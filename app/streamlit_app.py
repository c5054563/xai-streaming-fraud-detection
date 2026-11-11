"""Live demonstration dashboard: XAI-guided drift-adaptive streaming fraud detection.

Dinesh Gudapuri (C5054563), MSc Big Data Analytics, Sheffield Hallam University.
Run from the repository root:   streamlit run app/streamlit_app.py

The app replays the time-ordered evaluation stream (after the warm-up slice), scores every
transaction, explains alerts with TreeSHAP, and shows drift alarms from the label-free
SHAP-ADWIN detector next to the classic ADWIN and DDM detectors, which only see errors once
the (delayed) labels arrive.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudxai import cleansing, config, data, detectors, drift, features, models  # noqa: E402
from fraudxai.detectors import positive_class_shap  # noqa: E402

st.set_page_config(page_title="XAI fraud stream demo", layout="wide")
FEATS = features.MODEL_FEATURES


# ----------------------------------------------------------------------------- data and models
MODEL_FILE = config.RESULTS_DIR / "models" / "fraud_pipeline.pkl"


@st.cache_resource(show_spinner="Loading dataset and the trained model...")
def load_pipeline(seed: int):
    """Load the model saved by the notebook (results/models/fraud_pipeline.pkl); if it is
    missing, train the same warm-up Random Forest here so the demo still runs."""
    df, clean_log = cleansing.clean(data.load(data.download()))
    n_warm = int(len(df) * config.WARMUP_FRACTION)
    raw_warm, raw_eval = df.iloc[:n_warm].reset_index(drop=True), df.iloc[n_warm:].reset_index(drop=True)
    if MODEL_FILE.exists():
        saved = pickle.loads(MODEL_FILE.read_bytes())
        if saved.get("seed") == seed and saved.get("features") == FEATS:
            return raw_warm, raw_eval, saved["rf"], saved["shap_reference"], saved.get("arf"), "saved notebook model"
    warm = features.add_features(raw_warm)
    rf = models.make_batch_rf(seed=seed).fit(warm[FEATS].to_numpy(), warm.Class.to_numpy())
    ref = warm[FEATS].to_numpy()[np.random.default_rng(seed).choice(len(warm), 2000, replace=False)]
    return raw_warm, raw_eval, rf, ref, None, "trained in the app (no saved model for this seed)"


def tuned_arf_params() -> dict:
    """Use the tuned ARF setting from the notebook results if available."""
    f = config.RESULTS_DIR / "results.json"
    if f.exists():
        for r in json.loads(f.read_text()).get("tuning_best", []):
            if r["model"] == "ARF":
                import ast
                return ast.literal_eval(r["params"])
    return {}


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Stream controls")
scenario = st.sidebar.selectbox("Injected drift scenario", ["abrupt", "gradual", "adversarial", "none"])
delay_label = st.sidebar.selectbox("Label delay (chargeback latency)", ["0 s", "1 hour", "6 hours"], index=1)
delay_s = {"0 s": 0, "1 hour": 3600, "6 hours": 21600}[delay_label]
seed = st.sidebar.number_input("Seed", 0, 4, 0)
n_replay = st.sidebar.slider("Transactions to replay", 2_000, 60_000, 20_000, step=2_000)
drift_after = st.sidebar.slider("Drift starts after (transactions)", 1_000, 40_000, 8_000, step=1_000)
batch = st.sidebar.select_slider("Transactions per refresh", [250, 500, 1000, 2000], value=1000)
pause = st.sidebar.slider("Pause per refresh (s)", 0.0, 1.0, 0.1)
threshold = st.sidebar.slider("Alert threshold", 0.1, 0.9, config.THRESHOLD, step=0.05)
run_arf = st.sidebar.checkbox("Also score with online ARF (River, warm-up trained if saved)", value=False)
start = st.sidebar.button("Start replay", type="primary")

st.title("XAI-Guided Drift-Adaptive Streaming Fraud Detection")
st.caption("Dinesh Gudapuri (C5054563), Sheffield Hallam University. Data: ULB/Worldline credit card "
           "transactions (Dal Pozzolo et al.), ODbL v1.0 / DbCL v1.0. Drift is injected programmatically.")

raw_warm, raw_eval, rf, ref, saved_arf, model_source = load_pipeline(int(seed))
st.sidebar.caption(f"Model: {model_source}")
explainer_det = detectors.ShapADWIN(rf, ref, delta=0.002, sample_every=5)

c1, c2, c3, c4, c5 = st.columns(5)
m_proc, m_alert, m_caught, m_labels, m_drift = (c.empty() for c in (c1, c2, c3, c4, c5))
left, right = st.columns([3, 2])
score_box, signal_box = left.empty(), left.empty()
shap_box, alert_box = right.empty(), right.empty()
log_box = st.empty()

if not start:
    st.info(f"Ready: {len(raw_warm):,} warm-up transactions used to train the model; "
            f"{len(raw_eval):,} evaluation transactions available for replay. "
            "Choose the settings on the left and press **Start replay**.")
    st.stop()

# ----------------------------------------------------------------------------- replay
stream_df = raw_eval.iloc[:n_replay].reset_index(drop=True)
stream_df, drift_info = drift.inject(stream_df, scenario, t0=drift_after, seed=int(seed), width=5_000)
online = features.OnlineFeatures()
for x in raw_warm.iloc[-200:].to_dict("records"):      # prime the window features with recent history
    online.transform_one(x)
import copy
arf = (copy.deepcopy(saved_arf) if saved_arf is not None
       else models.make_online("ARF", seed=int(seed), **tuned_arf_params())) if run_arf else None
err_dets = {"ADWIN": models.make_detector("ADWIN"), "DDM": models.make_detector("DDM")}
pending: deque = deque()                                 # (time, label, prediction, feats)
scores, arf_scores, sig_pos, sig_val, alarms, alerts = [], [], [], [], [], []
labels_known = 0
last_shap = None

for start_i in range(0, len(stream_df), batch):
    chunk = stream_df.iloc[start_i:start_i + batch]
    rows = [online.transform_one(x) for x in chunk.to_dict("records")]
    X = pd.DataFrame(rows)[FEATS].to_numpy()
    p = rf.predict_proba(X)[:, 1]
    for j, (x_row, (_, r)) in enumerate(zip(rows, chunk.iterrows())):
        i = start_i + j
        # labels whose delay has passed arrive now: classic detectors see the error, ARF learns
        while pending and pending[0][0] + delay_s <= r.Time:
            tj, yj, pj, fj = pending.popleft()
            labels_known += 1
            err = int((pj >= threshold) != yj)
            for name, det in err_dets.items():
                det.update(err)
                if det.drift_detected:
                    alarms.append({"position": i, "detector": name, "note": "error-rate (needs labels)"})
            if arf is not None:
                arf.learn_one(fj, int(yj))
        if arf is not None:
            arf_scores.append(arf.predict_proba_one(x_row).get(1, 0.0))
        pending.append((r.Time, int(r.Class), p[j], x_row))
    scores.extend(p.tolist())
    # label-free SHAP signal on every 5th transaction
    idx = np.arange(0, len(X), explainer_det.sample_every)
    sig = explainer_det.signal(X[idx])
    for k, v in zip(idx, sig):
        explainer_det.adwin.update(float(v))
        sig_pos.append(start_i + int(k)); sig_val.append(float(v))
        if explainer_det.adwin.drift_detected:
            alarms.append({"position": start_i + int(k), "detector": "SHAP-ADWIN", "note": "label-free (SHAP)"})
    # alerts and explanation of the strongest new alert
    new = np.where(p >= threshold)[0]
    for k in new:
        alerts.append({"position": start_i + int(k), "hour": int(chunk.Time.iloc[k] // 3600) % 24,
                       "amount_eur": float(chunk.Amount.iloc[k]), "score": float(p[k]),
                       "true_label": "fraud" if chunk.Class.iloc[k] == 1 else "legit"})
    if len(new):
        k = int(new[np.argmax(p[new])])
        last_shap = (start_i + k, positive_class_shap(explainer_det.explainer, X[k:k + 1])[0])

    # ------------------------------------------------------------------ redraw
    done = start_i + len(chunk)
    caught = sum(a["true_label"] == "fraud" for a in alerts)
    m_proc.metric("Processed", f"{done:,}")
    m_alert.metric("Alerts", len(alerts))
    m_caught.metric("Alerts that were fraud", caught)
    m_labels.metric("Labels received", f"{labels_known:,}", help=f"Label delay: {delay_label}")
    m_drift.metric("Drift alarms", len(alarms))

    sdf = pd.DataFrame({"position": np.arange(done), "RF score": scores})
    if arf is not None:
        sdf["ARF score"] = arf_scores
    sdf = sdf.melt("position", var_name="model", value_name="fraud score")
    rule = alt.Chart(pd.DataFrame({"position": [drift_after]})).mark_rule(color="black", strokeDash=[4, 4])
    thr = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(color="red").encode(y="y")
    score_box.altair_chart((alt.Chart(sdf).mark_circle(size=12, opacity=0.6).encode(
        x="position", y=alt.Y("fraud score", scale=alt.Scale(domain=[0, 1])), color="model")
        + thr + (rule.encode(x="position") if scenario != "none" else thr)).properties(
        title="Fraud score per transaction (red line = alert threshold, dashed = drift start)", height=260),
        width="stretch")
    gdf = pd.DataFrame({"position": sig_pos, "SHAP drift signal": sig_val})
    gdf["rolling mean (200)"] = gdf["SHAP drift signal"].rolling(200, min_periods=1).mean()
    adf = pd.DataFrame(alarms) if alarms else pd.DataFrame(columns=["position", "detector", "note"])
    base = alt.Chart(gdf).mark_line(color="#999", opacity=0.4).encode(x="position", y="SHAP drift signal") + \
        alt.Chart(gdf).mark_line(color="#1565C0").encode(x="position", y="rolling mean (200)")
    marks = alt.Chart(adf).mark_rule(strokeWidth=2).encode(x="position:Q", color="detector:N")
    signal_box.altair_chart((base + marks + (rule.encode(x="position") if scenario != "none" else base)).properties(
        title="Label-free SHAP drift signal and detector alarms", height=220), width="stretch")
    if last_shap is not None:
        pos, sv = last_shap
        top = pd.DataFrame({"feature": FEATS, "SHAP": sv}).reindex(
            pd.Series(np.abs(sv)).sort_values(ascending=False).index[:10])
        shap_box.altair_chart(alt.Chart(top).mark_bar().encode(
            x="SHAP", y=alt.Y("feature", sort="-x"),
            color=alt.condition(alt.datum.SHAP > 0, alt.value("#C62828"), alt.value("#2E7D32"))).properties(
            title=f"Why transaction #{pos} was alerted (TreeSHAP, red pushes to fraud)", height=260),
            width="stretch")
    alert_box.dataframe(pd.DataFrame(alerts[-15:][::-1]), width="stretch", height=260)
    log_box.dataframe(adf.sort_values("position") if len(adf) else adf, width="stretch")
    time.sleep(pause)

st.success(f"Replay finished: {len(stream_df):,} transactions, {len(alerts)} alerts, "
           f"{len(alarms)} drift alarms. Drift: {drift_info}")

# XAI-Guided Drift-Adaptive Streaming Fraud Detection

**Bridging Black-Box Models and Regulatory Transparency in Real-Time Financial Systems**

MSc Big Data Analytics, Computing Research Project, Sheffield Hallam University
Author: **Dinesh Gudapuri** (C5054563, c5054563@hallam.shu.ac.uk) · Supervisor: Olamilekan Shobayo

This repository holds the code for a reproducible Python prototype that replays real card
transactions as a time-ordered stream, simulates delayed (chargeback) labels, injects controlled
concept drift, and compares a label-free, XAI-guided drift detector (ADWIN on windowed TreeSHAP
attributions, "SHAP-ADWIN") with the classic error-rate detectors ADWIN and DDM. Evaluation is
prequential (test-then-train) with PR-AUC, G-mean and Precision@k; accuracy and cross-validation
are not used (UREC2 ethics form).

## Research question

Can monitoring changes in SHAP explanations give an earlier indication that fraud patterns have
changed (concept drift) than classic error-rate drift detectors (ADWIN, DDM) in streaming fraud
detection with extreme imbalance and delayed feedback?

## Objectives and code traceability

| Objective | Description | Where in the code | Tests |
|---|---|---|---|
| O1 Baseline | Streaming pipeline (Hoeffding Tree, Adaptive Random Forest) measured with PR-AUC and G-mean | `src/fraudxai/data.py`, `cleansing.py`, `features.py`, `models.py`, `metrics.py`; notebook 3.18–3.21, 3.23 | `test_cleansing.py`, `test_features.py`, `test_metrics_stream.py` |
| O2 Latency | Delayed-feedback label simulation (chargeback latency) | `src/fraudxai/stream.py` (`prequential`, label queue), `experiments.py` (`label_arrival_positions`); notebook 3.22 | `test_delayed_labels_are_not_used_before_delay` |
| O3 Strategy | Incremental online learning vs batch retraining | `models.py` (`make_online`, `make_batch_rf`), `tuning.evaluate`; notebook preview section | — (Chapter 4) |
| O4 Trigger | XAI-guided drift detector on windowed TreeSHAP vs ADWIN and DDM | `src/fraudxai/detectors.py` (`ShapADWIN`), `drift.py`, `experiments.py`; notebook 3.22 and preview | `test_abrupt_drift_changes_only_after_t0`, `test_drift_is_repeatable_per_seed` |
| O5 Constraints | Alert-budget evaluation (Precision@k) | `metrics.precision_at_k`; notebook 3.24 | `test_g_mean_and_precision_at_k` |
| O6 Compliance | Explanation stability for regulatory relevance | `detectors.positive_class_shap`, Streamlit SHAP view (`app/`) | — (Chapter 5) |

Commit messages reference the objective numbers (for example `(O2)`), so `git log --grep "O4"`
lists every change for an objective.

## Repository layout

```
src/fraudxai/   config, data loading, cleansing, features, stream replay, drift injection,
                metrics, models, detectors, tuning, detector experiments
notebooks/      Dinesh_Gudapuri_C5054563_Fraud_XAI.ipynb (outputs stripped on commit)
tests/          pytest unit and integration tests
app/            streamlit_app.py: live demonstration dashboard
data/           raw dataset (git-ignored, never committed)
results/        generated figures and results.json (git-ignored)
```

## Dataset and licence

Credit Card Fraud Detection, Machine Learning Group (ULB) and Worldline: 284,807 card transactions
by European cardholders over two days in September 2013, 492 frauds (0.172 %), features V1–V28
(PCA), Time and Amount (Dal Pozzolo et al.).

* Source: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>; download mirror used by the
  code: OpenML data id 1597 (<https://www.openml.org/d/1597>).
* Licence: Open Database License (ODbL) v1.0 for the database
  (<https://opendatacommons.org/licenses/odbl/1-0/>) and Database Contents License (DbCL) v1.0
  for its contents (<https://opendatacommons.org/licenses/dbcl/1-0/>).
* **Raw data is not committed.** `data/` is in `.gitignore`; only code, configuration and
  aggregated results are shared. The data is already anonymised (PCA); no attempt is made to
  re-identify anyone. During the project the data is stored on the SHU F drive and deleted after
  the final mark (UREC2).

## How to run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
nbstripout --install                      # keep notebook outputs out of git

pytest -q                                 # unit and integration tests
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/Dinesh_Gudapuri_C5054563_Fraud_XAI.ipynb   # downloads data on first run
streamlit run app/streamlit_app.py        # live demonstration dashboard
```

The notebook downloads the dataset into `data/` on first run and writes figures and
`results/results.json`. Runtime: about 20–30 minutes on an 8-core machine (the hyperparameter
search and the detector comparison run in parallel processes).

## Method summary

* Time-ordered replay of the cleaned stream (stable sort by `Time`, exact duplicates dropped).
* Warm-up slice: first 20 % of the stream, used only for hyperparameter tuning and for fitting the
  TreeSHAP reference model; drift is injected only in the later evaluation part.
* Label delay: a transaction's label becomes available only `delay` seconds of stream time later.
* Drift scenarios (Gama et al., 2014): abrupt, gradual and adversarial (camouflage), all seeded.
* Metrics: PR-AUC, G-mean (threshold 0.5) and Precision@k (k = 10 alerts per 10,000 transactions);
  every result is mean ± SD over 5 seeds.

## Tools

Python, River (BSD-3-Clause), SHAP (MIT), scikit-learn (BSD-3-Clause), pandas, NumPy, Matplotlib,
Jupyter, pytest, Streamlit (Apache-2.0), Git/GitHub.

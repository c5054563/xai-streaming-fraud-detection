"""Unit tests for feature extraction (TC-U6..U9)."""
import numpy as np

from fraudxai import features


def test_hour_of_day_wraps_after_24h():
    assert list(features.hour_of_day([0, 3599, 3600, 86_399, 86_400, 172_799])) == [0, 0, 1, 23, 0, 23]


def test_window_features_use_only_past(tiny):
    f = features.add_features(tiny.drop_duplicates().reset_index(drop=True))
    # times 0, 30, 75, 100, 200: earlier rows within the previous 60 s
    assert list(f["tx_count_60s"]) == [0, 1, 1, 1, 0]
    assert f.loc[0, "amt_mean_100"] == 0.0                # no history for the first row
    assert np.isclose(f.loc[2, "amt_mean_100"], (10.0 + 0.0) / 2)


def test_log_amount_and_ratio(tiny):
    f = features.add_features(tiny)
    assert np.allclose(f["log_amount"], np.log1p(tiny["Amount"]))
    assert np.allclose(f["amt_ratio"], tiny["Amount"] / (f["amt_mean_100"] + 1))


def test_online_features_match_batch(tiny):
    batch = features.add_features(tiny)
    online = features.OnlineFeatures()
    for i, x in enumerate(tiny.to_dict("records")):
        o = online.transform_one(x)
        for k in features.ENGINEERED:
            assert np.isclose(o[k], batch.loc[i, k]), (i, k)

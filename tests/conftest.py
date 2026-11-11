import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudxai import config  # noqa: E402


@pytest.fixture
def tiny():
    """Six hand-made transactions in the real schema (one exact duplicate, one zero amount)."""
    rng = np.random.default_rng(0)
    rows = []
    for t, amt, cls in [(0, 10.0, 0), (30, 0.0, 0), (30, 0.0, 0), (75, 250.0, 1), (100, 5.0, 0), (200, 20.0, 0)]:
        r = {"Time": float(t), **{v: float(rng.normal()) for v in config.V_COLS}, "Amount": amt, "Class": cls}
        rows.append(r)
    rows[2] = dict(rows[1])            # exact duplicate of row 1
    return pd.DataFrame(rows)[config.RAW_COLS]

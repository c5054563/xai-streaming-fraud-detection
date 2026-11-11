"""Download and load the ULB/Worldline credit card dataset (Dal Pozzolo et al.).

Source: Kaggle mlg-ulb/creditcardfraud (ODbL v1.0 / DbCL v1.0); mirror used here: OpenML id 1597.
The raw file is stored in data/ and is never committed (see .gitignore).
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from . import config


def download(dest: Path = config.DATA_FILE, url: str = config.OPENML_PARQUET_URL) -> Path:
    """Download the OpenML parquet copy if it is not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path = config.DATA_FILE) -> pd.DataFrame:
    """Load the raw table with the original column order; Class as int (0 legit, 1 fraud)."""
    df = pd.read_parquet(path)
    df["Class"] = df["Class"].astype(int)
    return df[config.RAW_COLS]

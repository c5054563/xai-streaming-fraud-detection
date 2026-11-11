"""Data audit, validation rules and cleansing (report Section 3.20, objective O1).

Each rule returns a small record (id, rule, violations, action) so the notebook can show a
data-audit table and the unit tests can check every rule on tiny hand-made tables.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config


@dataclass
class RuleResult:
    rule_id: str
    rule: str
    violations: int
    action: str

    def as_dict(self) -> dict:
        return asdict(self)


def audit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level audit: type, missing, unique values, min, max."""
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique()),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> list[RuleResult]:
    """Apply validation rules R1-R9 and count violations (nothing is changed here)."""
    feats = [c for c in df.columns if c != config.TARGET]
    num = df[feats].apply(pd.to_numeric, errors="coerce")
    return [
        RuleResult("R1", "Schema: 31 expected columns present",
                   len(set(config.RAW_COLS) ^ set(df.columns)), "stop if > 0"),
        RuleResult("R2", "All features numeric (float)",
                   int((num.isna() & df[feats].notna()).sum().sum()), "stop if > 0"),
        RuleResult("R3", "No missing values",
                   int(df.isna().sum().sum()), "stop if > 0"),
        RuleResult("R4", "Class in {0, 1}",
                   int((~df[config.TARGET].isin([0, 1])).sum()), "stop if > 0"),
        RuleResult("R5", "0 <= Time <= 172,800 s and non-decreasing",
                   int(((df["Time"] < 0) | (df["Time"] > config.MAX_TIME_S)).sum()
                       + (df["Time"].diff().fillna(0) < 0).sum()), "sort by Time if > 0"),
        RuleResult("R6", "Amount >= 0",
                   int((df["Amount"] < 0).sum()), "stop if > 0"),
        RuleResult("R7", "V1-V28 and Amount finite",
                   int((~np.isfinite(num[config.V_COLS + ["Amount"]])).sum().sum()), "stop if > 0"),
        RuleResult("R8", "Exact duplicate rows",
                   int(df.duplicated().sum()), "drop, keep first"),
        RuleResult("R9", "Amount == 0 (zero-value transactions)",
                   int((df["Amount"] == 0).sum()), "keep (valid value), flag"),
    ]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleansing: enforce time order (stable), drop exact duplicates, reset the index."""
    n0, f0 = len(df), int(df[config.TARGET].sum())
    out = df.sort_values("Time", kind="stable")
    out = out.drop_duplicates(keep="first").reset_index(drop=True)
    log = {
        "rows_before": n0, "fraud_before": f0,
        "rows_after": len(out), "fraud_after": int(out[config.TARGET].sum()),
        "duplicates_dropped": n0 - len(out),
        "fraud_duplicates_dropped": f0 - int(out[config.TARGET].sum()),
    }
    return out, log

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "real_energy_risk_results.csv"

FEATURE_COLUMNS: List[str] = [
    "marketScore",
    "policyScore",
    "macroScore",
    "countrySpecificRiskScore",
    "marketWeight",
    "policyWeight",
    "macroWeight",
    "beta",
    "annualized_volatility",
    "gdp",
    "inflation",
    "debt",
    "fx",
    "policy_stability",
    "expected_annual_return",
    "estimated_sigma",
    "prob8",
]


def load_energy_risk_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median(numeric_only=True))
    df["country"] = df["country"].astype(str)
    df["energy"] = df["energy"].astype(str)
    df["region"] = df["region"].astype(str)
    return df


def minmax_scale(df: pd.DataFrame, columns: Iterable[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        values = out[col].astype(float)
        lo = values.min()
        hi = values.max()
        if np.isclose(hi, lo):
            out[col] = 0.0
        else:
            out[col] = (values - lo) / (hi - lo)
    return out


def row_to_state_vector(row: pd.Series) -> np.ndarray:
    return row[FEATURE_COLUMNS].astype(float).to_numpy(dtype=np.float32)


def discretize_row(row: pd.Series, bins: int = 4) -> tuple[int, ...]:
    selected = [
        "marketScore",
        "policyScore",
        "macroScore",
        "countrySpecificRiskScore",
        "beta",
        "annualized_volatility",
        "gdp",
        "inflation",
        "debt",
        "fx",
        "policy_stability",
        "prob8",
    ]
    values = row[selected].astype(float).to_numpy()
    clipped = np.clip(values, 0.0, 1.0)
    return tuple(np.minimum((clipped * bins).astype(int), bins - 1).tolist())


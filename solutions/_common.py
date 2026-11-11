"""Outils partagés par les corrigés."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from churn.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_or_create  # noqa: E402


def load_data():
    return load_or_create(ROOT / "data" / "customers.csv")


def make_pipeline(
    clf, extra_numeric: list[str] | None = None, steps_before: list | None = None, scale: bool = True
) -> Pipeline:
    """Pipeline standard (scaler + one-hot) autour d'un classifieur quelconque."""
    prep = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC_FEATURES + (extra_numeric or [])),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([*(steps_before or []), ("prep", prep), ("clf", clf)])


def cv(seed: int = 42, n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def title(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")

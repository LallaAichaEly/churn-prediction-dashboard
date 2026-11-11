"""Module 2 — Préparation des variables (feature engineering).

Idée clé : toute la préparation vit DANS un `Pipeline` scikit-learn. Ainsi les mêmes
transformations sont appliquées à l'entraînement et en production, et on évite la
*fuite de données* (data leakage) : le scaler n'apprend que sur le jeu d'entraînement.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des variables dérivées. Exercice : proposez-en d'autres et mesurez leur effet.

    - `avg_charge_per_month` : total payé / ancienneté (détecte les hausses de prix).
    - `calls_per_year` : intensité des appels au support, normalisée par l'ancienneté.
    """
    out = df.copy()
    out["avg_charge_per_month"] = out["total_charges"] / out["tenure_months"].clip(lower=1)
    out["calls_per_year"] = out["support_calls"] / (out["tenure_months"].clip(lower=1) / 12)
    return out


ENGINEERED_FEATURES = ["avg_charge_per_month", "calls_per_year"]


def build_preprocessor(use_engineered: bool = False) -> ColumnTransformer:
    """Normalise les numériques et encode les catégorielles en one-hot."""
    numeric = NUMERIC_FEATURES + (ENGINEERED_FEATURES if use_engineered else [])
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

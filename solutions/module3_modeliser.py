"""Corrigé — Module 3 : modéliser.

Usage : python solutions/module3_modeliser.py          (≈ 30 s, à cause de GridSearchCV)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score

try:
    from ._common import cv, load_data, make_pipeline, title
except ImportError:  # lancé comme script
    from _common import cv, load_data, make_pipeline, title

from churn.data import FEATURES, TARGET
from churn.model import build_pipeline, evaluate, split


def random_forest(seed: int = 42):
    """3.2 — Forêt aléatoire. `min_samples_leaf` évite des feuilles d'un seul client (sur-apprentissage)."""
    return make_pipeline(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=seed)
    )


def candidates(seed: int = 42) -> dict:
    return {
        "logistic": build_pipeline("logistic", seed=seed),
        "gboost": build_pipeline("gboost", seed=seed),
        "random_forest": random_forest(seed),
    }


def cross_validate_models(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """3.4 — ROC-AUC sur 5 plis pour chaque modèle (mêmes plis pour tous → comparaison appariée)."""
    X, y = df[FEATURES], df[TARGET]
    scores = {name: cross_val_score(p, X, y, cv=cv(seed), scoring="roc_auc") for name, p in candidates(seed).items()}
    return pd.DataFrame(scores, index=[f"pli {i + 1}" for i in range(5)])


def tune_boosting(df: pd.DataFrame, seed: int = 42, quick: bool = False) -> tuple[dict, float, float]:
    """3.5 — GridSearchCV sur le TRAIN uniquement, puis UNE évaluation finale sur le test.

    Le paramètre clé est `max_depth` : avec des arbres de profondeur 1 (« souches »), le boosting
    devient un modèle ADDITIF — exactement la forme du vrai processus (`_churn_logit`).
    Retourne (meilleurs paramètres, AUC test du boosting optimisé, AUC test de la logistique).
    """
    X_train, X_test, y_train, y_test = split(df, seed=seed)
    grid = {
        "clf__max_depth": [1, 3] if quick else [1, 2, 3, None],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_iter": [200] if quick else [200, 500],
    }
    search = GridSearchCV(build_pipeline("gboost", seed=seed), grid, scoring="roc_auc", cv=cv(seed), n_jobs=-1)
    search.fit(X_train, y_train)
    auc_boost = roc_auc_score(y_test, search.predict_proba(X_test)[:, 1])

    logistic = build_pipeline("logistic").fit(X_train, y_train)
    auc_logistic = roc_auc_score(y_test, logistic.predict_proba(X_test)[:, 1])
    params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
    return params, auc_boost, auc_logistic


def class_weight_effect(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """3.6 — `class_weight="balanced"` donne plus de poids aux churners (classe minoritaire)."""
    X_train, X_test, y_train, y_test = split(df, seed=seed)
    rows = {}
    for label, weight in [("sans pondération", None), ("class_weight='balanced'", "balanced")]:
        model = make_pipeline(LogisticRegression(max_iter=1000, class_weight=weight)).fit(X_train, y_train)
        m = evaluate(y_test, model.predict_proba(X_test)[:, 1], threshold=0.5)
        rows[label] = {"roc_auc": m.roc_auc, "precision": m.precision, "recall": m.recall, "f1": m.f1}
    return pd.DataFrame(rows).T


def main() -> None:
    df = load_data()

    title("3.2 + 3.4 — Validation croisée 5 plis (ROC-AUC)")
    scores = cross_validate_models(df)
    print(scores.round(4).to_string())
    print("\nmoyenne ± écart-type")
    for name in scores:
        print(f"  {name:<14} {scores[name].mean():.4f} ± {scores[name].std():.4f}")
    diff = scores["logistic"] - scores["gboost"]
    print(f"\nÉcart logistique − boosting, pli par pli : {np.round(diff.to_numpy(), 4)}")
    # Les plis partagent des données d'entraînement : un test de Student classique serait trop
    # optimiste. Le critère robuste et lisible : l'écart a-t-il toujours le même signe ?
    print(f"→ la logistique gagne sur {int((diff > 0).sum())}/5 plis "
          f"(écart moyen {diff.mean():.4f}, plus petit écart {diff.min():.4f})")

    title("3.5 — GridSearchCV sur le gradient boosting")
    params, auc_boost, auc_log = tune_boosting(df)
    print(f"Meilleurs paramètres : {params}")
    print(f"ROC-AUC test — boosting optimisé : {auc_boost:.4f} | logistique : {auc_log:.4f}")

    title("3.6 — Effet de class_weight='balanced' (seuil 0,5)")
    print(class_weight_effect(df).round(3).to_string())


if __name__ == "__main__":
    main()

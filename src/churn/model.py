"""Modules 3 & 4 : modélisation et évaluation.

Le parcours d'un data scientist :
1. Commencer par un modèle de référence simple et interprétable (régression logistique).
2. Essayer un modèle plus puissant (gradient boosting) et voir si le gain vaut la complexité.
3. Évaluer avec des métriques adaptées au déséquilibre des classes (ROC-AUC, précision,
   rappel), pas seulement l'accuracy.
4. Choisir un SEUIL de décision selon le coût métier, pas toujours 0,5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .data import FEATURES, TARGET
from .features import add_engineered_features, build_preprocessor

MODELS = {
    "logistic": "Régression logistique (référence interprétable)",
    "gboost": "Gradient boosting (plus puissant, moins lisible)",
}


def build_pipeline(model_name: str = "logistic", use_engineered: bool = False, seed: int = 42) -> Pipeline:
    if model_name == "logistic":
        clf = LogisticRegression(max_iter=1000)
    elif model_name == "gboost":
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
    else:
        raise ValueError(f"Modèle inconnu : {model_name!r}. Choix : {list(MODELS)}")

    steps = []
    if use_engineered:
        steps.append(("engineer", FunctionTransformer(add_engineered_features)))
    steps += [("prep", build_preprocessor(use_engineered)), ("clf", clf)]
    return Pipeline(steps)


def split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    """Découpage stratifié : même taux de churn dans le train et le test."""
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=test_size, stratify=df[TARGET], random_state=seed
    )


@dataclass
class Metrics:
    threshold: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


def evaluate(y_true, proba, threshold: float = 0.5) -> Metrics:
    """Calcule les métriques pour un seuil donné.

    - Précision : parmi les clients signalés « à risque », combien partent vraiment ?
    - Rappel    : parmi les clients qui partent, combien a-t-on détectés ?
    """
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return Metrics(
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y_true, proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def roc_points(y_true, proba) -> pd.DataFrame:
    fpr, tpr, thr = roc_curve(y_true, proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})


def threshold_table(y_true, proba, steps: int = 19) -> pd.DataFrame:
    """Précision / rappel / F1 pour une grille de seuils (compromis à visualiser)."""
    rows = [asdict(evaluate(y_true, proba, t)) for t in np.linspace(0.05, 0.95, steps)]
    return pd.DataFrame(rows)


def best_threshold_by_cost(y_true, proba, cost_fn: float, cost_fp: float) -> float:
    """Seuil qui minimise le coût total = FN × coût d'un client perdu + FP × coût d'une
    action de rétention inutile. C'est ici que la data science rencontre le métier."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in grid:
        m = evaluate(y_true, proba, t)
        costs.append(m.fn * cost_fn + m.fp * cost_fp)
    return float(grid[int(np.argmin(costs))])


def feature_importance(pipeline: Pipeline, X_test, y_test, seed: int = 42) -> pd.DataFrame:
    """Importance par permutation : de combien l'AUC chute-t-elle si on mélange une colonne ?

    Méthode agnostique du modèle → on peut comparer logistique et boosting.
    """
    result = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=seed
    )
    return (
        pd.DataFrame(
            {"feature": X_test.columns, "importance": result.importances_mean, "std": result.importances_std}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train(df: pd.DataFrame, model_name: str = "logistic", use_engineered: bool = False, seed: int = 42):
    """Entraîne et évalue. Retourne (pipeline, X_test, y_test, proba_test, metrics)."""
    X_train, X_test, y_train, y_test = split(df, seed=seed)
    pipeline = build_pipeline(model_name, use_engineered, seed)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return pipeline, X_test, y_test, proba, evaluate(y_test, proba)


def save(
    pipeline: Pipeline,
    metrics: Metrics,
    model_dir: str | Path = "models",
    name: str = "churn_model",
    model_name: str | None = None,
) -> Path:
    """Sauvegarde le pipeline (préparation + modèle) et ses métriques.

    On sauvegarde le PIPELINE entier, pas seulement le modèle : ainsi la normalisation et
    l'encodage appris à l'entraînement sont réappliqués à l'identique en production.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{name}.joblib"
    joblib.dump(pipeline, path)
    card = {"model": model_name, "features": list(FEATURES), **asdict(metrics)}
    (model_dir / f"{name}_metrics.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return path


def load(path: str | Path = "models/churn_model.joblib") -> Pipeline:
    return joblib.load(path)

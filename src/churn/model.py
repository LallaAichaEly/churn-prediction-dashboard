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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
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
    return round(float(grid[int(np.argmin(costs))]), 2)  # 0.07 plutôt que 0.06999999999999999


def tune_threshold(
    pipeline: Pipeline, X_train, y_train, cost_fn: float, cost_fp: float, cv: int = 5, seed: int = 42
) -> float:
    """Choisit le seuil SANS regarder le jeu de test.

    On obtient des probabilités « hors échantillon » sur le jeu d'entraînement par validation
    croisée, puis on y cherche le seuil le moins coûteux. Choisir le seuil directement sur le
    jeu de test donnerait une estimation trop optimiste : le test servirait deux fois.
    """
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipeline, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    return best_threshold_by_cost(y_train, oof, cost_fn, cost_fp)


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


def train(
    df: pd.DataFrame,
    model_name: str = "logistic",
    use_engineered: bool = False,
    seed: int = 42,
    costs: tuple[float, float] | None = None,
):
    """Entraîne et évalue. Retourne (pipeline, X_test, y_test, proba_test, metrics).

    Sans `costs`, les métriques sont calculées au seuil 0,5. Avec `costs=(coût_FN, coût_FP)`,
    le seuil est d'abord choisi par `tune_threshold` sur le jeu d'entraînement, puis les
    métriques du jeu de test sont calculées à ce seuil (évaluation honnête).
    """
    X_train, X_test, y_train, y_test = split(df, seed=seed)
    pipeline = build_pipeline(model_name, use_engineered, seed)
    threshold = 0.5
    if costs is not None:
        threshold = tune_threshold(pipeline, X_train, y_train, *costs, seed=seed)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return pipeline, X_test, y_test, proba, evaluate(y_test, proba, threshold)


def metrics_path(model_path: str | Path) -> Path:
    """`models/churn_model.joblib` → `models/churn_model_metrics.json`."""
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_metrics.json")


def save(pipeline: Pipeline, metrics: Metrics, model_dir: str | Path = "models", name: str = "churn_model") -> Path:
    """Enregistre le pipeline complet (préparation + modèle) et ses métriques, dont le seuil."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{name}.joblib"
    joblib.dump(pipeline, path)
    metrics_path(path).write_text(json.dumps(asdict(metrics), indent=2))
    return path


def load(path: str | Path = "models/churn_model.joblib") -> Pipeline:
    return joblib.load(path)


def load_metrics(path: str | Path = "models/churn_model.joblib") -> Metrics | None:
    """Relit les métriques enregistrées à côté du modèle (None si le fichier n'existe pas)."""
    mpath = metrics_path(path)
    if not mpath.exists():
        return None
    return Metrics(**json.loads(mpath.read_text()))

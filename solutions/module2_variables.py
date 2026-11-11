"""Corrigé — Module 2 : préparer les variables.

Usage : python solutions/module2_variables.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

try:
    from ._common import cv, load_data, make_pipeline, title
except ImportError:  # lancé comme script
    from _common import cv, load_data, make_pipeline, title

from churn.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from churn.model import split


def add_is_new_customer(df: pd.DataFrame) -> pd.DataFrame:
    """2.3 — 1 si le client a moins de 6 mois d'ancienneté."""
    out = df.copy()
    out["is_new_customer"] = (out["tenure_months"] < 6).astype(int)
    return out


def compare_is_new_customer(df: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    """2.3 — ROC-AUC en validation croisée, avec et sans la nouvelle variable.

    On compare en validation croisée (5 plis) et non sur un seul découpage : un écart de
    0,002 d'AUC sur un seul split n'est que du bruit.
    """
    X, y = df[FEATURES], df[TARGET]
    base = make_pipeline(LogisticRegression(max_iter=1000))
    with_new = make_pipeline(
        LogisticRegression(max_iter=1000),
        extra_numeric=["is_new_customer"],
        steps_before=[("new", FunctionTransformer(add_is_new_customer))],
    )
    return {
        "sans": cross_val_score(base, X, y, cv=cv(seed), scoring="roc_auc"),
        "avec is_new_customer": cross_val_score(with_new, X, y, cv=cv(seed), scoring="roc_auc"),
    }


def scaler_leakage(df: pd.DataFrame, seed: int = 42) -> dict[str, float]:
    """2.5 (a) — Normaliser AVANT le split. La fuite existe, mais elle est minuscule ici."""
    X_train, X_test, y_train, y_test = split(df, seed=seed)

    honest = make_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    auc_honest = roc_auc_score(y_test, honest.predict_proba(X_test)[:, 1])

    # ❌ Le scaler voit le jeu de test (moyennes et écarts-types calculés sur TOUTES les lignes).
    scaler = StandardScaler().fit(pd.concat([X_train, X_test])[NUMERIC_FEATURES])
    leak_train, leak_test = X_train.copy(), X_test.copy()
    leak_train[NUMERIC_FEATURES] = scaler.transform(X_train[NUMERIC_FEATURES])
    leak_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])
    # scale=False : le pipeline ne re-normalise pas (sinon il annulerait la fuite).
    leaky = make_pipeline(LogisticRegression(max_iter=1000), scale=False).fit(leak_train, y_train)
    auc_leaky = roc_auc_score(y_test, leaky.predict_proba(leak_test)[:, 1])
    return {"honnête": auc_honest, "scaler avant split": auc_leaky}


def target_encoder(seed: int) -> TargetEncoder:
    """Encodeur de cible reproductible. L'API a changé en scikit-learn 1.9 (`random_state`
    déprécié au profit d'un générateur de plis dans `cv`) : on s'adapte à la version installée."""
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 9):
        return TargetEncoder(cv=cv(seed))
    return TargetEncoder(random_state=seed)


def target_encoding_leakage(df: pd.DataFrame, n_codes: int = 2000, seed: int = 42) -> dict[str, float]:
    """2.5 (b) — Une fuite SPECTACULAIRE : l'encodage par la cible calculé sur tout le jeu.

    On ajoute un « code postal » 100 % aléatoire (aucun lien avec le churn), avec ~2-3 clients
    par code. On le remplace par le taux de churn moyen de son code :
    - calculé sur TOUT le jeu → la cible de chaque client de test fuit dans sa propre variable :
      l'AUC de test s'envole… mais c'est un mirage ;
    - calculé sur le TRAIN, appliqué au train → la cible de chaque client d'ENTRAÎNEMENT fuit :
      le modèle apprend à faire confiance à du bruit et s'effondre sur le test ;
    - calculé HORS-PLI (`TargetEncoder` de scikit-learn, qui fait du cross-fitting) → correct.
    """
    rng = np.random.default_rng(seed)
    data = df.assign(code_postal=rng.integers(0, n_codes, size=len(df)))
    train_idx, test_idx = (s.index for s in split(data, seed=seed)[:2])
    train, test = data.loc[train_idx], data.loc[test_idx]
    prior = train[TARGET].mean()

    def fit_eval(tr: pd.DataFrame, te: pd.DataFrame) -> float:
        cols = FEATURES + ["code_postal_te"]
        model = make_pipeline(LogisticRegression(max_iter=1000), extra_numeric=["code_postal_te"])
        model.fit(tr[cols], tr[TARGET])
        return roc_auc_score(te[TARGET], model.predict_proba(te[cols])[:, 1])

    # ❌ Fuite : moyenne de la cible par code calculée sur train + test.
    leaky_map = data.groupby("code_postal")[TARGET].mean()
    auc_leaky = fit_eval(
        train.assign(code_postal_te=train["code_postal"].map(leaky_map)),
        test.assign(code_postal_te=test["code_postal"].map(leaky_map)),
    )

    # ⚠️ Naïf : appris sur le train mais appliqué au train lui-même (fuite interne au train).
    train_map = train.groupby("code_postal")[TARGET].mean()
    auc_naive = fit_eval(
        train.assign(code_postal_te=train["code_postal"].map(train_map)),
        test.assign(code_postal_te=test["code_postal"].map(train_map).fillna(prior)),
    )

    # ✅ Correct : TargetEncoder encode chaque ligne du train avec des plis qui ne la contiennent pas.
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("te", target_encoder(seed), ["code_postal"]),
    ])
    model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
    cols = FEATURES + ["code_postal"]
    model.fit(train[cols], train[TARGET])
    auc_oof = roc_auc_score(test[TARGET], model.predict_proba(test[cols])[:, 1])
    return {
        "encodage cible sur tout le jeu": auc_leaky,
        "encodage sur le train (naïf)": auc_naive,
        "TargetEncoder hors-pli": auc_oof,
    }


def main() -> None:
    df = load_data()

    title("2.3 — Effet de is_new_customer (ROC-AUC, validation croisée 5 plis)")
    for name, scores in compare_is_new_customer(df).items():
        print(f"{name:<22} {scores.mean():.4f} ± {scores.std():.4f}")

    title("2.5 (a) — Fuite par normalisation avant le split (ROC-AUC test)")
    for name, auc in scaler_leakage(df).items():
        print(f"{name:<22} {auc:.4f}")

    title("2.5 (b) — Fuite par encodage de la cible (ROC-AUC test ; sans code postal ≈ 0,834)")
    for name, auc in target_encoding_leakage(df).items():
        print(f"{name:<32} {auc:.4f}")


if __name__ == "__main__":
    main()

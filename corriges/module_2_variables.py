"""Corrigé du module 2 — Préparer les variables.

Lancer : python corriges/module_2_variables.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _commun import donnees, titre
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from churn.model import build_pipeline, split


def add_is_new_customer(df: pd.DataFrame) -> pd.DataFrame:
    """2.3 — Variable binaire : client arrivé depuis moins de 6 mois."""
    out = df.copy()
    out["is_new_customer"] = (out["tenure_months"] < 6).astype(int)
    return out


def pipeline_is_new_customer() -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES + ["is_new_customer"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(add_is_new_customer)),
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def auc_test(pipeline: Pipeline, df: pd.DataFrame) -> float:
    X_train, X_test, y_train, y_test = split(df)
    pipeline.fit(X_train, y_train)
    return roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])


def main() -> None:
    df = donnees()

    titre("2.2 / 2.3 — Effet des variables dérivées (ROC-AUC sur le jeu de test)")
    print(f"Référence (variables brutes)          : {auc_test(build_pipeline('logistic'), df):.4f}")
    print(f"+ avg_charge_per_month, calls_per_year : {auc_test(build_pipeline('logistic', True), df):.4f}")
    print(f"+ is_new_customer                      : {auc_test(pipeline_is_new_customer(), df):.4f}")

    titre("2.5a — Fuite de données « douce » : normaliser AVANT le split")
    # Mauvais : le scaler voit la moyenne/l'écart-type du jeu de test.
    leaky = df.copy()
    leaky[NUMERIC_FEATURES] = StandardScaler().fit_transform(leaky[NUMERIC_FEATURES])
    X_train, X_test, y_train, y_test = split(leaky)
    model = Pipeline([
        ("prep", ColumnTransformer([("num", "passthrough", NUMERIC_FEATURES),
                                    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(X_train, y_train)
    print(f"Avec fuite (scaler sur tout le jeu) : {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
    print(f"Sans fuite (Pipeline)               : {auc_test(build_pipeline('logistic'), df):.4f}")
    print("→ Écart quasi nul : la fuite d'un StandardScaler est minime... mais le code reste faux.")

    titre("2.5b — Fuite de données « dure » : sélectionner des variables AVANT la validation")
    # Petit échantillon (300 clients) + 2 000 colonnes de bruit pur, sans AUCUN lien avec le churn :
    # la situation typique « beaucoup de variables, peu de lignes » (génomique, logs, texte...).
    rng = np.random.default_rng(0)
    y = df[TARGET].to_numpy()[:300]
    noise = rng.normal(size=(len(y), 2000))
    # Mauvais : on choisit les 20 colonnes « les plus liées au churn » sur TOUTES les lignes.
    chosen = SelectKBest(f_classif, k=20).fit(noise, y).get_support()
    leaky_auc = cross_val_score(LogisticRegression(max_iter=1000), noise[:, chosen], y,
                                cv=5, scoring="roc_auc").mean()
    # Bon : la sélection fait partie du pipeline, elle est refaite dans chaque pli.
    clean_auc = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)),
                                noise, y, cv=5, scoring="roc_auc").mean()
    print(f"Bruit pur, sélection hors pipeline : ROC-AUC = {leaky_auc:.3f}  ← illusion !")
    print(f"Bruit pur, sélection dans pipeline : ROC-AUC = {clean_auc:.3f}  ← la vérité (≈ 0,5 = hasard)")


if __name__ == "__main__":
    main()

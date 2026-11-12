"""Module 5 — Expliquer UNE prédiction (explication locale).

L'importance par permutation (`model.feature_importance`) répond à « quelles variables
comptent en général ? ». Ici on répond à « pourquoi CE client a-t-il ce score ? ».

Méthode « et si… ? » (agnostique du modèle, sans dépendance externe) :
pour chaque variable, on remplace la valeur du client par une valeur *typique* (médiane
pour les numériques, modalité la plus fréquente pour les catégorielles) et on regarde
de combien la probabilité de churn change.

    contribution = proba(client réel) − proba(client avec cette variable « normalisée »)

- contribution > 0 : cette caractéristique POUSSE vers le départ ;
- contribution < 0 : elle RETIENT le client.

Limites (à connaître pour un entretien !) :
- les variables sont modifiées une par une, donc les interactions et les corrélations
  (ex. `tenure_months` et `total_charges`) ne sont pas réparties équitablement ;
- tout dépend du client « typique » choisi : le contrat le plus fréquent est « Mensuel »,
  donc un client mensuel n'est PAS pénalisé par son contrat dans cette explication, alors
  que c'est la première cause de churn. On explique toujours « par rapport à quoi ».
SHAP (exercice 5.5) traite ces deux problèmes en moyennant sur de nombreuses références.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def typical_profile(reference: pd.DataFrame) -> pd.Series:
    """Le client « typique » : médiane des numériques, mode des catégorielles."""
    values = {c: reference[c].median() for c in NUMERIC_FEATURES}
    values.update({c: reference[c].mode().iloc[0] for c in CATEGORICAL_FEATURES})
    return pd.Series(values)[FEATURES]


def explain_prediction(pipeline: Pipeline, client: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Contribution de chaque variable à la probabilité de churn d'UN client.

    `client` : DataFrame d'une ligne avec les colonnes `FEATURES`.
    `reference` : données servant à définir le profil typique (ex. jeu d'entraînement).
    Retourne un DataFrame trié par impact absolu décroissant.
    """
    if len(client) != 1:
        raise ValueError("explain_prediction attend exactement un client (une ligne)")

    client = client[FEATURES].reset_index(drop=True)
    typical = typical_profile(reference)
    base_proba = float(pipeline.predict_proba(client)[0, 1])

    # Une ligne par variable « normalisée » → un seul appel à predict_proba (rapide).
    variants = pd.concat([client] * len(FEATURES), ignore_index=True)
    # Une médiane peut être décimale (ex. 12,5 mois) : on passe les numériques en float.
    variants[NUMERIC_FEATURES] = variants[NUMERIC_FEATURES].astype(float)
    for i, feature in enumerate(FEATURES):
        variants.loc[i, feature] = typical[feature]
    probas = pipeline.predict_proba(variants)[:, 1]

    out = pd.DataFrame(
        {
            "feature": FEATURES,
            "client_value": [client.loc[0, f] for f in FEATURES],
            "typical_value": [typical[f] for f in FEATURES],
            "contribution": base_proba - probas,
        }
    )
    order = out["contribution"].abs().sort_values(ascending=False).index
    return out.loc[order].reset_index(drop=True)

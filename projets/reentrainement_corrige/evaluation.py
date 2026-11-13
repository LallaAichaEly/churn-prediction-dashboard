"""Étape 3 (corrigé) — Mesurer un modèle sur un mois de test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .modeles import CIBLE, COUT_FN, COUT_FP, MOIS, SEUIL, VARIABLES, Mesures


def mesurer(modele, test: pd.DataFrame, seuil: float = SEUIL) -> Mesures:
    mois = test[MOIS].unique()
    if len(mois) != 1:
        raise ValueError(f"le test doit couvrir UN mois, pas {len(mois)} : comparer des modèles sur des mois "
                         f"différents ne veut rien dire")
    y = test[CIBLE].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(f"le mois {mois[0]} n'a qu'une seule classe : l'AUC n'est pas définie")

    # Seulement les VARIABLES : jamais la cible (le modèle lirait la réponse), ni le mois, ni l'identifiant.
    # La colonne de la classe 1 se lit dans classes_ : rien ne garantit qu'elle soit la deuxième.
    colonne = list(modele.classes_).index(1)
    proba = np.asarray(modele.predict_proba(test[list(VARIABLES)]))[:, colonne]

    alerte = proba >= seuil              # >= : « au moins 40 % » déclenche l'appel
    manques = int(((y == 1) & ~alerte).sum())
    fausses_alertes = int(((y == 0) & alerte).sum())
    return Mesures(
        mois=str(mois[0]),
        clients=len(y),
        auc=float(roc_auc_score(y, proba)),
        cout=(COUT_FN * manques + COUT_FP * fausses_alertes) / len(y),  # PAR client : comparable d'un mois à l'autre
        taux_alerte=float(alerte.mean()),
    )

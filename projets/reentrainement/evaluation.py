"""Étape 3 🟡 — Mesurer un modèle sur un mois de test.

Vérifier : python scripts/projet.py --projet 5 --etape 3
"""

from __future__ import annotations

import numpy as np  # noqa: F401 — vous en aurez besoin
import pandas as pd  # noqa: F401
from sklearn.metrics import roc_auc_score  # noqa: F401

from .modeles import CIBLE, COUT_FN, COUT_FP, MOIS, SEUIL, VARIABLES, Mesures  # noqa: F401


def mesurer(modele, test: pd.DataFrame, seuil: float = SEUIL) -> Mesures:
    """Renvoie les `Mesures` (modeles.py) du modèle sur `test`, qui couvre UN mois.

    1. `test` doit couvrir un seul mois (`test[MOIS].unique()`), sinon `ValueError` : on compare
       les modèles mois par mois. Si personne n'est parti ce mois-là (ou tout le monde), l'AUC n'est
       pas définie : `ValueError` aussi.
    2. Donnez au modèle `test[list(VARIABLES)]`, et RIEN d'autre : surtout pas la colonne `churn`
       (ce serait lui donner la réponse), ni le mois, ni l'identifiant.
    3. Prenez `modele.predict_proba(...)`, pas `predict` (qui décide au seuil 0,5, pas au nôtre). La
       colonne de la classe 1 (« part ») est `list(modele.classes_).index(1)` : rien ne garantit
       que ce soit la deuxième.
    4. On alerte quand la probabilité est `>= seuil` (les clients pile au seuil sont alertés).
    5. Remplissez :
       - `mois` : le mois de test ; `clients` : le nombre de lignes ;
       - `auc` : `roc_auc_score(y, proba)` (un float) ;
       - `cout` : `(COUT_FN × partants NON alertés + COUT_FP × restés alertés) / clients`, en euros
         PAR CLIENT (un mois de 300 clients se compare ainsi à un mois de 500) ;
       - `taux_alerte` : la part des clients alertés.
    6. Ne modifiez pas `test`.

    Exemple : 4 clients, probabilités [0,9 ; 0,4 ; 0,3 ; 0,1], churn [1, 0, 1, 0], seuil 0,4. On
    alerte les deux premiers ; le 3ᵉ part sans alerte (500 €), le 2ᵉ est alerté pour rien (50 €) :
    cout = 550 / 4 = 137,50 € par client, taux_alerte = 0,5, auc = 0,75.

    Ce que vous apprenez : `predict_proba` et `classes_`, une métrique qui ne dépend pas du seuil
    (AUC) et une qui en dépend (le coût), traduire un modèle en euros.
    """
    raise NotImplementedError

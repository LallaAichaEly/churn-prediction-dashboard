"""Entraîner un modèle sur un historique (DÉJÀ ÉCRIT : à lire).

C'est la recette du projet 1 (`src/churn/features.py` et `src/churn/model.py`) : variables numériques
centrées-réduites, variables catégorielles en indicatrices, régression logistique, le tout dans un
`Pipeline`. Le projet 5 ne change pas la recette : il change QUAND on s'en sert, sur QUELLES données,
et ce qu'on fait du résultat.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .modeles import CATEGORIELLES, CIBLE, NUMERIQUES, VARIABLES


def entrainer(train: pd.DataFrame) -> Pipeline:
    """Un modèle entraîné sur `train` (les colonnes `VARIABLES` et la cible `churn`).

    Le modèle ne reçoit que les VARIABLES : ni le mois (le modèle servirait mal le mois suivant, qu'il
    n'a jamais vu), ni l'identifiant (il apprendrait des clients par cœur). Il est déterministe :
    deux entraînements sur les mêmes données donnent le même modèle.
    """
    if train.empty:
        raise ValueError("impossible d'entraîner un modèle sans aucune ligne d'historique")
    preparation = ColumnTransformer([
        ("num", StandardScaler(), list(NUMERIQUES)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORIELLES)),
    ])
    modele = Pipeline([("prep", preparation), ("clf", LogisticRegression(max_iter=1000))])
    return modele.fit(train[list(VARIABLES)], train[CIBLE].astype(int))

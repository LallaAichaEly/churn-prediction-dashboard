"""Défi 01 — Faire parler les données.

Remplacez chaque `raise NotImplementedError(...)` par votre code, puis vérifiez :

    pytest defis/defi_01_explorer -q

Ne regardez `corrige.py` qu'après avoir obtenu tous les tests au vert (ou après 30 min bloqué).
"""

from __future__ import annotations

import pandas as pd

TARGET = "churn"


# --------------------------------------------------------------------------- Partie 1 🟢
def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Taux de churn pour chaque modalité de `column`.

    Retourne un DataFrame avec EXACTEMENT ces 3 colonnes, dans cet ordre :
        [column, "clients", "churn_rate"]
    trié par `churn_rate` décroissant, avec un index 0, 1, 2, ...

    Exemple avec column="contract" :
           contract  clients  churn_rate
        0   Mensuel     2794       0.509
        1    Annuel     1222       0.215
        2  Biannuel      984       0.089

    Indices : `groupby`, `.agg(clients="size", churn_rate="mean")`, `sort_values`, `reset_index`.
    """
    raise NotImplementedError("À vous de jouer ! (partie 1 — churn_rate_by)")


# --------------------------------------------------------------------------- Partie 2 🟢
def lift(df: pd.DataFrame, column: str) -> pd.Series:
    """Lift de chaque modalité = taux de churn du groupe / taux de churn global.

    Un lift de 1,4 signifie « ce groupe part 40 % plus que la moyenne » ;
    un lift de 0,5 signifie « deux fois moins ».

    Retourne une Series :
      - index  = les modalités de `column`
      - valeurs = le lift
      - nom     = "lift"
      - triée par lift décroissant

    Indice : réutilisez `churn_rate_by` !
    """
    raise NotImplementedError("À vous de jouer ! (partie 2 — lift)")


# --------------------------------------------------------------------------- Partie 3 🟡
def riskiest_segment(df: pd.DataFrame, columns: list[str], min_clients: int = 100) -> dict:
    """Le segment le plus à risque : la combinaison de modalités de `columns` qui a le plus
    fort taux de churn, PARMI les segments d'au moins `min_clients` clients.

    Exemple : columns=["contract", "payment_method"] → le segment gagnant pourrait être
    « contrat Mensuel ET paiement par Chèque ».

    Retourne un dictionnaire :
        {
            "segment": {"contract": "Mensuel", "payment_method": "Chèque"},
            "clients": 415,          # un `int` Python (pas numpy) : le résultat doit
            "churn_rate": 0.6096,    # pouvoir être envoyé en JSON à une API.
        }

    Lève `ValueError` si aucun segment n'atteint `min_clients`.

    ⚠️ Le piège : sans `min_clients`, un segment de 2 clients qui partent tous les deux
    (100 %) gagnerait. Une conclusion tirée de 2 personnes ne vaut rien.
    """
    raise NotImplementedError("À vous de jouer ! (partie 3 — riskiest_segment)")


# --------------------------------------------------------------------------- Partie 4 🟡
def check_monthly_charges(df: pd.DataFrame, low: float = 0.0, high: float = 500.0) -> list[str]:
    """Contrôle qualité (exercice 1.3 du PARCOURS) : une facture mensuelle doit être
    comprise entre `low` et `high` euros (bornes incluses).

    Retourne la liste des problèmes :
      - [] si tout va bien ;
      - sinon UN message qui contient le nombre de factures hors limites, par exemple
        "2 facture(s) mensuelle(s) hors de l'intervalle [0, 500] €".

    Règles :
      - les valeurs manquantes (NaN) sont ignorées ici (`validate()` les signale déjà) ;
      - si `low > high`, lever `ValueError` (programmation défensive).
    """
    raise NotImplementedError("À vous de jouer ! (partie 4 — check_monthly_charges)")

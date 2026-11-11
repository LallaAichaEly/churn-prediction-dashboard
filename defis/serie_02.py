"""Série 02 — Explorer les données, corrigés automatiquement.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que la série 01 :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py` (ou `python scripts/defis.py --defi 12` pour un seul défi).
3. Bloqué·e ? Relisez l'indice, testez dans un terminal Python, puis seulement en dernier
   recours ouvrez `defis/corrige_serie_02.py`.

Cette série approfondit le Module 1 : avant de modéliser, un·e data scientist sait déjà
**qui** part, **quand** et **dans quelle combinaison** de caractéristiques.

Pour expérimenter :
    >>> import sys; sys.path.insert(0, "src")
    >>> from churn.data import generate_customers
    >>> df = generate_customers(1000)
    >>> df.head()
"""

from __future__ import annotations

import pandas as pd

# =========================================================================== Module 1 — Données


def lift(df: pd.DataFrame, colonne: str, modalite) -> float:
    """Défi 11 🟡 — « Lift » d'un groupe = taux de churn du groupe ÷ taux de churn GLOBAL.

    - lift = 2   → ce groupe part deux fois plus que la moyenne ;
    - lift = 0.5 → deux fois moins.
    C'est l'indicateur préféré des équipes marketing pour cibler une campagne.

    Exemple (data/customers.csv) : lift(df, "contract", "Mensuel") ≈ 1.44

    Lever une `ValueError` si `modalite` n'existe pas dans `colonne` : sinon on calculerait
    la moyenne de 0 client, ce qui donne NaN… en silence. Une erreur bruyante vaut mieux
    qu'un résultat faux qui passe inaperçu.

    Indice : df[df[colonne] == modalite] sélectionne le groupe ; `.empty` dit s'il est vide.
    """
    raise NotImplementedError


def factures_hors_bornes(df: pd.DataFrame, minimum: float = 0.0, maximum: float = 500.0) -> pd.DataFrame:
    """Défi 12 🟡 (Parcours 1.3) — Lignes dont `monthly_charges` sort de [minimum, maximum].

    Règles :
    - les bornes sont acceptées : 500 € pile est normal, 500,01 € est suspect ;
    - une facture manquante (NaN) n'est PAS hors bornes : elle est déjà signalée par la règle
      « valeurs manquantes » de `validate()`, inutile de la compter deux fois ;
    - renvoyer les lignes suspectes avec toutes leurs colonnes, sans modifier `df`.

    C'est la règle à ajouter dans `validate()` pour l'exercice 1.3 du parcours.

    Indice : un filtre booléen `df[masque]`, avec `|` (ou) entre deux comparaisons.
    """
    raise NotImplementedError


def churn_par_tranche_anciennete(
    df: pd.DataFrame, bornes: tuple[int, ...] = (0, 6, 12, 24, 48, 72)
) -> pd.DataFrame:
    """Défi 13 🟡 — Taux de churn par tranche d'ancienneté (`tenure_months`).

    Renvoyer un DataFrame avec exactement 3 colonnes, une ligne par tranche, dans l'ordre :

        tranche   clients   taux_churn          (data/customers.csv)
        "1-6"        1415         0.51
        "7-12"        914         0.43
        "13-24"      1086         0.37
        "25-48"       950         0.23
        "49-72"       635         0.07

    La tranche ]a, b] s'écrit f"{a + 1}-{b}" (les mois sont des entiers) : un client de
    6 mois est dans « 1-6 », un client de 7 mois dans « 7-12 ».

    Indice : `pd.cut(df["tenure_months"], bins=..., labels=...)`, puis `groupby`
    et `.agg(clients="size", taux_churn="mean")`.
    """
    raise NotImplementedError


# ============================================================================== 🔴 Défi final


def segments_a_risque(
    df: pd.DataFrame, colonnes: list[str], min_clients: int = 50, top: int = 5
) -> pd.DataFrame:
    """Défi 14 🔴 — Les `top` segments (combinaisons de `colonnes`) qui partent le plus.

    Exemple avec colonnes=["contract", "payment_method"] (data/customers.csv) :

        contract  payment_method  clients  taux_churn
        Mensuel   Chèque              415        0.61
        Mensuel   Prélèvement         996        0.49
        ...

    Règles :
    - colonnes du résultat : les `colonnes` demandées, puis "clients", puis "taux_churn" ;
    - ignorer les segments de MOINS de `min_clients` clients : un taux calculé sur 8 personnes
      n'est pas fiable. C'est la leçon la plus importante de ce défi ;
    - trier par `taux_churn` décroissant, garder les `top` premiers, index 0, 1, 2…
    """
    raise NotImplementedError

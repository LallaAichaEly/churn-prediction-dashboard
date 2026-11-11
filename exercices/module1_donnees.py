"""Katas — Module 1 : comprendre les données.

Mode d'emploi
-------------
1. Choisissez une fonction ci-dessous (dans l'ordre : 🟢 puis 🟡 puis 🔴).
2. Remplacez `raise NotImplementedError` par votre code.
3. Lancez :  pytest exercices -q -rs
   - ✅ passed  : bravo, c'est juste ;
   - ❌ failed  : le message d'erreur explique ce qui ne va pas ;
   - ⏳ skipped : pas encore commencé.
4. Bloqué plus de 20 minutes ? Relisez l'indice, testez dans un terminal Python,
   puis seulement en dernier recours ouvrez `solutions/module1_donnees.py`.

Pour expérimenter :
    >>> import sys; sys.path.insert(0, "src")
    >>> from churn.data import generate_customers
    >>> df = generate_customers(1000)
    >>> df.head()
"""

from __future__ import annotations

import pandas as pd

TARGET = "churn"  # 1 = le client est parti, 0 = il est resté


# --------------------------------------------------------------------------- 🟢 Échauffement


def taux_churn(df: pd.DataFrame) -> float:
    """🟢 Proportion de clients partis (entre 0 et 1).

    Exemple : 3 clients partis sur 10 → 0.3

    Indice : la moyenne d'une colonne qui ne contient que des 0 et des 1, c'est… ?
    """
    raise NotImplementedError


def taux_churn_par(df: pd.DataFrame, colonne: str) -> pd.Series:
    """🟢 Taux de churn pour chaque modalité de `colonne`, du plus élevé au plus faible.

    Exemple pour `colonne="contract"` (données de `data/customers.csv`) :
        Mensuel     0.51
        Annuel      0.22
        Biannuel    0.09

    Indice : `df.groupby(...)[...].mean()`, puis `.sort_values(ascending=False)`.
    """
    raise NotImplementedError


def valeurs_manquantes(df: pd.DataFrame) -> pd.Series:
    """🟢 Nombre de valeurs manquantes par colonne.

    Ne garder que les colonnes qui en ont (> 0), de la plus touchée à la moins touchée.
    Si rien ne manque, renvoyer une Series vide.

    Indice : `df.isna().sum()`, puis un filtre booléen.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- 🟡 Analyse


def lift(df: pd.DataFrame, colonne: str, modalite) -> float:
    """🟡 « Lift » d'un groupe = taux de churn du groupe ÷ taux de churn global.

    - lift = 2   → ce groupe part deux fois plus que la moyenne ;
    - lift = 0.5 → deux fois moins.
    C'est l'indicateur préféré des équipes marketing pour cibler une campagne.

    Lever une `ValueError` si `modalite` n'existe pas dans `colonne`
    (sinon on diviserait 0 client par quelque chose et on obtiendrait NaN en silence).
    """
    raise NotImplementedError


def factures_hors_bornes(df: pd.DataFrame, minimum: float = 0.0, maximum: float = 500.0) -> pd.DataFrame:
    """🟡 Lignes dont `monthly_charges` est strictement en dehors de [minimum, maximum].

    Une facture de exactement 500 € est acceptée ; 500,01 € ne l'est pas.
    Ne pas modifier `df` : renvoyer seulement les lignes suspectes (avec toutes leurs colonnes).

    C'est la première étape de l'exercice 1.3 du parcours (ajouter une règle à `validate()`).
    """
    raise NotImplementedError


def churn_par_tranche_anciennete(
    df: pd.DataFrame, bornes: tuple[int, ...] = (0, 6, 12, 24, 48, 72)
) -> pd.DataFrame:
    """🟡 Taux de churn par tranche d'ancienneté (`tenure_months`).

    Renvoyer un DataFrame avec exactement 3 colonnes, une ligne par tranche, dans l'ordre :
        tranche   clients   taux_churn
        "1-6"     ...       ...
        "7-12"    ...       ...
        "13-24"   ...       ...
        ...
    L'étiquette d'une tranche ]a, b] s'écrit f"{a + 1}-{b}" (les mois sont entiers).

    Indice : `pd.cut(df["tenure_months"], bins=..., labels=...)`, puis `groupby`
    et `.agg(clients="size", taux_churn="mean")`.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- 🔴 Défi


def segments_a_risque(
    df: pd.DataFrame, colonnes: list[str], min_clients: int = 50, top: int = 5
) -> pd.DataFrame:
    """🔴 Les `top` segments (combinaisons de `colonnes`) qui partent le plus.

    Exemple avec `colonnes=["contract", "payment_method"]` (données de `data/customers.csv`) :
        contract  payment_method  clients  taux_churn
        Mensuel   Chèque              415        0.61
        Mensuel   Prélèvement         996        0.49
        ...

    Règles :
    - colonnes du résultat : les `colonnes` demandées, puis "clients", puis "taux_churn" ;
    - ignorer les segments de moins de `min_clients` clients (un taux calculé sur
      8 personnes n'est pas fiable : c'est la leçon la plus importante de ce kata) ;
    - trier par `taux_churn` décroissant, garder les `top` premiers, index 0, 1, 2…
    """
    raise NotImplementedError

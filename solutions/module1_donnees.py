"""Corrigé — Module 1 : comprendre les données.

⚠️ Ne lisez ce fichier qu'après avoir vraiment cherché. Comparez ensuite avec votre version :
il existe souvent plusieurs bonnes réponses, celle-ci n'est qu'une possibilité.
"""

from __future__ import annotations

import pandas as pd

TARGET = "churn"


def taux_churn(df: pd.DataFrame) -> float:
    # La moyenne d'une colonne 0/1 est exactement la proportion de 1.
    return float(df[TARGET].mean())


def taux_churn_par(df: pd.DataFrame, colonne: str) -> pd.Series:
    return df.groupby(colonne)[TARGET].mean().sort_values(ascending=False)


def valeurs_manquantes(df: pd.DataFrame) -> pd.Series:
    manquantes = df.isna().sum()
    return manquantes[manquantes > 0].sort_values(ascending=False)


def lift(df: pd.DataFrame, colonne: str, modalite) -> float:
    groupe = df[df[colonne] == modalite]
    if groupe.empty:
        raise ValueError(f"Modalité {modalite!r} absente de la colonne {colonne!r}")
    return float(groupe[TARGET].mean() / df[TARGET].mean())


def factures_hors_bornes(df: pd.DataFrame, minimum: float = 0.0, maximum: float = 500.0) -> pd.DataFrame:
    hors = (df["monthly_charges"] < minimum) | (df["monthly_charges"] > maximum)
    return df[hors]


def churn_par_tranche_anciennete(
    df: pd.DataFrame, bornes: tuple[int, ...] = (0, 6, 12, 24, 48, 72)
) -> pd.DataFrame:
    etiquettes = [f"{a + 1}-{b}" for a, b in zip(bornes[:-1], bornes[1:])]
    tranche = pd.cut(df["tenure_months"], bins=list(bornes), labels=etiquettes)
    resume = (
        df.groupby(tranche, observed=True)[TARGET]
        .agg(clients="size", taux_churn="mean")
        .reset_index()
        .rename(columns={"tenure_months": "tranche"})
    )
    resume["tranche"] = resume["tranche"].astype(str)
    return resume[["tranche", "clients", "taux_churn"]]


def segments_a_risque(
    df: pd.DataFrame, colonnes: list[str], min_clients: int = 50, top: int = 5
) -> pd.DataFrame:
    segments = (
        df.groupby(colonnes)[TARGET]
        .agg(clients="size", taux_churn="mean")
        .reset_index()
    )
    segments = segments[segments["clients"] >= min_clients]
    return (
        segments.sort_values("taux_churn", ascending=False)
        .head(top)
        .reset_index(drop=True)[[*colonnes, "clients", "taux_churn"]]
    )

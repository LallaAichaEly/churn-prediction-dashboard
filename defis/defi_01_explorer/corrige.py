"""Défi 01 — CORRIGÉ commenté.

⚠️ À lire seulement APRÈS avoir essayé. Pour vérifier que ce corrigé passe les tests :

    DEFI_CORRIGE=1 pytest defis/defi_01_explorer -q
"""

from __future__ import annotations

import pandas as pd

TARGET = "churn"


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupby(column, observed=True)[TARGET]
        # agg avec des noms : une seule passe, des colonnes déjà bien nommées.
        .agg(clients="size", churn_rate="mean")
        .sort_values("churn_rate", ascending=False)
        # reset_index fait de `column` une vraie colonne et renumérote 0..n-1.
        .reset_index()
    )


def lift(df: pd.DataFrame, column: str) -> pd.Series:
    global_rate = df[TARGET].mean()
    if global_rate == 0:
        raise ValueError("Aucun churn dans les données : le lift n'est pas défini")
    rates = churn_rate_by(df, column).set_index(column)["churn_rate"]
    # Déjà trié par churn_rate décroissant, donc aussi par lift décroissant.
    return (rates / global_rate).rename("lift")


def riskiest_segment(df: pd.DataFrame, columns: list[str], min_clients: int = 100) -> dict:
    stats = (
        df.groupby(columns, observed=True)[TARGET]
        .agg(clients="size", churn_rate="mean")
        .reset_index()
    )
    # On filtre AVANT de chercher le maximum : c'est toute la leçon de cette partie.
    big_enough = stats[stats["clients"] >= min_clients]
    if big_enough.empty:
        raise ValueError(f"Aucun segment n'a au moins {min_clients} clients")

    best = big_enough.loc[big_enough["churn_rate"].idxmax()]
    return {
        "segment": {col: best[col] for col in columns},
        # int() / float() : convertit les types numpy en types Python (JSON-sérialisables).
        "clients": int(best["clients"]),
        "churn_rate": float(best["churn_rate"]),
    }


def check_monthly_charges(df: pd.DataFrame, low: float = 0.0, high: float = 500.0) -> list[str]:
    if low > high:
        raise ValueError(f"Bornes incohérentes : low={low} > high={high}")
    charges = df["monthly_charges"].dropna()
    # between(..., inclusive="both") est la valeur par défaut : bornes incluses.
    n_bad = int((~charges.between(low, high)).sum())
    if n_bad == 0:
        return []
    return [f"{n_bad} facture(s) mensuelle(s) hors de l'intervalle [{low:g}, {high:g}] €"]

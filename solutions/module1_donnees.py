"""Corrigé — Module 1 : comprendre les données.

Usage : python solutions/module1_donnees.py
"""

from __future__ import annotations

import pandas as pd

try:
    from ._common import load_data, title
except ImportError:  # lancé comme script
    from _common import load_data, title

from churn.data import TARGET, validate

MAX_MONTHLY_CHARGE = 500


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """1.2 — Taux de churn ET effectif par modalité (un taux sans effectif peut tromper)."""
    return (
        df.groupby(col)[TARGET]
        .agg(taux_churn="mean", clients="size")
        .sort_values("taux_churn", ascending=False)
    )


def validate_v2(df: pd.DataFrame) -> list[str]:
    """1.3 — `validate()` + règle métier sur la facture mensuelle.

    `< 0 | > MAX` plutôt que `~between(0, MAX)` : une valeur manquante (NaN) est déjà
    signalée par la règle « valeurs manquantes », on ne veut pas la compter deux fois.
    """
    problems = validate(df)
    if "monthly_charges" in df.columns:
        charges = df["monthly_charges"]
        if ((charges < 0) | (charges > MAX_MONTHLY_CHARGE)).any():
            problems.append(f"Facture mensuelle hors de [0, {MAX_MONTHLY_CHARGE}] €")
    return problems


def main() -> None:
    df = load_data()

    title("1.1 — Premier contact")
    print(df.describe().round(2).T[["mean", "std", "min", "50%", "max"]].to_string())
    print(f"\nTaux de churn global : {df[TARGET].mean():.1%}")

    title("1.2 — Taux de churn par contrat et par service internet")
    print(churn_rate_by(df, "contract").round(3).to_string())
    print()
    print(churn_rate_by(df, "internet_service").round(3).to_string())

    title("1.3 — Nouvelle règle de validation")
    broken = df.head(3).copy()
    broken.loc[broken.index[0], "monthly_charges"] = 9_999
    print("Données d'origine :", validate_v2(df) or "OK")
    print("Facture à 9 999 € :", validate_v2(broken))


if __name__ == "__main__":
    main()

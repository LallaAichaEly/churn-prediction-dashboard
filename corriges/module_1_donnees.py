"""Corrigé du module 1 — Comprendre les données.

Lancer : python corriges/module_1_donnees.py
"""

from __future__ import annotations

import pandas as pd
from _commun import donnees, titre

from churn.data import TARGET, validate


def validate_avec_facture(df: pd.DataFrame) -> list[str]:
    """1.3 — `validate()` + une règle métier : facture mensuelle entre 0 et 500 €.

    Dans votre version, ajoutez simplement ces lignes à la fin de `validate()` dans
    `src/churn/data.py`, et le test `test_validate_rejette_facture_aberrante` ci-dessous
    dans `tests/test_churn.py`.
    """
    problems = validate(df)
    if "monthly_charges" in df.columns and not df["monthly_charges"].between(0, 500).all():
        problems.append("Facture mensuelle hors de [0 ; 500] €")
    return problems


def test_validate_rejette_facture_aberrante():
    from churn.data import generate_customers

    df = generate_customers(100, seed=0)
    assert validate_avec_facture(df) == []
    df.loc[0, "monthly_charges"] = 9999
    assert any("Facture" in p for p in validate_avec_facture(df))


def main() -> None:
    df = donnees()

    titre("1.1 — Statistiques descriptives")
    print(df.describe().round(2).to_string())
    print(f"\nTaux de churn global : {df[TARGET].mean():.1%}")

    titre("1.2 — Taux de churn par contrat et par service internet")
    for col in ["contract", "internet_service"]:
        stats = df.groupby(col)[TARGET].agg(taux="mean", clients="size").sort_values("taux", ascending=False)
        stats["taux"] = (stats["taux"] * 100).round(1).astype(str) + " %"
        print(stats.to_string(), "\n")

    titre("1.3 — Nouvelle règle de validation")
    test_validate_rejette_facture_aberrante()
    print("Règle « facture entre 0 et 500 € » : test OK")
    print("Problèmes sur les vraies données :", validate_avec_facture(df) or "aucun")


if __name__ == "__main__":
    main()

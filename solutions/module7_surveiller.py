"""Corrigé — Module 7 : surveiller le modèle en production (data drift).

Usage : python solutions/module7_surveiller.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from ._common import load_data, title
except ImportError:  # lancé comme script
    from _common import load_data, title

from churn.data import FEATURES, TARGET, generate_customers
from churn.model import evaluate, split, train
from churn.monitoring import drift_report, psi, simulate_batch

SCENARIOS = {
    "aucun changement": {},
    "prix +20 %": {"price_increase": 0.20},
    "75 % de mensuels": {"monthly_share": 0.75},
    "concurrent fibre": {"fibre_competition": 1.5},
}


def psi_noise(reference: pd.DataFrame, sizes=(200, 1000, 5000), repeats: int = 20) -> pd.DataFrame:
    """7.3 — PSI maximal entre la référence et des lots tirés du MÊME monde (aucune dérive)."""
    rows = []
    for n in sizes:
        worst = [drift_report(reference, generate_customers(n, seed=1000 + r))["psi"].max() for r in range(repeats)]
        rows.append({"taille_lot": n, "psi_max_moyen": np.mean(worst), "psi_max_pire": np.max(worst)})
    return pd.DataFrame(rows).set_index("taille_lot")


def scenario_table(df: pd.DataFrame, scenarios: dict | None = None) -> pd.DataFrame:
    """7.1, 7.4, 7.5 — Ce que voit la surveillance dans chaque scénario."""
    pipeline, _, _, proba_test, _ = train(df)
    reference = split(df)[0]
    rows = {}
    for name, kwargs in (scenarios or SCENARIOS).items():
        batch = simulate_batch(**kwargs)
        report = drift_report(reference, batch)
        proba = pipeline.predict_proba(batch[FEATURES])[:, 1]
        rows[name] = {
            "variable_max": report.iloc[0]["feature"],
            "psi_max": report.iloc[0]["psi"],
            "psi_scores": psi(proba_test, proba),
            "roc_auc": evaluate(batch[TARGET], proba).roc_auc,
            "churn_prédit": proba.mean(),
            "churn_réel": batch[TARGET].mean(),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["écart_réel_prédit"] = table["churn_réel"] - table["churn_prédit"]
    return table


def retrain_on_recent(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """7.4 / 7.5 — Ancien modèle contre modèle réentraîné sur 5 000 clients récents.

    Les deux sont évalués sur le MÊME lot de 2 000 nouveaux clients (jamais vu à l'entraînement).
    """
    batch = simulate_batch(**kwargs)
    old = train(df)[0]
    recent = simulate_batch(n=5000, seed=99, **kwargs)  # autre graine : d'autres clients que le lot évalué
    new = train(recent)[0]
    rows = {}
    for name, pipeline in [("ancien modèle", old), ("réentraîné", new)]:
        proba = pipeline.predict_proba(batch[FEATURES])[:, 1]
        rows[name] = {
            "roc_auc": evaluate(batch[TARGET], proba).roc_auc,
            "churn_prédit": proba.mean(),
            "churn_réel": batch[TARGET].mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    df = load_data()
    pd.set_option("display.width", 120)

    title("7.2 — PSI calculé à la main : 50/50 → 70/30")
    by_hand = (0.7 - 0.5) * np.log(0.7 / 0.5) + (0.3 - 0.5) * np.log(0.3 / 0.5)
    print(f"à la main : {by_hand:.4f}   psi() : {psi(['a'] * 50 + ['b'] * 50, ['a'] * 70 + ['b'] * 30):.4f}")

    title("7.3 — Bruit du PSI SANS dérive, selon la taille du lot")
    print(psi_noise(split(df)[0]).round(3).to_string())

    title("7.1 / 7.4 / 7.5 — Ce que voit la surveillance")
    print(scenario_table(df).round(3).to_string())

    for name in ("75 % de mensuels", "concurrent fibre"):
        title(f"Réentraîner ? — scénario « {name} »")
        print(retrain_on_recent(df, **SCENARIOS[name]).round(3).to_string())


if __name__ == "__main__":
    main()

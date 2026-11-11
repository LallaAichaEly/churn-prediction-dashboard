"""Corrigé — Module 4 : évaluer et décider.

Usage : python solutions/module4_evaluer.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

try:
    from ._common import load_data, title
except ImportError:  # lancé comme script
    from _common import load_data, title

from churn.model import best_threshold_by_cost, evaluate, train


def dummy_baseline(y_test) -> dict[str, float]:
    """4.2 — Le modèle « tout le monde reste » : bonne accuracy, rappel nul. Inutile !"""
    m = evaluate(y_test, np.zeros(len(y_test)), threshold=0.5)
    return {"accuracy": m.accuracy, "recall": m.recall}


def cost_scenarios(y_test, proba, scenarios: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """4.3 — Seuil optimal et coût total pour plusieurs hypothèses de coûts (FN, FP)."""
    rows = {}
    for name, (cost_fn, cost_fp) in scenarios.items():
        t = best_threshold_by_cost(y_test, proba, cost_fn, cost_fp)
        m, m05 = evaluate(y_test, proba, t), evaluate(y_test, proba, 0.5)
        rows[name] = {
            "seuil_optimal": t,
            "seuil_théorique": cost_fp / (cost_fp + cost_fn),
            "précision": m.precision,
            "rappel": m.recall,
            "coût_optimal": m.fn * cost_fn + m.fp * cost_fp,
            "coût_seuil_0.5": m05.fn * cost_fn + m05.fp * cost_fp,
        }
    return pd.DataFrame(rows).T


def pr_curve(y_test, proba) -> tuple[pd.DataFrame, float]:
    """4.4 — Courbe précision-rappel et sa surface (average precision).

    Référence du hasard : AP = taux de churn (et non 0,5 comme pour la ROC-AUC).
    """
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    curve = pd.DataFrame({"recall": recall[:-1], "precision": precision[:-1], "threshold": thresholds})
    return curve, float(average_precision_score(y_test, proba))


def calibration_table(y_test, proba, n_bins: int = 10) -> pd.DataFrame:
    """4.5 — Dans chaque tranche de probabilité prédite, quel taux de churn observe-t-on ?"""
    observed, predicted = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"proba_moyenne_prédite": predicted, "taux_churn_observé": observed,
                         "écart": observed - predicted})


def main() -> None:
    df = load_data()
    _, _, y_test, proba, _ = train(df, "logistic")
    _, _, _, proba_gb, _ = train(df, "gboost")

    title("4.2 — Pourquoi l'accuracy trompe")
    base = dummy_baseline(y_test)
    print(f"Modèle « personne ne part » : accuracy {base['accuracy']:.1%}, rappel {base['recall']:.0%}")
    print(f"Régression logistique       : accuracy {evaluate(y_test, proba).accuracy:.1%}, "
          f"rappel {evaluate(y_test, proba).recall:.0%}")

    title("4.3 — Seuil optimal selon les coûts métier")
    scenarios = {"FN 500 € / FP 50 €": (500, 50), "FN 100 € / FP 100 €": (100, 100)}
    print(cost_scenarios(y_test, proba, scenarios).round(3).to_string())

    title("4.4 — Courbe précision-rappel")
    curve, ap = pr_curve(y_test, proba)
    print(f"Average precision : {ap:.3f} (hasard = taux de churn = {y_test.mean():.3f})")
    for target in (0.8, 0.6, 0.4):
        row = curve[curve["recall"] >= target].iloc[-1]
        print(f"  pour un rappel ≥ {target:.0%} : précision {row['precision']:.1%} (seuil {row['threshold']:.2f})")

    title("4.5 — Calibration")
    for name, p in [("logistic", proba), ("gboost", proba_gb)]:
        table = calibration_table(y_test, p)
        print(f"\n{name} — Brier score {brier_score_loss(y_test, p):.4f} "
              f"| écart absolu moyen {table['écart'].abs().mean():.3f}")
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

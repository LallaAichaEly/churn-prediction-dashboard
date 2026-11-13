"""Corrigé du module 4 — Évaluer et décider.

Lancer : python corriges/module_4_evaluation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _commun import donnees, titre
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from churn.model import best_threshold_by_cost, evaluate, train


def precision_recall_points(y_true, proba) -> pd.DataFrame:
    """4.4 — Points de la courbe précision-rappel, prêts pour `px.line(x="recall", y="precision")`.

    Dans `app.py`, onglet Performance :
        pr = precision_recall_points(y_test, proba)
        fig = px.line(pr, x="recall", y="precision", title=f"Précision-rappel (AP = ...)")
        fig.add_hline(y=y_test.mean(), line_dash="dash")   # niveau du hasard = taux de churn
    """
    precision, recall, thr = precision_recall_curve(y_true, proba)
    return pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": thr})


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """4.5 — Probabilité prédite moyenne vs fréquence réelle de churn, par tranche."""
    reel, predit = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"proba_predite": predit, "churn_observe": reel, "ecart": reel - predit})


def campaign_profit(y_true, proba, threshold: float, client_value: float, action_cost: float,
                    success_rate: float) -> float:
    """4.6 — Gain net d'une campagne de rétention qui contacte les clients au-dessus du seuil.

    Plus réaliste que `best_threshold_by_cost`, qui suppose que l'action est gratuite et
    réussit à 100 % sur les vrais churners :
    - chaque client contacté coûte `action_cost` (churner ou non) ;
    - un churner contacté n'est retenu qu'avec une probabilité `success_rate`.
    Ne rien faire rapporte 0 € : c'est la référence.
    """
    y_true = np.asarray(y_true)
    contacted = np.asarray(proba) >= threshold
    saved = (contacted & (y_true == 1)).sum() * success_rate
    return round(float(saved * client_value - contacted.sum() * action_cost), 2)


def best_campaign(y_true, proba, client_value: float, action_cost: float, success_rate: float):
    """Seuil qui maximise le gain net. 1,0 = ne contacter personne (choisi en cas d'égalité)."""
    grid = np.concatenate([[1.0], np.linspace(0.01, 0.99, 99)])
    gains = [campaign_profit(y_true, proba, t, client_value, action_cost, success_rate) for t in grid]
    i = int(np.argmax(gains))
    return float(grid[i]), gains[i]


def main() -> None:
    df = donnees()
    _, _, y_test, proba, m = train(df)

    titre("4.2 — Pourquoi l'accuracy est trompeuse")
    naive = evaluate(y_test, np.zeros_like(proba))
    print(f"Modèle naïf « personne ne part » : accuracy {naive.accuracy:.1%}, rappel {naive.recall:.0%}")
    print(f"Notre modèle (seuil 0,5)         : accuracy {m.accuracy:.1%}, rappel {m.recall:.0%}")
    print(f"→ +{(m.accuracy - naive.accuracy) * 100:.0f} points d'accuracy seulement, "
          f"mais on passe de 0% à {m.recall:.0%} de départs détectés.")

    titre("4.3 — Seuil optimal selon les coûts métier (jeu de test)")
    for cost_fn, cost_fp in [(500, 50), (100, 100), (50, 500)]:
        t = best_threshold_by_cost(y_test, proba, cost_fn, cost_fp)
        b = evaluate(y_test, proba, t)
        print(f"client perdu {cost_fn:>3} € / action inutile {cost_fp:>3} € → seuil {t:.2f} | "
              f"précision {b.precision:.0%} | rappel {b.recall:.0%} | "
              f"coût {b.fn * cost_fn + b.fp * cost_fp:>7,.0f} €")
    print("→ Théorie : seuil ≈ coût_FP / (coût_FP + coût_FN) si le modèle est bien calibré :")
    print("  500/50 → 50/550 ≈ 0,09 ; 100/100 → 0,50 ; 50/500 → 0,91.")

    titre("4.6 — Et si l'action de rétention ne marchait pas toujours ? (client = 500 €, action = 50 €)")
    for success in [1.0, 0.5, 0.3, 0.1]:
        t, gain = best_campaign(y_test, proba, 500, 50, success)
        contactes = int((proba >= t).sum())
        print(f"taux de succès {success:4.0%} → seuil {t:.2f} | {contactes:4d} clients contactés | "
              f"gain net {gain:>9,.0f} € pour {len(y_test)} clients")
    print("→ Le seuil optimal MONTE quand l'action marche moins bien, et le gain fond.")
    print("  La donnée la plus précieuse n'est pas l'AUC, mais le taux de succès réel de la campagne")
    print("  (à mesurer par un test A/B : groupe contacté vs groupe témoin).")

    titre("4.4 — Courbe précision-rappel")
    pr = precision_recall_points(y_test, proba)
    print(f"Average precision : {average_precision_score(y_test, proba):.3f} "
          f"(hasard = taux de churn = {y_test.mean():.3f})")
    for r in [0.9, 0.8, 0.6]:
        best_p = pr.loc[pr["recall"] >= r, "precision"].max()
        print(f"Pour détecter au moins {r:.0%} des départs, la meilleure précision possible est {best_p:.0%}")

    titre("4.5 — Calibration : une probabilité de 0,7 signifie-t-elle 70 % ?")
    cal = calibration_table(y_test, proba)
    print(cal.round(3).to_string(index=False))
    par_tranche = len(y_test) / len(cal)
    bruit = np.sqrt(0.25 / par_tranche)  # écart-type d'une proportion, au pire (p = 0,5)
    print(f"\nÉcart absolu moyen : {cal['ecart'].abs().mean():.3f} | plus grand écart : "
          f"{cal['ecart'].abs().max():.3f} | Brier score : {brier_score_loss(y_test, proba):.4f}")
    print(f"Avec ~{par_tranche:.0f} clients par tranche, le bruit d'échantillonnage vaut ±{bruit:.3f} "
          f"(1 écart-type) : le plus grand écart vaut {cal['ecart'].abs().max() / bruit:.1f} fois ce bruit.")
    print("→ La régression logistique est naturellement bien calibrée (elle optimise la log-vraisemblance).")
    print("  Ce n'est pas le cas de tous les modèles : refaites l'exercice avec le gradient boosting !")


if __name__ == "__main__":
    main()

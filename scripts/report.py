"""Génère la fiche modèle (model card) : RAPPORT_MODELE.md.

Usage :
    python scripts/report.py
    python scripts/report.py --cost-fn 300 --cost-fp 30

Une fiche modèle résume ce qu'un modèle fait, comment il a été évalué, ses limites et
comment le reproduire. Elle est générée par du code : les chiffres ne peuvent donc pas
être recopiés de travers, et la fiche reste à jour à chaque réentraînement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn import __version__  # noqa: E402
from churn.data import load_or_create, validate  # noqa: E402
from churn.model import MODELS, best_threshold_by_cost, evaluate, feature_importance, train  # noqa: E402

# Variables présentes dans la « vérité cachée » (`_churn_logit` dans src/churn/data.py).
TRUE_CAUSES = {
    "contract", "tenure_months", "monthly_charges", "support_calls", "payment_method",
    "internet_service", "tech_support", "num_products", "senior",
}


# À performance égale, on garde le modèle le plus simple (principe de parcimonie).
AUC_TOLERANCE = 0.005


def num(x: float, decimals: int = 3) -> str:
    """Nombre à la française : virgule décimale, espace pour les milliers."""
    return f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")


def euros(x: float) -> str:
    return f"{num(x, 0)} €"


def pct(x: float) -> str:
    return f"{num(100 * x, 1)} %"


def build_report(cost_fn: float, cost_fp: float, seed: int = 42) -> str:
    df = load_or_create(ROOT / "data" / "customers.csv", seed=seed)
    problems = validate(df)
    if problems:
        raise SystemExit("Données invalides :\n- " + "\n- ".join(problems))

    # 1. Comparer tous les modèles, avec et sans variables dérivées.
    runs = {}
    for name in MODELS:
        for engineered in (False, True):
            runs[(name, engineered)] = train(df, name, engineered, seed)
    top_auc = max(run[4].roc_auc for run in runs.values())
    candidates = [k for k, run in runs.items() if run[4].roc_auc >= top_auc - AUC_TOLERANCE]
    best_key = min(candidates, key=lambda k: (k[0] != "logistic", k[1]))  # le plus simple
    pipeline, X_test, y_test, proba, m50 = runs[best_key]
    best_name, best_eng = best_key

    # 2. Seuil métier et impact financier sur le jeu de test.
    t_best = best_threshold_by_cost(y_test, proba, cost_fn, cost_fp)
    m_best = evaluate(y_test, proba, t_best)

    def cost(m) -> float:
        return m.fn * cost_fn + m.fp * cost_fp

    n_test, n_churn = len(y_test), int(y_test.sum())
    nobody = n_churn * cost_fn               # on ne fait rien : tous les départs sont perdus
    everybody = (n_test - n_churn) * cost_fp  # on contacte tout le monde
    best_without_model = min(nobody, everybody)
    savings = best_without_model - cost(m_best)
    theory = cost_fp / (cost_fp + cost_fn)

    # 3. Interprétation.
    imp = feature_importance(pipeline, X_test, y_test, seed=seed).head(6)

    lines = [
        "# 📋 Fiche modèle — Prédiction du churn",
        "",
        f"> Générée automatiquement par `python scripts/report.py` (paquet `churn` v{__version__}). "
        "Ne pas modifier à la main : relancez le script.",
        "",
        "## 1. Objectif",
        "",
        "Estimer, pour chaque client, la probabilité qu'il résilie son abonnement, afin de cibler "
        "les actions de rétention (appel, remise…) sur les clients les plus à risque.",
        "",
        "## 2. Données",
        "",
        f"- **{num(len(df), 0)} clients** synthétiques (`data/customers.csv`), "
        f"taux de churn **{pct(df['churn'].mean())}**.",
        f"- Découpage stratifié : {num(len(df) - n_test, 0)} clients pour l'entraînement, "
        f"**{num(n_test, 0)} pour le test** ({n_churn} départs).",
        "- 10 variables : ancienneté, facture mensuelle et totale, appels au support, nombre de produits, "
        "contrat, moyen de paiement, service internet, support technique, senior.",
        "",
        "## 3. Modèles comparés (jeu de test, seuil 0,5)",
        "",
        "| Modèle | Variables dérivées | ROC-AUC | Précision | Rappel | F1 |",
        "|---|:---:|---:|---:|---:|---:|",
    ]
    for (name, eng), (*_, m) in sorted(runs.items(), key=lambda kv: -kv[1][4].roc_auc):
        mark = "**" if (name, eng) == best_key else ""
        lines.append(
            f"| {mark}{name}{mark} | {'oui' if eng else 'non'} | {mark}{num(m.roc_auc)}{mark} | "
            f"{pct(m.precision)} | {pct(m.recall)} | {num(m.f1)} |"
        )
    lines += [
        "",
        f"**Modèle retenu : `{best_name}`{' avec' if best_eng else ' sans'} variables dérivées** "
        f"— ROC-AUC {num(m50.roc_auc)}. Règle : parmi les modèles à moins de {num(AUC_TOLERANCE)} "
        f"de la meilleure ROC-AUC ({num(top_auc)}), on garde le plus simple. C'est aussi celui que "
        "`scripts/train.py` sauvegarde et que l'API sert.",
        "",
        "## 4. Seuil de décision et impact financier",
        "",
        f"Hypothèses : un client perdu non détecté coûte **{euros(cost_fn)}**, une action de rétention "
        f"inutile coûte **{euros(cost_fp)}**, et une action sur un vrai partant le retient.",
        "",
        "| Stratégie | Clients contactés | Départs manqués | Coût total |",
        "|---|---:|---:|---:|",
        f"| Ne rien faire | 0 | {n_churn} | {euros(nobody)} |",
        f"| Contacter tout le monde | {num(n_test, 0)} | 0 | {euros(everybody)} |",
        f"| Modèle, seuil 0,50 | {m50.tp + m50.fp} | {m50.fn} | {euros(cost(m50))} |",
        f"| **Modèle, seuil {num(t_best, 2)} (optimal)** | **{num(m_best.tp + m_best.fp, 0)}** | "
        f"**{m_best.fn}** | **{euros(cost(m_best))}** |",
        "",
        f"Au seuil optimal **{num(t_best, 2)}** : précision {pct(m_best.precision)}, "
        f"rappel {pct(m_best.recall)}. Gain par rapport à la meilleure stratégie *sans* modèle : "
        f"**{euros(savings)}** sur {num(n_test, 0)} clients, soit **{euros(savings / n_test)} par client**.",
        "",
        f"💡 Pour un modèle bien calibré, la théorie donne un seuil optimal ≈ coût d'une action inutile ÷ "
        f"(coût action inutile + coût client perdu) = {num(theory, 2)}. Plus rater un départ coûte cher "
        "par rapport à une action, plus le seuil baisse — jusqu'à contacter presque tout le monde : "
        "le modèle sert alors surtout à **écarter les clients sûrs**.",
        "",
        "## 5. Facteurs de churn (importance par permutation)",
        "",
        "| Rang | Variable | Baisse de ROC-AUC | Vraie cause ? |",
        "|---:|---|---:|:---:|",
    ]
    for i, row in enumerate(imp.itertuples(), start=1):
        truth = "✅" if row.feature in TRUE_CAUSES else "❌ (corrélée)"
        lines.append(f"| {i} | `{row.feature}` | {num(row.importance, 4)} | {truth} |")
    lines += [
        "",
        "Les données étant synthétiques, on connaît les vraies causes (`_churn_logit` dans "
        "`src/churn/data.py`) : on peut vérifier que le modèle les retrouve.",
        "",
        "## 6. Limites et risques",
        "",
        "- **Données synthétiques** : les chiffres illustrent la méthode, pas un vrai marché.",
        "- **Coûts simplifiés** : on suppose qu'une action retient toujours le client et on ignore son coût "
        "quand elle réussit. Cela surestime le gain : à affiner avec le métier (taux de succès réel, "
        "coût de la remise).",
        "- **Corrélation ≠ causalité** : une variable importante n'est pas forcément un levier d'action.",
        "- **Dérive** : le modèle se dégrade si les clients changent. Surveiller avec "
        "`python scripts/check_drift.py` (et l'onglet « Surveillance » du tableau de bord).",
        "- **Un seul découpage train/test** : les écarts entre modèles de moins de ~0,01 d'AUC ne sont "
        "pas forcément significatifs (voir exercice 3.4, validation croisée).",
        "",
        "## 7. Reproduire",
        "",
        "```bash",
        "python scripts/train.py     # entraîne et sauvegarde le modèle",
        f"python scripts/report.py --cost-fn {cost_fn:g} --cost-fp {cost_fp:g}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cost-fn", type=float, default=500.0, help="coût d'un client perdu non détecté (€)")
    parser.add_argument("--cost-fp", type=float, default=50.0, help="coût d'une action inutile (€)")
    parser.add_argument("--output", type=Path, default=ROOT / "RAPPORT_MODELE.md")
    args = parser.parse_args()
    args.output.write_text(build_report(args.cost_fn, args.cost_fp), encoding="utf-8")
    print(f"Fiche modèle écrite : {args.output}")


if __name__ == "__main__":
    main()

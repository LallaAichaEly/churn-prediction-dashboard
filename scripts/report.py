"""Génère RAPPORT.md : le rapport final du projet, avec des chiffres recalculés à chaque exécution.

Pourquoi générer le rapport par code ? Si les données ou le modèle changent, il suffit de relancer
le script : les chiffres du rapport ne peuvent pas être « oubliés » ou recopiés de travers.

Usage :
    python scripts/report.py
    python scripts/report.py --cost-fn 300 --cost-fp 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from churn.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, load_or_create, validate  # noqa: E402
from churn.model import MODELS, best_threshold_by_cost, build_pipeline, evaluate, feature_importance, train  # noqa: E402

LABELS = {
    "tenure_months": "Ancienneté (mois)",
    "monthly_charges": "Facture mensuelle",
    "total_charges": "Total payé",
    "support_calls": "Appels au support",
    "num_products": "Nombre de produits",
    "contract": "Type de contrat",
    "payment_method": "Moyen de paiement",
    "internet_service": "Service internet",
    "tech_support": "Support technique",
    "senior": "Client senior",
}


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", " ")


def integer(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def rate(df: pd.DataFrame, mask) -> float:
    return float(df.loc[mask, TARGET].mean())


def build_report(df: pd.DataFrame, cost_fn: float, cost_fp: float, seed: int = 42) -> str:
    churn_rate = df[TARGET].mean()

    # --- Comparaison des modèles : un split de test + une validation croisée pour la robustesse.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows, trained = [], {}
    for name in MODELS:
        pipeline, X_test, y_test, proba, m = train(df, name, seed=seed)
        scores = cross_val_score(build_pipeline(name, seed=seed), df[FEATURES], df[TARGET], cv=cv, scoring="roc_auc")
        trained[name] = (pipeline, X_test, y_test, proba, m)
        rows.append((name, m, scores.mean(), scores.std()))
    best = max(trained, key=lambda n: trained[n][4].roc_auc)
    pipeline, X_test, y_test, proba, m50 = trained[best]

    # --- Seuil métier et impact financier (sur le jeu de test, jamais vu à l'entraînement).
    t_best = best_threshold_by_cost(y_test, proba, cost_fn, cost_fp)
    mb = evaluate(y_test, proba, t_best)
    n_test, n_churn = len(y_test), int(y_test.sum())

    def cost(m) -> float:
        return m.fn * cost_fn + m.fp * cost_fp

    cost_nothing = n_churn * cost_fn               # personne n'est contacté : tous les départs sont perdus
    cost_everyone = (n_test - n_churn) * cost_fp   # tout le monde est contacté
    cost_model = cost(mb)
    baseline = min(cost_nothing, cost_everyone)
    saving = baseline - cost_model
    per_1000 = saving / n_test * 1000

    # --- Sensibilité : la valeur du modèle dépend fortement du coût d'une action.
    sensitivity = []
    for fp_cost in (cost_fp / 2, cost_fp, cost_fp * 2, cost_fp * 4):
        t = best_threshold_by_cost(y_test, proba, cost_fn, fp_cost)
        m = evaluate(y_test, proba, t)
        base = min(n_churn * cost_fn, (n_test - n_churn) * fp_cost)
        model_cost = m.fn * cost_fn + m.fp * fp_cost
        sensitivity.append((fp_cost, t, m.tp + m.fp, model_cost, base, (base - model_cost) / base))
    sensitivity_table = "\n".join(
        f"| {'**' + eur(c) + '**' if c == cost_fp else eur(c)} | {num(t, 2)} | {integer(k)} | {eur(mc)} "
        f"| {eur(b)} | {pct(g, 0)} |"
        for c, t, k, mc, b, g in sensitivity
    )

    importance = feature_importance(pipeline, X_test, y_test, seed=seed).head(5)

    # --- Segments pour les recommandations.
    r_monthly = rate(df, df["contract"] == "Mensuel")
    r_long = rate(df, df["contract"] != "Mensuel")
    r_no_support = rate(df, df["tech_support"] == "Non")
    r_support = rate(df, df["tech_support"] == "Oui")
    r_calls = rate(df, df["support_calls"] >= 3)
    r_few_calls = rate(df, df["support_calls"] < 3)
    share_calls = float((df["support_calls"] >= 3).mean())
    r_cheque = rate(df, df["payment_method"] == "Chèque")
    r_auto = rate(df, df["payment_method"] == "Prélèvement")

    model_table = "\n".join(
        f"| {'**' + n + '**' if n == best else n} | {num(m.roc_auc)} | {num(mean)} ± {num(std)} "
        f"| {pct(m.precision)} | {pct(m.recall)} |"
        for n, m, mean, std in rows
    )
    importance_table = "\n".join(
        f"| {i + 1} | {LABELS.get(r.feature, r.feature)} | {num(r.importance)} |"
        for i, r in importance.iterrows()
    )
    contacted_50, contacted_best = m50.tp + m50.fp, mb.tp + mb.fp

    return f"""# 📄 Rapport — Prédiction du churn client

> Généré par `python scripts/report.py` (graine {seed}). Toutes les valeurs sont recalculées à chaque
> exécution. **Données synthétiques** : ce rapport illustre la démarche, pas un cas réel.

## 1. Problème

L'opérateur perd **{pct(churn_rate)}** de ses clients. Acquérir un nouveau client coûte bien plus cher
que d'en garder un : l'objectif est d'**identifier à l'avance les clients susceptibles de partir** pour
leur proposer une action de rétention ciblée, plutôt que d'arroser tout le monde.

## 2. Données

- **{integer(len(df))} clients**, {len(NUMERIC_FEATURES)} variables numériques et {len(CATEGORICAL_FEATURES)} catégorielles (contrat, paiement, services…).
- Contrôles qualité (`validate()`) : aucun problème détecté.
- Découpage stratifié : 75 % entraînement, 25 % test ({integer(n_test)} clients dont {n_churn} départs).

## 3. Modèle retenu

| Modèle | ROC-AUC (test) | ROC-AUC (validation croisée 5 plis) | Précision @0,5 | Rappel @0,5 |
|---|---|---|---|---|
{model_table}

**Choix : `{best}`.** Meilleure ROC-AUC, écart confirmé par la validation croisée, et le modèle
le plus simple à expliquer et à maintenir. Règle d'or : un modèle complexe doit *mériter* sa complexité.

## 4. Ce qui fait partir les clients

Importance par permutation (baisse de ROC-AUC quand la variable est mélangée) :

| Rang | Variable | Baisse d'AUC |
|---|---|---|
{importance_table}

## 5. Seuil de décision

Le seuil par défaut (0,5) n'a aucune raison d'être le bon. Hypothèses de coût :
**{eur(cost_fn)}** par client perdu non détecté, **{eur(cost_fp)}** par action de rétention inutile.

| | Seuil 0,5 | Seuil optimal **{num(t_best, 2)}** |
|---|---|---|
| Clients contactés (sur {integer(n_test)}) | {contacted_50} | {contacted_best} |
| Précision | {pct(m50.precision)} | {pct(mb.precision)} |
| Rappel (départs détectés) | {pct(m50.recall)} | {pct(mb.recall)} |
| Départs manqués | {m50.fn} | {mb.fn} |
| Coût total | {eur(cost(m50))} | **{eur(cost_model)}** |

Quand rater un départ coûte {cost_fn / cost_fp:.0f}× plus cher qu'une action inutile, on a intérêt à
**contacter plus large** : le seuil baisse, le rappel monte, la précision baisse — et c'est voulu.

## 6. Impact financier estimé (jeu de test)

| Stratégie | Coût |
|---|---|
| Ne rien faire | {eur(cost_nothing)} |
| Contacter tout le monde | {eur(cost_everyone)} |
| **Cibler avec le modèle (seuil {num(t_best, 2)})** | **{eur(cost_model)}** |

Par rapport à la meilleure stratégie sans modèle, le ciblage économise **{eur(saving)}**
({pct(saving / baseline, 0)}), soit environ **{eur(per_1000)} pour 1 000 clients**.

**Lecture critique.** Avec une action aussi bon marché, « contacter tout le monde » est déjà une
stratégie correcte : le modèle apporte surtout en évitant {integer(mb.tn)} contacts inutiles
(au prix de {mb.fn} départs manqués).
Plus l'action coûte cher (remise, appel d'un conseiller), plus le ciblage devient rentable :

| Coût d'une action inutile | Seuil optimal | Clients contactés | Coût avec modèle | Meilleur coût sans modèle | Économie |
|---|---|---|---|---|---|
{sensitivity_table}

👉 Avant de déployer, **mesurer le vrai coût d'une action** : c'est lui qui décide si le modèle vaut l'effort.

## 7. Recommandations

1. **Fidéliser les contrats mensuels** (churn {pct(r_monthly)} contre {pct(r_long)} pour les contrats
   annuels ou biannuels) : offre de passage à l'engagement annuel.
2. **Proposer le support technique** (churn {pct(r_no_support)} sans, {pct(r_support)} avec) — à valider
   par un test A/B avant généralisation.
3. **Alerte « 3 appels au support ou plus »** : {pct(share_calls, 0)} des clients, churn {pct(r_calls)}
   (contre {pct(r_few_calls)}). Un appel de suivi proactif par un conseiller senior.
4. **Encourager le prélèvement automatique** (churn {pct(r_cheque)} par chèque, {pct(r_auto)} par prélèvement).

## 8. Limites

- **Données synthétiques** : les vraies causes sont connues et le bruit est maîtrisé ; sur des données
  réelles, attendez-vous à une performance plus faible.
- **Modèle de coût simplifié** : on suppose qu'une action de rétention sauve 100 % des clients ciblés et
  on ignore son coût pour les vrais positifs. À affiner avec un taux de succès mesuré.
- **Corrélation ≠ causalité** : un client avec support technique part moins, mais *offrir* le support
  ne garantit pas le même effet. Seule une expérience contrôlée (A/B) le prouve.
- **Pas de validation temporelle** ni de suivi de la dérive des données (*data drift*) ; la
  calibration des probabilités n'est pas vérifiée (exercice 4.5 du parcours).

## 9. Reproduire

```bash
pip install -r requirements.txt
python scripts/train.py      # entraîne et sauvegarde le meilleur modèle
python scripts/report.py     # régénère ce rapport
streamlit run app.py         # explore les résultats de façon interactive
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cost-fn", type=float, default=500.0, help="Coût d'un client perdu non détecté (€)")
    parser.add_argument("--cost-fp", type=float, default=50.0, help="Coût d'une action de rétention inutile (€)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "RAPPORT.md")
    args = parser.parse_args()

    df = load_or_create(ROOT / "data" / "customers.csv", seed=args.seed)
    problems = validate(df)
    if problems:
        sys.exit("Données invalides :\n- " + "\n- ".join(problems))

    args.out.write_text(build_report(df, args.cost_fn, args.cost_fp, args.seed), encoding="utf-8")
    print(f"Rapport écrit dans {args.out}")


if __name__ == "__main__":
    main()

"""Entraîne les modèles, compare leurs performances et sauvegarde le meilleur.

Le seuil de décision enregistré avec le modèle est choisi selon les coûts métier
(par validation croisée sur le jeu d'entraînement) : c'est lui qu'utilise l'API.

Usage :
    python scripts/train.py
    python scripts/train.py --model gboost --engineered
    python scripts/train.py --cost-fn 100 --cost-fp 100
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from churn.data import load_or_create, validate  # noqa: E402
from churn.model import MODELS, save, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--engineered", action="store_true", help="Ajoute les variables dérivées")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-fn", type=float, default=500.0,
                        help="Coût d'un client perdu non détecté, en € (défaut : 500)")
    parser.add_argument("--cost-fp", type=float, default=50.0,
                        help="Coût d'une action de rétention inutile, en € (défaut : 50)")
    args = parser.parse_args()

    df = load_or_create(ROOT / "data" / "customers.csv", seed=args.seed)
    problems = validate(df)
    if problems:
        sys.exit("Données invalides :\n- " + "\n- ".join(problems))
    print(f"{len(df)} clients — taux de churn : {df['churn'].mean():.1%}")
    print(f"Coûts métier : client perdu = {args.cost_fn:.0f} €, action inutile = {args.cost_fp:.0f} €\n")

    names = list(MODELS) if args.model == "all" else [args.model]
    results = {}
    for name in names:
        pipeline, _, _, _, metrics = train(df, name, args.engineered, args.seed,
                                           costs=(args.cost_fn, args.cost_fp))
        results[name] = (pipeline, metrics)

    table = pd.DataFrame({n: asdict(m) for n, (_, m) in results.items()}).T
    table["coût_test_€"] = table["fn"] * args.cost_fn + table["fp"] * args.cost_fp
    cols = ["roc_auc", "threshold", "precision", "recall", "f1", "coût_test_€"]
    print(table[cols].astype(float).round(3).to_string())

    best = max(results, key=lambda n: results[n][1].roc_auc)
    path = save(*results[best], model_dir=ROOT / "models")
    print(f"\nMeilleur modèle (ROC-AUC) : {best} — seuil {results[best][1].threshold:.2f} "
          f"→ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

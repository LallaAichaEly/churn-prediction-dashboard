"""Entraîne les modèles, compare leurs performances et sauvegarde le meilleur.

Usage :
    python scripts/train.py
    python scripts/train.py --model gboost --engineered
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--engineered", action="store_true", help="Ajoute les variables dérivées")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_or_create(ROOT / "data" / "customers.csv", seed=args.seed)
    problems = validate(df)
    if problems:
        sys.exit("Données invalides :\n- " + "\n- ".join(problems))
    print(f"{len(df)} clients — taux de churn : {df['churn'].mean():.1%}\n")

    names = list(MODELS) if args.model == "all" else [args.model]
    results = {}
    for name in names:
        pipeline, _, _, _, metrics = train(df, name, args.engineered, args.seed)
        results[name] = (pipeline, metrics)

    table = pd.DataFrame({n: asdict(m) for n, (_, m) in results.items()}).T
    print(table[["roc_auc", "accuracy", "precision", "recall", "f1"]].round(3).to_string())

    best = max(results, key=lambda n: results[n][1].roc_auc)
    path = save(*results[best], model_dir=ROOT / "models")
    print(f"\nMeilleur modèle (ROC-AUC) : {best} → {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Compare de nouvelles données aux données d'entraînement et signale les dérives (PSI).

Usage :
    python scripts/check_drift.py --current nouveaux_clients.csv
    python scripts/check_drift.py --price-increase 0.2          # simulation
    python scripts/check_drift.py --extra-calls 1 --to-monthly 0.5

Code de sortie : 0 si aucune dérive significative, 1 sinon. Ce script peut donc
tourner chaque nuit (cron, GitHub Action…) et déclencher une alerte en cas d'échec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from churn.data import FEATURES, load_or_create  # noqa: E402
from churn.monitoring import SIGNIFICANT, drift_report, drift_status, psi_numeric, simulate_new_period  # noqa: E402
from churn.model import load  # noqa: E402

ICONS = {"stable": "✅", "modérée": "⚠️ ", "significative": "🔴"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", type=Path, help="CSV des nouveaux clients (sinon : simulation)")
    parser.add_argument("--price-increase", type=float, default=0.0, help="ex. 0.2 = +20 %%")
    parser.add_argument("--extra-calls", type=float, default=0.0, help="appels au support en plus (moyenne)")
    parser.add_argument("--to-monthly", type=float, default=0.0, help="part des contrats longs → mensuel")
    args = parser.parse_args()

    reference = load_or_create(ROOT / "data" / "customers.csv")
    if args.current:
        current = pd.read_csv(args.current)
        missing = [c for c in FEATURES if c not in current.columns]
        if missing:
            sys.exit(f"Colonnes manquantes dans {args.current} : {missing}")
        source = str(args.current)
    else:
        current = simulate_new_period(
            price_increase=args.price_increase, extra_support_calls=args.extra_calls, to_monthly=args.to_monthly
        )
        source = "simulation"

    print(f"Référence : {len(reference)} clients · Nouvelle période : {len(current)} clients ({source})\n")
    report = drift_report(reference, current)
    for row in report.itertuples():
        print(f"  {ICONS[row.statut]} {row.feature:<18} PSI = {row.psi:6.3f}  ({row.statut})")

    model_path = ROOT / "models" / "churn_model.joblib"
    if model_path.exists():
        pipeline = load(model_path)
        p_ref = pipeline.predict_proba(reference[FEATURES])[:, 1]
        p_cur = pipeline.predict_proba(current[FEATURES])[:, 1]
        score_psi = psi_numeric(p_ref, p_cur)
        print(
            f"\n  Score du modèle : PSI = {score_psi:.3f} ({drift_status(score_psi)}) · "
            f"churn moyen prédit {p_ref.mean():.1%} → {p_cur.mean():.1%}"
        )
    else:
        print("\n  (Pas de modèle sauvegardé : lancez python scripts/train.py pour suivre aussi le score.)")

    drifted = report.loc[report["psi"] >= SIGNIFICANT, "feature"].tolist()
    if drifted:
        print(f"\n🔴 Dérive significative : {', '.join(drifted)}. Envisagez de réentraîner le modèle.")
        return 1
    print("\n✅ Aucune dérive significative.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

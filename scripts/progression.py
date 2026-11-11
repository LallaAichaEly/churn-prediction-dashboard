"""Affiche votre progression sur les défis auto-corrigés.

Usage :
    python scripts/progression.py              # corrige vos réponses (defis/serie_*.py)
    python scripts/progression.py --defi 5     # détail d'un seul défi
    python scripts/progression.py --corrige    # vérifie le corrigé (tout doit être ✅)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defis.verif import ICONES, REUSSI, evaluer, series  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--defi", type=int, help="n'afficher qu'un défi, avec le détail")
    parser.add_argument("--corrige", action="store_true", help="évaluer le corrigé au lieu de vos réponses")
    args = parser.parse_args()

    total = reussis = 0
    prochain = None
    for serie in series():
        resultats = evaluer(serie, corrige=args.corrige)
        if args.defi is not None:
            resultats = [r for r in resultats if r.defi.numero == args.defi]
            if not resultats:
                continue
        ok = sum(r.statut == REUSSI for r in resultats)
        total, reussis = total + len(resultats), reussis + ok
        fichier = serie.module_corrige if args.corrige else serie.module_eleve
        print(f"\n🧭 Série {serie.code} — {serie.titre}  ({ok}/{len(resultats)} réussis)  · {fichier.replace('.', '/')}.py\n")
        for r in resultats:
            d = r.defi
            ligne = f"  {ICONES[r.statut]} {d.numero:>2} {d.niveau} {d.fonction:<24} Parcours {d.parcours}"
            print(ligne)
            if r.message:
                print(f"        ↳ {r.message}")
            if prochain is None and r.statut != REUSSI:
                prochain = (serie, d)

    if total == 0:
        sys.exit(f"Aucun défi n°{args.defi}.")
    barre = "█" * round(20 * reussis / total) + "░" * (20 - round(20 * reussis / total))
    print(f"\n  {barre}  {reussis}/{total}\n")
    if prochain:
        serie, d = prochain
        print(f"👉 Prochain défi : n°{d.numero} `{d.fonction}` dans defis/serie_{serie.code}.py")
        print("   Lisez sa docstring, codez, relancez ce script.\n")
    else:
        print("🏆 Tout est réussi ! Comparez vos solutions au corrigé et notez ce que vous avez appris.\n")


if __name__ == "__main__":
    main()

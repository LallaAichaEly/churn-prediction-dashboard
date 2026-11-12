"""Corrige votre projet guidé (projets/retention/) et affiche ✅ / ❌ / ⬜ pour chaque étape.

Usage :
    python scripts/projet.py              # votre progression
    python scripts/projet.py --etape 3    # une seule étape
    python scripts/projet.py --corrige    # vérifie le projet terminé (tout doit être ✅)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projets.verif import ICONES, REUSSI, evaluer, projets  # noqa: E402


def _modeles_modifie(projet) -> bool:
    """Le correcteur suppose que modeles.py (déjà écrit) n'a pas été modifié."""
    eleve = ROOT / projet.dossier() / "modeles.py"
    corrige = ROOT / projet.dossier(corrige=True) / "modeles.py"
    return eleve.exists() and corrige.exists() and eleve.read_bytes() != corrige.read_bytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--etape", type=int, help="n'afficher qu'une étape")
    parser.add_argument("--corrige", action="store_true", help="évaluer le corrigé au lieu de votre code")
    args = parser.parse_args(argv)

    for projet in projets():
        if args.etape is not None and all(e.numero != args.etape for e in projet.etapes):
            sys.exit(f"Aucune étape n°{args.etape} (1 à {len(projet.etapes)}).")
        resultats = evaluer(projet, corrige=args.corrige)
        reussis = sum(r.statut == REUSSI for r in resultats)
        print(f"\n🏗️  Projet {projet.numero} — {projet.titre}  ({reussis}/{len(resultats)} étapes)  · {projet.dossier(args.corrige)}\n")
        if not args.corrige and _modeles_modifie(projet):
            print(f"  ⚠️  {projet.dossier()}modeles.py a été modifié : le correcteur suppose le fichier d'origine.\n"
                  f"     Pour le restaurer : git checkout -- {projet.dossier()}modeles.py\n")

        largeur = max(len(e.module) for e in projet.etapes) + 4
        premier_echec = None
        for r in resultats:
            e = r.etape
            if args.etape is not None and e.numero != args.etape:
                continue
            print(f"  {ICONES[r.statut]} {e.numero} {e.niveau} {e.module + '.py':<{largeur}} {e.titre}")
            if r.message:
                print(f"        ↳ {r.message}")
            if r.statut != REUSSI and premier_echec is None:
                premier_echec = e

        plein = round(20 * reussis / len(resultats))
        print(f"\n  {'█' * plein}{'░' * (20 - plein)}  {reussis}/{len(resultats)}\n")
        if premier_echec:
            e = premier_echec
            print(f"👉 Prochaine étape : {e.numero} — {e.titre}, dans {projet.dossier(args.corrige)}{e.module}.py")
            print(f"   Lisez la docstring de {', '.join(f'`{f}`' for f in e.fonctions)}, codez, puis : "
                  f"python scripts/projet.py --etape {e.numero}\n")
        else:
            commande = projet.commande.replace(projet.paquet_eleve, projet.paquet_corrige) if args.corrige \
                else projet.commande
            print("🏁 Projet terminé ! Votre programme fonctionne pour de vrai :")
            print(f"   {commande}")
            print("   Comparez ensuite votre code au corrigé (projets/retention_corrige/) et notez dans JOURNAL.md\n"
                  "   ce que vous feriez autrement. Idées pour aller plus loin : projets/README.md\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

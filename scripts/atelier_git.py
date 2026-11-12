"""Atelier Git : 8 situations réelles à résoudre dans un bac à sable, corrigées automatiquement.

Usage :
    python scripts/atelier_git.py                  # votre progression (✅ / ❌ / ⬜) et le prochain scénario
    python scripts/atelier_git.py 1                # prépare le scénario 1, affiche la mission… puis le vérifie
    python scripts/atelier_git.py 1 --indice       # un indice (--indice 2 pour le suivant…)
    python scripts/atelier_git.py 1 --solution     # la solution commentée (après avoir cherché !)
    python scripts/atelier_git.py 1 --recommencer  # remet le scénario à zéro

Le bac à sable est créé dans ~/atelier-git (ou --dossier, ou la variable ATELIER_GIT), en dehors du
projet : une fausse manœuvre ne peut pas abîmer votre vrai dépôt.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atelier_git.atelier import (A_FAIRE, ICONES, REUSSI, SCENARIOS, Resultat, dossier, est_prepare,  # noqa: E402
                                 evaluer, preparer, scenario)
from atelier_git.outils import GitIntrouvable, racine_par_defaut, verifier_git_installe  # noqa: E402


def _commande() -> str:
    """La commande à recopier depuis n'importe quel dossier (chemin absolu du script)."""
    script = str(Path(__file__).resolve())
    return f'python "{script}"' if " " in script else f"python {script}"


def _bloc(texte: str, retrait: str = "   ", premiere_ligne: str | None = None) -> str:
    return textwrap.fill(texte, width=100, initial_indent=premiere_ligne or retrait, subsequent_indent=retrait)


def _afficher_mission(s, travail: Path, complet: bool = True) -> None:
    print(f"\n🌿 Scénario {s.numero} {s.niveau} — {s.titre}")
    if complet:
        print(f"   Vous pratiquez : {s.commandes}\n")
        print(_bloc(s.situation) + "\n")
    print("   🎯 Mission :")
    for m in s.mission:
        print(textwrap.fill(m, width=100, initial_indent="     • ", subsequent_indent="       "))
    print(f"   📁 {travail}\n")
    if not complet:
        return
    print(f"👉 Allez dans le bac à sable et travaillez avec git :\n     cd \"{travail}\"")
    print(f"   Puis vérifiez (depuis n'importe où) :\n     {_commande()} {s.numero}")
    print(f"   Bloqué·e ? {_commande()} {s.numero} --indice\n")


def _afficher_resultat(r: Resultat) -> None:
    s = r.scenario
    if r.statut == REUSSI:
        print(f"✅ Scénario {s.numero} réussi !\n")
        print("   📌 À retenir :")
        print(_bloc(s.a_retenir, "      ") + "\n")
        suivant = next((x for x in SCENARIOS if x.numero == s.numero + 1), None)
        if suivant:
            print(f"👉 Suivant : {suivant.numero} — {suivant.titre} : {_commande()} {suivant.numero}\n")
        else:
            print("🏆 Atelier terminé ! Notez dans JOURNAL.md la commande qui vous a le plus servi.\n")
    elif r.statut == A_FAIRE:
        print(f"⬜ {r.message[0].upper() + r.message[1:]}. Relisez la mission ci-dessus et lancez-vous.\n")
    else:
        print(f"{ICONES[r.statut]} Pas encore :")
        print(_bloc(r.message, "     ", premiere_ligne="   ↳ ") + "\n")
        print(f"   Indice : {_commande()} {s.numero} --indice · tout reprendre : --recommencer\n")


def _tableau(racine: Path) -> int:
    resultats = [evaluer(s, racine) for s in SCENARIOS]
    reussis = sum(r.statut == REUSSI for r in resultats)
    print(f"\n🌿 Atelier Git — {reussis}/{len(SCENARIOS)} scénarios réussis · bac à sable : {racine}\n")
    largeur = max(len(s.titre) for s in SCENARIOS) + 2
    for r in resultats:
        s = r.scenario
        print(f"  {ICONES[r.statut]} {s.numero} {s.niveau} {s.titre:<{largeur}} {s.commandes}")
        if r.statut != REUSSI and r.message and r.message != "pas encore préparé":
            print(textwrap.fill(r.message, width=110, initial_indent="        ↳ ", subsequent_indent="          "))
    plein = round(16 * reussis / len(SCENARIOS))
    print(f"\n  {'█' * plein}{'░' * (16 - plein)}  {reussis}/{len(SCENARIOS)}\n")
    prochain = next((r.scenario for r in resultats if r.statut != REUSSI), None)
    if prochain:
        verbe = "Reprenez" if est_prepare(prochain, racine) else "Commencez"
        print(f"👉 {verbe} le scénario {prochain.numero} — {prochain.titre} : {_commande()} {prochain.numero}\n")
    else:
        print("🏆 Les 8 scénarios sont réussis : vous avez les réflexes git d'un·e développeur·se en équipe.\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:  # la console Windows n'est pas toujours en UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("numero", type=int, nargs="?", help=f"numéro du scénario (1 à {len(SCENARIOS)})")
    parser.add_argument("--indice", type=int, nargs="?", const=1, metavar="N", help="affiche l'indice N (1 par défaut)")
    parser.add_argument("--solution", action="store_true", help="affiche la solution commentée")
    parser.add_argument("--recommencer", action="store_true", help="remet le scénario à zéro")
    parser.add_argument("--dossier", type=Path, help="dossier du bac à sable (défaut : ~/atelier-git)")
    args = parser.parse_args(argv)
    racine = (args.dossier or racine_par_defaut()).expanduser().resolve()

    if args.numero is None:
        if args.indice or args.solution or args.recommencer:
            parser.error("précisez le numéro du scénario, par exemple : atelier_git.py 1 --indice")
        try:
            verifier_git_installe()
        except GitIntrouvable as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 1
        return _tableau(racine)

    try:
        s = scenario(args.numero)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2

    if args.indice is not None:
        n = args.indice
        if not 1 <= n <= len(s.indices):
            print(f"Le scénario {s.numero} a {len(s.indices)} indices (1 à {len(s.indices)}).", file=sys.stderr)
            return 2
        print(f"\n💡 Scénario {s.numero}, indice {n}/{len(s.indices)} :\n")
        print(_bloc(s.indices[n - 1]) + "\n")
        if n < len(s.indices):
            print(f"   Encore bloqué·e ? {_commande()} {s.numero} --indice {n + 1}\n")
        return 0

    if args.solution:
        print(f"\n🔓 Solution du scénario {s.numero} — {s.titre}")
        print("   (d'autres chemins mènent aussi au ✅ : le correcteur vérifie le résultat, pas les commandes)\n")
        for pas in s.solution:
            if pas.est_commande:
                ligne = f"   $ {pas.texte}"
                print(f"{ligne:<60} # {pas.commentaire}" if pas.commentaire else ligne)
            else:
                print(f"   {pas.texte}")
        print("\n   📌 À retenir :")
        print(_bloc(s.a_retenir, "      ") + "\n")
        return 0

    try:
        if args.recommencer or not est_prepare(s, racine):
            deja = est_prepare(s, racine)
            travail = preparer(s, racine, _commande())
            print(f"\n🌱 Scénario {s.numero} {'remis à zéro' if deja else 'préparé'} dans {travail}"
                  + (" (si votre terminal y était, refaites le `cd`)" if deja else ""))
            _afficher_mission(s, travail)
            return 0
    except (GitIntrouvable, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    resultat = evaluer(s, racine)
    _afficher_mission(s, dossier(s, racine), complet=resultat.statut == A_FAIRE)
    _afficher_resultat(resultat)
    return 0


if __name__ == "__main__":
    sys.exit(main())

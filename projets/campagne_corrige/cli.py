"""Étape 6 (corrigé) — Un programme, trois commandes : préparer, importer, faire le bilan."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from .base import enregistrer_campagne, ouvrir
from .fichiers import ecrire_contacts, formater_ligne, lire_plan
from .mesure import compter, regrouper
from .modeles import PART_TEMOIN, TOUTES
from .resultats import importer_resultats
from .statistiques import effet


def _proportion(texte: str) -> float:
    """Type argparse : un nombre strictement entre 0 et 1 (« 0.2 » ou « 0,2 »)."""
    try:
        valeur = float(texte.replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError(f"« {texte} » n'est pas un nombre") from None
    if not 0 < valeur < 1:
        raise argparse.ArgumentTypeError(f"attendu un nombre strictement entre 0 et 1, pas « {texte} »")
    return valeur


def creer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m projets.campagne",
        description="Mesure l'effet d'une campagne de rétention grâce à un groupe témoin tiré au hasard.",
    )
    # Une sous-commande par moment de la campagne, comme `git commit` / `git push`.
    commandes = parser.add_subparsers(dest="commande", required=True, metavar="COMMANDE")

    preparer = commandes.add_parser("preparer", help="tire le groupe témoin et écrit les clients à contacter")
    preparer.add_argument("--plan", required=True, type=Path, help="plan de rétention (sortie du projet 2)")
    preparer.add_argument("--base", required=True, type=Path, help="base SQLite de la campagne (créée si besoin)")
    preparer.add_argument("--graine", required=True, help="texte qui fixe le tirage (ex. 2026-11)")
    preparer.add_argument("--part-temoin", type=_proportion, default=PART_TEMOIN,
                          help=f"part des clients mis de côté (défaut : {PART_TEMOIN})")
    preparer.add_argument("--contacts", type=Path, default=Path("a_contacter.csv"),
                          help="fichier des clients à contacter (défaut : a_contacter.csv)")

    importer = commandes.add_parser("importer", help="enregistre les résultats envoyés par le CRM")
    importer.add_argument("--base", required=True, type=Path)
    importer.add_argument("--resultats", required=True, type=Path, help="CSV : customer_id, parti")

    bilan = commandes.add_parser("bilan", help="affiche l'effet de chaque offre")
    bilan.add_argument("--base", required=True, type=Path)
    bilan.add_argument("--niveau", type=_proportion, default=0.95,
                       help="niveau de confiance des intervalles (défaut : 0.95)")
    return parser


def _preparer(args: argparse.Namespace) -> int:
    ciblages = lire_plan(args.plan)
    # closing() FERME la connexion à la sortie du bloc, même en cas d'erreur.
    with closing(ouvrir(args.base)) as conn:
        ajoutes = enregistrer_campagne(conn, ciblages, args.graine, args.part_temoin)
        ecrire_contacts(conn, args.contacts)
        offre, temoin = regrouper(compter(conn))[TOUTES]
    print(f"Plan lu        : {len(ciblages)} clients ({ajoutes} nouveaux dans {args.base})")
    print(f"Groupe offre   : {offre.clients} clients à contacter → {args.contacts}")
    print(f"Groupe témoin  : {temoin.clients} clients à NE PAS contacter")
    return 0


def _base_existante(chemin: Path) -> bool:
    # Sans ce test, sqlite3.connect créerait une base VIDE et le programme dirait « 0 résultat »
    # au lieu de signaler la faute de frappe.
    if chemin.exists():
        return True
    print(f"Erreur : base introuvable : {chemin}. Lancez d'abord la commande `preparer`.", file=sys.stderr)
    return False


def _importer(args: argparse.Namespace) -> int:
    if not _base_existante(args.base):
        return 1
    with closing(ouvrir(args.base)) as conn:
        enregistres, erreurs = importer_resultats(conn, args.resultats)
    for erreur in erreurs:  # stderr : stdout reste propre si on redirige le résumé
        print(f"Attention : {erreur}", file=sys.stderr)
    print(f"Résultats enregistrés : {enregistres} ({len(erreurs)} ligne(s) ignorée(s))")
    return 0


def _bilan(args: argparse.Namespace) -> int:
    if not _base_existante(args.base):
        return 1
    with closing(ouvrir(args.base)) as conn:
        groupes = regrouper(compter(conn))
    offre, temoin = groupes[TOUTES]
    if offre.clients + temoin.clients == 0:
        print("Erreur : la base ne contient aucune campagne. Lancez d'abord la commande `preparer`.",
              file=sys.stderr)
        return 1
    for nom, (offre, temoin) in groupes.items():
        try:
            mesure = effet(offre, temoin, args.niveau)
        except ValueError:  # pas de résultat dans l'un des groupes : on le dit, sans planter
            mesure = None
        print(formater_ligne(nom, offre, temoin, mesure))
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = creer_parser().parse_args(argv)
    commande = {"preparer": _preparer, "importer": _importer, "bilan": _bilan}[args.commande]
    # Les erreurs prévisibles deviennent un message clair et le code de sortie 1, sans traceback.
    try:
        return commande(args)
    except FileNotFoundError as exc:
        print(f"Erreur : fichier introuvable : {exc.filename}", file=sys.stderr)
    except ValueError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
    except sqlite3.Error as exc:  # ex. --base pointe vers un fichier qui n'est pas une base SQLite
        print(f"Erreur : base {args.base} : {exc}", file=sys.stderr)
    return 1

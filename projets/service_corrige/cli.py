"""Étape 6 (corrigé) — Le programme à trois commandes : importer, avancement, servir."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Callable, Mapping

from .api import creer_app
from .base import importer_contacts, ouvrir
from .config import lire_config
from .contacts import avancement
from .fichiers import formater_avancement, lire_contacts
from .modeles import ConfigInvalide

PROGRAMME = "python -m projets.service_corrige"


def _port(texte: str) -> int:
    try:
        port = int(texte)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{texte!r} n'est pas un numéro de port") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"le port doit être entre 1 et 65535, reçu {port}")
    return port


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAMME,
        description="Le service des conseillers. Configuration : variables SERVICE_CLES (obligatoire), "
                    "SERVICE_BASE et SERVICE_TAILLE_PAGE.")
    commandes = parser.add_subparsers(dest="commande", required=True)
    importer = commandes.add_parser("importer", help="répartir une liste de contacts entre les conseillers")
    importer.add_argument("--contacts", type=Path, required=True, help="le fichier a_contacter.csv du projet 3")
    commandes.add_parser("avancement", help="où en est l'équipe")
    servir = commandes.add_parser("servir", help="lancer l'API web")
    servir.add_argument("--hote", default="127.0.0.1", help="127.0.0.1 : accessible depuis cette machine seulement")
    servir.add_argument("--port", type=_port, default=8000)
    return parser


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None,
         lancer: Callable[..., object] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        # La configuration est lue UNE fois, ici ; le reste du programme reçoit `config`.
        config = lire_config(os.environ if env is None else env)
    except ConfigInvalide as exc:
        print(exc, file=sys.stderr)  # le message de l'étape 1 ne contient aucune clé : on peut l'afficher
        return 1

    if args.commande == "importer":
        try:
            liste = lire_contacts(args.contacts)
            with closing(ouvrir(config.base)) as conn:
                ajoutes = importer_contacts(conn, liste, config.conseillers)
                tableau = formater_avancement(avancement(conn))
        except (OSError, ValueError, sqlite3.Error) as exc:
            print(f"Import impossible : {exc}", file=sys.stderr)
            return 1
        print(f"Contacts lus : {len(liste)} ({ajoutes} nouveaux, répartis entre "
              f"{len(config.conseillers)} conseillers)\n")
        print(tableau)
        return 0

    # avancement et servir travaillent sur une base qui doit déjà exister : sqlite3.connect en
    # créerait une vide, et le service démarrerait… sans aucun contact.
    if not config.base.exists():
        print(f"La base {config.base} n'existe pas : lancez d'abord `{PROGRAMME} importer --contacts …` "
              "(ou vérifiez SERVICE_BASE).", file=sys.stderr)
        return 1
    try:
        # Ouvrir la base AVANT de démarrer : un fichier abîmé est signalé maintenant, pas à la
        # première requête d'un conseiller.
        with closing(ouvrir(config.base)) as conn:
            tableau = formater_avancement(avancement(conn))
    except sqlite3.Error as exc:
        print(f"Base illisible ({config.base}) : {exc}", file=sys.stderr)
        return 1

    if args.commande == "avancement":
        print(tableau)
        return 0

    if lancer is None:  # en vrai : uvicorn. Dans les tests : une fonction qui note ce qu'on lui passe.
        import uvicorn
        lancer = uvicorn.run
    print(f"Service des conseillers : http://{args.hote}:{args.port}/docs "
          f"({len(config.conseillers)} conseillers : {', '.join(config.conseillers)} ; base {config.base})")
    lancer(creer_app(config), host=args.hote, port=args.port)
    return 0

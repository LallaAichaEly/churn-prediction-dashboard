"""Étape 6 🟡 — Le programme à trois commandes : importer, avancement, servir.

Vérifier : python scripts/projet.py --projet 4 --etape 6
"""

from __future__ import annotations

import argparse  # noqa: F401
import os  # noqa: F401
import sqlite3  # noqa: F401
import sys  # noqa: F401
from contextlib import closing  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Callable, Mapping

from .api import creer_app  # noqa: F401
from .base import importer_contacts, ouvrir  # noqa: F401
from .config import lire_config  # noqa: F401
from .contacts import avancement  # noqa: F401
from .fichiers import formater_avancement, lire_contacts  # noqa: F401
from .modeles import ConfigInvalide  # noqa: F401

PROGRAMME = "python -m projets.service"


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None,
         lancer: Callable[..., object] | None = None) -> int:
    """Le point d'entrée du programme. Renvoie le code de sortie (0 = tout va bien).

        python -m projets.service importer --contacts projets/exemple_contacts.csv
        python -m projets.service avancement
        python -m projets.service servir [--hote 127.0.0.1] [--port 8000]

    `env` : les variables d'environnement (`os.environ` s'il vaut None). `lancer` : la fonction qui
    démarre le serveur (`uvicorn.run` s'il vaut None ; le correcteur passe la sienne, qui note ce
    qu'elle reçoit au lieu de démarrer un vrai serveur).

    1. Trois sous-commandes `argparse`, obligatoires (`add_subparsers(dest="commande", required=True)`).
       `servir --port` : un entier entre 1 et 65535, vérifié par argparse (une fonction pour `type=`
       qui lève `argparse.ArgumentTypeError`) ; `--hote` vaut 127.0.0.1 par défaut (le service n'est
       alors accessible que depuis cette machine), `--port` 8000.
    2. Lisez la configuration UNE fois : `lire_config(env)`. Une `ConfigInvalide` : affichez son
       message sur `sys.stderr`, code 1 (son message ne contient aucune clé, c'est l'étape 1).
    3. `importer` : `lire_contacts(args.contacts)`, puis `importer_contacts(conn, liste,
       config.conseillers)` ; affichez le nombre de contacts lus et de NOUVEAUX contacts, puis le
       tableau `formater_avancement(avancement(conn))`.
    4. `avancement` et `servir` travaillent sur une base qui doit exister : si `config.base` n'existe
       pas, message sur stderr qui conseille de lancer `importer` d'abord, et code 1 (sans créer la
       base : `sqlite3.connect` en créerait une vide). Ouvrez-la ensuite pour vérifier qu'elle est
       lisible : un fichier abîmé doit être signalé au démarrage, pas à la première requête.
    5. `avancement` : affichez le tableau `formater_avancement(avancement(conn))`.
    6. `servir` : affichez l'adresse et le NOM des conseillers configurés (jamais leurs clés), puis
       `lancer(creer_app(config), host=args.hote, port=args.port)`.
    7. Les erreurs prévisibles (fichier introuvable : `OSError` ; fichier mal formé : `ValueError` ;
       base illisible : `sqlite3.Error`) : un message clair sur stderr et le code 1, jamais une
       traceback. Fermez vos connexions (`with closing(ouvrir(...)) as conn:`).

    Ce que vous apprenez : assembler six modules, sous-commandes, configuration lue une fois,
    injection de dépendance (`lancer`) pour tester un programme qui démarre un serveur.
    """
    raise NotImplementedError

"""Étape 6 🟡 — Un programme, trois commandes : préparer, importer, faire le bilan.

Vérifier : python scripts/projet.py --projet 3 --etape 6
"""

from __future__ import annotations

import argparse  # noqa: F401 — vous en aurez besoin
import sqlite3  # noqa: F401
import sys  # noqa: F401
from contextlib import closing  # noqa: F401
from pathlib import Path  # noqa: F401

from .base import enregistrer_campagne, ouvrir  # noqa: F401
from .fichiers import ecrire_contacts, formater_ligne, lire_plan  # noqa: F401
from .mesure import compter, regrouper  # noqa: F401
from .modeles import PART_TEMOIN, TOUTES  # noqa: F401
from .resultats import importer_resultats  # noqa: F401
from .statistiques import effet  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    """Lit la ligne de commande (`argv`, ou `sys.argv` si None), exécute la commande demandée et
    renvoie le code de sortie : 0 si tout va bien, 1 pour une erreur prévisible.

    Trois sous-commandes, une par moment de la campagne (comme `git commit` et `git push`) :
    `parser.add_subparsers(dest="commande", required=True)`, puis un `add_parser` par commande.

    preparer --plan PLAN --base BASE --graine TEXTE [--part-temoin 0.2] [--contacts a_contacter.csv]
        1. `lire_plan(PLAN)` (déjà écrit, dans fichiers.py).
        2. `ouvrir(BASE)` puis `enregistrer_campagne(conn, ciblages, graine, part_temoin)`.
        3. `ecrire_contacts(conn, CONTACTS)` (déjà écrit) : seuls les clients du groupe offre.
        4. Affichez sur stdout : le nombre de clients du plan, combien sont nouveaux dans la base,
           et le nombre de clients de chaque groupe, avec le mot « offre » sur une ligne et le mot
           « témoin » sur une autre (lisez-les dans `regrouper(compter(conn))[TOUTES]`).
        `--graine` est obligatoire ; `--part-temoin` doit être strictement entre 0 et 1 (« 0,2 »
        accepté) : écrivez un `type=` personnalisé qui lève `argparse.ArgumentTypeError`.

    importer --base BASE --resultats FICHIER
        1. Si BASE n'existe pas : message sur stderr qui contient « preparer », code 1. (Sinon
           `sqlite3.connect` créerait une base vide et le programme dirait « 0 résultat » au lieu de
           signaler la faute de frappe.)
        2. `importer_resultats(conn, FICHIER)`. Chaque erreur de ligne va sur stderr ; le nombre de
           résultats enregistrés et le nombre de lignes ignorées vont sur stdout.

    bilan --base BASE [--niveau 0.95]
        1. Même contrôle de BASE que pour importer. Si la campagne est vide (0 client) : message sur
           stderr, code 1.
        2. Pour chaque `nom, (offre, temoin)` de `regrouper(compter(conn))` : calculez
           `effet(offre, temoin, niveau)` (None si `effet` lève ValueError) et affichez
           `formater_ligne(nom, offre, temoin, mesure)` (déjà écrit).

    Pour toutes les commandes : un fichier introuvable, un `ValueError` (fichier mal formé) ou une
    `sqlite3.Error` (BASE n'est pas une base SQLite) donne un message clair sur stderr et le code 1,
    jamais une traceback. Fermez vos connexions : `with conn:` NE ferme PAS la connexion (il gère la
    transaction) ; `with closing(ouvrir(...)) as conn:` la ferme.

    Ce que vous apprenez : sous-commandes argparse, assembler les modules des étapes 1 à 5, stdout
    contre stderr, codes de sortie, fermer ses ressources.
    """
    raise NotImplementedError

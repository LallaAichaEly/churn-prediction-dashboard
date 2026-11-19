"""Étape 6 🟡 — Le programme du début de mois.

Vérifier : python scripts/projet.py --projet 7 --etape 6
"""

from __future__ import annotations

import argparse  # noqa: F401 — vous en aurez besoin
import sys  # noqa: F401
from collections.abc import Iterable
from datetime import date
from pathlib import Path  # noqa: F401

from .formats import ecart_points, nombre, pourcentage  # noqa: F401
from .indicateurs import indicateurs_du_mois, par_segment  # noqa: F401
from .modeles import (DEPARTS_AFFICHES, ESPACE_MILLIERS, ESPACE_UNITE, MOIS_HISTORIQUE, Abonne,  # noqa: F401
                      Rapport, lire_abonnes)
from .page import page_html  # noqa: F401
from .periodes import decaler, fin_du_mois, lire_mois, nom_du_mois  # noqa: F401


def construire_rapport(abonnes: Iterable[Abonne], mois: date) -> Rapport:
    """Tout ce que la page doit montrer pour le mois de `mois` (n'importe quel jour de ce mois).

    - `mois` : le 1er du mois ;
    - `actuel` et `precedent` : les indicateurs du mois et du mois d'avant (étape 2) ;
    - `segments` : `par_segment` du mois ;
    - `historique` : les indicateurs des MOIS_HISTORIQUE derniers mois, du plus ancien au plus récent,
      le mois du rapport en DERNIER (un tuple) ;
    - `departs` : les abonnés partis pendant le mois (même règle qu'à l'étape 2), triés par mensualité
      DÉCROISSANTE puis, à mensualité égale, par identifiant client croissant ; les DEPARTS_AFFICHES
      premiers seulement (un tuple). Sans la règle d'égalité, deux lancements sur le même fichier
      pourraient donner deux rapports différents.

    Piège : `abonnes` peut être un générateur, qu'on ne parcourt qu'une fois. Faites-en une liste.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Le programme qu'on lance au début de chaque mois. Renvoie le code de sortie (0 = tout va bien).

        python -m projets.rapport --abonnes FICHIER.csv --mois AAAA-MM [--sortie DOSSIER] [--date-du-jour AAAA-MM-JJ]

    1. `--mois` se lit avec `lire_mois` (étape 1). `--sortie` vaut `rapports` par défaut.
       `--date-du-jour` (AAAA-MM-JJ, `date.fromisoformat`) vaut `date.today()` par défaut : il sert aux
       tests, et à refaire un vieux rapport « comme à l'époque ».
    2. Refusez un mois qui n'est pas terminé : si le dernier jour du mois n'est pas AVANT la date du
       jour, un rapport sur le mois en cours oublierait tous les départs des jours qui restent (et
       ferait croire que le churn baisse). Le 31 octobre au soir, octobre n'est pas fini.
    3. `lire_abonnes` (dans modeles.py) lit le fichier ; il lève `FileNotFoundError` ou `ValueError`.
    4. Refusez un mois sans données : si aucune date du fichier (`debut` ou `fin`) n'est dans le mois
       ou après, le fichier s'arrête avant ce mois (« 2027-10 » tapé au lieu de « 2026-10 » ?). Un
       rapport plein de zéros serait pris au sérieux.
    5. `construire_rapport` puis `page_html` ; écrivez la page dans `<sortie>/rapport_AAAA-MM.html`, en
       créant le dossier s'il le faut. Relancer la commande réécrit le même fichier.
       PRÉCISEZ l'encodage : `write_text(texte, encoding="utf-8")`. Sans lui, Python utilise celui du
       système : cp1252 sous Windows, qui ne connaît pas l'espace fine insécable → UnicodeEncodeError.
       Le correcteur relit votre code et refuse tout `open`, `write_text` ou `read_text` sans `encoding=`.
    6. Affichez un résumé écrit avec les fonctions de l'étape 3 (le mois, le nombre de clients au début, les
       départs, le taux et son écart avec le mois d'avant), puis le chemin du fichier écrit. Dans le TERMINAL,
       remplacez ESPACE_MILLIERS et ESPACE_UNITE par des espaces ordinaires : une console Windows dont la
       sortie est redirigée vers un fichier écrit en cp1252, et `print` planterait sur l'espace fine.

    Toute erreur prévue (mois illisible ou pas terminé, date du jour illisible, fichier introuvable ou
    faux, mois sans données) : UNE phrase sur stderr (`print(..., file=sys.stderr)`), code 1, aucun
    fichier écrit, jamais de traceback.

    Exemple :

        $ python -m projets.rapport --abonnes projets/exemple_abonnes.csv --mois 2026-10
        Octobre 2026 : 2 844 clients au début du mois, 89 départs
        Taux de churn : 3,0 % (+1,3 pt par rapport à septembre 2026)
        Rapport écrit dans rapports/rapport_2026-10.html

    Ce que vous apprenez : assembler les étapes, refuser de produire un chiffre faux plutôt que de le
    produire en silence, les encodages (le bug qui « marche sur ma machine »), stdout contre stderr.
    """
    raise NotImplementedError

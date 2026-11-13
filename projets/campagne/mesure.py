"""Étape 4 🟡 — Compter, en une requête SQL, ce que la base sait de chaque groupe.

Vérifier : python scripts/projet.py --projet 3 --etape 4
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

from .modeles import GROUPES, TOUTES, Comptage  # noqa: F401


def compter(conn: sqlite3.Connection) -> list[Comptage]:
    """Renvoie un `Comptage` par couple (offre, groupe) présent dans la table `campagne`, trié par
    offre puis par groupe (« offre » avant « temoin » : l'ordre alphabétique suffit).

    Pour chaque couple :
    - `clients`       : le nombre de clients de la campagne ;
    - `avec_resultat` : combien ont un résultat dans la table `resultats` ;
    - `partis`        : combien sont partis (`parti = 1`) ;
    - `cout`          : la somme de leur colonne `cout`, en float.

    Faites travailler SQLite : UNE seule requête, avec `GROUP BY`. Le correcteur compte vos
    requêtes : une requête par offre ou par client (le problème « N+1 » de la série 15) est refusée.

    Les trois pièges de cette requête (le correcteur a prévu des clients sans résultat) :
    - un `JOIN` simple fait DISPARAÎTRE les clients sans résultat : il faut un `LEFT JOIN` ;
    - `COUNT(*)` compte toutes les lignes ; `COUNT(colonne)` ne compte que les valeurs non NULL ;
    - `SUM` d'une colonne où tout est NULL vaut NULL (donc None en Python), pas 0 : `COALESCE`.

    Ce que vous apprenez : GROUP BY, LEFT JOIN, COUNT et NULL, COALESCE, construire des objets à
    partir des lignes d'une requête.
    """
    raise NotImplementedError


def regrouper(comptages: Iterable[Comptage]) -> dict[str, tuple[Comptage, Comptage]]:
    """Range les comptages par offre, sous la forme `{offre: (comptage_offre, comptage_temoin)}`.

    1. La première clé est `TOUTES` : la somme de toutes les offres, groupe par groupe (clients,
       avec_resultat, partis et cout additionnés), avec `offre=TOUTES`.
    2. Puis une clé par code d'offre, dans l'ordre alphabétique, quel que soit l'ordre reçu.
    3. Chaque valeur est un tuple `(groupe offre, groupe témoin)`, dans cet ordre (celui de `GROUPES`).
       S'il manque un groupe pour une offre (aucun témoin n'a été tiré pour une petite offre), mettez
       un `Comptage(code, groupe, 0, 0, 0, 0.0)` : l'appelant n'aura pas de cas particulier à gérer.
    4. Sans aucun comptage, renvoyez quand même `{TOUTES: (zéro offre, zéro témoin)}`.

    `comptages` peut être un générateur : on ne peut le parcourir qu'UNE fois. Transformez-le en
    liste si vous avez besoin de le parcourir plusieurs fois.

    Ce que vous apprenez : construire un dictionnaire (il garde l'ordre d'insertion), `sorted`,
    `dict.get` avec une valeur par défaut, additionner des objets champ par champ.
    """
    raise NotImplementedError

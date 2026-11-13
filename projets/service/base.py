"""Étape 2 🟡 — Ouvrir la base et répartir les contacts entre les conseillers.

Vérifier : python scripts/projet.py --projet 4 --etape 2
"""

from __future__ import annotations

import sqlite3  # noqa: F401 — vous en aurez besoin
from pathlib import Path
from typing import Sequence

from .modeles import SCHEMA  # noqa: F401


def ouvrir(chemin: str | Path) -> sqlite3.Connection:
    """Ouvre la base du service (la crée si besoin) et renvoie la connexion.

    1. `sqlite3.connect(chemin, check_same_thread=False)`. Pourquoi ce réglage ? À l'étape 5,
       FastAPI ouvrira la connexion dans un fil d'exécution (thread) et exécutera la route dans
       un autre ; sqlite3 refuse cela par défaut. C'est sans danger ici, parce que chaque requête
       HTTP aura SA connexion, jamais partagée.
    2. Activez les clés étrangères (`PRAGMA foreign_keys = ON`), qui ne valent que pour cette
       connexion, puis créez les tables : `conn.executescript(SCHEMA)`.
    3. Si l'une de ces requêtes échoue (le fichier n'est pas une base SQLite : `sqlite3.DatabaseError`),
       fermez la connexion avant de laisser remonter l'erreur.

    Ce que vous apprenez : se connecter, un réglage par connexion, créer un schéma (comme à l'étape 2
    du projet 3), et le premier piège de SQLite avec un serveur web.
    """
    raise NotImplementedError


def importer_contacts(conn: sqlite3.Connection, contacts: Sequence[tuple[str, str]],
                      conseillers: Sequence[str]) -> int:
    """Enregistre les NOUVEAUX contacts dans la table `contacts`, répartis entre les conseillers,
    et renvoie le nombre de contacts ajoutés.

    `contacts` : des couples (customer_id, offre), dans l'ordre du fichier (voir
    `fichiers.lire_contacts`). `conseillers` : les noms des conseillers qui reçoivent des contacts.

    1. Sans aucun conseiller : `ValueError`.
    2. Un contact déjà dans la base n'est pas touché (ni son conseiller, ni son statut : il a
       peut-être déjà été appelé). Un contact présent deux fois dans `contacts` : seul le premier compte.
    3. La répartition : chaque nouveau contact, dans l'ordre de `contacts`, va au conseiller qui a
       le MOINS de contacts à ce moment-là, en comptant ceux déjà dans la base (quel que soit leur
       statut) et ceux attribués juste avant dans cet appel. À égalité, le premier par ordre
       ALPHABÉTIQUE. Un conseiller qui est dans la base mais pas dans `conseillers` (parti de
       l'équipe) garde ses contacts et n'en reçoit plus.
       Astuce : un dict `charge` {conseiller: nombre}, et `min(charge, key=lambda nom: (charge[nom], nom))`.
    4. Peu de requêtes, quel que soit le nombre de contacts : lisez les identifiants déjà connus et
       la charge de chacun en une ou deux requêtes (`GROUP BY`), travaillez en mémoire, puis insérez
       tout d'un coup (`executemany`). Le correcteur importe 300 contacts et compte vos requêtes :
       une requête PAR contact (le piège « N+1 ») est refusée.
    5. Tout ou rien : si la base refuse une ligne (offre manquante : `NOT NULL`), laissez remonter
       `sqlite3.IntegrityError` et qu'AUCUN contact de cet appel ne reste dans la base (`with conn:`).
    6. Requêtes paramétrées (« ? ») uniquement : le correcteur essaie un identifiant avec une apostrophe.

    Ce que vous apprenez : répartir une charge (algorithme glouton), éviter les requêtes N+1,
    transaction, idempotence (relancer l'import ne change rien).
    """
    raise NotImplementedError

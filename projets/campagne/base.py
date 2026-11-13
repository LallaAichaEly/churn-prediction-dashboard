"""Étape 2 🟡 — Ranger la campagne dans une base SQLite, sans doublon et sans demi-mesure.

Vérifier : python scripts/projet.py --projet 3 --etape 2
"""

from __future__ import annotations

import sqlite3  # noqa: F401 — vous en aurez besoin
from pathlib import Path
from typing import Sequence

from .modeles import PART_TEMOIN, SCHEMA, Ciblage  # noqa: F401
from .tirage import groupe  # noqa: F401


def ouvrir(chemin: str | Path) -> sqlite3.Connection:
    """Ouvre la base de la campagne (la crée si besoin) et renvoie la connexion.

    1. Connectez-vous au fichier `chemin` avec `sqlite3.connect`. `chemin` peut être un texte, un
       `Path`, ou ":memory:" (une base en mémoire, pratique pour les tests).
    2. Activez les clés étrangères : `PRAGMA foreign_keys = ON`. SQLite les IGNORE par défaut, et ce
       réglage ne vaut que pour la connexion en cours : sans lui, la table `resultats` accepterait
       un client qui n'existe pas dans `campagne`.
    3. Créez les tables en exécutant `SCHEMA` (plusieurs instructions : `executescript`). Grâce à
       `IF NOT EXISTS`, rouvrir une base existante ne l'efface pas.

    Ce que vous apprenez : se connecter à une base, un réglage par connexion, créer un schéma.
    """
    raise NotImplementedError


def enregistrer_campagne(conn: sqlite3.Connection, ciblages: Sequence[Ciblage], graine: str,
                         part_temoin: float = PART_TEMOIN) -> int:
    """Enregistre chaque client du plan dans la table `campagne`, avec son groupe, et renvoie le
    nombre de clients AJOUTÉS.

    1. Le groupe de chaque client est `groupe(customer_id, graine, part_temoin)` (étape 1).
    2. Requêtes PARAMÉTRÉES uniquement : `VALUES (?, ?, ?, ?, ?)` et les valeurs à part. Jamais de
       f-string dans du SQL : le correcteur essaie un identifiant avec une apostrophe (« C'0042 »).
    3. Idempotence : relancer la même commande ne doit ni planter (la clé primaire refuse les
       doublons) ni changer le groupe d'un client déjà enregistré. Le client déjà présent est
       simplement ignoré : cherchez `ON CONFLICT (customer_id) DO NOTHING`.
    4. Tout ou rien : si une ligne est refusée par la base (un `CHECK` : coût négatif, probabilité
       de 1,4…), laissez remonter l'erreur `sqlite3.IntegrityError`, et AUCUN client de cet appel ne
       doit rester dans la base. Une transaction fait exactement ça : `with conn:` valide (COMMIT)
       si le bloc se termine bien et annule (ROLLBACK) s'il lève une exception.
    5. Le nombre renvoyé ne compte que les lignes vraiment insérées : 0 si on relance la même
       campagne. `curseur.rowcount` (après `execute` ou `executemany`) donne ce nombre.

    Ce que vous apprenez : INSERT paramétré, `executemany`, transactions, idempotence.

    Piège : sans `commit` (ou `with conn:`), tout semble marcher… jusqu'à ce qu'on ferme la
    connexion : les lignes disparaissent. Le correcteur rouvre la base pour vérifier.
    """
    raise NotImplementedError

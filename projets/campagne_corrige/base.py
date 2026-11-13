"""Étape 2 (corrigé) — Ranger la campagne dans une base SQLite, sans doublon et sans demi-mesure."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from .modeles import PART_TEMOIN, SCHEMA, Ciblage
from .tirage import groupe


def ouvrir(chemin: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(chemin)
    try:
        # SQLite n'applique PAS les clés étrangères par défaut, et ce réglage vaut pour UNE connexion :
        # il faut le redemander à chaque ouverture.
        conn.execute("PRAGMA foreign_keys = ON")
        # IF NOT EXISTS (dans SCHEMA) : rouvrir une base existante ne l'efface pas.
        conn.executescript(SCHEMA)
    except sqlite3.Error:
        # connect() réussit même sur un fichier qui n'est pas une base : l'erreur n'arrive qu'à la
        # première requête. On ferme alors la connexion avant de laisser remonter l'erreur, sinon
        # elle resterait ouverte (et le fichier verrouillé sous Windows).
        conn.close()
        raise
    return conn


def enregistrer_campagne(conn: sqlite3.Connection, ciblages: Sequence[Ciblage], graine: str,
                         part_temoin: float = PART_TEMOIN) -> int:
    lignes = [(c.customer_id, c.offre, c.cout, c.proba_churn, groupe(c.customer_id, graine, part_temoin))
              for c in ciblages]
    # `with conn:` ouvre une transaction : COMMIT si le bloc se termine normalement, ROLLBACK s'il
    # lève une exception. Une ligne refusée par un CHECK annule donc TOUTE la campagne, au lieu
    # d'en laisser la moitié dans la base. (Attention : `with conn:` ne FERME pas la connexion.)
    with conn:
        # Les « ? » : SQLite reçoit les valeurs à part, jamais collées dans le texte de la requête.
        # Un identifiant comme « C'0042 » ne casse rien, et l'injection SQL est impossible.
        # ON CONFLICT DO NOTHING : relancer la commande ne crée pas de doublon et, surtout, ne change
        # le groupe d'aucun client déjà tiré.
        curseur = conn.executemany(
            "INSERT INTO campagne (customer_id, offre, cout, proba_churn, groupe) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (customer_id) DO NOTHING",
            lignes,
        )
    # rowcount : le nombre de lignes VRAIMENT insérées (celles ignorées par DO NOTHING ne comptent pas).
    return max(curseur.rowcount, 0)

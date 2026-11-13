"""Étape 2 (corrigé) — Ouvrir la base et répartir les contacts entre les conseillers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from .modeles import SCHEMA


def ouvrir(chemin: str | Path) -> sqlite3.Connection:
    # check_same_thread=False : FastAPI peut ouvrir la connexion dans un fil d'exécution (thread)
    # et exécuter la route dans un autre. Sans ce réglage, sqlite3 refuse (ProgrammingError).
    # C'est sans danger ici parce que chaque requête HTTP a SA connexion, jamais partagée.
    conn = sqlite3.connect(chemin, check_same_thread=False)
    try:
        conn.execute("PRAGMA foreign_keys = ON")  # à redemander à CHAQUE connexion
        conn.executescript(SCHEMA)
    except sqlite3.Error:
        conn.close()  # un fichier qui n'est pas une base : on ne laisse pas la connexion ouverte
        raise
    return conn


def importer_contacts(conn: sqlite3.Connection, contacts: Sequence[tuple[str, str]],
                      conseillers: Sequence[str]) -> int:
    if not conseillers:
        raise ValueError("aucun conseiller à qui attribuer les contacts")

    # Deux requêtes en tout, quel que soit le nombre de contacts : les identifiants déjà connus et
    # la charge actuelle de chaque conseiller. Un SELECT par contact (« N+1 ») ferait 1 600 allers-
    # retours avec la liste d'exemple, et 1 600 000 avec un vrai fichier.
    deja_la = {cid for (cid,) in conn.execute("SELECT customer_id FROM contacts")}
    charge = dict.fromkeys(conseillers, 0)
    for conseiller, nombre in conn.execute("SELECT conseiller, COUNT(*) FROM contacts GROUP BY conseiller"):
        if conseiller in charge:  # un conseiller parti garde ses contacts, mais n'en reçoit plus
            charge[conseiller] = nombre

    nouveaux: list[tuple[str, str, str]] = []
    for customer_id, offre in contacts:
        if customer_id in deja_la:
            continue  # déjà attribué (ou en double dans le fichier) : on ne touche à rien
        # Le moins chargé ; à égalité, le premier par ordre alphabétique. Le tuple (charge, nom)
        # se compare élément par élément : c'est tout le départage.
        conseiller = min(charge, key=lambda nom: (charge[nom], nom))
        charge[conseiller] += 1
        deja_la.add(customer_id)
        nouveaux.append((customer_id, offre, conseiller))

    # Tout ou rien : si une ligne est refusée (offre vide…), aucun contact de cet appel n'est gardé.
    with conn:
        conn.executemany("INSERT INTO contacts (customer_id, offre, conseiller) VALUES (?, ?, ?)", nouveaux)
    return len(nouveaux)

    # Pourquoi pas `ON CONFLICT DO NOTHING` comme au projet 3 ? On a déjà écarté les contacts connus
    # (il fallait les connaître pour bien répartir la charge). Si un collègue importe le même fichier
    # au même moment, l'INSERT lèvera IntegrityError et la transaction sera annulée : il suffira de
    # relancer. Mieux vaut une erreur visible qu'une répartition fausse.

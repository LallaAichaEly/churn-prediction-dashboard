"""Étape 4 (corrigé) — Charger un fichier une fois, en entier ou pas du tout."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .chargement import charger_contrats, charger_tickets
from .contrats import lire_contrats
from .modeles import CHARGE, DEJA_CHARGE, REFUSE, SEUIL_REJETS, Bilan, empreinte
from .tickets import lire_tickets


def charger_fichier(connexion: sqlite3.Connection, chemin: str | Path) -> Bilan:
    chemin = Path(chemin)
    nom = chemin.name
    if nom.startswith("crm_") and nom.endswith(".csv"):
        lire, ranger = lire_contrats, charger_contrats
    elif nom.startswith("support_") and nom.endswith(".jsonl"):
        lire, ranger = lire_tickets, charger_tickets
    else:
        raise ValueError(f"{nom} : fichier inconnu (attendu crm_*.csv ou support_*.jsonl)")

    signature = empreinte(chemin)
    deja = connexion.execute("SELECT empreinte FROM fichiers WHERE nom = ?", (nom,)).fetchone()
    if deja is not None and deja[0] == signature:
        return Bilan(nom, DEJA_CHARGE, 0, [])  # même nom, même contenu : on ne relit même pas le fichier

    lecture = lire(chemin)
    lues = len(lecture.lignes) + len(lecture.rejets)
    if lues and len(lecture.rejets) / lues > SEUIL_REJETS:
        return Bilan(nom, REFUSE, 0, list(lecture.rejets))  # rien n'est écrit : le fichier corrigé sera chargé

    # Une seule transaction : les lignes ET la trace du fichier, ou rien du tout. Si quoi que ce soit
    # échoue au milieu, `with connexion` annule tout (rollback) et l'exception remonte : le fichier
    # n'est pas noté comme chargé, et il sera repris au prochain lancement.
    with connexion:
        changees = ranger(connexion, lecture.lignes)
        connexion.execute(
            "INSERT INTO fichiers (nom, empreinte, charge_le, lignes, rejets) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (nom) DO UPDATE SET empreinte = excluded.empreinte, charge_le = excluded.charge_le, "
            "lignes = excluded.lignes, rejets = excluded.rejets",
            (nom, signature, datetime.now().isoformat(timespec="seconds"), changees, len(lecture.rejets)),
        )
    return Bilan(nom, CHARGE, changees, list(lecture.rejets))

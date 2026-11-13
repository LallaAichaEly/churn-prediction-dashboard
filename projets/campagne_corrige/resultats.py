"""Étape 3 (corrigé) — Importer les résultats du CRM, même quand ils arrivent en plusieurs fois."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from .modeles import COLONNES_RESULTATS

_OUI = {"1", "oui", "true", "vrai"}
_NON = {"0", "non", "false", "faux"}


def _parti(texte: str | None) -> int:
    valeur = (texte or "").strip().lower()  # None : ligne coupée
    if valeur in _OUI:
        return 1
    if valeur in _NON:
        return 0
    raise ValueError(f"parti « {texte or ''} » illisible (attendu : 1/0, oui/non, true/false)")


def importer_resultats(conn: sqlite3.Connection, chemin: str | Path) -> tuple[int, list[str]]:
    chemin = Path(chemin)
    with chemin.open(encoding="utf-8-sig", newline="") as fichier:  # utf-8-sig : export Excel (BOM)
        lecteur = csv.DictReader(fichier)
        manquantes = [c for c in COLONNES_RESULTATS if c not in (lecteur.fieldnames or [])]
        if manquantes:
            raise ValueError(f"{chemin.name} : colonnes manquantes : {', '.join(manquantes)}")

        enregistres = 0
        erreurs: list[str] = []
        # Une seule transaction pour tout le fichier : si le programme plante à la ligne 5 000,
        # rien n'est gardé, et on peut relancer l'import sans se demander où il s'était arrêté.
        with conn:
            for ligne in lecteur:
                numero = lecteur.line_num
                identifiant = (ligne.get("customer_id") or "").strip()
                try:
                    if not identifiant:
                        raise ValueError("customer_id est vide")
                    parti = _parti(ligne.get("parti"))
                except ValueError as exc:
                    erreurs.append(f"ligne {numero} : {exc}")
                    continue
                try:
                    # « Upsert » : on insère, ou, si le client a déjà un résultat (import précédent,
                    # ou plus haut dans ce fichier), on remplace sa valeur. `excluded` désigne la
                    # ligne qu'on essayait d'insérer. Ainsi, réimporter un fichier ne change rien,
                    # et une correction du CRM remplace l'ancienne valeur.
                    conn.execute(
                        "INSERT INTO resultats (customer_id, parti) VALUES (?, ?) "
                        "ON CONFLICT (customer_id) DO UPDATE SET parti = excluded.parti",
                        (identifiant, parti),
                    )
                except sqlite3.IntegrityError:
                    # La clé étrangère (activée à l'étape 2) refuse un client absent de la campagne.
                    # Seule CETTE requête est annulée : la transaction continue.
                    erreurs.append(f"ligne {numero} : customer_id {identifiant} inconnu, absent de la campagne")
                    continue
                enregistres += 1
    return enregistres, erreurs

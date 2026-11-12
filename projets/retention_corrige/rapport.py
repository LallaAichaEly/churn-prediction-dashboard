"""Étape 5 (corrigé) — Résumer le plan et l'écrire dans un fichier CSV."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Sequence

from .modeles import COLONNES_PLAN, Action


def resumer(actions: Sequence[Action], nb_clients: int) -> dict:
    cout = sum(a.offre.cout for a in actions)
    gain = sum(a.gain_attendu for a in actions)
    return {
        "clients_analyses": nb_clients,
        "clients_cibles": len(actions),
        "cout_total": cout,
        "gain_attendu": gain,
        "roi": gain / cout if cout else 0.0,  # pas de ZeroDivisionError quand le plan est vide
        "par_offre": dict(Counter(a.offre.code for a in actions)),
    }


def ecrire_plan(actions: Sequence[Action], chemin: str | Path) -> Path:
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)  # crée resultats/2026/… si besoin
    with chemin.open("w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.writer(fichier)
        ecrivain.writerow(COLONNES_PLAN)
        for a in actions:
            ecrivain.writerow([
                a.client.customer_id,
                a.offre.code,
                f"{a.offre.cout:.2f}",
                f"{a.client.proba_churn:.3f}",
                f"{a.gain_attendu:.2f}",
            ])
    return chemin

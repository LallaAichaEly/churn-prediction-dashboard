"""Étape 2 (corrigé) — Lire l'export du support : une ligne JSON par ticket, et chaque ticket une seule fois."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .modeles import CHAMPS_TICKET, FORMAT_CLIENT, FORMAT_TICKET, Lecture, Rejet, Ticket


def _instant(objet: dict, cle: str) -> datetime:
    valeur = objet[cle]
    if not isinstance(valeur, str):
        raise ValueError(f"{cle} : date et heure illisibles {valeur!r}")
    try:
        return datetime.fromisoformat(valeur)
    except ValueError:
        raise ValueError(f"{cle} : date et heure illisibles {valeur!r}") from None


def _ticket(objet) -> Ticket:
    """Un objet JSON → un Ticket. Lève ValueError, avec la clé fautive, si l'objet est inutilisable."""
    if not isinstance(objet, dict):
        raise ValueError(f"un objet JSON {{...}} était attendu, pas {type(objet).__name__}")
    manquantes = [cle for cle in CHAMPS_TICKET if cle not in objet]
    if manquantes:
        raise ValueError(f"clé(s) manquante(s) : {', '.join(manquantes)}")
    # isinstance AVANT re.fullmatch : un numéro écrit 42 (sans guillemets) ferait planter fullmatch.
    for cle, motif in (("ticket", FORMAT_TICKET), ("client", FORMAT_CLIENT)):
        if not isinstance(objet[cle], str) or not re.fullmatch(motif, objet[cle]):
            raise ValueError(f"{cle} : identifiant illisible {objet[cle]!r}")
    ouvert, maj = _instant(objet, "ouvert"), _instant(objet, "maj")
    if maj < ouvert:
        raise ValueError(f"maj : {objet['maj']} est avant l'ouverture ({objet['ouvert']})")
    return Ticket(ticket=objet["ticket"], client=objet["client"], ouvert=ouvert, maj=maj)


def lire_tickets(chemin: str | Path) -> Lecture:
    chemin = Path(chemin)
    derniers: dict[str, Ticket] = {}
    rejets: list[Rejet] = []
    with chemin.open(encoding="utf-8") as f:
        for numero, texte in enumerate(f, start=1):
            if not texte.strip():
                continue
            try:
                objet = json.loads(texte.strip())
            except json.JSONDecodeError as exc:
                rejets.append(Rejet(chemin.name, numero, f"JSON illisible ({exc.msg})"))
                continue
            try:
                ticket = _ticket(objet)
            except ValueError as exc:
                rejets.append(Rejet(chemin.name, numero, str(exc)))
                continue
            deja = derniers.get(ticket.ticket)
            if deja is None or ticket.maj >= deja.maj:  # à égalité, la ligne la plus basse gagne
                derniers[ticket.ticket] = ticket
    return Lecture(lignes=[derniers[cle] for cle in sorted(derniers)], rejets=rejets)

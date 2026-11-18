"""Étape 3 (corrigé) — Ranger les lignes dans la base : la version la plus récente gagne, toujours."""

from __future__ import annotations

import sqlite3

from .modeles import Contrat, Ticket

# ON CONFLICT … DO UPDATE … WHERE : on n'écrase QUE si la ligne reçue est plus récente que celle de la
# base. Un vieux fichier rechargé par erreur ne remet donc pas un contrat dans un état ancien, et
# recharger les mêmes lignes ne change rien. Les « ? » laissent SQLite placer les valeurs : jamais
# de f-string dans une requête.
_CONTRATS = """
INSERT INTO contrats (client, debut, fin, contrat, paiement, internet, assistance, senior, produits, mensualite, maj)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (client) DO UPDATE SET
    debut = excluded.debut, fin = excluded.fin, contrat = excluded.contrat, paiement = excluded.paiement,
    internet = excluded.internet, assistance = excluded.assistance, senior = excluded.senior,
    produits = excluded.produits, mensualite = excluded.mensualite, maj = excluded.maj
WHERE excluded.maj > contrats.maj
"""

_TICKETS = """
INSERT INTO tickets (ticket, client, ouvert, maj) VALUES (?, ?, ?, ?)
ON CONFLICT (ticket) DO UPDATE SET client = excluded.client, ouvert = excluded.ouvert, maj = excluded.maj
WHERE excluded.maj > tickets.maj
"""


def _texte(instant) -> str:
    """date → "2026-06-30" ; datetime → "2026-06-30T15:04:00" (secondes toujours écrites)."""
    return instant.isoformat(timespec="seconds") if hasattr(instant, "hour") else instant.isoformat()


def charger_contrats(connexion: sqlite3.Connection, contrats: list[Contrat]) -> int:
    avant = connexion.total_changes
    connexion.executemany(_CONTRATS, [
        (c.client, _texte(c.debut), _texte(c.fin) if c.fin else None, c.contrat, c.paiement, c.internet,
         c.assistance, c.senior, c.produits, c.mensualite, _texte(c.maj))
        for c in contrats
    ])
    return connexion.total_changes - avant  # pas de commit : c'est l'appelant qui termine la transaction


def charger_tickets(connexion: sqlite3.Connection, tickets: list[Ticket]) -> int:
    avant = connexion.total_changes
    connexion.executemany(_TICKETS, [(t.ticket, t.client, _texte(t.ouvert), _texte(t.maj)) for t in tickets])
    return connexion.total_changes - avant

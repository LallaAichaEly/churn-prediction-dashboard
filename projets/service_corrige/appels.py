"""Étape 4 (corrigé) — Enregistrer le résultat d'un appel, même quand deux conseillers écrivent en même temps."""

from __future__ import annotations

import sqlite3

from .contacts import lire_contact
from .modeles import RESULTATS, Conflit, Contact


def enregistrer_appel(conn: sqlite3.Connection, conseiller: str, customer_id: str, statut: str, version: int,
                      commentaire: str = "") -> Contact:
    if statut not in RESULTATS:
        raise ValueError(f"résultat d'appel inconnu : {statut!r} (choix : {', '.join(RESULTATS)})")

    # UNE transaction pour la fiche ET le journal : les deux sont enregistrés, ou aucun.
    with conn:
        # Vérifier ET écrire dans la même instruction : personne ne peut s'intercaler au milieu d'un
        # UPDATE. Le `conseiller = ?` est DANS la condition : une fiche qui n'est pas la vôtre n'est
        # jamais modifiée, même une milliseconde.
        curseur = conn.execute(
            "UPDATE contacts SET statut = ?, commentaire = ?, tentatives = tentatives + 1, version = version + 1 "
            "WHERE customer_id = ? AND conseiller = ? AND version = ?",
            (statut, commentaire, customer_id, conseiller, version),
        )
        if curseur.rowcount == 1:
            conn.execute(
                "INSERT INTO appels (customer_id, conseiller, statut, commentaire) VALUES (?, ?, ?, ?)",
                (customer_id, conseiller, statut, commentaire),
            )
            # Si cet INSERT échoue (disque plein…), l'exception sort du `with` : ROLLBACK, et
            # l'UPDATE ci-dessus est annulé lui aussi.

    # Rien n'a été modifié ? lire_contact dit pourquoi (Introuvable, PasVotreClient). S'il ne lève
    # rien, c'est que la version a changé : un conflit. Après une écriture réussie, il renvoie la
    # fiche à jour.
    contact = lire_contact(conn, conseiller, customer_id)
    if curseur.rowcount != 1:
        raise Conflit(customer_id, contact.version)
    return contact

"""Étape 3 (corrigé) — Lire une fiche, la liste d'un conseiller page par page, l'avancement de l'équipe."""

from __future__ import annotations

import sqlite3

from .modeles import COLONNES_CONTACT, STATUTS, Contact, Introuvable, PasVotreClient


def lire_contact(conn: sqlite3.Connection, conseiller: str, customer_id: str) -> Contact:
    ligne = conn.execute(f"SELECT {COLONNES_CONTACT} FROM contacts WHERE customer_id = ?", (customer_id,)).fetchone()
    # (Le f-string ne colle ici que COLONNES_CONTACT, un texte écrit par nous ; la valeur venue de
    # l'utilisateur, elle, voyage à part, dans le « ? ».)
    if ligne is None:
        raise Introuvable(customer_id)
    contact = Contact(*ligne)
    if contact.conseiller != conseiller:
        raise PasVotreClient(f"{customer_id} est suivi par un autre conseiller")
    return contact


def page_contacts(conn: sqlite3.Connection, conseiller: str, statut: str | None = None,
                  apres: str | None = None, taille: int = 20) -> tuple[list[Contact], str | None]:
    if statut is not None and statut not in STATUTS:
        raise ValueError(f"statut inconnu : {statut!r} (choix : {', '.join(STATUTS)})")
    if isinstance(taille, bool) or not isinstance(taille, int) or taille < 1:
        raise ValueError(f"taille de page invalide : {taille!r} (entier ≥ 1 attendu)")

    # La requête se construit avec des MORCEAUX ÉCRITS PAR NOUS ; les valeurs vont dans `parametres`.
    conditions, parametres = ["conseiller = ?"], [conseiller]
    if statut is not None:
        conditions.append("statut = ?")
        parametres.append(statut)
    if apres is not None:
        # Pagination par curseur (« keyset ») : « après le dernier client vu », et non « sauter les
        # 20 premiers » (OFFSET). Si des contacts changent de statut entre deux pages, OFFSET saute
        # des clients ; le curseur, jamais.
        conditions.append("customer_id > ?")
        parametres.append(apres)
    # Une ligne de plus que demandé : si elle existe, il y a une page suivante.
    lignes = conn.execute(
        f"SELECT {COLONNES_CONTACT} FROM contacts WHERE {' AND '.join(conditions)} ORDER BY customer_id LIMIT ?",
        (*parametres, taille + 1),
    ).fetchall()
    contacts = [Contact(*ligne) for ligne in lignes[:taille]]
    suivant = contacts[-1].customer_id if len(lignes) > taille else None
    return contacts, suivant


def avancement(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    # UNE requête pour toute l'équipe : la base compte, Python met en forme.
    resultat: dict[str, dict[str, int]] = {}
    lignes = conn.execute(
        "SELECT conseiller, statut, COUNT(*) FROM contacts GROUP BY conseiller, statut ORDER BY conseiller"
    )
    for conseiller, statut, nombre in lignes:
        # setdefault : la première fois qu'on voit ce conseiller, tous ses statuts partent de 0
        # (un statut sans aucun contact n'apparaît pas dans un GROUP BY).
        resultat.setdefault(conseiller, dict.fromkeys(STATUTS, 0))[statut] = nombre
    return resultat

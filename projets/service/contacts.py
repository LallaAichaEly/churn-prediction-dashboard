"""Étape 3 🟡 — Lire une fiche, la liste d'un conseiller page par page, l'avancement de l'équipe.

Vérifier : python scripts/projet.py --projet 4 --etape 3
"""

from __future__ import annotations

import sqlite3

from .modeles import COLONNES_CONTACT, STATUTS, Contact, Introuvable, PasVotreClient  # noqa: F401


def lire_contact(conn: sqlite3.Connection, conseiller: str, customer_id: str) -> Contact:
    """Renvoie la fiche du contact `customer_id`, si elle appartient bien à `conseiller`.

    1. `SELECT {COLONNES_CONTACT} FROM contacts WHERE customer_id = ?` : les colonnes sont dans
       l'ordre des champs de `Contact`, donc `Contact(*ligne)` fabrique l'objet.
    2. Aucun contact : levez `Introuvable(customer_id)` (l'API répondra 404).
    3. Le contact d'un autre conseiller : levez `PasVotreClient(...)` (l'API répondra 403).

    Coller COLONNES_CONTACT (un texte écrit par nous) dans la requête est permis ; coller
    `customer_id` (qui viendra d'une adresse web) ne l'est jamais : « ? ».
    """
    raise NotImplementedError


def page_contacts(conn: sqlite3.Connection, conseiller: str, statut: str | None = None,
                  apres: str | None = None, taille: int = 20) -> tuple[list[Contact], str | None]:
    """Une page de la liste de `conseiller` : (contacts, suivant).

    1. Seulement les contacts de `conseiller`, triés par `customer_id` (ORDER BY : sans lui, SQLite
       rend les lignes dans l'ordre qui l'arrange).
    2. Si `statut` est donné, seulement les contacts dans ce statut.
    3. Si `apres` est donné, seulement les contacts STRICTEMENT après lui : c'est la pagination par
       curseur. Le conseiller demande « la suite après C10042 », jamais « la page 3 » : si des
       contacts changent de statut entre deux pages, un OFFSET sauterait des clients.
    4. Au plus `taille` contacts. `suivant` vaut le `customer_id` du dernier contact de la page s'il
       reste des contacts après, et `None` sinon. Comment savoir s'il en reste ? Demandez `taille + 1`
       lignes (LIMIT) : si la ligne en trop existe, il y a une suite (et on ne la renvoie pas).
    5. Un statut qui n'est pas dans `STATUTS`, ou une taille inférieure à 1 : `ValueError`.
    6. UNE requête par page. Construisez-la avec des morceaux écrits par vous (une liste de
       conditions, jointes par " AND ") et mettez les valeurs dans une liste de paramètres.

    Ce que vous apprenez : construire une requête dynamique SANS injection, pagination par curseur.
    """
    raise NotImplementedError


def avancement(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """Le nombre de contacts de chaque conseiller dans chaque statut.

        {"alice": {"a_appeler": 398, "accepte": 4, "refuse": 2, "injoignable": 1}, "bruno": {...}}

    1. UNE requête pour toute l'équipe : `GROUP BY conseiller, statut`. Une requête par conseiller
       ferait 50 allers-retours pour 50 conseillers.
    2. Chaque conseiller a TOUS les statuts de `STATUTS`, même à 0 : un statut sans aucun contact
       n'apparaît pas dans le résultat d'un GROUP BY (pensez à `dict.fromkeys(STATUTS, 0)`).
    3. Les conseillers par ordre alphabétique. Une base sans contact : `{}`.
    """
    raise NotImplementedError

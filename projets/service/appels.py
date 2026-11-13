"""Étape 4 🔴 — Enregistrer le résultat d'un appel, même quand deux écritures arrivent en même temps.

Vérifier : python scripts/projet.py --projet 4 --etape 4
"""

from __future__ import annotations

import sqlite3

from .contacts import lire_contact  # noqa: F401
from .modeles import RESULTATS, Conflit, Contact  # noqa: F401


def enregistrer_appel(conn: sqlite3.Connection, conseiller: str, customer_id: str, statut: str, version: int,
                      commentaire: str = "") -> Contact:
    """Enregistre le résultat d'un appel et renvoie la fiche À JOUR.

    `version` est la version de la fiche que le conseiller avait à l'écran. Entre le moment où il
    l'a lue et celui où il enregistre, la fiche a pu changer (il l'avait ouverte sur deux onglets,
    ou sur son téléphone) : on n'enregistre alors RIEN et on le prévient. C'est le verrou optimiste.

    1. `statut` doit être un des `RESULTATS` (« a_appeler » n'est pas un résultat d'appel) :
       sinon `ValueError`, avant de toucher à la base.
    2. Dans UNE transaction (`with conn:`) :
       a. Mettez la fiche à jour : nouveau statut et commentaire, `tentatives + 1`, `version + 1`.
          La condition fait tout le travail : `WHERE customer_id = ? AND conseiller = ? AND version = ?`.
          Vérifier la version ou le conseiller en Python, AVANT l'UPDATE, laisse une fenêtre où une
          autre écriture se glisse : le correcteur en glisse une exactement là.
       b. Si `curseur.rowcount == 1` (la fiche a bien été modifiée), ajoutez une ligne au journal :
          `INSERT INTO appels (customer_id, conseiller, statut, commentaire) VALUES (?, ?, ?, ?)`.
          Si cet INSERT échoue (le correcteur simule un disque plein), l'UPDATE doit être annulé
          lui aussi : c'est pour ça qu'ils sont dans la même transaction.
    3. Si rien n'a été modifié, dites pourquoi : `lire_contact` (étape 3) lève déjà `Introuvable`
       ou `PasVotreClient` ; s'il ne lève rien, c'est que la version a changé : levez
       `Conflit(customer_id, version_actuelle)`, avec la version ACTUELLE de la fiche, pour que le
       conseiller puisse la recharger.
    4. Renvoyez la fiche à jour (`lire_contact`).

    Ce que vous apprenez : le verrou optimiste dans un vrai programme (défi 96 de la série 15),
    une transaction qui couvre deux tables, choisir la bonne exception.
    """
    raise NotImplementedError

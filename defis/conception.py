"""Outils de la série 17, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Ils sont dans un fichier à part pour une raison précise : quand le correcteur fait tomber la
passerelle SMS en panne, votre code doit voir EXACTEMENT la classe `PasserelleIndisponible` qu'il
lève. Deux classes écrites à l'identique dans deux fichiers sont deux classes différentes pour Python.

Tout est réimporté dans `defis/serie_17.py` : vous n'avez pas besoin d'ouvrir ce fichier.
"""

from __future__ import annotations

#: Les trois contrats du projet, écrits comme dans `data/customers.csv` (colonne `contract`).
CONTRATS = ("Mensuel", "Annuel", "Biannuel")

#: Des scores de churn, pour essayer le portefeuille du défi 108.
SCORES_EXEMPLE = {"C001": 0.82, "C002": 0.15, "C003": 0.47, "C004": 0.91, "C005": 0.47}

#: Des clients, pour essayer les règles du défi 109 (des dictionnaires, comme une ligne de CSV).
CLIENTS_EXEMPLE = [
    {"client_id": "C001", "contrat": "Mensuel", "anciennete_mois": 3, "appels_support": 0, "facture_mensuelle": 89.0},
    {"client_id": "C002", "contrat": "Mensuel", "anciennete_mois": 40, "appels_support": 5, "facture_mensuelle": 45.5},
    {"client_id": "C003", "contrat": "Annuel", "anciennete_mois": 2, "appels_support": 4, "facture_mensuelle": 30.0},
    {"client_id": "C004", "contrat": "Mensuel", "anciennete_mois": 30, "appels_support": 1, "facture_mensuelle": 70.0},
    {"client_id": "C005", "contrat": "Biannuel", "anciennete_mois": 50, "appels_support": 0, "facture_mensuelle": 99.9},
]


class PasserelleIndisponible(Exception):
    """La passerelle SMS n'a pas pu ouvrir ou valider un lot (panne, refus de l'opérateur)."""


class Lot:
    """Un lot de SMS en préparation : on y ajoute des clients, rien ne part avant `valider`."""

    def __init__(self, passerelle: Passerelle, nom: str) -> None:
        self.nom = nom
        self.clients: list[str] = []
        self._passerelle = passerelle

    def ajouter(self, client_id: str) -> None:
        self._passerelle.journal.append(f"ajouter {client_id}")
        self.clients.append(client_id)

    def __repr__(self) -> str:
        return f"Lot({self.nom!r}, {len(self.clients)} clients)"


class Passerelle:
    """Une fausse passerelle SMS qui note tout ce qu'on lui demande (défi 107).

    Le vrai cycle de vie d'un envoi groupé :

        lot = passerelle.ouvrir("Fidélité")   # réserve un lot chez l'opérateur
        lot.ajouter("C001")                   # prépare (rien ne part encore)
        passerelle.valider(lot)               # les SMS partent : impossible de revenir en arrière
        passerelle.annuler(lot)               # OU : on jette le lot, aucun SMS ne part
        passerelle.fermer(lot)                # rend la réservation (TOUJOURS, une fois ouvert)

    - `journal` : la liste de tout ce qui a été demandé, dans l'ordre
      (`["ouvrir Fidélité", "ajouter C001", "valider", "fermer"]`) ;
    - `envoyes` : les clients qui ont VRAIMENT reçu un SMS (ceux des lots validés) ;
    - `lots` : les lots ouverts, dans l'ordre ;
    - `Passerelle(panne="ouvrir")` ou `Passerelle(panne="valider")` : cette étape lève
      `PasserelleIndisponible` (le correcteur s'en sert pour simuler une panne).
    """

    def __init__(self, panne: str | None = None) -> None:
        if panne not in (None, "ouvrir", "valider"):
            raise ValueError("panne doit valoir None, 'ouvrir' ou 'valider'")
        self.panne = panne
        self.journal: list[str] = []
        self.envoyes: list[str] = []
        self.lots: list[Lot] = []

    def ouvrir(self, nom: str) -> Lot:
        self.journal.append(f"ouvrir {nom}")
        if self.panne == "ouvrir":
            raise PasserelleIndisponible("la passerelle SMS ne répond pas")
        lot = Lot(self, nom)
        self.lots.append(lot)
        return lot

    def valider(self, lot: Lot) -> None:
        self.journal.append("valider")
        if self.panne == "valider":
            raise PasserelleIndisponible("l'opérateur a refusé le lot")
        self.envoyes.extend(lot.clients)

    def annuler(self, lot: Lot) -> None:
        self.journal.append("annuler")

    def fermer(self, lot: Lot) -> None:
        self.journal.append("fermer")

    def __repr__(self) -> str:
        return f"Passerelle({len(self.envoyes)} SMS envoyés)"

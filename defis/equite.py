"""Outils communs de la série 38 : les clients scorés de novembre, le modèle en production, et les résultats nommés.

Depuis la campagne de novembre, le modèle de churn décide qui reçoit l'offre de fidélité (−20 % pendant six mois) :
tout client dont le score atteint `SEUIL` la reçoit. Mme Roux, 71 ans, a écrit au service client : « Mon voisin a
reçu la remise, pas moi. Est-ce parce que je suis âgée ? » Le service juridique demande un audit chiffré avant la
campagne de décembre. Les données :

- `clients_exemple()` : les 6 000 clients scorés de novembre. Chacun est un dict : `customer_id`, `age` (une
  tranche : "18-29", "30-44", "45-64" ou "65+" ; ABSENTE pour 2 % des clients : None quand la boutique ne l'a pas
  saisie, NaN quand elle vient de l'export du CRM lu avec pandas), `zone` ("Urbaine", "Périurbaine", "Rurale"),
  `contrat` ("Mensuel", "Annuel", "Biannuel"), `paiement` ("Prélèvement", "Carte bancaire", "Virement",
  "Chèque"), `internet` ("Fibre", "ADSL", "Aucun"), `appels` (le nombre d'appels au support, de 0 à 5), `score`
  (la probabilité de départ donnée par le modèle : l'offre part si score >= SEUIL) et `parti` (1 si le client est
  parti dans les six mois qui ont suivi, 0 sinon) ;
- `modele(clients)` : le modèle en production, qui reçoit une LISTE de clients et renvoie un score par client,
  dans l'ordre (défi 234). Comme celui du projet (`senior` dans `src/churn/data.py`), il lit l'âge ;
- `SEUIL`, `AGES`, `ZONES`, `CONTRATS`, `PAIEMENTS`, `INTERNETS` et `COLONNES` : le seuil de l'offre, les
  valeurs possibles de chaque colonne, et les colonnes qu'on peut soupçonner d'être des indices de l'âge ;
- `Groupe` (défis 230 et 231), `Impact` (défi 231), `Standardisation` (défi 232), `Proxy` (défi 233), `Bascule`
  (défi 234) et `Reglage` (défi 235) : les résultats, des tuples nommés ;
- `generer_clients(n, graine)` : d'autres clients, reproductibles, pour vos propres essais ; `selection(s, n)` : un
  Groupe dont on ne connaît que les sélectionnés et l'effectif, pour essayer le défi 231 à la main.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> clients = clients_exemple()
    >>> len(clients), sum(c["score"] >= SEUIL for c in clients), sum(c["parti"] for c in clients)
    (6000, 1813, 2304)
    >>> clients[0]
    {'customer_id': 'C00001', 'age': '30-44', 'zone': 'Périurbaine', 'contrat': 'Annuel', 'paiement': 'Carte bancaire',
     'internet': 'ADSL', 'appels': 1, 'score': 0.161, 'parti': 0}
    >>> modele([clients[0], {**clients[0], "age": "65+"}])     # le même client, avec 65 ans et plus
    [0.161, 0.087]
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple

SEUIL = 0.5   # l'offre de fidélité part si score >= SEUIL

AGES = ("18-29", "30-44", "45-64", "65+")
ZONES = ("Urbaine", "Périurbaine", "Rurale")
CONTRATS = ("Mensuel", "Annuel", "Biannuel")
PAIEMENTS = ("Prélèvement", "Carte bancaire", "Virement", "Chèque")
INTERNETS = ("Fibre", "ADSL", "Aucun")
COLONNES = ("zone", "contrat", "paiement", "internet", "appels")   # tout ce que le modèle peut lire, sauf l'âge


class Groupe(NamedTuple):
    """Ce que le modèle a décidé pour un groupe de clients (défi 230). Les taux valent None quand on ne peut pas diviser."""

    effectif: int                # clients du groupe
    selectionnes: int            # ceux qui ont reçu l'offre (score >= seuil)
    positifs: int                # ceux qui sont partis (parti == 1)
    vrais_positifs: int          # sélectionnés ET partis : l'offre est allée à quelqu'un qui en avait besoin
    faux_positifs: int           # sélectionnés mais restés
    taux_selection: float        # selectionnes / effectif
    tvp: float | None            # taux de vrais positifs (le RAPPEL) : vrais_positifs / positifs
    tfp: float | None            # taux de faux positifs : faux_positifs / (effectif − positifs)
    precision: float | None      # vrais_positifs / selectionnes


class Impact(NamedTuple):
    """La règle des quatre cinquièmes (défi 231). Un tuple : (reference, ratios, signales, ignores)."""

    reference: object            # le groupe jugé au plus fort taux de sélection (None si aucun groupe n'est jugé)
    ratios: dict                 # groupe jugé → son taux / celui de la référence (None si la référence est à 0)
    signales: list               # les groupes jugés dont le ratio est STRICTEMENT inférieur à 4/5
    ignores: list                # les groupes trop petits pour être jugés


class Standardisation(NamedTuple):
    """Des taux de sélection comparables d'un groupe à l'autre (défi 232). Un tuple de quatre valeurs."""

    brut: dict                   # groupe → taux de sélection, tel quel
    standardise: dict            # groupe → taux qu'il aurait avec la répartition par strate de TOUTE la clientèle
    strates: list                # les strates communes à tous les groupes, utilisées
    exclues: list                # les strates absentes d'au moins un groupe, écartées


class Proxy(NamedTuple):
    """À quel point une colonne trahit l'attribut protégé (défi 233). Un tuple : (colonne, v, modalites)."""

    colonne: str
    v: float                     # le V de Cramér : 0 = aucun lien, 1 = l'une des deux se déduit entièrement de l'autre
    modalites: int               # le nombre de valeurs différentes de la colonne (l'absence compte pour une)


class Bascule(NamedTuple):
    """Le test de situation pour une paire de valeurs (défi 234). Un tuple : (origine, testee, testes, gagnent, perdent)."""

    origine: object              # la vraie valeur des clients testés
    testee: object               # la valeur qu'on leur donne à la place, toutes choses égales par ailleurs
    testes: int                  # clients dont la vraie valeur est `origine`
    gagnent: int                 # pas d'offre avec leur vraie valeur, l'offre avec `testee`
    perdent: int                 # l'offre avec leur vraie valeur, plus d'offre avec `testee`


class Reglage(NamedTuple):
    """Le seuil d'un groupe pour l'égalité des chances (défi 235). Un tuple : (seuil, tvp, tfp, taux_selection)."""

    seuil: float                 # le plus HAUT seuil qui donne au groupe un tvp >= la cible
    tvp: float                   # le taux de vrais positifs obtenu avec ce seuil
    tfp: float | None            # le taux de faux positifs obtenu (None si le groupe n'a aucun client resté)
    taux_selection: float        # la part du groupe qui recevrait l'offre


def selection(selectionnes: int, effectif: int) -> Groupe:
    """Un Groupe dont on ne connaît que l'effectif et les sélectionnés : de quoi essayer le défi 231 à la main.

        >>> selection(25, 30)
        Groupe(effectif=30, selectionnes=25, positifs=0, vrais_positifs=0, faux_positifs=0, taux_selection=0.8333333333333334, tvp=None, tfp=None, precision=None)
    """
    return Groupe(effectif, selectionnes, 0, 0, 0, selectionnes / effectif if effectif else 0.0, None, None, None)


# ---------------------------------------------------------------------------- le modèle en production

_COEFFICIENTS = {
    ("contrat", "Mensuel"): 1.5, ("contrat", "Biannuel"): -1.0,
    ("paiement", "Chèque"): 0.6,
    ("internet", "Fibre"): 0.4, ("internet", "Aucun"): -0.4,
    ("age", "65+"): -0.7,
}


def _logit(client: dict) -> float:
    logit = -2.1 + 0.45 * min(client.get("appels") or 0, 5)
    for (colonne, valeur), coefficient in _COEFFICIENTS.items():
        if client.get(colonne) == valeur:
            logit += coefficient
    return logit


def modele(clients: list[dict]) -> list[float]:
    """Le modèle de churn en production : la probabilité de départ de chaque client, arrondie au millième, dans l'ordre.

    Il a été entraîné sur les départs de 2024, quand les clients de 65 ans et plus partaient peu : il a appris à
    baisser leur score (−0.7 sur le logit). Depuis la fin de la facture papier, en 2025, ils partent autant que
    les autres (un peu plus, même), mais le modèle ne le sait pas. C'est un modèle en LOT : appelez-le une fois
    avec une liste, pas une fois par client (derrière, il y a une API).

        >>> modele([{"contrat": "Mensuel", "appels": 2, "age": "65+"}, {"contrat": "Mensuel", "appels": 2, "age": "30-44"}])
        [0.401, 0.574]
    """
    return [round(1 / (1 + math.exp(-_logit(client))), 3) for client in clients]


# ---------------------------------------------------------------------------- les clients


def _tirer(rng: random.Random, valeurs: tuple, poids: tuple):
    x, cumul = rng.random() * sum(poids), 0.0
    for valeur, p in zip(valeurs, poids):
        cumul += p
        if x < cumul:
            return valeur
    return valeurs[-1]


_ZONE = {"18-29": (55, 30, 15), "30-44": (45, 35, 20), "45-64": (40, 33, 27), "65+": (30, 30, 40), None: (45, 33, 22)}
_CONTRAT = {"18-29": (72, 18, 10), "30-44": (55, 25, 20), "45-64": (45, 30, 25), "65+": (28, 34, 38), None: (55, 25, 20)}
_PAIEMENT = {"18-29": (45, 45, 8, 2), "30-44": (55, 30, 10, 5), "45-64": (55, 20, 13, 12), "65+": (40, 5, 13, 42),
             None: (50, 25, 12, 13)}
_INTERNET = {"Urbaine": (75, 15, 10), "Périurbaine": (55, 35, 10), "Rurale": (30, 55, 15)}


def generer_clients(n: int = 6000, graine: int = 0) -> list[dict]:
    """`n` clients scorés, reproductibles : âge (parfois absent), zone, contrat, paiement, internet, appels, score, départ.

    Le départ est tiré selon la VRAIE probabilité, que le modèle ne connaît pas : pour les 65 ans et plus, elle
    est plus forte que leur score (le modèle les croit fidèles depuis 2024).

        >>> [sorted(c) == ["age", "appels", "contrat", "customer_id", "internet", "paiement", "parti", "score", "zone"]
        ...  for c in generer_clients(3, graine=1)]
        [True, True, True]
    """
    rng = random.Random(graine)
    clients = []
    for i in range(n):
        tirage = rng.random()
        age = _tirer(rng, AGES, (22, 30, 30, 18))
        if tirage < 0.01:
            age = None            # pas saisi en boutique
        elif tirage < 0.02:
            age = float("nan")    # vide dans l'export du CRM, lu avec pandas
        cle = age if age in AGES else None
        zone = _tirer(rng, ZONES, _ZONE[cle])
        poids_internet = _INTERNET[zone]
        if cle == "65+":
            fibre, adsl, aucun = poids_internet
            poids_internet = (fibre * 0.6, adsl, aucun + fibre * 0.4)
        client = {
            "customer_id": f"C{i + 1:05d}",
            "age": age,
            "zone": zone,
            "contrat": _tirer(rng, CONTRATS, _CONTRAT[cle]),
            "paiement": _tirer(rng, PAIEMENTS, _PAIEMENT[cle]),
            "internet": _tirer(rng, INTERNETS, poids_internet),
            "appels": _tirer(rng, (0, 1, 2, 3, 4, 5), (30, 28, 18, 12, 8, 4)),
        }
        client["score"] = modele([client])[0]
        logit_vrai = _logit(client) + (0.9 if cle == "65+" else 0.0)   # ce que le modèle ignore
        client["parti"] = 1 if rng.random() < 1 / (1 + math.exp(-logit_vrai)) else 0
        clients.append(client)
    return clients


def clients_exemple() -> list[dict]:
    """Les 6 000 clients scorés de novembre, avec leur départ (ou non) dans les six mois qui ont suivi."""
    return generer_clients(6000, graine=2611)

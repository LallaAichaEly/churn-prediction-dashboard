"""Outils communs de la série 35 : la campagne de novembre, les clients de décembre, et les résultats nommés.

Yasmine prépare la campagne de rétention de décembre. Celle de novembre (le projet 3) a gardé un groupe témoin
tiré au hasard : 20 % des clients n'ont rien reçu. La directrice propose de recommencer avec « les 300 clients
au plus fort score de churn ». Yasmine doute : les clients les plus décidés à partir sont-ils ceux qu'un appel
peut retenir ? Les données :

- `campagne_exemple()` : les 2 000 clients de la campagne de novembre. Chacun est un dict :
  `customer_id`, trois variables connues AVANT la campagne (`contrat`, `anciennete`, `appels` : le nombre
  d'appels au support des trois derniers mois, en tranches), `score` (la probabilité de départ donnée par le
  modèle du projet 1), `valeur` (ce que rapporte en un an un client qui reste, en euros), `offre` (1 s'il a
  été contacté, 0 s'il était dans le groupe témoin), `p_offre` (la probabilité qu'il avait, au tirage,
  d'être contacté : 0.8 pour tout le monde en novembre) et `parti` (1 s'il est parti dans le mois, 0 sinon) ;
- `campagne_stratifiee_exemple()` : une autre campagne, tirée AUTREMENT : 90 % d'offre pour les clients au
  score >= 0.5, 50 % pour les autres (défi 214) ;
- `prospects_exemple()` : les 1 000 clients de décembre, pas encore contactés : ni `offre`, ni `p_offre`, ni
  `parti` ;
- `VARIABLES`, les trois variables du ciblage, et `COUT_CONTACT`, le coût d'un appel avec son offre ;
- `Effet` (défi 212), `Qini` (défi 215) et `Plan` (défi 216) : les résultats, des tuples nommés ;
- `generer_campagne(n, graine, …)` : d'autres campagnes, reproductibles, pour vos propres essais ;
- `uplift_vrai(client)` et `risque_vrai(client)` : la VÉRITÉ, que personne ne connaît jamais dans une vraie
  entreprise. Ici, les données sont simulées : on peut s'en servir pour vérifier qu'une méthode retrouve ce
  qu'elle cherche. Vos fonctions, elles, ne doivent jamais les appeler.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> campagne = campagne_exemple()
    >>> len(campagne), sum(c["offre"] for c in campagne), sum(c["parti"] for c in campagne)
    (2000, 1561, 639)
    >>> campagne[0]
    {'customer_id': 'C00001', 'contrat': 'mensuel', 'anciennete': '< 1 an', 'appels': '1-2', 'score': 0.591,
     'valeur': 876, 'offre': 1, 'p_offre': 0.8, 'parti': 1}
    >>> risque_vrai(campagne[0]), uplift_vrai(campagne[0])   # l'offre fait passer son risque de 58 % à 40 %…
    (0.58, 0.18)

…et il est parti quand même (`parti` vaut 1). On ne saura jamais ce qu'il aurait fait sans l'appel : c'est
pour ça qu'on ne mesure un effet que sur des GROUPES.
"""

from __future__ import annotations

import random
from typing import NamedTuple

VARIABLES = ("contrat", "anciennete", "appels")   # les variables du ciblage, toutes connues AVANT la campagne
COUT_CONTACT = 30                                 # euros : vingt minutes de conseiller et le geste commercial

CONTRATS = ("mensuel", "annuel", "biennal")
ANCIENNETES = ("< 1 an", "1 à 3 ans", "> 3 ans")
APPELS = ("0", "1-2", "3+")


class Effet(NamedTuple):
    """L'effet observé de l'offre sur un segment (défi 212). Un tuple : (n_offre, n_temoin, taux_offre, taux_temoin, uplift)."""

    n_offre: int                  # clients du segment qui ont reçu l'offre
    n_temoin: int                 # clients du segment dans le groupe témoin
    taux_offre: float | None      # part de départs chez ceux qui ont reçu l'offre (None si n_offre = 0)
    taux_temoin: float | None     # part de départs dans le groupe témoin (None si n_temoin = 0)
    uplift: float | None          # taux_temoin − taux_offre : départs évités par client contacté (None si un groupe est vide)


class Qini(NamedTuple):
    """La courbe de Qini d'un classement (défi 215). Un tuple : (points, qini)."""

    points: list[tuple[int, float]]   # (k, départs évités estimés si l'on contacte les k premiers), de k = 0 à n
    qini: float                       # aire entre la courbe et la droite du hasard, divisée par n


class Plan(NamedTuple):
    """Un plan de contact (défi 216). Un tuple : (clients, gain, cout)."""

    clients: list[str]            # les customer_id à appeler, du plus rentable au moins rentable
    gain: float                   # le gain net attendu, en euros : somme des uplift × valeur − coût
    cout: int                     # ce que coûtent les appels, en euros


# ---------------------------------------------------------------------------- la vérité (simulée)

_RISQUE_CONTRAT = {"mensuel": 0.36, "annuel": 0.14, "biennal": 0.06}
_RISQUE_ANCIENNETE = {"< 1 an": 0.12, "1 à 3 ans": 0.04, "> 3 ans": 0.0}
_RISQUE_APPELS = {"0": 0.0, "1-2": 0.10, "3+": 0.34}


def risque_vrai(client: dict) -> float:
    """La VRAIE probabilité de départ du client s'il ne reçoit rien (connue ici seulement : données simulées)."""
    risque = (_RISQUE_CONTRAT[client["contrat"]] + _RISQUE_ANCIENNETE[client["anciennete"]]
              + _RISQUE_APPELS[client["appels"]])
    return round(min(risque, 0.95), 2)


def uplift_vrai(client: dict) -> float:
    """De combien l'offre fait VRAIMENT baisser sa probabilité de départ (connue ici seulement : données simulées).

    Les quatre profils du marketing de rétention :

    - les PERSUADABLES (contrat mensuel, pas encore excédés) : l'appel les retient vraiment ;
    - les CAUSES PERDUES (3 appels au support ou plus) : leur décision est prise, l'appel n'y change rien ;
    - les ACQUIS (annuels anciens, sans problème) : ils restent de toute façon, l'appel ne sert à rien ;
    - les CHIENS QUI DORMENT (biennaux) : l'appel leur rappelle qu'ils peuvent partir, et certains partent.
    """
    contrat, anciennete, appels = client["contrat"], client["anciennete"], client["appels"]
    if appels == "3+":
        return 0.0
    if contrat == "mensuel":
        return 0.18 if anciennete != "> 3 ans" else 0.10
    if contrat == "annuel":
        return 0.0 if anciennete == "> 3 ans" else 0.06
    return -0.05


# ---------------------------------------------------------------------------- les campagnes


def _tirer(rng: random.Random, valeurs: tuple, poids: tuple):
    x, cumul = rng.random(), 0.0
    for valeur, p in zip(valeurs, poids):
        cumul += p
        if x < cumul:
            return valeur
    return valeurs[-1]


def generer_campagne(n: int = 2000, graine: int = 0, p_offre: float = 0.8, stratifiee: bool = False,
                     prefixe: str = "C") -> list[dict]:
    """`n` clients d'une campagne passée, reproductibles : variables, score, valeur, offre tirée au hasard, départ.

    - `p_offre` : la probabilité, pour chaque client, d'être contacté (le reste forme le groupe témoin) ;
    - `stratifiee=True` : on tire AUTREMENT selon le score, 0.9 pour un score >= 0.5 et 0.5 sinon (`p_offre`
      est alors ignoré). C'est permis, à condition de noter pour chacun la probabilité qu'il avait (`p_offre`).

        >>> clients = generer_campagne(5, graine=1)
        >>> [c["customer_id"] for c in clients], sorted(clients[0])
        (['C00001', 'C00002', 'C00003', 'C00004', 'C00005'],
         ['anciennete', 'appels', 'contrat', 'customer_id', 'offre', 'p_offre', 'parti', 'score', 'valeur'])
        >>> {c["p_offre"] for c in generer_campagne(200, graine=2, stratifiee=True)}
        {0.5, 0.9}
    """
    rng = random.Random(graine)
    clients = []
    for i in range(n):
        client = {
            "customer_id": f"{prefixe}{i + 1:05d}",
            "contrat": _tirer(rng, CONTRATS, (0.5, 0.3, 0.2)),
            "anciennete": _tirer(rng, ANCIENNETES, (0.3, 0.4, 0.3)),
            "appels": _tirer(rng, APPELS, (0.5, 0.33, 0.17)),
        }
        risque = risque_vrai(client)
        client["score"] = round(min(max(risque + rng.uniform(-0.06, 0.06), 0.01), 0.99), 3)
        facture = {"mensuel": 70, "annuel": 60, "biennal": 55}[client["contrat"]] + rng.randint(-15, 15)
        client["valeur"] = 12 * facture
        p = (0.9 if client["score"] >= 0.5 else 0.5) if stratifiee else p_offre
        client["offre"] = 1 if rng.random() < p else 0
        client["p_offre"] = p
        client["parti"] = 1 if rng.random() < risque - client["offre"] * uplift_vrai(client) else 0
        clients.append(client)
    return clients


def campagne_exemple() -> list[dict]:
    """Les 2 000 clients de la campagne de novembre : 80 % ont reçu l'offre, 20 % forment le groupe témoin."""
    return generer_campagne(2000, graine=2611)


def campagne_stratifiee_exemple() -> list[dict]:
    """2 000 clients d'une campagne tirée selon le score : 90 % d'offre si score >= 0.5, 50 % sinon (défi 214)."""
    return generer_campagne(2000, graine=2612, stratifiee=True)


def prospects_exemple() -> list[dict]:
    """Les 1 000 clients de décembre, pas encore contactés : ni `offre`, ni `p_offre`, ni `parti`."""
    return [{cle: valeur for cle, valeur in client.items() if cle not in ("offre", "p_offre", "parti")}
            for client in generer_campagne(1000, graine=2612 + 1, prefixe="D")]

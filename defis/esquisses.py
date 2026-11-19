"""Outils communs de la série 40 : l'empreinte, les numéros contactés, les appels, les visites et les clients.

Le 12 novembre, à 19 h, la campagne de rétention s'est arrêtée net : le service d'envoi des SMS gardait dans un
`set` les 40 millions de numéros contactés depuis le début du mois (pour ne jamais écrire deux fois au même
client), et ce `set` avait fini par dépasser les 3 Go de mémoire de son serveur. Le même soir, le tableau de bord
annonçait 11 242 « visiteurs uniques » de la page de résiliation pour la semaine, en additionnant les visiteurs de
chaque jour (un client venu lundi et jeudi comptait deux fois ; ils n'étaient que 7 981) ; et le service de
score, passé de 4 à 5 serveurs avec `hash(client) % 5`, a vu 80 % de ses clients changer de serveur, et tous ses
caches repartir de zéro.
Inès, développeuse de la plateforme d'envoi, reprend tout avec des structures qui tiennent dans quelques
kilo-octets ou méga-octets, quitte à répondre « à peu près », mais avec une marge d'erreur CONNUE. Les données :

- `empreinte(texte, graine)` : un entier de 64 bits tiré de SHA-256, le même sur toutes les machines et à chaque
  lancement (pas `hash()`). Une autre graine donne une autre empreinte, indépendante : c'est ce qui fournit
  « plusieurs fonctions de hachage » aux défis 243, 245 et 246 ;
- `Dimension` (défi 242) : la taille d'un filtre de Bloom (bits, nombre de fonctions k, taux de faux positifs) ;
- `envois_exemple()` : les numéros contactés ce mois-ci par chacun des trois serveurs d'envoi (défi 243) ;
- `campagne_exemple()` : les numéros visés par la campagne de demain, dont une partie a déjà reçu un SMS ;
- `appels_exemple()` : les numéros des 60 000 appels reçus par le support ce mois-ci, dans l'ordre d'arrivée
  (défis 244 et 245) ;
- `visites_exemple()` : les visiteurs de la page de résiliation, jour par jour, sur une semaine (défi 246) ;
- `clients_exemple()` et `SERVEURS` : les clients et les serveurs du service de score (défi 247) ;
- `generer_numeros(n, graine)`, `generer_appels(n, distincts, graine)` et `generer_visiteurs(n, graine)` :
  d'autres données, reproductibles, pour vos propres essais.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> empreinte("0612345678"), empreinte("0612345678", 1)
    (8160095906645286627, 15777807589842033787)
    >>> envois = envois_exemple()
    >>> sorted(envois), sum(len(v) for v in envois.values())
    (['sms-1', 'sms-2', 'sms-3'], 12000)
    >>> appels = appels_exemple()
    >>> len(appels), len(set(appels))
    (60000, 15611)
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import NamedTuple

SERVEURS = ("score-1", "score-2", "score-3", "score-4")   # les serveurs du service de score (défi 247)

JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def empreinte(texte: str, graine: int = 0) -> int:
    """Un entier de 64 bits (0 à 2**64 - 1) : les 8 premiers octets de SHA-256(f"{graine}:{texte}").

    Le même résultat sur toutes les machines et à chaque lancement de Python, contrairement à `hash()`, qui est
    « salé » au démarrage (série 39, défi 240). Deux graines différentes donnent deux empreintes sans rapport :
    c'est une FAMILLE de fonctions de hachage, numérotées par la graine.

        >>> empreinte("C000001"), empreinte("C000001", graine=7)
        (5554835790419175625, 9385086558381087929)
        >>> int.from_bytes(hashlib.sha256("0:C000001".encode("utf-8")).digest()[:8], "big")
        5554835790419175625
    """
    if not isinstance(texte, str):
        raise TypeError(f"empreinte() attend un texte, pas {type(texte).__name__} : {texte!r}")
    if isinstance(graine, bool) or not isinstance(graine, int):
        raise TypeError(f"la graine est un entier, pas {type(graine).__name__}")
    return int.from_bytes(hashlib.sha256(f"{graine}:{texte}".encode("utf-8")).digest()[:8], "big")


class Dimension(NamedTuple):
    """La taille d'un filtre de Bloom (défi 242). Un tuple : (bits, k, taux_reel)."""

    bits: int          # m : le nombre de bits du filtre
    k: int             # le nombre de positions (de fonctions de hachage) par clé
    taux_reel: float   # le taux de faux positifs attendu avec CES bits et CE k, une fois les n clés ajoutées

    @property
    def octets(self) -> int:
        """La mémoire occupée par les bits, en octets (8 bits par octet, le dernier octet entamé compte)."""
        return (self.bits + 7) // 8


# ---------------------------------------------------------------------------- numéros et envois (242, 243)


def generer_numeros(n: int = 1000, graine: int = 0) -> list[str]:
    """`n` numéros de portable français DIFFÉRENTS (« 06 » ou « 07 » suivi de 8 chiffres), reproductibles.

        >>> generer_numeros(3, graine=1)
        ['0676397250', '0634234785', '0666496171']
    """
    rng = random.Random(graine)
    vus: set[str] = set()
    numeros = []
    while len(numeros) < n:
        numero = f"0{rng.choice((6, 7))}{rng.randrange(10 ** 8):08d}"
        if numero not in vus:
            vus.add(numero)
            numeros.append(numero)
    return numeros


def _tous_les_numeros() -> list[str]:
    return generer_numeros(15_000, graine=40)


def envois_exemple() -> dict[str, list[str]]:
    """{serveur: numéros} : les 12 000 numéros contactés ce mois-ci, répartis entre les trois serveurs d'envoi.

    Chaque serveur ne connaît que SES numéros : pour savoir si un client a déjà reçu un SMS, il faut réunir les
    trois (défi 243, `union`).

        >>> {serveur: len(numeros) for serveur, numeros in envois_exemple().items()}
        {'sms-1': 5000, 'sms-2': 4000, 'sms-3': 3000}
    """
    numeros = _tous_les_numeros()[:12_000]
    return {"sms-1": numeros[:5000], "sms-2": numeros[5000:9000], "sms-3": numeros[9000:]}


def campagne_exemple() -> list[str]:
    """Les 4 000 numéros visés par la campagne de demain : 1 000 ont déjà reçu un SMS ce mois-ci, 3 000 jamais.

        >>> campagne = campagne_exemple()
        >>> len(campagne), len(set(campagne)), campagne[:2]
        (4000, 4000, ['0755572495', '0608914958'])
    """
    tous = _tous_les_numeros()
    deja = tous[:12_000:12]                         # 1 000 numéros déjà contactés, pris chez les trois serveurs
    jamais = tous[12_000:15_000]
    campagne = deja + jamais
    random.Random(4000).shuffle(campagne)
    return campagne


# ---------------------------------------------------------------------------- appels au support (244, 245)


def generer_appels(n: int = 10_000, distincts: int = 2_000, graine: int = 0) -> list[str]:
    """`n` appels (des numéros, dans l'ordre d'arrivée) passés par environ `distincts` numéros : très peu de
    numéros appellent très souvent, la plupart une ou deux fois (une loi de Zipf, comme dans la vraie vie).

        >>> generer_appels(6, 3, graine=1)
        ['0676397250', '0666496171', '0634234785', '0676397250', '0676397250', '0676397250']
    """
    rng = random.Random(graine)
    numeros = generer_numeros(distincts, graine=graine)
    poids = [1 / (rang + 1) for rang in range(distincts)]
    return rng.choices(numeros, weights=poids, k=n)


def appels_exemple() -> list[str]:
    """Les numéros des 60 000 appels reçus par le support ce mois-ci, dans l'ordre d'arrivée.

    Quatre numéros appellent beaucoup plus que les autres : un robot d'appels automatiques (2 400 appels), un
    client en litige (1 500), une boutique partenaire (900) et un client qui rappelle à chaque coupure (700). Les
    autres viennent d'une longue traîne de plus de 15 000 numéros.

        >>> from collections import Counter
        >>> Counter(appels_exemple()).most_common(5)
        [('0700000000', 2400), ('0611111111', 1500), ('0622222222', 900), ('0633333333', 700), ('0713554669', 163)]
    """
    rng = random.Random(245)
    gros = ["0700000000"] * 2400 + ["0611111111"] * 1500 + ["0622222222"] * 900 + ["0633333333"] * 700
    numeros = generer_numeros(30_000, graine=244)
    traine = rng.choices(numeros, weights=[1 / (rang + 60) for rang in range(len(numeros))], k=60_000 - len(gros))
    appels = gros + traine
    rng.shuffle(appels)
    return appels


# ---------------------------------------------------------------------------- visites (246)


def generer_visiteurs(n: int = 1000, graine: int = 0) -> list[str]:
    """`n` identifiants de visiteurs (« V » suivi de 6 chiffres), reproductibles, avec des visites répétées.

        >>> generer_visiteurs(4, graine=1)
        ['V017611', 'V074606', 'V008271', 'V033432']
    """
    rng = random.Random(graine)
    return [f"V{rng.randrange(100_000):06d}" for _ in range(n)]


def visites_exemple() -> dict[str, list[str]]:
    """{jour: identifiants des visiteurs} : chaque visite de la page de résiliation, du lundi au dimanche.

    Beaucoup reviennent (on hésite avant de résilier) : le même jour, et d'un jour à l'autre. Additionner les
    visiteurs uniques de chaque jour compte deux fois un client venu lundi et jeudi.

        >>> visites = visites_exemple()
        >>> list(visites)[:3], len(visites["lundi"]), len(set(visites["lundi"]))
        (['lundi', 'mardi', 'mercredi'], 2600, 1986)
        >>> sum(len(set(v)) for v in visites.values()), len(set().union(*visites.values()))
        (11242, 7981)
    """
    rng = random.Random(246)
    habitues = [f"V{i:06d}" for i in range(1, 3_001)]           # ils reviennent toute la semaine
    visites = {}
    nouveau = 3_001
    for numero, jour in enumerate(JOURS):
        du_jour = []
        volume = (2600, 2500, 2400, 2300, 2700, 1200, 900)[numero]
        for _ in range(volume):
            if rng.random() < 0.5:
                du_jour.append(rng.choice(habitues))
            elif du_jour and rng.random() < 0.3:
                du_jour.append(rng.choice(du_jour))               # revenu le même jour
            else:
                du_jour.append(f"V{nouveau:06d}")
                nouveau += 1
        rng.shuffle(du_jour)
        visites[jour] = du_jour
    return visites


# ---------------------------------------------------------------------------- clients du service de score (247)


def clients_exemple(n: int = 20_000) -> list[str]:
    """Les identifiants des `n` clients du service de score : « C » suivi de 6 chiffres.

        >>> clients_exemple()[:3], len(clients_exemple())
        (['C000001', 'C000002', 'C000003'], 20000)
    """
    return [f"C{i:06d}" for i in range(1, n + 1)]


def taux_theorique(bits: int, k: int, n: int) -> float:
    """Le taux de faux positifs attendu d'un filtre de `bits` bits, à `k` positions, après `n` clés.

    La probabilité qu'un bit donné soit encore à 0 est (1 - 1/bits)**(k·n), presque exactement e**(-k·n/bits) ;
    un faux positif demande que les k positions d'une clé jamais vue soient toutes à 1.

        >>> round(taux_theorique(9586, 7, 1000), 5)
        0.01003
    """
    return (1 - math.exp(-k * n / bits)) ** k

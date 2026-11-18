"""Outils communs de la série 28 : des flux d'exemple, et de quoi vérifier qu'un code « tient le flux ».

Un flux, c'est une suite de valeurs qu'on lit UNE fois, dans l'ordre où elles arrivent, sans connaître sa
longueur : les connexions à l'application, les appels au support, les factures de la nuit. On ne le
charge pas en mémoire : on garde un petit RÉSUMÉ, mis à jour à chaque valeur.

- `UneSeuleFois(iterable)` : un flux qui ne se lit qu'une fois, et qui le DIT si on essaie de le relire (un
  générateur, lui, se tait : sa deuxième lecture est simplement vide). Il compte les valeurs lues (`.lus`) ;
- `pic_memoire(appel)` : ce que renvoie `appel()`, et la mémoire maximale (en octets) que Python a allouée
  PENDANT l'appel. C'est ainsi que le correcteur voit qu'un code garde tout le flux ;
- `connexions_exemple()`, `taux_quotidiens_exemple()`, `journal_exemple()`, `INSTANT_EXEMPLE` : les petits
  exemples des défis ;
- `generer_montants(n, graine)` et `generer_journal(n, n_clients, graine)` : de longs flux reproductibles,
  produits au fur et à mesure (des générateurs : rien n'est gardé en mémoire).

    >>> flux = UneSeuleFois([3, 1, 2])
    >>> sum(flux), flux.lus
    (6, 3)
    >>> sum(flux)
    Traceback (most recent call last):
    ...
    RuntimeError: ce flux a déjà été lu : un flux ne se relit pas. Gardez ce qu'il vous faut PENDANT la première lecture.
    >>> generateur = (x for x in [3, 1, 2])
    >>> sum(generateur), sum(generateur)              # un générateur relu est VIDE, sans erreur
    (6, 0)
    >>> resultat, pic = pic_memoire(lambda: sum(generer_montants(100_000)))
    >>> pic < 50_000                                  # 100 000 montants additionnés sans être gardés
    True
    >>> resultat, pic = pic_memoire(lambda: sum(list(generer_montants(100_000))))
    >>> pic > 2_000_000                               # la même somme, après les avoir tous gardés
    True
"""

from __future__ import annotations

import gc
import math
import random
import tracemalloc
from datetime import datetime, timedelta
from typing import Callable, Iterable, Iterator


class UneSeuleFois:
    """Un flux lisible une seule fois. Une deuxième lecture lève RuntimeError au lieu d'être vide en silence.

    Il n'a pas de longueur (`len(flux)` lève TypeError) : on ne sait pas combien de valeurs vont arriver.
    """

    def __init__(self, iterable: Iterable) -> None:
        self._source = iterable
        self._deja_lu = False
        self.lus = 0

    def __iter__(self) -> Iterator:
        if self._deja_lu:
            raise RuntimeError("ce flux a déjà été lu : un flux ne se relit pas. Gardez ce qu'il vous faut PENDANT "
                               "la première lecture.")
        self._deja_lu = True
        return self._lire()

    def _lire(self) -> Iterator:
        for valeur in self._source:
            self.lus += 1
            yield valeur

    def __repr__(self) -> str:
        return f"UneSeuleFois(déjà lu : {self._deja_lu}, {self.lus} valeurs lues)"


def pic_memoire(appel: Callable[[], object]) -> tuple[object, int]:
    """(résultat, pic) : ce que renvoie `appel()`, et le pic de mémoire allouée PENDANT l'appel, en octets.

    `tracemalloc` (bibliothèque standard) note chaque allocation faite par Python : le pic ne dépend pas de la
    vitesse de la machine. Ce qui existait avant l'appel n'est pas compté.
    """
    deja = tracemalloc.is_tracing()
    if not deja:
        tracemalloc.start()
    try:
        gc.collect()
        depart = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        resultat = appel()
        pic = tracemalloc.get_traced_memory()[1] - depart
    finally:
        if not deja:
            tracemalloc.stop()
    return resultat, max(pic, 0)


# ---------------------------------------------------------------------------- les petits exemples


def connexions_exemple() -> list[float]:
    """Les instants de 4 connexions d'une même rafale, en secondes depuis le 1er janvier 1970 (`datetime.timestamp()`).

    Moyenne 1 700 000 010, variance 30 : des nombres immenses qui diffèrent à peine. C'est le cas où la formule
    « moyenne des carrés moins carré de la moyenne » perd TOUS ses chiffres (défi 170).
    """
    return [1_700_000_004.0, 1_700_000_007.0, 1_700_000_013.0, 1_700_000_016.0]


def taux_quotidiens_exemple() -> list:
    """40 jours de taux de résiliation quotidien (en %), tels qu'ils arrivent chaque matin (défi 171).

    - jours 0 à 24 : la normale, autour de 2 % (1,75 à 2,25) ;
    - jour 8 : un pic isolé à 3,5 % (un lundi de fin de promotion) : il ne doit PAS déclencher l'alerte ;
    - jour 13 : None (l'export n'est pas arrivé), jour 29 : NaN (une division par zéro dans l'export) ;
    - à partir du jour 25 : une hausse DURABLE à 2,75 % (la concurrence a baissé ses prix) : elle doit la déclencher.

    Les valeurs sont des multiples de 0,25 : exactes en binaire, sans erreur d'arrondi qui brouillerait les
    comparaisons au seuil.
    """
    calme = [2.0, 2.25, 1.75, 2.0, 2.0, 2.25, 1.75, 2.0, 3.5, 2.0, 1.75, 2.0, 2.25, None, 2.0, 1.75, 2.0, 2.25, 2.0,
             1.75, 2.0, 2.0, 2.25, 1.75, 2.0]
    hausse = [2.75] * 15
    hausse[29 - 25] = math.nan
    return calme + hausse


INSTANT_EXEMPLE = datetime(2024, 3, 1, 8, 0)
"""Le 1er mars 2024 à 8 h : le moment où le modèle calcule les scores d'activité de `journal_exemple()`."""


def journal_exemple() -> list[tuple[str, datetime, float]]:
    """Le journal d'activité de 3 clients : (client, moment, poids), dans l'ordre d'ARRIVÉE (défi 172).

    - C1 se connecte toutes les semaines (poids 1) et a payé une facture (poids 3) ;
    - C2 était très actif… jusqu'à fin novembre. Depuis : rien (le signal de départ qu'on veut voir) ;
    - C3 : une connexion en boutique, le 28 février à 23 h, arrivée EN RETARD (après des événements plus
      récents de C1), et une autre le 29 février à 7 h. Entre les deux : 8 heures, pas « 0 jour ».
    """
    j = []
    for semaine in range(8):
        j.append(("C1", datetime(2024, 1, 5, 18, 30) + timedelta(weeks=semaine), 1.0))
    j.append(("C1", datetime(2024, 2, 15, 9, 0), 3.0))
    for jour in range(0, 60, 3):
        j.append(("C2", datetime(2023, 10, 1, 12, 0) + timedelta(days=jour), 1.0))
    j.append(("C3", datetime(2024, 2, 29, 7, 0), 1.0))
    j.append(("C1", datetime(2024, 2, 27, 20, 15), 1.0))
    j.append(("C3", datetime(2024, 2, 28, 23, 0), 1.0))  # arrivée en retard : plus ancien que l'événement précédent de C3
    j.append(("C1", datetime(2024, 2, 20, 10, 0), 1.0))  # en retard aussi
    return j


# ---------------------------------------------------------------------------- les longs flux


def generer_montants(n: int, graine: int = 0) -> Iterator[float]:
    """n montants de factures (en euros, au centime), produits un par un : un générateur, rien n'est gardé.

    Une loi log-normale (beaucoup de petites factures, quelques très grosses), comme de vraies factures.
    """
    rng = random.Random(graine)
    for _ in range(n):
        yield round(math.exp(rng.gauss(3.5, 0.6)), 2)


def generer_journal(n: int, n_clients: int = 200, graine: int = 0,
                    debut: datetime = datetime(2024, 1, 1)) -> Iterator[tuple[str, datetime, float]]:
    """n événements (client, moment, poids) de `n_clients` clients, à peu près dans l'ordre du temps.

    Environ 5 % arrivent en retard (jusqu'à 3 jours) : une boutique qui envoie ses données le lendemain. Les
    moments ont des heures et des minutes. Un générateur : rien n'est gardé en mémoire.
    """
    rng = random.Random(graine)
    moment = debut
    for _ in range(n):
        moment += timedelta(minutes=rng.randrange(1, 15))
        client = f"C{rng.randrange(n_clients):04d}"
        retard = timedelta(minutes=rng.randrange(1, 3 * 24 * 60)) if rng.random() < 0.05 else timedelta(0)
        yield client, moment - retard, rng.choice((1.0, 1.0, 1.0, 2.0, 5.0))

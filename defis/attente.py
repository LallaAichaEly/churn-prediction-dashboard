"""Outils communs de la série 44 : les appels du service client, la prévision de la journée et le planning.

Depuis septembre, le service client « tient ses chiffres » : il reçoit en moyenne 111 appels par heure, d'une
durée moyenne de 6 minutes, soit 11,1 heures de conversation par heure ; avec 14 conseillers au téléphone, le
tableur affiche 79 % d'occupation, et la direction conclut qu'on est large. Pourtant, l'après-midi, les clients
raccrochent avant qu'on leur réponde, et ceux qui raccrochent partent. Amina, responsable du service client, a
l'export du standard téléphonique de la semaine dernière et la prévision d'appels de l'équipe planning. Ce module
lui fournit :

- `appels_exemple()` : les appels de la semaine dernière (du lundi au vendredi, de 8 h à 20 h), tels que les
  exporte le standard : une liste de tuples (arrivée, durée, patience), en SECONDES, dans l'ordre des lignes
  téléphoniques (PAS dans l'ordre chronologique). `arrivee` compte les secondes depuis le lundi 8 h 00 (le mardi
  8 h 00 est à 86 400 s) ; `duree` est le temps de conversation qu'il faudrait au conseiller ; `patience` est le
  temps au bout duquel le client raccroche s'il attend encore (`math.inf` : il ne raccroche jamais) ;
- `departs_exemple()` : pour chaque appel, dans le même ordre, True si le client a résilié dans les deux mois ;
- `prevision_exemple()` : la prévision de l'équipe planning, une liste de 24 couples (« HH:MM », appels par
  heure), un par demi-heure de 8 h 00 à 19 h 30 ;
- `planning_actuel_exemple()` : le planning actuel, une liste de 24 entiers : le nombre de vacations de 4 heures
  (`VACATION` = 8 demi-heures) qui COMMENCENT à chaque demi-heure : 14 à 8 h, 14 à 12 h, 14 à 16 h ;
- `creneau(arrivee)` : le numéro de la demi-heure (0 pour 8 h 00, 23 pour 19 h 30) où tombe un appel ;
- `generer_appels(...)` : d'autres appels, reproductibles, pour vos propres essais ;
- les constantes `DUREE_MOYENNE_S` (360 s), `SEUIL_S` (20 s), `OBJECTIF` (0,8 : « 80 % des appels décrochés en 20
  secondes »), `CONSEILLERS_ACTUELS` (14) et `VACATION` (8 demi-heures).

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> appels = appels_exemple()
    >>> len(appels), appels[:3]
    (6806, [(10482, 37, 150), (31781, 1489, 230), (363476, 587, 277)])
    >>> sum(departs_exemple()), prevision_exemple()[14]
    (628, ('15:00', 180))
    >>> planning_actuel_exemple()[:9], creneau(10482), creneau(363476)
    ([14, 0, 0, 0, 0, 0, 0, 0, 14], 5, 9)
"""

from __future__ import annotations

import functools
import heapq
import math
import random

DUREE_MOYENNE_S = 360              # la durée moyenne d'une conversation : 6 minutes
SEUIL_S = 20                       # « décroché en 20 secondes »
OBJECTIF = 0.8                     # 80 % des appels décrochés en 20 secondes : la règle « 80/20 »
CONSEILLERS_ACTUELS = 14           # au téléphone en même temps, toute la journée, depuis septembre
VACATION = 8                       # une vacation dure 8 demi-heures : 4 heures
JOUR_S = 86_400                    # le mardi 8 h 00 est à 86 400 s du lundi 8 h 00
OUVERTURE_S = 12 * 3600            # de 8 h 00 à 20 h 00
PATIENCE_MOYENNE_S = 150

# La prévision de l'équipe planning : appels par heure, pour chaque demi-heure de 8 h 00 à 19 h 30.
_PREVISION = (60, 90, 130, 150, 150, 140, 120, 100, 70, 60, 70, 110,
              160, 175, 180, 170, 150, 130, 110, 100, 90, 70, 50, 30)


def _heure(k: int) -> str:
    minutes = 8 * 60 + 30 * k
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def creneau(arrivee) -> int:
    """Le numéro de la demi-heure (0 pour 8 h 00 … 23 pour 19 h 30) où tombe un appel arrivé à `arrivee` secondes
    du lundi 8 h 00, quel que soit le jour.

        >>> creneau(0), creneau(1799), creneau(1800), creneau(86_400 + 7 * 3600)      # lundi 8 h, mardi 15 h
        (0, 0, 1, 14)
    """
    return int(arrivee % JOUR_S // 1800)


def generer_appels(par_heure, jours: int = 1, graine: int = 0, duree_moyenne_s: float = DUREE_MOYENNE_S,
                   patience_moyenne_s: float = PATIENCE_MOYENNE_S, part_patients: float = 0.03,
                   melanger: bool = True) -> list[tuple]:
    """Des appels (arrivée, durée, patience) en secondes, reproductibles.

    `par_heure` est un nombre d'appels par heure (constant de 8 h à 20 h) ou une liste de 24 nombres (un par
    demi-heure). Les arrivées suivent un processus de Poisson (des appels indépendants), les durées et les patiences
    une loi exponentielle, arrondies à la seconde (au moins 1 s de conversation) ; une part `part_patients` des
    clients ne raccroche jamais (patience `math.inf`). Avec `melanger=True`, les appels sont rendus dans le désordre,
    comme dans l'export du standard.

        >>> generer_appels(60, graine=1)[:2]
        [(38253, 351, 166), (41167, 121, 184)]
    """
    taux = list(par_heure) if isinstance(par_heure, (list, tuple)) else [par_heure] * 24
    alea = random.Random(graine)
    appels = []
    for jour in range(jours):
        for k, par_h in enumerate(taux):
            if par_h <= 0:
                continue
            t = 0.0
            while True:
                t += alea.expovariate(par_h / 3600)
                if t >= 1800:
                    break
                arrivee = jour * JOUR_S + k * 1800 + int(t)
                duree = max(1, round(alea.expovariate(1 / duree_moyenne_s)))
                patience = math.inf if alea.random() < part_patients else round(alea.expovariate(1 / patience_moyenne_s))
                appels.append((arrivee, duree, patience))
    if melanger:
        alea.shuffle(appels)
    return appels


def _simuler(appels, conseillers: int) -> list[tuple[float, bool]]:
    """La file d'attente (premier arrivé, premier servi), pour fabriquer les données de l'exemple."""
    ordre = sorted(range(len(appels)), key=lambda i: appels[i][0])
    libres = [0.0] * conseillers
    resultats: list = [None] * len(appels)
    for i in ordre:
        arrivee, duree, patience = appels[i]
        debut = max(arrivee, libres[0])
        if debut - arrivee > patience:
            resultats[i] = (float(patience), True)
        else:
            resultats[i] = (float(debut - arrivee), False)
            heapq.heapreplace(libres, debut + duree)
    return resultats


@functools.lru_cache(maxsize=1)
def _semaine() -> dict:
    appels = generer_appels(_PREVISION, jours=5, graine=4412)
    resultats = _simuler(appels, CONSEILLERS_ACTUELS)
    alea = random.Random(4413)
    departs = []
    for attente, abandon in resultats:
        if abandon:
            p = 0.27                                   # raccroché sans réponse
        elif attente > 60:
            p = 0.14                                   # décroché, mais après plus d'une minute
        else:
            p = 0.06
        departs.append(alea.random() < p)
    return {"appels": tuple(appels), "departs": tuple(departs)}


def appels_exemple() -> list[tuple]:
    """Les 6 806 appels de la semaine dernière : [(arrivée, durée, patience), …] en secondes, dans le désordre."""
    return list(_semaine()["appels"])


def departs_exemple() -> list[bool]:
    """Pour chaque appel de `appels_exemple()`, dans le même ordre : le client a-t-il résilié dans les deux mois ?"""
    return list(_semaine()["departs"])


def prevision_exemple() -> list[tuple[str, int]]:
    """La prévision d'appels : [(« 08:00 », 60), (« 08:30 », 90), …] : 24 demi-heures, en appels PAR HEURE."""
    return [(_heure(k), n) for k, n in enumerate(_PREVISION)]


def planning_actuel_exemple() -> list[int]:
    """Le planning actuel : combien de vacations de 4 h commencent à chaque demi-heure (14 à 8 h, 12 h et 16 h)."""
    debuts = [0] * len(_PREVISION)
    for k in (0, 8, 16):
        debuts[k] = CONSEILLERS_ACTUELS
    return debuts

"""Outils communs de la série 44 : les appels d'un lundi au service client, et le planning des conseillers.

Depuis septembre, le service client répond moins bien : pour économiser, la direction a fixé l'équipe à 10
conseillers par créneau de 4 heures, toute la journée (ils étaient 13). Le tableau de bord du centre d'appels
affiche pourtant des chiffres rassurants. Awa, chargée de la planification, veut savoir combien de clients
attendent vraiment, combien raccrochent, si ceux-là partent plus que les autres, et combien de conseillers il
faudrait, heure par heure. Ce module lui fournit :

- `demande_exemple()` : les appels reçus un lundi, de 8 h à 20 h, avec ce que le standard ne voit pas mais
  qu'on estime d'après les semaines précédentes : la DURÉE de conversation dont chaque client a besoin, et sa
  PATIENCE (combien de secondes il attend avant de raccrocher). [{"client", "arrivee", "duree", "patience"}] ;
- `journal_exemple()` : ce qui s'est passé ce lundi-là, tel que l'exporte le standard : pour chaque appel,
  l'arrivée, le moment où un conseiller a DÉCROCHÉ (None si le client a raccroché avant), la FIN de l'appel (fin
  de la conversation, ou moment où le client a raccroché), et si le client est parti dans les trois mois ;
- `planning_actuel()` : le planning de septembre, au format du défi 270 : combien de conseillers COMMENCENT une
  vacation de 4 heures à chaque heure d'ouverture (10 à 8 h, 10 à 12 h, 10 à 16 h) ;
- `horaires(debuts, duree_h)` : ce planning en horaires de conseillers [(début, fin)] en secondes, pour le
  simulateur (défi 271) ;
- `generer_demande(...)` : d'autres journées, reproductibles, pour vos propres essais ;
- `en_heure(secondes)` : 3725 secondes après l'ouverture → '09:02:05', pour lire les résultats ;
- les constantes `HEURES` (12 heures d'ouverture), `SEUIL_S` (20 s : l'objectif « 80 % des appels décrochés en
  20 secondes »), `OBJECTIF` (0,8) et `DUREE_VACATION_H` (4).

Toutes les heures sont des nombres ENTIERS de secondes depuis l'ouverture (8 h 00 = 0, 20 h 00 = 43 200) : les
calculs sur des entiers sont exacts, et les égalités (un conseiller qui se libère à la seconde où un client
arrive) ne dépendent pas d'un arrondi. Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les
sans crainte.

    >>> demande = demande_exemple()
    >>> len(demande), demande[0]
    (1279, {'client': 'C14842', 'arrivee': 47, 'duree': 85, 'patience': 773})
    >>> journal = journal_exemple()
    >>> journal[0]
    {'client': 'C14842', 'arrivee': 47, 'decroche': 47, 'fin': 132, 'parti': True}
    >>> planning_actuel(), en_heure(journal[-1]["arrivee"])
    ([10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0], '19:59:11')
"""

from __future__ import annotations

import functools
import heapq
import math
import random

HEURES = 12                     # de 8 h à 20 h
OUVERTURE_H = 8
SEUIL_S = 20                    # « décroché en 20 secondes ou moins »
OBJECTIF = 0.8                  # 80 % des appels décrochés en 20 secondes : l'objectif du service client
DUREE_VACATION_H = 4            # une vacation dure 4 heures (8 h-12 h, 9 h-13 h…)

# Les appels par heure d'un lundi (le pic du matin, le creux du déjeuner, le pic de fin d'après-midi).
APPELS_PAR_HEURE_LUNDI = (40, 120, 190, 180, 90, 70, 100, 170, 160, 80, 50, 30)
_PLANNING_ACTUEL = (10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0)


def en_heure(secondes) -> str:
    """L'heure qu'il est `secondes` après l'ouverture (8 h 00), en « hh:mm:ss ».

        >>> en_heure(0), en_heure(3725), en_heure(12 * 3600)
        ('08:00:00', '09:02:05', '20:00:00')
    """
    s = int(round(secondes))
    return f"{OUVERTURE_H + s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def horaires(debuts, duree_h: int = DUREE_VACATION_H) -> list[tuple[int, int]]:
    """Le planning `debuts` (combien de conseillers commencent à chaque heure, défi 270) en horaires [(début, fin)]
    de conseillers, en secondes depuis l'ouverture : un conseiller par vacation, dans l'ordre des heures.

        >>> horaires([2, 0, 1], 2)
        [(0, 7200), (0, 7200), (7200, 14400)]
    """
    return [(h * 3600, (h + duree_h) * 3600) for h, nombre in enumerate(debuts) for _ in range(nombre)]


def generer_demande(appels_par_heure=APPELS_PAR_HEURE_LUNDI, duree_mediane_s: float = 240,
                    patience_moyenne_s: float = 360, graine: int = 0) -> list[dict]:
    """Une journée d'appels, reproductible : [{"client", "arrivee", "duree", "patience"}], rangés par arrivée.

    Les arrivées suivent un processus de Poisson (au hasard, `appels_par_heure[h]` appels en moyenne pendant
    l'heure h) ; les durées une loi log-normale (beaucoup d'appels courts, quelques très longs ; 30 s au moins) ;
    les patiences une loi exponentielle (5 s au moins). Tout est en secondes ENTIÈRES.

        >>> premier, second = generer_demande((60,), graine=1)[:2]
        >>> premier
        {'client': 'C14384', 'arrivee': 8, 'duree': 167, 'patience': 288}
        >>> second
        {'client': 'C16661', 'arrivee': 121, 'duree': 197, 'patience': 119}
    """
    alea = random.Random(graine)
    arrivees = []
    for h, taux in enumerate(appels_par_heure):
        if taux <= 0:
            continue
        t = h * 3600.0
        while True:
            t += alea.expovariate(taux / 3600)
            if t >= (h + 1) * 3600:
                break
            arrivees.append(int(t))
    numeros = alea.sample(range(1, max(20_000, 2 * len(arrivees))), len(arrivees))
    appels = []
    for arrivee, numero in zip(sorted(arrivees), numeros):
        duree = max(30, round(alea.lognormvariate(math.log(duree_mediane_s), 0.6)))
        patience = max(5, round(alea.expovariate(1 / patience_moyenne_s)))
        appels.append({"client": f"C{numero:05d}", "arrivee": arrivee, "duree": duree, "patience": patience})
    return appels


def _simuler(appels: list[dict], horaires_conseillers: list[tuple[int, int]]) -> list[dict]:
    """La file d'attente du standard (le défi 271 : ne lisez pas ceci avant de l'avoir écrit)."""
    tas = [(debut, k, fin) for k, (debut, fin) in enumerate(horaires_conseillers)]
    heapq.heapify(tas)
    resultat: list[dict] = [{} for _ in appels]
    for i in sorted(range(len(appels)), key=lambda i: appels[i]["arrivee"]):
        appel = appels[i]
        arrivee = appel["arrivee"]
        while tas and max(arrivee, tas[0][0]) >= tas[0][2]:
            heapq.heappop(tas)
        if tas and max(arrivee, tas[0][0]) - arrivee <= appel["patience"]:
            libre, k, fin = heapq.heappop(tas)
            decroche = max(arrivee, libre)
            heapq.heappush(tas, (decroche + appel["duree"], k, fin))
            resultat[i] = {**appel, "decroche": decroche, "fin": decroche + appel["duree"]}
        else:
            resultat[i] = {**appel, "decroche": None, "fin": arrivee + appel["patience"]}
    return resultat


@functools.lru_cache(maxsize=1)
def _lundi() -> dict:
    """Tout l'exemple, tiré une fois (graine fixe) : les objets publics en sont des COPIES."""
    demande = generer_demande(APPELS_PAR_HEURE_LUNDI, 240, 360, graine=44)
    alea = random.Random(4400)
    journal = []
    for appel in _simuler(demande, horaires(_PLANNING_ACTUEL)):
        if appel["decroche"] is None:
            risque = 0.22                                    # il a raccroché : il résilie souvent en ligne
        else:
            attente = appel["decroche"] - appel["arrivee"]
            risque = 0.05 if attente <= SEUIL_S else 0.05 + 0.15 * min(1.0, attente / 300)
        journal.append({"client": appel["client"], "arrivee": appel["arrivee"], "decroche": appel["decroche"],
                        "fin": appel["fin"], "parti": alea.random() < risque})
    return {"demande": tuple(demande), "journal": tuple(journal)}


def demande_exemple() -> list[dict]:
    """Les 1 279 appels du lundi : [{"client", "arrivee", "duree", "patience"}], rangés par arrivée (secondes).

        >>> from collections import Counter
        >>> demande = demande_exemple()
        >>> par_heure = Counter(appel["arrivee"] // 3600 for appel in demande)
        >>> [par_heure[h] for h in range(HEURES)]                      # le pic de 10 h, le creux de midi
        [45, 130, 204, 181, 85, 78, 89, 154, 154, 75, 51, 33]
        >>> round(sum(appel["duree"] for appel in demande) / len(demande), 1)     # la durée moyenne, en secondes
        281.2
    """
    return [dict(appel) for appel in _lundi()["demande"]]


def journal_exemple() -> list[dict]:
    """Le journal du standard pour ce lundi : [{"client", "arrivee", "decroche", "fin", "parti"}].

    `decroche` vaut None quand le client a raccroché avant qu'on lui réponde ; `fin` est alors le moment où il a
    raccroché. `parti` dit s'il a résilié dans les trois mois qui ont suivi.

        >>> journal = journal_exemple()
        >>> sum(appel["decroche"] is None for appel in journal)        # clients qui ont raccroché
        194
    """
    return [dict(appel) for appel in _lundi()["journal"]]


def planning_actuel() -> list[int]:
    """Le planning de septembre : combien de conseillers commencent une vacation de 4 heures à chaque heure.

        >>> horaires(planning_actuel())[:2], len(horaires(planning_actuel()))
        ([(0, 14400), (0, 14400)], 30)
    """
    return list(_PLANNING_ACTUEL)

"""Série 44 — Décrocher à temps : mesurer l'attente, la loi de Little, Erlang C, dimensionner, planifier, simuler.

📞 VOUS VOULEZ DÉVELOPPER ? UN JOUR, ON VOUS DEMANDERA « COMBIEN DE PERSONNES (OU DE SERVEURS, DE CAISSES, DE
WORKERS) FAUT-IL ? ». UNE FILE D'ATTENTE NE SE JUGE NI À L'ŒIL, NI À LA MOYENNE QUI ARRANGE. ICI, VOUS ÉCRIVEZ CE
QUE FONT LES LOGICIELS DE CENTRES D'APPELS (GENESYS, NICE, CALABRIO), ET CE QUE FAIT TOUT INGÉNIEUR QUI
DIMENSIONNE UN POOL DE CONNEXIONS OU UNE FLOTTE DE WORKERS.

Depuis septembre, le service client répond moins bien : pour économiser, la direction a fixé l'équipe à 10
conseillers par créneau de 4 heures, toute la journée (ils étaient 13). Le tableau de bord du centre d'appels
affiche 57 % d'appels décrochés en 20 secondes, et des conseillers occupés 70 % du temps : « il y a de la marge ».
Awa, chargée de la planification, a le journal du standard d'un lundi et l'historique des départs. Elle écrit les
six outils de cette série :

    266 mesurer                   les indicateurs de la journée, calculés sur TOUS les appels (abandons compris)
    267 file_attente              combien de clients attendent, en moyenne dans le TEMPS, et au plus fort
    268 erlang_c                  la probabilité d'attendre avec N conseillers, sans que les factorielles débordent
    269 conseillers_necessaires   le plus petit N qui décroche 80 % des appels en 20 s
    270 planning                  les vacations de 4 heures qui couvrent les besoins de chaque heure, au plus juste
    271 simuler                   rejouer la journée avec un autre planning : une simulation à événements discrets

Prérequis : la série 00 et les `dict`. Les tas (`heapq`, série 24), le tri stable et les sommes cumulées
l'éclairent ; la série 28 (résumer un flux) et la série 26 (conclure sans se tromper) en sont les cousines.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- les heures sont des SECONDES depuis l'ouverture (8 h 00 = 0, 20 h 00 = 43 200), en nombres entiers dans les
  exemples : `en_heure(3725)` donne '09:02:05' ;
- un appel du journal est un dict {"arrivee", "decroche", "fin"} (et parfois d'autres clés, à ignorer) :
  `decroche` vaut None si le client a raccroché avant qu'on lui réponde (un ABANDON), et `fin` est alors le
  moment où il a raccroché. L'ATTENTE d'un appel décroché va de l'arrivée au décroché ; celle d'un abandon, de
  l'arrivée à la fin ;
- le TRAFIC, en erlangs, est le nombre moyen de conversations en cours : appels par heure × durée moyenne d'un
  appel en heures. 190 appels par heure de 281 s chacun, c'est 190 × 281 / 3600 = 14,8 erlangs : il faut AU
  MOINS 15 conseillers, sinon la file grossit sans fin, et en pratique quelques-uns de plus ;
- la LOI DE LITTLE : le nombre moyen de clients en attente = le nombre d'arrivées par seconde × l'attente moyenne
  (L = λ × W). Elle vaut pour n'importe quelle file, sans hypothèse ;
- le défi 269 utilise `erlang_c` (défi 268). Pendant sa vérification, le correcteur lui substitue la sienne :
  chaque défi est noté seul, mais faites le 268 d'abord, sinon vos essais seront faux.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_44 import *
    >>> journal = journal_exemple()                # les 1 279 appels du lundi, tels que les a notés le standard
    >>> journal[0]
    {'client': 'C14842', 'arrivee': 47, 'decroche': 47, 'fin': 132, 'parti': True}
    >>> demande = demande_exemple()                # les mêmes appels, avec la durée et la patience de chacun
    >>> demande[0]
    {'client': 'C14842', 'arrivee': 47, 'duree': 85, 'patience': 773}

Quand les six défis sont réussis, Awa peut répondre à la direction :

    >>> m = mesurer(journal)
    >>> round(m["niveau_service"] * m["appels"] / m["decroches"], 3)     # le tableau de bord : 57 % « en 20 s »
    0.567
    >>> round(m["niveau_service"], 3), round(m["taux_abandon"], 3)       # la vérité : 48 %, et 15 % raccrochent
    (0.481, 0.152)
    >>> def taux_de_depart(appels):
    ...     return round(sum(a["parti"] for a in appels) / len(appels), 3)
    >>> taux_de_depart([a for a in journal if a["decroche"] is None])   # ceux qui ont raccroché sont partis à 16 %
    0.16
    >>> taux_de_depart([a for a in journal if a["decroche"] is not None and a["decroche"] - a["arrivee"] <= 20])
    0.062
    >>> occupe = sum(a["fin"] - a["decroche"] for a in journal if a["decroche"] is not None)
    >>> round(occupe / (30 * 4 * 3600), 2)                               # 30 vacations de 4 h : occupés à 70 %…
    0.7
    >>> moyenne, maximum = file_attente(journal, 2 * 3600, 3 * 3600)     # … mais de 10 h à 11 h
    >>> round(moyenne, 1), maximum
    (6.5, 17)
    >>> from collections import Counter
    >>> par_heure = Counter(a["arrivee"] // 3600 for a in demande)
    >>> duree = sum(a["duree"] for a in demande) / len(demande)          # 281 s par appel, en moyenne
    >>> round(par_heure[2] * duree / 3600, 1), erlang_c(par_heure[2] * duree / 3600, 10)   # à 10 h : 15,9 erlangs
    (15.9, 1.0)
    >>> besoins = [conseillers_necessaires(par_heure[h], duree) for h in range(HEURES)]
    >>> besoins
    [6, 14, 20, 18, 10, 9, 10, 16, 16, 9, 7, 5]
    >>> debuts = planning(besoins, DUREE_VACATION_H)
    >>> debuts, sum(debuts)
    ([6, 8, 6, 0, 0, 3, 7, 6, 5, 0, 0, 0], 41)
    >>> round(mesurer(simuler(demande, horaires(debuts)))["niveau_service"], 3)
    0.943
    >>> moins = planning([conseillers_necessaires(par_heure[h], duree, objectif=0.5) for h in range(HEURES)], 4)
    >>> sum(moins), round(mesurer(simuler(demande, horaires(moins)))["niveau_service"], 3)
    (36, 0.862)
    >>> round(mesurer(simuler(demande, horaires([12, 0, 0, 0] * 3)))["niveau_service"], 3)      # 36 aussi, à plat
    0.623

Sur la journée, les conseillers ne sont occupés que 70 % du temps : la direction a raison en moyenne, et tort à
chaque pic. De 10 h à 11 h, il arrive 15,9 erlangs de travail pour 10 conseillers : la file grossit sans fin
(Erlang C vaut 1), jusqu'à 17 clients en ligne, et le creux de 13 h n'y change rien. Résultat : 48 % des appels
décrochés en 20 s, et non 57 % (le tableau de bord ne compte pas les 194 clients qui ont raccroché), et ceux-là
partent à 16 %, contre 6 % de ceux qu'on a pris tout de suite.

Erlang C demande 41 vacations pour tenir 80 % chaque heure, et la simulation donne 94 % : Erlang ignore ceux qui
raccrochent, il est prudent. Avec 36 vacations (6 de plus qu'aujourd'hui, autant que 12 conseillers par créneau),
placées heure par heure, la simulation donne déjà 86 % ; les mêmes 36 vacations posées à plat, 62 %. Ce n'est pas
d'abord une question d'effectif : c'est une question d'HEURES.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites le 268 avant le 269 (qui s'en sert) ; les
   autres sont indépendants, et l'ordre proposé est le plus progressif (le 267 reprend les attentes du 266).
2. Lancez `python scripts/defis.py --defi 266` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_44.py`.

Les outils communs (`demande_exemple`, `journal_exemple`, `planning_actuel`, `horaires`, `generer_demande`,
`en_heure`, `HEURES`, `SEUIL_S`, `OBJECTIF` et `DUREE_VACATION_H`) sont dans `defis/centre_appels.py`, déjà
importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import heapq  # noqa: F401 — heapq.heapify, heapq.heappush, heapq.heappop (défi 271)
import math  # noqa: F401 — math.exp, math.floor, math.isfinite, math.inf

from defis.centre_appels import (  # noqa: F401
    DUREE_VACATION_H,
    HEURES,
    OBJECTIF,
    SEUIL_S,
    demande_exemple,
    en_heure,
    generer_demande,
    horaires,
    journal_exemple,
    planning_actuel,
)

# ============================================================================ Mesurer


def mesurer(appels, seuil_s=SEUIL_S) -> dict:
    """Défi 266 🟢 — Les indicateurs de la journée, calculés sur TOUS les appels, abandons compris.

        >>> journal = [{"arrivee": 0, "decroche": 5, "fin": 300},         # décroché en 5 s
        ...            {"arrivee": 10, "decroche": 30, "fin": 200},       # décroché en 20 s pile
        ...            {"arrivee": 20, "decroche": 140, "fin": 400},      # décroché au bout de 2 minutes
        ...            {"arrivee": 30, "decroche": None, "fin": 210}]     # a raccroché au bout de 3 minutes
        >>> mesurer(journal)        # doctest: +NORMALIZE_WHITESPACE
        {'appels': 4, 'decroches': 3, 'abandons': 1, 'taux_abandon': 0.25, 'attente_moyenne_s': 81.25,
         'attente_decroches_s': 48.333333333333336, 'niveau_service': 0.5}

    Ici : les quatre clients ont attendu 5, 20, 120 et 180 secondes (le dernier, jusqu'à ce qu'il raccroche) :
    81,25 s en moyenne. Le tableau de bord, qui ne regarde que les appels décrochés, affiche 48,3 s. Deux appels
    sur QUATRE ont été décrochés en 20 s ou moins (20 s pile compte) : 50 %, et non deux sur trois.

    Les règles :

    - `appels` est une liste (ou un tuple) de dict ; chacun a les clés "arrivee", "decroche" et "fin" (d'autres
      clés, comme "client" ou "parti", sont ignorées). "arrivee" et "fin" sont des nombres (int ou float, pas des
      bool), "decroche" un nombre ou None. Sinon TypeError ; une clé absente, un NaN, ou un appel incohérent
      (décroché avant d'arriver, terminé avant d'être décroché ou d'arriver) → ValueError ;
    - l'ATTENTE d'un appel décroché est decroche - arrivee ; celle d'un abandon (decroche None), fin - arrivee ;
    - la réponse est un dict, avec ces clés :
      "appels" : le nombre d'appels ; "decroches" : ceux qui ont eu un conseiller ; "abandons" : les autres ;
      "taux_abandon" : abandons / appels ;
      "attente_moyenne_s" : la moyenne des attentes de TOUS les appels ;
      "attente_decroches_s" : la moyenne des attentes des seuls appels décrochés (ce qu'affichent les tableaux de
      bord : l'ASA, « average speed of answer »), None s'il n'y en a aucun ;
      "niveau_service" : la part de TOUS les appels décrochés en `seuil_s` secondes ou moins (<=) ; un abandon
      compte comme un échec, même s'il a raccroché au bout de 3 secondes ;
    - `seuil_s` est un nombre >= 0 (sinon TypeError ou ValueError) ; aucun appel → ValueError (on ne divise pas
      par zéro) ; `appels` n'est pas modifié.

    Ce que vous apprenez : qu'un indicateur, c'est d'abord un DÉNOMINATEUR (tous les appels, pas les seuls
    décrochés), le biais du survivant (ceux qui ont raccroché ont attendu le plus longtemps, et ce sont eux qu'on
    oublie), et à valider un journal avant d'en tirer des chiffres.

    Indice : une boucle qui calcule l'attente de chaque appel ; deux listes (`toutes`, `des_decroches`) ;
    `sum(attente <= seuil_s for attente in des_decroches)` compte les True.
    """
    raise NotImplementedError


def file_attente(appels, debut, fin) -> tuple[float, int]:
    """Défi 267 🟡 — (nombre MOYEN de clients en attente entre `debut` et `fin`, nombre MAXIMAL au même instant).

        >>> journal = [{"arrivee": 0, "decroche": 60, "fin": 400},        # attend de 0 à 60
        ...            {"arrivee": 30, "decroche": 90, "fin": 300},       # attend de 30 à 90
        ...            {"arrivee": 60, "decroche": None, "fin": 120},     # attend de 60 à 120, puis raccroche
        ...            {"arrivee": 200, "decroche": 200, "fin": 500}]     # décroché tout de suite : n'attend pas
        >>> file_attente(journal, 0, 120)
        (1.5, 2)
        >>> file_attente(journal, 0, 600)
        (0.3, 2)

    Ici : entre 0 et 120 s, les trois premiers clients attendent 60 s chacun : 180 secondes d'attente en 120
    secondes, soit 1,5 client en ligne en moyenne. Au plus fort, ils sont 2 : à 60 s, le premier est décroché à
    l'instant où le troisième arrive, ils ne sont jamais 3. Sur 600 s, les mêmes 180 secondes ne font plus que 0,3
    client en moyenne. Et le quatrième n'a jamais attendu : il ne compte pas, même à 200 s.

    Les règles :

    - `appels` suit les règles du défi 266 (mêmes erreurs) ; `debut` et `fin` sont des nombres (sinon TypeError),
      avec debut < fin (sinon ValueError : une fenêtre vide n'a pas de moyenne) ;
    - un client est en attente sur l'intervalle [arrivee, fin de son attente[ : FERMÉ au début, OUVERT à la fin
      (la fin de l'attente est le décroché, ou la fin pour un abandon). À l'instant où l'un est décroché et où un
      autre arrive, ils ne sont pas deux ; un appel décroché à son arrivée n'attend jamais ;
    - la MOYENNE est une moyenne dans le TEMPS : chaque client compte pour la durée de son attente qui tombe DANS
      [debut, fin] (un client arrivé avant `debut` compte à partir de `debut`), divisée par fin - debut. Ce n'est
      PAS la moyenne des nombres relevés à chaque arrivée ou départ ;
    - le MAXIMUM est le plus grand nombre de clients en attente au même instant entre debut et fin (0 si
      personne) : un int ;
    - la réponse est un tuple (moyenne, maximum) ; un journal vide donne (0.0, 0) ; `appels` n'est pas modifié ;
    - le correcteur passe 10 000 appels et compte les lignes exécutées : comparer chaque appel à tous les autres
      (100 millions de paires) est trop lent. Triez UNE fois.

    Ce que vous apprenez : la différence entre une moyenne d'événements et une moyenne dans le temps (une seconde
    de rush ne vaut pas une heure de calme), le balayage d'événements triés (à instant égal, les départs AVANT les
    arrivées : trier des tuples `(instant, -1)` et `(instant, +1)` le fait tout seul), et la loi de Little :
    quand la fenêtre contient toutes les attentes, moyenne = (appels / durée) × attente moyenne du défi 266.

    Indice : pour chaque appel, `a, b = max(arrivee, debut), min(fin_attente, fin)` ; si `a < b`, ajoutez `b - a`
    au total, et les événements `(a, +1)` et `(b, -1)` à une liste ; `sorted(evenements)`, puis un compteur.
    """
    raise NotImplementedError


# ============================================================================ Dimensionner


def erlang_c(trafic, conseillers) -> float:
    """Défi 268 🟡 — La probabilité qu'un appel ATTENDE, pour `trafic` erlangs et `conseillers` conseillers.

        >>> round(erlang_c(0.5, 1), 4)          # un conseiller, occupé la moitié du temps : un client sur deux attend
        0.5
        >>> round(erlang_c(10, 12), 4)          # 10 erlangs, 12 conseillers : 45 % des clients attendent
        0.4494
        >>> round(erlang_c(950, 1000), 4)       # un très grand centre : la formule des manuels déborderait
        0.0683
        >>> erlang_c(15.9, 10)                  # lundi, 10 h : plus de travail que de conseillers
        1.0

    Ici : avec 10 erlangs (10 conversations en cours en moyenne) et 12 conseillers, 45 % des clients attendent un
    peu. Avec 950 erlangs et 1 000 conseillers, 6,8 % seulement, alors que chaque conseiller est occupé à 95 % :
    c'est l'économie d'échelle des grandes files. Et quand le trafic dépasse le nombre de conseillers, tout le
    monde attend : la file grossit sans fin.

    Les règles :

    - la formule d'Erlang C, avec A = trafic et N = conseillers :
          C = X / (S + X),  où  X = A^N / N! × N / (N - A)  et  S = Σ (k de 0 à N - 1) A^k / k! ;
      elle suppose que les appels arrivent au hasard (Poisson), durent un temps exponentiel, et que personne ne
      raccroche ;
    - mais A^N et N! débordent : `math.factorial(171)` ne tient plus dans un float, et `950.0 ** 1000` lève
      OverflowError. Le correcteur va jusqu'à 5 000 conseillers. Utilisez la récurrence d'ERLANG B (la probabilité de
      rejet sans file), où chaque terme reste entre 0 et 1 :
          B = 1, puis pour k de 1 à N :  B = A × B / (k + A × B),
      et enfin C = N × B / (N - A × (1 - B)) ;
    - si trafic >= conseillers, la réponse est 1.0 (la formule donnerait un nombre négatif ou plus grand que 1) ;
      si trafic vaut 0, 0.0 ;
    - `trafic` est un nombre >= 0 (int ou float, pas un bool : sinon TypeError ; négatif ou NaN → ValueError) ;
      `conseillers` un int >= 1 (pas un bool ni un float : sinon TypeError ou ValueError) ; la réponse est un float.

    Ce que vous apprenez : qu'une formule juste au papier peut être inutilisable en float (les factorielles
    débordent, leur quotient non), à la remplacer par une récurrence stable (chaque étape reste dans [0, 1]), et
    qu'une file d'attente n'est pas linéaire : à 95 % d'occupation, 10 conseillers sont débordés, 1 000 non.

    Indice : `b = 1.0` ; `for k in range(1, conseillers + 1): b = trafic * b / (k + trafic * b)`. Vérifiez vos
    résultats avec la formule du manuel pour de petites valeurs (`math.factorial` y suffit).
    """
    raise NotImplementedError


def conseillers_necessaires(appels_par_heure, duree_moyenne_s, seuil_s=SEUIL_S, objectif=OBJECTIF) -> int:
    """Défi 269 🟡 — Le plus petit nombre de conseillers qui décroche `objectif` des appels en `seuil_s` s ou moins

        >>> conseillers_necessaires(190, 281)            # lundi, 10 h : 190 appels de 281 s, 80 % en 20 s
        19
        >>> conseillers_necessaires(190, 281, objectif=0.5), conseillers_necessaires(190, 281, seuil_s=60)
        (17, 18)
        >>> conseillers_necessaires(1900, 281)           # dix fois plus d'appels : PAS dix fois plus de monde
        158
        >>> conseillers_necessaires(0, 281)              # personne n'appelle
        0

    Ici : 190 appels de 281 secondes font 14,8 erlangs. Avec 15 conseillers, la file tient à peine (presque tout
    le monde attend) ; il en faut 19 pour décrocher 80 % des appels en 20 s. Dix fois plus d'appels demandent 158
    conseillers, pas 190 : une grande équipe absorbe mieux les à-coups.

    Les règles :

    - le trafic, en erlangs : A = appels_par_heure × duree_moyenne_s / 3600 (des heures, pas des secondes) ;
    - avec N conseillers (N > A), le niveau de service est
          1 - erlang_c(A, N) × exp(-(N - A) × seuil_s / duree_moyenne_s) :
      `erlang_c` est la part des clients qui attendent, l'exponentielle la part de ceux-là qui attendent plus de
      `seuil_s` secondes ;
    - la réponse est le PLUS PETIT N (un int) dont le niveau de service est >= objectif ; essayez N = ⌊A⌋ + 1,
      puis N + 1, et ainsi de suite (pour N <= A, la file grossit sans fin : ce n'est jamais une réponse). Aucun
      appel (A = 0) → 0 ;
    - `appels_par_heure` est un nombre >= 0, `duree_moyenne_s` un nombre > 0, `seuil_s` un nombre >= 0, `objectif`
      un nombre strictement entre 0 et 1 (100 % ne s'atteint jamais) : sinon TypeError (pas un nombre, ou un
      bool) ou ValueError ;
    - utilisez `erlang_c` (défi 268) : pendant la vérification, le correcteur lui substitue la sienne.

    Ce que vous apprenez : à passer d'une probabilité à une DÉCISION (le plus petit N qui tient l'objectif), à
    garder les unités sous contrôle (des appels par HEURE et des durées en SECONDES), et pourquoi la règle de pouce
    « 85 % d'occupation » (N = ⌈A / 0,85⌉) se trompe : trop de monde dans une grande équipe, trop peu dans une
    petite.

    Indice : `trafic = appels_par_heure * duree_moyenne_s / 3600` ; `n = math.floor(trafic) + 1` ; une boucle
    `while` qui augmente n tant que le niveau de service est < objectif.
    """
    raise NotImplementedError


# ============================================================================ Planifier et rejouer


def planning(besoins, duree) -> list[int]:
    """Défi 270 🟡 — Combien de conseillers COMMENCENT une vacation de `duree` heures à chaque heure, au plus juste.

        >>> planning([2, 3, 3, 1, 4], 2)
        [2, 1, 2, 4, 0]
        >>> planning([6, 14, 20, 18, 10, 9, 10, 16, 16, 9, 7, 5], 4)      # lundi, des vacations de 4 heures
        [6, 8, 6, 0, 0, 3, 7, 6, 5, 0, 0, 0]

    Ici (1er exemple) : la journée dure 5 heures, une vacation 2. À l'heure 0, il en faut 2 : 2 conseillers
    commencent (ils restent les heures 0 et 1). À l'heure 1, il en manque 1 : il commence à 1. À l'heure 2, les
    deux premiers sont partis, il en faut 3 : 2 commencent. À l'heure 4, il en faut 4, mais une vacation commencée
    à 4 finirait après la fermeture : ils commencent à 3, la dernière heure possible. 9 vacations : impossible de
    faire moins. Faire commencer chaque renfort le plus TÔT possible (à h - duree + 1) en demanderait 11.

    Les règles :

    - `besoins[h]` est le nombre de conseillers nécessaires pendant l'heure h de la journée (un int >= 0) ; une
      vacation commencée à l'heure s couvre les heures s, s + 1, …, s + duree - 1, et doit finir avant la
      fermeture : s + duree <= len(besoins) ;
    - la réponse est une liste d'int de même longueur que `besoins` : `debuts[s]` conseillers commencent à
      l'heure s. Chaque heure h doit être couverte : le nombre de vacations en cours à h est >= besoins[h] ; et
      le NOMBRE TOTAL de vacations doit être le plus petit possible ;
    - la méthode (gloutonne, et optimale) : avancez heure par heure ; s'il manque k conseillers à l'heure h,
      faites-en commencer k le plus TARD possible, c'est-à-dire à h, ou à len(besoins) - duree si h est trop
      tard pour qu'une vacation tienne avant la fermeture. Le correcteur accepte tout planning valide et aussi
      court ; celui-là en est un ;
    - `besoins` est une liste (ou un tuple) d'int >= 0 (pas des bool ni des float : un besoin de 13,5 conseillers
      s'arrondit AVANT) ; `duree` un int >= 1 : sinon TypeError ou ValueError. Une vacation plus longue que la
      journée → ValueError ; aucune heure → [] ; `besoins` n'est pas modifié.

    Ce que vous apprenez : un algorithme glouton et ce qui le rend optimal (commencer plus tard couvre plus
    d'heures à venir), le cas du bord (la fin de journée), et suivre une fenêtre glissante avec un tableau des
    départs (`partent[s + duree] -= k`) plutôt qu'en recomptant tout à chaque heure.

    Indice : `debuts = [0] * n`, `partent = [0] * (n + 1)`, `presents = 0` ; à chaque heure,
    `presents += partent[h]`, puis s'il en manque : `s = min(h, n - duree)`, `debuts[s] += manque`,
    `presents += manque`, `partent[s + duree] -= manque`.
    """
    raise NotImplementedError


def simuler(appels, conseillers) -> list[dict]:
    """Défi 271 🔴 — Rejouer la journée : qui est décroché, quand, et qui raccroche, avec ces conseillers-là.

        >>> appels = [{"client": "A", "arrivee": 0, "duree": 300, "patience": 60},
        ...           {"client": "B", "arrivee": 10, "duree": 120, "patience": 60},
        ...           {"client": "C", "arrivee": 20, "duree": 200, "patience": 300},
        ...           {"client": "D", "arrivee": 40, "duree": 60, "patience": 30},
        ...           {"client": "E", "arrivee": 130, "duree": 100, "patience": 200}]
        >>> [(a["client"], a["decroche"], a["fin"]) for a in simuler(appels, 2)]
        [('A', 0, 300), ('B', 10, 130), ('C', 130, 330), ('D', None, 70), ('E', 300, 400)]
        >>> [(a["client"], a["decroche"], a["fin"]) for a in simuler(appels, [(0, 100), (0, 1000)])]
        [('A', 0, 300), ('B', 10, 130), ('C', 130, 330), ('D', None, 70), ('E', 330, 430)]
        >>> simuler(appels, 2)[0]
        {'client': 'A', 'arrivee': 0, 'duree': 300, 'patience': 60, 'decroche': 0, 'fin': 300}

    Ici, avec 2 conseillers : A et B sont décrochés à leur arrivée. C attend que B finisse (130 s). D, arrivé à
    40 s, devrait attendre jusqu'à 130 s au plus tôt, et C est avant lui : sa patience (30 s) est épuisée à 70 s,
    il raccroche. E arrive à 130 s, à l'instant où B se libère, mais C attendait avant lui : E attend la fin de A
    (300 s). Dans le 2e exemple, le premier conseiller termine son service à 100 s : il finit l'appel de A
    (jusqu'à 300 s), mais n'en prend plus ; E attend la fin de C (330 s).

    Les règles :

    - `appels` est une liste de dict avec les clés "arrivee", "duree" (la conversation dont le client a besoin)
      et "patience" (combien de secondes il accepte d'attendre) : des nombres, durée et patience >= 0 (sinon
      TypeError ou ValueError) ; les autres clés sont recopiées telles quelles ;
    - `conseillers` est un int >= 0 (autant de conseillers, présents toute la journée), ou une liste d'horaires
      (debut, fin), un par conseiller, en secondes (fin peut valoir `math.inf`) : il peut DÉCROCHER pendant
      [debut, fin[ (pas à fin pile), et termine l'appel en cours même après fin. Sinon TypeError ou ValueError ;
    - premier arrivé, premier servi : les clients sont pris dans l'ordre d'ARRIVÉE (pas forcément celui de la
      liste ; à arrivée égale, dans l'ordre de la liste). Chacun est décroché au PLUS TÔT possible : dès qu'un
      conseiller est libre et en service, à partir de son arrivée. S'il est libre à la seconde où le client arrive,
      le client est décroché tout de suite ;
    - plusieurs conseillers libres ? Celui qui est libre depuis le plus LONGTEMPS (à égalité, le premier de la
      liste, ou le plus petit numéro) ;
    - si le client devrait attendre plus que sa patience (strictement : attendre exactement sa patience, c'est
      encore être décroché), il raccroche à arrivee + patience, sans occuper personne. S'il n'y a plus aucun
      conseiller en service, il raccroche aussi ;
    - la réponse est une liste de dict dans l'ORDRE de `appels` : une copie de chaque appel, complétée de
      "decroche" (le moment du décroché, ou None) et "fin" (decroche + duree, ou arrivee + patience). Elle se lit
      avec `mesurer` et `file_attente`. `appels` et `conseillers` ne sont pas modifiés ;
    - le correcteur passe 14 000 appels et 100 conseillers et compte les lignes exécutées : ni horloge qui avance
      seconde par seconde, ni chaque appel comparé à tous les précédents, ni tous les conseillers passés en revue
      à chaque appel. Un TAS (`heapq`) de tuples (libre à partir de, numéro, fin de service) donne le bon
      conseiller en O(log n).

    Ce que vous apprenez : la simulation à événements discrets (sauter d'un événement au suivant, sans horloge),
    le tas comme file de priorité, à traiter dans un ordre (les arrivées) et répondre dans un autre (celui de la
    liste), et à rendre une simulation exacte et reproductible en fixant chaque égalité par une règle.

    Indice : `tas = [(debut, k, fin) for k, (debut, fin) in enumerate(horaires)]`, `heapq.heapify(tas)` ; pour
    chaque client (dans l'ordre `sorted(range(len(appels)), key=…)`), sortez d'abord du tas les conseillers dont
    le service est fini (`max(arrivee, tas[0][0]) >= tas[0][2]` : ils ne reviendront pas) ; puis le sommet du tas
    dit quand le client serait décroché.
    """
    raise NotImplementedError

"""Série 44 — Faire attendre le moins possible : Erlang C, effectif, simulation, indicateurs, planning, file prioritaire.

☎️ VOUS VOULEZ DÉVELOPPER ? UN JOUR, VOTRE CODE SERVIRA DES DEMANDES QUI ARRIVENT TOUTES EN MÊME TEMPS : DES APPELS,
DES REQUÊTES WEB, DES TÂCHES DANS UNE FILE. ET UNE FILE D'ATTENTE NE SE DIMENSIONNE PAS AVEC UNE MOYENNE. ICI, VOUS
ÉCRIVEZ CE QUE FONT LES LOGICIELS DES CENTRES D'APPELS (ERLANG C, LE PLANNING DES ÉQUIPES), LES SIMULATEURS DE
FILES D'ATTENTE ET LES ORDONNANCEURS À PRIORITÉ (CELUI DE LINUX, CELUI DE KUBERNETES, CELUI DE CELERY).

Depuis septembre, le service client « tient ses chiffres » : 111 appels par heure en moyenne, 6 minutes chacun,
soit 11,1 heures de conversation par heure ; avec 14 conseillers au téléphone, le tableur affiche 79 %
d'occupation, et la direction conclut qu'on est large. Pourtant, l'après-midi, les clients raccrochent avant
qu'on leur réponde, et ceux qui raccrochent partent. Amina, responsable du service client, a l'export du standard
de la semaine dernière et la prévision d'appels de l'équipe planning. Elle écrit les six outils de cette série :

    266 erlang_c                  la probabilité qu'un appel attende (la formule d'Erlang C, sans factorielle)
    267 conseillers_necessaires   combien de conseillers planifier pour décrocher 80 % des appels en 20 secondes
    268 simuler_file              rejouer la semaine appel par appel : qui attend, combien, qui raccroche
    269 indicateurs               les chiffres du tableau de bord, chacun avec SA définition
    270 planifier_vacations       le moins de vacations possible pour couvrir chaque demi-heure
    271 FilePrioritaire           les clients à risque d'abord, sans que personne n'attende indéfiniment

Prérequis : la série 00 et les `dict`. Les tas (`heapq`, série 24), le défi 112 (dichotomie) et les centimes de la
série 37 (pourquoi les float trahissent les arrondis) l'éclairent ; la série 16 (asyncio) en est la cousine : une
file d'attente, vue de l'intérieur.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- le TRAFIC se mesure en ERLANGS : des heures de conversation par heure. 120 appels par heure de 6 minutes (360 s),
  c'est 120 × 360 / 3600 = 12 erlangs : il faut au moins 12 conseillers occupés en permanence ;
- s'il y a moins de conseillers que d'erlangs (n <= A), la file grossit sans fin : chaque heure apporte plus de
  travail qu'on n'en fait. S'il y en a plus, la file reste stable, mais des clients attendent quand même : les
  appels n'arrivent pas à intervalles réguliers, ils arrivent au HASARD (un processus de Poisson) ;
- la formule d'ERLANG C (Agner Erlang, 1917, pour le téléphone de Copenhague) donne la probabilité qu'un appel
  attende, avec n conseillers et A erlangs. Et la part des appels décrochés en t secondes ou moins vaut
  1 - C × exp(-(n - A) × t / durée moyenne). On l'écrit par une récurrence (défi 266) ;
- ces formules supposent que personne ne raccroche. La réalité, si : chaque client a sa PATIENCE. Pour la
  mesurer, on rejoue les appels un par un (défi 268) ;
- les défis 267 et 270 utilisent d'autres défis : pendant la vérification du 267, le correcteur prête sa propre
  `erlang_c` à votre fichier : chaque défi est noté seul, mais faites le 266 d'abord.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_44 import *
    >>> appels = appels_exemple()                  # 6 806 appels : (arrivée, durée, patience), en secondes
    >>> appels[0]                                  # lundi 10 h 54 min 42 s, 37 s de conversation, 150 s de patience
    (10482, 37, 150)
    >>> departs = departs_exemple()                # pour chaque appel : le client est-il parti dans les deux mois ?
    >>> prevision_exemple()[14]                    # à 15 h 00 : 180 appels par heure prévus
    ('15:00', 180)

Quand les six défis sont réussis, Amina peut répondre à la direction :

    >>> round(erlang_c(111 * 360 / 3600, 14), 3)       # la MOYENNE : 11,1 erlangs pour 14 conseillers, ça passe
    0.318
    >>> erlang_c(180 * 360 / 3600, 14)                 # 15 h : 18 erlangs pour 14 conseillers, la file explose
    1.0
    >>> besoins = [conseillers_necessaires(n, DUREE_MOYENNE_S) for _, n in prevision_exemple()]
    >>> besoins                                        # pour tenir 80 % en 20 s, à chaque demi-heure
    [9, 13, 17, 19, 19, 18, 16, 14, 10, 9, 10, 15, 21, 22, 23, 22, 19, 17, 15, 14, 13, 10, 8, 6]
    >>> resultats = simuler_file(appels, CONSEILLERS_ACTUELS)
    >>> semaine = indicateurs(resultats)
    >>> round(semaine["taux_service"], 3), round(semaine["taux_abandon"], 3), semaine["p90"]
    (0.733, 0.113, 56.0)
    >>> apres_midi = indicateurs([r for a, r in zip(appels, resultats) if 12 <= creneau(a[0]) < 18])
    >>> round(apres_midi["taux_service"], 3), round(apres_midi["taux_abandon"], 3), apres_midi["p90"]
    (0.525, 0.205, 89.0)
    >>> raccroche = [d for (attente, abandon), d in zip(resultats, departs) if abandon]
    >>> vite = [d for (attente, abandon), d in zip(resultats, departs) if not abandon and attente <= 60]
    >>> round(sum(raccroche) / len(raccroche), 2), round(sum(vite) / len(vite), 2)
    (0.26, 0.06)
    >>> sum(planifier_vacations(besoins, VACATION)), sum(planning_actuel_exemple())
    (53, 42)

La moyenne de la journée disait « 32 % des appels attendent un peu » ; à 15 h, il arrive 18 heures de conversation
par heure pour 14 conseillers. Sur la semaine, le taux de service (73 %) n'a l'air « pas si loin » de l'objectif ;
entre 14 h et 17 h, il tombe à 52 %, et un client sur cinq raccroche. Or 26 % de ceux qui ont raccroché sont partis
dans les deux mois, contre 6 % de ceux décrochés en moins d'une minute. Il faut 23 conseillers à 15 h, mais 9 à
8 h : 53 vacations de 4 heures, placées aux bonnes heures, tiennent l'objectif à chaque demi-heure ; il en faudrait
69 pour avoir 23 conseillers toute la journée. Et en attendant les recrutements, la file prioritaire du défi 271
fait passer d'abord les clients que le modèle de churn juge à risque, sans laisser les autres attendre sans fin.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites le 266 d'abord (le 267 s'en sert) ; les
   autres sont indépendants, et l'ordre proposé est le plus progressif.
2. Lancez `python scripts/defis.py --defi 266` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_44.py`.

Les outils communs (`appels_exemple`, `departs_exemple`, `prevision_exemple`, `planning_actuel_exemple`,
`creneau`, `generer_appels`, `DUREE_MOYENNE_S`, `SEUIL_S`, `OBJECTIF`, `CONSEILLERS_ACTUELS` et `VACATION`) sont
dans `defis/attente.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import heapq  # noqa: F401 — heapq.heappush, heapq.heappop, heapq.heapreplace
import itertools  # noqa: F401 — itertools.count() : un compteur pour départager
import math  # noqa: F401 — math.exp, math.floor, math.ceil, math.isfinite

from defis.attente import (  # noqa: F401
    CONSEILLERS_ACTUELS,
    DUREE_MOYENNE_S,
    OBJECTIF,
    SEUIL_S,
    VACATION,
    appels_exemple,
    creneau,
    departs_exemple,
    generer_appels,
    planning_actuel_exemple,
    prevision_exemple,
)

# ============================================================================ Dimensionner


def erlang_c(trafic, agents) -> float:
    """Défi 266 🟢 — La probabilité qu'un appel attende, avec `agents` conseillers et un trafic de `trafic` erlangs.

        >>> round(erlang_c(2, 3), 4)                    # 2 erlangs, 3 conseillers : 4/9, calculable à la main
        0.4444
        >>> round(erlang_c(12, 14), 3)                  # 120 appels de 6 min par heure, 14 conseillers
        0.482
        >>> erlang_c(14, 14), erlang_c(18, 14)          # autant (ou plus) de travail que de conseillers
        (1.0, 1.0)
        >>> round(erlang_c(950, 1000), 4)               # un très grand centre : pas d'OverflowError
        0.0683

    Ici : avec 3 conseillers et 2 erlangs, 4 appels sur 9 attendent. Avec 14 conseillers et 12 erlangs (86 %
    d'occupation), presque un appel sur deux attend. À 14 erlangs pour 14 conseillers, tout le monde attend, et la
    file ne se vide jamais. Un centre de 1 000 conseillers à 950 erlangs (95 % d'occupation !) fait attendre moins
    de 7 % des appels : les grands centres sont plus efficaces que les petits.

    Les règles :

    - `trafic` est un nombre (int ou float, pas un bool) : TypeError sinon ; négatif, NaN ou infini → ValueError ;
      `agents` est un int (pas un bool, ni un float, même 14.0) : TypeError sinon ; moins de 1 → ValueError ;
    - si `trafic >= agents`, la réponse est 1.0 : la file grossit sans fin, tout le monde attend (la formule, elle,
      donnerait un nombre négatif ou plus grand que 1) ;
    - sinon, calculez d'abord ERLANG B (la probabilité qu'un appel soit rejeté s'il n'y avait pas de file) par
      récurrence : B = 1 au départ, puis pour k = 1, 2, …, agents : B = A × B / (k + A × B). Puis
      C = n × B / (n - A × (1 - B)), avec n = agents et A = trafic. Un float ;
    - pas de factorielles : `trafic ** agents / math.factorial(agents)` déborde (OverflowError) dès 171 conseillers,
      et bien avant pour un grand trafic (950 ** 1000 n'existe pas en float). Le correcteur essaie 1 000 conseillers.
      Avec la récurrence, chaque B reste entre 0 et 1.

    Ce que vous apprenez : la formule qu'utilisent tous les logiciels de centres d'appels (et le dimensionnement des
    serveurs : un « agent », c'est aussi un processus qui traite des requêtes), qu'une formule juste au papier peut
    être fausse en machine (A**n / n! : deux nombres immenses dont le quotient est raisonnable), et la récurrence qui
    les évite en ne gardant que leur rapport.

    Indice : `b = 1.0`, `for k in range(1, agents + 1): b = trafic * b / (k + trafic * b)` ; vérifiez les
    arguments d'abord, puis le cas `trafic >= agents`.
    """
    raise NotImplementedError


def conseillers_necessaires(appels_par_heure, duree_moyenne_s, objectif=OBJECTIF, seuil_s=SEUIL_S,
                            absents_pct: int = 0) -> int:
    """Défi 267 🟡 — Combien de conseillers PLANIFIER pour décrocher `objectif` des appels en `seuil_s` secondes.

        >>> conseillers_necessaires(120, 360)                     # 120 appels/h de 6 min, 80 % en 20 s
        16
        >>> conseillers_necessaires(180, 360)                     # le pic de 15 h
        23
        >>> conseillers_necessaires(120, 360, absents_pct=30)     # 30 % d'absents (congés, formations, pauses)
        23
        >>> conseillers_necessaires(160, 360, absents_pct=30)     # 21 présents : 30 planifiés, pas 31
        30
        >>> conseillers_necessaires(0, 360)
        0

    Ici : 120 appels de 6 minutes par heure font 12 erlangs. Avec 15 conseillers, 73 % des appels sont décrochés en
    20 secondes ; avec 16, 84 % : il en faut 16. S'il manque 30 % de l'équipe chaque jour (congés, formations,
    pauses), il faut en planifier 23 pour en avoir 16 présents (23 × 0,7 = 16,1). Pour 21 présents, 30 suffisent
    (30 × 0,7 = 21) ; `math.ceil(21 / (1 - 0.3))` répond pourtant 31, car en float 21 / 0,7 vaut 30.000000000000004.

    Les règles :

    - `appels_par_heure` : un nombre >= 0 ; `duree_moyenne_s` : un nombre > 0 (la durée moyenne d'une conversation,
      en SECONDES) ; `objectif` : un nombre strictement entre 0 et 1 ; `seuil_s` : un nombre >= 0 (en secondes) ;
      `absents_pct` : un int de 0 à 99 (un pourcentage entier). Un bool, un texte, un float pour `absents_pct` →
      TypeError ; une valeur hors de ces bornes, NaN ou infinie → ValueError ;
    - zéro appel → 0 conseiller. Sinon, le trafic en erlangs vaut A = appels_par_heure × duree_moyenne_s / 3600
      (des heures de conversation par heure : la durée est en secondes, l'heure en compte 3 600) ;
    - pour n conseillers PRÉSENTS (n > A), la part des appels décrochés en `seuil_s` secondes ou moins vaut
      1 - erlang_c(A, n) × exp(-(n - A) × seuil_s / duree_moyenne_s). Cherchez le PLUS PETIT n > A qui donne au
      moins `objectif` : en partant de n = ⌊A⌋ + 1 (le premier n qui laisse la file stable), puis n + 1… ;
    - enfin, s'il manque `absents_pct` % de l'équipe, la réponse est le plus petit m tel que m × (100 - absents_pct)
      / 100 >= n. Calculez-le en ENTIERS : ⌈100 × n / (100 - absents_pct)⌉, soit `-(-100 * n // (100 -
      absents_pct))`. Une division en float suivie de `math.ceil` se trompe (30.000000000000004 → 31) ;
    - la réponse est un int ; utilisez votre `erlang_c` (défi 266).

    Ce que vous apprenez : à convertir des unités avant de calculer (des appels PAR HEURE et des durées en SECONDES :
    la moitié des erreurs de dimensionnement), que « 85 % d'occupation » n'est pas un objectif de service (avec 12
    erlangs, la règle des 85 % donne 15 conseillers, et 73 % de taux de service), à chercher le plus petit n qui
    convient, et pourquoi les arrondis vers le haut se font en entiers.

    Indice : `trafic = appels_par_heure * duree_moyenne_s / 3600` ; `n = math.floor(trafic) + 1` ; `while 1 -
    erlang_c(trafic, n) * math.exp(-(n - trafic) * seuil_s / duree_moyenne_s) < objectif: n += 1`.
    """
    raise NotImplementedError


# ============================================================================ Rejouer la semaine


def simuler_file(appels, conseillers) -> list[tuple[float, bool]]:
    """Défi 268 🟡 — Rejouer les appels : [(attente, abandonné), …], un couple par appel, dans l'ordre de `appels`.

        >>> appels = [(0, 100, 50), (0, 100, 50), (10, 100, 30), (20, 100, 200)]
        >>> simuler_file(appels, 1)
        [(0.0, False), (50.0, True), (30.0, True), (80.0, False)]
        >>> simuler_file(appels, 2)
        [(0.0, False), (0.0, False), (30.0, True), (80.0, False)]
        >>> resultats = simuler_file(appels_exemple(), 14)
        >>> sum(abandon for attente, abandon in resultats), len(resultats)
        (770, 6806)

    Ici (avec 1 conseiller) : le 1er appel arrive à 0 s et est décroché tout de suite ; le conseiller est occupé
    jusqu'à 100 s. Le 2e, arrivé à 0 s aussi, devrait attendre 100 s : il raccroche au bout de sa patience, 50 s. Le
    3e (arrivé à 10 s, patience 30 s) raccroche après 30 s. Le 4e, arrivé à 20 s, est décroché à 100 s : 80 s
    d'attente, et sa patience (200 s) suffit. Les deux clients qui ont raccroché n'ont pris AUCUN conseiller : le 4e
    est décroché à 100 s, pas à 300 s. Avec 2 conseillers, les deux premiers sont décrochés tout de suite. Sur la
    semaine de l'exemple, avec 14 conseillers, 770 clients sur 6 806 ont raccroché.

    Les règles :

    - `appels` est une liste (ou un tuple) d'appels ; un appel est un tuple (ou une liste) de TROIS nombres
      (arrivée, durée, patience), en secondes : autre chose → TypeError ; une valeur négative ou NaN → ValueError ;
      l'arrivée et la durée sont finies, la patience peut valoir `math.inf` (le client ne raccroche jamais) ;
      `conseillers` est un int >= 1 (pas un bool) : TypeError ou ValueError ;
    - les appels sont servis par ordre d'ARRIVÉE (premier arrivé, premier servi), quel que soit leur ordre dans la
      liste (l'export du standard est rangé par ligne téléphonique) ; à arrivée égale, dans l'ordre de la liste ;
    - au départ, tous les conseillers sont libres. Un appel est décroché par le conseiller qui se libère le PREMIER,
      dès qu'il est libre et pas avant l'arrivée de l'appel : à l'instant max(arrivée, première libération). Un
      conseiller libéré à 100 s peut décrocher un appel arrivé à 100 s : 0 s d'attente ;
    - si l'attente dépasserait STRICTEMENT la patience, le client raccroche : le couple vaut (patience, True), et
      AUCUN conseiller n'est pris. Sinon (attente <= patience : décroché juste à temps), le couple vaut (attente,
      False), et ce conseiller est occupé jusqu'à début + durée ;
    - la réponse est une liste de tuples (attente en float, abandonné en bool), dans l'ordre de `appels` ; [] si
      aucun appel ; `appels` n'est pas modifié ;
    - le correcteur passe 30 000 appels et 400 conseillers, et compte les lignes exécutées : chercher le premier libre
      en parcourant les 400 conseillers à chaque appel est trop lent. Gardez un TAS (`heapq`) des instants où
      chaque conseiller se libère : `libres[0]` est le plus proche, `heapq.heapreplace` le remplace.

    Ce que vous apprenez : la simulation à événements discrets (on saute d'un événement à l'autre, sans avancer
    seconde par seconde), le tas comme « qui se libère en premier ? », pourquoi les formules d'Erlang (sans
    abandons) se trompent dès que les clients raccrochent, et à trier des INDICES pour rendre les résultats dans
    l'ordre d'origine.

    Indice : `ordre = sorted(range(len(appels)), key=lambda i: appels[i][0])` ; `libres = [0.0] * conseillers` ;
    `debut = max(arrivee, libres[0])` ; `resultats = [None] * len(appels)`, puis `resultats[i] = …`.
    """
    raise NotImplementedError


def indicateurs(resultats, seuil_s=SEUIL_S) -> dict:
    """Défi 269 🟡 — Les chiffres du tableau de bord : un dict de sept indicateurs, chacun avec SA définition.

        >>> indicateurs([(0, False), (30, False), (5, True), (60, True)], 20)
        {'appels': 4, 'decroches': 2, 'abandons': 2, 'taux_service': 0.3333333333333333, 'taux_abandon': 0.5, 'asa': 15.0, 'p90': 30}
        >>> ind = indicateurs(simuler_file(appels_exemple(), 14))
        >>> round(ind["taux_service"], 3), round(ind["asa"], 1), ind["p90"]
        (0.733, 14.2, 56.0)

    Ici : quatre appels. Deux décrochés, après 0 s et 30 s ; deux abandons, après 5 s et 60 s. Seul le premier est
    décroché en 20 s ou moins. Le client qui a raccroché après 5 s n'a pas laissé au centre le temps de répondre : il
    sort du taux de service. Celui qui a raccroché après 60 s, si : il compte comme un échec. Taux de service : 1 sur
    3. Le temps moyen avant décroché (l'« ASA ») vaut 15 s : la moyenne de 0 et 30, les abandons n'ont jamais été
    décrochés. Le 90e centile des attentes (« 90 % des clients décrochés ont attendu 30 s ou moins ») vaut 30.

    Les règles (le dict a EXACTEMENT ces sept clés) :

    - "appels" : le nombre de couples ; "decroches" : ceux qui ne sont pas des abandons ; "abandons" : les autres ;
    - "taux_service" : (décrochés avec attente <= seuil_s) / (décrochés + abandons avec attente > seuil_s) ; un
      abandon en `seuil_s` secondes ou moins ne compte NI au numérateur NI au dénominateur ; None si ce dénominateur
      est nul ;
    - "taux_abandon" : abandons / appels (tous les appels) ; None s'il n'y a aucun appel ;
    - "asa" (« average speed of answer ») : la moyenne des attentes des seuls DÉCROCHÉS ; None si aucun ;
    - "p90" : le 90e centile des attentes des DÉCROCHÉS, « au rang le plus proche » : rangez leurs attentes dans
      l'ordre croissant, et prenez celle de rang ⌈0,9 × k⌉ (k décrochés, rangs comptés à partir de 1 : l'indice
      ⌈0,9 × k⌉ - 1 en Python). C'est une attente VRAIE (pas une interpolation entre deux, comme le font
      `numpy.percentile` et `statistics.quantiles`) ; None si aucun décroché ;
    - `resultats` est une liste (ou un tuple) de couples (attente, abandonné), comme ceux du défi 268 : attente un
      nombre >= 0, abandonné un bool (True ou False, pas 0 ou 1) ; autre chose → TypeError, une attente négative,
      NaN ou infinie → ValueError ; `seuil_s` est un nombre >= 0 (TypeError ou ValueError).

    Ce que vous apprenez : qu'un indicateur n'existe que par sa définition écrite (trois centres d'appels calculent
    trois « taux de service » différents à partir des mêmes appels : sur tous les appels, sur les seuls décrochés, ou
    sans les abandons rapides), à choisir le bon dénominateur, à dire None plutôt que 0 quand il n'y a rien à
    mesurer, et qu'un centile n'est pas une moyenne : la moyenne (14 s) cache que 10 % des clients attendent près
    d'une minute.

    Indice : séparez d'abord `attentes` (décrochés) et `abandons` (attentes des abandons) ; `rangees =
    sorted(attentes)` ; `rangees[math.ceil(0.9 * len(rangees)) - 1]`.
    """
    raise NotImplementedError


# ============================================================================ Organiser


def planifier_vacations(besoins, duree=VACATION) -> list[int]:
    """Défi 270 🔴 — Combien de vacations COMMENCENT à chaque créneau, pour couvrir `besoins` avec le moins possible.

        >>> planifier_vacations([1, 3, 3, 1], 2)
        [1, 2, 1, 0]
        >>> planifier_vacations([0, 0, 5], 2)
        [0, 5, 0]
        >>> besoins = [conseillers_necessaires(n, DUREE_MOYENNE_S) for _, n in prevision_exemple()]
        >>> debuts = planifier_vacations(besoins, VACATION)
        >>> debuts
        [9, 4, 4, 2, 0, 0, 0, 0, 0, 3, 5, 7, 6, 1, 1, 0, 11, 0, 0, 0, 0, 0, 0, 0]
        >>> sum(debuts)
        53

    Ici (1er exemple) : quatre créneaux, des vacations de deux créneaux. Une vacation commence au créneau 0 (elle
    couvre 0 et 1), deux au créneau 1 (elles couvrent 1 et 2), une au créneau 2 (2 et 3) : 1, 3, 3 et 1 conseillers
    présents, exactement les besoins, avec 4 vacations ; impossible avec 3. Au 2e exemple, les 5 vacations
    nécessaires au DERNIER créneau ne peuvent pas commencer à ce créneau (elles finiraient après la fermeture) :
    elles commencent au créneau 1. Sur la journée de l'exemple (24 demi-heures, des vacations de 4 h), 53
    vacations couvrent chaque demi-heure ; le planning actuel en a 42, mais 14 conseillers à 15 h pour un besoin de 23.

    Les règles :

    - `besoins` est une liste (ou un tuple) d'int >= 0 : le nombre de conseillers nécessaires à chaque créneau (pas
      de bool : TypeError ; un besoin négatif → ValueError) ; `duree` est un int >= 1 (TypeError ou ValueError) : le
      nombre de créneaux CONSÉCUTIFS que couvre une vacation ;
    - aucun créneau → [] ; sinon, une vacation plus longue que la journée (`duree > len(besoins)`) → ValueError ;
    - la réponse `debuts` est une liste d'int >= 0, de la longueur de `besoins` : `debuts[s]` vacations commencent
      au créneau s et couvrent les créneaux s, s + 1, …, s + duree - 1 ;
    - aucune vacation ne déborde après le dernier créneau : `debuts[s]` vaut 0 pour s > len(besoins) - duree ;
    - à chaque créneau t, les vacations en cours couvrent le besoin : la somme des `debuts[s]` pour les s tels que
      s <= t < s + duree est >= besoins[t] ;
    - le NOMBRE TOTAL de vacations, `sum(debuts)`, est le plus petit possible. Plusieurs plannings peuvent y arriver :
      le correcteur vérifie ces propriétés, pas UN planning ;
    - la méthode : de gauche à droite, tenez à jour le nombre de présents à chaque créneau. Au créneau t, s'il en
      manque, faites commencer TOUTES les vacations manquantes le plus TARD possible tout en couvrant t : au créneau
      t, ou à `len(besoins) - duree` si elles déborderaient. Plus tard elles commencent, plus elles couvrent de
      créneaux À VENIR : c'est ce qui rend ce glouton optimal ;
    - le correcteur passe une semaine de 336 demi-heures avec des besoins de plusieurs milliers (un très grand
      centre), et compte les lignes exécutées : ajoutez les vacations manquantes d'un coup (`+= manque`), pas une
      par une dans une boucle.

    Ce que vous apprenez : un algorithme glouton et l'argument qui prouve qu'il est optimal (l'« échange » : toute
    vacation qui couvre t peut être décalée au plus tard sans rien perdre), le problème de couverture qu'on
    résoudrait sinon en programmation linéaire, et à vérifier une réponse par ses PROPRIÉTÉS quand il y en a plusieurs
    bonnes.

    Indice : `presents = [0] * len(besoins)` ; `manque = besoins[t] - presents[t]` ; `debut = min(t, len(besoins) -
    duree)` ; `for u in range(debut, debut + duree): presents[u] += manque`.
    """
    raise NotImplementedError


class FilePrioritaire:
    """Défi 271 🔴 — Les clients à risque d'abord, mais personne n'attend indéfiniment.

        >>> file = FilePrioritaire(vieillissement=6)          # 6 points gagnés par minute d'attente
        >>> file.ajouter("C1", 20, 0)                          # risque 20 (sur 100), en ligne depuis 0 s
        >>> file.ajouter("C2", 85, 60)                         # très à risque, appelle une minute plus tard
        >>> file.ajouter("C3", 30, 90)
        >>> file.suivant(120)                                  # à 120 s : C1 vaut 32, C2 91, C3 33
        'C2'
        >>> file.ajouter("C4", 55, 600)
        >>> file.retirer("C3")                                 # C3 a raccroché
        >>> len(file), file.suivant(600)                       # à 600 s : C1 vaut 20 + 60 = 80, C4 vaut 55
        (2, 'C1')
        >>> file.suivant(610), len(file)
        ('C4', 0)

    Ici : le modèle de churn donne à chaque client un risque de 0 à 100. La file fait passer les plus à risque
    d'abord, mais chaque minute d'attente rapporte `vieillissement` points : à 120 s, C1 (risque 20, deux minutes
    d'attente) vaut 20 + 6 × 2 = 32, C2 vaut 85 + 6 × 1 = 91 et C3 vaut 30 + 6 × 0,5 = 33 : C2 passe. À 600 s, C1
    attend depuis 10 minutes : il vaut 80 et passe devant C4 (55, qui vient d'appeler). Sans vieillissement, un
    client peu risqué attendrait tant que des clients à risque appellent : c'est la « famine » des ordonnanceurs.

    Ce qu'il faut écrire :

    - `__init__(vieillissement=0)` : une file vide ; `vieillissement` est le nombre de points gagnés par MINUTE
      d'attente (un nombre >= 0, fini) ;
    - `ajouter(client, risque, arrivee)` : le client (un identifiant : un texte, un nombre…) entre dans la file à
      l'instant `arrivee` (en secondes), avec son risque (un nombre >= 0, fini) ;
    - `retirer(client)` : le client a raccroché, il quitte la file ;
    - `suivant(maintenant)` : renvoie le client à servir à l'instant `maintenant` (en secondes), et le retire de la
      file ;
    - `len(file)` (`__len__`) : le nombre de clients en ligne.

    Les règles :

    - à l'instant t, la priorité d'un client vaut risque + vieillissement × (t - arrivee) / 60. `suivant` sert le
      client de plus GRANDE priorité ; à priorité égale, celui qui est entré dans la file le premier ;
    - l'horloge ne recule pas : chaque `arrivee` et chaque `maintenant` doit être >= la plus grande des valeurs
      déjà reçues (par `ajouter` ou par `suivant`), sinon ValueError ;
    - erreurs : un `vieillissement`, un `risque`, une `arrivee` ou un `maintenant` qui n'est pas un nombre (ou un
      bool) → TypeError ; négatif, NaN ou infini → ValueError ; `ajouter` un client déjà en ligne → ValueError ;
      `retirer` un client qui n'est pas en ligne → KeyError ; `suivant` sur une file vide → IndexError (vérifiez
      `maintenant` d'abord). Une opération refusée ne change rien à la file ;
    - un client retiré (ou servi) peut rappeler : `ajouter` le remet en ligne, avec son nouveau risque et sa nouvelle
      arrivée. Son ancienne place ne doit ni revenir ni l'empêcher d'être servi ;
    - le correcteur fait près de 40 000 opérations (20 000 clients, jusqu'à 5 000 en ligne en même temps), et compte
      les lignes exécutées : recalculer toutes les priorités à chaque `suivant` est trop lent. Or tous les clients gagnent les MÊMES points
      par minute : l'ORDRE entre deux clients ne change jamais avec le temps. Rangez-les UNE fois dans un tas
      (`heapq`), avec la clé `-(risque - vieillissement × arrivee / 60)` (négative : `heapq` sort le plus PETIT) ;
    - deux files ne partagent rien.

    Ce que vous apprenez : la file de priorité (le tas), le « vieillissement » qui évite la famine (l'ordonnanceur de
    Linux, les files de Kubernetes et de Celery le font), à repérer ce qui ne change PAS dans un calcul qui semble
    dépendre du temps, à départager avec un compteur pour ne jamais comparer deux clients (`heapq` comparerait
    "C1" et 42 : TypeError), et la suppression paresseuse : on MARQUE une entrée retirée au lieu de la chercher dans
    le tas.

    Indice : `self._tas = []`, `self._entrees = {}` (client → son entrée dans le tas) et `self._compteur =
    itertools.count()` ; une entrée est une LISTE `[cle, next(self._compteur), client]` ; pour retirer, remplacez son
    dernier élément par une marque, et faites `del self._entrees[client]` ; `suivant` saute les entrées marquées.
    """

    def __init__(self, vieillissement=0) -> None:
        raise NotImplementedError

    def ajouter(self, client, risque, arrivee) -> None:
        raise NotImplementedError

    def retirer(self, client) -> None:
        raise NotImplementedError

    def suivant(self, maintenant):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

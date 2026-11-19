"""Série 44 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_44.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours une
moyenne calculée sur les seuls appels décrochés, une file moyennée sur les événements au lieu du temps, un
`math.factorial` qui déborde à 171 conseillers, ou un conseiller qui prend un appel après la fin de son service.

Fil rouge de la série : une file d'attente ne se juge ni à l'œil, ni à la moyenne qui arrange :

1. un indicateur se calcule sur TOUS les appels : ceux qui ont raccroché ont attendu, eux aussi, et ce sont les
   plus mécontents. Le tableau de bord qui ne compte que les appels décrochés se flatte ;
2. le nombre moyen de clients en attente est une moyenne dans le TEMPS (la somme des attentes divisée par la durée),
   pas la moyenne des comptes relevés à chaque événement. C'est la loi de Little : L = λ × W ;
3. Erlang C donne la probabilité d'attendre avec N conseillers. La formule avec des factorielles déborde ; la
   récurrence d'Erlang B (un nombre entre 0 et 1 à chaque étape) ne déborde jamais ;
4. le nombre de conseillers nécessaires est le plus petit N qui tient l'objectif : on part de N > trafic (en
   dessous, la file grossit sans fin) et on monte ;
5. un planning se construit heure par heure, en faisant commencer chaque vacation qui manque le plus TARD possible
   (elle couvre ainsi le plus d'heures à venir), sans déborder de la fermeture ;
6. une simulation à événements discrets suit chaque appel dans l'ordre d'arrivée, avec un TAS des conseillers
   rangés par moment de disponibilité : pas d'horloge qui avance seconde par seconde.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import heapq
import math

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


def _nombre(valeur, nom: str) -> float:
    """Un vrai nombre (int ou float, pas un bool ni un texte), fini : sinon TypeError ou ValueError."""
    if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
        raise TypeError(f"{nom} doit être un nombre, pas {type(valeur).__name__}")
    if not math.isfinite(valeur):
        raise ValueError(f"{nom} doit être un nombre fini, pas {valeur!r}")
    return valeur


def _entier(valeur, nom: str, minimum: int) -> int:
    """Un vrai int (pas un bool, ni un float), au moins `minimum`."""
    if isinstance(valeur, bool) or not isinstance(valeur, int):
        raise TypeError(f"{nom} doit être un int, pas {type(valeur).__name__}")
    if valeur < minimum:
        raise ValueError(f"{nom} doit valoir au moins {minimum}, pas {valeur}")
    return valeur


def _liste(valeur, nom: str) -> list:
    if not isinstance(valeur, (list, tuple)):
        raise TypeError(f"{nom} doit être une liste, pas {type(valeur).__name__}")
    return list(valeur)


def _champ(appel, cle: str, numero: int):
    if not isinstance(appel, dict):
        raise TypeError(f"l'appel n°{numero} doit être un dict, pas {type(appel).__name__}")
    if cle not in appel:
        raise ValueError(f"l'appel n°{numero} n'a pas de clé {cle!r}")
    return appel[cle]


def _attentes(appels) -> list[tuple[float, float, bool]]:
    """(arrivée, fin de l'attente, décroché ?) pour chaque appel du journal, vérifié.

    L'attente d'un appel décroché va de son arrivée au décroché ; celle d'un appel abandonné, de son arrivée au
    moment où le client a raccroché (sa `fin`). Un journal incohérent (décroché avant l'arrivée, fin avant le
    décroché) est refusé : un indicateur calculé dessus serait faux sans que personne ne le voie.
    """
    resultat = []
    for numero, appel in enumerate(_liste(appels, "appels")):
        arrivee = _nombre(_champ(appel, "arrivee", numero), f"l'arrivée de l'appel n°{numero}")
        decroche = _champ(appel, "decroche", numero)
        fin = _nombre(_champ(appel, "fin", numero), f"la fin de l'appel n°{numero}")
        if decroche is not None:
            decroche = _nombre(decroche, f"le décroché de l'appel n°{numero}")
            if not arrivee <= decroche <= fin:
                raise ValueError(f"l'appel n°{numero} est incohérent : arrivée {arrivee}, décroché {decroche}, "
                                 f"fin {fin} (il faut arrivée <= décroché <= fin)")
            resultat.append((arrivee, decroche, True))
        else:
            if fin < arrivee:
                raise ValueError(f"l'appel n°{numero} se termine ({fin}) avant d'arriver ({arrivee})")
            resultat.append((arrivee, fin, False))
    return resultat


# ============================================================================ Défi 266


def mesurer(appels, seuil_s=SEUIL_S) -> dict:
    """Les indicateurs d'une journée, calculés sur TOUS les appels.

    Complexité : O(n), une passe. Les pièges sont des pièges de DÉNOMINATEUR :
    - l'attente moyenne honnête compte aussi les clients qui ont raccroché (ils ont attendu jusqu'à leur `fin`) ;
      la moyenne des seuls décrochés (l'ASA des tableaux de bord) oublie justement les attentes les plus longues ;
    - le niveau de service divise par TOUS les appels : un appel abandonné n'a pas été décroché en 20 s. Diviser
      par les seuls décrochés fait passer ce lundi de 48 % à 57 % sans que personne n'ait mieux répondu ;
    - « en 20 secondes ou moins » : `<=`. Le standard enregistre des secondes entières, et 20 s pile est un cas
      fréquent.
    """
    attentes = _attentes(appels)
    seuil = _nombre(seuil_s, "seuil_s")
    if seuil < 0:
        raise ValueError(f"seuil_s doit être positif ou nul, pas {seuil_s!r}")
    if not attentes:
        raise ValueError("aucun appel : pas d'indicateur sur une journée vide (on diviserait par zéro)")
    n = len(attentes)
    decroches = [fin - arrivee for arrivee, fin, decroche in attentes if decroche]
    return {
        "appels": n,
        "decroches": len(decroches),
        "abandons": n - len(decroches),
        "taux_abandon": (n - len(decroches)) / n,
        "attente_moyenne_s": sum(fin - arrivee for arrivee, fin, _ in attentes) / n,
        "attente_decroches_s": sum(decroches) / len(decroches) if decroches else None,
        "niveau_service": sum(attente <= seuil for attente in decroches) / n,
    }


# ============================================================================ Défi 267


def file_attente(appels, debut, fin) -> tuple[float, int]:
    """(nombre moyen, nombre maximal) de clients en attente entre `debut` et `fin`.

    Complexité : O(n log n), pour trier les événements du maximum (la moyenne seule se fait en O(n)).
    - la MOYENNE dans le temps : chaque client compte pendant la part de son attente qui tombe dans la fenêtre.
      L'intégrale du nombre de clients en attente est donc la SOMME de ces durées, et la moyenne, cette somme
      divisée par la durée de la fenêtre. Relever le nombre de clients à chaque arrivée ou départ, puis faire la
      moyenne de ces relevés, donne autant de poids à une seconde de rush qu'à une heure de calme ;
    - le MAXIMUM : on balaie les événements dans l'ordre du temps. Une attente est l'intervalle [arrivée, fin de
      l'attente[ (fermé à gauche, ouvert à droite) : à la même seconde, celui qui part passe AVANT celui qui
      arrive. Trier des tuples (instant, -1) et (instant, +1) le fait tout seul, car -1 < +1. Un appel décroché
      à la seconde même de son arrivée n'a jamais attendu : son intervalle est vide, il ne compte pas.
    - la loi de Little : si la fenêtre contient toutes les attentes, moyenne × durée = somme des attentes, soit
      L = λ × W (λ = appels / durée, W = attente moyenne du défi 266). Un bon test de cohérence.
    """
    attentes = _attentes(appels)
    debut, fin = _nombre(debut, "debut"), _nombre(fin, "fin")
    if not debut < fin:
        raise ValueError(f"la fenêtre est vide : debut ({debut}) doit être avant fin ({fin})")
    total = 0
    evenements = []
    for arrivee, fin_attente, _ in attentes:
        a, b = max(arrivee, debut), min(fin_attente, fin)     # la part de l'attente dans la fenêtre
        if a < b:
            total += b - a
            evenements += [(a, +1), (b, -1)]
    evenements.sort()                                         # à instant égal, -1 (départ) avant +1 (arrivée)
    en_attente = maximum = 0
    for _, variation in evenements:
        en_attente += variation
        maximum = max(maximum, en_attente)
    return total / (fin - debut), maximum


# ============================================================================ Défi 268


def erlang_c(trafic, conseillers) -> float:
    """La probabilité qu'un appel attende, avec `trafic` erlangs et `conseillers` conseillers (Erlang C).

    Complexité : O(conseillers). La formule du manuel,
        C = (A^N / N! × N / (N - A)) / (Σ_{k<N} A^k / k! + A^N / N! × N / (N - A)),
    est juste, mais A^N et N! débordent : 171! dépasse le plus grand float, et `400.0 ** 420` lève OverflowError.
    On passe par Erlang B, la probabilité de rejet SANS file, qui suit une récurrence où chaque terme reste entre
    0 et 1 :
        B(0) = 1,  B(k) = A × B(k-1) / (k + A × B(k-1)),
    puis C = N × B / (N - A × (1 - B)). Trois lignes, et 10 000 conseillers ne font pas peur.
    - trafic >= conseillers : les appels arrivent plus vite qu'on ne peut les traiter, la file grossit sans fin ;
      tout le monde finit par attendre : 1.0 (la formule donnerait un nombre négatif ou plus grand que 1) ;
    - trafic == 0 : personne n'appelle, personne n'attend : 0.0 (la récurrence le donne d'elle-même).
    """
    a = _nombre(trafic, "trafic")
    if a < 0:
        raise ValueError(f"le trafic (en erlangs) ne peut pas être négatif : {trafic!r}")
    n = _entier(conseillers, "conseillers", 1)
    if a >= n:
        return 1.0
    b = 1.0
    for k in range(1, n + 1):
        b = a * b / (k + a * b)
    return n * b / (n - a * (1 - b))


# ============================================================================ Défi 269


def conseillers_necessaires(appels_par_heure, duree_moyenne_s, seuil_s=SEUIL_S, objectif=OBJECTIF) -> int:
    """Le plus petit nombre de conseillers qui décroche `objectif` des appels en `seuil_s` secondes ou moins.

    Complexité : O(√A) essais de O(N) chacun : N dépasse rarement le trafic A de plus de quelques √A.
    - le TRAFIC, en erlangs, est le nombre moyen de conversations en cours : A = appels par heure × durée moyenne
      en HEURES (`duree_moyenne_s / 3600`). 190 appels de 281 s, c'est 14,8 conseillers occupés en permanence ;
    - le niveau de service avec N conseillers : 1 - C(A, N) × exp(-(N - A) × seuil_s / durée moyenne). C(A, N) est
      la part des appels qui attendent ; l'exponentielle, la part de ceux-là qui attendent PLUS que le seuil ;
    - on part de N = ⌊A⌋ + 1 (le premier N > A : en dessous, la file grossit sans fin) et on monte jusqu'à tenir
      l'objectif. La règle de pouce « 85 % d'occupation » (N = ⌈A / 0,85⌉) se trompe dans les deux sens : trop de
      monde pour une grosse équipe, trop peu pour une petite.
    """
    appels = _nombre(appels_par_heure, "appels_par_heure")
    duree = _nombre(duree_moyenne_s, "duree_moyenne_s")
    seuil = _nombre(seuil_s, "seuil_s")
    cible = _nombre(objectif, "objectif")
    if appels < 0:
        raise ValueError(f"appels_par_heure ne peut pas être négatif : {appels_par_heure!r}")
    if duree <= 0:
        raise ValueError(f"duree_moyenne_s doit être > 0 : {duree_moyenne_s!r}")
    if seuil < 0:
        raise ValueError(f"seuil_s doit être positif ou nul : {seuil_s!r}")
    if not 0 < cible < 1:
        raise ValueError(f"objectif doit être strictement entre 0 et 1 (100 % est impossible) : {objectif!r}")
    trafic = appels * duree / 3600
    if trafic == 0:
        return 0
    n = math.floor(trafic) + 1
    while 1 - erlang_c(trafic, n) * math.exp(-(n - trafic) * seuil / duree) < cible:
        n += 1
    return n


# ============================================================================ Défi 270


def planning(besoins, duree) -> list[int]:
    """Combien de conseillers commencent une vacation de `duree` heures à chaque heure : le moins possible.

    Complexité : O(heures). Le raisonnement glouton :
    - on avance heure par heure. À l'heure h, les présents sont ceux qui ont commencé entre h - duree + 1 et h ;
      s'il en manque, on en fait commencer exactement autant ;
    - QUAND les faire commencer ? Le plus TARD possible : à h. Une vacation qui couvre h commence entre
      h - duree + 1 et h ; celle qui commence à h couvre les heures h … h + duree - 1, toutes encore à venir,
      alors que commencer plus tôt gaspille des heures déjà couvertes. Personne ne peut faire mieux : c'est
      optimal (l'argument « on peut toujours décaler une vacation vers la droite sans rien découvrir ») ;
    - mais une vacation ne déborde pas de la fermeture : après l'heure len(besoins) - duree, on ne peut plus
      commencer à h, et la plus tardive qui couvre h commence à len(besoins) - duree ;
    - les présents se suivent avec un tableau des DÉPARTS (`partent[s + duree] -= k`), pas en recomptant toutes
      les vacations à chaque heure.
    """
    besoins = _liste(besoins, "besoins")
    for h, besoin in enumerate(besoins):
        _entier(besoin, f"besoins[{h}]", 0)
    duree = _entier(duree, "duree", 1)
    n = len(besoins)
    if n and duree > n:
        raise ValueError(f"une vacation de {duree} h ne tient pas dans une journée de {n} h")
    debuts = [0] * n
    partent = [0] * (n + 1)
    presents = 0
    for h, besoin in enumerate(besoins):
        presents += partent[h]
        manque = besoin - presents
        if manque > 0:
            s = min(h, n - duree)                             # le plus tard possible, sans dépasser la fermeture
            debuts[s] += manque
            presents += manque
            partent[s + duree] -= manque
    return debuts


# ============================================================================ Défi 271


def _horaires_conseillers(conseillers) -> list[tuple[float, float]]:
    if isinstance(conseillers, bool):
        raise TypeError("conseillers doit être un int ou une liste d'horaires, pas un bool")
    if isinstance(conseillers, int):
        if conseillers < 0:
            raise ValueError(f"un nombre de conseillers ne peut pas être négatif : {conseillers}")
        return [(0, math.inf)] * conseillers
    resultat = []
    for numero, horaire in enumerate(_liste(conseillers, "conseillers")):
        if not isinstance(horaire, (list, tuple)) or len(horaire) != 2:
            raise TypeError(f"l'horaire n°{numero} doit être un couple (début, fin), pas {horaire!r}")
        debut = _nombre(horaire[0], f"le début de l'horaire n°{numero}")
        fin = horaire[1]
        if not (isinstance(fin, float) and fin == math.inf):
            fin = _nombre(fin, f"la fin de l'horaire n°{numero}")
        if not debut < fin:
            raise ValueError(f"l'horaire n°{numero} se termine ({fin}) avant de commencer ({debut})")
        resultat.append((debut, fin))
    return resultat


def simuler(appels, conseillers) -> list[dict]:
    """La journée telle qu'elle se serait passée : le journal du standard, appel par appel.

    Complexité : O(n log n + n log c), pour n appels et c conseillers. C'est une simulation à ÉVÉNEMENTS
    DISCRETS : on ne fait pas avancer une horloge seconde par seconde (43 200 tours pour une journée, et faux dès
    qu'une heure n'est pas entière), on saute d'un appel au suivant.
    - les appels sont servis dans l'ordre d'ARRIVÉE (premier arrivé, premier servi), quel que soit l'ordre de la
      liste ; à arrivée égale, dans l'ordre de la liste (`sorted` est stable) ;
    - un TAS contient un triplet (libre à partir de, numéro, fin de service) par conseiller. Son sommet est le
      conseiller disponible le plus tôt, et, parmi ceux déjà libres, celui qui attend depuis le plus longtemps
      (la règle des standards : « longest idle agent ») ; à égalité, le plus petit numéro ;
    - un conseiller ne prend plus d'appel à partir de la fin de son service. Si le plus tôt qu'il pourrait
      décrocher tombe à sa fin ou après, il est sorti du tas pour de bon : les appels suivants arrivent plus tard
      encore, il ne leur servira pas non plus. Un appel commencé avant la fin est mené à son terme ;
    - le client décroché attend max(arrivée, libre) - arrivée ; s'il aurait dû attendre PLUS que sa patience, il
      raccroche à arrivée + patience, et n'occupe personne : le conseiller reste dans le tas, tel quel ;
    - le résultat suit l'ordre de la liste reçue : chaque appel est une copie, complétée de "decroche" et "fin".
    """
    liste = _liste(appels, "appels")
    for numero, appel in enumerate(liste):
        _nombre(_champ(appel, "arrivee", numero), f"l'arrivée de l'appel n°{numero}")
        for cle in ("duree", "patience"):
            valeur = _nombre(_champ(appel, cle, numero), f"la clé {cle!r} de l'appel n°{numero}")
            if valeur < 0:
                raise ValueError(f"la clé {cle!r} de l'appel n°{numero} est négative : {valeur!r}")
    tas = [(debut, k, fin) for k, (debut, fin) in enumerate(_horaires_conseillers(conseillers))]
    heapq.heapify(tas)
    resultat: list[dict] = [{} for _ in liste]
    for i in sorted(range(len(liste)), key=lambda i: liste[i]["arrivee"]):
        appel = liste[i]
        arrivee = appel["arrivee"]
        while tas and max(arrivee, tas[0][0]) >= tas[0][2]:    # son service est fini : pour de bon
            heapq.heappop(tas)
        if tas and max(arrivee, tas[0][0]) - arrivee <= appel["patience"]:
            libre, k, fin_service = heapq.heappop(tas)
            decroche = max(arrivee, libre)
            heapq.heappush(tas, (decroche + appel["duree"], k, fin_service))
            resultat[i] = {**appel, "decroche": decroche, "fin": decroche + appel["duree"]}
        else:                                                  # trop long, ou plus personne : il raccroche
            resultat[i] = {**appel, "decroche": None, "fin": arrivee + appel["patience"]}
    return resultat


# ============================================================================ Tests (à la pytest)


def test_mesurer_compte_les_abandons():
    journal = [{"arrivee": 0, "decroche": 20, "fin": 100}, {"arrivee": 10, "decroche": None, "fin": 70}]
    m = mesurer(journal)
    assert m["niveau_service"] == 0.5 and m["attente_moyenne_s"] == 40 and m["attente_decroches_s"] == 20


def test_file_attente_little():
    journal = [{"arrivee": 0, "decroche": 30, "fin": 90}, {"arrivee": 10, "decroche": None, "fin": 40}]
    moyenne, maximum = file_attente(journal, 0, 60)
    assert moyenne == (30 + 30) / 60 and maximum == 2


def test_erlang_c_mm1():
    assert math.isclose(erlang_c(0.5, 1), 0.5)                 # une seule file, un seul guichet : C = A
    assert erlang_c(5, 5) == 1.0 and 0 < erlang_c(950, 1000) < 1


def test_conseillers_necessaires_croissant():
    besoins = [conseillers_necessaires(n, 300) for n in (0, 50, 100, 200)]
    assert besoins[0] == 0 and besoins == sorted(besoins)


def test_planning_fin_de_journee():
    assert planning([0, 0, 0, 2], 2) == [0, 0, 2, 0]


def test_simuler_patience_egale():
    appels = [{"arrivee": 0, "duree": 10, "patience": 0}, {"arrivee": 0, "duree": 5, "patience": 10}]
    assert [a["decroche"] for a in simuler(appels, 1)] == [0, 10]

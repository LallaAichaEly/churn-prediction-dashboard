"""Série 44 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_44.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours une
factorielle qui déborde (défi 266), des secondes mélangées avec des heures (défi 267), un client qui raccroche mais
garde quand même un conseiller (défi 268), un taux calculé sur le mauvais dénominateur (défi 269), ou une file qui
recalcule toutes les priorités à chaque appel alors que leur ORDRE ne change jamais (défi 271).

Fil rouge de la série : une file d'attente ne se dimensionne pas avec une moyenne. 79 % d'occupation « en
moyenne », c'est 18 heures de conversation par heure à 15 h pour 14 conseillers : une file qui ne peut que grossir.

1. ERLANG C donne la probabilité d'attendre ; on la calcule par récurrence (Erlang B, puis C), jamais avec des
   factorielles, qui débordent dès 171 conseillers ; et quand le trafic dépasse le nombre de conseillers, elle vaut 1 ;
2. le nombre de conseillers se cherche : le plus petit n > A qui tient l'objectif (80 % en 20 s), avec A en
   ERLANGS (appels par heure × durée en HEURES), puis on le divise par la part de présents, en entiers ;
3. la SIMULATION rejoue chaque appel : un tas des heures où chaque conseiller se libère ; un client qui
   raccroche ne prend personne ;
4. un INDICATEUR n'existe que par sa définition : qui est au numérateur, qui est au dénominateur ;
5. le PLANNING se construit de gauche à droite : chaque vacation manquante commence le plus tard possible,
   sans déborder après la fermeture ;
6. une FILE PRIORITAIRE où tout le monde vieillit au même rythme garde le même ORDRE à tout instant : la clé
   `risque - vieillissement × arrivée` ne change jamais, et un tas suffit.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import heapq
import itertools
import math

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


def _nombre(valeur, nom: str, infini: bool = False) -> float:
    """Un vrai nombre (int ou float, pas un bool ni un texte), fini (ou +inf si `infini`) : TypeError ou ValueError."""
    if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
        raise TypeError(f"{nom} doit être un nombre, pas {type(valeur).__name__}")
    if math.isnan(valeur) or (math.isinf(valeur) and not (infini and valeur > 0)):
        raise ValueError(f"{nom} doit être un nombre fini, pas {valeur!r}")
    return valeur


def _positif(valeur, nom: str, infini: bool = False) -> float:
    if _nombre(valeur, nom, infini) < 0:
        raise ValueError(f"{nom} ne peut pas être négatif ({valeur!r})")
    return valeur


def _entier(valeur, nom: str, minimum: int) -> int:
    """Un vrai int (pas un bool, pas un float, même 14.0) >= minimum : TypeError ou ValueError."""
    if isinstance(valeur, bool) or not isinstance(valeur, int):
        raise TypeError(f"{nom} doit être un entier, pas {type(valeur).__name__}")
    if valeur < minimum:
        raise ValueError(f"{nom} doit valoir au moins {minimum} ({valeur!r})")
    return valeur


# ============================================================================ Dimensionner


def erlang_c(trafic, agents) -> float:
    """Défi 266 — La probabilité qu'un appel attende, avec `agents` conseillers et `trafic` erlangs."""
    _positif(trafic, "trafic")
    _entier(agents, "agents", 1)
    if trafic >= agents:
        # Plus de conversations à tenir que de conseillers : la file grossit sans fin, tout le monde attend.
        # (La formule, elle, donnerait un nombre négatif ou plus grand que 1.)
        return 1.0
    # Erlang B par récurrence : B(0) = 1, B(k) = A·B(k-1) / (k + A·B(k-1)). Chaque B reste entre 0 et 1 :
    # aucun débordement, là où A**n / n! déborde dès n = 171 (et A**n bien avant, pour un grand trafic).
    b = 1.0
    for k in range(1, agents + 1):
        b = trafic * b / (k + trafic * b)
    # Erlang C à partir d'Erlang B : la part d'appels qui attendent (au lieu d'être rejetés).
    return agents * b / (agents - trafic * (1 - b))


def _taux_de_service(trafic: float, n: int, duree: float, seuil: float) -> float:
    """La part des appels décrochés en `seuil` secondes ou moins (modèle d'Erlang C, sans abandon)."""
    return 1 - erlang_c(trafic, n) * math.exp(-(n - trafic) * seuil / duree)


def conseillers_necessaires(appels_par_heure, duree_moyenne_s, objectif=OBJECTIF, seuil_s=SEUIL_S,
                            absents_pct: int = 0) -> int:
    """Défi 267 — Combien de conseillers PLANIFIER pour décrocher `objectif` des appels en `seuil_s` secondes."""
    _positif(appels_par_heure, "appels_par_heure")
    if _positif(duree_moyenne_s, "duree_moyenne_s") == 0:
        raise ValueError("une conversation dure plus de 0 seconde")
    if not 0 < _nombre(objectif, "objectif") < 1:
        raise ValueError(f"l'objectif est une part strictement entre 0 et 1, pas {objectif!r}")
    _positif(seuil_s, "seuil_s")
    _entier(absents_pct, "absents_pct", 0)
    if absents_pct >= 100:
        raise ValueError("avec 100 % d'absents, personne ne répond")
    if appels_par_heure == 0:
        return 0
    # Le trafic en ERLANGS : des heures de conversation par heure. La durée est en secondes : / 3600.
    trafic = appels_par_heure * duree_moyenne_s / 3600
    # Le plus petit n qui laisse la file stable (n > A), puis on monte jusqu'à tenir l'objectif.
    n = math.floor(trafic) + 1
    while _taux_de_service(trafic, n, duree_moyenne_s, seuil_s) < objectif:
        n += 1
    # n conseillers doivent être PRÉSENTS. S'il manque absents_pct %, il en faut m tels que m × (100 - absents_pct)
    # / 100 >= n : m = ⌈100 n / (100 - absents_pct)⌉, en ENTIERS. En float, 21 / (1 - 0.3) vaut
    # 30.000000000000004, et math.ceil en fait 31.
    return -(-n * 100 // (100 - absents_pct))


# ============================================================================ Rejouer la journée


def _lire_appels(appels) -> list[tuple]:
    if not isinstance(appels, (list, tuple)):
        raise TypeError("appels est une liste de (arrivée, durée, patience)")
    lus = []
    for appel in appels:
        if not isinstance(appel, (list, tuple)) or len(appel) != 3:
            raise TypeError(f"un appel est un triplet (arrivée, durée, patience), pas {appel!r}")
        arrivee, duree, patience = appel
        lus.append((_positif(arrivee, "arrivée"), _positif(duree, "durée"), _positif(patience, "patience", True)))
    return lus


def simuler_file(appels, conseillers) -> list[tuple[float, bool]]:
    """Défi 268 — [(attente, abandonné), …] pour chaque appel, dans l'ordre de `appels`."""
    lus = _lire_appels(appels)
    _entier(conseillers, "conseillers", 1)
    # Premier arrivé, premier servi : on traite les appels dans l'ordre d'ARRIVÉE (sorted est stable : à arrivée
    # égale, l'ordre de la liste). L'export du standard, lui, est rangé par ligne téléphonique.
    ordre = sorted(range(len(lus)), key=lambda i: lus[i][0])
    # Un tas des instants où chaque conseiller se libère : libres[0] est le PREMIER libre (en O(1)) ; le
    # remplacer coûte O(log n). Au départ, tous sont libres.
    libres = [0.0] * conseillers
    resultats: list = [None] * len(lus)
    for i in ordre:
        arrivee, duree, patience = lus[i]
        debut = max(arrivee, libres[0])           # un conseiller libéré à 10.0 prend l'appel arrivé à 10.0
        attente = debut - arrivee
        if attente > patience:
            # Le client raccroche au bout de `patience` secondes. Il ne prend AUCUN conseiller : le tas ne change
            # pas, et le suivant de la file sera décroché au même instant `debut`.
            resultats[i] = (float(patience), True)
        else:
            resultats[i] = (float(attente), False)    # attente == patience : décroché juste à temps
            heapq.heapreplace(libres, debut + duree)  # libre à la fin de CETTE conversation (pas arrivée + durée)
    return resultats


def indicateurs(resultats, seuil_s=SEUIL_S) -> dict:
    """Défi 269 — Les chiffres du tableau de bord, chacun avec SA définition."""
    if not isinstance(resultats, (list, tuple)):
        raise TypeError("resultats est une liste de (attente, abandonné)")
    attentes, abandons = [], []
    for r in resultats:
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            raise TypeError(f"un résultat est un couple (attente, abandonné), pas {r!r}")
        attente, abandon = r
        _positif(attente, "attente")
        if not isinstance(abandon, bool):
            raise TypeError(f"« abandonné » est True ou False, pas {abandon!r}")
        (abandons if abandon else attentes).append(attente)
    _positif(seuil_s, "seuil_s")
    # Le taux de service : décrochés en `seuil_s` secondes ou moins, sur les appels qui ont attendu ASSEZ pour
    # qu'on puisse leur répondre à temps. Un client qui raccroche après 5 s n'a pas été mal servi : il sort du
    # calcul. Un client qui raccroche après 40 s, si : il compte au dénominateur.
    a_temps = sum(1 for a in attentes if a <= seuil_s)
    juges = len(attentes) + sum(1 for a in abandons if a > seuil_s)
    rangees = sorted(attentes)
    return {
        "appels": len(attentes) + len(abandons),
        "decroches": len(attentes),
        "abandons": len(abandons),
        "taux_service": a_temps / juges if juges else None,
        "taux_abandon": len(abandons) / (len(attentes) + len(abandons)) if resultats else None,
        # Le temps moyen avant décroché (ASA) : sur les DÉCROCHÉS seulement (un abandon n'a jamais été décroché).
        "asa": sum(attentes) / len(attentes) if attentes else None,
        # Le 90e centile « au rang le plus proche » : la plus petite attente telle qu'au moins 90 % des décrochés
        # ont attendu autant ou moins. Une VRAIE attente (pas une interpolation entre deux).
        "p90": rangees[math.ceil(0.9 * len(rangees)) - 1] if rangees else None,
    }


# ============================================================================ Organiser


def planifier_vacations(besoins, duree=VACATION) -> list[int]:
    """Défi 270 — Combien de vacations commencent à chaque créneau, pour couvrir `besoins` avec le moins possible."""
    if not isinstance(besoins, (list, tuple)):
        raise TypeError("besoins est une liste d'entiers, un par créneau")
    for b in besoins:
        _entier(b, "un besoin", 0)
    _entier(duree, "duree", 1)
    total = len(besoins)
    if not besoins:
        return []
    if duree > total:
        raise ValueError(f"une vacation de {duree} créneaux ne tient pas dans une journée de {total} créneaux")
    debuts = [0] * total
    presents = [0] * total
    dernier_debut = total - duree             # une vacation ne déborde pas après la fermeture
    for t in range(total):
        manque = besoins[t] - presents[t]
        if manque > 0:
            # Toutes les vacations commencées avant t sont déjà comptées. Celles qui manquent commencent le plus
            # TARD possible tout en couvrant t : à t, ou à `dernier_debut` en fin de journée. Plus tard, elles
            # couvrent plus de créneaux À VENIR : c'est ce qui rend ce glouton optimal.
            debut = min(t, dernier_debut)
            debuts[debut] += manque           # toutes d'un coup : pas une boucle « une vacation à la fois »
            for u in range(debut, debut + duree):
                presents[u] += manque
    return debuts


class FilePrioritaire:
    """Défi 271 — Les clients à risque d'abord, mais personne n'attend indéfiniment."""

    _RETIRE = object()                        # marque d'une entrée périmée (le client a raccroché)

    def __init__(self, vieillissement=0) -> None:
        self.vieillissement = _positif(vieillissement, "vieillissement")
        self._tas: list[list] = []
        self._entrees: dict = {}              # client -> son entrée DANS le tas (pour la retirer)
        self._compteur = itertools.count()    # à priorité égale : premier entré, premier servi
        self._horloge = 0

    def _verifier_heure(self, instant, nom: str) -> None:
        """L'horloge ne recule pas. On VÉRIFIE seulement : l'horloge n'avance que si l'opération réussit."""
        _positif(instant, nom)
        if instant < self._horloge:
            raise ValueError(f"l'horloge ne recule pas : {nom} = {instant!r} < {self._horloge!r}")

    def ajouter(self, client, risque, arrivee) -> None:
        _positif(risque, "risque")
        self._verifier_heure(arrivee, "arrivee")
        if client in self._entrees:
            raise ValueError(f"{client!r} est déjà en ligne")
        # À l'instant t, la priorité vaut risque + vieillissement × (t - arrivee) / 60. Tous les clients gagnent
        # le même nombre de points par minute : l'ORDRE entre deux clients ne dépend pas de t. On range donc par
        # risque - vieillissement × arrivee / 60, une fois pour toutes (négatif : heapq sort le plus PETIT).
        cle = -(risque - self.vieillissement * arrivee / 60)
        entree = [cle, next(self._compteur), client]   # le compteur départage : on ne compare jamais deux clients
        self._entrees[client] = entree
        heapq.heappush(self._tas, entree)
        self._horloge = arrivee

    def retirer(self, client) -> None:
        # Retirer du milieu d'un tas coûte O(n) : on MARQUE l'entrée, et suivant() la sautera. On l'oublie aussi
        # dans _entrees : si le client rappelle, sa nouvelle entrée ne sera pas confondue avec l'ancienne.
        entree = self._entrees.pop(client)    # KeyError s'il n'est pas en ligne
        entree[-1] = self._RETIRE

    def suivant(self, maintenant):
        self._verifier_heure(maintenant, "maintenant")
        while self._tas:
            _, _, client = heapq.heappop(self._tas)
            if client is not self._RETIRE:   # une entrée marquée par retirer() : on la jette et on continue
                del self._entrees[client]
                self._horloge = maintenant
                return client
        raise IndexError("personne n'attend")      # et l'horloge n'a pas bougé : l'opération est refusée

    def __len__(self) -> int:
        return len(self._entrees)             # les clients en ligne, pas les entrées périmées du tas


# ============================================================================ Tests (comme dans tests/test_*.py)


def test_erlang_c():
    assert abs(erlang_c(2, 3) - 4 / 9) < 1e-12              # calculable à la main
    assert erlang_c(0, 1) == 0.0 and erlang_c(14, 14) == 1.0 and erlang_c(18, 14) == 1.0
    assert 0 < erlang_c(950, 1000) < 1                       # pas d'OverflowError


def test_conseillers_necessaires():
    assert conseillers_necessaires(120, 360) == 16
    assert conseillers_necessaires(120, 360, absents_pct=30) == 23
    assert conseillers_necessaires(0, 360) == 0
    trafic = 120 * 360 / 3600
    assert _taux_de_service(trafic, 16, 360, 20) >= 0.8 > _taux_de_service(trafic, 15, 360, 20)


def test_simuler_file():
    appels = [(0, 100, 50), (0, 100, 50), (10, 100, 30), (20, 100, 200)]
    assert simuler_file(appels, 1) == [(0.0, False), (50.0, True), (30.0, True), (80.0, False)]
    assert simuler_file(list(reversed(appels)), 1)[0] == (80.0, False)


def test_indicateurs():
    r = indicateurs([(0, False), (30, False), (5, True), (60, True)], 20)
    assert r["taux_service"] == 1 / 3 and r["asa"] == 15 and r["p90"] == 30 and r["taux_abandon"] == 0.5


def test_planifier_vacations():
    assert planifier_vacations([1, 3, 3, 1], 2) == [1, 2, 1, 0]
    assert planifier_vacations([0, 0, 5], 2) == [0, 5, 0]


def test_file_prioritaire():
    file = FilePrioritaire(vieillissement=30)                # 30 points par minute : 0,5 par seconde
    file.ajouter("A", 10, 0)
    file.ajouter("B", 40, 30)                                # B passe devant : 40 - 15 > 10 - 0
    file.ajouter("C", 0, 60)
    file.retirer("A")
    assert len(file) == 2 and file.suivant(100) == "B" and file.suivant(100) == "C" and len(file) == 0

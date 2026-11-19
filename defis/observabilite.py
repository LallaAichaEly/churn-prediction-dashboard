"""Outils communs de la série 36 : le journal, les histogrammes, le compteur, les minutes et la trace de l'API de score.

Samir est d'astreinte cette semaine sur l'API de score (`api.py`), que les conseillers interrogent chaque fois
qu'un client appelle. Mardi, ils se sont plaints : la fiche client met parfois quatre secondes à s'afficher.
Le tableau de bord de l'équipe, lui, était tout vert. Les données :

- `journal_exemple()` : une heure du journal d'accès écrit par le répartiteur de charge, une ligne par requête
  (« 2026-11-17T14:00:05.531Z GET /clients/C05201/score 200 75ms »), avec ce qu'un vrai journal contient
  aussi : des lignes vides, une trace d'erreur Python, une ligne coupée (défi 218) ;
- `histogrammes_exemple()` : la durée des requêtes de la journée de mardi, comptée par chacun des trois serveurs
  dans des SEAUX, comme le fait Prometheus (défi 219) ;
- `compteur_exemple()` : le compteur `requetes_total` du serveur api-2, lu toutes les 15 secondes pendant dix
  minutes ; le serveur a redémarré pendant ce temps, et une lecture a échoué (défi 220) ;
- `minutes_exemple()` : les 43 200 minutes de novembre, chacune avec son nombre de requêtes et d'erreurs
  (défis 221 et 222) ;
- `trace_exemple()` : la trace d'UNE requête lente de mardi, découpée en `Span` (défi 223) ;
- `SEAUX`, `OBJECTIF` et `REGLES` : les bornes des seaux, l'objectif de disponibilité et les règles d'alerte ;
- `Route`, `Journal`, `Hausse`, `Budget`, `Regle`, `Alerte` et `Span` : les résultats, des tuples nommés ;
- `generer_journal`, `generer_histogramme`, `generer_minutes` et `generer_trace` : d'autres données,
  reproductibles, pour vos propres essais ; `horloge(minute)` : « 12 nov. 14:07 » pour la minute n° 16 687 ;
- `Comptee` : une liste en lecture seule qui compte les cases lues (le correcteur s'en sert au défi 222).

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> journal = journal_exemple()
    >>> len(journal), journal[0]
    (2024, '2026-11-17T14:00:01.871Z GET /health 200 924µs')
    >>> histogrammes_exemple()["api-1"][:3]           # (borne en secondes, requêtes qui ont duré AU PLUS autant)
    [(0.005, 0), (0.01, 0), (0.025, 280)]
    >>> compteur_exemple()[:3]                        # (secondes depuis le début, valeur du compteur)
    [(0, 18240), (15, 18319), (30, 18399)]
    >>> minutes = minutes_exemple()
    >>> len(minutes), minutes[16687], horloge(16687)   # (requêtes, erreurs) de la minute n° 16 687
    (43200, (290, 20), '12 nov. 14:07')
    >>> trace_exemple()[0]
    Span(id='s1', parent=None, service='passerelle', debut=0, fin=4180)
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import NamedTuple

SEAUX = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, math.inf)  # en secondes (ceux de Prometheus)
OBJECTIF = 0.999   # 99.9 % des requêtes du mois doivent réussir : c'est l'engagement pris auprès des conseillers


class Route(NamedTuple):
    """Ce que le journal dit d'une route (défi 218). Un tuple : (requetes, erreurs, durees_ms)."""

    requetes: int                 # nombre de requêtes lisibles sur cette route
    erreurs: int                  # parmi elles, celles dont le statut est 5xx (la faute du SERVEUR)
    durees_ms: list[float]        # la durée de chacune, en millisecondes, dans l'ordre du journal


class Journal(NamedTuple):
    """Le résumé d'un journal d'accès (défi 218). Un tuple : (routes, illisibles)."""

    routes: dict[str, Route]      # "MÉTHODE /chemin/{id}" → Route, dans l'ordre d'apparition
    illisibles: int               # lignes non vides qu'on n'a pas pu lire


class Hausse(NamedTuple):
    """Ce qu'un compteur a compté entre sa première et sa dernière lecture (défi 220). Un tuple : (total, par_seconde)."""

    total: float                  # ce que le compteur a compté, redémarrages compris
    par_seconde: float            # total / (durée entre la première et la dernière lecture VALIDE)


class Budget(NamedTuple):
    """Le budget d'erreur d'une période (défi 221). Un tuple de six valeurs."""

    requetes: int                 # requêtes de la période
    erreurs: int                  # erreurs de la période
    disponibilite: float | None   # 1 − erreurs / requetes (None sans requête)
    erreurs_permises: float       # (1 − objectif) × requetes : ce que l'objectif autorise
    consomme: float | None        # erreurs / erreurs_permises : la part du budget déjà brûlée (peut dépasser 1)
    restant: float | None         # 1 − consomme (négatif quand l'objectif est manqué)


class Regle(NamedTuple):
    """Une règle d'alerte à deux fenêtres (défi 222). Un tuple : (nom, longue, courte, seuil)."""

    nom: str
    longue: int                   # longueur de la fenêtre longue, en minutes
    courte: int                   # longueur de la fenêtre courte, en minutes
    seuil: float                  # vitesse de combustion à partir de laquelle on alerte


class Alerte(NamedTuple):
    """Une alerte (défi 222). Un tuple : (regle, debut, fin)."""

    regle: str                    # le nom de la règle qui s'est déclenchée
    debut: int                    # première minute où elle est allumée
    fin: int                      # dernière minute où elle est allumée (incluse)


class Span(NamedTuple):
    """Un morceau de trace (défi 223) : le travail d'UN service pour UNE requête. Un tuple de cinq valeurs."""

    id: str
    parent: str | None            # l'id du span qui l'a appelé (None : c'est la racine)
    service: str
    debut: float                  # en millisecondes depuis le début de la requête
    fin: float


# Les règles du « Site Reliability Workbook » de Google (chapitre 5), pour un objectif sur 30 jours : la page
# 1 h se déclenche quand on brûle 2 % du budget du mois en une heure, la page 6 h quand on en brûle 5 % en six heures,
# les tickets à 10 % en un jour et à 10 % en trois jours.
REGLES = (
    Regle("page 1 h", 60, 5, 14.4),
    Regle("page 6 h", 360, 30, 6.0),
    Regle("ticket 1 j", 1440, 120, 3.0),
    Regle("ticket 3 j", 4320, 360, 1.0),
)


class Comptee(Sequence):
    """Une liste en LECTURE SEULE qui compte les cases lues dans `.lues` (la même qu'à la série 30).

    `len()` ne coûte rien ; `liste[i]` compte 1 ; `liste[a:b]` compte le nombre de cases du morceau ;
    parcourir avec `for`, `list()`, `zip()` ou `sum()` compte chaque case.

        >>> minutes = Comptee([(300, 1), (310, 0), (295, 2)])
        >>> sum(e for r, e in minutes), minutes.lues
        (3, 3)
        >>> minutes[1:], minutes.lues
        ([(310, 0), (295, 2)], 5)
    """

    def __init__(self, valeurs=()) -> None:
        self._valeurs = list(valeurs)
        self.lues = 0

    def __len__(self) -> int:
        return len(self._valeurs)

    def __getitem__(self, i):
        if isinstance(i, slice):
            morceau = self._valeurs[i]
            self.lues += len(morceau)
            return morceau
        valeur = self._valeurs[i]
        self.lues += 1
        return valeur

    def __iter__(self):
        for valeur in self._valeurs:
            self.lues += 1
            yield valeur

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        if len(self._valeurs) > 6:
            return f"Comptee([{', '.join(map(repr, self._valeurs[:4]))}, … ({len(self._valeurs)} valeurs)])"
        return f"Comptee({self._valeurs!r})"


# ---------------------------------------------------------------------------- le journal (défi 218)


def duree_go(ms: float) -> str:
    """Une durée écrite comme l'écrit le répartiteur de charge (programmé en Go) : 850µs, 61.2ms, 4.18s.

        >>> duree_go(0.85), duree_go(61.24), duree_go(4180)
        ('850µs', '61.2ms', '4.18s')
    """
    if ms < 1:
        return f"{round(ms * 1000)}µs"
    if ms < 1000:
        return f"{round(ms, 1):g}ms"
    return f"{round(ms / 1000, 3):g}s"


def _client(rng: random.Random) -> str:
    return f"C{rng.randint(1, 9999):05d}"


def _requete(rng: random.Random, panne: bool) -> tuple[str, str, int, float]:
    """(méthode, chemin, statut, durée en ms) d'une requête tirée au hasard."""
    tirage = rng.random()
    if tirage < 0.55:                                   # la fiche client des conseillers
        chemin = f"/clients/{_client(rng)}/score"
        if rng.random() < 0.06:
            chemin += "/"
        if rng.random() < 0.10:
            chemin += f"?modele=v{rng.randint(2, 4)}"
        duree = rng.lognormvariate(math.log(60), 0.5)
        if rng.random() < 0.04:                         # le CRM est lent
            duree = rng.uniform(1500, 6000)
        statut = 200
        if rng.random() < 0.03:
            statut = 404                                # client inconnu : la faute de l'appelant
        if rng.random() < 0.006 or (panne and rng.random() < 0.4):
            statut = 503 if panne else 500
        return "GET", chemin, statut, duree
    if tirage < 0.73:
        statut = 422 if rng.random() < 0.05 else (500 if rng.random() < 0.008 else 200)
        return "POST", "/predict", statut, rng.lognormvariate(math.log(25), 0.4)
    if tirage < 0.85:
        return "GET", "/health", 200, rng.uniform(0.3, 0.95)
    if tirage < 0.93:
        statut = 404 if rng.random() < 0.02 else (500 if rng.random() < 0.006 else 200)
        return "GET", f"/v2/clients/{rng.randint(1, 99999)}", statut, rng.lognormvariate(math.log(40), 0.5)
    if tirage < 0.98:
        chemin = f"/clients?contrat={rng.choice(['mensuel', 'annuel', 'biennal'])}&page={rng.randint(1, 40)}"
        return "GET", chemin, 200, rng.lognormvariate(math.log(150), 0.4)
    return "GET", "/clients/", 200, rng.lognormvariate(math.log(150), 0.4)


_TRACE = ["Traceback (most recent call last):",
          '  File "/app/api.py", line 112, in predict',
          "TimeoutError: le CRM n'a pas répondu en 3 s"]


def generer_journal(n: int = 2000, graine: int = 0, illisibles: float = 0.01, debut: str = "2026-11-17T14",
                    panne: tuple[int, int] | None = None) -> list[str]:
    """`n` requêtes du journal d'accès, reproductibles, avec quelques lignes vides ou illisibles en plus.

    Une requête sur cent environ (`illisibles`) est suivie d'une ligne vide, d'une trace d'erreur Python (3 lignes)
    ou remplacée par une ligne coupée. Entre les requêtes n° panne[0] et panne[1], le CRM est en panne : 40 % des
    fiches client répondent 503.

        >>> generer_journal(3, graine=1)
        ['2026-11-17T14:06:50.227Z GET /health 200 796µs',
         '2026-11-17T14:26:40.367Z GET /clients/C06220/score?modele=v2 200 104.3ms',
         '2026-11-17T14:57:32.312Z POST /predict 200 34.7ms']
    """
    rng = random.Random(graine)
    lignes: list[str] = []
    secondes = 0.0
    for i in range(n):
        secondes = min(secondes + rng.uniform(0.1, 1.9) * 3600 / n, 3599.999)
        minute, seconde = divmod(secondes, 60)
        methode, chemin, statut, duree = _requete(rng, panne is not None and panne[0] <= i < panne[1])
        ligne = f"{debut}:{int(minute):02d}:{seconde:06.3f}Z {methode} {chemin} {statut} {duree_go(duree)}"
        if rng.random() < illisibles:
            accident = rng.randrange(3)
            if accident == 0:
                lignes += [ligne, rng.choice(["", "   "])]
            elif accident == 1:
                lignes += [ligne] + _TRACE
            else:
                lignes.append(ligne[: rng.randint(12, len(ligne) - 1)])
        else:
            lignes.append(ligne)
    return lignes


def journal_exemple() -> list[str]:
    """Une heure du journal de l'API de score, mardi 17 novembre de 14 h à 15 h (2 000 requêtes, 2 024 lignes).

    Vers 14 h 37, le CRM est tombé quelques minutes : la fiche client a répondu 503.
    """
    return generer_journal(2000, graine=17, illisibles=0.012, panne=(1220, 1300))


# ---------------------------------------------------------------------------- les histogrammes (défi 219)


def generer_histogramme(durees_s, seaux=SEAUX) -> list[tuple[float, int]]:
    """L'histogramme CUMULÉ de durées en secondes : [(borne, nombre de durées <= borne), …], comme Prometheus.

        >>> generer_histogramme([0.03, 0.07, 0.07, 0.4, 3.0], seaux=(0.05, 0.1, 0.5, math.inf))
        [(0.05, 1), (0.1, 3), (0.5, 4), (inf, 5)]
    """
    durees = sorted(durees_s)
    resultat, i = [], 0
    for borne in seaux:
        while i < len(durees) and durees[i] <= borne:
            i += 1
        resultat.append((borne, i))
    return resultat


def histogrammes_exemple() -> dict[str, list[tuple[float, int]]]:
    """La journée de mardi sur les trois serveurs de l'API : 12 000 requêtes chacun.

    api-1 et api-2 répondent en 60 ms environ. api-3 tourne sur la machine dont le disque fatigue : 15 % de ses
    requêtes durent entre 2 et 4.6 secondes.
    """
    rng = random.Random(36)
    histogrammes = {}
    for serveur in ("api-1", "api-2", "api-3"):
        durees = []
        for _ in range(12_000):
            if serveur == "api-3" and rng.random() < 0.15:
                durees.append(rng.uniform(2.0, 4.6))
            else:
                durees.append(rng.lognormvariate(math.log(0.06), 0.45))
        histogrammes[serveur] = generer_histogramme(durees)
    return histogrammes


# ---------------------------------------------------------------------------- le compteur (défi 220)


def compteur_exemple() -> list[tuple[int, float | None]]:
    """Le compteur `requetes_total` du serveur api-2, lu toutes les 15 secondes pendant 10 minutes (41 lectures).

    Le serveur a redémarré entre la lecture de 6 min (t = 360) et celle de 6 min 15 : son compteur est reparti
    de zéro. La lecture de 3 min 15 (t = 195) a échoué : elle vaut None.
    """
    rng = random.Random(220)
    valeurs: list[tuple[int, float | None]] = []
    valeur = 18_240
    for i in range(41):
        t = 15 * i
        if i > 0:
            valeur += rng.randint(68, 82)
        if t == 375:
            valeur = 41                                 # redémarré quelques secondes après t = 360
        valeurs.append((t, None if t == 195 else valeur))
    return valeurs


# ---------------------------------------------------------------------------- les minutes (défis 221 et 222)


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 40:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    seuil, k, p = math.exp(-lam), 0, rng.random()
    while p > seuil:
        k += 1
        p *= rng.random()
    return k


def _trafic(minute: int) -> float:
    """Le nombre moyen de requêtes par minute : 300 en journée de semaine, 10 la nuit, moins le week-end."""
    jour, reste = divmod(minute, 1440)
    heure = reste / 60
    base = 300.0 if 8 <= heure < 20 else (80.0 if 7 <= heure < 8 or 20 <= heure < 22 else 10.0)
    return base * (0.4 if jour % 7 in (0, 6) else 1.0)   # le 1er novembre 2026 est un dimanche


def generer_minutes(n: int = 43_200, graine: int = 0, taux: float = 0.0003,
                    incidents: tuple[tuple[int, int, float], ...] = ()) -> list[tuple[int, int]]:
    """`n` minutes (requêtes, erreurs), reproductibles. Chaque incident (debut, fin, taux) vaut pour les minutes
    debut à fin − 1 : leur taux d'erreur remplace `taux`.

        >>> generer_minutes(3, graine=1, incidents=((1, 2, 0.5),))
        [(4, 0), (3, 3), (5, 0)]
    """
    rng = random.Random(graine)
    minutes = []
    for m in range(n):
        requetes = _poisson(rng, _trafic(m))
        p = taux
        for debut, fin, p_incident in incidents:
            if debut <= m < fin:
                p = p_incident
        minutes.append((requetes, min(requetes, _poisson(rng, requetes * p))))
    return minutes


def _minute(jour: int, heure: int, mn: int) -> int:
    return (jour - 1) * 1440 + heure * 60 + mn


INCIDENTS = (
    (_minute(12, 14, 2), _minute(12, 14, 42), 0.08),    # une mise en production ratée : 8 % d'erreurs pendant 40 min
    (_minute(18, 10, 31), _minute(18, 10, 33), 0.10),   # deux minutes de hoquet
    (_minute(24, 8, 0), _minute(25, 4, 0), 0.005),      # un nouveau modèle qui plante sur les anciennetés manquantes
)


def minutes_exemple() -> list[tuple[int, int]]:
    """Les 43 200 minutes de novembre (du 1er à 0 h 00 au 30 à 23 h 59) : (requêtes, erreurs 5xx) de chacune.

    Trois incidents : le 12 de 14 h 02 à 14 h 41, une mise en production ratée (8 % d'erreurs) ; le 18 à 10 h 31,
    deux minutes de hoquet (10 %) ; du 24 à 8 h 00 au 25 à 4 h 00, un nouveau modèle qui plante pour les clients
    sans ancienneté (0.5 % d'erreurs pendant 20 heures). Le reste du temps, 0.03 % d'erreurs.
    """
    return generer_minutes(43_200, graine=2026, incidents=INCIDENTS)


def horloge(minute: int) -> str:
    """La minute n° `minute` de novembre, lisible.

        >>> horloge(0), horloge(16687), horloge(43199)
        ('1 nov. 00:00', '12 nov. 14:07', '30 nov. 23:59')
    """
    jour, reste = divmod(minute, 1440)
    return f"{jour + 1} nov. {reste // 60:02d}:{reste % 60:02d}"


# ---------------------------------------------------------------------------- la trace (défi 223)


def trace_exemple() -> list[Span]:
    """La trace d'une fiche client qui a mis 4.18 s à s'afficher, mardi à 14 h 21 (onze spans).

    La passerelle reçoit la requête et la passe à l'API. L'API lit le cache, interroge le CRM TROIS fois EN
    PARALLÈLE (fiche, contrats, factures), calcule le score, l'écrit en base, puis lance l'écriture du journal
    d'audit sans l'attendre : l'audit finit APRÈS la réponse. Le consommateur de la file d'audit a perdu
    l'identifiant de son parent en route : son span est orphelin.
    """
    return [
        Span("s1", None, "passerelle", 0, 4180),
        Span("s2", "s1", "api", 12, 4170),
        Span("s3", "s2", "cache", 15, 17),
        Span("s4", "s2", "crm", 20, 3610),
        Span("s5", "s2", "crm", 20, 3480),
        Span("s6", "s2", "crm", 22, 3790),
        Span("s7", "s2", "modele", 3800, 3815),
        Span("s8", "s2", "base", 3820, 3900),
        Span("s9", "s2", "audit", 3905, 4420),
        Span("s10", "s99", "file", 4000, 4300),
        Span("s11", "s8", "base", 3830, 3890),
    ]


def generer_trace(n: int = 200, graine: int = 0, paralleles: float = 0.5, debordements: float = 0.1,
                  orphelins: float = 0.05, entiers: bool = True) -> list[Span]:
    """Une trace de `n` spans, reproductible : des enfants souvent en parallèle, parfois qui débordent de leur
    parent, parfois orphelins (leur parent n'est pas dans la trace). Les spans sont mélangés : un enfant peut
    venir AVANT son parent.

        >>> [tuple(s) for s in generer_trace(3, graine=4)]
        [('t1', None, 'passerelle', 0, 1000), ('t3', 't1', 'crm', 277, 398), ('t2', 't1', 'base', 479, 707)]
    """
    rng = random.Random(graine)
    services = ["api", "crm", "base", "cache", "modele", "audit", "file"]
    spans = [Span("t1", None, "passerelle", 0, 1000)] if n else []
    for i in range(2, n + 1):
        parent = rng.choice(spans)
        longueur = parent.fin - parent.debut
        if rng.random() < orphelins:
            id_parent = f"perdu{rng.randint(1, 50)}"
        else:
            id_parent = parent.id
        if rng.random() < paralleles or longueur == 0:
            debut = parent.debut + rng.uniform(0, longueur * 0.3)
        else:
            debut = parent.debut + rng.uniform(0, longueur)
        fin = debut + rng.uniform(0, longueur * 0.7)
        if rng.random() < debordements:
            fin += rng.uniform(0, 300)
        if entiers:
            debut, fin = round(debut), round(fin)
            fin = max(fin, debut)
        spans.append(Span(f"t{i}", id_parent, rng.choice(services), debut, fin))
    if len(spans) > 2:
        reste = spans[1:]
        rng.shuffle(reste)
        spans[1:] = reste
    return spans

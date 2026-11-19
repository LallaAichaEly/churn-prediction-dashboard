"""Série 36 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_36.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours une
moyenne de rapports au lieu d'un rapport de sommes, une moyenne de p99, ou des appels en parallèle dont on a
additionné les durées.

Fil rouge de la série : un tableau de bord tout vert peut mentir. Chaque chiffre de surveillance se calcule
d'une façon précise, et presque toutes les autres façons donnent un nombre plausible et faux.

1. on compte par ROUTE, pas par URL (sinon autant de lignes que de clients), et une erreur est la faute du
   SERVEUR (5xx) : un 404 pour un client inconnu n'en est pas une ;
2. un quantile ne se moyenne pas : on additionne les HISTOGRAMMES des serveurs, puis on cherche le quantile ;
3. un compteur ne fait que monter, sauf quand le serveur redémarre : une baisse est un redémarrage, pas un
   nombre négatif de requêtes, et une lecture manquée n'est pas un zéro ;
4. la disponibilité, c'est le total des erreurs sur le total des requêtes : un rapport de SOMMES, pas la
   moyenne des minutes ;
5. on réveille quelqu'un quand le budget brûle vite sur une fenêtre LONGUE (c'est grave) ET sur une fenêtre
   COURTE (c'est encore en cours) ;
6. le temps propre d'un span, c'est sa durée moins la RÉUNION des intervalles de ses enfants : des appels en
   parallèle ne s'additionnent pas.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import math
import numbers

from defis.observabilite import (  # noqa: F401
    OBJECTIF,
    REGLES,
    SEAUX,
    Alerte,
    Budget,
    Comptee,
    Hausse,
    Journal,
    Regle,
    Route,
    Span,
    compteur_exemple,
    generer_histogramme,
    generer_journal,
    generer_minutes,
    generer_trace,
    histogrammes_exemple,
    horloge,
    journal_exemple,
    minutes_exemple,
    trace_exemple,
)


def _nombre(x) -> bool:
    """Un nombre réel, pas un bool (True est un int en Python)."""
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _objectif(objectif) -> float:
    if not _nombre(objectif) or not 0 < objectif < 1:
        raise ValueError(f"objectif = {objectif!r} : une fraction strictement entre 0 et 1 (0.999, pas 99.9) est attendue")
    return objectif


# ============================================================================ Défi 218 : lire le journal

# L'ordre compte : « 61.2ms » finit aussi par « s ». On essaie donc « µs » et « ms » AVANT « s ».
_UNITES = ("µs", "ms", "s")
_CHIFFRES = set("0123456789")
_MAJUSCULES = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _duree_ms(texte: str) -> float | None:
    for unite in _UNITES:
        if texte.endswith(unite):
            try:
                valeur = float(texte[: -len(unite)])
            except ValueError:
                return None
            # float() accepte "nan" et "inf" : une durée doit être un vrai nombre, positif ou nul.
            if not math.isfinite(valeur) or valeur < 0:
                return None
            if unite == "µs":
                return valeur / 1000          # 817 / 1000 donne 0.817 ; 817 × 0.001 donne 0.8170000000000001
            return valeur * 1000 if unite == "s" else valeur
    return None


def _identifiant(morceau: str) -> bool:
    """Un numéro (48213) ou un identifiant client (C04182) ; « v2 » n'en est pas un."""
    if morceau and set(morceau) <= _CHIFFRES:
        return True
    return len(morceau) >= 4 and morceau[0] in _MAJUSCULES and set(morceau[1:]) <= _CHIFFRES


def _route(methode: str, chemin: str) -> str:
    chemin = chemin.split("?", 1)[0]                        # les paramètres ne changent pas la route
    chemin = "/".join("{id}" if _identifiant(m) else m for m in chemin.split("/"))
    chemin = chemin.rstrip("/") or "/"                     # /clients/ et /clients sont la même route
    return f"{methode} {chemin}"


def _statut(texte: str) -> int | None:
    if len(texte) == 3 and set(texte) <= _CHIFFRES and 100 <= int(texte) <= 599:
        return int(texte)
    return None


def lire_journal(lignes: list[str]) -> Journal:
    """Défi 218 : une passe sur les lignes ; une ligne illisible se compte, elle n'arrête rien."""
    routes: dict[str, list] = {}
    illisibles = 0
    for ligne in lignes:
        champs = ligne.split()
        if not champs:                                      # une ligne vide n'est pas une requête ratée
            continue
        if len(champs) != 5 or not champs[2].startswith("/"):
            illisibles += 1
            continue
        _, methode, chemin, texte_statut, texte_duree = champs
        statut, duree = _statut(texte_statut), _duree_ms(texte_duree)
        if statut is None or duree is None:
            illisibles += 1
            continue
        compteurs = routes.setdefault(_route(methode, chemin), [0, 0, []])
        compteurs[0] += 1
        compteurs[1] += statut >= 500                       # 4xx : la faute de l'appelant, pas du service
        compteurs[2].append(duree)
    return Journal({route: Route(*c) for route, c in routes.items()}, illisibles)


# ============================================================================ Défi 219 : le quantile d'un histogramme


def _bornes(histogramme) -> list:
    return [borne for borne, _ in histogramme]


def quantile_histogramme(histogrammes: list[list[tuple[float, float]]], q: float) -> float | None:
    """Défi 219 : on additionne les histogrammes seau par seau, PUIS on cherche le quantile (comme Prometheus)."""
    if not _nombre(q) or not 0 < q <= 1:
        raise ValueError(f"q = {q!r} : un nombre dans ]0, 1] est attendu (0.99 pour le p99)")
    if not histogrammes:
        return None
    bornes = _bornes(histogrammes[0])
    if len(bornes) < 2 or bornes[-1] != math.inf or bornes[0] <= 0:
        raise ValueError("les bornes doivent être positives, avec au moins une borne finie, et finir par math.inf")
    if any(b >= suivante for b, suivante in zip(bornes, bornes[1:])):
        raise ValueError("les bornes doivent être strictement croissantes")
    cumuls = [0] * len(bornes)
    for n, histogramme in enumerate(histogrammes):
        if _bornes(histogramme) != bornes:
            raise ValueError(f"l'histogramme n°{n} n'a pas les mêmes seaux que le premier : on ne peut pas les additionner")
        precedent = 0
        for i, (_, compte) in enumerate(histogramme):
            if not _nombre(compte) or compte != compte or compte < precedent:
                raise ValueError(f"histogramme n°{n} : les comptes doivent être des nombres CUMULÉS (jamais en baisse)")
            precedent = compte
            cumuls[i] += compte
    total = cumuls[-1]
    if total == 0:
        return None
    rang = q * total
    for i in range(len(bornes) - 1):
        if cumuls[i] >= rang:
            bas, avant = (bornes[i - 1], cumuls[i - 1]) if i else (0.0, 0)
            # On suppose les durées réparties uniformément dans le seau : interpolation linéaire.
            return bas + (bornes[i] - bas) * (rang - avant) / (cumuls[i] - avant)
    return bornes[-2]                                       # dans le seau +Inf : on ne sait rien de plus


# ============================================================================ Défi 220 : un compteur qui redémarre


def augmentation(echantillons: list[tuple[float, float | None]]) -> Hausse | None:
    """Défi 220 : une baisse est un redémarrage (le compteur est reparti de 0) ; une lecture manquée est ignorée."""
    valides = []
    t_precedent = None
    for t, valeur in echantillons:
        if not _nombre(t) or (t_precedent is not None and t <= t_precedent):
            raise ValueError(f"les instants doivent être des nombres strictement croissants (t = {t!r})")
        t_precedent = t
        if valeur is None or valeur != valeur:              # lecture manquée : ni zéro, ni redémarrage
            continue
        if not _nombre(valeur) or valeur < 0:
            raise ValueError(f"t = {t} : un compteur vaut un nombre positif ou nul, pas {valeur!r}")
        valides.append((t, valeur))
    if len(valides) < 2:
        return None
    total = 0
    for (_, avant), (_, apres) in zip(valides, valides[1:]):
        total += apres - avant if apres >= avant else apres
    return Hausse(total, total / (valides[-1][0] - valides[0][0]))


# ============================================================================ Défi 221 : le budget d'erreur


def _minute(i: int, minute) -> tuple[int, int]:
    requetes, erreurs = minute
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (requetes, erreurs)) or \
            not 0 <= erreurs <= requetes:
        raise ValueError(f"minute n°{i} : {minute!r}, deux entiers avec 0 <= erreurs <= requêtes attendus")
    return requetes, erreurs


def budget_erreur(minutes: list[tuple[int, int]], objectif: float = OBJECTIF) -> Budget:
    """Défi 221 : un rapport de SOMMES, jamais une moyenne de rapports."""
    objectif = _objectif(objectif)
    requetes = erreurs = 0
    for i, minute in enumerate(minutes):
        r, e = _minute(i, minute)
        requetes += r
        erreurs += e
    permises = (1 - objectif) * requetes
    if requetes == 0:
        return Budget(0, 0, None, 0.0, None, None)
    consomme = erreurs / permises
    return Budget(requetes, erreurs, 1 - erreurs / requetes, permises, consomme, 1 - consomme)


# ============================================================================ Défi 222 : les alertes


def _regles(regles) -> list[Regle]:
    propres = []
    for regle in regles:
        nom, longue, courte, seuil = regle
        if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in (longue, courte)) \
                or courte > longue or not _nombre(seuil) or not seuil > 0:
            raise ValueError(f"règle {regle!r} : fenêtres entières >= 1, courte <= longue, seuil > 0 attendus")
        propres.append(Regle(nom, longue, courte, seuil))
    return propres


def alertes(minutes: list[tuple[int, int]], objectif: float = OBJECTIF, regles=REGLES) -> list[Alerte]:
    """Défi 222 : des sommes cumulées (une seule lecture), puis chaque fenêtre en deux soustractions."""
    permis = 1 - _objectif(objectif)
    regles = _regles(regles)
    # cumul_r[t] = requêtes des minutes 0 à t − 1. Les requêtes des minutes a à t (incluses) :
    # cumul_r[t + 1] − cumul_r[a]. Une seule lecture de chaque minute, quelle que soit la taille des fenêtres.
    cumul_r, cumul_e = [0], [0]
    for i, minute in enumerate(minutes):
        r, e = _minute(i, minute)
        cumul_r.append(cumul_r[-1] + r)
        cumul_e.append(cumul_e[-1] + e)
    n = len(cumul_r) - 1

    def combustion(t: int, fenetre: int) -> float:
        a = max(0, t - fenetre + 1)                        # au début, la fenêtre prend les minutes qui existent
        requetes = cumul_r[t + 1] - cumul_r[a]
        if requetes == 0:
            return 0.0                                      # sans requête, on ne brûle rien
        return (cumul_e[t + 1] - cumul_e[a]) / requetes / permis

    trouvees = []
    for rang, regle in enumerate(regles):
        debut = None
        for t in range(n):
            allumee = combustion(t, regle.longue) >= regle.seuil and combustion(t, regle.courte) >= regle.seuil
            if allumee and debut is None:
                debut = t
            elif not allumee and debut is not None:
                trouvees.append((debut, rang, Alerte(regle.nom, debut, t - 1)))
                debut = None
        if debut is not None:
            trouvees.append((debut, rang, Alerte(regle.nom, debut, n - 1)))
    trouvees.sort(key=lambda x: (x[0], x[1]))
    return [alerte for _, _, alerte in trouvees]


# ============================================================================ Défi 223 : le temps propre


def temps_propre(spans: list[Span]) -> dict[str, float]:
    """Défi 223 : les enfants groupés par parent en une passe, puis la réunion de leurs intervalles, triés."""
    enfants: dict = {}
    vus = set()
    for span in spans:
        if not (_nombre(span.debut) and _nombre(span.fin)) or not span.debut <= span.fin:
            raise ValueError(f"span {span.id!r} : debut <= fin attendus, pas {span.debut!r} et {span.fin!r}")
        if span.id in vus:
            raise ValueError(f"deux spans portent l'identifiant {span.id!r}")
        vus.add(span.id)
        enfants.setdefault(span.parent, []).append(span)
    resultat: dict[str, float] = {}
    for span in spans:
        # Chaque enfant, coupé à l'intervalle de son parent (un appel lancé sans attendre peut finir après lui).
        morceaux = sorted((max(e.debut, span.debut), min(e.fin, span.fin)) for e in enfants.get(span.id, ()))
        couvert, jusqu_a = 0, span.debut
        for debut, fin in morceaux:
            debut = max(debut, jusqu_a)                     # la partie déjà couverte ne compte pas deux fois
            if fin > debut:
                couvert += fin - debut
                jusqu_a = fin
        resultat[span.service] = resultat.get(span.service, 0) + (span.fin - span.debut) - couvert
    return resultat


# ============================================================================ Tests (à la façon de pytest)


def test_une_ligne_illisible_ne_bloque_rien():
    journal = lire_journal(["x GET /clients/C00001/score 200 61.2ms", "Traceback (most recent call last):", "",
                            "x GET /clients/C00002/score/?modele=v2 503 1.5s", "x GET /health 200 850µs"])
    assert journal.illisibles == 1
    assert journal.routes["GET /clients/{id}/score"] == Route(2, 1, [61.2, 1500.0])
    assert math.isclose(journal.routes["GET /health"].durees_ms[0], 0.85)


def test_un_404_n_est_pas_une_erreur_du_service():
    assert lire_journal(["x GET /v2/clients/12 404 3ms"]).routes == {"GET /v2/clients/{id}": Route(1, 0, [3.0])}


def test_le_p99_de_la_flotte_n_est_pas_la_moyenne_des_p99():
    rapide = generer_histogramme([0.05] * 99 + [0.2], seaux=(0.1, 1.0, math.inf))
    lent = generer_histogramme([0.05] * 50 + [0.9] * 50, seaux=(0.1, 1.0, math.inf))
    assert [quantile_histogramme([h], 0.99) for h in (rapide, lent)] == [0.1, 0.982]   # moyenne : 0.541
    assert math.isclose(quantile_histogramme([rapide, lent], 0.99), 0.1 + 0.9 * 49 / 51)  # 0.965


def test_un_redemarrage_n_est_pas_une_baisse():
    assert augmentation([(0, 100), (15, 160), (30, 20), (45, 50)]) == Hausse(110, 110 / 45)
    assert augmentation([(0, 100), (15, None), (30, 130)]) == Hausse(30, 1.0)
    assert augmentation([(0, 5)]) is None


def test_la_disponibilite_est_un_rapport_de_sommes():
    budget = budget_erreur([(1000, 1), (1, 1)], objectif=0.99)
    assert budget.requetes == 1001 and budget.erreurs == 2
    assert math.isclose(budget.disponibilite, 1 - 2 / 1001)
    assert math.isclose(budget.consomme, 2 / 10.01)


def test_une_alerte_demande_les_deux_fenetres():
    minutes = [(100, 0)] * 10 + [(100, 20)] * 3 + [(100, 0)] * 10
    regle = Regle("page", 6, 2, 8.0)
    assert alertes(minutes, objectif=0.99, regles=[regle]) == [Alerte("page", 12, 13)]
    assert alertes([], regles=[regle]) == []


def test_des_appels_en_parallele_ne_s_additionnent_pas():
    spans = [Span("a", None, "api", 0, 100), Span("b", "a", "crm", 10, 70), Span("c", "a", "crm", 20, 80),
             Span("d", "a", "audit", 90, 160), Span("e", "zz", "file", 0, 30)]
    assert temps_propre(spans) == {"api": 20, "crm": 120, "audit": 70, "file": 30}

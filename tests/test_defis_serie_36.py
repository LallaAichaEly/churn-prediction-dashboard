"""Série 36 (savoir si le service va bien) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 218`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (expressions régulières, NumPy, pandas, une
  fenêtre glissante avec `deque`, des tas…) ;
- le corrigé et les références du correcteur donnent les mêmes résultats que pandas (`diff`, `rolling`), NumPy
  (`cumsum`, `interp`) et une force brute (les intervalles dessinés minute par minute) ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent
  tient (la moyenne des p99 ment, une seule fenêtre d'alerte réveille pour rien ou trop longtemps).
"""

from __future__ import annotations

import copy
import doctest
import math
import re
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.checks_serie_36 as checks  # noqa: E402
import defis.corrige_serie_36 as corrige  # noqa: E402
import defis.observabilite as observabilite  # noqa: E402
from defis.checks_serie_36 import (  # noqa: E402
    SERIE,
    _combustions,
    _grande_trace,
    _ref_alertes,
    _ref_augmentation,
    _ref_budget,
    _ref_journal,
    _ref_quantile,
    _ref_temps_propre,
)
from defis.observabilite import (  # noqa: E402
    OBJECTIF,
    REGLES,
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
    generer_trace,
    histogrammes_exemple,
    horloge,
    journal_exemple,
    minutes_exemple,
    trace_exemple,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}), DEFIS[nom])


def _echec(nom, objet, *morceaux):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message
    return resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


# --------------------------------------------------------------------------- Défi 218 : lire le journal

_UNITES = (("µs", 1 / 1000), ("ms", 1), ("s", 1000))


def _journal_eleve(lignes, bug=""):
    """Une version « élève », écrite sans regarder le corrigé ; `bug` y glisse une erreur classique."""
    routes, illisibles = {}, 0
    for ligne in lignes:
        if bug == "split_espace" and not ligne.strip():
            continue
        champs = ligne.split(" ") if bug == "split_espace" else ligne.split()
        if not champs:
            illisibles += bug == "vides"
            continue
        if len(champs) != 5 or not champs[2].startswith("/"):
            illisibles += 1
            continue
        _, methode, chemin, statut, duree = champs
        if bug == "plante":
            if not 100 <= int(statut) <= 599:
                illisibles += 1
                continue
        elif not re.fullmatch(r"[0-9]{3}", statut) or not 100 <= int(statut) <= 599:
            if bug != "statut_libre" or not statut.lstrip("-").isdigit():
                illisibles += 1
                continue
        statut = int(statut)
        unites = (("s", 1000), ("ms", 1), ("µs", 1 / 1000)) if bug == "s_d_abord" else _UNITES
        if bug == "sans_micro":
            unites = _UNITES[1:]
        ms = None
        for unite, facteur in unites:
            if duree.endswith(unite):
                try:
                    valeur = float(duree[: -len(unite)])
                except ValueError:
                    break
                if (math.isfinite(valeur) or bug == "nan") and valeur >= 0 or (bug == "nan" and valeur != valeur):
                    ms = valeur if bug == "unite" else (valeur / 1000 if unite == "µs" else valeur * facteur)
                break
        if ms is None:
            illisibles += 1
            continue
        if bug != "url":
            if bug != "query":
                chemin = chemin.split("?")[0]
            morceaux = chemin.split("/")
            if bug == "chiffre":
                morceaux = ["{id}" if re.search(r"[0-9]", m) else m for m in morceaux]
            elif bug != "ids":
                morceaux = ["{id}" if re.fullmatch(r"[0-9]+|[A-Z][0-9]{3,}", m) else m for m in morceaux]
            chemin = "/".join(morceaux)
            if bug != "slash":
                chemin = chemin.rstrip("/") or "/"
        r = routes.setdefault(f"{methode} {chemin}", [0, 0, []])
        r[0] += 1
        r[1] += statut >= (400 if bug == "4xx" else 500)
        r[2].append(ms)
    if bug == "sans_illisibles":
        illisibles = 0
    if bug == "tri":
        routes = dict(sorted(routes.items()))
    if bug == "modifie":
        lignes.clear()
    return Journal({k: Route(*v) for k, v in routes.items()}, illisibles)


def test_lire_journal_juste_ecrit_autrement():
    _reussi("lire_journal", _journal_eleve)

    def avec_une_regex(lignes):
        motif = re.compile(r"\S+\s+(\S+)\s+(/\S*)\s+([1-5][0-9]{2})\s+([0-9.]+)(\u00b5s|ms|s)")
        routes, illisibles = {}, 0
        for ligne in lignes:
            if not ligne.strip():
                continue
            m = motif.fullmatch(ligne.strip())
            try:
                valeur = float(m.group(4)) if m else None
            except ValueError:
                valeur = None
            if valeur is None:
                illisibles += 1
                continue
            facteur = {"µs": 0.001, "ms": 1.0, "s": 1000.0}[m.group(5)]
            chemin = re.sub(r"(?<=/)([0-9]+|[A-Z][0-9]{3,})(?=/|$)", "{id}", m.group(2).split("?")[0])
            route = f"{m.group(1)} {chemin.rstrip('/') or '/'}"
            n, e, d = routes.get(route, (0, 0, []))
            routes[route] = (n + 1, e + (m.group(3)[0] == "5"), d + [valeur * facteur])
        return Journal({k: Route(*v) for k, v in routes.items()}, illisibles)

    _reussi("lire_journal", avec_une_regex)


@pytest.mark.parametrize("bug,morceaux", [
    ("4xx", ("seuls les statuts 5xx",)),
    ("url", ("par ROUTE, pas par URL",)),
    ("ids", ("devient « {id} »",)),
    ("chiffre", ("« v2 » n'est pas un identifiant",)),
    ("query", ("paramètres",)),
    ("slash", ("« / » finaux",)),
    ("unite", ("convertissez-la en millisecondes",)),
    ("s_d_abord", ("AVANT « s »",)),
    ("sans_micro", ("850µs",)),
    ("nan", ("math.isfinite",)),
    ("statut_libre", ("entre 100 et 599",)),
    ("vides", ("une ligne vide",)),
    ("sans_illisibles", ("comptez-la dans `illisibles`",)),
    ("split_espace", ("ligne.split()",)),
    ("plante", ("lève ValueError", "n'arrête pas la lecture des autres")),
    ("tri", ("l'ordre où chaque route APPARAÎT",)),
    ("modifie", ("a modifié les lignes",)),
])
def test_lire_journal_erreurs_classiques(bug, morceaux):
    _echec("lire_journal", lambda lignes: _journal_eleve(lignes, bug), *morceaux)


def test_la_reference_218_compte_comme_pandas():
    lignes = generer_journal(3000, graine=5, illisibles=0.05)
    champs = [ligne.split() for ligne in lignes if ligne.strip()]
    lisibles = pd.DataFrame([c for c in champs if len(c) == 5], columns=["t", "methode", "chemin", "statut", "duree"])
    extrait = lisibles["duree"].str.extract(r"^([0-9.]+)(µs|ms|s)$")
    lisibles["ms"] = extrait[0].astype(float) * extrait[1].map({"µs": 0.001, "ms": 1.0, "s": 1000.0})
    lisibles = lisibles.dropna(subset=["ms"])
    lisibles["route"] = lisibles["methode"] + " " + (
        lisibles["chemin"].str.split("?").str[0].str.replace(r"(?<=/)([0-9]+|[A-Z][0-9]{3,})(?=/|$)", "{id}", regex=True)
        .str.rstrip("/").replace("", "/"))
    attendu = lisibles.groupby("route", sort=False).agg(requetes=("ms", "size"),
                                                         erreurs=("statut", lambda s: (s.astype(int) >= 500).sum()))
    routes, illisibles = _ref_journal(lignes)
    assert illisibles == len(champs) - len(lisibles)
    assert list(routes) == list(attendu.index)
    assert [(n, e) for n, e, _ in routes.values()] == list(zip(attendu["requetes"], attendu["erreurs"]))
    assert corrige.lire_journal(lignes) == Journal({k: Route(*v) for k, v in routes.items()}, illisibles)


def test_compter_par_url_ferait_une_ligne_par_client():
    lignes = journal_exemple()
    urls = {tuple(ligne.split()[1:3]) for ligne in lignes if len(ligne.split()) == 5}
    journal = corrige.lire_journal(lignes)
    assert len(urls) > 1000 and len(journal.routes) == 5
    score = journal.routes["GET /clients/{id}/score"]
    quatre_cents = sum(1 for ligne in lignes if len(ligne.split()) == 5 and ligne.split()[3].startswith("4"))
    assert score.erreurs == 26 and quatre_cents > score.erreurs          # les 404 ne sont pas des pannes
    assert sum(score.durees_ms) / score.requetes < 300 < max(score.durees_ms) / 10   # la moyenne cache la queue


# --------------------------------------------------------------------------- Défi 219 : quantile d'histogramme


def _quantile_eleve(histogrammes, q, bug=""):
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 < q <= 1:
        raise ValueError("q")
    if not histogrammes:
        return None
    bornes = [b for b, _ in histogrammes[0]]
    if bornes[-1] != math.inf or len(bornes) < 2 or bornes[0] <= 0 or bornes != sorted(set(bornes)):
        raise ValueError("bornes")
    for h in histogrammes:
        comptes = [c for _, c in h]
        if [b for b, _ in h] != bornes or any(not isinstance(c, (int, float)) or c != c for c in comptes) \
                or any(a > b for a, b in zip([0] + comptes, comptes)):
            raise ValueError("histogramme")
    if bug in ("moyenne", "max"):
        valeurs = [v for v in (_quantile_eleve([h], q) for h in histogrammes) if v is not None]
        return (sum(valeurs) / len(valeurs) if bug == "moyenne" else max(valeurs)) if valeurs else None
    cumuls = [sum(h[i][1] for h in histogrammes) for i in range(len(bornes))]
    if bug == "non_cumule":
        cumuls = list(np.cumsum(cumuls))
    if cumuls[-1] == 0:
        return None
    rang = q * cumuls[-1]
    for i in range(len(bornes) - 1):
        if cumuls[i] >= rang:
            bas, avant = (bornes[i - 1], cumuls[i - 1]) if i else (0, 0)
            if bug == "haut":
                return bornes[i]
            if bug == "depuis_zero":
                return bornes[i] * rang / cumuls[i]
            return bas + (bornes[i] - bas) * (rang - avant) / (cumuls[i] - avant)
    return math.inf if bug == "infini" else bornes[-2]


def test_quantile_histogramme_juste_ecrit_autrement():
    _reussi("quantile_histogramme", _quantile_eleve)

    def en_numpy(histogrammes, q):
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 < q <= 1:
            raise ValueError("q")
        if not histogrammes:
            return None
        bornes = np.array([b for b, _ in histogrammes[0]], dtype=float)
        if not (np.isinf(bornes[-1]) and len(bornes) >= 2 and bornes[0] > 0 and np.all(np.diff(bornes) > 0)):
            raise ValueError("bornes")
        total = np.zeros(len(bornes))
        for h in histogrammes:
            if [b for b, _ in h] != list(bornes) or not all(isinstance(c, (int, float)) for _, c in h):
                raise ValueError("bornes différentes")
            comptes = np.array([c for _, c in h], dtype=float)
            if np.isnan(comptes).any() or (np.diff(comptes, prepend=0) < 0).any():
                raise ValueError("comptes")
            total += comptes
        if total[-1] == 0:
            return None
        rang = q * total[-1]
        i = int(np.searchsorted(total[:-1], rang, side="left"))
        if i == len(bornes) - 1:
            return float(bornes[-2])
        bas, avant = (bornes[i - 1], total[i - 1]) if i else (0.0, 0.0)
        return float(bas + (bornes[i] - bas) * (rang - avant) / (total[i] - avant))

    _reussi("quantile_histogramme", en_numpy)


@pytest.mark.parametrize("bug,morceaux", [
    ("moyenne", ("MOYENNE des quantiles",)),
    ("max", ("pire serveur",)),
    ("non_cumule", ("DÉJÀ cumulés",)),
    ("haut", ("borne haute",)),
    ("depuis_zero", ("borne du seau PRÉCÉDENT",)),
    ("infini", ("pas math.inf",)),
])
def test_quantile_histogramme_erreurs_classiques(bug, morceaux):
    _echec("quantile_histogramme", lambda h, q: _quantile_eleve(h, q, bug), *morceaux)


def test_quantile_histogramme_qui_ne_verifie_rien():
    def sans_controle(histogrammes, q):
        if not histogrammes:
            return None
        bornes = [b for b, _ in histogrammes[0]]
        cumuls = [sum(h[i][1] for h in histogrammes) for i in range(len(bornes))]
        return checks._ref_quantile_un(bornes, cumuls, q)

    _echec("quantile_histogramme", sans_controle, "devrait lever ValueError")


def test_la_reference_219_interpole_comme_numpy():
    rng = np.random.default_rng(219)
    for _ in range(200):
        seaux = sorted(rng.choice(observabilite.SEAUX[:-1], size=rng.integers(1, 8), replace=False)) + [math.inf]
        durees = rng.lognormal(np.log(rng.choice([0.03, 0.2, 1.0])), 0.8, size=rng.integers(1, 500))
        h = generer_histogramme(durees.tolist(), seaux)
        cumuls = np.array([c for _, c in h], dtype=float)
        q = float(rng.uniform(0.01, 1.0))
        rang = q * cumuls[-1]
        attendu = _ref_quantile([h], q)
        if rang > cumuls[-2]:
            assert attendu == seaux[-2]
        elif not np.isclose(cumuls, rang).any():   # hors des égalités exactes, histogram_quantile = interp de la CDF
            assert math.isclose(attendu, np.interp(rang, np.r_[0, cumuls[:-1]], np.r_[0, seaux[:-1]]), rel_tol=1e-9)


def test_la_moyenne_des_p99_ment():
    histogrammes = histogrammes_exemple()
    p99 = {s: corrige.quantile_histogramme([h], 0.99) for s, h in histogrammes.items()}
    flotte = corrige.quantile_histogramme(list(histogrammes.values()), 0.99)
    assert round(sum(p99.values()) / 3, 2) == 1.76 and round(flotte, 2) == 4.38
    assert max(p99.values()) > flotte > sum(p99.values()) / 3
    # La vérité (avec toutes les durées) est dans le même seau que l'estimation : 2.5 à 5 s.
    assert 2.5 < flotte <= 5.0


# --------------------------------------------------------------------------- Défi 220 : un compteur qui redémarre


def _augmentation_eleve(echantillons, bug=""):
    valides, precedent = [], None
    for t, v in echantillons:
        if not isinstance(t, (int, float)) or (precedent is not None and t <= precedent):
            raise ValueError("t")
        precedent = t
        if v is None or (isinstance(v, float) and math.isnan(v)):
            if bug == "none_zero":
                valides.append((t, 0))
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError("valeur")
        valides.append((t, v))
    if len(valides) < 2:
        return None
    if bug == "dernier_moins_premier":
        total = valides[-1][1] - valides[0][1]
    else:
        deltas = [b - a for (_, a), (_, b) in zip(valides, valides[1:])]
        if bug == "abs":
            total = sum(abs(d) for d in deltas)
        elif bug == "reset_zero":
            total = sum(d for d in deltas if d >= 0)
        else:
            total = sum(d if d >= 0 else b for d, (_, b) in zip(deltas, valides[1:]))
    if bug == "par_lecture":
        return Hausse(total, total / (len(valides) - 1))
    duree = (echantillons[-1][0] - echantillons[0][0]) if bug == "duree_totale" else valides[-1][0] - valides[0][0]
    return Hausse(total, total / duree)


def test_augmentation_juste_ecrite_autrement():
    _reussi("augmentation", _augmentation_eleve)

    def en_pandas(echantillons):
        t = pd.Series([t for t, _ in echantillons], dtype=float)
        if t.diff().le(0).any() or any(isinstance(v, bool) or not (v is None or isinstance(v, (int, float)))
                                       for _, v in echantillons):
            raise ValueError("lectures")
        valeurs = pd.Series([v for _, v in echantillons], dtype=float, index=t).dropna()
        if (valeurs < 0).any():
            raise ValueError("négatif")
        if len(valeurs) < 2:
            return None
        hausses = valeurs.diff().iloc[1:]
        hausses = hausses.where(hausses >= 0, valeurs.iloc[1:])
        total = hausses.sum()
        total = int(total) if all(isinstance(v, int) for _, v in echantillons if v is not None) else float(total)
        return Hausse(total, total / (valeurs.index[-1] - valeurs.index[0]))

    _reussi("augmentation", en_pandas)


@pytest.mark.parametrize("bug,morceaux", [
    ("dernier_moins_premier", ("la dernière lecture moins la première",)),
    ("reset_zero", ("Compter 0 pour cet intervalle",)),
    ("abs", ("valeur absolue",)),
    ("none_zero", ("une lecture manquée (None ou NaN) n'est pas un 0",)),
    ("par_lecture", ("PAR SECONDE",)),
    ("duree_totale", ("lecture VALIDE",)),
])
def test_augmentation_erreurs_classiques(bug, morceaux):
    _echec("augmentation", lambda e: _augmentation_eleve(e, bug), *morceaux)


def test_la_reference_220_est_celle_de_pandas():
    rng = np.random.default_rng(220)
    for _ in range(100):
        valeurs, courant = [], float(rng.integers(0, 1000))
        for _ in range(30):
            courant = float(rng.integers(0, 10)) if rng.random() < 0.1 else courant + float(rng.integers(0, 50))
            valeurs.append(courant)                                   # quelques redémarrages
        valeurs = np.array(valeurs)
        valeurs[rng.choice(30, size=2, replace=False)] = np.nan
        echantillons = [(15 * i, None if np.isnan(v) else float(v)) for i, v in enumerate(valeurs)]
        s = pd.Series(valeurs).dropna()
        d = s.diff()
        total = d.where(d >= 0, s).iloc[1:].sum()
        assert math.isclose(_ref_augmentation(echantillons)[0], total)
        assert corrige.augmentation(echantillons).total == pytest.approx(total)


def test_le_compteur_de_mardi():
    lectures = compteur_exemple()
    assert lectures[-1][1] - lectures[0][1] == -17095
    assert corrige.augmentation(lectures) == Hausse(2932, 2932 / 600)
    assert lectures[13] == (195, None) and lectures[25][1] == 41


# --------------------------------------------------------------------------- Défi 221 : budget d'erreur


def _budget_eleve(minutes, objectif=OBJECTIF, bug=""):
    if isinstance(objectif, bool) or not isinstance(objectif, (int, float)) or not 0 < objectif < 1:
        raise ValueError("objectif")
    for r, e in minutes:
        if type(r) is not int or type(e) is not int or not 0 <= e <= r:
            raise ValueError("minute")
    requetes, erreurs = sum(r for r, _ in minutes), sum(e for _, e in minutes)
    if not requetes:
        return Budget(0, 0, None, 0.0, None, None)
    if bug == "moyenne_minutes":
        taux = [e / r for r, e in minutes if r]
        dispo = 1 - sum(taux) / len(taux)
    else:
        dispo = 1 - erreurs / requetes
    permises = (objectif if bug == "objectif" else 1 - objectif) * requetes
    consomme = permises / erreurs if bug == "inverse" else erreurs / permises
    restant = permises - erreurs if bug == "restant_requetes" else 1 - consomme
    return Budget(requetes, erreurs, dispo, permises, consomme, restant)


def test_budget_erreur_juste_ecrit_autrement():
    _reussi("budget_erreur", _budget_eleve)

    def en_numpy(minutes, objectif=OBJECTIF):
        if isinstance(objectif, bool) or not isinstance(objectif, (int, float)) or not 0 < objectif < 1:
            raise ValueError("objectif")
        if any(type(x) is not int for m in minutes for x in m):
            raise ValueError("pas des int")
        tableau = np.array(minutes, dtype=np.int64).reshape(-1, 2)
        if (tableau < 0).any() or (tableau[:, 1] > tableau[:, 0]).any():
            raise ValueError("minute")
        requetes, erreurs = (int(x) for x in tableau.sum(axis=0))
        permises = (1 - objectif) * requetes
        if not requetes:
            return Budget(0, 0, None, 0.0, None, None)
        return Budget(requetes, erreurs, 1 - erreurs / requetes, permises, erreurs / permises, 1 - erreurs / permises)

    _reussi("budget_erreur", en_numpy)


@pytest.mark.parametrize("bug,morceaux", [
    ("moyenne_minutes", ("rapport de SOMMES",)),
    ("objectif", ("(1 − objectif) × requetes",)),
    ("inverse", ("pas l'inverse",)),
    ("restant_requetes", ("FRACTION du budget",)),
])
def test_budget_erreur_erreurs_classiques(bug, morceaux):
    _echec("budget_erreur", lambda m, o=OBJECTIF: _budget_eleve(m, o, bug), *morceaux)


def test_budget_erreur_sans_controle():
    def sans_controle(minutes, objectif=OBJECTIF):
        requetes, erreurs = sum(r for r, _ in minutes), sum(e for _, e in minutes)
        if not requetes:
            return Budget(0, 0, None, 0.0, None, None)
        permises = (1 - objectif) * requetes
        return Budget(requetes, erreurs, 1 - erreurs / requetes, permises, erreurs / permises, 1 - erreurs / permises)

    _echec("budget_erreur", sans_controle, "0.999, pas 99.9")


def test_le_budget_de_novembre():
    budget = corrige.budget_erreur(minutes_exemple())
    assert (budget.requetes, budget.erreurs, round(budget.erreurs_permises)) == (5794785, 3774, 5795)
    assert round(budget.consomme, 3) == 0.651 and round(budget.restant, 3) == 0.349
    temps = _ref_budget(minutes_exemple(), OBJECTIF, "moyenne_minutes")
    assert temps[2] > budget.disponibilite        # la moyenne des minutes flatte : les pannes ont eu lieu en journée


# --------------------------------------------------------------------------- Défi 222 : alertes


def _alertes_eleve(minutes, objectif=OBJECTIF, regles=REGLES, bug=""):
    """Une fenêtre glissante par fenêtre (deque), sans sommes cumulées ; `bug` y glisse une erreur classique."""
    if isinstance(objectif, bool) or not isinstance(objectif, (int, float)) or not 0 < objectif < 1:
        raise ValueError("objectif")
    for _, longue, courte, seuil in regles:
        if any(type(x) is not int or x < 1 for x in (longue, courte)) or courte > longue \
                or isinstance(seuil, bool) or not isinstance(seuil, (int, float)) or seuil <= 0:
            raise ValueError("règle")
    tailles = sorted({w for _, longue, courte, _ in regles for w in (longue, courte)})
    fenetres = {w: [deque(), 0, 0, 0.0] for w in tailles}
    allumees = {nom: None for nom, *_ in regles}
    trouvees = []
    n = 0
    for t, minute in enumerate(minutes):
        r, e = minute
        if type(r) is not int or type(e) is not int or not 0 <= e <= r:
            raise ValueError("minute")
        n = t + 1
        for w, f in fenetres.items():
            f[0].append((r, e))
            f[1] += r
            f[2] += e
            f[3] += e / r if r else 0
            taille = w + 1 if bug == "trop_large" else w
            while len(f[0]) > taille:
                vieux_r, vieux_e = f[0].popleft()
                f[1] -= vieux_r
                f[2] -= vieux_e
                f[3] -= vieux_e / vieux_r if vieux_r else 0

        def combustion(w):
            _, requetes, erreurs, ratios = fenetres[w]
            if bug == "moyenne":
                k = sum(1 for rr, _ in fenetres[w][0] if rr)
                return ratios / k / (1 - objectif) if k else 0.0
            if not requetes:
                return 0.0
            return erreurs / requetes / (1 if bug == "sans_objectif" else 1 - objectif)

        for rang, (nom, longue, courte, seuil) in enumerate(regles):
            if bug == "longue_seule":
                allumee = combustion(longue) >= seuil
            elif bug == "courte_seule":
                allumee = combustion(courte) >= seuil
            else:
                allumee = combustion(longue) >= seuil and combustion(courte) >= seuil
            if bug == "fenetres_pleines" and t < longue - 1:
                allumee = False
            if bug == "par_minute" and allumee:
                trouvees.append((t, rang, Alerte(nom, t, t)))
                continue
            if allumee and allumees[nom] is None:
                allumees[nom] = t
            elif not allumee and allumees[nom] is not None:
                trouvees.append((allumees[nom], rang, Alerte(nom, allumees[nom], t - 1)))
                allumees[nom] = None
    for rang, (nom, *_) in enumerate(regles):
        if allumees[nom] is not None:
            trouvees.append((allumees[nom], rang, Alerte(nom, allumees[nom], n - 1)))
    cle = (lambda x: (x[1], x[0])) if bug == "ordre_regles" else (lambda x: (x[0], x[1]))
    return [a for _, _, a in sorted(trouvees, key=cle)]


def test_alertes_juste_ecrites_autrement():
    _reussi("alertes", _alertes_eleve)

    def en_numpy(minutes, objectif=OBJECTIF, regles=REGLES):
        if isinstance(objectif, bool) or not isinstance(objectif, (int, float)) or not 0 < objectif < 1:
            raise ValueError("objectif")
        if any(type(x) is not int for m in minutes for x in m):
            raise ValueError("minute")
        tableau = np.array(list(minutes), dtype=np.int64).reshape(-1, 2)
        if (tableau < 0).any() or (tableau[:, 1] > tableau[:, 0]).any():
            raise ValueError("minute")
        cumul = np.vstack([[0, 0], np.cumsum(tableau, axis=0)])
        t = np.arange(len(tableau))
        resultat = []
        for rang, (nom, longue, courte, seuil) in enumerate(regles):
            if any(type(x) is not int or x < 1 for x in (longue, courte)) or courte > longue \
                    or isinstance(seuil, bool) or not isinstance(seuil, (int, float)) or not seuil > 0:
                raise ValueError("règle")
            allumee = np.ones(len(t), dtype=bool)
            for w in (longue, courte):
                a = np.maximum(0, t - w + 1)
                r, e = (cumul[t + 1] - cumul[a]).T
                with np.errstate(divide="ignore", invalid="ignore"):
                    combustion = np.where(r > 0, e / np.where(r > 0, r, 1) / (1 - objectif), 0.0)
                allumee &= combustion >= seuil
            bords = np.diff(np.r_[0, allumee.astype(int), 0])
            for debut, fin in zip(np.flatnonzero(bords == 1), np.flatnonzero(bords == -1) - 1):
                resultat.append((int(debut), rang, Alerte(nom, int(debut), int(fin))))
        return [a for _, _, a in sorted(resultat, key=lambda x: (x[0], x[1]))]

    _reussi("alertes", en_numpy)


@pytest.mark.parametrize("bug,morceaux", [
    ("longue_seule", ("Avec la seule fenêtre longue",)),
    ("courte_seule", ("Avec la seule fenêtre courte",)),
    ("moyenne", ("rapport de SOMMES",)),
    ("sans_objectif", ("divisez-le par (1 − objectif)",)),
    ("trop_large", ("pas t − W à t",)),
    ("fenetres_pleines", ("la fenêtre prend les minutes qui existent",)),
    ("par_minute", ("pas une alerte par minute",)),
    ("ordre_regles", ("triées par debut",)),
])
def test_alertes_erreurs_classiques(bug, morceaux):
    _echec("alertes", lambda m, o=OBJECTIF, r=REGLES: _alertes_eleve(m, o, r, bug), *morceaux)


def test_alertes_fenetre_decalee():
    def decalee(minutes, objectif=OBJECTIF, regles=REGLES):
        corrige.alertes([], objectif, regles)                       # mêmes contrôles
        liste = list(minutes)
        return [Alerte(*a) for a in _ref_alertes(liste, objectif, regles, "decalee")]

    _echec("alertes", decalee, "INCLUT la minute t")


def test_relire_chaque_fenetre_est_refuse():
    def naive(minutes, objectif=OBJECTIF, regles=REGLES):
        corrige.alertes([], objectif, regles)
        trouvees = []
        for rang, (nom, longue, courte, seuil) in enumerate(regles):
            debut = None
            for t in range(len(minutes) + 1):
                allumee = t < len(minutes)
                for w in (longue, courte):
                    if allumee:
                        fenetre = minutes[max(0, t - w + 1):t + 1]
                        r = sum(x for x, _ in fenetre)
                        allumee = bool(r) and sum(y for _, y in fenetre) / r / (1 - objectif) >= seuil
                if allumee and debut is None:
                    debut = t
                elif not allumee and debut is not None:
                    trouvees.append((debut, rang, Alerte(nom, debut, t - 1)))
                    debut = None
        return [a for _, _, a in sorted(trouvees, key=lambda x: (x[0], x[1]))]

    _echec("alertes", naive, "relisez chaque fenêtre à chaque minute")


def test_la_reference_222_est_celle_de_pandas_rolling():
    rng = np.random.default_rng(222)
    for _ in range(40):
        n = int(rng.integers(1, 300))
        minutes = checks._minutes_hasard(__import__("random").Random(int(rng.integers(10**6))), n)
        objectif = 0.99
        df = pd.DataFrame(minutes, columns=["r", "e"])
        for w in (1, 5, 17, 60):
            somme = df.rolling(w, min_periods=1).sum()
            attendu = np.where(somme["r"] > 0, somme["e"] / somme["r"].where(somme["r"] > 0, 1) / (1 - objectif), 0.0)
            combustion = _combustions(minutes, objectif)
            assert np.allclose([combustion(t, w) for t in range(n)], attendu)


def test_minutes_exemple_sans_egalite_douteuse():
    assert not checks._presque_a_egalite(minutes_exemple()[:4000], OBJECTIF, REGLES)
    rapide = corrige.alertes(minutes_exemple())
    lente = _ref_alertes(minutes_exemple(), OBJECTIF, REGLES)
    assert [tuple(a) for a in rapide] == lente


def test_une_seule_fenetre_reveille_pour_rien_ou_trop_longtemps():
    minutes = minutes_exemple()
    pages = [a for a in corrige.alertes(minutes) if a.regle.startswith("page")]
    assert [(a.regle, horloge(a.debut), horloge(a.fin)) for a in pages] == [
        ("page 1 h", "12 nov. 14:12", "12 nov. 14:45"), ("page 6 h", "12 nov. 14:27", "12 nov. 15:08")]
    courte = corrige.alertes(minutes, regles=[Regle("courte", 5, 5, 14.4)])
    longue = corrige.alertes(minutes, regles=[Regle("longue", 60, 60, 14.4)])
    assert len(courte) == 56
    nuit = [a for a in courte if not 8 <= (a.debut % 1440) // 60 < 20]
    assert len(nuit) >= 40
    assert any(horloge(a.debut) == "18 nov. 10:31" for a in courte)             # le hoquet réveille…
    assert not any(a.debut // 1440 == 17 for a in corrige.alertes(minutes))      # …mais pas avec deux fenêtres
    assert [(horloge(a.debut), horloge(a.fin)) for a in longue] == [("12 nov. 14:12", "12 nov. 15:30")]


# --------------------------------------------------------------------------- Défi 223 : temps propre


def _temps_eleve(spans, bug=""):
    if len({s.id for s in spans}) != len(spans):
        raise ValueError("id")
    for s in spans:
        if not all(isinstance(x, (int, float)) and x == x for x in (s.debut, s.fin)) or s.fin < s.debut:
            raise ValueError("durée")
    enfants = {}
    for s in spans:
        enfants.setdefault(s.parent, []).append(s)
    if bug == "orphelins":
        gardes, pile = [], [s for s in spans if s.parent is None]
        while pile:
            s = pile.pop()
            gardes.append(s)
            pile += enfants.get(s.id, [])
        gardes = [s for s in spans if s in gardes]
    elif bug == "quadratique":
        gardes = spans
        enfants = {s.id: [e for e in spans if e.parent == s.id] for s in spans}
    else:
        gardes = spans
    resultat = {}
    for s in gardes:
        mes_enfants = enfants.get(s.id, [])
        if bug == "total":
            attente = 0
        elif bug == "somme":
            attente = sum(e.fin - e.debut for e in mes_enfants)
        else:
            intervalles = sorted((e.debut, e.fin) if bug == "sans_coupe" else (max(e.debut, s.debut), min(e.fin, s.fin))
                                 for e in mes_enfants)
            attente, courant = 0, None
            for a, b in intervalles:
                if b <= a:
                    continue
                if courant is None or a > courant[1]:
                    if courant:
                        attente += courant[1] - courant[0]
                    courant = [a, b]
                else:
                    courant[1] = max(courant[1], b)
            if courant:
                attente += courant[1] - courant[0]
        resultat[s.service] = resultat.get(s.service, 0) + s.fin - s.debut - attente
    if bug == "tri":
        resultat = dict(sorted(resultat.items()))
    return resultat


def test_temps_propre_juste_ecrit_autrement():
    _reussi("temps_propre", _temps_eleve)


@pytest.mark.parametrize("bug,morceaux", [
    ("somme", ("RÉUNION",)),
    ("sans_coupe", ("partie COMMUNE",)),
    ("total", ("durée totale",)),
    ("orphelins", ("orphelin",)),
    ("tri", ("APPARAÎT dans `spans`",)),
    ("quadratique", ("beaucoup trop lent", "groupez-les par parent")),
])
def test_temps_propre_erreurs_classiques(bug, morceaux):
    _echec("temps_propre", lambda s: _temps_eleve(s, bug), *morceaux)


def test_temps_propre_recursif_est_refuse():
    def recursif(spans):
        for s in spans:
            if not all(isinstance(x, (int, float)) and x == x for x in (s.debut, s.fin)) or s.fin < s.debut:
                raise ValueError("durée")
        if len({s.id for s in spans}) != len(spans):
            raise ValueError("id")
        ids = {s.id for s in spans}
        enfants = {}
        for s in spans:
            enfants.setdefault(s.parent if s.parent in ids else None, []).append(s)
        resultat = {s.service: 0 for s in spans}

        def visiter(s):
            for e in enfants.get(s.id, []):
                visiter(e)
            resultat[s.service] += corrige.temps_propre([s] + [Span(e.id, s.id, "_", e.debut, e.fin)
                                                              for e in enfants.get(s.id, [])])[s.service]

        for racine in enfants.get(None, []):
            visiter(racine)
        return resultat

    _echec("temps_propre", recursif, "RecursionError")


def test_la_reference_223_est_celle_d_une_force_brute():
    """Avec des temps entiers, la réunion se dessine milliseconde par milliseconde."""
    for graine in range(60):
        spans = generer_trace(40, graine=graine, debordements=0.3, orphelins=0.1)
        attendu: dict = {}
        for s in spans:
            occupe = set()
            for e in spans:
                if e.parent == s.id:
                    occupe |= set(range(max(e.debut, s.debut), min(e.fin, s.fin)))
            attendu[s.service] = attendu.get(s.service, 0) + (s.fin - s.debut) - len(occupe)
        assert _ref_temps_propre(spans) == attendu == corrige.temps_propre(spans)


def test_la_trace_de_mardi():
    propre = corrige.temps_propre(trace_exemple())
    assert propre["passerelle"] + propre["api"] == 48 and max(propre, key=propre.get) == "crm"
    somme = _ref_temps_propre(trace_exemple(), "somme")
    assert somme["api"] == -7272                                 # additionner des appels parallèles : absurde
    assert propre["audit"] == 515 and propre["file"] == 300      # l'audit qui déborde et l'orphelin comptent


def test_la_grande_trace():
    spans = _grande_trace()
    assert len(spans) == 40_000 and len({s.id for s in spans}) == 40_000
    assert sum(1 for s in spans if s.parent == "lot") == 20_000
    propre = corrige.temps_propre(spans)
    assert propre == pytest.approx(_ref_temps_propre(spans))


# --------------------------------------------------------------------------- Les outils communs


def test_les_donnees_sont_reproductibles_et_neuves_a_chaque_appel():
    for fabrique in (journal_exemple, histogrammes_exemple, compteur_exemple, minutes_exemple, trace_exemple):
        a, b = fabrique(), fabrique()
        assert a == b and a is not b
    assert generer_journal(50, graine=3) == generer_journal(50, graine=3) != generer_journal(50, graine=4)
    assert generer_trace(30, graine=3) == generer_trace(30, graine=3)


def test_les_incidents_de_novembre_sont_la_ou_on_le_dit():
    minutes = minutes_exemple()

    def taux(debut, fin):
        return sum(e for _, e in minutes[debut:fin]) / sum(r for r, _ in minutes[debut:fin])

    debut, fin, _ = observabilite.INCIDENTS[0]
    assert horloge(debut) == "12 nov. 14:02" and horloge(fin - 1) == "12 nov. 14:41"
    assert 0.06 < taux(debut, fin) < 0.10
    assert taux(0, 10 * 1440) < 0.0006
    assert observabilite._trafic(_minute(12, 12, 0)) == 300 and observabilite._trafic(_minute(15, 12, 0)) == 120


def _minute(jour, heure, mn):
    return (jour - 1) * 1440 + heure * 60 + mn


def test_le_journal_contient_ce_qu_un_vrai_journal_contient():
    lignes = journal_exemple()
    assert any(not ligne.strip() for ligne in lignes)
    assert any(ligne.startswith("Traceback") for ligne in lignes)
    assert any("µs" in ligne for ligne in lignes) and any(re.search(r" [0-9.]+s$", ligne) for ligne in lignes)
    assert any("?" in ligne for ligne in lignes) and any("/score/ " in ligne for ligne in lignes)
    assert any(" 503 " in ligne for ligne in lignes) and any(" 404 " in ligne for ligne in lignes)


def test_comptee_compte_les_cases_lues():
    c = Comptee([(1, 0), (2, 1), (3, 0)])
    assert len(c) == 3 and c.lues == 0
    list(c)
    c[0:2]
    c[1]
    assert c.lues == 6


# --------------------------------------------------------------------------- La série


def test_les_references_rejouent_les_bonnes_erreurs():
    routes, _ = _ref_journal(["x GET /clients/C00001/score 404 5ms"], "4xx")
    assert routes["GET /clients/{id}/score"][1] == 1
    a = [(0.1, 60), (0.5, 90), (1.0, 100), (math.inf, 100)]
    b = [(0.1, 10), (0.5, 20), (1.0, 50), (math.inf, 60)]
    assert _ref_quantile([a], 0.5, "haut") == 0.1 and _ref_quantile([b], 0.99, "infini") == math.inf
    assert _ref_augmentation([(0, 100), (15, 20)], "dernier_moins_premier") == (-80, -80 / 15)
    assert _ref_budget([(1000, 1), (1, 1)], 0.999, "moyenne_minutes")[2] == pytest.approx(1 - (0.001 + 1) / 2)
    assert _ref_alertes([(100, 50)] * 3, 0.99, [Regle("p", 2, 1, 1.0)], "par_minute") == [
        ("p", 0, 0), ("p", 1, 1), ("p", 2, 2)]


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(218, 224))
    assert any(s is SERIE for s in series())            # enregistrée dans defis/verif.py


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_36 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(observabilite)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_36 import *` (dans les exemples) doit donner les fonctions du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 40 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    donnees = (journal_exemple()[:300], histogrammes_exemple(), compteur_exemple(), minutes_exemple()[:2000],
               trace_exemple())
    copies = copy.deepcopy(donnees)
    lignes, histogrammes, lectures, minutes, spans = donnees
    corrige.lire_journal(lignes)
    corrige.quantile_histogramme(list(histogrammes.values()), 0.99)
    corrige.augmentation(lectures)
    corrige.budget_erreur(minutes)
    corrige.alertes(minutes)
    corrige.temps_propre(spans)
    assert donnees == copies

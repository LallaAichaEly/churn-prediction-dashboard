"""Série 35 (cibler ceux qu'on peut convaincre) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 212`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (`defaultdict`, `Counter`, pandas, la formule
  de l'issue transformée en une ligne, une courbe de Qini en NumPy, `heapq.nsmallest`…) ;
- le corrigé et les références du correcteur donnent les mêmes résultats que pandas (`groupby`), NumPy, une
  recherche exhaustive (le plan optimal) et `PredefinedSplit` de scikit-learn (les plis) ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent
  tient (le score classe mal, le modèle jugé sur ses propres clients se flatte).
"""

from __future__ import annotations

import copy
import doctest
import heapq
import itertools
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.checks_serie_35 as checks  # noqa: E402
import defis.ciblage as ciblage  # noqa: E402
import defis.corrige_serie_35 as corrige  # noqa: E402
from defis.checks_serie_35 import (  # noqa: E402
    SERIE,
    _ref_effets,
    _ref_issue,
    _ref_plan,
    _ref_qini,
    _ref_t_learner,
)
from defis.ciblage import (  # noqa: E402
    COUT_CONTACT,
    VARIABLES,
    Effet,
    Plan,
    Qini,
    campagne_exemple,
    campagne_stratifiee_exemple,
    generer_campagne,
    prospects_exemple,
    risque_vrai,
    uplift_vrai,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet, **autres):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}, **autres), DEFIS[nom])


def _echec(nom, objet, *morceaux, **autres):
    resultat = _evaluer(nom, objet, **autres)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message
    return resultat.message


def _reussi(nom, objet, **autres):
    resultat = _evaluer(nom, objet, **autres)
    assert resultat.statut == REUSSI, resultat.message


def _zero_un(x):
    if x is True or x is False or (type(x) is int and x in (0, 1)):
        return int(x)
    raise ValueError(f"{x!r} n'est ni 0 ni 1")


def _manquant(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


# --------------------------------------------------------------------------- Défi 212 : effet_par_segment


def _effets(defaut=None):
    """Écrite autrement que le corrigé (defaultdict de Counter, deux passes), avec UN défaut."""

    def effet_par_segment(clients, variable):
        comptes = defaultdict(Counter)
        for c in clients:
            if defaut == "saisie":
                offre, parti = bool(c["offre"]), bool(c["parti"])
            else:
                offre, parti = _zero_un(c["offre"]), _zero_un(c["parti"])
            if defaut == "modifie":
                c["offre"] = int(offre)
                c["parti"] = int(parti)
            if defaut == "groupes_inverses":
                offre = not offre
            valeur = c[variable]
            if defaut == "sans_none" and valeur is None:
                continue
            groupe = "o" if offre else "t"
            comptes[valeur]["n" + groupe] += 1
            comptes[valeur]["d" + groupe] += parti
        n_t = sum(x["nt"] for x in comptes.values())
        d_t = sum(x["dt"] for x in comptes.values())
        resultat = {}
        cles = sorted(comptes, key=str) if defaut == "trie" else list(comptes)
        for valeur in cles:
            x = comptes[valeur]
            if defaut == "division":
                taux_o, taux_t = x["do"] / x["no"], x["dt"] / x["nt"]
            elif defaut == "zero":
                taux_o = x["do"] / x["no"] if x["no"] else 0.0
                taux_t = x["dt"] / x["nt"] if x["nt"] else 0.0
            elif defaut == "taux_segment":
                taux_o = x["do"] / (x["no"] + x["nt"]) if x["no"] else None
                taux_t = x["dt"] / (x["no"] + x["nt"]) if x["nt"] else None
            else:
                taux_o = x["do"] / x["no"] if x["no"] else None
                taux_t = x["dt"] / x["nt"] if x["nt"] else None
            if defaut == "temoin_global":
                taux_t = d_t / n_t if n_t else None
            if taux_o is None or taux_t is None:
                uplift = None
            elif defaut == "signe":
                uplift = taux_o - taux_t
            elif defaut == "relatif":
                uplift = (taux_t - taux_o) / taux_t if taux_t else None
            else:
                uplift = taux_t - taux_o
            resultat[valeur] = Effet(x["no"], x["nt"], taux_o, taux_t, uplift)
        if defaut == "liste":
            return list(resultat.items())
        return resultat

    return effet_par_segment


def _effets_pandas(clients, variable):
    """Une autre écriture juste : pandas, groupby(dropna=False, sort=False)."""
    for c in clients:
        _zero_un(c["offre"]), _zero_un(c["parti"])
    if not clients:
        return {}
    df = pd.DataFrame({"s": [c[variable] for c in clients], "o": [int(c["offre"]) for c in clients],
                       "p": [int(c["parti"]) for c in clients]})
    resultat = {}
    for valeur, groupe in df.groupby("s", dropna=False, sort=False):
        o, t = groupe[groupe.o == 1], groupe[groupe.o == 0]
        taux_o = o.p.mean() if len(o) else None
        taux_t = t.p.mean() if len(t) else None
        uplift = None if taux_o is None or taux_t is None else taux_t - taux_o
        resultat[None if isinstance(valeur, float) and math.isnan(valeur) else valeur] = Effet(
            len(o), len(t), taux_o, taux_t, uplift)
    return resultat


def test_effet_par_segment_juste_ecrit_autrement():
    _reussi("effet_par_segment", _effets())
    _reussi("effet_par_segment", _effets_pandas)


@pytest.mark.parametrize("defaut,morceaux", [
    ("signe", ["départs ÉVITÉS", "taux_temoin − taux_offre"]),
    ("taux_segment", ["SON groupe", "tout le segment"]),
    ("temoin_global", ["DE CE SEGMENT"]),
    ("zero", ["None", "pas 0.0"]),
    ("relatif", ["DIFFÉRENCE"]),
    ("groupes_inverses", ["inversé les deux groupes"]),
    ("trie", ["APPARAÎT", "ne le triez pas"]),
    ("sans_none", ["segment None"]),
    ("saisie", ["ValueError", "0 ou 1"]),
    ("modifie", ["modifié"]),
    ("liste", ["un dict"]),
    ("division", ["ZeroDivisionError"]),
])
def test_effet_par_segment_erreurs_classiques(defaut, morceaux):
    _echec("effet_par_segment", _effets(defaut), *morceaux)


def test_la_reference_212_est_celle_de_pandas():
    alea = random.Random(1)
    for _ in range(40):
        clients = generer_campagne(alea.randint(1, 300), graine=alea.randint(0, 999))
        for c in clients:
            if alea.random() < 0.1:
                c["contrat"] = None
        variable = alea.choice(VARIABLES)
        attendu = _effets_pandas(clients, variable)
        obtenu = _ref_effets(clients, variable)
        assert list(obtenu) == list(attendu)
        for valeur in attendu:
            for x, y in zip(obtenu[valeur], attendu[valeur]):
                assert (x is None and y is None) or math.isclose(x, y, abs_tol=1e-12)
        assert corrige.effet_par_segment(clients, variable) == {v: Effet(*e) for v, e in obtenu.items()}


def test_le_score_ne_dit_pas_qui_on_peut_retenir():
    """L'histoire de la docstring : les clients excédés partent le plus, et l'appel n'y change rien."""
    effets = corrige.effet_par_segment(campagne_exemple(), "appels")
    assert max(effets, key=lambda a: effets[a].taux_temoin) == "3+"
    assert effets["3+"].uplift < 0 < effets["1-2"].uplift < effets["0"].uplift


# --------------------------------------------------------------------------- Défi 213 : t_learner


def _t(defaut=None):
    """Écrite autrement que le corrigé (un Counter par (groupe, cellule)), avec UN défaut."""

    def t_learner(historique, nouveaux, variables, m=20):
        if defaut != "m_non_verifie" and (isinstance(m, bool) or not isinstance(m, (int, float)) or not m > 0):
            raise ValueError("m > 0")
        variables = list(variables)
        n, d = Counter(), Counter()
        for c in historique:
            g, y = _zero_un(c["offre"]), _zero_un(c["parti"])
            if defaut == "modifie":
                c["cellule"] = tuple(c[v] for v in variables)
            cell = tuple(c[v] for v in variables)
            n[g, cell] += 1
            d[g, cell] += y
            n[g] += 1
            d[g] += y
        if defaut != "groupes_non_verifies" and (not n[0] or not n[1]):
            raise ValueError("il faut les deux groupes")
        prior = {g: d[g] / n[g] for g in (0, 1)}
        if defaut == "a_priori_commun":
            prior = {g: (d[0] + d[1]) / (n[0] + n[1]) for g in (0, 1)}
        elif defaut == "a_priori_inverse":
            prior = {0: prior[1], 1: prior[0]}

        def risque(g, cell):
            if defaut == "sans_a_priori":
                return d[g, cell] / (n[g, cell] + m)
            if defaut == "brut":
                return d[g, cell] / n[g, cell] if n[g, cell] else prior[g]
            return (d[g, cell] + m * prior[g]) / (n[g, cell] + m)

        resultat = []
        for c in nouveaux:
            if defaut == "lit_nouveaux":
                _zero_un(c["offre"])
            cell = tuple(c[v] for v in variables)
            if defaut == "quadratique":
                nt = sum(1 for h in historique if h["offre"] == 0 and tuple(h[v] for v in variables) == cell)
                dt = sum(h["parti"] for h in historique if h["offre"] == 0 and tuple(h[v] for v in variables) == cell)
                no = sum(1 for h in historique if h["offre"] == 1 and tuple(h[v] for v in variables) == cell)
                do = sum(h["parti"] for h in historique if h["offre"] == 1 and tuple(h[v] for v in variables) == cell)
                resultat.append((dt + m * prior[0]) / (nt + m) - (do + m * prior[1]) / (no + m))
            elif defaut == "signe":
                resultat.append(risque(1, cell) - risque(0, cell))
            elif defaut == "risque_temoin":
                resultat.append(risque(0, cell))
            else:
                resultat.append(risque(0, cell) - risque(1, cell))
        return resultat

    return t_learner


def _t_pandas(historique, nouveaux, variables, m=20):
    """Une autre écriture juste : pandas (groupby par groupe et cellule, puis une jointure)."""
    if isinstance(m, bool) or not isinstance(m, (int, float)) or not m > 0:
        raise ValueError("m > 0")
    variables = list(variables)
    for c in historique:
        _zero_un(c["offre"]), _zero_un(c["parti"])
    h = pd.DataFrame({"cell": [tuple(c[v] for v in variables) for c in historique],
                      "g": [int(c["offre"]) for c in historique], "y": [int(c["parti"]) for c in historique]})
    if h.empty or h.g.nunique() < 2:
        raise ValueError("il faut les deux groupes")
    resultat = np.zeros(len(nouveaux))
    cellules = pd.Series([tuple(c[v] for v in variables) for c in nouveaux], dtype=object)
    for g, signe in ((0, 1), (1, -1)):
        groupe = h[h.g == g]
        prior = groupe.y.mean()
        stats = groupe.groupby("cell", dropna=False).y.agg(["sum", "count"])
        somme = cellules.map(stats["sum"]).fillna(0).to_numpy(dtype=float) if len(cellules) else np.zeros(0)
        nombre = cellules.map(stats["count"]).fillna(0).to_numpy(dtype=float) if len(cellules) else np.zeros(0)
        resultat += signe * (somme + m * prior) / (nombre + m)
    return [float(x) for x in resultat]


def test_t_learner_juste_ecrit_autrement():
    _reussi("t_learner", _t())
    _reussi("t_learner", _t_pandas)


def test_t_learner_en_pandas_donne_les_memes_predictions():
    campagne, prospects = campagne_exemple(), prospects_exemple()
    for variables, m in ((VARIABLES, 20), (("contrat",), 1), (("appels", "anciennete"), 7.5)):
        attendu = _ref_t_learner(campagne, prospects, variables, m)
        assert np.allclose(_t_pandas(campagne, prospects, variables, m), attendu, atol=1e-12)
        assert np.allclose(corrige.t_learner(campagne, prospects, variables, m), attendu, atol=1e-12)


@pytest.mark.parametrize("defaut,morceaux", [
    ("signe", ["SANS offre moins le risque AVEC"]),
    ("a_priori_commun", ["SON taux global"]),
    ("a_priori_inverse", ["croisé les deux a priori"]),
    ("sans_a_priori", ["m × P_g au numérateur"]),
    ("brut", ["LISSÉS", "taux brut"]),
    ("risque_temoin", ["RISQUE de départ", "soustraire"]),
    ("lit_nouveaux", ["NOUVEAU client", "`offre`"]),
    ("quadratique", ["beaucoup trop lent", "une fois par cellule"]),
    ("m_non_verifie", ["ValueError", "m est un nombre"]),
    ("groupes_non_verifies", ["ValueError", "pas d'effet mesurable"]),
    ("modifie", ["modifié"]),
])
def test_t_learner_erreurs_classiques(defaut, morceaux):
    _echec("t_learner", _t(defaut), *morceaux)


def test_t_learner_un_seul_modele_est_un_modele_de_churn():
    def un_seul_modele(historique, nouveaux, variables, m=20):
        return _ref_t_learner(historique, nouveaux, variables, m, "risque_global")

    _echec("t_learner", un_seul_modele, "UN risque", "modèle de churn")


def test_le_lissage_tire_une_petite_cellule_vers_son_groupe():
    """Une cellule d'un seul témoin parti : 100 % brut, ramené vers 50 % (le taux des témoins) par le lissage."""
    historique = [{"c": "a", "offre": 0, "parti": 1}, {"c": "b", "offre": 0, "parti": 0},
                  {"c": "b", "offre": 1, "parti": 0}, {"c": "a", "offre": 1, "parti": 0}]
    [petit_m] = corrige.t_learner(historique, [{"c": "a"}], ["c"], m=0.001)
    [grand_m] = corrige.t_learner(historique, [{"c": "a"}], ["c"], m=1000)
    assert math.isclose(petit_m, 1.0, abs_tol=1e-3) and math.isclose(grand_m, 0.5, abs_tol=1e-3)


# --------------------------------------------------------------------------- Défi 214 : issue_transformee


def _issue(defaut=None):
    """Écrite autrement que le corrigé (la formule en une ligne, qui fabrique des -0.0), avec UN défaut."""

    def issue_transformee(clients):
        resultat = []
        for c in clients:
            t, y, p = _zero_un(c["offre"]), _zero_un(c["parti"]), c["p_offre"]
            if defaut == "nan_accepte":
                if p is None or isinstance(p, str) or p <= 0 or p >= 1:
                    raise ValueError("p")
            elif defaut != "p_non_verifie" and (isinstance(p, bool) or not isinstance(p, (int, float))
                                                or not 0 < p < 1):
                raise ValueError("p")
            if defaut == "demi":
                p = 0.5
            elif defaut == "part_globale":
                p = sum(int(x["offre"]) for x in clients) / len(clients)
            if defaut == "signe":
                resultat.append(y * (t / p - (1 - t) / (1 - p)))
            elif defaut == "retenu":
                resultat.append((1 - y) * ((1 - t) / (1 - p) - t / p))
            elif defaut == "sans_un_moins_p":
                resultat.append(y * ((1 - t) - t / p))
            elif defaut == "inverse":
                resultat.append(y * ((1 - t) / p - t / (1 - p)))
            elif defaut == "autre_formule":                # −y (T − p) / (p (1 − p)) : la même chose, autrement
                resultat.append(-y * (t - p) / (p * (1 - p)))
            else:
                resultat.append(y * ((1 - t) / (1 - p) - t / p))
        return resultat

    return issue_transformee


def test_issue_transformee_juste_ecrite_autrement():
    _reussi("issue_transformee", _issue())
    _reussi("issue_transformee", _issue("autre_formule"))


@pytest.mark.parametrize("defaut,morceaux", [
    ("signe", ["départs ÉVITÉS", "inversés"]),
    ("retenu", ["`parti` qui porte l'information"]),
    ("demi", ["SA probabilité", "0.5 pour tout le monde"]),
    ("part_globale", ["SA probabilité", "pas la part"]),
    ("sans_un_moins_p", ["1 / (1 − p), pas 1"]),
    ("inverse", ["croisé p et 1 − p"]),
    ("p_non_verifie", ["ValueError", "0 < p < 1"]),
    ("nan_accepte", ["p_offre = nan", "refuse aussi NaN"]),
])
def test_issue_transformee_erreurs_classiques(defaut, morceaux):
    _echec("issue_transformee", _issue(defaut), *morceaux)


def test_issue_transformee_modifiee_en_place():
    def issue_transformee(clients):
        resultat = corrige.issue_transformee(clients)
        for c in clients:
            c["offre"] = bool(c["offre"])
        return resultat

    _echec("issue_transformee", issue_transformee, "modifié")


def test_la_moyenne_de_l_issue_transformee_est_l_uplift():
    """Avec p = la part EXACTE d'appelés d'un groupe, la moyenne de l'issue transformée EST l'écart des taux."""
    alea = random.Random(3)
    for _ in range(30):
        clients = generer_campagne(alea.randint(2, 200), graine=alea.randint(0, 999), p_offre=alea.choice([0.3, 0.8]))
        n_offre = sum(c["offre"] for c in clients)
        if n_offre in (0, len(clients)):
            continue
        for c in clients:
            c["p_offre"] = n_offre / len(clients)
        tous = [dict(c, tous=1) for c in clients]
        assert math.isclose(fmean(corrige.issue_transformee(clients)),
                            corrige.effet_par_segment(tous, "tous")[1].uplift, abs_tol=1e-12)


def test_l_issue_transformee_corrige_un_tirage_selon_le_score():
    """En moyenne sur beaucoup de campagnes stratifiées : sans biais repondérée, biaisée en différence brute."""
    brutes, ponderees, vraies = [], [], []
    for graine in range(40):
        clients = generer_campagne(3000, graine=graine, stratifiee=True)
        tous = [dict(c, tous=1) for c in clients]
        brutes.append(corrige.effet_par_segment(tous, "tous")[1].uplift)
        ponderees.append(fmean(corrige.issue_transformee(clients)))
        vraies.append(fmean(uplift_vrai(c) for c in clients))
    biais_brut, biais_pondere = fmean(brutes) - fmean(vraies), fmean(ponderees) - fmean(vraies)
    assert biais_brut < -0.05 and abs(biais_pondere) < 0.01


# --------------------------------------------------------------------------- Défi 215 : courbe_qini


def _qini_numpy(uplifts, clients):
    """Une autre écriture juste : NumPy (argsort stable, cumsum, fin de chaque groupe d'ex aequo)."""
    n = len(clients)
    if len(uplifts) != n or n == 0:
        raise ValueError("longueurs")
    u = np.array([np.nan if x is None else x for x in uplifts], dtype=float)
    if np.isnan(u).any():
        raise ValueError("NaN")
    t = np.array([_zero_un(c["offre"]) for c in clients])
    y = np.array([_zero_un(c["parti"]) for c in clients])
    ordre = np.argsort(-u, kind="stable")
    u, t, y = u[ordre], t[ordre], y[ordre]
    fins = np.r_[np.flatnonzero(u[1:] != u[:-1]), n - 1]
    n_t, d_t = np.cumsum(t)[fins], np.cumsum(t * y)[fins]
    n_c, d_c = np.cumsum(1 - t)[fins], np.cumsum((1 - t) * y)[fins]
    with np.errstate(divide="ignore", invalid="ignore"):
        gains = np.where(n_c > 0, d_c * n_t / np.where(n_c > 0, n_c, 1) - d_t, 0.0)
    k = np.r_[0, fins + 1]
    g = np.r_[0.0, gains]
    courbe = np.sum((k[1:] - k[:-1]) * (g[1:] + g[:-1]) / 2)
    return Qini(list(zip(k, g)), (courbe - n * g[-1] / 2) / n)


def _qini(defaut=None):
    """La référence du correcteur avec UN défaut, ou écrite naïvement (quadratique, NaN accepté)."""

    def courbe_qini(uplifts, clients):
        if defaut == "quadratique":
            n = len(clients)
            valeurs = sorted(set(uplifts), reverse=True)
            points = [(0, 0.0)]
            for v in valeurs:
                haut = [c for u, c in zip(uplifts, clients) if u >= v]
                n_t = sum(1 for c in haut if c["offre"])
                d_t = sum(c["parti"] for c in haut if c["offre"])
                n_c = len(haut) - n_t
                d_c = sum(c["parti"] for c in haut if not c["offre"])
                points.append((len(haut), d_c * n_t / n_c - d_t if n_c else 0.0))
            courbe = sum((k2 - k1) * (g1 + g2) / 2 for (k1, g1), (k2, g2) in zip(points, points[1:]))
            return Qini(points, (courbe - n * points[-1][1] / 2) / n)
        if defaut == "nan_accepte":
            if len(uplifts) != len(clients) or not clients:
                raise ValueError("longueurs")
            for c in clients:
                _zero_un(c["offre"]), _zero_un(c["parti"])
            return Qini(*_ref_qini(uplifts, clients))
        if defaut == "listes":
            points, qini = corrige.courbe_qini(uplifts, clients)
            return [[k, g] for k, g in points], qini
        return Qini(*checks._ref_qini(uplifts, clients, defaut)) if defaut else corrige.courbe_qini(uplifts, clients)

    return courbe_qini


def test_courbe_qini_juste_ecrite_autrement():
    _reussi("courbe_qini", _qini_numpy)
    _reussi("courbe_qini", lambda u, c: tuple(corrige.courbe_qini(u, c)))
    _reussi("courbe_qini", _qini("listes"))                      # des [k, gain] au lieu de (k, gain) : accepté


@pytest.mark.parametrize("defaut,morceaux", [
    ("ex_aequo", ["EX AEQUO", "ENSEMBLE"]),
    ("croissant", ["DÉCROISSANTE"]),
    ("sans_origine", ["(0, 0.0)"]),
    ("ratio_global", ["DES k PREMIERS", "totaux"]),
    ("sans_ratio", ["D_C × N_T / N_C"]),
    ("taux", ["courbe d'UPLIFT"]),
    ("nc_zero", ["N_C = 0", "pas −D_T"]),
    ("signe", ["départs ÉVITÉS", "l'inverse"]),
    ("qini_brut", ["AU-DESSUS de la droite du hasard"]),
    ("qini_non_normalise", ["diviser par n"]),
    ("quadratique", ["beaucoup trop lent", "UNE passe"]),
])
def test_courbe_qini_erreurs_classiques(defaut, morceaux):
    _echec("courbe_qini", _qini(defaut), *morceaux)


def test_courbe_qini_nan_accepte():
    message = _echec("courbe_qini", _qini("nan_accepte"), "ValueError")
    assert "NaN" in message or "None" in message


def test_la_reference_215_est_celle_de_numpy():
    alea = random.Random(5)
    for _ in range(60):
        clients = generer_campagne(alea.randint(1, 300), graine=alea.randint(0, 999))
        uplifts = [alea.choice([0.0, 0.1, 0.2, -0.1]) if alea.random() < 0.5 else alea.random() for _ in clients]
        points, qini = _ref_qini(uplifts, clients)
        q = _qini_numpy(uplifts, clients)
        assert [k for k, _ in points] == [int(k) for k, _ in q.points]
        assert np.allclose([g for _, g in points], [g for _, g in q.points], atol=1e-9)
        assert math.isclose(qini, q.qini, abs_tol=1e-9)
        assert corrige.courbe_qini(uplifts, clients) == Qini(points, qini)


def test_tout_le_monde_ex_aequo_c_est_le_hasard():
    clients = campagne_exemple()
    q = corrige.courbe_qini([0.0] * len(clients), clients)
    assert len(q.points) == 2 and q.qini == 0.0
    tous = [dict(c, tous=1) for c in clients]
    effet = corrige.effet_par_segment(tous, "tous")[1]
    assert math.isclose(q.points[-1][1], effet.uplift * effet.n_offre)   # le gain total : uplift × appelés


def test_l_histoire_des_docstrings_tient():
    """Le score classe mieux que le hasard mais bien moins que la vérité ; jugé sur ses clients, le modèle se flatte."""
    campagne = campagne_exemple()
    score = corrige.courbe_qini([c["score"] for c in campagne], campagne).qini
    vrai = corrige.courbe_qini([uplift_vrai(c) for c in campagne], campagne).qini
    deja_vus = corrige.courbe_qini(corrige.t_learner(campagne, campagne, VARIABLES), campagne).qini
    apprendre = lambda e, p: corrige.t_learner(e, p, VARIABLES)  # noqa: E731
    jamais_vus = corrige.courbe_qini(corrige.predictions_hors_pli(campagne, apprendre), campagne).qini
    assert 0 < score < jamais_vus < vrai < deja_vus


# --------------------------------------------------------------------------- Défi 216 : plan_de_contact


def _plan(defaut=None):
    """Écrite autrement que le corrigé (heapq.nsmallest), avec UN défaut."""

    def plan_de_contact(clients, uplifts, cout, budget):
        if defaut != "float_accepte":
            for x, minimum in ((cout, 1), (budget, 0)):
                if type(x) is not int or x < minimum:
                    raise ValueError("entier")
        if defaut != "zip" and len(clients) != len(uplifts):
            raise ValueError("longueurs")
        lignes = []
        for c, u in zip(clients, uplifts):
            if _manquant(u) or _manquant(c["valeur"]):
                raise ValueError("manquant")
            lignes.append((u * c["valeur"] - cout, c["customer_id"], u, c["score"]))
        n_max = int(budget // cout)
        if defaut == "par_score":
            choisis = heapq.nsmallest(n_max, lignes, key=lambda t: (-t[3], t[1]))
        elif defaut == "par_uplift":
            choisis = heapq.nsmallest(n_max, [t for t in lignes if t[0] > 0], key=lambda t: (-t[2], t[1]))
        elif defaut == "remplir":
            choisis = heapq.nsmallest(n_max, lignes, key=lambda t: (-t[0], t[1]))
        elif defaut == "zero_inclus":
            choisis = heapq.nsmallest(n_max, [t for t in lignes if t[0] >= 0], key=lambda t: (-t[0], t[1]))
        elif defaut == "egalite_inverse":
            choisis = sorted([t for t in lignes if t[0] > 0], key=lambda t: (t[0], t[1]), reverse=True)[:n_max]
        else:
            choisis = heapq.nsmallest(n_max, [t for t in lignes if t[0] > 0], key=lambda t: (-t[0], t[1]))
        if defaut == "ordre":
            choisis.sort(key=lambda t: t[1])
        gain = sum(t[0] for t in choisis)
        if defaut == "brut":
            gain = sum(t[0] + cout for t in choisis)
        depense = budget if defaut == "cout_budget" else len(choisis) * cout
        return Plan([t[1] for t in choisis], gain, depense)

    return plan_de_contact


def test_plan_de_contact_juste_ecrit_autrement():
    _reussi("plan_de_contact", _plan())


@pytest.mark.parametrize("defaut,morceaux", [
    ("par_score", ["SCORE de churn", "causes perdues"]),
    ("par_uplift", ["GAIN attendu", "pas par uplift seul"]),
    ("remplir", ["strictement positif", "chiens qui"]),
    ("zero_inclus", ["gain nul"]),
    ("egalite_inverse", ["CROISSANT"]),
    ("brut", ["NET"]),
    ("cout_budget", ["len(clients) × cout"]),
    ("ordre", ["mauvais ordre"]),
    ("float_accepte", ["ValueError", "entier d'euros"]),
    ("zip", ["ValueError", "`zip` tronquerait"]),
])
def test_plan_de_contact_erreurs_classiques(defaut, morceaux):
    _echec("plan_de_contact", _plan(defaut), *morceaux)


def test_le_plan_glouton_est_optimal_a_cout_egal():
    """Recherche exhaustive sur de petits cas : aucun autre choix de clients ne rapporte plus."""
    alea = random.Random(7)
    for _ in range(80):
        n = alea.randint(0, 9)
        clients = [{"customer_id": f"K{i}", "valeur": alea.choice([300, 600, 1000]), "score": 0.5} for i in range(n)]
        uplifts = [alea.choice([-0.1, 0.0, 0.03, 0.05, 0.1]) for _ in range(n)]
        cout, budget = 30, alea.choice([0, 30, 60, 100, 1000])
        plan = corrige.plan_de_contact(clients, uplifts, cout, budget)
        gains = [u * c["valeur"] - cout for c, u in zip(clients, uplifts)]
        meilleur = max(sum(gains[i] for i in choix) for r in range(min(n, budget // cout) + 1)
                       for choix in itertools.combinations(range(n), r))
        assert math.isclose(plan.gain, meilleur, abs_tol=1e-9)
        assert plan == Plan(*_ref_plan(clients, uplifts, cout, budget))


def test_appeler_par_l_uplift_rapporte_plus_que_par_le_score():
    """Vérifié avec la VÉRITÉ simulée : à budget égal (300 appels), le plan par l'uplift rapporte bien plus."""
    campagne, prospects = campagne_exemple(), prospects_exemple()
    plan = corrige.plan_de_contact(prospects, corrige.t_learner(campagne, prospects, VARIABLES), COUT_CONTACT, 9000)
    par_id = {c["customer_id"]: c for c in prospects}
    vrai_plan = sum(uplift_vrai(par_id[i]) * par_id[i]["valeur"] - COUT_CONTACT for i in plan.clients)
    par_score = sorted(prospects, key=lambda c: -c["score"])[:300]
    vrai_score = sum(uplift_vrai(c) * c["valeur"] - COUT_CONTACT for c in par_score)
    assert len(plan.clients) == 300 and vrai_plan > 1.4 * vrai_score
    assert plan.gain > vrai_plan               # le gain annoncé est optimiste : on a choisi les estimations hautes


# --------------------------------------------------------------------------- Défi 217 : predictions_hors_pli


def _oof(defaut=None):
    """Écrite autrement que le corrigé (un dict pli → indices), avec UN défaut."""

    def predictions_hors_pli(historique, apprendre, k=5):
        n = len(historique)
        if defaut != "k_non_verifie" and (type(k) is not int or not 2 <= k <= n):
            raise ValueError("k")
        plis = defaultdict(list)
        for i in range(n):
            plis[i * k // n if defaut == "blocs" else i % k].append(i)
        predictions = {}
        ordre = []
        for pli in range(k):
            indices = plis[pli]
            if defaut == "dans_l_echantillon":
                entrainement = list(historique)
            elif defaut == "tout_retire":
                entrainement = [{c: v for c, v in historique[i].items() if c not in ("offre", "parti")}
                                for i in range(n) if i not in indices]
            else:
                entrainement = [historique[i] for i in range(n) if i not in indices]
            if defaut == "ordre_entrainement":
                entrainement = entrainement[::-1]
            if defaut == "reponse_visible":
                a_predire = [historique[i] for i in indices]
            elif defaut == "del":
                a_predire = []
                for i in indices:
                    del historique[i]["parti"]
                    a_predire.append(historique[i])
            elif defaut == "trop_retire":
                a_predire = [{c: v for c, v in historique[i].items() if c not in ("offre", "parti", "p_offre")}
                             for i in indices]
            else:
                a_predire = [{c: v for c, v in historique[i].items() if c not in ("offre", "parti")} for i in indices]
            resultat = apprendre(entrainement, a_predire)
            if defaut != "longueur_non_verifiee" and len(resultat) != len(indices):
                raise ValueError("longueur")
            for i, u in zip(indices, resultat):
                predictions[i] = u
                ordre.append(u)
        if defaut == "bout_a_bout":
            return ordre
        return [predictions.get(i) for i in range(n)]

    return predictions_hors_pli


def test_predictions_hors_pli_juste_ecrit_autrement():
    _reussi("predictions_hors_pli", _oof())


@pytest.mark.parametrize("defaut,morceaux", [
    ("dans_l_echantillon", ["qu'il doit ensuite prédire", "triche"]),
    ("reponse_visible", ["`parti`", "finit toujours par la lire"]),
    ("del", ["modifié l'historique"]),
    ("trop_retire", ["on ne retire QUE"]),
    ("tout_retire", ["À PRÉDIRE", "rien à apprendre"]),
    ("blocs", ["i % k", "pas un bloc"]),
    ("bout_a_bout", ["mises bout à bout", "SA place"]),
    ("ordre_entrainement", ["dans le désordre"]),
    ("k_non_verifie", ["ValueError", "entre 2 et le nombre de clients"]),
    ("longueur_non_verifiee", ["ValueError", "une prédiction par client"]),
])
def test_predictions_hors_pli_erreurs_classiques(defaut, morceaux):
    _echec("predictions_hors_pli", _oof(defaut), *morceaux)


def test_les_plis_sont_ceux_de_predefined_split():
    from sklearn.model_selection import PredefinedSplit

    historique = generer_campagne(23, graine=4)
    vus = []

    def apprendre(entrainement, a_predire):
        vus.append(([c["customer_id"] for c in entrainement], [c["customer_id"] for c in a_predire]))
        return [0.0] * len(a_predire)

    corrige.predictions_hors_pli(historique, apprendre, k=4)
    ids = np.array([c["customer_id"] for c in historique])
    attendus = [(list(ids[a]), list(ids[b])) for a, b in PredefinedSplit([i % 4 for i in range(23)]).split()]
    assert vus == attendus


def test_l_espion_reconnait_chaque_client():
    espion = checks._Apprenti()
    clients = generer_campagne(4, graine=1)
    r = espion(clients[:2], [{"customer_id": "C00003"}, {"customer_id": "C00004"}])
    assert r == [checks._valeur_espion("C00003") + 2 / 10_000, checks._valeur_espion("C00004") + 2 / 10_000]
    assert checks._valeur_espion("C00003") != checks._valeur_espion("C00004")
    assert espion.appels[0][0] == ["C00001", "C00002"]


# --------------------------------------------------------------------------- Outils communs


def test_les_campagnes_sont_reproductibles_et_neuves_a_chaque_appel():
    a, b = campagne_exemple(), campagne_exemple()
    assert a == b and a is not b and a[0] is not b[0]
    a[0]["parti"] = 99
    assert campagne_exemple()[0]["parti"] != 99
    assert all(set(c) == {"customer_id", "contrat", "anciennete", "appels", "score", "valeur"} for c in prospects_exemple())
    assert {c["p_offre"] for c in campagne_exemple()} == {0.8}
    assert {c["p_offre"] for c in campagne_stratifiee_exemple()} == {0.5, 0.9}
    assert all((c["p_offre"] == 0.9) == (c["score"] >= 0.5) for c in campagne_stratifiee_exemple())


def test_la_verite_simulee_contient_les_quatre_profils():
    profils = [dict(zip(VARIABLES, p)) for p in itertools.product(ciblage.CONTRATS, ciblage.ANCIENNETES, ciblage.APPELS)]
    uplifts = {uplift_vrai(c) for c in profils}
    assert min(uplifts) < 0 < max(uplifts) and 0.0 in uplifts                   # chiens qui dorment, persuadables
    assert all(0 <= risque_vrai(c) - uplift_vrai(c) <= 1 for c in profils)       # une probabilité, avec ou sans offre
    perdus = [c for c in profils if c["appels"] == "3+"]
    assert all(uplift_vrai(c) == 0.0 and risque_vrai(c) >= 0.4 for c in perdus)  # causes perdues


def test_les_taux_simules_sont_ceux_de_la_verite():
    clients = generer_campagne(40_000, graine=11, p_offre=0.5)
    temoins = [c for c in clients if not c["offre"]]
    assert abs(fmean(c["parti"] for c in temoins) - fmean(risque_vrai(c) for c in temoins)) < 0.01


# --------------------------------------------------------------------------- La série


def test_les_references_rejouent_les_bonnes_erreurs():
    clients = [{"s": "a", "offre": 1, "parti": 0}, {"s": "a", "offre": 0, "parti": 1}]
    assert _ref_effets(clients, "s", "signe")["a"][4] == -1.0
    assert _ref_issue([{"offre": 0, "parti": 1, "p_offre": 0.8}], "demi") == [2.0]
    assert _ref_qini([0.2, 0.1], [{"offre": 1, "parti": 0}, {"offre": 0, "parti": 1}], "sans_origine")[0] == [
        (1, 0.0), (2, 1.0)]
    plan = _ref_plan([{"customer_id": "A", "valeur": 100, "score": 0.9}], [-0.5], 10, 100, "remplir")
    assert plan[0] == ["A"] and plan[1] == -60.0


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(212, 218))
    assert any(s is SERIE for s in series())            # enregistrée dans defis/verif.py


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_35 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(ciblage)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_35 import *` (dans les exemples) doit donner les fonctions du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 50 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    campagne, prospects = campagne_exemple()[:300], prospects_exemple()[:50]
    copies = copy.deepcopy((campagne, prospects))
    corrige.effet_par_segment(campagne, "contrat")
    predits = corrige.t_learner(campagne, prospects, VARIABLES)
    corrige.issue_transformee(campagne)
    corrige.courbe_qini([c["score"] for c in campagne], campagne)
    corrige.plan_de_contact(prospects, predits, 30, 600)
    corrige.predictions_hors_pli(campagne, lambda e, p: corrige.t_learner(e, p, VARIABLES), 5)
    assert (campagne, prospects) == copies

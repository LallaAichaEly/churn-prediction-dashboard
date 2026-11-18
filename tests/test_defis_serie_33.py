"""Série 33 (prévoir les départs) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 200`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (`itertools.cycle`, `statistics.fmean`, la
  forme « correction d'erreur » des lissages, une validation qui allonge la même liste, `numpy.quantile`…) ;
- le corrigé et les références du correcteur donnent les mêmes résultats que scikit-learn (métriques,
  `TimeSeriesSplit`), pandas (`ewm`) et numpy (`quantile(method="inverted_cdf")`) ;
- les exemples des docstrings, les outils communs et le corrigé sont justes.
"""

from __future__ import annotations

import copy
import doctest
import heapq
import itertools
import math
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.checks_serie_33 as checks  # noqa: E402
import defis.corrige_serie_33 as corrige  # noqa: E402
import defis.previsions as previsions  # noqa: E402
from defis.checks_serie_33 import (  # noqa: E402
    SERIE,
    _ref_erreurs,
    _ref_holt,
    _ref_marge,
    _ref_naive,
    _ref_ses,
    _ref_validation,
    _regle,
)
from defis.previsions import (  # noqa: E402
    DOSSIERS_PAR_CONSEILLER,
    Erreurs,
    Essai,
    Lissage,
    Marge,
    departs_exemple,
    generer_serie,
    mois_exemple,
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


def _manquant(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


# --------------------------------------------------------------------------- Défi 200 : prevision_naive


def _naive(defaut=None):
    """Écrite autrement que le corrigé (itertools.cycle et islice), avec UN défaut."""

    def prevision_naive(historique, horizon, saison=1):
        if defaut != "sans_controle":
            if horizon < 0 or saison < 1 or len(historique) < saison:
                raise ValueError("paramètres invalides")
            if defaut != "nan_accepte" and any(_manquant(x) for x in historique):
                raise ValueError("valeur manquante")
        if defaut == "modifie" and isinstance(historique, list) and historique:
            historique[-1] = historique[-1] + 1
        n = len(historique)
        if defaut == "a_rebours":
            return [float(historique[n - 1 - (h % saison)]) for h in range(horizon)]
        if defaut == "decale":
            return [float(historique[n - saison + (h + 1) % saison]) for h in range(horizon)]
        if defaut == "constante":
            return [float(historique[-saison])] * horizon
        if defaut == "dernier":
            return [float(historique[-1])] * horizon
        if defaut == "moyenne":
            return [statistics.fmean(historique[n - saison:])] * horizon
        if defaut == "premier_cycle":
            return [float(x) for x in itertools.islice(itertools.cycle(historique[:saison]), horizon)]
        if defaut == "sans_modulo":
            return [float(historique[n - saison + h]) for h in range(horizon)]
        if defaut == "tuple":
            return tuple(float(x) for x in itertools.islice(itertools.cycle(historique[n - saison:]), horizon))
        return [float(x) for x in itertools.islice(itertools.cycle(historique[n - saison:]), horizon)]

    return prevision_naive


def test_prevision_naive_juste_ecrite_autrement():
    _reussi("prevision_naive", _naive())


@pytest.mark.parametrize("defaut,morceaux", [
    ("a_rebours", ["ordre du CALENDRIER", "historique[-h]"]),
    ("decale", ["un mois trop loin"]),
    ("constante", ["h % saison"]),
    ("dernier", ["`saison` est ignoré"]),
    ("moyenne", ["RECOPIE"]),
    ("premier_cycle", ["DERNIER cycle"]),
    ("sans_modulo", ["IndexError", "on recommence"]),
    ("sans_controle", ["ValueError"]),
    ("nan_accepte", ["ValueError", "valeur manquante"]),
    ("modifie", ["modifié"]),
    ("tuple", ["list"]),
])
def test_prevision_naive_erreurs_classiques(defaut, morceaux):
    _echec("prevision_naive", _naive(defaut), *morceaux)


# --------------------------------------------------------------------------- Défi 201 : erreurs


def _erreurs(defaut=None):
    """Écrite autrement (statistics.fmean, une boucle par mesure), avec UN défaut."""

    def erreurs(reels, prevus):
        if defaut != "zip_tronque" and len(reels) != len(prevus):
            raise ValueError("longueurs différentes")
        if not reels:
            raise ValueError("vide")
        if any(_manquant(x) for x in list(reels) + list(prevus)):
            raise ValueError("valeur manquante")
        paires = list(zip(reels, prevus))
        ecarts = [p - r for r, p in paires]
        mae = statistics.fmean(abs(e) for e in ecarts)
        rmse = math.sqrt(statistics.fmean(e * e for e in ecarts))
        biais = statistics.fmean(ecarts)
        zero = any(r == 0 for r, _ in paires)
        mape = None if zero else statistics.fmean(abs(e / r) for e, (r, _) in zip(ecarts, paires))
        if defaut == "biais_inverse":
            biais = -biais
        elif defaut == "mae_signee":
            mae = statistics.fmean(ecarts)
        elif defaut == "mse":
            rmse = statistics.fmean(e * e for e in ecarts)
        elif defaut == "racine_avant":
            rmse = math.sqrt(sum(e * e for e in ecarts)) / len(ecarts)
        elif defaut == "rmse_est_mae":
            rmse = statistics.fmean(math.sqrt(e * e) for e in ecarts)
        elif defaut == "mape_prevu":
            mape = None if zero else statistics.fmean(abs(e) / abs(p) if p else math.inf for e, (_, p) in
                                                      zip(ecarts, paires))
        elif defaut == "mape_pourcent" and mape is not None:
            mape = 100 * mape
        elif defaut == "mape_sans_zeros":
            gardes = [abs(e) / abs(r) for e, (r, _) in zip(ecarts, paires) if r != 0]
            mape = statistics.fmean(gardes) if gardes else None
        elif defaut == "mape_zero" and zero:
            mape = 0.0
        elif defaut == "mape_signee" and not zero:
            mape = statistics.fmean(abs(e) / r for e, (r, _) in zip(ecarts, paires))
        if defaut == "ordre":
            return (rmse, mae, biais, mape)
        if defaut == "dict":
            return {"mae": mae, "rmse": rmse, "biais": biais, "mape": mape}
        return Erreurs(mae, rmse, biais, mape)

    return erreurs


def test_erreurs_juste_ecrite_autrement():
    _reussi("erreurs", _erreurs())


@pytest.mark.parametrize("defaut,morceaux", [
    ("biais_inverse", ["biais", "signe est inversé"]),
    ("mae_signee", ["mae", "ABSOLUES"]),
    ("mse", ["rmse", "MSE"]),
    ("racine_avant", ["rmse", "AVANT de prendre la racine"]),
    ("rmse_est_mae", ["rmse", "c'est la MAE"]),
    ("mape_prevu", ["mape", "par le RÉEL"]),
    ("mape_pourcent", ["mape", "100"]),
    ("mape_sans_zeros", ["mape", "None"]),
    ("mape_zero", ["mape", "None"]),
    ("mape_signee", ["abs(r)"]),
    ("ordre", ["mauvais ordre"]),
    ("zip_tronque", ["ValueError", "zip"]),
    ("dict", ["Erreurs(mae, rmse, biais, mape)"]),
])
def test_erreurs_erreurs_classiques(defaut, morceaux):
    _echec("erreurs", _erreurs(defaut), *morceaux)


def test_erreurs_comme_scikit_learn():
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

    alea = random.Random(1)
    for _ in range(100):
        n = alea.randint(1, 20)
        reels = [alea.uniform(1, 500) for _ in range(n)]
        prevus = [alea.uniform(1, 500) for _ in range(n)]
        e = corrige.erreurs(reels, prevus)
        assert math.isclose(e.mae, mean_absolute_error(reels, prevus))
        assert math.isclose(e.rmse, math.sqrt(mean_squared_error(reels, prevus)))
        assert math.isclose(e.mape, mean_absolute_percentage_error(reels, prevus))
        assert math.isclose(e.biais, float(np.mean(np.array(prevus) - np.array(reels))), abs_tol=1e-9)
        assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(e, _ref_erreurs(reels, prevus)))


# --------------------------------------------------------------------------- Défi 202 : lissage_exponentiel


def _ses(defaut=None):
    """Écrit sous la forme « correction d'erreur » (niveau += alpha × erreur), avec UN défaut."""

    def lissage_exponentiel(historique, alpha, horizon):
        if defaut != "alpha_zero_accepte" and not 0 < alpha <= 1:
            raise ValueError("alpha")
        if defaut == "alpha_zero_accepte" and not 0 <= alpha <= 1:
            raise ValueError("alpha")
        if horizon < 0 or not historique or any(_manquant(x) for x in historique):
            raise ValueError("entrée invalide")
        valeurs = list(historique)
        niveau = statistics.fmean(valeurs) if defaut == "moyenne_initiale" else valeurs[0]
        ajustees = [valeurs[0] if defaut == "premier_ajuste" else None]
        precedent = niveau
        for y in valeurs[1:]:
            if defaut != "fuite":
                ajustees.append(niveau)
            precedent = niveau
            if defaut == "inverse":
                niveau = (1 - alpha) * y + alpha * niveau
            else:
                niveau += alpha * (y - niveau)
            if defaut == "fuite":
                ajustees.append(niveau)
            if defaut == "lent":                      # tout le passé relissé à chaque mois
                verif = valeurs[0]
                for z in valeurs[1:len(ajustees)]:
                    verif += alpha * (z - verif)
        if defaut == "sans_premier":
            ajustees = ajustees[1:]
        final = niveau
        if defaut == "derniere_valeur":
            final = valeurs[-1]
        if defaut == "dernier_mois_oublie":
            final = precedent
        tendance = None if defaut == "tendance_none" else 0.0
        return Lissage(ajustees, [final] * horizon, final, tendance)

    return lissage_exponentiel


def test_lissage_exponentiel_juste_ecrit_autrement():
    _reussi("lissage_exponentiel", _ses())


@pytest.mark.parametrize("defaut,morceaux", [
    ("inverse", ["NOUVELLE valeur"]),
    ("fuite", ["AVANT de le voir", "connaît déjà"]),
    ("moyenne_initiale", ["AVENIR"]),
    ("premier_ajuste", ["ajustees[0] vaut None"]),
    ("sans_premier", ["autant d'éléments"]),
    ("derniere_valeur", ["dernier NIVEAU"]),
    ("dernier_mois_oublie", ["DERNIER mois"]),
    ("alpha_zero_accepte", ["ValueError", "alpha"]),
    ("tendance_none", ["tendance", "0.0"]),
    ("lent", ["trop lent", "Une seule passe"]),
])
def test_lissage_exponentiel_erreurs_classiques(defaut, morceaux):
    _echec("lissage_exponentiel", _ses(defaut), *morceaux)


def test_lissage_exponentiel_comme_pandas():
    alea = random.Random(2)
    for _ in range(50):
        serie = generer_serie(alea.randint(1, 60), graine=alea.randint(0, 999))
        alpha = alea.choice([0.1, 0.3, 0.5, 0.9, 1.0])
        lisse = pd.Series(serie).ewm(alpha=alpha, adjust=False).mean().tolist()
        r = corrige.lissage_exponentiel(serie, alpha, 2)
        assert math.isclose(r.niveau, lisse[-1]) and r.previsions == [r.niveau] * 2
        assert all(math.isclose(a, b) for a, b in zip(r.ajustees[1:], lisse[:-1]))
        ref = _ref_ses(serie, alpha, 2)
        assert ref[0][0] is None and all(math.isclose(a, b) for a, b in zip(ref[0][1:], r.ajustees[1:]))


# --------------------------------------------------------------------------- Défi 203 : holt


def _holt(defaut=None):
    """Écrit sous la forme « correction d'erreur » (niveau et tendance corrigés par l'erreur), avec UN défaut."""

    def holt(historique, alpha, beta, horizon):
        if not (0 < alpha <= 1 and 0 < beta <= 1) or horizon < 0:
            raise ValueError("paramètres")
        if len(historique) < 2 or any(_manquant(x) for x in historique):
            raise ValueError("historique")
        y = list(historique)
        if defaut == "tendance_zero":
            niveau, tendance, ajustees, debut = y[0], 0.0, [None], 1
        else:
            niveau, tendance, ajustees, debut = y[1], y[1] - y[0], [None, None], 2
        if defaut == "depart_ajuste":
            ajustees = [None, y[1]]
        for t in range(debut, len(y)):
            prevu = niveau + tendance
            if defaut != "fuite":
                ajustees.append(prevu)
            erreur = y[t] - prevu
            if defaut in (None, "h_oublie", "fuite", "depart_ajuste", "tendance_zero", "lent"):
                niveau, tendance = prevu + alpha * erreur, tendance + alpha * beta * erreur
            else:
                ancien = niveau
                if defaut == "alpha_inverse":
                    niveau = (1 - alpha) * y[t] + alpha * prevu
                elif defaut == "niveau_sans_tendance":
                    niveau = alpha * y[t] + (1 - alpha) * ancien
                else:
                    niveau = alpha * y[t] + (1 - alpha) * prevu
                if defaut == "ancien_ecrase":
                    ancien = niveau
                ecart = y[t] - y[t - 1] if defaut == "tendance_brute" else niveau - ancien
                if defaut == "beta_inverse":
                    tendance = (1 - beta) * ecart + beta * tendance
                else:
                    tendance = beta * ecart + (1 - beta) * tendance
            if defaut == "fuite":
                ajustees.append(niveau + tendance)
            if defaut == "lent":
                somme = 0.0
                for z in y[:t]:
                    somme += z
        if defaut == "h_oublie":
            return Lissage(ajustees, [niveau + tendance] * horizon, niveau, tendance)
        return Lissage(ajustees, [niveau + h * tendance for h in range(1, horizon + 1)], niveau, tendance)

    return holt


def test_holt_juste_ecrit_autrement():
    _reussi("holt", _holt())


@pytest.mark.parametrize("defaut,morceaux", [
    ("h_oublie", ["h × TENDANCE"]),
    ("ancien_ecrase", ["ANCIEN niveau", "ancien = niveau"]),
    ("tendance_brute", ["deux NIVEAUX"]),
    ("niveau_sans_tendance", ["oubliez la tendance"]),
    ("alpha_inverse", ["alpha s'applique"]),
    ("beta_inverse", ["beta s'applique"]),
    ("fuite", ["AVANT de voir", "connaît déjà"]),
    ("depart_ajuste", ["ajustees[1] vaut None"]),
    ("tendance_zero", ["pas une tendance nulle"]),
    ("lent", ["trop lent"]),
])
def test_holt_erreurs_classiques(defaut, morceaux):
    _echec("holt", _holt(defaut), *morceaux)


def test_holt_prolonge_une_droite_et_retrouve_la_derive():
    droite = [7.0 + 3 * t for t in range(20)]
    for alpha, beta in ((0.1, 0.1), (0.5, 0.9), (1.0, 1.0)):
        r = corrige.holt(droite, alpha, beta, 4)                # exact au dernier chiffre près (alpha × y en float)
        assert np.allclose(r.ajustees[2:], droite[2:]) and np.allclose(r.previsions, [67.0, 70.0, 73.0, 76.0])
        assert math.isclose(r.tendance, 3.0)
    serie = departs_exemple()                                  # alpha = beta = 1 : y + h × (dernier écart)
    r = corrige.holt(serie, 1.0, 1.0, 3)
    assert r.previsions == [serie[-1] + h * (serie[-1] - serie[-2]) for h in (1, 2, 3)]


def test_les_deux_formes_de_holt_sont_les_memes():
    alea = random.Random(3)
    juste = _holt()
    for _ in range(100):
        serie = generer_serie(alea.randint(2, 50), graine=alea.randint(0, 999), tendance=alea.choice([-1.0, 2.0]))
        alpha, beta = alea.choice([0.2, 0.5, 1.0]), alea.choice([0.1, 0.5, 1.0])
        a, b, ref = corrige.holt(serie, alpha, beta, 3), juste(serie, alpha, beta, 3), _ref_holt(serie, alpha, beta, 3)
        for x, y, z in zip(a.ajustees + a.previsions, b.ajustees + b.previsions, ref[0] + ref[1]):
            assert (x is None and y is None and z is None) or (math.isclose(x, y, abs_tol=1e-9)
                                                               and math.isclose(x, z, abs_tol=1e-9))


# --------------------------------------------------------------------------- Défi 204 : validation_glissante


def _validation(defaut=None):
    """Écrite autrement : UNE liste qui s'allonge mois après mois (pas de copie par tranche), avec UN défaut."""

    def validation_glissante(serie, methode, debut, horizon=1):
        if defaut != "sans_controle" and (horizon < 1 or debut < 1 or debut > len(serie) - horizon):
            raise ValueError("rien à tester")
        origines = list(range(debut, len(serie) - horizon + 1))
        if defaut == "derniere_oubliee":
            origines = origines[:-1]
        if defaut == "premiere_oubliee":
            origines = origines[1:]
        if defaut == "desordre":
            origines = origines[::-1]
        essais = []
        passe = list(serie[:origines[0]]) if origines else []
        for o in origines:
            if defaut == "desordre":
                vu = list(serie[:o])
            elif defaut == "fuite":
                vu = list(serie[:o + 1])
            elif defaut == "toute_la_serie":
                vu = list(serie)
            else:
                while len(passe) < o:
                    passe.append(serie[len(passe)])
                vu = passe
            if defaut == "double_appel":
                methode(vu, horizon)
            prevus = methode(vu, 1 if defaut == "horizon_un" else horizon)
            cible = o + horizon - 1
            prevu = prevus[0] if defaut in ("premier", "horizon_un") else prevus[horizon - 1]
            essais.append(Essai(o - 1 if defaut == "origine_index" else o, cible, prevu, serie[cible]))
        if defaut == "desordre":
            essais.reverse()
        if defaut == "trois":
            return [(e.origine, e.prevu, e.reel) for e in essais]
        return essais

    return validation_glissante


def test_validation_glissante_juste_ecrite_autrement():
    _reussi("validation_glissante", _validation())


@pytest.mark.parametrize("defaut,morceaux", [
    ("fuite", ["serie[:o + 1]", "fuite du futur"]),
    ("toute_la_serie", ["TOUTE la série"]),
    ("premier", ["DERNIÈRE de la liste"]),
    ("derniere_oubliee", ["len(serie) − horizon (incluse"]),
    ("premiere_oubliee", ["`debut` elle-même"]),
    ("origine_index", ["NOMBRE de mois connus"]),
    ("horizon_un", ["horizon=1 au lieu de"]),
    ("double_appel", ["UN appel par origine"]),
    ("desordre", ["désordre"]),
    ("sans_controle", ["ValueError"]),
    ("trois", ["Essai(origine, cible, prevu, reel)"]),
])
def test_validation_glissante_erreurs_classiques(defaut, morceaux):
    _echec("validation_glissante", _validation(defaut), *morceaux)


def test_validation_glissante_comme_timeseriessplit():
    """À horizon h, l'essai d'origine o, c'est le pli de TimeSeriesSplit(gap=h − 1, test_size=1) qui teste o + h − 1."""
    from sklearn.model_selection import TimeSeriesSplit

    alea = random.Random(4)
    for _ in range(40):
        n = alea.randint(3, 40)
        horizon = alea.randint(1, min(4, n - 2))
        debut = alea.randint(1, n - horizon - 1)                    # au moins 2 essais : TimeSeriesSplit en exige 2
        serie = generer_serie(n, graine=alea.randint(0, 999))
        essais = corrige.validation_glissante(serie, _regle, debut, horizon)
        plis = list(TimeSeriesSplit(n_splits=len(essais), gap=horizon - 1, test_size=1).split(serie))
        assert [(len(train), int(test[0])) for train, test in plis] == [(e.origine, e.cible) for e in essais]
        assert all(list(train) == list(range(e.origine)) for (train, _), e in zip(plis, essais))
        assert [tuple(e) for e in essais] == _ref_validation(serie, _regle, debut, horizon)


# --------------------------------------------------------------------------- Défi 205 : marge_de_securite


def _marge(defaut=None):
    """Écrite autrement (un plafond `math.ceil(niveau * n / 100)`, exact, ou heapq), avec UN défaut."""

    def marge_de_securite(serie, methode, debut, horizon, niveau=90):
        accepte = (int, float) if defaut == "fraction_acceptee" else int
        if not isinstance(niveau, accepte) or isinstance(niveau, bool) or not 0 < niveau < 100:
            raise ValueError("niveau")
        essais = corrige.validation_glissante(serie, methode, debut, 1 if defaut == "horizon_un" else horizon)
        if defaut == "absolu":
            ecarts = [abs(e.reel - e.prevu) for e in essais]
        elif defaut == "signe":
            ecarts = [e.prevu - e.reel for e in essais]
        else:
            ecarts = [e.reel - e.prevu for e in essais]
        n = len(ecarts)
        if defaut == "interpolation":
            marge = float(np.quantile(ecarts, niveau / 100))
        elif defaut == "plafond_float":
            marge = sorted(ecarts)[math.ceil(niveau / 100 * n) - 1]
        elif defaut == "plancher":
            marge = sorted(ecarts)[int(niveau / 100 * n) - 1]
        elif defaut == "k_suivant":
            marge = sorted(ecarts)[min(math.ceil(niveau * n / 100), n - 1)]
        elif defaut == "heapq":
            marge = heapq.nsmallest(math.ceil(niveau * n / 100), ecarts)[-1]
        elif defaut == "numpy":
            marge = float(np.quantile(ecarts, niveau / 100, method="inverted_cdf"))
        else:
            marge = sorted(ecarts)[math.ceil(niveau * n / 100) - 1]     # niveau * n est entier : / 100 est exact
        if defaut == "prevision_derniere_origine":
            prevision = essais[-1].prevu
        else:
            prevus = methode(list(serie), horizon)
            prevision = prevus[0] if defaut == "prevision_premiere" else prevus[horizon - 1]
        borne = prevision if defaut == "borne_oubliee" else prevision + marge
        return Marge(prevision, marge, borne)

    return marge_de_securite


@pytest.mark.parametrize("defaut", [None, "heapq"])
def test_marge_de_securite_juste_ecrite_autrement(defaut):
    _reussi("marge_de_securite", _marge(defaut))


@pytest.mark.parametrize("defaut,morceaux", [
    ("absolu", ["SANS valeur absolue"]),
    ("signe", ["prévu − réel : votre marge protège contre l'erreur inverse"]),
    ("interpolation", ["sans interpolation"]),
    ("plafond_float", ["ENTIERS", "7.000000000000001"]),
    ("numpy", ["ENTIERS", "inverted_cdf"]),
    ("plancher", ["PLAFOND"]),
    ("k_suivant", ["tries[k - 1]"]),
    ("horizon_un", ["celles à 1 mois"]),
    ("prevision_derniere_origine", ["TOUTE la série"]),
    ("prevision_premiere", ["DERNIÈRE de la liste"]),
    ("fraction_acceptee", ["ValueError", "ENTIER"]),
    ("borne_oubliee", ["borne", "prevision + marge"]),
])
def test_marge_de_securite_erreurs_classiques(defaut, morceaux):
    _echec("marge_de_securite", _marge(defaut), *morceaux)


def test_marge_qui_appelle_une_validation_boguee_le_dit():
    validation_boguee = _validation("derniere_oubliee")

    def marge_de_securite(serie, methode, debut, horizon, niveau=90):
        if not isinstance(niveau, int) or not 1 <= niveau <= 99:
            raise ValueError("niveau")
        ecarts = sorted(e.reel - e.prevu for e in validation_boguee(serie, methode, debut, horizon))
        prevision = methode(list(serie), horizon)[horizon - 1]
        marge = ecarts[(niveau * len(ecarts) + 99) // 100 - 1]
        return Marge(prevision, marge, prevision + marge)

    assert _evaluer("marge_de_securite", marge_de_securite).statut == ECHEC
    globales = {"validation_glissante": validation_boguee}
    assert "défi 204" in checks._aide_204(SimpleNamespace(__globals__=globales))
    assert checks._aide_204(SimpleNamespace(__globals__={"validation_glissante": corrige.validation_glissante})) == ""


def test_marge_qui_appelle_une_validation_pas_encore_ecrite():
    def validation_glissante(serie, methode, debut, horizon=1):
        raise NotImplementedError

    def marge_de_securite(serie, methode, debut, horizon, niveau=90):
        return validation_glissante(serie, methode, debut, horizon)

    message = _echec("marge_de_securite", marge_de_securite)
    assert "validation_glissante" in message and "204" in message


def test_marge_comme_numpy_sauf_quand_numpy_arrondit_en_float():
    from fractions import Fraction

    alea = random.Random(5)
    for _ in range(100):
        n = alea.randint(3, 40)
        serie = generer_serie(n, graine=alea.randint(0, 999))
        horizon = alea.randint(1, min(3, n - 1))
        debut = alea.randint(1, n - horizon)
        niveau = alea.randint(1, 99)
        m = corrige.marge_de_securite(serie, _regle, debut, horizon, niveau)
        ecarts = [e.reel - e.prevu for e in corrige.validation_glissante(serie, _regle, debut, horizon)]
        k = math.ceil(Fraction(niveau, 100) * len(ecarts))                          # le rang, en nombres exacts
        assert m.marge == sorted(ecarts)[k - 1]
        if (niveau, len(ecarts)) not in ((28, 25), (14, 50), (56, 25)):
            assert m.marge == float(np.quantile(ecarts, niveau / 100, method="inverted_cdf"))
        assert tuple(m) == _ref_marge(serie, _regle, debut, horizon, niveau)
        assert sum(e <= m.marge for e in ecarts) >= niveau * len(ecarts) / 100      # la couverture promise
    assert math.ceil(28 / 100 * 25) == 8 and (28 * 25 + 99) // 100 == 7          # le piège annoncé existe…
    assert np.quantile(np.arange(25.0), 0.28, method="inverted_cdf") == 7.0     # … numpy y tombe (le 8e, pas le 7e)


# --------------------------------------------------------------------------- Outils communs et exemples


def test_les_exemples_racontent_l_histoire_annoncee():
    departs = departs_exemple()
    resultats = {}
    for nom, methode in (("naive", lambda p, h: corrige.prevision_naive(p, h)),
                         ("an_dernier", lambda p, h: corrige.prevision_naive(p, h, saison=12)),
                         ("lissage", lambda p, h: corrige.lissage_exponentiel(p, 0.3, h).previsions),
                         ("holt", lambda p, h: corrige.holt(p, 0.3, 0.1, h).previsions)):
        essais = corrige.validation_glissante(departs, methode, 24, 3)
        resultats[nom] = (corrige.erreurs([e.reel for e in essais], [e.prevu for e in essais]),
                          corrige.marge_de_securite(departs, methode, 24, 3, 90))
    erreurs_an_dernier, marge_an_dernier = resultats["an_dernier"]
    assert min(resultats, key=lambda nom: resultats[nom][0].mae) == "an_dernier"      # la plus petite MAE…
    assert erreurs_an_dernier.biais == -erreurs_an_dernier.mae                        # … toujours trop basse
    assert min(resultats, key=lambda nom: resultats[nom][1].borne) == "an_dernier"    # et la marge la corrige
    assert math.ceil(marge_an_dernier.borne / DOSSIERS_PAR_CONSEILLER) == 13
    assert math.ceil(marge_an_dernier.prevision / DOSSIERS_PAR_CONSEILLER) == 12
    assert len(departs) == len(mois_exemple()) == 36 and mois_exemple()[26] == "2025-03"
    assert departs_exemple() is not departs


def test_generer_serie_est_reproductible_et_exacte():
    a, b = generer_serie(200, graine=7), generer_serie(200, graine=7)
    assert a == b and a != generer_serie(200, graine=8)
    assert all(x * 16 == int(x * 16) for x in a)
    assert generer_serie(4, saison=1, tendance=1.0, bruit=0.0) == [400.0, 401.0, 402.0, 403.0]
    plate = generer_serie(120, graine=1, tendance=0.0, bruit=0.0)
    assert plate[:12] == plate[12:24] and abs(statistics.fmean(plate[:12]) - 400) < 1 / 16


def test_les_references_rejouent_les_bonnes_erreurs():
    assert _ref_naive([10, 20, 30, 40, 50, 60], 3, 4, "a_rebours") == [60.0, 50.0, 40.0]
    assert _ref_naive([10, 20, 30, 40, 50, 60], 3, 4, "decale") == [40.0, 50.0, 60.0]
    assert _ref_ses([10.0, 20.0, 20.0], 0.5, 1, "fuite")[0] == [None, 15.0, 17.5]
    assert _ref_holt([10.0, 12.0, 20.0], 0.5, 0.5, 1, "ancien_ecrase")[3] == 1.0
    assert _ref_validation([5.0, 7.0, 6.0, 9.0], _regle, 2, 1, "derniere_oubliee") == [(2, 2, 7.0 + 0.5 + 2 / 64, 6.0)]
    assert _ref_erreurs([10, 20], [13, 16], "biais_inverse")[2] == 0.5


def test_l_espionne_voit_ce_qu_on_lui_donne():
    espionne = checks._Espionne()
    passe = [1.0, 2.0]
    assert espionne(passe, 2) == [2.0 + 0.5 + 2 / 64, 2.0 + 1.0 + 2 / 64]
    passe.append(3.0)                                    # elle a gardé une COPIE de ce qu'elle a reçu
    assert espionne.appels == [([1.0, 2.0], 2)]
    with pytest.raises(ValueError):
        espionne([], 1)


# --------------------------------------------------------------------------- La série


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(200, 206))
    assert any(s is SERIE for s in series())            # enregistrée dans defis/verif.py


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_33 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(previsions)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_33 import *` (dans les exemples) doit donner les fonctions du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 35 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    serie = [10.0, 14.0, 11.0, 15.0, 12.0, 16.0]
    reels, prevus = [1.0, 2.0], [2.0, 2.0]
    copies = copy.deepcopy((serie, reels, prevus))
    corrige.prevision_naive(serie, 4, saison=2)
    corrige.erreurs(reels, prevus)
    corrige.lissage_exponentiel(serie, 0.5, 2)
    corrige.holt(serie, 0.5, 0.5, 2)
    corrige.validation_glissante(serie, _regle, 2, 2)
    corrige.marge_de_securite(serie, _regle, 2, 2, 50)
    assert (serie, reels, prevus) == copies

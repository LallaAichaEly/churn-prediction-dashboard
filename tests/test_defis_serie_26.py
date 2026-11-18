"""Série 26 (conclure sans se tromper) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 158`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées ;
- les exemples des docstrings, les outils communs et le corrigé sont justes.
"""

from __future__ import annotations

import doctest
import itertools
import math
import random
import sys
import warnings
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.corrige_serie_26 as corrige  # noqa: E402
from defis.checks_serie_26 import SERIE  # noqa: E402
from defis.incertitude import (  # noqa: E402
    EFFET_REEL,
    attentes_support,
    auc,
    campagne_exemple,
    notes_deux_scripts,
    p_valeur_exacte,
    p_valeurs_segments,
    puissance_simulee,
    scores_deux_modeles,
)
from defis.verif import A_FAIRE, ECHEC, ERREUR, REUSSI, evaluer, evaluer_defi  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}), DEFIS[nom])


def _echec(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


# --------------------------------------------------------------------------- Défi 158 : intervalle_bootstrap


def _bootstrap(defaut=None):
    """Un bootstrap écrit autrement que le corrigé (`rng.choice`), avec UN défaut au choix."""

    def intervalle_bootstrap(valeurs, statistique=np.mean, n_tirages=2000, confiance=0.95, graine=0):
        if defaut == "etiquettes" and isinstance(valeurs, pd.Series):
            x = valeurs
        else:
            x = np.asarray(valeurs, dtype=float)
        if len(x) == 0:
            raise ValueError("vide")
        if defaut != "nan_accepte" and np.isnan(np.asarray(x, dtype=float)).any():
            raise ValueError("NaN")
        if defaut != "confiance_non_verifiee" and not 0 < confiance < 1:
            raise ValueError("confiance")
        if defaut == "confiance_ignoree":
            confiance = 0.95
        if defaut == "graine_globale":
            np.random.seed(graine)
        rng = np.random.default_rng() if defaut == "sans_graine" else np.random.default_rng(graine)
        if defaut == "melange_en_place":
            rng.shuffle(valeurs)
        n = len(x)
        resultats = []
        for _ in range(n_tirages):
            if defaut == "rng_dans_boucle":
                rng = np.random.default_rng(graine)
            if defaut == "sans_remise":
                tirage = rng.permutation(np.asarray(x))
            elif defaut == "graine_globale":
                tirage = np.random.choice(x, size=n, replace=True)
            elif defaut == "taille_n_tirages":
                tirage = rng.choice(x, size=n_tirages, replace=True)
            elif defaut == "etiquettes":
                tirage = x[rng.integers(0, n, size=n)]
            else:
                tirage = rng.choice(x, size=n, replace=True)
            if defaut == "statistique_ignoree":
                resultats.append(np.mean(tirage))
            elif defaut == "liste":
                resultats.append(statistique(list(tirage)))
            else:
                resultats.append(statistique(tirage))
        if defaut == "quatre_vingt_dix":
            bornes = [1 - confiance, confiance]
        elif defaut == "percentiles":
            bornes = [100 * (1 - confiance) / 2, 100 * (1 + confiance) / 2]
        else:
            bornes = [(1 - confiance) / 2, (1 + confiance) / 2]
        bas, haut = np.quantile(resultats, bornes)
        return float(bas), float(haut)

    return intervalle_bootstrap


@pytest.mark.parametrize("defaut,mots", [
    ("sans_remise", ("largeur nulle", "SANS remise")),
    ("rng_dans_boucle", ("largeur nulle", "DANS la boucle")),
    ("graine_globale", ("GLOBAL", "np.random.default_rng(graine)")),
    ("sans_graine", ("même graine",)),
    ("quatre_vingt_dix", ("90 %", "0.025")),
    ("percentiles", ("ValueError", "np.percentile")),
    ("taille_n_tirages", ("n = 400", "trop étroit")),
    ("statistique_ignoree", ("MOYENNE", "statistique(tirage)")),
    ("liste", ("TypeError", "TABLEAU")),
    ("etiquettes", ("POSITION",)),
    ("melange_en_place", ("modifié",)),
    ("nan_accepte", ("NaN", "devrait lever ValueError")),
    ("confiance_non_verifiee", ("confiance=1.0",)),
    ("confiance_ignoree", ("80 %", "confiance")),
])
def test_bootstrap_erreurs_classiques(defaut, mots):
    _echec("intervalle_bootstrap", _bootstrap(defaut), *mots)


def test_bootstrap_sans_defaut_la_fabrique_est_juste():
    _reussi("intervalle_bootstrap", _bootstrap())


def test_bootstrap_autres_ecritures_justes():
    def vectorise(valeurs, statistique=np.mean, n_tirages=2000, confiance=0.95, graine=0):
        x = np.asarray(valeurs, dtype=float)
        if x.size == 0 or np.isnan(x).any() or not 0 < confiance < 1:
            raise ValueError("entrée invalide")
        rng = np.random.default_rng(graine)
        tirages = x[rng.integers(0, len(x), size=(n_tirages, len(x)))]   # tous les tirages d'un coup
        stats = np.apply_along_axis(statistique, 1, tirages)
        alpha = 100 * (1 - confiance) / 2
        bas, haut = np.percentile(stats, [alpha, 100 - alpha])
        return bas, haut                                                  # des np.float64 : acceptés

    def trie_a_la_main(valeurs, statistique=np.mean, n_tirages=2000, confiance=0.95, graine=0):
        x = np.array(list(valeurs), dtype=float)
        if not len(x) or any(math.isnan(v) for v in x):
            raise ValueError("entrée invalide")
        if confiance <= 0 or confiance >= 1:
            raise ValueError("confiance")
        rng = np.random.default_rng(graine)
        stats = sorted(statistique(x[rng.integers(0, len(x), len(x))]) for _ in range(n_tirages))
        bas = stats[int((1 - confiance) / 2 * (n_tirages - 1))]
        haut = stats[int(math.ceil((1 + confiance) / 2 * (n_tirages - 1)))]
        return [bas, haut]                                                # une liste de deux nombres : acceptée

    _reussi("intervalle_bootstrap", vectorise)
    _reussi("intervalle_bootstrap", trie_a_la_main)


# --------------------------------------------------------------------------- Défi 159 : p_valeur_permutation


def _permutation(defaut=None):
    """Un test de permutation écrit autrement que le corrigé (on mélange des positions), avec UN défaut."""

    def p_valeur_permutation(a, b, n_permutations=10_000, graine=0):
        xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if len(xa) == 0 or len(xb) == 0:
            raise ValueError("groupe vide")
        if defaut != "nan_accepte" and (np.isnan(xa).any() or np.isnan(xb).any()):
            raise ValueError("NaN")
        if defaut == "graine_globale":
            np.random.seed(graine)
        rng = np.random.default_rng(graine)
        if defaut == "modifie_entree":
            rng.shuffle(xa)                       # « sans effet sur la moyenne »… mais xa EST le tableau reçu
        tous = np.concatenate([xa, xb])
        k = len(tous) // 2 if defaut == "coupe_milieu" else len(xa)
        observe = xb.mean() - xa.mean()
        compte = 0
        for _ in range(n_permutations):
            if defaut == "melange_separe":
                pa, pb = rng.permutation(xa), rng.permutation(xb)
            else:
                positions = np.random.permutation(len(tous)) if defaut == "graine_globale" else \
                    rng.permutation(len(tous))
                pa, pb = tous[positions[:k]], tous[positions[k:]]
            ecart = pb.mean() - pa.mean()
            if defaut == "unilateral":
                compte += ecart >= observe
            elif defaut == "strict":
                compte += abs(ecart) > abs(observe)
            else:
                compte += abs(ecart) >= abs(observe)
        if defaut == "sans_plus_un":
            return compte / n_permutations
        if defaut == "divise_par_n":
            return (compte + 1) / n_permutations
        return (compte + 1) / (n_permutations + 1)

    return p_valeur_permutation


@pytest.mark.parametrize("defaut,mots", [
    ("sans_plus_un", ("renvoie 0", "jamais 0")),
    ("divise_par_n", ("1/999", "n_permutations + 1")),
    ("strict", ("`>`", "`>=`")),
    ("unilateral", ("a > b", "VALEUR ABSOLUE")),
    ("melange_separe", ("ENSEMBLE",)),
    ("coupe_milieu", ("MILIEU", "len(a) = 3", "p_valeur_exacte")),
    ("graine_globale", ("GLOBAL",)),
    ("modifie_entree", ("modifié",)),
    ("nan_accepte", ("NaN", "devrait lever ValueError")),
])
def test_permutation_erreurs_classiques(defaut, mots):
    _echec("p_valeur_permutation", _permutation(defaut), *mots)


def test_permutation_sans_defaut_la_fabrique_est_juste():
    _reussi("p_valeur_permutation", _permutation())


def test_permutation_autres_ecritures_justes():
    def vectorisee(a, b, n_permutations=10_000, graine=0):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if not len(a) or not len(b) or np.isnan(np.r_[a, b]).any():
            raise ValueError("entrée invalide")
        rng = np.random.default_rng(graine)
        tous = np.r_[a, b]
        ordres = rng.random((n_permutations, len(tous))).argsort(axis=1)   # une permutation par ligne
        melanges = tous[ordres]
        ecarts = np.abs(melanges[:, len(a):].mean(axis=1) - melanges[:, :len(a)].mean(axis=1))
        return (np.sum(ecarts >= abs(b.mean() - a.mean())) + 1) / (n_permutations + 1)

    def avec_choice(a, b, n_permutations=10_000, graine=0):
        a, b = list(map(float, a)), list(map(float, b))
        if not a or not b or any(math.isnan(v) for v in a + b):
            raise ValueError("entrée invalide")
        rng = np.random.default_rng(graine)
        tous = np.array(a + b)
        observe = abs(sum(b) / len(b) - sum(a) / len(a))
        compte = 0
        for _ in range(n_permutations):
            m = rng.choice(tous, size=len(tous), replace=False)
            compte += bool(abs(m[len(a):].mean() - m[:len(a)].mean()) >= observe)
        return (compte + 1) / (n_permutations + 1)

    _reussi("p_valeur_permutation", vectorisee)
    _reussi("p_valeur_permutation", avec_choice)


# --------------------------------------------------------------------------- Défi 160 : taille_echantillon


def _taille(defaut=None):
    """La formule de l'énoncé avec scipy plutôt que NormalDist, et UN défaut au choix."""
    from scipy.stats import norm

    def taille_echantillon(p_temoin, baisse, alpha=0.05, puissance=0.80):
        if defaut != "sans_validation":
            for v in (p_temoin, baisse, alpha, puissance):
                if not 0 < v < 1:
                    raise ValueError("hors de ]0, 1[")
            if baisse >= p_temoin:
                raise ValueError("churn négatif")
        p_offre = {"hausse": p_temoin + baisse, "relative": p_temoin * (1 - baisse)}.get(defaut, p_temoin - baisse)
        p_moyen = (p_temoin + p_offre) / 2
        z_alpha = norm.ppf(1 - alpha) if defaut == "unilateral" else norm.ppf(1 - alpha / 2)
        z_beta = {"beta_inverse": norm.ppf(1 - puissance), "puissance_brute": puissance}.get(defaut,
                                                                                           norm.ppf(puissance))
        delta = abs(p_temoin - p_offre)
        if defaut == "simple_commune":
            n = (z_alpha + z_beta) ** 2 * 2 * p_moyen * (1 - p_moyen) / delta ** 2
        elif defaut == "simple_separee":
            n = (z_alpha + z_beta) ** 2 * (p_temoin * (1 - p_temoin) + p_offre * (1 - p_offre)) / delta ** 2
        else:
            n = (z_alpha * math.sqrt(2 * p_moyen * (1 - p_moyen))
                 + z_beta * math.sqrt(p_temoin * (1 - p_temoin) + p_offre * (1 - p_offre))) ** 2 / delta ** 2
        if defaut == "arrondi":
            return round(n)
        if defaut == "tronque":
            return int(n)
        if defaut == "flottant":
            return np.ceil(n)
        if defaut == "total":
            return 2 * math.ceil(n)
        return int(np.ceil(n))

    return taille_echantillon


@pytest.mark.parametrize("defaut,mots", [
    ("unilateral", ("UNILATÉRAL", "1 - alpha / 2", "puissance", "détectée")),
    ("beta_inverse", ("NÉGATIF",)),
    ("puissance_brute", ("QUANTILE",)),
    ("arrondi", ("VERS LE HAUT", "math.ceil")),
    ("tronque", ("int(...)", "math.ceil")),
    ("flottant", ("ENTIER", "math.ceil")),
    ("total", ("PAR GROUPE", "de trop")),
    ("simple_commune", ("formule simplifiée", "même variance")),
    ("simple_separee", ("formule simplifiée", "p_moyen")),
    ("hausse", ("BAISSER",)),
    ("relative", ("POINTS", "RELATIVE")),
    ("sans_validation", ("taille_echantillon(0.30, 0.30)", "churn nul")),
])
def test_taille_erreurs_classiques(defaut, mots):
    _echec("taille_echantillon", _taille(defaut), *mots)


def test_taille_sans_defaut_la_fabrique_est_juste():
    _reussi("taille_echantillon", _taille())


def test_taille_les_cas_ne_sont_pas_au_bord_d_un_entier():
    """Sinon, une écriture juste de la formule (dans un autre ordre) pourrait tomber à 1 près à côté."""
    from defis.checks_serie_26 import _CAS_TAILLE, _n_brut

    z = NormalDist().inv_cdf
    fractions = [_n_brut(p, p - b, z(1 - a / 2), z(w)) % 1 for p, b, a, w in _CAS_TAILLE]
    assert all(0.04 < f < 0.96 for f in fractions)
    assert any(f < 0.5 for f in fractions) and any(f > 0.5 for f in fractions)  # round ≠ ceil, int ≠ ceil


def test_la_puissance_simulee_confirme_la_formule():
    for p, b, a, w in [(0.30, 0.05, 0.05, 0.80), (0.10, 0.02, 0.05, 0.80), (0.50, 0.10, 0.01, 0.90)]:
        n = corrige.taille_echantillon(p, b, a, w)
        assert abs(puissance_simulee(n, p, p - b, a, n_experiences=20_000) - w) < 0.015


# --------------------------------------------------------------------------- Défi 161 : benjamini_hochberg


def _bh(defaut=None):
    """Benjamini-Hochberg en Python pur (sorted + parcours à l'envers), avec UN défaut au choix."""

    def benjamini_hochberg(p_valeurs, q=0.05):
        if defaut == "sort_en_place":
            p_valeurs.sort()
        p = [float(v) for v in p_valeurs]
        if defaut != "nan_accepte" and any(math.isnan(v) or not 0 <= v <= 1 for v in p):
            raise ValueError("p-valeur invalide")
        if not 0 < q < 1:
            raise ValueError("q")
        if defaut == "q_ignore":
            q = 0.05
        m = len(p)
        if defaut == "bonferroni":
            return [v <= q / m for v in p]
        if defaut == "brut":
            return [v <= q for v in p]
        if defaut == "quadratique":
            rangs = [sum(1 for w in p if w < v) + 1 for v in p]     # le rang de chacune : une boucle dans une boucle
            ordre = [i for _, i in sorted(zip(rangs, range(m)))]
        else:
            ordre = sorted(range(m), key=lambda i: p[i])
        k = 0
        if defaut == "descendante":
            for rang, i in enumerate(ordre, start=1):
                if p[i] > rang / m * q:
                    break
                k = rang
        else:
            for rang in range(m, 0, -1):
                seuil = rang / m * q
                if (p[ordre[rang - 1]] < seuil) if defaut == "strict" else (p[ordre[rang - 1]] <= seuil):
                    k = rang
                    break
        decisions = [False] * m
        for i in ordre[:k]:
            decisions[i] = True
        if defaut == "trie":
            return [decisions[i] for i in ordre]
        if defaut == "tableau":
            return np.array(decisions)
        return decisions

    return benjamini_hochberg


@pytest.mark.parametrize("defaut,mots", [
    ("descendante", ("PREMIÈRE", "PLUS GRAND k")),
    ("bonferroni", ("Bonferroni",)),
    ("brut", ("sans correction",)),
    ("strict", ("PILE", "`<=`")),
    ("trie", ("ordre REÇU",)),
    ("sort_en_place", ("modifié", ".sort()")),
    ("tableau", ("tableau NumPy", ".tolist()")),
    ("q_ignore", ("q = 0.05 au lieu",)),
    ("quadratique", ("trop lent", "boucle dans une boucle")),
    ("nan_accepte", ("NaN", "devrait lever ValueError")),
])
def test_bh_erreurs_classiques(defaut, mots):
    _echec("benjamini_hochberg", _bh(defaut), *mots)


def test_bh_sans_defaut_la_fabrique_est_juste():
    _reussi("benjamini_hochberg", _bh())


def test_bh_autre_ecriture_juste_avec_pandas():
    def avec_pandas(p_valeurs, q=0.05):
        p = pd.Series(np.asarray(p_valeurs, dtype=float))
        if p.isna().any() or ((p < 0) | (p > 1)).any() or not 0 < q < 1:
            raise ValueError("entrée invalide")
        if p.empty:
            return []
        rangs = p.rank(method="first")
        passe = p <= rangs / len(p) * q
        if not passe.any():
            return [False] * len(p)
        seuil = p[passe].max()                 # la plus grande p-valeur qui passe son seuil
        return (p <= seuil).tolist()

    _reussi("benjamini_hochberg", avec_pandas)


def test_bh_la_reference_suit_la_definition():
    """La référence du correcteur, comparée à la définition prise au mot (tous les k essayés)."""
    from defis.checks_serie_26 import _ref_bh

    alea = random.Random(5)
    for _ in range(500):
        m = alea.randint(1, 12)
        p = [round(alea.random() ** 3, 2) for _ in range(m)]
        q = alea.choice([0.05, 0.1, 0.2])
        tries = sorted(p)
        k = max([j for j in range(1, m + 1) if tries[j - 1] <= q * j / m], default=0)
        attendu = [v <= tries[k - 1] if k else False for v in p]
        assert _ref_bh(p, q) == attendu == corrige.benjamini_hochberg(p, q)


# --------------------------------------------------------------------------- Défi 162 : comparer_auc


def _comparer(defaut=None, aire=auc):
    """Le bootstrap apparié, avec UN défaut au choix (l'AUC rapide par défaut : roc_auc_score coûte 2 ms par appel)."""

    def comparer_auc(y, proba_a, proba_b, n_tirages=1000, confiance=0.95, graine=0):
        if defaut == "etiquettes" and isinstance(y, pd.Series):
            y_ = y
        else:
            y_ = np.asarray(y)
        a, b = np.asarray(proba_a, dtype=float), np.asarray(proba_b, dtype=float)
        if not len(y_) == len(a) == len(b) or np.isnan(a).any() or np.isnan(b).any():
            raise ValueError("entrée invalide")
        if not set(np.unique(y_)) <= {0, 1} or len(np.unique(y_)) < 2:
            raise ValueError("y")
        if defaut != "confiance_ignoree" and not 0 < confiance < 1:
            raise ValueError("confiance")
        if defaut == "confiance_ignoree":
            confiance = 0.95
        if defaut == "graine_globale":
            np.random.seed(graine)
        rng = np.random.default_rng(graine)
        n, ecarts, ignores = len(y_), [], 0
        for _ in range(n_tirages):
            if defaut == "sans_remise":
                i = rng.permutation(n)
            elif defaut == "graine_globale":
                i = np.random.randint(0, n, size=n)
            else:
                i = rng.integers(0, n, size=n)
            j = rng.integers(0, n, size=n) if defaut == "non_apparie" else i
            yi, yj = y_[i], y_[j]
            if defaut != "plante_une_classe" and (len(np.unique(yi)) < 2 or len(np.unique(yj)) < 2):
                if defaut != "sans_compter":
                    ignores += 1
                continue
            ecarts.append(aire(yj, b[j]) - aire(yi, a[i]))
        bornes = [1 - confiance, confiance] if defaut == "quatre_vingt_dix" else [(1 - confiance) / 2,
                                                                                (1 + confiance) / 2]
        bas, haut = np.quantile(ecarts, bornes)
        ecart = aire(y_, b) - aire(y_, a)
        if defaut == "signe":
            ecart = -ecart
        if defaut == "moyenne_tirages":
            ecart = float(np.mean(ecarts))
        return {"ecart": ecart, "bas": bas, "haut": haut, "tirages_ignores": ignores}

    return comparer_auc


@pytest.mark.parametrize("defaut,mots", [
    ("non_apparie", ("trop large", "DIFFÉRENTS", "UNE fois les positions")),
    ("signe", ("AUC(b) - AUC(a)",)),
    ("moyenne_tirages", ("TOUS les clients", "pas la moyenne des tirages")),
    ("plante_une_classe", ("ValueError", "AUCUN client parti")),
    ("sans_compter", ("vaut 0", "ne comptez pas")),
    ("sans_remise", ("largeur presque nulle", "AVEC remise")),
    ("graine_globale", ("GLOBAL",)),
    ("etiquettes", ("POSITION",)),
    ("quatre_vingt_dix", ("trop étroit", "90 %")),
    ("confiance_ignoree", ("confiance",)),
])
def test_comparer_auc_erreurs_classiques(defaut, mots):
    _echec("comparer_auc", _comparer(defaut), *mots)


def test_comparer_auc_sans_defaut_la_fabrique_est_juste():
    _reussi("comparer_auc", _comparer())


def test_comparer_auc_avec_roc_auc_score_est_accepte():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _reussi("comparer_auc", _comparer(aire=roc_auc_score))


def test_comparer_auc_une_auc_nan_est_expliquee():
    """Les versions récentes de scikit-learn renvoient NaN (avec un avertissement) pour une seule classe, sans lever."""

    def auc_ou_nan(y, score):
        return float("nan") if len(np.unique(y)) < 2 else auc(y, score)

    _echec("comparer_auc", _comparer("plante_une_classe", aire=auc_ou_nan), "NaN", "AUCUN client parti",
           "roc_auc_score")


def test_comparer_auc_l_ecart_non_apparie_est_bien_plus_large():
    """Ce que le défi 162 enseigne, mesuré : sur ces données, tirer A et B séparément multiplie la largeur par ~5."""
    from defis.checks_serie_26 import _ref_comparer

    df = scores_deux_modeles()
    y, a, b = df["churn"].to_numpy(), df["champion"].to_numpy(), df["challenger"].to_numpy()
    apparie = _ref_comparer(y, a, b, 1500, 0.95, 1)
    separe = _ref_comparer(y, a, b, 1500, 0.95, 1, apparie=False)
    assert (separe["haut"] - separe["bas"]) > 3.5 * (apparie["haut"] - apparie["bas"])
    assert apparie["bas"] > 0 > separe["bas"]  # apparié : b fait mieux ; séparé : « on ne peut pas conclure »


# --------------------------------------------------------------------------- Défi 163 : effet_cuped


def _cuped(defaut=None):
    """CUPED écrit avec pandas et une régression (np.polyfit) pour θ, avec UN défaut au choix."""

    def effet_cuped(y, x_avant, traite):
        if defaut == "etiquettes":
            df = pd.DataFrame({"y": y, "x": x_avant, "t": traite})
        else:
            df = pd.DataFrame({"y": np.asarray(y, dtype=float), "x": np.array(x_avant, dtype=float),
                               "t": np.asarray(traite)})
        if not len(np.asarray(y)) == len(np.asarray(x_avant)) == len(np.asarray(traite)) or df["y"].isna().any():
            raise ValueError("entrée invalide")
        if defaut != "deux_accepte" and not df["t"].isin([0, 1]).all():
            raise ValueError("traite")
        t = (df["t"] != 0).to_numpy()
        if defaut != "un_client" and (t.sum() < 2 or (~t).sum() < 2):
            raise ValueError("au moins 2 par groupe")
        if defaut == "en_place":
            x = np.asarray(x_avant, dtype=float)
            x[np.isnan(x)] = np.nanmean(x)
        else:
            x = df["x"].to_numpy(copy=True)
            if defaut == "nan_zero":
                x[np.isnan(x)] = 0.0
            elif defaut != "vide" and np.isnan(x).all():
                x[:] = 0.0
            else:
                x[np.isnan(x)] = np.nanmean(x)
        yv = df["y"].to_numpy()
        if defaut != "constant_plante" and np.ptp(x) == 0:
            theta = 0.0
        elif defaut == "theta_y":
            theta = np.cov(x, yv)[0, 1] / np.var(yv, ddof=1)
        elif defaut == "ddof":
            theta = np.cov(x, yv)[0, 1] / np.var(x)
        elif defaut == "theta_temoins":
            theta = np.polyfit(x[~t], yv[~t], 1)[0]
        else:
            theta = np.cov(x, yv)[0, 1] / np.var(x, ddof=1)
        centre = np.where(t, x[t].mean(), x[~t].mean()) if defaut == "centrage_groupe" else x.mean()
        ajuste = yv - theta * (x - centre)
        ddof = 0 if defaut == "se_ddof0" else 1

        def estimer(v):
            ecart = v[t].mean() - v[~t].mean()
            if defaut == "se_commune":
                erreur = math.sqrt(v.var(ddof=1) * (1 / t.sum() + 1 / (~t).sum()))
            else:
                erreur = math.sqrt(v[t].var(ddof=ddof) / t.sum() + v[~t].var(ddof=ddof) / (~t).sum())
            return (-ecart if defaut == "signe" else ecart), erreur

        effet, erreur = estimer(ajuste)
        brut, erreur_brute = estimer(yv)
        return {"effet": effet, "erreur_type": erreur, "effet_brut": brut, "erreur_type_brute": erreur_brute,
                "theta": theta}

    return effet_cuped


@pytest.mark.parametrize("defaut,mots", [
    ("signe", ("traités MOINS témoins",)),
    ("nan_zero", ("score de 0", "MOYENNE")),
    ("theta_y", ("variance de x",)),
    ("ddof", ("n / (n - 1)", "même ddof")),
    ("theta_temoins", ("TOUS les clients",)),
    ("centrage_groupe", ("SA moyenne", "effet brut")),
    ("se_ddof0", ("ddof=1",)),
    ("se_commune", ("CHAQUE groupe",)),
    ("en_place", ("modifié", "COPIE")),
    ("constant_plante", ("score constant", "x.max() == x.min()")),
    ("vide", ("aucun score connu", "VIDE")),
    ("etiquettes", ("ÉTIQUETTES", "POSITION")),
    ("deux_accepte", ("[2, 0, …]", "devrait lever ValueError")),
    ("un_client", ("un seul client traité", "au moins 2")),
])
def test_cuped_erreurs_classiques(defaut, mots):
    # les versions boguées divisent par 0 ou font la moyenne de rien : on fait taire leurs RuntimeWarning
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _echec("effet_cuped", _cuped(defaut), *mots)


def test_cuped_sans_defaut_la_fabrique_est_juste():
    _reussi("effet_cuped", _cuped())


def test_cuped_les_clients_sans_score_ne_changent_pas_theta():
    """Remplir les NaN par la moyenne puis calculer θ sur tous = calculer θ sur les seuls clients connus."""
    df = campagne_exemple()
    connus = df.dropna()
    theta_connus = np.cov(connus["score_avant"], connus["churn"])[0, 1] / connus["score_avant"].var()
    r = corrige.effet_cuped(df["churn"], df["score_avant"], df["offre"])
    assert math.isclose(r["theta"], theta_connus, rel_tol=1e-12)


def test_cuped_est_sans_biais_et_plus_precis_en_moyenne():
    """Sur 200 campagnes simulées : CUPED vise aussi juste que l'effet brut, avec moins de dispersion."""
    effets, bruts, erreurs = [], [], []
    for graine in range(200):
        df = campagne_exemple(1500, graine)
        r = corrige.effet_cuped(df["churn"], df["score_avant"], df["offre"])
        effets.append(r["effet"])
        bruts.append(r["effet_brut"])
        erreurs.append(r["erreur_type"])
    assert abs(np.mean(effets) - EFFET_REEL) < 3 * np.std(effets) / math.sqrt(200)
    assert np.std(effets) < 0.92 * np.std(bruts)
    assert abs(np.mean(erreurs) / np.std(effets) - 1) < 0.15  # l'erreur type annoncée est la vraie dispersion


# --------------------------------------------------------------------------- outils communs, corrigé, docstrings


def test_les_tests_du_corrige_passent():
    tests = [f for nom, f in list(vars(corrige).items()) if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6


def test_le_corrige_reussit_tout():
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6


def test_un_plantage_donne_un_message_lisible():
    def plante(*args, **kwargs):
        return 1 / 0

    assert _evaluer("taille_echantillon", plante).statut == ECHEC  # l'appel est enveloppé : on dit quel appel a planté
    assert "ZeroDivisionError" in _evaluer("taille_echantillon", plante).message
    assert _evaluer("taille_echantillon", None).statut == ERREUR


def test_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.incertitude as incertitude
    import defis.serie_26 as eleve

    for nom in ("EFFET_REEL", "attentes_support", "notes_deux_scripts", "p_valeurs_segments", "scores_deux_modeles",
                "campagne_exemple", "p_valeur_exacte", "puissance_simulee", "auc"):
        assert getattr(eleve, nom) is getattr(incertitude, nom), nom


def test_les_exemples_sont_neufs_a_chaque_appel():
    attentes_support()[0] = 99
    assert attentes_support()[0] == 2.0
    notes_deux_scripts()[0].append(1)
    assert len(notes_deux_scripts()[0]) == 14
    p_valeurs_segments().clear()
    assert len(p_valeurs_segments()) == 20
    df = campagne_exemple()
    df.loc[df.index[0], "churn"] = 7
    assert campagne_exemple()["churn"].max() == 1


def test_les_exemples_ont_les_proprietes_annoncees():
    attentes = attentes_support()
    assert np.median(attentes) == 7.5 and np.mean(attentes) > 10.5 and int((attentes > 15).sum()) == 9
    ancien, nouveau = notes_deux_scripts()
    assert (len(ancien), len(nouveau)) == (14, 12)
    assert sum(v < 0.05 for v in p_valeurs_segments().values()) == 5
    df = campagne_exemple()
    assert df["offre"].sum() == 2000 and df["score_avant"].isna().any() and set(df["churn"]) == {0, 1}
    assert df.loc[df["offre"], "score_avant"].mean() > df.loc[~df["offre"], "score_avant"].mean()  # la « malchance »
    scores = scores_deux_modeles()
    assert np.corrcoef(scores["champion"], scores["challenger"])[0, 1] > 0.95
    assert scores_deux_modeles(200, graine=3).equals(scores_deux_modeles(200, graine=3))


def test_auc_est_celle_de_scikit_learn():
    alea = np.random.default_rng(0)
    for _ in range(300):
        n = int(alea.integers(2, 80))
        y = alea.integers(0, 2, n)
        if y.min() == y.max():
            continue
        score = alea.integers(0, 6, n) / 5  # beaucoup d'ex aequo
        assert math.isclose(auc(y, score), roc_auc_score(y, score), abs_tol=1e-12)
    with pytest.raises(ValueError):
        auc([1, 1, 1], [0.1, 0.2, 0.3])


def test_p_valeur_exacte_compte_tous_les_partages():
    a, b = [1.0, 4.0], [2.0, 6.0, 7.0]
    observe = abs(np.mean(b) - np.mean(a))
    tous = a + b
    partages = list(itertools.combinations(range(5), 2))
    extremes = sum(abs(np.mean([tous[i] for i in range(5) if i not in g]) - np.mean([tous[i] for i in g]))
                   >= observe - 1e-9 for g in partages)
    assert p_valeur_exacte(a, b) == extremes / len(partages)
    with pytest.raises(ValueError):
        p_valeur_exacte(list(range(30)), list(range(30)))  # 1,2 × 10¹⁷ partages


def test_puissance_simulee_est_reproductible_et_vaut_alpha_sans_effet():
    assert puissance_simulee(500, 0.3, 0.25, graine=4) == puissance_simulee(500, 0.3, 0.25, graine=4)
    assert abs(puissance_simulee(500, 0.3, 0.3, n_experiences=20_000) - 0.05) < 0.01  # sans effet : 5 % de faux positifs


def test_les_exemples_des_docstrings_sont_justes():
    """Les exemples de defis/serie_26.py et defis/incertitude.py, rejoués avec le corrigé."""
    import defis.incertitude as incertitude
    import defis.serie_26 as eleve

    tests = doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(incertitude)
    globs = dict(vars(eleve))
    for nom in DEFIS:
        globs[nom] = getattr(corrige, nom)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs = dict(globs)
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 40 and resultat.failed == 0


def test_le_correcteur_ne_laisse_pas_le_hasard_global_change():
    """Le correcteur lui-même ne doit pas modifier np.random : sinon il accuserait l'élève à tort."""
    np.random.seed(123)
    avant = np.random.get_state()[1].copy()
    evaluer(SERIE, corrige=True)
    assert np.array_equal(np.random.get_state()[1], avant)

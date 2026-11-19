"""Série 38 (traiter tout le monde de la même façon) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 230`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (pandas avec `dropna=False`, `scipy` pour le V de
  Cramér, `heapq.nlargest`, une comparaison `vp / P >= cible`, un seul appel au modèle pour tous les lots…) ;
- les références du correcteur donnent les mêmes résultats que scikit-learn (matrices de confusion), SciPy (V de
  Cramér) et une recherche exhaustive (seuils de l'égalité des chances) ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (les 65 ans et plus sont signalés et le restent après standardisation, le chèque trahit l'âge, Mme Roux aurait
  eu l'offre avec 40 ans).
"""

from __future__ import annotations

import copy
import doctest
import heapq
import math
import random
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.checks_serie_38 as checks  # noqa: E402
import defis.corrige_serie_38 as corrige  # noqa: E402
import defis.equite as equite  # noqa: E402
from defis.checks_serie_38 import (  # noqa: E402
    SERIE,
    _modele_test,
    _ref_proxies,
    _ref_quatre,
    _ref_seuils,
    _ref_standard,
    _ref_taux,
    _ref_testing,
)
from defis.equite import (  # noqa: E402
    AGES,
    COLONNES,
    SEUIL,
    Bascule,
    Groupe,
    Impact,
    Proxy,
    Reglage,
    Standardisation,
    clients_exemple,
    generer_clients,
    modele,
    selection,
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


def _valider(clients, seuil=SEUIL, parti=True):
    corrige._seuil(seuil)
    for client in clients:
        corrige._score(client)
        if parti:
            corrige._parti(client)


# --------------------------------------------------------------------------- Défi 230 : taux_par_groupe


def _taux_bogue(bug):
    def taux_par_groupe(clients, colonne, seuil=SEUIL):
        _valider(clients, seuil)
        return _ref_taux(clients, colonne, seuil, bug)
    return taux_par_groupe


@pytest.mark.parametrize("bug,morceau", [
    ("strict", "STRICTEMENT"),
    ("nan_a_part", "NaN"),
    ("sans_absents", "disparu"),
    ("population", "TOTAL"),
    ("rappel_precision", "échangé"),
    ("tfp_effectif", "RESTÉS"),
    ("zero", "pas 0.0"),
    ("tri", "ordre"),
])
def test_taux_par_groupe_nomme_l_erreur(bug, morceau):
    _echec("taux_par_groupe", _taux_bogue(bug), morceau)


def _taux_pandas(dropna):
    """Ce qu'écrirait quelqu'un qui connaît pandas : un groupby, puis des sommes."""
    def taux_par_groupe(clients, colonne, seuil=SEUIL):
        _valider(clients, seuil)
        if not clients:
            return {}
        df = pd.DataFrame({"g": [c.get(colonne) for c in clients],
                           "sel": [c["score"] >= seuil for c in clients],
                           "parti": [c["parti"] == 1 for c in clients]})
        df["vp"], df["fp"] = df.sel & df.parti, df.sel & ~df.parti
        resultat = {}
        for g, lot in df.groupby("g", sort=False, dropna=dropna):
            n, s, p, vp, fp = (int(x) for x in (len(lot), lot.sel.sum(), lot.parti.sum(), lot.vp.sum(), lot.fp.sum()))
            cle = None if (isinstance(g, float) and math.isnan(g)) else g
            resultat[cle] = Groupe(n, s, p, vp, fp, s / n, vp / p if p else None,
                                   fp / (n - p) if n - p else None, vp / s if s else None)
        return resultat
    return taux_par_groupe


def test_taux_par_groupe_avec_pandas():
    _reussi("taux_par_groupe", _taux_pandas(dropna=False))
    _echec("taux_par_groupe", _taux_pandas(dropna=True), "dropna=False")


def test_taux_par_groupe_ecrit_par_un_eleve():
    def avec_zero(clients, colonne, seuil=SEUIL):
        resultat = corrige.taux_par_groupe(clients, colonne, seuil)
        return {g: x._replace(tvp=x.tvp or 0.0) for g, x in resultat.items()}

    def en_numpy(clients, colonne, seuil=SEUIL):
        resultat = corrige.taux_par_groupe(clients, colonne, seuil)
        return {g: x._replace(effectif=np.int64(x.effectif)) for g, x in resultat.items()}

    _echec("taux_par_groupe", avec_zero, "pas 0.0")
    _echec("taux_par_groupe", en_numpy, "effectif", "int")
    _echec("taux_par_groupe", lambda clients, colonne, seuil=SEUIL: None, "return")
    _echec("taux_par_groupe", lambda clients, colonne, seuil=SEUIL: list(corrige.taux_par_groupe(clients, colonne)),
           "dict")


def test_taux_par_groupe_qui_modifie_les_clients():
    def taux_par_groupe(clients, colonne, seuil=SEUIL):
        for c in clients:
            if checks._absent(c.get(colonne)):
                c[colonne] = None  # « nettoyer » les données de l'appelant
        return corrige.taux_par_groupe(clients, colonne, seuil)

    _echec("taux_par_groupe", taux_par_groupe, "modifié")


def test_taux_par_groupe_qui_accepte_un_score_impossible():
    def sans_controle(clients, colonne, seuil=SEUIL):
        return _ref_taux(clients, colonne, seuil)

    _echec("taux_par_groupe", sans_controle, "devrait lever")


def test_reference_taux_comme_scikit_learn():
    from sklearn.metrics import confusion_matrix

    clients = generer_clients(3000, graine=5)
    resultat = _ref_taux(clients, "zone", 0.4)
    for zone, groupe in resultat.items():
        lot = [c for c in clients if c["zone"] == zone]
        vn, fp, fn, vp = confusion_matrix([c["parti"] for c in lot], [int(c["score"] >= 0.4) for c in lot],
                                          labels=[0, 1]).ravel()
        assert (groupe.vrais_positifs, groupe.faux_positifs) == (vp, fp)
        assert math.isclose(groupe.tvp, vp / (vp + fn)) and math.isclose(groupe.tfp, fp / (fp + vn))
        assert math.isclose(groupe.precision, vp / (vp + fp))


# --------------------------------------------------------------------------- Défi 231 : quatre_cinquiemes


def _quatre_bogue(bug):
    def quatre_cinquiemes(groupes, min_effectif=30):
        corrige.quatre_cinquiemes(groupes, min_effectif)  # les mêmes erreurs que le corrigé
        return _ref_quatre(groupes, min_effectif, bug)
    return quatre_cinquiemes


@pytest.mark.parametrize("bug,morceau", [
    ("petits", "min_effectif"),
    ("ref_compte", "SÉLECTIONNÉS"),
    ("ref_premier", "premier groupe du dict"),
    ("ref_derniere", "PREMIER groupe"),
    ("moyenne", "MOYEN"),
    ("inverse", "à l'envers"),
    ("float", "0.7999999999999999"),
    ("large", "STRICTEMENT"),
])
def test_quatre_cinquiemes_nomme_l_erreur(bug, morceau):
    _echec("quatre_cinquiemes", _quatre_bogue(bug), morceau)


def test_quatre_cinquiemes_ecrit_par_un_eleve_en_float():
    def en_float(groupes, min_effectif=30):
        impact = corrige.quatre_cinquiemes(groupes, min_effectif)
        if impact.reference is None or not groupes[impact.reference].selectionnes:
            return impact
        taux = {g: groupes[g].selectionnes / groupes[g].effectif for g in impact.ratios}
        ratios = {g: t / taux[impact.reference] for g, t in taux.items()}
        return impact._replace(ratios=ratios, signales=[g for g, r in ratios.items() if r < 0.8])

    _echec("quatre_cinquiemes", en_float, "0.7999999999999999", "entiers")


def test_quatre_cinquiemes_avec_des_fractions():
    def en_fractions(groupes, min_effectif=30):
        corrige.quatre_cinquiemes(groupes, min_effectif)
        juges = [g for g, x in groupes.items() if x.effectif >= min_effectif]
        taux = {g: Fraction(groupes[g].selectionnes, groupes[g].effectif) for g in juges}
        if not juges:
            return Impact(None, {}, [], [g for g in groupes if g not in juges])
        reference = max(juges, key=lambda g: taux[g])  # max garde le PREMIER des ex aequo
        ignores = [g for g in groupes if g not in juges]
        if not taux[reference]:
            return Impact(reference, dict.fromkeys(juges), [], ignores)
        ratios = {g: taux[g] / taux[reference] for g in juges}
        return Impact(reference, {g: float(r) for g, r in ratios.items()},
                      [g for g in juges if ratios[g] < Fraction(4, 5)], ignores)

    _reussi("quatre_cinquiemes", en_fractions)


def test_la_limite_quatre_cinquiemes_piege_vraiment_le_float():
    assert (20 / 30) / (25 / 30) < 0.8 and Fraction(20, 30) / Fraction(25, 30) == Fraction(4, 5)
    assert (4 / 50) / (3 / 30) < 0.8
    assert (16 / 30) / (24 / 36) == 0.8  # celui-là, le float le calcule juste : il sépare « float » de « large »


# --------------------------------------------------------------------------- Défi 232 : taux_standardises


def _standard_bogue(bug):
    def taux_standardises(clients, colonne, strate, seuil=SEUIL):
        if colonne == strate:
            raise ValueError("colonne == strate")
        _valider(clients, seuil, parti=False)
        return _ref_standard(clients, colonne, strate, seuil, bug)
    return taux_standardises


@pytest.mark.parametrize("bug,morceau", [
    ("poids_du_groupe", "SA PROPRE répartition"),
    ("poids_egaux", "moyenne simple"),
    ("toutes_strates", "COMMUNES"),
    ("poids_toutes", "ne font plus 1"),
    ("sans_absents", "disparu"),
])
def test_taux_standardises_nomme_l_erreur(bug, morceau):
    _echec("taux_standardises", _standard_bogue(bug), morceau)


def test_taux_standardises_une_passe_par_case_est_trop_lente():
    def par_case(clients, colonne, strate, seuil=SEUIL):
        if colonne == strate:
            raise ValueError("colonne == strate")
        _valider(clients, seuil, parti=False)
        if not clients:
            return Standardisation({}, {}, [], [])
        def cle(client, nom):  # écrite dans le même fichier : ses lignes comptent, comme chez un élève
            valeur = client.get(nom)
            return None if checks._absent(valeur) else valeur

        groupes = list(dict.fromkeys(cle(c, colonne) for c in clients))
        strates = list(dict.fromkeys(cle(c, strate) for c in clients))

        def lot(g, s):
            return [c for c in clients if cle(c, colonne) == g and cle(c, strate) == s]

        return _ref_standard(clients, colonne, strate, seuil) if all(lot(g, s) is not None for g in groupes
                                                                    for s in strates) else None

    _echec("taux_standardises", par_case, "UNE passe")


def test_taux_standardises_a_la_main_avec_pandas():
    clients = clients_exemple()
    resultat = corrige.taux_standardises(clients, "age", "contrat")
    df = pd.DataFrame(clients)
    df["age"] = df["age"].astype(object).where(df["age"].notna(), "absent")
    df["sel"] = df["score"] >= SEUIL
    poids = df["contrat"].value_counts(normalize=True)
    taux = df.pivot_table(index="age", columns="contrat", values="sel", aggfunc="mean")
    for age, attendu in resultat.standardise.items():
        assert math.isclose((taux.loc["absent" if age is None else age] * poids).sum(), attendu)
    assert resultat.exclues == [] and math.isclose(sum(poids), 1)


# --------------------------------------------------------------------------- Défi 233 : proxies


def _proxies_bogue(bug):
    def proxies(clients, protegee, colonnes):
        corrige.proxies(clients, protegee, colonnes)  # les mêmes erreurs que le corrigé
        return _ref_proxies(clients, protegee, colonnes, bug)
    return proxies


@pytest.mark.parametrize("bug,morceau", [
    ("cases_observees", "case VIDE"),
    ("sans_racine", "RACINE"),
    ("k_max", "min, pas max"),
    ("phi", "(k − 1)"),
    ("sans_absents", "disparu"),
    ("croissant", "plus FORT"),
    ("sans_tri", "Triez"),
])
def test_proxies_nomme_l_erreur(bug, morceau):
    _echec("proxies", _proxies_bogue(bug), morceau)


def _proxies_scipy(clients, protegee, colonnes):
    """Le V de Cramér de SciPy (sans correction), sur le tableau croisé de pandas."""
    from scipy.stats.contingency import association

    corrige.proxies(clients, protegee, colonnes)
    df = pd.DataFrame([{k: ("∅" if checks._absent(c.get(k)) else str(c.get(k))) for k in [protegee, *colonnes]}
                       for c in clients])
    resultats = []
    for colonne in colonnes:
        table = pd.crosstab(df[protegee], df[colonne]).to_numpy()
        v = 0.0 if min(table.shape) == 1 else float(association(table, method="cramer", correction=False))
        resultats.append(Proxy(colonne, v, int(df[colonne].nunique())))
    return sorted(resultats, key=lambda p: -p.v)


def test_proxies_avec_scipy():
    _reussi("proxies", _proxies_scipy)


def test_reference_proxies_comme_scipy():
    clients = generer_clients(2000, graine=9)
    for proxy in _ref_proxies(clients, "age", COLONNES):
        assert math.isclose(proxy.v, next(p.v for p in _proxies_scipy(clients, "age", COLONNES)
                                          if p.colonne == proxy.colonne), rel_tol=1e-9)


def test_proxies_ecrit_par_un_eleve():
    def sans_les_cases_vides(clients, protegee, colonnes):
        corrige.proxies(clients, protegee, colonnes)
        resultats = []
        n = len(clients)
        for colonne in colonnes:
            paires = [(corrige._cle(c, protegee), corrige._cle(c, colonne)) for c in clients]
            a, b = {}, {}
            for i, j in paires:
                a[i], b[j] = a.get(i, 0) + 1, b.get(j, 0) + 1
            croise = {}
            for paire in paires:
                croise[paire] = croise.get(paire, 0) + 1
            chi2 = sum((m - a[i] * b[j] / n) ** 2 / (a[i] * b[j] / n) for (i, j), m in croise.items())
            k = min(len(a), len(b))
            resultats.append(Proxy(colonne, math.sqrt(chi2 / (n * (k - 1))) if k > 1 else 0.0, len(b)))
        return sorted(resultats, key=lambda p: -p.v)

    def en_float_sans_garde(clients, protegee, colonnes):
        corrige.proxies(clients, protegee, colonnes)
        resultats = []
        n = len(clients)
        for colonne in colonnes:
            a = checks.Counter(corrige._cle(c, protegee) for c in clients)
            b = checks.Counter(corrige._cle(c, colonne) for c in clients)
            croise = checks.Counter((corrige._cle(c, protegee), corrige._cle(c, colonne)) for c in clients)
            k = min(len(a), len(b))
            somme = sum(m * m / (a[i] * b[j]) for (i, j), m in croise.items())
            resultats.append(Proxy(colonne, math.sqrt(n * (somme - 1) / (n * (k - 1))) if k > 1 else 0.0, len(b)))
        return sorted(resultats, key=lambda p: -p.v)

    _echec("proxies", sans_les_cases_vides, "case VIDE")
    message = _evaluer("proxies", en_float_sans_garde).message
    assert message == "" or "math domain error" in message or "max(χ², 0.0)" in message


def test_proxies_tolere_un_arrondi_sur_un_ex_aequo():
    def avec_bruit(clients, protegee, colonnes):
        return [p._replace(v=p.v + 1e-15 * (len(colonnes) - i)) for i, p in enumerate(corrige.proxies(clients, protegee,
                                                                                                         colonnes))]

    _reussi("proxies", avec_bruit)


# --------------------------------------------------------------------------- Défi 234 : testing


def _testing_bogue(bug):
    def testing(modele, clients, colonne, valeurs, seuil=SEUIL):
        corrige._seuil(seuil)
        valeurs = list(valeurs)
        if len(valeurs) < 2 or len(set(valeurs)) != len(valeurs):
            raise ValueError("valeurs")
        return _ref_testing(lambda lot: corrige._scorer(modele, lot), clients, colonne, valeurs, seuil, bug)
    return testing


@pytest.mark.parametrize("bug,morceau", [
    ("score_stocke", "ancienne version"),
    ("decalage", "décalés"),
    ("sens", "échangé"),
    ("avec_soi", "elle-même"),
])
def test_testing_nomme_l_erreur(bug, morceau):
    _echec("testing", _testing_bogue(bug), morceau)


def test_testing_qui_modifie_puis_remet_le_client():
    def testing(modele, clients, colonne, valeurs, seuil=SEUIL):
        corrige._seuil(seuil)
        avant = [s >= seuil for s in modele(clients)] if clients else []
        resultats = []
        for o in valeurs:
            for t in valeurs:
                if o == t:
                    continue
                testes = [i for i, c in enumerate(clients) if corrige._cle(c, colonne) == o]
                anciens = [clients[i].get(colonne) for i in testes]
                for i in testes:
                    clients[i][colonne] = t
                apres = [s >= seuil for s in modele([clients[i] for i in testes])] if testes else []
                for i, ancien in zip(testes, anciens):
                    clients[i][colonne] = ancien  # « je remets tout comme avant »
                resultats.append(Bascule(o, t, len(testes), sum(a and not avant[i] for i, a in zip(testes, apres)),
                                         sum(avant[i] and not a for i, a in zip(testes, apres))))
        return resultats

    _echec("testing", testing, "ORIGINAL modifié", "COPIES")


def test_testing_un_appel_par_client():
    def testing(modele, clients, colonne, valeurs, seuil=SEUIL):
        return corrige.testing(lambda lot: [modele([c])[0] for c in lot], clients, colonne, valeurs, seuil)

    _echec("testing", testing, "au plus", "un appel par client")


def test_testing_en_un_seul_appel():
    def testing(modele, clients, colonne, valeurs, seuil=SEUIL):
        corrige._seuil(seuil)
        valeurs = list(valeurs)
        if len(valeurs) < 2 or len(set(valeurs)) != len(valeurs):
            raise ValueError("valeurs")
        testes = [c for c in clients if corrige._cle(c, colonne) in valeurs]
        lot = list(testes) + [{**c, colonne: t} for t in valeurs for c in testes]
        scores = list(modele(lot)) if lot else []
        if len(scores) != len(lot):
            raise ValueError("le modèle a perdu des scores")
        n = len(testes)
        avant = [s >= seuil for s in scores[:n]]
        resultats = []
        for o in valeurs:
            for t in valeurs:
                if o != t:
                    bloc = scores[n * (1 + valeurs.index(t)): n * (2 + valeurs.index(t))]
                    idx = [i for i, c in enumerate(testes) if corrige._cle(c, colonne) == o]
                    resultats.append(Bascule(o, t, len(idx), sum(bloc[i] >= seuil and not avant[i] for i in idx),
                                             sum(avant[i] and bloc[i] < seuil for i in idx)))
        return resultats

    _reussi("testing", testing)


def test_reference_testing_comme_une_boucle_client_par_client():
    clients = clients_exemple()[:800]
    attendu = {(b.origine, b.testee): b for b in _ref_testing(modele, clients, "age", AGES)}
    for (o, t), b in attendu.items():
        lot = [c for c in clients if c["age"] == o]
        gagnent = sum(modele([c])[0] < SEUIL <= modele([{**c, "age": t}])[0] for c in lot)
        perdent = sum(modele([{**c, "age": t}])[0] < SEUIL <= modele([c])[0] for c in lot)
        assert (b.testes, b.gagnent, b.perdent) == (len(lot), gagnent, perdent)


# --------------------------------------------------------------------------- Défi 235 : seuils_egalite_chances


def _seuils_bogue(bug):
    def seuils_egalite_chances(clients, colonne, cible):
        corrige._cible(cible)
        _valider(clients)
        return _ref_seuils(clients, colonne, cible, bug)
    return seuils_egalite_chances


@pytest.mark.parametrize("bug,morceau", [
    ("float", "55.00000000000001"),
    ("fraction_float", "Fraction(str(cible))"),
    ("division_float", "0.6666666666666666"),
    ("decale", "tries[k - 1]"),
    ("strict", "score >= t"),
    ("global", "PAR GROUPE"),
    ("tfp_effectif", "RESTÉS"),
    ("sans_absents", "disparu"),
])
def test_seuils_nomme_l_erreur(bug, morceau):
    _echec("seuils_egalite_chances", _seuils_bogue(bug), morceau)


def test_seuils_ecrit_par_un_eleve():
    def avec_ceil(clients, colonne, cible):
        corrige._cible(cible)
        _valider(clients)
        groupes = {}
        for c in clients:
            groupes.setdefault(corrige._cle(c, colonne), []).append(c)
        resultat = {}
        for g, lot in groupes.items():
            partis = sorted((c["score"] for c in lot if c["parti"] == 1), reverse=True)
            if not partis:
                resultat[g] = None
                continue
            t = partis[math.ceil(cible * len(partis)) - 1]
            restes = [c["score"] for c in lot if c["parti"] != 1]
            vp, fp = sum(s >= t for s in partis), sum(s >= t for s in restes)
            resultat[g] = Reglage(t, vp / len(partis), fp / len(restes) if restes else None, (vp + fp) / len(lot))
        return resultat

    def chaque_seuil(clients, colonne, cible):
        corrige._cible(cible)
        _valider(clients)
        groupes = {}
        for c in clients:
            groupes.setdefault(corrige._cle(c, colonne), []).append(c)
        resultat = {}
        for g, lot in groupes.items():
            partis = [c["score"] for c in lot if c["parti"] == 1]
            restes = [c["score"] for c in lot if c["parti"] != 1]
            if not partis:
                resultat[g] = None
                continue
            exacte = corrige._cible(cible)
            t = max(s for s in partis if Fraction(sum(x >= s for x in partis), len(partis)) >= exacte)  # P × P
            vp, fp = sum(s >= t for s in partis), sum(s >= t for s in restes)
            resultat[g] = Reglage(t, vp / len(partis), fp / len(restes) if restes else None, (vp + fp) / len(lot))
        return resultat

    _echec("seuils_egalite_chances", avec_ceil, "55.00000000000001")
    _echec("seuils_egalite_chances", chaque_seuil, "trop lent", "Triez")


def test_seuils_autrement_justes():
    def avec_heapq(clients, colonne, cible):
        exacte = corrige._cible(cible)
        _valider(clients)
        partis, restes = {}, {}
        for c in clients:
            g = corrige._cle(c, colonne)
            partis.setdefault(g, [])
            restes.setdefault(g, [])
            (partis if c["parti"] == 1 else restes)[g].append(c["score"])
        resultat = {}
        for g in partis:
            p, r = partis[g], restes[g]
            if not p:
                resultat[g] = None
                continue
            k = math.ceil(exacte * len(p))  # ceil d'une Fraction : exact
            t = heapq.nlargest(k, p)[-1]
            vp, fp = sum(s >= t for s in p), sum(s >= t for s in r)
            resultat[g] = Reglage(t, vp / len(p), fp / len(r) if r else None, (vp + fp) / (len(p) + len(r)))
        return resultat

    def en_parcourant(clients, colonne, cible):
        exacte = corrige._cible(cible)
        _valider(clients)
        par_groupe = {}
        for c in clients:
            par_groupe.setdefault(corrige._cle(c, colonne), []).append((c["score"], c["parti"] == 1))
        resultat = {}
        for g, lot in par_groupe.items():
            p = sum(parti for _, parti in lot)
            if not p:
                resultat[g] = None
                continue
            vus = 0
            for score, parti in sorted(lot, reverse=True):  # les partis d'abord à score égal : (s, True) > (s, False)
                vus += parti
                if parti and vus * exacte.denominator >= exacte.numerator * p:
                    t = score
                    break
            vp = sum(parti for s, parti in lot if s >= t)
            fp = sum(not parti for s, parti in lot if s >= t)
            resultat[g] = Reglage(t, vp / p, fp / (len(lot) - p) if len(lot) > p else None, (vp + fp) / len(lot))
        return resultat

    _reussi("seuils_egalite_chances", avec_heapq)
    _reussi("seuils_egalite_chances", en_parcourant)


def test_reference_seuils_comme_une_recherche_exhaustive():
    alea = random.Random(5)
    for _ in range(60):
        clients = [{"g": alea.choice("AB"), "score": alea.choice((0.1, 0.2, 0.3, 0.4, 0.5)), "parti": alea.randint(0, 1)}
                   for _ in range(alea.randint(1, 30))]
        cible = alea.choice((0.1, 0.2, 0.25, 0.5, 0.55, 0.9, 1))
        for g, reglage in _ref_seuils(clients, "g", cible).items():
            partis = [c["score"] for c in clients if c["g"] == g and c["parti"]]
            if not partis:
                assert reglage is None
                continue
            permis = [t for t in partis if Fraction(sum(s >= t for s in partis), len(partis)) >= Fraction(str(cible))]
            assert reglage.seuil == max(permis)


# --------------------------------------------------------------------------- les outils communs et l'histoire


def test_les_outils_communs():
    clients = clients_exemple()
    assert repr(clients) == repr(clients_exemple()) and clients is not clients_exemple()
    assert repr(generer_clients(50, graine=3)) == repr(generer_clients(50, graine=3))
    ages = [c["age"] for c in clients]
    assert sum(a is None for a in ages) > 20 and sum(isinstance(a, float) and math.isnan(a) for a in ages) > 20
    assert {a for a in ages if isinstance(a, str)} == set(AGES)
    assert all(0 <= c["score"] <= 1 and c["parti"] in (0, 1) for c in clients)
    assert [c["score"] for c in clients] == modele(clients)  # le champ "score" est celui du modèle d'aujourd'hui
    assert modele([]) == [] and selection(3, 4).taux_selection == 0.75
    assert set(COLONNES) | {"age", "customer_id", "score", "parti"} == set(clients[0])


def test_l_histoire_tient():
    clients = clients_exemple()
    groupes = corrige.taux_par_groupe(clients, "age")
    assert groupes["65+"].tvp < groupes["18-29"].tvp / 2                     # le modèle rate les seniors qui partent
    assert corrige.quatre_cinquiemes(groupes).signales == ["65+", "45-64"]
    standard = corrige.taux_standardises(clients, "age", "contrat").standardise
    meilleur = max(standard.values())
    assert standard["45-64"] / meilleur > 0.8 > standard["65+"] / meilleur   # le contrat explique l'un, pas l'autre
    assert corrige.proxies(clients, "age", COLONNES)[0].colonne == "paiement"
    bascules = corrige.testing(modele, clients, "age", AGES)
    assert all(b.gagnent == b.perdent == 0 for b in bascules if "65+" not in (b.origine, b.testee))
    assert [b.gagnent for b in bascules if b.origine == "65+"] == [98, 98, 98]
    roux = clients[193]
    assert (roux["customer_id"], roux["age"], roux["parti"]) == ("C00194", "65+", 1)
    assert roux["score"] < SEUIL <= modele([{**roux, "age": "30-44"}])[0]
    reglages = corrige.seuils_egalite_chances(clients, "age", 0.6)
    assert reglages["65+"].seuil < 0.25 < min(r.seuil for g, r in reglages.items() if g != "65+")
    vrais = [c for c in clients if c["age"] == "65+"]                          # le modèle les croit fidèles
    assert sum(c["score"] for c in vrais) < sum(c["parti"] for c in vrais) * 0.8


# --------------------------------------------------------------------------- le correcteur lui-même


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(230, 236))
    assert any(s is SERIE for s in series())            # enregistrée dans defis/verif.py


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_38 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(equite)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_38 import *` (dans les exemples) doit donner les fonctions du corrigé
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
    clients = clients_exemple()[:500]
    copies = copy.deepcopy(clients)
    valeurs, colonnes = list(AGES), list(COLONNES)
    corrige.taux_par_groupe(clients, "age")
    corrige.taux_standardises(clients, "age", "contrat")
    corrige.proxies(clients, "age", colonnes)
    corrige.testing(modele, clients, "age", valeurs)
    corrige.seuils_egalite_chances(clients, "age", 0.6)
    groupes = corrige.taux_par_groupe(clients, "age")
    avant = copy.deepcopy(groupes)
    corrige.quatre_cinquiemes(groupes)
    assert groupes == avant and valeurs == list(AGES) and colonnes == list(COLONNES)
    assert len(clients) == len(copies) and all(
        all(a == b or (checks._absent(a) and checks._absent(b)) for a, b in zip(c.values(), d.values()))
        for c, d in zip(clients, copies))


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(ValueError):
        corrige.seuils_egalite_chances([{"g": "A", "score": 0.5, "parti": 1}], "g", float("nan"))
    with pytest.raises(TypeError):
        corrige.seuils_egalite_chances([{"g": "A", "score": 0.5, "parti": 1}], "g", True)
    assert Fraction(str(0.55)) == Fraction(11, 20) and Fraction(0.55) > Fraction(11, 20)
    assert math.ceil(0.55 * 100) == 56                   # la raison du Fraction(str(cible)) du corrigé
    with pytest.raises(ValueError):
        corrige.testing(lambda lot: [0.5], [{"age": "65+"}, {"age": "18-29"}], "age", ["65+", "18-29"])
    with pytest.raises(ValueError):
        corrige.taux_standardises([{"g": "A", "s": "x", "score": 0.1}, {"g": "B", "s": "y", "score": 0.1}], "g", "s")


def test_un_modele_espion_voit_tout():
    clients = [{"age": "65+", "base": 0.6}, {"age": "18-29", "base": 0.4}]
    espion = checks._Espion(_modele_test, clients)
    espion([clients[0], {**clients[1], "age": "65+"}])
    assert espion.appels == 1 and espion.modifies == []
    clients[1]["age"] = "65+"
    espion(clients)
    assert espion.modifies == [{"age": "65+", "base": 0.4}]

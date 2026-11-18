"""Série 27 (retrouver le même client) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 164`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées ;
- les exemples des docstrings, les outils communs et le corrigé sont justes.
"""

from __future__ import annotations

import doctest
import functools
import itertools
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.corrige_serie_27 as corrige  # noqa: E402
from defis.checks_serie_27 import SERIE  # noqa: E402
from defis.doublons import (  # noqa: E402
    CLES,
    VERITE_EXEMPLE,
    fiches_exemple,
    generer_fiches,
    normaliser_email,
    normaliser_nom,
    normaliser_telephone,
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


# --------------------------------------------------------------------------- Défi 164 : distance_edition


def _distance(defaut=None):
    """Une distance écrite autrement que le corrigé (le tableau entier), avec UN défaut au choix."""

    def distance_edition(a, b):
        if defaut == "none_vide":
            a, b = a or "", b or ""
            a, b = str(a), str(b)
        elif not isinstance(a, str) or not isinstance(b, str):
            raise TypeError("deux textes")
        if defaut == "minuscules":
            a, b = a.lower(), b.lower()
        if defaut == "hamming":
            return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
        if defaut == "recursion_naive":
            def d(i, j):
                if i == 0 or j == 0:
                    return i + j
                return min(d(i - 1, j) + 1, d(i, j - 1) + 1, d(i - 1, j - 1) + (a[i - 1] != b[j - 1]))
            return d(len(a), len(b))
        if defaut == "recursion_memo":
            @functools.cache
            def d(i, j):
                if i == 0 or j == 0:
                    return i + j
                return min(d(i - 1, j) + 1, d(i, j - 1) + 1, d(i - 1, j - 1) + (a[i - 1] != b[j - 1]))
            return d(len(a), len(b))
        n, m = len(a), len(b)
        t = [[0] * (m + 1) for _ in range(n + 1)]
        if defaut != "zeros_bord":
            for i in range(n + 1):
                t[i][0] = i
            for j in range(m + 1):
                t[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cout = 0 if a[i - 1] == b[j - 1] else 1
                candidats = [t[i - 1][j] + 1, t[i][j - 1] + 1]
                if defaut == "sans_remplacement":
                    if cout == 0:
                        candidats.append(t[i - 1][j - 1])
                else:
                    candidats.append(t[i - 1][j - 1] + cout)
                t[i][j] = min(candidats)
                if defaut == "damerau" and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    t[i][j] = min(t[i][j], t[i - 2][j - 2] + 1)
        if defaut == "sans_return":
            return None
        return float(t[n][m]) if defaut == "float" else t[n][m]

    return distance_edition


@pytest.mark.parametrize("defaut,mots", [
    ("zeros_bord", ("insertions", "première ligne")),
    ("hamming", ("MÊME POSITION",)),
    ("sans_remplacement", ("remplacement", "coûte 2")),
    ("damerau", ("Damerau",)),
    ("minuscules", ("TELLES QUELLES",)),
    ("recursion_naive", ("trop lent", "3^n")),
    ("recursion_memo", ("RecursionError", "boucle")),
    ("none_vide", ("TypeError", "case vide")),
    ("float", ("int",)),
    ("sans_return", ("return",)),
])
def test_distance_erreurs_classiques(defaut, mots):
    _echec("distance_edition", _distance(defaut), *mots)


def test_distance_sans_defaut_le_tableau_entier_est_accepte():
    _reussi("distance_edition", _distance())


def test_distance_un_entier_numpy_est_accepte():
    import numpy as np

    _reussi("distance_edition", lambda a, b: np.int64(corrige.distance_edition(a, b)))


# --------------------------------------------------------------------------- Défi 165 : regrouper


def _regrouper(defaut=None):
    """Un regroupement « petits groupes dans les grands » (des sets, sans parent), avec UN défaut au choix."""

    def regrouper(identifiants, paires):
        if defaut == "trie_en_place":
            identifiants.sort()
        groupe_de = {}
        for i in identifiants:
            if i in groupe_de and defaut != "doublon_accepte":
                raise ValueError("doublon")
            groupe_de[i] = {i}
        if defaut == "sans_seuls":
            groupe_de = {}
        for a, b in paires:
            for x in (a, b):
                if x not in groupe_de:
                    if defaut in ("inconnu_ajoute", "sans_seuls"):
                        groupe_de[x] = {x}
                    else:
                        raise ValueError(x)
            ga, gb = groupe_de[a], groupe_de[b]
            if ga is gb:
                continue
            if defaut != "renumerote_tout" and len(ga) < len(gb):
                ga, gb = gb, ga
            elif defaut == "renumerote_tout" and len(ga) > len(gb):
                ga, gb = gb, ga  # le GRAND groupe rejoint le petit : chaque fiche du grand est renumérotée
            for x in gb:
                ga.add(x)
                groupe_de[x] = ga
        uniques = {id(g): g for g in groupe_de.values()}.values()
        if defaut == "pas_trie":
            return [list(g) for g in uniques]
        if defaut == "tuples":
            return sorted(tuple(sorted(g)) for g in uniques)
        return sorted(sorted(g) for g in uniques)

    return regrouper


def _regrouper_non_transitif(identifiants, paires):
    """Chaque paire « rattache b au numéro de groupe de a » : les fiches déjà rattachées à b ne suivent pas."""
    groupe = {}
    for i in identifiants:
        if i in groupe:
            raise ValueError(i)
        groupe[i] = i
    for a, b in paires:
        if a not in groupe or b not in groupe:
            raise ValueError((a, b))
        groupe[b] = groupe[a]
    groupes = {}
    for i, g in groupe.items():
        groupes.setdefault(g, []).append(i)
    return sorted(sorted(g) for g in groupes.values())


def _regrouper_recursif(identifiants, paires):
    parent = {}
    for i in identifiants:
        if i in parent:
            raise ValueError(i)
        parent[i] = i

    def racine(x):
        return x if parent[x] == x else racine(parent[x])

    for a, b in paires:
        if a not in parent or b not in parent:
            raise ValueError("inconnu")
        parent[racine(a)] = racine(b)
    groupes = {}
    for i in parent:
        groupes.setdefault(racine(i), []).append(i)
    return sorted(sorted(g) for g in groupes.values())


def _regrouper_deux_lectures(identifiants, paires):
    """Lit `identifiants` deux fois : juste sur une liste, faux sur un générateur."""
    if len(set(identifiants)) != len(list(identifiants)):
        raise ValueError("doublon")
    return corrige.regrouper(identifiants, paires)


@pytest.mark.parametrize("defaut,mots", [
    ("sans_seuls", ("oublie", "client à elle seule")),
    ("pas_trie", ("ordre demandé",)),
    ("tuples", ("LISTE de listes",)),
    ("inconnu_ajoute", ("devrait lever ValueError", "inconnue")),
    ("doublon_accepte", ("devrait lever ValueError", "clé cassée")),
    ("renumerote_tout", ("trop lent", "PETIT groupe")),
    ("trie_en_place", ("modifie",)),
])
def test_regrouper_erreurs_classiques(defaut, mots):
    _echec("regrouper", _regrouper(defaut), *mots)


def test_regrouper_non_transitif_est_explique():
    _echec("regrouper", _regrouper_non_transitif, "transitivité")


def test_regrouper_sans_defaut_les_sets_fusionnes_sont_acceptes():
    _reussi("regrouper", _regrouper())


def test_regrouper_recursif_est_refuse_sur_la_longue_chaine():
    _echec("regrouper", _regrouper_recursif, "RecursionError")


def test_regrouper_un_generateur_lu_deux_fois_est_explique():
    _echec("regrouper", _regrouper_deux_lectures, "GÉNÉRATEURS")


def test_regrouper_union_par_taille_sans_compression_est_acceptee():
    def regrouper(identifiants, paires):
        parent, taille = {}, {}
        for i in identifiants:
            if i in parent:
                raise ValueError(i)
            parent[i], taille[i] = i, 1

        def racine(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for a, b in paires:
            if a not in parent or b not in parent:
                raise ValueError((a, b))
            ra, rb = racine(a), racine(b)
            if ra != rb:
                if taille[ra] < taille[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                taille[ra] += taille[rb]
        groupes = {}
        for i in parent:
            groupes.setdefault(racine(i), []).append(i)
        return sorted(sorted(g) for g in groupes.values())

    _reussi("regrouper", regrouper)


def test_regrouper_compression_sans_union_par_taille_est_acceptee():
    def regrouper(identifiants, paires):
        parent = {}
        for i in identifiants:
            if i in parent:
                raise ValueError(i)
            parent[i] = i

        def racine(x):
            chemin = []
            while parent[x] != x:
                chemin.append(x)
                x = parent[x]
            for y in chemin:
                parent[y] = x
            return x

        for a, b in paires:
            if a not in parent or b not in parent:
                raise ValueError((a, b))
            parent[racine(a)] = racine(b)
        groupes = {}
        for i in parent:
            groupes.setdefault(racine(i), []).append(i)
        return sorted(sorted(g) for g in groupes.values())

    _reussi("regrouper", regrouper)


# --------------------------------------------------------------------------- Défi 166 : paires_candidates


def _paires(defaut=None):
    """Un blocage avec des boucles écrites à la main et une clé (position, valeur), avec UN défaut au choix."""

    def paires_candidates(fiches, cles=CLES, taille_max=50):
        if defaut != "taille_non_verifiee" and (isinstance(taille_max, bool) or not isinstance(taille_max, int)
                                                or taille_max < 2):
            raise ValueError("taille_max")
        blocs = {}
        valeurs = {}
        for position, cle in enumerate(cles):
            for ident, fiche in fiches.items():
                v = cle(fiche)
                if defaut != "cle_vide_comptee" and (v is None or v == ""):
                    continue
                cle_bloc = v if defaut == "dictionnaire_commun" else (position, v)
                blocs.setdefault(cle_bloc, []).append(ident)
                valeurs[(position, ident)] = v
        if defaut == "quadratique":  # chaque fiche comparée à chaque autre (le bon résultat, mais en n²)
            paires = set()
            ids = list(fiches)
            for position in range(len(cles)):
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        vi = valeurs.get((position, ids[i]))
                        if vi is not None and vi == valeurs.get((position, ids[j])) \
                                and len(blocs[(position, vi)]) <= taille_max:
                            paires.add(tuple(sorted((ids[i], ids[j]))))
            trop = [(p, v, len(m)) for (p, v), m in blocs.items() if len(m) > taille_max]
            return paires, sorted(trop, key=lambda t: (-t[2], t[0], t[1]))
        paires = [] if defaut == "liste" else set()
        trop_gros = []
        for cle_bloc, membres in blocs.items():
            trop = len(membres) >= taille_max if defaut == "superieur_ou_egal" else len(membres) > taille_max
            if trop:
                if defaut != "trop_gros_oublie":
                    position, v = cle_bloc if defaut != "dictionnaire_commun" else (0, cle_bloc)
                    trop_gros.append((position, v, len(membres)))
                continue
            for i in range(len(membres)):
                for j in range(i + 1, len(membres)):
                    a, b = membres[i], membres[j]
                    if defaut == "cle_par_paire":
                        position = cle_bloc[0]
                        if cles[position](fiches[a]) != cles[position](fiches[b]):
                            continue
                    paire = (a, b) if defaut == "non_ordonnees" else (min(a, b), max(a, b))
                    if defaut == "liste":
                        paires.append(paire)
                    else:
                        paires.add(paire)
        if defaut != "trop_gros_non_trie":
            trop_gros.sort(key=lambda t: (-t[2], t[0], t[1]))
        else:
            trop_gros.sort(key=lambda t: t[2])
        return paires, trop_gros

    return paires_candidates


@pytest.mark.parametrize("defaut,mots", [
    ("cle_vide_comptee", ("SANS téléphone",)),
    ("dictionnaire_commun", ("PREMIÈRE clé", "DEUXIÈME")),
    ("superieur_ou_egal", ("PLUS de taille_max",)),
    ("non_ordonnees", ("a < b",)),
    ("liste", ("set",)),
    ("trop_gros_non_trie", ("bon ordre", "plus gros")),
    ("trop_gros_oublie", ("trop_gros",)),
    ("cle_par_paire", ("UNE fois par fiche",)),
    ("quadratique", ("trop lent", "dictionnaire valeur")),
    ("taille_non_verifiee", ("devrait lever ValueError", "taille_max")),
])
def test_paires_erreurs_classiques(defaut, mots):
    _echec("paires_candidates", _paires(defaut), *mots)


def test_paires_sans_defaut_les_boucles_a_la_main_sont_acceptees():
    _reussi("paires_candidates", _paires())


def test_paires_des_cles_tuples_sont_acceptees():
    def paires_candidates(fiches, cles=CLES, taille_max=50):
        if not isinstance(taille_max, int) or isinstance(taille_max, bool) or taille_max < 2:
            raise ValueError
        blocs = {}
        for ident, fiche in fiches.items():
            for position, cle in enumerate(cles):
                v = cle(fiche)
                if v not in (None, ""):
                    blocs.setdefault((position, v), set()).add(ident)
        paires = set()
        trop = []
        for (position, v), membres in blocs.items():
            if len(membres) > taille_max:
                trop.append((position, v, len(membres)))
            else:
                paires |= set(itertools.combinations(sorted(membres), 2))
        return paires, sorted(trop, key=lambda t: (-t[2], t[0], t[1]))

    _reussi("paires_candidates", paires_candidates)


# --------------------------------------------------------------------------- Défi 167 : score_correspondance


def _texte(v):
    if v is None:
        return ""
    if not isinstance(v, str) and pd.isna(v):
        return ""
    return str(v).strip()


def _score(defaut=None, distance=None):
    """Un score écrit avec une table de règles, avec UN défaut au choix."""
    distance = distance or corrige.distance_edition

    def score_correspondance(a, b):
        if defaut == "modifie":
            a["email"] = normaliser_email(a.get("email"))
        total = 0
        if defaut == "brut":
            ea, eb = _texte(a.get("email")), _texte(b.get("email"))
        else:
            ea, eb = normaliser_email(a.get("email")), normaliser_email(b.get("email"))
        if defaut == "absents_egaux":
            total += 4 if ea == eb else (-1 if ea and eb else 0)
        elif defaut == "absent_desaccord":
            if ea or eb:
                total += 4 if ea == eb else -1
        elif ea and eb:
            total += 4 if ea == eb else -1
        ta, tb = normaliser_telephone(a.get("telephone")), normaliser_telephone(b.get("telephone"))
        if defaut == "telephone_brut":  # « les deux fiches ont un téléphone » lu sur le texte BRUT
            if _texte(a.get("telephone")) and _texte(b.get("telephone")):
                total += 3 if ta == tb else -1
        elif ta and tb:
            total += 3 if ta == tb else -1
        if defaut == "nan_vrai":
            na, nb = a.get("naissance"), b.get("naissance")
            na = "" if na is None else str(na).strip()
            nb = "" if nb is None else str(nb).strip()
        else:
            na, nb = _texte(a.get("naissance")), _texte(b.get("naissance"))
        if na and nb:
            total += 3 if na == nb else (-1 if defaut == "naissance_moins_un" else -8)
        if defaut in ("ordre_des_mots", "distance_brute"):
            ma, mb = normaliser_nom(a.get("nom")), normaliser_nom(b.get("nom"))
        else:
            ma, mb = (" ".join(sorted(normaliser_nom(x.get("nom")).split())) for x in (a, b))
        if ma and mb:
            if defaut == "distance_brute" and sorted(ma.split()) == sorted(mb.split()):
                total += 3
            elif ma == mb:
                total += 3
            elif (distance(ma, mb) < 2) if defaut == "strictement_2" else (distance(ma, mb) <= 2):
                total += 1
            else:
                total -= 4
        ca, cb = _texte(a.get("code_postal")), _texte(b.get("code_postal"))
        if ca and cb:
            total += 1 if ca == cb else (-1 if defaut == "cp_desaccord" else 0)
        return float(total) if defaut == "float" else total

    return score_correspondance


@pytest.mark.parametrize("defaut,mots", [
    ("absents_egaux", ("rien à comparer", "if va and vb")),
    ("absent_desaccord", ("absente", "0 point")),
    ("nan_vrai", ("NaN", "pd.isna")),
    ("brut", ("NORMALISÉES",)),
    ("telephone_brut", ("ne se lit pas",)),
    ("ordre_des_mots", ("TRIÉS", "à l'envers")),
    ("distance_brute", ("mots TRIÉS",)),
    ("strictement_2", ("<= 2",)),
    ("cp_desaccord", ("déménage",)),
    ("naissance_moins_un", ("-8",)),
    ("float", ("int",)),
    ("modifie", ("modifie les fiches",)),
])
def test_score_erreurs_classiques(defaut, mots):
    _echec("score_correspondance", _score(defaut), *mots)


def test_score_sans_defaut_la_fabrique_est_juste():
    _reussi("score_correspondance", _score())


def test_score_sans_le_defi_164_dit_de_le_faire_d_abord():
    def distance_edition(a, b):
        raise NotImplementedError

    resultat = _evaluer("score_correspondance", _score(distance=distance_edition))
    assert resultat.statut == ECHEC, resultat.message
    assert "distance_edition" in resultat.message and "faites d'abord" in resultat.message and "164" in resultat.message


def test_score_non_ecrit_reste_a_faire():
    def score_correspondance(a, b):
        raise NotImplementedError

    assert _evaluer("score_correspondance", score_correspondance).statut == A_FAIRE


# --------------------------------------------------------------------------- Défi 168 : fiche_de_reference


def _connu(v):
    return _texte(v) != ""


def _fiche(defaut=None):
    """Une fiche de référence écrite champ par champ, avec UN défaut au choix."""

    def fiche_de_reference(fiches):
        fiches = list(fiches)
        if defaut != "sans_validation":
            if not fiches or any("id" not in f or "maj" not in f for f in fiches):
                raise ValueError("fiches")
        if not fiches:
            return {}
        if defaut == "ordre_liste":
            recentes = fiches[::-1]
        elif defaut == "reverse_true":
            recentes = sorted(fiches, key=lambda f: f["maj"], reverse=True)
        else:
            recentes = [f for _, _, f in sorted(((f["maj"], i, f) for i, f in enumerate(fiches)),
                                                key=lambda t: (t[0], t[1]), reverse=True)]
        if defaut == "plus_recente_entiere":
            base = dict(recentes[0])
        else:
            base = {}
            for f in fiches:
                for champ in f:
                    base.setdefault(champ, None)
            for champ in base:
                for f in recentes:
                    v = f.get(champ)
                    if defaut == "si_valeur":
                        connu = bool(v) and _connu(v)
                    elif defaut == "nan_present":
                        connu = v is not None and not (isinstance(v, str) and not v.strip())
                    else:
                        connu = _connu(v)
                    if connu:
                        base[champ] = v
                        break
                else:
                    if defaut == "nan_reste":
                        base[champ] = ""
        ids = sorted(f["id"] for f in fiches)
        base["id"] = fiches[0]["id"] if defaut == "id_premier" else ids[0]
        base["anciens_ids"] = ids
        base["maj"] = max(f["maj"] for f in fiches)
        inscriptions = [f["inscription"] for f in fiches if _connu(f.get("inscription"))]
        if defaut == "inscription_recente":
            base["inscription"] = next((f["inscription"] for f in recentes if _connu(f.get("inscription"))), None)
        else:
            base["inscription"] = min(inscriptions) if inscriptions else None
        if defaut == "resilie_any":
            base["resilie"] = any(bool(f.get("resilie")) for f in fiches if _connu(f.get("resilie")))
        elif defaut == "resilie_plus_recente":
            base["resilie"] = bool(recentes[0].get("resilie")) if _connu(recentes[0].get("resilie")) else False
        else:
            base["resilie"] = all(_connu(f.get("resilie")) and bool(f["resilie"]) for f in fiches)
        if defaut == "modifie":
            fiches[0].update(base)
            return fiches[0]
        return base

    return fiche_de_reference


@pytest.mark.parametrize("defaut,mots", [
    ("plus_recente_entiere", ("CONNU",)),
    ("si_valeur", ("RÉPONSE", "CNIL")),
    ("nan_present", ("NaN",)),
    ("ordre_liste", ("par date",)),
    ("reverse_true", ("reverse=True", "stable")),
    ("inscription_recente", ("PLUS ANCIENNE", "ancienneté")),
    ("resilie_any", ("TOUTES",)),
    ("resilie_plus_recente", ("TOUTES",)),
    ("id_premier", ("plus PETIT",)),
    ("modifie", ("modifie les fiches",)),
    ("nan_reste", ("None",)),
    ("sans_validation", ("devrait lever ValueError",)),
])
def test_fiche_erreurs_classiques(defaut, mots):
    _echec("fiche_de_reference", _fiche(defaut), *mots)


def test_fiche_sans_defaut_la_fabrique_est_juste():
    _reussi("fiche_de_reference", _fiche())


def test_fiche_resilie_en_entier_est_refuse():
    def fiche_de_reference(fiches):
        r = corrige.fiche_de_reference(fiches)
        r["resilie"] = int(r["resilie"])
        return r

    _echec("fiche_de_reference", fiche_de_reference, "bool")


# --------------------------------------------------------------------------- Défi 169 : qualite_dedoublonnage


def _qualite(defaut=None):
    """Une qualité calculée avec des sets de fiches par groupe, avec UN défaut au choix."""

    def qualite_dedoublonnage(predits, verite):
        if defaut == "trie_en_place":
            for g in predits:
                if isinstance(g, list):
                    g.sort()
        etiq_p, etiq_v = {}, {}
        for etiq, groupes in ((etiq_p, predits), (etiq_v, verite)):
            for n, g in enumerate(groupes):
                for i in g:
                    if i in etiq and defaut != "sans_validation":
                        raise ValueError(i)
                    etiq[i] = n
        if defaut != "sans_validation" and set(etiq_p) != set(etiq_v):
            raise ValueError("pas les mêmes fiches")

        def paires(k):
            if defaut == "ordonnees":
                return k * (k - 1)
            if defaut == "k_carre":
                return k * k
            if defaut == "flottants":
                return k * (k - 1) / 2
            return k * (k - 1) // 2

        if defaut == "paires_fabriquees":
            pp = {frozenset(c) for g in predits for c in itertools.combinations(g, 2)}
            pv = {frozenset(c) for g in verite for c in itertools.combinations(g, 2)}
            p, v, j = len(pp), len(pv), len(pp & pv)
        elif defaut == "croisement_groupes":
            p = sum(paires(len(g)) for g in predits)
            v = sum(paires(len(g)) for g in verite)
            j = sum(paires(len(set(gp) & set(gv))) for gp in predits for gv in verite)
        else:
            p = sum(paires(len(g)) for g in predits)
            v = sum(paires(len(g)) for g in verite)
            j = sum(paires(k) for k in Counter((etiq_p[i], etiq_v[i]) for i in etiq_p).values())
        if defaut == "division_zero":
            precision, rappel = j / p, j / v
        else:
            precision = j / p if p else 1.0
            rappel = j / v if v else 1.0
        if defaut == "inverse":
            precision, rappel = rappel, precision
        if defaut == "moyenne":
            f1 = (precision + rappel) / 2
        else:
            f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
        return {"paires_predites": p, "paires_vraies": v, "paires_justes": j, "precision": float(precision),
                "rappel": float(rappel), "f1": float(f1)}

    return qualite_dedoublonnage


@pytest.mark.parametrize("defaut,mots", [
    ("ordonnees", ("DOUBLE", "k(k - 1)/2")),
    ("k_carre", ("k(k - 1)/2",)),
    ("flottants", ("int", "//")),
    ("division_zero", ("ZeroDivisionError", "division par zéro")),
    ("moyenne", ("HARMONIQUE",)),
    ("inverse", ("inversés",)),
    ("paires_fabriquees", ("trop lent", "fabriquez les paires")),
    ("croisement_groupes", ("trop lent", "croisez chaque groupe")),
    ("sans_validation", ("devrait lever ValueError",)),
    ("trie_en_place", ("modifie les groupes",)),
])
def test_qualite_erreurs_classiques(defaut, mots):
    _echec("qualite_dedoublonnage", _qualite(defaut), *mots)


def test_qualite_sans_defaut_la_fabrique_est_juste():
    _reussi("qualite_dedoublonnage", _qualite())


def test_qualite_avec_pandas_crosstab_est_acceptee():
    def qualite_dedoublonnage(predits, verite):
        lignes = []
        for cote, groupes in (("p", predits), ("v", verite)):
            for n, g in enumerate(groupes):
                for i in g:
                    lignes.append((cote, i, n))
        df = pd.DataFrame(lignes, columns=["cote", "fiche", "groupe"])
        if df.duplicated(["cote", "fiche"]).any():
            raise ValueError("doublon")
        large = df.pivot(index="fiche", columns="cote", values="groupe")
        if large.isna().any().any():
            raise ValueError("pas les mêmes fiches")

        def c2(s):
            return int((s * (s - 1) // 2).sum())

        p, v = c2(large["p"].value_counts()), c2(large["v"].value_counts())
        j = c2(large.groupby(["p", "v"]).size())
        precision = j / p if p else 1.0
        rappel = j / v if v else 1.0
        f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
        return {"paires_predites": p, "paires_vraies": v, "paires_justes": j, "precision": float(precision),
                "rappel": float(rappel), "f1": float(f1)}

    _reussi("qualite_dedoublonnage", qualite_dedoublonnage)


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

    assert _evaluer("regrouper", plante).statut == ECHEC  # l'appel est enveloppé : on dit quel appel a planté
    assert "ZeroDivisionError" in _evaluer("regrouper", plante).message
    assert _evaluer("regrouper", None).statut == ERREUR


def test_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.doublons as doublons
    import defis.serie_27 as eleve

    for nom in ("CLES", "SEUIL", "VERITE_EXEMPLE", "cle_email", "cle_nom_code_postal", "cle_telephone",
                "fiches_exemple", "generer_fiches", "normaliser_email", "normaliser_nom", "normaliser_telephone"):
        assert getattr(eleve, nom) is getattr(doublons, nom), nom


def test_les_exemples_sont_neufs_a_chaque_appel():
    fiches_exemple()["F01"]["nom"] = "X"
    assert fiches_exemple()["F01"]["nom"] == "Hélène Dupré"
    fiches, _ = generer_fiches(20, graine=2)
    fiches.clear()
    assert len(generer_fiches(20, graine=2)[0]) >= 20


def test_la_verite_de_l_exemple_couvre_chaque_fiche_une_fois():
    ids = [i for g in VERITE_EXEMPLE for i in g]
    assert sorted(ids) == sorted(fiches_exemple()) and len(ids) == len(set(ids)) == 14


def test_l_exemple_contient_les_pieges_annonces():
    f = fiches_exemple()
    assert math.isnan(f["F11"]["naissance"])                        # un NaN, pas None
    assert f["F03"]["optin_sms"] is False and f["F01"]["optin_sms"] is True
    assert normaliser_telephone(f["F12"]["telephone"]) == normaliser_telephone(f["F13"]["telephone"])
    assert normaliser_email(f["F04"]["email"]) == normaliser_email(f["F05"]["email"])
    assert f["F04"]["naissance"] != f["F05"]["naissance"]
    assert corrige.score_correspondance(f["F02"], f["F03"]) < 5     # la transitivité est nécessaire


def test_generer_fiches_est_reproductible_et_dit_la_verite():
    fiches, verite = generer_fiches(500, graine=11)
    assert verite == generer_fiches(500, graine=11)[1]
    assert sorted(i for g in verite for i in g) == sorted(fiches)
    assert verite == sorted(verite) and all(list(g) == sorted(g) for g in verite)
    assert sum(len(g) > 1 for g in verite) > 100                    # des doublons…
    noms = Counter(normaliser_nom(f["nom"]) for f in fiches.values())
    assert noms.most_common(1)[0][1] > 3                            # … et des homonymes
    assert any(isinstance(f["naissance"], float) for f in fiches.values())  # des NaN


@pytest.mark.parametrize("texte", ["06 12 34 56 78", "+33 6 12 34 56 78", "+33 (0)6 12 34 56 78", "0033612345678",
                                   "06.12.34.56.78", "6 12 34 56 78", "0612345678", 612345678, 612345678.0])
def test_normaliser_telephone_lit_toutes_les_ecritures(texte):
    assert normaliser_telephone(texte) == "0612345678"


@pytest.mark.parametrize("texte", [None, float("nan"), "", "   ", "06 12", "inconnu", "+44 20 7946 0958"])
def test_normaliser_telephone_rend_vide_ce_qui_ne_se_lit_pas(texte):
    assert normaliser_telephone(texte) == ""


def test_les_cles_de_blocage():
    assert [cle({}) for cle in CLES] == [None, None, None]
    fiche = {"nom": "Hélène  DUPRÉ", "email": " H@X.fr", "telephone": "+33612345678", "code_postal": " 75011 "}
    assert [cle(fiche) for cle in CLES] == ["h@x.fr", "0612345678", "dupre helene|75011"]


def test_le_dedoublonnage_complet_du_corrige_est_precis():
    fiches, verite = generer_fiches(1500, graine=27)
    groupes = [c["anciens_ids"] for c in corrige.dedoublonner(fiches)]
    q = corrige.qualite_dedoublonnage(groupes, verite)
    assert q["precision"] > 0.95 and q["rappel"] > 0.8


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    """Les exemples de defis/serie_27.py et defis/doublons.py, rejoués avec le corrigé."""
    import defis.doublons as doublons
    import defis.serie_27 as eleve

    tests = doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(doublons)
    for nom in DEFIS:  # `from defis.serie_27 import *` (dans les exemples) doit donner les fonctions du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 50 and resultat.failed == 0

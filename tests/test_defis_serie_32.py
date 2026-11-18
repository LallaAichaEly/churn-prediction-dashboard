"""Série 32 (regrouper les clients) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 194`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (`statistics.pstdev`, `zip(*points)`,
  `math.dist` au lieu des carrés, `accumulate` + `bisect`, matrice des distances, `min(…, key=…)`…) ;
- le corrigé et les références du correcteur donnent les mêmes résultats que scikit-learn (StandardScaler,
  KMeans, silhouette_score) ou qu'un calcul naïf écrit autrement ;
- les exemples des docstrings, les outils communs et le corrigé sont justes.
"""

from __future__ import annotations

import bisect
import copy
import doctest
import itertools
import math
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.checks_serie_32 as checks  # noqa: E402
import defis.corrige_serie_32 as corrige  # noqa: E402
import defis.segments as segments  # noqa: E402
from defis.checks_serie_32 import (  # noqa: E402
    SERIE,
    _ref_affecter,
    _ref_decrire,
    _ref_kmeans,
    _ref_kmpp,
    _ref_silhouette,
    _ref_standardiser,
)
from defis.segments import (  # noqa: E402
    VARIABLES,
    Partition,
    Segment,
    clients_exemple,
    generer_clients,
    generer_points,
    points_de,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}), DEFIS[nom])


def _echec(nom, objet, *morceaux):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


def _d2(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q))


# --------------------------------------------------------------------------- Défi 194 : standardiser


def _standardiser(defaut=None):
    """Écrit autrement que le corrigé (zip(*points), statistics), avec UN défaut."""

    def standardiser(points):
        if defaut != "sans_controle":
            if not points:
                raise ValueError("aucun point")
            d = len(points[0])
            for p in points:
                if len(p) != d or any(x is None or (isinstance(x, float) and math.isnan(x)) for x in p):
                    raise ValueError(f"point invalide : {p}")
        if defaut == "modifie":
            for p in points:
                if isinstance(p, list):
                    p[0] = p[0] * 1.0 + 0.5
        n = len(points)
        colonnes = list(zip(*points))
        moyennes, ecarts = [], []
        for col in colonnes:
            moyennes.append(statistics.fmean(col))
            if defaut == "ddof1":
                ecarts.append(statistics.stdev(col) if n > 1 else 0.0)
            elif defaut == "variance":
                ecarts.append(statistics.pvariance(col))
            else:
                ecarts.append(statistics.pstdev(col))
        if defaut == "ecart_un":
            ecarts = [e if e else 1.0 for e in ecarts]
        z = []
        for p in points:
            ligne = []
            for x, m, s in zip(p, moyennes, ecarts):
                if defaut == "lent":                     # la moyenne recalculée pour CHAQUE valeur
                    j = len(ligne)
                    m = sum(q[j] for q in points) / n
                    ligne.append(0.0 if s == 0 else (x - m) / s)
                elif defaut == "zero_div":
                    ligne.append((x - m) / s)
                elif defaut == "nan":
                    ligne.append(float("nan") if s == 0 else (x - m) / s)
                elif s == 0:
                    ligne.append(0.0)
                elif defaut == "sans_centrer":
                    ligne.append(x / s)
                elif defaut == "sans_reduire":
                    ligne.append(x - m)
                else:
                    ligne.append((x - m) / s)
            z.append(tuple(ligne) if defaut != "liste" else ligne)
        if defaut == "lignes":
            z = []
            for p in points:
                m = statistics.fmean(p)
                s = statistics.pstdev(p)
                z.append(tuple(0.0 if s == 0 else (x - m) / s for x in p))
        if defaut == "minmax":
            z = [tuple(0.0 if max(c) == min(c) else (x - min(c)) / (max(c) - min(c)) for x, c in zip(p, colonnes))
                 for p in points]
        if defaut == "colonnes":
            z = [tuple(col) for col in zip(*z)]
        if defaut == "ordre":
            return z, tuple(ecarts), tuple(moyennes)
        return z, tuple(moyennes), tuple(ecarts)

    return standardiser


def test_standardiser_juste_ecrit_autrement():
    _reussi("standardiser", _standardiser())
    _reussi("standardiser", _standardiser("liste"))          # des listes au lieu de tuples : accepté


@pytest.mark.parametrize("defaut,morceaux", [
    ("ddof1", ("n − 1",)),
    ("lignes", ("chaque VARIABLE",)),
    ("sans_centrer", ("retirer la moyenne",)),
    ("sans_reduire", ("diviser par l'écart-type",)),
    ("variance", ("RACINE",)),
    ("minmax", ("min-max",)),
    ("zero_div", ("ZeroDivisionError", "colonne constante")),
    ("nan", ("NaN",)),
    ("ecart_un", ("vraie valeur",)),
    ("ordre", ("(z, moyennes, ecarts)",)),
    ("colonnes", ("rangé par colonne",)),
    ("sans_controle", ("ValueError",)),
    ("modifie", ("a modifié",)),
    ("lent", ("trop lent", "UNE fois")),
])
def test_standardiser_erreurs_classiques(defaut, morceaux):
    _echec("standardiser", _standardiser(defaut), *morceaux)


def test_standardiser_comme_scikit_learn():
    from sklearn.preprocessing import StandardScaler

    alea = random.Random(1)
    for _ in range(50):
        n, d = alea.randint(2, 30), alea.randint(1, 5)
        points = [tuple(alea.uniform(-100, 100) for _ in range(d)) for _ in range(n)]
        z, moyennes, ecarts = corrige.standardiser(points)
        echelle = StandardScaler().fit(np.array(points))
        assert np.allclose(z, echelle.transform(np.array(points)))
        assert np.allclose(moyennes, echelle.mean_) and np.allclose(ecarts, np.sqrt(echelle.var_))
        assert _ref_standardiser(points)[1:] == (moyennes, ecarts)


# --------------------------------------------------------------------------- Défi 195 : affecter


def _affecter(defaut=None):
    """Écrit autrement (min avec key, math.dist), avec UN défaut."""

    def affecter(points, centres):
        if not centres:
            raise ValueError("aucun centre")
        d = len(centres[0])
        if defaut != "sans_controle" and any(len(x) != d for x in list(centres) + list(points)):
            raise ValueError("dimensions")
        if defaut == "modifie":
            centres.sort()
        resultat = []
        for p in points:
            if defaut == "manhattan":
                ds = [sum(abs(a - b) for a, b in zip(p, c)) for c in centres]
            elif defaut == "premiere":
                ds = [abs(p[0] - c[0]) for c in centres]
            elif defaut == "sans_controle":            # zip s'arrête au plus court (math.dist, lui, refuserait)
                ds = [_d2(p, c) for c in centres]
            else:
                ds = [math.dist(p, c) for c in centres]
            if defaut == "dernier":
                j = max(range(len(ds)), key=lambda i: (-ds[i], i))
            elif defaut == "max":
                j = max(range(len(ds)), key=ds.__getitem__)
            else:
                j = min(range(len(ds)), key=ds.__getitem__)
            resultat.append(centres[j] if defaut == "centres" else j)
        return resultat

    return affecter


def test_affecter_juste_ecrit_autrement():
    _reussi("affecter", _affecter())


@pytest.mark.parametrize("defaut,morceaux", [
    ("dernier", ("PLUS PETIT numéro", "`<` strict")),
    ("manhattan", ("CARRÉS", "Manhattan")),
    ("premiere", ("TOUTES les coordonnées",)),
    ("max", ("plus PROCHE",)),
    ("centres", ("NUMÉRO",)),
    ("sans_controle", ("ValueError", "zip")),
    ("modifie", ("a modifié les centres",)),
])
def test_affecter_erreurs_classiques(defaut, morceaux):
    _echec("affecter", _affecter(defaut), *morceaux)


# --------------------------------------------------------------------------- Défi 196 : kmeans


def _recentrer(points, etiquettes, centres, vide="garder"):
    nouveaux = []
    for j, ancien in enumerate(centres):
        membres = [p for p, e in zip(points, etiquettes) if e == j]
        if membres:
            nouveaux.append(tuple(statistics.fmean(col) for col in zip(*membres)))
        elif vide == "zero":
            nouveaux.append(tuple(0.0 for _ in ancien))
        elif vide == "crash":
            nouveaux.append(tuple(x / len(membres) for x in ancien))     # division par zéro
        elif vide != "retirer":
            nouveaux.append(tuple(ancien))
    return nouveaux


def _kmeans(defaut=None):
    """Écrit autrement (la boucle « affecter, comparer, recentrer », statistics.fmean), avec UN défaut."""
    affecter = corrige.affecter

    def kmeans(points, centres, max_iter=100):
        if max_iter < 1 or not points or not centres:
            raise ValueError("entrées invalides")
        if defaut == "modifie":
            courants = centres
        else:
            courants = [tuple(c) for c in centres]
        vide = {"vide_zero": "zero", "vide_retire": "retirer", "vide_crash": "crash"}.get(defaut, "garder")
        etiquettes = affecter(points, courants)
        iterations, inertie = 0, None
        while defaut == "sans_plafond" or iterations < (1 if defaut == "une_seule" else max_iter):
            if defaut == "inertie_avant":
                inertie = sum(_d2(p, courants[e]) for p, e in zip(points, etiquettes))
            nouveaux = _recentrer(points, etiquettes, courants, vide)
            iterations += 1
            if defaut == "plus_un":
                if nouveaux == list(courants):
                    break
                courants = nouveaux
                etiquettes = affecter(points, courants)
                continue
            if defaut == "modifie":
                courants[:] = nouveaux
            else:
                courants = nouveaux
            nouvelles = affecter(points, courants)
            stable = nouvelles == etiquettes
            if defaut != "etiquettes_avant" or stable:
                etiquettes = nouvelles
            if stable:
                break
        if defaut == "inertie_racine":
            inertie = sum(math.dist(p, courants[e]) for p, e in zip(points, etiquettes))
        elif defaut != "inertie_avant":
            inertie = sum(_d2(p, courants[e]) for p, e in zip(points, etiquettes))
        return Partition(list(courants), etiquettes, float(inertie), iterations)

    return kmeans


def _kmeans_eleve_typique():
    """La boucle qu'on écrit souvent : affecter, s'arrêter si rien ne change, recentrer. Juste ?"""

    def kmeans(points, centres, max_iter=100):
        if max_iter < 1 or not points or not centres:
            raise ValueError("entrées invalides")
        centres = [tuple(c) for c in centres]
        ancien, iterations = None, 0
        while True:
            etiquettes = corrige.affecter(points, centres)
            if etiquettes == ancien or iterations == max_iter:
                break
            centres = _recentrer(points, etiquettes, centres)
            iterations += 1
            ancien = etiquettes
        inertie = sum(_d2(p, centres[e]) for p, e in zip(points, etiquettes))
        return Partition(centres, etiquettes, inertie, iterations)

    return kmeans


def test_kmeans_juste_ecrit_autrement():
    _reussi("kmeans", _kmeans())
    _reussi("kmeans", _kmeans_eleve_typique())


def _kmeans_etiquettes_d_avant():
    """La même boucle, mais avec `for` : arrivé à max_iter, les étiquettes sont celles des centres d'AVANT."""

    def kmeans(points, centres, max_iter=100):
        if max_iter < 1 or not points or not centres:
            raise ValueError("entrées invalides")
        centres = [tuple(c) for c in centres]
        ancien, iterations = None, 0
        for _ in range(max_iter):
            etiquettes = corrige.affecter(points, centres)
            if etiquettes == ancien:
                break
            centres = _recentrer(points, etiquettes, centres)
            iterations += 1
            ancien = etiquettes
        inertie = sum(_d2(p, centres[e]) for p, e in zip(points, etiquettes))
        return Partition(centres, etiquettes, inertie, iterations)

    return kmeans


@pytest.mark.parametrize("defaut,morceaux", [
    ("une_seule", ("un seul recalcul", "tant qu'une étiquette change")),
    ("vide_zero", ("mis à zéro",)),
    ("vide_retire", ("ne disparaît pas",)),
    ("vide_crash", ("ZeroDivisionError", "garde alors sa place")),
    ("etiquettes_avant", ("pas celles des centres renvoyés",)),
    ("inertie_racine", ("AU CARRÉ",)),
    ("inertie_avant", ("RENVOYÉS",)),
    ("plus_un", ("un tour de trop",)),
    ("sans_plafond", ("PLAFOND",)),
    ("modifie", ("a modifié les centres",)),
])
def test_kmeans_erreurs_classiques(defaut, morceaux):
    _echec("kmeans", _kmeans(defaut), *morceaux)


def test_kmeans_etiquettes_d_avant_le_dernier_recentrage():
    _echec("kmeans", _kmeans_etiquettes_d_avant(), "pas celles des centres renvoyés", "max_iter")


def test_kmeans_qui_appelle_affecter_pas_encore_ecrite():
    def affecter(points, centres):
        raise NotImplementedError

    def kmeans(points, centres, max_iter=100):
        return affecter(points, centres)

    _echec("kmeans", kmeans, "`affecter`", "défi 195")


def test_kmeans_comme_scikit_learn():
    from sklearn.cluster import KMeans

    alea = random.Random(2)
    compares = 0
    for graine in range(60):
        points = generer_points(alea.randint(20, 80), k=3, graine=graine, dim=2)
        depart = points[:3]
        centres, etiquettes, inertie, iterations, douteux = _ref_kmeans(points, depart, 100)
        if douteux or len(set(etiquettes)) < 3 or len(set(depart)) < 3:
            continue
        modele = KMeans(n_clusters=3, init=np.array(depart), n_init=1, algorithm="lloyd", tol=0, max_iter=300)
        modele.fit(np.array(points))
        assert list(modele.labels_) == etiquettes
        assert np.allclose(modele.cluster_centers_, centres) and math.isclose(modele.inertia_, inertie, rel_tol=1e-9)
        assert corrige.kmeans(points, depart) == Partition(centres, etiquettes, inertie, iterations)
        compares += 1
    assert compares >= 30


def test_l_inertie_ne_remonte_jamais():
    for graine in range(40):
        points = generer_points(60, k=4, graine=graine, dim=2)
        inerties = [corrige.kmeans(points, points[:4], max_iter=m).inertie for m in range(1, 8)]
        assert all(b <= a + 1e-9 for a, b in zip(inerties, inerties[1:])), inerties


def test_les_cas_ecrits_a_la_main_n_ont_pas_d_egalite_douteuse(monkeypatch):
    """Le correcteur écarte les tirages ambigus ; ses cas écrits à la main ne doivent pas l'être non plus."""
    vus = []
    comparer = checks._comparer_kmeans

    def espion(r, points, centres, max_iter, quoi):
        vus.append((quoi, _ref_kmeans(points, centres, max_iter)[4]))
        return comparer(r, points, centres, max_iter, quoi)

    monkeypatch.setattr(checks, "_comparer_kmeans", espion)
    checks.verifier_kmeans(corrige.kmeans)
    assert len(vus) > 150 and not [quoi for quoi, douteux in vus if douteux]


# --------------------------------------------------------------------------- Défi 197 : kmeans_plus_plus


def _kmpp(defaut=None):
    """Écrit autrement (math.dist au carré, accumulate + bisect), avec UN défaut."""

    def kmeans_plus_plus(points, k, tirages):
        n = len(points)
        if not 1 <= k <= n or len(tirages) < k:
            raise ValueError("k ou tirages invalides")
        if defaut == "n_moins_un":
            premier = int(tirages[0] * (n - 1))
        elif defaut == "round":
            premier = min(round(tirages[0] * n), n - 1)
        elif defaut == "decale":
            premier = (int(tirages[0] * n) + 1) % n
        else:
            premier = int(tirages[0] * n)
        choisis = [premier]

        def poids_de(p, q):
            return math.dist(p, q) if defaut == "simple" else math.dist(p, q) ** 2

        poids = [poids_de(p, points[premier]) for p in points]
        for i in range(1, k):
            if defaut in ("dernier", "lent"):
                centres = choisis[-1:] if defaut == "dernier" else choisis
                poids = [min(poids_de(p, points[c]) for c in centres) for p in points]
            cumul = list(itertools.accumulate(poids))
            if cumul[-1] == 0 and defaut != "sans_controle":
                raise ValueError("plus assez de points différents")
            seuil = tirages[i] * cumul[-1]
            j = (bisect.bisect_left if defaut == "large" else bisect.bisect_right)(cumul, seuil)
            choisis.append(j)
            if defaut not in ("dernier", "lent"):
                poids = [min(w, poids_de(p, points[j])) for w, p in zip(poids, points)]
        return [points[c] for c in choisis] if defaut == "points" else choisis

    return kmeans_plus_plus


def test_kmeans_plus_plus_juste_ecrit_autrement():
    _reussi("kmeans_plus_plus", _kmpp())


@pytest.mark.parametrize("defaut,morceaux", [
    ("large", ("STRICTEMENT", "`>=`")),
    ("dernier", ("PLUS PROCHE", "pas au dernier")),
    ("simple", ("AU CARRÉ",)),
    ("n_moins_un", ("pas n − 1",)),
    ("round", ("on tronque",)),
    ("decale", ("tirages fournis",)),
    ("points", ("NUMÉROS",)),
    ("sans_controle", ("ValueError",)),
    ("lent", ("trop lent", "SEUL nouveau centre")),
])
def test_kmeans_plus_plus_erreurs_classiques(defaut, morceaux):
    _echec("kmeans_plus_plus", _kmpp(defaut), *morceaux)


def _kmpp_naif(points, k, tirages):
    """k-means++ écrit comme dans l'article : à chaque tour, D² recalculé depuis tous les centres."""
    choisis = [int(tirages[0] * len(points))]
    for i in range(1, k):
        poids = [min(_d2(p, points[c]) for c in choisis) for p in points]
        total = sum(poids)
        if total == 0:
            return None
        cible = tirages[i] * total
        choisis.append(next(j for j in range(len(points)) if sum(poids[:j + 1]) > cible))
    return choisis


def test_kmeans_plus_plus_comme_l_article():
    alea = random.Random(3)
    for _ in range(300):
        d = alea.choice((1, 2))
        points = [tuple(alea.randint(-6, 6) / 2 for _ in range(d)) for _ in range(alea.randint(1, 10))]
        k = alea.randint(1, len(points))
        tirages = [alea.randint(0, 15) / 16 for _ in range(k)]
        naif = _kmpp_naif(points, k, tirages)
        assert _ref_kmpp(points, k, tirages)[0] == naif
        if naif is None:
            with pytest.raises(ValueError):
                corrige.kmeans_plus_plus(points, k, tirages)
        else:
            assert corrige.kmeans_plus_plus(points, k, tirages) == naif


def test_kmeans_plus_plus_part_loin():
    """Le principe : sur trois groupes bien séparés, k-means++ prend presque toujours un point de chaque."""
    points = [(x, 0.0) for x in (0.0, 0.25, 0.5)] + [(x, 0.0) for x in (10.0, 10.25)] + [(0.0, 20.0), (0.25, 20.0)]
    groupes = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    alea = random.Random(4)
    bons = sum(len({groupes[i] for i in corrige.kmeans_plus_plus(points, 3, [alea.random() for _ in range(3)])}) == 3
               for _ in range(500))
    assert bons > 480


# --------------------------------------------------------------------------- Défi 198 : silhouette


def _silhouette(defaut=None):
    """Écrit autrement (matrice des distances, étiquettes regroupées par sorted/groupby), avec UN défaut."""

    def silhouette(points, etiquettes):
        if len(points) != len(etiquettes):
            raise ValueError("longueurs")
        n = len(points)
        noms = list(dict.fromkeys(etiquettes))
        if len(noms) < 2:
            raise ValueError("un seul segment")
        dist = _d2 if defaut == "carre" else math.dist
        matrice = [[dist(p, q) for q in points] for p in points]
        membres = {}
        for j, e in enumerate(etiquettes):
            membres.setdefault(e, []).append(j)
        if defaut == "index":                       # comme si les étiquettes valaient 0, 1, …, k − 1
            tailles = [0] * len(noms)
            for e in etiquettes:
                tailles[e] += 1
        total, par_segment = 0.0, {}
        for i in range(n):
            propres = membres[etiquettes[i]]

            def d(j):
                return matrice[i][j]

            if len(propres) == 1 and defaut != "singleton_un":
                s = 0.0
            else:
                a = 0.0 if len(propres) == 1 else sum(d(j) for j in propres) / (len(propres) - (defaut != "avec_soi"))
                if defaut == "b_tous":
                    autres = [j for j in range(n) if etiquettes[j] != etiquettes[i]]
                    b = sum(d(j) for j in autres) / len(autres)
                elif defaut == "b_point":
                    b = min(d(j) for j in range(n) if etiquettes[j] != etiquettes[i])
                else:
                    b = min(sum(d(j) for j in idx) / len(idx) for nom, idx in membres.items() if nom != etiquettes[i])
                if defaut == "sans_normaliser":
                    s = b - a
                elif defaut == "zero_div":
                    s = (b - a) / max(a, b)
                else:
                    s = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
            total += s
            par_segment.setdefault(etiquettes[i], []).append(s)
        if defaut == "macro":
            return statistics.fmean(statistics.fmean(v) for v in par_segment.values())
        return total / n

    return silhouette


def _silhouette_lente():
    """Juste, mais n³ : pour chaque paire (i, j), elle recompte les membres du segment de j."""

    def silhouette(points, etiquettes):
        if len(points) != len(etiquettes) or len(set(etiquettes)) < 2:
            raise ValueError("entrées invalides")
        n, total = len(points), 0.0
        for i in range(n):
            moyennes = {}
            for j in range(n):
                taille = len([m for m in range(n) if etiquettes[m] == etiquettes[j]])
                if etiquettes[j] == etiquettes[i]:
                    taille -= 1
                if j != i:
                    moyennes[etiquettes[j]] = moyennes.get(etiquettes[j], 0.0) + math.dist(points[i], points[j]) / taille
            if etiquettes[i] not in moyennes:
                continue
            a = moyennes.pop(etiquettes[i])
            b = min(moyennes.values())
            total += 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
        return total / n

    return silhouette


def test_silhouette_juste_ecrite_autrement():
    _reussi("silhouette", _silhouette())


@pytest.mark.parametrize("defaut,morceaux", [
    ("carre", ("VRAIES distances",)),
    ("avec_soi", ("AUTRES points", "taille − 1")),
    ("singleton_un", ("SEUL dans son segment",)),
    ("b_tous", ("segment par segment",)),
    ("b_point", ("distance MOYENNE",)),
    ("macro", ("TOUS les points",)),
    ("sans_normaliser", ("max(a, b)",)),
    ("index", ("3, 7 ou « A »",)),
    ("zero_div", ("ZeroDivisionError",)),
])
def test_silhouette_erreurs_classiques(defaut, morceaux):
    _echec("silhouette", _silhouette(defaut), *morceaux)


def test_silhouette_en_n3_est_trop_lente():
    _echec("silhouette", _silhouette_lente(), "trop lent", "UNE fois")


def test_silhouette_comme_scikit_learn():
    from sklearn.metrics import silhouette_score

    alea = random.Random(5)
    for _ in range(100):
        n = alea.randint(3, 25)
        points = [tuple(alea.randint(-32, 32) / 16 for _ in range(2)) for _ in range(n)]
        etiquettes = [alea.randint(0, 3) for _ in range(n)]
        if not 2 <= len(set(etiquettes)) <= n - 1:
            continue
        attendu = silhouette_score(np.array(points), etiquettes)
        assert math.isclose(corrige.silhouette(points, etiquettes), attendu, abs_tol=1e-9)
        assert math.isclose(_ref_silhouette(points, etiquettes), attendu, abs_tol=1e-9)


# --------------------------------------------------------------------------- Défi 199 : decrire_segments


def _decrire(defaut=None):
    """Écrit autrement (statistics.pstdev, sorted puis sort stable), avec UN défaut."""

    def decrire_segments(clients, etiquettes, variables=VARIABLES):
        if defaut != "sans_controle" and len(clients) != len(etiquettes):
            raise ValueError("longueurs")
        if not variables:
            raise ValueError("aucune variable")
        if not clients:
            return []
        if defaut == "modifie":
            for c, e in zip(clients, etiquettes):
                c["segment"] = e
        n = len(clients)
        globales = {v: statistics.fmean([c[v] for c in clients]) for v in variables}
        ecarts = {v: (statistics.stdev([c[v] for c in clients]) if n > 1 else 0.0) if defaut == "ddof1"
                  else statistics.pstdev([c[v] for c in clients]) for v in variables}
        fiches = []
        for numero in sorted(set(etiquettes)):
            membres = [c for c, e in zip(clients, etiquettes) if e == numero]
            taille = len(membres)
            if defaut == "standardise":
                moyennes = {v: statistics.fmean([(c[v] - globales[v]) / (ecarts[v] or 1) for c in membres])
                            for v in variables}
            else:
                moyennes = {v: statistics.fmean([c[v] for c in membres]) for v in variables}
            zs = []
            for v in variables:
                brute = statistics.fmean([c[v] for c in membres])
                if defaut == "plus_grande":
                    z = brute
                elif defaut == "brut":
                    z = brute - globales[v]
                elif defaut == "sigma_segment":
                    s = statistics.pstdev([c[v] for c in membres])
                    z = 0.0 if s == 0 else (brute - globales[v]) / s
                elif defaut == "zero_div":
                    z = (brute - globales[v]) / ecarts[v]
                else:
                    z = 0.0 if ecarts[v] == 0 else (brute - globales[v]) / ecarts[v]
                zs.append((v, abs(z) if defaut == "abs" else z))
            distinctif = max(zs, key=lambda vz: abs(vz[1]))     # max garde le PREMIER des ex aequo
            part = taille / n * (100 if defaut == "pourcent" else 1)
            taux = statistics.fmean([c["churn"] for c in membres])
            fiches.append(Segment(numero, taille, part, taux, moyennes, distinctif))
        if defaut == "reverse":
            return sorted(fiches, key=lambda s: (s.taux_churn, s.numero), reverse=True)
        if defaut == "croissant":
            return sorted(fiches, key=lambda s: s.taux_churn)
        if defaut == "numero":
            return fiches
        return sorted(fiches, key=lambda s: -s.taux_churn)        # sort stable : les numéros restent croissants

    return decrire_segments


def test_decrire_segments_juste_ecrit_autrement():
    _reussi("decrire_segments", _decrire())


@pytest.mark.parametrize("defaut,morceaux", [
    ("plus_grande", ("plus grande moyenne",)),
    ("brut", ("écart-type GLOBAL",)),
    ("sigma_segment", ("(global)",)),
    ("abs", ("SIGNE",)),
    ("ddof1", ("POPULATION",)),
    ("pourcent", ("pourcentage",)),
    ("reverse", ("`reverse=True`",)),
    ("croissant", ("DÉCROISSANT",)),
    ("numero", ("pas par numéro seulement",)),
    ("standardise", ("BRUTES",)),
    ("zero_div", ("ZeroDivisionError", "écart-type global nul")),
    ("sans_controle", ("ValueError",)),
    ("modifie", ("a modifié les clients",)),
])
def test_decrire_segments_erreurs_classiques(defaut, morceaux):
    _echec("decrire_segments", _decrire(defaut), *morceaux)


def test_decrire_segments_comme_pandas():
    import pandas as pd

    for graine in range(30):
        clients = generer_clients(60, graine=graine)
        etiquettes = [random.Random(graine + i).randint(0, 3) for i in range(60)]
        df = pd.DataFrame(clients).assign(segment=etiquettes)
        groupes = df.groupby("segment")
        fiches = {s.numero: s for s in corrige.decrire_segments(clients, etiquettes)}
        for numero, groupe in groupes:
            fiche = fiches[numero]
            assert fiche.taille == len(groupe) and math.isclose(fiche.taux_churn, groupe["churn"].mean())
            for v in VARIABLES:
                assert math.isclose(fiche.moyennes[v], groupe[v].mean())
            v, z = fiche.distinctif
            zs = ((groupe[list(VARIABLES)].mean() - df[list(VARIABLES)].mean()) / df[list(VARIABLES)].std(ddof=0))
            assert v == zs.abs().idxmax() and math.isclose(z, zs[v])


def test_les_tirages_de_decrire_ne_sont_jamais_ambigus(monkeypatch):
    vus = []
    comparer = checks._comparer_decrire

    def espion(r, clients, etiquettes, variables, quoi):
        if "tiré au hasard" in quoi:
            vus.append(checks._z_douteux(clients, etiquettes, variables))
        return comparer(r, clients, etiquettes, variables, quoi)

    monkeypatch.setattr(checks, "_comparer_decrire", espion)
    checks.verifier_decrire_segments(corrige.decrire_segments)
    assert len(vus) == 150 and not any(vus)


def test_un_arrondi_sur_un_z_nul_n_est_pas_reproche():
    """Segment = tous les clients : tous les z valent 0, mais un calcul différent peut donner 1e-17."""

    def decrire_segments(clients, etiquettes, variables=VARIABLES):
        fiches = corrige.decrire_segments(clients, etiquettes, variables)
        if len(fiches) == 1 and len(variables) > 1:
            f = fiches[0]
            return [f._replace(distinctif=(list(variables)[-1], 1e-17))]
        return fiches

    _reussi("decrire_segments", decrire_segments)


# --------------------------------------------------------------------------- Outils communs et exemples


def test_les_exemples_ont_les_pieges_annonces():
    clients = clients_exemple()
    points = points_de(clients)
    z = corrige.standardiser(points)[0]
    brut = corrige.affecter(points, [points[0], points[4], points[8]])
    juste = corrige.affecter(z, [z[0], z[4], z[8]])
    assert brut[3] == 2 and juste[3] == 0                     # C04 : la facture l'emporte sans standardiser
    assert clients_exemple() is not clients and clients_exemple()[0] is not clients[0]
    assert points_de(clients, ("appels",))[0] == (5.0,)


def test_generer_points_est_reproductible_et_exact():
    a, b = generer_points(200, k=3, graine=7, dim=3), generer_points(200, k=3, graine=7, dim=3)
    assert a == b and a != generer_points(200, k=3, graine=8, dim=3)
    assert all(-8 <= x <= 8 and x * 16 == int(x * 16) for p in a for x in p)
    clients = generer_clients(300, graine=1)
    assert generer_clients(300, graine=1) == clients and 0.1 < statistics.fmean(c["churn"] for c in clients) < 0.9


def test_les_references_rejouent_les_bonnes_erreurs():
    points = [(0.0,), (1.0,), (9.0,), (10.0,), (20.0,)]
    assert _ref_kmpp(points, 3, [0.0, 0.75, 0.5])[0] == [0, 4, 3]
    assert _ref_kmpp(points, 3, [0.0, 0.75, 0.5], "dernier")[0] == [0, 4, 1]
    assert _ref_kmpp([(0.0,), (0.0,), (4.0,)], 2, [0.0, 0.0], "large")[0] == [0, 0]
    assert _ref_affecter([(1.0,)], [(0.0,), (2.0,)], "dernier") == [1]
    assert _ref_silhouette([(0.0,), (0.0,), (10.0,)], [0, 0, 1], "singleton_un") == 1.0
    fiches = _ref_decrire(clients_exemple(), [0] * 4 + [1] * 4 + [2] * 4, VARIABLES, "plus_grande")
    assert fiches[0][5][0] == "facture"


# --------------------------------------------------------------------------- La série


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(194, 200))


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_32 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(segments)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_32 import *` (dans les exemples) doit donner les fonctions du corrigé
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
    points = [[2.0, 10.0], [6.0, 30.0], [4.0, 20.0]]
    centres = [[2.0, 10.0], [6.0, 30.0]]
    copies = copy.deepcopy((points, centres))
    corrige.standardiser(points)
    corrige.kmeans(points, centres)
    corrige.kmeans_plus_plus(points, 2, [0.5, 0.5])
    corrige.silhouette(points, [0, 1, 1])
    assert (points, centres) == copies

"""Série 28 (les données qui n'arrêtent pas d'arriver) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 170`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées ;
- les exemples des docstrings, les outils communs et le corrigé sont justes.
"""

from __future__ import annotations

import bisect
import doctest
import heapq
import math
import numbers
import random
import statistics
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.corrige_serie_28 as corrige  # noqa: E402
from defis.checks_serie_28 import SERIE  # noqa: E402
from defis.flux import (  # noqa: E402
    INSTANT_EXEMPLE,
    UneSeuleFois,
    connexions_exemple,
    generer_journal,
    generer_montants,
    journal_exemple,
    pic_memoire,
    taux_quotidiens_exemple,
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


def _verifier_nombre(x):
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise TypeError("un nombre")
    if not math.isfinite(x):
        raise ValueError("fini")
    return float(x)


# --------------------------------------------------------------------------- Défi 170 : MoyenneVariance


def _moyenne_variance(defaut=None):
    """Une MoyenneVariance écrite autrement que le corrigé (de simples attributs), avec UN défaut au choix."""

    class MoyenneVariance:
        etat_partage = {"n": 0, "moyenne": 0.0, "m2": 0.0}

        def __init__(self):
            self.etat = MoyenneVariance.etat_partage if defaut == "partage" else {"n": 0, "moyenne": 0.0, "m2": 0.0}
            self.valeurs = []
            self.somme, self.somme_carres = 0.0, 0.0

        def ajouter(self, x):
            if defaut == "modifie_avant":
                self.etat["n"] += 1
                _verifier_nombre(x)
                self.etat["n"] -= 1
            if defaut == "texte_converti":
                x = float(x)
            if defaut == "accepte_bool":
                if not isinstance(x, numbers.Real):  # True est un numbers.Real : il passe
                    raise TypeError("un nombre")
                if math.isnan(x) or math.isinf(x):
                    raise ValueError("fini")
            elif defaut == "accepte_nan":
                if isinstance(x, bool) or not isinstance(x, numbers.Real):
                    raise TypeError("un nombre")
            elif defaut != "texte_converti":
                _verifier_nombre(x)
            elif not math.isfinite(x):
                raise ValueError("fini")
            e = self.etat
            e["n"] += 1
            if defaut == "garde_les_valeurs":
                self.valeurs.append(float(x))
            self.somme += x
            self.somme_carres += x * x
            if defaut == "moyenne_fausse":
                ecart = x - e["moyenne"]
                e["moyenne"] = x if e["n"] == 1 else (e["moyenne"] + x) / 2
                e["m2"] += ecart * (x - e["moyenne"])
                return
            ecart = x - e["moyenne"]
            e["moyenne"] += ecart / e["n"]
            e["m2"] += ecart * (x - e["moyenne"])

        @property
        def n(self):
            return self.etat["n"]

        @property
        def moyenne(self):
            if self.etat["n"] == 0:
                if defaut == "zero_vide":
                    return 0.0
                if defaut != "division_par_zero":
                    return math.nan
            if defaut == "garde_les_valeurs":
                return statistics.fmean(self.valeurs)
            return self.etat["moyenne"] if defaut != "division_par_zero" else self.somme / self.etat["n"]

        @property
        def variance(self):
            n = self.etat["n"]
            if defaut == "division_par_zero":
                return self.etat["m2"] / (n - 1)
            if defaut == "variance_zero_un" and n == 1:
                return 0.0
            if n < 2:
                return math.nan
            if defaut == "garde_les_valeurs":
                return statistics.variance(self.valeurs)
            if defaut == "somme_carres":
                return (self.somme_carres - self.somme * self.somme / n) / (n - 1)
            if defaut == "ddof0":
                return self.etat["m2"] / n
            return self.etat["m2"] / (n - 1)

    return MoyenneVariance


@pytest.mark.parametrize("defaut,mots", [
    ("somme_carres", ("connexions_exemple", "moyenne des carrés", "Welford")),
    ("ddof0", ("divisez par n", "n - 1", "ddof=0")),
    ("zero_vide", ("nan", "0.0")),
    ("division_par_zero", ("ZeroDivisionError", "math.nan")),
    ("variance_zero_un", ("1 valeur", "n - 1", "nan")),
    ("garde_les_valeurs", ("Mo", "TROIS nombres")),
    ("accepte_nan", ("devrait lever ValueError", "POUR TOUJOURS")),
    ("modifie_avant", ("AVANT de vérifier",)),
    ("texte_converti", ("'12'", "TypeError", "texte")),
    ("accepte_bool", ("True", "isinstance(x, bool)")),
    ("partage", ("partagent", "__init__")),
    ("moyenne_fausse", ("moyenne vaut",)),
])
def test_moyenne_variance_erreurs_classiques(defaut, mots):
    _echec("MoyenneVariance", _moyenne_variance(defaut), *mots)


def test_moyenne_variance_sans_defaut_la_fabrique_est_juste():
    _reussi("MoyenneVariance", _moyenne_variance())


def test_moyenne_variance_par_decalage_est_acceptee():
    """Une autre méthode stable : des sommes calculées sur x - K, avec K la première valeur."""

    class MoyenneVariance:
        def __init__(self):
            self.n, self.k, self.s, self.s2 = 0, None, 0.0, 0.0

        def ajouter(self, x):
            x = _verifier_nombre(x)
            if self.k is None:
                self.k = x
            self.n += 1
            self.s += x - self.k
            self.s2 += (x - self.k) ** 2

        @property
        def moyenne(self):
            return self.k + self.s / self.n if self.n else math.nan

        @property
        def variance(self):
            return (self.s2 - self.s * self.s / self.n) / (self.n - 1) if self.n >= 2 else math.nan

    _reussi("MoyenneVariance", MoyenneVariance)


def test_moyenne_variance_non_ecrite_reste_a_faire():
    class MoyenneVariance:
        def __init__(self):
            raise NotImplementedError

    assert _evaluer("MoyenneVariance", MoyenneVariance).statut == A_FAIRE


def test_une_methode_oubliee_est_nommee():
    class MoyenneVariance(corrige.MoyenneVariance):
        @property
        def variance(self):
            raise NotImplementedError

    _echec("MoyenneVariance", MoyenneVariance, "MoyenneVariance.variance", "terminez-la")


# --------------------------------------------------------------------------- Défi 171 : alertes_cusum


def _cusum(defaut=None):
    """Un CUSUM écrit autrement que le corrigé (un compteur de position à la main), avec UN défaut au choix."""

    def alertes_cusum(valeurs, cible, marge, seuil):
        if defaut != "sans_validation":
            if defaut == "nan_passe":
                if marge < 0 or seuil <= 0:
                    raise ValueError("paramètres")
            elif not all(math.isfinite(v) for v in (cible, marge, seuil)) or marge < 0 or seuil <= 0:
                raise ValueError("paramètres")
        if defaut == "liste":
            valeurs = list(valeurs)
        if defaut == "len":
            len(valeurs)
        if defaut == "deux_lectures":
            sum(1 for _ in valeurs)
        cumul, alertes, jour = 0.0, [], -1
        for x in valeurs:
            manquant = x is None or x != x
            if manquant and defaut == "positions_filtrees":
                continue
            jour += 1
            if defaut in ("nan_remet_a_zero", "nan_empoisonne"):
                if x is None:  # None est sauté… mais pas NaN
                    continue
            elif defaut == "none_plante":
                if x != x:  # NaN est sauté… mais pas None
                    continue
            elif manquant:
                continue
            m = 0.0 if defaut == "sans_marge" else marge
            if defaut == "sans_plancher":
                cumul = cumul + x - cible - m
            elif defaut == "nan_empoisonne":
                cumul = max(cumul + x - cible - m, 0.0)
            else:
                cumul = max(0.0, cumul + x - cible - m)
            if (cumul >= seuil) if defaut == "large" else (cumul > seuil):
                alertes.append(jour + 1 if defaut == "a_partir_de_1" else jour)
                if defaut != "sans_remise":
                    cumul = 0.0
        if defaut == "sans_return":
            return None
        return tuple(alertes) if defaut == "tuple" else alertes

    return alertes_cusum


@pytest.mark.parametrize("defaut,mots", [
    ("sans_remise", ("repartir de 0", "TOUS les jours")),
    ("sans_plancher", ("max(0, …)", "crédit")),
    ("large", ("strictement",)),
    ("sans_marge", ("oubliez la marge",)),
    ("nan_remet_a_zero", ("remet votre cumul à 0", "AVANT le calcul")),
    ("nan_empoisonne", ("NaN pour toujours",)),
    ("positions_filtrees", ("TOUS les jours", "enumerate")),
    ("a_partir_de_1", ("commencent à 0",)),
    ("none_plante", ("TypeError", "None ou NaN")),
    ("liste", ("Mo", "list(valeurs)")),
    ("len", ("pas de longueur",)),
    ("deux_lectures", ("DEUX fois",)),
    ("nan_passe", ("devrait lever ValueError", "isfinite")),
    ("sans_validation", ("devrait lever ValueError", "marge négative")),
    ("sans_return", ("return",)),
    ("tuple", ("liste de positions",)),
])
def test_cusum_erreurs_classiques(defaut, mots):
    _echec("alertes_cusum", _cusum(defaut), *mots)


def test_cusum_sans_defaut_la_fabrique_est_juste():
    _reussi("alertes_cusum", _cusum())


def test_cusum_avec_un_iterateur_a_la_main_est_accepte():
    def alertes_cusum(valeurs, cible, marge, seuil):
        for v in (cible, marge, seuil):
            if v != v or v in (math.inf, -math.inf):
                raise ValueError("fini")
        if marge < 0 or seuil <= 0:
            raise ValueError("bornes")
        it, jour, s, alertes = iter(valeurs), 0, 0, []
        while True:
            try:
                x = next(it)
            except StopIteration:
                return alertes
            if not (x is None or math.isnan(x)):
                s = max(0, s + (x - cible - marge))
                if s > seuil:
                    alertes.append(jour)
                    s = 0
            jour += 1

    _reussi("alertes_cusum", alertes_cusum)


def test_cusum_non_ecrit_reste_a_faire():
    def alertes_cusum(valeurs, cible, marge, seuil):
        raise NotImplementedError

    assert _evaluer("alertes_cusum", alertes_cusum).statut == A_FAIRE


# --------------------------------------------------------------------------- Défi 172 : ActiviteRecente


def _activite(defaut=None):
    """Une ActiviteRecente écrite comme le corrigé, avec UN défaut au choix."""

    class ActiviteRecente:
        etat_partage = {}

        def __init__(self, demi_vie_jours=30.0):
            if defaut == "demi_vie_nan":
                if demi_vie_jours <= 0:
                    raise ValueError("demi-vie")
            elif not math.isfinite(demi_vie_jours) or demi_vie_jours <= 0:
                raise ValueError("demi-vie")
            self.demi_vie = demi_vie_jours
            self.etat = ActiviteRecente.etat_partage if defaut == "partage" else {}
            self.evenements = {}

        def _facteur(self, depuis, jusqu_a):
            if defaut == "days":
                jours = (jusqu_a - depuis).days
            elif defaut == "seconds":
                jours = (jusqu_a - depuis).seconds / 86400
            else:
                jours = (jusqu_a - depuis).total_seconds() / 86400
            if defaut == "exp":
                return math.exp(-jours / self.demi_vie)
            return 0.5 ** (jours / self.demi_vie)

        def ajouter(self, client, moment, poids=1.0):
            if defaut != "poids_nan_modifie" and not math.isfinite(poids):
                raise ValueError("poids")
            if defaut == "garde_evenements":
                self.evenements.setdefault(client, []).append((moment, poids))
            elif client not in self.etat:
                self.etat[client] = (poids, moment)
            else:
                self._mettre_a_jour(client, moment, poids)
            if not math.isfinite(poids):  # « poids_nan_modifie » : vérifié APRÈS avoir rangé le nouveau score
                raise ValueError("poids")

        def _mettre_a_jour(self, client, moment, poids):
            score, ref = self.etat[client]
            if defaut == "sans_vieillir_entre":
                self.etat[client] = (score + poids, max(ref, moment))
            elif moment >= ref:
                self.etat[client] = (score * self._facteur(ref, moment) + poids, moment)
            elif defaut == "refuse_retard":
                raise ValueError("événement plus ancien que le dernier reçu")
            elif defaut == "retard_plein_poids":
                self.etat[client] = (score + poids, ref)
            elif defaut == "recule_reference":
                self.etat[client] = (score * 0.5 ** ((moment - ref).total_seconds() / 86400 / self.demi_vie) + poids,
                                     moment)
            else:
                self.etat[client] = (score + poids * self._facteur(moment, ref), ref)

        def score(self, client, instant):
            if defaut == "garde_evenements":
                evenements = self.evenements.get(client, [])
                if any(m > instant for m, _ in evenements):
                    raise ValueError("futur")
                return sum(p * self._facteur(m, instant) for m, p in evenements)
            if client not in self.etat:
                if defaut == "keyerror":
                    raise KeyError(client)
                return 0.0
            score, ref = self.etat[client]
            if instant < ref and defaut != "sans_controle_futur":
                raise ValueError("futur")
            if defaut == "sans_vieillir_a_l_instant":
                return score
            vieilli = score * self._facteur(ref, instant)
            if defaut == "lecture_modifie":
                self.etat[client] = (vieilli, instant)
            return vieilli

    return ActiviteRecente


@pytest.mark.parametrize("defaut,mots", [
    ("days", (".days", "TRONQUE", "total_seconds")),
    ("seconds", (".seconds", "DERNIÈRE journée")),
    ("exp", ("math.exp", "1/e", "0.5 **")),
    ("sans_vieillir_a_l_instant", ("ne vieillit pas jusqu'à `instant`",)),
    ("sans_vieillir_entre", ("sans vieillir l'ancien score",)),
    ("retard_plein_poids", ("EN RETARD", "âge réel")),
    ("refuse_retard", ("ValueError", "EN RETARD")),
    ("recule_reference", ("devrait lever ValueError", "futur", "PLUS RÉCENT")),
    ("sans_controle_futur", ("devrait lever ValueError", "futur")),
    ("keyerror", ("KeyError", "0.0")),
    ("lecture_modifie", ("ne doit rien modifier",)),
    ("garde_evenements", ("Mo", "deux nombres")),
    ("poids_nan_modifie", ("Validez d'abord",)),
    ("demi_vie_nan", ("devrait lever ValueError", "nan <= 0")),
    ("partage", ("partagent", "__init__")),
])
def test_activite_erreurs_classiques(defaut, mots):
    _echec("ActiviteRecente", _activite(defaut), *mots)


def test_activite_sans_defaut_la_fabrique_est_juste():
    _reussi("ActiviteRecente", _activite())


def test_activite_par_epoque_fixe_est_acceptee():
    """Une autre écriture juste : chaque poids ramené à la date du premier événement du client (sans vieillir le score)."""

    class ActiviteRecente:
        def __init__(self, demi_vie_jours=30.0):
            if not (isinstance(demi_vie_jours, (int, float)) and math.isfinite(demi_vie_jours) and demi_vie_jours > 0):
                raise ValueError("demi-vie")
            self.d = demi_vie_jours
            self.clients = {}  # client -> [époque, somme des poids × 2 ** (âge depuis l'époque / d), plus récent]

        def _jours(self, a, b):
            return (b - a).total_seconds() / 86400

        def ajouter(self, client, moment, poids=1.0):
            if poids != poids:
                raise ValueError("poids")
            if client not in self.clients:
                self.clients[client] = [moment, 0.0, moment]
            c = self.clients[client]
            c[1] += poids * 2 ** (self._jours(c[0], moment) / self.d)
            c[2] = max(c[2], moment)

        def score(self, client, instant):
            if client not in self.clients:
                return 0.0
            epoque, somme, recent = self.clients[client]
            if instant < recent:
                raise ValueError("futur")
            return somme * 0.5 ** (self._jours(epoque, instant) / self.d)

    _reussi("ActiviteRecente", ActiviteRecente)


# --------------------------------------------------------------------------- Défi 173 : MedianeCourante


def _mediane(defaut=None):
    """Une MedianeCourante à deux tas, écrite comme le corrigé, avec UN défaut au choix."""

    class MedianeCourante:
        bas_partage, haut_partage = [], []

        def __init__(self):
            if defaut == "partage":
                self.bas, self.haut = MedianeCourante.bas_partage, MedianeCourante.haut_partage
            else:
                self.bas, self.haut = [], []
            self.toutes = []

        def ajouter(self, x):
            if x is None or isinstance(x, str):
                raise TypeError("un nombre")
            if x != x and defaut != "accepte_nan":
                raise ValueError("NaN")
            if defaut in ("tri_a_chaque_fois", "insertion_lineaire"):
                if defaut == "insertion_lineaire":
                    i = 0
                    while i < len(self.toutes) and self.toutes[i] < x:
                        i += 1
                    self.toutes.insert(i, x)
                else:
                    self.toutes.append(x)
                return
            signe = 1 if defaut == "sans_signe" else -1
            if not self.bas or x <= signe * self.bas[0]:
                heapq.heappush(self.bas, signe * x)
            else:
                heapq.heappush(self.haut, x)
            if defaut == "sans_reequilibrer":
                return
            if len(self.bas) > len(self.haut) + 1:
                heapq.heappush(self.haut, signe * heapq.heappop(self.bas))
            elif len(self.haut) > len(self.bas):
                heapq.heappush(self.bas, signe * heapq.heappop(self.haut))

        def mediane(self):
            if defaut in ("tri_a_chaque_fois", "insertion_lineaire"):
                if not self.toutes:
                    return math.nan
                return statistics.median(self.toutes if defaut == "insertion_lineaire" else sorted(self.toutes))
            if not self.bas:
                if defaut == "vide_plante":
                    return -self.bas[0]
                return 0.0 if defaut == "vide_zero" else math.nan
            signe = 1 if defaut == "sans_signe" else -1
            if len(self.bas) > len(self.haut):
                return float(signe * self.bas[0])
            if defaut == "pair_bas":
                return float(signe * self.bas[0])
            if defaut == "pair_haut":
                return float(self.haut[0])
            return (signe * self.bas[0] + self.haut[0]) / 2

        def __len__(self):
            if defaut in ("tri_a_chaque_fois", "insertion_lineaire"):
                return len(self.toutes)
            return len(self.bas) + len(self.haut)

    return MedianeCourante


@pytest.mark.parametrize("defaut,mots", [
    ("pair_bas", ("PAIR", "DEUX valeurs du milieu")),
    ("pair_haut", ("PAIR", "DEUX valeurs du milieu")),
    ("sans_reequilibrer", ("rééquilibrez pas",)),
    ("sans_signe", ("signes moins",)),
    ("tri_a_chaque_fois", ("trop lent", "comparaisons", "heapq")),
    ("insertion_lineaire", ("trop lent",)),
    ("vide_plante", ("IndexError", "math.nan")),
    ("vide_zero", ("nan", "0.0")),
    ("accepte_nan", ("devrait lever ValueError", "NaN n'est ni")),
    ("partage", ("partagent", "__init__")),
])
def test_mediane_erreurs_classiques(defaut, mots):
    _echec("MedianeCourante", _mediane(defaut), *mots)


def test_mediane_sans_defaut_la_fabrique_est_juste():
    _reussi("MedianeCourante", _mediane())


def test_mediane_avec_insort_est_acceptee():
    """bisect.insort : O(log n) comparaisons (le décalage de la liste se fait en C, très vite)."""

    class MedianeCourante:
        def __init__(self):
            self.v = []

        def ajouter(self, x):
            if not isinstance(x, (int, float)):
                raise TypeError("un nombre")
            if math.isnan(x):
                raise ValueError("NaN")
            bisect.insort(self.v, x)

        def mediane(self):
            n = len(self.v)
            if n == 0:
                return math.nan
            return self.v[n // 2] if n % 2 else (self.v[n // 2 - 1] + self.v[n // 2]) / 2

        def __len__(self):
            return len(self.v)

    _reussi("MedianeCourante", MedianeCourante)


def test_mediane_avec_le_surplus_en_haut_est_acceptee():
    class MedianeCourante:
        def __init__(self):
            self.bas, self.haut = [], []

        def ajouter(self, x):
            if not isinstance(x, (int, float)):
                raise TypeError("un nombre")
            if x != x:
                raise ValueError("NaN")
            heapq.heappush(self.haut, x)
            heapq.heappush(self.bas, -heapq.heappop(self.haut))
            if len(self.bas) > len(self.haut):
                heapq.heappush(self.haut, -heapq.heappop(self.bas))

        def mediane(self):
            if not self.haut:
                return math.nan
            return self.haut[0] if len(self.haut) > len(self.bas) else (self.haut[0] - self.bas[0]) / 2

        def __len__(self):
            return len(self.bas) + len(self.haut)

    _reussi("MedianeCourante", MedianeCourante)


# --------------------------------------------------------------------------- Défi 174 : echantillon_reservoir


def _reservoir(defaut=None):
    """L'algorithme R, écrit comme le corrigé, avec UN défaut au choix."""

    def echantillon_reservoir(flux, k, graine=None):
        if defaut != "accepte_bool" and isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k entier")
        if k < 0 and defaut != "k_negatif":
            raise ValueError("k positif")
        if defaut == "seed_global":
            random.seed(graine)
            rng = random
        else:
            rng = random.Random(0 if defaut == "graine_ignoree" else graine)
        if defaut == "len":
            len(flux)
        if defaut == "deux_lectures":
            sum(1 for _ in flux)
        if defaut == "liste_puis_sample":
            valeurs = list(flux)
            return valeurs if len(valeurs) <= k else rng.sample(valeurs, k)
        echantillon = []
        for i, x in enumerate(flux):
            if defaut == "rng_dans_boucle":
                rng = random.Random(graine)
            if i < k:
                echantillon.append(x)
            elif defaut == "plus_ancien":
                if rng.randrange(i + 1) < k:
                    echantillon.pop(0)
                    echantillon.append(x)
            elif defaut == "randrange_i_k0_protege" and k == 0:
                break
            else:
                j = rng.randrange(i if defaut in ("randrange_i", "randrange_i_k0_protege") else i + 1)
                if j < k:
                    echantillon[j] = x
        return echantillon

    return echantillon_reservoir


@pytest.mark.parametrize("defaut,mots", [
    ("randrange_i", ("randrange(0)", "randrange(i + 1)")),
    ("randrange_i_k0_protege", ("randrange(i + 1)", "TOUJOURS sous k", "JAMAIS gardée")),
    ("plus_ancien", ("plus ANCIENNE",)),
    ("rng_dans_boucle", ("devrait être choisie",)),
    ("seed_global", ("GLOBAL", "random.Random(graine)")),
    ("graine_ignoree", ("graine reçue",)),
    ("liste_puis_sample", ("Mo", "gardez le flux")),
    ("len", ("pas de longueur",)),
    ("deux_lectures", ("DEUX fois",)),
    ("accepte_bool", ("True", "isinstance(k, bool)")),
    ("k_negatif", ("devrait lever ValueError",)),
])
def test_reservoir_erreurs_classiques(defaut, mots):
    _echec("echantillon_reservoir", _reservoir(defaut), *mots)


def test_reservoir_sans_defaut_la_fabrique_est_juste():
    _reussi("echantillon_reservoir", _reservoir())


def test_reservoir_algorithme_l_est_accepte():
    """L'algorithme L (Li, 1994) saute directement à la prochaine valeur gardée : autre hasard, même loi."""

    def echantillon_reservoir(flux, k, graine=None):
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k entier")
        if k < 0:
            raise ValueError("k positif")
        if k == 0:
            return []
        rng = random.Random(graine)
        it = iter(flux)
        echantillon = []
        for x in it:
            echantillon.append(x)
            if len(echantillon) == k:
                break
        if len(echantillon) < k:
            return echantillon
        w = math.exp(math.log(rng.random()) / k)
        prochain = math.floor(math.log(rng.random()) / math.log(1 - w))
        for x in it:
            if prochain > 0:
                prochain -= 1
                continue
            echantillon[rng.randrange(k)] = x
            w *= math.exp(math.log(rng.random()) / k)
            prochain = math.floor(math.log(rng.random()) / math.log(1 - w))
        return echantillon

    _reussi("echantillon_reservoir", echantillon_reservoir)


def test_reservoir_tirage_en_deux_temps_est_accepte():
    def echantillon_reservoir(flux, k, graine=None):
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k entier")
        if k < 0:
            raise ValueError("k positif")
        rng = random.Random(graine)
        echantillon = []
        for vus, x in enumerate(flux, start=1):
            if len(echantillon) < k:
                echantillon.append(x)
            elif rng.random() < k / vus:
                echantillon[rng.randrange(k)] = x
        return echantillon

    _reussi("echantillon_reservoir", echantillon_reservoir)


# --------------------------------------------------------------------------- Défi 175 : fusionner


def _avec_classe(fonction, classe):
    """La même fonction, mais qui trouve `classe` sous le nom MoyenneVariance dans ses variables globales."""
    return types.FunctionType(fonction.__code__, {**fonction.__globals__, "MoyenneVariance": classe},
                              fonction.__name__, fonction.__defaults__, fonction.__closure__)


def _fusion(defaut=None):
    """Une fusion qui passe par des sommes pondérées (autrement que le corrigé), avec UN défaut au choix."""

    def fusionner(a, b):
        if defaut == "modifie_et_renvoie_a":
            r = a
        else:
            r = MoyenneVariance()  # noqa: F821 — fournie par _avec_classe
        n = a._n + b._n
        if n == 0 and defaut != "vide_division":
            return r
        if defaut == "vide_division":
            moyenne = (a._n * a._moyenne + b._n * b._moyenne) / n
        elif defaut == "moyenne_des_moyennes" and a._n and b._n:
            moyenne = (a._moyenne + b._moyenne) / 2
        else:
            moyenne = (a._n * a._moyenne + b._n * b._moyenne) / n
        ecart = b._moyenne - a._moyenne if a._n and b._n else 0.0
        m2 = a._m2 + b._m2 + ecart * ecart * a._n * b._n / n
        if defaut == "sans_ecart":
            m2 = a._m2 + b._m2
        if defaut == "moyenne_des_variances" and a._n > 1 and b._n > 1:
            m2 = (a.variance + b.variance) / 2 * (n - 1)
        if defaut == "oublie_m2":
            m2 = 0.0
        if defaut == "modifie_a":
            a._n = 0
        r._n, r._moyenne, r._m2 = n, moyenne, m2
        return r

    return _avec_classe(fusionner, corrige.MoyenneVariance)


@pytest.mark.parametrize("defaut,mots", [
    ("moyenne_des_moyennes", ("moyenne des deux moyennes", "pondérée")),
    ("sans_ecart", ("l'écart ENTRE",)),
    ("moyenne_des_variances", ("moyenne des deux variances",)),
    ("modifie_et_renvoie_a", ("lui-même", "NOUVEL objet")),
    ("modifie_a", ("a modifié a",)),
    ("vide_division", ("ZeroDivisionError",)),
    ("oublie_m2", ("variance vaut", "m2_a + m2_b")),
])
def test_fusion_erreurs_classiques(defaut, mots):
    _echec("fusionner", _fusion(defaut), *mots)


def test_fusion_sans_defaut_la_fabrique_est_juste():
    _reussi("fusionner", _fusion())


def test_fusion_avec_une_classe_naive_demande_le_defi_170():
    _echec("fusionner", _avec_classe(corrige.fusionner, _moyenne_variance("somme_carres")), "défi 170")


def test_fusion_sans_la_classe_demande_le_defi_170():
    import defis.serie_28 as eleve

    _echec("fusionner", _avec_classe(corrige.fusionner, eleve.MoyenneVariance), "défi 170")


def test_fusion_non_ecrite_reste_a_faire_meme_si_la_classe_l_est():
    import defis.serie_28 as eleve

    assert _evaluer("fusionner", _avec_classe(eleve.fusionner, corrige.MoyenneVariance)).statut == A_FAIRE
    assert _evaluer("fusionner", eleve.fusionner).statut == A_FAIRE


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

    assert _evaluer("alertes_cusum", plante).statut == ECHEC  # l'appel est enveloppé : on dit quel appel a planté
    assert "ZeroDivisionError" in _evaluer("alertes_cusum", plante).message
    assert _evaluer("alertes_cusum", None).statut == ERREUR


def test_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.flux as flux
    import defis.serie_28 as eleve

    for nom in ("INSTANT_EXEMPLE", "UneSeuleFois", "connexions_exemple", "generer_journal", "generer_montants",
                "journal_exemple", "pic_memoire", "taux_quotidiens_exemple"):
        assert getattr(eleve, nom) is getattr(flux, nom), nom


def test_un_flux_ne_se_lit_qu_une_fois_et_n_a_pas_de_longueur():
    flux = UneSeuleFois(range(3))
    assert list(flux) == [0, 1, 2] and flux.lus == 3
    with pytest.raises(RuntimeError, match="déjà été lu"):
        list(flux)
    with pytest.raises(TypeError):
        len(UneSeuleFois([1]))


def test_pic_memoire_voit_une_liste_et_pas_un_generateur():
    _, petit = pic_memoire(lambda: sum(generer_montants(50_000)))
    _, gros = pic_memoire(lambda: sum(list(generer_montants(50_000))))
    assert petit < 50_000 < 1_000_000 < gros


def test_les_longs_flux_sont_reproductibles():
    assert list(generer_montants(100, graine=3)) == list(generer_montants(100, graine=3))
    journal = list(generer_journal(2000, n_clients=20, graine=5))
    assert journal == list(generer_journal(2000, n_clients=20, graine=5))
    en_retard = sum(1 for (_, m1, _), (_, m2, _) in zip(journal, journal[1:]) if m2 < m1)
    assert en_retard > 50                                            # des événements en retard
    assert any(m.minute for _, m, _ in journal)                      # des heures et des minutes


def test_les_exemples_contiennent_les_pieges_annonces():
    taux = taux_quotidiens_exemple()
    assert len(taux) == 40 and taux[8] == 3.5 and taux[13] is None and math.isnan(taux[29])
    assert all(t is None or (t * 4).is_integer() for t in taux if not (isinstance(t, float) and math.isnan(t)))
    journal = journal_exemple()
    moments_c3 = [m for c, m, _ in journal if c == "C3"]
    assert moments_c3[1] < moments_c3[0]                             # C3 : un événement en retard
    assert (moments_c3[0] - moments_c3[1]).days == 0                 # … à 8 heures d'écart
    assert max(m for c, m, _ in journal if c == "C2") < datetime(2023, 12, 1)
    assert all(m < INSTANT_EXEMPLE for _, m, _ in journal)
    xs = connexions_exemple()
    assert sum(x * x for x in xs) / 4 - (sum(xs) / 4) ** 2 == 0.0    # la formule du manuel perd tout


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    """Les exemples de defis/serie_28.py et defis/flux.py, rejoués avec le corrigé."""
    import defis.flux as flux
    import defis.serie_28 as eleve

    tests = doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(flux)
    for nom in DEFIS:  # `from defis.serie_28 import *` (dans les exemples) doit donner les fonctions du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 50 and resultat.failed == 0

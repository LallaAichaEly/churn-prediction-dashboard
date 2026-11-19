"""Série 37 (facturer au centime près) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 224`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (`fractions.Fraction`, `heapq.nlargest`,
  `dateutil.easter`, `numpy.busday_offset`, une file `deque` par client…) ;
- les références du correcteur donnent les mêmes résultats que `dateutil` (Pâques, de 1583 à 9999), NumPy (les
  jours ouvrés) et des calculs exacts en fractions ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (l'ancien calcul se trompe sur près d'une facture sur deux, chaque centime du relevé est retrouvé).
"""

from __future__ import annotations

import copy
import doctest
import math
import re
import sys
from collections import deque
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.checks_serie_37 as checks  # noqa: E402
import defis.corrige_serie_37 as corrige  # noqa: E402
import defis.facturation as facturation  # noqa: E402
from defis.checks_serie_37 import (  # noqa: E402
    SERIE,
    _paques,
    _ref_facturer,
    _ref_feries,
    _ref_prorata,
    _ref_rapprocher,
    _ref_relances,
    _ref_repartir,
)
from defis.facturation import (  # noqa: E402
    TAUX_TVA,
    Affectation,
    Creance,
    Facture,
    Paiement,
    Rapprochement,
    Relances,
    TotalTaux,
    creances_exemple,
    euros,
    generer_lignes,
    lignes_exemple,
    paiements_exemple,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}
_entier, _jour = corrige._entier, corrige._jour


def _evaluer(nom, objet, **autres):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}, **autres), DEFIS[nom])


def _echec(nom, objet, *morceaux):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message
    return resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


# --------------------------------------------------------------------------- Défi 224 : prorata


def _valider_prorata(montant, debut, fin, jour):
    for nom, valeur in (("debut", debut), ("fin", fin), ("jour", jour)):
        _jour(valeur, nom)
    if _entier(montant, "montant") < 0 or not debut < fin or not debut <= jour <= fin:
        raise ValueError("arguments impossibles")


def _prorata_bogue(bug):
    def prorata(montant, debut, fin, jour):
        _valider_prorata(montant, debut, fin, jour)
        return _ref_prorata(montant, debut, fin, jour, bug)
    return prorata


@pytest.mark.parametrize("bug,morceau", [
    ("trente", "30 jours"),
    ("fin_incluse", "EXCLUE"),
    ("restants_plus_un", "jour restant de trop"),
    ("tronque", "tronquez"),
    ("float_divise", "montant / periode * restants"),
    ("float_fraction", "restants / periode n'est pas exact"),
    ("banquier", "PAIR"),
])
def test_prorata_nomme_l_erreur(bug, morceau):
    _echec("prorata", _prorata_bogue(bug), morceau)


def test_prorata_ecrit_par_un_eleve_avec_round():
    def prorata(montant, debut, fin, jour):
        _valider_prorata(montant, debut, fin, jour)
        return round(montant * (fin - jour).days / (fin - debut).days)

    _echec("prorata", prorata, "498.5", "PAIR")


def test_prorata_qui_renvoie_un_float_ou_rien():
    def en_float(montant, debut, fin, jour):
        return float(corrige.prorata(montant, debut, fin, jour))

    _echec("prorata", en_float, "float", "ENTIERS")
    _echec("prorata", lambda *a: None, "return")


def test_prorata_qui_accepte_un_datetime_ou_un_bool():
    def sans_datetime(montant, debut, fin, jour):
        if isinstance(montant, bool) or not isinstance(montant, int):
            raise TypeError("montant")
        for d in (debut, fin, jour):
            if not isinstance(d, date):
                raise TypeError("date")
        if montant < 0 or not debut < fin or not debut <= jour <= fin:
            raise ValueError("impossible")
        return _ref_prorata(montant, debut, fin, jour)

    _echec("prorata", sans_datetime, "datetime", "TypeError")

    def accepte_true(montant, debut, fin, jour):
        if not isinstance(montant, int):
            raise TypeError("montant")
        return corrige.prorata(int(montant), debut, fin, jour)

    _echec("prorata", accepte_true, "True")


def test_prorata_en_fractions_est_accepte():
    def prorata(montant, debut, fin, jour):
        _valider_prorata(montant, debut, fin, jour)
        return math.floor(Fraction(montant * (fin - jour).days, (fin - debut).days) + Fraction(1, 2))

    _reussi("prorata", prorata)


# --------------------------------------------------------------------------- Défi 225 : repartir


def _valider_poids(total, poids):
    _entier(total, "total")
    for p in poids:
        if _entier(p, "poids") < 0:
            raise ValueError("poids négatif")
    if sum(poids) == 0:
        raise ValueError("rien à répartir")


def _repartir_bogue(bug):
    def repartir(total, poids):
        _valider_poids(total, poids)
        return _ref_repartir(total, list(poids), bug)
    return repartir


@pytest.mark.parametrize("bug,morceau", [
    ("arrondi", "la somme des parts fait 3 au lieu de 2"),
    ("dernier", "DERNIÈRE part"),
    ("premier", "PREMIÈRES parts"),
    ("ex_aequo_dernier", "stable"),
    ("float", "0.30000000000000004"),
    ("negatif", "|total|"),
])
def test_repartir_nomme_l_erreur(bug, morceau):
    _echec("repartir", _repartir_bogue(bug), morceau)


def test_repartir_avec_un_max_par_centime_est_trop_lent():
    def repartir(total, poids):
        _valider_poids(total, poids)
        somme, signe, t = sum(poids), (-1 if total < 0 else 1), abs(total)
        parts = [t * p // somme for p in poids]
        restes = [t * p % somme for p in poids]
        donnes = set()
        for _ in range(t - sum(parts)):
            i = max((i for i in range(len(poids)) if i not in donnes), key=lambda i: restes[i])
            donnes.add(i)
            parts[i] += 1
        return [signe * x for x in parts]

    _echec("repartir", repartir, "trop lent", "UNE fois")


def test_repartir_qui_modifie_ou_renvoie_les_poids():
    def modifie(total, poids):
        parts = corrige.repartir(total, poids)
        poids[:] = parts
        return parts

    _echec("repartir", modifie, "modifié")

    def meme_liste(total, poids):
        parts = corrige.repartir(total, poids)
        poids[:] = parts
        return poids

    _echec("repartir", meme_liste)


def test_repartir_le_dernier_prend_le_reste_sur_plusieurs_centimes():
    """Le « dernier prend le reste » donne parfois plusieurs centimes à une seule part."""
    message = _echec("repartir", _repartir_bogue("dernier"))
    assert "repartir(2, [1, 1, 1]) renvoie [0, 0, 2]" in message


def test_repartir_avec_heapq_ou_fractions_est_accepte():
    import heapq

    def avec_tas(total, poids):
        _valider_poids(total, poids)
        somme, signe, t = sum(poids), (-1 if total < 0 else 1), abs(total)
        exactes = [Fraction(t * p, somme) for p in poids]
        parts = [math.floor(x) for x in exactes]
        for i in heapq.nlargest(t - sum(parts), range(len(poids)), key=lambda i: (exactes[i] - parts[i], -i)):
            parts[i] += 1
        return [signe * x for x in parts]

    _reussi("repartir", avec_tas)


def test_la_repartition_de_reference_respecte_ses_proprietes():
    import random

    alea = random.Random(0)
    for _ in range(2000):
        poids = [alea.randrange(0, 50) for _ in range(alea.randint(1, 9))]
        if not any(poids):
            continue
        total = alea.randrange(-500, 500)
        parts = _ref_repartir(total, poids)
        somme = sum(poids)
        assert sum(parts) == total
        assert all(math.floor(Fraction(total * p, somme)) <= x <= math.ceil(Fraction(total * p, somme))
                   for x, p in zip(parts, poids))
        assert parts == [-x for x in _ref_repartir(-total, poids)]
        assert parts == corrige.repartir(total, poids)


# --------------------------------------------------------------------------- Défi 226 : facturer


def _valider_lignes(lignes, remise):
    if _entier(remise, "remise") < 0:
        raise ValueError("remise")
    total = 0
    for ligne in lignes:
        if _entier(ligne["quantite"], "quantite") < 1:
            raise ValueError("quantité")
        prix = corrige._prix(ligne["prix_unitaire"])
        if ligne["tva"] not in TAUX_TVA:
            raise ValueError("taux")
        total += checks._demi_haut(prix * ligne["quantite"] * 100)
    if remise > total:
        raise ValueError("remise trop grande")


def _facturer_bogue(bug):
    def facturer(lignes, remise=0):
        _valider_lignes(lignes, remise)
        nets, par_taux, ht, tva, ttc = _ref_facturer(lignes, remise, bug)
        return Facture(nets, [TotalTaux(*t) for t in par_taux], ht, tva, ttc)
    return facturer


@pytest.mark.parametrize("bug,morceau", [
    ("float_prix", "100.49999999999999"),
    ("banquier", "ROUND_HALF_EVEN"),
    ("tva_par_ligne", "UNE fois par taux"),
    ("tva_round", "round(16.5) vaut 16"),
    ("tri_texte", "comme du TEXTE"),
    ("ordre_apparition", "plus FORT au plus faible"),
    ("remise_premiere", "une seule ligne"),
    ("remise_ttc", "après la TVA"),
    ("remise_arrondie", "repartir (défi 225)"),
])
def test_facturer_nomme_l_erreur(bug, morceau):
    _echec("facturer", _facturer_bogue(bug), morceau)


def test_facturer_comme_l_ancien_systeme():
    """Le float, round() et la TVA ligne par ligne : le code que Nadia remplace."""
    def facturer(lignes, remise=0):
        hts = [round(float(l["prix_unitaire"]) * l["quantite"] * 100) for l in lignes]
        par_taux = {}
        for l, ht in zip(lignes, hts):
            a, b = par_taux.get(l["tva"], (0, 0))
            par_taux[l["tva"]] = (a + ht, b + round(ht * float(l["tva"]) / 100))
        taux = [TotalTaux(t, a, b) for t, (a, b) in par_taux.items()]
        return Facture(hts, taux, sum(hts), sum(t.tva for t in taux), sum(hts) + sum(t.tva for t in taux))

    _echec("facturer", facturer, "float")


def test_facturer_signale_un_repartir_faux():
    def facturer(lignes, remise=0):
        return corrige.facturer(lignes, remise)

    faux_repartir = _repartir_bogue("dernier")
    original = corrige.repartir
    try:
        corrige.repartir = faux_repartir
        resultat = evaluer_defi(SimpleNamespace(__name__="faux", facturer=corrige.facturer), DEFIS["facturer"])
    finally:
        corrige.repartir = original
    assert resultat.statut == ECHEC and "défi 225" in resultat.message, resultat.message


def test_facturer_qui_appelle_un_repartir_pas_encore_ecrit():
    import types

    code = ("from defis.corrige_serie_37 import *\n"
            "def repartir(total, poids):\n    raise NotImplementedError\n"
            "def facturer(lignes, remise=0):\n"
            "    if remise:\n        repartir(remise, [1])\n"
            "    return corrige_facturer(lignes, remise)\n")
    module = types.ModuleType("eleve_faux")
    module.corrige_facturer = corrige.facturer
    exec(compile(code, "eleve_faux.py", "exec"), module.__dict__)
    resultat = evaluer_defi(module, DEFIS["facturer"])
    assert resultat.statut == ECHEC and "défi 225" in resultat.message, resultat.message


def test_facturer_qui_laisse_passer_invalid_operation():
    def facturer(lignes, remise=0):
        for ligne in lignes:
            if isinstance(ligne["prix_unitaire"], str) and "," in ligne["prix_unitaire"]:
                Decimal(ligne["prix_unitaire"])  # lève InvalidOperation, pas ValueError
        return corrige.facturer(lignes, remise)

    _echec("facturer", facturer, "InvalidOperation", "n'est PAS une ValueError")


def test_facturer_qui_modifie_les_lignes():
    def facturer(lignes, remise=0):
        resultat = corrige.facturer(lignes, remise)
        for ligne in lignes:
            ligne["prix_unitaire"] = str(ligne["prix_unitaire"]).strip()
            ligne["ht"] = 0
        return resultat

    _echec("facturer", facturer, "modifié")


def test_facturer_en_fractions_est_accepte():
    def facturer(lignes, remise=0):
        _valider_lignes(lignes, remise)
        hts = [math.floor(Fraction(Decimal(l["prix_unitaire"])) * l["quantite"] * 100 + Fraction(1, 2)) for l in lignes]
        parts = corrige.repartir(remise, hts) if remise else [0] * len(hts)
        nets = [h - p for h, p in zip(hts, parts)]
        bases = {}
        for l, n in zip(lignes, nets):
            bases[l["tva"]] = bases.get(l["tva"], 0) + n
        taux = [TotalTaux(t, b, math.floor(Fraction(b) * Fraction(t) / 100 + Fraction(1, 2)))
                for t, b in sorted(bases.items(), key=lambda kv: -Fraction(kv[0]))]
        ht, tva = sum(t.ht for t in taux), sum(t.tva for t in taux)
        return Facture(nets, taux, ht, tva, ht + tva)

    _reussi("facturer", facturer)


def test_l_ancien_calcul_se_trompe_sur_pres_d_une_facture_sur_deux():
    def ancien(lignes):
        hts = [round(float(l["prix_unitaire"]) * l["quantite"] * 100) for l in lignes]
        return sum(hts) + sum(round(h * float(l["tva"]) / 100) for h, l in zip(hts, lignes))

    fausses = sum(ancien(generer_lignes(6, graine=g)) != _ref_facturer(generer_lignes(6, graine=g))[4]
                  for g in range(1000))
    assert 400 <= fausses <= 600
    assert (ancien(lignes_exemple()), corrige.facturer(lignes_exemple()).total_ttc) == (14533, 14536)


# --------------------------------------------------------------------------- Défi 227 : jours fériés


def _valider_annee(annee):
    if not 1583 <= _entier(annee, "annee") <= 9999:
        raise ValueError("année")


def _feries_bogue(bug):
    def jours_feries(annee, alsace_moselle=False):
        _valider_annee(annee)
        return _ref_feries(annee, alsace_moselle, bug)
    return jours_feries


@pytest.mark.parametrize("bug,morceau", [
    ("ascension_40", "39 jours"),
    ("pentecote_49", "DIMANCHE de Pentecôte"),
    ("doublon", "set"),
])
def test_jours_feries_nomme_l_erreur(bug, morceau):
    _echec("jours_feries", _feries_bogue(bug), morceau)


def test_jours_feries_avec_une_division_en_float():
    """Un « / » à la place d'un « // » dans la formule de Pâques."""
    def jours_feries(annee, alsace_moselle=False):
        _valider_annee(annee)
        a = annee % 19
        b, c = divmod(annee, 100)
        d, e = divmod(b, 4)
        f = int((b + 8) / 25)  # au lieu de (b + 8) // 25 : juste… sauf quand ce n'est pas un entier
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = divmod(c, 4)
        l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
        m = round((a + 11 * h + 22 * l) / 451)  # round() au lieu de //
        mois, jour = divmod(h + l - 7 * m + 114, 31)
        p = date(annee, mois, jour + 1)
        jours = {date(annee, 1, 1), p + timedelta(1), date(annee, 5, 1), date(annee, 5, 8), p + timedelta(39),
                 p + timedelta(50), date(annee, 7, 14), date(annee, 8, 15), date(annee, 11, 1), date(annee, 11, 11),
                 date(annee, 12, 25)}
        if alsace_moselle:
            jours |= {p - timedelta(2), date(annee, 12, 26)}
        return sorted(jours)

    _echec("jours_feries", jours_feries, "Pâques tombe le dimanche", "//")


def test_jours_feries_non_tries_ou_en_cache_partage():
    def non_tries(annee, alsace_moselle=False):
        jours = corrige.jours_feries(annee, alsace_moselle)
        return jours[1:] + jours[:1]

    _echec("jours_feries", non_tries, "triés")
    cache = {}

    def cache_partage(annee, alsace_moselle=False):
        if (annee, alsace_moselle) not in cache:
            cache[annee, alsace_moselle] = corrige.jours_feries(annee, alsace_moselle)
        return cache[annee, alsace_moselle]

    _echec("jours_feries", cache_partage, "MÊME liste")


def test_jours_feries_avec_dateutil_est_accepte():
    from dateutil.easter import easter

    def jours_feries(annee, alsace_moselle=False):
        _valider_annee(annee)
        p = easter(annee)
        jours = {date(annee, m, j) for m, j in ((1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25))}
        jours |= {p + timedelta(days=n) for n in (1, 39, 50)}
        if alsace_moselle:
            jours |= {p - timedelta(days=2), date(annee, 12, 26)}
        return sorted(jours)

    _reussi("jours_feries", jours_feries)


def test_paques_est_celle_de_dateutil_de_1583_a_9999():
    from dateutil.easter import easter

    assert all(_paques(annee) == easter(annee) == corrige.paques(annee) for annee in range(1583, 10_000))


def test_les_jours_feries_de_reference_tombent_le_bon_jour_de_la_semaine():
    for annee in range(1583, 3000):
        p = _paques(annee)
        assert p.weekday() == 6 and date(annee, 3, 22) <= p <= date(annee, 4, 25)
        assert (p + timedelta(1)).weekday() == 0 and (p + timedelta(39)).weekday() == 3
        assert (p + timedelta(50)).weekday() == 0
        assert len(_ref_feries(annee)) == (10 if p + timedelta(39) in (date(annee, 5, 1), date(annee, 5, 8)) else 11)
    assert len(_ref_feries(2008)) == 10 and len(_ref_feries(2026)) == 11


# --------------------------------------------------------------------------- Défi 228 : relances


def _valider_relances(echec, delais, grace):
    _jour(echec, "echec")
    delais = list(delais)
    for d in delais:
        if _entier(d, "délai") < 1:
            raise ValueError("délai")
    if any(b <= a for a, b in zip(delais, delais[1:])) or _entier(grace, "grace") < 0:
        raise ValueError("délais ou grâce")


def _relances_bogue(bug):
    def relances(echec, delais=facturation.DELAIS, grace=facturation.GRACE):
        _valider_relances(echec, delais, grace)
        essais, suspension = _ref_relances(echec, tuple(delais), grace, bug)
        return Relances(essais, suspension)
    return relances


@pytest.mark.parametrize("bug,morceau", [
    ("sans_feries", "1er mai"),
    ("samedi", "weekday() >= 5"),
    ("annee_echec", "année SUIVANTE"),
    ("doublon", "même jour"),
    ("enchaine", "depuis le REFUS"),
    ("suspension_brute", "suspension tombe un jour non ouvré"),
    ("suspension_echec", "DERNIER ESSAI"),
])
def test_relances_nomme_l_erreur(bug, morceau):
    _echec("relances", _relances_bogue(bug), morceau)


def test_relances_signale_un_jours_feries_faux():
    def relances(echec, delais=facturation.DELAIS, grace=facturation.GRACE):
        return corrige.relances(echec, delais, grace)

    original = corrige.jours_feries
    try:
        corrige.jours_feries = _feries_bogue("ascension_40")
        resultat = evaluer_defi(SimpleNamespace(__name__="faux", relances=corrige.relances), DEFIS["relances"])
    finally:
        corrige.jours_feries = original
    assert resultat.statut == ECHEC and "défi 227" in resultat.message, resultat.message


def test_relances_qui_modifie_les_delais():
    def relances(echec, delais=facturation.DELAIS, grace=facturation.GRACE):
        resultat = corrige.relances(echec, delais, grace)
        if isinstance(delais, list):
            delais.sort(reverse=True)
        return resultat

    _echec("relances", relances, "modifié")


def test_relances_avec_numpy_est_accepte():
    def relances(echec, delais=facturation.DELAIS, grace=facturation.GRACE):
        _valider_relances(echec, delais, grace)
        feries = [j for annee in range(echec.year, echec.year + 3) for j in corrige.jours_feries(annee)]

        def premier(jour):
            return np.busday_offset(np.datetime64(jour), 0, roll="forward", holidays=feries).astype(date)

        essais = []
        for d in delais:
            prevu = echec + timedelta(days=d)
            essais.append(premier(max(prevu, essais[-1] + timedelta(days=1)) if essais else prevu))
        return Relances(essais, premier((essais[-1] if essais else echec) + timedelta(days=grace)))

    _reussi("relances", relances)


def test_les_relances_de_reference_sont_celles_de_numpy():
    import random

    alea = random.Random(1)
    for _ in range(500):
        echec = date(2024, 1, 1) + timedelta(days=alea.randrange(5000))
        delais = sorted(alea.sample(range(1, 40), alea.randint(1, 5)))
        grace = alea.randrange(0, 20)
        feries = [j for annee in range(echec.year, echec.year + 3) for j in _ref_feries(annee)]
        essais, suspension = _ref_relances(echec, delais, grace)
        precedent = echec
        for d, essai in zip(delais, essais):
            prevu = max(echec + timedelta(days=d), precedent + timedelta(days=1))
            assert essai == np.busday_offset(np.datetime64(prevu), 0, roll="forward", holidays=feries).astype(date)
            precedent = essai
        attendu = np.busday_offset(np.datetime64(essais[-1] + timedelta(days=grace)), 0, roll="forward",
                                   holidays=feries).astype(date)
        assert suspension == attendu


# --------------------------------------------------------------------------- Défi 229 : rapprocher


def _valider_rapprochement(creances, paiements):
    numeros, ids = set(), set()
    for c in creances:
        if _entier(c.montant, "montant") <= 0 or c.numero in numeros:
            raise ValueError("créance")
        numeros.add(c.numero)
    for p in paiements:
        if _entier(p.montant, "montant") <= 0 or p.id in ids:
            raise ValueError("paiement")
        ids.add(p.id)


def _rapprocher_bogue(bug):
    def rapprocher(creances, paiements):
        _valider_rapprochement(creances, paiements)
        affectations, restes, credits, non_affectes = _ref_rapprocher(creances, paiements, bug)
        return Rapprochement([Affectation(*a) for a in affectations], dict(restes), dict(credits), non_affectes)
    return rapprocher


@pytest.mark.parametrize("bug,morceau", [
    ("ordre_recu", "ordre de leur `jour`"),
    ("fifo_numero", "ÉCHÉANCE"),
    ("autre_client", "AUTRE client"),
    ("sans_bornes", "frontières"),
    ("casse", "re.IGNORECASE"),
    ("sans_citation", "D'ABORD"),
    ("credit_perdu", "CRÉDIT"),
    ("inconnu_credit", "non_affectes"),
    ("plafond", "au plus son reste dû"),
])
def test_rapprocher_nomme_l_erreur(bug, morceau):
    _echec("rapprocher", _rapprocher_bogue(bug), morceau)


def test_rapprocher_qui_cherche_dans_toutes_les_factures_est_trop_lent():
    motif = re.compile(r"\bFAC-\d{4}-\d{5}\b", re.IGNORECASE)

    def rapprocher(creances, paiements):
        _valider_rapprochement(creances, paiements)
        reste = {c.numero: c.montant for c in creances}
        affectations, credits, non_affectes = [], {}, []
        for p in sorted(paiements, key=lambda p: p.jour):
            siennes = sorted([c for c in creances if c.client == p.client], key=lambda c: (c.echeance, c.numero))
            if not siennes:
                non_affectes.append(p.id)
                continue
            disponible = p.montant
            cites = [m.upper() for m in motif.findall(p.libelle)]
            for c in sorted(siennes, key=lambda c: cites.index(c.numero) if c.numero in cites else len(cites)):
                if disponible and reste[c.numero] > 0:
                    x = min(disponible, reste[c.numero])
                    reste[c.numero] -= x
                    disponible -= x
                    affectations.append(Affectation(p.id, c.numero, x))
            if disponible:
                credits[p.client] = credits.get(p.client, 0) + disponible
        return Rapprochement(affectations, {n: r for n, r in reste.items() if r > 0}, credits, non_affectes)

    _echec("rapprocher", rapprocher, "trop lent", "dict {client")


def test_rapprocher_qui_trie_le_releve_sur_place():
    def rapprocher(creances, paiements):
        paiements.sort(key=lambda p: p.jour)
        return corrige.rapprocher(creances, paiements)

    _echec("rapprocher", rapprocher, "modifié", "sorted")


def test_rapprocher_avec_les_restes_dans_le_desordre():
    def rapprocher(creances, paiements):
        r = corrige.rapprocher(creances, paiements)
        return r._replace(restes=dict(sorted(r.restes.items(), key=lambda kv: -kv[1])))

    _echec("rapprocher", rapprocher, "pas dans le bon ordre")


def test_rapprocher_avec_une_file_par_client_est_accepte():
    motif = re.compile(r"(?<![0-9A-Za-z])FAC-[0-9]{4}-[0-9]{5}(?![0-9A-Za-z])", re.IGNORECASE)

    def rapprocher(creances, paiements):
        _valider_rapprochement(creances, paiements)
        reste = {c.numero: c.montant for c in creances}
        client_de = {c.numero: c.client for c in creances}
        files = {}
        for c in sorted(creances, key=lambda c: (c.echeance, c.numero)):
            files.setdefault(c.client, deque()).append(c.numero)
        affectations, credits, non_affectes = [], {}, []
        for p in sorted(paiements, key=lambda p: p.jour):
            if p.client not in files:
                non_affectes.append(p.id)
                continue
            disponible, vus = p.montant, set()
            cites = [n.upper() for n in motif.findall(p.libelle)]
            file = files[p.client]
            for numero in [n for n in cites if not (n in vus or vus.add(n))] + [None]:
                while numero is None and disponible and file:
                    if reste[file[0]] == 0:
                        file.popleft()
                        continue
                    x = min(disponible, reste[file[0]])
                    reste[file[0]] -= x
                    disponible -= x
                    affectations.append(Affectation(p.id, file[0], x))
                if numero is not None and disponible and client_de.get(numero) == p.client and reste[numero]:
                    x = min(disponible, reste[numero])
                    reste[numero] -= x
                    disponible -= x
                    affectations.append(Affectation(p.id, numero, x))
            if disponible:
                credits[p.client] = credits.get(p.client, 0) + disponible
        return Rapprochement(affectations, {n: r for n, r in reste.items() if r > 0}, credits, non_affectes)

    _reussi("rapprocher", rapprocher)


def test_le_rapprochement_ne_perd_ni_n_invente_un_centime():
    creances, paiements = creances_exemple(), paiements_exemple()
    r = corrige.rapprocher(creances, paiements)
    assert _ref_rapprocher(creances, paiements) == (
        [tuple(a) for a in r.affectations], list(r.restes.items()), list(r.credits.items()), r.non_affectes)
    par_facture, par_paiement = {}, {}
    for a in r.affectations:
        par_facture[a.numero] = par_facture.get(a.numero, 0) + a.montant
        par_paiement[a.paiement] = par_paiement.get(a.paiement, 0) + a.montant
    for c in creances:
        assert par_facture.get(c.numero, 0) + r.restes.get(c.numero, 0) == c.montant
    non_affectes = set(r.non_affectes)
    assert not non_affectes & set(par_paiement)
    recu = sum(p.montant for p in paiements if p.id not in non_affectes)
    assert recu == sum(par_paiement.values()) + sum(r.credits.values())
    assert sum(c.montant for c in creances) == sum(par_facture.values()) + sum(r.restes.values())


def test_les_impayes_de_novembre_sont_des_prelevements_refuses():
    creances = creances_exemple()
    r = corrige.rapprocher(creances, paiements_exemple())
    montant = {c.numero: c for c in creances}
    novembre = [n for n, reste in r.restes.items()
                if montant[n].echeance == date(2026, 11, 5) and reste == montant[n].montant]
    assert len(r.restes) == 303 and len(novembre) == 181
    assert corrige.relances(date(2026, 11, 5)).essais == [date(2026, 11, d) for d in (6, 9, 12, 19)]


# --------------------------------------------------------------------------- La série


def test_les_references_rejouent_les_bonnes_erreurs():
    feb = (date(2026, 2, 1), date(2026, 3, 1), date(2026, 2, 17))
    assert _ref_prorata(2999, *feb, "trente") == 1200 and _ref_prorata(2999, *feb) == 1285
    assert _ref_prorata(997, date(2026, 4, 1), date(2026, 5, 1), date(2026, 4, 16), "banquier") == 498
    assert _ref_prorata(999, date(2026, 4, 1), date(2026, 5, 1), date(2026, 4, 16), "float_divise") == 499
    assert _ref_repartir(3, [3, 13, 11, 3], "float") == [0, 2, 1, 0]
    assert _ref_repartir(-1000, [1, 1, 1], "negatif") == [-333, -333, -334]
    trois = [checks._ligne("12.37")] * 3
    assert _ref_facturer(trois)[3] == 742 and _ref_facturer(trois, bug="tva_par_ligne")[3] == 741
    assert _ref_feries(2008, bug="doublon").count(date(2008, 5, 1)) == 2
    assert _ref_relances(date(2026, 12, 31), bug="annee_echec")[0][0] == date(2027, 1, 1)


def test_les_outils_communs():
    assert euros(-123456) == "-1 234,56 €"
    assert generer_lignes(5, graine=3) == generer_lignes(5, graine=3)
    assert creances_exemple() == creances_exemple() and paiements_exemple() == paiements_exemple()
    numeros = [c.numero for c in creances_exemple()]
    assert len(set(numeros)) == len(numeros) and all(re.fullmatch(r"FAC-2026-\d{5}", n) for n in numeros)
    ids = [p.id for p in paiements_exemple()]
    assert len(set(ids)) == len(ids)
    assert [p.jour for p in paiements_exemple()] != sorted(p.jour for p in paiements_exemple())  # dans le désordre
    assert all(isinstance(p.montant, int) and p.montant > 0 for p in paiements_exemple())
    assert all(l["tva"] in TAUX_TVA for l in generer_lignes(200, graine=1))


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(224, 230))
    assert any(s is SERIE for s in series())            # enregistrée dans defis/verif.py


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_37 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(facturation)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_37 import *` (dans les exemples) doit donner les fonctions du corrigé
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
    donnees = (lignes_exemple(), creances_exemple()[:500], paiements_exemple()[:500], [3, 1, 2], [1, 3, 7])
    copies = copy.deepcopy(donnees)
    lignes, creances, paiements, poids, delais = donnees
    corrige.facturer(lignes, remise=1000)
    corrige.rapprocher(creances, paiements)
    corrige.repartir(100, poids)
    corrige.relances(date(2026, 4, 30), delais)
    assert donnees == copies


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(TypeError):
        corrige.prorata(2999, datetime(2026, 2, 1), date(2026, 3, 1), date(2026, 2, 17))
    with pytest.raises(ValueError):
        corrige.facturer([checks._ligne("12,99")])
    with pytest.raises(InvalidOperation):
        Decimal("12,99")
    assert not issubclass(InvalidOperation, ValueError)  # la raison du « except InvalidOperation » du corrigé
    with pytest.raises(ValueError):
        corrige.rapprocher([Creance("FAC-2026-00001", "A", 1, date(2026, 1, 1))] * 2, [])
    with pytest.raises(ValueError):
        corrige.rapprocher([], [Paiement("P", date(2026, 1, 1), "A", 1, "")] * 2)

"""Série 44 (faire attendre le moins possible) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 266`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (Erlang C par la formule directe en logarithmes,
  une recherche par dichotomie, une simulation sans tas mais avec `min`, des indicateurs avec `statistics`, un
  planning construit de droite à gauche ou par programmation linéaire, une file à la manière de la documentation de
  `heapq`…) ;
- les solutions trop lentes (parcourir tous les conseillers à chaque appel, ajouter les vacations une par une,
  recalculer toutes les priorités à chaque `suivant`) sont arrêtées et expliquées ;
- les références du correcteur tiennent face à des vérifications indépendantes : la formule exacte en fractions,
  une simulation sans abandon qui retrouve Erlang C, une recherche exhaustive, la programmation linéaire en nombres
  entiers de SciPy, une file naïve qui recalcule les priorités à l'instant t ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (73 % de taux de service sur la semaine, 52 % l'après-midi, 26 % de départs chez ceux qui ont raccroché contre
  6 %, 53 vacations au lieu de 42).
"""

from __future__ import annotations

import bisect
import doctest
import heapq
import itertools
import math
import random
import statistics
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import defis.attente as attente  # noqa: E402
import defis.corrige_serie_44 as corrige  # noqa: E402
from defis.attente import (  # noqa: E402
    CONSEILLERS_ACTUELS,
    DUREE_MOYENNE_S,
    VACATION,
    appels_exemple,
    creneau,
    departs_exemple,
    generer_appels,
    planning_actuel_exemple,
    prevision_exemple,
)
from defis.checks_serie_44 import (  # noqa: E402
    _BUGS_CONSEILLERS,
    _BUGS_ERLANG,
    _BUGS_FILE,
    _BUGS_INDICATEURS,
    _BUGS_PLANNING,
    _BUGS_SIMULATION,
    SERIE,
    _cas_266,
    _cas_267,
    _cas_268,
    _cas_269,
    _cas_270,
    _erlang_pretee,
    _jouer,
    _ref_conseillers,
    _ref_erlang_c,
    _ref_indicateurs,
    _ref_planifier,
    _ref_simuler,
    _RefFile,
    _scenarios_271,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


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


def _avec_les_gardes(nom, calcul):
    """Une version boguée qui garde les vérifications du corrigé (les erreurs attendues), pour isoler LE bug."""
    def fonction(*args, **kwargs):
        getattr(corrige, nom)(*args, **kwargs)        # lève les mêmes erreurs que le corrigé
        return calcul(*args, **kwargs)
    fonction.__name__ = nom
    return fonction


# --------------------------------------------------------------------------- Défi 266 : erlang_c


@pytest.mark.parametrize("bug,morceau", [
    ("occupation", "OCCUPATION"),
    ("erlang_b", "Erlang B"),
    ("un_de_moins", "range(1, agents + 1)"),
    ("somme_jusqu_a_n", "de k = 0 à n - 1"),
    ("instable", "AVANT"),
])
def test_erlang_nomme_l_erreur(bug, morceau):
    _echec("erlang_c", _avec_les_gardes("erlang_c", lambda a, n: _ref_erlang_c(a, n, bug)), morceau)


def test_erlang_par_les_factorielles_deborde():
    def erlang_c(trafic, agents):
        corrige.erlang_c(trafic, agents)
        if trafic >= agents:
            return 1.0
        haut = trafic ** agents / math.factorial(agents) * agents / (agents - trafic)
        return haut / (sum(trafic ** k / math.factorial(k) for k in range(agents)) + haut)

    _echec("erlang_c", erlang_c, "OverflowError", "récurrence")


def test_erlang_qui_commence_a_zero_divise_par_zero():
    def erlang_c(trafic, agents):
        corrige.erlang_c(trafic, agents)
        if trafic >= agents:
            return 1.0
        b = 1.0
        for k in range(agents + 1):                    # k = 0 : 0 / 0 quand le trafic est nul
            b = trafic * b / (k + trafic * b)
        return agents * b / (agents - trafic * (1 - b))

    _echec("erlang_c", erlang_c, "ZeroDivisionError", "k = 0")


def test_erlang_sans_les_verifications():
    def erlang_c(trafic, agents):
        return 1.0 if trafic >= agents else corrige.erlang_c(trafic, agents)

    _echec("erlang_c", erlang_c, "erlang_c(inf, 3)", "ValueError")


def test_erlang_accepte_la_formule_directe_en_logarithmes():
    def erlang_c(trafic, agents):
        corrige.erlang_c(trafic, agents)
        if trafic >= agents:
            return 1.0
        if trafic == 0:
            return 0.0
        logs = [k * math.log(trafic) - math.lgamma(k + 1) for k in range(agents)]
        dernier = agents * math.log(trafic) - math.lgamma(agents + 1) + math.log(agents / (agents - trafic))
        m = max(logs + [dernier])
        return math.exp(dernier - m) / (sum(math.exp(x - m) for x in logs) + math.exp(dernier - m))

    _reussi("erlang_c", erlang_c)


def test_erlang_accepte_la_recurrence_sur_l_inverse():
    def erlang_c(trafic, agents):
        corrige.erlang_c(trafic, agents)
        if trafic >= agents:
            return 1.0
        if trafic == 0:
            return 0.0
        inverse = 1.0                                   # 1 / B(k) = 1 + k / A × 1 / B(k - 1)
        for k in range(1, agents + 1):
            inverse = 1 + k / trafic * inverse
        b = 1 / inverse
        return b / (1 - trafic / agents * (1 - b))

    _reussi("erlang_c", erlang_c)


def test_la_reference_d_erlang_est_la_formule_exacte():
    for trafic, agents in [(Fraction(2), 3), (Fraction(12), 14), (Fraction(37, 4), 11), (Fraction(1, 3), 1),
                           (Fraction(199, 10), 25), (Fraction(0), 4)]:
        termes = [trafic ** k / math.factorial(k) for k in range(agents)]
        haut = trafic ** agents / math.factorial(agents) * agents / (agents - trafic)
        exact = haut / (sum(termes) + haut)
        assert math.isclose(_ref_erlang_c(float(trafic), agents), float(exact), rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(_ref_erlang_c(10, 11), 0.6821, abs_tol=5e-5)          # la valeur des manuels


def test_la_simulation_sans_abandon_retrouve_erlang_c():
    """Deux méthodes indépendantes : la formule (défi 266) et la simulation (défi 268), sur 100 jours d'appels."""
    appels = generer_appels(80, jours=100, graine=4466, part_patients=1.0)    # 8 erlangs, personne ne raccroche
    resultats = corrige.simuler_file(appels, 10)
    attendent = sum(a > 0 for a, _ in resultats) / len(resultats)
    a_temps = sum(a <= 20 for a, _ in resultats) / len(resultats)
    assert abs(attendent - corrige.erlang_c(8, 10)) < 0.02
    assert abs(a_temps - corrige._taux_de_service(8, 10, 360, 20)) < 0.02


# --------------------------------------------------------------------------- Défi 267 : conseillers_necessaires


@pytest.mark.parametrize("bug,morceau", [
    ("minutes", "3 600"),
    ("heures", "erlangs"),
    ("occupation", "85 %"),
    ("sans_attente", "SANS attendre"),
    ("sans_moins_a", "(n - A)"),
    ("absents_plus", "n / 0,7"),
    ("absents_float", "30.000000000000004"),
    ("absents_round", "vers le HAUT"),
])
def test_conseillers_nomme_l_erreur(bug, morceau):
    def calcul(appels, duree, objectif=0.8, seuil_s=20, absents_pct=0):
        return _ref_conseillers(appels, duree, objectif, seuil_s, absents_pct, bug)

    _echec("conseillers_necessaires", _avec_les_gardes("conseillers_necessaires", calcul), morceau)


def test_conseillers_un_de_moins():
    def calcul(appels, duree, objectif=0.8, seuil_s=20, absents_pct=0):
        n = _ref_conseillers(appels, duree, objectif, seuil_s)
        return 0 if appels == 0 else -(-(n - 1) * 100 // (100 - absents_pct))

    _echec("conseillers_necessaires", _avec_les_gardes("conseillers_necessaires", calcul), "PREMIER n")


def test_conseillers_un_float_est_refuse():
    _echec("conseillers_necessaires", _avec_les_gardes("conseillers_necessaires",
                                                       lambda *a, **k: float(corrige.conseillers_necessaires(*a, **k))),
           "un int")


def test_conseillers_zero_appel():
    def calcul(appels, duree, objectif=0.8, seuil_s=20, absents_pct=0):
        return max(1, corrige.conseillers_necessaires(appels, duree, objectif, seuil_s, absents_pct))

    _echec("conseillers_necessaires", _avec_les_gardes("conseillers_necessaires", calcul), "zéro conseiller")


def test_conseillers_utilise_l_erlang_c_prete(monkeypatch):
    """Pendant le 267, le correcteur prête SA erlang_c : un 266 faux ne fait pas échouer un 267 juste."""
    faux = SimpleNamespace(__name__="faux_module_267")

    def erlang_c(trafic, agents):
        raise NotImplementedError

    def conseillers_necessaires(appels, duree, objectif=0.8, seuil_s=20, absents_pct=0):
        corrige.conseillers_necessaires(appels, duree, objectif, seuil_s, absents_pct)
        if appels == 0:
            return 0
        trafic = appels * duree / 3600
        n = math.floor(trafic) + 1
        while 1 - faux.erlang_c(trafic, n) * math.exp(-(n - trafic) * seuil_s / duree) < objectif:
            n += 1
        return -(-n * 100 // (100 - absents_pct))

    faux.erlang_c = erlang_c
    conseillers_necessaires.__module__ = "faux_module_267"
    monkeypatch.setitem(sys.modules, "faux_module_267", faux)
    _reussi("conseillers_necessaires", conseillers_necessaires)
    assert faux.erlang_c is erlang_c                   # la vôtre est remise en place


def test_conseillers_accepte_une_dichotomie():
    def conseillers_necessaires(appels, duree, objectif=0.8, seuil_s=20, absents_pct=0):
        corrige.conseillers_necessaires(appels, duree, objectif, seuil_s, absents_pct)
        if appels == 0:
            return 0
        trafic = appels * duree / 3600

        def tient(n):
            return n > trafic and corrige._taux_de_service(trafic, n, duree, seuil_s) >= objectif

        haut = math.floor(trafic) + 1
        while not tient(haut):
            haut *= 2
        bas = math.floor(trafic)                      # bas ne tient pas, haut tient
        while haut - bas > 1:
            milieu = (bas + haut) // 2
            bas, haut = (bas, milieu) if tient(milieu) else (milieu, haut)
        return -(-haut * 100 // (100 - absents_pct))

    _reussi("conseillers_necessaires", conseillers_necessaires)


def test_conseillers_qui_partent_de_un_avec_un_trafic_en_secondes_sont_arretes():
    def conseillers_necessaires(appels, duree, objectif=0.8, seuil_s=20, absents_pct=0):
        corrige.conseillers_necessaires(appels, duree, objectif, seuil_s, absents_pct)
        if appels == 0:
            return 0
        trafic = appels * duree                        # des secondes, pas des erlangs
        n = 1
        while n <= trafic or 1 - corrige.erlang_c(trafic, n) * math.exp(-(n - trafic) * seuil_s / duree) < objectif:
            n += 1
        return n

    _echec("conseillers_necessaires", conseillers_necessaires, "trop lent", "3 600")


def test_la_reference_des_conseillers_est_une_recherche_exhaustive():
    for appels, duree, kwargs in _cas_267():
        objectif, seuil = kwargs.get("objectif", 0.8), kwargs.get("seuil_s", 20)
        pct = kwargs.get("absents_pct", 0)
        if appels == 0:
            assert _ref_conseillers(appels, duree, objectif, seuil, pct) == 0
            continue
        trafic = appels * duree / 3600
        n = next(n for n in range(1, 10_000) if n > trafic
                 and 1 - corrige.erlang_c(trafic, n) * math.exp(-(n - trafic) * seuil / duree) >= objectif)
        m = next(m for m in range(n, 10 * n + 2) if Fraction(m) * (100 - pct) / 100 >= n)
        assert _ref_conseillers(appels, duree, objectif, seuil, pct) == m, (appels, duree, kwargs)


def test_le_piege_de_l_arrondi_en_float_est_dans_les_cas():
    pieges = [(a, d, k) for a, d, k in _cas_267() if k.get("absents_pct")
              and _ref_conseillers(a, d, k.get("objectif", 0.8), k.get("seuil_s", 20), k["absents_pct"], "absents_float")
              != _ref_conseillers(a, d, k.get("objectif", 0.8), k.get("seuil_s", 20), k["absents_pct"])]
    assert len(pieges) >= 5
    assert math.ceil(21 / (1 - 0.3)) == 31 and -(-21 * 100 // 70) == 30


def test_l_erlang_pretee_est_juste_et_rapide():
    alea = random.Random(2670)
    for _ in range(500):
        n = alea.randint(1, 120)
        trafic = round(alea.uniform(0, n * 1.1), 3)
        assert math.isclose(_erlang_pretee(trafic, n), _ref_erlang_c(trafic, n), rel_tol=1e-12, abs_tol=1e-15)
    for n in (400, 401, 402, 399, 450):               # elle reprend où elle en était, ou repart de zéro
        assert math.isclose(_erlang_pretee(390.5, n), _ref_erlang_c(390.5, n), rel_tol=1e-12, abs_tol=1e-15)
    with pytest.raises(TypeError):
        _erlang_pretee(12, 14.0)


# --------------------------------------------------------------------------- Défi 268 : simuler_file


@pytest.mark.parametrize("bug,morceau", [
    ("dans_l_ordre", "rangé par ligne"),
    ("abandon_occupe", "AUCUN conseiller"),
    ("attente_reelle", "PATIENCE"),
    ("abandon_large", "EXACTEMENT"),
    ("sans_abandon", "ne raccrochent jamais"),
    ("fin_sans_attente", "DÉBUT + durée"),
    ("tourniquet", "tour de rôle"),
])
def test_simulation_nomme_l_erreur(bug, morceau):
    _echec("simuler_file", _avec_les_gardes("simuler_file", lambda appels, n: _ref_simuler(appels, n, bug)), morceau)


def test_simulation_qui_trie_la_liste_recue():
    def simuler_file(appels, conseillers):
        resultats = corrige.simuler_file(appels, conseillers)
        if isinstance(appels, list):
            appels.sort()
        return resultats

    _echec("simuler_file", simuler_file, "a modifié")


def test_simulation_qui_rend_des_listes():
    _echec("simuler_file", lambda appels, n: [list(r) for r in corrige.simuler_file(appels, n)], "TUPLE")


def test_simulation_qui_rend_0_et_1():
    _echec("simuler_file", lambda appels, n: [(a, int(b)) for a, b in corrige.simuler_file(appels, n)], "bool")


def test_simulation_accepte_min_sur_une_liste():
    def simuler_file(appels, conseillers):
        corrige.simuler_file(appels, conseillers)
        libres = [0] * conseillers
        resultats = [None] * len(appels)
        for i in sorted(range(len(appels)), key=lambda i: appels[i][0]):
            arrivee, duree, patience = appels[i]
            k = libres.index(min(libres))
            debut = max(arrivee, libres[k])
            if debut - arrivee > patience:
                resultats[i] = (float(patience), True)
            else:
                resultats[i] = (float(debut - arrivee), False)
                libres[k] = debut + duree
        return resultats

    _reussi("simuler_file", simuler_file)


def test_simulation_accepte_une_liste_triee_et_bisect():
    def simuler_file(appels, conseillers):
        corrige.simuler_file(appels, conseillers)
        libres = [0.0] * conseillers                   # toujours triée
        resultats = [None] * len(appels)
        ordre = sorted(enumerate(appels), key=lambda paire: paire[1][0])
        for i, (arrivee, duree, patience) in ordre:
            attente = max(0.0, libres[0] - arrivee)
            if attente > patience:
                resultats[i] = (patience * 1.0, True)
                continue
            libres.pop(0)
            bisect.insort(libres, arrivee + attente + duree)
            resultats[i] = (attente * 1.0, False)
        return resultats

    _reussi("simuler_file", simuler_file)


def test_simulation_qui_parcourt_tous_les_conseillers_est_arretee():
    def simuler_file(appels, conseillers):
        corrige.simuler_file(appels, conseillers)
        libres = [0.0] * conseillers
        resultats = [None] * len(appels)
        for i in sorted(range(len(appels)), key=lambda i: appels[i][0]):
            arrivee, duree, patience = appels[i]
            k = 0
            for j in range(conseillers):
                if libres[j] < libres[k]:
                    k = j
            debut = max(arrivee, libres[k])
            if debut - arrivee > patience:
                resultats[i] = (float(patience), True)
            else:
                resultats[i] = (float(debut - arrivee), False)
                libres[k] = debut + duree
        return resultats

    _echec("simuler_file", simuler_file, "trop lent", "TAS")


def test_la_reference_de_simulation_est_une_simulation_naive():
    """Une simulation écrite autrement : un conseiller = un objet, on cherche le premier libre en parcourant tout."""
    for appels, n in _cas_268()[:-6] + [(appels_exemple()[:1500], 9)]:
        fin = [0.0] * n
        attendu = [None] * len(appels)
        ordre = sorted(range(len(appels)), key=lambda i: (appels[i][0], i))
        for i in ordre:
            arrivee, duree, patience = appels[i]
            k = min(range(n), key=lambda j: fin[j])
            depart = fin[k] if fin[k] > arrivee else arrivee
            if depart - arrivee <= patience:
                fin[k] = depart + duree
                attendu[i] = (depart - arrivee, False)
            else:
                attendu[i] = (patience, True)
        obtenu = _ref_simuler(appels, n)
        assert [b for _, b in obtenu] == [b for _, b in attendu]
        assert all(math.isclose(x, y, abs_tol=1e-9) for (x, _), (y, _) in zip(obtenu, attendu))


# --------------------------------------------------------------------------- Défi 269 : indicateurs


@pytest.mark.parametrize("bug,morceau", [
    ("service_sur_tous", "TOUS les appels"),
    ("service_sur_decroches", "DÉCROCHÉS"),
    ("strict", "20 secondes pile"),
    ("asa_avec_abandons", "jamais été décroché"),
    ("asa_sur_tous", "nombre de décrochés"),
    ("p90_interpole", "INTERPOLÉ"),
    ("p90_rang", "un rang trop loin"),
    ("p90_tous", "compte les abandons"),
    ("abandon_sur_decroches", "TOUS les appels"),
])
def test_indicateurs_nomme_l_erreur(bug, morceau):
    _echec("indicateurs", _avec_les_gardes("indicateurs", lambda r, s=20: _ref_indicateurs(r, s, bug)), morceau)


def test_indicateurs_zero_au_lieu_de_none():
    def indicateurs(resultats, seuil_s=20):
        return {c: (0 if v is None else v) for c, v in corrige.indicateurs(resultats, seuil_s).items()}

    _echec("indicateurs", indicateurs, "None")


def test_indicateurs_avec_une_cle_en_trop():
    _echec("indicateurs", lambda r, s=20: {**corrige.indicateurs(r, s), "occupation": 0.8}, "EXACTEMENT", "occupation")


def test_indicateurs_qui_trie_la_liste_recue():
    def indicateurs(resultats, seuil_s=20):
        r = corrige.indicateurs(resultats, seuil_s)
        if isinstance(resultats, list):
            resultats.sort()
        return r

    _echec("indicateurs", indicateurs, "modifié")


def test_indicateurs_accepte_statistics_et_un_seul_passage():
    def indicateurs(resultats, seuil_s=20):
        corrige.indicateurs(resultats, seuil_s)
        decroches = sorted(a for a, abandon in resultats if not abandon)
        tardifs = sum(1 for a, abandon in resultats if abandon and a > seuil_s)
        juges = len(decroches) + tardifs
        k = len(decroches)
        return {
            "appels": len(resultats), "decroches": k, "abandons": len(resultats) - k,
            "taux_service": bisect.bisect_right(decroches, seuil_s) / juges if juges else None,
            "taux_abandon": (len(resultats) - k) / len(resultats) if resultats else None,
            "asa": statistics.fmean(decroches) if decroches else None,
            "p90": decroches[-(-9 * k // 10) - 1] if decroches else None,
        }

    _reussi("indicateurs", indicateurs)


def test_la_reference_des_indicateurs_compte_a_la_main():
    for resultats, seuil in _cas_269():
        r = _ref_indicateurs(resultats, seuil)
        decroches = [a for a, ab in resultats if not ab]
        for p in ([] if not decroches else [r["p90"]]):
            assert sum(a <= p for a in decroches) >= 0.9 * len(decroches)          # au moins 90 % en dessous…
            assert sum(a < p for a in decroches) < 0.9 * len(decroches)            # …et c'est la plus petite
            assert p in decroches
        if r["taux_service"] is not None:
            assert 0 <= r["taux_service"] <= 1
        assert r["appels"] == r["decroches"] + r["abandons"] == len(resultats)


# --------------------------------------------------------------------------- Défi 270 : planifier_vacations


@pytest.mark.parametrize("bug,morceau", [
    ("deborde", "fermeture"),
    ("au_plus_tot", "TÔT"),
    ("fins_oubliees", "finies"),
    ("pic", "PIC"),
])
def test_planning_nomme_l_erreur(bug, morceau):
    _echec("planifier_vacations", _avec_les_gardes("planifier_vacations",
                                                   lambda b, d=VACATION: _ref_planifier(b, d, bug)), morceau)


def test_planning_qui_modifie_les_besoins():
    def planifier_vacations(besoins, duree=VACATION):
        r = corrige.planifier_vacations(besoins, duree)
        if isinstance(besoins, list) and besoins:
            besoins[0] = 0
        return r

    _echec("planifier_vacations", planifier_vacations, "modifié")


def test_planning_une_vacation_a_la_fois_est_arrete():
    def planifier_vacations(besoins, duree=VACATION):
        corrige.planifier_vacations(besoins, duree)
        debuts, presents = [0] * len(besoins), [0] * len(besoins)
        for t in range(len(besoins)):
            while presents[t] < besoins[t]:
                debut = min(t, len(besoins) - duree)
                debuts[debut] += 1
                for u in range(debut, debut + duree):
                    presents[u] += 1
        return debuts

    _echec("planifier_vacations", planifier_vacations, "trop lent", "d'un coup")


def test_planning_accepte_le_glouton_de_droite_a_gauche():
    def planifier_vacations(besoins, duree=VACATION):
        corrige.planifier_vacations(besoins, duree)
        total = len(besoins)
        debuts, presents = [0] * total, [0] * total
        for t in reversed(range(total)):
            manque = besoins[t] - presents[t]
            if manque > 0:
                debut = max(0, t - duree + 1)
                debuts[debut] += manque
                for u in range(debut, debut + duree):
                    presents[u] += manque
        return debuts

    # c'est un AUTRE planning (les vacations commencent le plus tôt possible), avec le même nombre de vacations
    besoins = [corrige.conseillers_necessaires(n, DUREE_MOYENNE_S) for _, n in prevision_exemple()]
    assert planifier_vacations(besoins) != corrige.planifier_vacations(besoins)
    _reussi("planifier_vacations", planifier_vacations)


def test_planning_accepte_la_programmation_lineaire():
    optimize = pytest.importorskip("scipy.optimize")
    np = pytest.importorskip("numpy")

    def planifier_vacations(besoins, duree=VACATION):
        corrige.planifier_vacations(besoins, duree)
        total = len(besoins)
        if not total:
            return []
        n = total - duree + 1                         # les débuts possibles
        couverture = np.array([[1 if s <= t < s + duree else 0 for s in range(n)] for t in range(total)])
        contraintes = optimize.LinearConstraint(couverture, lb=np.array(besoins, dtype=float), ub=np.inf)
        r = optimize.milp(c=np.ones(n), constraints=contraintes, integrality=np.ones(n),
                          bounds=optimize.Bounds(0, np.inf))
        return [int(round(x)) for x in r.x] + [0] * (duree - 1)

    _reussi("planifier_vacations", planifier_vacations)


def test_le_glouton_de_reference_est_optimal():
    optimize = pytest.importorskip("scipy.optimize")
    np = pytest.importorskip("numpy")
    alea = random.Random(2700)
    cas = [(b, d) for b, d in _cas_270() if b] + [([alea.randint(0, 30) for _ in range(alea.randint(1, 40))], 0)
                                                for _ in range(60)]
    for besoins, duree in cas:
        duree = duree or alea.randint(1, len(besoins))
        total = len(besoins)
        n = total - duree + 1
        couverture = np.array([[1 if s <= t < s + duree else 0 for s in range(n)] for t in range(total)])
        r = optimize.milp(c=np.ones(n), constraints=optimize.LinearConstraint(couverture, lb=np.array(besoins, float),
                                                                              ub=np.inf),
                          integrality=np.ones(n), bounds=optimize.Bounds(0, np.inf))
        assert r.success
        assert sum(_ref_planifier(besoins, duree)) == round(r.fun), (besoins, duree)


# --------------------------------------------------------------------------- Défi 271 : FilePrioritaire


def _classe_boguee(bug):
    class FilePrioritaire(_RefFile):
        def __init__(self, vieillissement=0):
            corrige.FilePrioritaire(vieillissement)  # les mêmes erreurs que le corrigé
            super().__init__(vieillissement, bug)

    return FilePrioritaire


@pytest.mark.parametrize("bug,morceau", [
    ("sans_vieillissement", "famine"),
    ("fifo", "ordre d'arrivée"),
    ("a_l_envers", "ARRIVÉS TARD"),
    ("le_moins_prioritaire", "plus PETITE"),
    ("egalite_identifiant", "IDENTIFIANT"),
    ("egalite_dernier", "DERNIER entré"),
    ("fantome", "servi deux fois"),
    ("perdu", "n'est jamais servi"),
    ("len_tas", "EN LIGNE"),
])
def test_file_nomme_l_erreur(bug, morceau):
    _echec("FilePrioritaire", _classe_boguee(bug), morceau)


class _Naive:
    """Une file naïve : à chaque `suivant`, la priorité de chacun à l'instant t, et le maximum."""

    def __init__(self, vieillissement=0):
        corrige.FilePrioritaire(vieillissement)
        self.v, self.clients, self.horloge, self.numero = vieillissement, {}, 0, 0
        self.verif = corrige.FilePrioritaire(vieillissement)   # pour lever les mêmes erreurs, au bon moment

    def ajouter(self, client, risque, arrivee):
        self.verif.ajouter(client, risque, arrivee)
        self.clients[client] = (risque, arrivee, self.numero)
        self.numero += 1

    def retirer(self, client):
        self.verif.retirer(client)
        del self.clients[client]

    def suivant(self, maintenant):
        self.verif.suivant(maintenant)
        meilleur = max(self.clients, key=lambda c: (self.clients[c][0] + self.v * (maintenant - self.clients[c][1]) / 60,
                                                    -self.clients[c][2]))
        del self.clients[meilleur]
        return meilleur

    def __len__(self):
        return len(self.clients)


def test_la_file_naive_est_juste_mais_arretee():
    message = _echec("FilePrioritaire", _Naive, "trop lent", "ORDRE ne change jamais")
    assert "opérations sur 20 000 clients" in message


def test_la_reference_de_la_file_est_la_file_naive():
    """Recalculer la priorité de chacun à l'instant t (la définition) donne le même ordre que la clé fixe."""
    for v, operations in _scenarios_271():
        naive = _Naive.__new__(_Naive)
        naive.v, naive.clients, naive.horloge, naive.numero = v, {}, 0, 0
        naive.verif = corrige.FilePrioritaire(v)
        assert _jouer(naive, operations) == _jouer(_RefFile(v), operations) == _jouer(corrige.FilePrioritaire(v),
                                                                                      operations)


def test_file_accepte_la_recette_de_la_documentation_de_heapq():
    class FilePrioritaire:
        RETIRE = "<retiré>"

        def __init__(self, vieillissement=0):
            corrige.FilePrioritaire(vieillissement)
            self.v, self.tas, self.trouver, self.compteur = vieillissement, [], {}, itertools.count()
            self.verif = corrige.FilePrioritaire(vieillissement)

        def ajouter(self, client, risque, arrivee):
            self.verif.ajouter(client, risque, arrivee)
            entree = [-(risque * 60 - self.v * arrivee), next(self.compteur), client]   # la clé × 60 : même ordre
            self.trouver[client] = entree
            heapq.heappush(self.tas, entree)

        def retirer(self, client):
            self.verif.retirer(client)
            self.trouver.pop(client)[-1] = self.RETIRE

        def suivant(self, maintenant):
            client = self.verif.suivant(maintenant)
            while True:
                _, _, c = heapq.heappop(self.tas)
                if c is not self.RETIRE:
                    del self.trouver[c]
                    assert c == client
                    return c

        def __len__(self):
            return len(self.trouver)

    _reussi("FilePrioritaire", FilePrioritaire)


class _LeClientCommeCompteur:
    """Un faux compteur qui renvoie le client en cours d'ajout : les entrées deviennent [clé, client, client]."""

    courant = None

    def __iter__(self):
        return self

    def __next__(self):
        return self.courant


def test_file_sans_compteur_compare_les_clients():
    class FilePrioritaire(corrige.FilePrioritaire):
        def __init__(self, vieillissement=0):
            super().__init__(vieillissement)
            self._compteur = _LeClientCommeCompteur()

        def ajouter(self, client, risque, arrivee):
            self._compteur.courant = client
            super().ajouter(client, risque, arrivee)

    _echec("FilePrioritaire", FilePrioritaire, "IDENTIFIANT")


def test_file_qui_compare_des_clients_de_types_differents():
    class FilePrioritaire(corrige.FilePrioritaire):
        def suivant(self, maintenant):
            if len({type(c) for c in self._entrees}) > 1:      # ce que ferait heapq en comparant 7 et "sept"
                raise TypeError("'<' not supported between instances of 'str' and 'int'")
            return super().suivant(maintenant)

    _echec("FilePrioritaire", FilePrioritaire, "TypeError", "COMPTEUR")


def test_file_partagee_entre_instances():
    class FilePrioritaire(corrige.FilePrioritaire):
        _commun = {"tas": [], "entrees": {}}

        def __init__(self, vieillissement=0):
            super().__init__(vieillissement)
            self._tas, self._entrees = self._commun["tas"], self._commun["entrees"]

    _echec("FilePrioritaire", FilePrioritaire, "partagent")


def test_file_dont_l_horloge_avance_meme_quand_on_refuse():
    class FilePrioritaire(corrige.FilePrioritaire):
        def suivant(self, maintenant):
            if not self._tas or all(e[-1] is self._RETIRE for e in self._tas):
                self._horloge = max(self._horloge, maintenant)
            return super().suivant(maintenant)

    _echec("FilePrioritaire", FilePrioritaire, "suivant refusé")


def test_file_qui_verifie_le_vide_avant_l_heure():
    class FilePrioritaire(corrige.FilePrioritaire):
        def suivant(self, maintenant):
            if not len(self):
                raise IndexError("personne")
            return super().suivant(maintenant)

    _echec("FilePrioritaire", FilePrioritaire, "AVANT de regarder")


# --------------------------------------------------------------------------- outils, corrigé, histoire


def test_les_outils_communs():
    appels = appels_exemple()
    assert len(appels) == len(departs_exemple()) == 6806
    assert appels_exemple() is not appels and appels_exemple() == appels
    assert all(isinstance(a, int) and isinstance(d, int) and d >= 1 and (p == math.inf or isinstance(p, int))
               for a, d, p in appels)
    assert any(p == math.inf for _, _, p in appels)
    assert appels != sorted(appels)                    # l'export n'est pas dans l'ordre chronologique
    assert len({a for a, _, _ in appels}) < len(appels)       # des arrivées à la même seconde
    assert [h for h, _ in prevision_exemple()][:3] == ["08:00", "08:30", "09:00"] and len(prevision_exemple()) == 24
    assert sum(planning_actuel_exemple()) == 42
    assert generer_appels(60, graine=3) == generer_appels(60, graine=3)
    dans_l_ordre = [a for a, _, _ in generer_appels(60, graine=3, melanger=False)]
    assert dans_l_ordre == sorted(dans_l_ordre)
    assert {creneau(a) for a, _, _ in appels} == set(range(24))


def test_l_histoire_tient():
    besoins = [corrige.conseillers_necessaires(n, DUREE_MOYENNE_S) for _, n in prevision_exemple()]
    moyenne = sum(n for _, n in prevision_exemple()) / 24
    assert round(moyenne) == 111 and round(moyenne * 360 / 3600 / CONSEILLERS_ACTUELS, 2) == 0.79
    assert round(corrige.erlang_c(moyenne * 360 / 3600, CONSEILLERS_ACTUELS), 2) == 0.32
    assert max(besoins) == 23 and besoins[14] == 23 and min(besoins) == 6
    assert math.ceil(12 / 0.85) == 15 and round(corrige._taux_de_service(12, 15, 360, 20), 2) == 0.73

    appels, departs = appels_exemple(), departs_exemple()
    resultats = corrige.simuler_file(appels, CONSEILLERS_ACTUELS)
    semaine = corrige.indicateurs(resultats)
    assert (round(semaine["taux_service"], 2), round(semaine["taux_abandon"], 3)) == (0.73, 0.113)
    apres_midi = corrige.indicateurs([r for a, r in zip(appels, resultats) if 12 <= creneau(a[0]) < 18])
    matin = corrige.indicateurs([r for a, r in zip(appels, resultats) if creneau(a[0]) < 12])
    assert round(apres_midi["taux_service"], 2) == 0.52 and round(apres_midi["taux_abandon"], 1) == 0.2
    assert matin["taux_service"] > 0.8

    def taux(groupe):
        return sum(groupe) / len(groupe)

    raccroche = [d for (a, ab), d in zip(resultats, departs) if ab]
    long = [d for (a, ab), d in zip(resultats, departs) if not ab and a > 60]
    vite = [d for (a, ab), d in zip(resultats, departs) if not ab and a <= 60]
    assert (round(taux(raccroche), 2), round(taux(long), 2), round(taux(vite), 2)) == (0.26, 0.15, 0.06)

    debuts = corrige.planifier_vacations(besoins, VACATION)
    assert sum(debuts) == 53 and sum(planning_actuel_exemple()) == 42
    assert sum(corrige.planifier_vacations([max(besoins)] * 24, VACATION)) == 69
    presents_actuels = [sum(planning_actuel_exemple()[s] for s in range(max(0, t - 7), t + 1)) for t in range(24)]
    assert presents_actuels == [CONSEILLERS_ACTUELS] * 24

    plus = corrige.indicateurs(corrige.simuler_file(appels, 16))
    assert plus["taux_service"] > 0.8 > semaine["taux_service"]            # 16 toute la journée : 80 % en moyenne…


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    serie = next(s for s in series() if s.code == "44")
    assert [r.statut for r in evaluer(serie)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(serie, corrige=True)] == [REUSSI] * 6


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_44 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(attente)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_44 import *` (dans les exemples) doit donner les objets du corrigé
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
    appels = [[30, 10, 5], [0, 100, math.inf], [10, 5, 1]]
    resultats = [[5, False], [30, True]]
    besoins = [3, 1, 4, 1]
    copies = [repr(x) for x in (appels, resultats, besoins)]
    corrige.simuler_file(appels, 1)
    corrige.indicateurs(resultats)
    corrige.planifier_vacations(besoins, 2)
    assert [repr(x) for x in (appels, resultats, besoins)] == copies


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(OverflowError):                 # la raison d'être de la récurrence
        950.0 ** 1000
    with pytest.raises(TypeError):
        corrige.erlang_c(12, 14.0)
    with pytest.raises(ValueError):
        corrige.conseillers_necessaires(120, 360, objectif=80)
    with pytest.raises(TypeError):
        corrige.simuler_file([(0, 10)], 1)
    with pytest.raises(TypeError):
        corrige.indicateurs([(5, 1)])
    with pytest.raises(ValueError):
        corrige.planifier_vacations([1, 2], 3)
    file = corrige.FilePrioritaire()
    with pytest.raises(IndexError):
        file.suivant(0)
    with pytest.raises(KeyError):
        file.retirer("X")


def test_les_bugs_annonces_existent_tous():
    """Chaque erreur classique décrite par le correcteur a un test ci-dessus (et inversement)."""
    assert set(_BUGS_ERLANG) == {"occupation", "erlang_b", "un_de_moins", "somme_jusqu_a_n", "instable"}
    assert set(_BUGS_CONSEILLERS) == {"minutes", "heures", "occupation", "sans_attente", "sans_moins_a",
                                      "absents_plus", "absents_float", "absents_round"}
    assert set(_BUGS_SIMULATION) == {"dans_l_ordre", "abandon_occupe", "attente_reelle", "abandon_large",
                                     "sans_abandon", "fin_sans_attente", "tourniquet"}
    assert set(_BUGS_INDICATEURS) == {"service_sur_tous", "service_sur_decroches", "strict", "asa_avec_abandons",
                                      "asa_sur_tous", "p90_interpole", "p90_rang", "p90_tous", "abandon_sur_decroches"}
    assert set(_BUGS_PLANNING) == {"deborde", "au_plus_tot", "fins_oubliees", "pic"}
    assert set(_BUGS_FILE) == {"sans_vieillissement", "fifo", "a_l_envers", "le_moins_prioritaire",
                               "egalite_identifiant", "egalite_dernier", "fantome", "perdu", "len_tas"}


def test_chaque_bug_change_au_moins_un_cas():
    for bug in _BUGS_ERLANG:
        assert any(not math.isclose(_ref_erlang_c(a, n, bug), _ref_erlang_c(a, n), abs_tol=1e-9)
                   for a, n in _cas_266() if not (bug == "instable" and a >= n and a == n)), bug
    for bug in _BUGS_SIMULATION:
        assert any(_ref_simuler(a, n, bug) != _ref_simuler(a, n) for a, n in _cas_268()[:13]), bug
    for bug in _BUGS_INDICATEURS:
        assert any(_ref_indicateurs(r, s, bug) != _ref_indicateurs(r, s) for r, s in _cas_269()), bug
    for bug in _BUGS_PLANNING:
        assert any(_ref_planifier(b, d, bug) != _ref_planifier(b, d) for b, d in _cas_270()), bug
    for bug in _BUGS_FILE:
        assert any(_jouer(_RefFile(v, bug), ops) != _jouer(_RefFile(v), ops) for v, ops in _scenarios_271()), bug

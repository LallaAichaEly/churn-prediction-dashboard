"""Série 44 (décrocher à temps) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 266`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (pandas, une intégrale balayée, Erlang C en
  logarithmes avec `math.lgamma`, une recherche qui part de 1, un planning optimal différent, une simulation avec
  une liste triée et `bisect`…) ;
- les solutions trop lentes (chaque appel comparé à tous les autres, tous les conseillers passés en revue à chaque
  appel) sont arrêtées et expliquées ;
- les références du correcteur tiennent face à des vérifications indépendantes : un comptage seconde par seconde,
  la loi de Little, Erlang C en fractions exactes, le planning optimal d'un solveur en nombres entiers
  (`scipy.optimize.milp`), deux simulations écrites différemment ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (48 % de niveau de service et non 57 %, 15 % d'abandons, 16 % de départs chez ceux qui ont raccroché contre 6 %,
  70 % d'occupation mais une file sans fin à 10 h, 36 vacations bien placées qui donnent 86 % contre 62 % à plat).
"""

from __future__ import annotations

import bisect
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import defis.centre_appels as centre  # noqa: E402
import defis.corrige_serie_44 as corrige  # noqa: E402
from defis.centre_appels import (  # noqa: E402
    DUREE_VACATION_H,
    HEURES,
    demande_exemple,
    en_heure,
    generer_demande,
    horaires,
    journal_exemple,
    planning_actuel,
)
from defis.checks_serie_44 import (  # noqa: E402
    _BUGS_BESOIN,
    _BUGS_ERLANG,
    _BUGS_FILE,
    _BUGS_MESURER,
    _BUGS_PLANNING,
    _BUGS_SIMULER,
    SERIE,
    _cas_266,
    _cas_267,
    _cas_268,
    _cas_269,
    _cas_270,
    _cas_271,
    _couverture,
    _ErlangPretee,
    _journal_aleatoire,
    _ref_besoin,
    _ref_erlang,
    _ref_file,
    _ref_mesurer,
    _ref_planning,
    _ref_simuler,
    _ref_simuler_tas,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}

# Les écritures du défi 269 ci-dessous appellent `erlang_c` : le correcteur la remplace par la sienne pendant la
# vérification, comme dans votre fichier (et remet celle-ci ensuite).
erlang_c = corrige.erlang_c


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


def _echec_refus(nom, objet):
    """Une version sans vérification des entrées : refusée, parce qu'elle ne lève pas l'erreur attendue."""
    message = _echec(nom, objet)
    assert "devrait lever" in message or "au lieu de" in message, message


def _avec_les_gardes(nom, calcul):
    """Une version boguée qui garde les vérifications du corrigé (les erreurs attendues), pour isoler LE bug."""
    def fonction(*args, **kwargs):
        getattr(corrige, nom)(*args, **kwargs)        # lève les mêmes erreurs que le corrigé
        return calcul(*args, **kwargs)
    fonction.__name__ = nom
    return fonction


# --------------------------------------------------------------------------- Défi 266 : mesurer


@pytest.mark.parametrize("bug,morceau", [
    ("asa", "ASA"),
    ("abandon_zero", "comptée 0"),
    ("denominateur", "flatteur"),
    ("strict", "20 s pile"),
    ("abandons_comptes", "un abandon est un échec"),
    ("taux_sur_decroches", "taux d'abandon"),
])
def test_mesurer_nomme_l_erreur(bug, morceau):
    _echec("mesurer", _avec_les_gardes("mesurer", lambda appels, seuil_s=20: _ref_mesurer(list(appels), seuil_s, bug)),
           morceau)


def test_mesurer_zero_au_lieu_de_none_est_explique():
    def mesurer(appels, seuil_s=20):
        resultat = corrige.mesurer(appels, seuil_s)
        if resultat["attente_decroches_s"] is None:
            resultat["attente_decroches_s"] = 0.0
        return resultat

    _echec("mesurer", mesurer, "None (pas 0")


def test_mesurer_des_comptes_en_float_sont_refuses():
    def mesurer(appels, seuil_s=20):
        resultat = corrige.mesurer(appels, seuil_s)
        return {**resultat, "appels": float(resultat["appels"])}

    _echec("mesurer", mesurer, "un nombre d'appels est un int")


def test_mesurer_d_autres_ecritures_justes():
    def avec_pandas(appels, seuil_s=20):
        corrige.mesurer(appels, seuil_s)                       # les mêmes refus
        df = pd.DataFrame(list(appels), columns=["arrivee", "decroche", "fin"])
        decroche = df["decroche"].notna()
        attente = df["decroche"].where(decroche, df["fin"]) - df["arrivee"]
        return {"appels": len(df), "decroches": int(decroche.sum()), "abandons": int((~decroche).sum()),
                "taux_abandon": float((~decroche).mean()), "attente_moyenne_s": float(attente.mean()),
                "attente_decroches_s": float(attente[decroche].mean()) if decroche.any() else None,
                "niveau_service": float((decroche & (attente <= seuil_s)).mean())}

    def une_passe(appels, seuil_s=20):
        corrige.mesurer(appels, seuil_s)
        n = k = reussis = 0
        total = total_decroches = 0
        for appel in appels:
            n += 1
            if appel["decroche"] is None:
                total += appel["fin"] - appel["arrivee"]
            else:
                attente = appel["decroche"] - appel["arrivee"]
                k += 1
                total += attente
                total_decroches += attente
                reussis += attente <= seuil_s
        return {"niveau_service": reussis / n, "appels": n, "decroches": k, "abandons": n - k,
                "taux_abandon": 1 - k / n, "attente_moyenne_s": total / n,
                "attente_decroches_s": total_decroches / k if k else None}

    _reussi("mesurer", avec_pandas)
    _reussi("mesurer", une_passe)


def test_mesurer_refuse_ce_qu_il_faut():
    def sans_garde(appels, seuil_s=20):
        return _ref_mesurer(list(appels), seuil_s)

    _echec_refus("mesurer", sans_garde)


def test_mesurer_qui_modifie_le_journal_est_refuse():
    def mesurer(appels, seuil_s=20):
        resultat = corrige.mesurer(appels, seuil_s)
        for appel in appels:
            appel.pop("parti", None)
        return resultat

    _echec("mesurer", mesurer, "modifié le journal")


# --------------------------------------------------------------------------- Défi 267 : file_attente


@pytest.mark.parametrize("bug,morceau", [
    ("evenements", "moyenne dans le TEMPS"),
    ("ferme", "ouverte à droite"),
    ("conversation", "FIN de sa conversation"),
    ("sans_fenetre", "max(arrivee, debut)"),
    ("sans_tri", "sans les trier"),
])
def test_file_nomme_l_erreur(bug, morceau):
    _echec("file_attente", _avec_les_gardes("file_attente", lambda a, d, f: _ref_file(list(a), d, f, bug)), morceau)


def _file_quadratique(appels, debut, fin):
    """Juste, mais compte les clients en attente à chaque instant en reparcourant tout le journal."""
    corrige.file_attente(appels, debut, fin)
    attentes = []
    for appel in appels:
        b = appel["decroche"] if appel["decroche"] is not None else appel["fin"]
        attentes.append((max(appel["arrivee"], debut), min(b, fin)))
    total, maximum = 0, 0
    for a, b in attentes:
        if a < b:
            total += b - a
            compte = 0
            for x, y in attentes:
                if x <= a < y:
                    compte += 1
            maximum = max(maximum, compte)
    return total / (fin - debut), maximum


def test_file_quadratique_est_juste_mais_arretee():
    message = _echec("file_attente", _file_quadratique, "beaucoup trop lent", "triez UNE fois")
    assert "<10 000 appels>" in message


def test_file_d_autres_ecritures_justes():
    def integrale(appels, debut, fin):
        corrige.file_attente(appels, debut, fin)
        evenements = []
        for appel in appels:
            b = appel["decroche"] if appel["decroche"] is not None else appel["fin"]
            a, b = max(appel["arrivee"], debut), min(b, fin)
            if a < b:
                evenements += [(a, -0.5 + 1.5), (b, -1)]           # +1 et -1, écrits autrement
        evenements.sort(key=lambda e: (e[0], e[1]))
        aire, courant, maximum, avant = 0, 0, 0, debut
        for instant, variation in evenements:
            aire += courant * (instant - avant)
            courant += int(variation)
            maximum = max(maximum, courant)
            avant = instant
        return aire / (fin - debut), maximum

    def avec_numpy(appels, debut, fin):
        corrige.file_attente(appels, debut, fin)
        if not appels:
            return 0.0, 0
        a = np.array([x["arrivee"] for x in appels], dtype=float)
        b = np.array([x["decroche"] if x["decroche"] is not None else x["fin"] for x in appels], dtype=float)
        a, b = np.maximum(a, debut), np.minimum(b, fin)
        garde = a < b
        a, b = a[garde], b[garde]
        moyenne = float((b - a).sum() / (fin - debut))
        instants = np.concatenate([a, b])
        variations = np.concatenate([np.ones(len(a), dtype=int), -np.ones(len(b), dtype=int)])
        ordre = np.lexsort((variations, instants))
        cumul = np.cumsum(variations[ordre])
        return moyenne, int(cumul.max()) if len(cumul) else 0

    _reussi("file_attente", integrale)
    _reussi("file_attente", avec_numpy)


def test_file_refuse_ce_qu_il_faut():
    _echec_refus("file_attente", lambda a, d, f: _ref_file(list(a), d, f))


def _file_seconde_par_seconde(appels, debut, fin):
    """Pour des heures entières : le nombre de clients en attente est constant sur chaque [t, t + 1[."""
    comptes = []
    for t in range(debut, fin):
        comptes.append(sum(1 for x in appels if x["arrivee"] <= t < (x["decroche"] if x["decroche"] is not None
                                                                      else x["fin"])))
    return sum(comptes) / (fin - debut), max(comptes, default=0)


def test_la_file_de_reference_contre_un_comptage_seconde_par_seconde():
    for graine in range(5):
        journal = _journal_aleatoire(150, graine=5300 + graine, duree_s=1800)
        for debut, fin in ((0, 1800), (300, 900), (1799, 2400)):
            moyenne, maximum = _ref_file(journal, debut, fin)
            attendu = _file_seconde_par_seconde(journal, debut, fin)
            assert math.isclose(moyenne, attendu[0], rel_tol=1e-12) and maximum == attendu[1]


def test_la_loi_de_little_sur_le_lundi():
    journal = journal_exemple()
    m = corrige.mesurer(journal)
    moyenne, _ = corrige.file_attente(journal, 0, 12 * 3600)
    assert max(x["decroche"] if x["decroche"] is not None else x["fin"] for x in journal) <= 12 * 3600
    assert math.isclose(moyenne, m["appels"] / (12 * 3600) * m["attente_moyenne_s"], rel_tol=1e-12)


# --------------------------------------------------------------------------- Défi 268 : erlang_c


@pytest.mark.parametrize("bug,morceau", [
    ("erlang_b", "Erlang B"),
    ("occupation", "A / N"),
    ("somme_n", "k = N inclus"),
    ("poisson", "Poisson"),
])
def test_erlang_nomme_l_erreur(bug, morceau):
    _echec("erlang_c", _avec_les_gardes("erlang_c", lambda a, n: _ref_erlang(a, n, bug)), morceau)


def _erlang_manuel(trafic, conseillers):
    """La formule du manuel, avec des factorielles : juste, mais elle déborde."""
    corrige.erlang_c(trafic, conseillers)
    if trafic >= conseillers:
        return 1.0
    x = trafic ** conseillers / math.factorial(conseillers) * conseillers / (conseillers - trafic)
    s = sum(trafic ** k / math.factorial(k) for k in range(conseillers))
    return x / (s + x)


def test_erlang_qui_deborde_est_explique():
    message = _echec("erlang_c", _erlang_manuel, "débordé", "Erlang B")
    assert "erlang_c(380.5, 400)" in message                # le premier grand centre


def test_erlang_instable_est_explique():
    def sans_garde(trafic, conseillers):
        corrige.erlang_c(trafic, conseillers)
        if trafic == 0:
            return 0.0
        b = 1.0
        for k in range(1, conseillers + 1):
            b = trafic * b / (k + trafic * b)
        return conseillers * b / (conseillers - trafic * (1 - b))

    def division_par_zero(trafic, conseillers):
        corrige.erlang_c(trafic, conseillers)
        return sans_garde(trafic, conseillers) if trafic != conseillers else 1 / 0

    _echec("erlang_c", sans_garde, "file grossit sans fin", "renvoyez 1.0")
    _echec("erlang_c", division_par_zero, "division par zéro", "1.0")


def test_erlang_d_autres_ecritures_justes():
    def en_logarithmes(trafic, conseillers):
        corrige.erlang_c(trafic, conseillers)
        if trafic == 0:
            return 0.0
        if trafic >= conseillers:
            return 1.0
        logs = [k * math.log(trafic) - math.lgamma(k + 1) for k in range(conseillers + 1)]
        logs[conseillers] += math.log(conseillers / (conseillers - trafic))
        plus_grand = max(logs)
        termes = [math.exp(v - plus_grand) for v in logs]
        return termes[-1] / sum(termes)

    def par_le_haut(trafic, conseillers):
        corrige.erlang_c(trafic, conseillers)
        if trafic >= conseillers:
            return 1.0
        inverse_b = 1.0                                     # 1 / B(k) = 1 + k / A × 1 / B(k - 1)
        for k in range(1, conseillers + 1):
            inverse_b = 1 + k / trafic * inverse_b if trafic else math.inf
        b = 1 / inverse_b
        return conseillers * b / (conseillers - trafic * (1 - b))

    _reussi("erlang_c", en_logarithmes)
    _reussi("erlang_c", par_le_haut)


def test_erlang_refuse_ce_qu_il_faut():
    _echec_refus("erlang_c", lambda a, n: _ref_erlang(a, n))


def _erlang_exact(trafic: int, conseillers: int) -> Fraction:
    termes = [Fraction(trafic ** k, math.factorial(k)) for k in range(conseillers)]
    x = Fraction(trafic ** conseillers, math.factorial(conseillers)) * Fraction(conseillers, conseillers - trafic)
    return x / (sum(termes) + x)


def test_l_erlang_de_reference_contre_des_fractions_exactes():
    for trafic, conseillers in ((1, 2), (2, 3), (10, 12), (15, 20), (95, 100), (380, 400), (950, 1000)):
        assert math.isclose(_ref_erlang(trafic, conseillers), float(_erlang_exact(trafic, conseillers)), rel_tol=1e-11)


def test_l_economie_d_echelle():
    """À occupation égale (95 %), une grande équipe fait bien moins attendre qu'une petite."""
    assert _ref_erlang(9.5, 10) > 0.8 > 0.1 > _ref_erlang(950, 1000)


# --------------------------------------------------------------------------- Défi 269 : conseillers_necessaires


def _besoin_bogue(bug):
    def conseillers_necessaires(appels_par_heure, duree_moyenne_s, seuil_s=20, objectif=0.8):
        corrige.conseillers_necessaires(appels_par_heure, duree_moyenne_s, seuil_s, objectif)
        return _ref_besoin(appels_par_heure, duree_moyenne_s, seuil_s, objectif, bug)
    return conseillers_necessaires


@pytest.mark.parametrize("bug,morceau", [
    ("minutes", "divise par 60"),
    ("sans_conversion", "sans conversion"),
    ("occupation", "85 %"),
    ("sans_seuil", "n'attendent PAS du tout"),
    ("sans_duree", "divisez par la durée moyenne"),
])
def test_besoin_nomme_l_erreur(bug, morceau):
    _echec("conseillers_necessaires", _besoin_bogue(bug), morceau)


def test_besoin_qui_ne_rend_pas_zero_est_explique():
    def au_moins_un(appels_par_heure, duree_moyenne_s, seuil_s=20, objectif=0.8):
        return max(1, corrige.conseillers_necessaires(appels_par_heure, duree_moyenne_s, seuil_s, objectif))

    _echec("conseillers_necessaires", au_moins_un, "personne n'appelle")


def test_besoin_strict_ou_instable_est_explique():
    def un_de_moins(appels_par_heure, duree_moyenne_s, seuil_s=20, objectif=0.8):
        n = corrige.conseillers_necessaires(appels_par_heure, duree_moyenne_s, seuil_s, objectif)
        return n - 1 if n > 1 else n

    def le_trafic_arrondi(appels_par_heure, duree_moyenne_s, seuil_s=20, objectif=0.8):
        corrige.conseillers_necessaires(appels_par_heure, duree_moyenne_s, seuil_s, objectif)
        return math.floor(appels_par_heure * duree_moyenne_s / 3600)

    _echec("conseillers_necessaires", un_de_moins, "objectif")
    _echec("conseillers_necessaires", le_trafic_arrondi, "grossit sans fin")


def test_besoin_d_autres_ecritures_justes():
    def depuis_un(appels_par_heure, duree_moyenne_s, seuil_s=20, objectif=0.8):
        corrige.conseillers_necessaires(appels_par_heure, duree_moyenne_s, seuil_s, objectif)
        trafic = appels_par_heure * duree_moyenne_s / 3600
        if trafic == 0:
            return 0
        n = 1
        while n <= trafic or 1 - erlang_c(trafic, n) * math.exp(-(n - trafic) * seuil_s / duree_moyenne_s) < objectif:
            n += 1
        return n

    def par_dichotomie(appels_par_heure, duree_moyenne_s, seuil_s=20, objectif=0.8):
        corrige.conseillers_necessaires(appels_par_heure, duree_moyenne_s, seuil_s, objectif)
        trafic = appels_par_heure * duree_moyenne_s / 3600
        if trafic == 0:
            return 0

        def tient(n):
            return 1 - erlang_c(trafic, n) * math.exp(-(n - trafic) * seuil_s / duree_moyenne_s) >= objectif

        bas, haut = math.floor(trafic) + 1, math.floor(trafic) + 2
        while not tient(haut):
            bas, haut = haut, 2 * haut
        while bas < haut:
            milieu = (bas + haut) // 2
            if tient(milieu):
                haut = milieu
            else:
                bas = milieu + 1
        return bas

    _reussi("conseillers_necessaires", depuis_un)
    _reussi("conseillers_necessaires", par_dichotomie)
    assert erlang_c is corrige.erlang_c                       # rendue après la vérification


def test_l_erlang_pretee_remplace_celle_de_l_eleve_puis_la_rend():
    module = type(sys)("eleve_269")
    exec("def erlang_c(trafic, conseillers):\n    raise NotImplementedError\n\n"
         "def conseillers_necessaires(appels_par_heure, duree_moyenne_s, seuil_s=20, objectif=0.8):\n"
         "    import math\n"
         "    trafic = appels_par_heure * duree_moyenne_s / 3600\n"
         "    if trafic == 0:\n        return 0\n"
         "    n = math.floor(trafic) + 1\n"
         "    while 1 - erlang_c(trafic, n) * math.exp(-(n - trafic) * seuil_s / duree_moyenne_s) < objectif:\n"
         "        n += 1\n"
         "    return n\n", module.__dict__)
    sys.modules["eleve_269"] = module
    try:
        eleve = module.erlang_c
        resultat = evaluer_defi(module, DEFIS["conseillers_necessaires"])
        assert resultat.statut == ECHEC and "devrait lever" in resultat.message      # le calcul est juste ; pas les refus
        assert module.erlang_c is eleve
        with _ErlangPretee(module.conseillers_necessaires):
            assert module.conseillers_necessaires(190, 281) == 19
        assert module.erlang_c is eleve
    finally:
        del sys.modules["eleve_269"]


def test_les_besoins_de_reference_tiennent_l_objectif_au_plus_juste():
    for appels, duree, seuil, objectif in _cas_269():
        seuil = 20 if seuil is None else seuil
        objectif = 0.8 if objectif is None else objectif
        n = _ref_besoin(appels, duree, seuil, objectif)
        trafic = appels * duree / 3600
        if trafic == 0:
            assert n == 0
            continue

        def niveau(m):
            return 1 - _ref_erlang(trafic, m) * math.exp(-(m - trafic) * seuil / duree)

        assert n > trafic and niveau(n) >= objectif and (n - 1 <= trafic or niveau(n - 1) < objectif)


# --------------------------------------------------------------------------- Défi 270 : planning


def _planning_bogue(bug):
    def planning(besoins, duree):
        corrige.planning(besoins, duree)
        return _ref_planning(list(besoins), duree, bug)
    return planning


@pytest.mark.parametrize("bug,morceau", [
    ("plus_tot", "le plus TÔT possible"),
    ("sans_cumul", "COMMENCENT à l'heure h"),
    ("court", "une heure trop tôt"),
    ("long", "une heure de trop"),
    ("hors_horaires", "après la fermeture"),
])
def test_planning_nomme_l_erreur(bug, morceau):
    _echec("planning", _planning_bogue(bug), morceau)


def test_planning_a_plat_est_refuse():
    def au_pic(besoins, duree):
        corrige.planning(besoins, duree)
        n = len(besoins)
        debuts = [0] * n
        for s in range(0, n, duree):
            debuts[min(s, n - duree)] += max(besoins, default=0)
        return debuts

    _echec("planning", au_pic, "vacations, alors que")


def test_planning_un_autre_optimum_est_accepte():
    def le_plus_tot_qui_suffit(besoins, duree):
        """Un autre planning minimal : celui du corrigé, lu à l'envers (de la fin de journée vers le début)."""
        corrige.planning(besoins, duree)
        n = len(besoins)
        if not n:
            return []
        miroir = corrige.planning(list(reversed(besoins)), duree)
        debuts = [0] * n
        for s, nombre in enumerate(miroir):
            debuts[n - duree - s] += nombre
        return debuts

    exemple = le_plus_tot_qui_suffit([2, 3, 3, 1, 4], 2)
    assert exemple != corrige.planning([2, 3, 3, 1, 4], 2) and sum(exemple) == 9
    _reussi("planning", le_plus_tot_qui_suffit)


def test_planning_d_autres_ecritures_justes():
    def en_recomptant(besoins, duree):
        corrige.planning(besoins, duree)
        n = len(besoins)
        debuts = [0] * n
        for h in range(n):
            presents = sum(debuts[max(0, h - duree + 1):h + 1])
            if besoins[h] > presents:
                debuts[min(h, n - duree)] += besoins[h] - presents
        return debuts

    _reussi("planning", en_recomptant)


def test_planning_refuse_ce_qu_il_faut():
    _echec_refus("planning", lambda b, d: _ref_planning(list(b), d))


def _optimum_milp(besoins, duree):
    from scipy.optimize import Bounds, LinearConstraint, milp

    n = len(besoins)
    debuts = n - duree + 1
    matrice = np.array([[1 if s <= h < s + duree else 0 for s in range(debuts)] for h in range(n)])
    resultat = milp(c=np.ones(debuts), constraints=LinearConstraint(matrice, lb=np.array(besoins), ub=np.inf),
                    integrality=np.ones(debuts), bounds=Bounds(0, np.inf))
    assert resultat.success
    return round(resultat.fun)


def test_le_planning_de_reference_est_optimal():
    alea = random.Random(2700)
    for _ in range(60):
        n = alea.randint(1, 16)
        besoins, duree = [alea.randint(0, 9) for _ in range(n)], alea.randint(1, n)
        debuts = _ref_planning(besoins, duree)
        assert all(c >= b for c, b in zip(_couverture(debuts, duree, n), besoins))
        assert all(debuts[s] == 0 for s in range(n - duree + 1, n))
        assert sum(debuts) == _optimum_milp(besoins, duree), (besoins, duree)


def test_le_glouton_au_plus_tot_n_est_pas_optimal():
    """Pourquoi la règle dit « le plus TARD possible » : au plus tôt, il faut parfois plus de vacations."""
    perdants = [b for b, d in _cas_270() if sum(_ref_planning(list(b), d, "plus_tot")) > sum(_ref_planning(list(b), d))]
    assert perdants


# --------------------------------------------------------------------------- Défi 271 : simuler


def _simuler_bogue(bug):
    def simuler(appels, conseillers):
        corrige.simuler(appels, conseillers)
        return _ref_simuler_tas(appels, conseillers) if not bug else _ref_simuler(appels, conseillers, bug)
    return simuler


@pytest.mark.parametrize("bug,morceau", [
    ("sans_tri", "ordre des ARRIVÉES"),
    ("ordre_trie", "ordre de la liste reçue"),
    ("abandon_occupe", "occupe quand même"),
    ("patience_stricte", "EXACTEMENT sa patience"),
    ("sans_patience", "Personne ne raccroche"),
    ("un_seul", "qu'un client à la fois"),
    ("plus_petit_numero", "le plus LONGTEMPS"),
    ("fin_ignoree", "après la fin de leur service"),
    ("fin_incluse", "pas à `fin` pile"),
])
def test_simuler_nomme_l_erreur(bug, morceau):
    _echec("simuler", _simuler_bogue(bug), morceau)


def test_simuler_qui_renvoie_les_dict_recus_est_refuse():
    def simuler(appels, conseillers):
        resultat = corrige.simuler(appels, conseillers)
        for appel, sortie in zip(appels, resultat):
            appel.update(decroche=sortie["decroche"], fin=sortie["fin"])
        return list(appels)

    _echec("simuler", simuler, "COPIES")


def _simuler_quadratique(appels, conseillers):
    """Juste, mais retrouve la disponibilité des conseillers en relisant tous les appels déjà servis."""
    corrige.simuler(appels, conseillers)
    services = [(0, math.inf)] * conseillers if isinstance(conseillers, int) else list(conseillers)
    servis = []                                     # (numéro du conseiller, fin de l'appel, décroché)
    resultat = [None] * len(appels)
    for i in sorted(range(len(appels)), key=lambda i: appels[i]["arrivee"]):
        appel = appels[i]
        meilleur = None
        for k, (debut, fin) in enumerate(services):
            libre = debut
            for numero, fin_appel, _ in servis:
                if numero == k:
                    libre = max(libre, fin_appel)
            moment = max(appel["arrivee"], libre)
            if moment < fin and (meilleur is None or (moment, libre, k) < meilleur):
                meilleur = (moment, libre, k)
        if meilleur is not None and meilleur[0] - appel["arrivee"] <= appel["patience"]:
            servis.append((meilleur[2], meilleur[0] + appel["duree"], meilleur[0]))
            resultat[i] = {**appel, "decroche": meilleur[0], "fin": meilleur[0] + appel["duree"]}
        else:
            resultat[i] = {**appel, "decroche": None, "fin": appel["arrivee"] + appel["patience"]}
    return resultat


def test_simuler_lent_est_arrete():
    message = _echec("simuler", _simuler_quadratique, "beaucoup trop lent", "heapq")
    assert "100)" in message


def test_simuler_qui_passe_les_conseillers_en_revue_est_arrete():
    def en_revue(appels, conseillers):
        corrige.simuler(appels, conseillers)
        services = [(0, math.inf)] * conseillers if isinstance(conseillers, int) else list(conseillers)
        libres = [debut for debut, _ in services]
        resultat = [None] * len(appels)
        for i in sorted(range(len(appels)), key=lambda i: appels[i]["arrivee"]):
            appel = appels[i]
            meilleur = None
            for k in range(len(services)):
                moment = max(appel["arrivee"], libres[k])
                if moment < services[k][1] and (meilleur is None or (moment, libres[k], k) < meilleur):
                    meilleur = (moment, libres[k], k)
            if meilleur is not None and meilleur[0] - appel["arrivee"] <= appel["patience"]:
                libres[meilleur[2]] = meilleur[0] + appel["duree"]
                resultat[i] = {**appel, "decroche": meilleur[0], "fin": meilleur[0] + appel["duree"]}
            else:
                resultat[i] = {**appel, "decroche": None, "fin": appel["arrivee"] + appel["patience"]}
        return resultat

    _echec("simuler", en_revue, "beaucoup trop lent")
    assert _simulations(en_revue) == _simulations(corrige.simuler)       # juste, pourtant


def _simulations(simuler):
    return [[(x["decroche"], x["fin"]) for x in simuler(a, c)] for a, c in _cas_271()[:12]]


def test_simuler_d_autres_ecritures_justes():
    def deux_tas(appels, conseillers):
        """Un tas de (libre à partir de, numéro), les fins de service à part, et `heapreplace`."""
        corrige.simuler(appels, conseillers)
        services = [(0, math.inf)] * conseillers if isinstance(conseillers, int) else list(conseillers)
        tas = [(libre, k) for k, (libre, _) in enumerate(services)]
        heapq.heapify(tas)
        resultat = [None] * len(appels)
        for i in sorted(range(len(appels)), key=lambda i: (appels[i]["arrivee"], i)):
            appel = appels[i]
            while tas and max(appel["arrivee"], tas[0][0]) >= services[tas[0][1]][1]:
                heapq.heappop(tas)
            decroche = max(appel["arrivee"], tas[0][0]) if tas else None
            if decroche is not None and decroche <= appel["arrivee"] + appel["patience"]:
                _, k = heapq.heapreplace(tas, (decroche + appel["duree"], tas[0][1]))
                resultat[i] = dict(appel, decroche=decroche, fin=decroche + appel["duree"])
            else:
                resultat[i] = dict(appel, decroche=None, fin=appel["arrivee"] + appel["patience"])
        return resultat

    def liste_triee(appels, conseillers):
        """Une liste triée de (libre, numéro, fin), tenue avec bisect.insort : juste, et assez rapide ici."""
        corrige.simuler(appels, conseillers)
        services = [(0, math.inf)] * conseillers if isinstance(conseillers, int) else list(conseillers)
        libres = sorted((debut, k, fin) for k, (debut, fin) in enumerate(services))
        resultat = {}
        for i, appel in sorted(enumerate(appels), key=lambda e: e[1]["arrivee"]):
            while libres and max(appel["arrivee"], libres[0][0]) >= libres[0][2]:
                libres.pop(0)
            if libres and max(appel["arrivee"], libres[0][0]) - appel["arrivee"] <= appel["patience"]:
                libre, k, fin = libres.pop(0)
                decroche = max(appel["arrivee"], libre)
                bisect.insort(libres, (decroche + appel["duree"], k, fin))
                resultat[i] = {**appel, "decroche": decroche, "fin": decroche + appel["duree"]}
            else:
                resultat[i] = {**appel, "decroche": None, "fin": appel["arrivee"] + appel["patience"]}
        return [resultat[i] for i in range(len(appels))]

    _reussi("simuler", deux_tas)
    _reussi("simuler", liste_triee)


def test_simuler_refuse_ce_qu_il_faut():
    _echec_refus("simuler", lambda a, c: _ref_simuler_tas(a, c))


def test_les_deux_simulations_de_reference_sont_d_accord():
    alea = random.Random(2710)
    for graine in range(40):
        demande = generer_demande((alea.randint(20, 120), alea.randint(20, 120)), duree_mediane_s=alea.choice((60, 240)),
                                  patience_moyenne_s=alea.choice((30, 200, 600)), graine=graine)
        alea.shuffle(demande)
        if graine % 2:
            conseillers = alea.randint(0, 8)
        else:
            conseillers = [(alea.choice((0, 600, 1800, 3600)), alea.choice((3600, 5400, 7200, math.inf)))
                           for _ in range(alea.randint(1, 8))]
            conseillers = [(d, f) for d, f in conseillers if d < f]
        assert _ref_simuler_tas(demande, conseillers) == _ref_simuler(demande, conseillers)


def test_une_simulation_respecte_ses_invariants():
    demande = demande_exemple()
    services = horaires(planning_actuel())
    journal = _ref_simuler_tas(demande, services)
    for appel, sortie in zip(demande, journal):
        if sortie["decroche"] is None:
            assert sortie["fin"] == appel["arrivee"] + appel["patience"]
        else:
            assert 0 <= sortie["decroche"] - appel["arrivee"] <= appel["patience"]
            assert sortie["fin"] == sortie["decroche"] + appel["duree"]
    for t in range(0, 12 * 3600, 97):                         # jamais plus d'appels en cours que de conseillers
        en_cours = sum(1 for x in journal if x["decroche"] is not None and x["decroche"] <= t < x["fin"])
        decroches = sum(1 for x in journal if x["decroche"] == t)
        arrives = sum(1 for d, _ in services if d <= t)         # un appel en cours a un conseiller arrivé
        presents = sum(1 for d, f in services if d <= t < f)    # on ne décroche qu'en service
        assert en_cours <= arrives and decroches <= presents


def test_chaque_bug_de_simulation_a_son_petit_cas():
    petits = [(a, c) for a, c in _cas_271() if len(a) <= 6]
    for bug in _BUGS_SIMULER:
        assert any(_ref_simuler(a, c, bug) != _ref_simuler_tas(a, c) for a, c in petits), bug


# --------------------------------------------------------------------------- L'histoire et les outils


def test_les_outils_communs():
    demande, journal = demande_exemple(), journal_exemple()
    assert len(demande) == len(journal) == 1279
    copie = demande_exemple()
    copie[0]["duree"] = 0
    assert demande_exemple()[0]["duree"] == 85                   # de nouveaux objets à chaque appel
    assert [a["arrivee"] for a in demande] == sorted(a["arrivee"] for a in demande)
    assert all(isinstance(a[c], int) for a in demande for c in ("arrivee", "duree", "patience"))
    assert all(a["client"] == j["client"] and a["arrivee"] == j["arrivee"] for a, j in zip(demande, journal))
    assert generer_demande((50, 80), graine=3) == generer_demande((50, 80), graine=3)
    assert len({a["client"] for a in generer_demande((1500,) * 12, graine=4)}) == len(generer_demande((1500,) * 12,
                                                                                                     graine=4))
    assert en_heure(43_199) == "19:59:59" and horaires([1, 0, 2], 1) == [(0, 3600), (7200, 10800), (7200, 10800)]
    assert sum(planning_actuel()) == 30 and DUREE_VACATION_H == 4 and HEURES == 12


def test_le_journal_est_la_simulation_du_planning_actuel():
    simule = _ref_simuler_tas(demande_exemple(), horaires(planning_actuel()))
    journal = journal_exemple()
    assert [(s["decroche"], s["fin"]) for s in simule] == [(j["decroche"], j["fin"]) for j in journal]


def test_l_histoire_tient():
    journal, demande = journal_exemple(), demande_exemple()
    m = corrige.mesurer(journal)
    assert round(m["niveau_service"], 3) == 0.481 and round(m["niveau_service"] * m["appels"] / m["decroches"], 3) == 0.567
    assert m["abandons"] == 194 and round(m["taux_abandon"], 3) == 0.152

    def taux(groupe):
        return sum(a["parti"] for a in groupe) / len(groupe)

    raccroche = [a for a in journal if a["decroche"] is None]
    vite = [a for a in journal if a["decroche"] is not None and a["decroche"] - a["arrivee"] <= 20]
    lents = [a for a in journal if a["decroche"] is not None and a["decroche"] - a["arrivee"] > 20]
    assert taux(raccroche) > taux(lents) > taux(vite)
    assert round(taux(raccroche), 2) == 0.16 and round(taux(vite), 3) == 0.062

    occupe = sum(a["fin"] - a["decroche"] for a in journal if a["decroche"] is not None)
    assert round(occupe / (30 * 4 * 3600), 2) == 0.70
    moyenne, maximum = corrige.file_attente(journal, 2 * 3600, 3 * 3600)
    assert round(moyenne, 1) == 6.5 and maximum == 17

    par_heure = [sum(1 for a in demande if a["arrivee"] // 3600 == h) for h in range(HEURES)]
    duree = sum(a["duree"] for a in demande) / len(demande)
    assert corrige.erlang_c(par_heure[2] * duree / 3600, 10) == 1.0
    besoins = [corrige.conseillers_necessaires(par_heure[h], duree) for h in range(HEURES)]
    debuts = corrige.planning(besoins, DUREE_VACATION_H)
    assert sum(debuts) == 41
    assert round(corrige.mesurer(corrige.simuler(demande, horaires(debuts)))["niveau_service"], 3) == 0.943
    moins = corrige.planning([corrige.conseillers_necessaires(par_heure[h], duree, objectif=0.5) for h in range(HEURES)],
                             DUREE_VACATION_H)
    assert sum(moins) == 36 == sum([12, 0, 0, 0] * 3)
    bien_places = corrige.mesurer(corrige.simuler(demande, horaires(moins)))["niveau_service"]
    a_plat = corrige.mesurer(corrige.simuler(demande, horaires([12, 0, 0, 0] * 3)))["niveau_service"]
    assert round(bien_places, 3) == 0.862 and round(a_plat, 3) == 0.623
    avant = corrige.mesurer(corrige.simuler(demande, horaires([13, 0, 0, 0] * 3)))
    assert avant["niveau_service"] > m["niveau_service"]                       # avant septembre, c'était mieux


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    serie = next(s for s in series() if s.code == "44")
    assert [r.statut for r in evaluer(serie)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(serie, corrige=True)] == [REUSSI] * 6


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_44 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(centre)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_44 import *` (dans les exemples) doit donner les objets du corrigé
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
    journal = journal_exemple()[:50]
    demande = demande_exemple()[:50]
    besoins, services = [3, 1, 4], [(0, 600), (300, math.inf)]
    copies = [copy.deepcopy(x) for x in (journal, demande, besoins, services)]
    corrige.mesurer(journal)
    corrige.file_attente(journal, 0, 3600)
    corrige.planning(besoins, 2)
    sortie = corrige.simuler(demande, services)
    sortie[0]["arrivee"] = -1
    assert [journal, demande, besoins, services] == copies


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(ValueError):
        corrige.mesurer([])
    with pytest.raises(TypeError):
        corrige.file_attente([{"arrivee": "0", "decroche": None, "fin": 1}], 0, 10)
    with pytest.raises(ValueError):
        corrige.erlang_c(-1, 3)
    with pytest.raises(ValueError):
        corrige.conseillers_necessaires(100, 300, objectif=1)
    with pytest.raises(ValueError):
        corrige.planning([1, 2], 3)
    with pytest.raises(TypeError):
        corrige.simuler([{"arrivee": 0, "duree": 1, "patience": 1}], 1.0)


def test_les_bugs_annonces_existent_tous():
    """Chaque erreur classique décrite par le correcteur a un test ci-dessus (et inversement)."""
    assert set(_BUGS_MESURER) == {"asa", "abandon_zero", "denominateur", "strict", "abandons_comptes",
                                  "taux_sur_decroches"}
    assert set(_BUGS_FILE) == {"evenements", "ferme", "conversation", "sans_fenetre", "sans_tri"}
    assert set(_BUGS_ERLANG) == {"erlang_b", "occupation", "somme_n", "poisson"}
    assert set(_BUGS_BESOIN) == {"minutes", "sans_conversion", "occupation", "sans_seuil", "sans_duree"}
    assert set(_BUGS_PLANNING) == {"plus_tot", "sans_cumul", "court", "long", "hors_horaires"}
    assert set(_BUGS_SIMULER) == {"sans_tri", "ordre_trie", "abandon_occupe", "patience_stricte", "sans_patience",
                                  "un_seul", "plus_petit_numero", "fin_ignoree", "fin_incluse"}


def _differe(bogue, juste) -> bool:
    try:
        return bogue() != juste()
    except ZeroDivisionError:                                 # le rejeu n'a pas de sens sur ce cas
        return False


def test_chaque_bug_change_au_moins_un_cas():
    for bug in _BUGS_MESURER:
        assert any(_differe(lambda: _ref_mesurer(list(a), 20 if s is None else s, bug),
                            lambda: _ref_mesurer(list(a), 20 if s is None else s)) for a, s in _cas_266()), bug
    for bug in _BUGS_FILE:
        assert any(_ref_file(a, d, f, bug) != _ref_file(a, d, f) for a, d, f in _cas_267()), bug
    for bug in _BUGS_ERLANG:
        assert any(not math.isclose(_ref_erlang(a, n, bug), _ref_erlang(a, n), rel_tol=1e-7) for a, n in _cas_268()), bug
    for bug in _BUGS_BESOIN:
        assert any(_ref_besoin(a, d, 20 if s is None else s, 0.8 if o is None else o, bug)
                   != _ref_besoin(a, d, 20 if s is None else s, 0.8 if o is None else o)
                   for a, d, s, o in _cas_269()), bug
    for bug in _BUGS_PLANNING:
        assert any(_ref_planning(list(b), d, bug) != _ref_planning(list(b), d) for b, d in _cas_270()), bug

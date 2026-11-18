"""Série 31 (proposer la bonne offre) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 188`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (`permutations`, `Counter` de paires,
  lift calculé en deux divisions, tri d'une copie du journal, logarithme naturel dans le NDCG…) ;
- le corrigé et les références du correcteur donnent les mêmes résultats qu'un calcul naïf, écrit
  autrement (toutes les paires du catalogue, toutes les listes possibles, formules recopiées des manuels) ;
- les exemples des docstrings, les outils communs et le corrigé sont justes.
"""

from __future__ import annotations

import collections
import datetime as dt
import doctest
import heapq
import itertools
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.corrige_serie_31 as corrige  # noqa: E402
import defis.offres as offres  # noqa: E402
from defis.checks_serie_31 import (  # noqa: E402
    SERIE,
    _ref_cooc,
    _ref_decouper,
    _ref_diversifier,
    _ref_mesurer,
    _ref_recommander,
    _ref_regles,
)
from defis.offres import (  # noqa: E402
    LecturesEpuisees,
    Regle,
    SimilaritesComptees,
    abonnements_exemple,
    generer_abonnements,
    generer_paniers,
    paniers_exemple,
    populaires,
    similarites_cosinus,
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


# --------------------------------------------------------------------------- Défi 188 : cooccurrences


def _cooccurrences(defaut=None):
    """Écrit autrement que le corrigé (permutations : chaque paire ordonnée une fois), avec UN défaut."""

    def cooccurrences(paniers):
        cooc = {}
        for panier in paniers.values():
            if defaut == "none":
                offres_client = set(panier)
            else:
                if not panier:
                    continue
                offres_client = list(panier) if defaut == "doublons" else set(panier)
            if defaut == "modifie" and isinstance(panier, list):
                panier.sort()
            if defaut == "solitaire" and len(set(offres_client)) < 2:
                continue
            for a in offres_client:
                cooc.setdefault(a, {})
            if defaut == "soi":
                paires = itertools.product(offres_client, repeat=2)
            elif defaut == "un_sens":
                paires = itertools.combinations(sorted(set(offres_client)), 2)
            else:
                paires = itertools.permutations(offres_client, 2)
            for a, b in paires:
                if a == b and defaut != "soi":
                    continue
                cooc[a][b] = cooc[a].get(b, 0) + 1
                if defaut == "double":
                    cooc[b][a] = cooc[b].get(a, 0) + 1
        if defaut == "defaultdict":
            return collections.defaultdict(dict, cooc)
        if defaut == "zeros":
            for a in list(cooc):
                for b in cooc:
                    if b != a:
                        cooc[a].setdefault(b, 0)
        return cooc

    return cooccurrences


def _cooccurrences_lent(paniers):
    """Juste, mais essaie toutes les paires du catalogue."""
    detenteurs = {}
    for client, panier in paniers.items():
        for offre in set(panier or ()):
            detenteurs.setdefault(offre, set()).add(client)
    cooc = {offre: {} for offre in detenteurs}
    for a in detenteurs:
        for b in detenteurs:
            if a != b:
                communs = len(detenteurs[a] & detenteurs[b])
                if communs:
                    cooc[a][b] = communs
    return cooc


def _cooccurrences_counter(paniers):
    """Juste : un Counter de paires (frozenset), puis le dict de dict."""
    compte = collections.Counter()
    vues = set()
    for panier in paniers.values():
        offres_client = set(panier or ())
        vues |= offres_client
        compte.update(frozenset(p) for p in itertools.combinations(sorted(offres_client), 2))
    cooc = {offre: {} for offre in vues}
    for paire, n in compte.items():
        a, b = sorted(paire)
        cooc[a][b] = cooc[b][a] = n
    return cooc


def test_188_le_corrige_et_d_autres_ecritures_justes_reussissent():
    _reussi("cooccurrences", corrige.cooccurrences)
    _reussi("cooccurrences", _cooccurrences())
    _reussi("cooccurrences", _cooccurrences_counter)


@pytest.mark.parametrize("defaut,morceaux", [
    ("soi", ("sa propre voisine",)),
    ("un_sens", ("SYMÉTRIQUE", "DEUX sens")),
    ("double", ("DEUX fois",)),
    ("doublons", ("deux fois une option écrite deux fois", "set(panier)")),
    ("solitaire", ("'presse' n'est pas une clé", "{}")),
    ("defaultdict", ("defaultdict",)),
    ("zeros", ("vaut 0", ".get(b, 0)")),
    ("none", ("None", "set(panier or ())")),
    ("modifie", ("a modifié les paniers",)),
])
def test_188_chaque_erreur_classique_est_nommee(defaut, morceaux):
    _echec("cooccurrences", _cooccurrences(defaut), *morceaux)


def test_188_essayer_toutes_les_paires_du_catalogue_est_refuse():
    _echec("cooccurrences", _cooccurrences_lent, "beaucoup trop lent", "k(k-1)/2", "catalogue")


def test_188_la_reference_egale_le_calcul_naif():
    alea = random.Random(1)
    for _ in range(200):
        paniers = {f"C{i}": [alea.choice("abcde") for _ in range(alea.randint(0, 5))] for i in range(alea.randint(0, 7))}
        assert _ref_cooc(paniers) == _cooccurrences_lent(paniers) == corrige.cooccurrences(paniers)


# --------------------------------------------------------------------------- Défi 189 : regles_association


def _regles(defaut=None):
    """Écrit autrement que le corrigé (le lift en deux divisions), avec UN défaut."""

    def regles_association(paniers, support_min=2):
        if defaut != "valeur" and (not isinstance(support_min, int) or support_min < 1):
            raise ValueError("support_min")
        n = len(paniers)
        if defaut == "n_non_vides":
            n = sum(1 for p in paniers.values() if p)
        detenteurs = collections.Counter()
        paires = collections.Counter()
        for panier in paniers.values():
            offres_client = list(panier or ()) if defaut == "doublons" else sorted(set(panier or ()))
            detenteurs.update(offres_client)
            for a, b in itertools.permutations(offres_client, 2):
                if a != b:
                    paires[a, b] += 1
        regles = []
        for (si, alors), support in paires.items():
            if defaut == "un_sens" and si > alors:
                continue
            if support < support_min or (defaut == "strict" and support <= support_min):
                continue
            confiance = support / detenteurs[alors if defaut == "conf_sur_alors" else si]
            if defaut == "lift_sans_n":
                lift = support / (detenteurs[si] * detenteurs[alors])
            else:
                lift = (support / detenteurs[si]) / (detenteurs[alors] / n)
            valeur = support / len(paniers) if defaut == "support_fraction" else support
            regles.append(Regle(si, alors, valeur, confiance, lift))
        if defaut == "reverse":
            regles.sort(key=lambda r: (r.lift, r.confiance, r.si, r.alors), reverse=True)
        elif defaut == "tri_confiance":
            regles.sort(key=lambda r: (-r.confiance, -r.lift, r.si, r.alors))
        else:
            regles.sort(key=lambda r: (-round(r.lift, 9), -r.confiance, r.si, r.alors))
        return regles

    return regles_association


def _regles_lentes(paniers, support_min=2):
    """Juste, mais essaie toutes les paires d'options."""
    if not isinstance(support_min, int) or support_min < 1:
        raise ValueError
    detenteurs = {}
    for client, panier in paniers.items():
        for offre in set(panier or ()):
            detenteurs.setdefault(offre, set()).add(client)
    n, regles = len(paniers), []
    for a in detenteurs:
        for b in detenteurs:
            if a != b:
                support = len(detenteurs[a] & detenteurs[b])
                if support >= support_min:
                    regles.append(Regle(a, b, support, support / len(detenteurs[a]),
                                        support * n / (len(detenteurs[a]) * len(detenteurs[b]))))
    return sorted(regles, key=lambda r: (-r.lift, -r.confiance, r.si, r.alors))


def test_189_le_corrige_et_d_autres_ecritures_justes_reussissent():
    _reussi("regles_association", corrige.regles_association)
    _reussi("regles_association", _regles())


def test_189_des_tuples_simples_sont_acceptes():
    def regles_association(paniers, support_min=2):
        return [tuple(r) for r in corrige.regles_association(paniers, support_min)]

    _reussi("regles_association", regles_association)


@pytest.mark.parametrize("defaut,morceaux", [
    ("doublons", ("compte deux fois", "CLIENTS")),
    ("conf_sur_alors", ("n(si)", "n(alors)")),
    ("n_non_vides", ("TOUS les clients", "len(paniers)")),
    ("lift_sans_n", ("HASARD", "Il manque N")),
    ("strict", ("support >= support_min",)),
    ("un_sens", ("DEUX règles",)),
    ("support_fraction", ("NOMBRE de clients",)),
    ("tri_confiance", ("lift décroissant", "popularité")),
    ("reverse", ("ALPHABÉTIQUE", "reverse=True")),
    ("valeur", ("devrait lever ValueError", "ENTIER")),
])
def test_189_chaque_erreur_classique_est_nommee(defaut, morceaux):
    _echec("regles_association", _regles(defaut), *morceaux)


def test_189_essayer_toutes_les_paires_est_refuse():
    _echec("regles_association", _regles_lentes, "beaucoup trop lent", "cooccurrences")


def test_189_un_fichier_ou_le_188_n_est_pas_fait_le_dit():
    def cooccurrences(paniers):
        raise NotImplementedError

    def regles_association(paniers, support_min=2):
        if support_min < 1:
            raise ValueError
        return list(cooccurrences(paniers))

    resultat = evaluer_defi(SimpleNamespace(__name__="faux", regles_association=regles_association,
                                            cooccurrences=cooccurrences), DEFIS["regles_association"])
    assert resultat.statut == ECHEC and "défi 188" in resultat.message, resultat.message


def test_189_la_reference_egale_le_calcul_naif():
    alea = random.Random(2)
    for _ in range(150):
        paniers = {f"C{i}": [alea.choice("abcde") for _ in range(alea.randint(0, 5))] for i in range(alea.randint(1, 8))}
        support_min = alea.randint(1, 3)
        attendu = _regles_lentes(paniers, support_min)
        assert [tuple(r) for r in attendu] == _ref_regles(paniers, support_min)
        assert attendu == corrige.regles_association(paniers, support_min)


def test_189_le_lift_vaut_la_formule_du_manuel():
    # 14 clients ; cloud et antivirus : 2 clients chacun, toujours ensemble → P(a et b) / (P(a) P(b)) = (2/14) / (2/14)² = 7
    regle = corrige.regles_association(paniers_exemple(), 2)[0]
    assert (regle.si, regle.alors, regle.support) == ("antivirus", "cloud", 2)
    assert math.isclose(regle.lift, (2 / 14) / ((2 / 14) * (2 / 14))) and regle.lift == 7.0


# --------------------------------------------------------------------------- Défi 190 : recommander


def _recommander(defaut=None):
    """Écrit autrement que le corrigé (Counter et heapq.nsmallest), avec UN défaut."""

    def recommander(panier, voisins, populaires, k=3):
        if k < 0 and defaut != "negatif":
            raise ValueError("k")
        if defaut == "none":
            detenues = set(panier)
        else:
            detenues = set(panier or ())
        if defaut == "modifie" and isinstance(panier, list):
            panier.append("tv")
        scores = collections.Counter()
        source = list(panier or ()) if defaut == "doublons" else detenues
        for offre in source:
            ligne = voisins[offre] if defaut == "keyerror" else voisins.get(offre, {})
            for autre, s in ligne.items():
                if autre in detenues and defaut != "deja":
                    continue
                if defaut == "max":
                    scores[autre] = max(scores[autre], s)
                else:
                    scores[autre] += s
        if defaut == "zero":
            gardees = list(scores)
        else:
            gardees = [o for o in scores if scores[o] > 0]
        if defaut == "reverse":
            choix = sorted(gardees, key=lambda o: (scores[o], o), reverse=True)[:max(k, 0)]
        else:
            choix = heapq.nsmallest(max(k, 0), gardees, key=lambda o: (-scores[o], o))
        if defaut == "sans_plan_b":
            return choix
        plan_b = [o for o in populaires if o not in detenues or defaut == "deja"]
        if defaut == "plan_b_avant":
            tout = []
            for o in plan_b + choix:
                if o not in tout:
                    tout.append(o)
            return tout[:k]
        for o in plan_b:
            if len(choix) >= k:
                break
            if o not in choix or defaut == "plan_b_doublon":
                choix.append(o)
        return choix

    return recommander


def test_190_le_corrige_et_d_autres_ecritures_justes_reussissent():
    _reussi("recommander", corrige.recommander)
    _reussi("recommander", _recommander())


@pytest.mark.parametrize("defaut,morceaux", [
    ("deja", ("que le client a DÉJÀ",)),
    ("doublons", ("compte deux fois", "set(panier)")),
    ("max", ("SOMME",)),
    ("zero", ("similarité 0", "score > 0")),
    ("reverse", ("ALPHABÉTIQUE", "-scores[o], o")),
    ("plan_b_avant", ("APRÈS",)),
    ("sans_plan_b", ("complétez avec `populaires`",)),
    ("plan_b_doublon", ("deux fois",)),
    ("keyerror", ("KeyError", "voisins.get(offre, {})")),
    ("none", ("None", "set(panier or ())")),
    ("negatif", ("devrait lever ValueError",)),
    ("modifie", ("a modifié le panier",)),
])
def test_190_chaque_erreur_classique_est_nommee(defaut, morceaux):
    _echec("recommander", _recommander(defaut), *morceaux)


def test_190_la_reference_egale_une_recherche_exhaustive():
    """Le classement attendu, recalculé par une autre voie : score de CHAQUE option du catalogue, tri complet."""
    alea = random.Random(3)
    catalogue = list("abcdefg")
    for _ in range(300):
        voisins = {a: {b: alea.randint(1, 8) / 8 for b in alea.sample(catalogue, 3) if b != a}
                   for a in catalogue if alea.random() < 0.8}
        panier = [alea.choice(catalogue) for _ in range(alea.randint(0, 3))]
        pop = alea.sample(catalogue, len(catalogue))
        k = alea.randint(0, 8)
        detenues = set(panier)
        score = {o: sum(voisins.get(h, {}).get(o, 0.0) for h in sorted(detenues)) for o in catalogue if o not in detenues}
        proches = sorted((o for o in score if score[o] > 0), key=lambda o: (-score[o], o))
        attendu = (proches + [o for o in pop if o not in detenues and o not in proches])[:k]
        assert _ref_recommander(panier, voisins, pop, k) == attendu == corrige.recommander(panier, voisins, pop, k)


# --------------------------------------------------------------------------- Défi 191 : decouper_dans_le_temps


def _decouper(defaut=None):
    """Écrit autrement que le corrigé (on trie une COPIE du journal par date), avec UN défaut."""

    def decouper_dans_le_temps(abonnements, coupure):
        lignes = abonnements if defaut in ("ordre", "modifie") else sorted(abonnements, key=lambda x: x[2])
        if defaut == "modifie":
            abonnements.sort(key=lambda x: x[2])
        historique, a_deviner = {}, {}
        for client, offre, date in lignes:
            if defaut != "froids_absents":
                historique.setdefault(client, set())
            if (date <= coupure) if defaut == "inclusif" else (date < coupure):
                historique.setdefault(client, set()).add(offre)
            else:
                if defaut == "historique_tout":
                    historique.setdefault(client, set()).add(offre)
                if defaut == "cible_deja" or offre not in historique.get(client, set()) or defaut == "historique_tout":
                    a_deviner.setdefault(client, set()).add(offre)
                elif defaut == "vides":
                    a_deviner.setdefault(client, set())
        if defaut == "listes":
            return ({c: sorted(o) for c, o in historique.items()}, {c: sorted(o) for c, o in a_deviner.items()})
        return historique, a_deviner

    return decouper_dans_le_temps


def test_191_le_corrige_et_d_autres_ecritures_justes_reussissent():
    _reussi("decouper_dans_le_temps", corrige.decouper_dans_le_temps)
    _reussi("decouper_dans_le_temps", _decouper())


@pytest.mark.parametrize("defaut,morceaux", [
    ("inclusif", ("AVENIR", "date < coupure")),
    ("cible_deja", ("réabonnement",)),
    ("froids_absents", ("froid",)),
    ("vides", ("pas d'ensemble vide",)),
    ("ordre", ("DÉSORDRE",)),
    ("historique_tout", ("PASSÉ",)),
    ("listes", ("ensemble (set)",)),
    ("modifie", ("a modifié le journal",)),
])
def test_191_chaque_erreur_classique_est_nommee(defaut, morceaux):
    _echec("decouper_dans_le_temps", _decouper(defaut), *morceaux)


def test_191_la_reference_egale_la_definition():
    abonnements = generer_abonnements(500, graine=7)
    coupure = dt.date(2025, 7, 1)
    historique, a_deviner = _ref_decouper(abonnements, coupure)
    assert (historique, a_deviner) == corrige.decouper_dans_le_temps(abonnements, coupure)
    for client in {c for c, _, _ in abonnements}:
        avant = {o for c, o, d in abonnements if c == client and d < coupure}
        apres = {o for c, o, d in abonnements if c == client and d >= coupure} - avant
        assert historique[client] == avant
        assert a_deviner.get(client, set()) == apres and (client in a_deviner) == bool(apres)


# --------------------------------------------------------------------------- Défi 192 : mesurer


def _mesurer(defaut=None):
    """Écrit autrement que le corrigé (listes de scores par client, logarithme naturel), avec UN défaut."""

    def mesurer(recommandations, a_deviner, k=3):
        if defaut != "pas_de_valeur" and (k < 1 or not a_deviner):
            raise ValueError
        clients = list(a_deviner)
        if defaut == "moyenne_servis":
            clients = [c for c in a_deviner if recommandations.get(c)]
        if defaut == "tous_clients":
            clients = list(a_deviner) + [c for c in recommandations if c not in a_deviner]
        precisions, rappels, ndcgs = [], [], []
        for client in clients:
            cible = a_deviner.get(client, set())
            liste = recommandations.get(client) or []
            montrees = liste if defaut == "sans_coupe" else liste[:k]
            touches, dcg, vues = 0, 0.0, set()
            for i, offre in enumerate(montrees):
                if offre in cible and (offre not in vues or defaut == "doublons"):
                    touches += 1
                    rang = i + 1
                    if defaut == "log_zero":
                        dcg += 1 / math.log(rang)
                    elif defaut == "rang_decale":
                        dcg += 1 / math.log(rang + 2)
                    else:
                        dcg += 1 / math.log(rang + 1)          # ln au lieu de log2 : le NDCG ne change pas
                vues.add(offre)
            if not cible:
                precisions.append(0.0), rappels.append(0.0), ndcgs.append(0.0)
                continue
            nb = k if defaut == "ideal_k" else min(k, len(cible))
            ideal = sum(1 / math.log(r + 1) for r in range(1, nb + 1))
            precisions.append(touches / len(montrees) if defaut == "precision_longueur" and montrees
                              else 0.0 if defaut == "precision_longueur" else touches / k)
            rappels.append(touches / k if defaut == "rappel_k" else
                           touches / min(k, len(cible)) if defaut == "rappel_min" else touches / len(cible))
            ndcgs.append(dcg / ideal)
        m = len(clients) or 1
        resultat = {"precision": sum(precisions) / m, "rappel": sum(rappels) / m, "ndcg": sum(ndcgs) / m}
        if defaut == "cles":
            resultat["f1"] = 0.0
        return resultat

    return mesurer


def test_192_le_corrige_et_d_autres_ecritures_justes_reussissent():
    _reussi("mesurer", corrige.mesurer)
    _reussi("mesurer", _mesurer())                     # logarithme naturel : accepté, le NDCG n'en dépend pas


@pytest.mark.parametrize("defaut,morceaux", [
    ("moyenne_servis", ("TOUS les clients à deviner", "embellit")),
    ("tous_clients", ("Seuls les clients de `a_deviner`",)),
    ("sans_coupe", ("k PREMIÈRES",)),
    ("doublons", ("ne touche qu'une fois",)),
    ("precision_longueur", ("divise par k",)),
    ("rappel_k", ("taille de la CIBLE", "ni par k")),
    ("rappel_min", ("ni par k, ni par min(k",)),
    ("ideal_k", ("min(k, taille de la cible)",)),
    ("rang_decale", ("commence à 1",)),
    ("log_zero", ("log2(r + 1)", "divise par zéro")),
    ("pas_de_valeur", ("au lieu de ValueError",)),
    ("cles", ("exactement les clés",)),
])
def test_192_chaque_erreur_classique_est_nommee(defaut, morceaux):
    _echec("mesurer", _mesurer(defaut), *morceaux)


def test_192_la_reference_redonne_les_valeurs_calculees_a_la_main():
    a_deviner = {"C01": {"tv"}, "C02": {"musique"}, "C03": {"gaming"}}
    recos = {"C01": ["sport", "tv", "box2"], "C02": ["gaming", "musique"], "C04": ["sport"]}
    attendu = {"precision": (1 / 3 + 1 / 3 + 0) / 3, "rappel": (1 + 1 + 0) / 3,
               "ndcg": (1 / math.log2(3) + 1 / math.log2(3) + 0) / 3}
    for resultat in (_ref_mesurer(recos, a_deviner, 3), corrige.mesurer(recos, a_deviner, 3)):
        assert all(math.isclose(resultat[m], attendu[m]) for m in attendu)
    # deux bonnes réponses sur trois à deviner, aux rangs 1 et 3, k = 3
    resultat = corrige.mesurer({"A": ["tv", "x", "box2"]}, {"A": {"tv", "box2", "cloud"}}, 3)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(resultat["ndcg"], (1 + 1 / math.log2(4)) / idcg)
    assert math.isclose(resultat["precision"], 2 / 3) and math.isclose(resultat["rappel"], 2 / 3)


def test_192_le_ndcg_vaut_1_seulement_pour_la_meilleure_liste():
    alea = random.Random(4)
    for _ in range(200):
        cible = set(alea.sample("abcdef", alea.randint(1, 4)))
        k = alea.randint(1, 5)
        meilleure = sorted(cible) + [o for o in "ghij"]
        assert math.isclose(corrige.mesurer({"A": meilleure}, {"A": cible}, k)["ndcg"], 1.0)
        autre = [alea.choice("abcdefghij") for _ in range(k)]
        assert corrige.mesurer({"A": autre}, {"A": cible}, k)["ndcg"] <= 1.0 + 1e-12


# --------------------------------------------------------------------------- Défi 193 : diversifier


def _diversifier(defaut=None):
    """Écrit autrement que le corrigé (max avec une clé, et `set` des restantes), avec UN défaut."""

    def diversifier(candidats, similarites, k=3, lam=0.7):
        if defaut != "lam" and not 0 <= lam <= 1:
            raise ValueError
        if k < 0:
            raise ValueError
        ordre = sorted(candidats) if defaut == "nom_seul" else sorted(candidats, key=lambda o: (-candidats[o], o))
        if defaut == "reverse":
            ordre = sorted(candidats, key=lambda o: (candidats[o], o), reverse=True)
        penalite = {o: 0.0 for o in ordre}
        choix = []
        restantes = list(ordre)
        while restantes and len(choix) < k:
            def valeur(o):
                if defaut == "naif":
                    p = max((similarites.get(o, {}).get(c, 0.0) for c in choix), default=0.0)
                elif defaut == "max_vide":
                    p = max(similarites.get(o, {}).get(c, 0.0) for c in choix)
                else:
                    p = penalite[o]
                if defaut == "lam_inverse":
                    return (1 - lam) * candidats[o] - lam * p
                if defaut == "note_entiere":
                    return candidats[o] - (1 - lam) * p
                return lam * candidats[o] - (1 - lam) * p

            meilleure = max(restantes, key=valeur)       # max garde le PREMIER des ex aequo
            choix.append(meilleure)
            restantes.remove(meilleure)
            if defaut == "modifie":
                del candidats[meilleure]
            for o in restantes:
                s = similarites.get(o, {}).get(meilleure, 0.0)
                if defaut == "somme":
                    penalite[o] += s
                elif defaut == "derniere":
                    penalite[o] = s
                elif defaut == "premiere":
                    if len(choix) == 1:
                        penalite[o] = s
                else:
                    penalite[o] = max(penalite[o], s)
        return choix

    return diversifier


def test_193_le_corrige_et_d_autres_ecritures_justes_reussissent():
    _reussi("diversifier", corrige.diversifier)
    _reussi("diversifier", _diversifier())


@pytest.mark.parametrize("defaut,morceaux", [
    ("somme", ("PLUS FORTE", "`max`")),
    ("derniere", ("TOUTES les options déjà choisies", "MAXIMUM")),
    ("premiere", ("pas seulement de la première",)),
    ("lam_inverse", ("pondère la NOTE",)),
    ("note_entiere", ("multipliée par `lam`",)),
    ("nom_seul", ("meilleure NOTE d'abord",)),
    ("reverse", ("ALPHABÉTIQUE", "reverse=True")),
    ("max_vide", ("ValueError", "liste vide")),
    ("lam", ("devrait lever ValueError", "NaN")),
    ("modifie", ("a modifié les candidats",)),
    ("naif", ("beaucoup trop lent", "SEULE option")),
])
def test_193_chaque_erreur_classique_est_nommee(defaut, morceaux):
    _echec("diversifier", _diversifier(defaut), *morceaux)


def test_193_la_reference_incrementale_egale_le_calcul_naif():
    """Recalculer le maximum sur TOUTES les options choisies à chaque tour donne le même résultat, en plus lent."""
    alea = random.Random(5)
    for _ in range(300):
        noms = alea.sample("abcdefgh", alea.randint(0, 8))
        cand = {o: alea.randint(0, 16) / 16 for o in noms}
        sims = {a: {b: alea.randint(0, 16) / 16 for b in noms if b != a and alea.random() < 0.6} for a in noms}
        k, lam = alea.randint(0, 8), alea.choice((0.0, 0.25, 0.5, 0.75, 1.0))
        attendu = _diversifier("naif")(cand, sims, k, lam)
        assert _ref_diversifier(cand, sims, k, lam) == attendu == corrige.diversifier(cand, sims, k, lam)


def test_193_lam_1_redonne_le_classement_par_note():
    alea = random.Random(6)
    for _ in range(100):
        cand = {o: alea.randint(0, 8) / 8 for o in alea.sample("abcdefg", 5)}
        sims = {a: {b: 1.0 for b in cand if b != a} for a in cand}
        assert corrige.diversifier(cand, sims, 5, 1.0) == sorted(cand, key=lambda o: (-cand[o], o))


# --------------------------------------------------------------------------- Les outils communs


def test_similarites_comptees_comptent_et_s_arretent():
    sim = SimilaritesComptees({"tv": {"box2": 0.75, "sport": 0.5}}, budget=3)
    ligne = sim["tv"]
    assert sim.lues == 0 and len(ligne) == 2 and "presse" not in sim
    assert ligne.get("box2") == 0.75 and ligne["sport"] == 0.5 and sim.lues == 2
    assert "cloud" not in ligne and sim.lues == 3
    with pytest.raises(LecturesEpuisees):
        ligne.get("box2")
    assert not issubclass(LecturesEpuisees, Exception)            # un `except Exception` ne la cache pas
    sim = SimilaritesComptees({"tv": {"box2": 0.75, "sport": 0.5}})
    assert dict(sim["tv"].items()) == {"box2": 0.75, "sport": 0.5} and sim.lues == 2
    with pytest.raises(TypeError):
        sim["tv"]["box2"] = 1.0                                     # lecture seule


def test_similarites_cosinus_suit_la_formule():
    paniers = generer_paniers(800, graine=3)
    sim = similarites_cosinus(paniers)
    cooc = corrige.cooccurrences(paniers)
    nombre = collections.Counter(o for p in paniers.values() for o in set(p or ()))
    assert set(sim) == set(cooc) == set(nombre)
    for a in sim:
        assert set(sim[a]) == set(cooc[a])
        for b, valeur in sim[a].items():
            assert math.isclose(valeur, cooc[a][b] / math.sqrt(nombre[a] * nombre[b])) and sim[b][a] == valeur


def test_populaires_compte_les_clients_pas_les_lignes():
    assert populaires({"A": ["tv", "tv", "tv"], "B": ["box2"], "C": ["box2"], "D": None}) == ["box2", "tv"]
    assert populaires(paniers_exemple())[0] == "mobile"


def test_les_generateurs_sont_reproductibles_et_realistes():
    assert generer_paniers(300, graine=4) == generer_paniers(300, graine=4) != generer_paniers(300, graine=5)
    paniers = generer_paniers(3000, graine=4)
    assert 0.6 < sum(1 for p in paniers.values() if p and "mobile" in p) / len(paniers) < 0.8
    assert sum(1 for p in paniers.values() if p is None) > 20                      # des cases vides
    assert sum(1 for p in paniers.values() if p and len(p) != len(set(p))) > 20    # des options répétées
    regles = {(r.si, r.alors): r for r in corrige.regles_association(paniers, 30)}
    assert regles["sport", "tv"].lift > 1.5 and regles["cloud", "antivirus"].lift > 1.5     # les profils se voient
    assert regles["tv", "mobile"].confiance > 0.5 and 0.8 < regles["tv", "mobile"].lift < 1.25  # mobile : partout
    abonnements = generer_abonnements(400, graine=4)
    assert abonnements == generer_abonnements(400, graine=4)
    assert len(abonnements) != len(set(abonnements))                              # des lignes en double
    assert abonnements != sorted(abonnements, key=lambda x: x[2])


def test_les_exemples_ont_les_pieges_annonces():
    paniers = paniers_exemple()
    assert paniers["C08"].count("tv") == 2 and paniers["C09"] == [] and paniers["C11"] is None
    assert corrige.cooccurrences(paniers)["presse"] == {}
    assert paniers_exemple() is not paniers_exemple() and paniers_exemple()["C01"] is not paniers_exemple()["C01"]
    abonnements = abonnements_exemple()
    assert len(abonnements) != len(set(abonnements))
    assert any(d == dt.date(2026, 6, 1) for _, _, d in abonnements)


# --------------------------------------------------------------------------- La série


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(188, 194))


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_31 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(offres)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_31 import *` (dans les exemples) doit donner les fonctions du corrigé
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

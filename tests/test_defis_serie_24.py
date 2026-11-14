"""Série 24 (orchestrer le traitement de nuit) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 146`). Ils vérifient que :

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
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.corrige_serie_24 as corrige  # noqa: E402
from defis.checks_serie_24 import SERIE  # noqa: E402
from defis.nuit import (  # noqa: E402
    durees_exemple,
    generer_durees,
    generer_pipeline,
    pipeline_exemple,
    taches_du_graphe,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}), DEFIS[nom])


def _echec(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


# --------------------------------------------------------------------------- outils des versions boguées


def _graphe(dependances, doublons=False, chaine=False, seulement_cles=False):
    if not chaine:
        for p in dependances.values():
            if isinstance(p, str):
                raise TypeError("une chaîne")
    graphe = {}
    for t, p in dependances.items():
        graphe.setdefault(t, [] if doublons else set())
        if doublons:
            graphe[t].extend(p)
        else:
            graphe[t].update(p)
        if not seulement_cles:
            for nom in p:
                graphe.setdefault(nom, [] if doublons else set())
    return graphe


def _successeurs(graphe):
    successeurs = {t: [] for t in graphe}
    for t, p in graphe.items():
        for nom in set(p):
            if nom in successeurs:
                successeurs[nom].append(t)
    return successeurs


# --------------------------------------------------------------------------- Défi 146 : ordre_execution


def _ordre(defaut=None):
    def ordre_execution(dependances):
        if defaut == "modifie":
            for p in list(dependances.values()):
                for nom in p:
                    dependances.setdefault(nom, [])
        graphe = _graphe(dependances, doublons=defaut == "doublons", chaine=defaut == "chaine",
                         seulement_cles=defaut == "seulement_cles")
        successeurs = _successeurs(graphe)
        restants = {t: len(p) for t, p in graphe.items()}
        if defaut == "tri_alpha":
            return sorted(graphe)
        if defaut == "vagues":
            faites, ordre = set(), []
            while True:
                vague = sorted(t for t in graphe if t not in faites and set(graphe[t]) <= faites)
                if not vague:
                    break
                ordre += vague
                faites.update(vague)
        elif defaut == "dfs":
            ordre, vus = [], set()

            def visiter(t):
                if t not in vus:
                    vus.add(t)
                    for p in sorted(graphe[t], reverse=True):
                        visiter(p)
                    ordre.append(t)

            for t in sorted(graphe, reverse=True):
                visiter(t)
        elif defaut == "boucle_infinie":
            ordre, faites = [], set()
            while len(ordre) < len(graphe):
                for t in sorted(graphe):
                    if t not in faites and set(graphe[t]) <= faites:
                        ordre.append(t)
                        faites.add(t)
                        break
        elif defaut == "scan":
            ordre, faites, tri = [], set(), sorted(graphe)
            while True:
                suivante = None
                for t in tri:
                    if t not in faites and all(p in faites for p in graphe[t]):
                        suivante = t
                        break
                if suivante is None:
                    break
                ordre.append(suivante)
                faites.add(suivante)
        else:
            prets = [t for t, n in restants.items() if n == 0]
            if defaut != "min_chaque_tour":
                heapq.heapify(prets)
            ordre = []
            while prets:
                if defaut == "min_chaque_tour":
                    t = min(prets)
                    prets.remove(t)
                else:
                    t = heapq.heappop(prets)
                ordre.append(t)
                for s in successeurs[t]:
                    restants[s] -= 1
                    if restants[s] == 0:
                        if defaut == "min_chaque_tour":
                            prets.append(s)
                        else:
                            heapq.heappush(prets, s)
        if len(ordre) < len(graphe) and defaut != "cycle_partiel":
            if defaut == "sans_mot":
                raise ValueError("graphe invalide")
            if defaut == "sans_taches":
                raise ValueError("cycle détecté")
            raise ValueError(f"cycle : {', '.join(sorted(set(graphe) - set(ordre)))}")
        if defaut == "envers":
            return ordre[::-1]
        if defaut == "tuple":
            return tuple(ordre)
        return ordre

    return ordre_execution


@pytest.mark.parametrize("mutant,mots", [
    (_ordre("vagues"), ("vagues", "alerte_support", "extraire_clients")),
    (_ordre("dfs"), ("ordre possible", "première par ordre alphabétique")),
    (_ordre("seulement_cles"), ("PRÉREQUIS",)),
    (_ordre("doublons"), ("ne compte qu'une fois",)),
    (_ordre("chaine"), ("TypeError", "LETTRE PAR LETTRE")),
    (_ordre("cycle_partiel"), ("devrait lever ValueError",)),
    (_ordre("boucle_infinie"), ("ne s'arrête pas", "cycle")),
    (_ordre("sans_mot"), ("ne contient pas", "'cycle'")),
    (_ordre("sans_taches"), ("ne nomme aucune tâche",)),
    (_ordre("modifie"), ("MODIFIÉ",)),
    (_ordre("min_chaque_tour"), ("O(n²)", "heapq")),
    (_ordre("scan"), ("O(n²)", "compteur")),
    (_ordre("tri_alpha"), ("avant son prérequis",)),
    (_ordre("envers"), ("à l'envers",)),
    (_ordre("tuple"), ("au lieu d'une liste",)),
])
def test_ordre_erreurs_classiques(mutant, mots):
    _echec("ordre_execution", mutant, *mots)


# --------------------------------------------------------------------------- Défi 147 : taches_a_relancer


def _relancer(defaut=None):
    ordre_execution = corrige.ordre_execution

    def taches_a_relancer(dependances, modifiees):
        if isinstance(modifiees, str) and defaut != "chaine":
            raise TypeError("une chaîne")
        ordre = ordre_execution(dependances)
        graphe = _graphe(dependances)
        successeurs = _successeurs(graphe)
        if defaut == "generateur":
            for nom in modifiees:
                if nom not in graphe:
                    raise ValueError(nom)
            depart = list(modifiees)
        else:
            depart = list(modifiees)
            inconnues = [n for n in depart if n not in graphe]
            if inconnues and defaut != "inconnue_ignoree":
                raise ValueError(f"inconnue : {inconnues}")
            depart = [n for n in depart if n in graphe]
        if defaut == "ancetres":
            voisins = {t: list(p) for t, p in graphe.items()}
        else:
            voisins = successeurs
        if defaut == "enfants":
            a_relancer = set(depart) | {s for t in depart for s in successeurs[t]}
        elif defaut == "sans_vus":
            a_relancer, pile = set(), list(depart)
            while pile:
                t = pile.pop()
                a_relancer.add(t)
                pile.extend(voisins[t])
        else:
            a_relancer, file = set(depart), deque(depart)
            while file:
                for s in voisins[file.popleft()]:
                    if s not in a_relancer:
                        a_relancer.add(s)
                        file.append(s)
        if defaut == "sans_modifiees":
            a_relancer -= set(depart)
        if defaut == "sous_graphe":
            return ordre_execution({t: [p for p in graphe[t] if p in a_relancer] for t in a_relancer})
        if defaut == "tri":
            return sorted(a_relancer)
        return [t for t in ordre if t in a_relancer]

    return taches_a_relancer


@pytest.mark.parametrize("mutant,mots", [
    (_relancer("ancetres"), ("PRÉREQUIS", "ATTENDENT")),
    (_relancer("enfants"), ("DIRECTEMENT",)),
    (_relancer("sans_modifiees"), ("tâches modifiées elles-mêmes",)),
    (_relancer("sous_graphe"), ("PETIT", "graphe ENTIER")),
    (_relancer("tri"), ("triées par nom",)),
    (_relancer("generateur"), ("GÉNÉRATEUR",)),
    (_relancer("chaine"), ("TypeError", "lettre par lettre")),
    (_relancer("inconnue_ignoree"), ("ValueError", "ignorée en silence")),
    (_relancer("sans_vus"), ("ne s'arrête pas", "2^30", "déjà vu")),
])
def test_relancer_erreurs_classiques(mutant, mots):
    _echec("taches_a_relancer", mutant, *mots)


# --------------------------------------------------------------------------- Défi 148 : trouver_cycle


def _cycle(defaut=None):
    def trouver_cycle(dependances):
        graphe = {t: sorted(p) for t, p in _graphe(dependances).items()}
        if defaut in ("recursif", "limite"):
            if defaut == "limite":
                sys.setrecursionlimit(20_000)
            etat = {}

            def visiter(t, chemin):
                etat[t] = 1
                chemin.append(t)
                for p in graphe[t]:
                    if etat.get(p) == 1:
                        return chemin[chemin.index(p):] + [p]
                    if p not in etat:
                        trouve = visiter(p, chemin)
                        if trouve:
                            return trouve
                chemin.pop()
                etat[t] = 2
                return None

            for t in sorted(graphe):
                if t not in etat:
                    trouve = visiter(t, [])
                    if trouve:
                        return trouve
            return None
        etat = {}
        for depart in sorted(graphe):
            if depart in etat:
                continue
            chemin, pile = [depart], [iter(graphe[depart])]
            etat[depart] = 1
            while pile:
                suivante = next(pile[-1], None)
                if suivante is None:
                    fini = chemin.pop()
                    if defaut == "sans_terminees":
                        del etat[fini]
                    else:
                        etat[fini] = 2
                    pile.pop()
                elif etat.get(suivante) == 1 or (defaut == "deux_etats" and suivante in etat):
                    if defaut == "chemin_complet":
                        cycle = chemin + [suivante]
                    else:
                        debut = chemin.index(suivante) if suivante in chemin else len(chemin) - 1
                        cycle = chemin[debut:] + [suivante]  # « deux_etats » : un faux cycle sur une tâche déjà vue
                    if defaut == "sens":
                        cycle = cycle[::-1]
                    if defaut == "sans_repetition":
                        cycle = cycle[:-1]
                    return cycle
                elif suivante not in etat:
                    etat[suivante] = 1
                    chemin.append(suivante)
                    pile.append(iter(graphe[suivante]))
            if defaut == "sans_terminees":
                etat[depart] = 2
        return None

    return trouver_cycle


@pytest.mark.parametrize("mutant,mots", [
    (_cycle("deux_etats"), ("« déjà vue » ne veut pas dire « cycle »", "EN COURS")),
    (_cycle("chemin_complet"), ("MÈNE",)),
    (_cycle("sens"), ("mauvais sens",)),
    (_cycle("sans_repetition"), ("RÉPÉTER",)),
    (_cycle("recursif"), ("RecursionError", "pile explicite")),
    (_cycle("limite"), ("setrecursionlimit", "interdit")),
    (_cycle("sans_terminees"), ("ne s'arrête pas", "TERMINÉES")),
])
def test_cycle_erreurs_classiques(mutant, mots):
    limite = sys.getrecursionlimit()
    try:
        _echec("trouver_cycle", mutant, *mots)
    finally:
        assert sys.getrecursionlimit() == limite  # le correcteur a remis la limite en place


# --------------------------------------------------------------------------- Défi 149 : executer


def _executer(defaut=None):
    ordre_execution = corrige.ordre_execution

    def executer(taches, dependances):
        graphe = _graphe(dependances)
        if defaut != "valide_tard":
            manquantes = set(graphe) - set(taches)
            en_trop = set(taches) - set(graphe)
            if manquantes or (en_trop and defaut != "en_trop_accepte"):
                raise ValueError("taches")
        ordre = ordre_execution(dependances)
        statuts, resultats, erreurs = {}, {}, {}
        for t in ordre:
            if defaut == "valide_tard" and t not in taches:
                raise ValueError(t)
            if defaut == "enfants_seuls":
                bloque = any(statuts[p] == "echec" for p in graphe[t])
            elif defaut == "none_pas_ok":
                bloque = any(statuts[p] != "ok" or resultats[p] is None for p in graphe[t])
            else:
                bloque = any(statuts[p] != "ok" for p in graphe[t])
            if bloque:
                statuts[t] = "sautee"
                continue
            try:
                if defaut == "positionnel":
                    valeur = taches[t](*[resultats[p] for p in sorted(graphe[t])])
                else:
                    valeur = taches[t](**{p: resultats[p] for p in graphe[t]})
            except BaseException as exc:
                if defaut != "base_exception" and not isinstance(exc, Exception):
                    raise
                statuts[t] = "echec"
                erreurs[t] = str(exc) if defaut == "erreur_texte" else exc
                if defaut == "stop":
                    break
            else:
                statuts[t] = "ok"
                resultats[t] = valeur
        if defaut == "stop":
            for t in ordre:
                statuts.setdefault(t, "sautee")
        if defaut == "ordre_statuts":
            statuts = dict(sorted(statuts.items()))
        if defaut == "deux_retours":
            return statuts, resultats
        return statuts, resultats, erreurs

    return executer


@pytest.mark.parametrize("mutant,mots", [
    (_executer("positionnel"), ("POSITIONNELS", "NOMMÉS")),
    (_executer("stop"), ("ne dépendent PAS", "break")),
    (_executer("enfants_seuls"), ("SAUTÉ", "petits-enfants")),
    (_executer("base_exception"), ("Ctrl+C a été avalé", "`Exception`")),
    (_executer("valide_tard"), ("APRÈS avoir lancé",)),
    (_executer("en_trop_accepte"), ("devrait lever ValueError", "faute de")),
    (_executer("none_pas_ok"), ("renvoyé None", "STATUT")),
    (_executer("erreur_texte"), ("l'exception levée",)),
    (_executer("ordre_statuts"), ("pas dans l'ordre d'exécution",)),
    (_executer("deux_retours"), ("trois dictionnaires",)),
])
def test_executer_erreurs_classiques(mutant, mots):
    _echec("executer", mutant, *mots)


# --------------------------------------------------------------------------- Défi 150 : chemin_critique


def _critique(defaut=None):
    ordre_execution = corrige.ordre_execution

    def chemin_critique(durees, dependances):
        graphe = _graphe(dependances)
        if set(durees) != set(graphe) and not (defaut == "en_trop" and set(graphe) <= set(durees)):
            raise ValueError("durees")
        for t, d in durees.items():
            if (isinstance(d, bool) and defaut != "bool") or not isinstance(d, numbers.Real):
                raise TypeError(t)
            if (d < 0 if defaut == "nan" else not (math.isfinite(d) and d >= 0)):
                raise ValueError(t)
        if defaut in ("recursif", "memo"):
            ordre_execution(dependances)  # refuse les cycles d'abord, comme le demande l'énoncé
            memo = {}

            def fin(t):
                if defaut == "memo" and t in memo:
                    return memo[t]
                valeur = max((fin(p) for p in graphe[t]), default=0) + durees[t]
                memo[t] = valeur
                return valeur

            for t in sorted(graphe):
                fin(t)
            fins = memo
        ordre = ordre_execution(dependances)
        if not ordre:
            return 0, []
        fins, precedente = {}, {}
        for t in ordre:
            if defaut == "egalite":
                dernier = max(graphe[t], key=lambda p: (fins[p], p), default=None)
            elif defaut == "min":
                dernier = min(graphe[t], key=lambda p: (fins[p], p), default=None)
            else:
                dernier = min(graphe[t], key=lambda p: (-fins[p], p), default=None)
            precedente[t] = dernier
            fins[t] = (fins[dernier] if dernier is not None else 0) + durees[t]
        if defaut == "fin_egalite":
            t = max(fins, key=lambda x: (fins[x], x))
        else:
            t = min(fins, key=lambda x: (-fins[x], x))
        total, chemin = fins[t], []
        while t is not None:
            chemin.append(t)
            t = precedente[t]
        if defaut == "somme":
            total = sum(durees[x] for x in graphe)
        if defaut == "envers":
            return total, chemin
        return total, chemin[::-1]

    return chemin_critique


@pytest.mark.parametrize("mutant,mots", [
    (_critique("somme"), ("SOMME",)),
    (_critique("min"), ("MAXIMUM",)),
    (_critique("envers"), ("à l'envers", "chemin[::-1]")),
    (_critique("egalite"), ("à égalité", "première par ordre alphabétique")),
    (_critique("fin_egalite"), ("se termine par", "à égalité")),
    (_critique("recursif"), ("ne s'arrête pas", "UNE fois")),
    (_critique("memo"), ("RecursionError",)),
    (_critique("nan"), ("NaN",)),
    (_critique("bool"), ("True est un entier",)),
    (_critique("en_trop"), ("faute de frappe",)),
])
def test_critique_erreurs_classiques(mutant, mots):
    _echec("chemin_critique", mutant, *mots)


# --------------------------------------------------------------------------- Défi 151 : planifier


def _planifier(defaut=None):
    def planifier(durees, dependances, travailleurs=2):
        if (isinstance(travailleurs, bool) and defaut != "bool") or not isinstance(travailleurs, numbers.Integral):
            raise TypeError("travailleurs")
        if travailleurs < 1:
            raise ValueError("travailleurs")
        corrige.chemin_critique(durees, dependances)  # valide les durées et le graphe
        graphe = _graphe(dependances)
        successeurs = _successeurs(graphe)
        reste = corrige._reste(durees, graphe, corrige.ordre_execution(dependances))

        def cle(t):
            return (t,) if defaut == "alpha" else (-reste[t], t)

        restants = {t: len(p) for t, p in graphe.items()}
        prets = [(cle(t), t) for t, n in restants.items() if n == 0]
        heapq.heapify(prets)
        premier = 0 if defaut == "zero" else 1
        libres = list(range(premier, premier + int(travailleurs)))
        en_cours, planning, temps = [], {}, 0
        while prets or en_cours:
            while prets and libres:
                _, t = heapq.heappop(prets)
                w = heapq.heappop(libres)
                planning[t] = (temps, temps + durees[t], w)
                heapq.heappush(en_cours, (temps + durees[t], w, t))
            if defaut == "pas_a_pas":
                if en_cours[0][0] > temps:
                    temps += 1
                    continue
            else:
                temps = en_cours[0][0]
            while en_cours and en_cours[0][0] == temps:
                _, w, t = heapq.heappop(en_cours)
                heapq.heappush(libres, w)
                for s in successeurs[t]:
                    restants[s] -= 1
                    if restants[s] == 0:
                        heapq.heappush(prets, (cle(s), s))
                if defaut == "une_fin":
                    break
        if defaut == "ordre_dict":
            return dict(sorted(planning.items()))
        return planning

    return planifier


@pytest.mark.parametrize("mutant,mots", [
    (_planifier("alpha"), ("ALPHABÉTIQUE", "RESTE")),
    (_planifier("zero"), ("numérotés de 0",)),
    (_planifier("une_fin"), ("MÊME instant", "TOUTES")),
    (_planifier("pas_a_pas"), ("d'une unité",)),
    (_planifier("bool"), ("True",)),
    (_planifier("ordre_dict"), ("ordre des débuts",)),
])
def test_planifier_erreurs_classiques(mutant, mots):
    _echec("planifier", mutant, *mots)


# --------------------------------------------------------------------------- garde-fous


@pytest.mark.parametrize("nom,fabrique", [("ordre_execution", _ordre), ("taches_a_relancer", _relancer),
                                          ("trouver_cycle", _cycle), ("executer", _executer),
                                          ("chemin_critique", _critique), ("planifier", _planifier)])
def test_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Chaque version boguée ci-dessus ne diffère de la juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_le_compteur_rend_le_traceur_precedent():
    """Le correcteur remet en place le traceur d'avant (débogueur, couverture de code)."""
    def traceur(frame, evenement, arg):
        return None

    precedent = sys.gettrace()
    sys.settrace(traceur)
    try:
        _evaluer("ordre_execution", _ordre())
        _evaluer("ordre_execution", _ordre("boucle_infinie"))
        assert sys.gettrace() is traceur
    finally:
        sys.settrace(precedent)


def test_les_tests_du_corrige_passent():
    tests = [f for nom, f in list(vars(corrige).items()) if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6


def test_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.nuit as nuit
    import defis.serie_24 as eleve

    for nom in ("pipeline_exemple", "durees_exemple", "taches_du_graphe", "generer_pipeline", "generer_durees",
                "afficher_planning"):
        assert getattr(eleve, nom) is getattr(nuit, nom), nom


def test_les_exemples_sont_neufs_a_chaque_appel():
    p = pipeline_exemple()
    p["nettoyer"].append("x")
    p.clear()
    assert pipeline_exemple()["nettoyer"] == ["extraire_clients", "extraire_factures"]
    d = durees_exemple()
    d.clear()
    assert len(durees_exemple()) == 12


def test_l_exemple_a_tous_les_pieges_annonces():
    p = pipeline_exemple()
    assert "extraire_appels" not in p and "extraire_appels" in taches_du_graphe(p)
    assert set(durees_exemple()) == taches_du_graphe(p)
    ordre = corrige.ordre_execution(p)
    assert ordre.index("alerte_support") < ordre.index("extraire_clients")  # pas par vagues
    total, _ = corrige.chemin_critique(durees_exemple(), p)
    assert total == 150 and sum(durees_exemple().values()) == 200
    # evaluer et scorer finissent ensemble : l'égalité est départagée par l'alphabet
    plan = corrige.planifier(durees_exemple(), p, 20)
    assert plan["evaluer"][1] == plan["scorer"][1] == 135
    # par ordre alphabétique au lieu du reste, la nuit à 2 travailleurs durerait 160 minutes
    from defis.checks_serie_24 import _ref_planifier

    alpha = _ref_planifier(durees_exemple(), p, 2, priorite="alpha")
    assert max(fin for _, fin, _ in alpha.values()) == 160


def test_le_generateur_est_reproductible_et_sans_cycle():
    for n, graine in [(0, 1), (1, 1), (50, 2), (5000, 3)]:
        g = generer_pipeline(n, graine=graine)
        assert g == generer_pipeline(n, graine=graine)
        assert len(taches_du_graphe(g)) == n
        assert len(corrige.ordre_execution(g)) == n  # pas de cycle
    g = generer_pipeline(2000, graine=4)
    assert any(t not in g for t in taches_du_graphe(g))  # des tâches citées seulement comme prérequis
    assert any(len(set(p)) >= 2 for p in g.values())
    ordre = corrige.ordre_execution(g)
    assert ordre != sorted(ordre)  # l'alphabet n'a rien à voir avec les dépendances
    d = generer_durees(g, graine=4)
    assert set(d) == taches_du_graphe(g) and 0 in d.values() and all(isinstance(v, int) for v in d.values())


def test_les_exemples_des_docstrings_sont_justes():
    """Les exemples de defis/serie_24.py et defis/nuit.py, rejoués avec le corrigé."""
    import defis.nuit as nuit
    import defis.serie_24 as eleve

    tests = doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(nuit)
    globs = dict(vars(eleve))
    for nom in DEFIS:
        globs[nom] = getattr(corrige, nom)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs = dict(globs)
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 40 and resultat.failed == 0


def test_le_corrige_n_utilise_pas_graphlib():
    texte = Path(corrige.__file__).read_text(encoding="utf-8")
    assert "import graphlib" not in texte and "TopologicalSorter(" not in texte


# --------------------------------------------------------------------------- d'autres écritures justes


def test_autres_ecritures_justes_acceptees():
    def ordre_insort(dependances):                # une liste toujours triée (bisect.insort) au lieu d'un tas
        graphe = _graphe(dependances)
        successeurs = _successeurs(graphe)
        restants = {t: len(p) for t, p in graphe.items()}
        prets = sorted(t for t, n in restants.items() if n == 0)
        ordre = []
        while prets:
            t = prets.pop(0)                          # la plus petite est au début
            ordre.append(t)
            for s in successeurs[t]:
                restants[s] -= 1
                if restants[s] == 0:
                    bisect.insort(prets, s)           # O(log n) comparaisons pour trouver sa place
        if len(ordre) != len(graphe):
            raise ValueError("cycle entre " + ", ".join(sorted(set(graphe) - set(ordre))))
        return ordre

    def cycle_par_kahn(dependances):              # on retire tout ce qui n'est pas bloqué, puis on suit les bloquées
        graphe = _graphe(dependances)
        successeurs = _successeurs(graphe)
        restants = {t: len(p) for t, p in graphe.items()}
        file = deque(t for t, n in restants.items() if n == 0)
        while file:
            for s in successeurs[file.popleft()]:
                restants[s] -= 1
                if restants[s] == 0:
                    file.append(s)
        bloquees = {t for t, n in restants.items() if n > 0}
        if not bloquees:
            return None
        # chaque bloquée a un prérequis bloqué : on le suit jusqu'à repasser par une tâche déjà vue
        t, position, chemin = min(bloquees), {}, []
        while t not in position:
            position[t] = len(chemin)
            chemin.append(t)
            t = min(p for p in graphe[t] if p in bloquees)
        return chemin[position[t]:] + [t]

    def executer_par_descendance(taches, dependances):   # on calcule à part ce qu'un échec bloque
        graphe = _graphe(dependances)
        if set(taches) != set(graphe):
            raise ValueError("une fonction par tâche, ni plus ni moins")
        ordre = corrige.ordre_execution(dependances)
        bloquees, statuts, resultats, erreurs = set(), {}, {}, {}
        for t in ordre:
            if t in bloquees:
                statuts[t] = "sautee"
                continue
            try:
                resultats[t] = taches[t](**{p: resultats[p] for p in graphe[t]})
                statuts[t] = "ok"
            except Exception as exc:
                statuts[t] = "echec"
                erreurs[t] = exc
                bloquees |= set(corrige.taches_a_relancer(dependances, [t])) - {t}
        return statuts, resultats, erreurs

    def critique_avec_debuts(durees, dependances):   # début et fin rangés à part, chemin remonté par recherche
        graphe = _graphe(dependances)
        if set(durees) != set(graphe):
            raise ValueError("durées")
        for t, d in durees.items():
            if isinstance(d, bool) or not isinstance(d, (int, float)):
                raise TypeError(t)
            if math.isnan(d) or math.isinf(d) or d < 0:
                raise ValueError(t)
        debut, fin = {}, {}
        for t in corrige.ordre_execution(dependances):
            debut[t] = max((fin[p] for p in graphe[t]), default=0)
            fin[t] = debut[t] + durees[t]
        if not fin:
            return 0, []
        total = max(fin.values())
        t = sorted(x for x in fin if fin[x] == total)[0]
        chemin = [t]
        while graphe[t]:
            t = sorted(p for p in graphe[t] if fin[p] == debut[t])[0]
            chemin.append(t)
        return total, list(reversed(chemin))

    def planifier_par_instants(durees, dependances, travailleurs=2):  # un tas des INSTANTS, et des listes triées
        if type(travailleurs) is not int:
            raise TypeError("entier")
        if travailleurs < 1:
            raise ValueError("au moins 1")
        corrige.chemin_critique(durees, dependances)
        graphe = _graphe(dependances)
        successeurs = _successeurs(graphe)
        reste = corrige._reste(durees, graphe, corrige.ordre_execution(dependances))
        restants = {t: len(p) for t, p in graphe.items()}
        prets = [(-reste[t], t) for t, n in restants.items() if n == 0]
        heapq.heapify(prets)
        libres = list(range(1, travailleurs + 1))
        finissent: dict = {}
        instants: list = []
        planning, temps = {}, 0
        while True:
            for t, w in finissent.pop(temps, []):
                bisect.insort(libres, w)
                for s in successeurs[t]:
                    restants[s] -= 1
                    if restants[s] == 0:
                        heapq.heappush(prets, (-reste[s], s))
            while prets and libres:
                _, t = heapq.heappop(prets)
                w = libres.pop(0)
                planning[t] = (temps, temps + durees[t], w)
                if temps + durees[t] not in finissent:
                    heapq.heappush(instants, temps + durees[t])
                finissent.setdefault(temps + durees[t], []).append((t, w))
            if not instants:
                return planning
            temps = heapq.heappop(instants)

    for nom, objet in [("ordre_execution", ordre_insort), ("trouver_cycle", cycle_par_kahn),
                       ("executer", executer_par_descendance), ("chemin_critique", critique_avec_debuts),
                       ("planifier", planifier_par_instants)]:
        resultat = _evaluer(nom, objet)
        assert resultat.statut == REUSSI, (nom, resultat.message)


def test_relancer_par_balayage_dans_l_ordre_est_accepte():
    """Sans parcours de graphe : dans l'ordre d'exécution, une tâche est relancée si un de ses prérequis l'est."""

    def taches_a_relancer(dependances, modifiees):
        if isinstance(modifiees, (str, bytes)):
            raise TypeError("une liste de noms")
        depart = set(modifiees)
        graphe = {t: set(p) for t, p in corrige._graphe(dependances).items()}
        inconnues = depart - set(graphe)
        if inconnues:
            raise ValueError(f"tâches inconnues : {sorted(inconnues)}")
        relancees: list[str] = []
        marquees: set[str] = set()
        for t in corrige.ordre_execution(dependances):
            if t in depart or graphe[t] & marquees:
                marquees.add(t)
                relancees.append(t)
        return relancees

    resultat = _evaluer("taches_a_relancer", taches_a_relancer)
    assert resultat.statut == REUSSI, resultat.message

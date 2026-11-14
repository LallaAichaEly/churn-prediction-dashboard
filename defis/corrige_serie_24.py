"""Série 24 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_24.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md ce qui vous a coûté le plus de temps : c'est presque toujours un
doublon compté deux fois, une flèche prise dans le mauvais sens, ou un « déjà vu » pris pour un cycle.

Fil rouge de la série : un traitement de nuit est un GRAPHE ORIENTÉ SANS CYCLE (un « DAG », comme dans
Airflow, Dagster, `make` ou `dvc`). Cinq habitudes d'ingénieur·e :

1. on VALIDE le graphe une fois, au début, avec une fonction utilitaire (`_graphe`) que tous les défis
   réutilisent : une chaîne n'est pas une liste, un doublon ne compte qu'une fois, une tâche citée
   seulement comme prérequis existe quand même ;
2. on retourne les flèches une fois pour toutes (`_successeurs`) : on a souvent besoin de savoir « qui
   attend cette tâche ? », et relire tout le graphe à chaque fois coûte O(n²) ;
3. on traite les tâches dans l'ordre topologique (algorithme de Kahn : un compteur de prérequis
   restants par tâche, et un TAS pour toujours prendre la première par ordre alphabétique) ;
4. on garde une trace de ce qui est DÉJÀ fait (un `set`) : sans elle, un graphe « en losanges » se
   parcourt en un temps exponentiel ;
5. on n'utilise pas la récursion quand la profondeur dépend des données : Python s'arrête à 1 000
   appels imbriqués (RecursionError), et un traitement de nuit peut avoir une chaîne de 5 000 tâches.

La bibliothèque standard a un `graphlib.TopologicalSorter` (Python 3.9+) : en production, servez-vous
en. Ici, on l'écrit à la main pour comprendre ce qu'il fait, et parce que c'est une question
d'entretien classique (« course schedule », « build order »).

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import heapq
import math
import numbers
from collections import deque

from defis.nuit import (  # noqa: F401
    afficher_planning,
    durees_exemple,
    generer_durees,
    generer_pipeline,
    pipeline_exemple,
    taches_du_graphe,
)

# ============================================================================ Outils communs


def _graphe(dependances: dict) -> dict[str, set[str]]:
    """{tâche: ensemble de ses prérequis}, pour TOUTES les tâches (clés et prérequis), sans doublon.

    Vérifie les entrées une fois pour toutes. Ne modifie pas `dependances`.
    """
    if not isinstance(dependances, dict):
        raise TypeError(f"dependances doit être un dictionnaire, pas {type(dependances).__name__}.")
    graphe: dict[str, set[str]] = {}
    for tache, prerequis in dependances.items():
        # Le piège n°1 : "nettoyer" au lieu de ["nettoyer"]. Python parcourt une chaîne LETTRE PAR LETTRE,
        # et la tâche aurait pour prérequis "n", "e", "t", "o", "y", "r" : on refuse franchement.
        if isinstance(prerequis, (str, bytes)) or not isinstance(prerequis, (list, tuple, set, frozenset)):
            raise TypeError(f"Les prérequis de {tache!r} doivent être une liste de noms, pas {prerequis!r}.")
        for nom in (tache, *prerequis):
            if not isinstance(nom, str):
                raise TypeError(f"Un nom de tâche est une chaîne, pas {nom!r}.")
        graphe.setdefault(tache, set()).update(prerequis)  # un set : un doublon ne compte qu'une fois
        for nom in prerequis:
            graphe.setdefault(nom, set())  # une tâche citée seulement comme prérequis existe quand même
    return graphe


def _successeurs(graphe: dict[str, set[str]]) -> dict[str, list[str]]:
    """Les flèches retournées : {tâche: les tâches qui l'attendent}. Une seule lecture du graphe : O(n + m)."""
    successeurs: dict[str, list[str]] = {tache: [] for tache in graphe}
    for tache, prerequis in graphe.items():
        for nom in prerequis:
            successeurs[nom].append(tache)
    return successeurs


def _verifier_cles(cles, graphe: dict, quoi: str) -> None:
    """Exactement une entrée par tâche du graphe : ni oubli, ni faute de frappe."""
    manquantes, en_trop = sorted(set(graphe) - set(cles)), sorted(set(cles) - set(graphe))
    if manquantes or en_trop:
        raise ValueError(f"{quoi} ne correspond pas au graphe : manquantes {manquantes}, inconnues {en_trop}.")


# ============================================================================ Défi 146 : l'ordre d'exécution


def ordre_execution(dependances: dict) -> list[str]:
    graphe = _graphe(dependances)
    successeurs = _successeurs(graphe)
    # Algorithme de Kahn : on compte, pour chaque tâche, ses prérequis PAS ENCORE faits. Une tâche
    # devient prête quand son compteur tombe à 0. Le compteur vient de `graphe` (des ensembles) : un
    # prérequis écrit deux fois ne compte qu'une fois (sinon le compteur ne tomberait jamais à 0).
    restants = {tache: len(prerequis) for tache, prerequis in graphe.items()}
    prets = [tache for tache, n in restants.items() if n == 0]
    heapq.heapify(prets)  # un TAS : la plus petite (alphabétiquement) sort en O(log n), pas en O(n)
    ordre = []
    while prets:
        tache = heapq.heappop(prets)
        ordre.append(tache)
        for suivante in successeurs[tache]:
            restants[suivante] -= 1
            if restants[suivante] == 0:
                heapq.heappush(prets, suivante)  # elle passe AVANT les autres si son nom est plus petit
    if len(ordre) < len(graphe):
        # Il reste des tâches dont le compteur n'est jamais tombé à 0 : elles attendent, directement ou
        # non, une tâche de leur propre cycle. On les nomme : c'est la première chose qu'on cherchera.
        bloquees = sorted(tache for tache, n in restants.items() if n > 0)
        raise ValueError(f"Le graphe contient un cycle ; tâches bloquées : {', '.join(bloquees)}.")
    return ordre
    # Complexité : O((n + m) log n) pour n tâches et m flèches. La version « je relis toutes les tâches
    # pour trouver la prochaine prête » est en O(n²) : 25 millions de tours pour 5 000 tâches.


# ============================================================================ Défi 147 : ce qu'il faut relancer


def taches_a_relancer(dependances: dict, modifiees) -> list[str]:
    if isinstance(modifiees, (str, bytes)):
        raise TypeError(f"modifiees doit être une liste de noms, pas la chaîne {modifiees!r}.")
    ordre = ordre_execution(dependances)  # valide le graphe, et lève ValueError s'il y a un cycle
    successeurs = _successeurs(_graphe(dependances))
    depart = list(modifiees)  # UNE seule lecture : `modifiees` peut être un générateur
    inconnues = sorted({nom for nom in depart if nom not in successeurs})
    if inconnues:
        raise ValueError(f"Tâches inconnues : {', '.join(inconnues)}.")
    # Parcours en largeur dans le sens des flèches retournées (vers ce qui ATTEND la tâche). `a_relancer`
    # sert aussi de « déjà vu » : sans lui, chaque chemin serait reparcouru, et il y en a 2^k dans un
    # graphe à k étages en losange.
    a_relancer = set(depart)
    file = deque(a_relancer)
    while file:
        for suivante in successeurs[file.popleft()]:
            if suivante not in a_relancer:
                a_relancer.add(suivante)
                file.append(suivante)
    # L'ordre : celui dans lequel le traitement de nuit les lancera vraiment, c'est-à-dire l'ordre
    # d'exécution du graphe ENTIER, filtré. (L'ordre du petit graphe des seules tâches à relancer peut
    # être différent : une tâche relancée peut attendre un prérequis qui, lui, n'est pas relancé.)
    return [tache for tache in ordre if tache in a_relancer]


# ============================================================================ Défi 148 : trouver un cycle


def trouver_cycle(dependances: dict) -> list[str] | None:
    graphe = {tache: sorted(prerequis) for tache, prerequis in _graphe(dependances).items()}
    # Parcours en profondeur avec TROIS états, et SANS récursion :
    # - pas dans `etat` : jamais visitée ;
    # - EN_COURS : sur le chemin actuel (dans la pile). La revoir, c'est boucler : voilà le cycle ;
    # - TERMINEE : entièrement explorée, sans cycle en dessous. La revoir est normal (un losange) :
    #   « déjà vue » ne veut PAS dire « cycle ».
    EN_COURS, TERMINEE = 1, 2
    etat: dict[str, int] = {}
    for depart in sorted(graphe):
        if depart in etat:
            continue
        chemin = [depart]  # les tâches EN_COURS, dans l'ordre
        pile = [iter(graphe[depart])]  # pour chacune, les prérequis qu'il reste à explorer
        etat[depart] = EN_COURS
        while pile:
            suivante = next(pile[-1], None)
            if suivante is None:  # plus rien à explorer sous chemin[-1] : elle est terminée
                etat[chemin.pop()] = TERMINEE
                pile.pop()
            elif etat.get(suivante) == EN_COURS:
                # On revient sur une tâche du chemin : le cycle commence à SA position dans le chemin
                # (ce qui est avant n'est que la route qui y mène).
                return chemin[chemin.index(suivante):] + [suivante]
            elif suivante not in etat:
                etat[suivante] = EN_COURS
                chemin.append(suivante)
                pile.append(iter(graphe[suivante]))
    return None
    # Une pile explicite (`chemin` et `pile`) remplace les appels récursifs : la profondeur n'est plus
    # limitée par Python (1 000 appels), seulement par la mémoire. `sys.setrecursionlimit` n'est pas une
    # solution : au-delà de quelques milliers, c'est le programme entier qui plante, sans traceback.


# ============================================================================ Défi 149 : exécuter le traitement


def executer(taches: dict, dependances: dict) -> tuple[dict, dict, dict]:
    graphe = _graphe(dependances)
    if not isinstance(taches, dict):
        raise TypeError(f"taches doit être un dictionnaire {{nom: fonction}}, pas {type(taches).__name__}.")
    # TOUT vérifier AVANT de lancer quoi que ce soit : une faute de frappe découverte à 3 h du matin,
    # après l'envoi des SMS, laisse le traitement à moitié fait.
    _verifier_cles(taches, graphe, "taches")
    ordre = ordre_execution(dependances)  # lève ValueError s'il y a un cycle, avant tout lancement

    statuts: dict[str, str] = {}
    resultats: dict = {}
    erreurs: dict[str, Exception] = {}
    for tache in ordre:
        if any(statuts[p] != "ok" for p in graphe[tache]):
            # Un prérequis a échoué ou a été sauté : la tâche est sautée à son tour. C'est ce qui propage
            # l'échec à TOUTE la descendance, petits-enfants compris, sans parcours à part.
            statuts[tache] = "sautee"
            continue
        try:
            # Des paramètres NOMMÉS : `entrainer(variables=…)`. L'ordre des prérequis dans la liste ne
            # compte donc pas, et une tâche qui renomme ses paramètres échoue bruyamment.
            resultats[tache] = taches[tache](**{p: resultats[p] for p in graphe[tache]})
        except Exception as exc:  # pas BaseException : Ctrl+C (KeyboardInterrupt) doit tout arrêter
            statuts[tache] = "echec"
            erreurs[tache] = exc  # l'exception elle-même : son type, son message ET sa traceback
        else:
            statuts[tache] = "ok"  # même si elle renvoie None : « ok » parle du déroulement, pas du résultat
    return statuts, resultats, erreurs


# ============================================================================ Défi 150 : le chemin critique


def _verifier_durees(durees: dict, graphe: dict) -> None:
    if not isinstance(durees, dict):
        raise TypeError(f"durees doit être un dictionnaire {{nom: minutes}}, pas {type(durees).__name__}.")
    _verifier_cles(durees, graphe, "durees")
    for tache, duree in durees.items():
        if isinstance(duree, bool) or not isinstance(duree, numbers.Real):
            raise TypeError(f"La durée de {tache!r} doit être un nombre, pas {duree!r}.")
        if not (math.isfinite(duree) and duree >= 0):  # `not (… >= 0)` refuse aussi NaN
            raise ValueError(f"La durée de {tache!r} doit être positive ou nulle et finie, pas {duree!r}.")


def chemin_critique(durees: dict, dependances: dict) -> tuple[float, list[str]]:
    graphe = _graphe(dependances)
    _verifier_durees(durees, graphe)
    ordre = ordre_execution(dependances)
    if not ordre:
        return 0, []
    # Programmation dynamique dans l'ordre topologique : quand on arrive à une tâche, la fin de TOUS
    # ses prérequis est déjà connue. Chaque tâche est calculée une fois : O(n + m), sans récursion.
    fin: dict[str, float] = {}
    precedente: dict[str, str | None] = {}
    for tache in ordre:
        # Le prérequis qui finit le plus TARD décide du début ; à égalité, le premier par ordre
        # alphabétique (min sur (-fin, nom)).
        dernier = min(graphe[tache], key=lambda p: (-fin[p], p), default=None)
        precedente[tache] = dernier
        fin[tache] = (fin[dernier] if dernier is not None else 0) + durees[tache]
    # La fin du traitement, c'est la fin de la tâche qui finit la DERNIÈRE (à égalité, la première par
    # ordre alphabétique). On remonte ensuite de prérequis décisif en prérequis décisif.
    tache: str | None = min(fin, key=lambda t: (-fin[t], t))
    total = fin[tache]
    chemin = []
    while tache is not None:
        chemin.append(tache)
        tache = precedente[tache]
    return total, chemin[::-1]  # remonté de la fin vers le début : on le retourne


# ============================================================================ Défi 151 : planifier sur k travailleurs


def _reste(durees: dict, graphe: dict, ordre: list[str]) -> dict[str, float]:
    """Pour chaque tâche, la durée du plus long chemin qui part d'elle (elle comprise) jusqu'à la fin."""
    successeurs = _successeurs(graphe)
    reste: dict[str, float] = {}
    for tache in reversed(ordre):  # à l'envers : les successeurs sont calculés avant
        reste[tache] = durees[tache] + max((reste[s] for s in successeurs[tache]), default=0)
    return reste


def planifier(durees: dict, dependances: dict, travailleurs: int = 2) -> dict[str, tuple]:
    if isinstance(travailleurs, bool) or not isinstance(travailleurs, numbers.Integral):
        raise TypeError(f"travailleurs doit être un entier, pas {travailleurs!r}.")
    if travailleurs < 1:
        raise ValueError(f"Il faut au moins un travailleur (reçu {travailleurs}).")
    graphe = _graphe(dependances)
    _verifier_durees(durees, graphe)
    ordre = ordre_execution(dependances)
    successeurs = _successeurs(graphe)
    reste = _reste(durees, graphe, ordre)

    # Une SIMULATION PAR ÉVÉNEMENTS : on ne fait pas avancer le temps minute par minute (6 000 tours pour
    # une tâche de 100 heures, et faux avec des durées de 2,5 minutes), on saute directement à la
    # prochaine fin de tâche. Trois tas :
    restants = {tache: len(prerequis) for tache, prerequis in graphe.items()}
    prets = [(-reste[t], t) for t, n in restants.items() if n == 0]  # la plus prioritaire sort la première
    heapq.heapify(prets)
    libres = list(range(1, int(travailleurs) + 1))  # déjà un tas : le plus petit numéro sort le premier
    en_cours: list[tuple] = []  # (fin, travailleur, tâche)
    planning: dict[str, tuple] = {}
    temps = 0
    while prets or en_cours:
        # 1. Tant qu'il y a un travailleur libre ET une tâche prête : on attribue, à l'instant `temps`.
        while prets and libres:
            _, tache = heapq.heappop(prets)
            travailleur = heapq.heappop(libres)
            planning[tache] = (temps, temps + durees[tache], travailleur)
            heapq.heappush(en_cours, (temps + durees[tache], travailleur, tache))
        # 2. On saute à la prochaine fin, et on termine TOUTES les tâches qui finissent à cet instant
        #    AVANT d'attribuer quoi que ce soit (sinon une tâche prioritaire, débloquée par la 2e fin du
        #    même instant, arriverait après qu'un travailleur a pris une tâche moins urgente). Une tâche
        #    de durée 0 finit à l'instant où elle commence : elle est terminée ici, au même `temps`.
        temps = en_cours[0][0]
        while en_cours and en_cours[0][0] == temps:
            _, travailleur, tache = heapq.heappop(en_cours)
            heapq.heappush(libres, travailleur)
            for suivante in successeurs[tache]:
                restants[suivante] -= 1
                if restants[suivante] == 0:
                    heapq.heappush(prets, (-reste[suivante], suivante))
    return planning
    # Cette règle (« liste » avec la priorité du chemin critique) n'est pas toujours optimale (trouver le
    # meilleur planning est un problème NP-difficile), mais elle n'est jamais pire que 2 fois l'optimum,
    # et c'est celle qu'emploient la plupart des ordonnanceurs.


# ============================================================================ Tests (à la pytest)


def test_ordre_respecte_les_dependances_et_l_alphabet():
    ordre = ordre_execution(pipeline_exemple())
    assert ordre[:4] == ["extraire_appels", "alerte_support", "extraire_clients", "extraire_factures"]
    position = {tache: i for i, tache in enumerate(ordre)}
    for tache, prerequis in pipeline_exemple().items():
        assert all(position[p] < position[tache] for p in prerequis)


def test_un_cycle_est_refuse_et_nomme():
    try:
        ordre_execution({"a": ["b"], "b": ["c"], "c": ["a"], "d": []})
    except ValueError as exc:
        assert "cycle" in str(exc) and "a, b, c" in str(exc)
    else:
        raise AssertionError("un cycle doit lever ValueError")


def test_une_chaine_n_est_pas_une_liste():
    try:
        ordre_execution({"variables": "nettoyer"})
    except TypeError:
        pass
    else:
        raise AssertionError("une chaîne doit lever TypeError")


def test_relancer_suit_les_fleches_vers_l_aval():
    assert taches_a_relancer(pipeline_exemple(), ["evaluer"]) == ["evaluer", "rapport", "tableau_de_bord"]
    assert taches_a_relancer(pipeline_exemple(), iter(["sms"])) == ["sms"]  # un générateur, lu une fois


def test_trouver_cycle_ne_confond_pas_losange_et_cycle():
    assert trouver_cycle(pipeline_exemple()) is None  # evaluer et scorer partagent variables : pas un cycle
    assert trouver_cycle({"a": ["a"]}) == ["a", "a"]
    chaine = {f"t{i:04d}": [f"t{i + 1:04d}"] for i in range(5000)}
    assert trouver_cycle(chaine) is None  # 5 000 niveaux : pas de RecursionError
    chaine["t5000"] = ["t0000"]
    assert len(trouver_cycle(chaine)) == 5002


def test_executer_saute_toute_la_descendance_d_un_echec():
    appels = []

    def tache(nom, echoue=False):
        def f(**entrees):
            appels.append(nom)
            if echoue:
                raise RuntimeError("base indisponible")
            return nom

        return f

    dependances = {"b": ["a"], "c": ["b"], "d": []}
    statuts, resultats, erreurs = executer({"a": tache("a", echoue=True), "b": tache("b"), "c": tache("c"),
                                            "d": tache("d")}, dependances)
    assert statuts == {"a": "echec", "b": "sautee", "c": "sautee", "d": "ok"}
    assert appels == ["a", "d"] and resultats == {"d": "d"} and isinstance(erreurs["a"], RuntimeError)


def test_chemin_critique_du_traitement_de_nuit():
    total, chemin = chemin_critique(durees_exemple(), pipeline_exemple())
    assert total == 150
    assert chemin == ["extraire_factures", "nettoyer", "variables", "entrainer", "evaluer", "rapport",
                      "tableau_de_bord"]
    assert sum(durees_exemple()[t] for t in chemin) == total


def test_planifier_un_seul_travailleur_fait_tout_a_la_suite():
    plan = planifier(durees_exemple(), pipeline_exemple(), travailleurs=1)
    assert max(fin for _, fin, _ in plan.values()) == sum(durees_exemple().values())
    beaucoup = planifier(durees_exemple(), pipeline_exemple(), travailleurs=20)
    assert max(fin for _, fin, _ in beaucoup.values()) == chemin_critique(durees_exemple(), pipeline_exemple())[0]

"""Vérifications automatiques de la série 24 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur traite vos fonctions comme l'ordonnanceur d'une vraie nuit de production :

- il les appelle sur le traitement de nuit, puis sur des centaines de graphes tirés au hasard, et compare
  chaque réponse à sa propre version. Quand l'ordre diffère, il rejoue l'algorithme de Kahn jusqu'à la
  première différence et vous dit QUELLES tâches étaient prêtes à ce moment-là ;
- il glisse les pièges un par un : un prérequis écrit deux fois, une tâche citée seulement comme
  prérequis, une chaîne à la place d'une liste, un générateur qu'on ne lit qu'une fois, un cycle caché
  derrière des losanges, une chaîne de 5 000 tâches (la limite de récursion de Python est 1 000) ;
- au défi 149, ses fausses tâches notent chaque appel et ses paramètres : il sait qui a été appelé, dans
  quel ordre, avec quoi, et il lève lui-même des erreurs (et un Ctrl+C) au milieu de la nuit ;
- il COMPTE au lieu de chronométrer quand il le peut : les comparaisons entre noms de tâches (des chaînes
  « espionnes ») pour voir un `min` ou un `sorted` dans une boucle, et les lignes de VOTRE fichier
  exécutées (`sys.settrace`) pour voir une boucle qui relit tout, une exploration exponentielle des
  losanges, ou un temps qui avance d'une unité à la fois. Une boucle sans fin est arrêtée et expliquée.

Quand un résultat est faux, il rejoue les versions boguées classiques (par vagues au lieu du premier
prêt, flèches à l'envers, enfants directs seulement, l'ordre du petit graphe, « déjà vu » pris pour un
cycle, petits-enfants lancés quand même, priorité alphabétique…) pour vous dire LAQUELLE vous avez écrite.
"""

from __future__ import annotations

import copy
import heapq
import math
import random
import sys
import time

from .nuit import durees_exemple, generer_durees, generer_pipeline, pipeline_exemple
from .verif import Defi, Serie

# ---------------------------------------------------------------------------- outils


def _decrire(exc: BaseException) -> str:
    texte = str(exc).split("\n")[0]
    if len(texte) > 160:
        texte = texte[:159] + "…"
    return f"{type(exc).__name__}({texte!r})" if texte else type(exc).__name__


def _court(valeur, n: int = 160) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


def _milliers(n: float) -> str:
    """12 345 678 : des espaces entre les milliers, à la française."""
    return f"{n:,.0f}".replace(",", " ")


def _liste(noms) -> str:
    noms = list(noms)
    if len(noms) > 8:
        return ", ".join(map(str, noms[:8])) + f"… ({len(noms)} en tout)"
    return ", ".join(map(str, noms)) if noms else "(aucune)"


class _BoucleSansFin(BaseException):
    """Levée dans VOTRE fonction quand elle exécute trop de lignes (BaseException : `except Exception` ne l'attrape pas)."""


class _TropDeComparaisons(BaseException):
    """Levée quand votre fonction compare des noms de tâches bien plus souvent qu'un tas ne le ferait."""


class _Compteur:
    lignes = 0


def _trace(f, appel, lignes_max: int | None):
    """(résultat, lignes exécutées dans le fichier de f) ; arrête net au-delà de `lignes_max`."""
    code = getattr(f, "__code__", None)
    if code is None or lignes_max is None:  # sans limite : pas de traceur (il ralentirait le chronomètre)
        return appel(), 0
    fichier, compte = code.co_filename, [0]

    def local(frame, evenement, arg):
        if evenement == "line":
            compte[0] += 1
            if lignes_max is not None and compte[0] > lignes_max:
                raise _BoucleSansFin
        return local

    def global_(frame, evenement, arg):
        return local if frame.f_code.co_filename == fichier else None

    precedent = sys.gettrace()
    sys.settrace(global_)
    try:
        return appel(), compte[0]
    finally:
        sys.settrace(precedent)


_CONSEIL_RECURSION = ("Python s'arrête à 1 000 appels imbriqués, et un traitement de nuit peut enchaîner 5 000 tâches. "
                      "Remplacez la récursion par une boucle : une pile explicite (une liste, `append` / `pop`), ou un "
                      "parcours dans l'ordre de `ordre_execution`, où les prérequis sont toujours déjà calculés.")


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = 2_000_000,
           boucle: str = "") -> tuple:
    """(résultat, lignes) de f(*args), en transformant toute erreur en message lisible."""
    limite = sys.getrecursionlimit()
    try:
        return _trace(f, lambda: f(*args), lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} ne s'arrête pas : votre fonction a exécuté plus de {_milliers(lignes_max)} lignes "
                             f"(elle a été arrêtée). {boucle or conseil}".rstrip()) from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError. {_CONSEIL_RECURSION}") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None
    finally:
        if sys.getrecursionlimit() != limite:
            sys.setrecursionlimit(limite)
            raise AssertionError(f"{quoi} modifie la limite de récursion (`sys.setrecursionlimit`) : c'est interdit. Au-delà "
                                 f"de quelques milliers, ce n'est plus Python qui lève RecursionError, c'est le programme "
                                 f"entier qui plante, sans un message. {_CONSEIL_RECURSION}")


def _resultat(f, *args, quoi: str, **options):
    return _appel(f, args, quoi, **options)[0]


def _leve(f, args: tuple, attendues, quoi: str, conseil: str = "", mots: tuple = ()) -> BaseException:
    """Exige que f(*args) lève une des exceptions `attendues` (et que le message contienne `mots`)."""
    attendues = attendues if isinstance(attendues, tuple) else (attendues,)
    noms = " ou ".join(e.__name__ for e in attendues)
    try:
        resultat, _ = _trace(f, lambda: f(*args), 2_000_000)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} ne s'arrête pas (votre fonction a été arrêtée) : il devrait lever {noms}. "
                             f"{conseil}".rstrip()) from None
    except attendues as exc:
        texte = str(exc).lower()
        absents = [m for m in mots if m.lower() not in texte]
        if absents:
            raise AssertionError(f"{quoi} lève bien {type(exc).__name__}, mais son message ({str(exc)!r}) ne contient pas "
                                 f"{' ni '.join(repr(m) for m in absents)}. Un message d'erreur sert à celui qui le lira à "
                                 f"3 h du matin : nommez le problème et les tâches en cause.") from None
        return exc
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {noms}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {noms}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _intact(avant, apres, quoi: str) -> None:
    if avant != apres:
        raise AssertionError(f"{quoi} a MODIFIÉ le dictionnaire reçu. L'appelant s'en sert encore (le même graphe sert "
                             f"à l'ordre, aux relances, au planning) : travaillez sur VOS structures (`_graphe`), sans "
                             f"`setdefault`, `pop`, `remove` ni `append` sur ce que vous avez reçu.")


def _avec_copie(f, entree, quoi: str, *autres, **options):
    """Appelle f(entree, *autres) et vérifie qu'`entree` n'a pas été modifiée."""
    avant = copy.deepcopy(entree)
    resultat = _resultat(f, entree, *autres, quoi=quoi, **options)
    _intact(avant, entree, quoi)
    return resultat


# ---------------------------------------------------------------------------- références


def _graphe(dependances: dict) -> dict[str, set[str]]:
    graphe: dict[str, set[str]] = {}
    for tache, prerequis in dependances.items():
        graphe.setdefault(tache, set()).update(prerequis)
        for nom in prerequis:
            graphe.setdefault(nom, set())
    return graphe


def _successeurs(graphe: dict) -> dict[str, list[str]]:
    successeurs: dict[str, list[str]] = {t: [] for t in graphe}
    for tache, prerequis in graphe.items():
        for nom in prerequis:
            successeurs[nom].append(tache)
    return successeurs


def _ref_ordre(dependances: dict) -> list[str]:
    graphe = _graphe(dependances)
    successeurs = _successeurs(graphe)
    restants = {t: len(p) for t, p in graphe.items()}
    prets = sorted(t for t, n in restants.items() if n == 0)
    ordre = []
    while prets:
        tache = heapq.heappop(prets)
        ordre.append(tache)
        for s in successeurs[tache]:
            restants[s] -= 1
            if restants[s] == 0:
                heapq.heappush(prets, s)
    if len(ordre) < len(graphe):
        raise ValueError("cycle")
    return ordre


def _ordre_par_vagues(dependances: dict) -> list[str]:
    graphe, faites, ordre = _graphe(dependances), set(), []
    while len(faites) < len(graphe):
        vague = sorted(t for t in graphe if t not in faites and graphe[t] <= faites)
        if not vague:
            return ordre
        ordre += vague
        faites.update(vague)
    return ordre


def _ancetres(graphe: dict, depart) -> set[str]:
    vus, pile = set(depart), list(depart)
    while pile:
        for p in graphe[pile.pop()]:
            if p not in vus:
                vus.add(p)
                pile.append(p)
    return vus


def _descendants(graphe: dict, depart) -> set[str]:
    successeurs = _successeurs(graphe)
    vus, pile = set(depart), list(depart)
    while pile:
        for s in successeurs[pile.pop()]:
            if s not in vus:
                vus.add(s)
                pile.append(s)
    return vus


def _ref_relancer(dependances: dict, modifiees) -> list[str]:
    a_relancer = _descendants(_graphe(dependances), list(modifiees))
    return [t for t in _ref_ordre(dependances) if t in a_relancer]


def _fins(durees: dict, dependances: dict) -> tuple[dict, dict]:
    graphe = _graphe(dependances)
    fin, precedente = {}, {}
    for tache in _ref_ordre(dependances):
        dernier = min(graphe[tache], key=lambda p: (-fin[p], p), default=None)
        precedente[tache] = dernier
        fin[tache] = (fin[dernier] if dernier is not None else 0) + durees[tache]
    return fin, precedente


def _ref_critique(durees: dict, dependances: dict) -> tuple:
    fin, precedente = _fins(durees, dependances)
    if not fin:
        return 0, []
    tache = min(fin, key=lambda t: (-fin[t], t))
    total, chemin = fin[tache], []
    while tache is not None:
        chemin.append(tache)
        tache = precedente[tache]
    return total, chemin[::-1]


def _restes(durees: dict, dependances: dict) -> dict:
    successeurs = _successeurs(_graphe(dependances))
    reste: dict = {}
    for tache in reversed(_ref_ordre(dependances)):
        reste[tache] = durees[tache] + max((reste[s] for s in successeurs[tache]), default=0)
    return reste


def _ref_planifier(durees: dict, dependances: dict, travailleurs: int, priorite: str = "reste",
                   fins: str = "toutes", premier: int = 1) -> dict:
    """La simulation de référence ; `priorite`, `fins` et `premier` fabriquent les versions boguées classiques."""
    graphe = _graphe(dependances)
    successeurs = _successeurs(graphe)
    reste = _restes(durees, dependances)

    def cle(t):
        return (t,) if priorite == "alpha" else (-reste[t], t)

    restants = {t: len(p) for t, p in graphe.items()}
    prets = [(cle(t), t) for t, n in restants.items() if n == 0]
    heapq.heapify(prets)
    libres = list(range(premier, premier + travailleurs))
    en_cours: list = []
    planning: dict = {}
    temps = 0
    while prets or en_cours:
        while prets and libres:
            _, tache = heapq.heappop(prets)
            w = heapq.heappop(libres)
            planning[tache] = (temps, temps + durees[tache], w)
            heapq.heappush(en_cours, (temps + durees[tache], w, tache))
        temps = en_cours[0][0]
        while en_cours and en_cours[0][0] == temps:
            _, w, tache = heapq.heappop(en_cours)
            heapq.heappush(libres, w)
            for s in successeurs[tache]:
                restants[s] -= 1
                if restants[s] == 0:
                    heapq.heappush(prets, (cle(s), s))
            if fins == "une":
                break
    return planning


def _losanges(etages: int, sommet_en_premier: bool = False) -> dict[str, list[str]]:
    """a00 → (a01, b01) → (a02, b02) → … : chaque étage a besoin des DEUX tâches de l'étage d'avant (2^etages chemins).

    Avec `sommet_en_premier`, les étages sont numérotés à l'envers : la tâche qui a besoin de toutes les autres
    s'appelle « a00 », et un parcours en profondeur qui part de la première par ordre alphabétique descend tout.
    """
    def nom(lettre, i):
        return f"{lettre}{etages - i if sommet_en_premier else i:02d}"

    dependances: dict[str, list[str]] = {nom("a", 0): [], nom("b", 0): [nom("a", 0)]}
    for i in range(1, etages + 1):
        avant = [nom("a", i - 1), nom("b", i - 1)]
        dependances[nom("a", i)] = list(avant)
        dependances[nom("b", i)] = list(avant)
    return dependances


def _chaine(n: int, cycle: bool = False) -> dict[str, list[str]]:
    """t0000 a besoin de t0001, qui a besoin de t0002… (et, avec `cycle`, la dernière a besoin de la première)."""
    dependances = {f"t{i:04d}": [f"t{i + 1:04d}"] for i in range(n)}
    dependances[f"t{n:04d}"] = ["t0000"] if cycle else []
    return dependances


_CHAINE_MSG = "une chaîne de 5 000 tâches"


# ---------------------------------------------------------------------------- les noms « espions »


class _Nom(str):
    """Un nom de tâche qui compte les comparaisons `<`, `>`, `<=`, `>=` (un tas en fait O(n log n))."""

    compte = 0
    budget: float = math.inf

    def _compter(self):
        _Nom.compte += 1
        if _Nom.compte > _Nom.budget:
            raise _TropDeComparaisons

    def __lt__(self, autre):
        self._compter()
        return str.__lt__(self, autre)

    def __gt__(self, autre):
        self._compter()
        return str.__gt__(self, autre)

    def __le__(self, autre):
        self._compter()
        return str.__le__(self, autre)

    def __ge__(self, autre):
        self._compter()
        return str.__ge__(self, autre)

    __hash__ = str.__hash__


def _espionner(dependances: dict) -> dict:
    return {_Nom(t): [_Nom(p) for p in prerequis] for t, prerequis in dependances.items()}


# ============================================================================ Défi 146 : ordre_execution


def _exiger_liste(resultat, quoi: str) -> list:
    if resultat is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(resultat, list):
        pourquoi = ("Un générateur ne se lit qu'une fois : renvoyez une vraie liste." if hasattr(resultat, "__next__")
                    else "Renvoyez une liste (`list(...)`) : c'est ce que promet la signature.")
        raise AssertionError(f"{quoi} renvoie {type(resultat).__name__} ({_court(resultat, 80)}) au lieu d'une liste. "
                             f"{pourquoi}")
    return resultat


def _expliquer_ordre(obtenu: list, dependances: dict, quoi: str) -> None:
    """Lève une AssertionError qui explique en quoi `obtenu` n'est pas l'ordre attendu (ne fait rien s'il l'est)."""
    graphe = _graphe(dependances)
    attendu = _ref_ordre(dependances)
    if obtenu == attendu:
        return
    taches = set(graphe)
    vus, doublons = set(), []
    for t in obtenu:
        if t in vus:
            doublons.append(t)
        vus.add(t)
    if doublons:
        raise AssertionError(f"{quoi} contient deux fois {_liste(sorted(set(doublons)))}. Chaque tâche ne se lance qu'une "
                             f"fois : elle passe dans la liste des prêtes quand son compteur de prérequis restants "
                             f"tombe à 0, ce qui n'arrive qu'une fois.")
    inconnues = [t for t in obtenu if t not in taches]
    if inconnues:
        raise AssertionError(f"{quoi} contient des tâches qui n'existent pas : {_liste(inconnues)}.")
    manquantes = sorted(taches - vus)
    if manquantes:
        cles = set(dependances)
        seulement_prerequis = [t for t in manquantes if t not in cles]
        if seulement_prerequis:
            raise AssertionError(f"{quoi} oublie {_liste(seulement_prerequis)} : ces tâches ne sont pas des clés du "
                                 f"dictionnaire, elles ne sont citées que comme PRÉREQUIS. Ce sont des tâches comme les "
                                 f"autres : `graphe.setdefault(nom, set())` pour chaque prérequis.")
        raise AssertionError(f"{quoi} oublie {_liste(manquantes)}. Si vous arrêtez la boucle trop tôt, ou si un "
                             f"compteur ne tombe jamais à 0 (un prérequis écrit deux fois ?), des tâches restent en route.")
    position = {t: i for i, t in enumerate(obtenu)}
    for t in obtenu:
        for p in sorted(graphe[t]):
            if position[p] > position[t]:
                if all(position[q] > position[u] for u in graphe for q in graphe[u]) and any(graphe.values()):
                    raise AssertionError(f"{quoi} est à l'envers : chaque tâche y passe AVANT ses prérequis ({t!r} avant "
                                         f"{p!r}). La liste d'une tâche, ce sont les tâches dont elle a BESOIN : on lance "
                                         f"d'abord les prérequis.")
                raise AssertionError(f"{quoi} lance {t!r} (position {position[t] + 1}) avant son prérequis {p!r} (position "
                                     f"{position[p] + 1}). Une tâche n'est prête que quand TOUS ses prérequis sont faits.")
    # Un ordre valide, mais pas celui de la règle : on rejoue Kahn jusqu'à la première différence.
    i = next(k for k, (a, b) in enumerate(zip(obtenu, attendu)) if a != b)
    faites = set(attendu[:i])
    prets = sorted(t for t in graphe if t not in faites and graphe[t] <= faites)
    deja = f"après {_liste(attendu[:i])}" if i else "au départ"
    message = (f"{quoi} est un ordre possible, mais pas celui de la règle. {deja.capitalize()}, les tâches prêtes sont "
               f"{_liste(prets)} : on prend la première par ordre alphabétique, {attendu[i]!r}, et vous prenez "
               f"{obtenu[i]!r} (position {i + 1}).")
    if obtenu == _ordre_par_vagues(dependances):
        message += (" Vous raisonnez par « vagues » (toutes les prêtes du moment, puis les suivantes) : une tâche qui "
                    "devient prête passe avant les autres si son nom est plus petit. Un TAS (`heapq`) fait exactement ça.")
    elif obtenu == sorted(obtenu):
        message += " Vous triez toutes les tâches par nom : l'alphabet ne départage que les tâches PRÊTES."
    else:
        message += " Gardez les tâches prêtes dans un tas (`heapq.heappush` / `heapq.heappop`)."
    raise AssertionError(message)


def _verifier_ordre(f, dependances: dict, quoi: str, **options) -> list:
    try:
        obtenu = _exiger_liste(_avec_copie(f, dependances, quoi, **options), quoi)
    except AssertionError as exc:
        texte = str(exc)
        seulement = sorted(t for t in _graphe(dependances) if t not in dependances)
        if seulement and (" lève ValueError" in texte or " lève KeyError" in texte):
            raise AssertionError(f"{texte} Ce graphe n'a pourtant aucun cycle. Attention à {_liste(seulement)} : ce ne sont pas "
                                 f"des clés, elles ne sont citées que comme PRÉREQUIS. Ce sont des tâches comme les autres "
                                 f"(sans prérequis, donc prêtes dès le départ) : `graphe.setdefault(nom, set())` pour chaque "
                                 f"prérequis.") from None
        raise
    _expliquer_ordre(obtenu, dependances, quoi)
    return obtenu


_CONSEIL_SCAN = ("C'est O(n²) : à chaque étape, vous relisez toutes les tâches pour trouver la prochaine prête. Tenez un "
                 "compteur de prérequis restants par tâche, et des « successeurs » : quand une tâche est faite, seules "
                 "celles qui l'attendaient peuvent devenir prêtes.")

_BOUCLE_KAHN = ("Si le graphe a un cycle, aucune tâche du cycle ne devient jamais prête : une boucle `while` qui attend "
                "« la prochaine tâche prête » tourne alors sans fin. Arrêtez-vous quand il n'y a plus de tâche prête, et "
                "comparez `len(ordre)` au nombre de tâches.")


def _verifier_cycles_refuses(f, quoi_f: str) -> None:
    for dependances, du_cycle in [
        ({"a": ["b"], "b": ["c"], "c": ["a"], "d": []}, {"a", "b", "c"}),
        ({"x": ["a"], "a": ["b"], "b": ["a"]}, {"a", "b"}),
        ({"a": ["a"]}, {"a"}),
    ]:
        quoi = f"{quoi_f}({dependances})"
        exc = _leve(f, (dependances,), ValueError, quoi, _BOUCLE_KAHN, mots=("cycle",))
        if not any(t in str(exc) for t in du_cycle):
            raise AssertionError(f"{quoi} lève bien ValueError, mais son message ({str(exc)!r}) ne nomme aucune tâche du "
                                 f"cycle ({_liste(sorted(du_cycle))}). Nommez les tâches bloquées (celles dont le compteur "
                                 f"n'est jamais tombé à 0) : c'est la première chose qu'on cherchera.")


def _verifier_chaine_refusee(f, quoi_f: str, *autres) -> None:
    quoi = f"{quoi_f}({{'variables': 'nettoyer'}}{', …' if autres else ''})"
    _leve(f, ({"variables": "nettoyer"}, *autres), TypeError, quoi,
          "Les prérequis doivent être une liste : une chaîne se parcourt LETTRE PAR LETTRE, et « variables » aurait "
          "pour prérequis « n », « e », « t »… Vérifiez `isinstance(prerequis, str)` et levez TypeError.")


def verifier_ordre_execution(f) -> None:
    # 1. Les cas simples.
    obtenu = _exiger_liste(_avec_copie(f, {}, "ordre_execution({})"), "ordre_execution({})")
    if obtenu != []:
        raise AssertionError(f"ordre_execution({{}}) renvoie {_court(obtenu)} au lieu de [] (aucune tâche).")
    _verifier_ordre(f, {"c": ["a"], "b": ["a"], "a": []}, "ordre_execution({'c': ['a'], 'b': ['a'], 'a': []})",
                    boucle=_BOUCLE_KAHN)
    _verifier_ordre(f, {"b": ["a"]}, "ordre_execution({'b': ['a']})", boucle=_BOUCLE_KAHN)

    # 2. Le traitement de nuit (une tâche citée seulement comme prérequis, une tâche prête « en cours de vague »).
    _verifier_ordre(f, pipeline_exemple(), "ordre_execution(pipeline_exemple())", boucle=_BOUCLE_KAHN)

    # 3. Les doublons ne comptent qu'une fois.
    quoi = "ordre_execution({'a': [], 'b': ['a', 'a'], 'c': ['b', 'a', 'b']})"
    try:
        _verifier_ordre(f, {"a": [], "b": ["a", "a"], "c": ["b", "a", "b"]}, quoi, lignes_max=200_000)
    except AssertionError as exc:
        raise AssertionError(f"{exc} Un prérequis écrit deux fois ne compte qu'une fois : si `restants['b']` vaut 2, il "
                             f"ne tombera jamais à 0 et « b » ne sera jamais prête. Rangez les prérequis dans un `set`.") \
            from None

    # 4. Des tuples et des ensembles à la place des listes, des clés dans le désordre.
    melange = {t: tuple(p) if i % 2 else set(p) for i, (t, p) in enumerate(reversed(list(pipeline_exemple().items())))}
    _verifier_ordre(f, melange, "ordre_execution(<le traitement de nuit, avec des tuples et des ensembles>)")

    # 5. Les cycles et les entrées mal formées.
    _verifier_cycles_refuses(f, "ordre_execution")
    _verifier_chaine_refusee(f, "ordre_execution")

    # 6. Des graphes tirés au hasard.
    for graine in range(1, 31):
        graphe = generer_pipeline(random.Random(graine).randint(5, 120), graine=graine)
        _verifier_ordre(f, graphe, f"ordre_execution(generer_pipeline(…, graine={graine}))")

    # 7. Pas de boucle Python qui relit toutes les tâches à chaque étape (compte des lignes, pas de chronomètre).
    def lignes(n):
        # z00000 a besoin de z00001, qui a besoin de z00002… : la seule tâche prête est toujours la DERNIÈRE par
        # ordre alphabétique, celle qu'une boucle « je relis toutes les tâches dans l'ordre » trouve en dernier.
        chaine = {f"z{i:05d}": [f"z{i + 1:05d}"] for i in range(n - 1)}
        return _appel(f, (chaine,), "ordre_execution(<une chaîne>)", lignes_max=3_000_000, boucle=_CONSEIL_SCAN)[1]

    a, b = lignes(300), lignes(600)
    if b > 3.2 * a and b > 30_000:
        raise AssertionError(f"ordre_execution exécute {_milliers(a)} lignes pour une chaîne de 300 tâches, et "
                             f"{_milliers(b)} pour 600 : deux fois plus de tâches, {b / a:.1f} fois plus de travail. "
                             f"{_CONSEIL_SCAN}")

    # 8. Pas de min / sorted sur toutes les prêtes à chaque tour : on compte les comparaisons entre noms.
    n = 3000
    large = {f"t{i:05d}": [] for i in random.Random(7).sample(range(10 * n), n)}
    large.update({"zz_fin": sorted(large)[: n // 2]})
    espion = _espionner(large)
    budget = 8 * n * math.log2(n)
    _Nom.compte, _Nom.budget = 0, budget
    try:
        obtenu = _trace(f, lambda: f(espion), 5_000_000)[0]
    except _TropDeComparaisons:
        raise AssertionError(f"ordre_execution sur {n} tâches toutes prêtes au départ compare des noms de tâches plus de "
                             f"{_milliers(budget)} fois (un tas en fait environ {_milliers(2 * n * math.log2(n))}) : un "
                             f"`min(prets)`, un `sorted(prets)` ou un `prets.sort()` À CHAQUE TOUR relit toutes les tâches "
                             f"prêtes, soit O(n²). Un tas (`heapq`) sort la plus petite en O(log n).") from None
    except _BoucleSansFin:
        raise AssertionError(f"ordre_execution sur {n} tâches exécute des millions de lignes. {_CONSEIL_SCAN}") from None
    finally:
        _Nom.budget = math.inf
    _expliquer_ordre(_exiger_liste(obtenu, "ordre_execution(<3000 tâches>)"), large, "ordre_execution(<3000 tâches>)")

    # 9. Une vraie taille.
    grand = generer_pipeline(20_000, graine=146)
    debut = time.perf_counter()
    obtenu = _resultat(f, grand, quoi="ordre_execution(generer_pipeline(20000))", lignes_max=None)
    duree = time.perf_counter() - debut
    _expliquer_ordre(_exiger_liste(obtenu, "ordre_execution(<20000 tâches>)"), grand, "ordre_execution(<20000 tâches>)")
    if duree > 2.0:
        raise AssertionError(f"ordre_execution est juste, mais 20 000 tâches prennent {duree:.1f} s (environ 0,05 s "
                             f"attendu). Un `x in liste`, `liste.remove(x)` ou `liste.index(x)` sur une grande liste "
                             f"relit toute la liste : utilisez des `set` et des dictionnaires.")


# ============================================================================ Défi 147 : taches_a_relancer


def _verifier_relancer(f, dependances: dict, modifiees, quoi: str, depart=None, **options) -> list:
    """`depart` : la liste des tâches modifiées, à donner quand `modifiees` est un générateur (lu une seule fois)."""
    graphe = _graphe(dependances)
    depart = list(modifiees) if depart is None else depart
    avant = copy.deepcopy(dependances)
    obtenu = _exiger_liste(_resultat(f, dependances, modifiees, quoi=quoi, **options), quoi)
    _intact(avant, dependances, quoi)
    attendu = _ref_relancer(dependances, depart)
    if obtenu == attendu:
        return obtenu
    if not obtenu and attendu and not isinstance(modifiees, (list, set, tuple)):
        raise AssertionError(f"{quoi} renvoie [] au lieu de {_court(attendu)}. `modifiees` est ici un GÉNÉRATEUR : il ne "
                             f"se lit qu'une fois. Si vous le parcourez pour vérifier les noms, il est vide ensuite. "
                             f"Commencez par `depart = list(modifiees)`.")
    if len(set(obtenu)) < len(obtenu):
        raise AssertionError(f"{quoi} contient des doublons : {_court(obtenu)}. Chaque tâche n'est relancée qu'une fois.")
    if set(obtenu) == set(attendu):
        ordre_complet = _ref_ordre(dependances)
        sous_graphe = {t: [p for p in graphe[t] if p in set(attendu)] for t in attendu}
        if obtenu == _ref_ordre(sous_graphe):
            raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {_court(attendu)} : c'est l'ordre du PETIT "
                                 f"graphe des seules tâches à relancer. Le traitement, lui, suit l'ordre du graphe ENTIER "
                                 f"(une tâche relancée peut attendre un prérequis qui, lui, n'est pas relancé) : "
                                 f"`[t for t in ordre_execution(dependances) if t in a_relancer]`.")
        if obtenu == sorted(obtenu):
            raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {_court(attendu)} : triées par nom au lieu "
                                 f"de l'ordre d'exécution. Filtrez `ordre_execution(dependances)`.")
        raise AssertionError(f"{quoi} renvoie les bonnes tâches dans le désordre : {_court(obtenu)} au lieu de "
                             f"{_court(attendu)}. L'ordre attendu est celui de `ordre_execution` "
                             f"({_court([t for t in ordre_complet if t in set(obtenu)])}).")
    ancetres = _ancetres(graphe, depart)
    if set(obtenu) == ancetres and ancetres != set(attendu):
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} : ce sont les PRÉREQUIS (ce dont la tâche a besoin), pas ce "
                             f"qui en dépend. Les flèches sont à retourner : il faut les tâches qui ATTENDENT la tâche "
                             f"modifiée, puis celles qui les attendent… (`_successeurs`).")
    enfants = set(depart) | {t for t in graphe if graphe[t] & set(depart)}
    if set(obtenu) == enfants and enfants != set(attendu):
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {_court(attendu)} : seulement les tâches qui "
                             f"dépendent DIRECTEMENT de la tâche modifiée. Leurs propres descendantes ont été calculées "
                             f"avec des données fausses elles aussi : continuez le parcours (une file, `deque`).")
    if set(obtenu) == set(attendu) - set(depart):
        raise AssertionError(f"{quoi} oublie les tâches modifiées elles-mêmes ({_liste(depart)}) : c'est la première "
                             f"chose à relancer.")
    manquantes, en_trop = sorted(set(attendu) - set(obtenu)), sorted(set(obtenu) - set(attendu))
    raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {_court(attendu)} (manquantes : {_liste(manquantes)} ; "
                         f"en trop : {_liste(en_trop)}).")


_BOUCLE_LOSANGES = ("Dans un graphe en losanges (chaque étage a besoin des deux tâches de l'étage d'avant), il y a 2^30 "
                    "chemins : sans un ensemble « déjà vu », votre parcours repasse par les mêmes tâches un nombre "
                    "exponentiel de fois. N'ajoutez une tâche à la file que si elle n'y est pas déjà passée.")


def verifier_taches_a_relancer(f) -> None:
    p = pipeline_exemple
    _verifier_relancer(f, p(), ["evaluer"], "taches_a_relancer(pipeline_exemple(), ['evaluer'])")
    _verifier_relancer(f, p(), [], "taches_a_relancer(pipeline_exemple(), [])")
    _verifier_relancer(f, p(), ["nettoyer"], "taches_a_relancer(pipeline_exemple(), ['nettoyer'])")
    _verifier_relancer(f, p(), ["sms", "alerte_support"], "taches_a_relancer(pipeline_exemple(), ['sms', 'alerte_support'])")
    _verifier_relancer(f, p(), ["extraire_appels"], "taches_a_relancer(pipeline_exemple(), ['extraire_appels'])")
    _verifier_relancer(f, p(), ["variables", "entrainer", "variables"],
                       "taches_a_relancer(pipeline_exemple(), ['variables', 'entrainer', 'variables'])")
    # L'ordre du graphe entier, pas celui du petit graphe des tâches à relancer.
    _verifier_relancer(f, {"b": ["x"], "c": [], "x": []}, ["b", "c"],
                       "taches_a_relancer({'b': ['x'], 'c': [], 'x': []}, ['b', 'c'])")
    # Un ensemble, un générateur.
    _verifier_relancer(f, p(), {"scorer"}, "taches_a_relancer(pipeline_exemple(), {'scorer'})")
    _verifier_relancer(f, p(), (t for t in ["nettoyer"]), "taches_a_relancer(pipeline_exemple(), <un générateur>)",
                       depart=["nettoyer"])

    # Les erreurs.
    _leve(f, (p(), "nettoyer"), TypeError, "taches_a_relancer(pipeline_exemple(), 'nettoyer')",
          "Une chaîne seule se parcourt lettre par lettre (« n », « e »…) : refusez-la avec TypeError avant de la lire, "
          "l'appelant voulait sans doute écrire ['nettoyer'].")
    exc = _leve(f, (p(), ["netoyer"]), ValueError, "taches_a_relancer(pipeline_exemple(), ['netoyer'])",
                "Une tâche inconnue (une faute de frappe) ne doit pas être ignorée en silence.")
    if "netoyer" not in str(exc):
        raise AssertionError(f"taches_a_relancer(pipeline_exemple(), ['netoyer']) lève bien ValueError, mais le message "
                             f"({str(exc)!r}) ne nomme pas la tâche inconnue : 'netoyer'.")
    _leve(f, ({"a": ["b"], "b": ["a"], "c": []}, ["c"]), ValueError, "taches_a_relancer(<un graphe avec un cycle>, ['c'])",
          "Comme au défi 146 : un graphe avec un cycle ne peut pas s'exécuter, même si la tâche relancée n'en fait pas "
          "partie. Appelez `ordre_execution`, qui le refuse.")
    _verifier_chaine_refusee(f, "taches_a_relancer", ["variables"])

    # Des graphes tirés au hasard.
    for graine in range(1, 31):
        rng = random.Random(graine)
        dependances = generer_pipeline(rng.randint(5, 150), graine=graine)
        taches = sorted(_graphe(dependances))
        depart = rng.sample(taches, rng.randint(1, 3))
        _verifier_relancer(f, dependances, depart, f"taches_a_relancer(generer_pipeline(…, graine={graine}), {depart})")

    # Les losanges : pas d'explosion exponentielle.
    _verifier_relancer(f, _losanges(30), ["a00"], "taches_a_relancer(<un graphe en losanges de 30 étages>, ['a00'])",
                       lignes_max=200_000, boucle=_BOUCLE_LOSANGES)

    # Une vraie taille.
    grand = generer_pipeline(20_000, graine=147)
    depart = sorted(_graphe(grand))[:20]
    debut = time.perf_counter()
    obtenu = _resultat(f, grand, depart, quoi="taches_a_relancer(generer_pipeline(20000), …)", lignes_max=None)
    duree = time.perf_counter() - debut
    if obtenu != _ref_relancer(grand, depart):
        _verifier_relancer(f, grand, depart, "taches_a_relancer(generer_pipeline(20000), …)")
    if duree > 2.0:
        raise AssertionError(f"taches_a_relancer est juste, mais 20 000 tâches prennent {duree:.1f} s (environ 0,1 s "
                             f"attendu). Retournez les flèches UNE fois (`_successeurs`) au lieu de relire tout le graphe "
                             f"pour chaque tâche, et utilisez un `set` pour « déjà vu », pas une liste.")


# ============================================================================ Défi 148 : trouver_cycle


def _verifier_un_cycle(cycle, dependances: dict, quoi: str) -> None:
    graphe = _graphe(dependances)
    if cycle is None:
        raise AssertionError(f"{quoi} renvoie None, alors que ce graphe a un cycle (`ordre_execution` le refuse).")
    if not isinstance(cycle, list) or not all(isinstance(t, str) for t in cycle):
        raise AssertionError(f"{quoi} renvoie {_court(cycle)} au lieu d'une liste de noms de tâches.")
    if len(cycle) >= 2 and cycle[0] != cycle[-1] and cycle[-1] in cycle[:-1]:
        k = cycle.index(cycle[-1])
        raise AssertionError(f"{quoi} renvoie {_court(cycle)} : le cycle, c'est {_court(cycle[k:])}. Vous renvoyez aussi le "
                             f"chemin qui y MÈNE ({_court(cycle[:k])}) : coupez-le au premier passage de la tâche qui se "
                             f"répète, `chemin[chemin.index(t):] + [t]`.")
    if len(cycle) < 2 or cycle[0] != cycle[-1]:
        if cycle and len(set(cycle)) == len(cycle) and all(cycle[(i + 1) % len(cycle)] in graphe.get(t, ())
                                                           for i, t in enumerate(cycle)):
            raise AssertionError(f"{quoi} renvoie {_court(cycle)} : c'est bien un cycle, mais la règle demande de RÉPÉTER "
                                 f"la première tâche à la fin ({_court(cycle + cycle[:1])}) : on voit ainsi la flèche qui "
                                 f"referme la boucle.")
        raise AssertionError(f"{quoi} renvoie {_court(cycle)} : un cycle commence et finit par la même tâche.")
    inconnues = [t for t in cycle if t not in graphe]
    if inconnues:
        raise AssertionError(f"{quoi} renvoie {_court(cycle)}, qui contient des tâches inconnues : {_liste(inconnues)}.")
    corps = cycle[:-1]
    if all(cycle[i] in graphe[cycle[i + 1]] for i in range(len(corps))) and not all(
            cycle[i + 1] in graphe[cycle[i]] for i in range(len(corps))):
        raise AssertionError(f"{quoi} renvoie {_court(cycle)} : ce cycle est parcouru dans le mauvais sens. La règle : "
                             f"chaque tâche a BESOIN de la suivante ({cycle[1]!r} doit être un prérequis de {cycle[0]!r}). "
                             f"Retournez la liste, ou suivez les prérequis plutôt que les successeurs.")
    for i in range(len(corps)):
        if cycle[i + 1] not in graphe[cycle[i]]:
            raise AssertionError(f"{quoi} renvoie {_court(cycle)} : {cycle[i + 1]!r} n'est pas un prérequis de "
                                 f"{cycle[i]!r}, cette flèche n'existe pas.")
    if len(set(corps)) < len(corps):
        debut = next(t for t in corps if corps.count(t) > 1)
        k = corps.index(debut)
        raise AssertionError(f"{quoi} renvoie {_court(cycle)} : {debut!r} y apparaît deux fois. Vous renvoyez aussi le "
                             f"chemin qui MÈNE au cycle : coupez-le au premier passage de la tâche qui se répète, "
                             f"`chemin[chemin.index(t):] + [t]` (ici {_court(corps[k:k + corps[k + 1:].index(debut) + 2])}).")


_BOUCLE_DFS = ("Un parcours en profondeur doit marquer les tâches TERMINÉES (explorées jusqu'au bout sans cycle) et ne jamais "
               "y redescendre : dans un graphe en losanges de 30 étages, il y a 2^30 chemins.")


def _cycle(f, dependances: dict, quoi: str, **options):
    return _avec_copie(f, dependances, quoi, **options)


def verifier_trouver_cycle(f) -> None:
    # 1. Pas de cycle : le losange n'est pas un cycle.
    quoi = "trouver_cycle(pipeline_exemple())"
    obtenu = _cycle(f, pipeline_exemple(), quoi, boucle=_BOUCLE_DFS)
    if obtenu is not None:
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de None : le traitement de nuit n'a AUCUN cycle. "
                             f"`evaluer` et `scorer` ont toutes les deux besoin de `variables`, on la croise donc deux "
                             f"fois : « déjà vue » ne veut pas dire « cycle ». Il y a cycle seulement si on retombe sur une "
                             f"tâche EN COURS (sur le chemin actuel), pas sur une tâche TERMINÉE. Trois états, pas deux.")
    obtenu = _cycle(f, {}, "trouver_cycle({})")
    if obtenu is not None:
        raise AssertionError(f"trouver_cycle({{}}) renvoie {_court(obtenu)} au lieu de None.")

    # 2. Des cycles simples.
    for dependances in [{"a": ["a"]}, {"rapport": ["scorer"], "scorer": ["rapport"]},
                        {"a": ["b"], "b": ["c"], "c": ["a"], "d": ["a"]},
                        {"x": ["a"], "a": ["b"], "b": ["a"]},
                        {"a": ["b", "z"], "b": ["c"], "c": ["d"], "d": ["b"], "z": []}]:
        quoi = f"trouver_cycle({dependances})"
        _verifier_un_cycle(_cycle(f, dependances, quoi, boucle=_BOUCLE_DFS), dependances, quoi)

    # 3. Un cycle caché derrière des losanges, et un dernier cycle atteint seulement par la DERNIÈRE tâche.
    cache = _losanges(6)
    cache["a06"] = cache["a06"] + ["z_fin"]
    cache.update({"z_fin": ["z_milieu"], "z_milieu": ["z_debut"], "z_debut": ["b03", "z_fin"]})
    quoi = "trouver_cycle(<un cycle caché sous des losanges>)"
    _verifier_un_cycle(_cycle(f, cache, quoi, lignes_max=300_000, boucle=_BOUCLE_DFS), cache, quoi)
    sans_cycle = {"a": ["b"], "b": ["c"], "c": [], "d": ["a", "c"], "e": ["d", "b"]}
    obtenu = _cycle(f, sans_cycle, f"trouver_cycle({sans_cycle})")
    if obtenu is not None:
        raise AssertionError(f"trouver_cycle({sans_cycle}) renvoie {_court(obtenu)} au lieu de None : on peut arriver à "
                             f"'c' par plusieurs chemins sans qu'il y ait de cycle. « Déjà vue » ne veut pas dire « cycle ».")

    # 4. Les entrées mal formées.
    _verifier_chaine_refusee(f, "trouver_cycle")

    # 5. Des graphes tirés au hasard : sans cycle, puis avec une flèche qui en referme un.
    for graine in range(1, 41):
        rng = random.Random(graine)
        dependances = generer_pipeline(rng.randint(5, 150), graine=graine)
        quoi = f"trouver_cycle(generer_pipeline(…, graine={graine}))"
        obtenu = _cycle(f, dependances, quoi, boucle=_BOUCLE_DFS)
        if obtenu is not None:
            raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de None : ce graphe n'a pas de cycle "
                                 f"(`ordre_execution` l'accepte). « Déjà vue » ne veut pas dire « cycle ».")
        # Donner à une tâche un de ses DESCENDANTS comme prérequis ferme un cycle (ou elle-même, s'il n'y en a pas).
        source = rng.choice(_ref_ordre(dependances))
        descendants = sorted(_descendants(_graphe(dependances), [source]) - {source})
        cible = rng.choice(descendants) if descendants else source
        dependances = {t: list(p) for t, p in dependances.items()}
        dependances.setdefault(source, []).append(cible)
        quoi = f"trouver_cycle(generer_pipeline(…, graine={graine}) + la flèche {source!r} → {cible!r})"
        _verifier_un_cycle(_cycle(f, dependances, quoi, boucle=_BOUCLE_DFS), dependances, quoi)

    # 6. Les losanges : pas d'exploration exponentielle.
    quoi = "trouver_cycle(<un graphe en losanges de 30 étages>)"
    obtenu = _cycle(f, _losanges(30, sommet_en_premier=True), quoi, lignes_max=300_000, boucle=_BOUCLE_DFS)
    if obtenu is not None:
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de None. « Déjà vue » ne veut pas dire « cycle ».")

    # 7. 5 000 tâches à la file : pas de RecursionError.
    quoi = f"trouver_cycle({_CHAINE_MSG})"
    obtenu = _cycle(f, _chaine(5000), quoi)
    if obtenu is not None:
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de None.")
    boucle = _chaine(5000, cycle=True)
    quoi = f"trouver_cycle({_CHAINE_MSG} dont la dernière a besoin de la première)"
    cycle = _cycle(f, boucle, quoi)
    _verifier_un_cycle(cycle, boucle, quoi)
    if len(cycle) != 5002:
        raise AssertionError(f"{quoi} renvoie un cycle de {len(cycle) - 1} tâches au lieu de 5 001.")

    # 8. Une vraie taille.
    grand = generer_pipeline(20_000, graine=148)
    debut = time.perf_counter()
    obtenu = _resultat(f, grand, quoi="trouver_cycle(generer_pipeline(20000))", lignes_max=None)
    duree = time.perf_counter() - debut
    if obtenu is not None:
        raise AssertionError(f"trouver_cycle(generer_pipeline(20000)) renvoie {_court(obtenu)} au lieu de None.")
    if duree > 2.0:
        raise AssertionError(f"trouver_cycle est juste, mais 20 000 tâches prennent {duree:.1f} s (environ 0,05 s "
                             f"attendu). « En cours » et « terminée » se rangent dans un dictionnaire ou un `set`, pas "
                             f"dans une liste (`t in chemin` relit tout le chemin).")


# ============================================================================ Défi 149 : executer


class _Journal:
    """Les fausses tâches du correcteur : chacune note ses appels, et peut échouer ou être interrompue."""

    def __init__(self, dependances: dict, echecs=(), aucun_resultat=(), interruption=None):
        self.appels: list[tuple[str, tuple, dict]] = []
        self.valeurs: dict[str, object] = {}
        self.exceptions: dict[str, Exception] = {}
        self.taches = {}
        for nom in sorted(_graphe(dependances)):
            self.taches[nom] = self._fabriquer(nom, nom in echecs, nom in aucun_resultat, nom == interruption)

    def _fabriquer(self, nom, echoue, aucun, interrompt):
        def tache(*args, **kwargs):
            self.appels.append((nom, args, kwargs))
            if interrompt:
                raise KeyboardInterrupt
            if echoue:
                exc = RuntimeError(f"{nom} : la base ne répond pas")
                self.exceptions[nom] = exc
                raise exc
            valeur = None if aucun else {"tache": nom, "lot": len(self.appels)}
            self.valeurs[nom] = valeur
            return valeur

        tache.__name__ = nom
        return tache


def _ref_statuts(dependances: dict, echecs=()) -> dict:
    graphe, statuts = _graphe(dependances), {}
    for tache in _ref_ordre(dependances):
        if any(statuts[p] != "ok" for p in graphe[tache]):
            statuts[tache] = "sautee"
        else:
            statuts[tache] = "echec" if tache in echecs else "ok"
    return statuts


def _executer(f, journal: _Journal, dependances: dict, quoi: str):
    taches = dict(journal.taches)
    avant_t, avant_d = dict(taches), copy.deepcopy(dependances)
    retour = _resultat(f, taches, dependances, quoi=quoi)
    if taches != avant_t:
        raise AssertionError(f"{quoi} a MODIFIÉ le dictionnaire `taches` reçu. Rangez les résultats dans VOS dictionnaires.")
    _intact(avant_d, dependances, quoi)
    if retour is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return statuts, resultats, erreurs` ?")
    if not isinstance(retour, tuple) or len(retour) != 3 or not all(isinstance(d, dict) for d in retour):
        raise AssertionError(f"{quoi} renvoie {_court(retour, 100)} au lieu de trois dictionnaires : "
                             f"`return statuts, resultats, erreurs`.")
    return retour


def _verifier_execution(f, dependances: dict, quoi: str, echecs=(), aucun_resultat=()) -> None:
    journal = _Journal(dependances, echecs=echecs, aucun_resultat=aucun_resultat)
    statuts, resultats, erreurs = _executer(f, journal, dependances, quoi)
    graphe = _graphe(dependances)
    ordre = _ref_ordre(dependances)
    attendus = _ref_statuts(dependances, echecs)

    # Les appels : positionnels ? deux fois ? avec les bons paramètres ?
    for nom, args, kwargs in journal.appels:
        if args:
            raise AssertionError(f"{quoi} appelle la fonction de {nom!r} avec des paramètres POSITIONNELS {_court(args, 80)}. "
                                 f"Passez les résultats des prérequis en paramètres NOMMÉS, `f(**{{p: resultats[p] for p "
                                 f"in prerequis}})` : l'ordre des prérequis dans la liste n'a alors plus d'importance.")
    noms_appeles = [nom for nom, _, _ in journal.appels]
    doublons = sorted({n for n in noms_appeles if noms_appeles.count(n) > 1})
    if doublons:
        raise AssertionError(f"{quoi} appelle plusieurs fois {_liste(doublons)}. Chaque tâche ne tourne qu'UNE fois (si "
                             f"elle envoie des SMS, les clients les recevraient deux fois).")
    for nom, _, kwargs in journal.appels:
        if attendus.get(nom) == "sautee":
            prerequis_ko = sorted(p for p in graphe[nom] if attendus[p] != "ok")
            enfants_directs = [p for p in prerequis_ko if attendus[p] == "echec"]
            if not enfants_directs:
                raise AssertionError(f"{quoi} appelle {nom!r}, alors que son prérequis {prerequis_ko[0]!r} a été SAUTÉ. "
                                     f"Un échec se propage à toute la descendance (les petits-enfants aussi) : une tâche "
                                     f"est sautée dès qu'un de ses prérequis n'est pas « ok » (`!= \"ok\"`, pas "
                                     f"`== \"echec\"`).")
            raise AssertionError(f"{quoi} appelle {nom!r}, alors que son prérequis {enfants_directs[0]!r} a ÉCHOUÉ : elle "
                                 f"doit être sautée, sans appeler sa fonction.")
        attendu = set(graphe[nom])
        if set(kwargs) != attendu:
            raise AssertionError(f"{quoi} appelle {nom!r} avec les paramètres {sorted(kwargs)} au lieu de {sorted(attendu)} : "
                                 f"exactement les résultats de SES prérequis, un par prérequis, sous leur nom.")
        for p, valeur in kwargs.items():
            if valeur is not journal.valeurs.get(p, object()):
                raise AssertionError(f"{quoi} appelle {nom!r} avec {p}={_court(valeur, 80)} au lieu de ce qu'a renvoyé la "
                                     f"tâche {p!r} ({_court(journal.valeurs.get(p), 80)}).")
    a_appeler = [t for t in ordre if attendus[t] != "sautee"]
    if noms_appeles != a_appeler:
        manquantes = [t for t in a_appeler if t not in noms_appeles]
        if manquantes and aucun_resultat:
            raise AssertionError(f"{quoi} n'appelle pas {_liste(manquantes)}. La fonction de {_liste(sorted(aucun_resultat))} "
                                 f"a renvoyé None, mais elle n'a pas planté : elle est « ok », et ce qui en dépend tourne "
                                 f"normalement, avec None en paramètre. None est un résultat comme un autre : testez le "
                                 f"STATUT des prérequis (`statuts[p] != \"ok\"`), pas leur valeur.")
        if manquantes and echecs:
            raise AssertionError(f"{quoi} n'appelle pas {_liste(manquantes)} : ces tâches ne dépendent PAS de la tâche en "
                                 f"échec ({_liste(sorted(echecs))}). Un échec ne doit bloquer que sa descendance : on "
                                 f"continue la boucle (pas de `break`, pas de `raise`), et on saute seulement les tâches dont "
                                 f"un prérequis n'est pas « ok ».")
        if manquantes:
            raise AssertionError(f"{quoi} n'appelle pas {_liste(manquantes)}.")
        raise AssertionError(f"{quoi} appelle les tâches dans l'ordre {_liste(noms_appeles)} au lieu de celui de "
                             f"`ordre_execution` ({_liste(a_appeler)}).")

    # Les trois dictionnaires.
    if list(statuts.items()) != list(attendus.items()):
        if dict(statuts) == attendus:
            raise AssertionError(f"{quoi} : les statuts sont justes, mais pas dans l'ordre d'exécution ({_liste(statuts)} au "
                                 f"lieu de {_liste(attendus)}). Remplissez `statuts` en suivant `ordre_execution`.")
        for t in ordre:
            if statuts.get(t) != attendus[t]:
                if attendus[t] == "sautee" and statuts.get(t) == "echec":
                    ko = sorted(p for p in graphe[t] if attendus[p] != "ok")
                    raise AssertionError(f"{quoi} : le statut de {t!r} vaut 'echec' au lieu de 'sautee' : son prérequis "
                                         f"{ko[0]!r} a été {'SAUTÉ' if attendus[ko[0]] == 'sautee' else 'en échec'}, sa "
                                         f"fonction ne doit même pas être appelée. Un échec se propage à toute la "
                                         f"descendance (les petits-enfants aussi) : une tâche est sautée dès qu'un de ses "
                                         f"prérequis n'est pas « ok » (`!= \"ok\"`, pas `== \"echec\"`).")
                if attendus[t] == "ok" and t in aucun_resultat:
                    raise AssertionError(f"{quoi} : le statut de {t!r} vaut {statuts.get(t)!r} au lieu de 'ok'. Sa fonction "
                                         f"renvoie None, mais elle n'a pas planté : « ok » parle du déroulement, pas de "
                                         f"la valeur (`if resultat:` ou `if not resultats.get(p)` se trompent ici).")
                raise AssertionError(f"{quoi} : le statut de {t!r} vaut {statuts.get(t)!r} au lieu de {attendus[t]!r}.")
        raise AssertionError(f"{quoi} : statuts = {_court(statuts)} au lieu de {_court(attendus)}.")
    ok = {t for t, s in attendus.items() if s == "ok"}
    if set(resultats) != ok:
        en_trop, manquants = sorted(set(resultats) - ok), sorted(ok - set(resultats))
        if manquants and set(manquants) <= set(aucun_resultat):
            raise AssertionError(f"{quoi} : `resultats` n'a pas de clé {_liste(manquants)}. La fonction a renvoyé None, mais "
                                 f"elle a réussi : None est un résultat comme un autre (`resultats[t] = valeur`, sans `if`).")
        raise AssertionError(f"{quoi} : `resultats` doit contenir les tâches « ok », et elles seules (en trop : "
                             f"{_liste(en_trop)} ; manquantes : {_liste(manquants)}).")
    for t in ok:
        if resultats[t] is not journal.valeurs[t]:
            raise AssertionError(f"{quoi} : resultats[{t!r}] vaut {_court(resultats[t], 80)} au lieu de ce qu'a renvoyé sa "
                                 f"fonction ({_court(journal.valeurs[t], 80)}).")
    en_echec = {t for t, s in attendus.items() if s == "echec"}
    if set(erreurs) != en_echec:
        raise AssertionError(f"{quoi} : `erreurs` doit contenir les tâches en échec, et elles seules ({_liste(sorted(en_echec))}), "
                             f"pas {_liste(sorted(erreurs))}.")
    for t in en_echec:
        if erreurs[t] is not journal.exceptions[t]:
            raise AssertionError(f"{quoi} : erreurs[{t!r}] vaut {_court(erreurs[t], 80)} au lieu de l'exception levée "
                                 f"elle-même (`except Exception as exc: erreurs[t] = exc`) : son type, son message et sa "
                                 f"traceback servent à comprendre la panne.")


def _verifier_refus_avant_tout(f, dependances: dict, taches: dict, quoi: str, conseil: str) -> None:
    appels = []

    def espion(nom):
        def tache(**kwargs):
            appels.append(nom)
            return nom

        return tache

    taches = {nom: espion(nom) for nom in taches}
    _leve(f, (taches, dependances), ValueError, quoi, conseil)
    if appels:
        raise AssertionError(f"{quoi} lève bien ValueError, mais APRÈS avoir lancé {_liste(appels)}. Vérifiez tout AVANT de "
                             f"lancer la première tâche : un traitement arrêté au milieu laisse des SMS envoyés et un "
                             f"rapport à moitié refait.")


def verifier_executer(f) -> None:
    petit = {"trier": ["extraire"], "compter": ["extraire"], "rapport": ["compter"]}
    _verifier_execution(f, petit, "executer(<l'exemple de la docstring, tout réussit>)")
    _verifier_execution(f, petit, "executer(<l'exemple de la docstring>)", echecs={"compter"})
    p = pipeline_exemple
    _verifier_execution(f, p(), "executer(<le traitement de nuit, tout réussit>)")
    _verifier_execution(f, p(), "executer(<le traitement de nuit, 'entrainer' échoue>)", echecs={"entrainer"})
    _verifier_execution(f, p(), "executer(<le traitement de nuit, 'extraire_factures' échoue>)",
                        echecs={"extraire_factures"})
    _verifier_execution(f, p(), "executer(<le traitement de nuit, 'scorer' et 'alerte_support' échouent>)",
                        echecs={"scorer", "alerte_support"})
    _verifier_execution(f, p(), "executer(<le traitement de nuit, 'nettoyer' renvoie None>)", aucun_resultat={"nettoyer"})
    _verifier_execution(f, {"b": ["a", "a"], "c": ["b", "a"]}, "executer(<un prérequis écrit deux fois>)")
    _verifier_execution(f, {}, "executer({}, {})")

    # Ctrl+C : on n'attrape pas plus que `Exception`.
    journal = _Journal(p(), interruption="variables")
    quoi = "executer(<le traitement de nuit, Ctrl+C pendant 'variables'>)"
    try:
        retour, _ = _trace(f, lambda: f(dict(journal.taches), p()), 2_000_000)
    except KeyboardInterrupt:
        pass
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} ne s'arrête pas (votre fonction a été arrêtée).") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de laisser passer le KeyboardInterrupt.") from None
    else:
        raise AssertionError(f"{quoi} renvoie {_court(retour, 100)} : le Ctrl+C a été avalé, et la nuit a continué. "
                             f"Attrapez `Exception`, pas `BaseException` ni un `except:` nu : KeyboardInterrupt et "
                             f"SystemExit doivent tout arrêter.")
    apres = [nom for nom, _, _ in journal.appels][[nom for nom, _, _ in journal.appels].index("variables") + 1:]
    if apres:
        raise AssertionError(f"{quoi} : après le Ctrl+C, {_liste(apres)} ont encore été lancées. Il doit tout arrêter, "
                             f"tout de suite.")

    # Tout vérifier avant de lancer quoi que ce soit.
    taches = sorted(_graphe(p()))
    _verifier_refus_avant_tout(f, p(), [t for t in taches if t != "tableau_de_bord"],
                               "executer(<il manque la fonction de 'tableau_de_bord'>)",
                               "Une tâche du graphe sans fonction : on ne peut pas faire la nuit en entier.")
    _verifier_refus_avant_tout(f, p(), taches + ["netoyer"], "executer(<une fonction pour 'netoyer', qui n'existe pas>)",
                               "Une fonction pour une tâche absente du graphe, c'est presque toujours une faute de "
                               "frappe : la refuser évite de croire qu'elle a tourné.")
    cycle = {"a": ["b"], "b": ["a"], "c": []}
    _verifier_refus_avant_tout(f, cycle, ["a", "b", "c"], "executer(<un graphe avec un cycle>)",
                               "Appelez `ordre_execution` AVANT de lancer quoi que ce soit : il lève ValueError.")

    # Des graphes tirés au hasard, avec des pannes au hasard.
    for graine in range(1, 21):
        rng = random.Random(graine)
        dependances = generer_pipeline(rng.randint(5, 80), graine=graine)
        noms = sorted(_graphe(dependances))
        echecs = set(rng.sample(noms, min(len(noms), rng.randint(0, 3))))
        _verifier_execution(f, dependances, f"executer(<generer_pipeline(…, graine={graine}), pannes : "
                                            f"{_liste(sorted(echecs))}>)", echecs=echecs)


# ============================================================================ Défi 150 : chemin_critique


def _verifier_durees_refusees(f, quoi_f: str, *autres) -> None:
    p = pipeline_exemple()
    cas = []
    sans = durees_exemple()
    del sans["tableau_de_bord"]
    cas.append((sans, ValueError, "<une durée manquante>", "Chaque tâche du graphe a une durée."))
    en_trop = durees_exemple() | {"netoyer": 5}
    cas.append((en_trop, ValueError, "<une durée pour 'netoyer', qui n'existe pas>",
                "Une durée pour une tâche absente du graphe est une faute de frappe : refusez-la."))
    for valeur, erreur, conseil in [(-5, ValueError, "Une durée est positive ou nulle."),
                                    (float("nan"), ValueError, "NaN passe `duree < 0` (toute comparaison avec NaN est "
                                                               "fausse) : écrivez `not (duree >= 0)` ou `math.isnan`."),
                                    (float("inf"), ValueError, "Une tâche qui ne finit jamais : `math.isfinite`."),
                                    (True, TypeError, "True est un entier pour Python (True + 1 == 2) : refusez les "
                                                      "booléens AVANT de tester `numbers.Real`."),
                                    ("10", TypeError, "« 10 » est une chaîne, pas un nombre."),
                                    (None, TypeError, "None n'est pas une durée.")]:
        durees = durees_exemple()
        durees["nettoyer"] = valeur
        cas.append((durees, erreur, f"<'nettoyer' dure {valeur!r}>", conseil))
    for durees, erreur, nom, conseil in cas:
        _leve(f, (durees, p, *autres), erreur, f"{quoi_f}({nom}, pipeline_exemple())", conseil)
    cycle = {"a": ["b"], "b": ["a"]}
    _leve(f, ({"a": 1, "b": 2}, cycle, *autres), ValueError, f"{quoi_f}(…, <un graphe avec un cycle>)",
          "Appelez `ordre_execution`, qui refuse les cycles.")


def _verifier_critique(f, durees: dict, dependances: dict, quoi: str, **options) -> None:
    avant_d, avant_g = copy.deepcopy(durees), copy.deepcopy(dependances)
    retour = _resultat(f, durees, dependances, quoi=quoi, **options)
    if durees != avant_d:
        raise AssertionError(f"{quoi} a MODIFIÉ le dictionnaire `durees` reçu.")
    _intact(avant_g, dependances, quoi)
    if retour is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return total, chemin` ?")
    if not isinstance(retour, tuple) or len(retour) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(retour, 100)} au lieu d'un couple (durée totale, chemin).")
    total, chemin = retour
    attendu_total, attendu_chemin = _ref_critique(durees, dependances)
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        raise AssertionError(f"{quoi} : la durée totale vaut {_court(total, 60)} au lieu d'un nombre.")
    if not math.isclose(total, attendu_total, rel_tol=1e-9, abs_tol=1e-9):
        if math.isclose(total, sum(durees.values())):
            raise AssertionError(f"{quoi} : la durée totale vaut {total} au lieu de {attendu_total} : c'est la SOMME de "
                                 f"toutes les durées, comme si tout se faisait l'un après l'autre. Avec autant de machines "
                                 f"qu'on veut, une tâche commence dès que son DERNIER prérequis finit : "
                                 f"début = max(fin des prérequis).")
        fin_min = {}
        graphe = _graphe(dependances)
        for t in _ref_ordre(dependances):
            fin_min[t] = min((fin_min[p] for p in graphe[t]), default=0) + durees[t]
        if fin_min and math.isclose(total, max(fin_min.values())):
            raise AssertionError(f"{quoi} : la durée totale vaut {total} au lieu de {attendu_total}. Une tâche commence "
                                 f"quand TOUS ses prérequis sont finis, donc au MAXIMUM de leurs fins, pas au minimum.")
        raise AssertionError(f"{quoi} : la durée totale vaut {total} au lieu de {attendu_total}.")
    if not isinstance(chemin, list):
        raise AssertionError(f"{quoi} : le chemin vaut {_court(chemin, 80)} au lieu d'une liste de tâches.")
    if chemin == attendu_chemin:
        return
    if chemin == attendu_chemin[::-1]:
        raise AssertionError(f"{quoi} : le chemin est à l'envers ({_court(chemin)}). On le reconstruit en remontant de la "
                             f"fin vers le début : retournez-le (`chemin[::-1]`) pour qu'il aille du DÉBUT à la fin.")
    fin, precedente = _fins(durees, dependances)
    graphe = _graphe(dependances)
    if chemin and chemin[-1] != attendu_chemin[-1]:
        derniere = attendu_chemin[-1]
        egales = sorted(t for t in fin if fin[t] == fin[derniere])
        if chemin[-1] in egales:
            raise AssertionError(f"{quoi} : le chemin se termine par {chemin[-1]!r} au lieu de {derniere!r}. Elles finissent "
                                 f"toutes les deux à {fin[derniere]} : à égalité, la première par ordre alphabétique.")
        raise AssertionError(f"{quoi} : le chemin se termine par {chemin[-1]!r}, qui finit à {fin.get(chemin[-1])}, au lieu "
                             f"de {derniere!r}, qui finit à {fin[derniere]} (la plus tard).")
    for i in range(len(chemin) - 1, 0, -1):
        t, p = chemin[i], chemin[i - 1]
        attendu_p = precedente.get(t)
        if p != attendu_p:
            if p in graphe.get(t, ()) and attendu_p is not None and fin[p] == fin[attendu_p]:
                raise AssertionError(f"{quoi} : avant {t!r}, le chemin passe par {p!r} au lieu de {attendu_p!r}. Les deux "
                                     f"finissent à {fin[p]} : à égalité, la première par ordre alphabétique "
                                     f"(`min(prerequis, key=lambda p: (-fin[p], p))`).")
            if p in graphe.get(t, ()):
                raise AssertionError(f"{quoi} : avant {t!r}, le chemin passe par {p!r}, qui finit à {fin[p]}, au lieu de "
                                     f"{attendu_p!r}, qui finit à {fin[attendu_p]} : c'est le prérequis qui finit le plus "
                                     f"TARD qui décide du début de {t!r}.")
            raise AssertionError(f"{quoi} : le chemin {_court(chemin)} passe de {p!r} à {t!r}, mais {p!r} n'est pas un "
                                 f"prérequis de {t!r}.")
    raise AssertionError(f"{quoi} : le chemin vaut {_court(chemin)} au lieu de {_court(attendu_chemin)}. Il commence par "
                         f"une tâche SANS prérequis : remontez jusqu'au bout.")


def verifier_chemin_critique(f) -> None:
    _verifier_critique(f, {"a": 10, "b": 5, "c": 20}, {"c": ["a", "b"]},
                       "chemin_critique({'a': 10, 'b': 5, 'c': 20}, {'c': ['a', 'b']})")
    obtenu = _resultat(f, {}, {}, quoi="chemin_critique({}, {})")
    if obtenu is None or tuple(obtenu) != (0, []):
        raise AssertionError(f"chemin_critique({{}}, {{}}) renvoie {_court(obtenu)} au lieu de (0, []).")
    _verifier_critique(f, durees_exemple(), pipeline_exemple(), "chemin_critique(durees_exemple(), pipeline_exemple())")
    # Des égalités à la fin, des durées nulles et décimales.
    _verifier_critique(f, {"a": 5, "b": 3, "c": 2, "d": 0}, {"c": ["b"], "d": ["a"]},
                       "chemin_critique({'a': 5, 'b': 3, 'c': 2, 'd': 0}, {'c': ['b'], 'd': ['a']})")
    _verifier_critique(f, {"x": 2.5, "y": 0.5, "z": 1.25}, {"z": ["x", "y"]},
                       "chemin_critique({'x': 2.5, 'y': 0.5, 'z': 1.25}, {'z': ['x', 'y']})")
    _verifier_durees_refusees(f, "chemin_critique")
    for graine in range(1, 31):
        rng = random.Random(graine)
        dependances = generer_pipeline(rng.randint(5, 200), graine=graine)
        durees = generer_durees(dependances, graine=graine, maximum=rng.choice([3, 10, 60]))
        _verifier_critique(f, durees, dependances, f"chemin_critique(generer_durees(…), generer_pipeline(…, graine={graine}))")
    # Losanges et longues chaînes.
    losanges = _losanges(30, sommet_en_premier=True)
    _verifier_critique(f, {t: 1 for t in _graphe(losanges)}, losanges,
                       "chemin_critique(<des durées>, <un graphe en losanges de 30 étages>)", lignes_max=300_000,
                       boucle="Une fonction récursive sans mémoire recalcule la fin de chaque tâche autant de fois qu'il y "
                              "a de chemins pour y arriver (2^30). Calculez chaque fin UNE fois, dans l'ordre de "
                              "`ordre_execution`, et rangez-la dans un dictionnaire.")
    chaine = _chaine(5000)
    _verifier_critique(f, {t: 1 for t in chaine}, chaine, f"chemin_critique(<des durées>, {_CHAINE_MSG})")
    # Une vraie taille.
    grand = generer_pipeline(20_000, graine=150)
    durees = generer_durees(grand, graine=150)
    debut = time.perf_counter()
    obtenu = _resultat(f, durees, grand, quoi="chemin_critique(generer_durees(…), generer_pipeline(20000))", lignes_max=None)
    duree = time.perf_counter() - debut
    if obtenu is None or tuple(obtenu) != _ref_critique(durees, grand):
        _verifier_critique(f, durees, grand, "chemin_critique(generer_durees(…), generer_pipeline(20000))")
    if duree > 2.0:
        raise AssertionError(f"chemin_critique est juste, mais 20 000 tâches prennent {duree:.1f} s (environ 0,1 s "
                             f"attendu). Chaque tâche ne doit lire que SES prérequis, une fois.")


# ============================================================================ Défi 151 : planifier


def _verifier_planning(f, durees: dict, dependances: dict, travailleurs: int, quoi: str, **options) -> dict:
    avant_d, avant_g = copy.deepcopy(durees), copy.deepcopy(dependances)
    plan = _resultat(f, durees, dependances, travailleurs, quoi=quoi, **options)
    if durees != avant_d:
        raise AssertionError(f"{quoi} a MODIFIÉ le dictionnaire `durees` reçu.")
    _intact(avant_g, dependances, quoi)
    _expliquer_planning(plan, durees, dependances, travailleurs, quoi)
    return plan


def _expliquer_planning(plan, durees: dict, dependances: dict, travailleurs: int, quoi: str) -> None:
    if plan is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(plan, dict):
        raise AssertionError(f"{quoi} renvoie {type(plan).__name__} au lieu d'un dictionnaire {{tâche: (début, fin, "
                             f"travailleur)}}.")
    attendu = _ref_planifier(durees, dependances, travailleurs)
    graphe = _graphe(dependances)
    if set(plan) != set(attendu):
        manquantes, en_trop = sorted(set(attendu) - set(plan)), sorted(set(plan) - set(attendu))
        raise AssertionError(f"{quoi} : le planning doit contenir chaque tâche du graphe une fois (manquantes : "
                             f"{_liste(manquantes)} ; en trop : {_liste(en_trop)}).")
    for t, valeur in plan.items():
        if not isinstance(valeur, tuple) or len(valeur) != 3:
            raise AssertionError(f"{quoi} : plan[{t!r}] vaut {_court(valeur, 80)} au lieu d'un triplet (début, fin, "
                                 f"travailleur).")
    if dict(plan) == attendu:
        debuts = [debut for debut, _, _ in plan.values()]
        if debuts != sorted(debuts):
            raise AssertionError(f"{quoi} : le planning est juste, mais les tâches ne sont pas rangées dans l'ordre des "
                                 f"débuts. Ajoutez chaque tâche au dictionnaire au moment où elle COMMENCE.")
        return
    # Les versions boguées classiques.
    if any(w == 0 for _, _, w in plan.values()) and plan == _ref_planifier(durees, dependances, travailleurs, premier=0):
        raise AssertionError(f"{quoi} : les travailleurs sont numérotés de 0 à {travailleurs - 1} : la règle les numérote "
                             f"de 1 à {travailleurs} (`list(range(1, travailleurs + 1))`).")
    if plan == _ref_planifier(durees, dependances, travailleurs, priorite="alpha"):
        reste = _restes(durees, dependances)
        raise AssertionError(f"{quoi} : vous donnez la priorité à la première tâche prête par ordre ALPHABÉTIQUE. La plus "
                             f"prioritaire est celle dont le RESTE (la durée du plus long chemin qui part d'elle) est le "
                             f"plus long ; l'alphabet ne départage que les égalités. Par exemple : "
                             + ", ".join(f"reste({t}) = {reste[t]}" for t in sorted(reste, key=lambda t: -reste[t])[:3])
                             + ". Rangez des couples (-reste, nom) dans le tas.")
    if plan == _ref_planifier(durees, dependances, travailleurs, fins="une"):
        raise AssertionError(f"{quoi} : quand plusieurs tâches finissent au MÊME instant, vous attribuez du travail après la "
                             f"première, avant d'avoir terminé les autres. Une tâche prioritaire débloquée par la 2e fin "
                             f"arrive alors trop tard. Terminez TOUTES les tâches qui finissent à t, puis attribuez : "
                             f"`while en_cours and en_cours[0][0] == temps:`.")
    # Un diagnostic général : la première différence, avec l'état de la simulation à cet instant.
    rang = {t: i for i, t in enumerate(attendu)}
    ordre_ref = sorted(attendu, key=lambda t: (attendu[t][0], rang[t]))
    for t in ordre_ref:
        if plan[t] != attendu[t]:
            debut, fin, w = attendu[t]
            obtenu = plan[t]
            if len(obtenu) == 3 and obtenu[0] is not None and obtenu[1] is not None:
                try:
                    if not math.isclose(obtenu[1] - obtenu[0], durees[t]):
                        raise AssertionError(f"{quoi} : {t!r} va de {obtenu[0]} à {obtenu[1]}, alors qu'elle dure "
                                             f"{durees[t]} : fin = début + durée.")
                    for p in graphe[t]:
                        if plan[p][1] > obtenu[0]:
                            raise AssertionError(f"{quoi} : {t!r} commence à {obtenu[0]}, avant la fin de son prérequis "
                                                 f"{p!r} (à {plan[p][1]}).")
                except TypeError:
                    pass
            reste = _restes(durees, dependances)
            prets = sorted((x for x in attendu if attendu[x][0] >= debut and all(attendu[p][1] <= debut for p in graphe[x])),
                           key=lambda x: (-reste[x], x))
            raise AssertionError(
                f"{quoi} : {t!r} devrait commencer à {debut} sur le travailleur {w} (fin à {fin}), et vous la placez "
                f"{tuple(obtenu)} (début, fin, travailleur). À l'instant {debut}, les tâches prêtes, de la plus prioritaire "
                f"à la moins prioritaire, sont : "
                + ", ".join(f"{x} (reste {reste[x]})" for x in prets[:6])
                + ". Le travailleur libre de plus petit numéro prend la plus prioritaire. Vérifiez aussi que le temps "
                  "SAUTE à la prochaine fin (sans avancer d'une unité), et qu'une tâche de durée 0 libère ses "
                  "successeurs au même instant.")


_PAS_A_PAS = ("Faites-vous avancer le temps d'une unité à chaque tour ? Avec des durées en secondes, ça fait 540 000 tours "
              "pour rien, et avec des durées de 2,5 minutes, le temps passe à côté de la fin (2, puis 3) sans jamais la "
              "voir. Sautez directement à la prochaine fin de tâche (`temps = en_cours[0][0]`, le plus petit du tas des "
              "tâches en cours).")


def verifier_planifier(f) -> None:
    petit_d, petit_g = {"a": 10, "b": 5, "c": 20}, {"c": ["a", "b"]}
    _verifier_planning(f, petit_d, petit_g, 1, "planifier({'a': 10, 'b': 5, 'c': 20}, {'c': ['a', 'b']}, travailleurs=1)")
    _verifier_planning(f, petit_d, petit_g, 2, "planifier({'a': 10, 'b': 5, 'c': 20}, {'c': ['a', 'b']}, travailleurs=2)")
    for k in (2, 1, 3, 20):
        _verifier_planning(f, durees_exemple(), pipeline_exemple(), k,
                           f"planifier(durees_exemple(), pipeline_exemple(), travailleurs={k})")
    obtenu = _resultat(f, {}, {}, 2, quoi="planifier({}, {}, 2)")
    if obtenu != {}:
        raise AssertionError(f"planifier({{}}, {{}}, 2) renvoie {_court(obtenu)} au lieu de {{}}.")

    # Deux fins au même instant (a sur le travailleur 1, c sur le 2) : c débloque h, a débloque z. h et z ont le même
    # reste : h, la première par ordre alphabétique, revient au travailleur 1, à condition d'avoir terminé c AVANT
    # d'attribuer quoi que ce soit.
    meme_instant = ({"a": 5, "b": 2, "c": 3, "h": 4, "z": 4}, {"c": ["b"], "h": ["c"], "z": ["a"]})
    _verifier_planning(f, *meme_instant, 2, f"planifier({meme_instant[0]}, {meme_instant[1]}, travailleurs=2)")
    # Une tâche de durée 0 libère ses successeurs au même instant.
    zero = ({"go": 0, "long": 8, "court": 1, "autre": 3}, {"long": ["go"], "court": [], "autre": []})
    _verifier_planning(f, *zero, 1, f"planifier({zero[0]}, {zero[1]}, travailleurs=1)")
    _verifier_planning(f, *zero, 2, f"planifier({zero[0]}, {zero[1]}, travailleurs=2)")
    # Des durées décimales et des égalités de priorité.
    decimales = ({"a": 2.5, "b": 2.5, "c": 0.5, "d": 1.25}, {"d": ["a", "c"], "b": []})
    _verifier_planning(f, *decimales, 2, f"planifier({decimales[0]}, {decimales[1]}, travailleurs=2)",
                       lignes_max=100_000, boucle=_PAS_A_PAS)

    # Les entrées.
    for valeur, erreur in [(0, ValueError), (-1, ValueError), (1.5, TypeError), ("2", TypeError), (True, TypeError)]:
        _leve(f, (durees_exemple(), pipeline_exemple(), valeur), erreur,
              f"planifier(durees_exemple(), pipeline_exemple(), travailleurs={valeur!r})",
              "Un nombre de travailleurs est un entier >= 1 (et True n'en est pas un, même si True == 1).")
    _verifier_durees_refusees(f, "planifier", 2)

    # Des graphes tirés au hasard.
    for graine in range(1, 31):
        rng = random.Random(graine)
        dependances = generer_pipeline(rng.randint(5, 200), graine=graine)
        durees = generer_durees(dependances, graine=graine, maximum=rng.choice([3, 10, 60]))
        k = rng.randint(1, 6)
        _verifier_planning(f, durees, dependances, k,
                           f"planifier(generer_durees(…), generer_pipeline(…, graine={graine}), travailleurs={k})")

    # De grandes durées : on saute de fin en fin, on n'avance pas d'une unité à chaque tour.
    secondes = {t: d * 3600 for t, d in durees_exemple().items()}
    _verifier_planning(f, secondes, pipeline_exemple(), 2,
                       "planifier(<les durées en secondes, 150 heures en tout>, pipeline_exemple(), travailleurs=2)",
                       lignes_max=100_000, boucle=_PAS_A_PAS)

    # Une vraie taille.
    grand = generer_pipeline(20_000, graine=151)
    durees = generer_durees(grand, graine=151)
    debut = time.perf_counter()
    plan = _resultat(f, durees, grand, 8, quoi="planifier(generer_durees(…), generer_pipeline(20000), travailleurs=8)",
                     lignes_max=None)
    duree = time.perf_counter() - debut
    if plan != _ref_planifier(durees, grand, 8):
        _expliquer_planning(plan, durees, grand, 8, "planifier(generer_durees(…), generer_pipeline(20000), travailleurs=8)")
    if duree > 2.0:
        raise AssertionError(f"planifier est juste, mais 20 000 tâches et 8 travailleurs prennent {duree:.1f} s (environ "
                             f"0,2 s attendu). Trois tas (`heapq`) : les prêtes par priorité, les travailleurs libres par "
                             f"numéro, les tâches en cours par heure de fin. Un `min` ou un `sorted` sur une liste à chaque "
                             f"événement relit tout.")


SERIE = Serie(
    code="24",
    titre="Orchestrer le traitement de nuit : un graphe de dépendances, de l'ordre d'exécution au planning",
    module_eleve="defis.serie_24",
    module_corrige="defis.corrige_serie_24",
    defis=(
        Defi(146, "ordre_execution", "🟢", "—", verifier_ordre_execution),
        Defi(147, "taches_a_relancer", "🟢", "—", verifier_taches_a_relancer),
        Defi(148, "trouver_cycle", "🟡", "—", verifier_trouver_cycle),
        Defi(149, "executer", "🟡", "—", verifier_executer),
        Defi(150, "chemin_critique", "🟡", "—", verifier_chemin_critique),
        Defi(151, "planifier", "🔴", "—", verifier_planifier),
    ),
)

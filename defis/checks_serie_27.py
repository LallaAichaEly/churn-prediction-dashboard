"""Vérifications automatiques de la série 27 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur se comporte comme la personne qui reçoit le fichier dédoublonné et devra vivre avec :

- il compare chaque réponse à une version de référence sur les fiches d'exemple, puis sur des centaines
  de cas tirés au hasard ; quand ça ne va pas, il rejoue les erreurs classiques (distance de Hamming
  au lieu de Levenshtein, groupes non transitifs, valeur absente qui réunit tout le monde, `None == None`
  compté comme un accord, NaN « vrai », False pris pour une valeur absente, `reverse=True` et ses ex
  aequo, paires comptées deux fois…) pour vous dire LAQUELLE vous avez écrite ;
- au défi 167, il teste chaque champ SEUL (les autres absents) : il sait donc quelle règle est fausse ;
- il compte les lignes exécutées dans votre fichier, les appels à vos clés et les identifiants hachés :
  une boucle dans une boucle, une chaîne de 20 000 fiches parcourue à chaque recherche ou des paires
  fabriquées une par une se voient, quelle que soit la vitesse de la machine ;
- il vérifie que vos fonctions ne modifient pas les fiches reçues.
"""

from __future__ import annotations

import copy
import itertools
import math
import numbers
import random
import sys
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .doublons import (
    CLES,
    VERITE_EXEMPLE,
    fiches_exemple,
    generer_fiches,
    normaliser_email,
    normaliser_nom,
    normaliser_telephone,
)
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


class _BoucleSansFin(BaseException):
    """Levée dans VOTRE fonction quand elle travaille trop (BaseException : `except Exception` ne l'attrape pas)."""


def _trace(f, appel, lignes_max: int | None):
    """Le résultat de `appel()` ; arrête net au-delà de `lignes_max` lignes exécutées dans le fichier de f."""
    code = getattr(f, "__code__", None)
    if code is None or lignes_max is None:
        return appel()
    fichier, compte = code.co_filename, [0]

    def local(frame, evenement, arg):
        if evenement == "line":
            compte[0] += 1
            if compte[0] > lignes_max:
                raise _BoucleSansFin
        return local

    def global_(frame, evenement, arg):
        return local if frame.f_code.co_filename == fichier else None

    precedent = sys.gettrace()
    sys.settrace(global_)
    try:
        return appel()
    finally:
        sys.settrace(precedent)


_CONSEIL_LENT = "Votre fonction a été arrêtée : elle exécute beaucoup trop de lignes (une boucle dans une boucle ?)."


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = None, conseil_lent: str = ""):
    """f(*args), en transformant toute erreur en message lisible."""
    try:
        return _trace(f, lambda: f(*args), lignes_max)
    except NotImplementedError as exc:
        _pas_encore(f, exc)
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError : Python s'arrête vers 1 000 appels imbriqués. Remplacez "
                             f"la récursion par une boucle. {conseil}".rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None


def _pas_encore(f, exc: NotImplementedError) -> None:
    """Votre fonction est écrite, mais elle appelle une fonction qui ne l'est pas encore (le 167 utilise le 164)."""
    cadres = []
    tb = exc.__traceback__
    while tb is not None:
        cadres.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    nom = getattr(f, "__name__", "")
    if nom and nom in cadres and cadres[-1] != nom:
        numero = _NUMEROS.get(cadres[-1])
        defi = f" (défi {numero})" if numero else ""
        raise AssertionError(f"`{nom}` appelle `{cadres[-1]}`, qui n'est pas encore écrite (raise "
                             f"NotImplementedError) : faites d'abord ce défi-là{defi}.") from None


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu: type = ValueError) -> None:
    """Exige que f(*args) lève `attendu`."""
    try:
        resultat = f(*args)
    except NotImplementedError as exc:
        _pas_encore(f, exc)
        raise
    except attendu:
        return
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _entier(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _egal(a, b) -> bool:
    """==, mais NaN égal à NaN (pour comparer des fiches avant / après un appel)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_egal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_egal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


# ---------------------------------------------------------------------------- défi 164 : distance_edition


def _ref_distance(a: str, b: str) -> int:
    precedente = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        courante = [i]
        for j, y in enumerate(b, start=1):
            courante.append(min(precedente[j] + 1, courante[j - 1] + 1, precedente[j - 1] + (x != y)))
        precedente = courante
    return precedente[-1]


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def _sans_remplacement(a: str, b: str) -> int:
    """Insertions et suppressions seulement : remplacer coûte 2."""
    precedente = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        courante = [i]
        for j, y in enumerate(b, start=1):
            courante.append(min(precedente[j] + 1, courante[j - 1] + 1, precedente[j - 1] if x == y else math.inf))
        precedente = courante
    return precedente[-1]


def _damerau(a: str, b: str) -> int:
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


_EXEMPLES_DISTANCE = (
    ("", "abc"), ("abc", ""), ("", ""), ("Dupont", "Dupond"), ("chat", "chats"), ("ab", "ba"), ("kitten", "sitting"),
    ("Hélène", "Helene"), ("abc", "bc"), ("flaw", "lawn"), ("intention", "execution"), ("dupre helene", "dupre helene"),
    ("Martin", "martin"), ("abcdef", "azcdyf"), ("gumbo", "gambol"),
)

_CONSEIL_EXPONENTIEL = ("Une récursion sans mémoire recalcule les mêmes sous-problèmes : environ 3^n appels (43 "
                        "millions pour deux mots de 16 lettres). Remplissez un tableau ligne par ligne : chaque "
                        "case d[i][j] se calcule UNE fois, à partir de d[i - 1][j], d[i][j - 1] et d[i - 1][j - 1].")


def _diagnostic_distance(a: str, b: str, obtenu, attendu: int) -> str:
    cas = f"distance_edition({a!r}, {b!r}) renvoie {_court(obtenu)} au lieu de {attendu}."
    if attendu == len(b) and not a and obtenu == 0 or attendu == len(a) and not b and obtenu == 0:
        return (f"{cas} Passer de \"\" à un texte de n lettres, c'est n insertions : la première ligne du tableau "
                f"vaut 0, 1, 2, …, n (et la première colonne 0, 1, …, len(a)), pas des zéros.")
    if obtenu == _hamming(a, b):
        return (f"{cas} Vous comparez les lettres À LA MÊME POSITION (comme `zip(a, b)`) : une lettre manquante au "
                f"début décale toutes les suivantes (« abc » → « bc » : une seule suppression). Il faut le tableau "
                f"d[i][j] et ses trois voisins.")
    if obtenu == _sans_remplacement(a, b):
        return (f"{cas} Il manque le remplacement : sans lui, changer une lettre coûte 2 (supprimer, puis insérer). "
                f"Troisième candidat du min : d[i - 1][j - 1] + (0 si les lettres sont égales, 1 sinon).")
    if obtenu == _damerau(a, b) and a == "ab":
        return (f"{cas} Vous comptez l'inversion de deux lettres voisines comme UNE faute (c'est la distance de "
                f"Damerau) : ici, seules l'insertion, la suppression et le remplacement comptent.")
    if obtenu == _ref_distance(a.lower(), b.lower()) or obtenu == _ref_distance(normaliser_nom(a), normaliser_nom(b)):
        return (f"{cas} Comparez les lettres TELLES QUELLES : « é » et « e », « D » et « d » sont différentes. "
                f"Normaliser est le travail de l'appelant (défi 167), pas de la distance.")
    return f"{cas} Relisez les trois candidats du min (supprimer, insérer, remplacer) et les bords du tableau."


def verifier_distance_edition(f) -> None:
    for a, b in _EXEMPLES_DISTANCE:
        quoi = f"distance_edition({a!r}, {b!r})"
        r = _appel(f, (a, b), quoi, lignes_max=200_000, conseil_lent=_CONSEIL_EXPONENTIEL)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
        if not _entier(r):
            raise AssertionError(f"{quoi} renvoie {_court(r)} ({type(r).__name__}) : un nombre de fautes est un int.")
        attendu = _ref_distance(a, b)
        if r != attendu:
            raise AssertionError(_diagnostic_distance(a, b, r, attendu))

    a, b = "abcdefghijklmnop", "qrstuvwxyzABCDEF"
    _appel(f, (a, b), f"distance_edition({a!r}, {b!r})", lignes_max=40_000, conseil_lent=_CONSEIL_EXPONENTIEL)

    alea = random.Random(164)
    for _ in range(300):
        a = "".join(alea.choice("abcé ") for _ in range(alea.randrange(0, 9)))
        b = "".join(alea.choice("abcé ") for _ in range(alea.randrange(0, 9)))
        r = _appel(f, (a, b), f"distance_edition({a!r}, {b!r})", lignes_max=20_000, conseil_lent=_CONSEIL_EXPONENTIEL)
        attendu = _ref_distance(a, b)
        if r != attendu:
            raise AssertionError(_diagnostic_distance(a, b, r, attendu))

    a = "abc" * 366 + "ab"
    b = "abc" * 366 + "ad"
    quoi = f"distance_edition(un texte de {len(a)} lettres, le même avec sa dernière lettre changée)"
    r = _appel(f, (a, b), quoi, "Un tableau rempli avec deux boucles `for` n'a pas de limite de profondeur.")
    if r != 1:
        raise AssertionError(f"{quoi} renvoie {_court(r)} au lieu de 1.")

    for args, quoi in (((None, "Dupont"), "distance_edition(None, 'Dupont')"),
                       (("Dupont", float("nan")), "distance_edition('Dupont', nan)"),
                       ((12345, "12345"), "distance_edition(12345, '12345')")):
        _doit_lever(f, args, quoi, "Une case vide (None, NaN) ou un nombre n'est pas un texte : vérifiez avec "
                                   "`isinstance(a, str)` et levez TypeError. C'est l'appelant qui décide ce "
                                   "qu'une valeur absente veut dire.", attendu=TypeError)


# ---------------------------------------------------------------------------- défi 165 : regrouper


def _ref_regrouper(identifiants, paires) -> list[list]:
    parent = {}
    for i in identifiants:
        parent[i] = i

    def racine(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in paires:
        parent[racine(a)] = racine(b)
    groupes: dict = {}
    for i in parent:
        groupes.setdefault(racine(i), []).append(i)
    return sorted(sorted(g) for g in groupes.values())


def _sans_transitivite(identifiants, paires) -> list[list]:
    """L'erreur classique : chaque paire « rattache b au groupe de a », sans suivre les chaînes."""
    groupe = {i: i for i in identifiants}
    for a, b in paires:
        groupe[b] = groupe[a]
    groupes: dict = {}
    for i, g in groupe.items():
        groupes.setdefault(g, []).append(i)
    return sorted(sorted(g) for g in groupes.values())


def _resultat_regrouper(r, quoi: str) -> list[list]:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
    if not isinstance(r, list) or not all(isinstance(g, list) for g in r):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : il faut une LISTE de listes (chaque groupe une liste "
                             f"triée), par exemple [['F01', 'F02'], ['F03']].")
    return r


def _comparer_groupes(r: list, attendu: list, quoi: str) -> None:
    if r == attendu:
        return
    vus = [i for g in r for i in g]
    tous = {i for g in attendu for i in g}
    if set(vus) < tous:
        oublies = sorted(tous - set(vus))[:3]
        raise AssertionError(f"{quoi} oublie des fiches ({_court(oublies)}…) : une fiche qu'aucune paire ne cite est un "
                             f"client à elle seule, un groupe d'UNE fiche. Partez de TOUS les identifiants.")
    if len(vus) != len(set(vus)) or set(vus) != tous:
        raise AssertionError(f"{quoi} renvoie {_court(r)} : chaque fiche doit apparaître dans exactement un groupe "
                             f"(attendu : {_court(attendu)}).")
    if sorted(sorted(g) for g in r) == attendu:
        raise AssertionError(f"{quoi} trouve les bons groupes, mais pas dans l'ordre demandé : chaque groupe trié, "
                             f"puis les groupes triés (`sorted(sorted(g) for g in ...)`). Obtenu : {_court(r)}.")
    morceaux = {frozenset(g) for g in r}
    if len(r) > len(attendu) and all(any(m <= set(g) for g in attendu) for m in morceaux):
        raise AssertionError(f"{quoi} renvoie {_court(r)} au lieu de {_court(attendu)} : vous ne fusionnez pas assez. "
                             f"A = B et B = C, donc A = C (transitivité), quel que soit l'ordre des paires : il faut "
                             f"réunir les GROUPES entiers (union-find), pas seulement les deux fiches de la paire.")
    raise AssertionError(f"{quoi} renvoie {_court(r)} au lieu de {_court(attendu)}.")


_CONSEIL_CHAINE = ("Sur une chaîne de 20 000 fiches, vos recherches de racine parcourent toute la chaîne, ou vous "
                   "renumérotez tout un groupe à chaque paire : c'est quadratique (200 millions d'étapes). Faites "
                   "rejoindre le PETIT groupe au grand (un dictionnaire `taille`), et/ou faites pointer chaque fiche "
                   "traversée directement sur la racine (compression de chemin).")


def verifier_regrouper(f) -> None:
    cas = [
        (["F01", "F02", "F03", "F04"], [("F02", "F01"), ("F01", "F03")]),
        (["C", "B", "A", "D", "E"], [("A", "B"), ("D", "C"), ("B", "D")]),
        (["A", "B", "C", "D"], [("A", "B"), ("C", "D"), ("B", "C")]),
        (["F3", "F1", "F2"], []),
        ([], []),
        (["A", "B"], [("A", "A")]),
        ([5, 3, 9, 1, 7], [(9, 1), (3, 5), (1, 7)]),
    ]
    for identifiants, paires in cas:
        quoi = f"regrouper({identifiants!r}, {paires!r})"
        r = _resultat_regrouper(_appel(f, (list(identifiants), list(paires)), quoi), quoi)
        _comparer_groupes(r, _ref_regrouper(identifiants, paires), quoi)

    alea = random.Random(165)
    for _ in range(200):
        n = alea.randrange(1, 30)
        ids = [f"F{i:02d}" for i in range(n)]
        alea.shuffle(ids)
        paires = [tuple(alea.sample(ids, 2)) if n > 1 else (ids[0], ids[0]) for _ in range(alea.randrange(0, n + 3))]
        quoi = f"regrouper({_court(ids, 80)}, {_court(paires, 80)})"
        r = _resultat_regrouper(_appel(f, (list(ids), list(paires)), quoi), quoi)
        _comparer_groupes(r, _ref_regrouper(ids, paires), quoi)

    ids, paires = ["D", "B", "C", "A"], [("A", "B"), ("C", "B")]
    copie_ids, copie_paires = list(ids), list(paires)
    _appel(f, (ids, paires), "regrouper(liste, liste)")
    if ids != copie_ids or paires != copie_paires:
        raise AssertionError("regrouper modifie les listes reçues (un `.sort()` ou un `.pop()` ?) : l'appelant s'en sert "
                             "encore. Travaillez sur vos propres structures.")
    quoi = "regrouper(un générateur d'identifiants, un générateur de paires)"
    conseil = ("Ici, identifiants et paires sont des GÉNÉRATEURS : un générateur ne se lit qu'UNE fois (la deuxième "
               "lecture est vide). Parcourez-le une seule fois, ou faites-en d'abord une liste.")
    r = _resultat_regrouper(_appel(f, ((i for i in "EDCBA"), ((a, b) for a, b in [("A", "C"), ("E", "C")])), quoi,
                                   conseil), quoi)
    try:
        _comparer_groupes(r, [["A", "C", "E"], ["B"], ["D"]], quoi)
    except AssertionError as exc:
        raise AssertionError(f"{exc} {conseil}") from None

    _doit_lever(f, (["A", "B", "A"], []), "regrouper(['A', 'B', 'A'], [])",
                "Deux fiches avec le même identifiant, c'est une clé cassée : le dire, pas les fusionner en silence.")
    _doit_lever(f, (["A", "B"], [("A", "Z")]), "regrouper(['A', 'B'], [('A', 'Z')])",
                "Une paire qui cite une fiche inconnue (une faute de frappe, une fiche supprimée) : ValueError qui la "
                "nomme, plutôt qu'un groupe inventé ou une KeyError.")

    n = 20_000
    for nom, paires in (("(0, 1), (1, 2), …", [(i, i + 1) for i in range(n - 1)]),
                        ("(1, 0), (2, 1), …", [(i + 1, i) for i in range(n - 1)]),
                        ("toutes vers 0, puis en chaîne", [(i, 0) for i in range(1, n // 2)]
                         + [(i, i + 1) for i in range(n // 2 - 1, n - 1)])):
        quoi = f"regrouper(20 000 fiches reliées en chaîne {nom})"
        r = _appel(f, (range(n), paires), quoi, "Écrivez `racine(x)` avec une boucle `while parent[x] != x`.",
                   lignes_max=250 * n, conseil_lent=_CONSEIL_CHAINE)
        if r != [list(range(n))]:
            raise AssertionError(f"{quoi} devrait renvoyer un seul groupe de {n} fiches, triées.")


# ---------------------------------------------------------------------------- défi 166 : paires_candidates


def _ref_paires(fiches: dict, cles, taille_max: int) -> tuple[set, list]:
    paires, trop_gros = set(), []
    for position, cle in enumerate(cles):
        blocs: dict = {}
        for ident, fiche in fiches.items():
            valeur = cle(fiche)
            if valeur is None or valeur == "":
                continue
            blocs.setdefault(valeur, []).append(ident)
        for valeur, membres in blocs.items():
            if len(membres) > taille_max:
                trop_gros.append((position, valeur, len(membres)))
            else:
                paires.update(itertools.combinations(sorted(membres), 2))
    trop_gros.sort(key=lambda t: (-t[2], t[0], t[1]))
    return paires, trop_gros


class _Espion:
    """Une clé de blocage qui compte ses appels."""

    def __init__(self, cle):
        self.cle, self.appels = cle, 0
        self.__name__ = getattr(cle, "__name__", "cle")

    def __call__(self, fiche):
        self.appels += 1
        return self.cle(fiche)


def _resultat_paires(r, quoi: str) -> tuple[set, list]:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
    if not isinstance(r, tuple) or len(r) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(r)} : il faut un tuple `(paires, trop_gros)`.")
    paires, trop_gros = r
    if not isinstance(paires, (set, frozenset)):
        raise AssertionError(f"{quoi} : `paires` est un {type(paires).__name__} ; il faut un `set` de tuples (a, b), "
                             f"qui garde automatiquement une seule fois la paire que deux clés proposent.")
    if not isinstance(trop_gros, list):
        raise AssertionError(f"{quoi} : `trop_gros` est un {type(trop_gros).__name__} ; il faut une liste de tuples "
                             f"(position de la clé, valeur, nombre de fiches).")
    for p in paires:
        if not isinstance(p, tuple) or len(p) != 2:
            raise AssertionError(f"{quoi} : {_court(p)} n'est pas une paire : chaque élément est un tuple (a, b).")
        if not p[0] < p[1]:
            raise AssertionError(f"{quoi} contient la paire {p!r} : écrivez chaque paire (a, b) avec a < b (sinon (F02, "
                                 f"F01) et (F01, F02) comptent deux fois, et (F01, F01) n'est pas une paire). "
                                 f"`itertools.combinations(sorted(membres), 2)` les donne dans l'ordre.")
    return paires, trop_gros


def _comparer_paires(r, attendu, quoi: str, conseil: str = "") -> None:
    paires, trop_gros = r
    paires_attendues, trop_gros_attendus = attendu
    en_trop = sorted(paires - paires_attendues)
    manquent = sorted(paires_attendues - paires)
    if en_trop or manquent:
        detail = []
        if en_trop:
            detail.append(f"{len(en_trop)} paire(s) en trop, par exemple {en_trop[0]!r}")
        if manquent:
            detail.append(f"{len(manquent)} paire(s) manquante(s), par exemple {manquent[0]!r}")
        raise AssertionError(f"{quoi} : {', '.join(detail)}. {conseil}".rstrip())
    if trop_gros != trop_gros_attendus:
        if sorted(trop_gros, key=repr) == sorted(trop_gros_attendus, key=repr):
            raise AssertionError(f"{quoi} : les bons blocs dans `trop_gros`, mais pas dans le bon ordre : du plus gros "
                                 f"au plus petit, puis par position de clé, puis par valeur "
                                 f"(`sort(key=lambda t: (-t[2], t[0], t[1]))`). Attendu : {_court(trop_gros_attendus)}.")
        raise AssertionError(f"{quoi} : trop_gros = {_court(trop_gros)} au lieu de {_court(trop_gros_attendus)} (chaque "
                             f"bloc sauté : (position de la clé dans `cles`, valeur, nombre de fiches)). {conseil}".rstrip())


def verifier_paires_candidates(f) -> None:
    # Une clé absente ne réunit personne.
    sans = {f"S{i}": {"telephone": t} for i, t in enumerate([None, "", None, "", "0612345678", "+33612345678"])}
    quoi = "paires_candidates(6 fiches dont 4 sans téléphone (None ou \"\"), cles=[cle_telephone, lambda f: \"\"])"
    cles = [CLES[1], lambda fiche: ""]
    r = _resultat_paires(_appel(f, (sans, cles, 50), quoi), quoi)
    if ("S0", "S1") in r[0] or ("S0", "S2") in r[0]:
        raise AssertionError(f"{quoi} réunit des fiches SANS téléphone : une clé None ou \"\" ne met la fiche dans aucun "
                             f"bloc. Sinon, le « bloc » des fiches sans téléphone réunit la moitié du fichier (100 000 "
                             f"fiches, 5 milliards de paires).")
    _comparer_paires(r, _ref_paires(sans, cles, 50), quoi)

    # Deux clés différentes qui donnent la même valeur ne font pas un bloc.
    croises = {"C": {"x": "c", "y": "a"}, "A": {"x": "75011", "y": "a"}, "B": {"x": "b", "y": "75011"}}
    cles = [lambda fiche: fiche["x"], lambda fiche: fiche["y"]]
    quoi = ("paires_candidates({'C': {'x': 'c', 'y': 'a'}, 'A': {'x': '75011', 'y': 'a'}, 'B': {'x': 'b', 'y': '75011'}}, "
            "cles=[fiche['x'], fiche['y']])")
    r = _resultat_paires(_appel(f, (croises, cles, 50), quoi), quoi)
    if ("A", "B") in r[0]:
        raise AssertionError(f"{quoi} propose ('A', 'B') : « 75011 » est la valeur de la PREMIÈRE clé pour A et de la "
                             f"DEUXIÈME pour B. Ce n'est pas un bloc : utilisez un dictionnaire de blocs PAR clé (ou "
                             f"la clé (position, valeur)).")
    _comparer_paires(r, ({("A", "C")}, []), quoi)

    # Taille limite, ordre de trop_gros.
    blocs = {}
    for numero, (x, y) in enumerate([("p", "u")] * 3 + [("q", "u")] * 4 + [("r", "v")] * 4 + [("s", "u")] * 2):
        blocs[f"G{numero:02d}"] = {"x": x, "y": y}
    quoi = "paires_candidates(blocs de 3, 4, 4 et 2 fiches pour la clé 0, de 9 et 4 pour la clé 1, taille_max=3)"
    r = _resultat_paires(_appel(f, (blocs, cles, 3), quoi), quoi)
    attendu = _ref_paires(blocs, cles, 3)
    if any(t[2] == 3 for t in r[1]):
        raise AssertionError(f"{quoi} saute un bloc de 3 fiches : seul un bloc de PLUS de taille_max fiches est sauté "
                             f"(`len(membres) > taille_max`).")
    _comparer_paires(r, attendu, quoi)

    for taille_max in (50, 2):
        quoi = f"paires_candidates(fiches_exemple(), taille_max={taille_max})"
        r = _resultat_paires(_appel(f, (fiches_exemple(), CLES, taille_max), quoi), quoi)
        _comparer_paires(r, _ref_paires(fiches_exemple(), CLES, taille_max), quoi,
                         "Un bloc de PLUS de taille_max fiches (strictement) est sauté et signalé." if taille_max == 2
                         else "Deux fiches sont candidates si UNE des clés leur donne la même valeur (et pas None ni \"\").")


    # Chaque clé une fois par fiche.
    fiches, _ = generer_fiches(150, graine=166)
    espions = [_Espion(cle) for cle in CLES]
    quoi = "paires_candidates(150 clients tirés au hasard, taille_max=4)"
    r = _resultat_paires(_appel(f, (fiches, espions, 4), quoi), quoi)
    for position, espion in enumerate(espions):
        if espion.appels != len(fiches):
            raise AssertionError(f"{quoi} appelle la clé n°{position} ({espion.__name__}) {espion.appels} fois pour "
                                 f"{len(fiches)} fiches : calculez chaque clé UNE fois par fiche (un dictionnaire "
                                 f"valeur → fiches), jamais dans une boucle sur les paires.")
    _comparer_paires(r, _ref_paires(fiches, CLES, 4), quoi)

    # Passer à l'échelle.
    fiches, _ = generer_fiches(1500, graine=27)
    for ident in list(fiches)[::7]:
        fiches[ident] = dict(fiches[ident], telephone="01 00 00 00 00")  # le centre d'appels
    quoi = f"paires_candidates({len(fiches)} fiches, dont {len(range(0, len(fiches), 7))} avec le numéro du centre d'appels)"
    r = _resultat_paires(_appel(f, (fiches, CLES, 50), quoi, lignes_max=100 * len(fiches),
                                conseil_lent="Ne comparez pas chaque fiche à chaque autre (n² / 2 = 2 millions de paires "
                                             "ici, 45 milliards pour 300 000 fiches) : rangez les fiches dans un "
                                             "dictionnaire valeur → liste de fiches, puis ne formez des paires "
                                             "qu'à l'intérieur de chaque liste."), quoi)
    _comparer_paires(r, _ref_paires(fiches, CLES, 50), quoi)

    for taille in (1, 0, 2.5, True):
        _doit_lever(f, (fiches_exemple(), CLES, taille), f"paires_candidates(fiches, taille_max={taille!r})",
                    "taille_max : un entier d'au moins 2 (True est un bool, même si Python le compte comme 1).")


# ---------------------------------------------------------------------------- défi 167 : score_correspondance


def _texte(valeur) -> str:
    if valeur is None:
        return ""
    if not isinstance(valeur, str):
        try:
            if pd.isna(valeur):
                return ""
        except (TypeError, ValueError):
            pass
    return str(valeur).strip()


def _mots(nom) -> str:
    return " ".join(sorted(normaliser_nom(nom).split()))


def _ref_score(a: dict, b: dict) -> int:
    score = 0
    ea, eb = normaliser_email(a.get("email")), normaliser_email(b.get("email"))
    if ea and eb:
        score += 4 if ea == eb else -1
    ta, tb = normaliser_telephone(a.get("telephone")), normaliser_telephone(b.get("telephone"))
    if ta and tb:
        score += 3 if ta == tb else -1
    na, nb = _texte(a.get("naissance")), _texte(b.get("naissance"))
    if na and nb:
        score += 3 if na == nb else -8
    ma, mb = _mots(a.get("nom")), _mots(b.get("nom"))
    if ma and mb:
        score += 3 if ma == mb else 1 if _ref_distance(ma, mb) <= 2 else -4
    ca, cb = _texte(a.get("code_postal")), _texte(b.get("code_postal"))
    if ca and cb and ca == cb:
        score += 1
    return score


_NAN = float("nan")
_ABSENT = object()  # le champ n'existe pas du tout dans la fiche

_SANS = "Une valeur absente (None, \"\", espaces, NaN, ou une clé qui n'existe pas) ne compte PAS : 0 point."
_DEUX_ABSENTS = ("Quand aucune des deux fiches ne connaît ce champ ({champ}), elles n'ont rien à comparer : 0 point, ni "
                 "accord ni désaccord. Avec `va == vb`, \"\" == \"\" et None == None donnent un accord à toutes les "
                 "fiches incomplètes. Commencez chaque règle par `if va and vb:`.")
_NAN_VRAI = ("Un NaN (une case vide lue par pandas) est une valeur ABSENTE. Attention : `bool(float('nan'))` vaut True "
             "(il passe un `if`), `nan == nan` vaut False et `str(nan)` vaut 'nan'. Testez-le avec `pd.isna(v)`.")
_NORMALISER = "Comparez les valeurs NORMALISÉES (normaliser_email, normaliser_telephone, normaliser_nom), pas le texte brut."

# (champ, valeur a, valeur b, points attendus, conseil si faux)
_CAS_CHAMPS = (
    ("email", "helene.dupre@exemple.fr", "helene.dupre@exemple.fr", 4, ""),
    ("email", " Helene.Dupre@Exemple.FR ", "helene.dupre@exemple.fr", 4, _NORMALISER),
    ("email", "helene.dupre@exemple.fr", "h.dupre@exemple.fr", -1, ""),
    ("email", None, None, 0, _DEUX_ABSENTS.format(champ="e-mail")),
    ("email", "", "", 0, _DEUX_ABSENTS.format(champ="e-mail")),
    ("email", _ABSENT, _ABSENT, 0, _DEUX_ABSENTS.format(champ="e-mail")),
    ("email", "helene.dupre@exemple.fr", None, 0, _SANS),
    ("email", "helene.dupre@exemple.fr", "   ", 0, _SANS),
    ("email", "helene.dupre@exemple.fr", _ABSENT, 0, _SANS + " Lisez la fiche avec `fiche.get('email')`."),
    ("email", _NAN, _NAN, 0, _NAN_VRAI),
    ("telephone", "06 12 34 56 78", "06 12 34 56 78", 3, ""),
    ("telephone", "06 12 34 56 78", "+33 6 12 34 56 78", 3, _NORMALISER),
    ("telephone", "0612345678", "6 12 34 56 78", 3, _NORMALISER),
    ("telephone", "06 12 34 56 78", "07 98 76 54 32", -1, ""),
    ("telephone", None, None, 0, _DEUX_ABSENTS.format(champ="téléphone")),
    ("telephone", "", "", 0, _DEUX_ABSENTS.format(champ="téléphone")),
    ("telephone", "06 12", "06 12", 0, "« 06 12 » ne se lit pas : normaliser_telephone donne \"\", c'est une valeur "
                                       "ABSENTE (ni accord ni désaccord). Comparez les numéros normalisés."),
    ("telephone", "06 12 34 56 78", "inconnu", 0, "Un numéro illisible (normaliser_telephone donne \"\") ne contredit "
                                                  "rien : c'est une valeur absente."),
    ("telephone", _NAN, "06 12 34 56 78", 0, _NAN_VRAI),
    ("naissance", "1984-03-12", "1984-03-12", 3, ""),
    ("naissance", "1984-03-12", " 1984-03-12 ", 3, "Retirez les espaces autour : `str(v).strip()`."),
    ("naissance", date(1984, 3, 12), "1984-03-12", 3, "Une date (`datetime.date`) et son texte ISO sont la même date : "
                                                      "comparez `str(v).strip()`."),
    ("naissance", "1984-03-12", "1962-11-02", -8, "Deux dates de naissance différentes : -8 (c'est presque toujours "
                                                  "deux personnes, un parent et son enfant)."),
    ("naissance", None, None, 0, _DEUX_ABSENTS.format(champ="date de naissance")),
    ("naissance", "1984-03-12", None, 0, _SANS),
    ("naissance", _NAN, "1984-03-12", 0, _NAN_VRAI),
    ("naissance", _NAN, _NAN, 0, _NAN_VRAI),
    ("naissance", pd.NaT, pd.NaT, 0, _NAN_VRAI + " (pd.NaT est le NaN des dates.)"),
    ("naissance", "", "  ", 0, _DEUX_ABSENTS.format(champ="date de naissance")),
    ("nom", "Hélène Dupré", "Hélène Dupré", 3, ""),
    ("nom", "Hélène Dupré", "helene dupre", 3, _NORMALISER),
    ("nom", "Hélène Dupré", "DUPRE Helene", 3, "Un nom écrit à l'envers (« DUPRE Helene », en boutique) est le même "
                                              "nom : comparez les mots TRIÉS, `' '.join(sorted(normaliser_nom(n).split()))`."),
    ("nom", "Jean-Pierre Dupont", "Jean Pierre Dupont", 3, _NORMALISER),
    ("nom", "Jean Dupont", "Jean Dupond", 1, "Une ou deux fautes de frappe (distance_edition <= 2 sur les mots triés) : +1."),
    ("nom", "Jean Dupont", "Jean Dupontel", 1, "Distance de 2 exactement : c'est encore +1 (`<= 2`, pas `< 2`)."),
    ("nom", "Jean Dupont", "Jean Dupontelle", -4, "Distance de 4 : ce ne sont pas des fautes de frappe, -4."),
    ("nom", "Paul Martin", "Claire Martin", -4, "Deux prénoms différents dans la même famille : -4."),
    ("nom", "Lea Martin", "Martin Lena", 1, "Les mots triés (« lea martin » / « lena martin ») sont à une faute près : "
                                            "+1. Calculez la distance sur les mots TRIÉS, pas sur le texte brut."),
    ("nom", None, None, 0, _DEUX_ABSENTS.format(champ="nom")),
    ("nom", "Hélène Dupré", "   ", 0, _SANS),
    ("nom", _NAN, "Hélène Dupré", 0, _NAN_VRAI),
    ("code_postal", "75011", "75011", 1, ""),
    ("code_postal", "75011", " 75011", 1, "Retirez les espaces autour : `str(v).strip()`."),
    ("code_postal", 75011, "75011", 1, "Un code postal lu comme un nombre (75011) et le texte « 75011 » : comparez "
                                       "`str(v).strip()`."),
    ("code_postal", "75011", "75012", 0, "Deux codes postaux différents : 0 (on déménage), pas un désaccord."),
    ("code_postal", None, None, 0, _DEUX_ABSENTS.format(champ="code postal")),
    ("code_postal", _NAN, _NAN, 0, _NAN_VRAI),
)


def _fiche_un_champ(champ: str, valeur) -> dict:
    return {} if valeur is _ABSENT else {champ: valeur}


def _montrer(v) -> str:
    return "(champ absent)" if v is _ABSENT else _court(v, 40)


def _score(f, a: dict, b: dict, quoi: str) -> int:
    r = _appel(f, (a, b), quoi, "Un champ peut manquer dans la fiche (lisez-le avec `fiche.get(...)`), ou valoir None "
                                "ou NaN : normaliser_xxx(None) et normaliser_xxx(nan) donnent \"\".")
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
    if not _entier(r):
        raise AssertionError(f"{quoi} renvoie {_court(r)} ({type(r).__name__}) : le score est un int (une somme de points).")
    return int(r)


def verifier_score_correspondance(f) -> None:
    for champ, va, vb, attendu, conseil in _CAS_CHAMPS:
        a, b = _fiche_un_champ(champ, va), _fiche_un_champ(champ, vb)
        for x, y in ((a, b), (b, a)):
            quoi = f"score_correspondance({_court(x, 70)}, {_court(y, 70)})"
            r = _score(f, copy.deepcopy(x), copy.deepcopy(y), quoi)
            if r != attendu:
                raise AssertionError(f"{champ} : {_montrer(x.get(champ, _ABSENT))} et {_montrer(y.get(champ, _ABSENT))} "
                                     f"(tous les autres champs absents) donnent {r} au lieu de "
                                     f"{f'{attendu:+d}' if attendu else '0'}. "
                                     f"{conseil}".rstrip())

    fiches = fiches_exemple()
    for ia, ib in itertools.combinations(sorted(fiches), 2):
        a, b = fiches[ia], fiches[ib]
        avant_a, avant_b = copy.deepcopy(a), copy.deepcopy(b)
        quoi = f"score_correspondance(fiches['{ia}'], fiches['{ib}'])"
        r = _score(f, a, b, quoi)
        if not (_egal(a, avant_a) and _egal(b, avant_b)):
            raise AssertionError(f"{quoi} modifie les fiches reçues : normalisez dans des variables à vous, sans écrire "
                                 f"dans la fiche.")
        attendu = _ref_score(a, b)
        if r != attendu:
            raise AssertionError(f"{quoi} vaut {r} au lieu de {attendu}. Reprenez champ par champ : "
                                 f"{_court(a, 220)} / {_court(b, 220)}.")
        retour = _score(f, b, a, f"score_correspondance(fiches['{ib}'], fiches['{ia}'])")
        if not (_egal(a, avant_a) and _egal(b, avant_b)):
            raise AssertionError(f"score_correspondance(fiches['{ib}'], fiches['{ia}']) modifie les fiches reçues : "
                                 f"normalisez dans des variables à vous, sans écrire dans la fiche.")
        if retour != r:
            raise AssertionError(f"{quoi} vaut {r} mais score_correspondance(fiches['{ib}'], fiches['{ia}']) vaut "
                                 f"{retour} : le score doit être le même dans les deux sens.")

    fiches, verite = generer_fiches(400, graine=167)
    alea = random.Random(167)
    ids = sorted(fiches)
    paires = [tuple(g[:2]) for g in verite if len(g) > 1][:150] + [tuple(alea.sample(ids, 2)) for _ in range(150)]
    for ia, ib in paires:
        quoi = f"score_correspondance(fiches['{ia}'], fiches['{ib}']) (fiches tirées au hasard)"
        r = _score(f, fiches[ia], fiches[ib], quoi)
        attendu = _ref_score(fiches[ia], fiches[ib])
        if r != attendu:
            raise AssertionError(f"{quoi} vaut {r} au lieu de {attendu} : {_court(fiches[ia], 220)} / "
                                 f"{_court(fiches[ib], 220)}.")


# ---------------------------------------------------------------------------- défi 168 : fiche_de_reference


def _absent(valeur) -> bool:
    return _texte(valeur) == ""


def _ref_fiche(fiches: list) -> dict:
    ordre = sorted(range(len(fiches)), key=lambda i: (fiches[i]["maj"], i))
    recentes = [fiches[i] for i in reversed(ordre)]
    ids = sorted(f["id"] for f in fiches)
    resultat = {"id": ids[0], "anciens_ids": ids, "maj": max(f["maj"] for f in fiches)}
    inscriptions = [f["inscription"] for f in fiches if not _absent(f.get("inscription"))]
    resultat["inscription"] = min(inscriptions) if inscriptions else None
    resultat["resilie"] = all(not _absent(f.get("resilie")) and bool(f["resilie"]) for f in fiches)
    for champ in dict.fromkeys(c for f in fiches for c in f):
        if champ not in resultat:
            resultat[champ] = next((f[champ] for f in recentes if champ in f and not _absent(f[champ])), None)
    return resultat


_J = date(2024, 1, 1)


def _j(n: int) -> date:
    return _J + timedelta(days=n)


_PLUS_RECENTE_CONNUE = ("Chaque champ prend la valeur de la fiche la plus récente où il est CONNU, pas toute la fiche la "
                        "plus récente : une fiche du support sans e-mail ne doit pas effacer l'e-mail du site.")
_TRI = ("Rangez les fiches par date `maj` (la plus récente gagne), pas dans l'ordre de la liste : "
        "`sorted(range(len(fiches)), key=lambda i: (fiches[i]['maj'], i))`.")
_EX_AEQUO = ("À date égale, c'est la fiche qui vient en DERNIER dans la liste qui gagne. Piège : avec "
             "`sorted(fiches, key=..., reverse=True)`, les ex aequo GARDENT leur ordre d'origine (le tri est stable), "
             "donc la première des deux passe devant. Triez dans l'ordre croissant puis parcourez `reversed(...)`.")

# (description, fiches, conseil si faux)
_CAS_FICHES = (
    ("la fiche la plus récente n'a pas d'e-mail (None)",
     [{"id": "B", "maj": _j(1), "email": "a@x.fr"}, {"id": "A", "maj": _j(5), "email": None}], _PLUS_RECENTE_CONNUE),
    ("la fiche la plus récente a un e-mail vide (\"   \")",
     [{"id": "A", "maj": _j(1), "email": "a@x.fr"}, {"id": "B", "maj": _j(5), "email": "   "}], _PLUS_RECENTE_CONNUE),
    ("la fiche la plus récente n'a pas du tout de champ « email »",
     [{"id": "A", "maj": _j(1), "email": "a@x.fr"}, {"id": "B", "maj": _j(5)}], _PLUS_RECENTE_CONNUE),
    ("la fiche la plus récente a une naissance NaN (case vide lue par pandas)",
     [{"id": "A", "maj": _j(1), "naissance": "1984-03-12"}, {"id": "B", "maj": _j(5), "naissance": _NAN}],
     "NaN est une valeur ABSENTE (`pd.isna(v)`) : il passe un `if v:` parce que `bool(float('nan'))` vaut True."),
    ("la fiche la plus récente dit optin_sms = False (la cliente a REFUSÉ les SMS), une plus ancienne True",
     [{"id": "A", "maj": _j(1), "optin_sms": True}, {"id": "B", "maj": _j(5), "optin_sms": False}],
     "False est une RÉPONSE, pas une valeur absente : avec `if valeur:`, vous réactivez les SMS d'une cliente qui les a "
     "refusés (c'est interdit, et la CNIL met des amendes). Ne traitez comme absents que None, NaN et les textes vides."),
    ("la fiche la plus récente dit enfants = 0, une plus ancienne 2",
     [{"id": "A", "maj": _j(1), "enfants": 2}, {"id": "B", "maj": _j(5), "enfants": 0}],
     "0 est une valeur CONNUE, pas une valeur absente : n'utilisez pas `if valeur:`."),
    ("les fiches ne sont pas dans l'ordre des dates (la plus récente est la première de la liste)",
     [{"id": "B", "maj": _j(9), "email": "recent@x.fr"}, {"id": "A", "maj": _j(1), "email": "ancien@x.fr"},
      {"id": "C", "maj": _j(4), "email": "milieu@x.fr"}], _TRI),
    ("deux fiches ont la même date maj",
     [{"id": "A", "maj": _j(1), "email": "a@x.fr"}, {"id": "B", "maj": _j(3), "email": "premiere@x.fr"},
      {"id": "C", "maj": _j(3), "email": "derniere@x.fr"}], _EX_AEQUO),
    ("trois fiches, inscrites en 2016, 2019 et 2021",
     [{"id": "F03", "maj": _j(9), "inscription": date(2021, 11, 30)}, {"id": "F01", "maj": _j(5), "inscription":
      date(2019, 5, 2)}, {"id": "F02", "maj": _j(1), "inscription": date(2016, 9, 14)}],
     "L'inscription est la PLUS ANCIENNE date connue (`min`) : l'ancienneté compte depuis le premier contact. Avec la "
     "fiche la plus récente, une cliente de 2016 devient une « nouvelle cliente » de 2021, et le modèle la croit fragile."),
    ("une seule des fiches connaît l'inscription, les autres ont None ou NaN",
     [{"id": "A", "maj": _j(1), "inscription": None}, {"id": "B", "maj": _j(2), "inscription": date(2020, 1, 1)},
      {"id": "C", "maj": _j(3), "inscription": _NAN}],
     "Ignorez les inscriptions absentes (None, NaN) avant de prendre le `min`."),
    ("aucune fiche ne connaît l'inscription",
     [{"id": "A", "maj": _j(1)}, {"id": "B", "maj": _j(2), "inscription": None}],
     "Si aucune fiche ne connaît l'inscription : None."),
    ("la fiche boutique est résiliée, la fiche site ne l'est pas",
     [{"id": "A", "maj": _j(1), "resilie": False}, {"id": "B", "maj": _j(5), "resilie": True}],
     "Le client n'est parti que si TOUTES ses fiches sont résiliées (`all`, pas `any`, ni la plus récente) : quitter la "
     "boutique en restant client du site n'est pas partir. Sinon, le modèle apprend sur de faux départs."),
    ("toutes les fiches sont résiliées",
     [{"id": "A", "maj": _j(1), "resilie": True}, {"id": "B", "maj": _j(5), "resilie": 1}],
     "Toutes les fiches résiliées (True ou 1) : resilie = True."),
    ("deux fiches résiliées, une où l'information manque",
     [{"id": "A", "maj": _j(1), "resilie": True}, {"id": "B", "maj": _j(5), "resilie": None}, {"id": "C", "maj": _j(3),
                                                                                             "resilie": True}],
     "Une fiche où l'information de résiliation manque n'est PAS résiliée : resilie = False."),
    ("les identifiants arrivent dans le désordre",
     [{"id": "F14", "maj": _j(1)}, {"id": "F02", "maj": _j(5)}, {"id": "F09", "maj": _j(3)}],
     "\"id\" est le plus PETIT identifiant (`min`), et \"anciens_ids\" la liste TRIÉE de tous."),
    ("un champ n'existe que dans la plus ancienne fiche",
     [{"id": "A", "maj": _j(1), "commentaire": "rappeler le soir"}, {"id": "B", "maj": _j(5), "email": "b@x.fr"}],
     "Le résultat contient TOUS les champs présents dans au moins une fiche."),
    ("aucune fiche ne connaît le téléphone (None, NaN, \"\")",
     [{"id": "A", "maj": _j(1), "telephone": None}, {"id": "B", "maj": _j(5), "telephone": _NAN},
      {"id": "C", "maj": _j(3), "telephone": ""}],
     "Un champ qu'aucune fiche ne connaît vaut None dans le résultat (ni NaN, ni \"\")."),
    ("une seule fiche", [{"id": "A", "maj": _j(1), "email": "a@x.fr", "resilie": True, "inscription": _j(-100)}],
     "Avec une seule fiche, la fiche de référence est cette fiche (plus anciens_ids = [son id])."),
)


def _resultat_fiche(r, quoi: str) -> dict:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : il faut un dictionnaire.")
    return r


_CONSEILS_CALCULES = {
    "id": "\"id\" est le plus PETIT identifiant (`min`), pas celui de la première fiche de la liste.",
    "anciens_ids": "\"anciens_ids\" : la liste TRIÉE de tous les identifiants (`sorted(...)`, une liste).",
    "maj": "\"maj\" : la date de mise à jour la plus récente (`max`).",
    "inscription": ("L'inscription est la PLUS ANCIENNE date connue (`min` des inscriptions qui ne sont ni None ni NaN ; "
                    "None si aucune) : l'ancienneté compte depuis le premier contact. Avec la fiche la plus récente, une "
                    "cliente de 2016 devient une « nouvelle cliente » de 2021, et le modèle la croit fragile."),
    "resilie": ("Le client n'est parti que si TOUTES ses fiches sont résiliées (`all`, pas `any`, ni la plus récente ; une "
                "fiche où l'information manque n'est pas résiliée) : quitter la boutique en restant client du site "
                "n'est pas partir. Sinon, le modèle apprend sur de faux départs."),
}


def _comparer_fiche(r: dict, attendu: dict, quoi: str, conseil: str) -> None:
    for champ in attendu:
        conseil_champ = _CONSEILS_CALCULES.get(champ, conseil)
        if champ not in r:
            raise AssertionError(f"{quoi} : le résultat n'a pas de champ {champ!r}. {conseil_champ}")
        if not _egal(r[champ], attendu[champ]):
            if champ == "resilie" and r[champ] == attendu[champ]:
                raise AssertionError(f"{quoi} : resilie vaut {r[champ]!r} ({type(r[champ]).__name__}) ; renvoyez un "
                                     f"bool (True ou False) : `all(...)` en donne un.")
            raise AssertionError(f"{quoi} : {champ} = {_court(r[champ], 60)} au lieu de {_court(attendu[champ], 60)}. "
                                 f"{conseil_champ}")
    en_trop = [c for c in r if c not in attendu]
    if en_trop:
        raise AssertionError(f"{quoi} : champ(s) en trop {en_trop!r}. Le résultat a les champs des fiches, plus "
                             f"\"anciens_ids\" (et \"inscription\", \"resilie\").")


def verifier_fiche_de_reference(f) -> None:
    for description, fiches, conseil in _CAS_FICHES:
        quoi = f"fiche_de_reference(…) quand {description}"
        r = _resultat_fiche(_appel(f, (copy.deepcopy(fiches),), quoi), quoi)
        _comparer_fiche(r, _ref_fiche(fiches), quoi, conseil)

    exemple = fiches_exemple()
    for groupe in VERITE_EXEMPLE:
        fiches = [{"id": i, **exemple[i]} for i in groupe]
        quoi = f"fiche_de_reference(les fiches {', '.join(groupe)} de fiches_exemple())"
        avant = copy.deepcopy(fiches)
        r = _resultat_fiche(_appel(f, (fiches,), quoi), quoi)
        if not _egal(fiches, avant):
            raise AssertionError(f"{quoi} modifie les fiches reçues : construisez un NOUVEAU dictionnaire (pas de "
                                 f"`fiche.update(...)`, de `pop` ou de `fiches.sort()` chez l'appelant).")
        if any(r is fiche for fiche in fiches):
            raise AssertionError(f"{quoi} renvoie l'une des fiches reçues : construisez un NOUVEAU dictionnaire.")
        _comparer_fiche(r, _ref_fiche(fiches), quoi, "Relisez les règles champ par champ.")

    fiches, verite = generer_fiches(300, graine=168)
    for groupe in verite[:200]:
        liste = [{"id": i, **fiches[i]} for i in groupe]
        random.Random(len(groupe)).shuffle(liste)
        quoi = f"fiche_de_reference(les fiches {', '.join(groupe)} de generer_fiches(300, graine=168))"
        r = _resultat_fiche(_appel(f, (liste,), quoi), quoi)
        _comparer_fiche(r, _ref_fiche(liste), quoi, "Relisez les règles champ par champ.")

    quoi = "fiche_de_reference(un générateur de fiches)"
    gen = ({"id": i, "maj": _j(i)} for i in (3, 1, 2))
    r = _resultat_fiche(_appel(f, (gen,), quoi, "Un générateur ne se lit qu'une fois : `fiches = list(fiches)` d'abord."),
                        quoi)
    _comparer_fiche(r, _ref_fiche([{"id": i, "maj": _j(i)} for i in (3, 1, 2)]), quoi,
                    "Un générateur ne se lit qu'une fois : `fiches = list(fiches)` d'abord.")

    _doit_lever(f, ([],), "fiche_de_reference([])", "Aucune fiche : rien à fusionner.")
    _doit_lever(f, ([{"maj": _j(1), "email": "a@x.fr"}],), "fiche_de_reference([une fiche sans 'id'])",
                "Chaque fiche doit avoir un \"id\" (sinon, impossible de dire d'où vient la fiche fusionnée).")
    _doit_lever(f, ([{"id": "A", "maj": _j(1)}, {"id": "B", "email": "b@x.fr"}],),
                "fiche_de_reference([..., une fiche sans 'maj'])",
                "Chaque fiche doit avoir une date \"maj\" (sinon, impossible de savoir laquelle est la plus récente).")


# ---------------------------------------------------------------------------- défi 169 : qualite_dedoublonnage


def _c2(k: int) -> int:
    return k * (k - 1) // 2


def _ref_qualite(predits, verite) -> dict:
    ep = {i: n for n, g in enumerate(predits) for i in g}
    ev = {i: n for n, g in enumerate(verite) for i in g}
    p = sum(_c2(len(g)) for g in predits)
    v = sum(_c2(len(g)) for g in verite)
    j = sum(_c2(k) for k in Counter((ep[i], ev[i]) for i in ep).values())
    precision = j / p if p else 1.0
    rappel = j / v if v else 1.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
    return {"paires_predites": p, "paires_vraies": v, "paires_justes": j, "precision": precision, "rappel": rappel,
            "f1": f1}


_CLES_QUALITE = ("paires_predites", "paires_vraies", "paires_justes", "precision", "rappel", "f1")


class _Fiche:
    """Un identifiant qui compte combien de fois on le hache (fabriquer des paires une par une se voit)."""

    compteur = [0]
    limite = [None]

    __slots__ = ("n",)

    def __init__(self, n: int):
        self.n = n

    def __hash__(self):
        _Fiche.compteur[0] += 1
        if _Fiche.limite[0] is not None and _Fiche.compteur[0] > _Fiche.limite[0]:
            raise _BoucleSansFin
        return hash(self.n)

    def __eq__(self, autre):
        return isinstance(autre, _Fiche) and self.n == autre.n

    def __lt__(self, autre):
        return self.n < autre.n

    def __repr__(self):
        return f"F{self.n}"


def _resultat_qualite(r, quoi: str) -> dict:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
    if not isinstance(r, dict) or set(r) != set(_CLES_QUALITE):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : il faut un dictionnaire avec exactement les clés "
                             f"{', '.join(_CLES_QUALITE)}.")
    for c in _CLES_QUALITE[:3]:
        if not _entier(r[c]):
            raise AssertionError(f"{quoi} : {c} vaut {r[c]!r} ({type(r[c]).__name__}) ; un nombre de paires est un int "
                                 f"(`k * (k - 1) // 2`, avec `//`).")
    for c in _CLES_QUALITE[3:]:
        if not isinstance(r[c], float):
            raise AssertionError(f"{quoi} : {c} vaut {r[c]!r} ({type(r[c]).__name__}) ; il faut un float (1.0, pas 1).")
    return r


def _comparer_qualite(r: dict, attendu: dict, quoi: str, conseil: str = "") -> None:
    if all(r[c] == attendu[c] for c in _CLES_QUALITE[:3]) and all(
            math.isclose(r[c], attendu[c], rel_tol=1e-12, abs_tol=1e-12) for c in _CLES_QUALITE[3:]):
        return
    p, v, j = attendu["paires_predites"], attendu["paires_vraies"], attendu["paires_justes"]
    if p and (r["paires_predites"], r["paires_vraies"], r["paires_justes"]) == (2 * p, 2 * v, 2 * j):
        raise AssertionError(f"{quoi} : vos nombres de paires sont le DOUBLE des bons ({r['paires_predites']} au lieu de "
                             f"{p}…) : (A, B) et (B, A) sont la même paire. Un groupe de k fiches contient "
                             f"k(k - 1)/2 paires, pas k(k - 1).")
    if (r["paires_predites"], r["paires_vraies"]) != (p, v):
        raise AssertionError(f"{quoi} : {r['paires_predites']} paires prédites et {r['paires_vraies']} vraies, au lieu "
                             f"de {p} et {v}. Un groupe de k fiches contient k(k - 1)/2 paires (un groupe d'une "
                             f"fiche : 0). {conseil}".rstrip())
    if r["paires_justes"] != j:
        raise AssertionError(f"{quoi} : {r['paires_justes']} paires justes au lieu de {j}. Une paire est juste si les "
                             f"deux fiches sont dans le même groupe prédit ET dans le même vrai groupe : comptez les "
                             f"fiches par couple (groupe prédit, vrai groupe), puis k(k - 1)/2 par couple. {conseil}".rstrip())
    if not math.isclose(r["precision"], attendu["precision"]) or not math.isclose(r["rappel"], attendu["rappel"]):
        if math.isclose(r["precision"], attendu["rappel"]) and math.isclose(r["rappel"], attendu["precision"]):
            raise AssertionError(f"{quoi} : précision et rappel sont inversés. Précision = justes / PRÉDITES (parmi "
                                 f"les fusions faites, combien sont bonnes) ; rappel = justes / VRAIES (parmi les "
                                 f"fusions à faire, combien sont faites).")
        raise AssertionError(f"{quoi} : precision = {r['precision']!r}, rappel = {r['rappel']!r} au lieu de "
                             f"{attendu['precision']!r} et {attendu['rappel']!r}. {conseil}".rstrip())
    pr, ra = attendu["precision"], attendu["rappel"]
    if math.isclose(r["f1"], (pr + ra) / 2):
        raise AssertionError(f"{quoi} : f1 = {r['f1']!r} est la moyenne ordinaire de la précision et du rappel. Le F1 est "
                             f"leur moyenne HARMONIQUE, 2 × P × R / (P + R) = {attendu['f1']!r} : elle reste basse "
                             f"si l'une des deux l'est (tout fusionner ne donne pas un bon F1).")
    raise AssertionError(f"{quoi} : f1 = {r['f1']!r} au lieu de {attendu['f1']!r}. {conseil}".rstrip())


def verifier_qualite_dedoublonnage(f) -> None:
    verite = [["A", "B", "C"], ["D"]]
    cas = [
        ([["A", "B"], ["C"], ["D"]], verite, ""),
        ([["D"], ["C", "A"], ["B"]], verite, ""),
        ([["A", "B", "C", "D"]], verite, ""),
        ([["A", "B", "C"], ["D"]], verite, ""),
        ([["A"], ["B"], ["C"], ["D"]], verite,
         "Aucune paire prédite : la précision vaut 1.0 (aucune fusion n'est fausse), le rappel 0.0, et le F1 0.0 "
         "(P + R ne vaut pas 0 ici). Évitez la division par zéro avec un `if`."),
        ([["A"], ["B"]], [["B"], ["A"]],
         "Aucune paire nulle part : précision 1.0 et rappel 1.0 (rien à fusionner, rien de fusionné à tort), F1 1.0."),
        ([["A", "D"], ["B"], ["C"]], verite,
         "Aucune paire juste : précision 0.0, rappel 0.0, et F1 0.0 (P + R vaut 0 : évitez la division par zéro)."),
        ([("A", "C"), {"B", "D"}], (["D"], ("A", "B", "C")), "Les groupes peuvent être des listes, des tuples ou des sets."),
        ([["F01", "F02", "F03"], ["F04", "F05"], ["F06", "F07", "F08"], ["F09", "F11"], ["F10"], ["F12", "F14"],
          ["F13"]], VERITE_EXEMPLE, ""),
    ]
    for predits, vraie, conseil in cas:
        quoi = f"qualite_dedoublonnage({_court(predits, 90)}, {_court(vraie, 90)})"
        avant = (copy.deepcopy(predits), copy.deepcopy(vraie))
        r = _resultat_qualite(_appel(f, (predits, vraie), quoi, conseil), quoi)
        if (predits, vraie) != avant:
            raise AssertionError(f"{quoi} modifie les groupes reçus : ne triez pas, ne videz pas les listes de "
                                 f"l'appelant.")
        _comparer_qualite(r, _ref_qualite(predits, vraie), quoi, conseil)

    alea = random.Random(169)
    for _ in range(200):
        n = alea.randrange(1, 25)
        ids = [f"F{i:02d}" for i in range(n)]

        def partition():
            etiquettes = [alea.randrange(0, max(1, n // alea.choice((1, 2, 3)))) for _ in ids]
            groupes: dict = {}
            for i, e in zip(ids, etiquettes):
                groupes.setdefault(e, []).append(i)
            liste = list(groupes.values())
            alea.shuffle(liste)
            return liste

        predits, vraie = partition(), partition()
        quoi = f"qualite_dedoublonnage({_court(predits, 90)}, {_court(vraie, 90)})"
        r = _resultat_qualite(_appel(f, (predits, vraie), quoi), quoi)
        _comparer_qualite(r, _ref_qualite(predits, vraie), quoi)

    for predits, vraie, quoi, conseil in (
            ([["A", "B"], ["B", "C"]], [["A", "B", "C"]], "qualite_dedoublonnage([['A', 'B'], ['B', 'C']], …)",
             "B est dans deux groupes prédits : un regroupement met chaque fiche dans UN groupe."),
            ([["A", "A"], ["B"]], [["A"], ["B"]], "qualite_dedoublonnage([['A', 'A'], ['B']], …)",
             "A est deux fois dans le même groupe."),
            ([["A", "B"]], [["A"], ["B"], ["C"]], "qualite_dedoublonnage([['A', 'B']], [['A'], ['B'], ['C']])",
             "C n'est que dans la vérité : les deux regroupements doivent porter sur les MÊMES fiches."),
            ([["A", "B"], ["Z"]], [["A", "B"]], "qualite_dedoublonnage([['A', 'B'], ['Z']], [['A', 'B']])",
             "Z n'est que dans vos groupes : les deux regroupements doivent porter sur les MÊMES fiches.")):
        _doit_lever(f, (predits, vraie), quoi, conseil)

    # Passer à l'échelle (le correcteur compte les fiches hachées et les lignes exécutées).
    n = 3000
    fiches = [_Fiche(i) for i in range(n)]
    melange = fiches[:]
    random.Random(1690).shuffle(melange)
    for predits, vraie, quoi, lent in (
            ([fiches[0:1500], fiches[1500:2500], fiches[2500:3000]], [fiches[0:1000], fiches[1000:2000], fiches[2000:3000]],
             "qualite_dedoublonnage(3 groupes de 1 000 fiches, recoupés en 1 500 + 1 000 + 500)",
             "Vous fabriquez les paires une par une (1,6 million ici ; 600 millions pour trois groupes de 20 000, qui ne "
             "tiennent pas en mémoire). Comptez sans les fabriquer : k(k - 1)/2 par groupe, et pour les paires justes, "
             "un `Counter` des couples (groupe prédit, vrai groupe) de chaque fiche, puis k(k - 1)/2 par couple."),
            ([fiches[i:i + 3] for i in range(0, n, 3)], [melange[i:i + 3] for i in range(0, n, 3)],
             "qualite_dedoublonnage(1 000 groupes de 3 fiches, contre 1 000 autres groupes de 3)",
             "Vous croisez chaque groupe prédit avec chaque vrai groupe (1 million de croisements ici, 40 milliards pour "
             "200 000 clients). Donnez à chaque fiche le numéro de son vrai groupe (un dictionnaire), puis comptez les "
             "couples (groupe prédit, vrai groupe) avec un `Counter` : un seul passage sur les fiches.")):
        attendu = _ref_qualite(predits, vraie)
        _Fiche.compteur[0], _Fiche.limite[0] = 0, 40 * n
        try:
            r = _resultat_qualite(_appel(f, (predits, vraie), quoi, lignes_max=60 * n, conseil_lent=lent), quoi)
        finally:
            _Fiche.limite[0] = None
        _comparer_qualite(r, attendu, quoi)


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"distance_edition": 164, "regrouper": 165, "paires_candidates": 166, "score_correspondance": 167,
            "fiche_de_reference": 168, "qualite_dedoublonnage": 169}

SERIE = Serie(
    code="27",
    titre="Retrouver le même client : distance, union-find, blocage, score, fusion, qualité",
    module_eleve="defis.serie_27",
    module_corrige="defis.corrige_serie_27",
    defis=(
        Defi(164, "distance_edition", "🟢", "—", verifier_distance_edition),
        Defi(165, "regrouper", "🟢", "—", verifier_regrouper),
        Defi(166, "paires_candidates", "🟡", "—", verifier_paires_candidates),
        Defi(167, "score_correspondance", "🟡", "1.3", verifier_score_correspondance),
        Defi(168, "fiche_de_reference", "🟡", "2.3", verifier_fiche_de_reference),
        Defi(169, "qualite_dedoublonnage", "🔴", "—", verifier_qualite_dedoublonnage),
    ),
)

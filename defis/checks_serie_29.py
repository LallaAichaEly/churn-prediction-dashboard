"""Vérifications automatiques de la série 29 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur joue Git lui-même, et la collègue qui relira votre fusion :

- au défi 176, il vérifie que votre script de modifications redonne EXACTEMENT les deux fichiers, qu'il est
  le plus court possible (il connaît la longueur de la plus longue sous-suite commune), et il compte les
  lignes exécutées dans votre fichier : un tableau de 3 000 × 3 000 cases pour UNE ligne changée se voit,
  quelle que soit la machine ;
- aux défis 177 et 178, ses diffs sont ceux de difflib et de `git diff`, ses patchs s'appliquent comme avec
  GNU patch (c'est vérifié dans tests/test_defis_serie_29.py) ; il rejoue les erreurs classiques (« ,1 »,
  numéros à partir de 0, « -5,0 », blocs mal regroupés, indice négatif) pour nommer la vôtre ;
- au défi 179, ses fusions sont celles de `git merge-file --diff3`, sur des centaines de versions tirées au
  hasard ;
- aux défis 180 et 181, il teste chaque piège SEUL (NaN, 0.1 + 0.2, "1" et 1, lignes dans un autre ordre,
  None et clé absente, True et 1) pour nommer celui qui vous a eu·e ;
- partout, il vérifie que vos fonctions ne modifient pas ce qu'elles reçoivent.
"""

from __future__ import annotations

import copy
import math
import numbers
import random
import re
import sys

import numpy as np
import pandas as pd

from .verif import Defi, Serie
from .versions import (
    BASE,
    EUX,
    NOUS,
    SEPARATEUR,
    PatchRefuse,
    config_exemple,
    generer_versions,
    tables_exemple,
    versions_en_conflit,
    versions_exemple,
)

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


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = 2_000_000, conseil_lent: str = "",
           kwargs: dict | None = None):
    """f(*args, **kwargs), en transformant toute erreur en message lisible."""
    kwargs = kwargs or {}
    try:
        return _trace(f, lambda: f(*args, **kwargs), lignes_max)
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
    """Votre fonction est écrite, mais elle appelle une fonction qui ne l'est pas encore (le 179 utilise le 176)."""
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


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu=ValueError, kwargs: dict | None = None) -> None:
    """Exige que f(*args, **kwargs) lève `attendu` (une classe d'exception, ou un tuple de classes)."""
    noms = " ou ".join(c.__name__ for c in attendu) if isinstance(attendu, tuple) else attendu.__name__
    try:
        resultat = _trace(f, lambda: f(*args, **(kwargs or {})), 2_000_000)
    except NotImplementedError as exc:
        _pas_encore(f, exc)
        raise
    except attendu:
        return
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {_CONSEIL_LENT}") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {noms}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {noms}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _intact(avant, apres, quoi: str, nom: str) -> None:
    if avant != apres:
        raise AssertionError(f"{quoi} a modifié {nom} reçu (avant : {_court(avant, 100)}, après : {_court(apres, 100)}). "
                             "Une fonction ne touche pas à ce qu'on lui prête : travaillez sur une copie, ou "
                             "construisez une nouvelle liste.")


# ---------------------------------------------------------------------------- défi 176 : difference


def _lcs(a: list, b: list) -> int:
    precedente = [0] * (len(b) + 1)
    for x in a:
        courante = [0]
        for j, y in enumerate(b):
            courante.append(precedente[j] + 1 if x == y else max(precedente[j + 1], courante[j]))
        precedente = courante
    return precedente[-1]


def _ref_difference(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    debut = 0
    while debut < len(a) and debut < len(b) and a[debut] == b[debut]:
        debut += 1
    fin = 0
    while fin < len(a) - debut and fin < len(b) - debut and a[len(a) - 1 - fin] == b[len(b) - 1 - fin]:
        fin += 1
    ma, mb = a[debut:len(a) - fin], b[debut:len(b) - fin]
    n, m = len(ma), len(mb)
    t = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            t[i][j] = t[i + 1][j + 1] + 1 if ma[i] == mb[j] else max(t[i + 1][j], t[i][j + 1])
    milieu, i, j = [], 0, 0
    while i < n and j < m:
        if ma[i] == mb[j]:
            milieu.append((" ", ma[i]))
            i, j = i + 1, j + 1
        elif t[i + 1][j] >= t[i][j + 1]:
            milieu.append(("-", ma[i]))
            i += 1
        else:
            milieu.append(("+", mb[j]))
            j += 1
    milieu += [("-", x) for x in ma[i:]] + [("+", y) for y in mb[j:]]
    return [(" ", x) for x in a[:debut]] + _ranger(milieu) + [(" ", x) for x in a[len(a) - fin:]]


def _ranger(ops: list) -> list:
    """Dans chaque bloc de modifications, les « - » puis les « + » (l'ordre des lignes est gardé)."""
    sortie, bloc = [], []
    for op in ops:
        if op[0] == " ":
            sortie += sorted(bloc, key=lambda o: o[0] == "+")
            bloc = []
            sortie.append(op)
        else:
            bloc.append(op)
    return sortie + sorted(bloc, key=lambda o: o[0] == "+")


def _positionnel(a: list, b: list) -> list:
    ops = []
    for x, y in zip(a, b):
        ops += [(" ", x)] if x == y else [("-", x), ("+", y)]
    return ops + [("-", x) for x in a[len(b):]] + [("+", y) for y in b[len(a):]]


def _avec_difflib(a: list, b: list) -> list:
    import difflib

    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            ops += [(" ", x) for x in a[i1:i2]]
        else:
            ops += [("-", x) for x in a[i1:i2]] + [("+", y) for y in b[j1:j2]]
    return ops


def _resultat_ops(r, quoi: str) -> list[tuple[str, str]]:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
    if not isinstance(r, (list, tuple)):
        raise AssertionError(f"{quoi} renvoie {_court(r)} ({type(r).__name__}) : il faut une LISTE de paires "
                             "(opération, ligne). Un générateur se transforme avec list(...).")
    ops = []
    for element in r:
        if not isinstance(element, (list, tuple)) or len(element) != 2:
            raise AssertionError(f"{quoi} contient {_court(element)} : chaque élément est une paire (opération, ligne), "
                                 "par exemple (\"-\", \"    score += 2\").")
        op, ligne = element
        if op not in (" ", "-", "+"):
            raise AssertionError(f"{quoi} contient l'opération {op!r} : il n'y en a que trois, \" \" (une espace : ligne "
                                 "gardée), \"-\" (supprimée) et \"+\" (ajoutée).")
        if not isinstance(ligne, str):
            raise AssertionError(f"{quoi} contient la ligne {_court(ligne)} : une ligne est un texte (str).")
        ops.append((op, ligne))
    return ops


def _premiere_difference(x: list, y: list) -> int:
    for k, (u, v) in enumerate(zip(x, y)):
        if u != v:
            return k
    return min(len(x), len(y))


def _verifier_script(ops: list, a: list, b: list, quoi: str) -> None:
    for cote, attendu, signes, nom in ((" -", a, "« » et « - »", "l'ANCIEN fichier"),
                                       (" +", b, "« » et « + »", "le NOUVEAU fichier")):
        obtenu = [ligne for op, ligne in ops if op in cote]
        if obtenu != attendu and [x.strip() for x in obtenu] == [x.strip() for x in attendu]:
            raise AssertionError(f"{quoi} : « x = 1 » et « x = 1 » suivi d'une espace sont deux lignes DIFFÉRENTES "
                                 "(git diff les montre). Comparez les lignes telles quelles, sans strip().")
        if obtenu != attendu:
            k = _premiere_difference(obtenu, attendu)
            detail = (f"il en manque à partir de la ligne {k + 1} ({_court(attendu[k], 60)})" if k >= len(obtenu)
                      else f"il y en a trop à partir de la ligne {k + 1} ({_court(obtenu[k], 60)})" if k >= len(attendu)
                      else f"ligne {k + 1} : {_court(obtenu[k], 60)} au lieu de {_court(attendu[k], 60)}")
            raise AssertionError(f"{quoi} : les lignes {signes}, dans l'ordre, devraient redonner {nom} ; {detail}. "
                                 "Le script doit permettre de reconstruire les deux versions (c'est ce que fait "
                                 "`git apply`) : chaque ligne une fois, à sa place.")
    gardees, attendu = sum(op == " " for op, _ in ops), _lcs(a, b)
    if gardees < attendu:
        gardees = f"{gardees} ligne" if gardees <= 1 else f"{gardees} lignes"
        if _ranger(ops) == _ranger(_positionnel(a, b)):
            raise AssertionError(f"{quoi} garde {gardees} au lieu de {attendu} : vous comparez position par "
                                 "position. Une ligne ajoutée en haut décale toutes les autres, qui paraissent alors "
                                 "toutes modifiées. Cherchez la plus longue sous-suite commune (le tableau de la "
                                 "distance d'édition, défi 164), puis reconstruisez le script.")
        if _ranger(ops) == _ranger(_avec_difflib(a, b)):
            raise AssertionError(f"{quoi} garde {gardees} au lieu de {attendu} : difflib.SequenceMatcher cherche "
                                 "des blocs qui « se ressemblent », pas le PLUS COURT script (et il ignore les lignes "
                                 "trop fréquentes). Écrivez la plus longue sous-suite commune vous-même.")
        raise AssertionError(f"{quoi} garde {gardees}, mais {attendu} lignes sont communes aux deux versions, dans "
                             "le même ordre : votre script supprime puis rajoute des lignes qui pouvaient rester. "
                             "Reconstruisez-le à partir du tableau de la plus longue sous-suite commune.")
    precedent = " "
    for k, (op, _) in enumerate(ops):
        if op == "-" and precedent == "+":
            raise AssertionError(f"{quoi} : l'opération n°{k + 1} est un « - » écrit juste après un « + ». Dans chaque bloc "
                                 "de modifications (entre deux lignes gardées), écrivez d'abord TOUTES les lignes "
                                 "supprimées, puis toutes les lignes ajoutées, comme git diff.")
        precedent = op


_CONSEIL_TRONQUER = ("Votre fonction a été arrêtée sur un fichier de 3 000 lignes où UNE seule ligne change : sans "
                     "préparation, le tableau a 9 millions de cases. Retirez d'abord le début et la fin communs (deux "
                     "boucles `while`), puis ne remplissez le tableau que pour le MILIEU, ce qui reste. git diff fait "
                     "pareil.")


def _une_difference(f, a: list, b: list, quoi: str, lignes_max: int = 200_000, conseil_lent: str = "") -> None:
    copie_a, copie_b = list(a), list(b)
    r = _appel(f, (a, b), quoi, lignes_max=lignes_max, conseil_lent=conseil_lent)
    ops = _resultat_ops(r, quoi)
    _intact(copie_a, a, quoi, "l'ancien fichier")
    _intact(copie_b, b, quoi, "le nouveau fichier")
    _verifier_script(ops, a, b, quoi)


_CAS_DIFFERENCE = [
    ([], [], "deux fichiers vides"),
    (["a"], [], "tout supprimer"),
    ([], ["a", "b"], "tout créer"),
    (["}", "", "}", "", "}"], ["}", "", "x", "}", "", "}"], "des lignes répétées (des accolades, des lignes vides)"),
    (["x = 1", "y = 2", "z = 3"], ["x = 1 ", "y = 2", "z = 3"], "une espace ajoutée en fin de ligne"),
    (["a", "b", "c", "d"], ["a", "B", "C", "d"], "deux lignes remplacées par deux autres"),
    (["a", "b", "c"], ["a", "x", "b", "y", "c"], "deux insertions séparées"),
    (["a", "b"], ["b", "a"], "deux lignes échangées"),
    (["1", "2", "3", "4", "5", "6"], ["0", "1", "2", "4", "5", "6", "7"], "au début, au milieu et à la fin"),
]


def verifier_difference(f) -> None:
    base, nous, eux = versions_exemple()
    _une_difference(f, base, nous, "difference(base, nous) (versions_exemple)")
    _une_difference(f, base, eux, "difference(base, eux) (versions_exemple)")
    entete = ["# -*- coding: utf-8 -*-"] + base
    _une_difference(f, base, entete, "difference(base, [une ligne ajoutée en haut] + base)")
    for a, b, quoi in _CAS_DIFFERENCE:
        _une_difference(f, a, b, f"difference({_court(a, 50)}, {_court(b, 50)}) ({quoi})")

    for args, quoi in ((("a = 1\nb = 2\n", ["a = 1"]), "difference(\"a = 1\\nb = 2\\n\", [\"a = 1\"])"),
                       ((["a = 1"], "a = 1"), "difference([\"a = 1\"], \"a = 1\")")):
        _doit_lever(f, args, quoi, "Un texte entier n'est pas une liste de lignes : il serait comparé LETTRE par "
                                   "lettre. Vérifiez `isinstance(ancien, str)` et levez TypeError (l'appelant fera "
                                   "texte.splitlines()).", attendu=TypeError)

    alea = random.Random(176)
    lignes = ["}", "", "x = 1", "return x", "if x:"]
    for _ in range(300):
        a = [alea.choice(lignes) for _ in range(alea.randrange(0, 10))]
        b = [alea.choice(lignes) for _ in range(alea.randrange(0, 10))]
        _une_difference(f, a, b, f"difference({_court(a, 70)}, {_court(b, 70)})", lignes_max=60_000)

    gros = [f"ligne {i} : client C{i * 7919 % 10_000:04d}" for i in range(3000)]
    modifie = gros[:1500] + ["ligne 1500 : client MODIFIÉ"] + gros[1501:]
    _une_difference(f, gros, modifie, "difference(3 000 lignes, les mêmes avec la ligne 1501 modifiée)",
                    lignes_max=200_000, conseil_lent=_CONSEIL_TRONQUER)

    a = [f"l{i}" for i in range(150)]
    b = ["nouvelle 0"] + [x for i, x in enumerate(a) if i % 9] + ["nouvelle 1"]
    _une_difference(f, a, b, "difference(150 lignes, modifiées du début à la fin)", lignes_max=2_000_000,
                    conseil_lent="Votre fonction a été arrêtée sur deux fichiers de 150 lignes modifiés partout : une "
                                 "récursion sans mémoire refait les mêmes calculs des milliards de fois. Remplissez "
                                 "un tableau, case par case.")


# ---------------------------------------------------------------------------- défi 177 : diff_unifie


def _ref_unifie(ops: list, nom: str = "fichier", contexte: int = 3, defaut: str | None = None) -> list[str]:
    changements = [k for k, (op, _) in enumerate(ops) if op != " "]
    if not changements:
        return []
    num_a, num_b, x, y = [], [], 1, 1
    for op, _ in ops:
        num_a.append(x)
        num_b.append(y)
        x += op != "+"
        y += op != "-"
    limite = {"seuil_strict": 2 * contexte - 1, "seuil_large": 2 * contexte + 1}.get(defaut, 2 * contexte)
    groupes = [[changements[0], changements[0]]]
    for k in changements[1:]:
        if k - groupes[-1][1] - 1 <= limite:
            groupes[-1][1] = k
        else:
            groupes.append([k, k])

    def plage(debut: int, longueur: int) -> str:
        if defaut == "zero":
            debut -= 1
        if longueur == 0 and defaut != "vide":
            return f"{debut - 1},0"
        if longueur == 1 and defaut != "virgule_un":
            return str(debut)
        return f"{debut},{longueur}"

    sortie = [f"--- a/{nom}", f"+++ b/{nom}"]
    for premier, dernier in groupes:
        debut = premier - contexte if defaut == "negatif" else max(0, premier - contexte)
        fin = min(len(ops), dernier + 1 + contexte)
        bloc = ops[debut:fin]
        if defaut == "negatif" and debut < 0:
            debut = 0
        n_a = sum(op != "+" for op, _ in bloc)
        n_b = sum(op != "-" for op, _ in bloc)
        sortie.append(f"@@ -{plage(num_a[debut], n_a)} +{plage(num_b[debut], n_b)} @@")
        sortie += [op + ligne for op, ligne in bloc]
    return sortie


_DEFAUTS_UNIFIE = {
    "virgule_un": "Une plage d'UNE seule ligne s'écrit « -5 », pas « -5,1 » (git diff omet « ,1 »).",
    "zero": "Les lignes d'un fichier sont numérotées à partir de 1, pas de 0 : ajoutez 1 aux indices de la liste.",
    "vide": "Quand un côté n'a AUCUNE ligne dans le bloc, son numéro est celui de la ligne d'AVANT : « -5,0 » veut "
            "dire « rien, juste après la ligne 5 » (et « -0,0 » : au tout début du fichier).",
    "seuil_strict": "Deux changements séparés par EXACTEMENT 2 × contexte lignes communes vont dans le même bloc : "
                    "leurs contextes se touchent. Le test est `écart <= 2 * contexte`.",
    "seuil_large": "Deux changements séparés par 2 × contexte + 1 lignes communes font DEUX blocs : une ligne commune "
                   "au moins les sépare, qu'aucun contexte ne montre. Le test est `écart <= 2 * contexte`.",
    "negatif": "Un bloc ne commence pas avant la première ligne : `max(0, premier - contexte)`. En Python, un indice "
               "négatif repart de la FIN de la liste : ops[-2:5] est vide.",
}


def _resultat_unifie(r, quoi: str) -> list[str]:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
    if isinstance(r, str):
        raise AssertionError(f"{quoi} renvoie un seul texte : il faut une LISTE de lignes, sans « \\n » (le texte "
                             "complet serait \"\\n\".join(lignes)).")
    if not isinstance(r, (list, tuple)) or not all(isinstance(ligne, str) for ligne in r):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : il faut une liste de lignes (des str).")
    return list(r)


def _diagnostic_unifie(obtenu: list, attendu: list, ops: list, nom: str, contexte: int, quoi: str) -> str:
    if not attendu:
        return f"{quoi} renvoie {_court(obtenu)} : sans aucune modification, git diff n'écrit RIEN. Renvoyez []."
    if not obtenu:
        return f"{quoi} renvoie [] alors qu'il y a des modifications."
    for defaut, conseil in _DEFAUTS_UNIFIE.items():
        if _ref_unifie(ops, nom, contexte, defaut) == obtenu:
            return f"{quoi} : {conseil}"
    if obtenu[:2] != attendu[:2]:
        return (f"{quoi} commence par {_court(obtenu[:2], 80)} : les deux premières lignes sont les en-têtes "
                f"{attendu[:2]!r} (l'ancien fichier en « a/ », le nouveau en « b/ »).")
    blocs_obtenus = sum(ligne.startswith("@@") for ligne in obtenu)
    blocs_attendus = sum(ligne.startswith("@@") for ligne in attendu)
    if blocs_obtenus != blocs_attendus:
        return (f"{quoi} écrit {blocs_obtenus} bloc(s) « @@ » au lieu de {blocs_attendus}. Deux changements séparés par "
                f"au plus 2 × contexte lignes communes vont dans le même bloc ; plus loin, deux blocs.")
    k = _premiere_difference(obtenu, attendu)
    if k >= len(attendu) or k >= len(obtenu):
        return f"{quoi} : {len(obtenu)} lignes au lieu de {len(attendu)} (attendu : {_court(attendu, 200)})."
    if attendu[k].startswith("@@"):
        return (f"{quoi} : l'en-tête de bloc n°{sum(x.startswith('@@') for x in attendu[:k + 1])} devrait être "
                f"{attendu[k]!r}, vous écrivez {obtenu[k]!r}. « -a,b » : le bloc commence à la ligne a de l'ancien "
                "fichier et en montre b (contexte + lignes retirées) ; « +c,d » : pareil pour le nouveau (contexte + "
                "lignes ajoutées).")
    if obtenu[k] == attendu[k][1:]:
        return (f"{quoi} : la ligne {k + 1} devrait être {attendu[k]!r}. Chaque ligne d'un bloc commence par son signe : "
                "« » (une espace) pour le contexte, « - » ou « + » pour les modifications.")
    return f"{quoi} : la ligne {k + 1} devrait être {attendu[k]!r}, vous écrivez {obtenu[k]!r}."


def _un_unifie(f, ops: list, quoi: str, nom: str | None = None, contexte: int | None = None) -> None:
    kwargs = {}
    if nom is not None:
        kwargs["nom"] = nom
    if contexte is not None:
        kwargs["contexte"] = contexte
    copie = list(ops)
    r = _appel(f, (ops,), quoi, kwargs=kwargs, lignes_max=500_000)
    _intact(copie, ops, quoi, "la liste d'opérations")
    obtenu = _resultat_unifie(r, quoi)
    attendu = _ref_unifie(ops, "fichier" if nom is None else nom, 3 if contexte is None else contexte)
    if obtenu != attendu:
        raise AssertionError(_diagnostic_unifie(obtenu, attendu, ops, "fichier" if nom is None else nom,
                                                3 if contexte is None else contexte, quoi))


def verifier_diff_unifie(f) -> None:
    base, nous, eux = versions_exemple()
    _un_unifie(f, _ref_difference(base, nous), "diff_unifie(difference(base, nous), nom=\"score.py\")", nom="score.py")
    _un_unifie(f, _ref_difference(base, eux), "diff_unifie(difference(base, eux))")
    for contexte in (0, 1, 2):
        _un_unifie(f, _ref_difference(base, nous), f"diff_unifie(difference(base, nous), contexte={contexte})",
                   contexte=contexte)
    dix = [f"ligne {i}" for i in range(1, 11)]
    cas = [
        (["a", "b", "c"], ["a", "B", "c"], 0, "une ligne remplacée, contexte=0"),
        ([], ["x"], 3, "un fichier créé"),
        (["a", "b", "c"], [], 3, "un fichier vidé"),
        (dix, dix[:5] + ["nouvelle", "autre"] + dix[5:], 0, "deux lignes insérées après la ligne 5, contexte=0"),
        (dix, ["# en-tête"] + dix, 0, "une ligne insérée tout en haut, contexte=0"),
        (dix, dix[:-1], 0, "la dernière ligne supprimée, contexte=0"),
        (dix + dix, ["X"] + dix[1:] + dix, 3, "la ligne 1 modifiée, contexte=3"),
        (dix, dix[:1] + ["X"] + dix[2:8] + ["Y"] + dix[9:], 3, "deux changements séparés par 6 lignes, contexte=3"),
        (dix + ["fin"], dix[:1] + ["X"] + dix[2:9] + ["Y"] + dix[10:] + ["fin"], 3,
         "deux changements séparés par 7 lignes, contexte=3"),
        (dix, dix[:2] + ["X"] + dix[3:5] + ["Y"] + dix[6:], 1, "deux changements séparés par 2 lignes, contexte=1"),
        (dix, list(dix), 3, "aucune modification"),
    ]
    for a, b, contexte, quoi in cas:
        _un_unifie(f, _ref_difference(a, b), f"diff_unifie(…) ({quoi})", contexte=contexte)

    _doit_lever(f, (_ref_difference(["a"], ["b"]),), "diff_unifie(ops, contexte=-1)",
                "Un contexte négatif n'a pas de sens : levez ValueError.", kwargs={"contexte": -1})

    alea = random.Random(177)
    for _ in range(300):
        a = [f"l{i}" for i in range(alea.randrange(0, 25))]
        b = []
        for x in a:
            tirage = alea.random()
            if tirage < 0.1:
                continue
            if tirage < 0.2:
                b.append(x + " modifiée")
                continue
            if tirage < 0.25:
                b.append(f"insérée {alea.randrange(1000)}")
            b.append(x)
        contexte = alea.randrange(0, 5)
        _un_unifie(f, _ref_difference(a, b), f"diff_unifie(difference({_court(a, 50)}, {_court(b, 50)}), "
                                             f"contexte={contexte})", contexte=contexte)


# ---------------------------------------------------------------------------- défi 178 : appliquer_patch

_EN_TETE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*")


def _ref_appliquer(lignes: list[str], patch: list[str]) -> list[str]:
    blocs, k = [], 0
    while k < len(patch) and not patch[k].startswith("@@"):
        k += 1
    while k < len(patch):
        m = _EN_TETE.fullmatch(patch[k])
        if m is None:
            raise PatchRefuse(f"en-tête attendu : {patch[k]!r}")
        debut, longueur = int(m[1]), int(m[2] or 1)
        reste_a, reste_b = longueur, int(m[4] or 1)
        k += 1
        attendues, nouvelles = [], []
        while reste_a > 0 or reste_b > 0:
            if k >= len(patch):
                raise PatchRefuse("patch tronqué")
            op, texte = patch[k][:1], patch[k][1:]
            if op == " ":
                attendues.append(texte)
                nouvelles.append(texte)
                reste_a, reste_b = reste_a - 1, reste_b - 1
            elif op == "-":
                attendues.append(texte)
                reste_a -= 1
            elif op == "+":
                nouvelles.append(texte)
                reste_b -= 1
            else:
                raise PatchRefuse(f"ligne inattendue : {patch[k]!r}")
            if reste_a < 0 or reste_b < 0:
                raise PatchRefuse("en-tête faux")
            k += 1
        blocs.append((debut, longueur, attendues, nouvelles))
    sortie, lues, decalage = [], 0, 0
    for debut, longueur, attendues, nouvelles in blocs:
        nominal = debut if longueur == 0 else debut - 1
        prevu = nominal + decalage
        trouve = None
        for ecart in range(len(lignes) + 1):
            for position in ((prevu,) if ecart == 0 else (prevu + ecart, prevu - ecart)):
                if lues <= position <= len(lignes) - longueur and lignes[position:position + longueur] == attendues:
                    trouve = position
                    break
            if trouve is not None:
                break
        if trouve is None:
            raise PatchRefuse("bloc introuvable")
        decalage = trouve - nominal
        sortie += lignes[lues:trouve] + nouvelles
        lues = trouve + longueur
    return sortie + lignes[lues:]


def _un_patch(f, lignes: list, patch: list, quoi: str, conseil: str = "", attendu: list | None = None) -> None:
    copie_l, copie_p = list(lignes), list(patch)
    r = _appel(f, (lignes, patch), quoi, conseil, lignes_max=500_000)
    _intact(copie_l, lignes, quoi, "le fichier")
    _intact(copie_p, patch, quoi, "le patch")
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return` (renvoyez le NOUVEAU fichier, une liste).")
    if not isinstance(r, list) or not all(isinstance(x, str) for x in r):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : il faut une liste de lignes (des str).")
    if r is lignes:
        raise AssertionError(f"{quoi} renvoie la liste reçue elle-même : renvoyez une NOUVELLE liste.")
    attendu = _ref_appliquer(lignes, patch) if attendu is None else attendu
    if r != attendu:
        k = _premiere_difference(r, attendu)
        detail = (f"ligne {k + 1} : {_court(r[k], 60)} au lieu de {_court(attendu[k], 60)}"
                  if k < len(r) and k < len(attendu) else f"{len(r)} lignes au lieu de {len(attendu)}")
        raise AssertionError(f"{quoi} donne un mauvais fichier ({detail}). {conseil}".rstrip())


def _refuse(f, lignes: list, patch: list, quoi: str, conseil: str) -> None:
    copie = list(lignes)
    try:
        resultat = _trace(f, lambda: f(lignes, patch), 500_000)
    except NotImplementedError:
        raise
    except PatchRefuse:
        _intact(copie, lignes, quoi + " (refusé)", "le fichier")
        return
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {_CONSEIL_LENT}") from None
    except ValueError as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} : levez PatchRefuse (importée de defis.versions), pour "
                             "que l'appelant la reconnaisse. C'est une sorte de ValueError.") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de PatchRefuse. {conseil}") from None
    raise AssertionError(f"{quoi} devrait lever PatchRefuse, mais renvoie {_court(resultat)}. {conseil}")


def verifier_appliquer_patch(f) -> None:
    base, nous, eux = versions_exemple()
    patch = _ref_unifie(_ref_difference(base, nous), "score.py")
    _un_patch(f, base, patch, "appliquer_patch(base, diff_unifie(difference(base, nous)))", attendu=nous)
    patch_eux = _ref_unifie(_ref_difference(base, eux), "score.py", 1)
    _un_patch(f, base, patch_eux, "appliquer_patch(base, le patch de Malik, contexte=1)", attendu=eux)

    git = ["diff --git a/score.py b/score.py", "index 3b18e51..a9c2f4e 100644"] + patch[:2] + \
          [patch[2] + " def score_risque(client):"] + patch[3:]
    _un_patch(f, base, git, "appliquer_patch(base, un patch écrit par git diff)",
              "Tout ce qui précède le premier « @@ » est un en-tête (ignoré). Et git écrit le nom de la fonction APRÈS "
              "le second « @@ » : « @@ -1,5 +1,6 @@ def score_risque(client): ». Ce texte-là aussi est ignoré.",
              attendu=nous)

    dix = [f"ligne {i}" for i in range(1, 11)]
    une = dix[:4] + ["LIGNE 5"] + dix[5:]
    _un_patch(f, dix, _ref_unifie(_ref_difference(dix, une), "f", 0), "appliquer_patch(dix lignes, « @@ -5 +5 @@ »)",
              "« -5 » sans virgule veut dire « -5,1 » : une seule ligne (git omet « ,1 »).", attendu=une)
    insere = dix[:5] + ["nouvelle", "autre"] + dix[5:]
    _un_patch(f, dix, _ref_unifie(_ref_difference(dix, insere), "f", 0),
              "appliquer_patch(dix lignes, « @@ -5,0 +6,2 @@ » : insérer après la ligne 5)",
              "« -5,0 » : rien à retirer, on insère APRÈS la ligne 5, donc à l'indice 5 de la liste (et pas 4).",
              attendu=insere)
    _un_patch(f, dix, _ref_unifie(_ref_difference(dix, ["# en-tête"] + dix), "f", 0),
              "appliquer_patch(dix lignes, « @@ -0,0 +1 @@ » : insérer tout en haut)",
              "« -0,0 » : on insère avant la ligne 1, à l'indice 0.", attendu=["# en-tête"] + dix)
    _un_patch(f, ["a", "b"], _ref_unifie(_ref_difference(["a", "b"], []), "f"), "appliquer_patch([\"a\", \"b\"], "
              "« @@ -1,2 +0,0 @@ » : tout supprimer)", attendu=[])

    requete = ["SELECT contrat, COUNT(*) AS n", "-- total par contrat", "FROM clients", "GROUP BY contrat;"]
    sans_commentaire = [requete[0], requete[2], requete[3]]
    patch_sql = _ref_unifie(_ref_difference(requete, sans_commentaire), "requete.sql")
    _un_patch(f, requete, patch_sql, "appliquer_patch(une requête SQL, un patch qui retire « -- total par contrat »)",
              "Dans ce patch, « --- total par contrat » est une ligne SUPPRIMÉE (« - » suivi de « -- total… »), pas "
              "un en-tête de fichier : COMPTEZ les lignes annoncées par l'en-tête du bloc au lieu de regarder par "
              "quoi elles commencent.", attendu=sans_commentaire)

    trois = ["# fichier relu", "# par l'équipe", ""]
    _un_patch(f, trois + base, patch, "appliquer_patch(base avec 3 lignes ajoutées en haut, le patch d'Inès)",
              "Le fichier a 3 lignes de plus en haut que celui du patch : les lignes attendues ne sont plus au numéro "
              "annoncé. Cherchez-les autour : écart 0, +1, -1, +2, -2… (c'est ce que fait GNU patch).",
              attendu=trois + nous)
    long_ = [f"ligne {i}" for i in range(1, 41)]
    change = long_[:29] + ["LIGNE 30"] + long_[30:]
    _un_patch(f, long_[2:], _ref_unifie(_ref_difference(long_, change), "f"),
              "appliquer_patch(un fichier dont on a retiré les 2 premières lignes, un patch pour la ligne 30)",
              "Les lignes attendues sont 2 lignes PLUS HAUT que leur numéro : cherchez aussi vers le haut.",
              attendu=change[2:])

    # Le décalage trouvé pour un bloc vaut pour les suivants.
    motif = ["}", ""] * 12
    avant = ["def a():", "    x = 1", "    y = 2"] + motif + ["fin"]
    apres = list(avant)
    apres[1] = "    x = 10"
    apres[15] = "# milieu"
    patch_motif = _ref_unifie(_ref_difference(avant, apres), "f", 1)
    pre = ["# 1", "# 2", "# 3", "# 4"]
    _un_patch(f, pre + avant, patch_motif, "appliquer_patch(4 lignes ajoutées en haut d'un fichier très répétitif)",
              "Le bloc 1 n'a été trouvé que 4 lignes plus bas : cherchez le bloc 2 à partir de SON numéro + 4. Sur des "
              "lignes répétées (« } », lignes vides), le contexte se trouve à plusieurs endroits, et le plus proche du "
              "numéro brut n'est pas le bon.", attendu=pre + apres)
    motif = ["a", "b"] * 6
    apres = list(motif)
    apres[5] = "X"
    _un_patch(f, ["# 1"] + motif, _ref_unifie(_ref_difference(motif, apres), "f", 1),
              "appliquer_patch(une ligne ajoutée en haut, contexte trouvé à +1 ET à -1)",
              "À distance égale, essayez d'abord PLUS BAS dans le fichier (+1 avant -1), comme GNU patch : des lignes "
              "ajoutées en haut sont le cas le plus fréquent.", attendu=["# 1"] + apres)

    alea = random.Random(178)
    for _ in range(200):
        a = [alea.choice(["}", "", "x = 1", "return x"]) if alea.random() < 0.3 else f"l{i}"
             for i in range(alea.randrange(1, 30))]
        b = [x if alea.random() > 0.15 else x + " !" for x in a]
        if alea.random() < 0.3:
            b.insert(alea.randrange(len(b) + 1), "insérée")
        cible = [f"pre {i}" for i in range(alea.randrange(0, 3))] + a
        _un_patch(f, cible, _ref_unifie(_ref_difference(a, b), "f", alea.randrange(0, 4)),
                  f"appliquer_patch({_court(cible, 60)}, diff de {_court(a, 40)} vers {_court(b, 40)})",
                  "Revoyez les règles de la docstring : numéros de l'ANCIEN fichier, écart 0, +1, -1, +2…, le décalage "
                  "d'un bloc reporté sur le suivant, et jamais avant la fin du bloc précédent.")

    abime = base[:4] + base[5:]
    _refuse(f, abime, patch, "appliquer_patch(base sans sa ligne 5, le patch qui modifie la ligne 5)",
            "Vérifiez que les lignes « » et « - » du bloc sont bien dans le fichier AVANT de le modifier. Appliquer "
            "un patch à l'aveugle, c'est écraser la mauvaise ligne sans prévenir.")
    vingt = [f"ligne {i}" for i in range(1, 21)]
    deux = list(vingt)
    deux[1], deux[17] = "LIGNE 2", "LIGNE 18"
    patch_deux = _ref_unifie(_ref_difference(vingt, deux), "f", 1)
    autre = vingt[:16] + ["ligne 17 changée par quelqu'un d'autre"] + vingt[17:]
    _refuse(f, autre, patch_deux, "appliquer_patch(un fichier où seul le bloc 2 ne s'applique plus)",
            "Tout ou rien : vérifiez TOUS les blocs avant de construire le résultat, et ne modifiez jamais la liste "
            "reçue (un patch appliqué à moitié laisse un fichier que personne n'a écrit).")
    entete_faux = ["@@ -1,3 +1,3 @@", " a", "-b", "+B", "@@ -9 +9 @@", "-x", "+y"]
    _refuse(f, ["a", "b", "c", ".", ".", ".", ".", ".", "x", "."], entete_faux,
            "appliquer_patch(…, un bloc qui annonce 3 lignes et n'en a que 2)",
            "L'en-tête annonce combien de lignes suivent : s'il en manque, le patch est abîmé. Levez PatchRefuse.")
    _refuse(f, ["a", "b", "c"], ["@@ -1,2 +1,2 @@", " a", "-b"], "appliquer_patch(…, un patch coupé au milieu d'un bloc)",
            "Le patch s'arrête avant d'avoir donné toutes les lignes annoncées : levez PatchRefuse.")
    _refuse(f, ["a", "b", "c"], ["@@ -1,2 +1,2 @@", " a", "*b", "+B"], "appliquer_patch(…, une ligne « *b »)",
            "Dans un bloc, une ligne commence par « » (espace), « - » ou « + » ; sinon le patch est abîmé.")
    _refuse(f, ["a", "b", "c"], ["@@ -1,x +1,2 @@", " a", "-b"], "appliquer_patch(…, un en-tête « @@ -1,x +1,2 @@ »)",
            "Un en-tête mal formé : levez PatchRefuse (une expression régulière avec re.fullmatch le vérifie).")

    vide = ["a", "b"]
    _un_patch(f, vide, [], "appliquer_patch([\"a\", \"b\"], [])", attendu=["a", "b"])


# ---------------------------------------------------------------------------- défi 179 : fusion_trois_voies


def _correspondances(base: list, autre: list) -> dict:
    paires, i, j = {}, 0, 0
    for op, _ in _ref_difference(base, autre):
        if op == " ":
            paires[i] = j
        i += op != "+"
        j += op != "-"
    return paires


def _ref_fusion(base: list, nous: list, eux: list) -> tuple[list[str], int]:
    avec_n, avec_e = _correspondances(base, nous), _correspondances(base, eux)
    sortie, conflits, i, j, k = [], 0, 0, 0, 0
    while True:
        i2 = i
        while i2 < len(base) and not (i2 in avec_n and i2 in avec_e):
            i2 += 1
        j2, k2 = (avec_n[i2], avec_e[i2]) if i2 < len(base) else (len(nous), len(eux))
        b, n, e = base[i:i2], nous[j:j2], eux[k:k2]
        if n == b:
            sortie += e
        elif e == b or n == e:
            sortie += n
        else:
            sortie += [NOUS, *n, BASE, *b, SEPARATEUR, *e, EUX]
            conflits += 1
        if i2 == len(base):
            return sortie, conflits
        sortie.append(base[i2])
        i, j, k = i2 + 1, j2 + 1, k2 + 1


def _resultat_fusion(r, quoi: str) -> tuple[list[str], int]:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return` (renvoyez (lignes, nombre_de_conflits)).")
    if not isinstance(r, (tuple, list)) or len(r) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(r)} : il faut une paire (lignes fusionnées, nombre de conflits), "
                             "comme `git merge-file`, qui renvoie le nombre de conflits.")
    lignes, conflits = r
    if not isinstance(lignes, list) or not all(isinstance(x, str) for x in lignes):
        raise AssertionError(f"{quoi} : le premier élément est {_court(lignes)} ; il faut une liste de lignes (str).")
    if not isinstance(conflits, numbers.Integral) or isinstance(conflits, bool):
        raise AssertionError(f"{quoi} : le nombre de conflits est {_court(conflits)} ; il faut un entier.")
    return lignes, int(conflits)


def _une_fusion(f, base: list, nous: list, eux: list, quoi: str, conseil: str = "") -> None:
    copies = (list(base), list(nous), list(eux))
    r = _appel(f, (base, nous, eux), quoi, lignes_max=2_000_000)
    for copie, recu, nom in zip(copies, (base, nous, eux), ("base", "nous", "eux")):
        _intact(copie, recu, quoi, f"la version « {nom} »")
    lignes, conflits = _resultat_fusion(r, quoi)
    attendu, n_attendu = _ref_fusion(base, nous, eux)
    if lignes == attendu and conflits == n_attendu:
        return
    marques = [x for x in lignes if x.startswith(("<<<<<<<", "|||||||", "=======", ">>>>>>>"))]
    if n_attendu and not marques:
        raise AssertionError(f"{quoi} ne montre aucun conflit, alors qu'Inès et Malik ont modifié le même endroit "
                             f"différemment. Une fusion ne CHOISIT jamais en silence : encadrez les deux versions "
                             f"avec {NOUS!r}, {BASE!r}, {SEPARATEUR!r} et {EUX!r}, et comptez le conflit. {conseil}"
                             .rstrip())
    if n_attendu == 0 and marques:
        conseil = conseil or ("Si un seul côté a touché un morceau, on prend sa version ; si les deux ont fait "
                              "EXACTEMENT la même modification, on la prend une fois.")
        raise AssertionError(f"{quoi} signale un conflit qui n'en est pas un. {conseil}")
    if marques and BASE not in lignes and n_attendu:
        raise AssertionError(f"{quoi} : dans un conflit, montrez aussi la version de BASE, entre {BASE!r} et "
                             f"{SEPARATEUR!r} (le style « diff3 » de Git, `git config merge.conflictStyle diff3`) : "
                             "sans elle, impossible de savoir qui a changé quoi.")
    if marques and not set(marques) <= {NOUS, BASE, SEPARATEUR, EUX}:
        raise AssertionError(f"{quoi} : les marqueurs sont exactement {NOUS!r}, {BASE!r}, {SEPARATEUR!r} et {EUX!r} "
                             f"(les constantes NOUS, BASE, SEPARATEUR et EUX) ; vous écrivez {_court(sorted(set(marques)))}.")
    if conflits != n_attendu and lignes == attendu:
        raise AssertionError(f"{quoi} : les lignes sont justes, mais le nombre de conflits est {conflits} au lieu de "
                             f"{n_attendu} (un par bloc encadré de marqueurs).")
    if lignes == [ligne for _, ligne in _ref_difference(nous, eux)]:
        raise AssertionError(f"{quoi} : vous fusionnez « nous » et « eux » sans regarder la BASE. Une ligne supprimée "
                             "par l'un ressemble alors à une ligne ajoutée par l'autre, et elle revient. Comparez "
                             "CHAQUE côté à la base : difference(base, nous) et difference(base, eux).")
    k = _premiere_difference(lignes, attendu)
    detail = (f"ligne {k + 1} : {_court(lignes[k], 60)} au lieu de {_court(attendu[k], 60)}"
              if k < len(lignes) and k < len(attendu) else f"{len(lignes)} lignes au lieu de {len(attendu)}")
    raise AssertionError(f"{quoi} : fusion différente de celle de git merge-file ({detail} ; {conflits} conflit(s) au "
                         f"lieu de {n_attendu}). Attendu : {_court(attendu, 300)}. {conseil}".rstrip())


def verifier_fusion_trois_voies(f) -> None:
    base, nous, eux = versions_exemple()
    _une_fusion(f, base, nous, eux, "fusion_trois_voies(*versions_exemple())",
                "Inès et Malik ont modifié des endroits différents : la fusion doit contenir LES DEUX modifications, "
                "sans conflit.")
    base, nous, eux = versions_en_conflit()
    _une_fusion(f, base, nous, eux, "fusion_trois_voies(*versions_en_conflit())")

    cas = [
        (["a", "b", "c"], ["a", "B", "c"], ["a", "B", "c"],
         "la même modification des deux côtés",
         "Inès et Malik ont fait EXACTEMENT la même modification : ce n'est pas un conflit, on la prend une fois."),
        (["a", "b", "c"], ["a", "b", "c"], ["a", "c"], "Malik supprime une ligne, Inès ne touche à rien",
         "La ligne « b » supprimée par Malik doit rester supprimée. Sans la BASE, impossible de savoir si Malik l'a "
         "supprimée ou si Inès l'a ajoutée : c'est pour ça qu'une fusion compare chaque côté à la base, pas l'un à "
         "l'autre."),
        (["a", "b", "c", "d"], ["a", "B", "c", "d"], ["a", "b", "C", "d"], "deux lignes VOISINES modifiées",
         "Les deux modifications se touchent (aucune ligne inchangée entre elles) : Git signale un conflit, car rien "
         "ne garantit qu'elles aient un sens ensemble. Une ligne n'est un point d'appui que si les DEUX côtés l'ont "
         "gardée."),
        (["a", "b", "c", "d", "e"], ["a", "B", "c", "d", "e"], ["a", "b", "c", "D", "e"],
         "deux modifications séparées par une ligne gardée des deux côtés",
         "La ligne « c », gardée des deux côtés, sépare les deux modifications : pas de conflit."),
        (["a", "b", "c"], ["a", "c"], ["a", "B", "c"], "Inès supprime une ligne que Malik modifie",
         "Supprimer d'un côté, modifier de l'autre : c'est un conflit (la section « nous » est vide)."),
        (["a"], ["a", "x"], ["a", "y"], "deux lignes différentes ajoutées à la fin", ""),
        (["a"], ["a", "x"], ["a", "x"], "la même ligne ajoutée à la fin des deux côtés", ""),
        ([], ["x", "y"], ["z"], "un fichier créé des deux côtés, différemment", ""),
        ([], ["x"], [], "un fichier créé d'un seul côté", ""),
        (["a", "b"], ["a", "b"], [], "Malik vide le fichier, Inès n'y touche pas", ""),
        (["a", "b", "c"], ["x", "a", "b", "c"], ["a", "b", "c", "y"], "un ajout au début, un autre à la fin", ""),
    ]
    for b, n, e, quoi, conseil in cas:
        _une_fusion(f, b, n, e, f"fusion_trois_voies({b}, {n}, {e}) ({quoi})", conseil)

    alea = random.Random(179)
    for graine in range(200):
        base, nous, eux = generer_versions(alea.randrange(0, 40), graine=1790 + graine,
                                           taux=alea.choice([0.05, 0.15, 0.3]))
        _une_fusion(f, base, nous, eux, f"fusion_trois_voies(*generer_versions(…, graine={1790 + graine}))")


# ---------------------------------------------------------------------------- défi 180 : comparer_tables


def _absent(valeur) -> bool:
    try:
        return bool(pd.isna(valeur))
    except (TypeError, ValueError):
        return False


def _nombre(valeur) -> bool:
    return isinstance(valeur, numbers.Number) and not isinstance(valeur, (bool, np.bool_))


def _memes_valeurs(a, b) -> bool:
    if _absent(a) or _absent(b):
        return _absent(a) and _absent(b)
    if _nombre(a) and _nombre(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    if _nombre(a) or _nombre(b):
        return False
    return bool(a == b)


def _ref_tables(avant: pd.DataFrame, apres: pd.DataFrame, cle: str) -> dict:
    a, b = avant.set_index(cle), apres.set_index(cle)
    communes = [c for c in a.columns if c in b.columns]
    modifies = {}
    for k in sorted(set(a.index) & set(b.index)):
        changements = {c: (a.at[k, c], b.at[k, c]) for c in communes if not _memes_valeurs(a.at[k, c], b.at[k, c])}
        if changements:
            modifies[k] = changements
    return {"ajoutes": sorted(set(b.index) - set(a.index)), "supprimes": sorted(set(a.index) - set(b.index)),
            "modifies": modifies, "colonnes_ajoutees": sorted(c for c in b.columns if c not in a.columns),
            "colonnes_supprimees": sorted(c for c in a.columns if c not in b.columns)}


_CLES_TABLES = ("ajoutes", "supprimes", "modifies", "colonnes_ajoutees", "colonnes_supprimees")


def _resultat_tables(r, quoi: str) -> dict:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
    if not isinstance(r, dict) or set(r) != set(_CLES_TABLES):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : il faut un dict avec exactement les clés {list(_CLES_TABLES)}.")
    for nom in _CLES_TABLES:
        if nom != "modifies" and not isinstance(r[nom], list):
            raise AssertionError(f"{quoi} : r[{nom!r}] vaut {_court(r[nom])} ; il faut une liste TRIÉE (sorted(...)).")
    if not isinstance(r["modifies"], dict):
        raise AssertionError(f"{quoi} : r['modifies'] vaut {_court(r['modifies'])} ; il faut un dict "
                             "{clé: {colonne: (avant, après)}}.")
    for k, changements in r["modifies"].items():
        if not isinstance(changements, dict) or not changements:
            raise AssertionError(f"{quoi} : r['modifies'][{k!r}] vaut {_court(changements)} ; il faut un dict NON VIDE "
                                 "{colonne: (avant, après)} (un client sans changement n'apparaît pas).")
        for colonne, paire in changements.items():
            if not isinstance(paire, tuple) or len(paire) != 2:
                raise AssertionError(f"{quoi} : r['modifies'][{k!r}][{colonne!r}] vaut {_court(paire)} ; il faut une "
                                     "paire (valeur avant, valeur après).")
    return r


def _lisible(valeur):
    """Les valeurs NumPy affichées comme des valeurs Python : 75011 plutôt que np.int64(75011)."""
    if isinstance(valeur, dict):
        return {k: _lisible(v) for k, v in valeur.items()}
    if isinstance(valeur, tuple):
        return tuple(_lisible(v) for v in valeur)
    return valeur.item() if isinstance(valeur, np.generic) else valeur


def _comparer_tables(r: dict, attendu: dict, quoi: str, conseil: str = "") -> None:
    for nom in ("ajoutes", "supprimes", "colonnes_ajoutees", "colonnes_supprimees"):
        if list(r[nom]) != list(attendu[nom]):
            pas_trie = sorted(r[nom], key=str) == sorted(attendu[nom], key=str)
            raise AssertionError(f"{quoi} : r[{nom!r}] vaut {_court(r[nom])} au lieu de {_court(attendu[nom])}."
                                 + (" Triez la liste." if pas_trie else f" {conseil}".rstrip()))
    obtenus, attendus = r["modifies"], attendu["modifies"]
    en_trop = [k for k in obtenus if k not in attendus]
    if en_trop:
        k = en_trop[0]
        raise AssertionError(f"{quoi} : {k!r} est déclaré modifié ({_court(_lisible(obtenus[k]), 120)}), alors que rien n'a "
                             f"vraiment changé pour lui. {conseil}".rstrip())
    manquants = [k for k in attendus if k not in obtenus]
    if manquants:
        k = manquants[0]
        raise AssertionError(f"{quoi} : la modification de {k!r} n'est pas vue ({_court(_lisible(attendus[k]), 120)}). "
                             f"{conseil}".rstrip())
    for k, changements in attendus.items():
        if set(obtenus[k]) != set(changements):
            raise AssertionError(f"{quoi} : pour {k!r}, les colonnes modifiées sont {sorted(changements)}, vous dites "
                                 f"{sorted(obtenus[k])}. {conseil}".rstrip())
        for colonne, (x, y) in changements.items():
            ox, oy = obtenus[k][colonne]
            if not (_memes_valeurs(ox, x) and _memes_valeurs(oy, y)) or _nombre(ox) != _nombre(x) \
                    or _nombre(oy) != _nombre(y):
                raise AssertionError(f"{quoi} : pour {k!r}, la colonne {colonne!r} passe de {_lisible(x)!r} à "
                                     f"{_lisible(y)!r}, vous donnez {_lisible(ox)!r} → {_lisible(oy)!r} (dans cet "
                                     "ordre : avant, après).")


def _copie_table(t: pd.DataFrame) -> pd.DataFrame:
    return t.copy(deep=True)


def _table_intacte(copie: pd.DataFrame, t: pd.DataFrame, quoi: str, nom: str) -> None:
    if list(copie.columns) != list(t.columns) or not copie.index.equals(t.index) or not copie.equals(t) \
            or list(copie.dtypes) != list(t.dtypes):
        raise AssertionError(f"{quoi} a modifié la table {nom!r} reçue (index, ordre des lignes, colonnes ou "
                             "valeurs). Pas de `inplace=True` ni d'affectation dans la table de l'appelant : "
                             "set_index(...), sort_values(...) renvoient une NOUVELLE table.")


def _une_comparaison(f, avant: pd.DataFrame, apres: pd.DataFrame, cle: str, quoi: str, conseil: str = "") -> None:
    copies = (_copie_table(avant), _copie_table(apres))
    r = _appel(f, (avant, apres, cle), quoi, lignes_max=None)
    _table_intacte(copies[0], avant, quoi, "avant")
    _table_intacte(copies[1], apres, quoi, "apres")
    _comparer_tables(_resultat_tables(r, quoi), _ref_tables(avant, apres, cle), quoi, conseil)


def _deux(colonne: str, avant: list, apres: list, cles=("C1", "C2", "C3")) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.DataFrame({"client_id": list(cles), colonne: avant}),
            pd.DataFrame({"client_id": list(cles), colonne: apres}))


def verifier_comparer_tables(f) -> None:
    pieges = [
        (_deux("appels", [1.0, np.nan, 3.0], [1.0, np.nan, 3.0]), "un NaN des deux côtés",
         "NaN == NaN vaut False ! Deux valeurs absentes ne sont pas une modification : testez pd.isna(a) et "
         "pd.isna(b) avant de comparer."),
        (_deux("facture", [0.3, 1.0, 2.0], [0.1 + 0.2, 1.0, 2.0]), "0.3 et 0.1 + 0.2",
         "0.1 + 0.2 vaut 0.30000000000000004 : recalculée autrement, une facture diffère au 17e chiffre. Comparez les "
         "nombres avec math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)."),
        (_deux("code_postal", ["75011", "01000", "69003"], [75011, 1000, 69003]), "un texte devenu nombre",
         "\"01000\" et 1000 : le code postal de Bourg-en-Bresse a perdu son zéro, c'est une VRAIE régression (un "
         "read_csv sans dtype). Un texte n'est jamais égal à un nombre, même si str(1000) le laisse croire."),
        (_deux("anciennete", [3, 40, 12], [3.0, 40.0, 12.0]), "des entiers devenus des float",
         "3 et 3.0 sont le même nombre : une colonne d'entiers devient float dès qu'un NaN apparaît ailleurs. Ce "
         "n'est pas une modification (comparez des NOMBRES, pas des types)."),
        (_deux("contrat", ["Mensuel", None, "Annuel"], ["Mensuel", np.nan, "Annuel"]), "None d'un côté, NaN de l'autre",
         "None, NaN et NaT sont tous des valeurs absentes : pd.isna les voit toutes."),
        (_deux("resiliation", pd.to_datetime(["2026-01-05", None, "2026-03-01"]),
               pd.to_datetime(["2026-01-05", None, "2026-03-02"])), "des dates (et un NaT)",
         "NaT, comme NaN, n'est égal à rien : pd.isna le voit. Et le 1er mars devenu 2 mars est une vraie modification."),
        ((pd.DataFrame({"client_id": ["C1", "C2", "C3"], "contrat": ["Mensuel", "Annuel", "Deux ans"]}),
          pd.DataFrame({"client_id": ["C3", "C1", "C2"], "contrat": ["Deux ans", "Mensuel", "Annuel"]})),
         "les mêmes lignes dans un autre ordre",
         "Les lignes sont alignées par la CLÉ, pas par leur position : le nouveau code peut trier autrement. "
         "set_index(cle), puis comparez client par client."),
        ((pd.DataFrame({"client_id": ["C1", "C2"], "contrat": ["Mensuel", "Annuel"], "segment": ["A", "B"]}),
          pd.DataFrame({"client_id": ["C1", "C2"], "contrat": ["Mensuel", "Mensuel"], "score": [3, 1]})),
         "une colonne en moins, une colonne en plus",
         "Une colonne présente d'un seul côté va dans colonnes_ajoutees ou colonnes_supprimees ; on ne compare, "
         "client par client, que les colonnes COMMUNES."),
        ((pd.DataFrame({"id": [3, 1, 2], "x": [1.0, 2.0, 3.0]}), pd.DataFrame({"id": [1, 2, 4], "x": [2.0, 3.5, 0.0]})),
         "une clé numérique", "La clé n'est pas toujours un texte : ne supposez rien sur son type."),
        ((pd.DataFrame({"client_id": ["C5", "C1", "C3"], "x": [1, 2, 3]}),
          pd.DataFrame({"client_id": ["C4", "C3", "C2"], "x": [0, 3, 0]})), "deux clients ajoutés, deux supprimés",
         "Les listes de clés sont TRIÉES : sorted(...)."),
        ((pd.DataFrame({"client_id": pd.Series([], dtype=object), "x": pd.Series([], dtype=float)}),
          pd.DataFrame({"client_id": pd.Series([], dtype=object), "x": pd.Series([], dtype=float)})),
         "deux tables vides", ""),
    ]
    for (a, b), quoi, conseil in pieges:
        cle = "id" if "id" in a.columns else "client_id"
        _une_comparaison(f, a, b, cle, f"comparer_tables(…) ({quoi})", conseil)
    avant, apres = tables_exemple()
    _une_comparaison(f, avant, apres, "client_id", "comparer_tables(*tables_exemple(), cle=\"client_id\")",
                     "Une seule vraie modification : le contrat de C005. Relisez les pièges un par un : "
                     "lignes dans un autre ordre, NaN, 0.1 + 0.2, 12 et 12.0.")

    doublon = pd.DataFrame({"client_id": ["C1", "C2", "C1"], "x": [1, 2, 3]})
    simple = pd.DataFrame({"client_id": ["C1", "C2"], "x": [1, 2]})
    conseil = ("Une clé en double rend la comparaison ambiguë (lequel des deux C1 comparer ?), et une jointure "
               "fabriquerait des lignes en trop. Vérifiez avec .duplicated() et levez ValueError.")
    _doit_lever(f, (doublon, simple, "client_id"), "comparer_tables(avant avec C1 deux fois, apres, ...)", conseil)
    _doit_lever(f, (simple, doublon, "client_id"), "comparer_tables(avant, apres avec C1 deux fois, ...)", conseil)
    sans_cle = pd.DataFrame({"client_id": ["C1", None], "x": [1, 2]})
    _doit_lever(f, (sans_cle, simple, "client_id"), "comparer_tables(avant avec une clé absente, apres, ...)",
                "Une ligne sans clé (None, NaN) ne désigne aucun client : levez ValueError.")
    _doit_lever(f, (simple, simple, "numero"), "comparer_tables(avant, apres, cle=\"numero\") (colonne inexistante)",
                "La colonne clé doit exister dans les deux tables.", attendu=(KeyError, ValueError))

    alea = np.random.default_rng(180)
    for essai in range(60):
        n = int(alea.integers(0, 12))
        cles = [f"C{i:02d}" for i in alea.permutation(30)[:n]]
        a = pd.DataFrame({"client_id": cles, "contrat": alea.choice(["Mensuel", "Annuel"], n),
                          "facture": alea.integers(10, 100, n) / 10, "appels": alea.integers(0, 5, n)})
        b = a.sample(frac=1, random_state=essai).reset_index(drop=True)
        if n:
            b.loc[0, "facture"] = b.loc[0, "facture"] + alea.choice([0.0, 1e-12, 0.5])
            b = b.drop(index=list(range(1, min(n, 1 + int(alea.integers(0, 2))))))
        b = pd.concat([b, pd.DataFrame({"client_id": ["C99"], "contrat": ["Mensuel"], "facture": [9.9],
                                        "appels": [1]})], ignore_index=True)
        _une_comparaison(f, a, b, "client_id", f"comparer_tables(table tirée au hasard n°{essai})")


# ---------------------------------------------------------------------------- défi 181 : fusion_config

_ABSENTE = object()


def _identiques(a, b) -> bool:
    if a is _ABSENTE or b is _ABSENTE:
        return a is b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_identiques(a[c], b[c]) for c in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_identiques(x, y) for x, y in zip(a, b))
    return a == b


def _ref_config(base, nous: dict, eux: dict, chemin: str = "", conflits: list | None = None):
    racine = conflits is None
    conflits = [] if racine else conflits
    base = base if isinstance(base, dict) else {}
    fusion = {}
    for cle in [*nous, *(c for c in eux if c not in nous), *(c for c in base if c not in nous and c not in eux)]:
        b, n, e = base.get(cle, _ABSENTE), nous.get(cle, _ABSENTE), eux.get(cle, _ABSENTE)
        ici = f"{chemin}.{cle}" if chemin else str(cle)
        if _identiques(n, e) or _identiques(e, b):
            choix = n
        elif _identiques(n, b):
            choix = e
        elif isinstance(n, dict) and isinstance(e, dict):
            choix = _ref_config(b, n, e, ici, conflits)
        else:
            conflits.append(ici)
            choix = n
        if choix is not _ABSENTE:
            fusion[cle] = copy.deepcopy(choix)
    return (fusion, sorted(conflits)) if racine else fusion


def _objets(valeur) -> list:
    """Tous les dict et list contenus dans `valeur` (pour vérifier que rien n'est partagé)."""
    trouves = []
    if isinstance(valeur, (dict, list)):
        trouves.append(valeur)
        for x in (valeur.values() if isinstance(valeur, dict) else valeur):
            trouves += _objets(x)
    return trouves


def _une_config(f, base: dict, nous: dict, eux: dict, quoi: str, conseil: str = "") -> None:
    copies = [copy.deepcopy(x) for x in (base, nous, eux)]
    r = _appel(f, (base, nous, eux), quoi, lignes_max=500_000)
    for copie, recu, nom in zip(copies, (base, nous, eux), ("base", "nous", "eux")):
        if not _identiques(copie, recu):
            raise AssertionError(f"{quoi} a modifié la configuration « {nom} » reçue : travaillez sur un nouveau dict.")
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return` (renvoyez (fusion, conflits)).")
    if not isinstance(r, (tuple, list)) or len(r) != 2 or not isinstance(r[0], dict) or not isinstance(r[1], list):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : il faut une paire (dict fusionné, liste des conflits).")
    fusion, conflits = r
    attendu, conflits_attendus = _ref_config(base, nous, eux)
    if conflits != conflits_attendus:
        if sorted(map(str, conflits)) == conflits_attendus:
            raise AssertionError(f"{quoi} : triez la liste des conflits (sorted).")
        derniers = sorted(c.rsplit(".", 1)[-1] for c in conflits_attendus)
        if any("." in c for c in conflits_attendus) and sorted(map(str, conflits)) == derniers:
            raise AssertionError(f"{quoi} : conflits {_court(conflits)} au lieu de {conflits_attendus}. Le chemin d'une "
                                 "clé part de la racine, avec des points entre les sections : passez le chemin déjà "
                                 "parcouru à l'appel récursif.")
        raise AssertionError(f"{quoi} : conflits {_court(conflits)} au lieu de {conflits_attendus}. {conseil}".rstrip())
    if not _identiques(fusion, attendu):
        raise AssertionError(f"{quoi} donne {_court(fusion, 200)} au lieu de {_court(attendu, 200)}. {conseil}".rstrip())
    recus = {id(x) for source in (base, nous, eux) for x in _objets(source)}
    if any(id(x) in recus for x in _objets(fusion)):
        raise AssertionError(f"{quoi} : la configuration fusionnée PARTAGE un dict ou une liste avec une version reçue : "
                             "modifier l'une modifierait l'autre. Copiez ce que vous reprenez (copy.deepcopy).")


def verifier_fusion_config(f) -> None:
    base, nous, eux = config_exemple()
    _une_config(f, base, nous, eux, "fusion_config(*config_exemple())",
                "Inès et Malik ont changé des clés différentes de la même section : descendez dans les sections "
                "(récursion) au lieu de comparer des sections entières.")
    cas = [
        ({"a": 1}, {"a": None}, {"a": 1}, "Inès met une clé à None",
         "None (null en JSON) est une VRAIE valeur, différente d'une clé absente. `dict.get(cle)` les confond : "
         "utilisez un objet témoin, ou `cle in d`."),
        ({"a": None}, {}, {"a": None}, "Inès supprime une clé qui valait None",
         "La clé supprimée par Inès (elle valait None) doit disparaître : une clé absente n'est pas une clé à None."),
        ({"actif": 1}, {"actif": 1}, {"actif": True}, "Malik remplace 1 par True",
         "En Python, True == 1 ! Pour une configuration, ce sont deux valeurs différentes (un booléen, un nombre), "
         "et Malik a bien changé la sienne : comparez aussi les TYPES (type(a) is type(b))."),
        ({"seuil": 1}, {"seuil": 1}, {"seuil": 1.0}, "Malik remplace 1 par 1.0",
         "1 == 1.0, mais Malik a bien changé le type : comparez aussi les types."),
        ({"m": {"x": 1, "y": 2}}, {"m": {"x": 10, "y": 2}}, {"m": {"x": 1, "y": 20}}, "deux clés d'une même section",
         "Inès change m.x, Malik m.y : ce n'est pas un conflit. Quand les deux côtés ont changé une section (un dict), "
         "fusionnez-la clé par clé (récursion)."),
        ({"m": {"x": {"z": 1}}}, {"m": {"x": {"z": 2}}}, {"m": {"x": {"z": 3}}}, "un conflit en profondeur",
         "Le chemin d'une clé en conflit s'écrit avec des points : \"m.x.z\"."),
        ({"a": 1, "b": 2}, {"b": 2}, {"a": 5, "b": 2}, "Inès supprime ce que Malik modifie",
         "Supprimer d'un côté, modifier de l'autre : un conflit. On garde le choix de « nous » (ici : la clé absente)."),
        ({}, {"n": 1}, {"n": 1}, "la même clé ajoutée des deux côtés", ""),
        ({}, {"n": 1}, {"n": 2}, "la même clé ajoutée avec deux valeurs", "En conflit, on garde la valeur de « nous »."),
        ({"m": 1}, {"m": {"x": 1}}, {"m": 2}, "un nombre devenu section d'un côté, autre nombre de l'autre", ""),
        ({"l": [1, 2]}, {"l": [1, 2, 3]}, {"l": [0, 1, 2]}, "deux listes modifiées différemment",
         "Une liste se compare EN ENTIER (on ne fusionne pas leurs éléments) : deux listes différentes, un conflit."),
        ({"a": 1, "b": 2}, {"b": 2}, {"b": 2}, "la même clé supprimée des deux côtés", ""),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}, {"b": 2}, "Malik supprime une clé",
         "La clé supprimée par Malik doit disparaître : Inès n'y a pas touché."),
        ({"m": {"x": 1}}, {"m": {"x": 1}, "n": {"y": [1]}}, {"m": {"x": 2}}, "une section ajoutée, une modifiée", ""),
        ({"m": {"x": 1}}, {"m": {}}, {"m": {"x": 1, "y": 2}}, "une section vidée d'un côté, complétée de l'autre",
         "m.x supprimée par Inès, m.y ajoutée par Malik : les deux modifications se gardent, clé par clé."),
    ]
    for b, n, e, quoi, conseil in cas:
        _une_config(f, b, n, e, f"fusion_config({_court(b, 60)}, {_court(n, 60)}, {_court(e, 60)}) ({quoi})", conseil)

    for args, quoi in ((({"a": 1}, [("a", 1)], {"a": 1}), "fusion_config({...}, [(\"a\", 1)], {...})"),
                       ((None, {}, {}), "fusion_config(None, {}, {})")):
        _doit_lever(f, args, quoi, "Les trois versions sont des dict : levez TypeError sinon.", attendu=TypeError)

    alea = random.Random(181)
    valeurs = [0, 1, 1.0, True, False, None, "sms", [1, 2], {"x": 1}]

    def config(profondeur: int) -> dict:
        d = {}
        for cle in alea.sample("abcde", alea.randint(0, 4)):
            d[cle] = config(profondeur - 1) if profondeur and alea.random() < 0.3 else copy.deepcopy(alea.choice(valeurs))
        return d

    def modifier(d: dict, profondeur: int) -> dict:
        d = copy.deepcopy(d)
        for cle in list(d):
            tirage = alea.random()
            if tirage < 0.15:
                del d[cle]
            elif tirage < 0.3:
                d[cle] = copy.deepcopy(alea.choice(valeurs))
            elif isinstance(d[cle], dict) and profondeur:
                d[cle] = modifier(d[cle], profondeur - 1)
        if alea.random() < 0.3:
            d[alea.choice("abcdef")] = copy.deepcopy(alea.choice(valeurs))
        return d

    for essai in range(300):
        base = config(2)
        _une_config(f, base, modifier(base, 2), modifier(base, 2), f"fusion_config(configurations tirées au hasard "
                                                                     f"n°{essai})")


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"difference": 176, "diff_unifie": 177, "appliquer_patch": 178, "fusion_trois_voies": 179,
            "comparer_tables": 180, "fusion_config": 181}

SERIE = Serie(
    code="29",
    titre="Comparer et fusionner : diff, patch, fusion à trois voies, tables, configurations",
    module_eleve="defis.serie_29",
    module_corrige="defis.corrige_serie_29",
    defis=(
        Defi(176, "difference", "🟡", "—", verifier_difference),
        Defi(177, "diff_unifie", "🟢", "—", verifier_diff_unifie),
        Defi(178, "appliquer_patch", "🟡", "—", verifier_appliquer_patch),
        Defi(179, "fusion_trois_voies", "🔴", "—", verifier_fusion_trois_voies),
        Defi(180, "comparer_tables", "🟢", "6.2", verifier_comparer_tables),
        Defi(181, "fusion_config", "🔴", "—", verifier_fusion_config),
    ),
)

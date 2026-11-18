"""Vérifications automatiques de la série 25 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur se comporte comme le service marketing… et comme quelqu'un qui veut casser l'outil :

- il découpe et analyse les règles du marketing, puis des centaines de règles tirées au hasard, et compare
  chaque jeton, chaque arbre, chaque résultat à sa propre version. Quand ça diffère, il rejoue les
  versions boguées classiques (priorité inversée, « non » qui avale tout un « et », un nœud par « et » au
  lieu d'un seul, `<` puis `=` au lieu de `<=`, mot-clé reconnu au début d'un nom…) pour vous dire
  LAQUELLE vous avez écrite ;
- il glisse les pièges un par un : guillemets copiés depuis Word, texte jamais fermé, parenthèse en trop,
  5 000 « ou » à la suite, 5 000 parenthèses imbriquées (la limite de récursion de Python est 1 000),
  NaN qui n'est égal à rien, True qui vaut 1, des valeurs numpy, une région `x' OR 1=1 --`, une colonne
  qui s'appelle `order` ;
- au défi 154, son client « espion » note chaque colonne lue : il voit si votre « et » continue après un
  faux. Au défi 157, il exécute VOTRE requête sur une vraie base SQLite et compare les clients trouvés à
  ceux de l'interpréteur ;
- il arrête une boucle sans fin (il compte les lignes de VOTRE fichier exécutées) et l'explique.
"""

from __future__ import annotations

import copy
import difflib
import itertools
import math
import numbers
import random
import sqlite3
import sys
import time
import unicodedata

from .langage import (
    BOOLEENS,
    MOTS_CLES,
    OPERATEURS,
    ErreurSyntaxe,
    afficher_arbre,
    base_clients,
    clients_exemple,
    generer_arbre,
    generer_clients,
    regles_exemple,
    schema_clients,
)
from .verif import Defi, Serie

# ---------------------------------------------------------------------------- outils


def _decrire(exc: BaseException) -> str:
    texte = str(exc).split("\n")[0]
    if len(texte) > 160:
        texte = texte[:159] + "…"
    return f"{type(exc).__name__}({texte!r})" if texte else type(exc).__name__


def _court(valeur, n: int = 200) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


def _milliers(n: float) -> str:
    """12 345 678 : des espaces entre les milliers, à la française."""
    return f"{n:,.0f}".replace(",", " ")


def _sans_accents(texte: str) -> str:
    decompose = unicodedata.normalize("NFKD", str(texte))
    return "".join(c for c in decompose if not unicodedata.combining(c)).casefold()


def _montrer(texte: str, position: int, message: str = "") -> str:
    """Le morceau de règle autour de `position`, avec un « ^ » dessous (indenté pour l'affichage du correcteur)."""
    debut, fin = max(0, position - 45), min(len(texte), position + 45)
    avant = "…" if debut > 0 else ""
    extrait = "".join(" " if c in "\t\n\r" else c for c in texte[debut:fin])
    apres = "…" if fin < len(texte) else ""
    return f"          {avant}{extrait}{apres}\n          {' ' * (len(avant) + position - debut)}^ {message}".rstrip()


class _BoucleSansFin(BaseException):
    """Levée dans VOTRE fonction quand elle exécute trop de lignes (BaseException : `except Exception` ne l'attrape pas)."""


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


_CONSEIL_RECURSION = ("Python s'arrête à 1 000 appels imbriqués. Une suite de conditions au même niveau se lit avec une "
                      "BOUCLE (`while voir().type == \"OU\"`), pas avec un appel récursif par condition.")
_CONSEIL_PROFONDEUR = ("Une règle venue de l'extérieur peut imbriquer des milliers de parenthèses : comptez la profondeur "
                       "(+1 en entrant dans une parenthèse ou un `non`, -1 en sortant) et levez ErreurSyntaxe au-delà "
                       "de 100, AVANT que Python ne lève RecursionError (sur un serveur, c'est une requête qui plante).")
_CONSEIL_BOUCLE = ("Vérifiez que chaque tour de boucle fait AVANCER l'indice (ou consomme un jeton) : un caractère que "
                   "vous ne savez pas traiter doit lever ErreurSyntaxe, pas être laissé sur place.")


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = 2_000_000):
    """f(*args), en transformant toute erreur en message lisible."""
    limite = sys.getrecursionlimit()
    try:
        return _trace(f, lambda: f(*args), lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} ne s'arrête pas : votre fonction a exécuté plus de {_milliers(lignes_max)} lignes "
                             f"(elle a été arrêtée). {conseil or _CONSEIL_BOUCLE}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError. {conseil or _CONSEIL_RECURSION}") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None
    finally:
        if sys.getrecursionlimit() != limite:
            sys.setrecursionlimit(limite)
            raise AssertionError(f"{quoi} modifie la limite de récursion (`sys.setrecursionlimit`) : c'est interdit. "
                                 f"{_CONSEIL_PROFONDEUR}")


def _doit_lever(f, args: tuple, attendues, quoi: str, conseil: str = "", mots: tuple = ()) -> BaseException:
    """Exige que f(*args) lève une des exceptions `attendues`, avec les `mots` dans le message."""
    attendues = attendues if isinstance(attendues, tuple) else (attendues,)
    noms = " ou ".join(e.__name__ for e in attendues)
    try:
        resultat = _trace(f, lambda: f(*args), 2_000_000)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} ne s'arrête pas (votre fonction a été arrêtée) : elle devrait lever {noms}. "
                             f"{conseil}".rstrip()) from None
    except attendues as exc:
        absents = [m for m in mots if _sans_accents(m) not in _sans_accents(str(exc))]
        if absents:
            raise AssertionError(f"{quoi} lève bien {type(exc).__name__}, mais son message ({str(exc)!r}) ne contient "
                                 f"pas {' ni '.join(repr(m) for m in absents)}. Un message d'erreur doit dire OÙ chercher.") \
                from None
        return exc
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {noms}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {noms}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _erreur_syntaxe(f, texte, position: int, quoi: str, mots: tuple = (), conseil: str = "") -> ErreurSyntaxe:
    """Exige ErreurSyntaxe à `position` (et les `mots` dans son message)."""
    try:
        resultat = _trace(f, lambda: f(texte), 3_000_000)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} ne s'arrête pas (votre fonction a été arrêtée) : elle devrait lever ErreurSyntaxe. "
                             f"{conseil or _CONSEIL_BOUCLE}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError au lieu de ErreurSyntaxe. {_CONSEIL_PROFONDEUR}") from None
    except ErreurSyntaxe as exc:
        pos = getattr(exc, "position", None)
        if type(pos) is not int:
            raise AssertionError(f"{quoi} lève ErreurSyntaxe, mais sa position vaut {pos!r}. Écrivez "
                                 f"`ErreurSyntaxe(message, position)` : d'abord le message, puis l'indice (un int) du "
                                 f"caractère fautif.") from None
        if pos != position:
            raise AssertionError(f"{quoi} lève bien ErreurSyntaxe, mais à la position {pos} au lieu de {position}. "
                                 f"Vous montrez :\n{_montrer(texte, pos, str(exc))}\n        alors qu'il faut montrer :\n"
                                 f"{_montrer(texte, position)}\n        {conseil}".rstrip()) from None
        absents = [m for m in mots if _sans_accents(m) not in _sans_accents(str(exc))]
        if absents:
            raise AssertionError(f"{quoi} lève bien ErreurSyntaxe à la bonne position, mais son message ({str(exc)!r}) "
                                 f"ne contient pas {' ni '.join(repr(m) for m in absents)}. {conseil}".rstrip()) from None
        return exc
    except Exception as exc:
        indice = (" `ErreurSyntaxe(message, position)` est une ValueError qui retient la position : levez-la, pas une "
                  "ValueError simple." if type(exc) is ValueError else "")
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de ErreurSyntaxe.{indice} {conseil}".rstrip()) \
            from None
    raise AssertionError(f"{quoi} devrait lever ErreurSyntaxe à la position {position} :\n{_montrer(texte, position)}\n"
                         f"        mais renvoie {_court(resultat)}. {conseil}".rstrip())


# ---------------------------------------------------------------------------- références

_CHIFFRES = "0123456789"
_TYPO = "“”«»‘’„"


def _ref_decouper(texte: str) -> list[tuple]:
    if not isinstance(texte, str):
        raise TypeError("une règle est un str")
    jetons, i, n = [], 0, len(texte)
    while i < n:
        c = texte[i]
        if c.isspace():
            i += 1
        elif c in "(),":
            jetons.append((c, c, i))
            i += 1
        elif c in "<>=!":
            deux = texte[i:i + 2]
            if deux in ("<=", ">=", "!=", "==", "<>"):
                jetons.append(("OP", {"==": "=", "<>": "!="}.get(deux, deux), i))
                i += 2
            elif c == "!":
                raise ErreurSyntaxe("« ! » seul", i)
            else:
                jetons.append(("OP", c, i))
                i += 1
        elif c in _CHIFFRES or (c == "-" and i + 1 < n and texte[i + 1] in _CHIFFRES):
            j = i + 1
            while j < n and texte[j] in _CHIFFRES:
                j += 1
            if j + 1 < n and texte[j] == "." and texte[j + 1] in _CHIFFRES:
                j += 1
                while j < n and texte[j] in _CHIFFRES:
                    j += 1
                jetons.append(("NOMBRE", float(texte[i:j]), i))
            else:
                jetons.append(("NOMBRE", int(texte[i:j]), i))
            i = j
        elif c in "\"'":
            morceaux, j = [], i + 1
            while True:
                if j >= n:
                    raise ErreurSyntaxe("texte jamais fermé (guillemet)", i)
                if texte[j] == "\\" and j + 1 < n:
                    morceaux.append(texte[j + 1])
                    j += 2
                elif texte[j] == c:
                    break
                else:
                    morceaux.append(texte[j])
                    j += 1
            jetons.append(("TEXTE", "".join(morceaux), i))
            i = j + 1
        elif c.isalpha() or c == "_":
            j = i + 1
            while j < n and (texte[j].isalnum() or texte[j] == "_"):
                j += 1
            mot = texte[i:j]
            bas = mot.lower()
            if bas in MOTS_CLES:
                jetons.append((MOTS_CLES[bas], bas, i))
            elif bas in BOOLEENS:
                jetons.append(("BOOLEEN", BOOLEENS[bas], i))
            else:
                jetons.append(("NOM", mot, i))
            i = j
        elif c in _TYPO:
            raise ErreurSyntaxe("guillemet typographique", i)
        else:
            raise ErreurSyntaxe(f"caractère inattendu {c!r}", i)
    jetons.append(("FIN", None, n))
    return jetons


def _ref_analyser(texte: str, priorite: str = "normale", assoc: str = "nary", aplatir: bool = False,
                  non_large: bool = False, emballer: bool = False, feuille: str = "op_col") -> tuple:
    """L'analyseur de référence et, avec des options, les versions boguées classiques (pour les reconnaître)."""
    jetons = _ref_decouper(texte)
    pos, profondeur = 0, 0
    faible, fort = ("OU", "ET") if priorite == "normale" else ("ET", "OU")

    def voir():
        return jetons[pos]

    def avancer():
        nonlocal pos
        pos += 1
        return jetons[pos - 1]

    def entrer(jeton):
        nonlocal profondeur
        profondeur += 1
        if profondeur > 100:
            raise ErreurSyntaxe("trop de niveaux imbriqués", jeton[2])

    def combiner(sorte: str, enfants: list) -> tuple:
        if len(enfants) == 1 and not emballer:
            return enfants[0]
        if assoc == "gauche":
            arbre = enfants[0]
            for e in enfants[1:]:
                arbre = (sorte, arbre, e)
            return arbre
        if assoc == "droite":
            arbre = enfants[-1]
            for e in reversed(enfants[:-1]):
                arbre = (sorte, e, arbre)
            return arbre
        if aplatir:
            enfants = [x for e in enfants for x in (e[1:] if e[0] == sorte else (e,))]
        return (sorte, *enfants)

    def niveau_faible():
        enfants = [niveau_fort()]
        while voir()[0] == faible:
            avancer()
            enfants.append(niveau_fort())
        return combiner(faible.lower(), enfants)

    def niveau_fort():
        enfants = [non()]
        while voir()[0] == fort:
            avancer()
            enfants.append(non())
        return combiner(fort.lower(), enfants)

    def non():
        nonlocal profondeur
        if voir()[0] == "NON":
            entrer(avancer())
            interieur = niveau_fort() if non_large else non()
            profondeur -= 1
            return ("non", interieur)
        return primaire()

    def valeur():
        jeton = voir()
        if jeton[0] in ("NOMBRE", "TEXTE", "BOOLEEN"):
            avancer()
            return jeton[1]
        raise ErreurSyntaxe("valeur attendue (guillemet ?)" if jeton[0] == "NOM" else "valeur attendue", jeton[2])

    def primaire():
        nonlocal profondeur
        jeton = voir()
        if jeton[0] == "(":
            entrer(avancer())
            interieur = niveau_faible()
            if voir()[0] == "FIN":
                raise ErreurSyntaxe("parenthèse jamais fermée", jeton[2])
            if voir()[0] != ")":
                raise ErreurSyntaxe("« ) » attendue", voir()[2])
            avancer()
            profondeur -= 1
            return interieur
        if jeton[0] == "NOM":
            avancer()
            suivant = voir()
            if suivant[0] == "OP":
                avancer()
                v = valeur()
                return (jeton[1], suivant[1], v) if feuille == "col_op" else (suivant[1], jeton[1], v)
            if suivant[0] == "DANS":
                avancer()
                ouvrante = voir()
                if ouvrante[0] != "(":
                    raise ErreurSyntaxe("« ( » attendue", ouvrante[2])
                avancer()
                valeurs = [valeur()]
                while voir()[0] == ",":
                    avancer()
                    valeurs.append(valeur())
                if voir()[0] == "FIN":
                    raise ErreurSyntaxe("parenthèse jamais fermée", ouvrante[2])
                if voir()[0] != ")":
                    raise ErreurSyntaxe("« , » ou « ) » attendue", voir()[2])
                avancer()
                return ("dans", jeton[1], tuple(valeurs))
            raise ErreurSyntaxe("opérateur attendu", suivant[2])
        if jeton[0] in ("NOMBRE", "TEXTE", "BOOLEEN"):
            raise ErreurSyntaxe("une condition commence par une colonne", jeton[2])
        if jeton[0] == "FIN":
            raise ErreurSyntaxe("règle vide" if pos == 0 else "règle incomplète", jeton[2])
        raise ErreurSyntaxe("condition attendue", jeton[2])

    arbre = niveau_faible()
    if voir()[0] != "FIN":
        raise ErreurSyntaxe("en trop", voir()[2])
    return arbre


def _inconnu(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _genre(v) -> str:
    if isinstance(v, bool):
        return "booleen"
    if isinstance(v, numbers.Real):
        return "nombre"
    if isinstance(v, str):
        return "texte"
    raise TypeError(f"valeur de type {type(v).__name__}")


_COMPARER = {"=": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b,
             "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def _ref_evaluer(arbre: tuple, client) -> bool | None:
    sorte = arbre[0]
    if sorte in ("et", "ou"):
        decisif = sorte == "ou"  # True arrête un « ou », False arrête un « et »
        resultat = not decisif
        for enfant in arbre[1:]:
            r = _ref_evaluer(enfant, client)
            if r is decisif:
                return decisif
            if r is None:
                resultat = None
        return resultat
    if sorte == "non":
        r = _ref_evaluer(arbre[1], client)
        return None if r is None else not r
    if sorte == "dans":
        _, colonne, valeurs = arbre
        v = client[colonne]
        if _inconnu(v):
            return None
        if any(_genre(v) != _genre(w) for w in valeurs):
            raise TypeError(colonne)
        return any(bool(v == w) for w in valeurs)
    if sorte in _COMPARER:
        op, colonne, attendu = arbre
        v = client[colonne]
        if _inconnu(v):
            return None
        if _genre(v) != _genre(attendu) or (_genre(v) == "booleen" and op not in ("=", "!=")):
            raise TypeError(colonne)
        return bool(_COMPARER[op](v, attendu))
    raise ValueError(sorte)


_PRIORITE = {"ou": 1, "et": 2}


def _ref_valeur(v) -> str:
    if isinstance(v, bool):
        return "vrai" if v else "faux"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(v)
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise ValueError(v)


def _nom_valide(colonne) -> bool:
    return (isinstance(colonne, str) and bool(colonne) and (colonne[0].isalpha() or colonne[0] == "_")
            and all(c.isalnum() or c == "_" for c in colonne)
            and colonne.lower() not in MOTS_CLES and colonne.lower() not in BOOLEENS)


def _ref_texte(arbre: tuple, parentheses: str = "minimal") -> str:
    """minimal (attendu), strict (oublie « et » dans « et »), compose (toujours entre parenthèses)."""
    sorte = arbre[0]
    if sorte in _PRIORITE:
        if len(arbre) < 3:
            raise ValueError(arbre)
        morceaux = []
        for enfant in arbre[1:]:
            t = _ref_texte(enfant, parentheses)
            if enfant[0] in _PRIORITE and (
                    parentheses == "compose"
                    or (parentheses == "minimal" and _PRIORITE[enfant[0]] <= _PRIORITE[sorte])
                    or (parentheses == "strict" and _PRIORITE[enfant[0]] < _PRIORITE[sorte])):
                t = f"({t})"
            morceaux.append(t)
        return f" {sorte} ".join(morceaux)
    if sorte == "non":
        t = _ref_texte(arbre[1], parentheses)
        return f"non ({t})" if arbre[1][0] in _PRIORITE else f"non {t}"
    if sorte == "dans":
        if not arbre[2] or not _nom_valide(arbre[1]):
            raise ValueError(arbre)
        return f"{arbre[1]} dans ({', '.join(_ref_valeur(v) for v in arbre[2])})"
    if sorte in OPERATEURS:
        if not _nom_valide(arbre[1]):
            raise ValueError(arbre)
        return f"{arbre[1]} {sorte} {_ref_valeur(arbre[2])}"
    raise ValueError(sorte)


def _ref_problemes(arbre: tuple, schema: dict) -> list[tuple]:
    """Les problèmes attendus, sous forme structurée : ("inconnue", col, suggestion), ("genre", col, g, gv), ("ordre", col, op)."""
    problemes, pile = [], [arbre]
    while pile:
        noeud = pile.pop()
        sorte = noeud[0]
        if sorte in ("et", "ou", "non"):
            pile.extend(reversed(noeud[1:]))
            continue
        if sorte == "dans":
            _, colonne, valeurs = noeud
        elif sorte in OPERATEURS:
            _, colonne, v = noeud
            valeurs = (v,)
        else:
            raise ValueError(sorte)
        if colonne not in schema:
            proches = ([c for c in schema if c.lower() == colonne.lower()]
                       or difflib.get_close_matches(colonne, list(schema), n=1))
            problemes.append(("inconnue", colonne, proches[0] if proches else None))
            continue
        fautive = next((v for v in valeurs if _genre(v) != schema[colonne]), None)
        if fautive is not None:
            problemes.append(("genre", colonne, schema[colonne], _genre(fautive)))
        elif schema[colonne] == "booleen" and sorte not in ("=", "!=", "dans"):
            problemes.append(("ordre", colonne, sorte))
    return problemes


def _ref_sql(arbre: tuple, parentheses: bool = True) -> tuple[str, list]:
    params: list = []

    def sql(noeud):
        sorte = noeud[0]
        if sorte in _PRIORITE:
            texte = (" AND " if sorte == "et" else " OR ").join(sql(e) for e in noeud[1:])
            return f"({texte})" if parentheses else texte
        if sorte == "non":
            return f"NOT ({sql(noeud[1])})" if parentheses else f"NOT {sql(noeud[1])}"
        colonne = '"' + noeud[1].replace('"', '""') + '"'
        if sorte == "dans":
            params.extend(noeud[2])
            return f"{colonne} IN ({', '.join('?' * len(noeud[2]))})"
        params.append(noeud[2])
        return f"{colonne} {'<>' if sorte == '!=' else sorte} ?"

    return sql(arbre), params


# ============================================================================ Défi 152 : decouper


def _exiger_jetons(obtenu, quoi: str) -> list[tuple]:
    if not isinstance(obtenu, list):
        raise AssertionError(f"{quoi} renvoie {type(obtenu).__name__} au lieu d'une LISTE de jetons.")
    for jeton in obtenu:
        if not (isinstance(jeton, tuple) and len(jeton) == 3):
            raise AssertionError(f"{quoi} contient {_court(jeton)} : chaque élément doit être un "
                                 f"Jeton(type, valeur, position).")
    return [tuple(j) for j in obtenu]


def _meme_jeton(a: tuple, b: tuple) -> bool:
    # 6 == 6.0 et True == 1 : pour les nombres et les booléens, on compare aussi les types
    return a == b and (not isinstance(b[1], (int, float)) or type(a[1]) is type(b[1]))


def _expliquer_jetons(obtenu: list, attendu: list, texte: str, quoi: str) -> None:
    if len(obtenu) == len(attendu) and all(_meme_jeton(a, b) for a, b in zip(obtenu, attendu)):
        return
    if obtenu == attendu[:-1]:
        raise AssertionError(f"{quoi} oublie le dernier jeton : il faut toujours finir par "
                             f"Jeton('FIN', None, {len(texte)}) (la position est la longueur du texte). L'analyseur du "
                             f"défi 153 s'en sert pour savoir que la règle est finie.")
    k = next((i for i, (a, b) in enumerate(zip(obtenu, attendu)) if not _meme_jeton(a, b)), min(len(obtenu), len(attendu)))
    if k >= len(attendu):
        raise AssertionError(f"{quoi} donne des jetons en trop après la fin : {_court(obtenu[k:])}. Le dernier jeton est "
                             f"FIN.")
    a = attendu[k]
    if k >= len(obtenu):
        raise AssertionError(f"{quoi} s'arrête trop tôt : il manque {_court(attendu[k:])}.")
    o = obtenu[k]
    debut = f"{quoi} : au jeton n°{k + 1}, vous donnez {o}, il faut {a}."
    conseil = ""
    if o[0] == "OP" and a[0] == "OP" and o[1] in ("==", "<>"):
        conseil = "`==` et `<>` sont acceptés, mais leur VALEUR est l'opérateur de la règle : \"=\" et \"!=\"."
    elif a[0] == "OP" and o[0] == "OP" and len(a[1]) == 2 and o[1] == a[1][0]:
        conseil = (f"« {a[1]} » est UN opérateur : regardez les deux caractères `texte[i:i + 2]` AVANT le premier "
                   f"seul (on prend toujours le jeton le plus long).")
    elif a[0] == "NOM" and o[0] in ("ET", "OU", "NON", "DANS", "BOOLEEN"):
        conseil = (f"« {a[1]} » est un nom de colonne qui COMMENCE comme un mot-clé : lisez le mot entier (lettres, "
                   f"chiffres, `_`), puis seulement regardez si c'est un mot-clé. Jamais `texte.startswith('et', i)`.")
    elif a[0] in ("ET", "OU", "NON", "DANS", "BOOLEEN") and o[0] == "NOM":
        conseil = ("les mots-clés s'écrivent en majuscules aussi (`ET`, `Vrai`) : comparez `mot.lower()` aux mots-clés, "
                   "et `vrai` / `faux` donnent un jeton BOOLEEN de valeur True / False.")
    elif a[0] in ("ET", "OU", "NON", "DANS") and o[0] == a[0] and o[1] != a[1]:
        conseil = "la valeur d'un mot-clé est le mot en minuscules (« et », même écrit « ET »)."
    elif a[0] == "NOMBRE" and o[0] == "NOMBRE" and o[1] == a[1] and type(o[1]) is not type(a[1]):
        conseil = "sans point, un nombre est un int (`int(...)`) ; avec un point, un float."
    elif a[0] == "NOMBRE" and isinstance(a[1], (int, float)) and a[1] < 0 and o[0] != "NOMBRE":
        conseil = "un `-` collé devant un chiffre fait partie du nombre (`-3`)."
    elif a[0] == "TEXTE" and o[0] == "TEXTE" and isinstance(o[1], str) and o[1][1:-1] == a[1]:
        conseil = "la valeur d'un texte est SANS ses guillemets."
    elif a[0] == "TEXTE" and o[0] == "TEXTE" and "\\" in texte:
        conseil = ("une barre oblique inverse protège le caractère qui la suit : on garde ce caractère (sans la barre) "
                   "et on saute les deux. `\\\"` ne ferme donc pas le texte.")
    elif (o[0], o[1]) == (a[0], a[1]) and o[2] != a[2]:
        conseil = (f"le type et la valeur sont bons, mais la position est {o[2]} au lieu de {a[2]} : c'est l'indice (à "
                   f"partir de 0) du PREMIER caractère du jeton.\n{_montrer(texte, a[2])}")
    elif isinstance(o[1], str) and o[1].strip() == "" and o[1] != "":
        conseil = "les espaces ne produisent aucun jeton (`c.isspace()` : espace, tabulation, retour à la ligne…)."
    elif a[0] == "TEXTE":
        conseil = "un texte va d'un guillemet (double OU simple) au même guillemet, et peut contenir n'importe quoi."
    raise AssertionError(f"{debut} {conseil}".rstrip())


def _verifier_jetons(f, texte: str, quoi: str | None = None) -> None:
    quoi = quoi or f"decouper({texte!r})"
    obtenu = _exiger_jetons(_appel(f, (texte,), quoi), quoi)
    _expliquer_jetons(obtenu, _ref_decouper(texte), texte, quoi)


class _Texte(str):
    """Une règle « espionne » : compte les caractères recopiés par les tranches `texte[a:b]`."""

    copies = 0

    def __getitem__(self, cle):
        morceau = str.__getitem__(self, cle)
        if isinstance(cle, slice):
            _Texte.copies += len(morceau)
            return _Texte(morceau)
        return morceau


_CAS_DECOUPER = [
    "anciennete <= 6",
    "",
    "   ",
    'region dans ("Nord", "Est") ET fibre = vrai',
    "a<=1",
    "a>=-3.5 et b<-2",
    "a == 1 ou b <> 2 ou c != 3 ou d < 4 ou e > 5 ou f = 6 ou g >= 7",
    "etat et ouverture ou nonne non dansant dans fauxfuyant vraiment",
    "ET Ou NON Dans VRAI Faux",
    "Anciennete durée_2 _x é Région2",
    "6 79.5 -3 007 0.25 100",
    "'mensuel' \"l'été\" 'il a dit \"non\"' \"\"",
    '"a\\"b" "a\\\\b" "x et y <= 3 (ou non)"',
    "a\t=\n1\u00a0et\r\nb = 2",
]


def verifier_decouper(f) -> None:
    # 1. Les cas simples, puis chaque piège un par un.
    for texte in _CAS_DECOUPER:
        _verifier_jetons(f, texte)
    for texte in regles_exemple().values():
        _verifier_jetons(f, texte)

    # 2. Les erreurs, à la bonne position.
    _erreur_syntaxe(f, 'contrat = "mensuel', 10, "decouper('contrat = \"mensuel')", ("guillemet",),
                    "Un texte jamais fermé se signale au guillemet OUVRANT : retenez sa position avant de chercher la fin.")
    _erreur_syntaxe(f, "contrat = 'mensuel\"", 10, "decouper('contrat = \\'mensuel\"')", ("guillemet",),
                    "Un texte commencé par ' se termine par ', pas par \".")
    _erreur_syntaxe(f, 'a = "x\\"', 4, "decouper('a = \"x\\\\\"')", ("guillemet",),
                    "Le dernier guillemet est protégé par la barre oblique : il ne ferme pas le texte.")
    for typo in ('“mensuel”', '« mensuel »'):
        _erreur_syntaxe(f, f"contrat = {typo}", 10, f"decouper('contrat = {typo}')", ("guillemet",),
                        "Les guillemets typographiques viennent d'un copier-coller depuis Word : le message doit le dire, "
                        "sinon personne ne verra la différence avec des guillemets droits.")
    for texte, position in (("a # 1", 2), ("a ! 1", 2), ("a = 1.2.3", 7), ("a = 12.", 6), ("a - 1", 2)):
        _erreur_syntaxe(f, texte, position, f"decouper({texte!r})",
                        conseil="Un caractère qui ne commence aucun jeton lève ErreurSyntaxe à SA position.")
    for mauvais in (None, 42, b"a = 1", ["a", "=", "1"]):
        _doit_lever(f, (mauvais,), TypeError, f"decouper({mauvais!r})",
                    "Une règle est un `str` : vérifiez `isinstance(texte, str)` en premier.")

    # 3. Des règles tirées au hasard (écrites par la référence).
    for graine in range(40):
        _verifier_jetons(f, _ref_texte(generer_arbre(graine, 4)), f"decouper(<règle au hasard n°{graine}>)")

    # 4. Pas de `texte = texte[1:]` : on compte les caractères recopiés.
    regle = _Texte(" et ".join(f'appels_support >= {i % 9} ou region = "Nord"' for i in range(400)))
    _Texte.copies = 0
    obtenu = _exiger_jetons(_appel(f, (regle,), "decouper(<une règle de 18 000 caractères>)", lignes_max=5_000_000),
                            "decouper(<une règle de 18 000 caractères>)")
    if _Texte.copies > 20 * len(regle):
        raise AssertionError(f"decouper recopie {_milliers(_Texte.copies)} caractères pour lire une règle de "
                             f"{_milliers(len(regle))} : "
                             f"`texte = texte[i:]` (ou `re.match(motif, texte[i:])`) recopie TOUT le reste du texte à "
                             f"chaque jeton, soit O(n²). Gardez le texte entier et faites avancer un indice ; avec `re`, "
                             f"`motif.match(texte, i)` commence à la position i sans rien recopier.")
    _expliquer_jetons(obtenu, _ref_decouper(str(regle)), "<règle longue>", "decouper(<une règle de 18 000 caractères>)")

    # 5. Une vraie taille.
    regle = " ou ".join(f'(contrat = "mensuel" et anciennete <= {i % 72})' for i in range(5000))
    debut = time.perf_counter()
    obtenu = _appel(f, (regle,), "decouper(<200 000 caractères>)", lignes_max=None)
    duree = time.perf_counter() - debut
    _expliquer_jetons(_exiger_jetons(obtenu, "decouper(<200 000 caractères>)"), _ref_decouper(regle), regle,
                      "decouper(<200 000 caractères>)")
    if duree > 3.0:
        raise AssertionError(f"decouper est juste, mais 200 000 caractères prennent {duree:.1f} s (environ 0,2 s attendu). "
                             f"Une copie du reste du texte à chaque jeton (`texte[i:]`, `lstrip()`, `split(maxsplit=1)`) "
                             f"rend le découpage quadratique : faites avancer un indice.")


# ============================================================================ Défi 153 : analyser


def _en_tuples(x):
    if isinstance(x, (list, tuple)):
        return tuple(_en_tuples(e) for e in x)
    return x


def _memes_types(a, b) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_memes_types(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


_VARIANTES_ANALYSE = [
    (dict(priorite="inversee"),
     "Chez vous, « ou » passe avant « et ». C'est l'inverse : `et` lie plus fort, comme × avant + (`a ou b et c` se lit "
     "`a ou (b et c)`). La fonction du niveau le plus FAIBLE (`ou`) appelle celle du niveau plus fort (`et`), qui "
     "appelle `non`."),
    (dict(non_large=True),
     "Votre « non » s'applique à tout le « et » qui le suit. Il ne prend que la condition qui le suit immédiatement : "
     "`non()` appelle `non()` (pour `non non …`) ou `primaire()`, pas `et()`. Pour nier un groupe, on écrit "
     "`non (a et b)`."),
    (dict(assoc="gauche"),
     "Vous construisez un nœud par « et » : ('et', ('et', a, b), c). L'arbre attendu met toutes les conditions d'une même "
     "suite dans UN nœud : ('et', a, b, c). Rangez-les dans une liste pendant la boucle, puis `(\"et\", *enfants)`."),
    (dict(assoc="droite"),
     "Vous construisez ('et', a, ('et', b, c)) : un appel récursif par « et ». L'arbre attendu met toutes les conditions "
     "d'une même suite dans UN nœud ('et', a, b, c), et une boucle `while` évite la RecursionError sur 5 000 conditions."),
    (dict(aplatir=True),
     "Vous fusionnez « a et (b et c) » en un seul « et ». Les parenthèses écrites par l'utilisateur doivent rester "
     "visibles dans l'arbre : ne regroupez que les conditions d'une même boucle, pas celles qui viennent d'une parenthèse."),
    (dict(emballer=True),
     "Un « et » (ou un « ou ») à une seule condition n'en est pas un : quand la boucle n'a trouvé qu'un enfant, "
     "renvoyez cet enfant seul."),
    (dict(feuille="col_op"),
     "Une comparaison s'écrit (opérateur, colonne, valeur) : ('<', 'anciennete', 6), pas ('anciennete', '<', 6)."),
]


def _dessin(arbre) -> str:
    try:
        texte = afficher_arbre(arbre)
    except Exception:
        return ""
    return "" if texte.count("\n") > 12 else "\n" + "\n".join("          " + ligne for ligne in texte.split("\n"))


def _verifier_arbre(f, texte: str, quoi: str | None = None, lignes_max: int = 3_000_000, conseil: str = "") -> None:
    quoi = quoi or f"analyser({texte!r})"
    obtenu = _appel(f, (texte,), quoi, conseil=conseil, lignes_max=lignes_max)
    attendu = _ref_analyser(texte)
    if _memes_types(obtenu, attendu):
        return
    if obtenu == attendu:
        raise AssertionError(f"{quoi} renvoie le bon arbre, mais avec des valeurs du mauvais type ({_court(obtenu)}) : un "
                             f"nombre écrit sans point est un int, `vrai` est True. Vérifiez d'abord votre `decouper`.")
    if _en_tuples(obtenu) == attendu:
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} : l'arbre est fait de TUPLES (y compris les valeurs de "
                             f"« dans » : ('dans', 'region', ('Nord', 'Est'))), pas de listes. `tuple(valeurs)`.")
    for options, explication in _VARIANTES_ANALYSE:
        try:
            variante = _ref_analyser(texte, **options)
        except Exception:
            continue
        if variante != attendu and _en_tuples(obtenu) == variante:
            raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {_court(attendu)}. {explication}")
    raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {_court(attendu)}.{_dessin(attendu)}")


_CAS_ANALYSER = [
    "anciennete < 6",
    'contrat = "mensuel" et anciennete < 6',
    "a = 1 ou b = 2 et c = 3",
    "a = 1 et b = 2 ou c = 3",
    "non a = 1 et b = 2",
    "non (a = 1 et b = 2)",
    "non non a = 1",
    "a = 1 et b = 2 et c = 3",
    "a = 1 ou b = 2 ou c = 3 ou d = 4",
    "a = 1 et (b = 2 et c = 3)",
    "(a = 1 et b = 2) et c = 3",
    "((a = 1))",
    'region dans ("Nord")',
    'region dans ("Nord", "Est", "Sud") ou region dans (1, 2.5, vrai)',
    "fibre = vrai et fibre != FAUX",
    'x = -3.5 ou y >= 0 et z = "et ou non"',
    "a = 1 et non (b = 2 ou c = 3) et d = 4",
]

_ERREURS_ANALYSER = [
    # (texte, mots attendus dans le message, conseil)
    ("", ("vide",), "Une règle vide se signale à la fin du texte (position 0 ici)."),
    ("   ", ("vide",), "Que des espaces, c'est une règle vide : position len(texte), le jeton FIN."),
    ("(a = 1", ("parenthèse",), "Signalez la parenthèse OUVRANTE : retenez son jeton avant d'analyser l'intérieur."),
    ("a = 1 et (b = 2 ou (c = 3)", ("parenthèse",),
     "La parenthèse fermée est la deuxième ; celle qui ne l'est jamais est la PREMIÈRE (position 9)."),
    ('region dans ("Nord", "Est"', ("parenthèse",), "La parenthèse ouverte après « dans » n'est jamais fermée."),
    ("contrat = mensuel", ("guillemet",),
     "Un NOM là où il faut une valeur : presque toujours un texte écrit sans ses guillemets. Le message doit le dire."),
    ("6 > anciennete", ("colonne",), "Une condition commence par le nom d'une colonne."),
    ("a = 1 b = 2", (), "Après une règle complète, il doit rester FIN. Sinon « b = 2 » serait ignoré EN SILENCE."),
    ("a = 1)", (), "Une parenthèse fermante en trop : vérifiez qu'il ne reste que FIN à la fin."),
    ("a = 1 et", (), "Il manque la condition après « et » : l'erreur est à la fin du texte."),
    ("a <", (), "Il manque la valeur : l'erreur est à la fin du texte."),
    ("a 6", (), "Après un nom de colonne, il faut un opérateur ou « dans »."),
    ("a", (), "Après un nom de colonne, il faut un opérateur ou « dans »."),
    ('region dans "Nord"', (), "Après « dans », il faut une parenthèse."),
    ("region dans ()", (), "« dans » demande au moins une valeur."),
    ("a = 1 et et b = 2", (), "Deux « et » de suite : le second est à la place d'une condition."),
    ("a = (1)", (), "Une valeur ne se met pas entre parenthèses."),
    ("()", (), "Des parenthèses vides : il faut une condition à l'intérieur."),
    ("non", (), "Il manque la condition après « non »."),
    ('a = "x', ("guillemet",), "Les erreurs de `decouper` passent telles quelles."),
]


def verifier_analyser(f) -> None:
    # 1. Les cas simples, la priorité, les parenthèses.
    for texte in _CAS_ANALYSER:
        _verifier_arbre(f, texte)
    for texte in regles_exemple().values():
        _verifier_arbre(f, texte)

    # 2. Les erreurs, chacune à sa position.
    for texte, mots, conseil in _ERREURS_ANALYSER:
        try:
            _ref_analyser(texte)
        except ErreurSyntaxe as exc:
            position = exc.position
        _erreur_syntaxe(f, texte, position, f"analyser({texte!r})", mots, conseil)
    _doit_lever(f, (None,), TypeError, "analyser(None)", "Les erreurs de `decouper` passent telles quelles.")

    # 3. Des règles tirées au hasard : l'arbre lu doit être celui qui a servi à écrire la règle.
    for graine in range(150):
        arbre = generer_arbre(graine, random.Random(graine).randint(1, 5))
        texte = _ref_texte(arbre)
        _verifier_arbre(f, texte, f"analyser({_court(texte, 120)})")

    # 4. 5 000 « ou » à la suite : une boucle, pas une récursion.
    longue = " ou ".join(f"a = {i}" for i in range(5000))
    _verifier_arbre(f, longue, "analyser(<5 000 conditions reliées par « ou »>)", lignes_max=20_000_000,
                    conseil=_CONSEIL_RECURSION)

    # 5. L'imbrication : 100 niveaux passent, 101 non, et 5 000 ne font pas planter.
    for texte, quoi in (("(" * 100 + "a = 1" + ")" * 100, "analyser(<100 parenthèses imbriquées>)"),
                        ("non " * 100 + "a = 1", "analyser(<100 « non » à la suite>)"),
                        ("non (" * 50 + "a = 1" + ")" * 50, "analyser(<50 fois « non (»>)")):
        _verifier_arbre(f, texte, quoi, conseil="100 niveaux doivent passer : la limite est AU-DELÀ de 100.")
    for texte, quoi in (("(" * 101 + "a = 1" + ")" * 101, "analyser(<101 parenthèses imbriquées>)"),
                        ("non " * 101 + "a = 1", "analyser(<101 « non » à la suite>)"),
                        ("non (" * 51 + "a = 1" + ")" * 51, "analyser(<51 fois « non (»>)"),
                        ("(" * 5000 + "a = 1" + ")" * 5000, "analyser(<5 000 parenthèses imbriquées>)"),
                        ("non " * 5000 + "a = 1", "analyser(<5 000 « non » à la suite>)")):
        try:
            _ref_analyser(texte)
        except ErreurSyntaxe as exc:
            position = exc.position
        _erreur_syntaxe(f, texte, position, quoi, ("imbriqu",),
                        "Comptez chaque « ( » ouverte et chaque « non » en cours ; l'erreur est sur le jeton qui fait "
                        "passer à 101.")


# ============================================================================ Défi 154 : evaluer


class _Espion(dict):
    """Un client qui note les colonnes lues (pour voir le court-circuit)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lues: list = []

    def __getitem__(self, cle):
        self.lues.append(cle)
        return dict.__getitem__(self, cle)

    def get(self, cle, defaut=None):
        self.lues.append(cle)
        return dict.get(self, cle, defaut)


def _nom_resultat(r) -> str:
    return {True: "True (vrai)", False: "False (faux)", None: "None (inconnu)"}.get(r, repr(r)) \
        if r is None or type(r) is bool else repr(r)


def _verifier_resultat(f, arbre, client, quoi: str, conseil: str = "") -> None:
    obtenu = _appel(f, (arbre, client), quoi)
    attendu = _ref_evaluer(arbre, client)
    if obtenu is not None and type(obtenu) is not bool:
        raise AssertionError(f"{quoi} renvoie {obtenu!r} (de type {type(obtenu).__name__}) : il faut exactement True, "
                             f"False ou None. Avec des valeurs numpy, `np.float64(3) < 6` donne `np.True_`, qui n'est "
                             f"pas True (`r is True` vaut False) : convertissez avec `bool(...)`.")
    if obtenu is attendu:
        return
    raise AssertionError(f"{quoi} renvoie {_nom_resultat(obtenu)} au lieu de {_nom_resultat(attendu)}. {conseil}".rstrip())


_T, _F, _U = ("=", "v", 1), ("=", "f", 1), ("=", "u", 1)
_CLIENT_KLEENE = {"v": 1, "f": 0, "u": None}
_NOMS_KLEENE = {_T: "VRAI", _F: "FAUX", _U: "INCONNU"}


def _conseil_kleene(sorte: str, enfants: tuple, obtenu, attendu) -> str:
    noms = f" {sorte.upper()} ".join(_NOMS_KLEENE[e] for e in enfants)
    if sorte == "non":
        return ("NON INCONNU reste INCONNU : on ne sait pas, donc on ne sait pas non plus le contraire (`not None` vaut "
                "True).")
    if attendu is None and obtenu is False:
        return (f"{noms} vaut INCONNU : rien n'est faux, mais tout n'est pas sûr. `all()` et `and` traitent None comme "
                f"False : il faut distinguer `r is False` de `r is None`.")
    if attendu is None and obtenu is True:
        return f"{noms} vaut INCONNU : rien n'est vrai, mais tout n'est pas sûr. Distinguez `r is True` de `r is None`."
    if attendu is False:
        return f"{noms} vaut FAUX : un seul FAUX suffit à rendre un « et » faux, même s'il y a de l'inconnu."
    return f"{noms} vaut VRAI : un seul VRAI suffit à rendre un « ou » vrai, même s'il y a de l'inconnu."


def verifier_evaluer(f) -> None:
    clients = clients_exemple()

    # 1. Inconnu : None et NaN, avec tous les opérateurs.
    for op in OPERATEURS:
        _verifier_resultat(f, (op, "facture", 50), clients[5], f"evaluer(('{op}', 'facture', 50), <client 6 : facture NaN>)",
                           "NaN veut dire « on ne sait pas » : la comparaison vaut None. Attention, en Python "
                           "`nan != 50` vaut True et `nan < 50` vaut False : testez `math.isnan` AVANT de comparer.")
        _verifier_resultat(f, (op, "region", "Nord"), clients[3], f"evaluer(('{op}', 'region', 'Nord'), <client 4 : région None>)",
                           "None veut dire « on ne sait pas » : la comparaison vaut None, quel que soit l'opérateur.")
    _verifier_resultat(f, ("dans", "region", ("Nord", "Est")), clients[3], "evaluer(('dans', 'region', ('Nord', 'Est')), <client 4>)",
                       "Une région inconnue n'est ni dans la liste, ni hors de la liste : None.")
    _verifier_resultat(f, ("=", "appels_support", 0), clients[6], "evaluer(('=', 'appels_support', 0), <client 7>)",
                       "0 est une vraie valeur, pas une absence : `if not valeur` le confondrait avec None.")

    # 2. La logique à trois valeurs, combinaison par combinaison.
    for k in (2, 3):
        for enfants in itertools.product((_T, _F, _U), repeat=k):
            for sorte in ("et", "ou"):
                arbre = (sorte, *enfants)
                obtenu = _appel(f, (arbre, dict(_CLIENT_KLEENE)), f"evaluer({_court(arbre)}, …)")
                attendu = _ref_evaluer(arbre, _CLIENT_KLEENE)
                if obtenu is not attendu:
                    noms = ", ".join(f"<{_NOMS_KLEENE[e]}>" for e in enfants)
                    raise AssertionError(f"evaluer(('{sorte}', {noms}), …) renvoie "
                                         f"{_nom_resultat(obtenu)} au lieu de {_nom_resultat(attendu)}. "
                                         f"{_conseil_kleene(sorte, enfants, obtenu, attendu)}")
    for enfant in (_T, _F, _U):
        obtenu = _appel(f, (("non", enfant), dict(_CLIENT_KLEENE)), f"evaluer(('non', {_court(enfant)}), …)")
        attendu = _ref_evaluer(("non", enfant), _CLIENT_KLEENE)
        if obtenu is not attendu:
            raise AssertionError(f"evaluer(('non', <{_NOMS_KLEENE[enfant]}>), …) renvoie {_nom_resultat(obtenu)} au lieu "
                                 f"de {_nom_resultat(attendu)}. {_conseil_kleene('non', (enfant,), obtenu, attendu)}")

    # 3. Le court-circuit : l'espion note les colonnes lues.
    piege = ("=", "piege", 1)
    for arbre, lues_attendues, explication in (
            (("et", _F, piege), ["f"], "un « et » s'arrête au premier FAUX"),
            (("ou", _T, piege), ["v"], "un « ou » s'arrête au premier VRAI"),
            (("et", _U, _F, piege), ["u", "f"], "INCONNU n'arrête pas un « et » (un FAUX plus loin l'emporterait), "
                                               "mais le FAUX qui suit, si"),
            (("ou", ("et", _F, piege), _T, piege), ["f", "v"], "le « et » s'arrête à son FAUX, le « ou » à son VRAI")):
        espion = _Espion(_CLIENT_KLEENE, piege=1)
        _appel(f, (arbre, espion), f"evaluer({_court(arbre)}, …)")
        lues = list(dict.fromkeys(espion.lues))  # lire deux fois la même colonne n'est pas une faute
        if lues != lues_attendues:
            raise AssertionError(f"evaluer({_court(arbre)}, …) lit les colonnes {lues} au lieu de "
                                 f"{lues_attendues} : {explication}. Comme `and` et `or` en Python, on ne lit pas la suite "
                                 f"quand le résultat est déjà connu (et une colonne lente ou absente n'est même pas "
                                 f"touchée). `return False` dès le premier faux, dans la boucle.")

    # 4. Une colonne absente est une erreur, pas un inconnu.
    for arbre in (("=", "ville", "Paris"), ("dans", "ville", ("Paris",)), ("et", ("=", "contrat", "mensuel"), ("<", "anciennete_mois", 6))):
        _doit_lever(f, (arbre, clients_exemple()[0]), KeyError, f"evaluer({_court(arbre)}, <client 1>)",
                    "Une colonne qui n'existe pas chez le client, c'est une faute de frappe dans la règle : `client[colonne]` "
                    "lève KeyError. Avec `client.get(colonne)`, la règle ne ciblerait plus personne, sans que personne "
                    "ne le voie.")

    # 5. On ne compare que deux valeurs du même genre.
    client = dict(clients_exemple()[0], anciennete_texte="6")
    for arbre, explication in (
            (("=", "anciennete", "3"), "`3 == \"3\"` vaut False en Python, sans erreur : la règle ne ciblerait personne."),
            (("<", "contrat", 3), "Un texte et un nombre ne se comparent pas."),
            (("=", "anciennete_texte", 6), "La colonne contient le TEXTE \"6\" : une donnée mal typée doit se voir."),
            (("=", "fibre", 1), "Pour Python, `True == 1` : un booléen n'est pas un nombre ici. Testez `bool` AVANT `int`."),
            (("=", "appels_support", True), "Pour Python, `4 == True` vaut False, mais `1 == True` vaudrait True : "
                                            "un booléen n'est pas un nombre ici."),
            (("<", "fibre", True), "Sur deux booléens, seuls = et != ont un sens."),
            (("dans", "anciennete", ("3", 4)), "Dans « dans » aussi, chaque valeur doit être du genre de la colonne."),
            (("dans", "fibre", (1, 0)), "Pour Python, `True in (1, 0)` vaut True : un booléen n'est pas un nombre ici.")):
        _doit_lever(f, (arbre, client), TypeError, f"evaluer({_court(arbre)}, <client 1>)", explication,
                    mots=(arbre[1],))
    _verifier_resultat(f, ("=", "anciennete", 3.0), clients[0], "evaluer(('=', 'anciennete', 3.0), <client 1>)",
                       "3 et 3.0 sont deux nombres : on peut les comparer.")
    _doit_lever(f, (("xor", _T, _F), dict(_CLIENT_KLEENE)), ValueError, "evaluer(('xor', …), …)",
                "Un nœud inconnu lève ValueError : sans ça, une règle mal construite renverrait None en silence.")

    # 6. Un client lu depuis pandas : des nombres numpy.
    import numpy as np

    client_np = {"anciennete": np.int64(3), "facture": np.float64(89.9), "inconnue": np.float64("nan"), "fibre": True}
    for arbre in (("<", "anciennete", 6), (">", "facture", 100), ("=", "inconnue", 1), ("non", ("<", "facture", 50)),
                  ("dans", "anciennete", (3, 5)), ("et", ("<", "anciennete", 6), (">", "facture", 80))):
        _verifier_resultat(f, arbre, client_np, f"evaluer({_court(arbre)}, <client avec des nombres numpy>)",
                           "`numpy.float64('nan')` est un NaN (et un float) ; `numpy.int64` est un nombre "
                           "(`isinstance(x, numbers.Real)`).")

    # 7. Les règles du marketing sur les huit clients.
    for regle in regles_exemple().values():
        arbre = _ref_analyser(regle)
        for client in clients:
            _verifier_resultat(f, arbre, client, f"evaluer(analyser({regle!r}), <client {client['id']}>)")

    # 8. Des règles et des clients tirés au hasard.
    clients_hasard = generer_clients(60, graine=154)
    for graine in range(250):
        arbre = generer_arbre(graine, 4)
        for client in clients_hasard[graine % 6::6]:
            _verifier_resultat(f, arbre, client, f"evaluer(generer_arbre({graine}, 4), <client {client['id']} : "
                                                 f"{_court({k: v for k, v in client.items() if k != 'id'}, 150)}>)")


# ============================================================================ Défi 155 : vers_texte


def _verifier_texte(f, arbre, quoi: str | None = None) -> str:
    quoi = quoi or f"vers_texte({_court(arbre, 150)})"
    obtenu = _appel(f, (arbre,), quoi)
    attendu = _ref_texte(arbre)
    if obtenu == attendu:
        return obtenu
    if not isinstance(obtenu, str):
        raise AssertionError(f"{quoi} renvoie {type(obtenu).__name__} au lieu d'un texte (str).")
    conseil = ""
    if "True" in obtenu or "False" in obtenu:
        conseil = ("True et False s'écrivent `vrai` et `faux` : `True` serait relu comme un NOM de colonne. Testez "
                   "`isinstance(v, bool)` AVANT `int` (True est un int pour Python).")
    elif obtenu == _ref_texte(arbre, "strict"):
        conseil = ("Il manque des parenthèses : un « et » dans un « et » (ou un « ou » dans un « ou ») en a besoin. Sans "
                   "elles, la règle relue ne ferait plus qu'UN seul nœud : `analyser(vers_texte(arbre))` ne redonnerait "
                   "pas l'arbre. Comparez avec `<=`, pas `<`.")
    elif obtenu == _ref_texte(arbre, "compose") or obtenu == f"({attendu})":
        conseil = ("Trop de parenthèses : un « et » dans un « ou » n'en a pas besoin (« et » passe avant « ou »), ni la "
                   "règle entière. Le marketing veut la règle la plus lisible.")
    elif any(isinstance(v, str) and ("\\" in v or '"' in v) for v in _valeurs(arbre)):
        conseil = ("Un texte s'écrit entre guillemets doubles, avec une barre oblique inverse devant chaque `\\` et chaque "
                   "`\"` : `.replace(\"\\\\\", \"\\\\\\\\\")` D'ABORD (sinon les barres ajoutées pour les guillemets "
                   "seraient doublées à leur tour), puis `.replace('\"', '\\\\\"')`.")
    elif any(isinstance(v, float) for v in _valeurs(arbre)):
        conseil = ("Un float s'écrit avec `repr(v)` : le texte le plus court qui redonne EXACTEMENT le même float. "
                   "`round`, `:.2f` ou `:g` changent la règle.")
    raise AssertionError(f"{quoi} renvoie\n          {obtenu!r}\n        au lieu de\n          {attendu!r}\n        "
                         f"{conseil}".rstrip())


def _valeurs(arbre) -> list:
    if arbre[0] in ("et", "ou", "non"):
        return [v for e in arbre[1:] for v in _valeurs(e)]
    return list(arbre[2]) if arbre[0] == "dans" else [arbre[2]]


def verifier_vers_texte(f) -> None:
    a, b, c, d = ("=", "a", 1), ("=", "b", 2), ("=", "c", 3), ("=", "d", 4)
    # 1. Les valeurs, une par une.
    for arbre in (("<", "anciennete", 6), ("=", "fibre", True), ("!=", "fibre", False), (">=", "solde", -3),
                  ("<", "facture", 79.5), ("=", "taux", 0.1 + 0.2), ("=", "contrat", "mensuel"),
                  ("=", "nom", "l'été"), ("=", "note", 'il a dit "non"'), ("=", "chemin", "C:\\exports\\"),
                  ("=", "mixte", 'a\\"b'), ("=", "vide", ""), ("dans", "region", ("Nord",)),
                  ("dans", "region", ("Nord", "Est", "Sud")), ("dans", "x", (1, 2.5, True, "z"))):
        _verifier_texte(f, arbre)

    # 2. Les parenthèses : seulement là où l'arbre l'exige.
    for arbre in (("et", a, b), ("ou", a, ("et", b, c)), ("et", a, ("ou", b, c)), ("et", a, ("et", b, c)),
                  ("ou", ("ou", a, b), c), ("et", ("et", a, b), ("et", c, d)), ("non", a), ("non", ("non", a)),
                  ("non", ("et", a, b)), ("non", ("ou", a, b)), ("et", ("non", a), ("non", ("ou", b, c))),
                  ("ou", ("et", a, ("ou", b, c)), d)):
        _verifier_texte(f, arbre)
    for texte in regles_exemple().values():
        _verifier_texte(f, _ref_analyser(texte), f"vers_texte(analyser({texte!r}))")

    # 3. L'aller-retour sur des arbres tirés au hasard.
    for graine in range(200):
        arbre = generer_arbre(graine, random.Random(graine).randint(1, 5))
        texte = _verifier_texte(f, arbre, f"vers_texte(generer_arbre({graine}, …))")
        if _ref_analyser(texte) != arbre:
            raise AssertionError(f"vers_texte(generer_arbre({graine}, …)) ne fait pas l'aller-retour.")

    # 4. Ce qui ne se relirait pas.
    for arbre, explication in (
            (("=", "x", float("inf")), "« inf » n'est pas un nombre que l'analyseur sait lire."),
            (("=", "x", float("nan")), "« nan » n'est pas un nombre que l'analyseur sait lire."),
            (("=", "date début", 1), "Avec une espace, « date début » serait relu comme deux noms."),
            (("=", "", 1), "Un nom de colonne vide ne se relit pas."),
            (("=", "2024", 1), "« 2024 » serait relu comme un nombre."),
            (("=", "et", 1), "« et » serait relu comme le mot-clé."),
            (("=", "Vrai", 1), "« Vrai » serait relu comme le booléen True."),
            (("dans", "region", ()), "« dans » sans valeur ne se relit pas."),
            (("et", a), "Un « et » à une seule condition ne se relit pas comme un « et »."),
            (("xor", a, b), "Un nœud inconnu.")):
        _doit_lever(f, (arbre,), ValueError, f"vers_texte({_court(arbre)})",
                    f"{explication} Ce qui ne ferait pas l'aller-retour lève ValueError.")


# ============================================================================ Défi 156 : verifier_regle

_NOMS = {"nombre": "nombre", "texte": "texte", "booleen": "booleen"}


def _mots_attendus(probleme: tuple) -> tuple[list[str], list[str]]:
    """(mots qui doivent être dans le message, mots qui ne doivent pas y être)."""
    if probleme[0] == "inconnue":
        _, colonne, suggestion = probleme
        if suggestion:
            return [colonne, "inconnue", "vouliez-vous dire", suggestion], []
        return [colonne, "inconnue"], ["vouliez-vous dire"]
    if probleme[0] == "genre":
        _, colonne, genre, genre_valeur = probleme
        return [colonne, _NOMS[genre], _NOMS[genre_valeur]], []
    _, colonne, op = probleme
    return [colonne, "booleen", op], []


def _correspond(message: str, probleme: tuple) -> bool:
    oui, non = _mots_attendus(probleme)
    texte = _sans_accents(message)
    return all(_sans_accents(m) in texte for m in oui) and not any(_sans_accents(m) in texte for m in non)


def _decrire_probleme(p: tuple) -> str:
    if p[0] == "inconnue":
        return f"colonne « {p[1]} » inconnue" + (f" (suggestion : « {p[2]} »)" if p[2] else " (aucune suggestion)")
    if p[0] == "genre":
        return f"« {p[1]} » est un {p[2]}, comparée à un {p[3]}"
    return f"« {p[1]} » est un booléen, avec « {p[2]} »"


def _indice_oubli(probleme: tuple) -> str:
    if probleme[0] == "genre" and "booleen" in probleme[2:]:
        return (" Rappel : True n'est pas un nombre ici, et 1 n'est pas un booléen. `isinstance(True, int)` vaut True : "
                "testez `isinstance(v, bool)` AVANT les nombres.")
    if probleme[0] == "genre":
        return " Le genre d'une valeur : bool → « booleen », int ou float → « nombre », str → « texte »."
    if probleme[0] == "ordre":
        return " Sur une colonne booléenne, seuls =, != et « dans » ont un sens."
    return " Une colonne absente du schéma : `colonne not in schema`."


def _verifier_problemes(f, arbre, schema, quoi: str) -> None:
    avant_schema = copy.deepcopy(schema)
    obtenu = _appel(f, (arbre, schema), quoi,
                    "Un problème se SIGNALE (on l'ajoute à la liste) : on ne lève pas d'exception, on continue.")
    if schema != avant_schema:
        raise AssertionError(f"{quoi} a MODIFIÉ le schéma reçu : il sert à toutes les autres règles.")
    if not isinstance(obtenu, list) or not all(isinstance(m, str) for m in obtenu):
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} : il faut une LISTE de messages (des str).")
    attendus = _ref_problemes(arbre, schema)
    for i, probleme in enumerate(attendus):
        if i < len(obtenu) and _correspond(obtenu[i], probleme):
            continue
        if not obtenu and attendus:
            raise AssertionError(f"{quoi} renvoie [] alors qu'il y a {len(attendus)} problème(s) : "
                                 f"{'; '.join(_decrire_probleme(p) for p in attendus)}.{_indice_oubli(probleme)}")
        if len(obtenu) == 1 and len(attendus) > 1 and _correspond(obtenu[0], attendus[0]):
            raise AssertionError(f"{quoi} ne signale qu'UN problème sur {len(attendus)} : vous vous arrêtez au premier. "
                                 f"Le marketing veut TOUT corriger d'un coup (manque : "
                                 f"{'; '.join(_decrire_probleme(p) for p in attendus[1:])}).")
        ailleurs = [j for j, m in enumerate(obtenu) if _correspond(m, probleme)]
        if ailleurs:
            raise AssertionError(f"{quoi} donne les bons messages, mais pas dans l'ordre de lecture de la règle (de gauche "
                                 f"à droite) : {_court(obtenu, 300)}.")
        oui, non = _mots_attendus(probleme)
        if i < len(obtenu):
            texte = _sans_accents(obtenu[i])
            absents = [m for m in oui if _sans_accents(m) not in texte]
            presents = [m for m in non if _sans_accents(m) in texte]
            indice = ""
            if probleme[0] == "inconnue" and probleme[2] and absents:
                indice = (" La suggestion : d'abord une colonne qui ne diffère que par les majuscules, sinon "
                          "`difflib.get_close_matches(colonne, list(schema), n=1)`.")
            elif probleme[0] == "inconnue" and presents:
                indice = (" Aucune colonne n'est assez proche : pas de « vouliez-vous dire » (une suggestion absurde "
                          "égare). `difflib.get_close_matches` renvoie [] dans ce cas.")
            elif probleme[0] == "genre" and probleme[3] == "booleen":
                indice = " Rappel : True n'est pas un nombre ici (`isinstance(True, int)` vaut True : testez bool d'abord)."
            raise AssertionError(f"{quoi} : le message n°{i + 1} ({obtenu[i]!r}) devrait parler de "
                                 f"{_decrire_probleme(probleme)}"
                                 + (f" : il manque {', '.join(repr(m) for m in absents)}" if absents else "")
                                 + (f" ; il ne devrait pas contenir {', '.join(repr(m) for m in presents)}" if presents else "")
                                 + f".{indice}")
        raise AssertionError(f"{quoi} ne signale pas : {_decrire_probleme(probleme)} (vous renvoyez "
                             f"{_court(obtenu, 300)}).{_indice_oubli(probleme)}")
    if len(obtenu) > len(attendus):
        en_trop = obtenu[len(attendus):]
        indice = ""
        if any(p[0] == "inconnue" for p in attendus):
            indice = " Une condition dont la colonne est inconnue n'est pas examinée plus loin (son genre n'existe pas)."
        elif any(isinstance(v, bool) for v in _valeurs(arbre)):
            indice = _indice_oubli(("genre", "", "booleen", "nombre"))
        raise AssertionError(f"{quoi} signale {len(obtenu)} problème(s) au lieu de {len(attendus)} : en trop "
                             f"{_court(en_trop, 300)}. Un seul message par condition fautive, et rien pour une condition "
                             f"juste (6 et 6.5 sont des nombres ; « dans » et = / != vont avec un booléen).{indice}")


def _corrompre(arbre: tuple, rng: random.Random) -> tuple:
    """Le même arbre, avec des fautes : colonnes mal tapées, valeurs du mauvais genre, < sur un booléen."""
    sorte = arbre[0]
    if sorte in ("et", "ou", "non"):
        return (sorte, *(_corrompre(e, rng) for e in arbre[1:]))
    tirage = rng.random()
    colonne = arbre[1]
    if tirage < 0.15:
        colonne = colonne[:-1] if len(colonne) > 3 else colonne + "x"
    elif tirage < 0.2:
        colonne = colonne.upper()
    elif tirage < 0.25:
        colonne = "zz_" + colonne[::-1]
    if sorte == "dans":
        valeurs = tuple(rng.choice((v, str(v), 1, True)) if rng.random() < 0.2 else v for v in arbre[2])
        return ("dans", colonne, valeurs)
    valeur = arbre[2]
    if rng.random() < 0.2:
        valeur = rng.choice((str(valeur), 3, 2.5, True, "x"))
    op = sorte
    if rng.random() < 0.15:
        op = rng.choice(("<", ">="))
    return (op, colonne, valeur)


def verifier_verifier_regle(f) -> None:
    schema = schema_clients()
    # 1. Des règles justes : aucun problème.
    for texte in regles_exemple().values():
        _verifier_problemes(f, _ref_analyser(texte), schema_clients(), f"verifier_regle(analyser({texte!r}), …)")
    for arbre in (("<", "facture", 79.5), ("<", "facture", 80), ("dans", "fibre", (True,)), ("!=", "fibre", False),
                  ("dans", "anciennete", (1, 2.5))):
        _verifier_problemes(f, arbre, schema_clients(), f"verifier_regle({_court(arbre)}, schema_clients())")

    # 2. Chaque problème, seul.
    for arbre in (
            ("<", "ancienete", 6),          # une lettre en moins : difflib propose « anciennete »
            ("=", "FIBRE", True),           # les majuscules : difflib ne voit rien, mais c'est évidemment « fibre »
            ("=", "Region", "Nord"),
            ("=", "ville", "Paris"),        # rien de proche : pas de suggestion
            ("=", "facture", "80"),         # un texte pour un nombre
            ("=", "anciennete", True),      # True n'est pas un nombre
            ("=", "fibre", 1),              # 1 n'est pas un booléen
            ("<", "contrat", 3),
            ("dans", "region", ("Nord", 3)),
            ("<", "fibre", True), (">=", "fibre", False),
            ("<", "fibre", 3),              # deux défauts, un seul message : le genre d'abord
            ("=", "anciennte", "x")):       # colonne inconnue : on ne regarde pas plus loin
        _verifier_problemes(f, arbre, schema_clients(), f"verifier_regle({_court(arbre)}, schema_clients())")

    # 3. Plusieurs problèmes : tous, dans l'ordre de lecture, même cachés sous un « non » ou dans un « ou ».
    for arbre in (
            ("et", ("<", "ancienete", 6), ("=", "facture", "80"), (">", "fibre", False)),
            ("ou", ("non", ("=", "contrat", 1)), ("et", ("=", "regoin", "Nord"), ("<", "fibre", True)),
             ("dans", "paiement", ("carte", False))),
            ("ou", ("<", "ancienete", 3), (">", "ancienete", 50))):
        _verifier_problemes(f, arbre, schema_clients(), f"verifier_regle({_court(arbre, 150)}, schema_clients())")
    _doit_lever(f, (("xor", ("=", "fibre", True), ("=", "fibre", False)), schema), ValueError,
                "verifier_regle(('xor', …), schema_clients())", "Un nœud inconnu lève ValueError.")

    # 4. Un autre schéma, et des règles abîmées au hasard.
    autre = {"Montant": "nombre", "statut": "texte", "actif": "booleen", "nb_retours": "nombre"}
    _verifier_problemes(f, ("et", ("=", "montant", 3), ("=", "statu", "ok"), ("<", "actif", True)), autre,
                        "verifier_regle(…, {'Montant': 'nombre', 'statut': 'texte', …})")
    rng = random.Random(156)
    for graine in range(150):
        arbre = _corrompre(generer_arbre(graine, 3), rng)
        _verifier_problemes(f, arbre, schema_clients(), f"verifier_regle({_court(arbre, 150)}, schema_clients())")


# ============================================================================ Défi 157 : vers_sql

_BIZARRES = {"order": "nombre", "date fin": "texte", 'prix "TTC"': "nombre"}


def _clients_sql(n: int, graine: int) -> list[dict]:
    clients = generer_clients(n, graine)
    rng = random.Random(graine)
    for client in clients:
        client["order"] = rng.choice([None, *range(1, 10)])
        client["date fin"] = rng.choice([None, "2024-01-31", "2024-06-30", "2025-03-15"])
        client['prix "TTC"'] = rng.choice([None, 19.99, 29.99, 49.5])
    return clients


def _attendus(arbre, clients) -> list[int]:
    return [c["id"] for c in clients if _ref_evaluer(arbre, c) is True]


def _verifier_sql(f, arbre, clients, conn, quoi: str | None = None) -> None:
    quoi = quoi or f"vers_sql({_court(arbre, 150)})"
    obtenu = _appel(f, (arbre,), quoi)
    if not (isinstance(obtenu, (tuple, list)) and len(obtenu) == 2 and isinstance(obtenu[0], str)
            and isinstance(obtenu[1], (list, tuple))):
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} : il faut un couple (sql, params), le texte SQL et la LISTE "
                             f"des valeurs.")
    sql, params = obtenu
    _, params_attendus = _ref_sql(arbre)
    if sql.count("?") != len(params):
        raise AssertionError(f"{quoi} renvoie {sql!r} avec {sql.count('?')} « ? » mais {len(params)} valeur(s) "
                             f"({_court(params)}) : il faut une valeur par « ? », dans le même ordre.")
    for v in params_attendus:
        if isinstance(v, str) and len(v) >= 3 and v in sql:
            raise AssertionError(f"{quoi} renvoie {sql!r} : la valeur {v!r} est DANS le texte SQL. C'est l'injection SQL "
                                 f"de la série 13 : une valeur voyage toujours à part, à la place d'un « ? ».")
    try:
        trouves = [i for (i,) in conn.execute(f"SELECT id FROM clients WHERE {sql} ORDER BY id", list(params))]
    except sqlite3.Error as exc:
        indice = ""
        if any(nom in _colonnes_de(arbre) for nom in _BIZARRES):
            indice = (" Les noms de colonnes s'écrivent entre guillemets DOUBLES, chaque « \" » doublé : `order` est un "
                      "mot réservé de SQL, `date fin` contient une espace.")
        raise AssertionError(f"{quoi} renvoie {sql!r}, que SQLite refuse : {exc}.{indice}") from None
    attendus = _attendus(arbre, clients)
    if trouves == attendus:
        return
    indice = ""
    if sorted(map(repr, params)) == sorted(map(repr, params_attendus)) and list(params) != list(params_attendus):
        indice = " Les valeurs ne sont pas dans l'ordre des « ? » : ajoutez-les au moment où vous écrivez leur « ? »."
    elif any(m in sql.upper() for m in ("IS NULL", "IS NOT", "COALESCE", "IFNULL")):
        indice = (" Inutile de traiter NULL vous-même : SQL applique déjà la logique à trois valeurs du défi 154 (une "
                  "condition NULL n'est pas retenue par WHERE, NOT NULL reste NULL).")
    else:
        sans = _ref_sql(arbre, parentheses=False)[0]
        try:
            variante = [i for (i,) in conn.execute(f"SELECT id FROM clients WHERE {sans} ORDER BY id", params_attendus)]
        except sqlite3.Error:
            variante = None
        if variante == trouves:
            indice = (" Il manque des parenthèses : en SQL aussi, AND passe avant OR, et `NOT a AND b` se lit "
                      "`(NOT a) AND b`. Entourez chaque groupe AND / OR, et ce qui suit NOT, de parenthèses.")
    manquants = [i for i in attendus if i not in trouves][:8]
    en_trop = [i for i in trouves if i not in attendus][:8]
    exemple = next((c for c in clients if c["id"] in (manquants or en_trop)), None)
    raise AssertionError(f"{quoi} renvoie {sql!r} avec {_court(list(params))} : la requête trouve {len(trouves)} clients "
                         f"au lieu de {len(attendus)} (manquants : {manquants or 'aucun'} ; en trop : "
                         f"{en_trop or 'aucun'}). Par exemple le client {_court(exemple, 300)}, pour qui `evaluer` "
                         f"donne {_ref_evaluer(arbre, exemple) if exemple else '?'}.{indice}")


def _colonnes_de(arbre) -> set:
    if arbre[0] in ("et", "ou", "non"):
        return {c for e in arbre[1:] for c in _colonnes_de(e)}
    return {arbre[1]}


def verifier_vers_sql(f) -> None:
    # 1. Les exemples, sur les huit clients.
    clients = clients_exemple()
    conn = base_clients(clients)
    for texte in regles_exemple().values():
        _verifier_sql(f, _ref_analyser(texte), clients, conn, f"vers_sql(analyser({texte!r}))")

    # 2. Aucune valeur dans le texte SQL.
    piege = ("ou", ("=", "region", "x' OR 1=1 --"), ("=", "contrat", "mensuel"), (">", "facture", 123456.75),
             ("dans", "paiement", ('"); DROP TABLE clients; --', "chèque")))
    obtenu = _appel(f, (piege,), f"vers_sql({_court(piege)})")
    if isinstance(obtenu, (tuple, list)) and obtenu and isinstance(obtenu[0], str) and "123456" in obtenu[0]:
        raise AssertionError(f"vers_sql({_court(piege)}) renvoie {obtenu[0]!r} : le nombre 123456.75 est DANS le texte "
                             f"SQL. Les nombres aussi voyagent à part, à la place d'un « ? » (une valeur, c'est une "
                             f"valeur, quel que soit son type).")
    _verifier_sql(f, piege, clients, conn)

    # 3. Des noms de colonnes que SQL n'aime pas.
    clients = _clients_sql(200, graine=157)
    conn = base_clients(clients)
    for arbre in ((">", "order", 3), ("<=", "date fin", "2024-06-30"), ("!=", 'prix "TTC"', 29.99),
                  ("et", (">", "order", 3), ("dans", "date fin", ("2024-01-31", "2025-03-15"))),
                  ("ou", ("non", ("dans", "order", (1, 2, 3))), ("=", 'prix "TTC"', 49.5))):
        _verifier_sql(f, arbre, clients, conn)

    # 4. Des centaines de règles au hasard, avec des NULL : le même résultat que l'interpréteur.
    for graine in range(250):
        _verifier_sql(f, generer_arbre(graine, random.Random(graine).randint(1, 5)), clients, conn,
                      f"vers_sql(generer_arbre({graine}, …))")

    # 5. Ce qui ne se traduit pas.
    a = ("=", "contrat", "mensuel")
    for arbre, explication in ((("et", a), "Un « et » à une seule condition."),
                               (("dans", "region", ()), "« IN () » est une erreur de syntaxe dans la plupart des bases."),
                               (("xor", a, a), "Un nœud inconnu.")):
        _doit_lever(f, (arbre,), ValueError, f"vers_sql({_court(arbre)})", f"{explication} Levez ValueError.")


SERIE = Serie(
    code="25",
    titre="Écrire un petit langage : les règles de ciblage, du texte à la requête SQL",
    module_eleve="defis.serie_25",
    module_corrige="defis.corrige_serie_25",
    defis=(
        Defi(152, "decouper", "🟢", "—", verifier_decouper),
        Defi(153, "analyser", "🔴", "—", verifier_analyser),
        Defi(154, "evaluer", "🟡", "—", verifier_evaluer),
        Defi(155, "vers_texte", "🟡", "—", verifier_vers_texte),
        Defi(156, "verifier_regle", "🟡", "—", verifier_verifier_regle),
        Defi(157, "vers_sql", "🔴", "—", verifier_vers_sql),
    ),
)

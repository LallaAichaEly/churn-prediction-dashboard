"""Vérifications automatiques de la série 41 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est le lecteur de l'entrepôt de données : il relit ce que vous écrivez, octet par octet, et
il vous envoie les fichiers que le partenaire envoie vraiment, tronqués et abîmés compris.

- il compare vos plages, vos codes et vos octets à ceux d'une version de référence : les règles sont fixées pour
  qu'il n'y ait qu'une bonne réponse ;
- quand votre résultat est faux, il rejoue les erreurs classiques (la dernière plage oubliée, `==` au lieu de
  « même type », le dictionnaire qui confond 1 et True, le bit de poids fort, le dernier octet perdu, le varint
  écrit à l'envers, `abs` au lieu du zigzag, l'écart avec la première valeur, les codes lus dans l'arbre au lieu
  des codes canoniques, l'égalité mal départagée, le zigzag oublié à la lecture…) pour nommer la vôtre ;
- il vous donne des flux qui ne se lisent qu'une fois (`UneSeuleFois`), COMPTE les comparaisons du défi 249
  (chercher dans une liste se voit), et compte les lignes exécutées quand votre code risque de tourner sans
  fin (un varint de nombre négatif, un fichier de 200 000 octets 0xFF) ;
- au défi 253, il vous passe des fichiers vides, coupés au milieu d'un nombre, coupés entre deux nombres, avec
  des octets en trop, avec un nombre de 11 octets, ou qui annoncent 10**15 valeurs : chacun doit lever une
  ValueError, et rien d'autre.
"""

from __future__ import annotations

import heapq
import math
import random
import sys

from .compression import evenements_exemple, instantane_exemple, notes_exemple
from .flux import UneSeuleFois
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
    """Levée dans VOTRE code quand il travaille trop (BaseException : `except Exception` ne l'attrape pas)."""


def _fichier(objet) -> str | None:
    code = getattr(objet, "__code__", None) or getattr(getattr(objet, "__func__", None), "__code__", None)
    if code is not None:
        return code.co_filename
    module = sys.modules.get(getattr(objet, "__module__", None) or "")
    return getattr(module, "__file__", None)


def _trace(fichier: str | None, appel, lignes_max: int | None):
    """Le résultat de `appel()` ; arrête net au-delà de `lignes_max` lignes exécutées dans `fichier`."""
    if fichier is None or lignes_max is None:
        return appel()
    compte = [0]

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


_CONSEIL_LENT = "Votre code a été arrêté : il exécute beaucoup trop de lignes (une boucle qui ne s'arrête pas ?)."


def _conseil_exception(exc: BaseException) -> str:
    texte = str(exc)
    if isinstance(exc, RuntimeError) and "déjà été lu" in texte:
        return ("Vous lisez les valeurs DEUX fois. Un générateur relu ne lève pas d'erreur : il est VIDE, et votre "
                "deuxième boucle ne verrait rien, en silence. Une seule boucle `for v in valeurs:` suffit.")
    if isinstance(exc, TypeError) and "UneSeuleFois" in texte:
        return ("Un flux n'a pas de longueur et ne s'indexe pas (`len(valeurs)`, `valeurs[0]`, `valeurs[i - 1]`) : on "
                "ne sait pas combien de valeurs vont arriver. Une seule boucle `for v in valeurs:`, qui garde ce dont "
                "elle a besoin (la valeur précédente, la plage en cours).")
    if isinstance(exc, TypeError) and "'<' not supported" in texte:
        return ("Python compare deux valeurs qui ne se comparent pas : dans un tas, un élément (poids, ordre, …) dont "
                "l'ordre est UNIQUE n'a jamais besoin de comparer la suite. Et pour `distinctes`, l'ordre est celui "
                "de la première apparition : pas de tri.")
    if isinstance(exc, ValueError) and "byte must be in range" in texte:
        return ("Un octet vaut de 0 à 255 : vous ajoutez un nombre négatif ou trop grand à un bytearray. Un varint "
                "ne s'écrit que pour un nombre >= 0 : passez par le zigzag d'abord, et gardez 7 bits par octet "
                "(`n & 0x7F`).")
    return ""


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = None, conseil_lent: str = ""):
    """f(*args), en transformant toute erreur en message lisible (NotImplementedError : le défi est à faire)."""
    try:
        return _trace(_fichier(f) if lignes_max else None, lambda: f(*args), lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError. {conseil}".rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {_conseil_exception(exc) or conseil}".rstrip()) from None


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu: type = ValueError,
                lignes_max: int | None = None, conseil_lent: str = "") -> None:
    """Exige que f(*args) lève `attendu` (et rien d'autre)."""
    try:
        resultat = _trace(_fichier(f) if lignes_max else None, lambda: f(*args), lignes_max)
    except NotImplementedError:
        raise
    except attendu:
        return
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _premier_bug(obtenus, rejouer, bugs: dict, egal=None) -> str:
    """Le conseil du premier bug dont le rejeu donne EXACTEMENT vos réponses."""
    egal = egal or (lambda a, b: a == b)
    for bug, conseil in bugs.items():
        try:
            faux = rejouer(bug)
        except Exception:
            continue
        if egal(faux, obtenus):
            return conseil
    return ""


def _entier(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _meme(a, b) -> bool:
    return type(a) is type(b) and (a == b or (a != a and b != b))


_NAN = object()


def _cle(v):
    return (type(v), _NAN if v != v else v)


def _pareil(a, b) -> bool:
    """Égaux au sens des défis 248 et 249 (même type, NaN égal à NaN), dans des listes et des tuples aussi."""
    if isinstance(a, (list, tuple)) and type(a) is type(b):
        return len(a) == len(b) and all(_pareil(x, y) for x, y in zip(a, b))
    return _meme(a, b)


def _lignes(n: int) -> int:
    """Le nombre de lignes de VOTRE fichier qu'un code raisonnable exécute pour n éléments (très large)."""
    return 200 * (n + 2) + 1000


def _nan() -> float:
    return float("nan")       # un NOUVEL objet à chaque fois, comme dans une colonne pandas


def _texte(s: str) -> str:
    return "".join(list(s))   # un texte égal à s, mais un AUTRE objet (comme ceux lus dans un fichier)


# ============================================================================ défi 248 : plages


def _ref_plages(valeurs, bug: str = "") -> list[tuple]:
    valeurs = list(valeurs)
    if bug == "compte":
        comptes: dict = {}
        for v in valeurs:
            comptes.setdefault(_cle(v), [v, 0])[1] += 1
        return [(v, n) for v, n in comptes.values()]
    if bug == "vide" and not valeurs:
        return [(None, 0)]
    meme = {"egal": lambda a, b: a == b, "nan": lambda a, b: type(a) is type(b) and a == b,
            "identite": lambda a, b: a is b}.get(bug, _meme)
    resultat: list[tuple] = []
    courante, longueur = None, 0
    for v in valeurs:
        if longueur and meme(courante, v):
            longueur += 1
        else:
            if longueur:
                resultat.append((courante, longueur - 1 if bug == "zero" else longueur))
            courante, longueur = v, 1
    if longueur and bug != "derniere":
        resultat.append((courante, longueur - 1 if bug == "zero" else longueur))
    return resultat


_BUGS_PLAGES = {
    "derniere": "Il manque la DERNIÈRE plage : dans la boucle, une plage n'est rangée que quand une valeur différente "
                "arrive, et après la dernière valeur, il n'en arrive plus. Rangez-la APRÈS la boucle : "
                "`if longueur: resultat.append((courante, longueur))`.",
    "egal": "Vous comparez avec `==` seulement (c'est aussi ce que fait `itertools.groupby`) : `1 == True == 1.0`, donc "
            "[1, True] devient [(1, 2)], et la relecture rend [1, 1] : le True est perdu. Et deux NaN ne sont jamais "
            "égaux : chacun ouvre sa propre plage. La règle : `type(a) is type(b) and (a == b or (a != a and b != b))`.",
    "nan": "Le type est bien vérifié, mais deux NaN ne sont jamais égaux (`nan == nan` est faux) : chacun ouvre sa propre "
           "plage, et la colonne des motifs (presque que des NaN) ne se compresse plus. Ajoutez le cas « deux NaN » : "
           "`a != a and b != b` (seul NaN n'est pas égal à lui-même).",
    "identite": "Vous comparez avec `is` : deux textes égaux ne sont pas forcément le MÊME objet (ceux lus dans un fichier "
                "ou fabriqués par le programme ne le sont pas). Comparez les valeurs : `type(a) is type(b) and a == b`.",
    "compte": "Vous comptez TOUTES les apparitions de chaque valeur (comme `Counter`), pas les répétitions CONSÉCUTIVES : "
              "[M, M, A, M] donne [(M, 2), (A, 1), (M, 1)], pas [(M, 3), (A, 1)]. Une plage s'arrête dès que la "
              "valeur change, et la relecture doit rendre la colonne dans son ordre.",
    "zero": "Vos longueurs sont trop petites de 1 : la plage commence avec UNE valeur (`longueur = 1`), pas 0.",
}


def _cas_248() -> list[list]:
    cas = [
        ["Mensuel", "Mensuel", "Mensuel", "Annuel", "Annuel", "Mensuel"],
        [],
        [7],
        [_texte("Mensuel") for _ in range(4)] + [_texte("Annuel"), _texte("Annuel")],
        [1, 1, True, 1.0, None, None],
        [_nan(), _nan(), "Prix", _nan()],
        [None, None, "x", None],
        ["a"] * 5,
        [True, True, 1, 1, True],
        [0, False, 0, 0.0, "0"],
        [1.5, 1.5, 2, 2.0, "2", "2"],
        [_nan()],
        ["x", "y", "x", "y"],
    ]
    alea = random.Random(248)
    alphabets = (("a", "b"), (1, True, 1.0), ("x", None), ("p", "nan"), (0, False, "0"), ("Prix", "nan", None))
    for _ in range(40):
        alphabet = alea.choice(alphabets)
        colonne: list = []
        for _ in range(alea.randrange(0, 8)):
            v = alea.choice(alphabet)
            for _ in range(alea.randrange(1, 5)):
                colonne.append(_nan() if v == "nan" else _texte(v) if isinstance(v, str) else v)
        cas.append(colonne)
    return cas


def _verifier_forme_plages(r, quoi: str, vide: bool = False) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie {type(r).__name__} : attendu une LISTE de tuples (valeur, longueur).")
    for p in r:
        if isinstance(p, list) and len(p) == 2:
            raise AssertionError(f"{quoi} contient la plage {_court(p)} : une plage est un TUPLE (valeur, longueur), "
                                 "pas une liste.")
        if not isinstance(p, tuple) or len(p) != 2:
            raise AssertionError(f"{quoi} contient {_court(p)} : chaque plage est un tuple de DEUX éléments (valeur, "
                                 "longueur).")
        if not _entier(p[1]):
            raise AssertionError(f"{quoi} contient la plage {_court(p)} : la longueur est un int.")
        if p[1] < 1 and vide:
            raise AssertionError(f"{quoi} renvoie {_court(r)} : pour une colonne vide, la réponse est [] (pas "
                                 "[(None, 0)]) : une plage contient au moins une valeur.")
        if p[1] < 1:
            raise AssertionError(f"{quoi} contient la plage {_court(p)} : une plage contient au moins une valeur. Vos "
                                 "longueurs sont-elles trop petites de 1 ? Une plage commence avec UNE valeur "
                                 "(`longueur = 1`), pas 0.")


def verifier_plages(f) -> None:
    cas = _cas_248()
    obtenus = []
    for valeurs in cas:
        quoi = f"plages({_court(valeurs, 90)})"
        copie = list(valeurs)
        conseil = ("Une colonne vide n'a pas de première valeur : ne lisez pas `valeurs[0]`, commencez avec une plage "
                   "de longueur 0." if not valeurs else "")
        r = _appel(f, (valeurs,), quoi, conseil, lignes_max=_lignes(len(valeurs)))
        _verifier_forme_plages(r, quoi, vide=not valeurs)
        if not _pareil(valeurs, copie):
            raise AssertionError(f"{quoi} a modifié la colonne reçue : elle doit rester intacte.")
        obtenus.append(r)
    for valeurs, r in zip(cas, obtenus):
        attendu = _ref_plages(valeurs)
        if not _pareil(r, attendu):
            conseil = _premier_bug(obtenus, lambda bug: [_ref_plages(v, bug) for v in cas], _BUGS_PLAGES, _pareil)
            raise AssertionError(f"plages({_court(valeurs, 90)}) renvoie {_court(r)} au lieu de {_court(attendu)}. "
                                 f"{conseil}".rstrip())

    # ------------------------------------------------------------------ un flux, et l'instantané de la nuit
    contrats = instantane_exemple()["contrat"]
    r = _appel(f, (UneSeuleFois(contrats),), "plages(un flux des 20 000 contrats de l'instantané)",
               lignes_max=_lignes(len(contrats)))
    _verifier_forme_plages(r, "plages(un flux)")
    if not _pareil(r, _ref_plages(contrats)):
        raise AssertionError(f"Sur un flux (les 20 000 contrats), plages renvoie {_court(r)} au lieu de "
                             f"{_court(_ref_plages(contrats))}.")
    r = _appel(f, ("aaabcc",), "plages('aaabcc')")
    if r != [("a", 3), ("b", 1), ("c", 2)]:
        raise AssertionError(f"plages('aaabcc') renvoie {_court(r)} au lieu de [('a', 3), ('b', 1), ('c', 2)] : un "
                             "texte est un itérable de caractères.")
    motifs = instantane_exemple()["motif"]
    r = _appel(f, (motifs,), "plages(les motifs de l'instantané)")
    if not _pareil(r, _ref_plages(motifs)):
        raise AssertionError(f"Sur la colonne des motifs (des NaN, et 2 004 motifs), plages renvoie {len(r)} plages au "
                             f"lieu de {len(_ref_plages(motifs))}.")


# ============================================================================ défi 249 : dictionnaire


def _ref_dictionnaire(valeurs, bug: str = "") -> tuple[list, list[int]]:
    valeurs = list(valeurs)
    cle = {"valeur": lambda v: v, "nan": lambda v: (type(v), v)}.get(bug, _cle)
    if bug == "trie":
        distinctes, _ = _ref_dictionnaire(valeurs)
        distinctes = sorted(distinctes)
        index = {_cle(v): i for i, v in enumerate(distinctes)}
        return distinctes, [index[_cle(v)] for v in valeurs]
    distinctes: list = []
    codes: list[int] = []
    numeros: dict = {}
    for v in valeurs:
        k = cle(v)
        if k not in numeros:
            numeros[k] = len(distinctes)
            distinctes.append(v)
        codes.append(numeros[k] + (1 if bug == "un" else 0))
    return distinctes, codes


_BUGS_DICTIONNAIRE = {
    "valeur": "Votre dictionnaire a pour clé la valeur seule : `{1: 0, True: 1}` n'a qu'UNE clé (`hash(True) == hash(1)` "
              "et `True == 1`), donc True reçoit le numéro de 1 (et se relit en 1). Et deux `float('nan')` font deux "
              "clés, puisqu'un NaN n'est égal à rien. Mettez le type dans la clé, et un remplaçant pour NaN : "
              "`(type(v), 'NaN' if v != v else v)`.",
    "nan": "Le type est bien dans la clé, mais deux `float('nan')` font deux clés différentes (un NaN n'est égal à rien, "
           "pas même à un autre NaN) : chaque NaN reçoit un nouveau numéro. Remplacez NaN dans la clé : "
           "`(type(v), 'NaN' if v != v else v)`.",
    "un": "Vos numéros commencent à 1 : ce sont des PLACES dans la liste `distinctes`, de 0 à len(distinctes) - 1 "
          "(la relecture fait `distinctes[code]`).",
    "trie": "Vos valeurs distinctes sont TRIÉES : la règle est l'ordre de PREMIÈRE apparition (une colonne qui mélange "
            "textes et None ne se trie même pas).",
}


def _cas_249() -> list[list]:
    cas = [
        ["Chèque", "Virement", "Chèque", "Carte bancaire", "Virement"],
        [],
        [1, True, 1.0, 1, None],
        [_nan(), "Prix", _nan(), _nan()],
        ["b", "a", "c", "a", "b"],
        [None, None],
        [True, 1, False, 0, 0.0, "0"],
        [_texte("Virement"), _texte("Chèque"), _texte("Virement")],
        ["Prix", _nan(), None, "Prix", _nan(), None, 2, 2.0, 2],
    ]
    alea = random.Random(249)
    alphabets = (("z", "y", "x", "w"), (1, True, 1.0, 0, False), ("x", None, "nan"), ("Prix", "nan", 3, 3.0))
    for _ in range(40):
        alphabet = alea.choice(alphabets)
        cas.append([_nan() if v == "nan" else _texte(v) if isinstance(v, str) else v
                    for v in (alea.choice(alphabet) for _ in range(alea.randrange(0, 25)))])
    return cas


class _Compte(str):
    """Un texte qui compte combien de fois on le compare (`==`) : chercher dans une liste se voit."""

    comparaisons = 0
    limite: int | None = None

    def __eq__(self, autre):
        _Compte.comparaisons += 1
        if _Compte.limite is not None and _Compte.comparaisons > _Compte.limite:
            raise _BoucleSansFin
        return str.__eq__(self, autre)

    __hash__ = str.__hash__


def _verifier_forme_dictionnaire(r, quoi: str, n: int) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, (tuple, list)) or len(r) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(r)} : attendu le tuple (distinctes, codes), deux listes.")
    distinctes, codes = r
    if not isinstance(distinctes, list) or not isinstance(codes, list):
        raise AssertionError(f"{quoi} renvoie ({type(distinctes).__name__}, {type(codes).__name__}) : distinctes et "
                             "codes sont deux LISTES (un set ne garde pas l'ordre d'apparition).")
    if len(codes) != n:
        raise AssertionError(f"{quoi} renvoie {len(codes)} codes pour {n} valeurs : un code par valeur, dans l'ordre.")
    if not all(_entier(c) for c in codes):
        raise AssertionError(f"{quoi} : chaque code est un int (la place de la valeur dans `distinctes`).")


def verifier_dictionnaire(f) -> None:
    cas = _cas_249()
    obtenus = []
    for valeurs in cas:
        quoi = f"dictionnaire({_court(valeurs, 90)})"
        copie = list(valeurs)
        r = _appel(f, (valeurs,), quoi, lignes_max=_lignes(len(valeurs)))
        _verifier_forme_dictionnaire(r, quoi, len(valeurs))
        if not _pareil(valeurs, copie):
            raise AssertionError(f"{quoi} a modifié la colonne reçue : elle doit rester intacte.")
        obtenus.append((list(r[0]), list(r[1])))
    for valeurs, r in zip(cas, obtenus):
        attendu = _ref_dictionnaire(valeurs)
        if not _pareil(r, attendu):
            conseil = _premier_bug(obtenus, lambda bug: [_ref_dictionnaire(v, bug) for v in cas], _BUGS_DICTIONNAIRE,
                                   _pareil)
            distinctes, codes = r
            if not conseil and len(distinctes) == len(attendu[0]) and \
                    all(0 <= c < len(distinctes) and _meme(distinctes[c], v) for c, v in zip(codes, valeurs)):
                conseil = ("Vos codes relisent bien la colonne, mais vos valeurs distinctes ne sont pas dans l'ordre de "
                           "leur PREMIÈRE apparition (un `set` n'a pas d'ordre fiable) : la première valeur vue reçoit "
                           "0, la suivante 1…")
            raise AssertionError(f"dictionnaire({_court(valeurs, 90)}) renvoie {_court(r)} au lieu de "
                                 f"{_court(attendu)}. {conseil}".rstrip())

    # ------------------------------------------------------------------ un flux, et l'instantané de la nuit
    paiements = instantane_exemple()["paiement"]
    r = _appel(f, (UneSeuleFois(paiements),), "dictionnaire(un flux des 20 000 moyens de paiement)")
    _verifier_forme_dictionnaire(r, "dictionnaire(un flux)", len(paiements))
    if not _pareil((list(r[0]), list(r[1])), _ref_dictionnaire(paiements)):
        raise AssertionError("Sur un flux (les 20 000 moyens de paiement), dictionnaire donne une mauvaise réponse.")

    # ------------------------------------------------------------------ 200 000 valeurs, 20 000 distinctes
    alea = random.Random(2490)
    grandes = [_Compte(f"V{alea.randrange(20_000):05d}") for _ in range(200_000)]
    _Compte.comparaisons, _Compte.limite = 0, 5 * len(grandes)
    try:
        r = _appel(f, (grandes,), "dictionnaire(200 000 valeurs, dont 20 000 différentes)",
                   conseil_lent="Le correcteur compte les comparaisons (`==`) : vous en faites plus d'un million, parce "
                                "que vous cherchez chaque valeur dans la LISTE des valeurs distinctes (`liste.index(v)`, "
                                "`v in liste`, ou une boucle) : 20 000 comparaisons pour chacune des 200 000 valeurs. "
                                "Cherchez dans un `dict` (clé → numéro) : une seule étape, quelle que soit sa taille.")
    finally:
        _Compte.limite = None
    _verifier_forme_dictionnaire(r, "dictionnaire(200 000 valeurs)", len(grandes))
    attendu = _ref_dictionnaire(grandes)
    if list(r[0]) != attendu[0] or list(r[1]) != attendu[1]:
        raise AssertionError("Sur 200 000 valeurs dont 20 000 différentes, dictionnaire donne une mauvaise réponse.")


# ============================================================================ défi 250 : empaqueter


def _ref_empaqueter(codes, largeur: int, bug: str = "") -> bytes:
    codes = list(codes)
    if bug == "octet_par_code":
        taille = (largeur + 7) // 8
        return b"".join(c.to_bytes(taille, "little") for c in codes)
    total = 0
    for i, c in enumerate(codes):
        if bug == "code_inverse":
            c = int(format(c, f"0{largeur}b")[::-1], 2)
        total |= c << (i * largeur)
    nb = (len(codes) * largeur + 7) // 8
    if bug == "arrondi_bas":
        nb = len(codes) * largeur // 8
        total &= (1 << (8 * nb)) - 1
    if bug == "gros_bout":
        return total.to_bytes(nb, "big")
    octets = total.to_bytes(nb, "little")
    if bug == "msb":
        octets = bytes(int(f"{o:08b}"[::-1], 2) for o in octets)
    return octets


_BUGS_EMPAQUETER = {
    "msb": "Vous remplissez chaque octet à partir du bit de POIDS FORT (0x80) : la position p est le bit p % 8 en "
           "partant du bit de poids FAIBLE, celui qui vaut 1 (`1 << (p % 8)`). C'est l'ordre de Parquet.",
    "code_inverse": "Vous écrivez chaque code à l'envers (son bit de poids fort d'abord) : le bit j du code (celui qui "
                    "vaut `1 << j`) va à la position i · largeur + j. Le code 1 sur 3 bits allume la position 3i, pas "
                    "3i + 2.",
    "gros_bout": "Vos octets sont dans l'ordre inverse (« gros-boutiste », `to_bytes(n, 'big')`) : le premier code est "
                 "dans le PREMIER octet. Utilisez `to_bytes(n, 'little')`.",
    "arrondi_bas": "Il manque le dernier octet quand il n'est pas plein : n · largeur bits demandent ⌈n · largeur / 8⌉ "
                   "octets, soit `(n * largeur + 7) // 8`, le dernier complété par des zéros.",
    "octet_par_code": "Vous écrivez chaque code dans son propre octet (ou ses propres octets) : c'est justement ce qu'on "
                      "veut éviter. Les codes sont COLLÉS : 4 codes de 2 bits tiennent dans un seul octet.",
}


def _cas_250() -> list[tuple[list[int], int]]:
    cas = [([0, 1, 1, 3, 2], 2), ([], 3), ([5], 3), ([1] * 8, 1), ([1] * 9, 1), ([1, 0, 0, 0, 0, 0, 0, 0, 1], 1),
           ([255, 1], 8), ([0x1234], 16), ([5, 5, 5], 3), ([2 ** 64 - 1], 64), ([1, 2 ** 33 - 1], 33), ([3] * 7, 2),
           ([6, 1], 3), ([0] * 5, 4), ([1, 2, 3, 4, 5, 6, 7], 3), ([0x3FF, 0, 0x155], 10)]
    alea = random.Random(250)
    for _ in range(40):
        largeur = alea.choice((1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 20, 31, 32, 40, 63, 64))
        cas.append(([alea.randrange(2 ** largeur) for _ in range(alea.randrange(0, 30))], largeur))
    return cas


def _verifier_forme_octets(r, quoi: str) -> bytes:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, (bytes, bytearray)):
        raise AssertionError(f"{quoi} renvoie {type(r).__name__} ({_court(r, 80)}) : attendu des `bytes` (`bytes(sortie)` "
                             "si vous remplissez un bytearray).")
    return bytes(r)


def _montrer(octets: bytes) -> str:
    if not octets:
        return "b''"
    return octets.hex(" ") if len(octets) <= 24 else f"{octets[:24].hex(' ')}… ({len(octets)} octets)"


def verifier_empaqueter(f) -> None:
    cas = _cas_250()
    obtenus = []
    for codes, largeur in cas:
        quoi = f"empaqueter({_court(codes, 80)}, {largeur})"
        r = _verifier_forme_octets(_appel(f, (list(codes), largeur), quoi,
                                          lignes_max=_lignes(len(codes) * (largeur // 8 + 3))), quoi)
        obtenus.append(r)
    for (codes, largeur), r in zip(cas, obtenus):
        attendu = _ref_empaqueter(codes, largeur)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_empaqueter(c, w, bug) for c, w in cas], _BUGS_EMPAQUETER)
            if not conseil and len(r) != len(attendu):
                conseil = (f"{len(codes)} codes de {largeur} bits font {len(codes) * largeur} bits : il faut "
                           f"{len(attendu)} octets, pas {len(r)}.")
            raise AssertionError(f"empaqueter({_court(codes, 80)}, {largeur}) renvoie {_montrer(r)} au lieu de "
                                 f"{_montrer(attendu)} (en hexadécimal ; `bits_lus` montre les bits dans l'ordre de "
                                 f"lecture). {conseil}".rstrip())

    codes = _ref_dictionnaire(instantane_exemple()["paiement"])[1]
    r = _verifier_forme_octets(_appel(f, (UneSeuleFois(codes), 2), "empaqueter(un flux de 20 000 codes, 2)"),
                               "empaqueter(un flux)")
    if r != _ref_empaqueter(codes, 2):
        raise AssertionError("Sur un flux (les 20 000 codes des moyens de paiement), empaqueter donne une mauvaise "
                             "réponse.")

    for args, attendu, conseil in (
            (([4], 2), ValueError, "4 (0b100) ne tient pas dans 2 bits (de 0 à 3) : le tronquer (`& 0b11`) écrirait 0, "
                                   "et la relecture rendrait une AUTRE valeur, sans prévenir."),
            (([8, 1], 3), ValueError, "8 ne tient pas dans 3 bits : il déborderait sur le code suivant."),
            (([0, 1, 2 ** 64], 64), ValueError, "2**64 ne tient pas dans 64 bits."),
            (([-1], 3), ValueError, "un code négatif n'a pas d'écriture sur `largeur` bits ici."),
            (([1.0], 3), TypeError, "un code est un int, pas un float."),
            (([True], 1), TypeError, "True est un bool, pas un code (même si `True == 1`)."),
            ((["1"], 4), TypeError, "un code est un int, pas un texte."),
            (([None], 4), TypeError, "un code est un int."),
            (([1], 0), ValueError, "une largeur de 0 bit ne peut rien écrire."),
            (([1], 65), ValueError, "la largeur va de 1 à 64 bits."),
            (([1], -2), ValueError, "une largeur négative n'a pas de sens."),
            (([1], 2.0), TypeError, "la largeur est un int, pas un float."),
            (([1], True), TypeError, "la largeur est un int, pas un bool."),
            (([1], "2"), TypeError, "la largeur est un int, pas un texte.")):
        _doit_lever(f, args, f"empaqueter{args!r}", conseil, attendu)


# ============================================================================ défi 251 : encoder_entiers

_MIN_64, _MAX_64 = -(2 ** 63), 2 ** 63 - 1


def _ref_varint(n: int, bug: str = "") -> bytes:
    groupes = []
    while True:
        groupes.append(n & 0x7F)
        n >>= 7
        if n == 0:
            break
    if bug == "gros_bout":
        groupes.reverse()
    dernier = len(groupes) - 1
    if bug == "drapeau_inverse":
        return bytes(g | (0x80 if i == dernier else 0) for i, g in enumerate(groupes))
    return bytes(g | (0x80 if i < dernier else 0) for i, g in enumerate(groupes))


def _ref_zigzag(v: int, bug: str = "") -> int:
    if bug == "abs":
        return abs(v)
    if bug == "signe":
        return 2 * abs(v) + (v < 0)
    if bug == "complement":
        return v & (2 ** 64 - 1)
    return 2 * v if v >= 0 else -2 * v - 1


def _ref_encoder(valeurs, bug: str = "") -> bytes:
    valeurs = list(valeurs)
    corps = b""
    for i, v in enumerate(valeurs):
        if bug == "sans_ecart":
            ecart = v
        elif bug == "ecart_premier":
            ecart = v if i == 0 else v - valeurs[0]
        elif bug == "ecart_inverse":
            ecart = v if i == 0 else valeurs[i - 1] - v
        else:
            ecart = v if i == 0 else v - valeurs[i - 1]
        corps += _ref_varint(_ref_zigzag(ecart, bug), bug)
    n = len(valeurs)
    if bug == "sans_longueur":
        return corps
    if bug == "longueur_fin":
        return corps + _ref_varint(n)
    return _ref_varint(2 * n if bug == "longueur_zigzag" else n, bug) + corps


_BUGS_ENCODER = {
    "gros_bout": "Vos varints commencent par les bits de POIDS FORT : la règle (Protobuf) écrit d'abord les 7 bits de "
                 "DROITE (`n & 0x7F`), puis décale (`n >>= 7`). 300 s'écrit ac 02, pas 02 ac.",
    "drapeau_inverse": "Le bit 0x80 est sur le mauvais octet : il signifie « un autre octet SUIT », donc il est sur tous "
                       "les octets SAUF le dernier. 300 s'écrit ac 02 (0xac = 0x2c | 0x80).",
    "abs": "Vous utilisez `abs(écart)` : -2 et +2 s'écrivent pareil, et la relecture ne peut plus savoir si l'horodatage "
           "a reculé. Le zigzag garde le signe : 2v si v >= 0, -2v - 1 sinon.",
    "signe": "Vous écrivez 2·|v| + (v < 0) : c'est un autre code, pas le zigzag. Le zigzag de -1 vaut 1, celui de 1 vaut "
             "2 : 2v si v >= 0, et -2v - 1 sinon.",
    "complement": "Vous écrivez les nombres négatifs en complément à deux sur 64 bits (`v & (2**64 - 1)`) : -1 devient "
                  "un nombre de 64 bits à 1, soit 10 octets, pour un écart de -1 ! C'est le rôle du zigzag : 2v si v >= 0, "
                  "-2v - 1 sinon (-1 → 1, un seul octet).",
    "sans_longueur": "Il manque le NOMBRE de valeurs au début : le lecteur doit savoir quand s'arrêter. Écrivez "
                     "varint(n) devant le reste (n est connu à la fin de la lecture : préparez le corps, puis collez "
                     "la longueur devant).",
    "longueur_fin": "Le nombre de valeurs est écrit à la FIN : il doit être au début (le lecteur le lit en premier).",
    "longueur_zigzag": "Le nombre de valeurs passe par le zigzag : il est toujours >= 0, il s'écrit tel quel "
                       "(varint(n)) ; seuls les ÉCARTS passent par le zigzag.",
    "ecart_premier": "Vos écarts sont pris par rapport à la PREMIÈRE valeur : pour des horodatages, ils grandissent toute "
                     "la journée (jusqu'à 86 400 : 3 octets). L'écart se prend avec la valeur PRÉCÉDENTE (1 octet).",
    "ecart_inverse": "Vos écarts ont le signe inversé (précédente - valeur) : l'écart est valeur - précédente (positif "
                     "quand le temps avance).",
    "sans_ecart": "Vous écrivez les valeurs elles-mêmes, pas leurs écarts : chaque horodatage prend 5 octets au lieu "
                  "d'un. Chaque valeur (sauf la première) s'écrit sous forme d'écart avec la PRÉCÉDENTE.",
}

_CONSEIL_NEGATIF = ("Sur un nombre NÉGATIF, la boucle d'un varint ne s'arrête jamais : `-1 >> 7` vaut encore -1 (le "
                    "décalage garde le signe). C'est pour ça que chaque écart passe d'abord par le zigzag (2v si "
                    "v >= 0, -2v - 1 sinon), qui donne toujours un nombre >= 0.")


def _cas_251() -> list[list[int]]:
    cas = [[1000, 1003, 1001, 1001], [], [0], [1], [63], [64], [127], [300], [5, 5, 5, 5], [2 ** 40, 2 ** 40 + 1],
           [-1], [-64], [-65], [10, 9, 8, 7], [_MAX_64], [_MIN_64], [_MIN_64, _MAX_64, 0], [0] * 130,
           [1_794_960_000, 1_794_960_004, 1_794_959_990, 1_794_960_050]]
    alea = random.Random(251)
    for _ in range(40):
        v = alea.choice((0, alea.randrange(-1000, 1000), alea.randrange(2 ** 31), -alea.randrange(2 ** 50)))
        liste = []
        for _ in range(alea.randrange(0, 40)):
            v += alea.choice((alea.randrange(-3, 10), alea.randrange(-300, 300), alea.randrange(-2 ** 20, 2 ** 20)))
            liste.append(max(_MIN_64, min(_MAX_64, v)))
        cas.append(liste)
    return cas


def _lignes_251(n: int) -> int:
    return 150 * (n + 1) + 500


def verifier_encoder_entiers(f) -> None:
    cas = _cas_251()
    obtenus = []
    for valeurs in cas:
        quoi = f"encoder_entiers({_court(valeurs, 90)})"
        negatif = any(b < a for a, b in zip([0] + valeurs, valeurs))
        r = _appel(f, (list(valeurs),), quoi, lignes_max=_lignes_251(len(valeurs)),
                   conseil_lent=_CONSEIL_NEGATIF if negatif else _CONSEIL_LENT)
        obtenus.append(_verifier_forme_octets(r, quoi))
    for valeurs, r in zip(cas, obtenus):
        attendu = _ref_encoder(valeurs)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_encoder(v, bug) for v in cas], _BUGS_ENCODER)
            raise AssertionError(f"encoder_entiers({_court(valeurs, 90)}) renvoie {_montrer(r)} au lieu de "
                                 f"{_montrer(attendu)} (en hexadécimal). {conseil}".rstrip())

    evenements = evenements_exemple()
    r = _appel(f, (UneSeuleFois(evenements),), "encoder_entiers(un flux des 50 000 événements)",
               lignes_max=_lignes_251(len(evenements)), conseil_lent=_CONSEIL_NEGATIF)
    r = _verifier_forme_octets(r, "encoder_entiers(un flux)")
    if r != _ref_encoder(evenements):
        raise AssertionError(f"Sur un flux (les 50 000 événements), encoder_entiers renvoie {len(r)} octets différents "
                             f"de ceux attendus ({len(_ref_encoder(evenements))} octets).")

    for args, attendu, conseil in (
            (([1.0],), TypeError, "une valeur est un int, pas un float."),
            (([1, 3.0],), TypeError, "3.0 est un float, même s'il est « entier » : refusé (le lecteur rendrait 3)."),
            (([True],), TypeError, "True est un bool, pas un entier (même si `True == 1`)."),
            ((["3"],), TypeError, "une valeur est un int, pas un texte."),
            (([None],), TypeError, "une valeur est un int."),
            (([2 ** 63],), ValueError, "2**63 dépasse les entiers signés de 64 bits (jusqu'à 2**63 - 1)."),
            (([-2 ** 63 - 1],), ValueError, "-2**63 - 1 est en dessous des entiers signés de 64 bits."),
            (([0, 2 ** 64],), ValueError, "2**64 dépasse les entiers signés de 64 bits.")):
        _doit_lever(f, args, f"encoder_entiers{args!r}", conseil, attendu, lignes_max=5000)


# ============================================================================ défi 252 : codes_huffman


def _ref_longueurs(frequences: dict, bug: str = "") -> dict[str, int]:
    symboles = list(frequences) if bug == "insertion" else sorted(frequences)
    if len(symboles) <= 1:
        return {s: 0 if bug == "vide" else 1 for s in symboles}
    if bug == "shannon":
        total = sum(frequences.values())
        return {s: max(1, math.ceil(math.log2(total / frequences[s]))) for s in symboles}
    longueurs = dict.fromkeys(symboles, 0)
    n = len(symboles)
    if bug == "neufs_avant":                      # à poids égal, les éléments créés passent AVANT les feuilles
        tas = [(frequences[s], 1, i, [s]) for i, s in enumerate(symboles)]
    else:
        tas = [(frequences[s], 0, i, [s]) for i, s in enumerate(symboles)]
    heapq.heapify(tas)
    compteur = n
    while len(tas) > 1:
        pa, _, _, a = heapq.heappop(tas)
        pb, _, _, b = heapq.heappop(tas)
        for s in a + b:
            longueurs[s] += 1
        groupe = 0 if bug == "neufs_avant" else 1
        ordre = -compteur if bug == "recents_avant" else compteur
        heapq.heappush(tas, (pa + pb, groupe, ordre, a + b))
        compteur += 1
    return longueurs


def _ref_codes(frequences: dict, bug: str = "") -> dict[str, str]:
    longueurs = _ref_longueurs(frequences, bug)
    if bug == "vide":
        if len(longueurs) == 1:
            return {s: "" for s in longueurs}
        longueurs = _ref_longueurs(frequences)
    if bug in ("arbre", "arbre_inverse") and len(longueurs) > 1:
        return _ref_arbre(frequences, bug == "arbre_inverse")
    if bug == "tri_frequence":
        ordre = sorted(longueurs, key=lambda s: (longueurs[s], -frequences[s], s))
    else:
        ordre = sorted(longueurs, key=lambda s: (longueurs[s], s))
    codes: dict[str, str] = {}
    code = precedente = 0
    for s in ordre:
        if bug != "sans_decalage":
            code <<= longueurs[s] - precedente
        codes[s] = format(code + (1 if bug == "plus_un_avant" else 0), f"0{longueurs[s]}b")
        code += 1
        precedente = longueurs[s]
    return codes


def _ref_arbre(frequences: dict, inverse: bool) -> dict[str, str]:
    """Les codes lus dans l'arbre de Huffman (0 pour le premier retiré du tas, 1 pour le second) : pas canoniques."""
    symboles = sorted(frequences)
    codes = dict.fromkeys(symboles, "")
    tas = [(frequences[s], 0, i, [s]) for i, s in enumerate(symboles)]
    heapq.heapify(tas)
    compteur = len(tas)
    gauche, droite = ("1", "0") if inverse else ("0", "1")
    while len(tas) > 1:
        pa, _, _, a = heapq.heappop(tas)
        pb, _, _, b = heapq.heappop(tas)
        for s in a:
            codes[s] = gauche + codes[s]
        for s in b:
            codes[s] = droite + codes[s]
        heapq.heappush(tas, (pa + pb, 1, compteur, a + b))
        compteur += 1
    return codes


_BUGS_LONGUEURS = {
    "neufs_avant": "À poids égal, vos éléments CRÉÉS (les fusions) passent avant les feuilles : la règle donne aux "
                   "feuilles les ordres 0 à n - 1, et aux éléments créés les ordres suivants (n, n + 1…), donc à poids "
                   "égal une feuille sort du tas d'abord.",
    "recents_avant": "À poids égal, vos éléments créés les plus RÉCENTS sortent les premiers : l'ordre des éléments "
                     "créés AUGMENTE (n, n + 1, n + 2…), donc le plus ancien sort d'abord.",
    "insertion": "Vos feuilles sont numérotées dans l'ordre du dict reçu : la règle les numérote dans l'ordre de "
                 "`sorted(frequences)`, pour que la réponse ne dépende pas de l'ordre de saisie.",
    "shannon": "Vos longueurs viennent d'une formule (⌈log2(total / fréquence)⌉, le code de Shannon) : elles ne sont pas "
               "optimales. Seules les fusions successives des deux plus petits poids donnent le minimum.",
}

_BUGS_CODES = {
    "arbre": "Vos LONGUEURS sont justes, mais vos codes sont lus dans l'arbre (0 d'un côté, 1 de l'autre) : ce sont des "
             "codes valables, mais pas les codes CANONIQUES. Triez par (longueur, symbole) : le premier vaut 0, puis "
             "+1 à chaque symbole, décalé à gauche quand la longueur augmente.",
    "arbre_inverse": "Vos LONGUEURS sont justes, mais vos codes sont lus dans l'arbre (1 d'un côté, 0 de l'autre) : ce "
                     "sont des codes valables, mais pas les codes CANONIQUES. Triez par (longueur, symbole) : le premier "
                     "vaut 0, puis +1 à chaque symbole, décalé à gauche quand la longueur augmente.",
    "tri_frequence": "À longueur égale, vous rangez les symboles par fréquence : la règle les range par SYMBOLE "
                     "(`sorted(symboles, key=lambda s: (longueur[s], s))`), pour que le lecteur, qui ne connaît que "
                     "les longueurs, retrouve les mêmes codes.",
    "plus_un_avant": "Votre premier code ne vaut pas 0 : vous ajoutez 1 AVANT d'écrire le code. Écrivez-le, puis "
                     "`code += 1` pour le suivant.",
    "sans_decalage": "Vous ne décalez pas le code quand la longueur augmente : `code <<= longueur - precedente` avant de "
                     "l'écrire.",
    "vide": "Un seul symbole doit recevoir le code « 0 » : un code vide ne s'écrit pas, et le texte « xxxx » coderait "
            "0 bit.",
}


def _fibonacci(n: int) -> dict[str, int]:
    a, b = 1, 1
    resultat = {}
    for i in range(n):
        resultat[chr(ord("a") + i)] = a
        a, b = b, a + b
    return resultat


def _cas_252() -> list[dict]:
    from collections import Counter

    cas = [{"a": 5, "b": 2, "c": 1, "d": 1}, {"a": 1, "b": 1, "c": 2, "d": 2}, {"x": 7}, {},
           {"a": 1, "b": 1, "c": 1, "d": 1}, {"e": 1, "d": 1, "c": 1, "b": 1, "a": 1}, {"z": 3, "a": 3, "m": 3},
           {"a": 2, "b": 2, "c": 2, "d": 2, "e": 4, "f": 4}, {"a": 1, "b": 1, "c": 1, "d": 1, "e": 2, "f": 2, "g": 4},
           _fibonacci(20), {"é": 4, "e": 4, " ": 10, "\n": 1}, dict(Counter("abracadabra")),
           {"ab": 3, "a": 3, "b": 1}, dict(Counter(notes_exemple(300)))]
    alea = random.Random(252)
    lettres = [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["é", "è", " ", "\n", "A", "Z"]
    for _ in range(40):
        symboles = alea.sample(lettres, alea.randrange(2, 25))
        cas.append({s: alea.choice((1, 1, 2, 3, 3, alea.randrange(1, 50))) for s in symboles})
    return cas


def _cout(frequences: dict, codes: dict) -> int:
    return sum(frequences[s] * len(codes[s]) for s in frequences)


def _sans_prefixe(codes: dict) -> tuple[str, str] | None:
    """Deux symboles dont le code de l'un est le début du code de l'autre, ou None."""
    tries = sorted(codes.items(), key=lambda p: p[1])
    for (s1, c1), (s2, c2) in zip(tries, tries[1:]):
        if c2.startswith(c1):
            return s1, s2
    return None


def _diagnostic_huffman(frequences: dict, r: dict, quoi: str, obtenus: list, cas: list) -> str:
    attendu = _ref_codes(frequences)
    if set(r) != set(frequences):
        manque, trop = sorted(set(frequences) - set(r)), sorted(set(r) - set(frequences), key=repr)
        return (f"{quoi} n'a pas les bons symboles : manquants {manque}, en trop {trop}. Un code par symbole des "
                "fréquences.")
    if not all(isinstance(c, str) and set(c) <= {"0", "1"} for c in r.values()):
        return f"{quoi} renvoie {_court(r)} : chaque code est un texte de « 0 » et de « 1 » (`format(code, '03b')`)."
    tout = lambda bug: [_ref_codes(c, bug) for c in cas]  # noqa: E731
    if any(c == "" for c in r.values()):
        conseil = _BUGS_CODES["vide"] if len(r) == 1 else "Un code vide ne s'écrit pas."
        return f"{quoi} renvoie {_court(r)} au lieu de {_court(attendu)}. {conseil}"
    paire = _sans_prefixe(r)
    if paire:
        a, b = paire
        conseil = _premier_bug(obtenus, tout, {b: _BUGS_CODES[b] for b in ("sans_decalage", "plus_un_avant")})
        return (f"{quoi} renvoie {_court(r)} : le code de {a!r} ({r[a]}) est le début de celui de {b!r} ({r[b]}). À la "
                f"lecture, impossible de savoir où s'arrête {a!r}. {conseil}".rstrip())
    if _cout(frequences, r) > _cout(frequences, attendu):
        conseil = _premier_bug(obtenus, tout, {"shannon": _BUGS_LONGUEURS["shannon"]})
        return (f"{quoi} renvoie {_court(r)} au lieu de {_court(attendu)} : vos codes coûtent {_cout(frequences, r)} bits "
                f"au lieu de {_cout(frequences, attendu)} (Σ fréquence × longueur) : ils ne sont pas optimaux. "
                f"{conseil}".rstrip())
    longueurs = {s: len(c) for s, c in r.items()}
    if longueurs != _ref_longueurs(frequences):
        conseil = _premier_bug(obtenus, tout, {b: _BUGS_LONGUEURS[b] for b in ("neufs_avant", "recents_avant",
                                                                               "insertion")})
        return (f"{quoi} renvoie {_court(r)} au lieu de {_court(attendu)}. Vos codes sont optimaux (même coût), mais "
                f"plusieurs arbres le sont aussi, et la règle de départage (poids, puis ordre) en choisit un. "
                f"{conseil}".rstrip())
    conseil = _premier_bug(obtenus, tout, {b: _BUGS_CODES[b] for b in ("arbre", "arbre_inverse", "tri_frequence",
                                                                       "plus_un_avant")})
    return (f"{quoi} renvoie {_court(r)} au lieu de {_court(attendu)}. "
            f"{conseil or 'Vos longueurs sont justes, mais pas les codes : relisez la règle des codes canoniques.'}")


def verifier_codes_huffman(f) -> None:
    cas = _cas_252()
    obtenus = []
    for frequences in cas:
        quoi = f"codes_huffman({_court(frequences, 90)})"
        copie = dict(frequences)
        r = _appel(f, (copie,), quoi, lignes_max=50 * (len(frequences) + 2) ** 2 + 2000,
                   conseil_lent="Votre boucle sur le tas ne s'arrête pas ? Chaque tour RETIRE deux éléments et en remet "
                                "UN : le tas raccourcit d'un élément à chaque fusion, jusqu'à n'en garder qu'un.")
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, dict):
            raise AssertionError(f"{quoi} renvoie {type(r).__name__} : attendu un dict {{symbole: code}}.")
        if copie != frequences or list(copie) != list(frequences):
            raise AssertionError(f"{quoi} a modifié le dict des fréquences reçu : il doit rester intact.")
        obtenus.append(dict(r))
    for frequences, r in zip(cas, obtenus):
        if r != _ref_codes(frequences):
            raise AssertionError(_diagnostic_huffman(frequences, r, f"codes_huffman({_court(frequences, 90)})", obtenus,
                                                     cas))

    from collections import Counter

    r = _appel(f, (Counter("abracadabra"),), "codes_huffman(Counter('abracadabra'))")
    if r != _ref_codes(dict(Counter("abracadabra"))):
        raise AssertionError(f"codes_huffman(Counter('abracadabra')) renvoie {_court(r)} au lieu de "
                             f"{_court(_ref_codes(dict(Counter('abracadabra'))))} : un Counter est un dict.")

    for args, attendu, conseil in (
            (({"a": 0},), ValueError, "une fréquence de 0 : le symbole n'apparaît pas, il n'a pas besoin de code."),
            (({"a": -2, "b": 1},), ValueError, "une fréquence est un nombre d'apparitions, >= 1."),
            (({"a": 1.5},), TypeError, "une fréquence est un int (un nombre d'apparitions), pas un float."),
            (({"a": True},), TypeError, "une fréquence est un int, pas un bool."),
            (({"a": "3"},), TypeError, "une fréquence est un int, pas un texte."),
            (({1: 3, 2: 4},), TypeError, "un symbole est un texte (un caractère, le plus souvent)."),
            (({None: 1},), TypeError, "un symbole est un texte."),
            ((["a", "b"],), TypeError, "les fréquences sont un dict (ou un Counter), pas une liste."),
            (("abc",), TypeError, "les fréquences sont un dict : `codes_huffman(Counter(texte))`."),
            ((None,), TypeError, "les fréquences sont un dict.")):
        _doit_lever(f, args, f"codes_huffman{args!r}", conseil, attendu)


# ============================================================================ défi 253 : lire_entiers


def _ref_lire(octets: bytes, bug: str = "") -> list[int]:
    """La lecture d'un fichier BIEN FORMÉ (le correcteur ne rejoue les bugs que sur ceux-là)."""
    i = 0

    def varint() -> int:
        nonlocal i
        groupes = []
        while True:
            octet = octets[i]
            i += 1
            groupes.append(octet & 0x7F)
            if octet < 0x80:
                break
        if bug == "gros_bout":
            groupes.reverse()
        valeur = 0
        for j, g in enumerate(groupes):
            valeur |= g << (7 * j)
        return valeur

    n = varint()
    valeurs, precedente = [], 0
    for _ in range(n):
        z = varint()
        if bug == "sans_zigzag":
            ecart = z
        elif bug == "dezigzag_moins":
            ecart = z // 2 if z % 2 == 0 else -(z // 2)
        elif bug == "dezigzag_plus":
            ecart = z // 2 if z % 2 == 0 else -(z // 2) + 1
        else:
            ecart = z // 2 if z % 2 == 0 else -(z // 2) - 1
        if bug == "sans_cumul":
            valeurs.append(ecart)
        else:
            precedente += ecart
            valeurs.append(precedente)
    return valeurs


_BUGS_LIRE = {
    "sans_zigzag": "Vous oubliez de défaire le zigzag : un écart z se relit z // 2 s'il est pair, et -(z // 2) - 1 s'il "
                   "est impair (1 → -1, 2 → 1, 3 → -2).",
    "dezigzag_moins": "Votre zigzag inverse se trompe de 1 sur les impairs : 1 doit redonner -1 (et pas 0), 3 doit "
                      "redonner -2 : -(z // 2) - 1.",
    "dezigzag_plus": "Votre zigzag inverse se trompe sur les impairs : 1 → -1, 3 → -2, 5 → -3, soit -(z // 2) - 1.",
    "sans_cumul": "Vous rendez les ÉCARTS, pas les valeurs : chaque valeur est la précédente plus son écart (la première, "
                  "0 plus son écart).",
    "gros_bout": "Vous relisez les varints en commençant par les bits de poids fort : le premier octet porte les 7 bits "
                 "de DROITE (`valeur |= (octet & 0x7F) << decalage`, avec decalage = 0, 7, 14…).",
}

_CONSEIL_INDEX = ("Un fichier tronqué n'est pas un bug de VOTRE programme : c'est un fichier faux. Avant de lire un octet, "
                  "vérifiez qu'il existe (`if i >= len(octets): raise ValueError(...)`), et dites ce qui ne va pas.")

_CONSEIL_FLOT = ("Un varint fait au plus 10 octets : au 10e octet qui porte encore le bit 0x80, le fichier est abîmé (ou "
                 "malveillant). Arrêtez-vous là (`for _ in range(10):`), sans fabriquer un nombre de 1,4 million de bits.")


class _Erreur:
    """Une exception, notée comme un résultat : un bug rejoué qui plante sur le même fichier que vous « répond » pareil."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def __eq__(self, autre) -> bool:
        return isinstance(autre, _Erreur)

    __hash__ = None


def _ou_erreur(appel, fichier: str | None = None, lignes_max: int | None = None, quoi: str = ""):
    try:
        return _trace(fichier, appel, lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {_CONSEIL_LENT} Chaque octet lu doit faire avancer "
                             "la position (`i += 1`).") from None
    except Exception as exc:
        return _Erreur(exc)


_BIEN_FORME = "Ce fichier est BIEN FORMÉ (écrit selon les règles du défi 251) : il doit se relire sans erreur."


def verifier_lire_entiers(f) -> None:
    listes = _cas_251() + [evenements_exemple()]
    obtenus = []
    for valeurs in listes:
        fichier = _ref_encoder(valeurs)
        quoi = f"lire_entiers({_montrer(fichier)})"
        r = _ou_erreur(lambda fichier=fichier: f(fichier), _fichier(f), _lignes(len(fichier)), quoi)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, (list, _Erreur)) or isinstance(r, list) and not all(_entier(v) for v in r):
            raise AssertionError(f"{quoi} renvoie {_court(r)} : attendu une liste d'int.")
        obtenus.append(r)
    for valeurs, r in zip(listes, obtenus):
        if isinstance(r, _Erreur) or r != valeurs:
            conseil = _premier_bug(obtenus, lambda bug: [_ou_erreur(lambda v=v: _ref_lire(_ref_encoder(v), bug))
                                                         for v in listes], _BUGS_LIRE)
            quoi = f"lire_entiers({_montrer(_ref_encoder(valeurs))})"
            if isinstance(r, _Erreur):
                raise AssertionError(f"{quoi} lève {_decrire(r.exc)} au lieu de rendre {_court(valeurs)}. "
                                     f"{conseil or _conseil_exception(r.exc) or _BIEN_FORME}")
            raise AssertionError(f"{quoi} renvoie {_court(r)} au lieu de {_court(valeurs)}. {conseil}".rstrip())
    r = _appel(f, (bytearray(b"\x04\xd0\x0f\x06\x03\x00"),), "lire_entiers(bytearray(b'\\x04\\xd0\\x0f\\x06\\x03\\x00'))")
    if r != [1000, 1003, 1001, 1001]:
        raise AssertionError(f"Avec un bytearray, lire_entiers renvoie {_court(r)} au lieu de [1000, 1003, 1001, 1001].")

    fichier = _ref_encoder(evenements_exemple())
    enorme = _ref_varint(10 ** 15) + b"\x00\x00\x00"
    for octets, conseil, lignes in (
            (b"", "Un fichier vide n'a même pas le nombre de valeurs : ValueError. " + _CONSEIL_INDEX, None),
            (b"\x80", "Le nombre de valeurs commence (0x80 : « un octet suit ») mais le fichier s'arrête : ValueError. "
                      + _CONSEIL_INDEX, None),
            (b"\x04\xd0", "Le fichier s'arrête au milieu d'un nombre (0xd0 annonce un octet qui ne vient pas). "
                          + _CONSEIL_INDEX, None),
            (b"\x04\xd0\x0f\x06\x03", "Le fichier annonce 4 valeurs et n'en contient que 3 : c'est le fichier tronqué de "
                                      "lundi. Le lire à moitié, c'est perdre des données sans le savoir.", None),
            (b"\x02\x00", "Le fichier annonce 2 valeurs et n'en contient qu'une.", None),
            (fichier[:-1], "Les 50 000 événements, sans leur dernier octet : refusé.", None),
            (b"\x04\xd0\x0f\x06\x03\x00\x00", "Un octet EN TROP après les 4 valeurs annoncées : deux fichiers collés, ou "
                                              "une longueur abîmée. Vérifiez à la fin que tout a été lu "
                                              "(`if i != len(octets)`).", None),
            (b"\x00\x00", "Le fichier annonce 0 valeur, puis il reste un octet : refusé.", None),
            (fichier + b"\x00", "Les 50 000 événements, plus un octet : refusé.", None),
            (b"\x01" + b"\xff" * 10 + b"\x01", "Une valeur écrite sur 11 octets : un varint de 64 bits (et son signe) en "
                                               "prend 10 au plus. " + _CONSEIL_FLOT, None),
            (b"\xff" * 10 + b"\x01", "Le nombre de valeurs est écrit sur 11 octets : refusé. " + _CONSEIL_FLOT, None),
            (b"\xff" * 200_000, _CONSEIL_FLOT, 3000),
            (enorme, "Le fichier annonce 10**15 valeurs en 3 octets. Ne préparez jamais une liste de la taille annoncée "
                     "(`[0] * n`) : chaque valeur prend au moins un octet, une annonce plus grande que le nombre "
                     "d'octets restants est fausse.", None)):
        _doit_lever(f, (octets,), f"lire_entiers({_montrer(octets)})", conseil, ValueError,
                    lignes_max=lignes, conseil_lent=_CONSEIL_FLOT)

    for valeur, conseil in (("\x04\xd0\x0f\x06\x03\x00", "un texte n'est pas un fichier binaire : open(..., 'rb')."),
                            ([0], "une liste n'est pas des bytes."),
                            (None, "attendu des bytes."),
                            (4, "attendu des bytes, pas un int.")):
        _doit_lever(f, (valeur,), f"lire_entiers({valeur!r})", conseil, TypeError)


# ---------------------------------------------------------------------------- la série

SERIE = Serie(
    code="41",
    titre="Stocker moins sans rien perdre : plages, dictionnaire, bits, varints, Huffman, lecture défensive",
    module_eleve="defis.serie_41",
    module_corrige="defis.corrige_serie_41",
    defis=(
        Defi(248, "plages", "🟢", "—", verifier_plages),
        Defi(249, "dictionnaire", "🟡", "—", verifier_dictionnaire),
        Defi(250, "empaqueter", "🟡", "—", verifier_empaqueter),
        Defi(251, "encoder_entiers", "🟡", "—", verifier_encoder_entiers),
        Defi(252, "codes_huffman", "🔴", "—", verifier_codes_huffman),
        Defi(253, "lire_entiers", "🔴", "—", verifier_lire_entiers),
    ),
)

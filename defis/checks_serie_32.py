"""Vérifications automatiques de la série 32 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est Camille, de l'équipe rétention, qui va bâtir ses campagnes sur vos segments, et le
serveur qui compte ce que chaque calcul lui coûte :

- aux défis 194, 197 et 198, il compte les lignes exécutées dans votre fichier : recalculer une moyenne pour
  chaque valeur, la distance à TOUS les centres à chaque tour, ou les membres d'un segment pour chaque paire
  de points se voit, quelle que soit la machine ;
- partout, il rejoue les erreurs classiques (écart-type en n − 1, lignes standardisées au lieu des colonnes,
  division par zéro sur une colonne constante, égalité donnée au dernier centre, distance de Manhattan,
  centre vide retiré ou mis à zéro, étiquettes d'avant le dernier recentrage, inertie sans carré, distance
  au dernier centre au lieu du plus proche, « >= » qui reprend un centre, point compté dans sa propre
  moyenne, b moyenné sur tous les autres points, moyenne des moyennes, plus gros chiffre brut…) pour nommer
  la vôtre ;
- et il vérifie que vos fonctions ne modifient pas ce qu'elles reçoivent.

Ses points tirés au hasard ont des coordonnées multiples de 1/16 : leurs distances au carré sont EXACTES en
virgule flottante. Quand une égalité dépendrait de l'ordre de vos additions (des centres qui sont des
moyennes, par exemple), il écarte le tirage : il ne vous reproche jamais un arrondi.
"""

from __future__ import annotations

import copy
import math
import random
import sys

from .segments import clients_exemple, generer_clients, generer_points, points_de
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
    """Votre fonction est écrite, mais elle appelle une fonction qui ne l'est pas encore."""
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
        raise AssertionError(f"{quoi} a modifié {nom} reçu(s) (avant : {_court(avant, 100)}, après : "
                             f"{_court(apres, 100)}). Une fonction ne touche pas à ce qu'on lui prête : travaillez sur "
                             "une copie, ou construisez un nouvel objet.")


def _pas_none(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")


def _nombre(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _proche(x, y, tol: float = 1e-9) -> bool:
    return _nombre(x) and _nombre(y) and math.isclose(x, y, rel_tol=tol, abs_tol=tol)


def _memes(r, attendu) -> bool:
    """Deux suites de nombres égales à 1e-9 près."""
    try:
        return len(r) == len(attendu) and all(_proche(x, y) for x, y in zip(r, attendu))
    except TypeError:
        return False


def _memes_lignes(r, attendu) -> bool:
    try:
        return len(r) == len(attendu) and all(_memes(a, b) for a, b in zip(r, attendu))
    except TypeError:
        return False


def _d2(p, q) -> float:
    return sum((a - b) ** 2 for a, b in zip(p, q))


def _seize(rng: random.Random, bas: int = -64, haut: int = 64) -> float:
    """Un multiple de 1/16 entre bas/16 et haut/16 : exact en virgule flottante."""
    return rng.randint(bas, haut) / 16


def _fmt(x: float) -> str:
    return f"{x:.6g}"


# ---------------------------------------------------------------------------- défi 194 : standardiser


def _ref_standardiser(points, variante: str = ""):
    """(z, moyennes, ecarts), ou None si la variante n'a pas de sens ici (n − 1 avec un seul point)."""
    lignes = [tuple(float(x) for x in p) for p in points]
    n, d = len(lignes), len(lignes[0])
    if variante == "lignes":                       # chaque CLIENT standardisé, au lieu de chaque variable
        z = []
        for ligne in lignes:
            m = sum(ligne) / d
            s = math.sqrt(sum((x - m) ** 2 for x in ligne) / d)
            z.append(tuple(0.0 if s == 0 else (x - m) / s for x in ligne))
        return z, None, None
    moyennes = tuple(sum(ligne[j] for ligne in lignes) / n for j in range(d))
    diviseur = n - 1 if variante == "ddof1" else n
    if diviseur == 0:
        return None
    ecarts = tuple(math.sqrt(sum((ligne[j] - moyennes[j]) ** 2 for ligne in lignes) / diviseur) for j in range(d))
    if variante == "minmax":
        bas = [min(ligne[j] for ligne in lignes) for j in range(d)]
        haut = [max(ligne[j] for ligne in lignes) for j in range(d)]
        z = [tuple(0.0 if haut[j] == bas[j] else (ligne[j] - bas[j]) / (haut[j] - bas[j]) for j in range(d))
             for ligne in lignes]
        return z, moyennes, ecarts

    def une(x, m, s):
        if variante == "sans_reduire":
            return x - m
        if s == 0:
            return 0.0
        if variante == "variance":
            return (x - m) / (s * s)
        if variante == "sans_centrer":
            return x / s
        return (x - m) / s

    z = [tuple(une(x, m, s) for x, m, s in zip(ligne, moyennes, ecarts)) for ligne in lignes]
    return z, moyennes, ecarts


_VARIANTES_STANDARDISER = [
    ("ddof1", "l'écart-type est celui de la POPULATION : la racine de la moyenne des (x − moyenne)², en divisant "
              "par n, comme StandardScaler. Vous divisez par n − 1 (`statistics.stdev`, `pandas.std()`)."),
    ("lignes", "chaque VARIABLE (chaque colonne) se standardise séparément, avec SA moyenne et SON écart-type. Vous "
               "standardisez chaque client (chaque ligne) : on ne compare pas des mois à des euros."),
    ("sans_centrer", "z = (x − moyenne) / écart-type : vous oubliez de retirer la moyenne."),
    ("sans_reduire", "z = (x − moyenne) / écart-type : vous oubliez de diviser par l'écart-type."),
    ("variance", "divisez par l'écart-type (la RACINE de la variance), pas par la variance."),
    ("minmax", "c'est une mise à l'échelle min-max ((x − min) / (max − min)). On demande la standardisation : "
               "(x − moyenne) / écart-type."),
]

_CONSEIL_194 = ("Une colonne constante (tous les clients ont 0 appel) a un écart-type nul : ses z valent 0.0, sans "
                "division par zéro.")


def _comparer_standardiser(r, points, quoi: str) -> None:
    _pas_none(r, quoi)
    if not isinstance(r, tuple) or len(r) != 3:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : un tuple (z, moyennes, ecarts) est attendu.")
    z, moyennes, ecarts = r
    z_att, m_att, e_att = _ref_standardiser(points)
    n, d = len(points), len(points[0])
    if not isinstance(z, list):
        raise AssertionError(f"Dans {quoi}, z est un {type(z).__name__} : une list de tuples (un par point) est attendue.")
    if len(z) != n:
        if len(z) == d and n != d:
            raise AssertionError(f"Dans {quoi}, z a {len(z)} éléments, autant que de VARIABLES : il est rangé par colonne. "
                                 f"Renvoyez un tuple par POINT ({n} ici), dans l'ordre des points.")
        raise AssertionError(f"Dans {quoi}, z a {len(z)} éléments au lieu de {n} (un par point).")
    for i, ligne in enumerate(z):
        if not isinstance(ligne, (tuple, list)) or len(ligne) != d or not all(_nombre(x) for x in ligne):
            raise AssertionError(f"Dans {quoi}, z[{i}] vaut {_court(ligne, 80)} : un tuple de {d} nombres est attendu.")
        if any(x != x for x in ligne):
            raise AssertionError(f"Dans {quoi}, z[{i}] contient NaN ({_court(ligne, 80)}). {_CONSEIL_194} (0/0 en "
                                 "float donne NaN, qui contaminerait ensuite toutes les distances.)")
    for nom, valeur in (("moyennes", moyennes), ("ecarts", ecarts)):
        if not isinstance(valeur, (tuple, list)) or not all(_nombre(x) for x in valeur):
            raise AssertionError(f"Dans {quoi}, {nom} vaut {_court(valeur, 80)} : un tuple de {d} nombres (un par "
                                 "variable) est attendu.")
        if len(valeur) != d:
            pourquoi = (" : une valeur par POINT. Standardisez chaque COLONNE (variable), pas chaque ligne."
                        if len(valeur) == n else ".")
            raise AssertionError(f"Dans {quoi}, {nom} a {len(valeur)} valeurs au lieu de {d} (une par variable)"
                                 f"{pourquoi}")
    if _memes(moyennes, e_att) and _memes(ecarts, m_att) and not _memes(moyennes, m_att):
        raise AssertionError(f"{quoi} renvoie les écarts-types à la place des moyennes : l'ordre est (z, moyennes, ecarts).")
    if not _memes(moyennes, m_att):
        j = next(j for j in range(d) if not _proche(moyennes[j], m_att[j]))
        raise AssertionError(f"Dans {quoi}, la moyenne de la colonne n°{j} vaut {moyennes[j]!r} au lieu de "
                             f"{m_att[j]!r} (la moyenne des {n} valeurs de cette colonne).")
    if not _memes(ecarts, e_att):
        ddof1 = _ref_standardiser(points, "ddof1")
        if ddof1 is not None and _memes(ecarts, ddof1[2]):
            raise AssertionError(f"Dans {quoi}, les écarts valent {_court(tuple(ecarts), 80)} au lieu de "
                                 f"{_court(e_att, 80)} : {_VARIANTES_STANDARDISER[0][1]}")
        if _memes(ecarts, tuple(e * e for e in e_att)):
            raise AssertionError(f"Dans {quoi}, `ecarts` contient les variances : l'écart-type est leur RACINE "
                                 "(`math.sqrt`).")
        if all(_proche(e, 1.0) if a == 0 else _proche(e, a) for e, a in zip(ecarts, e_att)):
            j = next(j for j in range(d) if e_att[j] == 0)
            raise AssertionError(f"Dans {quoi}, la colonne n°{j} est constante : son écart-type vaut 0.0, et `ecarts` "
                                 "garde cette vraie valeur (c'est z qui vaut 0.0, sans diviser). Vous renvoyez 1.0.")
        j = next(j for j in range(d) if not _proche(ecarts[j], e_att[j]))
        raise AssertionError(f"Dans {quoi}, l'écart-type de la colonne n°{j} vaut {ecarts[j]!r} au lieu de "
                             f"{e_att[j]!r} : la racine de la moyenne des (x − moyenne)².")
    if _memes_lignes(z, z_att):
        return
    for variante, message in _VARIANTES_STANDARDISER[1:]:
        autre = _ref_standardiser(points, variante)
        if autre is not None and _memes_lignes(z, autre[0]) and not _memes_lignes(autre[0], z_att):
            raise AssertionError(f"{quoi} : {message}")
    i, j = next((i, j) for i in range(n) for j in range(d) if not _proche(z[i][j], z_att[i][j]))
    x = float(points[i][j])
    if e_att[j] == 0:
        raise AssertionError(f"Dans {quoi}, z[{i}][{j}] vaut {z[i][j]!r} : la colonne n°{j} est constante (écart-type "
                             f"nul), ses z valent 0.0.")
    raise AssertionError(f"Dans {quoi}, z[{i}][{j}] vaut {z[i][j]!r} au lieu de ({x!r} − {m_att[j]!r}) / {e_att[j]!r} "
                         f"= {z_att[i][j]!r}.")


def _un_standardiser(f, points, quoi: str, lignes_max: int | None = 2_000_000, conseil_lent: str = "") -> None:
    avant = copy.deepcopy(points)
    r = _appel(f, (points,), quoi, _CONSEIL_194, lignes_max=lignes_max, conseil_lent=conseil_lent)
    _intact(avant, points, quoi, "les points")
    _comparer_standardiser(r, points, quoi)


def verifier_standardiser(f) -> None:
    exemple = [(2, 10, 0), (2, 30, 0), (6, 10, 0), (6, 30, 0)]
    _un_standardiser(f, exemple, f"standardiser({exemple})")
    _un_standardiser(f, points_de(clients_exemple()), "standardiser(points_de(clients_exemple()))")
    for points in ([(5.0, 2.0)], [(1.0,), (3.0,)], [(3.0,), (1.0,), (2.0,)], [[2.0, 10.0], [6.0, 30.0]],
                   [(-1.5, 4.0), (0.5, 4.0), (2.5, 4.0)], [(1, 100), (2, 300), (3, 200), (4, 400)]):
        _un_standardiser(f, points, f"standardiser({points})")
    for points in ([], [(1.0, 2.0), (3.0,)], [(1.0, None), (2.0, 3.0)], [(1.0, 2.0), (float("nan"), 1.0)], [(None,)]):
        _doit_lever(f, (points,), f"standardiser({points})",
                    "Aucun point, des points de longueurs différentes, ou une valeur manquante (None, NaN) : "
                    "ValueError. (`x is None or x != x` repère les deux sortes de trous.)")
    alea = random.Random(194)
    for essai in range(200):
        n, d = alea.randint(1, 10), alea.randint(1, 4)
        constantes = {j: _seize(alea) for j in range(d) if alea.random() < 0.2}
        points = [tuple(constantes.get(j, _seize(alea)) for j in range(d)) for _ in range(n)]
        _un_standardiser(f, points, f"standardiser({_court(points, 120)}) (tiré au hasard n°{essai})")
    n, d = 5000, 5
    gros = [tuple(_seize(alea, -1600, 1600) for _ in range(d)) for _ in range(n)]
    _un_standardiser(f, gros, f"standardiser({n} points à {d} variables)", lignes_max=40 * n * d + 20_000,
                     conseil_lent="Calculez la moyenne et l'écart-type de chaque colonne UNE fois, avant de construire "
                                  "z. Les recalculer pour chaque valeur (`sum(…) / n` à l'intérieur de la boucle sur "
                                  "les points), c'est 5 000 fois le même travail.")


# ---------------------------------------------------------------------------- défi 195 : affecter


def _ref_affecter(points, centres, variante: str = "") -> list[int]:
    etiquettes = []
    for p in points:
        if variante == "manhattan":
            ds = [sum(abs(a - b) for a, b in zip(p, c)) for c in centres]
        elif variante == "premiere":
            ds = [(p[0] - c[0]) ** 2 for c in centres]
        else:
            ds = [_d2(p, c) for c in centres]
        if variante == "max":
            j = max(range(len(ds)), key=lambda i: (ds[i], -i))
        elif variante == "dernier":
            plus_petite = min(ds)
            j = max(i for i, x in enumerate(ds) if x == plus_petite)
        else:
            j = min(range(len(ds)), key=ds.__getitem__)
        etiquettes.append(j)
    return etiquettes


_VARIANTES_AFFECTER = [
    ("dernier", "à égalité de distance, c'est le PLUS PETIT numéro qui gagne (le premier centre de la liste). Avec "
                "`<=`, le dernier ex aequo prend le point : comparez avec `<` strict."),
    ("manhattan", "la distance est euclidienne : la somme des CARRÉS des écarts, pas des valeurs absolues (ça, c'est "
                  "la distance de Manhattan)."),
    ("premiere", "la distance se calcule sur TOUTES les coordonnées, pas sur la première seulement."),
    ("max", "chaque point va au centre le plus PROCHE (la plus petite distance), pas au plus lointain."),
]


def _comparer_affecter(r, points, centres, quoi: str) -> None:
    _pas_none(r, quoi)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list de numéros (un par point) est attendue.")
    if len(r) != len(points):
        raise AssertionError(f"{quoi} renvoie {len(r)} numéros au lieu de {len(points)} (un par point).")
    for x in r:
        if isinstance(x, (tuple, list)):
            raise AssertionError(f"{quoi} renvoie des centres ({_court(x, 60)}) : on attend leur NUMÉRO (0, 1, …), la "
                                 "position du centre dans la liste `centres`.")
        if not isinstance(x, int) or isinstance(x, bool):
            raise AssertionError(f"{quoi} renvoie {_court(x, 60)} : un numéro de centre (un int) est attendu.")
        if not 0 <= x < len(centres):
            raise AssertionError(f"{quoi} renvoie le numéro {x}, mais il n'y a que {len(centres)} centres (de 0 à "
                                 f"{len(centres) - 1}).")
    attendu = _ref_affecter(points, centres)
    if r == attendu:
        return
    for variante, message in _VARIANTES_AFFECTER:
        if r == _ref_affecter(points, centres, variante):
            raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)} : {message}")
    i = next(i for i in range(len(r)) if r[i] != attendu[i])
    dr, da = _d2(points[i], centres[r[i]]), _d2(points[i], centres[attendu[i]])
    if dr == da:
        raise AssertionError(f"Dans {quoi}, le point n°{i} {points[i]} est aussi près du centre {attendu[i]} que du "
                             f"centre {r[i]} : {_VARIANTES_AFFECTER[0][1]}")
    raise AssertionError(f"Dans {quoi}, le point n°{i} {_court(points[i], 60)} est affecté au centre {r[i]} "
                         f"(distance au carré {_fmt(dr)}) au lieu du centre {attendu[i]} (distance au carré {_fmt(da)}).")


def _un_affecter(f, points, centres, quoi: str, lignes_max: int | None = 2_000_000) -> None:
    copies = copy.deepcopy((points, centres))
    r = _appel(f, (points, centres), quoi, lignes_max=lignes_max)
    _intact(copies[0], points, quoi, "les points")
    _intact(copies[1], centres, quoi, "les centres")
    _comparer_affecter(r, points, centres, quoi)


def verifier_affecter(f) -> None:
    cas = [
        ([(0.0, 0.0), (9.0, 1.0), (4.0, 1.0)], [(0.0, 1.0), (8.0, 1.0)]),
        ([(1.0,), (3.0,)], [(0.0,), (2.0,)]),                       # 1 est à 1 de 0 et de 2 : le premier gagne
        ([(5.0,), (2.0,)], [(3.0,), (7.0,), (1.0,)]),
        ([(1.0, 1.0), (5.0, 5.0)], [(1.0, 1.0), (1.0, 1.0)]),       # deux centres identiques : toujours le 1er
        ([(0.0, 0.0)], [(3.0, 0.0), (2.0, 2.0)]),                   # 9 contre 8 ; Manhattan dirait 3 contre 4
        ([(0.0, 5.0)], [(0.0, 0.0), (1.0, 5.0)]),                   # la 1re coordonnée seule ne suffit pas
        ([], [(0.0,)]),
        ([[2.0, 2.0], [7.0, 1.0]], [[0.0, 0.0], [6.0, 0.0]]),
    ]
    for points, centres in cas:
        _un_affecter(f, points, centres, f"affecter({points}, {centres})")
    points = points_de(clients_exemple())
    _un_affecter(f, points, [points[0], points[4], points[8]], "affecter(points, [points[0], points[4], points[8]])")
    for points, centres in (([(1.0,)], []), ([(1.0, 2.0)], [(0.0,)]), ([(1.0,)], [(0.0,), (1.0, 2.0)]),
                            ([(1.0, 2.0), (3.0,)], [(0.0, 0.0)])):
        _doit_lever(f, (points, centres), f"affecter({points}, {centres})",
                    "Aucun centre, ou un point (ou un centre) qui n'a pas le même nombre de coordonnées que le premier "
                    "centre : ValueError. `zip` s'arrêterait au plus court sans rien dire.")
    alea = random.Random(195)
    for essai in range(300):
        d = alea.randint(1, 3)
        centres = [tuple(alea.randint(-8, 8) / 4 for _ in range(d)) for _ in range(alea.randint(1, 5))]
        points = [tuple(alea.randint(-8, 8) / 4 for _ in range(d)) for _ in range(alea.randint(0, 10))]
        _un_affecter(f, points, centres, f"affecter({_court(points, 90)}, {_court(centres, 60)}) (tiré au hasard "
                                         f"n°{essai})")
    gros = generer_points(3000, k=6, graine=195, dim=3)
    _un_affecter(f, gros, gros[:10], "affecter(3 000 points, 10 centres)", lignes_max=None)


# ---------------------------------------------------------------------------- défi 196 : kmeans


def _ref_kmeans(points, centres, max_iter: int, variante: str = ""):
    """(centres, etiquettes, inertie, iterations, douteux) ; douteux = une égalité de distances après le départ."""
    centres = [tuple(float(x) for x in c) for c in centres]
    d = len(centres[0])
    douteux = False

    def aff(cs, depart: bool):
        nonlocal douteux
        etiquettes = []
        for p in points:
            ds = [_d2(p, c) for c in cs]
            j = min(range(len(ds)), key=ds.__getitem__)
            if not depart and len(ds) > 1:
                deux = sorted(ds)[:2]
                if deux[1] - deux[0] <= 1e-9 * (1 + deux[0]):
                    douteux = True
            etiquettes.append(j)
        return etiquettes

    def recentrer(etiquettes, anciens):
        sommes = [[0.0] * d for _ in anciens]
        tailles = [0] * len(anciens)
        for p, e in zip(points, etiquettes):
            tailles[e] += 1
            for j, x in enumerate(p):
                sommes[e][j] += x
        nouveaux = []
        for somme, t, ancien in zip(sommes, tailles, anciens):
            if t:
                nouveaux.append(tuple(s / t for s in somme))
            else:
                nouveaux.append(tuple(0.0 for _ in ancien) if variante == "vide_zero" else ancien)
        return nouveaux

    etiquettes = aff(centres, depart=True)
    iterations = 0
    limite = 1 if variante == "une_seule" else max_iter
    while iterations < limite:
        centres = recentrer(etiquettes, centres)
        iterations += 1
        nouvelles = aff(centres, depart=False)
        stable = nouvelles == etiquettes
        etiquettes = nouvelles
        if stable:
            break
    inertie = float(sum(_d2(p, centres[e]) for p, e in zip(points, etiquettes)))
    return centres, etiquettes, inertie, iterations, douteux


_CONSEIL_196 = ("Un centre peut se retrouver sans aucun point : il garde alors sa place (pas de division par zéro). "
                "Aucun point, aucun centre ou max_iter < 1 : ValueError.")


def _coherentes(points, centres, etiquettes) -> int | None:
    """Le premier point dont l'étiquette n'est pas (à un arrondi près) celle de son centre le plus proche."""
    for i, (p, e) in enumerate(zip(points, etiquettes)):
        ds = [_d2(p, c) for c in centres]
        if ds[e] > min(ds) + 1e-9 * (1 + min(ds)):
            return i
    return None


def _comparer_kmeans(r, points, centres, max_iter: int, quoi: str) -> None:
    _pas_none(r, quoi)
    if not isinstance(r, tuple) or len(r) != 4:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : une Partition(centres, etiquettes, inertie, "
                             "iterations) est attendue (un tuple de 4 éléments).")
    c, e, inertie, iterations = r
    k, n, d = len(centres), len(points), len(centres[0])
    ref_c, ref_e, ref_i, ref_it, _ = _ref_kmeans(points, centres, max_iter)
    if not isinstance(c, list):
        raise AssertionError(f"Dans {quoi}, `centres` est un {type(c).__name__} : une list de tuples est attendue.")
    if len(c) != k:
        message = (" Un centre qui n'a plus aucun point ne disparaît pas : il garde sa place (sinon k change, et le "
                   "centre j n'est plus celui de l'étiquette j)." if len(c) < k else "")
        raise AssertionError(f"Dans {quoi}, il y a {len(c)} centre(s) au lieu de {k} (autant que de centres "
                             f"reçus).{message}")
    for j, centre in enumerate(c):
        if not isinstance(centre, (tuple, list)) or len(centre) != d or not all(_nombre(x) for x in centre):
            raise AssertionError(f"Dans {quoi}, le centre n°{j} vaut {_court(centre, 80)} : un tuple de {d} nombres est "
                                 "attendu.")
        if any(x != x for x in centre):
            raise AssertionError(f"Dans {quoi}, le centre n°{j} contient NaN. {_CONSEIL_196}")
    if not isinstance(e, list) or len(e) != n:
        raise AssertionError(f"Dans {quoi}, `etiquettes` vaut {_court(e, 80)} : une list de {n} numéros (un par point) "
                             "est attendue.")
    if not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < k for x in e):
        x = next(x for x in e if not (isinstance(x, int) and not isinstance(x, bool) and 0 <= x < k))
        raise AssertionError(f"Dans {quoi}, l'étiquette {_court(x, 40)} n'est pas un numéro de centre (un int de 0 à "
                             f"{k - 1}).")
    if not _nombre(inertie):
        raise AssertionError(f"Dans {quoi}, `inertie` vaut {_court(inertie, 60)} : un nombre est attendu.")
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise AssertionError(f"Dans {quoi}, `iterations` vaut {_court(iterations, 60)} : un int est attendu.")
    if iterations > max_iter:
        raise AssertionError(f"Dans {quoi}, iterations = {iterations} alors que max_iter = {max_iter} : max_iter est un "
                             "PLAFOND. Arrivé là, on s'arrête même si des étiquettes changent encore.")
    if iterations < 1:
        raise AssertionError(f"Dans {quoi}, iterations = {iterations} : les centres sont recalculés au moins une fois "
                             "(l'étape 2 a toujours lieu), donc iterations >= 1.")
    i = _coherentes(points, c, e)
    if i is not None:
        ds = [_d2(points[i], centre) for centre in c]
        bon = min(range(k), key=ds.__getitem__)
        raise AssertionError(f"Dans {quoi}, les étiquettes ne sont pas celles des centres renvoyés : le point n°{i} "
                             f"{_court(points[i], 50)} porte l'étiquette {e[i]}, mais le centre renvoyé le plus proche est "
                             f"le n°{bon}. Après le DERNIER recalcul des centres, réaffectez les points (même quand "
                             "vous vous arrêtez parce que max_iter est atteint) : ce qu'on renvoie doit décrire la même "
                             "chose.")
    vraie = sum(_d2(p, c[x]) for p, x in zip(points, e))
    if not _proche(inertie, vraie):
        if _proche(inertie, sum(math.sqrt(_d2(p, c[x])) for p, x in zip(points, e))):
            raise AssertionError(f"Dans {quoi}, l'inertie vaut {inertie!r} au lieu de {vraie!r} : c'est la somme des "
                                 "distances AU CARRÉ (sans racine). C'est elle que k-means fait baisser à chaque étape.")
        raise AssertionError(f"Dans {quoi}, l'inertie vaut {inertie!r} au lieu de {vraie!r} : la somme, sur tous les "
                             "points, de la distance au carré à LEUR centre, calculée avec les centres et les étiquettes "
                             "RENVOYÉS (après le dernier recalcul).")
    memes_centres = all(_memes(a, b) for a, b in zip(c, ref_c))
    if memes_centres and e == ref_e and iterations == ref_it:
        return
    zero = _ref_kmeans(points, centres, max_iter, "vide_zero")
    if not memes_centres and all(_memes(a, b) for a, b in zip(c, zero[0])):
        raise AssertionError(f"Dans {quoi}, un centre sans point a été mis à zéro : il doit garder sa place (le mettre à "
                             "0 l'envoie au milieu des données, où il volerait des points au hasard).")
    une = _ref_kmeans(points, centres, max_iter, "une_seule")
    if ref_it > 1 and iterations == 1 and all(_memes(a, b) for a, b in zip(c, une[0])):
        raise AssertionError(f"{quoi} s'arrête après un seul recalcul des centres, alors que des étiquettes changent "
                             f"encore (il en faut {ref_it}) : recommencez tant qu'une étiquette change (et que "
                             "max_iter n'est pas atteint).")
    if memes_centres and e == ref_e:
        if iterations == ref_it + 1:
            raise AssertionError(f"Dans {quoi}, iterations = {iterations} au lieu de {ref_it} : c'est le nombre de "
                                 "RECALCULS des centres. Dès que la réaffectation ne change aucune étiquette, c'est fini "
                                 "(attendre que les centres ne bougent plus coûte un tour de trop ; compter les "
                                 "affectations aussi).")
        raise AssertionError(f"Dans {quoi}, iterations = {iterations} au lieu de {ref_it} (le nombre de recalculs des "
                             "centres).")
    j = next((j for j in range(k) if not _memes(c[j], ref_c[j])), None)
    if j is not None:
        raise AssertionError(f"Dans {quoi}, le centre n°{j} vaut {_court(tuple(c[j]), 80)} au lieu de "
                             f"{_court(ref_c[j], 80)} (après {ref_it} recalcul(s) : la moyenne de SES points).")
    raise AssertionError(f"{quoi} renvoie les étiquettes {_court(e, 80)} au lieu de {_court(ref_e, 80)}.")


def _un_kmeans(f, points, centres, max_iter: int | None, quoi: str, lignes_max: int | None = 3_000_000) -> None:
    copies = copy.deepcopy((points, centres))
    kwargs = {} if max_iter is None else {"max_iter": max_iter}
    r = _appel(f, (points, centres), quoi, _CONSEIL_196, lignes_max=lignes_max, kwargs=kwargs,
               conseil_lent="Votre fonction ne s'arrête pas : arrêtez-vous dès qu'aucune étiquette ne change, et au "
                            "plus tard après max_iter recalculs.")
    _intact(copies[0], points, quoi, "les points")
    _intact(copies[1], centres, quoi, "les centres (la liste reçue)")
    _comparer_kmeans(r, points, centres, 100 if max_iter is None else max_iter, quoi)


def verifier_kmeans(f) -> None:
    six = [(0.0,), (1.0,), (2.0,), (10.0,), (11.0,), (12.0,)]
    for centres, max_iter in (([(0.0,), (1.0,)], None), ([(0.0,), (1.0,)], 1), ([(0.0,), (1.0,)], 2),
                              ([(12.0,), (0.0,)], None), ([(5.0,)], None)):
        _un_kmeans(f, six, centres, max_iter, f"kmeans({six}, {centres}{'' if max_iter is None else f', max_iter={max_iter}'})")
    cas = [
        ([(0.0,), (1.0,)], [(0.0,), (100.0,)], None),                       # un centre sans point ne bouge pas
        ([(0.0,), (1.0,), (10.0,)], [(0.0,), (0.0,)], None),                # deux centres identiques au départ
        ([(0.0,), (4.0,)], [(0.0,), (4.0,), (100.0,)], None),               # plus de centres que de points
        ([[0.0, 0.0], [0.0, 2.0], [8.0, 0.0], [8.0, 2.0]], [[0.0, 0.0], [8.0, 2.0]], None),
        ([(0.0,), (1.0,), (3.0,), (7.0,), (15.0,), (31.0,)], [(0.0,), (1.0,)], 2),
        ([(0.0,), (1.0,), (3.0,), (7.0,), (15.0,), (31.0,)], [(0.0,), (1.0,)], 3),
        ([(0.0,), (1.0,), (3.0,), (7.0,), (15.0,), (31.0,)], [(0.0,), (1.0,)], None),
        ([(2.0, 2.0), (2.0, 2.0), (2.0, 2.0)], [(2.0, 2.0), (5.0, 5.0)], None),
    ]
    for points, centres, max_iter in cas:
        _un_kmeans(f, points, centres, max_iter,
                   f"kmeans({points}, {centres}{'' if max_iter is None else f', max_iter={max_iter}'})")
    points = points_de(clients_exemple())
    z = _ref_standardiser(points)[0]
    for depart in ((0, 4, 8), (0, 6, 9), (0, 1, 2)):
        _un_kmeans(f, points, [points[i] for i in depart], None, f"kmeans(points_de(clients_exemple()), points {depart})")
        _un_kmeans(f, z, [z[i] for i in depart], None, f"kmeans(clients_exemple() standardisés, points {depart})")
    for points, centres, max_iter in (([(1.0,)], [(0.0,)], 0), ([(1.0,)], [(0.0,)], -1), ([], [(0.0,)], 10),
                                      ([(1.0,)], [], 10), ([(1.0, 2.0)], [(0.0,)], 10)):
        _doit_lever(f, (points, centres), f"kmeans({points}, {centres}, max_iter={max_iter})",
                    "max_iter < 1, aucun point, aucun centre, ou des longueurs différentes : ValueError.",
                    kwargs={"max_iter": max_iter})
    alea = random.Random(196)
    essai = 0
    while essai < 150:
        d, n = alea.randint(1, 3), alea.randint(1, 25)
        points = [tuple(_seize(alea) for _ in range(d)) for _ in range(n)]
        k = alea.randint(1, 5)
        if alea.random() < 0.6:
            centres = [points[alea.randrange(n)] for _ in range(k)]
        else:
            centres = [tuple(_seize(alea) for _ in range(d)) for _ in range(k)]
        max_iter = alea.choice((1, 2, 3, 100))
        if _ref_kmeans(points, centres, max_iter)[4]:
            continue                                   # une égalité qui dépendrait de vos arrondis : on retire
        _un_kmeans(f, points, centres, max_iter, f"kmeans({_court(points, 90)}, {_court(centres, 60)}, "
                                                 f"max_iter={max_iter}) (tiré au hasard n°{essai})")
        essai += 1
    graine = 196
    while True:
        gros = generer_points(400, k=4, graine=graine, dim=2)
        if not _ref_kmeans(gros, gros[:4], 100)[4]:
            break
        graine += 1
    _un_kmeans(f, gros, gros[:4], None, f"kmeans(generer_points(400, k=4, graine={graine}), les 4 premiers points)",
               lignes_max=None)


# ---------------------------------------------------------------------------- défi 197 : kmeans_plus_plus


def _ref_kmpp(points, k: int, tirages, variante: str = ""):
    """(numéros choisis ou None si ValueError, douteux) ; douteux = un seuil à 1e-9 près d'une somme cumulée."""
    n = len(points)
    if variante == "n_moins_un":
        premier = int(tirages[0] * (n - 1))
    elif variante == "round":
        premier = round(tirages[0] * n)
        if premier >= n:
            return None, False
    else:
        premier = int(tirages[0] * n)
    choisis = [premier]

    def poids_de(p, q):
        return math.sqrt(_d2(p, q)) if variante == "simple" else _d2(p, q)

    poids = [poids_de(p, points[premier]) for p in points]
    douteux = False
    for i in range(1, k):
        if variante == "dernier":
            poids = [poids_de(p, points[choisis[-1]]) for p in points]
        total = sum(poids)
        if total == 0:
            return None, douteux
        seuil = tirages[i] * total
        cumul, choisi = 0.0, None
        for j, w in enumerate(poids):
            cumul += w
            if abs(cumul - seuil) <= 1e-9 * total:
                douteux = True
            if choisi is None and (cumul >= seuil if variante == "large" else cumul > seuil):
                choisi = j
        if choisi is None:                             # inaccessible avec des tirages < 1
            choisi = n - 1
        choisis.append(choisi)
        if variante != "dernier":
            nouveau = points[choisi]
            poids = [min(w, poids_de(p, nouveau)) for w, p in zip(poids, points)]
    return choisis, douteux


_VARIANTES_KMPP = [
    ("large", "le point choisi est le premier dont la somme cumulée est STRICTEMENT plus grande que le seuil (`>`). "
              "Avec `>=`, un point de poids 0, donc déjà centre, peut être choisi quand le seuil tombe pile sur une "
              "somme cumulée."),
    ("dernier", "D² est la distance au carré au centre déjà choisi le PLUS PROCHE, pas au dernier choisi : un point "
                "collé au 1er centre ne doit plus avoir de chances, même si le 2e est loin de lui. Gardez D² pour chaque "
                "point et mettez-le à jour avec `min(ancien, distance²(p, nouveau))`."),
    ("simple", "les poids sont les distances AU CARRÉ (D²), pas les distances : c'est ce carré qui pousse k-means++ "
               "vers les points lointains."),
    ("n_moins_un", "le 1er centre est le point n° `int(tirages[0] * n)`, avec n le nombre de points (pas n − 1)."),
    ("round", "le 1er centre est le point n° `int(tirages[0] * n)` : on tronque (`int`), on n'arrondit pas (`round`)."),
]


def _detail_tour(points, choisis_avant: list[int], tirage: float) -> str:
    """Le détail d'un tour de k-means++, pour un petit exemple."""
    poids = [min(_d2(p, points[c]) for c in choisis_avant) for p in points]
    total = sum(poids)
    cumul, sommes = 0.0, []
    for w in poids:
        cumul += w
        sommes.append(cumul)
    return (f"les D² étaient {[float(_fmt(w)) for w in poids]}, total {_fmt(total)}, seuil {_fmt(tirage)} × "
            f"{_fmt(total)} = {_fmt(tirage * total)}, sommes cumulées {[float(_fmt(s)) for s in sommes]}")


def _comparer_kmpp(r, points, k: int, tirages, quoi: str) -> None:
    _pas_none(r, quoi)
    attendu, _ = _ref_kmpp(points, k, tirages)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list de numéros de points est attendue.")
    for x in r:
        if isinstance(x, (tuple, list)):
            raise AssertionError(f"{quoi} renvoie des points ({_court(x, 60)}) : on attend leurs NUMÉROS (des int), "
                                 "la position de chaque point choisi dans `points`.")
        if not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < len(points):
            raise AssertionError(f"{quoi} renvoie {_court(x, 40)} : un numéro de point (un int de 0 à "
                                 f"{len(points) - 1}) est attendu.")
    if len(r) != k:
        raise AssertionError(f"{quoi} renvoie {len(r)} numéros au lieu de k = {k}.")
    if r == attendu:
        return
    for variante, message in _VARIANTES_KMPP:
        autre, _ = _ref_kmpp(points, k, tirages, variante)
        if autre == r:
            raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)} : {message}")
    if r[0] != attendu[0]:
        raise AssertionError(f"{quoi} commence par le point n°{r[0]} au lieu du n°{attendu[0]} = int({tirages[0]!r} × "
                             f"{len(points)}). Utilisez les tirages fournis, pas `random`.")
    i = next(i for i in range(k) if r[i] != attendu[i])
    if r[i] in r[:i]:
        raise AssertionError(f"{quoi} choisit deux fois le point n°{r[i]} : un point déjà centre a D² = 0, il ne doit "
                             "jamais être repris (`cumul > seuil`, strict).")
    detail = f" ({_detail_tour(points, attendu[:i], tirages[i])})" if len(points) <= 12 else ""
    raise AssertionError(f"{quoi} : au tour n°{i}, vous choisissez le point n°{r[i]} au lieu du n°{attendu[i]}{detail}. "
                         "Le premier point dont la somme cumulée dépasse STRICTEMENT le seuil est choisi.")


def _un_kmpp(f, points, k: int, tirages, quoi: str, lignes_max: int | None = 2_000_000, conseil_lent: str = "") -> None:
    copies = copy.deepcopy((points, tirages))
    r = _appel(f, (points, k, tirages), quoi, "Des points peuvent être identiques : leur D² vaut 0. Si la somme des D² "
                                              "vaut 0 avant d'avoir k centres, levez ValueError.",
               lignes_max=lignes_max, conseil_lent=conseil_lent)
    _intact(copies[0], points, quoi, "les points")
    _intact(copies[1], tirages, quoi, "les tirages")
    _comparer_kmpp(r, points, k, tirages, quoi)


def verifier_kmeans_plus_plus(f) -> None:
    cas = [
        ([(0.0,), (1.0,), (5.0,), (6.0,), (20.0,)], 3, [0.0, 0.5, 0.5]),
        ([(0.0,), (0.0,), (4.0,)], 2, [0.0, 0.0]),                      # « >= » reprendrait le point n°0
        ([(3.0,), (0.0,), (3.0,)], 2, [0.0, 0.0]),
        ([(0.0,), (2.0,), (2.0,), (2.0,), (2.0,)], 2, [0.0, 0.25]),      # le seuil tombe pile sur une somme
        ([(0.0,), (1.0,), (9.0,), (10.0,), (20.0,)], 3, [0.0, 0.75, 0.5]),   # le plus proche, pas le dernier
        ([(0.0,), (1.0,), (2.0,), (3.0,), (4.0,)], 2, [0.0, 0.5]),       # D², pas D
        ([(0.0,), (1.0,), (2.0,), (3.0,)], 1, [0.9375]),
        ([(0.0,), (8.0,), (1.0,), (9.0,)], 4, [0.5, 0.5, 0.5, 0.5]),
        ([(1.0, 1.0), (1.0, 1.0), (3.0, 1.0), (1.0, 5.0)], 3, [0.0, 0.5, 0.0, 0.75]),
        ([(0.0,), (1.0,), (2.0,)], 2, (0.75, 0.25)),
        ([(0.0,), (1.0,), (2.0,), (3.0,), (4.0,)], 1, [0.75]),          # int(3.75) = 3, round(3.75) = 4
    ]
    for points, k, tirages in cas:
        _un_kmpp(f, points, k, tirages, f"kmeans_plus_plus({points}, {k}, {tirages})")
    points = points_de(clients_exemple())
    _un_kmpp(f, points, 3, [0.0, 0.5, 0.5], "kmeans_plus_plus(points_de(clients_exemple()), 3, [0.0, 0.5, 0.5])")
    for points, k, tirages in (([(0.0,), (1.0,)], 0, [0.5]), ([(0.0,), (1.0,)], 3, [0.5, 0.5, 0.5]),
                               ([(0.0,), (1.0,)], -1, []), ([(0.0,), (1.0,), (2.0,)], 3, [0.5, 0.5]),
                               ([(1.0,), (1.0,), (1.0,)], 2, [0.0, 0.5]),
                               ([(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)], 3, [0.0, 0.5, 0.5])):
        _doit_lever(f, (points, k, tirages), f"kmeans_plus_plus({points}, {k}, {tirages})",
                    "k hors de 1..n, moins de k tirages, ou plus assez de points différents (la somme des D² vaut 0) : "
                    "ValueError.")
    alea = random.Random(197)
    essai = 0
    while essai < 300:
        d, n = alea.choice((1, 1, 2, 3)), alea.randint(1, 12)
        points = [tuple(alea.randint(-6, 6) / 2 for _ in range(d)) for _ in range(n)]
        k = alea.randint(1, n)
        tirages = [alea.randint(0, 15) / 16 for _ in range(k)]
        attendu, douteux = _ref_kmpp(points, k, tirages)
        if d > 1 and douteux:
            continue                     # en 2D, `math.dist(p, q) ** 2` n'est pas exact : pas de seuil « pile »
        quoi = f"kmeans_plus_plus({_court(points, 90)}, {k}, {tirages}) (tiré au hasard n°{essai})"
        if attendu is None:
            _doit_lever(f, (points, k, tirages), quoi, "Plus assez de points différents : la somme des D² vaut 0 avant "
                                                       "d'avoir k centres. ValueError.")
        else:
            _un_kmpp(f, points, k, tirages, quoi)
        essai += 1
    n, k, graine = 1200, 60, 197
    gros = generer_points(n, k=6, graine=graine, dim=2)
    while True:
        tirages = [alea.randint(0, 1023) / 1024 for _ in range(k)]
        attendu, douteux = _ref_kmpp(gros, k, tirages)
        if attendu is not None and not douteux:
            break
    _un_kmpp(f, gros, k, tirages, f"kmeans_plus_plus({n} points, k={k}, tirages)", lignes_max=18 * n * k + 50_000,
             conseil_lent="Recalculer à chaque tour la distance de chaque point à TOUS les centres déjà choisis, c'est k "
                          "fois trop. Gardez D² pour chaque point, et mettez-le à jour avec le SEUL nouveau centre : "
                          "`min(ancien, distance²(p, nouveau))`.")


# ---------------------------------------------------------------------------- défi 198 : silhouette


def _ref_silhouette(points, etiquettes, variante: str = "") -> float:
    membres: dict = {}
    for i, e in enumerate(etiquettes):
        membres.setdefault(e, []).append(i)
    dist = _d2 if variante == "carre" else math.dist
    valeurs = []
    for i, p in enumerate(points):
        propre = etiquettes[i]
        taille = len(membres[propre])
        if taille == 1 and variante != "singleton_un":
            valeurs.append(0.0)
            continue
        sommes = {e: sum(dist(p, points[j]) for j in idx) for e, idx in membres.items()}
        if taille == 1:
            a = 0.0
        elif variante == "avec_soi":
            a = sommes[propre] / taille
        else:
            a = sommes[propre] / (taille - 1)
        if variante == "b_tous":
            autres = [j for j in range(len(points)) if etiquettes[j] != propre]
            b = sum(dist(p, points[j]) for j in autres) / len(autres)
        elif variante == "b_point":
            b = min(dist(p, points[j]) for j in range(len(points)) if etiquettes[j] != propre)
        else:
            b = min(sommes[e] / len(idx) for e, idx in membres.items() if e != propre)
        plus_grand = max(a, b)
        if variante == "sans_normaliser":
            valeurs.append(b - a)
        else:
            valeurs.append(0.0 if plus_grand == 0 else (b - a) / plus_grand)
    if variante == "macro":
        return sum(sum(valeurs[i] for i in idx) / len(idx) for idx in membres.values()) / len(membres)
    return sum(valeurs) / len(points)


_VARIANTES_SILHOUETTE = [
    ("carre", "la silhouette se calcule avec les VRAIES distances, avec la racine (`math.dist(p, q)`). Des distances "
              "au carré gonflent les plus lointaines."),
    ("avec_soi", "a(i) est la distance moyenne aux AUTRES points du segment : le point lui-même (à distance 0) n'en fait "
                 "pas partie. Divisez par taille − 1, pas par taille."),
    ("singleton_un", "un point SEUL dans son segment a s = 0 (sans autre point dans son segment, a n'existe pas). Avec "
                     "a = 0, la formule donnerait 1 : le meilleur score possible pour un segment d'un client."),
    ("b_tous", "b(i) se calcule segment par segment : la distance moyenne aux points de CHAQUE autre segment, puis la "
               "plus petite (le segment voisin). Pas la moyenne sur tous les points des autres segments."),
    ("b_point", "b(i) est une distance MOYENNE aux points du segment voisin, pas la distance au point le plus proche."),
    ("macro", "le résultat est la moyenne de s sur TOUS les points, pas la moyenne des moyennes de chaque segment (un "
              "segment de 2 clients pèserait autant qu'un de 2 000)."),
    ("sans_normaliser", "s(i) = (b − a) / max(a, b) : sans la division, s n'est pas entre -1 et 1."),
]


def _comparer_silhouette(r, points, etiquettes, quoi: str) -> None:
    _pas_none(r, quoi)
    if not _nombre(r):
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : un nombre (la silhouette moyenne) est attendu.")
    if r != r:
        raise AssertionError(f"{quoi} renvoie NaN : si max(a, b) vaut 0 (des points identiques dans deux segments), "
                             "s vaut 0.")
    attendu = _ref_silhouette(points, etiquettes)
    if _proche(r, attendu):
        return
    for variante, message in _VARIANTES_SILHOUETTE:
        autre = _ref_silhouette(points, etiquettes, variante)
        if _proche(r, autre) and not _proche(autre, attendu):
            raise AssertionError(f"{quoi} renvoie {r!r} au lieu de {attendu!r} : {message}")
    raise AssertionError(f"{quoi} renvoie {r!r} au lieu de {attendu!r}.")


def _une_silhouette(f, points, etiquettes, quoi: str, lignes_max: int | None = 2_000_000,
                    conseil_lent: str = "") -> None:
    copies = copy.deepcopy((points, etiquettes))
    r = _appel(f, (points, etiquettes), quoi,
               "Les étiquettes peuvent valoir 3, 7 ou « A » (pas forcément 0, 1, …) : rangez les points dans un dict "
               "{étiquette: numéros}. Un point seul dans son segment a s = 0, sans division par zéro.",
               lignes_max=lignes_max, conseil_lent=conseil_lent)
    _intact(copies[0], points, quoi, "les points")
    _intact(copies[1], etiquettes, quoi, "les étiquettes")
    _comparer_silhouette(r, points, etiquettes, quoi)


def verifier_silhouette(f) -> None:
    quatre = [(0.0,), (2.0,), (10.0,), (12.0,)]
    cas = [
        (quatre, [0, 0, 1, 1]), (quatre, ["A", "B", "A", "B"]), (quatre, [7, 7, 3, 3]),
        ([(0.0,), (0.0,), (10.0,)], [0, 0, 1]),                          # un point seul : s = 0
        ([(0.0,), (0.0,)], [0, 1]), ([(0.0,), (0.0,), (0.0,), (0.0,)], [0, 0, 1, 1]),   # max(a, b) = 0
        ([(0.0,), (1.0,), (5.0,), (6.0,), (20.0,)], [0, 0, 1, 1, 1]),
        ([(0.0,), (1.0,), (2.0,), (10.0,), (11.0,), (30.0,)], [0, 0, 0, 1, 1, 2]),   # b : le segment VOISIN
        ([(0.0, 0.0), (0.0, 1.0), (4.0, 0.0), (4.0, 1.0)], [0, 0, 1, 1]),
        ([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]], ["x", "x", "y"]),
        ([(0.0,), (1.0,), (2.0,), (3.0,), (50.0,), (51.0,)], [0, 0, 0, 0, 1, 2]),   # macro ≠ moyenne sur tous
    ]
    for points, etiquettes in cas:
        _une_silhouette(f, points, etiquettes, f"silhouette({points}, {etiquettes})")
    points = points_de(clients_exemple())
    for etiquettes in ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], [0, 0, 0, 2, 1, 1, 1, 1, 2, 2, 2, 2]):
        _une_silhouette(f, points, etiquettes, f"silhouette(points_de(clients_exemple()), {etiquettes})")
    for points, etiquettes in (([(0.0,), (1.0,)], [0, 0]), ([(0.0,), (1.0,)], [0, 1, 1]), ([], []),
                               ([(0.0,), (1.0,), (2.0,)], [0, 1])):
        _doit_lever(f, (points, etiquettes), f"silhouette({points}, {etiquettes})",
                    "Moins de 2 segments (pas de voisin : b n'existe pas), ou pas autant d'étiquettes que de points : "
                    "ValueError.")
    alea = random.Random(198)
    for essai in range(200):
        d, n = alea.randint(1, 3), alea.randint(2, 12)
        points = [tuple(_seize(alea) for _ in range(d)) for _ in range(n)]
        noms = alea.choice(([0, 1, 2, 3], ["A", "B", "C"], [5, 9]))
        etiquettes = [alea.choice(noms) for _ in range(n)]
        if len(set(etiquettes)) < 2:
            etiquettes[0], etiquettes[-1] = noms[0], noms[1]
        _une_silhouette(f, points, etiquettes, f"silhouette({_court(points, 90)}, {etiquettes}) (tiré au hasard "
                                               f"n°{essai})")
    n = 250
    gros = generer_points(n, k=4, graine=198, dim=2)
    etiquettes = _ref_affecter(gros, gros[:4])
    _une_silhouette(f, gros, etiquettes, f"silhouette({n} points, 4 segments)", lignes_max=30 * n * n + 50_000,
                    conseil_lent="n² distances suffisent : rangez les numéros des points par segment UNE fois (un dict), "
                                 "puis, pour chaque point, sommez ses distances aux membres de chaque segment. Refaire la "
                                 "liste des membres pour chaque paire de points, c'est n³.")


# ---------------------------------------------------------------------------- défi 199 : decrire_segments


def _ref_decrire(clients, etiquettes, variables, variante: str = "") -> list[tuple]:
    variables = list(variables)
    if not clients:
        return []
    n = len(clients)
    globales = {v: sum(c[v] for c in clients) / n for v in variables}
    diviseur = n - 1 if variante == "ddof1" and n > 1 else n
    ecarts = {v: math.sqrt(sum((c[v] - globales[v]) ** 2 for c in clients) / diviseur) for v in variables}
    groupes: dict = {}
    for client, e in zip(clients, etiquettes):
        groupes.setdefault(e, []).append(client)
    fiches = []
    for numero, membres in groupes.items():
        taille = len(membres)
        moyennes = {v: sum(c[v] for c in membres) / taille for v in variables}
        if variante == "sigma_segment":
            propres = {v: math.sqrt(sum((c[v] - moyennes[v]) ** 2 for c in membres) / taille) for v in variables}
        meilleur = None
        for v in variables:
            if variante == "brut":
                z = moyennes[v] - globales[v]
            elif variante == "plus_grande":
                z = moyennes[v]
            elif variante == "sigma_segment":
                z = 0.0 if propres[v] == 0 else (moyennes[v] - globales[v]) / propres[v]
            else:
                z = 0.0 if ecarts[v] == 0 else (moyennes[v] - globales[v]) / ecarts[v]
            if variante == "abs":
                z = abs(z)
            if meilleur is None or abs(z) > abs(meilleur[1]):
                meilleur = (v, z)
        part = taille / n * (100 if variante == "pourcent" else 1)
        fiches.append((numero, taille, part, sum(c["churn"] for c in membres) / taille, moyennes, meilleur))
    if variante == "reverse":
        fiches.sort(key=lambda s: (s[3], s[0]), reverse=True)
    elif variante == "croissant":
        fiches.sort(key=lambda s: (s[3], s[0]))
    elif variante == "numero":
        fiches.sort(key=lambda s: s[0])
    else:
        fiches.sort(key=lambda s: (-s[3], s[0]))
    return fiches


_VARIANTES_DISTINCTIF = [
    ("plus_grande", "la variable distinctive n'est pas celle qui a la plus grande moyenne (93 € est un gros chiffre, "
                    "mais tout le monde paie à peu près ça) : c'est celle dont la moyenne s'écarte le plus de celle de "
                    "TOUS les clients, en écarts-types."),
    ("brut", "comparer des écarts bruts, c'est comparer des mois, des euros et des appels : l'écart en euros gagne "
             "toujours. Divisez chaque écart par l'écart-type GLOBAL de sa variable : z = (moyenne du segment − "
             "moyenne globale) / écart-type global."),
    ("sigma_segment", "z se divise par l'écart-type de TOUS les clients (global), pas par celui du segment : un "
                      "segment très homogène aurait sinon des z immenses."),
]


def _comparer_decrire(r, clients, etiquettes, variables, quoi: str) -> None:
    _pas_none(r, quoi)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list de Segment est attendue.")
    attendu = _ref_decrire(clients, etiquettes, variables)
    for x in r:
        if not isinstance(x, tuple) or len(x) != 6:
            raise AssertionError(f"Dans {quoi}, l'élément {_court(x, 80)} n'est pas un Segment(numero, taille, part, "
                                 "taux_churn, moyennes, distinctif) (un tuple de 6 valeurs).")
    numeros = [x[0] for x in r]
    if len(set(numeros)) != len(numeros):
        double = next(x for x in numeros if numeros.count(x) > 1)
        raise AssertionError(f"Dans {quoi}, le segment {double!r} a plusieurs fiches : une seule par étiquette.")
    voulus = {x[0] for x in attendu}
    if set(numeros) != voulus:
        manque = sorted(voulus - set(numeros))
        if manque:
            raise AssertionError(f"Dans {quoi}, il manque la fiche du segment {manque[0]!r} (les étiquettes présentes : "
                                 f"{sorted(voulus)}).")
        raise AssertionError(f"Dans {quoi}, la fiche du segment {sorted(set(numeros) - voulus)[0]!r} ne devrait pas y "
                             "être : aucun client ne porte cette étiquette.")
    par_numero = {x[0]: x for x in attendu}
    variantes = {nom: {x[0]: x for x in _ref_decrire(clients, etiquettes, variables, nom)}
                 for nom in ("pourcent", "abs", "ddof1", "brut", "plus_grande", "sigma_segment")}
    for x in r:
        y = par_numero[x[0]]
        numero = x[0]
        if x[1] != y[1]:
            raise AssertionError(f"Dans {quoi}, le segment {numero!r} a une taille de {x[1]!r} au lieu de {y[1]}.")
        if not _proche(x[2], y[2]):
            if _proche(x[2], variantes["pourcent"][numero][2]):
                raise AssertionError(f"Dans {quoi}, la part du segment {numero!r} vaut {x[2]!r} : une proportion entre 0 "
                                     f"et 1 est attendue ({y[2]!r}), pas un pourcentage.")
            raise AssertionError(f"Dans {quoi}, la part du segment {numero!r} vaut {x[2]!r} au lieu de {y[2]!r} (sa taille "
                                 "divisée par le nombre TOTAL de clients).")
        if not _proche(x[3], y[3]):
            raise AssertionError(f"Dans {quoi}, le taux de churn du segment {numero!r} vaut {x[3]!r} au lieu de "
                                 f"{y[3]!r} (la moyenne de `churn` de SES clients).")
        if not isinstance(x[4], dict) or set(x[4]) != set(y[4]):
            raise AssertionError(f"Dans {quoi}, les moyennes du segment {numero!r} valent {_court(x[4], 90)} : un dict "
                                 f"{{variable: moyenne}} avec exactement les variables {list(variables)} est attendu.")
        for v in y[4]:
            if not _proche(x[4][v], y[4][v]):
                raise AssertionError(f"Dans {quoi}, la moyenne de {v!r} dans le segment {numero!r} vaut {x[4][v]!r} au "
                                     f"lieu de {y[4][v]!r} : la moyenne des valeurs BRUTES (en mois, en euros…) de ses "
                                     "clients, jamais des valeurs standardisées.")
        dist = x[5]
        if not isinstance(dist, tuple) or len(dist) != 2 or not _nombre(dist[1]):
            raise AssertionError(f"Dans {quoi}, le distinctif du segment {numero!r} vaut {_court(dist, 60)} : un tuple "
                                 "(variable, z) est attendu.")
        for nom, message in _VARIANTES_DISTINCTIF:   # la variable ET la valeur d'une erreur classique
            autre = variantes[nom][numero][5]
            if dist[0] != y[5][0] and dist[0] == autre[0] and _proche(dist[1], autre[1]):
                raise AssertionError(f"Dans {quoi}, le segment {numero!r} est décrit par {dist!r} au lieu de "
                                     f"{(y[5][0], round(y[5][1], 4))!r} : {message}")
        if dist[0] != y[5][0]:
            zs = _z_du_segment(clients, etiquettes, variables, numero)
            if dist[0] in zs and abs(y[5][1]) <= 1e-9 and abs(zs[dist[0]]) <= 1e-9:
                if not _proche(dist[1], zs[dist[0]]):
                    raise AssertionError(f"Dans {quoi}, le z du segment {numero!r} vaut {dist[1]!r} au lieu de "
                                         f"{zs[dist[0]]!r}.")
                continue                       # tout vaut 0 (à un arrondi près) : aucune variable ne le distingue
            for nom, message in _VARIANTES_DISTINCTIF:
                if dist[0] == variantes[nom][numero][5][0]:
                    raise AssertionError(f"Dans {quoi}, le segment {numero!r} est décrit par {dist[0]!r} au lieu de "
                                         f"{y[5][0]!r} : {message}")
            raise AssertionError(f"Dans {quoi}, le segment {numero!r} est décrit par {dist[0]!r} au lieu de {y[5][0]!r} "
                                 f"(z = {y[5][1]:.4f}) : la variable de plus grand |z| ; à égalité, la première de "
                                 "`variables`.")
        if not _proche(dist[1], y[5][1]):
            if _proche(dist[1], variantes["abs"][numero][5][1]):
                raise AssertionError(f"Dans {quoi}, le z du segment {numero!r} vaut {dist[1]!r} au lieu de {y[5][1]!r} : "
                                     "z garde son SIGNE. −1,3 en facture, ce sont de PETITES factures ; +1,3, de grosses.")
            if _proche(dist[1], variantes["ddof1"][numero][5][1]):
                raise AssertionError(f"Dans {quoi}, le z du segment {numero!r} vaut {dist[1]!r} au lieu de {y[5][1]!r} : "
                                     "l'écart-type global est celui de la POPULATION (diviser par n, comme au défi 194), "
                                     "pas n − 1.")
            if _proche(dist[1], variantes["brut"][numero][5][1]):
                raise AssertionError(f"Dans {quoi}, le z du segment {numero!r} vaut {dist[1]!r} : c'est un écart brut. "
                                     f"{_VARIANTES_DISTINCTIF[1][1]}")
            raise AssertionError(f"Dans {quoi}, le z du segment {numero!r} vaut {dist[1]!r} au lieu de {y[5][1]!r} "
                                 "(moyenne du segment − moyenne globale) / écart-type global.")
    if numeros == [x[0] for x in attendu]:
        return
    for nom, message in (("reverse", "à taux de churn égal, le numéro CROISSANT. `reverse=True` renverse aussi les "
                                     "numéros : triez avec `key=lambda s: (-s.taux_churn, s.numero)`."),
                         ("croissant", "le plus fort taux de churn d'abord (ordre DÉCROISSANT) : c'est là que Camille "
                                       "doit agir en premier."),
                         ("numero", "les fiches se trient par taux de churn décroissant, puis par numéro (pas par "
                                    "numéro seulement).")):
        if numeros == [x[0] for x in _ref_decrire(clients, etiquettes, variables, nom)]:
            raise AssertionError(f"Dans {quoi}, l'ordre des segments est {numeros} au lieu de "
                                 f"{[x[0] for x in attendu]} : {message}")
    raise AssertionError(f"Dans {quoi}, l'ordre des segments est {numeros} au lieu de {[x[0] for x in attendu]} : taux "
                         "de churn décroissant, puis numéro croissant.")


def _une_description(f, clients, etiquettes, variables, quoi: str) -> None:
    copies = copy.deepcopy((clients, etiquettes))
    r = _appel(f, (clients, etiquettes, variables), quoi,
               "Une variable peut avoir un écart-type global nul (tous les clients ont la même valeur) : son z vaut 0.0, "
               "sans division par zéro.")
    _intact(copies[0], clients, quoi, "les clients")
    _intact(copies[1], etiquettes, quoi, "les étiquettes")
    _comparer_decrire(r, clients, etiquettes, variables, quoi)


def _z_du_segment(clients, etiquettes, variables, numero) -> dict:
    """{variable: z} du segment `numero`, calculés comme le demande l'énoncé."""
    n = len(clients)
    membres = [c for c, e in zip(clients, etiquettes) if e == numero]
    zs = {}
    for v in variables:
        m = sum(c[v] for c in clients) / n
        s = math.sqrt(sum((c[v] - m) ** 2 for c in clients) / n)
        zs[v] = 0.0 if s == 0 else (sum(c[v] for c in membres) / len(membres) - m) / s
    return zs


def _z_douteux(clients, etiquettes, variables) -> bool:
    """Deux |z| égaux ou presque dans un segment tiré au hasard : le choix pourrait dépendre de vos arrondis.

    (Les ex aequo des cas écrits à la main, eux, sont exacts quelle que soit la façon de calculer.)
    """
    n = len(clients)
    globales = {v: sum(c[v] for c in clients) / n for v in variables}
    ecarts = {v: math.sqrt(sum((c[v] - globales[v]) ** 2 for c in clients) / n) for v in variables}
    groupes: dict = {}
    for client, e in zip(clients, etiquettes):
        groupes.setdefault(e, []).append(client)
    for membres in groupes.values():
        zs = sorted((abs((sum(c[v] for c in membres) / len(membres) - globales[v]) / ecarts[v]) if ecarts[v] else 0.0)
                    for v in variables)
        if len(zs) > 1 and zs[-1] > 1e-9 and zs[-1] - zs[-2] <= 1e-9 * zs[-1]:
            return True
    return False


def verifier_decrire_segments(f) -> None:
    clients = clients_exemple()
    variables = ("anciennete", "facture", "appels")
    for etiquettes in ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], [0, 0, 0, 2, 1, 1, 1, 1, 2, 2, 2, 2],
                       [5, 5, 5, 5, 9, 9, 9, 9, 5, 5, 5, 5], [0] * 12, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]):
        _une_description(f, clients, etiquettes, variables, f"decrire_segments(clients_exemple(), {etiquettes})")
    for vars_ in (("appels", "facture"), ("facture",), ["anciennete", "appels"]):
        _une_description(f, clients, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], vars_,
                         f"decrire_segments(clients_exemple(), [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], {vars_!r})")
    avec_options = [dict(c, options=2) for c in clients]                 # une variable constante : z = 0
    _une_description(f, avec_options, [0] * 6 + [1] * 6, ("options", "appels"),
                     "decrire_segments(clients avec options=2 pour tous, [0]*6 + [1]*6, ('options', 'appels'))")
    _une_description(f, avec_options, [0] * 6 + [1] * 6, ("options",),
                     "decrire_segments(clients avec options=2 pour tous, [0]*6 + [1]*6, ('options',))")
    egaux = [{"client": "A", "x": 1, "y": 10, "churn": 1}, {"client": "B", "x": 3, "y": 30, "churn": 0},
             {"client": "C", "x": 3, "y": 30, "churn": 1}, {"client": "D", "x": 1, "y": 10, "churn": 0}]
    _une_description(f, egaux, [2, 2, 1, 1], ("x", "y"), f"decrire_segments({egaux}, [2, 2, 1, 1], ('x', 'y'))")
    _une_description(f, egaux, [0, 1, 1, 0], ("y", "x"), f"decrire_segments({egaux}, [0, 1, 1, 0], ('y', 'x'))")
    _une_description(f, [], [], variables, "decrire_segments([], [], VARIABLES)")
    _doit_lever(f, (clients, [0, 1], variables), "decrire_segments(clients_exemple(), [0, 1], VARIABLES)",
                "Pas autant d'étiquettes que de clients : ValueError (zip s'arrêterait au plus court sans rien dire).")
    _doit_lever(f, (clients, [0] * 12, ()), "decrire_segments(clients_exemple(), [0] * 12, ())",
                "Aucune variable : rien pour décrire les segments, ValueError.")
    alea = random.Random(199)
    essai = 0
    while essai < 150:
        clients = generer_clients(alea.randint(1, 30), graine=alea.randint(0, 10_000))
        etiquettes = [alea.randint(0, 4) for _ in clients]
        vars_ = tuple(alea.sample(variables, alea.randint(1, 3)))
        if _z_douteux(clients, etiquettes, vars_):
            continue
        _une_description(f, clients, etiquettes, vars_, f"decrire_segments(generer_clients(…), {_court(etiquettes, 70)}, "
                                                        f"{vars_!r}) (tiré au hasard n°{essai})")
        essai += 1


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"standardiser": 194, "affecter": 195, "kmeans": 196, "kmeans_plus_plus": 197, "silhouette": 198,
            "decrire_segments": 199}

SERIE = Serie(
    code="32",
    titre="Regrouper les clients : standardiser, affecter, k-means, k-means++, silhouette, décrire",
    module_eleve="defis.serie_32",
    module_corrige="defis.corrige_serie_32",
    defis=(
        Defi(194, "standardiser", "🟢", "2.1", verifier_standardiser),
        Defi(195, "affecter", "🟢", "—", verifier_affecter),
        Defi(196, "kmeans", "🟡", "—", verifier_kmeans),
        Defi(197, "kmeans_plus_plus", "🔴", "—", verifier_kmeans_plus_plus),
        Defi(198, "silhouette", "🔴", "—", verifier_silhouette),
        Defi(199, "decrire_segments", "🟡", "—", verifier_decrire_segments),
    ),
)

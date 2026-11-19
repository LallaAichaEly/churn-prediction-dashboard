"""Vérifications automatiques de la série 40 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est la plateforme d'envoi, un soir de campagne : beaucoup de données, peu de mémoire, et
aucune tolérance pour un SMS envoyé deux fois.

- il compare vos bits, vos compteurs et vos registres à ceux d'une version de référence, case par case : les
  règles sont fixées pour qu'il n'y ait qu'une bonne réponse ;
- quand votre résultat est faux, il rejoue les erreurs classiques (`math.log10`, `int()` au lieu de `ceil`, k = 0,
  i qui part de 1, bit de poids fort, `any` au lieu de `all`, la ligne répétée par `*`, le maximum au lieu du
  minimum, les zéros comptés au lieu du rang du premier 1, `bin(w)` sans ses zéros de tête, la somme au lieu du
  maximum, `% n` au lieu de l'anneau, `bisect_right`…) pour nommer la vôtre ;
- il vous donne des flux qui ne se lisent qu'une fois (`UneSeuleFois`), MESURE la mémoire (`tracemalloc`) pendant
  200 000 valeurs différentes, et fouille vos attributs : une structure qui garde les clés se voit ;
- au défi 247, il compte les lignes exécutées dans votre fichier pour 20 000 clients sur 20 000 points : un
  parcours de tous les points à chaque client se voit ;
- et il vérifie qu'une valeur refusée n'a rien changé, et que deux objets ne partagent rien.
"""

from __future__ import annotations

import bisect
import math
import random
import sys
from collections import Counter

from .esquisses import (SERVEURS, appels_exemple, clients_exemple, empreinte, generer_appels, generer_numeros,
                        generer_visiteurs, taux_theorique, visites_exemple)
from .flux import UneSeuleFois, pic_memoire
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
    """Le fichier où `objet` (fonction, classe ou méthode) est écrit : c'est là qu'on compte les lignes."""
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


_NUMEROS = {"dimensionner": 242, "FiltreBloom": 243, "frequents": 244, "CompteurMin": 245, "HyperLogLog": 246,
            "Anneau": 247}


def _ou_leve(exc: BaseException) -> str:
    """Le nom qualifié de la fonction qui a levé `exc` (« FiltreBloom.union »)."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        return ""
    code = tb.tb_frame.f_code
    nom = getattr(code, "co_qualname", code.co_name)
    if nom == code.co_name and "self" in tb.tb_frame.f_locals:  # Python 3.10 : pas de co_qualname
        nom = f"{type(tb.tb_frame.f_locals['self']).__name__}.{code.co_name}"
    return nom


def _pas_encore(exc: NotImplementedError, a_faire: bool) -> None:
    """Un NotImplementedError dans __init__ (ou dans la fonction elle-même) : le défi est à faire. Ailleurs (une
    méthode oubliée d'une classe commencée) : on dit laquelle."""
    if a_faire:
        raise exc
    raise AssertionError(f"`{_ou_leve(exc) or '?'}` lève encore NotImplementedError : terminez-la.") from None


_CONSEIL_LENT = "Votre code a été arrêté : il exécute beaucoup trop de lignes (une boucle dans une boucle ?)."


def _conseil_exception(exc: BaseException) -> str:
    texte = str(exc)
    if isinstance(exc, RuntimeError) and "déjà été lu" in texte:
        return ("Vous lisez le flux DEUX fois. Un générateur relu ne lève pas d'erreur : il est VIDE, et votre deuxième "
                "boucle ne verrait rien, en silence. Une seule boucle `for x in flux:` suffit.")
    if isinstance(exc, TypeError) and "UneSeuleFois" in texte:
        return ("Un flux n'a pas de longueur et ne s'indexe pas : on ne sait pas combien de valeurs vont arriver. Une "
                "seule boucle : `for x in flux:`.")
    if isinstance(exc, RuntimeError) and "changed size during iteration" in texte:
        return ("On ne supprime pas de clé dans un dict PENDANT qu'on le parcourt : parcourez une copie de ses clés "
                "(`for cle in list(compteurs):`).")
    if isinstance(exc, IndexError):
        return ("Un indice dépasse : une clé située APRÈS le dernier point de l'anneau revient au PREMIER point (on fait "
                "le tour du cercle) : `i % len(points)`." if "list" in texte else "")
    return ""


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = None, conseil_lent: str = "",
           a_faire: bool = False, fichier: str | None = None):
    """f(*args), en transformant toute erreur en message lisible."""
    try:
        return _trace(fichier or (_fichier(f) if lignes_max else None), lambda: f(*args), lignes_max)
    except NotImplementedError as exc:
        _pas_encore(exc, a_faire)
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError. {conseil}".rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {_conseil_exception(exc) or conseil}".rstrip()) from None


def _creer(classe, args: tuple, quoi: str):
    """classe(*args) : un NotImplementedError dans __init__ veut dire « défi à faire »."""
    return _appel(classe, args, quoi, a_faire=True)


def _lire(objet, attribut: str, quoi: str):
    if not hasattr(objet, attribut):
        raise AssertionError(f"{quoi} n'a pas d'attribut `{attribut}` : créez `self.{attribut}` dans __init__ (relisez "
                             "« Ce qu'il faut écrire »).")
    return getattr(objet, attribut)


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu: type = ValueError) -> None:
    """Exige que f(*args) lève `attendu`."""
    try:
        resultat = f(*args)
    except NotImplementedError as exc:
        _pas_encore(exc, a_faire=False)
        raise
    except attendu:
        return
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _premier_bug(obtenus, rejouer, bugs: dict) -> str:
    """Le conseil du premier bug dont le rejeu donne EXACTEMENT vos réponses."""
    for bug, conseil in bugs.items():
        try:
            faux = rejouer(bug)
        except Exception:
            continue
        if faux == obtenus:
            return conseil
    return ""


def _entier(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _textes_gardes(objet, cles: set[str]) -> str:
    """Le nom d'un attribut de `objet` qui contient (directement ou dans un conteneur) l'une des `cles`, ou ''."""
    for nom, valeur in getattr(objet, "__dict__", {}).items():
        a_voir = [valeur]
        for _ in range(2):  # deux niveaux : un dict de listes, une liste de tuples…
            suivants = []
            for v in a_voir:
                if isinstance(v, str):
                    if v in cles:
                        return nom
                elif isinstance(v, dict):
                    suivants += list(v.keys()) + list(v.values())
                elif isinstance(v, (list, tuple, set, frozenset)):
                    suivants += list(v)
            a_voir = suivants
        if any(isinstance(v, str) and v in cles for v in a_voir):
            return nom
    return ""


def _pas_de_cles(objet, cles: list[str], quoi: str, structure: str) -> None:
    nom = _textes_gardes(objet, {c for c in cles if len(c) > 1})   # "" ou "a" pourraient être autre chose
    if nom:
        raise AssertionError(f"{quoi} garde les clés ajoutées (dans `self.{nom}`). Tout l'intérêt de {structure} est de "
                             "NE PAS les garder : 40 millions de numéros dans un `set` ou une liste, c'est le serveur "
                             "tombé le 12 novembre. Ne gardez que ce que l'énoncé décrit.")


# ============================================================================ défi 242 : dimensionner

_LN2 = math.log(2)


def _bits_possibles(n: int, taux: float) -> set[int]:
    """Les écritures justes de la formule (l'ordre des opérations change parfois le dernier chiffre du float)."""
    valeurs = (-n * math.log(taux) / math.log(2) ** 2, n * math.log(1 / taux) / math.log(2) ** 2,
               -n * math.log(taux) / (math.log(2) * math.log(2)), -(n * math.log(taux)) / math.log(2) ** 2,
               n * -math.log(taux) / math.log(2) ** 2, -n * math.log(taux) / 0.4804530139182014,
               -n / math.log(2) ** 2 * math.log(taux), math.log(taux) * -n / math.pow(math.log(2), 2))
    return {math.ceil(v) for v in valeurs}


def _k_possibles(bits: int, n: int) -> set[int]:
    return {max(1, round(v)) for v in (bits / n * _LN2, _LN2 * bits / n, bits * _LN2 / n, 0.6931471805599453 * bits / n)}


def _ref_dimension(n: int, taux: float, bug: str = "") -> tuple:
    """(bits, k, taux_reel arrondi à 12 chiffres) ; `bug` rejoue une erreur classique."""
    if bug == "log10":
        brut = -n * math.log10(taux) / _LN2 ** 2
    elif bug == "log10_partout":
        brut = -n * math.log10(taux) / math.log10(2) ** 2
    elif bug == "ln2":
        brut = -n * math.log(taux) / _LN2
    else:
        brut = -n * math.log(taux) / _LN2 ** 2
    bits = int(brut) if bug == "int" else round(brut) if bug == "round_bits" else math.ceil(brut)
    brut_k = n / bits * _LN2 if bug == "k_inverse" else bits / n * _LN2
    if bug == "k_zero":
        k = round(brut_k)
    elif bug == "k_int":
        k = max(1, int(brut_k))
    elif bug == "k_ceil":
        k = max(1, math.ceil(brut_k))
    else:
        k = max(1, round(brut_k))
    if bug == "taux_demande":
        reel = taux
    elif bug == "taux_sans_k":
        reel = 1 - math.exp(-k * n / bits)
    elif bug == "taux_exposant":
        reel = (1 - math.exp(-n / bits)) ** k
    elif bug == "taux_exact":
        reel = (1 - (1 - 1 / bits) ** (k * n)) ** k
    else:
        reel = taux_theorique(bits, k, n)
    return bits, k, reel


_BUGS_DIMENSION = {
    "log10": "Vous utilisez `math.log10` : la formule demande le logarithme NATUREL (`math.log`). Avec log10, le filtre "
             "est 2,3 fois trop petit, et le taux de faux positifs explose.",
    "log10_partout": "Vous utilisez `math.log10` partout : le résultat est 2,3 fois trop GRAND (de la mémoire gâchée). "
                     "La formule s'écrit avec le logarithme NATUREL : `math.log(taux)` et `math.log(2) ** 2`.",
    "ln2": "Vous divisez par ln 2 au lieu de (ln 2)² : `math.log(2) ** 2`.",
    "int": "Vous arrondissez le nombre de bits VERS LE BAS (`int()` coupe la partie décimale) : il manque parfois un "
           "bit, et le taux promis n'est plus tenu. Arrondissez vers le HAUT : `math.ceil`.",
    "round_bits": "Vous arrondissez le nombre de bits au plus proche (`round`) : arrondi vers le bas, il manque un bit. "
                  "Arrondissez vers le HAUT : `math.ceil`.",
    "k_zero": "Votre k vaut 0 quand le taux demandé est grand (ex. 0,9) : un filtre sans aucune position répondrait "
              "« présent » à TOUT. Gardez au moins une position : `max(1, round(...))`.",
    "k_int": "Vous tronquez k (`int()`) au lieu de l'arrondir au plus proche (`round`) : 6,64 doit donner 7.",
    "k_ceil": "Vous arrondissez k vers le haut (`math.ceil`) : la règle est l'entier le plus PROCHE (`round`).",
    "k_inverse": "Votre k est calculé avec n / bits au lieu de bits / n : k = (bits / n) · ln 2.",
    "taux_demande": "Vous renvoyez le taux DEMANDÉ : taux_reel est celui qu'on obtient vraiment avec CES bits et CE k "
                    "(arrondir k le déplace un peu) : `taux_theorique(bits, k, n)`.",
    "taux_sans_k": "Votre taux_reel oublie la puissance k : un faux positif demande que les k positions soient TOUTES à "
                   "1, soit (1 - e^(-k·n/bits)) ** k.",
    "taux_exposant": "Votre taux_reel oublie k dans l'exponentielle : chaque clé allume k bits, la part de bits à 0 est "
                     "e^(-k·n/bits).",
    "taux_exact": "Vous utilisez (1 - 1/bits) ** (k·n) : c'est la forme exacte, mais la règle (et `taux_theorique`) "
                  "utilise l'approximation e^(-k·n/bits), presque identique quand le filtre est grand.",
}

_CONSEIL_ZERO_BIT = ("Votre nombre de bits vaut-il 0 ? Pour n = 1 et taux = 0.99, -n·ln(taux) / (ln 2)² vaut 0,02 : "
                     "`int()` arrondit VERS LE BAS et `round` au plus proche, les deux donnent 0 bit (et k divise par "
                     "zéro). Arrondissez vers le HAUT, avec `math.ceil` : au moins 1 bit.")

_CAS_242 = [(1000, 0.01), (40_000_000, 0.001), (12_000, 0.01), (10, 0.9), (1, 0.5), (1, 0.99), (5, 0.6), (100, 0.2),
            (7, 1e-9), (123_456, 0.05), (2, 0.3), (1_000_000, 1e-6), (3, 0.75), (50, 0.1), (999, 0.0001), (8, 0.7),
            (20, 0.45), (1, 0.01), (30, 0.95), (4096, 0.02)]


def _batterie_242() -> list[tuple[int, float]]:
    cas = list(_CAS_242)
    alea = random.Random(242)
    for _ in range(60):
        cas.append((alea.choice((1, 2, 3, 10, 77, 500, 10_000, 3_000_000)),
                    alea.choice((alea.uniform(0.0001, 0.2), alea.uniform(0.2, 0.99), 10 ** -alea.randrange(1, 12)))))
    return cas


def verifier_dimensionner(f) -> None:
    cas = _batterie_242()
    obtenus = []
    for n, taux in cas:
        quoi = f"dimensionner({n}, {taux!r})"
        r = _appel(f, (n, taux), quoi, _CONSEIL_ZERO_BIT, a_faire=True)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, tuple) or len(r) != 3:
            raise AssertionError(f"{quoi} renvoie {_court(r)} : attendu un Dimension(bits, k, taux_reel) (de "
                                 "defis/esquisses.py), un tuple de trois valeurs.")
        bits, k, reel = r
        if not _entier(bits) or not _entier(k):
            raise AssertionError(f"{quoi} renvoie bits = {bits!r} et k = {k!r} : ce sont des ENTIERS (int). "
                                 "`math.ceil` et `round` (sans second argument) renvoient des int.")
        if not isinstance(reel, (int, float)) or isinstance(reel, bool) or math.isnan(reel):
            raise AssertionError(f"{quoi} renvoie taux_reel = {reel!r} : attendu un float.")
        obtenus.append((bits, k, reel))
    for (n, taux), (bits, k, reel) in zip(cas, obtenus):
        quoi = f"dimensionner({n}, {taux!r})"
        if bits not in _bits_possibles(n, taux) or k not in _k_possibles(bits, n) or \
                not math.isclose(reel, taux_theorique(bits, k, n), rel_tol=1e-9, abs_tol=1e-300):
            conseil = ""
            for bug, texte in _BUGS_DIMENSION.items():
                try:
                    rejoues = [_ref_dimension(a, b, bug) for a, b in cas]
                except Exception:
                    continue
                if all(o[:2] == r[:2] and math.isclose(o[2], r[2], rel_tol=1e-9, abs_tol=1e-300)
                       for o, r in zip(obtenus, rejoues)):
                    conseil = texte
                    break
            attendu = _ref_dimension(n, taux)
            raise AssertionError(f"{quoi} renvoie (bits={bits}, k={k}, taux_reel={reel:.6g}) au lieu de (bits="
                                 f"{attendu[0]}, k={attendu[1]}, taux_reel={attendu[2]:.6g}). {conseil}".rstrip())

    for args, attendu, conseil in (
            ((1000.0, 0.01), TypeError, "n est un nombre de clés : un int, pas un float."),
            (("1000", 0.01), TypeError, "n est un int, pas un texte."),
            ((True, 0.01), TypeError, "True n'est pas un nombre de clés (c'est un bool, même si `True == 1`)."),
            ((None, 0.01), TypeError, "n est un int."),
            ((1000, "0.01"), TypeError, "taux est un nombre, pas un texte."),
            ((1000, None), TypeError, "taux est un nombre."),
            ((1000, True), TypeError, "taux est un nombre, pas un bool."),
            ((0, 0.01), ValueError, "un filtre pour 0 clé n'a pas de sens (et diviserait par 0 pour k)."),
            ((-5, 0.01), ValueError, "n vaut au moins 1."),
            ((1000, 0), ValueError, "un taux de 0 demanderait une infinité de bits (log(0))."),
            ((1000, 0.0), ValueError, "un taux de 0 demanderait une infinité de bits."),
            ((1000, 1), ValueError, "un taux de 1 (100 % d'erreurs) ne demande aucun bit : refusé."),
            ((1000, 1.5), ValueError, "un taux est une proportion, strictement entre 0 et 1."),
            ((1000, -0.1), ValueError, "un taux négatif n'a pas de sens (et log(-0.1) lève une erreur peu claire)."),
            ((1000, math.nan), ValueError, "NaN n'est pas un taux. `0 < taux < 1` est FAUX pour NaN : ce test le "
                                           "refuse d'office, alors que `taux <= 0 or taux >= 1` le laisse passer."),
            ((1000, math.inf), ValueError, "l'infini n'est pas un taux.")):
        _doit_lever(f, args, f"dimensionner{args!r}", conseil, attendu)


# ============================================================================ défi 243 : FiltreBloom


def _ref_positions(cle: str, bits: int, k: int, bug: str = "") -> list[int]:
    if bug == "k_graines":
        return [empreinte(cle, i) % bits for i in range(k)]
    h1, h2 = empreinte(cle, 0), empreinte(cle, 1)
    if bug == "graines_1_2":
        h1, h2 = empreinte(cle, 1), empreinte(cle, 2)
    elif bug == "meme_graine":
        h2 = h1
    elif bug == "inverse":
        h1, h2 = h2, h1
    debut = 1 if bug == "i_un" else 0
    modulo = (bits + 7) // 8 * 8 if bug == "octets" else bits
    positions = [(h1 + i * h2) % modulo for i in range(debut, debut + k)]
    if bug == "set":
        positions = sorted(set(positions))
    elif bug == "trie":
        positions = sorted(positions)
    return positions


_BUGS_POSITIONS = {
    "i_un": "Vos positions commencent à i = 1 : la position numéro i vaut (h1 + i · h2) % bits pour i = 0, 1, …, "
            "k - 1 (la première est donc h1 % bits).",
    "meme_graine": "Vous utilisez la même empreinte pour h1 et h2 : h2 = `empreinte(cle, 1)` (graine 1), h1 = "
                   "`empreinte(cle, 0)`.",
    "inverse": "h1 et h2 sont inversés : h1 = `empreinte(cle, 0)`, h2 = `empreinte(cle, 1)`.",
    "graines_1_2": "Les graines sont 0 pour h1 et 1 pour h2 (pas 1 et 2).",
    "k_graines": "Vous calculez k empreintes indépendantes (`empreinte(cle, i)`) : c'est un vrai filtre de Bloom, mais "
                 "pas celui demandé. La règle est le DOUBLE hachage : deux empreintes h1 et h2, puis (h1 + i · h2) % "
                 "bits. Il coûte 2 empreintes au lieu de k.",
    "set": "Vous supprimez les positions en double (un `set`) : `positions` renvoie les k positions dans l'ordre "
           "i = 0, …, k - 1, doublons compris (une LISTE).",
    "trie": "Vous triez les positions : `positions` les renvoie dans l'ordre i = 0, 1, …, k - 1.",
    "octets": "Vous prenez le modulo par 8 × (nombre d'octets) au lieu de `bits` : les positions doivent rester entre "
              "0 et bits - 1 (les bits de bourrage du dernier octet ne font pas partie du filtre).",
}


def _ref_octets(bits: int, k: int, cles, bug: str = "") -> bytearray:
    octets = bytearray((bits + 7) // 8)
    for cle in cles:
        for p in _ref_positions(cle, bits, k):
            octets[p // 8] |= (0x80 >> (p % 8)) if bug == "msb" else (1 << (p % 8))
    return octets


def _ref_contient(octets: bytearray, bits: int, k: int, cle: str, bug: str = "") -> bool:
    lus = [octets[p // 8] >> (p % 8) & 1 for p in _ref_positions(cle, bits, k)]
    return any(lus) if bug == "any" else all(lus)


def _octets(filtre, quoi: str):
    octets = _lire(filtre, "octets", quoi)
    if not isinstance(octets, bytearray):
        raise AssertionError(f"{quoi}.octets est un {type(octets).__name__} : attendu un `bytearray` (8 bits par octet ; "
                             "une liste d'entiers coûte ~8 octets par case, un set de positions bien plus).")
    return octets


_CLES_243 = ["0612345678", "0699999999", "", "é", "0700000000", "C000001", "a" * 300, "0612345678 ", "06 12 34 56 78"]


def verifier_filtre_bloom(classe) -> None:
    f = _creer(classe, (20, 3), "FiltreBloom(20, 3)")
    if (_lire(f, "bits", "FiltreBloom(20, 3)"), _lire(f, "k", "FiltreBloom(20, 3)")) != (20, 3):
        raise AssertionError(f"FiltreBloom(20, 3) a bits = {f.bits!r} et k = {f.k!r} au lieu de 20 et 3.")
    octets = _octets(f, "FiltreBloom(20, 3)")
    if octets != bytearray(3):
        if len(octets) == 20:
            raise AssertionError("FiltreBloom(20, 3).octets a 20 octets : un octet PAR BIT, c'est 8 fois trop de mémoire. "
                                 "Il en faut ⌈20 / 8⌉ = 3 : `bytearray((bits + 7) // 8)`.")
        raise AssertionError(f"FiltreBloom(20, 3).octets vaut {_court(octets)} au lieu de bytearray(3) : ⌈bits / 8⌉ "
                             "octets à zéro, `bytearray((bits + 7) // 8)`.")
    _appel(f.ajouter, ("0612345678",), "FiltreBloom(20, 3).ajouter('0612345678')")
    if _octets(_creer(classe, (20, 3), "FiltreBloom(20, 3)"), "FiltreBloom(20, 3)") != bytearray(3):
        raise AssertionError("Deux filtres partagent leurs bits : après f.ajouter(...), un AUTRE filtre, tout juste créé, "
                             "n'est pas vide. Créez le bytearray dans __init__ (`self.octets = bytearray(...)`), pas au "
                             "niveau de la classe : écrit dans la classe, il est UN pour tous les filtres.")

    # ------------------------------------------------------------------ positions
    alea = random.Random(243)
    tailles = [(20, 3), (1001, 7), (115_021, 7), (8, 4), (1, 3), (64, 1), (9, 12), (1_000_003, 5), (1000, 2)]
    cles = _CLES_243 + generer_numeros(40, graine=243) + [f"V{alea.randrange(10 ** 6):06d}" for _ in range(20)]
    obtenus, attendus, appels = [], [], []
    for bits, k in tailles:
        filtre = _creer(classe, (bits, k), f"FiltreBloom({bits}, {k})")
        for cle in cles:
            quoi = f"FiltreBloom({bits}, {k}).positions({cle!r})"
            r = _appel(filtre.positions, (cle,), quoi)
            if not isinstance(r, list):
                conseil = ("un `set` perd l'ordre et les doublons. " if isinstance(r, (set, frozenset)) else "")
                raise AssertionError(f"{quoi} renvoie un {type(r).__name__} ({_court(r, 80)}) : attendu une LISTE de k "
                                     f"entiers. {conseil}".rstrip())
            obtenus.append(r)
            attendus.append(_ref_positions(cle, bits, k))
            appels.append((quoi, cle, bits, k))
    if obtenus != attendus:
        conseil = _premier_bug(obtenus, lambda bug: [_ref_positions(c, b, k, bug) for _, c, b, k in appels],
                               _BUGS_POSITIONS)
        i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        conseil = conseil or ("Avec h1 = empreinte(cle, 0) et h2 = empreinte(cle, 1) : [(h1 + i * h2) % self.bits for i in "
                              "range(self.k)]. Pas `hash()` : il change à chaque lancement de Python.")
        raise AssertionError(f"{appels[i][0]} renvoie {_court(obtenus[i], 80)} au lieu de {_court(attendus[i], 80)}. "
                             f"{conseil}")

    # ------------------------------------------------------------------ ajouter : les bons bits, dans le bon ordre
    for bits, k, ajoutees in ((20, 3, ["0612345678"]), (1001, 5, cles[:12]), (13, 2, ["a", "b", "c"]),
                              (115_021, 7, generer_numeros(300, graine=2430))):
        filtre = _creer(classe, (bits, k), f"FiltreBloom({bits}, {k})")
        for cle in ajoutees:
            _appel(filtre.ajouter, (cle,), f"FiltreBloom({bits}, {k}).ajouter({cle!r})")
        obtenu, attendu = _octets(filtre, f"FiltreBloom({bits}, {k})"), _ref_octets(bits, k, ajoutees)
        if obtenu != attendu:
            if obtenu == _ref_octets(bits, k, ajoutees, "msb"):
                conseil = ("Vous comptez les bits à partir du bit de POIDS FORT (0x80 >> j) : la position p est le bit "
                           "p % 8 en partant du bit de poids FAIBLE, `1 << (p % 8)`.")
            elif len(obtenu) != len(attendu):
                conseil = f"Le nombre d'octets a changé ({len(obtenu)} au lieu de {len(attendu)}) : ajouter ne fait qu'allumer des bits."
            else:
                conseil = "La position p est le bit p % 8 de l'octet p // 8 : `self.octets[p // 8] |= 1 << (p % 8)`."
            raise AssertionError(f"Après {len(ajoutees)} ajout(s) dans FiltreBloom({bits}, {k}), octets vaut "
                                 f"{_court(bytes(obtenu), 70)} au lieu de {_court(bytes(attendu), 70)}. {conseil}")
        # ----- in : aucun faux négatif, et exactement les réponses d'un filtre juste
        _pas_de_cles(filtre, ajoutees, f"FiltreBloom({bits}, {k})", "un filtre de Bloom")
        for cle in ajoutees:
            if _appel(lambda c: c in filtre, (cle,), f"{cle!r} in filtre") is not True:
                raise AssertionError(f"{cle!r} a été ajouté, mais `{cle!r} in filtre` ne renvoie pas True : un filtre de "
                                     "Bloom n'a JAMAIS de faux négatif (sinon, le client recevrait deux SMS). True si les "
                                     "k bits sont tous à 1 (`all`).")
        autres = generer_numeros(400, graine=bits)
        for cle in autres:
            obtenu_in = _appel(lambda c: c in filtre, (cle,), f"{cle!r} in filtre")
            if obtenu_in != _ref_contient(attendu, bits, k, cle):
                conseil = ("Vous répondez « présent » dès qu'UN bit est à 1 (`any`) : il faut que les k bits soient TOUS à "
                           "1 (`all`). Avec `any`, presque tout le monde est « déjà contacté »."
                           if obtenu_in == _ref_contient(attendu, bits, k, cle, "any") else
                           "True si les k bits de la clé sont tous à 1 : `self.octets[p // 8] >> (p % 8) & 1`.")
                raise AssertionError(f"Dans FiltreBloom({bits}, {k}) après {len(ajoutees)} ajout(s), `{cle!r} in "
                                     f"filtre` renvoie {obtenu_in!r} au lieu de {not obtenu_in!r}. {conseil}")

    # ------------------------------------------------------------------ remplissage
    for bits, k, ajoutees in ((20, 3, ["0612345678"]), (1001, 5, cles[:12]), (13, 2, ["a", "b", "c"]), (7, 1, [])):
        filtre = _creer(classe, (bits, k), f"FiltreBloom({bits}, {k})")
        for cle in ajoutees:
            _appel(filtre.ajouter, (cle,), f"ajouter({cle!r})")
        r = _appel(filtre.remplissage, (), f"FiltreBloom({bits}, {k}).remplissage()")
        uns = sum(bin(o).count("1") for o in _ref_octets(bits, k, ajoutees))
        if not isinstance(r, (int, float)) or isinstance(r, bool) or not math.isclose(r, uns / bits, rel_tol=1e-12):
            conseil = ""
            octets = (bits + 7) // 8
            if isinstance(r, (int, float)) and math.isclose(r, uns / (octets * 8), rel_tol=1e-12) and bits % 8:
                conseil = (f"Vous divisez par 8 × {octets} octets = {octets * 8} au lieu de bits = {bits} : les bits de "
                           "bourrage du dernier octet ne font pas partie du filtre.")
            elif isinstance(r, (int, float)) and math.isclose(r, uns, rel_tol=1e-12):
                conseil = "Vous renvoyez le NOMBRE de bits à 1 : la règle demande la PART (divisée par bits)."
            raise AssertionError(f"Après {len(ajoutees)} ajout(s), FiltreBloom({bits}, {k}).remplissage() renvoie "
                                 f"{_court(r)} au lieu de {uns}/{bits} = {uns / bits:.6g}. {conseil or 'La part des bits à 1 : (nombre de bits à 1) / bits ; `octet.bit_count()`.'}")

    # ------------------------------------------------------------------ union
    a, b = _creer(classe, (1001, 4), "FiltreBloom(1001, 4)"), _creer(classe, (1001, 4), "FiltreBloom(1001, 4)")
    cles_a, cles_b = generer_numeros(60, graine=1), generer_numeros(60, graine=2)
    for cle in cles_a:
        _appel(a.ajouter, (cle,), f"a.ajouter({cle!r})")
    if _octets(b, "b") != bytearray(126):
        raise AssertionError("Deux filtres partagent leurs bits : après a.ajouter(...), un AUTRE filtre b n'est plus vide. "
                             "Créez le bytearray dans __init__ (`self.octets = bytearray(...)`), pas au niveau de la "
                             "classe : écrit dans la classe, il est UN pour tous les filtres.")
    for cle in cles_b:
        _appel(b.ajouter, (cle,), f"b.ajouter({cle!r})")
    avant_a, avant_b = bytes(a.octets), bytes(b.octets)
    u = _appel(a.union, (b,), "a.union(b)")
    if not isinstance(u, classe):
        raise AssertionError(f"a.union(b) renvoie {_court(u, 80)} : attendu un NOUVEAU FiltreBloom.")
    if bytes(a.octets) != avant_a or bytes(b.octets) != avant_b or u is a or u is b:
        raise AssertionError("a.union(b) modifie a ou b (ou renvoie l'un des deux) : il faut un NOUVEAU filtre. Le "
                             "filtre de chaque serveur doit rester tel quel.")
    attendu = _ref_octets(1001, 4, cles_a + cles_b)
    if (getattr(u, "bits", None), getattr(u, "k", None)) != (1001, 4) or bytes(_octets(u, "a.union(b)")) != bytes(attendu):
        conseil = ("Vous gardez seulement les bits allumés DANS LES DEUX (`&`) : l'union garde les bits allumés dans l'UN "
                   "OU l'autre (`|`)." if bytes(u.octets) == bytes(x & y for x, y in zip(avant_a, avant_b)) else
                   "Chaque octet du résultat est le OU des deux : `bytearray(x | y for x, y in zip(...))`.")
        raise AssertionError(f"a.union(b) n'a pas les bons bits. {conseil}")
    for cle in cles_a + cles_b:
        if cle not in u:
            raise AssertionError(f"{cle!r} a été ajouté à a ou à b, mais n'est pas dans a.union(b).")
    for autre, attendu_exc, conseil in (
            ((1001, 5), ValueError, "un autre k : les mêmes bits ne veulent pas dire la même chose."),
            ((1000, 4), ValueError, "une autre taille : les positions ne correspondent pas."),):
        x = _creer(classe, autre, f"FiltreBloom{autre}")
        _doit_lever(a.union, (x,), f"FiltreBloom(1001, 4).union(FiltreBloom{autre})", conseil, attendu_exc)
    for autre in (None, set(cles_b), bytearray(126)):
        _doit_lever(a.union, (autre,), f"a.union({_court(autre, 40)})", "union attend un FiltreBloom.", TypeError)

    # ------------------------------------------------------------------ valider avant de modifier
    for args, attendu_exc in (((10.0, 3), TypeError), (("10", 3), TypeError), ((True, 3), TypeError),
                              ((10, None), TypeError), ((10, 2.0), TypeError), ((0, 3), ValueError),
                              ((-8, 3), ValueError), ((10, 0), ValueError), ((10, -1), ValueError)):
        _doit_lever(classe, args, f"FiltreBloom{args!r}", "bits et k sont des int >= 1.", attendu_exc)
    filtre = _creer(classe, (1001, 4), "FiltreBloom(1001, 4)")
    _appel(filtre.ajouter, ("0612345678",), "ajouter('0612345678')")
    avant = bytes(filtre.octets)
    for mauvaise in (612345678, None, b"0612345678", ("06", "12")):
        conseil = ("612345678 (un int) n'est pas le numéro « 0612345678 » (le 0 de tête a disparu) : refusez-le, sinon "
                   "il aurait d'autres positions et le client recevrait un second SMS.")
        _doit_lever(filtre.ajouter, (mauvaise,), f"ajouter({mauvaise!r})", conseil, TypeError)
        _doit_lever(filtre.positions, (mauvaise,), f"positions({mauvaise!r})", conseil, TypeError)
        _doit_lever(lambda c: c in filtre, (mauvaise,), f"{mauvaise!r} in filtre", conseil, TypeError)
    if bytes(filtre.octets) != avant:
        raise AssertionError("Une clé refusée (TypeError) a quand même allumé des bits : validez la clé AVANT d'en "
                             "allumer un seul.")


# ============================================================================ défi 244 : frequents


def _ref_frequents(flux, k: int, bug: str = "") -> dict:
    if bug == "tout":
        return dict(Counter(flux).most_common(k - 1))
    places = k if bug == "k" else k - 1
    compteurs: dict = {}
    for x in flux:
        if x in compteurs:
            compteurs[x] += 1
        elif len(compteurs) < places:
            compteurs[x] = 1
        elif bug == "remplace":
            plus_petit = min(compteurs, key=compteurs.get)
            compteurs[x] = compteurs.pop(plus_petit) + 1
        else:
            for cle in list(compteurs):
                compteurs[cle] -= 1
                if compteurs[cle] == 0 and bug != "zeros":
                    del compteurs[cle]
            if bug == "ajoute" and len(compteurs) < places:
                compteurs[x] = 1
    return compteurs


_BUGS_FREQUENTS = {
    "tout": "Vous comptez TOUT (un Counter de tous les éléments), puis gardez les k - 1 plus fréquents : les comptes "
            "sont exacts, mais la mémoire grandit avec le nombre d'éléments DIFFÉRENTS (15 611 numéros ce mois-ci, des "
            "millions demain). L'algorithme de Misra-Gries n'a jamais plus de k - 1 compteurs.",
    "k": "Vous gardez k compteurs : la règle en garde AU PLUS k - 1 (k = 2 : un seul compteur, le vote de Boyer-Moore).",
    "ajoute": "Après avoir retiré 1 à tous les compteurs, vous donnez une place à x : il n'entre PAS. Ce décompte est "
              "justement « payé » par x (il disparaît avec un exemplaire de chacun des k - 1 autres) : c'est ce qui "
              "donne la garantie n / k.",
    "zeros": "Vous ne supprimez pas les compteurs qui tombent à 0 : ils occupent une place (et descendent sous zéro). "
             "Supprimez-les (`del compteurs[cle]`), pour laisser entrer de nouveaux éléments.",
    "remplace": "Quand il n'y a plus de place, vous remplacez le plus petit compteur par x : c'est l'algorithme "
                "« Space-Saving », pas Misra-Gries. Ici, on retire 1 à TOUS les compteurs et x n'entre pas.",
}


def _batterie_244() -> list[tuple[list, int]]:
    cas = [(list("aabacad"), 2), (["x", "y", "x", "z", "x", "y", "x"], 3), ([], 3), (["a"], 2), (list("abcabcabc"), 3),
           (list("abcabcabc"), 4), (list("aaaaabbbbcccdde"), 3), (list("abababababc"), 2), (list("abcdefgh"), 5),
           ([1, 2, 1, 3, 1, 4, 1, 5, 2, 2, 2], 3), ([(1, 2), (1, 2), (3,), None, None, None, (1, 2)], 2),
           (list("zyxwvutsrqponm") * 2 + list("a" * 20), 4)]
    alea = random.Random(244)
    for i in range(40):
        n = alea.randrange(1, 400)
        cas.append((generer_appels(n, alea.choice((3, 20, 200)), graine=i), alea.choice((2, 3, 4, 7, 10, 50))))
    cas.append((appels_exemple(), 100))
    cas.append((appels_exemple()[:9000], 10))
    return cas


def verifier_frequents(f) -> None:
    cas = _batterie_244()
    obtenus = []
    for flux, k in cas:
        source = UneSeuleFois(flux)
        quoi = f"frequents({_court(flux, 70)}, {k})"
        r = _appel(f, (source, k), quoi, a_faire=True)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, dict):
            raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : attendu un dict {{élément: compteur}}.")
        valeurs = list(r.values())
        if any(not _entier(v) for v in valeurs):
            raise AssertionError(f"{quoi} renvoie des compteurs qui ne sont pas des int : {_court(r, 100)}.")
        if valeurs != sorted(valeurs, reverse=True):
            raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : les compteurs doivent aller du plus grand au plus "
                                 "petit. `dict(sorted(compteurs.items(), key=lambda p: p[1], reverse=True))`.")
        obtenus.append(dict(r))
    attendus = [_ref_frequents(flux, k) for flux, k in cas]
    if obtenus != attendus:
        conseil = _premier_bug(obtenus, lambda bug: [_ref_frequents(fl, k, bug) for fl, k in cas], _BUGS_FREQUENTS)
        i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        flux, k = cas[i]
        if not conseil and len(obtenus[i]) > k - 1:
            conseil = f"Il y a {len(obtenus[i])} compteurs : jamais plus de k - 1 = {k - 1}."
        raise AssertionError(f"frequents({_court(flux, 70)}, {k}) renvoie {_court(obtenus[i], 120)} au lieu de "
                             f"{_court(attendus[i], 120)}. {conseil or 'Relisez la règle pas à pas sur ce petit exemple.'}")

    # ------------------------------------------------------------------ la garantie, sur le mois d'appels
    appels = appels_exemple()
    vrais = Counter(appels)
    for k in (10, 100, 1000):
        r = _appel(f, (iter(appels), k), f"frequents(appels, {k})")
        seuil = len(appels) / k
        for numero, vrai in vrais.items():
            compteur = r.get(numero, 0)
            if not vrai - seuil <= compteur <= vrai:
                raise AssertionError(f"frequents(appels, {k}) : {numero} a appelé {vrai} fois, et son compteur vaut "
                                     f"{compteur} ; la garantie est vrai - n/k <= compteur <= vrai.")

    # ------------------------------------------------------------------ la mémoire ne dépend pas du nombre d'éléments
    def flux_distincts(n: int):
        return (f"0{6 + i % 2}{i:08d}" for i in range(n))

    _, pic = pic_memoire(lambda: _appel(f, (flux_distincts(200_000), 10), "frequents(200 000 numéros différents, 10)"))
    if pic > 1_500_000:
        raise AssertionError(f"frequents(200 000 numéros différents, 10) a eu besoin de {pic / 1e6:.1f} Mo : la mémoire "
                             "grandit avec le flux. Avec k = 10, il ne faut jamais plus de 9 compteurs, quelle que soit "
                             "la longueur du flux. Ne gardez ni les éléments vus (liste, Counter), ni une copie du flux "
                             "(`list(flux)`).")

    for k, attendu, conseil in ((1, ValueError, "k = 1 ne laisserait AUCUN compteur (k - 1 = 0)."),
                                (0, ValueError, "k vaut au moins 2."), (-3, ValueError, "k vaut au moins 2."),
                                (2.0, TypeError, "k est un int."), ("3", TypeError, "k est un int, pas un texte."),
                                (True, TypeError, "k est un int, pas un bool."), (None, TypeError, "k est un int.")):
        source = UneSeuleFois(list("abc"))
        _doit_lever(f, (source, k), f"frequents(flux, {k!r})", conseil, attendu)
        if source.lus:
            raise AssertionError(f"frequents(flux, {k!r}) lève bien l'erreur, mais APRÈS avoir lu le flux : validez k "
                                 "avant de lire quoi que ce soit (un flux lu est perdu).")


# ============================================================================ défi 245 : CompteurMin


def _ref_colonnes(cle: str, largeur: int, profondeur: int, bug: str = "") -> list[int]:
    if bug == "meme_graine":
        return [empreinte(cle, 0) % largeur] * profondeur
    if bug == "graine_plus_un":
        return [empreinte(cle, i + 1) % largeur for i in range(profondeur)]
    if bug == "graine_ligne_colonne":
        return [empreinte(f"{i}:{cle}") % largeur for i in range(profondeur)]
    return [empreinte(cle, i) % largeur for i in range(profondeur)]


_BUGS_COLONNES = {
    "meme_graine": "Vous utilisez la même graine pour toutes les lignes : chaque ligne subit alors les mêmes collisions, "
                   "et le minimum ne gagne rien. Ligne i : `empreinte(cle, i) % largeur`.",
    "graine_plus_un": "Les graines vont de 0 à profondeur - 1 : la ligne i utilise `empreinte(cle, i)` (pas i + 1).",
    "graine_ligne_colonne": "Utilisez la graine de `empreinte` : `empreinte(cle, i)`, pas `empreinte(f\"{i}:{cle}\")`.",
}


def _ref_tableau(largeur: int, profondeur: int, ajouts) -> list[list[int]]:
    tableau = [[0] * largeur for _ in range(profondeur)]
    for cle, n in ajouts:
        for i, j in enumerate(_ref_colonnes(cle, largeur, profondeur)):
            tableau[i][j] += n
    return tableau


def _ref_estimer(tableau, cle: str, bug: str = ""):
    cases = [tableau[i][j] for i, j in enumerate(_ref_colonnes(cle, len(tableau[0]), len(tableau)))]
    return {"max": max, "premiere": lambda c: c[0], "somme": sum,
            "moyenne": lambda c: sum(c) / len(c)}.get(bug, min)(cases)


_BUGS_ESTIMER = {
    "max": "Vous prenez le MAXIMUM des cases : chaque case surestime (elle cumule les collisions), le maximum garde la "
           "PIRE. Le minimum garde la moins polluée.",
    "moyenne": "Vous prenez la MOYENNE des cases : elle mélange les cases polluées aux autres. Le minimum garde la "
               "moins polluée (et ne descend jamais sous la vérité).",
    "somme": "Vous additionnez les cases des lignes : chaque ligne compte déjà TOUT le compte de la clé. Prenez le "
             "minimum.",
    "premiere": "Vous ne lisez que la première ligne : les autres lignes ne servent alors à rien. Prenez le minimum des "
                "cases de la clé dans TOUTES les lignes.",
}


def _tableau(cm, quoi: str) -> list:
    tableau = _lire(cm, "tableau", quoi)
    if not isinstance(tableau, list) or not all(isinstance(ligne, list) for ligne in tableau):
        raise AssertionError(f"{quoi}.tableau est un {type(tableau).__name__} : attendu une liste de listes (une par ligne).")
    return tableau


def _partagees(tableau) -> bool:
    return len({id(ligne) for ligne in tableau}) < len(tableau)


_CONSEIL_PARTAGE = ("Vos lignes sont UNE SEULE liste répétée : `[[0] * largeur] * profondeur` copie la RÉFÉRENCE, pas la "
                    "liste. Ajouter dans une ligne ajoute dans toutes, et chaque estimation est multipliée par la "
                    "profondeur. Écrivez `[[0] * largeur for _ in range(profondeur)]`.")


def _ref_pour_erreur(epsilon: float, delta: float, bug: str = "") -> tuple[int, int]:
    if bug == "inverse":
        return math.ceil(math.e / delta), math.ceil(math.log(1 / epsilon))
    largeur = math.ceil(1 / epsilon) if bug == "sans_e" else int(math.e / epsilon) if bug == "int" else \
        math.ceil(math.e / epsilon)
    profondeur = math.ceil(math.log10(1 / delta)) if bug == "log10" else int(math.log(1 / delta)) if bug == "int" else \
        math.ceil(-math.log2(delta)) if bug == "log2" else math.ceil(math.log(1 / delta))
    return largeur, profondeur


_BUGS_POUR_ERREUR = {
    "log10": "Vous utilisez `math.log10` pour la profondeur : ⌈ln(1 / δ)⌉ demande le logarithme NATUREL (`math.log`).",
    "log2": "Vous utilisez `math.log2` pour la profondeur : ⌈ln(1 / δ)⌉ demande le logarithme NATUREL (`math.log`).",
    "sans_e": "La largeur vaut ⌈e / ε⌉ (e = math.e ≈ 2,718), pas ⌈1 / ε⌉ : c'est e qui donne la garantie de Cormode "
              "et Muthukrishnan.",
    "int": "Vous arrondissez vers le BAS (`int()`) : la garantie demande ⌈…⌉, `math.ceil`.",
    "inverse": "Largeur et profondeur sont inversées : la largeur vient de ε (l'erreur), la profondeur de δ (la "
               "probabilité de la dépasser).",
}


def verifier_compteur_min(classe) -> None:
    cm = _creer(classe, (5, 3), "CompteurMin(5, 3)")
    for attribut, attendu in (("largeur", 5), ("profondeur", 3), ("total", 0)):
        if _lire(cm, attribut, "CompteurMin(5, 3)") != attendu:
            raise AssertionError(f"CompteurMin(5, 3).{attribut} vaut {getattr(cm, attribut)!r} au lieu de {attendu}.")
    tableau = _tableau(cm, "CompteurMin(5, 3)")
    if tableau != [[0] * 5 for _ in range(3)]:
        raise AssertionError(f"CompteurMin(5, 3).tableau vaut {_court(tableau)} au lieu de 3 lignes de 5 zéros.")
    if _partagees(tableau):
        raise AssertionError(f"CompteurMin(5, 3).tableau : {_CONSEIL_PARTAGE}")

    # ------------------------------------------------------------------ colonnes
    cles = ["0611111111", "0622222222", "", "é", "0600000009"] + generer_numeros(40, graine=245)
    dims = [(5, 3), (1, 4), (2719, 5), (7, 1), (64, 8)]
    obtenus, appels = [], []
    for largeur, profondeur in dims:
        x = _creer(classe, (largeur, profondeur), f"CompteurMin({largeur}, {profondeur})")
        for cle in cles:
            quoi = f"CompteurMin({largeur}, {profondeur}).colonnes({cle!r})"
            r = _appel(x.colonnes, (cle,), quoi)
            if not isinstance(r, list):
                raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : attendu une liste (une colonne par ligne).")
            obtenus.append(r)
            appels.append((quoi, cle, largeur, profondeur))
    attendus = [_ref_colonnes(c, la, p) for _, c, la, p in appels]
    if obtenus != attendus:
        conseil = _premier_bug(obtenus, lambda bug: [_ref_colonnes(c, la, p, bug) for _, c, la, p in appels],
                               _BUGS_COLONNES)
        i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        raise AssertionError(f"{appels[i][0]} renvoie {_court(obtenus[i])} au lieu de {_court(attendus[i])}. "
                             f"{conseil or 'Ligne i : `empreinte(cle, i) % self.largeur`.'}")

    # ------------------------------------------------------------------ ajouter, total, estimer
    alea = random.Random(2450)
    for largeur, profondeur, ajouts in (
            (5, 3, [("0611111111", 4), ("0622222222", 1)]),
            (7, 4, [(c, alea.randrange(0, 6)) for c in generer_appels(60, 15, graine=1)]),
            (50, 5, [(c, 1) for c in generer_appels(3000, 400, graine=2)]),
            (1, 2, [("a", 3), ("b", 2), ("a", 0)])):
        x = _creer(classe, (largeur, profondeur), f"CompteurMin({largeur}, {profondeur})")
        quoi = f"CompteurMin({largeur}, {profondeur})"
        for cle, n in ajouts:
            if n == 1 and alea.random() < 0.5:
                _appel(x.ajouter, (cle,), f"{quoi}.ajouter({cle!r})")
            else:
                _appel(x.ajouter, (cle, n), f"{quoi}.ajouter({cle!r}, {n})")
        tableau = _tableau(x, quoi)
        attendu = _ref_tableau(largeur, profondeur, ajouts)
        if tableau != attendu:
            conseil = _CONSEIL_PARTAGE if _partagees(tableau) else (
                "ajouter(cle, n) ajoute n (pas 1) dans chaque ligne, à la colonne de la clé pour CETTE ligne."
                if any(n > 1 for _, n in ajouts) else "Ajoutez n dans chaque ligne i, à la colonne colonnes(cle)[i].")
            raise AssertionError(f"Après {len(ajouts)} ajouts, {quoi}.tableau vaut {_court(tableau, 120)} au lieu de "
                                 f"{_court(attendu, 120)}. {conseil}")
        total = sum(n for _, n in ajouts)
        if _lire(x, "total", quoi) != total:
            raise AssertionError(f"Après {len(ajouts)} ajouts, {quoi}.total vaut {x.total!r} au lieu de {total} (la somme "
                                 "des n ajoutés).")
        vrais = Counter()
        for cle, n in ajouts:
            vrais[cle] += n
        sondees = list(vrais) + generer_numeros(30, graine=largeur)
        obtenus = [_appel(x.estimer, (c,), f"{quoi}.estimer({c!r})") for c in sondees]
        attendus = [_ref_estimer(attendu, c) for c in sondees]
        if obtenus != attendus:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_estimer(attendu, c, bug) for c in sondees], _BUGS_ESTIMER)
            i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
            raise AssertionError(f"{quoi}.estimer({sondees[i]!r}) renvoie {_court(obtenus[i])} au lieu de {attendus[i]}. "
                                 f"{conseil or 'Le MINIMUM des cases de la clé, une par ligne.'}")
        for c in vrais:
            if x.estimer(c) < vrais[c]:
                raise AssertionError(f"{quoi}.estimer({c!r}) vaut {x.estimer(c)}, MOINS que la vérité ({vrais[c]}) : "
                                     "Count-Min ne sous-estime jamais.")

    # ------------------------------------------------------------------ deux compteurs ne partagent rien
    a, b = _creer(classe, (5, 3), "CompteurMin(5, 3)"), _creer(classe, (5, 3), "CompteurMin(5, 3)")
    _appel(a.ajouter, ("0611111111", 2), "a.ajouter('0611111111', 2)")
    if _tableau(b, "b") != [[0] * 5 for _ in range(3)] or b.total != 0:
        raise AssertionError("Deux compteurs partagent leur tableau (ou leur total) : après a.ajouter(...), un AUTRE "
                             "compteur b a changé. Créez le tableau dans __init__, pas au niveau de la classe.")

    # ------------------------------------------------------------------ pour_erreur
    cas = [(0.001, 0.01), (0.01, 0.05), (0.5, 0.5), (0.1, 0.001), (0.0001, 1e-6), (0.3, 0.9), (0.002, 0.2),
           (0.05, 0.3)]
    obtenus = []
    for epsilon, delta in cas:
        quoi = f"CompteurMin.pour_erreur({epsilon}, {delta})"
        x = _appel(classe.pour_erreur, (epsilon, delta), quoi)
        if not isinstance(x, classe):
            raise AssertionError(f"{quoi} renvoie {_court(x, 80)} : attendu un CompteurMin (dans une @classmethod : "
                                 "`return cls(largeur, profondeur)`).")
        obtenus.append((x.largeur, x.profondeur))
    attendus = [_ref_pour_erreur(e, d) for e, d in cas]
    if obtenus != attendus:
        conseil = _premier_bug(obtenus, lambda bug: [_ref_pour_erreur(e, d, bug) for e, d in cas], _BUGS_POUR_ERREUR)
        i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        raise AssertionError(f"CompteurMin.pour_erreur{cas[i]!r} a largeur, profondeur = {obtenus[i]} au lieu de "
                             f"{attendus[i]}. {conseil or 'largeur = ⌈e / ε⌉, profondeur = ⌈ln(1 / δ)⌉.'}")

    # ------------------------------------------------------------------ valider avant de modifier
    for args, attendu_exc in (((5.0, 3), TypeError), (("5", 3), TypeError), ((True, 3), TypeError),
                              ((5, None), TypeError), ((0, 3), ValueError), ((5, 0), ValueError),
                              ((-1, 3), ValueError)):
        _doit_lever(classe, args, f"CompteurMin{args!r}", "largeur et profondeur sont des int >= 1.", attendu_exc)
    for args, attendu_exc in (((0, 0.01), ValueError), ((1, 0.01), ValueError), ((0.01, 0), ValueError),
                              ((0.01, 1), ValueError), ((math.nan, 0.01), ValueError), ((0.01, -0.5), ValueError),
                              (("0.01", 0.01), TypeError), ((0.01, None), TypeError), ((True, 0.01), TypeError)):
        _doit_lever(classe.pour_erreur, args, f"CompteurMin.pour_erreur{args!r}",
                    "epsilon et delta sont des nombres strictement entre 0 et 1.", attendu_exc)
    x = _creer(classe, (5, 3), "CompteurMin(5, 3)")
    _appel(x.ajouter, ("0611111111", 4), "ajouter('0611111111', 4)")
    for args, attendu_exc, conseil in (
            (("0611111111", -1), ValueError, "un compte négatif ferait passer l'estimation SOUS la vérité : la garantie "
                                             "de Count-Min ne tient que pour des ajouts >= 0."),
            (("0611111111", 1.5), TypeError, "n est un int."), (("0611111111", True), TypeError, "n est un int, pas un "
                                                                                                  "bool."),
            (("0611111111", "2"), TypeError, "n est un int."), ((611111111, 1), TypeError, "la clé est un texte."),
            ((None, 1), TypeError, "la clé est un texte.")):
        _doit_lever(x.ajouter, args, f"ajouter{args!r}", conseil, attendu_exc)
        if x.tableau != _ref_tableau(5, 3, [("0611111111", 4)]) or x.total != 4:
            raise AssertionError(f"ajouter{args!r} a bien été refusé ({attendu_exc.__name__}), mais a quand même modifié "
                                 "le tableau ou le total : validez la clé ET n avant de toucher à quoi que ce soit.")
    for mauvaise in (611111111, None):
        _doit_lever(x.estimer, (mauvaise,), f"estimer({mauvaise!r})", "la clé est un texte.", TypeError)
        _doit_lever(x.colonnes, (mauvaise,), f"colonnes({mauvaise!r})", "la clé est un texte.", TypeError)


# ============================================================================ défi 246 : HyperLogLog


def _registre_et_rang(cle: str, p: int, bug: str = "") -> tuple[int, int]:
    """(numéro du registre, rang) d'une clé ; `bug` rejoue une erreur classique."""
    x = empreinte(cle)
    if bug == "bas":
        j, w = x & ((1 << p) - 1), x >> p
    else:
        j, w = x >> (64 - p), x & ((1 << (64 - p)) - 1)
    if bug == "zeros":
        rang = (64 - p) - w.bit_length()
    elif bug == "bin":
        rang = bin(w)[2:].find("1") + 1
    elif bug == "droite":
        rang = ((w & -w).bit_length()) if w else 64 - p + 1
    elif bug == "w_64":
        rang = 64 - x.bit_length() + 1
    else:
        rang = (64 - p) - w.bit_length() + 1
    return j, rang


def _ref_registres(cles, p: int, bug: str = "") -> list[int]:
    registres = [0] * (1 << p)
    for cle in cles:
        j, rang = _registre_et_rang(cle, p, bug)
        registres[j] = rang if bug == "sans_max" else max(registres[j], rang)
    return registres


_BUGS_REGISTRES = {
    "zeros": "Vos registres valent 1 de moins que prévu : vous comptez les ZÉROS de tête, pas le RANG du premier 1. "
             "Rang = (nombre de zéros de tête) + 1 = (64 - p) - w.bit_length() + 1.",
    "bin": "Avec `bin(w)`, le premier 1 est TOUJOURS en tête : `bin` supprime les zéros de tête. Écrivez w sur 64 - p "
           "bits (`format(w, f\"0{64 - p}b\")`), ou calculez `(64 - p) - w.bit_length() + 1`.",
    "bas": "Vous prenez le registre dans les bits de DROITE de l'empreinte : la règle prend les p bits de GAUCHE "
           "(`x >> (64 - p)`), et w = les 64 - p bits restants (`x & ((1 << (64 - p)) - 1)`).",
    "droite": "Vous comptez les zéros à DROITE de w (la version de Flajolet et Martin) : la règle compte à partir de la "
              "GAUCHE de w écrit sur 64 - p bits.",
    "w_64": "Vous cherchez le premier 1 dans les 64 bits de l'empreinte : les p bits de gauche ont déjà servi à choisir "
            "le registre. Cherchez-le dans w, les 64 - p bits RESTANTS.",
    "sans_max": "Un registre garde le MAXIMUM des rangs vus (`max`) : en écrivant le dernier rang, une valeur plus "
                "petite efface une plus grande, et l'ordre d'arrivée change le résultat.",
}


def _alpha(m: int) -> float:
    return {16: 0.673, 32: 0.697, 64: 0.709}.get(m, 0.7213 / (1 + 1.079 / m))


def _ref_estimation(registres, bug: str = "") -> float:
    m = len(registres)
    alpha = 0.7213 / (1 + 1.079 / m) if bug == "alpha_fixe" else 1.0 if bug == "sans_alpha" else _alpha(m)
    if bug == "geometrique":
        brute = alpha * m * 2 ** (sum(registres) / m)
    else:
        brute = alpha * m * m / math.fsum(2.0 ** -r for r in registres)
    vides = list(registres).count(0)
    if bug == "sans_lineaire":
        return brute
    if vides and (bug == "toujours_lineaire" or brute <= 2.5 * m):
        return m * (math.log10(m / vides) if bug == "log10" else math.log(m / vides))
    return brute


_BUGS_ESTIMATION = {
    "sans_lineaire": "Avec peu de clés, il faut le « linear counting » : si E <= 2.5 · m et qu'il reste V > 0 registres "
                     "vides, la réponse est m · ln(m / V). Sans lui, 10 visiteurs sont estimés à ~0,7 · m.",
    "toujours_lineaire": "Le « linear counting » ne s'applique que si E <= 2.5 · m (et V > 0) : au-delà, la formule "
                         "brute E est plus juste.",
    "alpha_fixe": "α vaut 0.673 pour m = 16, 0.697 pour m = 32 et 0.709 pour m = 64 ; 0.7213 / (1 + 1.079 / m) "
                  "seulement au-delà.",
    "sans_alpha": "Vous oubliez α : E = α · m² / Σ 2^(-r) (sans α, l'estimation est ~40 % trop haute).",
    "geometrique": "Vous utilisez la moyenne GÉOMÉTRIQUE des 2^r (c'est LogLog, 2003) : HyperLogLog utilise la moyenne "
                   "HARMONIQUE, E = α · m² / Σ 2^(-r), bien moins sensible à un registre chanceux.",
    "log10": "Le « linear counting » utilise le logarithme NATUREL : m · math.log(m / V).",
}


def _registres(h, quoi: str) -> list[int]:
    registres = _lire(h, "registres", quoi)
    if not isinstance(registres, (bytearray, list)):
        raise AssertionError(f"{quoi}.registres est un {type(registres).__name__} : attendu un bytearray (ou une liste "
                             "d'entiers).")
    return list(registres)


def _remplir_hll(classe, p: int, cles, quoi: str):
    h = _creer(classe, (p,), f"HyperLogLog({p})")
    for cle in cles:
        _appel(h.ajouter, (cle,), f"{quoi} : ajouter({cle!r})")
    return h


def verifier_hyperloglog(classe) -> None:
    h = _creer(classe, (), "HyperLogLog()")
    if _lire(h, "precision", "HyperLogLog()") != 14 or _registres(h, "HyperLogLog()") != [0] * 16384:
        raise AssertionError("HyperLogLog() (sans argument) doit avoir precision = 14 et 2**14 = 16 384 registres à 0.")
    h4 = _creer(classe, (4,), "HyperLogLog(4)")
    if _registres(h4, "HyperLogLog(4)") != [0] * 16:
        raise AssertionError(f"HyperLogLog(4).registres vaut {_court(h4.registres)} au lieu de 16 registres à 0 (2**4).")
    _appel(h4.ajouter, ("V000001",), "HyperLogLog(4).ajouter('V000001')")
    if _registres(_creer(classe, (4,), "HyperLogLog(4)"), "HyperLogLog(4)") != [0] * 16:
        raise AssertionError("Deux HyperLogLog partagent leurs registres : après h.ajouter(...), un AUTRE objet, tout "
                             "juste créé, n'est pas vide. Créez les registres dans __init__, pas au niveau de la classe.")

    # ------------------------------------------------------------------ ajouter : les registres
    alea = random.Random(246)
    cas = [(4, ["V000001"]), (4, ["V000001", "V000002", "V000003", "V000001"]), (5, generer_visiteurs(40, graine=1)),
           (6, generer_visiteurs(200, graine=2)), (10, generer_visiteurs(3000, graine=3)),
           (16, generer_visiteurs(500, graine=4)), (4, ["", "é", "0612345678"] * 2),
           (8, [f"x{alea.randrange(10 ** 9)}" for _ in range(2000)])]
    obtenus = []
    for p, cles in cas:
        quoi = f"HyperLogLog({p}) après {len(cles)} ajouts"
        obtenus.append(_registres(_remplir_hll(classe, p, cles, quoi), quoi))
    attendus = [_ref_registres(cles, p) for p, cles in cas]
    if obtenus != attendus:
        conseil = _premier_bug(obtenus, lambda bug: [_ref_registres(c, p, bug) for p, c in cas], _BUGS_REGISTRES)
        i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        p, cles = cas[i]
        j = next(j for j, (o, a) in enumerate(zip(obtenus[i], attendus[i])) if o != a)
        raise AssertionError(f"HyperLogLog({p}) après ajouter({_court(cles, 60)}) : le registre {j} vaut "
                             f"{obtenus[i][j]} au lieu de {attendus[i][j]}. {conseil or 'Relisez la règle : j = x >> (64 - p), w = les 64 - p bits restants, rang = position du premier 1 de w (à partir de 1).'}")
    # ------------------------------------------------------------------ estimation, à registres justes
    cas_estimation = [(4, []), (4, ["V000001", "V000002", "V000003"]), (4, generer_visiteurs(30, graine=5)),
                      (4, generer_visiteurs(400, graine=6)), (5, generer_visiteurs(100, graine=7)),
                      (6, generer_visiteurs(1000, graine=8)), (10, generer_visiteurs(10, graine=9)),
                      (10, generer_visiteurs(2000, graine=10)), (10, [f"c{i}" for i in range(20_000)]),
                      (14, generer_visiteurs(5000, graine=11)), (12, [f"c{i}" for i in range(30_000)])]
    obtenus, registres = [], []
    for p, cles in cas_estimation:
        quoi = f"HyperLogLog({p}) après {len(cles)} ajouts"
        x = _remplir_hll(classe, p, cles, quoi)
        r = _appel(x.estimation, (), f"{quoi}, estimation()")
        if not isinstance(r, (int, float)) or isinstance(r, bool):
            raise AssertionError(f"{quoi}, estimation() renvoie {_court(r)} : attendu un nombre (float).")
        obtenus.append(r)
        registres.append(_registres(x, quoi))
    attendus = [_ref_estimation(r) for r in registres]
    faux = [i for i, (o, a) in enumerate(zip(obtenus, attendus)) if not math.isclose(o, a, rel_tol=1e-9, abs_tol=1e-9)]
    if faux:
        conseil = ""
        for bug, texte in _BUGS_ESTIMATION.items():
            rejoue = [_ref_estimation(r, bug) for r in registres]
            if all(math.isclose(o, a, rel_tol=1e-9, abs_tol=1e-9) for o, a in zip(obtenus, rejoue)):
                conseil = texte
                break
        i = faux[0]
        p, cles = cas_estimation[i]
        raise AssertionError(f"HyperLogLog({p}) après {len(cles)} ajouts ({len(set(cles))} différents) : estimation() "
                             f"renvoie {obtenus[i]:.6g} au lieu de {attendus[i]:.6g}. {conseil or 'E = α · m² / Σ 2^(-r) ; si E <= 2.5 · m et V > 0 registres vides : m · ln(m / V).'}")

    # un rang maximal (w = 0) : 64 - p + 1, ce qui tient dans un octet
    for p in (4, 14):
        m = 1 << p
        x = _creer(classe, (p,), f"HyperLogLog({p})")
        x.registres[3] = 64 - p + 1
        if not math.isclose(_appel(x.estimation, (), f"HyperLogLog({p}).estimation()"),
                            _ref_estimation([0, 0, 0, 64 - p + 1] + [0] * (m - 4)), rel_tol=1e-9):
            raise AssertionError(f"HyperLogLog({p}).estimation() est fausse quand un registre vaut {64 - p + 1} (le rang "
                                 "maximal, w = 0).")

    # ------------------------------------------------------------------ pas de clés gardées, mémoire constante
    cles = [f"V{i:07d}" for i in range(20_000)]
    x = _remplir_hll(classe, 10, cles, "HyperLogLog(10)")
    _pas_de_cles(x, cles, "HyperLogLog(10)", "HyperLogLog")
    if len(_registres(x, "HyperLogLog(10)")) != 1024:
        raise AssertionError("Après 20 000 ajouts, HyperLogLog(10) n'a plus 1 024 registres : leur nombre ne change "
                             "jamais.")

    # ------------------------------------------------------------------ fusionner
    lundi, mardi = generer_visiteurs(3000, graine=21), generer_visiteurs(3000, graine=22) + generer_visiteurs(500,
                                                                                                                graine=21)
    a, b = _remplir_hll(classe, 10, lundi, "a"), _remplir_hll(classe, 10, mardi, "b")
    avant_a, avant_b = _registres(a, "a"), _registres(b, "b")
    u = _appel(a.fusionner, (b,), "a.fusionner(b)")
    if not isinstance(u, classe):
        raise AssertionError(f"a.fusionner(b) renvoie {_court(u, 80)} : attendu un NOUVEL HyperLogLog.")
    if _registres(a, "a") != avant_a or _registres(b, "b") != avant_b or u is a or u is b:
        raise AssertionError("a.fusionner(b) modifie a ou b (ou renvoie l'un des deux) : il faut un NOUVEL objet. Le "
                             "résumé de lundi doit rester celui de lundi.")
    obtenu, attendu = _registres(u, "a.fusionner(b)"), _ref_registres(lundi + mardi, 10)
    if obtenu != attendu or _lire(u, "precision", "a.fusionner(b)") != 10:
        if obtenu == [x + y for x, y in zip(avant_a, avant_b)]:
            conseil = ("Vous ADDITIONNEZ les registres : un visiteur venu lundi ET mardi compterait deux fois (l'erreur "
                       "du tableau de bord). Le MAXIMUM, registre par registre, donne exactement le résumé d'un seul "
                       "HyperLogLog qui aurait vu les deux jours.")
        elif obtenu == [min(x, y) for x, y in zip(avant_a, avant_b)]:
            conseil = "Vous gardez le MINIMUM : c'est le MAXIMUM, registre par registre."
        else:
            conseil = "Le MAXIMUM, registre par registre : `bytearray(max(x, y) for x, y in zip(...))`."
        raise AssertionError(f"a.fusionner(b) n'a pas les bons registres. {conseil}")
    for autre, attendu_exc in ((_creer(classe, (11,), "HyperLogLog(11)"), ValueError), (None, TypeError),
                               (bytearray(1024), TypeError), (set(lundi), TypeError)):
        _doit_lever(a.fusionner, (autre,), f"HyperLogLog(10).fusionner({_court(autre, 40)})",
                    "même precision exigée ; un autre objet qu'un HyperLogLog → TypeError.", attendu_exc)

    # ------------------------------------------------------------------ deux objets ne partagent rien, erreurs
    a = _creer(classe, (4,), "HyperLogLog(4)")
    _appel(a.ajouter, ("V000001",), "a.ajouter('V000001')")
    for args, attendu_exc in (((3,), ValueError), ((17,), ValueError), ((0,), ValueError), ((14.0,), TypeError),
                              (("14",), TypeError), ((True,), TypeError), ((None,), TypeError)):
        _doit_lever(classe, args, f"HyperLogLog{args!r}".replace(",)", ")"), "precision est un int de 4 à 16.",
                    attendu_exc)
    for mauvaise in (12, None, b"V000001"):
        _doit_lever(a.ajouter, (mauvaise,), f"ajouter({mauvaise!r})", "la clé est un texte.", TypeError)
    if _registres(a, "a") != _ref_registres(["V000001"], 4):
        raise AssertionError("Une clé refusée (TypeError) a quand même modifié un registre : validez-la d'abord.")

    # ------------------------------------------------------------------ l'histoire : la semaine à partir des jours
    visites = visites_exemple()
    semaine = None
    for jour, visiteurs in visites.items():
        h = _remplir_hll(classe, 14, visiteurs, jour)
        semaine = h if semaine is None else semaine.fusionner(h)
    vrai = len(set().union(*visites.values()))
    estime = semaine.estimation()
    if not abs(estime - vrai) / vrai < 0.02:
        raise AssertionError(f"La semaine de visites_exemple() est estimée à {estime:.0f} visiteurs au lieu d'environ "
                             f"{vrai}.")


# ============================================================================ défi 247 : Anneau


class _RefAnneau:
    """L'anneau de référence ; `bug` rejoue une erreur classique."""

    def __init__(self, serveurs=(), repliques: int = 100, bug: str = "") -> None:
        self.repliques, self.bug, self.serveurs, self.points, self.positions = repliques, bug, [], [], []
        for s in serveurs:
            self.ajouter(s)

    def _position(self, serveur: str, i: int) -> int:
        if self.bug == "sans_diese":
            return empreinte(f"{serveur}{i}")
        return empreinte(f"{serveur}#{i}")

    def ajouter(self, serveur: str) -> None:
        n = 1 if self.bug == "sans_repliques" else self.repliques
        self.points = sorted(self.points + [(self._position(serveur, i), serveur) for i in range(n)])
        self.positions = [p for p, _ in self.points]
        self.serveurs.append(serveur)

    def retirer(self, serveur: str) -> None:
        self.points = [p for p in self.points if p[1] != serveur]
        self.positions = [p for p, _ in self.points]
        self.serveurs.remove(serveur)

    def serveur_pour(self, cle: str) -> str:
        h = empreinte(cle)
        if self.bug == "modulo":
            return self.serveurs[h % len(self.serveurs)]
        positions = self.positions
        if self.bug == "droite":
            i = bisect.bisect_right(positions, h)
        elif self.bug == "precedent":
            i = bisect.bisect_left(positions, h) - 1
        else:
            i = bisect.bisect_left(positions, h)
        return self.points[i % len(self.points)][1]


_BUGS_ANNEAU = {
    "modulo": "Vous répartissez avec `empreinte(cle) % len(serveurs)` : c'est l'erreur du passage à 5 serveurs (80 % des "
              "clients déplacés, tous les caches vidés). Placez les points des serveurs sur l'anneau, et donnez à chaque "
              "clé le premier point à une position >= la sienne.",
    "droite": "Une clé posée PILE sur un point appartient à ce point : `bisect_left` (la première position >= la clé), "
              "pas `bisect_right` (la première position > la clé).",
    "precedent": "Vous prenez le point PRÉCÉDENT : la clé va au premier point qu'on rencontre en tournant à partir d'elle "
                 "(position >= la sienne).",
    "sans_repliques": "Chaque serveur a UN seul point sur votre anneau : il en faut `repliques`, aux positions "
                      "`empreinte(f\"{serveur}#{i}\")` pour i de 0 à repliques - 1. Avec un seul point, un serveur peut "
                      "hériter de la moitié du cercle par malchance.",
    "sans_diese": "Le point i du serveur s est à `empreinte(f\"{s}#{i}\")`, avec le « # » (sinon « score-1 » point 10 et "
                  "« score-11 » point 0 seraient au même endroit).",
}


def _cles_sur_le_bord(points: list[tuple[int, str]], combien: int) -> list[str]:
    """Des clés dont la position dépasse le DERNIER point (elles font le tour) ou précède le premier."""
    dernier, premier = points[-1][0], points[0][0]
    trouvees, i = [], 0
    while len(trouvees) < combien and i < 2_000_000:
        cle = f"tour-{i}"
        if empreinte(cle) > dernier or empreinte(cle) < premier:
            trouvees.append(cle)
        i += 1
    return trouvees


def _comparer_anneau(anneau, ref: _RefAnneau, cles: list[str], contexte: str, bugs_args: tuple) -> None:
    obtenus = [_appel(anneau.serveur_pour, (c,), f"{contexte} : serveur_pour({c!r})") for c in cles]
    attendus = [ref.serveur_pour(c) for c in cles]
    if obtenus != attendus:
        serveurs, repliques, operations = bugs_args

        def rejouer(bug):
            r = _RefAnneau(serveurs, repliques, bug)
            for operation, s in operations:
                getattr(r, operation)(s)
            return [r.serveur_pour(c) for c in cles]

        conseil = _premier_bug(obtenus, rejouer, _BUGS_ANNEAU)
        i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        raise AssertionError(f"{contexte} : serveur_pour({cles[i]!r}) renvoie {obtenus[i]!r} au lieu de {attendus[i]!r}. "
                             f"{conseil or 'Le serveur du premier point dont la position est >= empreinte(cle) ; après le dernier point, le premier.'}")


def verifier_anneau(classe) -> None:
    anneau = _creer(classe, (list(SERVEURS),), f"Anneau({list(SERVEURS)})")
    if _appel(len, (anneau,), "len(anneau)") != 4:
        raise AssertionError(f"len(Anneau({list(SERVEURS)})) vaut {len(anneau)!r} au lieu de 4 (le nombre de serveurs).")
    clients = clients_exemple(3000)
    ref = _RefAnneau(SERVEURS)
    _comparer_anneau(anneau, ref, clients, "Anneau(SERVEURS)", (SERVEURS, 100, ()))

    # une clé pile sur un point, et des clés qui font le tour
    sur_les_points = [f"{s}#{i}" for s in SERVEURS for i in (0, 7, 42, 99)]
    _comparer_anneau(anneau, ref, sur_les_points, "Anneau(SERVEURS), clés posées pile sur un point",
                     (SERVEURS, 100, ()))
    bord = _cles_sur_le_bord(ref.points, 12)
    _comparer_anneau(anneau, ref, bord, "Anneau(SERVEURS), clés après le dernier point (ou avant le premier)",
                     (SERVEURS, 100, ()))

    # ------------------------------------------------------------------ ajouter : seule la part du nouveau bouge
    avant = {c: anneau.serveur_pour(c) for c in clients}
    _appel(anneau.ajouter, ("score-5",), "anneau.ajouter('score-5')")
    if len(anneau) != 5:
        raise AssertionError(f"Après ajouter('score-5'), len(anneau) vaut {len(anneau)} au lieu de 5.")
    operations = [("ajouter", "score-5")]
    ref.ajouter("score-5")
    _comparer_anneau(anneau, ref, clients + sur_les_points + ["score-5#3"], "Après ajouter('score-5')",
                     (SERVEURS, 100, tuple(operations)))
    deplaces = [c for c in clients if anneau.serveur_pour(c) != avant[c]]
    if any(anneau.serveur_pour(c) != "score-5" for c in deplaces):
        raise AssertionError("Après ajouter('score-5'), des clients sont passés d'un ANCIEN serveur à un autre ancien : "
                             "seuls ceux qui vont vers le nouveau doivent bouger.")

    # ------------------------------------------------------------------ retirer : seuls ses clients bougent
    avant = {c: anneau.serveur_pour(c) for c in clients}
    _appel(anneau.retirer, ("score-2",), "anneau.retirer('score-2')")
    ref.retirer("score-2")
    operations.append(("retirer", "score-2"))
    if len(anneau) != 4:
        raise AssertionError(f"Après retirer('score-2'), len(anneau) vaut {len(anneau)} au lieu de 4.")
    _comparer_anneau(anneau, ref, clients + sur_les_points, "Après ajouter('score-5') puis retirer('score-2')",
                     (SERVEURS, 100, tuple(operations)))
    for c in clients:
        if avant[c] != "score-2" and anneau.serveur_pour(c) != avant[c]:
            raise AssertionError(f"Après retirer('score-2'), le client {c} est passé de {avant[c]} à "
                                 f"{anneau.serveur_pour(c)} : seuls les clients de score-2 doivent changer. Retirez les "
                                 "points de score-2, et SEULEMENT eux.")
    # retirer puis remettre redonne exactement l'anneau d'avant
    _appel(anneau.ajouter, ("score-2",), "anneau.ajouter('score-2')")
    ref.ajouter("score-2")
    operations.append(("ajouter", "score-2"))
    _comparer_anneau(anneau, ref, clients, "Après avoir remis score-2", (SERVEURS, 100, tuple(operations)))

    # ------------------------------------------------------------------ d'autres repliques, d'autres itérables
    for serveurs, repliques in ((("a", "b"), 1), (("a", "b", "c"), 7), (("x",), 3), (("score-1", "score-11"), 12)):
        x = _creer(classe, (iter(serveurs), repliques), f"Anneau(iter({serveurs!r}), repliques={repliques})")
        _comparer_anneau(x, _RefAnneau(serveurs, repliques), clients[:600],
                         f"Anneau({serveurs!r}, repliques={repliques})", (serveurs, repliques, ()))
    vide = _creer(classe, (), "Anneau()")
    if _appel(len, (vide,), "len(Anneau())") != 0:
        raise AssertionError("len(Anneau()) devrait valoir 0.")
    _doit_lever(vide.serveur_pour, ("C000001",), "Anneau().serveur_pour('C000001')",
                "sans aucun serveur, aucune réponse possible : ValueError (pas IndexError).")
    _appel(vide.ajouter, ("seul",), "Anneau().ajouter('seul')")
    if vide.serveur_pour("C000001") != "seul":
        raise AssertionError("Un anneau d'un seul serveur doit tout lui envoyer.")
    _appel(vide.retirer, ("seul",), "retirer('seul')")
    _doit_lever(vide.serveur_pour, ("C000001",), "serveur_pour sur un anneau vidé par retirer",
                "sans aucun serveur : ValueError.")

    # ------------------------------------------------------------------ deux anneaux ne partagent rien
    a, b = _creer(classe, (["p"],), "Anneau(['p'])"), _creer(classe, (["q"],), "Anneau(['q'])")
    _appel(a.ajouter, ("r",), "a.ajouter('r')")
    if len(b) != 1 or {b.serveur_pour(c) for c in clients[:200]} != {"q"}:
        raise AssertionError("Deux anneaux partagent leurs points : après a.ajouter('r'), un AUTRE anneau b a changé. "
                             "Créez les listes dans __init__ (et pas `serveurs=[]` modifié sur place).")

    # ------------------------------------------------------------------ erreurs
    for args, attendu_exc, conseil in (
            ((SERVEURS, 0), ValueError, "repliques vaut au moins 1."),
            ((SERVEURS, 2.5), TypeError, "repliques est un int."),
            ((SERVEURS, True), TypeError, "repliques est un int, pas un bool."),
            ((["a", "a"],), ValueError, "le même serveur deux fois : ses points seraient en double."),
            (([""],), ValueError, "un nom vide n'est pas un serveur."),
            (([5],), TypeError, "un nom de serveur est un texte.")):
        _doit_lever(classe, args, f"Anneau{_court(args, 60)}", conseil, attendu_exc)
    x = _creer(classe, (["a", "b"],), "Anneau(['a', 'b'])")
    _doit_lever(x.ajouter, ("a",), "ajouter('a') alors que 'a' est déjà là", "un serveur déjà présent : ValueError.")
    _doit_lever(x.ajouter, ("",), "ajouter('')", "un nom vide : ValueError.")
    _doit_lever(x.ajouter, (None,), "ajouter(None)", "un nom de serveur est un texte.", TypeError)
    _doit_lever(x.retirer, ("c",), "retirer('c') alors que 'c' n'y est pas", "un serveur absent : ValueError.")
    _doit_lever(x.serveur_pour, (1,), "serveur_pour(1)", "la clé est un texte.", TypeError)
    if len(x) != 2 or {x.serveur_pour(c) for c in clients[:300]} != {"a", "b"}:
        raise AssertionError("Une opération refusée a modifié l'anneau : validez AVANT de toucher aux points.")

    # ------------------------------------------------------------------ 20 000 clients sur 20 000 points
    noms = [f"noeud-{i:03d}" for i in range(200)]
    grand = _creer(classe, (noms,), "Anneau(200 serveurs)")
    tous = clients_exemple(20_000)
    ref = _RefAnneau(noms)

    def demander():
        return [grand.serveur_pour(c) for c in tous]

    conseil = ("Le correcteur compte les lignes exécutées dans votre fichier : parcourir les 20 000 points à chaque client, "
               "c'est 400 millions d'étapes. Gardez les positions TRIÉES (une fois, dans ajouter) et cherchez avec "
               "`bisect.bisect_left(positions, empreinte(cle))` : une quinzaine d'étapes par client.")
    obtenus = _appel(demander, (), "serveur_pour pour 20 000 clients sur 200 serveurs × 100 points",
                     lignes_max=60 * len(tous), conseil_lent=conseil, fichier=_fichier(classe.serveur_pour))
    if obtenus != [ref.serveur_pour(c) for c in tous]:
        raise AssertionError("Sur 200 serveurs, serveur_pour donne de mauvaises réponses.")


# ---------------------------------------------------------------------------- la série

SERIE = Serie(
    code="40",
    titre="Compter sans tout retenir : filtre de Bloom, éléments fréquents, Count-Min, HyperLogLog, hachage cohérent",
    module_eleve="defis.serie_40",
    module_corrige="defis.corrige_serie_40",
    defis=(
        Defi(242, "dimensionner", "🟢", "—", verifier_dimensionner),
        Defi(243, "FiltreBloom", "🟡", "—", verifier_filtre_bloom),
        Defi(244, "frequents", "🟡", "—", verifier_frequents),
        Defi(245, "CompteurMin", "🟡", "—", verifier_compteur_min),
        Defi(246, "HyperLogLog", "🔴", "—", verifier_hyperloglog),
        Defi(247, "Anneau", "🔴", "—", verifier_anneau),
    ),
)

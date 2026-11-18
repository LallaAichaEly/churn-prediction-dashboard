"""Vérifications automatiques de la série 31 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est Nadia, de l'équipe marketing, qui va envoyer vos recommandations à 300 000 clients,
et le serveur qui compte ce que chaque calcul lui coûte :

- au défi 188, il compte les lignes exécutées dans votre fichier sur 4 000 clients et un catalogue de
  3 000 options : essayer toutes les paires du catalogue se voit, quelle que soit la machine ;
- au défi 193, il vous donne des `SimilaritesComptees` (des similarités qui comptent leurs lectures) et
  connaît le nombre de lectures qu'une mise à jour incrémentale demande ;
- partout, il rejoue les erreurs classiques (option écrite deux fois comptée deux fois, paire (a, a),
  un seul sens, confiance divisée par la mauvaise option, N sans les clients sans option, `>` au lieu de
  `>=`, `reverse=True` et ses ex aequo, option déjà détenue proposée, plan B placé avant, `<=` à la
  coupure, réabonnement compté comme une découverte, moyenne sur les seuls clients servis, DCG idéal
  calculé sur k, somme au lieu du maximum…) pour nommer la vôtre ;
- et il vérifie que vos fonctions ne modifient pas ce qu'elles reçoivent.

Les similarités et les notes de ses tirages au hasard sont des multiples de 1/16 : leurs sommes et leurs
produits sont EXACTS en virgule flottante, si bien qu'une égalité est une vraie égalité, quel que soit
l'ordre de vos additions.
"""

from __future__ import annotations

import copy
import datetime as dt
import math
import random
import sys
from collections import Counter

from .offres import (
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
    except (_BoucleSansFin, LecturesEpuisees):
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
        raise AssertionError(f"{quoi} a modifié {nom} reçu (avant : {_court(avant, 100)}, après : {_court(apres, 100)}). "
                             "Une fonction ne touche pas à ce qu'on lui prête : travaillez sur une copie, ou "
                             "construisez un nouvel objet.")


def _pas_none(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")


def _proche(x, y) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float))
            and math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12))


def _seize(rng: random.Random, haut: int = 16) -> float:
    """Un multiple de 1/16 entre 1/16 et haut/16 : exact en virgule flottante."""
    return rng.randint(1, haut) / 16


_CONSEIL_NONE = "Un panier peut être None (case vide de l'export) : `set(panier or ())` le traite comme []."


# ---------------------------------------------------------------------------- défi 188 : cooccurrences


def _ref_cooc(paniers: dict, doublons: bool = False) -> dict[str, dict[str, int]]:
    cooc: dict[str, dict[str, int]] = {}
    for panier in paniers.values():
        if not panier:
            continue
        offres = list(panier) if doublons else sorted(set(panier))
        for offre in offres:
            cooc.setdefault(offre, {})
        for i in range(len(offres)):
            for j in range(i + 1, len(offres)):
                a, b = offres[i], offres[j]
                if a != b:
                    cooc[a][b] = cooc[a].get(b, 0) + 1
                    cooc[b][a] = cooc[b].get(a, 0) + 1
    return cooc


def _comparer_cooc(r, paniers: dict, quoi: str) -> None:
    _pas_none(r, quoi)
    attendu = _ref_cooc(paniers)
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : un dict {{option: {{autre: nombre}}}} est attendu.")
    for a, voisines in r.items():
        if not isinstance(voisines, dict):
            raise AssertionError(f"Dans {quoi}, la valeur de {a!r} est un {type(voisines).__name__} "
                                 f"({_court(voisines, 60)}) : un dict {{autre option: nombre de clients}} est attendu.")
    if type(r) is not dict or any(type(v) is not dict for v in r.values()):
        raise AssertionError(f"{quoi} renvoie des defaultdict (ou une autre sorte de dict). Consulter une paire jamais "
                             "vue (cooc['tv']['presse']) y AJOUTERAIT une case à 0 : le résultat grossirait à chaque "
                             "lecture. Renvoyez de vrais dict : `{a: dict(v) for a, v in cooc.items()}`, ou "
                             "`setdefault` et `.get(b, 0)` dès le départ.")
    soi = sorted(a for a, v in r.items() if a in v)
    if soi:
        raise AssertionError(f"Dans {quoi}, {soi[0]!r} est sa propre voisine (cooc[{soi[0]!r}][{soi[0]!r}] = "
                             f"{r[soi[0]][soi[0]]}). Une paire, ce sont deux options DIFFÉRENTES : `itertools.product` "
                             "ou deux boucles sur tout le panier donnent aussi (a, a).")
    zeros = sorted((a, b) for a, v in r.items() for b, c in v.items() if c == 0)
    if zeros:
        a, b = zeros[0]
        raise AssertionError(f"Dans {quoi}, cooc[{a!r}][{b!r}] vaut 0 : seules les paires VUES sont stockées. Une paire "
                             "absente se lit avec `.get(b, 0)`.")
    for a in sorted(r):
        for b in sorted(r[a]):
            retour = r.get(b, {}).get(a)
            if retour != r[a][b]:
                raise AssertionError(f"Dans {quoi}, cooc[{a!r}][{b!r}] = {r[a][b]} mais cooc[{b!r}][{a!r}] = {retour} : le "
                                     "résultat doit être SYMÉTRIQUE. Avec `combinations` (chaque paire une fois), "
                                     "ajoutez 1 dans les DEUX sens : cooc[a][b] et cooc[b][a].")
    if r == attendu:
        return
    if set(r) == set(attendu) and all(set(r[a]) == set(attendu[a]) for a in attendu) and attendu and any(attendu.values()) \
            and all(r[a][b] == 2 * attendu[a][b] for a in attendu for b in attendu[a]):
        raise AssertionError(f"{quoi} compte chaque paire DEUX fois (cooc[a][b] double). Avec deux boucles sur tout le "
                             "panier (ou `permutations`), la paire (a, b) est déjà vue dans les deux sens : n'ajoutez "
                             "alors 1 que dans cooc[a][b]. Avec `combinations` (b après a), ajoutez 1 dans les deux sens.")
    if r == _ref_cooc(paniers, doublons=True) != attendu:
        client = next(c for c, p in paniers.items() if p and len(p) != len(set(p)))
        raise AssertionError(f"{quoi} compte deux fois une option écrite deux fois : le panier de {client} est "
                             f"{paniers[client]}. On compte des CLIENTS : `set(panier)` d'abord.")
    manquantes = sorted(set(attendu) - set(r))
    if manquantes:
        a = manquantes[0]
        if not attendu[a]:
            raise AssertionError(f"Dans {quoi}, l'option {a!r} n'est pas une clé : elle est détenue, mais jamais avec une "
                                 f"autre option. Elle doit quand même figurer, avec {{}} (`cooc.setdefault({a!r}, {{}})` "
                                 "pour chaque option du panier, même seule).")
        raise AssertionError(f"Dans {quoi}, l'option {a!r} manque (attendu : {_court(attendu[a], 80)}).")
    en_trop = sorted(set(r) - set(attendu))
    if en_trop:
        raise AssertionError(f"Dans {quoi}, l'option {en_trop[0]!r} est une clé, mais aucun client ne la détient.")
    for a in sorted(attendu):
        for b in sorted(set(attendu[a]) | set(r[a])):
            if r[a].get(b) != attendu[a].get(b):
                raise AssertionError(f"Dans {quoi}, cooc[{a!r}][{b!r}] vaut {r[a].get(b)} au lieu de {attendu[a].get(b)} "
                                     "(le nombre de clients qui ont les deux options).")
    raise AssertionError(f"{quoi} renvoie {_court(r)} au lieu de {_court(attendu)}.")


def _un_cooc(f, paniers: dict, quoi: str, lignes_max: int | None = 2_000_000, conseil_lent: str = "") -> None:
    avant = copy.deepcopy(paniers)
    r = _appel(f, (paniers,), quoi, _CONSEIL_NONE, lignes_max=lignes_max, conseil_lent=conseil_lent)
    _intact(avant, paniers, quoi, "les paniers")
    _comparer_cooc(r, paniers, quoi)


def _paniers_au_hasard(rng: random.Random, offres=("tv", "sport", "box2", "mobile", "musique", "cloud")) -> dict:
    paniers = {}
    for i in range(rng.randint(0, 8)):
        tirage = rng.random()
        if tirage < 0.1:
            paniers[f"C{i}"] = None
        else:
            paniers[f"C{i}"] = [rng.choice(offres) for _ in range(rng.randint(0, 5))]
    return paniers


def verifier_cooccurrences(f) -> None:
    _un_cooc(f, paniers_exemple(), "cooccurrences(paniers_exemple())")
    for paniers in ({}, {"A": None, "B": []}, {"A": ["tv"]}, {"A": ["tv", "tv"]}, {"A": ["tv", "sport"]},
                    {"A": ("tv", "sport"), "B": {"sport", "box2"}}):
        _un_cooc(f, paniers, f"cooccurrences({paniers!r})")
    alea = random.Random(188)
    for essai in range(250):
        paniers = _paniers_au_hasard(alea)
        _un_cooc(f, paniers, f"cooccurrences({_court(paniers, 120)}) (tiré au hasard n°{essai})")
    gros = generer_paniers(4000, graine=188, nb_offres=3000)
    paires = sum(len(set(p)) ** 2 for p in gros.values() if p)
    _un_cooc(f, gros, "cooccurrences(4 000 clients, 3 000 options)", lignes_max=8 * paires + 30 * len(gros) + 20_000,
             conseil_lent="Un client à k options ne donne que k(k-1)/2 paires : parcourez les paires DE CHAQUE PANIER. "
                          "Essayer toutes les paires du catalogue (plus de 2 millions ici) ou comparer les clients deux à "
                          "deux (8 millions), c'est compter surtout des zéros.")


# ---------------------------------------------------------------------------- défi 189 : regles_association


def _ref_regles(paniers: dict, support_min: int, variante: str = "") -> list[tuple]:
    detenteurs = Counter()
    for panier in paniers.values():
        detenteurs.update(list(panier or ()) if variante == "doublons" else set(panier or ()))
    n = len(paniers)
    if variante == "n_non_vides":
        n = sum(1 for p in paniers.values() if p)
    regles = []
    for si, voisines in _ref_cooc(paniers, doublons=variante == "doublons").items():
        for alors, support in voisines.items():
            if variante == "un_sens" and si > alors:
                continue
            if support < support_min or (variante == "strict" and support == support_min):
                continue
            confiance = support / detenteurs[alors if variante == "conf_sur_alors" else si]
            if variante == "lift_sans_n":
                lift = support / (detenteurs[si] * detenteurs[alors])
            else:
                lift = support * n / (detenteurs[si] * detenteurs[alors])
            valeur_support = support / len(paniers) if variante == "support_fraction" else support
            regles.append((si, alors, valeur_support, confiance, lift))
    regles.sort(key=lambda r: (-r[4], -r[3], r[0], r[1]))
    return regles


_VARIANTES_REGLES = [
    ("doublons", "une option écrite deux fois par l'export compte deux fois. On compte des CLIENTS : `set(panier)` "
                 "pour n(a) comme pour les paires."),
    ("conf_sur_alors", "la confiance de « si → alors » se divise par n(si), le nombre de clients qui ont `si` : "
                       "parmi EUX, la part qui a aussi `alors`. Vous divisez par n(alors)."),
    ("n_non_vides", "N doit compter TOUS les clients de `paniers`, y compris ceux sans option ([] ou None) : `len(paniers)`. "
                    "Ils auraient pu acheter ; les oublier gonfle le lift."),
    ("lift_sans_n", "le lift compare à ce que donnerait le HASARD : n(si, alors) × N / (n(si) × n(alors)). Il manque N : "
                    "sans lui, ce n'est pas un rapport de proportions."),
    ("strict", "« au moins support_min clients » : gardez support >= support_min (vous écartez aussi les règles qui "
               "ont exactement support_min)."),
    ("un_sens", "chaque paire donne DEUX règles, a → b et b → a (leurs confiances diffèrent). Vous n'en gardez qu'une."),
    ("support_fraction", "le support est un NOMBRE de clients (un entier), pas une proportion."),
]

_ORDRE_CONFIANCE = ("le tri se fait d'abord par lift décroissant ; la confiance ne départage que les lifts égaux. Trier "
                    "par confiance, c'est retomber dans le piège de la popularité.")
_ORDRE_REVERSE = ("à lift et confiance égaux, `si` puis `alors` par ordre ALPHABÉTIQUE. `reverse=True` renverse aussi "
                  "les noms : triez avec `key=lambda r: (-r.lift, -r.confiance, r.si, r.alors)`.")


def _avant(x: tuple, y: tuple) -> bool:
    """La règle x peut-elle précéder y ? (à 1e-9 près pour les flottants)"""
    if not math.isclose(x[4], y[4], rel_tol=1e-9):
        return x[4] > y[4]
    if not math.isclose(x[3], y[3], rel_tol=1e-9):
        return x[3] > y[3]
    return (x[0], x[1]) < (y[0], y[1])


def _memes_valeurs(r: list, attendu: list) -> bool:
    """Les mêmes règles, avec les mêmes valeurs (à 1e-9 près), dans n'importe quel ordre."""
    if len(r) != len(attendu):
        return False
    par_paire = {(x[0], x[1]): x for x in attendu}
    if len(par_paire) != len(attendu) or len({(x[0], x[1]) for x in r}) != len(r):
        return False
    for x in r:
        y = par_paire.get((x[0], x[1]))
        if y is None or not (x[2] == y[2] or _proche(x[2], y[2])) or not _proche(x[3], y[3]) or not _proche(x[4], y[4]):
            return False
    return True


def _memes_regles(r: list, attendu: list) -> bool:
    """Les mêmes règles, et dans un ordre correct (deux lifts égaux à 1e-9 près sont ex aequo)."""
    return _memes_valeurs(r, attendu) and all(_avant(r[i], r[i + 1]) for i in range(len(r) - 1))


def _comparer_regles(r, paniers: dict, support_min: int, quoi: str) -> None:
    _pas_none(r, quoi)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list de Regle(si, alors, support, confiance, "
                             "lift) est attendue.")
    for x in r:
        if not isinstance(x, tuple) or len(x) != 5:
            raise AssertionError(f"Dans {quoi}, l'élément {_court(x, 80)} n'est pas une Regle(si, alors, support, "
                                 "confiance, lift) (un tuple de 5 valeurs).")
    attendu = _ref_regles(paniers, support_min)
    if _memes_regles(r, attendu):
        return
    soi = [x for x in r if x[0] == x[1]]
    if soi:
        raise AssertionError(f"Dans {quoi}, la règle {soi[0][0]!r} → {soi[0][1]!r} relie une option à elle-même.")
    paires = Counter((x[0], x[1]) for x in r)
    double = [p for p, n in paires.items() if n > 1]
    if double:
        raise AssertionError(f"Dans {quoi}, la règle {double[0][0]!r} → {double[0][1]!r} apparaît {paires[double[0]]} fois.")
    if not _memes_valeurs(r, attendu):
        for variante, message in _VARIANTES_REGLES:
            autre = _ref_regles(paniers, support_min, variante)
            if _memes_valeurs(r, autre) and not _memes_valeurs(autre, attendu):
                raise AssertionError(f"{quoi} : {message}")
    par_paire = {(x[0], x[1]): x for x in attendu}
    for x in r:
        y = par_paire.get((x[0], x[1]))
        if y is None:
            raise AssertionError(f"Dans {quoi}, la règle {x[0]!r} → {x[1]!r} ne devrait pas y être (support "
                                 f"{x[2]!r} : il faut au moins {support_min} clients qui ont les deux options).")
        for i, nom in ((2, "support"), (3, "confiance"), (4, "lift")):
            if not (x[i] == y[i] or _proche(x[i], y[i])):
                raise AssertionError(f"Dans {quoi}, la règle {x[0]!r} → {x[1]!r} a un {nom} de {x[i]!r} au lieu de "
                                     f"{y[i]!r}.")
    if len(r) < len(attendu):
        vues = {(x[0], x[1]) for x in r}
        y = next(y for y in attendu if (y[0], y[1]) not in vues)
        raise AssertionError(f"Dans {quoi}, il manque la règle {y[0]!r} → {y[1]!r} (support {y[2]}, confiance "
                             f"{y[3]:.4f}, lift {y[4]:.4f}).")
    i = next(i for i in range(len(r) - 1) if not _avant(r[i], r[i + 1]))
    x, y = r[i], r[i + 1]
    pourquoi = "triez par lift décroissant, puis confiance décroissante, puis si et alors par ordre alphabétique."
    if not math.isclose(x[4], y[4], rel_tol=1e-9) and x[3] > y[3]:
        pourquoi = _ORDRE_CONFIANCE
    elif math.isclose(x[4], y[4], rel_tol=1e-9) and math.isclose(x[3], y[3], rel_tol=1e-9):
        pourquoi = _ORDRE_REVERSE
    raise AssertionError(f"Dans {quoi}, la règle {x[0]!r} → {x[1]!r} (lift {x[4]:.4f}, confiance {x[3]:.4f}) est placée "
                         f"avant {y[0]!r} → {y[1]!r} (lift {y[4]:.4f}, confiance {y[3]:.4f}) : {pourquoi}")


def _une_regle(f, paniers: dict, support_min: int, quoi: str, lignes_max: int | None = 2_000_000,
               conseil_lent: str = "") -> None:
    avant = copy.deepcopy(paniers)
    r = _appel(f, (paniers,), quoi, _CONSEIL_NONE, lignes_max=lignes_max, conseil_lent=conseil_lent,
               kwargs={"support_min": support_min})
    _intact(avant, paniers, quoi, "les paniers")
    _comparer_regles(r, paniers, support_min, quoi)


def verifier_regles_association(f) -> None:
    for support_min in (2, 1, 3, 4):
        _une_regle(f, paniers_exemple(), support_min, f"regles_association(paniers_exemple(), support_min={support_min})")
    for mauvais in (0, -1, 1.5, "2"):
        _doit_lever(f, (paniers_exemple(),), f"regles_association(paniers_exemple(), support_min={mauvais!r})",
                    "support_min doit être un ENTIER >= 1 : une règle vue chez zéro client n'existe pas.",
                    kwargs={"support_min": mauvais})
    for paniers in ({}, {"A": None, "B": []}, {"A": ["tv", "sport"]}, {"A": ["tv", "sport", "tv"], "B": ["tv"]}):
        _une_regle(f, paniers, 1, f"regles_association({paniers!r}, support_min=1)")
    alea = random.Random(189)
    for essai in range(200):
        paniers = _paniers_au_hasard(alea)
        support_min = alea.randint(1, 3)
        _une_regle(f, paniers, support_min, f"regles_association({_court(paniers, 110)}, support_min={support_min}) "
                                            f"(tiré au hasard n°{essai})")
    gros = generer_paniers(3000, graine=189, nb_offres=2000)
    paires = sum(len(set(p)) ** 2 for p in gros.values() if p)
    _une_regle(f, gros, 25, "regles_association(3 000 clients, 2 000 options, support_min=25)",
               lignes_max=10 * paires + 40 * len(gros) + 30_000,
               conseil_lent="Partez des cooccurrences (défi 188) : seules les paires VUES peuvent donner une règle. "
                            "Essayer toutes les paires du catalogue, c'est 4 millions de paires pour quelques centaines "
                            "de règles.")


# ---------------------------------------------------------------------------- défi 190 : recommander

_VOISINS = {
    "tv": {"sport": 0.5, "box2": 0.75, "mobile": 0.25, "assurance": 0.0},
    "sport": {"tv": 0.5, "box2": 0.25, "presse": 0.25},
    "box2": {"tv": 0.75, "sport": 0.25, "assurance": 0.0},
    "mobile": {"musique": 0.5, "tv": 0.25, "gaming": 0.25},
    "musique": {"mobile": 0.5, "gaming": 0.5},
    "gaming": {"musique": 0.5, "mobile": 0.25},
    "cloud": {"assurance": 0.75, "presse": 0.5},
    "antivirus": {"presse": 0.5},
}
_POPULAIRES = ["mobile", "tv", "musique", "sport", "cloud", "gaming", "box2", "antivirus", "presse", "assurance"]


def _ref_recommander(panier, voisins: dict, pop: list, k: int, variante: str = "") -> list[str]:
    detenues = set(panier or ())
    scores: dict[str, float] = {}
    for offre in (list(panier or ()) if variante == "doublons" else sorted(detenues)):
        for autre, s in voisins.get(offre, {}).items():
            if autre not in detenues or variante == "deja":
                scores[autre] = max(scores.get(autre, 0.0), s) if variante == "max" else scores.get(autre, 0.0) + s
    garder = (lambda s: s >= 0) if variante == "zero" else (lambda s: s > 0)
    if variante == "reverse":
        classees = sorted((o for o, s in scores.items() if garder(s)), key=lambda o: (scores[o], o), reverse=True)
    else:
        classees = sorted((o for o, s in scores.items() if garder(s)), key=lambda o: (-scores[o], o))
    plan_b = [o for o in pop if o not in detenues]
    if variante == "sans_plan_b":
        return classees[:k]
    if variante == "plan_b_avant":
        resultat = []
        for o in plan_b + classees:
            if o not in resultat:
                resultat.append(o)
        return resultat[:k]
    resultat = classees[:k]
    for o in plan_b:
        if len(resultat) >= k:
            break
        if o not in resultat:
            resultat.append(o)
    return resultat


_VARIANTES_RECOMMANDER = [
    ("doublons", "une option écrite deux fois dans le panier compte deux fois dans les scores. Chaque option du "
                 "client compte une fois : parcourez `set(panier)`."),
    ("max", "le score d'une option est la SOMME de ses similarités avec les options du client, pas la plus forte : "
            "une option proche de deux options du client passe avant une option proche d'une seule."),
    ("zero", "une option de similarité 0 n'est pas proposée pour son score : seules les options de score > 0 passent "
             "avant le plan B (les populaires, dans leur ordre)."),
    ("reverse", "à score égal, l'ordre ALPHABÉTIQUE. `reverse=True` renverse aussi les noms : triez avec "
                "`key=lambda o: (-scores[o], o)`."),
    ("plan_b_avant", "le plan B (les populaires) vient APRÈS les options proches de celles du client, jamais avant : "
                     "ce qu'on sait du client passe d'abord."),
    ("sans_plan_b", "quand les options proches ne suffisent pas pour arriver à k, complétez avec `populaires` (dans "
                    "l'ordre, sans doublon, sans ce que le client a déjà). Un client dont on ne sait rien doit recevoir "
                    "quelque chose."),
]


def _comparer_recommander(r, panier, voisins: dict, pop: list, k: int, quoi: str) -> None:
    _pas_none(r, quoi)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list d'options est attendue.")
    attendu = _ref_recommander(panier, voisins, pop, k)
    if r == attendu:
        return
    detenues = set(panier or ())
    deja = [o for o in r if o in detenues]
    if deja:
        raise AssertionError(f"{quoi} propose {deja[0]!r}, que le client a DÉJÀ (son panier : {panier!r}). Écartez les "
                             "options détenues, des scores comme du plan B.")
    doubles = [o for o, n in Counter(r).items() if n > 1]
    if doubles:
        raise AssertionError(f"{quoi} propose deux fois {doubles[0]!r} ({r}). Le plan B ne reprend pas une option déjà "
                             "proposée.")
    if len(r) > k:
        raise AssertionError(f"{quoi} renvoie {len(r)} options ({r}) : au plus k = {k}.")
    for variante, message in _VARIANTES_RECOMMANDER:
        if r == _ref_recommander(panier, voisins, pop, k, variante) != attendu:
            raise AssertionError(f"{quoi} renvoie {r} au lieu de {attendu} : {message}")
    if len(r) < len(attendu):
        raise AssertionError(f"{quoi} renvoie {r} ({len(r)} options) au lieu de {attendu} : il y avait de quoi en "
                             f"proposer {len(attendu)}.")
    raise AssertionError(f"{quoi} renvoie {r} au lieu de {attendu}.")


def _une_recommandation(f, panier, voisins: dict, pop: list, k: int, quoi: str) -> None:
    copies = copy.deepcopy((panier, voisins, pop))
    conseil = (_CONSEIL_NONE if panier is None else
               "Une option du client peut ne pas avoir de ligne dans `voisins` (une option toute nouvelle) : "
               "`voisins.get(offre, {})`.")
    r = _appel(f, (panier, voisins, pop), quoi, conseil, kwargs={"k": k})
    for avant, apres, nom in zip(copies, (panier, voisins, pop), ("le panier", "les voisins", "les populaires")):
        _intact(avant, apres, quoi, nom)
    _comparer_recommander(r, panier, voisins, pop, k, quoi)


def verifier_recommander(f) -> None:
    cas = [
        (["tv"], 3), (["tv", "sport"], 3), (["tv", "tv", "mobile"], 3), (["cloud", "antivirus"], 2), (["box2"], 3),
        ([], 3), (None, 2), (["presse"], 2), (["nouvelle_offre", "musique"], 4), (["tv"], 0), (["tv"], 20),
        (["mobile", "gaming"], 1), (("tv", "sport"), 2),
    ]
    for panier, k in cas:
        _une_recommandation(f, panier, _VOISINS, _POPULAIRES, k, f"recommander({panier!r}, voisins, populaires, k={k})")
    _doit_lever(f, (["tv"], _VOISINS, _POPULAIRES), "recommander(['tv'], voisins, populaires, k=-1)",
                "k < 0 n'a pas de sens : levez ValueError.", kwargs={"k": -1})
    paniers = paniers_exemple()
    sim, pop = similarites_cosinus(paniers), populaires(paniers)
    for client in ("C01", "C08", "C13", "C09"):
        _une_recommandation(f, paniers[client], sim, pop, 4, f"recommander(paniers_exemple()[{client!r}], "
                                                             "similarites_cosinus(…), populaires(…), k=4)")
    alea = random.Random(190)
    offres = ["tv", "sport", "box2", "mobile", "musique", "cloud", "gaming", "presse"]
    for essai in range(300):
        voisins = {}
        for a in offres:
            if alea.random() < 0.85:
                voisins[a] = {b: _seize(alea, 8) for b in alea.sample(offres, alea.randint(0, 5)) if b != a}
        panier = [alea.choice(offres + ["inconnue"]) for _ in range(alea.randint(0, 4))]
        pop = alea.sample(offres, alea.randint(3, len(offres)))
        k = alea.randint(0, 7)
        _une_recommandation(f, panier, voisins, pop, k, f"recommander({panier!r}, {_court(voisins, 90)}, {pop!r}, "
                                                        f"k={k}) (tiré au hasard n°{essai})")


# ---------------------------------------------------------------------------- défi 191 : decouper_dans_le_temps


def _ref_decouper(abonnements: list, coupure: dt.date, variante: str = "") -> tuple[dict, dict]:
    historique: dict[str, set] = {}
    apres: dict[str, set] = {}
    if variante == "ordre":                       # décide en lisant, comme si le journal était trié
        a_deviner: dict[str, set] = {}
        for client, offre, date in abonnements:
            historique.setdefault(client, set())
            if date < coupure:
                historique[client].add(offre)
            elif offre not in historique[client]:
                a_deviner.setdefault(client, set()).add(offre)
        return historique, a_deviner
    for client, offre, date in abonnements:
        if variante != "froids_absents":
            historique.setdefault(client, set())
        avant = date <= coupure if variante == "inclusif" else date < coupure
        if avant:
            historique.setdefault(client, set()).add(offre)
        else:
            apres.setdefault(client, set()).add(offre)
            if variante == "historique_tout":
                historique.setdefault(client, set()).add(offre)
    a_deviner = {}
    for client, offres in apres.items():
        nouvelles = offres if variante == "cible_deja" else offres - historique.get(client, set())
        if variante == "historique_tout":
            nouvelles = set(offres)
        if nouvelles or variante == "vides":
            a_deviner[client] = nouvelles
    return historique, a_deviner


_VARIANTES_DECOUPER = [
    ("ordre", "le journal est dans le DÉSORDRE : une souscription d'avant la coupure peut être lue APRÈS celle "
              "d'après. On ne sait ce qui est nouveau qu'à la fin : remplissez d'abord l'historique, puis faites la "
              "différence."),
    ("inclusif", "le jour de la coupure appartient à l'AVENIR : l'historique ne garde que les souscriptions "
                 "strictement avant (`date < coupure`)."),
    ("cible_deja", "une option que le client avait déjà avant la coupure (un réabonnement, une ligne rejouée) n'est "
                   "pas à deviner : sinon on récompense le modèle pour proposer ce que le client a déjà. Retirez "
                   "l'historique : `apres[client] - historique[client]`."),
    ("froids_absents", "TOUT client du journal figure dans l'historique, même sans souscription avant la coupure "
                       "(set() : un client « froid », à qui il faudra quand même recommander quelque chose)."),
    ("vides", "un client qui n'a rien de NOUVEAU après la coupure n'a rien à deviner : il ne figure pas dans "
              "a_deviner (pas d'ensemble vide)."),
    ("historique_tout", "l'historique ne contient que le PASSÉ (`date < coupure`) : y mettre l'avenir, c'est laisser "
                        "le modèle apprendre sur ce qu'il doit deviner."),
]


def _comparer_decouper(r, abonnements: list, coupure: dt.date, quoi: str) -> None:
    _pas_none(r, quoi)
    if not isinstance(r, tuple) or len(r) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : un tuple (historique, a_deviner) est attendu.")
    historique, a_deviner = r
    for nom, d in (("historique", historique), ("a_deviner", a_deviner)):
        if not isinstance(d, dict):
            raise AssertionError(f"Dans {quoi}, {nom} est un {type(d).__name__} : un dict {{client: set d'options}} est "
                                 "attendu.")
        for client, offres in d.items():
            if not isinstance(offres, (set, frozenset)):
                raise AssertionError(f"Dans {quoi}, {nom}[{client!r}] est un {type(offres).__name__} ({_court(offres, 60)}) : "
                                     "un ensemble (set) d'options est attendu (un client compte une fois par option).")
    attendu = _ref_decouper(abonnements, coupure)
    if (historique, a_deviner) == attendu:
        return
    for variante, message in _VARIANTES_DECOUPER:
        if (historique, a_deviner) == _ref_decouper(abonnements, coupure, variante) != attendu:
            raise AssertionError(f"{quoi} : {message}")
    for nom, d, ref in (("historique", historique, attendu[0]), ("a_deviner", a_deviner, attendu[1])):
        for client in sorted(set(d) | set(ref)):
            if d.get(client) != ref.get(client):
                obtenu = "absent" if client not in d else _court(d[client], 60)
                voulu = "absent" if client not in ref else _court(ref[client], 60)
                raise AssertionError(f"Dans {quoi}, {nom}[{client!r}] vaut {obtenu} au lieu de {voulu}.")
    raise AssertionError(f"{quoi} renvoie {_court(r)} au lieu de {_court(attendu)}.")


def _un_decoupage(f, abonnements: list, coupure: dt.date, quoi: str) -> None:
    avant = list(abonnements)
    r = _appel(f, (abonnements, coupure), quoi)
    _intact(avant, abonnements, quoi, "le journal")
    _comparer_decouper(r, abonnements, coupure, quoi)


def verifier_decouper_dans_le_temps(f) -> None:
    j = dt.date
    _un_decoupage(f, abonnements_exemple(), j(2026, 6, 1), "decouper_dans_le_temps(abonnements_exemple(), 2026-06-01)")
    cas = [
        [],
        [("A", "tv", j(2026, 6, 1))],
        [("A", "tv", j(2026, 5, 31))],
        [("A", "tv", j(2026, 8, 1)), ("A", "tv", j(2026, 1, 1))],          # l'avant est lu APRÈS l'après
        [("A", "tv", j(2026, 1, 1)), ("A", "tv", j(2026, 8, 1)), ("A", "sport", j(2026, 9, 1))],
        [("A", "tv", j(2026, 7, 1)), ("A", "tv", j(2026, 7, 1)), ("B", "sport", j(2026, 2, 1))],
    ]
    for abonnements in cas:
        _un_decoupage(f, abonnements, j(2026, 6, 1), f"decouper_dans_le_temps({_court(abonnements, 130)}, 2026-06-01)")
    alea = random.Random(191)
    for essai in range(200):
        abonnements = [(alea.choice("ABCD"), alea.choice(("tv", "sport", "box2")), j(2026, alea.randint(4, 8), 1))
                       for _ in range(alea.randint(0, 9))]
        _un_decoupage(f, abonnements, j(2026, 6, 1), f"decouper_dans_le_temps({_court(abonnements, 110)}, 2026-06-01) "
                                                     f"(tiré au hasard n°{essai})")
    _un_decoupage(f, generer_abonnements(2000, graine=191), j(2026, 1, 1),
                  "decouper_dans_le_temps(generer_abonnements(2000, graine=191), 2026-01-01)")


# ---------------------------------------------------------------------------- défi 192 : mesurer


def _ref_mesurer(recommandations: dict, a_deviner: dict, k: int, variante: str = "") -> dict[str, float]:
    clients = list(a_deviner)
    if variante == "moyenne_servis":
        clients = [c for c in a_deviner if recommandations.get(c)]
    if variante == "tous_clients":
        clients = list(a_deviner) + [c for c in recommandations if c not in a_deviner]
    if not clients:
        return {"precision": 0.0, "rappel": 0.0, "ndcg": 0.0}
    tp = tr = tn = 0.0
    for client in clients:
        cible = a_deviner.get(client, set())
        liste = list(recommandations.get(client) or [])
        montrees = liste if variante == "sans_coupe" else liste[:k]
        vues: set = set()
        touches, dcg = 0, 0.0
        for rang, offre in enumerate(montrees, start=1):
            if offre in cible and (offre not in vues or variante == "doublons"):
                touches += 1
                if variante == "log_rang":
                    dcg += math.inf if rang == 1 else 1 / math.log2(rang)
                else:
                    dcg += 1 / math.log2(rang + (2 if variante == "rang_decale" else 1))
            vues.add(offre)
        if not cible:
            continue
        nb_ideal = k if variante == "ideal_k" else min(k, len(cible))
        ideal = sum(1 / math.log2(r + 1) for r in range(1, nb_ideal + 1))
        if variante == "precision_longueur":
            tp += touches / len(montrees) if montrees else 0.0
        else:
            tp += touches / k
        if variante == "rappel_k":
            tr += touches / k
        elif variante == "rappel_min":
            tr += touches / min(k, len(cible))
        else:
            tr += touches / len(cible)
        tn += dcg / ideal
    m = len(clients)
    return {"precision": tp / m, "rappel": tr / m, "ndcg": tn / m}


_VARIANTES_MESURER = [
    ("moyenne_servis", "la moyenne porte sur TOUS les clients à deviner. Un client pour qui on n'a rien proposé "
                       "compte 0 : l'oublier embellit le résultat (`recommandations.get(client) or []`)."),
    ("tous_clients", "seuls les clients de `a_deviner` comptent : un client qui n'avait rien à deviner n'a pas de "
                     "note."),
    ("sans_coupe", "seules les k PREMIÈRES options proposées comptent (`liste[:k]`) : le client ne voit pas les "
                   "suivantes."),
    ("doublons", "une option proposée deux fois ne touche qu'une fois (gardez un set des options déjà vues) : la "
                 "deuxième occupe une place pour rien."),
    ("precision_longueur", "la précision@k se divise par k, même si la liste est plus courte : proposer moins n'est "
                           "pas récompensé."),
    ("rappel_k", "le rappel se divise par la taille de la CIBLE (ce qu'il y avait à trouver), ni par k, ni par "
                 "min(k, taille de la cible) : une cible d'une option, trouvée, vaut 1 ; une cible de 5 options dont "
                 "on trouve 3 (k = 3) vaut 0,6, car il en reste 2 à découvrir."),
    ("rappel_min", "le rappel se divise par la taille de la CIBLE (ce qu'il y avait à trouver), ni par k, ni par "
                   "min(k, taille de la cible) : une cible d'une option, trouvée, vaut 1 ; une cible de 5 options dont "
                   "on trouve 3 (k = 3) vaut 0,6, car il en reste 2 à découvrir."),
    ("log_rang", "une touche au rang r vaut 1 / log2(r + 1) : au rang 1, 1 / log2(2) = 1 ; au rang 2, 0,63. Avec "
                 "log2(r), le rang 2 vaut 1 et le rang 1 divise par zéro."),
    ("ideal_k", "le DCG idéal ne compte que min(k, taille de la cible) rangs : avec une seule option à deviner, la "
                "meilleure liste n'a qu'une touche (au rang 1), et son NDCG doit pouvoir valoir 1."),
    ("rang_decale", "le rang commence à 1 : la première option proposée vaut 1 / log2(1 + 1) = 1. Vous décalez d'un "
                    "rang (`enumerate(liste, start=1)` puis log2(rang + 1))."),
]


def _comparer_mesurer(r, recommandations: dict, a_deviner: dict, k: int, quoi: str) -> None:
    _pas_none(r, quoi)
    if not isinstance(r, dict) or set(r) != {"precision", "rappel", "ndcg"}:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : un dict avec exactement les clés 'precision', "
                             "'rappel' et 'ndcg' est attendu.")
    for nom, valeur in r.items():
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
            raise AssertionError(f"Dans {quoi}, {nom!r} vaut {_court(valeur, 60)} : un nombre (float) est attendu.")
    attendu = _ref_mesurer(recommandations, a_deviner, k)
    for nom in ("precision", "rappel", "ndcg"):
        if _proche(r[nom], attendu[nom]):
            continue
        for variante, message in _VARIANTES_MESURER:
            autre = _ref_mesurer(recommandations, a_deviner, k, variante)
            if _proche(r[nom], autre[nom]) and not _proche(autre[nom], attendu[nom]):
                raise AssertionError(f"{quoi} : {nom} vaut {r[nom]:.4f} au lieu de {attendu[nom]:.4f}. "
                                     f"{message[0].upper()}{message[1:]}")
        raise AssertionError(f"{quoi} : {nom} vaut {r[nom]!r} au lieu de {attendu[nom]!r}.")


def _une_mesure(f, recommandations: dict, a_deviner: dict, k: int, quoi: str) -> None:
    copies = copy.deepcopy((recommandations, a_deviner))
    r = _appel(f, (recommandations, a_deviner), quoi,
               "Avec `enumerate(…, start=1)`, le premier rang vaut 1 et log2(1 + 1) = 1 (log2(1) = 0 : division par "
               "zéro).", kwargs={"k": k})
    for avant, apres, nom in zip(copies, (recommandations, a_deviner), ("les recommandations", "a_deviner")):
        _intact(avant, apres, quoi, nom)
    _comparer_mesurer(r, recommandations, a_deviner, k, quoi)


def verifier_mesurer(f) -> None:
    a_deviner = {"C01": {"tv"}, "C02": {"musique"}, "C03": {"gaming"}}
    recommandations = {"C01": ["sport", "tv", "box2"], "C02": ["gaming", "musique"], "C04": ["sport"]}
    cas = [
        (recommandations, a_deviner, 3),
        ({"A": ["tv", "box2"]}, {"A": {"tv", "box2"}}, 2),
        ({}, {"A": {"tv"}}, 3),
        ({"A": []}, {"A": {"tv"}}, 3),
        ({"A": ["tv"]}, {"A": {"tv"}}, 3),                                      # liste plus courte que k
        ({"A": ["tv", "tv", "sport"]}, {"A": {"tv", "sport"}}, 3),              # « tv » deux fois
        ({"A": ["box2", "presse", "tv"]}, {"A": {"tv"}}, 2),                    # la touche est au-delà de k
        ({"A": ["tv", "sport", "box2"]}, {"A": {"tv", "sport", "box2", "mobile", "cloud"}}, 3),
        ({"A": ["box2", "tv"]}, {"A": {"tv"}}, 1),
        ({"A": ["box2", "tv"], "B": ["tv"]}, {"A": {"tv", "box2"}}, 2),         # B n'avait rien à deviner
        ({"A": ["tv"]}, {"A": {"tv"}, "B": {"box2"}}, 1),                       # B n'a rien reçu
    ]
    for recos, cibles, k in cas:
        _une_mesure(f, recos, cibles, k, f"mesurer({_court(recos, 90)}, {_court(cibles, 70)}, k={k})")
    for k in (0, -1):
        _doit_lever(f, (recommandations, a_deviner), f"mesurer(…, k={k})", "k doit être un entier >= 1.",
                    kwargs={"k": k})
    _doit_lever(f, ({"A": ["tv"]}, {}), "mesurer({'A': ['tv']}, {}, k=3)",
                "Sans aucun client à deviner, la moyenne n'existe pas : levez ValueError plutôt que de renvoyer 0 "
                "(ou de diviser par zéro).", kwargs={"k": 3})
    alea = random.Random(192)
    offres = ["tv", "sport", "box2", "mobile", "musique", "cloud"]
    for essai in range(250):
        clients = [f"C{i}" for i in range(alea.randint(1, 5))]
        cibles = {c: set(alea.sample(offres, alea.randint(1, 4))) for c in clients if alea.random() < 0.8}
        if not cibles:
            cibles = {clients[0]: {alea.choice(offres)}}
        recos = {c: [alea.choice(offres) for _ in range(alea.randint(0, 5))] for c in clients + ["X"]
                 if alea.random() < 0.8}
        k = alea.randint(1, 4)
        _une_mesure(f, recos, cibles, k, f"mesurer({_court(recos, 90)}, {_court(cibles, 70)}, k={k}) "
                                         f"(tiré au hasard n°{essai})")


# ---------------------------------------------------------------------------- défi 193 : diversifier


def _ref_diversifier(candidats: dict, similarites, k: int, lam: float, variante: str = "") -> list[str]:
    if variante == "nom_seul":
        restantes = sorted(candidats)
    elif variante == "reverse":
        restantes = sorted(candidats, key=lambda o: (candidats[o], o), reverse=True)
    else:
        restantes = sorted(candidats, key=lambda o: (-candidats[o], o))
    penalite = dict.fromkeys(restantes, 0.0)
    choix: list[str] = []
    while restantes and len(choix) < k:
        meilleure, valeur = None, -math.inf
        for o in restantes:
            if variante == "lam_inverse":
                v = (1 - lam) * candidats[o] - lam * penalite[o]
            elif variante == "note_entiere":
                v = candidats[o] - (1 - lam) * penalite[o]
            else:
                v = lam * candidats[o] - (1 - lam) * penalite[o]
            if v > valeur:
                meilleure, valeur = o, v
        choix.append(meilleure)
        restantes.remove(meilleure)
        for o in restantes:                       # la pénalité, mise à jour avec le seul nouveau choix
            ligne = similarites.get(o)
            s = 0.0 if ligne is None else ligne.get(meilleure, 0.0)
            if variante == "somme":
                penalite[o] += s
            elif variante == "derniere":
                penalite[o] = s
            elif variante == "premiere":
                penalite[o] = s if len(choix) == 1 else penalite[o]
            else:
                penalite[o] = max(penalite[o], s)
    return choix


def _erreur_reconnue(f, reference, variantes: list, batterie: list, cas: tuple, r) -> str:
    """Le message de L'erreur classique qui reproduit vos résultats sur TOUTE une batterie de cas, sinon "".

    Une seule entrée ne suffit pas : deux erreurs différentes peuvent donner la même réponse sur un exemple. On
    rejoue donc votre fonction sur des dizaines de cas, et on ne nomme une erreur que si elle les explique tous.
    """
    sorties = []
    for args in [cas] + batterie:
        try:
            sortie = r if args is cas else _trace(f, lambda: f(*copy.deepcopy(args[:2]), k=args[2], lam=args[3]),
                                                   200_000)
        except BaseException:  # noqa: BLE001 — un plantage ici ne dit rien : ce cas est simplement ignoré
            continue
        sorties.append((args, sortie))
    for variante, message in variantes:
        if all(sortie == reference(*args, variante=variante) for args, sortie in sorties) and \
                any(reference(*args, variante=variante) != reference(*args) for args, _ in sorties):
            return message
    return ""


def _batterie_diversifier() -> list[tuple]:
    alea = random.Random(1930)
    batterie = [
        ({"tv": 0.9, "box2": 0.85, "sport": 0.8, "musique": 0.5},
         {"tv": {"box2": 0.9, "sport": 0.6}, "box2": {"tv": 0.9, "sport": 0.5}, "sport": {"tv": 0.6, "box2": 0.5}},
         3, 0.7),
    ]
    noms = "abcdefg"
    for _ in range(60):
        cand = {o: alea.randint(0, 16) / 16 for o in alea.sample(noms, alea.randint(2, len(noms)))}
        sims = {a: {b: alea.randint(0, 16) / 16 for b in cand if b != a and alea.random() < 0.7} for a in cand}
        batterie.append((cand, sims, alea.randint(2, 6), alea.choice((0.25, 0.5, 0.75))))
    return batterie


_VARIANTES_DIVERSIFIER = [
    ("somme", "la pénalité est la similarité la PLUS FORTE avec une option déjà choisie (`max`), pas leur somme : "
              "sinon chaque nouveau choix pénalise un peu plus tout le monde, même ce qui ne ressemble à rien."),
    ("derniere", "la pénalité tient compte de TOUTES les options déjà choisies, pas seulement de la dernière : une "
                 "option presque identique à la première choisie doit le rester au 3e tour. Gardez le MAXIMUM : "
                 "`penalite[o] = max(penalite[o], similarité avec le nouveau choix)`."),
    ("premiere", "la pénalité tient compte de TOUTES les options déjà choisies, pas seulement de la première. "
                 "Mettez-la à jour après chaque choix."),
    ("lam_inverse", "`lam` pondère la NOTE (lam = 1 : seule la note compte), et 1 − lam la ressemblance : "
                    "valeur = lam × note − (1 − lam) × pénalité."),
    ("note_entiere", "la note est multipliée par `lam` : valeur = lam × note − (1 − lam) × pénalité."),
    ("nom_seul", "à valeur égale, la meilleure NOTE d'abord, et seulement ensuite l'ordre alphabétique : partez de "
                 "`sorted(candidats, key=lambda o: (-candidats[o], o))`."),
    ("reverse", "à valeur égale, la meilleure note puis l'ordre ALPHABÉTIQUE. `reverse=True` renverse aussi les "
                "noms : triez avec `key=lambda o: (-candidats[o], o)`."),
]


def _comparer_diversifier(r, f, candidats: dict, similarites: dict, k: int, lam: float, quoi: str) -> None:
    """`similarites` : un vrai dict (le correcteur ne puise pas dans le budget de lectures de votre fonction)."""
    _pas_none(r, quoi)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list d'options est attendue.")
    attendu = _ref_diversifier(candidats, similarites, k, lam)
    if r == attendu:
        return
    doubles = [o for o, n in Counter(r).items() if n > 1]
    if doubles:
        raise AssertionError(f"{quoi} choisit deux fois {doubles[0]!r} ({r}) : une option choisie quitte les candidates.")
    inconnues = [o for o in r if o not in candidats]
    if inconnues:
        raise AssertionError(f"{quoi} renvoie {inconnues[0]!r}, qui n'est pas une candidate.")
    if len(r) != len(attendu):
        raise AssertionError(f"{quoi} renvoie {len(r)} options ({_court(r, 80)}) au lieu de {len(attendu)} : "
                             "min(k, nombre de candidates).")
    message = _erreur_reconnue(f, _ref_diversifier, _VARIANTES_DIVERSIFIER, _batterie_diversifier(),
                               (candidats, similarites, k, lam), r)
    if message:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)} : {message}")
    i = next(i for i, (x, y) in enumerate(zip(r, attendu)) if x != y)
    raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)} : au tour {i + 1}, "
                         f"{attendu[i]!r} a la meilleure valeur lam × note − (1 − lam) × pénalité.")


def _une_diversification(f, candidats: dict, similarites, k: int, lam: float, quoi: str,
                         conseil_lent: str = "", brutes: dict | None = None) -> None:
    copies = copy.deepcopy(candidats), (None if isinstance(similarites, SimilaritesComptees) else
                                        copy.deepcopy(similarites))
    r = _appel(f, (candidats, similarites), quoi,
               "Une candidate peut ne pas avoir de ligne dans `similarites` : `similarites.get(o, {}).get(c, 0.0)`. "
               "Au premier tour, rien n'est choisi : la pénalité vaut 0 (`max` d'une liste vide lève ValueError).",
               lignes_max=None, conseil_lent=conseil_lent, kwargs={"k": k, "lam": lam})
    _intact(copies[0], candidats, quoi, "les candidats")
    if copies[1] is not None:
        _intact(copies[1], similarites, quoi, "les similarités")
    _comparer_diversifier(r, f, candidats, similarites if brutes is None else brutes, k, lam, quoi)


def verifier_diversifier(f) -> None:
    candidats = {"tv": 0.9, "box2": 0.85, "sport": 0.8, "musique": 0.5}
    similarites = {"tv": {"box2": 0.9, "sport": 0.6}, "box2": {"tv": 0.9, "sport": 0.5},
                   "sport": {"tv": 0.6, "box2": 0.5}}
    for k, lam in ((3, 0.7), (3, 1.0), (4, 0.0), (0, 0.7), (10, 0.7), (2, 0.5), (4, 0.3)):
        _une_diversification(f, candidats, similarites, k, lam, f"diversifier(candidats, similarites, k={k}, lam={lam})")
    cas = [
        ({}, {}, 3, 0.7),
        ({"tv": 0.5, "box2": 0.5, "sport": 0.5}, {}, 3, 0.5),                       # tout à égalité : l'alphabet
        ({"box2": 0.5, "tv": 0.75, "sport": 0.75}, {"sport": {"tv": 0.5}}, 3, 0.5),  # la note avant le nom
        ({"a": 1.0, "b": 0.75, "c": 0.75, "d": 0.5},
         {"b": {"a": 1.0}, "c": {"a": 0.25, "b": 1.0}, "d": {"a": 0.25}}, 4, 0.5),
        ({"a": 1.0, "b": 0.875, "c": 0.5, "d": 0.5},
         {"c": {"a": 0.25, "b": 0.25}, "d": {"a": 0.375}}, 3, 0.5),                   # somme ≠ max
        ({"a": 1.0, "b": 0.875, "c": 0.75, "d": 0.5}, {"c": {"b": 1.0}}, 3, 0.5),    # la 2e choisie compte aussi
        ({"a": 1.0, "b": 0.9375, "c": 0.875, "d": 0.25},
         {"c": {"a": 1.0}, "b": {"a": 0.125}, "d": {"a": 0.0}}, 4, 0.5),           # la 1re choisie pèse encore
    ]
    for cand, sims, k, lam in cas:
        _une_diversification(f, cand, sims, k, lam, f"diversifier({cand!r}, {sims!r}, k={k}, lam={lam})")
    for lam in (1.5, -0.1, float("nan")):
        _doit_lever(f, (candidats, similarites), f"diversifier(candidats, similarites, k=3, lam={lam})",
                    "lam doit être entre 0 et 1 : `if not 0 <= lam <= 1: raise ValueError(…)` refuse aussi NaN.",
                    kwargs={"k": 3, "lam": lam})
    _doit_lever(f, (candidats, similarites), "diversifier(candidats, similarites, k=-1)", "k < 0 : ValueError.",
                kwargs={"k": -1, "lam": 0.7})
    alea = random.Random(193)
    noms = ["tv", "sport", "box2", "mobile", "musique", "cloud", "gaming"]
    for essai in range(300):
        cand = {o: alea.randint(0, 16) / 16 for o in alea.sample(noms, alea.randint(0, len(noms)))}
        sims = {}
        for a in cand:
            if alea.random() < 0.85:
                sims[a] = {b: alea.randint(0, 16) / 16 for b in cand if b != a and alea.random() < 0.6}
        k, lam = alea.randint(0, 6), alea.choice((0.0, 0.25, 0.5, 0.75, 1.0))
        _une_diversification(f, cand, sims, k, lam, f"diversifier({_court(cand, 80)}, {_court(sims, 80)}, k={k}, "
                                                    f"lam={lam}) (tiré au hasard n°{essai})")
    n, k = 3000, 40
    cand = {f"offre{i:04d}": alea.randint(1, 64) / 64 for i in range(n)}
    noms_gros = list(cand)
    lignes = {o: {b: alea.randint(1, 16) / 16 for b in alea.sample(noms_gros, 12) if b != o} for o in noms_gros}
    comptees = SimilaritesComptees(lignes, budget=2 * k * n + 2 * n)
    _une_diversification(f, cand, comptees, k, 0.5, f"diversifier({n} candidates, similarités comptées, k={k}, lam=0.5)",
                         brutes=lignes,
                         conseil_lent=f"Plus de {2 * k * n + 2 * n} similarités lues. Relire à chaque tour la "
                                      "similarité de chaque candidate avec TOUTES les options déjà choisies (ou "
                                      "parcourir toute sa ligne), c'est k fois trop. Gardez pour chaque candidate sa "
                                      "pénalité la plus forte jusqu'ici, et mettez-la à jour avec la SEULE option "
                                      "qu'on vient de choisir.")


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"cooccurrences": 188, "regles_association": 189, "recommander": 190, "decouper_dans_le_temps": 191,
            "mesurer": 192, "diversifier": 193}

SERIE = Serie(
    code="31",
    titre="Proposer la bonne offre : cooccurrences, règles, recommandation, évaluation, diversité",
    module_eleve="defis.serie_31",
    module_corrige="defis.corrige_serie_31",
    defis=(
        Defi(188, "cooccurrences", "🟢", "—", verifier_cooccurrences),
        Defi(189, "regles_association", "🟡", "—", verifier_regles_association),
        Defi(190, "recommander", "🟡", "—", verifier_recommander),
        Defi(191, "decouper_dans_le_temps", "🟢", "—", verifier_decouper_dans_le_temps),
        Defi(192, "mesurer", "🔴", "—", verifier_mesurer),
        Defi(193, "diversifier", "🔴", "—", verifier_diversifier),
    ),
)

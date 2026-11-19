"""Vérifications automatiques de la série 44 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est la direction, qui ne recrutera que si vos calculs tiennent, et le standard téléphonique,
qui vous envoie ses exports tels qu'ils sont vraiment.

- il compare vos probabilités, vos effectifs, vos simulations et vos indicateurs à ceux d'une version de référence
  (à un milliardième près : les règles sont fixées pour qu'il n'y ait qu'une bonne réponse) ; pour le planning, où
  plusieurs réponses sont justes, il vérifie ses PROPRIÉTÉS (rien ne déborde, chaque créneau est couvert, le moins
  de vacations possible) ;
- quand votre résultat est faux, il rejoue les erreurs classiques (l'occupation prise pour une probabilité, Erlang B
  au lieu de C, une boucle qui s'arrête un conseiller trop tôt, des secondes prises pour des minutes, l'arrondi en
  float, l'appel abandonné qui garde son conseiller, le mauvais dénominateur, le centile interpolé, la vacation qui
  déborde, la file sans vieillissement, l'entrée fantôme d'un client qui rappelle…) pour nommer la vôtre ;
- il choisit des cas où chaque erreur se voit (1 000 conseillers, un trafic égal au nombre de conseillers, un
  conseiller libéré pile à l'arrivée d'un appel, une attente égale à la patience, des priorités à égalité), et
  compte les lignes exécutées dans votre fichier quand une solution naïve serait trop lente ;
- pendant le défi 267, il prête sa propre `erlang_c` à votre fichier : chaque défi est noté seul.
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
import sys

from .attente import (
    DUREE_MOYENNE_S,
    OBJECTIF,
    SEUIL_S,
    VACATION,
    appels_exemple,
    generer_appels,
    prevision_exemple,
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
    if isinstance(exc, OverflowError):
        return ("Un nombre a débordé : `trafic ** agents` et `math.factorial(agents)` sont immenses (950 ** 1000 n'existe "
                "pas en float, et 171! non plus), même si leur quotient est raisonnable. Calculez Erlang B par "
                "récurrence : B = 1, puis B = A × B / (k + A × B) pour k = 1 … agents. Chaque B reste entre 0 et 1.")
    if isinstance(exc, ZeroDivisionError):
        return ("Une division par zéro. Deux causes classiques : une récurrence qui commence à k = 0 (avec un trafic nul, "
                "A × B / (0 + A × B) vaut 0 / 0 : k va de 1 à agents), ou un trafic égal au nombre de conseillers, où "
                "n - A × (1 - B) peut s'annuler (traitez `trafic >= agents`, réponse 1.0, AVANT la formule). Sans "
                ")")
    if isinstance(exc, TypeError) and "not supported between" in texte:
        return ("Python ne sait pas comparer ces deux valeurs : dans un tas, à clé égale, `heapq` compare l'élément "
                "suivant du tuple. Mettez un COMPTEUR (`next(itertools.count())`) juste après la clé : deux entrées ne "
                "sont jamais égales jusqu'au client, et l'on ne compare jamais deux clients.")
    if isinstance(exc, IndexError):
        return "Un indice sort de la liste : vérifiez vos bornes (un rang commence à 1, un indice Python à 0)."
    return ""


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = None, conseil_lent: str = "",
           kwargs: dict | None = None, fichier: str | None = None, au_lieu: dict | None = None):
    """f(*args), en transformant toute erreur en message lisible (NotImplementedError : le défi est à faire).

    `au_lieu` : {type d'exception: conseil} propres à un défi, qui remplacent les conseils généraux.
    """
    kwargs = kwargs or {}
    try:
        return _trace((fichier or _fichier(f)) if lignes_max else None, lambda: f(*args, **kwargs), lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError. {conseil}".rstrip()) from None
    except Exception as exc:
        propre = next((texte for genre, texte in (au_lieu or {}).items() if isinstance(exc, genre)), "")
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {propre or _conseil_exception(exc) or conseil}".rstrip()) \
            from None


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu: type = ValueError, kwargs: dict | None = None):
    """Exige que f(*args) lève `attendu` (et rien d'autre). Renvoie l'exception, pour lire son message."""
    kwargs = kwargs or {}
    try:
        resultat = f(*args, **kwargs)
    except NotImplementedError:
        raise
    except attendu as exc:
        return exc
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _premier_bug(obtenus, rejouer, bugs: dict, egal) -> str:
    """Le conseil du premier bug dont le rejeu donne EXACTEMENT vos réponses."""
    for bug, conseil in bugs.items():
        try:
            faux = rejouer(bug)
        except Exception:
            continue
        if egal(faux, obtenus):
            return conseil
    return ""


def _nombre(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _proche(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return _nombre(a) and _nombre(b) and abs(a - b) <= tol * max(1.0, abs(b))


def _listes(egal):
    return lambda a, b: len(a) == len(b) and all(egal(x, y) for x, y in zip(a, b))


def _decrire_appel(nom: str, args: tuple, kwargs: dict | None = None, n: int = 60) -> str:
    morceaux = [_court(a, n) for a in args] + [f"{k}={v!r}" for k, v in (kwargs or {}).items()]
    return f"{nom}({', '.join(morceaux)})"


# ============================================================================ défi 266 : erlang_c


def _log_poisson(a: float, k: int) -> float:
    """log(A**k / k!), sans débordement (pour rejouer la formule « directe »)."""
    return k * math.log(a) - math.lgamma(k + 1) if a > 0 else (0.0 if k == 0 else -math.inf)


def _ref_erlang_c(a, n: int, bug: str = "") -> float:
    """La probabilité d'attendre ; `bug` rejoue une erreur classique."""
    if bug == "occupation":
        return min(1.0, a / n)
    if a >= n and bug != "instable":
        return 1.0
    if bug == "somme_jusqu_a_n":                 # la formule directe, mais la somme va jusqu'à n (au lieu de n - 1)
        termes = [_log_poisson(a, k) for k in range(n + 1)]
        dernier = _log_poisson(a, n) + math.log(n / (n - a))
        m = max(termes + [dernier])
        haut = math.exp(dernier - m)
        return haut / (sum(math.exp(t - m) for t in termes) + haut)
    b = 1.0
    for k in (range(1, n) if bug == "un_de_moins" else range(1, n + 1)):
        b = a * b / (k + a * b)
    if bug == "erlang_b":
        return b
    return n * b / (n - a * (1 - b))


_BUGS_ERLANG = {
    "occupation": "Vous renvoyez l'OCCUPATION (trafic / agents) : c'est la part du temps où un conseiller est au "
                  "téléphone, pas la probabilité qu'un appel attende. Avec 1 000 conseillers à 95 % d'occupation, moins "
                  "de 7 % des appels attendent. Erlang B par récurrence, puis C = n × B / (n - A × (1 - B)).",
    "erlang_b": "Vous renvoyez Erlang B (la probabilité qu'un appel soit REJETÉ quand il n'y a pas de file) : il faut "
                "ensuite Erlang C = n × B / (n - A × (1 - B)), la probabilité d'ATTENDRE (C est toujours plus grand "
                "que B).",
    "un_de_moins": "Votre récurrence s'arrête un conseiller trop tôt (`range(1, agents)` s'arrête à agents - 1) : vous "
                   "calculez Erlang B pour agents - 1 conseillers. Il faut k = 1, 2, …, agents : `range(1, agents + "
                   "1)`, en partant de B = 1.",
    "somme_jusqu_a_n": "Dans la formule directe, la somme du dénominateur va de k = 0 à n - 1 (pas jusqu'à n) : le "
                       "terme de rang n est déjà là, multiplié par n / (n - A). Plus simple et sans débordement : la "
                       "récurrence d'Erlang B, puis C = n × B / (n - A × (1 - B)).",
    "instable": "Quand le trafic atteint ou dépasse le nombre de conseillers (A >= n), la file grossit sans fin : la "
                "réponse est 1.0 (tout le monde attend). La formule, elle, donne alors un nombre négatif ou plus "
                "grand que 1 : traitez ce cas AVANT.",
}


def _cas_266() -> list[tuple]:
    cas = [(2, 3), (12, 14), (11.1, 14), (14, 14), (18, 14), (0, 1), (0, 5), (0.5, 1), (1, 1), (1, 2), (3, 4),
           (9.99, 10), (4.5, 5), (950, 1000), (950.5, 1000), (180.25, 200), (16.5, 16), (13, 14), (0.001, 1),
           (99.9, 100), (250, 300), (20, 15)]
    alea = random.Random(266)
    for _ in range(30):
        n = alea.randint(1, 60)
        cas.append((round(alea.uniform(0, n * 1.05), 3), n))
    return cas


def verifier_erlang_c(f) -> None:
    cas = _cas_266()
    obtenus = []
    for a, n in cas:
        quoi = f"erlang_c({a!r}, {n!r})"
        r = _appel(f, (a, n), quoi)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not _nombre(r):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) : attendu une probabilité (un "
                                 "float entre 0 et 1).")
        obtenus.append(r)
    for (a, n), r in zip(cas, obtenus):
        attendu = _ref_erlang_c(a, n)
        if not _proche(r, attendu):
            conseil = _premier_bug(obtenus, lambda bug: [_ref_erlang_c(x, m, bug) for x, m in cas], _BUGS_ERLANG,
                                   _listes(_proche))
            if not conseil and not 0 <= r <= 1:
                conseil = ("Une probabilité est entre 0 et 1." + (" Quand le trafic atteint le nombre de conseillers, la "
                                                                  "réponse est 1.0." if a >= n else ""))
            raise AssertionError(f"erlang_c({a!r}, {n!r}) vaut {r!r} au lieu de {attendu!r}. {conseil}".rstrip())

    for args, attendu, conseil in (
            ((-1, 3), ValueError, "Un trafic négatif n'existe pas."),
            ((float("nan"), 3), ValueError, "NaN n'est pas un trafic (et `nan >= 3` est FAUX : testez-le à part)."),
            ((float("inf"), 3), ValueError, "Un trafic infini n'est pas un trafic : ValueError, pas 1.0."),
            (("12", 14), TypeError, "« 12 » est un texte, pas un trafic."),
            ((True, 14), TypeError, "True n'est pas un trafic (`isinstance(True, int)` est vrai : excluez bool)."),
            ((None, 14), TypeError, "None n'est pas un trafic."),
            ((12, 14.0), TypeError, "14.0 conseillers : un nombre de conseillers est un int."),
            ((12, True), TypeError, "True n'est pas un nombre de conseillers."),
            ((12, "14"), TypeError, "« 14 » est un texte."),
            ((0, 0), ValueError, "Zéro conseiller : personne ne répond, la question n'a pas de sens."),
            ((12, -3), ValueError, "Un nombre de conseillers négatif n'existe pas.")):
        _doit_lever(f, args, f"erlang_c{_court(args, 40)}", conseil, attendu)


# ============================================================================ défi 267 : conseillers_necessaires


def _c_depuis_b(a: float, n: int, b: float) -> float:
    return 1.0 if a >= n else n * b / (n - a * (1 - b))


def _ref_conseillers(cph, aht, objectif=OBJECTIF, seuil=SEUIL_S, pct: int = 0, bug: str = "") -> int:
    """Le nombre de conseillers à planifier ; `bug` rejoue une erreur classique. Erlang B est calculé de proche en
    proche (B(n) à partir de B(n - 1)) : même un trafic mal converti (43 200 erlangs) se rejoue vite."""
    if cph == 0:
        return 0
    if bug == "minutes":
        a = cph * aht / 60
    elif bug == "heures":
        a = cph * aht
    else:
        a = cph * aht / 3600
    if bug == "occupation":
        n = math.ceil(a / 0.85)
    else:
        n = math.floor(a) + 1
        b = 1.0
        for k in range(1, n + 1):
            b = a * b / (k + a * b)
        while True:
            c = _c_depuis_b(a, n, b)
            if bug == "sans_attente":
                sl = 1 - c
            elif bug == "sans_moins_a":
                sl = 1 - c * math.exp(-n * seuil / aht)
            else:
                sl = 1 - c * math.exp(-(n - a) * seuil / aht)
            if sl >= objectif:
                break
            n += 1
            b = a * b / (n + a * b)
    if bug == "absents_plus":
        return -(-n * (100 + pct) // 100)
    if bug == "absents_float":
        return math.ceil(n / (1 - pct / 100))
    if bug == "absents_round":
        return round(n / (1 - pct / 100))
    return -(-n * 100 // (100 - pct))


_BUGS_CONSEILLERS = {
    "minutes": "Votre trafic est 60 fois trop grand : la durée est en SECONDES, et une heure en compte 3 600 (pas 60). "
               "A = appels_par_heure × duree_moyenne_s / 3600.",
    "heures": "Votre trafic n'est pas en erlangs : appels_par_heure × duree_moyenne_s compte des SECONDES de "
              "conversation par heure. Divisez par 3 600 : A = appels_par_heure × duree_moyenne_s / 3600.",
    "occupation": "Vous appliquez la règle du tableur (« 85 % d'occupation » : ⌈A / 0,85⌉) : ce n'est pas un objectif "
                  "de service. Avec 12 erlangs, elle donne 15 conseillers, et 73 % des appels décrochés en 20 s. "
                  "Cherchez le plus petit n > A dont le taux de service atteint l'objectif.",
    "sans_attente": "Votre taux de service est 1 - C : la part des appels décrochés SANS attendre. Un appel décroché en "
                    "12 secondes compte aussi : 1 - C × exp(-(n - A) × seuil_s / duree_moyenne_s).",
    "sans_moins_a": "Dans l'exponentielle, c'est (n - A) : exp(-(n - A) × seuil_s / duree_moyenne_s). Avec n seul, "
                    "vous comptez comme si tous les conseillers étaient libres, et sous-estimez l'effectif.",
    "absents_plus": "Avec 30 % d'absents, il ne faut pas n × 1,3 conseillers, mais n / 0,7 : on cherche m tel que 70 % "
                    "de m fassent n. 16 × 1,3 = 20,8 → 21 conseillers, dont 70 % font 14,7 : pas 16. La réponse : "
                    "⌈100 × n / (100 - absents_pct)⌉.",
    "absents_float": "Votre division se fait en float : 21 / (1 - 0.3) vaut 30.000000000000004, et `math.ceil` en "
                     "fait 31, alors que 30 conseillers dont 70 % sont présents font exactement 21. Calculez en "
                     "ENTIERS : `-(-100 * n // (100 - absents_pct))`.",
    "absents_round": "Vous ARRONDISSEZ au plus proche : 16 / 0,7 = 22,86 → 23, mais 17 / 0,8 = 21,25 → 21, et 21 × 0,8 = "
                     "16,8 présents : il en manque. Arrondissez vers le HAUT, en entiers : `-(-100 * n // (100 - "
                     "absents_pct))`.",
}


def _cas_267() -> list[tuple]:
    """(appels_par_heure, duree_moyenne_s, kwargs)."""
    cas = [(120, 360, {}), (180, 360, {}), (120, 360, {"absents_pct": 30}), (160, 360, {"absents_pct": 30}),
           (0, 360, {}), (0, 360, {"absents_pct": 30}), (12, 360, {}), (1, 30, {}), (60, 240, {"objectif": 0.9, "seuil_s": 15}),
           (300, 300, {"absents_pct": 25}), (1000, 180, {"objectif": 0.85, "seuil_s": 30, "absents_pct": 10}),
           (120, 360, {"seuil_s": 0}), (120.5, 359.5, {"objectif": 0.95}), (2500, 420, {"absents_pct": 35}),
           (40, 600, {"objectif": 0.5, "seuil_s": 60}), (120, 360, {"absents_pct": 20}), (150, 360, {"absents_pct": 20}),
           (90, 300, {"absents_pct": 45}), (45, 1200, {"objectif": 0.7, "seuil_s": 120, "absents_pct": 5})]
    cas += [(n, DUREE_MOYENNE_S, {}) for _, n in prevision_exemple()]
    # des cas où la division en float, puis math.ceil, se trompe d'un conseiller
    pieges = 0
    for appels in range(10, 400, 7):
        for pct in (10, 20, 30, 35, 45, 55, 68):
            n = _ref_conseillers(appels, 360)
            if math.ceil(n / (1 - pct / 100)) != -(-n * 100 // (100 - pct)) and pieges < 8:
                cas.append((appels, 360, {"absents_pct": pct}))
                pieges += 1
    alea = random.Random(267)
    for _ in range(20):
        cas.append((alea.choice([alea.randint(1, 400), round(alea.uniform(1, 400), 2)]), alea.choice([180, 240, 300, 360, 420]),
                    {"objectif": alea.choice([0.7, 0.8, 0.9]), "seuil_s": alea.choice([10, 20, 30]),
                     "absents_pct": alea.choice([0, 0, 15, 25, 30])}))
    return cas


class _ErlangPretee:
    """Remplace `erlang_c` dans le module de `f` le temps d'une vérification, puis remet la vôtre."""

    def __init__(self, f) -> None:
        self.module = sys.modules.get(getattr(f, "__module__", None) or "")
        self.avant = None

    def __enter__(self):
        if self.module is not None and hasattr(self.module, "erlang_c"):
            self.avant = self.module.erlang_c
            self.module.erlang_c = _erlang_pretee
        return self

    def __exit__(self, *exc) -> None:
        if self.avant is not None:
            self.module.erlang_c = self.avant


_DERNIER_B = {"trafic": None, "n": 0, "b": 1.0}


def _erlang_pretee(trafic, agents) -> float:
    """La `erlang_c` que le correcteur prête à votre fichier pendant le défi 267 (elle vérifie ses arguments).

    Elle reprend Erlang B là où elle l'avait laissé quand on lui demande le même trafic avec plus de conseillers
    (n, puis n + 1…) : même un trafic mal converti (un million d'« erlangs ») reste rapide à essayer.
    """
    if isinstance(trafic, bool) or not isinstance(trafic, (int, float)) or isinstance(agents, bool) \
            or not isinstance(agents, int):
        raise TypeError("erlang_c(trafic, agents) : un nombre et un int")
    if not math.isfinite(trafic) or trafic < 0 or agents < 1:
        raise ValueError("erlang_c(trafic, agents) : trafic >= 0 fini, agents >= 1")
    if trafic >= agents:
        return 1.0
    memo = _DERNIER_B
    if memo["trafic"] != trafic or memo["n"] > agents:
        memo.update(trafic=trafic, n=0, b=1.0)
    b = memo["b"]
    for k in range(memo["n"] + 1, agents + 1):
        b = trafic * b / (k + trafic * b)
    memo.update(n=agents, b=b)
    return agents * b / (agents - trafic * (1 - b))


def verifier_conseillers_necessaires(f) -> None:
    with _ErlangPretee(f):
        _verifier_conseillers(f)


def _verifier_conseillers(f) -> None:
    cas = _cas_267()
    obtenus = []
    for appels, duree, kwargs in cas:
        quoi = _decrire_appel("conseillers_necessaires", (appels, duree), kwargs)
        r = _appel(f, (appels, duree), quoi, kwargs=kwargs, lignes_max=2_000_000,
                   conseil_lent="Cherchez n à partir de ⌊A⌋ + 1 (pas de 1), et vérifiez que A est bien en erlangs "
                                "(appels par heure × durée en secondes / 3 600).")
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if isinstance(r, bool) or not isinstance(r, int):
            conseil = " (math.ceil et // donnent des int ; une division / donne un float)" if isinstance(r, float) else ""
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) : attendu un nombre de "
                                 f"conseillers, un int{conseil}.")
        obtenus.append(r)
    for (appels, duree, kwargs), r in zip(cas, obtenus):
        attendu = _ref_conseillers(appels, duree, kwargs.get("objectif", OBJECTIF), kwargs.get("seuil_s", SEUIL_S),
                                   kwargs.get("absents_pct", 0))
        if r != attendu:
            quoi = _decrire_appel("conseillers_necessaires", (appels, duree), kwargs)
            if appels == 0:
                raise AssertionError(f"{quoi} renvoie {r!r} : zéro appel, zéro conseiller (0).")

            def rejouer(bug):
                return [_ref_conseillers(x, d, k.get("objectif", OBJECTIF), k.get("seuil_s", SEUIL_S),
                                         k.get("absents_pct", 0), bug) for x, d, k in cas]

            conseil = _premier_bug(obtenus, rejouer, _BUGS_CONSEILLERS, lambda a, b: a == b)
            if not conseil:
                sans = _ref_conseillers(appels, duree, kwargs.get("objectif", OBJECTIF), kwargs.get("seuil_s", SEUIL_S))
                if r == sans - 1:
                    conseil = ("Un conseiller de moins que la réponse : gardez le PREMIER n dont le taux de service est "
                               ">= objectif (vérifiez la condition de votre boucle, et que n > A).")
                elif r == attendu + 1 and kwargs.get("absents_pct"):
                    conseil = _BUGS_CONSEILLERS["absents_float"]
            raise AssertionError(f"{quoi} renvoie {r!r} au lieu de {attendu!r}. {conseil}".rstrip())

    for args, kwargs, attendu, conseil in (
            ((-1, 360), {}, ValueError, "Un nombre d'appels négatif n'existe pas."),
            (("120", 360), {}, TypeError, "« 120 » est un texte."),
            ((True, 360), {}, TypeError, "True n'est pas un nombre d'appels."),
            ((float("nan"), 360), {}, ValueError, "NaN n'est pas un nombre d'appels."),
            ((120, 0), {}, ValueError, "Une conversation de 0 seconde : le trafic n'a pas de sens (durée > 0)."),
            ((0, 0), {}, ValueError, "Même sans appel, une durée de 0 seconde est une erreur : vérifiez TOUS les "
                                     "arguments avant de répondre 0."),
            ((120, -360), {}, ValueError, "Une durée négative n'existe pas."),
            ((120, float("inf")), {}, ValueError, "Une durée infinie n'existe pas."),
            ((120, 360), {"objectif": 80}, ValueError, "L'objectif est une PART (0,8), pas un pourcentage (80)."),
            ((120, 360), {"objectif": 1}, ValueError, "Un objectif de 100 % n'est jamais atteint : 0 < objectif < 1."),
            ((120, 360), {"objectif": 0}, ValueError, "Un objectif de 0 n'en est pas un : 0 < objectif < 1."),
            ((120, 360), {"objectif": "0.8"}, TypeError, "« 0.8 » est un texte."),
            ((120, 360), {"seuil_s": -20}, ValueError, "Un seuil négatif n'existe pas."),
            ((120, 360), {"absents_pct": 0.3}, TypeError, "absents_pct est un pourcentage ENTIER : 30, pas 0.3."),
            ((120, 360), {"absents_pct": 30.0}, TypeError, "absents_pct est un int : 30, pas 30.0."),
            ((120, 360), {"absents_pct": True}, TypeError, "True n'est pas un pourcentage."),
            ((120, 360), {"absents_pct": 100}, ValueError, "Avec 100 % d'absents, personne ne répond : de 0 à 99."),
            ((120, 360), {"absents_pct": -5}, ValueError, "Un pourcentage négatif d'absents n'existe pas.")):
        _doit_lever(f, args, _decrire_appel("conseillers_necessaires", args, kwargs), conseil, attendu, kwargs=kwargs)


# ============================================================================ défi 268 : simuler_file


def _ref_simuler(appels, conseillers: int, bug: str = "") -> list[tuple[float, bool]]:
    """La file « premier arrivé, premier servi », avec abandons ; `bug` rejoue une erreur classique."""
    if bug == "dans_l_ordre":
        ordre = list(range(len(appels)))
    else:
        ordre = sorted(range(len(appels)), key=lambda i: appels[i][0])
    libres = [0.0] * conseillers
    tour = itertools.count()
    resultats: list = [None] * len(appels)
    for i in ordre:
        arrivee, duree, patience = appels[i]
        if bug == "tourniquet":                       # l'appel i va au conseiller i % n, libre ou pas
            k = next(tour) % conseillers
            debut = max(arrivee, libres[k])
        else:
            debut = max(arrivee, libres[0])
        attente = debut - arrivee
        if bug == "sans_abandon":
            abandon = False
        elif bug == "abandon_large":
            abandon = attente >= patience
        else:
            abandon = attente > patience
        if abandon:
            resultats[i] = (float(attente if bug == "attente_reelle" else patience), True)
            if bug != "abandon_occupe":
                continue
        else:
            resultats[i] = (float(attente), False)
        fin = arrivee + duree if bug == "fin_sans_attente" else debut + duree
        if bug == "tourniquet":
            libres[k] = fin
        else:
            heapq.heapreplace(libres, fin)
    return resultats


_BUGS_SIMULATION = {
    "dans_l_ordre": "Vous servez les appels dans l'ordre de la LISTE : l'export du standard est rangé par ligne "
                    "téléphonique, pas par heure d'arrivée. Premier arrivé, premier servi : parcourez les indices triés "
                    "par arrivée (`sorted(range(len(appels)), key=…)`), et rangez chaque résultat à SA place.",
    "abandon_occupe": "Un client qui raccroche ne prend AUCUN conseiller : il n'a jamais été décroché. Ne touchez pas "
                      "au tas des conseillers quand il abandonne ; le suivant de la file est décroché au même instant.",
    "attente_reelle": "Pour un abandon, l'attente est la PATIENCE du client (il a raccroché au bout de ce temps), pas "
                      "l'attente qu'il aurait eue s'il était resté.",
    "abandon_large": "Un client dont l'attente vaut EXACTEMENT sa patience est décroché juste à temps : il raccroche "
                     "seulement si l'attente dépasse strictement sa patience (`attente > patience`).",
    "sans_abandon": "Vos clients ne raccrochent jamais : si l'attente dépasse la patience, le couple vaut (patience, "
                    "True), et le conseiller reste libre pour le suivant.",
    "fin_sans_attente": "Le conseiller est occupé jusqu'à DÉBUT + durée, pas arrivée + durée : un appel décroché après "
                        "80 s d'attente libère son conseiller 80 s plus tard.",
    "tourniquet": "Vous donnez les appels aux conseillers à tour de rôle (le i-ème au conseiller i % n), même s'il est "
                  "occupé alors qu'un autre est libre. Un appel va au conseiller qui se libère le PREMIER : le plus "
                  "petit élément d'un tas (`libres[0]`).",
}


def _cas_268() -> list[tuple]:
    inf = math.inf
    cas = [
        ([(0, 100, 50), (0, 100, 50), (10, 100, 30), (20, 100, 200)], 1),
        ([(0, 100, 50), (0, 100, 50), (10, 100, 30), (20, 100, 200)], 2),
        ([(20, 100, 200), (10, 100, 30), (0, 100, 50), (0, 100, 50)], 1),      # dans le désordre
        ([(0, 100, inf), (100, 50, 0)], 1),                                    # libéré pile à l'arrivée : 0 s
        ([(0, 100, inf), (40, 10, 60)], 1),                                    # attente == patience : décroché
        ([(0, 100, inf), (40, 10, 59.5)], 1),                                  # attente > patience : abandon
        ([(5, 30, inf), (5, 10, inf), (5, 20, inf)], 1),                       # même arrivée : l'ordre de la liste
        ([(0, 0, 0), (0, 0, 0), (0, 5, 0)], 1),                                # durées nulles
        ([(0.5, 10.25, 3.5), (1.0, 4.0, 20.0), (2.75, 1.5, 0.25), (3.0, 2.0, inf)], 2),
        ([], 3),
        ([(0, 300, inf), (10, 300, inf), (20, 300, 15), (30, 50, 400), (40, 60, 400)], 2),
        ((((0, 60, 10), (5, 60, 10), (6, 60, 100)), 1)),                       # un tuple de tuples
        ([[0, 60, 10], [5, 60, 70], [6, 60, 100]], 1),                         # une liste de listes
    ]
    cas.append((appels_exemple(), 14))
    cas.append((appels_exemple(), 16))
    for graine, par_heure, n in ((2681, 50, 4), (2682, 120, 10), (2683, 300, 31), (2684, 30, 1)):
        cas.append((generer_appels(par_heure, graine=graine), n))
    alea = random.Random(268)
    for _ in range(8):
        appels = [(alea.randint(0, 600), alea.randint(1, 120), alea.choice([inf, alea.randint(0, 90)]))
                  for _ in range(alea.randint(5, 40))]
        cas.append((appels, alea.randint(1, 4)))
    return cas


def _egal_simulation(a, b) -> bool:
    return len(a) == len(b) and all(_proche(x[0], y[0]) and x[1] == y[1] for x, y in zip(a, b))


def _verifier_une_simulation(f, appels, conseillers: int, lignes_max: int | None = None, conseil_lent: str = "") -> list:
    quoi = f"simuler_file({_court(appels, 90)}, {conseillers})"
    copie = repr(appels)
    r = _appel(f, (appels, conseillers), quoi, lignes_max=lignes_max, conseil_lent=conseil_lent)
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) : attendu une LISTE de couples.")
    if repr(appels) != copie:
        raise AssertionError(f"{quoi} a modifié la liste `appels` reçue : triez une COPIE (ou des indices).")
    if len(r) != len(appels):
        raise AssertionError(f"{quoi} renvoie {len(r)} couples pour {len(appels)} appels : un couple par appel, "
                             "abandons compris.")
    for k, couple in enumerate(r):
        if not isinstance(couple, tuple) or len(couple) != 2:
            raise AssertionError(f"{quoi} : l'élément {k} vaut {_court(couple, 60)} : attendu un TUPLE (attente, "
                                 "abandonné).")
        if not _nombre(couple[0]) or not isinstance(couple[1], bool):
            raise AssertionError(f"{quoi} : l'élément {k} vaut {_court(couple, 60)} : l'attente est un nombre (float) "
                                 "et « abandonné » un bool (True ou False).")
    return r


def verifier_simuler_file(f) -> None:
    cas = _cas_268()
    for appels, conseillers in cas:
        r = _verifier_une_simulation(f, appels, conseillers)
        attendu = _ref_simuler(appels, conseillers)
        if not _egal_simulation(r, attendu):
            k = next(i for i, (x, y) in enumerate(zip(r, attendu)) if not _egal_simulation([x], [y]))
            conseil = _premier_bug(r, lambda bug: _ref_simuler(appels, conseillers, bug), _BUGS_SIMULATION,
                                   _egal_simulation)
            raise AssertionError(f"simuler_file({_court(appels, 90)}, {conseillers}) : l'appel {k} "
                                 f"{_court(tuple(appels[k]), 40)} donne {r[k]!r} au lieu de {attendu[k]!r}. "
                                 f"{conseil}".rstrip())

    gros = generer_appels(2500, graine=2689)          # environ 30 000 appels, 250 erlangs
    conseil = ("Le correcteur compte les lignes exécutées dans votre fichier : chercher le premier conseiller libre en "
               "parcourant les 400 conseillers à chaque appel, c'est 12 millions d'étapes (et avancer seconde par "
               "seconde, bien plus). Gardez un TAS des instants de libération : `libres[0]` est le premier libre, "
               "`heapq.heapreplace(libres, fin)` le remplace en une quinzaine d'étapes.")
    r = _verifier_une_simulation(f, gros, 400, lignes_max=40 * len(gros), conseil_lent=conseil)
    if not _egal_simulation(r, _ref_simuler(gros, 400)):
        raise AssertionError(f"simuler_file(<{len(gros)} appels>, 400) donne de mauvais résultats (alors que les petits "
                             "cas passent) : les abandons, les égalités d'arrivée ?")
    r = _verifier_une_simulation(f, gros, 240, lignes_max=40 * len(gros), conseil_lent=conseil)
    if not _egal_simulation(r, _ref_simuler(gros, 240)):
        raise AssertionError(f"simuler_file(<{len(gros)} appels>, 240) donne de mauvais résultats.")

    for args, attendu, conseil in (
            (({"a": (0, 10, 5)}, 1), TypeError, "`appels` est une liste (ou un tuple) de triplets."),
            (([(0, 10)], 1), TypeError, "Un appel est un TRIPLET (arrivée, durée, patience)."),
            (([(0, 10, 5, 1)], 1), TypeError, "Un appel a trois nombres, pas quatre."),
            (([5], 1), TypeError, "5 n'est pas un appel (arrivée, durée, patience)."),
            (([("0", 10, 5)], 1), TypeError, "« 0 » est un texte, pas une heure d'arrivée."),
            (([(0, True, 5)], 1), TypeError, "True n'est pas une durée."),
            (([(0, 10, None)], 1), TypeError, "None n'est pas une patience (math.inf : il ne raccroche jamais)."),
            (([(-1, 10, 5)], 1), ValueError, "Une arrivée négative n'existe pas."),
            (([(0, -10, 5)], 1), ValueError, "Une durée négative n'existe pas."),
            (([(0, 10, -5)], 1), ValueError, "Une patience négative n'existe pas."),
            (([(0, float("nan"), 5)], 1), ValueError, "NaN n'est pas une durée."),
            (([(0, 10, float("nan"))], 1), ValueError, "NaN n'est pas une patience (et `attente > nan` est toujours FAUX)."),
            (([(float("inf"), 10, 5)], 1), ValueError, "Une arrivée infinie n'existe pas (seule la patience peut valoir "
                                                       "math.inf)."),
            (([(0, float("inf"), 5)], 1), ValueError, "Une conversation infinie n'existe pas."),
            (([(0, 10, 5), (3, 10, 5), (1, 10, -1)], 2), ValueError, "Le TROISIÈME appel a une patience négative : "
                                                                     "vérifiez tous les appels."),
            (([(0, 10, 5)], 0), ValueError, "Zéro conseiller : personne ne décroche."),
            (([(0, 10, 5)], 2.0), TypeError, "Un nombre de conseillers est un int (2, pas 2.0)."),
            (([(0, 10, 5)], True), TypeError, "True n'est pas un nombre de conseillers."),
            (([], 0), ValueError, "Même sans appel, zéro conseiller est une erreur : vérifiez `conseillers` aussi.")):
        _doit_lever(f, args, f"simuler_file{_court(args, 60)}", conseil, attendu)


# ============================================================================ défi 269 : indicateurs


_CLES = ("appels", "decroches", "abandons", "taux_service", "taux_abandon", "asa", "p90")


def _ref_indicateurs(resultats, seuil=SEUIL_S, bug: str = "") -> dict:
    attentes = [a for a, ab in resultats if not ab]
    abandons = [a for a, ab in resultats if ab]
    n = len(resultats)
    if bug == "strict":
        a_temps = sum(1 for a in attentes if a < seuil)
    else:
        a_temps = sum(1 for a in attentes if a <= seuil)
    if bug == "service_sur_tous":
        juges = n
    elif bug == "service_sur_decroches":
        juges = len(attentes)
    else:
        juges = len(attentes) + sum(1 for a in abandons if a > seuil)
    if bug == "asa_avec_abandons":
        asa = sum(a for a, _ in resultats) / n if n else None
    elif bug == "asa_sur_tous":
        asa = sum(attentes) / n if n else None
    else:
        asa = sum(attentes) / len(attentes) if attentes else None
    rangees = sorted(a for a, _ in resultats) if bug == "p90_tous" else sorted(attentes)
    k = len(rangees)
    if not k:
        p90 = None
    elif bug == "p90_interpole":
        position = 0.9 * (k - 1)
        bas = math.floor(position)
        haut = min(bas + 1, k - 1)
        p90 = rangees[bas] + (rangees[haut] - rangees[bas]) * (position - bas)
    elif bug == "p90_rang":
        p90 = rangees[min(int(0.9 * k), k - 1)]
    else:
        p90 = rangees[math.ceil(0.9 * k) - 1]
    if bug == "abandon_sur_decroches":
        taux_abandon = len(abandons) / len(attentes) if attentes else None
    else:
        taux_abandon = len(abandons) / n if n else None
    return {"appels": n, "decroches": len(attentes), "abandons": len(abandons),
            "taux_service": a_temps / juges if juges else None, "taux_abandon": taux_abandon, "asa": asa, "p90": p90}


_BUGS_INDICATEURS = {
    "service_sur_tous": "Votre taux de service divise par TOUS les appels : un client qui raccroche après 5 s n'a pas "
                        "laissé au centre le temps de répondre, il sort du calcul. Dénominateur : décrochés + abandons "
                        "APRÈS le seuil (attente > seuil_s).",
    "service_sur_decroches": "Votre taux de service ne divise que par les DÉCROCHÉS : un client qui raccroche après "
                             "60 s d'attente est un échec, il compte au dénominateur. Dénominateur : décrochés + "
                             "abandons après le seuil (attente > seuil_s).",
    "strict": "« Décroché en 20 secondes » inclut 20 secondes pile : `attente <= seuil_s` (et un abandon à 20 s pile "
              "sort du calcul : il n'est pas « après » le seuil).",
    "asa_avec_abandons": "Votre ASA (temps moyen avant décroché) compte les abandons : un client qui a raccroché n'a "
                         "jamais été décroché. Moyenne des attentes des seuls DÉCROCHÉS.",
    "asa_sur_tous": "Votre ASA additionne les attentes des décrochés, mais divise par TOUS les appels : divisez par le "
                    "nombre de décrochés.",
    "p90_interpole": "Votre 90e centile est INTERPOLÉ entre deux attentes (ce que font `numpy.percentile` et "
                     "`statistics.quantiles`) : la règle prend le rang le plus proche, une attente vraie : "
                     "`rangees[math.ceil(0.9 * k) - 1]`.",
    "p90_rang": "Votre 90e centile est pris un rang trop loin : `rangees[int(0.9 * k)]` saute une attente (avec 10 "
                "décrochés, il prend la 10e, la plus longue, au lieu de la 9e). Le rang est ⌈0,9 × k⌉, l'indice "
                "Python ⌈0,9 × k⌉ - 1.",
    "p90_tous": "Votre 90e centile compte les abandons : c'est celui des attentes des DÉCROCHÉS.",
    "abandon_sur_decroches": "Votre taux d'abandon divise par les décrochés : c'est abandons / TOUS les appels.",
}


def _cas_269() -> list[tuple]:
    cas = [
        ([(0, False), (30, False), (5, True), (60, True)], 20),
        ([], 20),
        ([(5, True)], 20),                                   # un seul abandon rapide : taux de service None
        ([(30, True)], 20),                                  # un seul abandon tardif : 0 %
        ([(20, False), (20, True), (21, True)], 20),          # 20 s pile
        ([(float(k), False) for k in range(10)], 20),        # 10 décrochés : 9e attente
        ([(float(k), False) for k in range(1, 21)] + [(100.0, True), (2.0, True)], 5),
        ([(0, False)] * 7 + [(45.5, False), (12.25, False), (300, True)], 30),
        ([(3.5, False)], 0),
        ((((12, False), (8, False), (40, True))), 10),
        ([[12, False], [8, False]], 10),
    ]
    cas.append((_ref_simuler(appels_exemple(), 14), 20))
    cas.append((_ref_simuler(appels_exemple(), 16), 30))
    alea = random.Random(269)
    for _ in range(12):
        k = alea.randint(1, 60)
        cas.append(([(alea.choice([0, alea.randint(0, 120), round(alea.uniform(0, 90), 2)]), alea.random() < 0.25)
                     for _ in range(k)], alea.choice([10, 15, 20, 30])))
    return cas


def _egal_indicateurs(a, b) -> bool:
    return isinstance(a, dict) and set(a) == set(b) and all(_proche(a[c], b[c]) for c in b)


def verifier_indicateurs(f) -> None:
    cas = _cas_269()
    obtenus = []
    for resultats, seuil in cas:
        quoi = f"indicateurs({_court(resultats, 90)}, {seuil!r})"
        copie = repr(resultats)
        r = _appel(f, (resultats, seuil), quoi, au_lieu={
            ZeroDivisionError: "Une division par zéro : sans décroché (ou sans appel, ou sans appel « jugé »), une "
                               "moyenne ou un taux n'existe pas. Renvoyez None au lieu de diviser par 0.",
            IndexError: "Un indice sort de la liste : sans décroché, il n'y a pas de 90e centile (None) ; avec k "
                        "décrochés, le rang ⌈0,9 × k⌉ va de 1 à k, l'indice Python de 0 à k - 1."})
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, dict):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) : attendu un dict.")
        if set(r) != set(_CLES):
            manque, trop = [c for c in _CLES if c not in r], [c for c in r if c not in _CLES]
            raise AssertionError(f"{quoi} : le dict doit avoir EXACTEMENT les clés {list(_CLES)}"
                                 + (f" ; il manque {manque}" if manque else "") + (f" ; en trop : {trop}" if trop else "")
                                 + ".")
        if repr(resultats) != copie:
            raise AssertionError(f"{quoi} a modifié la liste reçue : triez une copie (`sorted`), pas l'original.")
        for c in ("appels", "decroches", "abandons"):
            if isinstance(r[c], bool) or not isinstance(r[c], int):
                raise AssertionError(f"{quoi} : r[{c!r}] vaut {r[c]!r} : un compte est un int.")
        obtenus.append(r)
    for (resultats, seuil), r in zip(cas, obtenus):
        attendu = _ref_indicateurs(resultats, seuil)
        if _egal_indicateurs(r, attendu):
            continue
        cle = next(c for c in _CLES if not _proche(r[c], attendu[c]))
        quoi = f"indicateurs({_court(resultats, 90)}, {seuil!r})"
        if attendu[cle] is None:
            raise AssertionError(f"{quoi} : r[{cle!r}] vaut {r[cle]!r} au lieu de None : il n'y a rien à mesurer (un "
                                 "dénominateur nul). None dit « pas de mesure » ; 0 dirait « 0 % », ce qui est faux.")
        conseil = _premier_bug(obtenus, lambda bug: [_ref_indicateurs(x, s, bug) for x, s in cas], _BUGS_INDICATEURS,
                               _listes(_egal_indicateurs))
        if not conseil:   # le bug n'est peut-être que sur CETTE clé : on rejoue clé par clé
            for bug, texte in _BUGS_INDICATEURS.items():
                faux = [_ref_indicateurs(x, s, bug)[cle] for x, s in cas]
                if all(_proche(o[cle], v) for o, v in zip(obtenus, faux)):
                    conseil = texte
                    break
        raise AssertionError(f"{quoi} : r[{cle!r}] vaut {r[cle]!r} au lieu de {attendu[cle]!r}. {conseil}".rstrip())

    for args, attendu, conseil in (
            (({"a": (0, False)}, 20), TypeError, "`resultats` est une liste de couples (attente, abandonné)."),
            (([(0, False, 3)], 20), TypeError, "Un résultat est un COUPLE (attente, abandonné)."),
            (([5], 20), TypeError, "5 n'est pas un couple (attente, abandonné)."),
            (([(5, 1)], 20), TypeError, "« abandonné » vaut 1 : c'est True ou False (un bool), pas 0 ou 1."),
            (([(5, "non")], 20), TypeError, "« non » n'est pas un bool."),
            (([("5", False)], 20), TypeError, "« 5 » est un texte, pas une attente."),
            (([(True, False)], 20), TypeError, "True n'est pas une attente."),
            (([(-1, False)], 20), ValueError, "Une attente négative n'existe pas."),
            (([(float("nan"), False)], 20), ValueError, "NaN n'est pas une attente."),
            (([(float("inf"), True)], 20), ValueError, "Une attente infinie n'existe pas (un abandon a attendu sa "
                                                       "patience, un nombre fini)."),
            (([(5, False)], -1), ValueError, "Un seuil négatif n'existe pas."),
            (([(5, False)], "20"), TypeError, "« 20 » est un texte."),
            (([], -1), ValueError, "Même sans résultat, un seuil négatif est une erreur.")):
        _doit_lever(f, args, f"indicateurs{_court(args, 60)}", conseil, attendu)


# ============================================================================ défi 270 : planifier_vacations


def _ref_planifier(besoins, duree: int, bug: str = "") -> list[int]:
    """Le glouton « le plus tard possible » ; `bug` rejoue une erreur classique."""
    total = len(besoins)
    debuts, presents = [0] * total, [0] * total
    if bug == "pic":                                 # le pic de la journée, à chaque vague
        pic = max(besoins, default=0)
        for s in sorted({min(s, total - duree) for s in range(0, total, duree)}):
            debuts[s] = pic
        return debuts
    for t in range(total):
        if bug == "fins_oubliees":                   # les vacations commencées un jour restent comptées
            manque = besoins[t] - sum(debuts[: t + 1])
        else:
            manque = besoins[t] - presents[t]
        if manque <= 0:
            continue
        if bug == "au_plus_tot":
            debut = max(0, t - duree + 1)
        elif bug == "deborde":
            debut = t
        else:
            debut = min(t, total - duree)
        debuts[debut] += manque
        for u in range(debut, min(debut + duree, total)):
            presents[u] += manque
    return debuts


_BUGS_PLANNING = {
    "deborde": "Des vacations commencent trop tard et finissent après la fermeture. Au créneau t, si une vacation "
               "commencée en t déborderait, faites-la commencer au dernier créneau possible : `min(t, len(besoins) - "
               "duree)`.",
    "au_plus_tot": "Vous faites commencer les vacations manquantes le plus TÔT possible (t - duree + 1) : elles couvrent "
                   "des créneaux déjà couverts, et s'arrêtent avant ceux qui viennent. Faites-les commencer le plus "
                   "TARD possible tout en couvrant t : au créneau t (ou len(besoins) - duree en fin de journée).",
    "fins_oubliees": "Vous comptez comme présentes TOUTES les vacations déjà commencées, même celles qui sont finies : "
                     "une vacation commencée en s ne couvre que s … s + duree - 1. Tenez à jour `presents[u]` pour "
                     "chaque créneau couvert.",
    "pic": "Vous mettez le besoin du PIC à chaque vague de vacations : chaque créneau est couvert, mais avec beaucoup "
           "trop de monde aux heures creuses. Partez de la gauche, et n'ajoutez que ce qui MANQUE à chaque créneau.",
}


def _cas_270() -> list[tuple]:
    from .corrige_serie_44 import conseillers_necessaires as _necessaires    # (le corrigé : pour des besoins réalistes)

    journee = [_necessaires(n, DUREE_MOYENNE_S) for _, n in prevision_exemple()]
    cas = [([1, 3, 3, 1], 2), ([0, 0, 5], 2), (journee, VACATION), (journee, 14), (journee, 1), (journee, 24),
           ([0, 0, 0], 3), ([7], 1), ([2, 2, 2, 2, 2, 2], 3), ([5, 0, 0, 0, 0, 5], 2), ([0, 4, 0, 4, 0, 4, 0], 3),
           ((3, 1, 4, 1, 5, 9, 2, 6), 4), ([10, 1, 1, 1, 1, 1, 1, 10], 4)]
    alea = random.Random(270)
    for _ in range(25):
        t = alea.randint(1, 30)
        cas.append(([alea.choice([0, alea.randint(0, 9), alea.randint(5, 40)]) for _ in range(t)], alea.randint(1, t)))
    return cas


def _proprietes_planning(besoins, duree: int, debuts, quoi: str) -> None:
    """Vérifie un planning par ses propriétés : forme, rien qui déborde, chaque créneau couvert, le moins possible."""
    total = len(besoins)
    if not isinstance(debuts, list):
        raise AssertionError(f"{quoi} renvoie {_court(debuts, 60)} ({type(debuts).__name__}) : attendu une LISTE "
                             "d'entiers.")
    if len(debuts) != total:
        raise AssertionError(f"{quoi} renvoie {len(debuts)} nombres pour {total} créneaux : un nombre de vacations par "
                             "créneau (0 si aucune ne commence).")
    for s, d in enumerate(debuts):
        if isinstance(d, bool) or not isinstance(d, int) or d < 0:
            raise AssertionError(f"{quoi} : debuts[{s}] vaut {d!r} : un nombre de vacations est un int >= 0.")
    trop_tard = [s for s in range(total - duree + 1, total) if debuts[s]]
    conseil_bug = _premier_bug(debuts, lambda bug: _ref_planifier(besoins, duree, bug), _BUGS_PLANNING,
                               lambda a, b: a == b)
    if trop_tard:
        s = trop_tard[0]
        raise AssertionError(f"{quoi} : {debuts[s]} vacation(s) commencent au créneau {s} et finiraient au créneau "
                             f"{s + duree - 1}, après la fermeture (dernier créneau : {total - 1}). "
                             f"{conseil_bug or _BUGS_PLANNING['deborde']}")
    for t in range(total):
        presents = sum(debuts[s] for s in range(max(0, t - duree + 1), t + 1))
        if presents < besoins[t]:
            raise AssertionError(f"{quoi} : au créneau {t}, {presents} conseiller(s) présent(s) pour un besoin de "
                                 f"{besoins[t]} (les vacations commencées entre les créneaux {max(0, t - duree + 1)} et "
                                 f"{t}). {conseil_bug}".rstrip())
    minimum = sum(_ref_planifier(besoins, duree))
    if sum(debuts) > minimum:
        raise AssertionError(f"{quoi} : votre planning compte {sum(debuts)} vacations ; {minimum} suffisent à couvrir "
                             f"chaque créneau. {conseil_bug or _BUGS_PLANNING['au_plus_tot']}")
    if sum(debuts) < minimum:   # impossible si tout ce qui précède est juste : le correcteur se serait trompé
        raise AssertionError(f"{quoi} : {sum(debuts)} vacations couvrent tout ? Le correcteur en attendait {minimum}.")


def verifier_planifier_vacations(f) -> None:
    for besoins, duree in _cas_270():
        quoi = f"planifier_vacations({_court(besoins, 90)}, {duree})"
        copie = repr(besoins)
        r = _appel(f, (besoins, duree), quoi)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if repr(besoins) != copie:
            raise AssertionError(f"{quoi} a modifié la liste `besoins` reçue : tenez vos comptes dans une autre liste.")
        _proprietes_planning(besoins, duree, r, quoi)

    if _appel(f, ([], 8), "planifier_vacations([], 8)") != []:
        raise AssertionError("planifier_vacations([], 8) doit renvoyer [] : aucun créneau, aucune vacation.")

    alea = random.Random(2709)
    semaine = [alea.randint(800, 5000) if 2 <= k % 48 < 26 else alea.randint(0, 300) for k in range(336)]
    quoi = "planifier_vacations(<une semaine : 336 demi-heures, des besoins de plusieurs milliers>, 14)"
    conseil = ("Le correcteur compte les lignes exécutées dans votre fichier : ajouter les vacations manquantes UNE PAR "
               "UNE (une boucle `while` qui en ajoute une et recompte), c'est des millions d'étapes. Ajoutez-les toutes "
               "d'un coup : `debuts[debut] += manque`, puis `presents[u] += manque` sur les `duree` créneaux couverts.")
    r = _appel(f, (semaine, 14), quoi, lignes_max=60 * len(semaine) * 14, conseil_lent=conseil)
    _proprietes_planning(semaine, 14, r, quoi)

    for args, attendu, conseil in (
            (({0: 3}, 2), TypeError, "`besoins` est une liste (ou un tuple) d'entiers."),
            (([3, -1, 2], 2), ValueError, "Un besoin négatif n'existe pas."),
            (([3, 2.5, 2], 2), TypeError, "2,5 conseillers : un besoin est un int."),
            (([3, True, 2], 2), TypeError, "True n'est pas un besoin."),
            (([3, "2"], 1), TypeError, "« 2 » est un texte."),
            (([3, 2, 2], 0), ValueError, "Une vacation de 0 créneau ne couvre rien."),
            (([3, 2, 2], 2.0), TypeError, "La durée est un int (2, pas 2.0)."),
            (([3, 2, 2], True), TypeError, "True n'est pas une durée."),
            (([3, 2, 2], 4), ValueError, "Une vacation de 4 créneaux ne tient pas dans une journée de 3."),
            (([], 0), ValueError, "Même sans créneau, une durée de 0 est une erreur : vérifiez `duree`."),
            (([-2], 1), ValueError, "Un besoin négatif n'existe pas.")):
        _doit_lever(f, args, f"planifier_vacations{_court(args, 60)}", conseil, attendu)


# ============================================================================ défi 271 : FilePrioritaire


class _RefFile:
    """La file de référence (un tas, une clé fixe, la suppression paresseuse) ; `bug` rejoue une erreur classique."""

    def __init__(self, vieillissement=0, bug: str = "") -> None:
        self.v, self.bug = vieillissement, bug
        self.tas, self.entrees, self.retires, self.compteur, self.horloge = [], {}, set(), itertools.count(), 0

    def _cle(self, risque, arrivee):
        if self.bug == "sans_vieillissement":
            return -risque
        if self.bug == "fifo":
            return arrivee
        if self.bug == "a_l_envers":
            return -(risque + self.v * arrivee / 60)
        if self.bug == "le_moins_prioritaire":
            return risque - self.v * arrivee / 60
        return -(risque - self.v * arrivee / 60)

    def ajouter(self, client, risque, arrivee) -> None:
        if not _nombre(risque) or not _nombre(arrivee):
            raise TypeError("un nombre")
        if not math.isfinite(risque) or not math.isfinite(arrivee) or risque < 0 or arrivee < self.horloge:
            raise ValueError("refusé")
        if client in self.entrees:
            raise ValueError("déjà en ligne")
        if self.bug == "egalite_identifiant":
            depart = client
        elif self.bug == "egalite_dernier":
            depart = -next(self.compteur)
        else:
            depart = next(self.compteur)
        entree = [self._cle(risque, arrivee), depart, client, True]
        if self.bug == "perdu":
            pass                                         # « retires » n'est jamais vidé : le client rappelé est sauté
        else:
            self.retires.discard(client)
        self.entrees[client] = entree
        heapq.heappush(self.tas, entree)
        self.horloge = arrivee

    def retirer(self, client) -> None:
        entree = self.entrees.pop(client)
        if self.bug in ("fantome", "perdu"):
            self.retires.add(client)                     # on marque le CLIENT, pas son entrée
        else:
            entree[3] = False

    def suivant(self, maintenant):
        if not _nombre(maintenant):
            raise TypeError("un nombre")
        if not math.isfinite(maintenant) or maintenant < self.horloge:
            raise ValueError("refusé")
        while self.tas:
            _, _, client, valide = heapq.heappop(self.tas)
            if self.bug in ("fantome", "perdu"):
                valide = client not in self.retires
            if valide:
                self.entrees.pop(client, None)
                self.horloge = maintenant
                return client
        raise IndexError("vide")

    def __len__(self) -> int:
        return len(self.tas) if self.bug == "len_tas" else len(self.entrees)


_BUGS_FILE = {
    "sans_vieillissement": "Votre file ne tient compte que du RISQUE : un client peu risqué attendrait tant que des "
                           "clients à risque appellent (la « famine »). La priorité vaut risque + vieillissement × "
                           "(t - arrivee) / 60 : chaque minute d'attente rapporte des points.",
    "fifo": "Votre file sert les clients dans l'ordre d'arrivée : le risque compte aussi. La priorité vaut risque + "
            "vieillissement × (t - arrivee) / 60.",
    "a_l_envers": "Le vieillissement joue à l'envers : vous favorisez les clients ARRIVÉS TARD. Plus on est arrivé tôt, "
                  "plus on a attendu, plus on a de points : la clé est -(risque - vieillissement × arrivee / 60).",
    "le_moins_prioritaire": "Vous servez le client de plus PETITE priorité : `heapq` sort le plus petit élément. Rangez "
                            "l'OPPOSÉ de la priorité : -(risque - vieillissement × arrivee / 60).",
    "egalite_identifiant": "À priorité égale, vous départagez par l'IDENTIFIANT du client (« A1 » avant « Z9 ») : la "
                           "règle sert celui qui est entré le premier. Mettez un compteur (`itertools.count()`) juste "
                           "après la clé.",
    "egalite_dernier": "À priorité égale, vous servez le DERNIER entré : c'est le premier entré qui passe (un compteur "
                       "croissant après la clé, pas son opposé).",
    "fantome": "Un client qui a raccroché puis RAPPELÉ est servi deux fois, ou avec son ancienne priorité : vous "
               "marquez le CLIENT comme retiré, et son ancienne entrée, restée dans le tas, redevient valable quand il "
               "rappelle. Marquez l'ENTRÉE (une liste modifiable : `entree[-1] = RETIRE`), et oubliez-la dans le dict.",
    "perdu": "Un client qui a raccroché puis RAPPELÉ n'est jamais servi : vous gardez son identifiant dans un ensemble "
             "de « retirés », et sa nouvelle entrée est sautée comme l'ancienne. Marquez l'ENTRÉE retirée, pas le "
             "client.",
    "len_tas": "`len(file)` compte les entrées du tas, y compris celles des clients qui ont raccroché (marquées, mais "
               "encore dans le tas). Comptez les clients EN LIGNE (`len(self._entrees)`).",
}


def _jouer(file, operations) -> list:
    """Joue les opérations ; une sortie par opération (ce que renvoie suivant ou len, ou l'exception levée)."""
    sorties = []
    for op in operations:
        try:
            if op[0] == "len":
                sorties.append(len(file))
            elif op[0] == "suivant":
                sorties.append(file.suivant(op[1]))
            else:
                getattr(file, op[0])(*op[1:])
                sorties.append(None)
        except NotImplementedError:
            raise
        except _BoucleSansFin:
            raise
        except Exception as exc:
            sorties.append(("!", type(exc).__name__, _decrire(exc)))
    return sorties


def _scenarios_271() -> list[tuple]:
    """(vieillissement, opérations) : des scénarios où chaque erreur se voit."""
    exemple = (6, [("ajouter", "C1", 20, 0), ("ajouter", "C2", 85, 60), ("ajouter", "C3", 30, 90), ("suivant", 120),
                   ("ajouter", "C4", 55, 600), ("retirer", "C3"), ("len",), ("suivant", 600), ("suivant", 610),
                   ("len",)])
    famine = (30, [("ajouter", "lent", 0, 0)] + [op for k in range(1, 8) for op in
                                                   (("ajouter", f"R{k}", 80, 60 * k), ("suivant", 60 * k))]
              + [("suivant", 600), ("len",)])
    egalites = (30, [("ajouter", "Z9", 10, 0), ("ajouter", "A1", 10, 0), ("ajouter", "M5", 25, 30),
                     ("ajouter", "B2", 40, 60), ("ajouter", "Y8", 40, 60), ("suivant", 60), ("suivant", 60),
                     ("suivant", 61), ("suivant", 62), ("suivant", 63), ("len",)])
    rappel = (0, [("ajouter", "X", 5, 0), ("ajouter", "Y", 50, 0), ("ajouter", "W", 30, 0), ("retirer", "X"),
                  ("len",), ("ajouter", "X", 90, 10), ("len",), ("suivant", 20), ("suivant", 20), ("ajouter", "Y", 1, 30),
                  ("retirer", "Y"), ("ajouter", "Y", 2, 40), ("suivant", 50), ("suivant", 50), ("len",)])
    retires = (12, [("ajouter", k, 50 - k, 10 * k) for k in range(8)] + [("retirer", k) for k in (0, 2, 4)]
               + [("len",), ("suivant", 100), ("len",), ("suivant", 100), ("suivant", 100), ("suivant", 100),
                  ("suivant", 100), ("len",)])
    melange = (60, [("ajouter", 7, 10, 0), ("ajouter", "sept", 10, 0), ("ajouter", (7, 7), 10, 0), ("suivant", 0),
                    ("suivant", 0), ("suivant", 0)])
    scenarios = [exemple, famine, egalites, rappel, retires, melange]
    for graine, v in ((2711, 6), (2712, 0), (2713, 30), (2714, 2.5)):
        alea = random.Random(graine)
        ops, en_ligne, horloge, numero = [], [], 0, 0
        for _ in range(400):
            tirage = alea.random()
            if tirage < 0.45 or not en_ligne:
                horloge += alea.choice([0, 0, 1, 5, 30, 90])
                client = alea.choice([f"C{numero}", f"C{alea.randint(0, max(0, numero - 1))}"])
                numero += 1
                if client in en_ligne:
                    client = f"C{numero}-bis"
                risque = alea.randint(0, 10) * 10 if v in (0, 30) else round(alea.uniform(0, 100), 9)
                ops.append(("ajouter", client, risque, horloge))
                en_ligne.append(client)
            elif tirage < 0.6:
                client = alea.choice(en_ligne)
                en_ligne.remove(client)
                ops.append(("retirer", client))
            elif tirage < 0.95:
                horloge += alea.choice([0, 10, 60])
                ops.append(("suivant", horloge))
                ref = _RefFile(v)
                for op in ops[:-1]:
                    _jouer(ref, [op])
                en_ligne.remove(_RefFile.suivant(ref, horloge))
            else:
                ops.append(("len",))
        scenarios.append((v, ops))
    return scenarios


def _decrire_ops(operations, jusqu_a: int) -> str:
    morceaux = []
    for op in operations[max(0, jusqu_a - 5): jusqu_a + 1]:
        if op[0] == "len":
            morceaux.append("len(file)")
        else:
            morceaux.append(f"{op[0]}({', '.join(repr(x) for x in op[1:])})")
    return ("… " if jusqu_a > 5 else "") + " ; ".join(morceaux)


def _creer_file(classe, args: tuple, quoi: str):
    return _appel(classe, args, quoi)


def _sans_messages(sorties) -> list:
    """Les sorties, où une exception ne compte que par son type (pour comparer deux versions)."""
    return [s[:2] if isinstance(s, tuple) and s[:1] == ("!",) else s for s in sorties]


def verifier_file_prioritaire(classe) -> None:
    # ------------------------------------------------------------------ deux files ne partagent rien
    a, b = _creer_file(classe, (6,), "FilePrioritaire(6)"), _creer_file(classe, (6,), "FilePrioritaire(6)")
    _appel(a.ajouter, ("C1", 10, 0), "a.ajouter('C1', 10, 0)")
    if _appel(len, (b,), "len(b)") != 0:
        raise AssertionError("Deux files partagent leurs clients : après a.ajouter('C1', 10, 0), une AUTRE file b n'est "
                             "plus vide. Créez le tas et le dict dans __init__ (pas comme attributs de la classe).")

    # ------------------------------------------------------------------ les scénarios
    scenarios = _scenarios_271()
    obtenus = [_sans_messages(_jouer(_creer_file(classe, (v,), f"FilePrioritaire({v!r})"), ops))
               for v, ops in scenarios]
    attendus = [_sans_messages(_jouer(_RefFile(v), ops)) for v, ops in scenarios]
    for (v, operations), o_tout, a_tout in zip(scenarios, obtenus, attendus):
        if o_tout == a_tout:
            continue
        i = next(k for k, (o, a) in enumerate(zip(o_tout, a_tout)) if o != a)
        contexte = f"Avec FilePrioritaire({v!r}) : {_decrire_ops(operations, i)}"
        o, a = o_tout[i], a_tout[i]
        if isinstance(o, tuple) and o[:1] == ("!",):
            detail = _jouer(_creer_file(classe, (v,), f"FilePrioritaire({v!r})"), operations)[i]
            conseil = ""
            if "not supported between" in str(detail[2]):
                conseil = _conseil_exception(TypeError("not supported between"))
            elif o[1] == "KeyError" and operations[i][0] == "retirer":
                conseil = ("Ce client est en ligne : il a peut-être raccroché puis rappelé, ou vous l'avez servi à sa "
                           "place plus tôt.")
            raise AssertionError(f"{contexte} : la dernière opération lève {detail[2]} au lieu de "
                                 f"{'renvoyer ' + repr(a) if not isinstance(a, tuple) else 'lever ' + a[1]}. "
                                 f"{conseil}".rstrip())
        conseil = _premier_bug(obtenus, lambda bug: [_sans_messages(_jouer(_RefFile(x, bug), ops)) for x, ops in scenarios],
                               _BUGS_FILE, lambda x, y: x == y)
        quoi = "len(file)" if operations[i][0] == "len" else f"suivant({operations[i][1]!r})"
        raise AssertionError(f"{contexte} : {quoi} renvoie {o!r} au lieu de {a!r}. {conseil}".rstrip())

    if _appel(_suivant_sans_argument, (classe,), "FilePrioritaire()") != "B":
        raise AssertionError("FilePrioritaire() (sans argument) : le vieillissement par défaut vaut 0 : à risque égal, "
                             "premier entré, premier servi ; à risque différent, le plus risqué d'abord.")

    # ------------------------------------------------------------------ erreurs (et une opération refusée ne change rien)
    for args, attendu, conseil in (
            ((-1,), ValueError, "Un vieillissement négatif n'existe pas."),
            (("6",), TypeError, "« 6 » est un texte."),
            ((True,), TypeError, "True n'est pas un vieillissement."),
            ((float("nan"),), ValueError, "NaN n'est pas un vieillissement."),
            ((float("inf"),), ValueError, "Un vieillissement infini n'en est pas un.")):
        _doit_lever(classe, args, f"FilePrioritaire{_court(args, 30)}", conseil, attendu)
    file = _creer_file(classe, (6,), "FilePrioritaire(6)")
    _appel(file.ajouter, ("A", 50, 40), "file.ajouter('A', 50, 40)")
    for methode, args, attendu, conseil in (
            ("ajouter", ("B", -1, 50), ValueError, "Un risque négatif n'existe pas."),
            ("ajouter", ("B", "50", 50), TypeError, "« 50 » est un texte, pas un risque."),
            ("ajouter", ("B", True, 50), TypeError, "True n'est pas un risque."),
            ("ajouter", ("B", float("nan"), 50), ValueError, "NaN n'est pas un risque (et il casserait l'ordre du tas)."),
            ("ajouter", ("B", 50, float("nan")), ValueError, "NaN n'est pas une heure d'arrivée."),
            ("ajouter", ("B", 50, "50"), TypeError, "« 50 » est un texte, pas une heure d'arrivée."),
            ("ajouter", ("B", 50, 30), ValueError, "L'horloge ne recule pas : arrivée à 30 s, alors que A est arrivé à "
                                                   "40 s."),
            ("ajouter", ("A", 90, 100), ValueError, "A est déjà en ligne : on ne l'ajoute pas deux fois."),
            ("retirer", ("Z",), KeyError, "Z n'est pas en ligne : KeyError (comme `del d['Z']`)."),
            ("suivant", (39,), ValueError, "L'horloge ne recule pas : maintenant = 39 s, alors que A est arrivé à 40 s."),
            ("suivant", ("40",), TypeError, "« 40 » est un texte, pas une heure."),
            ("suivant", (float("inf"),), ValueError, "Un instant infini n'existe pas.")):
        _doit_lever(getattr(file, methode), args, f"file.{methode}{_court(args, 40)}", conseil, attendu)
    # l'ajout refusé de A à 100 s ne doit pas avoir avancé l'horloge, ni changé la file
    _appel(file.ajouter, ("B", 10, 50), "file.ajouter('B', 10, 50) après des opérations refusées",
           "Une opération refusée ne change rien : ni la file, ni l'horloge (l'ajout refusé de A à 100 s ne l'a pas "
           "avancée).")
    if _appel(len, (file,), "len(file)") != 2:
        raise AssertionError(f"Après des opérations refusées puis ajouter('B', …), len(file) vaut {len(file)} au lieu de "
                             "2 : une opération refusée ne change rien à la file.")
    if _appel(file.suivant, (60,), "file.suivant(60)") != "A":
        raise AssertionError("Après des opérations refusées, file.suivant(60) devrait servir A (risque 50) : une "
                             "opération refusée ne change rien à la file.")
    _appel(file.suivant, (60,), "file.suivant(60)")
    _doit_lever(file.suivant, (500,), "file.suivant(500) sur une file vide", "Personne n'attend : IndexError.",
                IndexError)
    _doit_lever(file.suivant, (10,), "file.suivant(10) sur une file vide, avant l'horloge",
                "Vérifiez `maintenant` AVANT de regarder si la file est vide : l'horloge ne recule pas (ValueError).")
    _appel(file.ajouter, ("C", 10, 200), "file.ajouter('C', 10, 200) après un suivant(500) refusé (file vide)",
           "Un suivant refusé (IndexError : file vide) ne fait pas avancer l'horloge : une arrivée à 200 s reste "
           "possible.")

    # ------------------------------------------------------------------ 60 000 opérations sur 20 000 clients
    alea = random.Random(2719)
    brutes, horloge = [], 0
    for k in range(20_000):
        horloge += alea.choice([0, 1, 2, 3])
        brutes.append(("ajouter", f"C{k:05d}", alea.uniform(0, 100), horloge))
        if k % 2 == 0:
            brutes.append(("retirer", f"C{alea.randrange(k + 1):05d}"))
        if k >= 8_000:
            brutes.append(("suivant", horloge))
    brutes += [("suivant", horloge + 1)] * 3_000 + [("len",)]
    operations = _sans_retraits_invalides(brutes)
    attendus = _jouer(_RefFile(6), operations)
    gros = _creer_file(classe, (6,), "FilePrioritaire(6)")
    conseil = ("Le correcteur compte les lignes exécutées dans votre fichier : recalculer la priorité de tous les "
               "clients à chaque `suivant` (ou chercher le maximum dans une liste), c'est des centaines de millions "
               "d'étapes. Tous les clients gagnent les MÊMES points par minute : leur ORDRE ne change jamais. Rangez-"
               "les une fois dans un tas avec la clé -(risque - vieillissement × arrivee / 60), et marquez les "
               "retirés au lieu de les chercher.")
    obtenus = _sans_messages(_appel(_jouer, (gros, operations), f"{len(operations)} opérations sur 20 000 clients",
                                    lignes_max=80 * len(operations), conseil_lent=conseil, fichier=_fichier(classe)))
    attendus = _sans_messages(attendus)
    if obtenus != attendus:
        i = next(k for k, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        raise AssertionError(f"Sur {len(operations)} opérations (20 000 clients), l'opération {i} "
                             f"({_decrire_ops(operations, i).split(' ; ')[-1]}) donne {_court(obtenus[i], 80)} au lieu de "
                             f"{_court(attendus[i], 80)} (alors que les petits scénarios passent).")


def _suivant_sans_argument(classe):
    """Une file créée SANS argument : à risque égal, premier entré, premier servi."""
    file = classe()
    file.ajouter("A", 10, 0)
    file.ajouter("B", 20, 5)
    file.ajouter("C", 20, 5)
    return file.suivant(10)


def _sans_retraits_invalides(operations) -> list:
    """Garde les `retirer` d'un client encore en ligne (selon la référence) : le scénario reste valable."""
    ref, gardees = _RefFile(6), []
    for op in operations:
        if op[0] == "retirer" and op[1] not in ref.entrees:
            continue
        _jouer(ref, [op])
        gardees.append(op)
    return gardees


# ---------------------------------------------------------------------------- la série

SERIE = Serie(
    code="44",
    titre="Faire attendre le moins possible : Erlang C, effectif, simulation, indicateurs, planning, file prioritaire",
    module_eleve="defis.serie_44",
    module_corrige="defis.corrige_serie_44",
    defis=(
        Defi(266, "erlang_c", "🟢", "—", verifier_erlang_c),
        Defi(267, "conseillers_necessaires", "🟡", "—", verifier_conseillers_necessaires),
        Defi(268, "simuler_file", "🟡", "—", verifier_simuler_file),
        Defi(269, "indicateurs", "🟡", "—", verifier_indicateurs),
        Defi(270, "planifier_vacations", "🔴", "—", verifier_planifier_vacations),
        Defi(271, "FilePrioritaire", "🔴", "—", verifier_file_prioritaire),
    ),
)

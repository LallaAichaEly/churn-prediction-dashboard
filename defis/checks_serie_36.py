"""Vérifications automatiques de la série 36 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est Samir, d'astreinte sur l'API de score, qui ne veut plus d'un tableau de bord tout vert
pendant que les conseillers attendent, et le serveur qui compte ce que chaque calcul lui coûte :

- au défi 218, il glisse dans le journal tout ce qu'un vrai journal contient : traces d'erreur, lignes coupées,
  lignes vides, durées « nan », statuts impossibles, identifiants qui n'en sont pas (« v2 ») ;
- au défi 222, il vous passe 14 400 minutes dans une `Comptee` (qui compte les cases lues) et compte les lignes
  exécutées dans votre fichier : relire chaque fenêtre à chaque minute se voit, quelle que soit la machine ;
- au défi 223, il vous passe une trace de 40 000 spans, dont un qui a 20 000 enfants, et une chaîne de 5 000
  appels imbriqués : chercher les enfants dans toute la liste, ou une fonction récursive, se voient ;
- partout, il rejoue les erreurs classiques (4xx comptés comme des erreurs, une route par URL, « 61.2ms » lu
  comme des secondes, moyenne des p99, comptes non cumulés, pas d'interpolation, dernier moins premier, lecture
  manquée prise pour un zéro, moyenne des minutes, une seule fenêtre, combustion sans l'objectif, fenêtre
  décalée d'une minute, durées des enfants additionnées, enfant qui déborde, orphelins oubliés…) pour nommer la
  vôtre ;
- et il vérifie que vos fonctions ne modifient pas ce qu'elles reçoivent.

Les durées et les taux se comparent à 1e-9 près : il ne vous reproche jamais un arrondi.
"""

from __future__ import annotations

import copy
import math
import numbers
import random
import sys

from .observabilite import (OBJECTIF, REGLES, SEAUX, Comptee, Regle, Span, compteur_exemple, generer_histogramme,
                            generer_journal, generer_minutes, generer_trace, histogrammes_exemple, journal_exemple,
                            minutes_exemple, trace_exemple)
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


class _Plafonnee(Comptee):
    """Une Comptee qui arrête net votre fonction au-delà de `plafond` cases lues."""

    def __init__(self, valeurs, plafond: int) -> None:
        super().__init__(valeurs)
        self.plafond = plafond

    def _compter(self, n: int) -> None:
        self.lues += n
        if self.lues > self.plafond:
            raise _BoucleSansFin

    def __getitem__(self, i):
        if isinstance(i, slice):
            morceau = self._valeurs[i]
            self._compter(len(morceau))
            return morceau
        valeur = self._valeurs[i]
        self._compter(1)
        return valeur

    def __iter__(self):
        for valeur in self._valeurs:
            self._compter(1)
            yield valeur


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
        raise AssertionError(f"{quoi} lève RecursionError : une fonction qui s'appelle elle-même une fois par niveau "
                             "s'arrête vers 1 000 niveaux. Pas besoin de parcourir l'arbre : groupez les enfants par "
                             "parent dans un dict, puis traitez chaque span avec une simple boucle.") from None
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
    """Exige que f(*args, **kwargs) lève `attendu`."""
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
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _egaux_nan(a, b) -> bool:
    """a == b, en considérant deux NaN comme égaux (dans des listes, tuples et dicts imbriqués)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_egaux_nan(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_egaux_nan(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return type(a) is type(b) and a == b


def _intact(avant, apres, quoi: str, nom: str) -> None:
    if not _egaux_nan(avant, apres):
        raise AssertionError(f"{quoi} a modifié {nom} reçu(s) (avant : {_court(avant, 100)}, après : "
                             f"{_court(apres, 100)}). Une fonction ne touche pas à ce qu'on lui prête : travaillez sur "
                             "une copie, ou construisez un nouvel objet.")


def _pas_none(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")


def _nombre(x) -> bool:
    """Un nombre réel, pas un bool (les entiers et flottants de NumPy sont acceptés)."""
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _proche(x, y, tol: float = 1e-9) -> bool:
    return _nombre(x) and _nombre(y) and math.isclose(x, y, rel_tol=tol, abs_tol=tol)


def _meme(x, y) -> bool:
    """Deux valeurs égales : None et None, deux infinis égaux, ou deux nombres à 1e-9 près."""
    if x is None or y is None:
        return x is None and y is None
    if _nombre(x) and _nombre(y) and x != x and y != y:   # deux NaN
        return True
    if _nombre(x) and _nombre(y) and (math.isinf(x) or math.isinf(y)):
        return x == y
    return _proche(x, y)


def _memes(r, attendu) -> bool:
    try:
        return len(r) == len(attendu) and all(_meme(x, y) for x, y in zip(r, attendu))
    except TypeError:
        return False


def _fmt(x) -> str:
    return f"{x:.6g}" if _nombre(x) else repr(x)


# ---------------------------------------------------------------------------- défi 218 : lire le journal

_MICRO = "µs"
_CHIFFRES = set("0123456789")
_MAJUSCULES = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _ref_duree(texte: str, variante: str):
    if variante == "unite":                                 # « on enlève les lettres et on lit le nombre »
        nombre = texte.rstrip("µsmµ")
        try:
            valeur = float(nombre)
        except ValueError:
            return None
        return valeur if math.isfinite(valeur) and valeur >= 0 else None
    if variante == "s_d_abord":
        unites = (("s", 1000.0), ("ms", 1.0), (_MICRO, 0.001))
    elif variante == "sans_micro":
        unites = (("ms", 1.0), ("s", 1000.0))
    else:
        unites = ((_MICRO, 0.001), ("ms", 1.0), ("s", 1000.0))
    for unite, facteur in unites:
        if texte.endswith(unite):
            try:
                valeur = float(texte[: -len(unite)])
            except ValueError:
                return None
            if variante != "nan" and (not math.isfinite(valeur) or valeur < 0):
                return None
            if variante == "nan" and valeur < 0:
                return None
            return valeur / 1000 if unite == _MICRO else valeur * facteur
    return None


def _ref_identifiant(morceau: str, variante: str) -> bool:
    if variante == "chiffre":
        return any(c in _CHIFFRES for c in morceau)
    if morceau and set(morceau) <= _CHIFFRES:
        return True
    return len(morceau) >= 4 and morceau[0] in _MAJUSCULES and set(morceau[1:]) <= _CHIFFRES


def _ref_route(methode: str, chemin: str, variante: str) -> str:
    if variante == "url":
        return f"{methode} {chemin}"
    if variante != "query":
        chemin = chemin.split("?", 1)[0]
    if variante != "ids":
        chemin = "/".join("{id}" if _ref_identifiant(m, variante) else m for m in chemin.split("/"))
    if variante != "slash":
        chemin = chemin.rstrip("/") or "/"
    return f"{methode} {chemin}"


def _ref_journal(lignes, variante: str = ""):
    """(routes, illisibles), routes = {route: (requetes, erreurs, durees)} ; `variante` rejoue une erreur classique."""
    routes: dict = {}
    illisibles = 0
    for ligne in lignes:
        if variante == "split_espace" and not ligne.strip():
            continue
        champs = ligne.split(" ") if variante == "split_espace" else ligne.split()
        if not champs:
            if variante == "vides":
                illisibles += 1
            continue
        if len(champs) != 5 or not champs[2].startswith("/"):
            illisibles += 1
            continue
        _, methode, chemin, texte_statut, texte_duree = champs
        if variante == "statut_libre":
            try:
                statut = int(texte_statut)
            except ValueError:
                statut = None
        else:
            statut = int(texte_statut) if (len(texte_statut) == 3 and set(texte_statut) <= _CHIFFRES
                                           and 100 <= int(texte_statut) <= 599) else None
        duree = _ref_duree(texte_duree, variante)
        if statut is None or duree is None:
            illisibles += 1
            continue
        c = routes.setdefault(_ref_route(methode, chemin, variante), [0, 0, []])
        c[0] += 1
        c[1] += (statut >= 400) if variante == "4xx" else (500 <= statut <= 599)
        c[2].append(duree)
    if variante == "sans_illisibles":
        illisibles = 0
    return {route: tuple(c) for route, c in routes.items()}, illisibles


_VARIANTES_JOURNAL = [
    ("4xx", "seuls les statuts 5xx sont des erreurs du SERVICE. Un 404 (client inconnu) ou un 422 (demande mal "
            "formée), c'est l'appelant qui s'est trompé : une requête, pas une erreur."),
    ("url", "on compte par ROUTE, pas par URL : retirez les paramètres (après « ? »), remplacez chaque identifiant "
            "par « {id} » et retirez les « / » finaux. Sinon, une ligne par client."),
    ("ids", "chaque morceau du chemin qui est un identifiant (un nombre, ou une majuscule suivie d'au moins 3 "
            "chiffres : C04182) devient « {id} » : sinon, une route par client."),
    ("chiffre", "« v2 » n'est pas un identifiant : seuls un nombre (que des chiffres) ou une majuscule suivie "
                "d'au moins 3 chiffres (C04182) deviennent « {id} ». Contenir un chiffre ne suffit pas."),
    ("query", "les paramètres (tout ce qui suit « ? ») ne font pas partie de la route : /clients?page=2 et "
              "/clients?page=3 sont la même route."),
    ("slash", "les « / » finaux ne font pas partie de la route : /clients/ et /clients sont la même route "
              "(mais « / » seul reste « / »)."),
    ("unite", "la durée est en µs, ms ou s : convertissez-la en millisecondes (4.18s → 4180.0 ; 850µs → 0.85). "
              "Vous lisez le nombre sans regarder son unité."),
    ("s_d_abord", "« 61.2ms » finit aussi par « s » : testez « µs » et « ms » AVANT « s », sinon toutes les "
                  "durées en ms deviennent illisibles."),
    ("sans_micro", "« 850µs » est une durée lisible : 850 microsecondes, soit 0.85 ms (copiez le « µ » de la "
                   "docstring : c'est le signe micro, U+00B5)."),
    ("nan", "float(\"nan\") et float(\"inf\") réussissent, mais « nanms » ou « infs » ne sont pas des durées : une "
            "durée doit être un nombre fini (`math.isfinite`), sinon la ligne est illisible."),
    ("statut_libre", "le statut est un nombre de 3 chiffres entre 100 et 599 : « 600 » ou « 99 » ne sont pas des "
                     "statuts HTTP, la ligne est illisible."),
    ("vides", "une ligne vide (ou faite d'espaces) n'est pas une requête ratée : on l'ignore, sans la compter "
              "dans `illisibles` (avec `ligne.split()`, sans argument, elle donne une liste vide)."),
    ("sans_illisibles", "une ligne illisible ne s'ignore pas en silence : comptez-la dans `illisibles` (c'est le "
                        "premier signe qu'un journal se dégrade)."),
    ("split_espace", "découpez avec `ligne.split()` (sans argument) : `split(\" \")` garde le « \\n » de fin de ligne "
                     "et fabrique des champs vides quand deux espaces se suivent."),
]


def _lire_resultat_journal(r, quoi: str):
    _pas_none(r, quoi)
    if not isinstance(r, tuple) or len(r) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : Journal(routes, illisibles) est attendu.")
    routes, illisibles = r
    if not isinstance(routes, dict):
        raise AssertionError(f"{quoi} : routes est un {type(routes).__name__}, un dict {{route: Route(...)}} est attendu.")
    if not isinstance(illisibles, int) or isinstance(illisibles, bool):
        raise AssertionError(f"{quoi} : illisibles vaut {illisibles!r}, un int (le nombre de lignes illisibles) est attendu.")
    for route, valeur in routes.items():
        if not isinstance(valeur, tuple) or len(valeur) != 3 or not isinstance(valeur[2], list):
            raise AssertionError(f"Dans {quoi}, la route {route!r} vaut {_court(valeur, 100)} : Route(requetes, "
                                 "erreurs, durees_ms) est attendu, durees_ms étant une liste.")
    return routes, illisibles


def _memes_journaux(a, b) -> bool:
    (routes_a, ill_a), (routes_b, ill_b) = a, b
    if ill_a != ill_b or list(routes_a) != list(routes_b):
        return False
    for route in routes_a:
        (n_a, e_a, d_a), (n_b, e_b, d_b) = routes_a[route], routes_b[route]
        if n_a != n_b or e_a != e_b or not _memes(d_a, d_b):
            return False
    return True


def _un_journal(f, lignes, quoi: str) -> None:
    avant = list(lignes)
    r = _appel(f, (lignes,), quoi, "Une ligne illisible se compte dans `illisibles` : elle n'arrête pas la lecture des "
                                   "autres (pas de `raise`, et attention à `int()` et `float()` sur un texte abîmé).")
    _intact(avant, lignes, quoi, "les lignes")
    obtenu = _lire_resultat_journal(r, quoi)
    attendu = _ref_journal(lignes)
    if _memes_journaux(obtenu, attendu):
        return
    for variante, message in _VARIANTES_JOURNAL:
        autre = _ref_journal(lignes, variante)
        if _memes_journaux(obtenu, autre) and not _memes_journaux(autre, attendu):
            raise AssertionError(f"{quoi} : {message}")
    routes, illisibles = obtenu
    routes_attendues, illisibles_attendues = attendu
    if set(routes) != set(routes_attendues):
        manque = [k for k in routes_attendues if k not in routes]
        trop = [k for k in routes if k not in routes_attendues]
        raise AssertionError(f"{quoi} : routes {_court(list(routes), 120)} au lieu de {_court(list(routes_attendues), 120)}"
                             f"{' (il manque ' + _court(manque, 60) + ')' if manque else ''}"
                             f"{' (en trop : ' + _court(trop, 60) + ')' if trop else ''}.")
    if list(routes) != list(routes_attendues):
        raise AssertionError(f"{quoi} range les routes dans l'ordre {_court(list(routes), 120)} au lieu de "
                             f"{_court(list(routes_attendues), 120)} : l'ordre où chaque route APPARAÎT dans le journal "
                             "(un dict garde l'ordre d'insertion ; ne le triez pas).")
    if illisibles != illisibles_attendues:
        raise AssertionError(f"{quoi} : illisibles vaut {illisibles} au lieu de {illisibles_attendues}. Une ligne non "
                             "vide est illisible si elle n'a pas 5 champs, si le chemin ne commence pas par « / », si le "
                             "statut n'est pas un nombre de 3 chiffres entre 100 et 599, ou si la durée ne se lit pas.")
    for route, (n, e, d) in routes_attendues.items():
        rn, re_, rd = routes[route]
        if rn != n:
            raise AssertionError(f"{quoi} : la route {route!r} a {rn!r} requêtes au lieu de {n}.")
        if re_ != e:
            raise AssertionError(f"{quoi} : la route {route!r} a {re_!r} erreurs au lieu de {e} (les statuts 500 à 599).")
        if not _memes(rd, d):
            i = next((i for i, (x, y) in enumerate(zip(rd, d)) if not _meme(x, y)), min(len(rd), len(d)))
            raise AssertionError(f"{quoi} : pour la route {route!r}, la durée n°{i} vaut "
                                 f"{_fmt(rd[i]) if i < len(rd) else 'rien'} au lieu de {_fmt(d[i]) if i < len(d) else 'rien'} "
                                 "(en millisecondes, dans l'ordre des lignes).")


_ILLISIBLES = [
    "Traceback (most recent call last):", '  File "/app/api.py", line 112, in predict', "TimeoutError: le CRM",
    "x GET /a 2OO 5ms", "x GET /a 600 5ms", "x GET /a 99 5ms", "x GET /a 2000 5ms", "x GET /a abc 5ms",
    "x GET /a 200 5", "x GET /a 200 5h", "x GET /a 200 ms", "x GET /a 200 -5ms", "x GET /a 200 nanms",
    "x GET /a 200 infs", "x GET /a 200 1.2.3s", "x GET /a 200 1m2.5s", "x GET a/b 200 5ms", "x GET /a 200",
    "x GET /a 200 5ms en trop", "2026-11-17T14:21:07.312Z GET /clients/C04182/sc", "x GET /a 200 nanµs",
]
_LISIBLES = [
    "x GET /clients/C123/score 200 5ms", "x GET /clients/C12/score 200 5ms", "x GET /clients/c1234 200 5ms",
    "x GET /clients/CC1234 200 5ms", "x GET / 200 1ms", "x GET /?page=2 200 1ms", "x GET /clients// 200 2ms",
    "x POST /predict 422 3.5ms", "x GET /v2/clients/7 503 0s", "x GET /v2/clients/7 599 0ms", "x GET /a 100 0µs",
    "x DELETE /sessions/42 204 12ms", "x GET /rapports/2026/11 200 1.25s", "x GET /health 200 999µs",
]


def verifier_lire_journal(f) -> None:
    doc = ["2026-11-17T14:00:01.874Z GET /clients/C04182/score 200 61.2ms",
           "2026-11-17T14:00:02.310Z GET /clients/C00007/score/?modele=v3 503 4.18s",
           "Traceback (most recent call last):", "",
           "2026-11-17T14:00:02.950Z GET /health 200 850µs",
           "2026-11-17T14:00:03.122Z GET /v2/clients/48213 404 12ms"]
    _un_journal(f, doc, "lire_journal(l'exemple de la docstring)")
    petits = [
        [], [""], ["x GET /clients/C00001/score 404 5ms", "x GET /clients/C00002/score 500 7ms"],
        ["x GET /v2/clients/12 200 5ms"], ["x GET /clients?contrat=mensuel&page=2 200 150ms", "x GET /clients 200 1ms"],
        ["x GET /clients/ 200 5ms", "x GET /clients 200 6ms"], ["x GET /a 200 1.5s", "x GET /a 200 2ms"],
        ["x GET /a 200 850µs", "x GET /a 200 2ms"], ["x GET /a 200 2ms", "x GET /a 200 nanms"],
        ["x GET /a 200 2ms", "x GET /a 600 2ms"], ["x GET /a 200 2ms", "", "   "],
        ["x GET /a 200 2ms", "Traceback (most recent call last):"], ["x GET /a 200 2.5s\n", "x  GET /a 200 3s"],
    ]
    for lignes in petits:
        _un_journal(f, lignes, f"lire_journal({_court(lignes, 110)})")
    for ligne in _ILLISIBLES + _LISIBLES:
        _un_journal(f, ["x GET /ok 200 1ms", ligne], f"lire_journal(['x GET /ok 200 1ms', {ligne!r}])")
    _un_journal(f, journal_exemple(), "lire_journal(journal_exemple())")
    alea = random.Random(218)
    for essai in range(60):
        lignes = generer_journal(alea.randint(0, 120), graine=alea.randint(0, 10**6), illisibles=alea.random() * 0.2,
                                 panne=(alea.randint(0, 50), alea.randint(50, 120)) if alea.random() < 0.3 else None)
        for _ in range(alea.randint(0, 6)):
            lignes.insert(alea.randint(0, len(lignes)), alea.choice(_ILLISIBLES + _LISIBLES))
        if alea.random() < 0.2:
            lignes = [ligne + "\n" for ligne in lignes]
        _un_journal(f, lignes, f"lire_journal(<{len(lignes)} lignes tirées au hasard>) (n°{essai})")


# ---------------------------------------------------------------------------- défi 219 : quantile d'un histogramme


def _ref_quantile_un(bornes, cumuls, q, variante: str = ""):
    total = cumuls[-1]
    if total == 0:
        return None
    rang = q * total
    for i in range(len(bornes) - 1):
        if cumuls[i] >= rang:
            bas, avant = (bornes[i - 1], cumuls[i - 1]) if i else (0.0, 0)
            if variante == "haut":
                return bornes[i]
            if variante == "bas":
                return bas
            if variante == "depuis_zero":
                return bornes[i] * rang / cumuls[i]
            return bas + (bornes[i] - bas) * (rang - avant) / (cumuls[i] - avant)
    return math.inf if variante == "infini" else bornes[-2]


def _ref_quantile(histogrammes, q, variante: str = ""):
    if not histogrammes:
        return None
    bornes = [b for b, _ in histogrammes[0]]
    if variante in ("moyenne", "max"):
        valeurs = [_ref_quantile_un(bornes, [c for _, c in h], q) for h in histogrammes]
        valeurs = [v for v in valeurs if v is not None]
        if not valeurs:
            return None
        return sum(valeurs) / len(valeurs) if variante == "moyenne" else max(valeurs)
    cumuls = [0] * len(bornes)
    for h in histogrammes:
        courant = 0
        for i, (_, c) in enumerate(h):
            courant = courant + c if variante == "non_cumule" else c
            cumuls[i] += courant
    return _ref_quantile_un(bornes, cumuls, q, variante)


_VARIANTES_QUANTILE = [
    ("moyenne", "vous faites la MOYENNE des quantiles des serveurs : elle ne correspond à aucune requête (un serveur "
                "lent et peu chargé pèse autant qu'un rapide très chargé). Additionnez les histogrammes seau par "
                "seau, PUIS cherchez le quantile dans la somme."),
    ("max", "vous prenez le plus grand quantile des serveurs : c'est le p99 du pire serveur, pas celui du service. "
            "Additionnez les histogrammes seau par seau, PUIS cherchez le quantile dans la somme."),
    ("non_cumule", "les comptes sont DÉJÀ cumulés : (0.5, 90) veut dire « 90 requêtes ont duré 0.5 s ou moins », "
                   "pas « 90 requêtes entre 0.1 et 0.5 s ». Ne les additionnez pas d'un seau au suivant."),
    ("haut", "vous renvoyez la borne haute du seau : il faut INTERPOLER dans le seau, bas + (haut − bas) × "
             "(rang − compte_precedent) / (compte_i − compte_precedent)."),
    ("bas", "vous renvoyez la borne basse du seau : il faut INTERPOLER dans le seau, bas + (haut − bas) × "
            "(rang − compte_precedent) / (compte_i − compte_precedent)."),
    ("depuis_zero", "l'interpolation part de la borne du seau PRÉCÉDENT (pas de 0) et ne compte que les requêtes "
                    "DE CE SEAU : bas + (haut − bas) × (rang − compte_precedent) / (compte_i − compte_precedent)."),
    ("infini", "quand le rang tombe dans le seau +Inf, on ne sait rien de plus que « au-delà de la dernière borne "
               "finie » : la réponse est cette borne finie, pas math.inf."),
]


def _un_quantile(f, histogrammes, q, quoi: str) -> None:
    avant = copy.deepcopy(histogrammes)
    r = _appel(f, (histogrammes, q), quoi)
    _intact(avant, histogrammes, quoi, "les histogrammes")
    attendu = _ref_quantile(histogrammes, q)
    if attendu is None:
        if r is not None:
            raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de None : sans aucune requête, il n'y a pas de "
                                 "quantile.")
        return
    _pas_none(r, quoi)
    if not _nombre(r):
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : un nombre (en secondes) est attendu.")
    if _meme(r, attendu):
        return
    for variante, message in _VARIANTES_QUANTILE:
        autre = _ref_quantile(histogrammes, q, variante)
        if autre is not None and _meme(r, autre) and not _meme(autre, attendu):
            raise AssertionError(f"{quoi} renvoie {_fmt(r)} au lieu de {_fmt(attendu)} : {message}")
    raise AssertionError(f"{quoi} renvoie {_fmt(r)} au lieu de {_fmt(attendu)}. Le rang vaut q × total (le compte "
                         "du seau +Inf, histogrammes ADDITIONNÉS) ; cherchez le premier seau fini dont le compte cumulé "
                         "est >= rang.")


def _histogrammes_hasard(rng: random.Random) -> tuple[list, float]:
    seaux = sorted(rng.sample(SEAUX[:-1], rng.randint(1, 6))) + [math.inf]
    histogrammes = []
    for _ in range(rng.randint(1, 4)):
        lenteur = rng.choice([0.02, 0.06, 0.3, 2.0])
        durees = [rng.lognormvariate(math.log(lenteur), rng.choice([0.3, 0.8, 1.5]))
                  for _ in range(rng.choice([0, 1, 3, 20, 200, 1000]))]
        h = generer_histogramme(durees, seaux)
        if rng.random() < 0.2:
            h = [(b, float(c)) for b, c in h]
        histogrammes.append(h)
    q = rng.choice([0.5, 0.9, 0.95, 0.99, 0.999, 1.0, rng.random() or 0.5])
    return histogrammes, q


def verifier_quantile_histogramme(f) -> None:
    a = [(0.1, 60), (0.5, 90), (1.0, 100), (math.inf, 100)]
    b = [(0.1, 10), (0.5, 20), (1.0, 50), (math.inf, 60)]
    for histogrammes, q in [([a], 0.5), ([a, b], 0.5), ([a, b], 0.99), ([a], 0.95), ([b], 0.9), ([a, b], 1.0)]:
        _un_quantile(f, histogrammes, q, f"quantile_histogramme({_court(histogrammes, 90)}, {q})")
    # Un cas construit pour que chaque erreur classique donne un résultat différent.
    rapide = [(0.1, 900), (0.25, 990), (1.0, 1000), (2.5, 1000), (math.inf, 1000)]
    lent = [(0.1, 10), (0.25, 40), (1.0, 60), (2.5, 95), (math.inf, 100)]
    for q in (0.5, 0.9, 0.97, 0.99):
        _un_quantile(f, [rapide, lent], q, f"quantile_histogramme([rapide, lent], {q}) (rapide : 1 000 requêtes, lent : 100)")
    histogrammes = histogrammes_exemple()
    for q in (0.5, 0.9, 0.99, 0.999):
        _un_quantile(f, list(histogrammes.values()), q, f"quantile_histogramme(<les 3 histogrammes de mardi>, {q})")
        _un_quantile(f, [histogrammes["api-3"]], q, f"quantile_histogramme([<api-3>], {q})")
    vide = [(0.1, 0), (1.0, 0), (math.inf, 0)]
    for cas in ([], [vide], [vide, vide]):
        _un_quantile(f, cas, 0.99, f"quantile_histogramme({_court(cas, 90)}, 0.99)")
    seul = [(0.5, 0), (1.0, 4), (math.inf, 4)]
    _un_quantile(f, [seul], 0.25, f"quantile_histogramme([{seul}], 0.25)")
    for q in (0, 1.5, -0.1, True, "0.99", None, math.nan):
        _doit_lever(f, ([a], q), f"quantile_histogramme([a], {q!r})",
                    "q est un nombre (pas un bool) avec 0 < q <= 1 : 0.99 pour le p99, pas 99.")
    mauvais = [
        ([[(0.1, 5), (1.0, 8)]], "la dernière borne doit être math.inf"),
        ([[(math.inf, 5)]], "il faut au moins une borne finie"),
        ([[(0.0, 1), (1.0, 3), (math.inf, 3)]], "la première borne doit être > 0"),
        ([[(0.5, 1), (0.1, 3), (math.inf, 3)]], "les bornes doivent être strictement croissantes"),
        ([[(0.1, 1), (1.0, 3), (math.inf, 3)], [(0.1, 1), (0.5, 3), (math.inf, 3)]],
         "deux histogrammes aux bornes différentes ne s'additionnent pas"),
        ([[(0.1, 5), (1.0, 3), (math.inf, 6)]], "un compte cumulé ne baisse jamais (ces comptes ne sont pas cumulés)"),
        ([[(0.1, -1), (1.0, 3), (math.inf, 3)]], "un compte n'est jamais négatif"),
        ([[(0.1, math.nan), (1.0, 3), (math.inf, 3)]], "un compte NaN n'est pas un nombre de requêtes"),
        ([[(0.1, "1"), (1.0, 3), (math.inf, 3)]], "un compte est un nombre"),
    ]
    for histogrammes, conseil in mauvais:
        _doit_lever(f, (histogrammes, 0.5), f"quantile_histogramme({_court(histogrammes, 90)}, 0.5)", conseil + ".")
    alea = random.Random(219)
    for essai in range(200):
        histogrammes, q = _histogrammes_hasard(alea)
        _un_quantile(f, histogrammes, q, f"quantile_histogramme(<{len(histogrammes)} histogrammes tirés au hasard>, "
                                         f"{q}) (n°{essai})")


# ---------------------------------------------------------------------------- défi 220 : un compteur qui redémarre


def _ref_augmentation(echantillons, variante: str = ""):
    valides = []
    for t, v in echantillons:
        if v is None or v != v:
            if variante == "none_zero":
                valides.append((t, 0))
            continue
        valides.append((t, v))
    if len(valides) < 2:
        return None
    if variante == "dernier_moins_premier":
        total = valides[-1][1] - valides[0][1]
    else:
        total = 0
        for (_, avant), (_, apres) in zip(valides, valides[1:]):
            if variante == "abs":
                total += abs(apres - avant)
            elif apres >= avant:
                total += apres - avant
            elif variante != "reset_zero":
                total += apres
    if variante == "par_lecture":
        return total, total / (len(valides) - 1)
    if variante == "duree_totale":
        duree = echantillons[-1][0] - echantillons[0][0]
    else:
        duree = valides[-1][0] - valides[0][0]
    return total, (total / duree if duree else None)


_VARIANTES_AUGMENTATION = [
    ("dernier_moins_premier", "« la dernière lecture moins la première » ne marche que sans redémarrage. Additionnez "
                              "les augmentations entre lectures CONSÉCUTIVES, et quand le compteur BAISSE, il est reparti "
                              "de 0 : l'augmentation vaut la nouvelle valeur."),
    ("reset_zero", "quand le compteur baisse, le serveur a redémarré et le compteur est reparti de 0 : depuis, il a "
                   "compté `apres` requêtes. Compter 0 pour cet intervalle les perd."),
    ("abs", "une baisse n'est pas « autant de requêtes dans l'autre sens » : le compteur est reparti de 0, "
            "l'augmentation vaut la nouvelle valeur (pas la différence en valeur absolue)."),
    ("none_zero", "une lecture manquée (None ou NaN) n'est pas un 0 : ignorez-la complètement. La compter comme un 0 "
                  "fait croire à un redémarrage, et compte deux fois tout ce qui précède."),
    ("par_lecture", "par_seconde est un débit PAR SECONDE : total / (t de la dernière lecture valide − t de la "
                    "première), pas total / nombre d'intervalles."),
    ("duree_totale", "par_seconde se calcule entre la première et la dernière lecture VALIDE : une lecture manquée au "
                     "début ou à la fin n'a rien compté."),
]


def _une_augmentation(f, echantillons, quoi: str) -> None:
    avant = copy.deepcopy(echantillons)
    r = _appel(f, (echantillons,), quoi)
    _intact(avant, echantillons, quoi, "les lectures")
    attendu = _ref_augmentation(echantillons)
    if attendu is None:
        if r is not None:
            raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de None : avec moins de deux lectures valides, "
                                 "on ne peut rien dire.")
        return
    _pas_none(r, quoi)
    if not isinstance(r, tuple) or len(r) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : Hausse(total, par_seconde) est attendu.")
    if _memes(r, attendu):
        return
    for variante, message in _VARIANTES_AUGMENTATION:
        autre = _ref_augmentation(echantillons, variante)
        if autre is not None and _memes(r, autre) and not _memes(autre, attendu):
            raise AssertionError(f"{quoi} renvoie {_court(tuple(r), 80)} au lieu de {_court(attendu, 80)} : {message}")
    if not _meme(r[0], attendu[0]):
        raise AssertionError(f"{quoi} : total vaut {_fmt(r[0])} au lieu de {_fmt(attendu[0])}.")
    raise AssertionError(f"{quoi} : par_seconde vaut {_fmt(r[1])} au lieu de {_fmt(attendu[1])} (total / durée entre la "
                         "première et la dernière lecture valide).")


def _compteur_hasard(rng: random.Random) -> list:
    t, valeur, lectures = rng.choice([0, 1_700_000_000]), rng.randint(0, 10**6), []
    flottant = rng.random() < 0.2
    for _ in range(rng.randint(0, 40)):
        t += rng.choice([15, 15, 15, 30, 7.5])
        if rng.random() < 0.08:
            valeur = rng.randint(0, 60)                     # redémarrage
        else:
            valeur += rng.randint(0, 120)
        v = valeur + 0.25 if flottant else valeur
        if rng.random() < 0.08:
            v = rng.choice([None, math.nan])
        lectures.append((t, v))
    return lectures


def verifier_augmentation(f) -> None:
    doc = [[(0, 100), (15, 160), (30, 175), (45, 20), (60, 50)], [(0, 100), (15, None), (30, 130)],
           [(0, 100), (15, None)]]
    for echantillons in doc:
        _une_augmentation(f, echantillons, f"augmentation({_court(echantillons, 100)})")
    petits = [
        [], [(0, 5)], [(0, None), (15, 5)], [(0, 5), (15, 5)], [(0, 5), (15, 0)], [(0, 50), (15, 0), (30, 0)],
        [(0, 100), (15, 30), (30, 10), (45, 40)], [(0, None), (15, 100), (30, 130), (45, math.nan)],
        [(0, 100), (15, math.nan), (30, 60)], [(0, 10.5), (10, 12.25), (20, 3.5)], [(0, 0), (15, None), (30, None)],
        [(100, 7), (115, 9), (130, 2), (145, None), (160, 11)],
    ]
    for echantillons in petits:
        _une_augmentation(f, echantillons, f"augmentation({_court(echantillons, 100)})")
    _une_augmentation(f, compteur_exemple(), "augmentation(compteur_exemple())")
    for echantillons, conseil in [
        ([(0, 1), (0, 2)], "deux lectures au même instant : les t doivent être strictement croissants"),
        ([(15, 1), (0, 2)], "les t doivent être strictement croissants"),
        ([(0, 1), (15, None), (10, 2)], "les t doivent être strictement croissants, lectures manquées comprises"),
        ([(0, 1), (15, -3)], "un compteur ne vaut jamais un nombre négatif"),
        ([(0, 1), (15, "12")], "une valeur est un nombre (ou None pour une lecture manquée)"),
    ]:
        _doit_lever(f, (echantillons,), f"augmentation({_court(echantillons, 80)})", conseil + " → ValueError.")
    alea = random.Random(220)
    for essai in range(200):
        lectures = _compteur_hasard(alea)
        _une_augmentation(f, lectures, f"augmentation(<{len(lectures)} lectures tirées au hasard>) (n°{essai})")


# ---------------------------------------------------------------------------- défi 221 : budget d'erreur


def _ref_budget(minutes, objectif, variante: str = ""):
    requetes = sum(r for r, _ in minutes)
    erreurs = sum(e for _, e in minutes)
    if requetes == 0:
        return (0, 0, None, 0.0, None, None)
    if variante == "moyenne_minutes":
        taux = [e / r for r, e in minutes if r]
        disponibilite = 1 - sum(taux) / len(taux)
    elif variante == "temps":
        avec = [(r, e) for r, e in minutes if r]
        mauvaises = sum(1 for r, e in avec if e / r > 1 - objectif)
        disponibilite = 1 - mauvaises / len(avec)
    else:
        disponibilite = 1 - erreurs / requetes
    permises = objectif * requetes if variante == "objectif" else (1 - objectif) * requetes
    if variante == "inverse":
        consomme = permises / erreurs if erreurs else None
    else:
        consomme = erreurs / permises
    restant = permises - erreurs if variante == "restant_requetes" else (None if consomme is None else 1 - consomme)
    return (requetes, erreurs, disponibilite, permises, consomme, restant)


_VARIANTES_BUDGET = [
    ("moyenne_minutes", "la disponibilité est un rapport de SOMMES : 1 − (total des erreurs) / (total des requêtes). La "
                        "moyenne des taux des minutes donne autant de poids à une minute de nuit à une requête qu'à une "
                        "minute de midi à 300."),
    ("temps", "l'objectif porte sur des REQUÊTES (99.9 % des requêtes réussissent), pas sur des minutes : compter les "
              "minutes « en panne » est une autre définition."),
    ("objectif", "l'objectif dit ce qui doit RÉUSSIR ; le budget d'erreur, c'est le reste : (1 − objectif) × requetes."),
    ("inverse", "consomme = erreurs / erreurs_permises (la part du budget déjà brûlée), pas l'inverse."),
    ("restant_requetes", "restant est une FRACTION du budget : 1 − consomme (pas un nombre de requêtes)."),
]

_NOMS_BUDGET = ("requetes", "erreurs", "disponibilite", "erreurs_permises", "consomme", "restant")


def _un_budget(f, minutes, objectif, quoi: str, comme_defaut: bool = False) -> None:
    avant = copy.deepcopy(minutes)
    r = _appel(f, (minutes,) if comme_defaut else (minutes, objectif), quoi)
    _intact(avant, minutes, quoi, "les minutes")
    _pas_none(r, quoi)
    if not isinstance(r, tuple) or len(r) != 6:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : Budget(requetes, erreurs, disponibilite, "
                             "erreurs_permises, consomme, restant) est attendu.")
    attendu = _ref_budget(minutes, objectif)
    if _memes(r, attendu):
        return
    for variante, message in _VARIANTES_BUDGET:
        autre = _ref_budget(minutes, objectif, variante)
        if _memes(r, autre) and not _memes(autre, attendu):
            raise AssertionError(f"{quoi} : {message}")
    i = next(i for i in range(6) if not _meme(r[i], attendu[i]))
    if attendu[0] == 0:
        raise AssertionError(f"{quoi} : {_NOMS_BUDGET[i]} vaut {_fmt(r[i])} au lieu de {_fmt(attendu[i])}. Sans aucune "
                             "requête : Budget(0, 0, None, 0.0, None, None).")
    raise AssertionError(f"{quoi} : {_NOMS_BUDGET[i]} vaut {_fmt(r[i])} au lieu de {_fmt(attendu[i])}.")


def _minutes_hasard(rng: random.Random, n: int) -> list[tuple[int, int]]:
    minutes, taux = [], rng.choice([0.0, 0.001, 0.02, 0.2])
    for _ in range(n):
        if rng.random() < 0.1:
            taux = rng.choice([0.0, 0.0005, 0.005, 0.05, 0.3, 1.0])
        r = rng.choice([0, rng.randint(0, 5), rng.randint(0, 30), rng.randint(50, 400)])
        e = sum(1 for _ in range(r) if rng.random() < taux) if r <= 60 else min(r, round(r * taux * rng.uniform(0, 2)))
        minutes.append((r, e))
    return minutes


def verifier_budget_erreur(f) -> None:
    _un_budget(f, [(1000, 1), (2000, 0), (1, 1)], 0.999, "budget_erreur([(1000, 1), (2000, 0), (1, 1)], 0.999)")
    petits = [([], 0.999), ([(0, 0), (0, 0)], 0.99), ([(100, 0)], 0.99), ([(100, 5)], 0.99), ([(1, 1), (999, 0)], 0.999),
              ([(10, 2), (1000, 3)], 0.9), ([(300, 0), (2, 2), (300, 1)], 0.995), ([(5, 5)], 0.5)]
    for minutes, objectif in petits:
        _un_budget(f, minutes, objectif, f"budget_erreur({_court(minutes, 80)}, {objectif})")
    _un_budget(f, [(400, 1), (3, 1)], OBJECTIF, "budget_erreur([(400, 1), (3, 1)]) (objectif par défaut)",
               comme_defaut=True)
    _un_budget(f, minutes_exemple(), OBJECTIF, "budget_erreur(minutes_exemple())", comme_defaut=True)
    for objectif in (99.9, 0, 1, 1.5, -0.5, True, "0.999", None, math.nan):
        _doit_lever(f, ([(10, 0)], objectif), f"budget_erreur([(10, 0)], {objectif!r})",
                    "l'objectif est une fraction strictement entre 0 et 1 (0.999, pas 99.9), et pas un bool.")
    for minute in ((5, 6), (-1, 0), (5, -1), (5.0, 1), (True, 0), (None, 0), (5, "1")):
        _doit_lever(f, ([(10, 0), minute],), f"budget_erreur([(10, 0), {minute!r}])",
                    "chaque minute est (requetes, erreurs) : deux int (pas des bool) avec 0 <= erreurs <= requetes.")
    alea = random.Random(221)
    for essai in range(150):
        minutes = _minutes_hasard(alea, alea.randint(0, 80))
        objectif = alea.choice([0.9, 0.99, 0.995, 0.999, 0.9999])
        _un_budget(f, minutes, objectif, f"budget_erreur(<{len(minutes)} minutes tirées au hasard>, {objectif}) "
                                         f"(n°{essai})")


# ---------------------------------------------------------------------------- défi 222 : alertes


def _combustions(minutes, objectif, variante: str = ""):
    """combustion(t, w) avec des sommes cumulées ; `variante` rejoue une erreur classique."""
    permis = 1 - objectif
    cr, ce, cq, cn = [0], [0], [0.0], [0]
    for r, e in minutes:
        cr.append(cr[-1] + r)
        ce.append(ce[-1] + e)
        cq.append(cq[-1] + (e / r if r else 0.0))
        cn.append(cn[-1] + (1 if r else 0))

    def combustion(t: int, w: int) -> float:
        if variante == "decalee":
            a, b = max(0, t - w), t
        elif variante == "trop_large":
            a, b = max(0, t - w), t + 1
        else:
            a, b = max(0, t - w + 1), t + 1
        if variante == "fenetres_pleines" and t - w + 1 < 0:
            return 0.0
        if variante == "moyenne":
            k = cn[b] - cn[a]
            if not k:
                return 0.0
            ratio = (cq[b] - cq[a]) / k
        else:
            requetes = cr[b] - cr[a]
            if not requetes:
                return 0.0
            ratio = (ce[b] - ce[a]) / requetes
        return ratio if variante == "sans_objectif" else ratio / permis

    return combustion


def _ref_alertes(minutes, objectif, regles, variante: str = "") -> list[tuple]:
    combustion = _combustions(minutes, objectif, variante)
    trouvees = []
    for rang, (nom, longue, courte, seuil) in enumerate(regles):
        debut = None
        for t in range(len(minutes) + 1):
            if t == len(minutes):
                allumee = False
            elif variante == "longue_seule":
                allumee = combustion(t, longue) >= seuil
            elif variante == "courte_seule":
                allumee = combustion(t, courte) >= seuil
            else:
                allumee = combustion(t, longue) >= seuil and combustion(t, courte) >= seuil
            if variante == "par_minute" and allumee:
                trouvees.append((t, rang, (nom, t, t)))
            elif allumee and debut is None:
                debut = t
            elif not allumee and debut is not None:
                trouvees.append((debut, rang, (nom, debut, t - 1)))
                debut = None
    cle = (lambda x: (x[1], x[0])) if variante == "ordre_regles" else (lambda x: (x[0], x[1]))
    return [a for _, _, a in sorted(trouvees, key=cle)]


_VARIANTES_ALERTES = [
    ("longue_seule", "une règle s'allume quand la fenêtre longue ET la fenêtre courte brûlent toutes deux au moins "
                     "`seuil`. Avec la seule fenêtre longue, l'alerte reste allumée longtemps après la fin de l'incident "
                     "(la fenêtre courte dit « c'est encore en cours »)."),
    ("moyenne", "la combustion d'une fenêtre est un rapport de SOMMES : (erreurs de la fenêtre / requêtes de la "
                "fenêtre) / (1 − objectif), pas la moyenne des taux des minutes."),
    ("courte_seule", "une règle s'allume quand la fenêtre longue ET la fenêtre courte brûlent toutes deux au moins "
                     "`seuil`. Avec la seule fenêtre courte, une ou deux erreurs parmi quelques requêtes de nuit "
                     "réveillent quelqu'un (la fenêtre longue dit « c'est grave »)."),
    ("fenetres_pleines", "au début, la fenêtre prend les minutes qui existent (max(0, t − W + 1) à t) : un incident "
                         "dans la première heure doit aussi déclencher l'alerte."),
    ("sans_objectif", "la combustion compare le taux d'erreur à ce que le budget permet : divisez-le par (1 − objectif). "
                      "Un taux de 2 % avec un objectif de 99.9 %, c'est une combustion de 20."),
    ("decalee", "la fenêtre de W minutes qui finit à la minute t INCLUT la minute t : ce sont les minutes t − W + 1 à t."),
    ("trop_large", "la fenêtre de W minutes qui finit à la minute t contient W minutes : t − W + 1 à t (pas t − W à t)."),
    ("par_minute", "une alerte couvre une période de minutes CONSÉCUTIVES allumées : Alerte(nom, debut, fin), pas une "
                   "alerte par minute."),
    ("ordre_regles", "les alertes sont triées par debut (puis, à debut égal, dans l'ordre des règles), pas règle par "
                     "règle."),
]


def _lire_alertes(r, quoi: str) -> list[tuple]:
    _pas_none(r, quoi)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une liste d'Alerte(regle, debut, fin) est attendue.")
    propres = []
    for a in r:
        if not isinstance(a, tuple) or len(a) != 3:
            raise AssertionError(f"{quoi} contient {_court(a, 80)} : chaque élément est une Alerte(regle, debut, fin).")
        propres.append(tuple(a))
    return propres


def _des_alertes(f, minutes, objectif, regles, quoi: str, kwargs: dict | None = None, lignes_max=2_000_000,
                 a_lire=None) -> None:
    avant = copy.deepcopy(minutes)
    args = (a_lire if a_lire is not None else minutes,) if kwargs is not None else (minutes, objectif, regles)
    r = _appel(f, args, quoi, kwargs=kwargs, lignes_max=lignes_max,
               conseil_lent="Lisez chaque minute UNE fois : avec des sommes cumulées, la somme de n'importe quelle fenêtre "
                            "se calcule en une soustraction.")
    _intact(avant, minutes, quoi, "les minutes")
    obtenu = _lire_alertes(r, quoi)
    attendu = _ref_alertes(minutes, objectif, regles)
    if obtenu == attendu:
        return
    for variante, message in _VARIANTES_ALERTES:
        autre = _ref_alertes(minutes, objectif, regles, variante)
        if obtenu == autre and autre != attendu:
            raise AssertionError(f"{quoi} : {message}")
    manque = [a for a in attendu if a not in obtenu]
    trop = [a for a in obtenu if a not in attendu]
    if not manque and not trop:
        raise AssertionError(f"{quoi} : les bonnes alertes, mais dans l'ordre {_court(obtenu, 120)} au lieu de "
                             f"{_court(attendu, 120)} (triées par debut, puis dans l'ordre des règles).")
    raise AssertionError(f"{quoi} renvoie {_court(obtenu, 120)} au lieu de {_court(attendu, 120)}"
                         f"{' (il manque ' + _court(manque, 80) + ')' if manque else ''}"
                         f"{' (en trop : ' + _court(trop, 80) + ')' if trop else ''}.")


def _presque_a_egalite(minutes, objectif, regles) -> bool:
    """Une combustion à 1e-9 près de son seuil : deux écritures justes pourraient ne pas trancher pareil."""
    combustion = _combustions(minutes, objectif)
    for _, longue, courte, seuil in regles:
        for t in range(len(minutes)):
            for w in (longue, courte):
                if abs(combustion(t, w) - seuil) <= 1e-9 * seuil:
                    return True
    return False


def _regles_hasard(rng: random.Random) -> list[Regle]:
    regles = []
    for i in range(rng.randint(1, 3)):
        longue = rng.randint(1, 30)
        regles.append(Regle(f"r{i}", longue, rng.randint(1, longue), rng.choice([0.7, 1.3, 2.9, 6.1, 14.3, 77.7])))
    return regles


def verifier_alertes(f) -> None:
    doc = [(100, 0)] * 10 + [(100, 20)] * 3 + [(100, 0)] * 10
    _des_alertes(f, doc, 0.99, [Regle("page", 6, 2, 8.0)],
                 "alertes(<10 minutes propres, 3 à 20 %, 10 propres>, 0.99, [Regle('page', 6, 2, 8.0)])")
    # Des cas construits pour que chaque erreur classique donne un résultat différent.
    construits = [
        ([(100, 0)] * 20 + [(100, 10)] * 4 + [(100, 0)] * 12, 0.99, [Regle("p", 10, 2, 2.5)]),
        ([(200, 0)] * 8 + [(2, 2)] + [(200, 0)] * 8, 0.99, [Regle("p", 6, 1, 5.5)]),
        ([(100, 30)] * 3 + [(100, 0)] * 10, 0.99, [Regle("p", 6, 3, 9.5)]),
        ([(100, 0)] * 5 + [(100, 3)] * 6 + [(0, 0)] * 3 + [(100, 3)] * 2, 0.99, [Regle("a", 4, 2, 2.5), Regle("b", 8, 1, 1.5)]),
        ([(50, 0)] * 6 + [(50, 5), (50, 0), (50, 5)] + [(50, 0)] * 6, 0.98, [Regle("p", 3, 1, 3.1), Regle("t", 9, 3, 1.1)]),
        ([], 0.999, list(REGLES)),
        ([(0, 0)] * 30, 0.999, list(REGLES)),
    ]
    for minutes, objectif, regles in construits:
        _des_alertes(f, minutes, objectif, regles, f"alertes({_court(minutes, 70)}, {objectif}, {_court(regles, 70)})")
    # 14 400 minutes (dix jours), lues par une Comptee : relire chaque fenêtre à chaque minute se voit.
    longues = generer_minutes(14_400, graine=222, incidents=((3_000, 3_040, 0.08), (9_000, 9_900, 0.006)))
    plafond = 30 * len(longues) + 1_000
    comptee = _Plafonnee(longues, plafond)
    quoi = "alertes(<14 400 minutes dans une Comptee>) (règles et objectif par défaut)"
    try:
        _des_alertes(f, longues, OBJECTIF, REGLES, quoi, kwargs={}, lignes_max=6_000_000, a_lire=comptee)
    except AssertionError as exc:
        if comptee.lues > plafond:
            raise AssertionError(f"{quoi} a lu plus de {plafond} cases ({len(longues)} minutes) avant d'être arrêtée : vous "
                                 "relisez chaque fenêtre à chaque minute. Lisez chaque minute UNE fois, et calculez les "
                                 "sommes des fenêtres avec des sommes cumulées (`cumul[t + 1] - cumul[a]`).") from None
        raise exc
    _des_alertes(f, minutes_exemple(), OBJECTIF, REGLES, "alertes(minutes_exemple())", kwargs={}, lignes_max=None)
    for regles in ([Regle("x", 5, 10, 2.0)], [Regle("x", 0, 0, 2.0)], [Regle("x", 10, 5, 0)], [Regle("x", 10, 5, -1.0)],
                   [Regle("x", 10.0, 5, 2.0)], [Regle("x", True, True, 2.0)], [Regle("x", 10, 5, "2")]):
        _doit_lever(f, ([(10, 0)], 0.999, regles), f"alertes([(10, 0)], 0.999, {_court(regles, 70)})",
                    "dans une règle, longue et courte sont des int (pas des bool) >= 1 avec courte <= longue, et le seuil "
                    "un nombre > 0.")
    for objectif in (99.9, 0, 1, True):
        _doit_lever(f, ([(10, 0)], objectif, list(REGLES)), f"alertes([(10, 0)], {objectif!r})",
                    "l'objectif est une fraction strictement entre 0 et 1, comme au défi 221.")
    for minute in ((5, 6), (-1, 0), (5.0, 1), (None, 0)):
        _doit_lever(f, ([(10, 0), minute], 0.999, list(REGLES)), f"alertes([(10, 0), {minute!r}])",
                    "chaque minute est (requetes, erreurs) : deux int avec 0 <= erreurs <= requetes, comme au défi 221.")
    alea = random.Random(222)
    essai = 0
    while essai < 120:
        minutes = _minutes_hasard(alea, alea.randint(0, 120))
        objectif, regles = alea.choice([0.9, 0.99, 0.999]), _regles_hasard(alea)
        if _presque_a_egalite(minutes, objectif, regles):
            continue
        _des_alertes(f, minutes, objectif, regles, f"alertes(<{len(minutes)} minutes tirées au hasard>, {objectif}, "
                                                   f"{_court(regles, 80)}) (n°{essai})")
        essai += 1


# ---------------------------------------------------------------------------- défi 223 : le temps propre


def _ref_temps_propre(spans, variante: str = "") -> dict:
    enfants: dict = {}
    for s in spans:
        enfants.setdefault(s.parent, []).append(s)
    gardes = spans
    if variante == "orphelins":                              # parcours depuis les racines : les orphelins se perdent
        atteints, pile = set(), [s for s in spans if s.parent is None]
        while pile:
            s = pile.pop()
            atteints.add(s.id)
            pile.extend(enfants.get(s.id, ()))
        gardes = [s for s in spans if s.id in atteints]
    resultat: dict = {}
    for s in gardes:
        if variante == "total":
            couvert = 0
        elif variante == "somme":
            couvert = sum(e.fin - e.debut for e in enfants.get(s.id, ()))
        elif variante == "somme_coupee":
            couvert = sum(max(0, min(e.fin, s.fin) - max(e.debut, s.debut)) for e in enfants.get(s.id, ()))
        else:
            if variante == "sans_coupe":
                morceaux = sorted((e.debut, e.fin) for e in enfants.get(s.id, ()))
            else:
                morceaux = sorted((max(e.debut, s.debut), min(e.fin, s.fin)) for e in enfants.get(s.id, ()))
            couvert, jusqu_a = 0, (-math.inf if variante == "sans_coupe" else s.debut)
            for debut, fin in morceaux:
                debut = max(debut, jusqu_a)
                if fin > debut:
                    couvert += fin - debut
                    jusqu_a = fin
        resultat[s.service] = resultat.get(s.service, 0) + (s.fin - s.debut) - couvert
    if variante == "tri":
        return dict(sorted(resultat.items()))
    return resultat


_VARIANTES_TEMPS = [
    ("somme", "des enfants qui tournent EN PARALLÈLE ne s'additionnent pas : retirez la longueur de la RÉUNION de leurs "
              "intervalles (triez-les par début, puis fusionnez ceux qui se chevauchent), pas la somme de leurs durées."),
    ("somme_coupee", "des enfants qui tournent EN PARALLÈLE ne s'additionnent pas : retirez la longueur de la RÉUNION de "
                     "leurs intervalles (triez-les par début, puis fusionnez ceux qui se chevauchent)."),
    ("sans_coupe", "un enfant qui commence avant son parent ou finit après lui (un audit lancé sans l'attendre) ne compte "
                   "que pour la partie COMMUNE : coupez chaque intervalle à [max(debut_enfant, debut), min(fin_enfant, fin)]."),
    ("total", "le temps PROPRE d'un span, c'est sa durée MOINS le temps où il attendait ses enfants. Vous renvoyez la "
              "durée totale : la passerelle aurait alors « travaillé » pendant toute la requête."),
    ("orphelins", "un span dont le parent n'est pas dans la trace (orphelin) compte comme une racine : son temps propre, "
                  "et celui de ses enfants, comptent. En partant des seules racines (parent None), vous les perdez."),
    ("tri", "le dict est rangé dans l'ordre où chaque service APPARAÎT dans `spans`, pas trié."),
]


def _lire_temps(r, quoi: str) -> dict:
    _pas_none(r, quoi)
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : un dict {{service: temps propre en ms}} est attendu.")
    for service, valeur in r.items():
        if not _nombre(valeur):
            raise AssertionError(f"Dans {quoi}, le service {service!r} vaut {_court(valeur, 60)} : un nombre (des ms) est "
                                 "attendu.")
    return r


def _memes_temps(a: dict, b: dict) -> bool:
    return list(a) == list(b) and all(_meme(a[k], b[k]) for k in a)


def _un_temps(f, spans, quoi: str, lignes_max: int | None = 2_000_000) -> None:
    avant = list(spans)
    r = _appel(f, (spans,), quoi, lignes_max=lignes_max,
               conseil_lent="Ne cherchez pas les enfants de chaque span dans toute la liste : groupez-les par parent dans "
                            "un dict, en une passe.")
    _intact(avant, spans, quoi, "les spans")
    obtenu = _lire_temps(r, quoi)
    attendu = _ref_temps_propre(spans)
    if _memes_temps(obtenu, attendu):
        return
    for variante, message in _VARIANTES_TEMPS:
        autre = _ref_temps_propre(spans, variante)
        if _memes_temps(obtenu, autre) and not _memes_temps(autre, attendu):
            raise AssertionError(f"{quoi} : {message}")
    if set(obtenu) != set(attendu):
        raise AssertionError(f"{quoi} : services {_court(list(obtenu), 100)} au lieu de {_court(list(attendu), 100)}.")
    if list(obtenu) != list(attendu):
        raise AssertionError(f"{quoi} range les services dans l'ordre {_court(list(obtenu), 100)} au lieu de "
                             f"{_court(list(attendu), 100)} : l'ordre où chaque service APPARAÎT dans `spans`.")
    service = next(k for k in attendu if not _meme(obtenu[k], attendu[k]))
    raise AssertionError(f"{quoi} : le service {service!r} a un temps propre de {_fmt(obtenu[service])} ms au lieu de "
                         f"{_fmt(attendu[service])}.")


def _grande_trace() -> list[Span]:
    """40 000 spans : une racine, un span à 20 000 enfants, une chaîne de 5 000 appels imbriqués, et du reste."""
    spans = [Span("r", None, "passerelle", 0, 100_000), Span("lot", "r", "api", 10, 90_000)]
    spans += [Span(f"e{i}", "lot", "crm", 10 + (i * 7) % 80_000, 10 + (i * 7) % 80_000 + 40) for i in range(20_000)]
    spans += [Span(f"c{i}", f"c{i - 1}" if i else "r", "base", 90_000 + i, 100_000 - i) for i in range(5_000)]
    spans += [Span(f"x{s.id}", s.id, "cache", s.debut, s.debut + 5) for s in generer_trace(14_998, graine=223)]
    return spans


def verifier_temps_propre(f) -> None:
    doc = [Span("a", None, "api", 0, 100), Span("b", "a", "crm", 10, 70), Span("c", "a", "crm", 20, 80),
           Span("d", "a", "audit", 90, 160), Span("e", "zz", "file", 0, 30)]
    _un_temps(f, doc, "temps_propre(l'exemple de la docstring)")
    petits = [
        [], [Span("a", None, "api", 5, 5)], [Span("a", None, "api", 0, 10), Span("b", "a", "crm", 2, 4)],
        [Span("b", "a", "crm", 2, 4), Span("a", None, "api", 0, 10)],
        [Span("a", None, "api", 0, 10), Span("b", "a", "crm", 2, 6), Span("c", "a", "crm", 4, 8), Span("d", "a", "crm", 5, 7)],
        [Span("a", None, "api", 0, 10), Span("b", "a", "crm", -5, 3), Span("c", "a", "crm", 8, 30)],
        [Span("a", None, "api", 0, 10), Span("b", "a", "crm", 20, 30)],
        [Span("a", None, "api", 0, 10), Span("b", "a", "crm", 2, 8), Span("c", "b", "base", 1, 9)],
        [Span("a", "perdu", "api", 0, 10), Span("b", "a", "crm", 2, 8), Span("c", None, "file", 0, 3)],
        [Span("a", None, "z", 0, 10.5), Span("b", "a", "a", 0.25, 1.75), Span("c", "a", "m", 1.5, 2.0)],
    ]
    for spans in petits:
        _un_temps(f, spans, f"temps_propre({_court(spans, 110)})")
    _un_temps(f, trace_exemple(), "temps_propre(trace_exemple())")
    for spans, conseil in [
        ([Span("a", None, "api", 10, 5)], "un span finit après avoir commencé : fin < debut"),
        ([Span("a", None, "api", 0, 10), Span("a", None, "crm", 0, 5)], "deux spans ont le même id"),
        ([Span("a", None, "api", None, 10)], "debut et fin sont des nombres"),
        ([Span("a", None, "api", 0, math.nan)], "debut et fin sont des nombres (NaN n'en est pas un utilisable)"),
    ]:
        _doit_lever(f, (spans,), f"temps_propre({_court(spans, 90)})", conseil + " → ValueError.")
    grande = _grande_trace()
    _un_temps(f, grande, "temps_propre(<40 000 spans, dont un à 20 000 enfants et une chaîne de 5 000 appels>)",
              lignes_max=4_000_000)
    alea = random.Random(223)
    for essai in range(150):
        spans = generer_trace(alea.randint(0, 60), graine=alea.randint(0, 10**6), paralleles=alea.random(),
                              debordements=alea.random() * 0.4, orphelins=alea.random() * 0.2,
                              entiers=alea.random() < 0.7)
        _un_temps(f, spans, f"temps_propre(<{len(spans)} spans tirés au hasard>) (n°{essai})")


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"lire_journal": 218, "quantile_histogramme": 219, "augmentation": 220, "budget_erreur": 221,
            "alertes": 222, "temps_propre": 223}

SERIE = Serie(
    code="36",
    titre="Savoir si le service va bien : journal, quantile d'histogramme, compteur, budget d'erreur, alertes, traces",
    module_eleve="defis.serie_36",
    module_corrige="defis.corrige_serie_36",
    defis=(
        Defi(218, "lire_journal", "🟢", "6.4", verifier_lire_journal),
        Defi(219, "quantile_histogramme", "🟡", "—", verifier_quantile_histogramme),
        Defi(220, "augmentation", "🟡", "—", verifier_augmentation),
        Defi(221, "budget_erreur", "🟢", "7.6", verifier_budget_erreur),
        Defi(222, "alertes", "🔴", "7.6", verifier_alertes),
        Defi(223, "temps_propre", "🔴", "—", verifier_temps_propre),
    ),
)

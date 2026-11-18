"""Vérifications automatiques de la série 33 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est Mariam, qui dirige le centre d'appels de rétention et recrute ses conseillers trois mois
à l'avance sur la foi de vos prévisions, et le serveur qui compte ce que chaque calcul lui coûte :

- aux défis 202 et 203, il compte les lignes exécutées dans votre fichier sur 20 000 mois : relisser tout le
  passé pour chaque mois se voit, quelle que soit la machine ;
- aux défis 204 et 205, il vous donne une méthode ESPIONNE : elle note ce qu'elle reçoit. Un passé qui
  contient le mois à prévoir (ou toute la série), un horizon qui n'est pas le bon, deux appels pour la même
  origine se voient ;
- partout, il rejoue les erreurs classiques (cycle lu à l'envers ou décalé d'un mois, biais dans le mauvais
  sens, racine prise avant la moyenne, MAPE divisée par la prévision ou en pourcentage, alpha appliqué à
  l'ancien niveau, prévision faite APRÈS avoir vu le mois, niveau de départ égal à la moyenne de toute la
  série, ancien niveau écrasé avant de calculer la tendance, h oublié dans niveau + h × tendance, dernière
  origine oubliée, prévision à 1 mois comparée au mois visé à 3 mois, valeur absolue des erreurs, quantile
  interpolé, plafond calculé en float…) pour nommer la vôtre ;
- et il vérifie que vos fonctions ne modifient pas ce qu'elles reçoivent.

Les valeurs qu'il tire au hasard sont des multiples de 1/16 : leurs différences sont EXACTES. Les lissages, eux,
multiplient par alpha : il compare donc les nombres à 1e-9 près, et ne vous reproche jamais un arrondi.
"""

from __future__ import annotations

import copy
import math
import random
import sys

from .previsions import departs_exemple, generer_serie
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
    if not _egaux_nan(avant, apres):
        raise AssertionError(f"{quoi} a modifié {nom} reçu(s) (avant : {_court(avant, 100)}, après : "
                             f"{_court(apres, 100)}). Une fonction ne touche pas à ce qu'on lui prête : travaillez sur "
                             "une copie, ou construisez un nouvel objet.")


def _egaux_nan(a, b) -> bool:
    """a == b, en considérant deux NaN comme égaux (une liste qui contient NaN n'est pas égale à sa copie)."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_egaux_nan(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


def _pas_none(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")


def _nombre(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _proche(x, y, tol: float = 1e-9) -> bool:
    return _nombre(x) and _nombre(y) and math.isclose(x, y, rel_tol=tol, abs_tol=tol)


def _memes(r, attendu) -> bool:
    """Deux suites de nombres (ou de None) égales à 1e-9 près."""
    try:
        return len(r) == len(attendu) and all((x is None and y is None) or _proche(x, y) for x, y in zip(r, attendu))
    except TypeError:
        return False


def _seize(rng: random.Random, bas: int = 0, haut: int = 16 * 800) -> float:
    """Un multiple de 1/16 entre bas/16 et haut/16 : exact en virgule flottante."""
    return rng.randint(bas, haut) / 16


def _serie_hasard(rng: random.Random, n: int) -> list[float]:
    """Une série de n mois : tantôt avec tendance et saison, tantôt sans rien, tantôt avec des entiers."""
    genre = rng.random()
    if genre < 0.5:
        return generer_serie(n, graine=rng.randint(0, 10_000), saison=rng.choice([1, 3, 4, 12]),
                             tendance=rng.choice([-2.0, 0.0, 0.5, 3.0]))
    if genre < 0.8:
        return [_seize(rng) for _ in range(n)]
    return [rng.randint(0, 60) for _ in range(n)]


def _fmt(x) -> str:
    return f"{x:.6g}" if _nombre(x) else repr(x)


# ---------------------------------------------------------------------------- défi 200 : prévision naïve


def _ref_naive(historique, horizon: int, saison: int = 1, variante: str = "") -> list[float]:
    valeurs = list(historique)
    cycle = valeurs[len(valeurs) - saison:]
    if variante == "a_rebours":                     # historique[-h] : le cycle lu à l'envers
        return [float(cycle[-1 - h % saison]) for h in range(horizon)]
    if variante == "decale":                        # historique[len - saison + h] avec h = 1, 2, … : un mois trop loin
        return [float(cycle[(h + 1) % saison]) for h in range(horizon)]
    if variante == "constante":                     # [historique[-saison]] * horizon
        return [float(cycle[0])] * horizon
    if variante == "dernier":                       # saison ignorée
        return [float(valeurs[-1])] * horizon
    if variante == "moyenne":
        return [sum(cycle) / saison] * horizon
    if variante == "premier_cycle":                 # le PREMIER cycle de l'historique, pas le dernier
        return [float(valeurs[h % saison]) for h in range(horizon)]
    return [float(cycle[h % saison]) for h in range(horizon)]


_VARIANTES_NAIVE = [
    ("a_rebours", "le cycle se relit dans l'ordre du CALENDRIER : le 1er mois prévu reprend la 1re valeur du "
                  "dernier cycle (historique[len - saison]), pas la dernière. `historique[-h]` lit à l'envers."),
    ("decale", "le 1er mois prévu (h = 0) reprend la 1re valeur du dernier cycle : vous commencez un mois trop "
               "loin (h = 1, 2, … au lieu de 0, 1, …)."),
    ("constante", "chaque mois prévu reprend SON mois du cycle (h % saison), pas toujours le premier."),
    ("dernier", "avec une saison de 12 mois, janvier se prévoit avec le janvier précédent, pas avec le dernier mois "
                "connu : `saison` est ignoré."),
    ("moyenne", "la prévision naïve RECOPIE le même mois du dernier cycle ; elle ne fait pas la moyenne du cycle."),
    ("premier_cycle", "on reprend le DERNIER cycle (les `saison` dernières valeurs), le plus récent, pas le premier "
                      "de l'historique."),
]

_CONSEIL_200 = ("Au-delà d'un cycle, on recommence : le mois prévu n°h (en comptant à partir de 0) reprend la valeur "
                "n°h % saison du dernier cycle.")


def _une_naive(f, historique, horizon: int, saison: int | None, quoi: str) -> None:
    avant = copy.deepcopy(historique)
    args = (historique, horizon) if saison is None else (historique, horizon, saison)
    r = _appel(f, args, quoi, _CONSEIL_200)
    _intact(avant, historique, quoi, "l'historique")
    _pas_none(r, quoi)
    s = 1 if saison is None else saison
    attendu = _ref_naive(historique, horizon, s)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list de {horizon} prévisions est attendue.")
    if len(r) != horizon:
        raise AssertionError(f"{quoi} renvoie {len(r)} prévisions au lieu de {horizon} (une par mois de l'horizon).")
    for x in r:
        if not _nombre(x):
            raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : des nombres (float) sont attendus.")
    if _memes(r, attendu):
        return
    for variante, message in _VARIANTES_NAIVE:
        autre = _ref_naive(historique, horizon, s, variante)
        if _memes(r, autre) and not _memes(autre, attendu):
            raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)} : {message}")
    h = next(h for h in range(horizon) if not _proche(r[h], attendu[h]))
    raise AssertionError(f"Dans {quoi}, le mois prévu n°{h} vaut {_fmt(r[h])} au lieu de {_fmt(attendu[h])} (la valeur "
                         f"n°{h % s} du dernier cycle de {s} mois).")


def verifier_prevision_naive(f) -> None:
    departs = departs_exemple()
    cas = [
        ([10, 20, 30, 40, 50, 60], 6, 4),
        (departs, 3, None),
        (departs, 3, 12),
        (departs, 15, 12),                         # au-delà d'un an : on recommence le cycle
        ([1.0, 2.0, 3.0], 2, None),
        ([7.5], 3, 1),
        ([1.0, 2.0, 3.0], 0, 2),
        ([4.0, 9.0, 1.0], 7, 3),                   # exactement un cycle connu
        ([5.0, 3.0, 8.0, 2.0, 6.0], 4, 2),
    ]
    for historique, horizon, saison in cas:
        appel = f"{horizon}" if saison is None else f"{horizon}, saison={saison}"
        _une_naive(f, historique, horizon, saison, f"prevision_naive({_court(historique, 60)}, {appel})")
    for historique, horizon, saison, pourquoi in (
            ([1.0, 2.0], -1, 1, "horizon < 0"), ([1.0, 2.0], 2, 0, "saison < 1"), ([1.0, 2.0], 2, -3, "saison < 1"),
            ([1.0, 2.0], 2, 3, "moins d'un cycle connu"), ([], 1, 1, "moins d'un cycle connu"),
            ([1.0, None, 3.0], 1, 1, "une valeur manquante"), ([float("nan"), 2.0], 1, 1, "une valeur manquante"),
            ([None], 1, 1, "une valeur manquante")):
        _doit_lever(f, (historique, horizon, saison), f"prevision_naive({historique}, {horizon}, saison={saison})",
                    f"Ici : {pourquoi}. Horizon négatif, saison < 1, moins de `saison` valeurs connues, ou une valeur "
                    "manquante (None ou NaN : `x is None or x != x`) → ValueError.")
    alea = random.Random(200)
    for essai in range(200):
        saison = alea.choice([1, 1, 2, 3, 4, 6, 12])
        historique = _serie_hasard(alea, alea.randint(saison, 40))
        horizon = alea.randint(0, 30)
        _une_naive(f, historique, horizon, saison, f"prevision_naive({_court(historique, 60)}, {horizon}, "
                                                   f"saison={saison}) (tiré au hasard n°{essai})")


# ---------------------------------------------------------------------------- défi 201 : erreurs


def _ref_erreurs(reels, prevus, variante: str = "") -> tuple:
    n = len(reels)
    ecarts = [p - r for r, p in zip(reels, prevus)]
    mae = sum(abs(e) for e in ecarts) / n
    rmse = math.sqrt(sum(e * e for e in ecarts) / n)
    biais = sum(ecarts) / n
    zero = any(r == 0 for r in reels)
    mape = None if zero else sum(abs(e) / abs(r) for e, r in zip(ecarts, reels)) / n
    if variante == "biais_inverse":
        biais = -biais
    elif variante == "mae_signee":
        mae = biais
    elif variante == "mse":
        rmse = sum(e * e for e in ecarts) / n
    elif variante == "racine_avant":
        rmse = math.sqrt(sum(e * e for e in ecarts)) / n
    elif variante == "rmse_est_mae":
        rmse = mae
    elif variante == "mape_prevu":
        mape = None if any(p == 0 for p in prevus) or zero else sum(abs(e) / abs(p) for e, p in zip(ecarts, prevus)) / n
    elif variante == "mape_pourcent":
        mape = None if zero else 100 * mape
    elif variante == "mape_sans_zeros":
        gardes = [(e, r) for e, r in zip(ecarts, reels) if r != 0]
        mape = sum(abs(e) / abs(r) for e, r in gardes) / len(gardes) if gardes else None
    elif variante == "mape_signee":
        mape = None if zero else sum(abs(e) / r for e, r in zip(ecarts, reels)) / n
    return mae, rmse, biais, mape


_VARIANTES_ERREURS = [
    ("biais_inverse", "biais", "le biais est la moyenne de prévu − réel (positif = on prévoit TROP). Vous calculez "
                               "réel − prévu : le signe est inversé, et Mariam recruterait dans le mauvais sens."),
    ("mae_signee", "mae", "la MAE est la moyenne des valeurs ABSOLUES des écarts : sans `abs`, +3 et −4 se compensent "
                          "et une méthode qui se trompe beaucoup peut afficher une MAE presque nulle."),
    ("mse", "rmse", "la RMSE est la RACINE de la moyenne des carrés : vous renvoyez la moyenne des carrés (la MSE), "
                    "qui n'est pas dans l'unité des données (des départs au carré)."),
    ("racine_avant", "rmse", "la RMSE est la racine de la MOYENNE des carrés : divisez par n AVANT de prendre la racine "
                             "(`math.sqrt(somme / n)`, pas `math.sqrt(somme) / n`)."),
    ("rmse_est_mae", "rmse", "la RMSE est la racine de la moyenne des CARRÉS. Prendre la racine de chaque carré puis la "
                             "moyenne, c'est la MAE."),
    ("mape_prevu", "mape", "la MAPE divise chaque erreur par le RÉEL, pas par la prévision : « on s'est trompé de 10 % "
                           "de ce qui s'est vraiment passé »."),
    ("mape_pourcent", "mape", "la MAPE est une fraction : 0.25 pour 25 %. Ne multipliez pas par 100."),
    ("mape_sans_zeros", "mape", "un réel vaut 0 : la MAPE n'existe pas, renvoyez None. Faire la moyenne sur les autres "
                                "mois cache le problème à celui qui lit le rapport."),
    ("mape_signee", "mape", "divisez par |réel| (`abs(r)`) : avec un réel négatif, une erreur deviendrait négative."),
]

_NOMS_ERREURS = ("mae", "rmse", "biais", "mape")


def _une_erreurs(f, reels, prevus, quoi: str) -> None:
    copies = copy.deepcopy((reels, prevus))
    r = _appel(f, (reels, prevus), quoi)
    _intact(copies[0], reels, quoi, "les réels")
    _intact(copies[1], prevus, quoi, "les prévisions")
    _pas_none(r, quoi)
    attendu = _ref_erreurs(reels, prevus)
    if not isinstance(r, tuple) or len(r) != 4:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : un Erreurs(mae, rmse, biais, mape) (ou un tuple de 4 "
                             "valeurs dans cet ordre) est attendu.")
    for nom, x in zip(_NOMS_ERREURS, r):
        if not (_nombre(x) or (nom == "mape" and x is None)):
            raise AssertionError(f"Dans {quoi}, {nom} vaut {_court(x, 60)} : un nombre est attendu"
                                 f"{' (ou None pour la MAPE)' if nom == 'mape' else ''}.")
    if _memes(r, attendu):
        return
    if attendu[3] is not None and all(_nombre(x) for x in r) and _memes(sorted(r), sorted(attendu)):
        raise AssertionError(f"{quoi} renvoie {_court(tuple(r), 100)} : les bonnes valeurs, dans le mauvais ordre. "
                             "L'ordre est (mae, rmse, biais, mape).")
    for variante, nom, message in _VARIANTES_ERREURS:
        autre = _ref_erreurs(reels, prevus, variante)
        if _memes(r, autre) and not _memes(autre, attendu):
            raise AssertionError(f"Dans {quoi}, {nom} vaut {_fmt(r[_NOMS_ERREURS.index(nom)])} au lieu de "
                                 f"{_fmt(attendu[_NOMS_ERREURS.index(nom)])} : {message}")
    i = next(i for i in range(4) if not _memes([r[i]], [attendu[i]]))
    if i == 3 and attendu[3] is None:
        raise AssertionError(f"Dans {quoi}, mape vaut {_fmt(r[3])} : un réel vaut 0, la MAPE n'existe pas. Renvoyez "
                             "None (ni 0, ni l'infini).")
    if i == 3 and r[3] is None:
        raise AssertionError(f"Dans {quoi}, mape vaut None, mais aucun réel ne vaut 0 : elle vaut {_fmt(attendu[3])}.")
    raise AssertionError(f"Dans {quoi}, {_NOMS_ERREURS[i]} vaut {_fmt(r[i])} au lieu de {_fmt(attendu[i])}.")


def verifier_erreurs(f) -> None:
    cas = [
        ([10, 20], [13, 16]),
        ([0, 5], [1, 5]),
        ([100.0], [90.0]),
        ([4.0, 4.0, 4.0], [4.0, 4.0, 4.0]),       # tout juste : 0 partout
        ([10.0, 10.0], [7.0, 13.0]),               # biais nul, MAE de 3
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 12.0]),   # une grosse erreur : la RMSE la voit
        ([-4.0, 2.0], [-2.0, 2.0]),                # un réel négatif : |réel|
        ([0.0, 0.0], [0.0, 0.0]),
        ((5.0, 6.0), (6.0, 6.0)),
    ]
    for reels, prevus in cas:
        _une_erreurs(f, reels, prevus, f"erreurs({reels}, {prevus})")
    departs = departs_exemple()
    _une_erreurs(f, departs[24:], departs[12:24], "erreurs(departs[24:], departs[12:24])")
    for reels, prevus, pourquoi in (([1.0, 2.0], [1.0], "pas autant de réels que de prévisions"),
                                    ([1.0], [1.0, 2.0], "pas autant de réels que de prévisions"),
                                    ([], [], "aucune prévision"),
                                    ([1.0, None], [1.0, 2.0], "une valeur manquante"),
                                    ([1.0, 2.0], [float("nan"), 2.0], "une valeur manquante")):
        _doit_lever(f, (reels, prevus), f"erreurs({reels}, {prevus})",
                    f"Ici : {pourquoi}. Des longueurs différentes (`zip` tronquerait sans rien dire), aucune prévision, "
                    "ou une valeur manquante (None, NaN) → ValueError.")
    alea = random.Random(201)
    for essai in range(200):
        n = alea.randint(1, 15)
        reels = [_seize(alea, -16 * 50, 16 * 50) if alea.random() < 0.2 else _seize(alea) for _ in range(n)]
        if alea.random() < 0.3:
            reels[alea.randrange(n)] = 0.0
        prevus = [_seize(alea) for _ in range(n)]
        _une_erreurs(f, reels, prevus, f"erreurs({_court(reels, 60)}, {_court(prevus, 60)}) (tiré au hasard n°{essai})")


# ---------------------------------------------------------------------------- défis 202 et 203 : lissages


def _ref_ses(historique, alpha: float, horizon: int, variante: str = "") -> tuple:
    valeurs = [float(x) for x in historique]
    niveau = sum(valeurs) / len(valeurs) if variante == "moyenne_initiale" else valeurs[0]
    ajustees = [valeurs[0] if variante == "premier_ajuste" else None]
    precedent = niveau
    for y in valeurs[1:]:
        if variante != "fuite":
            ajustees.append(niveau)
        precedent = niveau
        if variante == "inverse":
            niveau = (1 - alpha) * y + alpha * niveau
        else:
            niveau = alpha * y + (1 - alpha) * niveau
        if variante == "fuite":
            ajustees.append(niveau)
    if variante == "sans_premier":
        ajustees = ajustees[1:]
    final = niveau
    if variante == "derniere_valeur":
        final = valeurs[-1]
    elif variante == "dernier_mois_oublie":
        final = precedent if len(valeurs) > 1 else niveau
    return ajustees, [final] * horizon, (final if variante != "derniere_valeur" else niveau), 0.0


_VARIANTES_SES = [
    ("inverse", "le niveau devient alpha × (la NOUVELLE valeur) + (1 − alpha) × l'ancien niveau. Vous appliquez alpha à "
                "l'ancien niveau : avec alpha = 0.9, le niveau ne bougerait presque plus."),
    ("fuite", "ajustees[t] est la prévision du mois t faite AVANT de le voir : le niveau d'avant la mise à jour. Vous "
              "l'écrivez après avoir lissé historique[t] : la prévision connaît déjà le mois qu'elle prévoit."),
    ("moyenne_initiale", "le niveau de départ est historique[0]. La moyenne de toute la série contient l'AVENIR de "
                         "chaque mois : la prévision du mois 1 connaîtrait déjà le mois 35."),
    ("premier_ajuste", "ajustees[0] vaut None : le premier mois n'a pas de passé, personne ne pouvait le prévoir."),
    ("sans_premier", "`ajustees` a autant d'éléments que `historique`, en commençant par None pour le mois 0."),
    ("derniere_valeur", "les prévisions reprennent le dernier NIVEAU lissé, pas la dernière valeur de l'historique "
                        "(sinon, à quoi bon lisser ?)."),
    ("dernier_mois_oublie", "le dernier niveau inclut le DERNIER mois : il est mis à jour après la dernière prévision "
                            "ajustée. Vous prévoyez avec le niveau d'avant."),
]


def _ref_holt(historique, alpha: float, beta: float, horizon: int, variante: str = "") -> tuple:
    valeurs = [float(x) for x in historique]
    if variante == "tendance_zero":
        niveau, tendance, ajustees, debut = valeurs[0], 0.0, [None], 1
    else:
        niveau, tendance, ajustees, debut = valeurs[1], valeurs[1] - valeurs[0], [None, None], 2
    if variante == "depart_ajuste":
        ajustees = [None, valeurs[1]]
    for t in range(debut, len(valeurs)):
        y = valeurs[t]
        prevu = niveau + tendance
        if variante != "fuite":
            ajustees.append(prevu)
        ancien = niveau
        if variante == "alpha_inverse":
            niveau = (1 - alpha) * y + alpha * prevu
        elif variante == "niveau_sans_tendance":
            niveau = alpha * y + (1 - alpha) * ancien
        else:
            niveau = alpha * y + (1 - alpha) * prevu
        if variante == "ancien_ecrase":
            ecart = 0.0
        elif variante == "tendance_brute":
            ecart = y - valeurs[t - 1]
        else:
            ecart = niveau - ancien
        if variante == "beta_inverse":
            tendance = (1 - beta) * ecart + beta * tendance
        else:
            tendance = beta * ecart + (1 - beta) * tendance
        if variante == "fuite":
            ajustees.append(niveau + tendance)
    if variante == "h_oublie":
        previsions = [niveau + tendance] * horizon
    else:
        previsions = [niveau + h * tendance for h in range(1, horizon + 1)]
    return ajustees, previsions, niveau, tendance


_VARIANTES_HOLT = [
    ("h_oublie", "le mois h après la fin vaut niveau + h × TENDANCE : la tendance s'ajoute une fois par mois. Vous "
                 "prévoyez niveau + tendance pour tous les mois."),
    ("ancien_ecrase", "la tendance se lisse sur nouveau niveau − ANCIEN niveau. Vous écrasez le niveau avant de faire "
                      "la différence, qui vaut alors 0 : gardez `ancien = niveau` avant la mise à jour."),
    ("tendance_brute", "la tendance se lisse sur l'écart entre deux NIVEAUX lissés (nouveau − ancien), pas entre deux "
                       "valeurs brutes de l'historique (trop nerveuses)."),
    ("niveau_sans_tendance", "le nouveau niveau mélange la valeur du mois et la PRÉVISION (niveau + tendance) : "
                             "alpha × y + (1 − alpha) × (niveau + tendance). Vous oubliez la tendance."),
    ("alpha_inverse", "le nouveau niveau vaut alpha × (la valeur du mois) + (1 − alpha) × la prévision : alpha "
                      "s'applique à la nouvelle valeur."),
    ("beta_inverse", "la tendance vaut beta × (nouveau − ancien niveau) + (1 − beta) × l'ancienne tendance : beta "
                     "s'applique au nouvel écart."),
    ("fuite", "ajustees[t] est la prévision niveau + tendance faite AVANT de voir historique[t]. Vous l'écrivez après "
              "la mise à jour : elle connaît déjà le mois qu'elle prévoit."),
    ("depart_ajuste", "ajustees[1] vaut None : la tendance de départ (historique[1] − historique[0]) est calculée AVEC "
                      "le mois 1, qui ne peut donc pas être prévu honnêtement. Votre ajustees[1] est historique[1] "
                      "lui-même, une erreur de 0 qui flatte la méthode."),
    ("tendance_zero", "le départ est niveau = historique[1] et tendance = historique[1] − historique[0], pas une "
                      "tendance nulle."),
]


def _comparer_lissage(r, attendu, quoi: str, variantes, ref, args, tendance_nulle: bool) -> None:
    _pas_none(r, quoi)
    if not isinstance(r, tuple) or len(r) != 4:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : un Lissage(ajustees, previsions, niveau, tendance) "
                             "(ou un tuple de 4 valeurs dans cet ordre) est attendu.")
    ajustees, previsions, niveau, tendance = r
    a_att, p_att, n_att, t_att = attendu
    for nom, valeur, taille in (("ajustees", ajustees, len(a_att)), ("previsions", previsions, len(p_att))):
        if not isinstance(valeur, list):
            raise AssertionError(f"Dans {quoi}, {nom} est un {type(valeur).__name__} : une list est attendue.")
        for x in valeur:
            if not (_nombre(x) or (nom == "ajustees" and x is None)):
                raise AssertionError(f"Dans {quoi}, {nom} contient {_court(x, 60)} : des nombres sont attendus"
                                     f"{' (ou None au début)' if nom == 'ajustees' else ''}.")
    for nom, valeur in (("niveau", niveau), ("tendance", tendance)):
        if not _nombre(valeur):
            precision = " (0.0 pour le lissage simple, qui n'a pas de tendance)" if tendance_nulle else ""
            raise AssertionError(f"Dans {quoi}, {nom} vaut {_court(valeur, 60)} : un nombre est attendu{precision}.")
    if _memes(ajustees, a_att) and _memes(previsions, p_att) and _proche(niveau, n_att) and _proche(tendance, t_att):
        return
    for variante, message in variantes:
        autre = ref(*args, variante)
        if (_memes(ajustees, autre[0]) and _memes(previsions, autre[1])
                and not (_memes(autre[0], a_att) and _memes(autre[1], p_att))):
            raise AssertionError(f"{quoi} : {message}")
    if len(ajustees) != len(a_att):
        raise AssertionError(f"Dans {quoi}, ajustees a {len(ajustees)} éléments au lieu de {len(a_att)} : un par mois "
                             f"de l'historique, en commençant par {'None' if tendance_nulle else 'None, None'}.")
    for t, (x, y) in enumerate(zip(ajustees, a_att)):
        if y is None and x is not None:
            raise AssertionError(f"Dans {quoi}, ajustees[{t}] vaut {_fmt(x)} au lieu de None : personne ne pouvait "
                                 f"prévoir le mois {t}{'' if tendance_nulle else ', qui sert au départ de la tendance'}.")
        if x is None and y is not None:
            raise AssertionError(f"Dans {quoi}, ajustees[{t}] vaut None au lieu de {_fmt(y)}.")
        if not _memes([x], [y]):
            raise AssertionError(f"Dans {quoi}, ajustees[{t}] vaut {_fmt(x)} au lieu de {_fmt(y)} (la prévision du mois "
                                 f"{t}, faite avec les mois 0 à {t - 1}).")
    if tendance_nulle and not _proche(tendance, 0.0):
        raise AssertionError(f"Dans {quoi}, tendance vaut {_fmt(tendance)} : le lissage simple n'a pas de tendance, 0.0.")
    if not _proche(niveau, n_att):
        raise AssertionError(f"Dans {quoi}, niveau vaut {_fmt(niveau)} au lieu de {_fmt(n_att)} (le niveau APRÈS le "
                             "dernier mois).")
    if not _proche(tendance, t_att):
        raise AssertionError(f"Dans {quoi}, tendance vaut {_fmt(tendance)} au lieu de {_fmt(t_att)} (la tendance APRÈS "
                             "le dernier mois).")
    if len(previsions) != len(p_att):
        raise AssertionError(f"Dans {quoi}, previsions a {len(previsions)} éléments au lieu de {len(p_att)} (horizon).")
    h = next(h for h in range(len(p_att)) if not _proche(previsions[h], p_att[h]))
    raise AssertionError(f"Dans {quoi}, previsions[{h}] vaut {_fmt(previsions[h])} au lieu de {_fmt(p_att[h])}"
                         f"{'' if tendance_nulle else f' (niveau + {h + 1} × tendance)'}.")


def _un_ses(f, historique, alpha, horizon, quoi: str, lignes_max: int | None = 2_000_000, conseil_lent: str = "") -> None:
    avant = copy.deepcopy(historique)
    r = _appel(f, (historique, alpha, horizon), quoi, lignes_max=lignes_max, conseil_lent=conseil_lent)
    _intact(avant, historique, quoi, "l'historique")
    _comparer_lissage(r, _ref_ses(historique, alpha, horizon), quoi, _VARIANTES_SES, _ref_ses,
                      (historique, alpha, horizon), True)


def _un_holt(f, historique, alpha, beta, horizon, quoi: str, lignes_max: int | None = 2_000_000,
             conseil_lent: str = "") -> None:
    avant = copy.deepcopy(historique)
    r = _appel(f, (historique, alpha, beta, horizon), quoi, lignes_max=lignes_max, conseil_lent=conseil_lent)
    _intact(avant, historique, quoi, "l'historique")
    _comparer_lissage(r, _ref_holt(historique, alpha, beta, horizon), quoi, _VARIANTES_HOLT, _ref_holt,
                      (historique, alpha, beta, horizon), False)


_CONSEIL_LISSAGE_LENT = ("Une seule passe sur la série : à chaque mois, la prévision (le niveau actuel), puis la mise à "
                         "jour. Relisser tout le passé pour chaque mois (appeler le lissage sur historique[:t]), c'est "
                         "20 000 fois le même travail.")


def verifier_lissage_exponentiel(f) -> None:
    departs = departs_exemple()
    for historique, alpha, horizon in (([10.0, 20.0, 20.0], 0.5, 2), (departs, 1.0, 2), (departs, 0.3, 1),
                                       ([4.0], 0.5, 2), ([4.0, 8.0], 0.25, 0), ([3, 9, 6, 12], 0.8, 3),
                                       (departs, 0.1, 4), ([5.0, 1.0, 5.0, 1.0, 5.0], 0.5, 1)):
        _un_ses(f, historique, alpha, horizon, f"lissage_exponentiel({_court(historique, 60)}, {alpha}, {horizon})")
    for historique, alpha, horizon, pourquoi in (
            ([1.0, 2.0], 0, 1, "alpha = 0"), ([1.0, 2.0], -0.5, 1, "alpha < 0"), ([1.0, 2.0], 1.5, 1, "alpha > 1"),
            ([1.0, 2.0], float("nan"), 1, "alpha = NaN"), ([1.0, 2.0], 0.5, -1, "horizon < 0"),
            ([], 0.5, 1, "historique vide"), ([1.0, None, 2.0], 0.5, 1, "une valeur manquante"),
            ([1.0, float("nan")], 0.5, 1, "une valeur manquante")):
        _doit_lever(f, (historique, alpha, horizon), f"lissage_exponentiel({historique}, {alpha}, {horizon})",
                    f"Ici : {pourquoi}. Il faut 0 < alpha <= 1 (`not 0 < alpha <= 1` refuse aussi NaN), un horizon "
                    ">= 0, au moins une valeur, et aucune valeur manquante → sinon ValueError.")
    alea = random.Random(202)
    for essai in range(150):
        historique = _serie_hasard(alea, alea.randint(1, 30))
        alpha, horizon = alea.choice([0.05, 0.1, 0.25, 0.3, 0.5, 0.7, 0.9, 1.0]), alea.randint(0, 6)
        _un_ses(f, historique, alpha, horizon,
                f"lissage_exponentiel({_court(historique, 60)}, {alpha}, {horizon}) (tiré au hasard n°{essai})")
    n = 20_000
    long = generer_serie(n, graine=202)
    _un_ses(f, long, 0.3, 3, f"lissage_exponentiel(une série de {n} mois, 0.3, 3)", lignes_max=30 * n + 10_000,
            conseil_lent=_CONSEIL_LISSAGE_LENT)


def verifier_holt(f) -> None:
    departs = departs_exemple()
    for historique, alpha, beta, horizon in (
            ([10.0, 12.0, 14.0, 16.0], 0.5, 0.5, 3), ([10.0, 12.0, 20.0], 0.5, 0.5, 2), ([5.0, 8.0], 0.4, 0.2, 3),
            (departs, 0.3, 0.1, 3), (departs, 1.0, 1.0, 2), ([1, 4, 2, 8, 5, 7], 0.6, 0.3, 4),
            ([3.0, 3.0, 3.0, 3.0], 0.2, 0.9, 0), ([100.0, 90.0, 85.0, 70.0, 72.0], 0.7, 0.4, 5)):
        _un_holt(f, historique, alpha, beta, horizon,
                 f"holt({_court(historique, 60)}, {alpha}, {beta}, {horizon})")
    for historique, alpha, beta, horizon, pourquoi in (
            ([1.0, 2.0, 3.0], 0, 0.5, 1, "alpha = 0"), ([1.0, 2.0, 3.0], 0.5, 0, 1, "beta = 0"),
            ([1.0, 2.0, 3.0], 1.5, 0.5, 1, "alpha > 1"), ([1.0, 2.0, 3.0], 0.5, 1.2, 1, "beta > 1"),
            ([1.0, 2.0, 3.0], 0.5, float("nan"), 1, "beta = NaN"), ([1.0, 2.0, 3.0], 0.5, 0.5, -2, "horizon < 0"),
            ([5.0], 0.5, 0.5, 1, "une seule valeur"), ([], 0.5, 0.5, 1, "historique vide"),
            ([1.0, None, 3.0], 0.5, 0.5, 1, "une valeur manquante")):
        _doit_lever(f, (historique, alpha, beta, horizon), f"holt({historique}, {alpha}, {beta}, {horizon})",
                    f"Ici : {pourquoi}. Il faut 0 < alpha <= 1, 0 < beta <= 1, un horizon >= 0, au moins 2 valeurs "
                    "(pour une tendance) et aucune valeur manquante → sinon ValueError.")
    alea = random.Random(203)
    for essai in range(150):
        historique = _serie_hasard(alea, alea.randint(2, 30))
        alpha = alea.choice([0.1, 0.25, 0.3, 0.5, 0.8, 1.0])
        beta = alea.choice([0.05, 0.1, 0.2, 0.5, 1.0])
        horizon = alea.randint(0, 6)
        _un_holt(f, historique, alpha, beta, horizon,
                 f"holt({_court(historique, 60)}, {alpha}, {beta}, {horizon}) (tiré au hasard n°{essai})")
    n = 20_000
    long = generer_serie(n, graine=203, tendance=0.25)
    _un_holt(f, long, 0.3, 0.1, 3, f"holt(une série de {n} mois, 0.3, 0.1, 3)", lignes_max=40 * n + 10_000,
             conseil_lent=_CONSEIL_LISSAGE_LENT)


# ---------------------------------------------------------------------------- défi 204 : validation glissante


def _regle(passe: list, horizon: int) -> list[float]:
    """La méthode des tests : chaque prévision dépend du passé reçu ET du rang, et reste exacte (des 1/64)."""
    return [float(passe[-1]) + (i + 1) / 2 + len(passe) / 64 for i in range(horizon)]


class _Espionne:
    """Une méthode qui note ce qu'elle reçoit (une COPIE, prise au moment de l'appel)."""

    def __init__(self, regle=_regle):
        self.regle = regle
        self.appels: list[tuple[list, object]] = []

    def __call__(self, passe, horizon):
        vu = list(passe)
        self.appels.append((vu, horizon))
        if not vu:
            raise ValueError("la méthode a reçu un passé vide : il lui faut au moins un mois")
        return self.regle(vu, horizon)


def _ref_validation(serie, regle, debut: int, horizon: int, variante: str = "") -> list[tuple]:
    valeurs = list(serie)
    derniere = len(valeurs) - horizon
    fin = derniere if variante == "derniere_oubliee" else derniere + 1
    essais = []
    for origine in range(debut + 1 if variante == "premiere_oubliee" else debut, fin):
        if variante == "fuite":
            prevus = regle(valeurs[:origine + 1], horizon)
        else:
            prevus = regle(valeurs[:origine], horizon)
        cible = origine + horizon - 1
        prevu = prevus[0] if variante == "premier" else prevus[horizon - 1]
        if variante == "origine_index":
            essais.append((origine - 1, cible, prevu, valeurs[cible]))
        else:
            essais.append((origine, cible, prevu, valeurs[cible]))
    return essais


def _memes_essais(r, attendu) -> bool:
    try:
        return len(r) == len(attendu) and all(len(a) == 4 and _memes(a, b) for a, b in zip(r, attendu))
    except TypeError:
        return False


_VARIANTES_VALIDATION = [
    ("premier", "on garde la prévision à `horizon` mois, la DERNIÈRE de la liste (prevus[horizon - 1]) : c'est elle "
                "qui vise la cible. prevus[0] est la prévision à 1 mois, comparée ici au mauvais mois."),
    ("derniere_oubliee", "la dernière origine est len(serie) − horizon (incluse : sa cible est le dernier mois). "
                         "`range(debut, len(serie) - horizon + 1)`."),
    ("premiere_oubliee", "la première origine est `debut` elle-même : `range(debut, …)`, pas `range(debut + 1, …)`."),
    ("origine_index", "l'origine est le NOMBRE de mois connus (la méthode a reçu serie[:origine]), pas le numéro du "
                      "dernier mois connu."),
]


def _espionnage(espionne: _Espionne, serie, debut: int, horizon: int, quoi: str) -> None:
    """Ce que la méthode a reçu : jamais l'avenir, le bon horizon, un appel par origine, toutes les origines."""
    valeurs = list(serie)
    attendues = list(range(debut, len(valeurs) - horizon + 1))
    longueurs = [len(vu) for vu, _ in espionne.appels]
    for vu, h in espionne.appels:
        if not _egaux_nan(vu, valeurs[:len(vu)]):
            raise AssertionError(f"Dans {quoi}, la méthode a reçu {_court(vu, 80)}, qui n'est pas le début de la série : "
                                 "à chaque origine, elle reçoit serie[:origine], les mois connus, dans l'ordre.")
        if h != horizon:
            raise AssertionError(f"Dans {quoi}, la méthode est appelée avec horizon={_court(h, 30)} au lieu de {horizon} : "
                                 "elle doit prévoir jusqu'au mois visé, `methode(serie[:origine], horizon)`.")
    if longueurs == [o + 1 for o in attendues]:
        raise AssertionError(f"Dans {quoi}, la méthode a reçu des passés de {_court(longueurs, 60)} mois au lieu de "
                             f"{_court(attendues, 60)} : à l'origine o, vous lui donnez serie[:o + 1], qui contient le mois "
                             "n°o, celui qu'elle doit prévoir à 1 mois. C'est une fuite du futur : donnez-lui serie[:o].")
    for n in longueurs:
        if n >= len(valeurs):
            raise AssertionError(f"Dans {quoi}, la méthode a reçu TOUTE la série ({n} mois) : elle voit le mois qu'elle "
                                 "doit prévoir, et tous les suivants. À chaque origine, ne lui donnez que serie[:origine].")
    doubles = sorted({n for n in longueurs if longueurs.count(n) > 1})
    if doubles:
        raise AssertionError(f"Dans {quoi}, la méthode est appelée {longueurs.count(doubles[0])} fois avec les "
                             f"{doubles[0]} premiers mois : UN appel par origine suffit (une vraie méthode peut mettre "
                             "des minutes à s'entraîner).")
    if longueurs != attendues and sorted(longueurs) == attendues:
        raise AssertionError(f"Dans {quoi}, les origines sont parcourues dans le désordre ({_court(longueurs, 60)}) : "
                             "de `debut` à la dernière, dans l'ordre.")


def _une_validation(f, serie, debut: int, horizon: int | None, quoi: str, regle=_regle) -> None:
    avant = copy.deepcopy(serie)
    espionne = _Espionne(regle)
    args = (serie, espionne, debut) if horizon is None else (serie, espionne, debut, horizon)
    h = 1 if horizon is None else horizon
    r = _appel(f, args, quoi, "À l'origine o, la méthode reçoit serie[:o] ; la cible est le mois n°o + horizon − 1, "
                              "et la dernière origine est len(serie) − horizon.")
    _intact(avant, serie, quoi, "la série")
    _pas_none(r, quoi)
    attendu = _ref_validation(serie, regle, debut, h)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list d'Essai(origine, cible, prevu, reel) est "
                             "attendue.")
    for e in r:
        if not isinstance(e, tuple) or len(e) != 4:
            raise AssertionError(f"{quoi} renvoie {_court(e, 80)} dans sa liste : un Essai(origine, cible, prevu, reel) "
                                 "(ou un tuple de 4 valeurs dans cet ordre) est attendu.")
    _espionnage(espionne, serie, debut, h, quoi)
    if _memes_essais(r, attendu):
        return
    for variante, message in _VARIANTES_VALIDATION:
        autre = _ref_validation(serie, regle, debut, h, variante)
        if _memes_essais(r, autre) and not _memes_essais(autre, attendu):
            raise AssertionError(f"{quoi} : {message}")
    if len(r) != len(attendu):
        raise AssertionError(f"{quoi} renvoie {len(r)} essais au lieu de {len(attendu)} : une origine par valeur de "
                             f"{debut} à len(serie) − horizon = {len(serie) - h}, incluses.")
    i = next(i for i in range(len(r)) if not _memes(r[i], attendu[i]))
    o, c, p, v = attendu[i]
    raise AssertionError(f"Dans {quoi}, l'essai n°{i} vaut {_court(tuple(r[i]), 80)} au lieu de "
                         f"Essai(origine={o}, cible={c}, prevu={_fmt(p)}, reel={_fmt(v)}) : la méthode reçoit "
                         f"serie[:{o}], on garde sa prévision n°{h - 1} (la {h}e), à comparer à serie[{c}].")


def verifier_validation_glissante(f) -> None:
    departs = departs_exemple()
    cas = [
        ([5.0, 7.0, 6.0, 9.0], 2, None),
        ([5.0, 7.0, 6.0, 9.0], 1, 2),
        ([5.0, 7.0, 6.0, 9.0], 3, 1),              # une seule origine
        ([5.0, 7.0, 6.0, 9.0], 1, 3),
        (departs, 24, 3),
        (departs, 12, 1),
        (departs, 1, 12),
        ([2, 4, 3, 8, 6], 2, 2),
    ]
    for serie, debut, horizon in cas:
        appel = "" if horizon is None else f", horizon={horizon}"
        _une_validation(f, serie, debut, horizon, f"validation_glissante({_court(serie, 50)}, methode, {debut}{appel})")
    for serie, debut, horizon, pourquoi in (([1.0, 2.0, 3.0], 1, 0, "horizon < 1"), ([1.0, 2.0, 3.0], 0, 1, "debut < 1"),
                                            ([1.0, 2.0, 3.0], 3, 1, "aucune origine dont la cible existe"),
                                            ([1.0, 2.0, 3.0], 2, 2, "aucune origine dont la cible existe"),
                                            ([1.0], 1, 1, "aucune origine dont la cible existe")):
        _doit_lever(f, (serie, _Espionne(), debut, horizon),
                    f"validation_glissante({serie}, methode, {debut}, horizon={horizon})",
                    f"Ici : {pourquoi}. horizon < 1, debut < 1, ou aucune origine entre debut et len(serie) − horizon "
                    "→ ValueError (une évaluation sur zéro essai ne dit rien).")
    alea = random.Random(204)
    for essai in range(150):
        n = alea.randint(2, 40)
        horizon = alea.randint(1, min(5, n - 1))
        debut = alea.randint(1, n - horizon)
        serie = _serie_hasard(alea, n)
        _une_validation(f, serie, debut, horizon, f"validation_glissante({_court(serie, 50)}, methode, {debut}, "
                                                  f"horizon={horizon}) (tiré au hasard n°{essai})")


# ---------------------------------------------------------------------------- défi 205 : marge de sécurité


def _quantile(ecarts: list, niveau: int, variante: str = "") -> float:
    tries = sorted(ecarts)
    n = len(tries)
    if variante == "interpolation":                 # numpy.quantile par défaut (« linear »)
        position = (n - 1) * niveau / 100
        bas = math.floor(position)
        haut = min(bas + 1, n - 1)
        return tries[bas] + (tries[haut] - tries[bas]) * (position - bas)
    if variante == "plafond_float":
        return tries[math.ceil(niveau / 100 * n) - 1]
    if variante == "plancher":
        return tries[int(niveau / 100 * n) - 1]
    k = (niveau * n + 99) // 100
    if variante == "k_suivant":
        return tries[min(k, n - 1)]
    return tries[k - 1]


def _ref_marge(serie, regle, debut: int, horizon: int, niveau: int, variante: str = "") -> tuple:
    h_essais = 1 if variante == "horizon_un" else horizon
    essais = _ref_validation(serie, regle, debut, h_essais)
    if variante == "absolu":
        ecarts = [abs(v - p) for _, _, p, v in essais]
    elif variante == "signe":
        ecarts = [p - v for _, _, p, v in essais]
    else:
        ecarts = [v - p for _, _, p, v in essais]
    marge = _quantile(ecarts, niveau, variante)
    if variante == "prevision_derniere_origine":
        prevision = essais[-1][2]
    elif variante == "prevision_premiere":
        prevision = regle(list(serie), horizon)[0]
    else:
        prevision = regle(list(serie), horizon)[horizon - 1]
    return prevision, marge, prevision + marge


_VARIANTES_MARGE = [
    ("absolu", "les erreurs se prennent dans le sens qui coûte, réel − prévu, SANS valeur absolue : un mois où l'on a "
               "prévu trop n'oblige à aucune marge. Avec `abs`, la marge est trop grande."),
    ("signe", "les erreurs se prennent dans le sens réel − prévu (positif = des départs que personne n'avait prévus). "
              "Vous calculez prévu − réel : votre marge protège contre l'erreur inverse."),
    ("interpolation", "la marge est un écart qui a vraiment eu lieu, le k-ième des écarts triés, sans interpolation "
                      "entre deux écarts (ce que font `statistics.quantiles` et `numpy.quantile` par défaut)."),
    ("plafond_float", "calculez k en ENTIERS, `(niveau * n + 99) // 100` : en float, `niveau / 100 * n` peut dépasser "
                      "d'un cheveu un entier (28 / 100 * 25 = 7.000000000000001), et son plafond prend un écart de trop. "
                      "`numpy.quantile(…, method=\"inverted_cdf\")` fait le même calcul en float : même piège."),
    ("plancher", "k est le plus petit entier tel que k / n >= niveau / 100 : un PLAFOND. Arrondi vers le bas, la marge "
                 "couvre moins de `niveau` % des essais (et `tries[0 - 1]` prend même le plus grand)."),
    ("k_suivant", "la marge est le k-ième écart trié, `tries[k - 1]` (les indices commencent à 0), pas `tries[k]`."),
    ("horizon_un", "les erreurs doivent être celles à `horizon` mois (validation_glissante(…, horizon)) : celles à 1 mois "
                   "sont plus petites et ne disent rien de l'erreur à 3 mois."),
    ("prevision_derniere_origine", "la prévision se fait avec TOUTE la série, `methode(list(serie), horizon)` : "
                                   "maintenant, tout ce passé est connu. Vous reprenez la prévision du dernier essai, "
                                   "faite sans les derniers mois."),
    ("prevision_premiere", "la prévision retenue est celle à `horizon` mois, la DERNIÈRE de la liste renvoyée par la "
                           "méthode, pas la première."),
]


def _aide_204(f) -> str:
    """Si la validation glissante de l'élève ne passe pas, le dire : le défi 205 en dépend sans doute."""
    autre = getattr(f, "__globals__", {}).get("validation_glissante")
    if autre is None:
        return ""
    try:
        verifier_validation_glissante(autre)
    except (AssertionError, NotImplementedError):
        return (" (Votre validation_glissante, défi 204, ne passe pas encore ses vérifications : si vous l'appelez ici, "
                "commencez par elle.)")
    return ""


def _une_marge(f, serie, debut: int, horizon: int, niveau: int | None, quoi: str, regle=_regle) -> None:
    avant = copy.deepcopy(serie)
    args = (serie, _Espionne(regle), debut, horizon) + (() if niveau is None else (niveau,))
    r = _appel(f, args, quoi)
    _intact(avant, serie, quoi, "la série")
    _pas_none(r, quoi)
    nv = 90 if niveau is None else niveau
    attendu = _ref_marge(serie, regle, debut, horizon, nv)
    if not isinstance(r, tuple) or len(r) != 3 or not all(_nombre(x) for x in r):
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : une Marge(prevision, marge, borne) (trois nombres) est "
                             "attendue.")
    if _memes(r, attendu):
        return
    if not _proche(r[2], r[0] + r[1]):
        raise AssertionError(f"Dans {quoi}, borne vaut {_fmt(r[2])} : c'est prevision + marge = {_fmt(r[0])} + "
                             f"{_fmt(r[1])}.")
    for variante, message in _VARIANTES_MARGE:
        try:
            autre = _ref_marge(serie, regle, debut, horizon, nv, variante)
        except (IndexError, ValueError):
            continue
        if _memes(r, autre) and not _memes(autre, attendu):
            raise AssertionError(f"{quoi} renvoie {_court(tuple(r), 80)} au lieu de {_court(attendu, 80)} : {message}")
    if not _proche(r[0], attendu[0]):
        raise AssertionError(f"Dans {quoi}, prevision vaut {_fmt(r[0])} au lieu de {_fmt(attendu[0])} : la prévision à "
                             f"{horizon} mois de la méthode, avec toute la série.")
    raise AssertionError(f"Dans {quoi}, marge vaut {_fmt(r[1])} au lieu de {_fmt(attendu[1])} : le k-ième des écarts "
                         f"réel − prévu triés, avec k = (niveau × n + 99) // 100.{_aide_204(f)}")


def _naive12(passe, horizon):
    cycle = passe[len(passe) - 12:]
    return [float(cycle[h % 12]) for h in range(horizon)]


def verifier_marge_de_securite(f) -> None:
    departs = departs_exemple()
    comme_dernier = lambda passe, horizon: [passe[-1]] * horizon  # noqa: E731
    cas = [
        ([10.0, 14.0, 11.0, 15.0, 12.0, 16.0], 1, 1, 60, comme_dernier),
        (departs, 24, 3, 90, _naive12),
        (departs, 24, 3, None, _naive12),
        (departs, 12, 1, 50, _regle),
        (departs, 12, 3, 99, _regle),
        ([4.0, 6.0, 5.0, 9.0, 7.0, 8.0], 2, 2, 1, _regle),
        (generer_serie(26, graine=5), 1, 1, 28, comme_dernier),     # 25 essais : 28 / 100 * 25 = 7.000000000000001
        (generer_serie(52, graine=6), 2, 1, 14, comme_dernier),     # 50 essais : 14 / 100 * 50 aussi
    ]
    for serie, debut, horizon, niveau, regle in cas:
        appel = "" if niveau is None else f", {niveau}"
        _une_marge(f, serie, debut, horizon, niveau,
                   f"marge_de_securite({_court(serie, 50)}, methode, {debut}, {horizon}{appel})", regle)
    for niveau in (0, 100, 0.9, 150, -5, 90.0):
        _doit_lever(f, (departs, _Espionne(), 24, 3, niveau), f"marge_de_securite(departs, methode, 24, 3, {niveau!r})",
                    "niveau est un pourcentage ENTIER entre 1 et 99 (90, pas 0.9) : sinon ValueError.")
    for serie, debut, horizon in (([1.0, 2.0, 3.0], 1, 0), ([1.0, 2.0, 3.0], 0, 1), ([1.0, 2.0, 3.0], 3, 1)):
        _doit_lever(f, (serie, _Espionne(), debut, horizon, 90),
                    f"marge_de_securite({serie}, methode, {debut}, {horizon}, 90)",
                    "Les erreurs du défi 204 (horizon < 1, debut < 1, aucune origine possible) → ValueError.")
    alea = random.Random(205)
    for essai in range(150):
        n = alea.randint(3, 40)
        horizon = alea.randint(1, min(4, n - 1))
        debut = alea.randint(1, n - horizon)
        niveau = alea.choice([alea.randint(1, 99), 50, 80, 90, 95, 14, 28, 56])
        serie = _serie_hasard(alea, n)
        _une_marge(f, serie, debut, horizon, niveau, f"marge_de_securite({_court(serie, 50)}, methode, {debut}, "
                                                     f"{horizon}, {niveau}) (tiré au hasard n°{essai})")


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"prevision_naive": 200, "erreurs": 201, "lissage_exponentiel": 202, "holt": 203,
            "validation_glissante": 204, "marge_de_securite": 205}

SERIE = Serie(
    code="33",
    titre="Prévoir les départs : prévision naïve, erreurs, lissage, Holt, validation glissante, marge",
    module_eleve="defis.serie_33",
    module_corrige="defis.corrige_serie_33",
    defis=(
        Defi(200, "prevision_naive", "🟢", "—", verifier_prevision_naive),
        Defi(201, "erreurs", "🟢", "4.2", verifier_erreurs),
        Defi(202, "lissage_exponentiel", "🟡", "—", verifier_lissage_exponentiel),
        Defi(203, "holt", "🟡", "—", verifier_holt),
        Defi(204, "validation_glissante", "🔴", "3.4", verifier_validation_glissante),
        Defi(205, "marge_de_securite", "🔴", "4.3", verifier_marge_de_securite),
    ),
)

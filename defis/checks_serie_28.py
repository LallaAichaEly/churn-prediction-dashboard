"""Vérifications automatiques de la série 28 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur se comporte comme le flux de production :

- il vous donne des flux `UneSeuleFois` : une deuxième lecture, un `len()` ou un `list()` se voient ;
- il MESURE la mémoire (`tracemalloc`) pendant 200 000 valeurs : une liste qui grandit se voit, quelle que
  soit la vitesse de la machine ;
- au défi 173, il COMPTE les comparaisons entre vos valeurs et les lignes exécutées dans votre fichier : un
  tri à chaque valeur se voit ;
- il compare chaque réponse à une version de référence, puis rejoue les erreurs classiques (la formule de la
  « somme des carrés », n au lieu de n - 1, pas de max(0, …), `>=`, `.days`, `math.exp`, `randrange(i)`, la
  moyenne des moyennes…) pour vous dire LAQUELLE vous avez écrite ;
- au défi 174, il tire 20 000 échantillons et compte combien de fois chaque valeur est choisie ;
- il vérifie qu'une valeur refusée n'a rien changé, et que deux objets ne partagent rien.
"""

from __future__ import annotations

import heapq
import math
import numbers
import random
import sys
from datetime import datetime, timedelta

import numpy as np

from .flux import (
    INSTANT_EXEMPLE,
    UneSeuleFois,
    connexions_exemple,
    generer_journal,
    generer_montants,
    journal_exemple,
    pic_memoire,
    taux_quotidiens_exemple,
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
    """Le fichier où `objet` (fonction, classe ou méthode) est écrit : c'est là qu'on compte les lignes."""
    code = getattr(objet, "__code__", None)
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


_NUMEROS = {"MoyenneVariance": 170, "alertes_cusum": 171, "ActiviteRecente": 172, "MedianeCourante": 173,
            "echantillon_reservoir": 174, "fusionner": 175}


def _ou_leve(exc: BaseException) -> tuple[str, int | None]:
    """(nom qualifié de la fonction qui a levé `exc`, numéro du défi auquel elle appartient)."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        return "", None
    code = tb.tb_frame.f_code
    nom = getattr(code, "co_qualname", code.co_name)
    if nom == code.co_name and "self" in tb.tb_frame.f_locals:  # Python 3.10 : pas de co_qualname
        nom = f"{type(tb.tb_frame.f_locals['self']).__name__}.{code.co_name}"
    numeros = [_NUMEROS[morceau] for morceau in nom.split(".") if morceau in _NUMEROS]
    return nom, (numeros[-1] if numeros else None)


def _pas_encore(exc: NotImplementedError, defi: int, a_faire: bool) -> None:
    """Que faire d'un NotImplementedError levé pendant la vérification du défi `defi` ?

    - il vient d'un AUTRE défi (fusionner utilise MoyenneVariance) : on dit lequel faire d'abord ;
    - il vient de ce défi et `a_faire` : on le laisse passer (le défi est « à faire ») ;
    - sinon (une méthode oubliée d'une classe commencée) : on dit laquelle.
    """
    nom, numero = _ou_leve(exc)
    if numero is not None and numero != defi:
        raise AssertionError(f"`{nom}` n'est pas encore écrite (raise NotImplementedError) : faites d'abord le défi "
                             f"{numero}, ce défi-ci s'appuie dessus.") from None
    if a_faire:
        raise exc
    raise AssertionError(f"`{nom or '?'}` lève encore NotImplementedError : terminez-la.") from None


_CONSEIL_LENT = "Votre code a été arrêté : il exécute beaucoup trop de lignes (une boucle dans une boucle ?)."


def _conseil_flux(exc: BaseException) -> str:
    """Le conseil quand un flux `UneSeuleFois` a été mal lu (len, indice, deuxième lecture)."""
    texte = str(exc)
    if isinstance(exc, TypeError) and "UneSeuleFois" in texte:
        return ("Un flux n'a pas de longueur et ne s'indexe pas : on ne sait pas combien de valeurs vont arriver. Une "
                "seule boucle : `for i, x in enumerate(flux):`.")
    if isinstance(exc, ValueError) and "empty range" in texte:
        return ("`randrange(0)` n'a aucune valeur possible : avec `randrange(i)`, la toute première valeur (i = 0) plante "
                "quand k = 0. La valeur numéro i (à partir de 0) se tire parmi i + 1 possibilités : "
                "`rng.randrange(i + 1)`, de 0 à i inclus.")
    if isinstance(exc, RuntimeError) and "déjà été lu" in texte:
        return ("Vous lisez le flux DEUX fois (pour le compter, puis pour calculer ?). Un générateur relu ne lève pas "
                "d'erreur : il est VIDE, et votre deuxième boucle ne verrait rien, en silence. Une seule lecture suffit.")
    return ""


def _appel(f, args: tuple, quoi: str, defi: int, conseil: str = "", lignes_max: int | None = None,
           conseil_lent: str = "", a_faire: bool = False, fichier: str | None = None):
    """f(*args), en transformant toute erreur en message lisible."""
    try:
        return _trace(fichier or _fichier(f), lambda: f(*args), lignes_max)
    except NotImplementedError as exc:
        _pas_encore(exc, defi, a_faire)
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError. {conseil}".rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {_conseil_flux(exc) or conseil}".rstrip()) from None


def _mesurer(appel, quoi: str, defi: int) -> tuple[object, int]:
    """pic_memoire(appel), en transformant toute erreur en message lisible."""
    try:
        return pic_memoire(appel)
    except NotImplementedError as exc:
        _pas_encore(exc, defi, a_faire=False)
        raise
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {_conseil_flux(exc)}".rstrip()) from None


_CONSEILS_LECTURE = {
    "moyenne": "Sans aucune valeur, la moyenne vaut math.nan (pas de division par zéro).",
    "variance": "Avec moins de deux valeurs, la variance vaut math.nan (pas de division par zéro).",
}


def _lire(objet, attribut: str, quoi: str, defi: int):
    """objet.attribut (une propriété peut lever : on le dit)."""
    return _appel(lambda: getattr(objet, attribut), (), quoi, defi, fichier="",
                  conseil=_CONSEILS_LECTURE.get(attribut, ""))


def _creer(classe, args: tuple, quoi: str, defi: int):
    """classe(*args) : un NotImplementedError dans __init__ veut dire « défi à faire »."""
    return _appel(classe, args, quoi, defi, a_faire=True, fichier="")


def _doit_lever(f, args: tuple, quoi: str, defi: int, conseil: str = "", attendu: type = ValueError) -> None:
    """Exige que f(*args) lève `attendu`."""
    try:
        resultat = f(*args)
    except NotImplementedError as exc:
        _pas_encore(exc, defi, a_faire=False)
        raise
    except attendu:
        return
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _nombre(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _nan(v) -> bool:
    return _nombre(v) and math.isnan(v)


def _proche(obtenu, attendu: float, rel: float = 1e-9, abs_: float = 1e-12) -> bool:
    if not _nombre(obtenu):
        return False
    if math.isnan(attendu):
        return math.isnan(obtenu)
    return not math.isnan(obtenu) and math.isclose(obtenu, attendu, rel_tol=rel, abs_tol=abs_)


def _f(v) -> str:
    """Un nombre lisible dans un message."""
    if not _nombre(v):
        return _court(v, 60)
    if math.isnan(v):
        return "nan"
    return f"{v:.10g}"


# ---------------------------------------------------------------------------- défi 170 : MoyenneVariance


def _ref_mv(valeurs) -> tuple[int, float, float]:
    """(n, moyenne, variance d'échantillon), en deux passages exacts (math.fsum) : la référence."""
    valeurs = [float(v) for v in valeurs]
    n = len(valeurs)
    if n == 0:
        return 0, math.nan, math.nan
    moyenne = math.fsum(valeurs) / n
    if n < 2:
        return n, moyenne, math.nan
    return n, moyenne, math.fsum((v - moyenne) ** 2 for v in valeurs) / (n - 1)


def _etat_mv(mv, quoi: str, defi: int = 170) -> tuple:
    return (_lire(mv, "n", f"{quoi}.n", defi), _lire(mv, "moyenne", f"{quoi}.moyenne", defi),
            _lire(mv, "variance", f"{quoi}.variance", defi))


def _remplir_mv(classe, valeurs, quoi: str, defi: int = 170):
    mv = _creer(classe, (), "MoyenneVariance()", defi)
    for x in valeurs:
        _appel(mv.ajouter, (x,), f"{quoi} : ajouter({x!r})", defi)
    return mv


_RAPPEL_WELFORD = ("Welford : ecart = x - moyenne (l'ANCIENNE), puis moyenne += ecart / n, puis m2 += ecart * (x - "
                   "moyenne) (la NOUVELLE), et variance = m2 / (n - 1).")
_RAPPEL_FUSION = ("Avec ecart = moyenne_b - moyenne_a : moyenne = moyenne_a + ecart * nb / n et m2 = m2_a + m2_b + "
                  "ecart ** 2 * na * nb / n, puis variance = m2 / (n - 1).")


def _comparer_mv(obtenu: tuple, valeurs, quoi: str, rel: float = 1e-9, rappel: str = _RAPPEL_WELFORD) -> None:
    n_att, moy_att, var_att = _ref_mv(valeurs)
    n, moyenne, variance = obtenu
    if n != n_att or isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise AssertionError(f"{quoi} : n vaut {_court(n)} au lieu de {n_att} (le nombre de valeurs reçues, un int).")
    if not _proche(moyenne, moy_att, rel):
        if n_att == 0:
            raise AssertionError(f"{quoi} : sans aucune valeur, moyenne vaut {_court(moyenne)} au lieu de nan. Pas de "
                                 "valeur, pas de moyenne : renvoyez math.nan (comme pandas), pas 0.0 qui ressemble à "
                                 "une vraie moyenne.")
        raise AssertionError(f"{quoi} : moyenne vaut {_f(moyenne)} au lieu de {_f(moy_att)}. {rappel}")
    if not _proche(variance, var_att, rel):
        if n_att < 2:
            raise AssertionError(f"{quoi} : avec {n_att} valeur{'s' if n_att > 1 else ''}, variance vaut "
                                 f"{_court(variance)} au lieu de nan. La variance d'échantillon divise par n - 1 : "
                                 "elle n'est pas définie avant deux valeurs. Renvoyez math.nan (comme pandas `.var()`), "
                                 "ni 0.0 ni une ZeroDivisionError.")
        if _nombre(variance) and _proche(variance, var_att * (n_att - 1) / n_att, 1e-6):
            raise AssertionError(f"{quoi} : variance vaut {_f(variance)} au lieu de {_f(var_att)} : vous divisez par n. "
                                 "La variance d'un ÉCHANTILLON divise par n - 1 (pandas `.var()`, `statistics.variance`) ; "
                                 "c'est `np.var` qui divise par n par défaut (ddof=0).")
        raise AssertionError(f"{quoi} : variance vaut {_f(variance)} au lieu de {_f(var_att)}. {rappel}")


_CONSEIL_SOMME_CARRES = ("C'est la formule « moyenne des carrés moins carré de la moyenne » (ou Σx² - n·moyenne²) : "
                         "elle soustrait deux nombres de l'ordre de 10^19 qui ne diffèrent que de quelques unités, et "
                         "un float n'a que 15 à 16 chiffres. Il ne reste que du bruit (0.0 ici, parfois une variance "
                         "NÉGATIVE). Calculez sur des écarts à la moyenne courante : l'algorithme de Welford.")


def verifier_moyenne_variance(classe, _rapide: bool = False) -> None:
    # ------------------------------------------------------------------ deux résumés indépendants
    a = _creer(classe, (), "MoyenneVariance()", 170)
    b = _creer(classe, (), "MoyenneVariance()", 170)
    _appel(a.ajouter, (5.0,), "a.ajouter(5.0)", 170)
    _appel(a.ajouter, (9.0,), "a.ajouter(9.0)", 170)
    if _lire(b, "n", "b.n", 170) != 0:
        raise AssertionError("Deux résumés partagent leur état : après a.ajouter(5.0) et a.ajouter(9.0), un AUTRE résumé b "
                             f"a n = {_court(_lire(b, 'n', 'b.n', 170))}. Créez les attributs dans __init__ (self._n = 0…), "
                             "pas au niveau de la classe, et ne modifiez pas `MoyenneVariance._n`.")

    mv = _creer(classe, (), "MoyenneVariance()", 170)

    # ------------------------------------------------------------------ l'exemple, valeur après valeur
    vues: list = []
    for x in [4, 7, 13, 16]:
        _appel(mv.ajouter, (x,), f"ajouter({x})", 170)
        vues.append(x)
        _comparer_mv(_etat_mv(mv, "mv"), vues, f"Après ajouter({', '.join(map(str, vues))})")
    _comparer_mv(_etat_mv(_creer(classe, (), "MoyenneVariance()", 170), "MoyenneVariance()"), [],
                 "MoyenneVariance() sans aucune valeur")

    # ------------------------------------------------------------------ de grands nombres presque égaux
    xs = connexions_exemple()
    grand = _remplir_mv(classe, xs, "connexions_exemple()")
    n, moyenne, variance = _etat_mv(grand, "mv")
    if not (_nombre(variance) and _proche(variance, 30.0, 1e-6)):
        raise AssertionError(f"Sur connexions_exemple() (des secondes depuis 1970 : 1 700 000 004, …007, …013, …016), "
                             f"variance vaut {_f(variance)} au lieu de 30.0. {_CONSEIL_SOMME_CARRES}")
    _comparer_mv((n, moyenne, variance), xs, "Sur connexions_exemple()", rel=1e-6)
    alea = random.Random(170)
    for decalage, echelle in ((1e9, 0.01), (-3e12, 50.0), (1e6, 1e-4)):
        valeurs = [decalage + round(alea.gauss(0, echelle), 6) for _ in range(30)]
        mv = _remplir_mv(classe, valeurs, f"30 valeurs autour de {decalage:g}")
        n, moyenne, variance = _etat_mv(mv, "mv")
        attendu = _ref_mv(valeurs)[2]
        if not _proche(variance, attendu, 1e-4):
            raise AssertionError(f"Sur 30 valeurs autour de {decalage:g} (écart type {echelle:g}), variance vaut "
                                 f"{_f(variance)} au lieu de {_f(attendu)}. {_CONSEIL_SOMME_CARRES}")

    # ------------------------------------------------------------------ des flux au hasard
    for essai in range(40):
        taille = alea.randrange(0, 25)
        valeurs = [alea.choice((alea.randrange(-50, 50), round(alea.uniform(-1e3, 1e3), 3), np.float64(alea.random()),
                                np.int64(alea.randrange(10))))
                   for _ in range(taille)]
        mv = _remplir_mv(classe, valeurs, f"ajouter() de {_court(valeurs, 80)}")
        _comparer_mv(_etat_mv(mv, "mv"), valeurs, f"Après les valeurs {_court([float(v) for v in valeurs], 100)}", 1e-7)

    # ------------------------------------------------------------------ valider AVANT de modifier
    for mauvaise, attendu, conseil in (
            (None, TypeError, "None (une case vide) n'est pas une mesure."),
            ("12", TypeError, "Un texte n'est pas un nombre, même s'il en a l'air : `float(x)` l'accepterait en silence."),
            (True, TypeError, "True est un int pour Python (True + True == 2) : `isinstance(x, bool)` d'abord."),
            (math.nan, ValueError, "Un NaN accepté une fois rend la moyenne NaN POUR TOUJOURS (NaN + 1 = NaN) : "
                                   "`math.isfinite(x)`."),
            (np.float64("nan"), ValueError, "np.float64('nan') est un NaN aussi."),
            (math.inf, ValueError, "Une valeur infinie rend la variance NaN pour toujours (inf - inf = NaN)."),
            (-math.inf, ValueError, "Une valeur infinie rend la variance NaN pour toujours.")):
        mv = _remplir_mv(classe, [4, 7, 13], "[4, 7, 13]")
        quoi = f"Après [4, 7, 13], ajouter({mauvaise!r})"
        _doit_lever(mv.ajouter, (mauvaise,), quoi, 170, conseil, attendu)
        apres = _etat_mv(mv, "mv")
        if not (apres[0] == 3 and _proche(apres[1], 8.0) and _proche(apres[2], 21.0)):
            raise AssertionError(f"{quoi} lève bien {attendu.__name__}, mais le résumé a changé : n, moyenne, variance = "
                                 f"{_court(apres)} au lieu de (3, 8.0, 21.0). Vous modifiez l'état AVANT de vérifier "
                                 "la valeur : validez d'abord, modifiez ensuite.")

    if _rapide:
        return

    # ------------------------------------------------------------------ la mémoire ne grandit pas
    mv = _creer(classe, (), "MoyenneVariance()", 170)
    n_valeurs = 200_000

    def remplir():
        for x in generer_montants(n_valeurs, graine=1700):
            mv.ajouter(x)

    _, pic = _mesurer(remplir, f"ajouter() de {n_valeurs} montants", 170)
    if pic > 500_000:
        raise AssertionError(f"Après {n_valeurs:,} valeurs, votre résumé occupe {pic / 1e6:.1f} Mo : il garde les valeurs "
                             "(une liste qui grandit à chaque ajouter). Un résumé de flux garde TROIS nombres (n, la "
                             "moyenne, la somme des carrés des écarts), quel que soit le nombre de valeurs."
                             .replace(",", " "))
    _comparer_mv(_etat_mv(mv, "mv"), list(generer_montants(n_valeurs, graine=1700)),
                 f"Après {n_valeurs} montants de generer_montants(graine=1700)", rel=1e-9)


# ---------------------------------------------------------------------------- défi 171 : alertes_cusum


def _ref_cusum(valeurs, cible, marge, seuil, defaut: str | None = None) -> list[int]:
    """La référence, et les erreurs classiques (pour dire laquelle vous avez écrite).

    Pour « nan_remet_a_zero » et « nan_empoisonne », None est bien sauté (`if x is None: continue`) mais pas NaN.
    """
    cumul, alertes, position = 0.0, [], -1
    for x in valeurs:
        est_nan = isinstance(x, float) and math.isnan(x)
        manquant = x is None or est_nan
        if manquant and defaut == "positions_filtrees":
            continue
        position += 1
        if est_nan and defaut == "nan_remet_a_zero":
            cumul = 0.0  # max(0, cumul + NaN) vaut 0 : NaN > 0 est faux, max garde le premier
            continue
        if est_nan and defaut == "nan_empoisonne":
            cumul = math.nan  # max(cumul + NaN, 0) vaut NaN
            continue
        if manquant:
            continue
        m = 0.0 if defaut == "sans_marge" else marge
        if defaut == "sans_plancher":
            cumul = cumul + x - cible - m
        elif defaut == "nan_empoisonne":
            cumul = max(cumul + x - cible - m, 0.0)  # max(nan, 0.0) garde nan : le cumul reste NaN pour toujours
        else:
            cumul = max(0.0, cumul + x - cible - m)
        if (cumul >= seuil) if defaut == "large" else (cumul > seuil):
            alertes.append(position + (1 if defaut == "a_partir_de_1" else 0))
            if defaut != "sans_remise":
                cumul = 0.0
    return alertes


_DEFAUTS_CUSUM = (
    ("sans_remise", "Après une alerte, le cumul doit repartir de 0 : sinon il reste au-dessus du seuil et l'alerte sonne "
                    "TOUS les jours suivants (l'équipe finira par ignorer le message)."),
    ("sans_plancher", "Il manque le max(0, …) : sans lui, chaque jour calme (sous cible + marge) fait baisser le cumul "
                      "(-12 après les 25 premiers jours de taux_quotidiens_exemple()), et une vraie hausse doit d'abord "
                      "« rembourser » ce crédit. L'alerte arrive des semaines trop tard, ou jamais."),
    ("large", "« Dépasser » le seuil, c'est cumul > seuil (strictement) : un cumul égal au seuil ne sonne pas encore."),
    ("sans_marge", "Vous oubliez la marge : cumul + valeur - cible - marge. Sans elle, le moindre écart au-dessus de la "
                   "cible s'accumule, et un pic isolé suffit à faire sonner l'alerte."),
    ("nan_remet_a_zero", "Un jour sans donnée remet votre cumul à 0 : max(0, cumul + NaN) vaut 0 (NaN > 0 est faux, max "
                         "garde le premier). Un jour sans donnée ne prouve rien : sautez-le AVANT le calcul, le cumul ne "
                         "doit pas bouger."),
    ("nan_empoisonne", "Après un jour NaN, votre cumul est NaN pour toujours (NaN + 1 = NaN, et NaN > seuil est faux) : "
                       "plus aucune alerte ne sonnera, en silence. Sautez les jours sans donnée (None ou NaN) AVANT le "
                       "calcul."),
    ("positions_filtrees", "Vos positions sautent les jours sans donnée : la position d'un jour doit compter TOUS les "
                           "jours (c'est sa date). `enumerate` sur le flux d'origine, PUIS sauter les jours manquants."),
    ("a_partir_de_1", "Les positions commencent à 0 (le premier jour du flux est le jour 0), comme enumerate()."),
)


def _diagnostic_cusum(obtenu, valeurs, parametres: tuple, quoi: str) -> str:
    attendu = _ref_cusum(valeurs, *parametres)
    texte = f"{quoi} renvoie {_court(obtenu)} au lieu de {attendu}."
    for defaut, conseil in _DEFAUTS_CUSUM:
        if obtenu == _ref_cusum(valeurs, *parametres, defaut=defaut):
            return f"{texte} {conseil}"
    return (f"{texte} À chaque jour avec une valeur : cumul = max(0, cumul + valeur - cible - marge) ; si cumul > seuil, "
            "on note la position du jour et le cumul repart de 0.")


def _verifier_cusum_sur(f, valeurs, parametres: tuple, quoi: str, flux=None) -> None:
    r = _appel(f, (flux if flux is not None else UneSeuleFois(list(valeurs)), *parametres), quoi, 171, a_faire=True,
               conseil="Un jour sans donnée vaut None ou NaN : sautez-le (`x is None or math.isnan(x)`).")
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
    if not isinstance(r, list) or not all(isinstance(i, numbers.Integral) and not isinstance(i, bool) for i in r):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : on attend une liste de positions (des int).")
    if r != _ref_cusum(valeurs, *parametres):
        raise AssertionError(_diagnostic_cusum(r, valeurs, parametres, quoi))


def verifier_alertes_cusum(f) -> None:
    standard = (2.0, 0.5, 2.0)
    # L'ordre compte : chaque cas sépare une erreur classique de toutes celles qui n'ont pas encore été écartées.
    cas = (
        ([3.0] * 6, standard),                                    # >=, la marge, les positions à partir de 1
        ([3.0] * 12, standard),                                   # deux alertes : la remise à zéro
        ([3.0] * 4, standard),                                    # le cumul égale le seuil : pas d'alerte
        ([4.0, 2.0, 2.0, 4.0, 2.5], standard),                    # un pic, puis un autre : la marge
        ([1.0] * 20 + [3.0] * 6, standard),                       # un mois calme : le plancher
        ([3.0, 3.0, None, 3.0, math.nan, 3.0, 3.0, 3.0, 3.0, 3.0], standard),   # les jours sans donnée
        ([3.0, np.float64("nan"), np.float64(3.0), 3.0, 3.0, 3.0], standard),
        ([5.0, 5.0, 5.0], (2.0, 0.0, 2.5)),                       # marge nulle
        ([], standard),
        ([None, math.nan], standard),
        (taux_quotidiens_exemple(), standard),
    )
    for valeurs, parametres in cas:
        quoi = f"alertes_cusum({_court(valeurs, 90)}, cible={parametres[0]}, marge={parametres[1]}, seuil={parametres[2]})"
        _verifier_cusum_sur(f, valeurs, parametres, quoi)

    alea = random.Random(171)
    for _ in range(150):
        valeurs = [alea.choice((None, math.nan)) if alea.random() < 0.08 else alea.randrange(4, 15) / 4
                   for _ in range(alea.randrange(0, 40))]
        parametres = (alea.choice((2.0, 2.5)), alea.choice((0.0, 0.25, 0.5)), alea.choice((0.5, 1.0, 2.0, 3.25)))
        quoi = f"alertes_cusum({_court(valeurs, 90)}, cible={parametres[0]}, marge={parametres[1]}, seuil={parametres[2]})"
        _verifier_cusum_sur(f, valeurs, parametres, quoi)

    # ------------------------------------------------------------------ un flux se lit une fois, sans être gardé
    n = 300_000

    def long_flux():
        for i in range(n):
            yield 3.0 if i % 1000 > 990 else 2.0

    _, pic = _mesurer(lambda: f(long_flux(), *standard), f"alertes_cusum(un générateur de {n} jours, …)", 171)
    if pic > 1_000_000:
        raise AssertionError(f"Sur un flux de {n} jours, alertes_cusum occupe {pic / 1e6:.1f} Mo : vous gardez le flux "
                             "(list(valeurs) ?). CUSUM n'a besoin que du cumul et de la position du jour.")

    # ------------------------------------------------------------------ des paramètres absurdes
    for parametres, conseil in (
            ((2.0, -0.5, 2.0), "Une marge négative ferait sonner l'alerte quand le taux est SOUS la cible."),
            ((2.0, 0.5, 0.0), "Un seuil nul ou négatif ferait sonner l'alerte au moindre écart."),
            ((2.0, 0.5, -1.0), "Un seuil nul ou négatif ferait sonner l'alerte au moindre écart."),
            ((2.0, math.nan, 2.0), "`nan < 0` est faux : un test `if marge < 0` laisse passer NaN, et plus rien ne "
                                   "sonne jamais. `math.isfinite(...)` attrape NaN et l'infini."),
            ((2.0, 0.5, math.nan), "`nan <= 0` est faux : NaN passe le test du seuil, et `cumul > nan` est toujours "
                                   "faux. `math.isfinite(...)`."),
            ((math.nan, 0.5, 2.0), "Une cible NaN rend le cumul NaN : plus rien ne sonne. `math.isfinite(...)`.")):
        _doit_lever(f, ([3.0] * 6, *parametres),
                    f"alertes_cusum([3.0] * 6, cible={parametres[0]}, marge={parametres[1]}, seuil={parametres[2]})",
                    171, conseil)


# ---------------------------------------------------------------------------- défi 172 : ActiviteRecente


class _RefActivite:
    def __init__(self, demi_vie: float, defaut: str | None = None) -> None:
        self.demi_vie, self.defaut, self.etat = demi_vie, defaut, {}

    def _jours(self, depuis, jusqu_a) -> float:
        if self.defaut == "days":
            return (jusqu_a - depuis).days
        if self.defaut == "seconds":
            return (jusqu_a - depuis).seconds / 86_400
        return (jusqu_a - depuis).total_seconds() / 86_400

    def _facteur(self, depuis, jusqu_a) -> float:
        jours = self._jours(depuis, jusqu_a)
        if self.defaut == "exp":
            return math.exp(-jours / self.demi_vie)
        return 0.5 ** (jours / self.demi_vie)

    def ajouter(self, client, moment, poids=1.0) -> None:
        if client not in self.etat:
            self.etat[client] = (poids, moment)
            return
        score, ref = self.etat[client]
        if self.defaut == "sans_vieillir_entre":
            self.etat[client] = (score + poids, max(ref, moment))
        elif moment >= ref:
            self.etat[client] = (score * self._facteur(ref, moment) + poids, moment)
        elif self.defaut == "retard_plein_poids":
            self.etat[client] = (score + poids, ref)
        else:
            self.etat[client] = (score + poids * self._facteur(moment, ref), ref)

    def score(self, client, instant) -> float:
        if client not in self.etat:
            return 0.0
        score, ref = self.etat[client]
        if self.defaut == "sans_vieillir_a_l_instant":
            return score
        return score * self._facteur(ref, instant)


_DEFAUTS_ACTIVITE = (
    ("days", "Vous comptez l'âge avec `.days`, qui TRONQUE : 23 heures font 0 jour, 3 jours et 2 heures font 3 jours. "
             "Utilisez `(fin - debut).total_seconds() / 86_400`."),
    ("seconds", "Vous comptez l'âge avec `.seconds`, qui ne garde que les secondes de la DERNIÈRE journée : 3 jours et "
                "2 heures donnent 7 200 secondes. Utilisez `(fin - debut).total_seconds() / 86_400`."),
    ("exp", "Vous utilisez math.exp(-jours / demi_vie) : au bout d'une demi-vie, il reste 1/e ≈ 37 %, pas 50 %. C'est "
            "une « constante de temps ». Une DEMI-vie, c'est 0.5 ** (jours / demi_vie)."),
    ("sans_vieillir_a_l_instant", "Votre score ne vieillit pas jusqu'à `instant` : il garde la valeur du dernier "
                                  "événement. Un client parti il y a trois mois aurait encore le score de son dernier "
                                  "jour, et c'est exactement le signal de départ que le modèle doit voir. Multipliez par "
                                  "0.5 ** (jours entre le dernier événement et instant / demi_vie)."),
    ("retard_plein_poids", "Un événement arrivé EN RETARD compte avec son âge réel : vieillissez-le jusqu'à la référence "
                           "(poids × 0.5 ** (jours entre son moment et la référence / demi_vie)) avant de l'ajouter."),
    ("sans_vieillir_entre", "Vous additionnez les poids sans vieillir l'ancien score jusqu'au nouvel événement : "
                            "score = score × 0.5 ** (jours écoulés / demi_vie) + poids."),
)


def _reponses_activite(objet, scenario, requetes) -> list[float]:
    _jouer_activite(objet, scenario)
    return [objet.score(client, instant) for client, instant in requetes]


def _diagnostic_activite(obtenus: list, scenario, demi_vie, requetes, quoi: str) -> str:
    """Le message, quand la réponse à une requête est fausse : on compare TOUTES les réponses du scénario à celles
    de chaque erreur classique (une seule réponse ne suffit pas : `.days` et « ne pas vieillir » donnent toutes les
    deux 1.0 à 12 heures)."""
    attendus = _reponses_activite(_RefActivite(demi_vie), scenario, requetes)
    i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if not _proche(o, a, 1e-9))
    texte = f"{quoi[i]} renvoie {_f(obtenus[i])} au lieu de {_f(attendus[i])}."
    for defaut, conseil in _DEFAUTS_ACTIVITE:
        mutant = _reponses_activite(_RefActivite(demi_vie, defaut), scenario, requetes)
        if all(_proche(o, m, 1e-9) for o, m in zip(obtenus, mutant)):
            return f"{texte} {conseil}"
    return (f"{texte} score = somme des poids × 0.5 ** (âge en jours / demi_vie), l'âge allant du moment de chaque "
            "événement jusqu'à `instant`.")


def _jouer_activite(objet, scenario):
    for client, moment, poids in scenario:
        objet.ajouter(client, moment, poids)
    return objet


def _scenario_texte(scenario) -> str:
    return "; ".join(f"ajouter({c!r}, {m:%d/%m %H:%M}{'' if p == 1.0 else f', {p:g}'})" for c, m, p in scenario)


def verifier_activite_recente(classe) -> None:
    t0 = datetime(2024, 3, 1, 8, 0)
    a = _creer(classe, (7,), "ActiviteRecente(7)", 172)
    b = _creer(classe, (7,), "ActiviteRecente(7)", 172)
    _appel(a.ajouter, ("C1", t0, 5.0), "a.ajouter('C1', 01/03, 5.0)", 172)
    r = _appel(b.score, ("C1", t0), "b.score('C1', 01/03), sur un objet b qui n'a rien reçu", 172,
               conseil="Un client jamais vu a un score de 0.0 (pas KeyError) : le tableau de bord demande le score de "
                       "TOUS les clients, pas seulement des actifs.")
    if r != 0:
        raise AssertionError(f"Deux objets partagent leur état : après a.ajouter('C1', …), un AUTRE objet b donne "
                             f"b.score('C1', …) = {_f(r)}. Créez le dictionnaire dans __init__, pas au niveau de la classe.")

    # ------------------------------------------------------------------ des scénarios courts, chacun son piège
    scenarios = (
        (1.0, [("C1", t0, 1.0)], ((t0 + timedelta(days=1)), (t0 + timedelta(days=2)), t0, t0 + timedelta(hours=12),
                                  t0 + timedelta(days=3, hours=2), t0 + timedelta(hours=23))),
        (1.0, [("C1", t0, 1.0), ("C1", t0 + timedelta(days=1), 1.0)], (t0 + timedelta(days=1), t0 + timedelta(days=3))),
        (30.0, [("C1", t0, 2.0), ("C1", t0 + timedelta(days=10, hours=5), 1.0), ("C1", t0 + timedelta(days=40), 3.0)],
         (t0 + timedelta(days=40), t0 + timedelta(days=95))),
        (7.0, [("C3", t0 + timedelta(hours=23), 1.0), ("C3", t0 + timedelta(days=1, hours=7), 1.0)],
         (t0 + timedelta(days=2),)),
        (7.0, [("C3", t0 + timedelta(days=1, hours=7), 1.0), ("C3", t0 + timedelta(hours=23), 1.0)],   # en retard
         (t0 + timedelta(days=2), t0 + timedelta(days=1, hours=7))),
        (2.0, [("A", t0, 1.0), ("B", t0 + timedelta(days=5), 4.0), ("A", t0 + timedelta(days=2), 0.5),
               ("A", t0 + timedelta(days=1), 2.0), ("B", t0 + timedelta(days=1), 1.0)], (t0 + timedelta(days=6),)),
    )
    for demi_vie, scenario, instants in scenarios:
        objet = _creer(classe, (demi_vie,), f"ActiviteRecente({demi_vie:g})", 172)
        for client, moment, poids in scenario:
            args = (client, moment) if poids == 1.0 else (client, moment, poids)
            _appel(objet.ajouter, args, f"ajouter({client!r}, {moment:%d/%m/%Y %H:%M}…)", 172,
                   conseil="Un événement peut arriver EN RETARD (plus ancien que le dernier reçu) : il faut l'accepter. "
                           "Une boutique envoie souvent ses données le lendemain.")
        requetes = [(client, instant) for instant in instants for client in sorted({c for c, _, _ in scenario})]
        quoi = [f"Avec demi_vie_jours={demi_vie:g}, après {_scenario_texte(scenario)}, "
                f"score({client!r}, {instant:%d/%m %H:%M})" for client, instant in requetes]
        obtenus = [_appel(objet.score, (client, instant), q, 172,
                          conseil="score() ne lève ValueError que si instant est antérieur au dernier ÉVÉNEMENT du "
                                  "client. Lire un score ne doit rien modifier : ne rangez pas le score vieilli dans "
                                  "l'objet.")
                   for (client, instant), q in zip(requetes, quoi)]
        attendus = _reponses_activite(_RefActivite(demi_vie), scenario, requetes)
        if not all(_proche(o, a, 1e-9) for o, a in zip(obtenus, attendus)):
            raise AssertionError(_diagnostic_activite(obtenus, scenario, demi_vie, requetes, quoi))

    # ------------------------------------------------------------------ l'exemple
    objet = _creer(classe, (30,), "ActiviteRecente(30)", 172)
    reference = _RefActivite(30.0)
    for client, moment, poids in journal_exemple():
        _appel(objet.ajouter, (client, moment, poids), f"ajouter({client!r}, {moment:%d/%m/%Y %H:%M}, {poids})", 172)
        reference.ajouter(client, moment, poids)
    for client in ("C1", "C2", "C3"):
        quoi = f"Sur journal_exemple(), score({client!r}, INSTANT_EXEMPLE)"
        r = _appel(objet.score, (client, INSTANT_EXEMPLE), quoi, 172)
        if not _proche(r, reference.score(client, INSTANT_EXEMPLE), 1e-9):
            raise AssertionError(f"{quoi} renvoie {_f(r)} au lieu de {_f(reference.score(client, INSTANT_EXEMPLE))}.")

    # ------------------------------------------------------------------ jamais vu, le futur, lire sans modifier
    quoi = "score('jamais_vu', INSTANT_EXEMPLE)"
    r = _appel(objet.score, ("jamais_vu", INSTANT_EXEMPLE), quoi, 172,
               conseil="Un client sans événement a un score de 0.0 : le tableau de bord demande le score de TOUS les "
                       "clients, pas seulement des actifs.")
    if r != 0 or not _nombre(r):
        raise AssertionError(f"{quoi} renvoie {_court(r)} au lieu de 0.0 : un client sans événement n'a aucune activité.")

    objet = _creer(classe, (7,), "ActiviteRecente(7)", 172)
    _appel(objet.ajouter, ("C1", t0 + timedelta(days=10)), "ajouter('C1', 11/03 08:00)", 172)
    _appel(objet.ajouter, ("C1", t0), "ajouter('C1', 01/03 08:00) (en retard)", 172,
           conseil="Un événement peut arriver EN RETARD : il faut l'accepter.")
    _doit_lever(objet.score, ("C1", t0 + timedelta(days=5)),
                "Après ajouter('C1', 11/03) puis ajouter('C1', 01/03) (en retard), score('C1', 06/03)", 172,
                "C1 a un événement le 11/03, APRÈS le 06/03 : il est fondu dans le score et ne se retire plus. Le compter "
                "quand même, c'est utiliser le futur (la fuite du défi 55). Gardez comme référence l'événement le PLUS "
                "RÉCENT (un événement en retard ne la recule pas) et levez ValueError si instant < référence.")
    _appel(objet.score, ("C1", t0 + timedelta(days=10)), "score('C1', 11/03 08:00), l'instant de son dernier événement",
           172, conseil="À l'instant même du dernier événement, le score existe (âge 0).")

    objet = _creer(classe, (7,), "ActiviteRecente(7)", 172)
    _appel(objet.ajouter, ("C1", t0), "ajouter('C1', 01/03 08:00)", 172)
    plus_tard = _appel(objet.score, ("C1", t0 + timedelta(days=14)), "score('C1', 15/03)", 172)
    plus_tot = _appel(objet.score, ("C1", t0 + timedelta(days=7)), "puis score('C1', 08/03)", 172,
                      conseil="Lire un score ne doit rien modifier : après score(…, 15/03), on peut encore demander le "
                              "08/03 (aucun événement n'est postérieur au 08/03).")
    if not (_proche(plus_tard, 0.25) and _proche(plus_tot, 0.5)):
        raise AssertionError(f"Avec demi_vie_jours=7 et un événement le 01/03 : score('C1', 15/03) puis score('C1', 08/03) "
                             f"renvoient {_f(plus_tard)} et {_f(plus_tot)} au lieu de 0.25 et 0.5. Lire un score ne doit "
                             "rien modifier : ne rangez pas le score vieilli dans l'objet.")

    # ------------------------------------------------------------------ des valeurs absurdes
    for demi_vie in (0, -3, math.nan):
        _doit_lever(classe, (demi_vie,), f"ActiviteRecente({demi_vie!r})", 172,
                    "Une demi-vie nulle ou négative n'a pas de sens (division par zéro, ou des scores qui grandissent "
                    "avec le temps). Attention : `nan <= 0` est faux.")
    objet = _creer(classe, (7,), "ActiviteRecente(7)", 172)
    _appel(objet.ajouter, ("C1", t0, 2.0), "ajouter('C1', 01/03, 2.0)", 172)
    _doit_lever(objet.ajouter, ("C1", t0 + timedelta(days=1), math.nan), "ajouter('C1', 02/03, nan)", 172,
                "Un poids NaN rendrait le score de C1 NaN pour toujours.")
    r = _appel(objet.score, ("C1", t0 + timedelta(days=7)), "puis score('C1', 08/03)", 172,
               conseil="Le poids NaN a été refusé : il ne doit rien avoir changé, ni le score ni la référence.")
    if not _proche(r, 1.0):
        raise AssertionError(f"Après ajouter('C1', 01/03, 2.0) puis ajouter('C1', 02/03, nan) refusé, score('C1', 08/03) "
                             f"vaut {_f(r)} au lieu de 1.0 : le poids refusé a quand même modifié l'état. Validez d'abord, "
                             "modifiez ensuite.")

    # ------------------------------------------------------------------ à l'échelle : la mémoire
    n, n_clients = 100_000, 200
    objet = _creer(classe, (30,), "ActiviteRecente(30)", 172)

    def remplir():
        for client, moment, poids in generer_journal(n, n_clients=n_clients, graine=172):
            objet.ajouter(client, moment, poids)

    _, pic = _mesurer(remplir, f"ajouter() de generer_journal({n}, n_clients={n_clients})", 172)
    if pic > 2_000_000:
        raise AssertionError(f"Après {n} événements de {n_clients} clients, votre objet occupe {pic / 1e6:.1f} Mo : il garde "
                             "les événements. Par client, deux nombres suffisent : le score et l'instant de référence.")
    reference = _RefActivite(30.0)
    fin = datetime(2024, 1, 1)
    for client, moment, poids in generer_journal(n, n_clients=n_clients, graine=172):
        reference.ajouter(client, moment, poids)
        fin = max(fin, moment)
    instant = fin + timedelta(days=3, hours=5)
    for client in sorted(reference.etat):
        quoi = f"Après generer_journal({n}, n_clients={n_clients}, graine=172), score({client!r}, {instant:%d/%m/%Y %H:%M})"
        r = _appel(objet.score, (client, instant), quoi, 172)
        if not _proche(r, reference.score(client, instant), 1e-7):
            raise AssertionError(f"{quoi} renvoie {_f(r)} au lieu de {_f(reference.score(client, instant))} (le journal "
                                 "contient des événements en retard, jusqu'à 3 jours).")


# ---------------------------------------------------------------------------- défi 173 : MedianeCourante


class _Compte(float):
    """Un float qui compte ses comparaisons, et arrête tout au-delà d'une limite."""

    compteur = [0]
    limite: list = [None]

    def _compter(self) -> None:
        _Compte.compteur[0] += 1
        if _Compte.limite[0] is not None and _Compte.compteur[0] > _Compte.limite[0]:
            raise _BoucleSansFin

    def __lt__(self, autre):
        self._compter()
        return float.__lt__(self, autre)

    def __le__(self, autre):
        self._compter()
        return float.__le__(self, autre)

    def __gt__(self, autre):
        self._compter()
        return float.__gt__(self, autre)

    def __ge__(self, autre):
        self._compter()
        return float.__ge__(self, autre)

    __hash__ = float.__hash__


def _ref_mediane(valeurs) -> float:
    if not valeurs:
        return math.nan
    tries = sorted(float(v) for v in valeurs)
    milieu = len(tries) // 2
    return tries[milieu] if len(tries) % 2 else (tries[milieu - 1] + tries[milieu]) / 2


class _RefMediane:
    """Deux tas, et les erreurs classiques (pour dire laquelle vous avez écrite)."""

    def __init__(self, defaut: str | None = None) -> None:
        self.defaut, self.bas, self.haut = defaut, [], []

    def ajouter(self, x) -> None:
        signe = 1 if self.defaut == "sans_signe" else -1
        if not self.bas or x <= signe * self.bas[0]:
            heapq.heappush(self.bas, signe * x)
        else:
            heapq.heappush(self.haut, x)
        if self.defaut == "sans_reequilibrer":
            return
        if len(self.bas) > len(self.haut) + 1:
            heapq.heappush(self.haut, signe * heapq.heappop(self.bas))
        elif len(self.haut) > len(self.bas):
            heapq.heappush(self.bas, signe * heapq.heappop(self.haut))

    def mediane(self) -> float:
        signe = 1 if self.defaut == "sans_signe" else -1
        if not self.bas:
            return math.nan
        if len(self.bas) > len(self.haut) or self.defaut == "pair_bas":
            return float(signe * self.bas[0])
        if self.defaut == "pair_haut":
            return float(self.haut[0])
        return (signe * self.bas[0] + self.haut[0]) / 2


_DEFAUTS_MEDIANE = (  # dans cet ordre : chaque flux du correcteur a déjà écarté les précédentes
    ("sans_reequilibrer", "Vous ne rééquilibrez pas les deux tas : après chaque ajout, `bas` doit avoir autant de valeurs "
                          "que `haut`, ou une de plus. Sinon le sommet de `bas` n'est plus le milieu. Si l'un est trop "
                          "gros : heappop chez lui, heappush chez l'autre (avec le signe moins)."),
    ("pair_bas", "Avec un nombre PAIR de valeurs, la médiane est la moyenne des DEUX valeurs du milieu : "
                 "(-bas[0] + haut[0]) / 2, pas seulement la plus petite."),
    ("pair_haut", "Avec un nombre PAIR de valeurs, la médiane est la moyenne des DEUX valeurs du milieu : "
                  "(-bas[0] + haut[0]) / 2, pas seulement la plus grande."),
    ("sans_signe", "Le tas `bas` doit donner son MAXIMUM, et heapq ne fait que des tas min : rangez-y -x (et lisez "
                   "-bas[0]). Sans les signes moins, bas[0] est le plus PETIT de la moitié basse."),
)


def _diagnostic_mediane(obtenus: list, vues: list, quoi: str) -> str:
    """Le message, quand la médiane après la dernière valeur de `vues` est fausse : on compare toute la suite des
    médianes obtenues à celle de chaque erreur classique."""
    attendu = _ref_mediane(vues)
    texte = f"{quoi} renvoie {_f(obtenus[-1])} au lieu de {_f(attendu)}."
    for defaut, conseil in _DEFAUTS_MEDIANE:
        mutant = _RefMediane(defaut)
        suite = []
        for x in vues:
            mutant.ajouter(x)
            suite.append(mutant.mediane())
        if all(_proche(o, m, 1e-12) for o, m in zip(obtenus, suite)):
            return f"{texte} {conseil}"
    return (f"{texte} Après chaque ajout, `bas` (la moitié basse, avec des -x) doit avoir autant de valeurs que `haut`, ou "
            "une de plus, et tout `bas` doit être <= tout `haut`. Vérifiez le côté où vous rangez x (comparé à -bas[0]), "
            "le rééquilibrage, et les signes moins.")


def _suivre_mediane(classe, valeurs, quoi_flux: str) -> None:
    m = _creer(classe, (), "MedianeCourante()", 173)
    vues: list = []
    obtenus: list = []
    for x in valeurs:
        _appel(m.ajouter, (x,), f"{quoi_flux} : ajouter({x!r})", 173)
        vues.append(x)
        quoi = f"Après ajouter({', '.join(map(_f, vues)) if len(vues) <= 12 else '… ' + str(len(vues)) + ' valeurs'}), mediane()"
        obtenus.append(_appel(m.mediane, (), quoi, 173))
        if not _proche(obtenus[-1], _ref_mediane(vues), 1e-12):
            raise AssertionError(_diagnostic_mediane(obtenus, vues, quoi))
        taille = _appel(len, (m,), f"len() après {len(vues)} valeurs", 173)
        if taille != len(vues):
            raise AssertionError(f"Après {len(vues)} valeurs, len(m) vaut {_court(taille)} au lieu de {len(vues)}.")


def verifier_mediane_courante(classe) -> None:
    a = _creer(classe, (), "MedianeCourante()", 173)
    b = _creer(classe, (), "MedianeCourante()", 173)
    _appel(a.ajouter, (7,), "a.ajouter(7)", 173)
    if _appel(len, (b,), "len(b)", 173) != 0:
        raise AssertionError("Deux objets partagent leurs valeurs : après a.ajouter(7), un AUTRE objet b a len(b) = "
                             f"{_court(len(b))}. Créez les listes (les tas) dans __init__, pas au niveau de la classe : "
                             "`bas = []` écrit dans la classe est UNE liste partagée par tous les objets.")

    m = _creer(classe, (), "MedianeCourante()", 173)
    r = _appel(m.mediane, (), "MedianeCourante().mediane()", 173,
               conseil="Sans aucune valeur, renvoyez math.nan (comme pandas `.median()` sur une série vide).")
    if not _nan(r):
        raise AssertionError(f"MedianeCourante().mediane() renvoie {_court(r)} au lieu de nan : pas de valeur, pas de "
                             "médiane (0.0 ressemblerait à une vraie médiane).")
    if _appel(len, (m,), "len(MedianeCourante())", 173) != 0:
        raise AssertionError("len(MedianeCourante()) devrait valoir 0.")

    # L'ordre compte : [1, 2] sépare les erreurs de parité de toutes les autres (les tas y sont équilibrés d'office).
    for valeurs in ([1, 2], [2, 1], [5, 1, 9, 3, 7], [3, 3, 3, 1, 1, 5, 5], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8], [-2.5, 0, -7, 1e9, 3.25, -1e9], [4, 4, 1, 1]):
        _suivre_mediane(classe, valeurs, f"Le flux {valeurs}")
    alea = random.Random(173)
    for _ in range(60):
        valeurs = [alea.choice((alea.randrange(-5, 6), round(alea.uniform(-100, 100), 2)))
                   for _ in range(alea.randrange(1, 30))]
        _suivre_mediane(classe, valeurs, f"Le flux {_court(valeurs, 80)}")

    # ------------------------------------------------------------------ valider avant de ranger
    for mauvaise, attendu, conseil in (
            (None, TypeError, "None n'est pas un nombre : comparé à un nombre, il lèverait TypeError plus loin, au "
                              "milieu d'un tas à moitié modifié."),
            ("3", TypeError, "Un texte n'est pas un nombre."),
            (math.nan, ValueError, "NaN n'est ni plus petit ni plus grand que rien (toutes ses comparaisons sont "
                                   "fausses) : rangé dans un tas, il casserait l'ordre en silence.")):
        m = _creer(classe, (), "MedianeCourante()", 173)
        for x in (5, 1, 9):
            _appel(m.ajouter, (x,), f"ajouter({x})", 173)
        quoi = f"Après ajouter(5, 1, 9), ajouter({mauvaise!r})"
        _doit_lever(m.ajouter, (mauvaise,), quoi, 173, conseil, attendu)
        if _appel(len, (m,), "len()", 173) != 3 or not _proche(_appel(m.mediane, (), "mediane()", 173), 5.0):
            raise AssertionError(f"{quoi} lève bien {attendu.__name__}, mais la médiane ou len() a changé : la valeur "
                                 "refusée a été rangée. Validez d'abord, rangez ensuite.")

    # ------------------------------------------------------------------ pas de tri à chaque valeur
    n = 20_000
    alea = random.Random(1730)
    valeurs = [_Compte(alea.uniform(0, 1000)) for _ in range(n)]
    m = _creer(classe, (), "MedianeCourante()", 173)
    _Compte.compteur[0], _Compte.limite[0] = 0, 120 * n

    def tout_ajouter():
        derniere = None
        for x in valeurs:
            m.ajouter(x)
            derniere = m.mediane()
        return derniere

    conseil = ("Le correcteur compte les comparaisons entre vos valeurs : trier à chaque valeur (`sorted`, `.sort()`, "
               "`statistics.median`), chercher le maximum avec `max(...)` ou insérer à la main en parcourant la liste, "
               "c'est des milliers de comparaisons par valeur (des centaines de millions pour 20 000). Avec deux tas "
               "(`heapq`), chaque ajout coûte O(log n) : une quinzaine de comparaisons.")
    try:
        derniere = _appel(tout_ajouter, (), f"ajouter() puis mediane() pour {n} valeurs", 173, lignes_max=80 * n,
                          conseil_lent=conseil, fichier=_fichier(classe))
    finally:
        _Compte.limite[0] = None
    if not _proche(derniere, _ref_mediane(valeurs), 1e-12):
        raise AssertionError(f"Après {n} valeurs, mediane() renvoie {_f(derniere)} au lieu de {_f(_ref_mediane(valeurs))}.")


# ---------------------------------------------------------------------------- défi 174 : echantillon_reservoir


def _lire_hasard() -> tuple:
    return random.getstate(), np.random.get_state()[1].tobytes(), np.random.get_state()[2]


def _frequences(f, n: int, k: int, essais: int) -> list[float]:
    comptes = [0] * n
    for graine in range(essais):
        for x in f(iter(range(n)), k, graine):
            comptes[x] += 1
    return [c / essais for c in comptes]


def verifier_echantillon_reservoir(f) -> None:
    # ------------------------------------------------------------------ les cas simples
    quoi = "echantillon_reservoir(range(1000), 10, graine=1)"
    r = _appel(f, (UneSeuleFois(range(1000)), 10, 1), quoi, 174, a_faire=True)
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie {type(r).__name__} : on attend une liste.")
    if len(r) != 10 or len(set(r)) != 10 or not all(isinstance(x, int) and 0 <= x < 1000 for x in r):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : on attend 10 valeurs DIFFÉRENTES du flux (chaque position de "
                             "l'échantillon est remplacée, jamais dupliquée).")
    for flux, k, attendu in ((["a", "b", "c"], 5, ["a", "b", "c"]), (["a", "b", "c"], 3, ["a", "b", "c"]),
                             ([], 3, []), (list(range(10)), 0, [])):
        quoi = f"echantillon_reservoir({flux}, {k}, graine=4)"
        r = _appel(f, (UneSeuleFois(flux), k, 4), quoi, 174)
        if r != attendu:
            raise AssertionError(f"{quoi} renvoie {_court(r)} au lieu de {attendu} : avec k valeurs ou moins, on les garde "
                                 "toutes, dans leur ordre d'arrivée." if k else
                                 f"{quoi} renvoie {_court(r)} au lieu de [] : k = 0, c'est un échantillon vide.")
    r = _appel(f, (UneSeuleFois(range(50)), 5, None), "echantillon_reservoir(range(50), 5, graine=None)", 174)
    if not isinstance(r, list) or len(r) != 5:
        raise AssertionError(f"echantillon_reservoir(range(50), 5, graine=None) renvoie {_court(r)} : sans graine, le "
                             "hasard n'est pas reproductible, mais il faut bien 5 valeurs.")

    for k, attendu, conseil in ((-1, ValueError, "Un échantillon de -1 valeur n'a pas de sens."),
                                (2.5, TypeError, "k est un nombre de valeurs : un entier."),
                                ("10", TypeError, "k est un nombre de valeurs : un entier, pas un texte."),
                                (True, TypeError, "True est un int pour Python : `isinstance(k, bool)` d'abord.")):
        _doit_lever(f, (UneSeuleFois(range(10)), k, 1), f"echantillon_reservoir(range(10), {k!r}, graine=1)", 174,
                    conseil, attendu)

    # ------------------------------------------------------------------ reproductible, et sans toucher au hasard global
    avant = _lire_hasard()
    r1 = _appel(f, (UneSeuleFois(range(500)), 8, 2024), "echantillon_reservoir(range(500), 8, graine=2024)", 174)
    if _lire_hasard() != avant:
        raise AssertionError("echantillon_reservoir a changé le hasard GLOBAL du programme (random.seed, np.random.seed, ou "
                             "random.random() sans générateur à vous) : tout le reste du programme tirerait d'autres "
                             "nombres. Créez `rng = random.Random(graine)` et tirez avec `rng.randrange(...)`.")
    r2 = _appel(f, (UneSeuleFois(range(500)), 8, 2024), "echantillon_reservoir(range(500), 8, graine=2024), relancé", 174)
    if r1 != r2:
        raise AssertionError(f"Avec la même graine, deux appels donnent {_court(r1)} puis {_court(r2)} : le tirage doit être "
                             "reproductible. `rng = random.Random(graine)`, une seule fois, avant la boucle.")
    differents = {tuple(_appel(f, (UneSeuleFois(range(500)), 8, g), f"echantillon_reservoir(range(500), 8, graine={g})",
                               174)) for g in range(20)}
    if len(differents) < 10:
        raise AssertionError(f"Avec 20 graines différentes, vous ne tirez que {len(differents)} échantillons différents : "
                             "utilisez-vous bien la graine reçue (et pas une graine écrite en dur) ?")

    # ------------------------------------------------------------------ chaque valeur a la même chance
    configurations = ((10, 3, 20_000), (7, 1, 14_000))
    frequences = {}
    for n, k, essais in configurations:
        try:
            frequences[n, k] = _frequences(f, n, k, essais)
        except Exception as exc:
            raise AssertionError(f"echantillon_reservoir(range({n}), {k}, graine) lève {_decrire(exc)}.") from None

    def biais(n, k, essais) -> bool:
        attendu = k / n
        ecart_type = math.sqrt(attendu * (1 - attendu) / essais)
        return any(abs(p - attendu) > 5 * ecart_type for p in frequences[n, k])

    for n, k, essais in configurations:
        if not biais(n, k, essais):
            continue
        freq = frequences[n, k]
        detail = ", ".join(f"la valeur {i} : {p:.1%}" for i, p in enumerate(freq))
        if frequences[7, 1][0] == 0:
            conseil = (" Avec k=1, la première valeur n'est JAMAIS gardée : avec `randrange(i)`, le tirage va de 0 à "
                       "i - 1 et tombe TOUJOURS sous k pour i = k. La valeur numéro i doit entrer avec la probabilité "
                       "k / (i + 1) : `randrange(i + 1)`, de 0 à i inclus.")
        elif not biais(7, 1, 14_000):
            conseil = (" Avec k=1, tout va bien : c'est donc la PLACE remplacée qui est mal choisie (remplacez-vous "
                       "toujours la plus ANCIENNE, ou la dernière entrée ?). Elle doit être tirée au hasard : "
                       "echantillon[j], avec le même j < k que le tirage.")
        else:
            conseil = (" La valeur numéro i (en comptant à partir de 0) doit entrer avec la probabilité k / (i + 1) : "
                       "`rng.randrange(i + 1) < k`, avec UN générateur `rng` créé avant la boucle (pas un nouveau "
                       "`random.Random(graine)` à chaque valeur : il retirerait toujours les mêmes nombres).")
        raise AssertionError(f"Sur {str(essais)[:-3]} {str(essais)[-3:]} tirages de echantillon_reservoir(range({n}), {k}, graine), chaque valeur "
                             f"devrait être choisie {k / n:.1%} du temps ; vous obtenez : {detail}.{conseil}")

    # ------------------------------------------------------------------ une mémoire qui dépend de k, pas du flux
    n = 300_000
    resultat, pic = _mesurer(lambda: f((i * 7 for i in range(n)), 10, 74),
                             f"echantillon_reservoir(un générateur de {n} valeurs, 10, graine=74)", 174)
    if pic > 400_000:
        raise AssertionError(f"Sur un flux de {n} valeurs avec k=10, echantillon_reservoir occupe {pic / 1e6:.1f} Mo : vous "
                             "gardez le flux (list(flux), puis rng.sample ou un tirage à la fin ?). La mémoire doit "
                             "dépendre de k, pas de la longueur du flux : seules les k valeurs de l'échantillon restent.")
    if not isinstance(resultat, list) or len(resultat) != 10:
        raise AssertionError(f"Sur un flux de {n} valeurs avec k=10, echantillon_reservoir renvoie {_court(resultat)}.")


# ---------------------------------------------------------------------------- défi 175 : fusionner


def _classe_mv(f):
    classe = getattr(f, "__globals__", {}).get("MoyenneVariance")
    if classe is None:
        raise AssertionError("fusionner travaille sur VOTRE classe MoyenneVariance (défi 170) : elle doit être dans le "
                             "même fichier.")
    return classe


def verifier_fusionner(f) -> None:
    classe = _classe_mv(f)
    try:
        vide_a, vide_b = classe(), classe()
    except NotImplementedError:
        try:
            f(None, None)
        except NotImplementedError as exc:
            _pas_encore(exc, 175, a_faire=True)
        except Exception:
            pass
        raise AssertionError("fusionner fusionne des MoyenneVariance, et votre classe n'est pas encore écrite (raise "
                             "NotImplementedError) : faites d'abord le défi 170.") from None
    except Exception as exc:
        raise AssertionError(f"MoyenneVariance() lève {_decrire(exc)} : faites d'abord le défi 170.") from None
    try:
        f(vide_a, vide_b)
    except NotImplementedError as exc:
        _pas_encore(exc, 175, a_faire=True)  # fusionner n'est pas encore écrite : le défi est « à faire »
    except Exception:
        pass  # on vérifie d'abord la classe : l'erreur sera redite plus bas, avec son contexte
    try:
        verifier_moyenne_variance(classe, _rapide=True)
    except NotImplementedError:
        raise AssertionError("Votre MoyenneVariance (défi 170) n'est pas terminée : faites d'abord le défi 170.") from None
    except AssertionError as exc:
        raise AssertionError(f"Votre MoyenneVariance ne passe pas encore le défi 170 ({str(exc).split('. ')[0]}). "
                             "Réussissez-le d'abord : fusionner s'appuie dessus.") from None

    def fusion(a_valeurs, b_valeurs, quoi_a=None, quoi_b=None, rel=1e-9):
        quoi_a = quoi_a or _court(a_valeurs, 60)
        quoi_b = quoi_b or _court(b_valeurs, 60)
        a = _remplir_mv(classe, a_valeurs, "a", 175)
        b = _remplir_mv(classe, b_valeurs, "b", 175)
        avant_a, avant_b = _etat_mv(a, "a", 175), _etat_mv(b, "b", 175)
        quoi = f"fusionner(a = {quoi_a}, b = {quoi_b})"
        r = _appel(f, (a, b), quoi, 175)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : il manque le `return`.")
        if not isinstance(r, classe):
            raise AssertionError(f"{quoi} renvoie {type(r).__name__} : on attend un nouvel objet MoyenneVariance.")
        if r is a or r is b:
            raise AssertionError(f"{quoi} renvoie {'a' if r is a else 'b'} lui-même : le résultat doit être un NOUVEL objet. "
                                 "a et b servent encore (on fusionne a avec c, puis avec d…).")
        for nom, objet, avant in (("a", a, avant_a), ("b", b, avant_b)):
            apres = _etat_mv(objet, nom, 175)
            if not all(_proche(x, y, 0) if _nombre(y) else x == y for x, y in zip(apres, avant)):
                raise AssertionError(f"{quoi} a modifié {nom} : (n, moyenne, variance) passe de {_court(avant)} à "
                                     f"{_court(apres)}. Créez un nouvel objet, et ne touchez pas à a ni à b.")
        obtenu = _etat_mv(r, "fusionner(a, b)", 175)
        tout = list(a_valeurs) + list(b_valeurs)
        n_att, moy_att, var_att = _ref_mv(tout)
        if obtenu[0] == n_att and len(a_valeurs) and len(b_valeurs) and len(tout) >= 2:
            ma, mb = _ref_mv(a_valeurs)[1], _ref_mv(b_valeurs)[1]
            if len(a_valeurs) != len(b_valeurs) and _proche(obtenu[1], (ma + mb) / 2, 1e-9) and not _proche(
                    obtenu[1], moy_att, 1e-9):
                raise AssertionError(f"{quoi} : moyenne vaut {_f(obtenu[1])} au lieu de {_f(moy_att)} : c'est la moyenne "
                                     "des deux moyennes. Elle doit être pondérée par les effectifs (10 valeurs ne pèsent "
                                     "pas autant que 1 000) : moyenne_a + (moyenne_b - moyenne_a) * nb / n.")
            if _proche(obtenu[1], moy_att, 1e-9) and not _proche(obtenu[2], var_att, rel):
                m2a = _ref_mv(a_valeurs)[2] * (len(a_valeurs) - 1) if len(a_valeurs) > 1 else 0.0
                m2b = _ref_mv(b_valeurs)[2] * (len(b_valeurs) - 1) if len(b_valeurs) > 1 else 0.0
                sans_ecart = [(m2a + m2b) / (len(tout) - 1)]
                if len(tout) > 2:
                    sans_ecart.append((m2a + m2b) / (len(tout) - 2))
                va, vb = _ref_mv(a_valeurs)[2], _ref_mv(b_valeurs)[2]
                if any(_proche(obtenu[2], v, 1e-6) for v in sans_ecart):
                    raise AssertionError(f"{quoi} : variance vaut {_f(obtenu[2])} au lieu de {_f(var_att)} : il manque "
                                         "l'écart ENTRE les deux morceaux. m2 = m2_a + m2_b + ecart ** 2 * na * nb / n, avec "
                                         "ecart = moyenne_b - moyenne_a. Deux morceaux sans dispersion interne (10, 10 et "
                                         "1 000, 1 000) ont bien une variance !")
                if not (math.isnan(va) or math.isnan(vb)) and _proche(obtenu[2], (va + vb) / 2, 1e-6):
                    raise AssertionError(f"{quoi} : variance vaut {_f(obtenu[2])} au lieu de {_f(var_att)} : c'est la "
                                         "moyenne des deux variances. Fusionnez les sommes des carrés des écarts (m2), et "
                                         "ajoutez l'écart entre les moyennes : m2 = m2_a + m2_b + ecart ** 2 * na * nb / n.")
        _comparer_mv(obtenu, tout, quoi, rel, _RAPPEL_FUSION)
        return r, tout

    # ------------------------------------------------------------------ l'exemple, et les erreurs classiques
    fusion([4, 7], [13, 16, 40])
    fusion([10, 10], [1000, 1000, 1000, 1000])
    fusion([1, 2, 3, 4, 5, 6, 7, 8, 9], [100])
    fusion([5], [7])
    fusion([5], [])
    fusion([], [3, 8, 1])
    fusion([], [])
    fusion([2.5], [-1.5, 0.5])

    # ------------------------------------------------------------------ le résultat continue de vivre
    r, tout = fusion([4, 7], [13, 16, 40])
    _appel(r.ajouter, (100,), "fusionner(…).ajouter(100)", 175)
    _comparer_mv(_etat_mv(r, "r", 175), tout + [100], "Après r = fusionner(a = [4, 7], b = [13, 16, 40]) puis "
                 "r.ajouter(100) (l'objet fusionné doit avoir TOUT son état interne à jour : n, moyenne ET la somme des "
                 "carrés des écarts)", rappel=_RAPPEL_FUSION)
    r, tout = fusion([], [3, 8, 1])
    _appel(r.ajouter, (4,), "fusionner(vide, …).ajouter(4)", 175)
    _comparer_mv(_etat_mv(r, "r", 175), tout + [4], "Après r = fusionner(a = vide, b = [3, 8, 1]) puis r.ajouter(4)",
                 rappel=_RAPPEL_FUSION)

    # ------------------------------------------------------------------ autour de 1,7 milliard
    alea = random.Random(175)
    morceaux = [[1_700_000_000.0 + 1000 * m + alea.randrange(-5, 6) for _ in range(alea.randrange(1, 8))]
                for m in range(6)]
    a = _remplir_mv(classe, morceaux[0], "morceau 0", 175)
    for i, morceau in enumerate(morceaux[1:], start=1):
        a = _appel(f, (a, _remplir_mv(classe, morceau, f"morceau {i}", 175)), f"fusionner(…, morceau {i})", 175)
    tout = [x for m in morceaux for x in m]
    decales = [x - 1_700_000_000.0 for x in tout]  # soustraction exacte : la variance ne change pas
    attendu = _ref_mv(decales)[2]
    obtenu = _lire(a, "variance", "variance", 175)
    if not _proche(obtenu, attendu, 1e-6):
        raise AssertionError(f"Six morceaux de connexions autour de 1 700 000 000 (secondes), fusionnés un par un : variance "
                             f"vaut {_f(obtenu)} au lieu de {_f(attendu)}. Ne passez pas par des sommes de carrés "
                             "(Σx²) : fusionnez les moyennes et les m2 (sommes des carrés des ÉCARTS).")

    # ------------------------------------------------------------------ dix machines : en chaîne ou en arbre, même résultat
    valeurs = list(generer_montants(4000, graine=1750))
    resumes = []
    for i in range(0, len(valeurs), 125):
        resumes.append(_remplir_mv(classe, valeurs[i:i + 125], f"montants {i} à {i + 124}", 175))
    chaine = resumes[0]
    for r in resumes[1:]:
        chaine = _appel(f, (chaine, r), "fusionner en chaîne", 175)
    niveau = resumes
    while len(niveau) > 1:
        niveau = [_appel(f, (niveau[i], niveau[i + 1]), "fusionner en arbre", 175) if i + 1 < len(niveau) else niveau[i]
                  for i in range(0, len(niveau), 2)]
    for nom, resultat in (("en chaîne", chaine), ("deux à deux (en arbre)", niveau[0])):
        _comparer_mv(_etat_mv(resultat, "r", 175), valeurs,
                     f"32 résumés de 125 montants, fusionnés {nom}", rel=1e-9, rappel=_RAPPEL_FUSION)


# ---------------------------------------------------------------------------- la série

SERIE = Serie(
    code="28",
    titre="Les données qui n'arrêtent pas d'arriver : Welford, CUSUM, demi-vie, deux tas, réservoir, fusion",
    module_eleve="defis.serie_28",
    module_corrige="defis.corrige_serie_28",
    defis=(
        Defi(170, "MoyenneVariance", "🟢", "2.1", verifier_moyenne_variance),
        Defi(171, "alertes_cusum", "🟢", "7.6", verifier_alertes_cusum),
        Defi(172, "ActiviteRecente", "🟡", "2.3", verifier_activite_recente),
        Defi(173, "MedianeCourante", "🟡", "—", verifier_mediane_courante),
        Defi(174, "echantillon_reservoir", "🟡", "—", verifier_echantillon_reservoir),
        Defi(175, "fusionner", "🔴", "—", verifier_fusionner),
    ),
)

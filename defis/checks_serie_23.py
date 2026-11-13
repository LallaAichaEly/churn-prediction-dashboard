"""Vérifications automatiques de la série 23 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur traite vos fonctions comme un numéricien :

- il les appelle avec des nombres extrêmes (z = ±1000, un client « sûr à 100 % » qui part quand même)
  en transformant chaque `RuntimeWarning` (débordement, log de 0) en erreur : un calcul qui « marche
  avec un avertissement » donne tôt ou tard inf ou NaN au milieu d'un entraînement ;
- il compare votre gradient aux DIFFÉRENCES FINIES (`gradient_numerique`) et votre régression
  logistique à celle de scikit-learn : les coefficients doivent être les mêmes à 1e-5 près ;
- il compare vos coupures et vos arbres à ceux de scikit-learn (`DecisionTreeClassifier`), nœud par
  nœud : à chaque nœud, il recalcule toutes les coupures possibles et vérifie que la vôtre est la
  meilleure, ou à égalité avec elle ;
- il compte les lignes de VOTRE fichier exécutées pour 1 000 puis 10 000 clients (`sys.settrace`) : si
  ce nombre grandit avec les données, c'est une boucle Python par client, quelle que soit la vitesse de
  la machine ; puis il chronomètre sur des centaines de milliers de lignes, où une boucle dans une
  boucle se voit tout de suite.

Quand un résultat est faux, il rejoue les versions boguées classiques (la somme au lieu de la moyenne,
le signe du gradient, l'ordonnée à l'origine pénalisée, une coupure entre deux valeurs égales, des
enfants non pondérés, un arbre qui s'arrête dès que le gain est nul…) pour vous dire LAQUELLE vous avez
écrite.
"""

from __future__ import annotations

import math
import numbers
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .maison import NUMERIQUES, clients_exemple, generer_clients, gradient_numerique
from .verif import Defi, Serie

# ---------------------------------------------------------------------------- outils


def _decrire(exc: BaseException) -> str:
    texte = str(exc).split("\n")[0]
    if len(texte) > 160:
        texte = texte[:159] + "…"
    return f"{type(exc).__name__}({texte!r})" if texte else type(exc).__name__


def _court(valeur, n: int = 120) -> str:
    if isinstance(valeur, np.ndarray) and valeur.size > 8:
        return f"un tableau de forme {valeur.shape}"
    if isinstance(valeur, BaseEstimator):
        return f"un objet {type(valeur).__name__}"
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


def _nb(valeur) -> str:
    """Un nombre lisible : 4.248e-18, 0.5, 400, nan, inf."""
    try:
        valeur = float(valeur)
    except (TypeError, ValueError):
        return _court(valeur)
    if valeur != valeur:
        return "nan"
    if valeur in (math.inf, -math.inf):
        return "inf" if valeur > 0 else "-inf"
    if valeur == 0:
        return "0.0"
    if abs(valeur) < 1e-4 or abs(valeur) >= 1e6:
        return f"{valeur:.4g}"
    return f"{valeur:.6g}"


AVERTISSEMENT = ("Un RuntimeWarning n'est pas « juste un avertissement » : c'est un inf ou un NaN qui vient de naître, et "
                 "qui finira dans vos coefficients.")


def _appel(appel, quoi: str, conseil: str = "", conseil_avertissement: str = ""):
    """Appelle `appel()` en transformant les RuntimeWarning (débordement, log de 0…) en erreurs lisibles."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.simplefilter("error", RuntimeWarning)
            return appel()
    except NotImplementedError:
        raise
    except RuntimeWarning as exc:
        raise AssertionError(f"{quoi} déclenche « RuntimeWarning: {exc} ». {conseil_avertissement or AVERTISSEMENT}"
                             .rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None


def _leve(appel, attendues, quoi: str, conseil: str = "") -> None:
    """Exige que `appel()` lève une des exceptions `attendues`."""
    attendues = attendues if isinstance(attendues, tuple) else (attendues,)
    noms = " ou ".join(e.__name__ for e in attendues)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resultat = appel()
    except NotImplementedError:
        raise
    except attendues:
        return
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {noms}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {noms}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _chronometrer(appel, seuil: float):
    """(résultat, durée) : le meilleur de deux essais, pour ne pas pénaliser une machine occupée."""
    duree, resultat = math.inf, None
    for _ in range(2):
        debut = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resultat = appel()
        duree = min(duree, time.perf_counter() - debut)
        if duree < seuil:
            break
    return resultat, duree


def _lignes_executees(f, appel) -> int:
    """Nombre de lignes exécutées dans le FICHIER de f pendant appel() (sys.settrace).

    Une boucle Python par client exécute au moins une ligne par client ; numpy, lui, boucle en C et
    n'en exécute aucune. Ce compte ne dépend pas de la vitesse de la machine, contrairement au
    chronomètre : c'est lui qui décide s'il y a une boucle.
    """
    code = getattr(f, "__code__", None)
    if code is None:
        return 0
    fichier, compte = code.co_filename, [0]

    def local(frame, evenement, arg):
        if evenement == "line":
            compte[0] += 1
        return local

    def global_(frame, evenement, arg):
        return local if frame.f_code.co_filename == fichier else None

    precedent = sys.gettrace()
    sys.settrace(global_)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            appel()
    finally:
        sys.settrace(precedent)
    return compte[0]


def _boucle_par_ligne(f, appel_de_taille, petit: int = 1_000, grand: int = 10_000) -> tuple[int, int] | None:
    """(lignes pour `petit`, lignes pour `grand`) si le nombre de lignes exécutées grandit avec les données."""
    a = _lignes_executees(f, lambda: appel_de_taille(petit))
    b = _lignes_executees(f, lambda: appel_de_taille(grand))
    return (a, b) if b - a > (grand - petit) // 10 else None


def _proches(a, b, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    try:
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=rtol, atol=atol,
                                equal_nan=True))
    except (TypeError, ValueError):
        return False


def _verifier_intact(apres, avant, quoi: str) -> None:
    if not np.array_equal(np.asarray(apres), np.asarray(avant), equal_nan=True):
        raise AssertionError(f"{quoi} a MODIFIÉ le tableau reçu. Une fonction de calcul ne touche pas à ses entrées : "
                             f"l'appelant s'en sert encore (`z *= -1`, `x.sort()` ou `w -= …` modifient sur place ; "
                             f"`-z`, `np.argsort(x)` et `w - …` ne touchent pas au tableau reçu).")


def _exiger_nombre(resultat, quoi: str) -> float:
    if resultat is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if isinstance(resultat, (bool, np.bool_)) or np.ndim(resultat) != 0:
        raise AssertionError(f"{quoi} renvoie {_court(resultat)} au lieu d'un nombre (un float).")
    try:
        return float(resultat)
    except (TypeError, ValueError):
        raise AssertionError(f"{quoi} renvoie {_court(resultat)} au lieu d'un nombre (un float).") from None


# ---------------------------------------------------------------------------- outils scikit-learn (série 22)


def _verifier_parametres(objet, attendus: dict, quoi: str) -> None:
    if not hasattr(objet, "get_params"):
        raise AssertionError(f"{quoi} n'a pas de `get_params` : la classe doit hériter de `BaseEstimator` (à droite du "
                             f"mixin : `class X(ClassifierMixin, BaseEstimator):`), comme dans la série 22.")
    try:
        parametres = objet.get_params(deep=False)
    except (AttributeError, RuntimeError) as exc:
        raise AssertionError(f"{quoi}.get_params() lève {_decrire(exc)} : `__init__` doit ranger CHAQUE paramètre TEL "
                             f"QUEL sous SON nom (`self.pas = pas`).") from None
    if parametres != attendus:
        raise AssertionError(f"{quoi}.get_params() donne {parametres} au lieu de {attendus} : mêmes noms, mêmes valeurs "
                             f"par défaut, rangés tels quels.")


def _construit_sans_verifier(Classe, quoi: str, **parametres) -> None:
    try:
        objet = Classe(**parametres)
    except NotImplementedError:
        raise
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} dès la construction. Comme dans la série 22 : `__init__` range "
                             f"les paramètres tels quels, et c'est `fit` qui les vérifie (set_params ne repasse pas par "
                             f"`__init__`).") from None
    for nom, valeur in parametres.items():
        if getattr(objet, nom, object()) is not valeur:
            raise AssertionError(f"{quoi} : l'attribut `{nom}` ne vaut pas exactement la valeur reçue ({valeur!r}).")


def _verifier_pas_entraine(appel, quoi: str) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resultat = appel()
    except NotImplementedError:
        raise
    except NotFittedError:
        return
    except Exception as exc:
        raise AssertionError(f"{quoi} AVANT fit lève {_decrire(exc)} au lieu de NotFittedError : commencez par "
                             f"`check_is_fitted(self)`.") from None
    raise AssertionError(f"{quoi} AVANT fit renvoie {_court(resultat)} au lieu de lever NotFittedError : commencez par "
                         f"`check_is_fitted(self)`.")


def _verifier_fit_renvoie_self(objet, retour, quoi: str) -> None:
    if retour is not objet:
        raise AssertionError(f"{quoi} renvoie {_court(retour)} au lieu de l'objet lui-même : `fit` se termine par "
                             f"`return self`.")


def _verifier_classifieur(Classe, quoi: str) -> None:
    objet = _appel(lambda: Classe(), f"{quoi}()")
    mro = type(objet).__mro__
    if ClassifierMixin in mro and BaseEstimator in mro and mro.index(ClassifierMixin) > mro.index(BaseEstimator):
        raise AssertionError(f"{quoi} : le mixin se place À GAUCHE, `class {Classe.__name__}(ClassifierMixin, "
                             f"BaseEstimator):` (sinon scikit-learn ne voit pas que c'est un classifieur).")
    if not _appel(lambda: is_classifier(objet), f"is_classifier({quoi}())"):
        raise AssertionError(f"{quoi} n'est pas reconnu comme un classifieur : `class {Classe.__name__}(ClassifierMixin, "
                             f"BaseEstimator):` (le mixin À GAUCHE). Sans lui, `cross_val_score` ne découpe pas les plis "
                             f"en gardant la proportion de partants, et `score` n'existe pas.")


def _verifier_clone(objet, quoi: str):
    try:
        return clone(objet)
    except Exception as exc:
        raise AssertionError(f"clone({quoi}) lève {_decrire(exc)} : héritez de `BaseEstimator` et rangez les paramètres "
                             f"tels quels.") from None


def _standardiser(X):
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


# ============================================================================ Défi 140 : sigmoide


def _ref_sigmoide(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


_CONSEIL_SIGMOIDE = ("np.exp(-z) déborde dès que z < -709 : exp(1000) = inf. Écrivez la sigmoïde pour que l'exponentielle "
                     "ne reçoive JAMAIS un nombre positif : `e = np.exp(-np.abs(z))` est toujours entre 0 et 1, puis "
                     "`np.where(z >= 0, 1 / (1 + e), e / (1 + e))`. Attention : `np.where` calcule ses DEUX branches "
                     "partout, donc `np.where(z >= 0, 1 / (1 + np.exp(-z)), …)` déborde quand même.")


def verifier_sigmoide(f) -> None:
    # 1. Un nombre.
    obtenu = _exiger_nombre(_appel(lambda: f(0.0), "sigmoide(0.0)"), "sigmoide(0.0)")
    if obtenu != 0.5:
        raise AssertionError(f"sigmoide(0.0) vaut {_nb(obtenu)} au lieu de 0.5 : 1 / (1 + exp(-0)) = 1 / 2.")

    # 2. Les extrêmes, sans avertissement.
    quoi = "sigmoide(np.array([-1000.0, 0.0, 1000.0]))"
    obtenu = _appel(lambda: f(np.array([-1000.0, 0.0, 1000.0])), quoi, conseil_avertissement=_CONSEIL_SIGMOIDE)
    if obtenu is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(obtenu, np.ndarray):
        raise AssertionError(f"{quoi} renvoie {type(obtenu).__name__} au lieu d'un tableau numpy : pas de boucle ni de "
                             f"liste, une seule opération sur tout le tableau (`np.exp` agit case par case).")
    if obtenu.shape != (3,) or not _proches(obtenu, [0.0, 0.5, 1.0], rtol=0, atol=0):
        if obtenu.shape == (3,) and 0 < obtenu[0] < 1e-200:
            raise AssertionError(f"{quoi} donne {_court(obtenu)} : sigmoide(-1000) vaut {_nb(obtenu[0])} au lieu de 0. Vous "
                                 f"rabotez z (`np.clip` ?) avant de calculer : ça évite le débordement, mais fausse les "
                                 f"valeurs extrêmes (log(sigmoide(-600)) doit valoir -600, pas -500). Réécrivez plutôt la "
                                 f"formule : e = exp(-|z|) ne déborde jamais.")
        raise AssertionError(f"{quoi} donne {_court(obtenu)} au lieu de [0. , 0.5, 1. ].")

    # 3. Tout l'éventail, à la précision de la machine.
    z = np.array([-600.0, -100.0, -40.0, -5.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 5.0, 40.0, 100.0, 600.0])
    quoi = "sigmoide(z)"
    obtenu = np.asarray(_appel(lambda: f(z.copy()), quoi, conseil_avertissement=_CONSEIL_SIGMOIDE), dtype=float)
    attendu = _ref_sigmoide(z)
    if obtenu.shape != z.shape:
        raise AssertionError(f"sigmoide d'un tableau de {z.size} nombres renvoie un tableau de forme {obtenu.shape} : "
                             f"même forme que z.")
    for valeur, o, a in zip(z, obtenu, attendu):
        if not math.isclose(o, a, rel_tol=1e-12, abs_tol=0.0):
            if valeur < 0 and o == 0.0:
                raise AssertionError(
                    f"sigmoide({_nb(valeur)}) vaut 0.0 au lieu de {_nb(a)} : tous les chiffres sont perdus. Calculer "
                    f"`1 - (un nombre très proche de 1)` ne laisse que du bruit (1 - 1/(1 + 4e-18) = 1 - 1.0 = 0). Pour z < 0, "
                    f"écrivez directement e / (1 + e) avec e = exp(z) = exp(-|z|) : aucune soustraction.")
            if valeur == -600.0 and o > a * 1e6:
                raise AssertionError(
                    f"sigmoide(-600) vaut {_nb(o)} au lieu de {_nb(a)} : vous rabotez z (`np.clip` ?) avant de calculer. "
                    f"Ça évite le débordement, mais fausse les valeurs extrêmes : log(sigmoide(-600)) doit valoir -600, "
                    f"pas -500. Réécrivez plutôt la formule : e = exp(-|z|) ne déborde jamais.")
            raise AssertionError(f"sigmoide({_nb(valeur)}) vaut {_nb(o)} au lieu de {_nb(a)} (1 / (1 + exp(-z))).")

    # 4. N'importe quelle forme, n'importe quel type de nombres, une liste.
    z2 = np.arange(-6, 6).reshape(3, 4)
    quoi = "sigmoide(np.arange(-6, 6).reshape(3, 4))  (un tableau d'ENTIERS à 2 dimensions)"
    obtenu = _appel(lambda: f(z2.copy()), quoi, "Pas de boucle : `np.asarray(z, dtype=float)`, puis des opérations "
                                                "numpy, qui marchent quelle que soit la forme du tableau.")
    if not isinstance(obtenu, np.ndarray) or obtenu.shape != (3, 4) or obtenu.dtype.kind != "f":
        raise AssertionError(f"{quoi} doit renvoyer un tableau de floats de forme (3, 4), pas {_court(obtenu)}"
                             f"{' de type ' + str(obtenu.dtype) if isinstance(obtenu, np.ndarray) else ''}.")
    if not _proches(obtenu, _ref_sigmoide(z2), rtol=1e-12):
        raise AssertionError(f"{quoi} donne des valeurs fausses : avez-vous converti en float (`dtype=float`) ?")
    obtenu = _appel(lambda: f([0, 2]), "sigmoide([0, 2])  (une liste)",
                    "Commencez par `z = np.asarray(z, dtype=float)` : une liste devient un tableau.")
    if not isinstance(obtenu, np.ndarray) or not _proches(obtenu, _ref_sigmoide([0, 2]), rtol=1e-12):
        raise AssertionError(f"sigmoide([0, 2]) renvoie {_court(obtenu)} au lieu de {_court(_ref_sigmoide([0, 2]))} (un "
                             f"tableau numpy).")
    obtenu = _appel(lambda: f(np.array([np.nan, 1.0])), "sigmoide(np.array([nan, 1.0]))")
    if not (np.isnan(np.asarray(obtenu, dtype=float)[0]) and math.isclose(np.asarray(obtenu)[1], _ref_sigmoide(1.0))):
        raise AssertionError(f"sigmoide(np.array([nan, 1.0])) renvoie {_court(obtenu)} : NaN reste NaN (une donnée "
                             f"manquante ne devient pas une probabilité), et le reste est calculé normalement.")

    # 5. L'entrée est intacte.
    z3 = np.array([-3.0, 0.0, 2.5, 800.0])
    _appel(lambda: f(z3), "sigmoide(z)")
    _verifier_intact(z3, np.array([-3.0, 0.0, 2.5, 800.0]), "sigmoide(z)")

    # 6. Vectorisée : d'abord compter les lignes exécutées (ne dépend pas de la machine), puis chronométrer.
    scores = np.random.default_rng(1400).normal(0, 6, 10_000)
    boucle = _boucle_par_ligne(f, lambda n: f(scores[:n].copy()))
    if boucle:
        raise AssertionError(f"Les valeurs sont justes, mais votre code fait une boucle Python sur les scores : 1 000 scores "
                             f"exécutent {boucle[0]} lignes de votre fichier, 10 000 en exécutent {boucle[1]}. Sur 2 millions "
                             f"de scores, c'est 2 millions d'appels de fonction (environ 10 fois plus lent que numpy). "
                             f"`np.exp`, `np.abs` et `np.where` traitent tout le tableau d'un coup, en C : aucune ligne ne "
                             f"se répète.")
    grand = np.random.default_rng(140).normal(0, 6, 2_000_000)
    obtenu, duree = _chronometrer(lambda: f(grand), seuil=0.25)
    if duree >= 0.25:
        raise AssertionError(f"Les valeurs sont justes, mais 2 millions de scores prennent {duree:.2f} s (numpy : environ "
                             f"0,03 s). Une boucle Python appelle une fonction par case ; `np.exp`, `np.abs` et `np.where` "
                             f"traitent tout le tableau d'un coup, en C.")
    if not _proches(obtenu, _ref_sigmoide(grand), rtol=1e-12):
        raise AssertionError("sigmoide sur 2 millions de scores tirés au hasard donne des valeurs fausses.")


# ============================================================================ Défi 141 : perte_logistique


def _ref_perte(y, z, variante=None) -> float:
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    par_client = np.logaddexp(0.0, z) - y * z
    if variante == "somme":
        return float(np.sum(par_client))
    if variante == "un_terme":          # -y·log(p) seulement : les clients restés ne comptent pas
        return float(np.mean(y * np.logaddexp(0.0, -z)))
    if variante == "autre_terme":       # -(1-y)·log(1-p) seulement
        return float(np.mean((1 - y) * np.logaddexp(0.0, z)))
    if variante == "signe":
        return -float(np.mean(par_client))
    if variante == "inverse":           # y et 1 - y échangés
        return float(np.mean(np.logaddexp(0.0, z) - (1 - y) * z))
    if variante == "log10":
        return float(np.mean(par_client) / np.log(10))
    if variante == "log2":
        return float(np.mean(par_client) / np.log(2))
    if variante == "clip":
        p = np.clip(_ref_sigmoide(z), 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean(par_client))


_PIEGES_PERTE = [
    ("somme", "c'est la SOMME des pertes, pas leur MOYENNE (`np.mean`). Avec la somme, la perte d'un fichier de 1 million "
              "de clients est 1000 fois celle d'un fichier de 1000 : impossible de comparer deux modèles, et le pas de la "
              "descente de gradient devrait changer avec la taille des données."),
    ("un_terme", "seuls les clients PARTIS (y = 1) comptent : il manque le terme -(1 - y)·log(1 - p) des clients restés. "
                 "Un modèle qui prédit « tout le monde part » aurait une perte nulle."),
    ("autre_terme", "seuls les clients RESTÉS (y = 0) comptent : il manque le terme -y·log(p) des clients partis."),
    ("signe", "la perte est NÉGATIVE : il manque le signe moins devant y·log(p) + (1 - y)·log(1 - p) (un log de "
              "probabilité est négatif, la perte, elle, est positive)."),
    ("inverse", "y et 1 - y sont échangés : un client parti (y = 1) doit être puni quand p est PETIT, avec -log(p)."),
    ("log10", "c'est le logarithme en base 10 (`np.log10`) : la log-loss utilise le logarithme naturel (`np.log`)."),
    ("log2", "c'est le logarithme en base 2 (`np.log2`) : la log-loss utilise le logarithme naturel (`np.log`)."),
]

_CONSEIL_PERTE = ("Ne passez pas par p = sigmoide(z) : dès que |z| dépasse 37, p vaut EXACTEMENT 1.0 (ou 0.0) et log(1 - p) = "
                  "log(0) = -inf. Simplifiez d'abord la formule : -[y·log(p) + (1 - y)·log(1 - p)] = log(1 + exp(z)) - y·z, "
                  "et `np.logaddexp(0, z)` calcule log(1 + exp(z)) sans jamais calculer exp(z).")


def verifier_perte_logistique(f) -> None:
    # 1. Le cas d'école : un modèle qui ne sait rien (p = 0,5 pour tout le monde) a une perte de log(2).
    quoi = "perte_logistique([1, 0], [0.0, 0.0])"
    obtenu = _exiger_nombre(_appel(lambda: f([1, 0], [0.0, 0.0]), quoi), quoi)
    if not math.isclose(obtenu, math.log(2), rel_tol=1e-12):
        for variante, conseil in _PIEGES_PERTE:
            if math.isclose(obtenu, _ref_perte([1, 0], [0.0, 0.0], variante), rel_tol=1e-12):
                raise AssertionError(f"{quoi} vaut {_nb(obtenu)} au lieu de log(2) = 0.693147 : {conseil}")
        raise AssertionError(f"{quoi} vaut {_nb(obtenu)} au lieu de log(2) = 0.693147 : un modèle qui donne p = 0,5 à "
                             f"chacun perd log(2) par client.")

    # 2. Des scores ordinaires : comme sklearn.metrics.log_loss.
    rng = np.random.default_rng(141)
    y = rng.integers(0, 2, 60)
    z = rng.normal(0.3, 2.5, 60)
    quoi = "perte_logistique(y, z)  (60 clients, des scores entre -7 et 7)"
    obtenu = _exiger_nombre(_appel(lambda: f(y.copy(), z.copy()), quoi), quoi)
    attendu = _ref_perte(y, z)
    if not math.isclose(obtenu, attendu, rel_tol=1e-10):
        for variante, conseil in _PIEGES_PERTE:
            if math.isclose(obtenu, _ref_perte(y, z, variante), rel_tol=1e-10):
                raise AssertionError(f"{quoi} vaut {_nb(obtenu)} au lieu de {_nb(attendu)} : {conseil}")
        raise AssertionError(f"{quoi} vaut {_nb(obtenu)} au lieu de {_nb(attendu)} (c'est aussi ce que donne "
                             f"`sklearn.metrics.log_loss(y, sigmoide(z))` = {_nb(log_loss(y, _ref_sigmoide(z)))}).")

    # 3. Un modèle sûr de lui… et faux.
    y4, z4 = np.array([1, 0, 1, 0]), np.array([-800.0, 800.0, 50.0, -50.0])
    quoi = "perte_logistique([1, 0, 1, 0], [-800.0, 800.0, 50.0, -50.0])"
    obtenu = _exiger_nombre(_appel(lambda: f(y4.copy(), z4.copy()), quoi, conseil_avertissement=_CONSEIL_PERTE), quoi)
    if not math.isclose(obtenu, 400.0, rel_tol=1e-12):
        if not math.isfinite(obtenu):
            raise AssertionError(f"{quoi} vaut {_nb(obtenu)} au lieu de 400 : {_CONSEIL_PERTE}")
        if math.isclose(obtenu, _ref_perte(y4, z4, "clip"), rel_tol=1e-6):
            raise AssertionError(
                f"{quoi} vaut {_nb(obtenu)} au lieu de 400 : vous bornez p entre 1e-15 et 1 - 1e-15 (`np.clip`). Ça évite "
                f"log(0), mais plafonne la perte d'un client à 34,5 : un modèle sûr à 100 % et FAUX n'est pas assez puni, et "
                f"le gradient s'annule. Calculez à partir des scores : log(1 + exp(z)) - y·z avec `np.logaddexp(0, z)`.")
        raise AssertionError(f"{quoi} vaut {_nb(obtenu)} au lieu de 400 (les deux premiers clients coûtent 800 chacun, les "
                             f"deux autres presque rien). {_CONSEIL_PERTE}")

    # 4. Toutes les formes d'entrée, par POSITION.
    for forme, (yy, zz) in {
        "y en booléens": (y.astype(bool), z),
        "y et z en listes, z en entiers": (y.tolist(), np.round(z).astype(int).tolist()),
        "y et z en Series d'index DIFFÉRENTS": (pd.Series(y, index=np.arange(100, 160)), pd.Series(z)),
    }.items():
        quoi = f"perte_logistique(y, z)  ({forme})"
        obtenu = _exiger_nombre(_appel(lambda: f(yy, zz), quoi), quoi)
        attendu = _ref_perte(np.asarray(yy), np.asarray(zz))
        if not math.isclose(obtenu, attendu, rel_tol=1e-10):
            conseil = (" Deux Series s'alignent par ÉTIQUETTE d'index (100 ne rencontre jamais 0 : NaN) : le i-ème y va "
                       "avec le i-ème z. Convertissez d'abord : `np.asarray(y, dtype=float)`." if "Series" in forme else "")
            raise AssertionError(f"{quoi} vaut {_nb(obtenu)} au lieu de {_nb(attendu)}.{conseil}")

    # 5. Ce qu'il faut refuser.
    for yy, zz, pourquoi in (([1, 0, 1], [0.5, 0.2], "y et z de longueurs différentes"),
                             ([], [], "aucun client (la moyenne d'une liste vide n'existe pas)"),
                             ([0, 2, 1], [0.1, 0.2, 0.3], "y contient 2 : y vaut 0 ou 1"),
                             ([-1, 1, 1], [0.1, 0.2, 0.3], "y en -1 / +1 (l'autre convention) : convertissez d'abord"),
                             ([0.5, 1, 0], [0.1, 0.2, 0.3], "y contient 0.5 : y est une réponse, pas une probabilité")):
        _leve(lambda: f(yy, zz), ValueError, f"perte_logistique({yy}, {zz})",
              f"Refusez avec ValueError : {pourquoi}. Une perte calculée sur des données incohérentes est un chiffre "
              f"faux qui a l'air juste (`np.isin(y, (0, 1)).all()`).")

    # 6. Les entrées sont intactes, et c'est rapide.
    yy, zz = y.astype(float), z.copy()
    _appel(lambda: f(yy, zz), "perte_logistique(y, z)")
    _verifier_intact(yy, y.astype(float), "perte_logistique(y, z)")
    _verifier_intact(zz, z, "perte_logistique(y, z)")
    rng = np.random.default_rng(1412)
    petit_y, petit_z = rng.integers(0, 2, 10_000), rng.normal(0, 4, 10_000)
    boucle = _boucle_par_ligne(f, lambda n: f(petit_y[:n].copy(), petit_z[:n].copy()))
    if boucle:
        raise AssertionError(f"Les valeurs sont justes, mais votre code fait une boucle Python sur les clients : 1 000 "
                             f"clients exécutent {boucle[0]} lignes de votre fichier, 10 000 en exécutent {boucle[1]}. Sur "
                             f"2 millions de clients, c'est 2 millions de tours. Pas de boucle : une seule expression sur "
                             f"les tableaux entiers (`np.mean(np.logaddexp(0, z) - y * z)`).")
    grand_y = np.random.default_rng(1410).integers(0, 2, 2_000_000)
    grand_z = np.random.default_rng(1411).normal(0, 4, 2_000_000)
    obtenu, duree = _chronometrer(lambda: f(grand_y, grand_z), seuil=0.4)
    if duree >= 0.4:
        raise AssertionError(f"Les valeurs sont justes, mais 2 millions de clients prennent {duree:.2f} s (numpy : environ "
                             f"0,05 s). Pas de boucle : une seule expression sur les tableaux entiers.")
    if not math.isclose(float(obtenu), _ref_perte(grand_y, grand_z), rel_tol=1e-9):
        raise AssertionError("perte_logistique sur 2 millions de clients tirés au hasard donne une valeur fausse.")


# ============================================================================ Défi 142 : gradient_logistique


def _ref_gradient(X, y, w, b, alpha=0.0, variante=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(y)
    erreur = _ref_sigmoide(X @ w + (0.0 if variante == "sans_b" else b)) - y
    if variante == "signe":
        erreur = -erreur
    if variante == "somme":
        grad_w, grad_b = X.T @ erreur, float(erreur.sum())
    else:
        grad_w, grad_b = X.T @ erreur / n, float(erreur.mean())
    penalite = {"sans_alpha": 0.0, "deux_alpha": 2 * alpha * w, "demi_alpha": alpha / 2 * w}.get(variante, alpha * w)
    if variante == "penalite_divisee":
        penalite = alpha * w / n
    grad_w = grad_w + penalite
    if variante == "b_penalise":
        grad_b = grad_b + alpha * b
    return grad_w, grad_b


_PIEGES_GRADIENT = [
    ("somme", "vous additionnez sur les clients sans diviser par n : la perte est une MOYENNE, son gradient aussi "
              "(`X.T @ erreur / n` et `erreur.mean()`)."),
    ("signe", "le gradient a le mauvais SIGNE : la dérivée de la perte par rapport au score z est sigmoide(z) - y (la "
              "probabilité MOINS la réponse), pas y - sigmoide(z)."),
    ("sans_b", "le score oublie l'ordonnée à l'origine : z = X @ w + b."),
    ("b_penalise", "l'ordonnée à l'origine est pénalisée : la pénalité (alpha / 2)·||w||² ne porte que sur w (comme "
                   "dans scikit-learn). grad_b = moyenne de l'erreur, sans + alpha·b."),
    ("sans_alpha", "la pénalité est oubliée : la dérivée de (alpha / 2)·||w||² est alpha·w, à ajouter à grad_w."),
    ("deux_alpha", "la pénalité compte double : la dérivée de (alpha / 2)·||w||² est alpha·w (le 2 de w² et le 1/2 se "
                   "simplifient), pas 2·alpha·w."),
    ("demi_alpha", "la pénalité compte à moitié : la dérivée de (alpha / 2)·||w||² est alpha·w, pas (alpha / 2)·w."),
    ("penalite_divisee", "la pénalité est divisée par n : seule la PERTE est une moyenne sur les clients ; la dérivée de "
                         "(alpha / 2)·||w||² est alpha·w, tout simplement."),
]


_CONSEIL_FORMES = ("Les dimensions : X est (n, d), l'erreur (n,), donc `X.T @ erreur` est (d,), une somme par colonne — "
                   "et `X @ erreur` n'existe pas quand n ≠ d.")


def _exiger_gradient(resultat, d: int, quoi: str):
    if resultat is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return grad_w, grad_b` ?")
    if not isinstance(resultat, tuple) or len(resultat) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(resultat)} au lieu d'un couple (grad_w, grad_b).")
    grad_w, grad_b = resultat
    if not isinstance(grad_w, np.ndarray) or grad_w.shape != (d,):
        forme = grad_w.shape if isinstance(grad_w, np.ndarray) else type(grad_w).__name__
        raise AssertionError(f"{quoi} : grad_w doit être un tableau numpy de forme ({d},), une dérivée par coefficient, pas "
                             f"{forme}.")
    return grad_w, _exiger_nombre(grad_b, f"{quoi} : grad_b")


def _comparer_gradient(obtenu, X, y, w, b, alpha, quoi: str) -> None:
    grad_w, grad_b = obtenu
    attendu_w, attendu_b = _ref_gradient(X, y, w, b, alpha)
    if _proches(grad_w, attendu_w) and math.isclose(grad_b, attendu_b, rel_tol=1e-9, abs_tol=1e-12):
        return
    for variante, conseil in _PIEGES_GRADIENT:
        vw, vb = _ref_gradient(X, y, w, b, alpha, variante)
        if _proches(grad_w, vw) and math.isclose(grad_b, vb, rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(f"{quoi} : {conseil}")
    y_f, w_f = np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    X_f = np.asarray(X, dtype=float)

    def cout(v):
        return _ref_perte(y_f, X_f @ v[:-1] + v[-1]) + alpha / 2 * v[:-1] @ v[:-1]

    numerique = gradient_numerique(cout, np.append(w_f, b))
    raise AssertionError(f"{quoi} donne grad_w = {np.round(grad_w, 6)}, grad_b = {_nb(grad_b)} ; les différences finies "
                         f"(`gradient_numerique`) donnent grad_w ≈ {np.round(numerique[:-1], 6)}, grad_b ≈ "
                         f"{_nb(round(numerique[-1], 6))}. Rappel : grad_w = X.T @ (sigmoide(X @ w + b) - y) / n + alpha·w.")


def verifier_gradient_logistique(f) -> None:
    # 1. L'exemple de la docstring.
    X = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])
    y, w, b = np.array([1, 0, 1]), np.array([0.5, -0.25]), 0.1
    quoi = "gradient_logistique(X, y, w, b)  (l'exemple de la docstring)"
    obtenu = _exiger_gradient(_appel(lambda: f(X.copy(), y.copy(), w.copy(), b), quoi, _CONSEIL_FORMES), 2, quoi)
    _comparer_gradient(obtenu, X, y, w, b, 0.0, quoi)

    # 2. Des données au hasard, avec pénalité : plus de clients que de colonnes.
    rng = np.random.default_rng(142)
    for n, d, alpha in ((40, 3, 0.3), (7, 5, 2.5), (200, 1, 0.0)):
        X = rng.normal(0, 1.5, (n, d))
        y = rng.integers(0, 2, n)
        w, b = rng.normal(0, 1, d), float(rng.normal(0, 1))
        quoi = f"gradient_logistique(X, y, w, b, alpha={alpha})  ({n} clients, {d} colonnes)"
        obtenu = _exiger_gradient(_appel(lambda: f(X.copy(), y.copy(), w.copy(), b, alpha), quoi, _CONSEIL_FORMES), d,
                                  quoi)
        _comparer_gradient(obtenu, X, y, w, b, alpha, quoi)

    # 3. Des scores extrêmes : votre sigmoïde stable sert ici.
    X = np.array([[400.0, -2.0], [-500.0, 1.0], [3.0, 0.5]])
    y, w, b = np.array([0, 1, 1]), np.array([2.5, 0.0]), 0.0
    quoi = "gradient_logistique(X, y, w, b) avec des scores de ±1000"
    obtenu = _exiger_gradient(_appel(lambda: f(X, y, w, b), quoi, conseil_avertissement=(
        "Le score X @ w + b vaut ±1000 ici : calculez p avec VOTRE sigmoide (défi 140), qui ne déborde jamais.")), 2, quoi)
    _comparer_gradient(obtenu, X, y, w, b, 0.0, quoi)

    # 4. Un DataFrame, une Series d'un autre index, une liste : par position.
    clients = generer_clients(30, graine=142)
    Xdf = clients[NUMERIQUES] / 10
    ys = clients["churn"].reset_index(drop=True)
    wl, b = [0.2, -0.1, 0.3], -0.5
    quoi = "gradient_logistique(X, y, w, b)  (X : un DataFrame indexé par client ; y : une Series d'index 0 à 29 ; w : une liste)"
    obtenu = _exiger_gradient(_appel(lambda: f(Xdf, ys, wl, b, 0.1), quoi,
                                     "Convertissez d'abord : `np.asarray(X, dtype=float)`, idem pour y et w."), 3, quoi)
    attendu_w, attendu_b = _ref_gradient(Xdf, ys, wl, b, 0.1)
    if not (_proches(obtenu[0], attendu_w) and math.isclose(obtenu[1], attendu_b, rel_tol=1e-9)):
        raise AssertionError(f"{quoi} donne un résultat faux. `X @ w` sur un DataFrame renvoie une Series indexée par "
                             f"CLIENT, et `- y` l'aligne sur l'index 0 à 29 : aucune étiquette en commun, que des NaN. "
                             f"Convertissez tout en tableaux numpy d'abord (`np.asarray(..., dtype=float)`).")

    # 5. Ce qu'il faut refuser, et les entrées intactes.
    X, y, w = np.ones((4, 2)), np.array([0, 1, 1, 0]), np.zeros(2)
    _leve(lambda: f(X, y[:3], w, 0.0), ValueError, "gradient_logistique(X à 4 lignes, y à 3 valeurs, w, b)",
          "Chaque ligne de X va avec une réponse : comparez les longueurs (numpy le fait pour vous si vous ne cassez rien).")
    _leve(lambda: f(X, y, np.zeros(3), 0.0), ValueError, "gradient_logistique(X à 2 colonnes, y, w à 3 coefficients, b)",
          "Un coefficient par colonne : X @ w le vérifie tout seul.")
    X = rng.normal(0, 1, (20, 3))
    y, w = rng.integers(0, 2, 20), rng.normal(0, 1, 3)
    copies = X.copy(), y.copy(), w.copy()
    _appel(lambda: f(X, y, w, 0.3, 1.0), "gradient_logistique(X, y, w, b, alpha=1.0)")
    for apres, avant, nom in zip((X, y, w), copies, ("X", "y", "w")):
        _verifier_intact(apres, avant, f"gradient_logistique (sur {nom})")

    # 6. Vectorisé : d'abord compter les lignes exécutées (ne dépend pas de la machine), puis chronométrer.
    rng = np.random.default_rng(1421)
    pX, py, pw = rng.normal(0, 1, (10_000, 10)), rng.integers(0, 2, 10_000), rng.normal(0, 0.3, 10)
    boucle = _boucle_par_ligne(f, lambda n: f(pX[:n].copy(), py[:n].copy(), pw.copy(), 0.2, 0.01))
    if boucle:
        raise AssertionError(f"Les valeurs sont justes, mais votre code fait une boucle Python sur les clients : 1 000 "
                             f"clients exécutent {boucle[0]} lignes de votre fichier, 10 000 en exécutent {boucle[1]}. Sur "
                             f"300 000 clients et 10 colonnes, c'est 300 000 tours. Pas de boucle sur les clients : "
                             f"`X.T @ erreur` fait toutes les sommes d'un coup.")
    rng = np.random.default_rng(1420)
    X, y, w = rng.normal(0, 1, (300_000, 10)), rng.integers(0, 2, 300_000), rng.normal(0, 0.3, 10)
    obtenu, duree = _chronometrer(lambda: f(X, y, w, 0.2, 0.01), seuil=0.15)
    if duree >= 0.15:
        raise AssertionError(f"Les valeurs sont justes, mais 300 000 clients et 10 colonnes prennent {duree:.2f} s (numpy : "
                             f"environ 0,03 s). Pas de boucle sur les clients : `X.T @ erreur` fait toutes les sommes d'un "
                             f"coup.")
    _comparer_gradient(obtenu, X, y, w, 0.2, 0.01, "gradient_logistique sur 300 000 clients")


# ============================================================================ Défi 143 : RegressionLogistiqueGD


def _ref_descente(X, y01, alpha=0.01, pas=0.5, n_iter_max=10_000, tol=1e-6, variante=None):
    with warnings.catch_warnings(), np.errstate(all="ignore"):   # la « montée » diverge : inf, puis NaN
        warnings.simplefilter("ignore")
        return _descendre_pas_a_pas(X, y01, alpha, pas, n_iter_max, tol, variante)


def _descendre_pas_a_pas(X, y01, alpha, pas, n_iter_max, tol, variante):
    X = np.asarray(X, dtype=float)
    w, b, n_iter = np.zeros(X.shape[1]), 0.0, n_iter_max
    for iteration in range(n_iter_max):
        grad_w, grad_b = _ref_gradient(X, y01, w, b, alpha, "b_penalise" if variante == "b_penalise" else None)
        if variante == "norme_w":
            norme = float(np.sqrt(grad_w @ grad_w))
        elif variante == "norme_max":
            norme = max(float(np.abs(grad_w).max()), abs(grad_b))
        elif variante == "norme_carre":
            norme = float(grad_w @ grad_w + grad_b * grad_b)
        else:
            norme = float(np.sqrt(grad_w @ grad_w + grad_b * grad_b))
        if variante not in ("sans_arret", "arret_apres") and norme < tol:
            n_iter = iteration
            break
        if variante == "montee":
            w, b = w + pas * grad_w, b + pas * grad_b
        else:
            w = w - pas * grad_w
            b = b if variante == "b_fige" else b - pas * grad_b
        if variante == "arret_apres" and norme < tol:
            n_iter = iteration + 1
            break
    return w, b, n_iter


_PIEGES_DESCENTE = [
    ("montee", "les coefficients vont dans le mauvais sens : on DESCEND la pente, `w = w - pas * grad_w` (le gradient "
               "pointe vers la montée)."),
    ("b_fige", "l'ordonnée à l'origine ne bouge pas : à chaque pas, `b = b - pas * grad_b` aussi."),
    ("b_penalise", "l'ordonnée à l'origine est pénalisée : la pénalité ne porte que sur w (votre gradient_logistique, "
                   "défi 142, le fait déjà)."),
    ("sans_arret", "vous ne vous arrêtez jamais avant n_iter_max : quand la norme du gradient passe sous tol, on a trouvé le "
                   "minimum, `break` (et n_iter_ = le nombre de pas faits)."),
    ("arret_apres", "vous faites un pas de trop : on teste la norme du gradient AVANT de bouger (si elle est sous tol, on "
                    "s'arrête sans faire ce pas)."),
    ("norme_w", "la norme ne regarde que grad_w : c'est la norme de TOUT le gradient, w ET b : "
                "`np.sqrt(grad_w @ grad_w + grad_b ** 2)`."),
    ("norme_max", "vous comparez la plus grande coordonnée du gradient à tol : c'est la norme EUCLIDIENNE de tout le "
                  "gradient, `np.sqrt(grad_w @ grad_w + grad_b ** 2)`."),
    ("norme_carre", "vous comparez le CARRÉ de la norme à tol : prenez la racine, `np.sqrt(grad_w @ grad_w + grad_b ** 2)`."),
]


def _lire_modele(modele, d: int, quoi: str):
    for nom in ("coef_", "intercept_", "n_iter_", "classes_"):
        if not hasattr(modele, nom):
            raise AssertionError(f"Après {quoi}, l'attribut `{nom}` n'existe pas : `fit` range ce qu'il a appris dans "
                                 f"`coef_` (les d coefficients), `intercept_` (b), `n_iter_` (le nombre de pas) et "
                                 f"`classes_`.")
    coef = np.ravel(np.asarray(modele.coef_, dtype=float))
    intercept = np.ravel(np.asarray(modele.intercept_, dtype=float))
    if coef.shape != (d,) or intercept.shape != (1,):
        raise AssertionError(f"Après {quoi}, coef_ a la forme {np.shape(modele.coef_)} et intercept_ "
                             f"{np.shape(modele.intercept_)} : un coefficient par colonne ({d}) et un seul nombre pour b.")
    n_iter = modele.n_iter_
    if isinstance(n_iter, bool) or not isinstance(n_iter, numbers.Integral):
        raise AssertionError(f"Après {quoi}, n_iter_ vaut {_court(n_iter)} : le nombre de pas effectués (un entier).")
    return coef, float(intercept[0]), int(n_iter)


def _comparer_descente(modele, X, y01, parametres: dict, quoi: str, diagnostiquer: bool = True) -> None:
    coef, intercept, n_iter = _lire_modele(modele, X.shape[1], quoi)
    w, b, attendu_iter = _ref_descente(X, y01, **parametres)
    if _proches(coef, w, rtol=1e-7, atol=1e-10) and math.isclose(intercept, b, rel_tol=1e-7, abs_tol=1e-10) \
            and n_iter == attendu_iter:
        return
    if diagnostiquer:
        for variante, conseil in _PIEGES_DESCENTE:
            vw, vb, viter = _ref_descente(X, y01, variante=variante, **parametres)
            if _proches(coef, vw, rtol=1e-7, atol=1e-10) and math.isclose(intercept, vb, rel_tol=1e-7, abs_tol=1e-10) \
                    and n_iter == viter:
                raise AssertionError(f"{quoi} : {conseil}")
    if n_iter != attendu_iter:
        raise AssertionError(f"{quoi} : n_iter_ vaut {n_iter} au lieu de {attendu_iter}. À chaque tour : calculer le "
                             f"gradient, s'arrêter si sa norme (w ET b) est < tol, sinon faire UN pas ; n_iter_ = le nombre "
                             f"de pas faits (n_iter_max si on ne s'est jamais arrêté).")
    raise AssertionError(f"{quoi} : coef_ = {np.round(coef, 6)}, intercept_ = {_nb(intercept)} au lieu de {np.round(w, 6)} "
                         f"et {_nb(b)}. Partez de w = 0 et b = 0, puis à chaque pas : w = w - pas·grad_w, b = b - pas·grad_b.")


def _exemple_standardise():
    clients = clients_exemple()
    return _standardiser(clients[NUMERIQUES]), clients["churn"].to_numpy()


def verifier_RegressionLogistiqueGD(Classe) -> None:
    _verifier_classifieur(Classe, "RegressionLogistiqueGD")
    defauts = {"alpha": 0.01, "pas": 0.5, "n_iter_max": 10_000, "tol": 1e-6}
    _verifier_parametres(_appel(lambda: Classe(), "RegressionLogistiqueGD()"), defauts, "RegressionLogistiqueGD()")
    _construit_sans_verifier(Classe, "RegressionLogistiqueGD(pas=-1, n_iter_max=0)", pas=-1, n_iter_max=0)

    # 1. Un seul pas : on part de w = 0, b = 0, et on descend.
    X, y = _exemple_standardise()
    modele = Classe(n_iter_max=1)
    quoi = "RegressionLogistiqueGD(n_iter_max=1).fit(X, y)  (clients_exemple standardisés)"
    retour = _appel(lambda: modele.fit(X.copy(), y.copy()), quoi)
    _verifier_fit_renvoie_self(modele, retour, quoi)
    coef, intercept, n_iter = _lire_modele(modele, 3, quoi)
    w1 = 0.5 * X.T @ (y - 0.5) / len(y)
    if not _proches(coef, w1):
        if _proches(coef, -w1):
            complet = _appel(lambda: Classe().fit(X, y), "RegressionLogistiqueGD().fit(X, y)")
            if _proches(_lire_modele(complet, 3, "fit")[0], -_ref_descente(X, y)[0], rtol=1e-6, atol=1e-8):
                raise AssertionError(f"{quoi} : tous vos coefficients ont le signe opposé à ceux attendus. La classe "
                                     f"positive (y01 = 1) est classes_[1], la DEUXIÈME de `np.unique(y)` : "
                                     f"`y01 = (y == self.classes_[1])`.")
            raise AssertionError(f"{quoi} : après un pas, coef_ = {np.round(coef, 4)} au lieu de {np.round(w1, 4)}. Vous "
                                 f"MONTEZ la pente : w = w - pas·grad_w (en partant de w = 0, b = 0).")
        if _proches(coef, np.zeros(3)):
            raise AssertionError(f"{quoi} : après un pas, coef_ vaut encore 0 : partez de w = 0 et faites un pas "
                                 f"w = w - pas·grad_w (un appel de votre gradient_logistique).")
        raise AssertionError(f"{quoi} : après un pas depuis w = 0, b = 0, coef_ = {np.round(coef, 4)} au lieu de "
                             f"{np.round(w1, 4)} (= pas × X.T @ (y - 0,5) / n, car sigmoide(0) = 0,5).")
    if n_iter != 1:
        raise AssertionError(f"{quoi} : n_iter_ vaut {n_iter} au lieu de 1 (le nombre de pas effectués).")

    # 2. Jusqu'au bout, sur l'exemple puis sur des clients déséquilibrés (1 sur 3 part : b doit bouger).
    modele = _appel(lambda: Classe().fit(X, y), "RegressionLogistiqueGD().fit(X, y)")
    _verifier_intact(X, _exemple_standardise()[0], "RegressionLogistiqueGD().fit(X, y)")
    _comparer_descente(modele, X, y, {}, "RegressionLogistiqueGD().fit(X, y)  (clients_exemple standardisés)")
    clients = generer_clients(300, graine=143)
    Xg, yg = _standardiser(clients[NUMERIQUES]), clients["churn"].to_numpy()
    for parametres in ({"alpha": 0.1, "tol": 1e-3}, {"alpha": 0.0, "pas": 1.0, "tol": 1e-2}, {"n_iter_max": 7},
                       {"alpha": 0.05, "tol": 0.05}):
        texte = ", ".join(f"{k}={v}" for k, v in parametres.items())
        quoi = f"RegressionLogistiqueGD({texte}).fit(X, y)  (300 clients standardisés, un tiers de partants)"
        _comparer_descente(_appel(lambda: Classe(**parametres).fit(Xg, yg), quoi), Xg, yg, parametres, quoi)

    # 3. Les mêmes coefficients que scikit-learn.
    clients = generer_clients(600, graine=1430)
    Xs, ys = _standardiser(clients[NUMERIQUES]), clients["churn"].to_numpy()
    alpha = 0.02
    quoi = f"RegressionLogistiqueGD(alpha={alpha}, tol=1e-9, n_iter_max=100_000).fit(X, y)  (600 clients)"
    modele = _appel(lambda: Classe(alpha=alpha, tol=1e-9, n_iter_max=100_000).fit(Xs, ys), quoi)
    coef, intercept, _ = _lire_modele(modele, 3, quoi)
    reference = LogisticRegression(C=1 / (len(ys) * alpha), tol=1e-12, max_iter=10_000).fit(Xs, ys)
    if not (_proches(coef, reference.coef_[0], rtol=0, atol=1e-5)
            and math.isclose(intercept, reference.intercept_[0], abs_tol=1e-5)):
        raise AssertionError(f"{quoi} : coef_ = {np.round(coef, 5)}, intercept_ = {_nb(intercept)}, alors que scikit-learn "
                             f"(`LogisticRegression(C=1 / (n·alpha))`, la même perte écrite autrement) trouve "
                             f"{np.round(reference.coef_[0], 5)} et {_nb(reference.intercept_[0])}.")

    # 4. Les étiquettes : classes_[1] est la classe positive, predict renvoie des étiquettes.
    etiquettes = np.where(ys == 1, "oui", "non")
    quoi = "RegressionLogistiqueGD().fit(X, y)  (y vaut \"non\" ou \"oui\")"
    modele = _appel(lambda: Classe().fit(Xs, etiquettes), quoi,
                    "Les étiquettes ne sont pas forcément 0 et 1 : `self.classes_ = np.unique(y)`, puis "
                    "`y01 = (y == self.classes_[1])` (en float) pour le calcul.")
    if list(getattr(modele, "classes_", [])) != ["non", "oui"]:
        raise AssertionError(f"Après {quoi}, classes_ vaut {_court(getattr(modele, 'classes_', None))} au lieu de "
                             f"['non', 'oui'] (`np.unique(y)`, dans l'ordre).")
    _comparer_descente(modele, Xs, ys, {}, quoi, diagnostiquer=False)
    proba = np.asarray(_appel(lambda: modele.predict_proba(Xs[:50]), "modele.predict_proba(X)"), dtype=float)
    w, b, _ = _ref_descente(Xs, ys)
    attendu = _ref_sigmoide(Xs[:50] @ w + b)
    if proba.shape != (50, 2):
        raise AssertionError(f"predict_proba(X) de 50 clients renvoie la forme {proba.shape} au lieu de (50, 2) : une "
                             f"colonne par classe, dans l'ordre de classes_.")
    if not _proches(proba[:, 1], attendu, rtol=1e-6):
        if _proches(proba[:, 0], attendu, rtol=1e-6):
            raise AssertionError("predict_proba : les colonnes sont inversées. La colonne 0 est la probabilité de "
                                 "classes_[0] (1 - p), la colonne 1 celle de classes_[1] (p).")
        raise AssertionError("predict_proba(X)[:, 1] doit valoir sigmoide(X @ coef_ + intercept_).")
    prediction = np.asarray(_appel(lambda: modele.predict(Xs[:50]), "modele.predict(X)"))
    nets = np.abs(attendu - 0.5) > 1e-9
    if prediction.shape != (50,) or list(prediction[nets]) != list(np.where(attendu > 0.5, "oui", "non")[nets]):
        raise AssertionError(f"predict(X) renvoie {_court(prediction[:6].tolist())}… : des ÉTIQUETTES (\"oui\" si p > 0,5, "
                             f"sinon \"non\"), `self.classes_[(p > 0.5).astype(int)]`.")
    inverse = np.where(ys == 1, "reste", "part")    # classes_ = ['part', 'reste'] : 'reste' est positive
    quoi = "RegressionLogistiqueGD().fit(X, y)  (y vaut \"part\" ou \"reste\")"
    modele = _appel(lambda: Classe().fit(Xs, list(inverse)), quoi)
    _comparer_descente(modele, Xs, ys, {}, quoi, diagnostiquer=False)
    if list(getattr(modele, "classes_", [])) != ["part", "reste"]:
        raise AssertionError(f"Après {quoi}, classes_ doit valoir ['part', 'reste'] : la classe positive est classes_[1] "
                             f"(ici « reste », l'ordre alphabétique), quelle que soit l'idée qu'on s'en fait.")

    # 5. Chaque fit repart de zéro ; X intact ; avant fit ; clone et validation croisée.
    modele = Classe(alpha=0.05, tol=1e-4)
    _appel(lambda: modele.fit(X, y), "modele.fit(X_exemple, y_exemple)")
    copie_X = Xg.copy()
    quoi = "modele.fit(X_exemple, y_exemple) puis modele.fit(X, y)  (un deuxième fit sur d'autres clients)"
    _appel(lambda: modele.fit(Xg, yg), quoi)
    _verifier_intact(Xg, copie_X, "fit(X, y)")
    coef, intercept, n_iter = _lire_modele(modele, 3, quoi)
    w, b, it = _ref_descente(Xg, yg, alpha=0.05, tol=1e-4)
    if not (_proches(coef, w, rtol=1e-7, atol=1e-10) and n_iter == it):
        raise AssertionError(f"{quoi} ne donne pas le même modèle qu'un fit neuf : chaque `fit` repart de w = 0 et b = 0 "
                             f"(sinon le résultat dépend de l'historique de l'objet, et `clone` + `fit` dans la validation "
                             f"croisée ne donneraient pas la même chose).")
    _verifier_pas_entraine(lambda: Classe().predict_proba(X), "RegressionLogistiqueGD().predict_proba(X)")
    for parametres, pourquoi in (({"pas": 0}, "pas = 0 : on ne bougerait jamais"),
                                 ({"pas": -0.5}, "un pas négatif fait MONTER la perte"),
                                 ({"alpha": -1}, "une pénalité négative récompense les grands coefficients"),
                                 ({"n_iter_max": 0}, "il faut au moins un tour"),
                                 ({"n_iter_max": 2.5}, "un nombre de tours est entier"),
                                 ({"n_iter_max": True}, "True n'est pas un nombre de tours"),
                                 ({"tol": -1e-6}, "une norme n'est jamais négative")):
        texte = ", ".join(f"{k}={v!r}" for k, v in parametres.items())
        _leve(lambda: Classe(**parametres).fit(X, y), (ValueError, TypeError), f"RegressionLogistiqueGD({texte}).fit(X, y)",
              f"Vérifiez les paramètres au début de `fit` ({pourquoi}).")
    _leve(lambda: Classe().set_params(pas=-1).fit(X, y), ValueError, "RegressionLogistiqueGD().set_params(pas=-1).fit(X, y)",
          "C'est dans `fit` que les paramètres se vérifient.")
    _leve(lambda: Classe().fit(X, np.zeros(len(y), dtype=int)), ValueError, "RegressionLogistiqueGD().fit(X, <que des 0>)",
          "Il faut exactement deux classes dans y.")
    _leve(lambda: Classe().fit(X, np.arange(len(y)) % 3), ValueError, "RegressionLogistiqueGD().fit(X, <trois classes>)",
          "Il faut exactement deux classes dans y.")
    _verifier_clone(Classe(alpha=0.2), "RegressionLogistiqueGD(alpha=0.2)")
    clients = generer_clients(450, graine=1431)
    Xc, yc = clients[NUMERIQUES], clients["churn"]
    quoi = "cross_val_score(make_pipeline(StandardScaler(), RegressionLogistiqueGD()), X, y, cv=3, scoring='roc_auc')"
    obtenu = _appel(lambda: cross_val_score(make_pipeline(StandardScaler(), Classe(tol=1e-4)), Xc, yc, cv=3,
                                            scoring="roc_auc", error_score="raise"), quoi)
    attendu = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=1 / (300 * 0.01))), Xc, yc, cv=3,
                              scoring="roc_auc")
    if not _proches(obtenu, attendu, rtol=0, atol=2e-3):
        raise AssertionError(f"{quoi} donne {np.round(obtenu, 4)} au lieu d'environ {np.round(attendu, 4)}.")


# ============================================================================ Défi 144 : meilleure_coupure


def _gini(q):
    return 2.0 * q * (1.0 - q)


def _entropie(q):
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return np.nan_to_num(h)


def _coupures(x, y, min_feuille=1, variante=None):
    """Toutes les coupures possibles : (seuils, gains), dans l'ordre croissant des seuils."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ordre = np.arange(x.size) if variante == "non_trie" else np.argsort(x, kind="stable")
    xs, ys = x[ordre], y[ordre]
    n = xs.size
    if n < 2:
        return np.array([]), np.array([])
    n_g = np.arange(1, n)
    n_d = n - n_g
    partis_g = np.cumsum(ys)[:-1]
    p_g, p_d, p = partis_g / n_g, (ys.sum() - partis_g) / n_d, ys.mean()
    impurete = _entropie if variante == "entropie" else _gini
    if variante == "non_pondere":
        gains = impurete(p) - (impurete(p_g) + impurete(p_d)) / 2
    elif variante == "impurete":
        gains = (n_g / n) * impurete(p_g) + (n_d / n) * impurete(p_d)
    else:
        gains = impurete(p) - (n_g / n) * impurete(p_g) - (n_d / n) * impurete(p_d)
    distinct = np.ones(n - 1, dtype=bool) if variante == "entre_egales" else xs[1:] != xs[:-1]
    if variante == "sans_min":
        taille = np.ones(n - 1, dtype=bool)
    elif variante == "min_gauche":
        taille = n_g >= min_feuille
    else:
        taille = (n_g >= min_feuille) & (n_d >= min_feuille)
    possible = distinct & taille
    if variante == "valeur_gauche":
        seuils = xs[:-1]
    elif variante == "valeur_droite":
        seuils = xs[1:]
    else:
        seuils = (xs[:-1] + xs[1:]) / 2.0
    return seuils[possible], gains[possible]


def _ref_coupure(x, y, min_feuille=1, variante=None):
    seuils, gains = _coupures(x, y, min_feuille, variante)
    if seuils.size == 0:
        return None, 0.0
    if variante == "impurete":
        k = int(np.flatnonzero(gains <= gains.min() + 1e-12)[0])
    elif variante == "plus_grand":
        k = int(np.flatnonzero(gains >= gains.max() - 1e-12)[-1])
    else:
        k = int(np.flatnonzero(gains >= gains.max() - 1e-12)[0])
    return float(seuils[k]), float(gains[k])


_PIEGES_COUPURE = [
    ("non_trie", "x n'est pas trié ! Triez-le en gardant chaque y EN FACE de son x : `ordre = np.argsort(x)`, puis "
                 "`x[ordre]` et `y[ordre]`. Sans tri, « couper après les k premiers » ne veut rien dire."),
    ("entre_egales", "vous coupez ENTRE deux valeurs égales. Avec la règle x <= seuil, deux clients qui ont la même valeur "
                     "vont forcément du même côté : votre gain est celui d'une coupure qui n'existe pas. Une coupure "
                     "n'est possible qu'entre deux valeurs voisines DIFFÉRENTES (`xs[1:] > xs[:-1]`)."),
    ("valeur_gauche", "le seuil est la valeur de gauche : il est au MILIEU des deux valeurs voisines, (xs[k] + xs[k+1]) / 2, "
                      "comme dans scikit-learn (un nouveau client entre les deux tombe du côté le plus proche)."),
    ("valeur_droite", "le seuil est la valeur de droite : avec x <= seuil, ce client passerait à gauche. Le seuil est au "
                      "MILIEU des deux valeurs voisines, (xs[k] + xs[k+1]) / 2."),
    ("non_pondere", "les deux enfants comptent autant, quelle que soit leur taille : pondérez chaque impureté par la PART "
                    "des clients qui y vont, gini(parent) - n_g/n·gini(gauche) - n_d/n·gini(droite). Sinon, isoler un "
                    "seul client « pur » paraît toujours excellent."),
    ("entropie", "c'est l'ENTROPIE (log2) : la consigne demande l'impureté de GINI, 2·p·(1 - p) (celle de scikit-learn par "
                 "défaut)."),
    ("impurete", "vous renvoyez l'impureté des enfants, pas le GAIN : gain = gini(parent) - (impureté pondérée des "
                 "enfants). Le meilleur seuil est le même, mais le nombre renvoyé doit être la baisse d'impureté."),
    ("plus_grand", "à égalité de gain, gardez le PLUS PETIT seuil (le premier rencontré dans l'ordre croissant), comme "
                   "scikit-learn : `>` et pas `>=` quand vous comparez au meilleur gain."),
    ("sans_min", "min_feuille est ignoré : chaque côté doit garder AU MOINS min_feuille clients."),
    ("min_gauche", "min_feuille n'est vérifié qu'à GAUCHE : chaque côté doit garder au moins min_feuille clients."),
]


def _exiger_coupure(resultat, quoi: str):
    if resultat is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return seuil, gain` ? (Sans coupure possible, "
                             f"c'est `return None, 0.0`.)")
    if not isinstance(resultat, tuple) or len(resultat) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(resultat)} au lieu d'un couple (seuil, gain).")
    seuil, gain = resultat
    gain = _exiger_nombre(gain, f"{quoi} : le gain")
    if seuil is not None:
        seuil = _exiger_nombre(seuil, f"{quoi} : le seuil")
    return seuil, gain


def _diagnostiquer_coupure(obtenu, x, y, min_feuille, quoi: str, attendu) -> None:
    for variante, conseil in _PIEGES_COUPURE:
        v_seuil, v_gain = _ref_coupure(x, y, min_feuille, variante)
        if v_seuil is not None and obtenu[0] is not None and math.isclose(obtenu[0], v_seuil, abs_tol=1e-9) \
                and math.isclose(obtenu[1], v_gain, abs_tol=1e-9) and (v_seuil, v_gain) != attendu:
            attendu_texte = "(None, 0.0)" if attendu[0] is None else f"({_nb(attendu[0])}, {_nb(attendu[1])})"
            raise AssertionError(f"{quoi} renvoie ({_nb(obtenu[0])}, {_nb(obtenu[1])}) au lieu de {attendu_texte} : "
                                 f"{conseil}")


_CONSEIL_LECTURE_SEULE = ("Si l'erreur parle de « read-only » : vous modifiez x SUR PLACE (`x.sort()`), et pandas l'interdit "
                          "sur les données d'une Series. Ne touchez jamais au tableau reçu : `np.argsort(x)` donne l'ordre "
                          "sans rien modifier.")


def _verifier_coupure(f, x, y, min_feuille, quoi: str, exact: bool = False) -> None:
    # Les attendus sont calculés AVANT l'appel, sur des copies : si la fonction trie x sur place, on le dit.
    x_avant, y_avant = np.array(x, dtype=float), np.array(y, dtype=float)
    obtenu = _exiger_coupure(_appel(lambda: f(x, y, min_feuille) if min_feuille != 1 else f(x, y), quoi,
                                    _CONSEIL_LECTURE_SEULE), quoi)
    _verifier_intact(np.asarray(x, dtype=float), x_avant, quoi)
    _verifier_intact(np.asarray(y, dtype=float), y_avant, quoi)
    x, y = x_avant, y_avant
    attendu = _ref_coupure(x, y, min_feuille)
    seuils, gains = _coupures(x, y, min_feuille)
    if attendu[0] is None:
        if obtenu != (None, 0.0):
            raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de (None, 0.0) : aucune coupure n'est possible "
                                 f"ici (toutes les valeurs sont égales, ou un des côtés aurait moins de min_feuille "
                                 f"clients). Renvoyez exactement `None, 0.0`.")
        return
    if obtenu[0] is None:
        raise AssertionError(f"{quoi} renvoie (None, …) alors qu'une coupure est possible : ({_nb(attendu[0])}, "
                             f"{_nb(attendu[1])}).")
    ok = bool(np.any(np.isclose(seuils, obtenu[0], rtol=0, atol=1e-9) & (gains >= gains.max() - 1e-9)))
    if exact:
        ok = math.isclose(obtenu[0], attendu[0], abs_tol=1e-9)
    if ok:
        gain_du_seuil = float(gains[np.isclose(seuils, obtenu[0], rtol=0, atol=1e-9)][0])
        if math.isclose(obtenu[1], gain_du_seuil, abs_tol=1e-9):
            return
    _diagnostiquer_coupure(obtenu, x, y, min_feuille, quoi, attendu)
    raise AssertionError(f"{quoi} renvoie ({_nb(obtenu[0])}, {_nb(obtenu[1])}) au lieu de ({_nb(attendu[0])}, "
                         f"{_nb(attendu[1])}). Triez x, essayez chaque coupure entre deux valeurs voisines différentes, "
                         f"gain = gini(parent) - n_g/n·gini(gauche) - n_d/n·gini(droite), gardez le plus grand.")


def verifier_meilleure_coupure(f) -> None:
    # 1. Les exemples de la docstring.
    _verifier_coupure(f, [1, 1, 2, 3], [0, 1, 1, 1], 1, "meilleure_coupure([1, 1, 2, 3], [0, 1, 1, 1])", exact=True)
    _verifier_coupure(f, [1, 2, 3, 4], [0, 1, 1, 0], 1,
                      "meilleure_coupure([1, 2, 3, 4], [0, 1, 1, 0])  (deux coupures à égalité)", exact=True)
    clients = clients_exemple()
    for colonne in NUMERIQUES:
        _verifier_coupure(f, clients[colonne].to_numpy(), clients["churn"].to_numpy(), 1,
                          f"meilleure_coupure(clients['{colonne}'], clients['churn'])  (clients_exemple)", exact=True)

    # 2. x dans le désordre, et des Series aux index quelconques.
    _verifier_coupure(f, [3, 1, 2, 1], [1, 0, 1, 1], 1, "meilleure_coupure([3, 1, 2, 1], [1, 0, 1, 1])  (x n'est pas trié)",
                      exact=True)
    x = pd.Series([3.0, 1.0, 2.0, 1.0, 5.0, 4.0], index=[0, 1, 2, 3, 5, 4])
    y = pd.Series([0, 1, 1, 0, 1, 1], index=[10, 11, 12, 13, 14, 15])
    quoi = "meilleure_coupure(x, y)  (deux Series : x a l'index [0, 1, 2, 3, 5, 4], y l'index 10 à 15)"
    obtenu = _exiger_coupure(_appel(lambda: f(x, y), quoi, "Convertissez d'abord : `np.asarray(x, dtype=float)`."), quoi)
    attendu = _ref_coupure(x.to_numpy(), y.to_numpy())
    if obtenu[0] is None or not (math.isclose(obtenu[0], attendu[0]) and math.isclose(obtenu[1], attendu[1])):
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {attendu} : le i-ème x va avec le i-ème y, "
                             f"quel que soit l'index. Sur une Series, `x[ordre]` lit les ÉTIQUETTES d'index, pas les "
                             f"positions : convertissez en tableaux numpy (`np.asarray`) avant de trier.")

    # 3. Pas de coupure possible.
    for xx, yy, m, cas in (([5, 5, 5], [0, 1, 0], 1, "toutes les valeurs sont égales"),
                           ([7], [1], 1, "un seul client"),
                           ([1, 2, 3], [0, 1, 0], 2, "3 clients ne font pas deux côtés de 2")):
        quoi = f"meilleure_coupure({xx}, {yy}{', min_feuille=' + str(m) if m != 1 else ''})  ({cas})"
        obtenu = _appel(lambda: f(xx, yy, m) if m != 1 else f(xx, yy), quoi,
                        "Quand aucune coupure n'est possible, renvoyez `None, 0.0` (ce n'est pas une erreur : l'arbre "
                        "en fera une feuille).")
        if not (isinstance(obtenu, tuple) and len(obtenu) == 2 and obtenu[0] is None and obtenu[1] == 0.0):
            if isinstance(obtenu, tuple) and len(obtenu) == 2 and obtenu[0] is not None:
                try:
                    _diagnostiquer_coupure((float(obtenu[0]), float(obtenu[1])), xx, yy, m, quoi, (None, 0.0))
                except (TypeError, ValueError):
                    pass
            raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de (None, 0.0) : aucune coupure n'est "
                                 f"possible. Renvoyez exactement `None, 0.0` (pas NaN, pas d'exception : l'arbre en fera une "
                                 f"feuille).")

    # 4. min_feuille, des deux côtés.
    for xx, yy, m in (([1, 2, 3, 4, 5, 6], [1, 0, 0, 0, 0, 0], 2), ([1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 1], 2),
                      ([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 1, 0, 1, 1, 0], 3)):
        _verifier_coupure(f, xx, yy, m, f"meilleure_coupure({xx}, {yy}, min_feuille={m})", exact=True)

    # 5. Des centaines de cas tirés au hasard (valeurs répétées ou non), comparés à toutes les coupures possibles.
    rng = np.random.default_rng(144)
    for essai in range(40):
        n = int(rng.integers(2, 120))
        x = rng.integers(0, max(2, n // 3), n).astype(float) if essai % 2 else np.round(rng.normal(50, 20, n), 1)
        y = (rng.random(n) < 1 / (1 + np.exp(-(x - x.mean()) / (x.std() + 1)))).astype(int)
        m = int(rng.choice([1, 1, 2, 5]))
        _verifier_coupure(f, x, y, m, f"meilleure_coupure(x, y, min_feuille={m})  ({n} clients tirés au hasard, "
                                      f"x = {_court(list(x[:5]))[:-1]}, …])")

    # 6. Le même seuil que scikit-learn.
    clients = generer_clients(800, graine=1440)
    for colonne in NUMERIQUES:
        for m in (1, 40):
            x, y = clients[colonne].to_numpy(), clients["churn"].to_numpy()
            quoi = f"meilleure_coupure(clients['{colonne}'], clients['churn'], min_feuille={m})  (800 clients)"
            _verifier_coupure(f, x, y, m, quoi)
            souche = DecisionTreeClassifier(max_depth=1, min_samples_leaf=m).fit(x.reshape(-1, 1), y)
            obtenu = f(x, y, m)
            if not math.isclose(obtenu[0], souche.tree_.threshold[0], abs_tol=1e-9):
                raise AssertionError(f"{quoi} donne le seuil {_nb(obtenu[0])}, scikit-learn "
                                     f"(`DecisionTreeClassifier(max_depth=1)`) trouve {_nb(souche.tree_.threshold[0])}.")

    # 7. Ce qu'il faut refuser, et les entrées intactes.
    for xx, yy, m, pourquoi in (([1, 2, 3], [0, 1], 1, "x et y de longueurs différentes"),
                                ([], [], 1, "aucun client"),
                                ([1.0, np.nan, 3.0], [0, 1, 1], 1, "x contient NaN : où le mettre, à gauche ou à droite ?"),
                                ([1, 2, 3], [0, 2, 1], 1, "y vaut 0 ou 1"),
                                ([1, 2, 3], [0, 1, 1], 0, "min_feuille = 0 : une feuille vide n'a pas de sens"),
                                ([1, 2, 3], [0, 1, 1], True, "min_feuille = True n'est pas un nombre de clients"),
                                ([1, 2, 3], [0, 1, 1], 1.5, "min_feuille est un nombre ENTIER de clients")):
        _leve(lambda: f(xx, yy, m), ValueError, f"meilleure_coupure({xx}, {yy}, min_feuille={m!r})",
              f"Refusez avec ValueError : {pourquoi}.")
    x = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    y = np.array([1, 0, 1, 0, 1])
    _appel(lambda: f(x, y), "meilleure_coupure(x, y)")
    _verifier_intact(x, np.array([5.0, 1.0, 4.0, 2.0, 3.0]), "meilleure_coupure(x, y)")
    _verifier_intact(y, np.array([1, 0, 1, 0, 1]), "meilleure_coupure(x, y)")

    # 8. O(n log n) : un tri, puis des cumuls.
    rng = np.random.default_rng(1441)
    x, y = rng.normal(0, 1, 2_000), rng.integers(0, 2, 2_000)
    _, duree = _chronometrer(lambda: f(x, y), seuil=0.15)
    if duree >= 0.15:
        raise AssertionError(f"Les coupures sont justes, mais 2 000 clients prennent {duree:.2f} s (un tri et des cumuls : "
                             f"environ 0,001 s). Pour chaque coupure, vous recomptez tous les clients : c'est une boucle "
                             f"dans une boucle, O(n²) — 1 million de clients prendraient des heures. Triez UNE fois, puis "
                             f"`np.cumsum(y_trie)` donne le nombre de partants à gauche pour TOUTES les coupures d'un coup.")
    x, y = rng.normal(0, 1, 10_000), rng.integers(0, 2, 10_000)
    _, duree = _chronometrer(lambda: f(x, y), seuil=0.2)
    if duree >= 0.2:
        raise AssertionError(f"Les coupures sont justes, mais 10 000 clients prennent {duree:.2f} s (un tri et des cumuls : "
                             f"environ 0,002 s). Pour chaque seuil, vous refaites un calcul sur TOUS les clients (un masque "
                             f"`x <= seuil`, une moyenne…) : même vectorisé, c'est O(n²). Triez UNE fois, puis "
                             f"`np.cumsum(y_trie)` donne le nombre de partants à gauche pour toutes les coupures d'un coup.")
    x, y = np.round(rng.normal(60, 25, 200_000), 1), rng.integers(0, 2, 200_000)
    obtenu, duree = _chronometrer(lambda: f(x, y, 5), seuil=1.0)
    if duree >= 1.0:
        raise AssertionError(f"Les coupures sont justes, mais 200 000 clients prennent {duree:.2f} s (un tri et des cumuls : "
                             f"environ 0,02 s). Triez une fois, puis travaillez sur des tableaux entiers (`np.cumsum`).")
    _verifier_coupure(f, x, y, 5, "meilleure_coupure(x, y, min_feuille=5)  (200 000 clients)")


# ============================================================================ Défi 145 : ArbreDecision


def _chemin(chemin: tuple) -> str:
    return "à la racine" if not chemin else "au nœud racine → " + " → ".join(chemin)


def _regles(X, y01, profondeur: int, max_profondeur, min_feuille: int):
    """Ce que les règles imposent à ce nœud : ("feuille", raison) ou ("coupe", meilleur gain, coupures acceptables)."""
    if max_profondeur is not None and profondeur >= max_profondeur:
        return "feuille", "profondeur", None
    p = float(y01.mean())
    if p in (0.0, 1.0):
        return "feuille", "pur", None
    candidates = []
    for colonne in range(X.shape[1]):
        seuils, gains = _coupures(X[:, colonne], y01, min_feuille)
        candidates.extend((float(g), colonne, float(s)) for s, g in zip(seuils, gains))
    if not candidates:
        return "feuille", "aucune", None
    meilleur = max(g for g, _, _ in candidates)
    return "coupe", meilleur, [(c, s, g) for g, c, s in candidates if g >= meilleur - 1e-9]


def _exiger_noeud(noeud, ou: str) -> None:
    if not isinstance(noeud, dict):
        raise AssertionError(f"arbre_ : {ou}, on trouve {_court(noeud)} au lieu d'un dictionnaire. Une feuille est "
                             f"{{'proba': p, 'effectif': n}}, un nœud {{'colonne': j, 'seuil': s, 'gauche': …, 'droite': …}}.")
    cles = {"colonne", "seuil", "gauche", "droite"} if "colonne" in noeud else {"proba", "effectif"}
    if not cles <= set(noeud):
        raise AssertionError(f"arbre_ : {ou}, il manque les clés {sorted(cles - set(noeud))} (clés trouvées : "
                             f"{sorted(noeud)}). Une feuille est {{'proba': p, 'effectif': n}}, un nœud "
                             f"{{'colonne': j, 'seuil': s, 'gauche': …, 'droite': …}}.")


def _valider(noeud, X, y01, profondeur, max_profondeur, min_feuille, chemin=()) -> None:
    ou = f"{_chemin(chemin)} ({y01.size} clients, profondeur {profondeur})"
    _exiger_noeud(noeud, ou)
    regle = _regles(X, y01, profondeur, max_profondeur, min_feuille)
    if "colonne" not in noeud:  # une feuille
        if regle[0] == "coupe":
            meilleur, acceptables = regle[1], regle[2]
            colonne, seuil, _ = acceptables[0]
            if meilleur <= 1e-12:
                raise AssertionError(
                    f"arbre_ : {ou}, vous faites une feuille parce que le gain est nul. Il faut couper quand même (la "
                    f"colonne {colonne} au seuil {_nb(seuil)}) : sur un « ou exclusif », la première coupure ne gagne RIEN "
                    f"et la suivante gagne TOUT. Un nœud devient une feuille seulement s'il est pur, à la profondeur "
                    f"maximale, ou si aucune coupure n'est possible (seuil None pour toutes les colonnes). C'est aussi la "
                    f"règle de scikit-learn.")
            if max_profondeur is not None and profondeur == max_profondeur - 1:
                raise AssertionError(
                    f"arbre_ : {ou}, vous vous arrêtez une profondeur trop tôt. La racine est à la profondeur 0 ; avec "
                    f"max_profondeur={max_profondeur}, un nœud de profondeur {profondeur} peut encore être coupé (on "
                    f"s'arrête quand profondeur >= max_profondeur).")
            raise AssertionError(f"arbre_ : {ou}, vous faites une feuille alors qu'il fallait couper : colonne {colonne}, "
                                 f"seuil {_nb(seuil)} (gain {_nb(meilleur)}).")
        attendu = float(y01.mean())
        try:
            proba = float(noeud["proba"])
        except (TypeError, ValueError):
            raise AssertionError(f"arbre_ : {ou}, la proba de la feuille vaut {_court(noeud['proba'])} au lieu d'un "
                                 f"nombre.") from None
        if not math.isclose(proba, attendu, abs_tol=1e-12):
            if proba == round(attendu) and attendu not in (0.0, 1.0):
                conseil = ("c'est la classe MAJORITAIRE (0 ou 1) : la feuille garde la PROPORTION de partants, "
                           "`y.mean()`, pour que predict_proba donne une vraie probabilité.")
            elif math.isclose(proba, 1 - attendu, abs_tol=1e-12):
                conseil = "c'est la proportion de classes_[0] : la feuille garde celle de classes_[1] (la classe positive)."
            else:
                conseil = "c'est la proportion de classes_[1] parmi les clients qui arrivent dans CETTE feuille."
            raise AssertionError(f"arbre_ : {ou}, la feuille dit proba = {_nb(proba)} au lieu de {_nb(attendu)} : "
                                 f"{conseil}")
        if noeud["effectif"] != y01.size:
            raise AssertionError(f"arbre_ : {ou}, la feuille dit effectif = {_court(noeud['effectif'])} au lieu de "
                                 f"{y01.size} (le nombre de clients du train qui y arrivent).")
        return
    # un nœud qui coupe
    if regle[0] == "feuille":
        raison = {
            "profondeur": f"la profondeur maximale est atteinte (la racine est à la profondeur 0, et on s'arrête quand "
                          f"profondeur >= max_profondeur={max_profondeur})",
            "pur": "le nœud est PUR (tous ses clients sont de la même classe) : il n'y a plus rien à séparer",
            "aucune": f"aucune coupure n'est possible (chaque côté doit garder au moins min_feuille={min_feuille} "
                      f"clients, et on ne coupe pas entre deux valeurs égales)",
        }[regle[1]]
        raise AssertionError(f"arbre_ : {ou}, vous coupez encore, alors qu'il faut une feuille : {raison}.")
    try:
        colonne, seuil = int(noeud["colonne"]), float(noeud["seuil"])
    except (TypeError, ValueError):
        raise AssertionError(f"arbre_ : {ou}, colonne = {_court(noeud['colonne'])} et seuil = {_court(noeud['seuil'])} : "
                             f"un numéro de colonne (entier) et un nombre.") from None
    meilleur, acceptables = regle[1], regle[2]
    if not any(c == colonne and math.isclose(s, seuil, abs_tol=1e-9) for c, s, _ in acceptables):
        bonne_c, bonne_s, _ = acceptables[0]
        if not 0 <= colonne < X.shape[1]:
            raise AssertionError(f"arbre_ : {ou}, colonne {colonne} n'existe pas (X a {X.shape[1]} colonnes).")
        seuils, gains = _coupures(X[:, colonne], y01, min_feuille)
        ici = np.isclose(seuils, seuil, rtol=0, atol=1e-9)
        if not ici.any():
            raise AssertionError(f"arbre_ : {ou}, la coupure « colonne {colonne} <= {_nb(seuil)} » n'est pas une coupure "
                                 f"possible (seuil au milieu de deux valeurs voisines différentes, au moins min_feuille="
                                 f"{min_feuille} clients de chaque côté). La bonne : colonne {bonne_c} <= {_nb(bonne_s)}.")
        gain = float(gains[ici][0])
        meilleurs_par_colonne = [(_coupures(X[:, c], y01, min_feuille)[1].max(initial=-1.0), c) for c in range(X.shape[1])]
        if math.isclose(gain, gains.max(), abs_tol=1e-9):
            pire = min(g for g, _ in meilleurs_par_colonne if g >= 0)
            premiere = min(c for g, c in meilleurs_par_colonne if g >= 0)
            if math.isclose(gain, pire, abs_tol=1e-12) and colonne == premiere:
                conseil = (" C'est la PREMIÈRE colonne, et aussi celle au PLUS PETIT gain. Comparez les gains de TOUTES les "
                           "colonnes et gardez le plus GRAND.")
            elif math.isclose(gain, pire, abs_tol=1e-12):
                conseil = " Vous gardez le PLUS PETIT gain : c'est le plus GRAND qui gagne."
            else:
                conseil = " Comparez les gains de TOUTES les colonnes et gardez le plus GRAND."
            raise AssertionError(f"arbre_ : {ou}, vous coupez sur la colonne {colonne} (meilleur gain {_nb(gain)}), mais "
                                 f"la colonne {bonne_c} fait mieux : seuil {_nb(bonne_s)}, gain {_nb(meilleur)}.{conseil}")
        raise AssertionError(f"arbre_ : {ou}, la coupure « colonne {colonne} <= {_nb(seuil)} » a un gain de {_nb(gain)} ; la "
                             f"meilleure est « colonne {bonne_c} <= {_nb(bonne_s)} » (gain {_nb(meilleur)}). Chaque nœud "
                             f"cherche SA meilleure coupure, avec SES clients.")
    gauche = X[:, colonne] <= seuil
    _valider(noeud["gauche"], X[gauche], y01[gauche], profondeur + 1, max_profondeur, min_feuille, chemin + ("gauche",))
    _valider(noeud["droite"], X[~gauche], y01[~gauche], profondeur + 1, max_profondeur, min_feuille, chemin + ("droite",))


def _descendre(arbre: dict, ligne) -> float:
    noeud = arbre
    while "colonne" in noeud:
        noeud = noeud["gauche"] if ligne[int(noeud["colonne"])] <= noeud["seuil"] else noeud["droite"]
    return float(noeud["proba"])


def _seuils(noeud: dict):
    if "colonne" not in noeud:
        return []
    return [(int(noeud["colonne"]), float(noeud["seuil"]))] + _seuils(noeud["gauche"]) + _seuils(noeud["droite"])


def _mesures(noeud: dict):
    """(profondeur, nombre de feuilles)."""
    if "colonne" not in noeud:
        return 0, 1
    pg, fg = _mesures(noeud["gauche"])
    pd_, fd = _mesures(noeud["droite"])
    return 1 + max(pg, pd_), fg + fd


def _verifier_arbre(Classe, X, y, parametres: dict, quoi: str, X_test=None):
    """Entraîne, valide l'arbre nœud par nœud, puis predict_proba, predict, profondeur_ et n_feuilles_."""
    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y)
    modele = Classe(**parametres)
    copie = X_np.copy()
    retour = _appel(lambda: modele.fit(X, y), quoi)
    _verifier_fit_renvoie_self(modele, retour, quoi)
    _verifier_intact(np.asarray(X, dtype=float), copie, quoi)
    for nom in ("classes_", "arbre_", "profondeur_", "n_feuilles_"):
        if not hasattr(modele, nom):
            raise AssertionError(f"Après {quoi}, l'attribut `{nom}` n'existe pas : `fit` range classes_, arbre_ (le "
                                 f"dictionnaire racine), profondeur_ et n_feuilles_.")
    classes = np.unique(y_np)
    if list(modele.classes_) != list(classes):
        raise AssertionError(f"Après {quoi}, classes_ vaut {_court(list(modele.classes_))} au lieu de {list(classes)} "
                             f"(`np.unique(y)`).")
    y01 = (y_np == classes[1]).astype(float)
    _valider(modele.arbre_, X_np, y01, 0, parametres.get("max_profondeur", 3), parametres.get("min_feuille", 1))
    profondeur, feuilles = _mesures(modele.arbre_)
    if (modele.profondeur_, modele.n_feuilles_) != (profondeur, feuilles):
        raise AssertionError(f"Après {quoi}, profondeur_ = {_court(modele.profondeur_)} et n_feuilles_ = "
                             f"{_court(modele.n_feuilles_)}, mais votre arbre_ a une profondeur de {profondeur} (le nombre "
                             f"de coupures de la racine à la feuille la plus basse) et {feuilles} feuilles.")
    # predict_proba : sur le train, et sur des clients placés EXACTEMENT sur les seuils.
    lignes = [X_np]
    for colonne, seuil in _seuils(modele.arbre_)[:20]:
        sur_le_seuil = X_np[: min(len(X_np), 40)].copy()
        sur_le_seuil[:, colonne] = seuil
        lignes.append(sur_le_seuil)
    if X_test is not None:
        lignes.append(np.asarray(X_test, dtype=float))
    X_essai = np.vstack(lignes)
    proba = np.asarray(_appel(lambda: modele.predict_proba(X_essai), "modele.predict_proba(X)"), dtype=float)
    attendu = np.array([_descendre(modele.arbre_, ligne) for ligne in X_essai])
    if proba.shape != (len(X_essai), 2):
        raise AssertionError(f"predict_proba(X) de {len(X_essai)} clients renvoie la forme {proba.shape} au lieu de "
                             f"({len(X_essai)}, 2) : une colonne par classe, dans l'ordre de classes_.")
    if not _proches(proba[:, 1], attendu, rtol=0, atol=1e-12):
        if _proches(proba[:, 0], attendu, rtol=0, atol=1e-12):
            raise AssertionError("predict_proba : les colonnes sont inversées. Colonne 0 : 1 - p ; colonne 1 : p (la proba "
                                 "de la feuille, celle de classes_[1]).")
        rang = int(np.flatnonzero(~np.isclose(proba[:, 1], attendu, rtol=0, atol=1e-12))[0])
        if rang >= len(X_np):
            raise AssertionError("predict_proba se trompe pour un client placé EXACTEMENT sur un seuil : x == seuil va à "
                                 "GAUCHE (`x <= seuil`), comme pendant fit.")
        raise AssertionError("predict_proba(X)[:, 1] doit valoir la proba de la feuille où arrive chaque client (on descend "
                             "depuis la racine : x <= seuil → gauche, sinon droite).")
    prediction = np.asarray(_appel(lambda: modele.predict(X_essai), "modele.predict(X)"))
    if prediction.shape != (len(X_essai),) or list(prediction) != list(classes[(attendu > 0.5).astype(int)]):
        egalite = np.isclose(attendu, 0.5)
        if egalite.any() and list(prediction[~egalite]) == list(classes[(attendu[~egalite] > 0.5).astype(int)]):
            raise AssertionError(f"predict(X) : dans une feuille à 50 / 50, scikit-learn renvoie la PREMIÈRE classe "
                                 f"(classes_[0], ici {classes[0].item()!r}) : `p > 0.5`, pas `p >= 0.5`.")
        raise AssertionError(f"predict(X) renvoie {_court(prediction[:6].tolist())}… : des étiquettes, classes_[1] si la proba "
                             f"de la feuille dépasse 0,5, sinon classes_[0].")
    return modele


def verifier_ArbreDecision(Classe) -> None:
    _verifier_classifieur(Classe, "ArbreDecision")
    _verifier_parametres(_appel(lambda: Classe(), "ArbreDecision()"), {"max_profondeur": 3, "min_feuille": 1},
                         "ArbreDecision()")
    _construit_sans_verifier(Classe, "ArbreDecision(max_profondeur=0, min_feuille=-2)", max_profondeur=0, min_feuille=-2)

    # 1. L'exemple : la racine coupe sur `appels` (la DERNIÈRE colonne).
    clients = clients_exemple()
    X, y = clients[NUMERIQUES], clients["churn"]
    _verifier_arbre(Classe, X, y, {}, "ArbreDecision().fit(clients_exemple()[NUMERIQUES], clients_exemple()['churn'])")

    # 2. Le « ou exclusif » : la première coupure ne gagne rien, la deuxième gagne tout.
    X_xor = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
    y_xor = np.array(["non", "oui", "oui", "non"])
    modele = _verifier_arbre(Classe, X_xor, y_xor, {"max_profondeur": None},
                             "ArbreDecision(max_profondeur=None).fit(X, y)  (un « ou exclusif » de 4 clients)")
    if list(_appel(lambda: modele.predict(X_xor), "modele.predict(X)")) != list(y_xor):
        raise AssertionError("Sur le « ou exclusif », l'arbre sans limite de profondeur doit retrouver les 4 réponses.")

    # 3. Une feuille à 50 / 50 : predict renvoie la première classe.
    X_moitie = np.array([[1.0], [1.0], [2.0], [2.0]])
    _verifier_arbre(Classe, X_moitie, np.array(["oui", "non", "oui", "non"]), {},
                    "ArbreDecision().fit(X, y)  (des feuilles à 50 / 50)")

    # 4. Des clients tirés au hasard, plusieurs réglages.
    for graine, (profondeur, m) in enumerate([(1, 1), (3, 1), (4, 10), (None, 25), (2, 150), (6, 3)]):
        clients = generer_clients(500, graine=1450 + graine)
        X, y = clients[NUMERIQUES], clients["churn"]
        quoi = f"ArbreDecision(max_profondeur={profondeur}, min_feuille={m}).fit(X, y)  (500 clients tirés au hasard)"
        _verifier_arbre(Classe, X, y, {"max_profondeur": profondeur, "min_feuille": m}, quoi,
                        X_test=generer_clients(200, graine=2450 + graine)[NUMERIQUES])

    # 5. Le même arbre que scikit-learn.
    for graine, profondeur, m in ((5, 4, 10), (11, 5, 20)):
        clients = generer_clients(1500, graine=graine)
        X, y, X_test = clients[NUMERIQUES].iloc[:1000], clients["churn"].iloc[:1000], clients[NUMERIQUES].iloc[1000:]
        quoi = f"ArbreDecision(max_profondeur={profondeur}, min_feuille={m}).fit(X, y)  (1000 clients)"
        modele = _verifier_arbre(Classe, X, y, {"max_profondeur": profondeur, "min_feuille": m}, quoi)
        reference = DecisionTreeClassifier(max_depth=profondeur, min_samples_leaf=m, random_state=0).fit(X, y)
        if not _proches(modele.predict_proba(X_test), reference.predict_proba(X_test), rtol=0, atol=1e-12) \
                or modele.n_feuilles_ != reference.get_n_leaves():
            raise AssertionError(f"{quoi} : votre arbre a {modele.n_feuilles_} feuilles, celui de scikit-learn "
                                 f"{reference.get_n_leaves()}, et ils ne prédisent pas la même chose.")

    # 6. Des étiquettes, une Series d'un autre index, un deuxième fit, avant fit, X intact.
    clients = generer_clients(300, graine=1459)
    X, y = clients[NUMERIQUES], clients["churn"].map({0: "reste", 1: "part"}).reset_index(drop=True)
    modele = _verifier_arbre(Classe, X, y, {"max_profondeur": 2},
                             "ArbreDecision(max_profondeur=2).fit(X, y)  (y vaut \"part\" ou \"reste\", index 0 à 299)")
    autres = generer_clients(200, graine=1460)
    quoi = "modele.fit(X, y) puis modele.fit(autres_X, autres_y)  (un deuxième fit)"
    _appel(lambda: modele.fit(autres[NUMERIQUES].to_numpy(), autres["churn"].to_numpy()), quoi)
    _valider(modele.arbre_, autres[NUMERIQUES].to_numpy(dtype=float), autres["churn"].to_numpy(dtype=float), 0, 2, 1)
    _verifier_pas_entraine(lambda: Classe().predict_proba(X), "ArbreDecision().predict_proba(X)")
    for parametres, pourquoi in (({"max_profondeur": 0}, "une profondeur de 0, c'est une seule feuille : demandez au "
                                                         "moins 1 (ou None pour « sans limite »)"),
                                 ({"max_profondeur": 2.5}, "une profondeur est un nombre entier"),
                                 ({"max_profondeur": True}, "True n'est pas une profondeur"),
                                 ({"min_feuille": 0}, "une feuille vide n'a pas de sens"),
                                 ({"min_feuille": True}, "True n'est pas un nombre de clients")):
        texte = ", ".join(f"{k}={v!r}" for k, v in parametres.items())
        _leve(lambda: Classe(**parametres).fit(X, y), (ValueError, TypeError), f"ArbreDecision({texte}).fit(X, y)",
              f"Vérifiez les paramètres au début de `fit` ({pourquoi}).")
    _leve(lambda: Classe().fit(X, np.ones(len(X), dtype=int)), ValueError, "ArbreDecision().fit(X, <que des 1>)",
          "Il faut exactement deux classes dans y.")
    _verifier_clone(Classe(max_profondeur=4), "ArbreDecision(max_profondeur=4)")

    # 7. Une vraie taille : chaque nœud trie ses clients une fois par colonne.
    clients = generer_clients(5_000, graine=1461)
    X, y = clients[NUMERIQUES].to_numpy(dtype=float), clients["churn"].to_numpy()
    debut = time.perf_counter()
    _appel(lambda: Classe(max_profondeur=6, min_feuille=5).fit(X, y), "ArbreDecision(max_profondeur=6).fit(X, y)")
    duree = time.perf_counter() - debut
    if duree >= 4.0:
        raise AssertionError(f"L'arbre est juste, mais 5 000 clients et une profondeur de 6 prennent {duree:.1f} s (environ "
                             f"0,1 s attendu). Votre meilleure_coupure est-elle en O(n log n) (défi 144) ? Chaque nœud "
                             f"ne doit travailler que sur SES clients.")


SERIE = Serie(
    code="23",
    titre="Ouvrir la boîte noire : la régression logistique et l'arbre de décision recodés avec NumPy",
    module_eleve="defis.serie_23",
    module_corrige="defis.corrige_serie_23",
    defis=(
        Defi(140, "sigmoide", "🟢", "—", verifier_sigmoide),
        Defi(141, "perte_logistique", "🟢", "4.2", verifier_perte_logistique),
        Defi(142, "gradient_logistique", "🟡", "—", verifier_gradient_logistique),
        Defi(143, "RegressionLogistiqueGD", "🟡", "3.3", verifier_RegressionLogistiqueGD),
        Defi(144, "meilleure_coupure", "🟡", "3.3", verifier_meilleure_coupure),
        Defi(145, "ArbreDecision", "🔴", "3.3", verifier_ArbreDecision),
    ),
)

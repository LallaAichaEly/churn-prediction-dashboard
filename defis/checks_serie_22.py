"""Vérifications automatiques de la série 22 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur reçoit vos CLASSES (et, au défi 136, votre fonction), et s'en sert comme le ferait
scikit-learn lui-même :

- il appelle `fit` puis `transform` sur des données DIFFÉRENTES (un train et un test, dont les médianes,
  les effectifs et les taux de churn ne sont pas les mêmes) : ce qui est appris doit venir du train ;
- il appelle `transform` AVANT `fit`, `clone`, `get_params`, `set_params`, puis `fit` à nouveau, et
  vérifie que vos objets se comportent comme ceux de scikit-learn ;
- il branche vos briques dans un `Pipeline`, une validation croisée (`cross_val_score`) et une recherche
  de paramètres (`GridSearchCV`), et compare les scores à ceux du corrigé ;
- au défi 136, il note un faux modèle « déjà entraîné » qui compte ses appels à `fit` ; au défi 139, il
  vérifie que l'estimateur que vous recevez n'a JAMAIS été entraîné.

Quand un résultat est faux, il rejoue les versions boguées classiques (médiane du test, NaN devenu
« Autre », coût positif, mauvaise colonne de `predict_proba`, encodage qui voit sa propre réponse, seuil
choisi sur le train…) pour vous dire LAQUELLE vous avez écrite.
"""

from __future__ import annotations

import math
import numbers
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from .briques import CATEGORIES, NUMERIQUES, clients_exemple, generer_clients
from .verif import Defi, Serie

# ---------------------------------------------------------------------------- outils


def _decrire(exc: BaseException) -> str:
    texte = str(exc).split("\n")[0]
    if len(texte) > 160:
        texte = texte[:159] + "…"
    return f"{type(exc).__name__}({texte!r})" if texte else type(exc).__name__


def _appel(appel, quoi: str, conseil: str = ""):
    """Appelle `appel()` ; une exception inattendue devient un message qui dit QUEL appel a planté."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return appel()
    except NotImplementedError:
        raise
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


def _court(valeur, n: int = 120) -> str:
    if isinstance(valeur, pd.DataFrame):
        return f"un DataFrame de {valeur.shape[0]} lignes"
    if isinstance(valeur, BaseEstimator):
        return f"un objet {type(valeur).__name__}"
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


def _montrer(valeur) -> str:
    if valeur is None:
        return "None"
    try:
        if np.ndim(valeur) == 0 and pd.isna(valeur):
            return "NaN"
    except (TypeError, ValueError):
        pass
    if isinstance(valeur, (float, np.floating)):
        return f"{float(valeur):.4g}"
    if isinstance(valeur, np.integer):
        return str(int(valeur))
    return _court(valeur)


def _egal(obtenu, attendu) -> bool:
    try:
        o_manque, a_manque = bool(pd.isna(obtenu)), bool(pd.isna(attendu))
    except (TypeError, ValueError):
        return False
    if o_manque or a_manque:
        return o_manque and a_manque
    if isinstance(attendu, (float, np.floating)) and not isinstance(attendu, bool):
        try:
            return math.isclose(float(obtenu), float(attendu), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    try:
        return bool(obtenu == attendu)
    except (TypeError, ValueError):
        return False


def _premiere_difference(obtenu: pd.DataFrame, attendu: pd.DataFrame):
    """(étiquette, colonne, obtenu, attendu) de la première case qui diffère, ou None. Mêmes index supposés."""
    for colonne in attendu.columns:
        for etiquette in attendu.index:
            if not _egal(obtenu.at[etiquette, colonne], attendu.at[etiquette, colonne]):
                return etiquette, colonne, obtenu.at[etiquette, colonne], attendu.at[etiquette, colonne]
    return None


def _identiques(obtenu, attendu: pd.DataFrame) -> bool:
    try:
        return (isinstance(obtenu, pd.DataFrame) and list(obtenu.index) == list(attendu.index)
                and list(obtenu.columns) == list(attendu.columns) and _premiere_difference(obtenu, attendu) is None)
    except Exception:
        return False


def _exiger_table(resultat, attendu: pd.DataFrame, quoi: str) -> pd.DataFrame:
    """Le résultat d'un transform : un DataFrame, avec l'index et les colonnes de X."""
    if resultat is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if isinstance(resultat, np.ndarray):
        raise AssertionError(f"{quoi} renvoie un tableau numpy : renvoyez un DataFrame, avec l'index et les colonnes de "
                             f"X (les étapes suivantes s'en servent, et on garde l'identifiant de chaque client).")
    if not isinstance(resultat, pd.DataFrame):
        raise AssertionError(f"{quoi} doit renvoyer un DataFrame, pas un {type(resultat).__name__}.")
    if list(resultat.index) != list(attendu.index):
        raise AssertionError(f"{quoi} : l'index du résultat est {_court(list(resultat.index), 80)} au lieu de celui de X "
                             f"({_court(list(attendu.index), 80)}). Chaque ligne doit rester en face de SON client.")
    if list(resultat.columns) != list(attendu.columns):
        raise AssertionError(f"{quoi} : colonnes {list(resultat.columns)} au lieu de {list(attendu.columns)} (celles de "
                             f"X, dans le même ordre).")
    return resultat


def _difference_lisible(obtenu: pd.DataFrame, attendu: pd.DataFrame, quoi: str, conseil: str = "") -> None:
    difference = _premiere_difference(obtenu, attendu)
    if difference is None:
        return
    etiquette, colonne, o, a = difference
    raise AssertionError(f"{quoi} : ligne {etiquette!r}, colonne {colonne!r} : {_montrer(o)} au lieu de {_montrer(a)}. "
                         f"{conseil}".rstrip())


def _verifier_intact(apres, avant, quoi: str) -> None:
    try:
        if isinstance(avant, pd.DataFrame):
            pd.testing.assert_frame_equal(apres, avant)
        elif isinstance(avant, pd.Series):
            pd.testing.assert_series_equal(apres, avant)
        else:
            np.testing.assert_array_equal(np.asarray(apres), np.asarray(avant))
    except AssertionError:
        raise AssertionError(f"{quoi} a MODIFIÉ les données reçues. Une brique ne touche pas à ses entrées (l'appelant, "
                             f"ou l'étape suivante du Pipeline, s'en sert encore) : travaillez sur une copie, ou sur "
                             f"le résultat de `fillna` / `where` / `map`, qui sont de nouvelles tables.") from None


def _creer(Classe, quoi: str, **parametres):
    return _appel(lambda: Classe(**parametres), quoi)


def _verifier_fit_renvoie_self(objet, retour, quoi: str) -> None:
    if retour is None:
        raise AssertionError(f"{quoi} renvoie None : `fit` doit se terminer par `return self`. C'est ce qui permet "
                             f"d'écrire `Brique().fit(X).transform(X)`, et le Pipeline s'en sert.")
    if retour is not objet:
        raise AssertionError(f"{quoi} renvoie {_court(retour)} au lieu de l'objet lui-même : `fit` se termine par "
                             f"`return self` (ce qu'il a appris est rangé DANS l'objet, en attributs finissant par _).")


def _verifier_entraine(objet, quoi: str) -> None:
    try:
        check_is_fitted(objet)
    except NotFittedError:
        raise AssertionError(f"Après {quoi}, `check_is_fitted` croit l'objet encore NON entraîné : ce que `fit` apprend "
                             f"doit être rangé dans un attribut dont le nom FINIT PAR _ (par exemple `self.medianes_`). "
                             f"C'est la convention que scikit-learn vérifie.") from None


def _verifier_pas_entraine(appel, quoi: str) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resultat = appel()
    except NotImplementedError:
        raise
    except NotFittedError:
        return
    except AttributeError as exc:
        raise AssertionError(f"{quoi} AVANT fit lève {_decrire(exc)} : commencez par `check_is_fitted(self)`, qui lève une "
                             f"NotFittedError avec un message clair (« appelez fit d'abord »).") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} AVANT fit lève {_decrire(exc)} au lieu de NotFittedError : commencez par "
                             f"`check_is_fitted(self)`.") from None
    raise AssertionError(f"{quoi} AVANT fit renvoie {_court(resultat)} au lieu de lever NotFittedError. Rien n'a été "
                         f"appris : commencez par `check_is_fitted(self)`. (Un attribut en _ créé dans `__init__` ferait "
                         f"croire à scikit-learn que l'objet est déjà entraîné.)")


def _verifier_parametres(objet, attendus: dict, quoi: str) -> None:
    if not hasattr(objet, "get_params"):
        raise AssertionError(f"{quoi} n'a pas de `get_params` : la classe doit hériter de `BaseEstimator` (à droite du "
                             f"mixin : `class X(TransformerMixin, BaseEstimator):`).")
    try:
        parametres = objet.get_params(deep=False)
    except AttributeError as exc:
        raise AssertionError(f"{quoi}.get_params() lève {_decrire(exc)} : `__init__` doit ranger CHAQUE paramètre sous "
                             f"SON nom (`self.{next(iter(attendus))} = {next(iter(attendus))}`). `get_params` lit les noms "
                             f"dans la signature de `__init__`, puis va chercher l'attribut du même nom.") from None
    except RuntimeError as exc:
        raise AssertionError(f"{quoi}.get_params() lève {_decrire(exc)} : pas de `*args` dans `__init__`, chaque paramètre "
                             f"a un nom (et une valeur par défaut).") from None
    if set(parametres) != set(attendus):
        manquants, en_trop = sorted(set(attendus) - set(parametres)), sorted(set(parametres) - set(attendus))
        conseil = ("Pas de `**kwargs` : chaque paramètre est écrit dans la signature de `__init__`, car c'est elle que "
                   "`get_params` lit." if not parametres else
                   "Les paramètres sont EXACTEMENT ceux de la signature de `__init__`, avec leur nom.")
        raise AssertionError(f"{quoi}.get_params() donne {sorted(parametres)} : il manque {manquants or 'rien'}, en trop "
                             f"{en_trop or 'rien'}. {conseil}")
    for nom, valeur in attendus.items():
        obtenu = parametres[nom]
        if not (obtenu is valeur or (type(obtenu) is type(valeur) and _egal(obtenu, valeur))):
            raise AssertionError(f"{quoi}.get_params()[{nom!r}] vaut {_court(obtenu)} au lieu de {_court(valeur)} (la "
                                 f"valeur reçue). `__init__` range ses paramètres TELS QUELS : ni conversion, ni "
                                 f"vérification (elles ont leur place dans `fit`).")


def _construit_sans_verifier(Classe, quoi: str, **parametres) -> None:
    try:
        objet = Classe(**parametres)
    except NotImplementedError:
        raise
    except Exception as exc:
        raise AssertionError(
            f"{quoi} lève {_decrire(exc)} dès la construction. Pas de vérification dans `__init__` : `set_params` (que "
            f"GridSearchCV appelle) change les paramètres SANS repasser par `__init__`, et la vérification serait "
            f"contournée. Rangez la valeur telle quelle ; c'est `fit` qui la vérifie (et lève ValueError).") from None
    for nom, valeur in parametres.items():
        if getattr(objet, nom, object()) is not valeur:
            raise AssertionError(f"{quoi} : l'attribut `{nom}` ne vaut pas exactement la valeur reçue ({_court(valeur)}). "
                                 f"`__init__` la range telle quelle, sans la convertir.")


def _verifier_clone(objet, quoi: str):
    try:
        copie = clone(objet)
    except RuntimeError as exc:
        raise AssertionError(f"clone({quoi}) lève {_decrire(exc)} : `__init__` doit ranger chaque paramètre TEL QUEL, sous "
                             f"son nom, sans le modifier.") from None
    except Exception as exc:
        raise AssertionError(f"clone({quoi}) lève {_decrire(exc)}. La validation croisée clone vos briques à chaque "
                             f"pli : héritez de `BaseEstimator`, et rangez les paramètres tels quels.") from None
    return copie


def _x_y(n: int, graine: int):
    clients = generer_clients(n, graine=graine)
    return clients.drop(columns="churn"), clients["churn"]


# ============================================================================ Défi 134 : ImputeurMediane


def _train_test_numeriques():
    train = clients_exemple()[NUMERIQUES]
    test = pd.DataFrame({"anciennete": [np.nan, 50.0, 70.0, np.nan], "facture": [np.nan, 200.0, 210.0, 15.0],
                         "appels": [np.nan, 9.0, 8.0, np.nan]},
                        index=pd.Index(["T1", "T2", "T3", "T4"], name="client"))
    return train, test


def verifier_ImputeurMediane(Imputeur) -> None:
    train, test = _train_test_numeriques()
    imputeur = _creer(Imputeur, "ImputeurMediane()")
    quoi = "ImputeurMediane().fit(clients_exemple()[NUMERIQUES])"
    retour = _appel(lambda: imputeur.fit(train), quoi)
    _verifier_intact(train, _train_test_numeriques()[0], quoi)
    _verifier_fit_renvoie_self(imputeur, retour, quoi)
    _verifier_entraine(imputeur, quoi)

    # 1. Un test aux médianes différentes, avec un trou dans « appels » (aucun trou au train).
    quoi = "imputeur.transform(test)"
    obtenu = _exiger_table(_appel(lambda: imputeur.transform(test), quoi), test, quoi)
    _verifier_intact(test, _train_test_numeriques()[1], quoi)
    attendu = test.fillna(train.median())
    if not _identiques(obtenu, attendu):
        if obtenu.isna().any().any():
            colonne = obtenu.columns[obtenu.isna().any()][0]
            if colonne == "appels":
                raise AssertionError(
                    f"{quoi} laisse des NaN dans 'appels'. Cette colonne n'a aucun trou au train, mais le test en a : "
                    f"apprenez la médiane de TOUTES les colonnes, pas seulement de celles qui ont des trous.")
            raise AssertionError(f"{quoi} laisse des NaN dans {colonne!r}. `np.median` ne saute pas les NaN (une seule "
                                 f"case vide donne NaN) : utilisez `X.median()` de pandas, ou `np.nanmedian`.")
        pieges = [
            (test.fillna(test.median()),
             "les trous sont remplis avec la médiane du TEST. Ce qui est appris vient de `fit` (le train) ; en "
             "production, X n'a parfois qu'une ligne, et sa « médiane » serait le client lui-même. Rangez les "
             "médianes dans `fit` (`self.medianes_`), et réutilisez-les dans `transform`."),
            (test.fillna(train.mean()),
             "les trous sont remplis avec la MOYENNE. C10 paie 300 € : la moyenne des factures (80,81 €) en est tirée "
             "vers le haut, pas la médiane (60 €). Utilisez `X.median()`."),
            (test.fillna(0), "les trous sont remplis avec 0. Une ancienneté de 0 mois, c'est une information (un tout "
                             "nouveau client), pas « on ne sait pas » : utilisez la médiane du train."),
            (test.fillna(float(np.nanmedian(train.to_numpy(dtype=float)))),
             "les trous sont tous remplis avec la MÊME valeur : la médiane de toute la table. Chaque colonne a SA "
             "médiane (`X.median()` en donne une par colonne)."),
        ]
        for faux, conseil in pieges:
            if _identiques(obtenu, faux):
                raise AssertionError(f"{quoi} : {conseil}")
        _difference_lisible(obtenu, attendu, quoi, "Chaque NaN prend la médiane de sa colonne, apprise pendant fit ; "
                                                  "les autres valeurs ne changent pas.")

    # 2. Avant fit.
    _verifier_pas_entraine(lambda: _creer(Imputeur, "ImputeurMediane()").transform(test), "ImputeurMediane().transform(X)")

    # 3. Des colonnes qui ne sont pas celles du train.
    _leve(lambda: imputeur.transform(test.drop(columns="facture")), ValueError,
          "imputeur.transform(<X sans la colonne 'facture'>)",
          "X n'a pas les colonnes vues pendant fit : la préparation a changé, il faut le dire tout de suite (comparez "
          "`set(X.columns)` à `set(self.medianes_.index)`).")
    _leve(lambda: imputeur.transform(test.assign(remise=[np.nan, 1.0, 2.0, 3.0])), ValueError,
          "imputeur.transform(<X avec une colonne 'remise' en plus>)",
          "Une colonne en plus, jamais vue pendant fit, n'a pas de médiane : ses trous resteraient vides en silence.")

    # 4. Une colonne sans aucune valeur.
    vide = train.assign(facture=np.nan)
    _leve(lambda: _creer(Imputeur, "ImputeurMediane()").fit(vide), ValueError,
          "ImputeurMediane().fit(<X dont la colonne 'facture' est vide>)",
          "Sans aucune valeur, la médiane vaut NaN, et les trous ne seraient pas remplis. Refusez-le dans fit.")

    # 5. Réapprendre efface ce qu'on avait appris.
    autre = pd.DataFrame({"anciennete": [100.0, 102.0, 104.0], "facture": [10.0, 12.0, 14.0], "appels": [7, 9, 11]})
    quoi = "imputeur.fit(<un autre train>).transform(test)"
    obtenu = _exiger_table(_appel(lambda: imputeur.fit(autre).transform(test), quoi), test, quoi)
    if not _identiques(obtenu, test.fillna(autre.median())):
        raise AssertionError(f"{quoi} utilise encore les médianes du PREMIER train. Chaque `fit` réapprend tout (pas de "
                             f"`if not hasattr(self, ...)`).")

    # 6. fit_transform (TransformerMixin), clone et Pipeline.
    quoi = "ImputeurMediane().fit_transform(X)"
    if not hasattr(_creer(Imputeur, "ImputeurMediane()"), "fit_transform"):
        raise AssertionError("ImputeurMediane n'a pas de `fit_transform` : héritez de `TransformerMixin`, qui l'écrit pour "
                             "vous (`class ImputeurMediane(TransformerMixin, BaseEstimator):`).")
    obtenu = _exiger_table(_appel(lambda: _creer(Imputeur, "ImputeurMediane()").fit_transform(train), quoi), train, quoi)
    _difference_lisible(obtenu, train.fillna(train.median()), quoi)
    _verifier_clone(Imputeur(), "ImputeurMediane()")
    X, y = _x_y(300, 0)
    quoi = "cross_val_score(make_pipeline(ImputeurMediane(), LogisticRegression()), X, y, cv=3)"
    obtenu = _appel(lambda: cross_val_score(make_pipeline(Imputeur(), LogisticRegression()), X[NUMERIQUES], y, cv=3,
                                            scoring="roc_auc", error_score="raise"), quoi)
    attendu = cross_val_score(make_pipeline(SimpleImputer(strategy="median"), LogisticRegression()), X[NUMERIQUES], y,
                              cv=3, scoring="roc_auc")
    if not np.allclose(obtenu, attendu):
        raise AssertionError(f"{quoi} donne {np.round(obtenu, 4)} au lieu de {np.round(attendu, 4)} (le résultat de "
                             f"SimpleImputer(strategy=\"median\")).")

    # 7. Des trains et des tests tirés au hasard.
    rng = random.Random(134)
    for graine in range(3):
        X, _ = _x_y(rng.randint(40, 200), graine)
        coupe = len(X) // 2
        a_trous, b_trous = X[NUMERIQUES].iloc[:coupe], X[NUMERIQUES].iloc[coupe:]
        quoi = f"ImputeurMediane().fit(<{len(a_trous)} clients>).transform(<{len(b_trous)} autres>)"
        obtenu = _exiger_table(_appel(lambda: Imputeur().fit(a_trous).transform(b_trous), quoi), b_trous, quoi)
        _difference_lisible(obtenu, b_trous.fillna(a_trous.median()), quoi)


# ============================================================================ Défi 135 : RegroupeurRares


def _ref_regrouper(train: pd.DataFrame, test: pd.DataFrame, m, libelle="Autre", variante=None) -> pd.DataFrame:
    if variante == "commun":
        comptes = pd.concat([train[c] for c in train.columns]).value_counts()
    resultat = test.copy()
    for colonne in train.columns:
        if variante != "commun":
            comptes = (test if variante == "test" else train)[colonne].value_counts()
        gardees = set(comptes.index[comptes > m] if variante == "strict" else comptes.index[comptes >= m])
        garde = test[colonne].isin(gardees) if variante == "nan" else test[colonne].isin(gardees) | test[colonne].isna()
        resultat[colonne] = test[colonne].where(garde, libelle)
    return resultat


def _test_categories():
    return pd.DataFrame({"contrat": ["Mensuel", "Annuel", "Triennal", np.nan, "Biannuel"],
                         "ville": ["Lyon", "Quimper", np.nan, "Albi", "Paris"]},
                        index=pd.Index(["T1", "T2", "T3", "T4", "T5"], name="client"))


def verifier_RegroupeurRares(Regroupeur) -> None:
    train = clients_exemple()[CATEGORIES]

    # 1. Les paramètres : tels quels, sous leur nom, sans vérification.
    defaut = _creer(Regroupeur, "RegroupeurRares()")
    _verifier_parametres(defaut, {"min_effectif": 10, "libelle": "Autre"}, "RegroupeurRares()")
    for valeur in (0, "dix"):
        _construit_sans_verifier(Regroupeur, f"RegroupeurRares(min_effectif={valeur!r})", min_effectif=valeur)
    _verifier_parametres(Regroupeur(min_effectif=3, libelle="Rare"), {"min_effectif": 3, "libelle": "Rare"},
                         "RegroupeurRares(min_effectif=3, libelle='Rare')")

    # 2. L'exemple, avec un seuil de 3 (Lyon a pile 3 clients).
    regroupeur = Regroupeur(min_effectif=3)
    quoi = "RegroupeurRares(min_effectif=3).fit(clients_exemple()[CATEGORIES])"
    retour = _appel(lambda: regroupeur.fit(train), quoi)
    _verifier_intact(train, clients_exemple()[CATEGORIES], quoi)
    _verifier_fit_renvoie_self(regroupeur, retour, quoi)
    _verifier_entraine(regroupeur, quoi)
    test = _test_categories()
    quoi = "regroupeur.transform(test)"
    obtenu = _exiger_table(_appel(lambda: regroupeur.transform(test), quoi), test, quoi)
    _verifier_intact(test, _test_categories(), quoi)
    attendu = _ref_regrouper(train, test, 3)
    if not _identiques(obtenu, attendu):
        pieges = [
            ("nan", "une case VIDE (NaN) est devenue 'Autre'. `isin` vaut False pour NaN : gardez aussi les NaN "
                    "(`X[colonne].isin(gardees) | X[colonne].isna()`). « On ne sait pas » n'est pas « une petite ville »."),
            ("strict", "Lyon (pile 3 clients, min_effectif=3) est devenue 'Autre'. « Au moins min_effectif fois » : "
                       "`>=`, pas `>`."),
            ("test", "les effectifs sont comptés sur X reçu par TRANSFORM. Ils s'apprennent dans `fit` (le train) : en "
                     "production, une seule ligne n'a que des modalités « rares »."),
        ]
        for variante, conseil in pieges:
            if _identiques(obtenu, _ref_regrouper(train, test, 3, variante=variante)):
                raise AssertionError(f"{quoi} : {conseil}")
        _difference_lisible(obtenu, attendu, quoi, "Une valeur qui n'est pas retenue pour SA colonne (rare au train, ou "
                                                  "jamais vue) devient le libellé ; NaN reste NaN.")

    # 3. Chaque colonne a SES modalités.
    deux = pd.DataFrame({"a": ["x", "x", "y"], "b": ["y", "z", "z"]})
    quoi = "RegroupeurRares(min_effectif=2).fit_transform(pd.DataFrame({'a': ['x', 'x', 'y'], 'b': ['y', 'z', 'z']}))"
    obtenu = _exiger_table(_appel(lambda: Regroupeur(min_effectif=2).fit_transform(deux), quoi), deux, quoi)
    if not _identiques(obtenu, _ref_regrouper(deux, deux, 2)):
        if _identiques(obtenu, _ref_regrouper(deux, deux, 2, variante="commun")):
            raise AssertionError(f"{quoi} garde 'y' dans la colonne 'a', où il n'apparaît qu'une fois : les effectifs sont "
                                 f"comptés sur TOUTES les colonnes ensemble. Comptez colonne par colonne.")
        _difference_lisible(obtenu, _ref_regrouper(deux, deux, 2), quoi)

    # 4. Le libellé est un paramètre.
    quoi = "RegroupeurRares(min_effectif=3, libelle='Rare').fit(...).transform(test)"
    obtenu = _exiger_table(_appel(lambda: Regroupeur(min_effectif=3, libelle="Rare").fit(train).transform(test), quoi),
                           test, quoi)
    _difference_lisible(obtenu, _ref_regrouper(train, test, 3, "Rare"), quoi, "Utilisez `self.libelle`.")

    # 5. Les mauvaises valeurs sont refusées… dans fit, même après set_params.
    for valeur in (0, -1, 2.5, True, None, "10"):
        erreurs = ValueError if isinstance(valeur, numbers.Real) else (ValueError, TypeError)
        _leve(lambda: Regroupeur(min_effectif=valeur).fit(train), erreurs,
              f"RegroupeurRares(min_effectif={valeur!r}).fit(X)",
              "min_effectif doit être un entier supérieur ou égal à 1 (et `True` n'en est pas un, même si "
              "`isinstance(True, int)` vaut True).")
    _leve(lambda: Regroupeur().set_params(min_effectif=0).fit(train), ValueError,
          "RegroupeurRares().set_params(min_effectif=0).fit(X)",
          "`set_params` ne repasse pas par `__init__` : c'est dans `fit` que la valeur se vérifie.")
    _appel(lambda: Regroupeur(min_effectif=np.int64(3)).fit(train), "RegroupeurRares(min_effectif=np.int64(3)).fit(X)",
           "Un entier numpy est un entier : `isinstance(m, numbers.Integral)` l'accepte, `isinstance(m, int)` non.")

    # 6. set_params, clone, et réapprendre.
    quoi = "RegroupeurRares(min_effectif=3).set_params(min_effectif=4).fit(X).transform(test)"
    obtenu = _exiger_table(_appel(lambda: Regroupeur(min_effectif=3).set_params(min_effectif=4).fit(train).transform(test),
                                  quoi), test, quoi)
    if not _identiques(obtenu, _ref_regrouper(train, test, 4)):
        if _identiques(obtenu, _ref_regrouper(train, test, 3)):
            raise AssertionError(f"{quoi} se comporte encore comme avec min_effectif=3. `fit` doit lire `self.min_effectif` "
                                 f"(que `set_params` vient de changer), pas une copie faite dans `__init__`.")
        _difference_lisible(obtenu, _ref_regrouper(train, test, 4), quoi)
    quoi = "regroupeur.set_params(min_effectif=4).fit(X).transform(test)  (regroupeur déjà entraîné avec 3)"
    obtenu = _exiger_table(_appel(lambda: regroupeur.set_params(min_effectif=4).fit(train).transform(test), quoi),
                           test, quoi)
    if not _identiques(obtenu, _ref_regrouper(train, test, 4)):
        raise AssertionError(f"{quoi} garde des modalités apprises au fit PRÉCÉDENT (Lyon, retenue avec 3). Chaque `fit` "
                             f"repart de zéro : un NOUVEAU dictionnaire, pas `self.modalites_.update(...)`.")
    copie = _verifier_clone(regroupeur, "regroupeur")
    _verifier_pas_entraine(lambda: copie.transform(test), "clone(regroupeur).transform(X)")
    _verifier_pas_entraine(lambda: Regroupeur().transform(test), "RegroupeurRares().transform(X)")
    autre = pd.DataFrame({"contrat": ["Annuel"] * 4 + ["Mensuel"], "ville": ["Albi"] * 4 + ["Paris"]})
    quoi = "regroupeur.fit(<un autre train>).transform(test)"
    obtenu = _exiger_table(_appel(lambda: regroupeur.fit(autre).transform(test), quoi), test, quoi)
    if not _identiques(obtenu, _ref_regrouper(autre, test, 4)):
        raise AssertionError(f"{quoi} garde des modalités apprises au fit PRÉCÉDENT (Paris, Lyon…). Chaque `fit` "
                             f"repart de zéro : un NOUVEAU dictionnaire, pas `self.modalites_.update(...)`.")

    # 7. Dans une recherche de paramètres.
    X, y = _x_y(400, 1)
    grille = {"regroupeur__min_effectif": [1, 40]}
    quoi = ("GridSearchCV(Pipeline([('regroupeur', RegroupeurRares()), ('onehot', ...), ('modele', ...)]), "
            "{'regroupeur__min_effectif': [1, 40]})")

    def recherche(Classe):
        modele = Pipeline([("regroupeur", Classe()), ("onehot", OneHotEncoder(handle_unknown="ignore")),
                           ("modele", LogisticRegression())])
        return GridSearchCV(modele, grille, cv=3, scoring="roc_auc", error_score="raise").fit(X[CATEGORIES], y)

    obtenu = _appel(lambda: recherche(Regroupeur).cv_results_["mean_test_score"], quoi)
    attendu = recherche(_RegroupeurReference).cv_results_["mean_test_score"]
    if not np.allclose(obtenu, attendu):
        raise AssertionError(f"{quoi} donne les scores {np.round(obtenu, 4)} au lieu de {np.round(attendu, 4)}.")

    # 8. Des données tirées au hasard.
    rng = random.Random(135)
    for graine in range(3):
        X, _ = _x_y(rng.randint(50, 250), graine + 10)
        coupe, m = len(X) // 2, rng.randint(1, 12)
        a, b = X[CATEGORIES].iloc[:coupe], X[CATEGORIES].iloc[coupe:]
        quoi = f"RegroupeurRares(min_effectif={m}).fit(<{len(a)} clients>).transform(<{len(b)} autres>)"
        obtenu = _exiger_table(_appel(lambda: Regroupeur(min_effectif=m).fit(a).transform(b), quoi), b, quoi)
        _difference_lisible(obtenu, _ref_regrouper(a, b, m), quoi)


class _RegroupeurReference(BaseEstimator):
    """La référence du correcteur, pour comparer les scores d'une recherche de paramètres."""

    def __init__(self, min_effectif=10, libelle="Autre"):
        self.min_effectif = min_effectif
        self.libelle = libelle

    def fit(self, X, y=None):
        self.train_ = X.copy()
        return self

    def transform(self, X):
        return _ref_regrouper(self.train_, X, self.min_effectif, self.libelle)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# ============================================================================ Défi 136 : scoreur_cout


class _ModeleFactice(ClassifierMixin, BaseEstimator):
    """Un modèle « déjà entraîné » : X['p'] est la probabilité de la classe `positif`. Il compte ses appels."""

    def __init__(self, classes=(0, 1), positif=1):
        self.classes = classes
        self.positif = positif
        self.classes_ = np.array(list(classes), dtype=object if isinstance(classes[0], str) else None)
        self.appels = {"fit": 0, "predict": 0, "predict_proba": 0}

    def fit(self, X, y=None):
        self.appels["fit"] += 1
        return self

    def predict_proba(self, X):
        self.appels["predict_proba"] += 1
        p = np.asarray(X["p"], dtype=float)
        colonne = list(self.classes_).index(self.positif)
        proba = np.empty((len(p), 2))
        proba[:, colonne], proba[:, 1 - colonne] = p, 1 - p
        return proba

    def predict(self, X):
        self.appels["predict"] += 1
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


_P_FACTICES = [0.9, 0.3, 0.29, 0.1, 0.6, 0.05, 0.25, 0.2, 0.35, 0.7]
_Y_FACTICES = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]


def _cout_moyen(p, reel, seuil, cout_fn, cout_fp, strict=False):
    p, reel = np.asarray(p, dtype=float), np.asarray(reel, dtype=bool)
    alerte = p > seuil if strict else p >= seuil
    return (cout_fn * np.sum(reel & ~alerte) + cout_fp * np.sum(~reel & alerte)) / len(p)


def _scoreur_reference(cout_fn, cout_fp, seuil=0.5, positif=1):
    def scoreur(estimateur, X, y):
        proba = estimateur.predict_proba(X)[:, list(estimateur.classes_).index(positif)]
        return -float(_cout_moyen(proba, np.asarray(y) == positif, seuil, cout_fn, cout_fp))
    return scoreur


def _noter(scoreur, modele, X, y, quoi: str) -> float:
    valeur = _appel(lambda: scoreur(modele, X, y), quoi,
                    "y peut être une liste, un tableau numpy ou une Series (scikit-learn passe souvent un tableau) : "
                    "`np.asarray(y)` les traite tous de la même façon.")
    if valeur is None:
        raise AssertionError(f"{quoi} renvoie None : le scoreur doit RENVOYER le score (`return`).")
    if isinstance(valeur, bool) or not isinstance(valeur, numbers.Real):
        raise AssertionError(f"{quoi} renvoie {_court(valeur)} : un score est UN nombre (`float(...)`).")
    if modele.appels["fit"]:
        raise AssertionError(f"{quoi} a appelé `fit` sur l'estimateur. Le scoreur reçoit un modèle DÉJÀ entraîné (sur les "
                             f"autres plis) ; le réentraîner sur les données de test, c'est noter un modèle qui a vu "
                             f"les réponses.")
    return float(valeur)


def verifier_scoreur_cout(scoreur_cout) -> None:
    quoi = "scoreur_cout(500, 50, seuil=0.3)"
    scoreur = _appel(lambda: scoreur_cout(500, 50, seuil=0.3), quoi)
    if scoreur is None:
        raise AssertionError(f"{quoi} renvoie None : il doit renvoyer la FONCTION `scoreur` (`return scoreur`, sans "
                             f"parenthèses).")
    if not callable(scoreur):
        raise AssertionError(f"{quoi} renvoie {_court(scoreur)} : il doit renvoyer une FONCTION `scoreur(estimateur, X, "
                             f"y)`, que scikit-learn appellera sur chaque pli (`return scoreur`, sans parenthèses).")

    # 1. Un modèle factice : chaque erreur classique donne un chiffre différent.
    X = pd.DataFrame({"p": _P_FACTICES}, index=[f"T{i}" for i in range(10)])
    y = np.array(_Y_FACTICES)
    modele = _ModeleFactice()
    quoi = "scoreur_cout(500, 50, seuil=0.3)(modele, X, y)"
    obtenu = _noter(scoreur, modele, X, y.copy(), quoi)
    reel, p = y == 1, np.array(_P_FACTICES)
    attendu = -_cout_moyen(p, reel, 0.3, 500, 50)
    if not math.isclose(obtenu, attendu, abs_tol=1e-9):
        pieges = [
            (-attendu, "le score est POSITIF. scikit-learn MAXIMISE toujours le score : avec un coût positif, "
                       "GridSearchCV choisirait le modèle le plus CHER. Renvoyez MOINS le coût."),
            (attendu * len(p), "c'est le coût TOTAL. Les plis n'ont pas tous la même taille : divisez par le nombre de "
                               "clients (coût MOYEN par client)."),
            (-attendu * len(p), "c'est le coût TOTAL, et positif. Renvoyez MOINS le coût MOYEN par client."),
            (-_cout_moyen(p, reel, 0.3, 500, 50, strict=True),
             "le client à 0,30 pile n'a pas d'alerte. « Supérieure OU ÉGALE au seuil » : `>=`, pas `>`."),
            (-_cout_moyen(p, reel, 0.5, 500, 50),
             "les alertes sont celles de `predict`, c'est-à-dire au seuil de 0,5. Le seuil est un paramètre : "
             "comparez `predict_proba(X)[:, colonne]` à `seuil`."),
            (-_cout_moyen(1 - p, reel, 0.3, 500, 50),
             "vous lisez la probabilité de la MAUVAISE classe (la colonne de « reste »). Prenez la colonne de "
             "`positif` : `list(estimateur.classes_).index(positif)`."),
            (-_cout_moyen(p, reel, 0.3, 50, 500),
             "les deux coûts sont inversés. Faux négatif (parti sans alerte) : cout_fn ; faux positif (alerte pour "
             "rien) : cout_fp."),
        ]
        for faux, conseil in pieges:
            if math.isclose(obtenu, faux, abs_tol=1e-9):
                raise AssertionError(f"{quoi} vaut {_montrer(obtenu)} au lieu de {_montrer(attendu)} : {conseil}")
        raise AssertionError(f"{quoi} vaut {_montrer(obtenu)} au lieu de {_montrer(attendu)} (2 faux négatifs × 500 € + "
                             f"1 faux positif × 50 €, pour 10 clients, avec un signe moins).")
    _verifier_intact(y, np.array(_Y_FACTICES), quoi)

    # 2. y sous toutes ses formes.
    for forme, valeur in (("une liste", list(_Y_FACTICES)),
                          ("une Series d'index 0 à 9", pd.Series(_Y_FACTICES)),
                          ("une Series à l'index de X", pd.Series(_Y_FACTICES, index=X.index))):
        quoi = f"scoreur(modele, X, <y : {forme}>)"
        obtenu = _noter(scoreur, _ModeleFactice(), X, valeur, quoi)
        if not math.isclose(obtenu, attendu, abs_tol=1e-9):
            raise AssertionError(f"{quoi} vaut {_montrer(obtenu)} au lieu de {_montrer(attendu)}. `np.asarray(y)` traite "
                                 f"toutes les formes de y de la même façon (par position).")

    # 3. Des étiquettes en texte : la colonne de la classe positive se lit dans classes_.
    for classes, positif in ((("non", "oui"), "oui"), (("part", "reste"), "part")):
        negatif = classes[1] if positif == classes[0] else classes[0]
        etiquettes = np.array([positif if v else negatif for v in _Y_FACTICES], dtype=object)
        quoi = f"scoreur_cout(500, 50, seuil=0.3, positif={positif!r})(<modèle aux classes {list(classes)}>, X, y)"
        modele = _ModeleFactice(classes=classes, positif=positif)
        obtenu = _noter(_appel(lambda: scoreur_cout(500, 50, seuil=0.3, positif=positif), quoi), modele, X, etiquettes,
                        quoi)
        if not math.isclose(obtenu, attendu, abs_tol=1e-9):
            if positif == "part" and math.isclose(obtenu, -_cout_moyen(1 - p, reel, 0.3, 500, 50), abs_tol=1e-9):
                raise AssertionError(
                    f"{quoi} vaut {_montrer(obtenu)} au lieu de {_montrer(attendu)} : `predict_proba(X)[:, 1]` n'est pas "
                    f"toujours le churn. Les classes sont rangées dans l'ordre (classes_ = ['part', 'reste']) : ici, "
                    f"« part » est la colonne 0. Cherchez-la : `list(estimateur.classes_).index(positif)`.")
            raise AssertionError(f"{quoi} vaut {_montrer(obtenu)} au lieu de {_montrer(attendu)}. Une alerte, c'est la "
                                 f"probabilité de `positif` >= seuil ; un vrai départ, c'est `y == positif`.")
    quoi = "scoreur_cout(500, 50, positif='oui')(<modèle aux classes [0, 1]>, X, y)"
    _leve(lambda: scoreur_cout(500, 50, positif="oui")(_ModeleFactice(), X, y), ValueError, quoi,
          "La classe positive n'existe pas dans `estimateur.classes_` : c'est une erreur de configuration, à signaler.")

    # 4. Les paramètres se vérifient À LA CRÉATION.
    for arguments in ((-500, 50), (500, -1), (500, 50, 1.5), (500, 50, -0.1), (float("nan"), 50)):
        texte = ", ".join(repr(a) if not (isinstance(a, float) and math.isnan(a)) else "nan" for a in arguments)
        _leve(lambda: scoreur_cout(*arguments), ValueError, f"scoreur_cout({texte})",
              "Vérifiez les paramètres dès l'appel de `scoreur_cout` (coûts positifs ou nuls, seuil entre 0 et 1), pas "
              "dans le scoreur : l'erreur arriverait au milieu de la recherche.")
    for arguments in ((0, 0, 0.0), (500, 50, 1.0)):
        _appel(lambda: scoreur_cout(*arguments), f"scoreur_cout{arguments}", "Ces valeurs sont permises (bornes incluses).")

    # 5. Avec les vrais outils de scikit-learn.
    X, y = _x_y(600, 2)
    modele = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression())
    for seuil in (0.5, 0.15):
        quoi = f"cross_val_score(modele, X, y, cv=5, scoring=scoreur_cout(500, 50, seuil={seuil}))"
        obtenu = _appel(lambda: cross_val_score(modele, X[NUMERIQUES], y, cv=5, error_score="raise",
                                                scoring=scoreur_cout(500, 50, seuil=seuil)), quoi)
        attendu = cross_val_score(modele, X[NUMERIQUES], y, cv=5, scoring=_scoreur_reference(500, 50, seuil=seuil))
        if not np.allclose(obtenu, attendu):
            raise AssertionError(f"{quoi} donne {np.round(obtenu, 2)} au lieu de {np.round(attendu, 2)}.")
    quoi = "GridSearchCV(modele, {'logisticregression__C': [0.001, 1.0]}, scoring=scoreur_cout(500, 50, seuil=0.15))"
    recherche = _appel(lambda: GridSearchCV(modele, {"logisticregression__C": [0.001, 1.0]}, cv=3, error_score="raise",
                                            scoring=scoreur_cout(500, 50, seuil=0.15)).fit(X[NUMERIQUES], y), quoi)
    attendu = GridSearchCV(modele, {"logisticregression__C": [0.001, 1.0]}, cv=3,
                           scoring=_scoreur_reference(500, 50, seuil=0.15)).fit(X[NUMERIQUES], y)
    if recherche.best_params_ != attendu.best_params_:
        raise AssertionError(f"{quoi} choisit {recherche.best_params_} au lieu de {attendu.best_params_}.")

    # 6. Des probabilités et des coûts tirés au hasard.
    rng = random.Random(136)
    for _ in range(20):
        n = rng.randint(1, 40)
        X = pd.DataFrame({"p": [round(rng.random(), 2) for _ in range(n)]})
        y = np.array([rng.randint(0, 1) for _ in range(n)])
        cout_fn, cout_fp, seuil = rng.choice([500, 120.5, 0]), rng.choice([50, 7.25, 0]), rng.choice([0.0, 0.2, 0.5, 1])
        quoi = f"scoreur_cout({cout_fn}, {cout_fp}, seuil={seuil})(modele, <{n} clients tirés au hasard>, y)"
        obtenu = _noter(_appel(lambda: scoreur_cout(cout_fn, cout_fp, seuil=seuil), quoi), _ModeleFactice(), X, y, quoi)
        attendu = -_cout_moyen(X["p"], y == 1, seuil, cout_fn, cout_fp)
        if not math.isclose(obtenu, attendu, abs_tol=1e-9):
            raise AssertionError(f"{quoi} vaut {_montrer(obtenu)} au lieu de {_montrer(attendu)}.")


# ============================================================================ Défi 137 : EncodeurCible


def _ref_encodages(X: pd.DataFrame, y, lissage, variante=None):
    y = np.asarray(y, dtype=float)
    moyenne = float(y.mean())
    encodages = {}
    for colonne in X.columns:
        valeurs = X[colonne].fillna("(vide)") if variante == "nan_modalite" else X[colonne]
        stats = pd.Series(y).groupby(valeurs.to_numpy()).agg(["sum", "count"])
        if variante == "moyenne_des_taux":
            moyenne = float((stats["sum"] / stats["count"]).mean())
        if variante == "brut":
            encodages[colonne] = stats["sum"] / stats["count"]
        elif variante == "poids_inverses":
            encodages[colonne] = (stats["count"] * moyenne + lissage * stats["sum"] / stats["count"]) / (
                stats["count"] + lissage)
        else:
            encodages[colonne] = (stats["sum"] + lissage * moyenne) / (stats["count"] + lissage)
    return moyenne, encodages


def _ref_encoder(X_train, y, X_test, lissage=10.0, variante=None) -> pd.DataFrame:
    moyenne, encodages = _ref_encodages(X_train, y, lissage, variante)
    resultat = pd.DataFrame(index=X_test.index)
    for colonne, encodage in encodages.items():
        valeurs = X_test[colonne].fillna("(vide)") if variante == "nan_modalite" else X_test[colonne]
        code = valeurs.map(encodage).astype(float)
        resultat[colonne] = code if variante == "inconnu_nan" else code.fillna(0.0 if variante == "inconnu_zero" else moyenne)
    return resultat


def _nouveaux_clients():
    return pd.DataFrame({"contrat": ["Mensuel", "Annuel", np.nan, "Triennal"],
                         "ville": ["Albi", "Paris", "Quimper", np.nan]},
                        index=pd.Index(["N1", "N2", "N3", "N4"], name="client"))


def verifier_EncodeurCible(Encodeur) -> None:
    clients = clients_exemple()
    X, y = clients[CATEGORIES], clients["churn"]

    # 1. Le paramètre.
    _verifier_parametres(_creer(Encodeur, "EncodeurCible()"), {"lissage": 10.0}, "EncodeurCible()")
    _construit_sans_verifier(Encodeur, "EncodeurCible(lissage=-1)", lissage=-1)

    # 2. L'exemple : une ville d'un seul client, une ville inconnue, une case vide.
    encodeur = Encodeur()
    quoi = "EncodeurCible().fit(clients_exemple()[CATEGORIES], clients_exemple()['churn'])"
    retour = _appel(lambda: encodeur.fit(X, y), quoi)
    _verifier_intact(X, clients_exemple()[CATEGORIES], quoi)
    _verifier_fit_renvoie_self(encodeur, retour, quoi)
    _verifier_entraine(encodeur, quoi)
    nouveaux = _nouveaux_clients()
    quoi = "encodeur.transform(nouveaux)"
    obtenu = _exiger_table(_appel(lambda: encodeur.transform(nouveaux), quoi), nouveaux, quoi)
    _verifier_intact(nouveaux, _nouveaux_clients(), quoi)
    if not all(obtenu[c].dtype.kind == "f" for c in obtenu.columns):
        raise AssertionError(f"{quoi} : des colonnes de type {list(obtenu.dtypes.astype(str))} au lieu de nombres décimaux "
                             f"(`.astype(float)`).")
    attendu = _ref_encoder(X, y, nouveaux)
    if not _identiques(obtenu, attendu):
        pieges = [
            ("brut", "Albi (1 client, parti) vaut 1.0 : c'est le taux BRUT. Lissez vers la moyenne : (somme + lissage × "
                     "moyenne) / (effectif + lissage)."),
            ("poids_inverses", "le lissage est à l'envers : le poids `lissage` va à la MOYENNE GLOBALE, et l'effectif de "
                               "la ville à son propre taux. (somme + lissage × moyenne) / (effectif + lissage)."),
            ("moyenne_des_taux", "la « moyenne » est celle des taux des villes (chaque ville compte autant). C'est la "
                                 "moyenne de TOUT y : chaque CLIENT compte autant (`np.mean(y)`)."),
            ("inconnu_nan", "une ville jamais vue (Quimper) ou une case vide donne NaN. On ne sait rien d'elle : donnez-lui "
                            "la moyenne globale (`.fillna(self.moyenne_)`)."),
            ("inconnu_zero", "une ville jamais vue (Quimper) ou une case vide donne 0, c'est-à-dire « personne n'y part ». "
                             "On ne sait rien d'elle : donnez-lui la moyenne globale."),
            ("nan_modalite", "la case vide est encodée comme une ville (avec le churn de C12). Une case vide n'est pas une "
                             "modalité : elle reçoit la moyenne globale (`groupby` écarte les NaN tout seul)."),
        ]
        for variante, conseil in pieges:
            if _identiques(obtenu, _ref_encoder(X, y, nouveaux, variante=variante)):
                raise AssertionError(f"{quoi} : {conseil}")
        _difference_lisible(obtenu, attendu, quoi, "Encodage lissé de la modalité, dans SA colonne ; inconnue ou NaN : "
                                                  "la moyenne de tout y.")

    # 3. y par position, sous toutes ses formes.
    for forme, valeur in (("une liste", y.tolist()), ("un tableau numpy", y.to_numpy()),
                          ("une Series d'index 0 à 11 (après un reset_index)", y.reset_index(drop=True)),
                          ("une Series de booléens", y.astype(bool))):
        quoi = f"EncodeurCible().fit(X, <y : {forme}>).transform(nouveaux)"
        obtenu = _exiger_table(_appel(lambda: Encodeur().fit(X, valeur).transform(nouveaux), quoi,
                                      "y va avec X PAR POSITION, et peut être une liste, un tableau ou une Series : "
                                      "convertissez d'abord (`y = np.asarray(y, dtype=float)`)."), nouveaux, quoi)
        _difference_lisible(obtenu, attendu, quoi,
                            "y va avec X PAR POSITION (la 1re valeur avec la 1re ligne), quel que soit son index : "
                            "`np.asarray(y, dtype=float)`, puis grouper par `X[colonne].to_numpy()`.")

    # 4. Ce qu'il faut refuser.
    _leve(lambda: Encodeur().fit(X), ValueError, "EncodeurCible().fit(X)  (sans y)",
          "Un encodage par la CIBLE a besoin de la cible : refusez y=None avec un message clair.")
    _leve(lambda: Encodeur().fit(X, y.iloc[:-1]), ValueError, "EncodeurCible().fit(X, <y d'une ligne de moins>)",
          "Une ligne de X sans réponse, c'est un fichier mal assemblé : comparez `len(y)` à `len(X)`.")
    _leve(lambda: Encodeur().fit(X, y.tolist() + [1]), ValueError, "EncodeurCible().fit(X, <y d'une ligne de plus>)",
          "Une réponse sans ligne, c'est un fichier mal assemblé (et couper y en silence décale tout) : comparez "
          "`len(y)` à `len(X)`.")
    for valeur in (-1, True, "10"):
        erreurs = ValueError if isinstance(valeur, numbers.Real) else (ValueError, TypeError)
        _leve(lambda: Encodeur(lissage=valeur).fit(X, y), erreurs, f"EncodeurCible(lissage={valeur!r}).fit(X, y)",
              "lissage doit être un nombre positif ou nul (pas un booléen) : vérifiez-le dans fit.")
    _leve(lambda: Encodeur().set_params(lissage=-5).fit(X, y), ValueError,
          "EncodeurCible().set_params(lissage=-5).fit(X, y)", "C'est dans `fit` que le paramètre se vérifie.")
    _verifier_pas_entraine(lambda: Encodeur().transform(nouveaux), "EncodeurCible().transform(X)")

    # 5. lissage=0 : le taux brut ; réapprendre ; clone.
    quoi = "EncodeurCible(lissage=0).fit(X, y).transform(nouveaux)"
    obtenu = _exiger_table(_appel(lambda: Encodeur(lissage=0).fit(X, y).transform(nouveaux), quoi), nouveaux, quoi)
    _difference_lisible(obtenu, _ref_encoder(X, y, nouveaux, lissage=0), quoi, "Avec lissage=0, c'est le taux brut.")
    autre_y = 1 - y
    quoi = "encodeur.fit(X, 1 - y).transform(nouveaux)"
    obtenu = _exiger_table(_appel(lambda: encodeur.fit(X, autre_y).transform(nouveaux), quoi), nouveaux, quoi)
    _difference_lisible(obtenu, _ref_encoder(X, autre_y, nouveaux), quoi, "Chaque `fit` réapprend tout.")
    _verifier_clone(encodeur, "encodeur")

    # 6. Des données tirées au hasard, et dans un Pipeline.
    rng = random.Random(137)
    for graine in range(3):
        X_h, y_h = _x_y(rng.randint(60, 300), graine + 20)
        coupe, lissage = len(X_h) // 2, rng.choice([0, 1, 5.5, 20])
        a, b = X_h[CATEGORIES].iloc[:coupe], X_h[CATEGORIES].iloc[coupe:]
        quoi = f"EncodeurCible(lissage={lissage}).fit(<{len(a)} clients>, y).transform(<{len(b)} autres>)"
        obtenu = _exiger_table(_appel(lambda: Encodeur(lissage=lissage).fit(a, y_h.iloc[:coupe]).transform(b), quoi),
                               b, quoi)
        _difference_lisible(obtenu, _ref_encoder(a, y_h.iloc[:coupe], b, lissage), quoi)
    X_h, y_h = _x_y(400, 3)
    quoi = "cross_val_score(make_pipeline(EncodeurCible(), LogisticRegression()), X, y, cv=3)"
    obtenu = _appel(lambda: cross_val_score(make_pipeline(Encodeur(), LogisticRegression()), X_h[CATEGORIES], y_h, cv=3,
                                            scoring="roc_auc", error_score="raise"), quoi)
    attendu = cross_val_score(make_pipeline(_EncodeurReference(), LogisticRegression()), X_h[CATEGORIES], y_h, cv=3,
                              scoring="roc_auc")
    if not np.allclose(obtenu, attendu):
        raise AssertionError(f"{quoi} donne {np.round(obtenu, 4)} au lieu de {np.round(attendu, 4)}.")


class _EncodeurReference(BaseEstimator):
    def __init__(self, lissage=10.0):
        self.lissage = lissage

    def fit(self, X, y):
        self.train_, self.y_ = X.copy(), np.asarray(y, dtype=float)
        return self

    def transform(self, X):
        return _ref_encoder(self.train_, self.y_, X, self.lissage)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# ============================================================================ Défi 138 : EncodeurCibleHorsPli


def _ref_hors_pli(X, y, lissage=10.0, n_plis=5, graine=0, variante=None) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    if variante == "naif":
        return _ref_encoder(X, y, X, lissage)
    resultat = pd.DataFrame(np.nan, index=X.index, columns=list(X.columns), dtype=float)
    if variante == "loo":
        for i in range(len(X)):
            autres = np.r_[0:i, i + 1:len(X)]
            resultat.iloc[[i]] = _ref_encoder(X.iloc[autres], y[autres], X.iloc[[i]], lissage).to_numpy()
        return resultat
    if variante == "sans_melange":
        plis = KFold(n_splits=n_plis).split(X)
    elif variante == "stratifie":
        plis = StratifiedKFold(n_splits=n_plis, shuffle=True, random_state=graine).split(X, y)
    else:
        plis = KFold(n_splits=n_plis, shuffle=True, random_state=graine).split(X)
    moyenne_globale = float(y.mean())
    for train, test in plis:
        if variante == "moyenne_globale":   # lissage et modalités inconnues avec la moyenne de TOUT y
            pli = pd.DataFrame(index=X.index[test])
            for colonne in X.columns:
                stats = pd.Series(y[train]).groupby(X[colonne].iloc[train].to_numpy()).agg(["sum", "count"])
                encodage = (stats["sum"] + lissage * moyenne_globale) / (stats["count"] + lissage)
                pli[colonne] = X[colonne].iloc[test].map(encodage).astype(float).fillna(moyenne_globale)
        else:
            pli = _ref_encoder(X.iloc[train], y[train], X.iloc[test], lissage)
        resultat.iloc[test] = pli.to_numpy()
    return resultat


def verifier_EncodeurCibleHorsPli(Classe) -> None:
    clients = generer_clients(300, graine=5)
    X, y = clients[CATEGORIES], clients["churn"]

    # 1. Les paramètres, ceux du parent compris.
    objet = _creer(Classe, "EncodeurCibleHorsPli()")
    if not any(base.__name__ == "EncodeurCible" for base in type(objet).__mro__[1:]):
        raise AssertionError("EncodeurCibleHorsPli doit HÉRITER de votre EncodeurCible : "
                             "`class EncodeurCibleHorsPli(EncodeurCible):`.")
    _verifier_parametres(objet, {"lissage": 10.0, "n_plis": 5, "graine": 0}, "EncodeurCibleHorsPli()")
    _verifier_parametres(Classe(lissage=2.0, n_plis=3, graine=7), {"lissage": 2.0, "n_plis": 3, "graine": 7},
                         "EncodeurCibleHorsPli(lissage=2.0, n_plis=3, graine=7)")

    # 2. fit puis transform : ceux du parent (le défi 137 doit être juste).
    quoi = "EncodeurCibleHorsPli().fit(X, y).transform(X)"
    obtenu = _exiger_table(_appel(lambda: Classe().fit(X, y).transform(X), quoi), X, quoi)
    if not _identiques(obtenu, _ref_encoder(X, y, X)):
        raise AssertionError(f"{quoi} ne donne pas l'encodage du défi 137. `fit` et `transform` sont hérités de votre "
                             f"EncodeurCible : réussissez d'abord le défi 137 (`python scripts/defis.py --defi 137`).")

    # 3. fit_transform : hors pli.
    encodeur = Classe()
    quoi = "EncodeurCibleHorsPli().fit_transform(X, y)"
    obtenu = _exiger_table(_appel(lambda: encodeur.fit_transform(X, y), quoi,
                                  "Les plis donnent des POSITIONS : `X.iloc[train]`, `y[train]` (avec y en tableau "
                                  "numpy) et `resultat.iloc[test] = ...`."), X, quoi)
    _verifier_intact(X, generer_clients(300, graine=5)[CATEGORIES], quoi)
    attendu = _ref_hors_pli(X, y)
    if not _identiques(obtenu, attendu):
        pieges = [
            ("naif", "c'est exactement `fit(X, y).transform(X)` : chaque ligne est encodée avec SA propre réponse. "
                     "Redéfinissez `fit_transform` : chaque pli est encodé par un EncodeurCible appris sur les AUTRES "
                     "plis."),
            ("moyenne_globale", "la moyenne qui sert au lissage et aux villes inconnues est celle de TOUT y. Dans chaque "
                                "pli, tout vient des autres plis, moyenne comprise : `EncodeurCible(...).fit(X.iloc[train], "
                                "y[train])` s'en charge."),
            ("sans_melange", "les plis ne sont pas mélangés. Le correcteur attend exactement `KFold(n_splits=self.n_plis, "
                             "shuffle=True, random_state=self.graine)`."),
            ("stratifie", "les plis sont stratifiés. Le correcteur attend exactement `KFold(n_splits=self.n_plis, "
                          "shuffle=True, random_state=self.graine)` (pas StratifiedKFold)."),
            ("loo", "chaque ligne est encodée sans elle seule (« leave-one-out »). L'idée est bonne, mais elle fuit encore "
                    "un peu (la ligne se devine à partir de la moyenne de sa ville). Utilisez les plis de KFold."),
        ]
        for variante, conseil in pieges:
            if _identiques(obtenu, _ref_hors_pli(X, y, variante=variante)):
                raise AssertionError(f"{quoi} : {conseil}")
        _difference_lisible(obtenu, attendu, quoi, "Chaque pli de KFold(n_splits=self.n_plis, shuffle=True, "
                                                  "random_state=self.graine) est encodé par un EncodeurCible appris sur "
                                                  "les autres plis.")

    # 4. Après fit_transform, transform encode les nouveaux clients avec TOUT le train.
    nouveaux = generer_clients(40, graine=6)[CATEGORIES]
    quoi = "encodeur.fit_transform(X, y), puis encodeur.transform(<40 nouveaux clients>)"
    try:
        obtenu = encodeur.transform(nouveaux)
    except NotImplementedError:
        raise
    except NotFittedError:
        raise AssertionError(f"{quoi} lève NotFittedError : `fit_transform` doit AUSSI appeler `self.fit(X, y)`, pour que "
                             f"`transform` puisse ensuite encoder le test et les nouveaux clients.") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}.") from None
    obtenu = _exiger_table(obtenu, nouveaux, quoi)
    if not _identiques(obtenu, _ref_encoder(X, y, nouveaux)):
        raise AssertionError(f"{quoi} n'utilise pas l'encodage appris sur TOUT le train (celui du dernier pli, peut-être ?). "
                             f"Appelez `self.fit(X, y)` dans `fit_transform`, et laissez les encodeurs des plis de côté.")

    # 5. Les autres paramètres comptent.
    for parametres in ({"graine": 3}, {"n_plis": 3, "lissage": 1.0}):
        texte = ", ".join(f"{k}={v}" for k, v in parametres.items())
        quoi = f"EncodeurCibleHorsPli({texte}).fit_transform(X, y)"
        obtenu = _exiger_table(_appel(lambda: Classe(**parametres).fit_transform(X, y), quoi), X, quoi)
        _difference_lisible(obtenu, _ref_hors_pli(X, y, **parametres), quoi,
                            "Utilisez self.n_plis, self.graine et self.lissage.")
    for valeur in (1, True, 2.5):
        _leve(lambda: Classe(n_plis=valeur).fit_transform(X, y), ValueError,
              f"EncodeurCibleHorsPli(n_plis={valeur!r}).fit_transform(X, y)",
              "n_plis doit être un entier supérieur ou égal à 2 (et pas un booléen).")
    quoi = "EncodeurCibleHorsPli().fit_transform(X, <y : une liste>)"
    obtenu = _exiger_table(_appel(lambda: Classe().fit_transform(X, y.tolist()), quoi,
                                  "Convertissez y : `np.asarray(y, dtype=float)`."), X, quoi)
    _difference_lisible(obtenu, attendu, quoi)
    _verifier_clone(encodeur, "encodeur")

    # 6. Dans un Pipeline : fit appelle fit_transform (hors pli), predict appelle transform.
    X_h, y_h = _x_y(400, 7)
    colonne = pd.DataFrame({"client": X_h.index.to_numpy()}, index=X_h.index)
    quoi = "make_pipeline(EncodeurCibleHorsPli(lissage=1), LogisticRegression()).fit(<colonne client>, y)"
    modele = _appel(lambda: make_pipeline(Classe(lissage=1.0), LogisticRegression()).fit(colonne, y_h), quoi)
    reference = make_pipeline(_HorsPliReference(lissage=1.0), LogisticRegression()).fit(colonne, y_h)
    if not np.allclose(modele[-1].coef_, reference[-1].coef_):
        raise AssertionError(f"{quoi} : le modèle n'apprend pas sur l'encodage hors pli (coefficient "
                             f"{_montrer(modele[-1].coef_[0, 0])} au lieu de {_montrer(reference[-1].coef_[0, 0])}).")


class _HorsPliReference(BaseEstimator):
    def __init__(self, lissage=10.0, n_plis=5, graine=0):
        self.lissage, self.n_plis, self.graine = lissage, n_plis, graine

    def fit(self, X, y):
        self.train_, self.y_ = X.copy(), np.asarray(y, dtype=float)
        return self

    def transform(self, X):
        return _ref_encoder(self.train_, self.y_, X, self.lissage)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return _ref_hors_pli(X, y, self.lissage, self.n_plis, self.graine)


# ============================================================================ Défi 139 : ClassifieurSeuil


def _ref_seuil(estimateur, X, y, cout_fn=500.0, cout_fp=50.0, n_plis=5, graine=0, variante=None) -> float:
    y = np.asarray(y)
    classes = np.unique(y)
    if variante == "train":
        proba = clone(estimateur).fit(X, y).predict_proba(X)[:, 1]
    else:
        if variante == "plis_par_defaut":
            plis = n_plis
        elif variante == "kfold":
            plis = KFold(n_splits=n_plis, shuffle=True, random_state=graine)
        else:
            plis = StratifiedKFold(n_splits=n_plis, shuffle=True, random_state=graine)
        proba = cross_val_predict(clone(estimateur), X, y, cv=plis, method="predict_proba")[:, 1]
    reel = y == classes[1]
    grille = np.arange(1, 100) / 100
    if variante == "inverse":
        cout_fn, cout_fp = cout_fp, cout_fn
    alertes = [(proba > s) if variante == "strict" else (proba >= s) for s in grille]
    couts = np.array([cout_fn * np.sum(reel & ~a) + cout_fp * np.sum(~reel & a) for a in alertes])
    if variante == "plus_grand":
        return float(grille[len(couts) - 1 - int(np.argmin(couts[::-1]))])
    return float(grille[int(np.argmin(couts))])


def _modele_seuil():
    return make_pipeline(SimpleImputer(strategy="median"), LogisticRegression())


def _arbre():
    return make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(max_depth=6, random_state=0))


def _verifier_seuil(obtenu, attendu: float, quoi: str, estimateur, X, y, **parametres) -> None:
    if obtenu is None:
        raise AssertionError(f"Après {quoi}, `seuil_` vaut None.")
    if isinstance(obtenu, bool) or not isinstance(obtenu, numbers.Real):
        raise AssertionError(f"Après {quoi}, `seuil_` vaut {_court(obtenu)} : un nombre est attendu.")
    if math.isclose(float(obtenu), attendu, abs_tol=1e-9):
        return
    pieges = [
        ("train", "le seuil est choisi sur les probabilités que le modèle donne à ses PROPRES données d'entraînement. "
                  "Il y est trop sûr de lui (un arbre y donne 0 ou 1). Choisissez-le sur des probabilités HORS PLI : "
                  "`cross_val_predict(..., method=\"predict_proba\")`."),
        ("plis_par_defaut", "les plis ne sont pas mélangés (`cv=self.n_plis` ne mélange pas). Le correcteur attend "
                            "exactement `StratifiedKFold(n_splits=self.n_plis, shuffle=True, random_state=self.graine)`."),
        ("kfold", "les plis ne sont pas stratifiés. Le correcteur attend exactement `StratifiedKFold(n_splits=self.n_plis, "
                  "shuffle=True, random_state=self.graine)`."),
        ("inverse", "les deux coûts sont inversés : un faux négatif (client parti sans alerte) coûte cout_fn, un faux "
                    "positif (offre inutile) cout_fp."),
        ("strict", "une alerte, c'est une probabilité SUPÉRIEURE OU ÉGALE au seuil (`>=`)."),
        ("plus_grand", "en cas d'égalité de coût, gardez le PLUS PETIT seuil (`np.argmin` renvoie le premier)."),
    ]
    for variante, conseil in pieges:
        if math.isclose(float(obtenu), _ref_seuil(estimateur, X, y, variante=variante, **parametres), abs_tol=1e-9):
            raise AssertionError(f"Après {quoi}, seuil_ vaut {_montrer(obtenu)} au lieu de {_montrer(attendu)} : {conseil}")
    raise AssertionError(f"Après {quoi}, seuil_ vaut {_montrer(obtenu)} au lieu de {_montrer(attendu)} : le seuil le moins "
                         f"cher parmi 0,01 … 0,99, sur les probabilités hors pli de la classe positive.")


def verifier_ClassifieurSeuil(Classe) -> None:
    X, y = _x_y(500, 8)
    X = X[NUMERIQUES]

    # 1. Les paramètres, l'estimateur compris.
    recu = _modele_seuil()
    objet = _creer(Classe, "ClassifieurSeuil(modele)", estimateur=recu)
    _verifier_parametres(objet, {"estimateur": recu, "cout_fn": 500.0, "cout_fp": 50.0, "n_plis": 5, "graine": 0},
                         "ClassifieurSeuil(modele)")
    if "estimateur__logisticregression__C" not in objet.get_params(deep=True):
        raise AssertionError("ClassifieurSeuil(modele).get_params(deep=True) ne donne pas 'estimateur__logisticregression__C' "
                             ": l'estimateur doit être un paramètre nommé `estimateur`.")
    if not is_classifier(objet):
        raise AssertionError("scikit-learn ne reconnaît pas ClassifieurSeuil comme un classifieur (`is_classifier` vaut "
                             "False). Héritez de `ClassifierMixin`, et mettez-le À GAUCHE : `class ClassifieurSeuil(ClassifierMixin, BaseEstimator):`. "
                             "Sinon `cross_val_score` ne stratifie pas ses plis, et les scoreurs ne le reconnaissent pas.")
    _construit_sans_verifier(Classe, "ClassifieurSeuil(modele, cout_fn=-1)", estimateur=recu, cout_fn=-1)

    # 2. fit : le seuil hors pli, le modèle final sur tout le train, l'estimateur reçu intact.
    quoi = "ClassifieurSeuil(modele).fit(X, y)"
    retour = _appel(lambda: objet.fit(X, y), quoi)
    _verifier_intact(X, _x_y(500, 8)[0][NUMERIQUES], quoi)
    _verifier_fit_renvoie_self(objet, retour, quoi)
    if hasattr(recu[-1], "coef_"):
        raise AssertionError(f"{quoi} a ENTRAÎNÉ l'estimateur reçu. On n'entraîne que des COPIES (`clone(self.estimateur)`) "
                             f": l'appelant s'en sert peut-être ailleurs, et GridSearchCV le réutilise pour chaque "
                             f"combinaison de paramètres.")
    _verifier_entraine(objet, quoi)
    _verifier_seuil(getattr(objet, "seuil_", None), _ref_seuil(recu, X, y), quoi, recu, X, y)
    classes = getattr(objet, "classes_", None)
    if classes is None or list(classes) != [0, 1]:
        raise AssertionError(f"Après {quoi}, classes_ vaut {_court(classes)} au lieu de [0, 1] (`np.unique(y)`).")
    final = getattr(objet, "estimateur_", None)
    if final is None or final is recu:
        raise AssertionError(f"Après {quoi}, `estimateur_` doit être une COPIE de l'estimateur, entraînée sur tout le "
                             f"train : `self.estimateur_ = clone(self.estimateur).fit(X, y)`.")
    reference = clone(recu).fit(X, y)
    quoi = "modele.predict_proba(X)"
    proba = _appel(lambda: objet.predict_proba(X), quoi)
    if np.shape(proba) != (len(X), 2) or not np.allclose(proba, reference.predict_proba(X)):
        raise AssertionError(f"{quoi} ne donne pas les probabilités du modèle réentraîné sur TOUT le train (celui du "
                             f"dernier pli, peut-être ?). Après le choix du seuil : `clone(self.estimateur).fit(X, y)`.")
    quoi = "modele.predict(X)"
    predit = _appel(lambda: objet.predict(X), quoi)
    attendu = (reference.predict_proba(X)[:, 1] >= _ref_seuil(recu, X, y)).astype(int)
    if predit is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    predit = np.asarray(predit)
    if predit.shape != attendu.shape or not np.array_equal(predit.astype(int), attendu):
        if predit.shape == attendu.shape and np.array_equal((reference.predict(X)).astype(int), predit.astype(int)):
            raise AssertionError(f"{quoi} donne les prédictions au seuil de 0,5 (celles de l'estimateur). Utilisez "
                                 f"`self.seuil_` : `predict_proba(X)[:, 1] >= self.seuil_`.")
        raise AssertionError(f"{quoi} ne donne pas les bonnes classes : classes_[1] quand la probabilité de la classe "
                             f"positive est >= seuil_, sinon classes_[0].")

    # 3. Avant fit ; un arbre (sûr de lui sur le train) ; des étiquettes en texte.
    _verifier_pas_entraine(lambda: Classe(_modele_seuil()).predict(X), "ClassifieurSeuil(modele).predict(X)")
    arbre = _arbre()
    quoi = "ClassifieurSeuil(<un arbre de décision>).fit(X, y)"
    modele = _appel(lambda: Classe(arbre).fit(X, y), quoi)
    _verifier_seuil(getattr(modele, "seuil_", None), _ref_seuil(arbre, X, y), quoi, arbre, X, y)
    etiquettes = y.map({0: "non", 1: "oui"})
    quoi = "ClassifieurSeuil(modele).fit(X, <y en 'non' / 'oui'>).predict(X)"
    predit = _appel(lambda: Classe(_modele_seuil()).fit(X, etiquettes).predict(X), quoi)
    attendu = np.where(attendu == 1, "oui", "non")
    if predit is None or np.shape(predit) != attendu.shape or not all(a == b for a, b in zip(predit, attendu)):
        exemple = list(np.asarray(predit)[:3]) if predit is not None else None
        raise AssertionError(f"{quoi} commence par {exemple} au lieu de {list(attendu[:3])} : `predict` renvoie des "
                             f"ÉTIQUETTES, celles de y (`self.classes_[(proba >= self.seuil_).astype(int)]`), et la "
                             f"classe positive est classes_[1] (« oui »).")

    # 4. Ce qu'il faut refuser.
    for parametres in ({"cout_fn": -1}, {"cout_fp": -0.5}, {"n_plis": 1}, {"n_plis": True}):
        texte = ", ".join(f"{k}={v!r}" for k, v in parametres.items())
        _leve(lambda: Classe(_modele_seuil(), **parametres).fit(X, y), ValueError,
              f"ClassifieurSeuil(modele, {texte}).fit(X, y)",
              "Coûts positifs ou nuls, n_plis entier ≥ 2 (pas un booléen) : vérifiés dans fit.")
    for nom, cible in (("une seule classe", np.zeros(len(y), dtype=int)),
                       ("trois classes", np.arange(len(y)) % 3)):
        _leve(lambda: Classe(_modele_seuil()).fit(X, cible), ValueError, f"ClassifieurSeuil(modele).fit(X, <y : {nom}>)",
              "Il faut exactement 2 classes : `len(np.unique(y)) != 2` → ValueError.")

    # 5. Dans une recherche de paramètres, avec le scoreur du défi 136.
    quoi = ("GridSearchCV(ClassifieurSeuil(modele), {'estimateur__logisticregression__C': [0.001, 1.0]}, "
            "scoring=<coût moyen>, cv=3)")
    grille = {"estimateur__logisticregression__C": [0.001, 1.0]}
    recherche = _appel(lambda: GridSearchCV(Classe(_modele_seuil()), grille, cv=3, error_score="raise",
                                            scoring=_scoreur_reference(500, 50)).fit(X, y), quoi)
    attendu = GridSearchCV(_SeuilReference(_modele_seuil()), grille, cv=3, scoring=_scoreur_reference(500, 50)).fit(X, y)
    if not np.allclose(recherche.cv_results_["mean_test_score"], attendu.cv_results_["mean_test_score"]):
        raise AssertionError(f"{quoi} donne les scores {np.round(recherche.cv_results_['mean_test_score'], 2)} au lieu de "
                             f"{np.round(attendu.cv_results_['mean_test_score'], 2)}.")

    # 6. D'autres coûts, d'autres plis, d'autres clients.
    rng = random.Random(139)
    for graine in range(3):
        X_h, y_h = _x_y(rng.randint(150, 400), graine + 30)
        X_h = X_h[NUMERIQUES]
        parametres = {"cout_fn": rng.choice([500.0, 120, 30.0]), "cout_fp": rng.choice([50.0, 40, 100.0]),
                      "n_plis": rng.choice([2, 3, 5]), "graine": rng.randint(0, 99)}
        texte = ", ".join(f"{k}={v}" for k, v in parametres.items())
        quoi = f"ClassifieurSeuil(modele, {texte}).fit(<{len(X_h)} clients>, y)"
        estimateur = _modele_seuil()
        modele = _appel(lambda: Classe(estimateur, **parametres).fit(X_h, y_h), quoi)
        _verifier_seuil(getattr(modele, "seuil_", None), _ref_seuil(estimateur, X_h, y_h, **parametres), quoi,
                        estimateur, X_h, y_h, **parametres)


class _SeuilReference(ClassifierMixin, BaseEstimator):
    def __init__(self, estimateur, cout_fn=500.0, cout_fp=50.0, n_plis=5, graine=0):
        self.estimateur, self.cout_fn, self.cout_fp, self.n_plis, self.graine = estimateur, cout_fn, cout_fp, n_plis, graine

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        self.seuil_ = _ref_seuil(self.estimateur, X, y, self.cout_fn, self.cout_fp, self.n_plis, self.graine)
        self.estimateur_ = clone(self.estimateur).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.estimateur_.predict_proba(X)

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= self.seuil_).astype(int)]


SERIE = Serie(
    code="22",
    titre="Écrire ses propres briques scikit-learn : transformer, régler, noter, encoder, décider",
    module_eleve="defis.serie_22",
    module_corrige="defis.corrige_serie_22",
    defis=(
        Defi(134, "ImputeurMediane", "🟢", "2.5", verifier_ImputeurMediane),
        Defi(135, "RegroupeurRares", "🟢", "3.5", verifier_RegroupeurRares),
        Defi(136, "scoreur_cout", "🟡", "4.3", verifier_scoreur_cout),
        Defi(137, "EncodeurCible", "🟡", "—", verifier_EncodeurCible),
        Defi(138, "EncodeurCibleHorsPli", "🟡", "2.5", verifier_EncodeurCibleHorsPli),
        Defi(139, "ClassifieurSeuil", "🔴", "4.3", verifier_ClassifieurSeuil),
    ),
)

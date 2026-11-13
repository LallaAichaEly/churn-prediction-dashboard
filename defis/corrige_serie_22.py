"""Série 22 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_22.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la règle de scikit-learn que vous ne connaissiez pas.

Fil rouge de la série : une brique scikit-learn est un CONTRAT. Si vous le respectez, votre classe
entre dans un `Pipeline`, une validation croisée, un `GridSearchCV`, et elle y est automatiquement
protégée contre les fuites de données. Cinq règles :

1. `__init__` range ses paramètres TELS QUELS, sous LEUR nom, et ne fait rien d'autre. `get_params`,
   `set_params` et `clone` (que la validation croisée appelle à chaque pli) en dépendent ;
2. `fit` vérifie les paramètres, APPREND (sur le train seulement), range ce qu'il a appris dans des
   attributs finissant par `_` (`medianes_`) et renvoie `self` ;
3. `transform` / `predict` n'apprennent RIEN : ils appliquent ce que `fit` a appris, sans modifier
   l'entrée, et lèvent `NotFittedError` si `fit` n'a pas eu lieu (`check_is_fitted(self)`) ;
4. un méta-estimateur ne touche jamais à l'estimateur qu'on lui donne : il en entraîne des COPIES
   (`clone`) ;
5. tout ce qui est « choisi » (un encodage, un seuil) est choisi sur des données que le modèle n'a pas
   vues pendant l'apprentissage, sinon le choix est trop optimiste (validation hors pli).

Les mixins (`TransformerMixin`, `ClassifierMixin`) se placent À GAUCHE de `BaseEstimator`.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.utils.validation import check_is_fitted

from defis.briques import CATEGORIES, NUMERIQUES, clients_exemple, generer_clients  # noqa: F401

# ============================================================================ Le contrat fit / transform


class ImputeurMediane(TransformerMixin, BaseEstimator):
    # Pas de paramètre : pas besoin d'écrire `__init__` (celui de BaseEstimator suffit).

    def fit(self, X, y=None):
        # `y=None` : un transformateur non supervisé accepte quand même `y`, car le Pipeline le passe à
        # toutes les étapes. On l'ignore.
        # `X.median()` ignore les NaN (np.median ne le fait pas : une seule case vide donnerait NaN).
        medianes = X.median()
        vides = medianes.index[medianes.isna()]
        if len(vides):
            raise ValueError(f"La colonne {vides[0]!r} n'a aucune valeur : impossible d'en calculer la médiane.")
        # Ce qu'on a APPRIS finit par `_` : c'est ce que `check_is_fitted` cherche. Toutes les colonnes,
        # même celles sans trou dans le train : en production, un trou peut arriver n'importe où.
        self.medianes_ = medianes
        return self  # pour enchaîner : `ImputeurMediane().fit(X).transform(X)`, et le Pipeline le fait

    def transform(self, X):
        check_is_fitted(self)  # NotFittedError, avec un message clair, si fit n'a pas eu lieu
        attendues, recues = set(self.medianes_.index), set(X.columns)
        if attendues != recues:
            raise ValueError(f"Colonnes différentes de celles de fit : manquantes {sorted(attendues - recues)}, "
                             f"en trop {sorted(recues - attendues)}.")
        # Les médianes du TRAIN, jamais celles de X : en production, X n'a parfois qu'une ligne, et les
        # données de test ne doivent rien influencer. `fillna` renvoie une NOUVELLE table (même index).
        return X.fillna(self.medianes_)


# ============================================================================ Les paramètres et clone


class RegroupeurRares(TransformerMixin, BaseEstimator):
    def __init__(self, min_effectif=10, libelle="Autre"):
        # TELS QUELS, sous LEUR nom, et rien d'autre. Pas de vérification ici : `set_params` (que
        # GridSearchCV appelle) change l'attribut SANS repasser par `__init__`. Une vérification ici
        # serait contournée ; elle a sa place dans `fit`.
        self.min_effectif = min_effectif
        self.libelle = libelle

    def fit(self, X, y=None):
        m = self.min_effectif
        # `True` est un entier pour Python (isinstance(True, int) vaut True) : on l'écarte exprès.
        if isinstance(m, bool) or not isinstance(m, numbers.Integral) or m < 1:
            raise ValueError(f"min_effectif doit être un entier supérieur ou égal à 1 (reçu : {m!r}).")
        # Un NOUVEAU dictionnaire à chaque fit : réapprendre efface ce qu'on avait appris avant.
        # `value_counts()` ne compte pas les NaN : une case vide n'est pas une modalité.
        self.modalites_ = {}
        for colonne in X.columns:
            comptes = X[colonne].value_counts()
            self.modalites_[colonne] = set(comptes.index[comptes >= m])
        return self

    def transform(self, X):
        check_is_fitted(self)
        resultat = X.copy()
        for colonne, gardees in self.modalites_.items():
            # `isin` vaut False pour NaN : sans `| isna()`, les cases vides deviendraient « Autre », et le
            # modèle confondrait « on ne sait pas » avec « une petite ville ».
            garde = X[colonne].isin(gardees) | X[colonne].isna()
            resultat[colonne] = X[colonne].where(garde, self.libelle)
        return resultat


# ============================================================================ Un score métier


def scoreur_cout(cout_fn, cout_fp, seuil=0.5, positif=1):
    # On vérifie TOUT DE SUITE : une erreur au milieu d'une recherche de 2 heures, c'est 2 heures perdues.
    for nom, cout in (("cout_fn", cout_fn), ("cout_fp", cout_fp)):
        if isinstance(cout, bool) or not isinstance(cout, numbers.Real) or not cout >= 0:  # `not >=` : NaN
            raise ValueError(f"{nom} doit être un nombre positif ou nul (reçu : {cout!r}).")
    if isinstance(seuil, bool) or not isinstance(seuil, numbers.Real) or not 0 <= seuil <= 1:
        raise ValueError(f"seuil doit être compris entre 0 et 1 (reçu : {seuil!r}).")

    def scoreur(estimateur, X, y):
        # L'estimateur est DÉJÀ entraîné (sur les autres plis) : on ne l'entraîne surtout pas ici.
        # La colonne de `predict_proba` qui correspond à la classe positive se lit dans `classes_` (rangées
        # dans l'ordre : avec "non" / "oui", "non" est la colonne 0 ; avec "part" / "reste", "part" aussi).
        classes = list(estimateur.classes_)
        if positif not in classes:
            raise ValueError(f"La classe positive {positif!r} n'est pas dans classes_ = {classes}.")
        proba = estimateur.predict_proba(X)[:, classes.index(positif)]
        alerte = proba >= seuil
        reel = np.asarray(y) == positif
        cout = cout_fn * np.sum(reel & ~alerte) + cout_fp * np.sum(~reel & alerte)
        # Un coût MOYEN (les plis n'ont pas tous la même taille), et NÉGATIF : scikit-learn MAXIMISE
        # toujours le score. Avec un coût positif, GridSearchCV choisirait le modèle le plus cher.
        return -float(cout / len(reel))

    return scoreur


# Variante tout aussi juste, avec `make_scorer` :
#
#     def _cout_moyen(y, proba, pos_label, cout_fn, cout_fp, seuil):
#         reel, alerte = np.asarray(y) == pos_label, proba >= seuil
#         return (cout_fn * np.sum(reel & ~alerte) + cout_fp * np.sum(~reel & alerte)) / len(reel)
#
#     make_scorer(_cout_moyen, response_method="predict_proba", greater_is_better=False,
#                 pos_label=positif, cout_fn=cout_fn, cout_fp=cout_fp, seuil=seuil)
#
# `greater_is_better=False` met le signe moins, et `pos_label` choisit la bonne colonne de predict_proba.


# ============================================================================ L'encodage par la cible


def _verifier_lissage(lissage):
    if isinstance(lissage, bool) or not isinstance(lissage, numbers.Real) or not lissage >= 0:
        raise ValueError(f"lissage doit être un nombre positif ou nul (reçu : {lissage!r}).")


class EncodeurCible(TransformerMixin, BaseEstimator):
    def __init__(self, lissage=10.0):
        self.lissage = lissage

    def fit(self, X, y=None):
        _verifier_lissage(self.lissage)
        if y is None:
            raise ValueError("EncodeurCible a besoin de la cible : fit(X, y).")
        # Par POSITION, comme tout scikit-learn : la ligne i de X va avec la valeur i de y. Ainsi peu importe
        # que y soit une liste, un tableau numpy ou une Series dont l'index n'est pas celui de X.
        y = np.asarray(y, dtype=float)
        if len(y) != len(X):
            raise ValueError(f"X a {len(X)} lignes mais y en a {len(y)}.")
        # La moyenne GLOBALE : toutes les lignes comptent autant (pas la moyenne des moyennes des villes).
        self.moyenne_ = float(y.mean())
        m = self.lissage
        self.encodages_ = {}
        for colonne in X.columns:
            # `groupby` écarte les NaN : une case vide n'est pas une modalité, elle recevra la moyenne.
            stats = pd.Series(y).groupby(X[colonne].to_numpy()).agg(["sum", "count"])
            # Le lissage : une moyenne pondérée entre la modalité (n clients) et la moyenne globale (poids m).
            # Albi, 1 client parti : (1 + 10 × 0,42) / (1 + 10) ≈ 0,47 au lieu de 1,00.
            self.encodages_[colonne] = (stats["sum"] + m * self.moyenne_) / (stats["count"] + m)
        return self

    def transform(self, X):
        check_is_fitted(self)
        resultat = pd.DataFrame(index=X.index)
        for colonne, encodage in self.encodages_.items():
            # Modalité inconnue ou NaN : on ne sait rien d'elle, donc la moyenne globale (ni NaN, ni 0).
            resultat[colonne] = X[colonne].map(encodage).astype(float).fillna(self.moyenne_)
        return resultat


# ============================================================================ L'encodage hors pli


class EncodeurCibleHorsPli(EncodeurCible):
    def __init__(self, lissage=10.0, n_plis=5, graine=0):
        # Tous les paramètres dans la signature, y compris ceux du parent : get_params lit la signature.
        super().__init__(lissage=lissage)
        self.n_plis = n_plis
        self.graine = graine

    def fit_transform(self, X, y=None):
        n = self.n_plis
        if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 2:
            raise ValueError(f"n_plis doit être un entier supérieur ou égal à 2 (reçu : {n!r}).")
        # 1. L'encodage appris sur TOUT le train : c'est lui que `transform` appliquera au test et en
        #    production. `fit` vérifie aussi y et le lissage.
        self.fit(X, y)
        y = np.asarray(y, dtype=float)
        # 2. Les lignes du train, elles, sont encodées SANS leur propre cible : chaque pli est encodé par un
        #    encodeur appris sur les autres plis (moyenne globale comprise). Sinon, une ville rare « contient »
        #    la réponse de ses clients, et le modèle l'apprend par cœur.
        resultat = pd.DataFrame(np.nan, index=X.index, columns=list(X.columns), dtype=float)
        plis = KFold(n_splits=n, shuffle=True, random_state=self.graine)
        for train, test in plis.split(X):
            encodeur = EncodeurCible(lissage=self.lissage).fit(X.iloc[train], y[train])
            # Des POSITIONS : `iloc`, pas `loc` (l'index est fait d'identifiants de clients).
            resultat.iloc[test] = encodeur.transform(X.iloc[test]).to_numpy()
        return resultat


# ============================================================================ Un méta-estimateur


class ClassifieurSeuil(ClassifierMixin, BaseEstimator):
    def __init__(self, estimateur, cout_fn=500.0, cout_fp=50.0, n_plis=5, graine=0):
        # L'estimateur est un PARAMÈTRE comme les autres : get_params(deep=True) donne alors aussi
        # « estimateur__C », que GridSearchCV peut régler.
        self.estimateur = estimateur
        self.cout_fn = cout_fn
        self.cout_fp = cout_fp
        self.n_plis = n_plis
        self.graine = graine

    def fit(self, X, y):
        for nom, cout in (("cout_fn", self.cout_fn), ("cout_fp", self.cout_fp)):
            if isinstance(cout, bool) or not isinstance(cout, numbers.Real) or not cout >= 0:
                raise ValueError(f"{nom} doit être un nombre positif ou nul (reçu : {cout!r}).")
        n = self.n_plis
        if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 2:
            raise ValueError(f"n_plis doit être un entier supérieur ou égal à 2 (reçu : {n!r}).")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Il faut exactement 2 classes dans y (reçu : {list(self.classes_)}).")
        # 1. Des probabilités HORS PLI : chaque client est noté par un modèle qui ne l'a pas vu. Choisir le
        #    seuil sur les probabilités du train serait trop optimiste (un arbre profond y est sûr de tout).
        #    `cross_val_predict` clone l'estimateur pour chaque pli.
        plis = StratifiedKFold(n_splits=n, shuffle=True, random_state=self.graine)
        proba = cross_val_predict(clone(self.estimateur), X, y, cv=plis, method="predict_proba")[:, 1]
        # 2. Le seuil le moins cher parmi 0,01, 0,02 … 0,99 (le plus petit en cas d'égalité). C'est ce que
        #    fait `churn.model.best_threshold_by_cost` : vous pouvez réutiliser le code du projet.
        reel = y == self.classes_[1]
        grille = np.arange(1, 100) / 100
        couts = [self.cout_fn * np.sum(reel & (proba < s)) + self.cout_fp * np.sum(~reel & (proba >= s))
                 for s in grille]
        self.seuil_ = float(grille[int(np.argmin(couts))])
        # 3. Le modèle final, réentraîné sur TOUT le train. Une COPIE : l'objet reçu reste intact (l'appelant
        #    s'en sert peut-être ailleurs, et GridSearchCV le réutilise pour chaque combinaison).
        self.estimateur_ = clone(self.estimateur).fit(X, y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimateur_.predict_proba(X)

    def predict(self, X):
        # `predict_proba` d'abord : il vérifie que fit a eu lieu (`self.classes_` n'existe pas encore avant).
        positif = self.predict_proba(X)[:, 1] >= self.seuil_
        # Des ÉTIQUETTES (celles de y : "oui" / "non"…), pas des 0 / 1.
        return self.classes_[positif.astype(int)]


# ============================================================================ Tests (à la pytest)


def _x_y(n=300, graine=0):
    clients = generer_clients(n, graine=graine)
    return clients.drop(columns="churn"), clients["churn"]


def test_imputeur_utilise_les_medianes_du_train():
    train = pd.DataFrame({"a": [1.0, 2.0, 100.0, np.nan]})
    test = pd.DataFrame({"a": [np.nan, 50.0]}, index=[7, 8])
    imputeur = ImputeurMediane().fit(train)
    assert imputeur.transform(test)["a"].tolist() == [2.0, 50.0]
    assert test["a"].isna().iloc[0]                       # l'entrée est intacte


def test_imputeur_refuse_avant_fit():
    import pytest
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        ImputeurMediane().transform(pd.DataFrame({"a": [1.0]}))


def test_regroupeur_garde_les_nan_et_suit_set_params():
    X = pd.DataFrame({"v": ["a", "a", "b", np.nan]})
    regroupeur = clone(RegroupeurRares(min_effectif=2)).fit(X)
    assert regroupeur.transform(X)["v"].tolist()[:3] == ["a", "a", "Autre"]
    assert pd.isna(regroupeur.transform(X)["v"].iloc[3])
    assert regroupeur.set_params(min_effectif=3).fit(X).transform(X)["v"].tolist()[:2] == ["Autre", "Autre"]


def test_scoreur_cout_est_negatif_et_moyen():
    from sklearn.linear_model import LogisticRegression

    X, y = _x_y()
    modele = LogisticRegression(max_iter=1000).fit(X[["appels"]], y)
    score = scoreur_cout(500, 50, seuil=0.3)(modele, X[["appels"]], y)
    assert -500 <= score <= 0


def test_encodeur_cible_lisse_et_gere_l_inconnu():
    X = pd.DataFrame({"v": ["a", "a", "b"]})
    encodeur = EncodeurCible(lissage=1).fit(X, [1, 1, 0])
    moyenne = 2 / 3
    assert np.isclose(encodeur.transform(pd.DataFrame({"v": ["b"]}))["v"].iloc[0], (0 + moyenne) / 2)
    assert encodeur.transform(pd.DataFrame({"v": ["z"]}))["v"].iloc[0] == moyenne


def test_hors_pli_differe_de_fit_puis_transform():
    X, y = _x_y()
    encodeur = EncodeurCibleHorsPli(n_plis=5)
    hors_pli = encodeur.fit_transform(X[["ville"]], y)
    assert not np.allclose(hors_pli, EncodeurCible().fit(X[["ville"]], y).transform(X[["ville"]]))
    pd.testing.assert_frame_equal(encodeur.transform(X[["ville"]]),
                                  EncodeurCible().fit(X[["ville"]], y).transform(X[["ville"]]))


def test_classifieur_seuil_ne_touche_pas_a_l_estimateur_recu():
    from sklearn.linear_model import LogisticRegression

    X, y = _x_y()
    recu = LogisticRegression(max_iter=1000)
    modele = ClassifieurSeuil(recu).fit(X[["appels", "anciennete"]].fillna(0), y)
    assert not hasattr(recu, "coef_") and 0.01 <= modele.seuil_ <= 0.99
    assert set(modele.predict(X[["appels", "anciennete"]].fillna(0))) <= {0, 1}

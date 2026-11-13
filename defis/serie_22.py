"""Série 22 — Écrire ses propres briques scikit-learn : transformer, régler, noter, encoder, décider.

🧰 VOUS VOULEZ DÉVELOPPER ? UN JOUR, LA BRIQUE DONT VOUS AVEZ BESOIN N'EXISTERA PAS DANS SCIKIT-LEARN.

Le modèle de churn du projet tourne dans un `Pipeline` (voir `src/churn/model.py`). L'équipe veut
l'améliorer, et chaque idée demande une brique qu'il faut écrire soi-même :

- compléter les trous avec la médiane du TRAIN (celle du test, ce serait tricher) ;
- regrouper les 400 petites villes en « Autre », avec un seuil que GridSearchCV pourra régler ;
- noter chaque modèle en EUROS : un client perdu coûte 500 €, une offre inutile 50 € ;
- remplacer la ville par son taux de churn (« encodage par la cible »), sans que le modèle apprenne la
  réponse par cœur ;
- choisir le seuil d'alerte qui coûte le moins cher, sans le choisir sur les données d'entraînement.

Écrite au hasard, une telle brique « marche » dans un notebook… puis plante dans `cross_val_score`, ignore
le paramètre que GridSearchCV croit régler, ou fuit la cible et promet des résultats que la production ne
verra jamais. Écrite selon le CONTRAT de scikit-learn, elle entre partout : Pipeline, validation croisée,
recherche de paramètres, sauvegarde avec joblib.

    134 le contrat fit / transform (un imputeur)        135 les paramètres, get_params, set_params et clone
    136 un score métier, en euros, pour GridSearchCV    137 l'encodage par la cible, avec lissage
    138 l'encodage hors pli (fit_transform ≠ fit puis transform)
    139 un méta-estimateur qui choisit son seuil sans tricher

Prérequis : la série 00, la série 01 (fuite de données, validation croisée, coût métier : défis 3, 8 et 9)
et une classe (défi 48). Rien à installer : tout est dans scikit-learn et pandas.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- une brique scikit-learn est une CLASSE qui hérite de `BaseEstimator` (qui fournit `get_params`,
  `set_params`, et l'affichage) et d'un mixin : `TransformerMixin` (qui fournit `fit_transform`) ou
  `ClassifierMixin` (qui fournit `score`, et dit à scikit-learn « je suis un classifieur »). Les mixins se
  placent À GAUCHE : `class Imputeur(TransformerMixin, BaseEstimator):` ;
- `__init__` range ses paramètres TELS QUELS, sous LEUR nom (`self.lissage = lissage`), et ne fait RIEN
  d'autre : ni vérification, ni calcul. `get_params()` relit la signature de `__init__` et va chercher
  l'attribut du même nom ; `clone(objet)` fabrique un objet neuf avec les mêmes paramètres (la validation
  croisée le fait à chaque pli) ; `set_params(lissage=5)` change l'attribut SANS repasser par `__init__` ;
- `fit(X, y=None)` vérifie les paramètres, APPREND sur X (le train), range ce qu'il a appris dans des
  attributs dont le nom FINIT PAR `_` (`self.medianes_`) et renvoie `self` ;
- `transform(X)` (ou `predict(X)`) n'apprend RIEN : il applique ce que `fit` a appris. Il ne modifie pas
  X, et commence par `check_is_fitted(self)`, qui lève `NotFittedError` si aucun attribut en `_` n'existe ;
- un `Pipeline` appelle `fit_transform(X_train, y_train)` sur chaque étape, puis `transform(X_test)` :
  c'est ce qui garantit que le test n'influence jamais ce qui est appris.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_22 import *
    >>> clients = clients_exemple()                      # 12 clients, un piège par ligne (voir sa docstring)
    >>> StandardScaler().fit(clients[NUMERIQUES]).mean_  # une brique de scikit-learn : l'attribut appris finit par _
    array([19.6       , 80.80909091,  2.08333333])
    >>> StandardScaler().get_params()                    # les paramètres de __init__, relus par get_params
    {'copy': True, 'with_mean': True, 'with_std': True}

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Le défi 134 n'a pas de `__init__` : sans
   paramètre, celui de `BaseEstimator` suffit.
2. Lancez `python scripts/defis.py --defi 134` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_22.py`.

La table d'exemple et le générateur (`clients_exemple`, `generer_clients`, `NUMERIQUES`, `CATEGORIES`)
sont dans `defis/briques.py`, déjà importés ci-dessous, ainsi que tout ce dont vous aurez besoin dans
scikit-learn. Chaque appel renvoie une NOUVELLE table : vous pouvez la modifier sans rien abîmer.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import numbers  # noqa: F401 — utile pour vérifier qu'un paramètre est un nombre (numbers.Integral, numbers.Real)

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone  # noqa: F401
from sklearn.impute import SimpleImputer  # noqa: F401 — pour essayer vos briques dans un Pipeline
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.model_selection import (  # noqa: F401
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401
from sklearn.utils.validation import check_is_fitted  # noqa: F401

from defis.briques import CATEGORIES, NUMERIQUES, clients_exemple, generer_clients  # noqa: F401

# ============================================================================ Le contrat fit / transform


class ImputeurMediane(TransformerMixin, BaseEstimator):
    """Défi 134 🟢 — Compléter les trous avec la médiane… du TRAIN.

    Le modèle refuse les NaN. L'ancien script faisait `X.fillna(X.median())` juste avant de prédire :
    en production, X n'a souvent qu'UNE ligne, et sa « médiane » est la valeur du client lui-même (ou NaN).
    Il faut apprendre les médianes une fois, sur le train, et les réutiliser ensuite.

        >>> clients = clients_exemple()
        >>> imputeur = ImputeurMediane().fit(clients[NUMERIQUES])
        >>> imputeur.medianes_
        anciennete    15.0
        facture       60.0
        appels         1.5
        dtype: float64
        >>> imputeur.transform(clients[NUMERIQUES]).loc[["C03", "C04", "C09"]]
                anciennete  facture  appels
        client
        C03           15.0     55.0       1
        C04           12.0     60.0       2
        C09           15.0     30.0       0

    Les règles (X est un DataFrame de colonnes numériques, avec des NaN) :

    - `fit(X, y=None)` calcule la MÉDIANE de chaque colonne de X, en ignorant les NaN, la range dans
      `self.medianes_` et renvoie `self`. Toutes les colonnes, même celles qui n'ont aucun trou : en
      production, un trou peut arriver n'importe où. Une colonne sans AUCUNE valeur → `ValueError` ;
    - `transform(X)` renvoie un NOUVEAU DataFrame, même index, mêmes colonnes, où chaque NaN est remplacé
      par la médiane apprise pendant `fit` (jamais celle de X). Les autres valeurs ne changent pas, et X
      n'est pas modifié ;
    - `transform` avant `fit` → `NotFittedError` (c'est ce que lève `check_is_fitted(self)`) ;
    - si X n'a pas les mêmes colonnes que pendant `fit` (une en moins ou une en plus) → `ValueError` ;
    - `fit_transform(X)` existe (cadeau de `TransformerMixin`), et la classe s'utilise dans un Pipeline :
      `make_pipeline(ImputeurMediane(), LogisticRegression())`.

    Pourquoi la médiane ? C10 paie 300 € : la moyenne des factures (80,81 €) en est tirée vers le haut,
    pas la médiane (60 €). Le correcteur vérifie que vous n'avez pas pris la moyenne.

    Ce que vous apprenez : le contrat fit / transform, l'état appris (attributs en `_`), `return self`,
    `check_is_fitted`, et pourquoi c'est ce contrat qui empêche la fuite de données dans un Pipeline.

    Indice : `X.median()` ignore les NaN et renvoie une Series (colonne → médiane), et `X.fillna(serie)`
    remplit chaque colonne avec SA valeur, dans une nouvelle table. Comparez `set(X.columns)` à
    `set(self.medianes_.index)`.
    """

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


# ============================================================================ Les paramètres et clone


class RegroupeurRares(TransformerMixin, BaseEstimator):
    """Défi 135 🟢 — Regrouper les modalités rares, avec un seuil que GridSearchCV peut régler.

    La colonne `ville` compte 400 villes, dont 300 n'ont que 1 ou 2 clients. Le one-hot en fait 400
    colonnes, presque toutes vides, et le modèle apprend du bruit. On regroupe donc les villes rares en
    « Autre ». Mais à partir de combien de clients une ville est-elle « fréquente » ? 5 ? 20 ? 50 ? On veut
    laisser `GridSearchCV` choisir, donc le seuil doit être un vrai PARAMÈTRE scikit-learn.

        >>> clients = clients_exemple()
        >>> regroupeur = RegroupeurRares(min_effectif=3).fit(clients[CATEGORIES])
        >>> regroupeur.transform(clients[CATEGORIES])["ville"].tolist()
        ['Paris', 'Paris', 'Lyon', 'Autre', 'Paris', 'Autre', 'Lyon', 'Paris', 'Autre', 'Paris', 'Lyon', nan]
        >>> regroupeur.get_params()
        {'libelle': 'Autre', 'min_effectif': 3}
        >>> clone(regroupeur).set_params(min_effectif=4).fit(clients[CATEGORIES]).transform(
        ...     clients[CATEGORIES])["ville"].tolist()
        ['Paris', 'Paris', 'Autre', 'Autre', 'Paris', 'Autre', 'Autre', 'Paris', 'Autre', 'Paris', 'Autre', nan]

    Les règles :

    - `__init__(self, min_effectif=10, libelle="Autre")` range les deux paramètres TELS QUELS, sous LEUR
      nom, et ne fait RIEN d'autre (pas de vérification, pas de conversion, pas d'autre attribut). Ainsi
      `get_params()`, `set_params(...)` et `clone(...)` marchent sans que vous ayez à les écrire ;
    - `fit(X, y=None)` vérifie `min_effectif` : un entier (`int` ou entier numpy, pas un booléen, pas 2.5,
      pas "10") supérieur ou égal à 1, sinon `ValueError`. C'est ICI qu'on vérifie, pas dans `__init__` :
      `set_params(min_effectif=0)` ne repasse pas par `__init__`. Puis, pour CHAQUE colonne de X, il
      retient les modalités qui apparaissent au moins `min_effectif` fois dans X (le train), et renvoie
      `self`. Un nouveau `fit` oublie tout ce que le précédent avait appris ;
    - `transform(X)` renvoie un NOUVEAU DataFrame (même index, mêmes colonnes) où toute valeur qui n'est
      pas une modalité retenue pour SA colonne devient `libelle`, y compris une ville jamais vue pendant
      `fit`. Une case vide (NaN) RESTE vide : « on ne sait pas » n'est pas « une petite ville ». X n'est pas
      modifié ; `transform` avant `fit` → `NotFittedError`.

    Ce que vous apprenez : ce qu'est un paramètre pour scikit-learn, comment `get_params`, `set_params` et
    `clone` s'en servent, et pourquoi la validation croisée ne marche qu'avec des objets « clonables ».

    Indice : `X[colonne].value_counts()` (qui ignore les NaN) donne les effectifs ;
    `X[colonne].isin(gardees) | X[colonne].isna()` dit quoi garder ; `serie.where(garde, self.libelle)`
    remplace le reste. `isinstance(m, bool)` d'abord : `isinstance(True, numbers.Integral)` vaut True.
    """

    def __init__(self, min_effectif=10, libelle="Autre"):
        raise NotImplementedError

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


# ============================================================================ Un score métier


def scoreur_cout(cout_fn, cout_fp, seuil=0.5, positif=1):
    """Défi 136 🟡 — Noter un modèle en euros, pour `cross_val_score` et `GridSearchCV`.

    Un client perdu sans avoir été prévenu (faux négatif) coûte `cout_fn` €, une offre envoyée à un client
    qui serait resté (faux positif) coûte `cout_fp` €. L'AUC ne dit rien de ça. On veut comparer les modèles
    sur ce qu'ils coûtent, avec les outils de scikit-learn. Ils acceptent un « scoreur » : une fonction
    `scoreur(estimateur, X, y)` qui reçoit un modèle DÉJÀ entraîné et des données de test, et renvoie un
    nombre. Plus il est grand, mieux c'est : scikit-learn MAXIMISE toujours.

        >>> clients = generer_clients(1000)
        >>> X, y = clients[NUMERIQUES], clients["churn"]
        >>> modele = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression())
        >>> cross_val_score(modele, X, y, cv=5, scoring=scoreur_cout(500, 50)).round(1)
        array([-127. , -135. , -112.8, -119.2, -114.2])
        >>> cross_val_score(modele, X, y, cv=5, scoring=scoreur_cout(500, 50, seuil=0.2)).round(1)
        array([-29.5, -30. , -26.8, -41.2, -41.5])

    Avec un seuil à 0,2, on dépense plus en offres, mais le coût total est 4 fois plus faible : environ 30 €
    par client au lieu de 120.

    `scoreur_cout(cout_fn, cout_fp, seuil=0.5, positif=1)` renvoie une FONCTION `scoreur(estimateur, X, y)` :

    - une alerte est levée pour un client si la probabilité de la classe `positif` (le churn) donnée par
      `estimateur.predict_proba(X)` est SUPÉRIEURE OU ÉGALE à `seuil` ;
    - le scoreur renvoie MOINS le coût MOYEN par client (un `float`) : (cout_fn × nombre de faux négatifs
      + cout_fp × nombre de faux positifs) / nombre de clients, avec un signe moins. Moyen, car les plis
      n'ont pas tous la même taille ; négatif, car scikit-learn garde le score le plus GRAND ;
    - la colonne de `predict_proba` qui correspond à `positif` se lit dans `estimateur.classes_` (les
      classes, rangées dans l'ordre). Avec des étiquettes "non" / "oui", "oui" est la colonne 1 ; avec
      "part" / "reste", "part" est la colonne 0 ! Si `positif` n'est pas dans `classes_` → `ValueError` ;
    - le scoreur n'entraîne JAMAIS l'estimateur (il l'est déjà, sur les autres plis), et ne modifie ni X
      ni y. `y` peut être une liste, un tableau numpy ou une Series ;
    - les paramètres sont vérifiés tout de suite, à l'appel de `scoreur_cout` : un coût négatif, ou un
      `seuil` hors de [0, 1] → `ValueError`. Mieux vaut l'apprendre avant une recherche de 2 heures.

    Ce que vous apprenez : le protocole des scoreurs de scikit-learn, une fonction qui fabrique une
    fonction (une « fermeture »), `classes_` et l'ordre des colonnes de `predict_proba`, et pourquoi un
    coût se donne avec un signe moins.

    Indice : vérifiez les paramètres, puis `def scoreur(estimateur, X, y):` à l'intérieur de
    `scoreur_cout`, et `return scoreur`. Dedans : `colonne = list(estimateur.classes_).index(positif)`,
    `alerte = estimateur.predict_proba(X)[:, colonne] >= seuil`, `reel = np.asarray(y) == positif`, puis
    `np.sum(reel & ~alerte)` (faux négatifs) et `np.sum(~reel & alerte)` (faux positifs).
    """
    raise NotImplementedError


# ============================================================================ L'encodage par la cible


class EncodeurCible(TransformerMixin, BaseEstimator):
    """Défi 137 🟡 — Remplacer une catégorie par son taux de churn, sans croire les petits nombres.

    Le one-hot de 400 villes fait 400 colonnes. L'encodage par la cible (« target encoding ») n'en fait
    qu'UNE : chaque ville est remplacée par le taux de churn de ses clients. Mais Albi n'a qu'un client, et
    il est parti : un taux de 100 % sur UN client ne veut rien dire. On le « lisse » vers la moyenne :

        encodage(ville) = (somme des churns de la ville + lissage × moyenne) / (clients de la ville + lissage)

    Avec `lissage=10`, une ville de 1 client vaut presque la moyenne ; une ville de 1 000 clients vaut
    presque son propre taux. Avec `lissage=0`, c'est le taux brut.

        >>> encodeur = EncodeurCible().fit(clients_exemple()[["ville"]], clients_exemple()["churn"])
        >>> encodeur.moyenne_
        0.4166666666666667
        >>> nouveaux = pd.DataFrame({"ville": ["Albi", "Paris", "Quimper", np.nan]}, index=["N1", "N2", "N3", "N4"])
        >>> encodeur.transform(nouveaux).round(2)
            ville
        N1   0.47
        N2   0.41
        N3   0.42
        N4   0.42

    Albi : (1 + 10 × 0,4167) / (1 + 10) = 0,47 au lieu de 1,00. Quimper (jamais vue) et la case vide
    reçoivent la moyenne : on ne sait rien d'elles.

    Les règles :

    - `__init__(self, lissage=10.0)` range le paramètre tel quel (comme au défi 135) ;
    - `fit(X, y)` : `lissage` doit être un nombre positif ou nul (pas un booléen), sinon `ValueError` ;
      `y` (0 ou 1 pour chaque ligne de X) est OBLIGATOIRE : `y=None`, ou une longueur différente de celle
      de X → `ValueError`. `y` peut être une liste, un tableau numpy ou une Series, et va avec X PAR
      POSITION (la 1re valeur de y avec la 1re ligne de X), même si son index n'est pas celui de X ;
    - `fit` range dans `self.moyenne_` la moyenne de TOUT y (toutes les lignes comptent autant : ce n'est
      pas la moyenne des taux des villes), et, pour CHAQUE colonne de X, l'encodage lissé de chacune de
      ses modalités. Les cases vides (NaN) ne forment pas une modalité. Renvoie `self` ;
    - `transform(X)` renvoie un NOUVEAU DataFrame (même index, mêmes colonnes) de nombres décimaux : chaque
      valeur remplacée par l'encodage de sa modalité dans SA colonne ; une modalité jamais vue, ou NaN,
      reçoit `moyenne_` (ni NaN, ni 0). X n'est pas modifié ; avant `fit` → `NotFittedError`.

    Ce que vous apprenez : un transformateur SUPERVISÉ (`fit` a besoin de `y`), le lissage d'une moyenne
    calculée sur peu de cas, et l'alignement par position, la convention de scikit-learn.

    Indice : `y = np.asarray(y, dtype=float)`. Pour une colonne :
    `stats = pd.Series(y).groupby(X[colonne].to_numpy()).agg(["sum", "count"])` (le `.to_numpy()` aligne par
    position, et `groupby` écarte les NaN), puis `(stats["sum"] + m * moyenne) / (stats["count"] + m)`. Dans
    `transform` : `X[colonne].map(encodage).astype(float).fillna(self.moyenne_)`.
    """

    def __init__(self, lissage=10.0):
        raise NotImplementedError

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


# ============================================================================ L'encodage hors pli


class EncodeurCibleHorsPli(EncodeurCible):
    """Défi 138 🟡 — L'encodage par la cible, sans que chaque client « voie » sa propre réponse.

    Même lissé, l'encodage du défi 137 FUIT quand on l'applique aux lignes qui ont servi à le calculer : le
    taux de churn d'une petite ville CONTIENT la réponse de ses clients. Poussé à l'extrême, avec une colonne
    `client` (une modalité par ligne), `fit(X, y).transform(X)` donne à chaque client… son propre churn. Le
    modèle croit avoir trouvé la variable parfaite (AUC de 1,00 sur le train), et ne vaut rien sur de
    nouveaux clients (0,50). Sur une vraie colonne `ville`, l'effet est plus discret, mais il est là : le
    modèle fait trop confiance aux petites villes.

    La parade, celle du `TargetEncoder` de scikit-learn : PENDANT L'ENTRAÎNEMENT, chaque ligne est encodée
    par un encodeur appris SANS elle (validation « hors pli », comme `cross_val_predict`). Les nouveaux
    clients, eux, sont encodés avec ce qu'on a appris sur tout le train. Donc, exprès :
    `fit_transform(X, y)` ne donne PAS la même chose que `fit(X, y).transform(X)`.

        >>> clients = generer_clients(1000)
        >>> X, y = clients[["ville"]], clients["churn"]
        >>> encodeur = EncodeurCibleHorsPli()
        >>> pd.concat([X, EncodeurCible().fit(X, y).transform(X).add_suffix(" (fuite)"),
        ...            encodeur.fit_transform(X, y).add_suffix(" (hors pli)")], axis=1).head(3).round(3)
               ville  ville (fuite)  ville (hors pli)
        client
        G0001  Paris          0.287             0.292
        G0002  Paris          0.287             0.299
        G0003   Lyon          0.359             0.337

    Deux Parisiens n'ont pas le même encodage hors pli : ils ne sont pas dans le même pli.

    Les règles :

    - la classe HÉRITE de votre `EncodeurCible` (défi 137, qui doit d'abord être juste) : `fit` et
      `transform` ne changent pas ;
    - `__init__(self, lissage=10.0, n_plis=5, graine=0)` range les TROIS paramètres (`get_params()` doit
      les donner tous, `lissage` compris : `super().__init__(lissage=lissage)` s'en charge) ;
    - `fit_transform(X, y)` : `n_plis` doit être un entier supérieur ou égal à 2 (pas un booléen), sinon
      `ValueError`. Puis :
        1. `self.fit(X, y)` : l'encodage de TOUT le train, que `transform` appliquera ensuite au test et
           en production ;
        2. on découpe les lignes avec `KFold(n_splits=self.n_plis, shuffle=True, random_state=self.graine)`
           (exactement celui-là : le correcteur refait les mêmes plis) ;
        3. pour chaque pli, un `EncodeurCible(lissage=self.lissage)` appris sur les AUTRES plis (moyenne
           globale comprise) encode les lignes du pli ;
        4. on renvoie un DataFrame : même index et mêmes colonnes que X, chaque ligne à SA place ;
    - X n'est pas modifié. Dans un Pipeline, le `fit` du Pipeline appelle `fit_transform` (hors pli), et
      `predict` appelle `transform` (tout le train) : c'est exactement ce qu'on veut.

    Ce que vous apprenez : l'héritage d'une brique (`super().__init__`), la différence entre
    `fit_transform` et `fit` puis `transform`, et le « cross-fitting », qui empêche une variable de
    contenir la réponse.

    Indice : `for train, test in KFold(...).split(X):`, puis
    `EncodeurCible(lissage=self.lissage).fit(X.iloc[train], y[train])` (avec `y = np.asarray(y, dtype=float)`
    : `y[train]` prend des POSITIONS) et `resultat.iloc[test] = encodeur.transform(X.iloc[test]).to_numpy()`.
    Créez `resultat` avant la boucle : `pd.DataFrame(np.nan, index=X.index, columns=list(X.columns))`.
    """

    def __init__(self, lissage=10.0, n_plis=5, graine=0):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        raise NotImplementedError


# ============================================================================ Un méta-estimateur


class ClassifieurSeuil(ClassifierMixin, BaseEstimator):
    """Défi 139 🔴 — Un classifieur qui choisit lui-même son seuil d'alerte, sans tricher.

    Le seuil de 0,5 de `predict` n'a rien à voir avec le métier : avec 500 € par client perdu et 50 € par
    offre inutile, il vaut mieux alerter dès 10 % de risque. On veut donc un modèle qui, pendant `fit`,
    CHOISIT le seuil le moins cher, puis s'en sert dans `predict`. Un « méta-estimateur » : il reçoit un
    autre estimateur (un Pipeline, par exemple) et l'enveloppe. `GridSearchCV` sait alors régler en même
    temps l'estimateur intérieur (`estimateur__C`) et choisir le seuil.

    Le piège : choisir le seuil sur les probabilités que le modèle donne à ses PROPRES données
    d'entraînement. Un arbre de décision y est sûr de tout (0 ou 1) : n'importe quel seuil semble parfait,
    et on garde 0,01. Sur de nouveaux clients, c'est une catastrophe. Le seuil se choisit sur des
    probabilités HORS PLI.

        >>> clients = generer_clients(1000)
        >>> X, y = clients[NUMERIQUES], clients["churn"].map({0: "non", 1: "oui"})
        >>> modele = ClassifieurSeuil(make_pipeline(SimpleImputer(strategy="median"), LogisticRegression()))
        >>> modele.fit(X, y).seuil_
        0.1
        >>> modele.predict(X.head(3))
        array(['oui', 'oui', 'oui'], dtype=object)

    Les règles :

    - `__init__(self, estimateur, cout_fn=500.0, cout_fp=50.0, n_plis=5, graine=0)` range les cinq
      paramètres tels quels (l'estimateur aussi : `get_params(deep=True)` donne alors `estimateur__...`) ;
    - `fit(X, y)` vérifie les paramètres (`cout_fn` et `cout_fp` positifs ou nuls, `n_plis` entier ≥ 2,
      sinon `ValueError`) et `y` : exactement deux classes, sinon `ValueError`. Puis :
        1. `self.classes_` : les deux classes, dans l'ordre (`np.unique(y)`). La classe POSITIVE (le churn)
           est la seconde, `classes_[1]` ("oui", ou 1) ;
        2. des probabilités HORS PLI de la classe positive, pour chaque ligne du train :
           `cross_val_predict` avec `StratifiedKFold(n_splits=self.n_plis, shuffle=True,
           random_state=self.graine)` et `method="predict_proba"` ;
        3. `self.seuil_` : parmi 0,01, 0,02, …, 0,99 (`np.arange(1, 100) / 100`), le seuil dont le coût
           total cout_fn × FN + cout_fp × FP est le plus petit (alerte si probabilité >= seuil ; en cas
           d'égalité, le plus petit seuil) ;
        4. `self.estimateur_` : une COPIE de l'estimateur (`clone`), entraînée sur TOUT le train ;
      et renvoie `self`. L'estimateur reçu n'est JAMAIS entraîné ni modifié : on n'entraîne que des copies ;
    - `predict_proba(X)` : celles de `estimateur_` ; `predict(X)` : `classes_[1]` si la probabilité de la
      classe positive est >= `seuil_`, sinon `classes_[0]` (des ÉTIQUETTES, comme y : "oui" / "non", pas
      True / False). Avant `fit` → `NotFittedError` ;
    - `ClassifierMixin` À GAUCHE de `BaseEstimator` : sinon, depuis scikit-learn 1.6, il ne sait pas que
      c'est un classifieur (`is_classifier` vaut False), et `cross_val_score` ne stratifie plus ses plis.

    Ce que vous apprenez : écrire un méta-estimateur (`clone`, paramètres imbriqués `estimateur__C`), les
    probabilités hors pli (`cross_val_predict`), et que TOUT ce qu'on règle sur les données (un seuil comme
    un coefficient) doit l'être sans regarder les lignes qu'on va juger.

    Indice : `proba = cross_val_predict(clone(self.estimateur), X, y, cv=..., method="predict_proba")[:, 1]`,
    `reel = y == self.classes_[1]`, puis une boucle (ou une liste) sur les seuils et `np.argmin`. Le projet
    a déjà cette recherche : `churn.model.best_threshold_by_cost(reel.astype(int), proba, cout_fn, cout_fp)`.
    Pour `predict` : `self.classes_[(proba >= self.seuil_).astype(int)]`.
    """

    def __init__(self, estimateur, cout_fn=500.0, cout_fp=50.0, n_plis=5, graine=0):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

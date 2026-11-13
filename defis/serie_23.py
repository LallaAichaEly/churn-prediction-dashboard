"""Série 23 — Ouvrir la boîte noire : la régression logistique et l'arbre de décision recodés avec NumPy.

🔓 VOUS VOULEZ DÉVELOPPER ? UN JOUR, `fit` NE MARCHERA PAS, ET IL FAUDRA SAVOIR CE QU'IL Y A DEDANS.

Le projet entraîne une régression logistique et un modèle à base d'arbres (voir `src/churn/model.py`) en
une ligne : `modele.fit(X, y)`. Tant que tout va bien, c'est parfait. Mais le jour où :

- l'entraînement renvoie des coefficients NaN (« overflow encountered in exp ») ;
- la perte reste bloquée, ou le modèle dit « 100 % de chances de partir » à un client qui reste ;
- un recruteur vous demande : « codez-moi une régression logistique sans scikit-learn » ;

il faut savoir ce que fait `fit`. Cette série le recode, en NumPy, morceau par morceau :

    140 la sigmoïde, qui ne déborde jamais          141 la perte logistique, calculée sans log(0)
    142 le gradient, vérifié par différences finies  143 la descente de gradient (mêmes coefficients que scikit-learn)
    144 la meilleure coupure d'une colonne (Gini, en O(n log n))
    145 l'arbre de décision, récursif (le même arbre que scikit-learn)

Prérequis : la série 00, un peu de NumPy (série 01 ou 07), et pour 143 et 145 une classe scikit-learn
(série 22, défi 134 au moins). Pas besoin d'aimer les maths : chaque formule est donnée et expliquée.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- la RÉGRESSION LOGISTIQUE calcule un SCORE z = X @ w + b (une somme pondérée des colonnes, plus une
  constante b), puis le transforme en probabilité avec la SIGMOÏDE : p = 1 / (1 + exp(-z)). Un score de
  0 donne 0,5 ; un score très positif, presque 1 ; très négatif, presque 0 ;
- « entraîner », c'est chercher les w et b qui rendent la PERTE LOGISTIQUE (« log-loss ») la plus petite
  possible : en moyenne sur les clients, -log(p) pour un client parti, -log(1 - p) pour un client resté.
  Un client parti à qui l'on donnait p = 0,01 coûte -log(0,01) = 4,6 ; à qui l'on donnait 0,9, 0,1 ;
- la DESCENTE DE GRADIENT trouve ce minimum à petits pas : le GRADIENT (la liste des dérivées de la
  perte par rapport à chaque w et à b) indique la direction où la perte MONTE ; on fait un pas dans
  l'autre sens, `w = w - pas * grad_w`, et on recommence jusqu'à ce que la pente soit presque nulle ;
- un ARBRE DE DÉCISION pose des questions « x_j <= seuil ? ». À chaque nœud, il essaie toutes les
  colonnes et tous les seuils, garde la question qui sépare le mieux les partants des restants (au sens
  de l'impureté de GINI), puis recommence de chaque côté, RÉCURSIVEMENT ;
- les ordinateurs calculent avec des nombres à virgule de 64 bits : exp(710) vaut déjà `inf`, et
  1 - 1e-17 vaut exactement 1.0. Un bon code numérique réécrit ses formules pour ne jamais y tomber.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_23 import *
    >>> float(np.exp(710.0))                           # le plus grand float est environ 1,8e308
    inf
    >>> 1 - 1e-17 == 1.0                               # 16 chiffres significatifs, pas un de plus
    True
    >>> float(np.logaddexp(0.0, 1000.0))               # log(1 + exp(1000)), calculé sans exp(1000)
    1000.0

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites les défis DANS L'ORDRE : 142
   se sert de votre sigmoïde, 143 de votre gradient, 145 de votre meilleure coupure.
2. Lancez `python scripts/defis.py --defi 140` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_23.py`.

Les outils communs (`clients_exemple`, `generer_clients`, `NUMERIQUES`, `gradient_numerique`,
`afficher_arbre`) sont dans `defis/maison.py`, déjà importés ci-dessous. Vous avez le droit d'utiliser
scikit-learn pour PRÉPARER les données (`StandardScaler`) et pour COMPARER vos résultats, pas pour
apprendre à votre place.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import numbers  # noqa: F401 — utile pour vérifier qu'un paramètre est un nombre (numbers.Integral, numbers.Real)

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: F401
from sklearn.utils.validation import check_is_fitted  # noqa: F401

from defis.maison import (  # noqa: F401
    NUMERIQUES,
    afficher_arbre,
    clients_exemple,
    generer_clients,
    gradient_numerique,
)

# ============================================================================ La sigmoïde


def sigmoide(z):
    """Défi 140 🟢 — La sigmoïde, qui ne déborde jamais.

    La formule tient en une ligne : sigmoide(z) = 1 / (1 + exp(-z)). Écrite telle quelle, elle marche…
    jusqu'au premier client au score très négatif : `np.exp(1000)` vaut `inf` et numpy affiche
    « RuntimeWarning: overflow encountered in exp ». Ici le résultat (0) est encore juste par chance,
    mais la même erreur, deux lignes plus loin, donne NaN, et un seul NaN suffit à ruiner un entraînement.

        >>> sigmoide(0.0)
        0.5
        >>> sigmoide(np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0]))
        array([0.        , 0.11920292, 0.5       , 0.88079708, 1.        ])
        >>> float(sigmoide(-600.0))                   # exp(-600) : minuscule, mais pas nul
        2.6503965530043108e-261

    Les règles :

    - `z` est un nombre, une liste ou un tableau numpy de N'IMPORTE QUELLE forme (même à 2 dimensions,
      même d'entiers). Le résultat est un tableau de floats de la même forme ; pour un nombre, un float ;
    - aucun avertissement, même pour z = ±1000 : le correcteur transforme chaque RuntimeWarning en erreur ;
    - les résultats sont justes à la précision de la machine, y compris les tout petits : sigmoide(-40)
      vaut environ 4,2e-18, pas 0 (et sigmoide(-600) vaut environ 2,7e-261, pas sigmoide(-500)) ;
    - NaN reste NaN ; `z` n'est pas modifié ;
    - pas de boucle : 2 millions de scores en moins de 0,25 seconde.

    Ce que vous apprenez : le débordement des flottants, réécrire une formule pour qu'elle soit stable,
    `np.where` (qui calcule ses DEUX branches partout !), la vectorisation.

    Indice : pour z >= 0, 1 / (1 + exp(-z)) ne déborde pas (exp(-z) est entre 0 et 1). Pour z < 0,
    multipliez en haut et en bas par exp(z) : exp(z) / (1 + exp(z)), et là, c'est exp(z) qui est entre 0
    et 1. Dans les deux cas, l'exponentielle à calculer est e = exp(-|z|) : `e = np.exp(-np.abs(z))`.
    Commencez par `z = np.asarray(z, dtype=float)`, et finissez par `float(...)` si `resultat.ndim == 0`.
    """
    raise NotImplementedError


# ============================================================================ La perte logistique


def perte_logistique(y, z) -> float:
    """Défi 141 🟢 — La perte logistique (log-loss), calculée à partir des SCORES.

    C'est le chiffre que l'entraînement fait descendre : pour chaque client, -log(p) s'il est parti
    (y = 1), -log(1 - p) s'il est resté (y = 0), avec p = sigmoide(z) ; puis la MOYENNE sur les clients.
    Passer par p pose problème : dès que z dépasse 37 environ, p vaut EXACTEMENT 1.0 en flottant, et
    log(1 - p) = log(0) = -inf. Un modèle très sûr de lui et faux aurait une perte infinie (ou NaN).

    Bonne nouvelle : la formule se simplifie (voir l'indice) en log(1 + exp(z)) - y·z, qui ne passe
    jamais par p.

        >>> perte_logistique([1, 0], [0.0, 0.0])        # p = 0,5 pour tous : log(2) par client
        0.6931471805599453
        >>> perte_logistique([1, 0, 1], [2.0, -1.0, 0.5])
        0.3047555609137674
        >>> perte_logistique([1, 0], [-800.0, 800.0])   # sûr de lui, et faux : très cher, mais pas infini
        800.0

    Les règles :

    - `y` (les réponses, 0 ou 1) et `z` (les scores) : deux listes, tableaux ou Series de même longueur,
      associés par POSITION (le i-ème y avec le i-ème z, quel que soit l'index) ;
    - le résultat est un nombre : la MOYENNE des pertes des clients (pas la somme) ;
    - aucun avertissement, et un résultat fini même pour z = ±800 ;
    - `ValueError` si les longueurs diffèrent, s'il n'y a aucun client, ou si y contient autre chose que
      0 et 1 (un 2, la convention -1 / +1, une probabilité 0,5…) ;
    - y et z ne sont pas modifiés ; 2 millions de clients en moins de 0,4 seconde.

    Le correcteur compare votre résultat à `sklearn.metrics.log_loss(y, sigmoide(z))` sur des scores
    ordinaires.

    Ce que vous apprenez : ce que mesure la log-loss (et pourquoi la moyenne), simplifier une formule pour
    la rendre stable, `np.logaddexp`, et pourquoi on calcule les pertes à partir des scores (les
    bibliothèques le font : `BCEWithLogitsLoss` en PyTorch, par exemple).

    Indice : log(p) = -log(1 + exp(-z)) et log(1 - p) = -log(1 + exp(z)). En remplaçant, la perte d'un
    client devient log(1 + exp(z)) - y·z. Et `np.logaddexp(a, b)` calcule log(exp(a) + exp(b)) sans
    calculer les exponentielles : log(1 + exp(z)) = `np.logaddexp(0, z)`. Vérifiez y avec
    `np.isin(y, (0, 1)).all()`.
    """
    raise NotImplementedError


# ============================================================================ Le gradient


def gradient_logistique(X, y, w, b, alpha: float = 0.0):
    """Défi 142 🟡 — Le gradient de la perte, vérifié par différences finies.

    L'entraînement minimise le COÛT J(w, b) = perte_logistique(y, X @ w + b) + (alpha / 2)·||w||².
    Le deuxième terme est la « pénalité L2 » : elle empêche les coefficients de devenir énormes (c'est le
    `C` de scikit-learn, à l'envers). Pour descendre, il faut la PENTE de J dans chaque direction : ses
    dérivées par rapport à chaque coefficient de w, et par rapport à b. C'est le gradient.

    Les formules (démontrées dans le corrigé, en 3 lignes) :

        erreur = sigmoide(X @ w + b) - y           # une par client : la probabilité moins la réponse
        grad_w = X.T @ erreur / n + alpha * w       # une par colonne
        grad_b = moyenne de erreur                  # b n'est PAS pénalisé (comme dans scikit-learn)

        >>> X = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])
        >>> grad_w, grad_b = gradient_logistique(X, [1, 0, 1], np.array([0.5, -0.25]), 0.1)
        >>> grad_w.round(4), round(grad_b, 4)
        (array([-0.3642, -0.5808]), -0.0314)

    Comment savoir si un gradient écrit à la main est juste ? On le compare aux DIFFÉRENCES FINIES : on
    bouge chaque coefficient d'un tout petit h et on mesure la pente. `gradient_numerique` (dans
    `defis/maison.py`) le fait pour vous. Les 3 nombres doivent être les mêmes :

        >>> y = [1, 0, 1]
        >>> def cout(v):                                # v = [w0, w1, b]
        ...     return perte_logistique(y, X @ v[:2] + v[2])
        >>> gradient_numerique(cout, [0.5, -0.25, 0.1]).round(4)
        array([-0.3642, -0.5808, -0.0314])

    Les règles :

    - X : un tableau ou un DataFrame (n clients, d colonnes) ; y : n réponses 0 ou 1 ; w : d coefficients ;
      b : un nombre ; alpha : un nombre positif ou nul. Tout est associé par POSITION ;
    - renvoie le couple `(grad_w, grad_b)` : un tableau numpy de forme (d,) et un nombre ;
    - aucun avertissement même si les scores valent ±1000 (servez-vous de VOTRE `sigmoide`) ;
    - `ValueError` si les dimensions ne vont pas ensemble (numpy le fait pour vous si vous ne cassez rien) ;
    - X, y et w ne sont pas modifiés ; pas de boucle sur les clients : 300 000 clients et 10 colonnes en
      moins de 0,15 seconde.

    Ce que vous apprenez : la règle de la chaîne appliquée à un vrai modèle, le produit matriciel `@` et
    la transposée `.T` (une somme sur les clients pour chaque colonne, sans boucle), la pénalité L2, et le
    réflexe du « gradient check » par différences finies.

    Indice : faites un dessin des formes. X est (n, d) et erreur est (n,) : `X.T @ erreur` est (d,), une
    somme par colonne. `X @ erreur` ne marche que si n == d… et donne alors un résultat faux. La dérivée
    de (alpha / 2)·w² est alpha·w : le 2 et le 1/2 se simplifient.
    """
    raise NotImplementedError


# ============================================================================ La descente de gradient


class RegressionLogistiqueGD(ClassifierMixin, BaseEstimator):
    """Défi 143 🟡 — La régression logistique, entraînée par descente de gradient.

    On part de w = 0 et b = 0 (le modèle dit 0,5 à tout le monde), puis on répète :

        1. calculer le gradient (grad_w, grad_b) avec VOTRE gradient_logistique (défi 142) ;
        2. si la norme de TOUT le gradient, √(||grad_w||² + grad_b²), est < tol : c'est le minimum, on
           s'arrête (sans faire de pas) ;
        3. sinon, faire un pas : w = w - pas · grad_w et b = b - pas · grad_b.

    au plus `n_iter_max` fois. Avec des colonnes standardisées (moyenne 0, écart-type 1), on retrouve
    EXACTEMENT les coefficients de `LogisticRegression(C=1 / (n · alpha))` de scikit-learn : c'est la même
    perte, écrite autrement. Le correcteur le vérifie à 1e-5 près.

        >>> from sklearn.preprocessing import StandardScaler
        >>> clients = clients_exemple()
        >>> X = StandardScaler().fit_transform(clients[NUMERIQUES])
        >>> modele = RegressionLogistiqueGD().fit(X, clients["churn"])
        >>> modele.coef_.round(3), round(modele.intercept_, 3), modele.n_iter_
        (array([-0.186,  1.332,  1.299]), -0.062, 604)
        >>> modele.predict_proba(X[:3]).round(3)
        array([[0.138, 0.862],
               [0.992, 0.008],
               [0.607, 0.393]])
        >>> modele.predict(X[:4])
        array([1, 0, 0, 0])

    Les règles (le contrat scikit-learn de la série 22) :

    - `__init__(self, alpha=0.01, pas=0.5, n_iter_max=10_000, tol=1e-6)` range ses paramètres TELS QUELS ;
    - `fit(X, y)` vérifie les paramètres (`ValueError` si alpha < 0, pas <= 0, tol < 0, ou si n_iter_max
      n'est pas un entier >= 1 ; un booléen n'est pas un nombre), puis :
        · `self.classes_ = np.unique(y)` : il en faut exactement deux (sinon `ValueError`). La DEUXIÈME est
          la classe « positive » : les étiquettes peuvent être "non" / "oui", pas forcément 0 / 1 ;
        · la descente ci-dessus, en repartant de w = 0 et b = 0 à CHAQUE fit ;
        · `self.coef_` (tableau de forme (d,)), `self.intercept_` (un nombre), `self.n_iter_` (le nombre
          de pas FAITS : n_iter_max si on ne s'est jamais arrêté) ; renvoie `self` ;
    - `predict_proba(X)` : `check_is_fitted(self)`, puis un tableau (n, 2) : colonne 0 = 1 - p,
      colonne 1 = p = sigmoide(X @ coef_ + intercept_) ;
    - `predict(X)` : `classes_[1]` si p > 0,5, sinon `classes_[0]` (des étiquettes, comme scikit-learn) ;
    - X n'est jamais modifié.

    Ce que vous apprenez : ce que fait VRAIMENT `fit`, le critère d'arrêt, le rôle du pas et de la
    standardisation, et la correspondance alpha ↔ C entre deux écritures du même modèle.

    Indice : `X = np.asarray(X, dtype=float)`, `y01 = (np.asarray(y) == self.classes_[1]).astype(float)`.
    La boucle : `for iteration in range(self.n_iter_max):`, puis `self.n_iter_ = iteration` juste avant
    le `break`. Pour la norme : `np.sqrt(grad_w @ grad_w + grad_b ** 2)`. Essayez `pas=5` sur des données
    NON standardisées : vous verrez pourquoi on standardise.
    """

    def __init__(self, alpha=0.01, pas=0.5, n_iter_max=10_000, tol=1e-6):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


# ============================================================================ La meilleure coupure


def meilleure_coupure(x, y, min_feuille=1):
    """Défi 144 🟡 — La meilleure question « x <= seuil ? » pour une colonne, au sens de Gini.

    L'IMPURETÉ DE GINI d'un groupe où une proportion q de clients est partie vaut 2·q·(1 - q) : 0 si
    tout le monde est parti ou resté (groupe « pur »), 0,5 au pire (moitié-moitié). Couper le groupe en
    deux fait baisser l'impureté ; le GAIN d'une coupure mesure de combien :

        gain = gini(parent) - n_gauche / n · gini(gauche) - n_droite / n · gini(droite)

    (chaque enfant pèse selon sa PART des clients). La meilleure coupure est celle qui a le plus grand gain.

        >>> meilleure_coupure([1, 1, 2, 3], [0, 1, 1, 1])
        (1.5, 0.125)
        >>> clients = clients_exemple()
        >>> meilleure_coupure(clients["appels"], clients["churn"])
        (1.5, 0.25)
        >>> meilleure_coupure([4, 4, 4], [0, 1, 0])          # toutes les valeurs sont égales
        (None, 0.0)

    Les règles :

    - `x` : les valeurs d'une colonne (liste, tableau ou Series, PAS forcément triées) ; `y` : les
      réponses 0 ou 1, associées par POSITION ; un client va à GAUCHE si x <= seuil, à droite sinon ;
    - les seuils possibles sont au MILIEU de deux valeurs voisines DIFFÉRENTES (une fois x trié) : entre 1
      et 2, le seuil est 1,5. Jamais entre deux valeurs égales : deux clients identiques ne peuvent pas
      être séparés ;
    - chaque côté doit garder au moins `min_feuille` clients ;
    - renvoie `(seuil, gain)` pour la coupure au plus grand gain ; à égalité, le PLUS PETIT seuil (comme
      scikit-learn) ; s'il n'y a aucune coupure possible : `(None, 0.0)`. Un gain nul est un vrai gain :
      s'il existe une coupure, on la renvoie même si elle ne gagne rien ;
    - `ValueError` si les longueurs diffèrent, s'il n'y a aucun client, si x contient NaN, si y contient
      autre chose que 0 et 1, ou si min_feuille n'est pas un entier >= 1 ;
    - x et y ne sont pas modifiés ;
    - O(n log n) : 200 000 clients en moins d'une seconde. Essayer chaque seuil en recomptant tous les
      clients, c'est O(n²) : des heures pour un million de clients.

    Le correcteur compare vos seuils à ceux de `DecisionTreeClassifier(max_depth=1)`.

    Ce que vous apprenez : l'impureté de Gini, le critère qui fait pousser les arbres (et les forêts, et
    le gradient boosting) ; trier une fois puis cumuler (`np.argsort`, `np.cumsum`) au lieu d'une boucle
    dans une boucle ; les cas limites (valeurs égales, effectif minimum).

    Indice : `ordre = np.argsort(x)` (et pas `x.sort()`, qui modifie le tableau de l'appelant), puis
    `xs, ys = x[ordre], y[ordre]`. Couper après les k premiers clients (k = 1 … n-1) : à gauche, k
    clients dont `np.cumsum(ys)[k - 1]` partis. Calculez TOUS les gains d'un coup avec des tableaux, puis
    gardez les coupures possibles (`xs[1:] > xs[:-1]`, et les effectifs) et prenez `np.argmax`, qui
    renvoie la PREMIÈRE position du maximum.
    """
    raise NotImplementedError


# ============================================================================ L'arbre de décision


class ArbreDecision(ClassifierMixin, BaseEstimator):
    """Défi 145 🔴 — Un arbre de décision complet, récursif : le même que celui de scikit-learn.

    Pour construire le nœud qui reçoit un groupe de clients, à une profondeur donnée (la racine est à la
    profondeur 0, et reçoit tout le monde) :

        1. ce nœud devient une FEUILLE si la profondeur maximale est atteinte (profondeur >= max_profondeur),
           s'il est PUR (tous ses clients sont de la même classe), ou si AUCUNE coupure n'est possible
           (votre `meilleure_coupure` renvoie un seuil None pour toutes les colonnes) ;
        2. sinon, il essaie `meilleure_coupure` sur CHAQUE colonne, avec SES clients, et garde celle qui a
           le plus grand gain (à égalité, la première colonne) — même si ce gain est nul ;
        3. il envoie à gauche les clients tels que x_colonne <= seuil, à droite les autres, et construit
           ses deux enfants de la même façon, à la profondeur suivante : c'est la RÉCURSION.

        >>> clients = clients_exemple()
        >>> arbre = ArbreDecision(max_profondeur=2).fit(clients[NUMERIQUES], clients["churn"])
        >>> afficher_arbre(arbre.arbre_, NUMERIQUES)
        |--- appels <= 1.5
        |   |--- feuille : proba 0.0 (4 clients)
        |--- appels >  1.5
        |   |--- facture <= 42.5
        |   |   |--- feuille : proba 0.0 (1 client)
        |   |--- facture >  42.5
        |   |   |--- feuille : proba 0.857 (7 clients)
        >>> arbre.profondeur_, arbre.n_feuilles_
        (2, 3)
        >>> arbre.predict_proba(clients[NUMERIQUES].iloc[:2]).round(3)
        array([[0.143, 0.857],
               [1.   , 0.   ]])

    Les règles :

    - `__init__(self, max_profondeur=3, min_feuille=1)` range ses paramètres TELS QUELS ;
      `max_profondeur=None` veut dire « sans limite » ;
    - `fit(X, y)` vérifie les paramètres (`ValueError` si max_profondeur n'est ni None ni un entier >= 1,
      ou si min_feuille n'est pas un entier >= 1 ; un booléen n'est pas un entier), puis range :
        · `self.classes_ = np.unique(y)` (exactement deux classes, sinon `ValueError` ; la deuxième est la
          classe positive) ;
        · `self.arbre_` : le nœud racine, un DICTIONNAIRE. Une feuille vaut
          `{"proba": p, "effectif": n}` (p = la PROPORTION de clients de classes_[1] parmi les n clients
          du train qui y arrivent) ; un nœud qui coupe vaut
          `{"colonne": j, "seuil": s, "gauche": <nœud>, "droite": <nœud>}` (d'autres clés sont permises) ;
        · `self.profondeur_` (le nombre de coupures de la racine à la feuille la plus basse) et
          `self.n_feuilles_` ; renvoie `self` ;
    - `predict_proba(X)` : pour chaque client, on descend depuis la racine (x_colonne <= seuil → gauche,
      sinon droite) jusqu'à une feuille ; un tableau (n, 2) : [1 - p, p] ;
    - `predict(X)` : `classes_[1]` si p > 0,5, sinon `classes_[0]` (à 50 / 50, la PREMIÈRE classe, comme
      scikit-learn) ;
    - X n'est jamais modifié ; 5 000 clients et une profondeur de 6 en moins de 4 secondes.

    Le correcteur ne compare pas seulement le résultat final : il refait vos calculs à CHAQUE nœud (et
    vous dit à quel nœud l'arbre se trompe), puis compare vos prédictions à `DecisionTreeClassifier`
    (mêmes réglages : max_depth, min_samples_leaf). Seule différence permise : en cas d'égalité PARFAITE
    entre deux colonnes, scikit-learn tire au sort (c'est à ça que sert son `random_state`) ; le
    correcteur accepte n'importe laquelle des colonnes à égalité.

    Ce que vous apprenez : la récursion sur une vraie structure de données (un arbre de dictionnaires),
    les conditions d'arrêt, pourquoi un gain nul n'est pas une raison de s'arrêter (essayez le « ou
    exclusif » : y = x1 XOR x2), et ce que règlent vraiment max_depth et min_samples_leaf.

    Indice : écrivez une méthode `_construire(self, X, y01, profondeur)` qui renvoie un dictionnaire et
    s'appelle elle-même pour `gauche` et `droite`, avec `X[a_gauche]` et `y01[a_gauche]` (où
    `a_gauche = X[:, colonne] <= seuil`). Pour predict_proba, une boucle `while "colonne" in noeud:` par
    client suffit. profondeur_ et n_feuilles_ se calculent par deux petites fonctions récursives.
    """

    def __init__(self, max_profondeur=3, min_feuille=1):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

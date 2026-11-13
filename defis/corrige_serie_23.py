"""Série 23 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_23.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours
un signe, un `/ n`, ou un `<` à la place d'un `<=`.

Fil rouge de la série : `fit` n'est pas magique. Une régression logistique, c'est une PERTE (la
log-loss) qu'on fait descendre pas à pas en suivant son GRADIENT ; un arbre de décision, c'est une
COUPURE (la meilleure, au sens de Gini) répétée récursivement. Quatre habitudes de numéricien :

1. on ne calcule jamais `exp` d'un grand nombre : on réécrit la formule pour que l'exponentielle ne
   reçoive que des nombres négatifs (`exp(-|z|)`, `np.logaddexp`) ;
2. on calcule la perte à partir des SCORES (z), pas des probabilités arrondies à 0 ou 1 ;
3. on vérifie un gradient écrit à la main par différences finies (`gradient_numerique`) AVANT de
   s'en servir ;
4. on vectorise : une boucle Python par client est 100 fois plus lente qu'une opération NumPy, et un
   tri suivi d'un cumul (`np.cumsum`) remplace une boucle dans une boucle.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from defis.maison import NUMERIQUES, clients_exemple, generer_clients, gradient_numerique  # noqa: F401

# ============================================================================ La sigmoïde


def sigmoide(z):
    z = np.asarray(z, dtype=float)  # un nombre, une liste ou un tableau : tout devient un tableau de float
    # 1 / (1 + exp(-z)) déborde pour z = -1000 (exp(1000) = inf, avec un RuntimeWarning). L'astuce :
    # e = exp(-|z|) est TOUJOURS entre 0 et 1, et les deux écritures de la sigmoïde
    #     1 / (1 + exp(-z))   (pour z >= 0)      et      exp(z) / (1 + exp(z))   (pour z < 0)
    # deviennent 1 / (1 + e) et e / (1 + e). Aucune exponentielle d'un nombre positif : rien ne déborde,
    # et les très petits résultats (sigmoide(-600) ≈ 2,6e-261) restent exacts, au lieu de valoir
    # 1 - 1 = 0 comme avec `1 - sigmoide(600)`.
    e = np.exp(-np.abs(z))
    resultat = np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    # `np.where` calcule les DEUX branches partout : ici, aucune ne peut déborder, c'est pour ça que ça marche.
    # `scipy.special.expit(z)` fait la même chose (en C).
    return float(resultat) if resultat.ndim == 0 else resultat


# ============================================================================ La perte logistique


def perte_logistique(y, z) -> float:
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if y.ndim != 1 or y.shape != z.shape:
        raise ValueError(f"y et z doivent être deux listes de même longueur (reçu {y.shape} et {z.shape}).")
    if y.size == 0:
        raise ValueError("Aucun client : la moyenne d'une liste vide n'existe pas.")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y doit valoir 0 ou 1 (pas -1 / +1, pas une probabilité).")
    # Pour un client, -[y·log(p) + (1-y)·log(1-p)] avec p = sigmoide(z) se simplifie en
    #     log(1 + exp(z)) - y·z
    # (log(p) = -log(1 + exp(-z)) et log(1-p) = -log(1 + exp(z))). `np.logaddexp(0, z)` calcule
    # log(exp(0) + exp(z)) = log(1 + exp(z)) SANS jamais calculer exp(z) quand z est grand.
    # Passer par p arrondi fait log(0) = -inf dès que |z| dépasse 37 environ (p vaut alors exactement 1.0).
    # La MOYENNE, et pas la somme : la perte ne doit pas dépendre du nombre de clients (sinon le pas de la
    # descente de gradient et la pénalité alpha changent de sens avec la taille des données).
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


# ============================================================================ Le gradient


def gradient_logistique(X, y, w, b, alpha: float = 0.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[1] != w.shape[0]:
        raise ValueError(f"Dimensions incompatibles : X {X.shape}, y {y.shape}, w {w.shape}.")
    n = X.shape[0]
    # La démonstration, en 3 lignes (perte d'un client : log(1 + exp(z)) - y·z, défi 141) :
    #   1. d/dz log(1 + exp(z)) = exp(z) / (1 + exp(z)) = sigmoide(z)   et   d/dz (y·z) = y ;
    #   2. donc la dérivée de la perte par rapport au score est sigmoide(z) - y : « la probabilité moins la réponse » ;
    #   3. règle de la chaîne : z = x · w + b, donc dz/dw_j = x_j et dz/db = 1. On fait la moyenne sur les clients.
    erreur = sigmoide(X @ w + b) - y          # (n,) : une erreur par client
    # X.T @ erreur additionne, pour chaque colonne, x_ij × erreur_i sur tous les clients : c'est la boucle
    # « pour chaque client, pour chaque colonne », sans boucle. On divise par n (la perte est une MOYENNE).
    grad_w = X.T @ erreur / n + alpha * w     # dérivée de (alpha / 2)·||w||² : alpha·w (le 2 se simplifie)
    grad_b = float(erreur.mean())             # l'ordonnée à l'origine n'est PAS pénalisée (comme scikit-learn)
    return grad_w, grad_b


# ============================================================================ La descente de gradient


def _entier(valeur, minimum: int) -> bool:
    return not isinstance(valeur, bool) and isinstance(valeur, numbers.Integral) and valeur >= minimum


def _reel(valeur, minimum: float, strict: bool = False) -> bool:
    if isinstance(valeur, bool) or not isinstance(valeur, numbers.Real) or valeur != valeur:
        return False
    return valeur > minimum if strict else valeur >= minimum


class RegressionLogistiqueGD(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=0.01, pas=0.5, n_iter_max=10_000, tol=1e-6):
        # TELS QUELS (série 22) : la vérification a lieu dans `fit`.
        self.alpha = alpha
        self.pas = pas
        self.n_iter_max = n_iter_max
        self.tol = tol

    def fit(self, X, y):
        if not _reel(self.alpha, 0.0):
            raise ValueError(f"alpha doit être un nombre positif ou nul, pas {self.alpha!r}.")
        if not _reel(self.pas, 0.0, strict=True):
            raise ValueError(f"pas doit être un nombre strictement positif, pas {self.pas!r}.")
        if not _entier(self.n_iter_max, 1):
            raise ValueError(f"n_iter_max doit être un entier >= 1, pas {self.n_iter_max!r}.")
        if not _reel(self.tol, 0.0):
            raise ValueError(f"tol doit être un nombre positif ou nul, pas {self.tol!r}.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(y) != len(X):
            raise ValueError(f"X doit être un tableau à 2 dimensions avec une ligne par réponse (X {X.shape}, "
                             f"y {y.shape}).")
        # Les étiquettes peuvent être 0/1, "non"/"oui", False/True… `classes_` les range dans l'ordre (c'est
        # la convention de scikit-learn) et la DEUXIÈME est la classe « positive », celle de predict_proba[:, 1].
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(f"Il faut exactement deux classes dans y, pas {self.classes_.size}.")
        y01 = (y == self.classes_[1]).astype(float)

        # On repart de zéro à CHAQUE fit (sinon un deuxième fit continuerait le premier : « warm start »).
        w, b = np.zeros(X.shape[1]), 0.0
        self.n_iter_ = self.n_iter_max
        for iteration in range(self.n_iter_max):
            grad_w, grad_b = gradient_logistique(X, y01, w, b, self.alpha)
            # On s'arrête quand la pente est presque nulle : la norme de TOUT le gradient (w ET b).
            if np.sqrt(grad_w @ grad_w + grad_b * grad_b) < self.tol:
                self.n_iter_ = iteration
                break
            # On DESCEND : on va dans le sens opposé au gradient (le gradient pointe vers la montée).
            w = w - self.pas * grad_w
            b = b - self.pas * grad_b
        self.coef_ = w
        self.intercept_ = float(b)
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = sigmoide(self.decision_function(X))
        return np.column_stack([1.0 - p, p])  # colonne 0 : classes_[0] ; colonne 1 : classes_[1]

    def predict(self, X):
        # Comme scikit-learn : la classe positive si p > 0,5 (score > 0), sinon la première.
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]


# ============================================================================ La meilleure coupure


def meilleure_coupure(x, y, min_feuille=1):
    if not _entier(min_feuille, 1):
        raise ValueError(f"min_feuille doit être un entier >= 1, pas {min_feuille!r}.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"x et y doivent être deux listes de même longueur (reçu {x.shape} et {y.shape}).")
    if x.size == 0:
        raise ValueError("Aucun client : rien à couper.")
    if np.isnan(x).any():
        raise ValueError("x contient des NaN : où les mettre, à gauche ou à droite ? Complétez-les d'abord.")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y doit valoir 0 ou 1.")

    # 1. On TRIE une fois (O(n log n)) : `np.argsort` ne touche pas à x (`x.sort()` modifierait le tableau
    #    de l'appelant !). Ensuite, couper « après les k premiers » donne toutes les coupures possibles.
    ordre = np.argsort(x, kind="stable")
    xs, ys = x[ordre], y[ordre]
    n = xs.size
    # 2. Pour chaque k (de 1 à n-1 clients à gauche), le nombre de partants à gauche est un CUMUL :
    #    np.cumsum le calcule pour tous les k d'un coup, au lieu de recompter à chaque coupure (O(n²)).
    n_gauche = np.arange(1, n)
    n_droite = n - n_gauche
    partis_gauche = np.cumsum(ys)[:-1]
    partis = ys.sum()
    p_gauche = partis_gauche / n_gauche
    p_droite = (partis - partis_gauche) / n_droite
    p = partis / n

    def gini(q):
        return 2.0 * q * (1.0 - q)  # = 1 - q² - (1 - q)² : la probabilité de se tromper en tirant au hasard

    # 3. Le gain : l'impureté du parent MOINS celle des enfants, chacun PONDÉRÉ par sa part des clients.
    gain = gini(p) - (n_gauche / n) * gini(p_gauche) - (n_droite / n) * gini(p_droite)
    # 4. Une coupure n'est possible qu'entre deux valeurs DIFFÉRENTES (sinon deux clients identiques
    #    finiraient de part et d'autre), et si chaque côté garde au moins min_feuille clients.
    possible = (xs[1:] > xs[:-1]) & (n_gauche >= min_feuille) & (n_droite >= min_feuille)
    if not possible.any():
        return None, 0.0
    gain_possible = np.where(possible, gain, -np.inf)
    # 5. À égalité (au bruit de calcul près), le PLUS PETIT seuil : c'est le premier, puisque xs est trié.
    k = int(np.flatnonzero(gain_possible >= gain_possible.max() - 1e-12)[0])
    # Le seuil est au MILIEU des deux valeurs voisines (comme scikit-learn) : la règle est x <= seuil.
    return float((xs[k] + xs[k + 1]) / 2.0), float(gain[k])


# ============================================================================ L'arbre de décision


class ArbreDecision(ClassifierMixin, BaseEstimator):
    def __init__(self, max_profondeur=3, min_feuille=1):
        self.max_profondeur = max_profondeur
        self.min_feuille = min_feuille

    def fit(self, X, y):
        if self.max_profondeur is not None and not _entier(self.max_profondeur, 1):
            raise ValueError(f"max_profondeur doit être None ou un entier >= 1, pas {self.max_profondeur!r}.")
        if not _entier(self.min_feuille, 1):
            raise ValueError(f"min_feuille doit être un entier >= 1, pas {self.min_feuille!r}.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(y) != len(X):
            raise ValueError(f"X doit être un tableau à 2 dimensions avec une ligne par réponse (X {X.shape}, "
                             f"y {y.shape}).")
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(f"Il faut exactement deux classes dans y, pas {self.classes_.size}.")
        self.arbre_ = self._construire(X, (y == self.classes_[1]).astype(float), profondeur=0)
        self.profondeur_ = _profondeur(self.arbre_)
        self.n_feuilles_ = _n_feuilles(self.arbre_)
        return self

    def _construire(self, X, y, profondeur: int) -> dict:
        # Un nœud reçoit SES clients (X et y déjà filtrés) : c'est la récursion qui fait tout le travail.
        feuille = {"proba": float(y.mean()), "effectif": int(y.size)}
        # Les trois raisons de s'arrêter : profondeur atteinte, nœud pur (rien à séparer), aucune coupure
        # possible (toutes les colonnes constantes ici, ou trop peu de clients pour min_feuille).
        if self.max_profondeur is not None and profondeur >= self.max_profondeur:
            return feuille
        if feuille["proba"] in (0.0, 1.0):
            return feuille
        meilleure = None  # (gain, colonne, seuil)
        for colonne in range(X.shape[1]):
            seuil, gain = meilleure_coupure(X[:, colonne], y, self.min_feuille)
            # « > meilleur + 1e-12 » : à égalité, la PREMIÈRE colonne garde la place.
            if seuil is not None and (meilleure is None or gain > meilleure[0] + 1e-12):
                meilleure = (gain, colonne, seuil)
        if meilleure is None:
            return feuille
        gain, colonne, seuil = meilleure
        a_gauche = X[:, colonne] <= seuil
        return {
            "colonne": colonne,
            "seuil": seuil,
            "gain": gain,
            "effectif": int(y.size),
            "gauche": self._construire(X[a_gauche], y[a_gauche], profondeur + 1),
            "droite": self._construire(X[~a_gauche], y[~a_gauche], profondeur + 1),
        }

    def predict_proba(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        p = np.empty(len(X))
        for i, ligne in enumerate(X):
            noeud = self.arbre_
            while "colonne" in noeud:  # on descend jusqu'à une feuille : x <= seuil → à gauche
                noeud = noeud["gauche"] if ligne[noeud["colonne"]] <= noeud["seuil"] else noeud["droite"]
            p[i] = noeud["proba"]
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        # Comme scikit-learn (argmax des probabilités) : à 50 / 50, c'est la PREMIÈRE classe.
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]


def _profondeur(noeud: dict) -> int:
    if "colonne" not in noeud:
        return 0
    return 1 + max(_profondeur(noeud["gauche"]), _profondeur(noeud["droite"]))


def _n_feuilles(noeud: dict) -> int:
    if "colonne" not in noeud:
        return 1
    return _n_feuilles(noeud["gauche"]) + _n_feuilles(noeud["droite"])


# ============================================================================ Des tests comme dans tests/


def test_sigmoide_ne_deborde_pas():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # un RuntimeWarning (débordement) ferait échouer le test
        resultat = sigmoide([-1000.0, 0.0, 1000.0])
    assert resultat.tolist() == [0.0, 0.5, 1.0]
    assert abs(sigmoide(-600.0) / np.exp(-600.0) - 1) < 1e-12


def test_perte_logistique_reste_finie_quand_le_modele_se_trompe_avec_assurance():
    assert perte_logistique([1, 0], [-800.0, 800.0]) == 800.0
    assert abs(perte_logistique([1, 0], [0.0, 0.0]) - np.log(2)) < 1e-15


def test_gradient_logistique_egale_les_differences_finies():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(50, 3)), rng.integers(0, 2, 50)
    w, b = rng.normal(size=3), 0.3

    def cout(v):
        return perte_logistique(y, X @ v[:3] + v[3]) + 0.1 / 2 * v[:3] @ v[:3]

    grad_w, grad_b = gradient_logistique(X, y, w, b, alpha=0.1)
    assert np.allclose(np.append(grad_w, grad_b), gradient_numerique(cout, np.append(w, b)), atol=1e-7)


def test_descente_de_gradient_retrouve_scikit_learn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    clients = generer_clients(500, graine=3)
    X, y = StandardScaler().fit_transform(clients[NUMERIQUES]), clients["churn"]
    modele = RegressionLogistiqueGD(alpha=0.01, tol=1e-10, n_iter_max=50_000).fit(X, y)
    reference = LogisticRegression(C=1 / (len(X) * 0.01), tol=1e-12, max_iter=10_000).fit(X, y)
    assert np.allclose(modele.coef_, reference.coef_[0], atol=1e-6)
    assert abs(modele.intercept_ - reference.intercept_[0]) < 1e-6


def test_meilleure_coupure_ne_coupe_jamais_entre_deux_valeurs_egales():
    seuil, gain = meilleure_coupure([1, 1, 2, 3], [0, 1, 1, 1])
    assert seuil == 1.5 and abs(gain - 0.125) < 1e-12
    assert meilleure_coupure([4, 4, 4], [0, 1, 0]) == (None, 0.0)


def test_arbre_identique_a_scikit_learn():
    from sklearn.tree import DecisionTreeClassifier

    clients = generer_clients(1500, graine=5)
    X, y = clients[NUMERIQUES], clients["churn"]
    arbre = ArbreDecision(max_profondeur=4, min_feuille=10).fit(X.iloc[:1000], y.iloc[:1000])
    reference = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0)
    reference.fit(X.iloc[:1000], y.iloc[:1000])
    assert np.allclose(arbre.predict_proba(X.iloc[1000:]), reference.predict_proba(X.iloc[1000:]))
    assert arbre.n_feuilles_ == reference.get_n_leaves() and arbre.profondeur_ == reference.get_depth()

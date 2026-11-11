"""Série 01 — Les fondamentaux, corrigés automatiquement.

✍️  C'EST VOTRE FICHIER. Mode d'emploi :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/progression.py` : chaque défi passe à ✅ dès qu'il est juste,
   sinon ❌ avec un message qui explique ce qui ne va pas.
3. Bloqué·e ? Relisez l'indice, puis l'exercice du PARCOURS indiqué entre parenthèses.
   Le corrigé commenté est dans `defis/corrige_serie_01.py` : ouvrez-le seulement APRÈS
   avoir vraiment essayé. C'est en cherchant qu'on progresse.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé. Faites-les dans l'ordre que vous voulez.
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (vous en aurez besoin)
import pandas as pd

# =========================================================================== Module 1 — Données


def taux_churn_par(df: pd.DataFrame, colonne: str) -> pd.Series:
    """Défi 1 🟢 (Parcours 1.2) — Taux de churn par modalité d'une colonne.

    Retourne une Series indexée par les modalités de `colonne`, contenant la moyenne de la
    colonne `churn`, triée du taux le PLUS élevé au plus faible.

    Exemple : taux_churn_par(df, "contract") → Mensuel 0.52 · Annuel 0.18 · Biannuel 0.07

    Indice : groupby(...)["churn"].mean(), puis sort_values.
    """
    raise NotImplementedError


def taux_manquants(df: pd.DataFrame) -> pd.Series:
    """Défi 2 🟢 — Part de valeurs manquantes (entre 0 et 1) par colonne.

    Ne garder que les colonnes qui ont AU MOINS une valeur manquante, triées de la plus
    touchée à la moins touchée. Si rien ne manque, retourner une Series vide.

    Indice : df.isna() donne des booléens ; la moyenne d'un booléen est une proportion.
    """
    raise NotImplementedError


def valeurs_aberrantes(serie: pd.Series, k: float = 1.5) -> pd.Series:
    """Défi 3 🟡 — Masque des valeurs aberrantes selon la règle de Tukey (écart interquartile).

    Avec Q1 = quantile 25 %, Q3 = quantile 75 % et IQR = Q3 - Q1, une valeur est aberrante
    si elle est < Q1 - k × IQR  ou  > Q3 + k × IQR.

    Retourne une Series de booléens avec LE MÊME INDEX que `serie` (True = aberrante).

    Indice : serie.quantile(0.25)
    """
    raise NotImplementedError


# ========================================================================= Module 2 — Variables


def ajouter_nouveau_client(df: pd.DataFrame, seuil_mois: int = 6) -> pd.DataFrame:
    """Défi 4 🟢 (Parcours 2.3) — Ajoute la colonne `is_new_customer`.

    `is_new_customer` vaut 1 si `tenure_months` < `seuil_mois`, sinon 0.

    ⚠️ Piège classique : ne modifiez PAS `df` en place, retournez une copie.
    Un pipeline qui modifie ses entrées crée des bugs très difficiles à trouver.
    """
    raise NotImplementedError


def standardiser(train: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Défi 5 🟡 (Parcours 2.1 et 2.5) — Recodez StandardScaler à la main, SANS fuite de données.

    z = (x - moyenne) / écart-type

    La moyenne et l'écart-type sont calculés sur `train` UNIQUEMENT, puis appliqués à
    `train` ET à `test`. Utilisez l'écart-type « de population » (ddof=0), comme scikit-learn.

    Retourne (train_standardisé, test_standardisé).

    Pourquoi ? En production, on ne connaît pas encore les futurs clients : le modèle ne doit
    jamais avoir « vu » le jeu de test, même à travers une simple moyenne.
    """
    raise NotImplementedError


# ======================================================================= Module 4 — Évaluation


def precision_rappel(y_true, y_pred) -> tuple[float, float]:
    """Défi 6 🟢 — Précision et rappel, calculés à la main (sans scikit-learn).

    VP = vrais positifs, FP = faux positifs, FN = faux négatifs.
        précision = VP / (VP + FP)    « quand je dis "va partir", ai-je raison ? »
        rappel    = VP / (VP + FN)    « parmi ceux qui partent, combien ai-je trouvés ? »

    Si un dénominateur vaut 0, la métrique correspondante vaut 0.0.
    Retourne (précision, rappel).

    Indice : y_true = np.asarray(y_true), puis ((y_true == 1) & (y_pred == 1)).sum()
    """
    raise NotImplementedError


def accuracy_modele_naif(y_true) -> float:
    """Défi 7 🟢 (Parcours 4.2) — Accuracy d'un modèle qui prédit TOUJOURS la classe majoritaire.

    C'est la barre minimale à battre : avec 35 % de churn, ce modèle « idiot » fait déjà
    65 % d'accuracy. Voilà pourquoi l'accuracy seule est trompeuse.
    """
    raise NotImplementedError


def cout_total(y_true, proba, seuil: float, cout_fn: float, cout_fp: float) -> float:
    """Défi 8 🟡 (Parcours 4.3) — Coût métier d'un seuil de décision.

    Un client est ciblé par une action de rétention si proba >= seuil.
        coût = (nb de clients partis NON ciblés) × cout_fn
             + (nb de clients ciblés qui seraient restés) × cout_fp
    """
    raise NotImplementedError


# ===================================================================== Module 3 — Modélisation


def validation_croisee_auc(
    df: pd.DataFrame, model_name: str = "logistic", k: int = 5, seed: int = 42
) -> tuple[float, float]:
    """Défi 9 🔴 (Parcours 3.4) — ROC-AUC moyenne et écart-type sur k plis stratifiés.

    Outils :
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from churn.data import FEATURES, TARGET
        from churn.model import build_pipeline

    Utilisez StratifiedKFold(n_splits=k, shuffle=True, random_state=seed),
    scoring="roc_auc", X = df[FEATURES], y = df[TARGET].
    Retourne (moyenne, écart-type) des k scores (écart-type = np.std).

    Pourquoi ? Un seul découpage train/test peut être chanceux. L'écart-type vous dit si
    la différence entre deux modèles est réelle ou due au hasard.
    """
    raise NotImplementedError


def lift_top(y_true, proba, part: float = 0.1) -> float:
    """Défi 10 🔴 — Lift du top `part` des clients les plus à risque.

        lift = taux de churn parmi les clients aux plus fortes probas
             / taux de churn global

    Nombre de clients retenus : max(1, round(part × n)).

    Un lift de 3 sur le top 10 % = cibler ces clients est 3 fois plus efficace que cibler
    au hasard. C'est LA métrique qu'un responsable marketing comprend immédiatement.

    Indice : np.argsort(-proba) range les indices de la plus forte à la plus faible proba.
    """
    raise NotImplementedError

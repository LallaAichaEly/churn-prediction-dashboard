"""Série 01 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_01.py`.

Il n'y a pas une seule bonne réponse : si votre version passe les vérifications, elle est
juste. Comparez-la à celle-ci pour découvrir d'autres façons d'écrire la même idée, puis
notez dans votre JOURNAL.md ce que vous avez appris.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# =========================================================================== Module 1 — Données


def taux_churn_par(df: pd.DataFrame, colonne: str) -> pd.Series:
    # La moyenne d'une colonne 0/1 est directement une proportion : pas besoin de compter.
    return df.groupby(colonne)["churn"].mean().sort_values(ascending=False)


def taux_manquants(df: pd.DataFrame) -> pd.Series:
    taux = df.isna().mean()  # True compte pour 1, False pour 0 → proportion de manquants
    return taux[taux > 0].sort_values(ascending=False)


def valeurs_aberrantes(serie: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    # Comparer une Series à un nombre garde l'index d'origine : c'est exactement ce qu'on veut.
    return (serie < q1 - k * iqr) | (serie > q3 + k * iqr)


# ========================================================================= Module 2 — Variables


def ajouter_nouveau_client(df: pd.DataFrame, seuil_mois: int = 6) -> pd.DataFrame:
    out = df.copy()  # ← la ligne qui évite de modifier l'entrée de l'appelant
    out["is_new_customer"] = (out["tenure_months"] < seuil_mois).astype(int)
    return out
    # Variante sans copie explicite : return df.assign(is_new_customer=...)


def standardiser(train: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series]:
    # « fit » : on apprend les paramètres sur le train SEULEMENT…
    moyenne = train.mean()
    ecart_type = train.std(ddof=0)  # pandas utilise ddof=1 par défaut, scikit-learn ddof=0
    # … « transform » : on applique LES MÊMES paramètres aux deux jeux.
    return (train - moyenne) / ecart_type, (test - moyenne) / ecart_type
    # Erreur fréquente : calculer la moyenne sur pd.concat([train, test]). Le score paraît
    # meilleur pendant le développement… et s'effondre en production. C'est la fuite de données.


# ======================================================================= Module 4 — Évaluation


def precision_rappel(y_true, y_pred) -> tuple[float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = vp / (vp + fp) if vp + fp else 0.0
    rappel = vp / (vp + fn) if vp + fn else 0.0
    return precision, rappel


def accuracy_modele_naif(y_true) -> float:
    taux = float(np.mean(y_true))
    # Si moins de 50 % de 1, le modèle naïf prédit 0 et a raison sur tous les 0.
    return max(taux, 1 - taux)


def cout_total(y_true, proba, seuil: float, cout_fn: float, cout_fp: float) -> float:
    y_true = np.asarray(y_true)
    cible = np.asarray(proba) >= seuil
    rates = ((y_true == 1) & ~cible).sum()       # partis sans qu'on agisse
    inutiles = ((y_true == 0) & cible).sum()     # actions de rétention gaspillées
    return float(rates * cout_fn + inutiles * cout_fp)


# ===================================================================== Module 3 — Modélisation


def validation_croisee_auc(
    df: pd.DataFrame, model_name: str = "logistic", k: int = 5, seed: int = 42
) -> tuple[float, float]:
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from churn.data import FEATURES, TARGET
    from churn.model import build_pipeline

    # Le pipeline complet est ré-entraîné sur chaque pli : le scaler aussi.
    # Pas de fuite possible, c'est tout l'intérêt de tout mettre dans un Pipeline.
    plis = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = cross_val_score(
        build_pipeline(model_name, seed=seed), df[FEATURES], df[TARGET], cv=plis, scoring="roc_auc"
    )
    return float(scores.mean()), float(scores.std())


def lift_top(y_true, proba, part: float = 0.1) -> float:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    n_top = max(1, int(round(part * len(y_true))))
    top = np.argsort(-proba, kind="stable")[:n_top]  # indices des clients les plus à risque
    return float(y_true[top].mean() / y_true.mean())

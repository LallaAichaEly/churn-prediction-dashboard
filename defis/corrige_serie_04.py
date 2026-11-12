"""Série 04 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_04.py`.

Si votre version passe les vérifications, elle est juste. Comparez-la à celle-ci pour
découvrir d'autres écritures, puis notez dans JOURNAL.md ce que vous avez appris.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ======================================================================= Module 3 — Modéliser


def poids_classes(y) -> dict:
    classes, effectifs = np.unique(np.asarray(y), return_counts=True)
    n, k = int(effectifs.sum()), len(classes)
    # Une classe rare reçoit un grand poids : chaque classe pèse au total n / k.
    return {c.item(): n / (k * int(e)) for c, e in zip(classes, effectifs)}
    # Vérification : sklearn.utils.class_weight.compute_class_weight("balanced", classes=..., y=...)
    # donne les mêmes valeurs. Et LogisticRegression(class_weight=poids_classes(y)) apprend
    # exactement le même modèle que class_weight="balanced".


def choisir_C(X: pd.DataFrame, y, valeurs_C, n_plis: int = 5, seed: int = 0) -> tuple[pd.DataFrame, float]:
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from churn.model import build_pipeline

    # UN seul objet de découpage, réutilisé pour chaque C : shuffle=True + random_state fixe
    # produit les mêmes plis à chaque appel. Tous les C passent donc le même examen.
    cv = StratifiedKFold(n_splits=n_plis, shuffle=True, random_state=seed)
    lignes = []
    for C in valeurs_C:
        modele = build_pipeline("logistic").set_params(clf__C=C)  # clf__ : l'étape "clf" du Pipeline
        scores = cross_val_score(modele, X, y, cv=cv, scoring="roc_auc")
        lignes.append({"C": float(C), "auc_moyenne": scores.mean(), "auc_ecart_type": scores.std()})
    table = pd.DataFrame(lignes, columns=["C", "auc_moyenne", "auc_ecart_type"])

    meilleur = table.loc[table["auc_moyenne"].idxmax()]
    # L'écart-type mesure la dispersion d'UN pli ; l'incertitude sur la MOYENNE de n_plis plis
    # est plus petite : l'erreur standard, écart-type / √n_plis. (Approximation : les plis
    # partagent une partie de leurs données d'entraînement, ils ne sont pas indépendants.)
    tolerance = meilleur["auc_ecart_type"] / math.sqrt(n_plis)
    acceptables = table[table["auc_moyenne"] >= meilleur["auc_moyenne"] - tolerance]
    return table, float(acceptables["C"].min())  # le plus petit C = le modèle le plus régularisé
    # Avec GridSearchCV, la même règle s'écrit grâce à `refit=` qui accepte une fonction
    # recevant cv_results_ et renvoyant l'indice du modèle choisi.


# ================================================================ Module 6 — Industrialiser


def versions_compatibles(sauvegardee: str | None, installee: str) -> bool:
    if not sauvegardee:
        return False  # version inconnue : on ne peut rien garantir, on prévient
    # Comparer des ENTIERS : "1.10" > "1.9" est faux en texte, et float("1.10") == 1.1.
    majeur_mineur = [tuple(int(p) for p in v.split(".")[:2]) for v in (sauvegardee, installee)]
    return majeur_mineur[0] == majeur_mineur[1]
    # Les versions réelles peuvent contenir des suffixes ("1.6.0rc1", "1.5.2.post1") : en
    # production, utilisez `packaging.version.Version(v).release[:2]`, qui les gère tous.


def predire_lot(pipeline, clients: list[dict], seuil: float = 0.5) -> list[dict]:
    from churn.data import FEATURES

    if not clients:
        return []  # rien à prédire, et scikit-learn refuse un tableau de 0 ligne
    for i, client in enumerate(clients):
        manquantes = [f for f in FEATURES if client.get(f) is None]
        if manquantes:
            raise ValueError(f"client n°{i} : variable(s) manquante(s) {manquantes}")
    # pd.DataFrame(clients) crée un NOUVEAU tableau : les dictionnaires de l'appelant ne sont
    # pas touchés. [FEATURES] range les colonnes dans l'ordre d'entraînement et retire les
    # clés en trop comme "customer_id".
    lignes = pd.DataFrame(clients)[FEATURES]
    probas = pipeline.predict_proba(lignes)[:, 1]  # UN appel pour tout le lot
    return [{"churn_probability": float(p), "at_risk": bool(p >= seuil)} for p in probas]


# ===================================================================== Module 7 — Surveillance


def alerte_derive(rapport: pd.DataFrame, seuil: float = 0.25) -> tuple[int, list[str]]:
    # sort_values renvoie une COPIE triée : `rapport` reste intact. na_position="last" range
    # les PSI manquants après les vraies dérives.
    trie = rapport.sort_values("psi", ascending=False, na_position="last")
    alerte = (trie["psi"] >= seuil) | trie["psi"].isna()  # NaN >= seuil vaut False : à part
    variables = trie.loc[alerte, "feature"].tolist()
    return int(bool(variables)), variables
    # Dans scripts/monitor.py : `code, variables = alerte_derive(rapport)` puis `sys.exit(code)`.

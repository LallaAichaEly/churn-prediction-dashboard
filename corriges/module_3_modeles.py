"""Corrigé du module 3 — Modéliser.

Lancer : python corriges/module_3_modeles.py   (≈ 1 minute)
"""

from __future__ import annotations

import pandas as pd
from _commun import donnees, titre
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from churn.data import FEATURES, TARGET
from churn.features import build_preprocessor
from churn.model import evaluate, split

SEED = 42


def make(clf) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


# 3.2 — Dans votre version, ajoutez une branche `elif model_name == "forest":` dans
# `build_pipeline` et une entrée "forest" dans le dictionnaire MODELS.
CANDIDATS = {
    "logistic": LogisticRegression(max_iter=1000),
    "gboost": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED),
    "forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
}


def main() -> None:
    df = donnees()
    X, y = df[FEATURES], df[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    titre("3.2 / 3.4 — Comparaison en validation croisée à 5 plis (ROC-AUC)")
    lignes, scores = [], {}
    for nom, clf in CANDIDATS.items():
        scores[nom] = cross_val_score(make(clf), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        lignes.append({"modèle": nom, "moyenne": scores[nom].mean(), "écart-type": scores[nom].std(),
                       "plis": " ".join(f"{s:.3f}" for s in scores[nom])})
    print(pd.DataFrame(lignes).round(4).to_string(index=False))
    # Les plis sont les MÊMES pour tous les modèles : on compare donc pli par pli (test apparié).
    for autre in ["gboost", "forest"]:
        diff = scores["logistic"] - scores[autre]
        print(f"logistic - {autre:7s}: écart {diff.mean():+.4f} ± {diff.std():.4f}, "
              f"la logistique gagne sur {(diff > 0).sum()}/5 plis")
    print("→ Un écart constant sur tous les plis est un signal fiable, pas un coup de chance.")

    titre("3.5 — GridSearchCV sur le gradient boosting")
    grille = {
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__max_depth": [2, 3, None],
        "clf__max_iter": [100, 300],
        "clf__l2_regularization": [0.0, 1.0],
    }
    search = GridSearchCV(make(HistGradientBoostingClassifier(random_state=SEED)), grille,
                          cv=cv, scoring="roc_auc", n_jobs=-1).fit(X, y)
    print(f"Meilleurs paramètres : {search.best_params_}")
    print(f"ROC-AUC (CV) du boosting optimisé : {search.best_score_:.4f}")
    print("→ Des arbres peu profonds (max_depth=2-3) se rapprochent d'un modèle additif... comme la logistique.")

    titre("3.6 — Classes déséquilibrées : class_weight='balanced' (seuil 0,5, jeu de test)")
    X_train, X_test, y_train, y_test = split(df, seed=SEED)
    for cw in [None, "balanced"]:
        pipe = make(LogisticRegression(max_iter=1000, class_weight=cw)).fit(X_train, y_train)
        m = evaluate(y_test, pipe.predict_proba(X_test)[:, 1])
        print(f"class_weight={str(cw):9s} AUC {m.roc_auc:.3f} | précision {m.precision:.1%} | "
              f"rappel {m.recall:.1%} | signalés {m.tp + m.fp}")
    print("→ Même AUC (même classement), mais le seuil 0,5 « glisse » : plus de rappel, moins de précision.")
    print("  class_weight revient à déplacer le seuil. Choisir le seuil avec les coûts métier est plus explicite.")


if __name__ == "__main__":
    main()

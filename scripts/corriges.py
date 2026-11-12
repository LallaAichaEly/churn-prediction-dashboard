"""Reproduit tous les chiffres cités dans CORRIGES.md.

⚠️ Faites les exercices de PARCOURS.md AVANT de lancer ce script : il donne les réponses.

Usage :
    python scripts/corriges.py            # tous les modules
    python scripts/corriges.py --module 3 # un seul module
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler  # noqa: E402

from churn.data import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_or_create,
)
from churn.features import build_preprocessor  # noqa: E402
from churn.model import (  # noqa: E402
    best_threshold_by_cost,
    build_pipeline,
    evaluate,
    feature_importance,
    split,
    train,
)

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def title(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def module_1(df: pd.DataFrame) -> None:
    title("Module 1 — Comprendre les données")
    print(f"1.1  Taux de churn global : {df[TARGET].mean():.1%}  ({len(df)} clients)")
    print(df[NUMERIC_FEATURES].describe().round(1).to_string())
    print("\n1.2  Taux de churn par contrat :")
    print(df.groupby("contract")[TARGET].agg(["mean", "size"]).round(3).to_string())
    print("\n1.2  Taux de churn par service internet :")
    print(df.groupby("internet_service")[TARGET].agg(["mean", "size"]).round(3).to_string())
    print(f"\n1.3  Factures hors [0, 500] € : {((df['monthly_charges'] < 0) | (df['monthly_charges'] > 500)).sum()}")


def module_2(df: pd.DataFrame) -> None:
    title("Module 2 — Préparer les variables")
    for eng in (False, True):
        m = train(df, "logistic", eng)[-1]
        print(f"2.2  Logistique, variables dérivées={eng!s:5} → ROC-AUC {m.roc_auc:.3f}")

    # 2.3 — is_new_customer, testé en validation croisée pour ne pas conclure sur du bruit.
    def add_new(d: pd.DataFrame) -> pd.DataFrame:
        return d.assign(is_new_customer=(d["tenure_months"] < 6).astype(int))

    X, y = df[FEATURES], df[TARGET]
    base = build_pipeline("logistic")
    with_new = Pipeline([
        ("new", FunctionTransformer(add_new)),
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES + ["is_new_customer"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    for name, pipe in [("sans is_new_customer", base), ("avec is_new_customer", with_new)]:
        s = cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc")
        print(f"2.3  {name:22} → ROC-AUC CV {s.mean():.4f} ± {s.std():.4f}")

    # 2.5 — Fuite de données. (a) avec un simple scaler, (b) avec une sélection de variables
    # faite sur TOUT le jeu, en présence de nombreuses variables de bruit.
    X_train, X_test, y_train, y_test = split(df)
    prep_all = build_preprocessor().fit(df[FEATURES])  # ❌ appris sur train + test
    clf = LogisticRegression(max_iter=1000).fit(prep_all.transform(X_train), y_train)
    leak_auc = roc_auc_score(y_test, clf.predict_proba(prep_all.transform(X_test))[:, 1])
    clean_auc = train(df)[-1].roc_auc
    print(f"2.5a Scaler appris avant le split : AUC {leak_auc:.4f}  vs  pipeline propre : {clean_auc:.4f}")

    rng = np.random.default_rng(SEED)
    small = df.sample(300, random_state=SEED)
    noise = rng.normal(size=(len(small), 2000))  # 2000 variables PUREMENT aléatoires
    y_small = small[TARGET].to_numpy()
    top = SelectKBest(f_classif, k=20).fit(noise, y_small)  # ❌ sélection sur tout le jeu
    leaky = cross_val_score(LogisticRegression(max_iter=1000), top.transform(noise), y_small,
                            cv=CV, scoring="roc_auc").mean()
    clean = cross_val_score(Pipeline([("sel", SelectKBest(f_classif, k=20)),
                                      ("clf", LogisticRegression(max_iter=1000))]),
                            noise, y_small, cv=CV, scoring="roc_auc").mean()
    print(f"2.5b Bruit pur — sélection AVANT la CV : AUC {leaky:.3f}  vs  DANS le pipeline : {clean:.3f}")


def module_3(df: pd.DataFrame) -> None:
    title("Module 3 — Modéliser")
    X, y = df[FEATURES], df[TARGET]
    rf = Pipeline([("prep", build_preprocessor()),
                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                  random_state=SEED, n_jobs=-1))])
    models = {"logistic": build_pipeline("logistic"), "gboost": build_pipeline("gboost"), "random_forest": rf}

    X_train, X_test, y_train, y_test = split(df)
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        tr = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])
        te = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        print(f"3.2/3.3  {name:14} AUC train {tr:.3f} | test {te:.3f} | écart {tr - te:+.3f}")

    print()
    for name, pipe in models.items():
        s = cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc")
        print(f"3.4  {name:14} CV 5 plis : {s.mean():.3f} ± {s.std():.3f}   {np.round(s, 3)}")

    grid = GridSearchCV(
        Pipeline([("prep", build_preprocessor()),
                  ("clf", HistGradientBoostingClassifier(random_state=SEED))]),
        {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3, None],
         "clf__max_iter": [100, 300], "clf__l2_regularization": [0.0, 1.0]},
        cv=CV, scoring="roc_auc", n_jobs=-1,
    ).fit(X, y)
    print(f"\n3.5  Meilleur boosting (GridSearchCV) : AUC CV {grid.best_score_:.3f}  {grid.best_params_}")

    print()
    for cw in (None, "balanced"):
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000, class_weight=cw))])
        pipe.fit(X_train, y_train)
        m = evaluate(y_test, pipe.predict_proba(X_test)[:, 1])
        print(f"3.6  class_weight={cw!s:9} → précision {m.precision:.3f}  rappel {m.recall:.3f}  "
              f"F1 {m.f1:.3f}  AUC {m.roc_auc:.3f}")


def module_4(df: pd.DataFrame) -> None:
    title("Module 4 — Évaluer et décider")
    pipeline, X_test, y_test, proba, _ = train(df)
    for t in (0.3, 0.5, 0.7):
        m = evaluate(y_test, proba, t)
        print(f"4.1  seuil {t:.1f} → précision {m.precision:.3f}  rappel {m.recall:.3f}  "
              f"signalés {m.tp + m.fp}")
    print(f"\n4.2  Modèle « tout le monde reste » : accuracy {1 - y_test.mean():.3f}, rappel 0")

    for cfn, cfp in [(500, 50), (100, 100)]:
        t = best_threshold_by_cost(y_test, proba, cfn, cfp)
        m = evaluate(y_test, proba, t)
        m50 = evaluate(y_test, proba, 0.5)
        print(f"4.3  {cfn} €/{cfp} € → seuil {t:.2f} | coût {m.fn * cfn + m.fp * cfp:,.0f} € "
              f"(vs {m50.fn * cfn + m50.fp * cfp:,.0f} € à 0,5) | précision {m.precision:.2f} rappel {m.recall:.2f}")

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    print(f"\n4.5  Brier score : {brier_score_loss(y_test, proba):.4f}")
    print("     proba moyenne prédite → taux de churn observé")
    for p, f in zip(mean_pred, frac_pos):
        print(f"       {p:.2f} → {f:.2f}")


def module_5(df: pd.DataFrame) -> None:
    title("Module 5 — Interpréter")
    pipeline, X_test, y_test, _, _ = train(df)
    print("5.1  Importance par permutation (logistique) :")
    print(feature_importance(pipeline, X_test, y_test).round(4).to_string(index=False))

    # 5.3 — contribution de chaque variable au logit pour UN client.
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    client = X_test.iloc[[0]]
    contrib = pd.Series(prep.transform(client).ravel() * clf.coef_.ravel(), index=prep.get_feature_names_out())
    print(f"\n5.3  Client {client.index[0]} — proba {pipeline.predict_proba(client)[0, 1]:.2f}, "
          f"intercept {clf.intercept_[0]:+.2f}. Top contributions au logit :")
    print(contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5).round(3).to_string())

    # Auto-évaluation — chiffres pour argumenter des actions de rétention.
    print("\nAuto-éval  Taux de churn par support technique / moyen de paiement / appels au support :")
    for col in ("tech_support", "payment_method"):
        print(df.groupby(col)[TARGET].agg(["mean", "size"]).round(3).to_string(), "\n")
    calls = pd.cut(df["support_calls"], [-1, 1, 2, 100], labels=["0-1", "2", "3+"])
    print(df.groupby(calls, observed=True)[TARGET].agg(["mean", "size"]).round(3).to_string())
    monthly = df[df["contract"] == "Mensuel"]
    print(f"\n           Revenu mensuel des clients « Mensuel » qui partent : "
          f"{monthly.loc[monthly[TARGET] == 1, 'monthly_charges'].sum():,.0f} €")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=int, choices=range(1, 6))
    args = parser.parse_args()
    df = load_or_create(ROOT / "data" / "customers.csv")
    modules = {1: module_1, 2: module_2, 3: module_3, 4: module_4, 5: module_5}
    for k, fn in modules.items():
        if args.module in (None, k):
            fn(df)


if __name__ == "__main__":
    main()

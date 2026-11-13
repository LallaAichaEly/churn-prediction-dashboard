"""Les corrigés sont du code comme un autre : on vérifie qu'ils restent justes."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "corriges"), str(ROOT / "src")]

from churn.data import generate_customers  # noqa: E402
from churn.model import split, train  # noqa: E402
from module_1_donnees import test_validate_rejette_facture_aberrante  # noqa: E402, F401
from module_5_interpretation import explain_logistic  # noqa: E402


def test_contributions_add_up_to_the_logit():
    """Somme des contributions = logit(client) − logit(client moyen) : c'est un modèle linéaire."""
    df = generate_customers(800, seed=3)
    pipeline, X_test, *_ = train(df)
    X_train, *_ = split(df)
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    client = X_test.iloc[[0]]
    logit_client = clf.decision_function(prep.transform(client))[0]
    logit_moyen = clf.intercept_[0] + clf.coef_[0] @ np.asarray(prep.transform(X_train).mean(axis=0)).ravel()
    contributions = explain_logistic(pipeline, client, X_train)
    assert np.isclose(contributions.sum(), logit_client - logit_moyen)
    assert set(contributions.index) == set(X_train.columns)

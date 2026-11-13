"""Module 6 — Les tests. Un code de data science non testé est un code qui casse en silence."""

import json

import numpy as np
import pandas as pd
import pytest

from churn.data import FEATURES, TARGET, generate_customers, validate
from churn.features import add_engineered_features
from churn.model import best_threshold_by_cost, build_pipeline, evaluate, load, save, train

# La fixture `df` (1 500 clients, seed=0) est définie dans `conftest.py`.


def test_generation_is_reproducible():
    pd.testing.assert_frame_equal(generate_customers(200, seed=1), generate_customers(200, seed=1))


def test_generated_data_is_valid(df):
    assert validate(df) == []
    assert 0.15 < df[TARGET].mean() < 0.6, "taux de churn irréaliste"


def test_validate_detects_problems(df):
    broken = df.copy()
    broken.loc[0, "tenure_months"] = -3
    broken.loc[1, "monthly_charges"] = np.nan
    problems = validate(broken)
    assert any("négative" in p for p in problems)
    assert any("manquantes" in p for p in problems)


def test_engineered_features_do_not_mutate_input(df):
    before = df.copy()
    out = add_engineered_features(df)
    pd.testing.assert_frame_equal(df, before)
    assert {"avg_charge_per_month", "calls_per_year"} <= set(out.columns)


@pytest.mark.parametrize("model_name", ["logistic", "gboost"])
@pytest.mark.parametrize("engineered", [False, True])
def test_models_beat_random(df, model_name, engineered):
    _, _, _, proba, metrics = train(df, model_name, engineered)
    assert proba.min() >= 0 and proba.max() <= 1
    assert metrics.roc_auc > 0.7


def test_model_learns_true_signal(df):
    """Le contrat mensuel augmente le churn dans la « vérité cachée » : le modèle doit le savoir."""
    pipeline, X_test, *_ = train(df)
    base = X_test.iloc[[0]].copy()
    monthly, two_year = base.copy(), base.copy()
    monthly["contract"], two_year["contract"] = "Mensuel", "Biannuel"
    assert pipeline.predict_proba(monthly)[0, 1] > pipeline.predict_proba(two_year)[0, 1]


def test_evaluate_confusion_matrix_sums():
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.8, 0.4, 0.9]), threshold=0.5)
    assert (m.tn, m.fp, m.fn, m.tp) == (1, 1, 1, 1)
    assert m.precision == 0.5 and m.recall == 0.5


def test_costly_misses_lower_the_threshold(df):
    _, _, y_test, proba, _ = train(df)
    cheap_miss = best_threshold_by_cost(y_test, proba, cost_fn=10, cost_fp=100)
    costly_miss = best_threshold_by_cost(y_test, proba, cost_fn=1000, cost_fp=10)
    assert costly_miss < cheap_miss


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_pipeline("magic")


def test_unknown_category_is_ignored(df):
    """En production, une nouvelle modalité ne doit pas faire planter la prédiction."""
    pipeline, X_test, *_ = train(df)
    row = X_test.iloc[[0]].copy()
    row["payment_method"] = "Crypto"
    assert 0 <= pipeline.predict_proba(row[FEATURES])[0, 1] <= 1


@pytest.mark.parametrize("engineered", [False, True])
def test_saved_model_gives_same_predictions(df, tmp_path, engineered):
    """Ce qui est sauvegardé doit prédire EXACTEMENT comme ce qui a été entraîné."""
    pipeline, X_test, _, proba, metrics = train(df, "logistic", engineered)
    path = save(pipeline, metrics, model_dir=tmp_path, model_name="logistic")

    np.testing.assert_allclose(load(path).predict_proba(X_test)[:, 1], proba)
    card = json.loads((tmp_path / "churn_model_metrics.json").read_text())
    assert card["model"] == "logistic"
    assert card["features"] == FEATURES
    assert card["roc_auc"] == pytest.approx(metrics.roc_auc)

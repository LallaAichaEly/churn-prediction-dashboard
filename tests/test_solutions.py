"""Les corrigés font des affirmations (« la fuite gonfle l'AUC », « balanced augmente le rappel »…).
Ces tests vérifient qu'elles restent vraies si le code ou les données changent."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from churn.data import generate_customers  # noqa: E402
from churn.model import split, train  # noqa: E402
from solutions import (  # noqa: E402
    module1_donnees as m1,
    module2_variables as m2,
    module3_modeliser as m3,
    module4_evaluer as m4,
    module5_interpreter as m5,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate_customers(n=3000, seed=7)


def test_m1_monthly_contract_churns_most(df):
    rates = m1.churn_rate_by(df, "contract")
    assert rates.index[0] == "Mensuel"
    assert rates["clients"].sum() == len(df)


def test_m1_validate_v2_flags_absurd_invoice(df):
    assert m1.validate_v2(df) == []
    broken = df.copy()
    broken.loc[0, "monthly_charges"] = 600
    assert any("Facture" in p for p in m1.validate_v2(broken))


def test_m2_target_encoding_leak_inflates_then_oof_fixes_it(df):
    auc = m2.target_encoding_leakage(df)
    honest = auc["TargetEncoder hors-pli"]
    assert auc["encodage cible sur tout le jeu"] > honest + 0.05  # fuite → score gonflé
    assert auc["encodage sur le train (naïf)"] < honest - 0.05  # fuite interne → modèle trompé


def test_m2_scaler_leak_is_negligible_here(df):
    auc = m2.scaler_leakage(df)
    assert abs(auc["honnête"] - auc["scaler avant split"]) < 0.01


def test_m3_cross_validation_shape(df):
    scores = m3.cross_validate_models(df)
    assert scores.shape == (5, 3)
    assert (scores > 0.7).all().all()


def test_m3_stumps_make_boosting_competitive(df):
    params, auc_boost, auc_logistic = m3.tune_boosting(df, quick=True)
    assert params["max_depth"] == 1  # modèle additif, comme le vrai processus
    assert auc_boost > auc_logistic - 0.02


def test_m3_balanced_weights_trade_precision_for_recall(df):
    table = m3.class_weight_effect(df)
    plain, balanced = table.iloc[0], table.iloc[1]
    assert balanced["recall"] > plain["recall"]
    assert balanced["precision"] < plain["precision"]
    assert balanced["roc_auc"] == pytest.approx(plain["roc_auc"], abs=0.01)  # le classement change à peine


@pytest.fixture(scope="module")
def trained(df):
    return train(df, "logistic")


def test_m4_dummy_model_has_good_accuracy_but_zero_recall(trained):
    _, _, y_test, _, _ = trained
    base = m4.dummy_baseline(y_test)
    assert base["accuracy"] == pytest.approx(1 - y_test.mean())
    assert base["recall"] == 0


def test_m4_optimal_threshold_matches_theory(trained):
    _, _, y_test, proba, _ = trained
    table = m4.cost_scenarios(y_test, proba, {"a": (500, 50), "b": (100, 100)})
    assert table.loc["a", "seuil_optimal"] < table.loc["b", "seuil_optimal"]
    assert (table["seuil_optimal"] - table["seuil_théorique"]).abs().max() < 0.12  # modèle bien calibré
    assert (table["coût_optimal"] <= table["coût_seuil_0.5"]).all()


def test_m4_pr_curve_beats_chance(trained):
    _, _, y_test, proba, _ = trained
    curve, ap = m4.pr_curve(y_test, proba)
    assert ap > y_test.mean() + 0.2
    assert curve["precision"].between(0, 1).all()


def test_m4_logistic_is_reasonably_calibrated(trained):
    _, _, y_test, proba, _ = trained
    assert m4.calibration_table(y_test, proba)["écart"].abs().mean() < 0.06


def test_m5_recovers_signs_of_true_effects(trained):
    pipeline = trained[0]
    effects = m5.recovered_effects(pipeline).drop(index="total_charges (par €)")  # vrai effet nul
    assert (np.sign(effects["vrai_effet"]) == np.sign(effects["effet_appris"])).all()


def test_m5_contributions_sum_to_logit_difference(df, trained):
    pipeline, X_test, *_ = trained
    background = split(df)[0]
    client = X_test.iloc[[0]]
    contrib = m5.explain(pipeline, client, background)

    logit_client = pipeline.decision_function(client)[0]
    clf = pipeline.named_steps["clf"]
    logit_mean = clf.intercept_[0] + clf.coef_[0] @ m5._transform(pipeline, background).mean(axis=0)
    assert contrib.sum() == pytest.approx(logit_client - logit_mean)
    assert set(contrib.index) <= set(client.columns)

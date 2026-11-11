"""Module 7 — Tests de la surveillance : le PSI doit se taire quand rien ne change et parler quand ça dérive."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.data import FEATURES, TARGET, generate_customers  # noqa: E402
from churn.model import split, train  # noqa: E402
from churn.monitoring import PSI_DRIFT, PSI_WATCH, drift_report, psi, simulate_batch, status  # noqa: E402


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return generate_customers(n=4000, seed=0)


def test_psi_is_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) == pytest.approx(0, abs=1e-12)
    assert psi(["a", "b", "b"], ["a", "b", "b"]) == pytest.approx(0, abs=1e-12)


def test_psi_matches_hand_computation():
    """50/50 → 70/30 : (0,7−0,5)·ln(0,7/0,5) + (0,3−0,5)·ln(0,3/0,5) ≈ 0,1695."""
    value = psi(["a"] * 50 + ["b"] * 50, ["a"] * 70 + ["b"] * 30)
    assert value == pytest.approx(0.2 * np.log(0.7 / 0.5) - 0.2 * np.log(0.3 / 0.5))


def test_psi_flags_a_new_category():
    """Une modalité jamais vue (nouveau moyen de paiement) doit faire exploser le PSI."""
    assert psi(["Carte"] * 100, ["Carte"] * 70 + ["Crypto"] * 30) > PSI_DRIFT


def test_status_thresholds():
    assert [status(v) for v in (0.0, PSI_WATCH, PSI_DRIFT)] == ["stable", "à surveiller", "dérive"]


def test_no_drift_means_stable_report(reference):
    report = drift_report(reference[FEATURES], generate_customers(n=4000, seed=1))
    assert list(report.columns) == ["feature", "psi", "statut"]
    assert set(report["feature"]) == set(FEATURES)
    assert (report["statut"] == "stable").all()


def test_price_increase_is_detected_on_the_right_variable(reference):
    report = drift_report(reference, simulate_batch(price_increase=0.25)).set_index("feature")
    assert report.index[0] == "monthly_charges"
    assert report.loc["monthly_charges", "statut"] == "dérive"
    assert report.loc["contract", "statut"] == "stable"


def test_contract_mix_change_is_detected(reference):
    report = drift_report(reference, simulate_batch(monthly_share=0.85))
    assert report.iloc[0]["feature"] == "contract"
    assert report.iloc[0]["psi"] > PSI_WATCH


def test_default_generator_is_unchanged_by_drift_options():
    """Les options du module 7 ne doivent pas modifier les données des modules 1 à 6."""
    pd.testing.assert_frame_equal(
        generate_customers(300, seed=5),
        generate_customers(300, seed=5, contract_probs=(0.55, 0.25, 0.20), price_increase=0.0, fibre_competition=0.0),
    )


def test_concept_drift_is_invisible_to_input_psi_but_breaks_calibration(reference):
    """La leçon du module 7 : le PSI des entrées ne voit pas la dérive du concept."""
    pipeline, *_ = train(reference)
    before, after = simulate_batch(), simulate_batch(fibre_competition=1.5)

    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])  # mêmes clients…
    assert (drift_report(split(reference)[0], after)["statut"] == "stable").all()

    predicted = pipeline.predict_proba(after[FEATURES])[:, 1].mean()
    assert after[TARGET].mean() - predicted > 0.05  # … mais bien plus de départs que prévu


@pytest.mark.parametrize("kwargs", [{"contract_probs": (0.5, 0.5, 0.5)}, {"price_increase": -1.0}])
def test_generator_rejects_invalid_drift_options(kwargs):
    with pytest.raises(ValueError):
        generate_customers(10, **kwargs)

"""Module 7 — Tests de la surveillance du drift."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.data import FEATURES, TARGET, generate_customers  # noqa: E402
from churn.monitoring import (  # noqa: E402
    drift_report,
    drift_status,
    psi_categorical,
    psi_numeric,
    simulate_new_period,
)


@pytest.fixture(scope="module")
def reference():
    return generate_customers(n=4000, seed=0)


def test_psi_is_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=2000)
    assert psi_numeric(x, x) == pytest.approx(0)
    assert psi_categorical(["a", "b", "b"], ["a", "b", "b"]) == pytest.approx(0)


def test_psi_grows_with_the_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    small, large = psi_numeric(ref, ref + 0.2), psi_numeric(ref, ref + 1.0)
    assert 0 < small < large
    assert drift_status(large) == "significative"


def test_psi_handles_constant_reference():
    assert psi_numeric([1] * 100, [1] * 50 + [2] * 50) > 0.25


def test_new_category_is_detected():
    assert psi_categorical(["a", "b"] * 50, ["a", "b", "c"] * 50) > 0.25


def test_drift_status_thresholds():
    assert [drift_status(v) for v in (0.05, 0.15, 0.4)] == ["stable", "modérée", "significative"]


def test_no_change_means_no_drift(reference):
    report = drift_report(reference, simulate_new_period(n=4000))
    assert set(report["feature"]) == set(FEATURES)
    assert (report["statut"] == "stable").all()


@pytest.mark.parametrize(
    "change, feature",
    [
        ({"price_increase": 0.3}, "monthly_charges"),
        ({"extra_support_calls": 1.5}, "support_calls"),
        ({"to_monthly": 0.8}, "contract"),
    ],
)
def test_simulated_change_is_flagged(reference, change, feature):
    report = drift_report(reference, simulate_new_period(n=4000, **change))
    assert report.iloc[0]["feature"] == feature
    assert report.iloc[0]["statut"] == "significative"


def test_new_period_has_no_target():
    assert TARGET not in simulate_new_period(n=50).columns


@pytest.mark.parametrize("kwargs", [{"price_increase": -1}, {"to_monthly": 1.5}])
def test_invalid_simulation_parameters(kwargs):
    with pytest.raises(ValueError):
        simulate_new_period(n=10, **kwargs)

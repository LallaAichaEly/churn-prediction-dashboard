"""Module 6 — Tests de l'API : on vérifie le contrat (entrées/sorties), pas seulement le modèle."""

import sys
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api import Customer, create_app  # noqa: E402
from churn.data import CONTRACTS, INTERNET_SERVICES, PAYMENT_METHODS, generate_customers  # noqa: E402
from churn.model import train  # noqa: E402

RISKY = {
    "tenure_months": 2, "monthly_charges": 99.0, "total_charges": 198.0, "support_calls": 5,
    "num_products": 1, "contract": "Mensuel", "payment_method": "Chèque",
    "internet_service": "Fibre", "tech_support": "Non", "senior": "Oui",
}
LOYAL = {
    **RISKY, "tenure_months": 60, "total_charges": 3000.0, "support_calls": 0, "num_products": 4,
    "contract": "Biannuel", "payment_method": "Prélèvement", "tech_support": "Oui", "senior": "Non",
}
AVERAGE = {
    "tenure_months": 12, "monthly_charges": 70.0, "total_charges": 840.0, "support_calls": 1,
    "num_products": 2, "contract": "Annuel", "payment_method": "Carte bancaire",
    "internet_service": "ADSL", "tech_support": "Non", "senior": "Non",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    pipeline, *_ = train(generate_customers(n=1500, seed=0))
    return TestClient(create_app(pipeline))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_probability(client):
    body = client.post("/predict", json=RISKY).json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["threshold"] == 0.5
    assert body["at_risk"] == (body["churn_probability"] >= 0.5)


def test_risky_profile_scores_higher_than_loyal(client):
    risky, loyal = client.post("/predict/batch", json=[RISKY, LOYAL]).json()
    assert risky["churn_probability"] > loyal["churn_probability"]


def test_threshold_is_configurable(client):
    """Le même client (≈ 20 % de risque) est signalé ou non selon le seuil choisi par le métier."""
    low = client.post("/predict", params={"threshold": 0.01}, json=AVERAGE).json()
    high = client.post("/predict", params={"threshold": 0.99}, json=AVERAGE).json()
    assert low["at_risk"] is True and high["at_risk"] is False
    assert low["churn_probability"] == high["churn_probability"]


@pytest.mark.parametrize("threshold", [0, 1, 1.5])
def test_invalid_threshold_is_rejected(client, threshold):
    assert client.post("/predict", params={"threshold": threshold}, json=AVERAGE).status_code == 422


def test_empty_batch(client):
    assert client.post("/predict/batch", json=[]).json() == []


@pytest.mark.parametrize(
    "field, bad_value",
    [("tenure_months", -1), ("contract", "Hebdomadaire"), ("monthly_charges", 10_000), ("senior", "Peut-être")],
)
def test_invalid_input_is_rejected(client, field, bad_value):
    assert client.post("/predict", json={**RISKY, field: bad_value}).status_code == 422


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in RISKY.items() if k != "contract"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_api_categories_match_training_data():
    """Si on ajoute une modalité dans data.py, l'API doit suivre (sinon : 422 en production)."""
    fields = Customer.model_fields
    assert set(get_args(fields["contract"].annotation)) == set(CONTRACTS)
    assert set(get_args(fields["payment_method"].annotation)) == set(PAYMENT_METHODS)
    assert set(get_args(fields["internet_service"].annotation)) == set(INTERNET_SERVICES)

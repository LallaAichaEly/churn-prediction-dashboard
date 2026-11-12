"""Module 6 — Tester une API sans lancer de serveur, grâce au `TestClient` de FastAPI."""

import sys
from pathlib import Path
from typing import get_args

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from churn import api  # noqa: E402
from churn.data import CONTRACTS, INTERNET_SERVICES, PAYMENT_METHODS, generate_customers  # noqa: E402
from churn.model import save, train  # noqa: E402

CLIENT = dict(api.EXAMPLE_CUSTOMER)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory) -> Path:
    """Un petit modèle entraîné et sauvegardé dans un dossier temporaire."""
    pipeline, *_, metrics = train(generate_customers(n=1000, seed=3))
    return save(pipeline, metrics, model_dir=tmp_path_factory.mktemp("models"))


@pytest.fixture
def client(model_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("CHURN_MODEL_PATH", str(model_path))
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info_exposes_metrics(client):
    info = client.get("/model").json()
    assert info["estimator"] == "LogisticRegression"
    assert info["metrics"]["roc_auc"] > 0.7


def test_predict_returns_a_probability(client):
    r = client.post("/predict", json=CLIENT)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["at_risk"] == (body["churn_probability"] >= 0.5)


def test_threshold_is_applied(client):
    assert client.post("/predict", params={"threshold": 0}, json=CLIENT).json()["at_risk"] is True
    assert client.post("/predict", params={"threshold": 1}, json=CLIENT).json()["at_risk"] is False
    assert client.post("/predict", params={"threshold": 1.5}, json=CLIENT).status_code == 422


def test_model_signal_is_visible_through_the_api(client):
    """Bout en bout : un client mensuel doit sembler plus à risque qu'un client biannuel."""
    monthly = client.post("/predict", json={**CLIENT, "contract": "Mensuel"}).json()
    two_year = client.post("/predict", json={**CLIENT, "contract": "Biannuel"}).json()
    assert monthly["churn_probability"] > two_year["churn_probability"]


@pytest.mark.parametrize(
    "field, value",
    [("contract", "Hebdomadaire"), ("tenure_months", -1), ("monthly_charges", 0), ("senior", "Peut-être")],
)
def test_invalid_input_is_rejected(client, field, value):
    assert client.post("/predict", json={**CLIENT, field: value}).status_code == 422


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in CLIENT.items() if k != "contract"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_batch_keeps_order(client):
    customers = [{**CLIENT, "contract": c} for c in CONTRACTS]
    r = client.post("/predict/batch", json={"customers": customers})
    assert r.status_code == 200
    singles = [client.post("/predict", json=c).json() for c in customers]
    assert r.json() == singles


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_missing_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("CHURN_MODEL_PATH", str(tmp_path / "absent.joblib"))
    r = TestClient(api.app).post("/predict", json=CLIENT)
    assert r.status_code == 503
    assert "train.py" in r.json()["detail"]


def test_allowed_values_match_the_data():
    """Si on ajoute un contrat dans data.py sans mettre à jour l'API, ce test le signale."""
    fields = api.Customer.model_fields
    assert get_args(fields["contract"].annotation) == tuple(CONTRACTS)
    assert get_args(fields["payment_method"].annotation) == tuple(PAYMENT_METHODS)
    assert get_args(fields["internet_service"].annotation) == tuple(INTERNET_SERVICES)

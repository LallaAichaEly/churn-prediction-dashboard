"""Projet 2 — Tests de l'API. On teste le CONTRAT (ce que voit un client HTTP), pas l'implémentation."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import create_app, load_or_train_model  # noqa: E402
from api.schemas import EXAMPLE_CUSTOMER  # noqa: E402
from churn.data import FEATURES, generate_customers  # noqa: E402
from churn.model import save, train  # noqa: E402

LOYAL_CUSTOMER = {
    **EXAMPLE_CUSTOMER,
    "tenure_months": 60,
    "support_calls": 0,
    "num_products": 3,
    "contract": "Biannuel",
    "payment_method": "Prélèvement",
    "tech_support": "Oui",
}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory) -> Path:
    """Un petit modèle entraîné dans un dossier temporaire : les tests ne touchent pas à models/."""
    pipeline, *_, metrics = train(generate_customers(n=1500, seed=0))
    return save(pipeline, metrics, model_dir=tmp_path_factory.mktemp("models"))


@pytest.fixture(scope="module")
def client(model_path):
    with TestClient(create_app(model_path=model_path, threshold=0.5)) as c:  # `with` → lance le lifespan
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(client):
    info = client.get("/model").json()
    assert info["model_type"] == "LogisticRegression"
    assert info["features"] == FEATURES
    assert info["threshold"] == 0.5
    assert info["test_metrics"]["roc_auc"] > 0.7


def test_predict_returns_a_probability(client):
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["at_risk"] == (body["churn_probability"] >= body["threshold"])


def test_api_matches_the_trained_pipeline(client, model_path):
    """Garde-fou contre le *training-serving skew* : l'API doit donner la même réponse que le modèle."""
    expected = load_or_train_model(model_path)[0].predict_proba(
        pd.DataFrame([{**EXAMPLE_CUSTOMER, "total_charges": 89.9 * 3}])[FEATURES]
    )[0, 1]
    assert client.post("/predict", json=EXAMPLE_CUSTOMER).json()["churn_probability"] == pytest.approx(
        expected, abs=1e-4
    )


def test_risky_profile_scores_higher_than_loyal_one(client):
    risky = client.post("/predict", json=EXAMPLE_CUSTOMER).json()["churn_probability"]
    loyal = client.post("/predict", json=LOYAL_CUSTOMER).json()["churn_probability"]
    assert risky > loyal


@pytest.mark.parametrize(
    "field, value",
    [
        ("contract", "Hebdomadaire"),  # modalité inconnue
        ("tenure_months", -1),  # hors bornes
        ("monthly_charges", "beaucoup"),  # mauvais type
        ("couleur_preferee", "bleu"),  # champ inattendu
    ],
)
def test_invalid_input_is_rejected_with_422(client, field, value):
    response = client.post("/predict", json={**EXAMPLE_CUSTOMER, field: value})
    assert response.status_code == 422
    assert any(field in error["loc"] for error in response.json()["detail"])


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "contract"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_batch_prediction(client):
    response = client.post("/predict/batch", json={"customers": [EXAMPLE_CUSTOMER, LOYAL_CUSTOMER]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["n_at_risk"] == sum(p["at_risk"] for p in body["predictions"])


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_threshold_must_be_a_probability(model_path):
    with pytest.raises(ValueError):
        create_app(model_path=model_path, threshold=1.5)


def test_missing_model_is_trained_on_first_start(tmp_path):
    model_path = tmp_path / "models" / "churn_model.joblib"
    pipeline, metrics = load_or_train_model(model_path, data_path=tmp_path / "customers.csv")
    assert model_path.exists()
    assert metrics["roc_auc"] > 0.7
    assert hasattr(pipeline, "predict_proba")

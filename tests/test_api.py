"""Tests de l'API : on vérifie le contrat (entrées / sorties), pas seulement « ça tourne »."""

import pytest
from fastapi.testclient import TestClient

from churn.api import create_app
from churn.model import train

CLIENT = {
    "tenure_months": 3,
    "monthly_charges": 95.0,
    "support_calls": 4,
    "num_products": 1,
    "contract": "Mensuel",
    "payment_method": "Chèque",
    "internet_service": "Fibre",
    "tech_support": "Non",
    "senior": "Non",
}


@pytest.fixture(scope="module")
def client(df):
    pipeline, *_ = train(df)
    with TestClient(create_app(pipeline=pipeline, threshold=0.4)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "LogisticRegression"
    assert body["threshold"] == 0.4


def test_predict_returns_probability_and_decision(client):
    response = client.post("/predict", json=CLIENT)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["at_risk"] == (body["churn_probability"] >= 0.4)


def test_risky_profile_scores_higher_than_loyal_one(client):
    loyal = {**CLIENT, "tenure_months": 60, "support_calls": 0, "contract": "Biannuel",
             "payment_method": "Prélèvement", "tech_support": "Oui", "num_products": 4}
    risky = client.post("/predict", json=CLIENT).json()["churn_probability"]
    safe = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky > 0.5 > safe


def test_total_charges_is_optional_and_used_when_given(client):
    implicit = client.post("/predict", json=CLIENT).json()["churn_probability"]
    explicit = client.post("/predict", json={**CLIENT, "total_charges": 3 * 95.0}).json()["churn_probability"]
    assert implicit == pytest.approx(explicit)


@pytest.mark.parametrize(
    "field, value",
    [("contract", "Hebdomadaire"), ("tenure_months", -1), ("monthly_charges", 0), ("senior", "peut-être")],
)
def test_invalid_input_is_rejected(client, field, value):
    response = client.post("/predict", json={**CLIENT, field: value})
    assert response.status_code == 422
    assert field in str(response.json()["detail"])


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in CLIENT.items() if k != "contract"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_batch_matches_single_predictions(client):
    other = {**CLIENT, "contract": "Annuel"}
    batch = client.post("/predict/batch", json={"customers": [CLIENT, other]}).json()
    assert [p["churn_probability"] for p in batch] == [
        client.post("/predict", json=c).json()["churn_probability"] for c in (CLIENT, other)
    ]


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_missing_model_file_fails_fast(tmp_path):
    app = create_app(model_path=tmp_path / "absent.joblib")
    with pytest.raises(RuntimeError, match="introuvable"):
        with TestClient(app):
            pass


def test_invalid_threshold_is_refused():
    with pytest.raises(ValueError):
        create_app(threshold=1.5)

"""Tests de l'API : on vérifie le contrat (entrées, sorties, erreurs), pas seulement le modèle."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api  # noqa: E402

RISKY = {
    "tenure_months": 2, "monthly_charges": 95.0, "support_calls": 5, "num_products": 1,
    "contract": "Mensuel", "payment_method": "Chèque", "internet_service": "Fibre",
    "tech_support": "Non", "senior": "Oui",
}
LOYAL = {
    "tenure_months": 60, "monthly_charges": 30.0, "support_calls": 0, "num_products": 4,
    "contract": "Biannuel", "payment_method": "Prélèvement", "internet_service": "Aucun",
    "tech_support": "Oui", "senior": "Non",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Modèle et données dans un dossier temporaire : le test n'écrit rien dans le dépôt.
    tmp = tmp_path_factory.mktemp("api")
    mp = pytest.MonkeyPatch()
    mp.setenv("CHURN_MODEL_PATH", str(tmp / "model.joblib"))
    mp.setenv("CHURN_DATA_PATH", str(tmp / "customers.csv"))
    with TestClient(api.app) as c:
        yield c
    mp.undo()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_a_probability(client):
    body = client.post("/predict", json=RISKY).json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["threshold"] == 0.5
    assert body["model"] == "LogisticRegression"


def test_risky_customer_scores_higher_than_loyal_one(client):
    risky = client.post("/predict", json=RISKY).json()
    loyal = client.post("/predict", json=LOYAL).json()
    assert risky["churn_probability"] > 0.8 and risky["at_risk"]
    assert loyal["churn_probability"] < 0.2 and not loyal["at_risk"]


def test_threshold_changes_the_decision(client):
    p = client.post("/predict", json=RISKY).json()["churn_probability"]
    above = min(0.99, round(p + 0.005, 3))
    assert client.post(f"/predict?threshold={above}", json=RISKY).json()["at_risk"] is False


@pytest.mark.parametrize("field, bad_value", [
    ("contract", "À vie"),          # modalité inconnue
    ("tenure_months", -1),          # hors bornes
    ("monthly_charges", "cher"),    # mauvais type
])
def test_invalid_input_is_rejected(client, field, bad_value):
    assert client.post("/predict", json={**RISKY, field: bad_value}).status_code == 422


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in RISKY.items() if k != "contract"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_invalid_threshold_is_rejected(client):
    assert client.post("/predict?threshold=1.5", json=RISKY).status_code == 422

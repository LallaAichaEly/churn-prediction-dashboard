"""Tests de l'API : on vérifie le contrat (codes HTTP, format) autant que les prédictions."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import api  # noqa: E402
from churn.data import generate_customers  # noqa: E402
from churn.model import save, train  # noqa: E402

RISKY = {
    "tenure_months": 3, "monthly_charges": 100.0, "total_charges": 300.0, "support_calls": 5,
    "num_products": 1, "contract": "Mensuel", "payment_method": "Chèque",
    "internet_service": "Fibre", "tech_support": "Non", "senior": "Oui",
}
LOYAL = {
    **RISKY, "tenure_months": 60, "total_charges": 3000.0, "support_calls": 0, "num_products": 4,
    "contract": "Biannuel", "payment_method": "Prélèvement", "tech_support": "Oui", "senior": "Non",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Un modèle entraîné sur un petit jeu, enregistré dans un dossier temporaire."""
    model_dir = tmp_path_factory.mktemp("models")
    pipeline, *_, metrics = train(generate_customers(n=1500, seed=0))
    path = save(pipeline, metrics, model_dir=model_dir)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(api, "MODEL_PATH", path)
        with TestClient(api.app) as c:  # `with` déclenche le lifespan (chargement du modèle)
            yield c


def test_health_reports_metrics(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["metrics"]["roc_auc"] > 0.7


def test_predict_returns_a_probability(client):
    r = client.post("/predict", json=RISKY)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["at_risk"] == (body["churn_probability"] >= 0.5)


def test_risky_profile_scores_higher_than_loyal(client):
    p_risky = client.post("/predict", json=RISKY).json()["churn_probability"]
    p_loyal = client.post("/predict", json=LOYAL).json()["churn_probability"]
    assert p_risky > 0.5 > p_loyal


def test_threshold_controls_the_decision(client):
    assert client.post("/predict", params={"threshold": 0.0}, json=LOYAL).json()["at_risk"] is True
    assert client.post("/predict", params={"threshold": 1.0}, json=RISKY).json()["at_risk"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"contract": "Hebdomadaire"},  # modalité inconnue → refusée à la frontière
        {"tenure_months": -1},  # valeur impossible
        {"monthly_charges": 9999},  # hors bornes
        {"senior": None},  # champ obligatoire
    ],
)
def test_invalid_payload_is_rejected(client, change):
    assert client.post("/predict", json={**RISKY, **change}).status_code == 422


def test_invalid_threshold_is_rejected(client):
    assert client.post("/predict", params={"threshold": 1.5}, json=RISKY).status_code == 422


def test_load_or_train_creates_missing_model(tmp_path):
    data_path = tmp_path / "customers.csv"
    generate_customers(n=800, seed=3).to_csv(data_path, index=False)
    model_path = tmp_path / "models" / "demo.joblib"

    pipeline = api.load_or_train(model_path, data_path)

    assert model_path.exists()
    assert api.read_metrics(model_path)["roc_auc"] > 0.6
    assert hasattr(pipeline, "predict_proba")

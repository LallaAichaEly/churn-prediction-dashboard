"""Tests de l'API : on la lance « pour de faux » avec `TestClient`, sans serveur réseau."""

import pytest
from fastapi.testclient import TestClient

from churn.api import MAX_BATCH, create_app
from churn.data import generate_customers
from churn.model import save, train

CLIENT_RISQUE = {
    "tenure_months": 2, "monthly_charges": 110.0, "total_charges": 220.0, "support_calls": 6,
    "num_products": 1, "contract": "Mensuel", "payment_method": "Chèque",
    "internet_service": "Fibre", "tech_support": "Non", "senior": "Oui",
}
CLIENT_FIDELE = {
    "tenure_months": 70, "monthly_charges": 30.0, "total_charges": 2100.0, "support_calls": 0,
    "num_products": 4, "contract": "Biannuel", "payment_method": "Prélèvement",
    "internet_service": "Aucun", "tech_support": "Oui", "senior": "Non",
}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    pipeline, *_, metrics = train(generate_customers(n=1500, seed=0), costs=(500, 50))
    return save(pipeline, metrics, model_dir=tmp_path_factory.mktemp("models"))


@pytest.fixture(scope="module")
def client(model_path):
    with TestClient(create_app(model_path)) as c:  # `with` déclenche le chargement du modèle
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert 0 < body["threshold"] < 1
    assert body["roc_auc"] > 0.7


def test_predict_ranks_customers_sensibly(client):
    risky = client.post("/predict", json=CLIENT_RISQUE)
    loyal = client.post("/predict", json=CLIENT_FIDELE)
    assert risky.status_code == loyal.status_code == 200
    assert risky.json()["churn_probability"] > 0.8 > 0.2 > loyal.json()["churn_probability"]
    assert risky.json()["at_risk"] is True and loyal.json()["at_risk"] is False


def test_threshold_uses_saved_value_and_can_be_overridden(client):
    saved = client.get("/health").json()["threshold"]
    assert client.post("/predict", json=CLIENT_FIDELE).json()["threshold"] == saved
    forced = client.post("/predict", params={"threshold": 0.0}, json=CLIENT_FIDELE).json()
    assert forced["threshold"] == 0.0 and forced["at_risk"] is True


def test_batch_matches_single_predictions(client):
    batch = client.post("/predict/batch", json=[CLIENT_RISQUE, CLIENT_FIDELE]).json()
    singles = [client.post("/predict", json=c).json() for c in (CLIENT_RISQUE, CLIENT_FIDELE)]
    for b, s in zip(batch, singles):
        # approx : l'ordre des calculs en virgule flottante peut différer à la 15e décimale.
        assert b["churn_probability"] == pytest.approx(s["churn_probability"])
        assert b["at_risk"] == s["at_risk"]
    assert client.post("/predict/batch", json=[]).json() == []


def test_batch_size_is_limited(client):
    response = client.post("/predict/batch", json=[CLIENT_FIDELE] * (MAX_BATCH + 1))
    assert response.status_code == 413


@pytest.mark.parametrize(
    "change",
    [
        {"contract": "mensuel"},          # faute de frappe dans une modalité
        {"monthly_charges": -10},         # valeur hors bornes
        {"tenure_months": "beaucoup"},    # mauvais type
        {"couleur_preferee": "bleu"},     # champ inconnu
    ],
)
def test_invalid_input_is_rejected(client, change):
    assert client.post("/predict", json={**CLIENT_RISQUE, **change}).status_code == 422


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in CLIENT_RISQUE.items() if k != "contract"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_missing_model_fails_at_startup(tmp_path):
    with pytest.raises(RuntimeError, match="train.py"):
        with TestClient(create_app(tmp_path / "absent.joblib")):
            pass

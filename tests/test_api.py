"""Tests de l'API : on vérifie le contrat (entrées / sorties), pas seulement « ça répond »."""

import pytest
from fastapi.testclient import TestClient

from churn.model import save, train

RISKY = {
    "tenure_months": 2, "monthly_charges": 110, "support_calls": 6, "num_products": 1,
    "contract": "Mensuel", "payment_method": "Chèque", "internet_service": "Fibre",
    "tech_support": "Non", "senior": "Oui",
}
LOYAL = {
    "tenure_months": 60, "monthly_charges": 30, "support_calls": 0, "num_products": 4,
    "contract": "Biannuel", "payment_method": "Prélèvement", "internet_service": "Aucun",
    "tech_support": "Oui", "senior": "Non",
}


@pytest.fixture(scope="module")
def client(df, tmp_path_factory, monkeypatch_module):
    pipeline, *_, metrics = train(df)
    path = save(pipeline, metrics, model_dir=tmp_path_factory.mktemp("models"))
    monkeypatch_module.setenv("CHURN_MODEL_PATH", str(path))
    from api import app

    with TestClient(app) as c:  # le « with » déclenche le chargement du modèle (lifespan)
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_a_probability(client):
    r = client.post("/predict", json=RISKY)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["threshold"] == 0.5
    assert body["at_risk"] == (body["churn_probability"] >= 0.5)


def test_risky_profile_scores_higher_than_loyal(client):
    risky = client.post("/predict", json=RISKY).json()["churn_probability"]
    loyal = client.post("/predict", json=LOYAL).json()["churn_probability"]
    assert risky > 0.7 > 0.1 > loyal


def test_threshold_changes_the_decision(client):
    p = client.post("/predict", json=LOYAL).json()["churn_probability"]
    assert client.post("/predict", params={"threshold": p / 2}, json=LOYAL).json()["at_risk"] is True
    assert client.post("/predict", params={"threshold": 0.99}, json=LOYAL).json()["at_risk"] is False


def test_total_charges_defaults_to_monthly_times_tenure(client):
    explicit = {**RISKY, "total_charges": RISKY["monthly_charges"] * RISKY["tenure_months"]}
    assert client.post("/predict", json=explicit).json() == client.post("/predict", json=RISKY).json()


@pytest.mark.parametrize(
    "bad",
    [
        {**RISKY, "contract": "Hebdomadaire"},  # modalité inconnue
        {**RISKY, "tenure_months": -1},          # hors bornes
        {**RISKY, "monthly_charges": 900},       # hors bornes (cf. exercice 1.3)
        {k: v for k, v in RISKY.items() if k != "contract"},  # champ manquant
    ],
)
def test_invalid_input_is_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_invalid_threshold_is_rejected(client):
    assert client.post("/predict", params={"threshold": 1.5}, json=RISKY).status_code == 422

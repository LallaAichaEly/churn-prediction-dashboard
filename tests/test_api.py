"""Projet 2 — Tests de l'API : on la démarre avec un petit modèle entraîné pour l'occasion."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import create_app, to_frame  # noqa: E402
from api.schemas import EXAMPLE_CUSTOMER, MAX_BATCH, Customer  # noqa: E402
from churn.data import CONTRACTS, generate_customers  # noqa: E402
from churn.model import save, train  # noqa: E402


@pytest.fixture(scope="module")
def model_path(tmp_path_factory) -> Path:
    pipeline, _, _, _, metrics = train(generate_customers(n=1500, seed=0))
    return save(pipeline, metrics, model_dir=tmp_path_factory.mktemp("models"))


@pytest.fixture(scope="module")
def client(model_path):
    # `with` déclenche le démarrage (lifespan) : c'est là que le modèle est chargé.
    with TestClient(create_app(model_path, threshold=0.5)) as c:
        yield c


def customer(**changes) -> dict:
    return {**EXAMPLE_CUSTOMER, **changes}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client, model_path):
    info = client.get("/model").json()
    assert info["classifier"] == "LogisticRegression"
    assert info["path"] == str(model_path)
    assert info["threshold"] == 0.5
    assert 0.7 < info["metrics"]["roc_auc"] <= 1


def test_predict_returns_a_probability(client):
    r = client.post("/predict", json=customer())
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == EXAMPLE_CUSTOMER["customer_id"]
    assert 0 <= body["churn_probability"] <= 1
    assert body["at_risk"] == (body["churn_probability"] >= body["threshold"])


def test_api_matches_the_model_used_directly(client, model_path):
    """L'API ne doit RIEN changer à la prédiction (ordre des colonnes, conversions…)."""
    from churn.model import load

    expected = load(model_path).predict_proba(to_frame([Customer(**customer())]))[0, 1]
    got = client.post("/predict", json=customer()).json()["churn_probability"]
    assert got == pytest.approx(expected)


def test_monthly_contract_is_riskier(client):
    p = {c: client.post("/predict", json=customer(contract=c)).json()["churn_probability"] for c in CONTRACTS}
    assert p["Mensuel"] > p["Annuel"] > p["Biannuel"]


def test_total_charges_defaults_to_monthly_times_tenure(client):
    implicit = client.post("/predict", json=customer()).json()
    explicit_total = EXAMPLE_CUSTOMER["monthly_charges"] * EXAMPLE_CUSTOMER["tenure_months"]
    explicit = client.post("/predict", json=customer(total_charges=explicit_total)).json()
    assert implicit["churn_probability"] == pytest.approx(explicit["churn_probability"])


@pytest.mark.parametrize(
    "bad",
    [
        {"tenure_months": -1},             # hors bornes
        {"contract": "Hebdo"},             # modalité inconnue
        {"tech_support": "Peut-être"},
        {"tenure": 12},                    # champ mal orthographié → refusé (extra="forbid")
        {"monthly_charges": "beaucoup"},   # mauvais type
    ],
)
def test_invalid_input_is_rejected(client, bad):
    r = client.post("/predict", json=customer(**bad))
    assert r.status_code == 422


def test_missing_field_is_rejected(client):
    body = customer()
    del body["contract"]
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "contract"]


def test_batch_matches_single_predictions(client):
    customers = [customer(customer_id=f"C{i}", support_calls=i) for i in range(5)]
    batch = client.post("/predict/batch", json={"customers": customers}).json()
    assert batch["n_customers"] == 5
    assert [p["customer_id"] for p in batch["predictions"]] == [f"C{i}" for i in range(5)]
    assert batch["n_at_risk"] == sum(p["at_risk"] for p in batch["predictions"])
    for c, p in zip(customers, batch["predictions"]):
        single = client.post("/predict", json=c).json()
        assert p["churn_probability"] == pytest.approx(single["churn_probability"])


@pytest.mark.parametrize("size", [0, MAX_BATCH + 1])
def test_batch_size_limits(client, size):
    r = client.post("/predict/batch", json={"customers": [customer()] * size})
    assert r.status_code == 422


def test_threshold_changes_the_decision(model_path):
    with TestClient(create_app(model_path, threshold=0.01)) as low, \
         TestClient(create_app(model_path, threshold=0.99)) as high:
        assert low.post("/predict", json=customer()).json()["at_risk"] is True
        assert high.post("/predict", json=customer()).json()["at_risk"] is False


@pytest.mark.parametrize("threshold", [0, 1, 1.5])
def test_invalid_threshold_is_refused(model_path, threshold):
    with pytest.raises(ValueError):
        create_app(model_path, threshold=threshold)


def test_missing_model_gives_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="train.py"):
        with TestClient(create_app(tmp_path / "absent.joblib")):
            pass

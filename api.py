"""Module 6 — Exposer le modèle via une API REST (FastAPI).

Lancer :
    python scripts/train.py            # crée models/churn_model.joblib
    uvicorn api:app --reload           # http://localhost:8000/docs (documentation interactive)

Exemple d'appel :
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
      "tenure_months": 3, "monthly_charges": 95, "total_charges": 285, "support_calls": 4,
      "num_products": 1, "contract": "Mensuel", "payment_method": "Chèque",
      "internet_service": "Fibre", "tech_support": "Non", "senior": "Non"}'

Idées clés :
- Pydantic VALIDE les entrées : une ancienneté négative ou un contrat inconnu renvoie une
  erreur 422 claire au lieu d'une prédiction absurde.
- Le modèle est chargé UNE fois (pas à chaque requête).
- `create_app(pipeline=...)` permet d'injecter un modèle dans les tests (pas de fichier requis).
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from churn import __version__  # noqa: E402
from churn.data import FEATURES  # noqa: E402
from churn.model import load  # noqa: E402

MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", ROOT / "models" / "churn_model.joblib"))


class Customer(BaseModel):
    """Profil d'un client. Les bornes protègent le modèle des valeurs aberrantes."""

    tenure_months: int = Field(..., ge=0, le=120, description="Ancienneté en mois")
    monthly_charges: float = Field(..., ge=0, le=500, description="Facture mensuelle (€)")
    total_charges: float = Field(..., ge=0, description="Total facturé depuis l'arrivée (€)")
    support_calls: int = Field(..., ge=0, le=50)
    num_products: int = Field(..., ge=1, le=10)
    contract: Literal["Mensuel", "Annuel", "Biannuel"]
    payment_method: Literal["Prélèvement", "Carte bancaire", "Virement", "Chèque"]
    internet_service: Literal["Fibre", "ADSL", "Aucun"]
    tech_support: Literal["Oui", "Non"]
    senior: Literal["Oui", "Non"]


class Prediction(BaseModel):
    churn_probability: float
    at_risk: bool
    threshold: float


@lru_cache(maxsize=1)
def _load_default_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise HTTPException(503, f"Modèle introuvable ({MODEL_PATH}). Lancez d'abord : python scripts/train.py")
    return load(MODEL_PATH)


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    app = FastAPI(title="Churn Prediction API", version=__version__,
                  description="Prédit la probabilité qu'un client résilie son abonnement.")

    def get_pipeline() -> Pipeline:
        return pipeline if pipeline is not None else _load_default_model()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": pipeline is not None or MODEL_PATH.exists()}

    @app.post("/predict", response_model=Prediction)
    def predict(customer: Customer, threshold: float = Query(0.5, gt=0, lt=1)) -> Prediction:
        return predict_batch([customer], threshold)[0]

    @app.post("/predict/batch", response_model=list[Prediction])
    def predict_batch(customers: list[Customer], threshold: float = Query(0.5, gt=0, lt=1)) -> list[Prediction]:
        if not customers:
            return []
        X = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
        probas = get_pipeline().predict_proba(X)[:, 1]
        return [Prediction(churn_probability=round(float(p), 4), at_risk=bool(p >= threshold), threshold=threshold)
                for p in probas]

    return app


app = create_app()

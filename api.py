"""Module 6 — Le modèle en production : une API REST avec FastAPI.

Lancer :  uvicorn api:app --reload
Docs interactives (testez /predict depuis le navigateur) : http://localhost:8000/docs

Idées clés :
- La FRONTIÈRE du système valide les entrées (Pydantic) : une faute de frappe dans
  « contract » renvoie une erreur 422 claire au lieu d'une prédiction silencieusement fausse.
- Le modèle est chargé UNE fois au démarrage (lifespan), pas à chaque requête.
- Le chemin du modèle est configurable par variable d'environnement (12-factor app).
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, Query, Request  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from churn.data import CONTRACTS, FEATURES, INTERNET_SERVICES, PAYMENT_METHODS, load_or_create  # noqa: E402
from churn.model import load, save, train  # noqa: E402

MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", ROOT / "models" / "churn_model.joblib"))
DATA_PATH = Path(os.environ.get("CHURN_DATA_PATH", ROOT / "data" / "customers.csv"))
YES_NO = ("Oui", "Non")


class Customer(BaseModel):
    """Profil d'un client. Les bornes rejettent les valeurs absurdes (ex. ancienneté négative)."""

    tenure_months: int = Field(ge=0, le=600, description="Ancienneté en mois", examples=[5])
    monthly_charges: float = Field(ge=0, le=500, description="Facture mensuelle (€)", examples=[95.0])
    total_charges: float = Field(ge=0, description="Total facturé depuis le début (€)", examples=[475.0])
    support_calls: int = Field(ge=0, le=100, description="Appels au support", examples=[4])
    num_products: int = Field(ge=1, le=20, description="Nombre de produits souscrits", examples=[1])
    contract: Literal[tuple(CONTRACTS)]  # type: ignore[valid-type]
    payment_method: Literal[tuple(PAYMENT_METHODS)]  # type: ignore[valid-type]
    internet_service: Literal[tuple(INTERNET_SERVICES)]  # type: ignore[valid-type]
    tech_support: Literal[YES_NO]  # type: ignore[valid-type]
    senior: Literal[YES_NO]  # type: ignore[valid-type]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "tenure_months": 5, "monthly_charges": 95.0, "total_charges": 475.0,
                "support_calls": 4, "num_products": 1, "contract": "Mensuel",
                "payment_method": "Chèque", "internet_service": "Fibre",
                "tech_support": "Non", "senior": "Non",
            }]
        }
    }


class Prediction(BaseModel):
    churn_probability: float = Field(description="Probabilité estimée de départ (0 à 1)")
    at_risk: bool = Field(description="True si la probabilité dépasse le seuil")
    threshold: float


def load_or_train(model_path: Path = MODEL_PATH, data_path: Path = DATA_PATH) -> Pipeline:
    """Charge le modèle sauvegardé ; s'il n'existe pas encore, entraîne la référence et la sauvegarde."""
    model_path = Path(model_path)
    if not model_path.exists():
        pipeline, *_, metrics = train(load_or_create(data_path))
        save(pipeline, metrics, model_dir=model_path.parent, name=model_path.stem)
    return load(model_path)


def read_metrics(model_path: Path) -> dict | None:
    path = Path(model_path).with_name(f"{Path(model_path).stem}_metrics.json")
    return json.loads(path.read_text()) if path.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_path = MODEL_PATH
    app.state.pipeline = load_or_train(MODEL_PATH, DATA_PATH)
    yield


app = FastAPI(
    title="API Churn",
    description="Prédit la probabilité qu'un client résilie son abonnement.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    """Vérifie que l'API répond et indique les performances du modèle chargé."""
    return {"status": "ok", "metrics": read_metrics(request.app.state.model_path)}


@app.post("/predict", response_model=Prediction)
def predict(
    customer: Customer,
    request: Request,
    threshold: float = Query(0.5, ge=0, le=1, description="Seuil de décision « à risque »"),
) -> Prediction:
    row = pd.DataFrame([customer.model_dump()])[FEATURES]
    proba = float(request.app.state.pipeline.predict_proba(row)[0, 1])
    return Prediction(churn_probability=round(proba, 4), at_risk=proba >= threshold, threshold=threshold)

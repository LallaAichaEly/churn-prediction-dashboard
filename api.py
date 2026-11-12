"""Module 6 — API de prédiction (FastAPI).

Le tableau de bord sert aux humains ; l'API sert aux autres logiciels (CRM, outil marketing…).

Lancer :
    python scripts/train.py          # crée models/churn_model.joblib
    uvicorn api:app --reload         # puis ouvrir http://localhost:8000/docs

Exemple :
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
      "tenure_months": 3, "monthly_charges": 95, "support_calls": 4, "num_products": 1,
      "contract": "Mensuel", "payment_method": "Chèque", "internet_service": "Fibre",
      "tech_support": "Non", "senior": "Non"}'
"""

from __future__ import annotations

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

from churn import __version__  # noqa: E402
from churn.data import CONTRACTS, FEATURES, INTERNET_SERVICES, PAYMENT_METHODS  # noqa: E402
from churn.model import load  # noqa: E402

YesNo = Literal["Oui", "Non"]


def model_path() -> Path:
    """Chemin du modèle, surchargeable par variable d'environnement (utile en test / prod)."""
    return Path(os.environ.get("CHURN_MODEL_PATH", ROOT / "models" / "churn_model.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On charge le modèle UNE fois au démarrage, pas à chaque requête.
    path = model_path()
    if not path.exists():
        raise RuntimeError(f"Modèle introuvable ({path}). Lancez d'abord : python scripts/train.py")
    app.state.model = load(path)
    app.state.model_file = path.name
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Probabilité qu'un client résilie son abonnement.",
    version=__version__,
    lifespan=lifespan,
)


class Customer(BaseModel):
    """Profil d'un client. Pydantic rejette (HTTP 422) toute valeur hors de ces règles."""

    tenure_months: int = Field(ge=0, le=120, description="Ancienneté en mois", examples=[3])
    monthly_charges: float = Field(ge=0, le=500, description="Facture mensuelle (€)", examples=[95.0])
    total_charges: float | None = Field(
        default=None, ge=0, description="Total payé (€). Si absent : facture × ancienneté."
    )
    support_calls: int = Field(ge=0, le=50, examples=[4])
    num_products: int = Field(ge=1, le=10, examples=[1])
    contract: Literal[tuple(CONTRACTS)]  # type: ignore[valid-type]
    payment_method: Literal[tuple(PAYMENT_METHODS)]  # type: ignore[valid-type]
    internet_service: Literal[tuple(INTERNET_SERVICES)]  # type: ignore[valid-type]
    tech_support: YesNo
    senior: YesNo

    def to_frame(self) -> pd.DataFrame:
        row = self.model_dump()
        if row["total_charges"] is None:
            row["total_charges"] = row["monthly_charges"] * row["tenure_months"]
        return pd.DataFrame([row])[FEATURES]


class Prediction(BaseModel):
    churn_probability: float = Field(description="Entre 0 et 1")
    at_risk: bool = Field(description="probabilité ≥ seuil")
    threshold: float


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model": request.app.state.model_file, "version": __version__}


@app.post("/predict", response_model=Prediction)
def predict(
    customer: Customer,
    request: Request,
    threshold: float = Query(0.5, ge=0, le=1, description="Seuil de décision (voir onglet Performance)"),
) -> Prediction:
    proba = float(request.app.state.model.predict_proba(customer.to_frame())[0, 1])
    return Prediction(churn_probability=round(proba, 4), at_risk=proba >= threshold, threshold=threshold)

"""Module 6 — API de prédiction (FastAPI).

Lancer :
    uvicorn api:app --reload
puis ouvrir http://localhost:8000/docs pour tester l'API depuis le navigateur.

Exemple :
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \\
         -d '{"tenure_months": 3, "monthly_charges": 89.9, "support_calls": 4, "num_products": 1,
              "contract": "Mensuel", "payment_method": "Chèque", "internet_service": "Fibre",
              "tech_support": "Non", "senior": "Non"}'

Le modèle est lu dans `models/churn_model.joblib` (créé par `python scripts/train.py`).
S'il n'existe pas, il est entraîné au démarrage. Les variables d'environnement
CHURN_MODEL_PATH et CHURN_DATA_PATH permettent de changer ces emplacements.
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
from fastapi import FastAPI, Query  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from churn import __version__  # noqa: E402
from churn.data import FEATURES, load_or_create, validate  # noqa: E402
from churn.model import load, save, train  # noqa: E402

Contract = Literal["Mensuel", "Annuel", "Biannuel"]
Payment = Literal["Prélèvement", "Carte bancaire", "Virement", "Chèque"]
Internet = Literal["Fibre", "ADSL", "Aucun"]
YesNo = Literal["Oui", "Non"]


class Customer(BaseModel):
    """Profil d'un client. Les bornes refusent les valeurs absurdes (réponse HTTP 422)."""

    tenure_months: int = Field(ge=0, le=120, description="Ancienneté en mois")
    monthly_charges: float = Field(gt=0, le=500, description="Facture mensuelle (€)")
    total_charges: float | None = Field(
        default=None, ge=0, description="Total payé (€). Par défaut : facture × ancienneté"
    )
    support_calls: int = Field(ge=0, le=50, description="Appels au support")
    num_products: int = Field(ge=1, le=10, description="Nombre de produits souscrits")
    contract: Contract
    payment_method: Payment
    internet_service: Internet
    tech_support: YesNo
    senior: YesNo

    def to_frame(self) -> pd.DataFrame:
        row = self.model_dump()
        if row["total_charges"] is None:
            row["total_charges"] = row["monthly_charges"] * row["tenure_months"]
        return pd.DataFrame([row])[FEATURES]


class Prediction(BaseModel):
    churn_probability: float = Field(description="Probabilité de départ, entre 0 et 1")
    at_risk: bool = Field(description="Vrai si la probabilité dépasse le seuil")
    threshold: float
    model: str


def _get_pipeline():
    """Charge le modèle sauvegardé, ou l'entraîne s'il n'existe pas encore."""
    model_path = Path(os.environ.get("CHURN_MODEL_PATH", ROOT / "models" / "churn_model.joblib"))
    if model_path.exists():
        return load(model_path)
    df = load_or_create(os.environ.get("CHURN_DATA_PATH", ROOT / "data" / "customers.csv"))
    problems = validate(df)
    if problems:
        raise RuntimeError("Données invalides : " + " ; ".join(problems))
    pipeline, *_, metrics = train(df)
    save(pipeline, metrics, model_dir=model_path.parent, name=model_path.stem)
    return pipeline


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = _get_pipeline()
    yield
    state.clear()


app = FastAPI(
    title="API Churn",
    description="Prédit la probabilité qu'un client résilie son abonnement.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Sonde de santé : utilisée par les plateformes de déploiement et la supervision."""
    return {"status": "ok", "model_loaded": "pipeline" in state}


@app.post("/predict", response_model=Prediction)
def predict(
    customer: Customer,
    threshold: float = Query(0.5, gt=0, lt=1, description="Seuil de décision (voir l'onglet Performance)"),
) -> Prediction:
    pipeline = state["pipeline"]
    proba = float(pipeline.predict_proba(customer.to_frame())[0, 1])
    return Prediction(
        churn_probability=round(proba, 4),
        at_risk=proba >= threshold,
        threshold=threshold,
        model=type(pipeline.named_steps["clf"]).__name__,
    )

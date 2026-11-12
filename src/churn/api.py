"""Module 6 — Servir le modèle via une API REST (FastAPI).

Lancer :
    python scripts/train.py                        # crée models/churn_model.joblib
    uvicorn churn.api:app --app-dir src --reload   # puis ouvrir http://localhost:8000/docs

Idées clés :
- Pydantic valide chaque requête AVANT que le modèle ne la voie : un contrat inconnu ou une
  ancienneté négative renvoie une erreur 422 claire au lieu d'une prédiction absurde.
- Le modèle est fourni par une *dépendance* (`get_model`) : en test, on la remplace
  par un petit modèle entraîné à la volée (`app.dependency_overrides`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from . import __version__
from .data import FEATURES
from .model import load

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "churn_model.joblib"

EXAMPLE_CUSTOMER = {
    "tenure_months": 4,
    "monthly_charges": 89.9,
    "total_charges": 359.6,
    "support_calls": 3,
    "num_products": 1,
    "contract": "Mensuel",
    "payment_method": "Chèque",
    "internet_service": "Fibre",
    "tech_support": "Non",
    "senior": "Non",
}


class Customer(BaseModel):
    """Profil d'un client. Les valeurs autorisées reprennent celles de `churn.data`
    (un test vérifie que les deux listes restent synchronisées)."""

    model_config = {"json_schema_extra": {"examples": [EXAMPLE_CUSTOMER]}}

    tenure_months: int = Field(ge=0, le=120, description="Ancienneté (mois)")
    monthly_charges: float = Field(gt=0, le=500, description="Facture mensuelle (€)")
    total_charges: float = Field(ge=0, description="Total facturé depuis le début (€)")
    support_calls: int = Field(ge=0, le=100, description="Appels au support")
    num_products: int = Field(ge=1, le=10, description="Nombre de produits souscrits")
    contract: Literal["Mensuel", "Annuel", "Biannuel"]
    payment_method: Literal["Prélèvement", "Carte bancaire", "Virement", "Chèque"]
    internet_service: Literal["Fibre", "ADSL", "Aucun"]
    tech_support: Literal["Oui", "Non"]
    senior: Literal["Oui", "Non"]


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float = Field(description="Probabilité estimée de départ (0 à 1)")
    at_risk: bool = Field(description="Vrai si la probabilité atteint le seuil")
    threshold: float


@dataclass
class LoadedModel:
    pipeline: Pipeline
    metrics: dict | None
    path: Path


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime: float) -> LoadedModel:
    """Charge une seule fois par version du fichier (`mtime` change après un réentraînement)."""
    p = Path(path)
    metrics_path = p.with_name(f"{p.stem}_metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    return LoadedModel(pipeline=load(p), metrics=metrics, path=p)


def get_model() -> LoadedModel:
    """Dépendance FastAPI : le modèle sauvegardé (chemin modifiable via CHURN_MODEL_PATH)."""
    path = Path(os.environ.get("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Modèle introuvable ({path.name}). Lancez d'abord : python scripts/train.py",
        )
    return _load_cached(str(path), path.stat().st_mtime)


ModelDep = Annotated[LoadedModel, Depends(get_model)]
Threshold = Annotated[
    float, Query(ge=0, le=1, description="Seuil au-dessus duquel un client est « à risque »")
]

app = FastAPI(
    title="API Churn",
    version=__version__,
    description="Prédit la probabilité qu'un client résilie son abonnement.",
)


def _predict(pipeline: Pipeline, customers: list[Customer], threshold: float) -> list[Prediction]:
    X = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    proba = pipeline.predict_proba(X)[:, 1]
    return [
        Prediction(churn_probability=round(float(p), 4), at_risk=bool(p >= threshold), threshold=threshold)
        for p in proba
    ]


@app.get("/health")
def health() -> dict:
    """Le service répond-il ? (utilisé par les outils de supervision)"""
    return {"status": "ok", "version": __version__}


@app.get("/model")
def model_info(model: ModelDep) -> dict:
    """Quel modèle est servi, avec quelles variables et quelles performances ?"""
    return {
        "file": model.path.name,
        "estimator": type(model.pipeline.named_steps["clf"]).__name__,
        "features": FEATURES,
        "metrics": model.metrics,
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, model: ModelDep, threshold: Threshold = 0.5) -> Prediction:
    """Probabilité de churn d'un client."""
    return _predict(model.pipeline, [customer], threshold)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: BatchRequest, model: ModelDep, threshold: Threshold = 0.5) -> list[Prediction]:
    """Probabilités pour une liste de clients (1 à 1 000), dans le même ordre."""
    return _predict(model.pipeline, batch.customers, threshold)

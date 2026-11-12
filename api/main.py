"""Projet 2 — Servir le modèle de churn via une API REST (FastAPI).

Le tableau de bord (Projet 1) charge le modèle lui-même. En entreprise, on sépare souvent :
- l'ENTRAÎNEMENT : hors ligne, de temps en temps (`python scripts/train.py`) ;
- le SERVICE : en ligne, en continu, via une API que n'importe quelle application
  (CRM, site web, autre équipe) peut appeler.

Lancer :
    python scripts/train.py              # 1. entraîner et sauvegarder le modèle
    uvicorn api.main:app --reload        # 2. démarrer l'API
    http://localhost:8000/docs           # 3. documentation interactive (Swagger)

Variables d'environnement (optionnelles) :
    CHURN_MODEL_PATH   chemin du modèle (défaut : models/churn_model.joblib)
    CHURN_THRESHOLD    seuil de décision entre 0 et 1 (défaut : 0.5)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from churn.data import FEATURES
from churn.model import load

from . import __version__
from .schemas import (
    MAX_BATCH,
    BatchRequest,
    BatchResponse,
    Customer,
    HealthResponse,
    ModelInfo,
    Prediction,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "churn_model.joblib"

DESCRIPTION = """
Prédit la probabilité qu'un client parte (*churn*).

1. Essayez **POST /predict** : cliquez sur *Try it out* puis *Execute* (un exemple est pré-rempli).
2. Modifiez `contract` ou `support_calls` et observez la probabilité changer.
3. Envoyez une valeur absurde (`"tenure_months": -1`) et lisez l'erreur 422.
"""


def to_frame(customers: list[Customer]) -> pd.DataFrame:
    """Transforme des clients validés en DataFrame, colonnes dans l'ordre attendu par le modèle."""
    rows = []
    for customer in customers:
        row = customer.model_dump(exclude={"customer_id"})
        if row["total_charges"] is None:
            # Même règle que le formulaire du tableau de bord.
            row["total_charges"] = row["monthly_charges"] * row["tenure_months"]
        rows.append(row)
    return pd.DataFrame(rows)[FEATURES]


def _read_threshold(threshold: float | None) -> float:
    value = float(threshold if threshold is not None else os.getenv("CHURN_THRESHOLD", "0.5"))
    if not 0 < value < 1:
        raise ValueError(f"Le seuil doit être strictement entre 0 et 1 (reçu : {value})")
    return value


def _model_info(model_path: Path, pipeline, threshold: float) -> ModelInfo:
    metrics_path = model_path.with_name(f"{model_path.stem}_metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    return ModelInfo(
        path=str(model_path),
        classifier=type(pipeline[-1]).__name__,
        features=FEATURES,
        threshold=threshold,
        trained_at=datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc),
        metrics=metrics,
    )


def create_app(model_path: str | Path | None = None, threshold: float | None = None) -> FastAPI:
    """Fabrique de l'application.

    Passer le chemin du modèle en paramètre (plutôt que de le coder en dur) permet aux tests
    de démarrer l'API avec un petit modèle temporaire.
    """
    model_path = Path(model_path or os.getenv("CHURN_MODEL_PATH") or DEFAULT_MODEL_PATH)
    threshold = _read_threshold(threshold)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Le modèle est chargé UNE fois au démarrage, pas à chaque requête.
        if not model_path.exists():
            raise RuntimeError(
                f"Modèle introuvable : {model_path}\n"
                "Lancez d'abord :  python scripts/train.py"
            )
        app.state.pipeline = load(model_path)
        app.state.info = _model_info(model_path, app.state.pipeline, threshold)
        yield

    app = FastAPI(
        title="API de prédiction du churn",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    def predict_many(request: Request, customers: list[Customer]) -> list[Prediction]:
        proba = request.app.state.pipeline.predict_proba(to_frame(customers))[:, 1]
        return [
            Prediction(
                customer_id=customer.customer_id,
                churn_probability=float(p),
                at_risk=bool(p >= threshold),
                threshold=threshold,
            )
            for customer, p in zip(customers, proba)
        ]

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse, tags=["Service"],
             summary="L'API est-elle en vie ?")
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/model", response_model=ModelInfo, tags=["Service"],
             summary="Quel modèle est servi ?")
    def model_info(request: Request) -> ModelInfo:
        return request.app.state.info

    @app.post("/predict", response_model=Prediction, tags=["Prédiction"],
              summary="Probabilité de churn d'un client")
    def predict(customer: Customer, request: Request) -> Prediction:
        return predict_many(request, [customer])[0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["Prédiction"],
              summary=f"Probabilités de churn de 1 à {MAX_BATCH} clients")
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        # Un seul appel à predict_proba pour tout le lot : bien plus rapide qu'une boucle.
        predictions = predict_many(request, batch.customers)
        return BatchResponse(
            n_customers=len(predictions),
            n_at_risk=sum(p.at_risk for p in predictions),
            predictions=predictions,
        )

    return app


app = create_app()

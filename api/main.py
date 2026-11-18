"""Projet 2 — Servir le modèle avec une API REST (FastAPI).

Lancer :
    uvicorn api.main:app --reload
Puis ouvrir la documentation interactive : http://localhost:8000/docs

Ce qu'on apprend ici :
- charger le modèle UNE fois au démarrage (`lifespan`), pas à chaque requête ;
- valider les entrées avec Pydantic (voir `api/schemas.py`) ;
- réutiliser exactement le même pipeline qu'à l'entraînement → pas d'écart
  entraînement / production (*training-serving skew*) ;
- rendre l'application configurable par variables d'environnement (12-factor app) :
    CHURN_MODEL_PATH  chemin du modèle .joblib (défaut : models/churn_model.joblib)
    CHURN_THRESHOLD   seuil de décision entre 0 et 1 (défaut : 0.5)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request

from churn import __version__
from churn.data import FEATURES, load_or_create, validate
from churn.model import load, save, train

from .schemas import BatchRequest, BatchResponse, Customer, ModelInfo, Prediction

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "churn_model.joblib"
DEFAULT_DATA_PATH = ROOT / "data" / "customers.csv"
REPORTED_METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1"]


def load_or_train_model(model_path: str | Path, data_path: str | Path = DEFAULT_DATA_PATH):
    """Charge le modèle sauvegardé ; s'il n'existe pas encore, l'entraîne puis le sauvegarde.

    Retourne (pipeline, métriques de test).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        df = load_or_create(data_path)
        problems = validate(df)
        if problems:
            raise RuntimeError("Données invalides : " + " ; ".join(problems))
        pipeline, *_, metrics = train(df)
        save(pipeline, metrics, model_dir=model_path.parent, name=model_path.stem)

    metrics_path = model_path.with_name(f"{model_path.stem}_metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return load(model_path), {k: round(float(metrics[k]), 4) for k in REPORTED_METRICS if k in metrics}


def create_app(model_path: str | Path | None = None, threshold: float | None = None) -> FastAPI:
    """Fabrique d'application : les tests créent leur propre app avec un modèle temporaire."""
    model_path = Path(model_path or os.getenv("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))
    threshold = float(threshold if threshold is not None else os.getenv("CHURN_THRESHOLD", "0.5"))
    if not 0 < threshold < 1:
        raise ValueError(f"Le seuil doit être strictement entre 0 et 1 (reçu : {threshold})")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline, app.state.metrics = load_or_train_model(model_path)
        yield

    app = FastAPI(
        title="Churn Prediction API",
        version=__version__,
        description="Estime la probabilité qu'un client résilie son abonnement.",
        lifespan=lifespan,
    )

    def score(request: Request, customers: list[Customer]) -> list[Prediction]:
        X = pd.DataFrame([c.to_features() for c in customers])[FEATURES]
        proba = request.app.state.pipeline.predict_proba(X)[:, 1]
        return [
            Prediction(churn_probability=round(float(p), 4), at_risk=bool(p >= threshold), threshold=threshold)
            for p in proba
        ]

    @app.get("/health", tags=["Supervision"])
    def health() -> dict:
        """Sonde de vie : utilisée par Docker, Kubernetes ou un load balancer."""
        return {"status": "ok"}

    @app.get("/model", response_model=ModelInfo, tags=["Supervision"])
    def model_info(request: Request) -> ModelInfo:
        """Quel modèle tourne, avec quel seuil, et quelles performances au test ?"""
        clf = request.app.state.pipeline.named_steps["clf"]
        return ModelInfo(
            model_type=type(clf).__name__,
            features=FEATURES,
            threshold=threshold,
            test_metrics=request.app.state.metrics,
        )

    @app.post("/predict", response_model=Prediction, tags=["Prédiction"])
    def predict(customer: Customer, request: Request) -> Prediction:
        """Probabilité de churn d'un client."""
        return score(request, [customer])[0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["Prédiction"])
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        """Jusqu'à 1 000 clients en une requête (un seul appel au modèle : bien plus rapide)."""
        predictions = score(request, batch.customers)
        return BatchResponse(predictions=predictions, n_at_risk=sum(p.at_risk for p in predictions))

    return app


app = create_app()

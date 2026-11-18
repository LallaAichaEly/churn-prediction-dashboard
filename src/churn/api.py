"""Module 6 — Servir le modèle via une API web (FastAPI).

Un modèle n'a de valeur que s'il est utilisé. Une API permet à n'importe quelle application
(CRM, site web, outil marketing) de demander « ce client va-t-il partir ? ».

Lancer (après `python scripts/train.py`) :
    uvicorn churn.api:app --app-dir src --reload
puis ouvrir http://localhost:8000/docs : documentation interactive générée automatiquement.

Exemple :
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
      "tenure_months": 3, "monthly_charges": 95, "total_charges": 285, "support_calls": 4,
      "num_products": 1, "contract": "Mensuel", "payment_method": "Chèque",
      "internet_service": "Fibre", "tech_support": "Non", "senior": "Non"}'

Points clés :
- Pydantic valide chaque requête AVANT le modèle : une faute de frappe (« mensuel ») renvoie
  une erreur 422 claire au lieu d'une prédiction silencieusement fausse.
- Le modèle est chargé UNE fois au démarrage (`lifespan`), pas à chaque requête.
- Le seuil de décision vient du fichier de métriques enregistré par `scripts/train.py`.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .data import CONTRACTS, FEATURES, INTERNET_SERVICES, PAYMENT_METHODS
from .model import load, load_metrics

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "churn_model.joblib"
MAX_BATCH = 1000

# `Literal[tuple(...)]` construit le type à partir des constantes de `data.py` : une seule
# source de vérité pour les modalités autorisées.
Contract = Literal[tuple(CONTRACTS)]  # type: ignore[valid-type]
PaymentMethod = Literal[tuple(PAYMENT_METHODS)]  # type: ignore[valid-type]
InternetService = Literal[tuple(INTERNET_SERVICES)]  # type: ignore[valid-type]
YesNo = Literal["Oui", "Non"]


class Customer(BaseModel):
    """Profil d'un client — mêmes colonnes que `data.FEATURES`."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "tenure_months": 3, "monthly_charges": 95.0, "total_charges": 285.0,
                    "support_calls": 4, "num_products": 1, "contract": "Mensuel",
                    "payment_method": "Chèque", "internet_service": "Fibre",
                    "tech_support": "Non", "senior": "Non",
                }
            ]
        },
    )

    tenure_months: int = Field(ge=0, le=600, description="Ancienneté en mois")
    monthly_charges: float = Field(ge=0, le=500, description="Facture mensuelle (€)")
    total_charges: float = Field(ge=0, description="Total facturé depuis le début (€)")
    support_calls: int = Field(ge=0, le=100, description="Appels au support")
    num_products: int = Field(ge=1, le=20, description="Nombre de produits souscrits")
    contract: Contract
    payment_method: PaymentMethod
    internet_service: InternetService
    tech_support: YesNo
    senior: YesNo


class Prediction(BaseModel):
    churn_probability: float = Field(description="Probabilité estimée de départ (0 à 1)")
    at_risk: bool = Field(description="True si la probabilité atteint le seuil")
    threshold: float = Field(description="Seuil de décision utilisé")


class Health(BaseModel):
    status: str
    version: str
    model_path: str
    threshold: float
    roc_auc: float | None


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Fabrique l'application. Le chemin du modèle vient de l'argument, sinon de la variable
    d'environnement `CHURN_MODEL_PATH`, sinon de `models/churn_model.joblib`.

    Une *fabrique* (plutôt qu'une variable globale) permet aux tests de créer une API
    branchée sur un modèle temporaire.
    """
    path = Path(model_path or os.environ.get("CHURN_MODEL_PATH") or DEFAULT_MODEL_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not path.exists():
            raise RuntimeError(
                f"Modèle introuvable : {path}. Lancez d'abord `python scripts/train.py`."
            )
        metrics = load_metrics(path)
        app.state.pipeline = load(path)
        app.state.threshold = metrics.threshold if metrics else 0.5
        app.state.roc_auc = metrics.roc_auc if metrics else None
        yield

    app = FastAPI(
        title="API Churn",
        description="Prédit la probabilité qu'un client résilie son abonnement.",
        version=__version__,
        lifespan=lifespan,
    )

    def predict_many(request: Request, customers: list[Customer], threshold: float | None) -> list[Prediction]:
        t = request.app.state.threshold if threshold is None else threshold
        X = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
        proba = request.app.state.pipeline.predict_proba(X)[:, 1]
        return [Prediction(churn_probability=float(p), at_risk=bool(p >= t), threshold=t) for p in proba]

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        """Vérifie que l'API tourne et indique quel modèle est servi."""
        return Health(
            status="ok", version=__version__, model_path=str(path),
            threshold=request.app.state.threshold, roc_auc=request.app.state.roc_auc,
        )

    @app.post("/predict", response_model=Prediction)
    def predict(
        request: Request,
        customer: Customer,
        threshold: float | None = Query(None, ge=0, le=1, description="Remplace le seuil enregistré"),
    ) -> Prediction:
        """Probabilité de churn d'UN client."""
        return predict_many(request, [customer], threshold)[0]

    @app.post("/predict/batch", response_model=list[Prediction])
    def predict_batch(
        request: Request,
        customers: list[Customer],
        threshold: float | None = Query(None, ge=0, le=1, description="Remplace le seuil enregistré"),
    ) -> list[Prediction]:
        """Probabilités de churn de PLUSIEURS clients en un seul appel (plus efficace)."""
        if not customers:
            return []
        if len(customers) > MAX_BATCH:
            raise HTTPException(status_code=413, detail=f"Au plus {MAX_BATCH} clients par appel")
        return predict_many(request, customers, threshold)

    return app


app = create_app()

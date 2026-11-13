"""Module 6 — Servir le modèle via une API REST (FastAPI).

Lancer :
    python scripts/train.py                        # crée models/churn_model.joblib
    uvicorn churn.api:app --app-dir src --reload   # puis ouvrir http://localhost:8000/docs

Pourquoi une API ? Le tableau de bord sert aux humains ; une API sert aux autres logiciels
(CRM, outil marketing, application mobile...). C'est la forme la plus courante de
« mise en production » d'un modèle.

Trois réflexes professionnels illustrés ici :
1. **Valider les entrées** avec Pydantic : une requête mal formée reçoit une erreur 422 claire
   au lieu d'une prédiction absurde.
2. **Échouer tôt** : si le modèle est introuvable, l'API refuse de démarrer.
3. **Rendre testable** : `create_app()` accepte un pipeline déjà entraîné, ce qui permet de
   tester l'API sans fichier sur disque (voir `tests/test_api.py`).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from . import __version__
from .data import CONTRACTS, FEATURES, INTERNET_SERVICES, PAYMENT_METHODS
from .model import load

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "churn_model.joblib"

# Une seule source de vérité : les modalités acceptées viennent de `data.py`.
Contract = Literal[tuple(CONTRACTS)]  # type: ignore[valid-type]
PaymentMethod = Literal[tuple(PAYMENT_METHODS)]  # type: ignore[valid-type]
InternetService = Literal[tuple(INTERNET_SERVICES)]  # type: ignore[valid-type]
YesNo = Literal["Oui", "Non"]


class Customer(BaseModel):
    """Profil d'un client. Les bornes rejettent les valeurs manifestement fausses."""

    tenure_months: int = Field(ge=0, le=120, description="Ancienneté en mois", examples=[12])
    monthly_charges: float = Field(gt=0, le=500, description="Facture mensuelle (€)", examples=[70.0])
    total_charges: float | None = Field(
        default=None, ge=0, description="Total payé (€). Si absent : facture mensuelle × ancienneté."
    )
    support_calls: int = Field(ge=0, le=50, examples=[2])
    num_products: int = Field(ge=1, le=10, examples=[1])
    contract: Contract = Field(examples=["Mensuel"])
    payment_method: PaymentMethod = Field(examples=["Chèque"])
    internet_service: InternetService = Field(examples=["Fibre"])
    tech_support: YesNo = Field(examples=["Non"])
    senior: YesNo = Field(examples=["Non"])


class Batch(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float = Field(description="Probabilité estimée de départ (0 à 1)")
    at_risk: bool = Field(description="Vrai si la probabilité dépasse le seuil de décision")
    threshold: float


def to_frame(customers: list[Customer]) -> pd.DataFrame:
    """Convertit les requêtes en DataFrame au format attendu par le pipeline."""
    X = pd.DataFrame([c.model_dump() for c in customers])
    X["total_charges"] = pd.to_numeric(X["total_charges"]).fillna(X["monthly_charges"] * X["tenure_months"])
    return X[FEATURES]


def create_app(
    pipeline: Pipeline | None = None,
    model_path: str | Path | None = None,
    threshold: float | None = None,
) -> FastAPI:
    """Construit l'application.

    - `pipeline` : modèle déjà chargé (pratique pour les tests). Sinon, il est lu au démarrage
      depuis `model_path`, la variable d'environnement `CHURN_MODEL_PATH`, ou `models/`.
    - `threshold` : seuil de décision. Par défaut `CHURN_THRESHOLD` ou 0,5.
      Astuce : utilisez le seuil optimal trouvé dans l'onglet « Performance » du tableau de bord.
    """
    path = Path(model_path or os.environ.get("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))
    thr = float(threshold if threshold is not None else os.environ.get("CHURN_THRESHOLD", 0.5))
    if not 0 < thr < 1:
        raise ValueError(f"Le seuil doit être entre 0 et 1 (reçu : {thr})")
    state: dict[str, Pipeline | None] = {"pipeline": pipeline}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["pipeline"] is None:
            if not path.exists():
                raise RuntimeError(f"Modèle introuvable : {path}. Lancez d'abord `python scripts/train.py`.")
            state["pipeline"] = load(path)
        yield

    app = FastAPI(
        title="Churn API",
        version=__version__,
        description="Prédit la probabilité qu'un client résilie son abonnement.",
        lifespan=lifespan,
    )

    def predict_many(customers: list[Customer]) -> list[Prediction]:
        proba = state["pipeline"].predict_proba(to_frame(customers))[:, 1]
        return [
            Prediction(churn_probability=round(float(p), 4), at_risk=bool(p >= thr), threshold=thr)
            for p in proba
        ]

    @app.get("/health")
    def health() -> dict:
        """Vérifie que le service tourne et indique le modèle utilisé."""
        clf = state["pipeline"].named_steps["clf"] if state["pipeline"] is not None else None
        return {"status": "ok", "version": __version__, "model": type(clf).__name__, "threshold": thr}

    @app.post("/predict", response_model=Prediction)
    def predict(customer: Customer) -> Prediction:
        """Prédit le churn d'un client."""
        return predict_many([customer])[0]

    @app.post("/predict/batch", response_model=list[Prediction])
    def predict_batch(batch: Batch) -> list[Prediction]:
        """Prédit le churn de 1 à 1 000 clients en une seule requête (plus efficace)."""
        return predict_many(batch.customers)

    return app


app = create_app()

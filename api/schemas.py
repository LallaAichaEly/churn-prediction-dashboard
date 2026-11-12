"""Les « contrats » de l'API : ce qu'elle accepte et ce qu'elle renvoie (Pydantic).

Une API publique doit refuser les données absurdes AVANT qu'elles n'atteignent le modèle.
Pydantic s'en charge : une ancienneté négative ou un contrat « Hebdo » renvoient une erreur
422 claire, au lieu d'une prédiction silencieusement fausse.

Les valeurs autorisées (contrats, moyens de paiement…) viennent de `churn.data` : une seule
source de vérité pour le modèle, le tableau de bord et l'API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from churn.data import CONTRACTS, INTERNET_SERVICES, PAYMENT_METHODS

# `Literal[tuple(...)]` construit le type à partir des listes de `churn.data`.
Contract = Literal[tuple(CONTRACTS)]
PaymentMethod = Literal[tuple(PAYMENT_METHODS)]
InternetService = Literal[tuple(INTERNET_SERVICES)]
YesNo = Literal["Oui", "Non"]

MAX_BATCH = 1000

EXAMPLE_CUSTOMER = {
    "customer_id": "C00042",
    "tenure_months": 4,
    "monthly_charges": 89.9,
    "support_calls": 3,
    "num_products": 1,
    "contract": "Mensuel",
    "payment_method": "Chèque",
    "internet_service": "Fibre",
    "tech_support": "Non",
    "senior": "Non",
}


class Customer(BaseModel):
    """Profil d'un client à évaluer."""

    # extra="forbid" : un champ mal orthographié (« tenure » au lieu de « tenure_months »)
    # est rejeté au lieu d'être ignoré en silence.
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    customer_id: str | None = Field(
        None, max_length=64, description="Identifiant libre, renvoyé tel quel dans la réponse."
    )
    tenure_months: int = Field(ge=0, le=120, description="Ancienneté en mois.")
    monthly_charges: float = Field(ge=0, le=500, description="Facture mensuelle (€).")
    total_charges: float | None = Field(
        None, ge=0, description="Total payé depuis le début (€). Si absent : facture × ancienneté."
    )
    support_calls: int = Field(ge=0, le=100, description="Nombre d'appels au support.")
    num_products: int = Field(ge=1, le=20, description="Nombre de produits souscrits.")
    contract: Contract
    payment_method: PaymentMethod
    internet_service: InternetService
    tech_support: YesNo = Field(description="Le client a-t-il l'option support technique ?")
    senior: YesNo


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    customer_id: str | None = None
    churn_probability: float = Field(ge=0, le=1, description="Probabilité estimée de départ.")
    at_risk: bool = Field(description="Vrai si la probabilité atteint le seuil de décision.")
    threshold: float


class BatchResponse(BaseModel):
    n_customers: int
    n_at_risk: int
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ModelInfo(BaseModel):
    """Carte d'identité du modèle servi : indispensable pour savoir QUI a prédit QUOI."""

    path: str
    classifier: str
    features: list[str]
    threshold: float
    trained_at: datetime
    metrics: dict[str, float] | None = Field(
        None, description="Métriques sur le jeu de test au moment de l'entraînement."
    )

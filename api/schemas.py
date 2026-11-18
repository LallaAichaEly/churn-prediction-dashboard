"""Les contrats de l'API (Pydantic).

Idée clé : on valide les données À L'ENTRÉE de l'API. Une requête mal formée est rejetée
avec une erreur 422 qui dit précisément quel champ pose problème — elle n'atteint jamais
le modèle. C'est la première ligne de défense d'un modèle en production.

Les valeurs autorisées viennent de `churn.data` : une seule source de vérité, donc
impossible que l'API et les données d'entraînement divergent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from churn.data import CONTRACTS, INTERNET_SERVICES, PAYMENT_METHODS

Contract = Literal[tuple(CONTRACTS)]  # type: ignore[valid-type]
PaymentMethod = Literal[tuple(PAYMENT_METHODS)]  # type: ignore[valid-type]
InternetService = Literal[tuple(INTERNET_SERVICES)]  # type: ignore[valid-type]
YesNo = Literal["Oui", "Non"]

EXAMPLE_CUSTOMER = {
    "tenure_months": 3,
    "monthly_charges": 89.9,
    "support_calls": 4,
    "num_products": 1,
    "contract": "Mensuel",
    "payment_method": "Chèque",
    "internet_service": "Fibre",
    "tech_support": "Non",
    "senior": "Non",
}


class Customer(BaseModel):
    """Profil d'un client à évaluer."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    tenure_months: int = Field(ge=0, le=120, description="Ancienneté en mois")
    monthly_charges: float = Field(gt=0, le=500, description="Facture mensuelle (€)")
    total_charges: float | None = Field(
        default=None, ge=0, description="Total payé (€). Si absent : facture mensuelle × ancienneté"
    )
    support_calls: int = Field(ge=0, le=50, description="Nombre d'appels au support")
    num_products: int = Field(ge=1, le=10, description="Nombre de produits souscrits")
    contract: Contract
    payment_method: PaymentMethod
    internet_service: InternetService
    tech_support: YesNo
    senior: YesNo

    def to_features(self) -> dict:
        """Convertit le profil en ligne prête pour le pipeline scikit-learn."""
        row = self.model_dump()
        if row["total_charges"] is None:
            row["total_charges"] = row["monthly_charges"] * row["tenure_months"]
        return row


class Prediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1, description="Probabilité estimée de départ")
    at_risk: bool = Field(description="Vrai si la probabilité dépasse le seuil de décision")
    threshold: float = Field(description="Seuil de décision utilisé")


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    n_at_risk: int


class ModelInfo(BaseModel):
    model_type: str
    features: list[str]
    threshold: float
    test_metrics: dict[str, float] = Field(description="Métriques mesurées sur le jeu de test à l'entraînement")

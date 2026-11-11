"""Module 1 — Les données.

On génère un jeu de données *synthétique* d'opérateur télécom. Avantages pour apprendre :
- aucune dépendance réseau ni donnée personnelle ;
- on CONNAÎT les vraies causes du churn (voir `_churn_logit`), donc on peut vérifier
  si le modèle les retrouve. C'est un excellent moyen de développer son intuition.

Exercice : une fois le projet compris, remplacez ce générateur par un vrai jeu de données
(ex. « Telco Customer Churn » sur Kaggle) en gardant les mêmes noms de colonnes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "churn"
ID_COL = "customer_id"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "num_products",
]
CATEGORICAL_FEATURES = [
    "contract",
    "payment_method",
    "internet_service",
    "tech_support",
    "senior",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CONTRACTS = ["Mensuel", "Annuel", "Biannuel"]
PAYMENT_METHODS = ["Prélèvement", "Carte bancaire", "Virement", "Chèque"]
INTERNET_SERVICES = ["Fibre", "ADSL", "Aucun"]


def _churn_logit(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """La « vérité cachée » : comment le churn est réellement généré.

    Un logit positif = plus de chances de partir. Lisez ces coefficients, puis comparez-les
    à l'importance des variables trouvée par le modèle (onglet « Facteurs de churn »).
    """
    logit = (
        -1.2
        + 1.6 * (df["contract"] == "Mensuel")
        - 0.9 * (df["contract"] == "Biannuel")
        - 0.035 * df["tenure_months"]
        + 0.018 * (df["monthly_charges"] - 65)
        + 0.45 * df["support_calls"]
        + 0.7 * (df["payment_method"] == "Chèque")
        + 0.5 * (df["internet_service"] == "Fibre")
        - 0.8 * (df["tech_support"] == "Oui")
        - 0.25 * (df["num_products"] - 1)
        + 0.3 * (df["senior"] == "Oui")
    )
    noise = rng.normal(0, 0.6, size=len(df))
    return logit.to_numpy(dtype=float) + noise


def generate_customers(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Génère `n` clients fictifs, reproductibles grâce à `seed`."""
    if n <= 0:
        raise ValueError("n doit être strictement positif")

    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, size=n, p=[0.55, 0.25, 0.20])
    # Les clients sous contrat long sont en moyenne plus anciens.
    tenure_scale = np.select(
        [contract == "Mensuel", contract == "Annuel"], [14.0, 30.0], default=45.0
    )
    tenure = np.clip(rng.exponential(tenure_scale), 1, 72).round().astype(int)

    internet = rng.choice(INTERNET_SERVICES, size=n, p=[0.45, 0.35, 0.20])
    base_price = np.select(
        [internet == "Fibre", internet == "ADSL"], [80.0, 55.0], default=25.0
    )
    monthly = np.clip(base_price + rng.normal(0, 12, size=n), 15, 130).round(2)
    total = (monthly * tenure * rng.uniform(0.95, 1.05, size=n)).round(2)

    df = pd.DataFrame(
        {
            ID_COL: [f"C{i:05d}" for i in range(1, n + 1)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls": rng.poisson(1.3, size=n),
            "num_products": rng.integers(1, 5, size=n),
            "contract": contract,
            "payment_method": rng.choice(PAYMENT_METHODS, size=n, p=[0.35, 0.30, 0.20, 0.15]),
            "internet_service": internet,
            "tech_support": rng.choice(["Oui", "Non"], size=n, p=[0.35, 0.65]),
            "senior": rng.choice(["Oui", "Non"], size=n, p=[0.18, 0.82]),
        }
    )

    proba = 1.0 / (1.0 + np.exp(-_churn_logit(df, rng)))
    df[TARGET] = (rng.uniform(size=n) < proba).astype(int)
    return df


def load_or_create(path: str | Path = "data/customers.csv", n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Charge le CSV s'il existe, sinon le génère et l'enregistre."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    df = generate_customers(n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def validate(df: pd.DataFrame) -> list[str]:
    """Contrôles qualité simples. Retourne la liste des problèmes (vide = OK).

    Réflexe professionnel : toujours valider les données AVANT d'entraîner un modèle.
    """
    problems: list[str] = []
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        problems.append(f"Colonnes manquantes : {missing_cols}")
        return problems
    if df[FEATURES].isna().any().any():
        problems.append("Valeurs manquantes dans les variables explicatives")
    if not set(df[TARGET].unique()) <= {0, 1}:
        problems.append("La cible doit être binaire (0/1)")
    if (df["tenure_months"] < 0).any():
        problems.append("Ancienneté négative détectée")
    if ID_COL in df.columns and df[ID_COL].duplicated().any():
        problems.append("Identifiants clients dupliqués")
    return problems

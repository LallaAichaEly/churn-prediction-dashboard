"""Configuration partagée par tous les tests (pytest charge ce fichier automatiquement)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # pour importer api.py

from churn.data import generate_customers  # noqa: E402


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    """Petit jeu synthétique : assez gros pour apprendre, assez petit pour des tests rapides."""
    return generate_customers(n=1500, seed=0)

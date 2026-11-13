"""Fixtures partagées par tous les fichiers de tests."""

import pandas as pd
import pytest

from churn.data import generate_customers


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    """Petit jeu de données reproductible : assez grand pour apprendre, assez petit pour être rapide."""
    return generate_customers(n=1500, seed=0)

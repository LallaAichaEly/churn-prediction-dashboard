"""Configuration des katas : choix du module corrigé et bilan de progression en français."""

from __future__ import annotations

import functools
import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for chemin in (ROOT, ROOT / "src"):
    if str(chemin) not in sys.path:
        sys.path.insert(0, str(chemin))

from churn.data import generate_customers  # noqa: E402


class _AFaire:
    """Enveloppe un module : une fonction encore vide (NotImplementedError) → test « à faire »."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, nom):
        fonction = getattr(self._module, nom)

        @functools.wraps(fonction)
        def appel(*args, **kwargs):
            try:
                return fonction(*args, **kwargs)
            except NotImplementedError:
                pytest.skip(f"à faire : {nom}()")

        return appel


@pytest.fixture(scope="module")
def kata(request):
    """Votre code (`exercices/`) ou, avec CHURN_SOLUTIONS=1, le corrigé (`solutions/`)."""
    paquet = "solutions" if os.environ.get("CHURN_SOLUTIONS") == "1" else "exercices"
    nom = request.module.__name__.rsplit(".", 1)[-1].removeprefix("test_")
    return _AFaire(importlib.import_module(f"{paquet}.{nom}"))


@pytest.fixture(scope="session")
def clients() -> pd.DataFrame:
    """Jeu de données fixe (seed constante) : les résultats attendus sont reproductibles."""
    return generate_customers(n=3000, seed=7)


def pytest_terminal_summary(terminalreporter):
    stats = terminalreporter.stats
    reussis = len(stats.get("passed", []))
    rates = len(stats.get("failed", [])) + len(stats.get("error", []))
    a_faire = len(stats.get("skipped", []))
    total = reussis + rates + a_faire
    if not total:
        return
    terminalreporter.write_sep("=", "Progression")
    terminalreporter.write_line(
        f"✅ réussis : {reussis}/{total}   ❌ à corriger : {rates}   ⏳ à faire : {a_faire}"
    )
    if reussis == total:
        terminalreporter.write_line("🏆 Module terminé ! Notez ce que vous avez appris dans JOURNAL.md.")

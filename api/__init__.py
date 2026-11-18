"""Projet 2 — API de prédiction du churn (FastAPI)."""

import sys
from pathlib import Path

# Rend le paquet `churn` (dans src/) importable, comme dans app.py et scripts/train.py.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

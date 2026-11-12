"""Projet 2 — API de prédiction du churn (FastAPI).

Le paquet `churn` vit dans `src/` (comme pour `app.py` et `scripts/train.py`) : on l'ajoute
au chemin d'import ici, une seule fois, pour tous les modules de l'API.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

__version__ = "1.0.0"

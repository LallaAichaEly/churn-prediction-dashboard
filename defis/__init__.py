"""Défis auto-corrigés : écrivez du code, lancez `python scripts/defis.py`, progressez.

Voir `defis/README.md` pour le mode d'emploi.
"""

import sys
from pathlib import Path

# Rend le paquet `churn` (dossier src/) importable depuis vos solutions.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

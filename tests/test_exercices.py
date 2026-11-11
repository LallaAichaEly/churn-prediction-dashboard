"""Garde-fou : les katas doivent rester solubles quand le projet évolue.

On lance le correcteur des exercices contre les corrigés (`solutions/`) : tout doit passer.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_les_corriges_passent_tous_les_katas():
    resultat = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "exercices")],
        cwd=ROOT,
        env={**os.environ, "CHURN_SOLUTIONS": "1"},
        capture_output=True,
        text=True,
    )
    sortie = resultat.stdout + resultat.stderr
    assert resultat.returncode == 0, sortie
    assert "skipped" not in sortie, "un corrigé est incomplet :\n" + sortie

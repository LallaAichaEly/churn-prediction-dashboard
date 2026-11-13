"""Point d'entrée (DÉJÀ ÉCRIT) : c'est ce fichier qui s'exécute avec `python -m <paquet>`.

    python -m projets.campagne preparer --plan projets/exemple_plan.csv --base campagne.db --graine 2026-11
"""

import sys

from .cli import main

try:
    code = main()
except NotImplementedError:
    sys.exit("Le projet n'est pas encore terminé : lancez `python scripts/projet.py --projet 3` pour voir "
             "l'étape suivante.")
sys.exit(code)

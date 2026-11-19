"""Point d'entrée (DÉJÀ ÉCRIT) : c'est ce fichier qui s'exécute avec `python -m <paquet>`.

    python -m projets.rapport --abonnes projets/exemple_abonnes.csv --mois 2026-10
"""

import sys

from .cli import main

try:
    code = main()
except NotImplementedError:
    sys.exit("Le projet n'est pas encore terminé : lancez `python scripts/projet.py --projet 7` pour voir "
             "l'étape suivante.")
sys.exit(code)

"""Point d'entrée (DÉJÀ ÉCRIT) : c'est ce fichier qui s'exécute avec `python -m <paquet>`.

    python -m projets.entrepot charger --dossier projets/exemple_exports --base entrepot.db
"""

import sys

from .cli import main

try:
    code = main()
except NotImplementedError:
    sys.exit("Le projet n'est pas encore terminé : lancez `python scripts/projet.py --projet 6` pour voir "
             "l'étape suivante.")
sys.exit(code)

"""Point d'entrée (DÉJÀ ÉCRIT) : c'est ce fichier qui s'exécute avec `python -m <paquet>`.

    python -m projets.service importer --contacts projets/exemple_contacts.csv
"""

import sys

from .cli import main

try:
    code = main()
except NotImplementedError:
    sys.exit("Le projet n'est pas encore terminé : lancez `python scripts/projet.py --projet 4` pour voir "
             "l'étape suivante.")
sys.exit(code)

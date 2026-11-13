"""Point d'entrée (DÉJÀ ÉCRIT) : c'est ce fichier qui s'exécute avec `python -m <paquet>`.

    python -m projets.reentrainement entrainer --lots projets/exemple_lots.csv --mois 2026-03 --registre registre_modeles
"""

import sys

from .cli import main

try:
    code = main()
except NotImplementedError:
    sys.exit("Le projet n'est pas encore terminé : lancez `python scripts/projet.py --projet 5` pour voir "
             "l'étape suivante.")
sys.exit(code)

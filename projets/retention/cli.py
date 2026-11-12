"""Étape 6 🟡 — Assembler les étapes dans un vrai programme en ligne de commande.

Vérifier : python scripts/projet.py --etape 6
Essayer  : python -m projets.retention --clients projets/exemple_clients.csv --budget 1500
"""

from __future__ import annotations

import argparse  # noqa: F401 — vous en aurez besoin
import sys  # noqa: F401 — pour écrire sur sys.stderr
from pathlib import Path  # noqa: F401

from .allocation import allouer_budget  # noqa: F401 — étape 4
from .clients import lire_clients  # noqa: F401 — étape 1
from .modeles import HORIZON_MOIS  # noqa: F401
from .rapport import ecrire_plan, resumer  # noqa: F401 — étape 5
from .regles import meilleure_offre  # noqa: F401 — étape 3


def main(argv: list[str] | None = None) -> int:
    """Le programme complet. Renvoie le code de sortie : 0 si tout va bien, 1 en cas d'erreur.

    Utilisation :

        python -m projets.retention --clients FICHIER --budget MONTANT [--sortie FICHIER] [--horizon MOIS]

    Options (avec `argparse`) :
        --clients  obligatoire : le CSV des clients (étape 1)
        --budget   obligatoire : un montant en euros STRICTEMENT POSITIF ; « 1500 », « 1500.5 » et
                   « 1500,5 » sont acceptés ; « -100 », « 0 » ou « abc » sont refusés par argparse
                   (message d'erreur et code de sortie 2, c'est ce que fait `parser.error`)
        --sortie   facultatif : où écrire le plan (défaut : plan_retention.csv)
        --horizon  facultatif : un entier, le nombre de mois (défaut : HORIZON_MOIS)

    Déroulé :
    1. Lire les clients (étape 1). Si le fichier n'existe pas ou qu'il lui manque des colonnes :
       message clair sur `sys.stderr` et `return 1`. Pas de traceback pour une erreur prévisible.
    2. Afficher chaque ligne ignorée sur `sys.stderr` (`print(..., file=sys.stderr)`).
    3. Pour chaque client, la meilleure offre avec le bon horizon (étape 3) ; garder celles
       qui ne sont pas None.
    4. Répartir le budget (étape 4), écrire le plan (étape 5).
    5. Afficher un résumé sur la sortie standard (`print`). Il doit contenir au moins le nombre
       de clients ciblés et le chemin du fichier écrit ; ajoutez le coût, le gain, le détail par
       offre : c'est votre programme !
    6. `return 0`.

    `argv` : la liste des arguments, ex. ["--clients", "c.csv", "--budget", "1500"]. Avec None,
    argparse lit la vraie ligne de commande. C'est ce qui permet au correcteur (et à vos tests)
    d'appeler `main([...])` sans ouvrir de terminal.

    Ce que vous apprenez : assembler des modules, `argparse` (type personnalisé avec
    `argparse.ArgumentTypeError`), séparer stdout et stderr, les codes de sortie d'un programme.

    Indice : `parser.add_argument("--budget", required=True, type=une_fonction)` appelle
    `une_fonction(texte)` ; si elle lève `argparse.ArgumentTypeError("...")`, argparse affiche le
    message et quitte avec le code 2.
    """
    raise NotImplementedError

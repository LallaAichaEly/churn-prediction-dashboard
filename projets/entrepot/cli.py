"""Étape 6 🟡 — Le programme de la nuit : charger, puis exporter la table du projet 1.

Vérifier : python scripts/projet.py --projet 6 --etape 6
"""

from __future__ import annotations

import argparse  # noqa: F401 — vous en aurez besoin
import csv  # noqa: F401
import sys  # noqa: F401
from datetime import date  # noqa: F401
from pathlib import Path  # noqa: F401

from .fichier import charger_fichier  # noqa: F401
from .modeles import CHARGE, DEJA_CHARGE, HORIZON_JOURS, REFUSE, ouvrir_base  # noqa: F401
from .table import construire_table  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    """Le programme que l'équipe data lancera chaque nuit. Renvoie le code de sortie (0 = tout va bien).

    Deux sous-commandes (`argparse`, `add_subparsers(dest="commande", required=True)`) :

    `charger --dossier DOSSIER --base FICHIER.db [--rejets rejets.csv]`
      1. Si le dossier n'existe pas : une phrase sur stderr, code 1, et AUCUNE base créée (vérifiez le
         dossier AVANT `ouvrir_base`, qui crée le fichier).
      2. Les fichiers `crm_*.csv` et `support_*.jsonl` du dossier, triés par nom (les dates dans les
         noms font que l'ordre alphabétique est l'ordre chronologique). Les autres fichiers sont
         laissés de côté, sans erreur.
      3. Pour chacun, `charger_fichier` et une ligne qui dit ce qui s'est passé, avec le nom du
         fichier et l'un de ces mots : « chargé » (avec le nombre de lignes en base et de rejets),
         « déjà chargé » ou « refusé ». Une `ValueError` (mauvais en-tête) n'arrête pas les autres
         fichiers : le fichier compte comme refusé.
      4. Un résumé : combien de fichiers chargés, déjà chargés, refusés, et combien de contrats et
         de tickets dans la base.
      5. S'il y a eu des rejets (y compris ceux des fichiers refusés) : écrivez-les dans le fichier
         `--rejets`, un CSV avec l'en-tête `fichier,ligne,raison` (le module `csv`, séparateur
         virgule) et une ligne par rejet. Sinon, n'écrivez pas ce fichier.
      6. Code de sortie : 1 si au moins un fichier a été refusé, 0 sinon. Relancer la commande sur le
         même dossier affiche « déjà chargé » partout, ne change rien dans la base, et renvoie 0.

    `exporter --base FICHIER.db --photo AAAA-MM-JJ --sortie FICHIER.csv [--horizon 90]`
      1. La base doit exister (sinon stderr, code 1, et pas de base vide créée par erreur) ; la photo
         doit être une date AAAA-MM-JJ (`date.fromisoformat`), sinon stderr et code 1.
      2. `construire_table` ; sa `ValueError` (photo trop récente, base vide) → stderr, code 1, et
         pas de fichier écrit.
      3. Le CSV de sortie (`to_csv(..., index=False)`), en créant son dossier s'il n'existe pas : le
         fichier que lit le projet 1. Affichez le nombre de clients et le taux de départ.

    Les erreurs attendues s'affichent en UNE phrase sur stderr (`print(..., file=sys.stderr)`),
    jamais en traceback. Fermez la connexion à la fin (`connexion.close()`, dans un `finally`).

    Exemple de sortie de `charger` (le dossier `projets/exemple_exports/`) :

        ✅ crm_2026-01-05.csv        chargé : 757 ligne(s) en base, 0 rejet(s)
        ⚠️  crm_2026-04-01.csv        chargé : 51 ligne(s) en base, 3 rejet(s)
        ...
        20 fichier(s) : 20 chargé(s), 0 déjà chargé(s), 0 refusé(s)
        Base entrepot.db : 1026 contrats, 2053 tickets
        8 ligne(s) écartée(s), détail dans rejets.csv

    Ce que vous apprenez : assembler les étapes en un programme qu'on relance sans crainte, parcourir
    un dossier (`Path.iterdir`, `sorted`), ne pas créer de fichier par effet de bord, stdout contre
    stderr, les codes de sortie.
    """
    raise NotImplementedError

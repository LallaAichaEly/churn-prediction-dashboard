"""Étape 1 🟢 — Lire le fichier des clients sans s'arrêter à la première ligne fausse.

Vérifier : python scripts/projet.py --etape 1
"""

from __future__ import annotations

import csv  # noqa: F401 — vous en aurez besoin
from pathlib import Path

from .modeles import COLONNES_REQUISES, CONTRATS, Client  # noqa: F401


def lire_clients(chemin: str | Path) -> tuple[list[Client], list[str]]:
    """Lit un fichier CSV de clients et renvoie `(clients, erreurs)`.

    L'entrée typique est l'export « Clients à risque » du tableau de bord (projet 1) : il contient
    les colonnes de `COLONNES_REQUISES` (pas forcément dans cet ordre) et d'autres qu'on ignore.

        customer_id,tenure_months,monthly_charges,…,contract,…,support_calls,…,proba_churn
        C00042,5,95.5,…,Mensuel,…,4,…,0.912

    Le contrat :

    1. `chemin` peut être un texte ("data/clients.csv") ou un `Path`.
    2. S'il manque une colonne de `COLONNES_REQUISES`, levez `ValueError` avec un message qui
       NOMME les colonnes manquantes : le fichier entier est inutilisable.
    3. Chaque ligne valide devient un `Client`, avec les bons types : `proba_churn` et
       `monthly_charges` en float, `tenure_months` et `support_calls` en int, les textes sans
       espaces autour. Gardez l'ordre du fichier.
    4. Une ligne invalide n'arrête PAS la lecture : on l'ignore et on ajoute à `erreurs` un
       message qui commence par « ligne N : », N étant le numéro de la ligne DANS LE FICHIER
       (l'en-tête est la ligne 1, le premier client la ligne 2), suivi du nom de la colonne fautive.
       Une ligne est invalide si :
         - `customer_id` est vide ;
         - un nombre est vide ou illisible (« abc », « douze ») ;
         - `proba_churn` n'est pas entre 0 et 1 ;
         - `monthly_charges`, `tenure_months` ou `support_calls` est négatif ;
         - `contract` n'est pas dans `CONTRATS` ;
         - `customer_id` a déjà été vu plus haut (le message contient alors « double »).
       Un seul message par ligne suffit (le premier problème trouvé).

    Pourquoi ne pas s'arrêter à la première erreur ? Sur 50 000 lignes, refuser tout le fichier
    pour une case vide bloquerait l'équipe ; ignorer en silence serait pire. On écarte la ligne
    ET on le dit.

    Ce que vous apprenez : le module `csv` (`csv.DictReader`), `with open(...)`, convertir du texte
    (`int()`, `float()`), `try`/`except ValueError`, accumuler des erreurs au lieu de planter.

    Indices :
    - `with Path(chemin).open(encoding=..., newline="") as f:` puis `lecteur = csv.DictReader(f)`.
      `lecteur.fieldnames` donne les noms de colonnes, `lecteur.line_num` le numéro de la ligne lue.
    - Un fichier enregistré par Excel commence souvent par un caractère invisible (le « BOM ») :
      cherchez ce que fait `encoding="utf-8-sig"`.
    - Écrivez une petite fonction `_client(ligne) -> Client` qui lève `ValueError("...")` dès
      qu'elle trouve un problème : `lire_clients` n'a plus qu'à l'attraper.
    - Trois pièges que le correcteur a cachés dans ses fichiers : `float("nan")` NE lève PAS
      d'erreur ; une ligne coupée donne `None` (et non "") pour les colonnes absentes ;
      une probabilité de 0 et une ancienneté de 0 sont valides (bornes incluses).
    """
    raise NotImplementedError

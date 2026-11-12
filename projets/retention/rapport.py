"""Étape 5 🟢 — Résumer le plan et l'écrire dans un fichier CSV.

Vérifier : python scripts/projet.py --etape 5
"""

from __future__ import annotations

import csv  # noqa: F401 — vous en aurez besoin
from pathlib import Path
from typing import Sequence

from .modeles import COLONNES_PLAN, Action  # noqa: F401


def resumer(actions: Sequence[Action], nb_clients: int) -> dict:
    """Les chiffres que le manager lira en premier. Renvoie un dictionnaire avec ces clés :

        "clients_analyses" : nb_clients (le nombre de clients lus, ciblés ou non)
        "clients_cibles"   : le nombre d'actions du plan
        "cout_total"       : la somme des coûts des offres, en euros
        "gain_attendu"     : la somme des gains attendus, en euros (coûts déjà déduits)
        "roi"              : gain_attendu / cout_total, ou 0.0 si le plan est vide
        "par_offre"        : {code de l'offre: nombre de clients}, ex. {"APPEL": 12, "REMISE": 3}
                             (seules les offres présentes dans le plan)

    Pas d'arrondi ici non plus : c'est à l'affichage de le faire.

    Ce que vous apprenez : `sum()` avec un générateur, compter par catégorie (`dict.get(cle, 0)`
    ou `collections.Counter`), penser au cas vide (division par zéro).
    """
    raise NotImplementedError


def ecrire_plan(actions: Sequence[Action], chemin: str | Path) -> Path:
    """Écrit le plan dans un fichier CSV et renvoie son chemin (un `Path`).

    - Première ligne : les colonnes de `COLONNES_PLAN`
      (customer_id, offre, cout, proba_churn, gain_attendu).
    - Puis une ligne par action, dans l'ordre reçu : identifiant du client, CODE de l'offre, coût
      avec 2 décimales, probabilité avec 3 décimales, gain avec 2 décimales :
          C00042,ENGAGEMENT,60.00,0.912,410.22
    - Plan vide : le fichier ne contient que l'en-tête.
    - Si le dossier de `chemin` n'existe pas encore (ex. "resultats/2026-11/plan.csv"), créez-le.
    - Encodage UTF-8.

    Ce que vous apprenez : `csv.writer`, formater un nombre (`f"{x:.2f}"`), `newline=""` à
    l'écriture (sinon Windows ajoute des lignes vides), créer un dossier avec
    `Path.mkdir(parents=True, exist_ok=True)`.
    """
    raise NotImplementedError

"""Étape 3 🟡 — Importer les résultats du CRM, même quand ils arrivent en plusieurs fois.

Vérifier : python scripts/projet.py --projet 3 --etape 3
"""

from __future__ import annotations

import csv  # noqa: F401 — vous en aurez besoin
import sqlite3  # noqa: F401
from pathlib import Path

from .modeles import COLONNES_RESULTATS  # noqa: F401


def importer_resultats(conn: sqlite3.Connection, chemin: str | Path) -> tuple[int, list[str]]:
    """Lit le fichier du CRM (colonnes `customer_id` et `parti`) et enregistre chaque résultat dans
    la table `resultats`. Renvoie `(nombre de lignes enregistrées, erreurs)`.

    Un mois après la campagne, le CRM exporte qui est parti. Le fichier est saisi en partie à la main,
    exporté depuis Excel, et il arrive parfois en plusieurs fois. Le contrat :

    1. S'il manque une colonne de `COLONNES_RESULTATS`, levez `ValueError` en NOMMANT les colonnes
       manquantes, et n'enregistrez rien.
    2. `parti` vaut « 1 » ou « 0 », mais aussi « oui » / « non » ou « true » / « false », avec
       n'importe quelles majuscules et des espaces autour. Enregistrez 1 ou 0 (un entier).
    3. Une ligne fausse n'arrête PAS l'import : on l'ignore et on ajoute à `erreurs` un message qui
       commence par « ligne N : » (l'en-tête est la ligne 1). Une ligne est fausse si :
         - `customer_id` est vide (le message contient « customer_id ») ;
         - `parti` est illisible : « ? », « peut-être », vide (le message contient « parti ») ;
         - le client n'est pas dans la table `campagne` (le message contient « inconnu »).
    4. Un client qui a DÉJÀ un résultat (plus haut dans le fichier, ou lors d'un import précédent)
       voit sa valeur REMPLACÉE : le CRM corrige ses erreurs, la dernière valeur lue gagne. Importer
       deux fois le même fichier ne change donc rien. Cherchez « UPSERT » :
       `INSERT … ON CONFLICT (customer_id) DO UPDATE SET parti = excluded.parti`.
    5. Tout le fichier dans UNE transaction (`with conn:`) : si le programme plante au milieu, aucune
       ligne de ce fichier ne reste dans la base, et on peut relancer l'import sans réfléchir.
    6. Le nombre renvoyé compte les lignes enregistrées (insérées ou remplacées), pas les erreurs.

    Pour le client inconnu, deux solutions : vérifier avec un SELECT, ou laisser la clé étrangère
    (activée à l'étape 2) refuser l'INSERT et attraper `sqlite3.IntegrityError`. Seule la requête
    fautive est annulée : la transaction continue.

    Ce que vous apprenez : UPSERT, clé étrangère, `IntegrityError`, une transaction qui englobe une
    boucle, et ce qu'on a déjà vu au projet 2 (BOM d'Excel, lignes coupées, numéros de ligne).
    """
    raise NotImplementedError

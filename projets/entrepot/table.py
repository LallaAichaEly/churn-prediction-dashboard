"""Étape 5 🔴 — La table du projet 1, telle qu'on la connaissait le jour de la photo.

Vérifier : python scripts/projet.py --projet 6 --etape 5
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta  # noqa: F401 — vous en aurez besoin

import pandas as pd

from .modeles import COLONNES_TABLE, FENETRE_JOURS, HORIZON_JOURS, mois_entre  # noqa: F401


def construire_table(connexion: sqlite3.Connection, photo: date, horizon: int = HORIZON_JOURS) -> pd.DataFrame:
    """Construit, à partir de la base, la table qu'entraîne le projet 1 (`data/customers.csv`).

    La « photo » est la date à laquelle on se place. Chaque ligne est un client tel qu'on le
    connaissait CE JOUR-LÀ, et `churn` dit s'il est parti dans les `horizon` jours qui ont suivi.
    Tout ce qui s'est passé après la photo est interdit dans les variables : le modèle n'aura pas
    ces informations quand il servira. S'en servir, c'est une FUITE : le modèle paraît meilleur à
    l'entraînement qu'il ne le sera jamais en production.

    Les règles :

    1. Les dates de `contrats` sont du texte "AAAA-MM-JJ" ; `maj` va jusqu'à la seconde. Si la base
       ne contient aucun contrat, ou si le jour de la plus récente `maj` est AVANT `photo + horizon`
       jours, levez `ValueError` : on ne connaît pas encore tous les départs de l'horizon, et des
       clients partis seraient comptés comme restés.
    2. Une ligne par client ACTIF le jour de la photo : arrivé ce jour-là au plus tard
       (`debut <= photo`) et pas encore parti (`fin` NULL, ou `fin > photo` : un client parti le jour
       même de la photo n'est plus client).
    3. `support_calls` : le nombre de tickets de ce client ouverts dans les `FENETRE_JOURS` (365)
       jours qui précèdent la photo, le jour de la photo compris : `photo − 365 jours < jour
       d'ouverture <= photo`. Un client sans aucun ticket est dans la table avec 0. Attention :
       `ouvert` contient l'heure, et en comparant des textes "2026-06-30T15:04:00" <= "2026-06-30"
       est FAUX. Comparez le jour : `date(t.ouvert)` en SQL.
    4. `tenure_months` : `mois_entre(debut, photo)` (fourni), au moins 1.
    5. `monthly_charges` = `mensualite` ; `total_charges` = mensualité × `tenure_months`, arrondi à
       2 décimales (on n'a pas les factures : c'est une estimation).
    6. `num_products`, `contract`, `payment_method`, `internet_service`, `tech_support`, `senior` :
       les colonnes `produits`, `contrat`, `paiement`, `internet`, `assistance`, `senior`.
    7. `churn` = 1 si le client part dans l'horizon (`fin <= photo + horizon jours`), sinon 0.

    Renvoyez un DataFrame avec exactement les colonnes `COLONNES_TABLE`, dans cet ordre, une ligne
    par client triée par `customer_id`, un index 0, 1, 2… (des entiers pour les nombres de mois,
    d'appels, de produits et pour `churn`). Ne modifiez pas la base.

    Une seule requête SQL suffit pour les règles 2 et 3, avec un `LEFT JOIN` des tickets et un
    `GROUP BY`. Deux pièges classiques : `COUNT(*)` compte 1 pour un client sans ticket (la ligne de
    NULL du LEFT JOIN) ; et une condition sur les tickets placée dans le `WHERE` au lieu du `ON`
    élimine ces clients. Les règles 4 à 7 se font très bien en Python, ligne par ligne.

    Ce que vous apprenez : la validité « à date » (point-in-time), la fuite de données par le futur,
    `LEFT JOIN` contre `JOIN`, `COUNT(colonne)` contre `COUNT(*)`, comparer des dates rangées en
    texte, et quand une étiquette est « mûre ».
    """
    raise NotImplementedError

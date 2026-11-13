"""Étape 1 🟢 — Refuser un lot abîmé AVANT d'entraîner quoi que ce soit.

Vérifier : python scripts/projet.py --projet 5 --etape 1
"""

from __future__ import annotations

import pandas as pd  # noqa: F401 — vous en aurez besoin

from .modeles import CATEGORIELLES, CIBLE, COLONNES, IDENTIFIANT, MODALITES, MOIS, NUMERIQUES  # noqa: F401


def problemes(lots: pd.DataFrame) -> list[str]:
    """Renvoie la liste de TOUS les problèmes du lot, une phrase par problème ; liste vide = lot sain.

    En mars, un export du CRM est arrivé avec la colonne `churn` vide, lue comme 0 partout. Le modèle
    entraîné dessus a appris que « personne ne part » et a été mis en production : pendant trois
    semaines, aucun conseiller n'a reçu d'alerte. Cette fonction est le garde-fou qui l'aurait
    empêché. Elle ne corrige rien : elle dit tout ce qui ne va pas, et le programme refuse le lot.

    Les règles, dans cet ordre :

    0. Si des colonnes de `COLONNES` manquent, renvoyez TOUT DE SUITE une liste d'UNE phrase qui les
       cite toutes (par exemple « colonnes manquantes : churn, contract ») : les règles suivantes
       liraient ces colonnes et planteraient.
    1. Les cases vides : pour chaque colonne qui en a, UNE phrase avec la colonne et leur nombre
       (« monthly_charges : 2 valeur(s) manquante(s) »), pas une phrase par ligne.
    2. Le mois : un texte de la forme AAAA-MM avec un mois de 01 à 12. « 2026-9 », « 2026-13 » ou
       « 2026/09 » sont refusés. Indice : `lots[MOIS].dropna().astype(str).str.fullmatch(motif)`.
    3. La cible `churn` : 0 ou 1, rien d'autre (ni 2, ni « oui »). Indice :
       `pd.to_numeric(colonne, errors="coerce")` transforme ce qui n'est pas un nombre en NaN.
    4. Les colonnes numériques (`NUMERIQUES`) : lisibles comme des nombres (« 45,90 », écrit par
       Excel avec une virgule, ne l'est pas) et jamais négatives.
    5. Les colonnes catégorielles (`CATEGORIELLES`) : seulement les valeurs de `MODALITES`.
       « Mensual » ou « fibre » sont refusés : sans ce contrôle, le modèle IGNORERAIT la valeur
       inconnue en silence (`OneHotEncoder(handle_unknown="ignore")`).
    6. Les doublons : le même client deux fois dans le MÊME mois (citez un exemple : son
       customer_id). Le même client dans deux mois différents est normal : chaque mois est une
       nouvelle photo des clients.
    7. Un mois où personne n'est parti, ou tout le monde (taux de départ de 0 % ou 100 %) : citez
       le mois. C'est exactement l'incident de mars.

    Chaque phrase cite la colonne (ou le mois) concernée. Ne vous arrêtez pas au premier problème :
    rassemblez-les tous, pour que l'équipe du CRM corrige tout en une fois. Et ne modifiez pas
    `lots` (pas de `inplace=True`, pas de nouvelle colonne) : vérifier, ce n'est pas réparer.

    Ce que vous apprenez : valider des données avant de s'en servir, `isna`, `str.fullmatch`,
    `pd.to_numeric(errors="coerce")`, `isin`, `duplicated` sur plusieurs colonnes, `groupby`,
    rassembler des erreurs plutôt que s'arrêter à la première.
    """
    raise NotImplementedError

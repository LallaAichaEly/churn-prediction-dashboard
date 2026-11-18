"""Étape 5 (corrigé) — La table du projet 1, telle qu'on la connaissait le jour de la photo."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd

from .modeles import COLONNES_TABLE, FENETRE_JOURS, HORIZON_JOURS, mois_entre

# - LEFT JOIN : un client sans aucun ticket reste dans la table (avec 0), il ne disparaît pas.
# - COUNT(t.ticket) et pas COUNT(*) : COUNT(*) compterait 1 pour la ligne de NULL du LEFT JOIN.
# - La condition sur la date du ticket est dans le ON, pas dans un WHERE : dans un WHERE, elle
#   éliminerait les lignes de NULL, et le LEFT JOIN redeviendrait un JOIN.
# - date(t.ouvert) : "2026-06-30T15:04:00" <= "2026-06-30" est FAUX en comparant des textes ; on
#   compare donc le JOUR d'ouverture, pas le texte entier.
_REQUETE = """
SELECT c.client, c.debut, c.fin, c.contrat, c.paiement, c.internet, c.assistance, c.senior, c.produits,
       c.mensualite, COUNT(t.ticket) AS tickets
FROM contrats AS c
LEFT JOIN tickets AS t
       ON t.client = c.client
      AND date(t.ouvert) > :depuis
      AND date(t.ouvert) <= :photo
WHERE c.debut <= :photo
  AND (c.fin IS NULL OR c.fin > :photo)
GROUP BY c.client
ORDER BY c.client
"""


def construire_table(connexion: sqlite3.Connection, photo: date, horizon: int = HORIZON_JOURS) -> pd.DataFrame:
    connue = connexion.execute("SELECT max(maj) FROM contrats").fetchone()[0]
    if connue is None:
        raise ValueError("la base ne contient aucun contrat : chargez d'abord les exports")
    connue_jusqu_au = datetime.fromisoformat(connue).date()
    fin_horizon = photo + timedelta(days=horizon)
    if fin_horizon > connue_jusqu_au:
        raise ValueError(f"photo trop récente : il faut connaître les départs jusqu'au {fin_horizon.isoformat()}, "
                         f"mais la base ne va que jusqu'au {connue_jusqu_au.isoformat()}")

    parametres = {"photo": photo.isoformat(), "depuis": (photo - timedelta(days=FENETRE_JOURS)).isoformat()}
    lignes = []
    for (client, debut, fin, contrat, paiement, internet, assistance, senior, produits, mensualite,
         tickets) in connexion.execute(_REQUETE, parametres):
        anciennete = max(1, mois_entre(date.fromisoformat(debut), photo))
        part = fin is not None and date.fromisoformat(fin) <= fin_horizon
        lignes.append((client, anciennete, mensualite, round(mensualite * anciennete, 2), tickets, produits,
                       contrat, paiement, internet, assistance, senior, int(part)))
    table = pd.DataFrame(lignes, columns=list(COLONNES_TABLE))
    return table.astype({"tenure_months": "int64", "monthly_charges": "float64", "total_charges": "float64",
                         "support_calls": "int64", "num_products": "int64", "churn": "int64"})

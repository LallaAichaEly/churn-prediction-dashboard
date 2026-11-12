"""La base de l'opérateur pour la série 10 (vous n'avez pas besoin de modifier ce fichier).

Trois tables, comme dans le système d'information d'une vraie entreprise :

    clients   customer_id · contract · region (NULL si inconnue) · signup_date · churn_date (NULL = toujours client)
    factures  invoice_id · customer_id · invoice_date · amount
    appels    call_id · customer_id (NULL = appel d'un numéro inconnu) · call_date · motif

Les dates sont du texte au format ISO « AAAA-MM-JJ » : c'est la convention de SQLite, et elles se
comparent et se trient correctement comme du texte ('2026-01-31' < '2026-02-01').

Pour regarder les tables :           python -m defis.donnees_sql
Pour essayer une requête en Python :

    >>> from defis.donnees_sql import essayer
    >>> essayer("SELECT contract, COUNT(*) AS n FROM clients GROUP BY contract")
    >>> essayer("SELECT * FROM appels WHERE call_date < :date", date="2025-03-01")
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd

DATE_EXTRACTION = "2026-06-30"  # le jour où la base a été extraite
CONTRATS = ["Mensuel", "Annuel", "Biannuel"]
REGIONS = ["Nord", "Sud", "Est", "Ouest"]
MOTIFS = ["Facturation", "Panne", "Offre", "Résiliation"]
_PRIX = {"Mensuel": 45.0, "Annuel": 38.0, "Biannuel": 34.0}
_CHURN_PAR_AN = {"Mensuel": 0.35, "Annuel": 0.15, "Biannuel": 0.07}


def _mois_suivant(jour: date) -> date:
    return date(jour.year + jour.month // 12, jour.month % 12 + 1, 1)


def generer_tables(n_clients: int = 400, seed: int = 10,
                   date_extraction: str = DATE_EXTRACTION) -> dict[str, pd.DataFrame]:
    """Les trois tables, reproductibles grâce à `seed`."""
    rng = np.random.default_rng(seed)
    fin = date.fromisoformat(date_extraction)
    debut = date(2024, 1, 1)

    clients, factures, appels = [], [], []
    for i in range(1, n_clients + 1):
        cid = f"C{i:04d}"
        contrat = str(rng.choice(CONTRATS, p=[0.55, 0.25, 0.20]))
        region = None if rng.uniform() < 0.08 else str(rng.choice(REGIONS))
        inscription = debut + timedelta(days=int(rng.integers(0, (fin - debut).days - 30)))
        # Durée de vie tirée au hasard : au-delà de la date d'extraction, le client est toujours là.
        duree = rng.exponential(365 / _CHURN_PAR_AN[contrat])
        depart = inscription + timedelta(days=int(duree) + 1)
        churn_date = depart.isoformat() if depart <= fin else None
        derniere = depart if churn_date else fin
        clients.append((cid, contrat, region, inscription.isoformat(), churn_date))

        # Une facture le jour de l'inscription, puis chaque mois (le même jour, au plus le 28) jusqu'au départ.
        prix = _PRIX[contrat] + float(rng.normal(0, 6))
        mois = inscription
        while mois <= derniere:
            factures.append((len(factures) + 1, cid, mois.isoformat(), round(max(prix + float(rng.normal(0, 2)), 5.0), 2)))
            suivant = _mois_suivant(mois)
            mois = suivant.replace(day=min(inscription.day, 28))

        # Des appels au support, plus fréquents chez les clients qui finiront par partir.
        jours_actifs = max((derniere - inscription).days, 1)
        intensite = (2.5 if churn_date else 1.0) * jours_actifs / 365
        for _ in range(int(rng.poisson(intensite))):
            jour = inscription + timedelta(days=int(rng.integers(0, jours_actifs + 1)))
            appels.append([cid, jour.isoformat(), str(rng.choice(MOTIFS, p=[0.4, 0.3, 0.2, 0.1]))])
        # Quelques clients appellent plusieurs mois d'affilée : c'est le signal du défi 67.
        if churn_date and rng.uniform() < 0.3:
            serie = int(rng.integers(2, 6))
            mois = date(derniere.year, derniere.month, 1)
            for _ in range(serie):
                if mois < date(inscription.year, inscription.month, 1):
                    break
                for _ in range(int(rng.integers(1, 3))):
                    appels.append([cid, mois.replace(day=int(rng.integers(1, 29))).isoformat(), "Panne"])
                mois = date(mois.year - (mois.month == 1), (mois.month - 2) % 12 + 1, 1)

    # Les appels de numéros inconnus (client non identifié) : customer_id est NULL.
    for _ in range(max(n_clients // 40, 1)):
        jour = debut + timedelta(days=int(rng.integers(0, (fin - debut).days)))
        appels.append([None, jour.isoformat(), str(rng.choice(MOTIFS))])

    appels.sort(key=lambda a: (a[1], a[0] or ""))
    return {
        "clients": pd.DataFrame(clients, columns=["customer_id", "contract", "region", "signup_date", "churn_date"]),
        "factures": pd.DataFrame(factures, columns=["invoice_id", "customer_id", "invoice_date", "amount"]),
        "appels": pd.DataFrame(
            [(k + 1, *a) for k, a in enumerate(appels)], columns=["call_id", "customer_id", "call_date", "motif"]
        ),
    }


def charger(tables: dict[str, pd.DataFrame]) -> sqlite3.Connection:
    """Une base SQLite EN MÉMOIRE contenant ces tables : rien n'est écrit sur le disque."""
    connexion = sqlite3.connect(":memory:")
    for nom, table in tables.items():
        table.to_sql(nom, connexion, index=False)
    return connexion


def base_demo(n_clients: int = 400, seed: int = 10) -> sqlite3.Connection:
    """La base d'exemple de la série 10, prête à interroger."""
    return charger(generer_tables(n_clients, seed))


def essayer(requete: str, **params) -> pd.DataFrame:
    """Exécute une requête sur la base d'exemple et renvoie le résultat.

    Les paramètres nommés de la requête (`:date_extraction`, `:mois_min`…) se passent en mots-clés :
        essayer(requete_appels_recents(), date_extraction="2026-06-30")
    """
    return pd.read_sql_query(requete, base_demo(), params=params or None)


if __name__ == "__main__":
    with pd.option_context("display.width", 120, "display.max_columns", 10):
        for nom, table in generer_tables().items():
            print(f"\n=== {nom} ({len(table)} lignes)\n{table.head(8).to_string(index=False)}")
    print(f"\nDate d'extraction : {DATE_EXTRACTION}")

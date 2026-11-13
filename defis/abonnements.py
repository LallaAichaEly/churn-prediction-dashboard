"""Outils de la série 21, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

De petites tables d'exemple, construites pour que chaque piège de la série y apparaisse au moins une
fois, et deux générateurs pour essayer vos fonctions sur des centaines de clients. Chaque fonction
renvoie une NOUVELLE table : modifiez-la autant que vous voulez, l'exemple suivant sera intact.

    >>> from defis.serie_21 import *
    >>> abonnements_exemple().head(3)
      client      debut        fin
    0      A 2023-11-20        NaT
    1      B 2023-12-05 2024-02-01
    2      C 2024-01-01        NaT

Les colonnes des abonnements :

- `debut` : le jour où l'abonnement commence ;
- `fin` : le jour de la résiliation, ou `NaT` (« Not a Time », la valeur manquante des dates) si le
  client est TOUJOURS abonné. NaT n'est égal à rien, pas même à lui-même : `fin >= date` vaut False
  quand `fin` est NaT. C'est le piège principal de la série.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================================ Défi 128 : lire des dates


def dates_saisies() -> pd.Series:
    """Des dates tapées à la main dans un formulaire français (format JJ/MM/AAAA), avec ses accidents.

    La première vaut le 3 JANVIER 2024. Par défaut, pandas la lit… le 1er mars.
    """
    return pd.Series(
        ["03/01/2024", "3/1/2024", " 15/02/2024 ", "", None, "29/02/2024", "31/12/2023"],
        index=[101, 102, 103, 104, 105, 106, 107],
        name="date_saisie",
        dtype=object,
    )


# ============================================================================ Défis 129 à 131 : abonnements


def abonnements_exemple() -> pd.DataFrame:
    """Huit abonnements, un par situation à ne pas rater.

    A  arrivé avant la période étudiée, toujours abonné (fin = NaT)
    B  parti le 1er février : il était encore là fin janvier, il part EN février
    C  arrivé le 1er janvier : c'est un nouveau de janvier, pas un client « déjà là » au début de janvier
    D  arrivé le 10 janvier, parti le 25 : il n'était pas là au début du mois
    E  parti le 31 mars à 18 h : il part en mars
    F  arrivé le 29 février (2024 est bissextile)
    G  arrivé en mars, parti en avril
    H  arrivé en mars, toujours abonné
    """
    return pd.DataFrame(
        {
            "client": list("ABCDEFGH"),
            "debut": pd.to_datetime(["2023-11-20", "2023-12-05", "2024-01-01", "2024-01-10", "2024-01-15",
                                     "2024-02-29", "2024-03-03", "2024-03-15"], format="ISO8601"),
            "fin": pd.to_datetime([None, "2024-02-01", None, "2024-01-25", "2024-03-31 18:00",
                                   None, "2024-04-30", None], format="ISO8601"),
        }
    )


def generer_abonnements(n: int = 400, graine: int = 0) -> pd.DataFrame:
    """`n` abonnements tirés au hasard entre 2022 et 2024 (reproductibles grâce à `graine`).

    Environ un client sur trois est toujours abonné. Certaines dates tombent pile le 1er ou le dernier
    jour d'un mois, d'autres ont une heure (23 h 30 le dernier jour du mois, par exemple).
    """
    rng = np.random.default_rng(graine)
    origine = pd.Timestamp("2022-01-01")
    jours = rng.integers(0, 3 * 365, size=n)
    debut = origine + pd.to_timedelta(jours, unit="D")
    # Un quart des débuts tombe le 1er du mois, un dixième le dernier jour à 23 h 30.
    tirage = rng.random(n)
    debut = np.where(tirage < 0.25, debut.to_period("M").to_timestamp(), debut)
    debut = pd.DatetimeIndex(debut)
    dernier_jour = (debut.to_period("M") + 1).to_timestamp() - pd.Timedelta(minutes=30)
    debut = pd.DatetimeIndex(np.where(tirage > 0.90, dernier_jour, debut))

    duree = pd.to_timedelta(rng.exponential(220, size=n).round(), unit="D")
    fin = debut.normalize() + duree
    tirage = rng.random(n)
    fin = pd.DatetimeIndex(np.where(tirage < 0.2, fin.to_period("M").to_timestamp(), fin))  # le 1er du mois
    fin = pd.DatetimeIndex(np.where(fin < debut, debut, fin))
    toujours = (rng.random(n) < 0.33) | (fin > pd.Timestamp("2024-12-31"))
    fin = pd.Series(fin).mask(toujours)
    return pd.DataFrame({"client": [f"C{i:04d}" for i in range(n)], "debut": debut, "fin": fin.to_numpy()})


# ============================================================================ Défi 133 : le revenu mensuel


def factures_exemple() -> pd.DataFrame:
    """Les factures mensuelles de quatre clients, de janvier à avril 2024 (une ligne par facture).

    X  30 €, puis 45 € à partir de mars (il a pris une option : EXPANSION)
    Y  20 €, puis une option facturée À PART en février (deux lignes le même mois), absent en mars
       (PERDU), de retour en avril (RÉACTIVATION : ce n'est pas un nouveau client)
    Z  arrivé en février à 50 € (NOUVEAU), 40 € à partir de mars (CONTRACTION)
    W  arrivé en mars, parti en avril
    """
    lignes = [
        ("X", "2024-01", 30.0), ("X", "2024-02", 30.0), ("X", "2024-03", 45.0), ("X", "2024-04", 45.0),
        ("Y", "2024-01", 20.0), ("Y", "2024-02", 20.0), ("Y", "2024-02", 10.0), ("Y", "2024-04", 20.0),
        ("Z", "2024-02", 50.0), ("Z", "2024-03", 40.0), ("Z", "2024-04", 40.0),
        ("W", "2024-03", 25.0),
    ]
    table = pd.DataFrame(lignes, columns=["client", "mois", "montant"])
    table["mois"] = pd.PeriodIndex(table["mois"], freq="M")
    return table


def generer_factures(n_clients: int = 60, graine: int = 0) -> pd.DataFrame:
    """Les factures de `n_clients` clients sur 2023 et 2024, avec des mois sans facture, des retours,
    des hausses, des baisses et des options facturées sur une ligne à part. Lignes mélangées."""
    rng = np.random.default_rng(graine)
    mois = pd.period_range("2023-01", "2024-12", freq="M")
    lignes = []
    for i in range(n_clients):
        client = f"K{i:03d}"
        present = False
        montant = float(rng.choice([9.99, 19.99, 29.99, 49.99]))
        for m in mois:
            if not present:
                present = rng.random() < 0.12
            elif rng.random() < 0.10:
                present = False
                continue
            if not present:
                continue
            if rng.random() < 0.15:
                montant = float(rng.choice([9.99, 19.99, 29.99, 49.99, 79.99]))
            lignes.append((client, m, montant))
            if rng.random() < 0.08:
                lignes.append((client, m, 5.0))                 # une option facturée à part
    table = pd.DataFrame(lignes, columns=["client", "mois", "montant"])
    table["mois"] = table["mois"].astype("period[M]")
    return table.sample(frac=1.0, random_state=graine).reset_index(drop=True)

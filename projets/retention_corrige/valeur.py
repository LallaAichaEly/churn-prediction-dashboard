"""Étape 2 (corrigé) — Traduire une règle métier en code : combien vaut un client, combien rapporte une offre."""

from __future__ import annotations

from .modeles import HORIZON_MOIS, Client, Offre


def perte_attendue(client: Client, horizon: int = HORIZON_MOIS) -> float:
    # Probabilité de partir × ce qu'on perdrait s'il partait. Pas d'arrondi ici : on arrondit
    # seulement à l'affichage, sinon les petites erreurs s'additionnent sur des milliers de clients.
    return client.proba_churn * client.monthly_charges * horizon


def gain_attendu(client: Client, offre: Offre, horizon: int = HORIZON_MOIS) -> float:
    # On RÉUTILISE perte_attendue au lieu de recopier la formule : si elle change un jour
    # (ex. ajouter une marge), il n'y aura qu'un endroit à modifier.
    return offre.efficacite * perte_attendue(client, horizon) - offre.cout

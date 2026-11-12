"""Étape 3 (corrigé) — Règles d'éligibilité et choix de la meilleure offre pour un client."""

from __future__ import annotations

from typing import Sequence

from .modeles import HORIZON_MOIS, OFFRES, Action, Client, Offre
from .valeur import gain_attendu


def _est_eligible(client: Client, offre: Offre) -> bool:
    # Un tuple vide est « faux » en Python : `not offre.contrats` veut dire « aucune restriction ».
    bon_contrat = not offre.contrats or client.contract in offre.contrats
    # « au moins N » s'écrit >= : un client avec exactement N appels est éligible.
    return (bon_contrat
            and client.support_calls >= offre.appels_min
            and client.tenure_months >= offre.anciennete_min)


def offres_eligibles(client: Client, offres: Sequence[Offre] = OFFRES) -> list[Offre]:
    return [offre for offre in offres if _est_eligible(client, offre)]


def meilleure_offre(client: Client, offres: Sequence[Offre] = OFFRES, horizon: int = HORIZON_MOIS) -> Action | None:
    actions = [Action(client, offre, gain_attendu(client, offre, horizon)) for offre in offres_eligibles(client, offres)]
    rentables = [a for a in actions if a.gain_attendu > 0]
    if not rentables:
        return None
    # Clé de tri en tuple : d'abord le gain le plus grand, puis (à gain égal) le coût le plus
    # petit. À égalité parfaite, max() garde la première rencontrée : l'ordre du catalogue.
    return max(rentables, key=lambda a: (a.gain_attendu, -a.offre.cout))

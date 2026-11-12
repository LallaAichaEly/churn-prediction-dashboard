"""Étape 3 🟡 — Règles d'éligibilité et choix de la meilleure offre pour un client.

Vérifier : python scripts/projet.py --etape 3
"""

from __future__ import annotations

from typing import Sequence

from .modeles import HORIZON_MOIS, OFFRES, Action, Client, Offre  # noqa: F401
from .valeur import gain_attendu  # noqa: F401 — l'étape 2, réutilisée ici


def offres_eligibles(client: Client, offres: Sequence[Offre] = OFFRES) -> list[Offre]:
    """Les offres que l'on a le DROIT de proposer à ce client, dans l'ordre du catalogue `offres`.

    Une offre est éligible si les trois conditions sont vraies :
    - `offre.contrats` est vide (= tous les contrats) OU contient `client.contract` ;
    - le client a appelé le support AU MOINS `offre.appels_min` fois ;
    - le client a AU MOINS `offre.anciennete_min` mois d'ancienneté.

    Exemple avec le catalogue `OFFRES` de `modeles.py` : un client « Annuel », 2 mois d'ancienneté,
    3 appels → [APPEL, SUPPORT] (pas REMISE : moins de 3 mois ; pas ENGAGEMENT : réservé au Mensuel).

    Ce que vous apprenez : traduire des règles en conditions booléennes (`and`, `or`, `not`),
    « au moins » (>= et non >), le fait qu'un tuple vide est « faux » en Python.
    """
    raise NotImplementedError


def meilleure_offre(client: Client, offres: Sequence[Offre] = OFFRES, horizon: int = HORIZON_MOIS) -> Action | None:
    """L'`Action` qui rapporte le plus pour ce client, ou `None` si aucune ne rapporte.

    1. Ne considérez que les offres ÉLIGIBLES (fonction ci-dessus).
    2. Pour chacune, calculez `gain_attendu(client, offre, horizon)` (étape 2).
    3. Renvoyez `Action(client, offre, gain)` pour le plus grand gain. À gain égal, prenez l'offre
       la MOINS CHÈRE ; si l'égalité persiste, la première du catalogue.
    4. Si aucun gain n'est STRICTEMENT positif (ou aucune offre éligible), renvoyez `None` :
       on ne dépense pas d'argent pour en perdre.

    Ce que vous apprenez : construire des objets (`Action(...)`), `max()` avec `key=`, une clé de
    tri en tuple pour départager les ex aequo, renvoyer `None` pour dire « rien ».
    """
    raise NotImplementedError

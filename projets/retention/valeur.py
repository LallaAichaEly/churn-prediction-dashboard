"""Étape 2 🟢 — Traduire une règle métier en code : combien vaut un client, combien rapporte une offre.

Vérifier : python scripts/projet.py --etape 2
"""

from __future__ import annotations

from .modeles import HORIZON_MOIS, Client, Offre  # noqa: F401


def perte_attendue(client: Client, horizon: int = HORIZON_MOIS) -> float:
    """Ce que l'entreprise s'attend à perdre sur ce client pendant `horizon` mois, en euros.

        perte attendue = proba_churn × monthly_charges × horizon

    Exemple : un client à 0,5 de probabilité qui paie 80 €/mois, sur 12 mois → 0,5 × 80 × 12 = 480 €.

    Ne pas arrondir : on arrondit seulement à l'affichage (étape 5). Sinon les petites erreurs
    s'additionnent sur des milliers de clients.

    Ce que vous apprenez : lire une formule métier, les unités (€/mois × mois = €), les valeurs
    par défaut d'un paramètre (`horizon=HORIZON_MOIS`), utiliser un objet (`client.proba_churn`).
    """
    raise NotImplementedError


def gain_attendu(client: Client, offre: Offre, horizon: int = HORIZON_MOIS) -> float:
    """Ce que rapporte EN MOYENNE le fait de proposer `offre` à `client`, coût de l'offre déduit.

        gain attendu = efficacite de l'offre × perte attendue du client − coût de l'offre

    Exemple : même client (perte attendue 480 €), offre d'efficacité 0,30 qui coûte 45 € →
    0,30 × 480 − 45 = 99 €. Le résultat peut être NÉGATIF : l'offre coûte plus qu'elle ne sauve.

    Ce que vous apprenez : réutiliser une fonction (`perte_attendue`) au lieu de recopier sa formule,
    et transmettre `horizon` à la fonction appelée (sinon le paramètre ne sert à rien).
    """
    raise NotImplementedError

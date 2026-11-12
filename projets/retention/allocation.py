"""Étape 4 🟡 — Répartir un budget limité entre les clients.

Vérifier : python scripts/projet.py --etape 4
"""

from __future__ import annotations

from typing import Sequence

from .modeles import Action  # noqa: F401


def allouer_budget(actions: Sequence[Action], budget: float) -> list[Action]:
    """Choisit les actions à lancer sans dépasser `budget` (en euros). Renvoie une NOUVELLE liste.

    `actions` contient au plus une action par client (la meilleure, étape 3). Le marketing ne peut
    pas tout financer : il faut choisir. La règle (un algorithme « glouton ») :

    1. Si `budget` est négatif, levez `ValueError`.
    2. Écartez les actions dont le gain attendu n'est pas strictement positif.
    3. Triez les autres par RENTABILITÉ décroissante (`action.rentabilite`, euros gagnés par euro
       dépensé) ; à rentabilité égale, par `action.client.customer_id` croissant.
    4. Parcourez-les dans cet ordre : prenez l'action si son coût (`action.offre.cout`) tient dans
       ce qui reste du budget, sinon passez à la SUIVANTE (une moins chère peut encore rentrer).
    5. Renvoyez les actions retenues, dans l'ordre où vous les avez prises.

    Ne modifiez pas la liste reçue : l'appelant s'en sert peut-être encore.

    Pourquoi la rentabilité et pas le gain ? Avec 40 € : A coûte 10 € et rapporte 50 €, B coûte
    40 € et rapporte 100 €, C coûte 30 € et rapporte 90 €. Par gain, on prend B seul : 100 €. Par
    rentabilité (A = 5, C = 3, B = 2,5), on prend A puis C : 140 €.

    Ce que vous apprenez : `sorted()` contre `.sort()` (qui modifie la liste), trier avec une clé et
    un ordre décroissant, une boucle avec un « reste » qui diminue, `continue` contre `break`.

    Pour aller plus loin : cette règle est rapide mais pas toujours optimale (c'est le « problème
    du sac à dos »). Cherchez un exemple où elle se trompe : voir projets/README.md.
    """
    raise NotImplementedError

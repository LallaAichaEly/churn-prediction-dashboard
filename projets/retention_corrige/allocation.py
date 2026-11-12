"""Étape 4 (corrigé) — Répartir un budget limité entre les clients."""

from __future__ import annotations

from typing import Sequence

from .modeles import Action


def allouer_budget(actions: Sequence[Action], budget: float) -> list[Action]:
    if budget < 0:
        raise ValueError(f"le budget ne peut pas être négatif ({budget})")

    # sorted() renvoie une NOUVELLE liste : la liste reçue n'est pas modifiée
    # (actions.sort() l'aurait triée chez l'appelant, sans prévenir).
    # Le signe moins trie la rentabilité du plus grand au plus petit, et l'identifiant
    # départage les ex aequo pour que le résultat soit toujours le même.
    candidates = sorted((a for a in actions if a.gain_attendu > 0),
                        key=lambda a: (-a.rentabilite, a.client.customer_id))

    retenues: list[Action] = []
    reste = budget
    for action in candidates:
        if action.offre.cout <= reste:  # <= : une action qui coûte exactement le reste rentre
            retenues.append(action)
            reste -= action.offre.cout
        # Pas de `break` : une action trop chère n'empêche pas d'en prendre une moins chère ensuite.
    return retenues

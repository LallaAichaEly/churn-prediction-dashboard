"""Étape 4 (corrigé) — Décider si le challenger remplace le champion."""

from __future__ import annotations

from .fichiers import euros, nombre
from .modeles import REGLES, Decision, Mesures, Regles

#: 0.82 - 0.81 vaut 0.009999999999999898 en Python : sans cette marge, un gain d'« exactement » 0,01
#: serait refusé. 1e-9 est très petit devant les écarts qui comptent (0,001 d'AUC).
TOLERANCE = 1e-9


def decider(challenger: Mesures, champion: Mesures | None, regles: Regles = REGLES) -> Decision:
    if champion is not None and champion.mois != challenger.mois:
        raise ValueError(f"le champion est mesuré sur {champion.mois} et le challenger sur {challenger.mois} : "
                         f"mesurez les deux modèles sur le MÊME mois de test")

    # Toutes les règles sont examinées, même quand l'une a déjà refusé : l'historique doit dire TOUT ce
    # qui n'allait pas, pour qu'on sache quoi corriger avant le mois prochain.
    raisons: list[str] = []
    promouvoir = True

    if challenger.auc < regles.auc_min - TOLERANCE:
        promouvoir = False
        raisons.append(f"AUC {nombre(challenger.auc)} < {nombre(regles.auc_min)} : modèle inutilisable")

    if champion is None:
        if promouvoir:
            raisons.append("aucun champion en production : le premier modèle utilisable est promu")
        return Decision(promouvoir, tuple(raisons))

    gain = challenger.auc - champion.auc  # un écart ABSOLU d'AUC, pas un pourcentage
    if gain < regles.gain_auc_min - TOLERANCE:
        promouvoir = False
        raisons.append(f"gain d'AUC {nombre(gain)} < {nombre(regles.gain_auc_min)} exigé")
    else:
        raisons.append(f"gain d'AUC {nombre(gain)} >= {nombre(regles.gain_auc_min)}")

    hausse = challenger.cout - champion.cout
    if hausse > regles.hausse_cout_max + TOLERANCE:
        promouvoir = False
        raisons.append(f"coût par client {euros(hausse)} de plus, au-delà de la tolérance ({euros(regles.hausse_cout_max)})")
    else:
        raisons.append(f"coût par client : {euros(hausse)} (tolérance {euros(regles.hausse_cout_max)})")

    return Decision(promouvoir, tuple(raisons))

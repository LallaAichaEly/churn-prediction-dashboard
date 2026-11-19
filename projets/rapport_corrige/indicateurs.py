"""Étape 2 (corrigé) — Les chiffres du mois."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from .modeles import Abonne, Indicateurs
from .periodes import decaler, fin_du_mois


def indicateurs_du_mois(abonnes: Iterable[Abonne], mois: date) -> Indicateurs:
    premier = decaler(mois, 0)  # le 1er du mois, quel que soit le jour reçu
    dernier = fin_du_mois(premier)
    actifs_debut = arrivees = departs = actifs_fin = partis_parmi_actifs = 0
    revenu_perdu = 0.0
    for a in abonnes:
        actif_debut = a.debut < premier and (a.fin is None or a.fin >= premier)
        part = a.fin is not None and premier <= a.fin <= dernier
        actifs_debut += actif_debut
        arrivees += premier <= a.debut <= dernier
        actifs_fin += a.debut <= dernier and (a.fin is None or a.fin > dernier)
        if part:
            departs += 1
            revenu_perdu += a.mensualite
            partis_parmi_actifs += actif_debut  # arrivé ET parti ce mois-ci : pas au dénominateur, donc pas au numérateur
    taux = partis_parmi_actifs / actifs_debut if actifs_debut else None
    return Indicateurs(premier, actifs_debut, arrivees, departs, actifs_fin, taux, revenu_perdu)


def par_segment(abonnes: Iterable[Abonne], mois: date) -> dict[str, Indicateurs]:
    groupes: dict[str, list[Abonne]] = {}
    for a in abonnes:
        groupes.setdefault(a.segment, []).append(a)
    return {segment: indicateurs_du_mois(groupes[segment], mois) for segment in sorted(groupes)}

"""Étape 6 (corrigé) — Le programme du début de mois."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from .formats import ecart_points, nombre, pourcentage
from .indicateurs import indicateurs_du_mois, par_segment
from .modeles import (DEPARTS_AFFICHES, ESPACE_MILLIERS, ESPACE_UNITE, MOIS_HISTORIQUE, Abonne, Rapport,
                      lire_abonnes)
from .page import page_html
from .periodes import decaler, fin_du_mois, lire_mois, nom_du_mois


def construire_rapport(abonnes: Iterable[Abonne], mois: date) -> Rapport:
    abonnes = list(abonnes)  # on va les parcourir plusieurs fois : un générateur serait vide au 2e passage
    premier = decaler(mois, 0)
    dernier = fin_du_mois(premier)
    historique = tuple(indicateurs_du_mois(abonnes, decaler(premier, -k)) for k in range(MOIS_HISTORIQUE - 1, -1, -1))
    partis = sorted((a for a in abonnes if a.fin is not None and premier <= a.fin <= dernier),
                    key=lambda a: (-a.mensualite, a.client))
    return Rapport(mois=premier, actuel=indicateurs_du_mois(abonnes, premier),
                   precedent=indicateurs_du_mois(abonnes, decaler(premier, -1)),
                   segments=par_segment(abonnes, premier), historique=historique,
                   departs=tuple(partis[:DEPARTS_AFFICHES]))


def creer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m projets.rapport",
                                     description="Écrit le rapport churn d'un mois : une page HTML autonome.")
    parser.add_argument("--abonnes", required=True, type=Path, help="le fichier des abonnés (CSV)")
    parser.add_argument("--mois", required=True, help="le mois du rapport, AAAA-MM (ex. 2026-10)")
    parser.add_argument("--sortie", type=Path, default=Path("rapports"), help="dossier des rapports (défaut : rapports/)")
    parser.add_argument("--date-du-jour", help="AAAA-MM-JJ, pour les tests (défaut : aujourd'hui)")
    return parser


def _terminal(texte: str) -> str:
    """Les espaces insécables sont pour la page. Une console Windows dont la sortie est redirigée vers un
    fichier écrit en cp1252, qui ne connaît pas l'espace fine : print() planterait."""
    return texte.replace(ESPACE_MILLIERS, " ").replace(ESPACE_UNITE, " ")


def _erreur(message: str) -> int:
    print(f"Erreur : {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = creer_parser().parse_args(argv)
    try:
        mois = lire_mois(args.mois)
        aujourd_hui = date.fromisoformat(args.date_du_jour) if args.date_du_jour else date.today()
    except ValueError as exc:
        return _erreur(str(exc))
    if fin_du_mois(mois) >= aujourd_hui:
        return _erreur(f"{nom_du_mois(mois)} n'est pas terminé : un rapport fait trop tôt oublie les départs des "
                       f"derniers jours. Revenez le {decaler(mois, 1):%d/%m/%Y}.")

    try:
        abonnes = lire_abonnes(args.abonnes)
    except FileNotFoundError:
        return _erreur(f"fichier introuvable : {args.abonnes}")
    except ValueError as exc:
        return _erreur(str(exc))
    connues = [d for a in abonnes for d in (a.debut, a.fin) if d is not None]
    if not connues or max(connues) < mois:
        fin = f" : le fichier s'arrête au {max(connues):%d/%m/%Y}" if connues else " : le fichier est vide"
        return _erreur(f"aucune donnée pour {nom_du_mois(mois)}{fin}")

    rapport = construire_rapport(abonnes, mois)
    chemin = args.sortie / f"rapport_{mois:%Y-%m}.html"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(page_html(rapport), encoding="utf-8")  # sans encoding : cp1252 sous Windows

    a, p = rapport.actuel, rapport.precedent
    print(_terminal(f"{nom_du_mois(mois).capitalize()} : {nombre(a.actifs_debut, 'client')} au début du mois, "
                    f"{nombre(a.departs, 'départ')}"))
    print(_terminal(f"Taux de churn : {pourcentage(a.taux_churn)} ({ecart_points(p.taux_churn, a.taux_churn)} par "
                    f"rapport à {nom_du_mois(p.mois)})"))
    print(f"Rapport écrit dans {chemin}")
    return 0

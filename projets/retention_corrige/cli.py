"""Étape 6 (corrigé) — Assembler les étapes dans un vrai programme en ligne de commande."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .allocation import allouer_budget
from .clients import lire_clients
from .modeles import HORIZON_MOIS
from .regles import meilleure_offre
from .rapport import ecrire_plan, resumer


def _montant(texte: str) -> float:
    """Type argparse : un montant strictement positif (« 1500 », « 1500.5 » ou « 1500,5 »)."""
    try:
        valeur = float(texte.replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError(f"« {texte} » n'est pas un montant") from None
    if not (valeur > 0 and math.isfinite(valeur)):
        raise argparse.ArgumentTypeError(f"le budget doit être un montant positif, pas « {texte} »")
    return valeur


def creer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m projets.retention",
        description="Choisit quelle offre proposer à quels clients, dans la limite d'un budget.",
    )
    parser.add_argument("--clients", required=True, type=Path,
                        help="CSV des clients avec leur proba_churn (ex. export du tableau de bord)")
    parser.add_argument("--budget", required=True, type=_montant, help="budget total, en euros")
    parser.add_argument("--sortie", type=Path, default=Path("plan_retention.csv"),
                        help="fichier du plan à écrire (défaut : plan_retention.csv)")
    parser.add_argument("--horizon", type=int, default=HORIZON_MOIS,
                        help=f"mois de chiffre d'affaires à protéger (défaut : {HORIZON_MOIS})")
    return parser


def _euros(montant: float) -> str:
    return f"{montant:,.2f} €".replace(",", " ")


def main(argv: list[str] | None = None) -> int:
    args = creer_parser().parse_args(argv)  # argv=None : argparse lit sys.argv

    # Une erreur prévisible (fichier absent, mauvaises colonnes) mérite un message clair et un
    # code de sortie 1, pas une traceback de 20 lignes.
    try:
        clients, erreurs = lire_clients(args.clients)
    except FileNotFoundError:
        print(f"Erreur : fichier introuvable : {args.clients}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    # Les avertissements vont sur stderr : stdout reste propre si on redirige le résumé.
    for erreur in erreurs:
        print(f"Attention : {erreur}", file=sys.stderr)

    candidates = []
    for client in clients:
        action = meilleure_offre(client, horizon=args.horizon)
        if action is not None:
            candidates.append(action)
    plan = allouer_budget(candidates, args.budget)
    chemin = ecrire_plan(plan, args.sortie)

    r = resumer(plan, len(clients))
    print(f"Clients analysés  : {r['clients_analyses']} ({len(erreurs)} ligne(s) ignorée(s))")
    print(f"Clients ciblés    : {r['clients_cibles']}")
    print(f"Coût du plan      : {_euros(r['cout_total'])} sur un budget de {_euros(args.budget)}")
    print(f"Gain attendu      : {_euros(r['gain_attendu'])} (ROI {r['roi']:.1f})")
    for code, nombre in sorted(r["par_offre"].items()):
        print(f"  - {code:<11} {nombre}")
    print(f"Plan écrit dans   : {chemin}")
    return 0

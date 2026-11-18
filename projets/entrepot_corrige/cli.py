"""Étape 6 (corrigé) — Le programme de la nuit : charger, puis exporter la table du projet 1."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

from .fichier import charger_fichier
from .modeles import CHARGE, DEJA_CHARGE, HORIZON_JOURS, REFUSE, ouvrir_base
from .table import construire_table


def _charger(args) -> int:
    dossier = Path(args.dossier)
    if not dossier.is_dir():  # AVANT d'ouvrir la base : sinon on créerait une base vide pour rien
        print(f"Dossier introuvable : {dossier}", file=sys.stderr)
        return 1
    a_charger = sorted(p for p in dossier.iterdir() if p.is_file() and (
        (p.name.startswith("crm_") and p.suffix == ".csv") or (p.name.startswith("support_") and p.suffix == ".jsonl")))

    connexion = ouvrir_base(args.base)
    try:
        comptes = {CHARGE: 0, DEJA_CHARGE: 0, REFUSE: 0}
        tous_les_rejets = []
        largeur = max((len(p.name) for p in a_charger), default=0)
        for chemin in a_charger:
            try:
                bilan = charger_fichier(connexion, chemin)
            except ValueError as exc:  # en-tête inattendu : ce n'est pas le bon fichier
                print(f"❌ {chemin.name:<{largeur}}  refusé : {exc}", file=sys.stderr)
                comptes[REFUSE] += 1
                continue
            comptes[bilan.statut] += 1
            tous_les_rejets += bilan.rejets
            if bilan.statut == DEJA_CHARGE:
                print(f"⏭️  {bilan.fichier:<{largeur}}  déjà chargé")
            elif bilan.statut == REFUSE:
                print(f"❌ {bilan.fichier:<{largeur}}  refusé : {len(bilan.rejets)} ligne(s) illisible(s), "
                      f"plus de 10 % du fichier — rien n'a été chargé")
            else:
                icone = "⚠️ " if bilan.rejets else "✅"
                print(f"{icone} {bilan.fichier:<{largeur}}  chargé : {bilan.lignes} ligne(s) en base, "
                      f"{len(bilan.rejets)} rejet(s)")

        contrats = connexion.execute("SELECT count(*) FROM contrats").fetchone()[0]
        tickets = connexion.execute("SELECT count(*) FROM tickets").fetchone()[0]
    finally:
        connexion.close()

    print(f"\n{len(a_charger)} fichier(s) : {comptes[CHARGE]} chargé(s), {comptes[DEJA_CHARGE]} déjà chargé(s), "
          f"{comptes[REFUSE]} refusé(s)")
    print(f"Base {args.base} : {contrats} contrats, {tickets} tickets")
    if tous_les_rejets:
        with open(args.rejets, "w", encoding="utf-8", newline="") as f:
            ecrivain = csv.writer(f)
            ecrivain.writerow(["fichier", "ligne", "raison"])
            ecrivain.writerows((r.fichier, r.ligne, r.raison) for r in tous_les_rejets)
        print(f"{len(tous_les_rejets)} ligne(s) écartée(s), détail dans {args.rejets}")
    return 1 if comptes[REFUSE] else 0


def _exporter(args) -> int:
    if not Path(args.base).is_file():  # ouvrir_base créerait une base vide
        print(f"Base introuvable : {args.base} (lancez d'abord la commande charger)", file=sys.stderr)
        return 1
    try:
        photo = date.fromisoformat(args.photo)
    except ValueError:
        print(f"--photo : date illisible {args.photo!r}, attendu AAAA-MM-JJ", file=sys.stderr)
        return 1

    connexion = ouvrir_base(args.base)
    try:
        table = construire_table(connexion, photo, args.horizon)
    except ValueError as exc:
        print(f"Export impossible : {exc}", file=sys.stderr)
        return 1
    finally:
        connexion.close()

    sortie = Path(args.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(sortie, index=False)
    taux = f"{table['churn'].mean():.1%}".replace(".", ",").replace("%", " %") if len(table) else "—"
    print(f"{len(table)} clients au {photo.isoformat()}, départs dans les {args.horizon} jours : {taux}")
    print(f"Table écrite dans {sortie}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m projets.entrepot",
                                     description="L'entrepôt des clients : des exports bruts à la table du projet 1.")
    commandes = parser.add_subparsers(dest="commande", required=True)

    charger = commandes.add_parser("charger", help="charge les exports d'un dossier dans la base")
    charger.add_argument("--dossier", required=True)
    charger.add_argument("--base", required=True)
    charger.add_argument("--rejets", default="rejets.csv")

    exporter = commandes.add_parser("exporter", help="écrit la table du projet 1 à la date d'une photo")
    exporter.add_argument("--base", required=True)
    exporter.add_argument("--photo", required=True, help="AAAA-MM-JJ")
    exporter.add_argument("--sortie", required=True)
    exporter.add_argument("--horizon", type=int, default=HORIZON_JOURS)

    args = parser.parse_args(argv)
    return _charger(args) if args.commande == "charger" else _exporter(args)

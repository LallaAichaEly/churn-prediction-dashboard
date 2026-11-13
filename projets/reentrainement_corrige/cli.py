"""Étape 6 (corrigé) — Le programme à trois commandes : entrainer, historique, revenir."""

from __future__ import annotations

import argparse
import sys

from .decision import decider
from .decoupage import decouper
from .entrainement import entrainer
from .evaluation import mesurer
from .fichiers import champion, lire_lots, nombre, euros, tableau, versions
from .lots import problemes
from .modeles import HISTORIQUE, MOIS, ModeleAltere
from .registre import charger, enregistrer, promouvoir, revenir


def _entrainer(args) -> int:
    lots = lire_lots(args.lots)
    trouves = problemes(lots)
    if trouves:
        # Rien n'est entraîné, rien n'est écrit : un modèle appris sur un lot abîmé est pire que pas de modèle.
        print(f"Lot refusé : {len(trouves)} problème(s), aucun modèle entraîné.", file=sys.stderr)
        for probleme in trouves:
            print(f"  - {probleme}", file=sys.stderr)
        return 1

    train, test = decouper(lots, args.mois, args.historique)
    print(f"Entraînement : {len(train)} clients ({train[MOIS].min()} → {train[MOIS].max()}), "
          f"test : {len(test)} clients ({args.mois})")
    modele = entrainer(train)
    mesures = mesurer(modele, test)

    # Le champion est RE-MESURÉ sur le même mois que le challenger. Ses mesures enregistrées datent de
    # son propre mois de test : les comparer à celles du challenger, c'est comparer deux examens différents.
    actuel = champion(args.registre)
    mesures_champion = mesurer(charger(args.registre, actuel.numero), test) if actuel else None
    decision = decider(mesures, mesures_champion)

    # Enregistré même s'il n'est pas promu : on garde la trace de ce qui a été essayé et pourquoi refusé.
    version = enregistrer(args.registre, modele, mesures)
    if decision.promouvoir:
        promouvoir(args.registre, version.numero)

    lignes = [(f"champion v{actuel.numero:03d}", mesures_champion)] if actuel else []
    print()
    print(tableau(lignes + [(f"challenger v{version.numero:03d}", mesures)]))
    print()
    if decision.promouvoir:
        print(f"✅ v{version.numero:03d} promue championne :")
    else:
        print(f"❌ v{version.numero:03d} enregistrée, mais pas promue (le champion reste "
              f"v{actuel.numero:03d}) :" if actuel else f"❌ v{version.numero:03d} enregistrée, mais pas promue :")
    for raison in decision.raisons:
        print(f"   - {raison}")
    return 0


def _historique(args) -> int:
    toutes = versions(args.registre)
    if not toutes:
        print(f"Registre vide ({args.registre}) : lancez d'abord `entrainer`.")
        return 0
    actuel = champion(args.registre)
    for v in toutes:
        marque = "  ← champion" if actuel and v.numero == actuel.numero else ""
        m = v.mesures
        print(f"v{v.numero:03d}  testé sur {m.mois}  AUC {nombre(m.auc)}  coût {euros(m.cout)}/client{marque}")
    return 0


def _revenir(args) -> int:
    v = revenir(args.registre)
    print(f"Retour à v{v.numero:03d} (testé sur {v.mesures.mois}, AUC {nombre(v.mesures.auc)}) : c'est le "
          f"modèle en production.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m projets.reentrainement",
                                     description="Réentraîner le modèle de churn chaque mois, sans casser la production.")
    commandes = parser.add_subparsers(dest="commande", required=True)

    p = commandes.add_parser("entrainer", help="entraîne un challenger, le compare au champion, le promeut s'il est meilleur")
    p.add_argument("--lots", required=True, help="fichier CSV des lots mensuels")
    p.add_argument("--mois", required=True, help="mois de test, AAAA-MM (on apprend sur les mois d'avant)")
    p.add_argument("--registre", required=True, help="dossier du registre des modèles")
    p.add_argument("--historique", type=int, default=HISTORIQUE, help=f"mois d'historique (défaut : {HISTORIQUE})")
    p.set_defaults(action=_entrainer)

    p = commandes.add_parser("historique", help="liste les versions et le champion")
    p.add_argument("--registre", required=True)
    p.set_defaults(action=_historique)

    p = commandes.add_parser("revenir", help="remet en production le champion précédent")
    p.add_argument("--registre", required=True)
    p.set_defaults(action=_revenir)

    args = parser.parse_args(argv)
    try:
        return args.action(args)
    except (ValueError, ModeleAltere, OSError) as exc:
        # Une erreur attendue (mois absent, registre altéré, fichier introuvable) : une phrase, pas une traceback.
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

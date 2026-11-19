"""Étape 7 🔴 — Écrire le test qui protège `indicateurs_du_mois` (étape 2).

Vérifier : python scripts/projet.py --projet 7 --etape 7
"""

from __future__ import annotations

from datetime import date  # noqa: F401 — vous en aurez besoin

from .modeles import Abonne


def _abonne(client: str, debut: str, fin: str | None = None, mensualite: float = 50.0,
            segment: str = "Particulier") -> Abonne:
    """Fabrique un abonné de test (DÉJÀ ÉCRIT) : `_abonne("A", "2026-01-10", fin="2026-10-01", mensualite=30.0)`."""
    return Abonne(client=client, nom=f"Client {client}", segment=segment, debut=date.fromisoformat(debut),
                  fin=date.fromisoformat(fin) if fin else None, mensualite=mensualite)


def tester_indicateurs_du_mois(indicateurs_du_mois) -> None:
    """Vérifie, avec des `assert`, qu'une fonction `indicateurs_du_mois(abonnes, mois)` respecte le contrat
    de l'étape 2 (relisez sa docstring dans `indicateurs.py`).

    Le taux de churn est LE chiffre que la direction regarde chaque mois. Un bug ici ne fait rien
    planter : un dénominateur pris en fin de mois au lieu du début, un départ du 1er oublié, et le
    taux baisse un peu… Personne ne s'en aperçoit, et une décision se prend sur un chiffre faux.

    Comme aux projets précédents, c'est VOUS qui écrivez le test, et le correcteur l'exécute :
    - sur la version JUSTE d'`indicateurs_du_mois` : votre test doit passer ;
    - sur 13 versions BOGUÉES, toutes plausibles : votre test doit échouer sur CHACUNE.

    Fabriquez quelques abonnés avec `_abonne(...)`, choisissez un mois, et pour chaque règle placez un
    abonné PILE sur la frontière : arrivé le 1er, parti le 1er, parti le dernier jour, parti la veille
    du mois, arrivé le lendemain, arrivé ET parti pendant le mois… Calculez À LA MAIN chaque indicateur
    AVANT d'écrire les `assert`, puis pensez aux cas à part : un mois où personne n'était abonné au
    début, décembre, un jour qui n'est pas le 1er. Vérifiez aussi que la liste reçue n'a pas changé.

    Ce que vous apprenez : écrire un test qui protège une définition métier, viser les frontières,
    tester AUSSI les cas rares (décembre, le mois vide), où les bugs se cachent le mieux.
    """
    raise NotImplementedError

"""Étape 7 🔴 — Écrire le test qui protège `effet` (étape 5).

Vérifier : python scripts/projet.py --projet 3 --etape 7
"""

from __future__ import annotations

import math  # noqa: F401

from .modeles import GROUPE_OFFRE, GROUPE_TEMOIN, Comptage  # noqa: F401


def _comptage(groupe: str, clients: int, avec_resultat: int, partis: int) -> Comptage:
    """Fabrique un comptage de test (DÉJÀ ÉCRIT).

    Exemple : `_comptage(GROUPE_TEMOIN, clients=120, avec_resultat=100, partis=40)` : 120 témoins,
    dont 100 avec un résultat connu, dont 40 partis (taux de départ : 40 %).
    """
    return Comptage("TEST", groupe, clients, avec_resultat, partis, cout=10.0 * clients)


def tester_effet(effet) -> None:
    """Vérifie, avec des `assert`, qu'une fonction `effet(offre, temoin, niveau=0.95)` respecte le
    contrat de l'étape 5 (relisez sa docstring dans `statistiques.py`).

    Comme à l'étape 7 du projet 2, c'est VOUS qui écrivez le test, et le correcteur l'exécute :
    - sur une version JUSTE de `effet` : votre test doit passer ;
    - sur 7 versions BOGUÉES, toutes plausibles : votre test doit échouer sur CHACUNE.

    Calculez vos valeurs attendues à la main (ou à la calculatrice) AVANT d'écrire l'`assert`,
    jamais en recopiant ce que renvoie votre propre code. Pour comparer des nombres à virgule,
    utilisez `math.isclose(a, b, rel_tol=1e-5)` : 0.1 + 0.2 != 0.3 en Python.

    Choisissez des exemples qui séparent les erreurs : si `clients` et `avec_resultat` sont égaux
    dans tous vos comptages, une version qui confond les deux passera votre test.

    Ce que vous apprenez : tester du calcul numérique, choisir des exemples discriminants, tester
    un paramètre (`niveau`) et un cas d'erreur.
    """
    raise NotImplementedError

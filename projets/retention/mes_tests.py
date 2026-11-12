"""Étape 7 🔴 — Écrire le test qui protège `allouer_budget` (étape 4).

Vérifier : python scripts/projet.py --etape 7
"""

from __future__ import annotations

from .modeles import Action, Client, Offre  # noqa: F401


def _action(identifiant: str, cout: float, gain: float) -> Action:
    """Fabrique une action de test (DÉJÀ ÉCRIT). Rentabilité = gain / cout.

    Exemple : `_action("A", cout=10.0, gain=50.0)` a une rentabilité de 5.
    """
    client = Client(identifiant, 0.5, 50.0, 12, "Mensuel", 0)
    return Action(client, Offre(f"OFFRE_{identifiant}", "offre de test", cout, 0.3), gain)


def tester_allouer_budget(allouer) -> None:
    """Vérifie, avec des `assert`, qu'une fonction `allouer(actions, budget)` respecte la règle
    de l'étape 4 (relisez sa docstring dans `allocation.py`).

    Les rôles s'inversent : c'est VOUS qui écrivez le test. Le correcteur l'exécute :
    - sur une version JUSTE de `allouer_budget` : votre test doit passer (aucune erreur) ;
    - sur 7 versions BOGUÉES, toutes plausibles (elles ressemblent aux erreurs des débutants
      comme des confirmés) : votre test doit échouer sur CHACUNE.

    Utilisez `_action(...)` ci-dessus pour fabriquer vos exemples. Un bon test ne vérifie pas que
    « ça tourne » : chaque `assert` choisit un exemple où une version fausse donnerait un autre
    résultat. Relisez chaque phrase de la règle et demandez-vous : « quelle erreur pourrait-on
    faire ici, et quel exemple la révélerait ? »

    Pour un `ValueError` attendu, sans pytest :

        try:
            allouer(actions, -1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("un budget négatif doit lever ValueError")

    Ce que vous apprenez : écrire des tests qui attrapent vraiment des bugs (c'est l'idée du
    « mutation testing »), penser aux cas limites, tester ce qu'une fonction NE doit PAS faire.
    """
    raise NotImplementedError

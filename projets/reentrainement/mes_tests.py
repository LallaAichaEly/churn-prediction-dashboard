"""Étape 7 🔴 — Écrire le test qui protège `decider` (étape 4).

Vérifier : python scripts/projet.py --projet 5 --etape 7
"""

from __future__ import annotations

from .modeles import Mesures, Regles  # noqa: F401


def _mesures(auc: float, cout: float = 50.0, mois: str = "2026-09") -> Mesures:
    """Fabrique des mesures de test (DÉJÀ ÉCRIT).

    Exemple : `_mesures(0.80, cout=45.0)` : un modèle d'AUC 0,80 qui coûte 45 € par client, mesuré
    sur 2026-09.
    """
    return Mesures(mois=mois, clients=500, auc=auc, cout=cout, taux_alerte=0.3)


def tester_decider(decider) -> None:
    """Vérifie, avec des `assert`, qu'une fonction `decider(challenger, champion, regles=REGLES)`
    respecte le contrat de l'étape 4 (relisez sa docstring dans `decision.py`).

    `decider` est la fonction qui choisit, chaque mois, le modèle qui part en production. Un bug ici
    ne fait pas planter le programme : il met en production un modèle pire que l'ancien, sans
    que personne ne s'en aperçoive. C'est exactement le genre de code qu'on protège par un test.

    Comme aux projets 2 et 3, c'est VOUS qui écrivez le test, et le correcteur l'exécute :
    - sur la version JUSTE de `decider` : votre test doit passer ;
    - sur 11 versions BOGUÉES, toutes plausibles : votre test doit échouer sur CHACUNE.

    Pour chaque règle, cherchez un cas où une erreur classique donne une réponse DIFFÉRENTE de la
    bonne : un plancher oublié quand il n'y a pas de champion, un gain relatif au lieu d'absolu, un
    flottant comparé sans marge, un coût lu à l'envers, des règles écrites en dur au lieu de lire le
    paramètre `regles`, un examen qui s'arrête à la première règle enfreinte, deux mois différents
    acceptés… Pour vérifier qu'une erreur est levée :

        try:
            decider(...)
        except ValueError:
            pass
        else:
            raise AssertionError("decider aurait dû lever ValueError")

    Ce que vous apprenez : tester une règle métier, choisir des exemples qui séparent la bonne et la
    mauvaise version, tester un paramètre et un cas d'erreur.
    """
    raise NotImplementedError

"""Étape 7 (corrigé) — Un test qui attrape les bugs de `decider`.

Chaque bloc vise UN bug précis, avec des valeurs choisies pour que la bonne et la mauvaise version
donnent des réponses DIFFÉRENTES.
"""

from __future__ import annotations

from .modeles import Mesures, Regles


def _mesures(auc: float, cout: float = 50.0, mois: str = "2026-09") -> Mesures:
    return Mesures(mois=mois, clients=500, auc=auc, cout=cout, taux_alerte=0.3)


def tester_decider(decider) -> None:
    # 1. Pas de champion : un modèle utilisable est promu… mais pas un modèle sous le plancher (0,70).
    assert decider(_mesures(0.75), None).promouvoir
    assert not decider(_mesures(0.65), None).promouvoir

    # 2. Le plancher compte AUSSI face à un champion très mauvais : 0,69 bat 0,60 de loin, mais reste inutilisable.
    assert not decider(_mesures(0.69), _mesures(0.60)).promouvoir

    # 3. Le gain d'AUC est un écart ABSOLU : 0,758 contre 0,75, c'est +0,008 < 0,01 (alors que +1,07 % en relatif).
    assert not decider(_mesures(0.758), _mesures(0.75)).promouvoir
    assert decider(_mesures(0.80), _mesures(0.75)).promouvoir

    # 4. Un gain d'exactement 0,01 suffit : 0.82 - 0.81 vaut 0.00999…98 en Python, il faut une tolérance.
    assert decider(_mesures(0.82), _mesures(0.81)).promouvoir

    # 5. Le coût : un challenger meilleur en AUC mais plus cher est refusé ; moins cher, il passe.
    assert not decider(_mesures(0.85, cout=60.0), _mesures(0.80, cout=50.0)).promouvoir
    assert decider(_mesures(0.85, cout=40.0), _mesures(0.80, cout=50.0)).promouvoir

    # 6. Les règles passées en paramètre sont appliquées (pas les valeurs par défaut écrites en dur).
    souples = Regles(auc_min=0.70, gain_auc_min=0.0, hausse_cout_max=5.0)
    assert decider(_mesures(0.80, cout=54.0), _mesures(0.80, cout=50.0), souples).promouvoir
    strictes = Regles(auc_min=0.90, gain_auc_min=0.01, hausse_cout_max=0.0)
    assert not decider(_mesures(0.85), None, strictes).promouvoir

    # 7. Deux règles enfreintes : deux raisons (on ne s'arrête pas à la première).
    refus = decider(_mesures(0.65, cout=60.0), _mesures(0.66, cout=50.0))
    assert not refus.promouvoir and len(refus.raisons) >= 3   # plancher, gain et coût

    # 8. Deux mois différents : on refuse de comparer.
    try:
        decider(_mesures(0.80, mois="2026-09"), _mesures(0.75, mois="2026-08"))
    except ValueError:
        pass
    else:
        raise AssertionError("champion et challenger mesurés sur deux mois différents : decider doit lever ValueError")

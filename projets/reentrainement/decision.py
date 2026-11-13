"""Étape 4 🔴 — Décider si le challenger remplace le champion.

Vérifier : python scripts/projet.py --projet 5 --etape 4
"""

from __future__ import annotations

from .fichiers import euros, nombre  # noqa: F401 — pour écrire les raisons au format français
from .modeles import REGLES, Decision, Mesures, Regles  # noqa: F401


def decider(challenger: Mesures, champion: Mesures | None, regles: Regles = REGLES) -> Decision:
    """Renvoie une `Decision` : faut-il mettre le challenger en production, et pourquoi.

    `challenger` : les mesures du modèle tout juste entraîné ; `champion` : celles du modèle en
    production, mesuré sur le MÊME mois (ou None s'il n'y a encore aucun modèle en production).

    Les règles (lisez les seuils dans le paramètre `regles`, pas dans `REGLES` : l'équipe peut
    essayer d'autres règles) :

    0. Si `champion` n'est pas None et `champion.mois != challenger.mois`, levez `ValueError` :
       deux modèles mesurés sur deux mois différents ont passé deux examens différents.
    1. Le plancher : si `challenger.auc < regles.auc_min`, refus (le modèle est inutilisable). Cette
       règle vaut TOUJOURS, même sans champion, même face à un champion encore pire.
    2. Sans champion, un challenger qui passe le plancher est promu.
    3. Le gain : `challenger.auc - champion.auc` (un écart ABSOLU, pas un pourcentage) doit être
       au moins `regles.gain_auc_min`. Un gain minuscule peut venir du hasard du mois de test ; changer
       de modèle a un coût (les conseillers voient leurs listes changer) : on exige un vrai progrès.
    4. Le coût : `challenger.cout - champion.cout` ne doit pas dépasser `regles.hausse_cout_max`.
    5. Promu seulement si AUCUNE règle ne refuse.

    Piège des nombres à virgule : en Python, `0.82 - 0.81` vaut `0.009999999999999898`. Un gain
    d'exactement 0,01 serait refusé par `gain < 0.01`. Comparez avec une petite marge :
    `gain < regles.gain_auc_min - 1e-9` pour refuser (et de même pour le plancher et le coût).

    `Decision.raisons` est un TUPLE de phrases, jamais vide : une phrase par règle examinée (le
    plancher s'il refuse, puis le gain et le coût quand il y a un champion), y compris quand le
    modèle est promu (« aucun champion en production », « gain d'AUC 0,048 >= 0,010 »…). N'arrêtez
    pas l'examen à la première règle qui refuse : l'historique doit dire TOUT ce qui n'allait pas.
    `nombre(0.048)` et `euros(-13.1)` (fichiers.py) écrivent les nombres à la française.

    Ce que vous apprenez : écrire une règle métier comme une fonction pure (facile à tester),
    comparer des flottants, expliquer une décision automatique.
    """
    raise NotImplementedError

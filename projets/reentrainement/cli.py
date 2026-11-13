"""Étape 6 🟡 — Le programme à trois commandes : entrainer, historique, revenir.

Vérifier : python scripts/projet.py --projet 5 --etape 6
"""

from __future__ import annotations

import argparse  # noqa: F401 — vous en aurez besoin
import sys  # noqa: F401

from .decision import decider  # noqa: F401
from .decoupage import decouper  # noqa: F401
from .entrainement import entrainer  # noqa: F401
from .evaluation import mesurer  # noqa: F401
from .fichiers import champion, euros, lire_lots, nombre, tableau, versions  # noqa: F401
from .lots import problemes  # noqa: F401
from .modeles import HISTORIQUE, MOIS, ModeleAltere  # noqa: F401
from .registre import charger, enregistrer, promouvoir, revenir  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    """Le programme que l'équipe lancera chaque mois. Renvoie le code de sortie (0 = tout va bien).

    Trois sous-commandes (`argparse`, `add_subparsers(dest="commande", required=True)`) :

    `entrainer --lots CSV --mois AAAA-MM --registre DOSSIER [--historique N]`
      1. Lit les lots (`lire_lots`) et appelle `problemes`. S'il y en a : affichez-les tous sur
         stderr et renvoyez 1, SANS rien entraîner ni écrire.
      2. `decouper`, puis affichez combien de clients servent à l'entraînement et au test.
      3. `entrainer(train)` (entrainement.py, déjà écrit) donne le challenger ; `mesurer` le mesure
         sur test.
      4. S'il y a un champion (`champion(dossier)`, fichiers.py) : chargez-le (`charger`) et
         MESUREZ-LE SUR LE MÊME `test`. Ses mesures enregistrées datent de son propre mois de test :
         s'en servir, ce serait comparer deux examens différents (et `decider` refuserait).
      5. `decider(mesures_challenger, mesures_champion)`.
      6. Enregistrez le challenger dans tous les cas (`enregistrer`) : on garde la trace de ce qui a
         été essayé, même refusé. Promouvez-le seulement si la décision le dit.
      7. Affichez les mesures (`tableau`, fichiers.py), puis la décision : le numéro de la version
         (« v003 ») et chaque raison.

    `historique --registre DOSSIER`
      Une ligne par version (numéro « v001 », mois de test, AUC, coût) ; la ligne du champion
      actuel contient le mot « champion ». Un registre vide n'est pas une erreur : dites-le, et
      renvoyez 0.

    `revenir --registre DOSSIER`
      `revenir`, puis dites quelle version est de nouveau en production (« v001 »).

    Les erreurs attendues (`ValueError` : mois absent, aucun champion précédent… ; `ModeleAltere` ;
    `OSError` : fichier introuvable) s'affichent en UNE phrase sur stderr (`file=sys.stderr`) avec
    un code de sortie 1, jamais en traceback.

    Exemple de sortie de `entrainer` :

        Entraînement : 3300 clients (2025-10 → 2026-08), test : 300 clients (2026-09)

                            AUC   coût/client   alertes
        champion v001     0,789       73,83 €    40,3 %
        challenger v003   0,837       59,17 €    44,0 %

        ✅ v003 promue championne :
           - gain d'AUC 0,048 >= 0,010
           - coût par client : −14,67 € (tolérance 0,00 €)

    Ce que vous apprenez : assembler les étapes en un vrai programme, l'ordre des opérations quand
    une erreur peut survenir à chaque ligne, stdout contre stderr, les codes de sortie.
    """
    raise NotImplementedError

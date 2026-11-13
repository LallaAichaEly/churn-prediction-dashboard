"""Étape 5 🟡 — L'offre a-t-elle vraiment évité des départs ?

Vérifier : python scripts/projet.py --projet 3 --etape 5
"""

from __future__ import annotations

import math  # noqa: F401 — vous en aurez besoin
from statistics import NormalDist  # noqa: F401

from .modeles import Comptage, Effet  # noqa: F401


def effet(offre: Comptage, temoin: Comptage, niveau: float = 0.95) -> Effet:
    """Compare le groupe offre au groupe témoin et renvoie un `Effet` (voir `modeles.py`).

    Notations : n_o = `offre.avec_resultat`, n_t = `temoin.avec_resultat`.

    1. Si `niveau` n'est pas strictement entre 0 et 1, ou si n_o ou n_t vaut 0, levez `ValueError`
       (pas de `ZeroDivisionError` : l'appelant doit pouvoir attraper une erreur qui a un sens).
    2. Les taux de départ se calculent sur les résultats CONNUS :
           taux_offre = offre.partis / n_o        taux_temoin = temoin.partis / n_t
       Un client dont on ne sait rien n'est pas « resté » : le compter ferait baisser le taux.
    3. `ecart = taux_temoin − taux_offre` : positif quand il y a MOINS de départs avec l'offre.
    4. L'intervalle de confiance de l'écart (approximation normale, dite « de Wald ») :
           erreur = √( taux_offre × (1 − taux_offre) / n_o  +  taux_temoin × (1 − taux_temoin) / n_t )
           z      = NormalDist().inv_cdf((1 + niveau) / 2)      # ≈ 1,96 pour 0,95 ; ≈ 1,645 pour 0,90
           ic_bas = ecart − z × erreur          ic_haut = ecart + z × erreur
    5. `clients_sauves = ecart × offre.clients` : l'écart s'applique à tous les clients qui ont reçu
       l'offre (y compris ceux dont on attend le résultat), et à eux seuls : les témoins n'ont rien reçu.
    6. `significatif` est vrai si l'intervalle EXCLUT 0, dans un sens ou dans l'autre
       (`ic_bas > 0` : l'offre retient ; `ic_haut < 0` : elle fait fuir).
    7. `niveau` est recopié dans l'`Effet`.

    Pourquoi un intervalle ? Avec 10 témoins, 4 départs (40 %) contre 6 sur 20 (30 %) : l'écart de
    10 points peut venir du hasard. L'intervalle le dit : [−26 ; +46] (en points) contient 0. Annoncer « l'offre
    marche » sur un tel résultat, c'est décider d'un budget sur un coup de dé.

    Ce que vous apprenez : traduire des formules en code, `math.sqrt`, `statistics.NormalDist`,
    choisir le bon dénominateur, distinguer « écart positif » et « écart significatif ».
    """
    raise NotImplementedError

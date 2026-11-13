"""Étape 5 (corrigé) — L'offre a-t-elle vraiment évité des départs ?"""

from __future__ import annotations

import math
from statistics import NormalDist

from .modeles import Comptage, Effet


def effet(offre: Comptage, temoin: Comptage, niveau: float = 0.95) -> Effet:
    if not 0 < niveau < 1:
        raise ValueError(f"le niveau de confiance doit être strictement entre 0 et 1, pas {niveau}")
    if offre.avec_resultat == 0 or temoin.avec_resultat == 0:
        raise ValueError("il faut au moins un résultat connu dans chaque groupe")

    # Les taux se calculent sur les résultats CONNUS : un client dont on ne sait rien n'est pas
    # « resté ». Le compter comme resté ferait baisser artificiellement le taux de départ.
    n_o, n_t = offre.avec_resultat, temoin.avec_resultat
    p_o, p_t = offre.partis / n_o, temoin.partis / n_t
    ecart = p_t - p_o  # positif quand il y a MOINS de départs avec l'offre

    # Erreur type de la différence de deux proportions indépendantes : les variances s'ajoutent.
    erreur = math.sqrt(p_o * (1 - p_o) / n_o + p_t * (1 - p_t) / n_t)
    # z tel que la loi normale garde `niveau` au centre : 1,96 pour 95 %, 1,645 pour 90 %.
    z = NormalDist().inv_cdf((1 + niveau) / 2)
    bas, haut = ecart - z * erreur, ecart + z * erreur

    return Effet(
        niveau=niveau,
        taux_offre=p_o,
        taux_temoin=p_t,
        ecart=ecart,
        ic_bas=bas,
        ic_haut=haut,
        # Seuls les clients du groupe offre ont reçu l'offre : c'est sur eux que l'écart s'applique,
        # y compris ceux dont on ne connaît pas encore le résultat.
        clients_sauves=ecart * offre.clients,
        # Significatif si l'intervalle exclut 0, dans un sens ou dans l'autre. Un écart positif
        # ne suffit pas : avec 20 clients, il peut venir du hasard.
        significatif=bas > 0 or haut < 0,
    )

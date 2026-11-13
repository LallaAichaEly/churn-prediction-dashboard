"""Étape 7 (corrigé) — Un test qui attrape les bugs de `effet`.

Chaque bloc vise UN bug précis, avec des valeurs calculées à la main AVANT de lancer le code.
"""

from __future__ import annotations

import math

from .modeles import GROUPE_OFFRE, GROUPE_TEMOIN, Comptage


def _comptage(groupe: str, clients: int, avec_resultat: int, partis: int) -> Comptage:
    return Comptage("TEST", groupe, clients, avec_resultat, partis, cout=10.0 * clients)


def tester_effet(effet) -> None:
    # 500 clients ont reçu l'offre, on connaît le sort de 400 : 100 sont partis (25 %).
    # 120 témoins, résultat connu pour 100 : 40 sont partis (40 %).
    offre = _comptage(GROUPE_OFFRE, clients=500, avec_resultat=400, partis=100)
    temoin = _comptage(GROUPE_TEMOIN, clients=120, avec_resultat=100, partis=40)
    e = effet(offre, temoin)

    # 1. Les taux se calculent sur les résultats CONNUS (400 et 100), pas sur tous les clients.
    assert math.isclose(e.taux_offre, 0.25) and math.isclose(e.taux_temoin, 0.40)
    # 2. Le sens de l'écart : témoin − offre, positif quand l'offre retient.
    assert math.isclose(e.ecart, 0.15)
    # 3. Départs évités = écart × clients du groupe OFFRE (500) : ni les témoins, ni 400.
    assert math.isclose(e.clients_sauves, 75.0)
    # 4. L'intervalle : ± 1,96 erreur type, l'erreur type étant une RACINE carrée.
    erreur = math.sqrt(0.25 * 0.75 / 400 + 0.40 * 0.60 / 100)  # ≈ 0,0536
    assert math.isclose(e.ic_bas, 0.15 - 1.959964 * erreur, rel_tol=1e-5)
    assert math.isclose(e.ic_haut, 0.15 + 1.959964 * erreur, rel_tol=1e-5)
    assert e.significatif  # [0,045 ; 0,255] exclut 0
    # 5. Le niveau change la largeur : 1,645 erreur type à 90 %.
    e90 = effet(offre, temoin, niveau=0.90)
    assert math.isclose(e90.ic_haut - e90.ic_bas, 2 * 1.644854 * erreur, rel_tol=1e-5)
    # 6. Un écart positif n'est pas forcément significatif : 30 % contre 40 % sur 20 et 10 clients.
    petit = effet(_comptage(GROUPE_OFFRE, 20, 20, 6), _comptage(GROUPE_TEMOIN, 10, 10, 4))
    assert petit.ecart > 0 and not petit.significatif
    # 7. Aucun résultat dans un groupe : une erreur explicite, pas une division par zéro.
    try:
        effet(offre, _comptage(GROUPE_TEMOIN, 120, 0, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("aucun résultat témoin : effet doit lever ValueError")

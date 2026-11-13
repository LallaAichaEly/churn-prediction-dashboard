"""Étape 2 (corrigé) — Couper dans le temps : apprendre sur le passé, tester sur le mois suivant."""

from __future__ import annotations

import re

import pandas as pd

from .modeles import HISTORIQUE, MOIS


def decouper(lots: pd.DataFrame, mois_test: str, historique: int = HISTORIQUE) -> tuple[pd.DataFrame, pd.DataFrame]:
    # `type(...) is int` refuse True (bool est une sous-classe d'int) et 12.0.
    if type(historique) is not int or historique < 1:
        raise ValueError(f"historique doit être un nombre entier de mois >= 1, pas {historique!r}")
    if not isinstance(mois_test, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", mois_test):
        raise ValueError(f"mois_test doit être de la forme AAAA-MM, pas {mois_test!r}")

    # Des Period, pas des textes : « 12 mois avant 2026-03 » est un calcul sur des mois, que
    # `Period - 12` fait juste (2025-03), même à cheval sur deux années.
    periodes = pd.PeriodIndex(lots[MOIS], freq="M")
    test_p = pd.Period(mois_test, freq="M")
    debut = test_p - historique

    dans_test = periodes == test_p
    # `< test_p` et jamais `!= test_p` : les mois APRÈS le mois de test n'existaient pas encore le jour
    # où l'on aurait entraîné ce modèle. Les utiliser, c'est tricher (fuite temporelle).
    dans_train = (periodes >= debut) & (periodes < test_p)

    if not dans_test.any():
        raise ValueError(f"aucune ligne pour le mois de test {mois_test}")
    if not dans_train.any():
        raise ValueError(f"aucun historique dans les {historique} mois avant {mois_test} : impossible d'entraîner")
    # .copy() : l'appelant peut modifier train ou test sans abîmer `lots`.
    return lots[dans_train].copy(), lots[dans_test].copy()

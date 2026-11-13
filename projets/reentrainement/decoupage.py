"""Étape 2 🟡 — Couper dans le temps : apprendre sur le passé, tester sur le mois suivant.

Vérifier : python scripts/projet.py --projet 5 --etape 2
"""

from __future__ import annotations

import re  # noqa: F401 — vous en aurez besoin

import pandas as pd  # noqa: F401

from .modeles import HISTORIQUE, MOIS  # noqa: F401


def decouper(lots: pd.DataFrame, mois_test: str, historique: int = HISTORIQUE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie `(train, test)` : les lignes des `historique` mois AVANT `mois_test`, et celles de `mois_test`.

    Exemple : `decouper(lots, "2026-01", historique=12)` → train = les mois 2025-01 à 2025-12
    (12 mois), test = le mois 2026-01. Les mois après 2026-01, s'il y en a, ne vont NULLE PART.

    Pourquoi pas `train_test_split`, comme au projet 1 ? Parce que le modèle servira le mois
    PROCHAIN, avec ce qu'on sait AUJOURD'HUI. Un découpage au hasard mélange des clients de
    septembre dans l'entraînement d'un modèle testé sur août : il « connaît l'avenir » et paraît
    meilleur qu'il ne sera. On reproduit donc la vraie vie : apprendre sur le passé, tester sur le
    mois qui suit.

    1. `historique` doit être un entier >= 1 (ni True, ni 1.5, ni 0) ; `mois_test` un texte AAAA-MM
       (« 2026-1 » et « 2026-13 » sont refusés). Sinon, `ValueError`.
    2. Comptez en mois, pas en texte : `pd.Period("2026-02", freq="M") - 3` vaut 2025-11. Le premier
       mois gardé est `Period(mois_test) - historique`, inclus ; le dernier est le mois d'avant
       `mois_test`. Pour comparer toute la colonne : `pd.PeriodIndex(lots[MOIS], freq="M")`.
    3. Les lignes ne sont pas forcément triées par mois : sélectionnez avec des masques, pas par
       position.
    4. S'il y a moins de mois que demandé, gardez ceux qui existent. Mais si le mois de test n'a
       aucune ligne, ou si train serait vide, levez `ValueError` (on n'entraîne pas sur rien).
    5. Gardez toutes les colonnes, et renvoyez des COPIES (`.copy()`) : l'appelant peut modifier
       train ou test sans abîmer `lots`.

    Ce que vous apprenez : la validation temporelle (série 21, défi 132), `pd.Period` et
    `PeriodIndex`, les masques booléens, la fuite de données dans le temps.
    """
    raise NotImplementedError

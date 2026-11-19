"""Étape 1 🟢 — Les mois du calendrier.

Vérifier : python scripts/projet.py --projet 7 --etape 1
"""

from __future__ import annotations

import calendar  # noqa: F401 — vous en aurez besoin
import re  # noqa: F401
from datetime import date

from .modeles import MOIS_FR  # noqa: F401


def lire_mois(texte: str) -> date:
    """« 2026-10 » → `date(2026, 10, 1)` : le 1er du mois écrit AAAA-MM.

    Les espaces autour sont tolérés (`" 2026-10 "`, tapé avec une espace avant ou après). Tout le reste lève
    une `ValueError` avec un message qui dit ce qui est attendu :
    - « 2026-1 » : il faut DEUX chiffres pour le mois. Les rapports s'appellent `rapport_2026-01.html` :
      avec « 2026-1 », `rapport_2026-1.html` se rangerait APRÈS `rapport_2026-10.html` ;
    - « 2026-13 », « 2026-00 » : le mois va de 01 à 12 ;
    - « 10/2026 », « 2026-10-01 », « octobre » : ce n'est pas AAAA-MM.

    Piège : `datetime.strptime(texte, "%Y-%m")` accepte « 2026-1 ». Et `re.match` vérifie le DÉBUT du
    texte seulement : « 2026-10-01 » commence bien par AAAA-MM. Regardez `re.fullmatch`.
    """
    raise NotImplementedError


def fin_du_mois(mois: date) -> date:
    """Le dernier jour du mois de `mois` (qui peut être n'importe quel jour de ce mois).

        fin_du_mois(date(2026, 10, 15)) == date(2026, 10, 31)
        fin_du_mois(date(2028, 2, 1)) == date(2028, 2, 29)     # 2028 est bissextile

    Pièges : `mois + timedelta(days=30)` ; `date(annee, mois + 1, 1) - timedelta(days=1)`, qui plante
    en décembre (il n'y a pas de mois 13). `calendar.monthrange(annee, mois)` connaît la réponse.
    """
    raise NotImplementedError


def decaler(mois: date, n: int) -> date:
    """Le 1er du mois situé `n` mois après `mois` (`n` négatif : avant ; `n` = 0 : le 1er du même mois).

        decaler(date(2026, 10, 15), 0) == date(2026, 10, 1)
        decaler(date(2026, 1, 1), -1) == date(2025, 12, 1)
        decaler(date(2026, 3, 1), 12) == date(2027, 3, 1)

    Le rapport compare chaque mois au mois d'avant, et dessine les 12 derniers mois : cette fonction
    est appelée partout.

    Piège : `timedelta(days=30 * n)`. Un mois ne fait pas 30 jours : 12 fois 30 jours après le 1er mars
    2026, on est le 24 février 2027. Indice : comptez en mois depuis l'an 0 (`annee * 12 + mois - 1`),
    ajoutez `n`, puis revenez à (année, mois) avec `//` et `%`, qui gèrent aussi les nombres négatifs.
    """
    raise NotImplementedError


def nom_du_mois(mois: date) -> str:
    """« octobre 2026 » : le nom du mois, en français et en minuscules, puis l'année.

    Piège : `mois.strftime("%B %Y")` écrit le mois dans la langue de l'ORDINATEUR : « October 2026 » sur
    la plupart des serveurs, alors que c'était « octobre 2026 » sur votre portable. Utilisez `MOIS_FR`
    (dans `modeles.py`), et attention à l'indice : janvier est le mois 1, mais l'élément 0.
    """
    raise NotImplementedError

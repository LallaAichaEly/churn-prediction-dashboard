"""Étape 1 (corrigé) — Les mois du calendrier."""

from __future__ import annotations

import calendar
import re
from datetime import date

from .modeles import MOIS_FR


def lire_mois(texte: str) -> date:
    propre = texte.strip()
    if not re.fullmatch(r"\d{4}-\d{2}", propre):  # fullmatch : « 2026-10-01 » ne passe pas
        raise ValueError(f"mois {texte!r} illisible : attendu AAAA-MM, par exemple 2026-10")
    annee, mois = int(propre[:4]), int(propre[5:])
    if not 1 <= mois <= 12:
        raise ValueError(f"mois {texte!r} impossible : le mois va de 01 à 12")
    return date(annee, mois, 1)


def fin_du_mois(mois: date) -> date:
    return date(mois.year, mois.month, calendar.monthrange(mois.year, mois.month)[1])


def decaler(mois: date, n: int) -> date:
    # On compte en mois depuis l'an 0 : la division entière et le modulo de Python gèrent les n négatifs
    # (-1 // 12 vaut -1 et -1 % 12 vaut 11), donc janvier - 1 donne bien décembre de l'année d'avant.
    rang = mois.year * 12 + (mois.month - 1) + n
    return date(rang // 12, rang % 12 + 1, 1)


def nom_du_mois(mois: date) -> str:
    return f"{MOIS_FR[mois.month - 1]} {mois.year}"

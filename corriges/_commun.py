"""Outils partagés par les corrigés : chemins et chargement des données."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.data import load_or_create  # noqa: E402


def donnees():
    return load_or_create(ROOT / "data" / "customers.csv")


def titre(texte: str) -> None:
    print(f"\n{'=' * 70}\n{texte}\n{'=' * 70}")

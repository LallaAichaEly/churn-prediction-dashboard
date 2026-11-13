"""Étape 1 (corrigé) — Tirer le groupe témoin au hasard… de façon reproductible."""

from __future__ import annotations

import hashlib

from .modeles import GROUPE_OFFRE, GROUPE_TEMOIN, PART_TEMOIN


def tirage(customer_id: str, graine: str) -> float:
    # Le séparateur « : » compte : sans lui, ("ab", "c") et ("a", "bc") donneraient le même texte.
    texte = f"{graine}:{customer_id}"
    empreinte = hashlib.sha256(texte.encode("utf-8")).hexdigest()  # 64 caractères hexadécimaux
    # 16 caractères hexadécimaux = 8 octets = un entier entre 0 et 2**64 − 1. Divisé par 2**64,
    # il donne un nombre entre 0 (inclus) et 1 (exclu), réparti uniformément.
    # Surtout pas hash(texte) : depuis Python 3.3, hash() d'un texte change à chaque lancement
    # (PYTHONHASHSEED), donc un client changerait de groupe d'une exécution à l'autre.
    return int(empreinte[:16], 16) / 2**64


def groupe(customer_id: str, graine: str, part_temoin: float = PART_TEMOIN) -> str:
    # Écrit ainsi, le test refuse aussi nan (toute comparaison avec nan est fausse).
    if not 0 <= part_temoin <= 1:
        raise ValueError(f"part_temoin doit être entre 0 et 1, pas {part_temoin}")
    return GROUPE_TEMOIN if tirage(customer_id, graine) < part_temoin else GROUPE_OFFRE

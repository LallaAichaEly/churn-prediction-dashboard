"""Étape 3 (corrigé) — Écrire les nombres comme un humain."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .modeles import ESPACE_MILLIERS, ESPACE_UNITE, TIRET


def _arrondi(valeur: Decimal, decimales: int) -> Decimal:
    resultat = valeur.quantize(Decimal(1).scaleb(-decimales), rounding=ROUND_HALF_UP)
    return abs(resultat) if resultat == 0 else resultat  # -0,00 n'existe pas pour un lecteur


def _chiffres(arrondi: Decimal, decimales: int) -> str:
    # Le format anglais d'abord (1,234.50), puis on échange : la virgule des milliers devient une
    # espace fine insécable, PUIS le point devient une virgule. Dans l'autre ordre, tout se mélange.
    return f"{arrondi:,.{decimales}f}".replace(",", ESPACE_MILLIERS).replace(".", ",")


def _decimal(x: float) -> Decimal:
    # str(2.675) vaut « 2.675 » : le nombre tel qu'on l'a écrit. Decimal(2.675) vaudrait
    # 2.67499999999999982236431605997495353221893310546875, sa vraie valeur binaire.
    return Decimal(str(x))


def euros(montant: float | None) -> str:
    if montant is None:
        return TIRET
    return f"{_chiffres(_arrondi(_decimal(montant), 2), 2)}{ESPACE_UNITE}€"


def pourcentage(part: float | None, decimales: int = 1) -> str:
    if part is None:
        return TIRET
    return f"{_chiffres(_arrondi(_decimal(part) * 100, decimales), decimales)}{ESPACE_UNITE}%"


def ecart_points(avant: float | None, apres: float | None) -> str:
    if avant is None or apres is None:
        return TIRET
    ecart = _arrondi((_decimal(apres) - _decimal(avant)) * 100, 1)
    signe = "+" if ecart > 0 else ""  # le « - » des négatifs est déjà là ; zéro n'a pas de signe
    return f"{signe}{_chiffres(ecart, 1)}{ESPACE_UNITE}pt"


def nombre(n: int, mot: str) -> str:
    pluriel = mot if n < 2 or mot.endswith(("s", "x", "z")) else mot + "s"
    return f"{_chiffres(Decimal(n), 0)}{ESPACE_UNITE}{pluriel}"

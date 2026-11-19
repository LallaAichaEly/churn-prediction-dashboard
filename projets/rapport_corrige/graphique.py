"""Étape 4 (corrigé) — La courbe des 12 derniers mois, dessinée en SVG."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from decimal import Decimal

from .formats import pourcentage
from .modeles import (ECHELLE_PAR_DEFAUT, HAUTEUR, LARGEUR, MARGE_BAS, MARGE_DROITE, MARGE_GAUCHE, MARGE_HAUT,
                      SVG_NS)


def echelle_haute(maximum: float) -> float:
    if not maximum > 0:  # 0, un négatif… ou nan, pour qui toute comparaison est fausse
        return ECHELLE_PAR_DEFAUT
    k = math.floor(math.log10(maximum))
    # Par prudence, on essaie aussi la puissance d'avant et celle d'après : si log10 arrondit un cheveu trop
    # bas ou trop haut, on ne rate pas le bon candidat. Decimal fabrique 5 × 10⁻⁶ exactement, alors que
    # 5 * 10 ** -6 vaut 4.9999999999999996e-06, plus petit que 5e-06.
    for puissance in (k - 1, k, k + 1):
        for chiffre in ("1", "2", "2.5", "5"):
            candidat = float(Decimal(chiffre).scaleb(puissance))
            if candidat >= maximum:
                return candidat
    raise AssertionError("inatteignable : 10 ** (k + 1) dépasse toujours le maximum")


def _n(x: float) -> str:
    return f"{x:.1f}"


def _graduation(v: float) -> str:
    """Le libellé EXACT d'une graduation : 1,25 % (et non 1,3 %, qui mentirait sur la position de la ligne)."""
    decimales = -(Decimal(str(v)) * 100).normalize().as_tuple().exponent
    return pourcentage(v, max(1, decimales))


def courbe_svg(etiquettes: Sequence[str], valeurs: Sequence[float | None], titre: str,
               largeur: int = LARGEUR, hauteur: int = HAUTEUR) -> str:
    if len(etiquettes) != len(valeurs):
        raise ValueError(f"{len(etiquettes)} étiquette(s) pour {len(valeurs)} valeur(s)")
    haut = echelle_haute(max((v for v in valeurs if v is not None), default=0.0))
    gauche, droite, sommet, bas = MARGE_GAUCHE, largeur - MARGE_DROITE, MARGE_HAUT, hauteur - MARGE_BAS
    n = len(valeurs)

    def x(i: int) -> float:
        return gauche + (droite - gauche) * (i / (n - 1) if n > 1 else 0.5)

    def y(v: float) -> float:
        return bas - (bas - sommet) * v / haut  # en SVG, y grandit vers le BAS : 0 est en bas du rectangle

    e = html.escape
    morceaux = [f'<svg xmlns="{SVG_NS}" viewBox="0 0 {largeur} {hauteur}" width="{largeur}" height="{hauteur}" '
                f'role="img" font-family="system-ui, sans-serif" font-size="11">',
                f"<title>{e(titre)}</title>"]
    for v in (0.0, haut / 2, haut):
        morceaux.append(f'<line x1="{gauche}" y1="{_n(y(v))}" x2="{droite}" y2="{_n(y(v))}" stroke="#d0d4da"/>')
        morceaux.append(f'<text x="{gauche - 8}" y="{_n(y(v) + 4)}" text-anchor="end" fill="#5b6270">'
                        f"{e(_graduation(v))}</text>")
    for i, etiquette in enumerate(etiquettes):
        morceaux.append(f'<text x="{_n(x(i))}" y="{bas + 22}" text-anchor="middle" fill="#5b6270">{e(etiquette)}</text>')

    # Une ligne par suite d'au moins deux mois connus : un mois inconnu (None) COUPE la courbe ; il ne
    # la fait pas plonger à zéro, ce qui inventerait un mois sans aucun départ.
    suite: list[str] = []
    for i, v in enumerate([*valeurs, None]):
        if v is not None:
            suite.append(f"{_n(x(i))},{_n(y(v))}")
            continue
        if len(suite) >= 2:
            morceaux.append(f'<polyline fill="none" stroke="#2a78d6" stroke-width="2" points="{" ".join(suite)}"/>')
        suite = []
    for i, v in enumerate(valeurs):
        if v is not None:
            morceaux.append(f'<circle cx="{_n(x(i))}" cy="{_n(y(v))}" r="3.5" fill="#2a78d6">'
                            f"<title>{e(etiquettes[i])} : {e(pourcentage(v))}</title></circle>")
    morceaux.append("</svg>")
    return "\n".join(morceaux)

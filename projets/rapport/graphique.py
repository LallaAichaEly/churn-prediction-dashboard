"""Étape 4 🔴 — La courbe des 12 derniers mois, dessinée en SVG.

Vérifier : python scripts/projet.py --projet 7 --etape 4
"""

from __future__ import annotations

import html  # noqa: F401 — vous en aurez besoin
import math  # noqa: F401
from collections.abc import Sequence
from decimal import Decimal  # noqa: F401

from .formats import pourcentage  # noqa: F401
from .modeles import (ECHELLE_PAR_DEFAUT, HAUTEUR, LARGEUR, MARGE_BAS, MARGE_DROITE, MARGE_GAUCHE,  # noqa: F401
                      MARGE_HAUT, SVG_NS)


def echelle_haute(maximum: float) -> float:
    """Le haut de l'axe vertical : le plus petit nombre « rond » supérieur ou égal à `maximum`.

    « Rond » veut dire 1, 2, 2,5 ou 5 fois une puissance de 10 (… 0,01 ; 0,02 ; 0,025 ; 0,05 ; 0,1 ;
    0,2 ; 0,25 ; 0,5 ; 1 ; 2 ; 2,5 ; 5 ; 10 …). Avec un maximum de 3,0 % (0.03), l'axe monte à 5 %
    (0.05) : les graduations 0 %, 2,5 % et 5 % se lisent d'un coup d'œil, 0 %, 1,5 % et 3 % beaucoup
    moins, et le point le plus haut ne touche pas le bord du dessin.

        echelle_haute(0.03) == 0.05      echelle_haute(0.05) == 0.05     (déjà rond : on le garde)
        echelle_haute(0.0201) == 0.025   echelle_haute(0.7) == 1.0       echelle_haute(7) == 10.0

    Si `maximum` n'est pas strictement positif (aucune valeur, ou que des zéros) : ECHELLE_PAR_DEFAUT.
    Sans ce cas, une courbe de zéros diviserait par zéro, et `math.log10(0)` lève une ValueError.

    Indice : la puissance de 10 vient de `math.floor(math.log10(maximum))`. Essayez 1, 2, 2,5 et 5 fois
    cette puissance, puis la puissance suivante (et, par prudence, la précédente : si `log10` arrondit
    un cheveu trop bas ou trop haut, vous ne ratez pas le bon candidat). Fabriquez les candidats sans
    erreur d'arrondi : `5 * 10 ** -6` vaut 4.9999999999999996e-06, PLUS PETIT que 5e-06, alors que
    `float(Decimal(5).scaleb(-6))` vaut exactement 5e-06.
    """
    raise NotImplementedError


def courbe_svg(etiquettes: Sequence[str], valeurs: Sequence[float | None], titre: str,
               largeur: int = LARGEUR, hauteur: int = HAUTEUR) -> str:
    """Le dessin SVG d'une courbe : une valeur (un taux entre 0 et 1, ou None) par étiquette (un mois).

    Le SVG est du texte : des balises, comme le HTML, que le navigateur dessine. Pas de bibliothèque
    de graphiques à installer, pas d'image à joindre au courriel : le dessin vit DANS la page.

    Ce que le dessin doit contenir (le correcteur le lit avec un vrai lecteur XML, `xml.etree`) :

    1. Une balise racine `<svg xmlns="{SVG_NS}" viewBox="0 0 {largeur} {hauteur}" ...>` : sans `xmlns`,
       le navigateur n'affiche rien.
    2. Comme premier enfant, `<title>titre</title>` : c'est ce que lit un lecteur d'écran.
    3. L'échelle verticale va de 0 (en bas) à `haut = echelle_haute(<la plus grande valeur connue>)` (en haut).
       Le rectangle de la courbe va de x = MARGE_GAUCHE à x = largeur - MARGE_DROITE, et de y =
       MARGE_HAUT (la valeur `haut`) à y = hauteur - MARGE_BAS (la valeur 0). Donc :

           x(i) = MARGE_GAUCHE + i × (largeur - MARGE_GAUCHE - MARGE_DROITE) / (n - 1)   (n valeurs ;
                  une seule valeur se place au milieu, sans diviser par zéro)
           y(v) = (hauteur - MARGE_BAS) - v / haut × (hauteur - MARGE_BAS - MARGE_HAUT)

       Attention : en SVG, l'axe y va vers le BAS. Une valeur plus grande a un y plus PETIT.
    4. Pour chaque valeur connue, un point `<circle cx="..." cy="..." r="...">` qui contient
       `<title>étiquette : pourcentage(valeur)</title>` (l'infobulle au survol, ex. « oct. 26 : 3,0 % »).
    5. Les points connus CONSÉCUTIFS sont reliés par une `<polyline points="x1,y1 x2,y2 ..." fill="none" .../>`.
       Un mois inconnu (None) COUPE la courbe en deux polylines : le relier à zéro inventerait un mois
       sans aucun départ. Une suite d'un seul point n'a pas de polyline (son cercle suffit).
    6. Pour chaque étiquette, un `<text>` sous l'axe ; et pour 0, haut / 2 et haut, une ligne
       horizontale `<line .../>` et un `<text>` qui affiche `pourcentage(v, d)`, avec `d` le plus petit
       nombre de décimales (au moins 1) qui écrit v EXACTEMENT : « 0,0 % », « 2,5 % », « 5,0 % » quand
       haut vaut 5 %, mais « 1,25 % » au milieu quand haut vaut 2,5 %. « 1,3 % » mentirait sur la position
       de la ligne. Indice : `(Decimal(str(v)) * 100).normalize().as_tuple().exponent` vaut -2 pour 1,25.
    7. Les coordonnées avec UNE décimale au plus (`f"{x:.1f}"`) : 102.36363636363636 alourdit le fichier
       pour rien. Le correcteur tolère 0,1 pixel d'écart.
    8. Tout texte venu de l'extérieur (`titre`, `etiquettes`) passe par `html.escape` : un « & » nu ou un
       « < » rend le SVG illisible (« not well-formed »), et le navigateur n'affiche plus rien.

    `ValueError` si `etiquettes` et `valeurs` n'ont pas la même longueur. Couleurs, épaisseurs et
    polices : à votre goût (`stroke`, `fill`, `font-size`…), le correcteur ne les regarde pas.
    """
    raise NotImplementedError

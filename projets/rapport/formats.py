"""Étape 3 🟡 — Écrire les nombres comme un humain.

Vérifier : python scripts/projet.py --projet 7 --etape 3
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal  # noqa: F401 — vous en aurez besoin

from .modeles import ESPACE_MILLIERS, ESPACE_UNITE, TIRET  # noqa: F401

# Les quatre fonctions suivent les mêmes règles ; écrivez-les UNE fois, dans des fonctions à vous
# (par exemple `_arrondi` et `_chiffres`), au lieu de les recopier quatre fois.
#
# 1. ARRONDIR à la moitié supérieure, comme à l'école (en s'éloignant de zéro pour les négatifs),
#    le nombre tel qu'il s'écrit en décimal :
#
#        Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
#
#    - `round(0.125, 2)` vaut 0.12 : Python arrondit « au pair » (l'arrondi du banquier) ;
#    - `round(2.675, 2)` vaut 2.67 : le float 2.675 vaut en réalité 2.67499999999999982236431605997… ;
#    - `Decimal(2.675)` garde cette valeur binaire ; `Decimal(str(2.675))` vaut exactement 2.675.
#
# 2. Un résultat ARRONDI nul s'écrit sans signe : -0.001 € s'écrit « 0,00 € », jamais « -0,00 € ».
# 3. Une virgule décimale ; les milliers séparés par ESPACE_MILLIERS (une espace fine insécable) ;
#    ESPACE_UNITE (une espace insécable) avant l'unité. Avec une espace ordinaire, le navigateur peut
#    couper « 1 234 » en fin de ligne, ou laisser « € » tout seul au début de la suivante. Indice :
#    f"{valeur:,.2f}" écrit « 1,234.50 » (format anglais) ; reste à échanger les signes, dans le bon ordre.
# 4. None (une valeur qu'on ne connaît pas) s'écrit TIRET, jamais « None » ni « nan ».


def euros(montant: float | None) -> str:
    """Un montant en euros, au centime : 1234.5 → « 1 234,50 € » (« 1\\u202f234,50\\u00a0€ »).

        euros(0.125) == "0,13\\u00a0€"           euros(999.995) == "1\\u202f000,00\\u00a0€"
        euros(-1234.5) == "-1\\u202f234,50\\u00a0€"  euros(None) == "—"
    """
    raise NotImplementedError


def pourcentage(part: float | None, decimales: int = 1) -> str:
    """Une part (entre 0 et 1) en pourcentage : 0.1234 → « 12,3 % » (« 12,3\\u00a0% »).

    Multipliez par 100 APRÈS le passage en Decimal : en float, 0.0295 * 100 vaut 2.9499999999999997, qu'on
    arrondirait à « 2,9 % » au lieu de « 3,0 % ».
    `decimales` chiffres après la virgule ; avec 0, pas de virgule du tout : pourcentage(0.5, 0) == "50\\u00a0%".
    """
    raise NotImplementedError


def ecart_points(avant: float | None, apres: float | None) -> str:
    """L'écart entre deux taux, en POINTS de pourcentage, avec son signe : (0.020, 0.032) → « +1,2 pt ».

    Le taux de churn passe de 2,0 % à 3,2 % : c'est +1,2 point (une différence de pourcentages), et
    non « +60 % » (une variation relative, qui fait peur ou rassure selon ce qu'on veut montrer).

    - l'écart est (apres - avant) × 100, calculé en Decimal à partir des taux tels qu'ils s'écrivent
      (`Decimal(str(x))`, comme partout dans ce fichier), puis arrondi à 1 décimale à la moitié
      supérieure : (0.1, 0.1135) donne +1,4 pt. En float, 0.1135 - 0.1 vaut 0.013499999999999998, et
      Decimal(0.1135) - Decimal(0.1) vaut 0.01349999999999999811… : on écrirait +1,3 pt ;
    - « + » devant un écart positif, « - » devant un écart négatif, rien devant un écart nul APRÈS
      arrondi : (0.05, 0.0499) → « 0,0 pt », ni « +0,0 pt » ni « -0,0 pt » (`f"{x:+.1f}"` écrit les deux) ;
    - ESPACE_UNITE avant « pt », qui ne prend pas de « s » ;
    - TIRET si l'un des deux taux est None (un mois sans abonnés au début : il n'y a rien à comparer).
    """
    raise NotImplementedError


def nombre(n: int, mot: str) -> str:
    """Un nombre suivi de son nom, accordé : « 0 client », « 1 client », « 2 clients », « 1 234 clients ».

    - en français, 0 et 1 sont au singulier, le pluriel commence à 2 ;
    - un mot qui finit déjà par s, x ou z ne change pas : « 3 mois », « 2 prix » ;
    - ESPACE_MILLIERS entre les milliers, ESPACE_UNITE entre le nombre et le mot (pour que « 1 234 »
      et « clients » ne soient jamais séparés par un retour à la ligne).
    """
    raise NotImplementedError

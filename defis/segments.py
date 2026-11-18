"""Outils communs de la série 32 : des clients à regrouper, des générateurs, et les deux résultats nommés.

L'équipe rétention ne peut pas écrire une campagne par client, ni une seule campagne pour tous. Elle veut
quelques GROUPES de clients qui se ressemblent (des segments), chacun avec son message. Les données :

- `clients_exemple()` : douze clients décrits par trois variables, dans leurs unités : l'ancienneté
  (en mois), la facture mensuelle (en euros) et le nombre d'appels au support, plus `churn` (1 = parti) ;
- `VARIABLES` : ces trois variables, dans cet ordre ;
- `points_de(clients, variables)` : la liste des points (un tuple de nombres par client) qu'on donne à
  k-means ;
- `Partition` (défi 196) et `Segment` (défi 199) : les résultats, des tuples nommés ;
- `generer_points(n, k, graine)` et `generer_clients(n, graine)` : de gros volumes reproductibles, pour
  vos propres essais. Les coordonnées de `generer_points` sont des multiples de 1/16 : leurs sommes et
  leurs carrés sont EXACTS en virgule flottante.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> clients = clients_exemple()
    >>> clients[0]
    {'client': 'C01', 'anciennete': 2, 'facture': 85.0, 'appels': 5, 'churn': 1}
    >>> points = points_de(clients)
    >>> points[0], points[4]                     # (ancienneté, facture, appels)
    ((2.0, 85.0, 5.0), (60.0, 30.0, 0.0))
    >>> sum(c["churn"] for c in clients), len(clients)
    (4, 12)
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple

VARIABLES = ("anciennete", "facture", "appels")


class Partition(NamedTuple):
    """Le résultat de k-means (défi 196). Un tuple : (centres, etiquettes, inertie, iterations)."""

    centres: list[tuple[float, ...]]   # un centre par segment : la moyenne de ses points
    etiquettes: list[int]              # pour chaque point, le numéro de son centre (0, 1, …)
    inertie: float                     # somme des distances AU CARRÉ de chaque point à son centre
    iterations: int                    # nombre de recalculs des centres


class Segment(NamedTuple):
    """La fiche d'un segment (défi 199), dans les unités du métier."""

    numero: int                        # l'étiquette du segment
    taille: int                        # nombre de clients
    part: float                        # taille / nombre total de clients (entre 0 et 1)
    taux_churn: float                  # part des clients du segment qui sont partis
    moyennes: dict[str, float]         # {variable: moyenne dans le segment}, en mois, en euros, en appels
    distinctif: tuple[str, float]      # (variable, écart en écarts-types) : ce qui DISTINGUE le segment


#        client  ancienneté facture appels churn
_CLIENTS = [
    ("C01", 2, 85.0, 5, 1),    # récents, beaucoup d'appels : ils partent
    ("C02", 3, 90.0, 4, 1),
    ("C03", 1, 80.0, 6, 1),
    ("C04", 4, 118.0, 6, 0),   # récent, 6 appels… et une grosse facture
    ("C05", 60, 30.0, 0, 0),   # fidèles, petite facture, aucun souci
    ("C06", 48, 35.0, 1, 0),
    ("C07", 66, 25.0, 0, 0),
    ("C08", 54, 40.0, 1, 0),
    ("C09", 24, 120.0, 1, 0),  # grosse facture, ancienneté moyenne
    ("C10", 30, 110.0, 2, 1),
    ("C11", 18, 125.0, 1, 0),
    ("C12", 36, 115.0, 2, 0),
]


def clients_exemple() -> list[dict]:
    """Douze clients : quatre récents qui appellent beaucoup, quatre fidèles, quatre grosses factures."""
    return [{"client": c, "anciennete": a, "facture": f, "appels": n, "churn": y} for c, a, f, n, y in _CLIENTS]


def points_de(clients: list[dict], variables=VARIABLES) -> list[tuple[float, ...]]:
    """Un point par client : le tuple de ses `variables`, en float, dans l'ordre de `variables`.

        >>> points_de([{"anciennete": 3, "facture": 50.5, "appels": 1, "churn": 0}], ("appels", "facture"))
        [(1.0, 50.5)]
    """
    return [tuple(float(client[v]) for v in variables) for client in clients]


def generer_points(n: int = 300, k: int = 3, graine: int = 0, dim: int = 2) -> list[tuple[float, ...]]:
    """`n` points en `dim` dimensions, autour de `k` centres tirés au hasard (reproductible avec `graine`).

    Les coordonnées sont des multiples de 1/16 entre -8 et 8 : sommes et carrés exacts.

        >>> points = generer_points(5, k=2, graine=1)
        >>> len(points), len(points[0]), all(x * 16 == int(x * 16) for p in points for x in p)
        (5, 2, True)
    """
    rng = random.Random(graine)
    centres = [tuple(rng.randint(-5, 5) for _ in range(dim)) for _ in range(k)]
    points = []
    for _ in range(n):
        centre = rng.choice(centres)
        points.append(tuple(max(-8.0, min(8.0, c + rng.randint(-24, 24) / 16)) for c in centre))
    return points


def generer_clients(n: int = 500, graine: int = 0) -> list[dict]:
    """`n` clients au format de `clients_exemple()`, tirés de trois profils, avec un churn plausible.

        >>> clients = generer_clients(3, graine=2)
        >>> sorted(clients[0]), clients[0]["client"]
        (['anciennete', 'appels', 'churn', 'client', 'facture'], 'G0001')
    """
    rng = random.Random(graine)
    clients = []
    for i in range(1, n + 1):
        profil = rng.random()
        if profil < 0.35:                                  # récents, mécontents
            anciennete, facture, appels = rng.randint(1, 12), rng.randint(240, 400) / 4, rng.randint(2, 7)
        elif profil < 0.7:                                 # fidèles
            anciennete, facture, appels = rng.randint(30, 72), rng.randint(80, 200) / 4, rng.randint(0, 2)
        else:                                              # grosses factures
            anciennete, facture, appels = rng.randint(12, 48), rng.randint(400, 520) / 4, rng.randint(0, 3)
        logit = -1.0 - 0.04 * anciennete + 0.015 * (facture - 65) + 0.45 * appels
        churn = int(rng.random() < 1 / (1 + math.exp(-logit)))
        clients.append({"client": f"G{i:04d}", "anciennete": anciennete, "facture": facture, "appels": appels,
                        "churn": churn})
    return clients

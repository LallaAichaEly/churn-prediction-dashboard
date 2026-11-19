"""Outils communs de la série 42 : les relevés mensuels des clients, la matrice de l'an dernier, les marges.

Le directeur marketing veut offrir une remise de 120 € à chaque client « à risque ». La finance a répondu par la
formule de son tableur : « un client rapporte 23,50 € de marge par mois et 3,2 % des clients partent chaque mois,
donc un client vaut 23,50 / 0,032 = 734 € : on peut bien en dépenser 120 pour le garder ». Inès, contrôleuse de
gestion, n'est pas convaincue : un client « à risque » ne rapporte pas la marge moyenne, il part bien plus vite
que la moyenne, il revient parfois à la normale, et un euro touché dans dix ans ne vaut pas un euro aujourd'hui.
Elle décrit chaque client, chaque mois, par un ÉTAT, et compte comment on passe d'un état à l'autre :

- `ETATS` : « Fidèle », « Actif », « À risque » et « Parti ». Un client parti ne revient pas (s'il revient, c'est
  un nouveau client) : « Parti » est un état ABSORBANT ;
- `releves_exemple()` : les relevés mensuels de la base clients, de janvier 2025 à octobre 2026, une ligne
  (client, mois, état) par client et par mois. Ils sortent d'une requête sans ORDER BY : dans le DÉSORDRE. Le
  relevé d'août 2025 a été perdu (la panne de l'entrepôt), quelques relevés manquent çà et là, quelques-uns sont
  chargés deux fois, et un client parti n'a plus de relevé après celui de son départ (défis 254 et 259) ;
- `segments_exemple()` : l'offre de chaque client (« Essentiel », « Confort » ou « Pro ») ; l'offre Pro, toute
  récente, n'a que quelques dizaines de clients (défi 259) ;
- `matrice_exemple()` : la matrice de transition que l'équipe décisionnelle a estimée l'an dernier, pour
  travailler sans attendre le défi 254 (défis 255 à 258) ;
- `MARGES` : la marge mensuelle d'un client selon son état (un client à risque appelle le support et obtient des
  gestes commerciaux) ; `TAUX_ANNUEL` : le coût du capital, pour actualiser ;
- `afficher(matrice)` : la matrice en pourcentages, lisible dans un terminal ;
- `generer_releves(...)` : d'autres relevés, reproductibles, pour vos propres essais.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> releves = releves_exemple()
    >>> len(releves), releves[:2]
    (48434, [('C02024', '2026-01', 'Fidèle'), ('C01860', '2026-01', 'Actif')])
    >>> print(afficher(matrice_exemple()))
                 Fidèle    Actif À risque    Parti
    Fidèle        95.0%     4.0%     0.8%     0.2%
    Actif          3.0%    90.0%     5.0%     2.0%
    À risque       0.0%    25.0%    55.0%    20.0%
    Parti          0.0%     0.0%     0.0%   100.0%
"""

from __future__ import annotations

import functools
import random

ETATS = ("Fidèle", "Actif", "À risque", "Parti")
PARTI = "Parti"
OFFRES = ("Essentiel", "Confort", "Pro")

MARGES = {"Fidèle": 30.0, "Actif": 22.0, "À risque": 9.0, "Parti": 0.0}    # euros de marge par mois
TAUX_ANNUEL = 0.10                                                        # coût du capital : 10 % par an

PREMIER_MOIS, DERNIER_MOIS = "2025-01", "2026-10"
MOIS_PERDU = "2025-08"                                  # le relevé perdu dans la panne de l'entrepôt

_MATRICE_AN_DERNIER = {
    "Fidèle": (0.95, 0.04, 0.008, 0.002),
    "Actif": (0.03, 0.90, 0.05, 0.02),
    "À risque": (0.0, 0.25, 0.55, 0.20),
    "Parti": (0.0, 0.0, 0.0, 1.0),
}

# Les VRAIES probabilités, par offre, qu'utilise le générateur (et que le défi 254 retrouve à peu près).
_VRAIES = {
    "Essentiel": {
        "Fidèle": (0.93, 0.055, 0.012, 0.003),
        "Actif": (0.025, 0.88, 0.07, 0.025),
        "À risque": (0.0, 0.22, 0.53, 0.25),
    },
    "Confort": {
        "Fidèle": (0.965, 0.03, 0.004, 0.001),
        "Actif": (0.04, 0.915, 0.035, 0.01),
        "À risque": (0.0, 0.32, 0.55, 0.13),
    },
    "Pro": {
        "Fidèle": (0.985, 0.015, 0.0, 0.0),
        "Actif": (0.06, 0.935, 0.005, 0.0),
        "À risque": (0.0, 0.45, 0.50, 0.05),
    },
}
_DEPART = {"Essentiel": (0.15, 0.65, 0.20), "Confort": (0.35, 0.55, 0.10), "Pro": (0.40, 0.60, 0.0)}


def matrice_exemple() -> dict[str, dict[str, float]]:
    """La matrice de l'an dernier : `matrice[depuis][vers]` = la part des clients « depuis » qui sont « vers » le
    mois suivant. Chaque ligne fait 100 % ; la ligne « Parti » ne se quitte pas.

        >>> matrice = matrice_exemple()
        >>> matrice["À risque"]["Actif"], matrice["Actif"]["Parti"]
        (0.25, 0.02)
        >>> list(matrice) == list(ETATS) == list(matrice["Fidèle"])
        True
    """
    return {depuis: dict(zip(ETATS, ligne)) for depuis, ligne in _MATRICE_AN_DERNIER.items()}


def afficher(matrice) -> str:
    """La matrice en pourcentages, une ligne par état de départ (pour lire vos résultats dans un terminal).

        >>> print(afficher({"A": {"A": 0.5, "B": 0.5}, "B": {"A": 0.0, "B": 1.0}}))
                         A       B
        A            50.0%   50.0%
        B             0.0%  100.0%
    """
    colonnes = list(next(iter(matrice.values()), {}))
    largeur = max([8] + [len(c) + 1 for c in colonnes])
    lignes = [" " * 10 + "".join(f"{c:>{largeur}}" for c in colonnes)]
    for depuis, ligne in matrice.items():
        lignes.append(f"{depuis:<10}" + "".join(f"{100 * ligne.get(c, 0.0):>{largeur - 1}.1f}%" for c in colonnes))
    return "\n".join(lignes)


def _mois_suivants(debut: str, fin: str) -> list[str]:
    annee, mois = int(debut[:4]), int(debut[5:])
    tous = []
    while f"{annee:04d}-{mois:02d}" <= fin:
        tous.append(f"{annee:04d}-{mois:02d}")
        annee, mois = (annee + 1, 1) if mois == 12 else (annee, mois + 1)
    return tous


def generer_releves(n_clients: int = 500, graine: int = 0, debut: str = PREMIER_MOIS, fin: str = DERNIER_MOIS,
                    manquants: float = 0.02, doublons: float = 0.005,
                    mois_perdu: str | None = MOIS_PERDU) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """(relevés, segments) : `n_clients` clients suivis mois par mois, et l'offre de chacun.

    Chaque client arrive un mois au hasard, dans un état tiré selon son offre, puis change d'état chaque mois selon
    les vraies probabilités de son offre, jusqu'à son départ (son dernier relevé dit « Parti ») ou jusqu'à `fin`.
    Une part `manquants` des relevés est perdue, une part `doublons` est chargée deux fois, le mois `mois_perdu`
    disparaît en entier, et le tout est mélangé.

        >>> releves, segments = generer_releves(3, graine=8, debut="2025-10", fin="2026-02", mois_perdu=None)
        >>> for releve in sorted(releves):
        ...     print(releve)
        ('C00001', '2026-02', 'Fidèle')
        ('C00002', '2025-11', 'Actif')
        ('C00002', '2025-12', 'Actif')
        ('C00002', '2026-01', 'Actif')
        ('C00002', '2026-02', 'Actif')
        ('C00003', '2025-11', 'À risque')
        ('C00003', '2025-12', 'À risque')
        ('C00003', '2026-01', 'À risque')
        ('C00003', '2026-02', 'Parti')
        >>> segments
        {'C00001': 'Essentiel', 'C00002': 'Pro', 'C00003': 'Essentiel'}
    """
    rng = random.Random(graine)
    tous_les_mois = _mois_suivants(debut, fin)
    releves: list[tuple[str, str, str]] = []
    segments: dict[str, str] = {}
    for numero in range(1, n_clients + 1):
        client = f"C{numero:05d}"
        offre = rng.choices(OFFRES, weights=(0.6, 0.39, 0.01))[0]
        segments[client] = offre
        arrivee = min(int(rng.expovariate(1 / 6)), len(tous_les_mois) - 1)
        etat = rng.choices(ETATS[:3], weights=_DEPART[offre])[0]
        for mois in tous_les_mois[arrivee:]:
            if rng.random() >= manquants and mois != mois_perdu:
                releves.append((client, mois, etat))
                if rng.random() < doublons:
                    releves.append((client, mois, etat))
            if etat == PARTI:
                break
            etat = rng.choices(ETATS, weights=_VRAIES[offre][etat])[0]
    rng.shuffle(releves)
    return releves, segments


@functools.lru_cache(maxsize=1)
def _exemple() -> tuple[tuple[tuple[str, str, str], ...], tuple[tuple[str, str], ...]]:
    releves, segments = generer_releves(4000, graine=42)
    return tuple(releves), tuple(segments.items())


def releves_exemple() -> list[tuple[str, str, str]]:
    """Les relevés mensuels de 4 000 clients, de janvier 2025 à octobre 2026, dans le désordre (défis 254 et 259).

        >>> releves = releves_exemple()
        >>> for releve in sorted(r for r in releves if r[0] == "C00007"):     # pas d'août : la panne
        ...     print(releve)
        ('C00007', '2025-07', 'Actif')
        ('C00007', '2025-09', 'Actif')
        ('C00007', '2025-10', 'Actif')
        ('C00007', '2025-11', 'Actif')
        ('C00007', '2025-12', 'Actif')
        ('C00007', '2026-01', 'Actif')
        ('C00007', '2026-02', 'Parti')
        >>> sorted({mois for _, mois, _ in releves})[6:8]             # août 2025 manque pour tout le monde
        ['2025-07', '2025-09']
    """
    return list(_exemple()[0])


def segments_exemple() -> dict[str, str]:
    """L'offre de chacun des 4 000 clients des relevés (défi 259).

        >>> from collections import Counter
        >>> Counter(segments_exemple().values())
        Counter({'Essentiel': 2396, 'Confort': 1574, 'Pro': 30})
    """
    return dict(_exemple()[1])

"""Outils communs de la série 43 : la carte des clients, des antennes, de la fibre et des plaintes.

Dans la vallée, les départs ont doublé depuis la rentrée. L'équipe réseau répond que son tableau de bord est
« tout vert » : aucune antenne ne signale de panne. Yanis, analyste data, n'a qu'un tableur de clients avec leur
adresse géocodée, la liste des antennes et la carte de la fibre. Il veut savoir si les départs se concentrent
quelque part, et où envoyer le technicien. Ce module lui fournit :

- `clients_exemple()` : {client: {"position": (lat, lon), "service": …, "parti": bool}} : 6 000 clients, leur
  position en degrés (latitude d'abord, comme sur un GPS), leur service (« Fibre », « ADSL » ou « Box 4G ») et
  s'ils sont partis depuis septembre ;
- `antennes_exemple()` : {antenne: (lat, lon)} : les 280 antennes 4G de la vallée. Sur huit pylônes, deux
  antennes partagent EXACTEMENT la même position (défi 263) ;
- `zone_fibre_exemple()` : la zone où la fibre est ouverte, au format GeoJSON (RFC 7946), comme l'exporte le
  service SIG : un « MultiPolygon » (la ville et le bourg), des positions [LONGITUDE, latitude] (dans CET ordre :
  celui de GeoJSON, l'inverse d'un GPS), des anneaux FERMÉS (la dernière position répète la première), et un TROU
  dans la ville : le quartier des Tilleuls, pas encore raccordé (défi 261) ;
- `plaintes_exemple()` : la position de chaque appel « je n'ai plus de réseau » reçu par le support depuis
  septembre (défi 264) ;
- `sites_exemple()` et `DEPOT` : les antennes que le technicien doit visiter, et le dépôt d'où il part et où il
  revient le soir (défi 265) ;
- `RAYON_TERRE_KM` (le rayon moyen de la Terre, 6 371 km) et `BASE32_GEOHASH` (l'alphabet des geohash, défi 262) ;
- `generer_points(...)` : d'autres points, reproductibles, pour vos propres essais.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> clients = clients_exemple()
    >>> len(clients), clients["C00001"]
    (6000, {'position': (44.887531, 4.88975), 'service': 'Box 4G', 'parti': False})
    >>> zone = zone_fibre_exemple()
    >>> zone["type"], len(zone["coordinates"]), zone["coordinates"][0][0][:2]     # [lon, lat] : longitude d'abord
    ('MultiPolygon', 2, [[4.9274, 44.933], [4.939274, 44.937406]])
    >>> len(antennes_exemple()), len(plaintes_exemple()), len(sites_exemple())
    (280, 845, 24)
"""

from __future__ import annotations

import bisect
import functools
import math
import random

RAYON_TERRE_KM = 6371.0                     # le rayon MOYEN de la Terre (pas les 6 378 km de l'équateur)
BASE32_GEOHASH = "0123456789bcdefghjkmnpqrstuvwxyz"     # sans a, i, l ni o (défi 262)

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 44.45, 45.15, 4.55, 5.45      # la vallée
VILLE = (44.9330, 4.8920)                   # le centre de la ville
BOURG = (44.7290, 5.0230)                   # le centre du bourg
TILLEULS = (44.9395, 4.8790)                # le quartier pas encore raccordé (le trou de la zone fibre)
DEPOT = (44.9150, 4.9400)                   # le dépôt des techniciens

_KM_PAR_DEGRE = math.pi * RAYON_TERRE_KM / 180


def _decaler(centre: tuple[float, float], nord_km: float, est_km: float) -> tuple[float, float]:
    """Le point à `nord_km` au nord et `est_km` à l'est de `centre` (approximation locale, suffisante ici)."""
    lat, lon = centre
    return lat + nord_km / _KM_PAR_DEGRE, lon + est_km / (_KM_PAR_DEGRE * math.cos(math.radians(lat)))


def generer_points(n: int, graine: int = 0, centre: tuple[float, float] | None = None, ecart_km: float = 1.0,
                   arrondi: int = 6) -> list[tuple[float, float]]:
    """`n` points (lat, lon) : autour de `centre` (écart type `ecart_km` dans chaque direction), ou, sans `centre`,
    uniformément dans la vallée. Coordonnées arrondies à `arrondi` décimales (6 : à 11 cm près, comme un GPS).

        >>> generer_points(2, graine=1)
        [(44.544055, 5.31269), (44.984642, 4.779562)]
        >>> generer_points(1, graine=1, centre=VILLE, ecart_km=0.5)
        [(44.938792, 4.901207)]
    """
    alea = random.Random(graine)
    points = []
    for _ in range(n):
        if centre is None:
            lat, lon = alea.uniform(LAT_MIN, LAT_MAX), alea.uniform(LON_MIN, LON_MAX)
        else:
            lat, lon = _decaler(centre, alea.gauss(0, ecart_km), alea.gauss(0, ecart_km))
        points.append((round(lat, arrondi), round(lon, arrondi)))
    return points


def _anneau(centre: tuple[float, float], rayon_km: float, n_sommets: int, alea: random.Random,
            creux: float = 0.65) -> list[list[float]]:
    """Un anneau GeoJSON en étoile autour de `centre` : positions [lon, lat], fermé, dans le sens trigonométrique."""
    positions = []
    for k in range(n_sommets):
        angle = 2 * math.pi * k / n_sommets
        r = rayon_km * alea.uniform(creux, 1.0)
        lat, lon = _decaler(centre, r * math.sin(angle), r * math.cos(angle))
        positions.append([round(lon, 6), round(lat, 6)])
    return positions + [list(positions[0])]


def _dans_anneau(lat: float, lon: float, anneau: list[list[float]]) -> bool:
    dedans = False
    for (x1, y1), (x2, y2) in zip(anneau, anneau[1:]):
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            dedans = not dedans
    return dedans


def _haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 2 * RAYON_TERRE_KM * math.asin(math.sqrt(min(1.0, h)))


@functools.lru_cache(maxsize=1)
def _monde() -> dict:
    """Tout l'exemple, tiré une fois (graine fixe) : les objets publics en sont des COPIES."""
    alea = random.Random(43)

    # ------------------------------------------------------------------ la fibre : la ville (avec un trou), le bourg
    ville = [_anneau(VILLE, 4.2, 48, alea), _anneau(TILLEULS, 0.7, 16, alea, creux=0.8)]
    bourg = [_anneau(BOURG, 2.0, 32, alea)]
    zone = {"type": "MultiPolygon", "coordinates": [ville, bourg]}

    def fibre_ouverte(lat: float, lon: float) -> bool:
        return any(_dans_anneau(lat, lon, poly[0]) and not any(_dans_anneau(lat, lon, t) for t in poly[1:])
                   for poly in (ville, bourg))

    # ------------------------------------------------------------------ les antennes (et quelques pylônes partagés)
    antennes: dict[str, tuple[float, float]] = {}
    positions = generer_points(240, graine=431) + generer_points(22, graine=432, centre=VILLE, ecart_km=3.0) \
        + generer_points(10, graine=433, centre=BOURG, ecart_km=2.0)
    alea.shuffle(positions)
    for k, position in enumerate(positions, start=1):
        antennes[f"A{k:04d}"] = position
    partages = alea.sample(sorted(antennes), 8)
    for k, original in enumerate(partages, start=len(positions) + 1):      # même pylône, un autre identifiant
        antennes[f"A{k:04d}"] = antennes[original]
    # Trois antennes rurales, loin les unes des autres, ne servent plus personne depuis septembre (la sonde de
    # supervision ne vérifie que l'alimentation électrique : le tableau de bord reste vert).
    rurales = [a for a, p in antennes.items() if _haversine(p, VILLE) > 9 and _haversine(p, BOURG) > 7
               and LAT_MIN + 0.05 < p[0] < LAT_MAX - 0.05 and LON_MIN + 0.05 < p[1] < LON_MAX - 0.05
               and list(antennes.values()).count(p) == 1]
    en_panne: list[str] = []
    for antenne in alea.sample(rurales, len(rurales)):
        if all(_haversine(antennes[antenne], antennes[autre]) > 18 for autre in en_panne):
            en_panne.append(antenne)
        if len(en_panne) == 3:
            break
    n_antennes = len(positions)
    rangees = sorted((p[0], p[1], a) for a, p in antennes.items())
    latitudes = [r[0] for r in rangees]

    def antenne_de(position: tuple[float, float]) -> str:
        """L'antenne la plus proche (la recherche du défi 263 : ne lisez pas ceci avant de l'avoir écrite)."""
        haut = bisect.bisect_left(latitudes, position[0])
        bas, meilleur = haut - 1, (math.inf, "")
        while bas >= 0 or haut < len(rangees):
            if haut >= len(rangees) or (bas >= 0 and position[0] - latitudes[bas] <= latitudes[haut] - position[0]):
                k, bas = bas, bas - 1
            else:
                k, haut = haut, haut + 1
            if math.radians(abs(latitudes[k] - position[0])) * RAYON_TERRE_KM > meilleur[0] + 1e-9:
                break
            meilleur = min(meilleur, (_haversine(position, rangees[k][:2]), rangees[k][2]))
        return meilleur[1]

    # ------------------------------------------------------------------ les clients
    clients: dict[str, dict] = {}
    positions = generer_points(1900, graine=434, centre=VILLE, ecart_km=2.6) \
        + generer_points(700, graine=435, centre=BOURG, ecart_km=1.6) + generer_points(3400, graine=436)
    alea.shuffle(positions)
    for k, (lat, lon) in enumerate(positions, start=1):
        if fibre_ouverte(lat, lon) and alea.random() < 0.85:
            service = "Fibre"
        else:
            service = "ADSL" if alea.random() < 0.45 else "Box 4G"
        if service == "Fibre":
            risque = 0.05
        elif antenne_de((lat, lon)) in en_panne:
            risque = 0.48 if service == "Box 4G" else 0.16
        else:
            risque = 0.08
        clients[f"C{k:05d}"] = {"position": (lat, lon), "service": service, "parti": alea.random() < risque}

    # ------------------------------------------------------------------ les plaintes « plus de réseau »
    plaintes = []
    for antenne, n in zip(en_panne, (130, 95, 70)):
        plaintes += generer_points(n, graine=437 + n, centre=antennes[antenne], ecart_km=1.4)
    plaintes += generer_points(450, graine=438) + generer_points(50, graine=439, centre=VILLE, ecart_km=3.5) \
        + generer_points(50, graine=440, centre=BOURG, ecart_km=3.0)
    alea.shuffle(plaintes)

    # ------------------------------------------------------------------ la tournée : les pannes, puis l'entretien
    entretien = alea.sample(sorted(a for a in antennes if a not in en_panne and a <= f"A{n_antennes:04d}"), 21)
    sites = {a: antennes[a] for a in sorted(en_panne + entretien)}
    return {"zone": zone, "antennes": antennes, "clients": clients, "plaintes": plaintes, "sites": sites,
            "en_panne": tuple(sorted(en_panne))}


def clients_exemple() -> dict[str, dict]:
    """Les 6 000 clients de la vallée : {client: {"position": (lat, lon), "service": …, "parti": bool}}.

        >>> from collections import Counter
        >>> clients = clients_exemple()
        >>> Counter(c["service"] for c in clients.values())
        Counter({'Box 4G': 2691, 'ADSL': 2113, 'Fibre': 1196})
        >>> round(sum(c["parti"] for c in clients.values()) / len(clients), 3)       # le taux de départ
        0.075
    """
    return {client: dict(fiche) for client, fiche in _monde()["clients"].items()}


def antennes_exemple() -> dict[str, tuple[float, float]]:
    """Les 280 antennes : {antenne: (lat, lon)}. Huit pylônes portent deux antennes (deux identifiants, une
    position) : à distance égale, le défi 263 garde le plus petit identifiant.

        >>> antennes = antennes_exemple()
        >>> antennes["A0001"]
        (45.082572, 4.915994)
        >>> len(set(antennes.values()))                   # 280 antennes, 272 positions
        272
        >>> antennes["A0236"] == antennes["A0277"]        # deux antennes sur le même pylône
        True
    """
    return dict(_monde()["antennes"])


def zone_fibre_exemple() -> dict:
    """La zone fibre, en GeoJSON : {"type": "MultiPolygon", "coordinates": [ville, bourg]}.

    La ville est [anneau extérieur, trou des Tilleuls] ; le bourg, [anneau extérieur]. Chaque anneau est une liste
    de positions [LONGITUDE, latitude], fermée (la dernière position répète la première).

        >>> zone = zone_fibre_exemple()
        >>> ville, bourg = zone["coordinates"]
        >>> len(ville), len(bourg), [len(anneau) for anneau in ville + bourg]     # 48 + 1, 16 + 1, 32 + 1 positions
        (2, 1, [49, 17, 33])
        >>> ville[0][0] == ville[0][-1]                   # un anneau GeoJSON est FERMÉ
        True
    """
    zone = _monde()["zone"]
    return {"type": zone["type"],
            "coordinates": [[[list(p) for p in anneau] for anneau in poly] for poly in zone["coordinates"]]}


def plaintes_exemple() -> list[tuple[float, float]]:
    """La position (lat, lon) de chacune des 845 plaintes « plus de réseau » reçues depuis septembre (défi 264).

        >>> plaintes = plaintes_exemple()
        >>> len(plaintes), plaintes[0]
        (845, (44.675355, 5.072564))
    """
    return list(_monde()["plaintes"])


def sites_exemple() -> dict[str, tuple[float, float]]:
    """Les 24 antennes que le technicien doit visiter : {antenne: (lat, lon)}, rangées par identifiant (défi 265).

        >>> sites = sites_exemple()
        >>> list(sites)[:4]
        ['A0003', 'A0010', 'A0026', 'A0036']
    """
    return dict(_monde()["sites"])

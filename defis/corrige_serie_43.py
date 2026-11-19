"""Série 43 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_43.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours des degrés
passés à `math.sin` sans `math.radians`, une position GeoJSON lue (latitude, longitude) alors qu'elle est
[longitude, latitude], un sommet compté deux fois par le rayon du défi 261, ou une recherche qui s'arrête à la
première antenne trouvée au lieu de la plus proche.

Fil rouge de la série : sur une carte, « près » ne se mesure pas avec Pythagore sur des degrés, et « chercher le
plus proche » ne veut pas dire « comparer tout le monde à tout le monde » :

1. la DISTANCE entre deux points du globe est celle du grand cercle (haversine) : des radians, sin² du demi-écart,
   et `min(1.0, h)`, car un arrondi pousse h au-dessus de 1 entre deux points opposés ;
2. être DANS une zone, c'est croiser son bord un nombre IMPAIR de fois en partant vers l'est ; la règle « un bout
   strictement au-dessus, l'autre non » compte un sommet une seule fois et ignore les bords horizontaux ;
3. un GEOHASH découpe le monde en deux, encore et encore (longitude d'abord, `>=` va à droite) : des points
   proches partagent le début de leur code, et un `GROUP BY` sur ce début fait une carte ;
4. la plus PROCHE antenne se trouve sans tout comparer : rangées par latitude, on s'éloigne de part et d'autre,
   et l'on s'arrête quand l'écart de latitude SEUL dépasse déjà la meilleure distance (il la minore) ;
5. DBSCAN regroupe les plaintes : un point « central » a au moins `minimum` voisins (lui compris) à `rayon_km` ;
   un groupe s'étend de central en central, avec une file et non une récursion ; un point de bord n'étend rien ;
6. la TOURNÉE part du plus proche voisin, puis 2-opt décroise les chemins : retourner un morceau de tournée
   remplace deux arêtes par deux autres, et le gain se calcule avec ces QUATRE distances seulement.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import bisect
import math
from collections import deque

from defis.carte import (  # noqa: F401
    BASE32_GEOHASH,
    DEPOT,
    RAYON_TERRE_KM,
    antennes_exemple,
    clients_exemple,
    generer_points,
    plaintes_exemple,
    sites_exemple,
    zone_fibre_exemple,
)


def _nombre(valeur, nom: str) -> float:
    """Un vrai nombre (int ou float, pas un bool ni un texte), fini : sinon TypeError ou ValueError."""
    if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
        raise TypeError(f"{nom} doit être un nombre, pas {type(valeur).__name__}")
    if not math.isfinite(valeur):
        raise ValueError(f"{nom} doit être un nombre fini, pas {valeur!r}")
    return float(valeur)


def _point(point, nom: str = "point") -> tuple[float, float]:
    """(latitude, longitude) en degrés, vérifiés : TypeError si ce n'est pas un couple de nombres, ValueError hors
    du globe (une latitude au-delà de 90°, c'est souvent une longitude écrite à sa place)."""
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise TypeError(f"{nom} doit être un couple (latitude, longitude), pas {point!r}")
    lat, lon = _nombre(point[0], f"la latitude de {nom}"), _nombre(point[1], f"la longitude de {nom}")
    if not -90 <= lat <= 90:
        raise ValueError(f"latitude hors du globe pour {nom} : {lat!r} (entre -90 et 90)")
    if not -180 <= lon <= 180:
        raise ValueError(f"longitude hors du globe pour {nom} : {lon!r} (entre -180 et 180)")
    return lat, lon


# ============================================================================ Défi 260


def distance_km(a, b) -> float:
    """La distance du grand cercle (formule de haversine), en km.

    Complexité : O(1). Les pièges sont tous numériques :
    - `math.sin` attend des RADIANS : sin(45) n'est pas sin(45°) ;
    - Pythagore sur des degrés compte 1° de longitude pour 111 km, alors qu'à 45° de latitude il en vaut 79 ; et
      de part et d'autre de l'antiméridien (179,5° et -179,5°), l'écart de longitude n'est pas 359° mais 1° :
      sin²(Δλ / 2) le sait tout seul ;
    - la « loi des cosinus » acos(sin φ1 sin φ2 + cos φ1 cos φ2 cos Δλ) est juste sur le papier, mais pour deux
      points à un mètre l'un de l'autre, l'argument vaut 0,99999999999998 : acos perd presque tous les chiffres.
      sin² du DEMI-angle (le « haversine ») reste précis ;
    - entre deux points opposés, h vaut 1 au papier, et 1.0000000000000002 en float : 1 - h est négatif, et la
      variante 2 × R × atan2(√h, √(1 − h)) lève « math domain error » (asin(√h) n'y échappe que parce que √h
      s'arrondit à 1.0). `min(1.0, h)` protège toutes les écritures ;
    - 2 × R × asin(√h) : sans le 2, on trouve la moitié.
    """
    lat1, lon1 = _point(a, "a")
    lat2, lon2 = _point(b, "b")
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin((phi2 - phi1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * RAYON_TERRE_KM * math.asin(math.sqrt(min(1.0, h)))


# ============================================================================ Défi 261


def _anneau(anneau) -> list[tuple[float, float]]:
    """Les positions (lon, lat) d'un anneau GeoJSON, vérifié : au moins 4 positions, la dernière = la première."""
    if not isinstance(anneau, (list, tuple)) or len(anneau) < 4:
        raise ValueError(f"un anneau GeoJSON a au moins 4 positions (3 sommets + la fermeture), pas {anneau!r}")
    positions = []
    for position in anneau:
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise ValueError(f"position GeoJSON illisible : {position!r} (attendu [longitude, latitude])")
        # [lon, lat] ou [lon, lat, altitude] : on garde les DEUX premiers, dans l'ordre de GeoJSON.
        positions.append((_nombre(position[0], "une longitude"), _nombre(position[1], "une latitude")))
    if positions[0] != positions[-1]:
        raise ValueError("anneau GeoJSON non fermé : sa dernière position doit répéter la première")
    return positions


def _dans_anneau(lon: float, lat: float, anneau: list[tuple[float, float]]) -> bool:
    """Lancer de rayon vers l'est : dedans si l'on croise le bord un nombre impair de fois.

    `(y1 > lat) != (y2 > lat)` : l'arête a UN bout strictement au-dessus du rayon et l'autre non. Un sommet posé
    pile sur le rayon appartient alors à une seule des deux arêtes qui s'y rejoignent (pas de double compte), et une
    arête horizontale n'est jamais retenue (pas de division par zéro).
    """
    dedans = False
    for (x1, y1), (x2, y2) in zip(anneau, anneau[1:]):
        if (y1 > lat) != (y2 > lat):
            if lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):    # le croisement est à l'EST du point
                dedans = not dedans
    return dedans


def dans_zone(point, zone) -> bool:
    """Le point (lat, lon) est-il dans le Polygon ou le MultiPolygon GeoJSON `zone` ?

    Complexité : O(nombre de sommets). Toute la zone est vérifiée avant de répondre (une zone abîmée lève une
    erreur, même si le point est dans son premier polygone) : une erreur qui dépend du point testé passerait les
    essais et casserait en production.
    """
    lat, lon = _point(point)
    if not isinstance(zone, dict):
        raise TypeError(f"la zone doit être une géométrie GeoJSON (un dict), pas {type(zone).__name__}")
    genre, coordonnees = zone.get("type"), zone.get("coordinates")
    if genre == "Polygon":
        polygones = [coordonnees]
    elif genre == "MultiPolygon":
        polygones = coordonnees
    else:
        raise ValueError(f"géométrie {genre!r} : attendu « Polygon » ou « MultiPolygon »")
    if not isinstance(polygones, (list, tuple)):
        raise ValueError("les coordonnées d'une zone sont une liste de polygones")
    lus = []
    for polygone in polygones:
        if not isinstance(polygone, (list, tuple)) or not polygone:
            raise ValueError(f"polygone GeoJSON illisible : {polygone!r}")
        lus.append([_anneau(anneau) for anneau in polygone])
    # Dans un polygone : dans son anneau extérieur, et dans AUCUN de ses trous.
    return any(_dans_anneau(lon, lat, anneaux[0]) and not any(_dans_anneau(lon, lat, trou) for trou in anneaux[1:])
               for anneaux in lus)


# ============================================================================ Défi 262


def geohash(point, precision: int = 7) -> str:
    """Le geohash de `point` sur `precision` caractères.

    Complexité : O(precision). Cinq bits par caractère, pris tour à tour sur la longitude (d'ABORD) et la
    latitude : chaque bit dit si le point est dans la moitié haute (`>=` le milieu : 1) ou basse (0) de
    l'intervalle courant, qui est alors coupé en deux. Les bits forment des nombres de 0 à 31, bit de poids FORT
    d'abord, lus dans `BASE32_GEOHASH` (pas l'alphabet de `base64.b32encode`).
    """
    lat, lon = _point(point)
    if isinstance(precision, bool) or not isinstance(precision, int):
        raise TypeError(f"precision doit être un int, pas {type(precision).__name__}")
    if not 1 <= precision <= 12:
        raise ValueError(f"precision doit être entre 1 et 12, pas {precision}")
    intervalles = {"lon": [-180.0, 180.0], "lat": [-90.0, 90.0]}
    valeurs = {"lon": lon, "lat": lat}
    texte, valeur, bits, axe = [], 0, 0, "lon"
    while len(texte) < precision:
        bas, haut = intervalles[axe]
        milieu = (bas + haut) / 2
        if valeurs[axe] >= milieu:
            valeur = 2 * valeur + 1
            intervalles[axe][0] = milieu
        else:
            valeur = 2 * valeur
            intervalles[axe][1] = milieu
        axe = "lat" if axe == "lon" else "lon"
        bits += 1
        if bits == 5:
            texte.append(BASE32_GEOHASH[valeur])
            valeur, bits = 0, 0
    return "".join(texte)


# ============================================================================ Défi 263


def plus_proches(clients: dict, antennes: dict) -> dict:
    """{client: (antenne la plus proche, distance en km)}, sans comparer chaque client à chaque antenne.

    Complexité : O(m log m) pour ranger les m antennes par latitude, puis, pour chaque client, O(log m) pour se
    placer et quelques antennes de chaque côté. La distance du grand cercle est au moins R × |écart de latitude|
    (en radians) : dès que l'écart de latitude SEUL dépasse la meilleure distance trouvée, aucune antenne plus loin
    dans cette direction ne peut faire mieux. On avance toujours du côté le plus proche en latitude ; on ne s'arrête
    PAS à la première antenne trouvée (une antenne presque à la même latitude peut être à 30 km à l'est).
    """
    if not isinstance(clients, dict) or not isinstance(antennes, dict):
        raise TypeError("clients et antennes sont des dict {identifiant: (lat, lon)}")
    if not antennes:
        raise ValueError("aucune antenne : pas de plus proche antenne")
    rangees = sorted((*_point(position, f"l'antenne {ident!r}"), ident) for ident, position in antennes.items())
    latitudes = [lat for lat, _, _ in rangees]
    marge = 1e-9                                  # un arrondi ne doit pas faire sauter une antenne à égalité
    resultat = {}
    for client, position in clients.items():
        lat, lon = _point(position, f"le client {client!r}")
        haut = bisect.bisect_left(latitudes, lat)
        bas = haut - 1
        meilleur = None                           # (distance, identifiant) : le tuple trie aussi les égalités
        while bas >= 0 or haut < len(rangees):
            ecart_bas = lat - latitudes[bas] if bas >= 0 else math.inf
            ecart_haut = latitudes[haut] - lat if haut < len(rangees) else math.inf
            if ecart_bas <= ecart_haut:
                k, ecart, bas = bas, ecart_bas, bas - 1
            else:
                k, ecart, haut = haut, ecart_haut, haut + 1
            if meilleur is not None and RAYON_TERRE_KM * math.radians(ecart) > meilleur[0] + marge:
                break                             # le côté le plus proche est déjà trop loin : l'autre aussi
            candidat = (distance_km((lat, lon), rangees[k][:2]), rangees[k][2])
            if meilleur is None or candidat < meilleur:
                meilleur = candidat
        resultat[client] = (meilleur[1], meilleur[0])
    return resultat


# ============================================================================ Défi 264


def regrouper_pannes(plaintes, rayon_km: float, minimum: int) -> list[int]:
    """DBSCAN : l'étiquette du groupe de chaque plainte (0, 1, 2… dans l'ordre de création), -1 pour le bruit.

    Complexité : O(n log n + nombre de paires à moins de `rayon_km` en latitude). Les voisins de chaque point se
    cherchent dans une bande de latitude (bisect), comme au défi 263, au lieu de comparer les n² paires. Ensuite :
    - un point est CENTRAL s'il a au moins `minimum` voisins à `rayon_km` ou moins, LUI COMPRIS (comme
      `min_samples` de scikit-learn) ;
    - on parcourt les points dans l'ordre ; chaque point central sans groupe ouvre un nouveau groupe, qui s'étend
      avec une FILE (pas de récursion : 3 000 plaintes en chaîne feraient 3 000 appels imbriqués) ;
    - un voisin sans groupe rejoint le groupe ; s'il est central, il entre dans la file ; sinon c'est un point de
      BORD : il est pris, mais n'étend rien ;
    - un point d'abord jugé « bruit » peut être pris plus tard par un groupe (il n'est pas central, mais il est près
      d'un central) : le bruit n'est pas définitif, c'est seulement « pas encore de groupe » (-1) ;
    - un point de bord pris par un groupe n'est jamais repris par un autre.
    """
    rayon = _nombre(rayon_km, "rayon_km")
    if rayon <= 0:
        raise ValueError(f"rayon_km doit être > 0, pas {rayon_km!r}")
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise TypeError(f"minimum doit être un int, pas {type(minimum).__name__}")
    if minimum < 1:
        raise ValueError(f"minimum doit être >= 1, pas {minimum}")
    points = [_point(p, f"la plainte {k}") for k, p in enumerate(plaintes)]
    n = len(points)

    ordre = sorted(range(n), key=lambda k: points[k][0])
    latitudes = [points[k][0] for k in ordre]
    bande = math.degrees(rayon / RAYON_TERRE_KM) + 1e-9     # plus loin en latitude, on est forcément trop loin
    voisins = []
    for lat, lon in points:
        debut = bisect.bisect_left(latitudes, lat - bande)
        fin = bisect.bisect_right(latitudes, lat + bande)
        voisins.append([ordre[k] for k in range(debut, fin) if distance_km((lat, lon), points[ordre[k]]) <= rayon])
    central = [len(v) >= minimum for v in voisins]           # le point lui-même est dans sa liste (distance 0)

    etiquettes = [-1] * n
    groupe = 0
    for depart in range(n):
        if etiquettes[depart] != -1 or not central[depart]:
            continue
        etiquettes[depart] = groupe
        file = deque([depart])
        while file:
            for voisin in voisins[file.popleft()]:
                if etiquettes[voisin] == -1:
                    etiquettes[voisin] = groupe
                    if central[voisin]:
                        file.append(voisin)
        groupe += 1
    return etiquettes


# ============================================================================ Défi 265


def tournee(depart, sites: dict, tolerance: float = 1e-9) -> tuple[list, float]:
    """(ordre de visite, longueur en km) : plus proche voisin, puis 2-opt jusqu'à ce qu'aucun échange ne gagne.

    Complexité : O(n²) distances calculées UNE fois (une table), puis O(n²) par passe de 2-opt, chaque essai en
    O(1). Retourner la tournée entre les positions i et j remplace les arêtes (i - 1, i) et (j, j + 1) par
    (i - 1, j) et (i, j + 1) : le reste ne change pas de longueur (parcouru à l'envers, une route a la même
    longueur), donc le gain se lit sur ces QUATRE distances. Recalculer toute la tournée pour chaque essai ferait
    O(n³) par passe. `tolerance` : un gain de 1e-16 km n'est qu'un arrondi, et sans tolérance deux échanges
    pourraient se défaire l'un l'autre sans fin.
    """
    base = _point(depart, "depart")
    if not isinstance(sites, dict):
        raise TypeError("sites est un dict {identifiant: (lat, lon)}")
    noms = list(sites)
    if not noms:
        return [], 0.0
    points = [base] + [_point(sites[nom], f"le site {nom!r}") for nom in noms]
    n = len(points)
    table = [[distance_km(a, b) for b in points] for a in points]

    # Le plus proche voisin : à distance égale, le plus petit identifiant.
    route, restants = [0], set(range(1, n))
    while restants:
        dernier = route[-1]
        suivant = min(restants, key=lambda k: (table[dernier][k], noms[k - 1]))
        route.append(suivant)
        restants.remove(suivant)

    # 2-opt : la position 0 (le dépôt) ne bouge pas ; la tournée revient au dépôt après route[n - 1].
    ameliore = True
    while ameliore:
        ameliore = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b, c, d = route[i - 1], route[i], route[j], route[(j + 1) % n]
                if table[a][b] + table[c][d] - table[a][c] - table[b][d] > tolerance:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    ameliore = True
    longueur = sum(table[route[k]][route[(k + 1) % n]] for k in range(n))     # retour au dépôt compris
    return [noms[k - 1] for k in route[1:]], longueur


# ============================================================================ Tests (à la pytest)


def test_distance():
    assert distance_km((44.93, 4.89), (44.93, 4.89)) == 0.0
    assert math.isclose(distance_km((0, 0), (0, 1)), math.pi * RAYON_TERRE_KM / 180)       # 1° à l'équateur
    assert math.isclose(distance_km((0, 0), (0, 180)), math.pi * RAYON_TERRE_KM)          # la moitié du tour


def test_dans_zone_trou():
    carre = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    trou = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]
    zone = {"type": "Polygon", "coordinates": [carre, trou]}
    assert dans_zone((2, 2), zone) and not dans_zone((5, 5), zone) and not dans_zone((5, 11), zone)


def test_geohash_reference():
    assert geohash((57.64911, 10.40744), 11) == "u4pruydqqvj"
    assert geohash((0, 0), 5) == "s0000"


def test_plus_proche_egalite():
    antennes = {"B": (45.0, 5.01), "A": (45.0, 5.01), "C": (45.2, 5.0)}
    assert plus_proches({"x": (45.0, 5.0)}, antennes)["x"][0] == "A"


def test_dbscan_bord():
    plaintes = [(45.0, 5.0), (45.0, 5.001), (45.0, 5.002), (45.0, 5.012)]
    assert regrouper_pannes(plaintes, 0.1, 2) == [0, 0, 0, -1]


def test_tournee_deux_sites():
    ordre, longueur = tournee((45.0, 5.0), {"A": (45.0, 5.1)})
    assert ordre == ["A"] and math.isclose(longueur, 2 * distance_km((45.0, 5.0), (45.0, 5.1)))

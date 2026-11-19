"""Série 43 — Où partent les clients ? Distance, zone, geohash, plus proche antenne, foyers de pannes, tournée.

🗺️ VOUS VOULEZ DÉVELOPPER ? UN JOUR, VOS DONNÉES AURONT UNE ADRESSE. ET « PRÈS DE » NE SE CALCULE NI AVEC
PYTHAGORE SUR DES DEGRÉS, NI EN COMPARANT TOUT LE MONDE À TOUT LE MONDE. ICI, VOUS ÉCRIVEZ CE QUE FONT POSTGIS,
ELASTICSEARCH, REDIS (GEOADD), SCIKIT-LEARN (DBSCAN) ET LES LOGICIELS DE TOURNÉES DES TECHNICIENS.

Dans la vallée, les départs ont doublé depuis la rentrée. L'équipe réseau répond que son tableau de bord est
« tout vert » : aucune antenne ne signale de panne. Yanis, analyste data, a un tableur de clients avec leur
adresse géocodée (latitude, longitude), la liste des 280 antennes, la carte de la fibre exportée par le service
SIG, et la position de chaque appel « je n'ai plus de réseau » reçu par le support. Il écrit les six outils de
cette série :

    260 distance_km        la distance entre deux points du globe, en km (le grand cercle, pas Pythagore)
    261 dans_zone          ce client est-il dans la zone fibre ? (un polygone GeoJSON, avec un trou)
    262 geohash            le code de la case où tombe un point : « spgrx7f » (une carte = un GROUP BY)
    263 plus_proches       l'antenne la plus proche de chacun des 6 000 clients, sans 1,7 million de distances
    264 regrouper_pannes   les foyers de plaintes : DBSCAN, sans récursion ni n² comparaisons
    265 tournee            la tournée du technicien : plus proche voisin, puis 2-opt

Prérequis : la série 00, un peu de trigonométrie (rappelée ici) et les `dict`. La dichotomie (`bisect`, défi
112), le parcours en largeur avec une file (défi 61) et les tas de la série 24 l'éclairent ; la série 32
(k-means) est la cousine du défi 264.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un POINT est un couple (latitude, longitude) en DEGRÉS, comme sur un GPS : la latitude d'abord (de -90 au pôle
  Sud à 90 au pôle Nord), la longitude ensuite (de -180 à 180, 0 à Greenwich). La vallée est vers (44,9 ; 4,9) ;
- 1° de latitude vaut toujours 111,2 km ; 1° de longitude vaut 111,2 km × cos(latitude) : 78,7 km dans la
  vallée, 0 au pôle. Pythagore sur des degrés se trompe donc dès qu'on va vers l'est ;
- la distance « à vol d'oiseau » suit le grand cercle de la Terre (rayon R = 6 371 km). La formule de HAVERSINE :
  h = sin²(Δφ / 2) + cos φ1 × cos φ2 × sin²(Δλ / 2), puis d = 2 × R × asin(√h), avec φ les latitudes et Δλ
  l'écart de longitude, en RADIANS (`math.radians`). C'est celle de PostGIS (`ST_DistanceSphere`) ;
- le format GeoJSON (RFC 7946), celui de tous les SIG et de toutes les API de cartes, écrit les positions
  [LONGITUDE, latitude] : l'ordre INVERSE d'un GPS. C'est la première source de bugs en géographie ;
- les défis 263 à 265 utilisent `distance_km` (défi 260). Pendant leur vérification, le correcteur lui substitue
  la sienne : chaque défi est noté seul, mais faites le 260 d'abord, sinon vos essais seront faux.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_43 import *
    >>> clients = clients_exemple()                # 6 000 clients : position, service, parti ou non
    >>> clients["C00002"]
    {'position': (44.909572, 4.873351), 'service': 'Fibre', 'parti': False}
    >>> antennes = antennes_exemple()              # 280 antennes : {antenne: (lat, lon)}
    >>> zone = zone_fibre_exemple()                # GeoJSON : positions [lon, lat], un trou dans la ville
    >>> plaintes = plaintes_exemple()              # 845 appels « plus de réseau » : [(lat, lon), …]

Quand les six défis sont réussis, Yanis peut répondre à l'équipe réseau :

    >>> from collections import Counter
    >>> sum(dans_zone(c["position"], zone) for c in clients.values())        # clients dans la zone fibre
    1402
    >>> Counter(geohash(p, 5) for p in plaintes).most_common(3)               # les cases de 5 km les plus plaintives
    [('spgv9', 61), ('spgme', 57), ('u05b4', 50)]
    >>> etiquettes = regrouper_pannes(plaintes, rayon_km=1.0, minimum=10)
    >>> sorted(Counter(e for e in etiquettes if e >= 0).values())            # trois foyers de plaintes
    [60, 87, 117]
    >>> foyers = {}
    >>> for (lat, lon), e in zip(plaintes, etiquettes):
    ...     if e >= 0:
    ...         foyers.setdefault(e, []).append((lat, lon))
    >>> centres = {e: (sum(lat for lat, _ in pts) / len(pts), sum(lon for _, lon in pts) / len(pts))
    ...            for e, pts in foyers.items()}
    >>> sorted(a for a, d in plus_proches(centres, antennes).values())     # l'antenne au cœur de chaque foyer
    ['A0010', 'A0062', 'A0192']
    >>> rattachement = plus_proches({c: f["position"] for c, f in clients.items()}, antennes)
    >>> touches = [c for c, (a, d) in rattachement.items() if a in {"A0010", "A0062", "A0192"}]
    >>> len(touches), sum(clients[c]["parti"] for c in touches)              # 41 % de départs, contre 7 % ailleurs
    (73, 30)
    >>> ordre, km = tournee(DEPOT, sites_exemple())
    >>> len(ordre), round(km)                                                 # 941 km dans l'ordre du tableur
    (24, 354)

Les trois cases de 5 km d'où viennent le plus de plaintes, les trois foyers trouvés par DBSCAN et les trois antennes
au cœur de ces foyers disent la même chose : A0010, A0062 et A0192 ne servent plus personne. Leur sonde de
supervision ne vérifie que l'alimentation électrique : le tableau de bord reste vert. Les 73 clients qui leur sont
rattachés sont partis à 41 % (59 % de ceux qui ont une Box 4G), contre 7 % ailleurs ; dans la zone fibre, 5 %. Et le
technicien, qui aurait fait 941 km en suivant le tableur (deux jours de route), en fait 354 : une journée.

Remarquez aussi « spgv8 », juste après les trois premières : c'est la MÊME panne que « spgv9 », coupée en deux par
la grille des geohash. Une grille découpe le monde à des endroits arbitraires ; DBSCAN, lui, suit les points.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites le 260 d'abord (les défis 263 à 265 s'en
   servent) ; les autres sont indépendants, et l'ordre proposé est le plus progressif (le 264 reprend la recherche
   par bande de latitude du 263).
2. Lancez `python scripts/defis.py --defi 260` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_43.py`.

Les outils communs (`clients_exemple`, `antennes_exemple`, `zone_fibre_exemple`, `plaintes_exemple`,
`sites_exemple`, `generer_points`, `DEPOT`, `RAYON_TERRE_KM` et `BASE32_GEOHASH`) sont dans `defis/carte.py`,
déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import bisect  # noqa: F401 — bisect.bisect_left, bisect.bisect_right
import math  # noqa: F401 — math.radians, math.sin, math.asin, math.isfinite
from collections import deque  # noqa: F401 — une file pour le défi 264

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

# ============================================================================ Mesurer


def distance_km(a, b) -> float:
    """Défi 260 🟢 — La distance à vol d'oiseau entre deux points (lat, lon), en km : la formule de haversine.

        >>> round(distance_km((44.9330, 4.8920), (44.7290, 5.0230)), 2)       # la ville et le bourg
        24.93
        >>> round(distance_km((0, 0), (0, 1)), 3)                              # 1° le long de l'équateur
        111.195
        >>> round(distance_km((-17.5, 179.5), (-17.5, -179.5)), 1)             # de part et d'autre de l'antiméridien
        106.0
        >>> round(distance_km((44.9, 4.89), (-44.9, -175.11)), 1)              # la vallée et ses antipodes
        20015.1
        >>> round(distance_km((44.93, 4.89), (44.93001, 4.89)) * 1000, 3)      # en mètres : un pas
        1.112

    Ici : la ville et le bourg sont à 24,93 km. Aux îles Fidji, deux villages de part et d'autre de l'antiméridien
    (179,5° et -179,5°) sont à 106 km, pas à 359° de longitude. Les antipodes de la vallée, au large de la
    Nouvelle-Zélande, sont à un demi-tour de Terre (π × R = 20 015 km). Et deux points à un pas l'un de l'autre sont
    à 1,112 m : une formule imprécise trouve 0, ou 1,5 m.

    Les règles :

    - `a` et `b` sont des couples (latitude, longitude) en degrés : des tuples OU des listes de deux nombres (int
      ou float). Autre chose (un texte, un bool, un seul nombre, trois nombres) → TypeError ; un NaN, une
      latitude hors de [-90, 90] ou une longitude hors de [-180, 180] → ValueError ;
    - φ1, φ2 = les latitudes en radians, Δφ = φ2 − φ1, Δλ = l'écart de longitude en radians ;
      h = sin²(Δφ / 2) + cos φ1 × cos φ2 × sin²(Δλ / 2) ; la distance = 2 × R × asin(√h), avec R =
      `RAYON_TERRE_KM` (6 371 km, le rayon MOYEN : pas les 6 378 km de l'équateur) ; un float ;
    - entre deux points opposés, h vaut 1 au papier, mais l'arrondi des float donne 1.0000000000000002 (pour les
      antipodes de la vallée, par exemple) : 1 - h devient négatif, et `math.sqrt(1 - h)` (la variante en
      `atan2`) ou `math.acos(1 - 2 * h)` lèvent « math domain error ». Prenez `h = min(1.0, h)`, quelle que soit
      votre formule. Le correcteur essaie les antipodes.

    Ce que vous apprenez : que `math.sin` attend des radians, pourquoi un degré de longitude rétrécit vers les
    pôles, qu'une formule juste au papier (la « loi des cosinus », acos(sin φ1 sin φ2 + cos φ1 cos φ2 cos Δλ))
    peut perdre tous ses chiffres sur de petites distances (acos d'un nombre à 1e-14 de 1), et à protéger une
    fonction mathématique de son domaine (`min(1.0, h)` : un arrondi ne doit pas sortir h de [0, 1]).

    Indice : `phi1, phi2 = math.radians(lat1), math.radians(lat2)` ; `h = math.sin((phi2 - phi1) / 2) ** 2 +
    math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2`. Vérifiez les points avant.
    """
    raise NotImplementedError


def dans_zone(point, zone) -> bool:
    """Défi 261 🟡 — Le point (lat, lon) est-il dans la zone GeoJSON (un « Polygon » ou un « MultiPolygon ») ?

        >>> carre = [[4.0, 44.0], [6.0, 44.0], [6.0, 46.0], [4.0, 46.0], [4.0, 44.0]]      # [lon, lat], fermé
        >>> trou = [[4.5, 44.5], [5.0, 44.5], [5.0, 45.0], [4.5, 45.0], [4.5, 44.5]]
        >>> zone = {"type": "Polygon", "coordinates": [carre, trou]}
        >>> dans_zone((45.5, 5.5), zone), dans_zone((44.8, 4.8), zone), dans_zone((47.0, 5.0), zone)
        (True, False, False)
        >>> fibre = zone_fibre_exemple()
        >>> dans_zone((44.909572, 4.873351), fibre), dans_zone((44.937475, 4.880098), fibre)
        (True, False)

    Ici (1er exemple) : un carré de 2° de côté, percé d'un trou. (45,5 ; 5,5) est dans le carré, hors du trou :
    dedans. (44,8 ; 4,8) est dans le trou : dehors. (47 ; 5) est au nord du carré. Dans la zone fibre, le client
    C00002 est en ville ; le client C00003 habite le quartier des Tilleuls, le trou de la zone : pas encore de
    fibre, même s'il est au milieu de la ville.

    Les règles :

    - `point` suit les règles du défi 260 (TypeError ou ValueError) ; `zone` est un dict (sinon TypeError) dont
      "type" vaut "Polygon" ou "MultiPolygon" (sinon ValueError) ;
    - un Polygon : "coordinates" est une liste d'ANNEAUX, le premier est le bord extérieur, les suivants sont des
      TROUS. Un MultiPolygon : "coordinates" est une liste de polygones (chacun, une liste d'anneaux). Un point
      est dans un polygone s'il est dans son anneau extérieur et dans AUCUN de ses trous ; dans un MultiPolygon,
      s'il est dans l'un de ses polygones ;
    - un anneau est une liste d'au moins 4 positions, FERMÉE (la dernière égale la première) ; une position est
      [LONGITUDE, latitude] (dans cet ordre !), parfois suivie d'une altitude : [lon, lat, alt], qu'on ignore. Un
      anneau trop court ou non fermé, une position de moins de deux nombres → ValueError. Toute la zone est
      vérifiée, même si le point est déjà dans son premier polygone ;
    - pour savoir si un point est dans un anneau, partez du point vers l'EST et comptez combien de fois vous
      croisez le bord : impair = dedans (le « lancer de rayon »). Une arête (x1, y1) → (x2, y2) (x = longitude,
      y = latitude) compte si UN de ses bouts est strictement au-dessus du point et pas l'autre : `(y1 > lat) !=
      (y2 > lat)`, et si elle croise la ligne du point à l'est : `lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1)` ;
    - les zones restent petites (quelques dizaines de km) : on traite longitude et latitude comme x et y sur un
      plan. Le correcteur ne place jamais un point EXACTEMENT sur un bord, mais il en place à la hauteur exacte
      d'un sommet, et le long d'une arête horizontale.

    Ce que vous apprenez : à lire un format standard (RFC 7946 : l'ordre [lon, lat], les anneaux fermés, les
    trous), l'algorithme du lancer de rayon (pair / impair), et la règle « un bout strictement au-dessus » qui
    compte un sommet une seule fois et ignore les arêtes horizontales, là où `min(y1, y2) <= lat <= max(y1, y2)`
    compte deux fois le sommet partagé par deux arêtes (et divise par zéro sur une arête horizontale).

    Indice : `for (x1, y1), (x2, y2) in zip(anneau, anneau[1:])` parcourt les arêtes (l'anneau est fermé : la
    dernière arête revient au départ) ; `dedans = not dedans` à chaque croisement. Écrivez d'abord
    `dans_anneau(lon, lat, anneau)`, puis les trous, puis le MultiPolygon.
    """
    raise NotImplementedError


def geohash(point, precision: int = 7) -> str:
    """Défi 262 🟡 — Le geohash du point (lat, lon) : le nom de la case où il tombe, sur `precision` caractères.

        >>> geohash((57.64911, 10.40744), 11)          # l'exemple de référence de geohash.org
        'u4pruydqqvj'
        >>> geohash((44.9330, 4.8920)), geohash((44.9330, 4.8920), 5)   # la ville : 7 caractères, puis 5
        ('spgrx7f', 'spgrx')
        >>> geohash((0, 0), 5)                         # pile au milieu : `>=` va à droite
        's0000'

    Ici : un geohash coupe le monde en deux, encore et encore. Chaque caractère ajoute 5 coupures, et la case
    rétrécit : 5 caractères ≈ 4,9 km × 4,9 km, 7 caractères ≈ 153 m × 153 m (à l'équateur ; un peu moins larges
    dans la vallée, où un degré de longitude fait moins de km). Deux points proches partagent (presque toujours) le
    début de leur code : compter les plaintes par `geohash(p, 5)`, c'est dessiner une carte de chaleur avec un simple
    `Counter` (ou un GROUP BY : Elasticsearch, Redis et BigQuery font exactement ça).

    Les règles :

    - on part des intervalles [-180, 180] pour la longitude et [-90, 90] pour la latitude. À chaque étape, on
      coupe l'intervalle d'UN des deux axes en deux, en commençant par la LONGITUDE, puis la latitude, puis la
      longitude…  Si la coordonnée est >= au milieu, le bit vaut 1 et on garde la moitié haute ; sinon 0 et la
      moitié basse ;
    - les bits sont pris 5 par 5, le premier bit étant celui de poids FORT (`valeur = 2 * valeur + bit`) ; chaque
      groupe de 5 bits (un nombre de 0 à 31) devient un caractère de `BASE32_GEOHASH`
      ("0123456789bcdefghjkmnpqrstuvwxyz" : sans a, i, l ni o ; ce n'est PAS l'alphabet de `base64.b32encode`) ;
    - `point` suit les règles du défi 260 ; `precision` est un int (pas un bool, ni un float) → sinon TypeError,
      entre 1 et 12 → sinon ValueError ; la réponse est un str de `precision` caractères.

    Ce que vous apprenez : la dichotomie appliquée à l'espace (comme `bisect`, mais sur deux axes entrelacés), à
    fabriquer un nombre bit à bit (`2 * valeur + bit`), qu'un index spatial peut n'être qu'un texte trié (deux
    codes qui commencent pareil sont proches : un index B-arbre ordinaire suffit), et sa limite : deux points de
    part et d'autre d'une coupure (« spgv8 » et « spgv9 ») ont des codes différents, même à 10 m l'un de l'autre.

    Indice : `bas, haut = -180.0, 180.0` (et de même pour la latitude) ; `milieu = (bas + haut) / 2` ; un booléen
    `longitude = True` qui s'inverse à chaque bit ; `if nb_bits == 5: texte += BASE32_GEOHASH[valeur]`.
    """
    raise NotImplementedError


# ============================================================================ Chercher


def plus_proches(clients: dict, antennes: dict) -> dict:
    """Défi 263 🟡 — {client: (antenne la plus proche, distance en km)}, sans calculer toutes les distances.

        >>> antennes = {"A2": (45.0, 5.06), "A7": (45.03, 5.0), "A3": (45.03, 5.0), "A9": (45.3, 5.0)}
        >>> resultat = plus_proches({"C1": (45.0, 5.0), "C2": (45.04, 5.0)}, antennes)
        >>> {client: (antenne, round(km, 2)) for client, (antenne, km) in resultat.items()}
        {'C1': ('A3', 3.34), 'C2': ('A3', 1.11)}

    Ici : C1 est à 3,34 km de A3 et A7 (au nord, sur le même pylône) et à 4,72 km de A2 (à l'est, à la même
    latitude). A2 est la plus proche EN LATITUDE, mais pas la plus proche : une recherche qui s'arrête à la première
    antenne trouvée se trompe. A3 et A7 sont à égalité : la règle garde le plus petit identifiant, « A3 ». Sur
    l'exemple, 6 000 clients et 280 antennes, c'est 1,7 million de distances à tout comparer ; le correcteur passe
    200 clients et 20 000 antennes (4 millions), et compte les lignes exécutées.

    Les règles :

    - `clients` et `antennes` sont des dict {identifiant: (lat, lon)} (sinon TypeError), les points suivent les
      règles du défi 260 ; aucune antenne → ValueError (personne n'a d'antenne la plus proche) ; aucun client → {} ;
    - la réponse a une clé par client, dans l'ordre de `clients` ; la valeur est un TUPLE (antenne, distance), la
      distance étant celle de `distance_km`, en km ;
    - à distance égale (deux antennes sur le même pylône), l'antenne de plus PETIT identifiant (« A3 » avant
      « A7 ») : comparez des tuples `(distance, identifiant)` ;
    - pas de double boucle clients × antennes : rangez les antennes par latitude UNE fois, placez chaque client
      parmi elles avec `bisect`, puis éloignez-vous des deux côtés, toujours du côté le plus proche en latitude.
      La distance du grand cercle est AU MOINS l'écart de latitude, en km (`RAYON_TERRE_KM *
      math.radians(abs(lat - lat_antenne))`) : dès que cet écart seul dépasse la meilleure distance trouvée,
      aucune antenne plus loin en latitude ne peut faire mieux, et vous pouvez vous arrêter. Une grille de cases
      convient aussi, à condition de ne jamais manquer une antenne plus proche dans une case voisine.

    Ce que vous apprenez : la recherche du plus proche voisin (ce que font les « k-d tree » et « ball tree » de
    scikit-learn, et l'index GiST de PostGIS), à élaguer une recherche avec une BORNE inférieure qui ne se trompe
    jamais, et pourquoi trier une fois (O(m log m)) puis chercher (O(log m) + quelques voisins) bat n × m.

    Indice : `rangees = sorted((lat, lon, ident) for ident, (lat, lon) in antennes.items())`, `latitudes = [r[0]
    for r in rangees]` ; `haut = bisect.bisect_left(latitudes, lat)`, `bas = haut - 1` ; dans la boucle, prenez
    `bas` ou `haut` selon le plus petit écart de latitude, et `meilleur = min(meilleur, (distance, ident))`.
    """
    raise NotImplementedError


def regrouper_pannes(plaintes, rayon_km: float, minimum: int) -> list[int]:
    """Défi 264 🔴 — DBSCAN : le numéro du foyer de chaque plainte (0, 1, 2…), ou -1 pour une plainte isolée.

        >>> def a_km(nord_km):                         # un point à `nord_km` au nord de (45, 5)
        ...     return (45.0 + nord_km / 111.19492664455873, 5.0)
        >>> plaintes = [a_km(0.0), a_km(0.6), a_km(1.2), a_km(2.0), a_km(9.0), a_km(9.5), a_km(10.0), a_km(30)]
        >>> regrouper_pannes(plaintes, rayon_km=0.7, minimum=3)
        [0, 0, 0, -1, 1, 1, 1, -1]

    Ici : les plaintes sont posées le long d'une route nord-sud, à 0 ; 0,6 ; 1,2 ; 2 ; 9 ; 9,5 ; 10 et 30 km. Avec
    `rayon_km=0.7` et `minimum=3`, la plainte à 0,6 km a 3 voisines à 0,7 km ou moins (celles à 0 et 1,2 km, et
    elle-même) : elle est CENTRALE, et ouvre le foyer 0. Celles à 0 et 1,2 km n'ont que 2 voisines : ce sont des
    points de BORD, pris par le foyer 0, mais qui ne l'étendent pas. Celle à 2 km est à 0,8 km de la plus proche :
    isolée (-1). À 9,5 km, un deuxième foyer (1) ; à 30 km, une plainte isolée. Sur les 845 plaintes de l'exemple,
    `rayon_km=1.0` et `minimum=10` trouvent trois foyers de 60, 87 et 117 plaintes, et 581 plaintes isolées.

    Les règles (celles de `sklearn.cluster.DBSCAN`) :

    - les VOISINES d'une plainte sont les plaintes à `distance_km` <= `rayon_km` d'elle, ELLE COMPRISE. Une plainte
      est CENTRALE si elle a au moins `minimum` voisines ;
    - on parcourt les plaintes dans l'ordre de la liste. Chaque plainte centrale qui n'a pas encore de foyer en
      ouvre un nouveau (numérotés 0, 1, 2… dans l'ordre d'ouverture), qui s'étend de proche en proche : toute
      voisine d'une plainte centrale du foyer rejoint le foyer si elle n'en a pas encore ; si elle est centrale à
      son tour, ses voisines rejoignent aussi le foyer, et ainsi de suite. Une plainte de BORD (prise, mais pas
      centrale) n'étend rien ;
    - une plainte déjà prise par un foyer n'est jamais reprise par un autre ; une plainte vue « isolée » au début
      du parcours peut être prise PLUS TARD par un foyer (isolée veut dire « pas encore de foyer ») ;
    - la réponse est une liste d'int, une étiquette par plainte, dans l'ordre de `plaintes` ; [] si aucune ;
    - `plaintes` est une liste (ou un tuple) de points (règles du défi 260) ; `rayon_km` un int ou un float > 0 (sinon
      TypeError ou ValueError) ; `minimum` un int >= 1 (pas un bool : sinon TypeError ou ValueError) ;
    - le correcteur passe 3 000 plaintes en CHAÎNE (une récursion lèverait RecursionError : utilisez une file,
      `collections.deque`) et 4 000 plaintes dont il compte les lignes exécutées : comparer chaque plainte à
      toutes les autres (16 millions de paires) est trop lent. Cherchez les voisines dans une bande de latitude,
      comme au défi 263 : `math.degrees(rayon_km / RAYON_TERRE_KM)` degrés de chaque côté, puis `distance_km`.

    Ce que vous apprenez : le regroupement par DENSITÉ (il trouve des foyers de forme quelconque, et laisse le
    bruit de côté, là où k-means, série 32, range chaque point dans un groupe), le parcours en largeur avec une
    file (défi 61) plutôt qu'une récursion, et les deux pièges classiques de DBSCAN : le point qui oublie de se
    compter lui-même, et le point de bord qui étend le foyer de proche en proche jusqu'à tout avaler.

    Indice : calculez d'abord `voisines[i]` (une liste d'indices) pour chaque plainte, puis `centrale = [len(v) >=
    minimum for v in voisines]` ; `etiquettes = [-1] * n` ; pour ouvrir un foyer : `file = deque([i])`, puis
    `while file:` et `for k in voisines[file.popleft()]: if etiquettes[k] == -1: …`.
    """
    raise NotImplementedError


# ============================================================================ Se déplacer


def tournee(depart, sites: dict) -> tuple[list, float]:
    """Défi 265 🔴 — (ordre de visite, longueur en km) : la tournée du technicien, du dépôt au dépôt.

        >>> sites = {"N": (45.1, 5.0), "E": (45.0, 5.1), "S": (44.9, 5.0), "O": (45.0, 4.9)}
        >>> ordre, km = tournee((45.0, 5.0), sites)
        >>> ordre in (["N", "E", "S", "O"], ["N", "O", "S", "E"], ["E", "S", "O", "N"], ["O", "S", "E", "N"],
        ...           ["E", "N", "O", "S"], ["O", "N", "E", "S"], ["S", "E", "N", "O"], ["S", "O", "N", "E"])
        True
        >>> round(km, 1)
        59.8

    Ici : le dépôt est au centre d'une croix, les quatre sites à 7,9 ou 11,1 km au nord, à l'est, au sud et à
    l'ouest. La bonne tournée fait le tour (N, E, S, O, ou un ordre équivalent) : 59,8 km, retour au dépôt compris.
    Sur l'exemple, le technicien doit visiter 24 antennes : dans l'ordre du tableur (par identifiant), c'est 941 km ;
    en allant toujours au site le plus proche, 384 km ; après 2-opt, 354 km.

    Les règles :

    - `depart` est un point (règles du défi 260) ; `sites` un dict {identifiant: point} (sinon TypeError), qui
      n'est pas modifié ;
    - la réponse est un tuple (ordre, longueur) : `ordre` est la liste des identifiants de `sites`, chacun UNE
      fois (pas le dépôt) ; `longueur` est la longueur en km (avec `distance_km`) de la tournée dépôt → ordre[0] →
      … → ordre[-1] → dépôt : le RETOUR au dépôt compte. Aucun site → ([], 0.0) ;
    - la méthode : d'abord le PLUS PROCHE VOISIN (du dépôt, aller au site non visité le plus proche, et ainsi de
      suite ; à distance égale, le plus petit identifiant), puis 2-OPT : tant qu'il existe deux arêtes (a → b) et
      (c → d) de la tournée telles que d(a, b) + d(c, d) > d(a, c) + d(b, d) + 1e-9 (le chemin se croise), retourner
      le morceau b … c (`route[i:j + 1] = route[i:j + 1][::-1]`), ce qui remplace (a → b) et (c → d) par (a → c)
      et (b → d). Les arêtes qui partent du dépôt ou y reviennent comptent aussi ; le dépôt, lui, ne bouge pas ;
    - 2-opt s'arrête sur UN optimum local, qui dépend de l'ordre des essais : le correcteur n'exige pas UNE
      tournée. Il vérifie que `ordre` contient chaque site une fois, que `longueur` est bien celle de `ordre`, qu'il
      ne reste AUCUN échange 2-opt qui raccourcisse la tournée de plus d'un millimètre, et qu'elle n'est pas plus
      longue que celle du plus proche voisin ;
    - le correcteur passe 80 sites et compte les lignes exécutées : calculez chaque distance UNE fois (une table),
      et le gain d'un échange avec ses QUATRE distances, pas en recalculant toute la tournée (n fois plus lent).
      Le `1e-9` compte : sans lui, deux échanges peuvent se défaire l'un l'autre sans fin, à cause d'un arrondi.

    Ce que vous apprenez : une heuristique gloutonne (le plus proche voisin), une recherche locale (2-opt :
    améliorer tant qu'on peut, avec un gain calculé en O(1)), qu'un problème trop dur pour être résolu exactement
    (le voyageur de commerce) se résout « assez bien » en pratique, et à tester une réponse par ses PROPRIÉTÉS
    quand il y a plusieurs bonnes réponses.

    Indice : `points = [depart] + [sites[s] for s in noms]` ; `table = [[distance_km(p, q) for q in points] for p
    in points]` ; la route est une liste d'indices qui commence par 0 (le dépôt) ; pour 2-opt, `for i in range(1,
    n - 1): for j in range(i + 1, n):` avec a, b, c, d = route[i - 1], route[i], route[j], route[(j + 1) % n].
    """
    raise NotImplementedError

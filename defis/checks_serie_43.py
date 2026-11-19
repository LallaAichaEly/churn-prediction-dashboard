"""Vérifications automatiques de la série 43 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est l'équipe réseau, qui ne croira à une panne invisible que si vos calculs tiennent, et le
service SIG, qui vous envoie ses fichiers tels qu'ils sont vraiment.

- il compare vos distances, vos zones, vos geohash, vos antennes et vos foyers à ceux d'une version de référence
  (à un milliardième près pour les distances : les règles sont fixées pour qu'il n'y ait qu'une bonne réponse) ;
  pour la tournée, où plusieurs réponses sont justes, il vérifie ses PROPRIÉTÉS (chaque site une fois, la bonne
  longueur, plus aucun échange 2-opt qui raccourcisse, pas plus longue que le plus proche voisin) ;
- quand votre résultat est faux, il rejoue les erreurs classiques (des degrés passés à `sin`, Pythagore sur des
  degrés, la loi des cosinus, le rayon de l'équateur, [lon, lat] lu comme (lat, lon), les trous oubliés, un sommet
  compté deux fois, la latitude coupée d'abord, `>` au lieu de `>=`, la première antenne trouvée, le point qui ne
  se compte pas lui-même, le point de bord qui étend le foyer, le bruit définitif…) pour nommer la vôtre ;
- il choisit des points où chaque erreur se voit (les antipodes, l'antiméridien, deux points à un mètre, un sommet
  à la hauteur du point, une arête horizontale, deux antennes sur le même pylône, 3 000 plaintes en chaîne), et
  compte les lignes exécutées dans votre fichier quand comparer tout le monde à tout le monde serait trop lent ;
- pendant les défis 263 à 265, il prête sa propre `distance_km` à votre fichier : chaque défi est noté seul.
"""

from __future__ import annotations

import bisect
import math
import random
import sys
from collections import Counter, deque

from .carte import (
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
from .verif import Defi, Serie

_KM_PAR_DEGRE = math.pi * RAYON_TERRE_KM / 180

# ---------------------------------------------------------------------------- outils


def _decrire(exc: BaseException) -> str:
    texte = str(exc).split("\n")[0]
    if len(texte) > 160:
        texte = texte[:159] + "…"
    return f"{type(exc).__name__}({texte!r})" if texte else type(exc).__name__


def _court(valeur, n: int = 160) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


class _BoucleSansFin(BaseException):
    """Levée dans VOTRE code quand il travaille trop (BaseException : `except Exception` ne l'attrape pas)."""


def _fichier(objet) -> str | None:
    code = getattr(objet, "__code__", None) or getattr(getattr(objet, "__func__", None), "__code__", None)
    if code is not None:
        return code.co_filename
    module = sys.modules.get(getattr(objet, "__module__", None) or "")
    return getattr(module, "__file__", None)


def _trace(fichier: str | None, appel, lignes_max: int | None):
    """Le résultat de `appel()` ; arrête net au-delà de `lignes_max` lignes exécutées dans `fichier`."""
    if fichier is None or lignes_max is None:
        return appel()
    compte = [0]

    def local(frame, evenement, arg):
        if evenement == "line":
            compte[0] += 1
            if compte[0] > lignes_max:
                raise _BoucleSansFin
        return local

    def global_(frame, evenement, arg):
        return local if frame.f_code.co_filename == fichier else None

    precedent = sys.gettrace()
    sys.settrace(global_)
    try:
        return appel()
    finally:
        sys.settrace(precedent)


_CONSEIL_LENT = "Votre code a été arrêté : il exécute beaucoup trop de lignes (une boucle qui ne s'arrête pas ?)."


def _conseil_exception(exc: BaseException) -> str:
    texte = str(exc)
    if isinstance(exc, ValueError) and "math domain error" in texte:
        return ("`math.sqrt`, `math.asin` ou `math.acos` a reçu un nombre hors de son domaine : entre deux points "
                "opposés, h vaut 1 au papier mais 1.0000000000000002 en float, et 1 - h devient négatif. Bornez-le "
                "avant de vous en servir : `h = min(1.0, h)`.")
    if isinstance(exc, ZeroDivisionError):
        return ("Une division par zéro : une arête HORIZONTALE (y1 == y2) à la hauteur du point. Avec la règle `(y1 > "
                "lat) != (y2 > lat)`, une arête horizontale n'est jamais retenue, et l'on ne divise jamais par y2 - y1 "
                "= 0 (`min(y1, y2) <= lat <= max(y1, y2)` la retient).")
    if isinstance(exc, TypeError) and "not supported between" in texte:
        return ("Python ne sait pas comparer ces deux valeurs : comparez des tuples (distance, identifiant), avec une "
                "vraie distance dès le départ (pas None).")
    if isinstance(exc, KeyError):
        return "Une clé manque : la réponse a une clé par client (ou par site), et rien d'autre."
    return ""


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = None, conseil_lent: str = "",
           kwargs: dict | None = None):
    """f(*args), en transformant toute erreur en message lisible (NotImplementedError : le défi est à faire)."""
    kwargs = kwargs or {}
    try:
        return _trace(_fichier(f) if lignes_max else None, lambda: f(*args, **kwargs), lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError. {conseil}".rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {_conseil_exception(exc) or conseil}".rstrip()) from None


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu: type = ValueError, kwargs: dict | None = None):
    """Exige que f(*args) lève `attendu` (et rien d'autre). Renvoie l'exception, pour lire son message."""
    kwargs = kwargs or {}
    try:
        resultat = f(*args, **kwargs)
    except NotImplementedError:
        raise
    except attendu as exc:
        return exc
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _premier_bug(obtenus, rejouer, bugs: dict, egal) -> str:
    """Le conseil du premier bug dont le rejeu donne EXACTEMENT vos réponses."""
    for bug, conseil in bugs.items():
        try:
            faux = rejouer(bug)
        except Exception:
            continue
        if egal(faux, obtenus):
            return conseil
    return ""


def _nombre(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _proche(a, b, tol: float = 1e-9) -> bool:
    return _nombre(a) and _nombre(b) and abs(a - b) <= tol * max(1.0, abs(b))


def _listes(egal):
    return lambda a, b: len(a) == len(b) and all(egal(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------- les points


def _ref_point(point, nom: str = "point") -> tuple[float, float]:
    """(lat, lon) vérifiés comme l'exigent les règles : TypeError ou ValueError."""
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise TypeError(f"{nom} doit être un couple (latitude, longitude), pas {point!r}")
    for v in point:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"{nom} : {v!r} n'est pas un nombre")
        if not math.isfinite(v):
            raise ValueError(f"{nom} : {v!r} n'est pas un nombre fini")
    lat, lon = float(point[0]), float(point[1])
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"{nom} hors du globe : {point!r}")
    return lat, lon


def _distance_pretee(a, b) -> float:
    """La `distance_km` que le correcteur prête à votre fichier pendant les défis 263 à 265 (elle vérifie ses points)."""
    return _ref_distance(_ref_point(a, "a"), _ref_point(b, "b"))


class _DistancePretee:
    """Remplace `distance_km` dans le module de `f` le temps d'une vérification, puis remet la vôtre."""

    def __init__(self, f) -> None:
        self.module = sys.modules.get(getattr(f, "__module__", None) or "")
        self.avant = None

    def __enter__(self):
        if self.module is not None and hasattr(self.module, "distance_km"):
            self.avant = self.module.distance_km
            self.module.distance_km = _distance_pretee
        return self

    def __exit__(self, *exc) -> None:
        if self.avant is not None:
            self.module.distance_km = self.avant


def _refus_points(f, construire, nom: str) -> None:
    """Les points invalides : TypeError (pas un couple de nombres) ou ValueError (hors du globe, NaN)."""
    for point, attendu, conseil in (
            ((91.0, 4.9), ValueError, "Une latitude de 91° n'existe pas (souvent une longitude écrite à sa place)."),
            ((44.9, 181.0), ValueError, "Une longitude de 181° n'existe pas (entre -180 et 180)."),
            ((float("nan"), 4.9), ValueError, "NaN n'est pas une latitude : `-90 <= nan <= 90` est FAUX, testez-le."),
            (("44.9", 4.9), TypeError, "« 44.9 » est un texte : un point est un couple de NOMBRES."),
            ((True, 4.9), TypeError, "True n'est pas une latitude (`isinstance(True, int)` est vrai : excluez bool)."),
            ((44.9,), TypeError, "Un seul nombre n'est pas un point (latitude, longitude)."),
            ((44.9, 4.9, 250.0), TypeError, "Trois nombres, ce n'est pas un couple (latitude, longitude)."),
            (44.9, TypeError, "Un nombre seul n'est pas un point.")):
        args, kwargs = construire(point)
        _doit_lever(f, args, f"{nom} avec le point {point!r}", conseil, attendu, kwargs=kwargs)


# ============================================================================ défi 260 : distance_km


def _ref_distance(a, b, bug: str = "") -> float:
    """La distance de haversine ; `bug` rejoue une erreur classique."""
    (lat1, lon1), (lat2, lon2) = a, b
    if bug == "inverse":                       # (lat, lon) lus (lon, lat)
        lat1, lon1, lat2, lon2 = lon1, lat1, lon2, lat2
    if bug == "pythagore":
        return _KM_PAR_DEGRE * math.hypot(lat2 - lat1, lon2 - lon1)
    if bug == "degres":
        phi1, phi2, dlam = lat1, lat2, lon2 - lon1
    else:
        phi1, phi2, dlam = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    rayon = 6378.137 if bug == "equateur" else RAYON_TERRE_KM
    if bug == "equirectangulaire":
        return rayon * math.hypot(dlam * math.cos((phi1 + phi2) / 2), phi2 - phi1)
    if bug == "cosinus":
        c = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(dlam)
        return rayon * math.acos(max(-1.0, min(1.0, c)))
    if bug == "un_moins_cos":
        h = (1 - math.cos(phi2 - phi1)) / 2 + math.cos(phi1) * math.cos(phi2) * (1 - math.cos(dlam)) / 2
    else:
        h = math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    angle = math.asin(math.sqrt(max(0.0, min(1.0, h))))         # max(0.0, …) : pour les rejeux qui lisent mal
    return rayon * angle if bug == "sans_deux" else 2 * rayon * angle


_BUGS_DISTANCE = {
    "degres": "Vous passez des DEGRÉS à `math.sin` et `math.cos`, qui attendent des RADIANS : sin(45) n'est pas "
              "sin(45°). Convertissez avec `math.radians` (latitudes ET écart de longitude).",
    "pythagore": "Vous appliquez Pythagore sur les degrés (× 111 km) : 1° de longitude ne vaut 111 km qu'à l'équateur "
                 "(79 km dans la vallée, 0 au pôle), et la Terre est ronde. Formule de haversine : h = sin²(Δφ / 2) + "
                 "cos φ1 × cos φ2 × sin²(Δλ / 2), puis 2 × R × asin(√h).",
    "equirectangulaire": "Vous projetez sur un plan (Δλ × cos(latitude moyenne), Δφ) : presque juste pour quelques km, "
                         "faux au loin, et de part et d'autre de l'antiméridien (179,5° et -179,5°) l'écart de longitude "
                         "n'est pas 359° mais 1°. La formule de haversine (sin² du DEMI-écart) règle les deux.",
    "cosinus": "Vous utilisez la « loi des cosinus », R × acos(sin φ1 sin φ2 + cos φ1 cos φ2 cos Δλ) : juste au papier, "
               "mais pour deux points à un mètre l'argument vaut 0,99999999999998 et acos perd presque tous les "
               "chiffres (1,115 m au lieu de 1,112 m). Haversine : sin² du DEMI-angle reste précis.",
    "un_moins_cos": "Vous calculez h avec (1 - cos x) / 2 au lieu de sin²(x / 2) : c'est la même chose au papier, mais "
                    "1 - cos x, pour un x minuscule, soustrait deux nombres presque égaux et perd presque tous les "
                    "chiffres. `math.sin(x / 2) ** 2` reste précis.",
    "sans_deux": "Il manque le facteur 2 : la distance est 2 × R × asin(√h) (asin donne la MOITIÉ de l'angle au centre).",
    "equateur": "Votre rayon est celui de l'équateur (6 378 km) : la règle prend le rayon MOYEN, `RAYON_TERRE_KM` "
                "(6 371 km).",
    "inverse": "Vous lisez les points (longitude, latitude) : la règle est (LATITUDE, longitude), comme un GPS. "
               "a[0] est la latitude.",
}


def _cas_260() -> list[tuple]:
    cas = [
        ((44.9330, 4.8920), (44.7290, 5.0230)),          # la ville et le bourg
        ((0, 0), (0, 1)),                                # 1° le long de l'équateur
        ((0, 0), (1, 0)),                                # 1° le long d'un méridien
        ([44.93, 4.89], [44.93, 4.89]),                  # le même point, en listes
        ((-17.5, 179.5), (-17.5, -179.5)),               # de part et d'autre de l'antiméridien
        ((90, 0), (89, 120)),                            # au pôle, la longitude ne compte plus
        ((44.93, 4.89), (44.93001, 4.89)),               # un pas vers le nord
        ((44.93, 4.89), (44.93, 4.89001)),               # un pas vers l'est
        ((48.8566, 2.3522), (40.7128, -74.006)),         # Paris - New York
        ((-33.8688, 151.2093), (51.5074, -0.1278)),      # Sydney - Londres
        ((44.9, 4.89), (-44.9, -175.11)),                # la vallée et ses antipodes : h = 1.0000000000000002
        ((2.5, 0.0), (-2.5, 180.0)),                     # d'autres antipodes
        ((0, 0), (0, 180)),
        ((60, 10), (60.5, 11)),
        ((-45, -170), (-46, 175)),
    ]
    alea = random.Random(260)
    for _ in range(25):
        cas.append(((alea.uniform(-90, 90), alea.uniform(-180, 180)), (alea.uniform(-90, 90), alea.uniform(-180, 180))))
    cas += list(zip(generer_points(15, graine=2601), generer_points(15, graine=2602)))
    return cas


def verifier_distance_km(f) -> None:
    cas = _cas_260()
    obtenus = []
    for a, b in cas:
        quoi = f"distance_km({a!r}, {b!r})"
        copie_a, copie_b = list(a), list(b)
        r = _appel(f, (a, b), quoi)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not _nombre(r):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) : attendu un nombre de km (float).")
        if list(a) != copie_a or list(b) != copie_b:
            raise AssertionError(f"{quoi} a modifié ses arguments : une distance ne touche pas aux points reçus.")
        obtenus.append(r)
    for (a, b), r in zip(cas, obtenus):
        attendu = _ref_distance(a, b)
        if not _proche(r, attendu):
            conseil = _premier_bug(obtenus, lambda bug: [_ref_distance(x, y, bug) for x, y in cas], _BUGS_DISTANCE,
                                   _listes(_proche))
            if not conseil:
                rapports = {round(o / _ref_distance(x, y), 9) for (x, y), o in zip(cas, obtenus) if _ref_distance(x, y) > 1}
                if len(rapports) == 1 and rapports != {1.0}:
                    conseil = (f"Toutes vos distances sont celles attendues × {rapports.pop():g} : votre rayon (ou votre "
                               f"unité) n'est pas le bon. En km, avec R = RAYON_TERRE_KM = {RAYON_TERRE_KM:g} km, et 2 × R.")
            raise AssertionError(f"distance_km({a!r}, {b!r}) vaut {r!r} au lieu de {attendu!r}. {conseil}".rstrip())

    sens = [(b, a) for a, b in cas]
    for (a, b), r in zip(sens, obtenus):
        retour = _appel(f, (a, b), f"distance_km({a!r}, {b!r})")
        if not _proche(retour, r):
            raise AssertionError(f"distance_km(a, b) vaut {r!r} mais distance_km(b, a) vaut {retour!r} pour a = {b!r}, "
                                 f"b = {a!r} : une distance ne dépend pas du sens.")

    _refus_points(f, lambda p: ((p, (44.9, 4.9)), {}), "distance_km(point, (44.9, 4.9))")
    _refus_points(f, lambda p: (((44.9, 4.9), p), {}), "distance_km((44.9, 4.9), point)")


# ============================================================================ défi 261 : dans_zone


def _ref_lire_zone(zone) -> list[list[list[tuple[float, float]]]]:
    """[polygone][anneau][(x = lon, y = lat)], vérifié comme l'exigent les règles."""
    if not isinstance(zone, dict):
        raise TypeError("la zone est un dict GeoJSON")
    genre, coordonnees = zone.get("type"), zone.get("coordinates")
    if genre not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"type {genre!r}")
    polygones = [coordonnees] if genre == "Polygon" else coordonnees
    if not isinstance(polygones, (list, tuple)):
        raise ValueError("coordonnées illisibles")
    lus = []
    for polygone in polygones:
        if not isinstance(polygone, (list, tuple)) or not polygone:
            raise ValueError("polygone illisible")
        anneaux = []
        for anneau in polygone:
            if not isinstance(anneau, (list, tuple)) or len(anneau) < 4:
                raise ValueError("anneau trop court")
            positions = []
            for position in anneau:
                if not isinstance(position, (list, tuple)) or len(position) < 2 \
                        or not all(_nombre(v) for v in position[:2]):
                    raise ValueError("position illisible")
                positions.append((float(position[0]), float(position[1])))
            if positions[0] != positions[-1]:
                raise ValueError("anneau non fermé")
            anneaux.append(positions)
        lus.append(anneaux)
    return lus


def _ref_dans_anneau(lon: float, lat: float, anneau, bug: str = "") -> bool:
    if bug == "boite":
        xs, ys = [x for x, _ in anneau], [y for _, y in anneau]
        return min(xs) <= lon <= max(xs) and min(ys) <= lat <= max(ys)
    dedans = False
    for (x1, y1), (x2, y2) in zip(anneau, anneau[1:]):
        if bug == "sommet":
            if y1 == y2:
                continue
            croise = min(y1, y2) <= lat <= max(y1, y2)
        else:
            croise = (y1 > lat) != (y2 > lat)
        if croise and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            dedans = not dedans
    return dedans


def _ref_dans_zone(point, zone, bug: str = "") -> bool:
    lat, lon = point
    polygones = _ref_lire_zone(zone)
    if bug == "inverse":
        polygones = [[[(y, x) for x, y in anneau] for anneau in anneaux] for anneaux in polygones]
    if bug == "premier_polygone":
        polygones = polygones[:1]
    for anneaux in polygones:
        dedans = _ref_dans_anneau(lon, lat, anneaux[0], bug)
        if bug != "sans_trous":
            dedans = dedans and not any(_ref_dans_anneau(lon, lat, trou, bug) for trou in anneaux[1:])
        if dedans:
            return True
    return False


_BUGS_ZONE = {
    "inverse": "Vous lisez les positions GeoJSON comme (latitude, longitude) : GeoJSON (RFC 7946) écrit [LONGITUDE, "
               "latitude], l'ordre inverse d'un GPS. Dans un anneau, position[0] est la longitude (x), position[1] la "
               "latitude (y) ; le point, lui, reste (lat, lon).",
    "sans_trous": "Vous ignorez les TROUS : dans un polygone, les anneaux après le premier sont des trous (le quartier "
                  "des Tilleuls n'a pas la fibre, même au milieu de la ville). Dedans = dans l'anneau extérieur ET dans "
                  "aucun trou.",
    "premier_polygone": "Vous ne regardez que le PREMIER polygone du MultiPolygon : la zone fibre, c'est la ville ET le "
                        "bourg. Un point est dans un MultiPolygon s'il est dans l'UN de ses polygones.",
    "sommet": "Un sommet posé pile à la hauteur du point est compté DEUX fois (une fois par arête qui s'y rejoint) : "
              "`min(y1, y2) <= lat <= max(y1, y2)` retient les deux arêtes. La règle `(y1 > lat) != (y2 > lat)` "
              "attribue le sommet à une seule des deux (et ignore les arêtes horizontales).",
    "boite": "Vous testez le RECTANGLE qui entoure la zone, pas la zone : un point dans le creux d'une zone en U est "
             "dans le rectangle, pas dans la zone. Comptez les croisements du bord (lancer de rayon).",
}


def _carre(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _cas_261() -> list[tuple]:
    mince = {"type": "Polygon", "coordinates": [_carre(4.8, 44.9, 5.0, 44.95)]}
    troue = {"type": "Polygon", "coordinates": [_carre(4.0, 44.0, 6.0, 46.0), _carre(4.5, 44.5, 5.0, 45.0)]}
    en_u = {"type": "Polygon", "coordinates": [[[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3], [0, 0]]]}
    losange = {"type": "Polygon", "coordinates": [[[0, 1], [1, 0], [2, 1], [1, 2], [0, 1]]]}
    en_l = {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4], [0, 0]]]}
    deux = {"type": "MultiPolygon", "coordinates": [[_carre(4.8, 44.8, 4.9, 44.9)], [_carre(5.0, 45.0, 5.1, 45.1)]]}
    altitude = {"type": "Polygon", "coordinates": [[[4.8, 44.9, 210.0], [5.0, 44.9, 180.5], [5.0, 44.95, 250.0],
                                                     [4.8, 44.95, 300.0], [4.8, 44.9, 210.0]]]}
    tuples = {"type": "MultiPolygon", "coordinates": ((((4.8, 44.9), (5.0, 44.9), (4.9, 45.0), (4.8, 44.9)),),)}
    cas = [
        ((44.93, 4.9), mince), ((44.96, 4.9), mince), ((44.93, 5.01), mince),
        ((45.5, 5.5), troue), ((44.8, 4.8), troue), ((47.0, 5.0), troue), ((44.2, 4.7), troue),
        ((2, 1.5), en_u), ((2, 0.5), en_u), ((1, 0.5), en_u), ((0.5, 1.5), en_u), ((3.5, 0.5), en_u),
        ((1, 0.5), losange), ((1, 1.5), losange), ((1, -0.5), losange), ((1, 2.5), losange), ((1.5, 1.0), losange),
        ((2, 1), en_l), ((2, 5), en_l), ((2, -1), en_l), ((1, 3), en_l), ((3, 3), en_l),
        ((44.85, 4.85), deux), ((45.05, 5.05), deux), ((44.95, 4.95), deux),
        ((44.93, 4.9), altitude), ((44.97, 4.9), altitude),
        ((44.92, 4.9), tuples), ((44.99, 4.85), tuples),
        ([44.93, 4.9], mince),
    ]
    fibre = zone_fibre_exemple()
    for fiche in list(clients_exemple().values())[:400]:
        cas.append((fiche["position"], fibre))
    return cas


def verifier_dans_zone(f) -> None:
    cas = _cas_261()
    obtenus = []
    for point, zone in cas:
        quoi = f"dans_zone({point!r}, <{zone['type']} {_court(zone['coordinates'], 70)}>)"
        copie = repr(zone)
        r = _appel(f, (point, zone), quoi)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, bool):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) : attendu True ou False (un "
                                 "bool : `bool(...)` si besoin).")
        if repr(zone) != copie:
            raise AssertionError(f"{quoi} a modifié la zone reçue : elle appartient au service SIG.")
        obtenus.append(r)
    for (point, zone), r in zip(cas, obtenus):
        attendu = _ref_dans_zone(point, zone)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_dans_zone(p, z, bug) for p, z in cas], _BUGS_ZONE,
                                   lambda a, b: a == b)
            raise AssertionError(f"dans_zone({point!r}, {_court(zone, 200)}) renvoie {r} au lieu de {attendu}. "
                                 f"{conseil}".rstrip())

    carre = _carre(4.0, 44.0, 6.0, 46.0)
    for zone, attendu, conseil in (
            ([carre], TypeError, "La zone est une géométrie GeoJSON : un dict {\"type\": …, \"coordinates\": …}."),
            ({"type": "Point", "coordinates": [4.9, 44.9]}, ValueError, "Un « Point » n'est pas une zone : seuls "
                                                                         "« Polygon » et « MultiPolygon » le sont."),
            ({"type": "polygon", "coordinates": [carre]}, ValueError, "« polygon » n'est pas « Polygon » (GeoJSON "
                                                                       "distingue les majuscules)."),
            ({"type": "Polygon", "coordinates": [carre[:-1]]}, ValueError, "Un anneau GeoJSON est FERMÉ : sa dernière "
                                                                            "position répète la première."),
            ({"type": "Polygon", "coordinates": [[[4, 44], [6, 44], [4, 44]]]}, ValueError,
             "Un anneau a au moins 4 positions (3 sommets + la fermeture)."),
            ({"type": "Polygon", "coordinates": [[[4, 44], [6, 44], [6], [4, 44]]]}, ValueError,
             "[6] n'est pas une position [longitude, latitude]."),
            ({"type": "Polygon", "coordinates": []}, ValueError, "Un polygone a au moins un anneau (son bord)."),
            ({"type": "MultiPolygon", "coordinates": [[carre], [_carre(7, 47, 8, 48)[:-1]]]}, ValueError,
             "Le DEUXIÈME polygone a un anneau non fermé : vérifiez toute la zone, même si le point est déjà dans le "
             "premier (sinon l'erreur dépend du point testé, et passe vos essais).")):
        _doit_lever(f, ((45.0, 5.0), zone), f"dans_zone((45.0, 5.0), {_court(zone, 100)})", conseil, attendu)
    zone = {"type": "Polygon", "coordinates": [carre]}
    _refus_points(f, lambda p: ((p, zone), {}), "dans_zone(point, <un carré>)")


# ============================================================================ défi 262 : geohash


_ALPHABET_RFC4648 = "abcdefghijklmnopqrstuvwxyz234567"


def _ref_geohash(point, precision: int = 7, bug: str = "") -> str:
    lat, lon = point
    intervalles = {"lon": [-180.0, 180.0], "lat": [-90.0, 90.0]}
    valeurs = {"lon": lon, "lat": lat}
    axe = "lat" if bug == "latitude_dabord" else "lon"
    bits = []
    for _ in range(5 * precision):
        bas, haut = intervalles[axe]
        milieu = (bas + haut) / 2
        en_haut = valeurs[axe] > milieu if bug == "strict" else valeurs[axe] >= milieu
        bits.append(1 if en_haut else 0)
        intervalles[axe][0 if en_haut else 1] = milieu
        axe = "lat" if axe == "lon" else "lon"
    alphabet = _ALPHABET_RFC4648 if bug == "alphabet" else BASE32_GEOHASH
    texte = ""
    for k in range(0, len(bits), 5):
        groupe = bits[k:k + 5][::-1] if bug == "poids_faible" else bits[k:k + 5]
        texte += alphabet[int("".join(map(str, groupe)), 2)]
    return texte


_BUGS_GEOHASH = {
    "latitude_dabord": "Vous coupez d'abord la LATITUDE : un geohash commence par la LONGITUDE (bit 1 : longitude, bit "
                       "2 : latitude, bit 3 : longitude…).",
    "strict": "Un point pile au milieu d'un intervalle part en bas chez vous (`>`) : la règle est `>=` le milieu → bit 1, "
              "moitié haute. (0, 0) doit donner « s0000 », pas « 7zzzz ».",
    "alphabet": "Vous utilisez l'alphabet base32 de la RFC 4648 (celui de `base64.b32encode` : a à z puis 2 à 7). Le "
                "geohash a le sien : `BASE32_GEOHASH`, « 0123456789bcdefghjkmnpqrstuvwxyz » (sans a, i, l ni o).",
    "poids_faible": "Vos 5 bits sont lus à l'envers : le PREMIER bit calculé est celui de poids FORT "
                    "(`valeur = 2 * valeur + bit`, pas `valeur += bit << n`).",
}


def _cas_262() -> list[tuple]:
    cas = [((57.64911, 10.40744), 11), ((44.9330, 4.8920), None), ((44.9330, 4.8920), 5), ((0, 0), 5),
           ((0.0, 0.0), 12), ((-90, -180), 3), ((90, 180), 4), ((48.8566, 2.3522), 6), ((-33.8688, 151.2093), 9),
           ((44.9, 4.89), 1), ((45.0, 0.0), 6), ((22.5, 45.0), 8), ([44.93, 4.89], 7), ((-45.0, -90.0), 4),
           ((89.99999, -179.99999), 12), ((44.93, 4.89), 2)]
    alea = random.Random(262)
    for _ in range(30):
        cas.append(((round(alea.uniform(-90, 90), 6), round(alea.uniform(-180, 180), 6)), alea.randint(1, 12)))
    cas += [(p, 7) for p in generer_points(10, graine=2621)]
    return cas


def verifier_geohash(f) -> None:
    cas = _cas_262()
    obtenus = []
    for point, precision in cas:
        args = (point,) if precision is None else (point, precision)
        quoi = f"geohash({point!r}{'' if precision is None else f', {precision}'})"
        r = _appel(f, args, quoi)
        n = 7 if precision is None else precision
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, str):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) : attendu un texte (str).")
        if len(r) != n:
            raise AssertionError(f"{quoi} renvoie {r!r} ({len(r)} caractères) : attendu {n} caractères"
                                 + (" (par défaut, precision vaut 7)." if precision is None else
                                    " (`precision` compte des CARACTÈRES de 5 bits, pas des bits)."))
        obtenus.append(r)
    for (point, precision), r in zip(cas, obtenus):
        attendu = _ref_geohash(point, 7 if precision is None else precision)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_geohash(p, 7 if n is None else n, bug) for p, n in cas],
                                   _BUGS_GEOHASH, lambda a, b: a == b)
            raise AssertionError(f"geohash({point!r}{'' if precision is None else f', {precision}'}) renvoie {r!r} au "
                                 f"lieu de {attendu!r}. {conseil}".rstrip())

    for precision, attendu, conseil in ((0, ValueError, "Un geohash a au moins un caractère."),
                                        (13, ValueError, "Au-delà de 12 caractères, les float n'ont plus assez de "
                                                         "chiffres : la règle s'arrête à 12."),
                                        (-3, ValueError, "Une précision négative n'a pas de sens."),
                                        (7.0, TypeError, "7.0 est un float : `precision` est un int."),
                                        ("7", TypeError, "« 7 » est un texte : `precision` est un int."),
                                        (True, TypeError, "True n'est pas une précision (excluez bool).")):
        _doit_lever(f, ((44.93, 4.89), precision), f"geohash((44.93, 4.89), {precision!r})", conseil, attendu)
    _refus_points(f, lambda p: ((p, 7), {}), "geohash(point, 7)")


# ============================================================================ défi 263 : plus_proches


def _ref_plus_proches(clients: dict, antennes: dict, bug: str = "") -> dict:
    """{client: (antenne, km)} : par bande de latitude (rapide), ou la version boguée `bug`."""
    rangees = sorted((p[0], p[1], a) for a, p in antennes.items())
    latitudes = [r[0] for r in rangees]
    resultat = {}
    for client, (lat, lon) in clients.items():
        if bug == "premiere":
            k = min(range(len(rangees)), key=lambda i: (abs(rangees[i][0] - lat), rangees[i][2]))
            resultat[client] = (rangees[k][2], _ref_distance((lat, lon), rangees[k][:2]))
            continue
        if bug in ("pythagore", "egalite", "grille", "metres"):
            candidats = rangees
            if bug == "grille":
                case = (math.floor(lat / 0.02), math.floor(lon / 0.02))
                voisins = [r for r in rangees if abs(math.floor(r[0] / 0.02) - case[0]) <= 1
                           and abs(math.floor(r[1] / 0.02) - case[1]) <= 1]
                candidats = voisins or rangees
            if bug == "pythagore":
                lat_b, lon_b, ident = min(candidats, key=lambda r: ((r[0] - lat) ** 2 + (r[1] - lon) ** 2, r[2]))
                resultat[client] = (ident, _ref_distance((lat, lon), (lat_b, lon_b)))
            elif bug == "egalite":                # à égalité, le plus GRAND identifiant
                distances = [(_ref_distance((lat, lon), r[:2]), r[2]) for r in candidats]
                d = min(distance for distance, _ in distances)
                resultat[client] = (max(ident for distance, ident in distances if distance == d), d)
            else:
                d, ident = min((_ref_distance((lat, lon), r[:2]), r[2]) for r in candidats)
                resultat[client] = (ident, 1000 * d if bug == "metres" else d)
            continue
        haut = bisect.bisect_left(latitudes, lat)
        bas, meilleur = haut - 1, None
        while bas >= 0 or haut < len(rangees):
            ecart_bas = lat - latitudes[bas] if bas >= 0 else math.inf
            ecart_haut = latitudes[haut] - lat if haut < len(rangees) else math.inf
            if ecart_bas <= ecart_haut:
                k, ecart, bas = bas, ecart_bas, bas - 1
            else:
                k, ecart, haut = haut, ecart_haut, haut + 1
            if meilleur is not None and RAYON_TERRE_KM * math.radians(ecart) > meilleur[0] + 1e-9:
                break
            candidat = (_ref_distance((lat, lon), rangees[k][:2]), rangees[k][2])
            if meilleur is None or candidat < meilleur:
                meilleur = candidat
        resultat[client] = (meilleur[1], meilleur[0])
    return resultat


_BUGS_PROCHES = {
    "premiere": "Vous gardez l'antenne la plus proche EN LATITUDE (la première trouvée après `bisect`) : une antenne à "
                "la même latitude peut être à 5 km à l'est, et une autre à 3 km au nord. Continuez des deux côtés tant "
                "que l'écart de latitude SEUL (en km) ne dépasse pas la meilleure distance trouvée.",
    "pythagore": "Vous choisissez l'antenne avec Pythagore sur les degrés : 1° de longitude ne vaut que 79 km dans la "
                 "vallée, contre 111 km pour 1° de latitude, et vous préférez à tort les antennes au nord ou au sud. "
                 "Comparez des `distance_km`.",
    "egalite": "Deux antennes sur le même pylône (même position, même distance) : vous gardez la mauvaise. La règle "
               "garde le plus PETIT identifiant : comparez des tuples (distance, identifiant), et ne remplacez la "
               "meilleure que si le candidat est strictement plus petit.",
    "grille": "Vous ne regardez que les cases voisines d'une grille : l'antenne la plus proche peut être deux cases "
              "plus loin, quand une antenne d'une case voisine est dans un coin éloigné. Une grille doit s'élargir "
              "jusqu'à ce que la distance au bord des cases non vues dépasse la meilleure distance trouvée.",
    "metres": "Vos distances sont en mètres : la règle les veut en km (celles de `distance_km`).",
}


def _cas_263() -> list[tuple[dict, dict]]:
    cas = [
        ({"C1": (45.0, 5.0), "C2": (45.04, 5.0)},
         {"A2": (45.0, 5.06), "A7": (45.03, 5.0), "A3": (45.03, 5.0), "A9": (45.3, 5.0)}),
        ({"C": (45.0, 5.0)}, {"P": (45.0, 5.025), "Q": (45.022, 5.0)}),                     # Pythagore se trompe
        ({"C": (45.0, 5.0)}, {"E": (45.0005, 5.08), "N": (45.04, 5.0), "S": (44.95, 5.01)}),  # pas la 1re trouvée
        ({"C": (45.0005, 5.0005)}, {"LOIN": (45.039, 5.039), "PRES": (45.0005, 5.045)}),      # deux cases plus loin
        ({"X": (44.93, 4.89), "Y": (44.93, 4.89), "Z": (44.95, 4.95)},
         {"A0277": (44.94, 4.9), "A0236": (44.94, 4.9), "A0300": (44.95, 4.95)}),             # même pylône ; distance 0
        ({}, {"A": (45.0, 5.0)}),
        ({"seul": [44.9, 4.9]}, {"A": [45.2, 5.3]}),
        ({7: (44.5, 4.6), 3: (45.1, 5.4)}, {"B": (44.9, 4.9), "A": (44.6, 4.7), "C": (45.0, 5.3)}),
    ]
    for graine in range(4):
        antennes = {f"T{k:03d}": p for k, p in enumerate(generer_points(120, graine=2630 + graine))}
        for k in range(0, 120, 17):                                      # quelques pylônes partagés
            antennes[f"U{k:03d}"] = antennes[f"T{k:03d}"]
        clients = {f"K{k:03d}": p for k, p in enumerate(generer_points(150, graine=2640 + graine))}
        clients.update({f"P{k:03d}": antennes[f"T{k:03d}"] for k in range(0, 120, 17)})
        cas.append((clients, antennes))
    return cas


def _verifier_forme_proches(r, clients: dict, antennes: dict, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie {type(r).__name__} : attendu un dict {{client: (antenne, km)}}.")
    if list(r) != list(clients):
        manque = [c for c in clients if c not in r]
        raise AssertionError(f"{quoi} a pour clés {_court(list(r), 100)} : attendu une clé par client, dans l'ordre "
                             "de `clients`" + (f" (il manque {_court(manque, 60)})." if manque else "."))
    for client, valeur in r.items():
        if not isinstance(valeur, tuple) or len(valeur) != 2:
            raise AssertionError(f"{quoi} : [{client!r}] vaut {_court(valeur, 80)} : attendu un TUPLE (antenne, "
                                 "distance en km).")
        if valeur[0] not in antennes:
            raise AssertionError(f"{quoi} : [{client!r}] vaut {_court(valeur, 80)} : {valeur[0]!r} n'est pas un "
                                 "identifiant d'antenne (le tuple est (antenne, distance), dans cet ordre).")
        if not _nombre(valeur[1]):
            raise AssertionError(f"{quoi} : [{client!r}] vaut {_court(valeur, 80)} : la distance est un nombre.")


def _proches_egaux(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[k][0] == b[k][0] and _proche(a[k][1], b[k][1]) for k in b)


def _lignes_proches(n_clients: int, n_antennes: int) -> int:
    """Le corrigé : 480 000 lignes pour 200 clients et 20 000 antennes (dont 200 000 pour vérifier les antennes)."""
    return 1200 * n_clients + 40 * n_antennes + 50_000


def verifier_plus_proches(f) -> None:
    with _DistancePretee(f):
        _verifier_plus_proches(f)


def _verifier_plus_proches(f) -> None:
    cas = _cas_263()
    obtenus = []
    for clients, antennes in cas:
        quoi = f"plus_proches({_court(clients, 70)}, {_court(antennes, 90)})"
        copies = (dict(clients), dict(antennes))
        r = _appel(f, (clients, antennes), quoi)
        _verifier_forme_proches(r, clients, antennes, quoi)
        if (clients, antennes) != copies:
            raise AssertionError(f"{quoi} a modifié ses arguments : clients et antennes ne vous appartiennent pas.")
        obtenus.append(r)
    for (clients, antennes), r in zip(cas, obtenus):
        attendu = _ref_plus_proches(clients, antennes)
        if not _proches_egaux(r, attendu):
            conseil = _premier_bug(obtenus, lambda bug: [_ref_plus_proches(c, a, bug) for c, a in cas], _BUGS_PROCHES,
                                   _listes(_proches_egaux))
            client = next(k for k in attendu if r[k][0] != attendu[k][0] or not _proche(r[k][1], attendu[k][1]))
            raise AssertionError(f"plus_proches : le client {client!r} en {clients[client]!r} reçoit {r[client]!r} au "
                                 f"lieu de {attendu[client]!r} (antennes : {_court(antennes, 150)}). {conseil}".rstrip())

    # ------------------------------------------------------------------ de quoi compter, puis toute la vallée
    clients = {f"K{k:03d}": p for k, p in enumerate(generer_points(200, graine=2650))}
    antennes = {f"T{k:05d}": p for k, p in enumerate(generer_points(20_000, graine=2651))}
    r = _appel(f, (clients, antennes), "plus_proches(200 clients, 20 000 antennes)",
               lignes_max=_lignes_proches(len(clients), len(antennes)),
               conseil_lent="200 clients × 20 000 antennes, c'est 4 millions de distances. Rangez les antennes par "
                            "latitude UNE fois, placez chaque client avec `bisect`, éloignez-vous des deux côtés et "
                            "arrêtez-vous dès que RAYON_TERRE_KM × math.radians(écart de latitude) dépasse la meilleure "
                            "distance : il ne reste qu'une centaine d'antennes à essayer par client.")
    _verifier_forme_proches(r, clients, antennes, "plus_proches(200 clients, 20 000 antennes)")
    attendu = _ref_plus_proches(clients, antennes)
    if not _proches_egaux(r, attendu):
        client = next(k for k in attendu if r[k][0] != attendu[k][0] or not _proche(r[k][1], attendu[k][1]))
        raise AssertionError(f"Sur 200 clients et 20 000 antennes, le client {client!r} reçoit {r[client]!r} au lieu "
                             f"de {attendu[client]!r} : votre recherche s'arrête-t-elle trop tôt ?")
    clients = {c: fiche["position"] for c, fiche in clients_exemple().items()}
    antennes = antennes_exemple()
    r = _appel(f, (clients, antennes), "plus_proches(les 6 000 clients de l'exemple, les 280 antennes)",
               lignes_max=_lignes_proches(len(clients), len(antennes)))
    _verifier_forme_proches(r, clients, antennes, "plus_proches(les clients de l'exemple)")
    attendu = _ref_plus_proches(clients, antennes)
    if not _proches_egaux(r, attendu):
        client = next(k for k in attendu if r[k][0] != attendu[k][0] or not _proche(r[k][1], attendu[k][1]))
        conseil = _premier_bug(r, lambda bug: _ref_plus_proches(clients, antennes, bug), _BUGS_PROCHES, _proches_egaux)
        raise AssertionError(f"Sur l'exemple, le client {client!r} reçoit {r[client]!r} au lieu de {attendu[client]!r}. "
                             f"{conseil}".rstrip())

    # ------------------------------------------------------------------ ce qu'il faut refuser
    for args, attendu, conseil in (
            (({"C": (45.0, 5.0)}, {}), ValueError, "Sans antenne, personne n'a d'antenne la plus proche : ValueError "
                                                    "(pas un None ni une distance infinie, qui se propageraient)."),
            (([(45.0, 5.0)], {"A": (45.0, 5.0)}), TypeError, "`clients` est un dict {client: (lat, lon)}."),
            (({"C": (45.0, 5.0)}, [(45.0, 5.0)]), TypeError, "`antennes` est un dict {antenne: (lat, lon)}."),
            (({"C": (45.0, 5.0)}, {"A": (95.0, 5.0)}), ValueError, "Une antenne à 95° de latitude n'existe pas."),
            (({"C": ("45", 5.0)}, {"A": (45.0, 5.0)}), TypeError, "« 45 » est un texte, pas une latitude."),
            (({"C": (45.0, float("nan"))}, {"A": (45.0, 5.0)}), ValueError, "NaN n'est pas une longitude.")):
        _doit_lever(f, args, f"plus_proches({_court(args[0], 50)}, {_court(args[1], 50)})", conseil, attendu)


# ============================================================================ défi 264 : regrouper_pannes


def _voisines(points: list, rayon: float) -> list[list[int]]:
    ordre = sorted(range(len(points)), key=lambda k: points[k][0])
    latitudes = [points[k][0] for k in ordre]
    bande = math.degrees(rayon / RAYON_TERRE_KM) + 1e-9
    voisines = []
    for lat, lon in points:
        debut, fin = bisect.bisect_left(latitudes, lat - bande), bisect.bisect_right(latitudes, lat + bande)
        voisines.append(sorted(ordre[k] for k in range(debut, fin)
                               if _ref_distance((lat, lon), points[ordre[k]]) <= rayon))
    return voisines


def _ref_dbscan(points, rayon: float, minimum: int, bug: str = "") -> list[int]:
    points = [tuple(map(float, p)) for p in points]
    n = len(points)
    voisines = _voisines(points, rayon)
    if bug == "sans_soi":
        voisines = [[k for k in v if k != i] for i, v in enumerate(voisines)]
    centrale = [len(v) >= minimum for v in voisines]
    etiquettes, visite = [-1] * n, [False] * n
    groupe = 1 if bug == "a_partir_de_un" else 0
    for depart in range(n):
        if bug == "bruit_definitif":
            if visite[depart]:
                continue
            visite[depart] = True
            if not centrale[depart]:
                continue
        elif etiquettes[depart] != -1 or not centrale[depart]:
            continue
        etiquettes[depart] = groupe
        file = deque([depart])
        while file:
            for k in voisines[file.popleft()]:
                if bug == "vol":
                    prend = etiquettes[k] != groupe
                elif bug == "bruit_definitif":
                    prend = not visite[k]
                else:
                    prend = etiquettes[k] == -1
                if prend:
                    etiquettes[k] = groupe
                    visite[k] = True
                    if centrale[k] or bug == "bord_etend":
                        file.append(k)
        groupe += 1
    return etiquettes


_BUGS_DBSCAN = {
    "sans_soi": "Une plainte ne se compte pas elle-même parmi ses voisines chez vous : la règle (celle de "
                "`min_samples` dans scikit-learn) la compte. `distance_km(p, p)` vaut 0 <= rayon_km : elle est dans sa "
                "propre liste de voisines.",
    "bord_etend": "Un point de BORD (pris, mais pas central) étend le foyer chez vous : le foyer avance alors de proche "
                  "en proche à travers des plaintes isolées, et finit par tout avaler. Seules les plaintes CENTRALES "
                  "entrent dans la file.",
    "vol": "Un deuxième foyer reprend chez vous une plainte de bord déjà prise par le premier : une plainte qui a un "
           "foyer le garde (`if etiquettes[k] == -1:`).",
    "bruit_definitif": "Une plainte jugée « isolée » au début du parcours le reste chez vous, même quand un foyer "
                       "l'atteint ensuite : « isolée », c'est seulement « pas encore de foyer ». Un foyer prend toute "
                       "voisine dont l'étiquette vaut encore -1 (ne gardez pas un ensemble des points « déjà vus »).",
    "a_partir_de_un": "Vos foyers sont numérotés à partir de 1 : la règle les numérote 0, 1, 2… (comme scikit-learn), "
                      "et garde -1 pour les plaintes isolées.",
}


def _le_long(*km: float, lon: float = 5.0) -> list[tuple[float, float]]:
    """Des points sur un méridien, à `km` km au nord de (45, lon) : leurs écarts sont des km exacts."""
    return [(45.0 + k / _KM_PAR_DEGRE, lon) for k in km]


def _cas_264() -> list[tuple]:
    cas = [
        (_le_long(0.0, 0.6, 1.2, 2.0, 9.0, 9.5, 10.0, 30.0), 0.7, 3),           # l'exemple
        (_le_long(0.0, 0.3, 0.6), 1.0, 3),                                       # chacune se compte elle-même
        (_le_long(0.0, 0.6, 1.2), 0.7, 3),                                       # une isolée d'abord, prise ensuite
        (_le_long(0.0, 0.2, 0.4, 0.6, 2.5, 2.3, 2.1, 1.9, 1.25), 0.7, 4),        # un bord entre deux foyers
        (_le_long(1.9, 2.1, 2.3, 2.5, 0.0, 0.2, 0.4, 0.6, 1.25), 0.7, 4),        # le même, dans l'autre ordre
        (_le_long(-0.15, -0.05, 0.05, 0.15, 0.6, 1.05, 1.5), 0.5, 4),            # un bord n'étend rien
        (_le_long(1.5, 1.05, 0.6, 0.15, 0.05, -0.05, -0.15), 0.5, 4),
        (_le_long(0.0, 5.0, 10.0), 1.0, 1),                                      # minimum = 1 : chacun son foyer
        ([], 1.0, 3),
        (tuple(_le_long(0.0, 0.1, 0.2, 7.0)), 0.5, 2),                           # un tuple de points
        ([[45.0, 5.0], [45.0, 5.005], [45.0, 5.01]], 0.5, 3),                    # des listes, le long d'un parallèle
    ]
    alea = random.Random(264)
    for graine in range(5):
        points = generer_points(120, graine=2660 + graine)
        for _ in range(3):
            centre = (alea.uniform(44.6, 45.0), alea.uniform(4.7, 5.3))
            points += generer_points(alea.randint(15, 40), graine=alea.randrange(10**6), centre=centre,
                                     ecart_km=alea.uniform(0.3, 1.0))
        alea.shuffle(points)
        cas.append((points, alea.choice((0.6, 0.8, 1.0, 1.5)), alea.randint(3, 8)))
    cas.append((plaintes_exemple(), 1.0, 10))
    cas.append((plaintes_exemple(), 2.0, 25))
    return cas


_CONSEIL_PAIRES = ("Comparer chaque plainte à toutes les autres fait n² paires (16 millions pour 4 000 plaintes). Rangez "
                   "les plaintes par latitude, et ne cherchez les voisines que dans la bande de `math.degrees(rayon_km / "
                   "RAYON_TERRE_KM)` degrés de chaque côté (`bisect`), comme au défi 263.")


def _lignes_dbscan(n: int) -> int:
    """Le corrigé : 756 000 lignes pour les 4 000 plaintes (une bande de latitude d'une centaine de plaintes chacune)."""
    return 800 * n + 50_000


_LIGNES_CHAINE = 350_000        # le corrigé : 111 000 ; toutes les paires : 9 millions


def verifier_regrouper_pannes(f) -> None:
    with _DistancePretee(f):
        _verifier_regrouper_pannes(f)


def _verifier_dbscan_cas(f, cas: list) -> None:
    """Appelle f sur chaque cas, vérifie la forme, puis compare à la référence (et nomme le bug s'il le reconnaît)."""
    obtenus = []
    for points, rayon, minimum in cas:
        quoi = f"regrouper_pannes(<{len(points)} plaintes>, {rayon!r}, {minimum!r})"
        copie = list(points)
        r = _appel(f, (points, rayon, minimum), quoi,
                   conseil="DBSCAN s'écrit sans récursion : une file (`collections.deque`) des plaintes centrales à "
                           "étendre.")
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, list) or len(r) != len(points) or not all(type(e) is int for e in r):
            raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : attendu une LISTE de {len(points)} int (une "
                                 "étiquette par plainte, dans l'ordre).")
        if list(points) != copie:
            raise AssertionError(f"{quoi} a modifié la liste des plaintes reçue.")
        obtenus.append(r)
    for (points, rayon, minimum), r in zip(cas, obtenus):
        attendu = _ref_dbscan(points, rayon, minimum)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_dbscan(p, ra, m, bug) for p, ra, m in cas], _BUGS_DBSCAN,
                                   lambda a, b: a == b)
            if not conseil and sorted(Counter(r).values()) != sorted(Counter(attendu).values()):
                conseil = ("Vos foyers ne sont pas les bons : relisez qui est CENTRALE (au moins `minimum` voisines à "
                           "`rayon_km` ou moins, elle comprise) et qui rejoint un foyer.")
            elif not conseil:
                conseil = ("Vos foyers ont les bonnes tailles mais pas les bons numéros : ils se numérotent dans l'ordre "
                           "où on les OUVRE, en parcourant les plaintes dans l'ordre de la liste.")
            montre = _court(list(points), 200) if len(points) <= 12 else f"<{len(points)} plaintes>"
            raise AssertionError(f"regrouper_pannes({montre}, {rayon!r}, {minimum!r}) renvoie {_court(r, 120)} au lieu "
                                 f"de {_court(attendu, 120)}. {conseil}".rstrip())


def _verifier_regrouper_pannes(f) -> None:
    cas = _cas_264()
    _verifier_dbscan_cas(f, [c for c in cas if len(c[0]) < 400])

    # ------------------------------------------------------------------ une chaîne de 3 000 plaintes
    chaine = _le_long(*[0.5 * k for k in range(3000)])
    r = _appel(f, (chaine, 0.7, 3), "regrouper_pannes(<3 000 plaintes en chaîne, à 0,5 km les unes des autres>, 0.7, 3)",
               conseil="3 000 plaintes en chaîne font 3 000 appels récursifs imbriqués : Python s'arrête à 1 000. "
                       "Étendez le foyer avec une file (`collections.deque`) : `while file:` et `file.popleft()`.",
               lignes_max=_LIGNES_CHAINE, conseil_lent=_CONSEIL_PAIRES)
    if r != [0] * len(chaine):
        raise AssertionError("regrouper_pannes(<3 000 plaintes en chaîne, à 0,5 km>, 0.7, 3) : toutes forment UN foyer "
                             f"(chacune a ses deux voisines), attendu [0, 0, …, 0], obtenu {_court(r, 80)}.")

    # ------------------------------------------------------------------ 4 000 plaintes : de quoi compter
    points = generer_points(3200, graine=2670)
    for k, centre in enumerate(generer_points(4, graine=2671)):
        points += generer_points(200, graine=2672 + k, centre=centre, ecart_km=0.8)
    random.Random(2676).shuffle(points)
    r = _appel(f, (points, 1.0, 10), "regrouper_pannes(<4 000 plaintes>, 1.0, 10)", lignes_max=_lignes_dbscan(len(points)),
               conseil_lent=_CONSEIL_PAIRES)
    if r != _ref_dbscan(points, 1.0, 10):
        raise AssertionError("Sur 4 000 plaintes, regrouper_pannes(…, 1.0, 10) ne trouve pas les bons foyers : votre "
                             "bande de latitude est-elle assez large (en DEGRÉS : math.degrees(rayon_km / RAYON_TERRE_KM)) ?")

    # ------------------------------------------------------------------ les plaintes de la vallée
    _verifier_dbscan_cas(f, [c for c in cas if len(c[0]) >= 400])

    # ------------------------------------------------------------------ ce qu'il faut refuser
    base = _le_long(0.0, 0.3)
    for args, attendu, conseil in (
            ((base, 0, 3), ValueError, "Un rayon de 0 km ne réunit personne : rayon_km > 0."),
            ((base, -1.0, 3), ValueError, "Un rayon négatif n'a pas de sens."),
            ((base, float("nan"), 3), ValueError, "NaN n'est pas un rayon : `nan <= 0` est FAUX, testez-le."),
            ((base, "1", 3), TypeError, "« 1 » est un texte : rayon_km est un nombre."),
            ((base, True, 3), TypeError, "True n'est pas un rayon (excluez bool)."),
            ((base, 1.0, 0), ValueError, "minimum = 0 : tout le monde serait central, même sans voisine."),
            ((base, 1.0, 2.5), TypeError, "minimum est un int (un nombre de plaintes)."),
            ((base, 1.0, True), TypeError, "True n'est pas un minimum (excluez bool)."),
            ((base + [(95.0, 5.0)], 1.0, 3), ValueError, "Une plainte à 95° de latitude n'existe pas."),
            ((base + [("45", 5.0)], 1.0, 3), TypeError, "« 45 » est un texte, pas une latitude.")):
        _doit_lever(f, args, f"regrouper_pannes({_court(args[0], 60)}, {args[1]!r}, {args[2]!r})", conseil, attendu)


# ============================================================================ défi 265 : tournee


def _longueur(depart, sites: dict, ordre: list, retour: bool = True) -> float:
    points = [depart] + [sites[s] for s in ordre] + ([depart] if retour else [])
    return math.fsum(_ref_distance(a, b) for a, b in zip(points, points[1:]))


def _ref_voisin(depart, sites: dict) -> list:
    """L'ordre du plus proche voisin (à distance égale, le plus petit identifiant)."""
    ordre, ici, restants = [], depart, set(sites)
    while restants:
        suivant = min(restants, key=lambda s: (_ref_distance(ici, sites[s]), s))
        ordre.append(suivant)
        restants.remove(suivant)
        ici = sites[suivant]
    return ordre


def _meilleur_echange(depart, sites: dict, ordre: list, sans_depot: bool = False) -> tuple[float, int, int]:
    """(gain en km, i, j) du meilleur échange 2-opt sur la tournée dépôt → ordre → dépôt (0.0 s'il n'y en a pas).

    `sans_depot` : seulement les échanges qui ne touchent aucune des deux arêtes du dépôt.
    """
    points = [depart] + [sites[s] for s in ordre]
    n = len(points)
    meilleur = (0.0, 0, 0)
    for i in range(2 if sans_depot else 1, n - 1):
        for j in range(i + 1, n - 1 if sans_depot else n):
            a, b, c, d = points[i - 1], points[i], points[j], points[(j + 1) % n]
            gain = (_ref_distance(a, b) + _ref_distance(c, d)) - (_ref_distance(a, c) + _ref_distance(b, d))
            if gain > meilleur[0]:
                meilleur = (gain, i, j)
    return meilleur


def _ref_tournee(depart, sites: dict) -> tuple[list, float]:
    """UNE bonne réponse (le plus proche voisin, puis 2-opt) : le correcteur, lui, vérifie des propriétés."""
    ordre = _ref_voisin(depart, sites)
    while True:
        gain, i, j = _meilleur_echange(depart, sites, ordre)
        if gain <= 1e-9:
            return ordre, _longueur(depart, sites, ordre)
        ordre[i - 1:j] = ordre[i - 1:j][::-1]


def _cas_265() -> list[tuple]:
    croix = {"N": (45.1, 5.0), "E": (45.0, 5.1), "S": (44.9, 5.0), "O": (45.0, 4.9)}
    cas = [
        ((45.0, 5.0), croix),
        ((45.0, 5.0), {}),
        ((45.0, 5.0), {"A": (45.0, 5.1)}),
        ((45.0, 5.0), {"A": (45.0, 5.1), "B": (45.1, 5.0)}),
        ([44.9, 4.9], {"P1": [45.0, 5.0], "P2": [45.0, 5.0], "P3": [44.8, 5.2]}),              # deux sites au même endroit
        ((45.0, 5.0), {"A": (45.0, 5.05), "B": (45.02, 5.3), "C": (44.98, 5.3), "D": (45.0, 5.25)}),
        (DEPOT, sites_exemple()),
    ]
    alea = random.Random(265)
    for k in range(10):
        n = alea.randint(5, 30)
        sites = {f"S{i:02d}": p for i, p in enumerate(generer_points(n, graine=2680 + k))}
        cas.append((generer_points(1, graine=2690 + k)[0], sites))
    return cas


def _verifier_une_tournee(f, depart, sites: dict, quoi: str, lignes_max: int | None = None) -> None:
    copie = dict(sites)
    r = _appel(f, (depart, sites), quoi, lignes_max=lignes_max,
               conseil_lent="Calculez chaque distance UNE fois (une table n × n), et le gain d'un échange 2-opt avec ses "
                            "QUATRE distances : recalculer toute la tournée pour chaque essai est n fois plus lent. Et "
                            "n'échangez que si le gain dépasse 1e-9 : sans cette marge, un arrondi peut faire défaire et "
                            "refaire le même échange sans fin.")
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, tuple) or len(r) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : attendu un tuple (ordre, longueur en km).")
    ordre, longueur = r
    if sites != copie or list(sites) != list(copie):
        raise AssertionError(f"{quoi} a modifié le dict `sites` reçu.")
    if not isinstance(ordre, list):
        raise AssertionError(f"{quoi} : l'ordre vaut {_court(ordre, 80)} : attendu une LISTE d'identifiants.")
    if not _nombre(longueur):
        raise AssertionError(f"{quoi} : la longueur vaut {_court(longueur, 60)} : attendu un nombre de km.")
    if sorted(map(repr, ordre)) != sorted(map(repr, sites)) or len(ordre) != len(sites):
        absents = [s for s in sites if s not in ordre]
        doubles = sorted({repr(s) for s in ordre if ordre.count(s) > 1})
        raise AssertionError(f"{quoi} : l'ordre {_court(ordre, 120)} doit contenir chaque site UNE fois, et rien d'autre "
                             f"(pas le dépôt)" + (f" ; il manque {_court(absents, 60)}" if absents else "")
                             + (f" ; en double : {', '.join(doubles)}" if doubles else "") + ".")
    vraie = _longueur(depart, sites, ordre)
    if not _proche(longueur, vraie):
        if ordre and _proche(longueur, _longueur(depart, sites, ordre, retour=False)):
            raise AssertionError(f"{quoi} annonce {longueur!r} km, la longueur de votre ordre SANS le retour au dépôt : "
                                 f"le technicien rentre le soir, la tournée fait {vraie!r} km.")
        raise AssertionError(f"{quoi} annonce {longueur!r} km, mais l'ordre renvoyé ({_court(ordre, 80)}) fait {vraie!r} "
                             "km (retour compris). Si vous mettez la longueur à jour échange après échange, une erreur "
                             "s'y est glissée : recalculez-la à la fin, sur l'ordre final.")
    voisin = _ref_voisin(depart, sites)
    limite = _longueur(depart, sites, voisin)
    gain, i, j = _meilleur_echange(depart, sites, ordre)
    if gain > 1e-6:
        conseils = []
        if ordre == voisin:
            conseils.append("C'est la tournée du plus proche voisin : il faut ensuite l'améliorer par 2-opt, tant qu'un "
                            "échange raccourcit la tournée.")
        if _meilleur_echange(depart, sites, ordre, sans_depot=True)[0] <= 1e-6:
            conseils.append("Les seuls échanges qui raccourcissent encore touchent une arête qui part du DÉPÔT ou y "
                            "revient : ces arêtes-là peuvent aussi se croiser. Le dépôt est la position 0 de la route, "
                            "qui ne bouge pas ; essayez i de 1 à n - 2 et j de i + 1 à n - 1, avec d = route[(j + 1) % n].")
        elif ordre != voisin:
            conseils.append("Un passage de 2-opt ne suffit pas : après un échange, d'autres deviennent possibles. "
                            "Recommencez tant qu'un échange a raccourci la tournée.")
        conseil = " ".join(conseils)
        a = "le dépôt" if i == 1 else repr(ordre[i - 2])
        d = "le dépôt" if j == len(ordre) else repr(ordre[j])
        raise AssertionError(f"{quoi} : votre tournée {_court(ordre, 100)} se croise encore : remplacer les arêtes "
                             f"{a} → {ordre[i - 1]!r} et {ordre[j - 1]!r} → {d} par {a} → {ordre[j - 1]!r} et "
                             f"{ordre[i - 1]!r} → {d} (retourner le morceau entre les deux) la raccourcit de "
                             f"{1000 * gain:.1f} m. {conseil}")
    if longueur > limite + 1e-6:
        raise AssertionError(f"{quoi} : votre tournée fait {longueur:.3f} km, plus que celle du plus proche voisin "
                             f"({limite:.3f} km, {_court(voisin, 80)}). Partez du plus proche voisin (du dépôt, le site "
                             "non visité le plus proche ; à égalité, le plus petit identifiant), puis n'acceptez que les "
                             "échanges qui raccourcissent.")


_LIGNES_TOURNEE = 700_000       # le corrigé : 40 000 ; « le meilleur échange » à chaque tour : 200 000 ; recalculer : 1,6 million


def verifier_tournee(f) -> None:
    with _DistancePretee(f):
        _verifier_tournee(f)


def _verifier_tournee(f) -> None:
    for depart, sites in _cas_265():
        quoi = f"tournee({depart!r}, {_court(sites, 90)})"
        _verifier_une_tournee(f, depart, sites, quoi)

    sites = {f"S{k:02d}": p for k, p in enumerate(generer_points(80, graine=2699))}
    _verifier_une_tournee(f, DEPOT, sites, "tournee(DEPOT, <80 sites>)", lignes_max=_LIGNES_TOURNEE)

    for args, attendu, conseil in (
            (((45.0, 5.0), [(45.1, 5.0)]), TypeError, "`sites` est un dict {identifiant: (lat, lon)}."),
            (((95.0, 5.0), {"A": (45.1, 5.0)}), ValueError, "Un dépôt à 95° de latitude n'existe pas."),
            ((("45", 5.0), {"A": (45.1, 5.0)}), TypeError, "« 45 » est un texte, pas une latitude."),
            (((95.0, 5.0), {}), ValueError, "Même sans site à visiter, un dépôt à 95° de latitude est une erreur : "
                                            "vérifiez `depart` avant tout."),
            (((45.0, 5.0), {"A": (45.1, 5.0), "B": (45.1, 185.0)}), ValueError,
             "Un site à 185° de longitude n'existe pas.")):
        _doit_lever(f, args, f"tournee({_court(args[0], 40)}, {_court(args[1], 60)})", conseil, attendu)


# ---------------------------------------------------------------------------- la série

SERIE = Serie(
    code="43",
    titre="Où partent les clients ? Distance, zone, geohash, plus proche antenne, foyers de pannes, tournée",
    module_eleve="defis.serie_43",
    module_corrige="defis.corrige_serie_43",
    defis=(
        Defi(260, "distance_km", "🟢", "—", verifier_distance_km),
        Defi(261, "dans_zone", "🟡", "—", verifier_dans_zone),
        Defi(262, "geohash", "🟡", "—", verifier_geohash),
        Defi(263, "plus_proches", "🟡", "—", verifier_plus_proches),
        Defi(264, "regrouper_pannes", "🔴", "—", verifier_regrouper_pannes),
        Defi(265, "tournee", "🔴", "—", verifier_tournee),
    ),
)

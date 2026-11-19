"""Série 43 (où partent les clients ?) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 260`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (haversine en `atan2`, NumPy, un rayon lancé vers
  l'ouest, l'indice d'enroulement, un geohash fait d'une chaîne de bits, une grille qui s'élargit, une recherche
  vectorisée, DBSCAN avec une pile, 2-opt « meilleur échange »…) ;
- les solutions trop lentes (toutes les paires clients × antennes ou plainte × plainte, une récursion, une tournée
  recalculée à chaque essai) sont arrêtées et expliquées ;
- les références du correcteur tiennent face à des vérifications indépendantes : scikit-learn (`haversine_distances`,
  `DBSCAN`, `BallTree`), l'indice d'enroulement, le décodage des geohash, une recherche exhaustive, la tournée
  optimale trouvée en essayant tous les ordres ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (trois antennes muettes que retrouvent les geohash, DBSCAN et la recherche de la plus proche antenne ; 41 % de
  départs autour d'elles contre 7 % ailleurs ; 941 km de tournée au tableur, 384 au plus proche voisin, 354 après
  2-opt).
"""

from __future__ import annotations

import bisect
import doctest
import itertools
import textwrap
import types
import math
import random
import sys
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import defis.carte as carte  # noqa: E402
import defis.corrige_serie_43 as corrige  # noqa: E402
from defis.carte import (  # noqa: E402
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
from defis.checks_serie_43 import (  # noqa: E402
    _BUGS_DBSCAN,
    _BUGS_DISTANCE,
    _BUGS_GEOHASH,
    _BUGS_PROCHES,
    _BUGS_ZONE,
    SERIE,
    _cas_260,
    _cas_261,
    _cas_262,
    _cas_263,
    _cas_264,
    _cas_265,
    _DistancePretee,
    _le_long,
    _longueur,
    _meilleur_echange,
    _ref_dans_zone,
    _ref_dbscan,
    _ref_distance,
    _ref_geohash,
    _ref_lire_zone,
    _ref_plus_proches,
    _ref_tournee,
    _ref_voisin,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet, **autres):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}, **autres), DEFIS[nom])


def _echec(nom, objet, *morceaux):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message
    return resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


def _avec_les_gardes(nom, calcul):
    """Une version boguée qui garde les vérifications du corrigé (les erreurs attendues), pour isoler LE bug."""
    def fonction(*args, **kwargs):
        getattr(corrige, nom)(*args, **kwargs)        # lève les mêmes erreurs que le corrigé
        return calcul(*args, **kwargs)
    fonction.__name__ = nom
    return fonction


def _haversine(a, b):
    return corrige.distance_km(a, b)


# --------------------------------------------------------------------------- Défi 260 : distance_km


@pytest.mark.parametrize("bug,morceau", [
    ("degres", "RADIANS"),
    ("pythagore", "Pythagore"),
    ("equirectangulaire", "antiméridien"),
    ("cosinus", "loi des cosinus"),
    ("un_moins_cos", "1 - cos x"),
    ("sans_deux", "facteur 2"),
    ("equateur", "6 378"),
    ("inverse", "LATITUDE"),
])
def test_distance_nomme_l_erreur(bug, morceau):
    _echec("distance_km", _avec_les_gardes("distance_km", lambda a, b: _ref_distance(tuple(a), tuple(b), bug)), morceau)


def test_distance_sans_min_plante_aux_antipodes():
    def distance_km(a, b):                            # la variante en atan2, sans borner h
        corrige.distance_km(a, b)
        p1, p2 = math.radians(a[0]), math.radians(b[0])
        h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
        return 2 * RAYON_TERRE_KM * math.atan2(math.sqrt(h), math.sqrt(1 - h))

    _echec("distance_km", distance_km, "math domain error", "min(1.0, h)")


def test_distance_un_rayon_faux_est_reconnu():
    _echec("distance_km", _avec_les_gardes("distance_km", lambda a, b: 1000 * corrige.distance_km(a, b)), "× 1000")
    _echec("distance_km", _avec_les_gardes("distance_km", lambda a, b: corrige.distance_km(a, b) / 1.609344),
           "rayon (ou votre unité)")


def test_distance_d_autres_ecritures_justes():
    def en_atan2(a, b):
        corrige.distance_km(a, b)
        p1, p2 = math.radians(a[0]), math.radians(b[0])
        dp, dl = p2 - p1, math.radians(b[1]) - math.radians(a[1])
        h = min(1.0, math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
        return 2 * RAYON_TERRE_KM * math.atan2(math.sqrt(h), math.sqrt(1 - h))

    def en_numpy(a, b):
        corrige.distance_km(a, b)
        (p1, l1), (p2, l2) = np.radians(a), np.radians(b)
        h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
        return float(2 * RAYON_TERRE_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))

    _reussi("distance_km", en_atan2)
    _reussi("distance_km", en_numpy)


def test_distance_refuse_ce_qu_il_faut():
    def sans_verification(a, b):
        return corrige.distance_km(tuple(map(float, a)), tuple(map(float, b)))

    message = _echec("distance_km", sans_verification)
    assert "TypeError" in message or "ValueError" in message

    def nan_accepte(a, b):
        if any(isinstance(v, float) and math.isnan(v) for v in (*a, *b)):
            return float("nan")
        return corrige.distance_km(a, b)

    _echec("distance_km", nan_accepte, "NaN")

    def bool_accepte(a, b):
        return corrige.distance_km([float(v) for v in a], [float(v) for v in b]) if any(
            isinstance(v, bool) for v in (*a, *b)) else corrige.distance_km(a, b)

    _echec("distance_km", bool_accepte, "True")


def test_la_distance_de_reference_contre_scikit_learn():
    from sklearn.metrics.pairwise import haversine_distances

    alea = random.Random(0)
    for _ in range(300):
        a = (alea.uniform(-90, 90), alea.uniform(-180, 180))
        b = (alea.uniform(-90, 90), alea.uniform(-180, 180))
        attendu = RAYON_TERRE_KM * haversine_distances(np.radians([a]), np.radians([b]))[0, 0]
        assert math.isclose(_ref_distance(a, b), attendu, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(_ref_distance((0, 0), (0, 1)), math.pi * RAYON_TERRE_KM / 180)
    assert math.isclose(_ref_distance((0, 0), (90, 0)), math.pi * RAYON_TERRE_KM / 2)        # un quart de méridien
    assert _ref_distance((44.9, 4.89), (-44.9, -175.11)) == math.pi * RAYON_TERRE_KM


def test_les_antipodes_poussent_vraiment_h_au_dessus_de_1():
    """Sans ce cas, la règle `min(1.0, h)` ne serait pas vérifiée."""
    p1, p2 = math.radians(44.9), math.radians(-44.9)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(-175.11 - 4.89) / 2) ** 2
    assert h == 1.0000000000000002
    with pytest.raises(ValueError, match="math domain error"):
        math.sqrt(1 - h)
    with pytest.raises(ValueError, match="math domain error"):
        math.acos(1 - 2 * h)
    assert ((44.9, 4.89), (-44.9, -175.11)) in _cas_260()


# --------------------------------------------------------------------------- Défi 261 : dans_zone


@pytest.mark.parametrize("bug,morceau", [
    ("inverse", "[LONGITUDE"),
    ("sans_trous", "TROUS"),
    ("premier_polygone", "PREMIER polygone"),
    ("sommet", "DEUX fois"),
    ("boite", "RECTANGLE"),
])
def test_zone_nomme_l_erreur(bug, morceau):
    _echec("dans_zone", _avec_les_gardes("dans_zone", lambda p, z: _ref_dans_zone(tuple(p), z, bug)), morceau)


def test_zone_inclusive_qui_divise_par_zero_est_expliquee():
    def dans_zone(point, zone):
        corrige.dans_zone(point, zone)
        lat, lon = point
        anneaux = _ref_lire_zone(zone)
        resultat = False
        for polygone in anneaux:
            dedans = False
            for (x1, y1), (x2, y2) in zip(polygone[0], polygone[0][1:]):
                if min(y1, y2) <= lat <= max(y1, y2) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                    dedans = not dedans
            resultat = resultat or dedans
        return resultat

    _echec("dans_zone", dans_zone, "HORIZONTALE")


def _enroulement(lon, lat, anneau):
    """L'indice d'enroulement (Sunday) : une autre façon, indépendante, de savoir si un point est dedans."""
    tours = 0
    for (x1, y1), (x2, y2) in zip(anneau, anneau[1:]):
        cote = (x2 - x1) * (lat - y1) - (lon - x1) * (y2 - y1)
        if y1 <= lat < y2 and cote > 0:
            tours += 1
        elif y2 <= lat < y1 and cote < 0:
            tours -= 1
    return tours != 0


def _dans_zone_enroulement(point, zone):
    corrige.dans_zone(point, zone)
    lat, lon = point
    return any(_enroulement(lon, lat, p[0]) and not any(_enroulement(lon, lat, t) for t in p[1:])
               for p in _ref_lire_zone(zone))


def test_zone_d_autres_ecritures_justes():
    def vers_l_ouest(point, zone):
        corrige.dans_zone(point, zone)
        lat, lon = point

        def dedans(anneau):
            croisements = 0
            for (x1, y1), (x2, y2) in zip(anneau, anneau[1:]):
                if (y1 > lat) != (y2 > lat) and lon > x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                    croisements += 1
            return croisements % 2 == 1

        return any(dedans(p[0]) and not any(dedans(t) for t in p[1:]) for p in _ref_lire_zone(zone))

    _reussi("dans_zone", vers_l_ouest)
    _reussi("dans_zone", _dans_zone_enroulement)


def test_zone_refuse_ce_qu_il_faut():
    def sans_fermeture(point, zone):
        try:
            return corrige.dans_zone(point, zone)
        except ValueError as exc:
            if "non fermé" in str(exc):
                return False
            raise

    _echec("dans_zone", sans_fermeture, "FERMÉ")

    def premiere_erreur_seulement(point, zone):
        lat, lon = corrige._point(point)
        if not isinstance(zone, dict):
            raise TypeError("zone")
        polygones = [zone.get("coordinates")] if zone.get("type") == "Polygon" else zone.get("coordinates")
        if zone.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("type")
        for polygone in polygones:
            if not polygone:
                raise ValueError("vide")
            anneaux = [corrige._anneau(a) for a in polygone]
            if corrige._dans_anneau(lon, lat, anneaux[0]) and not any(corrige._dans_anneau(lon, lat, t)
                                                                        for t in anneaux[1:]):
                return True
        return False

    _echec("dans_zone", premiere_erreur_seulement, "DEUXIÈME polygone")


def test_la_zone_de_reference_contre_l_indice_d_enroulement():
    zone = zone_fibre_exemple()
    alea = random.Random(1)
    points = [(alea.uniform(44.85, 45.0), alea.uniform(4.82, 4.97)) for _ in range(3000)]
    points += [(alea.uniform(44.69, 44.77), alea.uniform(4.97, 5.08)) for _ in range(1000)]
    for point in points:
        assert _ref_dans_zone(point, zone) == _dans_zone_enroulement(point, zone)
    for point, zone_cas in _cas_261():
        assert _ref_dans_zone(point, zone_cas) == _dans_zone_enroulement(point, zone_cas), point


# --------------------------------------------------------------------------- Défi 262 : geohash


@pytest.mark.parametrize("bug,morceau", [
    ("latitude_dabord", "LONGITUDE"),
    ("strict", "`>=`"),
    ("alphabet", "RFC 4648"),
    ("poids_faible", "poids FORT"),
])
def test_geohash_nomme_l_erreur(bug, morceau):
    _echec("geohash", _avec_les_gardes("geohash", lambda p, n=7: _ref_geohash(tuple(p), n, bug)), morceau)


def test_geohash_precision_en_bits_est_expliquee():
    _echec("geohash", _avec_les_gardes("geohash", lambda p, n=7: corrige.geohash(p, max(1, n // 5) if n > 5 else n)),
           "CARACTÈRES de 5 bits")
    _echec("geohash", _avec_les_gardes("geohash", lambda p, n=5: corrige.geohash(p, n)), "precision vaut 7")


def test_geohash_d_autres_ecritures_justes():
    def en_chaine_de_bits(point, precision=7):
        corrige.geohash(point, precision)
        lat, lon = point
        bas = {"lat": -90.0, "lon": -180.0}
        haut = {"lat": 90.0, "lon": 180.0}
        valeur = {"lat": lat, "lon": lon}
        bits = ""
        for k in range(5 * precision):
            axe = "lon" if k % 2 == 0 else "lat"
            milieu = (bas[axe] + haut[axe]) / 2
            if valeur[axe] >= milieu:
                bits += "1"
                bas[axe] = milieu
            else:
                bits += "0"
                haut[axe] = milieu
        return "".join(BASE32_GEOHASH[int(bits[i:i + 5], 2)] for i in range(0, len(bits), 5))

    _reussi("geohash", en_chaine_de_bits)


def _decoder(code):
    """La case (lat_min, lat_max, lon_min, lon_max) d'un geohash."""
    lat, lon, longitude = [-90.0, 90.0], [-180.0, 180.0], True
    for caractere in code:
        valeur = BASE32_GEOHASH.index(caractere)
        for decalage in range(4, -1, -1):
            bit = (valeur >> decalage) & 1
            intervalle = lon if longitude else lat
            intervalle[1 - bit] = (intervalle[0] + intervalle[1]) / 2
            longitude = not longitude
    return lat[0], lat[1], lon[0], lon[1]


def test_le_geohash_de_reference_tombe_dans_sa_case():
    assert _ref_geohash((57.64911, 10.40744), 11) == "u4pruydqqvj"        # geohash.org
    assert _ref_geohash((42.6, -5.6), 5) == "ezs42"                       # l'exemple de Wikipédia
    alea = random.Random(2)
    for _ in range(2000):
        point = (alea.uniform(-90, 90), alea.uniform(-180, 180))
        code = _ref_geohash(point, alea.randint(1, 12))
        lat_min, lat_max, lon_min, lon_max = _decoder(code)
        assert lat_min <= point[0] <= lat_max and lon_min <= point[1] <= lon_max
        assert code.startswith(_ref_geohash(point, max(1, len(code) - 2)))     # un préfixe = une case plus grande


# --------------------------------------------------------------------------- Défi 263 : plus_proches


@pytest.mark.parametrize("bug,morceau", [
    ("premiere", "EN LATITUDE"),
    ("pythagore", "Pythagore"),
    ("egalite", "plus PETIT identifiant"),
    ("grille", "cases voisines"),
    ("metres", "mètres"),
])
def test_proches_nomme_l_erreur(bug, morceau):
    _echec("plus_proches", _avec_les_gardes("plus_proches", lambda c, a: _ref_plus_proches(
        {k: tuple(p) for k, p in c.items()}, {k: tuple(p) for k, p in a.items()}, bug)), morceau)


def test_chaque_bug_des_proches_a_son_petit_cas():
    """Chaque erreur se voit sur un des cinq petits cas : le correcteur la nomme sans attendre les grands."""
    petits = _cas_263()[:5]
    for bug in _BUGS_PROCHES:
        assert any(_ref_plus_proches(c, a, bug) != _ref_plus_proches(c, a) for c, a in petits), bug


def test_proches_tout_comparer_est_arrete():
    def plus_proches(clients, antennes):
        corrige.plus_proches({}, antennes)            # les mêmes erreurs pour les antennes
        if not isinstance(clients, dict):
            raise TypeError("dict")
        return {c: tuple(reversed(min((corrige.distance_km(p, q), a) for a, q in antennes.items())))
                for c, p in clients.items()}

    _echec("plus_proches", plus_proches, "4 millions", "bisect")


def test_proches_d_autres_ecritures_justes():
    def grille(clients, antennes):
        """Une grille de cases de 0,05°, élargie anneau par anneau tant qu'une case non vue peut faire mieux."""
        corrige.plus_proches({}, antennes) if antennes else None
        if not isinstance(clients, dict) or not isinstance(antennes, dict):
            raise TypeError("dict")
        if not antennes:
            raise ValueError("aucune antenne")
        taille = 0.05
        cases = {}
        for a, p in antennes.items():
            lat, lon = corrige._point(p)
            cases.setdefault((math.floor(lat / taille), math.floor(lon / taille)), []).append((lat, lon, a))
        resultat = {}
        for c, p in clients.items():
            lat, lon = corrige._point(p)
            ci, cj = math.floor(lat / taille), math.floor(lon / taille)
            meilleur, rang = None, 0
            while True:
                for i in range(ci - rang, ci + rang + 1):
                    for j in range(cj - rang, cj + rang + 1):
                        if max(abs(i - ci), abs(j - cj)) != rang:
                            continue
                        for la, lo, a in cases.get((i, j), ()):
                            candidat = (corrige.distance_km((lat, lon), (la, lo)), a)
                            if meilleur is None or candidat < meilleur:
                                meilleur = candidat
                # Toute case non vue est à plus de `rang` cases en latitude, ou en longitude : une borne sûre.
                borne = RAYON_TERRE_KM * math.radians(rang * taille) * min(
                    1.0, math.cos(math.radians(min(89.9, abs(lat) + (rang + 1) * taille))))
                if meilleur is not None and borne > meilleur[0] + 1e-9:
                    break
                rang += 1
            resultat[c] = (meilleur[1], meilleur[0])
        return resultat

    def vectorisee(clients, antennes):
        """NumPy : toutes les distances d'un client d'un coup (peu de lignes Python, beaucoup de calcul)."""
        if not isinstance(clients, dict) or not isinstance(antennes, dict):
            raise TypeError("dict")
        if not antennes:
            raise ValueError("aucune antenne")
        noms = sorted(antennes)
        pos = np.radians(np.array([corrige._point(antennes[a]) for a in noms]))
        resultat = {}
        for c, p in clients.items():
            lat, lon = np.radians(corrige._point(p))
            h = (np.sin((pos[:, 0] - lat) / 2) ** 2
                 + np.cos(lat) * np.cos(pos[:, 0]) * np.sin((pos[:, 1] - lon) / 2) ** 2)
            d = 2 * RAYON_TERRE_KM * np.arcsin(np.sqrt(np.minimum(1.0, h)))
            k = int(np.argmin(d))                      # le premier minimum : le plus petit identifiant
            resultat[c] = (noms[k], float(d[k]))
        return resultat

    _reussi("plus_proches", grille)
    _reussi("plus_proches", vectorisee)


def test_la_recherche_de_reference_est_exhaustive():
    """La bande de latitude ne rate rien : comparée à la recherche exhaustive (et à BallTree de scikit-learn)."""
    from sklearn.neighbors import BallTree

    antennes = antennes_exemple()
    clients = {c: f["position"] for c, f in list(clients_exemple().items())[:1500]}
    rapide = _ref_plus_proches(clients, antennes)
    for c, p in clients.items():
        d, a = min((_haversine(p, q), a) for a, q in antennes.items())
        assert rapide[c][0] == a and math.isclose(rapide[c][1], d, rel_tol=1e-12, abs_tol=1e-12)
    noms = sorted(set(antennes.values()))
    arbre = BallTree(np.radians(noms), metric="haversine")
    distances, _ = arbre.query(np.radians(list(clients.values())), k=1)
    for (c, (a, d)), attendu in zip(rapide.items(), distances[:, 0] * RAYON_TERRE_KM):
        assert math.isclose(d, attendu, rel_tol=1e-9)


def test_les_cas_n_ont_pas_de_quasi_egalite():
    """Hors des pylônes partagés (égalité EXACTE), aucune deuxième antenne à moins d'un nanomètre de la première :
    deux écritures justes de la distance ne peuvent pas choisir deux antennes différentes."""
    for clients, antennes in _cas_263():
        for p in clients.values():
            distances = sorted({_haversine(p, q) for q in antennes.values()})
            assert len(distances) < 2 or distances[1] - distances[0] > 1e-9


# --------------------------------------------------------------------------- Défi 264 : regrouper_pannes


@pytest.mark.parametrize("bug,morceau", [
    ("sans_soi", "elle-même"),
    ("bord_etend", "BORD"),
    ("vol", "reprend"),
    ("bruit_definitif", "pas encore de foyer"),
    ("a_partir_de_un", "à partir de 1"),
])
def test_dbscan_nomme_l_erreur(bug, morceau):
    _echec("regrouper_pannes", _avec_les_gardes("regrouper_pannes", lambda p, r, m: _ref_dbscan(p, r, m, bug)), morceau)


def _dbscan_pile(plaintes, rayon_km, minimum):
    """Une autre écriture juste : une PILE (parcours en profondeur) et une grille de cases pour les voisines."""
    corrige.regrouper_pannes([], rayon_km, minimum)
    points = [corrige._point(p) for p in plaintes]
    taille = math.degrees(rayon_km / RAYON_TERRE_KM)            # en latitude ; en longitude, on élargit
    cases = {}
    for k, (lat, lon) in enumerate(points):
        cases.setdefault(math.floor(lat / taille), []).append(k)
    voisines = []
    for lat, lon in points:
        i = math.floor(lat / taille)
        proches = [k for j in (i - 1, i, i + 1) for k in cases.get(j, ())]
        voisines.append([k for k in proches if corrige.distance_km((lat, lon), points[k]) <= rayon_km])
    etiquettes, groupe = [-1] * len(points), 0
    for depart in range(len(points)):
        if etiquettes[depart] != -1 or len(voisines[depart]) < minimum:
            continue
        etiquettes[depart] = groupe
        pile = [depart]
        while pile:
            for k in voisines[pile.pop()]:
                if etiquettes[k] == -1:
                    etiquettes[k] = groupe
                    if len(voisines[k]) >= minimum:
                        pile.append(k)
        groupe += 1
    return etiquettes


def test_dbscan_d_autres_ecritures_justes():
    _reussi("regrouper_pannes", _dbscan_pile)


def test_dbscan_recursif_est_explique():
    def regrouper_pannes(plaintes, rayon_km, minimum):
        corrige.regrouper_pannes([], rayon_km, minimum)
        points = [corrige._point(p) for p in plaintes]
        ordre = sorted(range(len(points)), key=lambda k: points[k][0])
        latitudes = [points[k][0] for k in ordre]
        bande = math.degrees(rayon_km / RAYON_TERRE_KM) + 1e-9
        voisines = [[ordre[k] for k in range(bisect.bisect_left(latitudes, p[0] - bande),
                                             bisect.bisect_right(latitudes, p[0] + bande))
                     if corrige.distance_km(p, points[ordre[k]]) <= rayon_km] for p in points]
        etiquettes = [-1] * len(points)

        def etendre(i, groupe):
            for k in voisines[i]:
                if etiquettes[k] == -1:
                    etiquettes[k] = groupe
                    if len(voisines[k]) >= minimum:
                        etendre(k, groupe)

        groupe = 0
        for i in range(len(points)):
            if etiquettes[i] == -1 and len(voisines[i]) >= minimum:
                etiquettes[i] = groupe
                etendre(i, groupe)
                groupe += 1
        return etiquettes

    _echec("regrouper_pannes", regrouper_pannes, "RecursionError", "deque")


def test_dbscan_toutes_les_paires_est_arrete():
    def regrouper_pannes(plaintes, rayon_km, minimum):
        corrige.regrouper_pannes([], rayon_km, minimum)
        points = [corrige._point(p) for p in plaintes]
        voisines = [[k for k, q in enumerate(points) if corrige.distance_km(p, q) <= rayon_km] for p in points]
        etiquettes, groupe = [-1] * len(points), 0
        for depart in range(len(points)):
            if etiquettes[depart] != -1 or len(voisines[depart]) < minimum:
                continue
            etiquettes[depart], file = groupe, deque([depart])
            while file:
                for k in voisines[file.popleft()]:
                    if etiquettes[k] == -1:
                        etiquettes[k] = groupe
                        if len(voisines[k]) >= minimum:
                            file.append(k)
            groupe += 1
        return etiquettes

    _echec("regrouper_pannes", regrouper_pannes, "beaucoup trop lent")


def test_le_dbscan_de_reference_contre_scikit_learn():
    from sklearn.cluster import DBSCAN

    jeux = [(plaintes_exemple(), 1.0, 10), (plaintes_exemple(), 2.0, 25)] + [c for c in _cas_264() if len(c[0]) > 50]
    for points, rayon, minimum in jeux:
        modele = DBSCAN(eps=rayon / RAYON_TERRE_KM, min_samples=minimum, metric="haversine", algorithm="ball_tree")
        attendu = modele.fit(np.radians(points)).labels_.tolist()
        assert _ref_dbscan(points, rayon, minimum) == attendu


def test_aucune_paire_n_est_pile_au_rayon():
    """Le correcteur évite les distances à un nanomètre du rayon : `<=` ne dépend pas de l'écriture de la distance."""
    for points, rayon, _ in _cas_264():
        for a, b in itertools.combinations(points[:300], 2):
            assert abs(_haversine(a, b) - rayon) > 1e-9


# --------------------------------------------------------------------------- Défi 265 : tournee


def _table(depart, sites):
    noms = list(sites)
    points = [corrige._point(depart)] + [corrige._point(sites[s]) for s in noms]
    return noms, points, [[corrige.distance_km(p, q) for q in points] for p in points]


def _plus_proche_voisin(table, noms):
    route, restants = [0], set(range(1, len(table)))
    while restants:
        k = min(restants, key=lambda k: (table[route[-1]][k], noms[k - 1]))
        route.append(k)
        restants.remove(k)
    return route


def _tournee_avec(deux_opt, longueur=None):
    """Une tournée : plus proche voisin, puis la variante `deux_opt(route, table)` ; `longueur(route, table)`."""
    def tournee(depart, sites):
        corrige.tournee(depart, sites)                 # les mêmes erreurs
        if not sites:
            return [], 0.0
        noms, _, table = _table(depart, sites)
        route = deux_opt(_plus_proche_voisin(table, noms), table)
        n = len(route)
        total = (longueur or (lambda r, t: sum(t[r[k]][r[(k + 1) % n]] for k in range(n))))(route, table)
        return [noms[k - 1] for k in route[1:]], total
    return tournee


def _passes(route, table, premier=1, dernier=None, une_seule=False):
    n = len(route)
    dernier = n if dernier is None else dernier
    ameliore = True
    while ameliore:
        ameliore = False
        for i in range(premier, n - 1):
            for j in range(i + 1, dernier):
                a, b, c, d = route[i - 1], route[i], route[j], route[(j + 1) % n]
                if table[a][b] + table[c][d] - table[a][c] - table[b][d] > 1e-9:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    ameliore = True
        if une_seule:
            break
    return route


def test_tournee_plus_proche_voisin_seul():
    _echec("tournee", _tournee_avec(lambda r, t: r), "plus proche voisin", "2-opt")


def test_tournee_une_seule_passe():
    _echec("tournee", _tournee_avec(lambda r, t: _passes(r, t, une_seule=True)), "Un passage de 2-opt ne suffit pas")


def test_tournee_sans_les_aretes_du_depot():
    _echec("tournee", _tournee_avec(lambda r, t: _passes(r, t, premier=2, dernier=len(r) - 1)), "DÉPÔT")


def test_tournee_sans_retour():
    n_retour = _tournee_avec(lambda r, t: _passes(r, t),
                             lambda r, t: sum(t[r[k]][r[k + 1]] for k in range(len(r) - 1)))
    _echec("tournee", n_retour, "SANS le retour")


def test_tournee_longueur_fausse():
    def longueur_du_voisin(route, table):
        voisin = _plus_proche_voisin(table, [str(k) for k in range(len(table) - 1)])
        n = len(voisin)
        return sum(table[voisin[k]][voisin[(k + 1) % n]] for k in range(n))

    _echec("tournee", _tournee_avec(lambda r, t: _passes(r, t), longueur_du_voisin), "recalculez-la à la fin")


def test_tournee_ordre_incomplet():
    def tournee(depart, sites):
        ordre, km = corrige.tournee(depart, sites)
        return (ordre[:-1] + ["DEPOT"] if len(ordre) > 2 else ordre), km

    _echec("tournee", tournee, "chaque site UNE fois")


def test_tournee_recalculee_a_chaque_essai_est_arretee():
    def deux_opt(route, table):
        n = len(route)

        def longueur(r):
            return sum(table[r[k]][r[(k + 1) % n]] for k in range(n))

        ameliore = True
        while ameliore:
            ameliore = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    nouvelle = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if longueur(nouvelle) < longueur(route) - 1e-9:
                        route, ameliore = nouvelle, True
        return route

    _echec("tournee", _tournee_avec(deux_opt), "QUATRE distances")


def test_tournee_d_autres_ecritures_justes():
    def meilleur_echange(route, table):
        n = len(route)
        while True:
            meilleur = (1e-9, 0, 0)
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    a, b, c, d = route[i - 1], route[i], route[j], route[(j + 1) % n]
                    gain = table[a][b] + table[c][d] - table[a][c] - table[b][d]
                    if gain > meilleur[0]:
                        meilleur = (gain, i, j)
            if meilleur[1] == 0:
                return route
            route[meilleur[1]:meilleur[2] + 1] = route[meilleur[1]:meilleur[2] + 1][::-1]

    _reussi("tournee", _tournee_avec(meilleur_echange))
    _reussi("tournee", _tournee_avec(lambda r, t: _passes(r, t)))


def test_tournee_plus_longue_que_le_voisin_est_refusee():
    """Partir de l'ordre du tableur et faire 2-opt donne un optimum local parfois plus long que le plus proche voisin."""
    def depuis_le_tableur(depart, sites):
        corrige.tournee(depart, sites)
        if not sites:
            return [], 0.0
        noms, _, table = _table(depart, sites)
        route = _passes(list(range(len(table))), table)
        n = len(route)
        return [noms[k - 1] for k in route[1:]], sum(table[route[k]][route[(k + 1) % n]] for k in range(n))

    resultat = _evaluer("tournee", depuis_le_tableur)
    assert resultat.statut in (ECHEC, REUSSI)
    if resultat.statut == ECHEC:
        assert "plus proche voisin" in resultat.message, resultat.message


def test_la_tournee_de_reference_est_un_optimum_local_et_proche_de_l_optimum():
    for depart, sites in _cas_265():
        ordre, longueur = _ref_tournee(depart, sites)
        assert sorted(ordre) == sorted(sites)
        assert math.isclose(longueur, _longueur(depart, sites, ordre))
        assert _meilleur_echange(depart, sites, ordre)[0] <= 1e-9
        assert longueur <= _longueur(depart, sites, _ref_voisin(depart, sites)) + 1e-9
    alea = random.Random(3)
    for k in range(6):                                        # 7 sites : on peut essayer les 5 040 ordres
        sites = {f"S{i}": p for i, p in enumerate(generer_points(7, graine=900 + k))}
        depart = (alea.uniform(44.5, 45.1), alea.uniform(4.6, 5.4))
        optimum = min(_longueur(depart, sites, list(o)) for o in itertools.permutations(sites))
        ordre, longueur = corrige.tournee(depart, sites)
        assert optimum - 1e-9 <= longueur <= 1.25 * optimum


# --------------------------------------------------------------------------- Le correcteur prête sa distance


def test_la_distance_pretee_est_rendue_meme_en_cas_d_echec():
    """Un défi 263 qui appelle un défi 260 encore « à faire » est quand même noté, et la distance de l'élève revient."""
    module = types.ModuleType("test_defis_serie_43_pret")
    exec(textwrap.dedent("""
        def distance_km(a, b):
            raise NotImplementedError

        def plus_proches(clients, antennes):
            return {c: ("faux", distance_km(p, p)) for c, p in clients.items()}
    """), module.__dict__)
    sys.modules[module.__name__] = module
    eleve = module.distance_km
    try:
        resultat = evaluer_defi(module, DEFIS["plus_proches"])
        assert resultat.statut == ECHEC and "identifiant d'antenne" in resultat.message, resultat.message
        assert module.distance_km is eleve
        with _DistancePretee(module.plus_proches):
            assert module.distance_km is not eleve
            assert module.distance_km((0, 0), (0, 1)) == _ref_distance((0, 0), (0, 1))
            with pytest.raises(ValueError):
                module.distance_km((91, 0), (0, 0))              # la distance prêtée vérifie ses points
        assert module.distance_km is eleve
    finally:
        del sys.modules[module.__name__]


# --------------------------------------------------------------------------- L'histoire et les outils


def test_les_outils_communs():
    assert len(clients_exemple()) == 6000 and len(antennes_exemple()) == 280 and len(plaintes_exemple()) == 845
    copie = clients_exemple()
    copie["C00001"]["parti"] = True
    assert clients_exemple()["C00001"]["parti"] is False                     # de nouveaux objets à chaque appel
    zone = zone_fibre_exemple()
    zone["coordinates"][0][0][0][0] = 0.0
    assert zone_fibre_exemple()["coordinates"][0][0][0][0] != 0.0
    for polygone in zone_fibre_exemple()["coordinates"]:
        for anneau in polygone:
            assert anneau[0] == anneau[-1] and len(anneau) >= 4
            assert all(4.5 < lon < 5.5 and 44.4 < lat < 45.2 for lon, lat in anneau)       # [lon, lat]
    lat_min, lat_max, lon_min, lon_max = carte.LAT_MIN, carte.LAT_MAX, carte.LON_MIN, carte.LON_MAX
    assert all(lat_min <= lat <= lat_max and lon_min <= lon <= lon_max for lat, lon in generer_points(500, graine=5))
    assert generer_points(10, graine=7) == generer_points(10, graine=7)
    assert set(sites_exemple()) >= set(carte._monde()["en_panne"])
    assert len(BASE32_GEOHASH) == 32 and not set("ailo") & set(BASE32_GEOHASH)


def test_l_histoire_tient():
    verite = set(carte._monde()["en_panne"])
    assert verite == {"A0010", "A0062", "A0192"}
    clients, antennes, plaintes = clients_exemple(), antennes_exemple(), plaintes_exemple()

    cases = Counter(corrige.geohash(p, 5) for p in plaintes).most_common(3)
    assert {code for code, _ in cases} == {corrige.geohash(antennes[a], 5) for a in verite}

    etiquettes = corrige.regrouper_pannes(plaintes, 1.0, 10)
    foyers = {}
    for p, e in zip(plaintes, etiquettes):
        if e >= 0:
            foyers.setdefault(e, []).append(p)
    assert len(foyers) == 3
    centres = {e: (sum(p[0] for p in f) / len(f), sum(p[1] for p in f) / len(f)) for e, f in foyers.items()}
    assert {a for a, _ in corrige.plus_proches(centres, antennes).values()} == verite

    rattachement = corrige.plus_proches({c: f["position"] for c, f in clients.items()}, antennes)
    touches = [c for c, (a, _) in rattachement.items() if a in verite]
    autres = [c for c in clients if c not in set(touches)]

    def taux(groupe):
        return sum(clients[c]["parti"] for c in groupe) / len(groupe)

    assert round(taux(touches), 2) == 0.41 and round(taux(autres), 2) == 0.07
    assert round(taux([c for c in touches if clients[c]["service"] == "Box 4G"]), 2) == 0.59
    zone = zone_fibre_exemple()
    fibre = [c for c, f in clients.items() if corrige.dans_zone(f["position"], zone)]
    assert len(fibre) == 1402 and round(taux(fibre), 2) == 0.05
    assert all(clients[c]["service"] != "Fibre" for c in set(clients) - set(fibre))

    sites = sites_exemple()
    assert round(_longueur(DEPOT, sites, sorted(sites))) == 941
    assert round(_longueur(DEPOT, sites, _ref_voisin(DEPOT, sites))) == 384
    assert round(corrige.tournee(DEPOT, sites)[1]) == 354


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    serie = next(s for s in series() if s.code == "43")
    assert [r.statut for r in evaluer(serie)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(serie, corrige=True)] == [REUSSI] * 6


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_43 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(carte)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_43 import *` (dans les exemples) doit donner les objets du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 40 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    zone, clients, antennes = zone_fibre_exemple(), {"C": [44.9, 4.9]}, {"A": [44.95, 4.95]}
    plaintes, sites = [[44.9, 4.9], [44.9001, 4.9]], {"S1": (45.0, 5.0), "S2": (44.8, 4.8)}
    copies = [repr(x) for x in (zone, clients, antennes, plaintes, sites)]
    corrige.dans_zone((44.9, 4.9), zone)
    corrige.plus_proches(clients, antennes)
    corrige.regrouper_pannes(plaintes, 1.0, 2)
    corrige.tournee(DEPOT, sites)
    assert [repr(x) for x in (zone, clients, antennes, plaintes, sites)] == copies


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(ValueError):
        corrige.distance_km((91, 0), (0, 0))
    with pytest.raises(TypeError):
        corrige.distance_km(("1", 0), (0, 0))
    with pytest.raises(ValueError):
        corrige.dans_zone((0, 0), {"type": "Point", "coordinates": [0, 0]})
    with pytest.raises(TypeError):
        corrige.geohash((0, 0), 7.0)
    with pytest.raises(ValueError):
        corrige.plus_proches({"C": (0, 0)}, {})
    with pytest.raises(ValueError):
        corrige.regrouper_pannes([(0, 0)], 0, 2)
    with pytest.raises(TypeError):
        corrige.tournee((0, 0), [(1, 1)])


def test_les_points_le_long_d_un_meridien_sont_a_des_km_exacts():
    a, b = _le_long(0.0, 0.6)
    assert math.isclose(_haversine(a, b), 0.6, rel_tol=1e-9)


def test_les_bugs_annonces_existent_tous():
    """Chaque erreur classique décrite par le correcteur a un test ci-dessus (et inversement)."""
    assert set(_BUGS_DISTANCE) == {"degres", "pythagore", "equirectangulaire", "cosinus", "un_moins_cos", "sans_deux",
                                   "equateur", "inverse"}
    assert set(_BUGS_ZONE) == {"inverse", "sans_trous", "premier_polygone", "sommet", "boite"}
    assert set(_BUGS_GEOHASH) == {"latitude_dabord", "strict", "alphabet", "poids_faible"}
    assert set(_BUGS_PROCHES) == {"premiere", "pythagore", "egalite", "grille", "metres"}
    assert set(_BUGS_DBSCAN) == {"sans_soi", "bord_etend", "vol", "bruit_definitif", "a_partir_de_un"}


def test_chaque_bug_change_au_moins_un_cas():
    for bug in _BUGS_DISTANCE:
        assert any(not math.isclose(_ref_distance(a, b, bug), _ref_distance(a, b), rel_tol=1e-9, abs_tol=1e-9)
                   for a, b in _cas_260()), bug
    for bug in _BUGS_ZONE:
        assert any(_ref_dans_zone(p, z, bug) != _ref_dans_zone(p, z) for p, z in _cas_261()), bug
    for bug in _BUGS_GEOHASH:
        assert any(_ref_geohash(p, 7 if n is None else n, bug) != _ref_geohash(p, 7 if n is None else n)
                   for p, n in _cas_262()), bug
    for bug in _BUGS_DBSCAN:
        assert any(_ref_dbscan(p, r, m, bug) != _ref_dbscan(p, r, m) for p, r, m in _cas_264()), bug

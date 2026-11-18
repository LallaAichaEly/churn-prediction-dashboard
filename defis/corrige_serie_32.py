"""Série 32 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_32.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours un
écart-type calculé avec n − 1, un point compté dans sa propre distance moyenne (silhouette), ou des
étiquettes renvoyées qui ne sont plus celles des centres renvoyés.

Fil rouge de la série : k-means ne voit que des DISTANCES. Tout se joue sur ce qu'on mesure, et comment.

1. avant de mesurer, mettre les variables à la même échelle : sinon la facture (des dizaines d'euros)
   écrase les appels (quelques unités), et les segments ne parlent que d'argent ;
2. une étape de k-means, c'est deux gestes : chaque point va au centre le plus proche (à égalité, le
   premier), puis chaque centre va au milieu de ses points (un centre sans point ne bouge pas) ;
3. on s'arrête quand plus aucune étiquette ne change, et ce qu'on renvoie est COHÉRENT : les étiquettes
   sont celles des centres renvoyés, l'inertie aussi ;
4. le départ compte : k-means++ choisit des centres éloignés les uns des autres, au hasard pondéré par la
   distance AU CARRÉ au centre le PLUS PROCHE, mise à jour avec le seul nouveau centre ;
5. un bon découpage se vérifie (silhouette), puis se raconte dans les unités du métier, par ce qui
   DISTINGUE chaque segment (en écarts-types), pas par ses plus gros chiffres.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import math

from defis.segments import (  # noqa: F401
    VARIABLES,
    Partition,
    Segment,
    clients_exemple,
    generer_clients,
    generer_points,
    points_de,
)


def _distance2(p, q) -> float:
    """La distance euclidienne AU CARRÉ : pas de racine, inutile pour comparer, et exacte sur des 1/16."""
    return sum((a - b) ** 2 for a, b in zip(p, q))


# ============================================================================ Défi 194 : standardiser


def standardiser(points: list[tuple[float, ...]]
                 ) -> tuple[list[tuple[float, ...]], tuple[float, ...], tuple[float, ...]]:
    """(z, moyennes, ecarts) : chaque COLONNE ramenée à une moyenne 0 et un écart-type 1."""
    lignes = [tuple(p) for p in points]
    if not lignes:
        raise ValueError("aucun point : la moyenne d'une colonne vide n'existe pas")
    d = len(lignes[0])
    for i, ligne in enumerate(lignes):
        if len(ligne) != d:
            raise ValueError(f"le point n°{i} a {len(ligne)} coordonnées au lieu de {d}")
        for x in ligne:
            if x is None or x != x:            # NaN est le seul nombre différent de lui-même
                raise ValueError(f"valeur manquante dans le point n°{i} : {ligne}")
    n = len(lignes)
    # Une moyenne et un écart-type PAR COLONNE (par variable), calculés UNE fois chacun. Les recalculer
    # pour chaque valeur ferait n fois le même travail : 5 000 clients, 25 millions d'additions.
    moyennes = tuple(sum(ligne[j] for ligne in lignes) / n for j in range(d))
    # L'écart-type de la POPULATION (diviser par n, comme StandardScaler) : on décrit CES clients, on
    # n'estime pas celui d'une population plus large. statistics.stdev divise par n − 1.
    ecarts = tuple(math.sqrt(sum((ligne[j] - moyennes[j]) ** 2 for ligne in lignes) / n) for j in range(d))
    # Une colonne constante (tous les clients ont 0 appel) n'a pas d'écart-type : elle ne distingue
    # personne, z = 0. (Avec des valeurs comme 0,1, la moyenne d'une colonne constante peut différer de
    # 1e-17 de ses valeurs : en vrai, on compare l'écart à un petit seuil, comme le fait scikit-learn.)
    z = [tuple(0.0 if s == 0 else (x - m) / s for x, m, s in zip(ligne, moyennes, ecarts)) for ligne in lignes]
    return z, moyennes, ecarts


# ============================================================================ Défi 195 : affecter


def affecter(points: list[tuple[float, ...]], centres: list[tuple[float, ...]]) -> list[int]:
    """Pour chaque point, le numéro du centre le plus proche (à égalité, le plus petit numéro)."""
    if not centres:
        raise ValueError("aucun centre : à quel segment affecter les points ?")
    d = len(centres[0])
    # `zip` s'arrête au plus court : sans ce contrôle, un point à 2 coordonnées serait comparé sur 2
    # coordonnées seulement, sans le moindre message. On refuse tout de suite.
    for nom, liste in (("centre", centres), ("point", points)):
        for i, x in enumerate(liste):
            if len(x) != d:
                raise ValueError(f"le {nom} n°{i} a {len(x)} coordonnées au lieu de {d}")
    etiquettes = []
    for p in points:
        meilleur, d_min = 0, _distance2(p, centres[0])
        for j in range(1, len(centres)):
            dj = _distance2(p, centres[j])
            if dj < d_min:                     # « < » strict : à égalité, le premier centre garde le point
                meilleur, d_min = j, dj
        etiquettes.append(meilleur)
    return etiquettes


# ============================================================================ Défi 196 : kmeans


def _recentrer(points, etiquettes, anciens):
    """Chaque centre au milieu de ses points ; un centre sans point reste où il était."""
    d = len(anciens[0])
    sommes = [[0.0] * d for _ in anciens]
    tailles = [0] * len(anciens)
    for p, e in zip(points, etiquettes):
        tailles[e] += 1
        for j, x in enumerate(p):
            sommes[e][j] += x
    # Un centre vide garde sa place (le retirer changerait k et décalerait tous les numéros ; le mettre à
    # zéro l'enverrait au milieu des données, où il volerait des points au hasard).
    return [tuple(s / t for s in somme) if t else tuple(ancien)
            for somme, t, ancien in zip(sommes, tailles, anciens)]


def kmeans(points: list[tuple[float, ...]], centres: list[tuple[float, ...]], max_iter: int = 100) -> Partition:
    """L'algorithme de Lloyd : affecter, recentrer, recommencer tant qu'une étiquette change."""
    if max_iter < 1:
        raise ValueError(f"max_iter doit valoir au moins 1, pas {max_iter}")
    if not points:
        raise ValueError("aucun point à regrouper")
    centres = [tuple(c) for c in centres]      # une COPIE : on ne touche pas à la liste reçue
    etiquettes = affecter(points, centres)     # lève ValueError s'il n'y a pas de centre
    iterations = 0
    while iterations < max_iter:
        centres = _recentrer(points, etiquettes, centres)
        iterations += 1
        nouvelles = affecter(points, centres)
        stable = nouvelles == etiquettes
        # Toujours garder les étiquettes des centres ACTUELS, même si l'on s'arrête faute d'itérations :
        # ce qu'on renvoie doit être cohérent (étiquettes, centres et inertie décrivent la même chose).
        etiquettes = nouvelles
        if stable:
            break
    # L'inertie : la somme des distances AU CARRÉ. C'est ce que k-means fait baisser à chaque étape.
    inertie = float(sum(_distance2(p, centres[e]) for p, e in zip(points, etiquettes)))
    return Partition(centres, etiquettes, inertie, iterations)


# ============================================================================ Défi 197 : kmeans_plus_plus


def kmeans_plus_plus(points: list[tuple[float, ...]], k: int, tirages: list[float]) -> list[int]:
    """Les numéros des k points choisis comme centres de départ, tirés au hasard pondéré par D²."""
    n = len(points)
    if not 1 <= k <= n:
        raise ValueError(f"k doit être entre 1 et le nombre de points ({n}), pas {k}")
    if len(tirages) < k:
        raise ValueError(f"il faut au moins k = {k} tirages, pas {len(tirages)}")
    choisis = [int(tirages[0] * n)]            # le premier : uniformément au hasard
    # poids[j] = distance AU CARRÉ de j au centre choisi le PLUS PROCHE. Un point déjà choisi pèse 0.
    centre = points[choisis[0]]
    poids = [_distance2(p, centre) for p in points]
    for i in range(1, k):
        total = sum(poids)
        if total == 0:                         # tous les points sont déjà des centres (doublons)
            raise ValueError(f"moins de k = {k} points distincts : impossible de choisir {k} centres")
        seuil = tirages[i] * total
        cumul = 0.0
        for j, w in enumerate(poids):
            cumul += w
            if cumul > seuil:                  # « > » strict : un point de poids 0 n'est jamais choisi
                break
        choisis.append(j)
        # Mise à jour INCRÉMENTALE : seul le nouveau centre peut rapprocher un point. Recalculer la
        # distance à TOUS les centres déjà choisis coûterait k fois plus.
        nouveau = points[j]
        poids = [min(w, _distance2(p, nouveau)) for w, p in zip(poids, points)]
    return choisis


# ============================================================================ Défi 198 : silhouette


def silhouette(points: list[tuple[float, ...]], etiquettes: list) -> float:
    """La moyenne, sur TOUS les points, de s = (b − a) / max(a, b)."""
    if len(points) != len(etiquettes):
        raise ValueError(f"{len(points)} points mais {len(etiquettes)} étiquettes")
    membres: dict = {}
    for i, e in enumerate(etiquettes):         # un dict : les étiquettes peuvent valoir 3, 7 ou « A »
        membres.setdefault(e, []).append(i)
    if len(membres) < 2:
        raise ValueError("la silhouette compare des segments entre eux : il en faut au moins deux")
    total = 0.0
    for i, p in enumerate(points):
        propre = etiquettes[i]
        taille = len(membres[propre])
        if taille == 1:                        # seul dans son segment : a n'existe pas, s = 0 par convention
            continue
        # Les VRAIES distances (avec la racine) : la silhouette compare des distances moyennes.
        sommes = {e: sum(math.dist(p, points[j]) for j in idx) for e, idx in membres.items()}
        a = sommes[propre] / (taille - 1)      # sa propre distance (0) est dans la somme : diviser par taille − 1
        b = min(sommes[e] / len(idx) for e, idx in membres.items() if e != propre)   # le segment VOISIN
        plus_grand = max(a, b)
        total += 0.0 if plus_grand == 0 else (b - a) / plus_grand
    return total / len(points)                 # sur TOUS les points, les seuls dans leur segment comptent 0


# ============================================================================ Défi 199 : decrire_segments


def decrire_segments(clients: list[dict], etiquettes: list[int], variables=VARIABLES) -> list[Segment]:
    """Une fiche par segment, dans les unités du métier, du plus fort taux de churn au plus faible."""
    if len(clients) != len(etiquettes):
        raise ValueError(f"{len(clients)} clients mais {len(etiquettes)} étiquettes")
    variables = list(variables)
    if not variables:
        raise ValueError("aucune variable pour décrire les segments")
    if not clients:
        return []
    n = len(clients)
    # La référence : TOUS les clients. Un segment se distingue par rapport à l'ensemble, et l'écart se
    # mesure en écarts-types GLOBAUX (de la population) : « 3 mois de moins » et « 3 € de moins » ne
    # pèsent pas pareil, « 1,4 écart-type de moins » si.
    globales = {v: sum(c[v] for c in clients) / n for v in variables}
    ecarts = {v: math.sqrt(sum((c[v] - globales[v]) ** 2 for c in clients) / n) for v in variables}
    groupes: dict[int, list[dict]] = {}
    for client, e in zip(clients, etiquettes):
        groupes.setdefault(e, []).append(client)
    segments = []
    for numero, membres in groupes.items():
        taille = len(membres)
        moyennes = {v: sum(c[v] for c in membres) / taille for v in variables}   # BRUTES : mois, euros, appels
        distinctif = None
        for v in variables:
            z = 0.0 if ecarts[v] == 0 else (moyennes[v] - globales[v]) / ecarts[v]
            if distinctif is None or abs(z) > abs(distinctif[1]):   # « > » : à égalité, la première variable
                distinctif = (v, z)                                  # le SIGNE dit dans quel sens
        segments.append(Segment(numero, taille, taille / n, sum(c["churn"] for c in membres) / taille,
                                moyennes, distinctif))
    segments.sort(key=lambda s: (-s.taux_churn, s.numero))
    return segments


# ============================================================================ Des tests, comme dans tests/


def test_sans_standardiser_la_facture_ecrase_les_appels():
    points = points_de(clients_exemple())
    brut = kmeans(points, [points[0], points[4], points[8]])
    z, _, _ = standardiser(points)
    juste = kmeans(z, [z[0], z[4], z[8]])
    assert juste.etiquettes == [0] * 4 + [1] * 4 + [2] * 4
    assert brut.etiquettes != juste.etiquettes          # C04 (70 €, 5 appels) part avec les fidèles


def test_une_colonne_constante_donne_des_zeros():
    z, moyennes, ecarts = standardiser([(1.0, 3.0), (3.0, 3.0)])
    assert z == [(-1.0, 0.0), (1.0, 0.0)] and moyennes == (2.0, 3.0) and ecarts == (1.0, 0.0)


def test_a_egalite_le_premier_centre_garde_le_point():
    assert affecter([(1.0,), (3.0,)], [(0.0,), (2.0,)]) == [0, 1]


def test_un_centre_sans_point_ne_bouge_pas():
    partition = kmeans([(0.0,), (1.0,)], [(0.0,), (100.0,)])
    assert partition.centres == [(0.5,), (100.0,)] and partition.etiquettes == [0, 0]


def test_kmeans_plus_plus_ne_reprend_jamais_un_centre():
    assert kmeans_plus_plus([(0.0,), (0.0,), (4.0,)], 2, [0.0, 0.0]) == [0, 2]


def test_un_client_seul_dans_son_segment_compte_zero():
    assert silhouette([(0.0,), (0.0,), (10.0,)], [0, 0, 1]) == 2 / 3


def test_le_segment_se_raconte_en_ecarts_types():
    segments = decrire_segments(clients_exemple(), [0] * 4 + [1] * 4 + [2] * 4)
    assert [s.numero for s in segments] == [0, 2, 1]
    assert segments[0].distinctif[0] == "appels" and segments[0].distinctif[1] > 0

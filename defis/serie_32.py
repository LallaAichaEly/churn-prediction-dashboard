"""Série 32 — Regrouper les clients : standardiser, affecter, k-means, k-means++, silhouette, décrire.

🧩 VOUS VOULEZ DÉVELOPPER ? LE `KMeans` DE scikit-learn : ICI, VOUS L'ÉCRIVEZ, ET VOUS SAVEZ QUAND IL SE TROMPE.

Camille, de l'équipe rétention, ne peut pas écrire une campagne par client, ni la même pour tous. Elle veut
trois ou quatre GROUPES de clients qui se ressemblent (des segments), chacun avec son message : « on vous
rappelle » pour ceux qui appellent sans cesse, une remise pour les grosses factures, rien pour les fidèles.
L'algorithme le plus utilisé pour ça, k-means, tient en vingt lignes. Mais il ne voit que des DISTANCES : il
suffit d'une variable en euros à côté d'une variable en appels pour qu'il ne parle plus que d'argent. Cette
série vous fait écrire les six pièces d'une segmentation, de la mise à l'échelle à la fiche qu'on présente :

    194 standardiser      chaque variable à la même échelle (moyenne 0, écart-type 1)
    195 affecter          chaque client rejoint le centre le plus proche
    196 kmeans            affecter, recentrer, recommencer : l'algorithme de Lloyd
    197 kmeans_plus_plus  bien choisir les centres de départ (le « k-means++ » de scikit-learn)
    198 silhouette        le découpage est-il net ? Une note entre -1 et 1
    199 decrire_segments  la fiche de chaque segment, dans les unités du métier

Prérequis : la série 00 (listes, tuples, dict, boucles). Le module 2 de PARCOURS.md (`StandardScaler`) et la
série 23 (recoder un algorithme de scikit-learn) l'éclairent, mais tout ce qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un POINT, c'est un client vu comme un tuple de nombres : (ancienneté, facture, appels). La distance entre
  deux points est la distance euclidienne : la racine de la somme des carrés des écarts. Pour COMPARER des
  distances, la racine ne sert à rien : on compare les distances AU CARRÉ (même ordre, calcul exact) ;
- ces nombres n'ont pas la même unité. 30 € d'écart de facture et 3 appels d'écart n'ont rien de commun,
  mais pour la distance, 30² pèse cent fois plus que 3². On STANDARDISE donc chaque variable :
  z = (x − moyenne) / écart-type. Un écart d'un écart-type pèse alors pareil partout ;
- k-means cherche k CENTRES. Une étape, c'est deux gestes : chaque point rejoint le centre le plus proche,
  puis chaque centre se place au milieu (la moyenne) de ses points. On recommence jusqu'à ce que plus aucun
  point ne change de centre. L'INERTIE (la somme des distances au carré de chaque point à son centre) baisse
  à chaque étape : l'algorithme s'arrête toujours ;
- le résultat dépend des centres de départ : k-means++ les tire au hasard, mais loin les uns des autres ;
- k-means trouve TOUJOURS k groupes, même quand il n'y en a pas : la silhouette dit si le découpage est net ;
- un segment se présente dans les unités du métier (« 2,5 mois d'ancienneté, 5 appels ») et se nomme par
  ce qui le DISTINGUE des autres clients, pas par son plus gros chiffre.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_32 import *
    >>> clients = clients_exemple()
    >>> points = points_de(clients)                    # (ancienneté en mois, facture en €, appels)
    >>> points[3]                                      # C04 : récent, 6 appels… et une grosse facture
    (4.0, 118.0, 6.0)

Quand les six défis sont réussis, vous avez une segmentation complète :

    >>> z, moyennes, ecarts = standardiser(points)
    >>> depart = kmeans_plus_plus(z, 3, [0.0, 0.5, 0.5])     # C01, puis C07 (loin), puis C10
    >>> depart
    [0, 6, 9]
    >>> partition = kmeans(z, [z[i] for i in depart])
    >>> partition.etiquettes                                 # récents / fidèles / grosses factures
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    >>> kmeans(points, [points[i] for i in depart]).etiquettes   # SANS standardiser : C04 change de camp
    [0, 0, 0, 2, 1, 1, 1, 1, 2, 2, 2, 2]
    >>> round(silhouette(z, partition.etiquettes), 3)
    0.702
    >>> fiche = decrire_segments(clients, partition.etiquettes)[0]
    >>> fiche.numero, fiche.taux_churn, fiche.distinctif[0]  # le segment qui part : il appelle beaucoup
    (0, 0.75, 'appels')

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les défis sont indépendants, sauf le 196 qui
   peut appeler votre 195 ; l'ordre proposé est le plus progressif.
2. Lancez `python scripts/defis.py --defi 194` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_32.py`.

Les outils communs (`clients_exemple`, `points_de`, `VARIABLES`, `Partition`, `Segment`, `generer_points`,
`generer_clients`) sont dans `defis/segments.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import math  # noqa: F401 — math.sqrt(x) : la racine ; math.dist(p, q) : la distance euclidienne (avec la racine)

from defis.segments import (  # noqa: F401
    VARIABLES,
    Partition,
    Segment,
    clients_exemple,
    generer_clients,
    generer_points,
    points_de,
)

# ============================================================================ Mettre à la même échelle


def standardiser(points: list[tuple[float, ...]]
                 ) -> tuple[list[tuple[float, ...]], tuple[float, ...], tuple[float, ...]]:
    """Défi 194 🟢 — (z, moyennes, ecarts) : chaque VARIABLE (chaque colonne) ramenée à moyenne 0, écart-type 1.

        >>> z, moyennes, ecarts = standardiser([(2, 10, 0), (2, 30, 0), (6, 10, 0), (6, 30, 0)])
        >>> z                                        # la 3e colonne est constante : des 0.0
        [(-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0)]
        >>> moyennes, ecarts
        ((4.0, 20.0, 0.0), (2.0, 10.0, 0.0))
        >>> z, moyennes, ecarts = standardiser(points_de(clients_exemple()))
        >>> [round(m, 2) for m in moyennes], [round(e, 2) for e in ecarts]
        ([28.83, 81.08, 2.42], [22.96, 37.03, 2.14])
        >>> [round(x, 2) for x in z[3]]              # C04 : 1,7 écart-type d'appels au-dessus de la moyenne
        [-1.08, 1.0, 1.68]

    Les règles :

    - chaque COLONNE séparément : sa moyenne, puis son écart-type de la POPULATION, la racine de la moyenne
      des (x − moyenne)². On divise par n, comme `StandardScaler` (`statistics.stdev` divise par n − 1 : pas
      celui-là) ;
    - z[i][j] = (points[i][j] − moyennes[j]) / ecarts[j]. Une colonne constante (tous les clients ont 0
      appel) a un écart-type nul : elle ne distingue personne, ses z valent 0.0, sans division par zéro, et
      `ecarts` garde sa vraie valeur, 0.0 ;
    - `z` est une liste de tuples (un par point, dans le même ordre) ; `moyennes` et `ecarts` sont des
      tuples (une valeur par variable) ;
    - aucun point → ValueError ; des points qui n'ont pas tous le même nombre de coordonnées → ValueError ;
      une valeur manquante (None ou NaN) → ValueError : k-means ne saurait qu'en faire, et un NaN contamine
      sans bruit toutes les distances (`x != x` n'est vrai que pour NaN) ;
    - ne modifiez pas `points` (ni la liste, ni les points eux-mêmes s'ils sont des listes) ;
    - 5 000 clients × 5 variables : le correcteur compte les lignes exécutées. Calculez chaque moyenne et
      chaque écart-type UNE fois par colonne, pas une fois par valeur.

    Ce que vous apprenez : pourquoi toute méthode fondée sur des distances (k-means, k plus proches voisins,
    SVM, régression pénalisée) exige des variables à la même échelle, l'écart-type de la population et celui
    de l'échantillon, et refuser tout de suite une donnée manquante plutôt que de calculer faux.

    Indice : `n, d = len(points), len(points[0])` ;
    `moyennes = tuple(sum(p[j] for p in points) / n for j in range(d))`, puis la même chose pour les écarts
    avec `math.sqrt(sum((p[j] - moyennes[j]) ** 2 for p in points) / n)`. Une compréhension construit `z`.
    """
    raise NotImplementedError


# ============================================================================ L'algorithme de Lloyd


def affecter(points: list[tuple[float, ...]], centres: list[tuple[float, ...]]) -> list[int]:
    """Défi 195 🟢 — Pour chaque point, le numéro (0, 1, …) du centre le plus proche.

        >>> affecter([(0.0, 0.0), (9.0, 1.0), (4.0, 1.0)], [(0.0, 1.0), (8.0, 1.0)])   # (4, 1) : à 4 des deux
        [0, 1, 0]
        >>> points = points_de(clients_exemple())
        >>> affecter(points, [points[0], points[4], points[8]])   # en euros et en mois : C04 part avec les
        [0, 0, 0, 2, 1, 1, 1, 1, 2, 2, 2, 2]
        >>> z, _, _ = standardiser(points)                        # grosses factures ; standardisé, il reste
        >>> affecter(z, [z[0], z[4], z[8]])                       # avec les récents qui appellent
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    Les règles :

    - la distance est euclidienne. Comparez les distances AU CARRÉ, `sum((a - b) ** 2 for a, b in zip(p, c))` :
      la racine ne change pas l'ordre ;
    - à égalité, le plus petit numéro (le premier centre de la liste) ;
    - aucun centre → ValueError ; un point ou un centre qui n'a pas le même nombre de coordonnées que le
      premier centre → ValueError (`zip` s'arrête au plus court : sans contrôle, un point à 2 coordonnées
      serait comparé sur 2 coordonnées seulement, sans le moindre message) ;
    - aucun point → [] ;
    - ne modifiez ni `points` ni `centres`.

    Ce que vous apprenez : l'étape d'affectation de k-means (et des k plus proches voisins), régler les
    égalités dans un ordre annoncé, pourquoi on se passe de la racine, et que `zip` tronque en silence.

    Indice : pour chaque point, `meilleur, d_min = 0, <distance² au centre 0>`, puis pour j à partir de 1 :
    si `dj < d_min` (strict), `meilleur, d_min = j, dj`. (`min(range(len(centres)), key=…)` renvoie aussi le
    premier des ex aequo.)
    """
    raise NotImplementedError


def kmeans(points: list[tuple[float, ...]], centres: list[tuple[float, ...]], max_iter: int = 100) -> Partition:
    """Défi 196 🟡 — L'algorithme de Lloyd, à partir des centres donnés : Partition(centres, etiquettes, inertie, iterations).

        >>> points = [(0.0,), (1.0,), (2.0,), (10.0,), (11.0,), (12.0,)]
        >>> kmeans(points, [(0.0,), (1.0,)])
        Partition(centres=[(1.0,), (11.0,)], etiquettes=[0, 0, 0, 1, 1, 1], inertie=4.0, iterations=2)
        >>> p = kmeans(points, [(0.0,), (1.0,)], max_iter=1)      # arrêté en route…
        >>> p.centres, p.etiquettes, p.iterations, round(p.inertie, 2)   # … mais cohérent
        ([(0.0,), (7.2,)], [0, 0, 0, 1, 1, 1], 1, 50.32)
        >>> kmeans([(0.0,), (1.0,)], [(0.0,), (100.0,)])          # un centre sans point ne bouge pas
        Partition(centres=[(0.5,), (100.0,)], etiquettes=[0, 0], inertie=0.5, iterations=1)

    L'algorithme :

    1. `etiquettes` : chaque point affecté au centre le plus proche (votre défi 195 fait l'affaire) ;
    2. RECENTRER : chaque centre devient la moyenne, coordonnée par coordonnée, de SES points. Un centre sans
       point garde sa place (le retirer changerait k et décalerait tous les numéros ; le mettre à 0
       l'enverrait au hasard au milieu des données) ;
    3. réaffecter. Si aucune étiquette n'a changé, c'est fini ; sinon, retour en 2.

    Les règles :

    - `iterations` = le nombre de fois où l'étape 2 a eu lieu : au moins 1, au plus `max_iter`. Arrivé à
      `max_iter`, on s'arrête même si des étiquettes changent encore ;
    - le résultat est COHÉRENT : les `etiquettes` sont celles des `centres` renvoyés (l'étape 3 suit
      toujours l'étape 2, même au dernier tour), et `inertie` est la somme, sur tous les points, de la
      distance AU CARRÉ à son centre dans `centres` ;
    - `centres` : une liste de k tuples (k = le nombre de centres reçus), dans le même ordre : le centre j
      est celui de l'étiquette j ;
    - `max_iter` < 1 → ValueError ; aucun point → ValueError ; aucun centre → ValueError ;
    - ne modifiez ni `points` ni `centres` (`centres[j] = …` sur la liste reçue changerait les centres de
      celui qui vous appelle : travaillez sur une copie).

    Ce que vous apprenez : l'algorithme de Lloyd (ce que fait `KMeans(init=…, n_init=1)`), pourquoi il
    s'arrête toujours (l'inertie baisse à chaque étape), le cas du segment vide, et renvoyer un résultat
    dont toutes les parties décrivent la même chose.

    Indice : `centres = [tuple(c) for c in centres]` (une copie), `etiquettes = affecter(points, centres)`,
    puis `while iterations < max_iter:`. Pour recentrer : `sommes = [[0.0] * d for _ in centres]` et
    `tailles = [0] * len(centres)`, remplies en un seul passage sur les points.
    """
    raise NotImplementedError


# ============================================================================ Bien démarrer


def kmeans_plus_plus(points: list[tuple[float, ...]], k: int, tirages: list[float]) -> list[int]:
    """Défi 197 🔴 — Les numéros des k points qui serviront de centres de départ (k-means++).

    `tirages` : k nombres au hasard entre 0 (inclus) et 1 (exclu), fournis pour que le résultat soit
    reproductible (en production : `[random.random() for _ in range(k)]`).

        >>> points = [(0.0,), (1.0,), (5.0,), (6.0,), (20.0,)]
        >>> kmeans_plus_plus(points, 3, [0.0, 0.5, 0.5])
        [0, 4, 3]

    Ici : le 1er centre est le point n°int(0.0 × 5) = 0. Les distances au carré à ce centre sont
    [0, 1, 25, 36, 400] : total 462, seuil 0,5 × 462 = 231. Les sommes cumulées 0, 1, 26, 62, 462 dépassent
    231 au point n°4 (20.0). Ensuite, chaque point garde sa distance au carré au centre le PLUS PROCHE :
    [0, 1, 25, 36, 0], total 62, seuil 31 : les sommes cumulées 0, 1, 26, 62 le dépassent au point n°3.

    Les règles :

    - le 1er centre est le point n° `int(tirages[0] * n)` (n = le nombre de points) ;
    - pour chacun des suivants : D²(p) = la distance AU CARRÉ de p au centre déjà choisi le PLUS PROCHE (pas
      au dernier choisi) ; seuil = tirages[i] × (la somme des D²) ; on prend le PREMIER point j dont la somme
      cumulée D²(0) + … + D²(j) est STRICTEMENT plus grande que le seuil. Un point loin de tous les centres
      occupe une grande part de la somme : il a plus de chances. Un point déjà choisi a D² = 0 : avec « > »,
      il n'est jamais repris ;
    - k doit être entre 1 et n, sinon ValueError ; moins de k tirages → ValueError ; si la somme des D² vaut
      0 avant d'avoir k centres (moins de k points différents), ValueError ;
    - le résultat : la liste des k numéros (des int), dans l'ordre du choix ; ne modifiez rien ;
    - 1 200 points et k = 60 : le correcteur compte les lignes exécutées. Recalculer à chaque tour, pour chaque
      point, la distance à TOUS les centres choisis, c'est k fois trop : gardez D² pour chaque point, et
      mettez-le à jour avec le SEUL nouveau centre : `min(ancien, distance²(p, nouveau))`.

    Ce que vous apprenez : l'initialisation k-means++ (celle de scikit-learn par défaut), tirer au hasard
    selon des poids (la somme cumulée, ce que fait `random.choices`), et la mise à jour incrémentale.

    Indice : `poids = [<distance²(p, premier centre)> for p in points]`. À chaque tour :
    `seuil = tirages[i] * sum(poids)`, puis `cumul += w` jusqu'à `cumul > seuil`.
    (`itertools.accumulate` et `bisect.bisect_right` font la même chose.)
    """
    raise NotImplementedError


# ============================================================================ Juger et raconter


def silhouette(points: list[tuple[float, ...]], etiquettes: list) -> float:
    """Défi 198 🔴 — La silhouette moyenne : près de 1 = segments nets, 0 = frontières floues, < 0 = mal rangés.

    Pour chaque point i :

    - a(i) = la distance moyenne de i aux AUTRES points de son segment (lui-même exclu) ;
    - b(i) = pour chaque AUTRE segment, la distance moyenne de i à ses points ; b(i) est la plus PETITE de ces
      moyennes (celle du segment voisin, où i irait s'il changeait) ;
    - s(i) = (b(i) − a(i)) / max(a(i), b(i)).

        >>> silhouette([(0.0,), (2.0,), (10.0,), (12.0,)], [0, 0, 1, 1])     # 0 : a = 2, b = 11, s = 9/11
        0.797979797979798
        >>> silhouette([(0.0,), (0.0,), (10.0,)], [0, 0, 1])                 # 10.0 est seul : il compte 0
        0.6666666666666666
        >>> silhouette([(0.0,), (2.0,), (10.0,), (12.0,)], ["A", "B", "A", "B"])   # un découpage absurde
        -0.4

    Les règles :

    - des distances euclidiennes AVEC la racine (`math.dist(p, q)`) : on fait des moyennes de distances, et
      des carrés gonfleraient les plus lointaines ;
    - un point seul dans son segment a s = 0 (sans voisin dans son segment, a n'existe pas) ; si
      max(a, b) = 0 (des points identiques dans deux segments), s = 0 ;
    - le résultat est la moyenne de s sur TOUS les points (ceux qui sont seuls comptent 0), pas la moyenne
      des moyennes de chaque segment ;
    - les étiquettes peuvent être n'importe quelles valeurs (0, 1, 7, « A ») : pas forcément 0, 1, …, k − 1 ;
    - moins de 2 segments différents → ValueError (pas de voisin : b n'existe pas) ; pas autant d'étiquettes
      que de points → ValueError ; ne modifiez rien ;
    - n² distances, c'est normal ; le correcteur refuse n³ (refaire la liste des membres d'un segment pour
      chaque paire de points, par exemple).

    Ce que vous apprenez : juger un regroupement sans « bonne réponse » à comparer, choisir k (celui dont la
    silhouette est la meilleure), et une question classique d'entretien : « comment savez-vous que vos
    clusters sont bons ? ».

    Indice : `membres = {}`, puis `membres.setdefault(e, []).append(i)`. Pour chaque point, la somme de ses
    distances aux membres de chaque segment ; `a = somme_du_sien / (taille - 1)`, et `b` est le `min` des
    `somme / taille` des autres segments.
    """
    raise NotImplementedError


def decrire_segments(clients: list[dict], etiquettes: list[int], variables=VARIABLES) -> list[Segment]:
    """Défi 199 🟡 — Une fiche par segment, dans les unités du métier, du plus fort taux de churn au plus faible.

        >>> fiches = decrire_segments(clients_exemple(), [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        >>> [(f.numero, f.taille, f.taux_churn) for f in fiches]
        [(0, 4, 0.75), (2, 4, 0.25), (1, 4, 0.0)]
        >>> fiches[0].moyennes                       # en mois, en euros, en appels : lisible par Camille
        {'anciennete': 2.5, 'facture': 93.25, 'appels': 5.25}
        >>> nom, ecart = fiches[0].distinctif       # 93,25 € est le plus gros chiffre, mais c'est banal ;
        >>> nom, round(ecart, 2)                     # 5,25 appels, c'est 1,3 écart-type au-dessus de tous
        ('appels', 1.32)
        >>> fiches[2].distinctif[0], round(fiches[2].distinctif[1], 2)   # les fidèles : de petites factures
        ('facture', -1.31)

    Les règles :

    - une fiche par étiquette présente : `Segment(numero, taille, part, taux_churn, moyennes, distinctif)` ;
    - `part` = taille / nombre total de clients ; `taux_churn` = la moyenne de `churn` (0 ou 1) des clients
      du segment ;
    - `moyennes` = {variable: moyenne dans le segment}, pour chacune des `variables`, sur les valeurs
      BRUTES (mois, euros) : jamais sur des valeurs standardisées, personne ne comprend « facture : 0,33 » ;
    - `distinctif` = (variable, z), avec z = (moyenne du segment − moyenne de TOUS les clients) / écart-type
      de TOUS les clients (de la population, comme au défi 194) : de combien d'écarts-types le segment
      s'écarte de l'ensemble. On garde la variable de plus grand |z| (à égalité, la première de
      `variables`) ; z garde son SIGNE (−1,3 en facture = de petites factures). Une variable dont l'écart-type
      global est nul a z = 0.0 ;
    - les fiches sont triées par taux de churn DÉCROISSANT, puis par numéro croissant (sans `reverse=True`) ;
    - pas autant d'étiquettes que de clients → ValueError ; aucune variable → ValueError ; aucun client → [] ;
    - ne modifiez pas `clients`.

    Ce que vous apprenez : raconter un regroupement à quelqu'un du métier (les « personas » du marketing),
    comparer des variables d'unités différentes (encore la standardisation, cette fois pour décrire), et
    pourquoi le plus gros chiffre n'est pas le plus important.

    Indice : d'abord la moyenne et l'écart-type de chaque variable sur TOUS les clients ; puis
    `groupes.setdefault(e, []).append(client)`. Pour chaque groupe : ses moyennes, puis une boucle sur
    `variables` qui garde (v, z) si `abs(z) > abs(meilleur_z)` (strict). Triez avec
    `key=lambda s: (-s.taux_churn, s.numero)`.
    """
    raise NotImplementedError

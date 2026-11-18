"""Série 28 — Les données qui n'arrêtent pas d'arriver : résumer un flux sans le garder.

🌊 VOUS VOULEZ DÉVELOPPER ? UN JOUR, LES DONNÉES NE TIENDRONT PLUS EN MÉMOIRE.

L'application envoie 40 millions d'événements par jour : connexions, appels au support, factures, clics. Le
tableau de bord veut, à chaque minute, le délai moyen de paiement et sa dispersion, la médiane des factures,
un score d'activité par client, et une alerte quand le taux de résiliation monte. Relire tout l'historique
avec `pd.read_csv` à chaque minute ? Impossible. On lit chaque événement UNE fois, au moment où il arrive,
et on met à jour un petit RÉSUMÉ. C'est un « algorithme en flux » (streaming), et cette série vous en fait
écrire six :

    170 MoyenneVariance        moyenne et variance en gardant 3 nombres… sans perdre tous ses chiffres
    171 alertes_cusum          sonner quand le churn monte DURABLEMENT, pas à chaque pic
    172 ActiviteRecente        un score qui s'efface avec le temps, même quand les données arrivent en retard
    173 MedianeCourante        la médiane après chaque valeur, sans tout retrier (deux tas)
    174 echantillon_reservoir  1 000 clients tirés au hasard dans un flux dont on ne connaît pas la longueur
    175 fusionner              dix machines, dix résumés, un seul résultat (l'idée de MapReduce)

Prérequis : la série 00, les classes (défi 48) et les générateurs (défi 47). La série 09 (complexité, `heapq`)
éclaire le défi 173 ; la série 21 (dates) le défi 172 ; la série 26 (le hasard reproductible) le défi 174.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un flux se lit UNE fois : pas de `len(flux)` (on ne sait pas combien de valeurs vont arriver), pas de
  `flux[i]`, et pas de deuxième boucle. Attention, un générateur relu ne lève pas d'erreur : il est VIDE, en
  silence. Le correcteur vous donne des flux `UneSeuleFois`, qui, eux, le disent ;
- on garde un résumé de taille fixe : quelques nombres, pas la liste des valeurs. Le correcteur mesure la
  mémoire (`pic_memoire`) : une liste de 300 000 valeurs se voit. Certaines choses se résument (le nombre, la
  moyenne, la variance, le maximum), d'autres non : la médiane EXACTE oblige à tout garder (défi 173) ;
- un float n'a que 15 à 16 chiffres significatifs. 1 700 000 016² ≈ 2,89 × 10^18 : ses derniers chiffres
  sont déjà perdus. Soustraire deux grands nombres presque égaux ne laisse que du bruit. On calcule donc sur
  des ÉCARTS (à la moyenne, au dernier instant), qui restent petits ;
- un NaN accepté une fois empoisonne le résumé POUR TOUJOURS (NaN + 1 = NaN). On VALIDE la valeur avant de
  toucher à l'état : une valeur refusée ne doit rien avoir changé ;
- le hasard reproductible : `rng = random.Random(graine)`, créé une fois, et `rng.randrange(...)`. Jamais
  `random.seed()`, qui change le hasard de TOUT le programme ;
- le temps : `(fin - debut).total_seconds() / 86_400` donne des jours AVEC les heures. `.days` tronque
  (23 heures → 0 jour) et `.seconds` oublie les jours (3 jours et 2 heures → 7 200 secondes).

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_28 import *
    >>> connexions_exemple()                         # des secondes depuis 1970 : immenses, presque égales
    [1700000004.0, 1700000007.0, 1700000013.0, 1700000016.0]
    >>> xs = connexions_exemple()
    >>> sum(x * x for x in xs) / 4 - (sum(xs) / 4) ** 2   # la formule du manuel : « aucune dispersion »
    0.0
    >>> flux = UneSeuleFois(generer_montants(5))
    >>> [m for m in flux], flux.lus
    ([58.27, 14.33, 22.02, 41.36, 18.0], 5)
    >>> resultat, pic = pic_memoire(lambda: max(generer_montants(100_000)))
    >>> pic < 50_000                                 # 100 000 montants lus, presque rien gardé
    True

Quand les six défis sont réussis, ils s'assemblent :

    >>> resume = MoyenneVariance()
    >>> for x in connexions_exemple():
    ...     resume.ajouter(x)
    >>> resume.n, resume.moyenne, resume.variance
    (4, 1700000010.0, 30.0)
    >>> alertes_cusum(taux_quotidiens_exemple(), cible=2.0, marge=0.5, seuil=2.0)
    [34]

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites le 170 avant le 175 (la fusion
   travaille sur vos résumés) ; les autres défis sont indépendants.
2. Lancez `python scripts/defis.py --defi 170` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_28.py`.

Les outils communs (`UneSeuleFois`, `pic_memoire`, `connexions_exemple`, `taux_quotidiens_exemple`,
`journal_exemple`, `INSTANT_EXEMPLE`, `generer_montants`, `generer_journal`) sont dans `defis/flux.py`, déjà
importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import heapq  # noqa: F401 — heapq.heappush(tas, x), heapq.heappop(tas) : le plus petit d'abord
import math  # noqa: F401 — math.nan, math.isnan(x), math.isfinite(x)
import numbers  # noqa: F401 — isinstance(x, numbers.Real) : int, float, np.float64… (et bool, attention)
import random  # noqa: F401 — random.Random(graine) : un générateur de hasard à vous
from datetime import datetime, timedelta  # noqa: F401

from defis.flux import (  # noqa: F401
    INSTANT_EXEMPLE,
    UneSeuleFois,
    connexions_exemple,
    generer_journal,
    generer_montants,
    journal_exemple,
    pic_memoire,
    taux_quotidiens_exemple,
)

# ============================================================================ Moyenne et variance


class MoyenneVariance:
    """Défi 170 🟢 — La moyenne et la variance d'un flux, en ne gardant que TROIS nombres.

        >>> mv = MoyenneVariance()
        >>> for x in [4, 7, 13, 16]:
        ...     mv.ajouter(x)
        >>> mv.n, mv.moyenne, mv.variance
        (4, 10.0, 30.0)
        >>> grand = MoyenneVariance()
        >>> for x in connexions_exemple():               # les mêmes écarts, autour de 1,7 milliard
        ...     grand.ajouter(x)
        >>> grand.moyenne, grand.variance
        (1700000010.0, 30.0)
        >>> vide = MoyenneVariance()
        >>> vide.n, vide.moyenne, vide.variance
        (0, nan, nan)

    Ce qu'il faut écrire :

    - `__init__()` : un résumé vide ;
    - `ajouter(x)` : prend en compte une valeur, en O(1) ;
    - `n`, `moyenne`, `variance` : à lire à tout moment (des `@property`, comme ci-dessous, ou de simples
      attributs mis à jour par `ajouter`, au choix).

    Les règles :

    - la variance est celle d'un ÉCHANTILLON : on divise par n - 1, comme pandas `.var()` ;
    - pas de valeur : `moyenne` vaut NaN (`math.nan`) ; moins de deux valeurs : `variance` vaut NaN. Ni 0.0
      (qui ressemble à un vrai résultat) ni ZeroDivisionError ;
    - aucune liste des valeurs : trois nombres, quel que soit le nombre de valeurs (le correcteur mesure la
      mémoire après 300 000 valeurs) ;
    - juste aussi sur `connexions_exemple()` : la formule « moyenne des carrés moins carré de la moyenne »
      y trouve 0.0. Il faut l'algorithme de Welford (ci-dessous) ;
    - `x` qui n'est pas un nombre (None, "12", True) → TypeError ; NaN ou infini → ValueError. Et dans les
      deux cas, le résumé n'a PAS changé ;
    - deux résumés ne partagent rien.

    L'algorithme de Welford (1962), à chaque nouvelle valeur x :

        n += 1
        ecart = x - moyenne              # écart à l'ANCIENNE moyenne
        moyenne += ecart / n
        m2 += ecart * (x - moyenne)      # fois l'écart à la NOUVELLE moyenne
        # variance = m2 / (n - 1)

    Ce que vous apprenez : qu'une formule juste sur le papier peut être fausse en float, et qu'un état
    se protège : on valide, PUIS on modifie.

    Indice : `isinstance(x, bool) or not isinstance(x, numbers.Real)` pour le type, `math.isfinite(x)` pour NaN
    et l'infini, et les trois attributs (`self._n`, `self._moyenne`, `self._m2`) créés dans `__init__`.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def ajouter(self, x) -> None:
        raise NotImplementedError

    @property
    def n(self) -> int:
        raise NotImplementedError

    @property
    def moyenne(self) -> float:
        raise NotImplementedError

    @property
    def variance(self) -> float:
        raise NotImplementedError


# ============================================================================ Alertes CUSUM


def alertes_cusum(valeurs, cible, marge, seuil) -> list[int]:
    """Défi 171 🟢 — Les positions des jours où l'alerte « le churn monte » doit sonner (méthode CUSUM).

    Chaque matin arrive le taux de résiliation de la veille. Un jour à 3,5 % au lieu de 2 % peut être un
    hasard ; deux semaines à 2,75 % ne le sont pas. CUSUM (« cumulative sum », Page, 1954) cumule les
    excès au-delà de `cible + marge` et ne sonne que quand le cumul dépasse `seuil` :

        cumul = 0
        pour chaque jour : cumul = max(0, cumul + valeur - cible - marge)
                           si cumul > seuil : ALERTE ce jour-là, et cumul = 0

        >>> taux = taux_quotidiens_exemple()             # 40 jours, voir defis/flux.py
        >>> taux[8], taux[13], taux[25:28]
        (3.5, None, [2.75, 2.75, 2.75])
        >>> alertes_cusum(taux, cible=2.0, marge=0.5, seuil=2.0)
        [34]
        >>> alertes_cusum([3.0, 3.0, 3.0, 3.0, 3.0, 3.0], cible=2.0, marge=0.5, seuil=2.0)
        [4]

    Dans le dernier exemple, le cumul vaut 0,5 ; 1 ; 1,5 ; 2 ; 2,5 : il DÉPASSE 2 au jour 4 (2 pile ne suffit
    pas), puis repart de 0 (et n'atteint que 0,5 au jour 5).

    Les règles :

    - `valeurs` est un flux : lisez-le UNE fois (pas de `len`, pas d'indice, pas de `list(valeurs)`) ;
    - la position d'un jour compte TOUS les jours, y compris ceux sans donnée ;
    - un jour sans donnée (None ou NaN) ne change pas le cumul et ne sonne jamais ;
    - le `max(0, …)` est indispensable : sans lui, un mois calme creuse le cumul (-12 sur l'exemple), et la
      hausse doit d'abord « rembourser » ce crédit. L'alerte arrive des semaines trop tard, ou jamais ;
    - après une alerte, le cumul repart de 0 : sinon l'alerte sonne TOUS les jours suivants ;
    - `marge` négative, `seuil` négatif ou nul, ou un paramètre NaN → ValueError (attention : `nan < 0` est
      faux, un test `if marge < 0` laisse passer NaN).

    Ce que vous apprenez : un détecteur de rupture qui tient en une variable, et pourquoi chaque détail
    (le max, la remise à zéro, la marge) a une raison d'être.

    Indice : `for jour, x in enumerate(valeurs):` (enumerate AVANT de sauter les jours manquants), et
    `x is None or math.isnan(x)` pour un jour sans donnée.
    """
    raise NotImplementedError


# ============================================================================ Activité récente


class ActiviteRecente:
    """Défi 172 🟡 — Un score d'activité par client, qui perd la moitié de sa valeur tous les `demi_vie_jours`.

    Chaque événement (connexion : poids 1, facture payée : poids 3…) compte d'abord son poids, puis s'efface :
    au bout d'une demi-vie il vaut la moitié, au bout de deux, le quart. Le score d'un client à un instant
    est la somme de ses événements ainsi effacés :

        score(client, instant) = somme des poids × 0.5 ** (âge en jours / demi_vie_jours)

        >>> activite = ActiviteRecente(demi_vie_jours=30)
        >>> for client, moment, poids in journal_exemple():
        ...     activite.ajouter(client, moment, poids)
        >>> [round(activite.score(c, INSTANT_EXEMPLE), 3) for c in ("C1", "C2", "C3", "C9")]
        [8.039, 1.252, 1.945, 0.0]
        >>> jour = ActiviteRecente(demi_vie_jours=1)
        >>> jour.ajouter("C1", datetime(2024, 3, 1, 8, 0))
        >>> jour.score("C1", datetime(2024, 3, 2, 8, 0)), jour.score("C1", datetime(2024, 3, 3, 8, 0))
        (0.5, 0.25)

    C2 a été très actif jusqu'à fin novembre : son score a fondu. C'est exactement le signal de départ que le
    modèle de churn doit voir.

    Ce qu'il faut écrire : `__init__(demi_vie_jours=30.0)`, `ajouter(client, moment, poids=1.0)` et
    `score(client, instant)`. Gardez, par client, DEUX nombres : le score, et l'instant auquel il est exprimé
    (le plus récent de ses événements). Pas la liste des événements : 40 millions par jour ne tiennent pas.

    Les règles :

    - l'âge se compte en jours AVEC les heures : `.total_seconds() / 86_400` (pas `.days`, pas `.seconds`) ;
    - c'est une DEMI-vie : 0.5 ** (jours / demi_vie), pas math.exp(-jours / demi_vie) ;
    - `score` vieillit le score jusqu'à `instant` : un client parti il y a trois mois n'a plus le score de son
      dernier jour ;
    - les événements arrivent parfois EN RETARD (une boutique envoie ses données le lendemain) : un événement
      plus ancien que le dernier connu est accepté, et compte avec son âge réel ;
    - `score(client, instant)` avec un `instant` antérieur au dernier événement reçu POUR CE CLIENT →
      ValueError. Les événements fondus ensemble ne se séparent plus : compter ceux d'après `instant`, ce
      serait utiliser le futur (la fuite de données du défi 55) ;
    - un client jamais vu a un score de 0.0 (pas KeyError) ;
    - `demi_vie_jours` nulle, négative ou NaN → ValueError ; `poids` NaN → ValueError, et rien n'a changé ;
    - deux objets ne partagent rien.

    Ce que vous apprenez : la moyenne mobile exponentielle quand les événements ne sont pas régulièrement
    espacés, et les pièges de `timedelta`.

    Indice : `facteur = 0.5 ** (jours / self.demi_vie_jours)`. Un nouvel événement : `score * facteur(ref →
    moment) + poids`. Un événement en retard : c'est LUI qu'on vieillit, `score + poids * facteur(moment → ref)`,
    et la référence ne bouge pas.
    """

    def __init__(self, demi_vie_jours: float = 30.0) -> None:
        raise NotImplementedError

    def ajouter(self, client, moment: datetime, poids: float = 1.0) -> None:
        raise NotImplementedError

    def score(self, client, instant: datetime) -> float:
        raise NotImplementedError


# ============================================================================ Médiane courante


class MedianeCourante:
    """Défi 173 🟡 — La médiane du flux, disponible après chaque valeur, en O(log n) par valeur.

        >>> m = MedianeCourante()
        >>> medianes = []
        >>> for x in [5, 1, 9, 3, 7]:
        ...     m.ajouter(x)
        ...     medianes.append(m.mediane())
        >>> medianes                                     # [5], [1, 5], [1, 5, 9], [1, 3, 5, 9], …
        [5.0, 3.0, 5.0, 4.0, 5.0]
        >>> len(m), MedianeCourante().mediane()
        (5, nan)

    La médiane résiste aux valeurs extrêmes (une facture de 40 000 € ne la déplace presque pas, alors qu'elle
    tire la moyenne) : c'est elle qu'on affiche pour « la facture typique ».

    Ce qu'il faut écrire : `__init__()`, `ajouter(x)`, `mediane()` et `__len__()`.

    Les règles :

    - nombre pair de valeurs : la moyenne des DEUX valeurs du milieu ;
    - pas de valeur : `mediane()` renvoie NaN ;
    - pas de tri à chaque valeur (`sorted`, `.sort()`, `statistics.median`) : sur 20 000 valeurs, c'est des
      centaines de millions de comparaisons. Le correcteur les COMPTE ;
    - None ou un texte → TypeError ; NaN → ValueError (NaN n'est ni plus petit ni plus grand que rien : il
      casserait l'ordre en silence), et rien n'a changé ;
    - deux objets ne partagent rien.

    La méthode : deux tas (`heapq`). `bas` garde la moitié basse et donne son MAXIMUM en O(1) ; `haut` garde
    la moitié haute et donne son minimum. `heapq` ne fait que des tas min : dans `bas`, rangez `-x`. Après
    chaque ajout, rééquilibrez pour que `bas` ait autant de valeurs que `haut`, ou une de plus. La médiane
    est alors `-bas[0]` (impair) ou `(-bas[0] + haut[0]) / 2` (pair).

    Ce que vous apprenez : la question d'entretien « Find Median from Data Stream », et qu'un résumé exact
    n'existe pas toujours : pour la médiane, il faut garder les valeurs (mais pas les retrier).

    Indice : ajoutez x dans `bas` si `bas` est vide ou si `x <= -bas[0]`, sinon dans `haut` ; puis, si l'un
    est trop gros, `heappop` chez lui et `heappush` chez l'autre (sans oublier le signe moins).
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def ajouter(self, x) -> None:
        raise NotImplementedError

    def mediane(self) -> float:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


# ============================================================================ Échantillon réservoir


def echantillon_reservoir(flux, k: int, graine=None) -> list:
    """Défi 174 🟡 — k valeurs du flux tirées au hasard, chacune avec la même chance, en UNE lecture.

    Le service qualité veut rappeler 1 000 clients tirés au hasard parmi ceux qui ont appelé aujourd'hui.
    On ne sait pas combien appelleront (peut-être 30 000), et on ne veut pas tous les garder pour tirer à la
    fin. L'échantillonnage « réservoir » (algorithme R, Vitter, 1985) :

        les k premières valeurs entrent dans l'échantillon ;
        la valeur numéro i (en comptant à partir de 0), pour i >= k :
            j = rng.randrange(i + 1)          # un entier entre 0 et i INCLUS
            si j < k : elle remplace l'échantillon[j]

        >>> echantillon_reservoir(range(1_000_000), 5, graine=28)
        [829384, 68297, 700653, 286138, 663301]
        >>> echantillon_reservoir(UneSeuleFois(["a", "b"]), 5, graine=28)   # moins de k valeurs : toutes
        ['a', 'b']
        >>> echantillon_reservoir(range(10), 0, graine=28)
        []

    Les règles :

    - chaque valeur du flux a la MÊME probabilité k / n d'être choisie, qu'elle arrive en premier ou en
      dernier. Le correcteur tire 20 000 échantillons et compte ;
    - une seule lecture du flux, et une mémoire qui dépend de k, pas de la longueur du flux (pas de
      `list(flux)`, même suivi de `rng.sample`) ;
    - moins de k valeurs : toutes, dans leur ordre d'arrivée ;
    - le hasard : `rng = random.Random(graine)`, créé UNE fois avant la boucle. Même graine, même échantillon.
      Jamais `random.seed(...)` ni `np.random.seed(...)` : le correcteur vérifie que le hasard du reste du
      programme n'a pas bougé ;
    - `k` négatif → ValueError ; `k` qui n'est pas un entier (2.5, "10", True) → TypeError.

    Ce que vous apprenez : un algorithme probabiliste court, sa preuve (une récurrence de trois lignes, dans
    le corrigé), et qu'une erreur d'une unité (`randrange(i)`) le rend biaisé sans jamais planter.

    Indice : `for i, x in enumerate(flux):`, puis `if i < k: echantillon.append(x)`, sinon le tirage.
    """
    raise NotImplementedError


# ============================================================================ Fusionner deux résumés


def fusionner(a: MoyenneVariance, b: MoyenneVariance) -> MoyenneVariance:
    """Défi 175 🔴 — Le résumé de TOUTES les valeurs vues par `a` et par `b`, sans revoir une seule valeur.

    Chaque nuit, dix machines lisent chacune un dixième des factures et en font un `MoyenneVariance` (défi 170).
    Il faut le résumé de l'ensemble : les valeurs ne sont plus là, seulement les dix résumés.

        >>> matin, soir = MoyenneVariance(), MoyenneVariance()
        >>> for x in [4, 7]:
        ...     matin.ajouter(x)
        >>> for x in [13, 16, 40]:
        ...     soir.ajouter(x)
        >>> journee = fusionner(matin, soir)
        >>> journee.n, journee.moyenne, journee.variance       # comme si un seul résumé avait tout vu
        (5, 16.0, 202.5)
        >>> matin.n, soir.n                                    # a et b n'ont pas changé
        (2, 3)

    Les règles :

    - le résultat est un NOUVEL objet `MoyenneVariance` ; `a` et `b` ne changent pas ;
    - il donne les mêmes `n`, `moyenne` et `variance` qu'un seul résumé qui aurait reçu toutes les valeurs ;
    - et il continue de fonctionner : on peut lui `ajouter` d'autres valeurs, ou le fusionner à nouveau ;
    - un côté vide : le résultat est une copie de l'autre (pas de division par zéro) ;
    - juste aussi autour de 1,7 milliard (pas de « somme des carrés »).

    Les formules (Chan, Golub et LeVeque, 1979), avec m2 la somme des carrés des écarts de chaque résumé :

        n = na + nb
        ecart = moyenne_b - moyenne_a
        moyenne = moyenne_a + ecart * nb / n          # pondérée : pas (moyenne_a + moyenne_b) / 2
        m2 = m2_a + m2_b + ecart ** 2 * na * nb / n   # le dernier terme : l'écart ENTRE les deux morceaux

    Ce que vous apprenez : ce qui rend un résumé « fusionnable » (et donc distribuable sur plusieurs
    machines, comme dans Spark ou un `groupby` sur des données trop grosses), et pourquoi on ne fait jamais
    la moyenne de deux moyennes ou de deux variances.

    Indice : `resultat = MoyenneVariance()`, puis remplissez ses attributs internes (ceux de VOTRE défi 170)
    avec les formules. La fonction est dans le même fichier que la classe : elle a le droit de les toucher.
    """
    raise NotImplementedError

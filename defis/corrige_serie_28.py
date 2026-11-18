"""Série 28 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_28.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours une
valeur refusée APRÈS avoir modifié l'état, un `.days` qui oublie les heures, ou un `randrange(i)` qui
devait être `randrange(i + 1)`.

Fil rouge de la série : un flux se lit une fois, et on n'en garde qu'un RÉSUMÉ. Quatre habitudes :

1. chaque valeur met à jour le résumé en O(1) (ou O(log n)) : jamais « garder la liste et tout recalculer » ;
2. on VALIDE une valeur avant de toucher à l'état. Un NaN accepté une fois empoisonne le résumé pour
   toujours, et une valeur refusée à moitié enregistrée le fausse en silence ;
3. on calcule sur des ÉCARTS (à la moyenne, au dernier instant), jamais sur de grands nombres presque égaux :
   1 700 000 016² et 1 700 000 004² ne diffèrent que dans des chiffres que le float n'a pas ;
4. un résumé bien choisi se FUSIONNE : dix machines résument dix morceaux, puis on réunit les résumés.
   C'est l'idée de MapReduce, de Spark, et de `groupby().agg()` sur des données qui ne tiennent pas en mémoire.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import heapq
import math
import numbers
import random
from datetime import datetime

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


def _nombre(x, nom: str = "la valeur") -> float:
    """x en float, après avoir vérifié que c'est un VRAI nombre, fini. Sinon TypeError ou ValueError."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        # bool est un int pour Python (True + True == 2) : on le refuse exprès, ce n'est pas une mesure.
        raise TypeError(f"{nom} doit être un nombre, pas {type(x).__name__} ({x!r})")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{nom} doit être un nombre fini, pas {x!r} : il empoisonnerait le résumé pour toujours")
    return x


# ============================================================================ Défi 170 : moyenne et variance


class MoyenneVariance:
    """La moyenne et la variance d'un flux, en gardant TROIS nombres (algorithme de Welford, 1962)."""

    def __init__(self) -> None:
        # Des attributs D'OBJET, créés dans __init__ : chaque résumé a les siens. Écrits au niveau de la classe,
        # ils seraient partagés par tous les objets (pour des nombres, `self._n += 1` crée un attribut d'objet
        # et ça « marche » ; pour une liste, `self._valeurs.append` modifierait celle de la classe).
        self._n = 0
        self._moyenne = 0.0
        self._m2 = 0.0  # somme des carrés des écarts à la moyenne courante

    def ajouter(self, x) -> None:
        x = _nombre(x)  # 1. valider : en cas d'erreur, RIEN n'a encore changé
        # 2. mettre à jour. Tout se calcule sur des ÉCARTS (x - moyenne), petits même quand x vaut 1,7 milliard.
        self._n += 1
        ecart_avant = x - self._moyenne
        self._moyenne += ecart_avant / self._n
        ecart_apres = x - self._moyenne  # avec la NOUVELLE moyenne
        self._m2 += ecart_avant * ecart_apres
        # La formule « somme des carrés » (Σx² - n·moyenne²) soustrait deux nombres de l'ordre de 10^19 qui ne
        # diffèrent que de quelques unités : le float n'a que 15-16 chiffres, et le résultat est du bruit
        # (0.0 sur `connexions_exemple()`, voire négatif). Welford n'additionne que des produits d'écarts.

    @property
    def n(self) -> int:
        return self._n

    @property
    def moyenne(self) -> float:
        # Pas de valeur : pas de moyenne. NaN (comme pandas : pd.Series([], dtype=float).mean()), pas 0.0 qui
        # ressemblerait à une vraie moyenne, ni une ZeroDivisionError qui ferait planter le tableau de bord.
        return self._moyenne if self._n else math.nan

    @property
    def variance(self) -> float:
        # Variance D'ÉCHANTILLON : on divise par n - 1, comme pandas `.var()` et `statistics.variance`.
        # (`np.var` divise par n par défaut : `ddof=0`.) Avec une seule valeur, elle n'est pas définie : NaN.
        return self._m2 / (self._n - 1) if self._n >= 2 else math.nan

    def __repr__(self) -> str:
        return f"MoyenneVariance(n={self.n}, moyenne={self.moyenne!r}, variance={self.variance!r})"


# ============================================================================ Défi 171 : alertes CUSUM


def _absente(x) -> bool:
    # None, ou NaN (float, np.float64… : NaN est le seul nombre différent de lui-même).
    return x is None or (isinstance(x, numbers.Real) and x != x)


def alertes_cusum(valeurs, cible, marge, seuil) -> list[int]:
    for nom, v in (("cible", cible), ("marge", marge), ("seuil", seuil)):
        # `marge < 0` est faux pour NaN : sans ce test, une marge NaN passerait et plus rien ne sonnerait.
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
            raise ValueError(f"{nom} doit être un nombre fini, pas {v!r}")
    if marge < 0:
        raise ValueError(f"marge doit être positive ou nulle, pas {marge!r}")
    if seuil <= 0:
        raise ValueError(f"seuil doit être strictement positif, pas {seuil!r} : sinon chaque jour normal sonnerait")

    cumul = 0.0
    alertes = []
    # enumerate AVANT de sauter les jours manquants : la position d'un jour, c'est sa date. Filtrer d'abord
    # décalerait toutes les alertes suivantes.
    for jour, x in enumerate(valeurs):  # une seule lecture : ni len(valeurs), ni valeurs[i], ni list(valeurs)
        if _absente(x):
            continue  # un jour sans donnée ne prouve rien : le cumul ne bouge pas
        # max(0, …) : un mois calme ne donne AUCUN crédit. Sans lui, 25 jours calmes creusent le cumul à -12, et
        # la hausse devrait d'abord le « rembourser » : l'alerte arriverait des semaines trop tard.
        cumul = max(0.0, cumul + x - cible - marge)
        if cumul > seuil:  # « dépasse » : strictement
            alertes.append(jour)
            cumul = 0.0  # on repart de zéro : sinon l'alerte sonnerait TOUS les jours suivants
    return alertes


# ============================================================================ Défi 172 : activité récente


class ActiviteRecente:
    """Un score d'activité par client, qui perd la moitié de sa valeur tous les `demi_vie_jours`.

    Pour chaque client, on garde DEUX nombres : le score, et l'instant auquel il est exprimé (le plus récent
    de ses événements). Le score à un autre instant s'en déduit : on le multiplie par 0,5 ** (jours / demi-vie).
    """

    def __init__(self, demi_vie_jours: float = 30.0) -> None:
        demi_vie = _nombre(demi_vie_jours, "demi_vie_jours")
        if demi_vie <= 0:
            raise ValueError(f"demi_vie_jours doit être strictement positive, pas {demi_vie_jours!r}")
        self.demi_vie_jours = demi_vie
        self._etat: dict = {}  # client -> (score, instant de référence = son événement le plus récent)

    def _facteur(self, depuis: datetime, jusqu_a: datetime) -> float:
        # total_seconds() : les jours ET les heures. `.days` tronque (23 h → 0 jour) et `.seconds` ne garde que
        # les secondes de la dernière journée (3 jours et 2 heures → 7 200 secondes).
        jours = (jusqu_a - depuis).total_seconds() / 86_400
        # 0.5 ** (jours / demi-vie) : au bout d'une demi-vie, le poids est divisé par 2. (math.exp(-jours / d)
        # le diviserait par e ≈ 2,718 : c'est une « constante de temps », pas une demi-vie.)
        return 0.5 ** (jours / self.demi_vie_jours)

    def ajouter(self, client, moment: datetime, poids: float = 1.0) -> None:
        poids = _nombre(poids, "poids")  # valider avant de toucher à l'état
        if client not in self._etat:
            self._etat[client] = (poids, moment)
            return
        score, reference = self._etat[client]
        if moment >= reference:
            # Le cas courant : on vieillit le score jusqu'au nouvel événement, puis on ajoute son poids.
            self._etat[client] = (score * self._facteur(reference, moment) + poids, moment)
        else:
            # Un événement EN RETARD : c'est lui qu'on vieillit jusqu'à la référence. On ne recule pas la
            # référence : elle doit rester l'événement le plus récent (voir score()).
            self._etat[client] = (score + poids * self._facteur(moment, reference), reference)

    def score(self, client, instant: datetime) -> float:
        if client not in self._etat:
            return 0.0  # jamais vu : aucune activité (pas KeyError : le tableau de bord demande pour tout le monde)
        score, reference = self._etat[client]
        if instant < reference:
            # On a fondu les événements ensemble : impossible de retirer ceux qui sont APRÈS `instant`. Les
            # compter quand même, ce serait utiliser le futur (la fuite du défi 55). On le dit.
            raise ValueError(f"le client {client!r} a un événement le {reference:%d/%m/%Y %H:%M}, après "
                             f"{instant:%d/%m/%Y %H:%M} : son score à cet instant n'est plus disponible")
        # On vieillit jusqu'à `instant` : un client parti il y a 3 mois n'a plus le score de son dernier jour.
        return score * self._facteur(reference, instant)


# ============================================================================ Défi 173 : médiane courante


class MedianeCourante:
    """La médiane d'un flux, avec deux tas : la moitié basse (tas MAX) et la moitié haute (tas min)."""

    def __init__(self) -> None:
        # heapq ne sait faire que des tas MIN : pour un tas max, on y range les valeurs avec un signe moins.
        self._bas: list = []   # -x pour chaque x de la moitié basse : -self._bas[0] est son MAXIMUM
        self._haut: list = []  # la moitié haute : self._haut[0] est son minimum

    def ajouter(self, x) -> None:
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise TypeError(f"la valeur doit être un nombre, pas {type(x).__name__} ({x!r})")
        if x != x:
            # NaN n'est ni plus petit ni plus grand que rien : rangé dans un tas, il casserait l'ordre en silence.
            raise ValueError("NaN n'a pas de place dans un classement : valeur refusée")
        # 1. ranger x du bon côté (O(log n))
        if not self._bas or x <= -self._bas[0]:
            heapq.heappush(self._bas, -x)
        else:
            heapq.heappush(self._haut, x)
        # 2. rééquilibrer : le bas a autant de valeurs que le haut, ou une de plus (O(log n))
        if len(self._bas) > len(self._haut) + 1:
            heapq.heappush(self._haut, -heapq.heappop(self._bas))
        elif len(self._haut) > len(self._bas):
            heapq.heappush(self._bas, -heapq.heappop(self._haut))

    def mediane(self) -> float:
        if not self._bas:
            return math.nan  # comme pandas : pd.Series([], dtype=float).median()
        if len(self._bas) > len(self._haut):
            return float(-self._bas[0])  # nombre impair : la valeur du milieu
        return (-self._bas[0] + self._haut[0]) / 2  # nombre pair : la moyenne des DEUX valeurs du milieu

    def __len__(self) -> int:
        return len(self._bas) + len(self._haut)


# ============================================================================ Défi 174 : échantillon réservoir


def echantillon_reservoir(flux, k: int, graine=None) -> list:
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise TypeError(f"k doit être un entier, pas {type(k).__name__} ({k!r})")
    if k < 0:
        raise ValueError(f"k doit être positif ou nul, pas {k}")
    # UN générateur, créé UNE fois, à nous : jamais random.seed(), qui changerait le hasard de tout le programme.
    rng = random.Random(graine)
    echantillon: list = []
    for i, x in enumerate(flux):  # i = nombre de valeurs vues AVANT x
        if i < k:
            echantillon.append(x)  # les k premières entrent toutes
        else:
            # x est la (i + 1)-ième valeur : elle doit être gardée avec une probabilité k / (i + 1).
            # randrange(i + 1) tire un entier entre 0 et i INCLUS : i + 1 possibilités, dont k sont < k.
            # (randrange(i) n'en a que i : la (k + 1)-ième valeur entrerait à coup sûr.)
            j = rng.randrange(i + 1)
            if j < k:
                echantillon[j] = x  # elle prend une place tirée au hasard (pas la plus ancienne)
    # Pourquoi c'est juste : par récurrence, après i valeurs, chacune est dans l'échantillon avec la
    # probabilité k / i. La suivante entre avec k / (i + 1) ; une valeur déjà là en sort avec
    # (k / (i + 1)) × (1 / k) = 1 / (i + 1). Elle reste donc avec (k / i) × (i / (i + 1)) = k / (i + 1).
    return echantillon


# ============================================================================ Défi 175 : fusionner deux résumés


def fusionner(a: MoyenneVariance, b: MoyenneVariance) -> MoyenneVariance:
    resultat = MoyenneVariance()  # un NOUVEL objet : a et b restent intacts (on peut fusionner a avec c ensuite)
    for source in (a, b):
        if source._n == 0:
            autre = b if source is a else a
            resultat._n, resultat._moyenne, resultat._m2 = autre._n, autre._moyenne, autre._m2
            return resultat  # un côté vide : le résultat est une COPIE de l'autre (et rien à diviser par 0)
    n = a._n + b._n
    ecart = b._moyenne - a._moyenne
    # La moyenne pondérée par les effectifs (pas (a + b) / 2 : 1 000 valeurs pèsent plus que 10).
    resultat._moyenne = a._moyenne + ecart * b._n / n
    # Chan, Golub et LeVeque (1979) : les écarts INTERNES de chaque morceau, plus l'écart ENTRE les morceaux.
    # Sans ce dernier terme, deux morceaux de moyennes 10 et 1 000 sans dispersion interne donneraient une
    # variance nulle.
    resultat._m2 = a._m2 + b._m2 + ecart * ecart * a._n * b._n / n
    resultat._n = n
    return resultat
    # Ce que ça permet : chaque machine résume sa part du flux, puis on fusionne les résumés deux à deux, dans
    # n'importe quel ordre (la fusion est associative, aux arrondis près). Les valeurs, elles, ne voyagent pas.


# ============================================================================ tests (comme dans tests/)


def test_welford_ne_perd_pas_ses_chiffres():
    mv = MoyenneVariance()
    for x in connexions_exemple():
        mv.ajouter(x)
    assert (mv.n, mv.moyenne, mv.variance) == (4, 1_700_000_010.0, 30.0)


def test_une_valeur_refusee_ne_change_rien():
    mv = MoyenneVariance()
    mv.ajouter(1.0)
    for mauvaise in (math.nan, math.inf):
        try:
            mv.ajouter(mauvaise)
        except ValueError:
            pass
    assert mv.n == 1 and mv.moyenne == 1.0


def test_cusum_ignore_le_pic_et_voit_la_hausse():
    assert alertes_cusum(taux_quotidiens_exemple(), cible=2.0, marge=0.5, seuil=2.0) == [34]


def test_activite_recente_vieillit_jusqu_a_l_instant():
    activite = ActiviteRecente(demi_vie_jours=1)
    activite.ajouter("C1", datetime(2024, 3, 1, 8, 0))
    assert activite.score("C1", datetime(2024, 3, 2, 8, 0)) == 0.5
    assert activite.score("C1", datetime(2024, 3, 1, 20, 0)) == 0.5 ** 0.5   # 12 heures : une demi-journée
    assert activite.score("inconnu", datetime(2024, 3, 2)) == 0.0


def test_mediane_courante_suit_le_flux():
    m = MedianeCourante()
    medianes = []
    for x in [5, 1, 9, 3, 7]:
        m.ajouter(x)
        medianes.append(m.mediane())
    assert medianes == [5, 3, 5, 4, 5]


def test_reservoir_garde_k_valeurs_et_tout_si_le_flux_est_court():
    assert len(echantillon_reservoir(range(1000), 10, graine=1)) == 10
    assert echantillon_reservoir(range(3), 10, graine=1) == [0, 1, 2]
    assert echantillon_reservoir(range(1000), 10, graine=7) == echantillon_reservoir(range(1000), 10, graine=7)


def test_fusionner_donne_le_resume_de_tout():
    a, b, tout = MoyenneVariance(), MoyenneVariance(), MoyenneVariance()
    for i, x in enumerate(generer_montants(1000, graine=3)):
        (a if i < 100 else b).ajouter(x)
        tout.ajouter(x)
    r = fusionner(a, b)
    assert r.n == tout.n and math.isclose(r.moyenne, tout.moyenne) and math.isclose(r.variance, tout.variance)
    assert a.n == 100 and b.n == 900

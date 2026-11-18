"""Outils de la série 26, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Des données d'exemple tirées de la vie du projet, et trois outils pour VÉRIFIER une réponse autrement
qu'en relisant son code :

- `p_valeur_exacte(a, b)` : la p-valeur d'un test de permutation calculée en essayant TOUS les partages
  possibles des clients entre les deux groupes (seulement pour de tout petits groupes) ;
- `puissance_simulee(n, p_temoin, p_offre)` : sur des milliers d'expériences simulées, la part où un
  test à 5 % voit vraiment la différence ;
- `auc(y, score)` : l'aire sous la courbe ROC, calculée par les rangs (rapide : utile pour les
  centaines de tirages du défi 162).

Chaque appel renvoie de NOUVELLES données : modifiez-les autant que vous voulez.

    >>> from defis.serie_26 import *
    >>> attentes = attentes_support()
    >>> len(attentes), float(np.median(attentes)), round(float(np.mean(attentes)), 1)
    (41, 7.5, 10.8)
"""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

EFFET_REEL = -0.04
"""Dans `campagne_exemple`, l'offre fait VRAIMENT baisser le churn de 4 points (c'est la réponse à trouver)."""


def attentes_support() -> np.ndarray:
    """41 durées d'attente au support (en minutes) mesurées un lundi : beaucoup de courtes, quelques très longues.

    La moyenne (10,8 min) est tirée vers le haut par quelques clients qui ont attendu une demi-heure ou plus ;
    la médiane (7,5 min) décrit mieux le client « ordinaire ». Mais aucune formule simple ne donne
    l'intervalle de confiance d'une médiane : c'est le travail du bootstrap (défi 158).

        >>> attentes = attentes_support()
        >>> float(attentes.min()), float(attentes.max())
        (1.0, 61.5)
    """
    return np.array([2.0, 1.0, 15.5, 21.0, 18.0, 5.0, 6.5, 5.5, 18.0, 13.0, 7.5, 11.5, 18.0, 9.0, 3.5, 6.5, 5.0,
                     2.5, 1.5, 8.0, 3.5, 9.5, 4.0, 12.5, 13.0, 9.0, 3.0, 1.5, 31.5, 6.5, 29.5, 6.0, 3.0, 7.0, 3.0,
                     7.0, 22.0, 13.0, 61.5, 9.0, 9.0])


def notes_deux_scripts() -> tuple[list[int], list[int]]:
    """Notes de satisfaction (de 1 à 10) après un appel au support : l'ancien script, puis le nouveau.

    14 clients ont eu l'ancien script et 12 le nouveau (tirés au sort). Le nouveau fait-il mieux, ou
    est-ce le hasard des clients tombés dans chaque groupe ? (défi 159)

        >>> ancien, nouveau = notes_deux_scripts()
        >>> len(ancien), len(nouveau), round(sum(nouveau) / 12 - sum(ancien) / 14, 2)
        (14, 12, 1.19)
    """
    ancien = [5, 5, 7, 5, 6, 6, 5, 10, 3, 7, 8, 8, 6, 5]
    nouveau = [8, 7, 6, 9, 8, 5, 8, 9, 7, 6, 9, 6]
    return ancien, nouveau


def p_valeurs_segments() -> dict[str, float]:
    """Le test A/B de l'offre, refait segment par segment : 20 segments, 20 p-valeurs.

    Cinq segments ont p < 0,05. Si l'offre n'avait AUCUN effet nulle part, on en attendrait déjà un en
    moyenne (20 × 5 %), et au moins un dans 64 % des cas (1 - 0,95²⁰). Lesquels croire ? (défi 161)

        >>> p = p_valeurs_segments()
        >>> len(p), sorted(s for s, v in p.items() if v < 0.05)
        (20, ['fibre_mensuel', 'jeunes_fibre', 'mensuel_cheque', 'region_est', 'seniors_adsl'])
    """
    return {
        "mensuel_cheque": 0.0004, "fibre_mensuel": 0.0021, "jeunes_fibre": 0.0063, "seniors_adsl": 0.031,
        "region_est": 0.044, "annuel_carte": 0.071, "adsl_mensuel": 0.12, "region_nord": 0.16,
        "biannuel": 0.22, "fibre_annuel": 0.27, "prelevement": 0.33, "seniors_fibre": 0.38,
        "region_ouest": 0.45, "sans_internet": 0.52, "jeunes_adsl": 0.58, "region_sud": 0.64,
        "virement": 0.71, "gros_clients": 0.79, "petits_clients": 0.86, "nouveaux": 0.93,
    }


def scores_deux_modeles(n: int = 600, graine: int = 0) -> pd.DataFrame:
    """Le même mois de test, noté par le modèle en production (`champion`) et par le nouveau (`challenger`).

    Colonnes : `churn` (0/1, ce qui s'est vraiment passé), `champion` et `challenger` (les probabilités
    prédites). Les deux modèles se trompent souvent sur les MÊMES clients : leurs scores sont très
    corrélés. C'est ce qui rend la comparaison APPARIÉE (mêmes clients tirés pour les deux) bien plus
    précise que deux intervalles séparés (défi 162).

        >>> df = scores_deux_modeles()
        >>> df.shape, sorted(df.columns)
        ((600, 3), ['challenger', 'champion', 'churn'])
        >>> round(auc(df["churn"], df["champion"]), 3), round(auc(df["churn"], df["challenger"]), 3)
        (0.76, 0.778)
    """
    rng = np.random.default_rng(graine)
    signal = rng.normal(0, 1.0, n)
    detail = rng.normal(0, 1.0, n)               # ce que seul le challenger voit (un peu)
    risque = 1.6 * signal + 0.5 * detail - 0.9
    churn = (rng.uniform(size=n) < 1 / (1 + np.exp(-risque))).astype(int)
    bruit = rng.normal(0, 0.35, n)               # le bruit commun aux deux modèles : ils se trompent ensemble
    champion = 1 / (1 + np.exp(-(1.5 * signal + bruit - 0.9 + rng.normal(0, 0.15, n))))
    challenger = 1 / (1 + np.exp(-(1.5 * signal + 0.25 * detail + bruit - 0.9 + rng.normal(0, 0.15, n))))
    return pd.DataFrame({"churn": churn, "champion": champion.round(4), "challenger": challenger.round(4)},
                        index=pd.Index([f"C{i:05d}" for i in range(1, n + 1)], name="client"))


def campagne_exemple(n: int = 4000, graine: int = 84) -> pd.DataFrame:
    """Le test A/B de l'offre de rétention : la moitié des clients (tirés au sort) a reçu l'offre.

    Colonnes :

    - `score_avant` : la probabilité de départ donnée par le modèle LA VEILLE de la campagne (donc
      impossible à influencer par l'offre). Les nouveaux clients n'ont pas encore de score : NaN ;
    - `offre` : True si le client a reçu l'offre (tiré au sort), False s'il est dans le groupe témoin ;
    - `churn` : 1 si le client est parti dans les 3 mois qui ont suivi.

    L'offre fait vraiment baisser le churn de 4 points (`EFFET_REEL`). Le score explique une partie de
    qui part : s'en servir réduit le bruit de la mesure (défi 163).

        >>> df = campagne_exemple()
        >>> df.shape, int(df["offre"].sum()), int(df["score_avant"].isna().sum())
        ((4000, 3), 2000, 321)
    """
    rng = np.random.default_rng(graine)
    proba = rng.beta(1.2, 2.8, n)                     # le vrai risque de chaque client (le modèle est bien calibré)
    offre = rng.permutation(np.arange(n) < n // 2)    # exactement la moitié, tirée au sort
    nouveau = rng.uniform(size=n) < 0.08
    churn = (rng.uniform(size=n) < np.clip(proba + EFFET_REEL * offre, 0, 1)).astype(int)
    score = np.where(nouveau, np.nan, proba.round(3))
    return pd.DataFrame({"score_avant": score, "offre": offre, "churn": churn},
                        index=pd.Index([f"C{i:05d}" for i in range(1, n + 1)], name="client"))


# ============================================================================ outils de vérification


def p_valeur_exacte(a, b) -> float:
    """La p-valeur EXACTE du test de permutation (bilatéral, sur l'écart des moyennes b - a).

    Essaie tous les partages possibles des clients en un groupe de `len(a)` et un groupe de `len(b)`,
    et renvoie la part de ceux dont l'écart (en valeur absolue) atteint l'écart observé. Le partage
    réellement observé en fait partie : la p-valeur n'est jamais 0. Limité à 200 000 partages.

        >>> p_valeur_exacte([1, 2, 3], [4, 5, 6])   # 2 partages sur 20 sont aussi extrêmes : l'observé et son miroir
        0.1
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    tous = np.concatenate([a, b])
    n, k = len(tous), len(a)
    if k == 0 or len(b) == 0:
        raise ValueError("chaque groupe doit contenir au moins une valeur")
    if math.comb(n, k) > 200_000:
        raise ValueError(f"{math.comb(n, k)} partages possibles : trop pour tous les essayer (200 000 au plus)")
    observe = abs(b.mean() - a.mean())
    total = tous.sum()
    extremes = partages = 0
    for groupe_a in itertools.combinations(range(n), k):
        somme_a = tous[list(groupe_a)].sum()
        ecart = (total - somme_a) / (n - k) - somme_a / k
        extremes += abs(ecart) >= observe - 1e-9
        partages += 1
    return float(extremes / partages)


def puissance_simulee(n: int, p_temoin: float, p_offre: float, alpha: float = 0.05,
                      n_experiences: int = 4000, graine: int = 0) -> float:
    """Sur `n_experiences` tests A/B simulés avec `n` clients par groupe, la part où le test voit la différence.

    Chaque expérience tire le nombre de départs de chaque groupe (loi binomiale), puis fait le test z de
    deux proportions (bilatéral, variance commune) au niveau `alpha`. C'est la PUISSANCE : la chance de
    conclure « l'offre marche » quand elle marche vraiment (de p_temoin à p_offre).

        >>> 0.78 < puissance_simulee(1251, 0.30, 0.25) < 0.82   # ≈ 0,80
        True
        >>> 0.26 < puissance_simulee(300, 0.30, 0.25) < 0.30    # 4 fois moins de clients : un test sur quatre voit l'effet
        True

    Le résultat exact (0,8035 ou 0,8058…) dépend de la version de NumPy : à graine égale, NumPy ne
    garantit pas les mêmes tirages d'une version à l'autre. On vérifie donc un intervalle (± 3 écarts-types
    de la simulation, √(0,8 × 0,2 / 4 000) ≈ 0,006), jamais une valeur arrondie à 2 décimales.
    """
    rng = np.random.default_rng(graine)
    departs_t = rng.binomial(n, p_temoin, n_experiences)
    departs_o = rng.binomial(n, p_offre, n_experiences)
    commun = (departs_t + departs_o) / (2 * n)
    erreur = np.sqrt(commun * (1 - commun) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(erreur > 0, (departs_o - departs_t) / n / erreur, 0.0)
    critique = NormalDist().inv_cdf(1 - alpha / 2)
    return float(np.mean(np.abs(z) > critique))


def auc(y, score) -> float:
    """L'aire sous la courbe ROC : la chance qu'un client parti ait un score plus haut qu'un client resté.

    Calculée par les rangs (statistique de Mann-Whitney), les ex aequo comptant pour moitié : le même
    résultat que `sklearn.metrics.roc_auc_score`, environ dix fois plus vite sur de petits tableaux.
    Lève ValueError si `y` ne contient qu'une seule classe (l'AUC n'existe pas).

        >>> auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        0.75
    """
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    positifs = int((y == 1).sum())
    negatifs = len(y) - positifs
    if positifs == 0 or negatifs == 0:
        raise ValueError("l'AUC demande au moins un client parti (1) et un client resté (0)")
    ordre = np.argsort(score, kind="mergesort")
    tries = score[ordre]
    # rang moyen de chaque valeur (les ex aequo partagent leur rang) : 1-based
    debut = np.r_[True, tries[1:] != tries[:-1]]
    groupe = np.cumsum(debut) - 1
    premiers = np.flatnonzero(debut)
    tailles = np.diff(np.r_[premiers, len(tries)])
    rang_moyen = premiers + (tailles + 1) / 2
    rangs = np.empty(len(score))
    rangs[ordre] = rang_moyen[groupe]
    somme = rangs[y == 1].sum()
    return float((somme - positifs * (positifs + 1) / 2) / (positifs * negatifs))

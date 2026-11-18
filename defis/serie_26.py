"""Série 26 — Conclure sans se tromper : l'incertitude, du bootstrap au test A/B à variance réduite.

🎲 VOUS VOULEZ DÉVELOPPER ? UN JOUR, ON VOUS DEMANDERA : « C'EST SIGNIFICATIF ? »

Le nouveau script du support a une note moyenne plus haute de 1,2 point. Le challenger gagne 0,018
d'AUC. L'offre de rétention a fait baisser le churn de 1,9 point. Cinq segments sur vingt réagissent à
l'offre. Chacune de ces phrases peut être vraie… ou n'être que le hasard des clients tombés dans chaque
groupe. Celle ou celui qui sait faire la différence (et le prouver) est écouté·e. Cette série vous fait
écrire les outils pour cela :

    158 intervalle_bootstrap   l'intervalle de confiance de N'IMPORTE QUELLE statistique (une médiane…)
    159 p_valeur_permutation   « l'écart est-il plus grand que ce que le hasard produit ? », sans formule
    160 taille_echantillon     combien de clients AVANT de lancer le test A/B (la puissance)
    161 benjamini_hochberg     20 segments, 20 tests : lesquels croire ?
    162 comparer_auc           le challenger bat-il VRAIMENT le champion ? (le bootstrap apparié)
    163 effet_cuped            le même test A/B, deux fois plus précis grâce au score calculé la veille

Prérequis : la série 00, et un peu de NumPy (`np.asarray`, `np.mean`, les masques `x[x > 0]`). La série
05 (intervalle de Wilson, test A/B) éclaire les défis 159 et 160 ; la série 03 (AUC) le défi 162.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un chiffre mesuré sur des clients n'est qu'UNE réalisation du hasard : avec d'autres clients, il aurait
  été un peu différent. L'INTERVALLE DE CONFIANCE à 95 % dit de combien : si l'on refaisait l'étude
  souvent, 95 % des intervalles ainsi calculés contiendraient la vraie valeur ;
- le BOOTSTRAP : on ne peut pas refaire l'étude, mais on peut tirer AVEC REMISE n clients parmi les n
  observés, des milliers de fois, et regarder comment la statistique varie d'un tirage à l'autre. Cela
  marche pour une moyenne, une médiane, une AUC… sans aucune formule ;
- la P-VALEUR : SI l'offre n'avait aucun effet, quelle serait la chance d'observer un écart au moins aussi
  grand ? Petite (< 0,05) : on doute du « aucun effet ». Ce n'est PAS la probabilité que l'offre ne marche
  pas, et p = 0,07 ne prouve pas l'absence d'effet : seulement qu'on n'a pas assez de données pour conclure ;
- un test à 5 % se trompe une fois sur 20 quand il n'y a RIEN (faux positif). Vingt tests sur du bruit :
  une « découverte » en moyenne. La PUISSANCE (80 %) est la chance de voir un effet qui existe vraiment :
  elle se règle AVANT l'expérience, en choisissant le nombre de clients ;
- on fixe ce nombre à l'avance et on ne s'arrête pas le jour où p passe sous 0,05 : regarder tous les jours
  et s'arrêter dès que « ça marche » fait monter les faux positifs à 20 % ou 30 % ;
- le hasard d'une fonction vient d'UN générateur créé dans la fonction : `rng = np.random.default_rng(graine)`,
  puis `rng.integers`, `rng.permutation`… Même graine, même résultat ; et jamais `np.random.seed` ni
  `np.random.choice`, qui changent le hasard de TOUT le programme (le code des autres compris).

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_26 import *
    >>> attentes = attentes_support()
    >>> float(np.median(attentes)), round(float(np.mean(attentes)), 1)
    (7.5, 10.8)
    >>> ancien, nouveau = notes_deux_scripts()
    >>> len(ancien), len(nouveau)
    (14, 12)
    >>> p_valeur_exacte([1, 2, 3], [4, 5, 6])
    0.1
    >>> rng = np.random.default_rng(0)
    >>> rng.integers(0, 5, size=5)              # 5 positions entre 0 et 4, AVEC remise (des doublons)
    array([4, 3, 2, 1, 1])

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les défis sont indépendants ; le 162
   ressemble au 158 (faites-le après).
2. Lancez `python scripts/defis.py --defi 158` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_26.py`.

Les outils communs (`attentes_support`, `notes_deux_scripts`, `p_valeurs_segments`, `scores_deux_modeles`,
`campagne_exemple`, `EFFET_REEL`, `p_valeur_exacte`, `puissance_simulee`, `auc`) sont dans
`defis/incertitude.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import math  # noqa: F401 — math.ceil, math.sqrt
import numbers  # noqa: F401 — isinstance(x, numbers.Real) : int, float, et les nombres de numpy
from statistics import NormalDist  # noqa: F401 — NormalDist().inv_cdf(0.975) vaut 1.959963984540054

import numpy as np

from defis.incertitude import (  # noqa: F401
    EFFET_REEL,
    attentes_support,
    auc,
    campagne_exemple,
    notes_deux_scripts,
    p_valeur_exacte,
    p_valeurs_segments,
    puissance_simulee,
    scores_deux_modeles,
)

# ============================================================================ Le bootstrap


def intervalle_bootstrap(valeurs, statistique=np.mean, n_tirages: int = 2000, confiance: float = 0.95,
                         graine: int = 0) -> tuple[float, float]:
    """Défi 158 🟢 — L'intervalle de confiance de n'importe quelle statistique, par le bootstrap.

    La médiane d'attente au support est de 7,5 minutes. À quelques secondes près, ou à 3 minutes près ?
    Aucune formule simple ne le dit pour une médiane. Le bootstrap, si :

    1. `rng = np.random.default_rng(graine)`, une fois, au début de la fonction ;
    2. `n_tirages` fois : tirer n valeurs AVEC REMISE parmi les n valeurs reçues (n = len(valeurs)) et
       calculer `statistique(tirage)` ;
    3. renvoyer `(bas, haut)` : les quantiles `(1 - confiance) / 2` et `(1 + confiance) / 2` de ces
       `n_tirages` résultats (2,5 % et 97,5 % pour 95 %), deux float, avec `np.quantile`.

        >>> bas, haut = intervalle_bootstrap(attentes_support(), np.median)
        >>> bas <= 7.5 <= haut, 5.5 <= bas <= 7.0, 8.5 <= haut <= 10.5
        (True, True, True)
        >>> intervalle_bootstrap(attentes_support(), np.median) == (bas, haut)   # même graine, même résultat
        True

    Les règles :

    - `valeurs` : une liste, un tableau NumPy ou une Series pandas (lue par POSITION, quel que soit son
      index : commencez par `np.asarray(valeurs, dtype=float)`), jamais modifiée. Le résultat ne dépend
      pas du type reçu : une liste et le tableau correspondant donnent le même intervalle ;
    - `statistique` : une fonction qui reçoit un TABLEAU NumPy et renvoie un nombre (`np.mean`,
      `np.median`, `lambda x: np.mean(x > 15)` pour la part d'attentes de plus d'un quart d'heure…) ;
    - tout le hasard vient de `rng` : même `graine`, même résultat ; aucun effet sur `np.random` ni sur
      `random` (pas de `np.random.seed`, de `np.random.choice`, de `random.shuffle`) ;
    - aucune valeur, ou un NaN dans `valeurs` → ValueError ; `confiance` hors de ]0, 1[ (95 au lieu de
      0.95) → ValueError.

    Ce que vous apprenez : rééchantillonner AVEC remise (sans remise, chaque tirage contient exactement les
    mêmes clients : un intervalle de largeur nulle), lire un intervalle, un hasard reproductible et local.

    Indice : `x = np.asarray(valeurs, dtype=float)`, puis dans une boucle
    `tirage = x[rng.integers(0, len(x), size=len(x))]` (des POSITIONS tirées avec remise) ;
    `np.quantile(resultats, [(1 - confiance) / 2, (1 + confiance) / 2])`.
    """
    raise NotImplementedError


# ============================================================================ Le test de permutation


def p_valeur_permutation(a, b, n_permutations: int = 10_000, graine: int = 0) -> float:
    """Défi 159 🟢 — La p-valeur d'un écart de moyennes, par permutation (sans aucune formule).

    Le nouveau script du support (`b`) a une note moyenne plus haute de 1,19 point que l'ancien (`a`).
    SI le script ne changeait rien, les notes seraient les mêmes quel que soit le groupe : n'importe quel
    partage des 26 notes en 14 + 12 aurait été aussi probable que celui observé. Le test :

    1. `observe = abs(moyenne(b) - moyenne(a))` (bilatéral : un écart dans un sens ou dans l'autre) ;
    2. `n_permutations` fois : mélanger toutes les notes ensemble (`rng.permutation`), appeler `a` les
       `len(a)` premières et `b` les autres, et compter les mélanges dont l'écart (en valeur absolue) est
       AU MOINS aussi grand que `observe` (`>=`) ;
    3. renvoyer `(compte + 1) / (n_permutations + 1)` : le partage observé est l'un des partages
       possibles, donc la p-valeur n'est jamais 0.

        >>> ancien, nouveau = notes_deux_scripts()
        >>> p = p_valeur_permutation(ancien, nouveau)
        >>> 0.05 < p < 0.10      # 1,19 point d'écart, mais avec 26 clients le hasard peut le faire
        True
        >>> p_valeur_permutation(range(20), range(100, 120), n_permutations=999)
        0.001

    Les règles :

    - `a` et `b` : listes, tableaux ou Series (lus par position), de tailles quelconques (14 et 12, ou 3
      et 40), jamais modifiés ;
    - `rng = np.random.default_rng(graine)` : même graine, même résultat ; `np.random` et `random` intacts ;
    - un groupe vide ou un NaN → ValueError ;
    - pour de petits groupes, votre résultat doit être proche de `p_valeur_exacte(a, b)`, qui essaie TOUS
      les partages possibles (c'est ce que le correcteur vérifie).

    Ce que vous apprenez : ce qu'est une p-valeur (« si rien ne se passait, à quelle fréquence verrait-on
    ça ? »), la simuler au lieu de chercher une formule, couper au bon endroit quand les groupes n'ont pas
    la même taille, pourquoi `>=` et pourquoi `+ 1`.

    Indice : `tous = np.concatenate([a, b])` (un NOUVEAU tableau : on peut le mélanger), `k = len(a)`,
    puis `melange = rng.permutation(tous)` et `melange[:k]`, `melange[k:]`.
    """
    raise NotImplementedError


# ============================================================================ La taille d'échantillon


def taille_echantillon(p_temoin: float, baisse: float, alpha: float = 0.05, puissance: float = 0.80) -> int:
    """Défi 160 🟡 — Combien de clients par groupe pour voir une baisse de churn ? (avant de lancer le test)

    Le churn est de 30 %. Le marketing espère que l'offre le fera baisser d'au moins 5 points (à 25 %). Avec
    trop peu de clients, le test conclura « pas d'effet démontré » même si l'offre marche : argent et temps
    perdus. Combien en faut-il, dans CHAQUE groupe (témoin et offre), pour avoir 80 % de chances de voir
    la baisse (la puissance) avec un test bilatéral à 5 % (alpha) ?

        p_offre = p_temoin - baisse                       p_moyen = (p_temoin + p_offre) / 2
        z_alpha = NormalDist().inv_cdf(1 - alpha / 2)     (1,96 pour alpha = 0,05 : bilatéral)
        z_beta  = NormalDist().inv_cdf(puissance)         (0,84 pour une puissance de 0,80)
        A = z_alpha × √(2 × p_moyen × (1 - p_moyen))
        B = z_beta  × √(p_temoin × (1 - p_temoin) + p_offre × (1 - p_offre))
        n = (A + B)² / baisse²

    arrondi À L'ENTIER SUPÉRIEUR (`math.ceil`), renvoyé comme un `int`.

        >>> taille_echantillon(0.30, 0.05)
        1251
        >>> taille_echantillon(0.30, 0.10)            # un effet 2 fois plus grand : 4 fois moins de clients
        294
        >>> 0.78 < puissance_simulee(1251, 0.30, 0.25) < 0.82   # vérifié sur 4 000 expériences simulées : ≈ 0,80
        True

    Les règles :

    - `baisse` est en POINTS, écrits comme une proportion : 5 points = 0.05 (et pas 5, ni « 5 % de 30 % ») ;
    - `p_temoin`, `baisse`, `alpha` et `puissance` sont des nombres strictement entre 0 et 1, sinon
      ValueError (un NaN aussi) ; une baisse qui rendrait le churn de l'offre nul ou négatif → ValueError ;
    - le résultat est le nombre de clients PAR GROUPE (il en faut deux fois plus en tout).

    Ce que vous apprenez : la puissance d'un test, l'influence de chaque paramètre (l'effet compte au CARRÉ),
    utiliser `statistics.NormalDist` au lieu de recopier 1,96, arrondir dans le bon sens, vérifier une
    formule par simulation (`puissance_simulee`).

    Indice : `loi = NormalDist()`, puis `loi.inv_cdf(1 - alpha / 2)` et `loi.inv_cdf(puissance)` ;
    `math.sqrt` ; `math.ceil(...)` renvoie déjà un int.
    """
    raise NotImplementedError


# ============================================================================ Benjamini-Hochberg


def benjamini_hochberg(p_valeurs, q: float = 0.05) -> list[bool]:
    """Défi 161 🟡 — Beaucoup de tests à la fois : lesquels croire ? (la procédure de Benjamini-Hochberg)

    Le test A/B a été refait sur 20 segments : 5 ont p < 0,05. Mais sur 20 tests, le hasard seul en donne
    déjà un en moyenne. Corriger à la Bonferroni (p < 0,05 / 20) protège, mais rejette presque tout.
    Benjamini-Hochberg garantit qu'EN MOYENNE, au plus une fraction `q` des découvertes annoncées est
    fausse (le « taux de fausses découvertes ») :

    1. m = nombre de p-valeurs ; on les range de la plus petite à la plus grande : p(1) <= … <= p(m) ;
    2. on cherche le PLUS GRAND k tel que p(k) <= k / m × q (même si un k plus petit ne passait pas) ;
    3. les k plus petites p-valeurs sont des découvertes (True), les autres non (False).

    Renvoie une liste de booléens, DANS L'ORDRE REÇU (la i-ème décision correspond à la i-ème p-valeur).

        >>> benjamini_hochberg([0.20, 0.036, 0.01, 0.03])
        [False, True, True, True]
        >>> segments = p_valeurs_segments()
        >>> [s for s, d in zip(segments, benjamini_hochberg(list(segments.values()))) if d]
        ['mensuel_cheque', 'fibre_mensuel', 'jeunes_fibre']

    Dans le premier exemple, 0,03 dépasse son seuil (2/4 × 0,05 = 0,025) mais 0,036 passe le sien
    (3/4 × 0,05 = 0,0375) : les trois plus petites sont retenues. S'arrêter au premier échec est une
    erreur classique.

    Les règles :

    - `p_valeurs` : une liste, un tuple, un tableau ou une Series (lue par position), jamais modifiée (pas
      de `.sort()` sur la liste reçue !) ; une liste vide donne `[]` ;
    - une p-valeur pile sur son seuil est retenue (`<=`) ; des p-valeurs égales reçoivent la même décision ;
    - un NaN ou une p-valeur hors de [0, 1] → ValueError ; `q` hors de ]0, 1[ → ValueError ;
    - 20 000 p-valeurs en un instant : triez UNE fois (pas de boucle dans une boucle).

    Ce que vous apprenez : le problème des comparaisons multiples, le taux de fausses découvertes, trier en
    gardant la trace des positions d'origine (`np.argsort`, ou `sorted(range(m), key=...)`), remettre les
    résultats dans l'ordre reçu.

    Indice : `ordre = np.argsort(p)` donne les POSITIONS de la plus petite à la plus grande ; `p[ordre]`
    les valeurs triées. Trouvez k, puis `decisions[ordre[:k]] = True` sur un tableau de False.
    """
    raise NotImplementedError


# ============================================================================ Comparer deux modèles


def comparer_auc(y, proba_a, proba_b, n_tirages: int = 1000, confiance: float = 0.95, graine: int = 0) -> dict:
    """Défi 162 🟡 — Le challenger bat-il VRAIMENT le champion ? L'intervalle de l'écart d'AUC, apparié.

    Sur le mois de test, le challenger (b) a une AUC de 0,778 et le champion (a) de 0,760 : +0,018. Chaque
    AUC seule a un intervalle large (±0,04) : les deux intervalles se chevauchent, et on serait tenté de
    conclure « pas de différence ». Mais les deux modèles notent les MÊMES clients et se trompent souvent
    ensemble : l'ÉCART, lui, est bien plus précis. Pour le mesurer, chaque tirage du bootstrap prend les
    MÊMES clients pour les deux modèles :

    1. `ecart = AUC(y, proba_b) - AUC(y, proba_a)` sur TOUS les clients (positif : b fait mieux) ;
    2. `n_tirages` fois : tirer n positions avec remise (`rng.integers(0, n, size=n)`), et calculer
       AUC(b) - AUC(a) sur CES positions, pour les deux modèles ;
    3. un tirage qui ne contient qu'une seule classe (que des 0, ou que des 1) n'a pas d'AUC : on l'écarte
       et on le COMPTE ;
    4. `bas`, `haut` : les quantiles `(1 - confiance) / 2` et `(1 + confiance) / 2` des écarts calculés.

    Renvoie `{"ecart": float, "bas": float, "haut": float, "tirages_ignores": int}`.

        >>> df = scores_deux_modeles()
        >>> r = comparer_auc(df["churn"], df["champion"], df["challenger"])
        >>> round(r["ecart"], 3), r["tirages_ignores"]
        (0.018, 0)
        >>> 0 < r["bas"] < r["ecart"] < r["haut"] < 0.04    # l'intervalle ne contient pas 0 : b fait mieux
        True

    Les règles :

    - `y` (des 0 et des 1), `proba_a`, `proba_b` : listes, tableaux ou Series (lus par POSITION), de même
      longueur, jamais modifiés ;
    - l'AUC : `auc(y, score)` (fourni) ou `sklearn.metrics.roc_auc_score`, comme vous voulez. `auc` est environ
      dix fois plus rapide sur de petits tableaux : avec des milliers de tirages, ça se sent (le correcteur
      en demande environ 2 500) ;
    - `rng = np.random.default_rng(graine)` : même graine, même résultat ; `np.random` et `random` intacts ;
    - longueurs différentes, `y` qui contient autre chose que 0 et 1, `y` d'une seule classe, un NaN dans
      les probabilités, `confiance` hors de ]0, 1[ → ValueError ; si TOUS les tirages sont écartés →
      ValueError.

    Ce que vous apprenez : la comparaison APPARIÉE (deux intervalles qui se chevauchent ne prouvent pas
    l'égalité), un bootstrap sur une statistique qui n'existe pas toujours, compter ce qu'on écarte.

    Indice : la même boucle qu'au défi 158, mais avec `i = rng.integers(0, n, size=n)` puis `y[i]`,
    `a[i]` et `b[i]` : un seul tirage de positions, utilisé pour les deux modèles.
    """
    raise NotImplementedError


# ============================================================================ CUPED


def effet_cuped(y, x_avant, traite) -> dict:
    """Défi 163 🔴 — Le même test A/B, plus précis : retirer le bruit déjà connu la veille (CUPED).

    Le test A/B de l'offre (`campagne_exemple()`) mesure un effet brut de -1,9 point, avec une erreur type
    de 1,4 point : « pas significatif ». Pourtant, une grande partie de qui part était PRÉVISIBLE : le
    modèle avait calculé un score pour chaque client LA VEILLE de la campagne, donc sans que l'offre puisse
    l'influencer. Par malchance, le tirage au sort a mis un peu plus de clients risqués dans le groupe offre.
    CUPED (« Controlled-experiment Using Pre-Experiment Data », utilisé par Microsoft, Netflix, Booking…)
    retire de chaque client la part de son résultat que le score expliquait déjà :

    1. les clients sans score (x NaN, les nouveaux) reçoivent la moyenne des scores connus (leur
       correction sera nulle) ;
    2. θ = cov(x, y) / var(x), calculé sur TOUS les clients (traités et témoins ensemble), avec le MÊME
       ddof en haut et en bas. Si x est constant (ou entièrement inconnu), θ = 0 ;
    3. `y_ajuste = y - θ × (x - moyenne(x))`, la moyenne de x sur TOUS les clients ;
    4. l'effet est l'écart des moyennes de `y_ajuste` : traités MOINS témoins ; son erreur type vaut
       √(var_traités / n_traités + var_témoins / n_témoins), variances avec `ddof=1` ;
    5. la même chose sur `y` sans ajustement donne l'effet brut et son erreur type.

    Renvoie `{"effet", "erreur_type", "effet_brut", "erreur_type_brute", "theta"}` (des float).

        >>> df = campagne_exemple()
        >>> r = effet_cuped(df["churn"], df["score_avant"], df["offre"])
        >>> round(r["effet_brut"], 4), round(r["erreur_type_brute"], 4)
        (-0.0185, 0.0143)
        >>> round(r["effet"], 4), round(r["erreur_type"], 4), round(r["theta"], 2)    # la vérité : -0.04
        (-0.0384, 0.013, 0.97)
        >>> round(1 - (r["erreur_type"] / r["erreur_type_brute"]) ** 2, 2)    # 18 % de variance en moins
        0.18

    Moins de variance, c'est moins de clients pour la même précision (défi 160) : ici 18 % de clients en
    moins, gratuitement. Et sur CE tirage, l'effet corrigé passe de « non significatif » à 3 erreurs types.
    En moyenne sur beaucoup de tirages, les deux estimations visent juste ; CUPED se trompe moins souvent.

    Les règles :

    - `y` (le résultat : churn 0/1, un montant…), `x_avant` (mesuré AVANT le tirage au sort : jamais une
      donnée que l'offre a pu changer) et `traite` (True / False ou 1 / 0) : listes, tableaux ou Series,
      lus par POSITION, de même longueur, jamais modifiés (attention : remplir les NaN de x « en place »
      modifierait le tableau de l'appelant) ;
    - longueurs différentes, un NaN dans `y`, une valeur de `traite` autre que True / False / 0 / 1, moins
      de 2 clients dans un groupe → ValueError.

    Ce que vous apprenez : réduire la variance d'une expérience avec une donnée d'avant (la même idée
    qu'une régression avec covariable), pourquoi centrer sur la moyenne de TOUS (centrer chaque groupe sur
    la sienne annule tout), `ddof` (`np.cov` et `np.var` n'ont pas le même par défaut), copier avant de
    modifier.

    Indice : `x = np.array(x_avant, dtype=float)` (`np.array` COPIE ; `np.asarray` peut renvoyer le même
    tableau), `np.isnan(x)`, `np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)`, puis `v[t].mean()` et
    `v[~t].var(ddof=1)` avec `t` un tableau de booléens.
    """
    raise NotImplementedError

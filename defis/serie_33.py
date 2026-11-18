"""Série 33 — Prévoir les départs : prévision naïve, erreurs, lissage, Holt, validation glissante, marge.

🔮 VOUS VOULEZ DÉVELOPPER ? UNE PRÉVISION : ICI, VOUS L'ÉCRIVEZ, VOUS LA JUGEZ, ET VOUS DÉCIDEZ AVEC.

Mariam dirige le centre d'appels de rétention : chaque client qui demande à résilier y est rappelé par un
conseiller. Un conseiller traite 40 dossiers par mois, et il faut TROIS mois pour en recruter un et le former.
En décembre, elle doit donc savoir combien de clients demanderont à partir en MARS. Le modèle de churn dit QUI
risque de partir ; elle a besoin de COMBIEN, et quand. Cette série vous fait écrire les six pièces d'une
prévision qu'on peut défendre devant elle, de la plus simple à la décision :

    200 prevision_naive        « comme le mois dernier », « comme l'an dernier » : la référence à battre
    201 erreurs                MAE, RMSE, biais, MAPE : de combien, et dans quel sens, on se trompe
    202 lissage_exponentiel    un niveau qui suit les données sans s'affoler
    203 holt                   un niveau ET une tendance
    204 validation_glissante   rejouer la méthode dans le passé, en ne lui montrant que ce qu'on savait alors
    205 marge_de_securite      la prévision, plus la marge que les erreurs passées rendent nécessaire

Prérequis : la série 00 (listes, boucles, tuples). La série 21 (le temps, la validation dans le temps) et la
série 26 (l'incertitude) l'éclairent, mais tout ce qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- une SÉRIE, c'est une liste de nombres, un par mois, du plus ancien au plus récent. `historique[t]` est le
  mois t (0 = le premier). L'HORIZON, c'est le nombre de mois qu'on prévoit après le dernier connu ;
- la règle d'or : une prévision ne connaît QUE le passé. Prévoir le mois t avec une formule qui contient le
  mois t, c'est tricher, et la triche ne se voit qu'en production, quand l'avenir n'est plus là pour aider ;
- toute méthode se compare à la prévision NAÏVE (« comme le mois dernier », ou « comme le même mois l'an
  dernier » quand il y a une saison). Une méthode savante qui ne la bat pas ne sert à rien ;
- une erreur, c'est prévu − réel. Sa moyenne SANS valeur absolue (le biais) dit dans quel sens on se trompe :
  pour Mariam, prévoir trop coûte des conseillers qui attendent ; pas assez, des clients qui partent sans
  avoir été rappelés ;
- un LISSAGE garde un « niveau » ℓ qui suit les données : à chaque mois, ℓ ← alpha × (la valeur du mois) +
  (1 − alpha) × ℓ. Alpha proche de 1 : ℓ suit le dernier mois (nerveux) ; proche de 0 : ℓ bouge à peine
  (lent, mais calme). Holt ajoute une TENDANCE lissée de la même façon ;
- pour juger une méthode, on la REJOUE dans le passé : « si je l'avais utilisée en décembre 2024, qu'aurait-
  elle prévu pour mars 2025 ? » Puis en janvier 2025, et ainsi de suite. C'est la validation glissante.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_33 import *
    >>> departs = departs_exemple()               # janvier 2023 à décembre 2025
    >>> departs[-3:], mois_exemple()[-1]
    ([523.0, 503.0, 505.0], '2025-12')

Quand les six défis sont réussis, vous pouvez répondre à Mariam :

    >>> comme_l_an_dernier = lambda passe, h: prevision_naive(passe, h, saison=12)
    >>> essais = validation_glissante(departs, comme_l_an_dernier, 24, horizon=3)
    >>> len(essais), essais[0]          # en décembre 2024 (24 mois connus), on prévoyait mars 2025 : 412
    (10, Essai(origine=24, cible=26, prevu=412.0, reel=452.0))
    >>> [round(x, 3) for x in erreurs([e.reel for e in essais], [e.prevu for e in essais])]
    [30.9, 32.807, -30.9, 0.064]
    >>> lisse = lambda passe, h: lissage_exponentiel(passe, 0.3, h).previsions
    >>> essais_lisse = validation_glissante(departs, lisse, 24, horizon=3)
    >>> round(erreurs([e.reel for e in essais_lisse], [e.prevu for e in essais_lisse]).mae, 1)   # moins bien
    38.8
    >>> marge_de_securite(departs, comme_l_an_dernier, 24, 3, 90)   # mars 2026 : 452 prévus, 40 de marge
    Marge(prevision=452.0, marge=40.0, borne=492.0)
    >>> math.ceil(492.0 / DOSSIERS_PAR_CONSEILLER)                  # 13 conseillers (12 sans la marge)
    13

« Comme l'an dernier » fait la plus petite erreur moyenne… et se trompe TOUJOURS dans le même sens (biais
−30,9 : chaque année, il y a un peu plus de départs). La marge, calculée sur ces erreurs-là, le corrige.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les défis sont indépendants, sauf le 205 qui
   peut appeler votre 204 ; l'ordre proposé est le plus progressif.
2. Lancez `python scripts/defis.py --defi 200` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_33.py`.

Les outils communs (`departs_exemple`, `mois_exemple`, `generer_serie`, `Erreurs`, `Lissage`, `Essai`, `Marge`,
`DOSSIERS_PAR_CONSEILLER`) sont dans `defis/previsions.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import math  # noqa: F401 — math.sqrt(x) : la racine ; math.ceil(x) : l'entier juste au-dessus

from defis.previsions import (  # noqa: F401
    DOSSIERS_PAR_CONSEILLER,
    Erreurs,
    Essai,
    Lissage,
    Marge,
    departs_exemple,
    generer_serie,
    mois_exemple,
)

# ============================================================================ La référence à battre


def prevision_naive(historique: list[float], horizon: int, saison: int = 1) -> list[float]:
    """Défi 200 🟢 — Les `horizon` mois suivants, chacun égal au même mois du DERNIER cycle de `saison` mois.

        >>> prevision_naive([10, 20, 30, 40, 50, 60], 6, saison=4)   # le dernier cycle : 30, 40, 50, 60
        [30.0, 40.0, 50.0, 60.0, 30.0, 40.0]
        >>> departs = departs_exemple()
        >>> prevision_naive(departs, 3)                # saison = 1 : « comme décembre », trois fois
        [505.0, 505.0, 505.0]
        >>> prevision_naive(departs, 3, saison=12)     # « comme l'an dernier » : janvier, février, mars 2025
        [532.0, 484.0, 452.0]

    Les règles :

    - le dernier cycle, ce sont les `saison` dernières valeurs de `historique`, dans l'ordre du calendrier.
      Le 1er mois prévu reprend la 1re valeur du cycle, le 2e la 2e, et ainsi de suite ; après la dernière, on
      RECOMMENCE le cycle (le mois prévu n°h, en comptant à partir de 0, reprend la valeur n°h % saison) ;
    - avec `saison = 1` (la valeur par défaut), le cycle est le dernier mois : « demain comme aujourd'hui » ;
    - le résultat est une liste de `horizon` float ; `horizon = 0` → [] ;
    - horizon < 0 → ValueError ; saison < 1 → ValueError ; moins de `saison` valeurs connues (pas un cycle
      complet) → ValueError ; une valeur manquante (None ou NaN) dans `historique` → ValueError ;
    - ne modifiez pas `historique`.

    Ce que vous apprenez : la prévision de référence (la « baseline » de toute étude de prévision),
    l'arithmétique modulo d'un calendrier, et pourquoi on n'annonce jamais une erreur sans la comparer à
    celle d'une méthode naïve.

    Indice : `cycle = historique[len(historique) - saison:]`, puis
    `[float(cycle[h % saison]) for h in range(horizon)]`. Attention à `historique[-h]`, qui lit le cycle à
    l'envers.
    """
    raise NotImplementedError


def erreurs(reels: list[float], prevus: list[float]) -> Erreurs:
    """Défi 201 🟢 — Erreurs(mae, rmse, biais, mape) des prévisions `prevus` face à ce qui s'est passé, `reels`.

        >>> erreurs([10, 20], [13, 16])          # écarts prévu − réel : +3 et −4
        Erreurs(mae=3.5, rmse=3.5355339059327378, biais=-0.5, mape=0.25)
        >>> erreurs([0, 5], [1, 5]).mape is None   # un mois sans aucun départ : la MAPE n'existe pas
        True

    Avec e = prevu − reel pour chaque mois, et n le nombre de mois :

    - mae = la moyenne des |e| : « on se trompe de 3,5 départs en moyenne » ;
    - rmse = la RACINE de la moyenne des e² : la moyenne d'abord, la racine ensuite. Elle pèse plus lourd
      quand quelques erreurs sont grosses (toujours >= mae) ;
    - biais = la moyenne des e, SANS valeur absolue : les erreurs se compensent, et c'est voulu. Positif, on
      prévoit TROP ; négatif, PAS ASSEZ. C'est prevu − reel, pas l'inverse ;
    - mape = la moyenne des |e| / |reel| : une erreur relative, 0.25 pour 25 % (pas 25). On divise par le
      RÉEL, pas par la prévision. Si un seul réel vaut 0, la MAPE n'existe pas : None (ni 0, ni l'infini, ni
      une moyenne sur les autres mois en faisant comme si de rien n'était) ;
    - le résultat est un `Erreurs` (ou un tuple dans cet ordre : mae, rmse, biais, mape) ;
    - pas autant de réels que de prévisions → ValueError (`zip` s'arrêterait au plus court sans rien dire) ;
      aucune prévision → ValueError ; une valeur manquante (None ou NaN) → ValueError ;
    - ne modifiez ni `reels` ni `prevus`.

    Ce que vous apprenez : les quatre mesures qu'on voit dans tous les rapports de prévision, ce que chacune
    dit et cache, et pourquoi la MAPE, si populaire, casse dès qu'un mois vaut 0.

    Indice : `ecarts = [p - r for r, p in zip(reels, prevus)]` (après avoir vérifié les longueurs), puis
    une somme par mesure.
    """
    raise NotImplementedError


# ============================================================================ Lisser


def lissage_exponentiel(historique: list[float], alpha: float, horizon: int) -> Lissage:
    """Défi 202 🟡 — Lissage exponentiel simple : Lissage(ajustees, previsions, niveau, tendance=0.0).

        >>> lissage_exponentiel([10.0, 20.0, 20.0], 0.5, 2)
        Lissage(ajustees=[None, 10.0, 15.0], previsions=[17.5, 17.5], niveau=17.5, tendance=0.0)
        >>> lissage_exponentiel(departs_exemple(), 1.0, 2).previsions     # alpha = 1 : la prévision naïve
        [505.0, 505.0]
        >>> round(lissage_exponentiel(departs_exemple(), 0.3, 1).niveau, 1)
        503.0

    Ici : le niveau part de 10 (la 1re valeur). Mois 1 : on prévoit 10, puis on voit 20, et le niveau devient
    0,5 × 20 + 0,5 × 10 = 15. Mois 2 : on prévoit 15, on voit 20, le niveau devient 17,5. Les mois suivants,
    sans nouvelle donnée, on prévoit le dernier niveau : 17,5.

    Les règles :

    - le niveau de départ est la 1re valeur, `historique[0]` (pas la moyenne de toute la série : elle
      contient l'AVENIR de chaque mois) ;
    - pour chaque mois t >= 1, DANS CET ORDRE : ajustees[t] = le niveau actuel (la prévision du mois t,
      faite AVANT de le voir), puis niveau = alpha × historique[t] + (1 − alpha) × niveau ;
    - ajustees[0] = None : le premier mois n'a pas de passé, personne ne pouvait le prévoir. `ajustees` a
      autant d'éléments que `historique` ;
    - previsions = `horizon` fois le DERNIER niveau (le lissage simple n'a pas de tendance : la prévision est
      plate), pas la dernière valeur de l'historique ; `niveau` = ce dernier niveau ; `tendance` = 0.0 ;
    - il faut 0 < alpha <= 1, sinon ValueError (alpha = 0 : le niveau ne bougerait jamais ; écrivez
      `not 0 < alpha <= 1`, qui refuse aussi NaN) ; horizon < 0 → ValueError ; historique vide →
      ValueError ; une valeur manquante (None ou NaN) → ValueError ;
    - ne modifiez pas `historique` ;
    - 20 000 mois : le correcteur compte les lignes exécutées. UNE passe sur la série : relisser tout le passé
      pour chaque mois, c'est 20 000 fois trop.

    Ce que vous apprenez : le lissage exponentiel (`pandas.Series.ewm(alpha=…, adjust=False)`, le
    `SimpleExpSmoothing` de statsmodels), le rôle d'alpha, et la discipline « prévoir, PUIS apprendre »
    qui empêche une prévision de voir ce qu'elle prévoit.

    Indice : `niveau = float(historique[0])`, `ajustees = [None]`, puis `for y in historique[1:]:` avec
    d'abord `ajustees.append(niveau)`, ensuite la mise à jour du niveau.
    """
    raise NotImplementedError


def holt(historique: list[float], alpha: float, beta: float, horizon: int) -> Lissage:
    """Défi 203 🟡 — La méthode de Holt : un niveau ET une tendance lissés. Lissage(ajustees, previsions, niveau, tendance).

        >>> holt([10.0, 12.0, 14.0, 16.0], 0.5, 0.5, 3)    # une droite : Holt la prolonge exactement
        Lissage(ajustees=[None, None, 14.0, 16.0], previsions=[18.0, 20.0, 22.0], niveau=16.0, tendance=2.0)
        >>> holt([10.0, 12.0, 20.0], 0.5, 0.5, 2)          # la hausse s'accélère : la tendance aussi
        Lissage(ajustees=[None, None, 14.0], previsions=[20.5, 24.0], niveau=17.0, tendance=3.5)

    Ici (2e exemple) : au départ, niveau = 12 (la 2e valeur) et tendance = 12 − 10 = 2. Mois 2 : on prévoit
    12 + 2 = 14, on voit 20 ; le niveau devient 0,5 × 20 + 0,5 × 14 = 17 ; la tendance devient
    0,5 × (17 − 12) + 0,5 × 2 = 3,5. Puis on prévoit 17 + 3,5 = 20,5, et 17 + 2 × 3,5 = 24.

    Les règles :

    - départ : niveau = historique[1], tendance = historique[1] − historique[0]. Ce départ utilise les mois
      0 ET 1 : ajustees[0] = ajustees[1] = None (personne ne pouvait les prévoir) ;
    - pour chaque mois t >= 2, DANS CET ORDRE :
        prevu = niveau + tendance ; ajustees[t] = prevu ;
        nouveau_niveau = alpha × historique[t] + (1 − alpha) × prevu ;
        tendance = beta × (nouveau_niveau − ANCIEN niveau) + (1 − beta) × tendance ;
      la tendance se lisse sur l'écart entre deux NIVEAUX successifs, pas entre deux valeurs brutes. Gardez
      l'ancien niveau avant de l'écraser : sinon nouveau − ancien vaut 0 ;
    - previsions : le mois h après la fin (h = 1, 2, …, horizon) vaut niveau + h × tendance (pas
      niveau + tendance pour tous) ; `niveau` et `tendance` = les derniers ;
    - il faut 0 < alpha <= 1 et 0 < beta <= 1, sinon ValueError ; horizon < 0 → ValueError ; moins de 2
      valeurs → ValueError (pas de tendance) ; une valeur manquante (None ou NaN) → ValueError ;
    - ne modifiez pas `historique` ; 20 000 mois : une seule passe, comme au défi 202.

    Ce que vous apprenez : la méthode de Holt (le `Holt` de statsmodels, la base de Holt-Winters), mettre à
    jour deux états qui dépendent l'un de l'autre dans le bon ordre, et la limite des méthodes à tendance :
    elles prolongent tout, même un pic de rentrée.

    Indice : `niveau, tendance = float(historique[1]), float(historique[1] - historique[0])`, `ajustees =
    [None, None]`, puis pour chaque y suivant : `prevu = niveau + tendance`, `ancien = niveau`, et les deux
    mises à jour.
    """
    raise NotImplementedError


# ============================================================================ Juger et décider


def validation_glissante(serie: list[float], methode, debut: int, horizon: int = 1) -> list[Essai]:
    """Défi 204 🔴 — Rejouer `methode` dans le passé : un Essai(origine, cible, prevu, reel) par origine.

    `methode(passe, horizon)` est une fonction qui reçoit les mois connus (une liste) et renvoie la liste de
    ses `horizon` prévisions, comme `lambda passe, h: prevision_naive(passe, h, saison=12)`.

        >>> comme_dernier = lambda passe, horizon: [passe[-1]] * horizon
        >>> validation_glissante([5.0, 7.0, 6.0, 9.0], comme_dernier, 2)
        [Essai(origine=2, cible=2, prevu=7.0, reel=6.0), Essai(origine=3, cible=3, prevu=6.0, reel=9.0)]
        >>> validation_glissante([5.0, 7.0, 6.0, 9.0], comme_dernier, 1, horizon=2)
        [Essai(origine=1, cible=2, prevu=5.0, reel=6.0), Essai(origine=2, cible=3, prevu=7.0, reel=9.0)]

    Ici (2e exemple) : à l'origine 1, on ne connaît que [5.0] ; la méthode prévoit 2 mois, et on garde le
    2e (5.0), à comparer au mois n°2 (6.0). À l'origine 2, on connaît [5.0, 7.0] : on garde 7.0, à comparer
    au mois n°3. L'origine 3 n'a pas de cible connue (le mois n°4 n'existe pas) : on s'arrête.

    Les règles :

    - une ORIGINE, c'est le nombre de mois connus au moment de prévoir. Pour chaque origine, de `debut` à
      len(serie) − horizon (incluse : la dernière dont la cible existe), dans l'ordre :
        prevus = methode(serie[:origine], horizon)    (un seul appel, avec les SEULS mois d'avant) ;
        cible = origine + horizon − 1                 (le numéro du mois visé) ;
        Essai(origine, cible, prevus[horizon − 1], serie[cible]) ;
    - on garde la prévision à `horizon` mois, la DERNIÈRE de la liste : Mariam recrute trois mois à l'avance,
      c'est l'erreur à trois mois qui l'intéresse, pas celle à un mois ;
    - la méthode ne doit JAMAIS recevoir le mois qu'elle prévoit, ni rien après : `serie[:origine + 1]`, ou
      toute la série, ce serait lui souffler la réponse. Le correcteur regarde ce qu'elle reçoit ;
    - horizon < 1 → ValueError ; debut < 1 → ValueError (une méthode a besoin d'au moins un mois) ; aucune
      origine possible (debut > len(serie) − horizon) → ValueError ;
    - le résultat est une liste d'`Essai`, dans l'ordre des origines ; ne modifiez pas `serie`.

    Ce que vous apprenez : la validation à origine glissante (« rolling origin », le `TimeSeriesSplit` de
    scikit-learn en plus fin), la seule façon honnête de juger une prévision, et repérer une fuite du futur
    en regardant ce que reçoit une fonction.

    Indice : `for origine in range(debut, len(serie) - horizon + 1):`. La cible d'une prévision à 1 mois
    faite avec `origine` mois connus est le mois n°`origine` (les numéros commencent à 0).
    """
    raise NotImplementedError


def marge_de_securite(serie: list[float], methode, debut: int, horizon: int, niveau: int = 90) -> Marge:
    """Défi 205 🔴 — La prévision à `horizon` mois, plus la marge qui aurait suffi dans `niveau` % des essais passés.

        >>> comme_dernier = lambda passe, horizon: [passe[-1]] * horizon
        >>> marge_de_securite([10.0, 14.0, 11.0, 15.0, 12.0, 16.0], comme_dernier, 1, 1, 60)
        Marge(prevision=16.0, marge=4.0, borne=20.0)
        >>> comme_l_an_dernier = lambda passe, h: prevision_naive(passe, h, saison=12)
        >>> marge_de_securite(departs_exemple(), comme_l_an_dernier, 24, 3, 90)
        Marge(prevision=452.0, marge=40.0, borne=492.0)

    Ici (1er exemple) : les 5 essais à 1 mois donnent les erreurs réel − prévu 4, −3, 4, −3, 4 ; triées :
    −3, −3, 4, 4, 4. Pour couvrir 60 % des cas, il faut k = 3 erreurs sur 5 (3 / 5 = 60 %) : la 3e, 4. La
    prévision faite avec toute la série est 16 : on s'équipe pour 20.

    Les règles :

    - les essais sont ceux de `validation_glissante(serie, methode, debut, horizon)` (appelez votre défi 204),
      au MÊME horizon que la prévision : les erreurs à 1 mois sont plus petites qu'à 3, elles ne disent rien
      de celles à 3 ;
    - les erreurs se prennent dans le sens qui coûte : réel − prévu (positif = des départs que personne
      n'avait prévus). Pas de valeur absolue : une méthode qui prévoit toujours trop n'a pas besoin de marge ;
    - marge = le plus petit des écarts e tel qu'au moins `niveau` % des écarts soient <= e. Avec les n
      écarts triés, c'est le k-ième, où k est le plus petit entier tel que k / n >= niveau / 100. Calculez k
      en ENTIERS, `(niveau * n + 99) // 100` : en float, `math.ceil(28 / 100 * 25)` vaut 8 au lieu de 7, car
      28 / 100 * 25 donne 7.000000000000001.
      Pas d'interpolation entre deux écarts (ce que font `statistics.quantiles` et `numpy.quantile` par
      défaut) : la marge est un écart qui a vraiment eu lieu. Elle peut être négative. (Même
      `numpy.quantile(…, method="inverted_cdf")`, qui ne fait pas d'interpolation, calcule son rang en float :
      il tombe dans le piège de 28 % sur 25 essais) ;
    - prevision = la prévision à `horizon` mois (la dernière de la liste) faite par `methode` avec TOUTE la
      série : maintenant, tout ce passé est connu ; borne = prevision + marge ;
    - niveau doit être un int entre 1 et 99 (un pourcentage : 90, pas 0.9), sinon ValueError ; les autres
      erreurs sont celles du défi 204 ;
    - le résultat est une `Marge(prevision, marge, borne)` ; ne modifiez pas `serie`.

    Ce que vous apprenez : transformer une prévision en DÉCISION (combien de conseillers ?), un intervalle de
    prévision tiré des erreurs passées (l'idée de la prévision « conforme »), un quantile sans interpolation,
    et pourquoi une méthode plus précise coûte moins cher : sa marge est plus petite.

    Indice : `ecarts = sorted(e.reel - e.prevu for e in validation_glissante(…))`, `k = …`,
    `marge = ecarts[k - 1]`, puis `methode(list(serie), horizon)[horizon - 1]`.
    """
    raise NotImplementedError

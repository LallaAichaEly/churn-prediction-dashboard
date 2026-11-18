"""Série 35 — Cibler ceux qu'on peut convaincre : effet par segment, T-learner, issue transformée, Qini, plan, hors pli.

🧲 VOUS VOULEZ DÉVELOPPER ? LE SCORE DE CHURN DIT QUI VA PARTIR. ICI, VOUS ÉCRIVEZ CE QUI DIT QUI ON PEUT RETENIR.

Yasmine prépare la campagne de rétention de décembre. La directrice veut « appeler les 300 clients au plus
fort score de churn ». Yasmine a un doute : un client qui a appelé le support quatre fois ce trimestre a déjà
pris sa décision, et un appel de plus n'y changera rien. Ce qui compte, ce n'est pas le RISQUE d'un client,
c'est de combien l'appel fait BAISSER ce risque : son « uplift ». La campagne de novembre (le projet 3) a gardé
un groupe témoin tiré au hasard ; c'est lui qui permet de le mesurer. Cette série vous fait écrire les six
pièces d'un ciblage par l'uplift, de la mesure à la décision :

    212 effet_par_segment       l'effet de l'offre, segment par segment : témoin − offre
    213 t_learner               prédire l'effet client par client : deux modèles de risque, avec et sans offre
    214 issue_transformee       quand tout le monde n'avait pas la même chance d'être contacté
    215 courbe_qini             juger un classement : les départs évités si l'on appelle les k premiers
    216 plan_de_contact         qui appeler, avec quel budget, et qui ne JAMAIS appeler
    217 predictions_hors_pli    juger un modèle sur des clients qu'il n'a pas vus

Prérequis : la série 00 (listes, dictionnaires, boucles). Le projet 3 (le groupe témoin), la série 05 (le test
A/B) et la série 22 (le lissage d'un encodage par la cible) l'éclairent, mais tout ce qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- chaque client de la campagne de novembre est un dict : ses variables (`contrat`, `anciennete`, `appels`),
  son `score` de churn, sa `valeur` (ce qu'il rapporte en un an s'il reste, en euros), `offre` (1 s'il a été
  appelé, 0 s'il était témoin), `p_offre` (la chance qu'il avait d'être appelé) et `parti` (1 ou 0) ;
- on ne voit jamais, pour un même client, ce qui se serait passé AVEC et SANS l'appel : il a reçu l'un OU
  l'autre. L'effet ne se mesure donc que sur des GROUPES comparables, et le tirage au hasard du témoin les
  rend comparables ;
- l'UPLIFT d'un groupe, c'est son taux de départ sans l'offre moins son taux de départ avec : les départs
  évités par client appelé. Positif, l'offre retient ; nul, elle ne sert à rien ; NÉGATIF, elle fait partir ;
- on distingue quatre profils : les PERSUADABLES (l'appel les retient), les CAUSES PERDUES (ils partent
  quoi qu'on fasse), les ACQUIS (ils restent quoi qu'on fasse) et les CHIENS QUI DORMENT (l'appel leur rappelle
  qu'ils peuvent partir). Un ciblage par le score appelle surtout des causes perdues ;
- un appel coûte `COUT_CONTACT` = 30 € ; il rapporte, en moyenne, uplift × valeur du client.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_35 import *
    >>> campagne = campagne_exemple()               # novembre : 2 000 clients, 80 % appelés
    >>> sorted(campagne[0])
    ['anciennete', 'appels', 'contrat', 'customer_id', 'offre', 'p_offre', 'parti', 'score', 'valeur']

Quand les six défis sont réussis, vous pouvez répondre à la directrice :

    >>> plus_forts_scores = sorted(range(2000), key=lambda i: -campagne[i]["score"])[:300]
    >>> marques = [dict(c, choisi=i in set(plus_forts_scores)) for i, c in enumerate(campagne)]
    >>> round(effet_par_segment(marques, "choisi")[True].uplift, 3)      # l'appel n'a rien changé pour eux
    -0.012
    >>> apprendre = lambda entrainement, a_predire: t_learner(entrainement, a_predire, VARIABLES, m=20)
    >>> predits = predictions_hors_pli(campagne, apprendre, k=5)
    >>> plus_forts_uplifts = sorted(range(2000), key=lambda i: -predits[i])[:300]
    >>> marques = [dict(c, choisi=i in set(plus_forts_uplifts)) for i, c in enumerate(campagne)]
    >>> round(effet_par_segment(marques, "choisi")[True].uplift, 3)      # 12 départs évités pour 100 appels
    0.118
    >>> plan = plan_de_contact(prospects_exemple(), t_learner(campagne, prospects_exemple(), VARIABLES), 30, 9000)
    >>> len(plan.clients), plan.cout                                     # les 300 appels de décembre
    (300, 9000)

Les 300 plus forts scores sont surtout des causes perdues : l'appel n'a évité AUCUN départ chez eux. Les 300
plus forts uplifts prédits (par un modèle qui ne les avait pas vus) en ont évité 12 pour 100 appels.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les défis sont indépendants, sauf le 217 qu'on
   essaie volontiers avec votre 213 ; l'ordre proposé est le plus progressif.
2. Lancez `python scripts/defis.py --defi 212` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_35.py`.

Les outils communs (`campagne_exemple`, `campagne_stratifiee_exemple`, `prospects_exemple`, `generer_campagne`,
`VARIABLES`, `COUT_CONTACT`, `Effet`, `Qini`, `Plan`, et la vérité simulée `uplift_vrai`, `risque_vrai`) sont
dans `defis/ciblage.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import math  # noqa: F401 — math.isnan(x) : x est-il NaN ?

from defis.ciblage import (  # noqa: F401
    COUT_CONTACT,
    VARIABLES,
    Effet,
    Plan,
    Qini,
    campagne_exemple,
    campagne_stratifiee_exemple,
    generer_campagne,
    prospects_exemple,
    risque_vrai,
    uplift_vrai,
)

# ============================================================================ Mesurer


def effet_par_segment(clients: list[dict], variable: str) -> dict:
    """Défi 212 🟢 — {valeur de `variable`: Effet(n_offre, n_temoin, taux_offre, taux_temoin, uplift)}.

        >>> clients = [{"contrat": "mensuel", "offre": 1, "parti": 0}, {"contrat": "mensuel", "offre": 1, "parti": 1},
        ...            {"contrat": "mensuel", "offre": 0, "parti": 1}, {"contrat": "annuel", "offre": 1, "parti": 0}]
        >>> effet_par_segment(clients, "contrat")
        {'mensuel': Effet(n_offre=2, n_temoin=1, taux_offre=0.5, taux_temoin=1.0, uplift=0.5),
         'annuel': Effet(n_offre=1, n_temoin=0, taux_offre=0.0, taux_temoin=None, uplift=None)}
        >>> for appels, e in effet_par_segment(campagne_exemple(), "appels").items():
        ...     print(f"{appels:>3} : {e.n_offre} appelés, {e.n_temoin} témoins, départs {e.taux_offre:.1%} avec "
        ...           f"l'offre, {e.taux_temoin:.1%} sans, uplift {e.uplift:+.3f}")
        1-2 : 505 appelés, 140 témoins, départs 29.3% avec l'offre, 35.7% sans, uplift +0.064
         3+ : 246 appelés, 62 témoins, départs 63.0% avec l'offre, 58.1% sans, uplift -0.049
          0 : 810 appelés, 237 témoins, départs 21.4% avec l'offre, 32.5% sans, uplift +0.111

    Ici (1er exemple) : chez les mensuels, 1 départ sur 2 appelés (0.5) et 1 sur 1 témoin (1.0) : l'offre a
    évité 0.5 départ par client appelé. Chez les annuels, il n'y a aucun témoin : on ne peut pas savoir ce qui
    se serait passé sans l'offre, l'effet est None.
    Sur la campagne de novembre : les clients qui ont appelé le support 3 fois ou plus partent le plus (58 %
    sans rien faire)… et l'appel n'y change rien. Le score de churn les met en tête de liste.

    Les règles :

    - un segment, c'est l'ensemble des clients qui ont la même valeur de `client[variable]` (None compris : une
      valeur manquante forme un segment comme un autre) ;
    - n_offre = le nombre de clients du segment avec offre == 1 ; n_temoin = avec offre == 0 ;
    - taux_offre = la part de partis (parti == 1) parmi les n_offre ; taux_temoin = la part parmi les n_temoin.
      Chaque taux se calcule dans SON groupe, pas sur tout le segment ;
    - uplift = taux_temoin − taux_offre (dans cet ordre : sans l'offre, moins avec). Positif = l'offre retient ;
    - un groupe vide n'a pas de taux : None (pas 0.0, qui voudrait dire « personne n'est parti »), et l'uplift
      vaut alors None ;
    - le résultat est un dict dont les clés sont rangées dans l'ordre où chaque segment APPARAÎT pour la
      première fois dans `clients` ; `clients` vide → {} ;
    - `offre` et `parti` valent 0 ou 1 (True et False sont acceptés) ; toute autre valeur (2, "oui", None…) →
      ValueError : c'est une erreur de saisie, pas un départ ;
    - ne modifiez pas `clients`.

    Ce que vous apprenez : la différence entre un RISQUE (le taux de départ) et un EFFET (l'écart de taux
    entre deux groupes comparables), lire un test A/B segment par segment, et ne pas inventer un zéro.

    Indice : un dict `compteurs` qui associe à chaque segment une liste [n_offre, departs_offre, n_temoin,
    departs_temoin] (`compteurs.setdefault(valeur, [0, 0, 0, 0])`), puis un second dict pour les Effet.
    """
    raise NotImplementedError


# ============================================================================ Prédire


def t_learner(historique: list[dict], nouveaux: list[dict], variables, m: float = 20) -> list[float]:
    """Défi 213 🟡 — L'uplift prédit de chaque client de `nouveaux` : risque sans offre − risque avec offre.

        >>> historique = [{"contrat": "mensuel", "offre": 1, "parti": 0}, {"contrat": "mensuel", "offre": 0, "parti": 1},
        ...               {"contrat": "annuel", "offre": 1, "parti": 0}, {"contrat": "annuel", "offre": 0, "parti": 0}]
        >>> [round(u, 3) for u in t_learner(historique, [{"contrat": "mensuel"}, {"contrat": "biennal"}], ["contrat"], m=2)]
        [0.667, 0.5]
        >>> campagne = campagne_exemple()
        >>> profils = [{"contrat": "mensuel"}, {"contrat": "annuel"}, {"contrat": "biennal"}]
        >>> [round(u, 3) for u in t_learner(campagne, profils, ["contrat"], m=20)]
        [0.121, 0.067, -0.044]

    Ici (1er exemple) : avec offre, 0 départ sur 2 clients, l'a priori P_offre vaut 0 ; sans offre, 1 départ
    sur 2, P_temoin vaut 0.5. Un mensuel : le seul témoin mensuel est parti (sans lissage, on dirait 100 %) ;
    lissé, (1 + 2 × 0.5) / (1 + 2) = 2/3. Avec offre : (0 + 2 × 0) / (1 + 2) = 0. Uplift : 2/3 − 0 = 0.667.
    Un biennal n'a jamais été vu : chaque risque vaut l'a priori de son groupe, 0.5 − 0 = 0.5.
    Sur la campagne : l'appel évite 12 départs pour 100 mensuels appelés, et en PROVOQUE 4 pour 100 biennaux.

    Les règles :

    - la CELLULE d'un client, c'est le tuple de ses valeurs pour `variables`, dans cet ordre :
      `tuple(client[v] for v in variables)`. Deux clients de la même cellule ont le même profil ;
    - on estime DEUX risques, un par groupe g (offre = 1, puis offre = 0), avec les seuls clients de
      `historique` de ce groupe. Pour le groupe g : P_g = le taux de départ de TOUT le groupe g (l'a priori),
      puis pour une cellule c, avec n_g(c) clients et d_g(c) départs de ce groupe dans c :
          risque_g(c) = (d_g(c) + m × P_g) / (n_g(c) + m)
      Une petite cellule est tirée vers l'a priori de SON groupe ; une grosse garde son propre taux. Une
      cellule jamais vue dans un groupe (n = d = 0) reçoit l'a priori : la formule le fait toute seule ;
    - uplift(client) = risque_temoin(sa cellule) − risque_offre(sa cellule) ;
    - le résultat est une liste de float, un par client de `nouveaux`, dans le même ordre ; `nouveaux` vide → [] ;
    - les nouveaux clients n'ont été ni appelés ni observés : ils n'ont ni `offre` ni `parti`. Ne lisez que
      leurs variables ;
    - m est un nombre (int ou float, pas un bool) > 0, sinon ValueError (m = 0 donnerait les taux bruts, et une division
      par zéro pour toute cellule sans témoin). `historique` sans aucun client avec offre, ou sans aucun
      témoin → ValueError (pas d'effet mesurable). `offre` ou `parti` qui ne vaut pas 0 ou 1 → ValueError ;
    - ne modifiez ni `historique` ni `nouveaux` ;
    - 20 000 clients connus et 20 000 nouveaux : le correcteur compte les lignes exécutées. Comptez UNE fois
      par cellule (un dict), puis répondez à chaque nouveau client par une simple lecture.

    Ce que vous apprenez : le « T-learner » (deux modèles, un par groupe, dont on soustrait les prédictions),
    le lissage d'un taux par un a priori (le même que l'encodage par la cible de la série 22), et pourquoi le
    modèle de churn seul (un seul risque, tous groupes confondus) ne dit rien de l'effet d'une offre.

    Indice : deux dicts par groupe (clients et départs par cellule) remplis en une passe, les deux a priori,
    puis `(d.get(c, 0) + m * a_priori) / (n.get(c, 0) + m)` pour chaque nouveau client.
    """
    raise NotImplementedError


def issue_transformee(clients: list[dict]) -> list[float]:
    """Défi 214 🟡 — Pour chaque client, un nombre dont la MOYENNE sur un groupe estime son uplift, même si p varie.

        >>> issue_transformee([{"offre": 0, "parti": 1, "p_offre": 0.5}, {"offre": 1, "parti": 1, "p_offre": 0.5},
        ...                    {"offre": 1, "parti": 0, "p_offre": 0.5}, {"offre": 0, "parti": 1, "p_offre": 0.75}])
        [2.0, -2.0, 0.0, 4.0]
        >>> from statistics import fmean
        >>> campagne = campagne_stratifiee_exemple()      # 90 % d'offre si score >= 0.5, 50 % sinon
        >>> tous = [dict(c, tous=1) for c in campagne]
        >>> round(effet_par_segment(tous, "tous")[1].uplift, 3)      # la différence brute : « l'offre ne sert à rien »
        -0.006
        >>> round(fmean(issue_transformee(campagne)), 3)             # repondérée : elle évite 7 à 8 départs sur 100
        0.076
        >>> round(fmean(uplift_vrai(c) for c in campagne), 3)        # la vérité (connue ici : données simulées)
        0.068

    Ici : cette campagne a appelé 90 % des clients à fort score et 50 % des autres. Le groupe appelé contient
    donc beaucoup plus de clients à risque que le témoin : il part plus, et la différence brute des taux
    conclut que l'offre ne sert à rien. Repondérer chaque client par l'inverse de la chance qu'il avait
    d'être dans son groupe corrige ce déséquilibre.

    Les règles, avec p = client["p_offre"], la probabilité qu'avait CE client d'être appelé au tirage :

    - issue = parti × ((1 − offre) / (1 − p) − offre / p). Autrement dit :
        un témoin parti vaut 1 / (1 − p) : il représente 1 / (1 − p) clients comme lui ;
        un appelé parti vaut −1 / p ;
        un client resté vaut 0 ;
    - la moyenne de ces nombres sur un groupe de clients estime son uplift (départs évités par appel), sans
      biais, même quand p change d'un client à l'autre. C'est la « pondération par l'inverse de la
      probabilité » (IPW) des études médicales ;
    - chaque client a SON p : n'utilisez ni 0.5, ni la part d'appelés de toute la liste. Il faut 0 < p < 1 :
      p = 0 ou p = 1, c'est un client qu'on ne pouvait pas comparer (il était sûr d'être dans son groupe), et
      un p manquant (None ou NaN) ou qui n'est pas un nombre → ValueError (écrivez `not 0 < p < 1`, qui refuse
      aussi NaN) ;
    - `offre` et `parti` valent 0 ou 1 (True et False acceptés), sinon ValueError ;
    - le résultat est une liste de float, un par client, dans l'ordre ; `clients` vide → [] ;
    - ne modifiez pas `clients`.

    Ce que vous apprenez : pourquoi une campagne tirée « au hasard, mais pas avec la même chance pour tous »
    ment si on la lit comme un test A/B ordinaire, la pondération par l'inverse de la probabilité, et une
    astuce puissante : cette issue transformée est une cible ordinaire, que n'importe quel modèle de
    RÉGRESSION peut apprendre pour prédire l'uplift directement.

    Indice : une compréhension de liste, après avoir vérifié p ; attention au signe (on compte des départs
    ÉVITÉS : un témoin parti fait monter l'effet de l'offre).
    """
    raise NotImplementedError


# ============================================================================ Juger et décider


def courbe_qini(uplifts: list[float], clients: list[dict]) -> Qini:
    """Défi 215 🔴 — Qini(points, qini) : combien de départs aurait-on évités en appelant les k premiers ?

    `uplifts[i]` est la prédiction (ou n'importe quel score de classement) du client `clients[i]`.

        >>> clients = [{"offre": 1, "parti": 0}, {"offre": 0, "parti": 1},
        ...            {"offre": 1, "parti": 1}, {"offre": 0, "parti": 0}]
        >>> courbe_qini([0.3, 0.3, 0.1, 0.1], clients)
        Qini(points=[(0, 0.0), (2, 1.0), (4, 0.0)], qini=0.5)
        >>> campagne = campagne_exemple()
        >>> round(courbe_qini([c["score"] for c in campagne], campagne).qini, 1)      # classer par le score
        13.6
        >>> q = courbe_qini([uplift_vrai(c) for c in campagne], campagne)              # par la vérité
        >>> [(k, round(g, 1)) for k, g in q.points], round(q.qini, 1)
        ([(0, 0.0), (603, 69.4), (866, 110.4), (1194, 142.6), (1670, 135.2), (2000, 103.6)], 44.0)

    Ici (1er exemple) : les deux premiers clients sont ex aequo (0.3) : ils entrent ENSEMBLE, k passe de 0 à 2.
    Parmi eux, 1 appelé (resté) et 1 témoin (parti) : sans l'offre, l'appelé serait parti comme le témoin,
    1 × 1 / 1 − 0 = 1 départ évité. Avec les 4 clients : 2 appelés (1 départ), 2 témoins (1 départ) :
    1 × 2 / 2 − 1 = 0. Aire sous la courbe : 1 + 1 = 2 ; sous la droite du hasard (de (0, 0) à (4, 0)) : 0 ;
    qini = (2 − 0) / 4 = 0.5.
    Sur la campagne : appeler TOUT le monde évite environ 104 départs. Classés par la vérité, les 1 194
    premiers suffisent à en éviter 143 (les suivants, des chiens qui dorment, en font partir). Le score de
    churn classe mieux que le hasard (13.6 > 0), mais bien moins bien que la vérité (44.0).

    Les règles :

    - on range les clients par prédiction DÉCROISSANTE. Des clients EX AEQUO (même prédiction) entrent
      ensemble : un seul point, après le dernier d'entre eux. Sinon la courbe dépendrait de l'ordre de la liste,
      que personne n'a choisi (et un modèle qui prédit la même chose pour tout le monde aurait l'air bon) ;
    - points commence par (0, 0.0). Puis, pour chaque valeur distincte de prédiction, de la plus haute à la
      plus basse : (k, gain), où k = le nombre de clients dont la prédiction est >= cette valeur, et
          gain = D_C × N_T / N_C − D_T
      avec, parmi ces k premiers : N_T appelés dont D_T partis, N_C témoins dont D_C partis. D_C / N_C est le
      taux de départ SANS offre en haut de la liste ; × N_T, ce qu'auraient fait les appelés sans l'appel ;
      − D_T, ce qu'ils ont fait. Tant qu'aucun témoin n'est dans les k premiers (N_C = 0), on ne peut rien
      estimer : gain = 0.0 ;
    - le DERNIER point est (n, gain de toute la campagne) ;
    - qini = (aire sous la courbe − aire sous la droite du hasard) / n, où :
        l'aire sous la courbe = la somme des trapèzes (k2 − k1) × (g1 + g2) / 2 entre points successifs ;
        la droite du hasard va de (0, 0) au dernier point (n, g) : son aire vaut n × g / 2 ;
      0 = pas mieux que le hasard ; négatif = pire (on appelle d'abord ceux que l'appel fait partir) ;
    - pas autant de prédictions que de clients → ValueError ; aucun client → ValueError ; une prédiction
      manquante (None ou NaN : un NaN se range n'importe où dans un tri) → ValueError ; `offre` ou `parti` qui
      ne vaut pas 0 ou 1 → ValueError ;
    - ne modifiez ni `uplifts` ni `clients` ;
    - 20 000 clients : le correcteur compte les lignes exécutées. Un tri, puis UNE passe en cumulant N_T, D_T,
      N_C, D_C : recompter les k premiers pour chaque k, c'est 20 000 fois trop.

    Ce que vous apprenez : la courbe de Qini (`qini_curve` de scikit-uplift, `get_qini` de causalml), juger un
    classement quand on ne voit jamais l'effet d'un client mais seulement celui d'un groupe, traiter les
    ex aequo comme la courbe ROC le fait, et l'aire au-dessus du hasard.

    Indice : `ordre = sorted(range(n), key=lambda i: -uplifts[i])`, quatre compteurs, et un point quand le
    client suivant de `ordre` a une prédiction différente (ou qu'il n'y en a plus).
    """
    raise NotImplementedError


def plan_de_contact(clients: list[dict], uplifts: list[float], cout: int, budget: int) -> Plan:
    """Défi 216 🟡 — Plan(clients, gain, cout) : qui appeler pour gagner le plus, dans le budget.

        >>> clients = [{"customer_id": "A", "valeur": 1000}, {"customer_id": "B", "valeur": 1000},
        ...            {"customer_id": "C", "valeur": 200}, {"customer_id": "D", "valeur": 1000}]
        >>> plan_de_contact(clients, [0.10, -0.05, 0.20, 0.04], cout=30, budget=1000)
        Plan(clients=['A', 'C', 'D'], gain=90.0, cout=90)
        >>> plan_de_contact(clients, [0.10, -0.05, 0.20, 0.04], cout=30, budget=59)
        Plan(clients=['A'], gain=70.0, cout=30)
        >>> campagne, prospects = campagne_exemple(), prospects_exemple()
        >>> predits = t_learner(campagne, prospects, VARIABLES, m=20)
        >>> plan = plan_de_contact(prospects, predits, COUT_CONTACT, 30_000)
        >>> len(plan.clients), plan.cout, round(plan.gain)               # 727 appels « rentables », pas 1 000
        (727, 21810, 50435)

    Ici (1er exemple) : A rapporte 0.10 × 1000 − 30 = 70 €, C 0.20 × 200 − 30 = 10 €, D 0.04 × 1000 − 30 = 10 €.
    B est un chien qui dort (0.05 départ EN PLUS par appel) : on ne l'appelle jamais, même s'il reste du budget.
    Avec 59 €, un seul appel : le plus rentable, A.
    Sur les clients de décembre : 727 appels rapportent plus qu'ils ne coûtent. Les 8 190 € restants ne sont
    pas dépensés, et c'est voulu : chaque appel de plus coûterait plus qu'il ne rapporterait.

    Les règles :

    - le gain attendu d'un appel au client i = uplifts[i] × client["valeur"] − cout : les départs évités (en
      probabilité), fois ce que rapporte un client gardé, moins le prix de l'appel ;
    - on n'appelle QUE les clients dont le gain attendu est strictement positif (> 0). Ni les chiens qui
      dorment (uplift < 0), ni ceux pour qui l'appel coûte plus qu'il ne rapporte, ni ceux à gain nul, même s'il
      reste du budget : dépenser tout le budget n'est pas un objectif ;
    - le budget permet au plus budget // cout appels : on garde les MEILLEURS gains. Ordre du plan : gain
      décroissant, puis customer_id croissant à gain égal ;
    - le résultat est un Plan(clients, gain, cout) : la liste des customer_id dans cet ordre, la somme de LEURS
      gains (nets : coût déjà déduit) et len(clients) × cout ;
    - cout est un entier d'euros >= 1 et budget un entier d'euros >= 0 (pas un bool, pas un float), sinon
      ValueError ; pas autant de prédictions que de clients → ValueError ; un uplift ou une valeur manquant
      (None ou NaN) → ValueError ;
    - ne modifiez ni `clients` ni `uplifts`.

    Ce que vous apprenez : transformer une prédiction en DÉCISION chiffrée en euros, l'erreur la plus coûteuse
    du ciblage (appeler les chiens qui dorment), le seuil de rentabilité, et un tri à deux clés.

    Indice : une liste de tuples `(-gain, customer_id, gain)` pour les seuls gains > 0, `sort()`, puis les
    `budget // cout` premiers.
    """
    raise NotImplementedError


def predictions_hors_pli(historique: list[dict], apprendre, k: int = 5) -> list[float]:
    """Défi 217 🔴 — La prédiction de chaque client par un modèle entraîné SANS lui (validation croisée en k plis).

    `apprendre(entrainement, a_predire)` entraîne un modèle sur la liste `entrainement` et renvoie la liste de
    ses prédictions pour les clients de `a_predire`, comme
    `lambda entrainement, a_predire: t_learner(entrainement, a_predire, VARIABLES, m=20)`.

        >>> taille = lambda entrainement, a_predire: [len(entrainement)] * len(a_predire)
        >>> predictions_hors_pli([{"offre": 1, "parti": 0} for _ in range(5)], taille, k=2)
        [2, 3, 2, 3, 2]
        >>> campagne = campagne_exemple()
        >>> apprendre = lambda entrainement, a_predire: t_learner(entrainement, a_predire, VARIABLES, m=20)
        >>> round(courbe_qini(t_learner(campagne, campagne, VARIABLES, m=20), campagne).qini, 1)   # déjà vus
        59.7
        >>> round(courbe_qini(predictions_hors_pli(campagne, apprendre, k=5), campagne).qini, 1)   # jamais vus
        25.4
        >>> for m in (1, 5, 20, 200):        # choisir m sur des clients jamais vus
        ...     a = lambda e, p, m=m: t_learner(e, p, VARIABLES, m)
        ...     print(m, round(courbe_qini(predictions_hors_pli(campagne, a), campagne).qini, 1))
        1 26.4
        5 27.0
        20 25.4
        200 9.0

    Ici (1er exemple) : 5 clients, 2 plis. Le pli 0 contient les clients n°0, 2 et 4 (i % 2 == 0) : on les
    prédit avec un modèle entraîné sur les 2 autres ; le pli 1, les clients n°1 et 3, avec les 3 autres.
    Sur la campagne : jugé sur les clients qui ont servi à l'entraîner, le T-learner obtient un Qini de 59.7,
    MEILLEUR que celui de la vérité elle-même (44.0, défi 215) : c'est impossible, il se flatte. Sur des
    clients qu'il n'a pas vus, il obtient 25.4 : c'est ce qu'il vaut vraiment en décembre.

    Les règles :

    - le pli du client n°i (sa position dans `historique`) est i % k. Pour chaque pli f, de 0 à k − 1 :
        entrainement = les clients dont le pli n'est PAS f (tous leurs champs, `offre` et `parti` compris) ;
        a_predire = les clients du pli f, dans l'ordre, chacun COPIÉ SANS ses clés "offre" et "parti" : un
        modèle qui peut lire la réponse finit toujours par la lire ;
        un seul appel à apprendre(entrainement, a_predire) ;
    - pas de `del client["parti"]` ni de `client.pop("offre")` : ce serait modifier l'historique de celui qui
      vous l'a prêté. Construisez de nouveaux dicts ;
    - `entrainement` garde l'ordre de `historique` ;
    - le résultat est une liste de la longueur de `historique` : la prédiction n°i est celle du client n°i
      (pas les plis mis bout à bout) ;
    - k est un int entre 2 et len(historique), sinon ValueError ; `apprendre` qui ne renvoie pas une prédiction
      par client à prédire → ValueError ;
    - ne modifiez pas `historique`.

    Ce que vous apprenez : la validation croisée écrite à la main (le `cross_val_predict` de scikit-learn),
    pourquoi un modèle s'évalue sur des données qu'il n'a pas vues (le surapprentissage se voit ici de façon
    spectaculaire), choisir un réglage (m) sur ces données-là, et une fonction qui reçoit une fonction.

    Indice : `indices = list(range(f, len(historique), k))`, une compréhension de dict pour les copies
    (`{cle: v for cle, v in client.items() if cle not in ("offre", "parti")}`), et une liste
    `predictions = [None] * len(historique)` remplie à la bonne place.
    """
    raise NotImplementedError

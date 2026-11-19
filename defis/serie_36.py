"""Série 36 — Savoir si le service va bien : journal, quantile d'histogramme, compteur, budget d'erreur, alertes, traces.

📟 VOUS VOULEZ DÉVELOPPER ? UN SERVICE EN PRODUCTION, ÇA SE SURVEILLE. ICI, VOUS ÉCRIVEZ CE QUE CALCULENT LES OUTILS DE SURVEILLANCE.

Samir est d'astreinte cette semaine sur l'API de score (`api.py`), que les conseillers interrogent chaque fois
qu'un client appelle. Mardi, ils se sont plaints : la fiche client met parfois quatre secondes à s'afficher.
Le tableau de bord de l'équipe, lui, était tout vert : « temps de réponse moyen : 150 ms », « p99 : 1.76 s ».
Les deux chiffres étaient vrais, et aucun ne disait ce que vivaient les conseillers. Cette série vous fait écrire
les six calculs d'une surveillance qui ne ment pas, ceux que font Prometheus, Grafana et Jaeger :

    218 lire_journal            compter par ROUTE, pas par URL ; une erreur est la faute du serveur
    219 quantile_histogramme    le p99 de toute la flotte, sans faire la moyenne des p99
    220 augmentation            ce qu'a compté un compteur qui repart de zéro à chaque redémarrage
    221 budget_erreur           combien d'erreurs l'objectif de 99.9 % autorise encore ce mois-ci
    222 alertes                 réveiller quelqu'un quand le budget brûle trop vite… et seulement là
    223 temps_propre            où passent les quatre secondes d'une requête

Prérequis : la série 00 (listes, dictionnaires, boucles, `str.split`). La série 14 (tenir en production) et le
projet 4 (le service des conseillers) l'éclairent, mais tout ce qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un JOURNAL d'accès a une ligne par requête : quand, quoi (méthode et chemin), le statut HTTP, la durée. Les
  statuts 2xx disent « ça a marché », 4xx « l'appelant s'est trompé » (404 : ce client n'existe pas ; 422 : la
  demande est mal formée), 5xx « le SERVEUR a échoué ». Seuls les 5xx sont des erreurs du service ;
- une moyenne cache la queue : 99 requêtes à 50 ms et une à 10 s font 150 ms de moyenne, alors qu'un client
  sur cent attend 10 s. On surveille donc des QUANTILES : le p99 est la durée que 99 % des requêtes ne
  dépassent pas ;
- pour ne pas garder toutes les durées, chaque serveur les compte dans des SEAUX (« combien de requêtes ont
  duré au plus 0.1 s, au plus 0.25 s… ») : c'est un HISTOGRAMME. Un COMPTEUR, lui, ne fait que monter (le
  nombre de requêtes depuis le démarrage du serveur) ; un outil comme Prometheus le relit toutes les 15 s ;
- l'OBJECTIF de disponibilité (le SLO) est un engagement : 99.9 % des requêtes du mois réussissent. Les 0.1 %
  restants forment le BUDGET D'ERREUR : on peut les « dépenser » (déployer, tenter, se tromper), pas plus ;
- la VITESSE DE COMBUSTION (burn rate) compare le taux d'erreur à ce que le budget permet : 1 veut dire « au
  bout du mois, le budget sera pile épuisé » ; 14.4 sur une heure veut dire « cette heure a brûlé 2 % du budget
  du mois » ;
- une TRACE suit une requête à travers les services : chaque service y ajoute un SPAN (son début, sa fin, le
  span qui l'a appelé). Des spans enfants peuvent tourner EN PARALLÈLE.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_36 import *
    >>> journal_exemple()[1]
    '2026-11-17T14:00:04.533Z POST /predict 200 14.8ms'
    >>> len(minutes_exemple()), OBJECTIF
    (43200, 0.999)

Quand les six défis sont réussis, Samir peut répondre aux conseillers :

    >>> journal = lire_journal(journal_exemple())
    >>> len(journal.routes), journal.illisibles          # 1 272 URL différentes, mais 5 routes
    (5, 25)
    >>> histogrammes = list(histogrammes_exemple().values())
    >>> round(quantile_histogramme(histogrammes, 0.99), 2)                       # le p99 du service
    4.38
    >>> round(sum(quantile_histogramme([h], 0.99) for h in histogrammes) / 3, 2)  # ce qu'affichait le tableau de bord
    1.76
    >>> augmentation(compteur_exemple()).total             # « la dernière lecture moins la première » dirait -17095
    2932
    >>> budget = budget_erreur(minutes_exemple())
    >>> f"{budget.disponibilite:.3%} de réussite, {budget.consomme:.0%} du budget de novembre brûlé"
    '99.935% de réussite, 65% du budget de novembre brûlé'
    >>> [(a.regle, horloge(a.debut), horloge(a.fin)) for a in alertes(minutes_exemple()) if a.regle.startswith("page")]
    [('page 1 h', '12 nov. 14:12', '12 nov. 14:45'), ('page 6 h', '12 nov. 14:27', '12 nov. 15:08')]
    >>> propre = temps_propre(trace_exemple())
    >>> max(propre, key=propre.get), propre["passerelle"] + propre["api"]
    ('crm', 48)

Une requête sur cent attend plus de 4 secondes, et c'est toujours le même serveur. Le budget du mois tient
encore (65 % brûlés) : on peut déployer le correctif. Dans la requête lente, l'API et la passerelle ne
travaillent que 48 ms : le reste, c'est le CRM, interrogé trois fois en parallèle.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les défis sont indépendants ; l'ordre proposé
   est le plus progressif (le 222 réutilise volontiers les idées du 221).
2. Lancez `python scripts/defis.py --defi 218` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_36.py`.

Les outils communs (`journal_exemple`, `histogrammes_exemple`, `compteur_exemple`, `minutes_exemple`,
`trace_exemple`, `generer_journal`, `generer_histogramme`, `generer_minutes`, `generer_trace`, `horloge`, `SEAUX`,
`OBJECTIF`, `REGLES`, `Comptee`, et les résultats `Route`, `Journal`, `Hausse`, `Budget`, `Regle`, `Alerte`,
`Span`) sont dans `defis/observabilite.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import math  # noqa: F401 — math.inf, math.isfinite(x)

from defis.observabilite import (  # noqa: F401
    OBJECTIF,
    REGLES,
    SEAUX,
    Alerte,
    Budget,
    Comptee,
    Hausse,
    Journal,
    Regle,
    Route,
    Span,
    compteur_exemple,
    generer_histogramme,
    generer_journal,
    generer_minutes,
    generer_trace,
    histogrammes_exemple,
    horloge,
    journal_exemple,
    minutes_exemple,
    trace_exemple,
)

# ============================================================================ Mesurer


def lire_journal(lignes: list[str]) -> Journal:
    """Défi 218 🟢 — Journal(routes, illisibles) : requêtes, erreurs et durées de chaque ROUTE.

        >>> journal = lire_journal(["2026-11-17T14:00:01.874Z GET /clients/C04182/score 200 61.2ms",
        ...                         "2026-11-17T14:00:02.310Z GET /clients/C00007/score/?modele=v3 503 4.18s",
        ...                         "Traceback (most recent call last):",
        ...                         "",
        ...                         "2026-11-17T14:00:02.950Z GET /health 200 850µs",
        ...                         "2026-11-17T14:00:03.122Z GET /v2/clients/48213 404 12ms"])
        >>> journal.illisibles
        1
        >>> journal.routes
        {'GET /clients/{id}/score': Route(requetes=2, erreurs=1, durees_ms=[61.2, 4180.0]),
         'GET /health': Route(requetes=1, erreurs=0, durees_ms=[0.85]),
         'GET /v2/clients/{id}': Route(requetes=1, erreurs=0, durees_ms=[12.0])}
        >>> for route, r in lire_journal(journal_exemple()).routes.items():
        ...     print(f"{route:<24} {r.requetes:>5} requêtes, {r.erreurs:>2} erreurs, la plus lente : {max(r.durees_ms):.0f} ms")
        GET /health                236 requêtes,  0 erreurs, la plus lente : 1 ms
        POST /predict              381 requêtes,  3 erreurs, la plus lente : 68 ms
        GET /clients/{id}/score   1095 requêtes, 26 erreurs, la plus lente : 5997 ms
        GET /clients               134 requêtes,  0 erreurs, la plus lente : 491 ms
        GET /v2/clients/{id}       147 requêtes,  1 erreurs, la plus lente : 151 ms

    Ici (1er exemple) : les deux premières lignes sont la MÊME route (la fiche client), pour deux clients
    différents ; la seconde a échoué (503) et duré 4.18 s. La trace Python est illisible ; la ligne vide ne
    compte pas. Le 404 n'est pas une erreur du service : ce client n'existe pas.
    Sur l'heure de mardi : 1 272 URL différentes, mais seulement 5 routes. Compter par URL donnerait un
    tableau d'une ligne par client, et un outil de surveillance qui s'écroule sous le nombre de séries.

    Les règles :

    - une ligne a 5 champs séparés par des espaces (`ligne.split()`) : horodatage, méthode, chemin, statut,
      durée. Une ligne vide (ou faite d'espaces) est ignorée : ni requête, ni illisible ;
    - une ligne est ILLISIBLE (on la compte dans `illisibles` et on passe à la suivante, sans lever d'erreur)
      si elle n'a pas 5 champs, si le chemin ne commence pas par « / », si le statut n'est pas un nombre de
      3 chiffres entre 100 et 599, ou si la durée n'est pas lisible ;
    - la durée est un nombre positif ou nul suivi de son unité, comme l'écrit le répartiteur de charge
      (programmé en Go) : « 850µs » (microsecondes : copiez le « µ » d'ici), « 61.2ms » (millisecondes) ou
      « 4.18s » (secondes). Convertissez-la en MILLISECONDES (float). Sans unité connue, ou si le nombre n'est
      pas lisible ou pas fini (`float("nan")` réussit !), la ligne est illisible ;
    - la ROUTE est "MÉTHODE chemin" où, dans le chemin : on retire tout ce qui suit « ? » (les paramètres) ;
      chaque morceau entre deux « / » qui est un IDENTIFIANT devient « {id} » ; on retire les « / » finaux
      (mais « / » seul reste « / »). Un identifiant est soit un nombre (que des chiffres 0 à 9 : 48213), soit
      une lettre majuscule suivie d'au moins 3 chiffres (C04182). « v2 » n'en est pas un ;
    - `erreurs` compte les statuts 5xx (500 à 599) ; les 4xx sont des requêtes, pas des erreurs ;
    - le résultat est Journal(routes, illisibles), où `routes` est un dict {route: Route(requetes, erreurs,
      durees_ms)} rangé dans l'ordre où chaque route APPARAÎT, et `durees_ms` les durées dans l'ordre des lignes ;
    - ne modifiez pas `lignes`.

    Ce que vous apprenez : lire un texte qui n'est pas toujours propre sans jamais planter (une ligne abîmée
    ne doit pas arrêter le comptage de toutes les autres), la « cardinalité » (un compteur par client, c'est
    un compteur de trop), et la différence entre une erreur du service et une erreur de l'appelant.

    Indice : une fonction `duree_ms(texte)` qui renvoie None quand elle ne sait pas lire (testez « µs » et
    « ms » AVANT « s » : « 61.2ms » finit aussi par « s »), une fonction `route(methode, chemin)`, puis
    `routes.setdefault(route, [0, 0, []])`.
    """
    raise NotImplementedError


def quantile_histogramme(histogrammes: list[list[tuple[float, float]]], q: float) -> float | None:
    """Défi 219 🟡 — Le quantile q (0.99 pour le p99) des durées comptées par TOUS les histogrammes, en secondes.

        >>> serveur_a = [(0.1, 60), (0.5, 90), (1.0, 100), (math.inf, 100)]   # 60 requêtes en 0.1 s ou moins…
        >>> serveur_b = [(0.1, 10), (0.5, 20), (1.0, 50), (math.inf, 60)]
        >>> quantile_histogramme([serveur_a], 0.5)            # 50 requêtes sur 100 : dans le 1er seau
        0.08333333333333333
        >>> quantile_histogramme([serveur_a, serveur_b], 0.5)  # les deux serveurs : 80e requête sur 160
        0.2
        >>> quantile_histogramme([serveur_a, serveur_b], 0.99)  # au-delà du dernier seau fini
        1.0
        >>> histogrammes = histogrammes_exemple()
        >>> {serveur: round(quantile_histogramme([h], 0.99), 2) for serveur, h in histogrammes.items()}
        {'api-1': 0.24, 'api-2': 0.24, 'api-3': 4.79}
        >>> round(quantile_histogramme(list(histogrammes.values()), 0.99), 2)
        4.38

    Ici (1er exemple) : le seau (0.1, 60) dit « 60 requêtes ont duré 0.1 s ou moins ». La médiane (q = 0.5) est la
    50e requête sur 100 : elle est dans le premier seau, qui va de 0 à 0.1 s et en contient 60 ; en supposant ces
    durées réparties régulièrement dans le seau, la 50e est aux 50/60 du seau : 0.0833 s. Avec le serveur b, on
    ADDITIONNE d'abord les seaux : (0.1, 70), (0.5, 110), (1.0, 150), (inf, 160). Le rang vaut 0.5 × 160 = 80 :
    70 requêtes ont duré 0.1 s ou moins, le seau 0.1 – 0.5 en contient 110 − 70 = 40, et la 80e est la 10e de ce
    seau : 0.1 + 0.4 × 10/40 = 0.2 s. Pour le p99, le rang 158.4 dépasse les 150 requêtes des seaux finis : la
    réponse est la plus grande borne finie, 1.0 s.
    Sur la journée de mardi : api-1 et api-2 ont un p99 de 0.24 s, api-3 de 4.79 s. Leur MOYENNE (1.76 s)
    était sur le tableau de bord ; le vrai p99 du service est 4.38 s.

    Les règles (celles de `histogram_quantile` de Prometheus) :

    - un histogramme est une liste de seaux (borne, compte), bornes en secondes strictement croissantes, la
      première > 0, la dernière `math.inf`, avec au moins une borne finie ; le compte d'un seau est CUMULÉ : le
      nombre de requêtes qui ont duré AU PLUS `borne` (il ne baisse donc jamais d'un seau au suivant) ;
    - on ADDITIONNE les histogrammes seau par seau (ils doivent avoir exactement les mêmes bornes), puis on
      cherche le quantile dans cette somme. Jamais de moyenne de quantiles : elle ne correspond à aucune
      requête ;
    - total = le compte du seau `math.inf` ; rang = q × total. On cherche le PREMIER seau fini i dont le
      compte cumulé est >= rang. Le seau i va de bas = la borne du seau précédent (0 pour le premier) à haut =
      sa borne, et contient compte_i − compte_precedent requêtes (compte_precedent = 0 pour le premier).
      Résultat : bas + (haut − bas) × (rang − compte_precedent) / (compte_i − compte_precedent) ;
    - si aucun seau fini n'atteint le rang (la requête cherchée est dans le seau `math.inf`), le résultat est
      la plus grande borne FINIE : on ne sait rien de plus ;
    - `histogrammes` vide, ou total = 0 → None (aucune requête : pas de quantile) ;
    - q est un nombre (pas un bool) avec 0 < q <= 1, sinon ValueError ; des bornes qui ne respectent pas la
      première règle, des histogrammes aux bornes différentes, un compte qui n'est pas un nombre, NaN, négatif
      ou en baisse (des comptes NON cumulés) → ValueError ;
    - ne modifiez pas `histogrammes`.

    Ce que vous apprenez : pourquoi un quantile ne se moyenne pas (mais un histogramme s'additionne), ce que
    calcule `histogram_quantile` de Prometheus (et pourquoi sa réponse n'est jamais plus précise que ses
    seaux), et l'interpolation linéaire.

    Indice : `bornes = [b for b, _ in histogrammes[0]]`, une liste `cumuls` remplie par une double boucle,
    puis un `for i in range(len(bornes) - 1)` qui s'arrête au premier `cumuls[i] >= rang`.
    """
    raise NotImplementedError


def augmentation(echantillons: list[tuple[float, float | None]]) -> Hausse | None:
    """Défi 220 🟡 — Hausse(total, par_seconde) : ce qu'a compté un compteur entre ses lectures, redémarrages compris.

        >>> augmentation([(0, 100), (15, 160), (30, 175), (45, 20), (60, 50)])
        Hausse(total=125, par_seconde=2.0833333333333335)
        >>> augmentation([(0, 100), (15, None), (30, 130)])
        Hausse(total=30, par_seconde=1.0)
        >>> augmentation([(0, 100), (15, None)]) is None
        True
        >>> h = augmentation(compteur_exemple())
        >>> h.total, round(h.par_seconde, 2)
        (2932, 4.89)

    Ici (1er exemple) : de 100 à 160, puis à 175 : 75 requêtes. Puis le compteur vaut 20 : il a BAISSÉ, donc
    le serveur a redémarré et son compteur est reparti de 0 ; depuis, il a compté 20 requêtes. Puis 30 de plus.
    Total : 60 + 15 + 20 + 30 = 125 requêtes en 60 secondes. « La dernière lecture moins la première »
    dirait 50 − 100 = −50 requêtes : absurde. Au 2e exemple, la lecture de t = 15 a échoué : on l'ignore
    (la compter comme un 0 ferait croire à un redémarrage, et compterait 130 requêtes au lieu de 30).
    Sur l'exemple de mardi : 2 932 requêtes en dix minutes, 4.89 par seconde ; le dernier moins le premier
    dirait −17 095.

    Les règles (celles d'`increase` et de `rate` de Prometheus, sans leur extrapolation) :

    - `echantillons` est une liste de lectures (t, valeur) : t en secondes, des nombres STRICTEMENT croissants
      (sinon ValueError) ; valeur est la valeur du compteur à cet instant ;
    - une valeur None ou NaN est une lecture manquée : on l'ignore complètement (ce n'est ni un 0 ni un
      redémarrage). Une valeur négative ou qui n'est pas un nombre → ValueError ;
    - entre deux lectures VALIDES consécutives, l'augmentation vaut apres − avant si apres >= avant ; sinon
      (le compteur a baissé : redémarrage), elle vaut apres (le compteur est reparti de 0) ;
    - total = la somme de ces augmentations ; par_seconde = total / (t de la dernière lecture valide − t de
      la première lecture valide) ;
    - moins de deux lectures valides → None (on ne peut rien dire) ;
    - ne modifiez pas `echantillons`.

    Ce que vous apprenez : ce qu'est un compteur (et pourquoi les outils de surveillance préfèrent un compteur
    qui ne fait que monter à un « nombre de requêtes depuis la dernière fois », qui se perd au moindre
    redémarrage), qu'une donnée manquante n'est pas un zéro, et ce qu'on perd quand même : les requêtes
    comptées entre la dernière lecture et le redémarrage. C'est pour ça qu'on lit souvent (toutes les 15 s).

    Indice : une première boucle qui garde les lectures valides (et vérifie les t), puis
    `for (_, avant), (_, apres) in zip(valides, valides[1:])`.
    """
    raise NotImplementedError


# ============================================================================ Décider


def budget_erreur(minutes: list[tuple[int, int]], objectif: float = OBJECTIF) -> Budget:
    """Défi 221 🟢 — Budget(requetes, erreurs, disponibilite, erreurs_permises, consomme, restant) de la période.

        >>> budget_erreur([(1000, 1), (2000, 0), (1, 1)], objectif=0.999)
        Budget(requetes=3001, erreurs=2, disponibilite=0.9993335554815062, erreurs_permises=3.0010000000000026,
               consomme=0.6664445184938348, restant=0.3335554815061652)
        >>> b = budget_erreur(minutes_exemple())
        >>> b.requetes, b.erreurs, round(b.erreurs_permises), f"{b.disponibilite:.3%}", f"{b.restant:.1%}"
        (5794785, 3774, 5795, '99.935%', '34.9%')

    Ici (1er exemple) : 2 erreurs sur 3 001 requêtes : 99.933 % de réussite. L'objectif de 99.9 % autorise
    0.1 % × 3 001 = 3.001 erreurs : on en a fait 2, les deux tiers du budget sont brûlés. La dernière minute,
    une seule requête, ratée, ne pèse que pour UNE requête : la moyenne des taux de réussite des trois minutes
    dirait (0.999 + 1 + 0) / 3 = 66.6 %.
    Sur novembre : 5 794 785 requêtes, 3 774 erreurs ; l'objectif en autorisait 5 795. Il reste 34.9 % du
    budget : on peut encore déployer, prudemment.

    Les règles :

    - `minutes` est une liste de (requetes, erreurs), deux int (pas des bool) avec 0 <= erreurs <= requetes,
      sinon ValueError ;
    - `objectif` est un nombre (pas un bool) strictement entre 0 et 1 (0.999 ; pas 99.9), sinon ValueError ;
    - requetes et erreurs sont les TOTAUX de la période ;
    - disponibilite = 1 − erreurs / requetes : un rapport de SOMMES. Jamais la moyenne des taux des minutes :
      une minute de nuit à une requête pèserait autant qu'une minute de midi à 300 ;
    - erreurs_permises = (1 − objectif) × requetes (un float) ;
    - consomme = erreurs / erreurs_permises (peut dépasser 1 : l'objectif est manqué) ; restant = 1 − consomme
      (négatif dans ce cas) ;
    - sans aucune requête (liste vide, ou que des minutes à 0) : Budget(0, 0, None, 0.0, None, None) ;
    - ne modifiez pas `minutes`.

    Ce que vous apprenez : ce qu'est un SLO et un budget d'erreur (l'outil qui permet à une équipe de choisir
    entre aller vite et être fiable, chiffres à l'appui), et pourquoi un taux global est un rapport de sommes.

    Indice : deux sommes dans une boucle (qui vérifie chaque minute), puis quatre lignes de calcul.
    """
    raise NotImplementedError


def alertes(minutes: list[tuple[int, int]], objectif: float = OBJECTIF, regles=REGLES) -> list[Alerte]:
    """Défi 222 🔴 — Les alertes déclenchées : Alerte(regle, debut, fin) pour chaque période où une règle est allumée.

        >>> minutes = [(100, 0)] * 10 + [(100, 20)] * 3 + [(100, 0)] * 10    # 3 minutes à 20 % d'erreurs
        >>> alertes(minutes, objectif=0.99, regles=[Regle("page", 6, 2, 8.0)])
        [Alerte(regle='page', debut=12, fin=13)]
        >>> for a in alertes(minutes_exemple()):
        ...     print(f"{a.regle:<10} de {horloge(a.debut)} à {horloge(a.fin)}")
        page 1 h   de 12 nov. 14:12 à 12 nov. 14:45
        ticket 3 j de 12 nov. 14:21 à 12 nov. 20:37
        page 6 h   de 12 nov. 14:27 à 12 nov. 15:08
        ticket 1 j de 12 nov. 14:28 à 12 nov. 16:36
        ticket 3 j de 14 nov. 04:52 à 14 nov. 05:05
        ticket 3 j de 14 nov. 07:07 à 14 nov. 07:19
        ticket 3 j de 24 nov. 11:56 à 25 nov. 07:32
        ticket 1 j de 24 nov. 15:21 à 25 nov. 00:29
        ticket 1 j de 25 nov. 02:10 à 25 nov. 04:18

    Ici (1er exemple) : les minutes 10, 11 et 12 ont 20 % d'erreurs. L'objectif de 99 % permet 1 % d'erreurs ; la
    règle s'allume quand le budget brûle au moins 8 fois trop vite (8 % d'erreurs) À LA FOIS sur les 6 dernières
    minutes et sur les 2 dernières. Minute 11 : la fenêtre longue (minutes 6 à 11) n'a que 40 erreurs sur 600
    requêtes (6.7 %) : éteinte, même si la courte dit 20 %. Minute 12 : 60 sur 600 (10 %) et 40 sur 200 (20 %) :
    allumée. Minute 13 : 10 % et 10 % : allumée. Minute 14 : la fenêtre longue dit encore 10 %, mais la courte
    (minutes 13 et 14) n'a plus d'erreur : l'incident est fini, l'alerte s'éteint.
    Sur novembre : le 12, la mise en production ratée de 14 h 02 (réparée à 14 h 41) déclenche « page 1 h » à
    14 h 12, éteinte à 14 h 45. Le hoquet de deux minutes du 18 ne réveille personne. La fuite lente du 24
    (0.5 % d'erreurs pendant 20 heures) ne réveille personne non plus, mais ouvre des tickets. Avec la seule
    fenêtre courte, Samir aurait été réveillé 56 fois dans le mois, presque toujours la nuit, pour UNE erreur
    parmi dix requêtes ; avec la seule fenêtre longue, la page du 12 serait restée allumée jusqu'à 15 h 30,
    49 minutes après la réparation.

    Les règles (les alertes « multi-fenêtres » du Site Reliability Workbook de Google) :

    - les minutes sont numérotées 0, 1, 2… dans l'ordre de `minutes` ; chacune est (requetes, erreurs), comme au
      défi 221 (mêmes vérifications, même `objectif`) ;
    - la combustion sur une fenêtre de W minutes qui finit à la minute t (incluse) couvre les minutes
      max(0, t − W + 1) à t (au début, la fenêtre prend les minutes qui existent). Elle vaut
      (erreurs de la fenêtre / requêtes de la fenêtre) / (1 − objectif) ; une fenêtre sans requête vaut 0 ;
    - une Regle(nom, longue, courte, seuil) est ALLUMÉE à la minute t si la combustion sur la fenêtre `longue`
      ET celle sur la fenêtre `courte` sont toutes deux >= seuil ;
    - pour chaque règle, chaque période de minutes CONSÉCUTIVES où elle est allumée donne une
      Alerte(nom, debut, fin) (debut et fin inclus) ;
    - le résultat est la liste de ces alertes, triée par debut, puis, à debut égal, dans l'ordre de `regles` ;
    - dans une règle, longue et courte sont des int (pas des bool) >= 1 avec courte <= longue, et seuil un
      nombre > 0, sinon ValueError ;
    - ne modifiez pas `minutes`. Le correcteur vous passe 14 400 minutes dans une `Comptee`, qui compte les
      cases lues : relire chaque fenêtre en entier à chaque minute (près de cent millions de lectures) est
      refusé. Lisez chaque minute UNE fois.

    Ce que vous apprenez : pourquoi une alerte doit dire « c'est grave » (fenêtre longue) ET « c'est encore en
    cours » (fenêtre courte), les sommes cumulées (la somme de n'importe quelle fenêtre en une soustraction),
    et regrouper des instants consécutifs en périodes.

    Indice : `cumul_r = [0]` puis `cumul_r.append(cumul_r[-1] + requetes)` pour chaque minute (idem pour les
    erreurs) : les requêtes des minutes a à t valent `cumul_r[t + 1] - cumul_r[a]`. Puis, règle par règle, une
    boucle sur t qui note `debut` quand la règle s'allume et ajoute l'Alerte quand elle s'éteint (sans oublier
    une alerte encore allumée à la dernière minute).
    """
    raise NotImplementedError


# ============================================================================ Comprendre


def temps_propre(spans: list[Span]) -> dict[str, float]:
    """Défi 223 🔴 — {service: son temps PROPRE total, en ms} : le temps où il travaillait lui-même, sans attendre un appel.

        >>> temps_propre([Span("a", None, "api", 0, 100),       # l'API répond en 100 ms
        ...               Span("b", "a", "crm", 10, 70),         # elle appelle le CRM deux fois EN PARALLÈLE
        ...               Span("c", "a", "crm", 20, 80),
        ...               Span("d", "a", "audit", 90, 160),      # et lance l'audit sans l'attendre
        ...               Span("e", "zz", "file", 0, 30)])       # un span dont le parent s'est perdu
        {'api': 20, 'crm': 120, 'audit': 70, 'file': 30}
        >>> temps_propre(trace_exemple())
        {'passerelle': 22, 'api': 26, 'cache': 2, 'crm': 10818, 'modele': 15, 'base': 80, 'audit': 515, 'file': 300}

    Ici (1er exemple) : l'API dure 100 ms. Pendant ce temps, elle attend le CRM de 10 à 80 (les deux appels se
    chevauchent : 70 ms d'attente, pas 60 + 60 = 120) et l'audit de 90 à 100 (il continue après la réponse de
    l'API, mais seule la partie DANS l'API la fait attendre). Temps propre de l'API : 100 − 70 − 10 = 20 ms.
    Le CRM a travaillé 60 + 60 = 120 ms (deux appels), l'audit 70 ms. Le span « file » est orphelin : il compte
    quand même.
    Sur la requête lente de mardi (4.18 s) : la passerelle et l'API ne travaillent que 48 ms. Additionner les
    durées des enfants dirait que l'API a un temps propre de −7 272 ms : ses trois appels au CRM, en parallèle,
    durent 10.8 s à eux trois.

    Les règles :

    - un Span(id, parent, service, debut, fin) : debut et fin en ms, `parent` l'id du span qui l'a appelé
      (None pour la racine). Les ENFANTS d'un span sont les spans dont `parent` est son id ;
    - temps propre d'un span = sa durée (fin − debut) moins la longueur de la RÉUNION des intervalles de ses
      enfants, chacun COUPÉ à l'intervalle de son parent : [max(debut_enfant, debut), min(fin_enfant, fin)]
      (un enfant qui commence ou finit hors du parent n'est compté que pour la partie commune) ;
    - seuls les enfants DIRECTS comptent (le temps des petits-enfants est déjà dans celui de leur parent) ;
    - un span dont le parent n'est pas dans la liste (orphelin) est traité comme une racine : son temps propre
      compte comme celui des autres ;
    - le résultat est un dict {service: somme des temps propres de ses spans}, rangé dans l'ordre où chaque
      service APPARAÎT dans `spans` ; `spans` vide → {} ;
    - un span avec fin < debut (ou un debut ou une fin qui n'est pas un nombre), ou deux spans avec le même id
      → ValueError ;
    - les spans peuvent être dans n'importe quel ordre (un enfant avant son parent) ;
    - ne modifiez pas `spans`. Le correcteur vous passe une trace de 40 000 spans, dont un qui a 20 000 enfants,
      et une chaîne de 5 000 appels imbriqués : chercher les enfants de chaque span dans toute la liste, ou
      parcourir l'arbre par une fonction récursive, se voit.

    Ce que vous apprenez : lire une trace distribuée (ce qu'affichent Jaeger, Zipkin ou Grafana Tempo), la
    réunion d'intervalles (trier, puis fusionner), et grouper par parent en une passe plutôt que chercher.

    Indice : un dict `enfants = {}` rempli en une passe (`enfants.setdefault(span.parent, []).append(span)`),
    puis, pour chaque span, `sorted(...)` des intervalles coupés de ses enfants et une variable `jusqu_a` : la
    fin de la partie déjà couverte.
    """
    raise NotImplementedError

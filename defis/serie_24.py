"""Série 24 — Orchestrer le traitement de nuit : un graphe de dépendances, de l'ordre d'exécution au planning.

🌙 VOUS VOULEZ DÉVELOPPER ? UN JOUR, VOTRE SCRIPT DEVIENDRA DIX SCRIPTS QUI DÉPENDENT LES UNS DES AUTRES.

Chaque nuit, le projet doit extraire les clients, les factures et les appels, nettoyer, calculer les
variables, réentraîner, évaluer, scorer, envoyer les SMS, puis refaire le rapport et le tableau de bord.
Chaque étape a besoin de certaines autres : c'est un GRAPHE DE DÉPENDANCES. Airflow, Dagster, Prefect,
`make`, `dvc` ou GitHub Actions (`needs:`) font tous la même chose : lire ce graphe et en déduire quoi
lancer, dans quel ordre, et quoi faire quand une étape plante. Cette série vous le fait écrire :

    146 l'ordre d'exécution (et le refus d'un cycle)     147 ce qu'il faut relancer après une modification
    148 trouver le cycle, même dans 5 000 tâches          149 exécuter le traitement sans qu'un échec bloque tout
    150 le chemin critique : la durée minimale de la nuit
    151 le planning sur k travailleurs (une simulation par événements)

C'est aussi une famille de questions d'entretien très classique (« course schedule », « build order »,
« détecter un cycle dans un graphe orienté »). Prérequis : la série 00, les dictionnaires et les `set`
(défi 45), et un tas `heapq` (série 09, défi 60, ou l'indice du défi 146). Pas de NumPy ni de pandas.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- le graphe est un DICTIONNAIRE {tâche: liste de ses prérequis}. `{"variables": ["nettoyer",
  "extraire_appels"]}` veut dire : « variables » ne peut commencer que quand « nettoyer » ET
  « extraire_appels » sont finies. Une tâche citée seulement comme prérequis (`extraire_appels`) est une
  tâche comme les autres, même si elle n'est pas une clé ;
- un ORDRE TOPOLOGIQUE est un ordre où chaque tâche vient après tous ses prérequis. Il n'existe que s'il
  n'y a pas de CYCLE (a attend b, b attend c, c attend a : personne ne peut commencer) ;
- l'algorithme de KAHN le construit : on compte, pour chaque tâche, ses prérequis pas encore faits. Une
  tâche dont le compte vaut 0 est PRÊTE. On en prend une, on la « fait », et on retire 1 au compte de
  chaque tâche qui l'attendait. S'il reste des tâches à la fin, elles sont bloquées par un cycle ;
- pour savoir vite « qui attend cette tâche ? », on RETOURNE les flèches une fois pour toutes :
  {tâche: liste des tâches qui en ont besoin} (les « successeurs ») ;
- un graphe peut avoir beaucoup de chemins qui mènent au même endroit (des « losanges ») : sans un
  ensemble « déjà vu », un parcours repasse par les mêmes tâches un nombre EXPONENTIEL de fois.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_24 import *
    >>> pipeline = pipeline_exemple()
    >>> pipeline["evaluer"]
    ['entrainer', 'variables']
    >>> import heapq
    >>> tas = ["sms", "rapport", "evaluer"]; heapq.heapify(tas)
    >>> heapq.heappop(tas)                         # la plus petite par ordre alphabétique
    'evaluer'
    >>> import graphlib                            # la version de la bibliothèque standard, pour comparer
    >>> len(list(graphlib.TopologicalSorter(pipeline).static_order()))
    12

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites les défis DANS L'ORDRE : 147, 149,
   150 et 151 se servent de votre `ordre_execution`. Écrivez aussi vos propres fonctions utilitaires
   (par exemple `_graphe` et `_successeurs`, voir l'indice du défi 146) : vous les réutiliserez partout.
2. Lancez `python scripts/defis.py --defi 146` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_24.py`.

Les outils communs (`pipeline_exemple`, `durees_exemple`, `taches_du_graphe`, `generer_pipeline`,
`generer_durees`, `afficher_planning`) sont dans `defis/nuit.py`, déjà importés ci-dessous.
`graphlib.TopologicalSorter` est INTERDIT dans vos réponses (c'est ce que vous écrivez), mais pratique
pour comparer.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import heapq  # noqa: F401 — un tas : heapq.heappush(tas, x), heapq.heappop(tas) rend le plus petit
import math  # noqa: F401
import numbers  # noqa: F401 — utile pour vérifier qu'un paramètre est un nombre (numbers.Real, numbers.Integral)
from collections import deque  # noqa: F401 — une file : file.append(x), file.popleft()

from defis.nuit import (  # noqa: F401
    afficher_planning,
    durees_exemple,
    generer_durees,
    generer_pipeline,
    pipeline_exemple,
    taches_du_graphe,
)

# ============================================================================ L'ordre d'exécution


def ordre_execution(dependances: dict) -> list[str]:
    """Défi 146 🟢 — Dans quel ordre lancer les tâches ? (l'algorithme de Kahn)

    Renvoie la liste de TOUTES les tâches du graphe (clés et prérequis), chacune une fois, dans un ordre
    où chaque tâche vient après ses prérequis. Plusieurs ordres conviennent souvent ; pour que deux nuits
    donnent le même (journaux comparables, tests reproductibles), la règle est : parmi les tâches PRÊTES
    (tous leurs prérequis faits), on prend toujours la PREMIÈRE PAR ORDRE ALPHABÉTIQUE.

        >>> ordre_execution({"c": ["a"], "b": ["a"], "a": []})
        ['a', 'b', 'c']
        >>> ordre_execution(pipeline_exemple())[:5]
        ['extraire_appels', 'alerte_support', 'extraire_clients', 'extraire_factures', 'nettoyer']

    Au départ, trois tâches sont prêtes (les trois « extraire_ »). Dès qu'`extraire_appels` est faite,
    `alerte_support` devient prête, et elle passe AVANT `extraire_clients` (« a » < « e ») : on ne
    raisonne pas par « vagues ».

    Les règles :

    - une tâche citée seulement comme prérequis fait partie du résultat ; un graphe vide donne [] ;
    - un prérequis écrit deux fois (`["nettoyer", "nettoyer"]`) ne compte qu'une fois ;
    - un CYCLE (y compris une tâche qui s'attend elle-même) lève `ValueError`, avec un message qui
      contient le mot « cycle » et le nom des tâches bloquées ;
    - les prérequis sont une liste (ou un tuple, ou un ensemble) de noms. Une CHAÎNE à la place
      (`{"variables": "nettoyer"}`) lève `TypeError` : sinon Python la lirait lettre par lettre, et la
      tâche aurait pour prérequis « n », « e », « t »… ;
    - `dependances` n'est pas modifié ;
    - 20 000 tâches en moins d'une seconde : pas de boucle qui relit toutes les tâches à chaque étape
      pour trouver la prochaine prête (O(n²)), pas de `sorted` ni de `min` sur toutes les tâches prêtes
      à chaque tour.

    Ce que vous apprenez : le tri topologique, retourner un graphe, un compteur par tâche, un tas
    (`heapq`) pour toujours sortir la plus petite en O(log n), refuser une entrée mal formée.

    Indice : écrivez d'abord deux outils, que les autres défis réutiliseront :
    `_graphe(dependances)` renvoie {tâche: set(prérequis)} pour TOUTES les tâches (un `set` élimine les
    doublons ; `graphe.setdefault(nom, set())` ajoute les tâches citées seulement comme prérequis), et
    `_successeurs(graphe)` renvoie {tâche: [les tâches qui l'attendent]}. Ensuite :
    `restants = {t: len(p) for t, p in graphe.items()}`, un tas des tâches à 0, et
    `while tas: t = heapq.heappop(tas)` … Si `len(ordre) < len(graphe)` à la fin : il y a un cycle.
    """
    raise NotImplementedError


# ============================================================================ Ce qu'il faut relancer


def taches_a_relancer(dependances: dict, modifiees) -> list[str]:
    """Défi 147 🟢 — On a corrigé une tâche : qu'est-ce qu'il faut relancer ?

    Le nettoyage avait un bogue, il est corrigé. Il faut relancer `nettoyer`… et TOUT ce qui en dépend,
    directement ou non (`variables`, puis `entrainer`, puis…), mais rien d'autre : les extractions, qui
    ne dépendent pas du nettoyage, sont restées justes. C'est ce que fait `make` après la modification
    d'un fichier, ou Airflow avec « clear downstream ».

        >>> taches_a_relancer(pipeline_exemple(), ["evaluer"])
        ['evaluer', 'rapport', 'tableau_de_bord']
        >>> taches_a_relancer(pipeline_exemple(), ["sms", "alerte_support"])
        ['alerte_support', 'sms']
        >>> taches_a_relancer(pipeline_exemple(), [])
        []

    Les règles :

    - le résultat contient les tâches de `modifiees` et toutes leurs DESCENDANTES (les tâches qui en
      dépendent, à n'importe quelle distance), chacune une fois ;
    - dans l'ordre où le traitement les lancera vraiment : l'ordre de `ordre_execution(dependances)`
      sur le graphe ENTIER (une tâche relancée peut attendre un prérequis qui, lui, ne l'est pas) ;
    - `modifiees` est n'importe quel itérable de noms : une liste, un ensemble… ou un générateur, qu'on
      ne peut lire qu'UNE fois. Une chaîne seule lève `TypeError` ;
    - une tâche inconnue lève `ValueError` (avec son nom) ; un cycle aussi (comme au défi 146) ;
    - un graphe « en losanges » de 30 étages (2^30 chemins) se traite en un instant.

    Ce que vous apprenez : parcourir un graphe dans le sens des flèches retournées (en largeur, avec
    une `deque`), l'ensemble « déjà vu » qui évite l'explosion exponentielle, lire un itérable une
    seule fois, réutiliser sa propre fonction.

    Indice : `depart = list(modifiees)` (une seule lecture), puis un parcours en largeur sur
    `_successeurs(...)` : `a_relancer = set(depart)`, `file = deque(depart)`, et on n'ajoute à la file
    qu'une tâche PAS ENCORE dans `a_relancer`. Pour l'ordre : `[t for t in ordre_execution(...) if t in
    a_relancer]`.
    """
    raise NotImplementedError


# ============================================================================ Trouver un cycle


def trouver_cycle(dependances: dict) -> list[str] | None:
    """Défi 148 🟡 — Le traitement refuse de démarrer : où est le cycle ?

    `ordre_execution` dit qu'il y a un cycle et nomme les tâches bloquées, mais dans un graphe de 400
    tâches, il en bloque peut-être 200 : il faut LE cycle, pour savoir quelle flèche supprimer. Renvoyez
    `None` s'il n'y en a pas, sinon une liste [t0, t1, …, tk] où chaque tâche a besoin de la suivante
    (t1 est un prérequis de t0, t2 de t1…) et où la dernière est la première (tk == t0).

        >>> trouver_cycle(pipeline_exemple()) is None
        True
        >>> trouver_cycle({"rapport": ["scorer"], "scorer": ["rapport"]}) in (["rapport", "scorer", "rapport"],
        ...                                                                   ["scorer", "rapport", "scorer"])
        True
        >>> trouver_cycle({"a": ["a"]})
        ['a', 'a']

    Les règles :

    - n'importe quel cycle convient s'il y en a plusieurs, mais c'est un VRAI cycle : chaque flèche
      existe, et aucune tâche n'y apparaît deux fois (sauf la première, répétée à la fin). Pas le chemin
      qui y MÈNE : avec `{"x": ["a"], "a": ["b"], "b": ["a"]}`, la réponse est [a, b, a] ou [b, a, b],
      pas [x, a, b, a] ;
    - « déjà vue » ne veut PAS dire « cycle » : `evaluer` et `scorer` ont toutes les deux besoin de
      `variables`, on la croise donc deux fois, et le traitement de nuit n'a pourtant aucun cycle ;
    - une chaîne de 5 000 tâches (`t0` a besoin de `t1`, qui a besoin de `t2`…) ne lève pas
      `RecursionError`. Python s'arrête à 1 000 appels imbriqués : une fonction récursive ne suffit
      pas, et `sys.setrecursionlimit` est interdit (au-delà de quelques milliers, c'est Python lui-même
      qui plante, sans message) ;
    - les mêmes entrées que le défi 146 (une chaîne à la place d'une liste lève `TypeError`) ;
      `dependances` n'est pas modifié ; un graphe en losanges de 30 étages se traite en un instant.

    Ce que vous apprenez : le parcours en PROFONDEUR avec trois états (jamais vue, en cours, terminée),
    remplacer la récursion par une pile explicite, extraire le cycle du chemin courant.

    Indice : un parcours en profondeur garde le CHEMIN actuel (les tâches « en cours »). Arriver sur une
    tâche en cours, c'est boucler : le cycle est `chemin[chemin.index(t):] + [t]`. Arriver sur une tâche
    TERMINÉE n'est pas un cycle : on l'ignore. Sans récursion : une pile d'itérateurs,
    `pile = [iter(prerequis_du_depart)]`, et à chaque tour `suivante = next(pile[-1], None)` ; `None`
    veut dire que la tâche du haut est terminée (on la retire du chemin ET de la pile).
    """
    raise NotImplementedError


# ============================================================================ Exécuter le traitement


def executer(taches: dict, dependances: dict) -> tuple[dict, dict, dict]:
    """Défi 149 🟡 — Exécuter le traitement de nuit, sans qu'un échec bloque ce qui n'en dépend pas.

    `taches` associe à chaque tâche une FONCTION. On les appelle dans l'ordre de `ordre_execution`, et
    chaque fonction reçoit les RÉSULTATS de ses prérequis en paramètres NOMMÉS : la fonction de
    « variables » est appelée `variables_f(nettoyer=<ce qu'a renvoyé nettoyer>, extraire_appels=<…>)`.
    Si une tâche plante, tout ce qui en dépend est SAUTÉ ; les tâches indépendantes, elles, tournent
    normalement (le SMS d'alerte part même si l'entraînement a planté).

        >>> def extraire():
        ...     return [3, 1, 2]
        >>> def trier(extraire):                  # le paramètre porte le NOM du prérequis
        ...     return sorted(extraire)
        >>> def compter(extraire):
        ...     raise RuntimeError("base indisponible")
        >>> def rapport(compter):
        ...     return f"{compter} clients"
        >>> statuts, resultats, erreurs = executer(
        ...     {"extraire": extraire, "trier": trier, "compter": compter, "rapport": rapport},
        ...     {"trier": ["extraire"], "compter": ["extraire"], "rapport": ["compter"]})
        >>> statuts                               # dans l'ordre d'exécution (défi 146)
        {'extraire': 'ok', 'compter': 'echec', 'rapport': 'sautee', 'trier': 'ok'}
        >>> resultats
        {'extraire': [3, 1, 2], 'trier': [1, 2, 3]}
        >>> erreurs
        {'compter': RuntimeError('base indisponible')}

    `rapport` n'a pas été appelée (son prérequis a échoué), mais `trier`, qui ne dépend pas de
    `compter`, a tourné normalement.

    Les règles :

    - renvoie trois dictionnaires : `statuts` ({tâche: "ok", "echec" ou "sautee"}, TOUTES les tâches,
      dans l'ordre d'exécution), `resultats` ({tâche: valeur renvoyée}, les tâches « ok » seulement, même
      si la valeur est None) et `erreurs` ({tâche: l'exception levée}, les tâches « echec » seulement) ;
    - chaque fonction est appelée au plus une fois, avec les résultats de ses prérequis en paramètres
      nommés, et rien d'autre ;
    - une tâche dont un prérequis a échoué OU a été sauté est « sautee », et sa fonction n'est pas
      appelée (les petits-enfants d'une tâche en échec aussi) ;
    - on attrape les `Exception`, pas plus : un Ctrl+C (`KeyboardInterrupt`) arrête tout, tout de suite ;
    - AVANT de lancer quoi que ce soit, on vérifie tout : un cycle, une tâche du graphe sans fonction,
      ou une fonction pour une tâche absente du graphe (une faute de frappe : « netoyer ») lèvent
      `ValueError`, et AUCUNE fonction n'a été appelée.

    Ce que vous apprenez : déléguer l'ordre à `ordre_execution`, propager un échec à toute la
    descendance sans parcours supplémentaire, appeler une fonction avec `f(**dictionnaire)`, la
    différence entre `Exception` et `BaseException`, valider avant d'agir.

    Indice : parcourez `ordre_execution(dependances)`. Pour chaque tâche : si un de ses prérequis n'a
    pas le statut "ok", elle est "sautee" (comme on suit l'ordre, les statuts des prérequis sont déjà
    connus : c'est ce qui propage l'échec aux petits-enfants). Sinon : `try: resultats[t] =
    taches[t](**{p: resultats[p] for p in prerequis})`, `except Exception as exc:` → "echec", et
    `else:` → "ok".
    """
    raise NotImplementedError


# ============================================================================ Le chemin critique


def chemin_critique(durees: dict, dependances: dict) -> tuple[float, list[str]]:
    """Défi 150 🟡 — Combien de temps dure la nuit, au minimum ? Et quelles tâches faut-il accélérer ?

    Avec autant de machines qu'on veut, chaque tâche commence dès que son DERNIER prérequis est fini :
    début(t) = la plus grande fin de ses prérequis (0 s'il n'en a pas), fin(t) = début(t) + durée(t). La
    nuit dure jusqu'à la plus grande fin. La suite de tâches qui y mène sans un instant de battement est
    le CHEMIN CRITIQUE : accélérer une autre tâche ne fait rien gagner.

        >>> chemin_critique({"a": 10, "b": 5, "c": 20}, {"c": ["a", "b"]})
        (30, ['a', 'c'])
        >>> total, chemin = chemin_critique(durees_exemple(), pipeline_exemple())
        >>> total, sum(durees_exemple().values())      # 150 minutes, au lieu de 200 l'une après l'autre
        (150, 200)
        >>> chemin
        ['extraire_factures', 'nettoyer', 'variables', 'entrainer', 'evaluer', 'rapport', 'tableau_de_bord']

    `evaluer` et `scorer` durent 15 minutes et commencent en même temps : elles finissent ensemble, à
    120 + 15 = 135. Le rapport attend les deux ; à égalité, on remonte par la première par ordre
    alphabétique (`evaluer`).

    Les règles :

    - renvoie (durée totale, chemin) ; le chemin va du DÉBUT à la fin : il commence par une tâche sans
      prérequis, chaque tâche y est le prérequis qui finit le plus tard de la suivante, et il se termine
      par la tâche qui finit le plus tard. À égalité, toujours la première par ordre alphabétique ;
    - un graphe vide donne (0, []) ;
    - `durees` a une entrée pour CHAQUE tâche du graphe, et aucune autre (sinon `ValueError` : une faute
      de frappe se voit tout de suite). Une durée est un nombre positif ou nul (0 est permis) : négative,
      NaN ou infinie → `ValueError` ; `True`, "10" ou None → `TypeError` ;
    - un cycle lève `ValueError` ; les entrées ne sont pas modifiées ;
    - une chaîne de 5 000 tâches (pas de `RecursionError`) et un graphe en losanges de 30 étages se
      traitent en un instant.

    Ce que vous apprenez : la programmation dynamique sur un graphe (calculer chaque tâche UNE fois,
    dans l'ordre topologique), retenir d'où vient le maximum pour reconstruire le chemin, départager
    les égalités.

    Indice : parcourez `ordre_execution(dependances)` : les fins des prérequis sont alors déjà connues.
    Pour chaque tâche, retenez le prérequis décisif : `min(prerequis, key=lambda p: (-fin[p], p),
    default=None)` (la plus grande fin, puis le plus petit nom). À la fin, partez de la tâche qui finit
    le plus tard et remontez les prérequis décisifs, puis retournez la liste (`chemin[::-1]`).
    """
    raise NotImplementedError


# ============================================================================ Le planning


def planifier(durees: dict, dependances: dict, travailleurs: int = 2) -> dict[str, tuple]:
    """Défi 151 🔴 — Le planning de la nuit sur k machines : une simulation par événements.

    En vrai, on n'a pas une machine par tâche, mais k « travailleurs », numérotés de 1 à k. Simulez la
    nuit, en suivant la règle de la plupart des ordonnanceurs (« ordonnancement par liste ») :

    1. à l'instant t, on TERMINE toutes les tâches dont la fin vaut t : leurs travailleurs sont libres,
       et les tâches qui les attendaient peuvent devenir prêtes ;
    2. puis, tant qu'il y a un travailleur libre ET une tâche prête : le travailleur libre de plus petit
       numéro prend la tâche prête la plus PRIORITAIRE, qui commence à t et finit à t + durée ;
    3. on saute à la prochaine fin de tâche, et on recommence. Une tâche de durée 0 finit à l'instant où
       elle commence : on revient à l'étape 1 SANS avancer le temps.

    La plus prioritaire est celle dont le RESTE est le plus long : reste(t) = durée(t) + le plus grand
    reste des tâches qui attendent t (0 si aucune), c'est-à-dire la durée du plus long chemin qui part
    d'elle. Commencer par elle, c'est attaquer le chemin critique au plus tôt. À égalité, la première par
    ordre alphabétique.

        >>> planifier({"a": 10, "b": 5, "c": 20}, {"c": ["a", "b"]}, travailleurs=1)
        {'a': (0, 10, 1), 'b': (10, 15, 1), 'c': (15, 35, 1)}
        >>> plan = planifier(durees_exemple(), pipeline_exemple(), travailleurs=2)
        >>> plan["extraire_factures"], plan["extraire_clients"], plan["extraire_appels"]
        ((0, 25, 1), (0, 10, 2), (10, 25, 2))
        >>> max(fin for debut, fin, travailleur in plan.values())     # 2 machines suffisent : 150 minutes
        150
        >>> afficher_planning(plan, largeur=80)                       # pour voir (pas besoin pour le défi)
        travailleur 1 : [extraire_fa][nettoyer ][variables     ][entrainer             ][evalue][rap][t]   0 → 150
        travailleur 2 : [ext][extrai][a]                                                [scorer][s]        0 → 150

    Au départ, les trois extractions sont prêtes : `extraire_factures` (reste 150), `extraire_clients`
    (135) et `extraire_appels` (120). Les deux travailleurs prennent les deux premières ; par ordre
    alphabétique, on aurait pris `extraire_appels` en premier, et la nuit aurait duré 160 minutes.

    Les règles :

    - renvoie {tâche: (début, fin, travailleur)} pour toutes les tâches, dans l'ordre des débuts ;
    - `travailleurs` est un entier >= 1 (0 → `ValueError` ; 1.5, "2" ou True → `TypeError`) ; `durees`
      suit les règles du défi 150 ; un cycle lève `ValueError` ; les entrées ne sont pas modifiées ;
    - avec 1 travailleur, la nuit dure la somme des durées ; avec autant de travailleurs que de tâches,
      elle dure exactement le chemin critique (défi 150) ;
    - les durées peuvent être grandes (des secondes : 90 000 pour 25 heures) ou décimales (2,5) : on
      SAUTE de fin en fin, on n'avance pas le temps d'une unité à chaque tour ;
    - 20 000 tâches et 8 travailleurs en moins de 2 secondes.

    Ce que vous apprenez : la simulation par événements, trois tas qui travaillent ensemble (les tâches
    prêtes par priorité, les travailleurs libres par numéro, les tâches en cours par heure de fin),
    traiter tous les événements d'un même instant avant de décider, une priorité calculée à rebours.

    Indice : calculez `reste` en parcourant `ordre_execution(...)` À L'ENVERS. Ensuite trois tas :
    `prets` contient des couples (-reste, nom) (le plus petit couple est le plus prioritaire), `libres`
    les numéros des travailleurs libres, `en_cours` des triplets (fin, travailleur, nom). La boucle :
    `while prets or en_cours:` attribuer tant que possible ; puis `temps = en_cours[0][0]` et
    `while en_cours and en_cours[0][0] == temps:` terminer (libérer, débloquer les successeurs).
    """
    raise NotImplementedError

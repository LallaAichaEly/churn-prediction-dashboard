"""Série 18 — Rendre le code rapide, preuves à l'appui : mesurer, profiler, bisect, cache, mémoire, optimiser.

⏱️ VOUS VOULEZ DÉVELOPPER ? UN CODE LENT, C'EST UN CODE QU'ON N'OSE PLUS LANCER.

Le rapport de rétention tournait en 2 minutes. Depuis que le fichier a grossi, il met 40 minutes, puis
la machine manque de mémoire. La tentation : réécrire au hasard ce qui « a l'air lent ». C'est presque
toujours au mauvais endroit. Un·e développeur·se procède dans l'ordre : MESURER (combien de temps,
vraiment ?), TROUVER où le temps passe (profiler), CHANGER d'algorithme là, et seulement là, puis
VÉRIFIER que le résultat n'a pas changé d'un octet. Les six défis suivent cette méthode :

    110 chronométrer sans se tromper              111 trouver le point chaud avec cProfile
    112 le bon tarif sans tout parcourir (bisect) 113 un cache qui ne grossit pas et ne ment pas
    114 un fichier de 10 Go avec 1 Mo de mémoire  115 accélérer une fonction sans rien changer

Prérequis : la série 00, les dictionnaires et les `set` (défi 45, série 07) et le décorateur du défi 49.
Rien à installer : tout fait partie de Python.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- `time.perf_counter()` est LE chronomètre : précis, et il ne recule jamais. `time.time()` donne l'heure
  du système, qui peut reculer (synchronisation de l'horloge) et n'est précise qu'à 16 ms sous Windows ;
- une mesure isolée ne vaut rien : un autre programme, le ramasse-miettes ou le disque ralentissent un
  passage au hasard. On répète, et on garde le MEILLEUR temps (le minimum) : c'est le coût du code
  lui-même, le reste est du bruit. C'est ce que fait `python -m timeit` ;
- un PROFILEUR (`cProfile`) note, pour chaque fonction, combien de fois elle est appelée (`ncalls`), le
  temps passé DANS son propre code (`tottime`) et le temps cumulé avec tout ce qu'elle appelle
  (`cumtime`). `python -m cProfile -s tottime mon_script.py` le fait sur un script entier ;
- la complexité compte plus que tout le reste : une boucle dans une boucle sur 400 000 clients, c'est
  160 milliards de tours. Remplacer la boucle intérieure par un dictionnaire (O(1)) ou une recherche
  dichotomique dans une liste triée (`bisect`, O(log n)) gagne des heures, là où « optimiser » une ligne
  gagne des millisecondes ;
- un CACHE évite de refaire un calcul, mais il doit rester borné (sinon il mange la mémoire) et
  périmer ses valeurs (sinon il sert des réponses fausses) ;
- `tracemalloc` mesure la mémoire allouée par Python, et son pic : c'est ainsi que le correcteur du
  défi 114 voit si vous avez chargé tout le fichier.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_18 import *
    >>> chronometrer(programme_exemple)                   # défi 110 : quelques centièmes de seconde
    >>> point_chaud(programme_exemple)                    # défi 111
    'calculer_scores'

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py --defi 110` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_18.py`.

Les outils d'essai (`programme_exemple`, `TARIFS_EXEMPLE`, `ecrire_factures`, `clients_a_rappeler_lent`,
`exemple_rappels` et l'horloge `HorlogeFactice`) sont dans `defis/performance.py`, déjà importés
ci-dessous : vous n'avez pas besoin d'ouvrir ce fichier.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import bisect  # noqa: F401 — utile au défi 112
import cProfile  # noqa: F401 — utile au défi 111
import csv  # noqa: F401 — utile au défi 114
import functools  # noqa: F401 — utile au défi 113
import pstats  # noqa: F401 — utile au défi 111
import time
from collections import Counter, OrderedDict  # noqa: F401 — utiles aux défis 113 et 115
from datetime import date
from pathlib import Path
from typing import Callable

from defis.performance import (  # noqa: F401 — tout est utile pour essayer vos fonctions
    CONTRATS,
    TARIFS_EXEMPLE,
    HorlogeFactice,
    clients_a_rappeler_lent,
    ecrire_factures,
    exemple_rappels,
    programme_exemple,
)

# ============================================================================ Mesurer


def chronometrer(fonction: Callable[[], object], repetitions: int = 5,
                 horloge: Callable[[], float] = time.perf_counter) -> float:
    """Défi 110 🟢 — Chronométrer sans se tromper : répéter, et garder le meilleur temps.

    « J'ai optimisé, c'est 3 fois plus rapide ! » Vraiment ? Une seule mesure peut tomber pendant que
    l'antivirus scanne le disque. Avant de toucher au code, il faut un chronomètre fiable.

    Lancez `fonction()` (sans argument) `repetitions` fois, mesurez CHAQUE appel séparément avec
    `horloge()` (lue juste avant et juste après l'appel), et renvoyez la durée la plus courte, en
    secondes (un float).

        >>> chronometrer(programme_exemple)                                 # en secondes
        0.026…
        >>> chronometrer(lambda: sorted(range(100_000)), repetitions=10)    # une fonction avec arguments ?
        0.003…                                                              # une lambda sans argument

    Les règles :

    - le paramètre `horloge` vaut `time.perf_counter` par défaut, et c'est LUI que vous appelez (pas
      `time.perf_counter()` directement) : le correcteur vous passe une fausse horloge qu'il avance à la
      main, pour savoir exactement ce que chaque appel « a duré » ;
    - le MINIMUM, pas la moyenne : une moyenne mélange le coût du code et les ralentissements venus
      d'ailleurs. Le minimum est la meilleure estimation de ce que coûte le code lui-même ;
    - `repetitions` inférieur à 1 lève ValueError, avec un message qui nomme `repetitions` ;
    - si `fonction()` lève une exception, elle remonte telle quelle : un chronomètre qui avale les
      erreurs mesure la vitesse d'un plantage.

    Ce que vous apprenez : à mesurer avant d'affirmer, et pourquoi `perf_counter` plutôt que `time`.

    Indice : une liste `durees`, une boucle `for _ in range(repetitions)`, `debut = horloge()`, l'appel,
    `durees.append(horloge() - debut)`, et `return min(durees)`. Vérifiez `repetitions` AVANT la boucle.
    """
    raise NotImplementedError


def point_chaud(programme: Callable[[], object]) -> str:
    """Défi 111 🟢 — Trouver où le temps passe : profiler avec `cProfile` et `pstats`.

    Personne ne devine juste. Le collègue était sûr que c'était la lecture du fichier ; c'était une
    fonction de mise en forme appelée une seule fois. Avant d'optimiser, on PROFILE.

    Lancez `programme()` une fois sous le profileur `cProfile`, et renvoyez le NOM (un texte, par
    exemple « calculer_scores ») de la fonction qui a passé le plus de temps dans SON PROPRE code : la
    colonne `tottime`.

        >>> point_chaud(programme_exemple)
        'calculer_scores'

    Pour lire les résultats : `pstats.Stats(profileur).stats` est un dictionnaire dont chaque clé est un
    triplet (fichier, ligne, nom de la fonction) et chaque valeur un tuple dont l'élément 2 est le
    `tottime` et l'élément 3 le `cumtime`.

    Les pièges, que le correcteur essaie :

    - `cumtime` (temps cumulé) désigne la fonction qui appelle les autres : `programme` lui-même, ou la
      fonction qui ne fait que déléguer. Ce n'est pas elle qu'il faut optimiser ;
    - la fonction appelée le plus souvent (`ncalls`) n'est pas forcément la plus lente : 3 000 appels à
      une fonction qui ne fait rien ne coûtent presque rien ;
    - si `programme()` lève une exception, elle doit remonter, ET le profileur doit être arrêté. Un
      profileur oublié ralentit toute la suite du programme (et depuis Python 3.12, empêche d'en lancer
      un autre : « Another profiling tool is already active »). `profileur.runcall(programme)` s'en
      charge pour vous ; sinon, `try / finally`.

    Ce que vous apprenez : à lire un profil, et la différence entre « ça prend du temps ici » (tottime)
    et « ça prend du temps en dessous » (cumtime).

    Indice : `profileur = cProfile.Profile()`, `profileur.runcall(programme)`, puis
    `stats = pstats.Stats(profileur).stats` et `max(stats, key=lambda cle: stats[cle][2])` donne la clé
    (fichier, ligne, nom) de la plus coûteuse ; renvoyez son élément 2.
    """
    raise NotImplementedError


# ============================================================================ Changer d'algorithme


def prix_en_vigueur(changements: list[tuple[date, float]], dates: list[date]) -> list[float]:
    """Défi 112 🟡 — Le bon tarif sans tout parcourir : une liste triée et `bisect`.

    Pour recalculer 200 000 factures, il faut le prix de l'offre au jour de chaque facture. L'historique
    contient 50 000 changements de tarif. Parcourir l'historique pour chaque facture, c'est 10 milliards
    de comparaisons. Dans une liste TRIÉE, la recherche dichotomique (`bisect`) trouve la bonne place en
    16 comparaisons, comme on cherche un mot dans un dictionnaire papier.

    `changements` est une liste de (date d'entrée en vigueur, prix), dans N'IMPORTE QUEL ordre : chaque
    prix s'applique à partir de sa date, incluse, jusqu'au changement suivant. Renvoyez la liste des prix
    en vigueur pour chaque date de `dates`, dans l'ordre de `dates`.

        >>> TARIFS_EXEMPLE                         # 1er janvier 2024 : 29.9, 1er juillet : 32.9, puis 34.9
        [(datetime.date(2025, 1, 1), 34.9), (datetime.date(2024, 1, 1), 29.9), (datetime.date(2024, 7, 1), 32.9)]
        >>> prix_en_vigueur(TARIFS_EXEMPLE, [date(2024, 3, 15), date(2024, 7, 1), date(2026, 1, 1)])
        [29.9, 32.9, 34.9]

    Les règles :

    - une date antérieure au premier changement n'a pas de prix : ValueError, avec la date dans le
      message ;
    - deux changements le même jour, c'est une erreur dans l'historique : ValueError (on ne choisit pas
      au hasard) ; un historique vide aussi ;
    - `changements` ne doit pas être modifié : l'appelant s'en sert encore (`sorted(...)` renvoie une
      copie triée, `liste.sort()` trie la liste de l'appelant) ;
    - triez UNE fois, puis une recherche dichotomique par date : le correcteur compte vos comparaisons
      sur 2 000 changements et 20 000 dates.

    Les pièges, que le correcteur essaie tous :

    - le jour même d'un changement, c'est le NOUVEAU prix : `bisect_left` et `bisect_right` ne donnent pas
      la même position quand la date est dans la liste ;
    - avant le premier changement, la position trouvée est 0, et `prix[0 - 1]` ne plante pas : en Python,
      l'indice -1 désigne le DERNIER élément. Le client paierait le tarif de 2025 en 2023, sans un message.

    Ce que vous apprenez : `bisect`, et le réflexe « trier une fois, chercher souvent ».

    Indice : `historique = sorted(changements)`, puis `jours = [d for d, _ in historique]` et
    `prix = [p for _, p in historique]`. Pour chaque date : `i = bisect.bisect_right(jours, d)` est le
    nombre de changements déjà en vigueur ce jour-là ; s'il vaut 0, ValueError ; sinon `prix[i - 1]`.
    Deux changements le même jour : deux éléments voisins égaux dans `jours`.
    """
    raise NotImplementedError


# ============================================================================ Ne pas refaire le travail


def cache_lru(taille_max: int, duree: float, horloge: Callable[[], float] = time.monotonic) -> Callable:
    """Défi 113 🟡 — Un cache qui ne grossit pas à l'infini et ne sert pas de vieilles réponses.

    Le cache du défi 49 garde TOUT, POUR TOUJOURS. Sur un serveur qui tourne des semaines, il finit par
    occuper toute la mémoire, et il sert le score d'un client calculé il y a un mois. Un vrai cache est
    BORNÉ (au plus `taille_max` résultats ; quand il est plein, il oublie celui qu'on a utilisé il y a le
    plus longtemps : « LRU », least recently used) et PÉRIME ses résultats (au bout de `duree` secondes,
    on recalcule).

    `cache_lru(taille_max, duree)` renvoie un décorateur :

        >>> horloge = HorlogeFactice()
        >>> @cache_lru(taille_max=2, duree=60, horloge=horloge)
        ... def score(client_id):
        ...     print("calcul", client_id)
        ...     return 0.5
        >>> [score(c) for c in ("C1", "C1", "C2")]          # le 2e C1 est réutilisé
        calcul C1
        calcul C2
        [0.5, 0.5, 0.5]
        >>> [score(c) for c in ("C1", "C3")]                # C1 vient de servir : c'est C2 qu'on oublie
        calcul C3
        [0.5, 0.5]
        >>> score.cache_info()
        {'calculs': 3, 'reutilisations': 2, 'taille': 2}
        >>> horloge.avancer(60)
        >>> score("C1")                                     # 60 secondes après son calcul : périmé
        calcul C1
        0.5

    Les règles :

    - les appels n'utilisent que des arguments positionnels (`score("C1")`), qui servent de clé ;
    - un résultat est réutilisable pendant `duree` secondes APRÈS SON CALCUL : à `calcul + duree`
      exactement, il est périmé. Le relire ne le rajeunit pas (sinon un score consulté chaque minute ne
      serait jamais recalculé) ;
    - utiliser un résultat en fait le plus récent ; quand il faut ranger un NOUVEAU résultat et que le
      cache est plein, on oublie le moins récemment utilisé. Le cache ne dépasse jamais `taille_max` ;
    - un résultat qui vaut None ou 0.0 se met en cache comme les autres ;
    - une exception n'est pas mise en cache : elle remonte, et l'appel suivant recalcule ;
    - `fonction.cache_info()` renvoie {"calculs": …, "reutilisations": …, "taille": …} ;
    - chaque fonction décorée a SON cache, même avec le même décorateur :
      `memo = cache_lru(10, 60)`, puis `@memo` sur deux fonctions ;
    - la fonction décorée garde son nom et sa documentation (`functools.wraps`) ;
    - `taille_max` doit être un entier d'au moins 1 et `duree` un nombre strictement positif, sinon
      ValueError, dès l'appel `cache_lru(...)` ;
    - l'heure se lit avec `horloge()` : le correcteur vous passe une `HorlogeFactice`.

    Ce que vous apprenez : `OrderedDict` (`move_to_end`, `popitem(last=False)`), un décorateur qui prend
    des paramètres (trois niveaux de fonctions), et à quel endroit créer l'état pour qu'il ne soit pas
    partagé.

    Indice :
        def cache_lru(taille_max, duree, horloge=time.monotonic):
            if ...: raise ValueError(...)
            def decorateur(fonction):
                memoire = OrderedDict()          # clé -> (instant du calcul, résultat) : UN par fonction
                @functools.wraps(fonction)
                def enveloppe(*args):
                    ...
                enveloppe.cache_info = ...
                return enveloppe
            return decorateur
    """
    raise NotImplementedError


# ============================================================================ La mémoire


def totaux_par_contrat(chemin: str | Path) -> dict[str, tuple[int, float]]:
    """Défi 114 🟡 — Un fichier de 10 Go avec 1 Mo de mémoire : lire en flux.

    L'export de nuit des factures pèse 10 Go ; le serveur a 4 Go de mémoire. `f.read()`, `f.readlines()`
    ou `pd.read_csv(chemin)` chargent TOUT avant de commencer : le programme est tué au milieu de la nuit.
    Un fichier se lit ligne par ligne : la mémoire utilisée ne dépend plus de la taille du fichier.

    Le fichier est un CSV en UTF-8, séparé par des virgules, avec une ligne d'en-tête. Il contient (au
    moins) les colonnes `contrat` et `montant`, dans un ordre qui change d'un export à l'autre. Renvoyez,
    pour chaque contrat, le nombre de factures et la somme de leurs montants arrondie au centime :

        >>> chemin = ecrire_factures("factures.csv")          # 100 000 lignes, 2,8 Mo
        >>> totaux_par_contrat(chemin)
        {'Biannuel': (19512, 1413713.44), 'Mensuel': (54010, 3914899.63), 'Annuel': (24509, 1772371.59)}

    Les règles :

    - un montant vide est une facture annulée : la ligne ne compte pas (ni dans le nombre, ni dans la
      somme) ; une ligne entièrement vide (la dernière, souvent) non plus ;
    - lisez les colonnes par leur NOM, pas par leur position : la colonne `montant` n'est pas toujours la
      dernière ;
    - un champ entre guillemets peut contenir une virgule (« "Fibre, TV" ») : `ligne.split(",")` décale
      alors toutes les colonnes. Le module `csv` connaît ces règles ; ne les réécrivez pas ;
    - la mémoire ne doit pas grandir avec le fichier : le correcteur mesure le pic de mémoire
      (`tracemalloc`) sur un export, puis sur un export deux fois plus long, et compare. pandas est
      permis, par paquets de lignes : `pd.read_csv(chemin, chunksize=10_000)` ;
    - le fichier doit être refermé : `with open(...)`.

    Ce que vous apprenez : à lire un fichier en flux (un objet fichier s'itère ligne par ligne),
    `csv.DictReader`, et à mesurer une mémoire au lieu de la deviner.

    Indice : `with open(chemin, newline="", encoding="utf-8") as fichier:` puis
    `for ligne in csv.DictReader(fichier):` ; `ligne["montant"]` est un texte. Gardez un dictionnaire
    contrat → [nombre, somme], et arrondissez avec `round(somme, 2)` à la fin seulement.
    """
    raise NotImplementedError


# ============================================================================ Optimiser sans rien changer


def clients_a_rappeler(clients: list[dict], appels: list[dict], deja_contactes: list[str],
                       seuil_appels: int = 3) -> list[tuple[str, int]]:
    """Défi 115 🔴 — Accélérer une fonction sans changer son résultat d'un octet.

    `clients_a_rappeler_lent` (lisez-la : `help(clients_a_rappeler_lent)`, ou dans `defis/performance.py`)
    est juste, testée, utilisée par le marketing… et met 40 minutes sur 400 000 clients. Écrivez
    `clients_a_rappeler`, qui renvoie EXACTEMENT la même chose (mêmes paires, même ordre) pour TOUTES les
    entrées, mais en un seul passage sur chaque liste.

        >>> clients, appels, deja = exemple_rappels()         # 2 000 clients, 20 000 appels
        >>> clients_a_rappeler(clients, appels, deja) == clients_a_rappeler_lent(clients, appels, deja)
        True
        >>> chronometrer(lambda: clients_a_rappeler(clients, appels, deja))   # contre 2 secondes
        0.002…

    Profilez la version lente (défi 111) : trois choses y coûtent cher.

    1. Pour CHAQUE client, elle relit TOUS les appels : comptez-les une fois pour toutes (un
       dictionnaire, ou `collections.Counter`) ;
    2. pour chaque client, elle recalcule la facture moyenne, qui est la même pour tous : sortez-la de
       la boucle ;
    3. pour chaque client, elle recopie `deja_contactes` en liste et la parcourt : un `set`, construit
       une fois.

    Le correcteur compare votre fonction à la version lente sur des centaines de cas tirés au hasard et
    sur les cas limites, puis compte vos lectures sur 2 000 clients : au-delà de 10 lectures par élément
    des listes, il arrête votre fonction.

    Les pièges : tout ce que la version lente fait sans le dire, et que la rapide doit refaire.

    - SANS client, la boucle ne tourne pas, donc la moyenne n'est jamais calculée : la version lente
      renvoie [] ; la moyenne sortie de la boucle, elle, divise par zéro ;
    - la moyenne porte sur TOUS les clients, y compris ceux déjà contactés ;
    - les appels pour « information » ne comptent pas, ni ceux de clients absents de `clients` ;
    - à égalité d'appels, l'ordre des clients est conservé (`sort` est stable, y compris avec
      `reverse=True`) : ne triez pas en plus par identifiant ;
    - le calcul exact de la moyenne compte aussi : `sum(...) / len(...)` et `statistics.mean(...)` ne
      donnent pas toujours le même dernier chiffre, et un client pile à la moyenne passerait d'un côté à
      l'autre. Optimiser, c'est changer la VITESSE, pas les calculs ;
    - ne modifiez pas les listes reçues.

    Ce que vous apprenez : la méthode complète (mesurer, profiler, changer d'algorithme, prouver que rien
    n'a changé en comparant à l'ancienne version sur des entrées au hasard), et pourquoi on garde
    l'ancienne version dans les tests.
    """
    raise NotImplementedError

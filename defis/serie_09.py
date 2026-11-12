"""Série 09 — Penser en algorithmes : les classiques de l'entretien technique.

🧠  APRÈS LES SÉRIES 06 À 08 : « ÇA MARCHE », CE N'EST PAS ENCORE « ÇA PASSE À L'ÉCHELLE ».

Les entretiens de développeur·se et de data scientist contiennent presque toujours un exercice
d'algorithmique : quelques lignes de code, mais une question précise derrière, « combien de temps
avec un million de lignes ? ». Ce sont les mêmes problèmes qu'au quotidien : retrouver une paire,
vérifier une formule, trouver un pic, fusionner des périodes, garder les meilleurs scores d'un flux
trop gros pour la mémoire, parcourir un réseau. Chaque défi a une solution « force brute » facile
et juste… que le correcteur refuse quand elle devient trop lente ou trop gourmande.

Prérequis : les séries 00 et 07 (dictionnaires, `set`, générateurs). Aucune notion de maths.

🗣️  LA MÉTHODE EN 5 ÉTAPES (à dire à voix haute en entretien, à appliquer seul·e ici) :

1. Reformuler : entrées, sortie, cas limites (liste vide ? doublons ? égalités ?).
2. Faire 2 ou 3 exemples à la main, dont un piège.
3. Écrire d'abord la force brute, et donner sa complexité.
4. Chercher la structure de données qui évite le travail répété (dict, pile, tas, file…).
5. Tester les cas limites, puis annoncer la complexité finale.

⏱️  LES COMPLEXITÉS À CONNAÎTRE (n = taille des données) :

    O(1)        x in un_set, un_dict[cle], liste.append, deque.popleft
    O(log n)    heapq.heappush / heappop, bisect (recherche dans une liste TRIÉE)
    O(n)        une boucle, x in une_liste, liste.pop(0), sum, max
    O(n log n)  sorted(...), liste.sort()
    O(n²)       une boucle dans une boucle : 1 000 lignes → 1 million d'opérations,
                1 million de lignes → 1 000 milliards (des heures)

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py --defi 56` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, faites l'exemple sur papier, puis seulement en dernier recours
   ouvrez `defis/corrige_serie_09.py`.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import heapq  # défi 60
from collections import deque  # défi 61
from collections.abc import Iterable
from datetime import date, timedelta  # défi 59

# ============================================================================ Table de hachage


def deux_offres(couts: list[int], budget: int) -> tuple[int, int] | None:
    """Défi 56 🟢 — Trouver deux gestes commerciaux dont la somme fait exactement le budget.

    Pour retenir un client, l'équipe peut lui offrir DEUX gestes commerciaux différents du catalogue
    (`couts` : le coût de chaque geste, en euros entiers ; deux gestes peuvent coûter pareil). Le
    budget accordé doit être dépensé exactement. Renvoyez les positions `(i, j)`, avec i < j, de deux
    gestes dont les coûts font `budget` au total, ou `None` s'il n'y en a pas :

        deux_offres([40, 10, 25, 15, 30], 55)  → (0, 3) ou (2, 4)   (40 + 15, ou 25 + 30)
        deux_offres([20, 35, 20], 40)          → (0, 2)             (deux gestes différents à 20 €)
        deux_offres([20, 35], 40)              → None               (on n'offre pas DEUX FOIS le même)

    - s'il y a plusieurs réponses, renvoyez-en une, n'importe laquelle ;
    - `None` quand c'est impossible (pas -1, pas False, pas une liste vide) ;
    - ne modifiez pas `couts` ;
    - le catalogue complet compte des milliers de gestes : pas de boucle dans une boucle.

    Ce que vous apprenez : LE classique des entretiens (« two sum »). La force brute teste toutes les
    paires, O(n²). Avec un dictionnaire « coût déjà vu → position », chaque geste demande juste
    « ai-je déjà vu le coût qui me complète ? », en O(1) : une seule boucle, O(n).

    Indice : parcourez `enumerate(couts)`. Pour le geste j de coût c, le complément est
    `budget - c` : s'il est dans le dictionnaire `vus`, vous avez votre paire. Sinon, enregistrez
    `vus[c] = j`… APRÈS la recherche (sinon un geste se complète lui-même). Attention : la position
    0 est « fausse » en Python, testez `complement in vus`, pas `if vus.get(complement)`.
    """
    raise NotImplementedError


# ============================================================================ Pile


def parenthesage_valide(formule: str) -> bool:
    """Défi 57 🟢 — Vérifier les parenthèses d'une formule avant de la calculer.

    Les variables dérivées du modèle sont décrites par des formules dans un fichier de configuration :

        "log(1 + [total_charges] / {tenure_months})"     → True
        "(support_calls * 2"                              → False  (une parenthèse jamais fermée)
        "[monthly_charges)"                               → False  (fermée par le mauvais signe)
        "(a + [b) - c]"                                   → False  (les paires se croisent)
        ")a + b("                                         → False  (fermée avant d'être ouverte)

    Renvoyez True si les trois sortes de délimiteurs, ( ), [ ] et { }, sont bien appariées :
    - chaque fermant ferme le DERNIER ouvrant encore ouvert, qui doit être du même type ;
    - à la fin, tout ce qui a été ouvert est fermé ;
    - les autres caractères sont ignorés ; une formule sans délimiteur (ou vide) est valide ;
    - renvoyez un vrai booléen (True / False) ;
    - les formules générées automatiquement peuvent imbriquer des milliers de niveaux.

    Ce que vous apprenez : la PILE (dernier entré, premier sorti), avec une simple liste Python
    (`append` pour empiler, `pop` pour dépiler, `pile[-1]` pour regarder le sommet). Compter les
    ouvrantes et les fermantes ne suffit pas : « )( » et « ([)] » ont les bons comptes.

    Indice : `PAIRES = {")": "(", "]": "[", "}": "{"}`. Pour chaque caractère : un ouvrant, on
    l'empile ; un fermant, on vérifie que la pile n'est pas vide ET que son sommet est l'ouvrant
    attendu, puis on dépile. À la fin : `return not pile`.
    """
    raise NotImplementedError


# ============================================================================ Fenêtre glissante


def pic_appels(jours: list[int], largeur: int = 7) -> int:
    """Défi 58 🟡 — Le pic d'appels au support sur une fenêtre de quelques jours.

    `jours` contient le jour de chaque appel reçu par le support (un entier : le numéro du jour,
    plusieurs appels possibles le même jour, dans le désordre). Renvoyez le nombre MAXIMAL d'appels
    reçus sur `largeur` jours consécutifs, c'est-à-dire du jour d au jour d + largeur - 1 inclus :

        pic_appels([1, 2, 2, 9, 3, 15, 8, 10], largeur=7)  → 4
            jours 1 à 7 : 1, 2, 2, 3 (4 appels)       jours 8 à 14 : 8, 9, 10 (3 appels)
            jours 4 à 10 : 8, 9, 10 (3 appels)        jours 9 à 15 : 9, 10, 15 (3 appels)

    - 0 si `jours` est vide ; `largeur` vaut au moins 1 (largeur=1 : le jour le plus chargé) ;
    - ne modifiez pas `jours` (ni tri sur place, ni suppression) ;
    - l'historique compte des milliers d'appels : pas de recomptage complet pour chaque jour.

    Ce que vous apprenez : la technique des DEUX POINTEURS, ou fenêtre glissante. Sur la liste
    TRIÉE, la fenêtre [gauche, droite] avance : `droite` ajoute un appel à chaque tour, `gauche`
    retire les appels trop anciens. Chaque appel entre une fois et sort une fois : O(n) après le
    tri, au lieu de O(n²) en recomptant la fenêtre pour chaque appel.

    Indice : `t = sorted(jours)` (une copie triée), `gauche = 0`, puis
    `for droite in range(len(t)):` et, tant que `t[droite] - t[gauche] >= largeur`, faites avancer
    `gauche`. La fenêtre contient `droite - gauche + 1` appels : gardez le maximum.
    """
    raise NotImplementedError


# ============================================================================ Intervalles


def fusionner_periodes(periodes: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Défi 59 🟡 — Reconstituer les vraies périodes d'abonnement d'un client.

    Le CRM garde une ligne par CONTRAT, `(debut, fin)`, deux dates INCLUSES (debut <= fin). Elles
    arrivent dans le désordre, se chevauchent parfois (une saisie en double, un avenant) et se
    suivent souvent (un renouvellement : un contrat finit le 31 janvier, le suivant commence le
    1er février). Renvoyez les périodes où le client a été abonné SANS INTERRUPTION :

        [(1er mars, 30 avril), (1er janv., 31 janv.), (1er févr., 31 mars), (1er juin, 30 juin)]
          → [(1er janv., 30 avril), (1er juin, 30 juin)]

    - deux périodes qui se chevauchent, se touchent ou se SUIVENT (lendemain) sont fusionnées ;
    - une période peut être entièrement contenue dans une autre ;
    - résultat : une liste de tuples `(debut, fin)` triée par date de début (liste vide si aucune
      période) ;
    - ne modifiez pas `periodes`.

    À quoi ça sert : l'ancienneté RÉELLE d'un client (sans les trous), et repérer les clients
    partis puis revenus (deux périodes séparées), un profil très particulier pour le churn.

    Ce que vous apprenez : « trier d'abord » transforme un problème compliqué (comparer chaque
    période à toutes les autres, O(n²)) en un seul parcours, O(n log n). Et `timedelta` pour
    calculer « le lendemain ».

    Indice : parcourez `sorted(periodes)` en gardant la période en cours `(debut, fin)`. Si la
    suivante commence au plus tard le lendemain de `fin` (`<= fin + timedelta(days=1)`), elle
    prolonge la période : `fin = max(fin, sa_fin)` (le max, à cause des périodes contenues). Sinon,
    la période en cours est terminée : ajoutez-la au résultat et repartez de la nouvelle.
    """
    raise NotImplementedError


# ============================================================================ Tas


def top_k_flux(flux: Iterable[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """Défi 60 🟡 — Les k clients les plus à risque, dans un flux trop gros pour la mémoire.

    Chaque nuit, le modèle note des millions de clients. Les scores arrivent de la base en FLUX :
    `flux` est un itérable (souvent un générateur, qu'on ne peut parcourir qu'UNE fois et qui n'a
    pas de `len`) de tuples `(customer_id, proba)`. L'équipe rétention ne peut appeler que `k`
    clients. Renvoyez les `k` tuples de plus forte probabilité, sous forme de liste triée par
    probabilité DÉCROISSANTE :

        top_k_flux(iter([("C1", 0.2), ("C2", 0.9), ("C3", 0.5), ("C4", 0.9)]), k=3)
          → [("C2", 0.9), ("C4", 0.9), ("C3", 0.5)]

    - à probabilité égale, le client arrivé le PREMIER dans le flux passe devant (C2 avant C4) ;
    - moins de k clients dans le flux : renvoyez-les tous (triés) ; k = 0 : liste vide ;
    - un seul passage sur le flux, et en mémoire au plus ~k clients à la fois : ni `list(flux)`,
      ni `sorted(flux)`, qui chargeraient les millions de lignes d'un coup.

    Ce que vous apprenez : le TAS (module `heapq`), la structure qui garde en permanence son plus
    PETIT élément au sommet (`tas[0]`). Un tas de taille k contient les k meilleurs vus jusqu'ici ;
    son sommet est le « moins bon des meilleurs », le seul qu'un nouveau venu peut détrôner.
    Mémoire O(k) au lieu de O(n), temps O(n log k).

    Indice : `heapq.nlargest(k, flux, key=...)` fait exactement cela, et il est stable (à égalité,
    il garde l'ordre d'arrivée). Pour comprendre ce qu'il fait, écrivez-le aussi à la main : un tas
    de tuples `(proba, -position, customer_id)`, `heapq.heappush` tant qu'il a moins de k éléments,
    puis `heapq.heappushpop` (ou `heapreplace`) quand un nouveau client bat le sommet `tas[0]`.
    Le `-position` règle les égalités : un nouveau venu à égalité ne détrône pas l'ancien.
    """
    raise NotImplementedError


# ============================================================================ Graphe


def clients_exposes(
    parrainages: list[tuple[str, str]],
    partis: Iterable[str],
    distance_max: int = 2,
) -> dict[str, int]:
    """Défi 61 🔴 — Repérer les clients proches d'un client parti, dans le réseau de parrainage.

    Le churn est contagieux : quand un client part, ses proches suivent plus souvent. Le programme
    de parrainage forme un réseau : `parrainages` est une liste de liens `(parrain, filleul)`, et le
    lien vaut dans LES DEUX SENS (on s'influence mutuellement). `partis` contient les clients partis
    ce mois-ci (une liste ou un set). Renvoyez le dictionnaire `{client: distance}` de tous les
    clients NON partis situés à au plus `distance_max` liens du client parti le PLUS PROCHE :

        A — B — C — D        E — F           partis = ["A", "F"], distance_max = 2
                                              → {"B": 1, "C": 2, "E": 1}

    - distance 1 = lien direct avec un client parti, 2 = un intermédiaire, etc. ; c'est le plus
      COURT chemin vers N'IMPORTE QUEL client parti qui compte ;
    - un client parti n'apparaît jamais dans le résultat, même s'il est voisin d'un autre parti ;
    - un client hors de portée (ici D, à distance 3) n'apparaît pas ;
    - le réseau a des cycles (A parraine B, B parraine C, C a parrainé A…) et des milliers de liens ;
    - ne modifiez pas les arguments.

    Ce que vous apprenez : le parcours en LARGEUR (BFS), l'algorithme des plus courts chemins quand
    chaque lien vaut 1. On part de TOUS les clients partis à la fois (BFS « multi-sources ») : la
    file traite d'abord tous les clients à distance 1, puis 2… donc la première fois qu'on atteint
    un client, c'est par le plus court chemin. Avec une liste d'adjacence construite une fois, O(V + E).
    Un parcours en PROFONDEUR (récursif) donne des distances fausses ici, et plante sur une longue chaîne.

    Indice : (1) construisez `voisins = {}` : pour chaque lien (a, b), ajoutez b aux voisins de a
    ET a aux voisins de b (`voisins.setdefault(a, []).append(b)`). (2) `distance = {p: 0 for p in
    partis}` et `file = deque(partis)`. (3) Tant que la file n'est pas vide : `client =
    file.popleft()` ; si sa distance vaut déjà `distance_max`, on ne va pas plus loin ; sinon, chaque
    voisin PAS ENCORE dans `distance` reçoit `distance[client] + 1` et entre dans la file.
    (4) Renvoyez `distance` sans les clients partis (distance 0).
    """
    raise NotImplementedError

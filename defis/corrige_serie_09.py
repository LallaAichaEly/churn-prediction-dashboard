"""Série 09 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_09.py`.

Si votre version passe les vérifications, elle est juste, même si elle ne ressemble pas à
celle-ci. Comparez surtout la COMPLEXITÉ : en entretien, c'est la question qui suit toujours
« votre code marche ? ». Notez dans JOURNAL.md la structure de données qui a tout débloqué.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterable
from datetime import date, timedelta

# ============================================================================ Table de hachage


def deux_offres(couts: list[int], budget: int) -> tuple[int, int] | None:
    # Force brute : for i… for j > i… → n²/2 paires. Ici, UNE boucle et un dictionnaire.
    vus: dict[int, int] = {}  # coût déjà rencontré → sa première position
    for j, cout in enumerate(couts):
        complement = budget - cout
        # `in` et pas `vus.get(complement)` : la position 0 est « fausse » en Python.
        if complement in vus:
            return vus[complement], j
        # On enregistre APRÈS avoir cherché : sinon, avec budget = 2 × cout, le geste j
        # se trouverait lui-même. setdefault garde la PREMIÈRE position d'un coût répété.
        vus.setdefault(cout, j)
    return None  # explicite : « pas de réponse » se lit mieux que la fin de fonction


# Complexité : O(n) en temps (une recherche dans un dict coûte O(1)), O(n) en mémoire.
# Variante sans mémoire supplémentaire : trier les positions par coût puis deux pointeurs
# (un au début, un à la fin) : O(n log n).


# ============================================================================ Pile

OUVRANTS = "([{"
PAIRES = {")": "(", "]": "[", "}": "{"}  # fermant → ouvrant attendu


def parenthesage_valide(formule: str) -> bool:
    pile: list[str] = []
    for caractere in formule:
        if caractere in OUVRANTS:
            pile.append(caractere)
        elif caractere in PAIRES:
            # Deux erreurs possibles : rien à fermer (« )a( »), ou le mauvais type (« [a) »).
            if not pile or pile.pop() != PAIRES[caractere]:
                return False
    # Ouvert mais jamais fermé (« (a + b ») : la pile n'est pas vide.
    return not pile


# Complexité : O(n). Une version récursive « naturelle » plante au-delà de ~1 000 niveaux
# d'imbrication (RecursionError) : la pile explicite n'a pas cette limite.
# L'astuce « while "()" in s: s = s.replace("()", "") » est juste, mais O(n²).


# ============================================================================ Fenêtre glissante


def pic_appels(jours: list[int], largeur: int = 7) -> int:
    t = sorted(jours)  # une COPIE triée : jours.sort() modifierait la liste de l'appelant
    meilleur = 0
    gauche = 0
    for droite, jour in enumerate(t):
        # La fenêtre couvre [jour - largeur + 1, jour] : on retire les appels trop anciens.
        while jour - t[gauche] >= largeur:
            gauche += 1
        meilleur = max(meilleur, droite - gauche + 1)
    return meilleur


# Complexité : O(n log n) pour le tri, puis O(n) : `gauche` et `droite` ne reculent jamais,
# chaque appel entre une fois dans la fenêtre et en sort une fois au plus.
# Piège du « + 1 » : 7 jours consécutifs à partir du jour d, c'est d à d + 6 ; d'où le
# `>= largeur` (un écart de 7 jours, c'est déjà 8 jours de fenêtre).


# ============================================================================ Intervalles

UN_JOUR = timedelta(days=1)


def fusionner_periodes(periodes: list[tuple[date, date]]) -> list[tuple[date, date]]:
    resultat: list[tuple[date, date]] = []
    # sorted(...) trie les tuples par début, puis par fin ; et ne touche pas à `periodes`.
    for debut, fin in sorted(periodes):
        if resultat and debut <= resultat[-1][1] + UN_JOUR:
            # Chevauchement, contact ou lendemain : on prolonge la période en cours.
            # max(), car une période peut être ENTIÈREMENT contenue dans la précédente.
            debut_courant, fin_courante = resultat[-1]
            resultat[-1] = (debut_courant, max(fin_courante, fin))
        else:
            resultat.append((debut, fin))
    return resultat


# Complexité : O(n log n), à cause du tri. Sans le tri, il faudrait comparer chaque période
# à toutes les autres, et recommencer tant que des fusions ont lieu.
# Pour aller plus loin : l'ancienneté réelle = sum((fin - debut).days + 1 for debut, fin in resultat).


# ============================================================================ Tas


def top_k_flux(flux: Iterable[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    # En une ligne, avec la bibliothèque standard (stable : à égalité, l'ordre d'arrivée) :
    #     return heapq.nlargest(k, flux, key=lambda client: client[1])
    # Ci-dessous, ce que fait nlargest, pour comprendre le tas.
    if k <= 0:
        return []
    tas: list[tuple[float, int, str]] = []  # (proba, -position, customer_id) : un tas MIN
    for position, (client, proba) in enumerate(flux):
        element = (proba, -position, client)
        if len(tas) < k:
            heapq.heappush(tas, element)
        elif element > tas[0]:
            # tas[0] est le « moins bon des meilleurs » : le nouveau venu le remplace.
            # À proba égale, -position est plus petit pour le nouveau : il ne passe pas.
            heapq.heapreplace(tas, element)
    # Du meilleur au moins bon : proba décroissante, puis arrivée la plus ancienne d'abord.
    return [(client, proba) for proba, _, client in sorted(tas, reverse=True)]


# Complexité : O(n log k) en temps, O(k) en mémoire. sorted(flux)[:k] : O(n log n) en temps et
# O(n) en mémoire, impossible avec 50 millions de clients.
# Pourquoi -position plutôt que customer_id pour départager ? Comparer les identifiants
# (« C10 » < « C9 » en texte !) donnerait un ordre arbitraire, pas l'ordre d'arrivée.


# ============================================================================ Graphe


def clients_exposes(
    parrainages: list[tuple[str, str]],
    partis: Iterable[str],
    distance_max: int = 2,
) -> dict[str, int]:
    # 1. La liste d'adjacence, construite UNE fois : O(E). Chercher les voisins en reparcourant
    #    `parrainages` à chaque client coûterait O(V × E).
    voisins: dict[str, list[str]] = {}
    for parrain, filleul in parrainages:
        voisins.setdefault(parrain, []).append(filleul)
        voisins.setdefault(filleul, []).append(parrain)  # le lien vaut dans les deux sens

    # 2. BFS multi-sources : tous les clients partis démarrent à distance 0, ensemble.
    distance: dict[str, int] = {}
    file: deque[str] = deque()
    for client in partis:
        if client not in distance:
            distance[client] = 0
            file.append(client)

    # 3. La file sert les clients par distance croissante : la première fois qu'on atteint un
    #    client, c'est par le plus court chemin. `distance` sert aussi d'ensemble « déjà vus »
    #    (indispensable : le réseau a des cycles).
    while file:
        client = file.popleft()  # O(1) ; liste.pop(0) décalerait toute la liste : O(n)
        if distance[client] == distance_max:
            continue  # inutile d'aller plus loin
        for voisin in voisins.get(client, ()):
            if voisin not in distance:
                distance[voisin] = distance[client] + 1
                file.append(voisin)

    # 4. Les clients partis (distance 0) ne sont pas « exposés » : ils sont déjà partis.
    return {client: d for client, d in distance.items() if d > 0}


# Complexité : O(V + E). Pourquoi pas un parcours en profondeur (DFS) ? Il atteint un client
# d'abord par un chemin quelconque, pas forcément le plus court, et une version récursive
# dépasse la limite de récursion de Python (~1 000) sur une longue chaîne de parrainages.

"""Outils de la série 24, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Le traitement de nuit du projet, décrit comme un GRAPHE DE DÉPENDANCES : un dictionnaire qui associe à
chaque tâche la liste des tâches dont elle a besoin (ses PRÉREQUIS). Chaque appel renvoie un NOUVEAU
dictionnaire : modifiez-le autant que vous voulez.

    >>> from defis.serie_24 import *
    >>> pipeline = pipeline_exemple()
    >>> pipeline["variables"]                      # « variables » a besoin de ces deux tâches
    ['nettoyer', 'extraire_appels']
    >>> "extraire_appels" in pipeline              # citée seulement comme prérequis : une tâche quand même
    False
    >>> sorted(taches_du_graphe(pipeline))[:4]
    ['alerte_support', 'entrainer', 'evaluer', 'extraire_appels']

Le même graphe, une ligne par tâche (à gauche ses prérequis, à droite la tâche qui en a besoin) :

    extraire_clients, extraire_factures  ─▶  nettoyer
    nettoyer, extraire_appels            ─▶  variables
    extraire_appels                      ─▶  alerte_support
    variables                            ─▶  entrainer
    entrainer, variables                 ─▶  evaluer
    entrainer, variables                 ─▶  scorer
    scorer                               ─▶  sms
    evaluer, scorer                      ─▶  rapport
    rapport, scorer                      ─▶  tableau_de_bord

`generer_pipeline` fabrique de grands graphes SANS cycle (des milliers de tâches) pour mesurer la vitesse
de votre code, et `afficher_planning` dessine le résultat du défi 151.
"""

from __future__ import annotations

import random


def pipeline_exemple() -> dict[str, list[str]]:
    """Le traitement de nuit : 11 clés, 12 tâches (`extraire_appels` n'est citée que comme prérequis).

    Les pièges qu'il contient :

    - `extraire_appels` n'est pas une clé : une tâche citée seulement comme prérequis existe quand même ;
    - `alerte_support` devient prête dès qu'`extraire_appels` est fait, AVANT `extraire_clients` dans
      l'ordre alphabétique : elle passe donc avant (défi 146), pas « à la vague suivante » ;
    - `evaluer` et `scorer` ont besoin d'`entrainer` ET de `variables` (déjà nécessaire à `entrainer`) :
      deux chemins mènent à elles, il ne faut les compter (et les lancer) qu'une fois.

        >>> pipeline_exemple()["tableau_de_bord"]
        ['rapport', 'scorer']
    """
    return {
        "extraire_clients": [],
        "extraire_factures": [],
        "nettoyer": ["extraire_clients", "extraire_factures"],
        "variables": ["nettoyer", "extraire_appels"],
        "alerte_support": ["extraire_appels"],
        "entrainer": ["variables"],
        "evaluer": ["entrainer", "variables"],
        "scorer": ["entrainer", "variables"],
        "sms": ["scorer"],
        "rapport": ["evaluer", "scorer"],
        "tableau_de_bord": ["rapport", "scorer"],
    }


def durees_exemple() -> dict[str, int]:
    """La durée de chaque tâche du traitement de nuit, en minutes (défis 150 et 151).

    `evaluer` et `scorer` durent 15 minutes toutes les deux, et commencent en même temps : elles finissent
    ENSEMBLE, à égalité. La somme des durées (tout faire l'un après l'autre) vaut 200 minutes.

        >>> sum(durees_exemple().values())
        200
    """
    return {
        "extraire_clients": 10,
        "extraire_factures": 25,
        "extraire_appels": 15,
        "alerte_support": 5,
        "nettoyer": 20,
        "variables": 30,
        "entrainer": 45,
        "evaluer": 15,
        "scorer": 15,
        "sms": 5,
        "rapport": 10,
        "tableau_de_bord": 5,
    }


def taches_du_graphe(dependances: dict) -> set[str]:
    """Toutes les tâches du graphe : les clés ET les noms cités comme prérequis.

    Un outil pour EXPLORER (il ne vérifie rien : une chaîne à la place d'une liste serait lue lettre par
    lettre). Dans vos défis, écrivez votre propre fonction, qui vérifie les entrées (défi 146).

        >>> sorted(taches_du_graphe({"b": ["a"], "c": ["a", "b"]}))
        ['a', 'b', 'c']
    """
    taches = set(dependances)
    for prerequis in dependances.values():
        taches.update(prerequis)
    return taches


def generer_pipeline(n: int = 1000, graine: int = 0, max_prerequis: int = 3) -> dict[str, list[str]]:
    """Un grand graphe SANS cycle de n tâches, reproductible (même graine → même graphe).

    Les noms (« t00042 »…) sont tirés au hasard : l'ordre alphabétique n'a RIEN à voir avec l'ordre des
    dépendances. Chaque tâche a de 0 à `max_prerequis` prérequis, pris parmi les tâches créées avant elle
    (c'est ce qui garantit l'absence de cycle). Certaines tâches ne sont citées que comme prérequis.

        >>> grand = generer_pipeline(5000, graine=1)
        >>> len(taches_du_graphe(grand))
        5000
        >>> grand == generer_pipeline(5000, graine=1)
        True
    """
    rng = random.Random(graine)
    noms = [f"t{i:05d}" for i in rng.sample(range(10 * n), n)] if n else []
    dependances: dict[str, list[str]] = {}
    cites: set[str] = set()
    for i, nom in enumerate(noms):
        voisins = range(max(0, i - 50), i)  # des prérequis « récents » : de longues chaînes de dépendances
        prerequis = [noms[j] for j in rng.sample(voisins, min(rng.randint(0, max_prerequis), len(voisins)))]
        dependances[nom] = prerequis
        cites.update(prerequis)
    for nom in noms:  # 30 % des tâches sans prérequis, mais citées par une autre, ne sont pas des clés
        if not dependances[nom] and nom in cites and rng.random() < 0.3:
            del dependances[nom]
    return dependances


def generer_durees(dependances: dict, graine: int = 0, maximum: int = 60) -> dict[str, int]:
    """Une durée entière entre 0 et `maximum` minutes pour chaque tâche du graphe (quelques durées nulles).

        >>> durees = generer_durees(generer_pipeline(200, graine=3), graine=3)
        >>> len(durees), min(durees.values()) >= 0
        (200, True)
    """
    rng = random.Random(graine)
    taches = sorted(taches_du_graphe(dependances))
    return {nom: (0 if rng.random() < 0.05 else rng.randint(1, maximum)) for nom in taches}


def afficher_planning(planning: dict, largeur: int = 60) -> None:
    """Dessine un planning du défi 151 ({tâche: (début, fin, travailleur)}) : une ligne par travailleur.

        >>> afficher_planning({"a": (0, 10, 1), "b": (0, 5, 2), "c": (10, 20, 1)}, largeur=20)
        travailleur 1 : [a       ][c       ]   0 → 20
        travailleur 2 : [b  ]                  0 → 20
    """
    if not planning:
        print("(planning vide)")
        return
    fin_totale = max(fin for _, fin, _ in planning.values()) or 1
    echelle = largeur / fin_totale
    for travailleur in sorted({t for _, _, t in planning.values()}):
        ligne = [" "] * largeur
        for nom, (debut, fin, qui) in sorted(planning.items(), key=lambda item: item[1][0]):
            if qui != travailleur:
                continue
            a, b = round(debut * echelle), max(round(fin * echelle), round(debut * echelle) + 1)
            case = "[" + nom[: b - a - 2].ljust(b - a - 2) + "]" if b - a >= 2 else "|"
            ligne[a:b] = list(case)
        print(f"travailleur {travailleur} : {''.join(ligne)}   0 → {fin_totale:g}")

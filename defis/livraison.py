"""Outils communs de la série 39 : les versions publiées, les commits, les dépendances, les drapeaux et les conseillers.

Le 3 novembre, le script de déploiement a choisi la « dernière » version du service de score avec `max(tags)` : il
a pris 2.9.4 au lieu de 2.10.1 (en texte, "2.9.4" > "2.10.1"). Pendant trois semaines, la production a tourné avec
l'ancien modèle. Le 24, Théo a corrigé à la main, publié le nouveau score et l'a activé pour TOUS les conseillers
d'un coup : l'appli des conseillers en version 4.1 ne savait pas l'afficher, et le centre d'appels est resté
aveugle une matinée. La directrice technique a tranché : plus de comparaison de versions « à l'œil », plus de
numéro de version choisi à la main, plus de mise en production « tout d'un coup ». Les données :

- `tags_exemple()` : les versions publiées du service de score, dans l'ordre où `git tag` les affiche (l'ordre
  ALPHABÉTIQUE, pas celui des versions) ;
- `historique_exemple()` : les messages des commits écrits depuis la 2.10.1 (« Conventional Commits », défi 237) ;
- `index_exemple()` : les paquets dont dépend le service de score, chacun avec ses versions publiées et, pour
  chaque version, ce qu'elle demande ({paquet: {version: {dépendance: contrainte}}}, défi 239) ;
- `drapeaux_exemple()` : la configuration des drapeaux de fonctionnalité (« feature flags ») de l'appli des
  conseillers, dont « nouveau_score » (défi 241) ;
- `conseillers_exemple()` : les 480 conseillers qui utilisent l'appli, chacun avec son `id`, son `site`, son
  `equipe`, son `role` (parfois absent) et la version de son `appli` (parfois absente, parfois mal écrite) ;
- `Decision` (défi 241) : la valeur servie et la raison ; `Conflit` (défi 239) : l'erreur levée quand aucune
  combinaison de versions ne convient ;
- `PALIERS` (défi 240) : le nombre de seaux entre lesquels on répartit les conseillers, et `OPERATEURS`
  (défi 241) : les opérateurs qu'une règle de drapeau peut utiliser ;
- `generer_versions(n, graine)` et `generer_conseillers(n, graine)` : d'autres données, reproductibles, pour vos
  propres essais.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> tags = tags_exemple()
    >>> len(tags), tags[:3], max(tags)
    (24, ['1.0.0', '1.1.0', '1.2.0'], '2.9.4')
    >>> conseillers = conseillers_exemple()
    >>> len(conseillers), conseillers[0]
    (480, {'id': 'U0001', 'site': 'Rennes', 'equipe': 'rétention', 'role': 'stagiaire', 'appli': '4.1.3'})
"""

from __future__ import annotations

import random
from typing import NamedTuple

PALIERS = 10_000   # affecter (défi 240) range chaque conseiller dans l'un de ces 10 000 seaux (0.01 % chacun)

OPERATEURS = ("dans", "pas_dans", "version>=", "version<")   # ce qu'une condition de règle peut tester (défi 241)

SITES = ("Lyon", "Nantes", "Lille", "Rennes")
EQUIPES = ("rétention", "facturation", "technique")
ROLES = ("conseiller", "superviseur", "stagiaire")


class Conflit(Exception):
    """Aucune combinaison de versions ne satisfait toutes les contraintes (défi 239)."""


class Decision(NamedTuple):
    """Ce qu'un drapeau sert à un conseiller, et pourquoi (défi 241). Un tuple : (valeur, raison)."""

    valeur: str                  # la variante servie : "on", "off", ou toute autre valeur de la configuration
    raison: str                  # "coupé", "prérequis <nom>", "ciblé", "règle <n>" (n à partir de 1) ou "défaut"


# ---------------------------------------------------------------------------- versions et commits (236 à 238)


def tags_exemple() -> list[str]:
    """Les 24 versions publiées du service de score, dans l'ordre de `git tag` : l'ordre ALPHABÉTIQUE.

        >>> tags_exemple()[-6:]
        ['2.5.0', '2.6.0', '2.7.0', '2.8.0', '2.9.0', '2.9.4']
    """
    publiees = ["1.0.0", "1.1.0", "1.2.0", "1.2.1", "2.0.0-rc.1", "2.0.0", "2.1.0", "2.2.0", "2.3.0", "2.4.0",
                "2.5.0", "2.6.0", "2.7.0", "2.8.0", "2.9.0", "2.9.4", "2.10.0", "2.10.1", "2.11.0-alpha",
                "2.11.0-alpha.1", "2.11.0-beta.2", "2.11.0-beta.11", "2.11.0-rc.1", "2.10.1+build.1103"]
    return sorted(publiees)


def historique_exemple() -> list[str]:
    """Les messages des 14 commits écrits depuis la 2.10.1, du plus récent au plus ancien (comme `git log`).

        >>> historique_exemple()[0]
        'ci: ajouter Python 3.13 à la matrice'
    """
    return [
        "ci: ajouter Python 3.13 à la matrice",
        "Merge branch 'main' into nouveau-score",
        "perf: charger le modèle une seule fois au démarrage",
        "refactor(api): ne plus renvoyer le champ « score »\n\n"
        "Les clients de l'API lisent « proba » depuis la 2.9.\n\n"
        "BREAKING CHANGE: le champ « score » disparaît des réponses de /predict.",
        "test: couvrir le cas d'un client sans contrat",
        "fixup! fix: arrondir la probabilité à trois décimales",
        "WIP",
        "Feat(tableau): afficher la date du dernier entraînement",
        "chore: mettre à jour scikit-learn",
        "docs: expliquer le seuil de 0.5 (voir la note BREAKING CHANGE de la 2.0)",
        "fix: arrondir la probabilité à trois décimales",
        "feat(api): renvoyer la probabilité dans le champ « proba »\n\n"
        "Le champ « score » reste là pour l'instant, pour la compatibilité.",
        "fix : corriger une faute dans le README",
        "feature: ajouter un export CSV",
    ]


def generer_versions(n: int = 50, graine: int = 0) -> list[str]:
    """`n` versions SemVer valides, reproductibles, avec beaucoup de pièges : pré-versions, numéros à plusieurs
    chiffres, identifiants numériques et alphanumériques mêlés, métadonnées de build.

        >>> generer_versions(4, graine=1)
        ['1.1.9-0a.beta', '11.11.0', '2.1.10-alpha', '11.2.11']
    """
    rng = random.Random(graine)
    mots = ("alpha", "beta", "rc", "x", "pre", "a1", "0a", "rc-2", "-", "Z", "build")
    versions = []
    for _ in range(n):
        noyau = ".".join(str(rng.choice((0, 1, 1, 2, 9, 10, 11, 100))) for _ in range(3))
        if rng.random() < 0.45:
            ids = [str(rng.choice((0, 1, 2, 9, 10, 11))) if rng.random() < 0.45 else rng.choice(mots)
                   for _ in range(rng.choice((1, 1, 2, 2, 3)))]
            noyau += "-" + ".".join(ids)
        if rng.random() < 0.15:
            noyau += "+" + rng.choice(("build.1", "20261103", "exp.sha.5114f85", "001"))
        versions.append(noyau)
    return versions


# ---------------------------------------------------------------------------- dépendances (239)


def index_exemple() -> dict:
    """Les paquets dont dépend le service de score : {paquet: {version: {dépendance: contrainte}}}.

        >>> index = index_exemple()
        >>> sorted(index), index["graphiques"]["1.4.0"]
        (['churn-modele', 'graphiques', 'journal', 'numerique', 'tableaux'], {'numerique': '^1.9.0'})
    """
    return {
        "churn-modele": {
            "3.2.0": {"numerique": "^2.1.0", "tableaux": "^4.0.0"},
            "3.1.0": {"numerique": "^2.0.0", "tableaux": "^3.2.0"},
            "3.0.0": {"numerique": "^1.8.0", "tableaux": "^3.0.0"},
            "2.10.1": {"numerique": "^1.8.0"},
        },
        "graphiques": {
            "1.4.0": {"numerique": "^1.9.0"},
            "1.3.1": {"numerique": ">=1.8.0, <3.0.0"},
            "1.2.0": {"numerique": "^1.8.0"},
            "2.0.0-beta.1": {"numerique": "^2.0.0"},
        },
        "tableaux": {
            "4.0.0": {"numerique": "^2.0.0"},
            "3.2.0": {"numerique": ">=1.9.0"},
            "3.0.0": {},
        },
        "numerique": {"2.2.0-rc.1": {}, "2.1.0": {}, "2.0.0": {}, "1.9.2": {}, "1.8.0": {}},
        "journal": {"1.0.0-beta.1": {}, "0.4.0": {}, "0.3.2": {}},
    }


# ---------------------------------------------------------------------------- drapeaux et conseillers (240, 241)


def drapeaux_exemple() -> dict:
    """La configuration des drapeaux de l'appli des conseillers, telle qu'elle est en production le 25 novembre.

        >>> drapeaux = drapeaux_exemple()
        >>> sorted(drapeaux), drapeaux["nouveau_score"]["defaut"]
        (['export_csv', 'fiche_v2', 'nouveau_score'], 'off')
    """
    return {
        "fiche_v2": {
            "actif": True,
            "coupe": "off",
            "regles": [{"si": [["appli", "version>=", "4.1.0"]], "alors": "on"}],
            "defaut": "off",
        },
        "nouveau_score": {
            "actif": True,
            "coupe": "off",
            "prerequis": {"fiche_v2": "on"},
            "cibles": {"on": ["U0024", "U0035", "U0061"], "off": ["U0011"]},   # les testeurs, et un refus
            "regles": [
                {"si": [["role", "dans", ["superviseur"]], ["appli", "version>=", "4.2.0"]], "alors": "on"},
                {"si": [["appli", "version>=", "4.2.0"], ["site", "dans", ["Lyon", "Nantes"]],
                        ["role", "pas_dans", ["stagiaire"]]],
                 "alors": {"on": 25, "off": 75}},
            ],
            "defaut": "off",
        },
        "export_csv": {"actif": False, "coupe": "off", "defaut": {"on": 50, "off": 50}},
    }


_APPLIS = ("4.1.0", "4.1.3", "4.2.0", "4.2.1", "4.9.1", "4.10.2", "4.2.0-beta.3", None, "4.10")
_POIDS_APPLIS = (14, 14, 18, 10, 10, 22, 4, 5, 3)


def _tirer(rng: random.Random, valeurs: tuple, poids: tuple):
    x, cumul = rng.random() * sum(poids), 0
    for valeur, p in zip(valeurs, poids):
        cumul += p
        if x < cumul:
            return valeur
    return valeurs[-1]


def generer_conseillers(n: int = 480, graine: int = 0) -> list[dict]:
    """`n` conseillers, reproductibles. Le rôle manque pour 3 % d'entre eux (la clé est absente) ; la version de
    l'appli vaut None quand l'appli ne l'a pas envoyée, et "4.10" (invalide : il manque le correctif) pour une
    vieille version de test.

        >>> [sorted(c) for c in generer_conseillers(2, graine=3)]
        [['appli', 'equipe', 'id', 'role', 'site'], ['appli', 'equipe', 'id', 'role', 'site']]
    """
    rng = random.Random(graine)
    conseillers = []
    for i in range(n):
        conseiller = {
            "id": f"U{i + 1:04d}",
            "site": _tirer(rng, SITES, (35, 25, 25, 15)),
            "equipe": _tirer(rng, EQUIPES, (60, 25, 15)),
            "role": _tirer(rng, ROLES, (80, 8, 12)),
            "appli": _tirer(rng, _APPLIS, _POIDS_APPLIS),
        }
        if rng.random() < 0.03:
            del conseiller["role"]
        conseillers.append(conseiller)
    return conseillers


def conseillers_exemple() -> list[dict]:
    """Les 480 conseillers qui utilisent l'appli (les identifiants vont de U0001 à U0480)."""
    return generer_conseillers(480, graine=2511)

"""Outils communs de la série 29 : trois versions d'un même fichier, d'une configuration, d'une table.

Deux personnes de l'équipe partent du MÊME fichier (la « base », le dernier commit commun) et le modifient
chacune de leur côté : Inès (« nous », la branche où l'on est) et Malik (« eux », la branche qu'on fusionne).
C'est exactement ce que Git a sous la main quand vous tapez `git merge` : trois versions, pas deux.

- `versions_exemple()` : (base, nous, eux), trois listes de lignes de `score_risque`, modifiées à des
  endroits différents (la fusion se fait toute seule) ;
- `versions_en_conflit()` : la même chose, mais Inès et Malik ont changé la MÊME ligne ;
- `config_exemple()` : (base, nous, eux), trois versions de la configuration du modèle (des dict imbriqués,
  comme un fichier JSON ou YAML) ;
- `tables_exemple()` : (avant, apres), la table des clients produite par l'ancien et par le nouveau code de
  préparation des données (deux DataFrame) ;
- `generer_versions(n, graine)` : trois longues versions tirées au hasard, pour vos propres tests ;
- `PatchRefuse` : l'erreur à lever quand un patch ne s'applique pas (défi 178) ;
- `NOUS`, `BASE`, `SEPARATEUR`, `EUX` : les quatre lignes qui encadrent un conflit (défi 179).

Chaque fonction renvoie de NOUVELLES listes (ou de nouveaux dict, de nouvelles tables) à chaque appel :
modifiez-les sans crainte.

    >>> base, nous, eux = versions_exemple()
    >>> len(base), len(nous), len(eux)
    (9, 10, 11)
    >>> base[4], nous[5]                              # Inès a changé la ligne 5 (et ajouté une ligne 2)
    ('    if client["appels_support"] >= 3:', '    if client["appels_support"] >= 4:')
    >>> print(montrer(eux[-3:], debut=9))
      9 |     if client["internet"] == "Fibre":
     10 |         score += 1
     11 |     return min(score, 5)
    >>> issubclass(PatchRefuse, ValueError)
    True
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

NOUS = "<<<<<<< nous"
BASE = "||||||| base"
SEPARATEUR = "======="
EUX = ">>>>>>> eux"


class PatchRefuse(ValueError):
    """Le patch ne s'applique pas à ce fichier (lignes attendues absentes, en-tête faux, patch abîmé)."""


_SCORE = [
    'def score_risque(client):',
    '    score = 0',
    '    if client["contrat"] == "Mensuel":',
    '        score += 2',
    '    if client["appels_support"] >= 3:',
    '        score += 2',
    '    if client["anciennete_mois"] < 6:',
    '        score += 1',
    '    return score',
]

_DOCSTRING = '    """Un score de 0 à 5 : plus il est haut, plus le client risque de partir."""'
_FIBRE = ['    if client["internet"] == "Fibre":', '        score += 1']


def versions_exemple() -> tuple[list[str], list[str], list[str]]:
    """(base, nous, eux) : Inès ajoute une docstring et passe le seuil d'appels à 4 ; Malik ajoute une règle
    pour la fibre et plafonne le score à 5. Rien ne se chevauche : la fusion se fait toute seule."""
    base = list(_SCORE)
    nous = [_SCORE[0], _DOCSTRING, *_SCORE[1:4], '    if client["appels_support"] >= 4:', *_SCORE[5:]]
    eux = [*_SCORE[:8], *_FIBRE, '    return min(score, 5)']
    return base, nous, eux


def versions_en_conflit() -> tuple[list[str], list[str], list[str]]:
    """(base, nous, eux) : Inès passe le seuil d'appels à 4, Malik à 5 (la même ligne : un conflit). Malik
    ajoute aussi la règle de la fibre, qui, elle, ne gêne personne."""
    base = list(_SCORE)
    nous = [*_SCORE[:4], '    if client["appels_support"] >= 4:', *_SCORE[5:]]
    eux = [*_SCORE[:4], '    if client["appels_support"] >= 5:', *_SCORE[5:8], *_FIBRE, _SCORE[8]]
    return base, nous, eux


def config_exemple() -> tuple[dict, dict, dict]:
    """(base, nous, eux) : trois versions de la configuration du modèle.

    Inès monte le seuil à 0.4 et le budget à 6 000 € ; Malik active les variables dérivées, passe la
    relance par e-mail, baisse le budget à 4 500 € (la même clé qu'Inès : un conflit) et ajoute une section
    « surveillance ».
    """
    base = {
        "modele": {"nom": "logistic", "seuil": 0.35, "variables_derivees": False},
        "relance": {"canal": "sms", "budget_euros": 5000, "exclure": ["Deux ans"]},
        "version": 3,
    }
    nous = {
        "modele": {"nom": "logistic", "seuil": 0.4, "variables_derivees": False},
        "relance": {"canal": "sms", "budget_euros": 6000, "exclure": ["Deux ans"]},
        "version": 3,
    }
    eux = {
        "modele": {"nom": "logistic", "seuil": 0.35, "variables_derivees": True},
        "relance": {"canal": "email", "budget_euros": 4500, "exclure": ["Deux ans"]},
        "version": 3,
        "surveillance": {"psi_max": 0.25},
    }
    return base, nous, eux


def tables_exemple() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(avant, apres) : la table des clients produite par l'ancien et par le nouveau code de préparation.

    Le nouveau code renvoie les lignes dans un autre ordre, a perdu la colonne `segment`, ajouté
    `score_risque`, et calcule la facture d'une autre façon (0.1 + 0.2 au lieu de 0.3). Une seule VRAIE
    modification : le contrat de C005. C006 a disparu, C009 est apparu.
    """
    avant = pd.DataFrame({
        "client_id": ["C001", "C002", "C003", "C004", "C005", "C006"],
        "contrat": ["Mensuel", "Annuel", "Mensuel", "Deux ans", "Mensuel", "Annuel"],
        "anciennete_mois": [3, 40, 12, 70, 8, 22],
        "facture": [89.9, 45.0, 0.3, 30.5, 70.0, 55.0],
        "appels_support": [4.0, 0.0, 1.0, np.nan, 2.0, 0.0],
        "segment": ["A", "B", "A", "C", "A", "B"],
    })
    apres = pd.DataFrame({
        "client_id": ["C005", "C001", "C009", "C002", "C004", "C003"],
        "contrat": ["Annuel", "Mensuel", "Mensuel", "Annuel", "Deux ans", "Mensuel"],
        "anciennete_mois": [8.0, 3.0, np.nan, 40.0, 70.0, 12.0],
        "facture": [70.0, 89.9, 99.0, 45.0, 30.5, 0.1 + 0.2],
        "appels_support": [2.0, 4.0, 0.0, 0.0, np.nan, 1.0],
        "score_risque": [3, 5, 4, 0, 1, 2],
    })
    return avant, apres


def generer_versions(n: int = 200, graine: int = 0, taux: float = 0.08) -> tuple[list[str], list[str], list[str]]:
    """(base, nous, eux) : `n` lignes de base, toutes différentes, modifiées au hasard des deux côtés.

    Chaque côté supprime, remplace ou insère des lignes (environ `taux` des lignes), sans jamais DÉPLACER une
    ligne, et chaque nouvelle ligne est unique (« n12 : … » pour nous, « e7 : … » pour eux), sauf quand les
    deux côtés font exprès la même modification (« m3 : … »). Avec des lignes uniques et sans déplacement, la
    plus longue sous-suite commune est unique : la bonne fusion aussi.
    """
    alea = random.Random(graine)
    base = [f"ligne {i} : client C{alea.randrange(10_000):04d}" for i in range(n)]
    compteurs = {"n": 0, "e": 0, "m": 0}

    def nouvelle(prefixe: str) -> str:
        compteurs[prefixe] += 1
        return f"{prefixe}{compteurs[prefixe]} : {alea.choice(['seuil', 'budget', 'canal', 'score'])}"

    communes = {}  # position de base -> modification faite à l'identique des deux côtés
    for i in range(n + 1):
        if alea.random() < taux / 4:
            communes[i] = [nouvelle("m") for _ in range(alea.randint(1, 2))]

    def un_cote(prefixe: str) -> list[str]:
        sortie = []
        for i in range(n + 1):
            if i in communes:
                sortie.extend(communes[i])
            if alea.random() < taux / 2:
                sortie.extend(nouvelle(prefixe) for _ in range(alea.randint(1, 3)))   # insérer
            if i == n:
                break
            tirage = alea.random()
            if tirage < taux / 2:
                continue                                                         # supprimer
            if tirage < taux:
                sortie.append(nouvelle(prefixe))                                  # remplacer
                continue
            sortie.append(base[i])
        return sortie

    return base, un_cote("n"), un_cote("e")


def montrer(lignes: list[str], debut: int = 1) -> str:
    """Les lignes numérotées, comme un éditeur (numérotées à partir de `debut`)."""
    return "\n".join(f"{debut + i:>3} | {ligne}" for i, ligne in enumerate(lignes))

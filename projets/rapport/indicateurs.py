"""Étape 2 🟡 — Les chiffres du mois.

Vérifier : python scripts/projet.py --projet 7 --etape 2
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from .modeles import Abonne, Indicateurs  # noqa: F401
from .periodes import decaler, fin_du_mois  # noqa: F401


def indicateurs_du_mois(abonnes: Iterable[Abonne], mois: date) -> Indicateurs:
    """Les chiffres d'un mois, pour une liste d'abonnés (relisez `Abonne` et `Indicateurs` dans modeles.py).

    `mois` peut être n'importe quel jour du mois : le mois entier compte, et `Indicateurs.mois` vaut
    le 1er du mois (`decaler(mois, 0)`). Avec `premier` le 1er du mois et `dernier` son dernier jour
    (`fin_du_mois`), et en se souvenant que `fin` est le DERNIER jour d'abonnement (le client est
    encore là ce jour-là) :

    - `actifs_debut` : les abonnés au début du mois, c'est-à-dire arrivés AVANT le 1er
      (`debut < premier`) et toujours là le 1er (`fin` vide, ou `fin >= premier`). Un client arrivé le
      1er n'était pas là « au début » ; un client parti le 1er, si ;
    - `arrivees` : les abonnés dont le `debut` tombe dans le mois (du 1er au dernier jour, inclus) ;
    - `departs` : les abonnés dont la `fin` tombe dans le mois (du 1er au dernier jour, inclus). TOUS,
      même ceux arrivés pendant le mois ;
    - `actifs_fin` : les abonnés encore là après le dernier jour (`debut <= dernier`, et `fin` vide ou
      `fin > dernier`). On doit toujours avoir actifs_fin = actifs_debut + arrivees - departs ;
    - `taux_churn` : parmi les `actifs_debut`, la part de ceux qui sont partis pendant le mois.
      `None` (et pas 0.0) s'il n'y avait personne au début du mois : 0 % voudrait dire « personne
      n'est parti », alors qu'on ne sait rien ;
    - `revenu_perdu` : la somme des mensualités de TOUS les départs du mois. Ne l'arrondissez pas :
      l'arrondi, c'est le travail de l'affichage (étape 3).

    Pourquoi un client arrivé le 3 et parti le 20 compte dans les départs mais pas dans le taux ? Il
    n'était pas au dénominateur (les abonnés du début) ; le compter au numérateur ferait un taux qui
    peut dépasser 100 %. C'est la définition la plus courante du churn mensuel : écrivez-la dans le
    rapport (le corrigé le fait), sinon chaque lecteur imaginera la sienne.

    Ne modifiez pas la liste reçue. Ce que vous apprenez : traduire une définition métier en
    conditions exactes, les frontières (le 1er, le dernier jour, la veille, le lendemain), None
    contre 0, un invariant qui vérifie vos comptes.
    """
    raise NotImplementedError


def par_segment(abonnes: Iterable[Abonne], mois: date) -> dict[str, Indicateurs]:
    """Les chiffres du mois pour chaque segment : `{segment: Indicateurs}`.

    Un segment par valeur de `Abonne.segment` présente dans `abonnes` (même si aucun de ses abonnés
    n'est actif ce mois-là : ses chiffres valent alors 0 et son taux None), clés dans l'ordre de
    `sorted`. Réutilisez `indicateurs_du_mois` sur les abonnés de chaque segment : la règle ne doit
    être écrite qu'à un seul endroit.

    Piège : si `abonnes` est un générateur, il est vide au deuxième passage. Regroupez d'abord.
    """
    raise NotImplementedError

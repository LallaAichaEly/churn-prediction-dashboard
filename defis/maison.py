"""Outils de la série 23, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Une petite table de 12 clients pour essayer vos fonctions à la main, un générateur pour les essayer sur
des milliers de clients, et `gradient_numerique`, l'outil qui sert à VÉRIFIER un gradient écrit à la
main. Chaque appel renvoie une NOUVELLE table : modifiez-la autant que vous voulez.

    >>> from defis.serie_23 import *
    >>> clients_exemple().head(4)
            anciennete  facture  appels  churn
    client
    C01              2     80.0       4      1
    C02             40     25.0       0      0
    C03              5     55.0       3      1
    C04             12     60.0       1      0

Les colonnes :

- `anciennete` (en mois, un entier), `facture` (en euros par mois, au demi-euro près) et `appels` (au
  support, un entier) : des nombres, SANS trou. Ici, on s'intéresse à l'algorithme, pas au nettoyage ;
- `churn` : la cible, 1 si le client est parti, 0 sinon.

Les valeurs sont des entiers ou des demi-entiers : elles s'écrivent exactement en binaire, ce qui permet
de comparer vos arbres à ceux de scikit-learn au chiffre près (scikit-learn range X en float32).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIQUES = ["anciennete", "facture", "appels"]


def clients_exemple() -> pd.DataFrame:
    """Douze clients : beaucoup d'appels, une grosse facture et peu d'ancienneté annoncent un départ.

    - `appels` sépare le mieux : aucun des 4 clients qui ont appelé 0 ou 1 fois n'est parti. Ce n'est
      pas la PREMIÈRE colonne : un arbre (défi 145) doit comparer toutes les colonnes ;
    - C04 et C07 ont la même facture (60 €) : on ne coupe jamais ENTRE deux valeurs égales (défi 144) ;
    - C05 (25 mois, 4 appels) est resté, C08 (26 mois, 4 appels) est parti : certaines feuilles ne
      seront pas pures, et leur probabilité ne vaudra ni 0 ni 1.

        >>> clients = clients_exemple()
        >>> float(clients["churn"].mean())
        0.5
        >>> clients.groupby("appels")["churn"].mean().round(2).to_dict()
        {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.5, 4: 0.67, 5: 1.0, 6: 1.0}
    """
    return pd.DataFrame(
        {
            "anciennete": [2, 40, 5, 12, 25, 1, 24, 26, 4, 60, 3, 8],
            "facture": [80.0, 25.0, 55.0, 60.0, 80.5, 90.0, 60.0, 70.0, 30.0, 100.0, 85.0, 50.0],
            "appels": [4, 0, 3, 1, 4, 5, 1, 4, 3, 2, 6, 1],
            "churn": [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        },
        index=pd.Index([f"C{i:02d}" for i in range(1, 13)], name="client"),
    )


def generer_clients(n: int = 1000, graine: int = 0) -> pd.DataFrame:
    """`n` clients tirés au hasard (reproductible avec `graine`), mêmes colonnes que l'exemple.

    Le départ suit une vraie régression logistique (moins d'ancienneté, plus d'appels, une facture plus
    élevée → plus de départs), avec assez de bruit pour qu'aucun modèle ne soit parfait. Environ un client
    sur trois part.

        >>> clients = generer_clients(2000, graine=1)
        >>> clients.shape
        (2000, 4)
        >>> bool(0.25 < clients["churn"].mean() < 0.45)
        True
        >>> list(clients.index[:2])
        ['G0001', 'G0002']
    """
    rng = np.random.default_rng(graine)
    anciennete = rng.integers(0, 73, size=n)
    facture = np.round(rng.normal(65, 20, size=n).clip(15, 140) * 2) / 2
    appels = rng.poisson(1.6, size=n).clip(0, 9)
    score = 0.1 - 0.05 * anciennete + 0.025 * (facture - 65) + 0.45 * appels
    churn = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    return pd.DataFrame(
        {"anciennete": anciennete, "facture": facture, "appels": appels, "churn": churn},
        index=pd.Index([f"G{i:04d}" for i in range(1, n + 1)], name="client"),
    )


def gradient_numerique(fonction, point, h: float = 1e-6) -> np.ndarray:
    """Le gradient de `fonction` en `point`, estimé par DIFFÉRENCES FINIES CENTRÉES.

    Pour chaque coordonnée i, on bouge le point de +h puis de -h et on mesure la pente :

        dérivée_i ≈ (f(point + h·e_i) - f(point - h·e_i)) / (2h)

    C'est lent (2 appels de `fonction` par coordonnée) mais SÛR : on s'en sert pour VÉRIFIER un gradient
    écrit à la main (le « gradient check »), jamais pour entraîner. Un écart de plus de 1e-6 entre les
    deux, c'est presque toujours une erreur de formule.

        >>> gradient_numerique(lambda v: v[0] ** 2 + 3 * v[1], [2.0, 5.0]).round(6)
        array([4., 3.])
    """
    point = np.asarray(point, dtype=float)
    gradient = np.empty_like(point)
    for i in range(point.size):
        plus, moins = point.copy(), point.copy()
        plus.flat[i] += h
        moins.flat[i] -= h
        gradient.flat[i] = (fonction(plus) - fonction(moins)) / (2 * h)
    return gradient


def afficher_arbre(arbre: dict, colonnes=None) -> None:
    """Affiche un arbre du défi 145 (votre `arbre_`) comme `sklearn.tree.export_text`.

    `colonnes` donne le nom de chaque colonne (sinon : « colonne 0 », « colonne 1 »…).

        >>> afficher_arbre({"colonne": 0, "seuil": 2.5, "effectif": 5,
        ...                 "gauche": {"proba": 0.0, "effectif": 2},
        ...                 "droite": {"proba": 0.6666666666666666, "effectif": 3}}, ["appels"])
        |--- appels <= 2.5
        |   |--- feuille : proba 0.0 (2 clients)
        |--- appels >  2.5
        |   |--- feuille : proba 0.667 (3 clients)
    """

    def nom(j: int) -> str:
        return colonnes[j] if colonnes is not None else f"colonne {j}"

    def afficher(noeud: dict, marge: str) -> None:
        if "colonne" not in noeud:
            n = noeud["effectif"]
            print(f"{marge}|--- feuille : proba {round(float(noeud['proba']), 3)} ({n} client{'s' if n > 1 else ''})")
            return
        seuil = round(float(noeud["seuil"]), 4)
        print(f"{marge}|--- {nom(noeud['colonne'])} <= {seuil}")
        afficher(noeud["gauche"], marge + "|   ")
        print(f"{marge}|--- {nom(noeud['colonne'])} >  {seuil}")
        afficher(noeud["droite"], marge + "|   ")

    afficher(arbre, "")

"""Outils de la série 22, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Une petite table de 12 clients, construite pour que chaque piège de la série y apparaisse au moins une
fois, et un générateur pour essayer vos briques sur des centaines de clients. Chaque appel renvoie une
NOUVELLE table : modifiez-la autant que vous voulez, l'exemple suivant sera intact.

    >>> from defis.serie_22 import *
    >>> clients_exemple().head(4)
            anciennete  facture  appels  contrat     ville  churn
    client
    C01            2.0     79.9       4  Mensuel     Paris      1
    C02           40.0     25.0       0   Annuel     Paris      0
    C03            NaN     55.0       1  Mensuel      Lyon      1
    C04           12.0      NaN       2  Mensuel  Nanterre      0

Les colonnes :

- `anciennete` (en mois), `facture` (en euros par mois), `appels` (au support) : des nombres, avec des
  trous (NaN) dans les deux premières ;
- `contrat` (Mensuel, Annuel, Biannuel) et `ville` : des catégories. La ville a une « longue traîne » :
  quelques grandes villes, beaucoup de villes avec un ou deux clients, et une case vide (NaN) ;
- `churn` : la cible, 1 si le client est parti, 0 sinon.

L'index (« client ») n'est PAS 0, 1, 2… : c'est l'identifiant du client. Une brique qui le perd, ou qui
range les lignes par position au lieu de leur étiquette (ou l'inverse), se voit tout de suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIQUES = ["anciennete", "facture", "appels"]
CATEGORIES = ["contrat", "ville"]


def clients_exemple() -> pd.DataFrame:
    """Douze clients, un piège par ligne ou presque.

    - C03 et C09 n'ont pas d'ancienneté, C04 n'a pas de facture : il faut les compléter (défi 134) ;
    - C10 a une facture de 300 € : la MOYENNE des factures en est tirée vers le haut, pas la médiane ;
    - Paris (5 clients) et Lyon (3) sont fréquentes ; Nanterre, Albi et Brest n'ont qu'un client, et C12
      n'a pas de ville (NaN) : une case vide n'est pas une « ville rare » (défi 135) ;
    - le seul client d'Albi est parti : sans lissage, « Albi » vaudrait 100 % de churn (défi 137).
    """
    return pd.DataFrame(
        {
            "anciennete": [2.0, 40.0, np.nan, 12.0, 30.0, 1.0, 24.0, 6.0, np.nan, 60.0, 3.0, 18.0],
            "facture": [79.9, 25.0, 55.0, np.nan, 45.0, 89.0, 60.0, 70.0, 30.0, 300.0, 85.0, 50.0],
            "appels": [4, 0, 1, 2, 0, 5, 1, 3, 0, 1, 6, 2],
            "contrat": ["Mensuel", "Annuel", "Mensuel", "Mensuel", "Biannuel", "Mensuel", "Annuel", "Mensuel",
                        "Annuel", "Biannuel", "Mensuel", "Annuel"],
            "ville": ["Paris", "Paris", "Lyon", "Nanterre", "Paris", "Albi", "Lyon", "Paris", "Brest", "Paris",
                      "Lyon", np.nan],
            "churn": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        },
        index=pd.Index([f"C{i:02d}" for i in range(1, 13)], name="client"),
    )


_VILLES = ["Paris", "Lyon", "Marseille", "Toulouse", "Nantes", "Lille", "Rennes", "Bordeaux", "Nice", "Brest",
           "Dijon", "Angers", "Reims", "Metz", "Caen", "Tours", "Pau", "Albi", "Vannes", "Niort", "Laval", "Agen",
           "Dax", "Sète", "Gap", "Foix", "Auch", "Millau", "Riom", "Dole"]


def generer_clients(n: int = 500, graine: int = 0) -> pd.DataFrame:
    """`n` clients tirés au hasard (reproductibles : même `graine`, même table), mêmes colonnes que l'exemple.

    Le churn dépend vraiment du contrat, des appels, de l'ancienneté… et de la ville : quelques villes ont
    un concurrent agressif. Une ville sur deux n'a qu'une poignée de clients (la longue traîne), environ
    5 % des anciennetés et des factures manquent, et 3 % des villes. L'index va de « G0001 » à « Gnnnn ».
    """
    rng = np.random.default_rng(graine)
    contrat = rng.choice(["Mensuel", "Annuel", "Biannuel"], size=n, p=[0.55, 0.27, 0.18])
    echelle = np.select([contrat == "Mensuel", contrat == "Annuel"], [14.0, 30.0], default=45.0)
    anciennete = np.clip(rng.exponential(echelle), 1, 72).round()
    facture = np.clip(rng.normal(62, 20, size=n), 15, 140).round(2)
    appels = rng.poisson(1.3, size=n)
    poids = 1 / np.arange(1, len(_VILLES) + 1) ** 1.3                  # loi de Zipf : la longue traîne
    ville = rng.choice(_VILLES, size=n, p=poids / poids.sum()).astype(object)
    effet_ville = dict(zip(_VILLES, rng.normal(0, 0.8, size=len(_VILLES))))
    logit = (-1.0 + 1.5 * (contrat == "Mensuel") - 0.8 * (contrat == "Biannuel") - 0.035 * anciennete
             + 0.45 * appels + 0.012 * (facture - 62) + np.array([effet_ville[v] for v in ville])
             + rng.normal(0, 0.5, size=n))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    anciennete[rng.random(n) < 0.05] = np.nan
    facture[rng.random(n) < 0.05] = np.nan
    ville[rng.random(n) < 0.03] = np.nan
    return pd.DataFrame(
        {"anciennete": anciennete, "facture": facture, "appels": appels, "contrat": contrat, "ville": ville,
         "churn": churn},
        index=pd.Index([f"G{i:04d}" for i in range(1, n + 1)], name="client"),
    )

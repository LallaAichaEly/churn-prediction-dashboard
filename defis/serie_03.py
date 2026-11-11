"""Série 03 — Évaluer, expliquer, surveiller, corrigés automatiquement.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les séries 01 et 02 :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py` (ou `python scripts/defis.py --defi 17` pour un seul défi).
3. Bloqué·e ? Relisez l'indice, testez dans un terminal Python, puis seulement en dernier
   recours ouvrez `defis/corrige_serie_03.py`.

Cette série accompagne les modules 4, 5 et 7. Vous recodez à la main ce que scikit-learn et
le tableau de bord font pour vous : on ne fait vraiment confiance qu'à un outil dont on
connaît la mécanique.

Les défis 15 à 17 reçoivent `y_true` et `proba` sous forme de listes, de pd.Series ou de
tableaux numpy : commencez par `y_true, proba = np.asarray(y_true), np.asarray(proba)`.

Pour expérimenter avec un vrai modèle :
    >>> import sys; sys.path.insert(0, "src")
    >>> from churn.data import generate_customers
    >>> from churn.model import train
    >>> pipeline, X_test, y_test, proba, metrics = train(generate_customers())  # 1 250 clients de test
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (vous en aurez besoin)
import pandas as pd

# ======================================================================= Module 4 — Évaluation


def precision_rappel_par_seuil(y_true, proba, seuils) -> pd.DataFrame:
    """Défi 15 🟡 (Parcours 4.4) — Les points d'une courbe précision-rappel.

    Pour chaque seuil de `seuils` (dans l'ordre donné), un client est ciblé si proba >= seuil.
    Renvoyer un DataFrame avec exactement ces 4 colonnes, une ligne par seuil :

        seuil   clients_cibles   precision   rappel     (régression logistique,
        0.2               823        0.50      0.93      1 250 clients de test)
        0.5               408        0.69      0.63
        0.9                 9        0.89      0.02

    Si personne n'est ciblé, la précision n'est pas définie : mettez NaN.
    (Au défi 6, on mettait 0 par convention. Sur une courbe, ce 0 inventé dessinerait une
    chute brutale qui n'existe pas : un trou dans la courbe est plus honnête.)

    Indice : une boucle sur les seuils suffit. `np.nan` est le NaN de numpy.
    """
    raise NotImplementedError


def table_calibration(y_true, proba, n_tranches: int = 10) -> pd.DataFrame:
    """Défi 16 🟡 (Parcours 4.5) — Le modèle dit 70 % : est-ce que 70 % de ces clients partent ?

    Découper [0, 1] en `n_tranches` tranches de même largeur. Avec 10 tranches :
    [0 ; 0,1[, [0,1 ; 0,2[, …, [0,9 ; 1]. La DERNIÈRE tranche inclut 1.

    Renvoyer un DataFrame avec exactement ces 4 colonnes :

        tranche   clients   proba_moyenne   taux_reel     (même modèle, mêmes clients)
              0       253            0.04        0.02
              1       174            0.15        0.14
            ...
              9         9            0.93        0.89

    - `tranche` : numéro de la tranche (0 pour [0 ; 0,1[, 9 pour [0,9 ; 1]) ;
    - `proba_moyenne` : moyenne des probas PRÉDITES dans la tranche (pas le milieu de la tranche) ;
    - `taux_reel` : part de clients réellement partis dans la tranche ;
    - seulement les tranches NON VIDES, de la plus basse à la plus haute, index 0, 1, 2…

    Un modèle bien calibré a proba_moyenne ≈ taux_reel dans chaque tranche.

    Indice : `(proba * n_tranches).astype(int)` donne le numéro de tranche… sauf pour une
    proba de 1 exactement. Puis `groupby` et `.agg(...)`.
    """
    raise NotImplementedError


def auc_par_paires(y_true, proba) -> float:
    """Défi 17 🔴 — La ROC-AUC SANS courbe ROC.

    Prenez un client parti et un client resté, au hasard. L'AUC est la probabilité que le
    modèle donne un score plus élevé au client parti. Donc :

        AUC = (paires où le parti a le score le plus haut + ½ × paires ex aequo)
              / (nombre de partis × nombre de restés)

    C'est pourquoi une AUC de 0,5 = hasard, et pourquoi l'AUC ne dépend pas du seuil.

    Lever une `ValueError` s'il n'y a qu'une seule classe dans `y_true` (aucune paire possible).

    Indice : séparez les scores des partis et des restés, puis comparez-les tous d'un coup
    avec le broadcasting numpy : `partis[:, None] > restes[None, :]` est une matrice de booléens.
    """
    raise NotImplementedError


# ==================================================================== Module 5 — Interprétation


def contributions_client(pipeline, client: pd.DataFrame, top: int | None = 3) -> pd.DataFrame:
    """Défi 18 🔴 (Parcours 5.3) — Pourquoi CE client est-il jugé à risque ?

    `pipeline` est une régression logistique entraînée, construite par
    `churn.model.build_pipeline("logistic", ...)` ; `client` est un DataFrame d'UNE ligne.

    Pour une régression logistique : logit = intercept + Σ coefficient × valeur transformée.
    Chaque terme « coefficient × valeur » est la contribution d'une variable :
    positive, elle pousse vers le départ ; négative, elle retient le client.

    Renvoyer un DataFrame avec les colonnes ["variable", "contribution"] :
    - `variable` : les noms donnés par `get_feature_names_out()` (ex. "num__tenure_months",
      "cat__contract_Mensuel") ;
    - sans les contributions nulles (modalités one-hot absentes chez ce client) ;
    - trié par contribution en VALEUR ABSOLUE décroissante (une variable qui retient fortement
      compte autant qu'une variable qui pousse fortement) ;
    - les `top` premières lignes (toutes si top=None), index 0, 1, 2…

    Outils :
        pipeline[:-1].transform(client)                    → les valeurs que le modèle voit vraiment
        pipeline[-1].coef_[0]                              → les coefficients
        pipeline.named_steps["prep"].get_feature_names_out() → les noms de ces valeurs

    Le pipeline peut contenir une étape de variables dérivées avant "prep" (use_engineered=True) :
    `pipeline[:-1]` applique toutes les étapes sauf le modèle, quel que soit leur nombre.

    Indice : `.transform()` peut renvoyer une matrice creuse (sparse) ; `.toarray()` la convertit.
    `sort_values(key=abs, ...)` trie par valeur absolue.
    """
    raise NotImplementedError


# ===================================================================== Module 7 — Surveillance


def psi_parts(parts_reference, parts_actuelles) -> float:
    """Défi 19 🟢 (Parcours 7.2) — Le PSI à partir des parts de chaque tranche.

        PSI = Σ (actuelle − référence) × ln(actuelle / référence)

    Exemple : une variable passe de 50 % / 50 % à 70 % / 30 % → PSI ≈ 0,1695 (à surveiller).

    Règles (les mêmes que `churn.monitoring.psi`) :
    - accepter des effectifs comme des parts : [50, 50] et [0.5, 0.5] donnent le même PSI.
      Divisez donc chaque liste par sa somme ;
    - une tranche vide donnerait ln(0) : remplacez les parts inférieures à 0,0001 par 0,0001 ;
    - lever une `ValueError` si les deux listes n'ont pas la même longueur.

    Indice : `np.asarray(..., dtype=float)`, `np.clip(parts, 1e-4, None)` et `np.log`
    (logarithme naturel).
    """
    raise NotImplementedError

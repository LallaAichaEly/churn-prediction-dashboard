"""Série 04 — Modéliser et industrialiser, corrigés automatiquement.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les séries précédentes :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py` (ou `python scripts/defis.py --defi 23` pour un seul défi).
3. Bloqué·e ? Relisez l'indice, testez dans un terminal Python, puis seulement en dernier
   recours ouvrez `defis/corrige_serie_04.py`.

Cette série accompagne les modules 3, 6 et 7. Un modèle qui marche dans un notebook n'est que la
moitié du travail : il faut le régler honnêtement, puis le faire tourner en production sans
qu'il se trompe en silence. Chaque défi correspond à une décision qu'on prend vraiment en
entreprise.

Pour expérimenter avec un vrai modèle :
    >>> import sys; sys.path.insert(0, "src")
    >>> from churn.data import FEATURES, TARGET, generate_customers
    >>> from churn.model import build_pipeline
    >>> df = generate_customers(n=2000, seed=11)
    >>> pipeline = build_pipeline("logistic").fit(df[FEATURES], df[TARGET])
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (vous en aurez besoin)
import pandas as pd

# ======================================================================= Module 3 — Modéliser


def poids_classes(y) -> dict:
    """Défi 20 🟢 (Parcours 3.6) — Ce que fait vraiment `class_weight="balanced"`.

    Avec ~35 % de churn, un modèle apprend surtout sur les clients restés. « balanced » donne à
    chaque classe un poids inversement proportionnel à son effectif :

        poids(classe) = n / (k × effectif(classe))

    n = nombre total d'exemples, k = nombre de classes présentes dans `y`.

    Renvoyer un dictionnaire {classe: poids}, par ex. pour y = [0, 0, 0, 1] :
        {0: 0.667, 1: 2.0}      → un client parti pèse 3 fois plus qu'un client resté

    Conséquence à vérifier vous-même : Σ poids × effectif vaut n pour chaque série de poids,
    et chaque classe pèse AU TOTAL n / k. Les partis pèsent autant que les restés.

    La formule marche pour plus de 2 classes (ex. resté / parti chez un concurrent / déménagé) :
    ne supposez pas que k vaut 2, lisez-le dans les données.

    Indice : `np.unique(y, return_counts=True)` donne les classes et leurs effectifs.
    `y` peut être une liste, une pd.Series ou un tableau numpy.
    """
    raise NotImplementedError


def choisir_C(X: pd.DataFrame, y, valeurs_C, n_plis: int = 5, seed: int = 0) -> tuple[pd.DataFrame, float]:
    """Défi 21 🔴 (Parcours 3.5) — Régler un hyperparamètre honnêtement.

    `C` règle la régularisation de la régression logistique : plus C est PETIT, plus le modèle
    est contraint (coefficients rétrécis), donc plus il est simple.

    1. Pour chaque C de `valeurs_C` (dans l'ordre donné), mesurer la ROC-AUC en validation
       croisée. Utilisez les MÊMES plis pour tous les C, sinon la comparaison n'est pas juste :
           cv = StratifiedKFold(n_splits=n_plis, shuffle=True, random_state=seed)
           modele = build_pipeline("logistic").set_params(clf__C=C)
       Renvoyer un DataFrame avec exactement les colonnes ["C", "auc_moyenne", "auc_ecart_type"],
       une ligne par C dans l'ordre de `valeurs_C`, écart-type = np.std des AUC des plis.

    2. Choisir C avec la règle de l'erreur standard (pratique courante, cf. *The Elements of
       Statistical Learning*, § 7.10) :
       - meilleur = la ligne de plus haute auc_moyenne ;
       - erreur standard du meilleur = son auc_ecart_type / √n_plis ;
       - C choisi = le PLUS PETIT C dont auc_moyenne >= meilleure moyenne − erreur standard.
       Autrement dit, le modèle le plus simple qui n'est pas significativement moins bon.

    Renvoyer (table, C_choisi).

    Pourquoi ne pas prendre le meilleur score ? Sur nos données, l'AUC gagne 0,0002 entre
    C=0,1 et C=1 : c'est du bruit, bien inférieur à l'écart entre les plis. Prendre le maximum,
    c'est choisir le C qui a eu de la chance sur CES plis.

    Outils :
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from churn.model import build_pipeline

    Indice : le défi 9 (série 01) fait la validation croisée pour UN modèle. Ici, une boucle sur C.
    """
    raise NotImplementedError


# ================================================================ Module 6 — Industrialiser


def versions_compatibles(sauvegardee: str | None, installee: str) -> bool:
    """Défi 22 🟢 (Parcours 6.2) — Peut-on charger ce modèle sans risque ?

    Un fichier joblib dépend du code interne de scikit-learn. On considère qu'un modèle est
    compatible si la version installée a le même nombre MAJEUR et MINEUR que celle qui l'a
    entraîné. Le 3e nombre (correctif) ne change pas le format.

        versions_compatibles("1.5.2", "1.5.0")   → True   (seul le correctif diffère)
        versions_compatibles("1.4.2", "1.5.0")   → False  (mineur différent)
        versions_compatibles("1.6", "1.6.1")     → True   (pas de correctif écrit)
        versions_compatibles(None, "1.5.0")      → False  (ancien modèle sans version notée :
                                                            on ne peut rien garantir)

    Attention : une version n'est PAS un nombre décimal. 1.10 vient après 1.9, et
    "1.10.1" n'est pas compatible avec "1.1.0".

    Indice : `"1.10.1".split(".")` donne ["1", "10", "1"] ; comparez des entiers.
    """
    raise NotImplementedError


def predire_lot(pipeline, clients: list[dict], seuil: float = 0.5) -> list[dict]:
    """Défi 23 🟡 (Parcours 6.4) — Le cœur d'une route `POST /predict/batch`.

    `clients` est une liste de dictionnaires, comme le JSON reçu par une API. Leurs clés peuvent
    être dans n'importe quel ordre et contenir des clés en plus (ex. "customer_id").

    Renvoyer une liste de dictionnaires, un par client, DANS LE MÊME ORDRE :
        [{"churn_probability": 0.8731, "at_risk": True}, {"churn_probability": 0.0412, "at_risk": False}]
    - at_risk = churn_probability >= seuil (même règle que l'API) ;
    - des types Python (float, bool), pas numpy : le résultat doit passer dans `json.dumps`.
      (`json.dumps(np.bool_(True))` plante : `bool(...)` convertit.)

    Trois exigences de production :
    1. Vectoriser : appeler `pipeline.predict_proba` UNE SEULE fois pour tout le lot.
       1 000 appels pour 1 000 clients est des dizaines de fois plus lent.
    2. Une liste vide renvoie [] (scikit-learn refuse un tableau de 0 ligne).
    3. Si un client n'a pas une variable de `FEATURES` (clé absente ou valeur None), lever une
       `ValueError` dont le message NOMME la variable. Sans ce contrôle, un client sans
       "contract" est prédit sans aucune erreur : la modalité manquante est traitée comme
       inconnue et ignorée, et la probabilité est fausse. Une erreur claire vaut mieux
       qu'une réponse fausse silencieuse.

    Et ne modifiez pas les dictionnaires reçus : ils appartiennent à l'appelant.

    Outils : `from churn.data import FEATURES` ; `pd.DataFrame(clients)[FEATURES]`.
    """
    raise NotImplementedError


# ===================================================================== Module 7 — Surveillance


def alerte_derive(rapport: pd.DataFrame, seuil: float = 0.25) -> tuple[int, list[str]]:
    """Défi 24 🟢 (Parcours 7.6) — Décider s'il faut réveiller quelqu'un.

    `rapport` est un DataFrame comme celui de `churn.monitoring.drift_report` : au moins les
    colonnes "feature" et "psi", une ligne par variable, dans un ordre quelconque (un rapport
    relu depuis un CSV n'est pas forcément trié).

    Renvoyer (code, variables) :
    - variables : les variables dont le PSI est >= seuil, de la plus dérivée à la moins dérivée,
      PUIS celles dont le PSI est manquant (NaN) ;
    - code : 1 s'il y a au moins une variable dans la liste, 0 sinon. C'est le code de sortie de
      `scripts/monitor.py` (exercice 7.6) : une tâche planifiée ou la CI alerte quand il vaut 1.

    Deux règles de surveillance :
    - >= et pas > : un PSI de 0,25 pile est déjà une dérive (voir `churn.monitoring.status`) ;
    - un PSI manquant veut dire qu'on n'a PAS PU mesurer. En surveillance, l'absence de mesure
      est une alerte, pas un feu vert. Or `NaN >= 0.25` vaut False : il faut le traiter à part.

    Utilisez `seuil` et pas la colonne "statut" (calculée pour 0,25 uniquement), et ne modifiez
    pas `rapport`.

    Indice : `sort_values("psi", ascending=False, na_position="last")`, puis un masque
    `(psi >= seuil) | psi.isna()`.
    """
    raise NotImplementedError

"""Série 05 — Prouver l'impact : SQL, incertitude, tests A/B, survie.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les séries précédentes :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py` (ou `python scripts/defis.py --defi 27` pour un seul défi).
3. Bloqué·e ? Relisez l'indice, testez dans un terminal Python, puis seulement en dernier
   recours ouvrez `defis/corrige_serie_05.py`.

Un modèle ne vaut que par les décisions qu'il fait prendre. Cette série couvre les compétences
qu'on vous demandera dès qu'il faudra CONVAINCRE : interroger la base de l'entreprise en SQL,
dire « 54 % ± combien ? », prouver qu'une offre de rétention marche vraiment, et répondre à
« combien de temps un client reste-t-il ? ». Ce sont des classiques des entretiens data.

Aucune installation : SQLite et `statistics` font partie de Python.

Pour expérimenter en SQL :
    >>> import sqlite3, sys; sys.path.insert(0, "src")
    >>> import pandas as pd
    >>> from churn.data import generate_customers
    >>> connexion = sqlite3.connect(":memory:")            # une base en mémoire
    >>> generate_customers(n=2000).to_sql("clients", connexion, index=False)
    >>> pd.read_sql_query("SELECT contract, COUNT(*) AS n FROM clients GROUP BY contract", connexion)
"""

from __future__ import annotations

import math  # noqa: F401  (vous en aurez besoin)
from statistics import NormalDist  # noqa: F401  (la loi normale, sans installer scipy)

import pandas as pd

# =========================================================================== SQL


def requete_churn_par_contrat() -> str:
    """Défi 25 🟢 (Parcours 1.2) — Votre premier GROUP BY en SQL.

    Dans une entreprise, les données sont rarement dans un CSV : elles sont dans une base, et
    le SQL est la compétence la plus demandée dans les offres data. Renvoyez le TEXTE d'une
    requête (une str) ; le correcteur l'exécute sur une table `clients` qui a les mêmes colonnes
    que `data/customers.csv`.

    Résultat attendu, une ligne par contrat, du taux de churn le plus élevé au plus faible :

        contract  clients  taux_churn       (sur generate_customers(n=2000))
        Mensuel      1094        0.52
        Annuel        493        0.21
        Biannuel      413        0.09

    C'est exactement le `df.groupby("contract")["churn"].mean()` de l'exercice 1.2.

    Piège classique : `SUM(churn) / COUNT(*)` divise deux ENTIERS, et SQL fait alors une division
    entière (2 / 3 vaut 0). Pensez à `AVG`.

    Indice : SELECT … AS clients, … AS taux_churn FROM clients GROUP BY … ORDER BY … DESC
    """
    raise NotImplementedError


def requete_top_risque() -> str:
    """Défi 26 🟡 — « Les k clients les plus à risque de CHAQUE contrat » (question d'entretien n°1).

    Chaque type de contrat a son équipe de rétention, qui peut appeler k clients cette semaine.
    La table `scores` a trois colonnes : customer_id, contract, proba_churn.

    Renvoyez le texte d'une requête qui donne, pour chaque contrat, ses k clients de plus forte
    proba_churn, avec exactement les colonnes ["contract", "rang", "customer_id", "proba_churn"] :
    - rang = 1 pour le plus risqué du contrat, puis 2, 3… sans ex aequo ni trou ;
    - en cas d'égalité de proba_churn, le plus petit customer_id passe devant (résultat
      reproductible, d'une exécution à l'autre) ;
    - un contrat qui a moins de k clients les renvoie tous ;
    - lignes triées par contract puis rang.

    k n'est PAS écrit dans la requête : utilisez le paramètre nommé `:k`. Le correcteur exécute
        pd.read_sql_query(votre_requete, connexion, params={"k": 3})
    Coller une valeur dans le texte (f-string) est la porte ouverte aux injections SQL.

    Pourquoi `LIMIT :k` ne suffit pas : il limite TOUT le résultat, pas chaque contrat.

    Outils : une fonction de fenêtre,
        ROW_NUMBER() OVER (PARTITION BY … ORDER BY … DESC, … ASC) AS rang
    et une sous-requête `WITH classement AS (SELECT …) SELECT … FROM classement WHERE rang <= :k`,
    car WHERE est évalué AVANT les fonctions de fenêtre.

    Question à vous poser : quelle différence entre ROW_NUMBER, RANK et DENSE_RANK quand trois
    clients ont la même probabilité ?
    """
    raise NotImplementedError


# =========================================================================== Incertitude


def intervalle_wilson(departs: int, clients: int, confiance: float = 0.95) -> tuple[float, float]:
    """Défi 27 🟡 — « 50 % de churn »… sur 10 clients ou sur 10 000 ?

    Un taux mesuré sur un échantillon est incertain. Renvoyez l'intervalle de confiance de
    Wilson (bas, haut) du taux departs / clients :

        p = departs / clients,  n = clients,  z = quantile de la loi normale (≈ 1,96 à 95 %)
        centre       = (p + z² / (2n)) / (1 + z² / n)
        demi_largeur = z / (1 + z² / n) × √( p(1 − p) / n + z² / (4n²) )
        intervalle   = (centre − demi_largeur, centre + demi_largeur)

        intervalle_wilson(5, 10)       → (0.237, 0.763)   « entre 24 % et 76 % » : on ne sait presque rien
        intervalle_wilson(500, 1000)   → (0.469, 0.531)
        intervalle_wilson(0, 12)       → (0.0, 0.243)     0 départ ne prouve pas 0 % de churn

    - z dépend de `confiance` et l'intervalle est BILATÉRAL (il laisse (1 − confiance) / 2 de
      chaque côté) : z = NormalDist().inv_cdf(1 - (1 - confiance) / 2).
    - Les bornes restent dans [0 ; 1] (bornez avec max / min contre les erreurs d'arrondi).
    - Lever une ValueError si clients <= 0 ou si departs n'est pas entre 0 et clients.

    Pourquoi pas p ± z·√(p(1−p)/n) (intervalle « de Wald ») ? Essayez-le sur (0, 12) et (1, 10).

    Lien avec le défi 14 (série 02) : `min_clients` écartait les petits segments au jugé ; un
    intervalle de confiance dit PRÉCISÉMENT à quel point un segment est incertain.
    """
    raise NotImplementedError


def comparer_ab(partis_temoin: int, n_temoin: int, partis_offre: int, n_offre: int) -> dict:
    """Défi 28 🟡 — L'offre de rétention a-t-elle VRAIMENT réduit le churn ?

    Le marketing a tiré au sort deux groupes de clients à risque : le groupe « offre » a reçu une
    remise, le groupe « témoin » rien. Trois mois plus tard :
        témoin : 120 partis sur 400 (30 %)      offre : 96 partis sur 400 (24 %)
    Six points de moins ! Effet de l'offre, ou hasard du tirage ?

    Test z de deux proportions (bilatéral). Renvoyez {"ecart": …, "z": …, "p_value": …} :
        ecart          = taux offre − taux témoin            (négatif = l'offre réduit le churn)
        p_commun       = (partis_temoin + partis_offre) / (n_temoin + n_offre)
        erreur_std     = √( p_commun (1 − p_commun) (1/n_temoin + 1/n_offre) )
        z              = ecart / erreur_std
        p_value        = 2 × (1 − Φ(|z|))                    Φ = NormalDist().cdf

    - Variance COMMUNE : le test suppose « l'offre ne change rien » (hypothèse nulle), donc un
      même taux pour les deux groupes, estimé sur tous les clients réunis.
    - Bilatéral : une offre maladroite peut aussi AUGMENTER le churn ; on teste les deux sens.
    - Si erreur_std vaut 0 (aucun départ, ou que des départs), renvoyer z = 0.0, p_value = 1.0.
    - Lever une ValueError si un groupe est vide.

    Résultat sur l'exemple : p_value ≈ 0,056 > 0,05. Avec 400 clients par groupe, un écart de
    6 points n'est PAS une preuve. Avec 1 000 par groupe (300 contre 240), p ≈ 0,003. La taille de
    l'échantillon se décide AVANT de lancer le test, pas après avoir vu le résultat.

    Vérifiez vous-même : scipy.stats.chi2_contingency([[120, 280], [96, 304]], correction=False)
    donne la même p-value.
    """
    raise NotImplementedError


# =========================================================================== Survie


def courbe_survie(df: pd.DataFrame) -> pd.DataFrame:
    """Défi 29 🔴 — Combien de temps un client reste-t-il ? (estimateur de Kaplan-Meier)

    Le taux de churn global (~35 %) ne répond pas à « quelle part des clients est encore là
    après 12 mois ? ». Ici, `tenure_months` est l'ancienneté et `churn` dit si le client est
    parti à ce moment-là (1) ou s'il est TOUJOURS client (0). Un client toujours là après 8 mois
    est « censuré » : on sait qu'il a tenu 8 mois, on ignore la suite.

    Renvoyer un DataFrame, une ligne par mois où au moins un client est parti, mois croissants,
    avec exactement les colonnes ["mois", "a_risque", "departs", "survie"] :
        a_risque(t) = nombre de clients avec tenure_months >= t   (encore là au début du mois t,
                                                                    censurés compris)
        departs(t)  = nombre de clients avec tenure_months == t et churn == 1
        survie(t)   = produit, pour tous les mois s <= t de la table, de (1 − departs(s) / a_risque(s))

    Exemple : anciennetés [1, 2, 2, 3, 4, 5], churn [1, 1, 0, 1, 0, 1]
        mois  a_risque  departs  survie
           1         6        1   0.833     5/6
           2         5        1   0.667     5/6 × 4/5
           3         3        1   0.444     … × 2/3   (le client resté à 2 mois est sorti du calcul)
           5         1        1   0.000

    Pourquoi pas « 1 − départs cumulés / tous les clients » ? Un client arrivé il y a 2 mois ne
    PEUT pas être parti au mois 12 : le compter au dénominateur fait croire que les clients
    restent plus longtemps qu'en réalité. Sur les 5 000 clients du projet, à 12 mois :
    Kaplan-Meier 74,7 %, formule naïve 77,8 %. Trois points d'optimisme injustifié.

    Si personne n'est parti, renvoyer un DataFrame vide qui a quand même ces 4 colonnes.
    Ne modifiez pas `df`.

    Pour aller plus loin : tracez la survie par contrat (Mensuel contre Biannuel), et trouvez la
    durée de vie médiane (premier mois où survie <= 0,5). Référence : lifelines.KaplanMeierFitter.

    Indice : une boucle sur `sorted(df.loc[df["churn"] == 1, "tenure_months"].unique())` et une
    variable `survie` qu'on multiplie à chaque tour.
    """
    raise NotImplementedError

"""Série 05 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_05.py`.

Si votre version passe les vérifications, elle est juste. Comparez-la à celle-ci pour
découvrir d'autres écritures, puis notez dans JOURNAL.md ce que vous avez appris.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import pandas as pd

# =========================================================================== SQL


def requete_churn_par_contrat() -> str:
    return """
        SELECT contract,
               COUNT(*)   AS clients,
               AVG(churn) AS taux_churn      -- AVG d'une colonne 0/1 = une proportion, en décimal
        FROM clients
        GROUP BY contract
        ORDER BY taux_churn DESC
    """
    # Le piège : SUM(churn) / COUNT(*) fait une division ENTIÈRE en SQLite (et en PostgreSQL) :
    # 2 / 3 vaut 0. Écrire 1.0 * SUM(churn) / COUNT(*) force le calcul en décimal.
    # À comparer avec pandas : df.groupby("contract")["churn"].agg(clients="size", taux_churn="mean").


def requete_top_risque() -> str:
    return """
        WITH classement AS (
            SELECT contract,
                   customer_id,
                   proba_churn,
                   ROW_NUMBER() OVER (
                       PARTITION BY contract                       -- un classement PAR contrat
                       ORDER BY proba_churn DESC, customer_id ASC  -- ex aequo : départage stable
                   ) AS rang
            FROM scores
        )
        SELECT contract, rang, customer_id, proba_churn
        FROM classement
        WHERE rang <= :k              -- un paramètre, pas une valeur collée dans le texte
        ORDER BY contract, rang
    """
    # Pourquoi une sous-requête (CTE) ? WHERE est évalué AVANT les fonctions de fenêtre : on ne
    # peut pas écrire `WHERE ROW_NUMBER() OVER (...) <= :k` directement.
    # ROW_NUMBER donne toujours 1, 2, 3… ; RANK donnerait 1, 2, 2, 2 en cas d'égalité, donc plus
    # de k clients pour un contrat : l'équipe recevrait plus d'appels qu'elle ne peut en passer.
    # Équivalent pandas :
    #   scores.sort_values(["proba_churn", "customer_id"], ascending=[False, True])
    #         .groupby("contract").head(k)


# =========================================================================== Incertitude


def _z_bilateral(confiance: float) -> float:
    # 95 % au centre → 2,5 % dans chaque queue → quantile 0,975 de la loi normale (≈ 1,96).
    return NormalDist().inv_cdf(1 - (1 - confiance) / 2)


def intervalle_wilson(departs: int, clients: int, confiance: float = 0.95) -> tuple[float, float]:
    if clients <= 0 or not 0 <= departs <= clients:
        raise ValueError(f"il faut 0 <= departs <= clients et clients > 0 (reçu {departs} / {clients})")
    z = _z_bilateral(confiance)
    p, n = departs / clients, clients
    denominateur = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominateur
    demi_largeur = z / denominateur * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    # max / min : protège des erreurs d'arrondi (ex. -1e-17 au lieu de 0).
    return max(0.0, centre - demi_largeur), min(1.0, centre + demi_largeur)
    # L'intervalle « de Wald » p ± z·√(p(1−p)/n) qu'on apprend souvent en premier donne [0 ; 0]
    # pour 0 départ sur 12 clients (une certitude absurde) et une borne négative pour 1 sur 10.
    # Wilson reste dans [0 ; 1] et n'est jamais de largeur nulle. Vérification possible :
    # statsmodels.stats.proportion.proportion_confint(departs, clients, method="wilson").


def comparer_ab(partis_temoin: int, n_temoin: int, partis_offre: int, n_offre: int) -> dict:
    if n_temoin <= 0 or n_offre <= 0:
        raise ValueError("chaque groupe doit contenir au moins un client")
    p_temoin, p_offre = partis_temoin / n_temoin, partis_offre / n_offre
    ecart = p_offre - p_temoin  # négatif = l'offre réduit le churn

    # Sous l'hypothèse nulle « l'offre ne change rien », les deux groupes ont le MÊME taux :
    # on l'estime sur les deux groupes réunis (variance commune, « pooled »).
    p_commun = (partis_temoin + partis_offre) / (n_temoin + n_offre)
    erreur_standard = math.sqrt(p_commun * (1 - p_commun) * (1 / n_temoin + 1 / n_offre))
    if erreur_standard == 0:  # 0 % ou 100 % de départs partout : aucune différence observable
        return {"ecart": ecart, "z": 0.0, "p_value": 1.0}

    z = ecart / erreur_standard
    p_value = 2 * (1 - NormalDist().cdf(abs(z)))  # bilatéral : un écart dans les deux sens compte
    return {"ecart": ecart, "z": z, "p_value": p_value}
    # Vérification : scipy.stats.chi2_contingency([[partis, restes], [partis, restes]],
    # correction=False) donne exactement la même p-value (le χ² à 1 degré de liberté vaut z²).


# =========================================================================== Survie


def courbe_survie(df: pd.DataFrame) -> pd.DataFrame:
    duree = df["tenure_months"].to_numpy()
    parti = df["churn"].to_numpy() == 1

    lignes, survie = [], 1.0
    for mois in sorted(set(duree[parti].tolist())):  # seulement les mois où quelqu'un part
        a_risque = int((duree >= mois).sum())   # encore clients au DÉBUT du mois (censurés compris)
        departs = int(((duree == mois) & parti).sum())
        survie *= 1 - departs / a_risque         # probabilité de passer ce mois, sachant qu'on y est arrivé
        lignes.append({"mois": int(mois), "a_risque": a_risque, "departs": departs, "survie": survie})
    return pd.DataFrame(lignes, columns=["mois", "a_risque", "departs", "survie"])
    # Le cœur de Kaplan-Meier : un client encore présent (churn = 0) après 8 mois nous apprend
    # qu'il a « survécu » 8 mois, et rien après. Il compte dans a_risque jusqu'au mois 8 puis
    # sort du calcul, sans être compté comme un départ.
    # Bibliothèque de référence : lifelines.KaplanMeierFitter().fit(durees, evenements).

"""Série 02 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_02.py`.

Si votre version passe les vérifications, elle est juste. Comparez-la à celle-ci pour
découvrir d'autres écritures, puis notez dans JOURNAL.md ce que vous avez appris.
"""

from __future__ import annotations

import pandas as pd

TARGET = "churn"

# =========================================================================== Module 1 — Données


def lift(df: pd.DataFrame, colonne: str, modalite) -> float:
    groupe = df[df[colonne] == modalite]
    if groupe.empty:
        # Sans ce garde-fou : groupe[TARGET].mean() vaut NaN, et NaN se propage partout
        # (tableaux, graphiques…) sans jamais lever d'erreur.
        raise ValueError(f"Modalité {modalite!r} absente de la colonne {colonne!r}")
    # On divise par le taux GLOBAL (tous les clients), pas par le taux « des autres ».
    return float(groupe[TARGET].mean() / df[TARGET].mean())


def factures_hors_bornes(df: pd.DataFrame, minimum: float = 0.0, maximum: float = 500.0) -> pd.DataFrame:
    charges = df["monthly_charges"]
    # Toute comparaison avec NaN vaut False : les manquants ne sont donc pas signalés ici.
    # Piège : `~charges.between(minimum, maximum)` les compterait, car between(NaN) vaut False.
    hors = (charges < minimum) | (charges > maximum)
    return df[hors]  # un filtre crée un nouveau DataFrame : `df` n'est pas modifié


def churn_par_tranche_anciennete(
    df: pd.DataFrame, bornes: tuple[int, ...] = (0, 6, 12, 24, 48, 72)
) -> pd.DataFrame:
    etiquettes = [f"{a + 1}-{b}" for a, b in zip(bornes[:-1], bornes[1:])]
    # pd.cut utilise des intervalles ]a, b] (right=True) : 6 tombe dans ]0, 6], 7 dans ]6, 12].
    tranche = pd.cut(df["tenure_months"], bins=list(bornes), labels=etiquettes)
    resume = (
        df.groupby(tranche, observed=True)[TARGET]
        .agg(clients="size", taux_churn="mean")
        .reset_index()
        .rename(columns={"tenure_months": "tranche"})
    )
    resume["tranche"] = resume["tranche"].astype(str)  # une catégorie pandas → du texte simple
    return resume[["tranche", "clients", "taux_churn"]]
    # À retenir : sur les vraies données, les clients de moins de 6 mois partent ~7 fois plus
    # que ceux de plus de 4 ans. Les premiers mois sont le moment où agir.


# ============================================================================== 🔴 Défi final


def segments_a_risque(
    df: pd.DataFrame, colonnes: list[str], min_clients: int = 50, top: int = 5
) -> pd.DataFrame:
    segments = df.groupby(colonnes)[TARGET].agg(clients="size", taux_churn="mean").reset_index()
    # Filtrer AVANT de trier : sinon un segment de 3 clients à 100 % prendrait la tête.
    segments = segments[segments["clients"] >= min_clients]
    return (
        segments.sort_values("taux_churn", ascending=False)
        .head(top)
        .reset_index(drop=True)[[*colonnes, "clients", "taux_churn"]]
    )
    # Pour aller plus loin : un intervalle de confiance du taux (ex. de Wilson) est plus
    # rigoureux qu'un seuil fixe de `min_clients`.

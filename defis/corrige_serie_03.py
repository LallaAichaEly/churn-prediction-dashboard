"""Série 03 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_03.py`.

Si votre version passe les vérifications, elle est juste. Comparez-la à celle-ci pour
découvrir d'autres écritures, puis notez dans JOURNAL.md ce que vous avez appris.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ======================================================================= Module 4 — Évaluation


def precision_rappel_par_seuil(y_true, proba, seuils) -> pd.DataFrame:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    lignes = []
    for seuil in seuils:
        cible = proba >= seuil                 # >= : même règle que churn.model.evaluate
        vp = int((cible & (y_true == 1)).sum())
        n_cibles = int(cible.sum())
        lignes.append({
            "seuil": float(seuil),
            "clients_cibles": n_cibles,
            # Personne ciblé : précision non définie. NaN laisse un trou dans la courbe au
            # lieu d'y dessiner un faux 0.
            "precision": vp / n_cibles if n_cibles else np.nan,
            "rappel": vp / int((y_true == 1).sum()),
        })
    return pd.DataFrame(lignes, columns=["seuil", "clients_cibles", "precision", "rappel"])


def table_calibration(y_true, proba, n_tranches: int = 10) -> pd.DataFrame:
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    # floor(proba × n) donne le numéro de tranche ; une proba de 1 donnerait n (tranche
    # inexistante) : on la range dans la dernière, n − 1.
    tranche = np.minimum((proba * n_tranches).astype(int), n_tranches - 1)
    table = (
        pd.DataFrame({"tranche": tranche, "proba": proba, "churn": y_true})
        # groupby sur une colonne d'entiers ne crée QUE les groupes présents : pas de tranche
        # vide. (Avec pd.cut, qui produit une catégorie, il faudrait observed=True.)
        .groupby("tranche")
        .agg(clients=("churn", "size"), proba_moyenne=("proba", "mean"), taux_reel=("churn", "mean"))
        .reset_index()
    )
    return table


def auc_par_paires(y_true, proba) -> float:
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    partis, restes = proba[y_true == 1], proba[y_true == 0]
    if len(partis) == 0 or len(restes) == 0:
        raise ValueError("Il faut au moins un client parti et un client resté pour calculer une AUC")
    # Matrice (nb partis × nb restés) : chaque case est une paire.
    gagnees = (partis[:, None] > restes[None, :]).sum()
    ex_aequo = (partis[:, None] == restes[None, :]).sum()
    return float((gagnees + 0.5 * ex_aequo) / (len(partis) * len(restes)))
    # Pour aller plus loin : cette formule coûte nb_partis × nb_restés comparaisons. Sur des
    # millions de clients, on trie les scores une fois et on utilise les rangs
    # (statistique de Mann-Whitney), comme le fait roc_auc_score.


# ==================================================================== Module 5 — Interprétation


def contributions_client(pipeline, client: pd.DataFrame, top: int | None = 3) -> pd.DataFrame:
    # pipeline[:-1] applique TOUTES les étapes avant le modèle, y compris les variables
    # dérivées si elles sont activées. named_steps["prep"].transform seul échouerait alors :
    # il attend des colonnes (avg_charge_per_month…) que seule l'étape précédente crée.
    valeurs = pipeline[:-1].transform(client)
    valeurs = valeurs.toarray() if hasattr(valeurs, "toarray") else np.asarray(valeurs)
    # Les noms, eux, viennent de "prep" : l'étape de variables dérivées ne sait pas nommer
    # ses sorties, donc pipeline[:-1].get_feature_names_out() lèverait une erreur.
    noms = pipeline.named_steps["prep"].get_feature_names_out()
    contributions = pd.Series(valeurs[0] * pipeline[-1].coef_[0], index=noms)
    contributions = contributions[contributions != 0].sort_values(key=abs, ascending=False)
    if top is not None:
        contributions = contributions.head(top)
    # Vérification possible : intercept + somme de TOUTES les contributions
    # = pipeline.decision_function(client), le logit de la proba prédite.
    return contributions.rename_axis("variable").reset_index(name="contribution")


# ===================================================================== Module 7 — Surveillance


def psi_parts(parts_reference, parts_actuelles) -> float:
    ref = np.asarray(parts_reference, dtype=float)
    act = np.asarray(parts_actuelles, dtype=float)
    if ref.shape != act.shape:
        raise ValueError(f"Autant de tranches de chaque côté : {len(ref)} contre {len(act)}")
    ref, act = ref / ref.sum(), act / act.sum()            # effectifs → parts
    ref, act = np.clip(ref, 1e-4, None), np.clip(act, 1e-4, None)  # pas de ln(0)
    # Chaque terme est positif ou nul : (a − r) et ln(a / r) ont toujours le même signe.
    return float(np.sum((act - ref) * np.log(act / ref)))

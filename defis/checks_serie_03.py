"""Vérifications automatiques de la série 03 (évaluer, expliquer, surveiller).

Lire ces tests est aussi un bon exercice : ils montrent les cas limites auxquels penser.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from .checks_serie_01 import _proche
from .checks_serie_02 import _dataframe
from .verif import Defi, Serie


@lru_cache(maxsize=1)
def _modele_cache():
    from churn.data import generate_customers
    from churn.model import train

    pipeline, X_test, y_test, proba, _ = train(generate_customers(n=2000, seed=11))
    return pipeline, X_test, y_test, proba


def _modele():
    """(pipeline, X_test, y_test, proba) d'une régression logistique. Copies : votre code ne
    peut pas abîmer les données des vérifications suivantes."""
    pipeline, X_test, y_test, proba = _modele_cache()
    return pipeline, X_test.copy(), y_test.copy(), proba.copy()


@lru_cache(maxsize=2)
def _pipelines_entraines(use_engineered: bool):
    from churn.data import FEATURES, TARGET, generate_customers
    from churn.model import build_pipeline

    df = generate_customers(n=2000, seed=11)
    client = df.loc[[df[df["contract"] == "Biannuel"]["tenure_months"].idxmax()], FEATURES]
    return build_pipeline("logistic", use_engineered=use_engineered).fit(df[FEATURES], df[TARGET]), client


def _colonnes(obtenu: pd.DataFrame, attendues: list[str], quoi: str) -> None:
    assert list(obtenu.columns) == attendues, f"{quoi} : colonnes attendues {attendues}, obtenu {list(obtenu.columns)}"


# ======================================================================= Module 4 — Évaluation


def verifier_precision_rappel_par_seuil(f) -> None:
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.6, 0.3, 0.6, 0.4, 0.2, 0.1, 0.05]
    seuils = [0.5, 0.6, 0.95, 0.1]  # volontairement dans le désordre
    res = _dataframe(f(y, p, seuils), "precision_rappel_par_seuil")
    _colonnes(res, ["seuil", "clients_cibles", "precision", "rappel"], "precision_rappel_par_seuil")
    assert len(res) == 4, f"4 seuils demandés : attendu 4 lignes, obtenu {len(res)}"
    assert [float(s) for s in res["seuil"]] == seuils, (
        f"Gardez les seuils dans l'ordre demandé {seuils}, obtenu {list(res['seuil'])}"
    )
    lignes = {float(r.seuil): r for r in res.itertuples()}
    if int(lignes[0.6].clients_cibles) == 1:
        raise AssertionError(
            "Au seuil 0,6, deux clients ont une proba de 0,6 pile : ils sont ciblés (proba >= seuil, "
            "pas >). C'est la règle de churn.model.evaluate et du défi 8."
        )
    attendus = {0.5: (3, 2 / 3, 2 / 3), 0.6: (3, 2 / 3, 2 / 3), 0.1: (7, 3 / 7, 1.0)}
    for seuil, (cibles, precision, rappel) in attendus.items():
        assert int(lignes[seuil].clients_cibles) == cibles, (
            f"Seuil {seuil} : {cibles} clients ciblés attendus, obtenu {lignes[seuil].clients_cibles}"
        )
        _proche(lignes[seuil].precision, precision, f"précision au seuil {seuil}")
        _proche(lignes[seuil].rappel, rappel, f"rappel au seuil {seuil}")
    personne = lignes[0.95]
    assert int(personne.clients_cibles) == 0, "Au seuil 0,95, personne n'est ciblé"
    if not pd.isna(personne.precision):
        raise AssertionError(
            f"Au seuil 0,95 personne n'est ciblé : la précision doit valoir NaN, obtenu {personne.precision}. "
            "Un 0 inventé ferait plonger la courbe à droite alors que le modèle ne s'est pas trompé."
        )
    _proche(personne.rappel, 0.0, "rappel au seuil 0,95 (aucun client trouvé)")

    from churn.model import evaluate

    _, _, y_test, proba = _modele()  # y_test : une pd.Series dont l'index n'est pas 0, 1, 2…
    grille = [round(s, 2) for s in np.linspace(0.05, 0.95, 19)]
    courbe = _dataframe(f(y_test, proba, grille), "precision_rappel_par_seuil sur un vrai modèle")
    assert courbe["rappel"].is_monotonic_decreasing, (
        "Le rappel ne peut que baisser quand le seuil monte (on cible de moins en moins de clients)"
    )
    for ligne in courbe.itertuples():
        m = evaluate(y_test, proba, ligne.seuil)
        assert int(ligne.clients_cibles) == m.tp + m.fp, (
            f"Seuil {ligne.seuil} : {m.tp + m.fp} clients ciblés selon churn.model.evaluate, obtenu {ligne.clients_cibles}"
        )
        _proche(ligne.rappel, m.recall, f"rappel au seuil {ligne.seuil} (vrai modèle)")
        if m.tp + m.fp:
            _proche(ligne.precision, m.precision, f"précision au seuil {ligne.seuil} (vrai modèle)")


def verifier_table_calibration(f) -> None:
    y = [0, 0, 1, 1, 0, 1, 1, 0]
    p = [0.0, 0.1, 0.5, 0.55, 0.7, 1.0, 0.9, 0.2]
    res = _dataframe(f(y, p, n_tranches=4), "table_calibration(n_tranches=4)")
    _colonnes(res, ["tranche", "clients", "proba_moyenne", "taux_reel"], "table_calibration")
    tranches = [int(t) for t in res["tranche"]]
    if 4 in tranches:
        raise AssertionError(
            "Une proba de 1 tombe dans la tranche 4, qui n'existe pas avec 4 tranches (0 à 3). "
            "La dernière tranche inclut 1 : np.minimum(numéro, n_tranches - 1)."
        )
    total = int(res["clients"].sum())
    if total != len(y):
        raise AssertionError(
            f"{total} clients répartis sur {len(y)} : chaque client doit être dans une tranche. "
            "Une proba de 0 ou de 1 exactement est-elle bien comptée ? (pd.cut exclut une des bornes.)"
        )
    if (res["clients"] == 0).any() or res["taux_reel"].isna().any():
        raise AssertionError(
            "Ne gardez que les tranches NON VIDES : aucun client n'a une proba entre 0,25 et 0,5. "
            "(Avec pd.cut, groupby(..., observed=True) supprime les catégories vides.)"
        )
    if tranches == [0, 1, 2, 3]:
        raise AssertionError(
            "Une proba de 0,5 pile appartient à la tranche [0,5 ; 0,75[, numéro 2, pas à la tranche 1 : "
            "les tranches incluent leur borne GAUCHE. (pd.cut : right=False.)"
        )
    assert tranches == [0, 2, 3], f"Tranches non vides attendues [0, 2, 3], obtenu {tranches}"
    assert list(res.index) == [0, 1, 2], "L'index doit être 0, 1, 2… : pensez à reset_index()"
    assert [int(c) for c in res["clients"]] == [3, 3, 2], f"Effectifs attendus [3, 3, 2], obtenu {list(res['clients'])}"
    if np.allclose(res["proba_moyenne"], [0.125, 0.625, 0.875]):
        raise AssertionError(
            "proba_moyenne vaut le MILIEU de chaque tranche. Il faut la moyenne des probas réellement "
            "prédites pour les clients de la tranche : ce sont elles qu'on compare au taux réel."
        )
    for obtenu, attendu, t in zip(res["proba_moyenne"], [0.1, (0.5 + 0.55 + 0.7) / 3, 0.95], tranches):
        _proche(obtenu, attendu, f"proba_moyenne de la tranche {t}")
    for obtenu, attendu, t in zip(res["taux_reel"], [0.0, 2 / 3, 1.0], tranches):
        _proche(obtenu, attendu, f"taux_reel de la tranche {t}")

    _, _, y_test, proba = _modele()
    vraie = _dataframe(f(y_test, proba), "table_calibration sur un vrai modèle (10 tranches par défaut)")
    numeros = np.minimum((proba * 10).astype(int), 9)
    attendue = (pd.DataFrame({"t": numeros, "p": proba, "y": np.asarray(y_test)})
                .groupby("t").agg(clients=("y", "size"), p=("p", "mean"), y=("y", "mean")))
    assert [int(t) for t in vraie["tranche"]] == list(attendue.index), (
        f"Tranches attendues {list(attendue.index)}, obtenu {list(vraie['tranche'])}"
    )
    assert [int(c) for c in vraie["clients"]] == list(attendue["clients"]), "Effectifs incorrects sur le vrai modèle"
    assert np.allclose(vraie["proba_moyenne"], attendue["p"]) and np.allclose(vraie["taux_reel"], attendue["y"]), (
        "proba_moyenne ou taux_reel incorrects sur le vrai modèle"
    )


def verifier_auc_par_paires(f) -> None:
    obtenu = f([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert obtenu is not None, "auc_par_paires : la fonction a retourné None. Avez-vous oublié `return` ?"
    if math.isclose(float(obtenu), 0.25):
        raise AssertionError(
            "Obtenu 0,25 au lieu de 0,75 : vous comptez les paires où le client RESTÉ a le score le "
            "plus haut. L'AUC compte celles où le client PARTI est mieux classé."
        )
    _proche(obtenu, 0.75, "AUC de y=[0, 0, 1, 1], proba=[0.1, 0.4, 0.35, 0.8] (3 paires gagnées sur 4)")

    obtenu = float(f([0, 1, 0, 1], [0.5, 0.5, 0.2, 0.9]))
    if math.isclose(obtenu, 0.75) or math.isclose(obtenu, 1.0):
        raise AssertionError(
            f"Avec une paire ex aequo (0,5 contre 0,5), attendu 0,875, obtenu {obtenu}. "
            "Une paire ex aequo compte pour ½ : le modèle n'a ni raison ni tort."
        )
    _proche(obtenu, 0.875, "AUC avec une paire ex aequo")

    from sklearn.metrics import roc_auc_score

    _, _, y_test, proba = _modele()
    arrondi = np.round(proba, 1)  # beaucoup d'ex aequo
    _proche(f(y_test, proba), roc_auc_score(y_test, proba), "AUC du vrai modèle (comparée à roc_auc_score)", tol=1e-9)
    _proche(f(list(y_test), list(arrondi)), roc_auc_score(y_test, arrondi),
            "AUC avec des probas arrondies à 0,1 près (beaucoup d'ex aequo)", tol=1e-9)

    try:
        resultat = f([1, 1, 1], [0.2, 0.5, 0.9])
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"Aucun client resté : attendu une ValueError, obtenu {type(exc).__name__}. "
            "Testez vous-même s'il y a des deux classes et levez `raise ValueError(...)`."
        ) from None
    raise AssertionError(
        f"Aucun client resté, donc aucune paire : la fonction a retourné {resultat!r} au lieu de lever une ValueError"
    )


# ==================================================================== Module 5 — Interprétation


def _contributions_attendues(pipeline, client) -> pd.Series:
    valeurs = pipeline[:-1].transform(client)
    valeurs = valeurs.toarray() if hasattr(valeurs, "toarray") else np.asarray(valeurs)
    noms = pipeline.named_steps["prep"].get_feature_names_out()
    contrib = pd.Series(valeurs[0] * pipeline[-1].coef_[0], index=noms)
    return contrib[contrib != 0]


def verifier_contributions_client(f) -> None:
    pipeline, client = _pipelines_entraines(False)  # contrat biannuel, 72 mois d'ancienneté
    attendu = _contributions_attendues(pipeline, client)
    logit = float(pipeline.decision_function(client)[0])
    intercept = float(pipeline[-1].intercept_[0])

    toutes = _dataframe(f(pipeline, client, top=None), "contributions_client(top=None)")
    _colonnes(toutes, ["variable", "contribution"], "contributions_client")
    variables = [str(v) for v in toutes["variable"]]
    assert all(v.startswith(("num__", "cat__")) for v in variables), (
        f"Noms inattendus {variables[:3]}… : utilisez pipeline.named_steps['prep'].get_feature_names_out()"
    )
    if len(toutes) == 3 < len(attendu):
        raise AssertionError(
            f"top=None doit renvoyer TOUTES les contributions non nulles ({len(attendu)}), pas seulement 3"
        )
    somme = intercept + float(toutes["contribution"].sum())
    assert math.isclose(somme, logit, abs_tol=1e-6), (
        f"intercept + Σ contributions = {somme:.3f}, mais le modèle calcule un logit de {logit:.3f} "
        "(pipeline.decision_function(client)). Multipliez les coefficients par les valeurs TRANSFORMÉES "
        "(pipeline[:-1].transform) : le modèle ne voit jamais les valeurs brutes."
    )
    assert (toutes["contribution"] != 0).all(), (
        "Retirez les contributions nulles : une modalité one-hot absente chez ce client (valeur 0) n'explique rien"
    )
    assert len(toutes) == len(attendu), f"top=None : {len(attendu)} contributions non nulles attendues, obtenu {len(toutes)}"

    top = _dataframe(f(pipeline, client), "contributions_client (top=3 par défaut)")
    ordre_attendu = list(attendu.sort_values(key=abs, ascending=False).index[:3])
    ordre_signe = list(attendu.sort_values(ascending=False).index[:3])
    assert ordre_attendu != ordre_signe  # garde-fou du correcteur : le client choisi distingue les deux tris
    obtenu = [str(v) for v in top["variable"]]
    if obtenu == ordre_signe:
        raise AssertionError(
            "Vous triez par contribution signée : les variables qui RETIENNENT ce client (contributions "
            "négatives, ex. 72 mois d'ancienneté) disparaissent. Triez par valeur absolue : "
            "sort_values(key=abs, ascending=False)."
        )
    assert obtenu == ordre_attendu, f"Top 3 attendu {ordre_attendu}, obtenu {obtenu}"
    assert list(top.index) == [0, 1, 2], "L'index doit être 0, 1, 2 : pensez à reset_index()"
    for v, c in zip(obtenu, top["contribution"]):
        _proche(c, attendu[v], f"contribution de {v}")
    assert len(_dataframe(f(pipeline, client, top=5), "contributions_client(top=5)")) == 5, "Le paramètre `top` est ignoré"

    avec_derivees, client = _pipelines_entraines(True)
    try:
        res = _dataframe(f(avec_derivees, client, top=None), "contributions_client avec variables dérivées")
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(
            f"Avec les variables dérivées (build_pipeline(..., use_engineered=True)), votre fonction plante : "
            f"{type(exc).__name__}: {exc}. Transformez avec pipeline[:-1] (toutes les étapes sauf le modèle) "
            "et prenez les noms dans pipeline.named_steps['prep']."
        ) from None
    assert "num__calls_per_year" in set(res["variable"]), (
        "Avec les variables dérivées, num__calls_per_year devrait apparaître dans les contributions"
    )
    logit = float(avec_derivees.decision_function(client)[0])
    somme = float(avec_derivees[-1].intercept_[0]) + float(res["contribution"].sum())
    assert math.isclose(somme, logit, abs_tol=1e-6), (
        f"Avec les variables dérivées, intercept + Σ contributions = {somme:.3f} au lieu de {logit:.3f}"
    )


# ===================================================================== Module 7 — Surveillance


def verifier_psi_parts(f) -> None:
    obtenu = f([0.5, 0.5], [0.7, 0.3])
    assert obtenu is not None, "psi_parts : la fonction a retourné None. Avez-vous oublié `return` ?"
    attendu = 0.2 * math.log(0.7 / 0.5) + (-0.2) * math.log(0.3 / 0.5)
    if math.isclose(float(obtenu), attendu / math.log(10), rel_tol=1e-6):
        raise AssertionError("ln est le logarithme NATUREL : np.log, pas np.log10")
    _proche(obtenu, attendu, "PSI de 50 % / 50 % vers 70 % / 30 % (exercice 7.2)")
    _proche(f([50, 50], [70, 30]), attendu,
            "PSI calculé à partir d'effectifs [50, 50] → [70, 30] (divisez chaque liste par sa somme)")
    _proche(f([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0.0, "PSI de deux répartitions identiques")

    obtenu = float(f([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))
    if not math.isfinite(obtenu):
        raise AssertionError(
            f"Une tranche vide dans la référence donne un PSI de {obtenu} : ln(0,2 / 0) est infini. "
            "Remplacez les parts nulles par 0,0001 : np.clip(parts, 1e-4, None)."
        )
    attendu = 2 * (-0.1 * math.log(0.8)) + (0.2 - 1e-4) * math.log(0.2 / 1e-4)
    _proche(obtenu, attendu, "PSI avec une tranche vide dans la référence (part remplacée par 0,0001)", tol=1e-3)

    from churn.data import generate_customers
    from churn.monitoring import psi, simulate_batch

    reference = generate_customers(n=2000, seed=5)["contract"]
    actuel = simulate_batch(n=2000, seed=6, monthly_share=0.75)["contract"]
    modalites = sorted(set(reference) | set(actuel))
    _proche(f(reference.value_counts().reindex(modalites, fill_value=0),
              actuel.value_counts().reindex(modalites, fill_value=0)),
            psi(reference, actuel), "PSI du contrat quand les mensuels passent à 75 % (comparé à churn.monitoring.psi)")

    try:
        resultat = f([0.5, 0.5], [0.2, 0.3, 0.5])
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"Listes de longueurs différentes : attendu une ValueError, obtenu {type(exc).__name__}"
        ) from None
    raise AssertionError(
        f"Listes de longueurs différentes (2 et 3 tranches) : la fonction a retourné {resultat!r} au lieu de "
        "lever une ValueError. Comparer des tranches qui ne se correspondent pas n'a pas de sens."
    )


SERIE = Serie(
    code="03",
    titre="Évaluer, expliquer, surveiller",
    module_eleve="defis.serie_03",
    module_corrige="defis.corrige_serie_03",
    defis=(
        Defi(15, "precision_rappel_par_seuil", "🟡", "4.4", verifier_precision_rappel_par_seuil),
        Defi(16, "table_calibration", "🟡", "4.5", verifier_table_calibration),
        Defi(17, "auc_par_paires", "🔴", "—", verifier_auc_par_paires),
        Defi(18, "contributions_client", "🔴", "5.3", verifier_contributions_client),
        Defi(19, "psi_parts", "🟢", "7.2", verifier_psi_parts),
    ),
)

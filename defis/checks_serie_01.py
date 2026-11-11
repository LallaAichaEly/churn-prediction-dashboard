"""Vérifications automatiques de la série 01.

Lire ces tests est aussi un bon exercice : ils montrent les cas limites auxquels penser.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .verif import Defi, Serie


def _proche(obtenu, attendu, quoi: str, tol: float = 1e-6) -> None:
    assert obtenu is not None, f"{quoi} : la fonction a retourné None. Avez-vous oublié `return` ?"
    assert math.isclose(float(obtenu), float(attendu), rel_tol=tol, abs_tol=tol), (
        f"{quoi} : attendu {attendu:.4f}, obtenu {float(obtenu):.4f}"
    )


def _series(obtenu, quoi: str) -> pd.Series:
    assert obtenu is not None, f"{quoi} : la fonction a retourné None. Avez-vous oublié `return` ?"
    assert isinstance(obtenu, pd.Series), f"{quoi} : attendu une pd.Series, obtenu {type(obtenu).__name__}"
    return obtenu


def _clients(n: int = 600, seed: int = 0) -> pd.DataFrame:
    from churn.data import generate_customers

    return generate_customers(n=n, seed=seed)


# =========================================================================== Module 1 — Données


def verifier_taux_churn_par(f) -> None:
    df = pd.DataFrame({"contract": list("AAABBCC"), "churn": [1, 1, 0, 0, 0, 1, 0]})
    s = _series(f(df, "contract"), "taux_churn_par")
    assert set(s.index) == {"A", "B", "C"}, f"L'index doit contenir les modalités A, B, C — obtenu {list(s.index)}"
    assert list(s.index) == ["A", "C", "B"], (
        f"Mauvais ordre : attendu A (67 %), C (50 %), B (0 %) — obtenu {list(s.index)}. "
        "Triez du taux le plus élevé au plus faible."
    )
    for cle, attendu in {"A": 2 / 3, "C": 0.5, "B": 0.0}.items():
        _proche(s[cle], attendu, f"taux de churn de {cle}")
    vrai = _series(f(_clients(), "internet_service"), "taux_churn_par sur les vraies données")
    assert vrai.is_monotonic_decreasing, "Sur les données clients, le résultat n'est pas trié décroissant"


def verifier_taux_manquants(f) -> None:
    df = pd.DataFrame(
        {"a": [1, np.nan, 3, np.nan], "b": [1, 2, 3, 4], "c": ["x", None, "y", "z"]}
    )
    s = _series(f(df), "taux_manquants")
    assert "b" not in s.index, "La colonne 'b' n'a aucune valeur manquante : elle ne doit pas apparaître"
    assert list(s.index) == ["a", "c"], f"Attendu les colonnes ['a', 'c'] dans cet ordre, obtenu {list(s.index)}"
    _proche(s["a"], 0.5, "taux de manquants de 'a'")
    _proche(s["c"], 0.25, "taux de manquants de 'c' (None compte comme manquant)")
    vide = _series(f(df[["b"]]), "taux_manquants sans aucun manquant")
    assert len(vide) == 0, "Quand rien ne manque, retournez une Series vide"


def verifier_valeurs_aberrantes(f) -> None:
    serie = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], index=[f"C{i}" for i in range(10)])
    masque = _series(f(serie), "valeurs_aberrantes")
    assert masque.dtype == bool, f"Attendu des booléens (True/False), obtenu le type {masque.dtype}"
    assert masque.index.equals(serie.index), "Le masque doit avoir le même index que la série d'entrée"
    assert list(masque[masque].index) == ["C9"], (
        f"Seule la valeur 100 (C9) est aberrante — obtenu {list(masque[masque].index)}. "
        "Rappel : Q1 = 3,25, Q3 = 7,75, IQR = 4,5, borne haute = 14,5"
    )
    basse = pd.Series([-50, 10, 11, 12, 13, 14, 15])
    assert bool(f(basse).iloc[0]), "Les valeurs trop BASSES sont aussi aberrantes (-50 ici)"
    assert not f(serie, k=100).any(), "Avec k=100, aucune valeur ne devrait être aberrante : utilisez-vous k ?"


# ========================================================================= Module 2 — Variables


def verifier_ajouter_nouveau_client(f) -> None:
    df = pd.DataFrame({"tenure_months": [1, 5, 6, 12]})
    avant = df.copy()
    out = f(df)
    assert out is not None, "La fonction a retourné None. Avez-vous oublié `return` ?"
    assert "is_new_customer" not in df.columns, "Vous avez modifié `df` en place : travaillez sur df.copy()"
    pd.testing.assert_frame_equal(df, avant, obj="df d'entrée (il ne doit pas changer)")
    assert "is_new_customer" in out.columns, "La colonne `is_new_customer` est absente du résultat"
    assert out["is_new_customer"].tolist() == [1, 1, 0, 0], (
        f"Attendu [1, 1, 0, 0] pour des anciennetés [1, 5, 6, 12] (6 n'est PAS < 6) — "
        f"obtenu {out['is_new_customer'].tolist()}"
    )
    assert f(df, seuil_mois=12)["is_new_customer"].tolist() == [1, 1, 1, 0], "Le paramètre `seuil_mois` est ignoré"


def verifier_standardiser(f) -> None:
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    test = pd.Series([3.0, 7.0])
    res = f(train, test)
    assert res is not None, "La fonction a retourné None. Avez-vous oublié `return` ?"
    assert len(res) == 2, "Retournez un tuple (train_standardisé, test_standardisé)"
    z_train, z_test = (np.asarray(x, dtype=float) for x in res)
    assert math.isclose(z_train.mean(), 0.0, abs_tol=1e-6), (
        f"La moyenne du train standardisé vaut {z_train.mean():.3f} au lieu de 0. "
        "Calculez-vous la moyenne sur le TRAIN uniquement ? Utiliser train + test, "
        "c'est une fuite de données."
    )
    if math.isclose(z_train.std(ddof=1), 1.0, abs_tol=1e-6):
        raise AssertionError("Écart-type calculé avec ddof=1 (défaut de pandas) : utilisez .std(ddof=0)")
    _proche(z_train.std(), 1.0, "écart-type du train standardisé")
    attendu = np.array([0.0, 4 / math.sqrt(2)])
    if not np.allclose(z_test, attendu):
        raise AssertionError(
            f"Test standardisé faux : attendu {attendu.round(3).tolist()}, obtenu {z_test.round(3).tolist()}. "
            "Le test doit être transformé avec la moyenne et l'écart-type DU TRAIN, "
            "pas avec les siens."
        )


# ======================================================================= Module 4 — Évaluation


def verifier_precision_rappel(f) -> None:
    def verifier(y_true, y_pred, attendu, cas):
        res = f(y_true, y_pred)
        assert res is not None, "La fonction a retourné None. Avez-vous oublié `return` ?"
        p, r = res
        _proche(p, attendu[0], f"précision ({cas})")
        _proche(r, attendu[1], f"rappel ({cas})")

    verifier([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], (2 / 3, 2 / 3), "cas simple")
    verifier([1, 0, 1, 1], [1, 0, 0, 0], (1.0, 1 / 3), "précision ≠ rappel — les avez-vous inversés ?")
    verifier([1, 0, 1], [0, 0, 0], (0.0, 0.0), "aucun positif prédit → précision 0, pas de division par zéro")
    verifier(np.array([0, 1]), np.array([0, 1]), (1.0, 1.0), "tableaux numpy")


def verifier_accuracy_modele_naif(f) -> None:
    _proche(f([0, 0, 0, 1]), 0.75, "accuracy naïve avec 1 churn sur 4")
    _proche(f([1, 1, 0]), 2 / 3, "accuracy naïve quand les 1 sont majoritaires")
    y = _clients()["churn"]
    _proche(f(y), 1 - y.mean(), "accuracy naïve sur les données clients")


def verifier_cout_total(f) -> None:
    y = [1, 0, 1, 0]
    proba = [0.9, 0.8, 0.2, 0.1]
    _proche(f(y, proba, 0.5, 500, 50), 550, "coût au seuil 0,5 (1 client raté + 1 action inutile)")
    _proche(f(y, proba, 0.95, 500, 50), 1000, "coût au seuil 0,95 (personne n'est ciblé)")
    _proche(f(y, proba, 0.0, 500, 50), 100, "coût au seuil 0 (tout le monde est ciblé)")
    _proche(f(y, proba, 0.8, 500, 50), 550, "coût au seuil 0,8 (proba >= seuil → ciblé, pensez au « = »)")


# ===================================================================== Module 3 — Modélisation


def verifier_validation_croisee_auc(f) -> None:
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from churn.data import FEATURES, TARGET
    from churn.model import build_pipeline

    df = _clients()
    for k in (5, 3):
        res = f(df, "logistic", k=k, seed=0)
        assert res is not None, "La fonction a retourné None. Avez-vous oublié `return` ?"
        moyenne, ecart = res
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        scores = cross_val_score(build_pipeline("logistic", seed=0), df[FEATURES], df[TARGET], cv=cv, scoring="roc_auc")
        assert 0.5 < float(moyenne) <= 1, f"Une ROC-AUC moyenne de {float(moyenne):.3f} n'est pas plausible"
        if not math.isclose(float(moyenne), scores.mean(), abs_tol=1e-6):
            raise AssertionError(
                f"AUC moyenne avec k={k} : attendu {scores.mean():.4f}, obtenu {float(moyenne):.4f}. "
                "Utilisez-vous StratifiedKFold(n_splits=k, shuffle=True, random_state=seed) ?"
            )
        _proche(ecart, scores.std(), f"écart-type des AUC avec k={k} (np.std)")


def verifier_lift_top(f) -> None:
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]                   # 20 % de churn
    proba = [0.95, 0.9, 0.85, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
    _proche(f(y, proba, 0.1), 5.0, "lift du top 10 % (1 client, qui part : 100 % / 20 %)")
    _proche(f(y, proba, 0.3), (2 / 3) / 0.2, "lift du top 30 % (2 départs sur 3 clients)")
    _proche(f(y, proba, 1.0), 1.0, "lift sur 100 % des clients (doit valoir 1)")
    _proche(f(y, proba, 0.01), 5.0, "lift avec part=1 % : on garde au moins 1 client")
    melange = [0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05]
    _proche(f(y, melange, 0.1), 0.0, "lift quand le client le plus à risque reste (triez-vous par proba ?)")


SERIE = Serie(
    code="01",
    titre="Les fondamentaux",
    module_eleve="defis.serie_01",
    module_corrige="defis.corrige_serie_01",
    defis=(
        Defi(1, "taux_churn_par", "🟢", "1.2", verifier_taux_churn_par),
        Defi(2, "taux_manquants", "🟢", "—", verifier_taux_manquants),
        Defi(3, "valeurs_aberrantes", "🟡", "—", verifier_valeurs_aberrantes),
        Defi(4, "ajouter_nouveau_client", "🟢", "2.3", verifier_ajouter_nouveau_client),
        Defi(5, "standardiser", "🟡", "2.1 · 2.5", verifier_standardiser),
        Defi(6, "precision_rappel", "🟢", "—", verifier_precision_rappel),
        Defi(7, "accuracy_modele_naif", "🟢", "4.2", verifier_accuracy_modele_naif),
        Defi(8, "cout_total", "🟡", "4.3", verifier_cout_total),
        Defi(9, "validation_croisee_auc", "🔴", "3.4", verifier_validation_croisee_auc),
        Defi(10, "lift_top", "🔴", "—", verifier_lift_top),
    ),
)

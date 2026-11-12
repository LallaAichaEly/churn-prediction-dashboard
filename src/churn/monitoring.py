"""Module 7 — Surveiller le modèle en production (data drift).

Un modèle est entraîné sur une photo du passé. Si les clients changent (hausse de prix,
nouvelle offre, crise…), ses prédictions se dégradent EN SILENCE : aucune erreur n'est levée.
On surveille donc la distribution des variables avec le PSI (Population Stability Index) :

    PSI = Σ (p_actuel − p_référence) × ln(p_actuel / p_référence)

où p = proportion de clients dans chaque intervalle (numérique) ou modalité (catégorielle).

Règle empirique : < 0,10 stable · 0,10 – 0,25 dérive modérée · > 0,25 dérive significative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, generate_customers

MODERATE = 0.10
SIGNIFICANT = 0.25
_EPS = 1e-4  # évite ln(0) quand une case est vide


def _psi_from_proportions(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(np.asarray(ref, dtype=float), _EPS, None)
    cur = np.clip(np.asarray(cur, dtype=float), _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(reference, current, bins: int = 10) -> float:
    """PSI d'une variable numérique.

    Les intervalles sont les quantiles de la RÉFÉRENCE : ~10 % des anciens clients dans chacun.
    On compte ensuite quelle part des nouveaux clients tombe dans chaque intervalle.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    # Bornes intérieures uniques (une variable discrète peut avoir des quantiles égaux).
    # Case i = ]borne(i-1), borne(i)] : chaque valeur entière d'une variable discrète
    # (ex. appels au support) obtient ainsi sa propre case.
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    n_bins = len(inner) + 1
    ref_counts = np.bincount(np.searchsorted(inner, ref, side="left"), minlength=n_bins)
    cur_counts = np.bincount(np.searchsorted(inner, cur, side="left"), minlength=n_bins)
    return _psi_from_proportions(ref_counts / len(ref), cur_counts / len(cur))


def psi_categorical(reference, current) -> float:
    """PSI d'une variable catégorielle : on compare les parts de chaque modalité."""
    ref = pd.Series(reference).value_counts(normalize=True)
    cur = pd.Series(current).value_counts(normalize=True)
    categories = ref.index.union(cur.index)  # une modalité nouvelle compte aussi
    return _psi_from_proportions(
        ref.reindex(categories, fill_value=0).to_numpy(),
        cur.reindex(categories, fill_value=0).to_numpy(),
    )


def drift_status(psi: float) -> str:
    if psi < MODERATE:
        return "stable"
    if psi < SIGNIFICANT:
        return "modérée"
    return "significative"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    """Tableau feature / type / PSI / statut, trié de la plus forte dérive à la plus faible."""
    rows = [
        {"feature": c, "type": "numérique", "psi": psi_numeric(reference[c], current[c])} for c in numeric
    ] + [
        {"feature": c, "type": "catégorielle", "psi": psi_categorical(reference[c], current[c])}
        for c in categorical
    ]
    report = pd.DataFrame(rows)
    report["statut"] = report["psi"].map(drift_status)
    return report.sort_values("psi", ascending=False).reset_index(drop=True)


def simulate_new_period(
    n: int = 5000,
    seed: int = 7,
    price_increase: float = 0.0,
    extra_support_calls: float = 0.0,
    to_monthly: float = 0.0,
) -> pd.DataFrame:
    """Fabrique des clients d'une « nouvelle période » avec des changements contrôlés.

    - `price_increase`      : hausse des prix (0,2 = +20 %) ;
    - `extra_support_calls` : appels au support supplémentaires en moyenne (loi de Poisson) ;
    - `to_monthly`          : part des contrats longs qui repassent en mensuel.

    La colonne `churn` est retirée : en production, on ne sait pas encore qui va partir.
    """
    if price_increase < -0.9:
        raise ValueError("price_increase doit être > -0,9")
    if not 0 <= to_monthly <= 1:
        raise ValueError("to_monthly doit être entre 0 et 1")

    rng = np.random.default_rng(seed + 1)
    df = generate_customers(n=n, seed=seed).drop(columns=TARGET)
    df["monthly_charges"] = (df["monthly_charges"] * (1 + price_increase)).round(2)
    df["total_charges"] = (df["total_charges"] * (1 + price_increase)).round(2)
    if extra_support_calls > 0:
        df["support_calls"] += rng.poisson(extra_support_calls, size=n)
    if to_monthly > 0:
        switch = (df["contract"] != "Mensuel") & (rng.uniform(size=n) < to_monthly)
        df.loc[switch, "contract"] = "Mensuel"
    return df

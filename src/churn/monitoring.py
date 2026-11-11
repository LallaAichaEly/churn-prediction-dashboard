"""Module 7 — Surveiller le modèle en production (data drift).

Un modèle est entraîné sur une PHOTO du passé. Six mois plus tard, les clients ont changé :
hausse des prix, nouvelle offre, nouveau moyen de paiement… On parle de *dérive* (drift).

Deux familles de dérive, à ne pas confondre :
- **dérive des entrées** (*data / covariate drift*) : la distribution des variables change.
  On la détecte TOUT DE SUITE, sans attendre de savoir qui est parti → indicateur PSI ci-dessous.
- **dérive du concept** (*concept drift*) : le LIEN entre variables et churn change (ex. un
  concurrent cible les clients fibre). Les entrées peuvent rester identiques : seule la
  performance réelle, mesurée quand les vrais départs sont connus, la révèle.

Indicateur utilisé : le PSI (*Population Stability Index*), standard en banque et assurance.
    PSI = Σ (part_actuelle − part_référence) × ln(part_actuelle / part_référence)
Règle empirique : < 0,10 stable · 0,10 à 0,25 à surveiller · > 0,25 dérive significative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES, generate_customers

PSI_WATCH = 0.10
PSI_DRIFT = 0.25
_EPS = 1e-4  # évite ln(0) quand une tranche est vide d'un côté


def _shares(reference: pd.Series, current: pd.Series, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Répartit les deux échantillons dans les MÊMES tranches, définies sur la référence."""
    numeric = pd.api.types.is_numeric_dtype(reference)
    if numeric and reference.nunique() > bins:
        # Variable continue : tranches de même effectif (déciles) calculées sur la référence.
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf  # les valeurs nouvelles tombent dans les tranches extrêmes
        ref_counts = pd.cut(reference, edges).value_counts(sort=False)
        cur_counts = pd.cut(current, edges).value_counts(sort=False)
    else:
        # Catégorielle (ou numérique à peu de valeurs) : une tranche par modalité, y compris
        # les modalités INCONNUES de la référence (ex. un nouveau moyen de paiement).
        categories = sorted(set(reference.dropna()) | set(current.dropna()), key=str)
        ref_counts = reference.value_counts().reindex(categories, fill_value=0)
        cur_counts = current.value_counts().reindex(categories, fill_value=0)
    return ref_counts.to_numpy() / len(reference), cur_counts.to_numpy() / len(current)


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index entre deux échantillons d'une même variable (0 = identiques)."""
    reference, current = pd.Series(reference).reset_index(drop=True), pd.Series(current).reset_index(drop=True)
    if reference.empty or current.empty:
        raise ValueError("Les deux échantillons doivent être non vides")
    ref, cur = _shares(reference, current, bins)
    ref, cur = np.clip(ref, _EPS, None), np.clip(cur, _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def status(value: float) -> str:
    if value < PSI_WATCH:
        return "stable"
    if value < PSI_DRIFT:
        return "à surveiller"
    return "dérive"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """PSI de chaque variable, de la plus dérivée à la plus stable."""
    features = features or FEATURES
    rows = [{"feature": f, "psi": psi(reference[f], current[f])} for f in features]
    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    report["statut"] = report["psi"].map(status)
    return report


def simulate_batch(
    n: int = 2000,
    seed: int = 2026,
    *,
    price_increase: float = 0.0,
    monthly_share: float = 0.55,
    fibre_competition: float = 0.0,
) -> pd.DataFrame:
    """Un lot de nouveaux clients « six mois plus tard », pour tester la surveillance.

    - `price_increase` : hausse des prix (0.20 = +20 %)              → dérive des entrées
    - `monthly_share`  : part de contrats mensuels (0.55 aujourd'hui) → dérive des entrées
    - `fibre_competition` : un concurrent attire les clients fibre    → dérive du concept
    Les contrats Annuel / Biannuel gardent entre eux leur proportion actuelle (25 : 20).
    """
    if not 0 <= monthly_share <= 1:
        raise ValueError("monthly_share doit être entre 0 et 1")
    rest = 1 - monthly_share
    probs = (monthly_share, rest * 25 / 45, rest * 20 / 45)
    return generate_customers(
        n, seed, contract_probs=probs, price_increase=price_increase, fibre_competition=fibre_competition
    )

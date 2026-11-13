"""Auto-correction du défi 01.

    pytest defis/defi_01_explorer -q                  # vérifie VOTRE code (exercice.py)
    DEFI_CORRIGE=1 pytest defis/defi_01_explorer -q   # vérifie le corrigé
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parents[1] / "src"))

from churn.data import generate_customers  # noqa: E402


def _charger():
    nom = "corrige" if os.environ.get("DEFI_CORRIGE") == "1" else "exercice"
    spec = importlib.util.spec_from_file_location(f"defi01_{nom}", ICI / f"{nom}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _charger()


@pytest.fixture
def mini() -> pd.DataFrame:
    """10 clients dont on connaît les taux par cœur.

    Mensuel  : 3 départs / 4  = 75 %   (dont Mensuel × Chèque : 2/2 = 100 %)
    Annuel   : 1 départ  / 4  = 25 %
    Biannuel : 0 départ  / 2  =  0 %
    Global   : 4 départs / 10 = 40 %
    """
    return pd.DataFrame(
        {
            "contract": ["Mensuel"] * 4 + ["Annuel"] * 4 + ["Biannuel"] * 2,
            "payment_method": ["Chèque", "Chèque", "Carte", "Carte"] + ["Carte"] * 6,
            "monthly_charges": [20.0, 35.5, 80.0, 99.9, 50.0, 60.0, 70.0, 45.0, 30.0, 25.0],
            "churn": [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        }
    )


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    return generate_customers(n=5000, seed=42)


# --------------------------------------------------------------------------- Partie 1
def test_p1_structure(mini):
    out = ex.churn_rate_by(mini, "contract")
    assert list(out.columns) == ["contract", "clients", "churn_rate"], (
        "Les colonnes doivent être exactement [column, 'clients', 'churn_rate'] — pensez à reset_index()"
    )
    assert list(out.index) == [0, 1, 2], "L'index doit être 0, 1, 2… (reset_index)"


def test_p1_values_and_order(mini):
    out = ex.churn_rate_by(mini, "contract")
    assert list(out["contract"]) == ["Mensuel", "Annuel", "Biannuel"], "Tri par churn_rate décroissant attendu"
    assert list(out["clients"]) == [4, 4, 2]
    np.testing.assert_allclose(out["churn_rate"], [0.75, 0.25, 0.0])


def test_p1_does_not_mutate_input(mini):
    before = mini.copy()
    ex.churn_rate_by(mini, "contract")
    pd.testing.assert_frame_equal(mini, before, obj="Votre fonction a modifié le DataFrame d'entrée !")


def test_p1_real_data(customers):
    out = ex.churn_rate_by(customers, "contract")
    assert out["clients"].sum() == len(customers), "Chaque client doit être compté exactement une fois"
    assert out.iloc[0]["contract"] == "Mensuel"


# --------------------------------------------------------------------------- Partie 2
def test_p2_values(mini):
    out = ex.lift(mini, "contract")
    assert isinstance(out, pd.Series)
    assert out.name == "lift"
    assert list(out.index) == ["Mensuel", "Annuel", "Biannuel"]
    np.testing.assert_allclose(out.to_numpy(), [0.75 / 0.4, 0.25 / 0.4, 0.0])


def test_p2_weighted_average_is_one(customers):
    """Propriété mathématique : la moyenne des lifts, pondérée par la taille des groupes, vaut 1.
    (Question de réflexion : pourquoi ?)"""
    out = ex.lift(customers, "payment_method")
    sizes = customers["payment_method"].value_counts()
    weighted = (out * sizes.reindex(out.index)).sum() / sizes.sum()
    assert weighted == pytest.approx(1.0)


# --------------------------------------------------------------------------- Partie 3
def test_p3_small_segments_win_without_filter(mini):
    seg = ex.riskiest_segment(mini, ["contract", "payment_method"], min_clients=1)
    assert seg["segment"] == {"contract": "Mensuel", "payment_method": "Chèque"}
    assert seg["clients"] == 2
    assert seg["churn_rate"] == pytest.approx(1.0)


def test_p3_min_clients_filters(mini):
    seg = ex.riskiest_segment(mini, ["contract", "payment_method"], min_clients=3)
    assert seg["segment"] == {"contract": "Annuel", "payment_method": "Carte"}, (
        "Seuls les segments d'au moins `min_clients` clients doivent concourir"
    )
    assert seg["clients"] == 4
    assert seg["churn_rate"] == pytest.approx(0.25)


def test_p3_no_segment_big_enough(mini):
    with pytest.raises(ValueError):
        ex.riskiest_segment(mini, ["contract", "payment_method"], min_clients=50)


def test_p3_json_serializable(mini):
    seg = ex.riskiest_segment(mini, ["contract"], min_clients=1)
    try:
        json.dumps(seg)
    except TypeError as err:
        pytest.fail(f"Le résultat doit être sérialisable en JSON (convertissez avec int()/float()) : {err}")


def test_p3_real_data_finds_hidden_cause(customers):
    """Dans `_churn_logit`, « Mensuel » (+1,6) et « Chèque » (+0,7) augmentent le churn."""
    seg = ex.riskiest_segment(customers, ["contract", "payment_method"])
    assert seg["segment"] == {"contract": "Mensuel", "payment_method": "Chèque"}
    assert seg["churn_rate"] > 0.55


# --------------------------------------------------------------------------- Partie 4
def test_p4_clean_data_has_no_problem(customers):
    assert ex.check_monthly_charges(customers) == []


def test_p4_detects_out_of_range(mini):
    broken = mini.copy()
    broken.loc[0, "monthly_charges"] = -5.0
    broken.loc[1, "monthly_charges"] = 600.0
    problems = ex.check_monthly_charges(broken)
    assert len(problems) == 1, "Un seul message attendu, qui résume le problème"
    assert "2" in problems[0], "Le message doit indiquer combien de factures sont hors limites"


def test_p4_bounds_are_inclusive_and_nan_ignored(mini):
    edge = mini.copy()
    edge.loc[0, "monthly_charges"] = 0.0
    edge.loc[1, "monthly_charges"] = 500.0
    edge.loc[2, "monthly_charges"] = np.nan
    assert ex.check_monthly_charges(edge) == []


def test_p4_custom_bounds(mini):
    problems = ex.check_monthly_charges(mini, low=30, high=90)
    assert len(problems) == 1 and "3" in problems[0], "20,0 ; 25,0 et 99,9 sont hors de [30, 90]"


def test_p4_inconsistent_bounds(mini):
    with pytest.raises(ValueError):
        ex.check_monthly_charges(mini, low=100, high=10)

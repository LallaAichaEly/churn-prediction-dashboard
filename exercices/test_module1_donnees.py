"""Correcteur automatique des katas du Module 1. Lisez les tests : ils sont aussi une leçon !

    pytest exercices -q -rs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------------- 🟢 taux_churn


def test_taux_churn_exemple_simple(kata):
    df = pd.DataFrame({"churn": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]})
    assert kata.taux_churn(df) == pytest.approx(0.3)


def test_taux_churn_jeu_complet(kata, clients):
    taux = kata.taux_churn(clients)
    assert isinstance(taux, float), "renvoyez un nombre (float), pas une Series"
    assert taux == pytest.approx(clients["churn"].sum() / len(clients))


# --------------------------------------------------------------------------- 🟢 taux_churn_par


def test_taux_churn_par_contrat(kata, clients):
    taux = kata.taux_churn_par(clients, "contract")
    assert isinstance(taux, pd.Series)
    assert set(taux.index) == {"Mensuel", "Annuel", "Biannuel"}, "une ligne par type de contrat"
    assert taux.between(0, 1).all(), "un taux est toujours entre 0 et 1"
    assert taux.is_monotonic_decreasing, "triez du taux le plus élevé au plus faible"
    assert taux.index[0] == "Mensuel", "le contrat mensuel est celui qui part le plus"


def test_taux_churn_par_valeurs_exactes(kata):
    df = pd.DataFrame({"senior": ["Oui", "Oui", "Non", "Non", "Non", "Non"], "churn": [1, 0, 1, 0, 0, 0]})
    taux = kata.taux_churn_par(df, "senior")
    assert taux["Oui"] == pytest.approx(0.5)
    assert taux["Non"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- 🟢 valeurs_manquantes


def test_valeurs_manquantes_detectees(kata, clients):
    abime = clients.copy()
    abime.loc[[0, 1, 2], "monthly_charges"] = np.nan
    abime.loc[5, "contract"] = None
    manquantes = kata.valeurs_manquantes(abime)
    assert manquantes.to_dict() == {"monthly_charges": 3, "contract": 1}
    assert list(manquantes.index) == ["monthly_charges", "contract"], "la plus touchée d'abord"


def test_valeurs_manquantes_aucune(kata, clients):
    assert kata.valeurs_manquantes(clients).empty, "données propres → Series vide"


# --------------------------------------------------------------------------- 🟡 lift


def test_lift_calcul(kata):
    df = pd.DataFrame({"contract": ["A", "A", "B", "B"], "churn": [1, 1, 0, 0]})
    # Taux global 0,5 ; taux du groupe A = 1 → lift 2.
    assert kata.lift(df, "contract", "A") == pytest.approx(2.0)
    assert kata.lift(df, "contract", "B") == pytest.approx(0.0)


def test_lift_retrouve_la_verite_cachee(kata, clients):
    assert kata.lift(clients, "contract", "Mensuel") > 1.2
    assert kata.lift(clients, "contract", "Biannuel") < 0.6
    assert kata.lift(clients, "tech_support", "Oui") < 1, "le support technique retient les clients"


def test_lift_modalite_inconnue(kata, clients):
    with pytest.raises(ValueError):
        kata.lift(clients, "contract", "Hebdomadaire")


# --------------------------------------------------------------------------- 🟡 factures_hors_bornes


def test_factures_hors_bornes(kata, clients):
    abime = clients.copy()
    abime.loc[10, "monthly_charges"] = 600.0
    abime.loc[20, "monthly_charges"] = -5.0
    abime.loc[30, "monthly_charges"] = 500.0  # pile sur la borne : accepté
    avant = abime.copy()

    suspectes = kata.factures_hors_bornes(abime)

    assert sorted(suspectes.index) == [10, 20]
    assert list(suspectes.columns) == list(abime.columns), "gardez toutes les colonnes"
    pd.testing.assert_frame_equal(abime, avant, obj="df ne doit pas être modifié")


def test_factures_bornes_personnalisees(kata, clients):
    suspectes = kata.factures_hors_bornes(clients, minimum=20, maximum=120)
    attendues = clients[(clients["monthly_charges"] < 20) | (clients["monthly_charges"] > 120)]
    assert sorted(suspectes.index) == sorted(attendues.index)


# --------------------------------------------------------------------------- 🟡 churn_par_tranche_anciennete


def test_tranches_structure(kata, clients):
    resume = kata.churn_par_tranche_anciennete(clients)
    assert list(resume.columns) == ["tranche", "clients", "taux_churn"]
    assert list(resume["tranche"]) == ["1-6", "7-12", "13-24", "25-48", "49-72"]
    assert resume["clients"].sum() == len(clients), "chaque client est dans exactement une tranche"
    assert resume["taux_churn"].between(0, 1).all()


def test_tranches_les_nouveaux_partent_plus(kata, clients):
    resume = kata.churn_par_tranche_anciennete(clients)
    assert resume["taux_churn"].iloc[0] > 2 * resume["taux_churn"].iloc[-1]


def test_tranches_bornes_personnalisees(kata, clients):
    resume = kata.churn_par_tranche_anciennete(clients, bornes=(0, 12, 72))
    assert list(resume["tranche"]) == ["1-12", "13-72"]
    nouveaux = clients[clients["tenure_months"] <= 12]
    assert resume["clients"].iloc[0] == len(nouveaux)
    assert resume["taux_churn"].iloc[0] == pytest.approx(nouveaux["churn"].mean())


# --------------------------------------------------------------------------- 🔴 segments_a_risque

COLONNES = ["contract", "payment_method"]


def test_segments_structure(kata, clients):
    segments = kata.segments_a_risque(clients, COLONNES)
    assert list(segments.columns) == [*COLONNES, "clients", "taux_churn"]
    assert len(segments) == 5
    assert list(segments.index) == [0, 1, 2, 3, 4], "pensez à reset_index(drop=True)"


def test_segments_tri_et_filtre(kata, clients):
    segments = kata.segments_a_risque(clients, COLONNES, min_clients=100, top=10)
    assert (segments["clients"] >= 100).all(), "les petits segments doivent être exclus"
    assert segments["taux_churn"].is_monotonic_decreasing
    assert segments.loc[0, "contract"] == "Mensuel"


def test_segments_valeurs_exactes(kata, clients):
    segments = kata.segments_a_risque(clients, COLONNES)
    premier = segments.iloc[0]
    groupe = clients[(clients["contract"] == premier["contract"])
                     & (clients["payment_method"] == premier["payment_method"])]
    assert premier["clients"] == len(groupe)
    assert premier["taux_churn"] == pytest.approx(groupe["churn"].mean())


def test_segments_le_filtre_change_le_classement(kata):
    """Un segment minuscule au taux de 100 % ne doit pas sortir en tête."""
    df = pd.DataFrame({
        "contract": ["Rare"] * 3 + ["Courant"] * 100,
        "churn": [1] * 3 + [1] * 40 + [0] * 60,
    })
    segments = kata.segments_a_risque(df, ["contract"], min_clients=50)
    assert list(segments["contract"]) == ["Courant"]


def test_segments_aucun_assez_grand(kata, clients):
    segments = kata.segments_a_risque(clients, COLONNES, min_clients=10**6)
    assert segments.empty
    assert list(segments.columns) == [*COLONNES, "clients", "taux_churn"]

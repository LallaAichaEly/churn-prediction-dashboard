"""Smoke tests du tableau de bord : chaque onglet s'affiche sans exception.

`AppTest` exécute `app.py` sans navigateur et permet de cliquer sur les widgets.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=180).run()


def test_dashboard_renders_all_tabs(app):
    assert not app.exception
    assert not app.error
    assert [t.label for t in app.tabs] == [
        "Vue d'ensemble", "Performance", "Facteurs de churn", "Clients à risque", "Prédire un client",
    ]
    labels = {m.label for m in app.metric}
    assert {"Clients", "Taux de churn", "ROC-AUC", "Rappel"} <= labels


def test_switching_model_and_features_keeps_working(app):
    app.sidebar.selectbox[0].set_value("gboost").run()
    app.sidebar.checkbox[0].check().run()
    app.sidebar.slider[0].set_value(0.3).run()
    assert not app.exception


def test_prediction_form_shows_a_probability(app):
    app.button[0].click().run()
    assert not app.exception
    proba = next(m for m in app.metric if m.label == "Probabilité de churn")
    assert proba.value.endswith("%")

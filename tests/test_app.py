"""Tests du tableau de bord : on exécute `app.py` sans navigateur grâce à `AppTest` de Streamlit.

Ces tests ne vérifient pas l'apparence, seulement que chaque onglet s'affiche sans erreur
quels que soient les réglages. C'est le filet de sécurité quand vous modifiez `app.py`
pendant les exercices du module 5.
"""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metric(at, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_dashboard_renders_every_tab(app):
    assert len(app.tabs) == 5
    assert not app.error
    assert metric(app, "Clients") == "5 000"
    assert float(metric(app, "ROC-AUC")) > 0.7


@pytest.mark.parametrize("model_name", ["logistic", "gboost"])
def test_sidebar_settings_do_not_break_the_app(app, model_name):
    app.sidebar.selectbox[0].set_value(model_name)
    app.sidebar.checkbox[0].set_value(True)
    app.sidebar.slider[0].set_value(0.3)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error


def test_lower_threshold_flags_more_customers(app):
    flagged_at_50 = int(metric(app, "Clients signalés (jeu de test)"))
    app.sidebar.slider[0].set_value(0.2).run()
    assert int(metric(app, "Clients signalés (jeu de test)")) > flagged_at_50


def test_prediction_form(app):
    app.button[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    proba = metric(app, "Probabilité de churn")
    assert proba.endswith("%") and 0 <= float(proba.rstrip("%")) <= 100

"""Test de fumée du tableau de bord : l'application s'exécute-t-elle sans erreur ?

`AppTest` exécute `app.py` sans navigateur. C'est rapide et ça attrape la majorité des
régressions (colonne renommée, fonction supprimée, option Streamlit obsolète...).
"""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    return at


def test_dashboard_runs_without_error(app):
    assert not app.exception
    assert len(app.tabs) == 5
    assert "ROC-AUC" in [m.label for m in app.metric]


def test_prediction_form(app):
    app.button[0].click().run()  # bouton « Prédire » avec les valeurs par défaut
    assert not app.exception
    assert any(m.label == "Probabilité de churn" for m in app.metric)


def test_threshold_changes_flagged_customers(app):
    def flagged() -> int:
        return int(next(m.value for m in app.metric if m.label.startswith("Clients signalés")))

    app.sidebar.slider[0].set_value(0.3).run()  # seuil de décision
    low = flagged()
    app.sidebar.slider[0].set_value(0.8).run()
    assert flagged() < low

"""Test de fumée du tableau de bord : le script Streamlit s'exécute-t-il sans erreur ?

`AppTest` exécute `app.py` sans navigateur. Ce test ne vérifie pas que les graphiques sont
« justes », seulement que rien ne plante — c'est déjà beaucoup : une mise à jour de
bibliothèque qui casse l'application sera détectée ici, avant vos utilisateurs.
"""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_dashboard_runs_without_error(app):
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.tabs) == 5


def flagged(app) -> int:
    return int(next(m.value for m in app.metric if m.label == "Clients signalés (jeu de test)"))


def test_raising_threshold_flags_fewer_customers(app):
    app.sidebar.slider[0].set_value(0.3).run()
    at_03 = flagged(app)
    app.sidebar.slider[0].set_value(0.8).run()
    assert not app.exception
    assert flagged(app) < at_03


def test_prediction_form(app):
    app.button[0].click().run()  # bouton « Prédire » du formulaire
    assert not app.exception
    assert any(m.label == "Probabilité de churn" for m in app.metric)

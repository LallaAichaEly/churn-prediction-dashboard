"""Test de fumée du tableau de bord : l'application s'exécute de bout en bout sans erreur."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(APP), default_timeout=180).run()


def test_app_runs_without_errors(app):
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert len(app.tabs) == 5


def test_changing_model_reruns_cleanly(app):
    app.sidebar.selectbox[0].select("gboost").run()
    assert not app.exception, [e.value for e in app.exception]

"""Test de fumée du tableau de bord : chaque réglage s'affiche sans exception.

`AppTest` exécute app.py sans navigateur. Lent (~10 s) mais attrape les régressions d'affichage.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_dashboard_renders(app):
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.tabs) == 5
    assert app.title[0].value.startswith("📉")


def test_switching_settings_does_not_crash(app):
    app.sidebar.selectbox[0].select("gboost").run()
    app.sidebar.checkbox[0].check().run()
    app.sidebar.slider[0].set_value(0.3).run()
    assert not app.exception, [e.value for e in app.exception]


def test_prediction_form(app):
    app.button[0].click().run()  # bouton « Prédire » du formulaire
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "Probabilité de churn" for m in app.metric)

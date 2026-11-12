"""Test de fumée du tableau de bord : l'application démarre-t-elle sans erreur ?

`AppTest` exécute app.py sans navigateur et permet de manipuler les widgets.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def metric_value(app: AppTest, label_start: str) -> str:
    return next(m.value for m in app.metric if m.label.startswith(label_start))


@pytest.fixture(scope="module")
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=180).run()


def test_app_runs_without_error(app):
    assert not app.exception
    assert [t.label for t in app.tabs][-1] == "Surveillance (drift)"


def test_drift_tab_flags_a_price_increase(app):
    assert metric_value(app, "Variables en dérive significative") == "0 / 10"
    next(s for s in app.slider if s.label.startswith("Hausse des prix")).set_value(30)
    app.run()
    assert not app.exception
    assert metric_value(app, "Variables en dérive significative") == "1 / 10"

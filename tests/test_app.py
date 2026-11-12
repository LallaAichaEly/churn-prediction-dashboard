"""Module 6 — Test de fumée du tableau de bord : chaque onglet s'affiche sans erreur.

`AppTest` exécute `app.py` comme le ferait Streamlit, mais sans navigateur.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.mark.parametrize("model_name", ["logistic", "gboost"])
def test_dashboard_renders_without_error(model_name):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.selectbox[0].select(model_name).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5
    assert any(m.label == "ROC-AUC" for m in at.metric)


def test_predict_form_shows_probability_and_explanation():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    submit = next(b for b in at.button if b.label == "Prédire")
    submit.click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Probabilité de churn" for m in at.metric)

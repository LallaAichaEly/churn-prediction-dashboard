"""Tests du suivi de progression (scripts/progression.py).

Ils utilisent un mini-projet factice dans un dossier temporaire : vos propres modifications
du code (exercices en cours) ne peuvent donc pas les faire échouer.
"""

import re
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import progression as prog  # noqa: E402


def write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


FAKE_DATA_OK = """
    import pandas as pd

    def generate_customers(n, seed=0):
        return pd.DataFrame({"tenure_months": [i % 12 + 1 for i in range(n)], "monthly_charges": [50.0] * n})

    def validate(df):
        return [] if df["monthly_charges"].between(0, 500).all() else ["Facture hors limites"]
"""
FAKE_FEATURES_OK = """
    ENGINEERED_FEATURES = ["is_new_customer"]

    def add_engineered_features(df):
        out = df.copy()
        out["is_new_customer"] = (out["tenure_months"] < 6).astype(int)
        return out
"""
FAKE_MODEL_OK = """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    MODELS = {"logistic": "...", "rf": "..."}

    def build_pipeline(name="logistic"):
        clf = RandomForestClassifier() if name == "rf" else LogisticRegression()
        return Pipeline([("clf", clf)])
"""


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    """Un projet où les exercices 1.3, 2.3 et 3.2 sont réussis."""
    write(tmp_path, "src/churn/__init__.py", "")
    write(tmp_path, "src/churn/data.py", FAKE_DATA_OK)
    write(tmp_path, "src/churn/features.py", FAKE_FEATURES_OK)
    write(tmp_path, "src/churn/model.py", FAKE_MODEL_OK)
    (tmp_path / "PARCOURS.md").write_text((ROOT / "PARCOURS.md").read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- Le parcours lui-même
def test_parcours_is_well_formed():
    modules, exercises = prog.parse_parcours(ROOT / "PARCOURS.md")
    assert list(modules) == ["1", "2", "3", "4", "5", "6"]
    ids = [ex.id for ex in exercises]
    assert len(ids) == len(set(ids)) >= 30, "identifiants d'exercices dupliqués ou manquants"
    assert all(ex.id.split(".")[0] == ex.module for ex in exercises)


def test_every_exercise_has_hints():
    _, exercises = prog.parse_parcours(ROOT / "PARCOURS.md")
    hints = (ROOT / "INDICES.md").read_text(encoding="utf-8")
    missing = [ex.id for ex in exercises if not re.search(rf"^### {re.escape(ex.id)}$", hints, re.M)]
    assert not missing, f"Exercices sans indices dans INDICES.md : {missing}"


def test_every_auto_check_targets_a_real_exercise():
    _, exercises = prog.parse_parcours(ROOT / "PARCOURS.md")
    ids = {ex.id for ex in exercises}
    auto_ids = set(prog.CODE_CHECKS) | set(prog.FILE_CHECKS) | set(prog.BEHAVIOR_IDS)
    assert auto_ids <= ids


# --------------------------------------------------------------------------- Vérifications
def test_behavior_checks_pass_on_solved_exercises(fake_project):
    results = prog.run_behavior_checks(fake_project)
    assert {k: ok for k, (ok, _) in results.items()} == {"1.3": True, "2.3": True, "3.2": True}


def test_behavior_checks_explain_failures(fake_project):
    write(fake_project, "src/churn/data.py", FAKE_DATA_OK.replace('.between(0, 500).all()', ".notna().all()"))
    write(fake_project, "src/churn/features.py", FAKE_FEATURES_OK.replace('= ["is_new_customer"]', "= []"))
    write(fake_project, "src/churn/model.py", "raise RuntimeError('oups')")
    results = prog.run_behavior_checks(fake_project)
    assert results["1.3"] == (False, "une facture de 900 € n'est pas détectée par validate()")
    assert not results["2.3"][0] and "ENGINEERED_FEATURES" in results["2.3"][1]
    assert not results["3.2"][0] and "RuntimeError" in results["3.2"][1]


def test_commented_code_does_not_count(tmp_path):
    write(tmp_path, "app.py", "x = 1  # TODO precision_recall_curve(y, p)\n")
    where, patterns, _ = prog.CODE_CHECKS["4.4"]
    assert prog.check_code(prog.code_files(tmp_path), where, patterns) == []

    write(tmp_path, "app.py", "p, r, t = precision_recall_curve(y, proba)  # 4.4\n")
    assert prog.check_code(prog.code_files(tmp_path), where, patterns) == ["app.py"]


def test_notebooks_are_scanned(tmp_path):
    write(tmp_path, "notebooks/cv.ipynb",
          '{"cells": [{"cell_type": "code", "source": ["s = cross_val_score(pipe, X, y, cv=5)\\n"]}]}')
    where, patterns, _ = prog.CODE_CHECKS["3.4"]
    assert prog.check_code(prog.code_files(tmp_path), where, patterns) == ["notebooks/cv.ipynb"]


def test_file_checks(tmp_path):
    assert not prog.check_ci(tmp_path)[0]
    write(tmp_path, ".github/workflows/tests.yml", "jobs:\n  t:\n    steps:\n      - run: pytest -q\n")
    assert prog.check_ci(tmp_path)[0]

    write(tmp_path, "README.md", "Démo : https://mon-churn.streamlit.app\n")
    assert prog.check_deployed(tmp_path)[0]

    write(tmp_path, "RAPPORT.md", "trop court")
    assert not prog.check_report(tmp_path)[0]
    write(tmp_path, "RAPPORT.md", "mot " * prog.REPORT_MIN_WORDS)
    assert prog.check_report(tmp_path)[0]


# --------------------------------------------------------------------------- Statuts et affichage
def test_ticked_box_is_trusted_only_when_not_verifiable():
    manual = prog.Exercise("1.1", "🟢", "…", checked=True, module="1")
    assert prog.status_of(manual, {}).done

    claimed = prog.Exercise("1.3", "🟡", "…", checked=True, module="1")
    status = prog.status_of(claimed, {"1.3": (False, "non détecté")})
    assert not status.done and "⚠️" in status.icon


def test_render_counts_progress_and_suggests_next(fake_project):
    parcours = fake_project / "PARCOURS.md"
    parcours.write_text(parcours.read_text(encoding="utf-8").replace("- [ ] 🟢 1.1", "- [x] 🟢 1.1"),
                        encoding="utf-8")
    text = prog.render(fake_project)
    assert "☑️  1.1" in text and "✅ 1.3" in text and "✅ 2.3" in text and "✅ 3.2" in text
    assert "— 4/" in text
    assert "👉 Prochain exercice : 1.2" in text

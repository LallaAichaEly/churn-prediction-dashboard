"""Tests du coach de progression (scripts/progression.py)."""

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("progression", ROOT / "scripts" / "progression.py")
progression = importlib.util.module_from_spec(_spec)
sys.modules["progression"] = progression  # requis par @dataclass avant l'exécution du module
_spec.loader.exec_module(progression)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Copie minimale du projet : le parcours et un dossier de code vide."""
    shutil.copy(ROOT / "PARCOURS.md", tmp_path / "PARCOURS.md")
    (tmp_path / "src").mkdir()
    return tmp_path


def tick(project: Path, exercise_id: str) -> None:
    path = project / "PARCOURS.md"
    text = path.read_text(encoding="utf-8")
    text, n = re.subn(rf"^- \[ \](?= \S+ {re.escape(exercise_id)}( ⚡)? — )", "- [x]", text, flags=re.M)
    assert n == 1, f"exercice {exercise_id} introuvable dans PARCOURS.md"
    path.write_text(text, encoding="utf-8")


def status_of(project: Path, exercise_id: str):
    _, statuses = progression.evaluate_progress(project)
    return next(s for s in statuses if s.exercise.id == exercise_id)


def test_every_exercise_has_a_guide_and_vice_versa():
    """Si on ajoute un exercice au parcours, il faut aussi lui écrire un indice."""
    modules, exercises = progression.parse_parcours((ROOT / "PARCOURS.md").read_text(encoding="utf-8"))
    assert {e.id for e in exercises} == set(progression.GUIDES)
    assert set(modules) == {e.module for e in exercises}
    assert all(e.level in progression.LEVELS for e in exercises)


def test_defi_marker_is_parsed_and_matches_the_defis():
    """Un exercice marqué ⚡ doit avoir un défi associé, et chaque défi doit pointer vers un exercice ⚡."""
    _, exercises = progression.parse_parcours("- [ ] 🟢 1.2 ⚡ — Faire ceci\n- [x] 🟡 1.3 — Faire cela\n")
    assert [(e.id, e.has_defi, e.ticked) for e in exercises] == [("1.2", True, False), ("1.3", False, True)]

    _, exercises = progression.parse_parcours((ROOT / "PARCOURS.md").read_text(encoding="utf-8"))
    score = progression.defis_score(ROOT)
    assert score is not None and not score.erreur and score.total >= 10
    assert {e.id for e in exercises if e.has_defi} == set(score.par_exercice)


def test_defis_are_optional(project):
    assert progression.defis_score(project) is None
    assert "Défis" not in progression.render(project)


def test_atelier_git_is_optional(project):
    assert progression.atelier_git_score(project) is None
    assert "Atelier Git" not in progression.render(project)


@pytest.mark.skipif(shutil.which("git") is None, reason="git n'est pas installé")
def test_atelier_git_score(tmp_path):
    score = progression.atelier_git_score(ROOT, racine=tmp_path / "atelier-git")
    assert score == progression.AtelierScore(0, 8)


def test_beginner_hint_only_before_any_progress():
    """Rien de fait : on propose la série 00. Dès le premier défi ou exercice réussi, on se tait."""
    zero = progression.DefiScore(0, 37, {})
    assert "série 00" in progression.conseil_debutant(0, zero)
    assert progression.conseil_debutant(0, progression.DefiScore(1, 37, {})) is None
    assert progression.conseil_debutant(1, zero) is None
    assert progression.conseil_debutant(0, None) is None
    assert progression.conseil_debutant(0, progression.DefiScore(0, 0, {}, "SyntaxError")) is None


def test_parse_ticked_boxes():
    text = "## Module 1 — Données (`x.py`)\n- [x] 🟢 1.1 — Faire **ceci**\n- [ ] 🟡 1.2 — Faire `cela`\n"
    modules, exercises = progression.parse_parcours(text)
    assert modules == {1: "Données"}
    assert [(e.id, e.ticked, e.title) for e in exercises] == [("1.1", True, "Faire ceci"), ("1.2", False, "Faire cela")]


def test_reflection_exercise_counts_when_ticked(project):
    assert not status_of(project, "1.1").done
    tick(project, "1.1")
    assert status_of(project, "1.1").done


def test_journal_mention_is_detected(project):
    (project / "JOURNAL.md").write_text("## Exercice 1.4\nMes hypothèses…\n", encoding="utf-8")
    assert status_of(project, "1.4").in_journal
    assert not status_of(project, "1.1").in_journal


def test_comment_does_not_validate_code_exercise(project):
    code = project / "src" / "tuning.py"
    code.write_text("# TODO : utiliser GridSearchCV\n", encoding="utf-8")
    assert not status_of(project, "3.5").done
    code.write_text("from sklearn.model_selection import GridSearchCV\nsearch = GridSearchCV(None, {})\n", encoding="utf-8")
    assert status_of(project, "3.5").done


def test_ticked_but_failing_is_flagged_and_comes_first(project):
    tick(project, "3.5")
    _, statuses = progression.evaluate_progress(project)
    flagged = next(s for s in statuses if s.exercise.id == "3.5")
    assert flagged.mismatch and not flagged.done
    assert progression.next_status(statuses).exercise.id == "3.5"


def test_next_exercise_follows_the_order(project):
    tick(project, "1.1")
    tick(project, "1.2")  # marqué ⚡ dans le parcours
    _, statuses = progression.evaluate_progress(project)
    assert progression.next_status(statuses).exercise.id == "1.3"


def test_ci_workflow_check(project):
    """Exercice 6.3 : la CI de base est fournie ; il faut l'étendre (Python 3.13 + ruff)."""
    workflows = project / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "tests.yml").write_text("on: push\njobs: {}\n", encoding="utf-8")
    assert not status_of(project, "6.3").done
    base = "on: push\njobs:\n  t:\n    steps:\n      - run: pytest -q\n"
    (workflows / "tests.yml").write_text(base + "# Exercice : ajoutez Python 3.13 et ruff\n", encoding="utf-8")
    assert not status_of(project, "6.3").done, "la CI fournie (ou un commentaire) ne valide pas l'exercice"
    (workflows / "tests.yml").write_text(
        base.replace("    steps:", '    strategy:\n      matrix:\n        python-version: ["3.12", "3.13"]\n    steps:')
        + "      - run: ruff check .\n",
        encoding="utf-8",
    )
    assert status_of(project, "6.3").done


def test_batch_route_needs_code_and_test(project):
    (project / "api.py").write_text('@app.post("/predict")\ndef predict(): ...\n', encoding="utf-8")
    assert not status_of(project, "6.4").done, "l'API fournie ne valide pas l'exercice"
    (project / "api.py").write_text('@app.post("/predict/batch")\ndef batch(): ...\n', encoding="utf-8")
    assert "test" in status_of(project, "6.4").result[1]
    (project / "tests").mkdir()
    (project / "tests" / "test_batch.py").write_text('client.post("/predict/batch", json=[])\n', encoding="utf-8")
    assert status_of(project, "6.4").done


def test_model_version_check(project):
    (project / "src" / "model.py").write_text("import sklearn\nmeta = {'v': sklearn.__version__}\n", encoding="utf-8")
    assert not status_of(project, "6.2").done, "enregistrer la version ne suffit pas : il faut avertir"
    (project / "src" / "model.py").write_text(
        "import sklearn, warnings\nif saved != sklearn.__version__:\n    warnings.warn('version')\n", encoding="utf-8"
    )
    assert status_of(project, "6.2").done


def test_monitor_script_check_runs_the_script(project):
    """Exercice 7.6 : le coach exécute le script ; un `sys.exit(1)` écrit au hasard ne suffit pas."""
    scripts = project / "scripts"
    scripts.mkdir()
    assert not status_of(project, "7.6").done
    header = (
        "import sys\nsys.path.insert(0, %r)\nimport pandas as pd\n"
        "from churn.data import generate_customers\nfrom churn.monitoring import drift_report\n"
        "report = drift_report(generate_customers(3000, seed=0), pd.read_csv(sys.argv[1]))\n"
    ) % str(ROOT / "src")
    (scripts / "monitor.py").write_text(header + "sys.exit(1)\n", encoding="utf-8")
    ok, message = status_of(project, "7.6").result
    assert not ok and "stable" in message, "toujours en alerte = faux positif sur le lot stable"
    (scripts / "monitor.py").write_text(header + "sys.exit(int(report['psi'].max() > 0.25))\n", encoding="utf-8")
    assert status_of(project, "7.6").done


def test_markdown_is_cleaned_for_the_terminal():
    assert progression._clean("*Dérive ≠ dégradation* : **AUC** de `model`") == "Dérive ≠ dégradation : AUC de model"
    assert progression._clean("2*3*4 et a * b") == "2*3*4 et a * b"


def test_validate_rule_check(project, monkeypatch):
    """Exercice 1.3 : vérifié en EXÉCUTANT validate() sur des factures aberrantes."""
    from churn import data

    assert not progression._check_validate_rule(project)[0]

    original = data.validate

    def validate_with_rule(df):
        problems = original(df)
        if (~df["monthly_charges"].between(0, 500)).any():
            problems.append("Facture hors bornes")
        return problems

    monkeypatch.setattr(data, "validate", validate_with_rule)
    ok, message = progression._check_validate_rule(project)
    assert not ok and "test" in message  # la règle marche, mais le test manque encore

    (project / "tests").mkdir()
    (project / "tests" / "test_bills.py").write_text('broken.loc[0, "monthly_charges"] = 900\n', encoding="utf-8")
    assert progression._check_validate_rule(project)[0]


def test_crashing_check_is_reported_not_raised(project):
    def boom(root):
        raise RuntimeError("code en chantier")

    ok, message = progression._run(boom, project)
    assert not ok and "code en chantier" in message


def test_cli_runs_on_this_repository():
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "progression.py")],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    assert "Progression :" in run.stdout
    detail = subprocess.run([sys.executable, str(ROOT / "scripts" / "progression.py"), "--exercice", "3.2"],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    assert "Indice" in detail.stdout
    unknown = subprocess.run([sys.executable, str(ROOT / "scripts" / "progression.py"), "--exercice", "9.9"],
                             capture_output=True, text=True, encoding="utf-8")
    assert unknown.returncode == 2

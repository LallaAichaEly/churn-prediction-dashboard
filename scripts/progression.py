"""Suivi de progression du parcours de compétences (PARCOURS.md).

Combine deux sources :
- les cases que vous cochez dans PARCOURS.md (`- [x]`) ;
- des vérifications automatiques 🤖 sur votre code, quand l'exercice s'y prête.

Usage :
    python scripts/progression.py              # vue d'ensemble
    python scripts/progression.py --module 3   # détail d'un module (avec les vérifications)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]

# Ces fichiers contiennent les motifs recherchés : on les exclut de l'analyse.
SELF_FILES = {"scripts/progression.py", "tests/test_progression.py"}
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
             "site-packages", "node_modules"}

MODULE_RE = re.compile(r"^## Module (?P<num>\d+) — (?P<title>.+?)(?:\s+\(.*\))?\s*$")
EXERCISE_RE = re.compile(
    r"^- \[(?P<mark>[ xX])\] (?P<level>🟢|🟡|🔴) (?P<id>\d+\.\d+) — (?P<title>.+)$"
)
REPORT_FILE = "RAPPORT.md"
REPORT_MIN_WORDS = 150


@dataclass
class Exercise:
    id: str
    level: str
    title: str
    checked: bool
    module: str


@dataclass
class Status:
    done: bool
    icon: str
    detail: str = ""


# --------------------------------------------------------------------------- Lecture du parcours
def parse_parcours(path: Path) -> tuple[dict[str, str], list[Exercise]]:
    """Retourne ({numéro de module: titre}, [exercices dans l'ordre])."""
    modules: dict[str, str] = {}
    exercises: list[Exercise] = []
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := MODULE_RE.match(line):
            current = m["num"]
            modules[current] = m["title"]
        elif (m := EXERCISE_RE.match(line)) and current:
            exercises.append(
                Exercise(m["id"], m["level"], m["title"], m["mark"].lower() == "x", current)
            )
    return modules, exercises


# --------------------------------------------------------------------------- Analyse du code
def _strip_comments(code: str) -> str:
    """Efface les commentaires `#` : mentionner une fonction en commentaire ne suffit pas !"""
    try:
        comments = [t for t in tokenize.generate_tokens(io.StringIO(code).readline)
                    if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, SyntaxError):
        return code
    lines = code.splitlines(keepends=True)
    for tok in comments:
        row, col = tok.start
        lines[row - 1] = lines[row - 1][:col] + "\n"
    return "".join(lines)


def _notebook_code(path: Path) -> str:
    try:
        cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    sources = []
    for cell in cells:
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            sources.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(_strip_comments(s) for s in sources)


def code_files(root: Path) -> dict[str, str]:
    """{chemin relatif: code sans commentaires} pour les .py et .ipynb du projet."""
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts) or rel in SELF_FILES:
            continue
        if path.suffix == ".py":
            try:
                files[rel] = _strip_comments(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                continue
        elif path.suffix == ".ipynb":
            files[rel] = _notebook_code(path)
    return files


def _in(*prefixes: str) -> Callable[[str], bool]:
    return lambda rel: any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in prefixes)


def _anywhere(rel: str) -> bool:
    return True


# id → (où chercher, motifs qui doivent TOUS apparaître dans un même fichier, message de réussite)
CODE_CHECKS: dict[str, tuple[Callable[[str], bool], list[str], str]] = {
    "3.4": (_anywhere, [r"\bcross_val(?:_score|idate)\s*\("], "validation croisée trouvée"),
    "3.5": (_anywhere, [r"\b(?:Grid|Randomized)SearchCV\s*\("], "recherche d'hyperparamètres trouvée"),
    "3.6": (_anywhere, [r"class_weight\s*=\s*['\"]balanced['\"]"], "class_weight='balanced' trouvé"),
    "4.4": (_in("app.py", "src"), [r"\bprecision_recall_curve\s*\(|\bPrecisionRecallDisplay\b"],
            "courbe précision-rappel trouvée"),
    "4.5": (_anywhere, [r"\bcalibration_curve\s*\(|\bCalibrationDisplay\b"], "calibration analysée"),
    "5.2": (_in("app.py"), [r"groupby\(\s*(?:by\s*=\s*)?['\"]internet_service['\"]"],
            "graphique par service internet trouvé"),
    "5.5": (_in("app.py", "src"), [r"^\s*(?:import\s+shap\b|from\s+shap\b)"], "SHAP intégré"),
    "6.2": (_in("tests"), [r"(?:\bjoblib\.load|\bload)\s*\(", r"\bpredict(?:_proba)?\s*\("],
            "test de rechargement du modèle trouvé"),
    "6.4": (_anywhere, [r"\bFastAPI\s*\(", r"['\"]/predict['\"]"], "API FastAPI /predict trouvée"),
}


def check_code(files: dict[str, str], where: Callable[[str], bool], patterns: list[str]) -> list[str]:
    """Fichiers où tous les motifs apparaissent."""
    regexes = [re.compile(p, re.MULTILINE) for p in patterns]
    return [rel for rel, code in files.items() if where(rel) and all(r.search(code) for r in regexes)]


# --------------------------------------------------------------------------- Vérifications « fichiers »
def check_ci(root: Path) -> tuple[bool, str]:
    workflows = root / ".github" / "workflows"
    for wf in sorted(workflows.glob("*.y*ml")) if workflows.is_dir() else []:
        if "pytest" in wf.read_text(encoding="utf-8"):
            return True, f"workflow {wf.name} lance pytest"
    return False, "aucun workflow dans .github/workflows/ ne lance pytest"


def check_deployed(root: Path) -> tuple[bool, str]:
    readme = root / "README.md"
    if readme.exists() and re.search(r"https?://[\w.-]+\.streamlit\.app", readme.read_text(encoding="utf-8")):
        return True, "lien .streamlit.app présent dans le README"
    return False, "aucun lien https://….streamlit.app dans le README"


def check_report(root: Path) -> tuple[bool, str]:
    path = root / REPORT_FILE
    if not path.exists():
        return False, f"{REPORT_FILE} n'existe pas encore"
    words = len(path.read_text(encoding="utf-8").split())
    if words < REPORT_MIN_WORDS:
        return False, f"{REPORT_FILE} ne fait que {words} mots (visez au moins {REPORT_MIN_WORDS})"
    return True, f"{REPORT_FILE} rédigé ({words} mots)"


FILE_CHECKS: dict[str, Callable[[Path], tuple[bool, str]]] = {"6.3": check_ci, "6.5": check_deployed}


# --------------------------------------------------------------------------- Vérifications « comportement »
# Exécutées dans un processus séparé : si votre code plante, le suivi, lui, continue.
BEHAVIOR_SCRIPT = r'''
import json, sys
sys.path.insert(0, sys.argv[1])

def ex_1_3():
    from churn.data import generate_customers, validate
    df = generate_customers(300, seed=0)
    if validate(df):
        return False, "validate() signale des problèmes sur des données saines"
    for bad in (900.0, -10.0):
        broken = df.copy()
        broken.loc[0, "monthly_charges"] = bad
        if not validate(broken):
            return False, f"une facture de {bad:g} € n'est pas détectée par validate()"
    return True, "factures hors de [0, 500] € détectées"

def ex_2_3():
    from churn import features
    from churn.data import generate_customers
    df = generate_customers(300, seed=0)
    out = features.add_engineered_features(df)
    if "is_new_customer" not in out.columns:
        return False, "add_engineered_features() ne crée pas la colonne is_new_customer"
    if "is_new_customer" not in features.ENGINEERED_FEATURES:
        return False, "is_new_customer manque dans ENGINEERED_FEATURES : le modèle ne la voit pas"
    expected = (df["tenure_months"] < 6).astype(int).to_numpy()
    if not (out["is_new_customer"].astype(int).to_numpy() == expected).all():
        return False, "valeurs incorrectes (attendu : 1 si ancienneté < 6 mois, sinon 0)"
    return True, "is_new_customer créée et utilisée par le modèle"

def ex_3_2():
    from sklearn.ensemble import RandomForestClassifier
    from churn.model import MODELS, build_pipeline
    for name in MODELS:
        if isinstance(build_pipeline(name).named_steps["clf"], RandomForestClassifier):
            return True, f"forêt aléatoire disponible sous le nom {name!r}"
    return False, "aucun modèle de MODELS n'utilise RandomForestClassifier"

results = {}
for key, fn in [("1.3", ex_1_3), ("2.3", ex_2_3), ("3.2", ex_3_2)]:
    try:
        results[key] = list(fn())
    except Exception as exc:
        results[key] = [False, f"erreur en exécutant votre code : {type(exc).__name__}: {exc}"]
print("@@RESULTS@@" + json.dumps(results, ensure_ascii=False))
'''
BEHAVIOR_IDS = ("1.3", "2.3", "3.2")


def run_behavior_checks(root: Path, timeout: int = 180) -> dict[str, tuple[bool, str]]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", BEHAVIOR_SCRIPT, str(root / "src")],
            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {k: (False, f"vérification interrompue après {timeout} s") for k in BEHAVIOR_IDS}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("@@RESULTS@@"):
            return {k: (bool(v[0]), str(v[1])) for k, v in json.loads(line[len("@@RESULTS@@"):]).items()}
    last = (proc.stderr.strip().splitlines() or ["erreur inconnue"])[-1]
    return {k: (False, f"impossible d'exécuter les vérifications : {last}") for k in BEHAVIOR_IDS}


# --------------------------------------------------------------------------- Synthèse
def auto_results(root: Path) -> dict[str, tuple[bool, str]]:
    """{id d'exercice: (réussi, explication)} pour tous les exercices vérifiables."""
    results = run_behavior_checks(root)
    files = code_files(root)
    for ex_id, (where, patterns, success) in CODE_CHECKS.items():
        found = check_code(files, where, patterns)
        results[ex_id] = (True, f"{success} ({', '.join(found[:2])})") if found else (False, "pas encore trouvé dans le code")
    for ex_id, check in FILE_CHECKS.items():
        results[ex_id] = check(root)
    return results


def status_of(ex: Exercise, auto: dict[str, tuple[bool, str]]) -> Status:
    if ex.id in auto:
        ok, detail = auto[ex.id]
        if ok:
            return Status(True, "✅", f"vérifié 🤖 : {detail}")
        if ex.checked:
            return Status(False, "⚠️ ", f"cochée, mais la vérification échoue : {detail}")
        return Status(False, "⬜", f"🤖 {detail}")
    if ex.checked:
        return Status(True, "☑️ ")
    return Status(False, "⬜")


def progress_bar(done: int, total: int, width: int = 20) -> str:
    filled = round(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def shorten(text: str, n: int = 72) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def render(root: Path, only_module: str | None = None) -> str:
    modules, exercises = parse_parcours(root / "PARCOURS.md")
    auto = auto_results(root)
    statuses = {ex.id: status_of(ex, auto) for ex in exercises}
    report_ok, report_detail = check_report(root)

    total = len(exercises) + 1
    done = sum(s.done for s in statuses.values()) + report_ok
    out = [f"🧭 Progression du parcours — {done}/{total} ({done / total:.0%})",
           f"   {progress_bar(done, total, 30)}", ""]

    for num, title in modules.items():
        if only_module and num != only_module:
            continue
        mod_ex = [ex for ex in exercises if ex.module == num]
        mod_done = sum(statuses[ex.id].done for ex in mod_ex)
        out.append(f"Module {num} — {title}  {progress_bar(mod_done, len(mod_ex), 10)} {mod_done}/{len(mod_ex)}")
        for ex in mod_ex:
            st = statuses[ex.id]
            out.append(f"  {st.icon} {ex.id} {ex.level} {shorten(ex.title)}")
            # Le détail s'affiche quand il apprend quelque chose : réussite, alerte, ou vue détaillée.
            if st.detail and (only_module or st.done or "⚠️" in st.icon):
                out.append(f"       ↳ {st.detail}")
        out.append("")

    if not only_module:
        out += [f"🏁 Projet final  {'✅' if report_ok else '⬜'} {report_detail}", ""]

    nxt = next((ex for ex in exercises if not statuses[ex.id].done), None)
    if nxt:
        out += [f"👉 Prochain exercice : {nxt.id} {nxt.level} — {shorten(nxt.title, 90)}",
                f"   Besoin d'un coup de pouce ? INDICES.md, section {nxt.id}."]
    elif not report_ok:
        out.append(f"👉 Tous les exercices sont faits ! Dernière étape : rédigez {REPORT_FILE}.")
    else:
        out.append("🎉 Parcours terminé. Ajoutez RAPPORT.md et JOURNAL.md à votre portfolio !")
    out.append("")
    out.append("Légende : ✅ vérifié 🤖 · ☑️ coché · ⚠️ coché mais vérification échouée · ⬜ à faire")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", help="N'afficher qu'un module, avec le détail des vérifications")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # emojis sous Windows
    except (AttributeError, ValueError):
        pass
    print(render(ROOT, args.module))


if __name__ == "__main__":
    main()

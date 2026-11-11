"""🧭 Coach de progression : où en êtes-vous dans PARCOURS.md, et que faire ensuite ?

Usage :
    python scripts/progression.py                 # bilan par module + prochain exercice avec indice
    python scripts/progression.py --exercice 3.2  # détail, indice et critère de validation d'un exercice

Comment le coach décide qu'un exercice est fait :
- Exercices de CODE (ex. 3.2, 4.4, 6.3) : vérifiés automatiquement, en inspectant ou en exécutant
  votre code. Cocher la case ne suffit pas, il faut que ça marche !
- Exercices de RÉFLEXION (ex. 1.4, 3.3) : comptent comme faits quand vous cochez leur case
  dans PARCOURS.md (`- [x]`). Notez vos réponses dans JOURNAL.md : le coach l'indique par 📝.

Les indices mettent sur la voie sans donner la solution : c'est en cherchant qu'on progresse.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CheckResult = tuple[bool, str]  # (réussi ?, explication pour l'apprenant)
Check = Callable[[Path], CheckResult]

# Les fichiers du coach citent les mots-clés recherchés : on les exclut des recherches.
COACH_FILES = {"progression.py", "test_progression.py"}
CODE = ("src/**/*.py", "scripts/*.py", "*.py")
TESTS = ("tests/*.py",)


# --------------------------------------------------------------------------- Lecture du code


def _strip_comments(source: str) -> str:
    """Retire les commentaires : un `# TODO: GridSearchCV` ne doit pas valider l'exercice."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return tokenize.untokenize(t for t in tokens if t.type != tokenize.COMMENT)
    except Exception:  # code en cours d'écriture, syntaxe invalide… on garde le texte brut
        return source


def _read_code(root: Path, globs: tuple[str, ...]) -> str:
    texts = []
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path.name not in COACH_FILES and ".venv" not in path.parts:
                texts.append(_strip_comments(path.read_text(encoding="utf-8", errors="replace")))
    return "\n".join(texts)


def _grep(patterns: list[str], globs: tuple[str, ...], ok: str, ko: str) -> Check:
    """Vérification statique : tous les motifs doivent apparaître dans le code."""

    def check(root: Path) -> CheckResult:
        code = _read_code(root, globs)
        return (True, ok) if all(re.search(p, code) for p in patterns) else (False, ko)

    return check


# --------------------------------------------------------------------------- Vérifications
# Les vérifications « à l'exécution » importent le paquet `churn` du projet (dossier src/).


def _check_validate_rule(root: Path) -> CheckResult:
    from churn import data

    df = data.generate_customers(200, seed=0)
    if data.validate(df):
        return False, "validate() rejette des données saines : " + " ; ".join(data.validate(df))
    for bad in (900.0, -5.0):
        broken = df.copy()
        broken.loc[0, "monthly_charges"] = bad
        if not data.validate(broken):
            return False, f"validate() ne signale pas une facture mensuelle à {bad:g} €"
    if not re.search(r"""["']monthly_charges["']\s*\]\s*=\s*-?\d""", _read_code(root, TESTS)):
        return False, "la règle marche ; il manque le test qui casse une facture (ex. 900 €)"
    return True, "validate() rejette les factures hors de [0, 500] € et un test le protège"


def _check_is_new_customer(root: Path) -> CheckResult:
    from churn import data, features

    df = data.generate_customers(200, seed=0)
    out = features.add_engineered_features(df)
    if "is_new_customer" not in out.columns:
        return False, "add_engineered_features() ne crée pas encore la colonne is_new_customer"
    expected = (df["tenure_months"] < 6).astype(int).to_numpy()
    if not (out["is_new_customer"].astype(int).to_numpy() == expected).all():
        return False, "is_new_customer doit valoir 1 si l'ancienneté est < 6 mois, 0 sinon"
    if "is_new_customer" not in features.ENGINEERED_FEATURES:
        return False, "la colonne existe, mais n'est pas dans ENGINEERED_FEATURES : le modèle l'ignore"
    return True, "is_new_customer est calculée correctement et utilisée par le modèle"


def _check_random_forest(root: Path) -> CheckResult:
    from sklearn.ensemble import RandomForestClassifier

    from churn import model

    for name in model.MODELS:
        if isinstance(model.build_pipeline(name)[-1], RandomForestClassifier):
            return True, f"le modèle « {name} » est une forêt aléatoire"
    return False, "aucun modèle de MODELS ne construit encore de RandomForestClassifier"


def _check_ci(root: Path) -> CheckResult:
    """Exercice 6.3 : la CI de base est fournie, il faut l'ÉTENDRE (Python 3.13 + lint ruff)."""
    workflows = [*(root / ".github" / "workflows").glob("*.yml"), *(root / ".github" / "workflows").glob("*.yaml")]
    if not workflows:
        return False, "aucun workflow dans .github/workflows/"
    # Les commentaires YAML ne comptent pas : « # TODO : ajouter ruff » ne valide rien.
    text = "\n".join(re.sub(r"(^|\s)#.*$", "", p.read_text(encoding="utf-8", errors="replace"), flags=re.M)
                     for p in workflows)
    if "pytest" not in text:
        return False, "un workflow existe, mais il ne lance pas pytest"
    missing = [label for label, pattern in (("Python 3.13", r"""["']?3\.13\b"""), ("une étape ruff", r"\bruff\b"))
               if not re.search(pattern, text)]
    if missing:
        return False, "la CI lance pytest ; il manque encore " + " et ".join(missing)
    return True, "la CI lance pytest, teste Python 3.13 et passe le lint ruff"


def _check_batch_route(root: Path) -> CheckResult:
    route = r"""["']/predict/batch["']"""
    if not re.search(route, _read_code(root, CODE)):
        return False, "pas encore de route \"/predict/batch\" dans l'API"
    if not re.search(route, _read_code(root, TESTS)):
        return False, "la route /predict/batch existe ; il manque son test dans tests/"
    return True, "route /predict/batch présente dans l'API et protégée par un test"


def _check_monitor_script(root: Path) -> CheckResult:
    """Exercice 7.6 : on EXÉCUTE le script sur un lot qui dérive et sur un lot stable."""
    import os
    import subprocess
    import tempfile

    from churn.monitoring import simulate_batch

    path = root / "scripts" / "monitor.py"
    if not path.exists():
        return False, "scripts/monitor.py n'existe pas encore"
    if "drift_report" not in _strip_comments(path.read_text(encoding="utf-8", errors="replace")):
        return False, "scripts/monitor.py n'appelle pas encore drift_report()"

    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")])}
    with tempfile.TemporaryDirectory() as tmp:
        cases = {"stable": (simulate_batch(2000, seed=2), 0),
                 "prix +30 %": (simulate_batch(2000, seed=1, price_increase=0.30), 1)}
        for label, (batch, expected) in cases.items():
            csv = Path(tmp) / f"lot_{expected}.csv"
            batch.to_csv(csv, index=False)
            run = subprocess.run([sys.executable, str(path), str(csv)], cwd=root, env=env,
                                 capture_output=True, text=True, timeout=180)
            if run.returncode not in (0, 1):
                last = (run.stderr.strip().splitlines() or ["(aucun message)"])[-1]
                return False, f"scripts/monitor.py plante sur le lot « {label} » : {last}"
            if run.returncode != expected:
                return False, (f"lot « {label} » : code de sortie {run.returncode}, attendu {expected} "
                               "(1 si une variable dépasse un PSI de 0,25, sinon 0)")
    return True, "scripts/monitor.py renvoie 1 sur un lot qui dérive et 0 sur un lot stable"


def _check_report(root: Path) -> CheckResult:
    path = root / "RAPPORT.md"
    if not path.exists():
        return False, "RAPPORT.md n'existe pas encore"
    words = len(path.read_text(encoding="utf-8", errors="replace").split())
    if words < 150:
        return False, f"RAPPORT.md ne fait que {words} mots (visez au moins 150)"
    return True, f"RAPPORT.md rédigé ({words} mots)"


@dataclass(frozen=True)
class Guide:
    hint: str
    check: Check | None = None  # None = exercice de réflexion, validé en cochant la case
    criterion: str = "cochez la case dans PARCOURS.md une fois l'exercice fait (exercice de réflexion)"


GUIDES: dict[str, Guide] = {
    # Module 1 — Données
    "1.1": Guide("Le CSV est créé par `python scripts/train.py`. La cible valant 0 ou 1, sa moyenne "
                 "est directement le taux de churn."),
    "1.2": Guide("`df.groupby(\"contract\")[\"churn\"].mean().sort_values()`. Pour croiser deux variables : "
                 "`pd.crosstab(df.contract, df.internet_service, values=df.churn, aggfunc=\"mean\")`."),
    "1.3": Guide("Inspirez-vous de la règle « Ancienneté négative » dans `validate()`. `(s < 0) | (s > 500)` "
                 "renvoie un booléen par ligne (le défi 12 montre pourquoi c'est plus sûr que `~s.between(0, 500)`). "
                 "Pour le test, partez de "
                 "`test_validate_detects_problems` et cassez une facture.",
                 _check_validate_rule,
                 "validate() signale une facture à 900 € et à -5 €, accepte les données saines, "
                 "et un test casse une facture"),
    "1.4": Guide("Le signe d'un coefficient donne le sens de l'effet, sa valeur la force… par unité ! "
                 "0,035 par mois d'ancienneté × 72 mois = 2,5 : c'est plus que le contrat mensuel (1,6)."),
    "1.5": Guide("Fichier Kaggle : `WA_Fn-UseC_-Telco-Customer-Churn.csv`. Pièges : `TotalCharges` est du texte "
                 "(`pd.to_numeric(errors=\"coerce\")`), `Churn` vaut \"Yes\"/\"No\", et `support_calls` / "
                 "`num_products` n'existent pas : retirez-les ou construisez un équivalent (nombre de services)."),
    # Module 2 — Variables
    "2.1": Guide("StandardScaler : (x − moyenne) / écart-type, appris sur le train seulement. OneHotEncoder : une "
                 "colonne 0/1 par modalité. Question guide : pourquoi la logistique a-t-elle besoin du premier, "
                 "et pas un arbre de décision ?"),
    "2.2": Guide("Comparez avec et sans la case, pour les deux modèles (ou `python scripts/train.py --engineered`). "
                 "Un écart de 0,002 d'AUC peut être du bruit : notez les deux valeurs dans JOURNAL.md."),
    "2.3": Guide("Deux endroits à modifier : la colonne dans `add_engineered_features` ET son nom dans "
                 "`ENGINEERED_FEATURES`, sinon le préprocesseur l'ignore. `(out[\"tenure_months\"] < 6).astype(int)`.",
                 _check_is_new_customer,
                 "la colonne is_new_customer existe, vaut 1 si ancienneté < 6 mois, et figure dans ENGINEERED_FEATURES"),
    "2.4": Guide("Retirez temporairement `handle_unknown=\"ignore\"`, relancez `pytest -q` et lisez le message "
                 "d'erreur. Que se passerait-il en production avec un nouveau moyen de paiement ?"),
    "2.5": Guide("Avec un StandardScaler sur 5 000 lignes, l'écart sera minuscule : c'est normal, et c'est une "
                 "leçon. Les fuites dangereuses viennent d'informations « du futur » ou d'un encodage calculé avec "
                 "la cible sur tout le jeu (ex. taux de churn moyen par contrat). Essayez celui-là !"),
    # Module 3 — Modélisation
    "3.1": Guide("`python scripts/train.py`. Le script garde le meilleur selon la ROC-AUC, mais regardez aussi "
                 "précision et rappel : le classement est-il le même ?"),
    "3.2": Guide("Trois endroits : le dictionnaire `MODELS`, une branche `elif` dans `build_pipeline`, et la liste "
                 "du test paramétré `test_models_beat_random`. Point de départ : "
                 "`RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)`.",
                 _check_random_forest,
                 "un des modèles de MODELS construit un RandomForestClassifier"),
    "3.3": Guide("Dans `_churn_logit`, le churn est une SOMME de termes linéaires passée dans une sigmoïde : "
                 "exactement la forme d'une régression logistique. Qu'est-ce qu'un modèle plus flexible "
                 "pourrait apprendre de plus… à part le bruit ?"),
    "3.4": Guide("Passez le PIPELINE complet (pas un X déjà transformé, sinon fuite !) à `cross_val_score`, avec "
                 "`cv=StratifiedKFold(5, shuffle=True, random_state=42)` et `scoring=\"roc_auc\"`. "
                 "Comparez moyenne ± écart-type des deux modèles.",
                 _grep([r"\bcross_val(_score|idate)\b"], CODE,
                       "validation croisée détectée dans le code", "cross_val_score / cross_validate pas encore utilisé"),
                 "cross_val_score ou cross_validate est utilisé dans src/, scripts/ ou app.py"),
    "3.5": Guide("Dans un pipeline, les hyperparamètres se nomment `clf__learning_rate`, `clf__max_depth`… "
                 "Commencez petit (2 × 3 valeurs), `scoring=\"roc_auc\"`, `cv=5`. Ne touchez au jeu de test "
                 "qu'une seule fois, à la toute fin.",
                 _grep([r"\b(GridSearchCV|RandomizedSearchCV|HalvingGridSearchCV)\b"], CODE,
                       "recherche d'hyperparamètres détectée", "GridSearchCV / RandomizedSearchCV pas encore utilisé"),
                 "GridSearchCV ou RandomizedSearchCV est utilisé dans src/, scripts/ ou app.py"),
    "3.6": Guide("`LogisticRegression(class_weight=\"balanced\", max_iter=1000)`. Attendez-vous à plus de rappel "
                 "et moins de précision. Question piège : est-ce vraiment différent de baisser le seuil ?",
                 _grep([r"class_weight\s*="], CODE,
                       "class_weight utilisé dans le code", "class_weight pas encore utilisé"),
                 "un modèle est construit avec class_weight=…"),
    # Module 4 — Évaluation
    "4.1": Guide("Seuil bas → on signale plus de clients : le rappel monte, la précision baisse. "
                 "Repérez le seuil où les deux courbes se croisent."),
    "4.2": Guide("Calculez `1 - df[\"churn\"].mean()` : c'est l'accuracy d'un modèle qui prédit toujours "
                 "« reste »… et qui ne détecte aucun client sur le départ."),
    "4.3": Guide("Saisissez les coûts dans l'onglet Performance, ou appelez "
                 "`best_threshold_by_cost(y_test, proba, 500, 50)`. Intuition : si rater un départ coûte 10× plus "
                 "qu'une action inutile, le seuil doit nettement baisser."),
    "4.4": Guide("`precision_recall_curve(y_test, proba)` renvoie précision, rappel et seuils (un seuil de moins "
                 "que de points !). Tracez le rappel en x et la précision en y avec `go.Scatter`, comme la courbe "
                 "ROC. Bonus : `average_precision_score`.",
                 _grep([r"\bprecision_recall_curve\b"], CODE,
                       "courbe précision-rappel détectée", "precision_recall_curve pas encore utilisé"),
                 "precision_recall_curve est utilisé dans app.py ou src/"),
    "4.5": Guide("`from sklearn.calibration import calibration_curve` puis `calibration_curve(y_test, proba, "
                 "n_bins=10)` : tracez la fréquence réelle de churn selon la probabilité prédite. "
                 "La diagonale = calibration parfaite.",
                 _grep([r"\b(calibration_curve|CalibrationDisplay)\b"], CODE,
                       "analyse de calibration détectée", "calibration_curve / CalibrationDisplay pas encore utilisé"),
                 "calibration_curve ou CalibrationDisplay est utilisé"),
    # Module 5 — Interprétation
    "5.1": Guide("Comparez le classement d'importance aux coefficients de `_churn_logit`, en pensant à la "
                 "fréquence de chaque cas : `senior` ne concerne que 18 % des clients, le paiement par chèque 15 %."),
    "5.2": Guide("Réutilisez la fonction `bar()` : copiez le bloc « par moyen de paiement » et remplacez "
                 "`payment_method` par `internet_service`.",
                 _grep([r"""groupby\(\s*["']internet_service["']"""], ("app.py",),
                       "graphique par service internet détecté", "pas encore de groupby sur internet_service dans app.py"),
                 "app.py calcule le taux de churn groupé par internet_service"),
    "5.3": Guide("`pipeline[:-1].transform(client)` donne les valeurs transformées (toutes les étapes sauf le "
                 "modèle, variables dérivées comprises), `pipeline.named_steps[\"clf\"].coef_[0]` les poids, "
                 "`pipeline.named_steps[\"prep\"].get_feature_names_out()` les noms. "
                 "Contribution = valeur × poids : triez par valeur absolue.",
                 _grep([r"\.coef_\b"], ("app.py",),
                       "contributions par coefficients détectées", "coef_ pas encore utilisé dans app.py"),
                 "app.py utilise les coefficients (coef_) du modèle"),
    "5.4": Guide("`st.sidebar.multiselect(\"Contrats\", CONTRACTS, default=CONTRACTS)`, puis filtrez `df` dans "
                 "l'onglet Vue d'ensemble seulement : le modèle, lui, doit rester entraîné sur toutes les données.",
                 _grep([r"sidebar\.\w+\([^)]*\bCONTRACTS\b"], ("app.py",),
                       "filtre par contrat détecté dans la barre latérale",
                       "pas encore de widget de barre latérale utilisant CONTRACTS"),
                 "un widget de la barre latérale (st.sidebar.…) propose les CONTRACTS"),
    "5.5": Guide("`shap.LinearExplainer` pour la logistique, `shap.TreeExplainer` pour le boosting. Ils travaillent "
                 "sur les données DÉJÀ transformées : appliquez d'abord `named_steps[\"prep\"]`. "
                 "Mettez le calcul en cache avec `st.cache_resource`.",
                 _grep([r"^\s*(import shap|from shap\b)"], CODE,
                       "SHAP est intégré", "shap n'est pas encore importé"),
                 "le module shap est importé dans app.py ou src/"),
    # Module 6 — Industrialisation
    "6.1": Guide("Pour chaque test, demandez-vous : « quel bug aurait-il attrapé ? ». Cassez volontairement le "
                 "code (ex. supprimez `stratify=` dans `split`) et regardez quel test échoue… ou pas !"),
    "6.2": Guide("Dans `save()`, ajoutez `sklearn.__version__` au JSON des métriques ; dans `load()`, relisez-le "
                 "et appelez `warnings.warn(...)` s'il diffère. Testez-le avec `pytest.warns(UserWarning)` en "
                 "réécrivant le JSON avec une fausse version.",
                 _grep([r"sklearn\.__version__|version\(\s*[\"']scikit-learn[\"']\s*\)", r"\bwarn"], CODE,
                       "la version de scikit-learn est enregistrée et vérifiée au chargement",
                       "la version de scikit-learn n'est pas encore enregistrée / comparée au chargement"),
                 "le code lit sklearn.__version__ et émet un avertissement (warn) en cas d'écart"),
    "6.3": Guide("Dans `.github/workflows/tests.yml` : ajoutez \"3.13\" à `matrix.python-version`, puis une étape "
                 "`pip install ruff` et `ruff check .` AVANT les tests (le lint est plus rapide qu'eux).",
                 _check_ci,
                 "un workflow lance pytest, teste Python 3.13 et passe le lint ruff"),
    "6.4": Guide("Une route `@app.post(\"/predict/batch\")` qui reçoit `list[Customer]` : construisez UN DataFrame "
                 "avec toutes les lignes et appelez `predict_proba` une seule fois (bien plus rapide qu'une boucle). "
                 "Testez-la avec `TestClient`, comme `tests/test_api.py`.",
                 _check_batch_route,
                 "\"/predict/batch\" apparaît dans le code de l'API et dans un fichier de tests/"),
    "6.5": Guide("Poussez sur GitHub, puis share.streamlit.io → « Create app » → fichier `app.py`. Les données "
                 "sont générées au démarrage : rien d'autre à configurer.",
                 _grep([r"https?://[\w-]+\.streamlit\.app"], ("README.md",),
                       "lien Streamlit Cloud présent dans le README", "pas encore de lien *.streamlit.app dans le README"),
                 "README.md contient un lien https://….streamlit.app"),
    # Module 7 — Surveillance
    "7.1": Guide("Le PSI compare des TRANCHES (déciles) calculées sur la référence. Combien de clients changent "
                 "de tranche quand on multiplie chaque valeur par 1,2 ? Comparez la largeur des déciles des deux "
                 "variables (`df[col].quantile([.1, .2, …])`)."),
    "7.2": Guide("PSI = Σ (actuel − référence) × ln(actuel / référence), une fois par modalité. Ici deux termes : "
                 "(0,7 − 0,5) × ln(0,7 / 0,5) et (0,3 − 0,5) × ln(0,3 / 0,5). Comparez ensuite aux seuils 0,10 et 0,25."),
    "7.3": Guide("Bouclez sur une vingtaine de graines et gardez à chaque fois le PSI MAXIMAL sur toutes les "
                 "variables (`drift_report(ref, lot)[\"psi\"].max()`) : c'est lui qui déclencherait une fausse alerte."),
    "7.4": Guide("L'AUC mesure la capacité à SÉPARER les clients. Si presque tout le monde a le même contrat, la "
                 "variable la plus discriminante sépare moins… même pour un modèle parfait. Regardez plutôt l'écart "
                 "entre churn prédit moyen et churn réel."),
    "7.5": Guide("Le lot « concurrent fibre » contient exactement les mêmes clients que le lot « aucun changement » : "
                 "seule leur DÉCISION de partir a changé. Que peut voir un PSI calculé sur les variables d'entrée ?"),
    "7.6": Guide("Lisez le CSV, appelez `drift_report(reference, lot)`, affichez-le, puis `sys.exit(1)` si "
                 "`report[\"psi\"].max() > PSI_DRIFT`. Un code de sortie ≠ 0 fait échouer une étape de CI ou alerte un cron.",
                 _check_monitor_script,
                 "scripts/monitor.py lot.csv renvoie le code 1 sur un lot qui dérive (prix +30 %) et 0 sur un lot stable"),
}


# --------------------------------------------------------------------------- Parcours


EXERCISE_RE = re.compile(
    r"^- \[(?P<tick>[ xX])\] (?P<level>\S+) (?P<id>\d+\.\d+)(?P<defi> ⚡)? — (?P<title>.+)$", re.M
)
MODULE_RE = re.compile(r"^## Module (?P<num>\d+) — (?P<title>.+)$", re.M)
LEVELS = {"🟢": "débutant", "🟡": "intermédiaire", "🔴": "avancé"}


@dataclass
class Exercise:
    id: str
    level: str
    title: str
    ticked: bool
    has_defi: bool = False  # marqué ⚡ dans PARCOURS.md : un défi auto-corrigé s'y prépare

    @property
    def module(self) -> int:
        return int(self.id.split(".")[0])


@dataclass
class Status:
    exercise: Exercise
    guide: Guide
    result: CheckResult | None  # None pour un exercice de réflexion
    in_journal: bool = False

    @property
    def done(self) -> bool:
        return self.result[0] if self.result else self.exercise.ticked

    @property
    def mismatch(self) -> bool:
        """Case cochée, mais la vérification automatique échoue."""
        return self.exercise.ticked and self.result is not None and not self.result[0]

    @property
    def icon(self) -> str:
        return "❗" if self.mismatch else "✅" if self.done else "⬜"

    @property
    def note(self) -> str:
        notes = []
        if self.result and (self.done or self.mismatch):
            notes.append(self.result[1])
        if self.result and self.done and not self.exercise.ticked:
            notes.append("pensez à cocher la case")
        if self.result is None and self.done:
            notes.append("coché")
        if self.in_journal:
            notes.append("📝")
        return " — " + ", ".join(notes) if notes else ""


def _clean(markdown: str) -> str:
    """Retire le gras, l'italique (*mot*) et le code (`x`) pour l'affichage en terminal."""
    text = markdown.replace("**", "").replace("`", "")
    return re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"\1", text)


def parse_parcours(text: str) -> tuple[dict[int, str], list[Exercise]]:
    modules = {
        int(m["num"]): _clean(re.sub(r"\s*\(.*\)\s*$", "", m["title"])) for m in MODULE_RE.finditer(text)
    }
    exercises = [
        Exercise(m["id"], m["level"], _clean(m["title"].strip()), m["tick"] != " ", bool(m["defi"]))
        for m in EXERCISE_RE.finditer(text)
    ]
    return modules, exercises


def _run(check: Check, root: Path) -> CheckResult:
    try:
        return check(root)
    except Exception as exc:  # le code de l'apprenant peut être en chantier : on explique, on ne plante pas
        return False, f"erreur pendant la vérification ({type(exc).__name__} : {exc})"


def evaluate_progress(root: Path = ROOT) -> tuple[dict[int, str], list[Status]]:
    parcours = root / "PARCOURS.md"
    if not parcours.exists():
        raise FileNotFoundError(f"{parcours} introuvable")
    modules, exercises = parse_parcours(parcours.read_text(encoding="utf-8"))
    journal_path = root / "JOURNAL.md"
    journal = journal_path.read_text(encoding="utf-8", errors="replace") if journal_path.exists() else ""

    statuses = []
    for ex in exercises:
        guide = GUIDES.get(ex.id, Guide("Pas encore d'indice pour cet exercice."))
        result = _run(guide.check, root) if guide.check else None
        in_journal = bool(re.search(rf"(?<![\d.]){re.escape(ex.id)}(?![\d])", journal))
        statuses.append(Status(ex, guide, result, in_journal))
    return modules, statuses


def next_status(statuses: list[Status]) -> Status | None:
    """D'abord réparer ce qui est coché mais ne marche pas, puis le premier exercice non fait."""
    return next((s for s in statuses if s.mismatch), None) or next((s for s in statuses if not s.done), None)


# --------------------------------------------------------------------------- Défis (defis/)


@dataclass(frozen=True)
class DefiScore:
    reussis: int
    total: int
    par_exercice: dict[str, list[str]]  # "2.5" -> ["✅ n°5 standardiser"]
    erreur: str = ""


def defis_score(root: Path = ROOT) -> DefiScore | None:
    """Score aux défis auto-corrigés, ou None si le dossier defis/ n'existe pas."""
    if not (root / "defis" / "verif.py").exists():
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from defis.verif import ICONES, REUSSI, evaluer, series

        resultats = [r for serie in series() for r in evaluer(serie)]
    except Exception as exc:  # ex. erreur de syntaxe dans defis/serie_01.py
        return DefiScore(0, 0, {}, f"{type(exc).__name__} : {exc}")
    par_exercice: dict[str, list[str]] = {}
    for r in resultats:
        for ex_id in re.findall(r"\d+\.\d+", r.defi.parcours):
            par_exercice.setdefault(ex_id, []).append(f"{ICONES[r.statut]} n°{r.defi.numero} {r.defi.fonction}")
    return DefiScore(sum(r.statut == REUSSI for r in resultats), len(resultats), par_exercice)


# --------------------------------------------------------------------------- Affichage


def _bar(done: int, total: int, width: int = 10) -> str:
    filled = round(width * done / total) if total else 0
    return "▰" * filled + "▱" * (width - filled)


def _short(text: str, limit: int = 72) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _detail(status: Status, modules: dict[int, str], indent: str = "   ", defis: DefiScore | None = None) -> str:
    ex = status.exercise
    lines = [
        f"{ex.id} {ex.level} ({LEVELS.get(ex.level, '')}) — Module {ex.module} — {modules.get(ex.module, '')}",
        f"{indent}{ex.title}",
        f"{indent}💡 Indice : {status.guide.hint}",
        f"{indent}🔎 Validé quand : {status.guide.criterion}",
    ]
    if status.result:
        lines.append(f"{indent}{'✅' if status.result[0] else '⬜'} État actuel : {status.result[1]}")
    if ex.has_defi:
        lies = ", ".join(defis.par_exercice.get(ex.id, [])) if defis else ""
        lines.append(f"{indent}⚡ Échauffement : le défi associé d'abord ({lies or 'voir defis/'}) — python scripts/defis.py")
    return "\n".join(lines)


def render(root: Path = ROOT) -> str:
    modules, statuses = evaluate_progress(root)
    out = ["🧭 Coach de progression — PARCOURS.md", ""]

    for num, title in modules.items():
        items = [s for s in statuses if s.exercise.module == num]
        done = sum(s.done for s in items)
        out.append(f"Module {num} — {title}  {done}/{len(items)} {_bar(done, len(items), len(items))}")
        for s in items:
            defi = " ⚡ " if s.exercise.has_defi else "    "  # ⚡ occupe deux colonnes
            out.append(f"  {s.icon} {s.exercise.id}{defi}{s.exercise.level} {_short(s.exercise.title)}{s.note}")
        out.append("")

    total = sum(s.done for s in statuses)
    by_level = " · ".join(
        f"{lvl} {sum(s.done for s in statuses if s.exercise.level == lvl)}"
        f"/{sum(s.exercise.level == lvl for s in statuses)}"
        for lvl in LEVELS
    )
    pct = total / len(statuses) if statuses else 0
    out.append(f"Progression : {total}/{len(statuses)} exercices ({pct:.0%}) {_bar(total, len(statuses))}  {by_level}")

    defis = defis_score(root)
    if defis and defis.erreur:
        out.append(f"⚡ Défis : 💥 impossible de les corriger ({defis.erreur}) — python scripts/defis.py")
    elif defis:
        out.append(f"⚡ Défis : {defis.reussis}/{defis.total} réussis {_bar(defis.reussis, defis.total)}"
                   "  — détail : python scripts/defis.py")
    report_ok, report_msg = _run(_check_report, root)
    out.append(f"🏁 Projet final (RAPPORT.md) : {'✅' if report_ok else '⬜'} {report_msg}")
    if not (root / "JOURNAL.md").exists():
        out.append("📝 Astuce : créez JOURNAL.md et notez-y vos résultats « avant / après » à chaque exercice.")
    out.append("")

    nxt = next_status(statuses)
    if nxt is None:
        out.append("🎉 Tous les exercices sont faits ! Il reste le projet final si ce n'est pas déjà fait.")
    else:
        prefix = "❗ À corriger d'abord : " if nxt.mismatch else "👉 Prochain exercice : "
        out.append(prefix + _detail(nxt, modules, defis=defis))
        out.append(f"   Plus de détails : python scripts/progression.py --exercice {nxt.exercise.id}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    try:  # la console Windows n'est pas toujours en UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Bilan de progression dans PARCOURS.md et prochain exercice.")
    parser.add_argument("--exercice", metavar="X.Y", help="affiche l'indice et l'état d'un exercice précis")
    args = parser.parse_args(argv)

    if args.exercice:
        modules, statuses = evaluate_progress(ROOT)
        match = next((s for s in statuses if s.exercise.id == args.exercice), None)
        if match is None:
            ids = ", ".join(s.exercise.id for s in statuses)
            print(f"Exercice inconnu : {args.exercice}. Choix possibles : {ids}", file=sys.stderr)
            return 2
        print(_detail(match, modules, defis=defis_score(ROOT)))
        return 0

    print(render(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

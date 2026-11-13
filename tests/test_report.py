"""Le rapport est généré par code : on vérifie qu'il reste générable et cohérent."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report.py"


def load_script():
    spec = importlib.util.spec_from_file_location("report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_contains_every_section(df):
    text = load_script().build_report(df, cost_fn=500, cost_fp=50)
    for section in ["Problème", "Données", "Modèle retenu", "Seuil de décision", "Impact financier",
                    "Recommandations", "Limites", "Reproduire"]:
        assert section in text
    assert "1 500 clients" in text  # formatage français des nombres


def test_french_number_formatting():
    report = load_script()
    assert report.pct(0.355) == "35,5 %"
    assert report.eur(221500) == "221 500 €"
    assert report.integer(1250) == "1 250"

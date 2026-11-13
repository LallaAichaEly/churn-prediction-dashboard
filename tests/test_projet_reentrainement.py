"""Garantit que le projet 5 (réentraîner le modèle sans casser la production) est faisable et bien corrigé.

Ces tests NE notent PAS votre projet (pour ça : `python scripts/projet.py --projet 5`) ; ils vérifient
le corrigé, les données d'exemple, et que chaque erreur classique reçoit le bon message.
"""

import itertools
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projets import checks_reentrainement as checks  # noqa: E402
from projets.reentrainement_corrige import decision, lots  # noqa: E402
from projets.reentrainement_corrige.modeles import IDENTIFIANT, MOIS, Mesures  # noqa: E402
from projets.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_etape  # noqa: E402

PROJET = checks.PROJET
LOTS = ROOT / "projets" / "exemple_lots.csv"
_compteur = itertools.count()


def _python(*args, cwd=ROOT):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          timeout=180)


# ---------------------------------------------------------------------------- Le projet est cohérent


@pytest.mark.parametrize("etape", PROJET.etapes, ids=lambda e: e.module)
def test_le_corrige_reussit(etape):
    resultat = evaluer_etape(PROJET.paquet_corrige, etape, PROJET.etapes)
    assert resultat.statut == REUSSI, resultat.message


def test_le_squelette_est_entierement_a_faire():
    assert [r.statut for r in evaluer(PROJET)] == [A_FAIRE] * len(PROJET.etapes)


@pytest.mark.parametrize("fichier", ["modeles.py", "fichiers.py", "entrainement.py", "__main__.py"])
def test_les_fichiers_deja_ecrits_sont_identiques(fichier):
    eleve = ROOT / "projets" / "reentrainement" / fichier
    corrige = ROOT / "projets" / "reentrainement_corrige" / fichier
    assert eleve.read_text(encoding="utf-8") == corrige.read_text(encoding="utf-8")


def test_les_lots_d_exemple_sont_reproductibles_et_sains():
    lus = pd.read_csv(LOTS, dtype={MOIS: str, IDENTIFIANT: str})
    pd.testing.assert_frame_equal(lus, checks.generer_lots(), check_dtype=False)
    assert lots.problemes(lus) == []
    assert sorted(lus[MOIS].unique()) == [str(pd.Period("2025-10", "M") + k) for k in range(12)]


def test_le_scenario_du_correcteur_couvre_promotion_refus_et_retour(tmp_path):
    """Le correcteur de l'étape 6 suppose : promu, refusé, promu (sinon `revenir` ne serait pas testé)."""
    donnees = checks.generer_lots(graine=2026, debut="2025-10", nb_mois=12, clients=150)
    issues = [checks._attendu_entrainer(donnees, mois, tmp_path / "r") for mois in ("2026-03", "2026-04", "2026-09")]
    assert issues == [(True, 1), (False, 1), (True, 3)]


# ---------------------------------------------------------------------------- Le programme terminé


def test_le_programme_termine_de_bout_en_bout(tmp_path):
    registre = tmp_path / "registre_modeles"
    paquet = "projets.reentrainement_corrige"

    def entrainer(mois):
        return _python("-m", paquet, "entrainer", "--lots", str(LOTS), "--mois", mois, "--registre", str(registre))

    res = entrainer("2026-03")
    assert res.returncode == 0, res.stderr
    assert "1500 clients (2025-10 → 2026-02)" in res.stdout and "✅ v001 promue" in res.stdout

    res = entrainer("2026-04")  # un mois sans dérive : le challenger ne fait pas vraiment mieux
    assert res.returncode == 0, res.stderr
    assert "❌ v002 enregistrée, mais pas promue (le champion reste v001)" in res.stdout
    assert "gain d'AUC 0,003 < 0,010" in res.stdout

    res = entrainer("2026-09")  # après la dérive : le vieux champion s'est dégradé
    assert res.returncode == 0, res.stderr
    assert "champion v001     0,789" in res.stdout and "challenger v003   0,837" in res.stdout
    assert "✅ v003 promue championne" in res.stdout

    res = _python("-m", paquet, "historique", "--registre", str(registre))
    assert res.returncode == 0 and res.stdout.count("\n") == 3 and "v003" in res.stdout.splitlines()[-1]
    assert "champion" in res.stdout.splitlines()[-1]

    res = _python("-m", paquet, "revenir", "--registre", str(registre))
    assert res.returncode == 0 and "Retour à v001" in res.stdout
    res = _python("-m", paquet, "revenir", "--registre", str(registre))
    assert res.returncode == 1 and "aucun champion précédent" in res.stderr and "Traceback" not in res.stderr
    assert sorted(p.name for p in registre.iterdir()) == ["registre.json", "v001.joblib", "v002.joblib", "v003.joblib"]


def test_un_lot_abime_est_refuse_sans_rien_ecrire(tmp_path):
    abime = pd.read_csv(LOTS, dtype={MOIS: str, IDENTIFIANT: str})
    abime.loc[abime[MOIS] == "2026-09", "churn"] = 0           # l'incident de mars
    abime.loc[5, "contract"] = "Mensual"
    abime.to_csv(tmp_path / "abime.csv", index=False)
    res = _python("-m", "projets.reentrainement_corrige", "entrainer", "--lots", str(tmp_path / "abime.csv"),
                  "--mois", "2026-09", "--registre", str(tmp_path / "r"))
    assert res.returncode == 1 and "Lot refusé : 2 problème(s)" in res.stderr
    assert "2026-09" in res.stderr and "Mensual" in res.stderr
    assert not (tmp_path / "r").exists()


def test_le_squelette_explique_qu_il_n_est_pas_termine(tmp_path):
    res = _python("-m", "projets.reentrainement", "historique", "--registre", str(tmp_path / "r"))
    assert res.returncode == 1
    assert "scripts/projet.py --projet 5" in res.stderr and "Traceback" not in res.stderr


def test_le_script_de_progression_connait_le_projet_5():
    res = _python("scripts/projet.py", "--projet", "5", "--corrige")
    assert res.returncode == 0 and "7/7" in res.stdout and "Projet 5" in res.stdout and "Projet 4" not in res.stdout
    res = _python("scripts/projet.py", "--projet", "5", "--etape", "4")
    assert res.returncode == 0 and "decision.py" in res.stdout and "lots.py" not in res.stdout


# ---------------------------------------------------------------------------- Les erreurs classiques


def _variante(tmp_path, monkeypatch, fichier: str, remplacements):
    """Le corrigé, avec UNE erreur classique introduite dans `fichier`, corrigé par le vrai moteur."""
    nom = f"reentrainement_variante_{next(_compteur)}"
    dossier = tmp_path / nom
    shutil.copytree(ROOT / "projets" / "reentrainement_corrige", dossier, ignore=shutil.ignore_patterns("__pycache__"))
    texte = (dossier / fichier).read_text(encoding="utf-8")
    for avant, apres in remplacements:
        assert avant in texte, f"{avant!r} introuvable dans {fichier}"
        texte = texte.replace(avant, apres)
    (dossier / fichier).write_text(texte, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    etape = next(e for e in PROJET.etapes if e.module == Path(fichier).stem)
    return evaluer_etape(nom, etape, PROJET.etapes)


_TRAIN = "dans_train = (periodes >= debut) & (periodes < test_p)"
_ENREGISTRER_PUIS_PROMOUVOIR = ("    version = enregistrer(args.registre, modele, mesures)\n"
                                "    if decision.promouvoir:\n        promouvoir(args.registre, version.numero)\n")

ERREURS_CLASSIQUES = [
    # Étape 1
    ("lots.py", [('return [f"colonnes manquantes : {\', \'.join(manquantes)}"]',
                  'return [f"colonne manquante : {c}" for c in manquantes]')], "UNE phrase"),
    ("lots.py", [('return [f"colonnes manquantes : {\', \'.join(manquantes)}"]', "pass")], "lève KeyError"),
    ("lots.py", [('_FORMAT_MOIS = r"\\d{4}-(0[1-9]|1[0-2])"', '_FORMAT_MOIS = r"\\d{4}-\\d{2}"')], "13ᵉ mois"),
    ("lots.py", [("~mois.str.fullmatch(_FORMAT_MOIS)", '~mois.str.contains("-")')], "2026-9"),
    ("lots.py", [("valeurs = pd.to_numeric(brutes, errors=\"coerce\")", "valeurs = brutes")], "lève TypeError"),
    ("lots.py", [("inconnues = valeurs[~valeurs.isin(MODALITES[colonne])]", "inconnues = valeurs[valeurs.isna()]")],
     "Mensual"),
    ("lots.py", [("~valeurs.isin(MODALITES[colonne])",
                  "~valeurs.str.lower().isin([v.lower() for v in MODALITES[colonne]])")],
     "« fibre » n'est pas « Fibre »"),
    ("lots.py", [("lots.duplicated([MOIS, IDENTIFIANT], keep=\"first\")", "lots.duplicated([IDENTIFIANT], keep=\"first\")")],
     "n'est PAS un doublon"),
    ("lots.py", [("if taux in (0.0, 1.0):", "if False:")], "incident de mars"),
    ("lots.py", [("        vides = int(lots[colonne].isna().sum())\n        if vides:\n"
                  "            trouves.append(f\"{colonne} : {vides} valeur(s) manquante(s)\")",
                  "        for ligne in lots.index[lots[colonne].isna()]:\n"
                  "            trouves.append(f\"{colonne} : case vide (ligne {ligne})\")")], "UNE phrase"),
    ("lots.py", [("f\"{colonne} : {vides} valeur(s) manquante(s)\"", "f\"{colonne} : valeur(s) manquante(s)\"")],
     "combien"),
    ("lots.py", [("    # 5. Les catégories", "    if trouves:\n        return trouves\n    # 5. Les catégories")],
     "rassemblez-les TOUS"),
    ("lots.py", [("    trouves: list[str] = []\n",
                  "    trouves: list[str] = []\n    lots[CIBLE] = pd.to_numeric(lots[CIBLE], errors=\"coerce\")\n")],
     "MODIFIÉ"),
    # Étape 2
    ("decoupage.py", [(_TRAIN, "dans_train = (periodes >= debut) & (periodes <= test_p)")], "AUSSI dans train"),
    ("decoupage.py", [(_TRAIN, "dans_train = (periodes >= debut) & (periodes != test_p)")], "fuite temporelle"),
    ("decoupage.py", [("debut = test_p - historique", "debut = test_p - historique + 1")], "12 mois de 2025-01"),
    ("decoupage.py", [("return lots[dans_train].copy(), lots[dans_test].copy()",
                       "return lots[dans_test].copy(), lots[dans_train].copy()")], r"renvoie \(test, train\)"),
    ("decoupage.py", [("if type(historique) is not int or historique < 1:", "if historique < 1:")], "historique=True"),
    ("decoupage.py", [('if not isinstance(mois_test, str) or not re.fullmatch(r"\\d{4}-(0[1-9]|1[0-2])", mois_test):',
                       "if not isinstance(mois_test, str):")], "'2026-1'"),
    ("decoupage.py", [("    if not dans_train.any():\n", "    if False:\n")], "le tout premier mois"),
    ("decoupage.py", [("historique: int = HISTORIQUE", "historique: int = 6")], "valeur par défaut"),
    # Étape 3
    ("evaluation.py", [("colonne = list(modele.classes_).index(1)", "colonne = 1")], "classes_"),
    ("evaluation.py", [("modele.predict_proba(test[list(VARIABLES)])", "modele.predict_proba(test)")], "LA RÉPONSE"),
    ("evaluation.py", [("alerte = proba >= seuil", "alerte = proba > seuil")], "pile au seuil"),
    ("evaluation.py", [("alerte = proba >= seuil", "alerte = proba >= SEUIL")], "seuil=0.7"),
    ("evaluation.py", [("fausses_alertes) / len(y),", "fausses_alertes),")], "coût TOTAL"),
    ("evaluation.py", [("(COUT_FN * manques + COUT_FP * fausses_alertes)", "(COUT_FP * manques + COUT_FN * fausses_alertes)")],
     "COUT_FN × partants"),
    ("evaluation.py", [("if len(mois) != 1:", "if len(mois) == 0:")], "DEUX mois"),
    # Étape 4
    ("decision.py", [("if gain < regles.gain_auc_min - TOLERANCE:", "if gain < regles.gain_auc_min:")], "exactement 0,01"),
    ("decision.py", [("if challenger.auc < regles.auc_min - TOLERANCE:",
                      "if champion is not None and challenger.auc < regles.auc_min - TOLERANCE:")],
     "même sous le plancher"),
    ("decision.py", [("gain = challenger.auc - champion.auc", "gain = challenger.auc / champion.auc - 1")], "RELATIF"),
    ("decision.py", [("hausse = challenger.cout - champion.cout", "hausse = champion.cout - challenger.cout")],
     "inverse le sens"),
    ("decision.py", [("    raisons: list[str] = []\n", "    regles = REGLES\n    raisons: list[str] = []\n")],
     "ignore les règles"),
    ("decision.py", [("            raisons.append(\"aucun champion en production : le premier modèle utilisable est promu\")",
                      "            pass")], "TUPLE d'au moins une"),
    ("decision.py", [("return Decision(promouvoir, tuple(raisons))", "return Decision(promouvoir, raisons)")], "TUPLE"),
    ("decision.py", [("if champion is not None and champion.mois != challenger.mois:", "if False:")],
     "deux mois différents"),
    # Étape 5
    ("registre.py", [("    tampon = io.BytesIO()\n    joblib.dump(modele, tampon)\n    contenu = tampon.getvalue()\n",
                      "    joblib.dump(modele, dossier / fichier)\n    contenu = (dossier / fichier).read_bytes()\n")],
     "tronqué"),
    ("registre.py", [('numero = max((v["numero"] for v in index["versions"]), default=0) + 1',
                      'numero = len(index["versions"]) + 1')], "le plus grand numéro"),
    ("registre.py", [("    contenu = (dossier / entree[\"fichier\"]).read_bytes()\n",
                      "    joblib.load(dossier / entree[\"fichier\"])\n"
                      "    contenu = (dossier / entree[\"fichier\"]).read_bytes()\n")], "ModeleAltere|piégé"),
    ("registre.py", [("    if hashlib.sha256(contenu).hexdigest() != entree[\"sha256\"]:\n", "    if False:\n")],
     "doit lever ModeleAltere"),
    ("registre.py", [("    index[\"versions\"].append(version.vers_dict())\n",
                      "    index[\"versions\"].append(version.vers_dict())\n    index[\"champions\"].append(numero)\n")],
     "n'est PAS promouvoir"),
    ("registre.py", [("    if index[\"champions\"] and index[\"champions\"][-1] == numero:\n        return",
                      "    if False:\n        return")], "deux fois"),
    ("registre.py", [("if len(index[\"champions\"]) < 2:", "if not index[\"champions\"]:")], "un seul champion"),
    ("registre.py", [("    _entree(index, numero)                 # une version inconnue", "    # une version inconnue")],
     r"promouvoir\(dossier, 9\)"),
    # Étape 6
    ("cli.py", [("mesures_champion = mesurer(charger(args.registre, actuel.numero), test) if actuel else None",
                 "mesures_champion = actuel.mesures if actuel else None")], "RE-MESURÉ"),
    ("cli.py", [(_ENREGISTRER_PUIS_PROMOUVOIR,
                 "    if not decision.promouvoir:\n        return 0\n"
                 "    version = enregistrer(args.registre, modele, mesures)\n"
                 "    promouvoir(args.registre, version.numero)\n")], "CHAQUE challenger"),
    ("cli.py", [("    if decision.promouvoir:\n        promouvoir(", "    if True:\n        promouvoir(")],
     "le champion devrait être"),
    ("cli.py", [("    if trouves:\n", "    if False:\n")], "refuser le lot"),
    ("cli.py", [("            print(f\"  - {probleme}\", file=sys.stderr)", "            pass")], "sans dire pourquoi"),
    ("cli.py", [("except (ValueError, ModeleAltere, OSError) as exc:", "except (ModeleAltere, OSError) as exc:")],
     "plante avec ValueError"),
    ("cli.py", [("print(f\"Erreur : {exc}\", file=sys.stderr)", "print(f\"Erreur : {exc}\")")], "stderr"),
    ("cli.py", [("lancez d'abord `entrainer`.\")\n        return 0", "lancez d'abord `entrainer`.\")\n        return 1")],
     "registre est vide"),
    # Étape 7
    ("mes_tests.py", [("    assert decider(_mesures(0.82), _mesures(0.81)).promouvoir\n", "")], "exactement 0,01"),
    ("mes_tests.py", [("    try:\n        decider(_mesures(0.80, mois=\"2026-09\"), _mesures(0.75, mois=\"2026-08\"))\n"
                       "    except ValueError:\n        pass\n    else:\n"
                       "        raise AssertionError(\"champion et challenger mesurés sur deux mois différents : decider "
                       "doit lever ValueError\")\n", "")], "deux mois différents"),
    ("mes_tests.py", [("    assert not decider(_mesures(0.65), None).promouvoir\n", ""),
                      ("    assert not decider(_mesures(0.85), None, strictes).promouvoir\n", "")],
     "même sous le plancher"),
]


@pytest.mark.parametrize("fichier,remplacements,motif", ERREURS_CLASSIQUES,
                         ids=[f"{f}-{m}" for f, _, m in ERREURS_CLASSIQUES])
def test_chaque_erreur_classique_est_expliquee(tmp_path, monkeypatch, fichier, remplacements, motif):
    resultat = _variante(tmp_path, monkeypatch, fichier, remplacements)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    assert re.search(motif, resultat.message), resultat.message


def _faux_module(**fonctions) -> types.ModuleType:
    module = types.ModuleType("projets.reentrainement_corrige.faux")
    module.__package__ = "projets.reentrainement_corrige"
    module.__dict__.update(fonctions)
    return module


@pytest.mark.parametrize("mutant,quoi,conseil", checks._MUTANTS_DECIDER, ids=[d for d, _, _ in checks._DEFAUTS_DECIDER])
def test_etape4_chaque_version_boguee_est_reconnue(mutant, quoi, conseil):
    # Celle qui s'arrête à la première règle décide juste : seul le nombre de raisons la trahit.
    motif = "une raison par règle" if quoi.startswith("s'arrête") else re.escape(quoi)
    with pytest.raises(AssertionError, match=motif):
        checks.verifier_decider(_faux_module(decider=mutant))


def test_etape4_chaque_version_boguee_differe_de_la_bonne():
    m = checks._modeles(decision)
    scenarios = checks._scenarios(m)
    juste = [checks._issue(decision.decider, args) for _, args in scenarios]
    vues = set()
    for mutant, quoi, _ in checks._MUTANTS_DECIDER:
        issues = tuple(checks._issue_mutant(mutant, args) for _, args in scenarios)
        assert list(issues) != juste or quoi.startswith("s'arrête"), quoi
        assert issues not in vues or quoi.startswith("s'arrête"), f"deux versions boguées indiscernables : {quoi}"
        vues.add(issues)


def test_etape7_un_test_trop_faible_est_refuse():
    def tester_faible(decider):  # ne teste que le cas sans champion
        assert decider(Mesures("2026-09", 500, 0.80, 50.0, 0.3), None).promouvoir
        assert not decider(Mesures("2026-09", 500, 0.65, 50.0, 0.3), None).promouvoir

    with pytest.raises(AssertionError, match="laisse passer"):
        checks.verifier_tester_decider(_faux_module(tester_decider=tester_faible))

    def tester_rien(decider):
        decider(Mesures("2026-09", 500, 0.80, 50.0, 0.3), None)

    with pytest.raises(AssertionError, match="TOUTES les versions boguées"):
        checks.verifier_tester_decider(_faux_module(tester_decider=tester_rien))

    def tester_faux(decider):  # oublie le plancher : croit qu'un premier modèle est toujours promu
        assert decider(Mesures("2026-09", 500, 0.60, 50.0, 0.3), None).promouvoir

    with pytest.raises(AssertionError, match="version JUSTE"):
        checks.verifier_tester_decider(_faux_module(tester_decider=tester_faux))


# ---------------------------------------------------------------------------- D'autres solutions justes passent

SOLUTIONS_DIFFERENTES = [
    ("lots.py", [("~mois.str.fullmatch(_FORMAT_MOIS)", "~mois.str.match(\"^\" + _FORMAT_MOIS + \"$\")")]),
    ("lots.py", [("lots.duplicated([MOIS, IDENTIFIANT], keep=\"first\")",
                  "lots.duplicated(subset=[IDENTIFIANT, MOIS])")]),
    ("decoupage.py", [("    periodes = pd.PeriodIndex(lots[MOIS], freq=\"M\")\n", ""),
                      ("    debut = test_p - historique\n", "    debut = str(test_p - historique)\n"),
                      ("    dans_test = periodes == test_p\n", "    dans_test = lots[MOIS] == mois_test\n"),
                      (_TRAIN, "dans_train = (lots[MOIS] >= debut) & (lots[MOIS] < mois_test)")]),
    ("evaluation.py", [("cout=(COUT_FN * manques + COUT_FP * fausses_alertes) / len(y),",
                        "cout=float(np.mean(np.where((y == 1) & ~alerte, COUT_FN, 0.0) "
                        "+ np.where((y == 0) & alerte, COUT_FP, 0.0))),")]),
    ("decision.py", [("TOLERANCE = 1e-9", "TOLERANCE = 1e-12")]),
    ("registre.py", [("    descripteur, temporaire = tempfile.mkstemp(dir=dossier, prefix=\".modele-\", suffix=\".tmp\")\n"
                      "    try:\n        with os.fdopen(descripteur, \"wb\") as sortie:\n",
                      "    temporaire = dossier / f\".{fichier}.tmp\"\n"
                      "    try:\n        with open(temporaire, \"wb\") as sortie:\n")]),
    ("cli.py", [("print(f\"✅ v{version.numero:03d} promue championne :\")",
                 "print(f\"La version v{version.numero:03d} part en production.\")"),
                ("        marque = \"  ← champion\" if", "        marque = \"  (champion actuel)\" if")]),
]


@pytest.mark.parametrize("fichier,remplacements", SOLUTIONS_DIFFERENTES,
                         ids=[f"{f}-{i}" for i, (f, _) in enumerate(SOLUTIONS_DIFFERENTES)])
def test_une_autre_solution_juste_est_acceptee(tmp_path, monkeypatch, fichier, remplacements):
    """Le correcteur juge le comportement, pas la ressemblance avec le corrigé."""
    resultat = _variante(tmp_path, monkeypatch, fichier, remplacements)
    assert resultat.statut == REUSSI, resultat.message

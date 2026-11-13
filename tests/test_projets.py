"""Garantit que le projet guidé est faisable et que sa correction automatique est fiable.

Ces tests NE notent PAS votre projet (pour ça : `python scripts/projet.py`) ; ils vérifient le
corrigé, le moteur de correction et que les erreurs classiques sont bien détectées.
"""

import csv
import math
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projets import checks_retention as checks  # noqa: E402
from projets.retention_corrige import allocation, cli, clients, rapport, regles, valeur  # noqa: E402
from projets.retention_corrige.modeles import CONTRATS, Action, Client, Offre  # noqa: E402
from projets.verif import A_FAIRE, ECHEC, ERREUR, REUSSI, evaluer, evaluer_etape, projets  # noqa: E402

PROJET = checks.PROJET
EXEMPLE = ROOT / "projets" / "exemple_clients.csv"


def _faux_module(**fonctions) -> types.ModuleType:
    """Un module « comme celui de l'apprenant·e », qui utilise les classes de modeles.py du corrigé."""
    module = types.ModuleType("projets.retention_corrige.faux")
    module.__package__ = "projets.retention_corrige"
    module.__dict__.update(fonctions)
    return module


def _echoue(verifier, module, motif: str) -> None:
    with pytest.raises(AssertionError, match=motif):
        verifier(module)


# ---------------------------------------------------------------------------- Le projet est cohérent


@pytest.mark.parametrize("etape", PROJET.etapes, ids=lambda e: e.module)
def test_le_corrige_reussit(etape):
    resultat = evaluer_etape(PROJET.paquet_corrige, etape, PROJET.etapes)
    assert resultat.statut == REUSSI, resultat.message


def test_le_squelette_est_entierement_a_faire():
    """Tel qu'il est livré, le code de l'apprenant·e importe sans erreur et tout est ⬜."""
    assert [r.statut for r in evaluer(PROJET)] == [A_FAIRE] * len(PROJET.etapes)


@pytest.mark.parametrize("fichier", ["modeles.py", "__main__.py"])
def test_les_fichiers_deja_ecrits_sont_identiques(fichier):
    eleve = ROOT / "projets" / "retention" / fichier
    corrige = ROOT / "projets" / "retention_corrige" / fichier
    assert eleve.read_text(encoding="utf-8") == corrige.read_text(encoding="utf-8")


def test_les_projets_sont_enregistres():
    assert [p.code for p in projets()] == ["retention", "campagne"]
    assert [p.numero for p in projets()] == [2, 3]
    assert len({e.numero for e in PROJET.etapes}) == len(PROJET.etapes)


def test_l_exemple_est_un_export_valide():
    lus, erreurs = clients.lire_clients(EXEMPLE)
    assert erreurs == []
    assert len(lus) >= 100
    assert {c.contract for c in lus} == set(CONTRATS)
    assert all(0 <= c.proba_churn <= 1 for c in lus)


# ---------------------------------------------------------------------------- Le programme terminé


def _python(*args, cwd=ROOT):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          timeout=120)


def test_le_programme_termine_produit_un_plan_dans_le_budget(tmp_path):
    plan = tmp_path / "plan.csv"
    res = _python("-m", "projets.retention_corrige", "--clients", str(EXEMPLE), "--budget", "1500",
                  "--sortie", str(plan))
    assert res.returncode == 0, res.stderr
    assert "Clients ciblés" in res.stdout and str(plan) in res.stdout
    with plan.open(encoding="utf-8", newline="") as f:
        lignes = list(csv.DictReader(f))
    assert lignes, "le plan ne devrait pas être vide avec 1 500 €"
    assert sum(float(x["cout"]) for x in lignes) <= 1500
    assert all(float(x["gain_attendu"]) > 0 for x in lignes)
    assert len({x["customer_id"] for x in lignes}) == len(lignes), "un client ciblé deux fois"


def test_le_squelette_explique_qu_il_n_est_pas_termine(tmp_path):
    res = _python("-m", "projets.retention", "--clients", str(EXEMPLE), "--budget", "1500",
                  "--sortie", str(tmp_path / "plan.csv"))
    assert res.returncode == 1
    assert "scripts/projet.py" in res.stderr and "Traceback" not in res.stderr


def test_le_script_de_progression(tmp_path):
    res = _python("scripts/projet.py", "--corrige")
    assert res.returncode == 0 and "7/7" in res.stdout and "Projet terminé" in res.stdout
    res = _python("scripts/projet.py")
    assert res.returncode == 0 and "0/7" in res.stdout and "Prochaine étape : 1" in res.stdout


def test_un_budget_plus_grand_ne_rapporte_jamais_moins():
    lus, _ = clients.lire_clients(EXEMPLE)
    candidates = [a for c in lus if (a := regles.meilleure_offre(c)) is not None]
    gains = [sum(a.gain_attendu for a in allocation.allouer_budget(candidates, b)) for b in (0, 200, 1000, 5000, 1e9)]
    assert gains == sorted(gains) and gains[0] == 0
    assert math.isclose(gains[-1], sum(a.gain_attendu for a in candidates))


# ---------------------------------------------------------------------------- Le moteur de correction


def _copie_du_squelette(tmp_path, monkeypatch, nom: str) -> Path:
    dossier = tmp_path / nom
    shutil.copytree(ROOT / "projets" / "retention", dossier, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.syspath_prepend(str(tmp_path))
    return dossier


def test_une_etape_qui_depend_d_une_etape_non_ecrite(tmp_path, monkeypatch):
    """regles.py est écrit, mais pas valeur.py : on le dit, au lieu d'afficher « à faire »."""
    dossier = _copie_du_squelette(tmp_path, monkeypatch, "eleve_dependance")
    shutil.copy(ROOT / "projets" / "retention_corrige" / "regles.py", dossier / "regles.py")
    resultat = evaluer_etape("eleve_dependance", PROJET.etapes[2], PROJET.etapes)
    assert resultat.statut == ECHEC
    assert "étape 2" in resultat.message and "gain_attendu" in resultat.message


def test_une_erreur_de_syntaxe_est_signalee(tmp_path, monkeypatch):
    dossier = _copie_du_squelette(tmp_path, monkeypatch, "eleve_syntaxe")
    (dossier / "valeur.py").write_text("def perte_attendue(client:\n", encoding="utf-8")
    resultat = evaluer_etape("eleve_syntaxe", PROJET.etapes[1], PROJET.etapes)
    assert resultat.statut == ERREUR and "SyntaxError" in resultat.message


def test_un_plantage_est_situe_dans_le_code_de_l_apprenant(tmp_path, monkeypatch):
    dossier = _copie_du_squelette(tmp_path, monkeypatch, "eleve_plantage")
    shutil.copy(ROOT / "projets" / "retention_corrige" / "valeur.py", dossier / "valeur.py")
    texte = (dossier / "valeur.py").read_text(encoding="utf-8")
    (dossier / "valeur.py").write_text(texte.replace("client.proba_churn *", "client.proba *"), encoding="utf-8")
    resultat = evaluer_etape("eleve_plantage", PROJET.etapes[1], PROJET.etapes)
    assert resultat.statut == ERREUR
    assert "AttributeError" in resultat.message and "valeur.py, ligne" in resultat.message


# ---------------------------------------------------------------------------- Étape 1 : erreurs classiques


def _lire_bogue(*, encodage="utf-8-sig", nan_accepte=False, debut=None, s_arrete=False, none_plante=False):
    def lire_clients(chemin):
        with Path(chemin).open(encoding=encodage, newline="") as f:
            lecteur = csv.DictReader(f)
            manquantes = [c for c in checks.ref_clients.COLONNES_REQUISES if c not in (lecteur.fieldnames or [])]
            if manquantes:
                raise ValueError(f"colonnes manquantes : {', '.join(manquantes)}")
            lus, erreurs, vus = [], [], set()
            for i, ligne in enumerate(lecteur):
                numero = lecteur.line_num if debut is None else debut + i
                if none_plante:
                    ligne = {k: v.strip() for k, v in ligne.items()}
                try:
                    if nan_accepte and ligne["proba_churn"] == "nan":
                        ligne = {**ligne, "proba_churn": "0.5"}
                    client = clients._client(ligne)
                except ValueError as exc:
                    if s_arrete:
                        raise
                    erreurs.append(f"ligne {numero} : {exc}")
                    continue
                if client.customer_id in vus:
                    erreurs.append(f"ligne {numero} : {client.customer_id} en double")
                    continue
                vus.add(client.customer_id)
                lus.append(client)
        return lus, erreurs

    return lire_clients


@pytest.mark.parametrize("options,motif", [
    ({"encodage": "utf-8"}, "BOM"),
    ({"nan_accepte": True}, "nan"),
    ({"debut": 1}, "décalés"),
    ({"s_arrete": True}, "IGNORÉE et SIGNALÉE"),
    ({"none_plante": True}, "None"),
])
def test_etape1_erreurs_classiques(options, motif):
    _echoue(checks.verifier_lire_clients, _faux_module(lire_clients=_lire_bogue(**options)), motif)


# ---------------------------------------------------------------------------- Étape 2


def test_etape2_erreurs_classiques():
    def perte_12(client, horizon=12):
        return client.proba_churn * client.monthly_charges * 12

    def perte_arrondie(client, horizon=12):
        return round(client.proba_churn * client.monthly_charges * horizon, 2)

    def gain_plancher(client, offre, horizon=12):
        return max(0.0, valeur.gain_attendu(client, offre, horizon))

    def gain_recopie(client, offre, horizon=12):
        return offre.efficacite * client.proba_churn * client.monthly_charges * horizon - offre.cout

    _echoue(checks.verifier_valeur, _faux_module(perte_attendue=perte_12, gain_attendu=valeur.gain_attendu), "en dur")
    _echoue(checks.verifier_valeur, _faux_module(perte_attendue=perte_arrondie, gain_attendu=valeur.gain_attendu),
            "N'arrondissez pas")
    _echoue(checks.verifier_valeur, _faux_module(perte_attendue=valeur.perte_attendue, gain_attendu=gain_plancher),
            "NÉGATIF")
    _echoue(checks.verifier_valeur, _faux_module(perte_attendue=valeur.perte_attendue, gain_attendu=gain_recopie),
            "recopie")


# ---------------------------------------------------------------------------- Étape 3


def test_etape3_erreurs_classiques():
    def eligibles_tuple_vide(client, offres=checks._modeles(regles).OFFRES):
        return [o for o in offres if client.contract in o.contrats
                and client.support_calls >= o.appels_min and client.tenure_months >= o.anciennete_min]

    def eligibles_strict(client, offres=checks._modeles(regles).OFFRES):
        return [o for o in offres if (not o.contrats or client.contract in o.contrats)
                and client.support_calls > o.appels_min and client.tenure_months >= o.anciennete_min]

    def meilleure_sans_horizon(client, offres=checks._modeles(regles).OFFRES, horizon=12):
        return regles.meilleure_offre(client, offres)

    def meilleure_sans_departage(client, offres=checks._modeles(regles).OFFRES, horizon=12):
        actions = [Action(client, o, valeur.gain_attendu(client, o, horizon)) for o in regles.offres_eligibles(client, offres)]
        actions = [a for a in actions if a.gain_attendu > 0]
        return max(actions, key=lambda a: a.gain_attendu) if actions else None

    ok = {"offres_eligibles": regles.offres_eligibles, "meilleure_offre": regles.meilleure_offre}
    _echoue(checks.verifier_regles, _faux_module(**{**ok, "offres_eligibles": eligibles_tuple_vide}), "tuple vide")
    _echoue(checks.verifier_regles, _faux_module(**{**ok, "offres_eligibles": eligibles_strict}), ">=")
    _echoue(checks.verifier_regles, _faux_module(**{**ok, "meilleure_offre": meilleure_sans_horizon}), "horizon")
    _echoue(checks.verifier_regles, _faux_module(**{**ok, "meilleure_offre": meilleure_sans_departage}), "MOINS CHÈRE")


# ---------------------------------------------------------------------------- Étapes 4 et 7 : les mêmes bugs


@pytest.mark.parametrize("mutant,motif", [
    (checks._mutant_break, "break"),
    (checks._mutant_par_gain, "GAIN"),
    (checks._mutant_negatifs, "strictement positifs"),
    (checks._mutant_strict, "<="),
    (checks._mutant_sans_budget, "dépensez"),
    (checks._mutant_modifie_entree, "MODIFIÉ"),
    (checks._mutant_budget_negatif, "ValueError"),
])
def test_etape4_chaque_bug_classique_est_explique(mutant, motif):
    _echoue(checks.verifier_allocation, _faux_module(allouer_budget=mutant), motif)


def test_etape7_un_test_trop_faible_est_refuse():
    def un_seul_exemple(allouer):
        a = checks._Fabrique(regles)("A", 10.0, 50.0)
        assert allouer([a], 10.0) == [a]  # n'attrape que la comparaison stricte (<)

    def rien(allouer):
        allouer([], 10.0)

    def faux(allouer):
        a = checks._Fabrique(regles)("A", 10.0, 50.0)
        assert allouer([a], 100.0) == []

    _echoue(checks.verifier_tester_allouer_budget, _faux_module(tester_allouer_budget=un_seul_exemple), "bugs attrapés")
    _echoue(checks.verifier_tester_allouer_budget, _faux_module(tester_allouer_budget=rien), "TOUTES")
    _echoue(checks.verifier_tester_allouer_budget, _faux_module(tester_allouer_budget=faux), "version JUSTE")


# ---------------------------------------------------------------------------- Étape 5


def test_etape5_erreurs_classiques(tmp_path):
    def resumer_double_cout(actions, nb_clients):
        r = rapport.resumer(actions, nb_clients)
        return {**r, "roi": (r["gain_attendu"] - r["cout_total"]) / r["cout_total"]} if actions else r

    def resumer_division(actions, nb_clients):
        r = rapport.resumer(actions, nb_clients)
        return {**r, "roi": r["gain_attendu"] / r["cout_total"]}

    def ecrire_sans_dossier(actions, chemin):
        with Path(chemin).open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return Path(chemin)

    def ecrire_libelle(actions, chemin):
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with chemin.open("w", encoding="utf-8", newline="") as f:
            ecrivain = csv.writer(f)
            ecrivain.writerow(checks.ref_rapport.COLONNES_PLAN)
            for a in actions:
                ecrivain.writerow([a.client.customer_id, a.offre.libelle, a.offre.cout, a.client.proba_churn,
                                   a.gain_attendu])
        return chemin

    ok = {"resumer": rapport.resumer, "ecrire_plan": rapport.ecrire_plan}
    _echoue(checks.verifier_rapport, _faux_module(**{**ok, "resumer": resumer_double_cout}), "DÉJÀ net")
    _echoue(checks.verifier_rapport, _faux_module(**{**ok, "resumer": resumer_division}), "ZeroDivisionError")
    _echoue(checks.verifier_rapport, _faux_module(**{**ok, "ecrire_plan": ecrire_sans_dossier}), "mkdir")
    _echoue(checks.verifier_rapport, _faux_module(**{**ok, "ecrire_plan": ecrire_libelle}), "CODE de l'offre")


# ---------------------------------------------------------------------------- Étape 6


def test_etape6_erreurs_classiques():
    def main_sans_return(argv=None):
        cli.main(argv)

    def main_budget_float(argv=None):
        if "1500,5" in argv:
            raise SystemExit(2)  # ce que fait argparse avec type=float sur « 1500,5 »
        return cli.main(argv)

    def main_sans_try(argv=None):
        args = cli.creer_parser().parse_args(argv)
        lus, _ = clients.lire_clients(args.clients)
        return cli.main(argv)

    _echoue(checks.verifier_cli, _faux_module(main=main_sans_return), "return 0")
    _echoue(checks.verifier_cli, _faux_module(main=main_budget_float), "française")
    _echoue(checks.verifier_cli, _faux_module(main=main_sans_try), "FileNotFoundError")


def test_etape6_la_reference_depend_bien_du_budget_et_de_l_horizon(tmp_path):
    """Les fichiers du correcteur sont choisis pour que chaque option change le plan."""
    chemin = checks._ecrire(tmp_path, "c.csv", checks._CLIENTS_CLI)
    plan_150, plan_150_h6, plan_large = (checks._plan_attendu(chemin, 150.0), checks._plan_attendu(chemin, 150.0, 6),
                                         checks._plan_attendu(chemin, 10_000.0))
    assert plan_150 != plan_150_h6
    assert len(plan_150) < len(plan_large)


def test_les_classes_partagees_sont_figees():
    client = Client("C1", 0.5, 50.0, 12, "Mensuel", 0)
    with pytest.raises(AttributeError):
        client.proba_churn = 0.1  # type: ignore[misc]
    action = Action(client, Offre("X", "x", 10.0, 0.5), 40.0)
    assert action.rentabilite == 4.0


def test_le_coach_affiche_la_progression_du_projet(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import progression

    assert progression.projets_scores(tmp_path) == []  # un dépôt sans dossier projets/
    retention, campagne = progression.projets_scores(ROOT)
    assert (retention.titre, retention.total, retention.erreur) == ("Projet 2 — Plan de rétention", 7, "")
    assert (campagne.titre, campagne.total, campagne.erreur) == ("Projet 3 — Mesurer l'effet de la campagne", 7, "")
    rendu = progression.render(ROOT)
    assert "Projet 2 — Plan de rétention : 0/7 étapes" in rendu
    assert "Projet 3 — Mesurer l'effet de la campagne : 0/7 étapes" in rendu

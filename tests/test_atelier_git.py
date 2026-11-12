"""Garantit que l'atelier Git est faisable et que sa correction automatique est fiable.

Ces tests NE notent PAS votre travail (pour ça : `python scripts/atelier_git.py`) : ils préparent
chaque scénario dans un dossier temporaire, appliquent la solution (qui doit donner ✅), puis les
erreurs classiques (qui doivent donner ❌ avec la bonne explication).
"""

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atelier_git.atelier import (A_FAIRE, ECHEC, REUSSI, SCENARIOS, dossier, evaluer, preparer,  # noqa: E402
                                 scenario)
from atelier_git.outils import Depot, preparer_racine, supprimer  # noqa: E402
from atelier_git.scenarios import message_vague  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git n'est pas installé")


@pytest.fixture(autouse=True)
def git_isole(tmp_path_factory, monkeypatch):
    """Ignore la configuration git de la machine (identité, alias, hooks…) : comme un·e débutant·e."""
    vide = tmp_path_factory.mktemp("config") / "gitconfig"
    vide.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(vide))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_EDITOR", "true")
    monkeypatch.setenv("GIT_PAGER", "cat")


@pytest.fixture
def racine(tmp_path):
    return tmp_path / "atelier-git"


def _lancer(depot: Path, commande: str, peut_echouer: bool = False) -> subprocess.CompletedProcess:
    argv = shlex.split(commande)
    if argv[0] == "python":
        argv[0] = sys.executable
    res = subprocess.run(argv, cwd=depot, capture_output=True, text=True, env=os.environ.copy())
    if res.returncode and not peut_echouer:
        raise AssertionError(f"`{commande}` a échoué ({res.returncode}) : {res.stdout}\n{res.stderr}")
    return res


def _preparer(numero: int, racine: Path) -> tuple[object, Path]:
    s = scenario(numero)
    return s, preparer(s, racine)


def _resoudre(s, racine: Path, jusqua: int | None = None) -> None:
    """Applique les étapes de la solution affichée par `--solution` (toutes, ou les `jusqua` premières)."""
    depot = Depot(dossier(s, racine))
    for pas in s.solution[:jusqua]:
        if pas.action is not None:
            pas.action(depot)
        else:
            _lancer(depot.chemin, pas.texte, pas.peut_echouer)


def _echoue(s, racine: Path, motif: str) -> None:
    resultat = evaluer(s, racine)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    assert motif in resultat.message, resultat.message


# ---------------------------------------------------------------------------- Chaque scénario est faisable


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.nom_dossier)
def test_un_scenario_prepare_est_a_faire(s, racine):
    travail = preparer(s, racine)
    assert (travail / "MISSION.md").exists()
    assert "MISSION.md" not in "\n".join(Depot(travail).statut()), "MISSION.md doit être ignoré par git"
    resultat = evaluer(s, racine)
    assert resultat.statut == A_FAIRE and "préparé" in resultat.message


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.nom_dossier)
def test_la_solution_affichee_reussit(s, racine):
    preparer(s, racine)
    _resoudre(s, racine)
    resultat = evaluer(s, racine)
    assert resultat.statut == REUSSI, resultat.message


@pytest.mark.parametrize("numero", [2, 4, 5, 8])
def test_la_solution_reussit_avec_une_config_git_de_windows(numero, racine, tmp_path, monkeypatch):
    """Git pour Windows convertit les fins de ligne (autocrlf) : la correction ne doit pas s'y tromper."""
    config = tmp_path / "gitconfig-windows"
    config.write_text("[core]\n\tautocrlf = true\n[merge]\n\tconflictStyle = zdiff3\n"
                      "[init]\n\tdefaultBranch = master\n[pull]\n\trebase = true\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    s, _ = _preparer(numero, racine)
    assert evaluer(s, racine).statut == A_FAIRE
    _resoudre(s, racine)
    resultat = evaluer(s, racine)
    assert resultat.statut == REUSSI, resultat.message


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.nom_dossier)
def test_les_textes_sont_complets(s):
    assert s.situation and s.a_retenir and len(s.mission) >= 2
    assert len(s.indices) >= 3, "au moins 3 indices, du plus léger au plus direct"
    assert any(p.est_commande for p in s.solution)


def test_non_prepare_et_recommencer(racine):
    s = scenario(1)
    assert evaluer(s, racine).statut == A_FAIRE
    _preparer(1, racine)
    _resoudre(s, racine)
    assert evaluer(s, racine).statut == REUSSI
    preparer(s, racine)  # --recommencer
    assert evaluer(s, racine).statut == A_FAIRE


def test_le_bac_a_sable_ne_touche_pas_a_un_dossier_etranger(tmp_path):
    etranger = tmp_path / "mes-documents"
    etranger.mkdir()
    (etranger / "important.txt").write_text("ne pas effacer")
    with pytest.raises(RuntimeError, match="n'a pas été créé par l'atelier"):
        preparer_racine(etranger)
    racine = tmp_path / "atelier-git"
    preparer_racine(racine)
    with pytest.raises(RuntimeError, match="Refus de supprimer"):
        supprimer(etranger, racine)
    assert (etranger / "important.txt").exists()


@pytest.mark.parametrize("message, vague", [
    ("wip", True), ("WIP tests", True), ("fix typo", True), ("update", True), ("modifs", True), ("Fix.", True),
    ("Corrige la division par zéro de taux_churn", False), ("Ajoute l'export Excel des clients", False),
])
def test_message_vague(message, vague):
    assert message_vague(message) is vague


# ---------------------------------------------------------------------------- Les erreurs classiques sont expliquées


def test_1_git_add_point_embarque_les_fichiers_indesirables(racine):
    s, travail = _preparer(1, racine)
    _lancer(travail, "git add .")
    _lancer(travail, 'git commit -m "Corrige la division par zéro de taux_churn"')
    _echoue(s, racine, "debug.log")


def test_1_test_oublie(racine):
    s, travail = _preparer(1, racine)
    (travail / ".gitignore").write_text("*.log\n__pycache__/\n")
    _lancer(travail, "git add churn.py .gitignore")
    _lancer(travail, 'git commit -m "Corrige la division par zéro de taux_churn"')
    _echoue(s, racine, "test_churn.py n'est pas commité")


def test_1_ignore_seulement_en_local(racine):
    s, travail = _preparer(1, racine)
    (travail / ".gitignore").write_text("*.log\n")
    with (travail / ".git" / "info" / "exclude").open("a") as f:
        f.write("__pycache__/\n")
    _lancer(travail, "git add churn.py test_churn.py .gitignore")
    _lancer(travail, 'git commit -m "Corrige la division par zéro de taux_churn"')
    _echoue(s, racine, "info/exclude")


def test_1_message_vague(racine):
    s, travail = _preparer(1, racine)
    (travail / ".gitignore").write_text("*.log\n__pycache__/\n")
    _lancer(travail, "git add churn.py test_churn.py .gitignore")
    _lancer(travail, 'git commit -m "wip"')
    _echoue(s, racine, "« wip »")


def test_2_reculer_main_avant_de_creer_la_branche(racine):
    s, travail = _preparer(2, racine)
    _lancer(travail, "git reset --hard origin/main")
    _echoue(s, racine, "git reflog")


def test_2_pousser_sur_main(racine):
    s, travail = _preparer(2, racine)
    _lancer(travail, "git push origin main")
    _echoue(s, racine, "origin/main a changé")


def test_2_branche_non_publiee(racine):
    s, _ = _preparer(2, racine)
    _resoudre(s, racine, jusqua=4)  # tout sauf le push
    _echoue(s, racine, "git push -u origin remise-fidelite")


def test_3_stash_sans_u_puis_git_add_point(racine):
    s, travail = _preparer(3, racine)
    _lancer(travail, "git stash")
    _lancer(travail, "git switch main")
    (travail / "churn.py").write_text((travail / "churn.py").read_text().replace("SEUIL = 5 ", "SEUIL = 0.5 "))
    _lancer(travail, "git add .")
    _lancer(travail, 'git commit -m "Corrige le seuil de contact"')
    _echoue(s, racine, "git stash -u")


def test_3_travail_oublie_dans_la_remise(racine):
    s, _ = _preparer(3, racine)
    _resoudre(s, racine, jusqua=6)  # tout sauf `git stash pop`
    _echoue(s, racine, "git stash pop")


def test_4_garder_un_seul_camp(racine):
    s, travail = _preparer(4, racine)
    _lancer(travail, "git switch main")
    _lancer(travail, "git merge offre-engagement", peut_echouer=True)
    _lancer(travail, "git checkout --ours offres.py")
    _lancer(travail, "git add offres.py")
    _lancer(travail, "git commit --no-edit")
    _echoue(s, racine, "Il manque ENGAGEMENT")


def test_4_marqueurs_commites(racine):
    s, travail = _preparer(4, racine)
    _lancer(travail, "git switch main")
    _lancer(travail, "git merge offre-engagement", peut_echouer=True)
    _lancer(travail, "git add offres.py")
    _lancer(travail, "git commit --no-edit")
    _echoue(s, racine, "marqueurs de conflit")


def test_4_fusion_pas_terminee(racine):
    s, travail = _preparer(4, racine)
    _lancer(travail, "git switch main")
    _lancer(travail, "git merge offre-engagement", peut_echouer=True)
    _echoue(s, racine, "git merge --abort")


def test_5_reset_et_push_force(racine):
    s, travail = _preparer(5, racine)
    _lancer(travail, "git reset --hard HEAD~2")
    _lancer(travail, "git push --force")
    _echoue(s, racine, "réécrit")


def test_5_revert_non_pousse(racine):
    s, _ = _preparer(5, racine)
    _resoudre(s, racine, jusqua=2)
    _echoue(s, racine, "git push")


def test_6_tag_sur_un_commit_deja_casse(racine):
    s, travail = _preparer(6, racine)
    _lancer(travail, "git tag coupable HEAD")
    _echoue(s, racine, "PREMIER mauvais commit")


def test_6_tag_sur_un_commit_encore_bon(racine):
    s, travail = _preparer(6, racine)
    _lancer(travail, "git tag coupable v1.0")
    _echoue(s, racine, "passe encore")


def test_6_bisect_pas_termine(racine):
    s, _ = _preparer(6, racine)
    _resoudre(s, racine, jusqua=4)  # tout sauf `git bisect reset`
    _echoue(s, racine, "git bisect reset")


def test_6_bisect_trouve_le_commit_attendu(racine):
    """Le correcteur et `git bisect run` désignent bien le même commit (le renommage « anodin »)."""
    s, travail = _preparer(6, racine)
    _resoudre(s, racine)
    assert _lancer(travail, "git log -1 --format=%s coupable").stdout.strip() == "Harmonise les noms de variables"


def test_7_branche_recuperee_mais_pas_main(racine):
    s, _ = _preparer(7, racine)
    _resoudre(s, racine, jusqua=2)
    _echoue(s, racine, "main ne contient pas")


def test_8_merge_au_lieu_de_rebase(racine):
    s, travail = _preparer(8, racine)
    _lancer(travail, "git fetch")
    _lancer(travail, "git merge --no-edit origin/main")
    _echoue(s, racine, "commit de fusion")


def test_8_rebase_sans_regrouper(racine):
    s, travail = _preparer(8, racine)
    _lancer(travail, "git fetch")
    _lancer(travail, "git rebase origin/main")
    _echoue(s, racine, "4 commits")


def test_8_pas_encore_pousse(racine):
    s, _ = _preparer(8, racine)
    _resoudre(s, racine, jusqua=5)  # tout sauf le push
    _echoue(s, racine, "--force-with-lease")


# ---------------------------------------------------------------------------- La ligne de commande


def _cli():
    spec = importlib.util.spec_from_file_location("atelier_git_cli", ROOT / "scripts" / "atelier_git.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_parcours_complet(racine, capsys):
    cli = _cli()
    assert cli.main(["--dossier", str(racine)]) == 0
    assert "0/8 scénarios réussis" in capsys.readouterr().out

    assert cli.main(["1", "--dossier", str(racine)]) == 0
    sortie = capsys.readouterr().out
    assert "préparé dans" in sortie and "Mission" in sortie and "cd " in sortie

    _resoudre(scenario(1), racine)
    assert cli.main(["1", "--dossier", str(racine)]) == 0
    sortie = capsys.readouterr().out
    assert "✅ Scénario 1 réussi" in sortie and "À retenir" in sortie

    assert cli.main(["--dossier", str(racine)]) == 0
    assert "1/8 scénarios réussis" in capsys.readouterr().out


def test_cli_indices_et_solution(capsys):
    cli = _cli()
    assert cli.main(["4", "--indice"]) == 0
    assert "indice 1/4" in capsys.readouterr().out
    assert cli.main(["4", "--indice", "9"]) == 2
    assert cli.main(["4", "--solution"]) == 0
    assert "$ git merge offre-engagement" in capsys.readouterr().out
    assert cli.main(["42"]) == 2

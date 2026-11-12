"""Préparer un scénario dans le bac à sable, puis le corriger (vous n'avez pas besoin de lire ce fichier)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atelier_git.outils import (DOSSIER_DISTANTS, MARQUEUR_SCENARIO, Depot, ErreurGit, preparer_racine, supprimer,
                                verifier_git_installe)
from atelier_git.scenarios import SCENARIOS, Scenario
from defis.verif import A_FAIRE, ECHEC, ERREUR, ICONES, REUSSI  # mêmes statuts et icônes que les défis

__all__ = ["A_FAIRE", "ECHEC", "ERREUR", "ICONES", "REUSSI", "Resultat", "SCENARIOS", "dossier", "evaluer",
           "est_prepare", "mission_markdown", "preparer", "scenario"]


@dataclass(frozen=True)
class Resultat:
    scenario: Scenario
    statut: str
    message: str = ""


def scenario(numero: int) -> Scenario:
    for s in SCENARIOS:
        if s.numero == numero:
            return s
    raise KeyError(f"Aucun scénario n°{numero} (1 à {len(SCENARIOS)}).")


def dossier(s: Scenario, racine: Path) -> Path:
    """Le dépôt où vous travaillez : c'est là qu'il faut faire `cd`."""
    return Path(racine) / s.nom_dossier


def _distants(s: Scenario, racine: Path) -> Path:
    """Le « GitHub » et le clone de Camille du scénario : on n'y touche pas."""
    return Path(racine) / DOSSIER_DISTANTS / s.nom_dossier


def _marqueur(s: Scenario, racine: Path) -> Path:
    return dossier(s, racine) / ".git" / MARQUEUR_SCENARIO


def est_prepare(s: Scenario, racine: Path) -> bool:
    return _marqueur(s, racine).exists()


def _empreinte(s: Scenario, racine: Path) -> str:
    """Un résumé de l'état des dépôts : s'il n'a pas bougé depuis la préparation, rien n'a été fait."""
    depot = Depot(dossier(s, racine))
    morceaux = [depot.refs(), depot.git("symbolic-ref", "-q", "HEAD", codes_ok=(0, 1)) or depot.rev("HEAD") or "",
                "\n".join(depot.statut())]
    distants = _distants(s, racine)
    if distants.exists():
        morceaux += [Depot(chemin).refs() for chemin in sorted(distants.iterdir()) if chemin.is_dir()]
    return "\n--\n".join(morceaux)


def mission_markdown(s: Scenario, commande: str) -> str:
    """Le contenu de MISSION.md, posé dans le dossier du scénario (et ignoré par git)."""
    puces = "\n".join(f"- [ ] {m}" for m in s.mission)
    return (f"# Scénario {s.numero} {s.niveau} — {s.titre}\n\n"
            f"Vous pratiquez : {s.commandes}\n\n"
            f"## La situation\n\n{s.situation}\n\n"
            f"## La mission\n\n{puces}\n\n"
            "## Et ensuite\n\n"
            f"- Vérifier votre travail : `{commande} {s.numero}`\n"
            f"- Un indice : `{commande} {s.numero} --indice` (puis `--indice 2`, `--indice 3`…)\n"
            f"- Tout remettre à zéro : `{commande} {s.numero} --recommencer`\n\n"
            "Ce fichier est ignoré par git (`.git/info/exclude`) : il n'apparaît pas dans `git status`.\n")


def preparer(s: Scenario, racine: Path, commande: str = "python scripts/atelier_git.py") -> Path:
    """(Re)crée le scénario dans le bac à sable et renvoie le dossier où travailler."""
    verifier_git_installe()
    racine = Path(racine)
    preparer_racine(racine)
    travail, distants = dossier(s, racine), _distants(s, racine)
    if travail.exists() and any(travail.iterdir()) and not est_prepare(s, racine):
        raise RuntimeError(f"{travail} existe et n'a pas été créé par l'atelier : déplacez-le ou supprimez-le.")
    try:
        supprimer(travail, racine)
        supprimer(distants, racine)
    except PermissionError as exc:
        raise RuntimeError(f"Impossible d'effacer l'ancien bac à sable ({exc}). Un terminal ou un éditeur est "
                           "peut-être ouvert dedans : fermez-le (ou faites `cd ..`), puis relancez.") from exc
    s.preparer(travail, distants)
    depot = Depot(travail)
    (travail / "MISSION.md").write_text(mission_markdown(s, commande), encoding="utf-8")
    depot.exclure("MISSION.md")
    _marqueur(s, racine).write_text(_empreinte(s, racine), encoding="utf-8")
    return travail


def evaluer(s: Scenario, racine: Path) -> Resultat:
    """✅ mission remplie · ❌ pas encore (le message dit pourquoi) · ⬜ pas commencé · 💥 dépôt illisible."""
    if not est_prepare(s, racine):
        return Resultat(s, A_FAIRE, "pas encore préparé")
    try:
        if _empreinte(s, racine) == _marqueur(s, racine).read_text(encoding="utf-8"):
            return Resultat(s, A_FAIRE, "préparé : à vous de jouer")
        s.verifier(dossier(s, racine), _distants(s, racine))
    except AssertionError as exc:
        return Resultat(s, ECHEC, str(exc))
    except ErreurGit as exc:
        return Resultat(s, ERREUR, f"{exc}. Le dépôt est peut-être abîmé : --recommencer le remet à neuf.")
    except Exception as exc:  # noqa: BLE001 - on montre l'erreur au lieu de planter
        return Resultat(s, ERREUR, f"{type(exc).__name__} : {exc}")
    return Resultat(s, REUSSI)

"""Outils du correcteur de l'atelier Git (vous n'avez pas besoin de lire ce fichier).

Le correcteur fabrique de vrais dépôts git (avec de vrais commits, branches, dépôts distants) puis
les inspecte avec des commandes git en lecture seule : il ne modifie jamais votre travail.
"""

from __future__ import annotations

import itertools
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MARQUEUR_RACINE = ".atelier-git"            # posé à la racine du bac à sable
MARQUEUR_SCENARIO = "atelier_etat_initial.txt"  # posé dans le .git de chaque scénario préparé
DOSSIER_DISTANTS = ".distants"              # dépôts « GitHub » et clones des collègues


@dataclass(frozen=True)
class Auteur:
    nom: str
    email: str


VOUS = Auteur("Vous", "vous@atelier-git.invalid")
CAMILLE = Auteur("Camille (collègue)", "camille@atelier-git.invalid")

# Dates fictives et croissantes pour les commits fabriqués : `git log` reste dans l'ordre.
_DEBUT = 1_767_258_000  # 1er janvier 2026, 9 h (UTC)
_horloge = itertools.count()


class GitIntrouvable(RuntimeError):
    """git n'est pas installé (ou pas dans le PATH)."""


class ErreurGit(RuntimeError):
    """Une commande git a échoué de façon inattendue."""


def verifier_git_installe() -> None:
    if shutil.which("git") is None:
        raise GitIntrouvable(
            "git est introuvable. Installez-le (https://git-scm.com/downloads), fermez et rouvrez le "
            "terminal, puis vérifiez avec : git --version")


def normaliser(texte: str | None) -> str | None:
    """Fins de ligne Windows (CRLF) et espaces en fin de ligne ignorés pour comparer deux fichiers."""
    if texte is None:
        return None
    lignes = texte.replace("\r\n", "\n").split("\n")
    return "\n".join(ligne.rstrip() for ligne in lignes).strip("\n")


class Depot:
    """Un dépôt git du bac à sable, piloté par le correcteur."""

    def __init__(self, chemin: Path, fabrication: bool = False):
        self.chemin = Path(chemin)
        # En fabrication (préparation d'un scénario), on neutralise la configuration git de la machine
        # (signature GPG, conversion des fins de ligne…). En vérification, git se comporte comme chez vous.
        self.fabrication = fabrication

    # ------------------------------------------------------------------ Exécution

    def git(self, *args: str, auteur: Auteur | None = None, codes_ok: tuple[int, ...] = (0,)) -> str:
        """Lance `git <args>` dans le dépôt et renvoie la sortie standard (sans espaces autour)."""
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_EDITOR": "true", "GIT_PAGER": "cat",
               "GIT_MERGE_AUTOEDIT": "no"}
        if auteur is not None:  # commit fabriqué : auteur et date fixés
            date = f"@{_DEBUT + 3600 * next(_horloge)} +0100"
            env.update(GIT_AUTHOR_NAME=auteur.nom, GIT_AUTHOR_EMAIL=auteur.email, GIT_AUTHOR_DATE=date,
                       GIT_COMMITTER_NAME=auteur.nom, GIT_COMMITTER_EMAIL=auteur.email, GIT_COMMITTER_DATE=date)
        options = ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", "-c", "core.autocrlf=false",
                   "-c", "advice.detachedHead=false"] if self.fabrication else []
        res = subprocess.run(["git", *options, *args], cwd=self.chemin, env=env, capture_output=True,
                             stdin=subprocess.DEVNULL)
        sortie = res.stdout.decode("utf-8", errors="replace").strip("\r\n")  # garde l'espace en tête de `status`
        if res.returncode not in codes_ok:
            erreur = res.stderr.decode("utf-8", errors="replace").strip()
            raise ErreurGit(f"`git {' '.join(args)}` a échoué dans {self.chemin} : {erreur or sortie}")
        return sortie

    def code(self, *args: str) -> int:
        """Le code de sortie de `git <args>` (0 = vrai, 1 = faux pour les commandes de test)."""
        res = subprocess.run(["git", *args], cwd=self.chemin, capture_output=True, stdin=subprocess.DEVNULL,
                             env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        return res.returncode

    # ------------------------------------------------------------------ Fabrication (préparation)

    @classmethod
    def creer(cls, chemin: Path) -> Depot:
        Path(chemin).mkdir(parents=True, exist_ok=True)
        depot = cls(chemin, fabrication=True)
        depot.git("init", "-q")
        depot.git("symbolic-ref", "HEAD", "refs/heads/main")
        depot._identite_par_defaut()
        return depot

    @classmethod
    def creer_nu(cls, chemin: Path) -> Depot:
        """Un dépôt « nu » (sans fichiers de travail) : il joue le rôle de GitHub."""
        Path(chemin).mkdir(parents=True, exist_ok=True)
        depot = cls(chemin, fabrication=True)
        depot.git("init", "-q", "--bare")
        depot.git("symbolic-ref", "HEAD", "refs/heads/main")
        return depot

    @classmethod
    def cloner(cls, origine: Depot, chemin: Path) -> Depot:
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        Depot(Path(chemin).parent, fabrication=True).git("clone", "-q", str(origine.chemin), str(chemin))
        depot = cls(chemin, fabrication=True)
        depot._identite_par_defaut()
        return depot

    def _identite_par_defaut(self) -> None:
        """Sans identité git configurée, `git commit` refuse : on en pose une, locale au bac à sable."""
        if not self.git("config", "user.email", codes_ok=(0, 1)):
            self.git("config", "user.name", "Apprenti·e de l'atelier")
            self.git("config", "user.email", "apprenti@atelier-git.invalid")

    def ecrire(self, chemin: str, contenu: str | bytes) -> None:
        fichier = self.chemin / chemin
        fichier.parent.mkdir(parents=True, exist_ok=True)
        fichier.write_bytes(contenu if isinstance(contenu, bytes) else contenu.encode("utf-8"))

    def commit(self, message: str, auteur: Auteur = VOUS) -> str:
        """`git add -A` puis un commit daté ; renvoie son identifiant (hash)."""
        self.git("add", "-A")
        self.git("commit", "-q", "--no-verify", "-m", message, auteur=auteur)
        return self.rev("HEAD")

    def exclure(self, motif: str) -> None:
        """Ignore un fichier pour ce dépôt seulement (.git/info/exclude), sans toucher au .gitignore."""
        exclude = Path(self.git("rev-parse", "--git-path", "info/exclude"))
        exclude = exclude if exclude.is_absolute() else self.chemin / exclude
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open("a", encoding="utf-8") as f:
            f.write(f"\n{motif}\n")

    # ------------------------------------------------------------------ Lecture (vérification)

    def existe(self) -> bool:
        return (self.chemin / ".git").exists() or (self.chemin / "HEAD").exists()

    def rev(self, ref: str) -> str | None:
        """Le commit désigné par `ref` (branche, tag, hash…), ou None s'il n'existe pas."""
        sortie = self.git("rev-parse", "--verify", "-q", f"{ref}^{{commit}}", codes_ok=(0, 1, 128))
        return sortie or None

    def branche_courante(self) -> str | None:
        """Le nom de la branche courante, ou None en « HEAD détachée »."""
        return self.git("symbolic-ref", "--short", "-q", "HEAD", codes_ok=(0, 1)) or None

    def fichier(self, ref: str, chemin: str) -> str | None:
        """Le contenu d'un fichier tel qu'enregistré dans un commit (None s'il n'y est pas)."""
        if not self.rev(ref):
            return None
        res = subprocess.run(["git", "show", f"{ref}:{chemin}"], cwd=self.chemin, capture_output=True,
                             stdin=subprocess.DEVNULL)
        return res.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n") if res.returncode == 0 else None

    def fichiers(self, ref: str) -> set[str]:
        sortie = self.git("ls-tree", "-r", "--name-only", ref)
        return set(sortie.splitlines())

    def est_ancetre(self, ancetre: str, descendant: str) -> bool:
        """Vrai si `ancetre` fait partie de l'historique de `descendant` (ou lui est égal)."""
        return self.code("merge-base", "--is-ancestor", ancetre, descendant) == 0

    def sujets(self, plage: str) -> list[tuple[str, str]]:
        """(hash, première ligne du message) des commits d'une plage, du plus récent au plus ancien."""
        sortie = self.git("log", "--format=%H%x1f%s", plage)
        return [tuple(ligne.split("\x1f", 1)) for ligne in sortie.splitlines() if ligne]  # type: ignore[misc]

    def contient_sujet(self, ref: str, sujet: str) -> bool:
        return any(s == sujet for _, s in self.sujets(ref))

    def fichiers_modifies(self, commit: str) -> list[str]:
        return self.git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root", commit).splitlines()

    def statut(self) -> list[str]:
        """`git status --porcelain` : une ligne par fichier modifié, prêt ou non suivi."""
        return [ligne for ligne in self.git("status", "--porcelain").splitlines() if ligne.strip()]

    def chemin_git(self, nom: str) -> Path:
        chemin = Path(self.git("rev-parse", "--git-path", nom))
        return chemin if chemin.is_absolute() else self.chemin / chemin

    def operation_en_cours(self) -> str | None:
        """Une fusion, un rebase, un revert… commencé et pas terminé ? Renvoie le conseil à suivre."""
        en_cours = [
            ("rebase-merge", "un rebase est en cours : terminez-le (`git rebase --continue`) ou annulez-le "
                             "(`git rebase --abort`)"),
            ("rebase-apply", "un rebase est en cours : terminez-le (`git rebase --continue`) ou annulez-le "
                             "(`git rebase --abort`)"),
            ("MERGE_HEAD", "une fusion est en cours : résolvez les conflits, `git add`, puis `git commit` "
                           "(ou annulez tout avec `git merge --abort`)"),
            ("REVERT_HEAD", "un `git revert` est en cours : terminez-le (`git revert --continue`) ou "
                            "annulez-le (`git revert --abort`)"),
            ("CHERRY_PICK_HEAD", "un `git cherry-pick` est en cours : `git cherry-pick --continue` ou `--abort`"),
            ("BISECT_LOG", "une recherche `git bisect` est en cours : terminez-la avec `git bisect reset`"),
        ]
        for nom, conseil in en_cours:
            if self.chemin_git(nom).exists():
                return conseil
        if any(ligne[:2] in {"UU", "AA", "DU", "UD", "AU", "UA", "DD"} for ligne in self.statut()):
            return ("des fichiers sont encore en conflit (`git status` les liste) : corrigez-les, puis "
                    "`git add <fichier>`")
        return None

    def refs(self) -> str:
        return self.git("for-each-ref", "--format=%(refname) %(objectname)")


# ---------------------------------------------------------------------- Bac à sable


def racine_par_defaut() -> Path:
    """Le bac à sable est HORS du projet : une commande ratée ne peut pas abîmer votre vrai dépôt."""
    return Path(os.environ.get("ATELIER_GIT") or Path.home() / "atelier-git").expanduser()


def preparer_racine(racine: Path) -> None:
    racine = Path(racine)
    if racine.exists() and any(racine.iterdir()) and not (racine / MARQUEUR_RACINE).exists():
        raise RuntimeError(f"{racine} existe déjà et n'a pas été créé par l'atelier : choisissez un autre "
                           "dossier avec --dossier (ou la variable d'environnement ATELIER_GIT).")
    racine.mkdir(parents=True, exist_ok=True)
    (racine / MARQUEUR_RACINE).write_text(
        "Bac à sable de l'atelier Git (scripts/atelier_git.py). Vous pouvez supprimer ce dossier.\n",
        encoding="utf-8")


def _rendre_inscriptible(fonction, chemin, _exc) -> None:
    """Sous Windows, git crée des fichiers en lecture seule que rmtree ne peut pas supprimer."""
    os.chmod(chemin, stat.S_IWRITE)
    fonction(chemin)


def supprimer(chemin: Path, racine: Path) -> None:
    """Supprime un dossier du bac à sable, en vérifiant qu'il est bien dedans."""
    chemin, racine = Path(chemin).resolve(), Path(racine).resolve()
    if not chemin.exists():
        return
    if racine not in chemin.parents or not (racine / MARQUEUR_RACINE).exists():
        raise RuntimeError(f"Refus de supprimer {chemin} : il n'est pas dans le bac à sable {racine}.")
    if sys.version_info >= (3, 12):
        shutil.rmtree(chemin, onexc=_rendre_inscriptible)
    else:
        shutil.rmtree(chemin, onerror=_rendre_inscriptible)

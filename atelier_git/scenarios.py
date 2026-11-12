"""Les 8 scénarios de l'atelier Git : la situation, la mission, les indices, la solution et la correction.

Chaque scénario fabrique un vrai dépôt git dans le bac à sable (`preparer`), puis l'inspecte une fois
que vous avez travaillé (`verifier`). La vérification lève une `AssertionError` dont le message
explique ce qui ne va pas et comment s'en sortir.

Lire ce fichier avant d'avoir cherché, c'est lire le corrigé : passez plutôt par
`python scripts/atelier_git.py N --indice`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from atelier_git.outils import CAMILLE, VOUS, Depot, normaliser


@dataclass(frozen=True)
class Pas:
    """Une étape de la solution : une commande à taper, ou une consigne (✏️) à suivre à la main."""

    texte: str
    commentaire: str = ""
    action: Callable[[Depot], None] | None = None  # pour les tests : consigne, ou commande à adapter (hash…)
    peut_echouer: bool = False                     # ex. `git merge` qui s'arrête sur un conflit

    @property
    def est_commande(self) -> bool:
        return self.texte.startswith(("git ", "python "))


@dataclass(frozen=True)
class Scenario:
    numero: int
    slug: str
    titre: str
    niveau: str                      # 🟢 🟡 🔴
    commandes: str                   # ce que vous pratiquez
    situation: str
    mission: tuple[str, ...]         # les critères de réussite, vérifiés un par un
    indices: tuple[str, ...]         # du plus léger au plus direct
    solution: tuple[Pas, ...]
    a_retenir: str
    preparer: Callable[[Path, Path], None]   # (dossier de travail, dossier des dépôts distants)
    verifier: Callable[[Path, Path], None]   # lève AssertionError (message explicatif) si la mission n'est pas remplie

    @property
    def nom_dossier(self) -> str:
        return f"{self.numero:02d}-{self.slug}"


# ============================================================================ Outils de vérification

MAIN = "refs/heads/main"
_VAGUES = {"wip", "update", "updates", "maj", "modif", "modifs", "modification", "modifications", "commit",
           "fix", "fixes", "bug", "test", "tests", "change", "changes", "divers", "save", "sauvegarde", "ok",
           "encore", "suite", "travail", "essai"}


def message_vague(sujet: str) -> bool:
    """Un message qui ne dit pas ce que fait le commit (« wip », « fix typo », « update »…)."""
    s = sujet.strip().lower().rstrip(".!… ")
    return len(s) < 10 or s in _VAGUES or s.startswith(("wip", "fix typo", "typo", "tmp", "temp", "test commit"))


def _verifier_messages(depot: Depot, plage: str) -> None:
    for h, sujet in depot.sujets(plage):
        if message_vague(sujet):
            raise AssertionError(
                f"Le message « {sujet} » (commit {h[:7]}) ne dit pas ce que fait le commit. Dans six mois, "
                "`git log --oneline` doit se lire comme un journal : « Corrige la division par zéro de "
                "taux_churn ». Pour réécrire le message du dernier commit (pas encore poussé) : "
                "git commit --amend -m \"…\"")


def _pas_d_operation_en_cours(depot: Depot) -> None:
    conseil = depot.operation_en_cours()
    if conseil:
        raise AssertionError(conseil[0].upper() + conseil[1:] + ".")


def _propre(depot: Depot) -> None:
    lignes = depot.statut()
    if lignes:
        liste = ", ".join(ligne[3:] for ligne in lignes[:4]) + (" …" if len(lignes) > 4 else "")
        raise AssertionError(f"`git status` n'est pas propre ({liste}). Tout doit être commité, ou ignoré "
                             "s'il ne doit pas l'être.")


def _racine(depot: Depot, ref: str = "HEAD") -> str:
    """Le tout premier commit de l'historique."""
    return depot.git("rev-list", "--max-parents=0", ref).split()[-1]


def _executer(source: str | None, nom: str) -> dict:
    """Exécute un fichier Python tel qu'il est enregistré dans un commit et renvoie ses variables."""
    if source is None:
        raise AssertionError(f"{nom} est introuvable dans ce commit.")
    espace: dict = {}
    try:
        exec(compile(source, nom, "exec"), espace)  # noqa: S102 - fichiers fabriqués par l'atelier
    except SyntaxError as exc:
        raise AssertionError(f"{nom} n'est plus du Python valide (ligne {exc.lineno} : {exc.msg}). "
                             "Un reste de marqueur de conflit ? Une virgule oubliée ?") from None
    return espace


def _identique(depot: Depot, ref: str, chemin: str, attendu: str) -> bool:
    return normaliser(depot.fichier(ref, chemin)) == normaliser(attendu)


def _hash_dans_reflog(depot: Depot, sujet: str) -> str:
    """Pour la solution testée du scénario 7 : le commit d'une ligne `commit: <sujet>` du reflog."""
    for ligne in depot.git("reflog", "--format=%H %gs").splitlines():
        h, _, action = ligne.partition(" ")
        if action == f"commit: {sujet}":
            return h
    raise AssertionError(f"« {sujet} » est absent du reflog")


def _publier(camille: Depot, origine: Depot) -> None:
    camille.git("remote", "add", "origin", str(origine.chemin))
    camille.git("push", "-q", "--no-verify", "-u", "origin", "main")


# ============================================================================ 1 — Un commit propre

CHURN_1 = '''"""Indicateurs de churn."""


def taux_churn(clients):
    """Part des clients partis, entre 0 et 1."""
    partis = sum(1 for client in clients if client["parti"])
    return partis / len(clients)
'''

CHURN_1_CORRIGE = '''"""Indicateurs de churn."""


def taux_churn(clients):
    """Part des clients partis, entre 0 et 1 (0 s'il n'y a aucun client)."""
    if not clients:
        return 0.0
    partis = sum(1 for client in clients if client["parti"])
    return partis / len(clients)
'''

TEST_1 = '''from churn import taux_churn


def test_taux_churn():
    clients = [{"parti": True}, {"parti": False}, {"parti": False}, {"parti": True}]
    assert taux_churn(clients) == 0.5


def test_taux_churn_sans_client():
    assert taux_churn([]) == 0.0
'''

PYC = "__pycache__/churn.cpython-312.pyc"


def _preparer_1(travail: Path, distants: Path) -> None:
    d = Depot.creer(travail)
    d.ecrire("README.md", "# Indicateurs de churn\n\nPetit module de calcul utilisé par le tableau de bord.\n")
    d.ecrire("churn.py", CHURN_1)
    d.commit("Version initiale")
    d.ecrire("churn.py", CHURN_1_CORRIGE)       # votre correction, pas encore commitée
    d.ecrire("test_churn.py", TEST_1)           # votre nouveau test
    d.ecrire("debug.log", "DEBUG clients=[] -> ZeroDivisionError\nDEBUG essai 2 : ok\n")
    d.ecrire(PYC, b"\xcb\r\r\n\x00\x00\x00\x00 fichier compile par Python, a ne jamais commiter")


def _verifier_1(travail: Path, distants: Path) -> None:
    d = Depot(travail)
    _pas_d_operation_en_cours(d)
    racine = _racine(d)
    nouveaux = d.sujets(f"{racine}..HEAD")
    if not nouveaux:
        raise AssertionError("Aucun nouveau commit pour l'instant. Préparez les bons fichiers "
                             "(`git add churn.py test_churn.py`), vérifiez avec `git status`, puis "
                             "`git commit -m \"…\"`.")
    for h, sujet in nouveaux:
        indesirables = [f for f in d.fichiers_modifies(h) if f == "debug.log" or f.startswith("__pycache__/")]
        if indesirables:
            raise AssertionError(
                f"Le commit « {sujet} » contient {', '.join(indesirables)} : un `git add .` (ou `-A`) a tout "
                "pris. Rien n'est encore partagé, vous pouvez refaire vos commits : "
                f"`git reset --soft {racine[:7]}` (vos modifications restent prêtes à commiter), "
                "`git rm -r --cached debug.log __pycache__` (retire du commit, pas du disque), créez le "
                ".gitignore, puis recommitez.")
    if _identique(d, "HEAD", "churn.py", CHURN_1):
        raise AssertionError("La correction de churn.py n'est pas dans vos commits : il est « modifié » mais "
                             "pas ajouté. git add churn.py, puis un commit.")
    if not _identique(d, "HEAD", "churn.py", CHURN_1_CORRIGE):
        raise AssertionError("churn.py, tel que commité, n'est plus la correction préparée. L'exercice porte "
                             "sur git : ne modifiez pas les fichiers (pour repartir de zéro : --recommencer).")
    if d.fichier("HEAD", "test_churn.py") is None:
        raise AssertionError("test_churn.py n'est pas commité. C'est un nouveau fichier (« untracked ») : git "
                             "ne le suit que si vous l'ajoutez avec git add test_churn.py.")
    if not _identique(d, "HEAD", "test_churn.py", TEST_1):
        raise AssertionError("test_churn.py, tel que commité, a été modifié : ne touchez pas au contenu "
                             "(pour repartir de zéro : --recommencer).")
    if d.fichier("HEAD", ".gitignore") is None:
        if (travail / ".gitignore").exists():
            raise AssertionError("Le .gitignore existe mais n'est pas commité : git add .gitignore, puis un "
                                 "commit. Il sert à toute l'équipe, il doit être dans le dépôt.")
        raise AssertionError("Pas de .gitignore commité. Créez un fichier texte nommé .gitignore avec un motif "
                             "par ligne (par exemple *.log et __pycache__/), puis commitez-le.")
    for chemin, motif in (("debug.log", "*.log"), (PYC, "__pycache__/")):
        sortie = d.git("check-ignore", "-v", "--no-index", chemin, codes_ok=(0, 1))
        source = sortie.split(":", 1)[0] if sortie else ""
        if not sortie:
            raise AssertionError(f"Votre .gitignore n'ignore pas {chemin}. Ajoutez-y la ligne {motif}.")
        if source != ".gitignore":
            raise AssertionError(
                f"{chemin} est ignoré par {source}, qui reste sur votre machine. Mettez le motif {motif} dans "
                "le .gitignore du projet (commité) : vos collègues en profiteront aussi.")
    _propre(d)
    _verifier_messages(d, f"{racine}..HEAD")


SCENARIO_1 = Scenario(
    numero=1, slug="commit-propre", titre="Un commit propre", niveau="🟢",
    commandes="status · diff · add · .gitignore · commit · log",
    situation=(
        "Vous avez corrigé un bug dans churn.py (une division par zéro quand la liste de clients est vide) "
        "et écrit le test qui le prouve, test_churn.py. En chemin, vos essais et Python ont laissé des "
        "fichiers qui n'ont rien à faire dans l'historique : debug.log et le dossier __pycache__/. Il faut "
        "enregistrer votre travail, et seulement lui."),
    mission=(
        "Un nouveau commit (ou deux) contient la correction de churn.py et le nouveau test_churn.py, tels quels.",
        "debug.log et __pycache__/ ne sont dans aucun commit, et `git status` ne les montre plus : un fichier "
        ".gitignore, commité lui aussi, les ignore.",
        "Chaque message de commit dit ce que fait le commit (pas « wip », « update » ni « modifs »).",
        "À la fin, `git status` affiche « nothing to commit, working tree clean ».",
    ),
    indices=(
        "Commencez toujours par `git status` : il sépare les fichiers « modifiés » (déjà suivis par git) des "
        "fichiers « non suivis » (untracked). Puis `git diff churn.py` montre ligne par ligne ce qui a changé.",
        "`git add .` ajoute TOUT, y compris les fichiers indésirables. Ajoutez fichier par fichier "
        "(`git add churn.py test_churn.py`) et relisez `git status` avant de commiter.",
        "Un .gitignore est un simple fichier texte, un motif par ligne : *.log ignore tous les fichiers .log, "
        "__pycache__/ ignore le dossier. Créez-le avec votre éditeur, puis `git add .gitignore`.",
        "Déjà commité un fichier de trop ? Tant que rien n'est poussé, on peut refaire : "
        "`git rm --cached debug.log` (le retire du dépôt, pas du disque), puis `git commit --amend`.",
    ),
    solution=(
        Pas("git status", "deux fichiers à garder, deux à ignorer"),
        Pas("git diff churn.py", "relire sa correction avant de l'enregistrer"),
        Pas("✏️ Créez le fichier .gitignore avec deux lignes : *.log et __pycache__/",
            action=lambda d: d.ecrire(".gitignore", "*.log\n__pycache__/\n")),
        Pas("git add churn.py test_churn.py .gitignore"),
        Pas("git status", "seuls ces 3 fichiers sont dans « Changes to be committed »"),
        Pas('git commit -m "Corrige la division par zéro de taux_churn et ajoute ses tests"'),
        Pas("git log --oneline"),
    ),
    a_retenir=(
        "`git status` avant chaque commit, `git add` fichier par fichier, un .gitignore dès le premier jour. "
        "Un commit = une intention, et son message la décrit."),
    preparer=_preparer_1, verifier=_verifier_1,
)


# ============================================================================ 2 — J'ai commité sur main

PRIX = '''"""Calcul des prix."""


def prix_mensuel(base, options=0):
    return base + options
'''

REMISE = '''"""Remise fidélité : 10 % après deux ans d'ancienneté."""


def remise_fidelite(prix, anciennete_mois):
    if anciennete_mois >= 24:
        return round(prix * 0.9, 2)
    return prix
'''

TEST_REMISE = '''from remise import remise_fidelite


def test_remise_apres_deux_ans():
    assert remise_fidelite(50, 24) == 45


def test_pas_de_remise_avant():
    assert remise_fidelite(50, 23) == 50
'''


def _preparer_2(travail: Path, distants: Path) -> None:
    origine = Depot.creer_nu(distants / "origine.git")
    camille = Depot.creer(distants / "camille")
    camille.ecrire("README.md", "# Tarifs\n\nRègle d'équipe : jamais de commit direct sur main.\n")
    camille.ecrire("prix.py", PRIX)
    camille.commit("Version initiale", CAMILLE)
    _publier(camille, origine)
    d = Depot.cloner(origine, travail)
    d.ecrire("remise.py", REMISE)
    d.commit("Ajoute la remise fidélité")
    d.ecrire("test_remise.py", TEST_REMISE)
    d.commit("Teste la remise fidélité")


def _verifier_2(travail: Path, distants: Path) -> None:
    d, origine = Depot(travail), Depot(distants / "origine.git")
    _pas_d_operation_en_cours(d)
    initial = Depot(distants / "camille").rev(MAIN)
    if origine.rev(MAIN) != initial:
        raise AssertionError(
            "origin/main a changé : vos commits ont été poussés sur main, ce que la règle de l'équipe interdit. "
            "Une fois publié, on ne revient pas en arrière sans réécrire l'historique partagé (voir le "
            "scénario 5) : recommencez ce scénario avec --recommencer.")
    branche = d.rev("refs/heads/remise-fidelite")
    if branche is None:
        if d.rev(MAIN) == initial:
            raise AssertionError(
                "main est revenu en arrière, mais il n'y a pas de branche remise-fidelite : vos 2 commits ne "
                "sont plus visibles. Ils ne sont pas perdus : `git reflog` les montre, et "
                "`git branch remise-fidelite <hash de « Teste la remise fidélité »>` les récupère.")
        raise AssertionError("Créez la branche à partir de main, qui contient vos commits : "
                             "git branch remise-fidelite")
    if not (d.est_ancetre(initial, branche) and _identique(d, branche, "remise.py", REMISE)
            and _identique(d, branche, "test_remise.py", TEST_REMISE)):
        raise AssertionError("La branche remise-fidelite doit partir de origin/main et contenir vos 2 commits "
                             "(remise.py et test_remise.py). Elle pointe sur le mauvais commit : "
                             "git branch -f remise-fidelite <hash>.")
    if d.rev(MAIN) != initial:
        raise AssertionError("main contient encore vos commits. Ils sont à l'abri sur remise-fidelite : "
                             "remettez main comme origin/main avec `git switch main` puis "
                             "`git reset --hard origin/main`.")
    distante = origine.rev("refs/heads/remise-fidelite")
    if distante is None:
        raise AssertionError("La branche n'est pas publiée sur origin : git push -u origin remise-fidelite")
    if distante != branche:
        raise AssertionError("origin/remise-fidelite n'est pas à jour avec votre branche : git push")
    amont = d.git("rev-parse", "--abbrev-ref", "remise-fidelite@{upstream}", codes_ok=(0, 128))
    if amont != "origin/remise-fidelite":
        raise AssertionError("La branche n'est pas « suivie » : `git status` ne peut pas dire si elle est à "
                             "jour. `git branch --set-upstream-to=origin/remise-fidelite remise-fidelite` "
                             "(la prochaine fois : git push -u).")
    _propre(d)


SCENARIO_2 = Scenario(
    numero=2, slug="commit-sur-main", titre="J'ai commité sur main par erreur", niveau="🟢",
    commandes="log --graph · branch · reset --hard · switch · push -u",
    situation=(
        "Dans votre équipe, la règle est simple : on ne commite jamais directement sur main. Chaque travail "
        "vit sur sa propre branche, puis passe en revue (pull request). Vous avez pourtant fait deux commits "
        "sur main : « Ajoute la remise fidélité » et « Teste la remise fidélité ». Heureusement, rien n'est "
        "encore parti sur origin, le dépôt partagé (comme GitHub)."),
    mission=(
        "Vos deux commits sont sur une nouvelle branche nommée remise-fidelite.",
        "main est revenu exactement à origin/main, sans vos commits, et n'a pas été poussé.",
        "La branche est publiée sur origin et suivie (git push -u).",
        "`git status` est propre.",
    ),
    indices=(
        "`git log --oneline --graph --all` dessine l'historique : main a deux commits d'avance sur origin/main. "
        "Gardez cette commande sous la main pour tous les scénarios.",
        "Une branche n'est qu'une étiquette posée sur un commit. `git branch remise-fidelite` pose une "
        "nouvelle étiquette là où vous êtes, donc sur vos 2 commits, sans changer de branche.",
        "Ensuite seulement, reculez main : `git reset --hard origin/main`. C'est sans danger ici, parce que vos "
        "commits sont gardés par la nouvelle branche et qu'aucune modification n'est en cours. Dans l'autre "
        "ordre, vos commits ne seraient plus visibles (le scénario 7 montre comment les retrouver).",
        "`git switch remise-fidelite` puis `git push -u origin remise-fidelite` : -u mémorise le lien, et les "
        "prochains `git push` et `git pull` sauront où aller.",
    ),
    solution=(
        Pas("git log --oneline --graph --all"),
        Pas("git branch remise-fidelite", "une étiquette sur vos 2 commits"),
        Pas("git reset --hard origin/main", "main recule ; la branche garde les commits"),
        Pas("git switch remise-fidelite"),
        Pas("git push -u origin remise-fidelite"),
        Pas("git log --oneline --graph --all"),
    ),
    a_retenir=(
        "Une branche est une étiquette qui avance avec vos commits : en créer une ne coûte rien, faites-en une "
        "par sujet. `git reset --hard` déplace la branche courante ET écrase vos fichiers : `git status` "
        "d'abord, toujours."),
    preparer=_preparer_2, verifier=_verifier_2,
)


# ============================================================================ 3 — Urgence : mettre son travail de côté

CHURN_3 = '''"""Règles de ciblage des clients à contacter."""

SEUIL = 5  # probabilité de churn à partir de laquelle on appelle le client


def a_contacter(proba_churn):
    return proba_churn >= SEUIL
'''

GRAPHIQUES_BASE = '''"""Graphiques du tableau de bord."""


def graphique_par_contrat(df):
    raise NotImplementedError("à faire")
'''

GRAPHIQUES_BRANCHE = '''"""Graphiques du tableau de bord."""


def graphique_par_contrat(df):
    """Taux de churn par type de contrat, du plus risqué au moins risqué."""
    taux = df.groupby("contract")["churn"].mean().sort_values(ascending=False)
    return {"x": list(taux.index), "y": [round(v, 3) for v in taux]}
'''

GRAPHIQUES_EN_COURS = GRAPHIQUES_BRANCHE + '''

def graphique_par_paiement(df):
    # TODO : finir après la réunion (trier ? ajouter les effectifs ?)
    taux = df.groupby("payment_method")["churn"].mean()
'''

BROUILLON = 'PALETTE = ["#2E5EAA", "#E4572E", "#76B041"]  # essais de couleurs, pas encore validés\n'
SUJET_GRAPHIQUES = "Ajoute le graphique par contrat"


def _preparer_3(travail: Path, distants: Path) -> None:
    d = Depot.creer(travail)
    d.ecrire("README.md", "# Tableau de bord\n")
    d.ecrire("churn.py", CHURN_3)
    d.ecrire("graphiques.py", GRAPHIQUES_BASE)
    d.commit("Version initiale", CAMILLE)
    d.git("switch", "-q", "-c", "graphiques")
    d.ecrire("graphiques.py", GRAPHIQUES_BRANCHE)
    d.commit(SUJET_GRAPHIQUES)
    d.ecrire("graphiques.py", GRAPHIQUES_EN_COURS)  # travail en cours, non commité
    d.ecrire("brouillon_couleurs.py", BROUILLON)    # nouveau fichier, non suivi


def _verifier_3(travail: Path, distants: Path) -> None:
    d = Depot(travail)
    _pas_d_operation_en_cours(d)
    fichiers_main = d.fichiers(MAIN)
    if "brouillon_couleurs.py" in fichiers_main:
        raise AssertionError(
            "brouillon_couleurs.py est dans un commit de main : votre brouillon part en production ! "
            "`git stash` seul ne range pas les fichiers non suivis : il est resté dans le dossier, et "
            "`git add .` l'a pris. Recommencez (--recommencer) avec `git stash -u`, et ajoutez seulement "
            "churn.py.")
    graphiques_main = normaliser(d.fichier(MAIN, "graphiques.py")) or ""
    if "graphique_par_paiement" in graphiques_main:
        raise AssertionError("Votre fonction inachevée graphique_par_paiement est dans main : le correctif ne "
                             "doit contenir QUE la correction du seuil. Recommencez avec --recommencer.")
    if graphiques_main != normaliser(GRAPHIQUES_BASE):
        raise AssertionError("graphiques.py a changé sur main : le correctif urgent ne doit toucher que "
                             "churn.py.")
    seuil = _executer(d.fichier(MAIN, "churn.py"), "churn.py (sur main)").get("SEUIL")
    if seuil == 5:
        raise AssertionError("Sur main, SEUIL vaut toujours 5 : le correctif n'est pas (encore) commité sur "
                             "main. Mettez votre travail de côté (`git stash -u`), puis `git switch main`.")
    if seuil != 0.5:
        raise AssertionError(f"Sur main, SEUIL vaut {seuil!r} au lieu de 0.5.")
    en_trop = fichiers_main - {"README.md", "churn.py", "graphiques.py"}
    if en_trop:
        raise AssertionError(f"Des fichiers en trop sur main : {', '.join(sorted(en_trop))}.")
    branche = d.branche_courante()
    if branche != "graphiques":
        raise AssertionError(f"Vous êtes sur {branche or 'une HEAD détachée'} : revenez sur votre branche "
                             "(git switch graphiques) et récupérez votre travail (git stash pop).")
    if not d.contient_sujet("refs/heads/graphiques", SUJET_GRAPHIQUES):
        raise AssertionError(f"La branche graphiques a perdu son commit « {SUJET_GRAPHIQUES} ».")
    remise = d.git("stash", "list")
    for chemin, attendu in (("graphiques.py", GRAPHIQUES_EN_COURS), ("brouillon_couleurs.py", BROUILLON)):
        fichier = travail / chemin
        if not fichier.exists() or normaliser(fichier.read_text(encoding="utf-8")) != normaliser(attendu):
            if remise:
                raise AssertionError(f"{chemin} n'a pas retrouvé votre travail en cours, qui est encore dans la "
                                     "remise (`git stash list`) : git stash pop")
            raise AssertionError(f"{chemin} ne contient plus votre travail en cours. S'il a été rangé avec "
                                 "`git stash`, `git stash list` le montre ; sinon, --recommencer.")


SCENARIO_3 = Scenario(
    numero=3, slug="urgence-stash", titre="Urgence : mettre son travail de côté", niveau="🟢",
    commandes="stash -u · switch · commit -a · stash pop",
    situation=(
        "Vous êtes en plein travail sur la branche graphiques : une fonction à moitié écrite dans "
        "graphiques.py et un fichier de brouillon, brouillon_couleurs.py, rien de commité. Urgence : en "
        "production, plus aucun client n'est contacté. Sur main, quelqu'un a écrit SEUIL = 5 au lieu de "
        "SEUIL = 0.5 dans churn.py (une probabilité ne dépasse jamais 1). Il faut corriger main tout de "
        "suite, sans y mélanger votre travail inachevé, puis reprendre là où vous en étiez."),
    mission=(
        "Sur main, un nouveau commit corrige SEUIL = 5 en SEUIL = 0.5 dans churn.py, et rien d'autre.",
        "Rien de votre travail en cours n'est sur main : ni graphique_par_paiement, ni brouillon_couleurs.py.",
        "Vous êtes revenu·e sur la branche graphiques et votre travail en cours est là, tel quel (non commité, "
        "ou commité sur graphiques : les deux conviennent).",
    ),
    indices=(
        "Essayez `git switch main` : git refuse, pour ne pas écraser vos modifications de graphiques.py. Il "
        "faut d'abord les mettre de côté.",
        "`git stash` range vos modifications dans une « remise » et vous rend un dossier propre. Mais par "
        "défaut, il ne prend PAS les fichiers non suivis comme brouillon_couleurs.py : ajoutez -u "
        "(--include-untracked).",
        "Sur main : corrigez churn.py, puis `git commit -am \"…\"` (-a ajoute les fichiers déjà suivis et "
        "modifiés, jamais les nouveaux).",
        "Retour sur graphiques, puis `git stash pop` : il ressort le dernier élément rangé et le retire de la "
        "remise. `git stash list` montre ce qu'elle contient.",
    ),
    solution=(
        Pas("git status"),
        Pas('git stash push -u -m "graphiques en cours"', "-u : prend aussi brouillon_couleurs.py"),
        Pas("git switch main"),
        Pas("✏️ Dans churn.py, remplacez SEUIL = 5 par SEUIL = 0.5",
            action=lambda d: d.ecrire("churn.py", CHURN_3.replace("SEUIL = 5 ", "SEUIL = 0.5 "))),
        Pas('git commit -am "Corrige le seuil de contact : 0.5 au lieu de 5"'),
        Pas("git switch graphiques"),
        Pas("git stash pop"),
        Pas("git status", "votre travail en cours est revenu"),
    ),
    a_retenir=(
        "`git stash -u` met tout de côté, y compris les nouveaux fichiers ; `git stash pop` le ressort. Et "
        "avant tout `git add .`, un `git status` : c'est là qu'on voit un brouillon prêt à partir en "
        "production."),
    preparer=_preparer_3, verifier=_verifier_3,
)


# ============================================================================ 4 — Résoudre un conflit

OFFRES_BASE = '''"""Offres de rétention : code de l'offre -> coût en euros."""

OFFRES = {
    "APPEL": 8,
    "REMISE": 45,
}
'''
OFFRES_ATTENDUES = {"APPEL": 8, "SUPPORT": 20, "REMISE": 45, "ENGAGEMENT": 60}
OFFRES_FUSIONNEES = OFFRES_BASE.replace('    "APPEL": 8,\n',
                                        '    "APPEL": 8,\n    "SUPPORT": 20,\n    "ENGAGEMENT": 60,\n')
SUJET_SUPPORT, SUJET_ENGAGEMENT = "Ajoute l'offre SUPPORT", "Ajoute l'offre ENGAGEMENT"

TEST_OFFRES = '''from offres import OFFRES


def test_offre_engagement():
    assert OFFRES["ENGAGEMENT"] == 60
'''


def _preparer_4(travail: Path, distants: Path) -> None:
    d = Depot.creer(travail)
    d.ecrire("README.md", "# Offres de rétention\n")
    d.ecrire("offres.py", OFFRES_BASE)
    d.commit("Version initiale", CAMILLE)
    d.git("switch", "-q", "-c", "offre-engagement")
    d.ecrire("offres.py", OFFRES_BASE.replace('    "APPEL": 8,\n', '    "APPEL": 8,\n    "ENGAGEMENT": 60,\n'))
    d.commit(SUJET_ENGAGEMENT)
    d.ecrire("test_offres.py", TEST_OFFRES)
    d.commit("Teste l'offre ENGAGEMENT")
    d.git("switch", "-q", "main")  # pendant ce temps, Camille travaille sur main
    d.ecrire("offres.py", OFFRES_BASE.replace('    "APPEL": 8,\n', '    "APPEL": 8,\n    "SUPPORT": 20,\n'))
    d.commit(SUJET_SUPPORT, CAMILLE)
    d.ecrire("README.md", "# Offres de rétention\n\nLes coûts sont en euros, par client contacté.\n")
    d.commit("Précise l'unité des coûts dans le README", CAMILLE)
    d.git("switch", "-q", "offre-engagement")


def _verifier_4(travail: Path, distants: Path) -> None:
    d = Depot(travail)
    _pas_d_operation_en_cours(d)
    if not d.contient_sujet(MAIN, SUJET_ENGAGEMENT):
        raise AssertionError("main ne contient pas encore votre travail. On fusionne depuis la branche qui "
                             "reçoit : git switch main, puis git merge offre-engagement.")
    if not d.contient_sujet(MAIN, SUJET_SUPPORT):
        raise AssertionError(f"main a perdu le commit de Camille « {SUJET_SUPPORT} ». Recommencez avec "
                             "--recommencer.")
    for chemin in sorted(d.fichiers(MAIN)):
        if re.search(r"^(<{7}|\|{7}|={7}|>{7})", d.fichier(MAIN, chemin) or "", re.MULTILINE):
            raise AssertionError(
                f"{chemin} a été commité avec des marqueurs de conflit (<<<<<<<, ======= ou >>>>>>>). git add "
                "ne vérifie rien : c'est à vous de nettoyer. Corrigez le fichier, puis "
                f"git commit -am \"Retire les marqueurs de conflit de {chemin}\".")
    offres = _executer(d.fichier(MAIN, "offres.py"), "offres.py (sur main)").get("OFFRES") or {}
    manquantes = [code for code in OFFRES_ATTENDUES if code not in offres]
    if manquantes:
        raise AssertionError(
            f"Il manque {' et '.join(manquantes)} dans offres.py : en résolvant le conflit, une seule des deux "
            "versions a été gardée. Un conflit ne veut pas dire « choisir un camp » : ici, il faut les deux "
            "lignes. Corrigez offres.py sur main, puis git commit -am \"Rétablit l'offre …\".")
    if offres != OFFRES_ATTENDUES:
        raise AssertionError(f"OFFRES vaut {offres} au lieu de {OFFRES_ATTENDUES}.")
    _propre(d)


SCENARIO_4 = Scenario(
    numero=4, slug="conflit", titre="Résoudre un conflit de fusion", niveau="🟡",
    commandes="merge · conflit · add · commit · merge --abort",
    situation=(
        "Vous avez ajouté l'offre ENGAGEMENT sur votre branche offre-engagement. Pendant ce temps, Camille a "
        "ajouté l'offre SUPPORT directement sur main, au même endroit du même fichier. Quand deux personnes "
        "modifient les mêmes lignes, git ne choisit pas à votre place : c'est un conflit, et c'est à vous de "
        "le résoudre."),
    mission=(
        "main contient votre travail (fusion de offre-engagement) et celui de Camille.",
        "Dans offres.py sur main, les 4 offres sont là : APPEL 8, SUPPORT 20, REMISE 45, ENGAGEMENT 60.",
        "Plus aucun marqueur de conflit (<<<<<<<, =======, >>>>>>>), la fusion est terminée et "
        "`git status` est propre.",
    ),
    indices=(
        "On fusionne depuis la branche qui reçoit : `git switch main`, puis `git merge offre-engagement`. Git "
        "fusionne seul le README et test_offres.py, mais s'arrête sur offres.py.",
        "Ouvrez offres.py. Entre <<<<<<< HEAD et ======= : la version de main (Camille). Entre ======= et "
        ">>>>>>> offre-engagement : la vôtre. Un conflit ne veut pas dire « choisir un camp » : ici, il faut "
        "garder les deux lignes.",
        "Supprimez les 3 lignes de marqueurs, vérifiez que le fichier est du Python valide "
        "(`python -c \"import offres; print(offres.OFFRES)\"`), puis `git add offres.py` : c'est ainsi "
        "qu'on dit à git « conflit résolu ».",
        "Terminez avec `git commit --no-edit`. Sans --no-edit, git ouvre un éditeur pour le message ; si c'est "
        "Vim, tapez :wq puis Entrée. Perdu·e ? `git merge --abort` remet tout comme avant la fusion.",
    ),
    solution=(
        Pas("git switch main"),
        Pas("git merge offre-engagement", "s'arrête : CONFLICT (content): Merge conflict in offres.py",
            peut_echouer=True),
        Pas("git status", "offres.py est « both modified »"),
        Pas('✏️ Dans offres.py, gardez les lignes "SUPPORT": 20, et "ENGAGEMENT": 60, puis supprimez les '
            "lignes <<<<<<<, ======= et >>>>>>>",
            action=lambda d: d.ecrire("offres.py", OFFRES_FUSIONNEES)),
        Pas("git add offres.py", "« conflit résolu »"),
        Pas("git commit --no-edit"),
        Pas("git log --oneline --graph --all"),
    ),
    a_retenir=(
        "Un conflit, c'est git qui vous demande de décider : lisez les deux versions, écrivez la bonne (souvent : "
        "les deux), `git add`, `git commit`. Et `git merge --abort` en cas de doute."),
    preparer=_preparer_4, verifier=_verifier_4,
)


# ============================================================================ 5 — Annuler un commit déjà partagé

REGLES_5 = '''"""Règles de ciblage."""

SEUIL = 0.5  # probabilité de churn à partir de laquelle on appelle le client


def a_contacter(proba_churn):
    return proba_churn >= SEUIL
'''

EXPORT_5 = '''"""Export CSV des clients à contacter."""

import csv


def exporter(clients, chemin):
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "proba"])
        for client in clients:
            writer.writerow([client["id"], client["proba"]])
'''

TEST_EXPORT_5 = '''import csv

from export import exporter


def test_exporter(tmp_path):
    chemin = tmp_path / "export.csv"
    exporter([{"id": "A1", "proba": 0.8}], chemin)
    assert list(csv.reader(chemin.open(encoding="utf-8"))) == [["id", "proba"], ["A1", "0.8"]]
'''
SUJET_FAUTIF, SUJET_TEST_CAMILLE = "Passe le seuil à 0.9 pour la démo", "Ajoute un test de l'export"


def _preparer_5(travail: Path, distants: Path) -> None:
    origine = Depot.creer_nu(distants / "origine.git")
    camille = Depot.creer(distants / "camille")
    camille.ecrire("README.md", "# Ciblage des clients\n")
    camille.ecrire("regles.py", REGLES_5)
    camille.commit("Version initiale", CAMILLE)
    camille.ecrire("export.py", EXPORT_5)
    camille.commit("Ajoute l'export CSV des clients à contacter", CAMILLE)
    _publier(camille, origine)
    d = Depot.cloner(origine, travail)
    d.ecrire("regles.py", REGLES_5.replace("SEUIL = 0.5", "SEUIL = 0.9"))
    d.commit(SUJET_FAUTIF)
    d.git("push", "-q", "--no-verify")
    camille.git("pull", "-q", "--ff-only")  # Camille a récupéré votre commit…
    camille.ecrire("test_export.py", TEST_EXPORT_5)
    camille.commit(SUJET_TEST_CAMILLE, CAMILLE)  # … et a poussé le sien par-dessus
    camille.git("push", "-q", "--no-verify")
    d.git("pull", "-q", "--ff-only")


def _verifier_5(travail: Path, distants: Path) -> None:
    d, origine, camille = Depot(travail), Depot(distants / "origine.git"), Depot(distants / "camille")
    _pas_d_operation_en_cours(d)
    publie = camille.rev(MAIN)  # ce que Camille a déjà récupéré : le dernier commit publié
    distant = origine.rev(MAIN)
    if distant is None or not origine.est_ancetre(publie, distant):
        raise AssertionError(
            "L'historique partagé a été réécrit (reset + push --force ?) : origin/main ne contient plus les "
            f"commits que Camille a déjà, dont « {SUJET_TEST_CAMILLE} ». Au prochain pull de Camille, les "
            "deux histoires se mélangeront et le commit fautif reviendra. Recommencez avec --recommencer, "
            "et cette fois ajoutez l'annulation au lieu d'effacer : git revert.")
    seuil_distant = _executer(origine.fichier(MAIN, "regles.py"), "regles.py (sur origin/main)").get("SEUIL")
    if seuil_distant != 0.5:
        local = _executer(d.fichier(MAIN, "regles.py"), "regles.py (sur votre main)").get("SEUIL")
        if local == 0.5:
            raise AssertionError("Votre annulation est commitée mais pas publiée : git push")
        raise AssertionError(f"Sur origin/main, SEUIL vaut {seuil_distant!r} au lieu de 0.5. Annulez le commit "
                             f"fautif « {SUJET_FAUTIF} » : git revert --no-edit <son hash>, puis git push.")
    for chemin, attendu in (("export.py", EXPORT_5), ("test_export.py", TEST_EXPORT_5)):
        if not _identique(origine, MAIN, chemin, attendu):
            raise AssertionError(f"{chemin} a changé ou disparu sur origin/main : l'annulation ne doit défaire "
                                 "que le commit fautif. Recommencez avec --recommencer.")
    if d.rev(MAIN) != distant:
        raise AssertionError("Votre main et origin/main diffèrent : git push (ou git pull s'il vous manque des "
                             "commits).")
    _propre(d)


SCENARIO_5 = Scenario(
    numero=5, slug="annuler-publie", titre="Annuler un commit déjà partagé", niveau="🟡",
    commandes="log · revert · push · (pourquoi pas reset + push --force)",
    situation=(
        "Hier, pour une démo, vous avez passé le seuil de contact à 0.9 (commit « Passe le seuil à 0.9 pour la "
        "démo ») et l'avez poussé sur main. Depuis, Camille a récupéré ce commit et a poussé le sien "
        "par-dessus. La démo est finie : il faut revenir à 0.5. L'historique de main est partagé : on ne le "
        "réécrit pas."),
    mission=(
        "Sur origin/main, SEUIL vaut de nouveau 0.5 et le commit de Camille (« Ajoute un test de l'export ») "
        "est toujours là.",
        "L'historique partagé n'est pas réécrit : les commits déjà publiés restent, l'annulation est un "
        "nouveau commit.",
        "Votre main local est poussé (identique à origin/main) et `git status` est propre.",
    ),
    indices=(
        "`git log --oneline` : repérez le commit fautif. Ce n'est pas le dernier : un `git reset` ferait "
        "aussi disparaître celui de Camille.",
        "`git revert <hash>` crée un NOUVEAU commit qui fait exactement l'inverse de <hash>. L'historique ne "
        "fait qu'avancer : chacun récupère l'annulation avec un simple `git pull`.",
        "`git revert` ouvre un éditeur pour le message. Pour garder le message proposé sans éditeur : "
        "`git revert --no-edit <hash>`. Puis `git push`.",
        "Pourquoi pas `git reset --hard` + `git push --force` ? Camille a déjà ces commits : son prochain pull "
        "mélangerait les deux histoires, et son prochain push remettrait le commit fautif. On ne réécrit que "
        "ce qui n'est pas encore partagé.",
    ),
    solution=(
        Pas("git log --oneline", "le fautif est l'avant-dernier"),
        Pas("git revert --no-edit HEAD~1", "HEAD~1 = l'avant-dernier commit ; son hash marche aussi"),
        Pas("git log --oneline", "un commit « Revert … » de plus, rien d'effacé"),
        Pas("git push"),
    ),
    a_retenir=(
        "Déjà poussé ? `git revert` : on ajoute l'annulation. Pas encore poussé ? `git reset` ou "
        "`git commit --amend` sont permis : on réécrit son propre brouillon. La question à se poser : "
        "« quelqu'un d'autre a-t-il déjà ces commits ? »"),
    preparer=_preparer_5, verifier=_verifier_5,
)


# ============================================================================ 6 — Trouver le coupable avec git bisect

VERIFIER_6 = '''"""Vérifie les calculs de churn.py. Code de sortie : 0 si tout est juste, 1 sinon.

    python verifier.py
    git bisect run python verifier.py     # git teste chaque commit tout seul
"""

import sys

sys.dont_write_bytecode = True  # relit churn.py à chaque fois (pas de cache .pyc périmé)

import churn  # noqa: E402

CLIENTS = [{"mois": 3, "parti": True}, {"mois": 12, "parti": False},
           {"mois": 40, "parti": False}, {"mois": 8, "parti": True}]
SEGMENTS = {3: "nouveau", 11: "nouveau", 12: "regulier", 35: "regulier", 36: "ancien"}

erreurs = []
if churn.taux_churn(CLIENTS) != 0.5:
    erreurs.append(f"taux_churn = {churn.taux_churn(CLIENTS)} au lieu de 0.5")
for mois, attendu in SEGMENTS.items():
    obtenu = churn.segment_anciennete(mois)
    if obtenu != attendu:
        erreurs.append(f"segment_anciennete({mois}) = {obtenu!r} au lieu de {attendu!r}")

if erreurs:
    print("ECHEC : " + " ; ".join(erreurs))
    sys.exit(1)
print("OK : tous les calculs sont justes")
'''
_CLIENTS_6 = [{"mois": 3, "parti": True}, {"mois": 12, "parti": False},
              {"mois": 40, "parti": False}, {"mois": 8, "parti": True}]
_SEGMENTS_6 = {3: "nouveau", 11: "nouveau", 12: "regulier", 35: "regulier", 36: "ancien"}


def churn_correct(source: str | None) -> bool:
    """Les mêmes contrôles que verifier.py, sur une version de churn.py (lue dans un commit)."""
    try:
        espace = _executer(source, "churn.py")
        return (espace["taux_churn"](_CLIENTS_6) == 0.5
                and all(espace["segment_anciennete"](m) == s for m, s in _SEGMENTS_6.items()))
    except Exception:
        return False


_BLOCS_6 = {
    "taux_churn": '''def taux_churn(clients):
    """Part des clients partis, entre 0 et 1."""
    if not clients:
        return 0.0
    return sum(1 for c in clients if c["parti"]) / len(clients)''',
    "segment_anciennete": '''def segment_anciennete(mois):
    """Segment d'un client selon son ancienneté en mois."""
    if mois < 12:
        return "nouveau"
    if mois < 36:
        return "regulier"
    return "ancien"''',
}

# (message, fonction ajoutée ou remplacée dans churn.py, ou ligne ajoutée au README / CHANGELOG / LICENSE)
_HISTOIRE_6: list[tuple[str, str, str]] = [
    ("Ajoute anciennete_moyenne", "anciennete_moyenne", '''def anciennete_moyenne(clients):
    """Ancienneté moyenne en mois (0 s'il n'y a aucun client)."""
    if not clients:
        return 0.0
    return sum(c["mois"] for c in clients) / len(clients)'''),
    ("Ajoute une section Utilisation au README", "README.md",
     "## Utilisation\n\n    import churn\n    churn.taux_churn(clients)\n"),
    ("Ajoute clients_partis", "clients_partis", '''def clients_partis(clients):
    """Les clients partis, dans l'ordre d'origine."""
    return [c for c in clients if c["parti"]]'''),
    ("Crée le CHANGELOG", "CHANGELOG.md", "# Historique des versions\n\n## 1.0\n\n- Première version.\n"),
    ("Ajoute taux_par_segment", "taux_par_segment", '''def taux_par_segment(clients):
    """Taux de churn de chaque segment d'ancienneté."""
    groupes = {}
    for c in clients:
        groupes.setdefault(segment_anciennete(c["mois"]), []).append(c)
    return {segment: taux_churn(groupe) for segment, groupe in groupes.items()}'''),
    ("Corrige une coquille dans le README", "README.md", "Chaque client est un dictionnaire.\n"),
    ("Ajoute facture_moyenne", "facture_moyenne", '''def facture_moyenne(clients):
    """Facture mensuelle moyenne en euros (0 s'il n'y a aucun client)."""
    if not clients:
        return 0.0
    return sum(c.get("facture", 0) for c in clients) / len(clients)'''),
    ("Met à jour le CHANGELOG", "CHANGELOG.md", "- Ajout de anciennete_moyenne et facture_moyenne.\n"),
    ("Ajoute top_risques", "top_risques", '''def top_risques(clients, k=10):
    """Les k clients au plus fort risque (clé « proba »), du plus au moins risqué."""
    return sorted(clients, key=lambda c: c.get("proba", 0), reverse=True)[:k]'''),
    ("Précise les unités dans le README", "README.md", "Les durées sont en mois, les montants en euros.\n"),
    ("Ajoute part_mensuels", "part_mensuels", '''def part_mensuels(clients):
    """Part des clients en contrat mensuel."""
    if not clients:
        return 0.0
    return sum(1 for c in clients if c.get("contrat") == "Mensuel") / len(clients)'''),
    ("Ajoute un exemple au README", "README.md",
     "Exemple : `taux_churn([{\"parti\": True}, {\"parti\": False}])` vaut 0.5.\n"),
    ("Simplifie taux_churn", "taux_churn", '''def taux_churn(clients):
    """Part des clients partis, entre 0 et 1 (0 s'il n'y a aucun client)."""
    return len(clients_partis(clients)) / len(clients) if clients else 0.0'''),
    ("Ajoute clients_a_contacter", "clients_a_contacter", '''def clients_a_contacter(clients, seuil=0.5):
    """Les clients dont la probabilité de churn atteint le seuil."""
    return [c for c in clients if c.get("proba", 0) >= seuil]'''),
    ("Harmonise les noms de variables", "*", ""),  # ← le coupable (voir _harmoniser)
    ("Documente les segments d'ancienneté", "segment_anciennete", '''def segment_anciennete(anciennete_mois):
    """Segment d'un client selon son ancienneté en mois.

    nouveau : moins d'un an · regulier : de 1 à 3 ans · ancien : 3 ans et plus.
    """
    if anciennete_mois <= 12:
        return "nouveau"
    if anciennete_mois < 36:
        return "regulier"
    return "ancien"'''),
    ("Prépare la version 1.1 dans le CHANGELOG", "CHANGELOG.md", "\n## 1.1\n\n- Noms de variables harmonisés.\n"),
    ("Ajoute une FAQ au README", "README.md",
     "## FAQ\n\n- Un client sans ancienneté ? `mois` vaut 0 : il est « nouveau ».\n"),
    ("Ajoute repartition_segments", "repartition_segments", '''def repartition_segments(clients):
    """Nombre de clients dans chaque segment d'ancienneté."""
    comptes = {}
    for client in clients:
        segment = segment_anciennete(client["mois"])
        comptes[segment] = comptes.get(segment, 0) + 1
    return comptes'''),
    ("Corrige la mise en forme du CHANGELOG", "CHANGELOG.md", "- Ajout de repartition_segments.\n"),
    ("Ajoute la licence", "LICENSE", "Usage interne : atelier Git.\n"),
    ("Ajoute part_contactes", "part_contactes", '''def part_contactes(clients, seuil=0.5):
    """Part des clients à contacter au seuil donné."""
    return len(clients_a_contacter(clients, seuil)) / len(clients) if clients else 0.0'''),
    ("Met à jour le README pour la version 1.1", "README.md", "Version actuelle : 1.1.\n"),
    ("Ajoute un exemple de sortie au README", "README.md",
     "Sortie de `python verifier.py` : « OK : tous les calculs sont justes ».\n"),
]


def _harmoniser(blocs: dict[str, str]) -> None:
    """Le commit coupable : un renommage anodin… et `<` devenu `<=` au passage."""
    for nom in ("anciennete_moyenne", "clients_partis", "taux_par_segment", "facture_moyenne", "part_mensuels",
                "clients_a_contacter"):
        blocs[nom] = (blocs[nom].replace("for c in", "for client in").replace("c[", "client[")
                      .replace("c.get", "client.get").replace("(c)", "(client)").replace("[c ", "[client "))
    blocs["segment_anciennete"] = (blocs["segment_anciennete"].replace("(mois)", "(anciennete_mois)")
                                   .replace("if mois < 12", "if anciennete_mois <= 12")
                                   .replace("if mois < 36", "if anciennete_mois < 36"))


def _rendre_churn(blocs: dict[str, str]) -> str:
    return '"""Indicateurs de churn utilisés par le tableau de bord."""\n\n\n' + "\n\n\n".join(blocs.values()) + "\n"


def _preparer_6(travail: Path, distants: Path) -> None:
    d = Depot.creer(travail)
    blocs = dict(_BLOCS_6)
    textes = {"README.md": "# Indicateurs de churn\n\n"}
    d.ecrire(".gitignore", "__pycache__/\n")
    d.ecrire("README.md", textes["README.md"])
    d.ecrire("churn.py", _rendre_churn(blocs))
    d.commit("Version initiale", CAMILLE)
    d.ecrire("verifier.py", VERIFIER_6)
    d.commit("Ajoute le script de vérification", CAMILLE)
    d.git("tag", "v1.0")
    for i, (message, cible, contenu) in enumerate(_HISTOIRE_6):
        if cible == "*":
            _harmoniser(blocs)
        elif "." in cible or cible == "LICENSE":
            textes[cible] = textes.get(cible, "") + ("\n" if textes.get(cible) else "") + contenu
            d.ecrire(cible, textes[cible])
        else:
            blocs[cible] = contenu
        d.ecrire("churn.py", _rendre_churn(blocs))
        d.commit(message, VOUS if i % 3 else CAMILLE)


def _verifier_6(travail: Path, distants: Path) -> None:
    d = Depot(travail)
    if d.chemin_git("BISECT_LOG").exists():
        raise AssertionError("La recherche est encore en cours : posez le tag (git tag coupable <hash>), puis "
                             "terminez avec git bisect reset, qui vous ramène sur main.")
    _pas_d_operation_en_cours(d)
    if d.branche_courante() != "main":
        raise AssertionError("Vous n'êtes pas sur main (HEAD détachée ?) : git switch main")
    v10 = d.rev("refs/tags/v1.0")
    commits = d.git("rev-list", "--reverse", "--first-parent", f"{v10}..{MAIN}").split()
    coupable = next((c for c in commits if not churn_correct(d.fichier(c, "churn.py"))), None)
    if coupable is None:
        raise AssertionError("verifier.py passe sur main : avez-vous corrigé le bug ? La mission est seulement "
                             "de trouver le coupable. Recommencez avec --recommencer.")
    tag = d.rev("refs/tags/coupable")
    if tag is None:
        raise AssertionError("Pas encore de tag coupable. Quand git bisect affiche « <hash> is the first bad "
                             "commit » : git tag coupable <hash>.")
    if tag != coupable:
        deplacer = " Pour déplacer le tag : git tag -f coupable <hash>."
        if tag != v10 and tag not in commits:
            raise AssertionError("Le tag coupable n'est pas sur un commit de main entre v1.0 et aujourd'hui."
                                 + deplacer)
        if churn_correct(d.fichier(tag, "churn.py")):
            raise AssertionError("À ce commit, verifier.py passe encore : le bug n'y est pas, le coupable est "
                                 "plus récent." + deplacer)
        raise AssertionError("À ce commit, verifier.py échoue bien… mais il échouait déjà au commit précédent : "
                             "ce n'est pas le PREMIER mauvais commit. Laissez git bisect chercher." + deplacer)
    _propre(d)


SCENARIO_6 = Scenario(
    numero=6, slug="bisect", titre="Trouver le commit coupable (git bisect)", niveau="🟡",
    commandes="bisect start / good / bad / run · tag · show",
    situation=(
        "verifier.py contrôle les calculs de churn.py. À la version v1.0, il passait ; aujourd'hui, sur main, "
        f"il échoue. Entre les deux : {len(_HISTOIRE_6)} commits. Les relire un par un prendrait une heure ; "
        "git bisect trouve le coupable en 5 essais, en coupant l'historique en deux à chaque fois."),
    mission=(
        "Un tag coupable est posé sur le PREMIER commit où `python verifier.py` échoue.",
        "La recherche est terminée (git bisect reset) et vous êtes de retour sur main.",
        "Rien d'autre ne change : pas de nouveau commit, pas de correction (ce sera pour la suite).",
    ),
    indices=(
        "Constatez d'abord : `python verifier.py` échoue sur main. `git switch --detach v1.0`, puis "
        "`python verifier.py` : il passe. Revenez avec `git switch main`.",
        "`git bisect start main v1.0` (le mauvais d'abord, puis le bon). Git vous place au milieu : lancez "
        "`python verifier.py`, puis répondez `git bisect good` ou `git bisect bad`. Recommencez jusqu'à "
        "« … is the first bad commit ».",
        "Plus rapide : `git bisect run python verifier.py` fait tout seul (code de sortie 0 = bon, 1 = mauvais). "
        "C'est pour ça qu'un bon script de vérification renvoie un code de sortie.",
        "À la fin, refs/bisect/bad désigne le coupable : `git tag coupable refs/bisect/bad` (ou "
        "`git tag coupable <hash>`), PUIS `git bisect reset` pour revenir sur main. Méfiez-vous des messages "
        "qui ont l'air suspects : le coupable n'est pas forcément celui qu'on croit.",
    ),
    solution=(
        Pas("python verifier.py", "échoue sur main", peut_echouer=True),
        Pas("git bisect start main v1.0", "le mauvais, puis le bon"),
        Pas("git bisect run python verifier.py", "git essaie ~5 commits et nomme le coupable"),
        Pas("git tag coupable refs/bisect/bad"),
        Pas("git bisect reset", "retour sur main"),
        Pas("git show coupable", "le diff du coupable : trouvez la ligne fautive"),
    ),
    a_retenir=(
        "git bisect transforme « ça marchait avant » en « c'est CE commit », en log₂(n) essais. Des petits "
        "commits et un script de vérification qui renvoie un code de sortie, et un bug se trouve en minutes."),
    preparer=_preparer_6, verifier=_verifier_6,
)


# ============================================================================ 7 — Retrouver un travail perdu

CHURN_7 = '''"""Indicateurs de churn."""


def taux_churn(clients):
    return sum(1 for c in clients if c["parti"]) / len(clients) if clients else 0.0
'''
MODELE_V2 = ['''"""Modèle v2 : régression logistique avec interactions."""

PARAMETRES = {"C": 1.0}
''', '''"""Modèle v2 : régression logistique avec interactions."""

PARAMETRES = {"C": 0.3, "class_weight": "balanced"}
''', '''"""Modèle v2 : régression logistique avec interactions."""

PARAMETRES = {"C": 0.3, "class_weight": "balanced"}
INTERACTIONS = [("contract", "tenure_months"), ("support_calls", "tech_support")]
''']
SUJETS_EXPERIENCE = ("Essaie un modèle v2", "Ajuste les paramètres du modèle v2", "Termine le modèle v2")
SUJETS_CALIBRATION = ("Ajoute la calibration", "Documente la calibration")


def _preparer_7(travail: Path, distants: Path) -> None:
    d = Depot.creer(travail)
    d.ecrire("README.md", "# Modèle de churn\n")
    d.ecrire("churn.py", CHURN_7)
    d.commit("Version initiale")
    d.git("switch", "-q", "-c", "experience-modele")
    for sujet, contenu in zip(SUJETS_EXPERIENCE, MODELE_V2, strict=True):
        d.ecrire("modele_v2.py", contenu)
        d.commit(sujet)
    d.git("switch", "-q", "main")
    d.ecrire("calibration.py", '"""Calibration des probabilités (méthode isotonique)."""\n\nMETHODE = "isotonic"\n')
    d.commit(SUJETS_CALIBRATION[0])
    d.ecrire("README.md", "# Modèle de churn\n\nLes probabilités sont calibrées : voir calibration.py.\n")
    d.commit(SUJETS_CALIBRATION[1])
    d.git("branch", "-q", "-D", "experience-modele")  # « elle était fusionnée, non ? »
    d.git("reset", "-q", "--hard", "HEAD~2")           # « trouvé sur un forum »


def _verifier_7(travail: Path, distants: Path) -> None:
    d = Depot(travail)
    _pas_d_operation_en_cours(d)
    branche = "refs/heads/experience-modele"
    if d.rev(branche) is None:
        raise AssertionError("La branche experience-modele n'existe pas encore. `git reflog` : cherchez la ligne "
                             f"« commit: {SUJETS_EXPERIENCE[-1]} », puis git branch experience-modele <son hash>.")
    if not d.contient_sujet(branche, SUJETS_EXPERIENCE[-1]):
        raise AssertionError(f"experience-modele existe mais ne contient pas « {SUJETS_EXPERIENCE[-1]} » : elle "
                             "pointe sur le mauvais commit. Déplacez-la : git branch -f experience-modele <hash>.")
    manquants = [s for s in SUJETS_CALIBRATION if not d.contient_sujet(MAIN, s)]
    if manquants:
        raise AssertionError(
            f"main ne contient pas {' ni '.join('« ' + s + ' »' for s in manquants)}. Dans `git reflog`, la ligne "
            "juste en dessous de « reset: moving to HEAD~2 » montre où était main avant l'accident. Sur main, "
            "avec un `git status` propre : git reset --hard <ce hash>.")
    _propre(d)


SCENARIO_7 = Scenario(
    numero=7, slug="reflog", titre="Retrouver un travail « perdu » (git reflog)", niveau="🔴",
    commandes="reflog · branch <nom> <hash> · reset --hard <hash>",
    situation=(
        "Mauvaise journée. Vous avez supprimé la branche experience-modele (git branch -D : trois commits sur un "
        "modèle v2), en la croyant fusionnée. Puis, voulant jeter une modification, vous avez copié d'un forum "
        "`git reset --hard HEAD~2` : main a perdu vos deux derniers commits sur la calibration. `git log` ne "
        "les montre plus. Pas de panique : git tient un journal de tous les endroits où HEAD est passé."),
    mission=(
        "La branche experience-modele existe de nouveau, avec ses 3 commits (le dernier : « Termine le modèle "
        "v2 »).",
        "main contient de nouveau « Ajoute la calibration » et « Documente la calibration ».",
        "`git status` est propre.",
    ),
    indices=(
        "`git reflog` liste, du plus récent au plus ancien, chaque déplacement de HEAD (commit, changement de "
        "branche, reset…), avec le hash du commit à chaque étape.",
        "Cherchez la ligne « commit: Termine le modèle v2 » : son hash est le dernier commit de la branche "
        "supprimée. `git branch experience-modele <hash>` la recrée : une branche n'est qu'une étiquette.",
        "Pour main : la ligne juste en dessous de « reset: moving to HEAD~2 » montre où était main avant "
        "l'accident (« commit: Documente la calibration »). `git reset --hard <hash>` y ramène main.",
        "Le reflog est local et temporaire (90 jours par défaut), et il ne connaît que ce qui a été commité : "
        "une modification jamais commitée puis effacée par `reset --hard` est perdue pour de bon. Commitez "
        "souvent, quitte à regrouper ensuite (scénario 8).",
    ),
    solution=(
        Pas("git reflog", "tout l'historique de HEAD, même ce que git log ne montre plus"),
        Pas("git branch experience-modele <hash de « commit: Termine le modèle v2 »>",
            action=lambda d: d.git("branch", "experience-modele", _hash_dans_reflog(d, SUJETS_EXPERIENCE[-1]))),
        Pas("git reset --hard <hash de « commit: Documente la calibration »>",
            action=lambda d: d.git("reset", "-q", "--hard", _hash_dans_reflog(d, SUJETS_CALIBRATION[-1]))),
        Pas("git log --oneline --graph --all", "tout est revenu"),
    ),
    a_retenir=(
        "Un commit ne se perd (presque) jamais : `git reflog` retrouve ce que `git log` ne montre plus. Ce qui "
        "n'a jamais été commité, en revanche, `reset --hard` l'efface pour de bon : commitez tôt, commitez "
        "souvent."),
    preparer=_preparer_7, verifier=_verifier_7,
)


# ============================================================================ 8 — Préparer sa branche pour la revue

EXPORT_8 = ['''"""Export Excel des clients à risque."""


def lignes_export(clients, seuil):
    pass
''', '''"""Export Excel des clients à risque."""


def lignes_export(clients, seuil):
    """Les lignes à exporter : les clients au-dessus du seuil, du plus risqé au moins risqué."""
    a_risque = [c for c in clients if c["proba"] >= seuil]
    a_risque.sort(key=lambda c: c["proba"], reverse=True)
    return [{"Client": c["id"], "Probabilité": f"{c['proba']:.0%}"} for c in a_risque]
''']
EXPORT_8.append(EXPORT_8[1].replace("risqé", "risqué"))

TEST_EXPORT_8 = '''from export import lignes_export


def test_lignes_export_filtre_et_trie():
    clients = [{"id": "A", "proba": 0.2}, {"id": "B", "proba": 0.9}, {"id": "C", "proba": 0.6}]
    assert [ligne["Client"] for ligne in lignes_export(clients, seuil=0.5)] == ["B", "C"]
'''
REQUIREMENTS_8 = "pandas>=2.1\n"
REQUIREMENTS_8_CAMILLE = "pandas>=2.1\nopenpyxl>=3.1\n"
SUJET_CAMILLE_8 = "Ajoute openpyxl aux dépendances"


def _preparer_8(travail: Path, distants: Path) -> None:
    origine = Depot.creer_nu(distants / "origine.git")
    camille = Depot.creer(distants / "camille")
    camille.ecrire("README.md", "# Tableau de bord\n\nToute branche passe en revue avant d'arriver sur main.\n")
    camille.ecrire("requirements.txt", REQUIREMENTS_8)
    camille.commit("Version initiale", CAMILLE)
    _publier(camille, origine)
    d = Depot.cloner(origine, travail)
    d.git("switch", "-q", "-c", "export-excel")
    for sujet, (chemin, contenu) in zip(("wip", "ajoute export", "fix typo", "wip tests"),
                                        [("export.py", c) for c in EXPORT_8] + [("test_export.py", TEST_EXPORT_8)],
                                        strict=True):
        d.ecrire(chemin, contenu)
        d.commit(sujet)
    d.git("push", "-q", "--no-verify", "-u", "origin", "export-excel")
    camille.ecrire("requirements.txt", REQUIREMENTS_8_CAMILLE)
    camille.commit(SUJET_CAMILLE_8, CAMILLE)
    camille.git("push", "-q", "--no-verify")


def _verifier_8(travail: Path, distants: Path) -> None:
    d, origine, camille = Depot(travail), Depot(distants / "origine.git"), Depot(distants / "camille")
    _pas_d_operation_en_cours(d)
    dernier_main = camille.rev(MAIN)
    if origine.rev(MAIN) != dernier_main:
        raise AssertionError("origin/main a changé : on ne pousse pas sur main sans revue (et jamais en force). "
                             "Recommencez avec --recommencer.")
    branche = d.rev("refs/heads/export-excel")
    if branche is None:
        raise AssertionError("La branche export-excel n'existe plus dans votre dépôt. git reflog la retrouvera "
                             "(scénario 7), sinon --recommencer.")
    if d.rev(dernier_main) is None:
        raise AssertionError(f"Votre dépôt ne connaît pas encore le commit de Camille « {SUJET_CAMILLE_8} » : "
                             "git fetch, puis git rebase origin/main.")
    if not d.est_ancetre(dernier_main, branche):
        raise AssertionError(f"export-excel ne part pas du dernier origin/main (« {SUJET_CAMILLE_8} » n'est pas "
                             "dans son historique) : git fetch, puis git rebase origin/main.")
    if d.git("rev-list", "--merges", f"{dernier_main}..{branche}"):
        raise AssertionError("La branche contient un commit de fusion (git merge ou git pull) : l'équipe veut un "
                             "historique linéaire. `git reset --soft origin/main` puis un seul `git commit` "
                             "remplace le tout par un commit propre.")
    commits = d.sujets(f"{dernier_main}..{branche}")
    if not commits:
        raise AssertionError("La branche ne contient plus aucun commit par rapport à main : votre travail a "
                             "disparu ? git reflog (scénario 7), sinon --recommencer.")
    if len(commits) > 2:
        raise AssertionError(f"La branche contient {len(commits)} commits : regroupez-les en 1 ou 2. "
                             "`git reset --soft origin/main` garde tout votre travail prêt à commiter, puis un "
                             "`git commit -m \"…\"` (ou `git rebase -i origin/main`, avec fixup).")
    _verifier_messages(d, f"{dernier_main}..{branche}")
    attendus = {"export.py": EXPORT_8[-1], "test_export.py": TEST_EXPORT_8, "requirements.txt": REQUIREMENTS_8_CAMILLE}
    for chemin, attendu in attendus.items():
        if not _identique(d, branche, chemin, attendu):
            raise AssertionError(f"{chemin} n'est plus identique à la version d'avant le nettoyage : réorganiser "
                                 "l'historique ne doit rien changer au contenu.")
    distante = origine.rev("refs/heads/export-excel")
    if distante != branche:
        raise AssertionError("origin/export-excel n'est pas votre nouvelle version : git push --force-with-lease "
                             "(un push normal est refusé, puisque l'historique de la branche a changé).")
    _propre(d)


SCENARIO_8 = Scenario(
    numero=8, slug="rebase-revue", titre="Préparer sa branche pour la revue", niveau="🔴",
    commandes="fetch · rebase · reset --soft · rebase -i · push --force-with-lease",
    situation=(
        "Votre branche export-excel est prête pour la revue (pull request). Mais son historique raconte vos "
        "tâtonnements : « wip », « ajoute export », « fix typo », « wip tests ». Et pendant ce temps, Camille a "
        "poussé un commit sur main. Avant de demander une revue, l'équipe attend une branche à jour et "
        "lisible."),
    mission=(
        "export-excel part du DERNIER origin/main (avec « Ajoute openpyxl aux dépendances »), sans commit de "
        "fusion.",
        "Elle contient 1 ou 2 commits au plus, aux messages clairs (plus de « wip » ni de « fix typo »).",
        "Le contenu est inchangé : export.py et test_export.py comme dans votre dernier commit.",
        "La branche est poussée sur origin (origin/export-excel = votre branche), et main n'a pas bougé.",
    ),
    indices=(
        "`git fetch` récupère les nouveautés d'origin sans rien toucher chez vous. Ensuite, "
        "`git log --oneline --graph --all` montre que origin/main a avancé.",
        "`git rebase origin/main` rejoue vos 4 commits par-dessus le dernier main : l'historique reste "
        "linéaire (au contraire de `git merge`, qui ajouterait un commit de fusion).",
        "Pour regrouper : `git reset --soft origin/main` recule la branche mais garde toutes vos modifications "
        "prêtes à commiter ; un seul `git commit -m \"…\"` les rassemble. Autre méthode : "
        "`git rebase -i origin/main`, en remplaçant pick par fixup.",
        "La branche était déjà poussée : son historique a changé, un `git push` normal est refusé. "
        "`git push --force-with-lease` remplace la branche distante seulement si personne n'y a poussé "
        "entre-temps. On ne force jamais sur main, ni sur la branche de quelqu'un d'autre.",
    ),
    solution=(
        Pas("git fetch"),
        Pas("git log --oneline --graph --all", "origin/main a avancé"),
        Pas("git rebase origin/main", "vos 4 commits rejoués par-dessus"),
        Pas("git reset --soft origin/main", "on recule, le travail reste prêt à commiter"),
        Pas('git commit -m "Ajoute l\'export Excel des clients à risque, avec son test"'),
        Pas("git push --force-with-lease"),
        Pas("git log --oneline --graph --all", "un seul commit, au-dessus du dernier main"),
    ),
    a_retenir=(
        "Réécrire l'historique est un outil, pas un tabou : sur VOTRE branche, avant la revue, rebase et "
        "reset --soft racontent une histoire claire. Sur une branche partagée, jamais (scénario 5). Et "
        "toujours --force-with-lease, jamais --force."),
    preparer=_preparer_8, verifier=_verifier_8,
)


SCENARIOS: tuple[Scenario, ...] = (SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4, SCENARIO_5, SCENARIO_6,
                                   SCENARIO_7, SCENARIO_8)

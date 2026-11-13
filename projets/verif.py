"""Moteur de correction des projets guidés (vous n'avez pas besoin de modifier ce fichier).

Un projet est un paquet Python découpé en étapes : chaque étape est un fichier (`clients.py`,
`valeur.py`…) dont on vérifie une ou plusieurs fonctions. Contrairement aux défis, les étapes
s'appuient les unes sur les autres, comme dans un vrai logiciel : l'étape 6 appelle le code
des étapes 1 à 5.
"""

from __future__ import annotations

import importlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

from defis.verif import A_FAIRE, ECHEC, ERREUR, ICONES, REUSSI  # mêmes statuts et icônes que les défis

__all__ = ["A_FAIRE", "ECHEC", "ERREUR", "ICONES", "REUSSI", "Etape", "Projet", "Resultat",
           "evaluer", "evaluer_etape", "projets"]


@dataclass(frozen=True)
class Etape:
    numero: int
    titre: str
    niveau: str                       # 🟢 🟡 🔴
    module: str                       # fichier du paquet, sans « .py » (ex. "clients")
    fonctions: tuple[str, ...]        # fonctions à écrire dans ce fichier
    verifier: Callable[[ModuleType], None]  # lève AssertionError (message explicatif) si faux


@dataclass(frozen=True)
class Projet:
    numero: int           # le projet 1 est le tableau de bord du churn (racine du dépôt)
    code: str
    titre: str
    paquet_eleve: str     # ex. "projets.retention" : le code que vous complétez
    paquet_corrige: str   # ex. "projets.retention_corrige" : le projet terminé
    commande: str         # comment lancer le programme une fois terminé
    etapes: tuple[Etape, ...]

    def dossier(self, corrige: bool = False) -> str:
        return (self.paquet_corrige if corrige else self.paquet_eleve).replace(".", "/") + "/"


@dataclass(frozen=True)
class Resultat:
    etape: Etape
    statut: str
    message: str = ""


def _dans_le_paquet(frame: traceback.FrameSummary, module: ModuleType) -> bool:
    return Path(frame.filename).resolve().parent == Path(module.__file__).resolve().parent


def _non_ecrite(exc: NotImplementedError, module: ModuleType, etape: Etape,
                etapes: tuple[Etape, ...]) -> Resultat:
    """Une fonction pas encore écrite : celle de l'étape (⬜), ou celle d'une étape précédente (❌)."""
    pile = traceback.extract_tb(exc.__traceback__)
    origine = pile[-1] if pile else None
    if origine is None or not _dans_le_paquet(origine, module) or Path(origine.filename).stem == etape.module:
        return Resultat(etape, A_FAIRE)
    fichier = Path(origine.filename).stem
    autre = next((e for e in etapes if e.module == fichier), None)
    quand = f"l'étape {autre.numero}" if autre else f"le fichier {fichier}.py"
    return Resultat(etape, ECHEC, f"cette étape utilise `{origine.name}` ({fichier}.py), qui n'est pas encore "
                                  f"écrite : terminez d'abord {quand}.")


def _plantage(exc: Exception, module: ModuleType) -> str:
    """Le message d'un plantage, situé dans VOTRE code si possible (fichier et ligne)."""
    pile = traceback.extract_tb(exc.__traceback__)
    chez_vous = [f for f in pile if _dans_le_paquet(f, module)]
    lieu = chez_vous[-1] if chez_vous else (pile[-1] if pile else None)
    ou = f" ({Path(lieu.filename).name}, ligne {lieu.lineno}, dans `{lieu.name}`)" if lieu else ""
    return f"{type(exc).__name__}: {exc}{ou}"


def evaluer_etape(paquet: str, etape: Etape, etapes: tuple[Etape, ...] = ()) -> Resultat:
    try:
        module = importlib.import_module(f"{paquet}.{etape.module}")
    except Exception as exc:  # erreur de syntaxe, import raté…
        return Resultat(etape, ERREUR, f"impossible d'importer {etape.module}.py — {type(exc).__name__}: {exc}")
    absentes = [f for f in etape.fonctions if not callable(getattr(module, f, None))]
    if absentes:
        return Resultat(etape, ERREUR, f"introuvable dans {etape.module}.py : {', '.join(absentes)}")
    try:
        etape.verifier(module)
    except NotImplementedError as exc:
        return _non_ecrite(exc, module, etape, etapes)
    except AssertionError as exc:
        return Resultat(etape, ECHEC, str(exc) or "résultat incorrect")
    except Exception as exc:  # votre code a planté : on montre où
        return Resultat(etape, ERREUR, _plantage(exc, module))
    return Resultat(etape, REUSSI)


def evaluer(projet: Projet, corrige: bool = False) -> list[Resultat]:
    paquet = projet.paquet_corrige if corrige else projet.paquet_eleve
    return [evaluer_etape(paquet, e, projet.etapes) for e in projet.etapes]


def projets() -> list[Projet]:
    """Tous les projets guidés, dans l'ordre. Les nouveaux projets s'ajoutent ici."""
    from .checks_campagne import PROJET as campagne
    from .checks_reentrainement import PROJET as reentrainement
    from .checks_retention import PROJET as retention
    from .checks_service import PROJET as service

    return [retention, campagne, service, reentrainement]

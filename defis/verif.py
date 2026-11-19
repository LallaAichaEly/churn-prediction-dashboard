"""Moteur de correction des défis (vous n'avez pas besoin de modifier ce fichier).

Chaque défi est associé à une fonction de vérification qui reçoit VOTRE fonction et lève
une `AssertionError` avec un message explicatif si le résultat est faux.
"""

from __future__ import annotations

import importlib
import traceback
from dataclasses import dataclass
from types import ModuleType
from typing import Callable

A_FAIRE, REUSSI, ECHEC, ERREUR = "a_faire", "reussi", "echec", "erreur"
ICONES = {REUSSI: "✅", ECHEC: "❌", ERREUR: "💥", A_FAIRE: "⬜"}


@dataclass(frozen=True)
class Defi:
    numero: int
    fonction: str          # nom de la fonction à écrire
    niveau: str            # 🟢 🟡 🔴
    parcours: str          # exercice correspondant dans PARCOURS.md ("—" si aucun)
    verifier: Callable     # verifier(votre_fonction) -> None, lève AssertionError si faux


@dataclass(frozen=True)
class Resultat:
    defi: Defi
    statut: str
    message: str = ""


@dataclass(frozen=True)
class Serie:
    code: str
    titre: str
    module_eleve: str      # le fichier que vous complétez
    module_corrige: str
    defis: tuple[Defi, ...]


def evaluer_defi(module: ModuleType, defi: Defi) -> Resultat:
    fonction = getattr(module, defi.fonction, None)
    if fonction is None:
        return Resultat(defi, ERREUR, f"fonction `{defi.fonction}` introuvable dans {module.__name__}")
    try:
        defi.verifier(fonction)
    except NotImplementedError:
        return Resultat(defi, A_FAIRE)
    except AssertionError as exc:
        return Resultat(defi, ECHEC, str(exc) or "résultat incorrect")
    except Exception as exc:  # votre code a planté : on montre où
        ligne = traceback.extract_tb(exc.__traceback__)[-1]
        return Resultat(defi, ERREUR, f"{type(exc).__name__}: {exc} (ligne {ligne.lineno})")
    return Resultat(defi, REUSSI)


def evaluer(serie: Serie, corrige: bool = False) -> list[Resultat]:
    nom = serie.module_corrige if corrige else serie.module_eleve
    try:
        module = importlib.import_module(nom)
    except Exception as exc:  # le fichier ne se charge pas : on le dit pour CETTE série, les autres restent corrigées
        message = f"{nom.replace('.', '/')}.py ne se charge pas : {_erreur_au_chargement(exc, nom)}"
        return [Resultat(d, ERREUR, message) for d in serie.defis]
    return [evaluer_defi(module, d) for d in serie.defis]


def _erreur_au_chargement(exc: Exception, nom: str) -> str:
    """L'erreur levée en chargeant le fichier, avec la ligne fautive DANS ce fichier.

    Une faute de syntaxe, mais aussi une erreur levée en DÉFINISSANT une classe (par exemple une liste
    comme valeur par défaut d'une dataclass : `options: list = []`) empêche tout le fichier de se charger.
    """
    if isinstance(exc, SyntaxError):
        return f"{type(exc).__name__}: {exc.msg} (ligne {exc.lineno})"
    fichier = nom.replace(".", "/") + ".py"
    lignes = [cadre.lineno for cadre in traceback.extract_tb(exc.__traceback__)
              if cadre.filename.replace("\\", "/").endswith(fichier)]
    ou = f" (ligne {lignes[-1]})" if lignes else ""
    return f"{type(exc).__name__}: {exc}{ou}"


def series() -> list[Serie]:
    """Toutes les séries disponibles, dans l'ordre. Les nouvelles séries s'ajoutent ici.

    La série 00 (remise à niveau) est arrivée après les autres : ses défis portent les numéros
    30 à 37, mais elle est affichée en premier, car c'est par elle qu'on commence.
    """
    from .checks_serie_00 import SERIE as serie_00
    from .checks_serie_01 import SERIE as serie_01
    from .checks_serie_02 import SERIE as serie_02
    from .checks_serie_03 import SERIE as serie_03
    from .checks_serie_04 import SERIE as serie_04
    from .checks_serie_05 import SERIE as serie_05
    from .checks_serie_06 import SERIE as serie_06
    from .checks_serie_07 import SERIE as serie_07
    from .checks_serie_08 import SERIE as serie_08
    from .checks_serie_09 import SERIE as serie_09
    from .checks_serie_10 import SERIE as serie_10
    from .checks_serie_11 import SERIE as serie_11
    from .checks_serie_12 import SERIE as serie_12
    from .checks_serie_13 import SERIE as serie_13
    from .checks_serie_14 import SERIE as serie_14
    from .checks_serie_15 import SERIE as serie_15
    from .checks_serie_16 import SERIE as serie_16
    from .checks_serie_17 import SERIE as serie_17
    from .checks_serie_18 import SERIE as serie_18
    from .checks_serie_19 import SERIE as serie_19
    from .checks_serie_20 import SERIE as serie_20
    from .checks_serie_21 import SERIE as serie_21
    from .checks_serie_22 import SERIE as serie_22
    from .checks_serie_23 import SERIE as serie_23
    from .checks_serie_24 import SERIE as serie_24
    from .checks_serie_25 import SERIE as serie_25
    from .checks_serie_26 import SERIE as serie_26
    from .checks_serie_27 import SERIE as serie_27
    from .checks_serie_28 import SERIE as serie_28
    from .checks_serie_29 import SERIE as serie_29
    from .checks_serie_30 import SERIE as serie_30
    from .checks_serie_31 import SERIE as serie_31
    from .checks_serie_32 import SERIE as serie_32
    from .checks_serie_33 import SERIE as serie_33
    from .checks_serie_34 import SERIE as serie_34
    from .checks_serie_35 import SERIE as serie_35
    from .checks_serie_36 import SERIE as serie_36
    from .checks_serie_37 import SERIE as serie_37
    from .checks_serie_38 import SERIE as serie_38
    from .checks_serie_39 import SERIE as serie_39
    from .checks_serie_40 import SERIE as serie_40
    from .checks_serie_41 import SERIE as serie_41
    from .checks_serie_42 import SERIE as serie_42

    return [serie_00, serie_01, serie_02, serie_03, serie_04, serie_05, serie_06, serie_07, serie_08, serie_09,
            serie_10, serie_11, serie_12, serie_13, serie_14, serie_15, serie_16, serie_17,
            serie_18, serie_19, serie_20, serie_21, serie_22, serie_23, serie_24, serie_25, serie_26, serie_27,
            serie_28, serie_29, serie_30, serie_31, serie_32, serie_33, serie_34, serie_35, serie_36, serie_37,
            serie_38, serie_39, serie_40, serie_41, serie_42]

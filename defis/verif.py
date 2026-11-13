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
    module = importlib.import_module(serie.module_corrige if corrige else serie.module_eleve)
    return [evaluer_defi(module, d) for d in serie.defis]


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

    return [serie_00, serie_01, serie_02, serie_03, serie_04, serie_05, serie_06, serie_07, serie_08, serie_09,
            serie_10, serie_11, serie_12, serie_13, serie_14, serie_15]

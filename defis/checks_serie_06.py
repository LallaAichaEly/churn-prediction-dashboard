"""Vérifications automatiques de la série 06 (développer comme un·e pro).

Les messages visent les réflexes qui manquent souvent quand on débute en développement :
lire une traceback par le haut, tester un seul exemple, attraper toutes les exceptions,
« améliorer » un code en changeant son résultat, oublier `type=` dans argparse, se tromper
d'un cran dans une boucle.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import io
import math
import sys
import textwrap
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd

from .verif import Defi, Serie


def _pas_none(resultat, quoi: str):
    if resultat is None:
        raise AssertionError(f"{quoi} : la fonction a retourné None. Avez-vous oublié `return` ?")
    return resultat


def _source(f) -> ast.FunctionDef | None:
    """L'arbre syntaxique de la fonction de l'apprenant·e (None si le code source est introuvable)."""
    try:
        arbre = ast.parse(textwrap.dedent(inspect.getsource(f)))
    except (OSError, TypeError, SyntaxError):
        return None
    return next((n for n in ast.walk(arbre) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)


# ============================================================================ Défi 38 : traceback

_TB_DOCSTRING = """\
Traceback (most recent call last):
  File "/projet/scripts/train.py", line 52, in <module>
    main()
  File "/projet/scripts/train.py", line 31, in main
    df = load_or_create(ROOT / "data" / "customers.csv")
  File "/projet/src/churn/data.py", line 118, in load_or_create
    return validate(pd.read_csv(path))
  File "/projet/src/churn/data.py", line 97, in validate
    if df["contrat"].isna().any():
       ~~^^^^^^^^^^
KeyError: 'contrat'
"""

_TB_WINDOWS = """\
Traceback (most recent call last):
  File "C:\\Users\\moi\\churn\\app.py", line 41, in <module>
    seuil = float(st.text_input("Seuil"))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: could not convert string to float: '0,4'
"""

_TB_CHAINEE = """\
Traceback (most recent call last):
  File "/projet/src/churn/model.py", line 60, in load
    return joblib.load(path)
  File "/usr/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 650, in load
    with open(filename, 'rb') as f:
FileNotFoundError: [Errno 2] No such file or directory: 'models/churn_model.joblib'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/projet/api.py", line 22, in <module>
    MODEL = load_model()
  File "/projet/api.py", line 18, in load_model
    raise RuntimeError("Modèle absent : lancez d'abord python scripts/train.py")
RuntimeError: Modèle absent : lancez d'abord python scripts/train.py
"""

_TB_LIBRAIRIE = """\
Traceback (most recent call last):
  File "/projet/scripts/train.py", line 29, in main
    df = pd.read_csv(chemin)
  File "/usr/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
pandas.errors.EmptyDataError: No columns to parse from file
"""

_TB_SANS_MESSAGE = """\
Traceback (most recent call last):
  File "/projet/scripts/defis.py", line 12, in <module>
    premier = next(iter([]))
StopIteration
"""

_TRACEBACKS = [
    ("l'exemple de la docstring", _TB_DOCSTRING,
     {"fichier": "/projet/src/churn/data.py", "ligne": 97, "fonction": "validate",
      "erreur": "KeyError", "message": "'contrat'"}),
    ("une traceback Windows", _TB_WINDOWS,
     {"fichier": "C:\\Users\\moi\\churn\\app.py", "ligne": 41, "fonction": "<module>",
      "erreur": "ValueError", "message": "could not convert string to float: '0,4'"}),
    ("une erreur levée dans pandas", _TB_LIBRAIRIE,
     {"fichier": "/usr/lib/python3.11/site-packages/pandas/io/parsers/readers.py", "ligne": 1026,
      "fonction": "read_csv", "erreur": "pandas.errors.EmptyDataError", "message": "No columns to parse from file"}),
    ("une erreur sans message", _TB_SANS_MESSAGE,
     {"fichier": "/projet/scripts/defis.py", "ligne": 12, "fonction": "<module>",
      "erreur": "StopIteration", "message": ""}),
    ("deux erreurs enchaînées", _TB_CHAINEE,
     {"fichier": "/projet/api.py", "ligne": 18, "fonction": "load_model",
      "erreur": "RuntimeError", "message": "Modèle absent : lancez d'abord python scripts/train.py"}),
]
_CLES = ["fichier", "ligne", "fonction", "erreur", "message"]


def _diagnostic_traceback(res: dict, attendu: dict, texte: str, nom: str) -> None:
    def ecart(cle: str) -> str:
        return f"Sur {nom}, « {cle} » vaut {res[cle]!r} au lieu de {attendu[cle]!r}."

    if res["ligne"] != attendu["ligne"] and str(res["ligne"]) == str(attendu["ligne"]):
        raise AssertionError(f"{ecart('ligne')} C'est du texte : convertissez-le avec int(…).")
    premier = texte.split('File "', 1)[1].split('"', 1)[0]
    if res["fichier"] == premier != attendu["fichier"]:
        raise AssertionError(
            f"{ecart('fichier')} Vous avez pris le PREMIER appel. « most recent call last » : le plus "
            "récent, celui qui a planté, est le DERNIER de la liste. Gardez la dernière ligne « File … »."
        )
    if "During handling" in texte and res["erreur"] != attendu["erreur"]:
        raise AssertionError(
            f"{ecart('erreur')} Quand une erreur survient pendant qu'on en traite une autre, Python "
            "affiche les deux : celle qui a arrêté le programme est TOUT EN BAS. Lisez la traceback par la fin."
        )
    for cle in ("fichier", "ligne", "fonction"):
        if res[cle] != attendu[cle]:
            indice = ""
            if isinstance(res[cle], str) and res[cle].strip('"') == attendu[cle]:
                indice = " Retirez les guillemets."
            elif isinstance(res[cle], str) and res[cle].strip() == attendu[cle]:
                indice = " Retirez les espaces avec .strip()."
            elif cle == "fichier" and ":" in attendu["fichier"]:
                indice = (" Un chemin Windows contient lui-même « : » (C:\\…) : découpez la ligne autour de "
                          "'\", line ' plutôt que sur « : »." )
            raise AssertionError(ecart(cle) + indice)
    if res["erreur"] != attendu["erreur"]:
        indice = ""
        if isinstance(res["erreur"], str) and ":" in res["erreur"]:
            indice = " Le nom de l'erreur s'arrête au premier « : »."
        elif not attendu["message"]:
            indice = " Une erreur peut n'avoir AUCUN message : la dernière ligne n'a alors pas de « : »."
        raise AssertionError(ecart("erreur") + indice)
    if res["message"] != attendu["message"]:
        indice = ""
        if isinstance(res["message"], str) and res["message"].strip() == attendu["message"]:
            indice = " Retirez l'espace qui suit « : » avec .strip()."
        elif isinstance(res["message"], str) and attendu["message"].startswith(res["message"].strip()):
            indice = (" Le message contient lui-même « : » : ne coupez qu'au PREMIER, avec "
                      "texte.split(\":\", 1) ou texte.partition(\":\").")
        raise AssertionError(ecart("message") + indice)


def verifier_analyser_traceback(f) -> None:
    for nom, texte, attendu in _TRACEBACKS:
        try:
            res = _pas_none(f(texte), "analyser_traceback")
        except (ValueError, IndexError, StopIteration) as exc:
            indice = ""
            if not attendu["message"]:
                indice = (" La dernière ligne n'a pas toujours de « : » : une erreur peut n'avoir AUCUN message. "
                          "texte.partition(\":\") ne plante jamais, contrairement à texte.split(\": \", 1) qu'on déballe.")
            raise AssertionError(f"Sur {nom}, votre fonction plante ({type(exc).__name__}: {exc}).{indice}") from None
        assert isinstance(res, dict), f"Attendu un dictionnaire, obtenu {type(res).__name__}"
        manquantes = [c for c in _CLES if c not in res]
        assert not manquantes, f"Clés manquantes dans le dictionnaire : {manquantes} (attendues : {_CLES})"
        _diagnostic_traceback(res, attendu, texte, nom)
        assert res["ligne"].__class__ is int, f"« ligne » doit être un int, obtenu {type(res['ligne']).__name__}"


# ============================================================================ Défi 39 : écrire un test


def _taux_correct(partis: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(100 * partis / total, 1)


def _taux_proportion(partis: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(partis / total, 3)


def _taux_plante_si_vide(partis: int, total: int) -> float:
    return round(100 * partis / total, 1)


def _taux_arrondi_entier(partis: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(100 * partis / total)


def _taux_toujours_35(partis: int, total: int) -> float:
    return 35.0


def _taux_sans_arrondi(partis: int, total: int) -> float:
    if total == 0:
        return 0.0
    return 100 * partis / total


_MUTANTS = [
    (_taux_proportion, "renvoie une PROPORTION (0.35) au lieu d'un pourcentage (35.0)",
     "Testez un cas normal et comparez au pourcentage attendu."),
    (_taux_plante_si_vide, "plante (ZeroDivisionError) quand il n'y a aucun client",
     "Testez le cas limite : taux_en_pourcentage(0, 0) doit valoir 0.0."),
    (_taux_arrondi_entier, "arrondit à l'unité (33 au lieu de 33.3)",
     "Choisissez un exemple dont le résultat n'est pas rond, par exemple 1 client sur 3."),
    (_taux_toujours_35, "renvoie toujours 35.0, quels que soient les arguments",
     "Un seul exemple ne suffit pas : ajoutez un deuxième cas normal avec un autre résultat."),
    (_taux_sans_arrondi, "n'arrondit pas (33.333… au lieu de 33.3)",
     "La fonction promet un arrondi à 1 décimale : vérifiez-le avec une valeur exacte, ex. == 33.3."),
]


def verifier_tester_taux_en_pourcentage(test) -> None:
    try:
        test(_taux_correct)
    except NotImplementedError:
        raise
    except AssertionError as exc:
        raise AssertionError(
            f"Votre test ÉCHOUE sur la version CORRECTE de la fonction ({str(exc) or 'un assert sans message'}). "
            "Un test qui crie au loup est pire que pas de test : vérifiez vos valeurs attendues "
            "(ex. taux_en_pourcentage(1, 3) vaut 33.3)."
        ) from None
    except Exception as exc:
        raise AssertionError(
            f"Votre test plante sur la version CORRECTE : {type(exc).__name__}: {exc}. "
            "Appelez la fonction reçue en paramètre avec deux entiers : taux_en_pourcentage(35, 100)."
        ) from None

    survivants = []
    for mutant, defaut, conseil in _MUTANTS:
        try:
            test(mutant)
        except Exception:  # noqa: BLE001 — n'importe quelle erreur = le test a détecté le bug
            continue
        survivants.append((defaut, conseil))
    if survivants:
        if len(survivants) == len(_MUTANTS):
            raise AssertionError(
                "Votre test passe sur TOUTES les versions boguées : il ne vérifie rien. Utilisez "
                "`assert taux_en_pourcentage(35, 100) == 35.0` pour comparer le résultat à la valeur attendue."
            )
        defaut, conseil = survivants[0]
        raise AssertionError(
            f"{len(_MUTANTS) - len(survivants)}/{len(_MUTANTS)} bugs détectés. Votre test laisse passer une version "
            f"qui {defaut}. {conseil}"
        )


# ============================================================================ Défi 40 : lire_seuil


def _seuil_refuse(f, texte: str, pourquoi: str) -> None:
    try:
        obtenu = f(texte)
    except ValueError as exc:
        saisie = texte.strip().rstrip("%").strip()  # « 150 % » : citer 150 suffit
        assert saisie in str(exc), (
            f"lire_seuil({texte!r}) lève bien une ValueError, mais son message ({exc}) ne cite pas la saisie. "
            "Un bon message dit CE QUI a été reçu : f\"Seuil illisible : {texte!r}\"."
        )
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"lire_seuil({texte!r}) lève {type(exc).__name__} : attendu une ValueError ({pourquoi})."
        ) from None
    if isinstance(obtenu, float) and math.isnan(obtenu):
        raise AssertionError(
            f"lire_seuil({texte!r}) renvoie nan. float('nan') ne plante pas, et toute comparaison avec nan est "
            "FAUSSE : `nan < 0` et `nan > 1` valent False. Écrivez `if not 0 <= valeur <= 1` (ou testez "
            "math.isfinite) pour le refuser."
        )
    if texte == "1.5" and isinstance(obtenu, (int, float)) and math.isclose(obtenu, 0.015):
        raise AssertionError(
            "lire_seuil('1.5') renvoie 0.015 : sans signe %, un nombre supérieur à 1 est une faute de saisie, "
            "pas un pourcentage. Deviner, c'est transformer une faute de frappe en seuil de 1,5 % sans prévenir."
        )
    if texte == "abc":
        raise AssertionError(
            f"lire_seuil('abc') renvoie {obtenu!r} au lieu de lever une ValueError. Une saisie invalide ne doit "
            "pas devenir la valeur par défaut en silence : la personne croira son seuil pris en compte."
        )
    raise AssertionError(f"lire_seuil({texte!r}) renvoie {obtenu!r} : attendu une ValueError ({pourquoi}).")


def _except_trop_large(f) -> str | None:
    fonction = _source(f)
    if fonction is None:
        return None
    for noeud in ast.walk(fonction):
        if not isinstance(noeud, ast.ExceptHandler):
            continue
        if noeud.type is None:
            return "`except:` tout seul"
        noms = noeud.type.elts if isinstance(noeud.type, ast.Tuple) else [noeud.type]
        for nom in noms:
            if isinstance(nom, ast.Name) and nom.id in {"Exception", "BaseException"}:
                return f"`except {nom.id}`"
    return None


def verifier_lire_seuil(f) -> None:
    cas = [
        (("0.4",), 0.4), ((" 0,35 ",), 0.35), (("40%",), 0.4), (("40 %",), 0.4), (("100%",), 1.0),
        (("0",), 0.0), (("1",), 1.0), ((None,), 0.5), (("   ",), 0.5), (("", 0.3), 0.3), ((None, 0.7), 0.7),
    ]
    for args, attendu in cas:
        appel = f"lire_seuil({', '.join(map(repr, args))})"
        try:
            obtenu = f(*args)
        except NotImplementedError:
            raise
        except ValueError as exc:
            indice = ""
            if args[0] and "," in args[0]:
                indice = " En français, on écrit 0,35 : remplacez la virgule par un point avant float()."
            elif args[0] and "%" in args[0]:
                indice = " Retirez le signe % (et les espaces) avant float(), puis divisez par 100."
            elif args[0] is None or not args[0].strip():
                indice = " Rien saisi (ou seulement des espaces) → renvoyez `defaut`, sans erreur."
            elif args[0] in ("0", "1"):
                indice = " Les bornes 0 et 1 sont des seuils valides (bornes comprises)."
            raise AssertionError(f"{appel} lève ValueError({exc}) au lieu de renvoyer {attendu}.{indice}") from None
        except (AttributeError, TypeError) as exc:
            raise AssertionError(
                f"{appel} plante ({type(exc).__name__}: {exc}). Traitez None et le texte vide AVANT d'appeler "
                "des méthodes comme .strip() : `if texte is None or not texte.strip(): return defaut`."
            ) from None
        _pas_none(obtenu, appel)
        assert not isinstance(obtenu, str), f"{appel} renvoie du texte ({obtenu!r}) : convertissez avec float()."
        assert isinstance(obtenu, (int, float)), f"{appel} : attendu un float, obtenu {type(obtenu).__name__}"
        assert math.isclose(obtenu, attendu, abs_tol=1e-12), f"{appel} renvoie {obtenu!r} au lieu de {attendu}"

    _seuil_refuse(f, "abc", "ce n'est pas un nombre")
    _seuil_refuse(f, "1.5", "un seuil est entre 0 et 1")
    _seuil_refuse(f, "-0.1", "un seuil est entre 0 et 1")
    _seuil_refuse(f, "150 %", "150 % dépasse 100 %")
    _seuil_refuse(f, "nan", "nan n'est pas un seuil")

    trop_large = _except_trop_large(f)
    if trop_large:
        raise AssertionError(
            f"Tout est juste, sauf un réflexe à prendre : {trop_large} attrape TOUTES les erreurs, y compris vos "
            "propres bugs (une faute de frappe dans un nom de variable devient « saisie invalide »). Nommez "
            "l'exception attendue : `except ValueError:`."
        )


# ============================================================================ Défi 41 : refactoriser

_COLONNES_ANCIENNES = ("monthly_charges", "tenure_months", "support_calls")


def _ligne_reference(partis: pd.Series, restes: pd.Series, variable: str) -> dict:
    m1, m0 = partis.mean(), restes.mean()
    return {"variable": variable, "moyenne_partis": round(m1, 2), "moyenne_restes": round(m0, 2),
            "ecart": round(m1 - m0, 2)}


def _reference(df: pd.DataFrame, colonnes=_COLONNES_ANCIENNES) -> pd.DataFrame:
    """Le comportement de `resume_kpis_ancien`, étendu à n'importe quelles colonnes."""
    partis, restes = df[df["churn"] == 1], df[df["churn"] == 0]
    return pd.DataFrame([_ligne_reference(partis[c], restes[c], c) for c in colonnes])


def _donnees_reelles() -> pd.DataFrame:
    racine = Path(__file__).resolve().parents[1]
    if str(racine / "src") not in sys.path:
        sys.path.insert(0, str(racine / "src"))
    from churn.data import generate_customers

    return generate_customers(n=1500, seed=6)


def _donnees_piegees() -> pd.DataFrame:
    """Moyennes choisies pour que round(m1, 2) - round(m0, 2) ≠ round(m1 - m0, 2)."""
    return pd.DataFrame({
        "churn": [1, 1, 0, 0],
        "monthly_charges": [1.004, 1.004, 0.996, 0.996],
        "tenure_months": [3, 4, 30, 31],
        "support_calls": [5, 4, 0, 1],
        "num_products": [1, 1, 3, 2],
    })


def _comparer_resume(res, attendu: pd.DataFrame, cas: str) -> None:
    _pas_none(res, "resume_kpis")
    assert isinstance(res, pd.DataFrame), f"{cas} : attendu un DataFrame, obtenu {type(res).__name__}"
    assert list(res.columns) == list(attendu.columns), (
        f"{cas} : colonnes {list(res.columns)} au lieu de {list(attendu.columns)} (mêmes noms, même ordre que l'ancien code)"
    )
    assert len(res) == len(attendu), f"{cas} : {len(res)} lignes au lieu de {len(attendu)} (une par variable)"
    assert res.index.equals(attendu.index), (
        f"{cas} : l'index vaut {list(res.index)} au lieu de {list(attendu.index)}. L'ancien code renvoie un index "
        "0, 1, 2… : ne changez pas l'index (pas de set_index), les utilisateurs de la fonction en dépendent."
    )
    assert list(res["variable"]) == list(attendu["variable"]), (
        f"{cas} : variables {list(res['variable'])} au lieu de {list(attendu['variable'])} (même ordre que `colonnes`)"
    )
    for col in ("moyenne_partis", "moyenne_restes", "ecart"):
        obtenu = pd.to_numeric(res[col], errors="coerce").to_numpy(dtype=float)
        voulu = attendu[col].to_numpy(dtype=float)
        nan_ok = np.array_equal(np.isnan(obtenu), np.isnan(voulu))
        if not nan_ok:
            raise AssertionError(
                f"{cas} : « {col} » vaut {list(res[col])} au lieu de {list(attendu[col])}. Quand un groupe est vide, "
                "l'ancien code donne NaN (la moyenne de rien) : un refactoring garde ce comportement."
            )
        proches = np.isclose(obtenu, voulu, rtol=0, atol=1e-9, equal_nan=True)
        if not proches.all():
            i = int(np.argmin(proches))
            indice = ""
            if col == "ecart":
                indice = (" Vous calculez sans doute l'écart avec les moyennes ARRONDIES : l'ancien code arrondit "
                          "m1 - m0, pas round(m1, 2) - round(m0, 2). Même un centième d'écart casse un rapport qui compare "
                          "les chiffres d'un mois sur l'autre.")
            raise AssertionError(
                f"{cas} : « {col} » de {res['variable'].iloc[i]} vaut {obtenu[i]!r} au lieu de {voulu[i]!r}.{indice}"
            )
        non_arrondis = [v for v in obtenu if not math.isnan(v) and round(v, 2) != v]
        if non_arrondis:
            raise AssertionError(
                f"{cas} : « {col} » contient {non_arrondis[0]!r}, qui n'est pas arrondi au centième comme dans l'ancien "
                "code. (Soustraire deux nombres arrondis redonne souvent des décimales parasites : arrondissez en DERNIER.)"
            )


def verifier_resume_kpis(f) -> None:
    reelles = _donnees_reelles()
    avant = reelles.copy()
    _comparer_resume(f(reelles), _reference(reelles), "Sur les données du projet")
    pd.testing.assert_frame_equal(reelles, avant, obj="DataFrame d'entrée (il ne doit pas être modifié)")

    piegees = _donnees_piegees()
    _comparer_resume(f(piegees), _reference(piegees), "Sur un petit tableau de 4 clients")

    personne_n_est_parti = reelles[reelles["churn"] == 0].head(50)
    try:
        res = f(personne_n_est_parti)
    except (KeyError, IndexError, TypeError) as exc:
        raise AssertionError(
            f"Quand AUCUN client n'est parti, votre version plante ({type(exc).__name__}: {exc}) alors que l'ancien "
            "code renvoie des NaN. Avec groupby(\"churn\"), le groupe 1 n'existe tout simplement pas : un refactoring "
            "doit garder le comportement de l'ancien code dans les cas limites aussi."
        ) from None
    _comparer_resume(res, _reference(personne_n_est_parti), "Quand aucun client n'est parti")

    autres = ["num_products", "total_charges"]
    try:
        res = f(reelles, autres)
    except TypeError as exc:
        raise AssertionError(f"resume_kpis(df, {autres}) plante ({exc}) : la fonction doit accepter `colonnes`.") from None
    _comparer_resume(res, _reference(reelles, autres), f"Avec colonnes={autres}")
    _comparer_resume(f(reelles, tuple(autres[::-1])), _reference(reelles, autres[::-1]), "Avec un tuple de colonnes")

    fonction = _source(f)
    if fonction is None:
        return
    corps = fonction.body
    if corps and isinstance(corps[0], ast.Expr) and isinstance(getattr(corps[0], "value", None), ast.Constant):
        corps = corps[1:]  # la docstring a le droit de citer des colonnes
    for noeud in (n for instruction in corps for n in ast.walk(instruction)):
        if isinstance(noeud, ast.Constant) and noeud.value in _COLONNES_ANCIENNES:
            raise AssertionError(
                f"Le résultat est juste, mais le nom de colonne {noeud.value!r} est écrit en dur dans le corps de la "
                "fonction. Parcourez `colonnes` avec une boucle : le jour où l'on ajoute une variable, rien à recopier."
            )
        if isinstance(noeud, ast.Name) and noeud.id == "resume_kpis_ancien":
            raise AssertionError("resume_kpis appelle resume_kpis_ancien : c'est l'ancien code qu'il faut remplacer.")


# ============================================================================ Défi 42 : argparse


def _analyser(parser: argparse.ArgumentParser, argv: list[str]):
    """(namespace, None) si la ligne est acceptée, (None, message d'erreur d'argparse) sinon."""
    erreurs = io.StringIO()
    try:
        with redirect_stderr(erreurs):
            return parser.parse_args(argv), None
    except SystemExit:  # argparse quitte le programme quand la ligne de commande est invalide
        return None, erreurs.getvalue().strip().splitlines()[-1] if erreurs.getvalue().strip() else "erreur"
    except NotImplementedError:
        raise AssertionError(
            "Votre option --seuil utilise lire_seuil, qui n'est pas encore codée : faites d'abord le défi 40 "
            "(python scripts/defis.py --defi 40), puis revenez ici."
        ) from None


def _accepte(parser, argv: list[str]) -> argparse.Namespace:
    args, erreur = _analyser(parser, argv)
    if args is None:
        raise AssertionError(f"`scorer.py {' '.join(argv)}` est refusé alors qu'il est valide. argparse dit : {erreur}")
    return args


def _refuse(parser, argv: list[str], pourquoi: str) -> None:
    args, _ = _analyser(parser, argv)
    if args is not None:
        raise AssertionError(f"`scorer.py {' '.join(argv)}` est accepté ({vars(args)}) au lieu d'être refusé : {pourquoi}.")


def verifier_creer_parser(f) -> None:
    parser = _pas_none(f(), "creer_parser")
    assert isinstance(parser, argparse.ArgumentParser), (
        f"Attendu un argparse.ArgumentParser, obtenu {type(parser).__name__}. Renvoyez le parser lui-même, "
        "sans appeler parse_args()."
    )

    args, erreur = _analyser(parser, ["clients.csv"])
    if args is None:
        raise AssertionError(
            f"`scorer.py clients.csv` est refusé ({erreur}). Les options (--seuil, --modele, --top) doivent avoir "
            "une valeur par défaut, et `fichier` doit être positionnel : add_argument(\"fichier\"), sans tirets."
        )
    attendus = {"fichier": "clients.csv", "seuil": 0.5, "modele": "logistic", "top": 10, "verbose": False}
    manquants = [cle for cle in attendus if not hasattr(args, cle)]
    assert not manquants, (
        f"Options introuvables : {manquants}. Le nom de l'attribut vient du nom long de l'option "
        "(--verbose → args.verbose)."
    )
    for cle, valeur in attendus.items():
        assert getattr(args, cle) == valeur, f"Par défaut, args.{cle} vaut {getattr(args, cle)!r} au lieu de {valeur!r}"

    complet = ["clients.csv", "--seuil", "0.3", "--modele", "gboost", "--top", "5", "-v"]
    args = _accepte(parser, complet)
    assert not isinstance(args.seuil, str), (
        f"args.seuil vaut le TEXTE {args.seuil!r}. argparse lit tout comme du texte : ajoutez type=float "
        "(ou type=lire_seuil)."
    )
    assert isinstance(args.seuil, float) and math.isclose(args.seuil, 0.3), f"--seuil 0.3 donne {args.seuil!r}"
    assert not isinstance(args.top, str), f"args.top vaut le TEXTE {args.top!r} : ajoutez type=int."
    assert args.top == 5 and isinstance(args.top, int), f"--top 5 donne {args.top!r}"
    assert args.modele == "gboost", f"--modele gboost donne {args.modele!r}"
    assert args.verbose is True, (
        f"-v donne verbose={args.verbose!r}. Un interrupteur s'écrit add_argument(\"-v\", \"--verbose\", "
        "action=\"store_true\")."
    )
    assert _accepte(parser, ["clients.csv", "--verbose"]).verbose is True, "--verbose (forme longue) doit marcher aussi"

    _refuse(parser, [], "le fichier est obligatoire")
    _refuse(parser, ["clients.csv", "--modele", "random_forest"], "seuls logistic et gboost existent (choices=[…])")
    _refuse(parser, ["clients.csv", "--top", "2.5"], "--top est un nombre ENTIER (type=int)")
    _refuse(parser, ["clients.csv", "--seuil", "abc"], "« abc » n'est pas un nombre")
    for hors_bornes in ("1.5", "-0.2"):
        _refuse(
            parser, ["clients.csv", "--seuil", hors_bornes],
            "un seuil est entre 0 et 1. type=float ne vérifie pas les bornes : donnez à type= une fonction qui "
            "lève une ValueError hors de [0, 1] (votre lire_seuil du défi 40, par exemple)",
        )

    sans_aide = [a.option_strings[-1] if a.option_strings else a.dest for a in parser._actions
                 if not isinstance(a, argparse._HelpAction) and not (a.help or "").strip()]
    assert not sans_aide, (
        f"Sans aide : {sans_aide}. Ajoutez help=\"…\" à chaque add_argument : c'est la documentation que lira "
        "la personne qui tape `python scorer.py --help` (souvent vous, dans six mois)."
    )


# ============================================================================ Défi 43 : bissection


def _identifiants(n: int) -> list[str]:
    """Des identifiants façon git (dans le désordre alphabétique : impossible de tricher en les triant)."""
    return [hashlib.sha1(f"commit-{i}".encode()).hexdigest()[:7] for i in range(n)]


class _Testeur:
    """Joue le rôle de `est_bon` : compte les appels et repère les boucles infinies."""

    def __init__(self, commits: list[str], premier_mauvais: int):
        self.position = {c: i for i, c in enumerate(commits)}
        self.premier_mauvais = premier_mauvais
        self.appels: list[int] = []
        self.limite = len(commits) + 5

    def __call__(self, commit: str) -> bool:
        if commit not in self.position:
            raise AssertionError(f"est_bon a reçu {commit!r}, qui n'est pas un commit de la liste. Passez commits[i], pas i.")
        self.appels.append(self.position[commit])
        if len(self.appels) > self.limite:
            raise AssertionError(
                f"est_bon a été appelé plus de {self.limite} fois pour {len(self.position)} commits : boucle infinie ? "
                "Vérifiez que `bon` ou `mauvais` change à CHAQUE tour (milieu = (bon + mauvais) // 2)."
            )
        return self.position[commit] < self.premier_mauvais


def _verifier_bissection(f, commits: list[str], premier_mauvais: int, cas: str) -> None:
    n = len(commits)
    maximum = math.ceil(math.log2(n - 1)) if n > 2 else 0
    copie = list(commits)
    testeur = _Testeur(commits, premier_mauvais)
    obtenu = _pas_none(f(copie, testeur), "premier_commit_fautif")
    assert copie == commits, "La liste `commits` a été modifiée : travaillez avec des positions (index)."
    if obtenu != commits[premier_mauvais]:
        if obtenu == commits[premier_mauvais - 1]:
            raise AssertionError(
                f"{cas}, vous renvoyez le DERNIER BON commit ({obtenu!r}) au lieu du PREMIER MAUVAIS "
                f"({commits[premier_mauvais]!r}). Erreur « à un près » : quand la boucle s'arrête, lequel de `bon` "
                "ou `mauvais` désigne la réponse ?"
            )
        position = testeur.position.get(obtenu, "hors de la liste")
        raise AssertionError(
            f"{cas}, le premier commit mauvais est en position {premier_mauvais} ({commits[premier_mauvais]!r}) ; "
            f"vous renvoyez {obtenu!r} (position {position})."
        )
    if 0 in testeur.appels or n - 1 in testeur.appels:
        raise AssertionError(
            "Vous testez le premier ou le dernier commit : on SAIT déjà que le premier est bon et le dernier "
            "mauvais. Chaque test coûte cher (un réentraînement) : ne testez que des commits entre les deux."
        )
    if len(testeur.appels) > maximum:
        pas = {b - a for a, b in zip(testeur.appels, testeur.appels[1:])}
        lineaire = len(testeur.appels) > 2 and pas in ({1}, {-1})
        raise AssertionError(
            f"{cas}, vous appelez est_bon {len(testeur.appels)} fois ; la bissection en demande au plus "
            f"{maximum}." + (" Vous les testez un par un, dans l'ordre : testez plutôt celui du MILIEU, puis "
                             "gardez la moitié qui contient le problème." if lineaire else
                             " Le commit testé doit couper l'intervalle restant en deux moitiés égales.")
        )


def verifier_premier_commit_fautif(f) -> None:
    for trop_court in ([], ["a1b2c3d"]):
        try:
            f(list(trop_court), _Testeur(trop_court, 1))
        except ValueError:
            continue
        except IndexError:
            raise AssertionError(
                f"Avec {len(trop_court)} commit(s), votre fonction plante (IndexError). Vérifiez la longueur au début : "
                "if len(commits) < 2: raise ValueError(…)."
            ) from None
        raise AssertionError(f"Avec {len(trop_court)} commit(s), attendu une ValueError (il faut un bon ET un mauvais).")

    _verifier_bissection(f, ["a1", "b2", "c3", "d4", "e5"], 2, "Sur l'exemple de la docstring")
    for n in range(2, 70):
        commits = _identifiants(n)
        for premier_mauvais in range(1, n):
            _verifier_bissection(f, commits, premier_mauvais, f"Avec {n} commits")


SERIE = Serie(
    code="06",
    titre="Développer comme un·e pro : lire une erreur, tester, structurer",
    module_eleve="defis.serie_06",
    module_corrige="defis.corrige_serie_06",
    defis=(
        Defi(38, "analyser_traceback", "🟢", "—", verifier_analyser_traceback),
        Defi(39, "tester_taux_en_pourcentage", "🟢", "6.1", verifier_tester_taux_en_pourcentage),
        Defi(40, "lire_seuil", "🟡", "—", verifier_lire_seuil),
        Defi(41, "resume_kpis", "🟡", "—", verifier_resume_kpis),
        Defi(42, "creer_parser", "🟡", "—", verifier_creer_parser),
        Defi(43, "premier_commit_fautif", "🔴", "—", verifier_premier_commit_fautif),
    ),
)

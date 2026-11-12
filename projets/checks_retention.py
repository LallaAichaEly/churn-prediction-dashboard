"""Vérifications automatiques du projet 2 (plan de rétention).

Les fichiers de test sont fabriqués à la volée, avec les pièges des vraies données : export
Excel (BOM), ligne coupée, `nan`, espaces, doublons. Les messages visent les erreurs les plus
fréquentes, pas seulement « faux ».

Lire ce fichier est permis (les cas testés sont des indices), mais essayez d'abord seul·e.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib
import inspect
import io
import math
import os
import random
import re
import tempfile
import textwrap
from pathlib import Path
from types import ModuleType

from .retention_corrige import allocation as ref_allocation
from .retention_corrige import clients as ref_clients
from .retention_corrige import regles as ref_regles
from .retention_corrige import rapport as ref_rapport
from .verif import Etape, Projet


def _modeles(module: ModuleType) -> ModuleType:
    """Le `modeles.py` du MÊME paquet que le module testé (le vôtre ou celui du corrigé)."""
    return importlib.import_module(f"{module.__package__}.modeles")


def _pas_none(resultat, quoi: str):
    if resultat is None:
        raise AssertionError(f"{quoi} a renvoyé None. Avez-vous oublié `return` ?")
    return resultat


def _proche(a, b, tol: float = 1e-6) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def _appelle(fonction, nom: str) -> bool | None:
    """La fonction appelle-t-elle `nom(...)` ? None si son code source est introuvable."""
    try:
        arbre = ast.parse(textwrap.dedent(inspect.getsource(fonction)))
    except (OSError, TypeError, SyntaxError):
        return None
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == nom
               for n in ast.walk(arbre))


@contextlib.contextmanager
def _dossier_temporaire():
    with tempfile.TemporaryDirectory(prefix="projet_retention_") as dossier:
        yield Path(dossier)


# ============================================================================ Étape 1 : lire_clients

_EN_TETE_EXPORT = ("customer_id,tenure_months,monthly_charges,total_charges,support_calls,num_products,"
                   "contract,payment_method,internet_service,tech_support,senior,churn,proba_churn")
_EXPORT = _EN_TETE_EXPORT + """
C00001,5,95.5,477.5,4,1,Mensuel,Chèque,Fibre,Non,Non,1,0.912
C00002,40,55.0,2200.0,0,2,Annuel,Prélèvement,ADSL,Oui,Non,0,0.081
C00003,2,70.25,140.5,1,1,Mensuel,Carte bancaire,Fibre,Non,Oui,0,0.64
C00004,60,25.0,1500.0,3,3,Biannuel,Virement,Aucun,Oui,Non,0,0.05
 C00005 , 12 , 80 ,960.0, 2 ,1, Mensuel ,Chèque,Fibre,Non,Non,1, 0.5
"""
_EXPORT_ATTENDU = [
    ("C00001", 0.912, 95.5, 5, "Mensuel", 4),
    ("C00002", 0.081, 55.0, 40, "Annuel", 0),
    ("C00003", 0.64, 70.25, 2, "Mensuel", 1),
    ("C00004", 0.05, 25.0, 60, "Biannuel", 3),
    ("C00005", 0.5, 80.0, 12, "Mensuel", 2),
]

# Chaque ligne fausse est numérotée comme dans un éditeur de texte (l'en-tête est la ligne 1).
_PIEGES = """customer_id,proba_churn,monthly_charges,tenure_months,contract,support_calls
C10,0.42,60,10,Mensuel,1
C11,,60,10,Mensuel,1
C12,1.4,60,10,Annuel,0
C13,0.3,60,10,Mensual,0
C14,nan,60,10,Mensuel,0
C15,0.3,60,douze,Mensuel,0
,0.3,60,10,Mensuel,0
C16,0.3
C10,0.9,60,10,Mensuel,1
C17,0.3,-5,10,Annuel,0
C19,0.55,45.5,3,Biannuel,0
C20,0.2,30,8,Annuel,-1
C21,0,30,0,Annuel,0
"""
_PIEGES_CLIENTS = ["C10", "C19", "C21"]
#: ligne → mot que le message doit contenir (None : n'importe quelle colonne fautive)
_PIEGES_ERREURS = {3: "proba_churn", 4: "proba_churn", 5: "contract", 6: "proba_churn", 7: "tenure_months",
                   8: "customer_id", 9: None, 10: "double", 11: "monthly_charges", 13: "support_calls"}


def _ecrire(dossier: Path, nom: str, texte: str, encodage: str = "utf-8") -> Path:
    chemin = dossier / nom
    chemin.write_text(texte, encoding=encodage, newline="")
    return chemin


def _lire(f, chemin, cas: str):
    try:
        res = f(chemin)
    except NotImplementedError:
        raise
    except (TypeError, AttributeError) as exc:
        if "NoneType" in str(exc):
            raise AssertionError(
                f"{cas} : TypeError ({exc}). Une ligne coupée (« C16,0.3 ») donne None, et non \"\", pour "
                "les colonnes absentes du dictionnaire de DictReader. Traitez None comme une case vide : "
                "`(ligne[colonne] or \"\").strip()`."
            ) from None
        raise
    res = _pas_none(res, "lire_clients")
    if not (isinstance(res, tuple) and len(res) == 2):
        raise AssertionError(f"{cas} : renvoyez un tuple (clients, erreurs), obtenu {type(res).__name__}.")
    clients, erreurs = res
    if not isinstance(clients, list) or not isinstance(erreurs, list):
        raise AssertionError(f"{cas} : clients et erreurs doivent être des listes.")
    return clients, erreurs


def _valeurs(client) -> tuple:
    return (client.customer_id, client.proba_churn, client.monthly_charges, client.tenure_months,
            client.contract, client.support_calls)


def _verifier_export(f, chemin, cas: str) -> None:
    try:
        clients, erreurs = _lire(f, chemin, cas)
    except ValueError as exc:
        if "\ufeff" in str(exc) or "customer_id" in str(exc):
            raise AssertionError(
                f"{cas} : ValueError ({exc}). Ce fichier commence par le BOM qu'ajoute Excel : un caractère "
                "invisible qui colle à la première colonne (\"\\ufeffcustomer_id\"). Ouvrez le fichier avec "
                "`encoding=\"utf-8-sig\"` : il lit les fichiers avec ET sans BOM."
            ) from None
        raise AssertionError(f"{cas} : ValueError inattendue sur un fichier valide : {exc}") from None
    except KeyError as exc:
        bom = " Ce fichier commence par le BOM d'Excel : ouvrez-le avec `encoding=\"utf-8-sig\"`." if "Excel" in cas else ""
        raise AssertionError(f"{cas} : KeyError {exc}. La colonne n'est pas trouvée sous ce nom.{bom}") from None
    if erreurs:
        raise AssertionError(f"{cas} : le fichier est valide, mais vous signalez des erreurs : {erreurs[:3]}. "
                             "Pensez à retirer les espaces autour des textes (`.strip()`).")
    if len(clients) != len(_EXPORT_ATTENDU):
        raise AssertionError(f"{cas} : {len(clients)} clients lus au lieu de {len(_EXPORT_ATTENDU)}.")
    for client, attendu in zip(clients, _EXPORT_ATTENDU):
        if not dataclasses.is_dataclass(client) or not hasattr(client, "proba_churn"):
            raise AssertionError(f"{cas} : chaque client doit être un objet `Client` (modeles.py), obtenu "
                                 f"{type(client).__name__}.")
        valeurs = _valeurs(client)
        if valeurs != attendu:
            raise AssertionError(f"{cas} : client lu {valeurs}, attendu {attendu}. Les colonnes ne sont pas dans "
                                 "le même ordre que dans Client : lisez-les par leur NOM (DictReader).")
        for champ, type_ in (("tenure_months", int), ("support_calls", int), ("proba_churn", float),
                             ("monthly_charges", float)):
            valeur = getattr(client, champ)
            if type(valeur) is not type_:
                raise AssertionError(f"{cas} : {champ} vaut {valeur!r} ({type(valeur).__name__}), il faut un "
                                     f"{type_.__name__}. Le CSV ne contient que du texte : convertissez.")


def _verifier_pieges(f, chemin) -> None:
    cas = "Fichier avec des lignes fausses"
    try:
        clients, erreurs = _lire(f, chemin, cas)
    except AssertionError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise AssertionError(
            f"{cas} : lire_clients s'arrête ({type(exc).__name__}: {exc}). Une ligne invalide doit être "
            "IGNORÉE et SIGNALÉE dans `erreurs`, pas faire planter la lecture : entourez la lecture d'une ligne "
            "d'un `try` / `except ValueError` (et transformez vos autres problèmes en ValueError)."
        ) from None

    ids = [c.customer_id for c in clients]
    if "C14" in ids:
        raise AssertionError(
            f"{cas} : le client C14 (proba_churn « nan ») a été accepté. float(\"nan\") ne lève pas d'erreur, et "
            "toute comparaison avec nan est fausse : `p < 0 or p > 1` laisse donc passer nan. Écrivez "
            "`if not 0 <= p <= 1:` (ou testez math.isnan)."
        )
    if ids.count("C10") > 1:
        raise AssertionError(f"{cas} : C10 apparaît deux fois (lignes 2 et 10). Gardez la première et signalez "
                             "la seconde (un `set` des identifiants déjà vus).")
    if "C21" not in ids and "C10" in ids:
        raise AssertionError(f"{cas} : C21 (proba_churn 0, tenure_months 0) a été refusé, or 0 est une valeur "
                             "valide : les bornes sont incluses. Attention aussi à `if not valeur:` qui rejette 0.")
    if ids != _PIEGES_CLIENTS:
        raise AssertionError(f"{cas} : clients retenus {ids}, attendu {_PIEGES_CLIENTS} (dans l'ordre du fichier).")

    numeros = {}
    for message in erreurs:
        trouve = re.match(r"\s*ligne\s+(\d+)\s*:", str(message), flags=re.IGNORECASE)
        if not trouve:
            raise AssertionError(f"{cas} : chaque erreur doit commencer par « ligne N : », obtenu {message!r}.")
        numeros[int(trouve.group(1))] = str(message)
    attendus = set(_PIEGES_ERREURS)
    if set(numeros) != attendus:
        decale = {n + 1 for n in numeros} == attendus or {n + 2 for n in numeros} == attendus
        conseil = (" Vos numéros sont décalés : l'en-tête est la ligne 1, donc le premier client est la ligne 2 "
                   "(`lecteur.line_num`, ou `enumerate(lecteur, start=2)`)." if decale else "")
        raise AssertionError(f"{cas} : erreurs signalées aux lignes {sorted(numeros)}, attendu "
                             f"{sorted(attendus)}.{conseil}")
    for numero, mot in _PIEGES_ERREURS.items():
        if mot and mot not in numeros[numero]:
            raise AssertionError(f"{cas} : le message de la ligne {numero} doit nommer « {mot} », obtenu "
                                 f"{numeros[numero]!r}. Un bon message dit OÙ et QUOI corriger.")


def verifier_lire_clients(module: ModuleType) -> None:
    f = module.lire_clients
    with _dossier_temporaire() as dossier:
        _verifier_export(f, str(_ecrire(dossier, "export.csv", _EXPORT)), "Export du tableau de bord")
        _verifier_export(f, _ecrire(dossier, "export_excel.csv", _EXPORT, "utf-8-sig"),
                         "Le même export enregistré par Excel")
        _verifier_pieges(f, _ecrire(dossier, "pieges.csv", _PIEGES))

        incomplet = _ecrire(dossier, "incomplet.csv", "customer_id,monthly_charges,tenure_months,contract\n"
                                                      "C1,50,3,Mensuel\n")
        try:
            f(incomplet)
        except ValueError as exc:
            if "proba_churn" not in str(exc) or "support_calls" not in str(exc):
                raise AssertionError(f"Colonnes manquantes : le message « {exc} » doit NOMMER les colonnes absentes "
                                     "(ici proba_churn et support_calls).") from None
        except KeyError as exc:
            raise AssertionError(f"Colonnes manquantes : KeyError {exc}. Comparez `lecteur.fieldnames` à "
                                 "COLONNES_REQUISES AVANT de lire les lignes, et levez ValueError.") from None
        else:
            raise AssertionError("Colonnes manquantes : il manque proba_churn et support_calls, le fichier est "
                                 "inutilisable. Levez `ValueError` en nommant les colonnes manquantes.")

        vide = _ecrire(dossier, "vide.csv", "")
        try:
            f(vide)
        except ValueError:
            pass
        except TypeError as exc:
            raise AssertionError(f"Fichier vide : TypeError ({exc}). Pour un fichier vide, `lecteur.fieldnames` vaut "
                                 "None : traitez-le comme « toutes les colonnes manquent » (ValueError).") from None
        else:
            raise AssertionError("Fichier vide : toutes les colonnes manquent, levez ValueError.")


# ============================================================================ Étape 2 : valeur


def verifier_valeur(module: ModuleType) -> None:
    m = _modeles(module)
    perte, gain = module.perte_attendue, module.gain_attendu
    client = m.Client("X", 0.5, 80.0, 10, "Mensuel", 0)
    remise = m.Offre("REMISE", "Remise", cout=45.0, efficacite=0.30)
    appel = m.Offre("APPEL", "Appel", cout=8.0, efficacite=0.10)

    res = _pas_none(perte(client), "perte_attendue")
    if not _proche(res, 480):
        raise AssertionError(f"perte_attendue (proba 0,5, 80 €/mois, 12 mois) = {res}, attendu 0,5 × 80 × 12 = 480.")
    res = perte(client, horizon=6)
    if _proche(res, 480):
        raise AssertionError("perte_attendue(client, horizon=6) renvoie encore 480 : vous avez écrit 12 en dur. "
                             "Utilisez le paramètre `horizon`.")
    if not _proche(res, 240):
        raise AssertionError(f"perte_attendue(client, horizon=6) = {res}, attendu 240.")
    precis = m.Client("Y", 0.337, 71.13, 10, "Annuel", 0)
    res, attendu = perte(precis), 0.337 * 71.13 * 12
    if not _proche(res, attendu, 1e-9):
        arrondi = _proche(res, attendu, 1e-2)
        raise AssertionError(f"perte_attendue = {res}, attendu {attendu}."
                             + (" N'arrondissez pas ici : on arrondit seulement à l'affichage." if arrondi else ""))

    res = _pas_none(gain(client, remise), "gain_attendu")
    if _proche(res, 0.30 * (480 - 45)) or _proche(res, 0.70 * 480 - 45):
        raise AssertionError(f"gain_attendu = {res}. Relisez la formule : efficacité × perte attendue, PUIS moins "
                             "le coût (0,30 × 480 − 45 = 99).")
    if not _proche(res, 99):
        raise AssertionError(f"gain_attendu (perte 480 €, efficacité 0,30, coût 45 €) = {res}, attendu 99.")
    res = gain(client, remise, horizon=6)
    if not _proche(res, 0.30 * 240 - 45):
        raise AssertionError(f"gain_attendu(client, offre, horizon=6) = {res}, attendu 0,30 × 240 − 45 = 27. "
                             "Transmettez `horizon` à perte_attendue.")
    petit = m.Client("Z", 0.05, 20.0, 10, "Annuel", 0)
    res = gain(petit, appel)
    if not _proche(res, 0.10 * 12 - 8):
        bloque = _proche(res, 0)
        raise AssertionError(f"gain_attendu (perte 12 €, offre à 8 € d'efficacité 0,10) = {res}, attendu −6,8."
                             + (" Un gain peut être NÉGATIF : ne le ramenez pas à 0, c'est l'étape 3 qui écartera "
                                "les offres qui font perdre de l'argent." if bloque else ""))
    if _appelle(gain, "perte_attendue") is False:
        raise AssertionError("Les résultats sont justes, mais gain_attendu recopie la formule de la perte au lieu "
                             "d'appeler perte_attendue(client, horizon). Une formule écrite deux fois finit "
                             "toujours par diverger le jour où on en modifie une seule.")


# ============================================================================ Étape 3 : regles


def _codes(offres) -> list:
    return [getattr(o, "code", o) for o in offres]


def verifier_regles(module: ModuleType) -> None:
    m = _modeles(module)
    eligibles, meilleure = module.offres_eligibles, module.meilleure_offre

    cas_eligibles = [
        (m.Client("E1", 0.5, 50.0, 2, "Annuel", 3), ["APPEL", "SUPPORT"],
         "Annuel, 2 mois, 3 appels"),
        (m.Client("E2", 0.5, 50.0, 3, "Mensuel", 0), ["APPEL", "REMISE", "ENGAGEMENT"],
         "Mensuel, 3 mois, 0 appel"),
        (m.Client("E3", 0.5, 50.0, 30, "Biannuel", 5), ["APPEL", "SUPPORT", "REMISE"],
         "Biannuel, 30 mois, 5 appels"),
    ]
    for client, attendu, decrit in cas_eligibles:
        res = _pas_none(eligibles(client), "offres_eligibles")
        if not isinstance(res, list):
            raise AssertionError(f"offres_eligibles doit renvoyer une liste, obtenu {type(res).__name__}.")
        if res and isinstance(res[0], str):
            raise AssertionError("offres_eligibles doit renvoyer les objets Offre eux-mêmes, pas leurs codes.")
        codes = _codes(res)
        if codes == attendu:
            continue
        conseil = ""
        if "APPEL" not in codes:
            conseil = (" APPEL n'a pas de restriction de contrat (contrats = ()) : un tuple vide veut dire "
                       "« tous les contrats ». Écrivez `not offre.contrats or client.contract in offre.contrats`.")
        elif ((decrit.startswith("Annuel") and "SUPPORT" not in codes)
              or (decrit.startswith("Mensuel") and "REMISE" not in codes)):
            conseil = " « Au moins 3 » inclut 3 : utilisez >= et non >."
        elif sorted(codes) == sorted(attendu):
            conseil = " Gardez l'ordre du catalogue."
        raise AssertionError(f"offres_eligibles({decrit}) = {codes}, attendu {attendu}.{conseil}")

    mini = (m.Offre("A", "a", cout=5.0, efficacite=0.1), m.Offre("B", "b", cout=5.0, efficacite=0.1, contrats=("Annuel",)))
    codes = _codes(eligibles(cas_eligibles[1][0], mini))
    if codes != ["A"]:
        raise AssertionError(f"offres_eligibles(client, offres=<2 offres A et B>) = {codes}, attendu ['A'] : "
                             "utilisez le paramètre `offres`, pas toujours le catalogue OFFRES.")

    client = m.Client("M1", 0.6, 80.0, 12, "Mensuel", 0)  # perte attendue 576 €
    action = _pas_none(meilleure(client), "meilleure_offre")
    if not all(hasattr(action, a) for a in ("client", "offre", "gain_attendu")):
        raise AssertionError(f"meilleure_offre doit renvoyer une Action(client, offre, gain), obtenu "
                             f"{type(action).__name__}.")
    if action.client is not client:
        raise AssertionError("L'Action renvoyée doit contenir le client reçu.")
    if action.offre.code != "ENGAGEMENT" or not _proche(action.gain_attendu, 199.2):
        raise AssertionError(f"meilleure_offre(Mensuel, proba 0,6, 80 €/mois) = {action.offre.code} "
                             f"(gain {action.gain_attendu}), attendu ENGAGEMENT (gain 0,45 × 576 − 60 = 199,2). "
                             "Prenez le plus grand GAIN, pas la plus grande efficacité ni le plus petit coût.")
    action = meilleure(m.Client("S1", 0.6, 80.0, 12, "Annuel", 3))
    if action is None or action.offre.code != "SUPPORT":
        raise AssertionError(f"Client Annuel avec 3 appels : attendu SUPPORT (gain 181,6), obtenu "
                             f"{getattr(getattr(action, 'offre', None), 'code', action)}.")

    pauvre = meilleure(m.Client("L1", 0.05, 20.0, 12, "Annuel", 0))
    if pauvre is not None:
        raise AssertionError(f"Client à faible risque (perte attendue 12 €) : toutes les offres font perdre de "
                             f"l'argent, renvoyez None. Obtenu {pauvre.offre.code} (gain {pauvre.gain_attendu:.2f}).")
    court = meilleure(client, horizon=1)
    if court is not None:
        raise AssertionError("meilleure_offre(client, horizon=1) : sur un seul mois, aucune offre n'est rentable "
                             "(attendu None). Transmettez `horizon` à gain_attendu.")

    egal = m.Client("T1", 0.5, 100.0, 12, "Annuel", 0)  # perte attendue 600 €
    cher, eco = m.Offre("CHER", "c", cout=200.0, efficacite=0.5), m.Offre("ECO", "e", cout=50.0, efficacite=0.25)
    res = meilleure(egal, (cher, eco))
    if res is None or res.offre.code != "ECO":
        raise AssertionError("Deux offres rapportent 100 € (CHER coûte 200 €, ECO 50 €) : à gain égal, prenez la "
                             f"MOINS CHÈRE. Obtenu {getattr(getattr(res, 'offre', None), 'code', res)}. Astuce : "
                             "`max(..., key=lambda a: (a.gain_attendu, -a.offre.cout))`.")
    premiere, seconde = m.Offre("PREMIERE", "p", cout=50.0, efficacite=0.25), m.Offre("SECONDE", "s", cout=50.0,
                                                                                      efficacite=0.25)
    res = meilleure(egal, (premiere, seconde))
    if res is None or res.offre.code != "PREMIERE":
        raise AssertionError("Égalité parfaite (même gain, même coût) : gardez la PREMIÈRE du catalogue. Dans une "
                             "boucle, comparez avec > (strictement) pour ne pas remplacer un ex aequo.")
    nul = meilleure(egal, (m.Offre("NUL", "n", cout=75.0, efficacite=0.125),))
    if nul is not None:
        raise AssertionError("Une offre au gain exactement nul (0,125 × 600 − 75 = 0) ne rapporte rien : il faut un "
                             "gain STRICTEMENT positif, renvoyez None.")


# ============================================================================ Étape 4 : allocation


class _Fabrique:
    """Fabrique des actions de test avec les classes de `modeles.py` du paquet testé."""

    def __init__(self, module: ModuleType):
        self.m = _modeles(module)

    def __call__(self, identifiant: str, cout: float, gain: float):
        client = self.m.Client(identifiant, 0.5, 50.0, 12, "Mensuel", 0)
        return self.m.Action(client, self.m.Offre(f"O_{identifiant}", "test", cout, 0.3), gain)


def _ids(actions) -> list:
    return [a.client.customer_id for a in actions]


def _allouer(f, actions, budget, cas: str):
    res = _pas_none(f(actions, budget), "allouer_budget")
    if not isinstance(res, list):
        raise AssertionError(f"{cas} : renvoyez une liste (obtenu {type(res).__name__}).")
    depense = sum(x.offre.cout for x in res)
    if depense > budget + 1e-9:
        raise AssertionError(f"{cas} : vous dépensez {depense:.0f} € ({_ids(res)}). Retirez le coût de chaque action "
                             "retenue de ce qui reste, et ne prenez une action que si son coût tient dans ce reste.")
    return res


def verifier_allocation(module: ModuleType) -> None:
    f, action = module.allouer_budget, _Fabrique(module)
    a, b, c = action("A", 10.0, 50.0), action("B", 40.0, 100.0), action("C", 30.0, 90.0)
    perte, nul = action("D", 5.0, -1.0), action("E", 5.0, 0.0)

    try:
        res = f([a], -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Budget négatif : levez ValueError (obtenu {res!r}). Un budget de −1 € est une erreur de "
                             "saisie : la signaler vaut mieux que renvoyer un plan vide en silence.")

    if _ids(_allouer(f, [a], 10.0, "Budget 10 €")) != ["A"]:
        raise AssertionError("Budget 10 € et une action à 10 € : elle rentre pile. Comparez avec <= et non <.")

    actions = [b, perte, c, a]
    copie = list(actions)
    res = _allouer(f, actions, 40.0, "Budget 40 €")
    if actions != copie:
        raise AssertionError("allouer_budget a MODIFIÉ la liste reçue (triée ou vidée). `actions.sort()` trie la liste "
                             "de l'appelant : utilisez `sorted(actions, ...)`, qui en renvoie une nouvelle.")
    if _ids(res) == ["B"]:
        raise AssertionError("Budget 40 € : vous avez pris B seul (gain 100 €) : vous triez par GAIN. Triez par "
                             "RENTABILITÉ (action.rentabilite) : A puis C rapportent 140 €.")
    if sorted(_ids(res)) == ["A", "C"] and _ids(res) != ["A", "C"]:
        raise AssertionError(f"Budget 40 € : les bonnes actions, mais dans le désordre ({_ids(res)}). Renvoyez-les "
                             "dans l'ordre où vous les prenez : rentabilité décroissante.")
    if _ids(res) != ["A", "C"]:
        raise AssertionError(f"Budget 40 € avec A (10 €, rentab. 5), B (40 €, rentab. 2,5), C (30 €, rentab. 3) et D "
                             f"(gain négatif) : obtenu {_ids(res)}, attendu ['A', 'C'].")

    gros, petit = action("X", 50.0, 200.0), action("Y", 10.0, 20.0)
    res = _allouer(f, [gros, petit], 30.0, "Budget 30 €")
    if _ids(res) == []:
        raise AssertionError("Budget 30 € : X (50 €) ne rentre pas, mais Y (10 €) oui. Vous arrêtez la boucle à la "
                             "première action trop chère (`break`) : passez simplement à la suivante.")
    if _ids(res) != ["Y"]:
        raise AssertionError(f"Budget 30 € avec X (50 €) et Y (10 €) : obtenu {_ids(res)}, attendu ['Y'].")

    if _ids(_allouer(f, [a, b, c], 50.0, "Budget 50 €")) != ["A", "C"]:
        raise AssertionError("Budget 50 € avec A (10 €), C (30 €) puis B (40 €) : attendu ['A', 'C'].")
    res = _allouer(f, [perte, nul, a], 1e6, "Budget illimité")
    if _ids(res) != ["A"]:
        raise AssertionError(f"Budget illimité : obtenu {_ids(res)}, attendu ['A']. D (gain −1 €) et E (gain 0 €) ne "
                             "rapportent rien : écartez les gains qui ne sont pas strictement positifs.")
    if _allouer(f, [a, c], 0.0, "Budget 0 €") != []:
        raise AssertionError("Budget 0 € : aucune action ne rentre, renvoyez [].")
    res = _allouer(f, [action("B2", 10.0, 20.0), action("A9", 10.0, 20.0)], 10.0, "Ex aequo")
    if _ids(res) != ["A9"]:
        raise AssertionError(f"Deux actions de même rentabilité, budget pour une seule : obtenu {_ids(res)}, attendu "
                             "['A9'] (customer_id croissant). Clé de tri : `(-a.rentabilite, a.client.customer_id)`.")

    hasard = random.Random(7)
    lot = [action(f"R{i:03d}", float(hasard.choice([8, 20, 45, 60])), float(hasard.randint(-20, 300)))
           for i in range(80)]
    attendu = ref_allocation.allouer_budget(lot, 700.0)
    res = _allouer(f, list(lot), 700.0, "80 actions, budget 700 €")
    if _ids(res) != _ids(attendu):
        raise AssertionError(f"80 actions au hasard, budget 700 € : {len(res)} actions retenues "
                             f"(gain {sum(x.gain_attendu for x in res):.0f} €), attendu {len(attendu)} "
                             f"(gain {sum(x.gain_attendu for x in attendu):.0f} €). Relisez la règle pas à pas.")


# ============================================================================ Étape 5 : rapport


def _actions_rapport(module: ModuleType) -> list:
    m = _modeles(module)
    appel, engagement = m.OFFRES[0], m.OFFRES[3]
    return [
        m.Action(m.Client("A1", 0.5124, 80.0, 12, "Mensuel", 0), appel, 41.6),
        m.Action(m.Client("A2", 0.6, 80.0, 12, "Mensuel", 0), engagement, 199.2),
        m.Action(m.Client("A3", 0.25, 60.0, 12, "Annuel", 0), appel, 10.0),
    ]


def verifier_rapport(module: ModuleType) -> None:
    actions = _actions_rapport(module)
    r = _pas_none(module.resumer(actions, 10), "resumer")
    if not isinstance(r, dict):
        raise AssertionError(f"resumer doit renvoyer un dictionnaire, obtenu {type(r).__name__}.")
    cles = {"clients_analyses", "clients_cibles", "cout_total", "gain_attendu", "roi", "par_offre"}
    if set(r) != cles:
        raise AssertionError(f"Clés de resumer : il manque {sorted(cles - set(r))}, en trop {sorted(set(r) - cles)}.")
    if r["clients_analyses"] != 10 or r["clients_cibles"] != 3:
        raise AssertionError(f"clients_analyses = {r['clients_analyses']} et clients_cibles = {r['clients_cibles']}, "
                             "attendu 10 (le paramètre nb_clients) et 3 (le nombre d'actions).")
    if not _proche(r["cout_total"], 76.0) or not _proche(r["gain_attendu"], 250.8):
        raise AssertionError(f"cout_total = {r['cout_total']} et gain_attendu = {r['gain_attendu']}, attendu 76 "
                             "(8 + 60 + 8) et 250,8 (41,6 + 199,2 + 10).")
    if _proche(r["roi"], (250.8 - 76) / 76):
        raise AssertionError("roi : le gain attendu est DÉJÀ net du coût des offres, ne le soustrayez pas une "
                             "deuxième fois : roi = gain_attendu / cout_total.")
    if not _proche(r["roi"], 250.8 / 76):
        raise AssertionError(f"roi = {r['roi']}, attendu gain_attendu / cout_total = 250,8 / 76 ≈ 3,3.")
    if dict(r["par_offre"]) != {"APPEL": 2, "ENGAGEMENT": 1}:
        raise AssertionError(f"par_offre = {r['par_offre']}, attendu {{'APPEL': 2, 'ENGAGEMENT': 1}} (les CODES des "
                             "offres comme clés).")
    try:
        vide = module.resumer([], 4)
    except ZeroDivisionError:
        raise AssertionError("resumer([], 4) : ZeroDivisionError. Un plan vide coûte 0 € : renvoyez roi = 0.0 "
                             "dans ce cas.") from None
    if not (vide["clients_cibles"] == 0 and _proche(vide["cout_total"], 0) and _proche(vide["roi"], 0)
            and dict(vide["par_offre"]) == {} and vide["clients_analyses"] == 4):
        raise AssertionError(f"Plan vide : obtenu {vide}.")

    lignes_attendues = [
        "customer_id,offre,cout,proba_churn,gain_attendu",
        "A1,APPEL,8.00,0.512,41.60",
        "A2,ENGAGEMENT,60.00,0.600,199.20",
        "A3,APPEL,8.00,0.250,10.00",
    ]
    with _dossier_temporaire() as dossier:
        chemin = dossier / "resultats" / "2026-11" / "plan.csv"
        try:
            res = module.ecrire_plan(actions, chemin)
        except FileNotFoundError:
            raise AssertionError("ecrire_plan : FileNotFoundError, le dossier resultats/2026-11/ n'existe pas encore. "
                                 "Créez-le : `chemin.parent.mkdir(parents=True, exist_ok=True)`.") from None
        if not isinstance(res, Path) or res != chemin:
            raise AssertionError(f"ecrire_plan doit renvoyer le chemin du fichier sous forme de Path, obtenu {res!r}.")
        brut = chemin.read_bytes()
        if b"\r\r\n" in brut or b"\n\n" in brut.replace(b"\r\n", b"\n"):
            raise AssertionError("Le plan contient des lignes vides : ouvrez le fichier avec `newline=\"\"`.")
        lignes = brut.decode("utf-8").splitlines()
        if lignes != lignes_attendues:
            differente = next((i for i, (x, y) in enumerate(zip(lignes, lignes_attendues)) if x != y), None)
            if differente is None:
                raise AssertionError(f"Le plan contient {len(lignes)} lignes, attendu {len(lignes_attendues)} "
                                     "(l'en-tête puis une ligne par action).")
            raise AssertionError(f"Ligne {differente + 1} du plan : {lignes[differente]!r}, attendu "
                                 f"{lignes_attendues[differente]!r}. Formatez avec f\"{{x:.2f}}\" (coût, gain) et "
                                 "f\"{x:.3f}\" (probabilité), et écrivez le CODE de l'offre.")
        seul = module.ecrire_plan([], str(dossier / "vide.csv"))
        if Path(seul).read_text(encoding="utf-8").splitlines() != lignes_attendues[:1]:
            raise AssertionError("Plan vide : le fichier doit contenir l'en-tête et rien d'autre.")


# ============================================================================ Étape 6 : cli

_CLIENTS_CLI = """customer_id,proba_churn,monthly_charges,tenure_months,contract,support_calls,payment_method
K01,0.91,95.5,5,Mensuel,4,Chèque
K02,0.08,55.0,40,Annuel,0,Prélèvement
K03,0.64,70.25,2,Mensual,1,Carte bancaire
K04,0.72,88.0,14,Mensuel,0,Chèque
K05,0.35,45.0,8,Annuel,3,Virement
K06,0.55,99.9,30,Biannuel,1,Prélèvement
K07,0.81,62.0,1,Mensuel,2,Carte bancaire
K08,0.22,30.0,20,Annuel,0,Prélèvement
K09,0.47,76.0,9,Mensuel,5,Chèque
K10,0.66,84.5,26,Annuel,1,Virement
K11,0.93,110.0,3,Mensuel,6,Chèque
K12,0.15,25.0,50,Biannuel,0,Virement
"""
_LIGNE_FAUSSE_CLI = 4  # K03 : contrat « Mensual »


def _plan_attendu(chemin_clients: Path, budget: float, horizon: int = 12) -> list[str]:
    """Le plan que produit le corrigé : la référence du correcteur."""
    clients, _ = ref_clients.lire_clients(chemin_clients)
    candidates = [a for c in clients if (a := ref_regles.meilleure_offre(c, horizon=horizon)) is not None]
    plan = ref_allocation.allouer_budget(candidates, budget)
    with _dossier_temporaire() as dossier:
        return ref_rapport.ecrire_plan(plan, dossier / "p.csv").read_text(encoding="utf-8").splitlines()


def _lancer(main, argv: list[str]) -> tuple[int, str, str]:
    """Exécute main(argv) comme le ferait un terminal : (code de sortie, stdout, stderr)."""
    sortie, erreurs = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(sortie), contextlib.redirect_stderr(erreurs):
        try:
            code = main(argv)
        except SystemExit as exc:  # argparse quitte avec SystemExit(2) sur une option invalide
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return code, sortie.getvalue(), erreurs.getvalue()


def _commande(argv: list[str]) -> str:
    return "python -m projets.retention " + " ".join(argv)


def verifier_cli(module: ModuleType) -> None:
    main = module.main
    with _dossier_temporaire() as dossier:
        clients = _ecrire(dossier, "clients.csv", _CLIENTS_CLI)
        plan = dossier / "sorties" / "plan.csv"

        argv = ["--clients", str(clients), "--budget", "150", "--sortie", str(plan)]
        code, out, err = _lancer(main, argv)
        if code is None:
            raise AssertionError(f"`{_commande(argv)}` : main renvoie None. Terminez par `return 0` : c'est le code de "
                                 "sortie qui dit au terminal (et aux scripts) que tout s'est bien passé.")
        if code != 0:
            raise AssertionError(f"`{_commande(argv)}` : code de sortie {code}, attendu 0. Message d'erreur : "
                                 f"{err.strip()[-300:] or '(aucun)'}")
        if not plan.exists():
            raise AssertionError(f"`{_commande(argv)}` : aucun fichier écrit à l'emplacement de --sortie.")
        attendu = _plan_attendu(clients, 150.0)
        lignes = plan.read_text(encoding="utf-8").splitlines()
        if lignes != attendu:
            raise AssertionError(f"`{_commande(argv)}` : le plan écrit ({len(lignes) - 1} clients) diffère du plan "
                                 f"attendu ({len(attendu) - 1} clients : {', '.join(x.split(',')[0] for x in attendu[1:])}). "
                                 "Vérifiez d'abord que les étapes 1 à 5 sont ✅, puis l'enchaînement dans main.")
        if str(len(attendu) - 1) not in out or plan.name not in out:
            raise AssertionError(f"Le résumé affiché doit contenir le nombre de clients ciblés ({len(attendu) - 1}) "
                                 f"et le chemin du plan. Affiché : {out.strip()[:300]!r}")
        if f"ligne {_LIGNE_FAUSSE_CLI}" not in err:
            raise AssertionError(f"La ligne {_LIGNE_FAUSSE_CLI} du fichier (contrat « Mensual ») est ignorée : affichez-le "
                                 "sur la sortie d'erreur, `print(erreur, file=sys.stderr)`, pour que l'utilisateur sache "
                                 f"quoi corriger. Sortie d'erreur : {err.strip()[:200]!r}")

        plan6 = dossier / "plan6.csv"
        argv = ["--clients", str(clients), "--budget", "150", "--sortie", str(plan6), "--horizon", "6"]
        code, _, err = _lancer(main, argv)
        if code != 0 or not plan6.exists() or plan6.read_text(encoding="utf-8").splitlines() != _plan_attendu(
                clients, 150.0, horizon=6):
            raise AssertionError(f"`{_commande(argv)}` : le plan ne tient pas compte de --horizon 6 (code {code}). "
                                 "Ajoutez l'option (type=int) et passez-la à meilleure_offre(client, horizon=...).")

        argv = ["--clients", str(clients), "--budget", "1500,5", "--sortie", str(dossier / "virgule.csv")]
        code, _, err = _lancer(main, argv)
        if code != 0:
            raise AssertionError(f"`{_commande(argv)}` : code {code}. Un budget écrit à la française (« 1500,5 ») doit "
                                 f"être accepté : remplacez la virgule par un point avant float(). {err.strip()[-200:]}")

        for budget, pourquoi in (("-100", "négatif"), ("0", "nul"), ("abc", "pas un nombre")):
            argv = ["--clients", str(clients), "--budget", budget, "--sortie", str(dossier / "refus.csv")]
            try:
                code, _, err = _lancer(main, argv)
            except ValueError as exc:
                raise AssertionError(f"`{_commande(argv)}` : ValueError ({exc}). Laissez argparse refuser le montant : "
                                     "dans votre fonction `type=`, levez `argparse.ArgumentTypeError`.") from None
            if code != 2:
                raise AssertionError(f"`{_commande(argv)}` (budget {pourquoi}) : code de sortie {code}, attendu 2 (refus "
                                     "par argparse). Écrivez une fonction de conversion pour `type=` qui lève "
                                     "`argparse.ArgumentTypeError` si le montant n'est pas strictement positif.")

        argv = ["--clients", str(clients)]
        code, _, _ = _lancer(main, argv)
        if code != 2:
            raise AssertionError(f"`{_commande(argv)}` : --budget est obligatoire (`required=True`), attendu le code 2, "
                                 f"obtenu {code}.")

        absent = dossier / "nexiste_pas.csv"
        argv = ["--clients", str(absent), "--budget", "100", "--sortie", str(dossier / "x.csv")]
        try:
            code, _, err = _lancer(main, argv)
        except FileNotFoundError:
            raise AssertionError(f"`{_commande(argv)}` : FileNotFoundError non attrapée, l'utilisateur voit une traceback. "
                                 "Attrapez-la, affichez un message clair sur sys.stderr et `return 1`.") from None
        if code != 1 or not err.strip():
            raise AssertionError(f"`{_commande(argv)}` (fichier absent) : attendu un message sur la sortie d'erreur et le "
                                 f"code 1, obtenu le code {code}.")

        incomplet = _ecrire(dossier, "incomplet.csv", "customer_id,contract\nC1,Mensuel\n")
        argv = ["--clients", str(incomplet), "--budget", "100", "--sortie", str(dossier / "y.csv")]
        try:
            code, _, err = _lancer(main, argv)
        except ValueError:
            raise AssertionError(f"`{_commande(argv)}` : la ValueError de lire_clients (colonnes manquantes) n'est pas "
                                 "attrapée. Affichez son message sur sys.stderr et `return 1`.") from None
        if code != 1:
            raise AssertionError(f"`{_commande(argv)}` (colonnes manquantes) : code {code}, attendu 1.")

        ici = Path.cwd()
        (dossier / "defaut").mkdir()
        try:
            os.chdir(dossier / "defaut")
            code, _, _ = _lancer(main, ["--clients", str(clients), "--budget", "150"])
        finally:
            os.chdir(ici)
        if code != 0 or not (dossier / "defaut" / "plan_retention.csv").exists():
            raise AssertionError("Sans --sortie, le plan doit être écrit dans plan_retention.csv (dossier courant) : "
                                 "`default=\"plan_retention.csv\"`.")


# ============================================================================ Étape 7 : tester allouer_budget


def _trier(actions):
    return sorted((a for a in actions if a.gain_attendu > 0), key=lambda a: (-a.rentabilite, a.client.customer_id))


def _mutant_break(actions, budget):
    if budget < 0:
        raise ValueError("budget négatif")
    retenues, reste = [], budget
    for a in _trier(actions):
        if a.offre.cout > reste:
            break
        retenues.append(a)
        reste -= a.offre.cout
    return retenues


def _mutant_par_gain(actions, budget):
    if budget < 0:
        raise ValueError("budget négatif")
    retenues, reste = [], budget
    for a in sorted((a for a in actions if a.gain_attendu > 0), key=lambda a: (-a.gain_attendu, a.client.customer_id)):
        if a.offre.cout <= reste:
            retenues.append(a)
            reste -= a.offre.cout
    return retenues


def _mutant_negatifs(actions, budget):
    if budget < 0:
        raise ValueError("budget négatif")
    retenues, reste = [], budget
    for a in sorted(actions, key=lambda a: (-a.rentabilite, a.client.customer_id)):
        if a.offre.cout <= reste:
            retenues.append(a)
            reste -= a.offre.cout
    return retenues


def _mutant_strict(actions, budget):
    if budget < 0:
        raise ValueError("budget négatif")
    retenues, reste = [], budget
    for a in _trier(actions):
        if a.offre.cout < reste:
            retenues.append(a)
            reste -= a.offre.cout
    return retenues


def _mutant_sans_budget(actions, budget):
    if budget < 0:
        raise ValueError("budget négatif")
    return _trier(actions)


def _mutant_modifie_entree(actions, budget):
    if isinstance(actions, list):
        actions.sort(key=lambda a: (-a.rentabilite, a.client.customer_id))
    return ref_allocation.allouer_budget(actions, budget)


def _mutant_budget_negatif(actions, budget):
    return ref_allocation.allouer_budget(actions, max(budget, 0.0))


_MUTANTS = [
    (_mutant_break, "s'arrête (`break`) à la première action trop chère",
     "Mettez une action chère et très rentable AVANT une action bon marché qui, elle, rentre dans le budget."),
    (_mutant_par_gain, "trie par gain au lieu de la rentabilité",
     "Prenez des actions où le plus gros gain n'est PAS le plus rentable (l'exemple A, B, C de la règle)."),
    (_mutant_negatifs, "garde les actions à gain négatif",
     "Ajoutez une action qui fait perdre de l'argent, avec un budget large, et vérifiez qu'elle n'est pas retenue."),
    (_mutant_strict, "refuse une action qui coûte exactement le budget restant (< au lieu de <=)",
     "Testez un budget égal pile au coût d'une action."),
    (_mutant_sans_budget, "ignore le budget",
     "Vérifiez qu'avec un petit budget, certaines actions rentables sont laissées de côté."),
    (_mutant_modifie_entree, "trie la liste de l'appelant (`actions.sort()`)",
     "Gardez une copie (`copie = list(actions)`) d'une liste NON triée, appelez la fonction, puis "
     "`assert actions == copie`."),
    (_mutant_budget_negatif, "accepte un budget négatif sans erreur",
     "Vérifiez qu'un budget de −1 lève ValueError (voir l'exemple try/except de la docstring)."),
]


def verifier_tester_allouer_budget(module: ModuleType) -> None:
    tester = module.tester_allouer_budget
    try:
        tester(ref_allocation.allouer_budget)
    except NotImplementedError:
        raise
    except AssertionError as exc:
        raise AssertionError(f"Votre test ÉCHOUE sur la version JUSTE de allouer_budget ({str(exc) or 'assert sans message'}). "
                             "Un test qui crie au loup est pire que pas de test : recalculez à la main vos valeurs "
                             "attendues en suivant la règle de l'étape 4.") from None
    except Exception as exc:
        raise AssertionError(f"Votre test plante sur la version JUSTE : {type(exc).__name__}: {exc}. Fabriquez vos "
                             "actions avec `_action(identifiant, cout=..., gain=...)`.") from None

    survivants = []
    for mutant, defaut, conseil in _MUTANTS:
        try:
            tester(mutant)
        except Exception:  # noqa: BLE001 — n'importe quelle erreur : le test a vu le bug
            continue
        survivants.append((defaut, conseil))
    if survivants:
        if len(survivants) == len(_MUTANTS):
            raise AssertionError("Votre test passe sur TOUTES les versions boguées : il ne vérifie rien. Comparez le "
                                 "résultat à la liste attendue : `assert allouer([b, c, a], 40.0) == [a, c]`.")
        defaut, conseil = survivants[0]
        raise AssertionError(f"{len(_MUTANTS) - len(survivants)}/{len(_MUTANTS)} bugs attrapés. Votre test laisse passer "
                             f"une version qui {defaut}. {conseil}")


# ============================================================================ Le projet

PROJET = Projet(
    numero=2,
    code="retention",
    titre="Plan de rétention",
    paquet_eleve="projets.retention",
    paquet_corrige="projets.retention_corrige",
    commande="python -m projets.retention --clients projets/exemple_clients.csv --budget 1500",
    etapes=(
        Etape(1, "Lire le fichier des clients", "🟢", "clients", ("lire_clients",), verifier_lire_clients),
        Etape(2, "Traduire la règle métier", "🟢", "valeur", ("perte_attendue", "gain_attendu"), verifier_valeur),
        Etape(3, "Choisir la meilleure offre", "🟡", "regles", ("offres_eligibles", "meilleure_offre"), verifier_regles),
        Etape(4, "Répartir le budget", "🟡", "allocation", ("allouer_budget",), verifier_allocation),
        Etape(5, "Résumer et écrire le plan", "🟢", "rapport", ("resumer", "ecrire_plan"), verifier_rapport),
        Etape(6, "Le programme en ligne de commande", "🟡", "cli", ("main",), verifier_cli),
        Etape(7, "Écrire le test qui protège l'étape 4", "🔴", "mes_tests", ("tester_allouer_budget",),
              verifier_tester_allouer_budget),
    ),
)

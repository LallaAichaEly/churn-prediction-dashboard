"""Vérifications automatiques de la série 14 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Le correcteur simule une nuit de production en une fraction de seconde : il fournit de fausses
variables d'environnement (et en cache de vraies dans `os.environ`, pour voir si vous les lisez),
un journal qui retient tout, un opérateur SMS qui tombe en panne sur commande, une horloge qu'il
avance à la main, des fuseaux horaires au moment du changement d'heure, et un générateur de lignes
qui « plante » au milieu de l'écriture d'un fichier.

Rien ne dort pour de vrai, rien n'est écrit en dehors d'un dossier temporaire.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import itertools
import logging
import math
import os
import random
import re
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .production import FAUX, PARIS, VRAIS, CircuitOuvert, Config, ErreurTemporaire, HorlogeFactice
from .verif import Defi, Serie

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")
JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def _court(valeur, n: int = 160) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


# ============================================================================ Défi 86 : configuration

_CLE = "cle-ultra-secrete-7f3a91"
_PIEGES_ENVIRON = {"CHURN_CLE_API": "cle-lue-dans-os-environ", "CHURN_SEUIL": "0.99", "CHURN_TAILLE_LOT": "7",
                   "CHURN_MODE_TEST": "oui"}
_DEFAUT = Config(cle_api=_CLE)


@contextlib.contextmanager
def _environ_piege():
    """Place de fausses variables CHURN_* dans le VRAI os.environ, le temps d'un appel."""
    avant = {nom: os.environ.get(nom) for nom in _PIEGES_ENVIRON}
    os.environ.update(_PIEGES_ENVIRON)
    try:
        yield
    finally:
        for nom, valeur in avant.items():
            if valeur is None:
                os.environ.pop(nom, None)
            else:
                os.environ[nom] = valeur


def _appeler_86(lire_config, env: dict):
    """Renvoie (config, None) ou (None, ValueError). Toute autre exception devient un message."""
    copie = dict(env)
    try:
        with _environ_piege():
            resultat = lire_config(copie)
    except NotImplementedError:
        raise
    except ValueError as erreur:
        resultat, probleme = None, erreur
    except Exception as erreur:  # noqa: BLE001 — on explique au lieu de planter
        conseil = (" Pour une variable absente, `env[nom]` lève une KeyError : utilisez `env.get(nom)`."
                   if isinstance(erreur, KeyError) else "")
        raise AssertionError(f"Avec env = {_court(env)}, votre fonction lève {type(erreur).__name__}: {erreur}. "
                             f"Une configuration invalide doit lever une ValueError, et une valide renvoyer une Config."
                             f"{conseil}") from None
    else:
        probleme = None
    if copie != env:
        raise AssertionError(f"Votre fonction MODIFIE le dictionnaire reçu : avant {_court(env)}, après {_court(copie)}. "
                             "En production, c'est `os.environ` : vous changeriez l'environnement de tout le programme "
                             "(et des programmes qu'il lance). Lisez `env` sans y écrire (`env.get`, pas `setdefault`).")
    if resultat is not None:
        _verifier_type_config(resultat, env)
        lus = {"cle_api": resultat.cle_api, "seuil": resultat.seuil, "taille_lot": resultat.taille_lot,
               "mode_test": resultat.mode_test}
        if lus["cle_api"] == _PIEGES_ENVIRON["CHURN_CLE_API"] or (
                "CHURN_SEUIL" not in env and lus["seuil"] == 0.99) or (
                "CHURN_TAILLE_LOT" not in env and lus["taille_lot"] == 7):
            raise AssertionError("Votre fonction lit `os.environ` au lieu du dictionnaire `env` reçu : le correcteur y a "
                                 f"caché CHURN_SEUIL=0.99 et CHURN_CLE_API={_PIEGES_ENVIRON['CHURN_CLE_API']!r}, et on les "
                                 f"retrouve dans votre Config avec env = {_court(env)}. Une fonction qui reçoit sa "
                                 "configuration se teste sans toucher au vrai environnement : n'utilisez que `env`.")
    return resultat, probleme


def _verifier_type_config(resultat, env) -> None:
    if not isinstance(resultat, Config):
        raise AssertionError(f"Avec env = {_court(env)}, votre fonction renvoie {_court(resultat)} "
                             f"({type(resultat).__name__}) au lieu d'une Config : "
                             "`return Config(cle_api=..., seuil=..., taille_lot=..., mode_test=...)`.")
    types = [("seuil", (int, float), "un nombre (float)"), ("taille_lot", int, "un entier (int)"),
             ("mode_test", bool, "un booléen (True ou False)"), ("cle_api", str, "un texte")]
    for champ, attendu, description in types:
        valeur = getattr(resultat, champ)
        if not isinstance(valeur, attendu) or (champ in ("seuil", "taille_lot") and isinstance(valeur, bool)):
            raise AssertionError(f"Avec env = {_court(env)}, le champ `{champ}` vaut {valeur!r} "
                                 f"({type(valeur).__name__}) : il faut {description}. Les variables d'environnement "
                                 "sont toujours du TEXTE : convertissez-les (`float(...)`, `int(...)`).")


def _attendre_config(lire_config, env, attendu: Config, quand: str) -> None:
    resultat, erreur = _appeler_86(lire_config, env)
    if erreur is not None:
        churn = {nom: texte for nom, texte in env.items() if nom.startswith("CHURN_")}
        citees = [nom for nom in churn if nom in str(erreur)] or list(churn)  # celles que VOTRE message accuse
        vides = [nom for nom in citees if not churn[nom].strip()]
        espaces = [nom for nom in citees if churn[nom].strip() and churn[nom] != churn[nom].strip()]
        mode = churn.get("CHURN_MODE_TEST", "").strip()
        if vides:
            conseil = (f" {', '.join(vides)} : une valeur vide ou faite d'espaces compte comme ABSENTE, donc on prend "
                       "la valeur par défaut. Testez la valeur après `.strip()` : `if not texte:`.")
        else:
            conseil = f" {', '.join(espaces)} : retirez d'abord les espaces autour de la valeur (`.strip()`)." if espaces else ""
            if "CHURN_MODE_TEST" in citees and mode.lower() != mode:
                conseil += " CHURN_MODE_TEST s'écrit en majuscules ou en minuscules : comparez `texte.lower()` à VRAIS et FAUX."
        raise AssertionError(f"{quand} : env = {_court(env)} est une configuration VALIDE, mais votre fonction lève "
                             f"ValueError({str(erreur)!r}). Attendu : {attendu!r} (clé d'API {attendu.cle_api!r})."
                             f"{conseil}")
    for champ in ("cle_api", "seuil", "taille_lot", "mode_test"):
        obtenu, voulu = getattr(resultat, champ), getattr(attendu, champ)
        if obtenu == voulu or (isinstance(voulu, float) and math.isclose(obtenu, voulu)):
            continue
        conseil = ""
        if champ == "mode_test" and obtenu is True and voulu is False:
            conseil = (" Si vous écrivez `bool(texte)` : tout texte non vide est vrai, `bool(\"false\")` et "
                       "`bool(\"0\")` valent True. Comparez avec les listes VRAIS et FAUX.")
        elif champ == "mode_test":
            conseil = " Mettez le texte en minuscules (`.lower()`) avant de le comparer à VRAIS et FAUX."
        elif isinstance(obtenu, str) and obtenu.strip() == voulu:
            conseil = " Retirez les espaces autour de la valeur : `.strip()`."
        elif champ in ("seuil", "taille_lot", "mode_test") and voulu == getattr(_DEFAUT, champ):
            conseil = (" Une variable vide ou faite d'espaces compte comme absente : on prend la valeur par défaut "
                       f"({voulu!r}).")
        raise AssertionError(f"{quand} : avec env = {_court(env)}, `{champ}` vaut {obtenu!r} au lieu de {voulu!r}."
                             f"{conseil}")


def _attendre_refus(lire_config, env, fautives: tuple[str, ...], pourquoi: str):
    resultat, erreur = _appeler_86(lire_config, env)
    if erreur is None:
        conseil = ""
        if "nan" in str(env.get("CHURN_SEUIL", "")).lower():
            conseil = (" NaN (« not a number ») n'est ni plus petit que 0 ni plus grand que 1 : le test "
                       "`seuil < 0 or seuil > 1` le laisse passer. Écrivez `if not 0 <= seuil <= 1:`, qui le refuse.")
        elif env.get("CHURN_TAILLE_LOT", "").strip() in ("2.5", "1e3", "7.9"):
            conseil = (" `int(float(texte))` transforme « 2.5 » en 2 sans prévenir : utilisez `int(texte)`, qui lève "
                       "une ValueError sur un nombre à virgule.")
        raise AssertionError(f"env = {_court(env)} doit être REFUSÉE ({pourquoi}), mais votre fonction renvoie "
                             f"{resultat!r}.{conseil}")
    message = str(erreur)
    oubliees = [nom for nom in fautives if nom not in message]
    if oubliees:
        if len(fautives) > 1 and len(oubliees) < len(fautives):
            raise AssertionError(f"Avec env = {_court(env)}, {len(fautives)} variables sont fausses "
                                 f"({', '.join(fautives)}), mais votre message ne cite pas {', '.join(oubliees)} : "
                                 f"{message!r}. Collectez TOUTES les erreurs dans une liste, puis levez UNE ValueError "
                                 "à la fin : sinon on corrige, on redéploie, et on découvre la faute suivante.")
        raise AssertionError(f"Avec env = {_court(env)} ({pourquoi}), votre ValueError ne cite pas le nom de la "
                             f"variable fautive ({', '.join(oubliees)}) : {message!r}. Sans le nom, la personne de garde "
                             "cherche au hasard parmi toutes les variables.")
    if _CLE in message:
        raise AssertionError(f"Votre message d'erreur contient la clé d'API en clair : {message!r}. Ce message finira "
                             "dans un journal (série 13, défi 81) : citez le NOM des variables, jamais la valeur de la clé.")
    return message


def verifier_lire_config(lire_config) -> None:
    # 1. L'utilisateur honnête : le minimum, puis une configuration complète.
    _attendre_config(lire_config, {"CHURN_CLE_API": _CLE}, _DEFAUT, "Seule la clé est donnée (valeurs par défaut)")
    _attendre_config(lire_config, {"CHURN_CLE_API": _CLE, "CHURN_SEUIL": "0.35", "CHURN_TAILLE_LOT": "1000",
                                   "CHURN_MODE_TEST": "true", "PATH": "/usr/bin", "HOME": "/home/lea"},
                     Config(_CLE, 0.35, 1000, True), "Configuration complète")
    _attendre_config(lire_config, {"CHURN_CLE_API": f"  {_CLE}\n", "CHURN_SEUIL": " 0.2 ", "CHURN_TAILLE_LOT": " 64 ",
                                   "CHURN_MODE_TEST": " Oui "}, Config(_CLE, 0.2, 64, True),
                     "Des espaces autour des valeurs (fréquent dans un fichier .env)")
    _attendre_config(lire_config, {"CHURN_CLE_API": _CLE, "CHURN_SEUIL": "0", "CHURN_TAILLE_LOT": "1"},
                     Config(_CLE, 0.0, 1, False), "Les bornes (seuil 0, lot de 1)")
    _attendre_config(lire_config, {"CHURN_CLE_API": _CLE, "CHURN_SEUIL": "1"}, Config(_CLE, 1.0), "Seuil de 1")
    _attendre_config(lire_config, {"CHURN_CLE_API": _CLE, "CHURN_SEUIL": "", "CHURN_TAILLE_LOT": "   ",
                                   "CHURN_MODE_TEST": ""}, _DEFAUT, "Des variables présentes mais vides")
    # 2. Les booléens : toutes les écritures prévues, en majuscules ou non.
    for texte in FAUX + tuple(t.upper() for t in FAUX) + ("False", "Non"):
        _attendre_config(lire_config, {"CHURN_CLE_API": _CLE, "CHURN_MODE_TEST": texte}, _DEFAUT,
                         f"CHURN_MODE_TEST={texte!r}")
    for texte in VRAIS + tuple(t.upper() for t in VRAIS) + ("True", "Oui"):
        _attendre_config(lire_config, {"CHURN_CLE_API": _CLE, "CHURN_MODE_TEST": texte}, Config(_CLE, mode_test=True),
                         f"CHURN_MODE_TEST={texte!r}")

    # 3. Chaque erreur, seule.
    _attendre_refus(lire_config, {}, ("CHURN_CLE_API",), "la clé d'API est obligatoire")
    _attendre_refus(lire_config, {"CHURN_CLE_API": "   "}, ("CHURN_CLE_API",), "une clé faite d'espaces est vide")
    for texte, pourquoi in [("0,35", "virgule"), ("1.5", "plus grand que 1"), ("-0.1", "négatif"), ("nan", "NaN"),
                            ("NaN", "NaN"), ("inf", "infini"), ("beaucoup", "pas un nombre")]:
        _attendre_refus(lire_config, {"CHURN_CLE_API": _CLE, "CHURN_SEUIL": texte}, ("CHURN_SEUIL",),
                        f"seuil {texte!r} : {pourquoi}")
    for texte, pourquoi in [("2.5", "pas entier"), ("0", "au moins 1"), ("-3", "négatif"), ("dix", "pas un nombre"),
                            ("1e3", "pas un entier écrit en chiffres")]:
        _attendre_refus(lire_config, {"CHURN_CLE_API": _CLE, "CHURN_TAILLE_LOT": texte}, ("CHURN_TAILLE_LOT",),
                        f"taille de lot {texte!r} : {pourquoi}")
    for texte in ("vrai", "2", "truee", "yes please"):
        _attendre_refus(lire_config, {"CHURN_CLE_API": _CLE, "CHURN_MODE_TEST": texte}, ("CHURN_MODE_TEST",),
                        f"{texte!r} n'est ni dans VRAIS ni dans FAUX")

    # 4. Toutes les erreurs d'un coup, et la clé jamais dans le message.
    _attendre_refus(lire_config, {"CHURN_SEUIL": "beaucoup", "CHURN_TAILLE_LOT": "2.5", "CHURN_MODE_TEST": "vrai"},
                    ("CHURN_CLE_API", "CHURN_SEUIL", "CHURN_TAILLE_LOT", "CHURN_MODE_TEST"), "quatre erreurs")
    _attendre_refus(lire_config, {"CHURN_CLE_API": _CLE, "CHURN_SEUIL": "2", "CHURN_MODE_TEST": "peut-être"},
                    ("CHURN_SEUIL", "CHURN_MODE_TEST"), "deux erreurs, et une vraie clé")


# ============================================================================ Défi 87 : journalisation


class _Memoire(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record)


def _texte(record: logging.LogRecord) -> str:
    try:
        return record.getMessage()
    except Exception as erreur:  # noqa: BLE001
        raise AssertionError(f"Un message de journal est mal formé : {record.msg!r} avec les arguments {record.args!r} "
                             f"({type(erreur).__name__}: {erreur}). Il faut autant de %s que de valeurs passées après "
                             "le message.") from None


_COMPTEUR_87 = itertools.count()


def _executer_87(scorer_lot, clients, score):
    """Appelle scorer_lot avec un journal espion. Renvoie (résultat ou exception, journal, racine, sortie, réglages)."""
    journal = logging.getLogger(f"defis.correcteur.lot_de_nuit_{next(_COMPTEUR_87)}")
    memoire, racine_memoire = _Memoire(), _Memoire()
    racine = logging.getLogger()
    journal.setLevel(logging.DEBUG)
    journal.propagate = False
    journal.addHandler(memoire)
    niveau_racine, gestionnaires_racine = racine.level, list(racine.handlers)
    racine.addHandler(racine_memoire)
    racine.setLevel(logging.DEBUG)
    basic_config, appels_basic = logging.basicConfig, []
    logging.basicConfig = lambda *a, **k: appels_basic.append((a, k))
    sortie = io.StringIO()
    try:
        with contextlib.redirect_stdout(sortie), contextlib.redirect_stderr(sortie):
            try:
                resultat = scorer_lot(clients, score, journal)
            except NotImplementedError:
                raise
            except BaseException as erreur:  # noqa: BLE001 — y compris KeyboardInterrupt, que l'on teste
                resultat = erreur
    finally:
        logging.basicConfig = basic_config
        racine.removeHandler(racine_memoire)
        reglages = {
            "basicConfig": bool(appels_basic),
            "gestionnaires du journal": [h for h in journal.handlers if h is not memoire],
            "niveau du journal": journal.level != logging.DEBUG,
            "gestionnaires racine": [h for h in racine.handlers if h not in gestionnaires_racine],
            "niveau racine": racine.level != logging.DEBUG,
        }
        for gestionnaire in list(journal.handlers):
            journal.removeHandler(gestionnaire)
        for gestionnaire in reglages["gestionnaires racine"]:
            racine.removeHandler(gestionnaire)
        racine.setLevel(niveau_racine)
    return resultat, memoire.messages, racine_memoire.messages, sortie.getvalue(), reglages


def _client(numero: int, facture="absente", anciennete: int = 12) -> dict:
    client = {"id": f"A1{numero:02d}", "email": f"client.a1{numero:02d}@exemple.fr", "anciennete_mois": anciennete}
    if facture != "absente":
        client["facture_mensuelle"] = facture
    return client


def _lot_87():
    clients = [_client(1, 70.0), _client(2, None), _client(3, 0.0), _client(4), _client(5, -5.0),
               _client(6, 30.0), _client(7, 55.0), _client(8, 80.0), _client(9, 12.0)]
    plantages = {"A106": ValueError("valeur manquante dans le vecteur de variables"), "A108": KeyError("contract")}
    appels: list[str] = []

    def score(client):
        appels.append(client["id"])
        if client["id"] in plantages:
            raise plantages[client["id"]]
        return round(client["facture_mensuelle"] / 100, 3)

    attendu = {"A101": 0.7, "A103": 0.0, "A107": 0.55, "A109": 0.12}
    return clients, score, appels, attendu, ("A102", "A104", "A105"), tuple(plantages)


def verifier_scorer_lot(scorer_lot) -> None:
    clients, score, appels, attendu, ignores, en_erreur = _lot_87()
    resultat, messages, racine, sortie, reglages = _executer_87(scorer_lot, iter(clients), score)

    # 1. Les mauvaises habitudes, visibles avant même le résultat.
    if sortie.strip():
        raise AssertionError(f"Votre fonction affiche du texte ({_court(sortie.strip(), 120)}) : la nuit, personne ne lit "
                             "l'écran. Remplacez chaque `print(...)` par `journal.warning(...)`, `journal.info(...)`…")
    if reglages["basicConfig"]:
        raise AssertionError("Votre fonction appelle `logging.basicConfig` : c'est au programme principal de configurer "
                             "les journaux, UNE fois. Une fonction qui le fait impose ses choix à toute l'application.")
    if reglages["gestionnaires du journal"] or reglages["gestionnaires racine"]:
        raise AssertionError("Votre fonction ajoute un gestionnaire (`addHandler`) au journal : à chaque lot, chaque "
                             "message s'écrirait une fois de plus. Configurer les journaux, c'est le rôle de "
                             "l'application ; la fonction se contente d'y écrire.")
    if reglages["niveau du journal"] or reglages["niveau racine"]:
        raise AssertionError("Votre fonction change le niveau du journal (`setLevel`) : c'est l'application qui décide "
                             "ce qu'elle veut voir (DEBUG pour enquêter, WARNING la nuit). La fonction écrit, c'est tout.")
    if racine:
        exemple = racine[0]
        raise AssertionError(f"Votre fonction écrit dans le journal racine (« {_texte(exemple)} », par "
                             f"`logging.{exemple.levelname.lower()}(...)` ou un autre journal) au lieu du journal "
                             f"REÇU : écrivez `journal.{exemple.levelname.lower()}(...)`.")
    if isinstance(resultat, BaseException):
        if isinstance(resultat, (ValueError, KeyError)) and str(resultat) in (
                "valeur manquante dans le vecteur de variables", "'contract'"):
            raise AssertionError(f"Le modèle a planté sur UN client ({type(resultat).__name__}: {resultat}) et votre "
                                 "fonction s'est arrêtée là : tout le reste du lot est perdu. Entourez `score(client)` "
                                 "d'un `try:` … `except Exception:` qui journalise et passe au client suivant.")
        raise AssertionError(f"Votre fonction lève {type(resultat).__name__}: {resultat}. Les clients invalides doivent "
                             "être ignorés (avec un avertissement), sans planter.")

    # 2. Le résultat.
    if resultat is None:
        raise AssertionError("Votre fonction renvoie None : avez-vous oublié `return scores` ?")
    if not isinstance(resultat, dict):
        raise AssertionError(f"Renvoyez un dictionnaire {{identifiant: score}}, pas {type(resultat).__name__}.")
    deja_notes = [i for i in ignores if i in appels]
    if deja_notes:
        raise AssertionError(f"Vous appelez `score` pour {', '.join(deja_notes)}, dont la facture est absente, None ou "
                             "négative. Vérifiez la facture AVANT : un vrai modèle planterait (ou pire, noterait un "
                             "client sur une facture fausse).")
    if "A103" not in resultat:
        raise AssertionError("Le client A103 a une facture de 0 € : elle est valide, il doit être noté. `if not facture` "
                             "écarte 0 (0 est « faux » en Python) : écrivez `if facture is None or facture < 0`.")
    if resultat != attendu:
        raise AssertionError(f"Votre fonction renvoie {_court(resultat)}, attendu {attendu} : les clients notés et "
                             "leur score, sans les ignorés ni ceux sur lesquels le modèle a planté.")

    # 3. Les avertissements : un par client ignoré, au niveau WARNING, avec l'identifiant.
    textes = [(m, _texte(m)) for m in messages]
    for identifiant in ignores:
        siens = [m for m, texte in textes if identifiant in texte]
        if not siens:
            raise AssertionError(f"Le client {identifiant} a été ignoré (facture invalide), mais aucun message du journal "
                                 "ne cite son identifiant. Le lendemain, comment savoir QUI n'a pas été noté ? "
                                 f"`journal.warning(\"client %s ignoré : ...\", identifiant)`.")
        if not any(m.levelno == logging.WARNING for m in siens):
            niveau = siens[0].levelname
            raise AssertionError(f"Le client {identifiant} (facture invalide) est journalisé au niveau {niveau} au lieu "
                                 "de WARNING. Une donnée invalide n'est pas une erreur du PROGRAMME (ERROR), et c'est "
                                 "plus qu'une simple information (INFO) : c'est un avertissement.")
    # 4. Les erreurs du modèle : niveau ERROR, avec l'identifiant ET la traceback.
    for identifiant in en_erreur:
        siens = [m for m, texte in textes if identifiant in texte and m.levelno >= logging.ERROR]
        if not siens:
            autres = [m.levelname for m, texte in textes if identifiant in texte]
            detail = f"(il est journalisé au niveau {autres[0]})" if autres else "(aucun message ne le cite)"
            raise AssertionError(f"Le modèle a planté sur le client {identifiant} {detail}. C'est une erreur du programme : "
                                 "écrivez-la au niveau ERROR, avec l'identifiant : `journal.exception(\"client %s : "
                                 "le modèle a échoué\", identifiant)` dans le `except`.")
        if not any(m.exc_info and m.exc_info[0] is not None for m in siens):
            raise AssertionError(f"L'erreur du client {identifiant} est journalisée sans sa traceback : on sait QUI a "
                                 "planté, mais pas OÙ ni POURQUOI. Dans le `except`, `journal.exception(...)` ajoute la "
                                 "traceback toute seule (c'est `journal.error(..., exc_info=True)`).")
    # 5. Le résumé : un seul message INFO, avec les trois compteurs.
    infos = [(m, texte) for m, texte in textes if m.levelno == logging.INFO]
    if not infos:
        raise AssertionError("Il manque le résumé du lot au niveau INFO (`journal.info(\"lot terminé : %d clients "
                             "notés, %d ignorés, %d en erreur\", ...)`). C'est la ligne que l'on cherche le matin.")
    if len(infos) > 1:
        raise AssertionError(f"Votre fonction écrit {len(infos)} messages au niveau INFO pour 9 clients (par exemple "
                             f"« {infos[0][1]} »). Sur 100 000 clients, c'est 100 000 lignes chaque nuit, et le résumé "
                             "s'y noie. Un message par client, c'est `journal.debug(...)`, ou rien ; gardez INFO pour "
                             "le résumé.")
    resume = infos[0][1]
    for nombre, quoi in ((4, "notés"), (3, "ignorés"), (2, "en erreur")):
        if not re.search(rf"(?<![\d.]){nombre}(?![\d.])", resume):
            raise AssertionError(f"Le résumé « {resume} » doit contenir le nombre de clients {quoi} ({nombre}) : "
                                 "4 notés, 3 ignorés, 2 en erreur.")
    # 6. Pas de données personnelles dans le journal.
    for m, texte in textes:
        if "@exemple.fr" in texte:
            raise AssertionError(f"Le journal contient une adresse e-mail : « {_court(texte, 120)} ». Les journaux sont "
                                 "lus par beaucoup de monde et gardés longtemps : journalisez l'IDENTIFIANT du client, "
                                 "jamais la ligne entière (RGPD).")

    # 7. Un lot vide, et Ctrl+C pendant le lot.
    resultat, messages, _, _, _ = _executer_87(scorer_lot, [], score)
    if resultat != {}:
        raise AssertionError(f"Pour un lot vide, renvoyez {{}} (et non {_court(resultat)}).")
    if sum(m.levelno == logging.INFO for m in messages) != 1:
        raise AssertionError("Même pour un lot vide, écrivez le résumé (« 0 clients notés… ») : un lot vide est "
                             "justement suspect, il faut qu'on le voie.")

    def interrompu(client):
        if client["id"] == "A103":
            raise KeyboardInterrupt
        return 0.5

    resultat, _, _, _, _ = _executer_87(scorer_lot, [_client(1, 10.0), _client(3, 10.0), _client(7, 10.0)], interrompu)
    if not isinstance(resultat, KeyboardInterrupt):
        raise AssertionError("Quelqu'un appuie sur Ctrl+C pendant le lot (KeyboardInterrupt), et votre fonction continue "
                             "comme si de rien n'était : `except:` tout court (ou `except BaseException`) attrape aussi "
                             "Ctrl+C. Écrivez `except Exception:`, qui laisse passer l'arrêt demandé.")


# ============================================================================ Défi 88 : reprises


class _Service88:
    """Un faux service : chaque appel consomme le prochain élément du scénario (exception ou valeur)."""

    def __init__(self, *scenario) -> None:
        self.scenario = list(scenario)
        self.appels = 0

    def __call__(self, *args, **kwargs):
        if args or kwargs:
            raise AssertionError(f"`appel` doit être appelé SANS argument (`appel()`), pas avec {args} {kwargs}.")
        self.appels += 1
        if not self.scenario:
            raise AssertionError("Votre fonction appelle le service plus de fois que prévu.")
        issue = self.scenario.pop(0)
        if isinstance(issue, BaseException):
            raise issue
        return issue


def _executer_88(fonction, service, **options):
    """Renvoie (résultat ou exception, délais demandés à `dormir`, appels à time.sleep)."""
    delais: list[float] = []
    vrais_sommeils: list[float] = []
    sleep = time.sleep
    time.sleep = vrais_sommeils.append
    try:
        resultat = fonction(service, dormir=delais.append, **options)
    except NotImplementedError:
        raise
    except Exception as erreur:  # noqa: BLE001
        resultat = erreur
    finally:
        time.sleep = sleep
    if isinstance(resultat, AttributeError) and "attente" in str(resultat):
        raise AssertionError("Une ConnectionError ou une TimeoutError n'a pas d'attribut `attente` : `erreur.attente` "
                             "plante. Écrivez `getattr(erreur, \"attente\", None)`, qui renvoie None quand l'attribut "
                             "n'existe pas.")
    if vrais_sommeils:
        raise AssertionError(f"Votre fonction appelle `time.sleep({vrais_sommeils[0]})` directement : en vrai, le test "
                             "attendrait pour de bon. Appelez la fonction reçue, `dormir(delai)` : c'est elle que le "
                             "correcteur (et vos tests) remplacent.")
    return resultat, delais


def _memes_delais(obtenus, attendus) -> bool:
    return len(obtenus) == len(attendus) and all(math.isclose(a, b) for a, b in zip(obtenus, attendus))


def verifier_appeler_avec_reprises(fonction) -> None:
    # 1. Tout va bien : le résultat, tout de suite, quel qu'il soit.
    for valeur in (object(), None, 0, ""):
        service = _Service88(valeur)
        resultat, delais = _executer_88(fonction, service)
        if isinstance(resultat, Exception) or resultat is not valeur:
            raise AssertionError(f"Le service répond {valeur!r} du premier coup, et votre fonction renvoie "
                                 f"{_court(resultat)}. Renvoyez le résultat de `appel()` tel quel, même s'il est None, "
                                 "0 ou vide : `return appel()`. Un résultat « faux » n'est pas un échec : seule une "
                                 "EXCEPTION en est un.")
        if delais or service.appels != 1:
            raise AssertionError(f"Le service répond {valeur!r} du premier coup, mais votre fonction "
                                 f"{'attend ' + str(delais) if delais else 'le rappelle'} : un résultat « faux » "
                                 "(None, 0, \"\") n'est pas un échec. Seule une EXCEPTION est un échec.")

    # 2. Deux pannes passagères, puis un succès.
    service = _Service88(ErreurTemporaire(), ErreurTemporaire(), "SMS envoyé")
    resultat, delais = _executer_88(fonction, service)
    if isinstance(resultat, ErreurTemporaire):
        raise AssertionError("Le service échoue deux fois (ErreurTemporaire) puis répond, mais votre fonction "
                             "abandonne dès le premier échec. Une erreur de TRANSITOIRES doit être réessayée.")
    if resultat != "SMS envoyé":
        raise AssertionError(f"Le service échoue deux fois puis répond « SMS envoyé » ; votre fonction renvoie "
                             f"{_court(resultat)}.")
    if not _memes_delais(delais, [0.5, 1.0]):
        conseil = (" On dirait que vous commencez à base * 2 au lieu de base : avant la PREMIÈRE reprise (k = 0), "
                   "on attend `base * 2**0` = base." if _memes_delais(delais, [1.0, 2.0]) else
                   " Le délai doit DOUBLER à chaque reprise (attente exponentielle) : base * 2**k." if
                   _memes_delais(delais, [0.5, 0.5]) else "")
        raise AssertionError(f"Deux échecs puis un succès : votre fonction attend {delais} au lieu de [0.5, 1.0] "
                             f"(base=0.5).{conseil}")

    # 3. Toutes les tentatives échouent : la DERNIÈRE erreur remonte, sans attente après elle.
    erreurs = [ErreurTemporaire(f"échec n° {i}") for i in range(1, 5)]
    service = _Service88(*erreurs, "jamais atteint")
    resultat, delais = _executer_88(fonction, service)
    if resultat == "jamais atteint" or service.appels > 4:
        raise AssertionError(f"Votre fonction appelle le service {service.appels} fois avec tentatives=4. `tentatives` "
                             "est le nombre TOTAL d'appels (1 appel + 3 reprises), pas le nombre de reprises.")
    if not isinstance(resultat, Exception):
        raise AssertionError(f"Les 4 tentatives échouent, et votre fonction renvoie {_court(resultat)} sans erreur : "
                             "l'appelant croira que le SMS est parti. Après le dernier échec, relancez l'erreur (`raise`).")
    if resultat is not erreurs[-1]:
        quoi = ("la PREMIÈRE erreur" if resultat in erreurs else
                f"une nouvelle erreur ({type(resultat).__name__}: {resultat})")
        raise AssertionError(f"Après 4 échecs, votre fonction lève {quoi} au lieu de la DERNIÈRE. Dans le `except`, un "
                             "simple `raise` relance l'erreur en cours, avec son type et sa traceback : l'appelant sait "
                             "exactement ce qui s'est passé (et peut l'attraper avec `except ErreurTemporaire`).")
    if len(delais) == 4:
        raise AssertionError(f"Après le DERNIER échec, votre fonction attend encore ({delais}) avant d'abandonner : "
                             f"{delais[-1]:g} s perdues pour rien à chaque abandon, et le lot entier ralentit d'autant. "
                             "N'attendez que s'il reste une tentative.")
    if not _memes_delais(delais, [0.5, 1.0, 2.0]):
        raise AssertionError(f"Quatre échecs : votre fonction attend {delais} au lieu de [0.5, 1.0, 2.0].")

    # 4. D'autres réglages, et le plafond.
    service = _Service88(*[TimeoutError("délai dépassé")] * 5, ConnectionError("connexion coupée"), "ok")
    resultat, delais = _executer_88(fonction, service, tentatives=7, base=1, plafond=5)
    if resultat != "ok":
        raise AssertionError(f"Avec TimeoutError et ConnectionError (des erreurs passagères de TRANSITOIRES), votre "
                             f"fonction renvoie {_court(resultat)} au lieu de réessayer. Attrapez `TRANSITOIRES`, pas "
                             "seulement ErreurTemporaire.")
    if not _memes_delais(delais, [1, 2, 4, 5, 5, 5]):
        raise AssertionError(f"Avec base=1 et plafond=5, six reprises : votre fonction attend {delais} au lieu de "
                             "[1, 2, 4, 5, 5, 5]. Le délai double, mais ne dépasse jamais le plafond : "
                             "`min(plafond, base * 2**k)`. Utilisez les paramètres reçus, pas 0.5 et 8 écrits en dur.")

    # 5. Les erreurs définitives remontent tout de suite.
    for scenario, nom in [((ValueError("numéro de téléphone invalide"), "ok"), "ValueError"),
                          ((ErreurTemporaire(), KeyError("client"), "ok"), "KeyError après une panne passagère")]:
        service = _Service88(*scenario)
        resultat, delais = _executer_88(fonction, service)
        attendue = next(e for e in scenario if isinstance(e, (ValueError, KeyError)))
        if resultat is not attendue:
            raise AssertionError(f"Scénario « {nom} » : votre fonction {'renvoie ' + _court(resultat) if not isinstance(resultat, Exception) else 'lève ' + type(resultat).__name__} "
                                 f"au lieu de laisser remonter l'erreur {type(attendue).__name__} tout de suite. Une "
                                 "erreur qui n'est pas dans TRANSITOIRES ne disparaîtra pas en réessayant : n'attrapez "
                                 "que `except TRANSITOIRES`.")
        if len(delais) != service.appels - 1:
            raise AssertionError(f"Scénario « {nom} » : votre fonction attend {delais} avant de laisser remonter "
                                 "l'erreur définitive. Elle doit remonter sans attendre.")

    # 6. Le service dit combien de temps attendre (Retry-After).
    for attente, plafond, attendu in [(3, 8.0, 3), (20, 8.0, 8.0), (0, 8.0, 0)]:
        service = _Service88(ErreurTemporaire("trop de requêtes", attente=attente), "ok")
        resultat, delais = _executer_88(fonction, service, plafond=plafond)
        if resultat != "ok" or not _memes_delais(delais, [attendu]):
            conseil = (" `if erreur.attente:` est faux quand attente vaut 0 : écrivez `if attente is not None`."
                       if attente == 0 else " Sans dépasser le plafond : `min(plafond, attente)`." if attente > plafond
                       else "")
            raise AssertionError(f"Le service lève ErreurTemporaire(attente={attente}) : il demande d'attendre "
                                 f"{attente} s. Avec plafond={plafond}, votre fonction attend {delais} au lieu de "
                                 f"[{attendu}].{conseil}")
    service = _Service88(ConnectionError("coupure"), "ok")
    resultat, delais = _executer_88(fonction, service)
    if isinstance(resultat, AttributeError):
        raise AssertionError("Une ConnectionError n'a pas d'attribut `attente` : `erreur.attente` plante. Écrivez "
                             "`getattr(erreur, \"attente\", None)`, qui renvoie None quand l'attribut n'existe pas.")

    # 7. Les cas limites de `tentatives`.
    service = _Service88(ErreurTemporaire("unique"), "jamais atteint")
    resultat, delais = _executer_88(fonction, service, tentatives=1)
    if not isinstance(resultat, ErreurTemporaire) or delais:
        raise AssertionError(f"Avec tentatives=1 : un seul appel, aucune attente, et l'erreur remonte. Votre fonction "
                             f"renvoie {_court(resultat)} après avoir attendu {delais}.")
    service = _Service88("jamais atteint")
    resultat, _ = _executer_88(fonction, service, tentatives=0)
    if not isinstance(resultat, ValueError) or service.appels:
        raise AssertionError(f"Avec tentatives=0, levez une ValueError sans appeler le service (votre fonction "
                             f"{'appelle le service et ' if service.appels else ''}renvoie {_court(resultat)}).")


# ============================================================================ Défi 89 : fuseaux horaires


def _attendu_89(maintenant: datetime) -> datetime:
    jour = maintenant.astimezone(PARIS).date()
    while True:
        candidat = datetime(jour.year, jour.month, jour.day, 10, tzinfo=PARIS)
        if candidat > maintenant and candidat.weekday() < 5:
            return candidat.astimezone(UTC)
        jour += timedelta(days=1)


def _paris(instant: datetime) -> str:
    local = instant.astimezone(PARIS)
    return f"{JOURS[local.weekday()]} {local:%d/%m/%Y %H:%M:%S} à Paris (UTC{local:%z})"


def _comparer_89(prochain_envoi, maintenant: datetime, quand: str = "") -> None:
    attendu = _attendu_89(maintenant)
    try:
        obtenu = prochain_envoi(maintenant)
    except NotImplementedError:
        raise
    except Exception as erreur:  # noqa: BLE001
        raise AssertionError(f"Pour maintenant = {maintenant.isoformat()}{quand}, votre fonction lève "
                             f"{type(erreur).__name__}: {erreur}.") from None
    contexte = (f"Pour maintenant = {maintenant.isoformat()}{quand}, soit {_paris(maintenant)}, le prochain envoi est "
                f"{_paris(attendu)}, c'est-à-dire {attendu:%Y-%m-%d %H:%M} UTC")
    if not isinstance(obtenu, datetime):
        raise AssertionError(f"{contexte}. Votre fonction renvoie {_court(obtenu)} au lieu d'un datetime.")
    if obtenu.tzinfo is None or obtenu.utcoffset() is None:
        raise AssertionError(f"{contexte}. Votre fonction renvoie une date NAÏVE ({obtenu!r}), sans fuseau : celui qui "
                             "la lit ne sait pas de quelle heure il s'agit. Renvoyez `candidat.astimezone(timezone.utc)`.")
    if obtenu.utcoffset() != timedelta(0):
        raise AssertionError(f"{contexte}. Votre fonction renvoie le bon instant ? Peut-être, mais exprimé en "
                             f"UTC{obtenu:%z} ({obtenu.isoformat()}) : le serveur travaille en UTC. Terminez par "
                             "`.astimezone(timezone.utc)`.")
    if obtenu == attendu:
        return
    local, voulu = obtenu.astimezone(PARIS), attendu.astimezone(PARIS)
    debut = f"{contexte}. Votre fonction renvoie {obtenu:%Y-%m-%d %H:%M:%S.%f} UTC, soit {_paris(obtenu)}"
    if local.date() == voulu.date() and abs(obtenu - attendu) == timedelta(hours=1):
        raise AssertionError(f"{debut} : une heure d'écart, c'est le CHANGEMENT D'HEURE. 10 h à Paris vaut 8 h UTC en "
                             "été et 9 h UTC en hiver : ne gardez pas le décalage de `maintenant`, et ne calculez pas "
                             "en UTC. Fabriquez le candidat en heure de Paris, "
                             "`datetime(jour.year, jour.month, jour.day, 10, tzinfo=PARIS)` : zoneinfo choisit le bon "
                             "décalage pour CE jour-là.")
    if local.date() == voulu.date() and (local.minute, local.second, local.microsecond) != (0, 0, 0):
        raise AssertionError(f"{debut} : le bon jour, mais pas 10 h 00 min 00 s pile. `replace(hour=10)` garde les "
                             "minutes, secondes et microsecondes de maintenant : fabriquez une date neuve à 10 h.")
    if local.date() == voulu.date():
        raise AssertionError(f"{debut} : le bon jour, mais pas 10 h à PARIS. `maintenant.replace(hour=10)` met 10 h "
                             "dans le fuseau de `maintenant` (UTC, New York…), et `replace(tzinfo=PARIS)` change "
                             "l'étiquette sans convertir l'heure. Convertissez d'abord : `maintenant.astimezone(PARIS)`.")
    autre_fuseau = maintenant.utcoffset() != maintenant.astimezone(PARIS).utcoffset()
    rappel = (f" Et `maintenant` est en UTC{maintenant:%z}, pas en heure de Paris : lisez l'heure APRÈS conversion, "
              "`maintenant.astimezone(PARIS)` (`replace(tzinfo=PARIS)` change l'étiquette sans convertir)."
              if autre_fuseau else "")
    if obtenu == maintenant:
        raise AssertionError(f"{debut} : maintenant lui-même. À 10 h pile, l'envoi du jour est déjà parti : il faut le "
                             f"suivant (`candidat > maintenant`, strictement).{rappel}")
    if obtenu < maintenant:
        raise AssertionError(f"{debut}, c'est-à-dire AVANT maintenant : ce SMS ne partira jamais. Souvent, c'est "
                             "`maintenant.replace(tzinfo=PARIS)` : il change l'étiquette du fuseau sans convertir "
                             "l'heure (8 h 30 UTC devient « 8 h 30 à Paris » au lieu de 10 h 30). Convertissez avec "
                             "`maintenant.astimezone(PARIS)`, et comparez le candidat à `maintenant` lui-même.")
    if local.weekday() >= 5:
        raise AssertionError(f"{debut} : un {JOURS[local.weekday()]}. Pas d'envoi le week-end, et le jour se lit à "
                             "Paris : `candidat.weekday() >= 5` (5 = samedi, 6 = dimanche) sur le candidat en heure "
                             "de Paris.")
    if voulu.weekday() >= 5 or maintenant.astimezone(PARIS).weekday() >= 5 or \
            maintenant.astimezone(PARIS).date() != maintenant.astimezone(UTC).date():
        raise AssertionError(f"{debut}. Le jour de la semaine et la date se lisent à PARIS : vendredi 23 h 30 UTC, c'est "
                             "déjà samedi à Paris. Partez de `maintenant.astimezone(PARIS).date()`.")
    raise AssertionError(f"{debut}.")


def verifier_prochain_envoi(prochain_envoi) -> None:
    # 1. Le refus d'une date naïve.
    naive = datetime(2026, 10, 21, 7, 0)
    try:
        obtenu = prochain_envoi(naive)
    except NotImplementedError:
        raise
    except ValueError:
        pass
    except Exception as erreur:  # noqa: BLE001
        raise AssertionError(f"Pour une date NAÏVE ({naive!r}, sans fuseau), levez une ValueError explicite ; votre "
                             f"fonction lève {type(erreur).__name__}: {erreur}. Testez `maintenant.tzinfo is None` "
                             "dès le début.") from None
    else:
        raise AssertionError(f"Pour une date NAÏVE ({naive!r}, sans fuseau), votre fonction renvoie {_court(obtenu)}. "
                             "Est-ce 7 h à Paris, à Londres, à Dakar ? Personne ne le sait : levez une ValueError.")

    # 2. Les cas qui expliquent chacun un piège.
    cas = [
        (datetime(2026, 10, 21, 7, 0, tzinfo=UTC), " (mercredi, 9 h à Paris)"),
        (datetime(2026, 10, 21, 7, 59, 59, 999_999, tzinfo=UTC), " (une microseconde avant 10 h à Paris)"),
        (datetime(2026, 10, 21, 8, 0, tzinfo=UTC), " (10 h pile à Paris : l'envoi du jour est parti)"),
        (datetime(2026, 10, 21, 7, 30, 45, 123_456, tzinfo=UTC), " (avec des secondes et des microsecondes)"),
        (datetime(2026, 1, 14, 12, 0, tzinfo=UTC), " (mercredi d'hiver, après l'envoi)"),
        (datetime(2026, 10, 23, 15, 0, tzinfo=UTC), " (vendredi : l'envoi suivant est APRÈS le changement d'heure)"),
        (datetime(2026, 3, 27, 12, 0, tzinfo=UTC), " (vendredi : passage à l'heure d'été le dimanche 29 mars)"),
        (datetime(2026, 10, 23, 22, 30, tzinfo=UTC), " (vendredi 22 h 30 UTC : déjà samedi à Paris)"),
        (datetime(2026, 12, 6, 23, 30, tzinfo=UTC), " (dimanche 23 h 30 UTC : déjà lundi à Paris)"),
        (datetime(2026, 10, 26, 6, 0, tzinfo=NEW_YORK), " (lundi 6 h à New York, 11 h à Paris)"),
        (datetime(2026, 10, 26, 18, 30, tzinfo=TOKYO), " (lundi 18 h 30 à Tokyo, 10 h 30 à Paris)"),
        (datetime(2026, 10, 21, 9, 0, tzinfo=PARIS), " (mercredi 9 h, déjà en heure de Paris)"),
        (datetime(2026, 10, 25, 2, 30, fold=1, tzinfo=PARIS), " (2 h 30 la nuit du changement d'heure, 2e passage)"),
    ]
    for maintenant, quand in cas:
        _comparer_89(prochain_envoi, maintenant, quand)

    # 3. Balayage : toutes les 47 minutes autour des changements d'heure, puis au hasard sur deux ans.
    for debut, fin in [(datetime(2026, 3, 20, tzinfo=UTC), datetime(2026, 4, 3, tzinfo=UTC)),
                       (datetime(2026, 10, 16, tzinfo=UTC), datetime(2026, 11, 3, tzinfo=UTC))]:
        instant = debut
        while instant < fin:
            _comparer_89(prochain_envoi, instant)
            instant += timedelta(minutes=47)
    hasard = random.Random(89)
    fuseaux = [UTC, PARIS, NEW_YORK, TOKYO, timezone(timedelta(hours=5, minutes=30)), timezone(timedelta(hours=-3))]
    for _ in range(400):
        instant = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=hasard.randrange(0, 730 * 86_400))
        _comparer_89(prochain_envoi, instant.astimezone(hasard.choice(fuseaux)))


# ============================================================================ Défi 90 : écriture atomique


class _Panne(Exception):
    """L'erreur levée exprès au milieu de l'écriture (disque plein, bogue dans le calcul des scores…)."""


_MODES_ECRITURE = set("wax+")
_FICHIERS_D_AUTRUI = ("scores.csv.tmp", "scores.tmp", "scores.csv.temp", "scores.csv.part", "scores.csv~",
                      ".scores.csv.tmp", "tmp")


@contextlib.contextmanager
def _espions_90(chemin: Path):
    """Note chaque ouverture de `chemin` en écriture et chaque renommage, pendant l'appel."""
    cible = os.path.realpath(chemin)
    ecritures: list[str] = []
    renommages: list[tuple[str, str, bytes | None]] = []
    open_builtin, open_io, replace, rename = builtins.open, io.open, os.replace, os.rename

    def chemin_de(argument):
        if isinstance(argument, (str, bytes, os.PathLike)):
            return os.path.realpath(os.fsdecode(argument))
        return None

    def open_espion(fichier, mode="r", *args, **kwargs):
        if chemin_de(fichier) == cible and _MODES_ECRITURE & set(str(mode)):
            ecritures.append(str(mode))
        return open_io(fichier, mode, *args, **kwargs)

    def noter(source, destination):
        # Ce que contient la destination JUSTE APRÈS le renommage : ce qu'un lecteur y trouverait à cet instant.
        arrivee = chemin_de(destination)
        contenu = Path(arrivee).read_bytes() if arrivee == cible and os.path.isfile(arrivee) else None
        renommages.append((chemin_de(source), arrivee, contenu))

    def replace_espion(source, destination, *args, **kwargs):
        resultat = replace(source, destination, *args, **kwargs)
        noter(source, destination)
        return resultat

    def rename_espion(source, destination, *args, **kwargs):
        resultat = rename(source, destination, *args, **kwargs)
        noter(source, destination)
        return resultat

    builtins.open = io.open = open_espion
    os.replace, os.rename = replace_espion, rename_espion
    try:
        yield ecritures, renommages
    finally:
        builtins.open, io.open, os.replace, os.rename = open_builtin, open_io, replace, rename


def _contenu(dossier: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(dossier.iterdir()) if p.is_file()}


def _ecrire_90(ecrire_atomique, chemin: Path, lignes):
    with _espions_90(chemin) as (ecritures, renommages):
        try:
            ecrire_atomique(chemin, lignes)
        except NotImplementedError:
            raise
        except Exception as erreur:  # noqa: BLE001
            return erreur, ecritures, renommages
    return None, ecritures, renommages


def _lire(chemin: Path) -> str | None:
    if not chemin.exists():
        return None
    with open(chemin, encoding="utf-8") as fichier:  # « \r\n » écrit sous Windows est relu comme « \n »
        return fichier.read()


def _surveiller(chemin: Path, ancien: bytes | None, lignes, panne: _Panne | None = None, apres: int = 0):
    """Un générateur de lignes qui, à CHAQUE ligne, vérifie que `chemin` n'a pas encore bougé."""
    abime: list[str] = []

    def generateur():
        for numero, ligne in enumerate(lignes):
            if panne is not None and numero == apres:
                raise panne
            actuel = chemin.read_bytes() if chemin.exists() else None
            if actuel != ancien and not abime:
                abime.append(_court(actuel.decode("utf-8", "replace") if actuel is not None else None, 80))
            yield ligne

    return generateur(), abime


def verifier_ecrire_atomique(ecrire_atomique) -> None:
    lignes = ["id;score;nom", "C001;0.82;Léa", "C003;0.15;Zébulon", "C004;0.97;D'Almeida"]
    attendu = "".join(f"{ligne}\n" for ligne in lignes)
    dossier = Path(tempfile.mkdtemp(prefix="defi90_"))
    try:
        chemin = dossier / "scores.csv"

        # 1. Un premier fichier : le bon contenu, et rien d'autre dans le dossier.
        erreur, ecritures, renommages = _ecrire_90(ecrire_atomique, chemin, list(lignes))
        if erreur is not None:
            raise AssertionError(f"Pour écrire {len(lignes)} lignes dans un dossier vide, votre fonction lève "
                                 f"{type(erreur).__name__}: {erreur}.")
        if ecritures:
            raise AssertionError(f"Votre fonction ouvre `chemin` lui-même en écriture (mode {ecritures[0]!r}) : pendant "
                                 "l'écriture, le tableau de bord lit un fichier à moitié écrit, et un plantage le laisse "
                                 "coupé. Écrivez dans un fichier TEMPORAIRE, puis `os.replace(temporaire, chemin)`.")
        contenu = _lire(chemin)
        if contenu is None:
            raise AssertionError("Après l'appel, `chemin` n'existe pas. Avez-vous oublié `os.replace(temporaire, chemin)` ?")
        if contenu != attendu:
            conseil = (" Le fichier est vide ou incomplet : renommez-le APRÈS l'avoir fermé (après le bloc `with`), "
                       "sinon une partie du texte est encore en mémoire." if attendu.startswith(contenu) else
                       " Chaque ligne est suivie d'un « \\n », y compris la dernière."
                       if contenu.replace("\n", "") == attendu.replace("\n", "") else
                       " Écrivez en UTF-8 : `encoding=\"utf-8\"`." if "Ã" in contenu else "")
            raise AssertionError(f"Le fichier contient {_court(contenu)} au lieu de {_court(attendu)}.{conseil}")
        restants = sorted(set(_contenu(dossier)) - {"scores.csv"})
        if restants:
            raise AssertionError(f"Après une écriture réussie, il reste {restants} dans le dossier : le fichier temporaire "
                                 "doit DEVENIR `chemin` (os.replace), pas rester à côté.")
        cible = os.path.realpath(chemin)
        vers_cible = [(source, octets) for source, destination, octets in renommages if destination == cible]
        if vers_cible and vers_cible[-1][1] is not None and vers_cible[-1][1] != attendu.encode("utf-8"):
            raise AssertionError(f"Juste après votre os.replace, le fichier contient "
                                 f"{_court(vers_cible[-1][1].decode('utf-8', 'replace'))} : le reste du texte était "
                                 "encore en mémoire, et n'est arrivé qu'à la fermeture. Un lecteur, à cet instant, "
                                 "trouvait un fichier incomplet. Renommez APRÈS l'avoir fermé (après le bloc `with`).")
        if vers_cible and os.path.dirname(vers_cible[-1][0]) != os.path.dirname(cible):
            raise AssertionError(f"Votre fichier temporaire est dans {os.path.dirname(vers_cible[-1][0])}, pas dans le dossier "
                                 f"de `chemin`. En production, /tmp est souvent sur un autre disque : os.replace échoue "
                                 "alors (« Invalid cross-device link »), ou devient une copie, qui n'est plus atomique. "
                                 "Créez-le à côté : `tempfile.mkstemp(dir=Path(chemin).parent)`.")

        # 2. Remplacer un fichier existant : pendant toute l'écriture, l'ancien reste intact.
        ancien = chemin.read_bytes()
        for nom in _FICHIERS_D_AUTRUI:
            (dossier / nom).write_text(f"fichier de {nom}, qui appartient à un autre traitement\n", encoding="utf-8")
        avant = _contenu(dossier)
        nouvelles = ["id;score;nom", "C002;0.41;Paul", "C005;0.08;Inès"]
        generateur, abime = _surveiller(chemin, ancien, nouvelles)
        erreur, ecritures, _ = _ecrire_90(ecrire_atomique, chemin, generateur)
        if erreur is not None:
            raise AssertionError(f"Pour remplacer un fichier existant, votre fonction lève {type(erreur).__name__}: "
                                 f"{erreur}. `os.replace` remplace la destination si elle existe (contrairement à "
                                 "`os.rename` sous Windows).")
        if abime or ecritures:
            raise AssertionError("PENDANT l'écriture, `chemin` ne contient déjà plus l'ancien fichier"
                                 f"{' (le tableau de bord y lirait ' + abime[0] + ')' if abime else ''} : écrivez tout "
                                 "dans le temporaire, et ne touchez à `chemin` qu'à la toute fin, d'un seul coup.")
        if _lire(chemin) != "".join(f"{ligne}\n" for ligne in nouvelles):
            raise AssertionError(f"Après remplacement, le fichier contient {_court(_lire(chemin))} au lieu des nouvelles "
                                 "lignes.")
        apres = _contenu(dossier)
        touches = [nom for nom in _FICHIERS_D_AUTRUI if apres.get(nom) != avant[nom]]
        if touches:
            raise AssertionError(f"Le fichier {touches[0]!r}, qui existait avant l'appel, a été écrasé ou supprimé. Un nom "
                                 "fixe pour le temporaire (« scores.csv.tmp ») entre en collision avec un autre "
                                 "traitement lancé en même temps, ou avec ses restes. Laissez Python choisir un nom "
                                 "unique : `tempfile.mkstemp(dir=...)` ou `NamedTemporaryFile(dir=..., delete=False)`.")
        restants = sorted(set(apres) - set(avant))
        if restants:
            raise AssertionError(f"Après une écriture réussie, il reste {restants} dans le dossier.")

        # 3. Une panne au milieu : l'ancien fichier intact, pas de temporaire, l'erreur remonte.
        ancien, avant = chemin.read_bytes(), _contenu(dossier)
        panne = _Panne("disque plein (simulé par le correcteur)")
        generateur, abime = _surveiller(chemin, ancien, ["id;score", "C001;0.9", "C002;0.1", "C003;0.5"], panne, apres=2)
        erreur, _, _ = _ecrire_90(ecrire_atomique, chemin, generateur)
        if not chemin.exists() or chemin.read_bytes() != ancien:
            raise AssertionError(f"Une panne survient après 2 lignes, et `chemin` contient maintenant "
                                 f"{_court(_lire(chemin))} au lieu de l'ancien fichier complet : la moitié des clients a "
                                 "disparu du tableau de bord. N'écrivez QUE dans le temporaire ; `chemin` ne change qu'à "
                                 "la fin, par os.replace, si tout s'est bien passé.")
        if erreur is None:
            raise AssertionError("Une panne survient au milieu de l'écriture, et votre fonction se termine sans erreur : "
                                 "le traitement de nuit croira que les scores sont à jour. Après avoir nettoyé, relancez "
                                 "l'erreur (`raise`).")
        if erreur is not panne:
            raise AssertionError(f"Une panne `_Panne(\"disque plein\")` survient, et votre fonction lève "
                                 f"{type(erreur).__name__}: {erreur} à la place. Relancez l'erreur d'origine avec un "
                                 "simple `raise` : on sait alors ce qui s'est vraiment passé.")
        restants = sorted(set(_contenu(dossier)) - set(avant))
        if restants:
            raise AssertionError(f"Après une panne, le fichier temporaire est resté dans le dossier ({restants}). Chaque "
                                 "nuit ratée en laisserait un de plus. Dans le `except`, supprimez-le "
                                 "(`Path(nom).unlink(missing_ok=True)`) avant le `raise`.")
        if _contenu(dossier) != avant:
            raise AssertionError("Après une panne, un fichier du dossier a changé : seul le temporaire peut être créé, "
                                 "puis supprimé.")

        # 4. Une panne alors que le fichier n'existait pas encore : il n'existe toujours pas.
        nouveau = dossier / "premiers_scores.csv"
        panne = _Panne("bogue dans le calcul des scores")
        generateur, _ = _surveiller(nouveau, None, ["id;score", "C001;0.9"], panne, apres=1)
        erreur, _, _ = _ecrire_90(ecrire_atomique, nouveau, generateur)
        if nouveau.exists():
            raise AssertionError(f"Une panne survient pendant la toute première écriture, et le fichier existe quand même "
                                 f"(il contient {_court(_lire(nouveau))}) : le tableau de bord croira qu'il est complet.")
        if erreur is not panne or sorted(set(_contenu(dossier)) - set(avant)):
            raise AssertionError("Une panne pendant la première écriture : l'erreur doit remonter, et le dossier rester "
                                 "comme avant (pas de temporaire oublié).")

        # 5. Zéro ligne : un fichier vide, mais qui existe.
        vide = dossier / "aucun_client.csv"
        erreur, _, _ = _ecrire_90(ecrire_atomique, vide, iter([]))
        if erreur is not None or _lire(vide) != "":
            raise AssertionError(f"Avec zéro ligne, le fichier doit exister et être vide ; votre fonction "
                                 f"{'lève ' + type(erreur).__name__ if erreur else 'écrit ' + _court(_lire(vide))}.")
    finally:
        shutil.rmtree(dossier, ignore_errors=True)


# ============================================================================ Défi 91 : disjoncteur


class _Service91:
    def __init__(self) -> None:
        self.appels: list[tuple] = []
        self.prochaine: BaseException | None = None

    def __call__(self, *args, **kwargs):
        self.appels.append((args, kwargs))
        if self.prochaine is not None:
            erreur, self.prochaine = self.prochaine, None
            raise erreur
        return ("réponse", args, kwargs)


def _appel_91(disjoncteur, service: _Service91, erreur: BaseException | None = None, *args, **kwargs):
    """Renvoie ("ok", résultat) ou ("erreur", exception)."""
    service.prochaine = erreur
    try:
        return "ok", disjoncteur.appeler(service, *args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        return "erreur", exc
    finally:
        service.prochaine = None


def _etat(disjoncteur) -> str:
    etat = disjoncteur.etat
    if callable(etat):
        raise AssertionError("`etat` est une méthode : on l'utilise comme un attribut (`sms.etat`, sans parenthèses). "
                             "Ajoutez `@property` juste au-dessus de `def etat(self)`.")
    if etat not in ("ferme", "ouvert", "semi_ouvert"):
        raise AssertionError(f"`etat` vaut {etat!r} : il doit valoir « ferme », « ouvert » ou « semi_ouvert » (sans "
                             "accent ni majuscule).")
    return etat


def _attendre_etat(disjoncteur, voulu: str, situation: str, conseil: str = "") -> None:
    etat = _etat(disjoncteur)
    if etat != voulu:
        raise AssertionError(f"{situation} : `etat` vaut {etat!r} au lieu de {voulu!r}.{' ' + conseil if conseil else ''}")


def _echouer(disjoncteur, service, situation: str, erreur: BaseException | None = None) -> None:
    """Un échec passager, qui doit remonter TEL QUEL (pas un CircuitOuvert, pas une autre erreur)."""
    erreur = erreur if erreur is not None else ErreurTemporaire("opérateur SMS indisponible")
    avant = len(service.appels)
    issue, valeur = _appel_91(disjoncteur, service, erreur)
    if len(service.appels) == avant:
        refus = f"lève {type(valeur).__name__}" if issue == "erreur" else f"renvoie {_court(valeur)}"
        raise AssertionError(f"{situation} : le service n'a pas été appelé (votre disjoncteur {refus}). "
                             "Seul l'état OUVERT refuse les appels.")
    if issue == "ok":
        raise AssertionError(f"{situation} : le service lève {type(erreur).__name__}, et `appeler` renvoie "
                             f"{_court(valeur)} au lieu de laisser remonter l'erreur. Le disjoncteur compte l'échec "
                             "puis le relance (`raise`) : l'appelant doit savoir que l'envoi a échoué.")
    if valeur is not erreur:
        conseil = (" C'est l'échec qui OUVRE le disjoncteur : il remonte tel quel. CircuitOuvert sert seulement à "
                   "refuser les appels SUIVANTS, sans appeler le service." if isinstance(valeur, CircuitOuvert) else "")
        raise AssertionError(f"{situation} : le service lève {type(erreur).__name__}({str(erreur)!r}), et `appeler` "
                             f"lève {type(valeur).__name__}({str(valeur)!r}) à la place. Relancez l'erreur d'origine "
                             f"avec `raise`.{conseil}")


def _reussir(disjoncteur, service, situation: str) -> None:
    issue, valeur = _appel_91(disjoncteur, service, None, "C001", texte="Votre offre")
    if issue == "erreur":
        raise AssertionError(f"{situation} : le service répond, mais `appeler` lève {type(valeur).__name__}: {valeur}.")
    if valeur != ("réponse", ("C001",), {"texte": "Votre offre"}):
        raise AssertionError(f"{situation} : `appeler(service, \"C001\", texte=\"Votre offre\")` doit renvoyer ce que "
                             f"renvoie `service(\"C001\", texte=\"Votre offre\")`. Votre disjoncteur renvoie "
                             f"{_court(valeur)}. Transmettez `*args, **kwargs` et faites `return`.")


def verifier_disjoncteur(cls) -> None:
    horloge = HorlogeFactice()
    try:
        sms = cls(horloge=horloge)
    except TypeError as erreur:
        raise AssertionError(f"`Disjoncteur(horloge=...)` plante ({erreur}) : gardez la signature "
                             "`__init__(self, seuil_echecs=3, delai=30.0, horloge=time.monotonic)`.") from None
    service = _Service91()

    # 1. Fermé : les appels passent, les échecs comptent, l'ouverture au 3e échec D'AFFILÉE.
    _attendre_etat(sms, "ferme", "Un disjoncteur neuf")
    _reussir(sms, service, "Disjoncteur fermé")
    _echouer(sms, service, "1er échec")
    _echouer(sms, service, "2e échec")
    _attendre_etat(sms, "ferme", "Après 2 échecs (seuil_echecs=3)",
                   "Il s'ouvre au 3e échec d'affilée, pas avant.")
    _reussir(sms, service, "Après 2 échecs, le service répond de nouveau")
    _echouer(sms, service, "Un échec après un succès")
    _attendre_etat(sms, "ferme", "Échec, échec, SUCCÈS, échec (seuil_echecs=3)",
                   "Ce sont les échecs D'AFFILÉE qui comptent : un succès remet le compteur à zéro.")
    _echouer(sms, service, "Un 2e échec après un succès")
    _attendre_etat(sms, "ferme", "Échec, échec, SUCCÈS, échec, échec (seuil_echecs=3)",
                   "Ce sont les échecs D'AFFILÉE qui comptent : un succès remet le compteur à zéro.")
    _echouer(sms, service, "3e échec d'affilée")
    _attendre_etat(sms, "ouvert", "Après 3 échecs d'affilée (seuil_echecs=3)",
                   "Au `seuil_echecs`-ième échec d'affilée, notez l'instant d'ouverture : `self._ouvert_depuis = "
                   "self._horloge()`.")

    # 2. Ouvert : refus immédiat, sans appeler le service.
    avant = len(service.appels)
    for _ in range(3):
        issue, valeur = _appel_91(sms, service, None)
        if len(service.appels) != avant:
            raise AssertionError("Le disjoncteur est OUVERT, et votre `appeler` appelle quand même le service : c'est tout "
                                 "ce qu'il doit éviter (attendre un service en panne, le surcharger). Levez "
                                 "CircuitOuvert AVANT d'appeler.")
        if issue != "erreur" or not isinstance(valeur, CircuitOuvert):
            raise AssertionError(f"Le disjoncteur est OUVERT : `appeler` doit lever CircuitOuvert ; il "
                                 f"{'renvoie ' + _court(valeur) if issue == 'ok' else 'lève ' + type(valeur).__name__}.")
    horloge.avancer(29.5)
    _attendre_etat(sms, "ouvert", "29,5 s après l'ouverture (delai=30)")

    # 3. Semi-ouvert au bout du délai, même sans appel, puis un essai réussi referme.
    horloge.avancer(0.5)
    etat = _etat(sms)
    if etat != "semi_ouvert":
        horloge.avancer(3600)
        plus_tard = _etat(sms)
        horloge.avancer(-3600)
        conseil = ("Une heure plus tard, il est toujours ouvert : calculez l'état avec l'horloge REÇUE (`self._horloge()`, "
                   "pas `time.monotonic()` ni `time.time()`), au moment où on le lit (dans la propriété `etat`, pas "
                   "seulement dans `appeler`)." if plus_tard == "ouvert" else
                   "À 30 s pile, le délai est écoulé : `horloge() - ouverture >= delai` (et non `>`).")
        raise AssertionError(f"30 s après l'ouverture (delai=30), sans aucun appel entre-temps, `etat` vaut {etat!r} au "
                             f"lieu de 'semi_ouvert'. {conseil}")
    _reussir(sms, service, "L'essai du disjoncteur semi-ouvert")
    _attendre_etat(sms, "ferme", "Après un essai RÉUSSI en semi-ouvert")
    _echouer(sms, service, "1er échec après la fermeture")
    _echouer(sms, service, "2e échec après la fermeture")
    _attendre_etat(sms, "ferme", "Refermé après un essai réussi, puis 2 échecs",
                   "La fermeture remet le compteur d'échecs à zéro : il en faut de nouveau 3.")

    # 4. Un essai raté rouvre tout de suite, et le délai repart de zéro.
    _echouer(sms, service, "3e échec")
    horloge.avancer(30)
    _attendre_etat(sms, "semi_ouvert", "30 s après la 2e ouverture")
    _echouer(sms, service, "L'essai du disjoncteur semi-ouvert échoue")
    _attendre_etat(sms, "ouvert", "Après un essai RATÉ en semi-ouvert",
                   "Un seul échec suffit à rouvrir (le service n'est pas revenu, inutile d'attendre 3 échecs), et le "
                   "délai repart de zéro : notez de nouveau l'instant d'ouverture.")
    horloge.avancer(29.5)
    _attendre_etat(sms, "ouvert", "29,5 s après l'essai raté",
                   "Le délai repart de zéro à la réouverture : notez de nouveau l'instant d'ouverture.")
    horloge.avancer(0.5)
    _attendre_etat(sms, "semi_ouvert", "30 s après l'essai raté")

    # 5. Les erreurs définitives ne changent pas l'état.
    for situation in ("En semi-ouvert, le service répond « numéro invalide » (ValueError)",):
        erreur = ValueError("numéro de téléphone invalide")
        issue, valeur = _appel_91(sms, service, erreur)
        if valeur is not erreur:
            raise AssertionError(f"{situation} : l'erreur doit remonter telle quelle ; votre disjoncteur "
                                 f"{'renvoie ' + _court(valeur) if issue == 'ok' else 'lève ' + type(valeur).__name__}.")
        _attendre_etat(sms, "semi_ouvert", situation,
                       "Une erreur hors de TRANSITOIRES ne change pas l'état : n'attrapez que `except TRANSITOIRES`.")
    _reussir(sms, service, "Puis l'essai réussit")
    _attendre_etat(sms, "ferme", "Après l'essai réussi")
    for numero in range(1, 6):
        erreur = KeyError("client")
        issue, valeur = _appel_91(sms, service, erreur)
        if valeur is not erreur:
            raise AssertionError(f"Le service lève KeyError : elle doit remonter telle quelle ; votre disjoncteur "
                                 f"{'renvoie ' + _court(valeur) if issue == 'ok' else 'lève ' + type(valeur).__name__}.")
        _attendre_etat(sms, "ferme", f"Après {numero} KeyError d'affilée",
                       "Seules les erreurs PASSAGÈRES (TRANSITOIRES) comptent comme des échecs : le service a répondu, "
                       "c'est la demande qui était mauvaise. Écrivez `except TRANSITOIRES:`, pas `except Exception:`.")

    # 6. D'autres réglages, toutes les erreurs passagères, et des disjoncteurs indépendants.
    horloge2 = HorlogeFactice(depart=50_000)
    email = cls(seuil_echecs=2, delai=5, horloge=horloge2)
    service2 = _Service91()
    _echouer(email, service2, "seuil_echecs=2, 1er échec")
    _attendre_etat(email, "ferme", "Un 2e disjoncteur (seuil_echecs=2) après 1 échec")
    _echouer(email, service2, "seuil_echecs=2, 2e échec")
    _attendre_etat(email, "ouvert", "seuil_echecs=2, après 2 échecs d'affilée",
                   "Utilisez `self.seuil_echecs` (reçu dans `__init__`), pas 3 écrit en dur.")
    _attendre_etat(sms, "ferme", "Le 1er disjoncteur, pendant que le 2e s'ouvre",
                   "Deux disjoncteurs ne partagent rien : créez chaque attribut dans `__init__` (`self.…`), "
                   "jamais au niveau de la CLASSE.")
    horloge2.avancer(5)
    _attendre_etat(email, "semi_ouvert", "delai=5, 5 s après l'ouverture",
                   "Utilisez `self.delai` (reçu dans `__init__`), pas 30 écrit en dur.")
    reseau = cls(seuil_echecs=2, delai=5, horloge=HorlogeFactice())
    _echouer(reseau, _Service91(), "ConnectionError", ConnectionError("connexion refusée"))
    _echouer(reseau, _Service91(), "TimeoutError", TimeoutError("délai dépassé"))
    _attendre_etat(reseau, "ouvert", "seuil_echecs=2, après une ConnectionError puis une TimeoutError",
                   "ConnectionError et TimeoutError sont aussi des pannes passagères : attrapez `TRANSITOIRES`.")
    neuf = cls(seuil_echecs=1, delai=10, horloge=HorlogeFactice())
    _attendre_etat(neuf, "ferme", "Un disjoncteur créé APRÈS que d'autres se sont ouverts",
                   "Son état ne doit dépendre que de lui : pas d'attribut de CLASSE partagé.")
    _echouer(neuf, _Service91(), "seuil_echecs=1, 1er échec")
    _attendre_etat(neuf, "ouvert", "seuil_echecs=1, après 1 échec")


SERIE = Serie(
    code="14",
    titre="Tenir en production : configuration, journaux, reprises, fuseaux horaires, écriture atomique, disjoncteur",
    module_eleve="defis.serie_14",
    module_corrige="defis.corrige_serie_14",
    defis=(
        Defi(86, "lire_config", "🟢", "—", verifier_lire_config),
        Defi(87, "scorer_lot", "🟢", "—", verifier_scorer_lot),
        Defi(88, "appeler_avec_reprises", "🟡", "—", verifier_appeler_avec_reprises),
        Defi(89, "prochain_envoi", "🟡", "—", verifier_prochain_envoi),
        Defi(90, "ecrire_atomique", "🟡", "—", verifier_ecrire_atomique),
        Defi(91, "Disjoncteur", "🔴", "—", verifier_disjoncteur),
    ),
)

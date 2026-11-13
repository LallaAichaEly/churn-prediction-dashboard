"""Vérifications automatiques du projet 4 (le service des conseillers).

Le correcteur fabrique ses propres bases et configurations, compte vos requêtes SQL, fait écrire
une collègue entre votre lecture et votre écriture, simule un disque plein, interroge votre API
comme le ferait un navigateur (et comme le ferait un attaquant), puis juge VOS tests sur des
versions boguées de l'API. Les messages visent les erreurs les plus fréquentes, pas seulement « faux ».

Lire ce fichier est permis (les cas testés sont des indices), mais essayez d'abord seul·e.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import logging
import os
import re
import sqlite3
import sys
import tempfile
import threading
import traceback
import types
from dataclasses import asdict, fields
from pathlib import Path
from types import ModuleType
from unittest import mock

from .checks_retention import _ecrire, _lancer, _pas_none
from .service_corrige import api as ref_api
from .service_corrige import base as ref_base
from .service_corrige import contacts as ref_contacts
from .service_corrige import fichiers as ref_fichiers
from .service_corrige.modeles import STATUTS
from .verif import Etape, Projet

_CORRIGE = Path(__file__).resolve().parent / "service_corrige"


def _modeles(module: ModuleType) -> ModuleType:
    """Le `modeles.py` du MÊME paquet que le module testé (le vôtre ou celui du corrigé)."""
    return importlib.import_module(f"{module.__package__}.modeles")


@contextlib.contextmanager
def _dossier_temporaire():
    with tempfile.TemporaryDirectory(prefix="projet_service_", ignore_cleanup_errors=True) as dossier:
        yield Path(dossier)


@contextlib.contextmanager
def _fermee(conn: sqlite3.Connection):
    try:
        yield conn
    finally:
        conn.close()


def _lieu(exc: BaseException, module: ModuleType) -> str:
    """« (api.py, ligne 42, dans `liste`) » : où votre code a planté, si c'est dans votre paquet."""
    dossier = Path(module.__file__).resolve().parent
    chez_vous = [f for f in traceback.extract_tb(exc.__traceback__) if Path(f.filename).resolve().parent == dossier]
    return f" ({Path(chez_vous[-1].filename).name}, ligne {chez_vous[-1].lineno}, dans `{chez_vous[-1].name}`)" \
        if chez_vous else ""


# ---------------------------------------------------------------------------- Une connexion qui compte vos requêtes


class _Curseur(sqlite3.Cursor):
    def execute(self, sql, parametres=()):
        self.connection._noter(sql)
        return super().execute(sql, parametres)

    def executemany(self, sql, lignes):
        self.connection._noter(sql)
        return super().executemany(sql, lignes)

    def executescript(self, script):
        self.connection._noter(script)
        return super().executescript(script)


class _Espion(sqlite3.Connection):
    """Une vraie connexion sqlite3 (même `isinstance`) qui note chaque requête et peut déclencher un
    événement juste avant l'une d'elles (une collègue qui écrit). `executemany` compte pour UNE."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requetes: list[str] = []
        self.avant = None  # avant(espion, sql), appelé avant chaque requête

    def _noter(self, sql: str) -> None:
        self.requetes.append(" ".join(str(sql).split()))
        if self.avant is not None:
            self.avant(self, sql)

    def cursor(self, factory=_Curseur):
        return super().cursor(factory)

    def execute(self, sql, parametres=()):
        return self.cursor().execute(sql, parametres)

    def executemany(self, sql, lignes):
        return self.cursor().executemany(sql, lignes)

    def executescript(self, script):
        return self.cursor().executescript(script)


def _espion(chemin: Path) -> _Espion:
    conn = sqlite3.connect(chemin, factory=_Espion, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.requetes.clear()
    return conn


# ---------------------------------------------------------------------------- Des bases connues


def _creer_base(chemin: Path, lignes) -> None:
    """Crée la base avec le schéma du projet et les contacts donnés (tuples dans l'ordre de Contact)."""
    with _fermee(ref_base.ouvrir(chemin)) as conn, conn:
        conn.executemany("INSERT INTO contacts (customer_id, offre, conseiller, statut, tentatives, commentaire, "
                         "version) VALUES (?, ?, ?, ?, ?, ?, ?)", lignes)


def _photo(chemin: Path) -> dict:
    """Le contenu des tables, vu depuis une AUTRE connexion : ce que voient les collègues."""
    with _fermee(sqlite3.connect(chemin, timeout=2)) as conn:
        return {
            "contacts": conn.execute("SELECT customer_id, offre, conseiller, statut, tentatives, commentaire, version "
                                     "FROM contacts ORDER BY customer_id").fetchall(),
            "appels": conn.execute("SELECT customer_id, conseiller, statut, commentaire FROM appels "
                                   "ORDER BY numero").fetchall(),
        }


def _fiche(photo: dict, customer_id: str):
    return next((ligne for ligne in photo["contacts"] if ligne[0] == customer_id), None)


def _differences(avant: dict, apres: dict) -> str:
    changees = [ligne for ligne in apres["contacts"] if ligne not in avant["contacts"]]
    texte = f"fiches modifiées : {changees[:3]}" if changees else ""
    if len(apres["appels"]) != len(avant["appels"]):
        texte += f"{' ; ' if texte else ''}{len(apres['appels']) - len(avant['appels'])} ligne(s) ajoutée(s) au journal"
    return texte or "rien de visible"


# ============================================================================ Étape 1 : lire_config

_CLE_A = "cle-alice-0123456789"
_CLE_B = "YWJjZGVmZ2hpamtsbW5vcA=="  # une clé en base64 : elle se termine par des « = »


def _lire_config(lire, env: dict, cas: str):
    try:
        return lire(env)
    except NotImplementedError:
        raise
    except KeyError as exc:
        raise AssertionError(f"{cas} : KeyError {exc}. Une variable absente n'est pas une erreur de programmation : "
                             "lisez-la avec env.get(nom), puis décidez (valeur par défaut, ou erreur de configuration "
                             "dans la liste).") from None


def _config_refusee(m: ModuleType, lire, env: dict, variable: str, cas: str) -> str:
    """Vérifie que `env` est refusé avec ConfigInvalide, et que le message nomme `variable` sans montrer de clé."""
    try:
        config = _lire_config(lire, env, cas)
    except m.ConfigInvalide as exc:
        message = str(exc)
    except ValueError as exc:
        raise AssertionError(f"{cas} : vous levez {type(exc).__name__}. Levez ConfigInvalide (dans modeles.py) : c'est "
                             "elle que la ligne de commande attrapera pour afficher un message propre.") from None
    else:
        raise AssertionError(f"{cas} : la configuration devait être refusée (ConfigInvalide), mais lire_config renvoie "
                             f"{config!r}. Un service mal configuré doit refuser de démarrer, pas tourner de travers.")
    if variable not in message:
        raise AssertionError(f"{cas} : le message d'erreur doit nommer la variable à corriger ({variable}). "
                             f"Message : {message!r}")
    secrets_env = [morceau.partition("=")[2].strip() for morceau in env.get("SERVICE_CLES", "").split(",")]
    fuite = next((s for s in secrets_env if len(s) >= 4 and s in message), None)
    if fuite:
        raise AssertionError(f"{cas} : votre message d'erreur CONTIENT une clé ({fuite!r}). Ce message finira dans un "
                             "journal ou une capture d'écran : nommez le conseiller (« la clé de 'alice' est trop "
                             "courte »), jamais la clé.")
    return message


def verifier_config(module: ModuleType) -> None:
    m = _modeles(module)
    lire = module.lire_config

    config = _pas_none(_lire_config(lire, {"SERVICE_CLES": f"alice={_CLE_A}"}, "une configuration minimale"),
                       "lire_config")
    if not isinstance(config, m.Config):
        raise AssertionError(f"lire_config doit renvoyer un objet Config (de modeles.py), obtenu {type(config).__name__}.")
    if config.cles != {"alice": _CLE_A}:
        raise AssertionError(f"SERVICE_CLES=\"alice={_CLE_A}\" : config.cles devrait valoir {{'alice': {_CLE_A!r}}}, "
                             f"obtenu {config.cles!r}.")
    if not isinstance(config.base, Path) or config.base != Path(m.BASE_PAR_DEFAUT):
        raise AssertionError(f"Sans SERVICE_BASE, config.base doit valoir Path({m.BASE_PAR_DEFAUT!r}) (un Path, pas un "
                             f"texte), obtenu {config.base!r}.")
    if config.taille_page != m.TAILLE_PAGE_PAR_DEFAUT:
        raise AssertionError(f"Sans SERVICE_TAILLE_PAGE, la taille de page vaut {m.TAILLE_PAGE_PAR_DEFAUT}, obtenu "
                             f"{config.taille_page!r}.")

    env = {"SERVICE_CLES": f" alice = {_CLE_A} , bruno={_CLE_B}, ", "SERVICE_BASE": " /srv/service.db ",
           "SERVICE_TAILLE_PAGE": " 50 "}
    config = _lire_config(lire, env, "une configuration complète")
    if config.cles.get("bruno") == _CLE_B.rstrip("="):
        raise AssertionError(f"La clé de bruno ({_CLE_B!r}) se termine par « == » (c'est du base64), et vous l'avez "
                             f"coupée en {config.cles['bruno']!r}. Coupez au PREMIER « = » seulement : "
                             "entree.partition(\"=\") ou entree.split(\"=\", 1).")
    if config.cles != {"alice": _CLE_A, "bruno": _CLE_B}:
        raise AssertionError(f"SERVICE_CLES={env['SERVICE_CLES']!r} : attendu {{'alice': …, 'bruno': …}} (espaces "
                             f"autour des noms et des clés retirés, virgule finale ignorée), obtenu {config.cles!r}.")
    if config.base != Path("/srv/service.db"):
        raise AssertionError(f"SERVICE_BASE=' /srv/service.db ' : config.base doit valoir Path('/srv/service.db') (sans "
                             f"les espaces), obtenu {config.base!r}.")
    if config.taille_page != 50 or isinstance(config.taille_page, bool):
        raise AssertionError(f"SERVICE_TAILLE_PAGE=' 50 ' : la taille de page doit être l'ENTIER 50, obtenu "
                             f"{config.taille_page!r}. Les variables d'environnement sont toujours du texte : int(...).")

    vides = {"SERVICE_CLES": f"alice={_CLE_A}", "SERVICE_BASE": "", "SERVICE_TAILLE_PAGE": "   "}
    config = _lire_config(lire, vides, "des variables vides")
    if config.base != Path(m.BASE_PAR_DEFAUT) or config.taille_page != m.TAILLE_PAGE_PAR_DEFAUT:
        raise AssertionError("Une variable vide (« SERVICE_BASE= » dans un fichier .env) compte comme absente : prenez "
                             f"la valeur par défaut. Obtenu base={config.base!r}, taille_page={config.taille_page!r}.")

    autre = {"SERVICE_CLES": "zoe=cle-de-zoe-0123456789", "SERVICE_TAILLE_PAGE": "7", "SERVICE_BASE": "ailleurs.db"}
    with mock.patch.dict(os.environ, autre):
        config = _lire_config(lire, {"SERVICE_CLES": f"alice={_CLE_A}"}, "os.environ différent de env")
    if "zoe" in config.cles or config.taille_page == 7 or config.base == Path("ailleurs.db"):
        raise AssertionError("lire_config lit os.environ : elle doit lire UNIQUEMENT le paramètre `env`. C'est ce qui la "
                             "rend testable avec un simple dict ; la ligne de commande lui passera os.environ.")

    cas_refuses = (
        ({}, "SERVICE_CLES", "sans SERVICE_CLES"),
        ({"SERVICE_CLES": " , "}, "SERVICE_CLES", "SERVICE_CLES sans aucun conseiller"),
        ({"SERVICE_CLES": "alice"}, "SERVICE_CLES", "SERVICE_CLES='alice' (pas de « = »)"),
        ({"SERVICE_CLES": "=cle-anonyme-0123456789"}, "SERVICE_CLES", "une clé sans nom de conseiller"),
        ({"SERVICE_CLES": "alice=secret-court"}, "SERVICE_CLES", f"une clé de 12 caractères (minimum {m.LONGUEUR_CLE_MIN})"),
        ({"SERVICE_CLES": f"alice={_CLE_A},alice=cle-alice-bis-0123456789"}, "SERVICE_CLES", "alice deux fois"),
        ({"SERVICE_CLES": f"alice={_CLE_A},bruno={_CLE_A}"}, "SERVICE_CLES",
         "deux conseillers avec la même clé (on ne saurait pas qui appelle)"),
        ({"SERVICE_CLES": f"alice={_CLE_A}", "SERVICE_TAILLE_PAGE": "vingt"}, "SERVICE_TAILLE_PAGE",
         "SERVICE_TAILLE_PAGE='vingt'"),
        ({"SERVICE_CLES": f"alice={_CLE_A}", "SERVICE_TAILLE_PAGE": "2.5"}, "SERVICE_TAILLE_PAGE",
         "SERVICE_TAILLE_PAGE='2.5' (pas un entier)"),
        ({"SERVICE_CLES": f"alice={_CLE_A}", "SERVICE_TAILLE_PAGE": "0"}, "SERVICE_TAILLE_PAGE", "SERVICE_TAILLE_PAGE=0"),
        ({"SERVICE_CLES": f"alice={_CLE_A}", "SERVICE_TAILLE_PAGE": str(m.TAILLE_PAGE_MAX + 1)}, "SERVICE_TAILLE_PAGE",
         f"SERVICE_TAILLE_PAGE={m.TAILLE_PAGE_MAX + 1} (maximum {m.TAILLE_PAGE_MAX})"),
    )
    for env, variable, cas in cas_refuses:
        _config_refusee(m, lire, env, variable, cas)

    deux = {"SERVICE_CLES": "alice=secret-court", "SERVICE_TAILLE_PAGE": "0"}
    message = _config_refusee(m, lire, deux, "SERVICE_CLES", "deux erreurs à la fois")
    if "SERVICE_TAILLE_PAGE" not in message:
        raise AssertionError("Avec DEUX erreurs (clé trop courte ET SERVICE_TAILLE_PAGE=0), le message ne cite que la "
                             f"première : {message!r}. Rassemblez les erreurs dans une liste et levez ConfigInvalide une "
                             "seule fois à la fin : l'équipe corrige tout en un seul redémarrage.")


# ============================================================================ Étape 2 : ouvrir, importer_contacts


def _ouvrir(ouvrir, chemin, cas: str) -> sqlite3.Connection:
    conn = _pas_none(ouvrir(chemin), "ouvrir")
    if not isinstance(conn, sqlite3.Connection):
        raise AssertionError(f"{cas} : ouvrir doit renvoyer la connexion (sqlite3.connect(...)), obtenu "
                             f"{type(conn).__name__}.")
    return conn


def _attribution(chemin_ou_conn) -> dict[str, str]:
    if isinstance(chemin_ou_conn, sqlite3.Connection):
        return dict(chemin_ou_conn.execute("SELECT customer_id, conseiller FROM contacts"))
    with _fermee(sqlite3.connect(chemin_ou_conn)) as conn:
        return dict(conn.execute("SELECT customer_id, conseiller FROM contacts"))


def _attribution_attendue(existants, contacts, conseillers) -> dict[str, str]:
    """Ce que donne le corrigé sur une base en mémoire qui contient déjà `existants`."""
    with _fermee(ref_base.ouvrir(":memory:")) as conn:
        with conn:
            conn.executemany("INSERT INTO contacts (customer_id, offre, conseiller, statut, version) "
                             "VALUES (?, 'APPEL', ?, ?, ?)", existants)
        ref_base.importer_contacts(conn, contacts, conseillers)
        return _attribution(conn)


def _importer(importer, conn, contacts, conseillers, cas: str):
    try:
        return importer(conn, contacts, conseillers)
    except sqlite3.OperationalError as exc:
        if "syntax" in str(exc) or "unrecognized token" in str(exc):
            raise AssertionError(f"{cas} : sqlite3.OperationalError ({exc}). Un identifiant contient une apostrophe "
                                 "(« K'07 ») : votre requête est fabriquée avec une f-string. Utilisez des « ? » et "
                                 "passez les valeurs à part.") from None
        raise
    except sqlite3.IntegrityError as exc:
        if "UNIQUE" in str(exc):
            raise AssertionError(f"{cas} : sqlite3.IntegrityError ({exc}). Un contact déjà dans la base (ou présent deux "
                                 "fois dans la liste) doit être ignoré, pas réinséré.") from None
        raise


def _comparer_attribution(obtenue: dict, attendue: dict, conseillers, cas: str) -> None:
    if obtenue == attendue:
        return
    nouveaux = [c for c in attendue if c in obtenue]
    if len(set(obtenue[c] for c in nouveaux)) == 1 and len(set(attendue.values())) > 1:
        raise AssertionError(f"{cas} : tous les contacts sont allés à {obtenue[nouveaux[0]]!r}. Chaque NOUVEAU contact va "
                             "au conseiller qui en a le MOINS à ce moment-là : tenez le compte à jour (un dict) à chaque "
                             "attribution.")
    faux = next(c for c in attendue if obtenue.get(c) != attendue[c])
    raise AssertionError(f"{cas} (conseillers : {list(conseillers)}) : {faux} devrait aller à {attendue[faux]!r}, il est "
                         f"chez {obtenue.get(faux)!r}. Règle : le conseiller qui a le moins de contacts (en comptant ceux "
                         "déjà dans la base) ; à égalité, le premier par ordre ALPHABÉTIQUE (pas l'ordre de la liste) : "
                         "min(charge, key=lambda nom: (charge[nom], nom)).")


def verifier_base(module: ModuleType) -> None:
    ouvrir, importer = module.ouvrir, module.importer_contacts
    with _dossier_temporaire() as dossier:
        chemin = dossier / "service.db"
        with _fermee(_ouvrir(ouvrir, chemin, "ouvrir(Path)")) as conn:
            tables = {nom for (nom,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            if not {"contacts", "appels"} <= tables:
                raise AssertionError(f"Après ouvrir, la base contient les tables {sorted(tables)} : exécutez SCHEMA avec "
                                     "conn.executescript(SCHEMA).")
            if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise AssertionError("Les clés étrangères ne sont pas activées (`PRAGMA foreign_keys = ON`) : le journal "
                                     "des appels accepterait un client qui n'existe pas.")
            erreurs: list[BaseException] = []
            fil = threading.Thread(target=lambda: _essayer(erreurs, conn.execute, "SELECT 1"))
            fil.start()
            fil.join()
            if erreurs and isinstance(erreurs[0], sqlite3.ProgrammingError):
                raise AssertionError("La connexion refuse de servir dans un autre fil d'exécution (thread) : "
                                     f"« {erreurs[0]} ». FastAPI ouvre souvent la connexion dans un thread et exécute la "
                                     "route dans un autre : connectez-vous avec sqlite3.connect(chemin, "
                                     "check_same_thread=False).")

            for vide in ([], ()):
                try:
                    importer(conn, [("K01", "APPEL")], vide)
                except ValueError:
                    continue
                except NotImplementedError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise AssertionError(f"importer_contacts sans aucun conseiller lève {type(exc).__name__} ({exc}) : "
                                         "levez ValueError, avec un message clair.") from None
                raise AssertionError("importer_contacts sans aucun conseiller doit lever ValueError : à qui attribuer "
                                     "les contacts ?")

            premiers = [(f"K{i:02d}", "APPEL") for i in range(1, 11)] + [("K'11", "REMISE")]
            conseillers = ["chloe", "alice", "bruno"]
            ajoutes = _importer(importer, conn, premiers, conseillers, "Premier import")
            if ajoutes != len(premiers):
                raise AssertionError(f"Premier import de {len(premiers)} contacts : renvoyez le nombre de contacts ajoutés "
                                     f"({len(premiers)}), obtenu {ajoutes!r}.")
        attendue = _attribution_attendue([], premiers, conseillers)
        obtenue = _attribution(chemin)
        if not obtenue:
            raise AssertionError("Les contacts importés ont disparu une fois la connexion fermée : il manque le COMMIT. "
                                 "Écrivez vos INSERT dans un bloc `with conn:`.")
        _comparer_attribution(obtenue, attendue, conseillers, "Premier import (11 contacts, 3 conseillers)")

        # Une semaine plus tard : alice a déjà traité K01, un nouveau conseiller arrive, la liste
        # contient les anciens contacts, des nouveaux, et un doublon.
        with _fermee(sqlite3.connect(chemin)) as autre, autre:
            autre.execute("UPDATE contacts SET statut = 'accepte', version = 2, tentatives = 1 WHERE customer_id = 'K01'")
        fiche_k01 = _fiche(_photo(chemin), "K01")
        seconds = premiers + [(f"N{i:02d}", "SUPPORT") for i in range(1, 7)] + [("N01", "APPEL")]
        conseillers = ["alice", "bruno", "chloe", "david"]
        with _fermee(_ouvrir(ouvrir, chemin, "ouvrir une base existante")) as conn:
            ajoutes = _importer(importer, conn, seconds, conseillers, "Second import (anciens + nouveaux + doublon)")
        if _fiche(_photo(chemin), "K01") != fiche_k01:
            raise AssertionError(f"Le second import a modifié K01, déjà traité par son conseiller : "
                                 f"{_fiche(_photo(chemin), 'K01')} au lieu de {fiche_k01}. Un contact déjà dans la base ne "
                                 "doit pas être touché (ni conseiller, ni statut, ni version).")
        if ajoutes != 6:
            raise AssertionError(f"Second import : 6 nouveaux contacts (N01 à N06 ; N01 est en double dans la liste), "
                                 f"obtenu {ajoutes!r}. Ne comptez que les contacts vraiment ajoutés.")
        existants = [(c, s, "a_appeler", 1) for c, s in attendue.items()]
        attendue2 = _attribution_attendue(existants, seconds, conseillers)
        _comparer_attribution(_attribution(chemin), attendue2, conseillers,
                              "Second import (david arrive avec 0 contact, les autres en ont déjà 3 ou 4)")

        # 300 contacts : combien de requêtes ?
        gros = dossier / "gros.db"
        ref_base.ouvrir(gros).close()
        with _fermee(_espion(gros)) as espion:
            _importer(importer, espion, [(f"G{i:03d}", "APPEL") for i in range(300)], ["alice", "bruno"], "300 contacts")
            if len(espion.requetes) > 6:
                exemple = max(set(espion.requetes), key=espion.requetes.count)
                raise AssertionError(f"Importer 300 contacts a demandé {len(espion.requetes)} requêtes à la base (la plus "
                                     f"répétée : {exemple[:90]!r}). Une requête par contact (le piège « N+1 ») devient "
                                     "très lent sur un vrai fichier : lisez les identifiants connus et les charges en "
                                     "1 ou 2 requêtes, travaillez en mémoire, puis insérez tout avec executemany.")

        # Tout ou rien : une ligne refusée par la base (offre manquante) au milieu de la liste.
        avant = _photo(chemin)
        fautifs = [("P01", "APPEL"), ("P02", None), ("P03", "APPEL")]
        with _fermee(_ouvrir(ouvrir, chemin, "ouvrir")) as conn:
            try:
                importer(conn, fautifs, ["alice"])
            except sqlite3.IntegrityError:
                pass
            except NotImplementedError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(f"Une ligne refusée par la base (offre manquante) : laissez remonter "
                                     f"sqlite3.IntegrityError, obtenu {type(exc).__name__}: {exc}.") from None
            else:
                raise AssertionError("Un contact sans offre (None) a été accepté ou ignoré en silence : laissez la base le "
                                     "refuser (NOT NULL) et l'erreur sqlite3.IntegrityError remonter.")
        if _photo(chemin) != avant:
            raise AssertionError(f"Une ligne refusée au milieu de l'import a laissé une partie des contacts dans la base "
                                 f"({_differences(avant, _photo(chemin))}). Tout ou rien : les INSERT dans un seul bloc "
                                 "`with conn:` (ROLLBACK automatique si une exception sort du bloc).")

        casse = _ecrire(dossier, "casse.db", "ceci n'est pas une base SQLite\n" * 40)
        try:
            ouvrir(casse).close()
        except sqlite3.DatabaseError:
            pass
        except NotImplementedError:
            raise
        else:
            raise AssertionError("ouvrir sur un fichier qui n'est pas une base SQLite doit lever sqlite3.DatabaseError "
                                 "(c'est executescript qui la lève) : ne l'attrapez pas, la ligne de commande s'en charge.")


def _essayer(erreurs: list, fonction, *args) -> None:
    try:
        fonction(*args)
    except BaseException as exc:  # noqa: BLE001
        erreurs.append(exc)


# ============================================================================ Étape 3 : lire_contact, page_contacts, avancement

#: alice a 9 contacts (insérés dans le désordre, l'un avec une apostrophe), bruno 3, chloe 2.
_CONTACTS_ETAPE3 = [
    ("A07", "APPEL", "alice", "a_appeler", 0, "", 1),
    ("B01", "REMISE", "bruno", "accepte", 1, "", 2),
    ("A02", "REMISE", "alice", "accepte", 1, "ravi", 2),
    ("A05", "SUPPORT", "alice", "a_appeler", 0, "", 1),
    ("A'01", "APPEL", "alice", "injoignable", 2, "messagerie", 3),
    ("A09", "ENGAGEMENT", "alice", "a_appeler", 0, "", 1),
    ("B02", "APPEL", "bruno", "a_appeler", 0, "", 1),
    ("A03", "APPEL", "alice", "refuse", 1, "", 2),
    ("C01", "SUPPORT", "chloe", "refuse", 1, "", 2),
    ("A01", "APPEL", "alice", "a_appeler", 0, "", 1),
    ("A08", "REMISE", "alice", "accepte", 1, "", 2),
    ("B03", "SUPPORT", "bruno", "a_appeler", 0, "", 1),
    ("A04", "APPEL", "alice", "a_appeler", 0, "", 1),
    ("C02", "APPEL", "chloe", "accepte", 1, "", 2),
]


def _ids(contacts) -> list:
    return [getattr(c, "customer_id", c) for c in contacts]


def _page(page_contacts, conn, conseiller, cas: str, **options):
    resultat = _pas_none(page_contacts(conn, conseiller, **options), "page_contacts")
    if not (isinstance(resultat, tuple) and len(resultat) == 2):
        raise AssertionError(f"{cas} : page_contacts doit renvoyer un couple (liste de Contact, suivant), obtenu "
                             f"{type(resultat).__name__}.")
    contacts, suivant = resultat
    return list(contacts), suivant


def _verifier_lire_contact(m: ModuleType, lire_contact, conn) -> None:
    contact = _pas_none(lire_contact(conn, "alice", "A'01"), "lire_contact")
    attendu = m.Contact(*_CONTACTS_ETAPE3[4])
    if not isinstance(contact, m.Contact):
        raise AssertionError(f"lire_contact doit renvoyer un Contact (modeles.py), obtenu {type(contact).__name__}. "
                             "Avec `SELECT {COLONNES_CONTACT} …`, Contact(*ligne) suffit.")
    if contact != attendu:
        raise AssertionError(f"lire_contact(conn, 'alice', \"A'01\") : attendu {attendu}, obtenu {contact}. Sélectionnez "
                             "les colonnes dans l'ordre de COLONNES_CONTACT.")
    for conseiller, customer_id, erreur, pourquoi in (
            ("alice", "Z99", m.Introuvable, "aucun contact n'a cet identifiant"),
            ("alice", "x' OR '1'='1", m.Introuvable, "un identifiant fabriqué par un attaquant (injection SQL)"),
            ("alice", "B01", m.PasVotreClient, "B01 est suivi par bruno"),
            ("bruno", "A02", m.PasVotreClient, "A02 est suivi par alice")):
        cas = f"lire_contact(conn, {conseiller!r}, {customer_id!r})"
        try:
            obtenu = lire_contact(conn, conseiller, customer_id)
        except erreur:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{cas} ({pourquoi}) : attendu {erreur.__name__}, obtenu {type(exc).__name__}: "
                                 f"{exc}.") from None
        raise AssertionError(f"{cas} ({pourquoi}) : attendu l'exception {erreur.__name__}, mais la fonction renvoie "
                             f"{obtenu!r}.")


def _verifier_pages(page_contacts, conn) -> None:
    tous_alice = sorted(c[0] for c in _CONTACTS_ETAPE3 if c[2] == "alice")  # 9 contacts

    contacts, suivant = _page(page_contacts, conn, "alice", "page_contacts(conn, 'alice', taille=4)", taille=4)
    if any(not hasattr(c, "customer_id") for c in contacts):
        raise AssertionError("page_contacts doit renvoyer des objets Contact (Contact(*ligne)), pas des tuples.")
    if any(c.conseiller != "alice" for c in contacts):
        raise AssertionError("page_contacts(conn, 'alice', …) renvoie des contacts d'autres conseillers : filtrez avec "
                             "`WHERE conseiller = ?`.")
    if _ids(contacts) != tous_alice[:4]:
        conseil = (" Sans ORDER BY, SQLite rend les lignes dans l'ordre où il les trouve : ajoutez ORDER BY customer_id."
                   if sorted(_ids(contacts)) != _ids(contacts) else "")
        raise AssertionError(f"Première page d'alice (taille=4) : attendu {tous_alice[:4]}, obtenu {_ids(contacts)}.{conseil}")
    if suivant != tous_alice[3]:
        raise AssertionError(f"Première page d'alice (taille=4) : il reste des contacts, `suivant` doit valoir le dernier "
                             f"identifiant de la page ({tous_alice[3]!r}), obtenu {suivant!r}.")

    vus, apres, pages = [], None, 0
    while True:
        pages += 1
        if pages > 5:
            raise AssertionError("En suivant les curseurs (taille=3), les pages ne s'arrêtent jamais : la dernière page doit "
                                 "renvoyer suivant=None.")
        contacts, apres = _page(page_contacts, conn, "alice", "pages de 3", taille=3, apres=apres)
        vus += _ids(contacts)
        if apres is None:
            break
    if vus != tous_alice:
        manquants, doubles = sorted(set(tous_alice) - set(vus)), sorted({c for c in vus if vus.count(c) > 1})
        raise AssertionError(f"En suivant les curseurs (taille=3), on obtient {vus} au lieu de {tous_alice} "
                             f"(manquants : {manquants}, en double : {doubles}). La page suivante commence STRICTEMENT "
                             "après le curseur : `customer_id > ?`.")
    if pages != 3:
        raise AssertionError(f"Alice a 9 contacts : avec taille=3, il faut exactement 3 pages, vous en renvoyez {pages}. La "
                             "dernière page est pleine, mais il n'y a rien après : `suivant` doit être None. Astuce : "
                             "demandez taille + 1 lignes ; la ligne en trop, si elle existe, prouve qu'il y a une suite.")

    contacts, suivant = _page(page_contacts, conn, "alice", "page_contacts(conn, 'alice') sans taille")
    if _ids(contacts) != tous_alice or suivant is not None:
        raise AssertionError(f"Sans taille, la page fait 20 contacts : toute la liste d'alice (9), suivant=None. Obtenu "
                             f"{_ids(contacts)}, suivant={suivant!r}.")

    a_appeler = sorted(c[0] for c in _CONTACTS_ETAPE3 if c[2] == "alice" and c[3] == "a_appeler")  # 5
    contacts, suivant = _page(page_contacts, conn, "alice", "statut='a_appeler', taille=2", statut="a_appeler", taille=2)
    if _ids(contacts) != a_appeler[:2]:
        raise AssertionError(f"page_contacts(conn, 'alice', statut='a_appeler', taille=2) : attendu {a_appeler[:2]}, "
                             f"obtenu {_ids(contacts)}. Le statut s'ajoute aux conditions : `AND statut = ?`.")
    contacts, _ = _page(page_contacts, conn, "alice", "statut + curseur", statut="a_appeler", taille=2, apres=suivant)
    if _ids(contacts) != a_appeler[2:4]:
        raise AssertionError(f"Page 2 des contacts « a_appeler » d'alice (apres={suivant!r}) : attendu {a_appeler[2:4]}, "
                             f"obtenu {_ids(contacts)}. Statut ET curseur doivent s'appliquer ensemble.")

    contacts, _ = _page(page_contacts, conn, "alice' OR '1'='1", "un conseiller fabriqué par un attaquant")
    if contacts:
        raise AssertionError("page_contacts(conn, \"alice' OR '1'='1\") renvoie des contacts : votre requête colle le nom "
                             "du conseiller dans le texte SQL (injection). Des « ? » et une liste de paramètres.")

    for options, pourquoi in (({"statut": "perdu"}, "un statut inconnu"), ({"taille": 0}, "taille=0"),
                              ({"taille": -5}, "taille=-5")):
        try:
            page_contacts(conn, "alice", **options)
        except ValueError:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"page_contacts avec {pourquoi} lève {type(exc).__name__} : levez ValueError.") from None
        raise AssertionError(f"page_contacts avec {pourquoi} doit lever ValueError (l'API, elle, répondra 422 avant).")


def verifier_contacts(module: ModuleType) -> None:
    m = _modeles(module)
    with _dossier_temporaire() as dossier:
        chemin = dossier / "service.db"
        _creer_base(chemin, _CONTACTS_ETAPE3)
        with _fermee(_espion(chemin)) as conn:
            _verifier_lire_contact(m, module.lire_contact, conn)
            _verifier_pages(module.page_contacts, conn)

            conn.requetes.clear()
            _page(module.page_contacts, conn, "alice", "une page", taille=5)
            if len(conn.requetes) > 1:
                raise AssertionError(f"Une page de contacts a demandé {len(conn.requetes)} requêtes : une seule suffit "
                                     "(SELECT … WHERE … ORDER BY … LIMIT ?).")
            if conn.requetes and "ORDER BY" not in conn.requetes[-1].upper():
                raise AssertionError("Votre requête de page n'a pas d'ORDER BY. Les contacts arrivent dans l'ordre aujourd'hui "
                                     "grâce à l'index de la table, par chance : SQL ne promet AUCUN ordre sans ORDER BY, "
                                     "et la pagination par curseur repose entièrement sur cet ordre.")

            with _fermee(sqlite3.connect(chemin)) as autre:
                attendu = ref_contacts.avancement(autre)
            conn.requetes.clear()
            obtenu = _pas_none(module.avancement(conn), "avancement")
            if len(conn.requetes) > 1:
                raise AssertionError(f"avancement a demandé {len(conn.requetes)} requêtes pour 3 conseillers : avec 50 "
                                     "conseillers, ce serait 50. Une seule requête : GROUP BY conseiller, statut.")
            if not isinstance(obtenu, dict) or obtenu != attendu:
                manque = next((c for c in attendu if c in (obtenu or {}) and set(obtenu[c]) != set(STATUTS)), None)
                conseil = (f" Chaque conseiller a TOUS les statuts, même à 0 (pour {manque} : {obtenu[manque]}). Un statut "
                           "sans contact n'apparaît pas dans le GROUP BY : partez de dict.fromkeys(STATUTS, 0)."
                           if manque else "")
                raise AssertionError(f"avancement : attendu {attendu}, obtenu {obtenu!r}.{conseil}")
            if list(obtenu) != sorted(obtenu):
                raise AssertionError(f"avancement : les conseillers doivent être dans l'ordre alphabétique, obtenu "
                                     f"{list(obtenu)}.")

        vide = dossier / "vide.db"
        ref_base.ouvrir(vide).close()
        with _fermee(sqlite3.connect(vide)) as conn:
            if module.avancement(conn) != {}:
                raise AssertionError(f"avancement d'une base sans contact : attendu {{}}, obtenu {module.avancement(conn)!r}.")


# ============================================================================ Étape 4 : enregistrer_appel


def _remplir_etape4(chemin: Path) -> None:
    _creer_base(chemin, [
        ("C1", "APPEL", "alice", "a_appeler", 0, "", 1),
        ("C2", "REMISE", "alice", "a_appeler", 0, "", 1),
        ("C3", "SUPPORT", "bruno", "a_appeler", 0, "", 1),
    ])


def _appeler(enregistrer, chemin: Path, args: tuple, kwargs: dict | None = None, avant=None):
    """Appelle votre fonction sur une connexion espionne NEUVE ; renvoie (résultat, exception, espion)."""
    conn = _espion(chemin)
    conn.avant = avant
    try:
        return enregistrer(conn, *args, **(kwargs or {})), None, conn
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        return None, exc, conn
    finally:
        if conn.in_transaction:
            conn.rollback()  # une transaction laissée ouverte : on l'annule, et on vous le dira
            conn.requetes.append("<transaction laissée ouverte>")
        conn.close()


def _appel_texte(args: tuple, kwargs: dict | None = None) -> str:
    morceaux = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in (kwargs or {}).items()]
    return f"enregistrer_appel(conn, {', '.join(morceaux)})"


def _transaction_ouverte(conn: _Espion, quand: str) -> None:
    if "<transaction laissée ouverte>" in conn.requetes:
        raise AssertionError(f"{quand} : votre fonction laisse une transaction ouverte (ni commit, ni rollback). Les "
                             "collègues ne voient rien, et la base reste verrouillée : écrivez dans un bloc `with conn:`.")


def _appel_refuse(m, enregistrer, chemin, args, attendue, pourquoi: str, conseil: str = "") -> BaseException:
    quand = _appel_texte(args)
    avant = _photo(chemin)
    resultat, erreur, conn = _appeler(enregistrer, chemin, args)
    apres = _photo(chemin)
    if erreur is None:
        change = f" Et la base a changé : {_differences(avant, apres)}." if apres != avant else ""
        raise AssertionError(f"{quand} : {pourquoi} ; votre fonction devait lever {attendue.__name__}, mais renvoie "
                             f"{resultat!r}.{change}{conseil}")
    if not isinstance(erreur, attendue):
        raise AssertionError(f"{quand} : {pourquoi} ; attendu {attendue.__name__}, obtenu {type(erreur).__name__}: "
                             f"{erreur}{_lieu(erreur, sys.modules[enregistrer.__module__])}.{conseil}")
    _transaction_ouverte(conn, quand)
    if apres != avant:
        raise AssertionError(f"{quand} : {attendue.__name__} est bien levée, mais la base a changé "
                             f"({_differences(avant, apres)}). Un enregistrement refusé ne modifie RIEN : la condition "
                             "(conseiller ET version) doit être DANS l'UPDATE, pas vérifiée après coup.")
    return erreur


def _appel_reussi(m, enregistrer, chemin, args, kwargs, attendu_fiche: tuple, attendu_journal: tuple) -> None:
    quand = _appel_texte(args, kwargs)
    avant = _photo(chemin)
    resultat, erreur, conn = _appeler(enregistrer, chemin, args, kwargs)
    if erreur is not None:
        raise AssertionError(f"{quand} devrait réussir, mais lève {type(erreur).__name__}: {erreur}"
                             f"{_lieu(erreur, sys.modules[enregistrer.__module__])}.")
    _transaction_ouverte(conn, quand)
    apres = _photo(chemin)
    fiche = _fiche(apres, args[1])
    if fiche != attendu_fiche:
        indice = ""
        if fiche and fiche[6] == attendu_fiche[6] - 1:
            indice = " La version doit augmenter de 1 à chaque enregistrement (`version = version + 1`)."
        elif fiche and fiche[4] == attendu_fiche[4] - 1:
            indice = " Le nombre de tentatives doit augmenter de 1 à chaque appel (`tentatives = tentatives + 1`)."
        elif fiche == _fiche(avant, args[1]):
            indice = " Rien n'a été enregistré (ou pas validé : `with conn:`)."
        raise AssertionError(f"{quand} : la fiche vaut {fiche} au lieu de {attendu_fiche} (customer_id, offre, conseiller, "
                             f"statut, tentatives, commentaire, version).{indice}")
    autres = [ligne for ligne in apres["contacts"] if ligne[0] != args[1]]
    if autres != [ligne for ligne in avant["contacts"] if ligne[0] != args[1]]:
        raise AssertionError(f"{quand} : d'AUTRES fiches ont changé. Votre UPDATE a-t-il un `WHERE customer_id = ?` ?")
    nouveaux = apres["appels"][len(avant["appels"]):]
    if nouveaux != [attendu_journal]:
        raise AssertionError(f"{quand} : le journal (table appels) devrait recevoir UNE ligne {attendu_journal} "
                             f"(customer_id, conseiller, statut, commentaire), il a reçu {nouveaux}.")
    if not isinstance(resultat, m.Contact) or tuple(getattr(resultat, f.name) for f in fields(m.Contact)) != fiche:
        raise AssertionError(f"{quand} doit renvoyer la fiche À JOUR (un Contact en version {attendu_fiche[6]}), obtenu "
                             f"{resultat!r}. Relisez-la après l'écriture (lire_contact).")


def verifier_appels(module: ModuleType) -> None:
    m = _modeles(module)
    enregistrer = module.enregistrer_appel
    with _dossier_temporaire() as dossier:
        chemin = dossier / "service.db"
        _remplir_etape4(chemin)

        _appel_reussi(m, enregistrer, chemin, ("alice", "C1", "injoignable", 1), {"commentaire": "messagerie"},
                      ("C1", "APPEL", "alice", "injoignable", 1, "messagerie", 2),
                      ("C1", "alice", "injoignable", "messagerie"))
        _appel_reussi(m, enregistrer, chemin, ("alice", "C1", "accepte", 2), {},
                      ("C1", "APPEL", "alice", "accepte", 2, "", 3), ("C1", "alice", "accepte", ""))

        erreur = _appel_refuse(m, enregistrer, chemin, ("alice", "C1", "refuse", 2), m.Conflit,
                               "C1 est passé en version 3 depuis la lecture (version 2)",
                               " Enregistrez seulement si la version n'a pas changé : `… AND version = ?` dans l'UPDATE, "
                               "puis regardez curseur.rowcount (0 = rien n'a été modifié).")
        if getattr(erreur, "version_actuelle", None) != 3:
            raise AssertionError(f"Le Conflit doit porter la version ACTUELLE de la fiche (3), pour que le conseiller "
                                 f"puisse recharger : Conflit(customer_id, version_actuelle). Obtenu "
                                 f"{getattr(erreur, 'version_actuelle', None)!r}.")
        _appel_refuse(m, enregistrer, chemin, ("alice", "C9", "accepte", 1), m.Introuvable,
                      "aucun contact C9 (ce n'est pas un conflit : il n'y a rien à recharger)",
                      " Quand l'UPDATE ne modifie rien, relisez la fiche (lire_contact, étape 3) pour savoir pourquoi.")
        _appel_refuse(m, enregistrer, chemin, ("alice", "C3", "accepte", 1), m.PasVotreClient,
                      "C3 est suivi par bruno, pas par alice",
                      " Mettez `AND conseiller = ?` DANS l'UPDATE : la fiche de bruno ne doit jamais être modifiée, "
                      "même une milliseconde.")
        _appel_refuse(m, enregistrer, chemin, ("alice", "C1' OR '1'='1", "refuse", 3), m.Introuvable,
                      "aucun client n'a cet identifiant (il vient d'une adresse web : entrée non fiable)",
                      " Collé dans le SQL, il devient `WHERE customer_id = 'C1' OR '1'='1' …` : des « ? » partout.")
        for statut in ("a_appeler", "perdu"):
            _appel_refuse(m, enregistrer, chemin, ("alice", "C2", statut, 1), ValueError,
                          f"« {statut} » n'est pas un résultat d'appel ({', '.join(m.RESULTATS)})",
                          " `if statut not in RESULTATS: raise ValueError(…)` AVANT de toucher à la base.")

        # La course : alice a aussi la fiche C2 ouverte sur son téléphone, et y enregistre « refuse »
        # entre votre lecture et votre écriture.
        etat = {"a_ecrit": False, "bloque": False}

        def telephone(espion: _Espion, sql: str) -> None:
            texte = " ".join(str(sql).split()).upper()
            if etat["a_ecrit"] or etat["bloque"] or not texte.startswith("UPDATE") or "CONTACTS" not in texte:
                return
            if not any(r.upper().startswith("SELECT") and "CONTACTS" in r.upper() for r in espion.requetes[:-1]):
                return  # rien lu avant d'écrire : pas de fenêtre où se glisser
            collegue = sqlite3.connect(chemin, timeout=0)
            try:
                with collegue:
                    collegue.execute("UPDATE contacts SET statut = 'refuse', version = version + 1, "
                                     "tentatives = tentatives + 1 WHERE customer_id = 'C2'")
                etat["a_ecrit"] = True
            except sqlite3.OperationalError:
                etat["bloque"] = True  # vous avez verrouillé la base : l'autre écriture attendra son tour
            finally:
                collegue.close()

        args = ("alice", "C2", "accepte", 1)
        quand = _appel_texte(args)
        resultat, erreur, conn = _appeler(enregistrer, chemin, args, avant=telephone)
        if etat["a_ecrit"]:
            if erreur is None:
                raise AssertionError(
                    f"{quand} : votre code a LU la fiche (version 1), puis alice a enregistré « refuse » depuis son "
                    "téléphone (version 2) juste avant votre UPDATE… que votre code a fait quand même : "
                    f"{_fiche(_photo(chemin), 'C2')}. Un appel est perdu sans que personne le sache. Comparer la version "
                    "en Python arrive trop tôt : la condition doit être DANS l'UPDATE (`AND version = ?`), et "
                    "curseur.rowcount dit si l'écriture a eu lieu.")
            if not isinstance(erreur, m.Conflit):
                raise AssertionError(f"{quand} : la fiche a changé entre votre lecture et votre écriture ; attendu "
                                     f"Conflit, obtenu {type(erreur).__name__}: {erreur}.")

        # Disque plein au moment d'écrire le journal : la fiche ne doit pas changer non plus.
        _remplir_panne(chemin)
        avant = _photo(chemin)
        cible = "C1"
        version = _fiche(avant, cible)[6]
        args = ("alice", cible, "refuse", version)
        resultat, erreur, conn = _appeler(enregistrer, chemin, args)
        apres = _photo(chemin)
        if erreur is None:
            raise AssertionError(f"{_appel_texte(args)} : l'écriture dans le journal a échoué (disque plein simulé), mais "
                                 "votre fonction n'a rien signalé. Laissez l'erreur sqlite3 remonter.")
        if _fiche(apres, cible) != _fiche(avant, cible):
            raise AssertionError(f"{_appel_texte(args)} : l'écriture dans le journal a échoué (disque plein simulé), mais la "
                                 f"fiche a quand même changé : {_fiche(apres, cible)}. Fiche et journal dans la MÊME "
                                 "transaction (un seul bloc `with conn:`) : les deux, ou aucun.")


def _remplir_panne(chemin: Path) -> None:
    with _fermee(sqlite3.connect(chemin)) as conn:
        conn.execute("CREATE TRIGGER disque_plein BEFORE INSERT ON appels BEGIN SELECT RAISE(ABORT, 'disque plein'); END")
        conn.commit()


# ============================================================================ Étape 5 : creer_app

_CLES_API = {"alice": "cle-alice-0123456789", "bruno": "cle-bruno-0123456789"}
_ALICE = {"X-Cle-API": _CLES_API["alice"]}
_BRUNO = {"X-Cle-API": _CLES_API["bruno"]}
_CONTACTS_API = [(f"A{i}", "APPEL", "alice", "a_appeler", 0, "", 1) for i in range(1, 6)] + \
                [("B1", "REMISE", "bruno", "a_appeler", 0, "", 1), ("B2", "APPEL", "bruno", "a_appeler", 0, "", 1)]


class _ConnexionSuivie(sqlite3.Connection):
    ouvertes: set = set()
    creees: int = 0

    def close(self) -> None:
        _ConnexionSuivie.ouvertes.discard(id(self))
        super().close()


def _ouvrir_suivie(chemin) -> sqlite3.Connection:
    conn = sqlite3.connect(chemin, factory=_ConnexionSuivie, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    _ConnexionSuivie.ouvertes.add(id(conn))
    _ConnexionSuivie.creees += 1
    return conn


class _Client:
    """Un client HTTP de test qui transforme un plantage de votre API en message clair."""

    def __init__(self, app, module: ModuleType):
        from fastapi.testclient import TestClient

        self.http = TestClient(app)
        self.module = module

    def __call__(self, methode: str, url: str, attendu: int | None, quoi: str, conseil: str = "", **options):
        try:
            reponse = self.http.request(methode, url, **options)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001 — une exception non traduite = une erreur 500
            raise AssertionError(f"`{methode} {url}` ({quoi}) : votre API plante — {type(exc).__name__}: {exc}"
                                 f"{_lieu(exc, self.module)}. Le client recevrait une erreur 500.{conseil}") from None
        if attendu is not None and reponse.status_code != attendu:
            raise AssertionError(f"`{methode} {url}` ({quoi}) : code {reponse.status_code}, attendu {attendu}. "
                                 f"Réponse : {reponse.text[:200]}")
        return reponse


def _json(reponse, cas: str):
    try:
        return reponse.json()
    except ValueError:
        raise AssertionError(f"{cas} : la réponse n'est pas du JSON ({reponse.text[:100]!r}).") from None


def _verifier_routes(m: ModuleType, appel: _Client, chemin: Path) -> None:
    sante = appel("GET", "/sante", 200, "sans clé : un robot de surveillance vérifie que le service répond")
    if _json(sante, "GET /sante") != {"statut": "ok"}:
        raise AssertionError(f"GET /sante doit répondre {{\"statut\": \"ok\"}}, obtenu {sante.text[:100]}.")

    appel("GET", "/contacts", 401, "sans en-tête X-Cle-API")
    for cle, quoi in (("mauvaise-cle-0123456789", "une clé inconnue"),
                      (_CLES_API["alice"][:-1], "la clé d'alice SANS son dernier caractère"),
                      (_CLES_API["alice"] + "0", "la clé d'alice avec un caractère EN PLUS"),
                      ("cle", "un morceau de clé")):
        reponse = appel("GET", "/contacts", None, quoi, headers={"X-Cle-API": cle})
        if reponse.status_code != 401:
            raise AssertionError(f"`GET /contacts` avec {quoi} : code {reponse.status_code}, attendu 401. Comparez la clé "
                                 "reçue à chaque clé connue EN ENTIER (pas startswith, pas `in`), avec "
                                 "secrets.compare_digest.")
    octets = (" secrets.compare_digest plante (TypeError) sur des textes non ASCII : comparez des octets, "
              "`compare_digest(recue.encode(\"utf-8\"), cle.encode(\"utf-8\"))`.")
    reponse = appel("GET", "/contacts", None, "une clé avec un accent", conseil=octets,
                    headers={"X-Cle-API": "clé-alice-0123456789".encode("utf-8")})
    if reponse.status_code != 401:
        raise AssertionError(f"`GET /contacts` avec une clé accentuée : code {reponse.status_code}, attendu 401.{octets}")

    page = _json(appel("GET", "/contacts", 200, "la clé d'alice", headers=_ALICE), "GET /contacts")
    if not isinstance(page, dict) or set(page) != {"contacts", "suivant"}:
        raise AssertionError(f"GET /contacts doit répondre {{\"contacts\": [...], \"suivant\": ...}}, obtenu "
                             f"{str(page)[:200]}.")
    attendus = [asdict(m.Contact(*c)) for c in _CONTACTS_API[:2]]
    if page["contacts"] != attendus or page["suivant"] != "A2":
        raise AssertionError(f"GET /contacts (alice, SERVICE_TAILLE_PAGE=2) : attendu les fiches A1 et A2 complètes "
                             f"(dataclasses.asdict) et suivant='A2', obtenu {str(page)[:300]}. La taille par défaut vient "
                             "de config.taille_page.")
    page = _json(appel("GET", "/contacts?apres=A4", 200, "la dernière page", headers=_ALICE), "GET /contacts?apres=A4")
    if _ids(c["customer_id"] for c in page["contacts"]) != ["A5"] or page["suivant"] is not None:
        raise AssertionError(f"GET /contacts?apres=A4 (alice) : attendu A5 seul et suivant=null, obtenu {str(page)[:200]}.")
    page = _json(appel("GET", "/contacts?taille=10", 200, "taille=10", headers=_BRUNO), "GET /contacts (bruno)")
    if [c["customer_id"] for c in page["contacts"]] != ["B1", "B2"]:
        raise AssertionError(f"GET /contacts?taille=10 avec la clé de bruno : attendu B1 et B2 (ses contacts à lui), "
                             f"obtenu {[c.get('customer_id') for c in page['contacts']]}.")
    for url, quoi in (("/contacts?taille=0", "taille=0"), (f"/contacts?taille={m.TAILLE_PAGE_MAX + 1}", "taille trop grande"),
                      ("/contacts?taille=dix", "taille=dix"), ("/contacts?statut=perdu", "un statut inconnu")):
        reponse = appel("GET", url, None, quoi, headers=_ALICE)
        if reponse.status_code != 422:
            raise AssertionError(f"`GET {url}` : code {reponse.status_code}, attendu 422. Laissez FastAPI valider : "
                                 f"`taille: int = Query(default=…, ge=1, le={m.TAILLE_PAGE_MAX})` et un statut de type "
                                 "Literal[...] (les valeurs de STATUTS).")

    fiche = _json(appel("GET", "/contacts/A1", 200, "une fiche d'alice", headers=_ALICE), "GET /contacts/A1")
    if fiche != asdict(m.Contact(*_CONTACTS_API[0])):
        raise AssertionError(f"GET /contacts/A1 : attendu {asdict(m.Contact(*_CONTACTS_API[0]))}, obtenu {fiche}.")
    appel("GET", "/contacts/B1", 403, "alice demande une fiche de bruno", headers=_ALICE)
    appel("GET", "/contacts/Z9", 404, "un client qui n'existe pas", headers=_ALICE)
    appel("GET", "/contacts/A1", 401, "sans clé")

    url = "/contacts/A3/appel"
    corps = {"statut": "accepte", "version": 1, "commentaire": "passe à l'annuel"}
    fiche = _json(appel("PUT", url, 200, "enregistrer un appel", headers=_ALICE, json=corps), f"PUT {url}")
    if fiche.get("statut") != "accepte" or fiche.get("version") != 2 or fiche.get("commentaire") != "passe à l'annuel":
        raise AssertionError(f"`PUT {url}` : la réponse doit être la fiche À JOUR (statut accepte, version 2), obtenu "
                             f"{fiche}.")
    if _fiche(_photo(chemin), "A3")[3] != "accepte":
        raise AssertionError(f"`PUT {url}` répond 200, mais la base n'a pas changé : appelez enregistrer_appel (étape 4).")
    conflit = _json(appel("PUT", url, 409, "la même version une seconde fois (périmée)", headers=_ALICE,
                          json={"statut": "refuse", "version": 1}), f"PUT {url}")
    if conflit.get("version_actuelle") != 2:
        raise AssertionError(f"`PUT {url}` en conflit : la réponse 409 doit contenir \"version_actuelle\": 2 pour que "
                             f"l'application du conseiller recharge la fiche. Obtenu {conflit}.")
    appel("PUT", "/contacts/B1/appel", 403, "alice enregistre un appel sur une fiche de bruno", headers=_ALICE,
          json={"statut": "accepte", "version": 1})
    appel("PUT", "/contacts/Z9/appel", 404, "un client qui n'existe pas", headers=_ALICE,
          json={"statut": "accepte", "version": 1})
    appel("PUT", "/contacts/A4/appel", 401, "sans clé", json={"statut": "accepte", "version": 1})
    for corps, quoi in (({"statut": "a_appeler", "version": 1}, "statut a_appeler"), ({"statut": "accepte"}, "sans version"),
                        ({"statut": "accepte", "version": 1, "commentaire": "x" * 501}, "un commentaire de 501 caractères")):
        reponse = appel("PUT", "/contacts/A4/appel", None, quoi, headers=_ALICE, json=corps)
        if reponse.status_code != 422:
            raise AssertionError(f"`PUT /contacts/A4/appel` avec {quoi} : code {reponse.status_code}, attendu 422. "
                                 "Déclarez le corps avec le modèle ResultatAppel (modeles.py) : FastAPI le valide pour vous.")
    if _fiche(_photo(chemin), "B1")[3] != "a_appeler" or _fiche(_photo(chemin), "A4")[3] != "a_appeler":
        raise AssertionError("Une requête refusée (403, 401 ou 422) a quand même modifié une fiche.")

    page = _json(appel("GET", "/contacts?statut=accepte", 200, "filtre par statut", headers=_ALICE), "?statut=accepte")
    if [c["customer_id"] for c in page["contacts"]] != ["A3"]:
        raise AssertionError(f"GET /contacts?statut=accepte (alice) : attendu A3 seul, obtenu "
                             f"{[c.get('customer_id') for c in page['contacts']]}. Passez `statut` à page_contacts.")

    appel("GET", "/avancement", 401, "sans clé")
    avancement = _json(appel("GET", "/avancement", 200, "l'avancement de l'équipe", headers=_BRUNO), "GET /avancement")
    with _fermee(ref_base.ouvrir(chemin)) as conn:
        attendu = ref_contacts.avancement(conn)
    if avancement != attendu:
        raise AssertionError(f"GET /avancement : attendu {attendu}, obtenu {avancement}.")


def verifier_api(module: ModuleType) -> None:
    m = _modeles(module)
    base_eleve = importlib.import_module(f"{module.__package__}.base")
    with _dossier_temporaire() as dossier:
        chemin = dossier / "service.db"
        _creer_base(chemin, _CONTACTS_API)
        config = m.Config(base=chemin, cles=dict(_CLES_API), taille_page=2)

        _ConnexionSuivie.ouvertes, _ConnexionSuivie.creees = set(), 0
        with _journal_capte() as messages, contextlib.ExitStack() as pile:
            pile.enter_context(mock.patch.object(base_eleve, "ouvrir", _ouvrir_suivie))
            if hasattr(module, "ouvrir"):
                pile.enter_context(mock.patch.object(module, "ouvrir", _ouvrir_suivie))
            app = _pas_none(module.creer_app(config), "creer_app")
            if not hasattr(app, "router"):
                raise AssertionError(f"creer_app doit renvoyer l'application FastAPI, obtenu {type(app).__name__}.")
            _verifier_routes(m, _Client(app, module), chemin)

        if _ConnexionSuivie.creees and _ConnexionSuivie.ouvertes:
            if _ConnexionSuivie.creees == 1:
                raise AssertionError("Votre API ouvre UNE connexion pour toutes les requêtes, et ne la ferme jamais. Deux "
                                     "requêtes simultanées mélangeraient leurs transactions : ouvrez une connexion PAR "
                                     "requête dans une dépendance (`def connexion(): conn = ouvrir(...); try: yield conn; "
                                     "finally: conn.close()`).")
            raise AssertionError(f"{len(_ConnexionSuivie.ouvertes)} connexion(s) sur {_ConnexionSuivie.creees} ne sont jamais "
                                 "fermées : avec des milliers de requêtes, le service finira par refuser d'ouvrir la base. "
                                 "`try: yield conn` puis `finally: conn.close()` dans la dépendance.")
        _verifier_journal(messages, (*_CLES_API.values(), "mauvaise-cle-0123456789"))
        appelle = {getattr(noeud.func, "attr", getattr(noeud.func, "id", None))
                   for noeud in _noeuds(module) if type(noeud).__name__ == "Call"}
        if "compare_digest" not in appelle:
            raise AssertionError("Vos clés sont comparées sans secrets.compare_digest. `==` s'arrête au premier caractère "
                                 "différent : en chronométrant des milliers d'essais, un attaquant devine la clé caractère "
                                 "par caractère. compare_digest prend toujours le même temps.")


class _Capteur(logging.Handler):
    def __init__(self, messages: list[str]):
        super().__init__(logging.DEBUG)
        self.messages = messages

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # noqa: BLE001
            self.messages.append(str(record.msg))


@contextlib.contextmanager
def _journal_capte():
    """Capte tout ce que votre code écrit dans le journal (`logging`), au lieu de l'afficher."""
    messages: list[str] = []
    capteur, racine = _Capteur(messages), logging.getLogger()
    niveau = racine.level
    racine.addHandler(capteur)
    racine.setLevel(logging.DEBUG)
    try:
        yield messages
    finally:
        racine.removeHandler(capteur)
        racine.setLevel(niveau)


def _verifier_journal(messages: list[str], cles) -> None:
    fuite = next((msg for msg in messages for cle in cles if cle in msg), None)
    if fuite:
        raise AssertionError(f"Votre journal contient une clé d'API : {fuite[:150]!r}. Un journal est lu par bien plus "
                             "de monde qu'un coffre à secrets : écrivez le NOM du conseiller, jamais sa clé (ni la clé "
                             "fausse reçue, qui est peut-être la vraie clé d'un collègue mal recopiée).")


def _noeuds(module: ModuleType):
    import ast

    try:
        return list(ast.walk(ast.parse(inspect.getsource(module))))
    except (OSError, TypeError, SyntaxError):
        return []


# ============================================================================ Étape 6 : main (cli)

_CLES_CLI = {"alice": "cle-alice-0123456789", "bruno": "cle-bruno-0123456789", "chloe": "cle-chloe-0123456789"}


def _env_cli(base: Path, **autres) -> dict:
    return {"SERVICE_CLES": ",".join(f"{n}={c}" for n, c in _CLES_CLI.items()), "SERVICE_BASE": str(base), **autres}


def _commande(argv: list[str]) -> str:
    return "python -m projets.service " + " ".join(argv)


class _Lanceur:
    def __init__(self):
        self.appels: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.appels.append((args, kwargs))


def _uvicorn_interdit(*_, **__):
    raise AssertionError("main a appelé uvicorn.run directement : le serveur aurait démarré pour de bon (et le correcteur "
                         "serait resté bloqué). Utilisez le paramètre `lancer` quand il est fourni, et uvicorn.run "
                         "seulement quand lancer vaut None.")


def _main(main, argv: list[str], env: dict | None, quoi: str, lancer=None) -> tuple[int, str, str]:
    import uvicorn

    try:
        with mock.patch.object(uvicorn, "run", _uvicorn_interdit):
            if env is None:
                return _lancer(lambda a: main(a, lancer=lancer) if lancer else main(a), argv)
            return _lancer(lambda a: main(a, env=env, lancer=lancer), argv)
    except NotImplementedError:
        raise
    except TypeError as exc:
        if "unexpected keyword" in str(exc) or "positional argument" in str(exc):
            raise AssertionError(f"main ne reçoit pas les bons paramètres ({exc}) : sa signature est "
                                 "`main(argv=None, env=None, lancer=None)`.") from None
        raise
    except AssertionError:
        raise
    except (OSError, ValueError, sqlite3.Error) as exc:
        nom = f"sqlite3.{type(exc).__name__}" if isinstance(exc, sqlite3.Error) else type(exc).__name__
        raise AssertionError(f"`{_commande(argv)}` ({quoi}) : {nom} non attrapée ({exc}), l'utilisateur voit une "
                             "traceback. Attrapez-la, affichez un message clair sur sys.stderr et `return 1`.") from None


def _sans_secret(argv, out: str, err: str) -> None:
    for nom, cle in _CLES_CLI.items():
        if cle in out or cle in err:
            raise AssertionError(f"`{_commande(argv)}` affiche la clé de {nom} : un terminal se partage, se filme, s'enregistre "
                                 "dans les journaux du serveur. Affichez le nom des conseillers, jamais leur clé.")


def verifier_cli(module: ModuleType) -> None:
    with _journal_capte() as messages:
        _verifier_cli(module.main)
    _verifier_journal(messages, _CLES_CLI.values())


def _verifier_cli(main) -> None:
    with _dossier_temporaire() as dossier:
        base = dossier / "service.db"
        env = _env_cli(base)
        liste = [(f"K{i:02d}", ("APPEL", "REMISE", "SUPPORT")[i % 3]) for i in range(1, 31)]
        contacts = _ecrire(dossier, "a_contacter.csv", "customer_id,offre\n" + "".join(f"{c},{o}\n" for c, o in liste),
                           encodage="utf-8-sig")

        code, _, _ = _main(main, [], env, "sans commande")
        if code != 2:
            raise AssertionError(f"`{_commande([])}` (sans commande) : attendu le code 2 (refus par argparse), obtenu "
                             f"{code!r}. `parser.add_subparsers(dest=\"commande\", required=True)`.")

        absente = dossier / "absente.db"
        for argv in (["avancement"], ["servir"]):
            lanceur = _Lanceur()
            code, out, err = _main(main, argv, _env_cli(absente), "base absente", lancer=lanceur)
            if code != 1 or "importer" not in err:
                raise AssertionError(f"`{_commande(argv)}` (la base n'existe pas encore) : attendu le code 1 et un message "
                                     f"sur stderr qui conseille « importer », obtenu le code {code!r} et {err.strip()!r}. "
                                     "Testez config.base.exists() AVANT d'ouvrir : sqlite3.connect créerait une base vide.")
            if absente.exists() or lanceur.appels:
                raise AssertionError(f"`{_commande(argv)}` (base absente) a créé la base ou lancé le serveur : vérifiez "
                                     "l'existence de la base d'abord.")

        argv = ["importer", "--contacts", str(contacts)]
        code, out, err = _main(main, argv, env, "importer")
        if code is None:
            raise AssertionError(f"`{_commande(argv)}` : main renvoie None. Terminez par `return 0`.")
        if code != 0:
            raise AssertionError(f"`{_commande(argv)}` : code {code}, attendu 0. Erreur : {err.strip()[-300:] or '(aucune)'}")
        _sans_secret(argv, out, err)
        with _fermee(sqlite3.connect(base)) as conn:
            charge = dict(conn.execute("SELECT conseiller, COUNT(*) FROM contacts GROUP BY conseiller"))
        if charge != {"alice": 10, "bruno": 10, "chloe": 10}:
            raise AssertionError(f"Après `{_commande(argv)}`, la base devrait répartir les 30 contacts entre alice, bruno et "
                                 f"chloe (10 chacun), obtenu {charge}. Les conseillers sont ceux de SERVICE_CLES : "
                                 "config.conseillers.")
        if "30" not in out or not all(nom in out for nom in _CLES_CLI):
            raise AssertionError(f"Après importer, affichez le nombre de contacts (30) et l'avancement (le tableau de "
                                 f"formater_avancement, avec chaque conseiller). Affiché : {out.strip()[:300]!r}")
        code, out, _ = _main(main, argv, env, "importer une seconde fois")
        # Le tableau d'avancement contient des 0 : on ne regarde que les lignes AU-DESSUS du tableau.
        resume = "\n".join(ligne for ligne in out.splitlines()
                           if ligne.split() and ligne.split()[0] not in (*_CLES_CLI, "conseiller", "TOTAL"))
        if code != 0 or not re.search(r"(?<![\d,.])0(?![\d,.])", resume):
            raise AssertionError(f"Relancer `{_commande(argv)}` : code {code!r}, et l'affichage doit indiquer 0 nouveau "
                                 f"contact. Affiché : {out.strip()[:200]!r}")

        with _fermee(ref_base.ouvrir(base)) as conn:
            tableau = ref_fichiers.formater_avancement(ref_contacts.avancement(conn))
        for env_ici, quoi in ((env, "avec env"), (None, "avec os.environ (env=None)")):
            with mock.patch.dict(os.environ, env):
                code, out, err = _main(main, ["avancement"], env_ici, quoi)
            if code != 0 or tableau not in out:
                raise AssertionError(f"`{_commande(['avancement'])}` ({quoi}) : code {code!r}, attendu 0 et le tableau "
                                     f"formater_avancement(avancement(conn)) :\n{tableau}\nAffiché : {out.strip()[:300]!r}"
                                     f"{' ; erreur : ' + err.strip()[:200] if err.strip() else ''}. Sans env, lisez "
                                     "os.environ.")

        lanceur = _Lanceur()
        argv = ["servir", "--port", "8123", "--hote", "0.0.0.0"]
        code, out, err = _main(main, argv, env, "servir", lancer=lanceur)
        if code != 0 or len(lanceur.appels) != 1:
            raise AssertionError(f"`{_commande(argv)}` : code {code!r} (attendu 0), et `lancer` appelé "
                                 f"{len(lanceur.appels)} fois (attendu 1). Erreur : {err.strip()[-200:] or '(aucune)'}")
        _sans_secret(argv, out, err)
        args, kwargs = lanceur.appels[0]
        app = args[0] if args else kwargs.get("app")
        if kwargs.get("host") != "0.0.0.0" or kwargs.get("port") != 8123:
            raise AssertionError(f"`{_commande(argv)}` : lancer doit être appelé comme uvicorn.run : "
                                 f"lancer(app, host=args.hote, port=args.port), obtenu args={args!r}, kwargs={kwargs!r}.")
        if not hasattr(app, "router"):
            raise AssertionError(f"`{_commande(argv)}` : le premier argument de lancer doit être l'application "
                                 f"(creer_app(config)), obtenu {type(app).__name__}.")
        from fastapi.testclient import TestClient

        reponse = TestClient(app).get("/contacts", headers={"X-Cle-API": _CLES_CLI["chloe"]})
        if reponse.status_code != 200 or len(reponse.json().get("contacts", [])) != 10:
            raise AssertionError(f"`{_commande(argv)}` : l'application lancée ne sert pas la bonne base (chloe devrait voir "
                                 f"ses 10 contacts, réponse {reponse.status_code}).")
        if not all(nom in out for nom in _CLES_CLI):
            raise AssertionError(f"Au démarrage, affichez l'adresse et le nom des conseillers configurés (jamais leurs "
                                 f"clés). Affiché : {out.strip()[:200]!r}")
        lanceur = _Lanceur()
        code, _, _ = _main(main, ["servir"], env, "servir sans option", lancer=lanceur)
        if code != 0 or not lanceur.appels or lanceur.appels[0][1].get("port") != 8000 \
                or lanceur.appels[0][1].get("host") != "127.0.0.1":
            raise AssertionError("`servir` sans option : hôte 127.0.0.1 (accessible depuis cette machine seulement) et port "
                                 f"8000 par défaut. Obtenu {lanceur.appels[:1]}.")

        for port in ("99999", "0", "huit"):
            lanceur = _Lanceur()
            argv = ["servir", "--port", port]
            code, _, _ = _main(main, argv, env, f"port {port}", lancer=lanceur)
            if code != 2 or lanceur.appels:
                raise AssertionError(f"`{_commande(argv)}` : code {code!r}, attendu 2 (refus par argparse) sans lancer le "
                                     "serveur. Une fonction pour `type=` qui lève argparse.ArgumentTypeError hors de "
                                     "1..65535.")

        for env_faux, quoi in (({"SERVICE_BASE": str(base)}, "sans SERVICE_CLES"),
                               (_env_cli(base, SERVICE_CLES="alice=secret-court"), "une clé trop courte"),
                               (_env_cli(base, SERVICE_TAILLE_PAGE="mille"), "SERVICE_TAILLE_PAGE=mille")):
            lanceur = _Lanceur()
            code, out, err = _main(main, ["servir"], env_faux, quoi, lancer=lanceur)
            if code != 1 or "SERVICE_" not in err or lanceur.appels:
                raise AssertionError(f"`{_commande(['servir'])}` ({quoi}) : attendu le message de ConfigInvalide sur stderr "
                                     f"et le code 1, sans lancer le serveur. Obtenu le code {code!r}, stderr "
                                     f"{err.strip()[:200]!r}.")
            if "secret-court" in out + err:
                raise AssertionError(f"`{_commande(['servir'])}` ({quoi}) affiche la clé fautive : nommez le conseiller.")

        casse = _ecrire(dossier, "casse.db", "ceci n'est pas une base SQLite\n" * 40)
        for argv in (["avancement"], ["servir"]):
            lanceur = _Lanceur()
            code, _, err = _main(main, argv, _env_cli(casse), "un fichier qui n'est pas une base", lancer=lanceur)
            if code != 1 or not err.strip() or lanceur.appels:
                raise AssertionError(f"`{_commande(argv)}` sur un fichier qui n'est pas une base SQLite : attrapez "
                                     f"sqlite3.Error, message sur stderr et code 1, sans lancer le serveur (obtenu {code!r}). "
                                     "Ouvrez la base AVANT de démarrer : mieux vaut échouer tout de suite qu'à la première "
                                     "requête d'un conseiller.")

        for fichier, quoi in ((dossier / "nexiste_pas.csv", "fichier introuvable"),
                              (_ecrire(dossier, "incomplet.csv", "customer_id\nK1\n"), "colonne offre manquante")):
            argv = ["importer", "--contacts", str(fichier)]
            code, _, err = _main(main, argv, env, quoi)
            if code != 1 or not err.strip():
                raise AssertionError(f"`{_commande(argv)}` ({quoi}) : attendu un message sur stderr et le code 1, obtenu "
                                     f"le code {code!r}.")


# ============================================================================ Étape 7 : tester_service


def _mutant(fichier: str, avant: str, apres: str):
    """`creer_app` du corrigé, avec UN bug introduit dans `fichier` (api, contacts ou appels)."""
    modules: dict[str, ModuleType] = {}
    for nom in ("contacts", "appels", "api"):
        chemin = _CORRIGE / f"{nom}.py"
        texte = chemin.read_text(encoding="utf-8")
        if nom == fichier:
            if texte.count(avant) != 1:
                raise RuntimeError(f"mutant introuvable dans {nom}.py : {avant!r}")
            texte = texte.replace(avant, apres)
        module = types.ModuleType(f"projets.service_corrige._mutant_{nom}")
        module.__package__ = "projets.service_corrige"
        module.__file__ = str(chemin)
        exec(compile(texte, str(chemin), "exec"), module.__dict__)  # noqa: S102 — notre propre code, modifié
        modules[nom] = module
    modules["appels"].lire_contact = modules["contacts"].lire_contact
    modules["api"].contacts, modules["api"].appels = modules["contacts"], modules["appels"]
    return modules["api"].creer_app


#: (fichier, texte d'origine, texte bogué, ce que fait la version boguée, conseil)
_MUTANTS = (
    ("api", "        if trouve is None:\n",
     "        if trouve is None and not x_cle_api:\n            return next(iter(config.cles))\n        if trouve is None:\n",
     "laisse entrer une requête SANS clé (elle passe pour le premier conseiller)",
     "Appelez GET /contacts sans en-tête X-Cle-API : attendu 401."),
    ("api", 'if secrets.compare_digest(recue, cle.encode("utf-8")):', 'if recue[:8] == cle.encode("utf-8")[:8]:',
     "accepte une clé fausse qui COMMENCE comme la vraie",
     "Un attaquant ne tente pas « mauvaise » : essayez la clé d'alice avec son dernier caractère changé (attendu 401)."),
    ("contacts", "    if contact.conseiller != conseiller:\n", "    if False:\n",
     "montre à alice la fiche d'un client suivi par bruno",
     "Demandez GET /contacts/C006 avec la clé d'alice : attendu 403."),
    ("api", "status_code=status.HTTP_404_NOT_FOUND", "status_code=status.HTTP_403_FORBIDDEN",
     "répond 403 (interdit) au lieu de 404 pour un client qui n'existe pas",
     "Demandez un client qui n'existe pas (GET /contacts/C999) : attendu 404."),
    ("appels", "AND conseiller = ? AND version = ?", "AND conseiller = ? AND version >= ?",
     "écrase la fiche même quand la version envoyée est périmée",
     "Enregistrez un appel (version 1 → 2), puis renvoyez la version 1 : attendu 409, et la fiche ne doit pas changer."),
    ("api", 'content={"detail": str(exc), "version_actuelle": exc.version_actuelle}', 'content={"detail": str(exc)}',
     "répond 409 sans dire la version actuelle de la fiche",
     "En cas de conflit, vérifiez aussi le CONTENU de la réponse : reponse.json()[\"version_actuelle\"] == 2."),
    ("contacts", "    suivant = contacts[-1].customer_id if len(lignes) > taille else None\n", "    suivant = None\n",
     "ne donne jamais le curseur de la page suivante (`suivant` vaut toujours null)",
     "Avec taille_page=2, la première page d'alice (5 contacts) doit avoir suivant == \"C002\"."),
    ("contacts", '    if statut is not None:\n        conditions.append("statut = ?")\n        parametres.append(statut)\n',
     "", "ignore le filtre ?statut=",
     "Enregistrez un appel « accepte », puis GET /contacts?statut=accepte : seul ce contact doit revenir."),
)


def verifier_tester_service(module: ModuleType) -> None:
    with _journal_capte():
        _verifier_tester_service(module, module.tester_service)


def _verifier_tester_service(module: ModuleType, tester) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tester(ref_api.creer_app)
    except NotImplementedError:
        raise
    except AssertionError as exc:
        raise AssertionError(f"Votre test ÉCHOUE sur la version JUSTE de l'API ({str(exc) or 'assert sans message'}"
                             f"{_lieu(exc, module)}). Relisez ce que promet chaque route (projets/README.md, étape 5) et ce "
                             "que contient la base de _preparer().") from None
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"Votre test plante sur la version JUSTE : {type(exc).__name__}: {exc}{_lieu(exc, module)}. "
                             "Créez le client avec `TestClient(creer_app(_preparer()))`.") from None

    survivants = []
    for fichier, avant, apres, defaut, conseil in _MUTANTS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tester(_mutant(fichier, avant, apres))
        except Exception:  # noqa: BLE001 — n'importe quelle erreur : le test a vu le bug
            continue
        survivants.append((defaut, conseil))
    if survivants:
        if len(survivants) == len(_MUTANTS):
            raise AssertionError("Votre test passe sur TOUTES les versions boguées : il ne vérifie rien. Pour chaque requête, "
                                 "vérifiez le code HTTP ET le contenu : `assert reponse.status_code == 401`.")
        defaut, conseil = survivants[0]
        raise AssertionError(f"{len(_MUTANTS) - len(survivants)}/{len(_MUTANTS)} bugs attrapés. Votre test laisse passer une "
                             f"version qui {defaut}. {conseil}")


# ============================================================================ Le projet

PROJET = Projet(
    numero=4,
    code="service",
    titre="Le service des conseillers",
    paquet_eleve="projets.service",
    paquet_corrige="projets.service_corrige",
    commande="python -m projets.service importer --contacts projets/exemple_contacts.csv",
    etapes=(
        Etape(1, "Lire la configuration sans montrer un secret", "🟢", "config", ("lire_config",), verifier_config),
        Etape(2, "Répartir les contacts entre les conseillers", "🟡", "base", ("ouvrir", "importer_contacts"),
              verifier_base),
        Etape(3, "Lire une fiche, une page, l'avancement", "🟡", "contacts",
              ("lire_contact", "page_contacts", "avancement"), verifier_contacts),
        Etape(4, "Enregistrer un appel sans écraser une collègue", "🔴", "appels", ("enregistrer_appel",),
              verifier_appels),
        Etape(5, "L'API web des conseillers", "🟡", "api", ("creer_app",), verifier_api),
        Etape(6, "Le programme à trois commandes", "🟡", "cli", ("main",), verifier_cli),
        Etape(7, "Écrire les tests qui protègent l'API", "🔴", "mes_tests", ("tester_service",),
              verifier_tester_service),
    ),
)

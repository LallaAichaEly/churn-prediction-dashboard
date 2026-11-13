"""Vérifications automatiques de la série 15 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Le correcteur travaille sur de vraies petites bases SQLite, dans un dossier temporaire. La connexion
qu'il vous passe est une vraie `sqlite3.Connection`, un peu espionne : elle note chaque requête,
compte les lignes que votre code lit, et peut faire écrire un collègue (ou un autre travailleur) au
pire moment : entre votre SELECT et votre UPDATE. Le correcteur fait aussi tomber en panne une
écriture, ou le serveur au milieu d'une migration.

Après chaque appel, il regarde la base depuis une AUTRE connexion : c'est ce que verront vos
collègues. Une écriture sans commit() y est invisible.
"""

from __future__ import annotations

import collections
import contextlib
import math
import random
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .persistance import BAIL, STATUTS, ConflitDeVersion, creer_schema
from .verif import Defi, Serie


def _court(valeur, n: int = 160) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


# ============================================================================ L'espion


class _TropDeRequetes(BaseException):
    """Levée par l'espion au-delà d'un nombre de requêtes : protège le correcteur d'une boucle sans fin.

    BaseException (et non Exception) pour traverser un `except Exception:` de votre code.
    """


class _PanneServeur(Exception):
    """Une panne simulée par le correcteur (le serveur s'arrête, le disque est plein…)."""


@dataclass
class _Releve:
    journal: list = field(default_factory=list)   # (sorte, sql) : execute, executemany, executescript, commit…
    lignes_lues: int = 0
    fermee: bool = False


def _sorte_requete(sql: str) -> str:
    """« lecture », « ecriture » ou « autre » (BEGIN, PRAGMA, CREATE…)."""
    debut = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
    if debut in ("SELECT", "WITH", "VALUES"):
        return "lecture"
    if debut in ("INSERT", "UPDATE", "DELETE", "REPLACE"):
        return "ecriture"
    return "autre"


def _touche(sql: str, table: str) -> bool:
    return re.search(rf"\b{table}\b", sql, re.IGNORECASE) is not None


class _Curseur(sqlite3.Cursor):
    """Un vrai curseur sqlite3, qui lit tout le résultat dès l'exécution (la base est ainsi libérée tout
    de suite), puis rend les lignes une à une, en les comptant."""

    def __init__(self, connexion: "_Espion") -> None:
        super().__init__(connexion)
        self._lignes = collections.deque()

    def execute(self, sql, *parametres):
        self.connection._noter("execute", sql)
        super().execute(sql, *parametres)
        self._lignes = collections.deque(super().fetchall())
        return self

    def executemany(self, sql, *parametres):
        self.connection._noter("executemany", sql)
        super().executemany(sql, *parametres)
        self._lignes = collections.deque()
        return self

    def executescript(self, script):
        self.connection._noter("executescript", script)
        self._lignes = collections.deque()
        return super().executescript(script)

    def fetchone(self):
        if not self._lignes:
            return None
        self.connection._releve.lignes_lues += 1
        return self._lignes.popleft()

    def fetchmany(self, size=None):
        n = self.arraysize if size is None else size
        return [self.fetchone() for _ in range(min(n, len(self._lignes)))]

    def fetchall(self):
        return [self.fetchone() for _ in range(len(self._lignes))]

    def __next__(self):
        if not self._lignes:
            raise StopIteration
        return self.fetchone()

    def close(self) -> None:
        self._lignes.clear()
        super().close()


class _Espion(sqlite3.Connection):
    """Une vraie connexion sqlite3 (même `isinstance`), qui note chaque requête et peut déclencher
    un événement juste avant l'une d'elles : une panne, ou un collègue qui écrit."""

    def _preparer(self, avant=None, limite: int = 300) -> None:
        self._avant = avant      # avant(espion, sorte, sql) : appelé avant chaque requête
        self._limite = limite
        self._releve = _Releve()

    def _noter(self, sorte: str, sql: str = "") -> None:
        releve = self._releve
        releve.journal.append((sorte, sql))
        if len(releve.journal) > self._limite:
            raise _TropDeRequetes(len(releve.journal))
        if self._avant is not None and sorte in ("execute", "executemany", "executescript"):
            self._avant(self, sorte, sql)

    def requetes(self, sorte: str | None = None) -> list[str]:
        """Le texte des requêtes exécutées, éventuellement filtrées (« lecture », « ecriture », « autre »)."""
        return [sql for quoi, sql in self._releve.journal
                if quoi in ("execute", "executemany") and (sorte is None or _sorte_requete(sql) == sorte)]

    def cursor(self, factory=None):
        return super().cursor(factory or _Curseur)

    def execute(self, sql, *parametres):
        return self.cursor().execute(sql, *parametres)

    def executemany(self, sql, *parametres):
        return self.cursor().executemany(sql, *parametres)

    def executescript(self, script):
        return self.cursor().executescript(script)

    def commit(self) -> None:
        self._noter("commit")
        super().commit()

    def rollback(self) -> None:
        self._noter("rollback")
        super().rollback()

    def close(self) -> None:
        # On ne ferme pas vraiment : on note, et on vous le dira.
        self._noter("close")
        self._releve.fermee = True

    def __exit__(self, type_, valeur, trace):
        self._releve.journal.append(("commit" if type_ is None else "rollback", "with conn"))
        return super().__exit__(type_, valeur, trace)


@dataclass
class _Appel:
    resultat: object
    erreur: BaseException | None
    espion: _Espion
    en_transaction: bool          # une transaction était-elle encore ouverte après l'appel ?
    photo_si_commit: dict | None  # si oui : ce que la base contiendrait après le prochain commit()
    fermee: bool                  # votre code a-t-il appelé conn.close() ?


@contextlib.contextmanager
def _base(remplir):
    """Crée une base dans un dossier temporaire, la remplit, et donne son chemin."""
    dossier = Path(tempfile.mkdtemp(prefix="correcteur-serie15-"))
    chemin = dossier / "retention.db"
    conn = sqlite3.connect(chemin)
    try:
        if remplir is not None:
            remplir(conn)
        conn.commit()
    finally:
        conn.close()
    try:
        yield chemin
    finally:
        shutil.rmtree(dossier, ignore_errors=True)


def _photo(chemin: Path) -> dict:
    """Le contenu de toutes les tables, vu depuis une AUTRE connexion (ce que voient les collègues)."""
    conn = sqlite3.connect(chemin, timeout=2)
    try:
        tables = [n for (n,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {t: sorted(conn.execute(f'SELECT * FROM "{t}"').fetchall(), key=repr) for t in tables}
    finally:
        conn.close()


def _differences(avant: dict, apres: dict, n: int = 4) -> str:
    morceaux = []
    for table in sorted(set(avant) | set(apres)):
        a, b = avant.get(table, []), apres.get(table, [])
        en_plus = [ligne for ligne in b if ligne not in a]
        en_moins = [ligne for ligne in a if ligne not in b]
        if en_moins:
            morceaux.append(f"table {table}, avant : {', '.join(map(repr, en_moins[:n]))}" + (" …" if len(en_moins) > n else ""))
        if en_plus:
            morceaux.append(f"table {table}, après : {', '.join(map(repr, en_plus[:n]))}" + (" …" if len(en_plus) > n else ""))
    return " ; ".join(morceaux) or "aucune différence"


def _appeler(fonction, chemin: Path, *args, avant=None, limite: int = 300, quand: str = "",
             simuler_commit: bool = True) -> _Appel:
    """Appelle votre fonction avec une connexion espionne, puis la ferme.

    Si une transaction est restée ouverte, `simuler_commit` joue le « prochain commit() » d'un autre
    code, pour montrer ce que la base contiendrait. Pour simuler un arrêt du serveur, on ne le fait
    pas : fermer la connexion annule la transaction, comme le ferait SQLite au redémarrage.
    """
    espion = sqlite3.connect(chemin, timeout=1, factory=_Espion)
    espion._preparer(avant=avant, limite=limite)
    try:
        try:
            resultat, erreur = fonction(espion, *args), None
        except NotImplementedError:
            raise
        except _TropDeRequetes as trop:
            raise AssertionError(
                f"{quand} : votre fonction a exécuté plus de {trop.args[0] - 1} requêtes sans s'arrêter : une boucle "
                f"sans fin ? Les dernières : {_court(espion.requetes()[-3:], 400)}. Si votre boucle recommence quand "
                "un UPDATE ne modifie rien, vérifiez que cet UPDATE PEUT réussir pour la ligne choisie par votre "
                "SELECT : sa condition doit être exactement la même.") from None
        except Exception as exc:  # noqa: BLE001 — on garde l'erreur pour l'examiner
            resultat, erreur = None, exc
        en_transaction = espion.in_transaction
        photo = None
        if en_transaction and simuler_commit:
            with contextlib.suppress(sqlite3.Error):
                sqlite3.Connection.commit(espion)  # le « prochain commit » de n'importe quel autre code
                photo = _photo(chemin)
        return _Appel(resultat, erreur, espion, en_transaction, photo, espion._releve.fermee)
    finally:
        sqlite3.Connection.close(espion)


def _verifier_rendue_propre(appel: _Appel, quand: str, avant: dict | None = None) -> None:
    if appel.fermee:
        raise AssertionError(f"{quand} : votre fonction FERME la connexion reçue (`conn.close()`). Elle ne vous "
                             "appartient pas : le reste du programme s'en sert encore après vous. Rendez-la ouverte.")
    if appel.en_transaction:
        if appel.erreur is None:
            raise AssertionError(
                f"{quand} : vos écritures ne sont pas validées : une transaction est encore ouverte après l'appel "
                "(`conn.in_transaction` vaut True), et une AUTRE connexion ne voit rien de ce que vous avez écrit. "
                "Ajoutez `conn.commit()` à la fin, ou mettez les écritures dans un bloc `with conn:`.")
        moitie = f" Après ce commit, la base contiendrait : {_differences(avant, appel.photo_si_commit)}." if (
            avant is not None and appel.photo_si_commit is not None and appel.photo_si_commit != avant) else ""
        raise AssertionError(
            f"{quand} : votre fonction lève {type(appel.erreur).__name__} mais laisse une transaction OUVERTE "
            "(`conn.in_transaction` vaut True). Ce qui était déjà écrit attend dans cette transaction, invisible… "
            f"jusqu'au prochain `conn.commit()` de n'importe quel autre code qui utilise la connexion.{moitie} "
            "En cas d'erreur, `conn.rollback()` avant de relancer (`raise`), ou un bloc `with conn:` qui le fait pour vous.")


def _requete_plantee(appel: _Appel, quand: str, valeurs=()) -> None:
    """Explique une erreur SQL de VOTRE requête (syntaxe, table inconnue…)."""
    if isinstance(appel.erreur, (sqlite3.OperationalError, sqlite3.ProgrammingError)):
        derniere = appel.espion.requetes()[-1] if appel.espion.requetes() else ""
        apostrophe = " Une valeur contient une apostrophe : collée dans le texte SQL, elle le coupe en deux. " \
                     "Passez les valeurs en paramètres : `conn.execute(\"… = ?\", (valeur,))` (série 13, défi 80)." \
            if any("'" in str(v) for v in valeurs) else ""
        raise AssertionError(f"{quand} : votre requête plante : {type(appel.erreur).__name__}: {appel.erreur} "
                             f"(dernière requête : {_court(derniere, 220)}).{apostrophe}")


# ============================================================================ Défi 92 : transaction

_CLIENTS = [
    ("C001", "Léa Martin", 0.82, "a_contacter", 1),
    ("C002", "Paul D'Almeida", 0.67, "offre_envoyee", 3),
    ("C003", "Zoé Zébulon", 0.15, "fidelise", 2),
    ("C004", "Inès Durand", 0.91, "a_contacter", 1),
    ("C005", "Noah Petit", 0.44, "contacte", 2),
    ("C006", "Emma Roux", 0.73, "a_contacter", 1),
    ("C007", "Hugo Lefèvre", 0.58, "a_contacter", 1),
    ("C008", "Chloé Bernard", 0.29, "parti", 4),
]


def _remplir_92(conn: sqlite3.Connection) -> None:
    creer_schema(conn)
    conn.executemany("INSERT INTO clients VALUES (?, ?, ?, ?, ?)", _CLIENTS)
    conn.executemany("INSERT INTO budget VALUES (?, ?)",
                     [("automne", 500.0), ("fidelite", 120.0), ("noël d'hiver", 200.0), ("printemps", 1000.0)])
    conn.execute("INSERT INTO offres (client_id, campagne, montant) VALUES ('C002', 'automne', 30.0)")
    # Deux pannes simulées : l'écriture dans `offres` échoue pour C006, celle du budget pour « printemps ».
    conn.execute("CREATE TRIGGER panne_offre BEFORE INSERT ON offres WHEN NEW.client_id = 'C006' "
                 "BEGIN SELECT RAISE(ABORT, 'panne simulée : disque plein (table offres)'); END")
    conn.execute("CREATE TRIGGER panne_budget BEFORE UPDATE ON budget WHEN OLD.campagne = 'printemps' "
                 "BEGIN SELECT RAISE(ABORT, 'panne simulée : disque plein (table budget)'); END")


def _budget(photo: dict) -> dict:
    return {campagne: reste for campagne, reste in photo["budget"]}


def _offres(photo: dict) -> dict:
    return {client: (identifiant, campagne, montant) for identifiant, client, campagne, montant in photo["offres"]}


def _reussite_92(accorder_offre, client, campagne, montant) -> None:
    quand = f"accorder_offre(conn, {client!r}, {campagne!r}, {montant!r})"
    with _base(_remplir_92) as chemin:
        avant = _photo(chemin)
        appel = _appeler(accorder_offre, chemin, client, campagne, montant, quand=quand)
        _requete_plantee(appel, quand, (client, campagne))
        if appel.erreur is not None:
            conseil = ""
            if isinstance(appel.erreur, ValueError) and montant == _budget(avant)[campagne]:
                conseil = (f" Le budget restant ({_budget(avant)[campagne]} €) couvre EXACTEMENT le montant : c'est "
                           "permis (le reste tombe à 0). Refusez seulement si `reste < montant`.")
            elif "panne simulée" in str(appel.erreur):
                conseil = (f" Le correcteur fait échouer toute écriture sur le budget de « printemps » et sur l'offre "
                           f"de C006 ; or l'offre concerne {client} et {campagne!r} : votre code a écrit sur d'AUTRES "
                           "campagnes (ou d'autres clients). Votre UPDATE a-t-il un `WHERE campagne = ?` ?")
            raise AssertionError(f"{quand} devrait réussir, mais lève {type(appel.erreur).__name__}: {appel.erreur}.{conseil}")
        _verifier_rendue_propre(appel, quand, avant)
        apres = _photo(chemin)
        offre = _offres(apres).get(client)
        if offre is None:
            raise AssertionError(f"{quand} : après l'appel, une autre connexion ne voit aucune offre pour {client}. "
                                 f"Différences : {_differences(avant, apres)}.")
        if offre[1:] != (campagne, montant):
            raise AssertionError(f"{quand} : l'offre enregistrée vaut {offre[1:]!r} (campagne, montant) au lieu de "
                                 f"{(campagne, montant)!r}.")
        reste_attendu = _budget(avant)[campagne] - montant
        if not math.isclose(_budget(apres)[campagne], reste_attendu):
            raise AssertionError(f"{quand} : le budget de {campagne!r} vaut {_budget(apres)[campagne]} au lieu de "
                                 f"{reste_attendu} ({_budget(avant)[campagne]} - {montant}). "
                                 "Retirez le montant : `UPDATE budget SET reste = reste - ? WHERE campagne = ?`.")
        autres = {k: v for k, v in _budget(apres).items() if k != campagne}
        if autres != {k: v for k, v in _budget(avant).items() if k != campagne}:
            raise AssertionError(f"{quand} : le budget d'AUTRES campagnes a changé : {_differences(avant, apres)}. "
                                 "Votre UPDATE a-t-il un `WHERE campagne = ?` ?")
        if len(apres["offres"]) != len(avant["offres"]) + 1:
            raise AssertionError(f"{quand} : il y a {len(apres['offres']) - len(avant['offres'])} offres de plus au "
                                 "lieu d'une seule.")
        if isinstance(appel.resultat, bool) or not isinstance(appel.resultat, int) or appel.resultat != offre[0]:
            raise AssertionError(f"{quand} renvoie {_court(appel.resultat)} au lieu de l'identifiant de la nouvelle "
                                 f"offre ({offre[0]}) : `curseur = conn.execute(\"INSERT …\", (…))` puis "
                                 "`return curseur.lastrowid`.")


def _refus_92(accorder_offre, client, campagne, montant, pourquoi: str, attendues: tuple, conseil: str = "") -> None:
    quand = f"accorder_offre(conn, {client!r}, {campagne!r}, {montant!r})"
    with _base(_remplir_92) as chemin:
        avant = _photo(chemin)
        appel = _appeler(accorder_offre, chemin, client, campagne, montant, quand=quand)
        if appel.erreur is None:
            apres = _photo(chemin) if not appel.en_transaction else appel.photo_si_commit
            change = "" if apres == avant else f" Pire : la base a changé ({_differences(avant, apres)})."
            raise AssertionError(f"{quand} : {pourquoi}, votre fonction aurait dû lever une erreur, mais elle renvoie "
                                 f"{_court(appel.resultat)}.{change} L'appelant doit SAVOIR que l'offre n'a pas été "
                                 f"accordée : ne cachez pas l'erreur (pas de `except: return None`).{conseil}")
        if not isinstance(appel.erreur, attendues):
            _requete_plantee(appel, quand, (client, campagne))
            noms = " ou ".join(t.__name__ for t in attendues)
            raise AssertionError(f"{quand} : {pourquoi}, on attend {noms}, mais votre fonction lève "
                                 f"{type(appel.erreur).__name__}: {appel.erreur}.{conseil}")
        _verifier_rendue_propre(appel, quand, avant)
        apres = _photo(chemin)
        if apres != avant:
            journal = [quoi if quoi not in ("execute", "executemany") else _sorte_requete(sql)
                       for quoi, sql in appel.espion._releve.journal]
            premier_commit = journal.index("commit") if "commit" in journal else len(journal)
            entre = (" Votre code a fait `commit()` ENTRE les deux écritures : la 1re est enregistrée pour de bon "
                     "avant que la 2e échoue. Un seul commit, à la fin, quand tout a réussi."
                     if "ecriture" in journal[:premier_commit] and "ecriture" in journal[premier_commit:] else "")
            raise AssertionError(f"{quand} : {pourquoi}, et pourtant la base a changé : {_differences(avant, apres)}. "
                                 f"Après une erreur, la base doit être EXACTEMENT comme avant.{entre}{conseil}")


def verifier_accorder_offre(accorder_offre) -> None:
    _reussite_92(accorder_offre, "C001", "automne", 50.0)
    _reussite_92(accorder_offre, "C004", "fidelite", 120.0)          # tout le reste : permis
    _reussite_92(accorder_offre, "C007", "noël d'hiver", 80.0)       # une apostrophe dans la campagne
    _refus_92(accorder_offre, "C004", "fidelite", 120.5, "le budget restant (120 €) ne couvre pas 120,5 €",
              (ValueError,), " Comparez le reste au montant AVANT d'écrire, ou levez la ValueError dans le bloc "
              "`with conn:` pour que tout soit annulé.")
    _refus_92(accorder_offre, "C004", "automne", 0.0, "un montant nul n'est pas une offre", (ValueError,),
              " Vérifiez `montant > 0` avant tout.")
    _refus_92(accorder_offre, "C004", "automne", -50.0, "un montant NÉGATIF ajouterait 50 € au budget "
              "(reste - (-50))", (ValueError,), " Vérifiez `montant > 0` avant tout.")
    _refus_92(accorder_offre, "C004", "hiver-2027", 10.0, "la campagne n'existe pas", (ValueError,),
              " Lisez d'abord le budget : `ligne = conn.execute(\"SELECT reste FROM budget WHERE campagne = ?\", "
              "(campagne,)).fetchone()` vaut None quand la campagne n'existe pas. (Un UPDATE sur une campagne "
              "inconnue ne modifie AUCUNE ligne, sans erreur : il ne vous préviendra pas.)")
    _refus_92(accorder_offre, "C002", "automne", 20.0, "C002 a déjà une offre (contrainte UNIQUE)",
              (sqlite3.IntegrityError, ValueError))
    _refus_92(accorder_offre, "C006", "automne", 20.0, "le correcteur a fait tomber en panne l'INSERT dans "
              "`offres` (une erreur que votre code ne pouvait pas prévoir)", (Exception,))
    _refus_92(accorder_offre, "C004", "printemps", 20.0, "le correcteur a fait tomber en panne l'UPDATE du "
              "budget (une erreur que votre code ne pouvait pas prévoir)", (Exception,))


# ============================================================================ Défi 93 : pagination


def _clients_93(n: int, graine: int = 15) -> list[tuple]:
    clients = [(f"C{k:03d}", f"Client n°{k}", round(random.Random(k).random(), 2), "a_contacter", 1) for k in range(1, n + 1)]
    random.Random(graine).shuffle(clients)  # inséré dans le désordre : seul ORDER BY garantit l'ordre
    return clients


def _remplir_93(n: int):
    def remplir(conn: sqlite3.Connection) -> None:
        creer_schema(conn)
        conn.executemany("INSERT INTO clients VALUES (?, ?, ?, ?, ?)", _clients_93(n))
    return remplir


def _attendu_93(chemin: Path, apres, taille) -> tuple[list[dict], str | None]:
    conn = sqlite3.connect(chemin)
    try:
        lignes = conn.execute("SELECT id, nom, score FROM clients WHERE ? IS NULL OR id > ? ORDER BY id",
                              (apres, apres)).fetchall()
    finally:
        conn.close()
    page = [{"id": i, "nom": n, "score": s} for i, n, s in lignes[:taille]]
    return page, (page[-1]["id"] if len(lignes) > taille else None)


def _page(page_clients, chemin: Path, apres, taille) -> tuple[list[dict], str | None]:
    quand = f"page_clients(conn, apres={apres!r}, taille={taille})"
    appel = _appeler(page_clients, chemin, apres, taille, quand=quand)
    _requete_plantee(appel, quand, (apres,))
    if appel.erreur is not None:
        raise AssertionError(f"{quand} lève {type(appel.erreur).__name__}: {appel.erreur}.")
    _verifier_rendue_propre(appel, quand)
    resultat = appel.resultat
    if not isinstance(resultat, tuple) or len(resultat) != 2:
        raise AssertionError(f"{quand} renvoie {_court(resultat)} au lieu d'un couple `(clients, suivant)` : "
                             "`return page, suivant`.")
    page, suivant = resultat
    if not isinstance(page, list) or not all(isinstance(c, dict) for c in page):
        raise AssertionError(f"{quand} : le 1er élément vaut {_court(page)} au lieu d'une LISTE de dictionnaires "
                             "`{'id': …, 'nom': …, 'score': …}`.")
    for client in page:
        if set(client) != {"id", "nom", "score"}:
            raise AssertionError(f"{quand} : un client vaut {_court(client)}. Il faut exactement les clés 'id', "
                                 "'nom' et 'score' : `{\"id\": id_, \"nom\": nom, \"score\": score}`.")
    attendu_page, attendu_suivant = _attendu_93(chemin, apres, taille)
    ids, ids_attendus = [c["id"] for c in page], [c["id"] for c in attendu_page]
    if ids != ids_attendus:
        if sorted(ids) == ids_attendus or ids != sorted(ids):
            conseil = (" Les clients ne sont pas triés par id : sans `ORDER BY id`, SQLite rend les lignes dans l'ordre "
                       "qui l'arrange (ici, l'ordre d'insertion), et LIMIT garde alors n'importe lesquels.")
        elif apres is not None and ids[:1] == [apres]:
            conseil = (f" La page commence par {apres!r}, le DERNIER client de la page précédente : il serait affiché "
                       "deux fois. Il faut les id STRICTEMENT plus grands : `id > ?`, pas `id >= ?`.")
        elif len(ids) > taille:
            conseil = f" {len(ids)} clients pour une page de {taille} : ne gardez que les `taille` premiers (`lignes[:taille]`)."
        elif not ids_attendus and ids:
            conseil = f" Il n'y a plus aucun client après {apres!r} : la page doit être vide."
        else:
            conseil = (" Triez par id (`ORDER BY id`), gardez les id > apres (`WHERE id > ?`), et au plus `taille` "
                       "lignes (`LIMIT ?`).")
        raise AssertionError(f"{quand} : la page contient {_court(ids, 200)} au lieu de {_court(ids_attendus, 200)}.{conseil}")
    for client, voulu in zip(page, attendu_page):
        if client != voulu:
            raise AssertionError(f"{quand} : le client {voulu['id']} vaut {_court(client)} au lieu de {_court(voulu)}.")
    if suivant != attendu_suivant:
        if attendu_suivant is None and suivant == (ids[-1] if ids else None):
            conseil = (f" Il ne reste AUCUN client après {suivant!r} : l'écran proposerait une page suivante vide. "
                       "Demandez UNE ligne de plus que la page (`LIMIT taille + 1`) : si elle arrive, il y a une suite.")
        elif attendu_suivant is not None and suivant is None:
            conseil = (f" Il reste des clients après {ids[-1]!r} : l'écran n'afficherait jamais la page suivante. "
                       "`LIMIT taille + 1` : s'il arrive une ligne de plus que `taille`, `suivant` est l'id du dernier "
                       "client GARDÉ.")
        else:
            conseil = f" `suivant` est l'id du DERNIER client de la page ({ids[-1]!r} ici), pas celui de la ligne en plus."
        raise AssertionError(f"{quand} : `suivant` vaut {suivant!r} au lieu de {attendu_suivant!r}.{conseil}")
    lues = appel.espion._releve.lignes_lues
    if lues > taille + 2:
        raise AssertionError(f"{quand} : votre code a lu {lues} lignes renvoyées par la base pour en afficher "
                             f"{len(page)}. Avec 2 millions de clients, chaque page lirait toute la table. Faites trier "
                             "ET limiter par la base : `ORDER BY id LIMIT ?` avec `taille + 1`.")
    return page, suivant


def _parcourir(page_clients, chemin: Path, taille: int) -> list[str]:
    vus, apres, pages = [], None, 0
    while True:
        page, apres = _page(page_clients, chemin, apres, taille)
        vus += [c["id"] for c in page]
        pages += 1
        if apres is None or pages > 200:
            return vus


def verifier_page_clients(page_clients) -> None:
    with _base(_remplir_93(45)) as chemin:
        _page(page_clients, chemin, None, 20)
        _page(page_clients, chemin, "C020", 20)
        _page(page_clients, chemin, "C040", 20)                  # dernière page, incomplète
        _page(page_clients, chemin, "C045", 20)                  # rien après le dernier
        _page(page_clients, chemin, None, 1)
        _page(page_clients, chemin, None, 100)                   # tout tient dans une page
        _page(page_clients, chemin, "C0105", 3)                  # un curseur qui n'est pas un id existant
        tous = _parcourir(page_clients, chemin, 7)
        if tous != [f"C{k:03d}" for k in range(1, 46)]:
            raise AssertionError(f"En suivant `suivant` page après page (taille 7), on voit {_court(tous, 300)} : "
                                 "chaque client doit apparaître une fois, dans l'ordre.")

        # Entre deux pages, un client de la 1re page part et deux arrivent : la suite doit rester juste.
        _page(page_clients, chemin, None, 10)
        conn = sqlite3.connect(chemin)
        with conn:
            conn.execute("DELETE FROM clients WHERE id IN ('C004', 'C010')")
            conn.executemany("INSERT INTO clients (id, nom, score) VALUES (?, ?, ?)",
                             [("C000", "Arrivé au début", 0.5), ("C0105", "Arrivé au milieu", 0.5)])
        conn.close()
        _page(page_clients, chemin, "C010", 10)                  # l'id du curseur vient d'être supprimé

        for taille in (0, -5, 101, 1000):
            quand = f"page_clients(conn, taille={taille})"
            appel = _appeler(page_clients, chemin, None, taille, quand=quand)
            if not isinstance(appel.erreur, ValueError):
                obtenu = f"lève {type(appel.erreur).__name__}" if appel.erreur else f"renvoie {_court(appel.resultat, 80)}"
                raise AssertionError(f"{quand} {obtenu} au lieu de lever une ValueError : la taille va de 1 à 100. Un "
                                     "`?taille=1000000` dans l'adresse ne doit pas faire lire la table entière.")
            if appel.espion.requetes():
                raise AssertionError(f"{quand} lève bien une ValueError, mais APRÈS avoir interrogé la base. Vérifiez "
                                     "la taille en PREMIER : une demande absurde ne doit rien coûter au serveur.")

        _page(page_clients, chemin, "C010'", 5)                  # entrée non fiable : une apostrophe
        _page(page_clients, chemin, "C030' --", 5)

    with _base(_remplir_93(40)) as chemin:                       # la dernière page est pleine
        _page(page_clients, chemin, None, 20)
        _page(page_clients, chemin, "C020", 20)
    with _base(_remplir_93(0)) as chemin:                        # aucun client
        _page(page_clients, chemin, None, 20)


# ============================================================================ Défi 94 : N+1

_MOIS = [f"2026-{m:02d}" for m in range(1, 11)]


def _remplir_94(conn: sqlite3.Connection) -> None:
    creer_schema(conn)
    hasard = random.Random(94)
    conn.executemany("INSERT INTO clients (id, nom, score) VALUES (?, ?, 0.5)",
                     [(f"C{k:03d}", f"Client n°{k}") for k in range(1, 601)])
    factures = []
    for k in range(1, 601):
        n = 0 if k in (4, 12, 40) else hasard.randint(0, 4)
        n = 3 if k in (1, 2, 3) else n
        factures += [(f"C{k:03d}", mois, round(hasard.uniform(15, 120), 2)) for mois in hasard.sample(_MOIS, n)]
    factures.append(("C777", "2026-05", 50.0))               # un client que personne ne demande
    hasard.shuffle(factures)                                 # dans le désordre : seul ORDER BY trie
    conn.executemany("INSERT INTO factures (client_id, mois, montant) VALUES (?, ?, ?)", factures)


def _attendu_94(chemin: Path, ids: list) -> dict:
    conn = sqlite3.connect(chemin)
    try:
        return {i: [{"mois": m, "montant": x} for m, x in conn.execute(
            "SELECT mois, montant FROM factures WHERE client_id = ? ORDER BY mois", (i,))] for i in dict.fromkeys(ids)}
    finally:
        conn.close()


_GENERATEUR = (" `ids` est ici un GÉNÉRATEUR, qu'on ne peut parcourir qu'UNE fois : si votre code l'a déjà parcouru "
               "(pour fabriquer les « ? », ou pour compter), il est épuisé, et la suite le voit vide. Commencez par "
               "`ids = list(dict.fromkeys(ids))`.")


def _factures(factures_par_client, chemin: Path, ids, description: str, ids_liste: list | None = None) -> None:
    quand = f"factures_par_client(conn, {description})"
    generateur = ids_liste is not None and not isinstance(ids, (list, tuple, set, frozenset, dict))
    ids_liste = list(ids) if ids_liste is None else ids_liste
    appel = _appeler(factures_par_client, chemin, ids, quand=quand, limite=2000)
    if generateur and appel.erreur is not None:
        raise AssertionError(f"{quand} lève {type(appel.erreur).__name__}: {str(appel.erreur).rstrip('.')}.{_GENERATEUR}")
    _requete_plantee(appel, quand, ids_liste)
    if appel.erreur is not None:
        conseil = (" Un id SANS facture doit avoir sa clé avec une liste vide : préparez `{id: [] for id in ids}` "
                   "AVANT de ranger les factures.") if isinstance(appel.erreur, KeyError) else ""
        raise AssertionError(f"{quand} lève {type(appel.erreur).__name__}: {appel.erreur}.{conseil}")
    _verifier_rendue_propre(appel, quand)
    obtenu, attendu = appel.resultat, _attendu_94(chemin, ids_liste)
    if not isinstance(obtenu, dict):
        raise AssertionError(f"{quand} renvoie {_court(obtenu)} au lieu d'un dictionnaire {{id: [factures]}}.")
    manquants = [i for i in attendu if i not in obtenu]
    if manquants:
        sans = [i for i in manquants if not attendu[i]]
        if generateur:
            conseil = _GENERATEUR
        elif sans and len(sans) == len(manquants):
            conseil = (f" {_court(sans, 120)} n'a aucune facture : il doit quand même avoir sa clé, avec `[]`. "
                       "L'écran n'a pas à tester `if id in resultat`. Préparez `{id: [] for id in ids}` d'abord.")
        else:
            conseil = " Chaque id demandé doit être une clé du résultat."
        raise AssertionError(f"{quand} : il manque les clés {_court(manquants, 160)}.{conseil}")
    en_trop = [i for i in obtenu if i not in attendu]
    if en_trop:
        raise AssertionError(f"{quand} : clés en trop : {_court(en_trop)}. Seuls les id DEMANDÉS sont des clés.")
    for i, voulues in attendu.items():
        recues = obtenu[i]
        if not isinstance(recues, list) or not all(isinstance(f, dict) and set(f) == {"mois", "montant"} for f in recues):
            raise AssertionError(f"{quand} : pour {i!r}, on obtient {_court(recues)}. Chaque facture est un dictionnaire "
                                 "`{\"mois\": …, \"montant\": …}`, dans une liste.")
        if recues == voulues:
            continue
        if len(recues) == 2 * len(voulues) and all(f in voulues for f in recues):
            conseil = (f" Chaque facture apparaît deux fois : {i!r} est présent deux fois dans `ids`. Retirez les "
                       "doublons d'abord (`dict.fromkeys(ids)`).")
        elif sorted(recues, key=lambda f: f["mois"]) == voulues:
            conseil = " Ce sont les bonnes factures, mais pas triées par mois : `ORDER BY client_id, mois`."
        elif not voulues:
            conseil = f" {i!r} n'a aucune facture : sa liste doit être vide."
        else:
            conseil = ""
        raise AssertionError(f"{quand} : pour {i!r}, on obtient {_court(recues, 220)} au lieu de {_court(voulues, 220)}.{conseil}")
    lectures = appel.espion.requetes("lecture")
    if len(lectures) > 2:
        raise AssertionError(
            f"{quand} : le résultat est juste, mais votre code a exécuté {len(lectures)} requêtes SELECT pour "
            f"{len(attendu)} clients (la 1re : {_court(lectures[0], 120)}). C'est le problème N+1 : pour 500 clients, "
            "500 allers-retours vers la base ; à 1 ms chacun sur un vrai serveur, la page met une demi-seconde. Une "
            "seule requête pour tous : `WHERE client_id IN (?, ?, …)` avec autant de « ? » que d'id.")


def verifier_factures_par_client(factures_par_client) -> None:
    with _base(_remplir_94) as chemin:
        _factures(factures_par_client, chemin, ["C001", "C004", "C002"], "['C001', 'C004', 'C002']")
        _factures(factures_par_client, chemin, ["C003"], "['C003']")
        _factures(factures_par_client, chemin, [], "[]")
        _factures(factures_par_client, chemin, ["C999"], "['C999'] (client inconnu)")
        _factures(factures_par_client, chemin, ["C001", "C001", "C003"], "['C001', 'C001', 'C003']")
        _factures(factures_par_client, chemin, {"C002", "C012"}, "{'C002', 'C012'} (un ensemble)", ["C002", "C012"])
        ids = ["C010", "C011", "C012"]
        _factures(factures_par_client, chemin, (i for i in ids), "un GÉNÉRATEUR de ['C010', 'C011', 'C012']", ids)
        _factures(factures_par_client, chemin, ["C001", "C9' OR '1'='1", "L'Hôte"], "['C001', \"C9' OR '1'='1\", \"L'Hôte\"]")
        cinq_cents = [f"C{k:03d}" for k in range(1, 501)]
        _factures(factures_par_client, chemin, cinq_cents, "500 clients ['C001', …, 'C500']")


# ============================================================================ Défi 95 : migrations

_M = [
    ("CREATE TABLE clients (id TEXT PRIMARY KEY, nom TEXT NOT NULL)",),
    ["ALTER TABLE clients ADD COLUMN score REAL NOT NULL DEFAULT 0",
     "CREATE INDEX clients_par_score ON clients (score)"],
    ("CREATE TABLE offres (id INTEGER PRIMARY KEY, client_id TEXT NOT NULL UNIQUE, montant REAL NOT NULL)",
     "INSERT INTO offres (client_id, montant) VALUES ('C001', 30.0)"),
    ("CREATE TABLE taches (id INTEGER PRIMARY KEY, client_id TEXT NOT NULL)",),
]
_CASSEE = (
    "CREATE TABLE journal (id INTEGER PRIMARY KEY, message TEXT)",
    "INSERT INTO journal (message) VALUES ('migration 3')",
    "ALTER TABLE table_qui_n_existe_pas ADD COLUMN x TEXT",
)


def _etat(conn: sqlite3.Connection) -> tuple:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    schema = sorted(conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"))
    tables = [nom for type_, nom, _, _ in schema if type_ == "table"]
    donnees = {t: sorted(conn.execute(f'SELECT * FROM "{t}"').fetchall(), key=repr) for t in tables}
    return version, schema, donnees


def _etat_fichier(chemin: Path) -> tuple:
    conn = sqlite3.connect(chemin)
    try:
        return _etat(conn)
    finally:
        conn.close()


def _etat_reference(migrations, version: int) -> tuple:
    """L'état d'une base neuve après les `version` premières migrations, appliquées correctement."""
    conn = sqlite3.connect(":memory:")
    try:
        for i in range(version):
            conn.execute("BEGIN")
            for instruction in migrations[i]:
                conn.execute(instruction)
            conn.execute(f"PRAGMA user_version = {i + 1}")
            conn.commit()
        return _etat(conn)
    finally:
        conn.close()


def _decrire_etat(etat: tuple) -> str:
    version, schema, _ = etat
    objets = [f"{type_} {nom}" for type_, nom, _, _ in schema]
    return f"version {version}, {', '.join(objets) or 'aucune table'}"


def _quelle_version(etat: tuple, migrations) -> int | None:
    """Le nombre de migrations dont le schéma correspond (en ignorant le numéro de version), ou None."""
    for k in range(len(migrations) + 1):
        if _etat_reference(migrations, k)[1:] == etat[1:]:
            return k
    return None


def _preparer_95(version: int, migrations=_M):
    def remplir(conn: sqlite3.Connection) -> None:
        for i in range(version):
            conn.execute("BEGIN")
            for instruction in migrations[i]:
                conn.execute(instruction)
            conn.execute(f"PRAGMA user_version = {i + 1}")
            conn.commit()
    return remplir


def _a_utilise_executescript(appel: _Appel) -> bool:
    return any(quoi == "executescript" for quoi, _ in appel.espion._releve.journal)


def _migrer_ok(migrer, chemin: Path, migrations, attendu_nombre: int, version_finale: int, quand: str) -> None:
    avant = _etat_fichier(chemin)
    appel = _appeler(migrer, chemin, migrations, quand=quand)
    if appel.erreur is not None:
        if "already exists" in str(appel.erreur) or "duplicate column" in str(appel.erreur):
            raise AssertionError(
                f"{quand} : la base était en version {avant[0]}, et votre fonction lève "
                f"{type(appel.erreur).__name__}: {appel.erreur}. Vous rejouez des migrations DÉJÀ appliquées : lisez "
                "la version (`conn.execute(\"PRAGMA user_version\").fetchone()[0]`) et commencez à "
                "`range(version, len(migrations))`. Et la version doit être écrite à chaque migration réussie.")
        if isinstance(appel.erreur, sqlite3.OperationalError) and "transaction" in str(appel.erreur):
            raise AssertionError(f"{quand} lève {type(appel.erreur).__name__}: {appel.erreur}. Un `BEGIN` alors qu'une "
                                 "transaction est déjà ouverte : faites `commit()` (ou `rollback()`) à la fin de CHAQUE "
                                 "migration, avant le `BEGIN` de la suivante.")
        raise AssertionError(f"{quand} : la base était en version {avant[0]}, et votre fonction lève "
                             f"{type(appel.erreur).__name__}: {appel.erreur}.")
    _verifier_rendue_propre(appel, quand)
    apres = _etat_fichier(chemin)
    attendu = _etat_reference(migrations, version_finale)
    if apres != attendu:
        schema = _quelle_version(apres, migrations)
        if schema == version_finale and apres[0] != version_finale:
            conseil = (f" Le schéma est bon, mais `PRAGMA user_version` vaut {apres[0]} : écrivez "
                       "`conn.execute(f\"PRAGMA user_version = {i + 1}\")` à chaque migration.")
        else:
            conseil = ""
        raise AssertionError(f"{quand} : la base était en version {avant[0]} ; après l'appel elle est en "
                             f"« {_decrire_etat(apres)} » au lieu de « {_decrire_etat(attendu)} ».{conseil}")
    if isinstance(appel.resultat, bool) or appel.resultat != attendu_nombre:
        raise AssertionError(f"{quand} renvoie {_court(appel.resultat)} au lieu de {attendu_nombre} : le NOMBRE de "
                             f"migrations appliquées par CET appel (la base était en version {avant[0]}).")


def verifier_migrer(migrer) -> None:
    with _base(None) as chemin:
        _migrer_ok(migrer, chemin, _M, 4, 4, "Sur une base neuve, migrer(conn, 4 migrations)")
        _migrer_ok(migrer, chemin, _M, 0, 4, "Une 2e fois, sur la base déjà à jour, migrer(conn, 4 migrations)")
        _migrer_ok(migrer, chemin, _M + [("CREATE TABLE notes (id INTEGER PRIMARY KEY)",)], 1, 5,
                   "Quand une 5e migration arrive, migrer(conn, 5 migrations)")
    with _base(_preparer_95(1)) as chemin:
        _migrer_ok(migrer, chemin, _M, 3, 4, "Sur une base en version 1, migrer(conn, 4 migrations)")
    with _base(None) as chemin:
        _migrer_ok(migrer, chemin, [], 0, 0, "Sur une base neuve, migrer(conn, [])")

    # Une migration qui échoue à sa 3e instruction.
    with _base(None) as chemin:
        migrations = [_M[0], _M[1], _CASSEE, _M[3]]
        quand = "migrer(conn, [m1, m2, m3_cassée, m4]) sur une base neuve (m3 échoue à sa 3e instruction)"
        appel = _appeler(migrer, chemin, migrations, quand=quand)
        apres = _etat_fichier(chemin)
        if appel.erreur is None:
            raise AssertionError(f"{quand} : votre fonction renvoie {_court(appel.resultat)} sans lever d'erreur. "
                                 "L'erreur doit remonter telle quelle (`raise`) : le déploiement doit s'arrêter, et "
                                 f"quelqu'un doit corriger la migration. (Base obtenue : {_decrire_etat(apres)}.)")
        if not isinstance(appel.erreur, sqlite3.Error):
            raise AssertionError(f"{quand} : votre fonction lève {type(appel.erreur).__name__}: {appel.erreur} au "
                                 "lieu de laisser remonter l'erreur de SQLite telle quelle (`raise` tout court).")
        _verifier_rendue_propre(appel, quand)
        bon = _etat_reference(migrations, 2)
        if apres != bon:
            noms = [nom for _, nom, _, _ in apres[1]]
            if "journal" in noms:
                if _a_utilise_executescript(appel):
                    conseil = (" Vous utilisez `executescript` : il commence par valider (COMMIT) la transaction en "
                               "cours, puis enregistre chaque instruction dès qu'elle s'exécute. Exécutez les "
                               "instructions une par une avec `conn.execute`.")
                else:
                    conseil = (" La table `journal`, créée par la 1re instruction de m3, est RESTÉE : en Python, sqlite3 "
                               "n'ouvre pas de transaction avant un CREATE TABLE, donc il est enregistré aussitôt et "
                               "`rollback()` n'a rien à annuler. Ouvrez la transaction vous-même : "
                               "`conn.execute(\"BEGIN\")` avant les instructions de chaque migration.")
                raise AssertionError(f"{quand} : la migration 3 est restée À MOITIÉ appliquée ({_decrire_etat(apres)}).{conseil}")
            if "taches" in noms:
                raise AssertionError(f"{quand} : après l'échec de m3, votre fonction a continué et appliqué m4 "
                                     f"({_decrire_etat(apres)}). Une migration suppose que les précédentes sont faites : "
                                     "on s'arrête à la première erreur.")
            schema = _quelle_version(apres, migrations)
            if schema == 2:
                raise AssertionError(f"{quand} : m1 et m2 sont bien appliquées, mais `PRAGMA user_version` vaut "
                                     f"{apres[0]} au lieu de 2. Au prochain lancement, m1 serait rejouée (« table "
                                     "clients already exists »). Écrivez la version DANS la transaction de chaque "
                                     "migration, pas une fois à la fin.")
            if schema == 0:
                raise AssertionError(f"{quand} : m1 et m2, qui avaient réussi, ont été annulées aussi "
                                     f"({_decrire_etat(apres)}). Chaque migration a SA transaction : `commit()` à la fin "
                                     "de chacune, pas une seule transaction pour tout l'appel.")
            raise AssertionError(f"{quand} : la base est en « {_decrire_etat(apres)} » au lieu de « {_decrire_etat(bon)} ».")
        _migrer_ok(migrer, chemin, _M, 2, 4, "Puis, une fois m3 corrigée, migrer(conn, [m1, m2, m3, m4])")

    # Le serveur s'arrête au moment où la version 2 est écrite.
    with _base(None) as chemin:
        quand = "Le serveur s'arrête pendant la migration 2 (au moment d'écrire sa version)"

        def couper(espion, sorte, sql):
            ecrit = re.search(r"user_version\s*=\s*(\d+)", sql, re.IGNORECASE)
            if ecrit and int(ecrit.group(1)) >= 2:
                raise _PanneServeur("panne simulée : le serveur s'arrête")

        appel = _appeler(migrer, chemin, _M, avant=couper, quand=quand, simuler_commit=False)  # fermée comme à l'arrêt
        apres = _etat_fichier(chemin)
        schema = _quelle_version(apres, _M)
        if schema != apres[0]:
            raise AssertionError(
                f"{quand} : au redémarrage, la base dit « version {apres[0]} » mais contient le schéma de la version "
                f"{schema if schema is not None else '?'} ({_decrire_etat(apres)}). Au prochain lancement, une "
                "migration déjà faite serait rejouée (« table already exists »), ou une autre sautée. Les instructions "
                "d'une migration ET `PRAGMA user_version = …` doivent être validées par le MÊME commit : "
                "`BEGIN`, les instructions, le PRAGMA, puis `commit()`.")
        if appel.erreur is None:
            raise AssertionError(f"{quand} : la panne ne remonte pas : votre fonction renvoie {_court(appel.resultat)}. "
                                 "Ne cachez pas une erreur pendant une migration : `rollback()` puis `raise`.")
        _migrer_ok(migrer, chemin, _M, 4 - apres[0], 4, f"{quand}, puis redémarre et relance migrer(conn, 4 migrations)")

    # Une base plus récente que le programme.
    with _base(_preparer_95(4)) as chemin:
        quand = "Sur une base en version 4, migrer(conn, 3 migrations) (programme plus ancien que la base)"
        avant = _etat_fichier(chemin)
        appel = _appeler(migrer, chemin, _M[:3], quand=quand)
        if not isinstance(appel.erreur, ValueError):
            obtenu = (f"lève {type(appel.erreur).__name__}: {appel.erreur}" if appel.erreur
                      else f"renvoie {_court(appel.resultat)}")
            raise AssertionError(f"{quand} : votre fonction {obtenu}, au lieu de lever une ValueError. La base a été "
                                 "migrée par un programme plus récent : `if version > len(migrations): raise ValueError(…)`.")
        _verifier_rendue_propre(appel, quand)
        if _etat_fichier(chemin) != avant:
            raise AssertionError(f"{quand} : la ValueError est levée, mais la base a changé ({_decrire_etat(_etat_fichier(chemin))}). "
                                 "Vérifiez la version AVANT de toucher à quoi que ce soit.")


# ============================================================================ Défi 96 : verrou optimiste


def _remplir_96(conn: sqlite3.Connection) -> None:
    creer_schema(conn)
    conn.executemany("INSERT INTO clients VALUES (?, ?, ?, ?, ?)", _CLIENTS)


def _fiche(photo: dict, client: str):
    return next(((statut, version) for i, _, _, statut, version in photo["clients"] if i == client), None)


def _collegue_96(chemin: Path, client: str, nouveau: str):
    """Un collègue qui enregistre `nouveau` sur la fiche, juste avant VOTRE écriture, si vous avez lu la fiche avant."""
    etat = {"a_ecrit": False, "bloque": False}

    def avant(espion, sorte, sql):
        if etat["a_ecrit"] or etat["bloque"] or _sorte_requete(sql) != "ecriture" or not _touche(sql, "clients"):
            return
        if not any(_touche(s, "clients") for s in espion.requetes("lecture")):
            return  # vous n'avez rien lu avant d'écrire : pas de fenêtre où se glisser
        collegue = sqlite3.connect(chemin, timeout=0)
        try:
            with collegue:
                collegue.execute("UPDATE clients SET statut = ?, version = version + 1 WHERE id = ?", (nouveau, client))
            etat["a_ecrit"] = True
        except sqlite3.OperationalError:
            etat["bloque"] = True  # vous avez verrouillé la base (BEGIN IMMEDIATE…) : le collègue attendra son tour
        finally:
            collegue.close()

    return avant, etat


def _changer(changer_statut, chemin: Path, client, statut, version, avant=None):
    quand = f"changer_statut(conn, {client!r}, {statut!r}, version_lue={version})"
    photo_avant = _photo(chemin)
    appel = _appeler(changer_statut, chemin, client, statut, version, avant=avant, quand=quand)
    return quand, photo_avant, appel


def _changement_ok(changer_statut, chemin, client, statut, version) -> None:
    quand, avant, appel = _changer(changer_statut, chemin, client, statut, version)
    _requete_plantee(appel, quand, (client, statut))
    if appel.erreur is not None:
        conseil = ""
        if isinstance(appel.erreur, ConflitDeVersion):
            conseil = f" La fiche est bien en version {_fiche(avant, client)[1]} : ce n'est pas un conflit."
        raise AssertionError(f"{quand} devrait réussir (la fiche est en version {version}), mais lève "
                             f"{type(appel.erreur).__name__}: {appel.erreur}.{conseil}")
    _verifier_rendue_propre(appel, quand, avant)
    apres = _photo(chemin)
    if _fiche(apres, client) != (statut, version + 1):
        conseil = (" La version doit augmenter de 1 À CHAQUE enregistrement (`version = version + 1` dans l'UPDATE) : "
                   "c'est elle qui permet au prochain enregistrement de détecter le vôtre."
                   if _fiche(apres, client) == (statut, version) else "")
        raise AssertionError(f"{quand} : la fiche vaut {_fiche(apres, client)!r} (statut, version) au lieu de "
                             f"{(statut, version + 1)!r}.{conseil}")
    autres_avant = [ligne for ligne in avant["clients"] if ligne[0] != client]
    autres_apres = [ligne for ligne in apres["clients"] if ligne[0] != client]
    if autres_avant != autres_apres:
        raise AssertionError(f"{quand} : d'AUTRES fiches ont changé : {_differences({'clients': autres_avant}, {'clients': autres_apres})}. "
                             "Votre UPDATE a-t-il un `WHERE id = ?` ?")
    if isinstance(appel.resultat, bool) or appel.resultat != version + 1:
        raise AssertionError(f"{quand} renvoie {_court(appel.resultat)} au lieu de la NOUVELLE version ({version + 1}).")


def _changement_refuse(changer_statut, chemin, client, statut, version, attendue, pourquoi, conseil="") -> None:
    quand, avant, appel = _changer(changer_statut, chemin, client, statut, version)
    if appel.erreur is None:
        apres = _photo(chemin) if not appel.en_transaction else appel.photo_si_commit
        change = f" Et la base a changé : {_differences(avant, apres)}." if apres != avant else ""
        raise AssertionError(f"{quand} : {pourquoi} ; votre fonction devait lever {attendue.__name__}, mais renvoie "
                             f"{_court(appel.resultat)}.{change}{conseil}")
    if not isinstance(appel.erreur, attendue):
        _requete_plantee(appel, quand, (client, statut))
        raise AssertionError(f"{quand} : {pourquoi} ; on attend {attendue.__name__}, mais votre fonction lève "
                             f"{type(appel.erreur).__name__}: {appel.erreur}.{conseil}")
    _verifier_rendue_propre(appel, quand, avant)
    apres = _photo(chemin)
    if apres != avant:
        raise AssertionError(f"{quand} : {attendue.__name__} est bien levée, mais la base a changé : "
                             f"{_differences(avant, apres)}. Un enregistrement refusé ne modifie RIEN.")


def verifier_changer_statut(changer_statut) -> None:
    with _base(_remplir_96) as chemin:
        _changement_ok(changer_statut, chemin, "C001", "contacte", 1)
        _changement_refuse(changer_statut, chemin, "C001", "parti", 1, ConflitDeVersion,
                           "la fiche est passée en version 2 depuis cette lecture (version 1)",
                           " Enregistrez seulement si la version n'a pas changé : `… WHERE id = ? AND version = ?`, puis "
                           "regardez `curseur.rowcount` (0 = rien n'a été modifié).")
        _changement_ok(changer_statut, chemin, "C001", "offre_envoyee", 2)
        _changement_refuse(changer_statut, chemin, "C002", "fidelise", 7, ConflitDeVersion,
                           "la fiche de C002 est en version 3, pas 7")
        _changement_refuse(changer_statut, chemin, "C999", "contacte", 1, KeyError,
                           "le client C999 n'existe pas (ce n'est pas un conflit : il n'y a rien à recharger)",
                           " Quand l'UPDATE ne modifie rien, un SELECT dit si le client existe : sinon "
                           "`raise KeyError(client_id)`.")
        _changement_refuse(changer_statut, chemin, "C0' OR '1'='1", "parti", 1, KeyError,
                           "aucun client n'a cet identifiant (il vient d'une adresse : entrée non fiable)",
                           " Collé dans le texte SQL, cet « identifiant » devient `WHERE id = 'C0' OR '1'='1' AND …` : "
                           "il modifie TOUTES les fiches. Paramètres `?` obligatoires (série 13, défi 80).")
        _changement_refuse(changer_statut, chemin, "C004", "perdu", 1, ValueError,
                           f"« perdu » n'est pas un statut ({', '.join(STATUTS)})",
                           " `if statut not in STATUTS: raise ValueError(…)` avant de toucher à la base.")

        # La course : un collègue enregistre sur la fiche de C004 entre votre lecture et votre écriture.
        collegue, etat = _collegue_96(chemin, "C004", "offre_envoyee")
        quand, avant, appel = _changer(changer_statut, chemin, "C004", "contacte", 1, avant=collegue)
        apres = _photo(chemin)
        if etat["a_ecrit"]:
            if appel.erreur is None:
                raise AssertionError(
                    f"{quand} : votre code a LU la fiche, puis un collègue a enregistré « offre_envoyee » (version 2) "
                    "juste avant votre UPDATE… que votre code a fait quand même : la fiche vaut maintenant "
                    f"{_fiche(apres, 'C004')!r}. Le travail du collègue est perdu sans que personne le sache. Comparer "
                    "la version en Python arrive une milliseconde trop tôt : la condition doit être DANS l'UPDATE "
                    "(`WHERE id = ? AND version = ?`), et `curseur.rowcount` dit si l'écriture a eu lieu.")
            if not isinstance(appel.erreur, ConflitDeVersion):
                _requete_plantee(appel, quand, ("C004",))
                raise AssertionError(f"{quand} : un collègue a enregistré la fiche entre votre lecture et votre écriture ; "
                                     f"on attend ConflitDeVersion, mais votre fonction lève {type(appel.erreur).__name__}: "
                                     f"{appel.erreur}.")
            _verifier_rendue_propre(appel, quand)
            if _fiche(apres, "C004") != ("offre_envoyee", 2):
                raise AssertionError(f"{quand} : la fiche du collègue a été modifiée : {_fiche(apres, 'C004')!r}.")
        else:
            if appel.erreur is not None:
                _requete_plantee(appel, quand, ("C004",))
                raise AssertionError(f"{quand} devrait réussir, mais lève {type(appel.erreur).__name__}: {appel.erreur}.")
            _verifier_rendue_propre(appel, quand)
            if _fiche(apres, "C004") != ("contacte", 2):
                raise AssertionError(f"{quand} : la fiche vaut {_fiche(apres, 'C004')!r} au lieu de ('contacte', 2).")


# ============================================================================ Défi 97 : file de tâches

_TACHES = [  # id, client_id, priorite, statut, travailleur, pris_a, essais
    (1, "C001", 0, "a_faire", None, None, 0),
    (2, "C002", 5, "a_faire", None, None, 0),
    (3, "C003", 5, "a_faire", None, None, 0),
    (4, "C004", 9, "faite", "poste-1", 400.0, 1),
    (5, "C005", 1, "en_cours", "poste-2", 900.0, 1),
    (6, "C006", 2, "en_cours", "poste-3", 650.0, 1),
    (7, "C007", 0, "a_faire", None, None, 0),
    (8, "C008", 7, "en_cours", "poste-2", 950.0, 2),
]


def _remplir_97(taches=_TACHES):
    def remplir(conn: sqlite3.Connection) -> None:
        creer_schema(conn)
        conn.executemany("INSERT INTO taches VALUES (?, ?, ?, ?, ?, ?, ?)", taches)
    return remplir


def _disponibles(photo: dict, maintenant: float, bail: float) -> list[tuple]:
    libres = [t for t in photo["taches"]
              if t[3] == "a_faire" or (t[3] == "en_cours" and t[5] is not None and t[5] + bail <= maintenant)]
    return sorted(libres, key=lambda t: (-t[2], t[0]))


def _reclamer(chemin: Path, travailleur: str, maintenant: float, bail: float) -> int | None:
    """Ce que fait un AUTRE travailleur, correctement, sur sa propre connexion."""
    conn = sqlite3.connect(chemin, timeout=0)
    try:
        dispo = "(statut = 'a_faire' OR (statut = 'en_cours' AND pris_a + ? <= ?))"
        with conn:
            ligne = conn.execute(f"SELECT id FROM taches WHERE {dispo} ORDER BY priorite DESC, id LIMIT 1",
                                 (bail, maintenant)).fetchone()
            if ligne is None:
                return None
            conn.execute(f"UPDATE taches SET statut = 'en_cours', travailleur = ?, pris_a = ?, essais = essais + 1 "
                         f"WHERE id = ? AND {dispo}", (travailleur, maintenant, ligne[0], bail, maintenant))
        return ligne[0]
    finally:
        conn.close()


def _decrire_tache(t: tuple) -> str:
    id_, client, prio, statut, travailleur, pris_a, essais = t
    suite = f", prise par {travailleur!r} à {pris_a}" if statut != "a_faire" else ""
    return f"tâche {id_} ({client}, priorité {prio}, {statut}{suite}, {essais} essai{'s' if essais > 1 else ''})"


def _prendre(prendre_tache, chemin: Path, travailleur: str, maintenant: float, bail: float | None = None,
             avant=None, quand: str | None = None):
    args = (travailleur, maintenant) if bail is None else (travailleur, maintenant, bail)
    quand = quand or (f"prendre_tache(conn, {travailleur!r}, maintenant={maintenant}"
                      + ("" if bail is None else f", bail={bail}") + ")")
    photo_avant = _photo(chemin)
    appel = _appeler(prendre_tache, chemin, *args, avant=avant, limite=200, quand=quand)
    _requete_plantee(appel, quand, (travailleur,))
    if appel.erreur is not None:
        raise AssertionError(f"{quand} lève {type(appel.erreur).__name__}: {appel.erreur}.")
    _verifier_rendue_propre(appel, quand, photo_avant)
    return quand, photo_avant, appel


def _verifier_prise(quand, avant, appel, apres, attendue, travailleur, maintenant, bail) -> None:
    resultat = appel.resultat
    taches_avant = {t[0]: t for t in avant["taches"]}
    taches_apres = {t[0]: t for t in apres["taches"]}
    changees = [i for i in taches_avant if taches_avant[i] != taches_apres.get(i)]
    if attendue is None:
        if resultat is not None:
            raise AssertionError(f"{quand} renvoie {_court(resultat)} alors qu'aucune tâche n'est disponible : il faut "
                                 f"None. (Disponible : « a_faire », ou « en_cours » avec `pris_a + bail <= maintenant`.)")
        if changees:
            raise AssertionError(f"{quand} : aucune tâche n'était disponible, mais la base a changé : "
                                 f"{_differences(avant, apres)}.")
        return
    tache = taches_avant[attendue]
    if resultat is None:
        if tache[3] == "en_cours" and tache[5] + bail == maintenant:
            pourquoi = (f" : son bail expire PILE maintenant ({tache[5]} + {bail} = {maintenant}). La règle est "
                        "`pris_a + bail <= maintenant`, avec <=.")
        elif tache[3] == "en_cours":
            pourquoi = (f" : son bail a expiré ({tache[5]} + {bail} <= {maintenant}), son travailleur est sans doute "
                        "tombé en panne.")
        else:
            pourquoi = "."
        raise AssertionError(f"{quand} renvoie None, alors que la {_decrire_tache(tache)} est disponible{pourquoi}")
    if not isinstance(resultat, dict) or set(resultat) != {"id", "client_id", "essais"}:
        raise AssertionError(f"{quand} renvoie {_court(resultat)} au lieu d'un dictionnaire "
                             "`{\"id\": …, \"client_id\": …, \"essais\": …}`.")
    if resultat["id"] != attendue:
        prise = taches_avant.get(resultat["id"])
        if prise is None:
            conseil = ""
        elif prise[3] == "faite":
            conseil = " Une tâche « faite » n'est jamais reprise."
        elif prise[3] == "en_cours" and prise[5] + bail > maintenant:
            conseil = (f" Celle-ci est « en_cours » et son bail court encore ({prise[5]} + {bail} > {maintenant}) : son "
                       "travailleur est peut-être en train d'envoyer le SMS.")
        elif tache[3] == "en_cours" and tache[5] + bail == maintenant:
            conseil = (f" La tâche {attendue} est disponible : son bail expire PILE maintenant "
                       f"({tache[5]} + {bail} = {maintenant}). La règle est `pris_a + bail <= maintenant`, avec <=.")
        elif tache[3] == "en_cours":
            conseil = (f" La tâche {attendue} est disponible : son travailleur ne l'a pas finie à temps "
                       f"({tache[5]} + {bail} <= {maintenant}), on considère qu'il est tombé en panne.")
        else:
            conseil = (" Ordre : la plus grande priorité d'abord, puis le plus petit id : "
                       "`ORDER BY priorite DESC, id`.")
        raise AssertionError(f"{quand} renvoie la tâche {resultat['id']} au lieu de la {_decrire_tache(tache)}.{conseil}")
    attendu_apres = (tache[0], tache[1], tache[2], "en_cours", travailleur, maintenant, tache[6] + 1)
    if taches_apres[attendue] != attendu_apres:
        obtenu = taches_apres[attendue]
        if obtenu[6] == tache[6]:
            conseil = " `essais` doit augmenter de 1 (`essais = essais + 1`)."
        elif obtenu[5] != maintenant:
            conseil = (" `pris_a` doit valoir `maintenant` (le correcteur avance l'horloge à la main) : n'utilisez pas "
                       "`time.time()`.")
        else:
            conseil = ""
        raise AssertionError(f"{quand} : dans la base, la tâche {attendue} vaut {obtenu!r} au lieu de {attendu_apres!r} "
                             f"(id, client, priorité, statut, travailleur, pris_a, essais).{conseil}")
    if resultat != {"id": attendue, "client_id": tache[1], "essais": tache[6] + 1}:
        raise AssertionError(f"{quand} renvoie {_court(resultat)} au lieu de "
                             f"{ {'id': attendue, 'client_id': tache[1], 'essais': tache[6] + 1} } (essais APRÈS la prise).")
    if changees != [attendue]:
        autres = [taches_avant[i] for i in changees if i != attendue]
        raise AssertionError(f"{quand} : d'AUTRES tâches ont changé en même temps ({', '.join(f'tâche {t[0]}' for t in autres)}) : "
                             f"{_differences({'taches': autres}, {'taches': [taches_apres[t[0]] for t in autres]})}. On prend UNE "
                             "tâche : votre UPDATE doit viser son id (`WHERE id = ? AND …`).")


def _suite_97(prendre_tache, chemin, travailleur, maintenant, bail=None) -> None:
    bail_effectif = BAIL if bail is None else bail
    avant = _photo(chemin)
    dispo = _disponibles(avant, maintenant, bail_effectif)
    attendue = dispo[0][0] if dispo else None
    quand, avant, appel = _prendre(prendre_tache, chemin, travailleur, maintenant, bail)
    _verifier_prise(quand, avant, appel, _photo(chemin), attendue, travailleur, maintenant, bail_effectif)


def verifier_prendre_tache(prendre_tache) -> None:
    with _base(_remplir_97()) as chemin:
        for _ in range(6):                                    # 2, 3, 6 (bail expiré), 1, 7, puis None
            _suite_97(prendre_tache, chemin, "poste d'Inès", 1000.0)
        _suite_97(prendre_tache, chemin, "poste-4", 1200.0)    # la 5 (prise à 900) est libérée, pile à l'heure
        _suite_97(prendre_tache, chemin, "poste-4", 1200.0)    # plus rien
        _suite_97(prendre_tache, chemin, "poste-4", 1250.0)    # la 8 (prise à 950)
        _suite_97(prendre_tache, chemin, "poste-4", 1300.0)    # celles prises à 1000 sont reprises
    with _base(_remplir_97()) as chemin:                       # un autre bail
        _suite_97(prendre_tache, chemin, "poste-5", 1000.0, bail=60.0)
        _suite_97(prendre_tache, chemin, "poste-5", 1000.0, bail=60.0)
        _suite_97(prendre_tache, chemin, "poste-5", 1010.0, bail=60.0)
    with _base(_remplir_97([(1, "C001", 0, "faite", "poste-1", 400.0, 1)])) as chemin:
        _suite_97(prendre_tache, chemin, "poste-1", 5000.0)

    # La course : un autre travailleur prend la meilleure tâche entre votre SELECT et votre UPDATE.
    with _base(_remplir_97()) as chemin:
        etat = {"pris": None, "bloque": False}

        def concurrent(espion, sorte, sql):
            if etat["pris"] is not None or etat["bloque"] or _sorte_requete(sql) != "ecriture" or not _touche(sql, "taches"):
                return
            if not any(_touche(s, "taches") for s in espion.requetes("lecture")):
                return  # vous n'avez rien lu avant d'écrire : pas de fenêtre où se glisser
            try:
                etat["pris"] = _reclamer(chemin, "poste-concurrent", 1000.0, BAIL)
            except sqlite3.OperationalError:
                etat["bloque"] = True  # vous avez verrouillé la base : l'autre attend son tour

        appel_seul = "prendre_tache(conn, 'poste-1', maintenant=1000.0)"
        quand = f"{appel_seul}, pendant qu'un autre travailleur prend la tâche 2 entre votre SELECT et votre UPDATE"
        avant = _photo(chemin)
        appel = _appeler(prendre_tache, chemin, "poste-1", 1000.0, avant=concurrent, limite=200, quand=quand)
        _requete_plantee(appel, quand, ("poste-1",))
        if appel.erreur is not None:
            raise AssertionError(f"{quand} lève {type(appel.erreur).__name__}: {appel.erreur}.")
        _verifier_rendue_propre(appel, quand, avant)
        apres = _photo(chemin)
        taches = {t[0]: t for t in apres["taches"]}
        if etat["pris"] is None:
            _verifier_prise(appel_seul, avant, appel, apres, 2, "poste-1", 1000.0, BAIL)
            return
        if taches[2][4] != "poste-concurrent" or (appel.resultat or {}).get("id") == 2:
            raise AssertionError(
                f"{quand} : l'autre travailleur a pris la tâche 2, puis votre code l'a prise AUSSI "
                f"(dans la base : {_decrire_tache(taches[2])}). Le client recevra deux SMS. Votre UPDATE doit RÉPÉTER "
                "la condition « disponible » : `UPDATE taches SET … WHERE id = ? AND (statut = 'a_faire' OR …)` ; "
                "si `rowcount` vaut 0, quelqu'un a été plus rapide.")
        if appel.resultat is None:
            raise AssertionError(
                f"{quand} : l'autre travailleur vous a pris la tâche 2, et votre fonction renvoie None. Mais il reste "
                "des tâches (la 3, la 6…) : quand votre UPDATE ne modifie rien, recommencez (une boucle) avec la "
                "meilleure tâche ENCORE disponible.")
        avant_sans_concurrent = dict(avant)
        avant_sans_concurrent["taches"] = [taches[2] if t[0] == 2 else t for t in avant["taches"]]
        _verifier_prise(quand, avant_sans_concurrent, appel, apres, 3, "poste-1", 1000.0, BAIL)


SERIE = Serie(
    code="15",
    titre="Enregistrer sans rien perdre : transactions, pagination, N+1, migrations, verrou optimiste, file de tâches",
    module_eleve="defis.serie_15",
    module_corrige="defis.corrige_serie_15",
    defis=(
        Defi(92, "accorder_offre", "🟢", "—", verifier_accorder_offre),
        Defi(93, "page_clients", "🟢", "—", verifier_page_clients),
        Defi(94, "factures_par_client", "🟡", "—", verifier_factures_par_client),
        Defi(95, "migrer", "🟡", "—", verifier_migrer),
        Defi(96, "changer_statut", "🟡", "—", verifier_changer_statut),
        Defi(97, "prendre_tache", "🔴", "—", verifier_prendre_tache),
    ),
)

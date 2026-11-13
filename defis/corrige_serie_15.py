"""Série 15 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_15.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la situation à laquelle vous n'aviez pas pensé.

Fil rouge de la série : la base est PARTAGÉE (avec le reste du programme, avec des collègues, avec
d'autres travailleurs). Trois réflexes reviennent dans chaque défi :

1. tracer soi-même les bords de la transaction (commit / rollback), et ne jamais rendre une
   connexion avec une transaction ouverte ;
2. faire faire à la base ce qu'elle seule peut faire sans être interrompue : vérifier ET écrire
   dans la même instruction (WHERE … AND version = ?), plutôt que lire, décider en Python, écrire ;
3. des paramètres `?` partout, et du SQL fabriqué uniquement à partir de morceaux écrits par vous.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Iterable, Sequence

from defis.persistance import (  # noqa: F401
    BAIL,
    MIGRATION_CASSEE,
    MIGRATIONS_EXEMPLE,
    STATUTS,
    ConflitDeVersion,
    afficher,
    base_exemple,
    deux_connexions,
)

# ============================================================================ Défi 92 : transaction


def accorder_offre(conn: sqlite3.Connection, client_id: str, campagne: str, montant: float) -> int:
    # `not montant > 0` plutôt que `montant <= 0` : refuse aussi NaN (série 14, défi 86). Un montant
    # négatif n'est pas un détail : « reste - (-50) » AJOUTERAIT 50 € au budget.
    if not montant > 0:
        raise ValueError(f"montant invalide : {montant!r} (il doit être strictement positif)")

    # `with conn:` = commit() si le bloc se termine normalement, rollback() si une erreur en sort,
    # QUELLE QU'ELLE SOIT : notre ValueError, l'IntegrityError d'un doublon, ou une panne que
    # personne n'avait prévue. C'est ce dernier cas qui justifie la transaction.
    with conn:
        ligne = conn.execute("SELECT reste FROM budget WHERE campagne = ?", (campagne,)).fetchone()
        if ligne is None:
            raise ValueError(f"campagne inconnue : {campagne!r}")
        if ligne[0] < montant:  # dépenser exactement tout le reste est permis : < et non <=
            raise ValueError(f"budget insuffisant pour la campagne {campagne!r} : "
                             f"{ligne[0]} € restants, {montant} € demandés")
        curseur = conn.execute("INSERT INTO offres (client_id, campagne, montant) VALUES (?, ?, ?)",
                               (client_id, campagne, montant))
        conn.execute("UPDATE budget SET reste = reste - ? WHERE campagne = ?", (montant, campagne))
    return curseur.lastrowid

    # Pour aller plus loin (défi 96) : avec plusieurs connexions, deux conseillers peuvent lire
    # « 120 € restants » en même temps et dépenser 2 × 100 €. La version sûre fait vérifier la base
    # elle-même, dans l'UPDATE :
    #     UPDATE budget SET reste = reste - ? WHERE campagne = ? AND reste >= ?
    # puis `if curseur.rowcount == 0:` → campagne inconnue ou budget insuffisant (un SELECT dit lequel).


# ============================================================================ Défi 93 : pagination


def page_clients(conn: sqlite3.Connection, apres: str | None = None, taille: int = 20) -> tuple[list[dict], str | None]:
    if not 1 <= taille <= 100:
        raise ValueError(f"taille invalide : {taille!r} (de 1 à 100)")

    # On demande UNE ligne de plus que la page : si elle arrive, il existe une page suivante.
    # Deux requêtes écrites en entier plutôt qu'un morceau de WHERE ajouté à la main : chacune se
    # lit d'un coup d'œil, et aucune valeur ne s'approche du texte SQL.
    if apres is None:
        lignes = conn.execute("SELECT id, nom, score FROM clients ORDER BY id LIMIT ?",
                              (taille + 1,)).fetchall()
    else:
        lignes = conn.execute("SELECT id, nom, score FROM clients WHERE id > ? ORDER BY id LIMIT ?",
                              (apres, taille + 1)).fetchall()

    page = [{"id": id_, "nom": nom, "score": score} for id_, nom, score in lignes[:taille]]
    suivant = page[-1]["id"] if len(lignes) > taille else None
    return page, suivant

    # Avec un tri sur une colonne qui n'est pas unique (les plus gros scores d'abord), le curseur
    # devient le COUPLE (score, id) : WHERE (score < ?) OR (score = ? AND id > ?)
    # ORDER BY score DESC, id. L'id départage les égalités : sans lui, des clients de même score
    # seraient sautés ou répétés d'une page à l'autre.


# ============================================================================ Défi 94 : N+1


def factures_par_client(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, list[dict]]:
    # dict.fromkeys : une seule lecture de `ids` (un générateur ne se relit pas), doublons
    # retirés, ordre d'arrivée conservé. Une liste, parce qu'on va la parcourir deux fois.
    ids = list(dict.fromkeys(ids))
    resultat: dict[str, list[dict]] = {id_: [] for id_ in ids}  # chaque id a sa clé, même sans facture
    if not ids:
        return resultat

    # Le f-string ne colle QUE des « ? » (autant que d'id) : les valeurs voyagent en paramètres.
    marques = ", ".join("?" * len(ids))
    lignes = conn.execute(
        f"SELECT client_id, mois, montant FROM factures WHERE client_id IN ({marques}) ORDER BY client_id, mois",
        ids,
    )
    for client_id, mois, montant in lignes:
        resultat[client_id].append({"mois": mois, "montant": montant})
    return resultat

    # Très longues listes : SQLite accepte 32 766 paramètres (999 avant sa version 3.32). Au-delà,
    # on découpe en paquets (série 07, défi 47), ou on passe toute la liste en UN paramètre JSON :
    #     WHERE client_id IN (SELECT value FROM json_each(?))    avec    (json.dumps(ids),)
    # Avec un ORM (SQLAlchemy, Django), le même problème a un nom : « eager loading »
    # (selectinload, prefetch_related) contre « lazy loading », une requête par objet.


# ============================================================================ Défi 95 : migrations


def migrer(conn: sqlite3.Connection, migrations: Sequence[Sequence[str]]) -> int:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > len(migrations):
        # La base vient du futur : un collègue l'a migrée avec un programme plus récent. Y toucher
        # avec un code plus ancien, c'est la casser. On s'arrête net, sans rien modifier.
        raise ValueError(f"la base est en version {version}, ce programme ne connaît que "
                         f"{len(migrations)} migration(s) : mettez le programme à jour")

    for i in range(version, len(migrations)):
        # BEGIN explicite : sans lui, Python laisse chaque CREATE / ALTER s'enregistrer aussitôt,
        # et rollback() n'aurait plus rien à annuler. SQLite, lui, sait annuler un CREATE TABLE.
        conn.execute("BEGIN")
        try:
            for instruction in migrations[i]:
                conn.execute(instruction)  # une par une : pas d'executescript, qui ferait COMMIT
            # La nouvelle version est écrite DANS la même transaction : schéma et numéro de
            # version sont enregistrés ensemble, ou pas du tout. `i + 1` est un entier calculé
            # ici : le f-string est sans risque (PRAGMA n'accepte pas de « ? »).
            conn.execute(f"PRAGMA user_version = {i + 1}")
            conn.commit()
        except BaseException:  # même Ctrl+C : on ne laisse pas une migration à moitié faite ouverte
            conn.rollback()
            raise
    return len(migrations) - version

    # En vrai : Alembic (SQLAlchemy), les migrations de Django ou Flyway font exactement cela, avec
    # une table de versions au lieu de PRAGMA user_version. Une règle d'équipe : on ne MODIFIE
    # jamais une migration déjà partagée ; on en ajoute une nouvelle qui corrige.


# ============================================================================ Défi 96 : verrou optimiste


def changer_statut(conn: sqlite3.Connection, client_id: str, statut: str, version_lue: int) -> int:
    if statut not in STATUTS:
        raise ValueError(f"statut inconnu : {statut!r} (choix : {', '.join(STATUTS)})")

    # Vérifier ET écrire dans la MÊME instruction : la base ne laisse personne s'intercaler au
    # milieu d'un UPDATE. Lire la version d'abord (SELECT) puis écrire laisserait une fenêtre d'une
    # milliseconde à un collègue : c'est dans cette fenêtre que le correcteur le fait écrire.
    with conn:
        curseur = conn.execute(
            "UPDATE clients SET statut = ?, version = version + 1 WHERE id = ? AND version = ?",
            (statut, client_id, version_lue),
        )
    if curseur.rowcount == 1:
        return version_lue + 1

    # Rien n'a été modifié. Pourquoi ? Ce SELECT vient APRÈS l'écriture ratée : il ne sert qu'à
    # choisir le bon message, plus aucune décision d'écriture n'en dépend.
    if conn.execute("SELECT 1 FROM clients WHERE id = ?", (client_id,)).fetchone() is None:
        raise KeyError(client_id)
    raise ConflitDeVersion(f"{client_id} a changé depuis votre lecture (version {version_lue}) : rechargez la fiche")

    # L'autre famille, le verrou PESSIMISTE : bloquer la ligne pendant qu'on la lit et la modifie
    # (SELECT … FOR UPDATE en PostgreSQL, BEGIN IMMEDIATE en SQLite). Sûr, mais le collègue attend :
    # impensable quand « lire » veut dire « garder la fiche ouverte 5 minutes à l'écran ».


# ============================================================================ Défi 97 : file de tâches

# Morceau de SQL écrit par nous (aucune valeur dedans) : on peut le coller dans les deux requêtes,
# qui doivent dire EXACTEMENT la même chose de « disponible ». Les deux « ? » valent (bail, maintenant).
_DISPONIBLE = "(statut = 'a_faire' OR (statut = 'en_cours' AND pris_a + ? <= ?))"


def prendre_tache(conn: sqlite3.Connection, travailleur: str, maintenant: float, bail: float = BAIL) -> dict | None:
    while True:
        candidate = conn.execute(
            f"SELECT id FROM taches WHERE {_DISPONIBLE} ORDER BY priorite DESC, id LIMIT 1",
            (bail, maintenant),
        ).fetchone()
        if candidate is None:
            return None
        (tache_id,) = candidate

        # « Réclamer » la tâche : l'UPDATE RÉPÈTE la condition. Si un autre travailleur l'a prise
        # entre notre SELECT et cet UPDATE, elle n'est plus disponible, et rowcount vaut 0.
        with conn:
            prise = conn.execute(
                f"UPDATE taches SET statut = 'en_cours', travailleur = ?, pris_a = ?, essais = essais + 1 "
                f"WHERE id = ? AND {_DISPONIBLE}",
                (travailleur, maintenant, tache_id, bail, maintenant),
            )
        if prise.rowcount == 1:
            client_id, essais = conn.execute("SELECT client_id, essais FROM taches WHERE id = ?",
                                             (tache_id,)).fetchone()
            return {"id": tache_id, "client_id": client_id, "essais": essais}
        # Perdu : quelqu'un a été plus rapide. Il reste peut-être d'autres tâches : on recommence.
        # (La boucle finit toujours : chaque tour perdu, c'est une tâche de moins dans la file.)

    # Avec SQLite 3.35 ou plus récent (sqlite3.sqlite_version), une seule instruction suffit, sans
    # boucle, parce que la base choisit et prend la tâche sans qu'on puisse s'intercaler :
    #     UPDATE taches SET … WHERE id = (SELECT id FROM taches WHERE <disponible>
    #                                     ORDER BY priorite DESC, id LIMIT 1)
    #     RETURNING id, client_id, essais
    # PostgreSQL ajoute FOR UPDATE SKIP LOCKED au sous-SELECT, pour que les travailleurs ne
    # s'attendent même pas les uns les autres.


# ============================================================================ Tests (comme dans tests/)
# `with conn:` ne FERME pas la connexion (il valide ou annule la transaction). Pour la fermer à la fin
# du bloc : `with closing(conn):`. Un test qui oublie de fermer ses connexions finit par manquer de
# fichiers ouverts, et Windows refuse alors de supprimer la base temporaire.


def test_une_offre_refusee_ne_laisse_aucune_trace():
    with closing(base_exemple()) as conn:
        assert accorder_offre(conn, "C001", "automne", 50) == 2
        try:
            accorder_offre(conn, "C002", "automne", 20)  # C002 a déjà une offre : l'INSERT échoue
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("un doublon aurait dû être refusé")
        assert not conn.in_transaction
        assert conn.execute("SELECT reste FROM budget WHERE campagne = 'automne'").fetchone() == (450.0,)


def test_les_pages_se_suivent_sans_trou_ni_doublon():
    with closing(base_exemple()) as conn:
        vus, apres = [], None
        while True:
            page, apres = page_clients(conn, apres, taille=3)
            vus += [client["id"] for client in page]
            if apres is None:
                break
    assert vus == [f"C00{n}" for n in range(1, 9)]


def test_les_factures_arrivent_en_une_requete():
    with closing(base_exemple()) as conn:
        requetes = []
        conn.set_trace_callback(requetes.append)  # note chaque instruction SQL exécutée
        factures = factures_par_client(conn, ["C001", "C004", "C001"])
    assert len(requetes) == 1
    assert [f["mois"] for f in factures["C001"]] == ["2026-07", "2026-08", "2026-09"] and factures["C004"] == []


def test_une_migration_cassee_est_annulee_en_entier():
    with closing(sqlite3.connect(":memory:")) as conn:
        assert migrer(conn, MIGRATIONS_EXEMPLE) == 3 and migrer(conn, MIGRATIONS_EXEMPLE) == 0
        try:
            migrer(conn, MIGRATIONS_EXEMPLE + [MIGRATION_CASSEE])
        except sqlite3.OperationalError:
            pass
        assert conn.execute("PRAGMA user_version").fetchone() == (3,)
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'journal'").fetchone() is None


def test_le_second_conseiller_ne_peut_pas_ecraser_le_premier():
    lea, hugo = deux_connexions()
    with closing(lea), closing(hugo):
        assert changer_statut(hugo, "C001", "offre_envoyee", version_lue=1) == 2
        try:
            changer_statut(lea, "C001", "contacte", version_lue=1)  # Léa avait lu la version 1
        except ConflitDeVersion:
            pass
        else:
            raise AssertionError("Léa aurait dû être prévenue du conflit")
        assert lea.execute("SELECT statut, version FROM clients WHERE id = 'C001'").fetchone() == ("offre_envoyee", 2)


def test_deux_travailleurs_ne_prennent_jamais_la_meme_tache():
    poste_1, poste_2 = deux_connexions()
    with closing(poste_1), closing(poste_2):
        pris = []
        for _ in range(3):
            for conn, nom in ((poste_1, "poste-1"), (poste_2, "poste-2")):
                tache = prendre_tache(conn, nom, maintenant=1000.0)
                if tache is not None:
                    pris.append(tache["id"])
        assert pris == [2, 3, 6, 1]  # chaque tâche une seule fois ; la 5 a encore son bail
        assert prendre_tache(poste_1, "poste-1", maintenant=1200.0)["id"] == 5

"""Outils de la série 15, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Ils sont dans un fichier à part pour une raison précise : quand le correcteur attend une
`ConflitDeVersion`, il doit reconnaître EXACTEMENT la classe que votre code lève. Deux classes
écrites à l'identique dans deux fichiers sont deux classes différentes pour Python.

Tout est réimporté dans `defis/serie_15.py` : vous n'avez pas besoin d'ouvrir ce fichier.

La base de la série est celle de l'application de rétention : des clients, leurs factures, le budget
des campagnes, les offres accordées et la file des SMS de relance à envoyer.

    >>> conn = base_exemple()
    >>> afficher(conn, "SELECT * FROM budget")
    campagne     | reste
    -------------+------
    automne      | 500.0
    fidelite     | 120.0
    noël d'hiver | 200.0
    (3 lignes)
"""

from __future__ import annotations

import atexit
import shutil
import sqlite3
import tempfile
from pathlib import Path

# Les statuts possibles d'une fiche client (défi 96), dans l'ordre du parcours de rétention.
STATUTS = ("a_contacter", "contacte", "offre_envoyee", "fidelise", "parti")

# Durée du « bail » d'une tâche (défi 97) : au-delà, on considère que le travailleur qui l'a prise
# est tombé en panne, et un autre peut la reprendre.
BAIL = 300.0


class ConflitDeVersion(Exception):
    """La fiche a été modifiée par quelqu'un d'autre depuis que vous l'avez lue (défi 96).

    L'application affiche alors : « Ce client vient d'être modifié par un collègue. Rechargez la
    fiche avant d'enregistrer. » C'est mieux que d'écraser son travail sans prévenir.
    """


SCHEMA = (
    """CREATE TABLE clients (
        id      TEXT PRIMARY KEY,
        nom     TEXT NOT NULL,
        score   REAL NOT NULL,
        statut  TEXT NOT NULL DEFAULT 'a_contacter',
        version INTEGER NOT NULL DEFAULT 1
    )""",
    """CREATE TABLE factures (
        id        INTEGER PRIMARY KEY,
        client_id TEXT NOT NULL,
        mois      TEXT NOT NULL,
        montant   REAL NOT NULL
    )""",
    "CREATE INDEX factures_client ON factures (client_id)",
    """CREATE TABLE budget (
        campagne TEXT PRIMARY KEY,
        reste    REAL NOT NULL
    )""",
    """CREATE TABLE offres (
        id        INTEGER PRIMARY KEY,
        client_id TEXT NOT NULL UNIQUE,
        campagne  TEXT NOT NULL,
        montant   REAL NOT NULL
    )""",
    """CREATE TABLE taches (
        id          INTEGER PRIMARY KEY,
        client_id   TEXT NOT NULL,
        priorite    INTEGER NOT NULL DEFAULT 0,
        statut      TEXT NOT NULL DEFAULT 'a_faire',
        travailleur TEXT,
        pris_a      REAL,
        essais      INTEGER NOT NULL DEFAULT 0
    )""",
)

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
_FACTURES = [  # volontairement dans le désordre : seul ORDER BY garantit un ordre
    ("C001", "2026-09", 89.9), ("C002", "2026-08", 45.0), ("C001", "2026-07", 79.9),
    ("C003", "2026-09", 19.99), ("C001", "2026-08", 89.9), ("C002", "2026-09", 45.0),
    ("C005", "2026-09", 60.0), ("C006", "2026-07", 35.5), ("C006", "2026-09", 35.5),
]
_BUDGET = [("automne", 500.0), ("fidelite", 120.0), ("noël d'hiver", 200.0)]
_OFFRES = [("C002", "automne", 30.0)]
_TACHES = [  # (client_id, priorite, statut, travailleur, pris_a, essais)
    ("C001", 0, "a_faire", None, None, 0),
    ("C004", 5, "a_faire", None, None, 0),
    ("C006", 5, "a_faire", None, None, 0),
    ("C003", 9, "faite", "poste-1", 400.0, 1),
    ("C005", 1, "en_cours", "poste-2", 900.0, 1),
    ("C007", 2, "en_cours", "poste-3", 700.0, 1),
]


def creer_schema(conn: sqlite3.Connection) -> None:
    """Crée les tables de la série 15 dans une base vide, et valide (commit)."""
    for instruction in SCHEMA:
        conn.execute(instruction)
    conn.commit()


def base_exemple(chemin: str | Path = ":memory:") -> sqlite3.Connection:
    """Une base remplie, prête pour essayer vos fonctions : 8 clients, 9 factures, 3 campagnes, 6 tâches.

    Par défaut la base est en mémoire (elle disparaît à la fin du programme). Donnez un chemin de
    fichier qui n'existe pas encore pour la garder, ou pour l'ouvrir depuis deux connexions.

        >>> conn = base_exemple()
        >>> accorder_offre(conn, "C001", "automne", 50)      # défi 92
        2
    """
    conn = sqlite3.connect(chemin)
    creer_schema(conn)
    conn.executemany("INSERT INTO clients (id, nom, score, statut, version) VALUES (?, ?, ?, ?, ?)", _CLIENTS)
    conn.executemany("INSERT INTO factures (client_id, mois, montant) VALUES (?, ?, ?)", _FACTURES)
    conn.executemany("INSERT INTO budget (campagne, reste) VALUES (?, ?)", _BUDGET)
    conn.executemany("INSERT INTO offres (client_id, campagne, montant) VALUES (?, ?, ?)", _OFFRES)
    conn.executemany("INSERT INTO taches (client_id, priorite, statut, travailleur, pris_a, essais) "
                     "VALUES (?, ?, ?, ?, ?, ?)", _TACHES)
    conn.commit()
    return conn


def deux_connexions() -> tuple[sqlite3.Connection, sqlite3.Connection]:
    """Deux connexions à la MÊME base (un fichier temporaire) : deux conseillers, ou deux travailleurs.

    Pour voir de vos yeux ce qu'une connexion voit (ou ne voit pas encore) de l'autre :

        >>> lea, hugo = deux_connexions()
        >>> lea.execute("UPDATE budget SET reste = 0 WHERE campagne = 'automne'")   # pas encore de commit
        >>> afficher(hugo, "SELECT * FROM budget WHERE campagne = 'automne'")        # Hugo voit encore 500
        >>> lea.commit()
        >>> afficher(hugo, "SELECT * FROM budget WHERE campagne = 'automne'")        # maintenant 0
    """
    dossier = Path(tempfile.mkdtemp(prefix="serie15-"))
    atexit.register(shutil.rmtree, dossier, True)  # le fichier est supprimé à la fin du programme
    chemin = dossier / "retention.db"
    base_exemple(chemin).close()
    return sqlite3.connect(chemin), sqlite3.connect(chemin)


def afficher(conn: sqlite3.Connection, sql: str, parametres: tuple | dict = ()) -> None:
    """Exécute une requête de LECTURE et affiche le résultat sous forme de petit tableau.

        >>> afficher(conn, "SELECT id, nom FROM clients WHERE score > ?", (0.8,))
    """
    curseur = conn.execute(sql, parametres)
    colonnes = [d[0] for d in curseur.description or ()]
    lignes = [["NULL" if v is None else str(v) for v in ligne] for ligne in curseur.fetchall()]
    if not colonnes:
        print("(aucun résultat à afficher : ce n'est pas une requête de lecture)")
        return
    largeurs = [max(len(c), *(len(ligne[i]) for ligne in lignes)) if lignes else len(c) for i, c in enumerate(colonnes)]
    print(" | ".join(c.ljust(n) for c, n in zip(colonnes, largeurs)).rstrip())
    print("-+-".join("-" * n for n in largeurs))
    for ligne in lignes:
        print(" | ".join(v.ljust(n) for v, n in zip(ligne, largeurs)).rstrip())
    print(f"({len(lignes)} ligne{'s' if len(lignes) > 1 else ''})")


# Pour essayer le défi 95. Chaque migration est une suite d'instructions SQL : la migration n° k
# (en partant de 1) fait passer la base de la version k - 1 à la version k.
MIGRATIONS_EXEMPLE = [
    # 1 : la table des clients
    ("CREATE TABLE clients (id TEXT PRIMARY KEY, nom TEXT NOT NULL)",),
    # 2 : le score de churn, et un index pour lister vite les plus à risque
    ("ALTER TABLE clients ADD COLUMN score REAL NOT NULL DEFAULT 0",
     "CREATE INDEX clients_par_score ON clients (score)"),
    # 3 : la table des offres de rétention
    ("CREATE TABLE offres (id INTEGER PRIMARY KEY, client_id TEXT NOT NULL UNIQUE, montant REAL NOT NULL)",),
]

# Une migration qui échoue à sa 2e instruction (la table n'existe pas) : la 1re ne doit pas rester.
MIGRATION_CASSEE = (
    "CREATE TABLE journal (id INTEGER PRIMARY KEY, message TEXT)",
    "ALTER TABLE table_qui_n_existe_pas ADD COLUMN x TEXT",
)

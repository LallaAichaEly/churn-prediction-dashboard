"""Les briques du projet 6 : formats des exports, base SQLite, table du projet 1 (DÉJÀ ÉCRIT : à lire, pas à modifier).

Lisez ce fichier en premier. Il définit :

- le format des deux exports bruts, celui du CRM (CSV) et celui du support (JSON Lines), et les
  valeurs permises (étapes 1 et 2) ;
- les petites classes figées (`frozen=True`) qui circulent entre les étapes ;
- le schéma de la base SQLite et `ouvrir_base` (étapes 3 à 6) ;
- `empreinte` (étape 4) et `mois_entre` (étape 5) ;
- les colonnes de la table que le projet 1 sait lire (étape 5).

Le correcteur suppose que ce fichier n'a pas changé.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------- L'export du CRM (étape 1)

#: Chaque début de mois, le CRM dépose `crm_AAAA-MM-JJ.csv` : les contrats CRÉÉS ou MODIFIÉS depuis
#: l'export précédent, une ligne par contrat, dans son état au moment de l'export (le tout premier
#: fichier contient tous les contrats). Le fichier est écrit pour Excel :
#:
#:     client;debut;fin;contrat;paiement;internet;assistance;senior;produits;mensualite;maj
#:     C00001;14/03/2024;;Mensuel;Prélèvement;Fibre;Non;Non;2;79,90;2026-01-04T17:42:00
#:
#: - séparateur `;`, encodage UTF-8 AVEC une marque d'ordre des octets (BOM) au début du fichier ;
#: - `debut` et `fin` au format JJ/MM/AAAA ; `fin` vide = toujours client ;
#: - `mensualite` en euros avec une virgule décimale ;
#: - `maj` : date et heure de la dernière modification du contrat dans le CRM (ISO 8601).
COLONNES_CRM = ("client", "debut", "fin", "contrat", "paiement", "internet", "assistance", "senior",
                "produits", "mensualite", "maj")

#: Un identifiant client : la lettre C suivie de 5 chiffres (C00042).
FORMAT_CLIENT = r"C\d{5}"

#: Les seules valeurs permises pour les colonnes catégorielles (celles du projet 1).
MODALITES = {
    "contrat": ("Mensuel", "Annuel", "Biannuel"),
    "paiement": ("Prélèvement", "Carte bancaire", "Virement", "Chèque"),
    "internet": ("Fibre", "ADSL", "Aucun"),
    "assistance": ("Oui", "Non"),
    "senior": ("Oui", "Non"),
}

# ---------------------------------------------------------------------------- L'export du support (étape 2)

#: L'outil du support dépose `support_AAAA-MM-JJ.jsonl` : un ticket par ligne, au format JSON.
#:
#:     {"ticket": "T000123", "client": "C00042", "ouvert": "2026-02-14T10:03:00", "maj": "2026-02-15T08:00:00", "statut": "résolu"}
#:
#: Un ticket est RENVOYÉ à chaque mise à jour (même `ticket`, `maj` plus récente) : il peut donc
#: apparaître plusieurs fois, dans le même fichier ou dans des fichiers différents. Les clés autres
#: que celles-ci (`statut`, et d'autres qui viendront) ne nous servent pas.
CHAMPS_TICKET = ("ticket", "client", "ouvert", "maj")

#: Un identifiant de ticket : la lettre T suivie de 6 chiffres (T000123).
FORMAT_TICKET = r"T\d{6}"


# ---------------------------------------------------------------------------- Ce qui circule entre les étapes


@dataclass(frozen=True)
class Contrat:
    """Une ligne lisible de l'export du CRM, avec les bons types (étape 1)."""

    client: str
    debut: date
    fin: date | None          # None = toujours client
    contrat: str
    paiement: str
    internet: str
    assistance: str
    senior: str
    produits: int
    mensualite: float         # en euros
    maj: datetime


@dataclass(frozen=True)
class Ticket:
    """Un ticket du support, dans sa version la plus récente (étape 2)."""

    ticket: str
    client: str
    ouvert: datetime
    maj: datetime


@dataclass(frozen=True)
class Rejet:
    """Une ligne écartée : le fichier (son nom, sans le dossier), le numéro de la ligne dans le
    fichier (la toute première ligne est la ligne 1) et la raison, en une phrase."""

    fichier: str
    ligne: int
    raison: str


@dataclass(frozen=True)
class Lecture:
    """Ce que renvoie la lecture d'un export : les lignes lisibles et les lignes écartées."""

    lignes: list
    rejets: list


#: Les trois issues possibles du chargement d'un fichier (étape 4).
CHARGE = "charge"            # lu et chargé dans la base
DEJA_CHARGE = "deja_charge"  # ce fichier, avec exactement ce contenu, est déjà dans la base
REFUSE = "refuse"            # trop de lignes illisibles : rien n'a été chargé

#: Au-delà de cette part de lignes écartées, on refuse tout le fichier (étape 4).
SEUIL_REJETS = 0.10


@dataclass(frozen=True)
class Bilan:
    """Ce qu'a donné le chargement d'un fichier (étape 4).

    `lignes` : le nombre de lignes de la base créées ou modifiées par ce fichier (0 si rien n'a
    changé) ; `rejets` : les lignes écartées à la lecture.
    """

    fichier: str
    statut: str
    lignes: int
    rejets: list


# ---------------------------------------------------------------------------- La base SQLite (étapes 3 à 6)

#: Les dates sont rangées en TEXTE ISO 8601 : "2026-06-30" pour un jour, "2026-06-30T15:04:00" pour
#: une date et une heure. Deux textes à ce format se comparent comme les dates qu'ils représentent,
#: et les fonctions `date()` et `datetime()` de SQLite savent les lire.
SCHEMA = """
CREATE TABLE IF NOT EXISTS contrats (
    client      TEXT PRIMARY KEY,
    debut       TEXT NOT NULL,
    fin         TEXT,
    contrat     TEXT NOT NULL,
    paiement    TEXT NOT NULL,
    internet    TEXT NOT NULL,
    assistance  TEXT NOT NULL,
    senior      TEXT NOT NULL,
    produits    INTEGER NOT NULL CHECK (produits >= 1),
    mensualite  REAL NOT NULL CHECK (mensualite > 0),
    maj         TEXT NOT NULL,
    CHECK (fin IS NULL OR fin >= debut)
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket  TEXT PRIMARY KEY,
    client  TEXT NOT NULL,
    ouvert  TEXT NOT NULL,
    maj     TEXT NOT NULL
);

-- Un fichier déjà chargé : son nom, l'empreinte de son contenu, quand et ce qu'il a donné.
CREATE TABLE IF NOT EXISTS fichiers (
    nom        TEXT PRIMARY KEY,
    empreinte  TEXT NOT NULL,
    charge_le  TEXT NOT NULL,
    lignes     INTEGER NOT NULL,
    rejets     INTEGER NOT NULL
);
"""


def ouvrir_base(chemin: str | Path) -> sqlite3.Connection:
    """Ouvre (ou crée) la base et ses trois tables. `":memory:"` donne une base en mémoire, pour les tests.

    Attention : `sqlite3.connect` CRÉE le fichier s'il n'existe pas. Pour seulement lire une base
    existante, vérifiez d'abord qu'elle existe (étape 6).
    """
    connexion = sqlite3.connect(str(chemin))
    connexion.executescript(SCHEMA)
    return connexion


def empreinte(chemin: str | Path) -> str:
    """L'empreinte SHA-256 du contenu d'un fichier (64 caractères hexadécimaux), lu par blocs.

    Deux fichiers ont la même empreinte si et seulement s'ils ont exactement le même contenu : c'est
    ainsi qu'on sait qu'un fichier a déjà été chargé, même s'il a été copié ou déplacé (étape 4).
    """
    calcul = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            calcul.update(bloc)
    return calcul.hexdigest()


# ---------------------------------------------------------------------------- La table du projet 1 (étape 5)

#: Les colonnes, dans cet ordre, du fichier `data/customers.csv` qu'entraîne le projet 1.
COLONNES_TABLE = ("customer_id", "tenure_months", "monthly_charges", "total_charges", "support_calls",
                  "num_products", "contract", "payment_method", "internet_service", "tech_support", "senior",
                  "churn")

#: churn = 1 si le client part dans les HORIZON_JOURS qui suivent la photo.
HORIZON_JOURS = 90

#: support_calls = le nombre de tickets ouverts dans les FENETRE_JOURS qui précèdent la photo.
FENETRE_JOURS = 365


def mois_entre(debut: date, fin: date) -> int:
    """Le nombre de mois COMPLETS entre deux dates.

    >>> mois_entre(date(2026, 1, 15), date(2026, 3, 14))
    1
    >>> mois_entre(date(2026, 1, 15), date(2026, 3, 15))
    2
    >>> mois_entre(date(2026, 6, 20), date(2026, 6, 30))
    0
    """
    mois = (fin.year - debut.year) * 12 + fin.month - debut.month
    return mois - 1 if fin.day < debut.day else mois

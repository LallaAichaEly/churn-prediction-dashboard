"""Les objets que tout le projet 4 manipule (DÉJÀ ÉCRIT : à lire, pas à modifier).

Lisez ce fichier en premier : chaque étape s'appuie sur ces constantes, ce schéma, ces classes et
ces exceptions. Le correcteur suppose qu'il n'a pas changé.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------- Les statuts d'un contact

A_APPELER = "a_appeler"      # personne n'a encore réussi à joindre le client
ACCEPTE = "accepte"          # le client a accepté l'offre
REFUSE = "refuse"            # le client a refusé l'offre
INJOIGNABLE = "injoignable"  # le conseiller a appelé, sans réponse : on pourra réessayer

STATUTS = (A_APPELER, ACCEPTE, REFUSE, INJOIGNABLE)
RESULTATS = (ACCEPTE, REFUSE, INJOIGNABLE)  # ce qu'un conseiller peut enregistrer après un appel

# ---------------------------------------------------------------------------- La configuration

BASE_PAR_DEFAUT = "service.db"
TAILLE_PAGE_PAR_DEFAUT = 20
TAILLE_PAGE_MAX = 100
LONGUEUR_CLE_MIN = 16  # une clé plus courte se devine trop facilement

# ---------------------------------------------------------------------------- La base

SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    customer_id TEXT PRIMARY KEY,
    offre       TEXT NOT NULL,
    conseiller  TEXT NOT NULL,
    statut      TEXT NOT NULL DEFAULT 'a_appeler'
                CHECK (statut IN ('a_appeler', 'accepte', 'refuse', 'injoignable')),
    tentatives  INTEGER NOT NULL DEFAULT 0 CHECK (tentatives >= 0),
    commentaire TEXT NOT NULL DEFAULT '',
    version     INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1)
);

-- La liste d'un conseiller, triée par client : la requête la plus fréquente du service.
CREATE INDEX IF NOT EXISTS contacts_par_conseiller ON contacts (conseiller, customer_id);

-- Le journal : une ligne par appel enregistré, jamais modifiée ni effacée.
CREATE TABLE IF NOT EXISTS appels (
    numero      INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL REFERENCES contacts (customer_id),
    conseiller  TEXT NOT NULL,
    statut      TEXT NOT NULL,
    commentaire TEXT NOT NULL DEFAULT '',
    le          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

#: Les colonnes de `contacts`, dans l'ordre des champs de `Contact` :
#: `Contact(*ligne)` fonctionne pour chaque ligne de `SELECT {COLONNES_CONTACT} FROM contacts …`.
COLONNES_CONTACT = "customer_id, offre, conseiller, statut, tentatives, commentaire, version"


# ---------------------------------------------------------------------------- Les objets


@dataclass(frozen=True)
class Config:
    """La configuration du service, lue UNE fois au démarrage (étape 1).

    `repr=False` sur les clés : un `print(config)` ou un journal qui affiche la configuration ne
    montre jamais un secret, seulement le nom des conseillers.
    """

    base: Path
    cles: dict[str, str] = field(repr=False)  # conseiller → sa clé d'API
    taille_page: int = TAILLE_PAGE_PAR_DEFAUT

    @property
    def conseillers(self) -> list[str]:
        return sorted(self.cles)


@dataclass(frozen=True)
class Contact:
    """Un client à appeler, tel qu'il est enregistré dans la table `contacts`."""

    customer_id: str
    offre: str
    conseiller: str
    statut: str
    tentatives: int
    commentaire: str
    version: int


class ResultatAppel(BaseModel):
    """Le corps de la requête `PUT /contacts/{customer_id}/appel` (étape 5).

    FastAPI le valide AVANT d'appeler votre route : un statut inconnu, une version absente ou un
    commentaire de 10 000 caractères donnent une erreur 422, sans une ligne de code de votre part.
    """

    statut: Literal["accepte", "refuse", "injoignable"]
    version: int = Field(ge=1, description="la version de la fiche lue par le conseiller")
    commentaire: str = Field(default="", max_length=500)


# ---------------------------------------------------------------------------- Les erreurs


class ConfigInvalide(ValueError):
    """La configuration est fausse : le service refuse de démarrer (étape 1)."""


class Introuvable(LookupError):
    """Aucun contact n'a cet identifiant (→ 404 dans l'API)."""


class PasVotreClient(PermissionError):
    """Le contact existe, mais il est attribué à un autre conseiller (→ 403 dans l'API)."""


class Conflit(Exception):
    """La fiche a changé depuis que le conseiller l'a lue (→ 409 dans l'API).

    `version_actuelle` permet au conseiller de recharger la fiche et de réessayer.
    """

    def __init__(self, customer_id: str, version_actuelle: int):
        super().__init__(f"{customer_id} a été modifié entre-temps (version actuelle : {version_actuelle}) : "
                         "rechargez la fiche avant d'enregistrer")
        self.customer_id = customer_id
        self.version_actuelle = version_actuelle

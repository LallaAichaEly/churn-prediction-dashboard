"""Étape 7 (corrigé) — Des tests qui attrapent les bugs de l'API.

Chaque bloc vise UN comportement promis par l'API, avec des valeurs connues à l'avance grâce à
`_preparer` : on sait qui suit quel client, et dans quel ordre ils arrivent.
"""

from __future__ import annotations

import atexit
import shutil
import sqlite3
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient  # noqa: F401 — l'outil de l'étape 7

from .modeles import SCHEMA, Config

CLE_ALICE = "cle-alice-0123456789"
CLE_BRUNO = "cle-bruno-0123456789"
ALICE = {"X-Cle-API": CLE_ALICE}
BRUNO = {"X-Cle-API": CLE_BRUNO}


def _preparer() -> Config:
    """Une base de test NEUVE, dans un dossier temporaire, et la configuration qui va avec (DÉJÀ ÉCRIT).

    - alice suit C001 à C005 (5 contacts), bruno suit C006 et C007 ;
    - tous sont « a_appeler », en version 1 ;
    - taille_page=2 : une liste de 5 contacts tient sur 3 pages (C001-C002, C003-C004, C005).
    """
    dossier = Path(tempfile.mkdtemp(prefix="service_test_"))
    atexit.register(shutil.rmtree, dossier, ignore_errors=True)
    base = dossier / "test.db"
    conn = sqlite3.connect(base)
    try:
        conn.executescript(SCHEMA)
        with conn:
            conn.executemany("INSERT INTO contacts (customer_id, offre, conseiller) VALUES (?, ?, ?)", [
                ("C001", "APPEL", "alice"), ("C002", "REMISE", "alice"), ("C003", "SUPPORT", "alice"),
                ("C004", "ENGAGEMENT", "alice"), ("C005", "APPEL", "alice"),
                ("C006", "REMISE", "bruno"), ("C007", "APPEL", "bruno"),
            ])
    finally:
        conn.close()
    return Config(base=base, cles={"alice": CLE_ALICE, "bruno": CLE_BRUNO}, taille_page=2)


def _ids(reponse) -> list[str]:
    return [c["customer_id"] for c in reponse.json()["contacts"]]


def tester_service(creer_app) -> None:
    client = TestClient(creer_app(_preparer()))

    # 1. Sans clé, ou avec une clé presque juste : 401, et rien n'est montré.
    assert client.get("/contacts").status_code == 401
    assert client.get("/contacts", headers={"X-Cle-API": CLE_ALICE[:-1] + "X"}).status_code == 401

    # 2. La première page d'alice : 2 contacts (taille_page=2), et un curseur pour la suite.
    reponse = client.get("/contacts", headers=ALICE)
    assert reponse.status_code == 200
    assert _ids(reponse) == ["C001", "C002"] and reponse.json()["suivant"] == "C002"
    # La dernière page : plus de suivant.
    derniere = client.get("/contacts", params={"apres": "C004"}, headers=ALICE)
    assert _ids(derniere) == ["C005"] and derniere.json()["suivant"] is None
    # bruno ne voit que les siens.
    assert _ids(client.get("/contacts", params={"taille": 10}, headers=BRUNO)) == ["C006", "C007"]

    # 3. La fiche d'un client de bruno : 403 pour alice. Un client qui n'existe pas : 404.
    assert client.get("/contacts/C006", headers=ALICE).status_code == 403
    assert client.get("/contacts/C999", headers=ALICE).status_code == 404

    # 4. Enregistrer un appel : la fiche revient à jour, en version 2.
    reponse = client.put("/contacts/C003/appel", json={"statut": "accepte", "version": 1}, headers=ALICE)
    assert reponse.status_code == 200
    assert reponse.json()["statut"] == "accepte" and reponse.json()["version"] == 2

    # 5. Réenregistrer avec la version 1 (périmée) : 409, et la version actuelle pour recharger.
    reponse = client.put("/contacts/C003/appel", json={"statut": "refuse", "version": 1}, headers=ALICE)
    assert reponse.status_code == 409 and reponse.json()["version_actuelle"] == 2
    assert client.get("/contacts/C003", headers=ALICE).json()["statut"] == "accepte"  # rien n'a changé

    # 6. Le filtre par statut : seul C003 est « accepte ».
    assert _ids(client.get("/contacts", params={"statut": "accepte"}, headers=ALICE)) == ["C003"]

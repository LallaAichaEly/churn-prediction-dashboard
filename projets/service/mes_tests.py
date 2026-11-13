"""Étape 7 🔴 — Écrire les tests qui protègent l'API (étape 5).

Vérifier : python scripts/projet.py --projet 4 --etape 7
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
    """Vérifie, avec des `assert`, qu'une fonction `creer_app(config)` respecte le contrat de l'API
    (relisez la docstring de l'étape 5, dans `api.py`).

    Comme aux étapes 7 des projets 2 et 3, c'est VOUS qui écrivez le test, et le correcteur l'exécute :
    - sur l'API JUSTE : votre test doit passer ;
    - sur 8 versions BOGUÉES, toutes plausibles : votre test doit échouer sur CHACUNE.

    Pour commencer (`_preparer`, `ALICE`, `BRUNO` et `_ids` sont déjà écrits, plus haut) :

        client = TestClient(creer_app(_preparer()))
        reponse = client.get("/contacts", headers=ALICE)
        assert reponse.status_code == 200
        assert _ids(reponse) == ["C001", "C002"]

    Vérifiez le code HTTP ET le contenu de la réponse. Pensez comme un attaquant (une requête sans
    clé, une clé PRESQUE juste, la fiche d'un client de quelqu'un d'autre) et comme un collègue
    distrait (une version périmée, la page suivante, un filtre). Calculez ce que vous attendez à
    partir de la base de `_preparer`, jamais en recopiant ce que renvoie votre propre code.

    Ce que vous apprenez : tester une API de l'extérieur avec `TestClient`, préparer des données de
    test connues, choisir des cas qui séparent les bugs.
    """
    raise NotImplementedError

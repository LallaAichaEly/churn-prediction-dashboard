"""Étape 5 🟡 — L'API web des conseillers.

Vérifier : python scripts/projet.py --projet 4 --etape 5
Essayer pour de vrai (une fois l'étape 6 faite) : python -m projets.service servir, puis
http://127.0.0.1:8000/docs, où FastAPI affiche chaque route et permet de l'essayer.
"""

from __future__ import annotations

import logging
import secrets  # noqa: F401
import sqlite3  # noqa: F401
from dataclasses import asdict  # noqa: F401
from typing import Iterator, Literal  # noqa: F401

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status  # noqa: F401
from fastapi.responses import JSONResponse  # noqa: F401

from . import appels, contacts  # noqa: F401
from .base import ouvrir  # noqa: F401
from .modeles import TAILLE_PAGE_MAX, Config, Conflit, Introuvable, PasVotreClient, ResultatAppel  # noqa: F401

journal = logging.getLogger("projets.service")


def creer_app(config: Config) -> FastAPI:
    """Fabrique l'application FastAPI du service et la renvoie.

    Les DÉPENDANCES (des fonctions que FastAPI appelle pour vous, avec `Depends`) :

    - `connexion()` : ouvre la base (`ouvrir(config.base)`), la donne à la route avec `yield`, et
      la ferme dans un `finally`. Une connexion PAR REQUÊTE, toujours fermée : le correcteur compte
      les connexions ouvertes et jamais fermées.
    - `conseiller_connecte(x_cle_api: str | None = Header(default=None))` : lit l'en-tête
      `X-Cle-API` et renvoie le NOM du conseiller à qui appartient la clé. Clé absente ou inconnue :
      `HTTPException(401)`. Comparez avec `secrets.compare_digest`, qui met le même temps que la clé
      soit presque juste ou complètement fausse (avec `==`, un attaquant qui chronomètre devine la
      clé caractère par caractère). Comparez des OCTETS (`.encode("utf-8")`) : sur des textes,
      compare_digest plante dès qu'un caractère n'est pas ASCII, et une clé « é… » donnerait 500.
      Le correcteur essaie aussi la vraie clé sans son dernier caractère, ou avec un en plus.

    Les ROUTES (toutes, sauf /sante, exigent la clé) :

    - `GET /sante` → `{"statut": "ok"}`, sans clé : un robot vérifie que le service répond.
    - `GET /contacts?statut=&apres=&taille=` → `{"contacts": [fiche, ...], "suivant": ...}` : une page
      des contacts DU conseiller connecté (`contacts.page_contacts`).
      `taille` vaut `config.taille_page` par défaut, entre 1 et `TAILLE_PAGE_MAX`, sinon 422
      (`Query(default=..., ge=..., le=...)`). Un `statut` inconnu donne 422 : déclarez-le avec un type
      `Literal["a_appeler", ...]` (les valeurs de STATUTS), FastAPI vérifie pour vous.
    - `GET /contacts/{customer_id}` → la fiche ; 404 si elle n'existe pas, 403 si elle est suivie par
      un autre conseiller.
    - `PUT /contacts/{customer_id}/appel` → corps `ResultatAppel` (modeles.py : FastAPI le valide, 422
      sinon) ; enregistre l'appel (`appels.enregistrer_appel`) et renvoie la fiche À JOUR. Erreurs :
      404, 403, et 409 en cas de conflit, avec `{"detail": ..., "version_actuelle": n}`.
    - `GET /avancement` → `contacts.avancement(conn)` : l'avancement de toute l'équipe.

    Une fiche se renvoie avec `asdict(contact)`.

    Traduire les erreurs du métier en codes HTTP : plutôt qu'un try/except dans chaque route,
    déclarez UNE fois, par exception, un gestionnaire :

        @app.exception_handler(Introuvable)
        def introuvable(requete: Request, exc: Introuvable) -> JSONResponse:
            return JSONResponse(status_code=404, content={"detail": ...})

    Le journal : `journal.info(...)` ou `journal.warning(...)` si vous voulez, avec le NOM du
    conseiller, jamais une clé (ni celle reçue, qui est peut-être la vraie clé d'un collègue mal
    recopiée) : le correcteur lit votre journal.

    Ce que vous apprenez : assembler une API sur une vraie base (série 12), authentifier par clé,
    une ressource par requête (`yield`), traduire des exceptions en codes HTTP.
    """
    raise NotImplementedError

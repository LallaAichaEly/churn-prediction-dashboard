"""Étape 5 (corrigé) — L'API web des conseillers."""

from __future__ import annotations

import logging
import secrets
import sqlite3
from dataclasses import asdict
from typing import Iterator, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from . import appels, contacts
from .base import ouvrir
from .modeles import TAILLE_PAGE_MAX, Config, Conflit, Introuvable, PasVotreClient, ResultatAppel

journal = logging.getLogger("projets.service")

# Les valeurs de modeles.STATUTS, écrites en toutes lettres pour que FastAPI les valide (et les
# affiche dans /docs) : `?statut=perdu` donne une erreur 422 sans une ligne de code.
Statut = Literal["a_appeler", "accepte", "refuse", "injoignable"]


def creer_app(config: Config) -> FastAPI:
    app = FastAPI(title="Service des conseillers",
                  description="Chaque conseiller voit SES clients à appeler et enregistre le résultat de ses appels.")

    # ------------------------------------------------------------------ Les dépendances

    def connexion() -> Iterator[sqlite3.Connection]:
        # Une connexion PAR REQUÊTE, fermée quoi qu'il arrive (le `finally` s'exécute après la
        # réponse, même si la route a levé une exception). Une connexion partagée entre requêtes
        # simultanées mélangerait leurs transactions.
        conn = ouvrir(config.base)
        try:
            yield conn
        finally:
            conn.close()

    def conseiller_connecte(x_cle_api: str | None = Header(default=None)) -> str:
        # FastAPI lit l'en-tête « X-Cle-API » (les _ deviennent des -, la casse ne compte pas).
        trouve = None
        if x_cle_api:
            recue = x_cle_api.encode("utf-8")
            for nom, cle in config.cles.items():
                # compare_digest met le même temps que la clé soit presque juste ou complètement
                # fausse : impossible de la deviner caractère par caractère en chronométrant. Sur
                # des octets, parce qu'avec des str il plante (TypeError) dès qu'un caractère n'est
                # pas ASCII : une clé « é… » donnerait une erreur 500 au lieu de 401.
                if secrets.compare_digest(recue, cle.encode("utf-8")):
                    trouve = nom
        if trouve is None:
            journal.warning("requête refusée : clé d'API absente ou inconnue")  # jamais la clé reçue !
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Clé d'API absente ou invalide (en-tête X-Cle-API)")
        return trouve

    # ------------------------------------------------------------------ Les erreurs du métier → codes HTTP
    # Un seul endroit traduit les exceptions des étapes 3 et 4 : les routes restent courtes.

    @app.exception_handler(Introuvable)
    def introuvable(_: Request, exc: Introuvable) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": f"contact inconnu : {exc}"})

    @app.exception_handler(PasVotreClient)
    def pas_votre_client(_: Request, exc: PasVotreClient) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(exc)})

    @app.exception_handler(Conflit)
    def conflit(_: Request, exc: Conflit) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT,
                            content={"detail": str(exc), "version_actuelle": exc.version_actuelle})

    # ------------------------------------------------------------------ Les routes

    @app.get("/sante")
    def sante() -> dict:
        return {"statut": "ok"}

    @app.get("/contacts")
    def liste(statut: Statut | None = None, apres: str | None = None,
              taille: int = Query(default=config.taille_page, ge=1, le=TAILLE_PAGE_MAX),
              conseiller: str = Depends(conseiller_connecte),
              conn: sqlite3.Connection = Depends(connexion)) -> dict:
        page, suivant = contacts.page_contacts(conn, conseiller, statut, apres, taille)
        return {"contacts": [asdict(c) for c in page], "suivant": suivant}

    @app.get("/contacts/{customer_id}")
    def fiche(customer_id: str, conseiller: str = Depends(conseiller_connecte),
              conn: sqlite3.Connection = Depends(connexion)) -> dict:
        return asdict(contacts.lire_contact(conn, conseiller, customer_id))

    @app.put("/contacts/{customer_id}/appel")
    def appel(customer_id: str, resultat: ResultatAppel, conseiller: str = Depends(conseiller_connecte),
              conn: sqlite3.Connection = Depends(connexion)) -> dict:
        contact = appels.enregistrer_appel(conn, conseiller, customer_id, resultat.statut, resultat.version,
                                           resultat.commentaire)
        journal.info("appel enregistré : %s par %s → %s", customer_id, conseiller, resultat.statut)
        return asdict(contact)

    @app.get("/avancement")
    def equipe(conseiller: str = Depends(conseiller_connecte),
               conn: sqlite3.Connection = Depends(connexion)) -> dict:
        return contacts.avancement(conn)

    return app

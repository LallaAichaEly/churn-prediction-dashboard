"""Série 12 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_12.py`.

Si vos routes passent les vérifications, elles sont bonnes, même si elles ne ressemblent pas à
celles-ci. Comparez : notez dans JOURNAL.md le code HTTP ou le piège auquel vous n'aviez pas pensé.

Fil rouge de la série : une API est un CONTRAT. Pour chaque requête possible, y compris les
mauvaises, on décide du code et du contenu de la réponse. Un code 500 n'est jamais une réponse :
c'est un contrat oublié.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`, avec `TestClient`.
"""

from __future__ import annotations

import copy
import itertools
import secrets
from typing import Callable, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Response, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field

CONTRATS = ("Mensuel", "Annuel", "Biannuel")
OFFRES = ("remise_10", "mois_offert", "appel_conseiller")
Contrat = Literal["Mensuel", "Annuel", "Biannuel"]
Offre = Literal["remise_10", "mois_offert", "appel_conseiller"]

_EXEMPLE = {
    "C001": {"id": "C001", "anciennete_mois": 3, "facture_mensuelle": 89.9, "contrat": "Mensuel", "appels_support": 4},
    "C002": {"id": "C002", "anciennete_mois": 40, "facture_mensuelle": 45.0, "contrat": "Biannuel", "appels_support": 0},
    "C003": {"id": "C003", "anciennete_mois": 14, "facture_mensuelle": 70.5, "contrat": "Annuel", "appels_support": 2},
    "C004": {"id": "C004", "anciennete_mois": 1, "facture_mensuelle": 99.0, "contrat": "Mensuel", "appels_support": 1},
}


def exemple_clients() -> dict[str, dict]:
    return copy.deepcopy(_EXEMPLE)


def essayer(routeur: APIRouter) -> TestClient:
    app = FastAPI()
    app.include_router(routeur)
    return TestClient(app)


# ============================================================================ Lire une ressource


def routes_lecture(clients: dict[str, dict]) -> APIRouter:
    routeur = APIRouter()

    @routeur.get("/clients/{client_id}")
    def lire_client(client_id: str) -> dict:
        # On teste l'absence AVANT de lire : `clients[client_id]` lèverait KeyError → 500.
        if client_id not in clients:
            # Le message cite l'identifiant : dans les journaux de l'appelant, « Client C999
            # introuvable » se comprend tout de suite ; « Not found » oblige à chercher.
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Client {client_id} introuvable")
        return clients[client_id]

    return routeur


# ============================================================================ Chercher : paramètres de requête


def routes_recherche(clients: dict[str, dict]) -> APIRouter:
    routeur = APIRouter()

    @routeur.get("/clients")
    def chercher_clients(
        contrat: str | None = None,                     # facultatif : None = « pas de filtre »
        min_appels: int = Query(0, ge=0),               # -1 → 422 automatiquement
        limite: int = Query(20, ge=1, le=100),          # protège le serveur : jamais 1 million de lignes
    ) -> list[dict]:
        trouves = [
            c for c in clients.values()
            if (contrat is None or c["contrat"] == contrat) and c["appels_support"] >= min_appels
        ]
        # Trier rend la réponse prévisible : deux appels identiques donnent la même liste, ce qui
        # est indispensable dès qu'on ajoute une pagination. Couper APRÈS le filtre, sinon
        # `limite=2` pourrait renvoyer 0 résultat alors que 3 clients correspondent.
        return sorted(trouves, key=lambda c: c["id"])[:limite]

    return routeur


# ============================================================================ Créer : le corps de la requête


class NouveauClient(BaseModel):
    # extra="forbid" : un champ inconnu (faute de frappe) donne 422 au lieu d'être ignoré.
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^C\d{3}$", examples=["C042"])
    anciennete_mois: int = Field(ge=0)
    facture_mensuelle: float = Field(gt=0, le=500)      # gt : strictement positif ; le : au plus
    contrat: Contrat                                    # "Mensual" → 422, avec la liste des valeurs permises
    appels_support: int = Field(default=0, ge=0)        # une valeur par défaut rend le champ facultatif


def routes_creation(clients: dict[str, dict]) -> APIRouter:
    routeur = APIRouter()

    @routeur.post("/clients", status_code=status.HTTP_201_CREATED)
    def creer_client(nouveau: NouveauClient, response: Response) -> dict:
        # Ici, `nouveau` est déjà validé : si on arrive dans la fonction, les données sont propres.
        if nouveau.id in clients:
            # 409 : la requête est correcte, mais impossible dans l'état actuel de la base.
            # Surtout pas d'écrasement silencieux : on perdrait l'historique d'un vrai client.
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Le client {nouveau.id} existe déjà")
        client = nouveau.model_dump()       # un dict ordinaire, comme les autres clients de la base
        clients[nouveau.id] = client
        response.headers["Location"] = f"/clients/{nouveau.id}"
        return client

    return routeur


# ============================================================================ Modifier : PATCH partiel


class ModificationClient(BaseModel):
    """Tous les champs sont facultatifs (valeur par défaut), mais aucun n'accepte `null`.

    `float = Field(default=None, ...)` : Pydantic ne valide pas la valeur par défaut, donc un champ
    absent est accepté ; mais un `null` ENVOYÉ est validé comme un float, et refusé.
    Avec `float | None = None`, un `{"facture_mensuelle": null}` effacerait la facture.
    """

    model_config = ConfigDict(extra="forbid")          # refuse aussi "id" : pas de changement d'identifiant

    anciennete_mois: int = Field(default=None, ge=0)
    facture_mensuelle: float = Field(default=None, gt=0, le=500)
    contrat: Contrat = None
    appels_support: int = Field(default=None, ge=0)


def routes_modification(clients: dict[str, dict]) -> APIRouter:
    routeur = APIRouter()

    @routeur.patch("/clients/{client_id}")
    def modifier_client(client_id: str, modification: ModificationClient) -> dict:
        if client_id not in clients:
            raise HTTPException(status_code=404, detail=f"Client {client_id} introuvable")
        # exclude_unset=True : seulement les champs ENVOYÉS. Sans lui, model_dump() renverrait aussi
        # les champs absents, à None, et les écrirait sur le client.
        changements = modification.model_dump(exclude_unset=True)
        # Pas de « rien à moitié » à gérer : Pydantic a validé TOUT le corps avant d'entrer ici.
        clients[client_id] = {**clients[client_id], **changements}
        return clients[client_id]

    return routeur


# ============================================================================ Protéger : clé d'API et Depends


def routes_admin(clients: dict[str, dict], cles_valides: set[str]) -> APIRouter:
    def verifier_cle(x_api_key: str | None = Header(default=None)) -> None:
        # FastAPI lit l'en-tête « X-API-Key » (les _ du nom deviennent des -, la casse est ignorée).
        if x_api_key is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clé d'API manquante (en-tête X-API-Key)")
        # On relit `cles_valides` à chaque requête : une clé révoquée est refusée tout de suite.
        # compare_digest prend le même temps quelle que soit la clé : un attaquant ne peut pas
        # deviner la bonne clé caractère par caractère en chronométrant les réponses.
        if not any(secrets.compare_digest(x_api_key, cle) for cle in cles_valides):
            # On ne dit pas POURQUOI (clé expirée, mal tapée…) : moins on en dit à un intrus, mieux c'est.
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accès refusé")

    # La dépendance est déclarée sur le ROUTEUR : toute route ajoutée plus tard est protégée aussi.
    routeur = APIRouter(prefix="/admin", dependencies=[Depends(verifier_cle)])

    @routeur.get("/clients")
    def tous_les_clients() -> list[dict]:
        return sorted(clients.values(), key=lambda c: c["id"])

    @routeur.delete("/clients/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
    def supprimer_client(client_id: str) -> None:
        # On n'arrive ici qu'avec une clé valide : un intrus ne peut pas savoir si C999 existe.
        if client_id not in clients:
            raise HTTPException(status_code=404, detail=f"Client {client_id} introuvable")
        del clients[client_id]

    return routeur


# ============================================================================ Rejouer sans risque : idempotence


class DemandeRelance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str
    offre: Offre


def routes_relances(clients: dict[str, dict], envoyer: Callable[[str, str], None]) -> APIRouter:
    routeur = APIRouter()
    # La mémoire vit DANS la fabrique : chaque routeur a la sienne (règle 8). Une variable globale
    # du module mélangerait les clés de deux applications, et de deux tests.
    deja_vu: dict[str, tuple[dict, dict]] = {}          # clé → (corps reçu, réponse envoyée)
    numeros = itertools.count(1)

    @routeur.post("/relances", status_code=status.HTTP_201_CREATED)
    def relancer(demande: DemandeRelance, idempotency_key: str | None = Header(default=None)) -> dict:
        if not idempotency_key:
            raise HTTPException(status_code=400, detail="En-tête Idempotency-Key obligatoire")
        corps = demande.model_dump()

        if idempotency_key in deja_vu:
            corps_precedent, reponse_precedente = deja_vu[idempotency_key]
            if corps != corps_precedent:
                # Même clé, autre demande : bug chez l'appelant. On refuse plutôt que de deviner.
                raise HTTPException(status_code=422, detail="Cette Idempotency-Key a déjà servi pour une autre demande")
            return reponse_precedente                   # rejeu : même réponse, AUCUN nouvel envoi

        if demande.client_id not in clients:
            raise HTTPException(status_code=404, detail=f"Client {demande.client_id} introuvable")

        try:
            envoyer(demande.client_id, demande.offre)
        except Exception as exc:
            # On NE mémorise PAS la clé : rien n'est parti, l'appelant doit pouvoir réessayer.
            raise HTTPException(status_code=503, detail="Service d'envoi indisponible, réessayez plus tard") from exc

        reponse = {"relance_id": f"R{next(numeros)}", **corps}
        deja_vu[idempotency_key] = (corps, reponse)     # seulement APRÈS un envoi réussi
        return reponse

    return routeur


# Pour aller plus loin :
# - En production, `deja_vu` serait une table de base de données (ou Redis) avec une durée de vie
#   (24 h chez Stripe), pour survivre à un redémarrage et être partagée par plusieurs serveurs.
# - Deux requêtes avec la même clé peuvent arriver EN MÊME TEMPS. La norme recommande alors 409
#   pour la seconde : il faut « réserver » la clé (avec un verrou) avant d'envoyer.


# ============================================================================ Dans un vrai fichier de tests
#
# Chaque test crée sa propre base (exemple_clients()) : aucun test ne dépend d'un autre.
# On vérifie le CONTRAT (code et contenu), y compris pour les requêtes fautives.


def test_lire_client_inconnu_donne_404():
    client = essayer(routes_lecture(exemple_clients()))
    reponse = client.get("/clients/C999")
    assert reponse.status_code == 404
    assert "C999" in reponse.json()["detail"]


def test_creer_un_client_existant_ne_l_ecrase_pas():
    base = exemple_clients()
    client = essayer(routes_creation(base))
    doublon = {"id": "C001", "anciennete_mois": 0, "facture_mensuelle": 10.0, "contrat": "Annuel"}
    assert client.post("/clients", json=doublon).status_code == 409
    assert base["C001"] == _EXEMPLE["C001"]


def test_patch_ne_change_que_les_champs_envoyes():
    base = exemple_clients()
    client = essayer(routes_modification(base))
    assert client.patch("/clients/C001", json={"contrat": "Annuel"}).status_code == 200
    assert base["C001"] == {**_EXEMPLE["C001"], "contrat": "Annuel"}


def test_une_mauvaise_cle_ne_supprime_rien():
    base = exemple_clients()
    client = essayer(routes_admin(base, {"bonne-cle"}))
    assert client.delete("/admin/clients/C001", headers={"X-API-Key": "mauvaise"}).status_code == 403
    assert "C001" in base


def test_rejouer_une_relance_n_envoie_pas_deux_sms():
    envois = []
    client = essayer(routes_relances(exemple_clients(), lambda c, o: envois.append((c, o))))
    requete = {"json": {"client_id": "C001", "offre": "remise_10"}, "headers": {"Idempotency-Key": "k1"}}
    premiere, seconde = client.post("/relances", **requete), client.post("/relances", **requete)
    assert premiere.status_code == seconde.status_code == 201
    assert premiere.json() == seconde.json()
    assert envois == [("C001", "remise_10")]

"""Série 12 — Construire une API web : lire, chercher, créer, modifier, protéger, rejouer sans risque.

🌐 VOUS VOULEZ DÉVELOPPER ? CONSTRUISEZ CE QUE LES AUTRES APPELLENT.

Le tableau de bord sert aux humains, l'API sert aux autres programmes : le CRM qui veut le score
d'un client, l'outil marketing qui envoie les offres, l'appli mobile. Dans une équipe de dev, écrire
une route HTTP propre fait partie du quotidien. `api.py` en contient deux ; ici, vous écrivez les
vôtres, une notion à la fois, sur une petite « base » de clients en mémoire (un dictionnaire).

Une route se juge sur son CONTRAT : pour chaque requête, le bon code HTTP et le bon contenu.
Les codes que vous allez utiliser :

    200 OK                  tout va bien, voici la réponse
    201 Created             j'ai créé la ressource (en-tête Location : son adresse)
    204 No Content          c'est fait, je n'ai rien à renvoyer (suppression)
    400 Bad Request         votre requête est mal formée (un en-tête obligatoire manque…)
    401 Unauthorized        je ne sais pas qui vous êtes (pas de clé)
    403 Forbidden           je sais qui vous êtes, mais c'est non (mauvaise clé)
    404 Not Found           cette ressource n'existe pas
    409 Conflict            impossible dans l'état actuel (cet identifiant existe déjà)
    422 Unprocessable       les données ne respectent pas les règles (FastAPI le fait pour vous)
    500 Internal Error      VOTRE code a planté : ce n'est jamais une réponse voulue
    503 Unavailable         un service dont je dépends est en panne, réessayez plus tard

Prérequis : la série 00 (fonctions, dictionnaires) et le défi 40 (série 06 : les exceptions).
Aucune notion de machine learning ni de réseau : tout se passe dans votre terminal.

🧩 Comment est construit chaque défi

Dans un vrai projet, les routes sont rangées par thème dans des `APIRouter`, puis branchées sur
l'application avec `app.include_router(...)`. Chaque défi vous demande d'écrire une fonction qui
FABRIQUE un routeur : elle reçoit la base de clients, déclare ses routes, et renvoie le routeur.

    from fastapi import APIRouter, HTTPException

    def routes_bonjour(clients: dict[str, dict]) -> APIRouter:
        routeur = APIRouter()

        @routeur.get("/bonjour/{prenom}")        # GET /bonjour/Aïcha
        def dire_bonjour(prenom: str) -> dict:   # FastAPI lit `prenom` dans l'adresse
            return {"message": f"Bonjour {prenom}", "nb_clients": len(clients)}

        return routeur                           # ← ne l'oubliez pas

La fonction `dire_bonjour` est définie DANS `routes_bonjour` : elle voit donc `clients` (c'est une
« fermeture »). Elle lit le dictionnaire au moment de chaque requête, donc toujours à jour.

▶️  Essayer vos routes sans lancer de serveur, dans un terminal Python :

    >>> from defis.serie_12 import *
    >>> client = essayer(routes_lecture(exemple_clients()))
    >>> reponse = client.get("/clients/C001")
    >>> reponse.status_code, reponse.json()
    (200, {'id': 'C001', 'anciennete_mois': 3, ...})

`essayer` renvoie un `TestClient` : il parle à votre API comme le ferait un vrai programme, sans
réseau. C'est exactement l'outil des tests de `tests/test_api.py`. Pour voir la documentation
interactive de vos routes dans le navigateur : `python -m defis.serie_12`, puis ouvrez
http://localhost:8000/docs (Ctrl+C pour arrêter).

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Essayez vos routes avec `essayer(...)` comme ci-dessus.
3. Lancez `python scripts/defis.py --defi 74` (le numéro du défi) : ✅ ou ❌ avec l'explication.
4. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_12.py`.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import copy
from typing import Callable, Literal  # noqa: F401 — Literal : pour vos modèles (défis 76, 77, 79)

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Response, status  # noqa: F401
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field  # noqa: F401

CONTRATS = ("Mensuel", "Annuel", "Biannuel")
OFFRES = ("remise_10", "mois_offert", "appel_conseiller")

_EXEMPLE = {
    "C001": {"id": "C001", "anciennete_mois": 3, "facture_mensuelle": 89.9, "contrat": "Mensuel", "appels_support": 4},
    "C002": {"id": "C002", "anciennete_mois": 40, "facture_mensuelle": 45.0, "contrat": "Biannuel", "appels_support": 0},
    "C003": {"id": "C003", "anciennete_mois": 14, "facture_mensuelle": 70.5, "contrat": "Annuel", "appels_support": 2},
    "C004": {"id": "C004", "anciennete_mois": 1, "facture_mensuelle": 99.0, "contrat": "Mensuel", "appels_support": 1},
}


def exemple_clients() -> dict[str, dict]:
    """Une petite base de 4 clients, NEUVE à chaque appel : vos essais ne s'accumulent pas."""
    return copy.deepcopy(_EXEMPLE)


def essayer(routeur: APIRouter) -> TestClient:
    """Branche votre routeur sur une application et renvoie un client pour l'appeler.

    Si votre route plante, l'erreur et sa traceback s'affichent directement dans le terminal.
    """
    app = FastAPI()
    app.include_router(routeur)
    return TestClient(app)


# ============================================================================ Lire une ressource


def routes_lecture(clients: dict[str, dict]) -> APIRouter:
    """Défi 74 🟢 (Parcours 6.4) — Votre première route : `GET /clients/{client_id}`.

    Renvoyez un routeur avec une route `GET /clients/{client_id}` :

    - client connu   → 200, et le dictionnaire du client tel quel (FastAPI le convertit en JSON) ;
    - client inconnu → 404, avec un message qui cite l'identifiant demandé, par exemple
      « Client C999 introuvable ».

        >>> client = essayer(routes_lecture(exemple_clients()))
        >>> client.get("/clients/C002").json()["contrat"]
        'Biannuel'
        >>> client.get("/clients/C999").status_code
        404

    Ce que vous apprenez : les paramètres de chemin (`{client_id}` devient un argument de votre
    fonction), et à répondre une ERREUR avec le bon code. Pour un client inconnu, deux réflexes
    sont faux : `clients[client_id]` plante (KeyError) et l'appelant reçoit une erreur 500, « bug
    du serveur » ; `clients.get(client_id)` renvoie None, donc 200 avec `null`, et l'appelant croit
    que tout s'est bien passé.

    Indice : `raise HTTPException(status_code=404, detail="...")` interrompt la route et envoie
    `{"detail": "..."}` avec le code 404. Le module `status` donne des noms lisibles :
    `status.HTTP_404_NOT_FOUND`.
    """
    raise NotImplementedError


# ============================================================================ Chercher : paramètres de requête


def routes_recherche(clients: dict[str, dict]) -> APIRouter:
    """Défi 75 🟢 (Parcours 6.4) — Chercher des clients : `GET /clients?contrat=…&min_appels=…`.

    Renvoyez un routeur avec une route `GET /clients` qui renvoie la LISTE des clients qui
    correspondent aux trois paramètres de requête (après le `?` dans l'adresse) :

    - `contrat`    : facultatif. S'il est donné, ne garder que ce type de contrat ;
    - `min_appels` : entier, 0 par défaut, jamais négatif. Garder les clients qui ont AU MOINS ce
                     nombre d'appels au support ;
    - `limite`     : entier entre 1 et 100, 20 par défaut. Nombre maximum de clients renvoyés.

    Les clients sont triés par identifiant. Une recherche sans résultat n'est pas une erreur :
    200 et une liste vide. Une valeur interdite (`min_appels=-1`, `limite=0`, `limite=500`,
    `min_appels=beaucoup`) donne 422, sans que vous ayez à écrire un seul `if`.

        >>> client = essayer(routes_recherche(exemple_clients()))
        >>> [c["id"] for c in client.get("/clients?contrat=Mensuel&min_appels=2").json()]
        ['C001']
        >>> client.get("/clients?limite=0").status_code
        422

    Ce que vous apprenez : tout argument de la fonction qui n'est pas dans le chemin devient un
    paramètre de requête. Son TYPE et sa valeur par défaut font le travail : `int` convertit
    "2" en 2 (et refuse "beaucoup"), une valeur par défaut le rend facultatif, et
    `Query(ge=…, le=…)` pose des bornes. La validation se fait à la frontière, une fois pour toutes.

    Indice : `contrat: str | None = None`, `min_appels: int = Query(0, ge=0)`. Triez avec
    `sorted(clients.values(), key=...)`, puis coupez avec `[:limite]` APRÈS avoir filtré.
    """
    raise NotImplementedError


# ============================================================================ Créer : le corps de la requête


# Défi 76 : complétez ce modèle. Il décrit le JSON qu'on a le droit d'envoyer pour créer un client.
class NouveauClient(BaseModel):
    """Corps de `POST /clients`. Chaque champ : un type, et des bornes avec `Field(...)`.

    Exemple de champ :  anciennete_mois: int = Field(ge=0)
    """

    # Remplacez `pass` par les 5 champs décrits dans le défi 76.
    pass


def routes_creation(clients: dict[str, dict]) -> APIRouter:
    """Défi 76 🟡 (Parcours 6.4) — Créer un client : `POST /clients`, validé par Pydantic.

    1. Complétez le modèle `NouveauClient` juste au-dessus. Le JSON envoyé contient :

       - `id`               : texte de la forme « C » suivi de 3 chiffres (C042) ;
       - `anciennete_mois`  : entier, 0 ou plus ;
       - `facture_mensuelle`: nombre strictement positif, au plus 500 ;
       - `contrat`          : exactement « Mensuel », « Annuel » ou « Biannuel » ;
       - `appels_support`   : entier, 0 ou plus ; FACULTATIF, 0 par défaut.

       Tout autre champ est refusé : une faute de frappe comme « apels_support » ne doit pas
       passer inaperçue (sinon le client est créé avec 0 appel, et personne ne le saura).

    2. Écrivez la route `POST /clients` qui reçoit un `NouveauClient` :

       - données valides → enregistrez le client dans `clients` (un dictionnaire ordinaire, comme
         les autres), puis répondez 201 avec le client créé, et l'en-tête
         `Location: /clients/C042` (l'adresse où le relire) ;
       - identifiant déjà pris → 409, et le client existant n'est PAS écrasé ;
       - données invalides → 422 (Pydantic s'en charge), et rien n'est enregistré.

        >>> base = exemple_clients()
        >>> client = essayer(routes_creation(base))
        >>> nouveau = {"id": "C010", "anciennete_mois": 0, "facture_mensuelle": 59.9, "contrat": "Annuel"}
        >>> reponse = client.post("/clients", json=nouveau)
        >>> reponse.status_code, reponse.headers["location"], base["C010"]["appels_support"]
        (201, '/clients/C010', 0)

    Ce que vous apprenez : décrire un corps JSON avec un modèle Pydantic. FastAPI le lit, le
    convertit, le valide et répond 422 en détaillant chaque champ fautif, avant même d'appeler votre
    fonction. Votre code ne reçoit que des données propres. Et le bon code de succès pour une
    création : 201, pas 200.

    Indice : `Field(pattern=r"^C\\d{3}$")`, `Field(gt=0, le=500)`, `Literal["Mensuel", ...]`, et
    `model_config = ConfigDict(extra="forbid")` dans la classe. Pour la route :
    `@routeur.post("/clients", status_code=201)`, un argument `nouveau: NouveauClient`,
    `nouveau.model_dump()` pour obtenir un dictionnaire, et un argument `response: Response` pour
    écrire `response.headers["Location"] = ...`.
    """
    raise NotImplementedError


# ============================================================================ Modifier : PATCH partiel


def routes_modification(clients: dict[str, dict]) -> APIRouter:
    """Défi 77 🟡 — Modifier une partie d'un client : `PATCH /clients/{client_id}`.

    Le service client change le contrat de C001, et seulement son contrat :

        PATCH /clients/C001    {"contrat": "Annuel"}

    Écrivez la route (et le modèle Pydantic de son corps, à vous de le définir) :

    - client inconnu → 404 ;
    - les champs envoyés parmi `anciennete_mois`, `facture_mensuelle`, `contrat` et
      `appels_support` sont modifiés, avec les mêmes règles qu'au défi 76 ; les champs NON envoyés
      gardent leur valeur ; réponse 200 avec le client complet après modification ;
    - une valeur invalide → 422, et le client reste exactement comme avant (rien à moitié) ;
    - un champ envoyé à `null` → 422 : `{"facture_mensuelle": null}` effacerait la facture ;
    - `id` (ou tout autre champ inconnu) dans le corps → 422 : on ne change pas l'identifiant
      d'un client.

        >>> base = exemple_clients()
        >>> client = essayer(routes_modification(base))
        >>> client.patch("/clients/C001", json={"contrat": "Annuel"}).json()["facture_mensuelle"]
        89.9

    Ce que vous apprenez : PUT remplace toute la ressource, PATCH n'en change qu'une partie. Le
    piège classique : un modèle où tous les champs valent None par défaut, puis
    `modele.model_dump()`, qui renvoie AUSSI les champs non envoyés (à None) et efface les autres
    valeurs du client. Il faut distinguer « champ absent » de « champ envoyé ».

    Indice : `modele.model_dump(exclude_unset=True)` ne garde que les champs réellement envoyés.
    Pour refuser `null`, gardez un type non facultatif avec une valeur par défaut :
    `facture_mensuelle: float = Field(default=None, gt=0, le=500)` (Pydantic ne valide pas la
    valeur par défaut, mais refuse un `null` envoyé). N'oubliez pas `extra="forbid"`.
    """
    raise NotImplementedError


# ============================================================================ Protéger : clé d'API et Depends


def routes_admin(clients: dict[str, dict], cles_valides: set[str]) -> APIRouter:
    """Défi 78 🟡 — Protéger des routes avec une clé d'API : `Depends`, 401 contre 403.

    Deux routes réservées à l'équipe data, qui envoie sa clé dans l'en-tête `X-API-Key` :

    - `GET /admin/clients`              → 200 et la liste de tous les clients, triés par identifiant ;
    - `DELETE /admin/clients/{client_id}` → 204 sans contenu, et le client est supprimé ;
                                            404 s'il n'existe pas.

    Les deux routes refusent l'accès, AVANT de faire quoi que ce soit :

    - aucune clé    → 401 (« qui êtes-vous ? ») ;
    - clé inconnue  → 403 (« vous n'avez pas le droit ») ; et surtout, rien n'est supprimé !

    `cles_valides` peut changer pendant que l'API tourne : une clé retirée de l'ensemble (une clé
    volée, que l'on révoque) doit être refusée dès la requête suivante.

        >>> base = exemple_clients()
        >>> client = essayer(routes_admin(base, {"cle-equipe-data"}))
        >>> client.get("/admin/clients").status_code
        401
        >>> client.delete("/admin/clients/C004", headers={"X-API-Key": "cle-equipe-data"}).status_code
        204

    Ce que vous apprenez : une dépendance (`Depends`) est une fonction que FastAPI appelle AVANT
    la route. Si elle lève une HTTPException, la route ne s'exécute jamais. On écrit le contrôle
    une seule fois et on le branche sur toutes les routes qui en ont besoin : impossible d'oublier
    une route. Et l'ordre compte : un intrus qui reçoit 404 pour C999 mais 403 pour C001 apprend
    quels clients existent. On vérifie l'accès d'abord.

    Indice : une fonction `verifier_cle(x_api_key: str | None = Header(default=None))` (FastAPI
    lit l'en-tête `X-API-Key` : les `_` deviennent des `-`), puis
    `APIRouter(dependencies=[Depends(verifier_cle)])` protège toutes les routes du routeur. Pour 204 :
    `@routeur.delete(..., status_code=204)` et la fonction ne renvoie rien. Bonus de pro : comparez
    les clés avec `secrets.compare_digest`, qui prend le même temps que la clé soit presque juste ou
    complètement fausse.
    """
    raise NotImplementedError


# ============================================================================ Rejouer sans risque : idempotence


def routes_relances(clients: dict[str, dict], envoyer: Callable[[str, str], None]) -> APIRouter:
    """Défi 79 🔴 — Une relance qu'on peut renvoyer sans risque : l'en-tête `Idempotency-Key`.

    L'outil marketing appelle `POST /relances` pour proposer une offre à un client à risque :

        POST /relances      Idempotency-Key: 7f3c-…      {"client_id": "C001", "offre": "remise_10"}

    Votre route appelle `envoyer(client_id, offre)` (fourni : il envoie vraiment un SMS) et
    répond 201 avec `{"relance_id": …, "client_id": …, "offre": …}`. `relance_id` est un texte
    unique de votre choix (« R1 », « R2 »… ou `uuid.uuid4().hex`).

    Le problème : le réseau coupe parfois APRÈS l'envoi, mais AVANT que la réponse n'arrive.
    L'outil marketing ne sait pas si la relance est partie, alors il réessaie. Sans précaution, le
    client reçoit deux SMS. La solution des API de paiement : l'appelant joint à chaque requête un
    identifiant unique, l'en-tête `Idempotency-Key`, et le RÉUTILISE quand il réessaie.

    Règles (ce sont celles de la norme en préparation à l'IETF, et de Stripe) :

    1. Pas d'en-tête `Idempotency-Key` → 400 (ou 422), et rien n'est envoyé.
    2. Client inconnu → 404 ; offre hors de `OFFRES` → 422. Rien n'est envoyé.
    3. Première requête avec une clé → `envoyer` est appelé UNE fois, réponse 201.
    4. Même clé, même corps → la MÊME réponse (même code, même `relance_id`), sans renvoyer de SMS.
    5. Même clé, corps DIFFÉRENT → 422 (ou 409) : c'est une erreur de l'appelant, rien n'est envoyé.
    6. Nouvelle clé → nouvelle relance, même si le corps est identique (l'appelant le veut vraiment).
    7. `envoyer` lève une exception (panne de l'opérateur SMS) → 503, et la clé n'est PAS
       retenue : quand l'appelant réessaie avec la même clé, l'envoi est tenté à nouveau.
    8. Deux routeurs (deux applications, deux bases) ne partagent RIEN : la mémoire des clés vit
       dans `routes_relances`, pas dans une variable globale du module.

        >>> envois = []
        >>> client = essayer(routes_relances(exemple_clients(), lambda c, o: envois.append((c, o))))
        >>> corps, entete = {"client_id": "C001", "offre": "remise_10"}, {"Idempotency-Key": "k1"}
        >>> premiere = client.post("/relances", json=corps, headers=entete).json()
        >>> client.post("/relances", json=corps, headers=entete).json() == premiere, len(envois)
        (True, 1)

    Ce que vous apprenez : une opération est IDEMPOTENTE si la faire deux fois revient à la faire une
    fois. GET, PUT et DELETE le sont par nature ; POST non. Les systèmes réels réessaient (réseau,
    file de messages, utilisateur qui clique deux fois) : tout effet de bord coûteux (paiement,
    e-mail, SMS) doit le supporter. Et l'ordre des opérations décide de tout : on ne mémorise une
    réponse qu'une fois le travail réussi.

    Indice : un dictionnaire `deja_vu = {}` créé dans `routes_relances` (pas au niveau du module)
    associe chaque clé à `(corps, réponse)`. Le corps se compare facilement avec
    `demande.model_dump()`. Entourez l'appel à `envoyer` d'un `try` / `except Exception`, et ne
    remplissez `deja_vu` qu'après un envoi réussi.
    """
    raise NotImplementedError


if __name__ == "__main__":  # python -m defis.serie_12 → documentation sur http://localhost:8000/docs
    import uvicorn

    app = FastAPI(title="Série 12 — vos routes")
    base = exemple_clients()
    for fabrique in (
        lambda: routes_lecture(base),
        lambda: routes_recherche(base),
        lambda: routes_creation(base),
        lambda: routes_modification(base),
        lambda: routes_admin(base, {"cle-equipe-data"}),
        lambda: routes_relances(base, lambda c, o: print(f"📱 SMS envoyé à {c} : {o}")),
    ):
        try:
            app.include_router(fabrique())
        except NotImplementedError:
            pass  # défi pas encore fait : ses routes n'apparaissent pas
    uvicorn.run(app, port=8000)

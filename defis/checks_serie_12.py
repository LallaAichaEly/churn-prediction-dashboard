"""Vérifications automatiques de la série 12 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Le correcteur fabrique votre routeur sur une petite base de clients, le branche sur une application
FastAPI, puis l'appelle avec `TestClient`, comme le ferait un vrai programme. Pour chaque requête,
il compare le code HTTP et le contenu au contrat décrit dans la docstring, puis regarde ce que la
requête a changé dans la base (une requête refusée ne doit RIEN changer).
"""

from __future__ import annotations

import copy
import random
import traceback
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from .verif import Defi, Serie

_BASE = {
    "C001": {"id": "C001", "anciennete_mois": 3, "facture_mensuelle": 89.9, "contrat": "Mensuel", "appels_support": 4},
    "C002": {"id": "C002", "anciennete_mois": 40, "facture_mensuelle": 45.0, "contrat": "Biannuel", "appels_support": 0},
    "C003": {"id": "C003", "anciennete_mois": 14, "facture_mensuelle": 70.5, "contrat": "Annuel", "appels_support": 2},
    "C004": {"id": "C004", "anciennete_mois": 1, "facture_mensuelle": 99.0, "contrat": "Mensuel", "appels_support": 1},
}
_CONTRATS = ("Mensuel", "Annuel", "Biannuel")


def _base() -> dict[str, dict]:
    return copy.deepcopy(_BASE)


def _grande_base(n: int = 30, seed: int = 12) -> dict[str, dict]:
    """30 clients insérés dans le DÉSORDRE : l'ordre du dictionnaire n'est pas l'ordre des identifiants."""
    rng = random.Random(seed)
    ids = [f"C{i:03d}" for i in range(1, n + 1)]
    rng.shuffle(ids)
    return {
        cid: {"id": cid, "anciennete_mois": rng.randint(0, 60), "facture_mensuelle": round(rng.uniform(20, 120), 2),
              "contrat": _CONTRATS[int(cid[1:]) % 3], "appels_support": int(cid[1:]) % 6}
        for cid in ids
    }


# ============================================================================ Outils communs


def _fabriquer(f: Callable, *args) -> APIRouter:
    routeur = f(*args)
    if routeur is None:
        raise AssertionError("Votre fonction renvoie None : avez-vous oublié `return routeur` à la fin ?")
    if isinstance(routeur, FastAPI):
        raise AssertionError("Votre fonction renvoie une application FastAPI : renvoyez un `APIRouter()`, que "
                             "l'application branchera avec `app.include_router(...)`.")
    if not isinstance(routeur, APIRouter):
        raise AssertionError(f"Votre fonction doit renvoyer un `APIRouter`, pas un objet de type {type(routeur).__name__}.")
    return routeur


def _segments(chemin: str) -> list[str]:
    return ["{}" if s.startswith("{") and s.endswith("}") else s for s in chemin.strip("/").split("/")]


def _exiger_route(routeur: APIRouter, methode: str, chemin: str) -> None:
    """Le nom du paramètre de chemin est libre ({client_id} ou {id}) ; la forme de l'adresse, non."""
    routes = [(m, getattr(r, "path", "")) for r in routeur.routes for m in sorted(getattr(r, "methods", None) or ())]
    if any(m == methode and _segments(p) == _segments(chemin) for m, p in routes):
        return
    trouvees = ", ".join(f"{m} {p}" for m, p in routes) or "aucune route"
    meme_chemin = [m for m, p in routes if _segments(p) == _segments(chemin)]
    conseil = (f" Le chemin existe, mais avec {', '.join(meme_chemin)} : le décorateur doit être `@routeur.{methode.lower()}`."
               if meme_chemin else " Vérifiez le décorateur (`@routeur.get`, `@routeur.post`…) et le chemin, à la lettre.")
    raise AssertionError(f"Route `{methode} {chemin}` introuvable dans votre routeur (il contient : {trouvees}).{conseil}")


def _client(routeur: APIRouter) -> TestClient:
    app = FastAPI()
    app.include_router(routeur)
    return TestClient(app)  # raise_server_exceptions=True : un plantage remonte jusqu'à _appel


def _ligne_eleve(exc: BaseException) -> str:
    cadres = [c for c in traceback.extract_tb(exc.__traceback__)
              if "-packages" not in c.filename and Path(c.filename).name != Path(__file__).name]
    return f", ligne {cadres[-1].lineno}" if cadres else ""


def _appel(client: TestClient, methode: str, url: str, si_plantage: str = "", **kwargs):
    """Envoie la requête. Si la route plante, l'appelant aurait reçu une erreur 500 : on l'explique."""
    try:
        return client.request(methode, url, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001 — n'importe quel plantage de la route de l'élève
        raise AssertionError(
            f"Votre route a planté pendant `{methode} {url}` ({type(exc).__name__}: {exc}{_ligne_eleve(exc)}). "
            f"L'appelant reçoit une erreur 500, « bug du serveur », sans savoir ce qu'il a fait de travers. {si_plantage}"
        ) from None


def _resume(reponse) -> str:
    texte = reponse.text if len(reponse.text) <= 160 else reponse.text[:157] + "…"
    return f"{reponse.status_code} {texte}"


def _code(reponse, attendus: int | tuple[int, ...], requete: str, conseils: dict[int, str] | None = None) -> None:
    attendus = (attendus,) if isinstance(attendus, int) else attendus
    if reponse.status_code in attendus:
        return
    conseil = (conseils or {}).get(reponse.status_code, "")
    if not conseil and reponse.status_code == 404 and reponse.text == '{"detail":"Not Found"}':
        conseil = "Ce « Not Found » est la réponse de FastAPI quand AUCUNE route ne correspond à l'adresse : vérifiez le chemin."
    if not conseil and reponse.status_code == 422:
        conseil = "FastAPI a refusé la requête avant même d'appeler votre fonction : lisez le `detail` ci-dessus."
    attendu = " ou ".join(map(str, attendus))
    raise AssertionError(f"`{requete}` doit répondre {attendu}, or votre route répond {_resume(reponse)}. {conseil}".rstrip())


def _base_intacte(base: dict, avant: dict, requete: str, pourquoi: str) -> None:
    if base != avant:
        changes = sorted(k for k in set(base) | set(avant) if base.get(k) != avant.get(k))
        raise AssertionError(f"Après `{requete}`, la base a changé (clients touchés : {', '.join(changes)}) alors que "
                             f"la requête devait être refusée. {pourquoi}")


# ============================================================================ Défi 74 : lire


def verifier_routes_lecture(f) -> None:
    base = _base()
    routeur = _fabriquer(f, base)
    _exiger_route(routeur, "GET", "/clients/{client_id}")
    client = _client(routeur)

    r = _appel(client, "GET", "/clients/C002")
    _code(r, 200, "GET /clients/C002")
    if r.json() != _BASE["C002"]:
        raise AssertionError(f"`GET /clients/C002` renvoie {r.json()} au lieu du client C002 : {_BASE['C002']}. "
                             "Renvoyez `clients[client_id]` tel quel.")

    base["C050"] = {**_BASE["C001"], "id": "C050"}
    r = _appel(client, "GET", "/clients/C050")
    if r.status_code != 200 or (r.json() or {}).get("id") != "C050":
        raise AssertionError("Un client ajouté APRÈS la création du routeur est introuvable. Votre route doit lire le "
                             "dictionnaire `clients` au moment de chaque requête, pas une copie faite à l'avance.")

    avant = copy.deepcopy(base)
    r = _appel(client, "GET", "/clients/C999",
               si_plantage="Pour un client inconnu, testez `if client_id not in clients:` puis "
                           "`raise HTTPException(status_code=404, detail=...)`.")
    if r.status_code == 200 and r.json() is None:
        raise AssertionError("`GET /clients/C999` répond 200 avec `null` : l'appelant croit que la requête a réussi et "
                             "plantera plus loin, sans comprendre pourquoi. Un client inconnu, c'est 404 : "
                             "`raise HTTPException(status_code=404, detail=...)`.")
    _code(r, 404, "GET /clients/C999", {
        400: "400 signifie « requête mal formée » ; ici elle est correcte, c'est la ressource qui n'existe pas : 404.",
        200: "Un client inconnu, c'est 404 : `raise HTTPException(status_code=404, detail=...)`.",
    })
    if "C999" not in str(r.json().get("detail", "")):
        raise AssertionError(f"Le message d'erreur ({r.json()}) doit citer l'identifiant demandé, par exemple "
                             "« Client C999 introuvable » : dans les journaux de l'appelant, c'est ce qui fait gagner du temps.")
    _base_intacte(base, avant, "GET /clients/C999", "Une lecture ne modifie jamais la base.")


# ============================================================================ Défi 75 : chercher


def _attendu_recherche(base: dict, contrat=None, min_appels=0, limite=20, *, strict=False, trier=True,
                       couper_avant=False) -> list[str]:
    """Résultat attendu, ou celui d'une erreur classique (strict, trier, couper_avant) pour la reconnaître."""
    ids = sorted(base) if trier else list(base)
    if couper_avant:
        ids = ids[:limite]
    garder = [i for i in ids if (contrat is None or base[i]["contrat"] == contrat)
              and (base[i]["appels_support"] > min_appels if strict else base[i]["appels_support"] >= min_appels)]
    return garder[:limite]


def _ids(reponse, requete: str) -> list[str]:
    donnees = reponse.json()
    if not isinstance(donnees, list) or not all(isinstance(c, dict) and "id" in c for c in donnees):
        raise AssertionError(f"`{requete}` doit renvoyer une LISTE de clients (des dictionnaires complets), "
                             f"pas {str(donnees)[:120]}.")
    return [c["id"] for c in donnees]


def _diagnostic_recherche(base: dict, obtenu: list[str], params: dict) -> str:
    """Reconnaît l'erreur classique qui donne exactement ce résultat, pour la nommer."""
    erreurs = [
        ({"strict": True}, "« Au moins N appels » s'écrit `>=` : avec `>`, les clients à exactement N appels "
                           "disparaissent (et ceux à 0 appel quand min_appels vaut 0)."),
        ({"couper_avant": True}, "Filtrez d'abord, coupez ENSUITE avec `[:limite]` : couper avant de filtrer fait "
                                 "perdre des résultats."),
        ({"trier": False}, "Triez par identifiant AVANT de couper : l'ordre d'insertion d'un dictionnaire n'est pas "
                           "une promesse de l'API. `sorted(..., key=lambda c: c[\"id\"])`."),
        ({"limite": 10**6}, "`limite` fixe le nombre MAXIMUM de clients renvoyés, 20 par défaut : "
                            "`limite: int = Query(20, ge=1, le=100)`, puis `[:limite]`."),
    ]
    for variante, conseil in erreurs:
        if obtenu == _attendu_recherche(base, **{**params, **variante}):
            return conseil
    attendu = _attendu_recherche(base, **params)
    if sorted(obtenu) == sorted(attendu):
        return ("Les bons clients, mais pas dans l'ordre des identifiants : triez avec "
                "`sorted(..., key=lambda c: c[\"id\"])` pour une réponse prévisible.")
    if params.get("contrat") and any(base[i]["contrat"] != params["contrat"] for i in obtenu if i in base):
        return f"Certains clients renvoyés n'ont pas le contrat {params['contrat']} : appliquez le filtre `contrat`."
    return "Filtrez (contrat, min_appels), triez par identifiant, puis gardez les `limite` premiers."


def verifier_routes_recherche(f) -> None:
    base = _grande_base()
    routeur = _fabriquer(f, base)
    _exiger_route(routeur, "GET", "/clients")
    client = _client(routeur)

    cas = [
        ("", {}, "Sans aucun paramètre, la recherche doit marcher : donnez une valeur par défaut à chacun "
                 "(`contrat: str | None = None`, `min_appels: int = Query(0, ge=0)`…)."),
        ("contrat=Annuel", {"contrat": "Annuel"}, "`contrat` est du texte : `contrat: str | None = None`."),
        ("min_appels=3", {"min_appels": 3}, "`min_appels: int = Query(0, ge=0)`."),
        ("limite=5", {"limite": 5}, "`limite: int = Query(20, ge=1, le=100)`."),
        ("contrat=Mensuel&min_appels=2&limite=4", {"contrat": "Mensuel", "min_appels": 2, "limite": 4},
         "Les trois paramètres peuvent être donnés ensemble."),
        ("limite=100", {"limite": 100}, "100 est la limite maximale AUTORISÉE : `le=100`, pas `lt=100`."),
        ("limite=1", {"limite": 1}, "1 est la plus petite limite autorisée : `ge=1`."),
    ]
    for requete, params, si_refus in cas:
        url = f"/clients?{requete}" if requete else "/clients"
        r = _appel(client, "GET", url)
        _code(r, 200, f"GET {url}", {422: si_refus})
        obtenu, attendu = _ids(r, f"GET {url}"), _attendu_recherche(base, **params)
        if obtenu != attendu:
            montrer = (lambda ids: f"{ids[:6]}… ({len(ids)} clients)" if len(ids) > 6 else str(ids))
            raise AssertionError(f"`GET {url}` renvoie {montrer(obtenu)} au lieu de {montrer(attendu)}. "
                                 f"{_diagnostic_recherche(base, obtenu, params)}")
        if obtenu and r.json()[0] != base[obtenu[0]]:
            raise AssertionError(f"Chaque élément de la liste doit être le client complet, comme dans la base : "
                                 f"{base[obtenu[0]]}, et non {r.json()[0]}.")

    r = _appel(client, "GET", "/clients?contrat=Hebdomadaire")
    _code(r, 200, "GET /clients?contrat=Hebdomadaire", {
        404: "Une recherche sans résultat n'est pas une erreur : 200 et une liste vide `[]`. (404 voudrait dire que "
             "l'adresse /clients elle-même n'existe pas.)"})
    if r.json() != []:
        raise AssertionError(f"Aucun client n'a le contrat « Hebdomadaire » : la réponse doit être `[]`, pas {r.json()}.")

    for requete, conseil in [
        ("min_appels=-1", "Un nombre d'appels négatif n'a pas de sens : `min_appels: int = Query(0, ge=0)`."),
        ("min_appels=beaucoup", "Déclarez le type : `min_appels: int`. FastAPI convertit « 2 » en 2 et refuse « beaucoup »."),
        ("limite=0", "Une limite de 0 ne renverrait jamais rien : `Query(20, ge=1, le=100)`."),
        ("limite=500", "Sans plafond, un appel `limite=1000000` peut faire tomber le serveur : `le=100`."),
    ]:
        r = _appel(client, "GET", f"/clients?{requete}")
        _code(r, 422, f"GET /clients?{requete}", {200: conseil, 400: f"FastAPI répond déjà 422 si vous déclarez le "
                                                                     f"type et les bornes. {conseil}"})


# ============================================================================ Défi 76 : créer


_NOUVEAU = {"id": "C010", "anciennete_mois": 0, "facture_mensuelle": 59.9, "contrat": "Annuel"}


def verifier_routes_creation(f) -> None:
    base = _base()
    routeur = _fabriquer(f, base)
    _exiger_route(routeur, "POST", "/clients")
    client = _client(routeur)
    attendu = {**_NOUVEAU, "appels_support": 0}

    r = _appel(client, "POST", "/clients", json=_NOUVEAU, si_plantage="Avez-vous enregistré `nouveau.model_dump()` ?")
    _code(r, 201, "POST /clients (client valide, sans appels_support)", {
        200: "Une création réussie répond 201 Created : `@routeur.post(\"/clients\", status_code=201)`.",
        422: "Ce client est valide : `appels_support` est FACULTATIF (`Field(default=0, ge=0)`). Vérifiez aussi les "
             "noms des champs, à la lettre.",
    })
    if "C010" not in base:
        raise AssertionError("La réponse est 201, mais le client C010 n'est pas dans la base : `clients[nouveau.id] = ...`.")
    if not isinstance(base["C010"], dict):
        raise AssertionError(f"La base contient un objet {type(base['C010']).__name__} pour C010 : enregistrez un "
                             "dictionnaire ordinaire (`nouveau.model_dump()`), comme les autres clients. Les autres "
                             "routes et le reste du code attendent des dictionnaires.")
    if base["C010"] != attendu:
        raise AssertionError(f"Client enregistré : {base['C010']} ; attendu : {attendu} "
                             "(`appels_support` vaut 0 quand il n'est pas envoyé).")
    if r.json() != attendu:
        raise AssertionError(f"La réponse doit contenir le client créé ({attendu}), pas {r.json()}.")
    location = r.headers.get("location")
    if location != "/clients/C010":
        raise AssertionError(f"L'en-tête `Location` vaut {location!r} au lieu de '/clients/C010' : il indique à "
                             "l'appelant où relire ce qu'il vient de créer. Ajoutez un argument `response: Response` "
                             "et écrivez `response.headers[\"Location\"] = f\"/clients/{nouveau.id}\"`.")

    r = _appel(client, "POST", "/clients", json={**_NOUVEAU, "id": "C011", "appels_support": 3, "facture_mensuelle": 500})
    _code(r, 201, "POST /clients (appels_support 3, facture 500 €)", {
        422: "500 € est AUTORISÉ (« au plus 500 ») : `le=500`, pas `lt=500`."})
    if base.get("C011", {}).get("appels_support") != 3:
        raise AssertionError("Quand `appels_support` est envoyé, c'est sa valeur qui doit être enregistrée.")

    avant = copy.deepcopy(base)
    doublon = {**_NOUVEAU, "id": "C001", "facture_mensuelle": 10.0}
    r = _appel(client, "POST", "/clients", json=doublon)
    if r.status_code in (200, 201):
        raise AssertionError("`POST /clients` avec l'identifiant C001, déjà pris, a été accepté : vous avez ÉCRASÉ un "
                             "vrai client. Testez `if nouveau.id in clients:` et répondez 409.")
    _code(r, 409, "POST /clients (identifiant C001 déjà pris)", {
        400: "La requête est bien formée : c'est l'état de la base qui l'empêche. Le code juste est 409 Conflict.",
        422: "Les données sont valides : c'est l'état de la base qui empêche la création. Le code juste est 409 Conflict.",
    })
    _base_intacte(base, avant, "POST /clients (C001 déjà pris)", "Le client existant ne doit pas être modifié.")

    # Des identifiants libres (C090) : un refus doit venir de la validation, pas du doublon C010.
    libre = {**_NOUVEAU, "id": "C090"}
    invalides = [
        ({**_NOUVEAU, "id": "42"}, "un id qui ne commence pas par C", "`Field(pattern=r\"^C\\d{3}$\")`."),
        ({**_NOUVEAU, "id": "C0420"}, "l'id C0420 (4 chiffres)",
         "Un `pattern` est cherché N'IMPORTE OÙ dans le texte : ancrez-le avec ^ (début) et $ (fin) : r\"^C\\d{3}$\"."),
        ({**_NOUVEAU, "id": "XC042"}, "l'id XC042", "Ancrez le motif au début du texte avec ^ : r\"^C\\d{3}$\"."),
        ({**libre, "anciennete_mois": -1}, "une ancienneté de -1 mois", "`anciennete_mois: int = Field(ge=0)`."),
        ({**libre, "facture_mensuelle": 0}, "une facture de 0 €",
         "« Strictement positif » : `gt=0` (greater than), pas `ge=0` (greater or equal)."),
        ({**libre, "facture_mensuelle": 500.01}, "une facture de 500,01 €", "Au plus 500 : `le=500`."),
        ({**libre, "contrat": "Mensual"}, "le contrat « Mensual » (faute de frappe)",
         "`contrat: Literal[\"Mensuel\", \"Annuel\", \"Biannuel\"]` : toute autre valeur est refusée."),
        ({k: v for k, v in libre.items() if k != "contrat"}, "un client sans contrat",
         "Un champ sans valeur par défaut est obligatoire : ne donnez pas de défaut à `contrat`."),
        ({**libre, "apels_support": 3}, "le champ « apels_support » (faute de frappe)",
         "Par défaut, Pydantic IGNORE les champs inconnus : le client serait créé avec 0 appel, en silence. Ajoutez "
         "`model_config = ConfigDict(extra=\"forbid\")` dans `NouveauClient`."),
        ({**libre, "appels_support": -2}, "-2 appels au support", "`appels_support: int = Field(default=0, ge=0)`."),
    ]
    for corps, quoi, conseil in invalides:
        avant = copy.deepcopy(base)
        r = _appel(client, "POST", "/clients", json=corps)
        _code(r, 422, f"POST /clients avec {quoi}", {201: f"Ce client aurait dû être refusé (422). {conseil}",
                                                     200: f"Ce client aurait dû être refusé (422). {conseil}"})
        _base_intacte(base, avant, f"POST /clients avec {quoi}", "Rien ne doit être enregistré.")


# ============================================================================ Défi 77 : modifier


def verifier_routes_modification(f) -> None:
    base = _base()
    routeur = _fabriquer(f, base)
    _exiger_route(routeur, "PATCH", "/clients/{client_id}")
    client = _client(routeur)

    attendu = {**_BASE["C001"], "contrat": "Annuel"}
    r = _appel(client, "PATCH", "/clients/C001", json={"contrat": "Annuel"})
    _code(r, 200, "PATCH /clients/C001 {\"contrat\": \"Annuel\"}", {
        422: "Les autres champs ne sont pas envoyés : ils doivent être FACULTATIFS dans votre modèle "
             "(une valeur par défaut pour chacun, par exemple `Field(default=None, ge=0)`)."})
    if base["C001"] != attendu:
        efface = [k for k, v in base["C001"].items() if v is None]
        if efface:
            raise AssertionError(f"Après ce PATCH, {', '.join(efface)} valent None : vous avez écrit les champs NON "
                                 "envoyés. Utilisez `modification.model_dump(exclude_unset=True)`, qui ne garde que "
                                 "les champs réellement présents dans le JSON.")
        raise AssertionError(f"Après `PATCH /clients/C001 {{\"contrat\": \"Annuel\"}}`, le client vaut {base['C001']} "
                             f"au lieu de {attendu}.")
    if r.json() != attendu:
        raise AssertionError(f"La réponse doit être le client COMPLET après modification ({attendu}), pas {r.json()}.")

    attendu = {**_BASE["C003"], "facture_mensuelle": 75.0, "appels_support": 0}
    r = _appel(client, "PATCH", "/clients/C003", json={"facture_mensuelle": 75.0, "appels_support": 0})
    _code(r, 200, "PATCH /clients/C003 (facture et appels)")
    if base["C003"] != attendu:
        conseil = (" 0 est une VALEUR, pas une absence : un `if valeur:` ou `if v` l'ignore. Ne testez pas les "
                   "valeurs, utilisez `exclude_unset=True`." if base["C003"]["appels_support"] == 2 else "")
        raise AssertionError(f"Après `PATCH /clients/C003 {{\"facture_mensuelle\": 75.0, \"appels_support\": 0}}`, le "
                             f"client vaut {base['C003']} au lieu de {attendu}.{conseil}")

    avant = copy.deepcopy(base)
    r = _appel(client, "PATCH", "/clients/C999", json={"contrat": "Annuel"},
               si_plantage="Testez `if client_id not in clients:` et répondez 404 AVANT toute modification.")
    _code(r, 404, "PATCH /clients/C999")
    _base_intacte(base, avant, "PATCH /clients/C999", "On ne crée pas un client en le modifiant.")

    refus = [
        ({"facture_mensuelle": -5}, "une facture négative", "Mêmes règles qu'au défi 76 : `Field(default=None, gt=0, le=500)`."),
        ({"contrat": "Mensual"}, "le contrat « Mensual »", "Utilisez le même `Literal` qu'au défi 76."),
        ({"anciennete_mois": 12, "facture_mensuelle": 0}, "une ancienneté valide ET une facture de 0 €",
         "Tout ou rien : si un champ est invalide, AUCUN ne doit changer. Laissez Pydantic valider tout le corps avant "
         "d'entrer dans la fonction, plutôt que de vérifier et d'écrire champ par champ."),
        ({"facture_mensuelle": None}, "`\"facture_mensuelle\": null`",
         "Avec `float | None = None`, un `null` envoyé est accepté : selon votre code, il efface la facture ou il est "
         "ignoré en silence. Refusez-le : gardez un type non facultatif avec "
         "une valeur par défaut : `facture_mensuelle: float = Field(default=None, gt=0, le=500)` (Pydantic ne valide "
         "pas la valeur par défaut, mais refuse un `null` envoyé)."),
        ({"contrat": None}, "`\"contrat\": null`", "Un `null` ne doit jamais effacer une donnée : `contrat: Literal[...] = None`."),
        ({"id": "C777"}, "`\"id\": \"C777\"`",
         "On ne change pas l'identifiant d'un client. `model_config = ConfigDict(extra=\"forbid\")` refuse tout champ "
         "inconnu du modèle, y compris `id`."),
        ({"contrat": "Annuel", "telephone": "0600000000"}, "un champ inconnu « telephone »",
         "Un champ inconnu doit être refusé, pas ignoré : `model_config = ConfigDict(extra=\"forbid\")`."),
    ]
    for corps, quoi, conseil in refus:
        avant = copy.deepcopy(base)
        r = _appel(client, "PATCH", "/clients/C004", json=corps, si_plantage=conseil)
        _code(r, 422, f"PATCH /clients/C004 avec {quoi}", {200: f"Cette modification devait être refusée (422). {conseil}"})
        _base_intacte(base, avant, f"PATCH /clients/C004 avec {quoi}", conseil)


# ============================================================================ Défi 78 : protéger


def verifier_routes_admin(f) -> None:
    base = _base()
    cles = {"cle-equipe-data", "cle-support"}
    routeur = _fabriquer(f, base, cles)
    _exiger_route(routeur, "GET", "/admin/clients")
    _exiger_route(routeur, "DELETE", "/admin/clients/{client_id}")
    client = _client(routeur)
    bonne, autre, fausse = {"X-API-Key": "cle-equipe-data"}, {"X-API-Key": "cle-support"}, {"X-API-Key": "cle-volee"}
    _inverses = ("401 et 403 sont inversés : 401 = « je ne sais pas qui vous êtes » (aucune clé), "
                 "403 = « je sais, et c'est non » (clé fausse).")
    inverse = {403: _inverses}          # attendu 401, obtenu 403
    inverse_403 = {401: _inverses}      # attendu 403, obtenu 401
    tout_trie = [_BASE[k] for k in sorted(_BASE)]

    r = _appel(client, "GET", "/admin/clients", headers=bonne)
    _code(r, 200, "GET /admin/clients avec une bonne clé", {
        401: "La clé est bien envoyée dans l'en-tête `X-API-Key`. Lisez-la avec `x_api_key: str | None = "
             "Header(default=None)` : FastAPI change les `_` du nom en `-`."})
    if r.json() != tout_trie:
        raise AssertionError(f"`GET /admin/clients` doit renvoyer tous les clients, triés par identifiant : "
                             f"{[c['id'] for c in tout_trie]}, pas {r.json()}.")
    _code(_appel(client, "GET", "/admin/clients", headers=autre), 200, "GET /admin/clients avec la clé cle-support",
          {403: "Toutes les clés de `cles_valides` donnent accès, pas seulement la première."})

    r = _appel(client, "GET", "/admin/clients",
               si_plantage="Sans en-tête, la clé vaut None : traitez ce cas en premier (401).")
    if r.status_code == 200:
        raise AssertionError("`GET /admin/clients` SANS clé répond 200 : n'importe qui lit la liste des clients ! "
                             "Écrivez une dépendance `verifier_cle` et branchez-la : `APIRouter(dependencies="
                             "[Depends(verifier_cle)])`.")
    _code(r, 401, "GET /admin/clients sans clé", {**inverse, 422: "Une clé absente n'est pas une donnée invalide : "
                                                  "déclarez `Header(default=None)` et levez vous-même une HTTPException(401)."})
    r = _appel(client, "GET", "/admin/clients", headers=fausse)
    if r.status_code == 200:
        raise AssertionError("`GET /admin/clients` avec une clé FAUSSE répond 200 : vérifiez que la clé fait partie de "
                             "`cles_valides`, pas seulement qu'elle est présente.")
    _code(r, 403, "GET /admin/clients avec une clé fausse", inverse_403)
    r = _appel(client, "GET", "/admin/clients?x_api_key=cle-equipe-data")
    _code(r, 401, "GET /admin/clients?x_api_key=… (clé dans l'adresse, pas dans l'en-tête)", {
        200: "La clé doit venir de l'EN-TÊTE `X-API-Key` (`Header`), pas de l'adresse (`Query`) : une adresse finit dans "
             "les journaux des serveurs et l'historique du navigateur, et la clé avec."})

    for entetes, code, quoi in [(fausse, 403, "une clé fausse"), ({}, 401, "sans clé")]:
        avant = copy.deepcopy(base)
        r = _appel(client, "DELETE", "/admin/clients/C004", headers=entetes)
        if r.status_code in (200, 204):
            raise AssertionError(f"`DELETE /admin/clients/C004` {quoi} a SUPPRIMÉ le client ! La route de suppression "
                                 "doit être protégée elle aussi : mettez la dépendance sur le routeur "
                                 "(`APIRouter(dependencies=[Depends(verifier_cle)])`), ainsi aucune route n'est oubliée.")
        _code(r, code, f"DELETE /admin/clients/C004 {quoi}", inverse if code == 401 else inverse_403)
        _base_intacte(base, avant, f"DELETE /admin/clients/C004 {quoi}", "Un accès refusé ne fait rien.")

    for entetes, code, quoi in [({}, 401, "sans clé"), (fausse, 403, "avec une clé fausse")]:
        r = _appel(client, "DELETE", "/admin/clients/C999", headers=entetes)
        _code(r, code, f"DELETE /admin/clients/C999 {quoi}", {
            404: "Vous vérifiez l'existence du client AVANT l'accès : un intrus qui reçoit 404 pour C999 mais 403 pour "
                 "C001 apprend quels clients existent. Contrôlez l'accès d'abord (une dépendance s'exécute avant la route)."})

    r = _appel(client, "DELETE", "/admin/clients/C004", headers=bonne,
               si_plantage="Pour supprimer : `del clients[client_id]` (après avoir vérifié qu'il existe).")
    _code(r, 204, "DELETE /admin/clients/C004 avec une bonne clé", {
        200: "Une suppression réussie qui n'a rien à renvoyer répond 204 No Content : "
             "`@routeur.delete(..., status_code=204)`, et la fonction ne renvoie rien."})
    if r.content:
        raise AssertionError(f"Une réponse 204 n'a PAS de contenu, or la vôtre contient {r.text!r}. Ne renvoyez rien.")
    if "C004" in base:
        raise AssertionError("`DELETE /admin/clients/C004` répond 204, mais C004 est toujours dans la base : "
                             "`del clients[client_id]`.")
    r = _appel(client, "DELETE", "/admin/clients/C004", headers=bonne,
               si_plantage="Le client a déjà été supprimé : testez `if client_id not in clients:` et répondez 404.")
    _code(r, 404, "DELETE /admin/clients/C004 une seconde fois (déjà supprimé)")

    cles.discard("cle-support")
    r = _appel(client, "GET", "/admin/clients", headers=autre)
    _code(r, 403, "GET /admin/clients avec la clé cle-support, RÉVOQUÉE (retirée de cles_valides)", {
        200: "Une clé retirée de `cles_valides` est encore acceptée : vous avez copié l'ensemble à la création du "
             "routeur. Relisez `cles_valides` à chaque requête, dans la dépendance."})
    cles.add("cle-nouvelle")
    r = _appel(client, "GET", "/admin/clients", headers={"X-API-Key": "cle-nouvelle"})
    _code(r, 200, "GET /admin/clients avec une clé ajoutée à cles_valides après la création du routeur", {
        403: "Relisez `cles_valides` à chaque requête : une clé ajoutée pendant que l'API tourne doit fonctionner."})


# ============================================================================ Défi 79 : idempotence


class _FauxEnvoi:
    """Remplace l'opérateur SMS : compte les envois et peut simuler une panne."""

    def __init__(self) -> None:
        self.envois: list[tuple[str, str]] = []
        self.tentatives = 0
        self.en_panne = False

    def __call__(self, client_id: str, offre: str) -> None:
        self.tentatives += 1
        if self.en_panne:
            raise ConnectionError("opérateur SMS injoignable")
        self.envois.append((client_id, offre))


def _relance(client: TestClient, corps: dict, cle: str | None, si_plantage: str = ""):
    entetes = {} if cle is None else {"Idempotency-Key": cle}
    return _appel(client, "POST", "/relances", json=corps, headers=entetes, si_plantage=si_plantage)


def _verifier_reponse_relance(r, corps: dict, requete: str) -> dict:
    donnees = r.json()
    if not isinstance(donnees, dict) or not donnees.get("relance_id") or not isinstance(donnees.get("relance_id"), str):
        raise AssertionError(f"`{requete}` doit renvoyer un objet avec un `relance_id` (texte non vide), "
                             f"`client_id` et `offre`, pas {donnees}.")
    if donnees.get("client_id") != corps["client_id"] or donnees.get("offre") != corps["offre"]:
        raise AssertionError(f"La réponse de `{requete}` doit reprendre `client_id` et `offre` : {donnees}.")
    return donnees


def verifier_routes_relances(f) -> None:
    base = _base()
    sms = _FauxEnvoi()
    routeur = _fabriquer(f, base, sms)
    _exiger_route(routeur, "POST", "/relances")
    client = _client(routeur)
    corps = {"client_id": "C001", "offre": "remise_10"}

    r = _relance(client, corps, None, si_plantage="Sans en-tête, la clé vaut None : répondez 400 avant tout le reste.")
    _code(r, (400, 422), "POST /relances sans en-tête Idempotency-Key", {
        201: "L'en-tête Idempotency-Key est obligatoire : sans lui, impossible de reconnaître un rejeu. Répondez 400."})
    if sms.tentatives:
        raise AssertionError("Sans en-tête Idempotency-Key, rien ne doit être envoyé : vérifiez l'en-tête AVANT d'appeler `envoyer`.")

    r = _relance(client, {"client_id": "C001", "offre": "voyage_offert"}, "k-offre")
    _code(r, 422, "POST /relances avec l'offre « voyage_offert » (inconnue)", {
        201: "L'offre doit faire partie de OFFRES : `offre: Literal[\"remise_10\", \"mois_offert\", \"appel_conseiller\"]`."})
    if sms.tentatives:
        raise AssertionError("Une offre invalide a été envoyée au client : validez le corps avec un modèle Pydantic.")

    r = _relance(client, {"client_id": "C999", "offre": "remise_10"}, "k-inconnu",
                 si_plantage="Vérifiez que `demande.client_id in clients` et répondez 404 sinon.")
    _code(r, 404, "POST /relances pour le client C999 (inconnu)")
    if sms.tentatives:
        raise AssertionError("Un SMS est parti vers un client inconnu : vérifiez qu'il existe AVANT d'appeler `envoyer`.")

    r = _relance(client, corps, "k1", si_plantage="Appelez `envoyer(demande.client_id, demande.offre)`.")
    _code(r, 201, "POST /relances (première requête, clé k1)", {200: "Une relance créée répond 201 : `status_code=201`."})
    premiere = _verifier_reponse_relance(r, corps, "POST /relances (clé k1)")
    if sms.envois != [("C001", "remise_10")]:
        raise AssertionError(f"La première requête doit appeler `envoyer(\"C001\", \"remise_10\")` exactement une fois. "
                             f"Envois constatés : {sms.envois}.")

    r = _relance(client, corps, "k1")
    if len(sms.envois) > 1:
        raise AssertionError("La même requête rejouée avec la MÊME clé (k1) a envoyé un DEUXIÈME SMS. C'est tout l'enjeu : "
                             "mémorisez chaque clé avec sa réponse (dans un dictionnaire), et si la clé est déjà connue, "
                             "renvoyez la réponse mémorisée SANS appeler `envoyer`.")
    _code(r, 201, "POST /relances rejouée avec la même clé k1 et le même corps", {
        409: "L'appelant réessaie parce qu'il n'a pas reçu la première réponse : il ne sait pas que tout s'est bien "
             "passé. Renvoyez-lui la MÊME réponse (201, même relance_id), pas une erreur.",
        422: "L'appelant réessaie parce qu'il n'a pas reçu la première réponse : il ne sait pas que tout s'est bien "
             "passé. Renvoyez-lui la MÊME réponse (201, même relance_id), pas une erreur."})
    if r.json() != premiere:
        raise AssertionError(f"Le rejeu avec la clé k1 renvoie {r.json()} au lieu de la réponse d'origine {premiere}. "
                             "Mémorisez la réponse elle-même et renvoyez-la telle quelle.")

    r = _relance(client, {"client_id": "C001", "offre": "mois_offert"}, "k1")
    if len(sms.envois) > 1:
        raise AssertionError("La clé k1, réutilisée pour une AUTRE offre, a déclenché un envoi. Une clé désigne UNE "
                             "demande : même clé et corps différent = erreur de l'appelant (422).")
    _code(r, (422, 409), "POST /relances avec la clé k1 déjà utilisée, mais pour l'offre « mois_offert »", {
        201: "Même clé, corps différent : refusez (422). Renvoyer l'ancienne réponse ferait croire à l'appelant que "
             "« mois_offert » est parti. Comparez le corps reçu au corps mémorisé avec la clé."})

    autre = {"client_id": "C003", "offre": "appel_conseiller"}
    r = _relance(client, corps, "k2")
    _code(r, 201, "POST /relances avec une NOUVELLE clé k2 (même corps que k1)", {
        409: "Une nouvelle clé, c'est une nouvelle demande, même si le corps est identique : l'appelant veut vraiment "
             "une deuxième relance. Ne dédoublonnez que sur la clé, pas sur le contenu.",
        422: "Une nouvelle clé, c'est une nouvelle demande, même si le corps est identique : l'appelant veut vraiment "
             "une deuxième relance. Ne dédoublonnez que sur la clé, pas sur le contenu."})
    seconde = _verifier_reponse_relance(r, corps, "POST /relances (clé k2)")
    if len(sms.envois) != 2:
        raise AssertionError("Une nouvelle clé (k2) doit déclencher un nouvel envoi : ne dédoublonnez que sur la clé.")
    if seconde["relance_id"] == premiere["relance_id"]:
        raise AssertionError(f"Deux relances différentes (clés k1 et k2) ont le même relance_id {premiere['relance_id']!r} : "
                             "chaque relance doit avoir un identifiant unique (un compteur, ou `uuid.uuid4().hex`).")

    sms.en_panne = True
    r = _relance(client, autre, "k3",
                 si_plantage="Entourez l'appel à `envoyer` d'un `try` / `except Exception` et répondez 503 : un 500 dit "
                             "« bug », un 503 dit « service indisponible, réessayez plus tard ».")
    _code(r, 503, "POST /relances (clé k3) pendant une panne de l'opérateur SMS", {
        201: "L'envoi a échoué (`envoyer` a levé une exception) : répondre 201 fait croire que le SMS est parti."})
    sms.en_panne = False
    tentatives = sms.tentatives
    r = _relance(client, autre, "k3")
    if sms.tentatives == tentatives:
        raise AssertionError("Après la panne, l'appelant réessaie avec la même clé k3, mais votre route ne retente pas "
                             "l'envoi : vous avez mémorisé la clé AVANT l'envoi. Ne mémorisez la "
                             "réponse qu'après un envoi RÉUSSI : sinon, ce client ne sera jamais relancé.")
    _code(r, 201, "POST /relances (clé k3) rejouée après la fin de la panne", {
        422: "Vous avez mémorisé la clé k3 malgré l'échec de l'envoi. Ne la mémorisez qu'après un envoi réussi.",
        409: "Vous avez mémorisé la clé k3 malgré l'échec de l'envoi. Ne la mémorisez qu'après un envoi réussi."})
    if sms.envois[-1:] != [("C003", "appel_conseiller")] or len(sms.envois) != 3:
        raise AssertionError(f"Après la panne, le rejeu de k3 doit envoyer le SMS (une fois). Envois : {sms.envois}.")
    r3 = r.json()
    if _relance(client, autre, "k3").json() != r3 or len(sms.envois) != 3:
        raise AssertionError("Une fois l'envoi de k3 réussi, un nouveau rejeu doit renvoyer la même réponse sans renvoyer de SMS.")

    base["C999"] = {**_BASE["C001"], "id": "C999"}
    r = _relance(client, {"client_id": "C999", "offre": "remise_10"}, "k-inconnu")
    _code(r, 201, "POST /relances (clé k-inconnu) rejouée une fois le client C999 créé", {
        404: "La première tentative avec cette clé a échoué (client inconnu) : rien n'a été fait, donc rien ne doit "
             "être mémorisé. Ne retenez une clé qu'après un envoi réussi."})
    if sms.envois[-1:] != [("C999", "remise_10")]:
        raise AssertionError("La relance de C999 (clé k-inconnu, client créé entre-temps) répond 201 sans avoir envoyé de "
                             "SMS : la première tentative avait échoué, rien ne doit en rester. Ne mémorisez une clé "
                             "qu'après un envoi réussi.")

    sms_b = _FauxEnvoi()
    client_b = _client(_fabriquer(f, _base(), sms_b))
    r = _relance(client_b, corps, "k1")
    if r.status_code == 201 and not sms_b.envois:
        raise AssertionError("Une deuxième application (autre routeur, autre base) a reconnu la clé k1 de la première et "
                             "n'a rien envoyé : la mémoire des clés est partagée. Créez le dictionnaire DANS "
                             "`routes_relances`, pas au niveau du module.")
    _code(r, 201, "POST /relances (clé k1) sur une DEUXIÈME application, indépendante de la première", {
        422: "La mémoire des clés est partagée entre deux routeurs : créez le dictionnaire DANS `routes_relances`, "
             "pas au niveau du module."})


SERIE = Serie(
    code="12",
    titre="Construire une API web : lire, chercher, créer, modifier, protéger, rejouer sans risque",
    module_eleve="defis.serie_12",
    module_corrige="defis.corrige_serie_12",
    defis=(
        Defi(74, "routes_lecture", "🟢", "6.4", verifier_routes_lecture),
        Defi(75, "routes_recherche", "🟢", "6.4", verifier_routes_recherche),
        Defi(76, "routes_creation", "🟡", "6.4", verifier_routes_creation),
        Defi(77, "routes_modification", "🟡", "—", verifier_routes_modification),
        Defi(78, "routes_admin", "🟡", "—", verifier_routes_admin),
        Defi(79, "routes_relances", "🔴", "—", verifier_routes_relances),
    ),
)

"""Vérifications automatiques de la série 16 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Le correcteur exécute vos coroutines sur une boucle asyncio à HORLOGE VIRTUELLE (defis/concurrence.py) :
quand toutes les tâches attendent, l'horloge saute au prochain réveil. Il mesure ainsi, exactement et en
une fraction de seconde, combien de temps votre code aurait pris, et combien d'appels il a faits EN MÊME
TEMPS. Les faux services notent tout : appels, annulations, refus.

Il regarde aussi ce qui reste APRÈS votre retour : une tâche oubliée qui tourne encore, un appel lent
jamais annulé. Et si tout s'arrête (toutes les tâches attendent, rien ne les réveillera), il le voit :
c'est un interblocage, qu'un vrai programme subirait en silence.

Pendant la correction, `time.sleep` est remplacé par une fonction qui signale l'erreur : il bloquerait
toute la boucle.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import heapq
import inspect
import time
import traceback
import warnings
from dataclasses import dataclass

from .concurrence import (
    SCORES_EXEMPLE,
    Interblocage,
    NumeroInvalide,
    OperateurSMS,
    Pages,
    ServeurAuth,
    ServiceIndisponible,
    ServiceScores,
    Surcharge,
    TempsDepasse,
    executer,
)
from .verif import Defi, Serie

EPS = 1e-6
TEXTE = "Votre offre fidélité vous attend : -20 % pendant 6 mois. STOP au 36180"

_INTERBLOCAGE = ("Cherchez un `await` qui ne peut plus aboutir : `file.get()` sur une file que plus personne ne "
                 "remplit, `file.join()` alors qu'un travailleur est mort sur une erreur, un verrou jamais rendu "
                 "(préférez `async with verrou:` à acquire() / release()).")


def _court(valeur, n: int = 160) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


def _s(duree: float) -> str:
    """Une durée en secondes, à la française : 0,2 s ; 1,6 s."""
    return f"{duree:.3f}".rstrip("0").rstrip(".").replace(".", ",") + " s"


# ============================================================================ Exécuter votre code


class _AppelBloquant(BaseException):
    """Levée à la place de time.sleep pendant la correction. BaseException pour traverser un `except Exception`."""


@contextlib.contextmanager
def _sans_time_sleep():
    vrai = time.sleep

    def bloquant(secondes=0, *_args):
        raise _AppelBloquant(secondes)

    time.sleep = bloquant
    try:
        yield
    finally:
        time.sleep = vrai


class _PasUneCoroutine(Exception):
    """Une méthode qui devait être `async def` a renvoyé directement une valeur (défi 101)."""

    def __init__(self, valeur) -> None:
        super().__init__(_court(valeur))
        self.valeur = valeur


@dataclass
class _Appel:
    resultat: object = None
    erreur: Exception | None = None
    duree: float = 0.0
    restantes: int = 0       # tâches encore en vie juste après votre retour
    observe: object = None   # ce que `observer()` a relevé juste après votre retour (avant que la boucle annule tout)
    probleme: str = ""       # "", "pas_async", "bloquant", "interblocage", "temps"
    detail: str = ""


def _lancer(appeler, *, observer=None, limite: float = 86_400.0) -> _Appel:
    """Exécute `appeler()` (qui doit renvoyer une coroutine) sur une boucle virtuelle neuve, et note tout.

    `observer()` est appelé juste après votre retour, AVANT que la boucle n'annule les tâches oubliées : c'est le
    moment de regarder si un faux service travaille encore.
    """
    appel = _Appel()

    async def principal():
        boucle = asyncio.get_running_loop()
        debut = boucle.time()
        try:
            objet = appeler()
            if not inspect.isawaitable(objet):
                appel.probleme, appel.resultat = "pas_async", objet
                return
            appel.resultat = await objet
        except Exception as exc:
            appel.erreur = exc
        appel.duree = boucle.time() - debut
        moi = asyncio.current_task()
        for _ in range(3):  # laisse aux annulations demandées (tache.cancel()) le temps d'arriver
            await asyncio.sleep(0)
        appel.restantes = sum(1 for t in asyncio.all_tasks() if t is not moi and not t.done())
        if observer is not None:
            appel.observe = observer()

    # Une coroutine jamais attendue (votre code l'a fabriquée sans `await`) déclencherait plus tard un avertissement
    # sans rapport avec le défi en cours : le message du correcteur l'explique déjà.
    with _sans_time_sleep(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="coroutine .* was never awaited", category=RuntimeWarning)
        try:
            executer(principal(), limite=limite, silencieux=True)
        except Interblocage:
            appel.probleme = "interblocage"
        except TempsDepasse as exc:
            appel.probleme, appel.detail = "temps", str(exc)
        except _AppelBloquant as exc:
            appel.probleme, appel.detail = "bloquant", str(exc)
        if appel.erreur is not None and appel.erreur.__traceback__ is not None:
            traceback.clear_frames(appel.erreur.__traceback__)  # libère tout de suite les coroutines de vos variables locales
    return appel


def _fermer_coroutines(valeur, profondeur: int = 0) -> object | None:
    """Renvoie la première coroutine (ou tâche) trouvée dans le résultat, et ferme les coroutines trouvées."""
    if inspect.isawaitable(valeur):
        if inspect.iscoroutine(valeur):
            valeur.close()
        return valeur
    if profondeur > 2:
        return None
    if isinstance(valeur, dict):
        elements = list(valeur.values())
    elif isinstance(valeur, (list, tuple, set)):
        elements = list(valeur)
    else:
        return None
    trouve = None
    for element in elements:
        objet = _fermer_coroutines(element, profondeur + 1)
        trouve = trouve or objet
    return trouve


def _verifier_appel(appel: _Appel, quand: str, interblocage: str | None = None) -> None:
    """Les problèmes communs à tous les défis : pas `async`, time.sleep, interblocage, asyncio.run…"""
    if isinstance(appel.erreur, NotImplementedError):
        raise appel.erreur  # défi pas encore commencé
    if appel.probleme == "pas_async":
        _fermer_coroutines(appel.resultat)
        raise AssertionError(
            f"{quand} : votre fonction a renvoyé {_court(appel.resultat)} au lieu d'une coroutine. Déclarez-la avec "
            "`async def` : c'est le correcteur qui l'exécute, dans sa boucle.")
    if appel.probleme == "bloquant":
        raise AssertionError(
            f"{quand} : votre code appelle time.sleep(). Il BLOQUE toute la boucle : pendant ce temps, aucune autre tâche "
            "n'avance (ni les autres appels, ni le serveur web). Dans une coroutine : `await asyncio.sleep(…)`.")
    if appel.probleme == "interblocage":
        raise AssertionError(
            f"{quand} : votre fonction ne se termine JAMAIS. Toutes les tâches attendent, et rien n'est prévu pour les "
            f"réveiller (interblocage). {interblocage or _INTERBLOCAGE}")
    if appel.probleme == "temps":
        raise AssertionError(f"{quand} : votre fonction ne s'arrête pas ({appel.detail}).")
    erreur = appel.erreur
    if isinstance(erreur, _PasUneCoroutine):
        raise AssertionError(
            f"{quand} : `jeton()` a renvoyé {erreur} directement, au lieu d'une coroutine. Déclarez la méthode avec "
            "`async def jeton(self)` : elle doit pouvoir attendre le serveur.")
    if isinstance(erreur, RuntimeError):
        texte = str(erreur)
        if "running event loop" in texte:
            raise AssertionError(
                f"{quand} : n'appelez pas asyncio.run() dans votre fonction : elle tourne DÉJÀ dans une boucle (celle du "
                "correcteur ici, celle de FastAPI en vrai). Écrivez `await …` à la place.")
        if "different event loop" in texte or "different loop" in texte:
            raise AssertionError(
                f"{quand} : RuntimeError: {texte}. Un objet asyncio (Lock, Semaphore, Queue…) créé une seule fois, au niveau "
                "du module ou de la CLASSE, reste lié à la première boucle qui s'en est servie, et il est partagé par "
                "tous. Créez-le dans la fonction (ou dans `__init__`, un par objet).")
    if isinstance(erreur, TypeError) and "coroutines is forbidden" in str(erreur):
        raise AssertionError(
            f"{quand} : asyncio.wait() n'accepte plus de coroutines depuis Python 3.11 : donnez-lui des TÂCHES, créées "
            "avec `asyncio.create_task(…)`.")


def _decrire_erreur(erreur: BaseException) -> str:
    if isinstance(erreur, BaseExceptionGroup):
        return f"{type(erreur).__name__}({', '.join(_decrire_erreur(e) for e in erreur.exceptions)})"
    return f"{type(erreur).__name__}: {erreur}"


def _sans_erreur(appel: _Appel, quand: str) -> None:
    if appel.erreur is not None:
        raise AssertionError(f"{quand} : votre fonction lève {_decrire_erreur(appel.erreur)}.")


def _sans_coroutine(valeur, quand: str) -> None:
    objet = _fermer_coroutines(valeur)
    if objet is not None:
        raise AssertionError(
            f"{quand} : votre résultat contient {_court(objet, 80)}. Il manque un `await` : appeler une coroutine ne fait "
            "que la fabriquer, c'est `await` (ou gather) qui l'exécute et donne son résultat.")


def _contient(erreur: BaseException | None, attendue) -> bool:
    """`attendue` est une classe d'exception ou une exception précise ; on cherche aussi dans les ExceptionGroup."""
    if erreur is None:
        return False
    if isinstance(erreur, BaseExceptionGroup):
        return any(_contient(e, attendue) for e in erreur.exceptions)
    if isinstance(attendue, type):
        return isinstance(erreur, attendue)
    return erreur is attendue


def _comparer_scores(resultat, attendu: dict, quand: str) -> None:
    if resultat is None:
        raise AssertionError(f"{quand} : votre fonction renvoie None. Oubli du `return` ?")
    if not isinstance(resultat, dict):
        raise AssertionError(
            f"{quand} : renvoyez un dictionnaire {{client: score}}, pas {type(resultat).__name__} ({_court(resultat)}).")
    manquants = [c for c in attendu if c not in resultat]
    en_trop = [c for c in resultat if c not in attendu]
    if manquants or en_trop:
        raise AssertionError(
            f"{quand} : il manque {len(manquants)} client(s) {_court(manquants[:5], 60)} et il y en a {len(en_trop)} en "
            f"trop {_court(en_trop[:5], 60)}.")
    faux = {c: resultat[c] for c in attendu if resultat[c] != attendu[c]}
    if faux:
        client = next(iter(faux))
        raise AssertionError(
            f"{quand} : le score de {client} vaut {_court(faux[client], 60)} au lieu de {attendu[client]!r} (le service "
            "renvoie le bon : gardez-le tel quel).")


# ============================================================================ Défi 98 : gather


def verifier_scores_en_parallele(f) -> None:
    ids = list(SCORES_EXEMPLE)
    service = ServiceScores(latence=0.2)
    quand = "Avec les 8 clients de SCORES_EXEMPLE (0,2 s par réponse)"
    appel = _lancer(lambda: f(service, list(ids)))
    _verifier_appel(appel, quand)
    _sans_erreur(appel, quand)
    _sans_coroutine(appel.resultat, quand)
    _comparer_scores(appel.resultat, SCORES_EXEMPLE, quand)
    if list(appel.resultat) != ids:
        raise AssertionError(
            f"{quand} : les clients doivent être dans l'ordre de la liste reçue {ids}, pas {list(appel.resultat)}. "
            "gather renvoie les résultats dans l'ordre des coroutines qu'on lui donne (as_completed, dans l'ordre "
            "d'arrivée).")
    if appel.duree > 0.2 + EPS:
        if appel.duree >= 1.6 - EPS:
            raise AssertionError(
                f"{quand}, votre fonction met {_s(appel.duree)} : les 8 appels sont faits l'un APRÈS l'autre "
                f"(service.max_en_cours = {service.max_en_cours}). Chaque `await` dans la boucle attend la réponse avant "
                "de lancer l'appel suivant. Lancez-les tous, puis attendez-les ensemble : asyncio.gather.")
        raise AssertionError(
            f"{quand}, votre fonction met {_s(appel.duree)} au lieu de 0,2 s : au plus {service.max_en_cours} appels "
            "sont partis en même temps. Les 8 doivent partir ensemble.")

    service = ServiceScores(latence=0.2, latences={"C001": 0.5, "C003": 0.1, "C005": 0.3})
    ids = ["C003", "C001", "C003", "C005"]
    quand = f"Avec client_ids = {ids} (C001 répond en 0,5 s)"
    appel = _lancer(lambda: f(service, list(ids)))
    _verifier_appel(appel, quand)
    _sans_erreur(appel, quand)
    _sans_coroutine(appel.resultat, quand)
    demandes = collections.Counter(client for client, _ in service.appels)
    doublons = [client for client, n in demandes.items() if n > 1]
    if doublons:
        raise AssertionError(
            f"{quand} : {doublons[0]} a été demandé {demandes[doublons[0]]} fois au service. Un client en double n'est "
            "demandé qu'une fois : `dict.fromkeys(client_ids)` retire les doublons en gardant l'ordre.")
    _comparer_scores(appel.resultat, {"C003": 0.47, "C001": 0.82, "C005": 0.0}, quand)
    if list(appel.resultat) != ["C003", "C001", "C005"]:
        raise AssertionError(f"{quand} : l'ordre attendu est ['C003', 'C001', 'C005'], pas {list(appel.resultat)}.")
    if appel.duree > 0.5 + EPS:
        raise AssertionError(
            f"{quand}, votre fonction met {_s(appel.duree)} au lieu de 0,5 s (la réponse la plus lente) : les appels ne "
            "partent pas tous ensemble.")

    service = ServiceScores(latence=0.2)
    quand = "Avec une liste vide"
    appel = _lancer(lambda: f(service, []))
    _verifier_appel(appel, quand)
    _sans_erreur(appel, quand)
    if appel.resultat != {} or not isinstance(appel.resultat, dict):
        raise AssertionError(f"{quand} : renvoyez {{}}, pas {_court(appel.resultat)}.")
    if service.appels:
        raise AssertionError(f"{quand} : aucun appel au service n'est nécessaire.")

    service = ServiceScores(latence=0.2)
    quand = "Avec ['C001', 'C999', 'C002'] (C999 n'existe pas)"
    appel = _lancer(lambda: f(service, ["C001", "C999", "C002"]))
    _verifier_appel(appel, quand)
    if appel.erreur is None:
        _fermer_coroutines(appel.resultat)
        raise AssertionError(
            f"{quand} : le service lève KeyError pour C999, et votre fonction renvoie {_court(appel.resultat)}. Laissez "
            "l'erreur remonter : un score absent ou inventé sur la page serait pire qu'un message d'erreur.")
    if not _contient(appel.erreur, KeyError):
        raise AssertionError(
            f"{quand} : votre fonction lève {_decrire_erreur(appel.erreur)} au lieu de laisser remonter la KeyError du "
            "service.")


# ============================================================================ Défi 99 : délai maximal


def verifier_score_ou_defaut(f) -> None:
    service = ServiceScores(latence=0.2)
    quand = "Avec une réponse en 0,2 s et delai=1.0"
    appel = _lancer(lambda: f(service, "C001", 1.0))
    _verifier_appel(appel, quand)
    if isinstance(appel.erreur, TypeError) and "defaut" in str(appel.erreur):
        raise AssertionError(f"{quand} : {appel.erreur}. Le paramètre `defaut` est facultatif (None par défaut).")
    _sans_erreur(appel, quand)
    _sans_coroutine(appel.resultat, quand)
    if appel.resultat != 0.82:
        raise AssertionError(
            f"{quand} : la réponse arrive à temps (0,2 s < 1 s), renvoyez-la (0.82), pas {_court(appel.resultat)}.")
    if appel.duree > 0.2 + EPS:
        if appel.duree >= 1.0 - EPS:
            raise AssertionError(
                f"{quand}, votre fonction met {_s(appel.duree)} : elle attend toujours la fin du délai. Renvoyez la "
                "réponse DÈS qu'elle arrive (asyncio.wait_for n'attend pas plus que nécessaire).")
        raise AssertionError(f"{quand}, votre fonction met {_s(appel.duree)} au lieu de 0,2 s.")

    lent = ServiceScores(latence=30.0)
    quand = "Avec une réponse en 30 s et delai=1.0 (defaut non précisé)"
    appel = _lancer(lambda: f(lent, "C001", 1.0), observer=lambda: lent.en_cours)
    _verifier_appel(appel, quand)
    if isinstance(appel.erreur, TimeoutError):
        raise AssertionError(
            f"{quand} : votre fonction laisse remonter TimeoutError. C'est justement le cas prévu : attrapez-la "
            "(`except TimeoutError:`) et renvoyez `defaut`.")
    _sans_erreur(appel, quand)
    _sans_coroutine(appel.resultat, quand)
    if appel.duree >= 30 - EPS:
        raise AssertionError(
            f"{quand}, votre fonction a attendu la réponse lente ({_s(appel.duree)}) : le délai n'est pas appliqué. "
            "`await asyncio.wait_for(service.score(client_id), timeout=delai)`.")
    if appel.resultat is not None:
        raise AssertionError(f"{quand} : renvoyez `defaut` (None par défaut), pas {_court(appel.resultat)}.")
    if abs(appel.duree - 1.0) > EPS:
        raise AssertionError(f"{quand}, votre fonction rend la main au bout de {_s(appel.duree)}, au lieu de delai = 1 s.")
    if appel.observe or appel.restantes:
        raise AssertionError(
            f"{quand} : vous avez bien renvoyé `defaut` au bout d'1 s… mais l'appel lent tourne ENCORE en arrière-plan "
            "(il occupe le service pour une réponse que personne n'attend plus). Annulez-le : asyncio.wait_for et "
            "asyncio.timeout le font ; avec asyncio.wait(…, timeout=…), il faut appeler `tache.cancel()` vous-même.")

    lent = ServiceScores(latence=2.0)
    quand = "Avec une réponse en 2 s, delai=0.5 et defaut=0.5"
    appel = _lancer(lambda: f(lent, "C004", 0.5, defaut=0.5))
    _verifier_appel(appel, quand)
    _sans_erreur(appel, quand)
    if appel.resultat != 0.5:
        raise AssertionError(f"{quand} : renvoyez le paramètre `defaut` (0.5), pas {_court(appel.resultat)}.")

    service = ServiceScores(latence=0.1)
    quand = "Pour C005 (score 0.0, réponse en 0,1 s), delai=1.0 et defaut=0.5"
    appel = _lancer(lambda: f(service, "C005", 1.0, defaut=0.5))
    _verifier_appel(appel, quand)
    _sans_erreur(appel, quand)
    if appel.resultat == 0.5:
        raise AssertionError(
            f"{quand} : vous renvoyez defaut. Mais 0.0 est un VRAI score, arrivé à temps. Seulement, 0.0 est « faux » en "
            "Python : `score or defaut` et `if score:` le remplacent. Ne testez pas le score : seul le délai dépassé "
            "donne defaut.")
    if appel.resultat != 0.0 or appel.resultat is None:
        raise AssertionError(f"{quand} : renvoyez 0.0, pas {_court(appel.resultat)}.")

    service = ServiceScores(latence=0.1)
    quand = "Pour C999 (client inconnu : le service lève KeyError en 0,1 s), delai=1.0 et defaut=0.5"
    appel = _lancer(lambda: f(service, "C999", 1.0, defaut=0.5))
    _verifier_appel(appel, quand)
    if appel.erreur is None:
        raise AssertionError(
            f"{quand} : votre fonction renvoie {_court(appel.resultat)}. SEUL le dépassement du délai donne defaut : une "
            "erreur n'est pas une lenteur. Attrapez TimeoutError, pas Exception.")
    if not _contient(appel.erreur, KeyError):
        raise AssertionError(f"{quand} : laissez remonter la KeyError du service, pas {_decrire_erreur(appel.erreur)}.")

    panne = ServiceIndisponible("service des scores indisponible (503)")
    service = ServiceScores(latence=0.3, erreurs={"C002": panne})
    quand = "Quand le service répond par une erreur (ServiceIndisponible en 0,3 s), delai=1.0 et defaut=0.5"
    appel = _lancer(lambda: f(service, "C002", 1.0, defaut=0.5))
    _verifier_appel(appel, quand)
    if appel.erreur is None:
        raise AssertionError(
            f"{quand} : votre fonction renvoie {_court(appel.resultat)}. Le service est en PANNE, il n'est pas lent : "
            "l'erreur doit remonter (on affichera « service indisponible », et l'équipe sera prévenue).")
    if not _contient(appel.erreur, panne):
        raise AssertionError(f"{quand} : laissez remonter l'erreur du service, pas {_decrire_erreur(appel.erreur)}.")

    for delai in (0, -1.0):
        service = ServiceScores(latence=0.1)
        quand = f"Avec delai={delai!r}"
        appel = _lancer(lambda: f(service, "C001", delai))
        _verifier_appel(appel, quand)
        if not isinstance(appel.erreur, ValueError):
            obtenu = _decrire_erreur(appel.erreur) if appel.erreur else f"un résultat ({_court(appel.resultat)})"
            raise AssertionError(f"{quand} : levez ValueError (un délai doit être strictement positif), pas {obtenu}.")


# ============================================================================ Défi 100 : Semaphore


def _au_plus_tot(latences: list[float], places: int) -> float:
    """Durée d'une campagne où chaque envoi part dès qu'une place se libère, dans l'ordre de la liste."""
    fins: list[float] = []
    for latence in latences:
        debut = heapq.heappop(fins) if len(fins) >= places else 0.0
        heapq.heappush(fins, debut + latence)
    return max(fins, default=0.0)


def _par_paquets(latences: list[float], places: int) -> float:
    return sum(max(latences[i:i + places]) for i in range(0, len(latences), places))


def _lancer_campagne(f, operateur, clients, quand, *args, **kwargs):
    appel = _lancer(lambda: f(operateur, list(clients), TEXTE, *args, **kwargs))
    _verifier_appel(appel, quand)
    if appel.erreur is not None:
        if _contient(appel.erreur, NumeroInvalide):
            raise AssertionError(
                f"{quand} : un numéro invalide a arrêté TOUTE la campagne (votre fonction lève "
                f"{_decrire_erreur(appel.erreur)}). gather(…) s'arrête à la première erreur, et laisse les autres envois "
                "tourner sans personne pour les attendre. Utilisez `return_exceptions=True`, ou un try/except dans la "
                "coroutine qui envoie UN SMS.")
        if _contient(appel.erreur, Surcharge):
            raise AssertionError(
                f"{quand} : l'opérateur a refusé des SMS (Surcharge) : plus de {operateur.max_simultanes} envois en même "
                "temps. Limitez-les avec `asyncio.Semaphore(max_simultanes)` et `async with`.")
        _sans_erreur(appel, quand)
    _sans_coroutine(appel.resultat, quand)
    resultat = appel.resultat
    if not (isinstance(resultat, (tuple, list)) and len(resultat) == 2 and all(isinstance(d, dict) for d in resultat)):
        raise AssertionError(
            f"{quand} : renvoyez un couple (envoyes, echecs) de deux dictionnaires, pas {_court(resultat)}.")
    if operateur.refus:
        raise AssertionError(
            f"{quand} : l'opérateur a refusé {len(operateur.refus)} SMS (Surcharge) : vous avez lancé plus de "
            f"{operateur.max_simultanes} envois en même temps. Limitez-les avec `asyncio.Semaphore(max_simultanes)` et "
            "`async with` autour de l'envoi.")
    recus = collections.Counter(client for client, _, _ in operateur.envoyes)
    doublons = [client for client, n in recus.items() if n > 1]
    if doublons:
        raise AssertionError(
            f"{quand} : {doublons[0]} a reçu {recus[doublons[0]]} SMS. Un client présent deux fois dans la liste n'en "
            "reçoit qu'un : retirez les doublons d'abord (dict.fromkeys).")
    if any(texte != TEXTE for _, texte, _ in operateur.envoyes):
        raise AssertionError(f"{quand} : chaque SMS doit contenir `texte`, tel quel.")
    return appel


def verifier_envoyer_campagne(f) -> None:
    clients = [f"C{n:03d}" for n in range(1, 31)]
    latences = {c: 0.2 + (0.8 if n % 7 == 3 else 0.0) + (0.3 if n % 5 == 0 else 0.0) for n, c in enumerate(clients, 1)}
    invalides = {"C007", "C018"}
    operateur = OperateurSMS(max_simultanes=10, latence=0.2, latences=latences, invalides=invalides)
    quand = "Avec 30 clients, 2 numéros invalides (C007, C018) et l'opérateur limité à 10 envois simultanés"
    appel = _lancer_campagne(f, operateur, clients, quand)  # max_simultanes par défaut : 10
    envoyes, echecs = appel.resultat
    identifiants = {client: identifiant for client, _, identifiant in operateur.envoyes}
    attendus = [c for c in clients if c not in invalides]
    manquants = [c for c in attendus if c not in envoyes]
    if manquants or set(envoyes) - set(attendus):
        raise AssertionError(
            f"{quand} : `envoyes` doit contenir les 28 clients qui ont reçu leur SMS. Il manque "
            f"{_court(manquants[:5], 60)}, et il y a en trop {_court(sorted(set(envoyes) - set(attendus))[:5], 60)}.")
    faux = [c for c in attendus if envoyes[c] != identifiants.get(c)]
    if faux:
        raise AssertionError(
            f"{quand} : envoyes[{faux[0]!r}] vaut {_court(envoyes[faux[0]], 60)} au lieu de l'identifiant renvoyé par "
            f"operateur.envoyer ({identifiants.get(faux[0])!r}).")
    if list(envoyes) != attendus:
        raise AssertionError(
            f"{quand} : les clients de `envoyes` doivent être dans l'ordre de la liste `clients` (gather garde cet "
            f"ordre), pas {list(envoyes)[:6]}…")
    if set(echecs) != invalides:
        raise AssertionError(
            f"{quand} : `echecs` doit contenir exactement C007 et C018 (numéros invalides), pas {sorted(echecs)}.")
    for client in sorted(invalides):
        valeur = echecs[client]
        if not isinstance(valeur, NumeroInvalide):
            raise AssertionError(
                f"{quand} : echecs[{client!r}] vaut {_court(valeur, 80)}. Gardez l'EXCEPTION elle-même (le NumeroInvalide "
                "levé par l'opérateur), pas son texte : demain, on relancera seulement les erreurs passagères.")
        if not any(valeur is erreur for erreur in operateur.erreurs):
            raise AssertionError(f"{quand} : echecs[{client!r}] doit être l'exception levée par l'opérateur, pas une copie.")
    duree_par_client = [latences[c] for c in clients]
    au_plus_tot = _au_plus_tot(duree_par_client, 10)
    if appel.duree > au_plus_tot + EPS:
        if appel.duree >= sum(duree_par_client) - EPS:
            raise AssertionError(
                f"{quand}, la campagne met {_s(appel.duree)} : les SMS partent un par un (operateur.max_en_cours = "
                f"{operateur.max_en_cours}). Lancez-les ensemble, en limitant à max_simultanes : Semaphore + gather.")
        if appel.duree >= _par_paquets(duree_par_client, 10) - EPS:
            raise AssertionError(
                f"{quand}, la campagne met {_s(appel.duree)} au lieu de {_s(au_plus_tot)} : vous envoyez par PAQUETS de "
                "10, et chaque paquet attend son SMS le plus lent pendant que les autres places restent vides. Avec un "
                "asyncio.Semaphore, dès qu'un envoi finit, le suivant part.")
        raise AssertionError(
            f"{quand}, la campagne met {_s(appel.duree)} au lieu de {_s(au_plus_tot)} : dès qu'un envoi finit, le suivant "
            "doit partir (au plus max_simultanes en même temps, et le plus possible).")

    operateur = OperateurSMS(max_simultanes=10, latence=0.2)
    clients = [f"C{n:03d}" for n in range(1, 13)]
    quand = "Avec 12 clients et max_simultanes=3 (l'opérateur en accepterait 10)"
    appel = _lancer_campagne(f, operateur, clients, quand, max_simultanes=3)
    if operateur.max_en_cours > 3:
        raise AssertionError(
            f"{quand} : jusqu'à {operateur.max_en_cours} envois en même temps. Respectez le paramètre max_simultanes "
            "(un Semaphore créé une fois pour toutes, hors de la fonction, l'ignore).")
    if len(appel.resultat[0]) != 12:
        raise AssertionError(f"{quand} : les 12 clients doivent recevoir leur SMS ({len(appel.resultat[0])} envoyés).")
    if appel.duree > _au_plus_tot([0.2] * 12, 3) + EPS:
        raise AssertionError(
            f"{quand}, la campagne met {_s(appel.duree)} au lieu de 0,8 s (4 vagues de 3) : utilisez les 3 places.")

    operateur = OperateurSMS(max_simultanes=10, latence=0.2)
    clients = ["C001", "C002", "C001", "C003", "C002"]
    quand = f"Avec clients = {clients}"
    appel = _lancer_campagne(f, operateur, clients, quand)
    if list(appel.resultat[0]) != ["C001", "C002", "C003"]:
        raise AssertionError(f"{quand} : `envoyes` doit valoir les clients ['C001', 'C002', 'C003'], pas "
                             f"{list(appel.resultat[0])}.")

    operateur = OperateurSMS(max_simultanes=10, latence=0.2)
    quand = "Avec une liste de clients vide"
    appel = _lancer_campagne(f, operateur, [], quand)
    if list(appel.resultat) != [{}, {}] or operateur.tentatives:
        raise AssertionError(f"{quand} : renvoyez ({{}}, {{}}) sans contacter l'opérateur, pas {_court(appel.resultat)}.")

    for maximum in (0, -2):
        operateur = OperateurSMS(max_simultanes=10, latence=0.2)
        quand = f"Avec max_simultanes={maximum}"
        appel = _lancer(lambda: f(operateur, ["C001"], TEXTE, max_simultanes=maximum))
        _verifier_appel(appel, quand, interblocage="Avec 0 place, aucun envoi ne peut partir : levez ValueError dès le début.")
        if not isinstance(appel.erreur, ValueError):
            obtenu = _decrire_erreur(appel.erreur) if appel.erreur else f"un résultat ({_court(appel.resultat)})"
            raise AssertionError(f"{quand} : levez ValueError (il faut au moins une place), pas {obtenu}.")


# ============================================================================ Défi 101 : Lock


async def _jeton(cache):
    objet = cache.jeton()
    if not inspect.isawaitable(objet):
        raise _PasUneCoroutine(objet)
    return await objet


def _verifier_jeton(valeur, quand: str) -> None:
    if isinstance(valeur, tuple):
        raise AssertionError(
            f"{quand} : jeton() renvoie {_court(valeur)}. nouveau_jeton() renvoie un couple (jeton, durée de vie) : "
            "gardez la durée pour vous, et ne renvoyez que le jeton (« jeton-1 »).")
    _sans_coroutine(valeur, quand)


def verifier_CacheJeton(Classe) -> None:
    _VERROU_RENDU = ("Après une panne du serveur, le verrou n'a jamais été rendu : `async with self._verrou:` le rend "
                     "même quand une erreur sort du bloc (acquire() puis release() sans try/finally, non).")

    # 1. Cinquante tâches pressées
    serveur = ServeurAuth(latence=0.5)

    async def cinquante():
        cache = Classe(serveur)
        return await asyncio.gather(*(_jeton(cache) for _ in range(50)))

    quand = "Quand 50 tâches appellent jeton() en même temps (le serveur répond en 0,5 s)"
    appel = _lancer(cinquante)
    _verifier_appel(appel, quand)
    _sans_erreur(appel, quand)
    jetons = appel.resultat
    for valeur in jetons:
        _verifier_jeton(valeur, quand)
    demandes = len(serveur.demandes)
    if demandes > 1:
        if appel.duree <= 0.5 + EPS:
            raise AssertionError(
                f"{quand}, le serveur a reçu {demandes} demandes au lieu d'une. Entre « pas de jeton valable » et la "
                "réponse du serveur, il y a un `await` : pendant ce temps, les autres tâches passent le même test, et "
                "demandent chacune leur jeton. Protégez la demande avec un verrou : `self._verrou = asyncio.Lock()` "
                "dans __init__, `async with self._verrou:` dans jeton().")
        raise AssertionError(
            f"{quand}, le serveur a reçu {demandes} demandes, l'une APRÈS l'autre ({_s(appel.duree)}). Le verrou est là, "
            "mais chaque tâche qui l'obtient redemande un jeton sans regarder si la précédente vient d'en obtenir un. "
            "Revérifiez la validité du jeton APRÈS avoir obtenu le verrou (double vérification).")
    if demandes == 0:
        raise AssertionError(f"{quand} : le serveur n'a reçu aucune demande. D'où vient le jeton {_court(jetons[0])} ?")
    if None in jetons and "jeton-1" in jetons:
        raise AssertionError(
            f"{quand} : {jetons.count(None)} appels ont reçu None. Pendant que la première tâche attendait le serveur, "
            "les autres ont trouvé un jeton « valable » qui n'était pas encore arrivé : vous notez l'heure d'expiration "
            "AVANT d'avoir la réponse. Ne la notez qu'une fois le jeton reçu, en même temps que lui.")
    if set(jetons) != {"jeton-1"}:
        raise AssertionError(
            f"{quand} : les 50 appels doivent recevoir le jeton du serveur ('jeton-1'), pas "
            f"{_court(dict(collections.Counter(map(repr, jetons))))}.")
    if appel.duree > 0.5 + EPS:
        raise AssertionError(f"{quand}, les 50 appels mettent {_s(appel.duree)} au lieu de 0,5 s.")

    # 2. Garder le jeton, puis le renouveler à temps
    serveur = ServeurAuth(latence=0.5, duree_de_vie=3600.0)

    async def une_heure():
        boucle = asyncio.get_running_loop()
        cache = Classe(serveur)
        constat = {"premier": await _jeton(cache)}
        for nom, instant in (("t1000", 1000.0), ("t3530", 3530.0), ("t3550", 3550.0), ("t3700", 3700.0)):
            await asyncio.sleep(instant - boucle.time())
            avant, debut = len(serveur.demandes), boucle.time()
            constat[nom] = (await _jeton(cache), len(serveur.demandes) - avant, boucle.time() - debut)
            if nom == "t3550" and constat[nom][1]:
                break
        return constat

    quand = "Avec un jeton valable 1 h (3600 s) et la marge par défaut (60 s)"
    appel = _lancer(une_heure)
    _verifier_appel(appel, quand, interblocage=_INTERBLOCAGE)
    _sans_erreur(appel, quand)
    constat = appel.resultat
    _verifier_jeton(constat.pop("premier"), quand)
    for valeur, _, _ in constat.values():
        _verifier_jeton(valeur, quand)
    if constat["t1000"][1]:
        raise AssertionError(
            f"{quand} : à t = 1000 s, le jeton est encore bon pour 2600 s, et vous en redemandez un. Gardez-le (dans "
            "l'objet : self._jeton) avec son heure d'expiration, et renvoyez-le tant qu'il est valable.")
    if constat["t1000"][0] != "jeton-1":
        raise AssertionError(f"{quand} : à t = 1000 s, renvoyez le jeton gardé ('jeton-1'), pas {_court(constat['t1000'][0])}.")
    if constat["t3530"][1]:
        raise AssertionError(
            f"{quand} : à t = 3530 s, le jeton expire dans 70 s, plus que la marge (60 s) : il est encore bon, ne le "
            "renouvelez pas encore. Renouvelez quand `maintenant >= expiration - marge`.")
    if not constat["t3550"][1]:
        if "t3700" in constat and not constat["t3700"][1]:
            raise AssertionError(
                f"{quand} : à t = 3700 s, le jeton a expiré depuis 100 s et vous renvoyez encore "
                f"{_court(constat['t3700'][0])}. Gardez l'heure d'expiration (heure de la demande + duree_de_vie) et "
                "comparez-la à `asyncio.get_running_loop().time()`. (time.time() et time.monotonic() n'avancent pas sur "
                "l'horloge virtuelle du correcteur ; en vrai, loop.time() EST time.monotonic().)")
        raise AssertionError(
            f"{quand} : à t = 3550 s, le jeton expire dans 50 s, moins que la marge (60 s) : renouvelez-le MAINTENANT. "
            "Sinon, une requête partie à 3599 s arriverait chez l'opérateur avec un jeton mort.")
    if constat["t3550"][0] != "jeton-2":
        raise AssertionError(f"{quand} : à t = 3550 s, renvoyez le nouveau jeton ('jeton-2'), pas {_court(constat['t3550'][0])}.")

    # 3. Une panne du serveur
    serveur = ServeurAuth(latence=0.5, pannes=1)

    async def panne():
        cache = Classe(serveur)
        try:
            premier = ("resultat", await _jeton(cache))
        except ServiceIndisponible as exc:
            premier = ("erreur", exc)
        try:
            ensuite = ("resultat", await _jeton(cache))
        except ServiceIndisponible as exc:
            ensuite = ("erreur", exc)
        return premier, ensuite

    quand = "Quand la première demande au serveur échoue (ServiceIndisponible)"
    appel = _lancer(panne)
    _verifier_appel(appel, quand, interblocage=_VERROU_RENDU)
    _sans_erreur(appel, quand)
    (sorte, valeur), (sorte_ensuite, ensuite) = appel.resultat
    if sorte == "resultat":
        raise AssertionError(
            f"{quand} : jeton() renvoie {_court(valeur)}. Laissez l'erreur remonter : sans jeton, l'appel à l'opérateur "
            "échouerait de toute façon, plus loin et moins clairement.")
    if sorte_ensuite == "erreur":
        raise AssertionError(
            f"{quand} : le serveur est revenu, mais l'appel suivant lève encore {_decrire_erreur(ensuite)} "
            f"({len(serveur.demandes)} demande(s) au serveur). Vous avez gardé l'échec en cache (une tâche ou un Future "
            "qui a échoué ?). Après une panne, rien ne doit rester en cache : l'appel suivant redemande un jeton.")
    _verifier_jeton(ensuite, quand)
    if len(serveur.demandes) != 2 or ensuite != "jeton-2":
        raise AssertionError(
            f"{quand} : l'appel suivant doit redemander un jeton et renvoyer 'jeton-2' (il a renvoyé {_court(ensuite)}, "
            f"après {len(serveur.demandes)} demande(s) au total). Après une panne, rien ne doit rester en cache.")

    # 4. Deux caches indépendants, et une marge personnalisée
    serveur_a, serveur_b = ServeurAuth(latence=0.5), ServeurAuth(latence=0.5)

    async def deux_caches():
        boucle = asyncio.get_running_loop()
        a, b = Classe(serveur_a, marge=10.0), Classe(serveur_b, marge=10.0)
        jetons = await asyncio.gather(_jeton(a), _jeton(b))
        duree = boucle.time()
        await asyncio.sleep(3570.0 - boucle.time())
        await _jeton(a)
        trop_tot = len(serveur_a.demandes) - 1
        await asyncio.sleep(3595.0 - boucle.time())
        await _jeton(a)
        return jetons, duree, trop_tot, len(serveur_a.demandes) - 1 - trop_tot

    quand = "Avec deux caches (deux serveurs) et marge=10"
    appel = _lancer(deux_caches)
    _verifier_appel(appel, quand)
    _sans_erreur(appel, quand)
    jetons, duree, trop_tot, renouvele = appel.resultat
    if duree > 0.5 + EPS:
        raise AssertionError(
            f"{quand} : les deux premiers jetons mettent {_s(duree)} au lieu de 0,5 s : un cache a attendu l'autre. Votre "
            "verrou est partagé par tous les caches (un attribut de CLASSE ?) : créez-le dans __init__, "
            "`self._verrou = asyncio.Lock()`.")
    if trop_tot:
        raise AssertionError(
            f"{quand} : à 30 s de l'expiration, le jeton est encore bon (marge de 10 s) et vous le renouvelez. Utilisez "
            "`self.marge`, pas 60 écrit en dur.")
    if not renouvele:
        raise AssertionError(f"{quand} : à 5 s de l'expiration (moins que la marge de 10 s), renouvelez le jeton.")


# ============================================================================ Défi 102 : le premier qui répond


def _trois_serveurs(erreurs=(None, None, None)) -> list[ServiceScores]:
    noms, latences = ("paris", "francfort", "dublin"), (0.3, 0.1, 0.5)
    return [ServiceScores(latence=latence, nom=nom, erreurs={"C004": erreur} if erreur else None)
            for nom, latence, erreur in zip(noms, latences, erreurs)]


def _lancer_serveurs(f, serveurs, client_id):
    return _lancer(lambda: f(list(serveurs), client_id), observer=lambda: [(s.nom, s.en_cours) for s in serveurs])


def _orphelins(serveurs, appel, quand) -> None:
    for nom, en_cours in appel.observe or ():
        if en_cours:
            raise AssertionError(
                f"{quand} : vous avez votre réponse, mais le serveur {nom} travaille encore pour rien "
                f"(en_cours = {en_cours}). Annulez les tâches perdantes : `tache.cancel()` (dans un `finally`, "
                "pour le faire dans tous les cas).")
    if appel.restantes:
        raise AssertionError(f"{quand} : {appel.restantes} tâche(s) tournent encore après votre retour : annulez-les.")


def verifier_premier_qui_repond(f) -> None:
    serveurs = _trois_serveurs()
    paris, francfort, dublin = serveurs
    quand = "Avec paris (0,3 s), francfort (0,1 s) et dublin (0,5 s)"
    appel = _lancer_serveurs(f, serveurs, "C004")
    _verifier_appel(appel, quand)
    _sans_erreur(appel, quand)
    _sans_coroutine(appel.resultat, quand)
    oublies = [s.nom for s in serveurs if not s.appels]
    if oublies:
        raise AssertionError(
            f"{quand} : {', '.join(oublies)} n'ont pas été interrogés. Interrogez-les TOUS, en même temps "
            "(asyncio.create_task pour chacun), et gardez la première réponse.")
    if appel.resultat != 0.91:
        raise AssertionError(f"{quand} : renvoyez le score de C004 (0.91), pas {_court(appel.resultat)}.")
    if appel.duree > 0.1 + EPS:
        if appel.duree >= 0.5 - EPS:
            raise AssertionError(
                f"{quand}, votre fonction met {_s(appel.duree)} : elle attend les 3 réponses (gather ?). La page est alors "
                "aussi lente que le plus LENT des serveurs. Gardez la première : asyncio.as_completed, ou asyncio.wait("
                "…, return_when=asyncio.FIRST_COMPLETED).")
        if appel.duree >= 0.3 - EPS:
            raise AssertionError(
                f"{quand}, votre fonction met {_s(appel.duree)} : elle attend le premier serveur DE LA LISTE (paris), pas "
                "le premier qui RÉPOND (francfort, 0,1 s). asyncio.as_completed(taches) donne les tâches dans l'ordre où "
                "elles finissent.")
        raise AssertionError(f"{quand}, votre fonction met {_s(appel.duree)} au lieu de 0,1 s.")
    _orphelins(serveurs, appel, quand)

    serveurs = _trois_serveurs(erreurs=(None, ServiceIndisponible("francfort : 503"), None))
    quand = "Quand francfort (le plus rapide, 0,1 s) répond par une ERREUR"
    appel = _lancer_serveurs(f, serveurs, "C004")
    _verifier_appel(appel, quand)
    if _contient(appel.erreur, ServiceIndisponible):
        raise AssertionError(
            f"{quand}, votre fonction lève {_decrire_erreur(appel.erreur)}. Une erreur n'est pas une réponse : attendez "
            "les autres serveurs (paris répond à 0,3 s). Ne levez une erreur que si TOUS ont échoué.")
    _sans_erreur(appel, quand)
    if appel.resultat != 0.91:
        raise AssertionError(f"{quand} : renvoyez la réponse de paris (0.91), pas {_court(appel.resultat)}.")
    if appel.duree > 0.3 + EPS:
        raise AssertionError(
            f"{quand}, votre fonction met {_s(appel.duree)} au lieu de 0,3 s (la réponse de paris) : n'attendez pas dublin.")
    _orphelins(serveurs, appel, quand)

    erreurs = (ServiceIndisponible("paris : 503"), ServiceIndisponible("francfort : 503"),
               ServiceIndisponible("dublin : 503"))
    serveurs = _trois_serveurs(erreurs=erreurs)
    quand = "Quand les 3 serveurs échouent"
    appel = _lancer_serveurs(f, serveurs, "C004")
    _verifier_appel(appel, quand)
    if appel.erreur is None:
        _fermer_coroutines(appel.resultat)
        raise AssertionError(
            f"{quand}, votre fonction renvoie {_court(appel.resultat)}. Levez une erreur (celle d'un des serveurs) : un "
            "score inventé serait pire qu'un message « service indisponible ».")
    if not any(_contient(appel.erreur, erreur) for erreur in erreurs):
        raise AssertionError(
            f"{quand}, votre fonction lève {_decrire_erreur(appel.erreur)}. Levez l'une des erreurs reçues des serveurs "
            "(ou un ExceptionGroup qui les contient).")

    serveur = ServiceScores(latence=0.2, nom="paris")
    quand = "Avec un seul serveur, pour C005 (score 0.0)"
    appel = _lancer(lambda: f([serveur], "C005"))
    _verifier_appel(appel, quand)
    _sans_erreur(appel, quand)
    if appel.resultat != 0.0 or appel.resultat is None or appel.resultat is False:
        raise AssertionError(
            f"{quand} : renvoyez 0.0, pas {_court(appel.resultat)}. 0.0 est un vrai score (mais « faux » pour un `if`).")

    quand = "Avec une liste de serveurs vide"
    appel = _lancer(lambda: f([], "C001"))
    _verifier_appel(appel, quand)
    if not isinstance(appel.erreur, ValueError):
        obtenu = _decrire_erreur(appel.erreur) if appel.erreur else f"un résultat ({_court(appel.resultat)})"
        raise AssertionError(f"{quand} : levez ValueError, pas {obtenu}.")


# ============================================================================ Défi 103 : Queue et travailleurs


_INTERBLOCAGE_103 = (
    "Le plus souvent : les travailleurs attendent `await file.get()` alors que le lecteur de pages a fini (prévoyez un "
    "signal de fin par travailleur, ou annulez-les une fois le travail fait), ou le lecteur attend `file.put()` alors "
    "que plus aucun travailleur ne vide la file.")
_INTERBLOCAGE_ERREUR_103 = (
    "Un travailleur est mort sur cette erreur, et le reste du programme l'attend : `file.join()` attend un "
    "`task_done()` qui ne viendra jamais, ou le lecteur attend une place dans la file pleine. Lancez lecteur et "
    "travailleurs dans un `asyncio.TaskGroup` : à la première erreur, il annule les autres et fait remonter l'erreur.")


def _clients_103(n: int = 40):
    ids = [f"C{k:03d}" for k in range(1, n + 1)]
    scores = {c: round((k * 37 % 100) / 100, 2) for k, c in enumerate(ids, 1)}
    latences = {c: 1.2 for k, c in enumerate(ids, 1) if k % 6 == 0}
    return ids, scores, latences


def _verifier_arret_sur_erreur(appel, service, attendue, quand, interblocage) -> None:
    _verifier_appel(appel, quand, interblocage=interblocage)
    if appel.erreur is None:
        nombre = len(appel.resultat) if isinstance(appel.resultat, dict) else _court(appel.resultat)
        raise AssertionError(
            f"{quand}, votre fonction renvoie {nombre} scores comme si de rien n'était. L'erreur doit remonter : un "
            "client sans score passerait inaperçu jusqu'à la campagne.")
    if not _contient(appel.erreur, attendue):
        raise AssertionError(
            f"{quand}, votre fonction lève {_decrire_erreur(appel.erreur)} au lieu de l'erreur reçue "
            f"({_decrire_erreur(attendue) if isinstance(attendue, BaseException) else attendue.__name__}).")
    if appel.observe or appel.restantes:
        raise AssertionError(
            f"{quand} : l'erreur remonte, mais {max(appel.observe or 0, appel.restantes)} tâche(s) tournent encore "
            "derrière vous (travailleurs, lecteur de pages). Arrêtez-les : un `asyncio.TaskGroup` annule toutes les "
            "tâches du groupe dès que l'une échoue.")


def verifier_scorer_tout(f) -> None:
    ids, scores, latences = _clients_103()
    service = ServiceScores(scores, latence=0.2, latences=latences)
    pages = Pages(ids, taille=8, latence=0.3, service=service)
    quand = "Avec 40 clients en 5 pages de 8 (0,3 s par page), nb_travailleurs=4 et max_en_attente=6"
    appel = _lancer(lambda: f(pages, service, nb_travailleurs=4, max_en_attente=6), observer=lambda: service.en_cours)
    _verifier_appel(appel, quand, interblocage=_INTERBLOCAGE_103)
    _sans_erreur(appel, quand)
    _sans_coroutine(appel.resultat, quand)
    demandes = collections.Counter(client for client, _ in service.appels)
    doublons = [client for client, n in demandes.items() if n > 1]
    if doublons:
        raise AssertionError(f"{quand} : {doublons[0]} a été scoré {demandes[doublons[0]]} fois. Une fois suffit.")
    _comparer_scores(appel.resultat, scores, quand)
    if service.max_en_cours > 4:
        raise AssertionError(
            f"{quand} : jusqu'à {service.max_en_cours} appels au service en même temps, pour nb_travailleurs=4. Le "
            "service des scores tomberait : seuls les travailleurs l'appellent, chacun un client à la fois.")
    avances = [avance for _, avance in pages.demandes]
    pire = max(avances, default=0)
    if pire > 6:
        numero = avances.index(pire) + 1
        raise AssertionError(
            f"{quand} : quand vous avez demandé la page {numero}, {pire} clients déjà lus attendaient encore leur "
            "travailleur (max_en_attente=6). Avec 2 millions de clients, vous les chargeriez tous en mémoire. Bornez la "
            "file : `asyncio.Queue(maxsize=max_en_attente)`, dont le `put()` ATTEND quand elle est pleine.")
    if service.max_en_cours < 4:
        raise AssertionError(
            f"{quand} : jamais plus de {service.max_en_cours} appel(s) au service en même temps. Vos 4 travailleurs "
            "doivent travailler ENSEMBLE : chacun dans sa tâche (create_task ou TaskGroup), avec sa boucle "
            "`while … await file.get()`.")
    lecture = pages.latence * (len(pages.pages) + 1)
    travail = sum(latences.get(c, 0.2) for c in ids) / 4
    borne = lecture + travail + max(latences.values())
    if appel.duree > borne + EPS:
        raise AssertionError(
            f"{quand}, le traitement met {_s(appel.duree)} (on attend au plus {_s(borne)}). Des travailleurs restent sans "
            "rien faire : chacun doit prendre le client suivant dès qu'il a fini (pas de paquets qui attendent le plus "
            "lent), et la page suivante se lit PENDANT qu'ils travaillent.")
    if appel.restantes or appel.observe:
        raise AssertionError(
            f"{quand} : à la fin, {max(appel.restantes, appel.observe or 0)} tâche(s) tournent encore (des travailleurs qui "
            "attendent `file.get()` pour toujours ?). Arrêtez-les : un signal de fin par travailleur, ou cancel().")

    panne = ServiceIndisponible("le service des scores ne répond plus (503)")
    service = ServiceScores(scores, latence=0.2, latences=latences, erreurs={"C013": panne})
    pages = Pages(ids, taille=8, latence=0.3, service=service)
    quand = "Quand le service lève ServiceIndisponible pour C013"
    appel = _lancer(lambda: f(pages, service, nb_travailleurs=4, max_en_attente=6), observer=lambda: service.en_cours)
    _verifier_arret_sur_erreur(appel, service, panne, quand, _INTERBLOCAGE_ERREUR_103)
    instant = next(t for client, t, issue in service.reponses if client == "C013" and issue == "erreur")
    apres = [client for client, t in service.appels if t > instant + EPS]
    if apres:
        raise AssertionError(
            f"{quand} (à {_s(instant)}) : {len(apres)} nouveaux appels au service APRÈS l'erreur ({', '.join(apres[:3])}…). "
            "Arrêtez tout dès la première erreur, au lieu de continuer à travailler pour rien.")

    service = ServiceScores(scores, latence=0.2, latences=latences)
    pages = Pages(ids, taille=8, latence=0.3, service=service, panne_a_la_page=2)
    quand = "Quand la lecture de la 3e page échoue (ServiceIndisponible)"
    appel = _lancer(lambda: f(pages, service, nb_travailleurs=4, max_en_attente=6), observer=lambda: service.en_cours)
    _verifier_arret_sur_erreur(
        appel, service, ServiceIndisponible, quand,
        "Le lecteur de pages est mort sur cette erreur, et vos travailleurs attendent un client qui n'arrivera jamais "
        "(ou vous attendez `file.join()`). Un `asyncio.TaskGroup` annule tout le groupe dès qu'une tâche échoue.")

    service = ServiceScores(latence=0.2)
    quand = "Sans aucune page"
    appel = _lancer(lambda: f(Pages([], service=service), service))
    _verifier_appel(appel, quand, interblocage=_INTERBLOCAGE_103)
    _sans_erreur(appel, quand)
    if appel.resultat != {} or not isinstance(appel.resultat, dict):
        raise AssertionError(f"{quand} : renvoyez {{}}, pas {_court(appel.resultat)}.")

    service = ServiceScores(latence=0.2)
    pages = Pages([], service=service)
    pages.pages = [[], ["C001"], []]
    quand = "Avec des pages vides : [[], ['C001'], []]"
    appel = _lancer(lambda: f(pages, service))
    _verifier_appel(appel, quand, interblocage=_INTERBLOCAGE_103)
    _sans_erreur(appel, quand)
    _comparer_scores(appel.resultat, {"C001": 0.82}, quand)

    ids = [f"C00{k}" for k in range(1, 9)]
    service = ServiceScores(latence=0.2)
    pages = Pages(ids, taille=3, latence=0.1, service=service)
    quand = "Avec nb_travailleurs=1 et max_en_attente=1 (8 clients en pages de 3)"
    appel = _lancer(lambda: f(pages, service, nb_travailleurs=1, max_en_attente=1))
    _verifier_appel(appel, quand, interblocage=_INTERBLOCAGE_103)
    _sans_erreur(appel, quand)
    _comparer_scores(appel.resultat, SCORES_EXEMPLE, quand)
    if service.max_en_cours > 1:
        raise AssertionError(f"{quand} : {service.max_en_cours} appels en même temps, pour un seul travailleur.")
    pire = max((avance for _, avance in pages.demandes), default=0)
    if pire > 1:
        raise AssertionError(f"{quand} : {pire} clients lus d'avance, pour max_en_attente=1.")

    for parametres in ({"nb_travailleurs": 0}, {"max_en_attente": 0}):
        service = ServiceScores(latence=0.2)
        quand = f"Avec {', '.join(f'{k}={v}' for k, v in parametres.items())}"
        appel = _lancer(lambda: f(Pages(["C001"], service=service), service, **parametres))
        _verifier_appel(appel, quand, interblocage=f"{quand}, levez ValueError dès le début.")
        if not isinstance(appel.erreur, ValueError):
            obtenu = _decrire_erreur(appel.erreur) if appel.erreur else f"un résultat ({_court(appel.resultat)})"
            raise AssertionError(f"{quand} : levez ValueError, pas {obtenu}.")


SERIE = Serie(
    code="16",
    titre="Faire plusieurs choses à la fois : gather, délais, Semaphore, Lock, annulation, Queue (asyncio)",
    module_eleve="defis.serie_16",
    module_corrige="defis.corrige_serie_16",
    defis=(
        Defi(98, "scores_en_parallele", "🟢", "—", verifier_scores_en_parallele),
        Defi(99, "score_ou_defaut", "🟢", "—", verifier_score_ou_defaut),
        Defi(100, "envoyer_campagne", "🟡", "—", verifier_envoyer_campagne),
        Defi(101, "CacheJeton", "🟡", "—", verifier_CacheJeton),
        Defi(102, "premier_qui_repond", "🟡", "—", verifier_premier_qui_repond),
        Defi(103, "scorer_tout", "🔴", "—", verifier_scorer_tout),
    ),
)

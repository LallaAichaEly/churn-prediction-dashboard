"""Série 16 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_16.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la situation à laquelle vous n'aviez pas pensé.

Fil rouge de la série : asyncio ne fait RIEN en parallèle tout seul. Une coroutine avance jusqu'à
son prochain `await`, puis rend la main à la boucle, qui fait avancer une autre tâche. Trois réflexes
reviennent dans chaque défi :

1. LANCER d'abord (gather, create_task, TaskGroup), ATTENDRE ensuite. `await` dans une boucle `for`,
   c'est un appel après l'autre ;
2. tout ce qu'on lance, on le termine : une tâche abandonnée (réponse trop lente, serveur perdant,
   travailleur après une erreur) continue de tourner et de consommer, sauf si on l'annule ;
3. entre deux `await`, le monde a pu changer : ce qu'on a vérifié AVANT un `await` doit être
   revérifié APRÈS (défi 101), et une ressource partagée se protège (Semaphore, Lock, Queue).

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Sequence

from defis.concurrence import (  # noqa: F401
    SCORES_EXEMPLE,
    Interblocage,
    NumeroInvalide,
    OperateurSMS,
    Pages,
    ServeurAuth,
    ServiceIndisponible,
    ServiceScores,
    Surcharge,
    essayer,
    executer,
    service_exemple,
)

# ============================================================================ Défi 98 : gather


async def scores_en_parallele(service: ServiceScores, client_ids: Sequence[str]) -> dict[str, float]:
    # dict.fromkeys garde le premier ordre d'apparition et retire les doublons (série 07).
    uniques = list(dict.fromkeys(client_ids))
    # On CRÉE toutes les coroutines d'abord, puis gather les fait avancer ensemble : pendant qu'une
    # attend sa réponse, les autres envoient leur demande. Les résultats reviennent dans l'ordre des
    # coroutines (pas dans l'ordre d'arrivée) : zip les remet en face des bons clients.
    scores = await asyncio.gather(*(service.score(client_id) for client_id in uniques))
    return dict(zip(uniques, scores))

    # Le piège : `{c: await service.score(c) for c in client_ids}`. C'est juste… et séquentiel : chaque
    # `await` attend la réponse avant de demander le score suivant. 8 × 0,2 s = 1,6 s au lieu de 0,2 s.
    # Une erreur (client inconnu) : gather la fait remonter telle quelle, c'est le comportement voulu.


# ============================================================================ Défi 99 : délai maximal


async def score_ou_defaut(service: ServiceScores, client_id: str, delai: float, defaut: float | None = None):
    if not delai > 0:  # « not … > 0 » refuse aussi NaN
        raise ValueError(f"le délai doit être strictement positif, pas {delai!r}")
    try:
        # wait_for ANNULE l'appel s'il dépasse le délai, et attend que l'annulation soit faite.
        # Équivalent depuis Python 3.11 : `async with asyncio.timeout(delai): return await service.score(…)`.
        return await asyncio.wait_for(service.score(client_id), timeout=delai)
    except TimeoutError:  # depuis Python 3.11, asyncio.TimeoutError EST TimeoutError
        return defaut

    # Deux pièges :
    # - `except Exception: return defaut` transforme un client inconnu (KeyError) ou un service en
    #   panne en « pas de réponse à temps ». La page afficherait un score par défaut au lieu d'une erreur ;
    # - `return await … or defaut` : un score de 0.0 est « faux » en Python, il serait remplacé.


# ============================================================================ Défi 100 : Semaphore


async def envoyer_campagne(operateur: OperateurSMS, clients: Iterable[str], texte: str,
                           max_simultanes: int = 10) -> tuple[dict[str, str], dict[str, Exception]]:
    if max_simultanes < 1:
        raise ValueError(f"max_simultanes doit valoir au moins 1, pas {max_simultanes!r}")
    uniques = list(dict.fromkeys(clients))  # un client présent deux fois ne reçoit qu'un SMS
    # Créé ICI, dans la fonction : un sémaphore par campagne, avec la limite demandée.
    places = asyncio.Semaphore(max_simultanes)

    async def envoyer_un(client_id: str) -> str:
        async with places:  # attend qu'une des max_simultanes places se libère ; la rend même en cas d'erreur
            return await operateur.envoyer(client_id, texte)

    # return_exceptions=True : une erreur devient un RÉSULTAT comme un autre, au lieu d'interrompre
    # gather (qui laisserait les autres envois tourner sans personne pour les attendre).
    resultats = await asyncio.gather(*(envoyer_un(c) for c in uniques), return_exceptions=True)
    envoyes: dict[str, str] = {}
    echecs: dict[str, Exception] = {}
    for client_id, resultat in zip(uniques, resultats):
        if isinstance(resultat, Exception):
            echecs[client_id] = resultat  # l'exception elle-même : on saura laquelle relancer demain
        else:
            envoyes[client_id] = resultat
    return envoyes, echecs

    # Le piège « par paquets » : `for i in range(0, n, 10): await gather(*paquet[i:i+10])`. La limite
    # est respectée, mais chaque paquet attend son SMS le plus lent, pendant que 9 places restent vides.
    # Avec le sémaphore, dès qu'un envoi finit, le suivant part.


# ============================================================================ Défi 101 : Lock


class CacheJeton:
    def __init__(self, serveur: ServeurAuth, marge: float = 60.0) -> None:
        self.serveur = serveur
        self.marge = marge
        self._jeton: str | None = None
        self._expire_a = float("-inf")
        # Un verrou PAR cache, créé dans __init__ : deux caches de deux serveurs ne s'attendent pas.
        self._verrou = asyncio.Lock()

    def _valable(self) -> bool:
        maintenant = asyncio.get_running_loop().time()
        return self._jeton is not None and maintenant < self._expire_a - self.marge

    async def jeton(self) -> str:
        if self._valable():  # chemin rapide : aucun verrou quand le jeton est bon (le cas le plus fréquent)
            return self._jeton
        async with self._verrou:
            # DOUBLE VÉRIFICATION : pendant que nous attendions le verrou, une autre tâche a peut-être
            # déjà renouvelé le jeton. Sans cette ligne, les 50 tâches feraient 50 demandes… une par une.
            if not self._valable():
                debut = asyncio.get_running_loop().time()
                jeton, duree_de_vie = await self.serveur.nouveau_jeton()
                # On compte depuis la DEMANDE (prudent : le serveur a créé le jeton pendant notre attente).
                # Si nouveau_jeton lève une erreur, on n'arrive jamais ici : rien n'est gardé, et
                # `async with` rend le verrou quand même.
                self._jeton, self._expire_a = jeton, debut + duree_de_vie
            return self._jeton

    # Le piège : `if not self._valable(): self._jeton, … = await self.serveur.nouveau_jeton()` sans
    # verrou. Entre le test et la réponse, il y a un `await` : les 49 autres tâches passent le même
    # test (« pas de jeton ») et demandent chacune le leur. En threads, on appelle ça une
    # « situation de compétition » ; asyncio ne l'évite pas, il la rend juste plus rare… et plus
    # surprenante, car elle n'arrive qu'aux `await`.


# ============================================================================ Défi 102 : le premier qui répond


async def premier_qui_repond(serveurs: Sequence[ServiceScores], client_id: str) -> float:
    if not serveurs:
        raise ValueError("il faut au moins un serveur")
    taches = [asyncio.create_task(serveur.score(client_id)) for serveur in serveurs]
    try:
        derniere_erreur: Exception | None = None
        # as_completed donne les tâches dans l'ORDRE OÙ ELLES FINISSENT : la première réponse d'abord.
        for prochaine in asyncio.as_completed(taches):
            try:
                return await prochaine
            except Exception as exc:  # ce serveur a échoué : ce n'est pas une réponse, on attend les autres
                derniere_erreur = exc
        raise derniere_erreur  # tous ont échoué
    finally:
        # Quoi qu'il arrive (réponse, erreur, ou annulation de NOTRE appel), les serveurs plus lents
        # n'ont plus de raison de travailler. cancel() sur une tâche déjà finie ne fait rien.
        for tache in taches:
            tache.cancel()

    # Variante avec asyncio.wait(…, return_when=FIRST_COMPLETED) dans une boucle : on regarde les
    # tâches finies, on renvoie la première réussie, sinon on recommence avec celles « en attente ».


# ============================================================================ Défi 103 : Queue et travailleurs


async def scorer_tout(pages, service: ServiceScores, nb_travailleurs: int = 4,
                      max_en_attente: int = 20) -> dict[str, float]:
    if nb_travailleurs < 1 or max_en_attente < 1:
        raise ValueError("il faut au moins un travailleur et une place dans la file")
    # maxsize : put() ATTEND quand la file est pleine. C'est la « contre-pression » : le lecteur de
    # pages ralentit au rythme des travailleurs, au lieu de charger 2 millions de clients en mémoire.
    file: asyncio.Queue = asyncio.Queue(maxsize=max_en_attente)
    scores: dict[str, float] = {}
    FIN = object()  # le signal de fin : un objet unique, impossible à confondre avec un client

    async def lire_les_pages() -> None:
        async for page in pages:
            for client_id in page:
                await file.put(client_id)
        for _ in range(nb_travailleurs):  # un signal de fin par travailleur
            await file.put(FIN)

    async def travailleur() -> None:
        while (client_id := await file.get()) is not FIN:
            scores[client_id] = await service.score(client_id)

    # TaskGroup (Python 3.11) : si UNE tâche échoue, il ANNULE toutes les autres, attend qu'elles
    # soient terminées, puis lève l'erreur (dans un ExceptionGroup, attrapable avec `except*`).
    # Sans lui, un travailleur mort sur une erreur laisserait les autres attendre `file.get()`
    # pour toujours.
    async with asyncio.TaskGroup() as groupe:
        groupe.create_task(lire_les_pages())
        for _ in range(nb_travailleurs):
            groupe.create_task(travailleur())
    return scores

    # Le piège classique : `await file.join()` pour attendre la fin, sans TaskGroup. Si un travailleur
    # meurt sur une erreur, son `task_done()` n'est jamais appelé et `join()` attend pour toujours :
    # le programme se fige, sans message. C'est ce que le correcteur appelle un « interblocage ».


# ============================================================================ Tests (comme dans tests/)


def test_les_scores_arrivent_en_parallele():
    service = ServiceScores(latence=0.2)
    scores, duree = executer(scores_en_parallele(service, ["C001", "C004", "C001", "C005"]))
    assert scores == {"C001": 0.82, "C004": 0.91, "C005": 0.0} and list(scores) == ["C001", "C004", "C005"]
    assert duree == 0.2 and len(service.appels) == 3  # 3 appels en même temps, pas 4 l'un après l'autre


def test_un_score_trop_lent_est_abandonne_et_annule():
    lent = ServiceScores(latence=5.0)
    score, duree = executer(score_ou_defaut(lent, "C001", delai=1.0, defaut=0.5))
    assert score == 0.5 and duree == 1.0
    assert lent.en_cours == 0 and lent.annules == ["C001"]
    zero, _ = executer(score_ou_defaut(ServiceScores(latence=0.1), "C005", delai=1.0, defaut=0.5))
    assert zero == 0.0


def test_la_campagne_respecte_la_limite_et_continue_malgre_un_numero_invalide():
    operateur = OperateurSMS(max_simultanes=3, invalides={"C002"})
    clients = [f"C00{n}" for n in range(1, 9)]
    (envoyes, echecs), duree = executer(envoyer_campagne(operateur, clients, "Offre", max_simultanes=3))
    assert list(envoyes) == [c for c in clients if c != "C002"]
    assert list(echecs) == ["C002"] and isinstance(echecs["C002"], NumeroInvalide)
    assert operateur.refus == [] and operateur.max_en_cours == 3
    assert abs(duree - 0.6) < 1e-9  # 8 envois de 0,2 s, 3 à la fois : 3 « vagues »


def test_cinquante_demandes_simultanees_un_seul_jeton():
    serveur = ServeurAuth(latence=0.5)

    async def scenario():
        cache = CacheJeton(serveur)
        jetons = await asyncio.gather(*(cache.jeton() for _ in range(50)))
        await asyncio.sleep(3550)  # le jeton expire à 3600 s, la marge est de 60 s : il faut le renouveler
        return jetons, await cache.jeton()

    (jetons, plus_tard), _ = executer(scenario())
    assert set(jetons) == {"jeton-1"} and plus_tard == "jeton-2" and len(serveur.demandes) == 2


def test_le_serveur_le_plus_rapide_gagne_et_les_autres_sont_annules():
    paris = ServiceScores(latence=0.3, nom="paris")
    francfort = ServiceScores(latence=0.1, nom="francfort", erreurs={"C004": ServiceIndisponible("503")})
    dublin = ServiceScores(latence=0.5, nom="dublin")

    async def scenario():
        score = await premier_qui_repond([paris, francfort, dublin], "C004")
        await asyncio.sleep(0)  # laisse à l'annulation le temps d'arriver
        return score

    score, duree = executer(scenario())
    assert score == 0.91 and duree == 0.3  # francfort a échoué : c'est paris qui répond
    assert dublin.annules == ["C004"] and dublin.en_cours == 0


def test_scorer_tout_lit_peu_d_avance_et_s_arrete_sur_une_erreur():
    ids = [f"C{n:03d}" for n in range(1, 31)]
    service = ServiceScores({c: 0.5 for c in ids}, latence=0.2)
    pages = Pages(ids, taille=5, latence=0.1, service=service)
    scores, _ = executer(scorer_tout(pages, service, nb_travailleurs=3, max_en_attente=4))
    assert scores == {c: 0.5 for c in ids} and service.max_en_cours == 3
    assert max(avance for _, avance in pages.demandes) <= 4

    en_panne = ServiceScores({c: 0.5 for c in ids}, latence=0.2, erreurs={"C010": ServiceIndisponible("503")})
    try:
        executer(scorer_tout(Pages(ids, service=en_panne), en_panne, nb_travailleurs=3))
    except* ServiceIndisponible:
        pass
    else:
        raise AssertionError("l'erreur du service aurait dû remonter")
    assert en_panne.en_cours == 0  # les autres travailleurs ont été annulés : rien ne tourne encore

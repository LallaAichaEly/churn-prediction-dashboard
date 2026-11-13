"""Série 16 — Faire plusieurs choses à la fois : asyncio (gather, délais, Semaphore, Lock, annulation, Queue).

⚡ VOUS VOULEZ DÉVELOPPER ? UN PROGRAMME PASSE L'ESSENTIEL DE SON TEMPS À ATTENDRE LES AUTRES.

Demander un score au service de modélisation : 0,2 s. Envoyer un SMS : 0,3 s. Lire une page de
clients sur une API : 0,1 s. Pendant ces attentes, votre processeur ne fait RIEN. Pour 5 000 clients,
un appel après l'autre, la campagne de rétention met 25 minutes à partir ; en lançant les appels
ensemble, 10 secondes. C'est le travail d'`asyncio`, et c'est aussi ce que fait FastAPI (série 12)
quand il sert plusieurs requêtes à la fois.

Les six défis suivent la campagne de rétention, de la plus simple des attentes à un vrai pipeline :

     98 demander 8 scores en même temps         99 ne pas attendre une réponse plus d'une seconde
    100 envoyer 500 SMS, 10 à la fois           101 un seul jeton d'accès pour 50 tâches pressées
    102 interroger 3 serveurs, garder le 1er    103 lire des pages, scorer avec 4 travailleurs, sans déborder

Prérequis : la série 00 et les exceptions du défi 40 (série 06). La classe du défi 48 (série 07) aide
pour le défi 101. Rien à installer : `asyncio` fait partie de Python (3.11 ou plus récent).

📌 Ce qu'il faut savoir d'asyncio (une page, à relire au besoin) :

- `async def f(): ...` définit une COROUTINE. Appeler `f()` n'exécute RIEN : ça fabrique un objet
  coroutine, qu'il faut attendre avec `await f()`. Oublier `await`, c'est obtenir l'objet au lieu du
  résultat (Python le signale par « coroutine 'f' was never awaited ») ;
- `await` veut dire : « je m'arrête ici le temps que ça arrive ; pendant ce temps, faites avancer les
  autres ». Il n'y a qu'UN seul fil d'exécution : la BOUCLE (event loop) passe d'une tâche à l'autre,
  et uniquement aux `await`. Entre deux `await`, votre code n'est jamais interrompu ;
- `await` dans une boucle `for` = un appel APRÈS l'autre. Pour avancer ensemble, on lance d'abord,
  on attend ensuite : `await asyncio.gather(f(1), f(2), f(3))` renvoie la liste des 3 résultats, DANS
  L'ORDRE où on les a donnés ; `tache = asyncio.create_task(f(1))` lance en arrière-plan, `await tache`
  récupère le résultat plus tard ;
- `asyncio.sleep(s)` attend sans bloquer les autres. `time.sleep(s)` BLOQUE toute la boucle : jamais
  dans une coroutine (pareil pour `requests.get`, qui attend sans rendre la main) ;
- `tache.cancel()` ANNULE une tâche : une `asyncio.CancelledError` est levée dans la tâche, à son
  `await` en cours. Une tâche qu'on n'attend plus et qu'on n'annule pas continue de tourner ;
- outils du module, vus dans les défis : `asyncio.wait_for` et `asyncio.timeout` (délai maximal),
  `asyncio.Semaphore` (au plus N à la fois), `asyncio.Lock` (un seul à la fois), `asyncio.as_completed`
  et `asyncio.wait` (dans l'ordre d'arrivée), `asyncio.Queue` (file d'attente entre tâches) et
  `asyncio.TaskGroup` (un groupe de tâches qui finissent ensemble, ou échouent ensemble) ;
- asyncio sert à ATTENDRE à plusieurs (réseau, base de données, fichiers). Pour CALCULER plus vite
  (entraîner un modèle), il ne sert à rien : un seul fil, un seul cœur.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_16 import *
    >>> service = service_exemple()                 # le service des scores : 8 clients, 0,2 s par réponse
    >>> essayer(service.score("C001"))
    ⏱️  0,2 s (horloge virtuelle : rien n'a vraiment attendu)
    0.82
    >>> essayer(scores_en_parallele(service, ["C001", "C002", "C003"]))     # défi 98

`essayer(...)` exécute votre coroutine sur une HORLOGE VIRTUELLE : quand toutes les tâches attendent,
l'horloge saute au prochain réveil. Il affiche la durée qu'aurait prise votre code, sans attendre pour
de vrai : 500 SMS de 0,3 s, c'est instantané. Le correcteur utilise la même horloge, et il mesure. Pour
attendre pour de vrai, comme dans un programme : `asyncio.run(scores_en_parallele(service, [...]))`.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code (gardez `async def`).
2. Lancez `python scripts/defis.py --defi 98` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_16.py`.

Les faux services (ServiceScores, OperateurSMS, ServeurAuth, Pages), leurs erreurs et `essayer` sont
dans `defis/concurrence.py`, déjà importés ci-dessous : vous n'avez pas besoin d'ouvrir ce fichier.
Chaque faux service note ce que vous lui demandez : `service.appels`, `service.max_en_cours` (le plus
grand nombre d'appels EN MÊME TEMPS), `service.annules`… Regardez-les après `essayer`.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import asyncio  # noqa: F401 — vous en aurez besoin dans chaque défi
from typing import Iterable, Sequence

from defis.concurrence import (  # noqa: F401 — tout est utile pour essayer vos fonctions
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
    service_exemple,
)

# ============================================================================ gather


async def scores_en_parallele(service: ServiceScores, client_ids: Sequence[str]) -> dict[str, float]:
    """Défi 98 🟢 — Demander les scores de plusieurs clients EN MÊME TEMPS.

    La page « Clients à risque » affiche le score de churn de 8 clients. Chaque score vient d'un
    service distant qui met 0,2 s à répondre (`await service.score(client_id)` renvoie un nombre).
    Un appel après l'autre : 8 × 0,2 = 1,6 s d'attente. Tous ensemble : 0,2 s.

    Renvoyez un dictionnaire `{client_id: score}` :

    - les clients dans l'ORDRE de `client_ids` (l'ordre de la page) ;
    - un client présent deux fois n'est demandé qu'UNE fois au service (et n'apparaît qu'une fois) ;
    - tous les appels partent ensemble : la durée totale est celle de la réponse la plus lente ;
    - une liste vide donne `{}`, sans appeler le service ;
    - un client inconnu fait lever `KeyError` au service : laissez l'erreur remonter.

        >>> service = service_exemple()                     # 8 clients, 0,2 s par réponse
        >>> essayer(scores_en_parallele(service, ["C001", "C004", "C001"]))
        ⏱️  0,2 s (horloge virtuelle : rien n'a vraiment attendu)
        {'C001': 0.82, 'C004': 0.91}
        >>> len(service.appels), service.max_en_cours        # 2 appels, les 2 en même temps
        (2, 2)

    Le piège : `{c: await service.score(c) for c in client_ids}`. Le résultat est juste, mais chaque
    `await` attend la réponse avant de demander le score suivant : c'est un appel après l'autre.

    Ce que vous apprenez : `await` ATTEND. Pour avancer à plusieurs, on lance d'abord, on attend ensuite.

    Indice : `uniques = list(dict.fromkeys(client_ids))` retire les doublons en gardant l'ordre. Puis
    `scores = await asyncio.gather(*(service.score(c) for c in uniques))` : l'étoile passe chaque
    coroutine comme un argument, et gather renvoie les résultats dans le même ordre. `zip` fait le reste.
    """
    raise NotImplementedError


# ============================================================================ Délai maximal et annulation


async def score_ou_defaut(service: ServiceScores, client_id: str, delai: float, defaut: float | None = None):
    """Défi 99 🟢 — Ne pas attendre une réponse plus de `delai` secondes.

    Le service des scores répond d'habitude en 0,2 s… mais parfois en 30 s, quand il est surchargé.
    Le conseiller, lui, ne va pas regarder une page blanche pendant 30 s : au-delà de `delai`
    secondes, on affiche le client avec `defaut` (« score indisponible ») et on passe à la suite.

    Renvoyez le score si la réponse arrive dans les `delai` secondes, sinon `defaut` :

    - si la réponse arrive vite, renvoyez-la TOUT DE SUITE (n'attendez pas la fin du délai) ;
    - si le délai est dépassé, l'appel lent est ANNULÉ : il ne doit pas continuer en arrière-plan,
      à occuper le service pour une réponse que personne n'attend plus ;
    - SEUL le dépassement du délai donne `defaut`. Un client inconnu (`KeyError`) ou un service en
      panne (`ServiceIndisponible`) sont des erreurs, pas des lenteurs : elles remontent ;
    - un score de 0.0 est un vrai score (le client C005) : renvoyez 0.0, pas `defaut` ;
    - `delai` doit être strictement positif, sinon ValueError.

        >>> essayer(score_ou_defaut(service_exemple(latence=0.2), "C001", delai=1.0))
        ⏱️  0,2 s (horloge virtuelle : rien n'a vraiment attendu)
        0.82
        >>> lent = service_exemple(latence=30.0)
        >>> essayer(score_ou_defaut(lent, "C001", delai=1.0, defaut=0.5))
        ⏱️  1 s (horloge virtuelle : rien n'a vraiment attendu)
        0.5

    Ce que vous apprenez : tout appel réseau a besoin d'un délai maximal, et « abandonner » une
    tâche, c'est l'ANNULER (sinon elle continue). Une erreur n'est pas une lenteur.

    Indice : `await asyncio.wait_for(service.score(client_id), timeout=delai)` lève `TimeoutError`
    au-delà du délai, APRÈS avoir annulé l'appel. Attrapez CETTE erreur-là seulement (depuis Python
    3.11, `asyncio.TimeoutError` et `TimeoutError` sont la même classe). Autre écriture possible :
    `async with asyncio.timeout(delai): ...`.
    """
    raise NotImplementedError


# ============================================================================ Semaphore


async def envoyer_campagne(operateur: OperateurSMS, clients: Iterable[str], texte: str,
                           max_simultanes: int = 10) -> tuple[dict[str, str], dict[str, Exception]]:
    """Défi 100 🟡 — Envoyer la campagne de SMS, au plus `max_simultanes` à la fois.

    500 clients à risque vont recevoir le même SMS. L'opérateur répond en 0,2 à 1 s par SMS… et
    REFUSE tout envoi au-delà de 10 en cours (`Surcharge`, l'erreur HTTP 429). Un par un, la campagne
    est trop lente ; tous d'un coup, 490 SMS sont refusés.

    `await operateur.envoyer(client_id, texte)` renvoie l'identifiant du SMS (« sms-0001 »).
    Renvoyez un couple `(envoyes, echecs)` :

    - `envoyes` : `{client_id: identifiant du SMS}`, dans l'ordre de `clients` ;
    - `echecs` : `{client_id: l'exception}` pour les envois refusés (un `NumeroInvalide`, par
      exemple). Gardez l'EXCEPTION elle-même, pas son texte : demain, on relancera seulement les
      erreurs passagères, et il faudra savoir lesquelles c'étaient ;
    - jamais plus de `max_simultanes` envois en cours en même temps, et le reste du temps, le plus
      possible : dès qu'un envoi finit, le suivant part ;
    - un échec n'arrête PAS la campagne : les autres clients reçoivent leur SMS ;
    - un client présent deux fois dans `clients` ne reçoit qu'UN SMS ;
    - `max_simultanes` doit valoir au moins 1, sinon ValueError ; une liste vide donne `({}, {})`.

        >>> operateur = OperateurSMS(max_simultanes=3, invalides={"C002"})
        >>> envoyes, echecs = essayer(envoyer_campagne(operateur, list(SCORES_EXEMPLE), "Offre fidélité", 3))
        ⏱️  0,6 s (horloge virtuelle : rien n'a vraiment attendu)
        >>> echecs
        {'C002': NumeroInvalide('numéro de téléphone invalide pour C002')}
        >>> len(envoyes), operateur.max_en_cours, operateur.refus     # 7 SMS, 3 à la fois, aucun refus
        (7, 3, [])

    Deux pièges : `asyncio.gather(...)` sans précaution, qui s'arrête à la PREMIÈRE erreur (et
    abandonne les autres envois en cours) ; et les « paquets de 10 » (gather sur 10, puis les 10
    suivants…), qui respectent la limite mais attendent à chaque fois le SMS le plus lent du paquet.

    Ce que vous apprenez : limiter la concurrence avec un sémaphore, et traiter une erreur comme un
    résultat parmi d'autres quand on travaille par lots.

    Indice : `places = asyncio.Semaphore(max_simultanes)`, créé DANS la fonction. Une petite coroutine
    `async def envoyer_un(client_id)` qui fait `async with places: return await operateur.envoyer(...)`.
    Puis `await asyncio.gather(*(...), return_exceptions=True)` : une erreur devient alors un élément
    de la liste des résultats (testez-le avec `isinstance(resultat, Exception)`).
    """
    raise NotImplementedError


# ============================================================================ Lock


class CacheJeton:
    """Défi 101 🟡 — Un seul jeton d'accès, même quand 50 tâches le demandent en même temps.

    Pour appeler l'API de l'opérateur, il faut un jeton d'accès, délivré par un serveur
    d'authentification : `jeton, duree_de_vie = await serveur.nouveau_jeton()` (0,5 s d'attente ; le
    jeton est valable `duree_de_vie` secondes, une heure en général). Le serveur limite le nombre de
    demandes : on garde le jeton, et on ne le redemande que quand il va expirer.

    Écrivez la classe :

    - `CacheJeton(serveur, marge=60.0)` : ne contacte PAS le serveur à la création ;
    - `await cache.jeton()` renvoie le jeton (le texte « jeton-1 », pas le couple). Il demande un
      nouveau jeton s'il n'en a pas encore, ou si le sien expire dans moins de `marge` secondes
      (pour ne jamais envoyer un jeton qui expire en route). Sinon, il renvoie celui qu'il a, sans
      rien demander ;
    - l'heure, c'est celle de la boucle : `asyncio.get_running_loop().time()` (en vrai, c'est la même
      que `time.monotonic()` ; chez le correcteur, elle avance à toute vitesse). Comptez la durée de
      vie depuis la demande ou depuis la réponse, au choix : la marge couvre la différence ;
    - si 50 tâches appellent `jeton()` EN MÊME TEMPS sans jeton valable, le serveur ne reçoit qu'UNE
      demande, et les 50 reçoivent le même jeton, au bout de 0,5 s (pas 50 × 0,5 s) ;
    - si la demande échoue (`ServiceIndisponible`), l'erreur remonte, rien n'est gardé, et le
      prochain appel redemande ;
    - deux caches (deux serveurs) sont indépendants : l'un n'attend jamais l'autre.

        >>> serveur = ServeurAuth(latence=0.5)                # jetons valables une heure
        >>> async def cinquante_appels():
        ...     cache = CacheJeton(serveur)
        ...     return await asyncio.gather(*(cache.jeton() for _ in range(50)))
        >>> set(essayer(cinquante_appels()))
        ⏱️  0,5 s (horloge virtuelle : rien n'a vraiment attendu)
        {'jeton-1'}
        >>> len(serveur.demandes)                             # une seule demande pour 50 appels
        1

    Le piège : « si pas de jeton valable, en demander un ». Entre ce test et la réponse du serveur, il
    y a un `await` : les 49 autres tâches passent le même test pendant ce temps, et demandent chacune
    leur jeton. C'est une situation de compétition (race condition), version asyncio.

    Ce que vous apprenez : un état partagé qu'on vérifie AVANT un `await` peut avoir changé APRÈS ;
    on le protège avec un verrou, et on le revérifie une fois le verrou obtenu.

    Indice : dans `__init__`, un verrou par cache : `self._verrou = asyncio.Lock()`. Dans `jeton()` :
    `async with self._verrou:` (il est rendu même en cas d'erreur), PUIS on regarde si le jeton est
    valable. Ainsi, les 49 tâches qui attendaient le verrou trouvent le jeton tout neuf de la première.
    """

    def __init__(self, serveur: ServeurAuth, marge: float = 60.0) -> None:
        raise NotImplementedError

    async def jeton(self) -> str:
        raise NotImplementedError


# ============================================================================ Le premier qui répond


async def premier_qui_repond(serveurs: Sequence[ServiceScores], client_id: str) -> float:
    """Défi 102 🟡 — Interroger plusieurs serveurs à la fois et garder la PREMIÈRE bonne réponse.

    Le modèle est déployé sur 3 serveurs (Paris, Francfort, Dublin). D'habitude ils répondent tous
    vite ; parfois l'un rame ou tombe en panne. Pour l'écran du conseiller, on interroge les 3 en même
    temps et on garde la première réponse : la page est aussi rapide que le plus rapide des serveurs.

    Renvoyez le score de `client_id` (`await serveur.score(client_id)`) :

    - les serveurs sont interrogés EN MÊME TEMPS, et vous renvoyez dès la première bonne réponse ;
    - les serveurs plus lents sont alors ANNULÉS : leur travail ne sert plus à rien ;
    - un serveur qui ÉCHOUE (`ServiceIndisponible`…) n'a pas répondu : attendez les autres ;
    - si TOUS échouent, levez une erreur (l'une des leurs, ou un ExceptionGroup qui les contient) ;
    - une liste de serveurs vide : ValueError.

        >>> paris = ServiceScores(latence=0.3, nom="paris")
        >>> francfort = ServiceScores(latence=0.1, nom="francfort")
        >>> dublin = ServiceScores(latence=0.5, nom="dublin")
        >>> essayer(premier_qui_repond([paris, francfort, dublin], "C004"))
        ⏱️  0,1 s (horloge virtuelle : rien n'a vraiment attendu)
        0.91

    Trois pièges : `gather` (attend les 3, donc le plus LENT) ; essayer les serveurs l'un après
    l'autre ; et prendre la première tâche TERMINÉE sans regarder si elle a réussi.

    Ce que vous apprenez : `asyncio.create_task` pour garder la main sur chaque appel, les attendre
    dans l'ordre où ils finissent, et ne laisser aucune tâche orpheline derrière soi.

    Indice : `taches = [asyncio.create_task(s.score(client_id)) for s in serveurs]`, puis
    `for prochaine in asyncio.as_completed(taches):` donne les tâches dans l'ordre où elles FINISSENT ;
    `return await prochaine` dans un `try` (en cas d'erreur, on passe à la suivante). Un `finally` qui
    appelle `tache.cancel()` sur toutes les tâches annule les perdantes (sans effet sur celles qui
    sont déjà finies).
    """
    raise NotImplementedError


# ============================================================================ Queue et travailleurs


async def scorer_tout(pages, service: ServiceScores, nb_travailleurs: int = 4,
                      max_en_attente: int = 20) -> dict[str, float]:
    """Défi 103 🔴 — Lire les clients page par page, et les scorer avec plusieurs travailleurs.

    Le traitement de nuit score TOUS les clients : 2 millions, lus sur une API par pages de 500
    (`async for page in pages:`, chaque page est une liste d'identifiants). Charger tous les clients
    d'abord ferait exploser la mémoire ; lancer 2 millions d'appels d'un coup ferait tomber le service
    des scores. On monte donc une petite usine : UN lecteur de pages remplit une file d'attente, et
    `nb_travailleurs` travailleurs la vident en appelant `service.score(client_id)`.

    Renvoyez le dictionnaire `{client_id: score}` de tous les clients (dans n'importe quel ordre) :

    - jamais plus de `nb_travailleurs` appels au service en même temps, et `nb_travailleurs` au
      travail dès qu'il y a de quoi faire ;
    - jamais plus de `max_en_attente` clients lus d'avance : quand vous demandez une nouvelle page,
      au plus `max_en_attente` clients déjà lus attendent encore leur travailleur ;
    - la lecture de la page suivante se fait PENDANT que les travailleurs scorent ;
    - chaque client est scoré une seule fois ; des pages vides, ou aucune page, sont possibles ;
    - à la fin, plus rien ne tourne : ni lecteur, ni travailleur ;
    - si le service lève une erreur pour un client (ou si une page ne peut pas être lue), l'erreur
      remonte (seule, ou dans un ExceptionGroup), RAPIDEMENT : plus aucun nouvel appel au service
      après elle, et toutes vos tâches sont arrêtées. Votre fonction ne doit surtout pas rester
      bloquée à attendre un travailleur mort ;
    - `nb_travailleurs` et `max_en_attente` valent au moins 1, sinon ValueError.

        >>> ids = [f"C{n:03d}" for n in range(1, 21)]
        >>> service = ServiceScores({c: 0.5 for c in ids}, latence=0.2)
        >>> pages = Pages(ids, taille=5, service=service)          # 4 pages de 5 clients, 0,1 s par page
        >>> scores = essayer(scorer_tout(pages, service, nb_travailleurs=4, max_en_attente=5))
        ⏱️  1,1 s (horloge virtuelle : rien n'a vraiment attendu)
        >>> len(scores), service.max_en_cours, max(avance for _, avance in pages.demandes)
        (20, 4, 5)

    Le piège : `await file.join()` pour attendre que la file soit vide. Si un travailleur meurt sur une
    erreur, il ne marque jamais sa tâche comme faite, et `join()` attend pour toujours : le traitement
    de nuit est figé au matin, sans aucun message. Le correcteur appelle ça un « interblocage ».

    Ce que vous apprenez : le modèle producteur / consommateurs, la contre-pression (une file bornée
    qui ralentit celui qui la remplit), et l'arrêt propre d'un groupe de tâches, même en cas d'erreur.

    Indice : `file = asyncio.Queue(maxsize=max_en_attente)` : `await file.put(x)` attend quand elle est
    pleine, `await file.get()` attend quand elle est vide. Une coroutine qui lit les pages et `put`
    chaque client, puis un signal de fin par travailleur (`FIN = object()`). Une coroutine travailleur :
    `while (client_id := await file.get()) is not FIN: ...`. Lancez tout dans
    `async with asyncio.TaskGroup() as groupe:` (`groupe.create_task(...)`) : si une tâche échoue, le
    groupe annule les autres et fait remonter l'erreur.
    """
    raise NotImplementedError

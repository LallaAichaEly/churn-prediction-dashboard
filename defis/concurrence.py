"""Outils de la série 16, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Ils sont dans un fichier à part pour une raison précise : quand le correcteur fait lever une
`ServiceIndisponible` à un faux service, votre code doit reconnaître EXACTEMENT la même classe. Deux
classes écrites à l'identique dans deux fichiers sont deux classes différentes pour Python.

Tout est réimporté dans `defis/serie_16.py` : vous n'avez pas besoin d'ouvrir ce fichier.

Il contient :

- quatre FAUX SERVICES distants (le service des scores, l'opérateur SMS, le serveur d'authentification
  et la liste des clients page par page). Leurs réponses mettent du temps à arriver, comme sur un vrai
  réseau : c'est `await asyncio.sleep(latence)` qui simule l'attente. Ils notent tout ce que vous leur
  demandez (combien d'appels, combien EN MÊME TEMPS, lesquels ont été annulés) ;
- une boucle asyncio à HORLOGE VIRTUELLE : `essayer(...)` exécute votre coroutine et affiche la durée
  qu'elle aurait prise, sans attendre pour de vrai. Quand toutes les tâches dorment, l'horloge saute
  directement au prochain réveil. 500 SMS de 0,2 s s'« envoient » ainsi en une fraction de seconde.

    >>> service = ServiceScores(latence=0.2)
    >>> essayer(service.score("C001"))
    ⏱️  0,2 s (horloge virtuelle : rien n'a vraiment attendu)
    0.82
"""

from __future__ import annotations

import asyncio
import itertools
import selectors
from typing import Awaitable, Iterable, Mapping, Sequence

# Les scores de churn que renvoie le faux service (C005 vaut exactement 0.0 : un score comme les autres).
SCORES_EXEMPLE = {
    "C001": 0.82, "C002": 0.15, "C003": 0.47, "C004": 0.91,
    "C005": 0.0, "C006": 0.63, "C007": 0.38, "C008": 0.74,
}


# ============================================================================ Les erreurs des services


class ServiceIndisponible(Exception):
    """Le service distant a répondu par une erreur (HTTP 503) : il est en panne ou en maintenance."""


class Surcharge(Exception):
    """L'opérateur SMS refuse l'envoi : trop de SMS en cours en même temps (HTTP 429 « Too Many Requests »)."""


class NumeroInvalide(Exception):
    """L'opérateur SMS refuse l'envoi : le numéro de téléphone du client n'existe pas."""


def _maintenant() -> float:
    """L'heure de la boucle asyncio en cours (virtuelle dans `essayer` et chez le correcteur)."""
    return asyncio.get_running_loop().time()


# ============================================================================ Les faux services


class ServiceScores:
    """Le service distant qui calcule le score de churn d'un client : `await service.score("C001")`.

    Chaque appel prend `latence` secondes (ou `latences[client_id]` si on la précise pour ce client).
    Un client absent de `scores` lève `KeyError` ; un client présent dans `erreurs` lève l'erreur indiquée
    (une `ServiceIndisponible` par exemple), APRÈS la latence, comme une vraie réponse d'erreur.

    Ce que le service note (le correcteur s'en sert, vous pouvez le regarder en essayant) :
    `appels` (client, instant du début), `reponses` (client, instant, "ok" | "erreur" | "annule"),
    `en_cours` (appels pas encore terminés) et `max_en_cours` (le plus grand nombre d'appels simultanés).

        >>> service = ServiceScores(latence=0.2)
        >>> essayer(service.score("C005"))
        ⏱️  0,2 s (horloge virtuelle : rien n'a vraiment attendu)
        0.0
    """

    def __init__(self, scores: Mapping[str, float] | None = None, latence: float = 0.2, *,
                 latences: Mapping[str, float] | None = None, erreurs: Mapping[str, BaseException] | None = None,
                 nom: str = "scores") -> None:
        self.scores = dict(SCORES_EXEMPLE if scores is None else scores)
        self.latence = latence
        self.latences = dict(latences or {})
        self.erreurs = dict(erreurs or {})
        self.nom = nom
        self.appels: list[tuple[str, float]] = []
        self.reponses: list[tuple[str, float, str]] = []
        self.en_cours = 0
        self.max_en_cours = 0

    def __repr__(self) -> str:
        return f"ServiceScores({self.nom!r}, {len(self.scores)} clients, latence {self.latence} s)"

    @property
    def annules(self) -> list[str]:
        """Les clients dont l'appel a été annulé avant la réponse."""
        return [client for client, _, issue in self.reponses if issue == "annule"]

    async def score(self, client_id: str) -> float:
        self.appels.append((client_id, _maintenant()))
        self.en_cours += 1
        self.max_en_cours = max(self.max_en_cours, self.en_cours)
        try:
            await asyncio.sleep(self.latences.get(client_id, self.latence))
        except asyncio.CancelledError:
            self.reponses.append((client_id, _maintenant(), "annule"))
            raise
        finally:
            self.en_cours -= 1
        if client_id in self.erreurs:
            self.reponses.append((client_id, _maintenant(), "erreur"))
            raise self.erreurs[client_id]
        if client_id not in self.scores:
            self.reponses.append((client_id, _maintenant(), "erreur"))
            raise KeyError(f"client inconnu : {client_id}")
        self.reponses.append((client_id, _maintenant(), "ok"))
        return self.scores[client_id]


class OperateurSMS:
    """L'opérateur qui envoie les SMS de relance : `await operateur.envoyer("C001", "Votre offre…")`.

    Il renvoie l'identifiant du SMS (« sms-0001 »…). Chaque envoi prend `latence` secondes (ou
    `latences[client_id]`). Deux refus possibles :

    - `Surcharge`, IMMÉDIATEMENT, si `max_simultanes` envois sont déjà en cours (HTTP 429) ;
    - `NumeroInvalide`, après la latence, pour un client de `invalides`.

    Il note `tentatives`, `envoyes` (client, texte, identifiant), `refus` (clients refusés pour
    surcharge), `erreurs` (les exceptions levées), `en_cours` et `max_en_cours`.
    """

    def __init__(self, max_simultanes: int = 10, latence: float = 0.2, *, latences: Mapping[str, float] | None = None,
                 invalides: Iterable[str] = ()) -> None:
        self.max_simultanes = max_simultanes
        self.latence = latence
        self.latences = dict(latences or {})
        self.invalides = set(invalides)
        self.tentatives: list[tuple[str, float]] = []
        self.envoyes: list[tuple[str, str, str]] = []
        self.refus: list[str] = []
        self.erreurs: list[BaseException] = []
        self.en_cours = 0
        self.max_en_cours = 0
        self._numeros = itertools.count(1)

    def __repr__(self) -> str:
        return f"OperateurSMS({self.max_simultanes} envois simultanés au maximum, latence {self.latence} s)"

    async def envoyer(self, client_id: str, texte: str) -> str:
        self.tentatives.append((client_id, _maintenant()))
        if self.en_cours >= self.max_simultanes:
            self.refus.append(client_id)
            erreur = Surcharge(f"{self.en_cours} SMS déjà en cours : {self.max_simultanes} au maximum en même temps")
            self.erreurs.append(erreur)
            raise erreur
        self.en_cours += 1
        self.max_en_cours = max(self.max_en_cours, self.en_cours)
        try:
            await asyncio.sleep(self.latences.get(client_id, self.latence))
        finally:
            self.en_cours -= 1
        if client_id in self.invalides:
            erreur = NumeroInvalide(f"numéro de téléphone invalide pour {client_id}")
            self.erreurs.append(erreur)
            raise erreur
        identifiant = f"sms-{next(self._numeros):04d}"
        self.envoyes.append((client_id, texte, identifiant))
        return identifiant


class ServeurAuth:
    """Le serveur d'authentification : `jeton, duree_de_vie = await serveur.nouveau_jeton()`.

    Chaque demande prend `latence` secondes et renvoie un nouveau jeton (« jeton-1 », « jeton-2 »…),
    valable `duree_de_vie` secondes. Les `pannes` prochaines demandes échouent (`ServiceIndisponible`).
    Il note l'instant de chaque demande dans `demandes`.
    """

    def __init__(self, duree_de_vie: float = 3600.0, latence: float = 0.5, *, pannes: int = 0) -> None:
        self.duree_de_vie = duree_de_vie
        self.latence = latence
        self.pannes = pannes
        self.demandes: list[float] = []
        self.en_cours = 0
        self.max_en_cours = 0

    def __repr__(self) -> str:
        return f"ServeurAuth(jetons valables {self.duree_de_vie} s, latence {self.latence} s)"

    async def nouveau_jeton(self) -> tuple[str, float]:
        self.demandes.append(_maintenant())
        numero = len(self.demandes)
        self.en_cours += 1
        self.max_en_cours = max(self.max_en_cours, self.en_cours)
        try:
            await asyncio.sleep(self.latence)
        finally:
            self.en_cours -= 1
        if self.pannes > 0:
            self.pannes -= 1
            raise ServiceIndisponible("serveur d'authentification indisponible (503)")
        return f"jeton-{numero}", self.duree_de_vie


class Pages:
    """La liste des clients, lue page par page sur une API lente : `async for page in pages: ...`.

    Chaque page est une liste d'identifiants, et met `latence` secondes à arriver. Comme un flux
    réseau, un objet `Pages` ne se lit qu'UNE fois, et par une seule boucle à la fois.

    Si on lui donne le `service` de scores, il note à chaque demande de page combien de clients ont
    été lus mais pas encore envoyés au service (`demandes` : instant, clients lus d'avance).

        >>> pages = Pages(["C001", "C002", "C003", "C004", "C005"], taille=2)
        >>> async def tout_lire():
        ...     return [page async for page in pages]
        >>> essayer(tout_lire())
        ⏱️  0,4 s (horloge virtuelle : rien n'a vraiment attendu)
        [['C001', 'C002'], ['C003', 'C004'], ['C005']]
    """

    def __init__(self, ids: Sequence[str] = (), taille: int = 5, latence: float = 0.1, *,
                 service: ServiceScores | None = None, panne_a_la_page: int | None = None) -> None:
        self.pages = [list(ids[i:i + taille]) for i in range(0, len(ids), taille)]
        self.latence = latence
        self.service = service
        self.panne_a_la_page = panne_a_la_page
        self.demandes: list[tuple[float, int]] = []
        self.lus = 0
        self._suivante = 0
        self._en_lecture = False

    def __repr__(self) -> str:
        return f"Pages({sum(map(len, self.pages))} clients en {len(self.pages)} pages, latence {self.latence} s)"

    def __aiter__(self) -> "Pages":
        return self

    async def __anext__(self) -> list[str]:
        if self._en_lecture:
            raise RuntimeError("deux lectures de page en même temps : une seule boucle `async for` doit lire les pages")
        envoyes = len(self.service.appels) if self.service is not None else 0
        self.demandes.append((_maintenant(), self.lus - envoyes))
        self._en_lecture = True
        try:
            await asyncio.sleep(self.latence)
        finally:
            self._en_lecture = False
        if self._suivante == self.panne_a_la_page:
            raise ServiceIndisponible(f"la page {self._suivante + 1} n'a pas pu être lue (503)")
        if self._suivante >= len(self.pages):
            raise StopAsyncIteration
        page = list(self.pages[self._suivante])
        self._suivante += 1
        self.lus += len(page)
        return page


def service_exemple(latence: float = 0.2) -> ServiceScores:
    """Le service des scores des 8 clients de `SCORES_EXEMPLE`, qui répond en `latence` secondes."""
    return ServiceScores(SCORES_EXEMPLE, latence)


# ============================================================================ La boucle à horloge virtuelle


class Interblocage(RuntimeError):
    """Plus rien ne peut avancer : toutes les tâches attendent, et rien n'est prévu pour les réveiller.

    Exemples : `await file.get()` sur une file vide que plus personne ne remplit, `await file.join()`
    alors qu'un travailleur est mort sur une erreur, un verrou jamais rendu. Avec une vraie boucle, le
    programme resterait figé pour toujours ; la boucle virtuelle s'en aperçoit et le dit.
    """


class TempsDepasse(RuntimeError):
    """La coroutine a dépassé la durée maximale (virtuelle) autorisée : elle ne s'arrête sans doute jamais."""


class _SelecteurVirtuel(selectors.BaseSelector):
    """Le « sélecteur » attend les événements réseau pour la boucle. Le nôtre n'attend jamais : quand la
    boucle veut dormir `timeout` secondes, il avance l'horloge virtuelle d'autant."""

    def __init__(self, boucle: "BoucleVirtuelle") -> None:
        self._vrai = selectors.DefaultSelector()
        self._boucle = boucle

    def register(self, fileobj, events, data=None):
        return self._vrai.register(fileobj, events, data)

    def unregister(self, fileobj):
        return self._vrai.unregister(fileobj)

    def modify(self, fileobj, events, data=None):
        return self._vrai.modify(fileobj, events, data)

    def get_map(self):
        return self._vrai.get_map()

    def close(self) -> None:
        self._vrai.close()

    def select(self, timeout=None):
        boucle = self._boucle
        evenements = self._vrai.select(0)  # un vrai événement en attente (réveil depuis un autre thread…) ?
        if evenements:
            return evenements
        boucle.tours += 1
        if boucle.tours > boucle.tours_max:
            raise TempsDepasse(f"plus de {boucle.tours_max} tours de boucle : votre code tourne sans fin "
                               "(une boucle `while` avec `await asyncio.sleep(0)` qui n'attend rien ?)")
        if timeout is None:
            if boucle._default_executor is not None:  # un thread travaille peut-être (asyncio.to_thread) : on l'attend
                evenements = self._vrai.select(5.0)
                if evenements:
                    return evenements
            raise Interblocage("toutes les tâches attendent, et rien n'est prévu pour les réveiller")
        if timeout > 0:
            boucle.maintenant += timeout
            if boucle.maintenant > boucle.limite:
                raise TempsDepasse(f"plus de {boucle.limite:g} s (virtuelles) se sont écoulées : "
                                   "votre code ne s'arrête sans doute jamais")
        return []


class BoucleVirtuelle(asyncio.SelectorEventLoop):
    """Une vraie boucle asyncio, dont l'horloge (`loop.time()`) est virtuelle : elle part de 0 et saute
    directement au prochain réveil prévu quand toutes les tâches attendent.

    Tout ce qui repose sur cette horloge marche comme d'habitude : `asyncio.sleep`, `asyncio.wait_for`,
    `asyncio.timeout`, `asyncio.wait(..., timeout=...)`. Seul `time.sleep` (qui bloque pour de vrai) ne
    la fait pas avancer : dans une coroutine, on ne l'utilise jamais.
    """

    def __init__(self, limite: float = 86_400.0, tours_max: int = 200_000) -> None:
        self.maintenant = 0.0
        self.tours = 0
        self.limite = limite
        self.tours_max = tours_max
        super().__init__(_SelecteurVirtuel(self))

    def time(self) -> float:
        return self.maintenant


def _fermer(boucle: BoucleVirtuelle) -> None:
    """Annule les tâches restées en vie, puis ferme la boucle (même après un interblocage)."""
    try:
        taches = [t for t in asyncio.all_tasks(boucle) if not t.done()]
        for tache in taches:
            tache.cancel()
        if taches:
            boucle.run_until_complete(asyncio.gather(*taches, return_exceptions=True))
        boucle.run_until_complete(boucle.shutdown_asyncgens())
    except (Interblocage, TempsDepasse):
        pass  # une tâche refuse de s'arrêter : on ferme quand même
    finally:
        boucle.close()


def executer(principal: Awaitable, *, limite: float = 86_400.0, silencieux: bool = False) -> tuple[object, float]:
    """Exécute `principal` (une coroutine) sur une boucle à horloge virtuelle neuve.

    Renvoie `(resultat, duree)`, la durée étant celle de l'horloge virtuelle. Lève `Interblocage` si tout
    s'arrête, `TempsDepasse` au-delà de `limite` secondes virtuelles. Les tâches encore en vie à la fin
    sont annulées. `silencieux=True` : les erreurs des tâches oubliées ne sont pas affichées.
    """
    boucle = BoucleVirtuelle(limite)
    if silencieux:
        boucle.set_exception_handler(lambda _boucle, _contexte: None)
    try:
        resultat = boucle.run_until_complete(principal)
        return resultat, boucle.maintenant
    finally:
        _fermer(boucle)


def _secondes(duree: float) -> str:
    return f"{duree:.3f}".rstrip("0").rstrip(".").replace(".", ",")


def essayer(principal: Awaitable):
    """Exécute votre coroutine sur l'horloge virtuelle, affiche la durée qu'elle aurait prise et renvoie son résultat.

        >>> essayer(scores_en_parallele(service_exemple(), ["C001", "C002", "C003"]))   # défi 98
        ⏱️  0,2 s (horloge virtuelle : rien n'a vraiment attendu)
        {'C001': 0.82, 'C002': 0.15, 'C003': 0.47}

    Pour attendre pour de vrai, comme dans un programme : `asyncio.run(scores_en_parallele(...))`.

    Si des tâches tournent encore quand votre coroutine a fini (un appel lent jamais annulé…),
    `essayer` les annule et vous prévient : un vrai programme les aurait laissées travailler pour rien.
    """
    duree = [0.0]
    oubliees: list[asyncio.Task] = []

    async def suivre():
        boucle = asyncio.get_running_loop()
        resultat = await principal
        duree[0] = boucle.time()
        for _ in range(3):  # laisse aux annulations demandées le temps d'arriver
            await asyncio.sleep(0)
        moi = asyncio.current_task()
        oubliees.extend(t for t in asyncio.all_tasks() if t is not moi and not t.done())
        return resultat

    try:
        resultat, _ = executer(suivre())
    except Interblocage:
        print("🧊 Interblocage : toutes les tâches attendent, et rien n'est prévu pour les réveiller.")
        raise
    print(f"⏱️  {_secondes(duree[0])} s (horloge virtuelle : rien n'a vraiment attendu)")
    if oubliees:
        print(f"⚠️  {len(oubliees)} tâche(s) tournaient encore à la fin : essayer les a annulées, mais votre code "
              "les avait oubliées.")
    return resultat

"""Vérifications automatiques de la série 18 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur ne chronomètre presque jamais pour de vrai : un temps dépend de la machine, et un
correcteur qui dit « trop lent » sur un vieil ordinateur et « bravo » sur un neuf ne vaut rien. Il
préfère COMPTER, ce qui donne le même chiffre partout :

- au défi 110, il vous passe une fausse horloge qu'il avance lui-même pendant chaque appel ;
- aux défis 112 et 115, il vous passe des dates, des clients et des listes « espions » qui comptent
  chaque comparaison et chaque lecture, et il ARRÊTE votre fonction quand elle dépasse le budget
  d'une solution efficace (une boucle dans une boucle le dépasse au bout de quelques clients) ;
- au défi 114, il mesure le pic de mémoire avec `tracemalloc`.

Au défi 111, il profile pour de vrai des programmes dont la fonction lente l'est 10 fois plus que
toutes les autres réunies : le résultat ne dépend pas de la machine.
"""

from __future__ import annotations

import csv
import functools
import gc
import inspect
import math
import random
import shutil
import sys
import tempfile
import time
import tracemalloc
import warnings
from datetime import date
from pathlib import Path

from .performance import HorlogeFactice, TARIFS_EXEMPLE, clients_a_rappeler_lent, ecrire_factures
from .verif import Defi, Serie


def _nombre(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _court(valeur, n: int = 160) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


def _decrire(exc: BaseException) -> str:
    return f"{type(exc).__name__}({str(exc)!r})" if str(exc) else type(exc).__name__


def _leve(appel, attendue, quoi: str, conseil: str = "") -> BaseException:
    """Exige que `appel()` lève `attendue` ; renvoie l'exception. NotImplementedError remonte (« à faire »)."""
    try:
        resultat = appel()
    except NotImplementedError:
        raise
    except attendue as exc:
        return exc
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendue.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendue.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


class _BudgetDepasse(BaseException):
    """Levée par les objets espions quand le budget est dépassé.

    Elle hérite de BaseException (comme KeyboardInterrupt) pour qu'un `except Exception` dans votre code
    ne puisse pas l'attraper : le correcteur doit pouvoir arrêter une fonction trop lente.
    """


class _Compteur:
    """Compte des opérations par catégorie et lève _BudgetDepasse au-delà du budget total."""

    def __init__(self, budget: float = math.inf) -> None:
        self.budget = budget
        self.total = 0
        self.par_categorie: dict[str, int] = {}

    def __call__(self, categorie: str, n: int = 1) -> None:
        self.total += n
        self.par_categorie[categorie] = self.par_categorie.get(categorie, 0) + n
        if self.total > self.budget:
            raise _BudgetDepasse


# ============================================================================ Défi 110 : chronométrer


def verifier_chronometrer(chronometrer) -> None:
    horloge = HorlogeFactice()
    durees = [0.30, 0.10, 0.12, 0.11, 0.50]
    appels = []

    def fonction():
        horloge.avancer(durees[len(appels)] if len(appels) < len(durees) else 1.0)
        appels.append(1)

    resultat = chronometrer(fonction, repetitions=5, horloge=horloge)
    if resultat is None:
        raise AssertionError("chronometrer renvoie None : avez-vous oublié le `return` ?")
    if len(appels) != 5:
        raise AssertionError(f"Avec repetitions=5, fonction() a été appelée {len(appels)} fois au lieu de 5.")
    if isinstance(resultat, bool) or not isinstance(resultat, (int, float)):
        raise AssertionError(f"chronometrer doit renvoyer une durée en secondes (un float), pas {_court(resultat)}.")
    if not math.isclose(resultat, 0.10, abs_tol=1e-9):
        situation = ("Les 5 appels ont duré 0.30, 0.10, 0.12, 0.11 et 0.50 seconde (d'après l'horloge passée en "
                     f"paramètre), et vous renvoyez {resultat!r} au lieu de 0.1.")
        diagnostics = (
            (0.226, "C'est la MOYENNE. Une moyenne mélange le coût du code et les ralentissements venus d'ailleurs "
                    "(le premier et le dernier appel ont été dérangés) : gardez le minimum, min(durees)."),
            (1.13, "C'est la durée TOTALE. Mesurez chaque appel séparément et gardez le plus court."),
            (0.30, "C'est la PREMIÈRE mesure seulement : une mesure isolée peut tomber pendant une perturbation. "
                   "Gardez la plus courte des 5."),
            (0.50, "C'est la plus LONGUE (ou la dernière). Gardez la plus courte : min(durees)."),
        )
        for valeur, conseil in diagnostics:
            if math.isclose(resultat, valeur, abs_tol=1e-9):
                raise AssertionError(f"{situation} {conseil}")
        if abs(resultat) < 0.01:
            raise AssertionError(f"{situation} Vous semblez lire une vraie horloge (time.perf_counter() ou "
                                 "time.time()) au lieu du paramètre `horloge` : appelez horloge(), c'est lui que le "
                                 "correcteur avance.")
        raise AssertionError(situation)

    for repetitions in (1, 3, 7):
        horloge, appels = HorlogeFactice(), []

        def fonction():
            appels.append(1)
            horloge.avancer(0.2 + 0.1 * (len(appels) % 3))

        resultat = chronometrer(fonction, repetitions=repetitions, horloge=horloge)
        if len(appels) != repetitions:
            raise AssertionError(f"Avec repetitions={repetitions}, fonction() a été appelée {len(appels)} fois.")
        attendu = min(0.2 + 0.1 * (i % 3) for i in range(1, repetitions + 1))
        if not math.isclose(resultat, attendu, abs_tol=1e-9):
            raise AssertionError(f"Avec repetitions={repetitions}, chronometrer renvoie {resultat!r} au lieu de "
                                 f"{attendu!r} (la plus courte des durées mesurées).")

    for mauvais in (0, -2):
        appels = []
        exc = _leve(lambda: chronometrer(lambda: appels.append(1), repetitions=mauvais, horloge=HorlogeFactice()),
                    ValueError, f"chronometrer(fonction, repetitions={mauvais})",
                    "Vérifiez `repetitions` avant la boucle.")
        if "repetitions" not in str(exc):
            raise AssertionError(f"repetitions={mauvais} lève bien ValueError, mais le message doit nommer "
                                 f"`repetitions` : obtenu {str(exc)!r}. (« min() arg is an empty sequence » ne dit pas "
                                 "à l'appelant ce qu'il a mal fait.)")
        if appels:
            raise AssertionError(f"Avec repetitions={mauvais}, fonction() a quand même été appelée.")

    class Panne(Exception):
        pass

    appels = []

    def en_panne():
        appels.append(1)
        if len(appels) == 2:
            raise Panne("le service de scores ne répond plus")

    _leve(lambda: chronometrer(en_panne, repetitions=5, horloge=HorlogeFactice()), Panne,
          "Quand fonction() lève une exception au 2e appel, chronometrer",
          "Ne l'attrapez pas : un chronomètre qui avale les erreurs mesure la vitesse d'un plantage.")

    parametres = inspect.signature(chronometrer).parameters
    for nom, defaut in (("repetitions", 5), ("horloge", time.perf_counter)):
        if nom not in parametres:
            raise AssertionError(f"chronometrer doit avoir un paramètre `{nom}`.")
    defaut = parametres["horloge"].default
    if defaut in (time.time, time.monotonic, time.process_time):
        raise AssertionError(f"Le paramètre horloge vaut time.{defaut.__name__} par défaut : prenez time.perf_counter. "
                             "time.time() donne l'heure du système, qui peut reculer quand l'horloge se synchronise, "
                             "et n'est précise qu'à 16 ms sous Windows ; perf_counter est faite pour chronométrer.")
    if parametres["repetitions"].default != 5:
        raise AssertionError(f"repetitions doit valoir 5 par défaut (obtenu : {parametres['repetitions'].default!r}).")
    reel = chronometrer(lambda: sum(range(1000)))
    if not (isinstance(reel, float) and 0 <= reel < 1):
        raise AssertionError(f"Avec l'horloge par défaut, chronometrer(lambda: sum(range(1000))) renvoie {_court(reel)} "
                             "au lieu d'une petite durée en secondes.")


# ============================================================================ Défi 111 : profiler
# Les programmes à profiler. Pas de liste en compréhension ici : avant Python 3.12, chacune apparaît dans
# le profil comme une fonction à part (<listcomp>).


def nettoyer_ligne(valeur: int) -> int:
    return valeur


def lire_clients() -> int:
    total = 0
    for i in range(5_000):
        total += nettoyer_ligne(i)     # 5 000 appels : c'est la fonction la plus APPELÉE, pas la plus lente
    return total


def calculer_risques(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += (i % 7) * 0.5
    return total


_LANCEMENTS: list[str] = []   # chaque programme note ici qu'il a été lancé


def programme_rapport() -> float:
    _LANCEMENTS.append("programme_rapport")
    lire_clients()
    return calculer_risques(500_000)


def mettre_en_forme(n: int) -> int:
    total = 0
    for i in range(n):
        total += i & 3
    return total


def preparer_rapport() -> int:
    return mettre_en_forme(600_000)     # ne fait que déléguer : gros cumtime, tottime presque nul


def envoyer_rapport(rapport: int) -> int:
    total = 0
    for i in range(20_000):
        total ^= i
    return total + rapport


def programme_envoi() -> int:
    _LANCEMENTS.append("programme_envoi")
    return envoyer_rapport(preparer_rapport())


def programme_en_panne() -> None:
    calculer_risques(10_000)
    raise RuntimeError("panne au milieu du rapport")


def _profileurs_actifs() -> bool:
    if sys.getprofile() is not None:
        return True
    monitoring = getattr(sys, "monitoring", None)       # Python 3.12 et suivants
    return monitoring is not None and monitoring.get_tool(monitoring.PROFILER_ID) is not None


def _arreter_les_profileurs() -> None:
    """Arrête un profileur oublié, pour que la suite (et les autres défis) ne soit pas ralentie ou bloquée."""
    import cProfile

    for objet in gc.get_objects():
        if isinstance(objet, cProfile.Profile):
            try:
                objet.disable()
            except Exception:
                pass
    sys.setprofile(None)


def verifier_point_chaud(point_chaud) -> None:
    for programme, attendu, cumule, plus_appelee in (
            (programme_rapport, "calculer_risques", "programme_rapport", "nettoyer_ligne"),
            (programme_envoi, "mettre_en_forme", "programme_envoi", None)):
        _LANCEMENTS.clear()
        resultat = point_chaud(programme)
        lancements = list(_LANCEMENTS)
        if resultat is None:
            raise AssertionError("point_chaud renvoie None : avez-vous oublié le `return` ?")
        if len(lancements) != 1:
            raise AssertionError(f"point_chaud a lancé le programme {len(lancements)} fois : une seule suffit "
                                 "(sur un vrai programme, chaque lancement peut prendre des minutes).")
        if isinstance(resultat, tuple) or (isinstance(resultat, str) and "(" in resultat and attendu in resultat):
            raise AssertionError(f"point_chaud renvoie {_court(resultat)} : renvoyez seulement le NOM de la fonction "
                                 f"(« {attendu} »), c'est-à-dire l'élément 2 de la clé (fichier, ligne, nom).")
        if not isinstance(resultat, str):
            raise AssertionError(f"point_chaud doit renvoyer un texte (le nom d'une fonction), pas {_court(resultat)}.")
        if resultat == attendu:
            continue
        description = (f"Sur le programme `{programme.__name__}`, point_chaud renvoie « {resultat} » au lieu de "
                       f"« {attendu} », la fonction qui passe le plus de temps dans son propre code.")
        if resultat in (cumule, "preparer_rapport") or "<" in resultat:
            raise AssertionError(f"{description} « {resultat} » a le plus grand temps CUMULÉ (cumtime, élément 3) : "
                                 "elle ne fait qu'appeler les autres. Prenez le tottime (élément 2 des valeurs).")
        if plus_appelee and resultat == plus_appelee:
            raise AssertionError(f"{description} « {resultat} » est la plus APPELÉE (5 000 fois), mais chaque appel ne "
                                 "coûte presque rien. Classez par tottime (élément 2 des valeurs), pas par ncalls.")
        raise AssertionError(description)

    _arreter_les_profileurs()
    try:
        _leve(lambda: point_chaud(programme_en_panne), RuntimeError,
              "Quand le programme lève RuntimeError, point_chaud",
              "L'erreur du programme doit remonter : on ne profile pas un programme qui plante.")
        if _profileurs_actifs():
            raise AssertionError("Quand le programme lève une exception, elle remonte bien, mais votre profileur est "
                                 "resté ACTIF : tout le reste du programme est ralenti (et depuis Python 3.12, aucun autre "
                                 "profileur ne peut démarrer). Utilisez profileur.runcall(programme), ou "
                                 "profileur.enable() puis try: … finally: profileur.disable().")
    finally:
        _arreter_les_profileurs()


# ============================================================================ Défi 112 : bisect


class _Jour(date):
    """Une date qui compte ses comparaisons (le correcteur la donne à votre fonction)."""

    compteur: _Compteur | None = None

    def _compter(self) -> None:
        if _Jour.compteur is not None:
            _Jour.compteur("comparaisons")

    def __lt__(self, autre):
        self._compter()
        return date.__lt__(self, autre)

    def __le__(self, autre):
        self._compter()
        return date.__le__(self, autre)

    def __gt__(self, autre):
        self._compter()
        return date.__gt__(self, autre)

    def __ge__(self, autre):
        self._compter()
        return date.__ge__(self, autre)

    def __eq__(self, autre):
        self._compter()
        return date.__eq__(self, autre)

    def __ne__(self, autre):
        self._compter()
        return date.__ne__(self, autre)

    __hash__ = date.__hash__


def _prix_reference(changements, dates):
    import bisect

    historique = sorted(changements, key=lambda c: c[0].toordinal())
    jours = [j.toordinal() for j, _ in historique]
    return [historique[bisect.bisect_right(jours, d.toordinal()) - 1][1] for d in dates]


def _appeler_prix(prix_en_vigueur, changements, dates, quoi: str):
    copie = list(changements)
    try:
        resultat = prix_en_vigueur(changements, dates)
    except NotImplementedError:
        raise
    except ValueError as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}, alors que chaque date a un prix connu.") from None
    if resultat is None:
        raise AssertionError("prix_en_vigueur renvoie None : avez-vous oublié le `return` ?")
    if changements != copie:
        raise AssertionError(f"{quoi} a MODIFIÉ la liste `changements` de l'appelant (elle est maintenant triée, ou "
                             "raccourcie). `liste.sort()` trie la liste reçue ; `sorted(liste)` renvoie une copie triée.")
    return list(resultat)


def verifier_prix_en_vigueur(prix_en_vigueur) -> None:
    exemple = [date(2024, 3, 15), date(2024, 7, 1), date(2026, 1, 1)]
    resultat = _appeler_prix(prix_en_vigueur, list(TARIFS_EXEMPLE), exemple, "Sur TARIFS_EXEMPLE, prix_en_vigueur")
    if resultat != [29.9, 32.9, 34.9]:
        if resultat == [29.9, 29.9, 34.9]:
            raise AssertionError("Le 1er juillet 2024, jour où le tarif passe à 32.9, vous donnez l'ANCIEN prix (29.9). "
                                 "Le prix s'applique à partir de sa date, incluse : bisect_left s'arrête AVANT une date "
                                 "égale, bisect_right APRÈS. Il vous faut bisect_right.")
        raise AssertionError(f"Sur TARIFS_EXEMPLE et les dates {exemple}, prix_en_vigueur renvoie {_court(resultat)} au lieu "
                             "de [29.9, 32.9, 34.9]. L'historique n'est pas trié : triez-le d'abord.")

    premier = [date(2024, 1, 1)]
    resultat = _appeler_prix(prix_en_vigueur, list(TARIFS_EXEMPLE), premier, "Le jour du tout premier tarif, prix_en_vigueur")
    if resultat != [29.9]:
        raise AssertionError(f"Le 1er janvier 2024, jour du premier tarif (29.9), vous donnez {_court(resultat)} : le prix "
                             "s'applique à partir de sa date, incluse (bisect_right).")

    avant = date(2023, 12, 31)
    exc = _leve(lambda: prix_en_vigueur(list(TARIFS_EXEMPLE), [date(2024, 5, 1), avant]), ValueError,
                "Pour le 31 décembre 2023, la veille du premier tarif, prix_en_vigueur",
                "Si la position trouvée vaut 0, prix[0 - 1] ne plante pas : l'indice -1 est le DERNIER élément, et le "
                "client paierait le tarif de 2025 en 2023. Testez la position et levez ValueError.")
    if "2023-12-31" not in str(exc):
        raise AssertionError(f"La veille du premier tarif lève bien ValueError, mais le message doit contenir la date "
                             f"fautive (« 2023-12-31 ») : obtenu {str(exc)!r}. Sur 200 000 factures, sans la date, "
                             "on cherche longtemps.")

    doublon = [(date(2024, 1, 1), 10.0), (date(2024, 6, 1), 12.0), (date(2024, 6, 1), 11.0)]
    _leve(lambda: prix_en_vigueur(doublon, [date(2024, 7, 1)]), ValueError,
          "Avec deux tarifs différents le 1er juin 2024, prix_en_vigueur",
          "L'historique est faux : on ne choisit pas un des deux au hasard. Après le tri, deux dates égales sont voisines.")
    _leve(lambda: prix_en_vigueur([], [date(2024, 7, 1)]), ValueError, "Avec un historique vide, prix_en_vigueur")
    vide = _appeler_prix(prix_en_vigueur, list(TARIFS_EXEMPLE), [], "Sans aucune date, prix_en_vigueur")
    if vide != []:
        raise AssertionError(f"Sans aucune date, prix_en_vigueur doit renvoyer [] (obtenu : {_court(vide)}).")

    rng = random.Random(18)
    for _ in range(60):
        k = rng.randint(1, 8)
        jours = rng.sample(range(737_000, 737_400), k)
        changements = [(date.fromordinal(j), round(rng.uniform(10, 60), 2)) for j in jours]
        debut = min(jours)
        dates = [date.fromordinal(rng.randint(debut, 737_450)) for _ in range(rng.randint(0, 12))]
        dates += [date.fromordinal(rng.choice(jours)) for _ in range(2)]   # des jours de changement exactement
        rng.shuffle(dates)
        attendu = _prix_reference(changements, dates)
        resultat = _appeler_prix(prix_en_vigueur, changements, dates, "prix_en_vigueur")
        if resultat != attendu:
            raise AssertionError(f"Avec changements={_court(changements, 300)} et dates={_court(dates, 300)}, "
                                 f"prix_en_vigueur renvoie {_court(resultat)} au lieu de {_court(attendu)}. Les résultats "
                                 "sont-ils dans l'ordre de `dates` ?")

    # Le budget : 2 000 changements, 20 000 dates. Trier, puis bisect, fait environ 300 000 comparaisons ;
    # parcourir l'historique pour chaque date en fait 20 millions.
    k, n = 2_000, 20_000
    jours = rng.sample(range(730_000, 760_000), k)
    changements = [(_Jour.fromordinal(j), float(i % 50 + 10)) for i, j in enumerate(jours)]
    debut = min(jours)
    dates = [_Jour.fromordinal(rng.randint(debut, 761_000)) for _ in range(n)]
    attendu = _prix_reference(changements, dates)
    _Jour.compteur = _Compteur(budget=1_500_000)
    depart = time.perf_counter()
    try:
        resultat = prix_en_vigueur(changements, dates)
    except _BudgetDepasse:
        raise AssertionError(f"Sur {_nombre(k)} changements et {_nombre(n)} dates, votre fonction a dépassé 1 500 000 "
                             "comparaisons de dates (trier puis bisect en fait environ 300 000) : le correcteur l'a "
                             "arrêtée. Parcourez-vous l'historique pour chaque date ? Triez-le UNE fois, puis "
                             "bisect.bisect_right pour chaque date.") from None
    finally:
        _Jour.compteur = None
    duree = time.perf_counter() - depart
    if list(resultat) != attendu:
        raise AssertionError(f"Sur {k} changements et {n} dates, les prix renvoyés sont faux.")
    if duree > 3:
        raise AssertionError(f"Sur {k} changements et {n} dates, votre fonction met {duree:.1f} s (une solution avec "
                             "bisect met moins d'une demi-seconde, comptage des comparaisons compris). Parcourez-vous "
                             "l'historique pour chaque date ?")


# ============================================================================ Défi 113 : cache borné et périmé


def _fabriquer(cache_lru, taille_max, duree, horloge, valeur=lambda *args: ("résultat", *args)):
    calculs = []

    def fonction(*args):
        """Calcule un score (documentation d'origine)."""
        calculs.append(args)
        return valeur(*args)

    return cache_lru(taille_max, duree, horloge=horloge)(fonction), calculs


def verifier_cache_lru(cache_lru) -> None:
    # ------------------------------------------------------------------ l'exemple de l'énoncé
    horloge = HorlogeFactice()
    decorateur = cache_lru(2, 60, horloge=horloge)
    if decorateur is None or not callable(decorateur):
        raise AssertionError("cache_lru(2, 60) doit renvoyer un DÉCORATEUR (une fonction qui reçoit la fonction à "
                             "décorer) : avez-vous oublié `return decorateur` ?")
    score, calculs = _fabriquer(cache_lru, 2, 60, horloge, valeur=lambda *a: 0.5)
    if score is None or not callable(score):
        raise AssertionError("Le décorateur doit renvoyer la fonction enveloppée : avez-vous oublié `return enveloppe` ?")
    resultats = [score(c) for c in ("C1", "C1", "C2", "C1", "C3")]
    if resultats != [0.5] * 5:
        raise AssertionError(f"La fonction décorée renvoie {_court(resultats)} au lieu des résultats de la fonction "
                             "([0.5, 0.5, 0.5, 0.5, 0.5]).")
    noms = [a[0] for a in calculs]
    if noms != ["C1", "C2", "C3"]:
        if noms[:2] == ["C1", "C1"]:
            raise AssertionError("Le 2e appel score(\"C1\") a recalculé au lieu de réutiliser le résultat.")
        raise AssertionError(f"Pour les appels C1, C1, C2, C1, C3 (taille_max=2), la fonction a calculé {noms} au lieu "
                             "de ['C1', 'C2', 'C3'].")
    if not callable(getattr(score, "cache_info", None)):
        raise AssertionError("La fonction décorée doit avoir une méthode cache_info() : "
                             "`enveloppe.cache_info = lambda: {...}` avant le `return enveloppe`.")
    info = score.cache_info()
    if info != {"calculs": 3, "reutilisations": 2, "taille": 2}:
        raise AssertionError(f"Après C1, C1, C2, C1, C3, cache_info() renvoie {_court(info)} au lieu de "
                             "{'calculs': 3, 'reutilisations': 2, 'taille': 2}.")
    score("C1")
    if len(calculs) != 3:
        raise AssertionError("Après C1, C1, C2, C1, C3 avec taille_max=2, C1 (utilisé juste avant C3) doit encore être "
                             "en cache : c'est C2, le moins récemment UTILISÉ, qu'il fallait oublier. Vous oubliez le plus "
                             "ancien CALCULÉ (une file, FIFO) : à chaque réutilisation, déplacez la clé à la fin "
                             "(OrderedDict.move_to_end).")
    score("C2")
    if len(calculs) != 4:
        raise AssertionError("Après C1, C1, C2, C1, C3 avec taille_max=2, C2 aurait dû être oublié (le cache ne garde que "
                             "2 résultats), mais il est encore réutilisé.")

    # ------------------------------------------------------------------ borné
    f, calculs = _fabriquer(cache_lru, 10, 1_000, HorlogeFactice())
    for i in range(1_000):
        f(i)
    taille = f.cache_info()["taille"]
    if taille != 10:
        raise AssertionError(f"Après 1 000 clients différents avec taille_max=10, le cache contient {taille} résultats "
                             "au lieu de 10. Un cache qui ne s'arrête jamais de grossir finit par occuper toute la "
                             "mémoire du serveur : quand il dépasse taille_max, oubliez le moins récemment utilisé "
                             "(OrderedDict.popitem(last=False)).")
    avant = len(calculs)
    for i in range(990, 1_000):
        f(i)
    f(989)
    if len(calculs) != avant + 1:
        raise AssertionError("Avec taille_max=10, après 1 000 clients, les 10 derniers doivent être en cache et le "
                             "11e avant la fin (989) oublié.")

    # ------------------------------------------------------------------ la mémoire reste bornée pour de vrai
    f = cache_lru(10, 1_000, horloge=HorlogeFactice())(lambda client_id: None)   # ne garde rien elle-même
    deja = tracemalloc.is_tracing()
    if not deja:
        tracemalloc.start()
    try:
        f(-1)
        gc.collect()
        depart = tracemalloc.get_traced_memory()[0]
        for i in range(30_000):
            f(i)
        gc.collect()
        retenu = tracemalloc.get_traced_memory()[0] - depart
    finally:
        if not deja:
            tracemalloc.stop()
    if retenu > 300_000:
        raise AssertionError(f"Avec taille_max=10, après 30 000 clients différents, votre cache retient encore "
                             f"{retenu / 1e6:.1f} Mo de mémoire : quelque chose garde la trace de chaque clé (une liste "
                             "`ordre` qui ne fait que grandir, un dictionnaire de dates…). Tout ce que garde le cache "
                             "doit être borné par taille_max.")

    # ------------------------------------------------------------------ périmé, même relu
    horloge = HorlogeFactice()
    f, calculs = _fabriquer(cache_lru, 5, 60, horloge)
    f("C1")
    horloge.avancer(59.5)
    f("C1")
    if len(calculs) != 1:
        raise AssertionError("Un résultat calculé il y a 59,5 secondes avec duree=60 a été recalculé : il est encore "
                             "valable.")
    horloge.avancer(0.5)
    f("C1")
    if len(calculs) != 2:
        raise AssertionError("Un résultat calculé il y a 60 secondes PILE avec duree=60 a été réutilisé : à calcul + "
                             "duree exactement, il est périmé (âge >= duree). S'il a été relu 0,5 s plus tôt, la relecture "
                             "ne le rajeunit pas : sinon un score consulté chaque minute ne serait JAMAIS recalculé. Et "
                             "lisez l'heure avec horloge(), le paramètre : c'est elle que le correcteur avance.")
    horloge.avancer(30)
    f("C1")
    if len(calculs) != 2:
        raise AssertionError("Un résultat recalculé il y a 30 secondes (duree=60) a été recalculé encore : après un "
                             "recalcul, l'âge repart de zéro.")

    # ------------------------------------------------------------------ recalculer un périmé n'expulse personne
    horloge = HorlogeFactice()
    f, calculs = _fabriquer(cache_lru, 2, 60, horloge)
    f("A")                    # t = 0
    horloge.avancer(10)
    f("B")                    # t = 10
    horloge.avancer(10)
    f("A")                    # t = 20 : réutilisé ; ordre d'utilisation : B, A
    horloge.avancer(41)
    f("A")                    # t = 61 : A est périmé, on le recalcule ; B (51 s) est encore valable
    horloge.avancer(1)
    f("B")                    # t = 62 : B doit être réutilisé
    if [a[0] for a in calculs] != ["A", "B", "A"]:
        raise AssertionError("Avec taille_max=2 : A calculé à t=0, B à t=10, A relu à t=20, A périmé recalculé à t=61. "
                             f"À t=62, B (calculé il y a 52 s, duree=60) devait être réutilisé, mais la fonction a calculé "
                             f"{[a[0] for a in calculs]}. Recalculer A ne fait pas grossir le cache (A y était déjà) : "
                             "personne ne doit être oublié. Supprimez l'ancienne valeur avant de ranger la nouvelle, et "
                             "n'oubliez le plus ancien que si la taille DÉPASSE taille_max après l'ajout.")

    # ------------------------------------------------------------------ None, 0.0, exceptions
    for valeur in (None, 0.0, 0, ""):
        f, calculs = _fabriquer(cache_lru, 5, 60, HorlogeFactice(), valeur=lambda *a, v=valeur: v)
        if f("C1") != valeur or f("C1") != valeur:
            raise AssertionError(f"La fonction décorée doit renvoyer {valeur!r}, comme la fonction d'origine.")
        if len(calculs) != 1:
            raise AssertionError(f"Un résultat qui vaut {valeur!r} a été recalculé : il se met en cache comme les autres. "
                                 "`if memoire.get(cle):` est faux pour None et 0.0 : testez `if cle in memoire:`.")

    class Panne(Exception):
        pass

    pannes = []

    def capricieuse(client_id):
        pannes.append(client_id)
        if len(pannes) == 1:
            raise Panne("modèle indisponible")
        return 0.7

    g = cache_lru(5, 60, horloge=HorlogeFactice())(capricieuse)
    _leve(lambda: g("C1"), Panne, "Quand la fonction lève une exception, la fonction décorée",
          "L'erreur doit remonter à l'appelant.")
    if g("C1") != 0.7 or len(pannes) != 2:
        raise AssertionError("Après une exception, l'appel suivant doit RECALCULER (une erreur passagère ne se met pas "
                             "en cache) et renvoyer le résultat.")

    # ------------------------------------------------------------------ plusieurs arguments, plusieurs fonctions
    f, calculs = _fabriquer(cache_lru, 5, 60, HorlogeFactice())
    if f("C1", 0.3) != ("résultat", "C1", 0.3) or f("C1", 0.7) != ("résultat", "C1", 0.7) or len(calculs) != 2:
        raise AssertionError("f(\"C1\", 0.3) et f(\"C1\", 0.7) sont deux calculs différents : la clé, c'est TOUS les "
                             "arguments (le tuple args).")

    horloge = HorlogeFactice()
    memo = cache_lru(10, 60, horloge=horloge)

    @memo
    def risque(client_id):
        return "risque"

    @memo
    def segment(client_id):
        return "segment"

    if (risque("C1"), segment("C1")) != ("risque", "segment"):
        raise AssertionError("Deux fonctions décorées avec le MÊME `memo = cache_lru(10, 60)` partagent leur cache : "
                             "segment(\"C1\") renvoie le résultat de risque(\"C1\"). Créez le dictionnaire DANS "
                             "decorateur(fonction) : il en faut un par fonction décorée, pas un par appel à cache_lru.")
    if risque.cache_info()["taille"] != 1:
        raise AssertionError("Deux fonctions décorées avec le même `memo` doivent avoir chacune leur cache_info().")

    f, _ = _fabriquer(cache_lru, 5, 60, HorlogeFactice())
    if f.__name__ != "fonction" or f.__doc__ != "Calcule un score (documentation d'origine).":
        raise AssertionError(f"La fonction décorée s'appelle « {f.__name__} » et a perdu sa documentation : ajoutez "
                             "@functools.wraps(fonction) au-dessus de `def enveloppe(*args):`.")

    # ------------------------------------------------------------------ paramètres invalides
    for args, quoi in (((0, 60), "cache_lru(0, 60)"), ((-1, 60), "cache_lru(-1, 60)"),
                       ((10, 0), "cache_lru(10, 0)"), ((10, -5), "cache_lru(10, -5)")):
        _leve(lambda args=args: cache_lru(*args), ValueError, quoi,
              "Vérifiez taille_max et duree dès l'appel à cache_lru, avant de décorer quoi que ce soit.")


# ============================================================================ Défi 114 : lire en flux


def _totaux_reference(chemin) -> dict[str, tuple[int, float]]:
    nombres: dict[str, int] = {}
    sommes: dict[str, float] = {}
    with open(chemin, newline="", encoding="utf-8") as fichier:
        for ligne in csv.DictReader(fichier):
            if ligne["montant"]:
                nombres[ligne["contrat"]] = nombres.get(ligne["contrat"], 0) + 1
                sommes[ligne["contrat"]] = sommes.get(ligne["contrat"], 0.0) + float(ligne["montant"])
    return {c: (nombres[c], round(sommes[c], 2)) for c in nombres}


@functools.lru_cache(maxsize=1)
def _dossier_commun() -> Path:
    """Un dossier temporaire pour toute la session (le gros fichier n'est écrit qu'une fois), effacé à la fin."""
    import atexit

    dossier = Path(tempfile.mkdtemp(prefix="serie18_"))
    atexit.register(shutil.rmtree, dossier, ignore_errors=True)
    return dossier


@functools.lru_cache(maxsize=1)
def _gros_fichiers() -> tuple[Path, Path]:
    """Deux exports de nuit, le second deux fois plus long que le premier (écrits une fois par session)."""
    return (ecrire_factures(_dossier_commun() / "factures_nuit.csv", lignes=30_000, graine=114),
            ecrire_factures(_dossier_commun() / "factures_nuit_x2.csv", lignes=60_000, graine=115))


def _pic_memoire(totaux_par_contrat, chemin, quoi: str):
    """Appelle votre fonction sous tracemalloc ; renvoie (résultat, pic de mémoire en octets)."""
    deja = tracemalloc.is_tracing()
    gc.collect()
    if deja:
        tracemalloc.reset_peak()
        depart = tracemalloc.get_traced_memory()[0]
    else:
        tracemalloc.start()
        depart = 0
    try:
        resultat = _appeler_totaux(totaux_par_contrat, chemin, quoi)
        return resultat, tracemalloc.get_traced_memory()[1] - depart
    finally:
        if not deja:
            tracemalloc.stop()


def _appeler_totaux(totaux_par_contrat, chemin, quoi: str):
    with warnings.catch_warnings(record=True) as alertes:
        warnings.simplefilter("always", ResourceWarning)
        try:
            resultat = totaux_par_contrat(chemin)
        except NotImplementedError:
            raise
        except (ValueError, IndexError, KeyError) as exc:
            raise AssertionError(f"{quoi} lève {_decrire(exc)}. Le fichier contient des offres entre guillemets qui "
                                 "contiennent une virgule (« \"Fibre, TV\" ») et des montants vides (factures annulées) : "
                                 "`ligne.split(\",\")` décale les colonnes. Utilisez le module csv (csv.DictReader), "
                                 "et ignorez les montants vides.") from None
        gc.collect()
    if any(issubclass(a.category, ResourceWarning) for a in alertes):
        raise AssertionError(f"{quoi} laisse le fichier OUVERT (Python le signale par un ResourceWarning). Sur un "
                             "serveur, les fichiers jamais refermés finissent par bloquer le programme (« Too many open "
                             "files ») : `with open(chemin, ...) as fichier:` le referme à la sortie du bloc.")
    if resultat is None:
        raise AssertionError("totaux_par_contrat renvoie None : avez-vous oublié le `return` ?")
    return resultat


def _comparer_totaux(resultat, attendu, quoi: str) -> None:
    if not isinstance(resultat, dict):
        raise AssertionError(f"{quoi} renvoie {_court(resultat)} : renvoyez un dictionnaire contrat → (nombre, total).")
    if set(resultat) != set(attendu):
        raise AssertionError(f"{quoi} renvoie les contrats {sorted(map(str, resultat))} au lieu de {sorted(attendu)}. "
                             "Lisez-vous la bonne colonne, par son NOM (ligne[\"contrat\"]) ?")
    for contrat, (nombre, total) in attendu.items():
        valeur = resultat[contrat]
        try:
            n_obtenu, total_obtenu = valeur
        except (TypeError, ValueError):
            raise AssertionError(f"{quoi} : pour « {contrat} », la valeur doit être un couple (nombre, total), pas "
                                 f"{_court(valeur)}.") from None
        if n_obtenu != nombre:
            conseil = ""
            if n_obtenu > nombre:
                conseil = (" Vous comptez sans doute les factures annulées (montant vide) : elles ne comptent pas. Avec "
                           "pandas, `count()` ignore les valeurs manquantes mais `size()` les compte.")
            raise AssertionError(f"{quoi} : {n_obtenu} factures « {contrat} » au lieu de {nombre}.{conseil}")
        if not isinstance(total_obtenu, (int, float)) or abs(total_obtenu - total) > 0.011:
            raise AssertionError(f"{quoi} : total « {contrat} » de {_court(total_obtenu)} au lieu de {total}.")


def verifier_totaux_par_contrat(totaux_par_contrat) -> None:
    with tempfile.TemporaryDirectory() as dossier:
        petit = ecrire_factures(Path(dossier) / "factures.csv", lignes=3_000, graine=7)
        attendu = _totaux_reference(petit)
        _comparer_totaux(_appeler_totaux(totaux_par_contrat, petit, "Sur un fichier de 3 000 factures, totaux_par_contrat"),
                         attendu, "Sur un fichier de 3 000 factures, totaux_par_contrat")
        autre_ordre = ecrire_factures(Path(dossier) / "factures_v2.csv", lignes=3_000, graine=7,
                                      colonnes=("montant", "offre", "contrat", "client_id"))
        quoi = ("Sur le même fichier avec les colonnes dans un autre ordre (montant, offre, contrat, client_id), "
                "totaux_par_contrat")
        try:
            resultat = _appeler_totaux(totaux_par_contrat, autre_ordre, quoi)
            _comparer_totaux(resultat, attendu, quoi)
        except AssertionError as exc:
            raise AssertionError(f"{exc} Lisez les colonnes par leur NOM (csv.DictReader : ligne[\"montant\"]), "
                                 "pas par leur position : l'export peut changer d'ordre.") from None
        windows = Path(dossier) / "factures_windows.csv"
        windows.write_bytes(petit.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        _comparer_totaux(_appeler_totaux(totaux_par_contrat, windows, "Sur un fichier aux fins de ligne Windows (\\r\\n)"),
                         attendu, "Sur un fichier aux fins de ligne Windows (\\r\\n)")
        resultat = _appeler_totaux(totaux_par_contrat, str(petit), "Avec un chemin donné en texte (str)")
        _comparer_totaux(resultat, attendu, "Avec un chemin donné en texte (str)")

    # La mémoire ne doit pas grandir avec le fichier : on mesure le pic sur un export, puis sur un export
    # deux fois plus long. Lire en flux (ou par paquets de taille fixe) donne le même pic ; tout charger le double.
    petit, gros = _gros_fichiers()
    mesures = []
    for chemin in (petit, gros):
        quoi = f"Sur l'export de {_nombre(30_000 if chemin == petit else 60_000)} factures, totaux_par_contrat"
        resultat, pic = _pic_memoire(totaux_par_contrat, chemin, quoi)
        _comparer_totaux(resultat, _totaux_reference(chemin), quoi)
        mesures.append((chemin.stat().st_size, pic))
    (taille_1, pic_1), (taille_2, pic_2) = mesures
    if pic_2 > 1.3 * pic_1 + 200_000 or pic_2 > 5_000_000:
        raise AssertionError(f"Sur un export de {taille_1 / 1e6:.1f} Mo, votre fonction a occupé jusqu'à {pic_1 / 1e6:.1f} Mo "
                             f"de mémoire ; sur un export deux fois plus long ({taille_2 / 1e6:.1f} Mo), jusqu'à "
                             f"{pic_2 / 1e6:.1f} Mo. La mémoire grandit avec le fichier : votre fonction le charge en entier "
                             "avant de compter (f.read(), f.readlines(), list(csv.reader(...)), pd.read_csv sans chunksize). "
                             "Sur l'export de 10 Go, le serveur manquerait de mémoire. Parcourez le fichier ligne par ligne "
                             "(for ligne in csv.DictReader(fichier)), ou par paquets : pd.read_csv(..., chunksize=10_000).")


# ============================================================================ Défi 115 : optimiser sans rien changer


class _Ligne(dict):
    """Un dictionnaire espion : chaque lecture est comptée."""

    __slots__ = ("_compteur", "_categorie")

    def _compter(self, n: int = 1) -> None:
        self._compteur(self._categorie, n)

    def __getitem__(self, cle):
        self._compter()
        return dict.__getitem__(self, cle)

    def get(self, cle, defaut=None):
        self._compter()
        return dict.get(self, cle, defaut)

    def __contains__(self, cle):
        self._compter()
        return dict.__contains__(self, cle)

    def __iter__(self):
        self._compter(len(self))
        return dict.__iter__(self)

    def keys(self):
        self._compter(len(self))
        return dict.keys(self)

    def values(self):
        self._compter(len(self))
        return dict.values(self)

    def items(self):
        self._compter(len(self))
        return dict.items(self)


class _Liste(list):
    """Une liste espionne : chaque élément parcouru, chaque recherche (`in`, `index`, `count`) est comptée."""

    __slots__ = ("_compteur", "_categorie")

    def _compter(self, n: int = 1) -> None:
        self._compteur(self._categorie, n)

    def __iter__(self):
        for element in list.__iter__(self):
            self._compter()
            yield element

    def __contains__(self, x):
        self._compter(len(self))
        return list.__contains__(self, x)

    def __getitem__(self, i):
        self._compter(len(range(len(self))[i]) if isinstance(i, slice) else 1)
        return list.__getitem__(self, i)

    def index(self, *args):
        self._compter(len(self))
        return list.index(self, *args)

    def count(self, x):
        self._compter(len(self))
        return list.count(self, x)

    def copy(self):
        self._compter(len(self))
        return list.copy(self)

    def __reversed__(self):
        self._compter(len(self))
        return list.__reversed__(self)


def _espionner(compteur, categorie: str, elements, en_lignes: bool) -> _Liste:
    liste = _Liste()
    for element in elements:
        if en_lignes:
            ligne = _Ligne(element)
            ligne._compteur, ligne._categorie = compteur, categorie
            element = ligne
        list.append(liste, element)
    liste._compteur, liste._categorie = compteur, categorie
    return liste


def _reference_rapide(clients, appels, deja_contactes, seuil_appels=3):
    """Le même résultat que clients_a_rappeler_lent (vérifié par tests/test_defis.py), sans déclencher les espions."""
    lire = dict.__getitem__
    tous = list(list.__iter__(clients))
    if not tous:
        return []
    nb: dict[str, int] = {}
    for appel in list.__iter__(appels):
        if lire(appel, "motif") != "information":
            nb[lire(appel, "client_id")] = nb.get(lire(appel, "client_id"), 0) + 1
    moyenne = sum(lire(c, "facture") for c in tous) / len(tous)
    exclus = set(list.__iter__(deja_contactes))
    resultat = [(lire(c, "client_id"), nb.get(lire(c, "client_id"), 0)) for c in tous
                if lire(c, "client_id") not in exclus]
    factures = [lire(c, "facture") for c in tous if lire(c, "client_id") not in exclus]
    resultat = [paire for paire, facture in zip(resultat, factures) if paire[1] >= seuil_appels and facture >= moyenne]
    resultat.sort(key=lambda paire: paire[1], reverse=True)
    return resultat


def _cl(client_id: str, facture: float) -> dict:
    return {"client_id": client_id, "facture": facture, "contrat": "Mensuel"}


def _ap(client_id: str, motif: str = "panne", fois: int = 1) -> list[dict]:
    return [{"client_id": client_id, "motif": motif} for _ in range(fois)]


#: (description, clients, appels, deja_contactes, seuil_appels, conseil si le résultat diffère)
_CAS_LIMITES = [
    ("sans aucun client", [], _ap("C1", fois=4), [], 3,
     "Sans client, la version lente ne calcule jamais la moyenne (la boucle ne tourne pas) et renvoie []. Sortie de "
     "la boucle, la moyenne divise par zéro : traitez ce cas avant."),
    ("avec des clients pile à la moyenne", [_cl("C1", 50), _cl("C2", 50)], _ap("C1", fois=3) + _ap("C2", fois=4), [], 3,
     "Une facture ÉGALE à la moyenne compte (>=), et 3 appels pour un seuil de 3 aussi."),
    ("avec des appels pour « information »", [_cl("C1", 60), _cl("C2", 60), _cl("C3", 30)],
     _ap("C1", "information", 3) + _ap("C1", "panne", 2) + _ap("C2", "facture", 3), [], 3,
     "Les appels dont le motif est « information » ne comptent pas."),
    ("avec un client déjà contacté qui tire la moyenne vers le haut",
     [_cl("C1", 100), _cl("C2", 60), _cl("C3", 40)], _ap("C1", fois=5) + _ap("C2", fois=5) + _ap("C3", fois=5), ["C1"], 3,
     "La moyenne porte sur TOUS les clients (100, 60 et 40 : 66,67), y compris ceux déjà contactés : C2 (60) est "
     "sous la moyenne. Calculez-la avant d'écarter qui que ce soit."),
    ("avec des égalités d'appels", [_cl("Z9", 80), _cl("A1", 80), _cl("C0", 10), _cl("M5", 80), _cl("B2", 80)],
     _ap("B2", fois=5) + _ap("Z9", fois=4) + _ap("A1", fois=4) + _ap("M5", fois=4), [], 3,
     "À égalité d'appels, la version lente garde l'ordre des clients (Z9, A1, M5) : son tri est stable, même avec "
     "reverse=True. Ne triez pas en plus par identifiant."),
    ("avec 3 factures de 0.1 €", [_cl("C1", 0.1), _cl("C2", 0.1), _cl("C3", 0.1)],
     _ap("C1", fois=3) + _ap("C2", fois=3) + _ap("C3", fois=3), [], 3,
     "En flottants, sum([0.1, 0.1, 0.1]) / 3 vaut 0.10000000000000002 : pour la version lente, aucun client n'atteint "
     "la moyenne. statistics.mean (ou numpy) donne 0.1 tout rond et change le résultat. Optimiser, c'est changer la "
     "vitesse, pas le calcul : reprenez exactement sum(...) / len(...)."),
    ("avec seuil_appels=0 et des clients sans aucun appel", [_cl("C1", 70), _cl("C2", 90), _cl("C3", 10)],
     _ap("C1", fois=1), [], 0,
     "Avec seuil_appels=0, un client sans aucun appel compte (0 appel >= 0). Un dictionnaire d'appels ne le "
     "contient pas : nb.get(client_id, 0), ou un Counter, qui renvoie 0."),
    ("avec deux fois le même client dans la liste", [_cl("C1", 80), _cl("C2", 20), _cl("C1", 80)],
     _ap("C1", fois=3), [], 3,
     "La liste contient deux fois C1 : la version lente le renvoie deux fois. Ne regroupez pas les clients par "
     "identifiant (un dictionnaire client_id → client en garde un seul)."),
    ("avec des appels de clients inconnus et des déjà-contactés inconnus", [_cl("C1", 60), _cl("C2", 40)],
     _ap("C9", fois=6) + _ap("C1", fois=3), ["C7", "C8"], 3,
     "Les appels de clients absents de la liste ne changent rien, pas plus que des identifiants déjà contactés "
     "inconnus."),
]


def _tirer_cas(rng: random.Random):
    ids = [f"C{i}" for i in range(rng.randint(0, 9))]
    factures = (20.0, 35.5, 50.0, 50.0, 80.0, 0.1, 99.99)
    clients = [_cl(cid, rng.choice(factures)) for cid in ids]
    if clients and rng.random() < 0.2:
        clients.append(dict(rng.choice(clients)))              # un doublon
    rng.shuffle(clients)
    possibles = ids + ["X1", "X2"]
    appels = [{"client_id": rng.choice(possibles), "motif": rng.choice(("panne", "facture", "information", "resiliation"))}
              for _ in range(rng.randint(0, 40))]
    deja = rng.sample(possibles, rng.randint(0, len(possibles)))
    return clients, appels, deja, rng.randint(0, 5)


def _appeler_rappel(fonction, clients, appels, deja, seuil):
    import copy

    avant = copy.deepcopy((clients, appels, deja))
    try:
        resultat = fonction(clients, appels, deja, seuil)
    except NotImplementedError:
        raise
    except ZeroDivisionError:
        raise AssertionError("Sans aucun client, votre fonction divise par zéro, alors que la version lente renvoie [] "
                             "(sa moyenne est calculée DANS la boucle, qui ne tourne pas). En sortant un calcul de la "
                             "boucle, vous l'exécutez aussi quand la boucle ne tourne pas : traitez ce cas.") from None
    except KeyError as exc:
        raise AssertionError(f"Votre fonction lève KeyError({exc}) avec seuil_appels={seuil}, clients={_court(clients, 200)}, "
                             f"appels={_court(appels, 200)}. Un client sans appel n'est pas dans votre dictionnaire "
                             "d'appels : nb.get(client_id, 0), ou un Counter, qui renvoie 0.") from None
    if (clients, appels, deja) != avant:
        raise AssertionError("Votre fonction a MODIFIÉ une des listes reçues (clients, appels ou deja_contactes) : "
                             "l'appelant s'en sert encore. Travaillez sur des copies, ou ne touchez à rien.")
    return resultat


def verifier_clients_a_rappeler(clients_a_rappeler) -> None:
    for description, clients, appels, deja, seuil, conseil in _CAS_LIMITES:
        attendu = clients_a_rappeler_lent(clients, appels, deja, seuil)
        resultat = _appeler_rappel(clients_a_rappeler, clients, appels, deja, seuil)
        if resultat is None:
            raise AssertionError("clients_a_rappeler renvoie None : avez-vous oublié le `return` ?")
        if resultat != attendu:
            if isinstance(resultat, list) and [tuple(r) for r in resultat if isinstance(r, (list, tuple))] == attendu:
                raise AssertionError(f"Votre fonction renvoie {_court(resultat)} : la version lente renvoie des TUPLES "
                                     f"{_court(attendu)}. Même résultat, c'est aussi même type.")
            raise AssertionError(f"Cas {description} (seuil_appels={seuil}) : votre fonction renvoie {_court(resultat)}, "
                                 f"la version lente {_court(attendu)}. {conseil}")

    parametres = inspect.signature(clients_a_rappeler).parameters
    if "seuil_appels" not in parametres or parametres["seuil_appels"].default != 3:
        raise AssertionError("clients_a_rappeler doit garder la même signature que la version lente : "
                             "seuil_appels=3 par défaut.")

    rng = random.Random(115)
    for numero in range(300):
        clients, appels, deja, seuil = _tirer_cas(rng)
        attendu = clients_a_rappeler_lent(clients, appels, deja, seuil)
        resultat = _appeler_rappel(clients_a_rappeler, clients, appels, deja, seuil)
        if resultat != attendu:
            raise AssertionError(f"Cas tiré au hasard n°{numero + 1} : clients={_court(clients, 400)}, "
                                 f"appels={_court(appels, 400)}, deja_contactes={deja}, seuil_appels={seuil}. "
                                 f"Votre fonction renvoie {_court(resultat)}, la version lente {_court(attendu)}. "
                                 "Rejouez ce cas à la main avec les deux fonctions pour voir où elles divergent.")

    # Le budget : 2 000 clients, 20 000 appels, 200 déjà contactés.
    from .performance import exemple_rappels

    clients, appels, deja = exemple_rappels(n_clients=2_000, graine=115)
    deja = deja[:200]
    n, m, k = len(clients), len(appels), len(deja)
    budget = 10 * (n + m + k)
    compteur = _Compteur(budget=budget)
    espions = (_espionner(compteur, "clients", clients, True), _espionner(compteur, "appels", appels, True),
               _espionner(compteur, "deja_contactes", deja, False))
    attendu = _reference_rapide(*espions)
    depart = time.perf_counter()
    try:
        resultat = clients_a_rappeler(*espions)
    except _BudgetDepasse:
        detail = ", ".join(f"{cat} : {_nombre(nb)}" for cat, nb in
                           sorted(compteur.par_categorie.items(), key=lambda kv: -kv[1]))
        taille = {"clients": n, "appels": m, "deja_contactes": k}
        pire = max(compteur.par_categorie, key=compteur.par_categorie.get)
        conseils = {
            "appels": "Vous relisez les appels pour chaque client : comptez-les en UN passage (Counter, ou un dictionnaire "
                      "client_id → nombre), avant la boucle sur les clients.",
            "clients": "Vous relisez tous les clients pour chaque client : la facture moyenne est la même pour tous, "
                       "calculez-la une fois, avant la boucle.",
            "deja_contactes": "`client_id in deja_contactes` parcourt toute la liste, pour chaque client : construisez "
                              "un set une fois (exclus = set(deja_contactes)), puis `client_id in exclus` est immédiat.",
        }
        raise AssertionError(f"Sur {_nombre(n)} clients, {_nombre(m)} appels et {k} déjà contactés, votre fonction a "
                             f"dépassé {_nombre(budget)} lectures (10 par élément) : le correcteur l'a arrêtée. Lectures au "
                             f"moment de l'arrêt : {detail}. La liste « {pire} » ({_nombre(taille[pire])} éléments) est lue "
                             f"bien plus d'une fois. {conseils[pire]}") from None
    duree = time.perf_counter() - depart
    if resultat != attendu:
        raise AssertionError(f"Sur {n} clients et {m} appels, votre fonction ne renvoie pas le même résultat que la "
                             "version lente.")
    if duree > 2:
        raise AssertionError(f"Sur {n} clients et {m} appels, votre fonction met {duree:.1f} s (une version en un passage "
                             "met moins d'une demi-seconde, comptage des lectures compris). Un tri ou une recherche dans "
                             "une liste à l'intérieur de la boucle ?")


SERIE = Serie(
    code="18",
    titre="Rendre le code rapide, preuves à l'appui : mesurer, profiler, bisect, cache, mémoire, optimiser",
    module_eleve="defis.serie_18",
    module_corrige="defis.corrige_serie_18",
    defis=(
        Defi(110, "chronometrer", "🟢", "—", verifier_chronometrer),
        Defi(111, "point_chaud", "🟢", "—", verifier_point_chaud),
        Defi(112, "prix_en_vigueur", "🟡", "—", verifier_prix_en_vigueur),
        Defi(113, "cache_lru", "🟡", "—", verifier_cache_lru),
        Defi(114, "totaux_par_contrat", "🟡", "—", verifier_totaux_par_contrat),
        Defi(115, "clients_a_rappeler", "🔴", "—", verifier_clients_a_rappeler),
    ),
)

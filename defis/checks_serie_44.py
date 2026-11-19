"""Vérifications automatiques de la série 44 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est la direction financière, qui ne paiera des conseillers de plus que si vos chiffres tiennent,
et le standard téléphonique, qui vous envoie ses journaux tels qu'ils sont vraiment.

- il compare vos indicateurs, vos files, vos probabilités, vos effectifs et vos journaux simulés à ceux d'une
  version de référence (à un milliardième près pour les nombres à virgule) ; pour le planning, où plusieurs
  réponses sont justes, il vérifie ses PROPRIÉTÉS (chaque heure couverte, aucune vacation après la fermeture, et
  pas une vacation de plus que nécessaire) ;
- quand votre résultat est faux, il rejoue les erreurs classiques (la moyenne des seuls appels décrochés, le
  niveau de service divisé par les décrochés, `<` au lieu de `<=`, la moyenne des événements au lieu de la moyenne
  dans le temps, l'arrivée comptée avant le départ, Erlang B à la place d'Erlang C, l'occupation prise pour une
  probabilité, des secondes prises pour des heures, la règle des 85 %, le renfort qui commence trop tôt, l'appel
  abandonné qui occupe un conseiller, le conseiller qui décroche après la fin de son service…) pour nommer la
  vôtre ;
- il choisit des journées où chaque erreur se voit (un client décroché à 20 s pile, un autre qui raccroche au bout
  de 8 s, un conseiller libéré à la seconde où un client arrive, 1 000 conseillers, une fin de journée qui force
  la dernière vacation, deux conseillers libres en même temps), et compte les lignes exécutées dans votre fichier
  quand comparer chaque appel à tous les autres serait trop lent ;
- pendant le défi 269, il prête sa propre `erlang_c` à votre fichier : chaque défi est noté seul.
"""

from __future__ import annotations

import copy
import heapq
import math
import random
import sys

from .centre_appels import (
    DUREE_VACATION_H,
    SEUIL_S,
    demande_exemple,
    generer_demande,
    horaires,
    journal_exemple,
    planning_actuel,
)
from .verif import Defi, Serie

# ---------------------------------------------------------------------------- outils


def _decrire(exc: BaseException) -> str:
    texte = str(exc).split("\n")[0]
    if len(texte) > 160:
        texte = texte[:159] + "…"
    return f"{type(exc).__name__}({texte!r})" if texte else type(exc).__name__


def _court(valeur, n: int = 160) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


class _BoucleSansFin(BaseException):
    """Levée dans VOTRE code quand il travaille trop (BaseException : `except Exception` ne l'attrape pas)."""


def _fichier(objet) -> str | None:
    code = getattr(objet, "__code__", None) or getattr(getattr(objet, "__func__", None), "__code__", None)
    if code is not None:
        return code.co_filename
    module = sys.modules.get(getattr(objet, "__module__", None) or "")
    return getattr(module, "__file__", None)


def _trace(fichier: str | None, appel, lignes_max: int | None):
    """Le résultat de `appel()` ; arrête net au-delà de `lignes_max` lignes exécutées dans `fichier`."""
    if fichier is None or lignes_max is None:
        return appel()
    compte = [0]

    def local(frame, evenement, arg):
        if evenement == "line":
            compte[0] += 1
            if compte[0] > lignes_max:
                raise _BoucleSansFin
        return local

    def global_(frame, evenement, arg):
        return local if frame.f_code.co_filename == fichier else None

    precedent = sys.gettrace()
    sys.settrace(global_)
    try:
        return appel()
    finally:
        sys.settrace(precedent)


_CONSEIL_LENT = "Votre code a été arrêté : il exécute beaucoup trop de lignes (une boucle qui ne s'arrête pas ?)."


def _conseil_exception(exc: BaseException) -> str:
    texte = str(exc)
    if isinstance(exc, OverflowError) or "too large" in texte:
        return ("Un nombre a débordé : A^N et N! ne tiennent plus dans un float au-delà de quelques centaines "
                "(`math.factorial(171)` dépasse déjà le plus grand float). Utilisez la récurrence d'Erlang B : b = 1.0, "
                "puis `b = trafic * b / (k + trafic * b)` pour k de 1 à N ; chaque étape reste entre 0 et 1.")
    if isinstance(exc, TypeError) and "NoneType" in texte:
        return ("Un calcul a reçu None : pour un client qui a raccroché, 'decroche' vaut None. Testez `appel['decroche'] "
                "is None` AVANT de soustraire ou de comparer (son attente va jusqu'à sa 'fin').")
    if isinstance(exc, ZeroDivisionError):
        return ("Une division par zéro : aucun appel décroché ('attente_decroches_s' vaut alors None), une fenêtre vide, "
                "ou N = A dans Erlang C (pour trafic >= conseillers, la réponse est 1.0, sans calcul).")
    if isinstance(exc, KeyError):
        return ("Une clé manque : un appel du journal a 'arrivee', 'decroche' et 'fin' ; un appel à simuler a 'arrivee', "
                "'duree' et 'patience'. Le résultat de `simuler` ajoute 'decroche' et 'fin' à une COPIE de l'appel.")
    if isinstance(exc, IndexError):
        return ("Un indice déborde : une vacation commencée trop tard finirait après la fermeture ; la dernière qui tient "
                "commence à len(besoins) - duree (et prévoyez `partent` de longueur len(besoins) + 1).")
    if isinstance(exc, TypeError) and "not supported between" in texte:
        return ("Python ne sait pas comparer ces deux valeurs : dans un tas, mettez des tuples de nombres (libre à partir "
                "de, numéro, fin de service), pas des dict.")
    return ""


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = None, conseil_lent: str = "",
           kwargs: dict | None = None):
    """f(*args), en transformant toute erreur en message lisible (NotImplementedError : le défi est à faire)."""
    kwargs = kwargs or {}
    try:
        return _trace(_fichier(f) if lignes_max else None, lambda: f(*args, **kwargs), lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError. {conseil}".rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {_conseil_exception(exc) or conseil}".rstrip()) from None


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu: type = ValueError, kwargs: dict | None = None):
    """Exige que f(*args) lève `attendu` (et rien d'autre). Renvoie l'exception, pour lire son message."""
    kwargs = kwargs or {}
    try:
        resultat = _trace(_fichier(f), lambda: f(*args, **kwargs), 2_000_000)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} ne s'arrête pas : il devrait lever {attendu.__name__}. {conseil}".rstrip()) from None
    except attendu as exc:
        return exc
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _premier_bug(obtenus, rejouer, bugs: dict, egal) -> str:
    """Le conseil du premier bug dont le rejeu donne EXACTEMENT vos réponses."""
    for bug, conseil in bugs.items():
        try:
            faux = rejouer(bug)
        except Exception:
            continue
        if egal(faux, obtenus):
            return conseil
    return ""


def _nombre(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _entier(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _proche(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return _nombre(a) and _nombre(b) and abs(a - b) <= tol * max(1.0, abs(b))


def _listes(egal):
    return lambda a, b: len(a) == len(b) and all(egal(x, y) for x, y in zip(a, b))


def _apercu(appels, n: int = 3) -> str:
    """Un journal en quelques appels, pour un message lisible."""
    if len(appels) <= n:
        return _court(appels, 200)
    return f"<{len(appels)} appels : {_court(list(appels[:2]), 120)[:-1]}, …]>"


# ---------------------------------------------------------------------------- les journaux


def _attente(appel) -> float:
    """L'attente d'un appel du journal : jusqu'au décroché, ou jusqu'à la fin pour un abandon."""
    return (appel["decroche"] if appel["decroche"] is not None else appel["fin"]) - appel["arrivee"]


def _journal(*appels) -> list[dict]:
    """Des appels (arrivee, decroche, fin) en dict."""
    return [{"arrivee": a, "decroche": d, "fin": f} for a, d, f in appels]


def _journal_aleatoire(n: int, graine: int, duree_s: int = 43_200) -> list[dict]:
    """Un journal synthétique de `n` appels, en secondes entières, dans le désordre."""
    alea = random.Random(graine)
    appels = []
    for _ in range(n):
        arrivee = alea.randrange(duree_s)
        attente = alea.choice((0, 0, alea.randrange(1, 40), alea.randrange(1, 600)))
        if alea.random() < 0.2:
            appels.append({"arrivee": arrivee, "decroche": None, "fin": arrivee + attente})
        else:
            appels.append({"arrivee": arrivee, "decroche": arrivee + attente, "fin": arrivee + attente
                           + alea.randrange(30, 900)})
    return appels


def _refus_journal(f, construire, nom: str) -> None:
    """Les journaux invalides : TypeError (pas une liste de dict, pas des nombres) ou ValueError (incohérents)."""
    for appels, attendu, conseil in (
            ("journal", TypeError, "`appels` est une LISTE de dict."),
            ([(0, 5, 10)], TypeError, "Chaque appel est un dict {'arrivee', 'decroche', 'fin'}, pas un tuple."),
            ([{"arrivee": 0, "fin": 10}], ValueError, "Un appel sans clé 'decroche' : le journal est abîmé "
                                                      "(ValueError, avec un message qui dit quelle clé manque)."),
            ([{"arrivee": 0, "decroche": 5, "fin": "10"}], TypeError, "« 10 » est un texte, pas une heure."),
            ([{"arrivee": True, "decroche": 5, "fin": 10}], TypeError,
             "True n'est pas une heure (`isinstance(True, int)` est vrai : excluez bool)."),
            ([{"arrivee": float("nan"), "decroche": None, "fin": 10}], ValueError, "NaN n'est pas une heure."),
            ([{"arrivee": 10, "decroche": 5, "fin": 20}], ValueError, "Décroché (5) avant d'arriver (10) : impossible."),
            ([{"arrivee": 0, "decroche": 30, "fin": 20}], ValueError, "Terminé (20) avant d'être décroché (30)."),
            ([{"arrivee": 30, "decroche": None, "fin": 20}], ValueError, "Raccroché (20) avant d'arriver (30).")):
        args, kwargs = construire(appels)
        _doit_lever(f, args, f"{nom} avec appels = {_court(appels, 80)}", conseil, attendu, kwargs=kwargs)


# ============================================================================ défi 266 : mesurer


_CLES_MESURES = ("appels", "decroches", "abandons", "taux_abandon", "attente_moyenne_s", "attente_decroches_s",
                 "niveau_service")


def _ref_mesurer(appels, seuil=SEUIL_S, bug: str = "") -> dict:
    """Les indicateurs ; `bug` rejoue une erreur classique."""
    n = len(appels)
    toutes = [_attente(a) for a in appels]
    decroches = [_attente(a) for a in appels if a["decroche"] is not None]
    k = len(decroches)
    if bug == "asa":
        moyenne = sum(decroches) / k
    elif bug == "abandon_zero":
        moyenne = sum(decroches) / n
    else:
        moyenne = sum(toutes) / n
    if bug == "strict":
        reussis = sum(w < seuil for w in decroches)
    elif bug == "abandons_comptes":
        reussis = sum(w <= seuil for w in toutes)
    else:
        reussis = sum(w <= seuil for w in decroches)
    return {
        "appels": n,
        "decroches": k,
        "abandons": n - k,
        "taux_abandon": (n - k) / k if bug == "taux_sur_decroches" else (n - k) / n,
        "attente_moyenne_s": moyenne,
        "attente_decroches_s": sum(decroches) / k if k else None,
        "niveau_service": reussis / k if bug == "denominateur" else reussis / n,
    }


_BUGS_MESURER = {
    "asa": "Votre 'attente_moyenne_s' ne compte que les appels décrochés : c'est l'ASA des tableaux de bord (la clé "
           "'attente_decroches_s'). L'attente moyenne honnête compte TOUS les appels : un client qui a raccroché a "
           "attendu jusqu'à sa 'fin', et ce sont justement les attentes les plus longues.",
    "abandon_zero": "Vous divisez bien par tous les appels, mais sans ajouter l'attente des clients qui ont raccroché "
                    "(comptée 0) : un abandon a attendu de son 'arrivee' à sa 'fin'.",
    "denominateur": "Votre niveau de service divise par les appels DÉCROCHÉS : c'est le chiffre flatteur du tableau de "
                    "bord. Divisez par TOUS les appels : un client qui a raccroché n'a pas été décroché en 20 s.",
    "strict": "Vous comptez les appels décrochés en MOINS de seuil_s secondes (<) : « en 20 secondes ou moins » inclut "
              "20 s pile (<=), un cas fréquent quand le standard enregistre des secondes entières.",
    "abandons_comptes": "Vous comptez comme réussis des clients qui ont raccroché vite (en moins de seuil_s secondes) : "
                        "seul un appel DÉCROCHÉ en seuil_s secondes ou moins est un succès ; un abandon est un échec.",
    "taux_sur_decroches": "Votre taux d'abandon divise par les appels décrochés : divisez par TOUS les appels "
                          "(abandons / appels).",
}


def _cas_266() -> list[tuple]:
    """(appels, seuil_s ou None pour la valeur par défaut)."""
    exemple = _journal((0, 5, 300), (10, 30, 200), (20, 140, 400), (30, None, 210))
    melange = _journal((0, 0, 60), (5, 25, 90), (7, None, 15), (9, 70, 200), (12, None, 312), (40, 61, 100),
                       (41, 41, 41), (50, 60, 75))
    abandons = _journal((0, None, 30), (10, None, 12), (20, None, 400))
    tout_vite = _journal((0, 0, 100), (1, 3, 50), (2, 22, 99))
    virgules = _journal((0.5, 20.5, 80.25), (1.25, 22.0, 30.0), (3.0, None, 3.5), (4.75, 35.25, 100.0))
    avec_cles = [dict(a, client=f"C{k}", parti=k % 2 == 0) for k, a in enumerate(melange)]
    simule = [{"arrivee": a["arrivee"], "decroche": a["decroche"], "fin": a["fin"]}
              for a in _ref_simuler_tas(generer_demande((90, 140), graine=2661), 6)]
    return [(exemple, None), (melange, None), (melange, 30), (abandons, None), (tout_vite, 0), (virgules, 20.5),
            (tuple(melange), 21), (avec_cles, None), (simule, None), (journal_exemple(), None)]


def _verifier_un_resultat_266(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} ({type(r).__name__}) : attendu un dict d'indicateurs.")
    manquantes = [c for c in _CLES_MESURES if c not in r]
    if manquantes:
        raise AssertionError(f"{quoi} : il manque la clé {manquantes[0]!r} (les clés : {', '.join(_CLES_MESURES)}).")
    en_trop = [c for c in r if c not in _CLES_MESURES]
    if en_trop:
        raise AssertionError(f"{quoi} : clé inattendue {en_trop[0]!r} (les clés : {', '.join(_CLES_MESURES)}).")
    for cle in ("appels", "decroches", "abandons"):
        if not _entier(r[cle]):
            raise AssertionError(f"{quoi} : {cle!r} vaut {r[cle]!r} ({type(r[cle]).__name__}) : un nombre d'appels est un "
                                 "int.")
    for cle in ("taux_abandon", "attente_moyenne_s", "niveau_service"):
        if not _nombre(r[cle]):
            raise AssertionError(f"{quoi} : {cle!r} vaut {_court(r[cle], 60)} : attendu un nombre.")
    if r["attente_decroches_s"] is not None and not _nombre(r["attente_decroches_s"]):
        raise AssertionError(f"{quoi} : 'attente_decroches_s' vaut {_court(r['attente_decroches_s'], 60)} : attendu un "
                             "nombre, ou None s'il n'y a aucun appel décroché.")


def _mesures_egales(a: dict, b: dict) -> bool:
    return all(_proche(a.get(c), b.get(c)) for c in _CLES_MESURES)


def verifier_mesurer(f) -> None:
    cas = _cas_266()
    obtenus, vus = [], []
    for appels, seuil in cas:
        args, kwargs = (appels,), ({} if seuil is None else {"seuil_s": seuil})
        quoi = f"mesurer({_apercu(appels)}{'' if seuil is None else f', seuil_s={seuil!r}'})"
        avant = copy.deepcopy(appels)
        r = _appel(f, args, quoi, kwargs=kwargs)
        if appels != avant:
            raise AssertionError(f"{quoi} a modifié le journal reçu : un indicateur se calcule sans toucher aux données.")
        _verifier_un_resultat_266(r, quoi)
        obtenus.append(r)
        vus.append((appels, SEUIL_S if seuil is None else seuil))
        attendu = _ref_mesurer(appels, SEUIL_S if seuil is None else seuil)
        for cle in _CLES_MESURES:
            if not _proche(r[cle], attendu[cle]):
                conseil = _premier_bug(obtenus, lambda bug: [_ref_mesurer(a, s, bug) for a, s in vus], _BUGS_MESURER,
                                       _listes(_mesures_egales))
                if not conseil and cle == "attente_decroches_s" and attendu[cle] is None:
                    conseil = "Aucun appel n'a été décroché : 'attente_decroches_s' vaut None (pas 0, pas NaN)."
                if not conseil and seuil is None and cle == "niveau_service":
                    conseil = f"Sans `seuil_s`, le seuil par défaut est SEUIL_S = {SEUIL_S} secondes."
                raise AssertionError(f"{quoi} : {cle!r} vaut {r[cle]!r} au lieu de {attendu[cle]!r}. {conseil}".rstrip())

    _refus_journal(f, lambda appels: ((appels,), {}), "mesurer")
    _doit_lever(f, ([],), "mesurer([])", "Aucun appel : pas d'indicateur (on diviserait par zéro). ValueError.")
    journal = _journal((0, 5, 30))
    for seuil, attendu, conseil in ((-1, ValueError, "Un seuil négatif n'a pas de sens."),
                                    ("20", TypeError, "« 20 » est un texte, pas un nombre de secondes."),
                                    (True, TypeError, "True n'est pas un nombre de secondes.")):
        _doit_lever(f, (journal, seuil), f"mesurer({journal!r}, {seuil!r})", conseil, attendu)


# ============================================================================ défi 267 : file_attente


def _ref_file(appels, debut, fin, bug: str = "") -> tuple[float, int]:
    """(moyenne dans le temps, maximum) des clients en attente ; `bug` rejoue une erreur classique."""
    total = 0
    evenements = []
    for appel in appels:
        a = appel["arrivee"]
        b = appel["fin"] if (bug == "conversation" or appel["decroche"] is None) else appel["decroche"]
        x, y = max(a, debut), min(b, fin)
        if bug == "sans_fenetre":
            if a < b and a < fin and b > debut:
                total += b - a
        elif x < y:
            total += y - x
        if x < y or (bug == "ferme" and x <= y):
            evenements += [(x, +1), (y, -1)]
    if bug == "ferme":
        evenements.sort(key=lambda e: (e[0], -e[1]))          # l'arrivée avant le départ
    elif bug != "sans_tri":
        evenements.sort()
    en_attente = maximum = 0
    releves = []
    for _, variation in evenements:
        en_attente += variation
        maximum = max(maximum, en_attente)
        releves.append(en_attente)
    moyenne = total / (fin - debut)
    if bug == "evenements":
        moyenne = sum(releves) / len(releves) if releves else 0.0
    return moyenne, maximum


_BUGS_FILE = {
    "evenements": "Votre moyenne est celle des nombres de clients relevés à chaque arrivée ou départ : un relevé pendant "
                  "le rush pèse autant qu'une heure de calme. La moyenne dans le TEMPS est la somme des durées d'attente "
                  "tombées dans la fenêtre, divisée par fin - debut.",
    "ferme": "À la même seconde, vous comptez l'arrivée AVANT le départ (ou vous comptez un client décroché à l'instant de "
             "son arrivée) : une attente est [arrivée, fin de l'attente[, ouverte à droite. Triez des tuples (instant, -1) "
             "et (instant, +1) : -1 passe avant +1 ; et une attente de durée nulle ne compte pas.",
    "conversation": "Vous comptez un client décroché jusqu'à la FIN de sa conversation : il n'attend plus dès qu'un "
                    "conseiller décroche. Son attente finit à 'decroche' ('fin' seulement pour un abandon).",
    "sans_fenetre": "Vous comptez les attentes entières, même la part qui déborde de [debut, fin] : un client arrivé "
                    "avant debut ne compte qu'à partir de debut (max(arrivee, debut)), et jusqu'à fin au plus "
                    "(min(fin_attente, fin)).",
    "sans_tri": "Vous parcourez les événements dans l'ordre du journal, sans les trier par instant : le maximum se lit "
                "sur la ligne du TEMPS. Triez les (instant, ±1) d'abord.",
}


def _cas_267() -> list[tuple]:
    """(appels, debut, fin)."""
    exemple = _journal((0, 60, 400), (30, 90, 300), (60, None, 120), (200, 200, 500))
    egalites = _journal((0, 10, 50), (10, 30, 70), (20, 20, 90), (25, None, 40), (40, 40, 41), (40, 55, 99))
    virgules = _journal((0.5, 2.5, 9.0), (1.0, None, 3.25), (2.5, 4.0, 4.5), (3.25, 3.25, 8.0))
    lundi = journal_exemple()
    melange = _journal_aleatoire(300, graine=2671, duree_s=3600)
    cas = [(exemple, 0, 120), (exemple, 0, 600), (exemple, 45, 75), (egalites, 0, 100), (egalites, 15, 45),
           (virgules, 0, 10), (virgules, 1.0, 3.0), (exemple, 500, 900), ([], 0, 60),
           (lundi, 0, 12 * 3600), (lundi, 2 * 3600, 3 * 3600), (lundi, 7 * 3600 + 1800, 8 * 3600 + 900),
           (melange, 0, 3600), (melange, 600, 1200)]
    alea = random.Random(267)
    alea.shuffle(egalites)
    cas.append((egalites, 0, 100))
    return cas


def _verifier_un_resultat_267(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, tuple) or len(r) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : attendu un TUPLE (moyenne, maximum).")
    if not _nombre(r[0]):
        raise AssertionError(f"{quoi} : la moyenne vaut {_court(r[0], 60)} : attendu un nombre.")
    if not _entier(r[1]):
        raise AssertionError(f"{quoi} : le maximum vaut {_court(r[1], 60)} ({type(r[1]).__name__}) : un nombre de clients "
                             "est un int.")


_LIGNES_FILE = 2_000_000        # le corrigé : 320 000 ; chaque appel comparé à tous les autres : 100 millions


def verifier_file_attente(f) -> None:
    cas = _cas_267()
    obtenus, vus = [], []
    for appels, debut, fin in cas:
        quoi = f"file_attente({_apercu(appels)}, {debut!r}, {fin!r})"
        avant = copy.deepcopy(appels)
        r = _appel(f, (appels, debut, fin), quoi)
        if appels != avant:
            raise AssertionError(f"{quoi} a modifié le journal reçu.")
        _verifier_un_resultat_267(r, quoi)
        obtenus.append(r)
        vus.append((appels, debut, fin))
        attendu = _ref_file(appels, debut, fin)
        if not (_proche(r[0], attendu[0]) and r[1] == attendu[1]):
            conseil = _premier_bug(obtenus, lambda bug: [_ref_file(a, d, e, bug) for a, d, e in vus], _BUGS_FILE,
                                   _listes(lambda x, y: _proche(x[0], y[0]) and x[1] == y[1]))
            if not conseil and _proche(r[0], attendu[0]):
                conseil = ("Votre moyenne est juste, pas votre maximum : balayez les événements triés, et à instant égal "
                           "faites partir avant d'arriver.")
            raise AssertionError(f"{quoi} vaut {r!r} au lieu de {attendu!r}. {conseil}".rstrip())

    grand = _journal_aleatoire(10_000, graine=2672)
    quoi = "file_attente(<10 000 appels>, 0, 43200)"
    r = _appel(f, (grand, 0, 43_200), quoi, lignes_max=_LIGNES_FILE,
               conseil_lent="Comparer chaque appel à tous les autres fait 100 millions de paires, et avancer seconde par "
                            "seconde 43 200 tours fois le nombre d'appels. La moyenne est une simple somme (une passe) ; "
                            "pour le maximum, triez UNE fois les événements (instant, ±1), puis un compteur.")
    _verifier_un_resultat_267(r, quoi)
    attendu = _ref_file(grand, 0, 43_200)
    if not (_proche(r[0], attendu[0]) and r[1] == attendu[1]):
        raise AssertionError(f"{quoi} vaut {r!r} au lieu de {attendu!r}.")

    _refus_journal(f, lambda appels: ((appels, 0, 100), {}), "file_attente(…, 0, 100)")
    journal = _journal((0, 5, 30))
    for debut, fin, attendu, conseil in ((10, 10, ValueError, "Une fenêtre vide (debut == fin) n'a pas de moyenne."),
                                         (20, 10, ValueError, "debut doit être AVANT fin."),
                                         ("0", 10, TypeError, "« 0 » est un texte, pas une heure."),
                                         (0, float("nan"), ValueError, "NaN n'est pas une heure de fin."),
                                         (0, None, TypeError, "None n'est pas une heure de fin.")):
        _doit_lever(f, (journal, debut, fin), f"file_attente({journal!r}, {debut!r}, {fin!r})", conseil, attendu)


# ============================================================================ défi 268 : erlang_c


def _ref_erlang(trafic, conseillers, bug: str = "") -> float:
    """Erlang C par la récurrence d'Erlang B ; `bug` rejoue une erreur classique."""
    a, n = trafic, conseillers
    if bug == "occupation":
        return min(1.0, a / n)
    if a >= n:
        return 1.0
    if bug == "poisson":                                       # P(au moins n appels en cours), sans file
        if a == 0:
            return 0.0
        return 1 - sum(math.exp(k * math.log(a) - a - math.lgamma(k + 1)) for k in range(n))
    b = 1.0
    for k in range(1, n + 1):
        b = a * b / (k + a * b)
    if bug == "erlang_b":
        return b
    if bug == "somme_n":                                       # le terme en N compté deux fois
        r = b * n / (n - a)
        return r / (1 + r)
    return n * b / (n - a * (1 - b))


_BUGS_ERLANG = {
    "erlang_b": "Vous renvoyez Erlang B (la probabilité qu'un appel soit REJETÉ quand il n'y a pas de file), pas Erlang C "
                "(la probabilité d'ATTENDRE quand il y en a une). Après la récurrence : C = N × B / (N - A × (1 - B)).",
    "occupation": "Vous renvoyez A / N, l'occupation des conseillers : la probabilité d'attendre est bien plus faible dans "
                  "une grande équipe (950 erlangs et 1 000 conseillers : 6,8 % des clients attendent, pas 95 %). C'est la "
                  "formule d'Erlang C qu'il faut.",
    "somme_n": "Votre somme S va jusqu'à k = N inclus : S = Σ A^k / k! s'arrête à N - 1, et le terme en N est remplacé "
               "par X = A^N / N! × N / (N - A).",
    "poisson": "Vous calculez la probabilité qu'au moins N appels soient en cours (une loi de Poisson) : l'idée est bonne, "
               "mais elle ignore la file. Erlang C : C = X / (S + X), avec X = A^N / N! × N / (N - A).",
}


def _cas_268() -> list[tuple]:
    cas = [(0.5, 1), (2, 3), (10, 12), (1, 100), (0.01, 1), (7, 8), (11.25, 15), (14.83, 15), (2.9999, 3), (0, 1),
           (0.0, 7), (3, 3), (4, 3), (15.9, 10), (40, 41), (100, 110), (380.5, 400), (950, 1000), (4999.5, 5000)]
    alea = random.Random(268)
    for _ in range(15):
        a = round(alea.uniform(0.1, 60), 3)
        cas.append((a, math.floor(a) + alea.randint(1, 20)))
    return cas


def verifier_erlang_c(f) -> None:
    cas = _cas_268()
    obtenus, vus = [], []
    for a, n in cas:
        quoi = f"erlang_c({a!r}, {n!r})"
        r = _appel(f, (a, n), quoi)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not _nombre(r):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) : attendu une probabilité (float).")
        obtenus.append(r)
        vus.append((a, n))
        attendu = _ref_erlang(a, n)
        if _proche(r, attendu):
            continue
        if a >= n:
            raise AssertionError(f"{quoi} vaut {r!r} au lieu de 1.0 : avec un trafic >= au nombre de conseillers, les "
                                 "appels arrivent plus vite qu'on ne peut les traiter, la file grossit sans fin et tout le "
                                 "monde attend. La formule n'a plus de sens (elle donne un nombre négatif ou plus grand que "
                                 "1) : renvoyez 1.0 avant de calculer.")
        if a == 0:
            raise AssertionError(f"{quoi} vaut {r!r} au lieu de 0.0 : sans trafic, personne n'attend.")
        conseil = _premier_bug(obtenus, lambda bug: [_ref_erlang(x, m, bug) for x, m in vus], _BUGS_ERLANG,
                               _listes(lambda x, y: _proche(x, y, 1e-7)))
        if not conseil and not 0 <= r <= 1:
            conseil = "Une probabilité est entre 0 et 1."
        raise AssertionError(f"{quoi} vaut {r!r} au lieu de {attendu!r}. {conseil}".rstrip())

    for args, attendu, conseil in (
            (("10", 12), TypeError, "« 10 » est un texte, pas un trafic."),
            ((True, 12), TypeError, "True n'est pas un trafic (excluez bool)."),
            ((-1, 12), ValueError, "Un trafic négatif n'existe pas."),
            ((float("nan"), 12), ValueError, "NaN n'est pas un trafic : `nan >= conseillers` est FAUX, testez-le."),
            ((float("inf"), 12), ValueError, "Un trafic infini n'est pas un nombre fini."),
            ((10, 0), ValueError, "Il faut au moins un conseiller."),
            ((10, 12.0), TypeError, "12.0 conseillers : un nombre de conseillers est un int."),
            ((10, True), TypeError, "True n'est pas un nombre de conseillers."),
            ((10, "12"), TypeError, "« 12 » est un texte.")):
        _doit_lever(f, args, f"erlang_c({args[0]!r}, {args[1]!r})", conseil, attendu)


# ============================================================================ défi 269 : conseillers_necessaires


class _Erlang:
    """Erlang C par la récurrence d'Erlang B, qui reprend là où le dernier calcul du même trafic s'est arrêté.

    Une recherche qui essaie N, puis N + 1, puis N + 2… ne coûte ainsi qu'une étape de récurrence par essai, même
    pour un trafic d'un million d'erlangs (des secondes prises pour des heures). Les nombres sont exactement ceux de
    `_ref_erlang` : les mêmes opérations, dans le même ordre.
    """

    def __init__(self) -> None:
        self.dernier: dict = {}

    def __call__(self, trafic, conseillers) -> float:
        a, n = trafic, conseillers
        if a >= n:
            return 1.0
        k, b = self.dernier.get(a, (0, 1.0))
        if k > n:
            k, b = 0, 1.0
        for j in range(k + 1, n + 1):
            b = a * b / (j + a * b)
        if len(self.dernier) > 10_000:
            self.dernier.clear()
        self.dernier[a] = (n, b)
        return n * b / (n - a * (1 - b))


_ERLANG = _Erlang()


def _erlang_pretee(trafic, conseillers) -> float:
    """La `erlang_c` que le correcteur prête à votre fichier pendant le défi 269 (elle vérifie ses arguments)."""
    if isinstance(trafic, bool) or not isinstance(trafic, (int, float)):
        raise TypeError(f"erlang_c : le trafic doit être un nombre, pas {type(trafic).__name__}")
    if isinstance(conseillers, bool) or not isinstance(conseillers, int):
        raise TypeError(f"erlang_c : le nombre de conseillers doit être un int, pas {conseillers!r} "
                        f"({type(conseillers).__name__}) : `math.floor(trafic) + 1` est un int, `trafic + 1` non")
    if not math.isfinite(trafic) or trafic < 0 or conseillers < 1:
        raise ValueError(f"erlang_c({trafic!r}, {conseillers!r}) : arguments hors du domaine")
    return _ERLANG(trafic, conseillers)


class _ErlangPretee:
    """Remplace `erlang_c` dans le module de `f` le temps d'une vérification, puis remet la vôtre."""

    def __init__(self, f) -> None:
        self.module = sys.modules.get(getattr(f, "__module__", None) or "")
        self.avant = None

    def __enter__(self):
        if self.module is not None and hasattr(self.module, "erlang_c"):
            self.avant = self.module.erlang_c
            self.module.erlang_c = _erlang_pretee
        return self

    def __exit__(self, *exc) -> None:
        if self.avant is not None:
            self.module.erlang_c = self.avant


def _ref_besoin(appels, duree, seuil=SEUIL_S, objectif=0.8, bug: str = "") -> int:
    """Le plus petit N qui tient l'objectif ; `bug` rejoue une erreur classique."""
    if bug == "minutes":
        trafic = appels * duree / 60
    elif bug == "sans_conversion":
        trafic = appels * duree
    else:
        trafic = appels * duree / 3600
    if trafic == 0:
        return 0
    if bug == "occupation":
        return math.ceil(trafic / 0.85)
    n = math.floor(trafic) + 1
    while True:
        c = _ERLANG(trafic, n)
        if bug == "sans_seuil":
            niveau = 1 - c
        elif bug == "sans_duree":
            niveau = 1 - c * math.exp(-(n - trafic) * seuil)
        else:
            niveau = 1 - c * math.exp(-(n - trafic) * seuil / duree)
        if niveau >= objectif:
            return n
        n += 1


_BUGS_BESOIN = {
    "minutes": "Votre trafic divise par 60 : la durée est en SECONDES et les appels sont comptés par HEURE. A = "
               "appels_par_heure × duree_moyenne_s / 3600.",
    "sans_conversion": "Votre trafic multiplie des appels par HEURE par une durée en SECONDES, sans conversion : A = "
                       "appels_par_heure × duree_moyenne_s / 3600 (190 appels de 281 s font 14,8 erlangs, pas 53 390).",
    "occupation": "Vous appliquez la règle de pouce « 85 % d'occupation » (N = ⌈A / 0,85⌉) : elle ignore l'objectif et "
                  "se trompe dans les deux sens (trop peu de monde dans une petite équipe, trop dans une grande). Cherchez "
                  "le plus petit N dont le niveau de service 1 - C × exp(-(N - A) × seuil / durée) tient l'objectif.",
    "sans_seuil": "Votre niveau de service est 1 - erlang_c(A, N), la part des clients qui n'attendent PAS du tout : un "
                  "client décroché en 15 s compte aussi. Multipliez C par exp(-(N - A) × seuil_s / duree_moyenne_s), la "
                  "part de ceux qui attendent plus que le seuil.",
    "sans_duree": "Dans l'exponentielle, divisez par la durée moyenne : exp(-(N - A) × seuil_s / duree_moyenne_s). Sans "
                  "elle, l'attente se compte en « durées d'appel », et vous sous-estimez nettement les besoins.",
}


def _cas_269() -> list[tuple]:
    """(appels_par_heure, duree_moyenne_s, seuil_s ou None, objectif ou None), du plus petit au plus grand."""
    return [(30, 240, None, None), (0, 300, None, None), (1, 30, None, None), (45, 281, None, None),
            (12, 600, 0, 0.9), (55.5, 312.5, 15, 0.75), (100, 180, 30, 0.95), (190, 281, None, None),
            (190, 281, None, 0.5), (190, 281, 60, None), (204, 281.2, 20, 0.8), (400, 240, 20, 0.8),
            (1900, 281, None, None), (5000, 200, 10, 0.9)]


_LIGNES_BESOIN = 200_000


def verifier_conseillers_necessaires(f) -> None:
    with _ErlangPretee(f):
        _verifier_conseillers_necessaires(f)


def _verifier_conseillers_necessaires(f) -> None:
    obtenus, vus, quois = [], [], []
    for appels, duree, seuil, objectif in _cas_269():        # toutes les réponses d'abord : elles nomment mieux l'erreur
        kwargs = {}
        if seuil is not None:
            kwargs["seuil_s"] = seuil
        if objectif is not None:
            kwargs["objectif"] = objectif
        texte = ", ".join([repr(appels), repr(duree)] + [f"{k}={v!r}" for k, v in kwargs.items()])
        quoi = f"conseillers_necessaires({texte})"
        r = _appel(f, (appels, duree), quoi, kwargs=kwargs, lignes_max=_LIGNES_BESOIN)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not _entier(r):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) : un nombre de conseillers est un "
                                 "int.")
        obtenus.append(r)
        vus.append((appels, duree, SEUIL_S if seuil is None else seuil, 0.8 if objectif is None else objectif))
        quois.append(quoi)
    for r, (appels, duree, s, o), quoi in zip(obtenus, vus, quois):
        attendu = _ref_besoin(appels, duree, s, o)
        if r == attendu:
            continue
        trafic = appels * duree / 3600
        if attendu == 0:
            raise AssertionError(f"{quoi} vaut {r!r} au lieu de 0 : personne n'appelle, personne au téléphone.")
        conseil = _premier_bug(obtenus, lambda bug: [_ref_besoin(*v, bug=bug) for v in vus], _BUGS_BESOIN,
                               lambda x, y: x == y)
        if not conseil and r <= trafic:
            conseil = (f"Avec {r} conseillers pour {trafic:.2f} erlangs, les appels arrivent plus vite qu'on ne les traite : "
                       "la file grossit sans fin. N doit dépasser le trafic (commencez à ⌊A⌋ + 1).")
        if not conseil:
            conseil = ("Le plus petit N dont le niveau de service 1 - erlang_c(A, N) × exp(-(N - A) × seuil_s / "
                       "duree_moyenne_s) est >= objectif (>=, pas >).")
        raise AssertionError(f"{quoi} vaut {r!r} au lieu de {attendu!r}. {conseil}".rstrip())

    for args, kwargs, attendu, conseil in (
            ((190, 281), {"objectif": 1}, ValueError, "100 % ne s'atteint jamais : la boucle ne s'arrêterait pas."),
            ((190, 281), {"objectif": 80}, ValueError, "80 au lieu de 0.8 : un objectif est entre 0 et 1."),
            ((190, 281), {"objectif": 0}, ValueError, "Un objectif de 0 % n'a pas de sens (strictement entre 0 et 1)."),
            ((190, 0), {}, ValueError, "Une durée moyenne de 0 s : le trafic ne veut plus rien dire."),
            ((-10, 281), {}, ValueError, "Un nombre d'appels négatif n'existe pas."),
            ((190, 281), {"seuil_s": -5}, ValueError, "Un seuil négatif n'a pas de sens."),
            (("190", 281), {}, TypeError, "« 190 » est un texte."),
            ((True, 281), {}, TypeError, "True n'est pas un nombre d'appels."),
            ((190, float("nan")), {}, ValueError, "NaN n'est pas une durée."),
            ((190, 281), {"objectif": "0.8"}, TypeError, "« 0.8 » est un texte.")):
        texte = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
        _doit_lever(f, args, f"conseillers_necessaires({texte})", conseil, attendu, kwargs=kwargs)


# ============================================================================ défi 270 : planning


def _ref_planning(besoins, duree, bug: str = "") -> list[int]:
    """Le planning glouton « le plus tard possible » ; `bug` rejoue une erreur classique."""
    n = len(besoins)
    debuts = [0] * n
    partent = [0] * (n + duree + 2)
    presents = 0
    for h in range(n):
        if bug == "sans_cumul":
            presents = debuts[h]
        else:
            presents += partent[h]
        manque = besoins[h] - presents
        if manque <= 0:
            continue
        if bug == "plus_tot":
            s = max(0, h - duree + 1)
        elif bug == "hors_horaires":
            s = h
        else:
            s = min(h, n - duree)
        debuts[s] += manque
        presents += manque
        couvre = {"court": duree - 1, "long": duree + 1}.get(bug, duree)
        if s + couvre > h:
            partent[s + couvre] -= manque
    return debuts


def _couverture(debuts, duree: int, n: int) -> list[int]:
    return [sum(debuts[s] for s in range(max(0, h - duree + 1), h + 1)) for h in range(n)]


_BUGS_PLANNING = {
    "plus_tot": "Vous faites commencer les renforts le plus TÔT possible (à h - duree + 1) : leurs premières heures tombent "
                "sur des heures déjà couvertes, et il en faut d'autres ensuite. Faites-les commencer le plus TARD "
                "possible : à h (ou à len(besoins) - duree en fin de journée).",
    "sans_cumul": "Vous ne comptez comme présents que ceux qui COMMENCENT à l'heure h : ceux qui ont commencé dans les "
                  "duree - 1 heures précédentes sont encore là. Suivez les présents, et leurs départs "
                  "(`partent[s + duree] -= k`).",
    "court": "Vous faites partir les conseillers une heure trop tôt : une vacation commencée à s couvre les heures s, …, "
             "s + duree - 1 (duree heures), et part à s + duree.",
    "long": "Vous gardez les conseillers une heure de trop : une vacation commencée à s couvre les heures s à "
            "s + duree - 1 seulement, et part à s + duree.",
    "hors_horaires": "Une vacation commence à l'heure h même quand elle finirait après la fermeture : en fin de journée, la "
                     "dernière vacation qui tient commence à len(besoins) - duree.",
}

_LUNDI = [6, 14, 20, 18, 10, 9, 10, 16, 16, 9, 7, 5]


def _cas_270() -> list[tuple]:
    cas = [(_LUNDI, DUREE_VACATION_H), ([2, 3, 3, 1, 4], 2), (_LUNDI, 6), (_LUNDI, 1), (_LUNDI, 12), ([], 3),
           ([0, 0, 0], 1), ([5], 1), ([1, 2, 3, 4, 5, 6], 6), ([3, 0, 0, 3], 2), ([0, 0, 0, 2], 2), ([4, 1, 1, 1, 4], 3),
           ([1, 5, 1, 5, 1, 5, 1], 2), ((3, 1, 4, 1, 5, 9, 2, 6), 3), ([0, 7, 0, 0, 0, 0], 4)]
    alea = random.Random(270)
    for _ in range(40):
        n = alea.randint(1, 24)
        cas.append(([alea.randint(0, 12) for _ in range(n)], alea.randint(1, n)))
    return cas


def verifier_planning(f) -> None:
    obtenus, vus = [], []
    for besoins, duree in _cas_270():
        quoi = f"planning({_court(besoins, 80)}, {duree!r})"
        avant = copy.deepcopy(besoins)
        r = _appel(f, (besoins, duree), quoi)
        if besoins != avant:
            raise AssertionError(f"{quoi} a modifié la liste des besoins.")
        n = len(besoins)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, list) or len(r) != n:
            raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : attendu une liste de {n} int (combien commencent à "
                                 "chaque heure).")
        if not all(_entier(x) and x >= 0 for x in r):
            raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : chaque case est un nombre de conseillers, un int >= 0.")
        obtenus.append(r)
        vus.append((list(besoins), duree))

        def conseil_rejeu() -> str:
            return _premier_bug(obtenus, lambda bug: [_ref_planning(b, d, bug) for b, d in vus], _BUGS_PLANNING,
                                lambda x, y: x == y)

        tard = [s for s in range(n) if r[s] > 0 and s + duree > n]
        if tard:
            s = tard[0]
            raise AssertionError(f"{quoi} vaut {r!r} : {r[s]} conseiller(s) commencent à l'heure {s}, et leur vacation de "
                                 f"{duree} h finirait après la fermeture (heure {n}). "
                                 f"{conseil_rejeu() or _BUGS_PLANNING['hors_horaires']}".rstrip())
        couverture = _couverture(r, duree, n)
        trous = [h for h in range(n) if couverture[h] < besoins[h]]
        if trous:
            h = trous[0]
            raise AssertionError(f"{quoi} vaut {r!r} : à l'heure {h}, {couverture[h]} conseiller(s) en vacation pour un "
                                 f"besoin de {besoins[h]}. {conseil_rejeu()}".rstrip())
        minimum = sum(_ref_planning(list(besoins), duree))
        if sum(r) > minimum:
            raise AssertionError(f"{quoi} vaut {r!r} : {sum(r)} vacations, alors que {minimum} suffisent (par exemple "
                                 f"{_ref_planning(list(besoins), duree)!r}). {conseil_rejeu() or _BUGS_PLANNING['plus_tot']}"
                                 .rstrip())
        if sum(r) < minimum:                                  # impossible si les heures sont couvertes
            raise AssertionError(f"{quoi} vaut {r!r} : moins de vacations que le minimum, c'est qu'une heure n'est pas "
                                 "couverte.")

    for args, attendu, conseil in (
            (([1, 2.5, 3], 2), TypeError, "2,5 conseillers n'existent pas : un besoin est un int (arrondissez avant)."),
            (([1, -1, 3], 2), ValueError, "Un besoin négatif n'a pas de sens."),
            (([1, True, 3], 2), TypeError, "True n'est pas un nombre de conseillers."),
            (("123", 2), TypeError, "« 123 » est un texte, pas une liste de besoins."),
            (([1, 2, 3], 0), ValueError, "Une vacation d'au moins une heure."),
            (([1, 2, 3], 2.0), TypeError, "2.0 heures : une durée de vacation est un int."),
            (([1, 2, 3], True), TypeError, "True n'est pas une durée."),
            (([1, 2, 3], 4), ValueError, "Une vacation de 4 h ne tient pas dans une journée de 3 h.")):
        _doit_lever(f, args, f"planning({args[0]!r}, {args[1]!r})", conseil, attendu)


# ============================================================================ défi 271 : simuler


def _ref_horaires(conseillers) -> list[tuple]:
    if isinstance(conseillers, int):
        return [(0, math.inf)] * conseillers
    return [tuple(h) for h in conseillers]


def _ref_simuler_tas(appels, conseillers) -> list[dict]:
    """La simulation de référence, avec un tas (rapide)."""
    tas = [(debut, k, fin) for k, (debut, fin) in enumerate(_ref_horaires(conseillers))]
    heapq.heapify(tas)
    resultat: list[dict] = [{} for _ in appels]
    for i in sorted(range(len(appels)), key=lambda i: appels[i]["arrivee"]):
        appel = appels[i]
        arrivee = appel["arrivee"]
        while tas and max(arrivee, tas[0][0]) >= tas[0][2]:
            heapq.heappop(tas)
        if tas and max(arrivee, tas[0][0]) - arrivee <= appel["patience"]:
            libre, k, fin = heapq.heappop(tas)
            decroche = max(arrivee, libre)
            heapq.heappush(tas, (decroche + appel["duree"], k, fin))
            resultat[i] = {**appel, "decroche": decroche, "fin": decroche + appel["duree"]}
        else:
            resultat[i] = {**appel, "decroche": None, "fin": arrivee + appel["patience"]}
    return resultat


def _ref_simuler(appels, conseillers, bug: str = "") -> list[dict]:
    """La même simulation, écrite autrement (chaque conseiller passé en revue) ; `bug` rejoue une erreur classique."""
    horaires_ = _ref_horaires(conseillers)
    if bug == "un_seul":                                       # un seul client en ligne à la fois
        horaires_ = horaires_[:1]
    libres = [debut for debut, _ in horaires_]
    if bug == "sans_tri":
        ordre = list(range(len(appels)))
    else:
        ordre = sorted(range(len(appels)), key=lambda i: appels[i]["arrivee"])
    resultat: list[dict] = [{} for _ in appels]
    for i in ordre:
        appel = appels[i]
        arrivee = appel["arrivee"]
        meilleur = None
        for k, (_, fin) in enumerate(horaires_):
            moment = max(arrivee, libres[k])
            if bug == "fin_ignoree":
                en_service = True
            elif bug == "fin_incluse":
                en_service = moment <= fin
            else:
                en_service = moment < fin
            if en_service:
                cle = (moment, k) if bug == "plus_petit_numero" else (moment, libres[k], k)
                if meilleur is None or cle < meilleur:
                    meilleur = cle
        accepte = False
        if meilleur is not None:
            attente = meilleur[0] - arrivee
            if bug == "sans_patience":
                accepte = True
            elif bug == "patience_stricte":
                accepte = attente < appel["patience"]
            else:
                accepte = attente <= appel["patience"]
        if accepte:
            moment, k = meilleur[0], meilleur[-1]
            libres[k] = moment + appel["duree"]
            resultat[i] = {**appel, "decroche": moment, "fin": moment + appel["duree"]}
        else:
            if bug == "abandon_occupe" and meilleur is not None:
                libres[meilleur[-1]] = meilleur[0] + appel["duree"]
            resultat[i] = {**appel, "decroche": None, "fin": arrivee + appel["patience"]}
    if bug == "ordre_trie":
        return [resultat[i] for i in ordre]
    return resultat


_BUGS_SIMULER = {
    "sans_tri": "Vous servez les appels dans l'ordre de la LISTE : premier arrivé, premier servi, c'est l'ordre des "
                "ARRIVÉES (`sorted(range(len(appels)), key=…)` ; `sorted` est stable, à arrivée égale l'ordre de la liste "
                "est gardé).",
    "ordre_trie": "Vos résultats sont rangés par arrivée : la réponse suit l'ordre de la liste reçue (traitez dans un "
                  "ordre, rangez dans l'autre : `resultat[i] = …`).",
    "abandon_occupe": "Un client qui raccroche occupe quand même un conseiller : il n'a jamais été décroché, le conseiller "
                      "reste libre pour le suivant (ne le sortez du tas que si le client est décroché).",
    "patience_stricte": "Un client dont l'attente vaut EXACTEMENT sa patience raccroche chez vous : il est encore décroché "
                        "(attente <= patience). Il ne raccroche que s'il devait attendre strictement plus.",
    "sans_patience": "Personne ne raccroche chez vous : un client qui devrait attendre plus que sa 'patience' raccroche à "
                     "arrivee + patience, avec 'decroche' à None.",
    "un_seul": "Vous ne servez qu'un client à la fois : avec plusieurs conseillers, plusieurs clients sont en ligne en "
               "même temps. Chaque conseiller a SA disponibilité (un tas de tuples (libre à partir de, numéro, fin de "
               "service)), et le client prend celui qui est libre le plus tôt.",
    "plus_petit_numero": "Quand plusieurs conseillers sont libres, vous prenez le plus petit numéro : la règle est celui "
                         "qui est libre depuis le plus LONGTEMPS (le plus petit « libre à partir de »), le numéro ne "
                         "départage que les égalités. Un tas de tuples (libre à partir de, numéro, fin) le fait seul.",
    "fin_ignoree": "Vos conseillers décrochent après la fin de leur service : un conseiller dont le service finit à `fin` "
                   "ne prend plus d'appel à partir de `fin` (il termine seulement celui en cours).",
    "fin_incluse": "Un conseiller décroche chez vous à la seconde même où son service finit : il décroche pendant "
                   "[debut, fin[, pas à `fin` pile.",
}


def _a(arrivee, duree, patience, **autres) -> dict:
    return {"arrivee": arrivee, "duree": duree, "patience": patience, **autres}


def _cas_271() -> list[tuple]:
    """(appels, conseillers)."""
    desordre = [_a(50, 30, 100, client="tard"), _a(0, 60, 100, client="tôt")]
    exemple = [_a(0, 300, 60, client="A"), _a(10, 120, 60, client="B"), _a(20, 200, 300, client="C"),
               _a(40, 60, 30, client="D"), _a(130, 100, 200, client="E")]
    egalite_arrivee = [_a(0, 50, 100, client="premier"), _a(0, 50, 100, client="second"), _a(0, 50, 49, client="pressé")]
    libere_a_l_arrivee = [_a(0, 30, 0), _a(30, 30, 0), _a(60, 10, 5)]
    patience_juste = [_a(0, 100, 10), _a(10, 50, 90), _a(20, 50, 129), _a(25, 10, 174)]
    plus_longtemps = [_a(0, 10, 100), _a(20, 100, 100), _a(30, 5, 100), _a(60, 10, 200)]
    services = [(0, 50), (0, 1000)]
    arrivee_tardive = [_a(0, 20, 150), _a(0, 20, 50), _a(130, 20, 10)]
    fin_pile = [_a(0, 100, 0), _a(50, 10, 100), _a(100, 10, 0)]
    abandons = [_a(0, 300, 10), _a(5, 300, 10), _a(8, 300, 400), _a(9, 30, 5), _a(320, 30, 0)]
    zero = [_a(0, 0, 0), _a(0, 0, 0), _a(0, 5, 0), _a(3, 1, 2)]
    virgules = [_a(0.5, 10.25, 3.0), _a(1.0, 4.5, 9.75), _a(1.0, 2.0, 0.5), _a(14.75, 1.0, 0.0)]
    melange = generer_demande((70, 110), duree_mediane_s=300, patience_moyenne_s=200, graine=2711)
    random.Random(2712).shuffle(melange)
    cas = [(exemple, 2), (desordre, 1), (exemple, [(0, 100), (0, 1000)]), (exemple, 1),
           (egalite_arrivee, 1), (list(reversed(egalite_arrivee)), 1), (libere_a_l_arrivee, 1), (patience_juste, 1),
           (plus_longtemps, services), (arrivee_tardive, [(100, 200)]), (fin_pile, [(0, 100)]),
           (fin_pile, [(0, 100), (200, math.inf)]), (abandons, 2), (abandons, [(0, 310), (0, 320)]), (zero, 1),
           (virgules, 1), (exemple, 0), (exemple, []), ([], 3), (melange, 5),
           (melange, horaires([3, 1, 0, 2], 2)), (melange, [(0, 3600), (1800, 7200), (0, 7200), (3600, 5400)])]
    for graine in range(3):
        demande = generer_demande((40, 90, 60), graine=2713 + graine)
        cas.append((demande, horaires([2 + graine, 1, 0], 2)))
    cas.append((demande_exemple(), horaires(planning_actuel())))
    cas.append((demande_exemple(), 13))
    return cas


def _resume(appels) -> list:
    return [(a.get("decroche"), a.get("fin")) for a in appels]


def _simulations_egales(a: list, b: list) -> bool:
    return len(a) == len(b) and all(_proche(x.get("decroche"), y.get("decroche")) and _proche(x.get("fin"), y.get("fin"))
                                    for x, y in zip(a, b))


def _verifier_une_simulation(f, appels, conseillers, quoi: str, obtenus: list, vus: list,
                             lignes_max: int | None = None, conseil_lent: str = "") -> None:
    avant_appels, avant_conseillers = copy.deepcopy(appels), copy.deepcopy(conseillers)
    r = _appel(f, (appels, conseillers), quoi, lignes_max=lignes_max, conseil_lent=conseil_lent)
    if appels != avant_appels:
        raise AssertionError(f"{quoi} a modifié les appels reçus : la réponse est faite de COPIES (`dict(appel)` ou "
                             "`{**appel, …}`), complétées de 'decroche' et 'fin'.")
    if conseillers != avant_conseillers:
        raise AssertionError(f"{quoi} a modifié la liste des horaires reçue.")
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, list) or len(r) != len(appels):
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : attendu une liste de {len(appels)} dict, un par appel.")
    for i, (appel, sortie) in enumerate(zip(appels, r)):
        if not isinstance(sortie, dict):
            raise AssertionError(f"{quoi} : le résultat n°{i} est {_court(sortie, 60)} : attendu un dict.")
        if sortie is appel:
            raise AssertionError(f"{quoi} : le résultat n°{i} est le dict reçu lui-même : renvoyez une COPIE complétée.")
        for cle, valeur in appel.items():
            if cle not in sortie or sortie[cle] != valeur:
                autre = next((j for j, x in enumerate(appels) if all(sortie.get(c) == v for c, v in x.items())), None)
                if autre is not None:
                    raise AssertionError(f"{quoi} : le résultat n°{i} est celui de l'appel n°{autre}. "
                                         f"{_BUGS_SIMULER['ordre_trie']}")
                raise AssertionError(f"{quoi} : le résultat n°{i} n'a pas gardé {cle!r}: {valeur!r} (chaque résultat est "
                                     "une copie de l'appel reçu, complétée de 'decroche' et 'fin').")
        for cle in ("decroche", "fin"):
            if cle not in sortie:
                raise AssertionError(f"{quoi} : le résultat n°{i} n'a pas de clé {cle!r}.")
        if sortie["decroche"] is not None and not _nombre(sortie["decroche"]):
            raise AssertionError(f"{quoi} : 'decroche' du résultat n°{i} vaut {_court(sortie['decroche'], 40)} : un "
                                 "nombre, ou None si le client a raccroché.")
        if not _nombre(sortie["fin"]):
            raise AssertionError(f"{quoi} : 'fin' du résultat n°{i} vaut {_court(sortie['fin'], 40)} : attendu un nombre.")
    obtenus.append(r)
    vus.append((appels, conseillers))
    attendu = _ref_simuler_tas(appels, conseillers)
    for i, (sortie, juste) in enumerate(zip(r, attendu)):
        if _proche(sortie["decroche"], juste["decroche"]) and _proche(sortie["fin"], juste["fin"]):
            continue
        conseil = _premier_bug(obtenus, lambda bug: [_ref_simuler(a, c, bug) for a, c in vus], _BUGS_SIMULER,
                               _listes(_simulations_egales))
        if not conseil and juste["decroche"] is None and sortie["decroche"] is not None \
                and sortie["fin"] != sortie["decroche"] + appels[i]["duree"]:
            conseil = "Un appel décroché finit à decroche + duree."
        raise AssertionError(
            f"{quoi} : l'appel n°{i} ({_court(appels[i], 90)}) donne decroche = {sortie['decroche']!r}, fin = "
            f"{sortie['fin']!r} au lieu de decroche = {juste['decroche']!r}, fin = {juste['fin']!r}. {conseil}".rstrip())


_LIGNES_SIMULER = 1_600_000     # le corrigé : 550 000 ; 100 conseillers passés en revue à chaque appel : 2 à 3 millions


def verifier_simuler(f) -> None:
    obtenus, vus = [], []
    for appels, conseillers in _cas_271():
        quoi = f"simuler({_apercu(appels)}, {_court(conseillers, 60)})"
        _verifier_une_simulation(f, appels, conseillers, quoi, obtenus, vus)

    grand = generer_demande((1200,) * 12, duree_mediane_s=240, patience_moyenne_s=300, graine=2719)
    _verifier_une_simulation(
        f, grand, 100, f"simuler(<{len(grand)} appels>, 100)", [], [], lignes_max=_LIGNES_SIMULER,
        conseil_lent="Une horloge qui avance seconde par seconde fait 43 200 tours, comparer chaque appel aux précédents "
                     "des centaines de millions, et passer en revue les 100 conseillers à chaque appel 1,4 million. Un tas "
                     "(`heapq`) de tuples (libre à partir de, numéro, fin de service) donne le bon conseiller en "
                     "O(log n).")

    appel = [_a(0, 60, 30)]
    for args, attendu, conseil in (
            (("appels", 2), TypeError, "`appels` est une LISTE de dict."),
            (([(0, 60, 30)], 2), TypeError, "Chaque appel est un dict {'arrivee', 'duree', 'patience'}."),
            (([{"arrivee": 0, "duree": 60}], 2), ValueError, "Un appel sans 'patience' : ValueError."),
            (([_a(0, "60", 30)], 2), TypeError, "« 60 » est un texte, pas une durée."),
            (([_a(0, 60, -1)], 2), ValueError, "Une patience négative n'existe pas."),
            (([_a(0, -5, 30)], 2), ValueError, "Une durée négative n'existe pas."),
            (([_a(float("nan"), 60, 30)], 2), ValueError, "NaN n'est pas une heure d'arrivée."),
            ((appel, -1), ValueError, "Un nombre de conseillers négatif n'existe pas."),
            ((appel, True), TypeError, "True n'est pas un nombre de conseillers (excluez bool)."),
            ((appel, 2.0), TypeError, "2.0 conseillers : un int, ou une liste d'horaires."),
            ((appel, "2"), TypeError, "« 2 » est un texte."),
            ((appel, [(0,)]), TypeError, "Un horaire est un couple (début, fin)."),
            ((appel, [(100, 50)]), ValueError, "Un service qui finit (50) avant de commencer (100)."),
            ((appel, [("0", 100)]), TypeError, "« 0 » est un texte, pas une heure.")):
        _doit_lever(f, args, f"simuler({_court(args[0], 60)}, {args[1]!r})", conseil, attendu)


# ---------------------------------------------------------------------------- la série

SERIE = Serie(
    code="44",
    titre="Décrocher à temps : mesurer l'attente, la loi de Little, Erlang C, dimensionner, planifier, simuler",
    module_eleve="defis.serie_44",
    module_corrige="defis.corrige_serie_44",
    defis=(
        Defi(266, "mesurer", "🟢", "—", verifier_mesurer),
        Defi(267, "file_attente", "🟡", "—", verifier_file_attente),
        Defi(268, "erlang_c", "🟡", "—", verifier_erlang_c),
        Defi(269, "conseillers_necessaires", "🟡", "—", verifier_conseillers_necessaires),
        Defi(270, "planning", "🟡", "—", verifier_planning),
        Defi(271, "simuler", "🔴", "—", verifier_simuler),
    ),
)

"""Vérifications automatiques de la série 37 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est la comptable, qui recompte tout au centime près, et la banque, qui envoie des relevés
dans le désordre :

- partout, il choisit des montants où chaque erreur classique se voit : un demi-centime (498.5), un prix que le
  float déforme (`1.005 * 100` vaut 100.49999999999999), une période de 28 ou 29 jours, des restes à égalité
  que le float sépare (0.3 contre 0.30000000000000004), une Ascension le 1er mai (2008), un refus le
  31 décembre, un numéro de facture tapé avec un chiffre de trop, un relevé qui n'est pas dans l'ordre des dates ;
- quand votre résultat est faux, il rejoue les erreurs classiques (30 jours par mois, fin incluse, `round()`
  qui arrondit au pair, float, centimes donnés au premier ou au dernier, TVA ligne par ligne, taux triés comme
  du texte, remise déduite d'une seule ligne, Pentecôte un dimanche, fériés de la mauvaise année, essais le
  même jour, numéro d'un autre client, factures soldées par numéro et pas par échéance…) pour nommer la vôtre ;
- au défi 225, il répartit un montant sur 100 000 poids, et au défi 229 il rapproche 20 000 factures et autant
  de paiements, en comptant les lignes exécutées dans votre fichier : une boucle dans une boucle se voit, quelle
  que soit la machine ;
- et il vérifie que vos fonctions ne modifient pas ce qu'elles reçoivent.

Les montants sont des centimes entiers : ici, aucune tolérance. Un centime de trop est un centime de trop.
"""

from __future__ import annotations

import copy
import math
import random
import re
import sys
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation

from .facturation import (DELAIS, GRACE, Creance, Paiement, creances_exemple, generer_creances, generer_lignes,
                          generer_paiements, lignes_exemple, paiements_exemple)
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
    """Levée dans VOTRE fonction quand elle travaille trop (BaseException : `except Exception` ne l'attrape pas)."""


def _trace(f, appel, lignes_max: int | None):
    """Le résultat de `appel()` ; arrête net au-delà de `lignes_max` lignes exécutées dans le fichier de f."""
    code = getattr(f, "__code__", None)
    if code is None or lignes_max is None:
        return appel()
    fichier, compte = code.co_filename, [0]

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


_CONSEIL_LENT = "Votre fonction a été arrêtée : elle exécute beaucoup trop de lignes (une boucle dans une boucle ?)."


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = 2_000_000, conseil_lent: str = "",
           kwargs: dict | None = None):
    """f(*args, **kwargs), en transformant toute erreur en message lisible."""
    kwargs = kwargs or {}
    try:
        return _trace(f, lambda: f(*args, **kwargs), lignes_max)
    except NotImplementedError as exc:
        _pas_encore(f, exc)
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None


def _pas_encore(f, exc: NotImplementedError) -> None:
    """Votre fonction est écrite, mais elle appelle une fonction qui ne l'est pas encore."""
    cadres = []
    tb = exc.__traceback__
    while tb is not None:
        cadres.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    nom = getattr(f, "__name__", "")
    if nom and nom in cadres and cadres[-1] != nom:
        numero = _NUMEROS.get(cadres[-1])
        defi = f" (défi {numero})" if numero else ""
        raise AssertionError(f"`{nom}` appelle `{cadres[-1]}`, qui n'est pas encore écrite (raise "
                             f"NotImplementedError) : faites d'abord ce défi-là{defi}.") from None


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu=ValueError, kwargs: dict | None = None) -> None:
    """Exige que f(*args, **kwargs) lève `attendu`."""
    try:
        resultat = _trace(f, lambda: f(*args, **(kwargs or {})), 2_000_000)
    except NotImplementedError as exc:
        _pas_encore(f, exc)
        raise
    except attendu:
        return
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {_CONSEIL_LENT}") from None
    except InvalidOperation as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__} : decimal.InvalidOperation "
                             "n'est PAS une ValueError. Attrapez-la (`except InvalidOperation:`) et levez une "
                             f"ValueError à la place. {conseil}".rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _intact(avant, apres, quoi: str, nom: str) -> None:
    if avant != apres:
        raise AssertionError(f"{quoi} a modifié {nom} reçu(s) (avant : {_court(avant, 100)}, après : "
                             f"{_court(apres, 100)}). Une fonction ne touche pas à ce qu'on lui prête : travaillez sur "
                             "une copie, ou construisez un nouvel objet (`sorted(...)` plutôt que `.sort()`).")


def _pas_none(resultat, quoi: str) -> None:
    if resultat is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")


def _entier(valeur, quoi: str, nom: str = "le résultat") -> None:
    if type(valeur) is not int:
        raise AssertionError(f"{quoi} : {nom} vaut {_court(valeur)} ({type(valeur).__name__}) ; attendu un int de "
                             "centimes. Un montant se compte en ENTIERS : pas de float, pas de Decimal, pas de bool.")


def _demi_haut(x: Decimal) -> int:
    return int(x.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _fonction_eleve(f, nom: str):
    """La fonction `nom` du fichier de l'élève (celui où f est écrite), ou None."""
    fonction = getattr(f, "__globals__", {}).get(nom)
    return fonction if callable(fonction) else None


# ---------------------------------------------------------------------------- défi 224 : prorata


def _ref_prorata(montant: int, debut: date, fin: date, jour: date, bug: str = "") -> int:
    """La bonne réponse ; `bug` rejoue une erreur classique."""
    periode = 30 if bug == "trente" else (fin - debut).days
    restants = (fin - jour).days
    if bug == "fin_incluse":
        periode, restants = periode + 1, restants + 1
    if bug == "restants_plus_un":
        restants += 1
    if bug == "tronque":
        return montant * restants // periode
    if bug == "banquier":
        return round(montant * restants / periode)
    if bug == "float_divise":
        return math.floor(montant / periode * restants + 0.5)
    if bug == "float_fraction":
        return math.floor(montant * (restants / periode) + 0.5)
    return (2 * montant * restants + periode) // (2 * periode)


_BUGS_PRORATA = {
    "trente": "Vous divisez par 30 jours. Comptez les jours RÉELS de la période : (fin - debut).days (28 en février "
              "2026, 29 en février 2028, 365 ou 366 pour un an).",
    "fin_incluse": "Vous comptez un jour de trop : `fin` est EXCLUE, c'est le premier jour de la période suivante. La "
                   "période compte (fin - debut).days jours, et il en reste (fin - jour).days, sans + 1.",
    "restants_plus_un": "Vous comptez un jour restant de trop : du `jour` à `fin` (exclue), il reste (fin - jour).days "
                        "jours. Le dernier jour de la période est la veille de `fin`.",
    "float_divise": "Vous passez par un float : montant / periode * restants donne par exemple 499.49999999999994 "
                    "au lieu de 499.5, et l'arrondi tombe du mauvais côté. Restez en entiers : "
                    "(2 * montant * restants + periode) // (2 * periode).",
    "float_fraction": "Vous passez par un float : restants / periode n'est pas exact (17 / 28 = 0.6071428571428571…), "
                      "et l'arrondi d'un demi-centime tombe du mauvais côté. Restez en entiers : "
                      "(2 * montant * restants + periode) // (2 * periode).",
    "banquier": "`round()` arrondit les demis au nombre PAIR (round(498.5) vaut 498 : l'« arrondi du banquier »). Ici, "
                "un demi-centime s'arrondit vers le HAUT : (2 * montant * restants + periode) // (2 * periode).",
    "tronque": "Vous tronquez (int() ou //) au lieu d'arrondir au plus proche : 856.71 centimes doivent donner 857, "
               "pas 856.",
}


def _un_prorata(f, args: tuple, quoi: str) -> None:
    attendu = _ref_prorata(*args)
    obtenu = _appel(f, args, quoi)
    _pas_none(obtenu, quoi)
    if obtenu == attendu and type(obtenu) is not int:
        _entier(obtenu, quoi)
    if obtenu != attendu:
        montant, debut, fin, jour = args
        detail = (f"{quoi} renvoie {_court(obtenu)} au lieu de {attendu} (période de {(fin - debut).days} jours, "
                  f"{(fin - jour).days} restants : {montant} × {(fin - jour).days} / {(fin - debut).days} = "
                  f"{montant * (fin - jour).days / (fin - debut).days:.4f}).")
        for bug, conseil in _BUGS_PRORATA.items():
            if obtenu == _ref_prorata(*args, bug=bug):
                raise AssertionError(f"{detail} {conseil}")
        raise AssertionError(f"{detail} Arrondissez au centime le plus proche, les demis vers le haut.")
    _entier(obtenu, quoi)


def verifier_prorata(f) -> None:
    d = date
    cas = [
        (2999, d(2026, 2, 1), d(2026, 3, 1), d(2026, 2, 17)),     # 12 / 28 → 1285.29
        (1999, d(2026, 2, 1), d(2026, 3, 1), d(2026, 2, 17)),     # 856.71 → 857 (tronquer donne 856)
        (2999, d(2026, 2, 1), d(2026, 3, 1), d(2026, 2, 1)),      # tout
        (2999, d(2026, 2, 1), d(2026, 3, 1), d(2026, 3, 1)),      # rien
        (2999, d(2028, 2, 1), d(2028, 3, 1), d(2028, 2, 17)),     # bissextile : 13 / 29
        (23988, d(2026, 1, 1), d(2027, 1, 1), d(2026, 7, 1)),     # un an : 184 / 365
        (23988, d(2028, 1, 1), d(2029, 1, 1), d(2028, 7, 1)),     # un an bissextile : 184 / 366
        (999, d(2026, 4, 1), d(2026, 5, 1), d(2026, 4, 16)),      # 499.5 : le float divisé donne 499
        (2450, d(2026, 2, 1), d(2026, 3, 1), d(2026, 2, 12)),     # 1487.5 : montant * (17 / 28) donne 1487
        (997, d(2026, 4, 1), d(2026, 5, 1), d(2026, 4, 16)),      # 498.5 : round() donne 498
        (0, d(2026, 1, 1), d(2026, 2, 1), d(2026, 1, 20)),
        (1, d(2026, 1, 1), d(2026, 1, 3), d(2026, 1, 2)),         # 0.5 → 1
        (10**15 + 7, d(2026, 1, 1), d(2026, 1, 31), d(2026, 1, 9)),  # un float perdrait les derniers chiffres
    ]
    for args in cas:
        _un_prorata(f, args, f"prorata({args[0]}, {args[1]}, {args[2]}, {args[3]})")
    alea = random.Random(224)
    for n in range(300):
        debut = d(2024, 1, 1) + timedelta(days=alea.randrange(1500))
        fin = debut + timedelta(days=alea.choice((28, 29, 30, 31, 7, 91, 365, 366)))
        jour = debut + timedelta(days=alea.randrange((fin - debut).days + 1))
        args = (alea.choice((999, 1499, 1999, 2999, 4999, alea.randrange(100_000))), debut, fin, jour)
        _un_prorata(f, args, f"prorata{args} (tirage n° {n})")
    base = (d(2026, 2, 1), d(2026, 3, 1), d(2026, 2, 17))
    for montant, attendu, conseil in (
            (-1, ValueError, "un montant négatif n'a pas de prorata."),
            (29.99, TypeError, "un montant en float a déjà perdu des centimes : exigez un int."),
            (True, TypeError, "True est un int pour Python (isinstance(True, int) est vrai) : refusez-le à part."),
            ("2999", TypeError, "un texte n'est pas un montant.")):
        _doit_lever(f, (montant, *base), f"prorata({montant!r}, …)", conseil, attendu)
    for args, attendu, conseil in (
            ((2999, datetime(2026, 2, 1), datetime(2026, 3, 1), datetime(2026, 2, 17, 12)), TypeError,
             "avec des heures, (fin - jour).days vaut 11 et pas 12 : refusez les datetime. Un datetime EST une date "
             "pour isinstance : testez `isinstance(x, datetime)` d'abord."),
            ((2999, d(2026, 2, 1), d(2026, 3, 1), datetime(2026, 2, 17, 12)), TypeError,
             "un datetime EST une date pour isinstance : testez `isinstance(x, datetime)` d'abord."),
            ((2999, "2026-02-01", d(2026, 3, 1), d(2026, 2, 17)), TypeError, "un texte n'est pas une date."),
            ((2999, d(2026, 3, 1), d(2026, 3, 1), d(2026, 3, 1)), ValueError, "une période vide n'a pas de prorata."),
            ((2999, d(2026, 3, 1), d(2026, 2, 1), d(2026, 2, 17)), ValueError, "debut doit être avant fin."),
            ((2999, d(2026, 2, 1), d(2026, 3, 1), d(2026, 1, 31)), ValueError, "le jour doit être dans la période."),
            ((2999, d(2026, 2, 1), d(2026, 3, 1), d(2026, 3, 2)), ValueError, "le jour doit être dans la période.")):
        _doit_lever(f, args, f"prorata{args}", conseil, attendu)


# ---------------------------------------------------------------------------- défi 225 : repartir


def _ref_repartir(total: int, poids: list[int], bug: str = "") -> list[int]:
    """La méthode du plus fort reste ; `bug` rejoue une erreur classique."""
    somme = sum(poids)
    if bug == "arrondi":
        return [round(total * p / somme) for p in poids]
    if bug == "negatif" and total < 0:
        parts = [total * p // somme for p in poids]
        restes = [total * p % somme for p in poids]
        signe, a_repartir = 1, total
    else:
        signe, a_repartir = (-1 if total < 0 else 1), abs(total)
        parts = [a_repartir * p // somme for p in poids]
        restes = [a_repartir * p % somme for p in poids]
    manque = a_repartir - sum(parts)
    n = len(poids)
    if bug == "dernier":
        parts[-1] += manque
    elif bug == "premier":
        for i in range(manque):
            parts[i % n] += 1
    elif bug == "float":
        flottants = [a_repartir * p / somme - math.floor(a_repartir * p / somme) for p in poids]
        for i in sorted(range(n), key=lambda i: -flottants[i])[:manque]:
            parts[i] += 1
    elif bug == "ex_aequo_dernier":
        for i in sorted(range(n), key=lambda i: (restes[i], i), reverse=True)[:manque]:
            parts[i] += 1
    else:
        for i in sorted(range(n), key=lambda i: -restes[i])[:manque]:
            parts[i] += 1
    return [signe * x for x in parts]


_BUGS_REPARTIR = {
    "negatif": "Pour un total négatif, répartissez |total| comme d'habitude, puis remettez le signe moins : −1000 en "
               "trois → [−334, −333, −333], le symétrique de [334, 333, 333]. (En Python, // arrondit vers −∞ : "
               "−1000 // 3 vaut −334 pour chaque part.)",
    "dernier": "Vous donnez tous les centimes qui manquent à la DERNIÈRE part (« le dernier prend le reste »). Ils "
               "vont, un chacun, aux parts dont le reste (la partie après la virgule) est le plus GRAND.",
    "premier": "Vous donnez les centimes qui manquent aux PREMIÈRES parts de la liste. Ils vont aux parts dont le "
               "reste (la partie après la virgule) est le plus GRAND ; l'ordre de la liste ne sert qu'à départager "
               "les égalités.",
    "ex_aequo_dernier": "À égalité de reste, c'est la PREMIÈRE part de la liste qui passe avant. Trier par "
                        "(reste, position) avec reverse=True met la dernière devant ; `sorted(..., key=lambda i: "
                        "-restes[i])` est stable et garde l'ordre des ex aequo.",
    "float": "Vous comparez des restes en float : 3 × 13 / 30 − 1 vaut 0.30000000000000004, un peu plus que "
             "3 × 3 / 30 = 0.3, et l'égalité disparaît. Calculez des restes ENTIERS : "
             "`q, r = divmod(total * p, somme)`.",
}


def _une_repartition(f, total: int, poids: list[int], quoi: str, lignes_max: int | None = 2_000_000) -> None:
    attendu = _ref_repartir(total, poids)
    copie = list(poids)
    obtenu = _appel(f, (total, copie), quoi, lignes_max=lignes_max,
                    conseil_lent="Votre fonction a été arrêtée : chercher le plus grand reste avec max() pour chaque "
                                 "centime à donner fait n × centimes opérations. Triez les positions UNE fois par "
                                 "reste décroissant (ou heapq.nlargest), puis donnez un centime aux premières.")
    _pas_none(obtenu, quoi)
    _intact(poids, copie, quoi, "les poids")
    if obtenu is copie:
        raise AssertionError(f"{quoi} renvoie la liste `poids` elle-même : construisez une NOUVELLE liste.")
    if not isinstance(obtenu, list):
        raise AssertionError(f"{quoi} renvoie un {type(obtenu).__name__} ; attendu une liste d'int.")
    if len(obtenu) != len(poids):
        raise AssertionError(f"{quoi} renvoie {len(obtenu)} parts pour {len(poids)} poids : une part par poids, "
                             "dans le même ordre, même pour un poids nul.")
    for x in obtenu:
        _entier(x, quoi, "une part")
    if sum(obtenu) != total:
        message = (f"{quoi} : la somme des parts fait {sum(obtenu)} au lieu de {total}. Arrondir chaque part "
                   "séparément perd (ou invente) des centimes. Prenez la partie ENTIÈRE de chaque part, puis donnez "
                   "les centimes qui manquent un par un.")
        if obtenu == _ref_repartir(total, poids, "arrondi"):
            message += " (`round()` sur chaque part : c'est exactement l'erreur que la répartition doit éviter.)"
        raise AssertionError(message)
    if obtenu == attendu:
        return
    detail = f"{quoi} renvoie {_court(obtenu)} au lieu de {_court(attendu)}."
    for bug, conseil in _BUGS_REPARTIR.items():
        if obtenu == _ref_repartir(total, poids, bug):
            raise AssertionError(f"{detail} {conseil}")
    somme = sum(poids)
    for i, (x, p) in enumerate(zip(obtenu, poids)):
        if abs(x * somme - total * p) >= somme:  # |x − part exacte| >= 1
            exacte = total * p / somme
            raise AssertionError(f"{quoi} : la part n° {i} vaut {x} alors que sa part exacte est {exacte:.4f}. Chaque "
                                 "part vaut l'arrondi inférieur ou supérieur de sa part exacte : donner « le reste » "
                                 "à la dernière part (ou à la première) peut lui faire recevoir bien plus.")
    raise AssertionError(f"{detail} Les centimes qui manquent vont aux plus grands restes, le premier à égalité.")


def verifier_repartir(f) -> None:
    cas = [
        (2, [1, 1, 1]),                                  # égalité : les premiers
        (1000, [1, 1, 1]),
        (100, [0, 1, 0, 2]),                             # un poids nul ne reçoit rien
        (1000, [3999, 5553, 1249, 999, 300, 302]),
        (10, [1, 2, 3, 4]),                              # tombe juste
        (7, [5]),
        (0, [5, 7]),
        (1, [0, 0, 3, 0]),
        (5, [2, 1, 2, 1, 2, 1, 1]),                      # les plus grands restes ne sont pas au début
        (3, [3, 13, 11, 3]),                             # le float casse l'égalité des restes
        (-1000, [1, 1, 1]),
        (-5, [1, 2, 2]),
        (10**15 + 1, [1, 1, 1]),
    ]
    for total, poids in cas:
        _une_repartition(f, total, poids, f"repartir({total}, {poids})")
    alea = random.Random(225)
    for n in range(300):
        poids = [alea.choice((0, 1, 2, 3, 10, alea.randrange(1, 10_000))) for _ in range(alea.randint(1, 12))]
        if not any(poids):
            poids[0] = 1
        total = alea.choice((alea.randrange(-5000, 5000), alea.randrange(len(poids) + 1), 1000))
        _une_repartition(f, total, poids, f"repartir({total}, {_court(poids, 80)}) (tirage n° {n})")
    grand = [alea.randrange(1, 10_000) for _ in range(100_000)]
    _une_repartition(f, 10_000_000 + 49_999, grand, "repartir(10 049 999, <100 000 poids>)", lignes_max=5_000_000)
    for args, attendu, conseil in (
            ((100, []), ValueError, "aucun poids : on ne peut rien répartir."),
            ((100, [0, 0]), ValueError, "des poids tous nuls : on ne peut rien répartir (et pas de division par zéro)."),
            ((100, [3, -1, 2]), ValueError, "un poids négatif n'a pas de sens."),
            ((10.5, [1, 2]), TypeError, "total est un int de centimes."),
            ((True, [1, 2]), TypeError, "True n'est pas un montant (même si isinstance(True, int) est vrai)."),
            ((100, [1.5, 2]), TypeError, "les poids sont des int : en float, les restes ne sont plus exacts."),
            ((100, [1, True]), TypeError, "True n'est pas un poids.")):
        _doit_lever(f, args, f"repartir{args}", conseil, attendu)


# ---------------------------------------------------------------------------- défi 226 : facturer


def _ligne(prix: str, tva: str = "20", quantite: int = 1, libelle: str = "x") -> dict:
    return {"libelle": libelle, "quantite": quantite, "prix_unitaire": prix, "tva": tva}


def _ref_facturer(lignes: list[dict], remise: int = 0, bug: str = "") -> tuple:
    """(lignes, par_taux, total_ht, total_tva, total_ttc) en tuples ; `bug` rejoue une erreur classique."""
    bruts, taux = [], []
    for ligne in lignes:
        prix, quantite = ligne["prix_unitaire"], ligne["quantite"]
        if bug == "float_prix":
            bruts.append(round(float(prix) * quantite * 100))
        elif bug == "banquier":
            bruts.append(int((Decimal(prix) * quantite * 100).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)))
        else:
            bruts.append(_demi_haut(Decimal(prix) * quantite * 100))
        taux.append(ligne["tva"])
    total_brut = sum(bruts)
    if not remise or bug == "remise_ttc":
        parts = [0] * len(bruts)
    elif bug == "remise_premiere":
        parts = [remise] + [0] * (len(bruts) - 1)
    elif bug == "remise_arrondie":
        parts = [round(remise * b / total_brut) for b in bruts]
    else:
        parts = _ref_repartir(remise, bruts)
    nets = [b - p for b, p in zip(bruts, parts)]
    groupes: dict[str, list[int]] = {}
    for t, n in zip(taux, nets):
        groupes.setdefault(t, []).append(n)
    if bug == "tri_texte":
        ordre = sorted(groupes, reverse=True)
    elif bug == "ordre_apparition":
        ordre = list(groupes)
    else:
        ordre = sorted(groupes, key=Decimal, reverse=True)
    par_taux = []
    for t in ordre:
        ht = sum(groupes[t])
        if bug == "tva_par_ligne":
            tva = sum(_demi_haut(Decimal(n) * Decimal(t) / 100) for n in groupes[t])
        elif bug == "tva_round":
            tva = round(ht * float(t) / 100)
        else:
            tva = _demi_haut(Decimal(ht) * Decimal(t) / 100)
        par_taux.append((t, ht, tva))
    total_ht = sum(x[1] for x in par_taux)
    total_tva = sum(x[2] for x in par_taux)
    total_ttc = total_ht + total_tva - (remise if bug == "remise_ttc" else 0)
    return (nets, par_taux, total_ht, total_tva, total_ttc)


_BUGS_FACTURER = {
    "banquier": "Un demi-centime est arrondi au PAIR (998.5 → 998) : `quantize(Decimal(1))` sans `rounding=` utilise "
                "ROUND_HALF_EVEN, comme round(). Écrivez `quantize(Decimal(1), rounding=ROUND_HALF_UP)`.",
    "float_prix": "Le prix passe par un float : float(\"1.005\") * 100 vaut 100.49999999999999, et la ligne perd un "
                  "centime. Lisez le prix avec Decimal(texte), qui est exact.",
    "tva_par_ligne": "Vous arrondissez la TVA ligne par ligne puis additionnez. La TVA se calcule UNE fois par taux, sur "
                     "la somme HT des lignes à ce taux : trois lignes à 12,37 € donnent 7,42 € de TVA, pas 7,41 €.",
    "tva_round": "La TVA est arrondie avec round() sur un float : round(16.5) vaut 16 (arrondi au pair). Calculez "
                 "Decimal(ht) * Decimal(taux) / 100, arrondi avec ROUND_HALF_UP : 300 centimes à 5,5 % → 17.",
    "tri_texte": "Les taux sont triés comme du TEXTE (\"5.5\" > \"20\" > \"2.1\" > \"10\"). Triez-les comme des "
                 "nombres : `key=lambda t: Decimal(t)` (ou float), du plus fort au plus faible.",
    "ordre_apparition": "Les taux sont dans l'ordre où ils apparaissent ; attendu : du plus FORT au plus faible "
                        "(20, 10, 5.5, 2.1), comme sur une vraie facture.",
    "remise_premiere": "Toute la remise est déduite d'une seule ligne : la base HT de son taux baisse trop, celle des "
                       "autres taux pas du tout, et la TVA est fausse. Répartissez la remise sur TOUTES les lignes, au "
                       "prorata de leur HT, avec repartir (défi 225).",
    "remise_ttc": "La remise est déduite du TTC, après la TVA : la TVA est alors calculée sur un prix que le client ne "
                  "paie pas. Répartissez-la sur les lignes (repartir, défi 225) AVANT de calculer la TVA.",
    "remise_arrondie": "La remise est répartie avec round() sur chaque ligne : la somme des parts ne fait plus la "
                       "remise. Utilisez votre repartir (défi 225), qui tombe juste.",
}


def _normaliser_facture(resultat, quoi: str) -> tuple:
    champs = ("lignes", "par_taux", "total_ht", "total_tva", "total_ttc")
    if not all(hasattr(resultat, c) for c in champs):
        raise AssertionError(f"{quoi} renvoie {_court(resultat)} ; attendu Facture(lignes, par_taux, total_ht, "
                             "total_tva, total_ttc) (importée de defis.facturation).")
    if not isinstance(resultat.lignes, list) or not isinstance(resultat.par_taux, list):
        raise AssertionError(f"{quoi} : `lignes` et `par_taux` doivent être des listes.")
    for x in resultat.lignes:
        _entier(x, quoi, "le HT d'une ligne")
    par_taux = []
    for t in resultat.par_taux:
        if not isinstance(t, tuple) or len(t) != 3:
            raise AssertionError(f"{quoi} : chaque élément de par_taux est un TotalTaux(taux, ht, tva), pas {_court(t)}.")
        if not isinstance(t[0], str):
            raise AssertionError(f"{quoi} : le taux {t[0]!r} doit rester le texte reçu (\"20\", \"5.5\"…).")
        _entier(t[1], quoi, f"le HT du taux {t[0]}")
        _entier(t[2], quoi, f"la TVA du taux {t[0]}")
        par_taux.append(tuple(t))
    for c in champs[2:]:
        _entier(getattr(resultat, c), quoi, c)
    return (list(resultat.lignes), par_taux, resultat.total_ht, resultat.total_tva, resultat.total_ttc)


def _une_facture(f, lignes: list[dict], remise, quoi: str) -> None:
    attendu = _ref_facturer(lignes, remise)
    copie = copy.deepcopy(lignes)
    obtenu = _appel(f, (copie,), quoi, kwargs={"remise": remise} if remise else {})
    _pas_none(obtenu, quoi)
    _intact(lignes, copie, quoi, "les lignes")
    obtenu = _normaliser_facture(obtenu, quoi)
    if obtenu == attendu:
        return
    detail = f"{quoi} renvoie {_court(obtenu, 300)} ; attendu {_court(attendu, 300)}."
    for bug, conseil in _BUGS_FACTURER.items():
        if obtenu == _ref_facturer(lignes, remise, bug):
            raise AssertionError(f"{detail} {conseil}")
    if remise:
        bruts = _ref_facturer(lignes)[0]
        repartir_eleve = _fonction_eleve(f, "repartir")
        if repartir_eleve is not None:
            try:
                parts = repartir_eleve(remise, list(bruts))
            except Exception:  # noqa: BLE001 — on cherche seulement un indice
                parts = None
            if parts != _ref_repartir(remise, bruts):
                raise AssertionError(f"{detail} Votre `repartir` (défi 225) renvoie {_court(parts)} pour "
                                     f"repartir({remise}, {bruts}) au lieu de {_ref_repartir(remise, bruts)} : "
                                     "réussissez d'abord le défi 225.")
    if obtenu[2:] != attendu[2:] and obtenu[:2] == attendu[:2]:
        raise AssertionError(f"{detail} Les totaux sont les sommes de par_taux : total_ttc = total_ht + total_tva.")
    raise AssertionError(detail)


def verifier_facturer(f) -> None:
    cas = [
        ([_ligne("29.99")], 0),
        ([_ligne("0.045", quantite=1234)], 0),                       # 55.53 € tout rond
        ([_ligne("1.005", quantite=3)], 0),                          # 301.5 → 302 ; en float : 301
        ([_ligne("1.005")], 0),                                      # 100.5 → 101 ; en float : 100
        ([_ligne("9.985", "2.1")], 0),                               # 998.5 → 999 ; arrondi au pair : 998
        ([_ligne("12.37")] * 3, 0),                                  # TVA 742, pas 3 × 247
        ([_ligne("3.00", "5.5")], 0),                                # TVA 16.5 → 17
        ([_ligne("0.25", "10")], 0),                                 # TVA 2.5 → 3
        ([_ligne("29.99", "20"), _ligne("9.99", "2.1"), _ligne("4.00", "5.5"), _ligne("12.49", "10")], 0),
        ([_ligne("1.00", "2.1"), _ligne("1.00", "10")], 0),
        ([], 0),
        ([_ligne("0", "20"), _ligne("5", "10")], 0),                 # un taux à 0 € reste sur la facture
        ([_ligne("29.99"), _ligne("9.99", "2.1"), _ligne("3.00", "5.5")], 1000),
        ([_ligne("39.99"), _ligne("12.49", "10")], 1),
        ([_ligne("10.00"), _ligne("10.00", "10"), _ligne("10.00", "5.5")], 1000),   # 333.33 × 3 : round() en perd un
        ([_ligne("10.00"), _ligne("10.00", "5.5")], 2000),           # remise de tout
        (lignes_exemple(), 0),
        (lignes_exemple(), 1000),
    ]
    for lignes, remise in cas:
        quoi = f"facturer({_court(lignes, 110)}{f', remise={remise}' if remise else ''})"
        _une_facture(f, lignes, remise, quoi)
    alea = random.Random(226)
    for n in range(120):
        lignes = generer_lignes(alea.randint(1, 15), graine=alea.randrange(10**6))
        total = sum(_demi_haut(Decimal(l["prix_unitaire"]) * l["quantite"] * 100) for l in lignes)
        remise = alea.choice((0, 0, alea.randrange(total + 1)))
        _une_facture(f, lignes, remise, f"facturer(generer_lignes(…), remise={remise}) (tirage n° {n})")
    erreurs = (
        ([_ligne("29.99", quantite=0)], 0, ValueError, "une quantité est au moins 1."),
        ([_ligne("29.99", quantite=1.0)], 0, TypeError, "une quantité est un int."),
        ([_ligne("29.99", quantite=True)], 0, TypeError, "True n'est pas une quantité."),
        ([_ligne(29.99)], 0, TypeError, "un prix en float a peut-être déjà perdu des centimes : exigez du texte."),
        ([_ligne("12,99")], 0, ValueError, "le prix s'écrit avec un point ; Decimal(\"12,99\") lève InvalidOperation."),
        ([_ligne("abc")], 0, ValueError, "un prix illisible."),
        ([_ligne("NaN")], 0, ValueError, "Decimal(\"NaN\") se lit, mais n'est pas un prix (Decimal.is_finite())."),
        ([_ligne("Infinity")], 0, ValueError, "Decimal(\"Infinity\") se lit, mais n'est pas un prix."),
        ([_ligne("-1.00")], 0, ValueError, "un prix négatif."),
        ([_ligne("29.99", "19.6")], 0, ValueError, "19,6 % n'existe plus depuis 2014 : taux inconnu."),
        ([_ligne("29.99", 20)], 0, ValueError, "le taux est un TEXTE de TAUX_TVA ; le nombre 20 n'en est pas un."),
        ([_ligne("29.99")], -1, ValueError, "une remise négative est une augmentation."),
        ([_ligne("29.99")], 3000, ValueError, "une remise plus grande que la facture."),
        ([_ligne("29.99")], 10.0, TypeError, "une remise est un int de centimes."),
    )
    for lignes, remise, attendu, conseil in erreurs:
        _doit_lever(f, (lignes,), f"facturer({_court(lignes, 110)}, remise={remise!r})", conseil, attendu,
                    kwargs={"remise": remise})


# ---------------------------------------------------------------------------- défi 227 : jours fériés


def _paques(annee: int) -> date:
    a = annee % 19
    b, c = divmod(annee, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    mois, jour = divmod(h + l - 7 * m + 114, 31)
    return date(annee, mois, jour + 1)


def _ref_feries(annee: int, alsace_moselle: bool = False, bug: str = "") -> list[date]:
    p = _paques(annee)
    ascension = p + timedelta(days=40 if bug == "ascension_40" else 39)
    pentecote = p + timedelta(days=49 if bug == "pentecote_49" else 50)
    jours = [date(annee, 1, 1), p + timedelta(days=1), date(annee, 5, 1), date(annee, 5, 8), ascension, pentecote,
             date(annee, 7, 14), date(annee, 8, 15), date(annee, 11, 1), date(annee, 11, 11), date(annee, 12, 25)]
    if alsace_moselle:
        jours += [p - timedelta(days=2), date(annee, 12, 26)]
    return sorted(jours) if bug == "doublon" else sorted(set(jours))


def _un_jeu_de_feries(f, annee: int, alsace_moselle: bool, quoi: str) -> None:
    attendu = _ref_feries(annee, alsace_moselle)
    obtenu = _appel(f, (annee,), quoi, kwargs={"alsace_moselle": True} if alsace_moselle else {})
    _pas_none(obtenu, quoi)
    if not isinstance(obtenu, list):
        raise AssertionError(f"{quoi} renvoie un {type(obtenu).__name__} ; attendu une LISTE de dates triée "
                             "(`sorted(ensemble)` en fabrique une).")
    for j in obtenu:
        if type(j) is not date:
            raise AssertionError(f"{quoi} contient {_court(j)} ({type(j).__name__}) ; attendu des datetime.date.")
    if obtenu == attendu:
        return
    p = _paques(annee)
    if obtenu == _ref_feries(annee, alsace_moselle, "doublon"):
        raise AssertionError(f"{quoi} contient deux fois le {_court(sorted(j for j in set(obtenu) if obtenu.count(j) > 1))}"
                             f" : en {annee}, deux fêtes tombent le même jour (l'Ascension le 1er mai). Une date "
                             "fériée ne compte qu'une fois : passez par un `set`.")
    if sorted(obtenu) == attendu:
        raise AssertionError(f"{quoi} renvoie les bons jours, mais pas triés : `sorted(...)`.")
    if obtenu == _ref_feries(annee, alsace_moselle, "ascension_40"):
        raise AssertionError(f"{quoi} : l'Ascension tombe 39 jours après Pâques (le jeudi {p + timedelta(days=39)}), "
                             f"pas 40 (le {p + timedelta(days=40)}, un vendredi).")
    if obtenu == _ref_feries(annee, alsace_moselle, "pentecote_49"):
        raise AssertionError(f"{quoi} : Pâques + 49 jours est le DIMANCHE de Pentecôte ; le jour férié est le LUNDI, "
                             f"Pâques + 50 jours ({p + timedelta(days=50)}).")
    manquants = sorted(set(attendu) - set(obtenu))
    en_trop = sorted(set(obtenu) - set(attendu))
    if p + timedelta(days=1) in manquants:
        lundis = [j for j in en_trop if j.weekday() == 0 and date(annee, 3, 1) < j < date(annee, 5, 1)]
        vu = f" (vous donnez le {lundis[0]})" if lundis else ""
        raise AssertionError(f"{quoi} : en {annee}, Pâques tombe le dimanche {p}, donc le lundi de Pâques le "
                             f"{p + timedelta(days=1)}{vu}. Recopiez la formule de Meeus, Jones et Butcher sans en "
                             "changer une opération : que des `//` et des `%`, jamais `/`.")
    if alsace_moselle and p - timedelta(days=2) in manquants:
        raise AssertionError(f"{quoi} : le Vendredi saint est Pâques − 2 jours ({p - timedelta(days=2)}).")
    raise AssertionError(f"{quoi} : il manque {_court(manquants)} et il y a en trop {_court(en_trop)}.")


def verifier_jours_feries(f) -> None:
    _un_jeu_de_feries(f, 2026, False, "jours_feries(2026)")
    _un_jeu_de_feries(f, 2008, False, "jours_feries(2008)")
    _un_jeu_de_feries(f, 2026, True, "jours_feries(2026, alsace_moselle=True)")
    _un_jeu_de_feries(f, 2008, True, "jours_feries(2008, alsace_moselle=True)")
    for annee in list(range(1583, 2600)) + [2800, 3000, 4100, 5555, 9999]:
        _un_jeu_de_feries(f, annee, annee % 7 == 0, f"jours_feries({annee}{', alsace_moselle=True' if annee % 7 == 0 else ''})")
    premiere = _appel(f, (2026,), "jours_feries(2026)")
    premiere.clear()
    if len(_appel(f, (2026,), "jours_feries(2026)")) != 11:
        raise AssertionError("jours_feries(2026) renvoie la MÊME liste à chaque appel : si l'appelant la modifie, le "
                             "suivant reçoit une liste abîmée. Renvoyez une nouvelle liste (une copie de votre cache).")
    for annee, attendu, conseil in (
            (1582, ValueError, "avant 1583, le calendrier grégorien ne s'appliquait pas."),
            (10000, ValueError, "datetime.date s'arrête en 9999."),
            ("2026", TypeError, "une année est un int."),
            (2026.0, TypeError, "une année est un int."),
            (True, TypeError, "True n'est pas une année.")):
        _doit_lever(f, (annee,), f"jours_feries({annee!r})", conseil, attendu)


# ---------------------------------------------------------------------------- défi 228 : relances


def _ref_relances(echec: date, delais=DELAIS, grace: int = GRACE, bug: str = "") -> tuple:
    """(essais, suspension) ; `bug` rejoue une erreur classique."""
    cache: dict[int, set[date]] = {}

    def ouvre(jour: date) -> bool:
        if jour.weekday() >= (6 if bug == "samedi" else 5):
            return False
        if bug == "sans_feries":
            return True
        annee = echec.year if bug == "annee_echec" else jour.year
        if annee not in cache:
            cache[annee] = set(_ref_feries(annee))
        return jour not in cache[annee]

    def premier(jour: date) -> date:
        while not ouvre(jour):
            jour += timedelta(days=1)
        return jour

    essais: list[date] = []
    precedent_delai = 0
    for d in delais:
        if bug == "enchaine" and essais:
            prevu = essais[-1] + timedelta(days=d - precedent_delai)
        else:
            prevu = echec + timedelta(days=d)
        if essais and bug != "doublon":
            prevu = max(prevu, essais[-1] + timedelta(days=1))
        essais.append(premier(prevu))
        precedent_delai = d
    depart = echec if bug == "suspension_echec" or not essais else essais[-1]
    suspension = depart + timedelta(days=grace)
    if bug != "suspension_brute":
        suspension = premier(suspension)
    return (essais, suspension)


_BUGS_RELANCES = {
    "annee_echec": "Vous prenez les fériés de l'année du REFUS : un refus le 31 décembre se relance en janvier, et le "
                   "1er janvier de l'année SUIVANTE est férié aussi. Prenez les fériés de l'année du jour testé "
                   "(jours_feries(jour.year)).",
    "suspension_brute": "La suspension tombe un jour non ouvré : elle aussi a lieu le premier jour ouvré à partir du "
                        "dernier essai + grace.",
    "sans_feries": "Un jour férié n'est pas un jour ouvré : la banque ne prélève pas le 1er mai. Utilisez votre "
                   "jours_feries (défi 227).",
    "samedi": "Le samedi n'est pas un jour ouvré : `jour.weekday()` vaut 5 le samedi et 6 le dimanche, le week-end "
              "est donc `weekday() >= 5` (et pas `> 5`).",
    "doublon": "Deux essais tombent le même jour : un essai a lieu strictement APRÈS le précédent (le premier jour "
               "ouvré à partir de max(date prévue, essai précédent + 1 jour)).",
    "enchaine": "Vous comptez chaque essai depuis l'essai précédent, déjà décalé : les décalages s'accumulent. Chaque "
                "essai est prévu à echec + delais[k], compté depuis le REFUS.",
    "suspension_echec": "La suspension se compte depuis le DERNIER ESSAI (dernier essai + grace), pas depuis le refus.",
}


def _normaliser_relances(resultat, quoi: str) -> tuple:
    if not (hasattr(resultat, "essais") and hasattr(resultat, "suspension")):
        raise AssertionError(f"{quoi} renvoie {_court(resultat)} ; attendu Relances(essais, suspension) (importée de "
                             "defis.facturation).")
    if not isinstance(resultat.essais, list):
        raise AssertionError(f"{quoi} : `essais` doit être une liste de dates, pas un {type(resultat.essais).__name__}.")
    for j in [*resultat.essais, resultat.suspension]:
        if type(j) is not date:
            raise AssertionError(f"{quoi} contient {_court(j)} ({type(j).__name__}) ; attendu des datetime.date.")
    return (list(resultat.essais), resultat.suspension)


def _une_relance(f, echec: date, delais, grace, quoi: str, par_defaut: bool = False) -> None:
    attendu = _ref_relances(echec, delais, grace)
    if par_defaut:
        obtenu = _appel(f, (echec,), quoi)
    else:
        copie = list(delais)
        obtenu = _appel(f, (echec,), quoi, kwargs={"delais": copie, "grace": grace})
        _intact(list(delais), copie, quoi, "les délais")
    _pas_none(obtenu, quoi)
    obtenu = _normaliser_relances(obtenu, quoi)
    if obtenu == attendu:
        return
    detail = f"{quoi} renvoie {_court(obtenu, 300)} ; attendu {_court(attendu, 300)}."
    for bug, conseil in _BUGS_RELANCES.items():
        if obtenu == _ref_relances(echec, delais, grace, bug):
            raise AssertionError(f"{detail} {conseil}")
    feries_eleve = _fonction_eleve(f, "jours_feries")
    if feries_eleve is not None:
        for annee in sorted({echec.year, *(j.year for j in attendu[0]), attendu[1].year}):
            try:
                juste = feries_eleve(annee) == _ref_feries(annee)
            except Exception:  # noqa: BLE001 — on cherche seulement un indice
                juste = False
            if not juste:
                raise AssertionError(f"{detail} Votre jours_feries({annee}) (défi 227) est faux : réussissez d'abord "
                                     "le défi 227.")
    raise AssertionError(detail)


def verifier_relances(f) -> None:
    d = date
    _une_relance(f, d(2026, 4, 30), DELAIS, GRACE, "relances(date(2026, 4, 30))", par_defaut=True)
    _une_relance(f, d(2026, 12, 31), DELAIS, GRACE, "relances(date(2026, 12, 31))", par_defaut=True)
    cas = [
        (d(2026, 11, 13), (1, 3), 0),          # vendredi : samedi, puis deux essais le même lundi
        (d(2026, 5, 11), (1,), 13),            # suspension le lundi de Pentecôte
        (d(2026, 7, 7), (), 7),                # aucun essai : suspension depuis le refus, le 14 juillet
        (d(2026, 8, 15), DELAIS, GRACE),       # un samedi férié
        (d(2026, 11, 5), DELAIS, GRACE),
        (d(2027, 12, 24), (1, 2, 3, 4, 5), 0),  # Noël, le week-end, puis la nouvelle année
        (d(2026, 5, 13), (1, 2), 1),           # l'Ascension, puis le vendredi
    ]
    for echec, delais, grace in cas:
        _une_relance(f, echec, delais, grace, f"relances(date({echec.year}, {echec.month}, {echec.day}), "
                                              f"delais={delais}, grace={grace})")
    alea = random.Random(228)
    for n in range(300):
        echec = d(2025, 1, 1) + timedelta(days=alea.randrange(4000))
        delais = tuple(sorted(alea.sample(range(1, 30), alea.randint(0, 5))))
        grace = alea.choice((0, 1, 3, 7, 10, 30))
        _une_relance(f, echec, delais, grace, f"relances(date({echec.year}, {echec.month}, {echec.day}), "
                                              f"delais={delais}, grace={grace}) (tirage n° {n})")
    base = d(2026, 4, 30)
    erreurs = (
        ((datetime(2026, 4, 30, 9),), {}, TypeError, "un datetime EST une date pour isinstance : testez-le d'abord."),
        (("2026-04-30",), {}, TypeError, "un texte n'est pas une date."),
        ((base,), {"delais": (3, 1)}, ValueError, "les délais sont strictement croissants."),
        ((base,), {"delais": (1, 1)}, ValueError, "les délais sont STRICTEMENT croissants."),
        ((base,), {"delais": (0, 2)}, ValueError, "un essai le jour même du refus n'a pas de sens : délai >= 1."),
        ((base,), {"delais": (1.5,)}, TypeError, "un délai est un nombre entier de jours."),
        ((base,), {"delais": (True, 3)}, TypeError, "True n'est pas un délai."),
        ((base,), {"grace": -1}, ValueError, "la grâce est >= 0."),
        ((base,), {"grace": 2.0}, TypeError, "la grâce est un int."),
    )
    for args, kwargs, attendu, conseil in erreurs:
        quoi = f"relances({args[0]!r}{''.join(f', {k}={v!r}' for k, v in kwargs.items())})"
        _doit_lever(f, args, quoi, conseil, attendu, kwargs=kwargs)


# ---------------------------------------------------------------------------- défi 229 : rapprocher

_MOTIFS = {
    "": re.compile(r"\bFAC-\d{4}-\d{5}\b", re.IGNORECASE),
    "sans_bornes": re.compile(r"FAC-\d{4}-\d{5}", re.IGNORECASE),
    "casse": re.compile(r"\bFAC-\d{4}-\d{5}\b"),
}


def _ref_rapprocher(creances, paiements, bug: str = "") -> tuple:
    """(affectations, restes, credits, non_affectes) en tuples et listes ; `bug` rejoue une erreur classique."""
    reste = {c.numero: c.montant for c in creances}
    proprietaire = {c.numero: c.client for c in creances}
    par_client: dict[str, list] = {}
    for c in creances:
        par_client.setdefault(c.client, []).append(c)
    cle = (lambda c: c.numero) if bug == "fifo_numero" else (lambda c: (c.echeance, c.numero))
    files = {k: [c.numero for c in sorted(v, key=cle)] for k, v in par_client.items()}
    motif = _MOTIFS.get(bug, _MOTIFS[""])
    affectations, credits, non_affectes = [], {}, []
    ordre = list(paiements) if bug == "ordre_recu" else sorted(paiements, key=lambda p: p.jour)
    for p in ordre:
        if p.client not in files:
            if bug == "inconnu_credit":
                credits[p.client] = credits.get(p.client, 0) + p.montant
            else:
                non_affectes.append(p.id)
            continue
        disponible = p.montant
        cites = [] if bug == "sans_citation" else list(dict.fromkeys(m.upper() for m in motif.findall(p.libelle or "")))
        for numero in cites:
            a_lui = proprietaire.get(numero) == p.client or (bug == "autre_client" and numero in reste)
            if disponible and a_lui and reste[numero] > 0:
                montant = disponible if bug == "plafond" else min(reste[numero], disponible)
                reste[numero] -= montant
                disponible -= montant
                affectations.append((p.id, numero, montant))
        for numero in files[p.client]:
            if not disponible:
                break
            if reste[numero] > 0:
                montant = min(reste[numero], disponible)
                reste[numero] -= montant
                disponible -= montant
                affectations.append((p.id, numero, montant))
        if disponible and bug != "credit_perdu":
            credits[p.client] = credits.get(p.client, 0) + disponible
    return (affectations, [(n, r) for n, r in reste.items() if r > 0], list(credits.items()), non_affectes)


_BUGS_RAPPROCHER = {
    "ordre_recu": "Vous traitez les paiements dans l'ordre du relevé. La banque exporte dans le désordre : traitez-les "
                  "dans l'ordre de leur `jour` (`sorted(paiements, key=lambda p: p.jour)`, qui garde l'ordre reçu pour "
                  "un même jour).",
    "fifo_numero": "Vous soldez les factures par NUMÉRO. Le paiement va à la plus ancienne ÉCHÉANCE d'abord (à "
                   "échéance égale, par numéro) : une régularisation émise après coup peut être due avant.",
    "autre_client": "Vous payez une facture citée qui appartient à un AUTRE client (une faute de frappe, un numéro "
                    "copié sur la facture du voisin). Une facture citée ne compte que si elle est à CE client.",
    "sans_bornes": "Votre motif trouve « FAC-2026-00012 » dans « FAC-2026-000127 », une faute de frappe (six chiffres). "
                   "Ajoutez des frontières : r\"\\bFAC-\\d{4}-\\d{5}\\b\".",
    "casse": "Les numéros tapés en minuscules (« fac-2026-00012 ») sont des citations aussi : re.IGNORECASE, puis "
             ".upper() pour retrouver le numéro.",
    "sans_citation": "Vous ignorez les numéros cités dans le libellé : un paiement règle D'ABORD les factures qu'il "
                     "cite (si elles sont à ce client), puis les plus anciennes.",
    "credit_perdu": "Ce qui reste d'un paiement après avoir soldé toutes les factures du client disparaît. C'est un "
                    "CRÉDIT du client : credits[client] += reste.",
    "inconnu_credit": "Un paiement d'un client inconnu (None, ou sans aucune facture) n'est pas un crédit : on ne sait "
                      "pas à qui il est. Son id va dans `non_affectes`.",
    "plafond": "Une facture citée reçoit plus que ce qu'elle doit : chaque facture reçoit au plus son reste dû "
               "(min(reste, disponible)), et le surplus continue vers les autres factures.",
}


def _normaliser_rapprochement(resultat, quoi: str) -> tuple:
    champs = ("affectations", "restes", "credits", "non_affectes")
    if not all(hasattr(resultat, c) for c in champs):
        raise AssertionError(f"{quoi} renvoie {_court(resultat)} ; attendu Rapprochement(affectations, restes, "
                             "credits, non_affectes) (importée de defis.facturation).")
    if not isinstance(resultat.affectations, list) or not isinstance(resultat.non_affectes, list):
        raise AssertionError(f"{quoi} : `affectations` et `non_affectes` doivent être des listes.")
    if not isinstance(resultat.restes, dict) or not isinstance(resultat.credits, dict):
        raise AssertionError(f"{quoi} : `restes` et `credits` doivent être des dict.")
    affectations = []
    for a in resultat.affectations:
        if not isinstance(a, tuple) or len(a) != 3:
            raise AssertionError(f"{quoi} : chaque affectation est une Affectation(paiement, numero, montant), pas "
                                 f"{_court(a)}.")
        _entier(a[2], quoi, "le montant d'une affectation")
        affectations.append(tuple(a))
    for valeur in [*resultat.restes.values(), *resultat.credits.values()]:
        _entier(valeur, quoi, "un reste ou un crédit")
    return (affectations, list(resultat.restes.items()), list(resultat.credits.items()), list(resultat.non_affectes))


def _un_rapprochement(f, creances, paiements, quoi: str, lignes_max: int | None = 2_000_000) -> None:
    attendu = _ref_rapprocher(creances, paiements)
    copies = (list(creances), list(paiements))
    obtenu = _appel(f, copies, quoi, lignes_max=lignes_max,
                    conseil_lent="Votre fonction a été arrêtée : chercher les factures d'un client dans TOUTES les "
                                 "factures, pour chaque paiement, fait des centaines de millions d'opérations. "
                                 "Rangez-les une fois dans un dict {client: numéros triés} et avancez un curseur.")
    _pas_none(obtenu, quoi)
    _intact((list(creances), list(paiements)), copies, quoi, "les créances ou les paiements")
    obtenu = _normaliser_rapprochement(obtenu, quoi)
    if obtenu == attendu:
        return
    detail = ""
    for nom, a, b in zip(("affectations", "restes", "credits", "non_affectes"), obtenu, attendu):
        if a != b:
            if sorted(a, key=repr) == sorted(b, key=repr):
                ordre = {"restes": "dans l'ordre des créances reçues", "credits": "dans l'ordre où chaque crédit est "
                         "apparu", "affectations": "dans l'ordre où on les fait", "non_affectes": "dans l'ordre de "
                         "traitement"}[nom]
                detail = f"{quoi} : `{nom}` a les bonnes valeurs, mais pas dans le bon ordre ({ordre})."
            else:
                detail = f"{quoi} : `{nom}` vaut {_court(a, 220)} ; attendu {_court(b, 220)}."
            break
    for bug, conseil in _BUGS_RAPPROCHER.items():
        if obtenu == _ref_rapprocher(creances, paiements, bug):
            raise AssertionError(f"{detail} {conseil}")
    raise AssertionError(detail)


def _f(numero: int, client: str, montant: int, mois: int, jour: int = 5) -> Creance:
    return Creance(f"FAC-2026-{numero:05d}", client, montant, date(2026, mois, jour))


def _p(id: str, jour: int, client, montant: int, libelle: str = "VIR", mois: int = 11) -> Paiement:
    return Paiement(id, date(2026, mois, jour), client, montant, libelle)


def verifier_rapprocher(f) -> None:
    deux = [_f(1, "A", 1000, 10), _f(2, "A", 1500, 11), _f(3, "B", 1500, 11)]
    cas = [
        ("un prélèvement", [_f(1, "A", 2999, 11)], [_p("P1", 5, "A", 2999, "PRLV SEPA FAC-2026-00001")]),
        ("un paiement sans numéro", deux, [_p("P1", 6, "A", 1000)]),
        ("la facture citée d'abord", deux, [_p("P1", 6, "A", 1500, "VIR FAC-2026-00002")]),
        ("un paiement plus grand que la facture citée", deux, [_p("P1", 6, "A", 3000, "VIR FAC-2026-00002")]),
        ("le numéro d'un autre client", deux, [_p("P1", 6, "A", 1500, "VIR FAC-2026-00003")]),
        ("une faute de frappe", deux, [_p("P1", 6, "A", 1500, "VIR FAC-2026-000021")]),
        ("des minuscules", deux, [_p("P1", 6, "A", 1500, "vir fac-2026-00002")]),
        ("deux numéros, dont un cité deux fois", deux,
         [_p("P1", 6, "A", 2000, "FAC-2026-00002/FAC-2026-00001 RAPPEL FAC-2026-00002")]),
        ("un payeur inconnu", deux, [_p("P1", 6, None, 1500, "VIR INST"), _p("P2", 7, "Z", 900, "VIR")]),
        ("un relevé dans le désordre", deux,
         [_p("P2", 10, "A", 1000, "VIR FAC-2026-00001"), _p("P1", 3, "A", 1000)]),
        ("deux paiements le même jour", deux, [_p("P2", 6, "A", 1200), _p("P1", 6, "A", 1300)]),
        ("une régularisation due avant", [_f(1, "A", 1000, 11), _f(2, "A", 400, 10, 20)], [_p("P1", 9, "A", 400)]),
        ("des crédits qui s'additionnent", deux,
         [_p("P1", 6, "B", 2000), _p("P2", 7, "A", 3000), _p("P3", 8, "B", 100), _p("P4", 9, "A", 1)]),
        ("une facture déjà soldée", deux, [_p("P1", 6, "A", 1000), _p("P2", 7, "A", 800, "VIR FAC-2026-00001")]),
        ("rien à rapprocher", deux, []),
    ]
    for nom, creances, paiements in cas:
        _un_rapprochement(f, creances, paiements, f"rapprocher(…) sur « {nom} » ({_court(paiements, 200)})")
    alea = random.Random(229)
    for n in range(40):
        creances = generer_creances(alea.randint(1, 40), graine=alea.randrange(10**6))
        paiements = generer_paiements(creances, graine=alea.randrange(10**6), inconnus=alea.randint(0, 3))
        _un_rapprochement(f, creances, paiements, f"rapprocher(generer_creances(…), generer_paiements(…)) (tirage n° {n})")
    _un_rapprochement(f, creances_exemple(), paiements_exemple(), "rapprocher(creances_exemple(), paiements_exemple())")
    grandes = generer_creances(10_000, graine=37)
    _un_rapprochement(f, grandes, generer_paiements(grandes, graine=37),
                      "rapprocher(<20 000 factures>, <19 000 paiements>)", lignes_max=5_000_000)
    erreurs = (
        ([_f(1, "A", 1000, 11), _f(1, "B", 500, 11)], [], ValueError, "deux factures ont le même numéro."),
        ([_f(1, "A", 1000, 11)], [_p("P1", 6, "A", 500), _p("P1", 7, "A", 500)], ValueError,
         "deux paiements ont le même id."),
        ([_f(1, "A", 1000, 11)], [_p("P1", 6, "A", 0)], ValueError, "un paiement de 0 centime."),
        ([_f(1, "A", 1000, 11)], [_p("P1", 6, "A", -500)], ValueError, "un paiement négatif est un remboursement."),
        ([_f(1, "A", 1000, 11)], [_p("P1", 6, "A", 10.5)], TypeError, "un montant est un int de centimes."),
        ([_f(1, "A", 1000, 11)], [_p("P1", 6, "A", True)], TypeError, "True n'est pas un montant."),
        ([_f(1, "A", -5, 11)], [], ValueError, "une facture de montant négatif."),
        ([_f(1, "A", 99.5, 11)], [], TypeError, "un montant est un int de centimes."),
    )
    for creances, paiements, attendu, conseil in erreurs:
        _doit_lever(f, (creances, paiements), f"rapprocher({_court(creances, 90)}, {_court(paiements, 90)})",
                    conseil, attendu)


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"prorata": 224, "repartir": 225, "facturer": 226, "jours_feries": 227, "relances": 228,
            "rapprocher": 229}

SERIE = Serie(
    code="37",
    titre="Facturer au centime près : prorata, répartition, TVA, jours fériés, relances, rapprochement",
    module_eleve="defis.serie_37",
    module_corrige="defis.corrige_serie_37",
    defis=(
        Defi(224, "prorata", "🟢", "—", verifier_prorata),
        Defi(225, "repartir", "🟡", "—", verifier_repartir),
        Defi(226, "facturer", "🟡", "—", verifier_facturer),
        Defi(227, "jours_feries", "🟢", "—", verifier_jours_feries),
        Defi(228, "relances", "🟡", "—", verifier_relances),
        Defi(229, "rapprocher", "🔴", "—", verifier_rapprocher),
    ),
)

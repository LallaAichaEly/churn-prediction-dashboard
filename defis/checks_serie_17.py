"""Vérifications automatiques de la série 17 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur ne reçoit pas une fonction mais une CLASSE (sauf au défi 107). Il s'en sert comme
le ferait un collègue : il crée des objets, les compare, les met dans un `set`, les additionne, les
parcourt avec `for`, les combine avec `&`… et vérifie que chaque mauvaise utilisation est refusée avec
la bonne erreur (TypeError pour un mauvais type, ValueError pour une mauvaise valeur).

Au défi 107, il lève lui-même des erreurs DANS votre bloc `with`, y compris un Ctrl+C simulé
(`KeyboardInterrupt`), et regarde dans le journal de la fausse passerelle ce que votre code a fait.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import math
import time
from decimal import Decimal
from enum import Enum

from .conception import CLIENTS_EXEMPLE, CONTRATS, Passerelle, PasserelleIndisponible
from .verif import Defi, Serie


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
        noms = " ou ".join(e.__name__ for e in (attendue if isinstance(attendue, tuple) else (attendue,)))
        regle = ""
        if {type(exc).__name__, noms} == {"TypeError", "ValueError"}:
            regle = " (TypeError : le TYPE ne va pas ; ValueError : le type va, mais pas la VALEUR)"
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {noms}{regle}. {conseil}".rstrip()) from None
    noms = " ou ".join(e.__name__ for e in (attendue if isinstance(attendue, tuple) else (attendue,)))
    raise AssertionError(f"{quoi} devrait lever {noms}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


# ============================================================================ Défi 104 : dataclass

_CHAMPS_CLIENT = ["client_id", "anciennete_mois", "facture_mensuelle", "contrat", "options"]
_BASE_CLIENT = {"client_id": "C001", "anciennete_mois": 3, "facture_mensuelle": 49.9}


def _client_texte(champs: dict) -> str:
    return "Client(" + ", ".join(f"{k}={v!r}" for k, v in champs.items()) + ")"


def verifier_Client(Client) -> None:
    client = Client("C001", 3, 49.9)
    if not dataclasses.is_dataclass(Client):
        raise AssertionError("Client doit être une dataclass : écrivez @dataclass(frozen=True) juste au-dessus de "
                             "`class Client:`, puis les champs (`client_id: str`, …) et supprimez le `__init__`.")
    noms = [f.name for f in dataclasses.fields(Client)]
    if noms != _CHAMPS_CLIENT:
        raise AssertionError(f"Les champs doivent être, dans cet ordre : {', '.join(_CHAMPS_CLIENT)} "
                             f"(obtenu : {', '.join(noms)}). L'ordre compte : Client(\"C001\", 3, 49.9) les remplit dans "
                             "l'ordre.")
    if client.contrat != "Mensuel":
        raise AssertionError(f"Sans contrat, le client doit être « Mensuel » (obtenu : {_court(client.contrat)}).")
    if client.options != () or not isinstance(client.options, tuple):
        raise AssertionError(f"Sans options, le client doit avoir options == () (obtenu : {_court(client.options)}). "
                             "Un tuple vide est immuable : il peut servir de valeur par défaut, contrairement à [].")

    for champs in ({"anciennete_mois": 0}, {"facture_mensuelle": 0}, {"facture_mensuelle": 0.0},
                   {"client_id": "C42"}, {"contrat": "Biannuel"}, {"contrat": "Annuel"}, {"options": ("tv", "fibre")}):
        complet = _BASE_CLIENT | champs
        try:
            Client(**complet)
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"{_client_texte(complet)} est valide, mais lève {_decrire(exc)}.") from None

    invalides = [
        ({"client_id": 1}, TypeError, ""),
        ({"client_id": ""}, ValueError, ""),
        ({"client_id": "C"}, ValueError, "Il faut au moins un chiffre après le C."),
        ({"client_id": "c001"}, ValueError, ""),
        ({"client_id": "C001 "}, ValueError, "re.match accepte une fin quelconque : utilisez re.fullmatch."),
        ({"client_id": "XC001"}, ValueError, "re.search trouve « C001 » au milieu : utilisez re.fullmatch."),
        ({"anciennete_mois": -3}, ValueError, ""),
        ({"anciennete_mois": "12"}, TypeError, ""),
        ({"anciennete_mois": 12.0}, TypeError, "12.0 est un float : une ancienneté se compte en mois entiers."),
        ({"anciennete_mois": True}, TypeError,
         "bool hérite de int (isinstance(True, int) vaut True) : excluez-le avec isinstance(x, bool)."),
        ({"facture_mensuelle": -1}, ValueError, ""),
        ({"facture_mensuelle": math.nan}, ValueError,
         "NaN n'est ni < 0 ni >= 0 : `if facture < 0` le laisse passer. Utilisez math.isfinite (ou math.isnan)."),
        ({"facture_mensuelle": math.inf}, ValueError, "L'infini est >= 0 : math.isfinite l'écarte, avec NaN."),
        ({"facture_mensuelle": "49,9"}, TypeError, ""),
        ({"facture_mensuelle": None}, TypeError, ""),
        ({"contrat": "mensuel"}, ValueError, "Les contrats s'écrivent comme dans CONTRATS (« Mensuel »)."),
        ({"contrat": "Bi-annuel"}, ValueError, ""),
        ({"options": "tv"}, TypeError,
         "tuple(\"tv\") donne ('t', 'v') : un seul texte doit être refusé, pas découpé en lettres."),
        ({"options": ["tv", 3]}, TypeError, "Chaque option doit être un texte."),
    ]
    for champs, erreur, conseil in invalides:
        complet = _BASE_CLIENT | champs
        quoi = _client_texte(complet)
        exc = _leve(lambda: Client(**complet), erreur, quoi, conseil)
        nom = next(iter(champs))
        if nom not in str(exc):
            raise AssertionError(f"{quoi} lève bien {erreur.__name__}, mais le message doit NOMMER le champ fautif "
                                 f"(« {nom} ») : obtenu {str(exc)!r}. Un message précis fait gagner des heures.")

    try:
        client.anciennete_mois = 4
    except AttributeError:   # dataclasses.FrozenInstanceError hérite d'AttributeError
        pass
    else:
        raise AssertionError("client.anciennete_mois = 4 a modifié le client : écrivez @dataclass(frozen=True). Un client "
                             "gelé ne peut pas être abîmé en douce par une autre fonction.")

    if Client(client_id="C001", anciennete_mois=3, facture_mensuelle=49.9) != client:
        raise AssertionError("Deux clients aux champs identiques doivent être égaux (==), qu'on les crée avec ou sans "
                             "les noms des champs.")

    options = ["tv", "fibre"]
    avec_liste = Client("C002", 12, 30, "Annuel", options)
    options.append("mobile")
    if not isinstance(avec_liste.options, tuple):
        raise AssertionError(f"Client(..., options=['tv', 'fibre']).options vaut {_court(avec_liste.options)} : gardez "
                             "un TUPLE. Avec frozen=True, l'écriture consacrée dans __post_init__ est "
                             "object.__setattr__(self, \"options\", tuple(self.options)).")
    if avec_liste.options != ("tv", "fibre"):
        raise AssertionError(f"Après options.append(\"mobile\") chez l'appelant, le client a les options "
                             f"{_court(avec_liste.options)} au lieu de ('tv', 'fibre') : le client doit garder sa propre "
                             "copie, dans l'ordre reçu.")
    try:
        hash(avec_liste)
    except TypeError as exc:
        raise AssertionError(f"hash(client) lève {_decrire(exc)} : un client gelé doit être hachable (pour un set ou "
                             "une clé de dictionnaire). Gardez les options dans un tuple.") from None
    if len({avec_liste, Client("C002", 12, 30, "Annuel", ("tv", "fibre"))}) != 1:
        raise AssertionError("Un set de deux clients identiques doit n'en garder qu'un : avec frozen=True (et eq=True, "
                             "par défaut), @dataclass écrit __eq__ et __hash__ à partir des champs.")


# ============================================================================ Défi 105 : Enum


def verifier_Contrat(Contrat) -> None:
    premier = Contrat.depuis_texte("Mensuel")
    if not (isinstance(Contrat, type) and issubclass(Contrat, Enum)):
        raise AssertionError("Contrat doit hériter d'Enum : `class Contrat(Enum):`.")
    membres = list(Contrat)
    noms = [m.name for m in membres]
    if len(membres) != 3:
        intrus = [n for n in noms if n not in ("MENSUEL", "ANNUEL", "BIANNUEL")]
        conseil = (f" Intrus : {', '.join(intrus)}. Dans une Enum, TOUT attribut de classe (un dictionnaire, une "
                   "constante, même avec un _ devant) devient un membre. Mettez-le hors de la classe, ou utilisez "
                   "if / match dans la propriété.") if intrus else ""
        raise AssertionError(f"Contrat doit avoir exactement 3 membres (obtenu : {len(membres)} : "
                             f"{', '.join(noms) or 'aucun'}).{conseil}")
    if noms != ["MENSUEL", "ANNUEL", "BIANNUEL"]:
        raise AssertionError(f"Les membres doivent s'appeler MENSUEL, ANNUEL et BIANNUEL, dans cet ordre "
                             f"(obtenu : {', '.join(noms)}).")
    valeurs = [m.value for m in membres]
    if valeurs != list(CONTRATS):
        raise AssertionError(f"Les valeurs doivent être les textes du CSV : {list(CONTRATS)} (obtenu : {valeurs}). "
                             "Ainsi Contrat(\"Annuel\") retrouve le membre à partir de la valeur lue dans le fichier.")
    mensuel, annuel, biannuel = membres

    if premier is not mensuel:
        if premier == "Mensuel":
            raise AssertionError("Contrat.depuis_texte(\"Mensuel\") renvoie le texte \"Mensuel\" : renvoyez le MEMBRE "
                                 "(Contrat.MENSUEL). C'est tout l'intérêt : ensuite, plus aucun texte à comparer.")
        raise AssertionError(f"Contrat.depuis_texte(\"Mensuel\") renvoie {_court(premier)} au lieu de Contrat.MENSUEL.")
    for texte, attendu in (("mensuel", mensuel), ("  ANNUEL ", annuel), ("Biannuel\n", biannuel),
                           ("bIaNnUeL", biannuel), ("annuel", annuel)):
        try:
            obtenu = Contrat.depuis_texte(texte)
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"Contrat.depuis_texte({texte!r}) lève {_decrire(exc)} au lieu de renvoyer {attendu!r}. "
                                 "Ignorez les majuscules et les espaces autour : texte.strip().casefold().") from None
        if obtenu is not attendu:
            raise AssertionError(f"Contrat.depuis_texte({texte!r}) renvoie {_court(obtenu)} au lieu de {attendu!r}. "
                                 "Ignorez les majuscules et les espaces autour : texte.strip().casefold().")
    if Contrat.depuis_texte(annuel) is not annuel:
        raise AssertionError("Contrat.depuis_texte(Contrat.ANNUEL) doit renvoyer Contrat.ANNUEL tel quel : l'appelant "
                             "n'a pas à savoir s'il a déjà converti son contrat.")
    for texte in ("Mensuelle", "Mens", "Bi-annuel", "", "   ", "Mensuel Annuel", "MENSUEL!"):
        exc = _leve(lambda: Contrat.depuis_texte(texte), ValueError, f"Contrat.depuis_texte({texte!r})",
                    "Seule la valeur exacte (aux majuscules et espaces près) est acceptée : pas de startswith ni de `in`.")
        if not all(v in str(exc) for v in CONTRATS):
            raise AssertionError(f"Le message de Contrat.depuis_texte({texte!r}) doit donner les valeurs acceptées "
                                 f"({', '.join(CONTRATS)}) : obtenu {str(exc)!r}.")
    for valeur in (None, 12):
        _leve(lambda: Contrat.depuis_texte(valeur), TypeError, f"Contrat.depuis_texte({valeur!r})",
              "Vérifiez isinstance(texte, str) avant d'appeler .strip().")

    for membre, mois in zip(membres, (1, 12, 24)):
        valeur = membre.engagement_mois
        if callable(valeur):
            raise AssertionError(f"Contrat.{membre.name}.engagement_mois renvoie une méthode : écrivez @property juste "
                                 "au-dessus de `def engagement_mois(self)`, pour la lire sans parenthèses.")
        if valeur != mois or isinstance(valeur, bool):
            raise AssertionError(f"Contrat.{membre.name}.engagement_mois vaut {_court(valeur)} au lieu de {mois}.")

    for membre in membres:
        if str(membre) != membre.value or f"{membre}" != membre.value:
            raise AssertionError(f"str(Contrat.{membre.name}) vaut {str(membre)!r} et f\"{{...}}\" donne {f'{membre}'!r} : "
                                 f"les deux doivent donner {membre.value!r}. Écrivez __str__ (return self.value).")


# ============================================================================ Défi 106 : objet valeur


def _comparer_montant(Montant, obtenu, centimes: int, devise: str, quoi: str) -> None:
    if not isinstance(obtenu, Montant):
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} : renvoyez un Montant (Montant({centimes}, {devise!r})).")
    if obtenu.centimes != centimes or obtenu.devise != devise or type(obtenu.centimes) is not int:
        raise AssertionError(f"{quoi} donne {_court(obtenu)} au lieu de Montant(centimes={centimes}, devise={devise!r}).")


def verifier_Montant(Montant) -> None:
    dix_neuf = Montant(1999)
    if dix_neuf.devise != "EUR":
        raise AssertionError(f"Sans devise, un montant est en euros : Montant(1999).devise vaut {_court(dix_neuf.devise)}.")

    # ------------------------------------------------------------------ validation
    for valide in ((0,), (-500,), (1999, "USD"), (10**12, "GBP")):
        try:
            Montant(*valide)
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"Montant{valide!r} est valide (un montant négatif est un remboursement), mais lève "
                                 f"{_decrire(exc)}.") from None
    for args, erreur, conseil in (
        ((19.99,), TypeError, "Pour des euros, on passe par Montant.depuis_euros(\"19.99\")."),
        ((True,), TypeError, "bool hérite de int : excluez-le avec isinstance(x, bool)."),
        (("1999",), TypeError, ""),
        ((None,), TypeError, ""),
        ((100, "eur"), ValueError, "La devise s'écrit en 3 lettres MAJUSCULES."),
        ((100, "EURO"), ValueError, "3 lettres exactement : re.fullmatch(r\"[A-Z]{3}\", devise)."),
        ((100, "€"), ValueError, ""),
        ((100, ""), ValueError, ""),
    ):
        _leve(lambda: Montant(*args), erreur, f"Montant{args!r}", conseil)

    try:
        dix_neuf.centimes = 5
    except AttributeError:
        pass
    else:
        raise AssertionError("montant.centimes = 5 a modifié le montant : gardez @dataclass(frozen=True).")
    if Montant(1999) != Montant(1999, "EUR") or Montant(1999) == Montant(1999, "USD"):
        raise AssertionError("Deux montants sont égaux si et seulement si centimes ET devise sont égaux.")
    if len({Montant(1999), Montant(1999), Montant(1999, "USD")}) != 2:
        raise AssertionError("Un set de montants doit fusionner les montants égaux : gardez frozen=True (et eq=True).")

    # ------------------------------------------------------------------ depuis_euros
    bons = [("19,99", "EUR", 1999), ("19.99", "EUR", 1999), ("1 234,50", "EUR", 123450),
            ("1 234,5", "EUR", 123450), ("1 234,50", "EUR", 123450), (" 20 ", "EUR", 2000),
            ("-5", "EUR", -500), ("0,1", "EUR", 10), ("19,9", "EUR", 1990), ("1 000 000", "EUR", 100_000_000),
            ("0,07", "EUR", 7), (Decimal("19.99"), "EUR", 1999), (20, "EUR", 2000), (0, "EUR", 0), ("12", "USD", 1200)]
    for euros, devise, centimes in bons:
        quoi = f"Montant.depuis_euros({euros!r}{'' if devise == 'EUR' else f', {devise!r}'})"
        try:
            obtenu = Montant.depuis_euros(euros) if devise == "EUR" else Montant.depuis_euros(euros, devise)
        except NotImplementedError:
            raise
        except Exception as exc:
            conseil = ""
            if isinstance(euros, str) and (" " in euros or " " in euros):
                conseil = (" Excel écrit les milliers avec des espaces INSÉCABLES (\\u00a0, \\u202f) : retirez-les "
                           "comme l'espace ordinaire.")
            raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de renvoyer Montant({centimes}).{conseil}") from None
        if isinstance(obtenu, Montant) and obtenu.centimes == centimes - 1:
            raise AssertionError(f"{quoi} donne {obtenu.centimes} centimes au lieu de {centimes} : vous passez par un "
                                 "float (int(19.99 * 100) vaut 1998). Lisez le texte avec Decimal, qui est exact.")
        _comparer_montant(Montant, obtenu, centimes, devise, quoi)
    for euros in (19.99, 0.5, 20.0):
        _leve(lambda: Montant.depuis_euros(euros), TypeError, f"Montant.depuis_euros({euros!r})",
              "Un float n'est pas exact (0.1 + 0.2 vaut 0.30000000000000004, int(19.99 * 100) vaut 1998) : refusez-le, "
              "l'appelant passera \"19.99\" ou Decimal(\"19.99\").")
    for euros in (True, None, [19]):
        _leve(lambda: Montant.depuis_euros(euros), TypeError, f"Montant.depuis_euros({euros!r})")
    for euros, conseil in (("19,999", "Plus de 2 décimales : quel centime choisir ? On refuse plutôt que d'arrondir."),
                           ("0,001", ""), ("abc", ""), ("", ""), ("   ", ""), ("1,2,3", ""),
                           ("NaN", "Decimal(\"NaN\") existe : valeur.is_finite() l'écarte."),
                           ("Infinity", "Decimal(\"Infinity\") existe : valeur.is_finite() l'écarte.")):
        try:
            _leve(lambda: Montant.depuis_euros(euros), ValueError, f"Montant.depuis_euros({euros!r})", conseil)
        except AssertionError as erreur:
            if "InvalidOperation" in str(erreur) or "ConversionSyntax" in str(erreur):
                raise AssertionError(f"Montant.depuis_euros({euros!r}) laisse passer decimal.InvalidOperation, qui n'est "
                                     "PAS une ValueError (elle hérite d'ArithmeticError) : attrapez-la et levez ValueError "
                                     "(`except InvalidOperation: raise ValueError(...) from None`).") from None
            raise

    # ------------------------------------------------------------------ str
    for montant, attendu in ((Montant(1999), "19,99 €"), (Montant(123450), "1 234,50 €"), (Montant(5), "0,05 €"),
                             (Montant(0), "0,00 €"), (Montant(100), "1,00 €"), (Montant(-5), "-0,05 €"),
                             (Montant(-123450), "-1 234,50 €"), (Montant(100_000_000), "1 000 000,00 €"),
                             (Montant(1200, "USD"), "12,00 USD"), (Montant(-99, "GBP"), "-0,99 GBP")):
        obtenu = str(montant)
        if obtenu != attendu:
            conseil = ""
            if montant.centimes < 0 and obtenu.startswith("-") and obtenu != attendu:
                conseil = (" -5 // 100 vaut -1 et -5 % 100 vaut 95 : la division entière arrondit vers le bas. "
                           "Calculez sur abs(centimes), puis remettez le signe.")
            elif "." in obtenu:
                conseil = " En français, la virgule sépare les décimales, et une espace les milliers."
            raise AssertionError(f"str({montant!r}) vaut {obtenu!r} au lieu de {attendu!r}.{conseil}")
        if f"{montant}" != attendu:
            raise AssertionError(f"f\"{{montant}}\" vaut {f'{montant}'!r} au lieu de {attendu!r}.")

    # ------------------------------------------------------------------ +
    _comparer_montant(Montant, Montant(1999) + Montant(1), 2000, "EUR", "Montant(1999) + Montant(1)")
    _comparer_montant(Montant, Montant(100, "USD") + Montant(-30, "USD"), 70, "USD",
                      "Montant(100, 'USD') + Montant(-30, 'USD')")
    exc = _leve(lambda: Montant(100, "USD") + Montant(100), ValueError, "Montant(100, 'USD') + Montant(100)",
                "On n'additionne pas des dollars et des euros sans taux de change.")
    if "USD" not in str(exc) or "EUR" not in str(exc):
        raise AssertionError(f"Le message de Montant(100, 'USD') + Montant(100) doit citer les deux devises : {str(exc)!r}.")
    for autre in (1, 1.5, "1", None):
        _leve(lambda: Montant(100) + autre, TypeError, f"Montant(100) + {autre!r}",
              "Renvoyez NotImplemented (sans le lever) quand l'autre n'est pas un Montant : Python lèvera TypeError.")
    for autre in (1, 1.5):
        _leve(lambda: autre + Montant(100), TypeError, f"{autre!r} + Montant(100)",
              "__radd__ n'accepte que le 0 de départ de sum() ; sinon, return NotImplemented.")
    try:
        total = sum([Montant(1999), Montant(1), Montant(-500)])
    except NotImplementedError:
        raise
    except TypeError as exc:
        raise AssertionError(f"sum([Montant(1999), Montant(1), Montant(-500)]) lève {_decrire(exc)} : sum() commence à 0 "
                             "et calcule 0 + Montant(1999). int ne sait pas faire, Python appelle alors "
                             "Montant(1999).__radd__(0) : acceptez ce 0 et renvoyez self.") from None
    _comparer_montant(Montant, total, 1500, "EUR", "sum([Montant(1999), Montant(1), Montant(-500)])")
    _comparer_montant(Montant, sum([Montant(100, "USD")] * 3), 300, "USD", "sum([Montant(100, 'USD')] * 3)")
    _comparer_montant(Montant, sum([Montant(5), Montant(6)], Montant(0)), 11, "EUR",
                      "sum([Montant(5), Montant(6)], Montant(0))")
    _leve(lambda: sum([Montant(1), Montant(1, "USD")]), ValueError, "sum([Montant(1), Montant(1, 'USD')])")

    # ------------------------------------------------------------------ *
    for calcul, quoi, centimes, devise in ((lambda: Montant(1999) * 3, "Montant(1999) * 3", 5997, "EUR"),
                                           (lambda: 3 * Montant(1999), "3 * Montant(1999)", 5997, "EUR"),
                                           (lambda: Montant(1999) * 0, "Montant(1999) * 0", 0, "EUR"),
                                           (lambda: Montant(-5) * -2, "Montant(-5) * -2", 10, "EUR"),
                                           (lambda: 2 * Montant(150, "USD"), "2 * Montant(150, 'USD')", 300, "USD")):
        try:
            obtenu = calcul()
        except NotImplementedError:
            raise
        except TypeError as exc:
            conseil = (" `3 * montant` : int ne sait pas multiplier un Montant, Python appelle montant.__rmul__(3) ; "
                       "la multiplication est commutative : __rmul__ peut faire comme __mul__.") if quoi[0].isdigit() else ""
            raise AssertionError(f"{quoi} lève {_decrire(exc)}.{conseil}") from None
        _comparer_montant(Montant, obtenu, centimes, devise, quoi)
    for calcul, quoi in ((lambda: Montant(100) * 1.5, "Montant(100) * 1.5"), (lambda: 1.5 * Montant(100), "1.5 * Montant(100)"),
                         (lambda: Montant(100) * Montant(2), "Montant(100) * Montant(2)"),
                         (lambda: Montant(100) * "3", "Montant(100) * '3'")):
        _leve(calcul, TypeError, quoi, "On ne multiplie un montant que par un ENTIER (une quantité) : 1,5 × 1,00 € tombe "
              "pile, mais 1,5 × 0,01 € tombe entre deux centimes. Sinon, return NotImplemented.")

    # ------------------------------------------------------------------ comparaisons
    a, b, c = Montant(100), Montant(250), Montant(-30)
    for quoi, calcul, attendu in (("Montant(100) < Montant(250)", lambda: a < b, True),
                                  ("Montant(250) < Montant(100)", lambda: b < a, False),
                                  ("Montant(100) <= Montant(100)", lambda: a <= Montant(100), True),
                                  ("Montant(250) > Montant(100)", lambda: b > a, True),
                                  ("Montant(-30) >= Montant(100)", lambda: c >= a, False),
                                  ("Montant(100) < Montant(100)", lambda: a < Montant(100), False)):
        try:
            obtenu = calcul()
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"{quoi} lève {_decrire(exc)} : écrivez __lt__ (et gardez @total_ordering, qui en déduit "
                                 "<=, > et >=).") from None
        if obtenu is not attendu:
            raise AssertionError(f"{quoi} vaut {_court(obtenu)} au lieu de {attendu}.")
    if sorted([b, c, a]) != [c, a, b] or max([a, b, c]) != b or min([a, b, c]) != c:
        raise AssertionError("sorted, max et min doivent ranger les montants par centimes croissants.")
    for calcul, quoi in ((lambda: Montant(100, "USD") < Montant(200), "Montant(100, 'USD') < Montant(200)"),
                         (lambda: Montant(300, "USD") > Montant(200), "Montant(300, 'USD') > Montant(200)"),
                         (lambda: Montant(300, "USD") <= Montant(200), "Montant(300, 'USD') <= Montant(200)"),
                         (lambda: sorted([Montant(2), Montant(1, "USD")]), "sorted([Montant(2), Montant(1, 'USD')])")):
        _leve(calcul, ValueError, quoi, "@dataclass(order=True) compare (centimes, devise) comme des tuples : 1 $ < 2 € "
              "vaudrait True. Écrivez __lt__ vous-même, et vérifiez la devise.")
    _leve(lambda: Montant(100) < 5, TypeError, "Montant(100) < 5", "Renvoyez NotImplemented si l'autre n'est pas un Montant.")
    try:
        egal = Montant(100) == 100
    except Exception as exc:
        raise AssertionError(f"Montant(100) == 100 lève {_decrire(exc)} : == doit simplement valoir False.") from None
    if egal is not False:
        raise AssertionError("Montant(100) == 100 doit valoir False (un montant n'est pas un nombre).")


# ============================================================================ Défi 107 : gestionnaire de contexte


def _attendu(journal: list[str], attendu: list[str], quand: str, conseil: str = "") -> None:
    if journal != attendu:
        raise AssertionError(f"{quand}, la passerelle a reçu {journal} au lieu de {attendu}. {conseil}".rstrip())


def verifier_campagne(campagne) -> None:
    passerelle = Passerelle()
    gestionnaire = campagne(passerelle, "Fidélité")
    if passerelle.journal:
        raise AssertionError(f"campagne(passerelle, \"Fidélité\") a déjà appelé {passerelle.journal} AVANT d'entrer dans le "
                             "`with` : rien ne doit être ouvert tant qu'on n'y est pas entré.")
    with gestionnaire as lot:
        if not passerelle.lots:
            raise AssertionError("En entrant dans le `with`, ouvrez le lot : lot = passerelle.ouvrir(nom).")
        if lot is not passerelle.lots[-1]:
            raise AssertionError(f"`with campagne(...) as lot` reçoit {_court(lot)} : le `as` doit recevoir le lot renvoyé "
                                 "par passerelle.ouvrir(nom) (avec @contextmanager : `yield lot`).")
        lot.ajouter("C001")
        lot.ajouter("C002")
    quand = "Quand le corps du `with` va au bout"
    if "fermer" in passerelle.journal and "valider" not in passerelle.journal:
        raise AssertionError(f"{quand}, le lot n'a pas été validé : aucun SMS n'est parti ({passerelle.journal}).")
    _attendu(passerelle.journal, ["ouvrir Fidélité", "ajouter C001", "ajouter C002", "valider", "fermer"], quand)
    if passerelle.envoyes != ["C001", "C002"]:
        raise AssertionError(f"{quand}, les SMS envoyés sont {passerelle.envoyes} au lieu de ['C001', 'C002'].")

    # ------------------------------------------------------------------ erreur dans le corps
    for erreur, quand in ((ValueError("fichier de cibles corrompu"), "Quand le corps du `with` lève ValueError"),
                          (KeyboardInterrupt(), "Quand l'utilisateur appuie sur Ctrl+C (KeyboardInterrupt) dans le `with`")):
        passerelle = Passerelle()
        remontee: BaseException | None = None
        try:
            with campagne(passerelle, "Relance") as lot:
                lot.ajouter("C001")
                raise erreur
        except BaseException as exc:  # noqa: B036 — on veut voir passer même KeyboardInterrupt
            remontee = exc
        journal = passerelle.journal
        if "valider" in journal:
            raise AssertionError(f"{quand}, le lot a été VALIDÉ ({journal}) : un envoi à moitié préparé est parti. Ne validez "
                                 "que si le corps est allé au bout : dans le `else:` du try, pas dans le `finally:`.")
        if "annuler" not in journal and "fermer" not in journal:
            raise AssertionError(f"{quand}, le lot n'a été ni annulé ni fermé ({journal}). L'erreur du corps est relevée AU "
                                 "NIVEAU DU `yield` : les lignes qui le suivent ne s'exécutent pas. Entourez le yield d'un "
                                 "try / except / finally.")
        if "annuler" not in journal:
            conseil = (" `except Exception` ne voit pas KeyboardInterrupt, qui hérite directement de BaseException. Ici on "
                       "attrape pour nettoyer, puis on relance : `except BaseException:` est donc permis."
                       if isinstance(erreur, KeyboardInterrupt) else "")
            raise AssertionError(f"{quand}, le lot n'a pas été annulé ({journal}).{conseil}")
        if "fermer" not in journal:
            raise AssertionError(f"{quand}, le lot n'a pas été fermé ({journal}) : `fermer` va dans un `finally:`.")
        _attendu(journal, ["ouvrir Relance", "ajouter C001", "annuler", "fermer"], quand)
        if remontee is None:
            raise AssertionError(f"{quand}, l'erreur a disparu : votre code l'a AVALÉE, et l'appelant croit que la campagne "
                                 "est partie. Après avoir annulé, relancez-la avec `raise` (sans argument). Avec une "
                                 "classe, __exit__ doit renvoyer False ou None, pas True.")
        if remontee is not erreur:
            raise AssertionError(f"{quand}, c'est {_decrire(remontee)} qui remonte au lieu de l'erreur d'origine "
                                 f"({_decrire(erreur)}) : relancez la MÊME, avec `raise` seul.")

    # ------------------------------------------------------------------ pannes de la passerelle
    passerelle = Passerelle(panne="ouvrir")
    corps = False
    quand = "Quand passerelle.ouvrir() lève PasserelleIndisponible"
    try:
        with campagne(passerelle, "Panne"):
            corps = True
    except PasserelleIndisponible:
        pass
    except BaseException as exc:  # noqa: B036
        raise AssertionError(f"{quand}, c'est {_decrire(exc)} qui remonte. Si ouvrir() échoue, il n'y a pas de lot : "
                             "ouvrez AVANT le try, pour ne pas annuler ou fermer un lot qui n'existe pas.") from None
    else:
        raise AssertionError(f"{quand}, l'erreur a disparu{' et le corps du with s’est exécuté' if corps else ''} : "
                             "elle doit remonter.")
    if corps:
        raise AssertionError(f"{quand}, le corps du `with` ne doit pas s'exécuter.")
    _attendu(passerelle.journal, ["ouvrir Panne"], quand, "Si ouvrir() échoue, il n'y a pas de lot : ouvrez AVANT le "
             "try, pour ne pas annuler ou fermer un lot qui n'existe pas.")

    passerelle = Passerelle(panne="valider")
    quand = "Quand passerelle.valider() lève PasserelleIndisponible"
    try:
        with campagne(passerelle, "Refus") as lot:
            lot.ajouter("C003")
    except PasserelleIndisponible:
        pass
    except BaseException as exc:  # noqa: B036
        raise AssertionError(f"{quand}, c'est {_decrire(exc)} qui remonte au lieu de PasserelleIndisponible.") from None
    else:
        raise AssertionError(f"{quand}, l'erreur a disparu : l'appelant doit savoir que l'opérateur a refusé le lot.")
    if "fermer" not in passerelle.journal:
        raise AssertionError(f"{quand}, le lot n'a pas été fermé ({passerelle.journal}) : `fermer` va dans un `finally:`, "
                             "qui s'exécute même si valider() échoue.")
    if "annuler" in passerelle.journal:
        raise AssertionError(f"{quand}, le lot a été annulé ({passerelle.journal}) alors que l'opérateur l'a déjà refusé. "
                             "valider() est sans doute DANS le try, juste après le yield : son erreur passe alors par votre "
                             "except. Mettez-la dans le `else:` du try : il ne s'exécute que si le corps a réussi, et ses "
                             "erreurs ne passent pas par l'except.")
    _attendu(passerelle.journal, ["ouvrir Refus", "ajouter C003", "valider", "fermer"], quand)

    # ------------------------------------------------------------------ return et réutilisation
    passerelle = Passerelle()

    def premier_lot():
        with campagne(passerelle, "Retour") as lot:
            lot.ajouter("C007")
            return "parti"

    if premier_lot() != "parti":
        raise AssertionError("Un `return` dans le corps du `with` doit renvoyer sa valeur normalement.")
    _attendu(passerelle.journal, ["ouvrir Retour", "ajouter C007", "valider", "fermer"],
             "Quand le corps du `with` en sort par un `return` (c'est une sortie normale)")
    with campagne(passerelle, "Encore") as lot:
        lot.ajouter("C008")
    if passerelle.envoyes != ["C007", "C008"] or len(passerelle.lots) != 2:
        raise AssertionError(f"Deux campagnes à la suite sur la même passerelle : SMS envoyés {passerelle.envoyes} au lieu "
                             "de ['C007', 'C008'] (chaque `with` ouvre son propre lot).")


# ============================================================================ Défi 108 : collection


def _paires(valeur) -> list:
    return [tuple(p) if isinstance(p, (list, tuple)) else p for p in valeur]


def verifier_Portefeuille(Portefeuille) -> None:
    vide = Portefeuille()
    p = Portefeuille([("C003", 0.4), ("C001", 0.9), ("C002", 0.4)])
    attendu = [("C001", 0.9), ("C002", 0.4), ("C003", 0.4)]

    if len(vide) != 0 or len(p) != 3:
        raise AssertionError(f"len() vaut {len(vide)} pour un portefeuille vide et {len(p)} pour 3 clients.")
    if bool(vide) or not bool(p):
        raise AssertionError("Un portefeuille vide doit être « faux » (if p: …) et un portefeuille rempli « vrai ».")
    for objet, texte in ((vide, "Portefeuille(vide)"), (p, "Portefeuille(3 clients)"),
                         (Portefeuille([("C001", 0.5)]), "Portefeuille(1 client)")):
        if repr(objet) != texte:
            raise AssertionError(f"repr() vaut {repr(objet)!r} au lieu de {texte!r}.")

    # ------------------------------------------------------------------ in et []
    if "C001" not in p:
        conseil = ""
        if not hasattr(type(p), "__contains__"):
            conseil = (" Sans __contains__, Python fait `in` en parcourant __iter__, et compare \"C001\" à chaque couple "
                       "(\"C001\", 0.9) : jamais égal. Écrivez __contains__.")
        raise AssertionError(f"\"C001\" in portefeuille vaut False alors que C001 y est.{conseil}")
    if "C999" in p or "C001" in vide:
        raise AssertionError("`in` répond True pour un client absent.")
    if p["C001"] != 0.9 or p["C003"] != 0.4:
        raise AssertionError(f"portefeuille[\"C001\"] vaut {_court(p['C001'])} au lieu de 0.9.")
    _leve(lambda: p["C999"], KeyError, "portefeuille[\"C999\"]",
          "Comme un dictionnaire : un client inconnu lève KeyError (renvoyer None cacherait l'erreur plus loin).")

    # ------------------------------------------------------------------ for
    premier = list(p)
    if _paires(premier) != attendu:
        raise AssertionError(f"list(portefeuille) vaut {_court(premier)} au lieu de {attendu} : du plus risqué au moins "
                             "risqué, et à score égal dans l'ordre des identifiants. Avec sorted : "
                             "key=lambda paire: (-paire[1], paire[0]).")
    if premier != attendu:
        raise AssertionError(f"Chaque élément doit être un couple (tuple) (client_id, score) : obtenu {_court(premier)}.")
    deuxieme = list(p)
    if deuxieme != attendu:
        raise AssertionError(f"Au DEUXIÈME parcours, list(portefeuille) vaut {_court(deuxieme)} : votre __iter__ renvoie "
                             "sans doute self (un itérateur, qui s'épuise). Renvoyez un NOUVEL itérateur à chaque appel : "
                             "return iter(sorted(...)), ou écrivez __iter__ avec yield.")
    couples = [(a, b) for a, _ in p for b, _ in p]
    if len(couples) != 9:
        raise AssertionError(f"Deux boucles imbriquées sur un portefeuille de 3 clients donnent {len(couples)} couples au "
                             "lieu de 9 : les deux boucles partagent le même itérateur. __iter__ doit en fabriquer un neuf.")

    # ------------------------------------------------------------------ ajouter
    _leve(lambda: p.ajouter("C001", 0.1), ValueError, "portefeuille.ajouter(\"C001\", 0.1) (C001 y est déjà)",
          "Un doublon cache souvent une erreur dans le fichier source : on refuse, on ne remplace pas.")
    if p["C001"] != 0.9 or len(p) != 3:
        raise AssertionError("Après un ajout refusé, le portefeuille ne doit pas avoir changé.")
    _leve(lambda: Portefeuille([("C1", 0.5), ("C1", 0.6)]), ValueError, "Portefeuille([(\"C1\", 0.5), (\"C1\", 0.6)])")

    q = Portefeuille([("C001", 0.9), ("C002", 0.5)])
    try:
        for client_id, _ in q:
            if client_id == "C001":
                q.ajouter("C010", 0.95)
    except RuntimeError as exc:
        raise AssertionError(f"Ajouter un client pendant un parcours lève {_decrire(exc)} : vous parcourez le dictionnaire "
                             "lui-même. Parcourez une copie (sorted(...) en fabrique une).") from None
    if "C010" not in q or q["C010"] != 0.95 or len(q) != 3:
        raise AssertionError("Un client ajouté pendant un parcours doit être dans le portefeuille ensuite.")

    # ------------------------------------------------------------------ au_dessus
    tous = p.au_dessus(0.4)
    if not isinstance(tous, Portefeuille):
        raise AssertionError(f"portefeuille.au_dessus(0.4) renvoie {_court(tous)} : renvoyez un Portefeuille.")
    if _paires(tous) != attendu:
        raise AssertionError(f"portefeuille.au_dessus(0.4) contient {_court(list(tous))} au lieu de {attendu} : le seuil "
                             "est inclus (score >= seuil).")
    alertes = p.au_dessus(0.5)
    if _paires(alertes) != [("C001", 0.9)]:
        raise AssertionError(f"portefeuille.au_dessus(0.5) contient {_court(list(alertes))} au lieu de [('C001', 0.9)].")
    alertes.ajouter("C050", 0.99)
    p.ajouter("C060", 0.8)
    if "C050" in p or "C060" in alertes or len(p) != 4 or len(alertes) != 2:
        raise AssertionError("Après au_dessus, les deux portefeuilles doivent être indépendants : un ajout dans l'un ne "
                             "doit pas apparaître dans l'autre (ne partagez pas le même dictionnaire).")
    if repr(p.au_dessus(1.0)) != "Portefeuille(vide)":
        raise AssertionError("portefeuille.au_dessus(1.0) doit être un portefeuille vide.")

    # ------------------------------------------------------------------ 20 000 clients
    n, limite = 20_000, 1.0
    grand = Portefeuille()
    debut = time.perf_counter()
    for k in range(n):
        grand.ajouter(f"C{k:05d}", (k * 7919 % 10007) / 10007)
        if time.perf_counter() - debut > limite:
            raise AssertionError(f"Seulement {k + 1} clients ajoutés en {limite:.0f} s : ajouter() parcourt sans doute une "
                                 "liste pour chercher les doublons (O(n) par ajout, O(n²) en tout). Un dictionnaire "
                                 "répond en O(1).")
    if len(grand) != n:
        raise AssertionError(f"Après {n} ajouts, len() vaut {len(grand)}.")
    cherches = [f"C{(k * 37) % (2 * n):05d}" for k in range(5_000)]   # la moitié existe, l'autre non
    for nom, operation in (("`in`", lambda cid: cid in grand), ("[]", lambda cid: cid in grand and grand[cid])):
        debut = time.perf_counter()
        for k, client_id in enumerate(cherches):
            operation(client_id)
            if time.perf_counter() - debut > 0.5:
                raise AssertionError(f"Seulement {k + 1} recherches avec {nom} en 0,5 s sur {n} clients : votre code "
                                     "parcourt une liste à chaque fois (O(n)). Un dictionnaire répond en O(1).")
    ordre = [score for _, score in grand]
    if len(ordre) != n or any(x < y for x, y in zip(ordre, ordre[1:])):
        raise AssertionError("Sur 20 000 clients, le parcours n'est pas trié du plus risqué au moins risqué.")


# ============================================================================ Défi 109 : opérateurs


def _clients_109() -> list[dict]:
    return [{"client_id": f"C{k:03d}", "contrat": contrat, "anciennete_mois": anciennete, "appels_support": appels}
            for k, (contrat, anciennete, appels) in enumerate(itertools.product(CONTRATS, (2, 30), (0, 5)), start=1)]


def verifier_Regle(Regle) -> None:
    toujours = Regle("toujours", lambda c: True)

    # ------------------------------------------------------------------ création et appel
    for nom in ("", "   "):
        _leve(lambda: Regle(nom, lambda c: True), ValueError, f"Regle({nom!r}, ...)",
              "Une règle sans nom serait illisible dans les rapports.")
    for condition in ("contrat == 'Mensuel'", None):
        _leve(lambda: Regle("x", condition), TypeError, f"Regle(\"x\", {condition!r})", "Vérifiez callable(condition).")
    if str(toujours) != "toujours":
        raise AssertionError(f"str(Regle(\"toujours\", ...)) vaut {str(toujours)!r} au lieu de 'toujours'.")
    appels = Regle("appels", lambda c: c["appels_support"])
    for client, attendu in (({"appels_support": 3}, True), ({"appels_support": 0}, False)):
        obtenu = appels(client)
        if obtenu is not attendu:
            raise AssertionError(f"Avec une condition qui renvoie {client['appels_support']}, regle(client) renvoie "
                                 f"{_court(obtenu)} au lieu de {attendu} : renvoyez un VRAI booléen, bool(...).")
    if Regle("vide", lambda c: "")({}) is not False:
        raise AssertionError("Avec une condition qui renvoie \"\", regle(client) doit renvoyer False.")

    # ------------------------------------------------------------------ combinaisons et affichage
    compteurs: collections.Counter[str] = collections.Counter()

    def suivie(nom, condition):
        def condition_notee(client):
            compteurs[nom] += 1
            return condition(client)

        return Regle(nom, condition_notee)

    mensuel = suivie("mensuel", lambda c: c["contrat"] == "Mensuel")
    recent = suivie("récent", lambda c: c["anciennete_mois"] < 6)
    mecontent = suivie("mécontent", lambda c: c["appels_support"] >= 3)

    et = mensuel & recent
    if not isinstance(et, Regle):
        raise AssertionError(f"mensuel & récent renvoie {_court(et)} : renvoyez une Regle (méthode __and__).")
    if et is mensuel or et is recent:
        raise AssertionError("mensuel & récent renvoie l'une des deux règles : renvoyez une NOUVELLE Regle, sans modifier "
                             "les deux autres (elles servent peut-être ailleurs).")
    for regle, texte in ((et, "(mensuel & récent)"), (mensuel | recent, "(mensuel | récent)"), (~mensuel, "~mensuel"),
                         (~(mensuel & recent), "~(mensuel & récent)"), (~~mensuel, "~~mensuel"),
                         (mensuel | recent & mecontent, "(mensuel | (récent & mécontent))"),
                         (~mensuel & recent, "(~mensuel & récent)"),
                         (mensuel & (recent | mecontent), "(mensuel & (récent | mécontent))")):
        if not isinstance(regle, Regle):
            raise AssertionError(f"La combinaison {texte} renvoie {_court(regle)} au lieu d'une Regle.")
        if str(regle) != texte:
            raise AssertionError(f"str() de la combinaison vaut {str(regle)!r} au lieu de {texte!r} : une paire de "
                                 "parenthèses par & ou |, et ~ collé devant.")
    if (str(mensuel), str(recent)) != ("mensuel", "récent"):
        raise AssertionError(f"Après les combinaisons, mensuel s'affiche {str(mensuel)!r} : la règle d'origine a été "
                             "modifiée. & | ~ doivent renvoyer une NOUVELLE règle.")

    clients = _clients_109()
    for client in clients:
        resultat = mensuel(client)
        if resultat is not (client["contrat"] == "Mensuel"):
            raise AssertionError(f"Après les combinaisons, mensuel({client}) vaut {_court(resultat)} : la règle d'origine "
                                 "a été modifiée (par exemple en ajoutant l'autre règle à une liste DANS mensuel).")

    def m(c):
        return c["contrat"] == "Mensuel"

    def r(c):
        return c["anciennete_mois"] < 6

    def x(c):
        return c["appels_support"] >= 3

    table = [
        ("mensuel & récent", lambda: mensuel & recent, lambda c: m(c) and r(c)),
        ("mensuel | récent", lambda: mensuel | recent, lambda c: m(c) or r(c)),
        ("~mensuel", lambda: ~mensuel, lambda c: not m(c)),
        ("mensuel & (récent | mécontent)", lambda: mensuel & (recent | mecontent), lambda c: m(c) and (r(c) or x(c))),
        ("~(mensuel | récent) & mécontent", lambda: ~(mensuel | recent) & mecontent,
         lambda c: (not (m(c) or r(c))) and x(c)),
        ("(mensuel & récent) | (~mensuel & mécontent)", lambda: (mensuel & recent) | (~mensuel & mecontent),
         lambda c: (m(c) and r(c)) or ((not m(c)) and x(c))),
        ("~~mensuel", lambda: ~~mensuel, m),
    ]
    for texte, fabriquer, reference in table:
        regle = fabriquer()
        for client in clients:
            obtenu, attendu = regle(client), reference(client)
            if obtenu is not attendu:
                raise AssertionError(f"({texte})(client) vaut {_court(obtenu)} au lieu de {attendu} pour le client "
                                     f"{client}.")

    # ------------------------------------------------------------------ court-circuit
    client = clients[0]   # Mensuel, 2 mois, 0 appel
    annuel = suivie("annuel", lambda c: c["contrat"] == "Annuel")
    for texte, fabriquer, evaluees in (("(annuel & mécontent)", lambda: annuel & mecontent, 0),
                                       ("(mensuel | mécontent)", lambda: mensuel | mecontent, 0),
                                       ("(mensuel & mécontent)", lambda: mensuel & mecontent, 1),
                                       ("(annuel | mécontent)", lambda: annuel | mecontent, 1)):
        regle = fabriquer()
        compteurs.clear()
        regle(client)
        if compteurs["mécontent"] != evaluees:
            if evaluees == 0:
                raise AssertionError(f"Pour un client mensuel, {texte}(client) a évalué « mécontent » alors que le résultat "
                                     "était déjà connu avec la première règle. `a(c) & b(c)` évalue toujours les deux "
                                     "côtés ; `a(c) and b(c)` (ou `or`) s'arrête dès que c'est joué. La seconde règle peut "
                                     "être lente (un appel au modèle) ou planter sur un client que la première écarte.")
            raise AssertionError(f"{texte}(client) a évalué « mécontent » {compteurs['mécontent']} fois au lieu de 1.")
        if max(compteurs.values(), default=0) > 1:
            raise AssertionError(f"{texte}(client) a évalué une règle {max(compteurs.values())} fois : une fois suffit.")

    # ------------------------------------------------------------------ autre chose qu'une règle
    for calcul, quoi in ((lambda: mensuel & True, "mensuel & True"), (lambda: True & mensuel, "True & mensuel"),
                         (lambda: mensuel | (lambda c: True), "mensuel | (lambda c: True)"),
                         (lambda: mensuel & "récent", "mensuel & \"récent\"")):
        _leve(calcul, TypeError, quoi, "Si l'autre n'est pas une Regle, `return NotImplemented` : Python lèvera TypeError.")

    # ------------------------------------------------------------------ and, or, not, if
    for calcul, quoi in ((lambda: bool(mensuel), "bool(mensuel)"), (lambda: mensuel and recent, "mensuel and récent"),
                         (lambda: mensuel or recent, "mensuel or récent"), (lambda: not mensuel, "not mensuel"),
                         (lambda: (mensuel & recent) and mecontent, "(mensuel & récent) and mécontent")):
        exc = _leve(calcul, TypeError, quoi,
                    "Python ne permet pas de redéfinir and / or / not : ils demandent bool(règle), et par défaut tout objet "
                    "est vrai, donc `mensuel and récent` renvoie récent, sans un mot. Écrivez __bool__ et levez TypeError.")
        if "&" not in str(exc):
            raise AssertionError(f"{quoi} lève bien TypeError, mais le message doit dire d'utiliser & | ~ : "
                                 f"obtenu {str(exc)!r}.")

    exemple = mensuel & (recent | mecontent)
    cibles = [c["client_id"] for c in CLIENTS_EXEMPLE if exemple(c)]
    if cibles != ["C001", "C002"]:
        raise AssertionError(f"Sur CLIENTS_EXEMPLE, mensuel & (récent | mécontent) cible {cibles} au lieu de "
                             "['C001', 'C002'].")


SERIE = Serie(
    code="17",
    titre="Concevoir ses propres types : dataclass, Enum, objets valeur, with, collections, opérateurs",
    module_eleve="defis.serie_17",
    module_corrige="defis.corrige_serie_17",
    defis=(
        Defi(104, "Client", "🟢", "—", verifier_Client),
        Defi(105, "Contrat", "🟢", "—", verifier_Contrat),
        Defi(106, "Montant", "🟡", "—", verifier_Montant),
        Defi(107, "campagne", "🟡", "—", verifier_campagne),
        Defi(108, "Portefeuille", "🟡", "—", verifier_Portefeuille),
        Defi(109, "Regle", "🔴", "—", verifier_Regle),
    ),
)

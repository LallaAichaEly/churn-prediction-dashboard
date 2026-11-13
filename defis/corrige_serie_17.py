"""Série 17 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_17.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la situation à laquelle vous n'aviez pas pensé.

Fil rouge de la série : un TYPE, c'est une promesse. Quand une fonction reçoit un `Client`, un
`Montant` ou un `Contrat`, elle n'a plus à se demander si l'ancienneté est négative, si le prix est
en euros ou en centimes, si « mensuel » s'écrit avec une majuscule : c'est vérifié UNE fois, à la
création, et plus jamais ensuite. Trois réflexes reviennent dans chaque défi :

1. refuser tôt et clairement : TypeError pour un mauvais type, ValueError pour une mauvaise valeur,
   et un message qui nomme le champ fautif ;
2. un objet qu'on ne peut pas modifier (frozen) ne peut pas être abîmé en douce par une autre
   fonction ; et ce qu'il contient doit l'être aussi (un tuple, pas une liste) ;
3. les méthodes spéciales (`__add__`, `__lt__`, `__iter__`, `__enter__`, `__bool__`…) branchent
   vos objets sur la syntaxe de Python : `+`, `<`, `for`, `with`, `if`. Quand une opération n'a pas
   de sens, on renvoie `NotImplemented` (Python lèvera TypeError) ou on lève une erreur explicite,
   mais on ne renvoie jamais un résultat faux.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import total_ordering
from typing import Any, Callable, Iterable, Iterator

from defis.conception import (  # noqa: F401
    CLIENTS_EXEMPLE,
    CONTRATS,
    SCORES_EXEMPLE,
    Lot,
    Passerelle,
    PasserelleIndisponible,
)

# ============================================================================ dataclass


@dataclass(frozen=True)
class Client:
    client_id: str
    anciennete_mois: int
    facture_mensuelle: float
    contrat: str = "Mensuel"
    options: tuple[str, ...] = ()   # un tuple vide est immuable : il peut servir de valeur par défaut

    def __post_init__(self) -> None:
        # __post_init__ est appelée par le __init__ que @dataclass a écrit, juste après avoir rangé
        # les champs. C'est LE seul endroit où valider : aucun Client invalide ne peut exister.
        if not isinstance(self.client_id, str):
            raise TypeError(f"client_id doit être un texte, pas {type(self.client_id).__name__}")
        if not re.fullmatch(r"C\d+", self.client_id):
            raise ValueError(f"client_id invalide : {self.client_id!r} (attendu : C suivi de chiffres, comme 'C001')")

        # bool hérite de int : isinstance(True, int) vaut True. Il faut l'exclure à la main.
        if isinstance(self.anciennete_mois, bool) or not isinstance(self.anciennete_mois, int):
            raise TypeError(f"anciennete_mois doit être un entier, pas {type(self.anciennete_mois).__name__}")
        if self.anciennete_mois < 0:
            raise ValueError(f"anciennete_mois ne peut pas être négative : {self.anciennete_mois}")

        if isinstance(self.facture_mensuelle, bool) or not isinstance(self.facture_mensuelle, (int, float)):
            raise TypeError(f"facture_mensuelle doit être un nombre, pas {type(self.facture_mensuelle).__name__}")
        # NaN n'est ni < 0 ni >= 0 : `if facture < 0` le laisse passer. isfinite écarte NaN et l'infini.
        if not math.isfinite(self.facture_mensuelle) or self.facture_mensuelle < 0:
            raise ValueError(f"facture_mensuelle doit être un nombre fini et positif : {self.facture_mensuelle}")

        if self.contrat not in CONTRATS:
            raise ValueError(f"contrat inconnu : {self.contrat!r} (attendu : {', '.join(CONTRATS)})")

        # Un texte est itérable : tuple("tv") donnerait ('t', 'v'). On le refuse avant de convertir.
        if isinstance(self.options, str):
            raise TypeError("options doit être une collection de textes, pas un seul texte : ('tv',) et non 'tv'")
        options = tuple(self.options)
        if not all(isinstance(option, str) for option in options):
            raise TypeError("options ne doit contenir que des textes")
        # frozen=True interdit self.options = … (même ici). object.__setattr__ contourne l'interdiction :
        # c'est l'écriture consacrée, et seulement dans __post_init__. Le tuple rend le client hachable,
        # et la liste de l'appelant peut changer ensuite sans toucher au client.
        object.__setattr__(self, "options", options)


# ============================================================================ Enum


class Contrat(Enum):
    MENSUEL = "Mensuel"
    ANNUEL = "Annuel"
    BIANNUEL = "Biannuel"

    # Attention : TOUT attribut écrit ici (un dictionnaire des engagements, une constante…) deviendrait
    # un quatrième membre. Les méthodes, elles, ne sont pas des membres.

    @classmethod
    def depuis_texte(cls, texte: str | Contrat) -> Contrat:
        if isinstance(texte, cls):        # déjà un Contrat : on le rend tel quel
            return texte
        if not isinstance(texte, str):
            raise TypeError(f"un contrat s'écrit avec un texte, pas {type(texte).__name__}")
        cle = texte.strip().casefold()    # casefold : lower() en plus strict (ß → ss), fait pour comparer
        for membre in cls:                # parcourir une Enum donne ses membres, dans l'ordre
            if membre.value.casefold() == cle:
                return membre
        attendus = ", ".join(membre.value for membre in cls)
        raise ValueError(f"contrat inconnu : {texte!r} (attendu : {attendus})")

    @property
    def engagement_mois(self) -> int:
        # match sur les membres : lisible, et une faute de frappe (Contrat.ANUEL) lève AttributeError.
        match self:
            case Contrat.MENSUEL:
                return 1
            case Contrat.ANNUEL:
                return 12
            case Contrat.BIANNUEL:
                return 24
        raise AssertionError(f"engagement inconnu pour {self!r}")  # pas atteignable : garde-fou

    def __str__(self) -> str:
        return self.value


# ============================================================================ Objet valeur


@total_ordering        # écrit <=, > et >= à partir de __lt__ et de __eq__ (écrit par @dataclass)
@dataclass(frozen=True)  # PAS order=True : il comparerait (centimes, devise) comme un tuple, devises mêlées
class Montant:
    centimes: int
    devise: str = "EUR"

    def __post_init__(self) -> None:
        if isinstance(self.centimes, bool) or not isinstance(self.centimes, int):
            raise TypeError(f"centimes doit être un entier, pas {type(self.centimes).__name__} : "
                            "pour des euros, utilisez Montant.depuis_euros(...)")
        if not isinstance(self.devise, str) or not re.fullmatch(r"[A-Z]{3}", self.devise):
            raise ValueError(f"devise invalide : {self.devise!r} (3 lettres majuscules, comme 'EUR')")

    @classmethod
    def depuis_euros(cls, euros: str | int | Decimal, devise: str = "EUR") -> Montant:
        # Un float n'est pas exact : 19.99 est en mémoire 19.989999999999998…, et int(19.99 * 100)
        # vaut 1998. On le refuse plutôt que de deviner : l'appelant passe "19.99" ou Decimal("19.99").
        if isinstance(euros, bool) or not isinstance(euros, (str, int, Decimal)):
            raise TypeError(f"montant en {type(euros).__name__} refusé : passez un texte ('19,99'), "
                            "un entier ou un Decimal (un float n'est pas exact)")
        if isinstance(euros, str):
            texte = euros.strip()
            for espace in (" ", " ", " "):   # espaces des milliers, y compris insécables (Excel)
                texte = texte.replace(espace, "")
            try:
                valeur = Decimal(texte.replace(",", "."))
            except InvalidOperation:
                # InvalidOperation n'hérite PAS de ValueError : on traduit, sinon l'appelant ne l'attrape pas.
                raise ValueError(f"montant illisible : {euros!r}") from None
        else:
            valeur = Decimal(euros)
        if not valeur.is_finite():   # Decimal("NaN") et Decimal("Infinity") existent
            raise ValueError(f"montant illisible : {euros!r}")
        centimes = valeur * 100
        if centimes != centimes.to_integral_value():
            raise ValueError(f"au plus 2 décimales : {euros!r}")
        return cls(int(centimes), devise)

    def _meme_devise(self, autre: Montant, action: str) -> None:
        if self.devise != autre.devise:
            raise ValueError(f"impossible de {action} {self.devise} et {autre.devise} sans taux de change")

    def __add__(self, autre: object) -> Montant:
        if not isinstance(autre, Montant):
            return NotImplemented   # « je ne sais pas faire » : Python essaiera autre.__radd__, puis TypeError
        self._meme_devise(autre, "additionner")
        return Montant(self.centimes + autre.centimes, self.devise)

    def __radd__(self, autre: object) -> Montant:
        # sum(montants) calcule 0 + m1 + m2… : int.__add__(0, m1) ne sait pas faire, Python appelle
        # alors m1.__radd__(0). On accepte ce 0-là, et rien d'autre.
        if type(autre) is int and autre == 0:
            return self
        return NotImplemented

    def __mul__(self, n: object) -> Montant:
        if isinstance(n, bool) or not isinstance(n, int):
            return NotImplemented   # 1,5 × 19,99 € tomberait entre deux centimes : à l'appelant d'arrondir
        return Montant(self.centimes * n, self.devise)

    __rmul__ = __mul__   # 3 * montant : int.__mul__ ne sait pas, Python appelle montant.__rmul__(3)

    def __lt__(self, autre: object) -> bool:
        if not isinstance(autre, Montant):
            return NotImplemented
        self._meme_devise(autre, "comparer")
        return self.centimes < autre.centimes

    def __str__(self) -> str:
        # divmod sur la valeur ABSOLUE : -5 // 100 vaut -1 et -5 % 100 vaut 95 (« -1,95 € » au lieu de « -0,05 € »).
        signe = "-" if self.centimes < 0 else ""
        euros, centimes = divmod(abs(self.centimes), 100)
        entier = f"{euros:,}".replace(",", " ")
        symbole = "€" if self.devise == "EUR" else self.devise
        return f"{signe}{entier},{centimes:02d} {symbole}"


# ============================================================================ Gestionnaire de contexte


@contextmanager
def campagne(passerelle: Passerelle, nom: str) -> Iterator[Lot]:
    # Tout ce qui est avant le yield s'exécute en ENTRANT dans le `with`, tout ce qui est après en SORTANT.
    # Si ouvrir() échoue, on n'est pas encore dans le try : rien à annuler ni à fermer.
    lot = passerelle.ouvrir(nom)
    try:
        yield lot                  # le corps du `with` s'exécute ici ; une erreur du corps est relevée ICI
    except BaseException:
        # BaseException, pas Exception : un Ctrl+C (KeyboardInterrupt) au milieu de la préparation ne doit
        # pas envoyer un lot à moitié rempli. On n'avale rien : `raise` relance la MÊME erreur.
        passerelle.annuler(lot)
        raise
    else:
        passerelle.valider(lot)    # seulement si le corps est allé au bout (un `return` compte aussi)
    finally:
        passerelle.fermer(lot)     # dans tous les cas, même si valider() échoue

    # La même chose avec une classe (utile quand l'objet a d'autres méthodes) :
    #
    #   class Campagne:
    #       def __init__(self, passerelle, nom): self.passerelle, self.nom = passerelle, nom
    #       def __enter__(self):
    #           self.lot = self.passerelle.ouvrir(self.nom)
    #           return self.lot
    #       def __exit__(self, type_erreur, erreur, traceback):
    #           try:
    #               if type_erreur is None:
    #                   self.passerelle.valider(self.lot)
    #               else:
    #                   self.passerelle.annuler(self.lot)
    #           finally:
    #               self.passerelle.fermer(self.lot)
    #           return False   # True AVALERAIT l'erreur du corps : l'appelant croirait que tout s'est bien passé


# ============================================================================ Collection


class Portefeuille:
    def __init__(self, paires: Iterable[tuple[str, float]] = ()) -> None:
        # Un dictionnaire : `in` et `[]` en O(1), quel que soit le nombre de clients (série 07, défi 45).
        self._scores: dict[str, float] = {}
        for client_id, score in paires:
            self.ajouter(client_id, score)

    def ajouter(self, client_id: str, score: float) -> None:
        if client_id in self._scores:
            raise ValueError(f"{client_id} est déjà dans le portefeuille (score {self._scores[client_id]})")
        self._scores[client_id] = score

    def __len__(self) -> int:
        return len(self._scores)       # donne aussi bool() : un portefeuille vide est « faux »

    def __contains__(self, client_id: object) -> bool:
        # Sans cette méthode, `in` parcourrait __iter__ et comparerait "C001" à ("C001", 0.82) : toujours False.
        return client_id in self._scores

    def __getitem__(self, client_id: str) -> float:
        return self._scores[client_id]  # KeyError pour un client inconnu, comme un dictionnaire

    def __iter__(self) -> Iterator[tuple[str, float]]:
        # Un NOUVEL itérateur à chaque appel : on peut parcourir deux fois, ou imbriquer deux boucles.
        # sorted() fabrique une liste : c'est une photo, qu'un ajouter() pendant la boucle ne dérange pas.
        return iter(sorted(self._scores.items(), key=lambda paire: (-paire[1], paire[0])))

    def au_dessus(self, seuil: float) -> Portefeuille:
        # type(self)(…) plutôt que Portefeuille(…) : une sous-classe obtiendrait un objet de SON type.
        return type(self)((client_id, score) for client_id, score in self._scores.items() if score >= seuil)

    def __repr__(self) -> str:
        n = len(self)
        return f"{type(self).__name__}({'vide' if n == 0 else f'{n} client' + ('s' if n > 1 else '')})"


# ============================================================================ Opérateurs


class Regle:
    def __init__(self, nom: str, condition: Callable[[Any], object]) -> None:
        if not isinstance(nom, str) or not nom.strip():
            raise ValueError("une règle a besoin d'un nom (il apparaît dans les rapports)")
        if not callable(condition):
            raise TypeError(f"condition doit être une fonction client -> bool, pas {type(condition).__name__}")
        self.nom = nom
        self._condition = condition

    def __call__(self, client: Any) -> bool:
        return bool(self._condition(client))   # un vrai booléen, même si la condition renvoie 3 ou ""

    def __and__(self, autre: object) -> Regle:
        if not isinstance(autre, Regle):
            return NotImplemented
        # Une NOUVELLE règle : `mensuel & recent` ne modifie ni mensuel ni recent.
        # Le `and` de Python court-circuite : si la première est fausse, la seconde n'est pas évaluée.
        return Regle(f"({self} & {autre})", lambda client: self(client) and autre(client))

    def __or__(self, autre: object) -> Regle:
        if not isinstance(autre, Regle):
            return NotImplemented
        return Regle(f"({self} | {autre})", lambda client: self(client) or autre(client))

    def __invert__(self) -> Regle:
        return Regle(f"~{self}", lambda client: not self(client))

    def __str__(self) -> str:
        return self.nom

    def __repr__(self) -> str:
        return f"Regle({self.nom!r})"

    def __bool__(self) -> bool:
        # `and`, `or`, `not` et `if` ne PEUVENT PAS être redéfinis : ils demandent bool(règle). Par défaut,
        # tout objet est « vrai », et `mensuel and recent` renverrait… recent, en silence. pandas fait pareil :
        # « The truth value of a Series is ambiguous ». On préfère une erreur claire.
        raise TypeError(f"la règle {self.nom!r} n'est ni vraie ni fausse : combinez avec & | ~ (pas and, or, not), "
                        "et évaluez-la sur un client : regle(client)")


# ============================================================================ Tests (comme dans tests/)


def test_un_client_est_valide_immuable_et_hachable():
    options = ["tv"]
    client = Client("C001", 3, 49, options=options)
    options.append("fibre")
    assert client.options == ("tv",) and client.contrat == "Mensuel"
    assert len({client, Client("C001", 3, 49.0, "Mensuel", ("tv",))}) == 1
    for mauvais, erreur in ((dict(anciennete_mois=True), TypeError), (dict(facture_mensuelle=math.nan), ValueError),
                            (dict(options="tv"), TypeError), (dict(contrat="mensuel"), ValueError)):
        champs = dict(client_id="C001", anciennete_mois=3, facture_mensuelle=49.0) | mauvais
        try:
            Client(**champs)
        except erreur as exc:
            assert next(iter(mauvais)) in str(exc)
        else:
            raise AssertionError(f"{mauvais} aurait dû être refusé")


def test_un_contrat_se_lit_comme_un_humain_l_ecrit():
    assert Contrat.depuis_texte("  mensuel ") is Contrat.MENSUEL
    assert Contrat("Biannuel").engagement_mois == 24 and f"{Contrat.ANNUEL}" == "Annuel"
    assert [c.value for c in Contrat] == list(CONTRATS)
    try:
        Contrat.depuis_texte("Mensuelle")
    except ValueError as exc:
        assert "Biannuel" in str(exc)
    else:
        raise AssertionError("un contrat inconnu aurait dû être refusé")


def test_un_montant_ne_perd_jamais_un_centime():
    assert Montant.depuis_euros("19,99") == Montant(1999)
    assert str(sum([Montant(1999)] * 3)) == "59,97 €" and str(Montant(-5)) == "-0,05 €"
    assert str(Montant.depuis_euros("1 234,5") * 2) == "2 469,00 €"
    assert max(Montant(100), Montant(250), Montant(-30)) == Montant(250)
    for appel, erreur in ((lambda: Montant.depuis_euros(19.99), TypeError),
                          (lambda: Montant.depuis_euros("abc"), ValueError),
                          (lambda: Montant(100, "USD") < Montant(200), ValueError),
                          (lambda: Montant(100) + 1, TypeError)):
        try:
            appel()
        except erreur:
            pass
        else:
            raise AssertionError("aurait dû lever " + erreur.__name__)


def test_une_campagne_n_envoie_que_si_tout_s_est_bien_passe():
    passerelle = Passerelle()
    with campagne(passerelle, "Fidélité") as lot:
        lot.ajouter("C001")
    try:
        with campagne(passerelle, "Relance") as lot:
            lot.ajouter("C002")
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert passerelle.envoyes == ["C001"]
    assert passerelle.journal[-3:] == ["ajouter C002", "annuler", "fermer"]


def test_le_portefeuille_se_parcourt_du_plus_risque_au_moins_risque():
    portefeuille = Portefeuille(SCORES_EXEMPLE.items())
    assert [c for c, _ in portefeuille][:3] == ["C004", "C001", "C003"]  # 0,47 et 0,47 : ordre des id
    assert list(portefeuille) == list(portefeuille)                      # on peut le parcourir deux fois
    assert "C002" in portefeuille and portefeuille["C002"] == 0.15
    alertes = portefeuille.au_dessus(0.47)
    alertes.ajouter("C009", 0.99)
    assert len(alertes) == 5 and len(portefeuille) == 5 and "C009" not in portefeuille
    assert repr(Portefeuille()) == "Portefeuille(vide)"


def test_les_regles_se_combinent_sans_and_ni_or():
    appels: list[str] = []

    def mensuel_note(client):
        appels.append(client["client_id"])
        return client["contrat"] == "Mensuel"

    mensuel = Regle("mensuel", mensuel_note)
    recent = Regle("récent", lambda c: c["anciennete_mois"] < 6)
    mecontent = Regle("mécontent", lambda c: c["appels_support"] >= 3)
    cible = ~mensuel & (recent | mecontent)
    assert str(cible) == "(~mensuel & (récent | mécontent))"
    assert [c["client_id"] for c in CLIENTS_EXEMPLE if cible(c)] == ["C003"]
    assert appels == [c["client_id"] for c in CLIENTS_EXEMPLE]  # chaque client n'est testé qu'une fois
    try:
        bool(mensuel and recent)
    except TypeError as exc:
        assert "&" in str(exc)
    else:
        raise AssertionError("`and` aurait dû lever TypeError")

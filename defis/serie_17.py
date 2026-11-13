"""Série 17 — Concevoir ses propres types : dataclass, Enum, objets valeur, `with`, collections, opérateurs.

🧩 VOUS VOULEZ DÉVELOPPER ? UN BON TYPE, C'EST UNE VÉRIFICATION QU'ON N'ÉCRIT QU'UNE FOIS.

Dans un script, un client est un dictionnaire, un prix un float, un contrat un texte. Ça marche…
jusqu'au jour où quelqu'un écrit « mensuel » sans majuscule, passe une ancienneté de -3 mois, ou
additionne des euros et des dollars. Chaque fonction doit alors tout revérifier, et l'une d'elles
oublie. En équipe, on crée plutôt ses propres TYPES : un `Client` qui refuse d'exister s'il est
invalide, un `Montant` qui ne perd jamais un centime, un `Contrat` qui ne peut valoir que trois choses.
Une fois créés, on peut leur faire confiance partout.

Les six défis suivent la campagne de rétention :

    104 un Client qui refuse d'être invalide       105 un Contrat qui ne peut valoir que 3 choses
    106 un Montant qui ne perd jamais un centime    107 un envoi de SMS qui ne part qu'en entier
    108 un portefeuille qu'on parcourt avec `for`   109 des règles de ciblage qu'on combine avec & | ~

Prérequis : la série 00, les exceptions du défi 40 (série 06) et la classe du défi 48 (série 07).
Rien à installer : tout fait partie de Python.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- `@dataclass` écrit pour vous `__init__`, `__repr__` (l'affichage) et `__eq__` (== compare les
  CHAMPS, pas l'identité). Avec `frozen=True`, l'objet ne peut plus être modifié après sa création,
  et il devient hachable (utilisable dans un `set` ou comme clé de dictionnaire). `__post_init__`
  est appelée juste après `__init__` : c'est l'endroit où valider ;
- TypeError quand le TYPE ne va pas (un texte au lieu d'un entier), ValueError quand le type va mais
  pas la VALEUR (une ancienneté de -3). C'est la convention de Python : suivez-la, l'appelant saura
  quoi attraper ;
- les MÉTHODES SPÉCIALES (« dunder », pour double underscore) branchent vos objets sur la syntaxe :
  `a + b` appelle `a.__add__(b)`, `a < b` → `a.__lt__(b)`, `x in a` → `a.__contains__(x)`,
  `a[k]` → `a.__getitem__(k)`, `len(a)` → `a.__len__()`, `for x in a` → `a.__iter__()`,
  `with a:` → `a.__enter__()` puis `a.__exit__(...)`, `if a:` → `a.__bool__()`, `str(a)` → `a.__str__()` ;
- quand une opération n'a pas de sens pour votre objet (`montant + "abc"`), la méthode RENVOIE
  (sans lever) la constante `NotImplemented`. Python essaie alors l'opération dans l'autre sens
  (`"abc".__radd__(montant)`), puis lève une TypeError claire ;
- `@classmethod` : une méthode appelée sur la CLASSE (`Montant.depuis_euros("19,99")`), qui reçoit la
  classe en premier argument (`cls`) ; parfait pour un autre moyen de créer un objet.
  `@property` : une méthode qu'on lit comme un attribut (`contrat.engagement_mois`, sans parenthèses).

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_17 import *
    >>> Client("C001", 3, 49.9)                         # défi 104
    Client(client_id='C001', anciennete_mois=3, facture_mensuelle=49.9, contrat='Mensuel', options=())
    >>> Client("C001", -3, 49.9)
    ValueError: anciennete_mois ne peut pas être négative : -3

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code (pour le défi 104, la classe entière).
2. Lancez `python scripts/defis.py --defi 104` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_17.py`.

La fausse passerelle SMS du défi 107 (`Passerelle`, `Lot`, `PasserelleIndisponible`) et les données
d'exemple (`CONTRATS`, `SCORES_EXEMPLE`, `CLIENTS_EXEMPLE`) sont dans `defis/conception.py`, déjà
importés ci-dessous : vous n'avez pas besoin d'ouvrir ce fichier.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import math  # noqa: F401 — utile au défi 104
import re  # noqa: F401 — utile aux défis 104 et 106
from contextlib import contextmanager
from dataclasses import dataclass, field  # noqa: F401 — field() si vous en avez besoin
from decimal import Decimal, InvalidOperation  # noqa: F401 — utiles au défi 106
from enum import Enum
from functools import total_ordering
from typing import Any, Callable, Iterable, Iterator

from defis.conception import (  # noqa: F401 — tout est utile pour essayer vos classes
    CLIENTS_EXEMPLE,
    CONTRATS,
    SCORES_EXEMPLE,
    Lot,
    Passerelle,
    PasserelleIndisponible,
)

# ============================================================================ dataclass


class Client:
    """Défi 104 🟢 — Un Client qui refuse d'être invalide : `@dataclass(frozen=True)` et `__post_init__`.

    Un client invalide qui entre dans le programme (ancienneté négative, facture NaN, contrat mal
    orthographié) fait des dégâts loin de l'endroit où il est entré, et on met des heures à remonter
    jusqu'à lui. La parade : un type qui vérifie tout À LA CRÉATION. Si l'objet existe, il est valide.

    Remplacez TOUTE cette classe (y compris le `__init__` ci-dessous) par une dataclass gelée :

        @dataclass(frozen=True)
        class Client:
            client_id: str
            ...

    Les champs, dans cet ordre (on peut créer un client avec ou sans les noms) :

    - `client_id` : un texte, « C » suivi de chiffres (« C001 », « C42 ») ;
    - `anciennete_mois` : un entier, 0 ou plus ;
    - `facture_mensuelle` : un nombre (entier ou float), fini et positif ou nul ;
    - `contrat` : « Mensuel », « Annuel » ou « Biannuel » (voir `CONTRATS`), par défaut « Mensuel » ;
    - `options` : les options souscrites, par défaut aucune. On peut les donner sous forme de liste ou
      de tuple, mais le client les GARDE en tuple (une liste pourrait être modifiée par l'appelant,
      et un objet qui contient une liste n'est pas hachable).

    Une valeur du mauvais TYPE lève TypeError, une mauvaise VALEUR lève ValueError ; dans les deux
    cas, le message NOMME le champ fautif (« anciennete_mois ne peut pas être négative : -3 »).

        >>> client = Client("C001", 3, 49.9, options=["tv"])
        >>> client.options
        ('tv',)
        >>> client.anciennete_mois = 4                    # gelé : on ne modifie pas un client
        dataclasses.FrozenInstanceError: cannot assign to field 'anciennete_mois'
        >>> client == Client("C001", 3, 49.9, "Mensuel", ("tv",))     # == compare les champs
        True

    Les pièges, que le correcteur essaie tous :

    - `True` est un entier pour Python (`isinstance(True, int)` vaut True) : une ancienneté `True`
      doit être refusée ;
    - `float("nan") < 0` vaut False, comme `float("nan") >= 0` : `if facture < 0` laisse passer NaN ;
    - `options: list = []` : Python refuse une liste comme valeur par défaut d'une dataclass (elle
      serait partagée par tous les clients) ; et `tuple("tv")` donne `('t', 'v')` : un seul texte
      doit être refusé, pas découpé en lettres.

    Ce que vous apprenez : à faire d'un type la seule porte d'entrée des données, et la différence
    entre TypeError et ValueError.

    Indice : dans `def __post_init__(self) -> None:`, une suite de `if … : raise …`. Pour exclure les
    booléens : `isinstance(x, bool) or not isinstance(x, int)`. Pour NaN et l'infini : `math.isfinite`.
    Une classe gelée refuse `self.options = …`, même dans `__post_init__` : l'écriture consacrée est
    `object.__setattr__(self, "options", tuple(self.options))`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError


# ============================================================================ Enum


class Contrat(Enum):
    """Défi 105 🟢 — Un Contrat qui ne peut valoir que trois choses : `Enum`.

    Dans le CSV, le contrat est un texte. Un texte peut valoir n'importe quoi : « mensuel »,
    « Mensuel  », « Mensuelle », « Bi-annuel »… Chaque `if contrat == "Mensuel"` du programme est alors
    une occasion de se tromper en silence. Une `Enum` (énumération) fixe la liste des valeurs
    possibles : `Contrat.MENSUEL` existe, `Contrat.MENSUELLE` lève une erreur dès qu'on l'écrit.

    Complétez la classe :

    - trois MEMBRES, dans cet ordre : `MENSUEL`, `ANNUEL` et `BIANNUEL`, dont les valeurs sont les
      textes du CSV (« Mensuel », « Annuel », « Biannuel »). Ainsi `Contrat("Annuel")` retrouve le
      membre à partir de la valeur lue dans le fichier ;
    - `Contrat.depuis_texte(texte)` lit un contrat saisi par un humain : sans tenir compte des
      majuscules ni des espaces autour (« mensuel », « ANNUEL\\n »). Elle renvoie le MEMBRE (pas le
      texte). Un `Contrat` reçu est renvoyé tel quel. Un texte inconnu (« Mensuelle », « ») lève
      ValueError, avec un message qui donne les valeurs acceptées. Autre chose qu'un texte (None, 12)
      lève TypeError ;
    - `contrat.engagement_mois`, une PROPRIÉTÉ (sans parenthèses) : 1, 12 ou 24 ;
    - `str(contrat)` donne la valeur (« Annuel »), pour écrire le contrat dans un SMS ou un CSV
      (par défaut, Python écrirait « Contrat.ANNUEL »).

        >>> Contrat.depuis_texte(" mensuel ")
        <Contrat.MENSUEL: 'Mensuel'>
        >>> Contrat.depuis_texte("Bi-annuel")
        ValueError: contrat inconnu : 'Bi-annuel' (attendu : Mensuel, Annuel, Biannuel)
        >>> Contrat.ANNUEL.engagement_mois, f"Contrat {Contrat.ANNUEL}"
        (12, 'Contrat Annuel')
        >>> [c.value for c in Contrat]                    # parcourir une Enum donne ses membres
        ['Mensuel', 'Annuel', 'Biannuel']

    Le piège : écrire dans la classe un dictionnaire des engagements, `_ENGAGEMENTS = {"Mensuel": 1, …}`.
    Dans une Enum, TOUT attribut de classe (sauf les méthodes) devient un membre : votre Enum aurait
    quatre contrats, dont un qui vaut un dictionnaire. Autre piège : `texte.startswith("mens")` ou
    `in` accepterait « Mensuelle ».

    Ce que vous apprenez : remplacer un texte « magique » par un type qui n'admet que des valeurs
    connues, et à écrire une `@classmethod` et une `@property`.

    Indice : `for membre in cls:` parcourt les membres, `membre.value` donne le texte. Pour comparer
    sans tenir compte des majuscules : `texte.strip().casefold() == membre.value.casefold()`. Pour
    l'engagement, un `if self is Contrat.MENSUEL:`… ou un `match self:` suffit.
    """

    # Vos trois membres ici, par exemple : MENSUEL = "Mensuel"

    @classmethod
    def depuis_texte(cls, texte: str | Contrat) -> Contrat:
        raise NotImplementedError

    @property
    def engagement_mois(self) -> int:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


# ============================================================================ Objet valeur


@total_ordering
@dataclass(frozen=True)
class Montant:
    """Défi 106 🟡 — Un Montant qui ne perd jamais un centime : un « objet valeur ».

    `0.1 + 0.2 == 0.3` vaut False en Python, et `int(19.99 * 100)` vaut 1998. Sur une facture, ce
    centime perdu devient une réclamation. La règle des développeurs : un montant se range en
    CENTIMES ENTIERS, avec sa devise, et ne s'additionne qu'avec un montant de la même devise.

    La dataclass est déjà là (gelée : deux montants égaux sont interchangeables, comme deux billets
    de 10 €). Complétez :

    - `__post_init__` : `centimes` est un entier (pas un booléen, pas un float : TypeError) ; la
      devise s'écrit avec 3 lettres majuscules (« EUR », « USD » ; sinon ValueError). Un montant
      négatif est permis (un remboursement) ;
    - `Montant.depuis_euros(euros, devise="EUR")` crée un montant à partir d'euros : un texte
      (« 19,99 », « 19.99 », « 1 234,50 », avec ou sans espaces des milliers, y compris les espaces
      insécables d'Excel « \\u00a0 » et « \\u202f »), un entier ou un `Decimal`. Un float lève
      TypeError (il n'est pas exact). Plus de 2 décimales (« 19,999 »), un texte illisible (« abc »,
      « », « NaN ») lèvent ValueError ;
    - `+` entre deux montants de même devise donne un nouveau Montant ; devises différentes :
      ValueError ; avec autre chose qu'un Montant : TypeError. `sum(montants)` doit marcher ;
    - `*` par un ENTIER (une quantité), dans les deux sens : `montant * 3` et `3 * montant`. Par un
      float ou un autre montant : TypeError ;
    - `<`, `<=`, `>`, `>=` entre montants de même devise (pour `sorted` et `max`) ; devises
      différentes : ValueError. (`==` entre devises différentes vaut simplement False.) ;
    - `str(montant)` au format français : « 1 234,50 € », « -0,05 € », « 12,00 USD » (le symbole
      € pour l'euro, le code pour les autres devises, une espace pour les milliers).

        >>> Montant.depuis_euros("1 234,5")
        Montant(centimes=123450, devise='EUR')
        >>> print(sum([Montant(1999), Montant(1)]), Montant(-5), 3 * Montant(1999, "USD"))
        20,00 € -0,05 € 59,97 USD
        >>> Montant(100, "USD") < Montant(200)
        ValueError: impossible de comparer USD et EUR sans taux de change

    Les pièges, que le correcteur essaie tous :

    - `@dataclass(order=True)` écrit `<` pour vous… en comparant `(centimes, devise)` comme des
      tuples : 1 $ < 2 € renverrait True. C'est pour ça qu'on écrit `__lt__` à la main (et
      `@total_ordering`, déjà en place, en déduit `<=`, `>` et `>=`) ;
    - `sum(montants)` calcule `0 + montant1 + …` : il faut accepter le 0 de départ (`__radd__`) ;
    - `decimal.InvalidOperation` (levée par `Decimal("abc")`) n'est PAS une ValueError ;
    - `-5 // 100` vaut -1 et `-5 % 100` vaut 95 : « -1,95 € » au lieu de « -0,05 € ».

    Ce que vous apprenez : NotImplemented et les opérateurs « dans l'autre sens » (`__radd__`,
    `__rmul__`), et pourquoi on ne range jamais de l'argent dans un float.

    Indice : `Decimal(texte.replace(",", "."))` lit le nombre exactement ; `valeur * 100` doit alors
    être entier (`centimes == centimes.to_integral_value()`), et `valeur.is_finite()` écarte NaN.
    Pour l'affichage : `euros, cts = divmod(abs(self.centimes), 100)`, puis `f"{euros:,}"` met des
    virgules, que `.replace(",", " ")` transforme en espaces.
    """

    centimes: int
    devise: str = "EUR"

    def __post_init__(self) -> None:
        raise NotImplementedError

    @classmethod
    def depuis_euros(cls, euros: str | int | Decimal, devise: str = "EUR") -> Montant:
        raise NotImplementedError

    def __add__(self, autre: object) -> Montant:
        raise NotImplementedError

    def __radd__(self, autre: object) -> Montant:
        raise NotImplementedError

    def __mul__(self, n: object) -> Montant:
        raise NotImplementedError

    def __rmul__(self, n: object) -> Montant:
        raise NotImplementedError

    def __lt__(self, autre: object) -> bool:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


# ============================================================================ Gestionnaire de contexte


@contextmanager
def campagne(passerelle: Passerelle, nom: str) -> Iterator[Lot]:
    """Défi 107 🟡 — Un envoi de SMS qui ne part qu'en entier : un gestionnaire de contexte (`with`).

    Pour un envoi groupé, la passerelle SMS fonctionne en 4 temps : on OUVRE un lot, on y AJOUTE des
    clients, puis on le VALIDE (les SMS partent) ou on l'ANNULE (rien ne part), et dans tous les cas
    on le FERME (sinon la réservation chez l'opérateur reste bloquée). Si le programme plante au
    milieu de la préparation, il ne faut surtout pas valider : 200 clients sur 500 recevraient l'offre,
    et les 300 autres jamais.

    Ce découpage « préparer → tout valider ou tout annuler → toujours fermer » revient partout
    (fichiers, transactions, verrous). On l'écrit UNE fois, dans un gestionnaire de contexte :

        with campagne(passerelle, "Fidélité") as lot:
            for client_id in cibles:
                lot.ajouter(client_id)
        # ici : validé et fermé si tout s'est bien passé ; annulé et fermé sinon

    Écrivez `campagne(passerelle, nom)` :

    - en entrant dans le `with` : `lot = passerelle.ouvrir(nom)`, et le `with … as lot` reçoit ce lot.
      Rien n'est ouvert tant qu'on n'est pas entré dans le `with` ;
    - si le corps du `with` va au bout (ou en sort par un `return` ou un `break`) :
      `passerelle.valider(lot)`, puis `passerelle.fermer(lot)` ;
    - si le corps lève une erreur, QUELLE QU'ELLE SOIT (même un Ctrl+C, `KeyboardInterrupt`) :
      `passerelle.annuler(lot)`, puis `passerelle.fermer(lot)`, puis la MÊME erreur continue son chemin
      (ne l'avalez pas, ne la remplacez pas : l'appelant doit savoir que la campagne n'est pas partie) ;
    - si `ouvrir` échoue (`PasserelleIndisponible`) : l'erreur remonte, et il n'y a rien à annuler ni
      à fermer ;
    - si `valider` échoue : l'erreur remonte, et le lot est quand même fermé (pas annulé : l'opérateur
      l'a déjà refusé).

        >>> passerelle = Passerelle()
        >>> with campagne(passerelle, "Fidélité") as lot:
        ...     lot.ajouter("C001")
        ...     lot.ajouter("C002")
        >>> passerelle.journal
        ['ouvrir Fidélité', 'ajouter C001', 'ajouter C002', 'valider', 'fermer']
        >>> passerelle.envoyes
        ['C001', 'C002']

    Le piège : écrire `lot = …ouvrir(nom)`, `yield lot`, `…valider(lot)`, `…fermer(lot)` à la suite.
    Ça marche… tant que rien ne plante : une erreur dans le corps du `with` est relevée AU NIVEAU DU
    `yield`, et les lignes suivantes ne s'exécutent jamais (le lot reste ouvert). Autre piège :
    `except Exception:` ne voit pas KeyboardInterrupt, qui hérite directement de BaseException.

    Ce que vous apprenez : `try / except / else / finally` au complet, et à écrire vos propres `with`.

    Indice : avec `@contextmanager` (déjà en place), tout ce qui est avant le `yield` s'exécute en
    entrant, tout ce qui est après en sortant. Ouvrez AVANT le `try`, puis :
    `try: yield lot` / `except BaseException: annuler ; raise` / `else: valider` / `finally: fermer`.
    Attraper BaseException est permis ici, PARCE QU'on relance aussitôt : on nettoie, on n'avale rien.
    """
    raise NotImplementedError
    yield  # (jamais atteint : cette ligne fait de la fonction un générateur, ce qu'attend @contextmanager)


# ============================================================================ Collection


class Portefeuille:
    """Défi 108 🟡 — Un portefeuille de clients qu'on utilise comme une collection Python.

    L'équipe rétention manipule des milliers de clients avec leur score de churn. Plutôt qu'un
    dictionnaire nu (que n'importe qui peut modifier n'importe comment), on écrit une classe… qui
    s'utilise pourtant aussi naturellement qu'une liste : `len`, `in`, `[]` et `for`.

    Écrivez la classe :

    - `Portefeuille(paires=())` : un portefeuille, éventuellement rempli à partir de couples
      `(client_id, score)` (par exemple `SCORES_EXEMPLE.items()`) ;
    - `p.ajouter(client_id, score)` ; un client déjà présent lève ValueError (et son score ne change
      pas) : un doublon cache souvent une erreur dans le fichier source ;
    - `len(p)` : le nombre de clients (et donc `if p:` est faux pour un portefeuille vide) ;
    - `"C001" in p` : le client est-il dans le portefeuille ? ;
    - `p["C001"]` : son score ; un client inconnu lève KeyError, comme un dictionnaire ;
    - `for client_id, score in p:` parcourt les clients du PLUS risqué au moins risqué (à score égal,
      dans l'ordre des identifiants). On doit pouvoir parcourir le portefeuille autant de fois qu'on
      veut, et même imbriquer deux boucles. Ajouter un client pendant un parcours ne doit pas faire
      planter ce parcours ;
    - `p.au_dessus(seuil)` : un NOUVEAU portefeuille avec les clients dont le score est ≥ seuil.
      Les deux portefeuilles sont ensuite indépendants ;
    - `repr(p)` : « Portefeuille(3 clients) », « Portefeuille(1 client) », « Portefeuille(vide) ».

    Le correcteur remplit aussi un portefeuille de 20 000 clients et y fait 5 000 recherches : il
    faut que ce soit instantané.

        >>> p = Portefeuille(SCORES_EXEMPLE.items())
        >>> p, len(p), "C002" in p, p["C002"]
        (Portefeuille(5 clients), 5, True, 0.15)
        >>> list(p)[:3]                                   # 0,47 et 0,47 : ordre des identifiants
        [('C004', 0.91), ('C001', 0.82), ('C003', 0.47)]
        >>> p.au_dessus(0.8)
        Portefeuille(2 clients)

    Les pièges : sans `__contains__`, Python fait `in` en parcourant `__iter__`… et compare « C001 »
    à chaque couple `("C001", 0.82)` : toujours False. Un `__iter__` qui renvoie `self` (avec un
    `__next__`) ne se parcourt qu'UNE fois : la deuxième boucle est vide, sans erreur. Et parcourir
    un dictionnaire pendant qu'on y ajoute lève « dictionary changed size during iteration ».

    Ce que vous apprenez : le protocole des collections (`__len__`, `__contains__`, `__getitem__`,
    `__iter__`), la différence entre un itérable (qu'on parcourt autant qu'on veut) et un itérateur
    (qui s'épuise), et le choix de la structure de données.

    Indice : un dictionnaire `self._scores` à l'intérieur. `__iter__` peut renvoyer
    `iter(sorted(self._scores.items(), key=lambda paire: (-paire[1], paire[0])))` : une NOUVELLE
    liste triée à chaque parcours (c'est aussi une photo, qu'un ajout ne dérange pas).
    """

    def __init__(self, paires: Iterable[tuple[str, float]] = ()) -> None:
        raise NotImplementedError

    def ajouter(self, client_id: str, score: float) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __contains__(self, client_id: object) -> bool:
        raise NotImplementedError

    def __getitem__(self, client_id: str) -> float:
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[str, float]]:
        raise NotImplementedError

    def au_dessus(self, seuil: float) -> Portefeuille:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


# ============================================================================ Opérateurs


class Regle:
    """Défi 109 🔴 — Des règles de ciblage qu'on combine avec & | ~ (et pas avec and, or, not).

    Le marketing veut cibler « les clients mensuels, récents OU mécontents, sauf les gros comptes ».
    Demain, ce sera autre chose. Plutôt que de réécrire un `if` à chaque campagne, on écrit de petites
    règles nommées, et on les COMBINE :

        mensuel = Regle("mensuel", lambda c: c["contrat"] == "Mensuel")
        recent = Regle("récent", lambda c: c["anciennete_mois"] < 6)
        mecontent = Regle("mécontent", lambda c: c["appels_support"] >= 3)
        cible = mensuel & (recent | mecontent)

    C'est exactement ce que fait pandas avec `df[(df.contrat == "Mensuel") & (df.anciennete < 6)]`.

    Écrivez la classe :

    - `Regle(nom, condition)` : `condition` est une fonction client → vrai/faux. Un nom vide (ou fait
      d'espaces) lève ValueError ; une condition qui n'est pas une fonction lève TypeError ;
    - `regle(client)` (méthode `__call__`) évalue la règle et renvoie un VRAI booléen (True ou False),
      même si la condition renvoie 3 ou "" ;
    - `a & b` (les deux), `a | b` (l'une ou l'autre) et `~a` (pas a) renvoient une NOUVELLE règle,
      sans modifier `a` ni `b`. Comme `and` et `or`, elles s'arrêtent dès que le résultat est connu :
      `(a & b)(client)` n'évalue pas `b` si `a` est fausse (b peut être lente, ou planter sur un client
      que a écarte), et `(a | b)(client)` n'évalue pas `b` si `a` est vraie. Combiner avec autre chose
      qu'une règle (`a & True`) lève TypeError ;
    - `str(regle)` : le nom pour une règle simple ; `"(a & b)"`, `"(a | b)"` et `"~a"` pour les
      combinaisons (une paire de parenthèses par & ou |, même à l'extérieur) ;
    - `bool(regle)`, et donc `if regle:`, `a and b`, `a or b`, `not a`, lève TypeError, avec un message
      qui dit d'utiliser & | ~.

        >>> cible = mensuel & (recent | mecontent)
        >>> str(cible), str(~cible)
        ('(mensuel & (récent | mécontent))', '~(mensuel & (récent | mécontent))')
        >>> [c["client_id"] for c in CLIENTS_EXEMPLE if cible(c)]
        ['C001', 'C002']
        >>> mensuel and recent
        TypeError: la règle 'mensuel' n'est ni vraie ni fausse : combinez avec & | ~ (pas and, or, not)…

    Pourquoi interdire `and` ? Python ne permet PAS de redéfinir `and`, `or` et `not` : ils demandent
    `bool(a)`, et par défaut tout objet est « vrai ». `mensuel and recent` renverrait donc… `recent`,
    sans un mot, et la campagne partirait vers tous les clients récents, mensuels ou non. pandas a fait
    le même choix (« The truth value of a Series is ambiguous »).

    Les pièges : `a & b` qui ajoute b à une liste DANS a et renvoie a (a est modifiée : chaque règle qui
    s'en sert change de sens) ; `lambda c: a(c) & b(c)` (le & entre deux booléens évalue les deux côtés,
    sans court-circuit) ; `__call__` qui renvoie la valeur brute de la condition.

    Ce que vous apprenez : la surcharge d'opérateurs (`__and__`, `__or__`, `__invert__`, `__call__`),
    les fermetures (une lambda qui se souvient de a et b), et `__bool__`.

    Indice : `def __and__(self, autre):` vérifie `isinstance(autre, Regle)` (sinon
    `return NotImplemented`), puis `return Regle(f"({self} & {autre})", lambda client: self(client) and
    autre(client))`. Ici `self(client)` renvoie déjà un booléen : le `and` de Python court-circuite.
    """

    def __init__(self, nom: str, condition: Callable[[Any], object]) -> None:
        raise NotImplementedError

    def __call__(self, client: Any) -> bool:
        raise NotImplementedError

    def __and__(self, autre: object) -> Regle:
        raise NotImplementedError

    def __or__(self, autre: object) -> Regle:
        raise NotImplementedError

    def __invert__(self) -> Regle:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def __bool__(self) -> bool:
        raise NotImplementedError

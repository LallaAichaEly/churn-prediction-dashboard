"""Outils communs de la série 31 : les options des clients, des similarités toutes faites, des générateurs.

Un client qui a plusieurs options part moins : dans `_churn_logit` (src/churn/data.py), chaque produit en
plus retire 0,25 au logit du churn. L'équipe marketing veut donc proposer à chaque client « l'offre
suivante », celle qu'il a le plus de chances de prendre. Les données :

- `paniers_exemple()` : {client: liste des options détenues}. Un export réel, avec ses défauts : une option
  écrite deux fois (export rejoué), un client sans option ([]), une case vide (None), une option que
  personne d'autre n'a (« presse ») ;
- `abonnements_exemple()` : le journal des souscriptions [(client, option, date)], dans le désordre ;
- `populaires(paniers)` : les options de la plus détenue à la moins détenue (le plan B quand on ne sait
  rien d'un client) ;
- `similarites_cosinus(paniers)` : {option: {autre option: similarité entre 0 et 1}}, prête à l'emploi
  pour les défis 190 et 193 ;
- `SimilaritesComptees(similarites)` : les mêmes, en lecture seule, qui COMPTENT les similarités lues
  (`.lues`). C'est ainsi que le correcteur du défi 193 voit qu'un calcul relit tout à chaque tour ;
- `generer_paniers(n, graine)` et `generer_abonnements(n, graine)` : de gros volumes reproductibles, pour
  vos propres essais.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> paniers = paniers_exemple()
    >>> paniers["C08"]                                 # « tv » deux fois : l'export a été rejoué
    ['mobile', 'tv', 'tv']
    >>> paniers["C09"], paniers["C11"]                 # un client sans option, une case vide
    ([], None)
    >>> populaires(paniers)[:4]                        # « mobile » est partout
    ['mobile', 'tv', 'gaming', 'musique']
    >>> sim = similarites_cosinus(paniers)
    >>> round(sim["cloud"]["antivirus"], 3), round(sim["tv"]["sport"], 3)
    (1.0, 0.866)
    >>> "presse" in sim, sim["presse"]                 # détenue, mais jamais avec une autre option
    (True, {})
"""

from __future__ import annotations

import datetime as dt
import math
import random
from collections import Counter
from collections.abc import Mapping
from typing import NamedTuple

CATALOGUE = ("antivirus", "assurance", "box2", "cloud", "gaming", "mobile", "musique", "presse", "sport", "tv")


class Regle(NamedTuple):
    """« Les clients qui ont `si` ont aussi `alors` » (défi 189). Un tuple : (si, alors, support, confiance, lift)."""

    si: str
    alors: str
    support: int        # nombre de clients qui ont les DEUX options
    confiance: float    # parmi les clients qui ont `si`, la part qui a aussi `alors`
    lift: float         # combien de fois plus souvent que le hasard


_PANIERS: dict[str, list[str] | None] = {
    "C01": ["mobile", "tv", "sport"],
    "C02": ["musique", "mobile"],
    "C03": ["mobile", "musique", "gaming"],
    "C04": ["tv", "sport", "box2"],
    "C05": ["mobile"],
    "C06": ["mobile", "cloud", "antivirus"],
    "C07": ["cloud", "antivirus", "assurance"],
    "C08": ["mobile", "tv", "tv"],
    "C09": [],
    "C10": ["gaming", "mobile", "musique"],
    "C11": None,
    "C12": ["presse"],
    "C13": ["mobile", "tv", "sport", "box2"],
    "C14": ["mobile", "gaming"],
}

_ABONNEMENTS: list[tuple[str, str, dt.date]] = [
    ("C01", "mobile", dt.date(2026, 1, 10)),
    ("C02", "musique", dt.date(2026, 7, 2)),
    ("C01", "tv", dt.date(2026, 6, 1)),          # le jour même de la coupure
    ("C02", "mobile", dt.date(2025, 11, 3)),
    ("C03", "gaming", dt.date(2026, 8, 15)),     # C03 n'avait rien avant : un client « froid »
    ("C01", "mobile", dt.date(2026, 9, 1)),      # réabonnement : C01 avait déjà « mobile »
    ("C04", "tv", dt.date(2026, 2, 1)),
    ("C02", "mobile", dt.date(2025, 11, 3)),     # la même ligne deux fois (export rejoué)
]


def paniers_exemple() -> dict[str, list[str] | None]:
    """Quatorze clients et leurs options, avec les défauts d'un vrai export (voir le haut du fichier)."""
    return {client: (None if panier is None else list(panier)) for client, panier in _PANIERS.items()}


def abonnements_exemple() -> list[tuple[str, str, dt.date]]:
    """Huit souscriptions (client, option, date), dans le désordre, avec un doublon et un réabonnement."""
    return list(_ABONNEMENTS)


def _detenteurs(paniers: dict) -> dict[str, set]:
    """{option: ensemble des clients qui la détiennent}."""
    detenteurs: dict[str, set] = {}
    for client, panier in paniers.items():
        for offre in set(panier or ()):
            detenteurs.setdefault(offre, set()).add(client)
    return detenteurs


def populaires(paniers: dict) -> list[str]:
    """Les options détenues, de la plus répandue à la moins répandue (à égalité : ordre alphabétique).

    Un client compte une fois par option, même si l'export l'a écrite deux fois.

        >>> populaires({"A": ["tv", "mobile"], "B": ["mobile", "mobile"], "C": None, "D": ["box2"]})
        ['mobile', 'box2', 'tv']
    """
    nombre = Counter()
    for panier in paniers.values():
        nombre.update(set(panier or ()))
    return sorted(nombre, key=lambda offre: (-nombre[offre], offre))


def similarites_cosinus(paniers: dict) -> dict[str, dict[str, float]]:
    """{option: {autre: similarité cosinus}} : (clients qui ont les deux) / √(clients de l'une × de l'autre).

    1.0 : toujours ensemble ; 0 : jamais (la paire n'est alors pas stockée). Symétrique. Chaque option
    détenue est une clé, même si elle n'a aucune voisine.

    Cette version compare TOUTES les paires d'options : simple, mais lente sur un catalogue de milliers
    d'options. Le défi 188 compte les paires sans les essayer toutes.

        >>> sim = similarites_cosinus({"A": ["tv", "sport"], "B": ["tv"], "C": ["tv", "sport", "tv"]})
        >>> round(sim["tv"]["sport"], 4)                 # 2 / √(3 × 2)
        0.8165
    """
    detenteurs = _detenteurs(paniers)
    sim: dict[str, dict[str, float]] = {offre: {} for offre in sorted(detenteurs)}
    offres = sorted(detenteurs)
    for i, a in enumerate(offres):
        for b in offres[i + 1:]:
            communs = len(detenteurs[a] & detenteurs[b])
            if communs:
                valeur = communs / math.sqrt(len(detenteurs[a]) * len(detenteurs[b]))
                sim[a][b] = sim[b][a] = valeur
    return sim


# ---------------------------------------------------------------------------- des similarités qui comptent


class LecturesEpuisees(BaseException):
    """Levée quand une `SimilaritesComptees` dépasse son budget de lectures.

    BaseException, et pas Exception : un `except Exception` dans votre code ne la cache pas.
    """


class _Ligne(Mapping):
    """{autre option: similarité} en lecture seule ; chaque similarité lue compte une lecture."""

    def __init__(self, valeurs: dict, compteur: SimilaritesComptees) -> None:
        self._valeurs = dict(valeurs)
        self._compteur = compteur

    def __getitem__(self, cle):
        self._compteur._lire()
        return self._valeurs[cle]

    def __contains__(self, cle) -> bool:
        self._compteur._lire()
        return cle in self._valeurs

    def get(self, cle, defaut=None):
        self._compteur._lire()
        return self._valeurs.get(cle, defaut)

    def __iter__(self):
        for cle in self._valeurs:
            self._compteur._lire()
            yield cle

    def items(self):  # type: ignore[override]
        for cle, valeur in self._valeurs.items():
            self._compteur._lire()
            yield cle, valeur

    def __len__(self) -> int:
        return len(self._valeurs)

    def __repr__(self) -> str:
        return f"<ligne de {len(self._valeurs)} similarités>"


class SimilaritesComptees(Mapping):
    """{option: {autre: similarité}} en LECTURE SEULE, qui compte les similarités lues dans `.lues`.

    `similarites[o]` (ou `.get(o)`) ne coûte rien ; lire UNE similarité dans la ligne (`ligne[a]`,
    `ligne.get(a, 0)`, `a in ligne`) coûte 1 ; parcourir une ligne coûte une lecture par case. Au-delà de
    `budget` lectures, `LecturesEpuisees` est levée : le calcul relit trop.

        >>> sim = SimilaritesComptees({"tv": {"box2": 0.75, "sport": 0.5}, "box2": {"tv": 0.75}})
        >>> sim["tv"].get("box2", 0.0), sim["tv"].get("cloud", 0.0), sim.lues
        (0.75, 0.0, 2)
        >>> sim.get("presse") is None, sim.lues            # une option sans ligne : rien de lu
        (True, 2)
    """

    def __init__(self, similarites: dict, budget: int | None = None) -> None:
        self._lignes = {offre: _Ligne(ligne, self) for offre, ligne in similarites.items()}
        self.lues = 0
        self.budget = budget

    def _lire(self) -> None:
        self.lues += 1
        if self.budget is not None and self.lues > self.budget:
            raise LecturesEpuisees(f"plus de {self.budget} similarités lues")

    def __getitem__(self, offre):
        return self._lignes[offre]

    def __iter__(self):
        return iter(self._lignes)

    def __len__(self) -> int:
        return len(self._lignes)

    def __repr__(self) -> str:
        return f"SimilaritesComptees({len(self._lignes)} options, {self.lues} lues)"


# ---------------------------------------------------------------------------- de gros volumes

# Trois familles de clients, chacune avec ses options et la probabilité de les avoir. « mobile » est à part :
# 70 % des clients l'ont, quel que soit leur profil. C'est elle que la confiance seule recommanderait à tous.
_PROFILS = {
    "famille": (("tv", 0.8), ("sport", 0.5), ("box2", 0.4), ("presse", 0.2), ("assurance", 0.2)),
    "jeune": (("musique", 0.7), ("gaming", 0.5), ("cloud", 0.2)),
    "pro": (("cloud", 0.7), ("antivirus", 0.6), ("assurance", 0.4), ("presse", 0.2)),
    "simple": (),
}
_POIDS_PROFILS = (0.35, 0.30, 0.15, 0.20)


def _tirer_panier(rng: random.Random, extras: list[str]) -> list[str]:
    profil = rng.choices(list(_PROFILS), weights=_POIDS_PROFILS)[0]
    panier = [offre for offre, p in _PROFILS[profil] if rng.random() < p]
    if rng.random() < 0.7:
        panier.append("mobile")
    for offre in CATALOGUE:                                   # un peu de bruit : des goûts imprévus
        if rng.random() < 0.03 and offre not in panier:
            panier.append(offre)
    if extras:                                                # la longue traîne d'un grand catalogue
        for _ in range(rng.choice((0, 1, 1, 2, 3))):
            offre = extras[int(len(extras) * rng.random() ** 2)]
            if offre not in panier:
                panier.append(offre)
    rng.shuffle(panier)
    return panier


def generer_paniers(n: int = 2000, graine: int = 0, nb_offres: int | None = None) -> dict[str, list[str] | None]:
    """`n` clients {identifiant: options} tirés au hasard, reproductibles.

    Trois profils (famille : tv, sport, box2 ; jeune : musique, gaming ; pro : cloud, antivirus), « mobile »
    chez 70 % des clients, un peu de bruit. Environ 2 % de cases vides (None) et 3 % d'options écrites deux
    fois. Avec `nb_offres` plus grand que le catalogue, des options « offre0001 »… s'ajoutent : un catalogue
    de milliers d'options, dont la plupart sont rares.

        >>> paniers = generer_paniers(500, graine=1)
        >>> len(paniers), paniers == generer_paniers(500, graine=1)
        (500, True)
        >>> populaires(paniers)[0]
        'mobile'
        >>> len(populaires(generer_paniers(2000, graine=1, nb_offres=1000))) > 500
        True
    """
    rng = random.Random(graine)
    extras = [f"offre{j:04d}" for j in range(1, (nb_offres or 0) - len(CATALOGUE) + 1)]
    paniers: dict[str, list[str] | None] = {}
    for i in rng.sample(range(1, 10 * n + 10), n):
        client = f"C{i:06d}"
        if rng.random() < 0.02:
            paniers[client] = None
            continue
        panier = _tirer_panier(rng, extras)
        if panier and rng.random() < 0.03:
            panier.insert(rng.randrange(len(panier) + 1), rng.choice(panier))
        paniers[client] = panier
    return paniers


def generer_abonnements(n: int = 1000, graine: int = 0) -> list[tuple[str, str, dt.date]]:
    """Les souscriptions (client, option, date) de `n` clients entre 2024 et 2026, dans le désordre.

    Mêmes profils que `generer_paniers`. Environ 3 % de lignes en double et 3 % de réabonnements (la même
    option, plus tard) : ce que contient un vrai journal.

        >>> abonnements = generer_abonnements(300, graine=2)
        >>> abonnements == generer_abonnements(300, graine=2)
        True
        >>> min(d for _, _, d in abonnements) >= dt.date(2024, 1, 1)
        True
    """
    rng = random.Random(graine)
    debut, jours = dt.date(2024, 1, 1), 3 * 365
    lignes: list[tuple[str, str, dt.date]] = []
    for i in rng.sample(range(1, 10 * n + 10), n):
        client = f"C{i:06d}"
        for offre in _tirer_panier(rng, []):
            date = debut + dt.timedelta(days=rng.randrange(jours))
            lignes.append((client, offre, date))
            if rng.random() < 0.03:
                lignes.append((client, offre, date))
            if rng.random() < 0.03:
                lignes.append((client, offre, date + dt.timedelta(days=rng.randrange(1, 400))))
    rng.shuffle(lignes)
    return lignes

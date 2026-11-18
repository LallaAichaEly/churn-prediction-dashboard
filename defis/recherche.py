"""Outils communs de la série 30 : les notes des conseillers, le découpage en mots, une liste qui compte.

Après chaque appel, le conseiller ou la conseillère écrit une note libre : « Client mécontent : la FIBRE
coupe tous les soirs. Menace de résilier. » L'équipe de rétention veut y CHERCHER : toutes les notes qui
parlent de résilier ET de fibre, les mots qui commencent par « rési » pendant qu'on tape, les meilleures
notes d'abord, le passage qui répond, surligné.

- `notes_exemple()` : dix notes {identifiant: texte}, dans le désordre (une vide, une sans aucun mot, une
  écrite avec un accent en deux caractères) ;
- `normaliser(texte)` : la forme d'un mot pour la recherche (minuscules, sans accents) ;
- `mots_et_positions(texte)` : les mots du texte, normalisés, avec leur position (début, fin) dans le
  texte D'ORIGINE ;
- `mots(texte)` : seulement les mots normalisés. L'index ET la requête passent par lui : deux découpages
  différents, et « Résilier, » ne retrouve plus « résilier » ;
- `index_frequences(notes)` : (fréquences, longueurs), ce dont le classement du défi 186 a besoin ;
- `Comptee(valeurs)` : une liste en lecture seule qui COMPTE les cases lues (`.lues`). C'est ainsi que le
  correcteur voit qu'une recherche parcourt toute une liste de 100 000 identifiants ;
- `generer_notes(n, graine)` et `generer_vocabulaire(n, graine)` : de gros volumes reproductibles, pour
  vos propres essais.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> notes = notes_exemple()
    >>> notes[17]
    'Client mécontent : la FIBRE coupe tous les soirs. Menace de résilier.'
    >>> mots(notes[17])
    ['client', 'mecontent', 'la', 'fibre', 'coupe', 'tous', 'les', 'soirs', 'menace', 'de', 'resilier']
    >>> mots("Résilier ? L'offre fidélité, -10 %.")
    ['resilier', 'l', 'offre', 'fidelite', '10']
    >>> mots_et_positions("La FIBRE !")
    [('la', 0, 2), ('fibre', 3, 8)]
    >>> "La FIBRE !"[3:8]
    'FIBRE'
    >>> liste = Comptee([3, 8, 17, 30])
    >>> liste[2], len(liste), liste.lues            # len() est gratuit ; chaque case lue compte
    (17, 4, 1)
    >>> 30 in liste, liste.lues                      # « in » lit les cases une par une
    (True, 5)
"""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

# Un mot : une lettre ou un chiffre, puis d'autres lettres, chiffres… ou accents écrits À PART (« e » suivi
# de U+0301, l'accent aigu combinant : deux caractères à l'écran pour un seul « é »). Le « _ » n'est pas une
# lettre : « [^\W_] », c'est \w sans le soulignement.
_MOT = re.compile(r"[^\W_](?:[^\W_]|[̀-ͯ])*")

_NOTES = {
    17: "Client mécontent : la FIBRE coupe tous les soirs. Menace de résilier.",
    3: "Appel pour une hausse de prix. Souhaite résilier si pas de geste commercial.",
    42: "",
    8: "Demande de résiliation reçue par courrier ; propose l'offre fidélité.",
    25: "Fibre installée, client satisfait. Aucune suite.",
    11: "Résilier ? Non : le client voulait juste changer de forfait mobile.",
    30: "Troisième appel pour la fibre : coupure, technicien, coupure. Client à bout, veut résilier la fibre "
        "et le mobile.",
    5: "Prélèvement refusé, client injoignable.",
    19: "Résilier n'est plus d'actualité : geste commercial accepté (prix -10 %).",
    36: "?!",
}


def notes_exemple() -> dict[int, str]:
    """Dix notes {identifiant: texte}, rangées dans le DÉSORDRE (comme un export qui ne trie rien).

    La note 42 est vide, la 36 ne contient aucun mot (« ?! »), la 19 écrit « Résilier » avec un « e » suivi
    d'un accent combinant (deux caractères, un seul « é » à l'écran), la 30 contient deux fois « fibre ».
    """
    return dict(_NOTES)


def normaliser(texte: str) -> str:
    """La forme d'un texte pour la recherche : sans accents, en minuscules (`casefold` : « ß » → « ss »).

        >>> normaliser("RÉSILIER"), normaliser("Re\\u0301silier"), normaliser("Straße"), normaliser("ﬁbre")
        ('resilier', 'resilier', 'strasse', 'fibre')
    """
    if not isinstance(texte, str):
        raise TypeError(f"un texte (str) est attendu, pas {type(texte).__name__}")
    decompose = unicodedata.normalize("NFKD", texte)            # « é » → « e » + accent ; « ﬁ » → « fi »
    sans_accents = "".join(c for c in decompose if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", sans_accents.casefold())


def mots_et_positions(texte: str) -> list[tuple[str, int, int]]:
    """[(mot normalisé, début, fin)] : `texte[début:fin]` est le mot tel qu'il est ÉCRIT dans le texte.

        >>> mots_et_positions("Résilier ? Non.")
        [('resilier', 0, 8), ('non', 11, 14)]
        >>> mots_et_positions("")
        []

    Un texte qui n'est pas un str (None, NaN : une case vide d'un CSV) → TypeError.
    """
    if not isinstance(texte, str):
        raise TypeError(f"un texte (str) est attendu, pas {type(texte).__name__} ({texte!r})")
    return [(normaliser(m.group()), m.start(), m.end()) for m in _MOT.finditer(texte)]


def mots(texte: str) -> list[str]:
    """Les mots normalisés du texte, dans l'ordre, doublons compris.

        >>> mots("Fibre installée, client satisfait.")
        ['fibre', 'installee', 'client', 'satisfait']
    """
    return [mot for mot, _, _ in mots_et_positions(texte)]


def index_frequences(notes: dict) -> tuple[dict[str, dict[int, int]], dict[int, int]]:
    """(fréquences, longueurs) : ce qu'un moteur de recherche calcule UNE fois, avant toute requête.

    - fréquences[mot] = {identifiant: nombre de fois où le mot apparaît dans cette note} ;
    - longueurs[identifiant] = nombre de mots de la note, pour TOUTES les notes (0 pour une note vide, ou
      qui n'est pas un texte).

        >>> frequences, longueurs = index_frequences(notes_exemple())
        >>> frequences["fibre"]
        {17: 1, 25: 1, 30: 2}
        >>> longueurs[30], longueurs[42], len(longueurs)
        (18, 0, 10)
    """
    frequences: dict[str, dict[int, int]] = {}
    longueurs: dict[int, int] = {}
    for identifiant, texte in notes.items():
        liste = mots(texte) if isinstance(texte, str) else []
        longueurs[identifiant] = len(liste)
        for mot, nombre in Counter(liste).items():
            frequences.setdefault(mot, {})[identifiant] = nombre
    return frequences, longueurs


class Comptee(Sequence):
    """Une liste en LECTURE SEULE qui compte les cases lues dans `.lues`.

    `len()` ne coûte rien ; `liste[i]` compte 1 ; `liste[a:b]` compte le nombre de cases du morceau ;
    parcourir avec `for`, `in`, `list()`, `set()` ou `sorted()` compte chaque case. `bisect.bisect_left`
    marche dessus (et ne lit qu'une vingtaine de cases sur un million).

        >>> import bisect
        >>> grande = Comptee(range(0, 2_000_000, 2))
        >>> bisect.bisect_left(grande, 777_778), grande.lues
        (388889, 20)
        >>> grande.lues = 0
        >>> 777_778 in set(grande), grande.lues           # un set se construit en lisant TOUT
        (True, 1000000)
    """

    def __init__(self, valeurs=()) -> None:
        self._valeurs = list(valeurs)
        self.lues = 0

    def __len__(self) -> int:
        return len(self._valeurs)

    def __getitem__(self, i):
        if isinstance(i, slice):
            morceau = self._valeurs[i]
            self.lues += len(morceau)
            return morceau
        valeur = self._valeurs[i]
        self.lues += 1
        return valeur

    def __iter__(self):
        for valeur in self._valeurs:
            self.lues += 1
            yield valeur

    def __eq__(self, autre) -> bool:
        if isinstance(autre, Comptee):
            autre = autre._valeurs
        if not isinstance(autre, (list, tuple)):
            return NotImplemented
        self.lues += len(self._valeurs)
        return self._valeurs == list(autre)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        if len(self._valeurs) > 8:
            return f"Comptee([{', '.join(map(repr, self._valeurs[:6]))}, … ({len(self._valeurs)} valeurs)])"
        return f"Comptee({self._valeurs!r})"


# ---------------------------------------------------------------------------- de gros volumes

_COURANTS = ["client", "appel", "pour", "la", "le", "de", "et", "les", "un", "une", "veut", "demande",
             "pas", "sur", "avec", "son", "mois", "depuis"]
_SUJETS = ["fibre", "mobile", "prix", "facture", "résilier", "résiliation", "coupure", "technicien",
           "offre", "fidélité", "remise", "retard", "box", "débit", "forfait", "concurrent", "geste",
           "commercial", "prélèvement", "rappel", "mécontent", "satisfait", "déménagement", "contrat"]
_SYLLABES = ["ba", "ri", "co", "lu", "té", "mon", "fa", "ser", "vi", "pa", "lé", "dor", "ni", "ca", "sol",
             "gu", "zé", "tra", "mi", "pé"]


# Les sujets suivent une loi de Zipf : le poids du k-ième (à partir de 0) est 1 / (k + 3). « fibre » finit dans
# une note sur trois, « contrat » dans une sur vingt.
_POIDS_SUJETS = [1 / (k + 3) for k in range(len(_SUJETS))]


def _mot_rare(rng: random.Random) -> str:
    return "".join(rng.choice(_SYLLABES) for _ in range(rng.randint(2, 4)))


def generer_notes(n: int = 1000, graine: int = 0) -> dict[int, str]:
    """`n` notes {identifiant: texte} tirées au hasard, reproductibles, rangées dans le désordre.

    Des mots très courants (« client » est dans la plupart des notes), des sujets moins fréquents
    (« fibre », « résilier »), et beaucoup de mots rares : la distribution des vrais textes, où quelques
    mots sont partout et la plupart presque nulle part. Environ 2 % des notes sont vides.

        >>> notes = generer_notes(500, graine=1)
        >>> len(notes), notes == generer_notes(500, graine=1)
        (500, True)
        >>> list(notes)[:3] != sorted(notes)[:3]              # pas dans l'ordre des identifiants
        True
    """
    rng = random.Random(graine)
    rares = [_mot_rare(rng) for _ in range(400)]
    identifiants = rng.sample(range(1, 10 * n + 10), n)
    notes = {}
    for identifiant in identifiants:
        if rng.random() < 0.02:
            notes[identifiant] = ""
            continue
        morceaux = []
        for _ in range(rng.randint(4, 20)):
            tirage = rng.random()
            if tirage < 0.45:
                morceaux.append(rng.choice(_COURANTS))
            elif tirage < 0.7:
                morceaux.append(rng.choices(_SUJETS, weights=_POIDS_SUJETS)[0])
            else:
                morceaux.append(rares[int(len(rares) * rng.random() ** 2)])
        if rng.random() < 0.5:                                  # « client » : dans la plupart des notes
            morceaux.insert(rng.randrange(len(morceaux) + 1), "client")
        if rng.random() < 0.5:
            morceaux[0] = morceaux[0].capitalize()
        notes[identifiant] = " ".join(morceaux) + rng.choice([".", " !", " ?", "…", ""])
    return notes


def generer_vocabulaire(n: int = 50_000, graine: int = 0) -> tuple[list[str], dict[str, int]]:
    """(vocabulaire, fréquences) : `n` mots normalisés DISTINCTS, TRIÉS, et le nombre de notes de chacun.

        >>> vocabulaire, frequences = generer_vocabulaire(1000, graine=2)
        >>> len(vocabulaire), vocabulaire == sorted(set(vocabulaire))
        (1000, True)
        >>> all(frequences[mot] >= 1 for mot in vocabulaire)
        True
    """
    rng = random.Random(graine)
    syllabes = [normaliser(s) for s in _SYLLABES]          # normalisées une fois, pas à chaque mot
    vus: set[str] = set()
    while len(vus) < n:
        vus.add("".join(rng.choice(syllabes) for _ in range(rng.randint(1, 5))))
    vocabulaire = sorted(vus)
    frequences = {mot: max(1, int(math.exp(rng.random() * 9))) for mot in vocabulaire}
    return vocabulaire, frequences

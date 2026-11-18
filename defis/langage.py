"""Outils de la série 25, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Le service marketing veut écrire LUI-MÊME ses règles de ciblage, sans demander un développeur à chaque
campagne. Une règle ressemble à ceci :

    contrat = "mensuel" et anciennete < 6 et (appels_support >= 3 ou facture > 80)

La série 25 vous fait écrire tout ce qu'il faut pour la comprendre, en quatre étages :

    le texte  ──decouper──▶  des JETONS  ──analyser──▶  un ARBRE  ──evaluer / vers_sql / vers_texte──▶ …

1. Les JETONS (défi 152) : le texte découpé en morceaux qui ont un sens, chacun avec sa position.

    >>> from defis.serie_25 import *
    >>> Jeton("OP", "<", 30)                       # l'opérateur « < », 31e caractère de la règle (position 30)
    Jeton(type='OP', valeur='<', position=30)

   Les types possibles : "NOM" (le nom d'une colonne), "NOMBRE" (valeur int ou float), "TEXTE" (valeur
   sans les guillemets), "BOOLEEN" (True ou False, écrits `vrai` et `faux`), "OP" (valeur parmi
   OPERATEURS), les mots-clés "ET", "OU", "NON", "DANS" (valeur : le mot en minuscules), la ponctuation
   "(", ")", "," (valeur : le caractère lui-même), et un dernier jeton "FIN" (valeur None, position : la
   longueur du texte).

2. L'ARBRE (défi 153) : de simples tuples, faciles à afficher et à comparer.

    ("=", "contrat", "mensuel")                    une comparaison : (opérateur, colonne, valeur)
    ("dans", "region", ("Nord", "Est"))            l'appartenance à une liste : un TUPLE de valeurs
    ("non", condition)                             la négation
    ("et", condition, condition, …)                au moins DEUX conditions, toutes vraies
    ("ou", condition, condition, …)                au moins DEUX conditions, l'une au moins vraie

   La règle du haut devient :

    >>> print(afficher_arbre(("et", ("=", "contrat", "mensuel"), ("<", "anciennete", 6),
    ...                           ("ou", (">=", "appels_support", 3), (">", "facture", 80)))))
    et
    ├── contrat = "mensuel"
    ├── anciennete < 6
    └── ou
        ├── appels_support >= 3
        └── facture > 80

3. Une erreur de syntaxe se signale par `ErreurSyntaxe(message, position)`, qui est une `ValueError` :

    >>> erreur = ErreurSyntaxe("parenthèse jamais fermée", 20)
    >>> print(souligner('contrat = "mensuel" (anciennete < 6', erreur))
    contrat = "mensuel" (anciennete < 6
                        ^ parenthèse jamais fermée

`clients_exemple`, `schema_clients` et `regles_exemple` donnent de quoi essayer ; `generer_clients` et
`generer_arbre` fabriquent autant de clients et de règles valides que vous voulez pour tester votre code ;
`base_clients` range des clients dans une base SQLite en mémoire pour essayer le défi 157.
"""

from __future__ import annotations

import math
import random
import sqlite3
from typing import NamedTuple


class Jeton(NamedTuple):
    """Un morceau de la règle. Un NamedTuple : `Jeton("NOM", "x", 0) == ("NOM", "x", 0)` vaut True."""

    type: str
    valeur: object
    position: int


class ErreurSyntaxe(ValueError):
    """Une règle mal écrite. `position` : l'indice (à partir de 0) du caractère où l'erreur a été trouvée."""

    def __init__(self, message: str, position: int):
        super().__init__(message)
        self.message = message
        self.position = position


MOTS_CLES = {"et": "ET", "ou": "OU", "non": "NON", "dans": "DANS"}
BOOLEENS = {"vrai": True, "faux": False}
OPERATEURS = ("=", "!=", "<", "<=", ">", ">=")


def souligner(texte: str, erreur: ErreurSyntaxe) -> str:
    """La règle, et en dessous un « ^ » sous le caractère fautif, suivi du message (comme Python le fait)."""
    ligne = "".join(" " if c in "\t\n\r" else c for c in texte)
    return f"{ligne}\n{' ' * erreur.position}^ {erreur.message}"


# ---------------------------------------------------------------------------- des clients pour essayer


def schema_clients() -> dict[str, str]:
    """Le genre de chaque colonne : "nombre", "texte" ou "booleen" (défi 156). Un NOUVEAU dictionnaire à chaque appel."""
    return {
        "contrat": "texte",
        "anciennete": "nombre",
        "facture": "nombre",
        "appels_support": "nombre",
        "fibre": "booleen",
        "region": "texte",
        "paiement": "texte",
    }


def clients_exemple() -> list[dict]:
    """Huit clients (une NOUVELLE liste à chaque appel). None et NaN veulent dire « on ne sait pas ».

    Les pièges : le client 4 n'a pas de région (None), la facture du client 6 est NaN (une case vide lue
    par pandas), et le client 7 a 0 appel au support (0 est une vraie valeur, pas une absence).

        >>> [c["id"] for c in clients_exemple() if c["contrat"] == "mensuel"]
        [1, 2, 4, 6]
    """
    return [
        {"id": 1, "contrat": "mensuel", "anciennete": 3, "facture": 89.9, "appels_support": 4, "fibre": True,
         "region": "Nord", "paiement": "carte"},
        {"id": 2, "contrat": "mensuel", "anciennete": 14, "facture": 45.5, "appels_support": 1, "fibre": False,
         "region": "Sud", "paiement": "prélèvement"},
        {"id": 3, "contrat": "annuel", "anciennete": 2, "facture": 99.0, "appels_support": 5, "fibre": True,
         "region": "Est", "paiement": "carte"},
        {"id": 4, "contrat": "mensuel", "anciennete": 5, "facture": 70.25, "appels_support": 3, "fibre": True,
         "region": None, "paiement": "chèque"},
        {"id": 5, "contrat": "biannuel", "anciennete": 40, "facture": 25.0, "appels_support": 0, "fibre": False,
         "region": "Ouest", "paiement": "prélèvement"},
        {"id": 6, "contrat": "mensuel", "anciennete": 1, "facture": float("nan"), "appels_support": 2,
         "fibre": True, "region": "Nord", "paiement": "carte"},
        {"id": 7, "contrat": "annuel", "anciennete": 26, "facture": 81.0, "appels_support": 0, "fibre": True,
         "region": "Centre", "paiement": "carte"},
        {"id": 8, "contrat": "biannuel", "anciennete": 60, "facture": 112.4, "appels_support": 7, "fibre": True,
         "region": "Est", "paiement": "prélèvement"},
    ]


def regles_exemple() -> dict[str, str]:
    """Des règles écrites par le marketing (un NOUVEAU dictionnaire à chaque appel).

    La dernière est un piège de PRIORITÉ : « et » passe avant « ou », comme × avant + en calcul. Elle se
    lit donc `contrat = "mensuel" ou (contrat = "annuel" et facture > 100)`.
    """
    return {
        "nouveaux_mensuels": 'contrat = "mensuel" et anciennete < 6',
        "mecontents": "appels_support >= 3 ou (facture > 80 et non fibre = vrai)",
        "regions_a_relancer": 'region dans ("Nord", "Est") et non contrat = "biannuel"',
        "campagne_ete": ('contrat = "mensuel" et anciennete < 6 et (appels_support >= 3 ou facture > 80)'),
        "piege_priorite": 'contrat = "mensuel" ou contrat = "annuel" et facture > 100',
    }


_DOMAINES = {
    "contrat": ("mensuel", "annuel", "biannuel"),
    "region": ("Nord", "Sud", "Est", "Ouest", "Centre"),
    "paiement": ("carte", "prélèvement", "chèque"),
}


def _valeur_au_hasard(rng: random.Random, colonne: str, genre: str):
    if genre == "booleen":
        return rng.random() < 0.5
    if genre == "texte":
        return rng.choice(_DOMAINES.get(colonne, ("a", "b", "c")))
    if colonne == "facture":
        return round(rng.uniform(20, 120), 2) if rng.random() < 0.6 else rng.randint(20, 120)
    if colonne == "appels_support":
        return rng.randint(0, 8)
    return rng.randint(0, 72)


def generer_clients(n: int, graine: int = 0) -> list[dict]:
    """n clients tirés au hasard (reproductibles), avec environ 8 % de valeurs inconnues (None, ou NaN pour la facture)."""
    rng = random.Random(graine)
    schema = schema_clients()
    clients = []
    for i in range(1, n + 1):
        client = {"id": i}
        for colonne, genre in schema.items():
            if rng.random() < 0.08:
                client[colonne] = float("nan") if colonne == "facture" and rng.random() < 0.5 else None
            else:
                client[colonne] = _valeur_au_hasard(rng, colonne, genre)
        clients.append(client)
    return clients


def generer_arbre(graine: int = 0, profondeur: int = 3, schema: dict | None = None) -> tuple:
    """Un arbre de règle VALIDE tiré au hasard (reproductible) : des valeurs du bon genre pour chaque colonne.

        >>> generer_arbre(graine=5, profondeur=1)
        ('ou', ('!=', 'region', 'Nord'), ('dans', 'appels_support', (3, 6)))
    """
    rng = random.Random(graine)
    schema = schema_clients() if schema is None else schema
    colonnes = sorted(schema)

    def feuille() -> tuple:
        colonne = rng.choice(colonnes)
        genre = schema[colonne]
        if rng.random() < 0.2:
            valeurs = {_valeur_au_hasard(rng, colonne, genre) for _ in range(rng.randint(1, 3))}
            return ("dans", colonne, tuple(sorted(valeurs, key=repr)))
        op = rng.choice(("=", "!=") if genre == "booleen" else OPERATEURS)
        return (op, colonne, _valeur_au_hasard(rng, colonne, genre))

    def noeud(p: int) -> tuple:
        if p <= 0 or rng.random() < 0.3:
            return feuille()
        r = rng.random()
        if r < 0.2:
            return ("non", noeud(p - 1))
        return ("et" if r < 0.6 else "ou", *(noeud(p - 1) for _ in range(rng.randint(2, 3))))

    return noeud(profondeur)


def base_clients(clients: list[dict]) -> sqlite3.Connection:
    """Une base SQLite EN MÉMOIRE avec une table `clients` (une colonne par clé ; NaN et None deviennent NULL).

        >>> conn = base_clients(clients_exemple())
        >>> conn.execute('SELECT id FROM clients WHERE "anciennete" < ? ORDER BY id', [6]).fetchall()
        [(1,), (3,), (4,), (6,)]
    """
    colonnes = list(dict.fromkeys(c for client in clients for c in client))
    conn = sqlite3.connect(":memory:")
    noms = ", ".join('"' + c.replace('"', '""') + '"' for c in colonnes)
    conn.execute(f"CREATE TABLE clients ({noms})")
    trous = ", ".join("?" * len(colonnes))

    def propre(v):
        return None if isinstance(v, float) and math.isnan(v) else v

    conn.executemany(f"INSERT INTO clients ({noms}) VALUES ({trous})",
                     [[propre(client.get(c)) for c in colonnes] for client in clients])
    return conn


# ---------------------------------------------------------------------------- afficher un arbre


def _feuille_en_texte(noeud: tuple) -> str:
    def valeur(v):
        if isinstance(v, bool):
            return "vrai" if v else "faux"
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        return repr(v)

    if noeud[0] == "dans":
        return f"{noeud[1]} dans ({', '.join(valeur(v) for v in noeud[2])})"
    return f"{noeud[1]} {noeud[0]} {valeur(noeud[2])}"


def afficher_arbre(arbre: tuple) -> str:
    """L'arbre dessiné sur plusieurs lignes (une branche par condition). Sans récursion : la profondeur importe peu."""
    lignes: list[str] = []
    pile = [(arbre, "", "", True)]
    while pile:
        noeud, prefixe, marque, dernier = pile.pop()
        if isinstance(noeud, tuple) and noeud and noeud[0] in ("et", "ou", "non"):
            lignes.append(prefixe + marque + noeud[0])
            suite = prefixe + ("" if not marque else ("    " if dernier else "│   "))
            enfants = noeud[1:]
            for i in range(len(enfants) - 1, -1, -1):
                fin = i == len(enfants) - 1
                pile.append((enfants[i], suite, "└── " if fin else "├── ", fin))
        else:
            texte = _feuille_en_texte(noeud) if isinstance(noeud, tuple) and len(noeud) == 3 else repr(noeud)
            lignes.append(prefixe + marque + texte)
    return "\n".join(lignes)

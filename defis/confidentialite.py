"""Outils communs de la série 34 : les clients à partager, le fichier de l'attaquant, les hiérarchies, les résultats.

Le laboratoire d'économie de l'université veut étudier le churn sur de vraies données. Inès, la déléguée à la
protection des données (DPO), doit signer avant que le moindre fichier parte. Les données :

- `clients_exemple()` : 20 clients de l'agglomération lyonnaise, avec leurs identifiants DIRECTS (numéro client,
  nom, e-mail, téléphone), leurs QUASI-IDENTIFIANTS (âge, sexe, code postal : rien de secret, mais ensemble ils
  désignent souvent une seule personne), et des informations SENSIBLES (le motif de départ, dont « impayé ») ;
- `annuaire_exemple()` : le fichier de prospection qu'un concurrent a acheté (nom, âge, sexe, code postal). C'est
  l'arme de l'attaquant : il ne contient aucune donnée sensible, mais il a des noms ;
- `IDENTIFIANTS_DIRECTS`, `QUASI_IDENTIFIANTS`, `COLONNES_ETUDE` : les colonnes de chaque sorte ;
- `HIERARCHIES` : pour chaque quasi-identifiant, la liste des façons de le rendre moins précis, du niveau 0 (la
  valeur exacte) au dernier niveau (« * », tout le monde pareil). `appliquer(ligne, HIERARCHIES, niveaux)` les
  applique à une ligne ;
- `comptages_exemple()` : le tableau des départs par commune et par motif qu'on veut publier (défi 210) ;
- `Classe`, `Audit`, `Generalisation` (des tuples nommés), `BudgetEpuise` (une exception) : les résultats ;
- `tirages_fixes(…)`, `tirage_hasard(graine)` : des sources de hasard pour le défi 211 ;
- `generer_clients(n, graine)`, `generer_annuaire(clients, graine)` : de longs fichiers reproductibles pour vos
  propres essais.

Toutes les personnes sont FICTIVES. Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans
crainte.

    >>> clients = clients_exemple()
    >>> len(clients), sorted(clients[0])
    (20, ['age', 'anciennete', 'churn', 'code_postal', 'contrat', 'customer_id', 'email', 'facture', 'motif', 'nom',
          'sexe', 'telephone'])
    >>> clients[0]["nom"], clients[0]["email"], clients[0]["age"], clients[0]["code_postal"]
    ('Camille Bernard', 'camille.bernard@exemple.fr', 34, '69003')
    >>> len(annuaire_exemple()), annuaire_exemple()[0]
    (23, {'nom': 'Camille Bernard', 'age': 34, 'sexe': 'F', 'code_postal': '69003'})
    >>> HIERARCHIES["age"][2](34), HIERARCHIES["code_postal"][2]("69003"), HIERARCHIES["sexe"][1]("F")
    ('30-39', '690**', '*')
    >>> appliquer({"age": 34, "sexe": "F", "motif": None}, HIERARCHIES, {"age": 1, "sexe": 0})
    {'age': '30-34', 'sexe': 'F', 'motif': None}
"""

from __future__ import annotations

import random
from typing import Callable, NamedTuple

IDENTIFIANTS_DIRECTS = frozenset({"customer_id", "nom", "email", "telephone"})   # ne partent JAMAIS
QUASI_IDENTIFIANTS = ("age", "sexe", "code_postal")                              # anodins seuls, pas ensemble
COLONNES_ETUDE = ("age", "sexe", "code_postal", "contrat", "anciennete", "churn", "motif")   # ce que demande le labo
CLE_EXEMPLE = b"cle-reservee-au-labo-2026-a-garder-au-coffre"                    # 44 octets


class Classe(NamedTuple):
    """Une classe d'équivalence (défi 208) : les lignes qui partagent les mêmes quasi-identifiants."""

    valeurs: tuple                # les valeurs des quasi-identifiants (None si la valeur manque)
    effectif: int                 # combien de lignes : l'attaquant hésite entre `effectif` personnes
    diversite: int                # combien de valeurs DIFFÉRENTES de l'attribut sensible dans la classe


class Audit(NamedTuple):
    """Le risque d'un fichier (défi 208)."""

    k: int                        # le plus petit effectif : le fichier est « k-anonyme »
    l: int                        # la plus petite diversité : le fichier est « l-divers » (l comme le L de l-diversité)
    classes: list                 # les Classe, les plus exposées d'abord


class Generalisation(NamedTuple):
    """Le résultat du défi 209."""

    niveaux: dict                 # {quasi-identifiant: niveau choisi dans sa hiérarchie}
    lignes: list                  # les lignes publiables (copies généralisées, sans les lignes supprimées)
    supprimees: int               # combien de lignes ont été retirées du fichier


class BudgetEpuise(Exception):
    """Levée quand une question coûterait plus de confidentialité qu'il n'en reste (défi 211)."""


def _manquant(x) -> bool:
    return x is None or (isinstance(x, float) and x != x)


# ---------------------------------------------------------------------------- hiérarchies de généralisation


def exact(x):
    """Niveau 0 : la valeur telle quelle (une valeur manquante, None ou NaN, devient None)."""
    return None if _manquant(x) else x


def masquer(x) -> str:
    """Le dernier niveau : « * » pour tout le monde (valeur manquante comprise)."""
    return "*"


def _tranche(largeur: int) -> Callable:
    def tranche(age):
        if _manquant(age):
            return None
        debut = int(age) // largeur * largeur
        return f"{debut}-{debut + largeur - 1}"

    tranche.__name__ = f"age_par_{largeur}"
    debut = 34 // largeur * largeur
    tranche.__doc__ = f"L'âge en tranches de {largeur} ans : 34 → '{debut}-{debut + largeur - 1}'."
    return tranche


def _prefixe(chiffres: int) -> Callable:
    def prefixe(code):
        if _manquant(code):
            return None
        code = str(code)
        return code[:chiffres] + "*" * (len(code) - chiffres)

    prefixe.__name__ = f"cp_{chiffres}_chiffres"
    exemple = "69003"[:chiffres] + "*" * (5 - chiffres)
    prefixe.__doc__ = f"Les {chiffres} premiers chiffres du code postal : '69003' → '{exemple}'."
    return prefixe


age_par_5, age_par_10, age_par_20 = _tranche(5), _tranche(10), _tranche(20)
cp_4_chiffres, cp_3_chiffres, cp_2_chiffres = _prefixe(4), _prefixe(3), _prefixe(2)

HIERARCHIES: dict[str, list[Callable]] = {
    "age": [exact, age_par_5, age_par_10, age_par_20, masquer],                         # niveaux 0 à 4
    "sexe": [exact, masquer],                                                           # niveaux 0 et 1
    "code_postal": [exact, cp_4_chiffres, cp_3_chiffres, cp_2_chiffres, masquer],       # niveaux 0 à 4
}


def appliquer(ligne: dict, hierarchies: dict, niveaux: dict) -> dict:
    """Une COPIE de `ligne` où chaque colonne de `niveaux` est généralisée au niveau demandé (les autres intactes).

        >>> appliquer({"age": 34, "code_postal": "69003", "churn": 1}, HIERARCHIES, {"age": 2, "code_postal": 3})
        {'age': '30-39', 'code_postal': '69***', 'churn': 1}
    """
    copie = dict(ligne)
    for colonne, niveau in niveaux.items():
        copie[colonne] = hierarchies[colonne][niveau](ligne.get(colonne))
    return copie


# ---------------------------------------------------------------------------- les clients et l'annuaire

_CHAMPS = ("customer_id", "nom", "age", "sexe", "code_postal", "contrat", "anciennete", "facture", "churn", "motif")

#            id        nom                     âge sexe  cp       contrat    anc.  facture churn motif
_CLIENTS = [
    ("C00412", "Camille Bernard",       34, "F", "69003", "mensuel",   8,  64.5, 1, "prix"),
    ("C00418", "Lucas Moreau",          41, "M", "69003", "annuel",   30,  89.0, 0, None),
    ("C00433", "Inès Haddad",           36, "F", "69007", "mensuel",   3,  71.0, 1, "concurrent"),
    ("C00440", "Hugo Lefebvre",         29, "M", "69007", "mensuel",  14,  45.5, 0, None),
    ("C00457", "Chloé Garnier",         52, "F", "69008", "biennal",  60, 102.0, 0, None),
    ("C00461", "Mehdi Benali",          47, "M", "69008", "mensuel",   5,  58.0, 1, "impayé"),
    ("C00478", "Léa Fontaine",          23, "F", "69100", "mensuel",   2,  39.0, 1, "prix"),
    ("C00483", "Nathan Roux",           26, "M", "69100", "annuel",   18,  52.5, 0, None),
    ("C00495", "Manon Girard",          38, "F", "69100", "mensuel",  11,  77.0, 1, "impayé"),
    ("C00502", "Karim Mansouri",        64, "M", "69120", "mensuel",   7,  49.0, 1, "impayé"),
    ("C00516", "Sophie Petit",          67, "F", "69120", "mensuel",   4,  55.5, 1, "impayé"),
    ("C00521", "Paul Mercier",          71, "M", "69120", "annuel",   40,  61.0, 1, "impayé"),
    ("C00537", "Julie Lambert",         45, "F", "69300", "biennal",  72, 118.0, 0, None),
    ("C00544", "Antoine Dubois",        58, "M", "69300", "annuel",   36,  95.5, 0, None),
    ("C00559", "Sarah Cohen",           31, "F", "69500", "mensuel",   6,  66.0, 1, "concurrent"),
    ("C00563", "Thomas Faure",          33, "M", "69500", "mensuel",   9,  62.5, 0, None),
    ("C00578", "Emma Rousseau",         19, "F", "69500", "mensuel",   1,  35.0, 1, "déménagement"),
    ("C00586", "Youssef Amrani",        44, "M", "69008", "annuel",   26,  84.0, 0, None),
    ("C00591", "Claire Vidal",          None, "F", "69005", "mensuel", 12,  70.0, 0, None),
    ("C00605", "Louis Chevalier",       82, "M", "69300", "biennal",  96,  40.0, 1, "déménagement"),
]

#              nom                  âge sexe cp
_ANNUAIRE = [
    ("Camille Bernard",     34, "F", "69003"),
    ("Lucas Moreau",        41, "M", "69003"),
    ("Inès Haddad",         36, "F", "69007"),
    ("Hugo Lefebvre",       29, "M", "69007"),
    ("Mehdi Benali",        47, "M", "69008"),
    ("Léa Fontaine",        23, "F", "69100"),
    ("Manon Girard",        38, "F", "69100"),
    ("Karim Mansouri",      64, "M", "69120"),
    ("Sophie Petit",        67, "F", "69120"),
    ("Paul Mercier",        71, "M", "69120"),
    ("Sarah Cohen",         31, "F", "69500"),
    ("Emma Rousseau",       19, "F", "69500"),
    ("Louis Chevalier",     82, "M", "69300"),
    ("Antoine Dubois",      58, "M", "69300"),
    ("Pierre Bernard",      41, "M", "69003"),     # même âge, même sexe, même quartier que Lucas Moreau
    ("Marie Lopez",         36, "F", "69003"),
    ("Kevin Morel",         26, "M", "69120"),
    ("Aurélie Simon",       52, "F", "69007"),
    ("Jean Masson",         58, "M", "69300"),     # le jumeau statistique d'Antoine Dubois
    ("Nadia Brahimi",       45, "F", "69500"),
    ("Olivier Blanc",       33, "M", "69100"),
    ("Lina Caron",          31, "F", "69008"),
    ("Claire Vidal",        49, "F", "69005"),     # son âge a été effacé du fichier… pas son quartier
]


def _email(nom: str) -> str:
    prenom, _, famille = nom.partition(" ")
    sans = str.maketrans("éèêëàâîïôöûüç", "eeeeaaiioouuc")
    return f"{prenom.lower().translate(sans)}.{famille.lower().translate(sans)}@exemple.fr"


def clients_exemple() -> list[dict]:
    """Les 20 clients du fichier demandé par le laboratoire, identifiants directs compris."""
    clients = []
    for i, valeurs in enumerate(_CLIENTS):
        ligne = dict(zip(_CHAMPS, valeurs))
        ligne["email"] = _email(ligne["nom"])
        ligne["telephone"] = f"06 12 34 {50 + i:02d} {17 + 3 * i:02d}"
        clients.append(ligne)
    return clients


def annuaire_exemple() -> list[dict]:
    """Le fichier de prospection de l'attaquant : 23 personnes, avec leur nom (et rien de sensible)."""
    return [dict(zip(("nom", "age", "sexe", "code_postal"), valeurs)) for valeurs in _ANNUAIRE]


# ---------------------------------------------------------------------------- le tableau à publier (défi 210)

COMMUNES = ("Lyon 3e", "Lyon 8e", "Villeurbanne", "Vaulx-en-Velin", "Bron")
MOTIFS = ("prix", "concurrent", "déménagement", "impayé")

_COMPTAGES = [
    [41, 23, 12,  8],      # Lyon 3e
    [35, 19,  9,  2],      # Lyon 8e         : 2 impayés, une case à cacher
    [52, 31, 15, 11],      # Villeurbanne
    [18,  7,  1,  4],      # Vaulx-en-Velin  : 1 déménagement, 4 impayés
    [27, 14,  6,  9],      # Bron
]


def comptages_exemple() -> list[list[int]]:
    """Les départs de l'année : une ligne par commune (COMMUNES), une colonne par motif (MOTIFS)."""
    return [list(ligne) for ligne in _COMPTAGES]


# ---------------------------------------------------------------------------- le hasard du défi 211


def tirages_fixes(valeurs) -> Callable[[], float]:
    """Une source de « hasard » qui renvoie `valeurs` dans l'ordre (pratique pour tester). Chacune dans ]0, 1[.

        >>> tirage = tirages_fixes([0.75, 0.5])
        >>> tirage(), tirage()
        (0.75, 0.5)
    """
    file = list(valeurs)
    for u in file:
        if not 0 < u < 1:
            raise ValueError(f"tirage {u!r} : il faut 0 < u < 1")

    def tirage() -> float:
        if not file:
            raise RuntimeError("plus aucun tirage prévu : la fonction en demande plus que les questions posées")
        return file.pop(0)

    return tirage


def tirage_hasard(graine: int = 0) -> Callable[[], float]:
    """Une source de hasard reproductible : chaque appel renvoie un nombre uniforme dans ]0, 1[ (jamais 0)."""
    rng = random.Random(graine)
    return lambda: rng.randrange(1, 2 ** 53) / 2 ** 53


# ---------------------------------------------------------------------------- de longs fichiers pour s'entraîner

_PRENOMS = ("Camille", "Lucas", "Inès", "Hugo", "Chloé", "Mehdi", "Léa", "Nathan", "Manon", "Karim", "Sophie", "Paul",
            "Julie", "Antoine", "Sarah", "Thomas", "Emma", "Youssef", "Claire", "Louis", "Nadia", "Olivier", "Lina")
_NOMS = ("Bernard", "Moreau", "Haddad", "Lefebvre", "Garnier", "Benali", "Fontaine", "Roux", "Girard", "Mansouri",
         "Petit", "Mercier", "Lambert", "Dubois", "Cohen", "Faure", "Rousseau", "Amrani", "Vidal", "Chevalier")
_CODES = ("69001", "69002", "69003", "69004", "69005", "69006", "69007", "69008", "69009", "69100", "69120",
          "69200", "69300", "69500", "69600")


def generer_clients(n: int = 1000, graine: int = 0, manquants: float = 0.02) -> list[dict]:
    """`n` clients fictifs reproductibles, avec quelques valeurs manquantes (None) dans l'âge et le code postal."""
    rng = random.Random(graine)
    clients = []
    for i in range(n):
        nom = f"{rng.choice(_PRENOMS)} {rng.choice(_NOMS)}"
        churn = int(rng.random() < 0.35)
        ligne = {
            "customer_id": f"C{i:06d}",
            "nom": nom,
            "email": _email(nom).replace("@", f"{i}@"),
            "telephone": f"06 {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}",
            "age": rng.randint(18, 90),
            "sexe": rng.choice("FM"),
            "code_postal": rng.choice(_CODES),
            "contrat": rng.choice(("mensuel", "mensuel", "annuel", "biennal")),
            "anciennete": rng.randint(1, 72),
            "facture": rng.randint(60, 240) / 2,
            "churn": churn,
            "motif": rng.choice(("prix", "concurrent", "déménagement", "impayé")) if churn else None,
        }
        for colonne in ("age", "code_postal"):
            if rng.random() < manquants:
                ligne[colonne] = None
        clients.append(ligne)
    return clients


def generer_annuaire(clients: list[dict], graine: int = 0, part: float = 0.6, autres: int | None = None) -> list[dict]:
    """Un fichier de l'attaquant : une `part` des clients (nom, âge, sexe, code postal) et `autres` inconnus."""
    rng = random.Random(graine)
    annuaire = [{c: ligne.get(c) for c in ("nom", "age", "sexe", "code_postal")}
                for ligne in clients if rng.random() < part]
    for _ in range(len(clients) // 2 if autres is None else autres):
        annuaire.append({"nom": f"{rng.choice(_PRENOMS)} {rng.choice(_NOMS)}", "age": rng.randint(18, 90),
                         "sexe": rng.choice("FM"), "code_postal": rng.choice(_CODES)})
    rng.shuffle(annuaire)
    return annuaire

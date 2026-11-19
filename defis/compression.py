"""Outils communs de la série 41 : l'instantané des clients, le journal des événements, les notes du support.

Chaque nuit, l'instantané des clients part en CSV vers l'entrepôt de données et vers le partenaire qui gère les
relances, et la facture de stockage a triplé en un an. Le journal des événements de l'application arrive toutes
les heures des antennes de zone rurale, où les transferts échouent une fois sur trois. Et lundi, le fichier du
partenaire est arrivé tronqué : le script de lecture a planté à 3 h du matin, et toute la chaîne de nuit s'est
arrêtée. Ousmane, ingénieur données, écrit les encodages de Parquet et de Protobuf pour comprendre ce qu'ils
gagnent, et un lecteur qui refuse proprement un fichier abîmé. Les données :

- `instantane_exemple()` : l'instantané de la nuit, colonne par colonne (un dict de listes), rangé par contrat
  puis par numéro de client, comme le demande l'équipe décisionnelle (défis 248 à 250) ;
- `evenements_exemple()` : les horodatages (en secondes) des 50 000 événements d'une journée, dans l'ordre où ils
  sont ARRIVÉS au serveur : presque triés, mais quelques-uns arrivent en retard (défis 251 et 253) ;
- `notes_exemple()` : le texte des notes saisies par le support dans la journée (défi 252) ;
- `taille_csv(valeurs)` : le nombre d'octets qu'occupe une colonne écrite en texte, une valeur par ligne
  (c'est contre elle que l'on mesure ce que chaque encodage fait gagner) ;
- `bits_lus(octets)` : les bits d'une suite d'octets, dans l'ordre où un lecteur de Parquet les lit (le bit de
  POIDS FAIBLE de chaque octet d'abord), pour déboguer le défi 250 ;
- `generer_colonne(n, valeurs, graine)` et `generer_horodatages(n, graine)` : d'autres données, reproductibles,
  pour vos propres essais.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> instantane = instantane_exemple()
    >>> sorted(instantane), len(instantane["contrat"])
    (['anciennete', 'contrat', 'motif', 'numero', 'paiement'], 20000)
    >>> instantane["contrat"][:2], instantane["numero"][:3], instantane["motif"][:3]
    (['Mensuel', 'Mensuel'], [1, 6, 8], [nan, nan, nan])
    >>> taille_csv(instantane["contrat"]), taille_csv(instantane["paiement"])
    (158927, 248499)
    >>> evenements = evenements_exemple()
    >>> len(evenements), evenements[:3], taille_csv(evenements)
    (50000, [1794960000, 1794960004, 1794960006], 550000)
"""

from __future__ import annotations

import math
import random

CONTRATS = ("Mensuel", "Annuel", "Biannuel")
PAIEMENTS = ("Prélèvement", "Carte bancaire", "Virement", "Chèque")
MOTIFS = ("Prix", "Concurrence", "Déménagement", "Qualité du service")

MINUIT = 1_794_960_000          # le 18 novembre 2026 à 0 h (UTC), en secondes depuis le 1er janvier 1970


def taille_csv(valeurs) -> int:
    """Le nombre d'octets de la colonne écrite en texte UTF-8, une valeur par ligne (NaN et None : ligne vide).

    C'est ce que coûte la colonne dans le CSV de la nuit : « Prélèvement » prend 14 octets (11 caractères, dont
    « é » et « è » qui en prennent deux chacun, plus le retour à la ligne), à chacune de ses 7 047 apparitions.

        >>> taille_csv(["Mensuel", "Prélèvement", 1794960000, math.nan, None])     # 8 + 14 + 11 + 1 + 1
        35
    """
    total = 0
    for v in valeurs:
        vide = v is None or (isinstance(v, float) and math.isnan(v))
        total += (0 if vide else len(str(v).encode("utf-8"))) + 1
    return total


def bits_lus(octets) -> str:
    """Les bits de `octets`, octet par octet, chacun lu à partir de son bit de POIDS FAIBLE (le bit 0, qui vaut 1).

    C'est l'ordre dans lequel Parquet range les codes du défi 250 : la position 0 du flux est le bit 0 du premier
    octet, la position 8 le bit 0 du deuxième. Lisez donc chaque code de gauche à droite, mais chaque nombre
    À L'ENVERS (bit de poids faible d'abord).

        >>> bits_lus(b"\\x01\\x80")                     # 1 = bit 0 du premier octet, 0x80 = bit 7 du second
        '10000000 00000001'
        >>> bits_lus(bytes([0b11010100, 0b00000010]))  # les codes 0, 1, 1, 3, 2 sur 2 bits : 00 10 10 11 01
        '00101011 01000000'
    """
    if not isinstance(octets, (bytes, bytearray)):
        raise TypeError(f"bits_lus() attend des bytes, pas {type(octets).__name__}")
    return " ".join("".join(str(o >> j & 1) for j in range(8)) for o in octets)


# ---------------------------------------------------------------------------- l'instantané (248 à 250)


def generer_colonne(n: int = 1000, valeurs=("a", "b", "c"), graine: int = 0) -> list:
    """`n` valeurs tirées au hasard (et de façon reproductible) parmi `valeurs`, en plages de longueurs variées.

        >>> generer_colonne(10, ("x", "y"), graine=4)
        ['x', 'x', 'y', 'y', 'y', 'x', 'y', 'y', 'y', 'y']
    """
    rng = random.Random(graine)
    colonne: list = []
    while len(colonne) < n:
        colonne += [rng.choice(valeurs)] * min(n - len(colonne), 1 + int(rng.expovariate(0.3)))
    return colonne


def _tirer(rng: random.Random, valeurs, poids):
    return rng.choices(valeurs, weights=poids)[0]


def instantane_exemple(n: int = 20_000) -> dict[str, list]:
    """L'instantané de la nuit : `n` clients, colonne par colonne, rangés par contrat puis par numéro de client.

    - "numero" : le numéro du client (un entier, attribué à l'inscription : les plus anciens ont les petits) ;
    - "contrat" : « Mensuel », « Annuel » ou « Biannuel » (rangés dans cet ordre : trois longues plages) ;
    - "paiement" : l'un des quatre moyens de paiement, dans le désordre ;
    - "anciennete" : en mois ;
    - "motif" : le motif de résiliation, et NaN (`float("nan")`, comme dans pandas) pour les clients restés.

        >>> instantane = instantane_exemple(8)
        >>> instantane["contrat"]
        ['Mensuel', 'Mensuel', 'Mensuel', 'Annuel', 'Annuel', 'Annuel', 'Biannuel', 'Biannuel']
        >>> instantane["numero"]                   # croissant DANS chaque contrat : il redescend deux fois
        [1, 6, 8, 2, 3, 5, 4, 7]
    """
    rng = random.Random(41)
    lignes = []
    for numero in range(1, n + 1):
        contrat = _tirer(rng, CONTRATS, (0.55, 0.25, 0.20))
        paiement = _tirer(rng, PAIEMENTS, (0.35, 0.30, 0.20, 0.15))
        anciennete = 1 + int(rng.expovariate(1 / (14, 30, 45)[CONTRATS.index(contrat)]))
        risque = (0.14, 0.05, 0.02)[CONTRATS.index(contrat)] + (0.04 if paiement == "Chèque" else 0.0)
        motif = _tirer(rng, MOTIFS, (0.4, 0.3, 0.1, 0.2)) if rng.random() < risque else None
        lignes.append((CONTRATS.index(contrat), numero, contrat, paiement, min(anciennete, 72), motif))
    lignes.sort()
    return {
        "numero": [ligne[1] for ligne in lignes],
        "contrat": [ligne[2] for ligne in lignes],
        "paiement": [ligne[3] for ligne in lignes],
        "anciennete": [ligne[4] for ligne in lignes],
        # un NOUVEL objet NaN par ligne, comme dans une colonne pandas : deux NaN ne sont jamais égaux
        "motif": [ligne[5] if ligne[5] is not None else float("nan") for ligne in lignes],
    }


# ---------------------------------------------------------------------------- le journal des événements (251, 253)


def generer_horodatages(n: int = 1000, graine: int = 0, debut: int = MINUIT, retards: float = 0.03) -> list[int]:
    """`n` horodatages (en secondes) dans l'ordre d'ARRIVÉE au serveur : une part `retards` arrive en retard.

    Les événements se produisent toutes les 1,7 seconde en moyenne. La plupart arrivent aussitôt ; certains
    (un téléphone qui retrouve le réseau) arrivent jusqu'à 90 secondes plus tard, après des événements plus
    récents qu'eux : l'horodatage RECULE. C'est pour eux qu'il faut le zigzag du défi 251.

        >>> generer_horodatages(6, graine=5, debut=0, retards=0.3)     # l'événement de la seconde 9 arrive en retard
        [1, 6, 6, 8, 10, 9]
    """
    rng = random.Random(graine)
    t = 0.0
    evenements = []
    for _ in range(n):
        t += rng.expovariate(1 / 1.7)
        arrivee = t + (rng.uniform(1, 90) if rng.random() < retards else rng.uniform(0, 0.5))
        evenements.append((arrivee, debut + int(t)))
    evenements.sort()
    return [horodatage for _, horodatage in evenements]


def evenements_exemple() -> list[int]:
    """Les horodatages des 50 000 événements du 18 novembre, dans l'ordre d'arrivée (défis 251 et 253).

        >>> evenements = evenements_exemple()
        >>> sum(b < a for a, b in zip(evenements, evenements[1:]))     # fois où l'horodatage recule
        1733
    """
    return generer_horodatages(50_000, graine=41)


# ---------------------------------------------------------------------------- les notes du support (252)

_DEBUTS = ("Client mécontent", "Cliente mécontente", "Appel entrant", "Rappel demandé", "Client fidèle",
           "Réclamation", "Demande de résiliation", "Question sur la facture")
_SUJETS = ("du prix", "de la hausse de tarif", "d'une coupure de réseau", "du débit en soirée", "de la box",
           "d'un prélèvement en double", "du délai d'intervention", "de l'offre concurrente")
_SUITES = ("propose une remise de 10 %", "geste commercial accordé", "technicien envoyé", "à rappeler jeudi",
           "transfert au service fidélisation", "aucune solution trouvée", "explications données, client rassuré",
           "dossier escaladé")


def notes_exemple(n: int = 2000) -> str:
    """Le texte des `n` notes saisies par le support dans la journée, une par ligne.

        >>> print(notes_exemple(2))
        Demande de résiliation : à propos d'un prélèvement en double, à rappeler jeudi.
        Appel entrant : à propos du délai d'intervention, transfert au service fidélisation.
    """
    rng = random.Random(41)
    notes = [f"{rng.choice(_DEBUTS)} : à propos {rng.choice(_SUJETS)}, {rng.choice(_SUITES)}." for _ in range(n)]
    return "\n".join(notes)

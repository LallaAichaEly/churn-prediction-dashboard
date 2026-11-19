"""Outils communs de la série 37 : les factures, les jours fériés, les créances et le relevé bancaire de novembre.

Nadia vient d'arriver au service facturation. Deux nouvelles l'attendent. La comptable a trouvé des écarts de
quelques centimes sur près d'une facture sur deux. Et l'équipe data a remarqué qu'une partie des départs du mois
n'en étaient pas : un prélèvement refusé (carte expirée, compte vide un 28 du mois) que personne n'a relancé. On
appelle ça le CHURN INVOLONTAIRE : le client voulait rester. Les données :

- `lignes_exemple()` : les lignes de la facture de novembre du client C04217 : son forfait, ses SMS hors forfait
  facturés 0.045 € pièce, son bouquet TV, son kiosque presse et un livre audio, à quatre taux de TVA (défi 226) ;
- `creances_exemple()` : les 6 035 factures d'octobre et de novembre de 3 000 clients (une par mois, plus 35
  régularisations), avec leur numéro, leur montant TTC et leur échéance (défi 229) ;
- `paiements_exemple()` : le relevé bancaire de ces deux mois, DANS LE DÉSORDRE (la banque exporte par lot) :
  prélèvements, virements avec ou sans numéro de facture, numéros tapés en minuscules ou avec une faute de
  frappe, clients qui paient deux mois d'un coup, trop ou pas assez, et virements qu'on ne sait pas attribuer ;
- `TAUX_TVA`, `DELAIS` et `GRACE` : les taux de TVA français, les délais des relances après un prélèvement refusé
  et le délai avant de suspendre l'abonnement (défi 228) ;
- `TotalTaux`, `Facture`, `Relances`, `Creance`, `Paiement`, `Affectation` et `Rapprochement` : les données et
  les résultats, des tuples nommés ;
- `generer_lignes`, `generer_creances` et `generer_paiements` : d'autres données, reproductibles, pour vos
  propres essais ; `euros(centimes)` : « 1 234,56 € » pour afficher un montant.

Tous les montants sont des NOMBRES ENTIERS DE CENTIMES : 29,99 € s'écrit 2999. Un float ne sait pas écrire 0,1
exactement, et une facture n'a pas le droit de perdre un centime.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> lignes_exemple()[1]
    {'libelle': 'SMS hors forfait', 'quantite': 1234, 'prix_unitaire': '0.045', 'tva': '20'}
    >>> creances = creances_exemple()
    >>> len(creances), creances[0]
    (6035, Creance(numero='FAC-2026-00001', client='C00001', montant=2498, echeance=datetime.date(2026, 10, 5)))
    >>> paiements = paiements_exemple()
    >>> len(paiements), paiements[0]                  # le premier de l'export : un virement anonyme
    (5532, Paiement(id='B002669', jour=datetime.date(2026, 10, 11), client=None, montant=2999, libelle='VIR INST 312845'))
    >>> euros(123456), euros(-350), euros(5)
    ('1 234,56 €', '-3,50 €', '0,05 €')
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import NamedTuple

TAUX_TVA = ("20", "10", "5.5", "2.1")  # en %, écrits en TEXTE : normal, intermédiaire (TV), réduit (livres), presse
DELAIS = (1, 3, 7, 14)                 # jours (calendaires) entre le prélèvement refusé et chaque nouvel essai
GRACE = 7                              # jours entre le dernier essai et la suspension de l'abonnement


class TotalTaux(NamedTuple):
    """Le total d'un taux de TVA sur une facture (défi 226). Un tuple : (taux, ht, tva)."""

    taux: str                 # "20", "10", "5.5" ou "2.1"
    ht: int                   # la somme HT des lignes à ce taux, remise déduite, en centimes
    tva: int                  # la TVA de ce taux, calculée UNE fois sur `ht` et arrondie au centime


class Facture(NamedTuple):
    """Une facture calculée (défi 226). Un tuple de cinq valeurs, toutes en centimes."""

    lignes: list[int]         # le HT de chaque ligne, remise déduite, dans l'ordre reçu
    par_taux: list[TotalTaux]  # un total par taux utilisé, du taux le plus FORT au plus faible
    total_ht: int
    total_tva: int
    total_ttc: int            # total_ht + total_tva


class Relances(NamedTuple):
    """Le calendrier des relances d'un prélèvement refusé (défi 228). Un tuple : (essais, suspension)."""

    essais: list[date]        # les jours où l'on représente le prélèvement, dans l'ordre
    suspension: date          # le jour où l'abonnement est suspendu si tout a échoué


class Creance(NamedTuple):
    """Une facture émise, qu'on attend d'être payée (défi 229)."""

    numero: str               # « FAC-2026-00123 » : FAC, l'année, cinq chiffres
    client: str
    montant: int              # TTC, en centimes
    echeance: date            # le jour où elle doit être payée


class Paiement(NamedTuple):
    """Une ligne du relevé bancaire : de l'argent reçu (défi 229)."""

    id: str                   # l'identifiant de l'opération à la banque (unique)
    jour: date
    client: str | None        # le client que la banque a reconnu (None : on ne sait pas qui a payé)
    montant: int              # en centimes, toujours > 0
    libelle: str              # le texte libre du virement ou du prélèvement


class Affectation(NamedTuple):
    """« Tant de centimes de tel paiement règlent telle facture » (défi 229). Un tuple : (paiement, numero, montant)."""

    paiement: str             # l'id du paiement
    numero: str               # le numéro de la facture
    montant: int              # en centimes, > 0


class Rapprochement(NamedTuple):
    """Le lettrage du relevé (défi 229). Un tuple de quatre valeurs."""

    affectations: list[Affectation]  # dans l'ordre où on les a faites
    restes: dict[str, int]    # numéro → ce qui reste dû (> 0 seulement), dans l'ordre des créances reçues
    credits: dict[str, int]   # client → trop-perçu (> 0 seulement), dans l'ordre où chaque crédit est apparu
    non_affectes: list[str]   # les id des paiements qu'on ne sait pas attribuer, dans l'ordre de traitement


def euros(centimes: int) -> str:
    """Un montant en centimes, écrit à la française.

        >>> euros(2999), euros(1234567), euros(0)
        ('29,99 €', '12 345,67 €', '0,00 €')
    """
    signe = "-" if centimes < 0 else ""
    unites, cents = divmod(abs(centimes), 100)
    return f"{signe}{unites:,}".replace(",", " ") + f",{cents:02d} €"


# ---------------------------------------------------------------------------- la facture (défi 226)


def lignes_exemple() -> list[dict]:
    """Les lignes de la facture de novembre de C04217 (les prix unitaires sont HT, en euros, écrits en TEXTE).

        >>> [(l["libelle"], l["tva"]) for l in lignes_exemple()]
        [('Forfait Fibre + Mobile', '20'), ('SMS hors forfait', '20'), ('Bouquet Cinéma', '10'), ('Kiosque presse', '2.1'), ('Livre audio', '5.5'), ('Option sécurité', '20')]
    """
    return [
        {"libelle": "Forfait Fibre + Mobile", "quantite": 1, "prix_unitaire": "39.99", "tva": "20"},
        {"libelle": "SMS hors forfait", "quantite": 1234, "prix_unitaire": "0.045", "tva": "20"},
        {"libelle": "Bouquet Cinéma", "quantite": 1, "prix_unitaire": "12.49", "tva": "10"},
        {"libelle": "Kiosque presse", "quantite": 1, "prix_unitaire": "9.985", "tva": "2.1"},
        {"libelle": "Livre audio", "quantite": 1, "prix_unitaire": "3.00", "tva": "5.5"},
        {"libelle": "Option sécurité", "quantite": 3, "prix_unitaire": "1.005", "tva": "20"},
    ]


def generer_lignes(n: int = 20, graine: int = 0) -> list[dict]:
    """`n` lignes de facture au hasard, reproductibles : quantités, prix au millième d'euro, quatre taux.

        >>> generer_lignes(2, graine=1)
        [{'libelle': 'ligne 1', 'quantite': 1, 'prix_unitaire': '37.303', 'tva': '20'}, {'libelle': 'ligne 2', 'quantite': 3, 'prix_unitaire': '7.727', 'tva': '10'}]
    """
    rng = random.Random(graine)
    lignes = []
    for i in range(1, n + 1):
        quantite = rng.choice((1, 1, 1, 2, 3, 12, 250, 1234))
        milliemes = rng.randrange(0, 60_000 if quantite < 12 else 200)
        prix = f"{milliemes // 1000}.{milliemes % 1000:03d}"
        lignes.append({"libelle": f"ligne {i}", "quantite": quantite, "prix_unitaire": prix,
                       "tva": rng.choice(("20", "20", "20", "10", "5.5", "2.1"))})
    return lignes


# ---------------------------------------------------------------------------- les créances et le relevé (défi 229)

_PRIX = (1499, 1999, 2499, 2999, 3299, 3999, 4999)


def generer_creances(n_clients: int = 3000, graine: int = 0, annee: int = 2026, mois: tuple[int, ...] = (10, 11),
                     premier_client: int = 1) -> list[Creance]:
    """Une facture par client et par mois, à échéance le 5 ; quelques clients ont une facture de régularisation.

    Les numéros se suivent dans l'ordre d'émission : une régularisation émise après coup peut avoir un numéro
    plus grand qu'une facture de novembre ET une échéance plus ancienne.

        >>> [c.numero for c in generer_creances(2, graine=3)]
        ['FAC-2026-00001', 'FAC-2026-00002', 'FAC-2026-00003', 'FAC-2026-00004']
    """
    rng = random.Random(graine)
    prix = {k: rng.choice(_PRIX) for k in range(premier_client, premier_client + n_clients)}
    creances: list[Creance] = []

    def emettre(client: str, montant: int, echeance: date) -> None:
        creances.append(Creance(f"FAC-{annee}-{len(creances) + 1:05d}", client, montant, echeance))

    for m in mois:
        for k in range(premier_client, premier_client + n_clients):
            emettre(f"C{k:05d}", prix[k] + rng.choice((0, 0, 0, 0, 300, 500, 999)), date(annee, m, 5))
    for k in range(premier_client, premier_client + n_clients):
        if rng.random() < 0.01:  # une régularisation, échéance le 20 du premier mois
            emettre(f"C{k:05d}", rng.choice((450, 1200, 2500)), date(annee, mois[0], 20))
    return creances


def generer_paiements(creances: list[Creance], graine: int = 0, inconnus: int = 15) -> list[Paiement]:
    """Le relevé bancaire qui va avec `creances`, dans le désordre, avec tout ce qu'un vrai relevé contient.

    Pour chaque client, un comportement tiré au sort : prélèvements (la plupart), prélèvement refusé (rien),
    virement qui règle tout d'un coup (avec ou sans numéros), trop-perçu, paiement partiel, numéro en minuscules
    ou avec une faute de frappe (six chiffres), numéro d'un AUTRE client ; plus `inconnus` virements anonymes.
    """
    rng = random.Random(graine)
    par_client: dict[str, list[Creance]] = {}
    for c in creances:
        par_client.setdefault(c.client, []).append(c)
    lignes: list[tuple[date, str | None, int, str]] = []
    numeros = [c.numero for c in creances]
    for client, factures in par_client.items():
        factures = sorted(factures, key=lambda c: (c.echeance, c.numero))
        total = sum(c.montant for c in factures)
        fin = max(c.echeance for c in factures) + timedelta(days=15)
        tirage = rng.random()
        if tirage < 0.80:  # prélèvement de chaque facture, à l'échéance ou un ou deux jours après
            for c in factures:
                lignes.append((c.echeance + timedelta(days=rng.choice((0, 0, 1, 2))), client, c.montant,
                               f"PRLV SEPA {c.numero} {client}"))
        elif tirage < 0.84:  # le dernier prélèvement est refusé : rien n'arrive (churn involontaire)
            for c in factures[:-1]:
                lignes.append((c.echeance, client, c.montant, f"PRLV SEPA {c.numero} {client}"))
        elif tirage < 0.86:  # tout est refusé
            pass
        elif tirage < 0.90:  # un virement règle tout, avec les numéros
            refs = " ".join(c.numero for c in factures)
            lignes.append((fin, client, total, f"VIR {client} REGLEMENT {refs}"))
        elif tirage < 0.93:  # un virement règle tout, sans numéro
            lignes.append((fin, client, total, f"VIREMENT DE {client}"))
        elif tirage < 0.95:  # trop-perçu : arrondi aux 10 € supérieurs
            lignes.append((fin, client, total + 1000 - total % 1000, f"VIR {client} SOLDE"))
        elif tirage < 0.97:  # paiement partiel du dernier mois
            for c in factures[:-1]:
                lignes.append((c.echeance, client, c.montant, f"PRLV SEPA {c.numero} {client}"))
            lignes.append((fin, client, factures[-1].montant // 2, f"VIR {client} ACOMPTE {factures[-1].numero}"))
        elif tirage < 0.98:  # numéros en minuscules
            for c in factures:
                lignes.append((c.echeance + timedelta(days=3), client, c.montant, f"vir {c.numero.lower()}"))
        elif tirage < 0.99:  # faute de frappe : un chiffre de trop, la facture n'est pas citée
            c = factures[-1]
            for d in factures[:-1]:
                lignes.append((d.echeance, client, d.montant, f"PRLV SEPA {d.numero} {client}"))
            lignes.append((fin, client, c.montant, f"VIR {client} {c.numero}7"))
        else:  # le numéro d'un autre client (copié sur la facture du voisin)
            lignes.append((fin, client, total, f"VIR {client} {rng.choice(numeros)}"))
    for i in range(inconnus):
        jour = min(c.echeance for c in creances) + timedelta(days=rng.randrange(40)) if creances else date(2026, 1, 1)
        lignes.append((jour, None, rng.choice((1000, 2999, 5000, 12000)), f"VIR INST {rng.randrange(10**6):06d}"))
    lignes.sort(key=lambda l: l[0])
    paiements = [Paiement(f"B{i:06d}", jour, client, montant, libelle)
                 for i, (jour, client, montant, libelle) in enumerate(lignes, start=1)]
    rng.shuffle(paiements)
    return paiements


def creances_exemple() -> list[Creance]:
    """Les factures d'octobre et de novembre 2026 des clients C00001 à C03000 (6 000 factures)."""
    return generer_creances(3000, graine=2026)


def paiements_exemple() -> list[Paiement]:
    """Le relevé bancaire d'octobre et de novembre 2026, dans le désordre."""
    return generer_paiements(creances_exemple(), graine=2026)

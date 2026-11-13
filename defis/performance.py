"""Outils de la série 18, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Tout est réimporté dans `defis/serie_18.py` : vous n'avez pas besoin d'ouvrir ce fichier. Vous y
trouverez de quoi ESSAYER vos fonctions en les écrivant :

    >>> from defis.serie_18 import *
    >>> point_chaud(programme_exemple)                          # défi 111
    'calculer_scores'
    >>> clients, appels, deja = exemple_rappels()               # défi 115
    >>> chronometrer(lambda: clients_a_rappeler_lent(clients, appels, deja), repetitions=1)
    2.1…                                                        # secondes : à vous de faire mieux
"""

from __future__ import annotations

import csv
import random
from datetime import date
from pathlib import Path

from defis.production import HorlogeFactice  # noqa: F401 — réexportée : l'horloge qu'on avance à la main

CONTRATS = ("Mensuel", "Annuel", "Biannuel")
MOTIFS = ("panne", "facture", "resiliation", "information")

# ============================================================================ Défi 111 : un programme à profiler


def charger_clients() -> list[int]:
    return list(range(2_000))


def nettoyer(valeur: int) -> int:
    return valeur


def calculer_scores(clients: list[int]) -> float:
    total = 0.0
    for i in range(400_000):        # la boucle lente : c'est ici que le programme passe son temps
        total += (i % 7) * 0.5
    return total + len(clients)


def programme_exemple() -> float:
    """Un petit programme dont on veut savoir OÙ il passe son temps (essayez `point_chaud` dessus)."""
    clients = [nettoyer(c) for c in charger_clients()]
    return calculer_scores(clients)


# ============================================================================ Défi 112 : historique des tarifs

#: Les changements de tarif de l'offre Fibre, dans le désordre (comme dans l'export du service facturation).
#: (date d'entrée en vigueur, prix mensuel) : le prix s'applique À PARTIR de ce jour-là, inclus.
TARIFS_EXEMPLE = [
    (date(2025, 1, 1), 34.9),
    (date(2024, 1, 1), 29.9),
    (date(2024, 7, 1), 32.9),
]

# ============================================================================ Défi 114 : un gros fichier de factures

#: Les colonnes du fichier de factures (l'ordre peut changer d'un export à l'autre : lisez-les par leur nom).
COLONNES_FACTURES = ("client_id", "contrat", "offre", "montant")


def ecrire_factures(chemin: str | Path, lignes: int = 100_000, graine: int = 0,
                    colonnes: tuple[str, ...] = COLONNES_FACTURES) -> Path:
    """Écrit un fichier de factures au format de l'export de nuit, et renvoie son chemin.

    Comme le vrai : en UTF-8, séparé par des virgules, une offre sur cinq contient une virgule (« Fibre, TV »,
    donc entre guillemets), environ une facture sur cinquante est annulée (montant vide) et le fichier finit
    par une ligne vide.

        >>> chemin = ecrire_factures("factures.csv")          # 100 000 lignes, 2,8 Mo
        >>> totaux_par_contrat(chemin)                        # défi 114
    """
    rng = random.Random(graine)
    chemin = Path(chemin)
    with chemin.open("w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.writer(fichier, lineterminator="\n")
        ecrivain.writerow(colonnes)
        for i in range(lignes):
            ligne = {
                "client_id": f"C{i % 40_000:05d}",
                "contrat": rng.choices(CONTRATS, weights=(55, 25, 20))[0],
                "offre": "Fibre, TV" if rng.random() < 0.2 else rng.choice(("Fibre", "ADSL", "Mobile")),
                "montant": "" if rng.random() < 0.02 else f"{rng.uniform(15, 130):.2f}",
            }
            ecrivain.writerow([ligne[c] for c in colonnes])
        fichier.write("\n")
    return chemin


# ============================================================================ Défi 115 : la fonction à accélérer


def clients_a_rappeler_lent(clients: list[dict], appels: list[dict], deja_contactes: list[str],
                            seuil_appels: int = 3) -> list[tuple[str, int]]:
    """La version d'origine, écrite par un collègue : JUSTE, mais 40 minutes sur le vrai fichier.

    Les clients à rappeler sont ceux qui ont appelé le support au moins `seuil_appels` fois (les appels pour
    simple « information » ne comptent pas), dont la facture est au moins égale à la facture moyenne de TOUS
    les clients, et qu'on n'a pas déjà contactés. Le résultat : des paires (client_id, nombre d'appels), du
    plus grand nombre d'appels au plus petit.

    C'est la RÉFÉRENCE : votre version rapide (défi 115) doit renvoyer exactement la même chose, dans tous
    les cas, y compris ceux auxquels personne n'a pensé.
    """
    resultat = []
    for client in clients:
        nb_appels = 0
        for appel in appels:
            if appel["client_id"] == client["client_id"] and appel["motif"] != "information":
                nb_appels += 1
        facture_moyenne = sum(c["facture"] for c in clients) / len(clients)
        if client["client_id"] in [d for d in deja_contactes]:
            continue
        if nb_appels >= seuil_appels and client["facture"] >= facture_moyenne:
            resultat.append((client["client_id"], nb_appels))
    resultat.sort(key=lambda paire: paire[1], reverse=True)
    return resultat


def exemple_rappels(n_clients: int = 2_000, graine: int = 0) -> tuple[list[dict], list[dict], list[str]]:
    """Des données pour essayer le défi 115 : (clients, appels, deja_contactes).

    10 appels par client, et un client sur dix déjà contacté. Avec 2 000 clients, la version lente met déjà
    deux secondes ; le vrai fichier en compte 400 000.
    """
    rng = random.Random(graine)
    clients = [{"client_id": f"C{i:05d}", "facture": round(rng.uniform(15, 130), 2),
                "contrat": rng.choice(CONTRATS)} for i in range(n_clients)]
    appels = [{"client_id": f"C{rng.randrange(n_clients):05d}", "motif": rng.choice(MOTIFS)}
              for _ in range(10 * n_clients)]
    deja_contactes = [c["client_id"] for c in rng.sample(clients, n_clients // 10)]
    return clients, appels, deja_contactes

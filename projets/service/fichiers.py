"""Lire la liste des contacts, mettre en forme l'avancement (DÉJÀ ÉCRIT : à lire).

Ces fonctions ne sont pas le cœur du projet, mais elles sont de bons exemples à imiter.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .modeles import STATUTS

COLONNES_CONTACTS = ("customer_id", "offre")


def lire_contacts(chemin: str | Path) -> list[tuple[str, str]]:
    """Lit la liste des clients à contacter écrite par le projet 3 (`a_contacter.csv`).

    Renvoie les couples (customer_id, offre), dans l'ordre du fichier. Ce fichier vient de NOTRE
    programme : une ligne fausse veut dire un bug en amont, on refuse donc tout le fichier
    (ValueError) au lieu d'écarter la ligne.
    """
    chemin = Path(chemin)
    with chemin.open(encoding="utf-8-sig", newline="") as fichier:
        lecteur = csv.DictReader(fichier)
        manquantes = [c for c in COLONNES_CONTACTS if c not in (lecteur.fieldnames or [])]
        if manquantes:
            raise ValueError(f"{chemin.name} : colonnes manquantes : {', '.join(manquantes)}")
        contacts: list[tuple[str, str]] = []
        deja_vus: set[str] = set()
        for ligne in lecteur:
            identifiant, offre = (ligne["customer_id"] or "").strip(), (ligne["offre"] or "").strip()
            if not identifiant or identifiant in deja_vus or not offre:
                raise ValueError(f"{chemin.name}, ligne {lecteur.line_num} : customer_id vide ou en double, "
                                 "ou offre vide")
            deja_vus.add(identifiant)
            contacts.append((identifiant, offre))
    return contacts


def formater_avancement(avancement: dict[str, dict[str, int]]) -> str:
    """Le tableau d'avancement de l'équipe, une ligne par conseiller, puis le total.

        conseiller      a_appeler      accepte       refuse  injoignable   traités
        alice                 398            4            2            1     1,7 %
        …
        TOTAL                1593           12            5            2     1,2 %
    """
    largeur = max([len("conseiller"), len("TOTAL"), *map(len, avancement)]) + 2
    lignes = [f"{'conseiller':<{largeur}}" + "".join(f"{s:>13}" for s in STATUTS) + f"{'traités':>10}"]
    totaux = dict.fromkeys(STATUTS, 0)
    for conseiller, compte in avancement.items():
        lignes.append(_ligne(conseiller, compte, largeur))
        for statut in STATUTS:
            totaux[statut] += compte.get(statut, 0)
    lignes.append(_ligne("TOTAL", totaux, largeur))
    return "\n".join(lignes)


def _ligne(nom: str, compte: dict[str, int], largeur: int) -> str:
    total = sum(compte.get(s, 0) for s in STATUTS)
    traites = total - compte.get(STATUTS[0], 0)
    part = f"{100 * traites / total:.1f} %".replace(".", ",") if total else "—"
    return f"{nom:<{largeur}}" + "".join(f"{compte.get(s, 0):>13}" for s in STATUTS) + f"{part:>10}"

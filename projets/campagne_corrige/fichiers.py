"""Lire le plan, écrire la liste des contacts, mettre en forme le bilan (DÉJÀ ÉCRIT : à lire).

Ces fonctions ne sont pas le cœur du projet, mais elles sont de bons exemples à imiter : lecture
stricte d'un fichier produit par un autre programme, requête paramétrée, nombres au format français.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from .modeles import COLONNES_PLAN, GROUPE_OFFRE, Ciblage, Comptage, Effet


def lire_plan(chemin: str | Path) -> list[Ciblage]:
    """Lit le plan de rétention écrit par le projet 2 (`plan_retention.csv`).

    Contrairement au fichier du CRM (étape 3), ce fichier vient de NOTRE programme : une ligne fausse
    veut dire un bug en amont. On refuse donc tout le fichier (ValueError) au lieu d'écarter la ligne.
    """
    chemin = Path(chemin)
    with chemin.open(encoding="utf-8-sig", newline="") as fichier:
        lecteur = csv.DictReader(fichier)
        manquantes = [c for c in COLONNES_PLAN if c not in (lecteur.fieldnames or [])]
        if manquantes:
            raise ValueError(f"{chemin.name} : colonnes manquantes : {', '.join(manquantes)}")
        ciblages: list[Ciblage] = []
        deja_vus: set[str] = set()
        for ligne in lecteur:
            ou = f"{chemin.name}, ligne {lecteur.line_num}"
            identifiant = (ligne["customer_id"] or "").strip()
            if not identifiant or identifiant in deja_vus:
                raise ValueError(f"{ou} : customer_id vide ou en double ({identifiant!r})")
            try:
                cout, proba = float(ligne["cout"]), float(ligne["proba_churn"])
            except (TypeError, ValueError):
                raise ValueError(f"{ou} : cout ou proba_churn illisible") from None
            if not (cout >= 0 and 0 <= proba <= 1):
                raise ValueError(f"{ou} : cout négatif ou proba_churn hors de [0, 1]")
            deja_vus.add(identifiant)
            ciblages.append(Ciblage(identifiant, (ligne["offre"] or "").strip(), cout, proba))
    return ciblages


def ecrire_contacts(conn: sqlite3.Connection, chemin: str | Path) -> int:
    """Écrit les clients du groupe OFFRE (les seuls à contacter) et renvoie leur nombre.

    Les clients du groupe témoin n'apparaissent pas : s'ils recevaient l'offre « par erreur »,
    la mesure de l'étape 5 serait faussée.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    # Requête paramétrée : la valeur voyage à part (le « ? »), jamais collée dans le texte SQL.
    lignes = conn.execute(
        "SELECT customer_id, offre FROM campagne WHERE groupe = ? ORDER BY customer_id", (GROUPE_OFFRE,)
    ).fetchall()
    with chemin.open("w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.writer(fichier)
        ecrivain.writerow(("customer_id", "offre"))
        ecrivain.writerows(lignes)
    return len(lignes)


# ---------------------------------------------------------------------------- Nombres en français


def entier(n: int) -> str:
    """12345 → « 12 345 »."""
    return f"{n:,}".replace(",", " ")


def euros(montant: float) -> str:
    """1234.5 → « 1 234,50 € »."""
    return f"{montant:,.2f}".replace(",", " ").replace(".", ",") + " €"


def pourcent(part: float) -> str:
    """0.2314 → « 23,1 % »."""
    return f"{100 * part:.1f} %".replace(".", ",")


def points(ecart: float) -> str:
    """0.158 → « 15,8 points » : un écart entre deux pourcentages se mesure en points."""
    valeur = round(100 * ecart, 1)
    return f"{valeur:.1f} point{'s' if abs(valeur) >= 2 else ''}".replace(".", ",")


def _sans_unite(ecart: float) -> str:
    return f"{100 * ecart:.1f}".replace(".", ",")


def formater_ligne(nom: str, offre: Comptage, temoin: Comptage, effet: Effet | None) -> str:
    """Le bloc de texte du bilan pour une offre (ou pour TOUTES). `effet` vaut None si on n'a pas pu
    le calculer (aucun résultat connu dans l'un des deux groupes)."""
    lignes = [
        f"[{nom}] {entier(offre.clients)} clients avec l'offre, {entier(temoin.clients)} témoins "
        f"(résultats connus : {entier(offre.avec_resultat)} et {entier(temoin.avec_resultat)})"
    ]
    if effet is None:
        lignes.append("  ⚠️  effet impossible à mesurer : il faut des résultats dans les deux groupes")
        return "\n".join(lignes)

    niveau = f"{100 * effet.niveau:g}".replace(".", ",")
    lignes.append(
        f"  départs : {pourcent(effet.taux_offre)} avec l'offre, {pourcent(effet.taux_temoin)} sans "
        f"(écart : {points(effet.ecart)} ; IC {niveau} % : de {_sans_unite(effet.ic_bas)} à {_sans_unite(effet.ic_haut)})"
    )
    if effet.clients_sauves > 0:
        lignes.append(f"  départs évités : ≈ {effet.clients_sauves:.0f}, pour {euros(offre.cout)} d'offres "
                      f"({euros(offre.cout / effet.clients_sauves)} par départ évité)")
    else:
        lignes.append(f"  départs évités : aucun, pour {euros(offre.cout)} d'offres")
    if effet.ic_bas > 0:
        lignes.append("  ✅ l'offre réduit les départs : l'écart ne s'explique pas par le hasard")
    elif effet.ic_haut < 0:
        lignes.append("  ❌ l'offre AUGMENTE les départs : à arrêter")
    else:
        lignes.append("  🤷 impossible de conclure : l'écart peut venir du hasard (il faudrait plus de clients)")
    return "\n".join(lignes)

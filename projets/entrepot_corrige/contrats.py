"""Étape 1 (corrigé) — Lire l'export du CRM sans s'arrêter à la première ligne abîmée."""

from __future__ import annotations

import csv
import math
import re
from datetime import date, datetime
from pathlib import Path

from .modeles import COLONNES_CRM, FORMAT_CLIENT, MODALITES, Contrat, Lecture, Rejet


def _jour(texte: str, colonne: str) -> date:
    try:
        return datetime.strptime(texte, "%d/%m/%Y").date()
    except ValueError:
        raise ValueError(f"{colonne} : date illisible {texte!r}, attendu JJ/MM/AAAA") from None


def _contrat(valeurs: dict[str, str]) -> Contrat:
    """Une ligne de texte → un Contrat. Lève ValueError, avec la colonne fautive, si la ligne est abîmée."""
    client = valeurs["client"]
    if not re.fullmatch(FORMAT_CLIENT, client):
        raise ValueError(f"client : identifiant illisible {client!r}")

    debut = _jour(valeurs["debut"], "debut")
    fin = _jour(valeurs["fin"], "fin") if valeurs["fin"] else None
    if fin is not None and fin < debut:
        raise ValueError(f"fin : {fin.isoformat()} est avant le début ({debut.isoformat()})")

    for colonne, permises in MODALITES.items():
        if valeurs[colonne] not in permises:
            raise ValueError(f"{colonne} : valeur inconnue {valeurs[colonne]!r}")

    if not re.fullmatch(r"[0-9]+", valeurs["produits"]) or int(valeurs["produits"]) < 1:
        raise ValueError(f"produits : {valeurs['produits']!r} n'est pas un entier >= 1")

    try:
        # « 79,90 » écrit par Excel ; float accepte aussi « nan » et « inf », qu'on refuse ensuite.
        mensualite = float(valeurs["mensualite"].replace(",", "."))
    except ValueError:
        raise ValueError(f"mensualite : nombre illisible {valeurs['mensualite']!r}") from None
    if not math.isfinite(mensualite) or mensualite <= 0:
        raise ValueError(f"mensualite : {valeurs['mensualite']!r} n'est pas un montant strictement positif")

    try:
        maj = datetime.fromisoformat(valeurs["maj"])
    except ValueError:
        raise ValueError(f"maj : date et heure illisibles {valeurs['maj']!r}") from None

    return Contrat(client=client, debut=debut, fin=fin, contrat=valeurs["contrat"], paiement=valeurs["paiement"],
                   internet=valeurs["internet"], assistance=valeurs["assistance"], senior=valeurs["senior"],
                   produits=int(valeurs["produits"]), mensualite=mensualite, maj=maj)


def lire_contrats(chemin: str | Path) -> Lecture:
    chemin = Path(chemin)
    lignes: list[Contrat] = []
    rejets: list[Rejet] = []
    # utf-8-sig retire la BOM ; newline="" laisse le module csv gérer les fins de ligne (\n ou \r\n).
    with chemin.open(encoding="utf-8-sig", newline="") as f:
        lecteur = csv.reader(f, delimiter=";")
        entete = [c.strip() for c in next(lecteur, [])]
        if tuple(entete) != COLONNES_CRM:
            raise ValueError(f"{chemin.name} : en-tête inattendu {entete!r}, attendu {list(COLONNES_CRM)!r}")
        for champs in lecteur:
            numero = lecteur.line_num  # le numéro de la ligne dans le fichier (l'en-tête est la ligne 1)
            if not any(c.strip() for c in champs):
                continue  # ligne vide (Excel en ajoute souvent une à la fin) : ni contrat, ni rejet
            if len(champs) != len(COLONNES_CRM):
                rejets.append(Rejet(chemin.name, numero, f"{len(champs)} champs au lieu de {len(COLONNES_CRM)}"))
                continue
            valeurs = {colonne: champ.strip() for colonne, champ in zip(COLONNES_CRM, champs)}
            try:
                lignes.append(_contrat(valeurs))
            except ValueError as exc:
                rejets.append(Rejet(chemin.name, numero, str(exc)))
    return Lecture(lignes=lignes, rejets=rejets)

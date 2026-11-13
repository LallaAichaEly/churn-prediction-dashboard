"""Étape 4 (corrigé) — Compter, en une requête SQL, ce que la base sait de chaque groupe."""

from __future__ import annotations

import sqlite3
from typing import Iterable

from .modeles import GROUPES, TOUTES, Comptage

# LEFT JOIN garde les clients SANS résultat (un JOIN simple les ferait disparaître).
# COUNT(*) compte les lignes, donc tous les clients ; COUNT(r.customer_id) ne compte que les
# valeurs non NULL, donc ceux qui ont un résultat. SUM d'une colonne où tout est NULL vaut NULL,
# d'où COALESCE(…, 0). Pas de risque de doublons : chaque client a au plus un résultat (clé primaire).
_REQUETE = """
SELECT c.offre,
       c.groupe,
       COUNT(*)                  AS clients,
       COUNT(r.customer_id)      AS avec_resultat,
       COALESCE(SUM(r.parti), 0) AS partis,
       SUM(c.cout)               AS cout
FROM campagne AS c
LEFT JOIN resultats AS r ON r.customer_id = c.customer_id
GROUP BY c.offre, c.groupe
ORDER BY c.offre, c.groupe
"""


def compter(conn: sqlite3.Connection) -> list[Comptage]:
    # Une seule requête : SQLite fait les comptes là où sont les données, au lieu de rapatrier
    # chaque client dans Python (ou, pire, de lancer une requête par offre).
    return [Comptage(offre, groupe, clients, avec, partis, float(cout))
            for offre, groupe, clients, avec, partis, cout in conn.execute(_REQUETE)]


def _somme(offre: str, groupe: str, comptages: Iterable[Comptage]) -> Comptage:
    choisis = [c for c in comptages if c.groupe == groupe]
    return Comptage(offre, groupe,
                    clients=sum(c.clients for c in choisis),
                    avec_resultat=sum(c.avec_resultat for c in choisis),
                    partis=sum(c.partis for c in choisis),
                    cout=float(sum(c.cout for c in choisis)))


def regrouper(comptages: Iterable[Comptage]) -> dict[str, tuple[Comptage, Comptage]]:
    comptages = list(comptages)  # on le parcourt plusieurs fois : un générateur serait vide au 2e tour
    par_cle = {(c.offre, c.groupe): c for c in comptages}
    # Un dict garde l'ordre d'insertion : TOUTES d'abord, puis les offres dans l'ordre alphabétique.
    bilan = {TOUTES: tuple(_somme(TOUTES, g, comptages) for g in GROUPES)}
    for code in sorted({c.offre for c in comptages}):
        # Une offre sans aucun témoin (petite offre, pas de chance au tirage) : un Comptage à zéro,
        # pour que l'appelant n'ait pas à gérer de cas particulier.
        bilan[code] = tuple(par_cle.get((code, g), Comptage(code, g, 0, 0, 0, 0.0)) for g in GROUPES)
    return bilan

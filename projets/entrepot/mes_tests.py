"""Étape 7 🔴 — Écrire le test qui protège `construire_table` (étape 5).

Vérifier : python scripts/projet.py --projet 6 --etape 7
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

from .modeles import Contrat, Ticket, ouvrir_base


def _contrat(client: str, debut: str, fin: str | None = None, mensualite: float = 50.0,
             maj: str = "2026-12-31T00:00:00") -> Contrat:
    """Fabrique un contrat de test (DÉJÀ ÉCRIT) : seuls le client, les dates et la mensualité comptent.

    Exemple : `_contrat("C00001", "2024-01-15", fin="2026-08-01")`. Par défaut, `maj` est le
    31/12/2026 : la base « connaît » donc les départs jusqu'à cette date.
    """
    return Contrat(client=client, debut=date.fromisoformat(debut), fin=date.fromisoformat(fin) if fin else None,
                   contrat="Mensuel", paiement="Prélèvement", internet="Fibre", assistance="Non", senior="Non",
                   produits=1, mensualite=mensualite, maj=datetime.fromisoformat(maj))


def _ticket(ticket: str, client: str, ouvert: str) -> Ticket:
    """Fabrique un ticket de test (DÉJÀ ÉCRIT) : `_ticket("T000001", "C00001", "2026-06-30T15:00:00")`."""
    instant = datetime.fromisoformat(ouvert)
    return Ticket(ticket=ticket, client=client, ouvert=instant, maj=instant)


def _base(contrats: list[Contrat], tickets: list[Ticket] = ()) -> sqlite3.Connection:
    """Une base EN MÉMOIRE qui contient exactement ces contrats et ces tickets (DÉJÀ ÉCRIT).

    Elle n'utilise pas vos étapes 3 et 4 : un bug là-bas ne fausse pas ce test-ci.
    """
    connexion = ouvrir_base(":memory:")
    connexion.executemany("INSERT INTO contrats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
        (c.client, c.debut.isoformat(), c.fin.isoformat() if c.fin else None, c.contrat, c.paiement, c.internet,
         c.assistance, c.senior, c.produits, c.mensualite, c.maj.isoformat(timespec="seconds")) for c in contrats])
    connexion.executemany("INSERT INTO tickets VALUES (?, ?, ?, ?)", [
        (t.ticket, t.client, t.ouvert.isoformat(timespec="seconds"), t.maj.isoformat(timespec="seconds"))
        for t in tickets])
    connexion.commit()
    return connexion


def tester_construire_table(construire_table) -> None:
    """Vérifie, avec des `assert`, qu'une fonction `construire_table(connexion, photo, horizon=90)`
    respecte le contrat de l'étape 5 (relisez sa docstring dans `table.py`).

    `construire_table` fabrique les données sur lesquelles le modèle apprend. Un bug ici ne fait
    rien planter : il fausse le modèle EN SILENCE. Un client oublié, un ticket du futur compté, un
    client déjà parti gardé… et l'AUC change, sans que personne ne sache pourquoi. C'est exactement
    le genre de code qu'on protège par un test.

    Comme aux projets précédents, c'est VOUS qui écrivez le test, et le correcteur l'exécute :
    - sur la version JUSTE de `construire_table` : votre test doit passer ;
    - sur 12 versions BOGUÉES, toutes plausibles : votre test doit échouer sur CHACUNE.

    Construisez une petite base avec `_base([_contrat(...), ...], [_ticket(...), ...])`, choisissez
    une photo (par exemple `date(2026, 6, 30)`), et pour chaque règle placez un client ou un ticket
    PILE sur la frontière : un ticket ouvert le jour de la photo l'après-midi, un autre le lendemain,
    un autre exactement 365 jours avant ; un client parti le jour de la photo, un autre arrivé le
    lendemain ; un départ le dernier jour de l'horizon, un autre le jour d'après ; un client sans
    aucun ticket ; un client arrivé dix jours avant la photo… Calculez à la main ce qu'il faut
    trouver, puis écrivez les `assert`. Pour vérifier qu'une erreur est levée :

        try:
            construire_table(base, date(2026, 10, 3))
        except ValueError:
            pass
        else:
            raise AssertionError("construire_table aurait dû refuser une photo trop récente")

    Ce que vous apprenez : tester des requêtes SQL sur une base de test en mémoire, viser les
    frontières (le jour même, le lendemain, 365 jours pile), calculer l'attendu à la main AVANT.
    """
    raise NotImplementedError

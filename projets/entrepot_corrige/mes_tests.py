"""Étape 7 (corrigé) — Le test qui protège `construire_table` (étape 5)."""

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
    photo = date(2026, 6, 30)  # horizon de 90 jours : jusqu'au 28/09/2026 compris
    # Un ticket « frontière » par client : si deux erreurs se compensaient sur le même client (un
    # ticket oublié d'un côté, un compté en trop de l'autre), le total resterait juste.
    base = _base(
        [
            _contrat("C00001", "2024-01-15"),                          # aucun ticket : reste, avec 0
            _contrat("C00002", "2025-12-30", mensualite=40.0),          # un ticket le jour même, l'après-midi
            _contrat("C00003", "2020-01-01", fin="2026-06-30"),         # parti le jour de la photo : exclu
            _contrat("C00004", "2026-07-01"),                           # arrivé le lendemain : exclu
            _contrat("C00005", "2023-05-10", fin="2026-09-28"),         # part le dernier jour de l'horizon
            _contrat("C00006", "2023-05-10", fin="2026-09-29"),         # part le jour d'après : resté
            _contrat("C00007", "2026-06-20"),                           # 10 jours d'ancienneté : 1 mois
            _contrat("C00008", "2019-03-01", fin="2025-11-02"),         # parti depuis longtemps : exclu
            _contrat("C00009", "2022-02-10"),                           # un ticket à 364 jours, un vieux
        ],
        [
            _ticket("T000001", "C00002", "2026-06-30T15:00:00"),  # le jour même, l'après-midi : compte
            _ticket("T000002", "C00005", "2026-07-01T09:00:00"),  # le lendemain : le futur, ne compte pas
            _ticket("T000003", "C00006", "2025-06-30T10:00:00"),  # 365 jours avant : hors de la fenêtre
            _ticket("T000004", "C00009", "2025-07-01T10:00:00"),  # 364 jours avant : compte
            _ticket("T000005", "C00009", "2024-02-01T10:00:00"),  # bien trop ancien
            _ticket("T000006", "C99999", "2026-06-01T10:00:00"),  # client inconnu du CRM : ignoré
        ],
    )

    table = construire_table(base, photo)
    assert list(table.columns) == ["customer_id", "tenure_months", "monthly_charges", "total_charges",
                                   "support_calls", "num_products", "contract", "payment_method",
                                   "internet_service", "tech_support", "senior", "churn"]
    assert list(table["customer_id"]) == ["C00001", "C00002", "C00005", "C00006", "C00007", "C00009"]
    lignes = table.set_index("customer_id")
    assert list(lignes["support_calls"]) == [0, 1, 0, 0, 0, 1]
    assert list(lignes["churn"]) == [0, 0, 1, 0, 0, 0]
    assert lignes.loc["C00002", "tenure_months"] == 6       # du 30/12/2025 au 30/06/2026
    assert lignes.loc["C00002", "total_charges"] == 240.0   # 40 € × 6 mois
    assert lignes.loc["C00007", "tenure_months"] == 1       # 0 mois complet, mais au moins 1

    construire_table(base, date(2026, 10, 2))  # 02/10 + 90 jours = 31/12 : la base le connaît
    try:
        construire_table(base, date(2026, 10, 3))  # 03/10 + 90 jours = 01/01/2027 : pas encore connu
    except ValueError:
        pass
    else:
        raise AssertionError("construire_table aurait dû refuser une photo trop récente")

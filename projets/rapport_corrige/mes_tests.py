"""Étape 7 (corrigé) — Le test qui protège `indicateurs_du_mois`."""

from __future__ import annotations

from datetime import date

from .modeles import Abonne


def _abonne(client: str, debut: str, fin: str | None = None, mensualite: float = 50.0,
            segment: str = "Particulier") -> Abonne:
    """Fabrique un abonné de test (DÉJÀ ÉCRIT) : `_abonne("A", "2026-01-10", fin="2026-10-01", mensualite=30.0)`."""
    return Abonne(client=client, nom=f"Client {client}", segment=segment, debut=date.fromisoformat(debut),
                  fin=date.fromisoformat(fin) if fin else None, mensualite=mensualite)


def tester_indicateurs_du_mois(indicateurs_du_mois) -> None:
    octobre = date(2026, 10, 1)
    abonnes = [
        _abonne("A", "2026-01-10"),                           # ancien, toujours là
        _abonne("B", "2026-02-01", "2026-10-01", 30.0),       # parti LE 1er : abonné au début, départ d'octobre
        _abonne("C", "2026-03-05", "2026-10-31", 40.0),       # parti LE DERNIER JOUR : départ d'octobre
        _abonne("D", "2026-10-01"),                           # arrivé LE 1er : pas abonné « au début »
        _abonne("E", "2026-10-03", "2026-10-20", 25.0),       # arrivé ET parti en octobre
        _abonne("F", "2025-06-01", "2026-09-30", 99.0),       # parti LA VEILLE : rien en octobre
        _abonne("G", "2026-11-01"),                           # arrivé le lendemain du mois
        _abonne("H", "2026-04-01", "2026-11-01"),             # part en novembre : là au début et à la fin
    ]
    copie = list(abonnes)
    i = indicateurs_du_mois(abonnes, octobre)
    assert i.mois == octobre
    assert i.actifs_debut == 4, i            # A, B, C, H
    assert i.arrivees == 2, i                # D, E
    assert i.departs == 3, i                 # B, C, E
    assert i.actifs_fin == 3, i              # A, D, H
    assert i.taux_churn == 0.5, i            # B et C sur A, B, C, H (E n'était pas là au début)
    assert abs(i.revenu_perdu - 95.0) < 1e-9, i  # 30 + 40 + 25 : TOUS les départs
    assert abonnes == copie, "la liste reçue a été modifiée"

    # N'importe quel jour du mois désigne le mois entier.
    assert indicateurs_du_mois(abonnes, date(2026, 10, 15)) == i

    # Personne au début du mois : pas de taux (None), et surtout pas 0 %, ni une division par zéro.
    vide = indicateurs_du_mois([_abonne("N", "2026-10-05")], octobre)
    assert vide.actifs_debut == 0 and vide.taux_churn is None, vide

    # Décembre : le mois d'après est en janvier de l'année suivante.
    decembre = indicateurs_du_mois([_abonne("X", "2026-01-01", "2026-12-31", 20.0), _abonne("Y", "2026-12-31")],
                                   date(2026, 12, 1))
    assert (decembre.actifs_debut, decembre.arrivees, decembre.departs, decembre.actifs_fin) == (1, 1, 1, 1), decembre
    assert decembre.taux_churn == 1.0, decembre

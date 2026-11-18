"""Outils communs de la série 33 : les départs de chaque mois, un générateur, et les quatre résultats nommés.

Mariam dirige le centre d'appels de rétention : chaque client qui demande à résilier y est rappelé par un
conseiller. Un conseiller traite `DOSSIERS_PAR_CONSEILLER` dossiers par mois, et il faut TROIS mois pour en
recruter et en former un. Elle doit donc savoir, en décembre, combien de clients demanderont à partir en mars.
Les données :

- `departs_exemple()` : le nombre de demandes de résiliation de chaque mois, de janvier 2023 à décembre 2025
  (36 mois). On y voit une TENDANCE (un peu plus de départs chaque année) et une SAISON (des pics en janvier et
  en septembre, la fin des engagements et la rentrée ; un creux l'été) ;
- `mois_exemple()` : les 36 mois correspondants, « 2023-01 » à « 2025-12 » ;
- `Erreurs` (défi 201), `Lissage` (défis 202 et 203), `Essai` (défi 204) et `Marge` (défi 205) : les résultats,
  des tuples nommés ;
- `generer_serie(n, graine, …)` : de longues séries reproductibles (tendance, saison, bruit) pour vos propres
  essais. Ses valeurs sont des multiples de 1/16 : leurs sommes et leurs différences sont EXACTES en virgule
  flottante.

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> departs = departs_exemple()
    >>> len(departs), departs[:3]                 # janvier, février, mars 2023
    (36, [463.0, 403.0, 391.0])
    >>> mois_exemple()[8], departs[8]             # le pic de la rentrée
    ('2023-09', 519.0)
    >>> sum(departs[:12]), sum(departs[24:])      # d'une année sur l'autre : la tendance
    (5036.0, 5855.0)
"""

from __future__ import annotations

import random
from typing import NamedTuple

DOSSIERS_PAR_CONSEILLER = 40      # dossiers de résiliation qu'un conseiller traite en un mois


class Erreurs(NamedTuple):
    """Les erreurs d'une série de prévisions (défi 201). Un tuple : (mae, rmse, biais, mape)."""

    mae: float                    # erreur absolue moyenne, dans l'unité des données (des départs)
    rmse: float                   # racine de l'erreur quadratique moyenne : punit les grosses erreurs
    biais: float                  # moyenne de (prévu − réel) : > 0, on prévoit trop ; < 0, pas assez
    mape: float | None            # moyenne de |prévu − réel| / |réel| (0.05 = 5 %) ; None si un réel vaut 0


class Lissage(NamedTuple):
    """Le résultat d'un lissage (défis 202 et 203). Un tuple : (ajustees, previsions, niveau, tendance)."""

    ajustees: list[float | None]  # ajustees[t] : la prévision de historique[t] faite AVANT de le voir (ou None)
    previsions: list[float]       # les `horizon` mois qui suivent l'historique
    niveau: float                 # le dernier niveau lissé
    tendance: float               # la dernière tendance lissée (0.0 pour le lissage simple)


class Essai(NamedTuple):
    """Une prévision rejouée dans le passé (défi 204)."""

    origine: int                  # le nombre de mois connus au moment de prévoir : serie[:origine]
    cible: int                    # le numéro du mois prévu : origine + horizon − 1
    prevu: float                  # ce que la méthode avait prévu pour ce mois-là
    reel: float                   # ce qui s'est vraiment passé : serie[cible]


class Marge(NamedTuple):
    """Une prévision et sa marge de sécurité (défi 205)."""

    prevision: float              # la prévision du mois visé, faite avec TOUTE la série
    marge: float                  # ce qu'on ajoute : un quantile des erreurs passées (réel − prévu)
    borne: float                  # prevision + marge : le nombre de départs pour lequel on s'équipe


#                 jan  fév  mar  avr  mai  juin juil août sep  oct  nov  déc
_DEPARTS = [463, 403, 391, 371, 393, 406, 371, 371, 519, 470, 443, 435,   # 2023
            510, 450, 412, 420, 431, 435, 429, 408, 545, 508, 472, 470,   # 2024
            532, 484, 452, 466, 460, 472, 437, 437, 584, 523, 503, 505]   # 2025


def departs_exemple() -> list[float]:
    """Les demandes de résiliation de chaque mois, de janvier 2023 à décembre 2025 (36 nombres)."""
    return [float(x) for x in _DEPARTS]


def mois_exemple() -> list[str]:
    """Les 36 mois de `departs_exemple()`, au format « AAAA-MM »."""
    return [f"{annee}-{mois:02d}" for annee in (2023, 2024, 2025) for mois in range(1, 13)]


def generer_serie(n: int = 120, graine: int = 0, saison: int = 12, tendance: float = 1.0,
                  amplitude: float = 40.0, bruit: float = 10.0, depart: float = 400.0) -> list[float]:
    """`n` valeurs mensuelles : départ + tendance × t + un motif de `saison` mois + du bruit (reproductible).

    Toutes les valeurs sont des multiples de 1/16 : sommes et différences exactes.

        >>> serie = generer_serie(24, graine=1)
        >>> len(serie), all(x * 16 == int(x * 16) for x in serie)
        (24, True)
        >>> generer_serie(5, graine=2, saison=1, tendance=2.0, bruit=0.0)     # sans saison ni bruit : une droite
        [400.0, 402.0, 404.0, 406.0, 408.0]
    """
    rng = random.Random(graine)
    motif = [rng.uniform(-amplitude, amplitude) for _ in range(max(saison, 1))] if saison > 1 else [0.0]
    moyenne = sum(motif) / len(motif)
    motif = [m - moyenne for m in motif]                       # une saison ne monte ni ne descend en moyenne
    serie = []
    for t in range(n):
        x = depart + tendance * t + motif[t % len(motif)] + rng.uniform(-bruit, bruit)
        serie.append(round(x * 16) / 16)
    return serie

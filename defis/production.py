"""Outils de la série 14, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Ils sont dans un fichier à part pour une raison précise : quand le correcteur lève une
`ErreurTemporaire`, votre `except TRANSITOIRES` doit reconnaître EXACTEMENT la même classe. Deux
classes écrites à l'identique dans deux fichiers sont deux classes différentes pour Python.

Tout est réimporté dans `defis/serie_14.py` : vous n'avez pas besoin d'ouvrir ce fichier.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
VRAIS = ("1", "true", "oui", "yes", "on")
FAUX = ("0", "false", "non", "no", "off")


@dataclass(frozen=True)
class Config:
    """La configuration du traitement de nuit et de l'API (défi 86).

    `frozen=True` : on ne peut plus la modifier une fois créée (personne ne change le seuil en
    cours de route). `repr=False` : `print(config)` n'affiche pas la clé d'API (série 13, défi 81).
    """

    cle_api: str = field(repr=False)
    seuil: float = 0.5
    taille_lot: int = 500
    mode_test: bool = False


class ErreurTemporaire(Exception):
    """Le service n'a pas pu répondre MAINTENANT (surcharge, erreur 503, coupure) : réessayer a du sens.

    `attente` : le nombre de secondes que le service demande d'attendre avant de réessayer (l'en-tête
    HTTP `Retry-After`), ou None s'il ne dit rien.
    """

    def __init__(self, message: str = "service momentanément indisponible", attente: float | None = None) -> None:
        super().__init__(message)
        self.attente = attente


class CircuitOuvert(ErreurTemporaire):
    """Levée par le disjoncteur (défi 91) quand il refuse un appel sans même essayer.

    C'est une `ErreurTemporaire` : le service sera peut-être revenu dans quelques secondes.
    """


# Les erreurs « passagères » : réessayer plus tard peut marcher. Une ValueError, une KeyError ou une
# réponse « numéro de téléphone invalide » ne disparaîtront pas en réessayant : elles n'y sont pas.
TRANSITOIRES = (ErreurTemporaire, ConnectionError, TimeoutError)


class HorlogeFactice:
    """Une horloge que VOUS avancez : `horloge()` donne l'heure, `horloge.avancer(30)` fait passer 30 s.

        >>> horloge = HorlogeFactice()
        >>> horloge()
        1000.0
        >>> horloge.avancer(30)
        >>> horloge()
        1030.0

    C'est l'équivalent de `time.monotonic`, mais on n'attend pas 30 vraies secondes pour tester.
    """

    def __init__(self, depart: float = 1000.0) -> None:
        self.maintenant = float(depart)

    def __call__(self) -> float:
        return self.maintenant

    def avancer(self, secondes: float) -> None:
        self.maintenant += secondes


def service_capricieux(echecs: int = 2, resultat: str = "SMS envoyé", erreur: type[Exception] = ErreurTemporaire):
    """Un faux service d'envoi de SMS qui échoue `echecs` fois, puis répond `resultat` (défi 88).

        >>> appel = service_capricieux(echecs=2)
        >>> appeler_avec_reprises(appel, dormir=print)       # affiche les attentes au lieu d'attendre
        0.5
        1.0
        'SMS envoyé'
    """
    restants = [echecs]

    def appel():
        if restants[0] > 0:
            restants[0] -= 1
            raise erreur("opérateur SMS surchargé")
        return resultat

    return appel


def exemple_lot() -> list[dict]:
    """Six clients du lot de nuit (défi 87) : deux factures invalides et un client qui fait planter le modèle."""
    return [
        {"id": "C001", "email": "lea.martin@exemple.fr", "facture_mensuelle": 89.9, "anciennete_mois": 5},
        {"id": "C002", "email": "paul.dalmeida@exemple.fr", "facture_mensuelle": None, "anciennete_mois": 30},
        {"id": "C003", "email": "zoe.zebulon@exemple.fr", "facture_mensuelle": 0.0, "anciennete_mois": 12},
        {"id": "C004", "email": "ines.durand@exemple.fr", "facture_mensuelle": -12.5, "anciennete_mois": 2},
        {"id": "C005", "email": "noah.petit@exemple.fr", "facture_mensuelle": 45.0, "anciennete_mois": 0},
        {"id": "C006", "email": "emma.roux@exemple.fr", "facture_mensuelle": 60.0, "anciennete_mois": 48},
    ]


def score_exemple(client: dict) -> float:
    """Un faux modèle (défi 87). Il plante sur une ancienneté de 0 mois, comme un vrai bogue."""
    return round(min(1.0, client["facture_mensuelle"] / 100 / (client["anciennete_mois"] / 12)), 3)


def journal_console(nom: str = "lot_de_nuit") -> logging.Logger:
    """Un journal qui affiche tout dans le terminal, pour voir vos messages en essayant le défi 87.

    C'est le travail de l'APPLICATION (le script lancé chaque nuit) de décider où vont les messages
    et à partir de quel niveau. Votre fonction, elle, se contente d'écrire dans le journal reçu.
    """
    journal = logging.getLogger(nom)
    if not journal.handlers:
        sortie = logging.StreamHandler(sys.stdout)
        sortie.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s", "%H:%M:%S"))
        journal.addHandler(sortie)
        journal.setLevel(logging.DEBUG)
        journal.propagate = False
    return journal

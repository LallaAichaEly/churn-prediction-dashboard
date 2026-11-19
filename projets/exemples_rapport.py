"""Le fichier d'exemple du projet 7 : `projets/exemple_abonnes.csv` vaut `generer_abonnes(...)`.

Le générateur simule près de trois ans d'abonnements (janvier 2024 à octobre 2026) chez un opérateur :
un stock de clients anciens, de nouveaux clients chaque mois, et des départs. Les clients des trois
premiers mois partent plus souvent ; les professionnels (segment « Pro & TPE ») partent beaucoup plus à
partir d'octobre 2026, après la hausse de leurs tarifs au 1er octobre. C'est ce que le rapport
d'octobre doit faire voir à la direction.

Les noms sont saisis en boutique : on y trouve des « & », des apostrophes, des guillemets, des
chevrons, et un compte de test au nom de `<script>…</script>`, tapé par un stagiaire pour « essayer »,
qui résilie en octobre avec l'une des plus grosses mensualités.

Le hasard vient du module `random` de la bibliothèque standard, dont les tirages ne changent pas d'une
version de Python à l'autre, pour que le fichier soit reproductible octet par octet :

    python -m projets.exemples_rapport      # réécrit projets/exemple_abonnes.csv
"""

from __future__ import annotations

import calendar
import csv
import random
from datetime import date
from pathlib import Path

CHEMIN_EXEMPLE = Path(__file__).resolve().parent / "exemple_abonnes.csv"

#: Le dernier jour connu du fichier : l'export est fait début novembre 2026.
DERNIER_JOUR = date(2026, 10, 31)
PREMIER_MOIS = date(2024, 1, 1)
HAUSSE_PRO = date(2026, 10, 1)

#: (segment, part des nouveaux clients, risque de départ par mois, mensualités possibles)
_SEGMENTS = (
    ("Particulier", 0.60, 0.021, (19.99, 24.99, 29.99, 34.99, 39.99, 49.99)),
    ("Pro & TPE", 0.25, 0.015, (59.0, 79.0, 99.0, 129.0, 179.0, 249.0)),
    ("Senior", 0.15, 0.012, (14.99, 19.99, 24.99)),
)
_RISQUE_DEBUTANT = 0.02    # en plus, pendant les trois premiers mois
_RISQUE_PRO_APRES_HAUSSE = 0.065

_PRENOMS = ("Zoé", "Hélène", "Jérôme", "Anaïs", "Noël", "Loïc", "Inès", "Amina", "Yanis", "Chloé", "Léa",
            "Mathéo", "Gaëlle", "Françoise", "Karim", "Thibault", "Maëlys", "Joël", "Sophie", "Lucas")
_NOMS = ("Martin", "Bernard", "Dubois", "Lefèvre", "N'Diaye", "O'Connor", "Le Goff", "Müller", "Moreau",
         "Fontaine", "Benali", "Nguyen", "Da Silva", "Perrin", "Lemaître", "Chevalier", "Roux", "Gauthier")
_ENTREPRISES = ("{nom} & Fils", "Boulangerie {nom}", "Garage {nom}", "SARL {NOM}", "Cabinet « {nom} »",
                "L'Atelier de {prenom}", "Café \"Chez {prenom}\"", "{nom} Plomberie", "Pharmacie {nom}")

#: Les comptes qu'on retrouvera dans le rapport d'octobre 2026, parmi les plus gros départs.
_SAISIES_PIEGES = (
    ('<script>alert("coucou")</script>', "Pro & TPE", date(2025, 2, 3), date(2026, 10, 14), 249.0),
    ("Bio & Co <Lyon>", "Pro & TPE", date(2023, 6, 12), date(2026, 10, 2), 249.0),
)


def _jour_au_hasard(rng: random.Random, mois: date, apres: date | None = None) -> date:
    premier = 1 if apres is None or apres < mois else apres.day
    return mois.replace(day=rng.randint(premier, calendar.monthrange(mois.year, mois.month)[1]))


def _mois_suivant(mois: date) -> date:
    return date(mois.year + mois.month // 12, mois.month % 12 + 1, 1)


def _nom(rng: random.Random, segment: str) -> str:
    prenom, nom = rng.choice(_PRENOMS), rng.choice(_NOMS)
    if segment == "Pro & TPE":
        return rng.choice(_ENTREPRISES).format(prenom=prenom, nom=nom, NOM=nom.upper())
    return f"{prenom} {nom}"


def _segment(rng: random.Random) -> tuple:
    tirage, cumul = rng.random(), 0.0
    for segment in _SEGMENTS:
        cumul += segment[1]
        if tirage < cumul:
            return segment
    return _SEGMENTS[-1]


def _fin(rng: random.Random, segment: tuple, debut: date) -> date | None:
    """Le dernier jour d'abonnement, tiré mois par mois depuis le premier mois simulé (ou None)."""
    nom, _, risque, _ = segment
    mois = max(PREMIER_MOIS, debut.replace(day=1))
    while mois <= DERNIER_JOUR:
        anciennete = (mois.year - debut.year) * 12 + mois.month - debut.month
        p = _RISQUE_PRO_APRES_HAUSSE if nom == "Pro & TPE" and mois >= HAUSSE_PRO else risque
        p += _RISQUE_DEBUTANT if anciennete < 3 else 0.0
        if anciennete == 0:
            p *= 0.3  # partir le mois même de son arrivée arrive, mais rarement
        if rng.random() < p:
            return _jour_au_hasard(rng, mois, apres=debut)
        mois = _mois_suivant(mois)
    return None


def _simuler(rng: random.Random, stock: int) -> list[tuple]:
    lignes = []
    for _ in range(stock):  # les clients déjà là le 1er janvier 2024
        segment = _segment(rng)
        debut = date.fromordinal(rng.randint(date(2019, 1, 1).toordinal(), date(2023, 12, 31).toordinal()))
        lignes.append((debut, _nom(rng, segment[0]), segment, rng.choice(segment[3])))
    mois = PREMIER_MOIS
    while mois <= DERNIER_JOUR:
        for _ in range(rng.randint(60, 95)):
            segment = _segment(rng)
            lignes.append((_jour_au_hasard(rng, mois), _nom(rng, segment[0]), segment, rng.choice(segment[3])))
        mois = _mois_suivant(mois)
    abonnes = [(debut, nom, segment[0], _fin(rng, segment, debut), mensualite)
               for debut, nom, segment, mensualite in lignes]
    abonnes += [(debut, nom, segment, fin, mensualite) for nom, segment, debut, fin, mensualite in _SAISIES_PIEGES]
    abonnes.sort(key=lambda a: (a[0], a[1]))
    return abonnes


def generer_abonnes(chemin: str | Path = CHEMIN_EXEMPLE, graine: int = 2026, stock: int = 1800) -> Path:
    """Écrit le fichier des abonnés au format de `modeles.COLONNES_ABONNES` et renvoie son chemin."""
    rng = random.Random(graine)
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8", newline="") as f:
        ecrivain = csv.writer(f, lineterminator="\n")
        ecrivain.writerow(("client", "nom", "segment", "debut", "fin", "mensualite"))
        for numero, (debut, nom, segment, fin, mensualite) in enumerate(_simuler(rng, stock), start=1):
            ecrivain.writerow((f"C{numero:05d}", nom, segment, debut.isoformat(), fin.isoformat() if fin else "",
                               f"{mensualite:.2f}"))
    return chemin


if __name__ == "__main__":
    print(f"Écrit : {generer_abonnes()}")

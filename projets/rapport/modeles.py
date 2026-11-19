"""Les briques du projet 7 : abonnés, indicateurs, rapport, règles d'écriture (DÉJÀ ÉCRIT : à lire, pas à modifier).

Lisez ce fichier en premier. Il définit :

- le fichier des abonnés et `lire_abonnes`, qui le lit (la lecture n'est pas le sujet de ce projet :
  les projets 2 et 6 l'ont entraînée) ;
- les petites classes figées (`frozen=True`) qui circulent entre les étapes : `Abonne`,
  `Indicateurs`, `Rapport` ;
- les règles d'écriture françaises (noms des mois, espaces insécables) et les dimensions du graphique.

Le correcteur suppose que ce fichier n'a pas changé.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------- Le fichier des abonnés

#: Une ligne par abonné, en UTF-8, séparateur virgule, dates ISO (AAAA-MM-JJ), point décimal :
#:
#:     client,nom,segment,debut,fin,mensualite
#:     C00042,Durand & Fils,Pro & TPE,2024-03-14,2026-10-05,129.90
#:     C00043,Zoé Martin,Particulier,2025-01-02,,34.99
#:
#: - `debut` : le PREMIER jour d'abonnement ;
#: - `fin` : le DERNIER jour d'abonnement (le client est encore abonné ce jour-là) ; vide = toujours abonné ;
#: - `nom` : saisi à la main par les conseillers en boutique. Il peut contenir n'importe quoi :
#:   « & », des guillemets, des chevrons… et même du code HTML tapé pour « essayer ».
COLONNES_ABONNES = ("client", "nom", "segment", "debut", "fin", "mensualite")


@dataclass(frozen=True)
class Abonne:
    client: str
    nom: str
    segment: str
    debut: date
    fin: date | None      # dernier jour d'abonnement, inclus ; None : toujours abonné
    mensualite: float     # euros par mois


def lire_abonnes(chemin: str | Path) -> list[Abonne]:
    """Lit le fichier des abonnés (DÉJÀ ÉCRIT), dans l'ordre du fichier.

    Lève `FileNotFoundError` si le fichier n'existe pas, et `ValueError` (avec le numéro de la ligne)
    si l'en-tête ou une ligne est fausse : un rapport envoyé à la direction ne se fabrique pas sur un
    fichier douteux.
    """
    chemin = Path(chemin)
    abonnes: list[Abonne] = []
    with chemin.open(encoding="utf-8-sig", newline="") as f:
        lecteur = csv.DictReader(f)
        if tuple(lecteur.fieldnames or ()) != COLONNES_ABONNES:
            raise ValueError(f"{chemin.name} : en-tête {lecteur.fieldnames} au lieu de {list(COLONNES_ABONNES)}")
        for ligne in lecteur:
            ou = f"{chemin.name}, ligne {lecteur.line_num}"
            if None in ligne or any(v is None for v in ligne.values()):
                raise ValueError(f"{ou} : nombre de champs incorrect")
            try:
                debut = date.fromisoformat(ligne["debut"].strip())
                fin = date.fromisoformat(ligne["fin"].strip()) if ligne["fin"].strip() else None
                mensualite = float(ligne["mensualite"])
            except ValueError as exc:
                raise ValueError(f"{ou} : {exc}") from None
            if fin is not None and fin < debut:
                raise ValueError(f"{ou} : fin ({fin}) avant le début ({debut})")
            if not math.isfinite(mensualite) or mensualite < 0:
                raise ValueError(f"{ou} : mensualité impossible ({ligne['mensualite']!r})")
            client = ligne["client"].strip()
            if not client:
                raise ValueError(f"{ou} : identifiant client vide")
            abonnes.append(Abonne(client, ligne["nom"].strip(), ligne["segment"].strip(), debut, fin, mensualite))
    return abonnes


# ---------------------------------------------------------------------------- Les chiffres d'un mois (étape 2)


@dataclass(frozen=True)
class Indicateurs:
    mois: date                 # le 1er du mois
    actifs_debut: int          # abonnés au début du mois (voir l'étape 2 pour la règle exacte)
    arrivees: int              # abonnements commencés pendant le mois
    departs: int               # abonnements terminés pendant le mois (TOUS)
    actifs_fin: int            # abonnés à la fin du dernier jour du mois
    taux_churn: float | None   # entre 0 et 1 ; None quand personne n'était abonné au début du mois
    revenu_perdu: float        # euros par mois : la somme des mensualités de TOUS les départs du mois


# ---------------------------------------------------------------------------- Le rapport (étapes 5 et 6)


@dataclass(frozen=True)
class Rapport:
    mois: date                          # le 1er du mois du rapport
    actuel: Indicateurs                 # les chiffres du mois
    precedent: Indicateurs              # ceux du mois d'avant, pour comparer
    segments: dict[str, Indicateurs]    # segment → chiffres du mois, segments dans l'ordre de `sorted`
    historique: tuple[Indicateurs, ...] # les MOIS_HISTORIQUE derniers mois, du plus ancien au mois du rapport inclus
    departs: tuple[Abonne, ...]         # les départs du mois aux plus grosses mensualités (DEPARTS_AFFICHES au plus)


#: Le graphique montre les 12 derniers mois ; le tableau, les 10 plus gros départs.
MOIS_HISTORIQUE = 12
DEPARTS_AFFICHES = 10

# ---------------------------------------------------------------------------- Écrire à la française (étapes 1 et 3)

#: Les noms des mois, en minuscules comme le veut le français. `strftime("%B")` dépend de la langue
#: configurée sur l'ordinateur : le même programme écrirait « October » sur un serveur.
MOIS_FR = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
           "novembre", "décembre")
MOIS_COURTS = ("janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc.")

#: Entre les milliers : une espace FINE insécable (1 234 567). Avant « € » et « % » : une espace
#: insécable (12,5 %). « Insécable » : le navigateur ou le logiciel de courriel ne coupera jamais la
#: ligne à cet endroit, et « € » ne se retrouvera jamais seul au début d'une ligne.
ESPACE_MILLIERS = " "
ESPACE_UNITE = " "

#: Ce qu'on écrit quand une valeur n'existe pas (None) : un tiret cadratin, jamais « None » ni « nan ».
TIRET = "—"


def etiquette_courte(mois: date) -> str:
    """« oct. 26 » : l'étiquette d'un mois sous l'axe du graphique (DÉJÀ ÉCRIT)."""
    return f"{MOIS_COURTS[mois.month - 1]} {mois.year % 100:02d}"


# ---------------------------------------------------------------------------- Le graphique (étape 4)

#: L'espace de noms obligatoire d'un dessin SVG : sans lui, un navigateur affiche… du texte.
SVG_NS = "http://www.w3.org/2000/svg"

#: Dimensions du dessin et marges (en pixels) : la courbe vit dans le rectangle intérieur, les marges
#: accueillent les graduations (à gauche) et les étiquettes des mois (en bas).
LARGEUR, HAUTEUR = 640, 240
MARGE_HAUT, MARGE_DROITE, MARGE_BAS, MARGE_GAUCHE = 16, 32, 36, 56

#: Le haut de l'échelle quand il n'y a rien à montrer (aucune valeur, ou que des zéros) : 5 %.
ECHELLE_PAR_DEFAUT = 0.05

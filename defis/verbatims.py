"""Outils communs de la série 45 : les commentaires libres des clients, la règle actuelle et le modèle.

Chaque mois, l'enquête de satisfaction se termine par une question ouverte : « Avez-vous quelque chose à nous dire ? ».
Des milliers de clients répondent, en quelques mots, sans relire, souvent depuis leur téléphone. Aujourd'hui, une
seule règle les exploite : un commentaire qui contient « résili » déclenche l'appel d'un conseiller fidélisation.
Inès, chargée de la relation client, a les réponses des six derniers mois, et pour chacune elle sait si le client est
parti dans les deux mois qui ont suivi. Ce module lui fournit :

- `commentaires_exemple()` : les 4 000 réponses des six derniers mois, une liste de couples (texte, étiquette), où
  l'étiquette vaut "part" (le client a résilié dans les deux mois) ou "reste". Les textes sont tels que les clients
  les ont tapés : sans accents, en majuscules, avec des apostrophes typographiques, des émojis, « la 3ᵉ coupure »,
  et parfois la ligature « ﬁ » d'un texte copié depuis un PDF ;
- `nouveaux_exemple()` : les 1 000 réponses du mois suivant, au même format, pour juger le modèle sur des
  commentaires qu'il n'a jamais vus ;
- `regle_actuelle(texte)` : la règle d'aujourd'hui ("part" si le texte contient « résili », avec ou sans accent) ;
- `bilan(predites, vraies)` : ce que valent des alertes (combien de signalés, la précision, le rappel) ;
- `generer_commentaires(n, graine)` : d'autres commentaires, reproductibles, pour vos propres essais ;
- `Modele` : ce que renvoie l'entraînement du défi 274 (les classes, les log-probabilités a priori et celles de
  chaque mot) ;
- les constantes `NEGATIONS`, `FINS_DE_PORTEE`, `PONCTUATION` et `PREFIXE` (défi 273), `LETTRES_LIEES` (défi 272).

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> commentaires = commentaires_exemple()
    >>> len(commentaires), commentaires[10]
    (4000, ('Pas question de résilier, je suis bien ici.', 'reste'))
    >>> commentaires[0], commentaires[36][1]
    (('MON FILS UTILISE SURTOUT LA TÉLÉ.', 'reste'), 'part')
    >>> sum(etiquette == "part" for _, etiquette in commentaires), len(nouveaux_exemple())
    (785, 1000)
    >>> regle_actuelle("Je RESILIE !"), regle_actuelle("Aucune envie de résilier."), regle_actuelle("Mon code RIO ?")
    ('part', 'part', 'reste')
"""

from __future__ import annotations

import functools
import random
import unicodedata
from dataclasses import dataclass

PARTI, RESTE = "part", "reste"

# Défi 272 : les lettres qui ne se décomposent pas (ni NFD ni NFKD ne séparent « œ » en « o » + « e »).
LETTRES_LIEES = {"œ": "oe", "æ": "ae"}

# Défi 273 : les mots qui ouvrent une négation, ceux qui la referment, et les signes qui terminent une proposition.
# « plus » n'y est pas : « c'est plus cher » n'est pas une négation (« je ne veux plus » l'est grâce à « ne »).
# « n » y est : l'élision « n' » de « je n'ai jamais » devient le mot « n » (défi 272).
NEGATIONS = frozenset({"ne", "n", "pas", "jamais", "rien", "aucun", "aucune", "sans", "ni"})
FINS_DE_PORTEE = frozenset({"mais", "car", "donc", "sauf", "pourtant", "cependant"})
PONCTUATION = ".,;:!?…()\n"
PREFIXE = "non_"


@dataclass
class Modele:
    """Un classifieur bayésien naïf entraîné (défi 274).

    - `classes` : les étiquettes vues à l'entraînement, dans l'ordre croissant (`sorted`) ;
    - `log_a_priori` : pour chaque classe, le LOGARITHME (népérien) de la part des documents de cette classe ;
    - `log_proba` : pour chaque classe, un dict {mot: log P(mot | classe)} qui contient TOUS les mots du vocabulaire
      (les mots vus à l'entraînement, toutes classes confondues), avec le lissage de Laplace.
    """

    classes: list
    log_a_priori: dict
    log_proba: dict


# ---------------------------------------------------------------------------- la règle d'aujourd'hui


def regle_actuelle(texte: str) -> str:
    """La règle d'aujourd'hui : "part" si le texte contient « résili » (sans tenir compte des accents ni des majuscules).

        >>> regle_actuelle("Merci de traiter ma résiliation."), regle_actuelle("Je pars chez un concurrent.")
        ('part', 'reste')
    """
    sans_accents = "".join(c for c in unicodedata.normalize("NFKD", texte.casefold()) if not unicodedata.combining(c))
    return PARTI if "resili" in sans_accents else RESTE


def bilan(predites, vraies, classe: str = PARTI) -> dict:
    """Ce que valent des alertes : combien de commentaires `classe` sont signalés, combien le sont à raison, la
    PRÉCISION (la part des signalés qui sont vraiment `classe`) et le RAPPEL (la part des vrais `classe` signalés),
    arrondies à 0,001 près (None s'il n'y a rien à diviser).

        >>> bilan(["part", "part", "reste", "reste"], ["part", "reste", "part", "reste"])
        {'signales': 2, 'trouves': 1, 'precision': 0.5, 'rappel': 0.5}
    """
    paires = list(zip(predites, vraies))
    signales = sum(p == classe for p, _ in paires)
    trouves = sum(p == classe and v == classe for p, v in paires)
    a_trouver = sum(v == classe for _, v in paires)
    return {"signales": signales, "trouves": trouves,
            "precision": round(trouves / signales, 3) if signales else None,
            "rappel": round(trouves / a_trouver, 3) if a_trouver else None}


# ---------------------------------------------------------------------------- les phrases des clients

_SERVICES = ["internet", "la box", "la fibre", "le réseau mobile", "la télé", "le wifi", "la 4G", "la ligne fixe"]
_DUREES = ["deux jours", "trois jours", "une semaine", "dix jours", "ce matin", "hier soir", "lundi"]
_ANCIENNETES = ["2012", "2015", "2018", "2020", "dix ans", "cinq ans", "toujours"]

# Chaque famille : des phrases à trous ({service}, {duree}, {anciennete}, {rang}).
_NEUTRES = [
    "J'ai appelé le service client pour ma facture.",
    "Le technicien est passé mardi.",
    "J'ai changé de forfait le mois dernier.",
    "Ma box a redémarré après la mise à jour.",
    "Je suis client depuis {anciennete}.",
    "Mon fils utilise surtout {service}.",
    "J'ai reçu le nouveau décodeur.",
    "J'utilise {service} pour le télétravail.",
    "Nous sommes quatre à la maison.",
    "J'ai déménagé en septembre.",
    "J'ai une question sur l'option sport.",
    "Le courrier est arrivé en retard.",
]
_POSITIFS = [
    "Le débit est excellent.",
    "Conseiller très aimable, merci !",
    "Tout fonctionne parfaitement.",
    "Service rapide et efficace.",
    "{service} est top 👍",
    "Je recommande à mes proches.",
    "Je suis satisfait du service.",
    "Le débit est plus rapide qu'avant.",
    "Installation rapide, technicien au top.",
    "Très content de mon forfait.",
    "Merci pour le geste commercial.",
    "Bon rapport qualité prix.",
    "La conseillère a résolu mon problème en cinq minutes.",
]
_NEGATIFS = [
    "La facture a encore augmenté.",
    "Panne : {service} depuis {duree}.",
    "Le débit est très lent le soir.",
    "Trop d'attente au téléphone.",
    "Le prix est trop cher pour ce que c'est.",
    "C'est plus cher qu'avant.",
    "C'est la {rang} coupure ce mois-ci.",
    "Personne ne répond au service client.",
    "{service} coupe sans arrêt.",
    "Les frais de dossier sont abusifs.",
    "Encore une hausse de tarif.",
    "Le technicien n'est jamais venu.",
]
_DEPARTS = [
    "Je vais résilier mon abonnement.",
    "J'ai demandé mon code RIO.",
    "Je pars chez un concurrent.",
    "Je résilie dès la fin de mon engagement.",
    "Une offre concurrente est moins chère, je change.",
    "Merci de traiter ma résiliation.",
    "Je change d'opérateur le mois prochain.",
    "La portabilité est en cours.",
    "Ma lettre de résiliation est partie.",
    "J'ai trouvé moins cher ailleurs.",
]
_RESTES_NIES = [
    "Je ne vais pas résilier.",
    "Je ne compte pas résilier.",
    "Aucune envie de partir.",
    "Je n'ai jamais eu de coupure.",
    "Pas de problème avec {service}.",
    "Je ne vais pas changer d'opérateur.",
    "Sans souci depuis {anciennete}.",
    "Aucun problème de facture.",
    "Rien à redire.",
    "Pas question de résilier, je suis bien ici.",
    "Je n'ai aucune raison de résilier.",
]
_PARTS_NIES = [
    "Je ne suis pas satisfait.",
    "{service} ne marche jamais.",
    "Rien ne fonctionne correctement.",
    "Je n'ai aucune réponse à mes mails.",
    "Pas de réseau à la maison.",
    "Je ne recommande pas.",
    "Le débit n'est pas excellent du tout.",
    "Le service n'est pas rapide.",
    "Je ne suis plus content du tout.",
    "Personne ne m'a rappelé.",
]
_MAIS = [
    "Pas de coupure, mais la facture est trop chère.",
    "Je ne résilie pas, mais le débit est lent.",
    "Aucun souci avec {service}, mais le prix a augmenté.",
    "Le débit est excellent, mais je pars chez un concurrent.",
]
_EMOJIS = [" 😡", " 🙂", " 👍", " 😤", " !!!", " ?!", ""]
_RANGS = ["2ᵉ", "3ᵉ", "4ᵉ", "5ᵉ", "deuxième", "troisième"]

# Probabilité de tirer une phrase de chaque famille, selon que le client part ou reste.
_TIRAGES = {
    PARTI: [(_DEPARTS, 0.5), (_NEGATIFS, 0.4), (_PARTS_NIES, 0.5), (_NEUTRES, 0.25), (_POSITIFS, 0.05),
            (_RESTES_NIES, 0.05), (_MAIS, 0.08)],
    RESTE: [(_POSITIFS, 0.5), (_NEUTRES, 0.35), (_NEGATIFS, 0.25), (_RESTES_NIES, 0.4), (_DEPARTS, 0.02),
            (_PARTS_NIES, 0.04), (_MAIS, 0.1)],
}


def _remplir(phrase: str, alea: random.Random) -> str:
    texte = phrase.format(service=alea.choice(_SERVICES), duree=alea.choice(_DUREES),
                          anciennete=alea.choice(_ANCIENNETES), rang=alea.choice(_RANGS))
    return texte[0].upper() + texte[1:]


def _abimer(texte: str, alea: random.Random) -> str:
    """Le texte tel qu'un client le tape : sans accents, en majuscules, apostrophes courbes, ligature d'un PDF…"""
    if alea.random() < 0.3:                      # tapé sans accents (« resilier »)
        texte = "".join(c for c in unicodedata.normalize("NFD", texte) if not unicodedata.combining(c))
        texte = unicodedata.normalize("NFC", texte)
    if alea.random() < 0.4:                      # apostrophe typographique (traitement de texte, téléphone)
        texte = texte.replace("'", "’")
    if alea.random() < 0.03:                     # copié depuis un PDF : la ligature « ﬁ » (U+FB01)
        texte = texte.replace("fi", "ﬁ")
    if alea.random() < 0.05:                     # EN COLÈRE
        texte = texte.upper()
    elif alea.random() < 0.1:
        texte = texte.lower()
    return texte + alea.choice(_EMOJIS) if alea.random() < 0.15 else texte


def _commentaire(etiquette: str, alea: random.Random) -> str:
    phrases = [_remplir(alea.choice(famille), alea) for famille, p in _TIRAGES[etiquette] if alea.random() < p]
    if not phrases:
        phrases = [_remplir(alea.choice(_NEUTRES), alea)]
    alea.shuffle(phrases)
    return _abimer(" ".join(phrases[:3]), alea)


def generer_commentaires(n: int, graine: int = 0, part_partis: float = 0.2) -> list[tuple[str, str]]:
    """`n` commentaires (texte, étiquette), reproductibles ; une part `part_partis` des clients part.

        >>> generer_commentaires(2, graine=1)
        [('le service n’est pas rapide. sans souci depuis toujours. !!!', 'part'), ('NOUS SOMMES QUATRE A LA MAISON.', 'part')]
    """
    alea = random.Random(graine)
    commentaires = []
    for _ in range(n):
        etiquette = PARTI if alea.random() < part_partis else RESTE
        commentaires.append((_commentaire(etiquette, alea), etiquette))
    return commentaires


@functools.lru_cache(maxsize=1)
def _corpus() -> dict:
    return {"anciens": tuple(generer_commentaires(4000, graine=4501)),
            "nouveaux": tuple(generer_commentaires(1000, graine=4502))}


def commentaires_exemple() -> list[tuple[str, str]]:
    """Les 4 000 réponses des six derniers mois : [(texte, "part" ou "reste"), …]."""
    return list(_corpus()["anciens"])


def nouveaux_exemple() -> list[tuple[str, str]]:
    """Les 1 000 réponses du mois suivant, au même format : pour juger le modèle sur des commentaires jamais vus."""
    return list(_corpus()["nouveaux"])

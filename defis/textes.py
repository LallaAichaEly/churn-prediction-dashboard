"""Outils de la série 20, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Des tables qu'on ne retient pas par cœur (l'alphabet des SMS, les lettres sans décomposition) et des
textes tels qu'ils arrivent vraiment : saisis au clavier, copiés d'une page web, venus d'un Mac, d'un
export Excel ou d'un formulaire en ligne. Tout est réimporté dans `defis/serie_20.py`.

    >>> from defis.serie_20 import *
    >>> NOMS_EXEMPLE[0] == NOMS_EXEMPLE[1]          # identiques à l'écran…
    False
    >>> len(NOMS_EXEMPLE[0]), len(NOMS_EXEMPLE[1])  # … mais pas pour Python
    (12, 15)
    >>> "€" in GSM_BASE, "€" in GSM_EXTENSION
    (False, True)
"""

from __future__ import annotations

import unicodedata

# ============================================================================ Défis 122 et 123 : comparer, chercher

#: Le même client, saisi six fois par six personnes : à l'écran, les cinq premiers se lisent pareil.
NOMS_EXEMPLE = [
    "Hélène Dupré",                                  # tapé au clavier (forme composée, NFC)
    "Hélène Dupré",                # venu d'un Mac : chaque accent est un caractère à part (NFD)
    "HÉLÈNE  DUPRÉ ",                                # en majuscules, deux espaces, une espace finale
    "Hélène Dupré",                             # espace insécable, copiée d'un document Word
    "Hélène Dupré​",                            # espace de largeur nulle, copiée d'une page web
    "Hélène Dupre",                                  # une AUTRE orthographe : sans l'accent final
]

#: Lettres qui ne se décomposent PAS en « lettre + accent » : NFD ne les sépare pas (défi 123).
LETTRES_LIEES = {"œ": "oe", "æ": "ae", "ø": "o", "ł": "l", "đ": "d"}

#: Apostrophes typographiques (Word, macOS, téléphones) à ramener à l'apostrophe droite « ' » (défi 123).
APOSTROPHES = "‘’ʼ"               # ‘ ’ ʼ

#: Tirets et signe moins à ramener au trait d'union « - » (défi 123).
TIRETS = "‐‑‒–—−"  # ‐ ‑ ‒ – — −

# ============================================================================ Défi 124 : numéros de téléphone

#: Le même numéro, tel qu'il arrive de sept sources différentes (formulaire, Excel, fiche papier…).
NUMEROS_EXEMPLE = [
    "06 12 34 56 78",
    "06.12.34.56.78",
    "+33 6 12 34 56 78",
    "+33 (0)6 12 34 56 78",
    "0033 6 12 34 56 78",
    "06 12 34 56 78",           # espaces fines insécables (typographie française)
    "612345678",                                     # Excel a mangé le 0 du début
]

# ============================================================================ Défi 125 : le prix d'un SMS

#: L'alphabet GSM 03.38 (norme 3GPP TS 23.038) : chacun de ces 127 caractères coûte 1 unité sur 7 bits.
#: Retour à la ligne et retour chariot compris. Remarquez ce qui MANQUE : ê â î ô û ë ï ç œ ’ « » …
GSM_BASE = (
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?"
    "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)

#: Les caractères d'EXTENSION de l'alphabet GSM : ils passent en GSM-7, mais coûtent 2 unités chacun
#: (un caractère d'échappement, puis le caractère). L'euro en fait partie.
GSM_EXTENSION = "\f^{}\\[]~|€"

#: Trois SMS de relance. Le 2e ne diffère du 1er que par une apostrophe… et coûte le double.
MESSAGES_EXEMPLE = [
    "Bonjour Hélène, votre offre fidélité vous attend : -20 % pendant 12 mois. Répondez OUI avant dimanche.",
    "Bonjour Hélène, votre offre fidélité vous attend : -20 % pendant 12 mois. Répondez OUI avant l’été.",
    "Bonjour Hélène, votre offre fidélité vous attend : -20 € pendant 12 mois. Répondez OUI avant dimanche.",
]

# ============================================================================ Défi 126 : l'export abîmé

#: Un export CSV en UTF-8, ouvert par erreur comme du Windows-1252 (cp1252), puis réenregistré.
EXPORT_ABIME = [
    "HÃ©lÃ¨ne DuprÃ©",                               # Hélène Dupré
    "Prix : 20 â‚¬ par mois",                        # Prix : 20 € par mois
    "Ã\u0081LVAREZ Ã\u0081ngel",                    # ÁLVAREZ Ángel (0x81 : un « trou » de cp1252)
    "FranÃƒÂ§ois",                                   # François, abîmé DEUX fois
    "Hélène Dupré",                                  # cette ligne-là était saine
]

# ============================================================================ Défi 127 : couper un texte

ZWJ = "‍"  # « zero width joiner » : colle deux emojis en un seul (👨 + ZWJ + 👩 + ZWJ + 👧 = 👨‍👩‍👧)


def montrer(texte: str) -> list[str]:
    """Les caractères d'un texte, un par un, avec leur nom Unicode : pour VOIR ce que `len` compte.

        >>> montrer("é")
        ['U+0065 LATIN SMALL LETTER E', 'U+0301 COMBINING ACUTE ACCENT']
    """
    return [f"U+{ord(c):04X} {unicodedata.name(c, '(sans nom)')}" for c in texte]

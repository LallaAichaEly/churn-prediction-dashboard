"""Série 20 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_20.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md le caractère auquel vous n'aviez pas pensé.

Fil rouge de la série : un « caractère » n'existe pas. Il y a des POINTS DE CODE (ce que `len` compte),
des OCTETS (ce que le disque et le réseau voient, selon l'encodage), des UNITÉS (ce que l'opérateur SMS
facture) et des GRAPHÈMES (ce que l'œil voit). Trois réflexes :

1. normaliser AVANT de comparer (`unicodedata.normalize`, `casefold`, espaces) : deux textes identiques
   à l'écran ne sont pas forcément égaux pour `==` ;
2. savoir quelle « longueur » on mesure, et choisir la bonne : points de code, octets, unités UTF-16,
   graphèmes ;
3. ne jamais deviner en silence : un texte qu'on ne sait pas lire se refuse (ValueError) ou se rend
   tel quel ; on ne le « corrige » pas à moitié.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import unicodedata

from defis.textes import (  # noqa: F401
    APOSTROPHES,
    EXPORT_ABIME,
    GSM_BASE,
    GSM_EXTENSION,
    LETTRES_LIEES,
    MESSAGES_EXEMPLE,
    NOMS_EXEMPLE,
    NUMEROS_EXEMPLE,
    TIRETS,
    ZWJ,
    montrer,
)

# ============================================================================ Comparer et chercher


def cle_comparaison(texte: str) -> str:
    # NFKC rassemble ce qui s'écrit de plusieurs façons : « é » en un ou deux points de code (NFC le
    # ferait aussi), mais aussi les formes « de compatibilité » : ligature ﬁ → fi, lettres pleine chasse
    # Ｍ → M, espaces insécables → espace. casefold (et pas lower) : « ß » devient « ss ».
    texte = unicodedata.normalize("NFKC", texte).casefold()
    # Les caractères de « format » (catégorie Cf) sont invisibles : espace de largeur nulle, BOM, trait
    # d'union conditionnel… Ils n'ont rien à faire dans une clé.
    texte = "".join(c for c in texte if unicodedata.category(c) != "Cf")
    # On renormalise : dans de rares cas (ǰ, ΐ), casefold rend un texte qui n'est plus normalisé.
    # split() sans argument coupe sur TOUS les blancs Unicode, et en enlève aux deux bouts.
    return " ".join(unicodedata.normalize("NFKC", texte).split())


_TRADUCTION = str.maketrans({**LETTRES_LIEES, **{a: "'" for a in APOSTROPHES}, **{t: "-" for t in TIRETS}})


def cle_recherche(texte: str) -> str:
    # 1. tout ce que fait déjà la clé de comparaison (inutile de le réécrire, et ça reste cohérent) ;
    # 2. NFD sépare chaque lettre de ses accents, qu'on retire (ce sont des « marques » combinantes) ;
    # 3. les lettres liées (œ, ø…) n'ont pas d'accent à retirer : une table de traduction s'en charge,
    #    avec les apostrophes et les tirets typographiques ;
    # 4. NFC recompose ce qui peut l'être, pour rendre un texte « normal ».
    #
    # ⚠️ PAS `texte.encode("ascii", "ignore")` : sur un texte composé, il supprime la LETTRE avec son
    # accent (« Hélène » → « Hlne »), et efface tout ce qui n'est pas latin (« Анна » → « »).
    decompose = unicodedata.normalize("NFD", cle_comparaison(texte))
    sans_accents = "".join(c for c in decompose if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", sans_accents.translate(_TRADUCTION))


# ============================================================================ Normaliser une saisie

_SEPARATEURS = ".-/"


def numero_e164(texte: str) -> str:
    # On garde les chiffres et le « + » en retirant les séparateurs connus. Tout le reste est une erreur :
    # on ne devine JAMAIS un numéro (un SMS envoyé au mauvais numéro, c'est un client qui n'est pas le bon).
    brut = "".join(c for c in texte if not c.isspace() and c not in _SEPARATEURS)
    for prefixe in ("+33(0)", "0033(0)"):            # « (0) » : seulement juste après l'indicatif
        if brut.startswith(prefixe):
            brut = prefixe[:-3] + brut[len(prefixe):]
    # « 0123456789 » et pas isdigit() : "²".isdigit() et "٣".isdigit() sont vrais, et int("٣") vaut 3.
    if brut.count("+") > (1 if brut.startswith("+") else 0) or any(c not in "+0123456789" for c in brut):
        raise ValueError(f"numéro illisible : {texte!r}")
    if brut.startswith("+"):
        if not brut.startswith("+33"):
            raise ValueError(f"indicatif non pris en charge : {texte!r} (seul +33 l'est)")
        national = brut[3:]
    elif brut.startswith("0033"):
        national = brut[4:]
    elif len(brut) == 10 and brut.startswith("0"):
        national = brut[1:]
    elif len(brut) == 9:                             # Excel a pris le numéro pour un nombre : le 0 a disparu
        national = brut
    else:
        raise ValueError(f"numéro illisible : {texte!r}")
    if len(national) != 9 or national[0] == "0":
        raise ValueError(f"numéro français invalide : {texte!r} (9 chiffres après +33, le premier n'est pas 0)")
    return "+33" + national


# ============================================================================ Compter ce qu'on paie

_GSM_BASE, _GSM_EXTENSION = set(GSM_BASE), set(GSM_EXTENSION)


def _segments(tailles: list[int], seul: int, par_segment: int) -> int:
    """Nombre de segments : remplis dans l'ordre, sans jamais couper un caractère de 2 unités."""
    if sum(tailles) <= seul:
        return 1
    segments, place = 1, par_segment
    for taille in tailles:
        if taille > place:                           # il ne rentre pas en entier : segment suivant
            segments, place = segments + 1, par_segment
        place -= taille
    return segments


def cout_sms(message: str) -> tuple[str, int]:
    # On compte le message tel qu'il PARTIRA : pas de normalisation ici. Un « é » en NFD (e + accent
    # combinant) n'est pas dans l'alphabet GSM : ce SMS-là part vraiment en UCS-2.
    if message == "":
        raise ValueError("message vide")
    if all(c in _GSM_BASE or c in _GSM_EXTENSION for c in message):
        return "GSM-7", _segments([2 if c in _GSM_EXTENSION else 1 for c in message], 160, 153)
    # UCS-2 (en pratique UTF-16) : 1 unité par caractère, 2 pour ceux au-delà de U+FFFF (les emojis).
    return "UCS-2", _segments([2 if ord(c) > 0xFFFF else 1 for c in message], 70, 67)


# ============================================================================ Réparer un texte abîmé


def _octets_cp1252(texte: str) -> bytes:
    """Les octets qu'avait le texte avant d'être lu en cp1252. UnicodeEncodeError si ce n'est pas possible."""
    octets = bytearray()
    for c in texte:
        try:
            octets += c.encode("cp1252")
        except UnicodeEncodeError:
            # Les « trous » de cp1252 (0x81, 0x8D, 0x8F, 0x90, 0x9D) : Windows les lit comme U+0081…
            # Plus généralement, un caractère < 256 que cp1252 ne sait pas écrire redevient son octet
            # (c'est aussi ce qui arrive à un texte lu en latin-1).
            if ord(c) > 0xFF:
                raise
            octets.append(ord(c))
    return bytes(octets)


def reparer_mojibake(texte: str) -> str:
    # Tant que le texte se répare EN ENTIER, on recommence (un texte abîmé deux fois). Chaque tour
    # raccourcit le texte (plusieurs caractères redeviennent un seul) : la boucle s'arrête forcément.
    while True:
        try:
            repare = _octets_cp1252(texte).decode("utf-8")
        except UnicodeError:                         # UnicodeEncodeError ET UnicodeDecodeError
            return texte                             # ne se répare pas : on n'y touche pas
        if repare == texte:                          # texte en ASCII pur : rien à réparer
            return texte
        texte = repare


# ============================================================================ Couper sans abîmer

_TEINTES = range(0x1F3FB, 0x1F3FF + 1)
_DRAPEAUX = range(0x1F1E6, 0x1F1FF + 1)


def _graphemes(texte: str) -> list[str]:
    """Le texte découpé en caractères VISIBLES (règles simplifiées du défi 127)."""
    groupes: list[str] = []
    colle = False                                    # le caractère précédent était un ZWJ
    for c in texte:
        attache = groupes and (
            colle
            or unicodedata.category(c).startswith("M")  # accent combinant, sélecteur de variante, keycap
            or ord(c) in _TEINTES
            or c == ZWJ
            or (ord(c) in _DRAPEAUX and len(groupes[-1]) == 1 and ord(groupes[-1]) in _DRAPEAUX)
        )
        if attache:
            groupes[-1] += c
        else:
            groupes.append(c)
        colle = c == ZWJ
    return groupes


def tronquer(texte: str, limite: int) -> str:
    if limite < 1:
        raise ValueError(f"limite doit valoir au moins 1 (reçu : {limite})")
    graphemes = _graphemes(texte)
    if len(graphemes) <= limite:                     # ça tient : on rend le texte TEL QUEL
        return texte
    return "".join(graphemes[: limite - 1]).rstrip() + "…"


# ============================================================================ Les tests qu'on écrirait


def test_les_cinq_saisies_du_meme_nom_donnent_la_meme_cle():
    assert {cle_comparaison(nom) for nom in NOMS_EXEMPLE[:5]} == {"hélène dupré"}
    assert cle_comparaison(NOMS_EXEMPLE[5]) != cle_comparaison(NOMS_EXEMPLE[0])   # l'accent compte
    assert cle_comparaison("JÜRGEN STRASSE") == cle_comparaison("Jürgen Straße")


def test_la_recherche_ignore_accents_ligatures_et_typographie():
    assert cle_recherche("Hélène Dupré") == "helene dupre"
    assert cle_recherche("Cœur d’Alène – Ørsted") == "coeur d'alene - orsted"
    assert cle_recherche("Анна Ёлкина") == "анна елкина"                    # rien n'est effacé
    assert all(cle_recherche("dupre") in cle_recherche(nom) for nom in NOMS_EXEMPLE)


def test_tous_les_formats_donnent_le_meme_numero():
    assert {numero_e164(n) for n in NUMEROS_EXEMPLE} == {"+33612345678"}
    for faux in ("06 12 34 56 7O", "06 12 34 56", "+32 470 12 34 56", "٠٦١٢٣٤٥٦٧٨", "+33 06 12 34 56 78", ""):
        try:
            numero_e164(faux)
        except ValueError:
            continue
        raise AssertionError(f"{faux!r} aurait dû être refusé")


def test_une_apostrophe_double_le_prix_du_sms():
    assert [cout_sms(m) for m in MESSAGES_EXEMPLE] == [("GSM-7", 1), ("UCS-2", 2), ("GSM-7", 1)]
    assert cout_sms("a" * 160) == ("GSM-7", 1) and cout_sms("a" * 161) == ("GSM-7", 2)
    assert cout_sms("a" * 152 + "€" + "a" * 152) == ("GSM-7", 3)                  # l'euro ne se coupe pas
    assert cout_sms("ê" * 70) == ("UCS-2", 1) and cout_sms("ê" * 69 + "😀") == ("UCS-2", 2)  # emoji : 2 unités


def test_l_export_se_repare_et_le_texte_sain_reste_intact():
    assert [reparer_mojibake(ligne) for ligne in EXPORT_ABIME] == [
        "Hélène Dupré", "Prix : 20 € par mois", "ÁLVAREZ Ángel", "François", "Hélène Dupré"]
    assert reparer_mojibake("Łódź") == "Łódź"                                    # ni cp1252, ni abîmé


def test_tronquer_ne_coupe_jamais_un_caractere_visible():
    famille = "👨" + ZWJ + "👩" + ZWJ + "👧"
    assert tronquer("Votre offre expire bientôt", 10) == "Votre off…"
    assert tronquer("Votre offre expire bientôt", 7) == "Votre…"
    assert tronquer(NOMS_EXEMPLE[1], 12) == NOMS_EXEMPLE[1]                       # 15 points de code, 12 graphèmes
    assert tronquer(f"Bravo {famille} !", 8) == f"Bravo {famille}…"
    assert tronquer("🇫🇷🇧🇪🇨🇭", 2) == "🇫🇷…"

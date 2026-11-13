"""Vérifications automatiques de la série 20 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Le correcteur vous envoie des textes tels qu'ils arrivent vraiment, avec ce que l'œil ne voit pas :

- aux défis 122 et 123, le même nom en forme composée et décomposée, en majuscules, avec une espace
  insécable, une espace de largeur nulle, un BOM, une ligature « ﬁ », des lettres pleine chasse, un « ß »,
  un « œ », une apostrophe de Word, et des noms en cyrillique et en grec qui ne doivent pas disparaître ;
- au défi 124, des numéros dans tous les formats… et des chiffres arabes, un « ² », la lettre O à la place
  d'un zéro, un numéro belge, un « +33 06 » ;
- au défi 125, des messages pile à la frontière d'un segment, avec un « € » ou un emoji à cheval sur deux
  segments ;
- au défi 126, des textes abîmés une ou deux fois, des textes sains qu'il ne faut pas toucher, et des
  textes illisibles en cp1252. Une boucle qui ne s'arrête jamais est détectée et arrêtée ;
- au défi 127, des accents décomposés, des familles d'emojis, des teintes de peau, des drapeaux et des
  touches « 1️⃣ » à la limite exacte de la coupe.

Chaque fonction est aussi comparée au corrigé sur des centaines de textes tirés au hasard.
"""

from __future__ import annotations

import random
import sys
import unicodedata

from .textes import (APOSTROPHES, EXPORT_ABIME, GSM_BASE, GSM_EXTENSION, LETTRES_LIEES, MESSAGES_EXEMPLE,
                     NOMS_EXEMPLE, NUMEROS_EXEMPLE, TIRETS, ZWJ)
from .verif import Defi, Serie


def _court(valeur, n: int = 160) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


def _decrire(exc: BaseException) -> str:
    return f"{type(exc).__name__}({str(exc)!r})" if str(exc) else type(exc).__name__


def _appel(appel, quoi: str, conseil: str = ""):
    """Appelle `appel()` ; une exception inattendue devient un message qui dit QUEL appel a planté."""
    try:
        return appel()
    except NotImplementedError:
        raise
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None


def _exiger_texte(resultat, quoi: str) -> str:
    if resultat is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(resultat, str):
        raise AssertionError(f"{quoi} doit renvoyer un texte (str), pas {type(resultat).__name__} : {_court(resultat)}.")
    return resultat


# ---------------------------------------------------------------------------- les références (le corrigé, recopié)


def _ref_comparaison(texte: str) -> str:
    texte = unicodedata.normalize("NFKC", texte).casefold()
    texte = "".join(c for c in texte if unicodedata.category(c) != "Cf")
    return " ".join(unicodedata.normalize("NFKC", texte).split())


_TRADUCTION = str.maketrans({**LETTRES_LIEES, **{a: "'" for a in APOSTROPHES}, **{t: "-" for t in TIRETS}})


def _ref_recherche(texte: str) -> str:
    decompose = unicodedata.normalize("NFD", _ref_comparaison(texte))
    return unicodedata.normalize("NFC", "".join(c for c in decompose if not unicodedata.combining(c))
                                 .translate(_TRADUCTION))


def _ref_segments(tailles: list[int], seul: int, par_segment: int) -> int:
    if sum(tailles) <= seul:
        return 1
    segments, place = 1, par_segment
    for taille in tailles:
        if taille > place:
            segments, place = segments + 1, par_segment
        place -= taille
    return segments


def _ref_cout(message: str) -> tuple[str, int]:
    if all(c in GSM_BASE or c in GSM_EXTENSION for c in message):
        return "GSM-7", _ref_segments([2 if c in GSM_EXTENSION else 1 for c in message], 160, 153)
    return "UCS-2", _ref_segments([2 if ord(c) > 0xFFFF else 1 for c in message], 70, 67)


def _ref_octets(texte: str) -> bytes:
    octets = bytearray()
    for c in texte:
        try:
            octets += c.encode("cp1252")
        except UnicodeEncodeError:
            if ord(c) > 0xFF:
                raise
            octets.append(ord(c))
    return bytes(octets)


def _ref_reparer(texte: str) -> str:
    while True:
        try:
            repare = _ref_octets(texte).decode("utf-8")
        except UnicodeError:
            return texte
        if repare == texte:
            return texte
        texte = repare


def _ref_graphemes(texte: str) -> list[str]:
    groupes: list[str] = []
    colle = False
    for c in texte:
        code = ord(c)
        if groupes and (colle or unicodedata.category(c).startswith("M") or 0x1F3FB <= code <= 0x1F3FF or c == ZWJ
                        or (0x1F1E6 <= code <= 0x1F1FF and len(groupes[-1]) == 1
                            and 0x1F1E6 <= ord(groupes[-1]) <= 0x1F1FF)):
            groupes[-1] += c
        else:
            groupes.append(c)
        colle = c == ZWJ
    return groupes


def _ref_tronquer(texte: str, limite: int) -> str:
    graphemes = _ref_graphemes(texte)
    if len(graphemes) <= limite:
        return texte
    return "".join(graphemes[: limite - 1]).rstrip() + "…"


# ---------------------------------------------------------------------------- textes tirés au hasard

_PRENOMS = ["Hélène", "Zoé", "Noël", "Chloé", "Anaïs", "Joël", "François", "Gaëlle", "Jérôme", "Maëlys", "Bérénice",
            "Ève", "Côme", "Inès", "Raphaël", "Léa", "Œdipe", "Søren", "Łucja", "Đorđe", "Ørjan", "Ægir", "Jürgen",
            "Анна", "Ёлкина", "Αθηνά", "Ελένη", "Ἀθῆναι", "Soﬁane", "Ｍａｒｔｉｎ", "Straße", "O’Brien", "N’Diaye",
            "Jean‑Pierre", "Marie–Anne", "Ali", "Mei"]
_BLANCS = [" ", "  ", " ", " ", "\t", "\n", "  "]
_INVISIBLES = ["", "", "", "​", "﻿", "­", "‎"]


def _nom_au_hasard(rng: random.Random) -> str:
    morceaux = []
    for _ in range(rng.randint(1, 3)):
        mot = rng.choice(_PRENOMS)
        mot = rng.choice([mot, mot.upper(), mot.lower(), mot.title()])
        if rng.random() < 0.4:
            mot = unicodedata.normalize("NFD", mot)
        morceaux.append(mot + rng.choice(_INVISIBLES))
    debut, fin = rng.choice(["", "", " ", "\t"]), rng.choice(["", "", " ", "\n"])
    return debut + "".join(m + rng.choice(_BLANCS) for m in morceaux[:-1]) + morceaux[-1] + fin


def _comparer_au_corrige(fonction, nom: str, entrees, reference, conseil: str = "") -> None:
    for entree in entrees:
        args = entree if isinstance(entree, tuple) else (entree,)
        attendu = reference(*args)
        quoi = f"{nom}({', '.join(_court(a, 90) for a in args)})"
        obtenu = _appel(lambda: fonction(*args), quoi)
        if obtenu != attendu:
            raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {_court(attendu)}. {conseil}".rstrip())


# ============================================================================ Défi 122 : cle_comparaison


def verifier_cle_comparaison(cle) -> None:
    quoi = "cle_comparaison('Hélène Dupré')"
    obtenu = _exiger_texte(_appel(lambda: cle("Hélène Dupré"), quoi), quoi)
    if obtenu != "hélène dupré":
        if unicodedata.normalize("NFC", obtenu) == "hélène dupré":
            conseil = ("Votre clé est en forme DÉCOMPOSÉE (chaque accent est un caractère à part) : normalisez avec "
                       "NFKC (ou NFC), pas NFD ni NFKD. Une clé doit rester un texte qu'on peut afficher et stocker.")
        elif obtenu.casefold() == "hélène dupré":
            conseil = "La clé ne doit pas tenir compte des majuscules : `.casefold()`."
        elif obtenu == "helene dupre":
            conseil = ("Gardez les accents : « Dupré » et « Dupre » sont deux orthographes, on ne fusionne pas deux "
                       "personnes sur une supposition (ignorer les accents, c'est pour la RECHERCHE, au défi 123).")
        else:
            conseil = ""
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de 'hélène dupré'. {conseil}".rstrip())

    cas = [
        ("Hélène Dupré", "hélène dupré",
         "Ce nom vient d'un Mac : chaque accent est un caractère à part (forme décomposée). Les deux s'affichent "
         "pareil : `unicodedata.normalize(\"NFKC\", texte)` les met d'accord."),
        ("HÉLÈNE DUPRÉ", "hélène dupré", "La clé ne doit pas tenir compte des majuscules : `.casefold()`."),
        ("Jürgen Straße", "jürgen strasse",
         "« ß » doit devenir « ss », comme dans « STRASSE » : c'est ce que fait `.casefold()`, et pas `.lower()` "
         "(qui garde le « ß »)."),
        ("JÜRGEN STRASSE", "jürgen strasse", "La clé ne doit pas tenir compte des majuscules : `.casefold()`."),
        ("Soﬁane", "sofiane",
         "Ce nom, copié d'un PDF, contient la ligature « ﬁ » (un seul caractère). NFC la garde ; NFKC la ramène à "
         "« fi » : normalisez en NFKC."),
        ("Ｍａｒｔｉｎ", "martin",
         "Lettres « pleine chasse » (saisies avec un clavier japonais) : NFKC les ramène aux lettres ordinaires, "
         "NFC non."),
        ("Hélène Dupré", "hélène dupré",
         "Ce nom contient une espace INSÉCABLE (U+00A0, copiée d'un document Word) : NFKC la remplace par une "
         "espace ordinaire, et `split()` la reconnaît comme un blanc."),
        ("Hélène Dupré​", "hélène dupré",
         "Ce nom se termine par une espace de LARGEUR NULLE (U+200B, copiée d'une page web), invisible à l'écran. "
         "Ni `strip()` ni NFKC ne la retirent : écartez les caractères dont `unicodedata.category(c)` vaut « Cf »."),
        ("﻿Hélène Dupré", "hélène dupré",
         "Ce nom commence par un BOM (U+FEFF), le marqueur qu'Excel met en tête d'un fichier UTF-8 et qui se retrouve "
         "collé à la première cellule. C'est un caractère de format : catégorie « Cf », à écarter."),
        ("Jean­Pierre", "jeanpierre",
         "Ce nom contient un trait d'union CONDITIONNEL (U+00AD), invisible sauf en fin de ligne : catégorie « Cf », "
         "à écarter."),
        ("HÉLÈNE  DUPRÉ ", "hélène dupré",
         "Une suite de blancs devient UNE espace, et on retire ceux des bords : `\" \".join(texte.split())`."),
        ("  Anne\tMarie\n", "anne marie",
         "Tabulations et retours à la ligne sont des blancs aussi : `split()` SANS argument coupe sur tous les blancs "
         "Unicode (`split(\" \")` ne coupe que sur l'espace)."),
        ("Anne  Marie", "anne marie",
         "Espaces fines (U+202F, U+2009) : `split()` sans argument les reconnaît, et une suite de blancs donne UNE "
         "espace."),
        ("", "", "Un texte vide donne une clé vide."),
        (" \t\n", "", "Un texte qui ne contient que des blancs donne une clé vide."),
    ]
    for entree, attendu, conseil in cas:
        quoi = f"cle_comparaison({entree!r})"
        obtenu = _exiger_texte(_appel(lambda: cle(entree), quoi), quoi)
        if obtenu != attendu:
            raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {attendu!r}. {conseil}")

    for a, b in [("Hélène Dupré", "Hélène Dupre"), ("Noël", "Noel"), ("Zoé", "Zoe")]:
        if _appel(lambda: cle(a), f"cle_comparaison({a!r})") == _appel(lambda: cle(b), f"cle_comparaison({b!r})"):
            raise AssertionError(f"cle_comparaison donne la même clé à {a!r} et à {b!r} : gardez les accents. Ce sont "
                                 "deux orthographes ; ignorer les accents, c'est pour la RECHERCHE (défi 123).")

    rng = random.Random(122)
    noms = [_nom_au_hasard(rng) for _ in range(300)]
    _comparer_au_corrige(cle, "cle_comparaison", noms, _ref_comparaison)
    for nom in noms[:50]:
        une_fois = cle(nom)
        if cle(une_fois) != une_fois:
            raise AssertionError(f"cle_comparaison n'est pas stable : appliquée à sa propre clé {une_fois!r}, elle donne "
                                 f"{cle(une_fois)!r}. Une clé doit être sa propre clé.")


# ============================================================================ Défi 123 : cle_recherche


def verifier_cle_recherche(cle) -> None:
    quoi = "cle_recherche('Hélène Dupré')"
    obtenu = _exiger_texte(_appel(lambda: cle("Hélène Dupré"), quoi), quoi)
    if obtenu != "helene dupre":
        if obtenu.strip() in ("hlne dupr", "Hlne Dupr"):
            conseil = ("Les LETTRES accentuées ont disparu avec leur accent : c'est ce que fait "
                       "`encode(\"ascii\", \"ignore\")` sur un texte en forme composée. Décomposez d'abord (NFD), puis "
                       "écartez seulement les caractères combinants (`unicodedata.combining(c)` non nul).")
        elif obtenu == "hélène dupré":
            conseil = ("Les accents sont toujours là. En forme composée, « é » est UN caractère : décomposez d'abord "
                       "avec `unicodedata.normalize(\"NFD\", ...)`, puis écartez les caractères dont "
                       "`unicodedata.combining(c)` est non nul.")
        elif obtenu.casefold() == "helene dupre":
            conseil = "La recherche ne tient pas compte des majuscules : partez de `cle_comparaison(texte)` (défi 122)."
        elif unicodedata.normalize("NFC", obtenu) == "hélène dupré":
            conseil = ("Le texte est décomposé, mais les accents combinants sont restés : écartez les caractères dont "
                       "`unicodedata.combining(c)` est non nul.")
        else:
            conseil = ""
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de 'helene dupre'. {conseil}".rstrip())

    cas = [
        ("Hélène Dupré", "helene dupre", "Ce nom vient d'un Mac (forme décomposée)."),
        ("HÉLÈNE  DUPRÉ​ ", "helene dupre",
         "Tout ce que fait `cle_comparaison` (défi 122) s'applique aussi : réutilisez-la."),
        ("Jürgen Straße", "jurgen strasse", "Tout ce que fait `cle_comparaison` (défi 122) s'applique aussi : « ß » → « ss »."),
        ("Cœur", "coeur",
         "« œ » ne se décompose pas (ce n'est pas « o » + un accent) : remplacez-le grâce à `LETTRES_LIEES`."),
        ("Ørsted", "orsted", "« ø » ne se décompose pas : `LETTRES_LIEES` le ramène à « o »."),
        ("ÆSIR", "aesir", "« Æ » devient « æ » avec casefold, puis « ae » grâce à `LETTRES_LIEES`."),
        ("Łódź", "lodz", "« ł » ne se décompose pas : `LETTRES_LIEES`. « ó » et « ź », eux, se décomposent."),
        ("Đorđe", "dorde", "« đ » ne se décompose pas : `LETTRES_LIEES`."),
        ("Jeanne d’Arc", "jeanne d'arc",
         "L'apostrophe de Word (’, U+2019) doit devenir l'apostrophe droite « ' » : voyez `APOSTROPHES`."),
        ("Nʼ Diaye", "n' diaye", "« ʼ » (U+02BC) fait partie des `APOSTROPHES` à ramener à « ' »."),
        ("Saint–Étienne", "saint-etienne", "Le tiret demi-cadratin « – » doit devenir « - » : voyez `TIRETS`."),
        ("Jean‑Pierre", "jean-pierre",
         "Trait d'union insécable (U+2011) : NFKC le transforme en U+2010, qui fait partie des `TIRETS`."),
        ("Анна Ёлкина", "анна елкина",
         "Les autres alphabets gardent leurs lettres : seul le tréma de « Ё » s'en va. Rien ne doit disparaître (si "
         "tout a disparu, c'est `encode(\"ascii\", \"ignore\")`)."),
        ("Ἀθῆναι", "αθηναι", "Le grec perd ses accents et ses esprits, et garde ses lettres."),
        ("", "", "Un texte vide donne une clé vide."),
    ]
    for entree, attendu, conseil in cas:
        quoi = f"cle_recherche({entree!r})"
        obtenu = _exiger_texte(_appel(lambda: cle(entree), quoi), quoi)
        if obtenu != attendu:
            raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {attendu!r}. {conseil}")

    for nom in NOMS_EXEMPLE:
        if _appel(lambda: cle("dupre"), "cle_recherche('dupre')") not in cle(nom):
            raise AssertionError(f"Chercher « dupre » ne trouve pas {nom!r} : `cle_recherche('dupre') in "
                                 f"cle_recherche({nom!r})` doit être vrai.")

    rng = random.Random(123)
    _comparer_au_corrige(cle, "cle_recherche", [_nom_au_hasard(rng) for _ in range(300)], _ref_recherche)


# ============================================================================ Défi 124 : numero_e164


def _doit_refuser(numero, texte: str, conseil: str) -> None:
    quoi = f"numero_e164({texte!r})"
    try:
        resultat = numero(texte)
    except NotImplementedError:
        raise
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de ValueError. Un numéro invalide se refuse TOUJOURS "
                             f"par une ValueError, que l'appelant sait attraper. {conseil}") from None
    raise AssertionError(f"{quoi} devrait lever ValueError, mais renvoie {_court(resultat)}. {conseil}")


def verifier_numero_e164(numero) -> None:
    quoi = "numero_e164('06 12 34 56 78')"
    obtenu = _exiger_texte(_appel(lambda: numero("06 12 34 56 78"), quoi), quoi)
    if obtenu != "+33612345678":
        conseil = ""
        if obtenu.replace(" ", "") in ("0612345678", "33612345678", "+330612345678"):
            conseil = ("Le format E.164, c'est « + », l'indicatif 33, puis le numéro SANS son 0 et sans séparateur : "
                       "« +33612345678 ».")
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de '+33612345678'. {conseil}".rstrip())

    valides = [
        ("0612345678", "+33612345678", ""),
        ("06.12.34.56.78", "+33612345678", "Le point est un séparateur accepté."),
        ("06-12-34-56-78", "+33612345678", "Le tiret est un séparateur accepté."),
        ("06/12/34/56/78", "+33612345678", "La barre oblique est un séparateur accepté."),
        ("06 12 34 56 78", "+33612345678",
         "Ce numéro est séparé par des espaces INSÉCABLES (U+00A0) : retirez tous les blancs (`c.isspace()`), pas "
         "seulement l'espace ordinaire."),
        ("06 12 34 56 78", "+33612345678",
         "Ce numéro est séparé par des espaces fines insécables (U+202F, la typographie française) : retirez tous les "
         "blancs (`c.isspace()`)."),
        ("06\t12 34 56 78", "+33612345678", "La tabulation est un blanc : retirez tous les blancs (`c.isspace()`)."),
        (" 06 12 34 56 78 ", "+33612345678", "Les blancs des bords se retirent aussi."),
        ("01 23 45 67 89", "+33123456789", "Les numéros fixes (01 à 05, 09) sont des numéros français aussi."),
        ("+33 6 12 34 56 78", "+33612345678", "« +33 » suivi des 9 chiffres nationaux est déjà au bon format."),
        ("+33612345678", "+33612345678", "Un numéro déjà au format E.164 se rend tel quel."),
        ("+33 (0)6 12 34 56 78", "+33612345678", "« (0) » est accepté juste après l'indicatif : retirez-le."),
        ("+33(0)612345678", "+33612345678", "« (0) » est accepté juste après l'indicatif, même collé."),
        ("0033 6 12 34 56 78", "+33612345678", "« 0033 » est l'indicatif international écrit sans « + »."),
        ("0033 (0)6 12 34 56 78", "+33612345678", "« (0) » est accepté juste après l'indicatif, « 0033 » compris."),
        ("612345678", "+33612345678",
         "9 chiffres seuls : Excel a pris le numéro pour un nombre et a mangé le 0 du début. C'est un numéro valide."),
        ("7 12 34 56 78", "+33712345678", "9 chiffres seuls (le 0 mangé par Excel), avec des séparateurs."),
    ]
    for entree, attendu, conseil in valides:
        quoi = f"numero_e164({entree!r})"
        obtenu = _exiger_texte(_appel(lambda: numero(entree), quoi, conseil), quoi)
        if obtenu != attendu:
            raise AssertionError(f"{quoi} renvoie {_court(obtenu)} au lieu de {attendu!r}. {conseil}".rstrip())

    obtenus = [_appel(lambda: numero(n), f"numero_e164({n!r})") for n in NUMEROS_EXEMPLE]
    if obtenus != ["+33612345678"] * len(NUMEROS_EXEMPLE):
        raise AssertionError(f"Sur NUMEROS_EXEMPLE, vous obtenez {_court(obtenus, 300)} : les sept formats désignent le "
                             "même numéro, '+33612345678'.")

    chiffres = ("Seuls les chiffres 0 à 9 sont acceptés : `str.isdigit()` et `\\d` (dans une regex) acceptent aussi "
                "les chiffres arabes, pleine chasse ou en exposant. Écrivez `c in \"0123456789\"` ou `[0-9]`.")
    invalides = [
        ("", "Un texte vide n'est pas un numéro."),
        ("   ", "Des blancs seuls ne sont pas un numéro."),
        ("06 12 34 56 7O", "La lettre O n'est pas un zéro : on ne devine JAMAIS un numéro."),
        ("06 12 34 56", "8 chiffres : il en manque un."),
        ("06 12 34 56 78 9", "11 chiffres : il y en a un de trop."),
        ("+33 06 12 34 56 78",
         "Après +33, le 0 national n'a rien à faire : seul « +33 (0)6… » est accepté."),
        ("0033 06 12 34 56 78", "Après 0033, le 0 national n'a rien à faire."),
        ("+32 470 12 34 56", "Un autre indicatif que +33 (ici la Belgique) n'est pas pris en charge : ValueError."),
        ("+44 20 7946 0958", "Un autre indicatif que +33 n'est pas pris en charge : ValueError."),
        ("33612345678", "11 chiffres sans « + » : est-ce +33 ? On ne devine pas."),
        ("012345678", "9 chiffres commençant par 0 : le premier chiffre national ne peut pas être 0."),
        ("00 12 34 56 78", "Le premier chiffre après le 0 national ne peut pas être 0."),
        ("٠٦١٢٣٤٥٦٧٨", chiffres + " Ici : des chiffres arabes (`\"٠٦١٢٣٤٥٦٧٨\".isdigit()` est vrai)."),
        ("０６１２３４５６７８", chiffres + " Ici : des chiffres pleine chasse."),
        ("06 12 34 56 7²", chiffres + " Ici : un « ² » (`\"²\".isdigit()` est vrai)."),
        ("(06) 12 34 56 78", "Les parenthèses ne sont acceptées que pour « (0) », juste après l'indicatif."),
        ("06 12 (0)34 56 78", "« (0) » n'est accepté que juste après l'indicatif."),
        ("06+12 34 56 78", "Un « + » ailleurs qu'au début est une erreur de saisie."),
        ("++33 6 12 34 56 78", "Un seul « + », au début."),
        ("06_12_34_56_78", "Le tiret bas n'est pas un séparateur accepté (liste blanche : blancs . - /)."),
        ("06,12,34,56,78", "La virgule n'est pas un séparateur accepté (liste blanche : blancs . - /)."),
        ("tel: 06 12 34 56 78", "Des lettres : on refuse, on ne devine pas."),
    ]
    for entree, conseil in invalides:
        _doit_refuser(numero, entree, conseil)

    rng = random.Random(124)
    entrees = []
    for _ in range(300):
        national = str(rng.randint(1, 9)) + "".join(rng.choice("0123456789") for _ in range(8))
        separateur = rng.choice(["", " ", ".", "-", "/", " ", " "])
        groupes = [national[0]] + [national[i:i + 2] for i in range(1, 9, 2)]
        corps = separateur.join(groupes)
        debut = rng.choice(["0", "+33" + separateur, "+33" + separateur + "(0)", "0033" + separateur, ""])
        entrees.append(debut + corps)
    _comparer_au_corrige(numero, "numero_e164", entrees, lambda texte: "+33" + "".join(
        c for c in texte.replace("(0)", "") if c in "0123456789")[-9:])


# ============================================================================ Défi 125 : cout_sms


def _cout(cout, message: str, conseil: str = "") -> tuple:
    quoi = f"cout_sms({_court(message, 70)})"
    obtenu = _appel(lambda: cout(message), quoi, conseil)
    if obtenu is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(obtenu, tuple) or len(obtenu) != 2:
        raise AssertionError(f"{quoi} doit renvoyer un tuple `(encodage, segments)`, pas {_court(obtenu)}.")
    return obtenu


def verifier_cout_sms(cout) -> None:
    obtenu = _cout(cout, "Bonjour")
    if obtenu != ("GSM-7", 1):
        conseil = ""
        if isinstance(obtenu[0], str) and obtenu[0].upper().replace("-", "").replace("_", "") == "GSM7" and obtenu[1] == 1:
            conseil = "L'encodage s'écrit exactement « GSM-7 » (et « UCS-2 »)."
        raise AssertionError(f"cout_sms('Bonjour') renvoie {_court(obtenu)} au lieu de ('GSM-7', 1). {conseil}".rstrip())

    a152_euro = "a" * 152 + "€" + "a" * 152
    e66_emoji = "ê" * 66 + "😀" + "ê" * 66
    cas = [
        ("é" * 160, ("GSM-7", 1), "« é », « è », « à », « ù » font partie de l'alphabet GSM : utilisez `GSM_BASE`."),
        ("Ligne 1\nLigne 2", ("GSM-7", 1), "Le retour à la ligne fait partie de l'alphabet GSM (`GSM_BASE`)."),
        ("a" * 160, ("GSM-7", 1), "Jusqu'à 160 unités, un seul segment."),
        ("a" * 161, ("GSM-7", 2), "Au-delà de 160 unités, le message est découpé."),
        ("a" * 306, ("GSM-7", 2), "Deux segments de 153 unités contiennent 306 unités."),
        ("a" * 307, ("GSM-7", 3),
         "Au-delà de 160 unités, chaque segment ne contient plus que 153 unités (7 servent à recoller les morceaux "
         "à l'arrivée) : 307 unités font 3 segments."),
        ("€", ("GSM-7", 1), "« € » fait partie de `GSM_EXTENSION` : le message reste en GSM-7."),
        ("a" * 159 + "€", ("GSM-7", 2),
         "« € » fait partie de `GSM_EXTENSION` et coûte 2 unités : 159 + 2 = 161 unités, donc 2 segments. `len()` "
         "compte les caractères, pas les unités."),
        ("[" * 80, ("GSM-7", 1), "80 crochets (extension) font 160 unités : un seul segment."),
        (a152_euro, ("GSM-7", 3),
         "152 « a » remplissent le 1er segment à une unité près : « € » (2 unités) ne se coupe JAMAIS entre deux "
         "segments, il commence le 2e. Comptez la place restante dans le segment en cours."),
        ("ê", ("UCS-2", 1), "« ê » n'est PAS dans l'alphabet GSM (ni « â », « î », « ô », « û », « ç »…) : UCS-2."),
        ("Jeanne d’Arc", ("UCS-2", 1),
         "L'apostrophe de Word (’) n'est PAS dans l'alphabet GSM : tout le message passe en UCS-2."),
        ("ê" * 70, ("UCS-2", 1), "En UCS-2, un seul segment contient jusqu'à 70 unités."),
        ("ê" * 71, ("UCS-2", 2), "En UCS-2, un seul segment contient au plus 70 unités (pas 160)."),
        ("ê" * 134, ("UCS-2", 2), "En UCS-2, deux segments de 67 unités contiennent 134 unités."),
        ("ê" * 135, ("UCS-2", 3), "En UCS-2, au-delà de 70 unités, chaque segment ne contient que 67 unités."),
        (MESSAGES_EXEMPLE[1], ("UCS-2", 2),
         "Avec une apostrophe typographique, ce message de 99 caractères passe en UCS-2, où un segment ne contient que "
         "70 unités."),
        ("😀" * 35, ("UCS-2", 1), "35 emojis font 70 unités UTF-16 : un seul segment."),
        ("😀" * 36, ("UCS-2", 2),
         "Un emoji (au-delà de U+FFFF) coûte 2 unités en UCS-2 : 36 emojis font 72 unités. `len()` compte les "
         "points de code, pas les unités UTF-16 (`ord(c) > 0xFFFF`)."),
        (e66_emoji, ("UCS-2", 3),
         "66 « ê » remplissent le 1er segment à une unité près : l'emoji (2 unités) ne se coupe JAMAIS entre deux "
         "segments, il commence le 2e."),
        ("é", ("UCS-2", 1),
         "Ce « é » est décomposé (e + accent combinant U+0301), et l'accent combinant n'est pas dans l'alphabet GSM : "
         "ce SMS partira VRAIMENT en UCS-2. Comptez le message tel qu'il partira, sans le normaliser."),
        ("Bonjour\t!", ("UCS-2", 1), "La tabulation n'est pas dans l'alphabet GSM (`GSM_BASE`)."),
    ]
    for message, attendu, conseil in cas:
        obtenu = _cout(cout, message, conseil)
        if obtenu != attendu:
            raise AssertionError(f"cout_sms({_court(message, 70)}) renvoie {_court(obtenu)} au lieu de {attendu}. "
                                 f"{conseil}")

    if [_cout(cout, m) for m in MESSAGES_EXEMPLE] != [("GSM-7", 1), ("UCS-2", 2), ("GSM-7", 1)]:
        raise AssertionError("Les trois MESSAGES_EXEMPLE doivent coûter [('GSM-7', 1), ('UCS-2', 2), ('GSM-7', 1)].")

    quoi = "cout_sms('')"
    try:
        resultat = cout("")
    except NotImplementedError:
        raise
    except ValueError:
        pass
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de ValueError.") from None
    else:
        raise AssertionError(f"{quoi} devrait lever ValueError (un message vide n'est pas un SMS), mais renvoie "
                             f"{_court(resultat)}.")

    rng = random.Random(125)
    alphabets = [GSM_BASE, GSM_BASE + GSM_EXTENSION, GSM_BASE + "êâçô’«»…", "😀👍🏽❤️ab"]
    messages = []
    for _ in range(300):
        alphabet = rng.choice(alphabets)
        messages.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 480))))
    _comparer_au_corrige(cout, "cout_sms", messages, _ref_cout)


# ============================================================================ Défi 126 : reparer_mojibake


class _BoucleSansFin(BaseException):
    """Levée dans VOTRE fonction quand elle tourne trop longtemps (BaseException : un `except Exception` ne l'attrape pas)."""


def _sans_boucle_infinie(fonction, *args, lignes_max: int = 200_000):
    """Appelle fonction(*args) en arrêtant net une boucle qui ne finit jamais (sys.settrace)."""
    code = getattr(fonction, "__code__", None)
    if code is None:
        return fonction(*args)
    fichier, compte = code.co_filename, [0]

    def local(frame, evenement, arg):
        if evenement == "line":
            compte[0] += 1
            if compte[0] > lignes_max:
                raise _BoucleSansFin
        return local

    def global_(frame, evenement, arg):
        return local if frame.f_code.co_filename == fichier else None

    precedent = sys.gettrace()
    sys.settrace(global_)
    try:
        return fonction(*args)
    finally:
        sys.settrace(precedent)


def _reparer(reparer, texte: str, conseil: str = "") -> str:
    quoi = f"reparer_mojibake({texte!r})"
    try:
        resultat = _sans_boucle_infinie(reparer, texte)
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} ne s'arrête jamais : votre boucle tourne sans fin. Un texte en ASCII pur se « répare » "
                             "en lui-même : arrêtez-vous quand la réparation ne CHANGE plus le texte.") from None
    except NotImplementedError:
        raise
    except UnicodeError as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. Un texte qui ne se répare pas se rend TEL QUEL : attrapez "
                             "`UnicodeError`, qui couvre les deux échecs (UnicodeEncodeError en écrivant en cp1252, "
                             "UnicodeDecodeError en relisant en UTF-8).") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None
    return _exiger_texte(resultat, quoi)


def verifier_reparer_mojibake(reparer) -> None:
    sain = "Hélène Dupré"
    obtenu = _reparer(reparer, "HÃ©lÃ¨ne DuprÃ©")
    if obtenu != sain:
        raise AssertionError(f"reparer_mojibake('HÃ©lÃ¨ne DuprÃ©') renvoie {_court(obtenu)} au lieu de 'Hélène Dupré'. "
                             "Réparez avec `texte.encode(\"cp1252\").decode(\"utf-8\")`.")

    obtenu = _reparer(reparer, sain)
    if obtenu != sain:
        conseil = ""
        if "�" in obtenu or len(obtenu) < len(sain):
            conseil = (" Jamais `errors=\"ignore\"` ni `errors=\"replace\"` : on répare TOUT le texte, ou on n'y touche "
                       "pas. Si une étape échoue, rendez le texte tel quel.")
        raise AssertionError(f"reparer_mojibake('Hélène Dupré') renvoie {_court(obtenu)} : ce texte était SAIN, il doit "
                             f"rester tel quel.{conseil}")

    cas = [
        ("Bonjour", "Bonjour", "Un texte en ASCII pur n'a rien à réparer."),
        ("", "", "Un texte vide reste vide."),
        ("Łódź", "Łódź",
         "« Ł » n'existe pas en cp1252 : l'écriture échoue (UnicodeEncodeError). Ce texte n'était pas abîmé de cette "
         "façon : rendez-le tel quel."),
        ("東京", "東京", "Ce texte ne s'écrit pas en cp1252 : rendez-le tel quel."),
        ("Ça coûte 5 €", "Ça coûte 5 €", "Ce texte était sain : ses octets cp1252 ne sont pas de l'UTF-8, on n'y touche pas."),
        ("Prix : 20 â‚¬", "Prix : 20 €",
         "« ‚ » (U+201A) est l'octet 0x82 en cp1252, mais n'existe pas en latin-1 : réparez avec cp1252, pas latin-1."),
        ("â€œOuiâ€\u009d", "“Oui”",
         "Les guillemets typographiques abîmés : « ” » s'écrit E2 80 9D en UTF-8, et 0x9D est un des TROUS de cp1252 "
         "(lu comme U+009D)."),
        ("ðŸ˜€ Merci", "😀 Merci", "Un emoji abîmé (4 octets en UTF-8) se répare de la même façon."),
        ("FranÃƒÂ§ois", "François",
         "Ce texte a été abîmé DEUX fois : recommencez tant que la réparation réussit ET change le texte."),
        ("FranÃƒÆ’Ã‚Â§ois", "François",
         "Ce texte a été abîmé TROIS fois : recommencez tant que la réparation réussit ET change le texte."),
        ("Ã\u0081LVAREZ Ã\u0081ngel", "ÁLVAREZ Ángel",
         "« Á » s'écrit C3 81 en UTF-8, et 0x81 est un des « trous » de cp1252 : Windows l'a lu comme U+0081, que "
         "Python refuse d'écrire en cp1252. Un caractère que cp1252 ne sait pas écrire mais dont le code est inférieur "
         "à 256 redevient l'octet de même valeur : `bytes([ord(c)])`."),
        ("Prix : 20 â\u0082¬", "Prix : 20 €",
         "Ce texte a été lu en latin-1 : « ‚ » y est devenu U+0082, que cp1252 ne sait pas écrire. Règle : un caractère "
         "de code inférieur à 256 que cp1252 refuse redevient l'octet de même valeur (pas seulement les 5 trous)."),
        ("HÃ©lÃ¨ne et Hélène", "HÃ©lÃ¨ne et Hélène",
         "Ce texte mélange une partie abîmée et une partie saine : il ne se répare pas EN ENTIER, on n'y touche pas "
         "(réparer « au mieux » mot par mot, c'est deviner)."),
    ]
    for entree, attendu, conseil in cas:
        obtenu = _reparer(reparer, entree, conseil)
        if obtenu != attendu:
            raise AssertionError(f"reparer_mojibake({entree!r}) renvoie {_court(obtenu)} au lieu de {attendu!r}. {conseil}")

    attendus = ["Hélène Dupré", "Prix : 20 € par mois", "ÁLVAREZ Ángel", "François", "Hélène Dupré"]
    obtenus = [_reparer(reparer, ligne) for ligne in EXPORT_ABIME]
    if obtenus != attendus:
        raise AssertionError(f"Sur EXPORT_ABIME, vous obtenez {_court(obtenus, 300)} au lieu de {attendus}.")

    rng = random.Random(126)
    sains = ["Hélène", "Noël à Zoé", "François", "Prix 20 €", "Œuvre", "naïve", "ÁLVAREZ", "Ölçer", "“Oui”", "ça",
             "déjà-vu", "Łódź", "Ελένη", "東京", "😀", "Straße", "Jürgen", "Ñandú", "ÍÑIGO", "Ýmir", "ÐÆ"]

    entrees = []
    for _ in range(300):
        texte = " ".join(rng.choice(sains) for _ in range(rng.randint(1, 4)))
        for _ in range(rng.choice([0, 1, 1, 2])):
            texte = _abimer(texte)
        entrees.append(texte)
    for texte in entrees:
        attendu = _ref_reparer(texte)
        obtenu = _reparer(reparer, texte)
        if obtenu != attendu:
            raise AssertionError(f"reparer_mojibake({_court(texte, 90)}) renvoie {_court(obtenu)} au lieu de "
                                 f"{_court(attendu)}.")


def _abimer(texte: str) -> str:
    """Ce que fait Windows en lisant de l'UTF-8 comme du cp1252 : les 5 octets « trous » deviennent U+0081…"""
    def lire(octet: int) -> str:
        try:
            return bytes([octet]).decode("cp1252")
        except UnicodeDecodeError:
            return chr(octet)

    return "".join(lire(octet) for octet in texte.encode("utf-8"))


# ============================================================================ Défi 127 : tronquer

_FAMILLE = "👨" + ZWJ + "👩" + ZWJ + "👧"
_COUPLE = "👩" + ZWJ + "❤️" + ZWJ + "👨"


def verifier_tronquer(tronquer) -> None:
    def appel(texte: str, limite: int, conseil: str = "") -> str:
        quoi = f"tronquer({texte!r}, {limite})"
        return _exiger_texte(_appel(lambda: tronquer(texte, limite), quoi, conseil), quoi)

    obtenu = appel("Votre offre expire bientôt", 10)
    if obtenu != "Votre off…":
        conseil = ""
        if obtenu == "Votre off...":
            conseil = "Le « … » est UN caractère (U+2026), pas trois points : il prend moins de place."
        elif obtenu == "Votre offr…":
            conseil = ("Le « … » compte dans la limite : gardez `limite - 1` graphèmes, puis ajoutez « … » (le résultat a "
                       "au plus `limite` graphèmes).")
        elif obtenu == "Votre offre expire bientôt":
            conseil = "Le texte a 26 caractères visibles : il est trop long pour une limite de 10."
        raise AssertionError(f"tronquer('Votre offre expire bientôt', 10) renvoie {_court(obtenu)} au lieu de "
                             f"'Votre off…'. {conseil}".rstrip())

    cas = [
        ("Votre offre expire bientôt", 7, "Votre…",
         "Les 6 premiers graphèmes sont « Votre » suivi d'une espace : retirez les blancs de fin avant d'ajouter "
         "« … » (`rstrip()`)."),
        ("Bonjour", 7, "Bonjour", "Le texte tient dans la limite : rendez-le TEL QUEL, sans « … »."),
        ("Bonjour", 30, "Bonjour", "Le texte tient dans la limite : rendez-le TEL QUEL, sans « … »."),
        ("Merci  ", 10, "Merci  ", "Le texte tient dans la limite : rendez-le TEL QUEL, espaces finales comprises."),
        ("Bonjour", 1, "…", "Avec une limite de 1, il ne reste de la place que pour « … »."),
        (NOMS_EXEMPLE[1], 12, NOMS_EXEMPLE[1],
         "Ce nom vient d'un Mac : 15 points de code (`len`), mais 12 caractères VISIBLES (chaque accent combinant "
         "s'attache à sa lettre). Il tient dans la limite : rendez-le tel quel."),
        (NOMS_EXEMPLE[1], 3, "Hé…",
         "Ne coupez jamais une lettre de son accent combinant : une marque (`unicodedata.category(c)` commence par "
         "« M ») s'attache au point de code précédent."),
        ("👍🏽👍🏽👍🏽", 2, "👍🏽…",
         "Un modificateur de teinte (U+1F3FB à U+1F3FF) s'attache à l'emoji précédent : « 👍🏽 » est UN caractère visible."),
        ("❤️❤️❤️", 2, "❤️…",
         "Le sélecteur de variante U+FE0F est une marque (catégorie « Mn ») : il s'attache au cœur."),
        ("1️⃣2️⃣3️⃣", 2, "1️⃣…",
         "« 1️⃣ » = « 1 » + U+FE0F + U+20E3 (une marque « Me ») : un seul caractère visible."),
        (_FAMILLE * 3, 2, _FAMILLE + "…",
         "« 👨‍👩‍👧 » = 👨 + ZWJ + 👩 + ZWJ + 👧 : le ZWJ s'attache au précédent, ET le point de code qui le suit "
         "s'attache aussi. C'est UN caractère visible."),
        (f"Bravo {_FAMILLE} !", 8, f"Bravo {_FAMILLE}…", "Une famille d'emojis est UN caractère visible."),
        (_COUPLE + "!!", 2, _COUPLE + "…",
         "« 👩‍❤️‍👨 » mélange ZWJ et sélecteur de variante : un seul caractère visible."),
        ("🇫🇷🇧🇪🇨🇭", 2, "🇫🇷…",
         "Deux indicateurs régionaux qui se suivent forment UN drapeau : « 🇫🇷🇧🇪🇨🇭 » fait 3 caractères visibles."),
        ("🇫🇷🇧🇪🇨🇭", 3, "🇫🇷🇧🇪🇨🇭", "« 🇫🇷🇧🇪🇨🇭 » fait 3 caractères visibles (6 points de code) : il tient."),
        ("🇫🇷🇧", 2, "🇫🇷🇧",
         "Un indicateur s'attache au précédent seulement si ce précédent est un indicateur SEUL : « 🇫🇷🇧 », c'est un "
         "drapeau puis un indicateur orphelin, soit 2 caractères visibles."),
        ("🇫🇷🇧🇪", 1, "…", "Avec une limite de 1, il ne reste de la place que pour « … »."),
        ("́abc", 2, "́…",
         "Un accent combinant en tout début de texte n'a pas de « précédent » : il commence un graphème (vérifiez que "
         "la liste n'est pas vide avant `groupes[-1]`)."),
        (ZWJ + "abc", 2, ZWJ + "a…",
         "Un ZWJ en début de texte commence un graphème, et le point de code qui le suit s'y attache."),
    ]
    for texte, limite, attendu, conseil in cas:
        obtenu = appel(texte, limite, conseil)
        if obtenu != attendu:
            raise AssertionError(f"tronquer({texte!r}, {limite}) renvoie {_court(obtenu)} au lieu de {attendu!r}. {conseil}")

    for limite in (0, -3):
        quoi = f"tronquer('Bonjour', {limite})"
        try:
            resultat = tronquer("Bonjour", limite)
        except NotImplementedError:
            raise
        except ValueError:
            continue
        except Exception as exc:
            raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de ValueError.") from None
        raise AssertionError(f"{quoi} devrait lever ValueError (on ne peut même pas écrire « … »), mais renvoie "
                             f"{_court(resultat)}.")

    rng = random.Random(127)
    morceaux = ["a", "b", " ", "é", "é", "ộ", "👍🏽", _FAMILLE, _COUPLE, "🇫🇷", "🇧🇪", "🇨", "❤️",
                "1️⃣", "Bonjour", "  ", ZWJ, "́"]
    entrees = []
    for _ in range(300):
        texte = "".join(rng.choice(morceaux) for _ in range(rng.randint(0, 14)))
        entrees.append((texte, rng.randint(1, 16)))
    _comparer_au_corrige(tronquer, "tronquer", entrees, _ref_tronquer)


SERIE = Serie(
    code="20",
    titre="Le texte sans surprises : comparer, chercher, normaliser, compter, réparer, couper",
    module_eleve="defis.serie_20",
    module_corrige="defis.corrige_serie_20",
    defis=(
        Defi(122, "cle_comparaison", "🟢", "—", verifier_cle_comparaison),
        Defi(123, "cle_recherche", "🟢", "—", verifier_cle_recherche),
        Defi(124, "numero_e164", "🟡", "—", verifier_numero_e164),
        Defi(125, "cout_sms", "🟡", "—", verifier_cout_sms),
        Defi(126, "reparer_mojibake", "🟡", "—", verifier_reparer_mojibake),
        Defi(127, "tronquer", "🔴", "—", verifier_tronquer),
    ),
)

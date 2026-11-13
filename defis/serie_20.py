"""Série 20 — Le texte sans surprises : comparer, chercher, normaliser, compter, réparer, couper.

🔤 VOUS VOULEZ DÉVELOPPER ? UN JOUR, UN TEXTE QUI « A L'AIR JUSTE » CASSERA VOTRE PROGRAMME.

Lundi matin, l'équipe rétention vous transmet cinq tickets :

- « Hélène Dupré existe deux fois dans la base. » Les deux fiches s'affichent pareil, mais `==` dit non ;
- « Quand je cherche "helene", je ne trouve personne. » ;
- « 30 % des SMS partent en erreur : numéro invalide. » Les numéros arrivent sous sept formes ;
- « La facture SMS a doublé ce mois-ci. » Quelqu'un a écrit les messages dans Word ;
- « L'export affiche HÃ©lÃ¨ne, et l'aperçu de la notification coupe un emoji en deux. »

Tous ont la même cause : on a cru qu'un « caractère » était une chose simple. Il y en a au moins quatre :

    les POINTS DE CODE   ce que `len()` compte          « é » vaut 1… ou 2 (e + accent combinant)
    les OCTETS           ce que le disque voit          « é » vaut 2 octets en UTF-8, 1 en Windows-1252
    les UNITÉS           ce que l'opérateur facture     « € » vaut 2 unités dans un SMS, « 😀 » aussi
    les GRAPHÈMES        ce que l'œil voit              « 👨‍👩‍👧 » est UN caractère visible, et 5 points de code

    122 comparer deux noms saisis par des gens différents   123 chercher sans se soucier des accents
    124 ramener sept formats de numéro à un seul             125 savoir combien coûte un SMS AVANT de l'envoyer
    126 réparer un export abîmé, sans abîmer le reste        127 couper un texte sans couper un caractère

Prérequis : la série 00 et le défi 40 (`try` / `except`). Les défis 122 et 123 prolongent le défi 37
(nettoyer une saisie), le défi 126 prolonge le défi 46 (lire un export Excel en cp1252), et le défi 125
chiffre les SMS envoyés dans les séries 12 à 19. Rien à installer : `unicodedata` fait partie de Python.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un texte Python (`str`) est une suite de POINTS DE CODE Unicode : `ord("é")` vaut 233, `chr(233)` vaut
  « é ». Les octets (`bytes`) n'apparaissent qu'à l'entrée et à la sortie : `texte.encode("utf-8")`,
  `octets.decode("utf-8")`. Un décodage avec le mauvais encodage ne plante pas toujours : il donne « Ã© » ;
- « é » s'écrit de deux façons : un seul point de code (U+00E9, forme composée, NFC), ou « e » suivi d'un
  accent combinant (U+0301, forme décomposée, NFD, celle que produit souvent macOS). Elles s'affichent
  pareil et ne sont pas égales. `unicodedata.normalize("NFC", texte)` les met d'accord ;
- NFKC va plus loin que NFC : il ramène aussi les formes « de compatibilité » à la forme simple : ligature
  « ﬁ » → « fi », lettres pleine chasse « Ｍ » → « M », espace insécable → espace ;
- `casefold()` est le `lower()` fait pour COMPARER : `"Straße".lower()` garde le « ß »,
  `"Straße".casefold()` donne « strasse », comme `"STRASSE".casefold()` ;
- `unicodedata.category(c)` dit ce qu'est un caractère : « Lu » lettre majuscule, « Mn » marque (accent
  combinant), « Zs » espace, « Cf » caractère de format INVISIBLE (espace de largeur nulle U+200B, BOM) ;
- `str.isdigit()` et `\\d` dans une regex acceptent tous les chiffres Unicode : « ² », « ٣ » (arabe), « ３ »
  (pleine chasse). Pour des chiffres 0 à 9, écrivez `c in "0123456789"` ou `[0-9]` ;
- un texte abîmé ou illisible ne se « corrige » pas au hasard : on le répare ENTIÈREMENT et sûrement, ou
  on le laisse tel quel, ou on le refuse (`ValueError`). Jamais `errors="ignore"` sur des données clients.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_20 import *
    >>> NOMS_EXEMPLE[0], NOMS_EXEMPLE[1], NOMS_EXEMPLE[0] == NOMS_EXEMPLE[1]
    ('Hélène Dupré', 'Hélène Dupré', False)
    >>> montrer(NOMS_EXEMPLE[1])[:3]                       # voir les points de code un par un
    ['U+0048 LATIN CAPITAL LETTER H', 'U+0065 LATIN SMALL LETTER E', 'U+0301 COMBINING ACUTE ACCENT']
    >>> cle_comparaison(NOMS_EXEMPLE[0]) == cle_comparaison(NOMS_EXEMPLE[1])      # défi 122
    True

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py --defi 122` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_20.py`.

Les tables et les textes d'exemple (`NOMS_EXEMPLE`, `GSM_BASE`, `EXPORT_ABIME`…) et la fonction `montrer`
sont dans `defis/textes.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import unicodedata  # noqa: F401 — utile à tous les défis

from defis.textes import (  # noqa: F401 — tout est utile pour essayer vos fonctions
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
    """Défi 122 🟢 — Deux saisies du même nom doivent donner la même clé.

    « Hélène Dupré » existe deux fois dans la base : une fiche saisie au clavier, l'autre importée depuis
    un Mac. À l'écran, rien ne les distingue ; pour Python, ce sont deux textes différents. Écrivez la
    CLÉ qui sert à comparer des noms (pour repérer les doublons, ou comme clé d'un dictionnaire) :

        >>> {cle_comparaison(nom) for nom in NOMS_EXEMPLE[:5]}
        {'hélène dupré'}
        >>> cle_comparaison("JÜRGEN STRASSE") == cle_comparaison("Jürgen Straße")
        True
        >>> cle_comparaison("Hélène Dupre") == cle_comparaison("Hélène Dupré")   # une autre orthographe
        False

    Les règles, que le correcteur essaie toutes :

    - normalisez avec `unicodedata.normalize("NFKC", texte)` : « é » en un ou deux points de code donne la
      même chose, et les formes de compatibilité aussi (« ﬁ » → « fi », « Ｍａｒｔｉｎ » → « Martin ») ;
    - comparez sans les majuscules avec `casefold()`, pas `lower()` : « ß » doit devenir « ss » ;
    - supprimez les caractères INVISIBLES de format, ceux dont `unicodedata.category(c)` vaut « Cf » : espace
      de largeur nulle (U+200B, copiée d'une page web), BOM (U+FEFF), trait d'union conditionnel (U+00AD) ;
    - remplacez chaque suite de blancs (espaces, tabulations, retours à la ligne, espaces insécables) par
      UNE espace, et retirez ceux du début et de la fin ;
    - gardez les accents : « Dupré » et « Dupre » sont deux orthographes, la clé doit les distinguer (on ne
      fusionne pas deux personnes sur une supposition ; ignorer les accents, c'est pour la RECHERCHE,
      au défi 123) ;
    - la clé reste du texte lisible, en forme composée (on peut l'afficher, la stocker, l'envoyer).

    Ce que vous apprenez : la normalisation Unicode (NFC, NFKC), `casefold`, les catégories de caractères,
    et pourquoi `==` ne suffit pas pour comparer du texte saisi par des humains.

    Indice : trois lignes. `texte = unicodedata.normalize("NFKC", texte).casefold()`, puis un
    `"".join(c for c in texte if ...)` qui écarte la catégorie « Cf », puis `" ".join(texte.split())` :
    `split()` SANS argument coupe sur tous les blancs Unicode, et ignore ceux des bords.
    """
    raise NotImplementedError


def cle_recherche(texte: str) -> str:
    """Défi 123 🟢 — Chercher « helene » doit trouver « Hélène ».

    Les conseillers tapent vite, sans accents, sur un clavier qui n'a pas de « œ ». La recherche doit
    ignorer les accents et la typographie. Écrivez la clé de RECHERCHE : on cherche ensuite
    `cle_recherche(requete) in cle_recherche(nom)`.

        >>> cle_recherche("Hélène Dupré")
        'helene dupre'
        >>> cle_recherche("Cœur d’Alène – Ørsted")
        "coeur d'alene - orsted"
        >>> [nom for nom in ["Hélène Dupré", "Anne Martin"] if cle_recherche("DUPRE") in cle_recherche(nom)]
        ['Hélène Dupré']

    Les règles :

    - tout ce que fait déjà `cle_comparaison` (défi 122) : réutilisez-la, c'est fait pour ;
    - retirez les accents : `unicodedata.normalize("NFD", texte)` sépare chaque lettre de ses accents, qui
      deviennent des caractères « combinants » (`unicodedata.combining(c)` est non nul) : écartez-les ;
    - certaines lettres ne se décomposent pas : « œ », « æ », « ø », « ł », « đ ». Remplacez-les grâce à la
      table `LETTRES_LIEES` (« œ » → « oe ») ;
    - ramenez les apostrophes typographiques (`APOSTROPHES` : ‘ ’ ʼ) à l'apostrophe droite « ' », et les
      tirets (`TIRETS` : ‐ ‑ ‒ – — −) au trait d'union « - » ;
    - les autres alphabets gardent leurs lettres : « Анна Ёлкина » donne « анна елкина » (seul le tréma de
      « Ё » s'en va). Rien ne doit disparaître ;
    - rendez un texte en forme composée (`unicodedata.normalize("NFC", ...)` à la fin).

    Le piège : `texte.encode("ascii", "ignore").decode()` semble marcher sur « Helene ». Sur « Hélène »
    écrit en forme composée, il supprime la LETTRE avec son accent : « Hlne ». Et « Анна » devient « ».

    Ce que vous apprenez : décomposer (NFD) pour séparer les accents, `str.translate` avec
    `str.maketrans`, et la différence entre une clé de comparaison et une clé de recherche.

    Indice : `decompose = unicodedata.normalize("NFD", cle_comparaison(texte))`, un `"".join(...)` qui
    garde les `c` dont `unicodedata.combining(c)` vaut 0, puis `.translate(table)` avec une table construite
    une fois par `str.maketrans({**LETTRES_LIEES, ...})` (un dictionnaire : un caractère → son
    remplacement), et enfin NFC.
    """
    raise NotImplementedError


# ============================================================================ Normaliser une saisie


def numero_e164(texte: str) -> str:
    """Défi 124 🟡 — Ramener sept formats de numéro de téléphone à un seul (E.164), ou refuser.

    30 % des SMS partent en erreur. Les numéros arrivent du formulaire web, de la fiche papier recopiée,
    du téléphone du conseiller, d'un fichier Excel… Le nouvel opérateur n'accepte que le format
    international E.164 : « + », l'indicatif du pays, puis le numéro sans son 0, sans séparateur.

        >>> [numero_e164(n) for n in NUMEROS_EXEMPLE]
        ['+33612345678', '+33612345678', '+33612345678', '+33612345678', '+33612345678', '+33612345678', '+33612345678']
        >>> numero_e164("06 12 34 56 7O")       # la lettre O à la place d'un zéro
        Traceback (most recent call last):
        ValueError: numéro illisible : '06 12 34 56 7O'

    Les règles :

    - séparateurs acceptés, à retirer : tous les blancs (`c.isspace()` : espace, espace insécable U+00A0,
      espace fine insécable U+202F, tabulation…), le point, le tiret et la barre oblique `/` ;
    - « (0) » est accepté SEULEMENT juste après l'indicatif : « +33 (0)6 12 34 56 78 » ;
    - formes reconnues, une fois les séparateurs retirés (N = 9 chiffres, le premier n'est pas 0) :
      « 0N » (national), « +33N », « 0033N », et « N » seul : Excel a pris le numéro pour un nombre et a
      mangé le 0 du début. Le résultat est toujours « +33N » ;
    - tout le reste est REFUSÉ par une `ValueError` (jamais `None`, jamais `False`, jamais une autre
      exception) : un caractère inconnu (lettre, parenthèse ailleurs, « + » au milieu), trop ou pas assez de
      chiffres, « +33 06… » (le 0 n'a rien à faire après +33), un autre indicatif que +33 (« +32 » : la
      Belgique, non prise en charge), un texte vide ;
    - des chiffres 0 à 9 SEULEMENT. `"٠٦١٢٣٤٥٦٧٨".isdigit()` est vrai (ce sont des chiffres arabes), `\\d`
      les accepte aussi dans une regex, et `int("٣")` vaut 3 : l'opérateur, lui, les refuserait.

    On ne devine JAMAIS un numéro : un SMS de relance envoyé au mauvais numéro, c'est un inconnu qui reçoit
    l'offre d'un client (et un client qui ne la reçoit pas).

    Ce que vous apprenez : valider une saisie en refusant franchement (liste blanche plutôt que liste
    noire), les blancs Unicode, et le piège de `isdigit` et de `\\d`.

    Indice : `brut = "".join(c for c in texte if not c.isspace() and c not in ".-/")`, puis le cas de
    « (0) » (`startswith("+33(0)")`), puis refusez tout caractère hors de « +0123456789 ». Ensuite, selon le
    début (`+`, `0033`, `0`, ou 9 chiffres), extrayez les 9 chiffres nationaux et vérifiez-les.
    """
    raise NotImplementedError


# ============================================================================ Compter ce qu'on paie


def cout_sms(message: str) -> tuple[str, int]:
    """Défi 125 🟡 — Combien coûte ce SMS ? (encodage et nombre de segments facturés)

    La facture SMS a doublé ce mois-ci. Un SMS n'est pas « 160 caractères » : tout dépend des caractères.

        >>> [cout_sms(m) for m in MESSAGES_EXEMPLE]
        [('GSM-7', 1), ('UCS-2', 2), ('GSM-7', 1)]

    Les deux premiers messages ne diffèrent que par une apostrophe : « ’ » (typographique, ajoutée par
    Word) n'est pas dans l'alphabet des SMS. Tout le message bascule alors en UCS-2, où un segment ne
    contient que 70 caractères. Sur 200 000 clients, c'est 200 000 SMS de plus.

    Renvoyez `(encodage, segments)` :

    - si CHAQUE caractère est dans `GSM_BASE` ou `GSM_EXTENSION`, l'encodage est « GSM-7 ». Un caractère de
      `GSM_BASE` coûte 1 unité, un caractère de `GSM_EXTENSION` (dont « € ») en coûte 2. Jusqu'à 160 unités :
      1 segment. Au-delà, le message est découpé en segments de 153 unités (7 unités par segment servent à
      recoller les morceaux à l'arrivée) ;
    - sinon, l'encodage est « UCS-2 » : 1 unité par caractère, mais 2 pour un caractère au-delà de U+FFFF
      (les emojis : `ord(c) > 0xFFFF`), car il s'écrit avec deux unités UTF-16. Jusqu'à 70 unités : 1 segment ;
      au-delà, des segments de 67 unités ;
    - on remplit les segments dans l'ordre, et un caractère de 2 unités ne se coupe JAMAIS entre deux
      segments : s'il ne reste qu'une unité de place, il commence le segment suivant (la place perdue est
      perdue). C'est pourquoi `ceil(unités / 153)` ne suffit pas toujours ;
    - on compte le message TEL QU'IL PARTIRA : ne le normalisez pas. Un « é » décomposé (e + U+0301) n'est pas
      dans l'alphabet GSM : ce SMS-là part vraiment en UCS-2 (normaliser, c'est le travail de celui qui écrit
      le message) ;
    - un message vide n'est pas un SMS : `ValueError`.

    Ce que vous apprenez : qu'une « longueur » dépend de ce qu'on mesure (points de code, unités UTF-16,
    unités facturées), les jeux de caractères, et chiffrer un choix technique en euros.

    Indice : construisez la liste des tailles (1 ou 2) de chaque caractère, puis écrivez UNE fonction
    `_segments(tailles, seul, par_segment)` utilisée pour les deux encodages : si `sum(tailles) <= seul`,
    1 ; sinon parcourez les tailles avec un compteur de segments et la place restante dans le segment en
    cours.
    """
    raise NotImplementedError


# ============================================================================ Réparer un texte abîmé


def reparer_mojibake(texte: str) -> str:
    """Défi 126 🟡 — Réparer « HÃ©lÃ¨ne » sans abîmer « Hélène ».

    Quelqu'un a ouvert l'export UTF-8 dans un outil réglé en Windows-1252 (cp1252), puis l'a réenregistré.
    Chaque « é » (2 octets en UTF-8 : C3 A9) est devenu deux caractères : « Ã » (C3) et « © » (A9). C'est ce
    qu'on appelle du mojibake. La bonne nouvelle : ça se défait. Le texte abîmé, réécrit en cp1252, redonne
    les octets d'origine ; relus en UTF-8, ils redonnent le texte.

        >>> [reparer_mojibake(ligne) for ligne in EXPORT_ABIME]
        ['Hélène Dupré', 'Prix : 20 € par mois', 'ÁLVAREZ Ángel', 'François', 'Hélène Dupré']

    Les règles :

    - réparez avec `texte.encode("cp1252").decode("utf-8")` ;
    - si l'une des deux étapes échoue (le texte n'était pas abîmé, ou pas de cette façon), rendez le texte
      TEL QUEL, sans lever d'exception. Les deux échecs existent : `"Hélène".encode("cp1252")` marche mais
      ses octets ne sont pas de l'UTF-8 (UnicodeDecodeError) ; `"Łódź".encode("cp1252")` échoue car « Ł »
      n'existe pas en cp1252 (UnicodeEncodeError). Les deux sont des `UnicodeError` ;
    - jamais `errors="ignore"` ni `errors="replace"` : « Hélène » deviendrait « Hlne » ou « H�l�ne ». On
      répare TOUT le texte, ou on n'y touche pas ;
    - recommencez tant que la réparation réussit ET change le texte : « FranÃƒÂ§ois » a été abîmé deux fois ;
    - les « trous » de cp1252 : cinq octets (0x81, 0x8D, 0x8F, 0x90 et 0x9D) n'y correspondent à aucun
      caractère. Windows les lit quand même, comme les caractères U+0081, U+008D… (c'est aussi ce que fait
      latin-1), et « Á » (C3 81) devient « Ã\\x81 ». Python, lui, refuse d'écrire U+0081 en cp1252. Règle :
      un caractère que cp1252 ne sait pas écrire mais dont le code est inférieur à 256 redevient l'octet de
      même valeur (`bytes([ord(c)])`). Au-delà de 255, c'est un échec : rendez le texte tel quel.

    Ce que vous apprenez : la différence entre texte et octets, ce que fait vraiment un encodage, et
    pourquoi un décodage « réussi » peut quand même être faux.

    Indice : écrivez d'abord `_octets(texte) -> bytes` qui encode caractère par caractère (un `try` autour de
    `c.encode("cp1252")`, et dans le `except UnicodeEncodeError`, la règle des trous). Puis une boucle
    `while True` : `try: repare = _octets(texte).decode("utf-8")`, `except UnicodeError: return texte`, et
    si `repare == texte`, c'est fini.
    """
    raise NotImplementedError


# ============================================================================ Couper sans abîmer


def tronquer(texte: str, limite: int) -> str:
    """Défi 127 🔴 — Couper un texte trop long sans jamais couper un caractère visible.

    L'aperçu d'une notification affiche au plus 30 caractères. L'ancien code faisait `texte[:29] + "…"` :
    il coupait « 👨‍👩‍👧 » en « 👨‍👩 » plus un ZWJ orphelin, « 🇫🇷 » en un « 🇫 » qui s'affiche comme un carré,
    et « é » en « e » quand le nom venait d'un Mac. Un « caractère visible » (un graphème) peut être fait de
    plusieurs points de code :

        >>> famille = "👨" + ZWJ + "👩" + ZWJ + "👧"
        >>> len(famille), len("🇫🇷"), len(NOMS_EXEMPLE[1])
        (5, 2, 15)
        >>> tronquer("Votre offre expire bientôt", 10)
        'Votre off…'
        >>> tronquer("Votre offre expire bientôt", 7)          # « Votre » + espace : l'espace s'en va
        'Votre…'
        >>> tronquer(f"Bravo {famille} !", 8)
        'Bravo 👨\\u200d👩\\u200d👧…'
        >>> tronquer(NOMS_EXEMPLE[1], 12) == NOMS_EXEMPLE[1]    # 15 points de code, mais 12 graphèmes
        True

    Découpez d'abord le texte en graphèmes (règles simplifiées, mais justes pour le français et les emojis
    courants). Chaque point de code commence un nouveau graphème, SAUF :

    - une marque (`unicodedata.category(c)` commence par « M ») : accent combinant (U+0301), sélecteur de
      variante (U+FE0F, dans « ❤️ »), cadre de touche (U+20E3, dans « 1️⃣ ») : elle s'attache au précédent ;
    - un modificateur de teinte (de U+1F3FB à U+1F3FF, dans « 👍🏽 ») : il s'attache au précédent ;
    - le ZWJ (U+200D, `ZWJ`) s'attache au précédent, ET le point de code qui le SUIT s'attache aussi ;
    - deux indicateurs régionaux (de U+1F1E6 à U+1F1FF) qui se suivent forment UN drapeau : un indicateur
      s'attache au précédent si ce précédent est un indicateur seul. « 🇫🇷🇧🇪 », c'est deux drapeaux ;
    - en tout début de texte, il n'y a pas de « précédent » : le point de code commence un graphème.

    Puis :

    - si le texte a au plus `limite` graphèmes, rendez-le TEL QUEL (même ses espaces finales) ;
    - sinon, gardez les `limite - 1` premiers graphèmes, retirez les blancs de fin, et ajoutez « … » (UN seul
      caractère, U+2026, pas trois points). Le « … » compte : le résultat a au plus `limite` graphèmes ;
    - `limite` inférieure à 1 : ValueError (on ne peut même pas écrire « … »).

    Ce que vous apprenez : la différence entre point de code et graphème, les séquences d'emojis, et à
    traduire des règles écrites en un petit automate (l'état « le précédent était un ZWJ »).

    Indice : écrivez `_graphemes(texte) -> list[str]` : une liste `groupes`, et pour chaque `c` du texte,
    soit `groupes[-1] += c` (il s'attache), soit `groupes.append(c)`. Gardez une variable `colle` qui dit si
    le point de code précédent était un ZWJ. Ensuite, `tronquer` tient en quatre lignes.
    """
    raise NotImplementedError

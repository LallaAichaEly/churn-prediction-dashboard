"""Série 41 — Stocker moins sans rien perdre : plages, dictionnaire, bits, varints, Huffman, et un lecteur qui ne plante pas.

🗜️ VOUS VOULEZ DÉVELOPPER ? UN JOUR, ON VOUS DEMANDERA POURQUOI LE MÊME TABLEAU PÈSE 2 GO EN CSV ET 90 MO EN
PARQUET. ICI, VOUS ÉCRIVEZ LES ENCODAGES DE PARQUET, DE PROTOBUF ET DE GZIP, PUIS LE LECTEUR QUI REFUSE
PROPREMENT UN FICHIER ABÎMÉ AU LIEU DE PLANTER À 3 H DU MATIN.

Chaque nuit, l'instantané des 2 millions de clients part en CSV vers l'entrepôt de données et vers le
partenaire qui gère les relances : 1,9 Go par nuit, et la facture de stockage a triplé en un an. Le journal des
événements de l'application (connexions, clics sur « Résilier ») arrive toutes les heures des antennes de zone
rurale, où un transfert sur trois échoue. Et lundi, le fichier du partenaire est arrivé tronqué : le script de
lecture a planté sur un `IndexError` à 3 h du matin, et toute la chaîne de nuit s'est arrêtée. Ousmane,
ingénieur données, a lu que Parquet ferait 20 fois mieux sans rien perdre. Avant de l'adopter, il veut
comprendre COMMENT. Cette série vous fait écrire ses six outils :

    248 plages           « Mensuel » 10 959 fois de suite : une valeur et un nombre (le codage par plages)
    249 dictionnaire     quatre moyens de paiement : chaque valeur devient un petit numéro
    250 empaqueter       un numéro de 0 à 3 dans 2 BITS, et pas dans 12 octets de texte (le « bit-packing »)
    251 encoder_entiers  des horodatages presque triés : les écarts, le zigzag et les varints de Protobuf
    252 codes_huffman    les lettres fréquentes en codes courts : le code de Huffman canonique de gzip
    253 lire_entiers     relire le fichier du 251, et refuser un fichier tronqué, abîmé ou malveillant

Prérequis : la série 00. Le « two sum » du défi 56 (un `dict` comme index), les opérations sur les bits de la
série 40 (défis 243 et 246), les flux qu'on ne lit qu'une fois (série 28) et le tas `heapq` (défis 60 et 146)
l'éclairent, mais tout ce qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un encodage SANS PERTE est un contrat : le lecteur doit retrouver EXACTEMENT ce que l'écrivain a reçu. Le
  même type compris : `True == 1` et `1 == 1.0` en Python, mais une colonne de booléens qui revient en entiers
  est une colonne abîmée. Et les NaN d'une colonne pandas, qui ne sont égaux à rien (pas même à eux-mêmes :
  `x != x` n'est vrai QUE pour NaN), doivent quand même rester des NaN ;
- on gagne de la place en profitant d'une RÉGULARITÉ : des répétitions (plages), peu de valeurs différentes
  (dictionnaire), des petits nombres (bits, varints), des écarts plus petits que les valeurs, des symboles plus
  fréquents que d'autres (Huffman). Sans régularité, rien ne se compresse ;
- un octet, ce sont 8 bits, numérotés de 0 (le bit de POIDS FAIBLE, qui vaut 1) à 7 (qui vaut 128). `x & 0x7F`
  garde les 7 bits de droite, `x >> 7` les retire, `x | 0x80` allume le bit 7, `x << 3` décale de 3 bits vers
  la gauche (multiplie par 8). `b"\\x04\\xd0"` est une suite de deux octets (4 et 208) ; `octets.hex(" ")` les
  affiche (« 04 d0 ») ; `bytearray()` est une suite d'octets qu'on remplit avec `.append(octet)` ;
- `bits_lus(octets)` (defis/compression.py) montre les bits dans l'ordre où Parquet les lit : bit 0 d'abord.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_41 import *
    >>> instantane = instantane_exemple()
    >>> instantane["contrat"][:2], instantane["paiement"][:2], instantane["motif"][:3]
    (['Mensuel', 'Mensuel'], ['Prélèvement', 'Virement'], [nan, nan, nan])
    >>> taille_csv(instantane["contrat"]), taille_csv(instantane["paiement"])      # en octets, dans le CSV
    (158927, 248499)

Quand les six défis sont réussis, Ousmane sait ce que Parquet gagne, et pourquoi :

    >>> plages(instantane["contrat"])                         # 158 927 octets de texte, ou trois plages
    [('Mensuel', 10959), ('Annuel', 5057), ('Biannuel', 3984)]
    >>> len(plages(instantane["motif"]))                      # avec `==`, les NaN seraient 19 916 plages
    3645
    >>> distinctes, codes = dictionnaire(instantane["paiement"])
    >>> distinctes, codes[:6]
    (['Prélèvement', 'Virement', 'Chèque', 'Carte bancaire'], [0, 1, 1, 1, 0, 1])
    >>> len(empaqueter(codes, 2))                             # 2 bits par client : 50 fois moins que le CSV
    5000
    >>> evenements = evenements_exemple()
    >>> fichier = encoder_entiers(evenements)
    >>> len(fichier), taille_csv(evenements), 8 * len(evenements)    # varints, texte, entiers de 8 octets
    (50870, 550000, 400000)
    >>> lire_entiers(fichier) == evenements
    True
    >>> notes = notes_exemple()
    >>> codes = codes_huffman(Counter(notes))
    >>> codes[" "], codes["é"], codes["Q"]                    # « é » : 5 bits au lieu de 16 en UTF-8
    ('000', '11010', '111111011')
    >>> sum(len(codes[c]) for c in notes) // 8, len(notes.encode("utf-8"))
    (83296, 156897)
    >>> lire_entiers(fichier[:-1])                            # le fichier tronqué de lundi
    Traceback (most recent call last):
    ...
    ValueError: fichier tronqué : un nombre commencé n'est pas terminé (octet 50869 manquant)

La colonne des contrats tient en trois plages au lieu de 159 Ko ; les moyens de paiement en 5 000 octets au lieu
de 248 Ko ; les 50 000 horodatages en 51 Ko au lieu de 550 Ko (10,8 fois moins, et 7,9 fois moins que des
entiers de 8 octets) ; les notes du support en 83 Ko au lieu de 157 Ko. Rien n'est perdu : tout se relit à
l'identique. Et le fichier tronqué est refusé avec un message qui dit quoi et où, au lieu d'un `IndexError`.
C'est, à peu de chose près, ce que fait Parquet : plages et dictionnaire pour les colonnes répétitives, bits
collés pour les numéros, écarts et varints pour les entiers, et un compresseur général (gzip, zstd) par-dessus.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les six défis sont indépendants ; l'ordre proposé
   est le plus progressif (le 249 donne les codes que le 250 empaquette, le 253 relit ce qu'écrit le 251).
2. Lancez `python scripts/defis.py --defi 248` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_41.py`.

Les outils communs (`instantane_exemple`, `evenements_exemple`, `notes_exemple`, `taille_csv`, `bits_lus`,
`generer_colonne`, `generer_horodatages`, `CONTRATS`, `PAIEMENTS`, `MOTIFS` et `MINUIT`) sont dans
`defis/compression.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import heapq  # noqa: F401 — heapq.heappush(tas, x), heapq.heappop(tas) : toujours le plus PETIT élément d'abord
import math  # noqa: F401 — math.isnan
from collections import Counter  # noqa: F401 — Counter("abaca") == {"a": 3, "b": 1, "c": 1}

from defis.compression import (  # noqa: F401
    CONTRATS,
    MINUIT,
    MOTIFS,
    PAIEMENTS,
    bits_lus,
    evenements_exemple,
    generer_colonne,
    generer_horodatages,
    instantane_exemple,
    notes_exemple,
    taille_csv,
)

# ============================================================================ Les colonnes répétitives


def plages(valeurs) -> list[tuple]:
    """Défi 248 🟢 — [(valeur, longueur), ...] : le codage par plages (« run-length encoding », RLE).

        >>> plages(["Mensuel", "Mensuel", "Mensuel", "Annuel", "Annuel", "Mensuel"])
        [('Mensuel', 3), ('Annuel', 2), ('Mensuel', 1)]
        >>> plages("aaabcc")
        [('a', 3), ('b', 1), ('c', 2)]
        >>> plages([1, 1, True, 1.0, None, None])            # True n'est pas 1, ni 1.0 : trois types, trois plages
        [(1, 2), (True, 1), (1.0, 1), (None, 2)]
        >>> plages([float("nan"), float("nan"), "Prix", float("nan")])
        [(nan, 2), ('Prix', 1), (nan, 1)]
        >>> plages([])
        []

    Ici : trois « Mensuel » de suite deviennent une seule plage ('Mensuel', 3). Le « Mensuel » de la fin n'est
    pas collé aux autres : c'est une NOUVELLE plage (on ne compte pas les apparitions, on compte les
    répétitions consécutives). Relire, c'est écrire chaque valeur autant de fois que sa longueur : on retrouve
    exactement la colonne. Dans l'instantané, rangé par contrat, les 20 000 contrats tiennent en trois plages.

    Les règles :

    - `valeurs` est n'importe quel itérable (une liste, un texte, un générateur…) : lisez-le UNE seule fois, sans
      `len` ni indice (le correcteur vous passe un flux qui ne se relit pas) ;
    - deux valeurs voisines vont dans la même plage si elles sont du MÊME TYPE et égales (`type(a) is type(b)
      and a == b`), ou si ce sont deux NaN (`a != a and b != b`). Sinon, une nouvelle plage commence ;
    - chaque plage est un tuple (valeur, longueur) : la PREMIÈRE valeur de la plage (l'objet lui-même), et le
      nombre de valeurs (un int >= 1) ;
    - aucune valeur → [] ; la colonne reçue n'est pas modifiée.

    Ce que vous apprenez : la boucle qui garde « l'élément en cours » et le range quand il change (et la
    dernière plage, qu'il faut ranger APRÈS la boucle), et ce que veut dire « sans perte » : `1 == True` en
    Python, mais un encodage qui rend `1` à la place de `True` a perdu quelque chose. `itertools.groupby`
    compare avec `==` : il ferait cette erreur, et couperait la colonne des motifs en 19 916 plages (ses NaN ne
    sont pas égaux entre eux) au lieu de 3 645.

    Indice : `courante, longueur = None, 0` avant la boucle ; `if longueur and meme(courante, v):` ; après la
    boucle, `if longueur: resultat.append((courante, longueur))`.
    """
    raise NotImplementedError


def dictionnaire(valeurs) -> tuple[list, list[int]]:
    """Défi 249 🟡 — (distinctes, codes) : le codage par dictionnaire. Chaque valeur devient le numéro de sa place.

        >>> dictionnaire(["Chèque", "Virement", "Chèque", "Carte bancaire", "Virement"])
        (['Chèque', 'Virement', 'Carte bancaire'], [0, 1, 0, 2, 1])
        >>> dictionnaire([1, True, 1.0, 1, None])
        ([1, True, 1.0, None], [0, 1, 2, 0, 3])
        >>> dictionnaire([float("nan"), "Prix", float("nan")])
        ([nan, 'Prix'], [0, 1, 0])
        >>> dictionnaire([])
        ([], [])

    Ici : « Chèque » est vu en premier, il reçoit le numéro 0 ; « Virement » le 1 ; « Carte bancaire » le 2.
    La colonne devient [0, 1, 0, 2, 1], plus la liste des trois textes, écrits UNE fois. Relire, c'est
    `[distinctes[c] for c in codes]`. Dans l'instantané, 20 000 moyens de paiement (248 Ko de texte)
    deviennent quatre textes et 20 000 numéros de 0 à 3 (que le défi 250 range dans 2 bits chacun).

    Les règles :

    - `distinctes` : les valeurs différentes, dans l'ordre de leur PREMIÈRE apparition (pas triées : on ne
      saurait pas trier une colonne qui mélange textes et None) ; `codes` : pour chaque valeur, dans l'ordre, la
      place de sa valeur dans `distinctes` (de 0 à len(distinctes) - 1) ; la réponse est le tuple (distinctes,
      codes), deux listes ;
    - « la même valeur » suit la règle du défi 248 : même type et égales, ou deux NaN. 1, True et 1.0 ont donc
      trois numéros différents, et tous les NaN un seul ;
    - `valeurs` est n'importe quel itérable de valeurs hashables (textes, nombres, None…) : lisez-le UNE fois ;
    - le correcteur passe 200 000 valeurs dont 20 000 différentes, et COMPTE les comparaisons : chercher chaque
      valeur dans la liste (`distinctes.index(v)`, `v in distinctes`) en fait des milliards. Cherchez dans un
      `dict`, qui répond en une étape.

    Ce que vous apprenez : le dictionnaire comme index (le même principe que les index d'une base de données),
    et deux pièges des clés de `dict` : `{1: "a", True: "b"}` ne contient qu'UNE clé (`hash(True) == hash(1)`
    et `True == 1`), et deux `float("nan")` font deux clés (un NaN n'est égal à rien). Une clé qui contient le
    type, avec un remplaçant unique pour NaN, règle les deux.

    Indice : `cle = (type(v), "NaN" if v != v else v)` (le type départage "NaN" texte et NaN) ;
    `numero = numeros.get(cle)` puis `if numero is None:` (pas `if not numero` : le numéro 0 est « faux »).
    """
    raise NotImplementedError


def empaqueter(codes, largeur: int) -> bytes:
    """Défi 250 🟡 — Les codes collés les uns aux autres, `largeur` bits chacun, bit de poids faible d'abord.

        >>> empaqueter([0, 1, 1, 3, 2], 2)
        b'\\xd4\\x02'
        >>> bits_lus(empaqueter([0, 1, 1, 3, 2], 2))       # 00 10 10 11 01 : chaque code se lit à l'envers
        '00101011 01000000'
        >>> empaqueter([5, 5, 5], 3).hex(" ")                # 9 bits : 2 octets, le second presque vide
        '6d 01'
        >>> empaqueter([], 3)
        b''

    Ici : chaque code prend 2 bits. Le code numéro i occupe les positions 2i et 2i + 1 du flux, et la position p
    est le bit p % 8 de l'octet p // 8, en partant du bit de POIDS FAIBLE. Le code 1 (le deuxième) allume la
    position 2, le code 1 suivant la position 4, le code 3 les positions 6 et 7 : 4 + 16 + 64 + 128 = 212 =
    0xd4. Le code 2 (0b10) allume la position 9, le bit 1 du deuxième octet : 0x02. Cinq codes de 2 bits
    font 10 bits : il faut 2 octets, le dernier entamé est complété par des zéros.

    Les règles (celles du « bit-packing » de Parquet) :

    - le bit j du code numéro i (celui qui vaut `1 << j`) va à la position i · largeur + j du flux ; la position
      p est le bit p % 8 (celui qui vaut `1 << (p % 8)`) de l'octet p // 8 ;
    - la réponse fait exactement ⌈n · largeur / 8⌉ octets (n codes) ; les bits de bourrage valent 0 ; des
      `bytes` (un `bytearray` est accepté) ;
    - `codes` est n'importe quel itérable d'entiers, lu une fois ;
    - erreurs : largeur qui n'est pas un int (un float, un bool) → TypeError, hors de 1 à 64 → ValueError ; un
      code qui n'est pas un int (un float, un bool, un texte) → TypeError ; un code qui ne tient pas dans
      `largeur` bits (négatif, ou >= 2**largeur) → ValueError : le tronquer en silence (`& masque`) rendrait
      une AUTRE valeur à la lecture.

    Ce que vous apprenez : ranger des bits dans des octets (`<<`, `>>`, `&`, `|`), l'ordre « petit-boutiste »
    (little-endian) de Parquet, de x86 et de la plupart des formats, et pourquoi la largeur se calcule :
    `(len(distinctes) - 1).bit_length()` bits suffisent pour numéroter les valeurs distinctes (2 pour 4
    valeurs, 3 pour 5).

    Indice : un `tampon` (les bits pas encore écrits) et `en_attente` (leur nombre) : `tampon |= code <<
    en_attente`, puis tant qu'il y a 8 bits, `sortie.append(tampon & 0xFF)` et `tampon >>= 8`. Ou, plus court :
    un seul grand entier `sum(code << (i * largeur) ...)`, et `.to_bytes(nb_octets, "little")`.
    """
    raise NotImplementedError


# ============================================================================ Les entiers presque triés


def encoder_entiers(valeurs) -> bytes:
    """Défi 251 🟡 — Des entiers presque triés en quelques octets : les écarts, le zigzag et les varints.

        >>> encoder_entiers([1000, 1003, 1001, 1001]).hex(" ")
        '04 d0 0f 06 03 00'
        >>> encoder_entiers([]), encoder_entiers([-1]), encoder_entiers([300])
        (b'\\x00', b'\\x01\\x01', b'\\x01\\xd8\\x04')

    Ici : on écrit d'abord le NOMBRE de valeurs (4 → 04), puis chaque valeur sous forme d'écart avec la
    précédente (la première, avec 0) : 1000, +3, -2, 0. Le zigzag rend chaque écart positif (1000 → 2000,
    3 → 6, -2 → 3, 0 → 0) ; le varint écrit 2000 en deux octets (d0 0f), et chaque petit écart en un seul. Un
    horodatage de l'exemple (1 794 960 000) prend 11 octets en texte et 8 en entier de 64 bits ; son écart avec
    le précédent, presque toujours 1 octet.

    Les règles (celles de Protobuf, de SQLite et des fichiers Parquet) :

    - zigzag(v) = 2v si v >= 0, et -2v - 1 sinon : 0, -1, 1, -2, 2… deviennent 0, 1, 2, 3, 4… ;
    - varint(n), pour n >= 0 : les 7 bits de DROITE de n d'abord ; chaque octet porte 7 bits, plus le bit 0x80
      quand un autre octet SUIT (le dernier octet est donc < 0x80). 0 → 00, 127 → 7f, 128 → 80 01, 2000 → d0 0f ;
    - la réponse : varint(nombre de valeurs), puis varint(zigzag(écart)) pour chaque valeur, où l'écart de la
      première est la valeur elle-même (l'écart avec 0), et celui des suivantes la différence avec la PRÉCÉDENTE
      (pas avec la première) ;
    - `valeurs` est n'importe quel itérable d'entiers, lu UNE fois : on ne connaît leur nombre qu'à la fin, et
      on l'écrit DEVANT (préparez le corps, puis collez la longueur devant) ;
    - erreurs : une valeur qui n'est pas un int (un float, même 3.0, un bool, un texte) → TypeError ; hors des
      entiers signés de 64 bits (de -2**63 à 2**63 - 1) → ValueError.

    Ce que vous apprenez : qu'un nombre n'a pas de taille fixe (un varint prend 1 octet pour 0 à 127, 10 au
    plus), que des données presque triées se résument par leurs écarts, et pourquoi le zigzag : un varint
    d'un nombre NÉGATIF ne s'arrête jamais (`-1 >> 7` vaut -1), et en complément à deux, -1 prendrait 10
    octets.

    Indice : `while n >= 0x80: sortie.append(n & 0x7F | 0x80); n >>= 7` puis `sortie.append(n)` ; un
    `bytearray` pour le corps, et `bytes(tete + corps)` à la fin.
    """
    raise NotImplementedError


# ============================================================================ Les lettres fréquentes


def codes_huffman(frequences: dict) -> dict[str, str]:
    """Défi 252 🔴 — {symbole: code} : le code de Huffman CANONIQUE, des codes courts pour les symboles fréquents.

        >>> codes_huffman({"a": 5, "b": 2, "c": 1, "d": 1})
        {'a': '0', 'b': '10', 'c': '110', 'd': '111'}
        >>> codes_huffman({"a": 1, "b": 1, "c": 2, "d": 2})
        {'a': '00', 'b': '01', 'c': '10', 'd': '11'}
        >>> codes_huffman({"x": 7}), codes_huffman({})
        ({'x': '0'}, {})

    Ici (1er exemple) : c et d (1 fois chacun) sont les plus rares, on les fusionne (poids 2) ; puis b (2) et
    {c, d} (2) : poids 4 ; puis a (5) et {b, c, d} (4). Chaque fusion allonge d'un bit les codes de tous les
    symboles fusionnés : a a 1 bit, b 2, c et d 3. Le texte « aaaaabbcd » coûte 5·1 + 2·2 + 3 + 3 = 15 bits,
    au lieu de 9 × 8 = 72 en ASCII (ou 18 avec 2 bits fixes). Aucun code n'est le début d'un autre : « 0110 »
    se lit « a » puis « 110 » = c, sans séparateur. Dans les notes du support, l'espace a 3 bits, « é » 5 (au
    lieu de 16 en UTF-8), et le texte entier passe de 157 Ko à 83 Ko.

    Les règles :

    1. LES LONGUEURS (Huffman, 1952) : un tas (`heapq`) d'éléments (poids, ordre, symboles). Au départ, une
       feuille par symbole, avec ordre = sa place dans `sorted(frequences)` (0, 1, 2…). Tant qu'il reste plus
       d'un élément : on retire les DEUX plus petits, on allonge d'un bit chacun de leurs symboles, et on remet
       (somme des poids, ordre suivant, symboles des deux) ; les ordres des éléments créés continuent après ceux
       des feuilles (n, n + 1…). Le tas compare d'abord le poids, puis l'ordre : à poids égal, une feuille
       passe avant un élément créé, et un ancien avant un récent. Cette règle fixe les longueurs quand
       plusieurs arbres sont aussi bons (2e exemple : 2, 2, 2, 2 et non 3, 3, 2, 1) ;
    2. LES CODES CANONIQUES (ceux de gzip et de JPEG) : triez les symboles par (longueur, symbole). Le premier
       reçoit le code 0 écrit sur sa longueur ; chaque suivant reçoit le code précédent + 1, décalé vers la
       gauche (`<<`) d'autant de bits que sa longueur dépasse la précédente. Un code est un texte de « 0 » et de
       « 1 », complété à gauche par des zéros jusqu'à sa longueur (`format(code, f"0{longueur}b")`) ;
    3. un seul symbole → son code est « 0 » (un code vide ne s'écrirait pas) ; aucun → {} ;
    4. erreurs : `frequences` qui n'est pas un dict (un `Counter` en est un) → TypeError ; un symbole qui n'est
       pas un texte → TypeError ; une fréquence qui n'est pas un int (un float, un bool) → TypeError, < 1 →
       ValueError ; le dict reçu n'est pas modifié.

    Ce que vous apprenez : l'algorithme glouton le plus célèbre (prouvé optimal : aucun code sans ambiguïté ne
    fait moins de bits), le tas de priorité, et l'idée des codes canoniques : comme ils se déduisent des seules
    LONGUEURS, gzip n'écrit que les longueurs en tête de chaque bloc, pas l'arbre.

    Indice : `longueurs = dict.fromkeys(sorted(frequences), 0)` ; dans la boucle, `for s in a + b:
    longueurs[s] += 1` ; pour les codes, `code <<= longueur - precedente`, puis `code += 1` APRÈS l'avoir écrit.
    """
    raise NotImplementedError


# ============================================================================ Relire un fichier qu'on ne contrôle pas


def lire_entiers(octets) -> list[int]:
    """Défi 253 🔴 — Relit ce qu'écrit `encoder_entiers`, et REFUSE (ValueError) tout fichier tronqué ou abîmé.

        >>> lire_entiers(b"\\x04\\xd0\\x0f\\x06\\x03\\x00")
        [1000, 1003, 1001, 1001]
        >>> lire_entiers(b"\\x00")
        []
        >>> lire_entiers(b"\\x04\\xd0\\x0f\\x06\\x03")          # la dernière valeur manque
        Traceback (most recent call last):
        ...
        ValueError: fichier tronqué : un nombre commencé n'est pas terminé (octet 5 manquant)
        >>> lire_entiers(b"\\x04\\xd0\\x0f\\x06\\x03\\x00\\x2a")  # un octet en trop
        Traceback (most recent call last):
        ...
        ValueError: 1 octet(s) en trop après les 4 valeurs annoncées

    Ici : 04 annonce 4 valeurs ; d0 0f est un varint de deux octets (d0 a le bit 0x80 : un octet suit) qui vaut
    0x50 + 0x0f · 128 = 2000, et zigzag inverse donne 1000 ; puis 06 → +3, 03 → -2, 00 → 0. On additionne les
    écarts : 1000, 1003, 1001, 1001. Le 3e exemple s'arrête au milieu de la 4e valeur : c'est le fichier
    tronqué de lundi. Il faut le DIRE (ValueError, avec un message), pas planter (IndexError) ni rendre trois
    valeurs comme si de rien n'était.

    Les règles :

    - le format est celui du défi 251 : varint(n), puis n varints zigzag d'écarts ; dézigzag(z) = z // 2 si z
      est pair, -(z // 2) - 1 sinon ; la réponse est la liste des n valeurs (des int) ;
    - `octets` est des `bytes` ou un `bytearray` : autre chose (un texte, une liste) → TypeError ;
    - tout fichier mal formé → ValueError (jamais IndexError, MemoryError, ni une réponse incomplète) :
        · rien du tout (b"") : même pas le nombre de valeurs ;
        · un varint qui n'est pas terminé quand le fichier s'arrête (son dernier octet a encore le bit 0x80) ;
        · moins de valeurs que le nombre annoncé ;
        · des octets EN TROP après la dernière valeur ;
        · un varint de plus de 10 octets (64 bits, plus le signe de l'écart, en demandent 10 au plus) : arrêtez-
          vous au 10e octet, sans lire la suite. Le correcteur vous passe 200 000 octets 0xFF et compte les
          lignes exécutées ;
    - ne faites confiance à AUCUNE longueur lue dans le fichier : un fichier de 5 octets peut annoncer 10**15
      valeurs. Ne préparez jamais une liste de la taille annoncée (`[0] * n` lève MemoryError) ; comme chaque
      valeur prend au moins un octet, vous pouvez refuser tout de suite une annonce plus grande que le nombre
      d'octets qui restent.

    Ce que vous apprenez : écrire un lecteur DÉFENSIF pour des données qui viennent de l'extérieur (un
    partenaire, un réseau, un attaquant), faire la différence entre un fichier faux (ValueError, avec ce qui ne
    va pas) et un bug du programme (IndexError), et le principe de toutes les failles de lecture de fichiers :
    une longueur crue sur parole.

    Indice : une fonction `lire_varint(octets, i)` qui renvoie (valeur, i_suivant) : `for _ in range(10):`,
    `if i >= len(octets): raise ValueError(...)`, `valeur |= (octet & 0x7F) << decalage`, et `return` dès
    qu'un octet est < 0x80 ; après la boucle, `raise ValueError`. À la fin, `if i != len(octets)`.
    """
    raise NotImplementedError

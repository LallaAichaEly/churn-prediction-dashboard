"""Série 41 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_41.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours une
dernière plage oubliée après la boucle, un `True` confondu avec `1` par un dictionnaire, un bit rangé du
mauvais côté de l'octet, ou un varint qui ne s'arrête jamais sur un nombre négatif.

Fil rouge de la série : un encodage SANS PERTE est un contrat entre celui qui écrit et celui qui lit. Il gagne
de la place en profitant d'une régularité des données, et il doit rendre EXACTEMENT ce qu'on lui a donné :

1. une colonne rangée répète la même valeur des milliers de fois : on écrit la valeur une fois, et le nombre
   de répétitions (les plages). « Exactement » veut dire aussi : le même TYPE (`True` n'est pas `1`), et les
   NaN d'une colonne pandas, qui ne sont égaux à rien, vont ensemble ;
2. une colonne qui n'a que quelques valeurs différentes se remplace par des numéros (le dictionnaire), qu'on
   cherche dans un `dict` (pas avec `liste.index`, qui relit toute la liste à chaque valeur) ;
3. un numéro de 0 à 3 tient dans 2 bits : on les colle les uns aux autres, bit de poids faible d'abord ;
4. des horodatages presque triés se ressemblent : on écrit les ÉCARTS, petits ; le zigzag rend positifs ceux
   qui reculent, et le varint écrit un petit nombre dans un seul octet ;
5. les lettres fréquentes reçoivent des codes courts (Huffman), et les codes CANONIQUES se reconstruisent à
   partir de leurs seules longueurs : c'est tout ce que gzip écrit en tête de chaque bloc ;
6. un fichier vient de l'extérieur : il peut être tronqué, abîmé, ou malveillant. Le lecteur ne fait confiance
   à aucune longueur, s'arrête au 10e octet d'un varint, et lève une ValueError qui dit ce qui ne va pas, au
   lieu d'un IndexError à 3 h du matin.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter

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

MIN_64, MAX_64 = -(2 ** 63), 2 ** 63 - 1    # les entiers signés de 64 bits (le type BIGINT des bases de données)


def _meme(a, b) -> bool:
    """La règle des défis 248 et 249 : même type ET égaux, ou deux NaN.

    `type(a) is type(b)` sépare True de 1 et 1.0 de 1 (`True == 1 == 1.0` en Python) ; `a != a` n'est vrai que
    pour NaN (et c'est la façon la plus sûre de le reconnaître : `x == math.nan` est TOUJOURS faux).
    """
    return type(a) is type(b) and (a == b or (a != a and b != b))


_NAN = object()   # une clé unique pour tous les NaN : `float("nan")` n'est égal à rien, pas même à lui-même


def _cle(valeur):
    """La clé de dictionnaire qui respecte `_meme` : (type, valeur), et tous les NaN d'un même type ensemble.

    `{1: "a", True: "b"}` ne contient qu'UNE clé (`hash(True) == hash(1)` et `True == 1`) : sans le type dans la
    clé, `True` prendrait le numéro de `1`, et le décodage rendrait `1` à sa place.
    """
    return (type(valeur), _NAN if valeur != valeur else valeur)


# ============================================================================ Défi 248 : plages


def plages(valeurs) -> list[tuple]:
    """Défi 248 🟢 — Le codage par plages (« run-length encoding ») : [(valeur, longueur), ...].

    Une seule lecture de `valeurs` (un flux ne se relit pas) : on garde la plage EN COURS (sa valeur et sa
    longueur), et on la range quand une valeur différente arrive. La dernière plage n'est suivie d'aucune
    valeur différente : il faut la ranger APRÈS la boucle (l'oubli le plus fréquent).

    `itertools.groupby` ne convient pas tel quel : il compare avec `==` (il met 1 et True ensemble, et sépare
    deux NaN). Avec `groupby(valeurs, key=_cle)`, il conviendrait.
    """
    resultat: list[tuple] = []
    courante, longueur = None, 0
    for valeur in valeurs:
        if longueur and _meme(courante, valeur):
            longueur += 1
        else:
            if longueur:
                resultat.append((courante, longueur))
            courante, longueur = valeur, 1
    if longueur:                                  # la dernière plage : personne ne l'a rangée
        resultat.append((courante, longueur))
    return resultat


# ============================================================================ Défi 249 : dictionnaire


def dictionnaire(valeurs) -> tuple[list, list[int]]:
    """Défi 249 🟡 — Le codage par dictionnaire : (valeurs distinctes, numéro de chaque valeur).

    Le `dict` `numeros` donne le numéro d'une valeur déjà vue en une étape, quelle que soit la taille du
    dictionnaire. `distinctes.index(valeur)` donnerait le même résultat, mais en relisant la liste à chaque
    valeur : 200 000 valeurs et 20 000 distinctes, c'est 2 milliards de comparaisons.
    """
    distinctes: list = []
    codes: list[int] = []
    numeros: dict = {}
    for valeur in valeurs:
        cle = _cle(valeur)
        numero = numeros.get(cle)
        if numero is None:                        # pas `if not numero` : le numéro 0 est « faux »
            numero = numeros[cle] = len(distinctes)
            distinctes.append(valeur)
        codes.append(numero)
    return distinctes, codes


# ============================================================================ Défi 250 : empaqueter


def _entier(valeur, nom: str) -> int:
    if isinstance(valeur, bool) or not isinstance(valeur, int):
        raise TypeError(f"{nom} est un entier (int), pas {type(valeur).__name__} : {valeur!r}")
    return valeur


def empaqueter(codes, largeur: int) -> bytes:
    """Défi 250 🟡 — Les codes collés les uns aux autres, `largeur` bits chacun, bit de poids faible d'abord.

    Un « tampon » garde les bits pas encore écrits (les plus anciens à droite) : chaque code s'ajoute à
    GAUCHE de ce qui attend (`code << en_attente`), et chaque fois qu'il y a 8 bits, l'octet de droite part.
    À la fin, le dernier octet entamé part aussi, complété par des zéros (il en faut ⌈n · largeur / 8⌉).

    Une écriture plus courte, juste aussi : `sum(c << (i * largeur) for i, c in enumerate(codes))` est un seul
    grand entier, que `.to_bytes(nb_octets, "little")` découpe dans le bon ordre.
    """
    _entier(largeur, "largeur")
    if not 1 <= largeur <= 64:
        raise ValueError(f"la largeur va de 1 à 64 bits, pas {largeur}")
    limite = 1 << largeur
    sortie = bytearray()
    tampon = en_attente = 0
    for code in codes:
        _entier(code, "chaque code")
        if not 0 <= code < limite:
            raise ValueError(f"le code {code} ne tient pas dans {largeur} bits (de 0 à {limite - 1})")
        tampon |= code << en_attente
        en_attente += largeur
        while en_attente >= 8:
            sortie.append(tampon & 0xFF)
            tampon >>= 8
            en_attente -= 8
    if en_attente:                                # le dernier octet entamé, complété par des zéros
        sortie.append(tampon)
    return bytes(sortie)


# ============================================================================ Défi 251 : encoder_entiers


def _zigzag(v: int) -> int:
    """0, -1, 1, -2, 2… → 0, 1, 2, 3, 4… : un entier POSITIF, petit quand v est proche de 0."""
    return 2 * v if v >= 0 else -2 * v - 1


def _varint(n: int, sortie: bytearray) -> None:
    """n >= 0 en groupes de 7 bits, les bits de poids faible d'abord ; le bit 0x80 dit « un octet suit ».

    Sur un n NÉGATIF, cette boucle ne s'arrêterait jamais si elle testait `n != 0` : `-1 >> 7` vaut -1. C'est
    pour ça qu'on passe TOUJOURS par le zigzag avant.
    """
    while n >= 0x80:
        sortie.append(n & 0x7F | 0x80)
        n >>= 7
    sortie.append(n)


def encoder_entiers(valeurs) -> bytes:
    """Défi 251 🟡 — varint(nombre de valeurs), puis varint(zigzag(écart)) pour chaque valeur.

    On ne connaît le nombre de valeurs qu'à la FIN de la lecture (un flux ne se relit pas) : on écrit le corps
    d'abord, et on colle la longueur devant à la fin. Le premier écart est pris par rapport à 0.
    """
    corps = bytearray()
    precedente = n = 0
    for valeur in valeurs:
        _entier(valeur, "chaque valeur")
        if not MIN_64 <= valeur <= MAX_64:
            raise ValueError(f"{valeur} ne tient pas dans un entier signé de 64 bits")
        _varint(_zigzag(valeur - precedente), corps)
        precedente = valeur
        n += 1
    tete = bytearray()
    _varint(n, tete)
    return bytes(tete + corps)


# ============================================================================ Défi 252 : codes_huffman


def codes_huffman(frequences: dict) -> dict[str, str]:
    """Défi 252 🔴 — {symbole: code} : le code de Huffman CANONIQUE des fréquences.

    1. Les longueurs : un tas de (poids, ordre, symboles du sous-arbre). On fusionne toujours les deux plus
       légers ; chaque fusion allonge d'un bit les codes de tous leurs symboles. `ordre` (les feuilles dans
       l'ordre de `sorted`, puis les nœuds dans l'ordre de leur création) départage les égalités, et empêche
       Python de comparer deux listes.
    2. Les codes : triés par (longueur, symbole), le premier vaut 0, chaque suivant vaut le précédent + 1,
       décalé à gauche quand la longueur augmente. Deux longueurs égales donnent deux codes voisins ; un code
       ne peut pas être le début d'un autre (il n'y a jamais d'ambiguïté à la lecture).
    """
    if not isinstance(frequences, dict):
        raise TypeError(f"les fréquences sont un dict (ou un Counter), pas {type(frequences).__name__}")
    for symbole, frequence in frequences.items():
        if not isinstance(symbole, str):
            raise TypeError(f"un symbole est un texte, pas {type(symbole).__name__} : {symbole!r}")
        _entier(frequence, f"la fréquence de {symbole!r}")
        if frequence < 1:
            raise ValueError(f"la fréquence de {symbole!r} vaut {frequence} : au moins 1")
    symboles = sorted(frequences)
    if len(symboles) <= 1:                        # un seul symbole : un bit quand même (« 0 »), sinon il ne coûte rien
        return {s: "0" for s in symboles}

    longueurs = dict.fromkeys(symboles, 0)
    tas = [(frequences[s], ordre, [s]) for ordre, s in enumerate(symboles)]
    heapq.heapify(tas)
    ordre = len(tas)
    while len(tas) > 1:
        poids_a, _, a = heapq.heappop(tas)
        poids_b, _, b = heapq.heappop(tas)
        for s in a + b:
            longueurs[s] += 1
        heapq.heappush(tas, (poids_a + poids_b, ordre, a + b))
        ordre += 1

    codes: dict[str, str] = {}
    code = precedente = 0
    for s in sorted(symboles, key=lambda s: (longueurs[s], s)):
        code <<= longueurs[s] - precedente        # on allonge le code quand la longueur augmente
        codes[s] = format(code, f"0{longueurs[s]}b")
        code += 1
        precedente = longueurs[s]
    return codes


# ============================================================================ Défi 253 : lire_entiers


def _lire_varint(octets, i: int) -> tuple[int, int]:
    """(valeur, position suivante) ; ValueError si le varint est coupé ou dépasse 10 octets."""
    valeur = decalage = 0
    for _ in range(10):
        if i >= len(octets):
            raise ValueError(f"fichier tronqué : un nombre commencé n'est pas terminé (octet {i} manquant)")
        octet = octets[i]
        i += 1
        valeur |= (octet & 0x7F) << decalage
        if octet < 0x80:
            return valeur, i
        decalage += 7
    raise ValueError(f"fichier abîmé : un nombre de plus de 10 octets (avant l'octet {i})")


def _dezigzag(z: int) -> int:
    return z >> 1 if z % 2 == 0 else -(z >> 1) - 1


def lire_entiers(octets) -> list[int]:
    """Défi 253 🔴 — Relit ce qu'écrit `encoder_entiers`, et refuse proprement un fichier tronqué ou abîmé.

    Chaque valeur prend au moins un octet : un fichier qui annonce plus de valeurs qu'il ne lui reste d'octets
    est faux, et on le dit AVANT de lire quoi que ce soit (on ne prépare jamais une liste de la taille annoncée :
    `[0] * n` avec n = 10**15 lève MemoryError). Les octets en trop après la dernière valeur sont une erreur
    aussi : deux fichiers collés, ou une longueur abîmée.
    """
    if not isinstance(octets, (bytes, bytearray)):
        raise TypeError(f"un fichier binaire se lit en bytes (open(..., 'rb')), pas en {type(octets).__name__}")
    n, i = _lire_varint(octets, 0)
    if n > len(octets) - i:
        raise ValueError(f"le fichier annonce {n} valeurs, mais il ne reste que {len(octets) - i} octets")
    valeurs: list[int] = []
    precedente = 0
    for _ in range(n):
        z, i = _lire_varint(octets, i)
        precedente += _dezigzag(z)
        valeurs.append(precedente)
    if i != len(octets):
        raise ValueError(f"{len(octets) - i} octet(s) en trop après les {n} valeurs annoncées")
    return valeurs


# ============================================================================ Tests (à mettre dans tests/)


def test_plages_et_dictionnaire_gardent_le_type_et_les_nan():
    nan_a, nan_b = float("nan"), float("nan")
    resultat = plages([1, True, 1.0, 1, nan_a, nan_b, None])
    assert [(type(v), n) for v, n in resultat] == [(int, 1), (bool, 1), (float, 1), (int, 1), (float, 2),
                                                   (type(None), 1)]
    distinctes, codes = dictionnaire([1, True, nan_a, 1, nan_b])
    assert distinctes[:2] == [1, True] and type(distinctes[1]) is bool and codes == [0, 1, 2, 0, 2]
    assert plages([]) == [] and dictionnaire([]) == ([], [])


def test_l_instantane_se_compresse():
    instantane = instantane_exemple()
    assert len(plages(instantane["contrat"])) == 3
    distinctes, codes = dictionnaire(instantane["paiement"])
    assert len(distinctes) == 4 and len(empaqueter(codes, 2)) == 5000


def test_empaqueter_bit_de_poids_faible_d_abord():
    assert empaqueter([0, 1, 1, 3, 2], 2) == b"\xd4\x02"
    assert bits_lus(empaqueter([1, 0, 0, 0, 0, 0, 0, 0, 1], 1)) == "10000000 10000000"
    assert empaqueter([], 5) == b""


def test_encoder_et_lire_sont_reciproques():
    evenements = evenements_exemple()
    fichier = encoder_entiers(evenements)
    assert lire_entiers(fichier) == evenements
    assert len(fichier) < taille_csv(evenements) / 7
    assert encoder_entiers([1000, 1003, 1001, 1001]) == b"\x04\xd0\x0f\x06\x03\x00"
    assert lire_entiers(encoder_entiers([MIN_64, MAX_64, 0])) == [MIN_64, MAX_64, 0]


def test_lire_refuse_les_fichiers_abimes():
    for abime in (b"", b"\x04\xd0", b"\x04\xd0\x0f\x06\x03", b"\x04\xd0\x0f\x06\x03\x00\x00", b"\xff" * 11):
        try:
            lire_entiers(abime)
        except ValueError:
            continue
        raise AssertionError(f"{abime!r} aurait dû être refusé")


def test_huffman_canonique_et_optimal():
    codes = codes_huffman({"a": 5, "b": 2, "c": 1, "d": 1})
    assert codes == {"a": "0", "b": "10", "c": "110", "d": "111"}
    assert codes_huffman({"a": 1, "b": 1, "c": 2, "d": 2}) == {"a": "00", "b": "01", "c": "10", "d": "11"}
    assert codes_huffman({"x": 7}) == {"x": "0"}
    notes = notes_exemple()
    codes = codes_huffman(Counter(notes))
    bits = sum(len(codes[c]) for c in notes)
    assert bits < 0.6 * 8 * len(notes.encode("utf-8"))
    tries = sorted(codes.values())
    assert all(not b.startswith(a) for a, b in zip(tries, tries[1:]))   # aucun code n'est le début d'un autre


def test_le_motif_se_compresse_seulement_si_les_nan_vont_ensemble():
    motif = instantane_exemple()["motif"]
    assert len(plages(motif)) < len(motif) / 4
    assert math.isnan(plages(motif)[0][0])

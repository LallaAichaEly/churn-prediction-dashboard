"""Série 41 (stocker moins sans rien perdre) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 248`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (`itertools.groupby` avec une bonne clé, un
  `setdefault`, un seul grand entier et `to_bytes`, un varint écrit bit par bit, un arbre de Huffman à nœuds, un
  lecteur à base de `memoryview`…) ;
- les références du correcteur tiennent face à des vérifications indépendantes : les varints de la documentation
  de Protobuf, le zigzag, l'ordre des bits relu avec `bits_lus`, l'aller-retour écrire → relire, et l'optimalité
  de Huffman (inégalité de Kraft, coût égal à la somme des fusions, comparaison à toutes les longueurs possibles
  sur de petits alphabets) ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (trois plages pour 20 000 contrats, 3 645 plages de motifs au lieu de 19 916, 5 000 octets pour les moyens de
  paiement, 50 870 octets pour 50 000 horodatages, 83 Ko pour 157 Ko de notes).
"""

from __future__ import annotations

import doctest
import heapq
import itertools
import math
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import defis.compression as compression  # noqa: E402
import defis.corrige_serie_41 as corrige  # noqa: E402
from defis.checks_serie_41 import (  # noqa: E402
    SERIE,
    _cas_251,
    _cle,
    _ref_codes,
    _ref_dictionnaire,
    _ref_empaqueter,
    _ref_encoder,
    _ref_lire,
    _ref_longueurs,
    _ref_plages,
    _ref_varint,
    _ref_zigzag,
)
from defis.compression import bits_lus, evenements_exemple, instantane_exemple, notes_exemple, taille_csv  # noqa: E402
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}), DEFIS[nom])


def _echec(nom, objet, *morceaux):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message
    return resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


def _meme(a, b):
    return type(a) is type(b) and (a == b or (a != a and b != b))


# --------------------------------------------------------------------------- Défi 248 : plages


def _plages_bogue(bug):
    def plages(valeurs):
        return _ref_plages(valeurs, bug)
    return plages


@pytest.mark.parametrize("bug,morceau", [
    ("derniere", "DERNIÈRE plage"),
    ("egal", "groupby"),
    ("nan", "deux NaN ne sont jamais égaux"),
    ("identite", "`is`"),
    ("compte", "CONSÉCUTIVES"),
    ("zero", "trop petites de 1"),
])
def test_plages_nomme_l_erreur(bug, morceau):
    _echec("plages", _plages_bogue(bug), morceau)


def test_plages_vide_ou_en_listes_ou_qui_indexe():
    _echec("plages", _plages_bogue("vide"), "[(None, 0)]")

    def en_listes(valeurs):
        return [list(p) for p in corrige.plages(valeurs)]

    _echec("plages", en_listes, "TUPLE")

    def premiere(valeurs):
        resultat, courante, longueur = [], valeurs[0], 0
        for v in valeurs:
            if _meme(v, courante):
                longueur += 1
            else:
                resultat.append((courante, longueur))
                courante, longueur = v, 1
        return resultat + [(courante, longueur)]

    _echec("plages", premiere, "IndexError", "première valeur")

    def avec_indices(valeurs):
        valeurs = list(valeurs) if isinstance(valeurs, str) else valeurs
        resultat, i = [], 0
        while i < len(valeurs):
            j = i
            while j < len(valeurs) and _meme(valeurs[j], valeurs[i]):
                j += 1
            resultat.append((valeurs[i], j - i))
            i = j
        return resultat

    _echec("plages", avec_indices, "flux", "for v in valeurs")


def test_plages_qui_modifie_la_colonne_ou_oublie_le_return():
    def videe(valeurs):
        resultat = corrige.plages(list(valeurs))
        if isinstance(valeurs, list):
            valeurs.clear()
        return resultat

    _echec("plages", videe, "modifié")
    _echec("plages", lambda valeurs: None, "return")


def test_plages_ecrites_par_un_eleve():
    def avec_groupby(valeurs):
        resultat = []
        for _, groupe in itertools.groupby(valeurs, key=_cle):     # groupby, mais avec la bonne clé
            groupe = list(groupe)
            resultat.append((groupe[0], len(groupe)))
        return resultat

    def avec_la_derniere_plage(valeurs):
        resultat = []
        for v in valeurs:
            if resultat:
                derniere, n = resultat[-1]
                deux_nan = isinstance(v, float) and math.isnan(v) and isinstance(derniere, float) and math.isnan(derniere)
                if type(v) is type(derniere) and (v == derniere or deux_nan):
                    resultat[-1] = (derniere, n + 1)
                    continue
            resultat.append((v, 1))
        return resultat

    _reussi("plages", avec_groupby)
    _reussi("plages", avec_la_derniere_plage)


def test_reference_plages_se_relit_a_l_identique():
    alea = random.Random(1)
    for _ in range(200):
        colonne = [alea.choice((1, True, 1.0, None, "a", float("nan"))) for _ in range(alea.randrange(0, 30))]
        relue = [v for v, n in _ref_plages(colonne) for _ in range(n)]
        assert len(relue) == len(colonne) and all(_meme(a, b) for a, b in zip(relue, colonne))
        assert all(not _meme(a[0], b[0]) for a, b in zip(_ref_plages(colonne), _ref_plages(colonne)[1:]))


# --------------------------------------------------------------------------- Défi 249 : dictionnaire


def _dictionnaire_bogue(bug):
    def dictionnaire(valeurs):
        return _ref_dictionnaire(valeurs, bug)
    return dictionnaire


@pytest.mark.parametrize("bug,morceau", [
    ("valeur", "hash(True) == hash(1)"),
    ("nan", "deux `float('nan')` font deux clés"),
    ("un", "commencent à 1"),
    ("trie", "apparition"),
])
def test_dictionnaire_nomme_l_erreur(bug, morceau):
    _echec("dictionnaire", _dictionnaire_bogue(bug), morceau)


def test_dictionnaire_dans_le_desordre_ou_trop_lent():
    def dans_le_desordre(valeurs):
        distinctes, codes = corrige.dictionnaire(valeurs)
        if len(distinctes) < 2:
            return distinctes, codes
        ordre = distinctes[1:] + distinctes[:1]               # le même ensemble, dans un autre ordre
        place = {_cle(v): i for i, v in enumerate(ordre)}
        return ordre, [place[_cle(distinctes[c])] for c in codes]

    _echec("dictionnaire", dans_le_desordre, "PREMIÈRE apparition")

    def avec_index(valeurs):
        distinctes, codes = [], []
        for v in valeurs:
            for i, d in enumerate(distinctes):
                if _meme(d, v):
                    codes.append(i)
                    break
            else:
                codes.append(len(distinctes))
                distinctes.append(v)
        return distinctes, codes

    _echec("dictionnaire", avec_index, "comparaisons", "dict")

    def avec_liste_index(valeurs):
        distinctes, cles, codes = [], [], []
        for v in valeurs:
            cle = _cle(v)
            if cle not in cles:
                cles.append(cle)
                distinctes.append(v)
            codes.append(cles.index(cle))
        return distinctes, codes

    _echec("dictionnaire", avec_liste_index, "comparaisons")


def test_dictionnaire_qui_renvoie_un_set_ou_lit_deux_fois():
    _echec("dictionnaire", lambda valeurs: (set(valeurs), []), "LISTES")

    def deux_fois(valeurs):
        distinctes = []
        for v in valeurs:
            if not any(_meme(v, d) for d in distinctes):
                distinctes.append(v)
        place = {_cle(v): i for i, v in enumerate(distinctes)}
        return distinctes, [place[_cle(v)] for v in valeurs]

    _echec("dictionnaire", deux_fois, "DEUX fois")


def test_dictionnaire_ecrit_par_un_eleve():
    def avec_setdefault(valeurs):
        numeros, distinctes, codes = {}, [], []
        for v in valeurs:
            cle = (type(v), "NaN" if v != v else v)
            numero = numeros.setdefault(cle, len(numeros))
            if numero == len(distinctes):
                distinctes.append(v)
            codes.append(numero)
        return distinctes, codes

    _reussi("dictionnaire", avec_setdefault)


# --------------------------------------------------------------------------- Défi 250 : empaqueter


def _empaqueter_bogue(bug):
    def empaqueter(codes, largeur):
        codes = list(codes)
        corrige.empaqueter(codes, largeur)               # les mêmes validations que le corrigé
        return _ref_empaqueter(codes, largeur, bug)
    return empaqueter


@pytest.mark.parametrize("bug,morceau", [
    ("msb", "POIDS FORT"),
    ("code_inverse", "à l'envers"),
    ("gros_bout", "'little'"),
    ("arrondi_bas", "(n * largeur + 7) // 8"),
    ("octet_par_code", "COLLÉS"),
])
def test_empaqueter_nomme_l_erreur(bug, morceau):
    _echec("empaqueter", _empaqueter_bogue(bug), morceau)


def test_empaqueter_qui_tronque_ou_accepte_un_bool():
    def tronque(codes, largeur):
        return corrige.empaqueter([c & ((1 << largeur) - 1) if isinstance(c, int) else c for c in codes], largeur)

    _echec("empaqueter", tronque, "devrait lever ValueError", "tronquer")

    def accepte_bool(codes, largeur):
        return corrige.empaqueter([int(c) if isinstance(c, bool) else c for c in codes], largeur)

    _echec("empaqueter", accepte_bool, "bool")
    _echec("empaqueter", lambda codes, largeur: list(corrige.empaqueter(codes, largeur)), "bytes")


def test_empaqueter_ecrit_par_un_eleve():
    def grand_entier(codes, largeur):
        if type(largeur) is not int:
            raise TypeError("largeur")
        if not 1 <= largeur <= 64:
            raise ValueError("largeur")
        total = n = 0
        for i, c in enumerate(codes):
            if type(c) is not int:
                raise TypeError("code")
            if c < 0 or c.bit_length() > largeur:
                raise ValueError("code")
            total |= c << (i * largeur)
            n += 1
        return total.to_bytes((n * largeur + 7) // 8, "little")

    def bit_par_bit(codes, largeur):
        if isinstance(largeur, bool) or not isinstance(largeur, int):
            raise TypeError
        if largeur < 1 or largeur > 64:
            raise ValueError
        sortie, position = bytearray(), 0
        for code in codes:
            if isinstance(code, bool) or not isinstance(code, int):
                raise TypeError
            if code < 0 or code >= 2 ** largeur:
                raise ValueError
            for j in range(largeur):
                if position % 8 == 0:
                    sortie.append(0)
                if code >> j & 1:
                    sortie[-1] |= 1 << (position % 8)
                position += 1
        return sortie

    _reussi("empaqueter", grand_entier)
    _reussi("empaqueter", bit_par_bit)


def test_reference_empaqueter_relue_avec_bits_lus():
    alea = random.Random(2)
    for _ in range(300):
        largeur = alea.randrange(1, 65)
        codes = [alea.randrange(2 ** largeur) for _ in range(alea.randrange(0, 20))]
        octets = _ref_empaqueter(codes, largeur)
        flux = bits_lus(octets).replace(" ", "")
        attendu = "".join(format(c, f"0{largeur}b")[::-1] for c in codes)   # chaque code, bit de poids faible d'abord
        assert len(octets) == math.ceil(len(codes) * largeur / 8)
        assert flux[:len(attendu)] == attendu and set(flux[len(attendu):]) <= {"0"}


# --------------------------------------------------------------------------- Défi 251 : encoder_entiers


def _encoder_bogue(bug):
    def encoder_entiers(valeurs):
        valeurs = list(valeurs)
        corrige.encoder_entiers(valeurs)
        return _ref_encoder(valeurs, bug)
    return encoder_entiers


@pytest.mark.parametrize("bug,morceau", [
    ("gros_bout", "7 bits de DROITE"),
    ("drapeau_inverse", "SAUF le dernier"),
    ("abs", "abs(écart)"),
    ("signe", "autre code"),
    ("complement", "complément à deux"),
    ("sans_longueur", "NOMBRE de valeurs"),
    ("longueur_fin", "à la FIN"),
    ("longueur_zigzag", "tel quel"),
    ("ecart_premier", "PREMIÈRE valeur"),
    ("ecart_inverse", "signe inversé"),
    ("sans_ecart", "pas leurs écarts"),
])
def test_encoder_entiers_nomme_l_erreur(bug, morceau):
    _echec("encoder_entiers", _encoder_bogue(bug), morceau)


def test_encoder_entiers_sans_zigzag_tourne_sans_fin():
    def sans_zigzag(valeurs):
        corps, precedente, n = bytearray(), 0, 0
        for v in valeurs:
            e = v - precedente
            while True:                      # un varint qui ne s'arrête qu'à 0 : -1 >> 7 vaut -1
                groupe = e & 0x7F
                e >>= 7
                if e == 0:
                    corps.append(groupe)
                    break
                corps.append(groupe | 0x80)
            precedente, n = v, n + 1
        return bytes([n]) + bytes(corps)

    _echec("encoder_entiers", sans_zigzag, "NÉGATIF", "-1 >> 7")

    def octet_negatif(valeurs):
        corps, precedente, n = bytearray(), 0, 0
        for v in valeurs:
            e = v - precedente
            while e >= 0x80:
                corps.append(e & 0x7F | 0x80)
                e >>= 7
            corps.append(e)
            precedente, n = v, n + 1
        return bytes([n]) + bytes(corps)

    _echec("encoder_entiers", octet_negatif, "zigzag")


def test_encoder_entiers_qui_accepte_un_bool_ou_lit_deux_fois():
    def accepte_bool(valeurs):
        return corrige.encoder_entiers([int(v) if isinstance(v, bool) else v for v in valeurs])

    _echec("encoder_entiers", accepte_bool, "bool")

    def deux_fois(valeurs):
        sum(1 for _ in valeurs)                          # une première lecture pour compter…
        return corrige.encoder_entiers(valeurs)          # … et une seconde pour écrire

    _echec("encoder_entiers", deux_fois, "DEUX fois")


def test_encoder_entiers_ecrit_par_un_eleve():
    def varint(n):
        sortie = []
        while True:
            septet = n % 128
            n //= 128
            sortie.append(septet + (128 if n else 0))
            if not n:
                return sortie

    def encoder_entiers(valeurs):
        liste = []
        for v in valeurs:
            if type(v) is not int:
                raise TypeError("pas un entier")
            if v.bit_length() > 63 and v != -(2 ** 63):
                raise ValueError("hors 64 bits")
            liste.append(v)
        sortie = varint(len(liste))
        for i, v in enumerate(liste):
            ecart = v - (liste[i - 1] if i else 0)
            sortie += varint(ecart * 2 if ecart >= 0 else -ecart * 2 - 1)
        return bytes(sortie)

    _reussi("encoder_entiers", encoder_entiers)


def test_reference_varint_et_zigzag_comme_protobuf():
    # https://protobuf.dev/programming-guides/encoding/ : 1 → 01, 150 → 96 01, 300 → ac 02
    assert [_ref_varint(n).hex(" ") for n in (0, 1, 127, 128, 150, 300, 2 ** 64 - 1)] == \
        ["00", "01", "7f", "80 01", "96 01", "ac 02", "ff ff ff ff ff ff ff ff ff 01"]
    assert [_ref_zigzag(v) for v in (0, -1, 1, -2, 2, 2 ** 63 - 1, -(2 ** 63))] == [0, 1, 2, 3, 4, 2 ** 64 - 2, 2 ** 64 - 1]
    for valeurs in _cas_251():
        assert _ref_lire(_ref_encoder(valeurs)) == valeurs
        assert all(len(_ref_varint(_ref_zigzag(b - a))) <= 10 for a, b in zip([0] + valeurs, valeurs))


# --------------------------------------------------------------------------- Défi 252 : codes_huffman


def _huffman_bogue(bug):
    def codes_huffman(frequences):
        corrige.codes_huffman(frequences)
        return _ref_codes(dict(frequences), bug)
    return codes_huffman


@pytest.mark.parametrize("bug,morceaux", [
    ("neufs_avant", ("optimaux", "feuille sort du tas d'abord")),
    ("recents_avant", ("optimaux", "le plus ancien sort d'abord")),
    ("insertion", ("optimaux", "sorted(frequences)")),
    ("shannon", ("pas optimaux", "Shannon")),
    ("arbre", ("LONGUEURS sont justes", "CANONIQUES")),
    ("arbre_inverse", ("LONGUEURS sont justes", "CANONIQUES")),
    ("tri_frequence", ("par SYMBOLE",)),
    ("plus_un_avant", ("ne vaut pas 0",)),
    ("sans_decalage", ("est le début de celui de", "code <<=")),
    ("vide", ("« 0 »",)),
])
def test_codes_huffman_nomme_l_erreur(bug, morceaux):
    _echec("codes_huffman", _huffman_bogue(bug), *morceaux)


def test_codes_huffman_qui_modifie_les_frequences_ou_compare_des_listes():
    def videe(frequences):
        resultat = corrige.codes_huffman(dict(frequences))
        frequences.clear()
        return resultat

    _echec("codes_huffman", videe, "modifié")

    def sans_ordre(frequences):
        corrige.codes_huffman(frequences)
        tas = [(f, s) for s, f in sorted(frequences.items())]          # (poids, symbole) : pas d'ordre unique
        heapq.heapify(tas)
        while len(tas) > 1:
            a, b = heapq.heappop(tas), heapq.heappop(tas)
            heapq.heappush(tas, (a[0] + b[0], [a[1], b[1]]))           # un poids égal compare un texte à une liste
        return corrige.codes_huffman(frequences)

    _echec("codes_huffman", sans_ordre, "ordre est UNIQUE")


def test_codes_huffman_ecrit_par_un_eleve():
    class Noeud:
        def __init__(self, poids, ordre, symbole=None, gauche=None, droite=None):
            self.poids, self.ordre, self.symbole, self.gauche, self.droite = poids, ordre, symbole, gauche, droite

        def __lt__(self, autre):
            return (self.poids, self.ordre) < (autre.poids, autre.ordre)

    def codes_huffman(frequences):
        if not isinstance(frequences, dict):
            raise TypeError
        if any(not isinstance(s, str) for s in frequences):
            raise TypeError
        if any(type(f) is not int for f in frequences.values()):
            raise TypeError
        if any(f <= 0 for f in frequences.values()):
            raise ValueError
        if len(frequences) == 1:
            return {s: "0" for s in frequences}
        tas = [Noeud(frequences[s], i, s) for i, s in enumerate(sorted(frequences))]
        heapq.heapify(tas)
        ordre = len(tas)
        while len(tas) > 1:
            a, b = heapq.heappop(tas), heapq.heappop(tas)
            heapq.heappush(tas, Noeud(a.poids + b.poids, ordre, gauche=a, droite=b))
            ordre += 1
        longueurs, pile = {}, [(tas[0], 0)] if tas else []
        while pile:
            noeud, profondeur = pile.pop()
            if noeud.symbole is not None:
                longueurs[noeud.symbole] = profondeur
            else:
                pile += [(noeud.gauche, profondeur + 1), (noeud.droite, profondeur + 1)]
        codes, code, avant = {}, 0, None
        for longueur, s in sorted((l, s) for s, l in longueurs.items()):
            if avant is not None:
                code = (code + 1) << (longueur - avant)
            codes[s] = bin(code)[2:].zfill(longueur)
            avant = longueur
        return codes

    _reussi("codes_huffman", codes_huffman)


def _cout_optimal(frequences):
    """Le coût de Huffman par une autre méthode : la somme des poids de toutes les fusions (une liste triée)."""
    poids = sorted(frequences.values())
    total = 0
    while len(poids) > 1:
        a, b = poids.pop(0), poids.pop(0)
        total += a + b
        poids.append(a + b)
        poids.sort()
    return total


def _meilleur_cout_brut(frequences):
    """Le plus petit Σ f·L parmi TOUTES les longueurs de 1 à n - 1 qui respectent Kraft (petits alphabets)."""
    symboles = list(frequences)
    meilleur = None
    for longueurs in itertools.product(range(1, len(symboles)), repeat=len(symboles)):
        if sum(2.0 ** -l for l in longueurs) <= 1:
            cout = sum(frequences[s] * l for s, l in zip(symboles, longueurs))
            meilleur = cout if meilleur is None else min(meilleur, cout)
    return meilleur


def test_reference_huffman_optimale_et_canonique():
    alea = random.Random(3)
    for _ in range(300):
        symboles = alea.sample("abcdefghijklmnop", alea.randrange(2, 12))
        frequences = {s: alea.randrange(1, 30) for s in symboles}
        codes = _ref_codes(frequences)
        longueurs = _ref_longueurs(frequences)
        assert sum(2.0 ** -longueurs[s] for s in symboles) == 1.0                       # Kraft : l'arbre est complet
        assert sum(frequences[s] * len(codes[s]) for s in symboles) == _cout_optimal(frequences)
        tries = sorted(codes.values())
        assert all(not b.startswith(a) for a, b in zip(tries, tries[1:]))
        assert {s: len(c) for s, c in codes.items()} == longueurs
        par_longueur = sorted(symboles, key=lambda s: (longueurs[s], s))
        assert set(codes[par_longueur[0]]) == {"0"}                                     # canonique : le premier vaut 0
        for a, b in zip(par_longueur, par_longueur[1:]):                                 # puis +1, décalé
            assert int(codes[b], 2) == (int(codes[a], 2) + 1) << (longueurs[b] - longueurs[a])
    for _ in range(40):
        frequences = {s: alea.randrange(1, 12) for s in alea.sample("abcdef", alea.randrange(2, 6))}
        assert _cout_optimal(frequences) == _meilleur_cout_brut(frequences)


def test_l_egalite_choisit_bien_entre_deux_arbres_optimaux():
    frequences = {"a": 1, "b": 1, "c": 2, "d": 2}
    assert _ref_longueurs(frequences) == {"a": 2, "b": 2, "c": 2, "d": 2}
    assert _ref_longueurs(frequences, "neufs_avant") == {"a": 3, "b": 3, "c": 2, "d": 1}
    assert _cout_optimal(frequences) == 12 == sum(f * l for f, l in zip((1, 1, 2, 2), (3, 3, 2, 1)))


# --------------------------------------------------------------------------- Défi 253 : lire_entiers


def _lire_bogue(bug):
    def lire_entiers(octets):
        corrige.lire_entiers(octets)                     # les mêmes refus que le corrigé
        return _ref_lire(bytes(octets), bug)
    return lire_entiers


@pytest.mark.parametrize("bug,morceau", [
    ("sans_zigzag", "défaire le zigzag"),
    ("dezigzag_moins", "se trompe de 1"),
    ("dezigzag_plus", "-(z // 2) - 1"),
    ("sans_cumul", "ÉCARTS"),
    ("gros_bout", "7 bits de DROITE"),
])
def test_lire_entiers_nomme_l_erreur(bug, morceau):
    _echec("lire_entiers", _lire_bogue(bug), morceau)


def _lecteur(bornes=True, reste=True, limite=10, prealloue=False, partiel=False, types=True):
    """Un lecteur écrit à la main, avec ou sans chacune de ses protections."""

    def lire_entiers(octets):
        if types and not isinstance(octets, (bytes, bytearray)):
            raise TypeError("bytes attendus")
        i = 0

        def varint():
            nonlocal i
            valeur = decalage = lus = 0
            while True:
                if limite and lus == limite:
                    raise ValueError("trop long")
                if bornes and i >= len(octets):
                    raise ValueError("tronqué")
                octet = octets[i]
                i, lus = i + 1, lus + 1
                valeur |= (octet & 0x7F) << decalage
                decalage += 7
                if octet < 0x80:
                    return valeur

        n = varint()
        valeurs = [0] * n if prealloue else []
        precedente = 0
        for k in range(n):
            if partiel and i >= len(octets):
                break
            z = varint()
            precedente += z // 2 if z % 2 == 0 else -(z // 2) - 1
            if prealloue:
                valeurs[k] = precedente
            else:
                valeurs.append(precedente)
        if reste and i != len(octets):
            raise ValueError("en trop")
        return valeurs

    return lire_entiers


def test_lire_entiers_ecrit_par_un_eleve():
    _reussi("lire_entiers", _lecteur())

    def avec_memoryview(octets):
        if not isinstance(octets, (bytes, bytearray)):
            raise TypeError
        vue = memoryview(octets)
        n, i = corrige._lire_varint(vue, 0)
        if n > len(vue) - i:
            raise ValueError("annonce trop grande")
        valeurs, total = [], 0
        for _ in range(n):
            z, i = corrige._lire_varint(vue, i)
            total += (z >> 1) ^ -(z & 1)
            valeurs.append(total)
        if i < len(vue):
            raise ValueError("reste")
        return valeurs

    _reussi("lire_entiers", avec_memoryview)


def test_lire_entiers_sans_protection():
    _echec("lire_entiers", _lecteur(bornes=False), "IndexError", "fichier faux")
    _echec("lire_entiers", _lecteur(reste=False), "EN TROP")
    _echec("lire_entiers", _lecteur(limite=0), "10 octets")
    _echec("lire_entiers", _lecteur(limite=9), "BIEN FORMÉ")
    _echec("lire_entiers", _lecteur(prealloue=True), "MemoryError", "taille annoncée")
    _echec("lire_entiers", _lecteur(partiel=True), "tronqué de lundi")
    _echec("lire_entiers", _lecteur(types=False), "TypeError")


# --------------------------------------------------------------------------- les outils communs et l'histoire


def test_les_outils_communs():
    instantane = instantane_exemple()
    assert {len(colonne) for colonne in instantane.values()} == {20_000}
    assert instantane["contrat"] == instantane_exemple()["contrat"]
    assert instantane["motif"] is not instantane_exemple()["motif"]
    assert all(isinstance(m, str) or math.isnan(m) for m in instantane["motif"])
    nans = [m for m in instantane["motif"] if not isinstance(m, str)]
    assert len({id(m) for m in nans}) == len(nans)                     # un NOUVEL objet NaN par ligne
    numeros = instantane["numero"]
    assert sorted(numeros) == list(range(1, 20_001))
    evenements = evenements_exemple()
    assert len(evenements) == 50_000 and evenements == evenements_exemple()
    assert max(a - b for a, b in zip(evenements, evenements[1:])) <= 90
    assert taille_csv([]) == 0 and taille_csv(["é"]) == 3
    assert bits_lus(b"") == "" and bits_lus(bytearray(b"\x03")) == "11000000"
    with pytest.raises(TypeError):
        bits_lus("01")
    assert len(compression.generer_colonne(37)) == 37


def test_l_histoire_tient():
    instantane = instantane_exemple()
    assert corrige.plages(instantane["contrat"]) == [("Mensuel", 10959), ("Annuel", 5057), ("Biannuel", 3984)]
    assert len(corrige.plages(instantane["motif"])) == 3645
    assert sum(1 for _ in itertools.groupby(instantane["motif"])) == 19916        # avec `==` : les NaN séparés
    distinctes, codes = corrige.dictionnaire(instantane["paiement"])
    assert len(distinctes) == 4 and len(corrige.empaqueter(codes, 2)) == 5000
    assert taille_csv(instantane["paiement"]) == 248499 and taille_csv(instantane["contrat"]) == 158927
    evenements = evenements_exemple()
    fichier = corrige.encoder_entiers(evenements)
    assert len(fichier) == 50870 and taille_csv(evenements) == 550000
    assert sum(b < a for a, b in zip(evenements, evenements[1:])) == 1733
    assert corrige.lire_entiers(fichier) == evenements
    notes = notes_exemple()
    codes = corrige.codes_huffman(Counter(notes))
    assert sum(len(codes[c]) for c in notes) // 8 == 83296 and len(notes.encode("utf-8")) == 156897
    assert (codes[" "], codes["é"], codes["Q"]) == ("000", "11010", "111111011")


# --------------------------------------------------------------------------- le correcteur lui-même


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(248, 254))
    assert any(s is SERIE for s in series())            # enregistrée dans defis/verif.py


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_41 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(compression)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_41 import *` (dans les exemples) doit donner les objets du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 35 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    colonne = [1, True, float("nan"), "a", "a"]
    copie = list(colonne)
    corrige.plages(colonne)
    corrige.dictionnaire(colonne)
    assert len(colonne) == 5 and all(a is b for a, b in zip(colonne, copie))
    codes = [0, 1, 2]
    corrige.empaqueter(codes, 2)
    corrige.encoder_entiers(codes)
    assert codes == [0, 1, 2]
    frequences = {"b": 1, "a": 2}
    corrige.codes_huffman(frequences)
    assert list(frequences.items()) == [("b", 1), ("a", 2)]


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(ValueError):
        corrige.empaqueter([4], 2)
    with pytest.raises(TypeError):
        corrige.empaqueter([True], 1)
    with pytest.raises(ValueError):
        corrige.empaqueter([1], 65)
    with pytest.raises(TypeError):
        corrige.encoder_entiers([3.0])
    with pytest.raises(ValueError):
        corrige.encoder_entiers([2 ** 63])
    with pytest.raises(ValueError):
        corrige.codes_huffman({"a": 0})
    with pytest.raises(TypeError):
        corrige.codes_huffman({1: 2})
    with pytest.raises(ValueError):
        corrige.lire_entiers(b"")
    with pytest.raises(ValueError):
        corrige.lire_entiers(b"\xff" * 11)
    with pytest.raises(TypeError):
        corrige.lire_entiers("\x00")


def test_les_boucles_sans_fin_sont_arretees_et_expliquees():
    def tas_qui_ne_raccourcit_pas(frequences):
        corrige.codes_huffman(frequences)
        tas = [(f, i) for i, f in enumerate(frequences.values())]
        heapq.heapify(tas)
        while len(tas) > 1:                  # on retire un élément… et on le remet
            heapq.heappush(tas, heapq.heappop(tas))
        return corrige.codes_huffman(frequences)

    _echec("codes_huffman", tas_qui_ne_raccourcit_pas, "trop lent", "RETIRE deux éléments")

    def position_qui_n_avance_pas(octets):
        i = valeur = 0
        while octets[i] >= 0x80:             # i n'avance jamais : boucle sans fin sur un varint de 2 octets
            valeur = valeur * 128 + (octets[i] & 0x7F)
        return corrige.lire_entiers(octets)

    _echec("lire_entiers", position_qui_n_avance_pas, "trop lent", "i += 1")

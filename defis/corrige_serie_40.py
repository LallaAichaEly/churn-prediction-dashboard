"""Série 40 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_40.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours un
décalage de un (compter les zéros au lieu de la position du premier 1, `bisect_right` au lieu de
`bisect_left`), une liste partagée sans le savoir (`[[0] * largeur] * profondeur`), ou un logarithme dans la
mauvaise base.

Fil rouge de la série : quand on ne peut pas tout garder, on garde un RÉSUMÉ, dont l'erreur est connue à
l'avance, et qui se FUSIONNE (trois serveurs, sept jours). Chaque structure échange un peu d'exactitude contre
énormément de mémoire, mais jamais n'importe comment :

1. on DIMENSIONNE avant de coder : n clés et un taux d'erreur accepté donnent la taille (bits) et le nombre de
   positions (k). 40 millions de numéros à 0,1 % tiennent dans 72 Mo, au lieu des 3 Go du `set` ;
2. un filtre de Bloom peut se tromper dans UN seul sens : il peut dire « déjà contacté » à tort (un faux
   positif : un client n'aura pas son SMS), jamais « jamais contacté » à tort (un doublon). On choisit ce sens ;
3. Misra-Gries trouve à coup sûr tout ce qui dépasse n / k avec k - 1 compteurs : rien de fréquent ne
   s'échappe, et la deuxième lecture (si on peut la faire) donne les comptes exacts ;
4. Count-Min ne sous-estime jamais : chaque case cumule la vraie valeur plus des collisions, et le minimum des
   lignes garde la case la moins polluée ;
5. HyperLogLog compte des millions de visiteurs différents dans 16 Ko, avec 0,8 % d'erreur, et deux résumés se
   fusionnent par un MAXIMUM : c'est ce qui permet d'avoir la semaine à partir des jours ;
6. le hachage cohérent fait qu'ajouter un serveur ne déplace que la part qui lui revient (1 / 5), au lieu de
   presque tout (`% n`).

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import bisect
import math
import numbers

from defis.esquisses import (  # noqa: F401
    JOURS,
    SERVEURS,
    Dimension,
    appels_exemple,
    campagne_exemple,
    clients_exemple,
    empreinte,
    envois_exemple,
    generer_appels,
    generer_numeros,
    generer_visiteurs,
    taux_theorique,
    visites_exemple,
)


def _entier(valeur, nom: str, minimum: int) -> int:
    """`valeur` si c'est un int (pas un bool) >= minimum ; TypeError ou ValueError sinon."""
    if isinstance(valeur, bool) or not isinstance(valeur, int):
        raise TypeError(f"{nom} est un entier, pas {type(valeur).__name__} : {valeur!r}")
    if valeur < minimum:
        raise ValueError(f"{nom} vaut au moins {minimum} : {valeur!r}")
    return valeur


def _texte(cle, nom: str = "la clé") -> str:
    if not isinstance(cle, str):
        raise TypeError(f"{nom} est un texte, pas {type(cle).__name__} : {cle!r}")
    return cle


def _proportion(valeur, nom: str) -> float:
    """Un nombre strictement entre 0 et 1 ; TypeError (pas un nombre) ou ValueError (0, 1, NaN, hors bornes)."""
    if isinstance(valeur, bool) or not isinstance(valeur, numbers.Real):
        raise TypeError(f"{nom} est un nombre, pas {type(valeur).__name__} : {valeur!r}")
    if not 0 < valeur < 1:      # NaN rend les deux comparaisons fausses : il est refusé ici aussi
        raise ValueError(f"{nom} est strictement entre 0 et 1 : {valeur!r}")
    return float(valeur)


# ============================================================================ défi 242 : dimensionner


def dimensionner(n: int, taux: float) -> Dimension:
    _entier(n, "n", 1)
    taux = _proportion(taux, "taux")
    # m = -n·ln(p) / (ln 2)² : le LOGARITHME NATUREL (math.log), pas math.log10, qui donne un filtre 2,3 fois trop
    # petit. Et math.ceil, pas int() : int() arrondit vers le bas, et un bit de moins peut dépasser le taux promis.
    bits = math.ceil(-n * math.log(taux) / math.log(2) ** 2)
    # k = (m / n)·ln 2, arrondi ; au moins 1 (avec un taux de 90 %, la formule donne 0, et un filtre sans aucune
    # position répondrait « présent » à tout).
    k = max(1, round(bits / n * math.log(2)))
    return Dimension(bits, k, taux_theorique(bits, k, n))


# ============================================================================ défi 243 : FiltreBloom


class FiltreBloom:
    def __init__(self, bits: int, k: int) -> None:
        self.bits = _entier(bits, "bits", 1)
        self.k = _entier(k, "k", 1)
        # Un bytearray : 8 bits par octet, 72 Mo pour 575 millions de bits. Créé ICI, dans __init__ : écrit au
        # niveau de la classe, il serait partagé par tous les filtres.
        self.octets = bytearray((bits + 7) // 8)

    def positions(self, cle: str) -> list[int]:
        _texte(cle)
        # Le double hachage de Kirsch et Mitzenmacher (2006) : deux empreintes suffisent pour fabriquer k positions
        # aussi bonnes que k fonctions de hachage indépendantes. Pas hash() : il change à chaque lancement de Python,
        # et un filtre enregistré sur disque ne reconnaîtrait plus rien après un redémarrage.
        h1, h2 = empreinte(cle, 0), empreinte(cle, 1)
        return [(h1 + i * h2) % self.bits for i in range(self.k)]

    def ajouter(self, cle: str) -> None:
        for p in self.positions(cle):            # positions() valide la clé AVANT de toucher au moindre bit
            self.octets[p // 8] |= 1 << (p % 8)  # le bit p % 8 (en partant du bit de poids faible) de l'octet p // 8

    def __contains__(self, cle: str) -> bool:
        return all(self.octets[p // 8] >> (p % 8) & 1 for p in self.positions(cle))

    def remplissage(self) -> float:
        # Divisé par self.bits, pas par len(self.octets) * 8 : les bits de bourrage du dernier octet ne sont pas à
        # nous. int.bit_count() (Python 3.10+) compte les 1 d'un entier.
        return sum(octet.bit_count() for octet in self.octets) / self.bits

    def union(self, autre: FiltreBloom) -> FiltreBloom:
        if not isinstance(autre, FiltreBloom):
            raise TypeError(f"union() attend un FiltreBloom, pas {type(autre).__name__}")
        if (self.bits, self.k) != (autre.bits, autre.k):
            # Avec une autre taille ou un autre k, les mêmes bits ne veulent pas dire la même chose.
            raise ValueError(f"filtres incompatibles : ({self.bits}, {self.k}) et ({autre.bits}, {autre.k})")
        resultat = type(self)(self.bits, self.k)      # type(self) : une sous-classe reste une sous-classe
        resultat.octets = bytearray(a | b for a, b in zip(self.octets, autre.octets))
        return resultat


# ============================================================================ défi 244 : frequents


def frequents(flux, k: int) -> dict:
    _entier(k, "k", 2)
    compteurs: dict = {}
    for x in flux:                      # UNE lecture : le flux peut être un générateur
        if x in compteurs:
            compteurs[x] += 1
        elif len(compteurs) < k - 1:    # k - 1 compteurs au plus
            compteurs[x] = 1
        else:
            # Plus de place : on retire 1 à TOUS les compteurs, et x « paie » sa part sans entrer. Chaque fois, k
            # occurrences disparaissent (les k - 1 décomptées et x) : cela ne peut pas arriver plus de n / k fois.
            # D'où la garantie : un élément vu plus de n / k fois survit forcément.
            for cle in list(compteurs):  # list() : on ne supprime pas dans un dict qu'on est en train de parcourir
                compteurs[cle] -= 1
                if compteurs[cle] == 0:
                    del compteurs[cle]
    # sorted est STABLE, même avec reverse=True : à égalité, l'ordre du dict est conservé.
    return dict(sorted(compteurs.items(), key=lambda paire: paire[1], reverse=True))


# ============================================================================ défi 245 : CompteurMin


class CompteurMin:
    def __init__(self, largeur: int, profondeur: int) -> None:
        self.largeur = _entier(largeur, "largeur", 1)
        self.profondeur = _entier(profondeur, "profondeur", 1)
        # UNE NOUVELLE liste par ligne. `[[0] * largeur] * profondeur` répéterait la MÊME liste : ajouter dans une
        # ligne ajouterait dans toutes, et chaque estimation serait multipliée par la profondeur.
        self.tableau = [[0] * largeur for _ in range(profondeur)]
        self.total = 0

    @classmethod
    def pour_erreur(cls, epsilon: float, delta: float) -> CompteurMin:
        epsilon = _proportion(epsilon, "epsilon")
        delta = _proportion(delta, "delta")
        # Cormode et Muthukrishnan (2005) : largeur = ⌈e / ε⌉ et profondeur = ⌈ln(1 / δ)⌉ garantissent
        # estimation <= vrai compte + ε·total, avec une probabilité d'au moins 1 - δ.
        return cls(math.ceil(math.e / epsilon), math.ceil(math.log(1 / delta)))

    def colonnes(self, cle: str) -> list[int]:
        _texte(cle)
        # Une graine PAR LIGNE : avec la même graine partout, toutes les lignes subiraient les mêmes collisions,
        # et le minimum ne servirait à rien.
        return [empreinte(cle, ligne) % self.largeur for ligne in range(self.profondeur)]

    def ajouter(self, cle: str, n: int = 1) -> None:
        colonnes = self.colonnes(cle)          # valide la clé
        _entier(n, "n", 0)                     # un compte négatif casserait la garantie « jamais en dessous »
        for ligne, colonne in enumerate(colonnes):
            self.tableau[ligne][colonne] += n
        self.total += n

    def estimer(self, cle: str) -> int:
        # Chaque case contient le vrai compte PLUS celui des clés qui partagent sa colonne : toutes surestiment.
        # Le MINIMUM est la moins polluée (le maximum ou la moyenne gardent la pire pollution).
        return min(self.tableau[ligne][colonne] for ligne, colonne in enumerate(self.colonnes(cle)))


# ============================================================================ défi 246 : HyperLogLog


def _alpha(m: int) -> float:
    """La constante de correction de Flajolet et al. (2007) pour m registres."""
    return {16: 0.673, 32: 0.697, 64: 0.709}.get(m, 0.7213 / (1 + 1.079 / m))


class HyperLogLog:
    def __init__(self, precision: int = 14) -> None:
        _entier(precision, "precision", 4)
        if precision > 16:
            raise ValueError(f"precision vaut au plus 16 : {precision!r}")
        self.precision = precision
        self.registres = bytearray(1 << precision)   # 2**14 registres = 16 Ko, quel que soit le nombre de visiteurs

    def ajouter(self, cle: str) -> None:
        x = empreinte(_texte(cle))
        p = self.precision
        j = x >> (64 - p)                    # les p bits de GAUCHE choisissent le registre
        w = x & ((1 << (64 - p)) - 1)        # les 64 - p bits restants servent à compter
        # Le rang du premier 1 en partant de la gauche, sur 64 - p bits : (64 - p) - w.bit_length() zéros, PLUS 1.
        # bin(w) ne convient pas tel quel : il supprime les zéros de tête, et le premier 1 y serait toujours en tête.
        rang = (64 - p) - w.bit_length() + 1
        if rang > self.registres[j]:
            self.registres[j] = rang

    def estimation(self) -> float:
        m = len(self.registres)
        # La moyenne HARMONIQUE des 2**registre : un registre exceptionnellement haut (un coup de chance) ne pèse
        # presque rien, alors qu'il ferait exploser une moyenne ordinaire.
        brute = _alpha(m) * m * m / sum(2.0 ** -r for r in self.registres)
        vides = self.registres.count(0)
        if brute <= 2.5 * m and vides:
            # Peu de visiteurs : beaucoup de registres sont encore vides, et compter les vides (« linear counting »)
            # est bien plus précis que la formule brute (qui annoncerait ~0,7·m visiteurs pour 10).
            return m * math.log(m / vides)
        return brute

    def fusionner(self, autre: HyperLogLog) -> HyperLogLog:
        if not isinstance(autre, HyperLogLog):
            raise TypeError(f"fusionner() attend un HyperLogLog, pas {type(autre).__name__}")
        if self.precision != autre.precision:
            raise ValueError(f"précisions différentes : {self.precision} et {autre.precision}")
        resultat = type(self)(self.precision)
        # Le MAXIMUM, registre par registre : exactement ce qu'aurait donné un seul résumé qui aurait vu les deux
        # flux. Un visiteur venu les deux jours tombe deux fois dans le même registre avec le même rang : il ne
        # compte qu'une fois. Additionner compterait deux fois.
        resultat.registres = bytearray(max(a, b) for a, b in zip(self.registres, autre.registres))
        return resultat


# ============================================================================ défi 247 : Anneau


class Anneau:
    def __init__(self, serveurs=(), repliques: int = 100) -> None:
        self.repliques = _entier(repliques, "repliques", 1)
        self._points: list[tuple[int, str]] = []    # (position, serveur), TRIÉS par position
        self._positions: list[int] = []             # les seules positions, pour bisect
        self._serveurs: list[str] = []
        for serveur in serveurs:
            self.ajouter(serveur)

    def _ranger(self, points: list[tuple[int, str]]) -> None:
        self._points = sorted(points)
        self._positions = [position for position, _ in self._points]

    def ajouter(self, serveur: str) -> None:
        _texte(serveur, "le nom du serveur")
        if not serveur:
            raise ValueError("le nom du serveur est vide")
        if serveur in self._serveurs:
            raise ValueError(f"le serveur {serveur!r} est déjà sur l'anneau")
        # `repliques` points par serveur (des « nœuds virtuels ») : avec un seul point, un serveur peut hériter d'un
        # arc immense par malchance ; avec 100, les arcs se compensent et chacun reçoit à peu près sa part.
        nouveaux = [(empreinte(f"{serveur}#{i}"), serveur) for i in range(self.repliques)]
        self._ranger(self._points + nouveaux)
        self._serveurs.append(serveur)

    def retirer(self, serveur: str) -> None:
        _texte(serveur, "le nom du serveur")
        if serveur not in self._serveurs:
            raise ValueError(f"le serveur {serveur!r} n'est pas sur l'anneau")
        self._ranger([point for point in self._points if point[1] != serveur])
        self._serveurs.remove(serveur)

    def serveur_pour(self, cle: str) -> str:
        position = empreinte(_texte(cle))
        if not self._points:
            raise ValueError("aucun serveur sur l'anneau")
        # Le premier point À une position >= celle de la clé : bisect_left (bisect_right sauterait un point situé
        # pile sur la clé). Après le dernier point, on repart du premier : c'est un ANNEAU (d'où le modulo).
        i = bisect.bisect_left(self._positions, position)
        return self._points[i % len(self._points)][1]

    def __len__(self) -> int:
        return len(self._serveurs)


# ============================================================================ tests (pytest)


def test_dimensionner():
    d = dimensionner(40_000_000, 0.001)
    assert (d.bits, d.k) == (575_103_503, 10) and d.octets < 72 * 2 ** 20   # 72 Mo au lieu de plus de 3 Go
    assert dimensionner(10, 0.9).k == 1                                     # jamais 0 position
    assert dimensionner(1000, 0.01).taux_reel <= 0.0101


def test_filtre_bloom_sans_faux_negatif():
    envois = envois_exemple()
    filtres = []
    for numeros in envois.values():
        filtre = FiltreBloom(*dimensionner(12_000, 0.01)[:2])
        for numero in numeros:
            filtre.ajouter(numero)
        filtres.append(filtre)
    tous = filtres[0].union(filtres[1]).union(filtres[2])
    assert all(numero in tous for numeros in envois.values() for numero in numeros)   # jamais de doublon
    deja = set().union(*envois.values())
    faux_positifs = sum(numero in tous for numero in campagne_exemple() if numero not in deja)
    assert faux_positifs < 3000 * 0.02


def test_frequents_garantit_les_gros_appelants():
    appels = appels_exemple()
    resultat = frequents(iter(appels), 100)                 # tout ce qui dépasse 60 000 / 100 = 600 appels
    assert len(resultat) <= 99
    assert {"0700000000", "0611111111", "0622222222", "0633333333"} <= set(resultat)
    assert frequents("aabacad", 2) == {"a": 1}             # « a » est majoritaire : il reste seul


def test_compteur_min_ne_sous_estime_jamais():
    cm = CompteurMin.pour_erreur(0.001, 0.01)
    assert (cm.largeur, cm.profondeur) == (2719, 5)
    from collections import Counter
    appels = appels_exemple()
    for numero in appels:
        cm.ajouter(numero)
    for numero, vrai in Counter(appels).items():
        assert vrai <= cm.estimer(numero) <= vrai + 0.001 * cm.total * 3


def test_hyperloglog_fusion_par_maximum():
    visites = visites_exemple()
    jours = []
    for visiteurs in visites.values():
        h = HyperLogLog(14)
        for v in visiteurs:
            h.ajouter(v)
        jours.append(h)
    semaine = jours[0]
    for h in jours[1:]:
        semaine = semaine.fusionner(h)
    vrai = len(set().union(*visites.values()))
    assert abs(semaine.estimation() - vrai) / vrai < 0.03


def test_anneau_ne_deplace_que_la_part_du_nouveau():
    anneau = Anneau(SERVEURS)
    clients = clients_exemple()
    avant = {c: anneau.serveur_pour(c) for c in clients}
    anneau.ajouter("score-5")
    deplaces = [c for c in clients if anneau.serveur_pour(c) != avant[c]]
    assert all(anneau.serveur_pour(c) == "score-5" for c in deplaces)
    assert 0.15 < len(deplaces) / len(clients) < 0.25
    assert anneau.serveur_pour("score-2#7") == "score-2"     # une clé posée pile sur un point lui appartient

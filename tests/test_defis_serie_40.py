"""Série 40 (compter sans tout retenir) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 242`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (une classe à `__slots__`, des registres en
  liste, le rang lu dans une chaîne de bits complétée par des zéros, Misra-Gries écrit avec `Counter`, un anneau
  rangé avec `bisect.insort`…) ;
- les références du correcteur tiennent face à des vérifications indépendantes : aucun faux négatif et un taux de
  faux positifs proche de la théorie, la garantie n / k de Misra-Gries contre un comptage exact, une erreur de
  HyperLogLog de l'ordre de 1,04 / √m, un anneau qui ne déplace que la part du nouveau serveur ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (1 000 numéros bloqués à raison et 35 à tort, les quatre gros appelants trouvés, la semaine comptée à 0,4 % près
  alors que l'addition des jours se trompe de 40 %, 18,6 % des clients déplacés au lieu de 80 %).
"""

from __future__ import annotations

import bisect
import doctest
import itertools
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import defis.checks_serie_40 as checks  # noqa: E402
import defis.corrige_serie_40 as corrige  # noqa: E402
import defis.esquisses as esquisses  # noqa: E402
from defis.checks_serie_40 import (  # noqa: E402
    SERIE,
    _bits_possibles,
    _ref_dimension,
    _ref_estimation,
    _ref_frequents,
    _ref_positions,
    _ref_registres,
    _RefAnneau,
)
from defis.esquisses import (  # noqa: E402
    SERVEURS,
    Dimension,
    appels_exemple,
    campagne_exemple,
    clients_exemple,
    empreinte,
    envois_exemple,
    generer_appels,
    generer_visiteurs,
    visites_exemple,
)
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


# --------------------------------------------------------------------------- Défi 242 : dimensionner


def _dimensionner_bogue(bug):
    def dimensionner(n, taux):
        corrige.dimensionner(n, taux)                 # les mêmes validations que le corrigé
        return Dimension(*_ref_dimension(n, taux, bug))
    return dimensionner


@pytest.mark.parametrize("bug,morceau", [
    ("log10", "NATUREL"),
    ("log10_partout", "2,3 fois trop GRAND"),
    ("ln2", "(ln 2)²"),
    ("int", "VERS LE BAS"),
    ("round_bits", "au plus proche"),
    ("k_zero", "max(1, round"),
    ("k_int", "tronquez"),
    ("k_ceil", "vers le haut"),
    ("k_inverse", "n / bits"),
    ("taux_demande", "DEMANDÉ"),
    ("taux_sans_k", "puissance k"),
    ("taux_exposant", "exponentielle"),
    ("taux_exact", "approximation"),
])
def test_dimensionner_nomme_l_erreur(bug, morceau):
    _echec("dimensionner", _dimensionner_bogue(bug), morceau)


def test_dimensionner_ecrit_par_un_eleve():
    def dimensionner(n, taux):
        if type(n) is not int:
            raise TypeError("n")
        if type(taux) not in (int, float):
            raise TypeError("taux")
        if n <= 0 or not (taux > 0 and taux < 1):
            raise ValueError("bornes")
        bits = math.ceil(n * math.log(1 / taux) / (math.log(2) * math.log(2)))
        k = max(1, round(math.log(2) * bits / n))
        return (bits, k, (1 - math.exp(-k * n / bits)) ** k)          # un simple tuple suffit

    _reussi("dimensionner", dimensionner)


def test_dimensionner_sans_return_ou_qui_accepte_un_bool():
    _echec("dimensionner", lambda n, taux: None, "return")

    def accepte_bool(n, taux):
        return corrige.dimensionner(int(n), taux) if n is True else corrige.dimensionner(n, taux)

    _echec("dimensionner", accepte_bool, "bool")

    def nan_accepte(n, taux):
        if taux != taux:
            return Dimension(1, 1, 1.0)
        return corrige.dimensionner(n, taux)

    _echec("dimensionner", nan_accepte, "NaN")


def test_les_formules_acceptees_contiennent_la_reference():
    for n, taux in checks._batterie_242():
        bits, k, reel = _ref_dimension(n, taux)
        assert bits in _bits_possibles(n, taux)
        assert math.isclose(reel, esquisses.taux_theorique(bits, k, n))
        assert reel <= taux * 1.6 or bits < 20          # k arrondi : on reste près du taux promis


# --------------------------------------------------------------------------- Défi 243 : FiltreBloom


def _bloom_positions(bug):
    class Faux(corrige.FiltreBloom):
        def positions(self, cle):
            corrige._texte(cle)
            return _ref_positions(cle, self.bits, self.k, bug)
    return Faux


@pytest.mark.parametrize("bug,morceau", [
    ("i_un", "i = 1"),
    ("meme_graine", "même empreinte"),
    ("inverse", "inversés"),
    ("graines_1_2", "pas 1 et 2"),
    ("k_graines", "DOUBLE hachage"),
    ("set", "doublons compris"),
    ("trie", "Vous triez"),
])
def test_filtre_bloom_nomme_l_erreur_de_positions(bug, morceau):
    _echec("FiltreBloom", _bloom_positions(bug), morceau)


def test_filtre_bloom_positions_en_set_ou_avec_hash():
    class EnSet(corrige.FiltreBloom):
        def positions(self, cle):
            return set(super().positions(cle))

    _echec("FiltreBloom", EnSet, "LISTE", "set")

    class AvecHash(corrige.FiltreBloom):
        def positions(self, cle):
            corrige._texte(cle)
            h1, h2 = hash(cle) % 2 ** 64, hash(cle + "!") % 2 ** 64
            return [(h1 + i * h2) % self.bits for i in range(self.k)]

    _echec("FiltreBloom", AvecHash, "hash()")


def test_filtre_bloom_modulo_par_les_octets():
    class Faux(corrige.FiltreBloom):
        def positions(self, cle):
            corrige._texte(cle)
            return _ref_positions(cle, self.bits, self.k, "octets")

        def ajouter(self, cle):
            for p in self.positions(cle):
                if p // 8 < len(self.octets):
                    self.octets[p // 8] |= 1 << (p % 8)

    _echec("FiltreBloom", Faux, "bourrage")


def test_filtre_bloom_bit_de_poids_fort():
    class Faux(corrige.FiltreBloom):
        def ajouter(self, cle):
            for p in self.positions(cle):
                self.octets[p // 8] |= 0x80 >> (p % 8)

        def __contains__(self, cle):
            return all(self.octets[p // 8] & (0x80 >> (p % 8)) for p in self.positions(cle))

    _echec("FiltreBloom", Faux, "POIDS FORT")


def test_filtre_bloom_any_au_lieu_de_all():
    class Faux(corrige.FiltreBloom):
        def __contains__(self, cle):
            return any(self.octets[p // 8] >> (p % 8) & 1 for p in self.positions(cle))

    _echec("FiltreBloom", Faux, "`any`")


def test_filtre_bloom_un_octet_par_bit():
    class Faux(corrige.FiltreBloom):
        def __init__(self, bits, k):
            super().__init__(bits, k)
            self.octets = bytearray(bits)

    _echec("FiltreBloom", Faux, "8 fois trop")


def test_filtre_bloom_remplissage_par_les_octets():
    class Faux(corrige.FiltreBloom):
        def remplissage(self):
            return sum(bin(o).count("1") for o in self.octets) / (len(self.octets) * 8)

    _echec("FiltreBloom", Faux, "bourrage")

    class Compte(corrige.FiltreBloom):
        def remplissage(self):
            return sum(bin(o).count("1") for o in self.octets)

    _echec("FiltreBloom", Compte, "NOMBRE")


def test_filtre_bloom_union_fausse():
    class Et(corrige.FiltreBloom):
        def union(self, autre):
            r = corrige.FiltreBloom.union(self, autre)
            r.__class__ = Et
            r.octets = bytearray(a & b for a, b in zip(self.octets, autre.octets))
            return r

    _echec("FiltreBloom", Et, "(`&`)")

    class SurPlace(corrige.FiltreBloom):
        def union(self, autre):
            corrige.FiltreBloom.union(self, autre)          # les mêmes vérifications
            for i, o in enumerate(autre.octets):
                self.octets[i] |= o
            return self

    _echec("FiltreBloom", SurPlace, "NOUVEAU filtre")

    class SansControle(corrige.FiltreBloom):
        def union(self, autre):
            r = type(self)(self.bits, self.k)
            r.octets = bytearray(a | b for a, b in zip(self.octets, autre.octets))
            return r

    _echec("FiltreBloom", SansControle, "ValueError")


def test_filtre_bloom_qui_garde_les_cles_ou_partage_ses_bits():
    class Garde(corrige.FiltreBloom):
        def __init__(self, bits, k):
            super().__init__(bits, k)
            self.vus = set()

        def ajouter(self, cle):
            super().ajouter(cle)
            self.vus.add(cle)

    _echec("FiltreBloom", Garde, "self.vus", "12 novembre")

    class Partage(corrige.FiltreBloom):
        _commun: dict = {}

        def __init__(self, bits, k):
            super().__init__(bits, k)
            self.octets = Partage._commun.setdefault((bits, k), bytearray((bits + 7) // 8))

    _echec("FiltreBloom", Partage, "partagent")


def test_filtre_bloom_qui_convertit_un_numero_entier():
    class Faux(corrige.FiltreBloom):
        def positions(self, cle):
            return super().positions(str(cle))

    _echec("FiltreBloom", Faux, "612345678", "TypeError")


def test_filtre_bloom_ecrit_par_un_eleve():
    class FiltreBloom:
        __slots__ = ("bits", "k", "octets")

        def __init__(self, bits, k):
            for valeur in (bits, k):
                if type(valeur) is not int:
                    raise TypeError(valeur)
                if valeur < 1:
                    raise ValueError(valeur)
            self.bits, self.k = bits, k
            self.octets = bytearray(-(-bits // 8))

        def positions(self, cle):
            if not isinstance(cle, str):
                raise TypeError(cle)
            h1 = int.from_bytes(__import__("hashlib").sha256(f"0:{cle}".encode()).digest()[:8], "big")
            h2 = int.from_bytes(__import__("hashlib").sha256(f"1:{cle}".encode()).digest()[:8], "big")
            resultat, p = [], h1 % self.bits
            for _ in range(self.k):                     # (h1 + i·h2) % bits, mis à jour pas à pas
                resultat.append(p)
                p = (p + h2) % self.bits
            return resultat

        def ajouter(self, cle):
            for p in self.positions(cle):
                self.octets[p >> 3] |= 1 << (p & 7)

        def __contains__(self, cle):
            return all(self.octets[p >> 3] & (1 << (p & 7)) for p in self.positions(cle))

        def remplissage(self):
            return sum(bin(o).count("1") for o in self.octets) / self.bits

        def union(self, autre):
            if not isinstance(autre, FiltreBloom):
                raise TypeError(autre)
            if autre.bits != self.bits or autre.k != self.k:
                raise ValueError("incompatibles")
            r = FiltreBloom(self.bits, self.k)
            r.octets[:] = self.octets
            for i, o in enumerate(autre.octets):
                r.octets[i] |= o
            return r

    _reussi("FiltreBloom", FiltreBloom)


def test_filtre_bloom_sans_faux_negatif_et_au_bon_taux():
    for n, taux in ((2000, 0.01), (500, 0.1), (3000, 0.001)):
        d = corrige.dimensionner(n, taux)
        f = corrige.FiltreBloom(d.bits, d.k)
        cles = [f"in-{i}" for i in range(n)]
        for c in cles:
            f.ajouter(c)
        assert all(c in f for c in cles)
        essais = 40_000
        faux = sum(f"out-{i}" in f for i in range(essais)) / essais
        ecart = 4 * math.sqrt(d.taux_reel / essais) + 0.0005
        assert abs(faux - d.taux_reel) < ecart, (n, taux, faux, d.taux_reel)
        assert abs(f.remplissage() - (1 - math.exp(-d.k * n / d.bits))) < 0.02


# --------------------------------------------------------------------------- Défi 244 : frequents


def _frequents_bogue(bug):
    def frequents(flux, k):
        corrige._entier(k, "k", 2)
        resultat = _ref_frequents(list(flux) if bug != "tout" else flux, k, bug)
        return dict(sorted(resultat.items(), key=lambda p: p[1], reverse=True))
    return frequents


@pytest.mark.parametrize("bug,morceau", [
    ("tout", "Counter"),
    ("k", "AU PLUS k - 1"),
    ("ajoute", "n'entre PAS"),
    ("zeros", "tombent à 0"),
    ("remplace", "Space-Saving"),
])
def test_frequents_nomme_l_erreur(bug, morceau):
    _echec("frequents", _frequents_bogue(bug), morceau)


def test_frequents_qui_lit_deux_fois_ou_modifie_en_parcourant():
    def deux_fois(flux, k):
        _ = sum(1 for _ in flux)
        return corrige.frequents(flux, k)

    _echec("frequents", deux_fois, "DEUX fois")

    def en_parcourant(flux, k):
        compteurs = {}
        for x in flux:
            if x in compteurs:
                compteurs[x] += 1
            elif len(compteurs) < k - 1:
                compteurs[x] = 1
            else:
                for cle in compteurs:
                    compteurs[cle] -= 1
                    if compteurs[cle] == 0:
                        del compteurs[cle]
        return compteurs

    _echec("frequents", en_parcourant, "list(compteurs)")


def test_frequents_non_trie_ou_trop_gourmand_ou_qui_valide_trop_tard():
    def non_trie(flux, k):
        return dict(sorted(corrige.frequents(flux, k).items()))

    _echec("frequents", non_trie, "plus grand au plus petit")

    def gourmand(flux, k):
        return corrige.frequents(list(flux), k)               # une copie du flux : la mémoire grandit

    _echec("frequents", gourmand, "Mo")

    def trop_tard(flux, k):
        valeurs = iter(flux)
        premier = next(valeurs, None)                       # une valeur lue AVANT de valider k
        corrige._entier(k, "k", 2)
        return corrige.frequents(itertools.chain([] if premier is None else [premier], valeurs), k)

    _echec("frequents", trop_tard, "APRÈS avoir lu")


def test_frequents_ecrit_par_un_eleve_avec_counter():
    def frequents(flux, k):
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(k)
        if k < 2:
            raise ValueError(k)
        compteurs = Counter()
        for x in flux:
            if x in compteurs or len(compteurs) < k - 1:
                compteurs[x] += 1
            else:
                compteurs -= Counter(dict.fromkeys(compteurs, 1))   # la soustraction de Counter retire les zéros
        return dict(compteurs.most_common())

    _reussi("frequents", frequents)


def test_reference_frequents_tient_sa_garantie():
    alea = random.Random(7)
    for _ in range(200):
        flux = generer_appels(alea.randrange(1, 2000), alea.choice((2, 10, 100)), graine=alea.randrange(10 ** 6))
        k = alea.choice((2, 3, 5, 10, 30))
        resultat, vrais = _ref_frequents(flux, k), Counter(flux)
        assert len(resultat) <= k - 1
        for x, vrai in vrais.items():
            assert vrai - len(flux) / k <= resultat.get(x, 0) <= vrai
            if vrai > len(flux) / k:
                assert x in resultat


# --------------------------------------------------------------------------- Défi 245 : CompteurMin


def test_compteur_min_lignes_partagees():
    class Faux(corrige.CompteurMin):
        def __init__(self, largeur, profondeur):
            super().__init__(largeur, profondeur)
            self.tableau = [[0] * largeur] * profondeur

    _echec("CompteurMin", Faux, "RÉFÉRENCE")


@pytest.mark.parametrize("bug,morceau", [
    ("meme_graine", "même graine"),
    ("graine_plus_un", "pas i + 1"),
    ("graine_ligne_colonne", "la graine de `empreinte`"),
])
def test_compteur_min_nomme_l_erreur_de_colonnes(bug, morceau):
    class Faux(corrige.CompteurMin):
        def colonnes(self, cle):
            corrige._texte(cle)
            return checks._ref_colonnes(cle, self.largeur, self.profondeur, bug)

    _echec("CompteurMin", Faux, morceau)


@pytest.mark.parametrize("bug,morceau", [
    ("max", "MAXIMUM"),
    ("moyenne", "MOYENNE"),
    ("somme", "additionnez"),
    ("premiere", "première ligne"),
])
def test_compteur_min_nomme_l_erreur_d_estimation(bug, morceau):
    class Faux(corrige.CompteurMin):
        def estimer(self, cle):
            corrige._texte(cle)
            return checks._ref_estimer(self.tableau, cle, bug)

    _echec("CompteurMin", Faux, morceau)


@pytest.mark.parametrize("bug,morceau", [
    ("log10", "NATUREL"),
    ("log2", "log2"),
    ("sans_e", "math.e"),
    ("inverse", "inversées"),
])
def test_compteur_min_nomme_l_erreur_de_dimension(bug, morceau):
    class Faux(corrige.CompteurMin):
        @classmethod
        def pour_erreur(cls, epsilon, delta):
            corrige.CompteurMin.pour_erreur(epsilon, delta)
            return cls(*checks._ref_pour_erreur(epsilon, delta, bug))

    _echec("CompteurMin", Faux, morceau)


def test_compteur_min_qui_accepte_un_negatif_ou_modifie_avant_de_valider():
    class Negatif(corrige.CompteurMin):
        def ajouter(self, cle, n=1):
            for ligne, colonne in enumerate(self.colonnes(cle)):
                self.tableau[ligne][colonne] += n
            self.total += n

    _echec("CompteurMin", Negatif, "SOUS la vérité")

    class TropTot(corrige.CompteurMin):
        def ajouter(self, cle, n=1):
            for ligne, colonne in enumerate(self.colonnes(cle)):
                self.tableau[ligne][colonne] += n if isinstance(n, int) else 0
            corrige._entier(n, "n", 0)
            self.total += n

    _echec("CompteurMin", TropTot, "validez la clé ET n")

    class SansTotal(corrige.CompteurMin):
        def ajouter(self, cle, n=1):
            total = self.total
            super().ajouter(cle, n)
            self.total = total + 1

    _echec("CompteurMin", SansTotal, "total")

    class PasUneClassmethod(corrige.CompteurMin):
        @classmethod
        def pour_erreur(cls, epsilon, delta):
            return corrige.CompteurMin.pour_erreur(epsilon, delta).__dict__

    _echec("CompteurMin", PasUneClassmethod, "cls(largeur, profondeur)")


def test_compteur_min_ecrit_par_un_eleve():
    class CompteurMin:
        def __init__(self, largeur, profondeur):
            for v in (largeur, profondeur):
                if type(v) is not int:
                    raise TypeError(v)
                if v < 1:
                    raise ValueError(v)
            self.largeur, self.profondeur, self.total = largeur, profondeur, 0
            self.tableau = []
            for _ in range(profondeur):
                self.tableau.append([0 for _ in range(largeur)])

        @classmethod
        def pour_erreur(cls, epsilon, delta):
            for v in (epsilon, delta):
                if type(v) not in (int, float):
                    raise TypeError(v)
                if not 0 < v < 1:
                    raise ValueError(v)
            return cls(math.ceil(math.exp(1) / epsilon), math.ceil(-math.log(delta)))

        def colonnes(self, cle):
            if type(cle) is not str:
                raise TypeError(cle)
            return [empreinte(cle, graine=i) % self.largeur for i in range(self.profondeur)]

        def ajouter(self, cle, n=1):
            colonnes = self.colonnes(cle)
            if type(n) is not int:
                raise TypeError(n)
            if n < 0:
                raise ValueError(n)
            for i in range(self.profondeur):
                self.tableau[i][colonnes[i]] += n
            self.total += n

        def estimer(self, cle):
            colonnes = self.colonnes(cle)
            return min(ligne[c] for ligne, c in zip(self.tableau, colonnes))

    _reussi("CompteurMin", CompteurMin)


def test_compteur_min_ne_sous_estime_jamais_et_tient_sa_garantie():
    for graine in range(5):
        flux = generer_appels(5000, 800, graine=graine)
        cm = corrige.CompteurMin.pour_erreur(0.01, 0.01)
        for x in flux:
            cm.ajouter(x)
        vrais = Counter(flux)
        depassements = sum(cm.estimer(x) > vrai + 0.01 * len(flux) for x, vrai in vrais.items())
        assert all(cm.estimer(x) >= vrai for x, vrai in vrais.items())
        assert depassements <= 0.01 * len(vrais) + 1


# --------------------------------------------------------------------------- Défi 246 : HyperLogLog


def _hll_registres(bug):
    class Faux(corrige.HyperLogLog):
        def ajouter(self, cle):
            j, rang = checks._registre_et_rang(corrige._texte(cle), self.precision, bug)
            self.registres[j] = rang if bug == "sans_max" else max(self.registres[j], rang)
    return Faux


@pytest.mark.parametrize("bug,morceau", [
    ("zeros", "ZÉROS de tête"),
    ("bin", "`bin(w)`"),
    ("bas", "bits de DROITE"),
    ("droite", "zéros à DROITE"),
    ("w_64", "les 64 bits"),
    ("sans_max", "MAXIMUM des rangs"),
])
def test_hyperloglog_nomme_l_erreur_de_registre(bug, morceau):
    _echec("HyperLogLog", _hll_registres(bug), morceau)


@pytest.mark.parametrize("bug,morceau", [
    ("sans_lineaire", "linear counting"),
    ("toujours_lineaire", "ne s'applique que si"),
    ("alpha_fixe", "0.673"),
    ("sans_alpha", "oubliez α"),
    ("geometrique", "GÉOMÉTRIQUE"),
    ("log10", "logarithme NATUREL"),
])
def test_hyperloglog_nomme_l_erreur_d_estimation(bug, morceau):
    class Faux(corrige.HyperLogLog):
        def estimation(self):
            return _ref_estimation(list(self.registres), bug)

    _echec("HyperLogLog", Faux, morceau)


def test_hyperloglog_fusion_fausse():
    def avec(operation):
        class Faux(corrige.HyperLogLog):
            def fusionner(self, autre):
                corrige.HyperLogLog.fusionner(self, autre)
                r = Faux(self.precision)
                r.registres = bytearray(operation(a, b) for a, b in zip(self.registres, autre.registres))
                return r
        return Faux

    _echec("HyperLogLog", avec(lambda a, b: a + b), "ADDITIONNEZ", "deux fois")
    _echec("HyperLogLog", avec(min), "MINIMUM")

    class SurPlace(corrige.HyperLogLog):
        def fusionner(self, autre):
            corrige.HyperLogLog.fusionner(self, autre)
            self.registres = bytearray(max(a, b) for a, b in zip(self.registres, autre.registres))
            return self

    _echec("HyperLogLog", SurPlace, "NOUVEL objet")


def test_hyperloglog_qui_garde_les_cles_ou_partage_ses_registres():
    class Exact(corrige.HyperLogLog):
        def __init__(self, precision=14):
            super().__init__(precision)
            self.vus = set()

        def ajouter(self, cle):
            super().ajouter(cle)
            self.vus.add(cle)

    _echec("HyperLogLog", Exact, "self.vus")

    class Partage(corrige.HyperLogLog):
        _communs: dict = {}

        def __init__(self, precision=14):
            super().__init__(precision)
            self.registres = Partage._communs.setdefault(precision, bytearray(1 << precision))

    _echec("HyperLogLog", Partage, "partagent")

    class SansBorne(corrige.HyperLogLog):
        def __init__(self, precision=14):
            if precision == 17:
                precision = 16
            super().__init__(precision)

    _echec("HyperLogLog", SansBorne, "4 à 16")


def test_hyperloglog_ecrit_par_un_eleve():
    class HyperLogLog:
        def __init__(self, precision=14):
            if type(precision) is not int:
                raise TypeError(precision)
            if not 4 <= precision <= 16:
                raise ValueError(precision)
            self.precision = precision
            self.registres = [0] * 2 ** precision                    # une liste : acceptée

        def ajouter(self, cle):
            if type(cle) is not str:
                raise TypeError(cle)
            p = self.precision
            bits = format(empreinte(cle), "064b")                   # 64 caractères, zéros de tête compris
            j, w = int(bits[:p], 2), bits[p:]
            rang = w.find("1") + 1 if "1" in w else len(w) + 1
            self.registres[j] = max(self.registres[j], rang)

        def estimation(self):
            m = len(self.registres)
            alpha = 0.673 if m == 16 else 0.697 if m == 32 else 0.709 if m == 64 else 0.7213 / (1 + 1.079 / m)
            e = alpha * m ** 2 / sum(1 / 2 ** r for r in self.registres)
            v = self.registres.count(0)
            return m * math.log(m / v) if v > 0 and e <= 5 * m / 2 else e

        def fusionner(self, autre):
            if not isinstance(autre, HyperLogLog):
                raise TypeError(autre)
            if autre.precision != self.precision:
                raise ValueError("precision")
            r = HyperLogLog(self.precision)
            r.registres = [max(a, b) for a, b in zip(self.registres, autre.registres)]
            return r

    _reussi("HyperLogLog", HyperLogLog)


def test_hyperloglog_se_trompe_de_l_ordre_de_1_04_sur_racine_de_m():
    for p in (10, 12):
        erreur_type = 1.04 / math.sqrt(2 ** p)
        for n in (5_000, 40_000):
            h = corrige.HyperLogLog(p)
            for i in range(n):
                h.ajouter(f"essai-{p}-{n}-{i}")
            assert abs(h.estimation() - n) / n < 4 * erreur_type, (p, n, h.estimation())
    petit = corrige.HyperLogLog(14)
    for i in range(10):
        petit.ajouter(f"x{i}")
    assert abs(petit.estimation() - 10) < 0.1                       # le linear counting, quasi exact
    assert _ref_estimation([0] * 16) == 0.0


# --------------------------------------------------------------------------- Défi 247 : Anneau


def _anneau_bogue(bug):
    class Faux(corrige.Anneau):
        def __init__(self, serveurs=(), repliques=100):
            self._ref = _RefAnneau((), repliques, bug)
            super().__init__(serveurs, repliques)

        def ajouter(self, serveur):
            super().ajouter(serveur)
            self._ref.ajouter(serveur)

        def retirer(self, serveur):
            super().retirer(serveur)
            self._ref.retirer(serveur)

        def serveur_pour(self, cle):
            super().serveur_pour(cle)                           # les mêmes validations
            return self._ref.serveur_pour(cle)
    return Faux


@pytest.mark.parametrize("bug,morceau", [
    ("modulo", "80 %"),
    ("droite", "`bisect_left`"),
    ("precedent", "PRÉCÉDENT"),
    ("sans_repliques", "UN seul point"),
    ("sans_diese", "avec le « # »"),
])
def test_anneau_nomme_l_erreur(bug, morceau):
    _echec("Anneau", _anneau_bogue(bug), morceau)


def test_anneau_sans_tour_ou_trop_lent():
    class SansTour(corrige.Anneau):
        def serveur_pour(self, cle):
            position = empreinte(corrige._texte(cle))
            if not self._points:
                raise ValueError("vide")
            return self._points[bisect.bisect_left(self._positions, position)][1]

    _echec("Anneau", SansTour, "tour du cercle")

    class Lent(corrige.Anneau):
        def serveur_pour(self, cle):
            position = empreinte(corrige._texte(cle))
            if not self._points:
                raise ValueError("vide")
            for p, serveur in self._points:
                if p >= position:
                    return serveur
            return self._points[0][1]

    _echec("Anneau", Lent, "beaucoup trop lent", "bisect")


def test_anneau_qui_retire_trop_ou_accepte_un_doublon():
    class RetireTrop(corrige.Anneau):
        def retirer(self, serveur):
            super().retirer(serveur)
            if self._points:
                self._ranger(self._points[1:])               # un point d'un autre serveur disparaît aussi

    _echec("Anneau", RetireTrop)

    class Doublon(corrige.Anneau):
        def ajouter(self, serveur):
            if serveur in self._serveurs:
                return
            super().ajouter(serveur)

    _echec("Anneau", Doublon, "ValueError")

    class Partage(corrige.Anneau):
        _points_communs: list = []

        def _ranger(self, points):
            Partage._points_communs[:] = sorted(points)
            self._points = Partage._points_communs
            self._positions = [p for p, _ in self._points]

    _echec("Anneau", Partage)


def test_anneau_ecrit_par_un_eleve():
    class Anneau:
        def __init__(self, serveurs=(), repliques=100):
            if type(repliques) is not int:
                raise TypeError(repliques)
            if repliques < 1:
                raise ValueError(repliques)
            self.repliques = repliques
            self.positions = []                              # triées, tenues à jour par insort
            self.proprietaire = {}                           # position → serveur
            self.serveurs = set()
            for s in serveurs:
                self.ajouter(s)

        def ajouter(self, serveur):
            if not isinstance(serveur, str):
                raise TypeError(serveur)
            if serveur == "" or serveur in self.serveurs:
                raise ValueError(serveur)
            for i in range(self.repliques):
                position = empreinte(f"{serveur}#{i}")
                bisect.insort(self.positions, position)
                self.proprietaire[position] = serveur
            self.serveurs.add(serveur)

        def retirer(self, serveur):
            if not isinstance(serveur, str):
                raise TypeError(serveur)
            if serveur not in self.serveurs:
                raise ValueError(serveur)
            for i in range(self.repliques):
                position = empreinte(f"{serveur}#{i}")
                del self.positions[bisect.bisect_left(self.positions, position)]
                del self.proprietaire[position]
            self.serveurs.discard(serveur)

        def serveur_pour(self, cle):
            if not isinstance(cle, str):
                raise TypeError(cle)
            if not self.positions:
                raise ValueError("aucun serveur")
            i = bisect.bisect_left(self.positions, empreinte(cle))
            if i == len(self.positions):
                i = 0
            return self.proprietaire[self.positions[i]]

        def __len__(self):
            return len(self.serveurs)

    _reussi("Anneau", Anneau)


def test_l_anneau_repartit_et_ne_deplace_que_la_part_du_nouveau():
    clients = clients_exemple()
    for n in (3, 4, 8):
        serveurs = [f"s{i}" for i in range(n)]
        anneau = _RefAnneau(serveurs)
        avant = {c: anneau.serveur_pour(c) for c in clients}
        charges = Counter(avant.values())
        assert max(charges.values()) < 1.5 * len(clients) / n                       # 100 points : à peu près équilibré
        anneau.ajouter("nouveau")
        deplaces = [c for c in clients if anneau.serveur_pour(c) != avant[c]]
        assert {anneau.serveur_pour(c) for c in deplaces} == {"nouveau"}
        assert 0.5 / (n + 1) < len(deplaces) / len(clients) < 1.5 / (n + 1)
        modulo_avant = {c: serveurs[empreinte(c) % n] for c in clients}
        modulo_apres = {c: (serveurs + ["nouveau"])[empreinte(c) % (n + 1)] for c in clients}
        assert sum(modulo_avant[c] != modulo_apres[c] for c in clients) > len(clients) * n / (n + 1) * 0.9


def test_les_cles_du_bord_font_bien_le_tour():
    ref = _RefAnneau(SERVEURS)
    for cle in checks._cles_sur_le_bord(ref.points, 5):
        assert ref.serveur_pour(cle) == ref.points[0][1]


# --------------------------------------------------------------------------- les outils communs et l'histoire


def test_l_empreinte_est_la_meme_d_un_lancement_a_l_autre():
    code = "from defis.esquisses import empreinte; print(empreinte('C000001'), empreinte('C000001', 3))"
    sorties = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                              env={"PYTHONHASHSEED": graine, "PATH": ""}).stdout for graine in ("1", "2")}
    assert sorties == {f"{empreinte('C000001')} {empreinte('C000001', 3)}\n"}
    with pytest.raises(TypeError):
        empreinte(612345678)
    with pytest.raises(TypeError):
        empreinte("x", graine=True)


def test_les_outils_communs():
    envois = envois_exemple()
    tous = [n for numeros in envois.values() for n in numeros]
    assert len(tous) == len(set(tous)) == 12_000 and envois is not envois_exemple()
    campagne = campagne_exemple()
    assert len(set(campagne)) == 4000 and len(set(campagne) & set(tous)) == 1000
    assert all(len(n) == 10 and n[:2] in ("06", "07") and n.isdigit() for n in tous + campagne)
    appels = appels_exemple()
    assert len(appels) == 60_000 and appels == appels_exemple()
    assert [n for n, _ in Counter(appels).most_common(4)] == ["0700000000", "0611111111", "0622222222", "0633333333"]
    assert Counter(appels).most_common(5)[4][1] < 600                    # aucun autre ne dépasse n / 100
    visites = visites_exemple()
    assert list(visites) == list(esquisses.JOURS)
    assert generer_visiteurs(50, graine=3) == generer_visiteurs(50, graine=3)
    assert clients_exemple(3) == ["C000001", "C000002", "C000003"]
    assert Dimension(9, 1, 0.5).octets == 2


def test_l_histoire_tient():
    envois, campagne = envois_exemple(), campagne_exemple()
    deja = set().union(*envois.values())
    d = corrige.dimensionner(12_000, 0.01)
    filtres = []
    for numeros in envois.values():
        f = corrige.FiltreBloom(d.bits, d.k)
        for numero in numeros:
            f.ajouter(numero)
        filtres.append(f)
    union = filtres[0].union(filtres[1]).union(filtres[2])
    bloques = [n for n in campagne if n in union]
    assert sum(n in deja for n in bloques) == 1000                        # aucun doublon possible
    assert len(bloques) - 1000 == 35                                      # et 35 clients privés de SMS, à tort
    assert sum(n in filtres[0] for n in campagne) < 1000                  # un seul serveur ne suffit pas
    assert corrige.dimensionner(40_000_000, 0.001).octets < 72 * 10 ** 6

    appels = appels_exemple()
    assert list(corrige.frequents(appels, 100))[:4] == ["0700000000", "0611111111", "0622222222", "0633333333"]
    assert "0700000000" not in corrige.frequents(appels, 10)            # k = 10 ne promet rien sous 6 000 appels

    visites = visites_exemple()
    jours = []
    for visiteurs in visites.values():
        h = corrige.HyperLogLog(14)
        for v in visiteurs:
            h.ajouter(v)
        jours.append(h)
    semaine = jours[0]
    for h in jours[1:]:
        semaine = semaine.fusionner(h)
    vrai = len(set().union(*visites.values()))
    addition = sum(len(set(v)) for v in visites.values())
    assert (vrai, addition) == (7981, 11242)
    assert abs(semaine.estimation() - vrai) / vrai < 0.005
    assert addition / vrai > 1.4                                          # l'erreur du tableau de bord

    anneau = corrige.Anneau(SERVEURS)
    clients = clients_exemple()
    avant = {c: anneau.serveur_pour(c) for c in clients}
    anneau.ajouter("score-5")
    assert sum(anneau.serveur_pour(c) != avant[c] for c in clients) == 3714
    cinq = SERVEURS + ("score-5",)
    assert sum(SERVEURS[empreinte(c) % 4] != cinq[empreinte(c) % 5] for c in clients) == 16028


# --------------------------------------------------------------------------- le correcteur lui-même


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(242, 248))
    assert any(s is SERIE for s in series())            # enregistrée dans defis/verif.py


def test_une_methode_oubliee_est_nommee():
    class Commence(corrige.FiltreBloom):
        def remplissage(self):
            raise NotImplementedError

    message = _echec("FiltreBloom", Commence, "remplissage", "terminez-la")
    assert "NotImplementedError" in message


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_40 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(esquisses)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_40 import *` (dans les exemples) doit donner les objets du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 40 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    appels = appels_exemple()
    copie = list(appels)
    corrige.frequents(appels, 50)
    assert appels == copie
    a, b = corrige.FiltreBloom(100, 3), corrige.FiltreBloom(100, 3)
    a.ajouter("x")
    b.ajouter("y")
    octets_a, octets_b = bytes(a.octets), bytes(b.octets)
    a.union(b)
    assert (bytes(a.octets), bytes(b.octets)) == (octets_a, octets_b)
    h, g = corrige.HyperLogLog(4), corrige.HyperLogLog(4)
    h.ajouter("x")
    g.ajouter("y")
    registres = (bytes(h.registres), bytes(g.registres))
    h.fusionner(g)
    assert (bytes(h.registres), bytes(g.registres)) == registres
    serveurs = ["a", "b"]
    corrige.Anneau(serveurs).ajouter("c")
    assert serveurs == ["a", "b"]


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(ValueError):
        corrige.dimensionner(1000, math.nan)
    with pytest.raises(TypeError):
        corrige.dimensionner(True, 0.01)
    with pytest.raises(TypeError):
        corrige.FiltreBloom(100, 3).ajouter(612345678)
    with pytest.raises(ValueError):
        corrige.FiltreBloom(100, 3).union(corrige.FiltreBloom(100, 4))
    with pytest.raises(ValueError):
        corrige.frequents([], 1)
    with pytest.raises(ValueError):
        corrige.CompteurMin(5, 3).ajouter("x", -1)
    with pytest.raises(ValueError):
        corrige.HyperLogLog(17)
    with pytest.raises(ValueError):
        corrige.HyperLogLog(4).fusionner(corrige.HyperLogLog(5))
    with pytest.raises(ValueError):
        corrige.Anneau(["a", "a"])
    with pytest.raises(ValueError):
        corrige.Anneau().serveur_pour("C000001")

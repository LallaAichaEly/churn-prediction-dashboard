"""Série 30 (chercher dans les notes) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 182`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (fusion de listes équilibrées, dernière
  position vue, insertion en partant de la fin…) ;
- le corrigé et les références du correcteur donnent les mêmes résultats qu'un calcul naïf, écrit
  autrement (relire toutes les notes, parcourir tout le vocabulaire, essayer tous les passages), et que
  BM25 donne, sur un petit exemple, la valeur calculée à la main ;
- les exemples des docstrings, les outils communs et le corrigé sont justes.
"""

from __future__ import annotations

import bisect
import collections
import doctest
import heapq
import math
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.corrige_serie_30 as corrige  # noqa: E402
import defis.recherche as recherche  # noqa: E402
from defis.checks_serie_30 import SERIE, _ref_classer, _ref_completer, _ref_extrait  # noqa: E402
from defis.recherche import (  # noqa: E402
    Comptee,
    generer_notes,
    generer_vocabulaire,
    index_frequences,
    mots,
    mots_et_positions,
    normaliser,
    notes_exemple,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}), DEFIS[nom])


def _echec(nom, objet, *morceaux):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


# --------------------------------------------------------------------------- Défi 182 : index_inverse


def _index_inverse(defaut=None):
    """Un index écrit autrement que le corrigé (on trie À LA FIN), avec UN défaut."""

    def index_inverse(notes):
        index = collections.defaultdict(list)
        cles = list(notes) if defaut == "ordre" else sorted(notes)
        for identifiant in cles:
            texte = notes[identifiant]
            if defaut == "str_none":
                texte = str(texte)
            elif defaut == "modifie" and not isinstance(texte, str):
                del notes[identifiant]
                continue
            elif defaut != "nan_plante" and not isinstance(texte, str):
                continue
            decoupe = texte.lower().split() if defaut == "split" else mots(texte)
            for mot in (decoupe if defaut == "doublons" else dict.fromkeys(decoupe)):
                index[mot].append(identifiant)
        if defaut == "oublie_return":
            return None
        if defaut == "defaultdict":
            return index
        if defaut == "set":
            return {mot: set(ids) for mot, ids in index.items()}
        return {mot: list(ids) for mot, ids in index.items()}

    return index_inverse


@pytest.mark.parametrize("defaut,morceaux", [
    ("split", ("mots(texte)", "jamais trouvée")),
    ("doublons", ("qu'une fois", "set(mots(texte))")),
    ("ordre", ("pas trié", "sorted(notes)")),
    ("defaultdict", ("defaultdict", "dict(index)")),
    ("set", ("set", "dichotomie")),
    ("nan_plante", ("None ou NaN", "isinstance(texte, str)")),
    ("str_none", ("str(None)", "AUCUN mot")),
    ("modifie", ("a modifié",)),
    ("oublie_return", ("return",)),
])
def test_index_inverse_nomme_l_erreur(defaut, morceaux):
    _echec("index_inverse", _index_inverse(defaut), *morceaux)


def test_index_inverse_accepte_d_autres_ecritures():
    _reussi("index_inverse", _index_inverse())

    def avec_compteur(notes):
        paires = sorted({(mot, i) for i, texte in notes.items() if isinstance(texte, str) for mot in mots(texte)})
        index = {}
        for mot, i in paires:
            index.setdefault(mot, []).append(i)
        return index

    _reussi("index_inverse", avec_compteur)


# --------------------------------------------------------------------------- Défi 183 : rechercher


def _rechercher(defaut=None):
    """Une recherche juste, écrite autrement (dichotomie SANS repartir de la dernière position), avec UN défaut."""

    def rechercher(index, requete):
        termes = requete.lower().split() if defaut == "split" else mots(requete)
        termes = list(dict.fromkeys(termes))
        if not termes:
            return sorted({i for ids in index.values() for i in ids}) if defaut == "vide_tout" else []
        if defaut == "keyerror":
            listes = [index[t] for t in termes]
        else:
            if any(t not in index for t in termes):
                return [] if defaut != "union" else sorted({i for t in termes for i in index.get(t, [])})
            listes = [index[t] for t in termes]
        if defaut == "union":
            return sorted({i for liste in listes for i in liste})
        if defaut == "tri_en_place":
            listes[0].sort()
        if defaut == "reaffecte":
            for t in termes:
                index[t] = sorted(index[t])
            listes = [index[t] for t in termes]
        if defaut == "ensembles":
            return sorted(set.intersection(*(set(liste) for liste in listes)))
        listes = sorted(listes, key=len, reverse=defaut == "plus_longue_dabord")
        if len(listes) == 1 and defaut == "meme_liste":
            return listes[0]
        if defaut == "in_liste":
            return [i for i in listes[0] if all(i in autre for autre in listes[1:])]
        if defaut == "fusion":
            resultat = list(listes[0])
            for autre in listes[1:]:
                a, b, commun = 0, 0, []
                while a < len(resultat) and b < len(autre):
                    x, y = resultat[a], autre[b]
                    if x == y:
                        commun.append(x)
                        a, b = a + 1, b + 1
                    elif x < y:
                        a += 1
                    else:
                        b += 1
                resultat = commun
            return resultat
        resultat = []
        for candidat in listes[0]:
            if all((p := bisect.bisect_left(autre, candidat)) < len(autre) and autre[p] == candidat
                   for autre in listes[1:]):
                resultat.append(candidat)
        return resultat[::-1] if defaut == "decroissant" else resultat

    return rechercher


@pytest.mark.parametrize("defaut,morceaux", [
    ("keyerror", ("index.get(mot)",)),
    ("union", ("UNION", "TOUS les mots")),
    ("split", ("mots(requete)",)),
    ("vide_tout", ("sans aucun mot",)),
    ("decroissant", ("pas trié",)),
    ("meme_liste", ("ELLE-MÊME",)),
    ("tri_en_place", ("ne se modifient pas", "sorted(listes, key=len)")),
    ("reaffecte", ("a modifié l'index",)),
    ("ensembles", ("TOUTE une liste longue", "bisect.bisect_left")),
    ("in_liste", ("RELISEZ", "bisect.bisect_left")),
    ("fusion", ("TOUTE une liste longue", "plus COURTE")),
    ("plus_longue_dabord", ("plus COURTE",)),
])
def test_rechercher_nomme_l_erreur(defaut, morceaux):
    _echec("rechercher", _rechercher(defaut), *morceaux)


def test_rechercher_premiere_ecriture_d_un_debutant():
    """Partir du premier mot de la requête, et `in` sur les autres : juste, mais la liste de 5 est relue 200 000 fois."""

    def rechercher(index, requete):
        termes = mots(requete)
        if not termes:
            return []
        return [i for i in index.get(termes[0], []) if all(i in index.get(t, []) for t in termes[1:])]

    _echec("rechercher", rechercher, "RELISEZ", "qui n'a que 5 identifiants")


def test_rechercher_accepte_d_autres_ecritures():
    _reussi("rechercher", _rechercher())

    def adaptative(index, requete):
        """Fusion quand les listes ont des tailles proches, dichotomie sinon : ce que font les vrais moteurs."""
        termes = set(mots(requete))
        if not termes or any(t not in index for t in termes):
            return []
        listes = sorted((index[t] for t in termes), key=len)
        resultat = list(listes[0])
        for autre in listes[1:]:
            if len(autre) <= 4 * len(resultat):
                a = b = 0
                commun = []
                while a < len(resultat) and b < len(autre):
                    y = autre[b]
                    if resultat[a] == y:
                        commun.append(y)
                        a, b = a + 1, b + 1
                    elif resultat[a] < y:
                        a += 1
                    else:
                        b += 1
            else:
                commun, depart = [], 0
                for x in resultat:
                    depart = bisect.bisect_left(autre, x, depart)
                    if depart < len(autre) and autre[depart] == x:
                        commun.append(x)
            resultat = commun
        return resultat

    _reussi("rechercher", adaptative)


def test_le_budget_de_lecture_laisse_passer_la_fusion_de_listes_equilibrees():
    """Deux listes de 5 000 : les parcourir ensemble (10 000 lectures) est raisonnable, et accepté."""
    index = {"fibre": Comptee(range(0, 10_000, 2)), "mobile": Comptee(range(0, 15_000, 3))}
    assert _rechercher("fusion")(index, "fibre mobile") == list(range(0, 10_000, 6))
    index_frais = {"fibre": Comptee(range(0, 10_000, 2)), "mobile": Comptee(range(0, 15_000, 3))}
    from defis.checks_serie_30 import _recherche_comptee
    _recherche_comptee(_rechercher("fusion"), {"fibre": list(range(0, 10_000, 2)), "mobile": list(range(0, 15_000, 3))},
                       "fibre mobile", "rechercher(…)")
    assert index_frais["fibre"].lues == 0


def test_rechercher_donne_le_meme_resultat_qu_une_lecture_de_toutes_les_notes():
    notes = generer_notes(800, graine=30)
    index = corrige.index_inverse(notes)
    alea = random.Random(30)
    vocabulaire = sorted(index)
    mots_des_notes = {i: set(mots(texte)) if isinstance(texte, str) else set() for i, texte in notes.items()}
    for _ in range(300):
        requete = " ".join(alea.choice(vocabulaire) for _ in range(alea.randint(1, 3)))
        naif = sorted(i for i, contenus in mots_des_notes.items() if set(mots(requete)) <= contenus)
        assert corrige.rechercher(index, requete) == naif, requete


# --------------------------------------------------------------------------- Défi 184 : completer


def _completer(defaut=None):
    """Une complétion juste, écrite autrement (la fin de plage par dichotomie), avec UN défaut."""

    def completer(vocabulaire, frequences, prefixe, k=5):
        if defaut == "sans_normaliser":
            p = prefixe
        elif defaut == "sans_strip":
            p = normaliser(prefixe)
        else:
            p = normaliser(prefixe).strip()
        if k <= 0 and defaut != "k_negatif":
            return []
        if defaut == "tout_lire":
            candidats = [m for m in vocabulaire if m.startswith(p)]
        else:
            debut = (bisect.bisect_right if defaut == "bisect_right" else bisect.bisect_left)(vocabulaire, p)
            fin_de_plage = {"fin_z": p + "z", "fin_tilde": p + "~"}.get(defaut, p + "\U0010ffff")
            fin = bisect.bisect_left(vocabulaire, fin_de_plage, debut)
            candidats = vocabulaire[debut:fin]
        if defaut == "reverse":
            return sorted(candidats, key=lambda m: (frequences[m], m), reverse=True)[:k]
        if defaut == "croissant":
            return sorted(candidats, key=lambda m: (frequences[m], m))[:k]
        return sorted(candidats, key=lambda m: (-frequences[m], m))[:k]

    return completer


@pytest.mark.parametrize("defaut,morceaux", [
    ("sans_normaliser", ("Normalisez", "sans accents")),
    ("sans_strip", ("Normalisez", "strip")),
    ("bisect_right", ("bisect_left", "égal au préfixe")),
    ("fin_z", ("préfixe + 'z'", "startswith")),
    ("fin_tilde", ("cœur",)),
    ("reverse", ("reverse=True", "INVERSE")),
    ("croissant", ("FRÉQUENTS",)),
    ("k_negatif", ("k <= 0",)),
    ("tout_lire", ("bisect.bisect_left", "TRIÉ")),
])
def test_completer_nomme_l_erreur(defaut, morceaux):
    _echec("completer", _completer(defaut), *morceaux)


def test_completer_accepte_d_autres_ecritures():
    _reussi("completer", _completer())

    def caractere_suivant(vocabulaire, frequences, prefixe, k=5):
        """La fin de plage : le préfixe dont la DERNIÈRE lettre est remplacée par la suivante (« ra » → « rb »)."""
        p = normaliser(prefixe).strip()
        if k <= 0:
            return []
        debut = bisect.bisect_left(vocabulaire, p)
        fin = len(vocabulaire) if not p else bisect.bisect_left(vocabulaire, p[:-1] + chr(ord(p[-1]) + 1))
        return heapq.nsmallest(k, (vocabulaire[i] for i in range(debut, fin)), key=lambda m: (-frequences[m], m))

    _reussi("completer", caractere_suivant)


def test_la_reference_de_completer_egale_un_parcours_complet():
    vocabulaire, frequences = generer_vocabulaire(5_000, graine=3)
    for prefixe in ("", "b", "ba", "zé", "Tra", "  co ", "xyz", vocabulaire[1234], vocabulaire[42][:3]):
        p = normaliser(prefixe).strip()
        naif = sorted((m for m in vocabulaire if m.startswith(p)), key=lambda m: (-frequences[m], m))[:7]
        assert _ref_completer(vocabulaire, frequences, prefixe, 7) == naif
        assert corrige.completer(vocabulaire, frequences, prefixe, 7) == naif


# --------------------------------------------------------------------------- Défi 185 : surligner


def _surligner(defaut=None):
    """Un surlignage juste, écrit autrement (insertions en partant de la FIN), avec UN défaut."""

    def surligner(texte, requete, avant="[", apres="]"):
        termes = set(mots(requete))
        if defaut == "replace":
            for terme in sorted(termes, key=len, reverse=True):
                texte = texte.replace(terme, f"{avant}{terme}{apres}")
            return texte
        if defaut == "replace_formes":                       # les bonnes formes… remplacées PARTOUT
            for forme in {texte[d:f] for mot, d, f in mots_et_positions(texte) if mot in termes}:
                texte = texte.replace(forme, f"{avant}{forme}{apres}")
            return texte
        if defaut == "regex_ignorecase":
            for terme in termes:
                texte = re.sub(rf"\b({re.escape(terme)})\b", lambda m: f"{avant}{m.group(1)}{apres}", texte,
                               flags=re.IGNORECASE)
            return texte
        if defaut == "regex_w":
            trouves = [(normaliser(m.group()), m.start(), m.end()) for m in re.finditer(r"\w+", texte)]
        else:
            if not isinstance(texte, str):
                raise TypeError("texte")
            trouves = mots_et_positions(texte)
        positions = [(debut, fin) for mot, debut, fin in trouves if mot in termes]
        if defaut == "normalise":
            texte = normaliser(texte)
        if defaut == "decale":
            for debut, fin in positions:                     # de gauche à droite : les suivantes se décalent
                texte = texte[:debut] + avant + texte[debut:fin] + apres + texte[fin:]
            return texte
        for debut, fin in reversed(positions):               # de droite à gauche : rien ne bouge avant
            texte = texte[:debut] + avant + texte[debut:fin] + apres + texte[fin:]
        return texte

    return surligner


@pytest.mark.parametrize("defaut,morceaux", [
    ("replace", ("str.replace", "NORMALISÉE")),
    ("replace_formes", ("intérieur d'un mot", "ENTIERS")),
    ("regex_ignorecase", ("NORMALISÉE",)),
    ("normalise", ("D'ORIGINE",)),
    ("decale", ("décalées", "FIN")),
    ("regex_w", ("U+0301", "mots_et_positions")),
])
def test_surligner_nomme_l_erreur(defaut, morceaux):
    _echec("surligner", _surligner(defaut), *morceaux)


def test_surligner_accepte_d_autres_ecritures():
    _reussi("surligner", _surligner())


# --------------------------------------------------------------------------- Défi 186 : classer


def _classer(defaut=None):
    """BM25 écrit autrement que le corrigé (note par note, puis un tri complet), avec UN défaut."""

    def classer(requete, frequences, longueurs, k=10, k1=1.2, b=0.75):
        if defaut == "validation_naive":
            if k1 < 0 or b < 0 or b > 1:
                raise ValueError("paramètres")
        elif defaut != "sans_validation":
            if not (k1 >= 0 and 0 <= b <= 1):
                raise ValueError("paramètres")
        termes = requete.lower().split() if defaut == "split" else mots(requete)
        if defaut != "doublons":
            termes = list(dict.fromkeys(termes))
        n = len(longueurs)
        if n == 0 or (k <= 0 and defaut != "k_ignore"):
            return []
        valeurs = [x for x in longueurs.values() if x > 0] if defaut == "moyenne_non_vides" else list(longueurs.values())
        if defaut == "moyenne_candidats":
            valeurs = [longueurs[i] for i in {i for t in termes for i in frequences.get(t, {})}]
        if not valeurs or sum(valeurs) == 0:
            return []
        moyenne = sum(valeurs) / len(valeurs)
        idf = {}
        for terme in termes:
            if terme in frequences:
                df = len(frequences[terme])
                if defaut == "log10":
                    idf[terme] = math.log10(1 + (n - df + 0.5) / (df + 0.5))
                elif defaut == "idf_classique":
                    idf[terme] = math.log((n - df + 0.5) / (df + 0.5))
                elif defaut == "idf_tfidf":
                    idf[terme] = math.log(n / df)
                else:
                    idf[terme] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        notes = longueurs if defaut == "tous" else list(dict.fromkeys(i for t in idf for i in frequences[t]))
        resultat = []
        for i in notes:
            score = 0.0
            for terme in termes:
                if terme not in idf or i not in frequences[terme]:
                    continue
                tf = 1 if defaut == "presence" else frequences[terme][i]
                rapport = {"longueur_inversee": moyenne / max(longueurs[i], 1),
                           "sans_longueur": 1.0}.get(defaut, longueurs[i] / moyenne)
                facteur = 1 if defaut == "sans_k1_plus_1" else k1 + 1
                score += idf[terme] * tf * facteur / (tf + k1 * (1 - b + b * rapport))
            resultat.append((i, score))
        if defaut == "egalites_inverse":
            resultat.sort(key=lambda p: (p[1], p[0]), reverse=True)
        elif defaut == "egalites_par_insertion":
            resultat.sort(key=lambda p: -p[1])        # tri stable : les ex aequo restent dans l'ordre du dict
        else:
            resultat.sort(key=lambda p: (-p[1], p[0]))
        if defaut == "seulement_les_ids":
            return [i for i, _ in resultat[:k]]
        return resultat if defaut == "k_ignore" else resultat[:k]

    return classer


@pytest.mark.parametrize("defaut,morceaux", [
    ("log10", ("math.log10", "NATUREL")),
    ("idf_classique", ("NÉGATIF", "ln(1 + (N - df + 0.5) / (df + 0.5))")),
    ("idf_tfidf", ("TF-IDF",)),
    ("moyenne_non_vides", ("vides", "TOUTES les notes")),
    ("moyenne_candidats", ("TOUTES les notes",)),
    ("longueur_inversee", ("à l'envers",)),
    ("sans_longueur", ("longueur des notes",)),
    ("presence", ("tf = 1",)),
    ("sans_k1_plus_1", ("(k1 + 1)",)),
    ("doublons", ("DISTINCTS",)),
    ("egalites_inverse", ("CROISSANT",)),
    ("egalites_par_insertion", ("CROISSANT",)),
    ("split", ("mots(requete)",)),
    ("tous", ("au moins un mot",)),
    ("k_ignore", ("au plus k",)),
    ("seulement_les_ids", ("paire (identifiant, score)",)),
    ("validation_naive", ("NaN", "not k1 >= 0")),
    ("sans_validation", ("ValueError",)),
])
def test_classer_nomme_l_erreur(defaut, morceaux):
    _echec("classer", _classer(defaut), *morceaux)


def test_classer_accepte_d_autres_ecritures():
    _reussi("classer", _classer())


def test_bm25_redonne_la_valeur_calculee_a_la_main():
    """N = 3 notes d'un mot : moyenne = 1, df(fibre) = 1. Avec dl = moyenne, tf = 1 : score = idf = ln(8/3)."""
    frequences, longueurs = index_frequences({1: "fibre", 2: "client", 3: "client"})
    ((identifiant, score),) = corrige.classer("fibre", frequences, longueurs)
    assert identifiant == 1 and score == pytest.approx(math.log(1 + 2.5 / 1.5)) == pytest.approx(math.log(8 / 3))
    # tf = 2 dans une note deux fois plus longue que la moyenne (b = 0.75, k1 = 1.2) :
    frequences, longueurs = index_frequences({1: "fibre fibre", 2: "", 3: ""})   # N = 3, moyenne = 2/3
    ((_, score),) = corrige.classer("fibre", frequences, longueurs)
    idf = math.log(1 + 2.5 / 1.5)
    assert score == pytest.approx(idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / (2 / 3))))


def test_bm25_du_corrige_egale_un_calcul_fait_depuis_le_texte_des_notes():
    """Le même score, recalculé en relisant le TEXTE des notes (Counter), sans passer par index_frequences."""
    notes = generer_notes(400, graine=186)
    frequences, longueurs = index_frequences(notes)
    textes = {i: (mots(t) if isinstance(t, str) else []) for i, t in notes.items()}
    moyenne = sum(map(len, textes.values())) / len(textes)
    for requete in ("client fibre", "résilier prix", "facture retard concurrent"):
        termes = set(mots(requete))
        attendu = {}
        for i, liste in textes.items():
            compte = collections.Counter(liste)
            score = 0.0
            for t in termes:
                if compte[t]:
                    df = sum(1 for autre in textes.values() if t in autre)
                    idf = math.log(1 + (len(textes) - df + 0.5) / (df + 0.5))
                    score += idf * compte[t] * 2.2 / (compte[t] + 1.2 * (0.25 + 0.75 * len(liste) / moyenne))
            if any(compte[t] for t in termes):
                attendu[i] = score
        classement = corrige.classer(requete, frequences, longueurs, k=1000)
        assert [i for i, _ in classement] == sorted(attendu, key=lambda i: (-attendu[i], i))
        assert all(s == pytest.approx(attendu[i], rel=1e-12) for i, s in classement)
        reference = _ref_classer(requete, frequences, longueurs, k=1000)
        assert [i for i, _ in reference] == [i for i, _ in classement]
        assert all(s == pytest.approx(r, rel=1e-12) for (_, s), (_, r) in zip(classement, reference))


def test_l_idf_classique_devient_negatif_pour_un_mot_frequent():
    """La raison d'être du « 1 + » : sans lui, « client » (dans la plupart des notes) ferait baisser le score."""
    frequences, longueurs = index_frequences(generer_notes(1500, graine=186))
    n, df = len(longueurs), len(frequences["client"])
    assert df > n / 2 and math.log((n - df + 0.5) / (df + 0.5)) < 0 < math.log(1 + (n - df + 0.5) / (df + 0.5))


# --------------------------------------------------------------------------- Défi 187 : meilleur_extrait


def _meilleur_extrait(defaut=None):
    """La dernière position vue de chaque mot (O(n × nombre de mots)), écrite autrement, avec UN défaut."""

    def meilleur_extrait(texte, requete):
        jetons = mots_et_positions(texte)
        termes = set(mots(requete))
        presents = termes & {m for m, _, _ in jetons}
        if defaut == "absent_none" and presents != termes:
            return None
        if not presents:
            return None
        if defaut == "quadratique":
            meilleur = None
            for i in range(len(jetons)):
                vus = set()
                for j in range(i, len(jetons)):
                    if jetons[j][0] in presents:
                        vus.add(jetons[j][0])
                    if len(vus) == len(presents):
                        if meilleur is None or j - i < meilleur[1] - meilleur[0]:
                            meilleur = (i, j)
                        break
        else:
            derniere, meilleur = {}, None
            for j, (mot, debut, fin) in enumerate(jetons):
                if mot in presents:
                    derniere[mot] = j
                    if len(derniere) == len(presents):
                        i = min(derniere.values())
                        taille = (fin - jetons[i][1]) if defaut == "caracteres" else j - i
                        if meilleur is None or taille < meilleur[2] or (defaut == "derniere" and taille == meilleur[2]):
                            meilleur = (i, j, taille)
        i, j = meilleur[0], meilleur[1]
        if defaut == "indices_mots":
            return i, j
        if defaut == "fin_incluse":
            return jetons[i][1], jetons[j][2] - 1
        if defaut == "liste":
            return [jetons[i][1], jetons[j][2]]
        return jetons[i][1], jetons[j][2]

    return meilleur_extrait


@pytest.mark.parametrize("defaut,morceaux", [
    ("quadratique", ("trop lent", "O(n)")),
    ("indices_mots", ("numéros de MOTS",)),
    ("fin_incluse", ("EXCLUE",)),
    ("caracteres", ("en MOTS",)),
    ("derniere", ("PREMIER", "« < »")),
    ("absent_none", ("ABSENTS du texte sont ignorés",)),
    ("liste", ("tuple",)),
])
def test_meilleur_extrait_nomme_l_erreur(defaut, morceaux):
    _echec("meilleur_extrait", _meilleur_extrait(defaut), *morceaux)


def test_meilleur_extrait_accepte_d_autres_ecritures():
    _reussi("meilleur_extrait", _meilleur_extrait())


def test_le_corrige_de_meilleur_extrait_egale_la_reference_qui_essaie_tout():
    alea = random.Random(7)
    for _ in range(2_000):
        texte = " ".join(alea.choice(["fibre", "Résilier", "prix", "la", "x"]) for _ in range(alea.randint(0, 18)))
        requete = " ".join(alea.sample(["fibre", "resilier", "prix", "wifi"], alea.randint(0, 3)))
        assert corrige.meilleur_extrait(texte, requete) == _ref_extrait(texte, requete), (texte, requete)


# --------------------------------------------------------------------------- Les outils communs


def test_mots_normalise_et_garde_les_positions_d_origine():
    texte = "  RÉSILIER, la fibre ! Résilier ; l'offre_spéciale ﬁbre 😡 Straße"
    jetons = mots_et_positions(texte)
    assert [m for m, _, _ in jetons] == ["resilier", "la", "fibre", "resilier", "l", "offre", "speciale", "fibre",
                                         "strasse"]
    assert [texte[d:f] for _, d, f in jetons][:4] == ["RÉSILIER", "la", "fibre", "Résilier"]
    assert mots(texte) == [m for m, _, _ in jetons]
    with pytest.raises(TypeError):
        mots(None)
    assert normaliser("Œuvre") == "œuvre"          # « œ » n'est pas « oe » : c'est une lettre à part entière


def test_comptee_compte_chaque_case_lue_et_ne_se_modifie_pas():
    liste = Comptee([2, 4, 6, 8])
    assert len(liste) == 4 and liste.lues == 0
    assert liste[1:3] == [4, 6] and liste.lues == 2
    assert list(liste) == [2, 4, 6, 8] and liste.lues == 6
    assert 5 not in liste and liste.lues == 10
    assert bisect.bisect_left(liste, 6) == 2
    assert liste == [2, 4, 6, 8] and liste != [1]
    for operation in (lambda: liste.sort(), lambda: liste.append(3), lambda: liste.__setitem__(0, 1)):
        with pytest.raises((AttributeError, TypeError)):
            operation()
    assert "1000000 valeurs" in repr(Comptee(range(1_000_000)))


def test_les_generateurs_sont_reproductibles_et_realistes():
    assert generer_notes(300, graine=4) == generer_notes(300, graine=4) != generer_notes(300, graine=5)
    notes = generer_notes(2000, graine=4)
    frequences, longueurs = index_frequences(notes)
    assert len(frequences["client"]) > 1000 > len(frequences["fibre"]) > 50     # des mots partout, d'autres rares
    assert sum(1 for x in longueurs.values() if x == 0) > 10                   # quelques notes vides
    vocabulaire, freq = generer_vocabulaire(3000, graine=4)
    assert vocabulaire == sorted(set(vocabulaire)) and all(normaliser(m) == m for m in vocabulaire)
    assert set(freq) == set(vocabulaire)


def test_les_exemples_ont_les_pieges_annonces():
    notes = notes_exemple()
    assert list(notes) != sorted(notes)
    assert notes[42] == "" and mots(notes[36]) == [] and "́" in notes[19]
    assert mots(notes[30]).count("fibre") == 2
    assert notes_exemple() is not notes_exemple()


# --------------------------------------------------------------------------- La série


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(182, 188))


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_30 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(recherche)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_30 import *` (dans les exemples) doit donner les fonctions du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 60 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()

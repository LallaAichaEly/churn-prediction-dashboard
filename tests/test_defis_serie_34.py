"""Série 34 (partager sans exposer) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 206`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (`hmac.digest`, des index construits d'avance,
  `defaultdict`, une recherche par hauteur, une comptabilité en `Fraction`…) ;
- le corrigé fait comme les outils de référence : pandas (`groupby(dropna=False)`), une jointure pandas pour
  l'attaque, la fonction de répartition de Laplace ;
- les exemples des docstrings, les outils communs et le corrigé sont justes.
"""

from __future__ import annotations

import copy
import doctest
import hashlib
import hmac
import itertools
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.checks_serie_34 as checks  # noqa: E402
import defis.confidentialite as confidentialite  # noqa: E402
import defis.corrige_serie_34 as corrige  # noqa: E402
from defis.checks_serie_34 import SERIE, _ref_audit, _ref_generaliser, _ref_reid, _ref_secret  # noqa: E402
from defis.confidentialite import (  # noqa: E402
    CLE_EXEMPLE,
    COLONNES_ETUDE,
    HIERARCHIES,
    IDENTIFIANTS_DIRECTS,
    QUASI_IDENTIFIANTS,
    BudgetEpuise,
    Classe,
    annuaire_exemple,
    appliquer,
    clients_exemple,
    comptages_exemple,
    generer_annuaire,
    generer_clients,
    tirage_hasard,
    tirages_fixes,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet, **autres):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}, **autres), DEFIS[nom])


def _echec(nom, objet, *morceaux, **autres):
    resultat = _evaluer(nom, objet, **autres)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message
    return resultat.message


def _reussi(nom, objet, **autres):
    resultat = _evaluer(nom, objet, **autres)
    assert resultat.statut == REUSSI, resultat.message


def _manquant(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


# --------------------------------------------------------------------------- Défi 206 : preparer_export


def _export(defaut=None):
    """Écrite autrement que le corrigé (hmac.digest, compréhension de dict), avec UN défaut."""

    def preparer_export(lignes, colonnes, cle):
        if defaut != "cle_non_verifiee" and (not isinstance(cle, bytes) or len(cle) < 16):
            raise ValueError("clé trop courte")
        if defaut != "directs_acceptes" and IDENTIFIANTS_DIRECTS.intersection(colonnes):
            raise ValueError("identifiant direct demandé")
        sorties = []
        for ligne in lignes:
            identifiant = ligne.get("customer_id")
            if not isinstance(identifiant, str) or not identifiant.strip():
                if defaut == "sans_id_accepte":
                    continue
                raise ValueError("pas d'identifiant")
            texte = identifiant.strip().upper()
            if defaut == "sans_normaliser":
                texte = identifiant
            elif defaut == "lower":
                texte = identifiant.strip().lower()
            octets = texte.encode("latin-1" if defaut == "latin1" else "utf-8")
            if defaut == "sha256":
                pseudo = hashlib.sha256(octets).hexdigest()
            elif defaut == "cle_concatenee":
                pseudo = hashlib.sha256(cle + octets).hexdigest()
            elif defaut == "md5":
                pseudo = hmac.new(cle, octets, hashlib.md5).hexdigest()
            elif defaut == "cle_exemple":
                pseudo = hmac.digest(CLE_EXEMPLE, octets, "sha256").hex()
            elif defaut == "digest":
                pseudo = hmac.digest(cle, octets, "sha256")
            elif defaut == "hash_python":
                pseudo = hash((cle, texte))
            else:
                pseudo = hmac.digest(cle, octets, "sha256").hex()
            if defaut == "tronque":
                pseudo = pseudo[:16]
            if defaut == "liste_noire":
                sortie = {"pseudo": pseudo, **{c: v for c, v in ligne.items() if c not in IDENTIFIANTS_DIRECTS}}
            elif defaut == "tout":
                sortie = {"pseudo": pseudo, **ligne}
            else:
                sortie = {"pseudo": pseudo, **{c: ligne.get(c) for c in colonnes}}
            if defaut == "modifie":
                ligne.pop("email", None)
            sorties.append(sortie)
        return sorties

    return preparer_export


def test_preparer_export_juste_ecrit_autrement():
    _reussi("preparer_export", _export())


@pytest.mark.parametrize("defaut,morceaux", [
    ("sha256", ["SANS clé", "un million"]),
    ("cle_concatenee", ["pas un HMAC", "défi 85"]),
    ("sans_normaliser", ["Normalisez", "strip().upper()"]),
    ("lower", ["pas `.lower()`"]),
    ("md5", ["MD5"]),
    ("latin1", ["UTF-8"]),
    ("cle_exemple", ["CLE_EXEMPLE"]),
    ("digest", [".hexdigest()"]),
    ("hash_python", ["hash()"]),
    ("tronque", ["gardez les 64", "anniversaires"]),
    ("liste_noire", ["LISTE BLANCHE", "pas une liste noire"]),
    ("tout", ["identifiant direct", "JAMAIS"]),
    ("cle_non_verifiee", ["ValueError", "16 octets"]),
    ("directs_acceptes", ["ValueError", "identifiant direct"]),
    ("sans_id_accepte", ["ValueError", "sans identifiant"]),
    ("modifie", ["modifié"]),
])
def test_preparer_export_erreurs_classiques(defaut, morceaux):
    _echec("preparer_export", _export(defaut), *morceaux)


def test_un_pseudonyme_sans_cle_se_retourne_en_essayant_tous_les_identifiants():
    """Ce que dit le message du défi 206 : sha256 sans clé ne cache rien, le HMAC si."""
    sans_cle = {hashlib.sha256(f"C{i:05d}".encode()).hexdigest(): f"C{i:05d}" for i in range(1000)}
    export = corrige.preparer_export(clients_exemple(), [], CLE_EXEMPLE)
    cibles = [hashlib.sha256(c["customer_id"].encode()).hexdigest() for c in clients_exemple()]
    assert [sans_cle[c] for c in cibles] == [c["customer_id"] for c in clients_exemple()]   # tout le monde retrouvé
    assert not any(ligne["pseudo"] in sans_cle for ligne in export)                         # personne


# --------------------------------------------------------------------------- Défi 207 : reidentifier


def _reid(defaut=None):
    """Écrite autrement que le corrigé : TOUS les index construits d'avance (defaultdict), avec UN défaut."""

    def manquant(x):
        return x is None or (x != x if defaut != "nan_valeur" else False)

    def reidentifier(publication, annuaire, quasi_identifiants):
        quasi = list(quasi_identifiants)
        if not quasi and defaut != "sans_controle":
            raise ValueError("aucun quasi-identifiant")
        if defaut == "modifie":
            for ligne in publication:
                ligne.pop("motif", None)
        if defaut == "double_boucle":
            trouves = {}
            for i, ligne in enumerate(publication):
                noms = [p["nom"] for p in annuaire
                        if all(manquant(ligne.get(q)) or ligne.get(q) == p.get(q) for q in quasi)]
                if len(noms) == 1:
                    trouves[i] = noms[0]
            return trouves
        index = {}
        for motif in itertools.product((True, False), repeat=len(quasi)):
            noms = defaultdict(list)
            for personne in annuaire:
                cle = []
                for q, connu in zip(quasi, motif):
                    if connu:
                        v = personne.get(q)
                        cle.append("JOKER" if defaut == "annuaire_joker" and manquant(v) else
                                   (None if manquant(v) else v))
                noms[tuple(cle)].append(personne["nom"])
            index[motif] = noms
        trouves = {}
        for i, ligne in enumerate(publication):
            if defaut == "none_valeur":
                motif = tuple(True for _ in quasi)
                cle = tuple(None if manquant(ligne.get(q)) else ligne.get(q) for q in quasi)
            else:
                motif = tuple(not manquant(ligne.get(q)) for q in quasi)
                cle = tuple(ligne.get(q) for q, c in zip(quasi, motif) if c)
            noms = list(index[motif].get(cle, []))
            if defaut == "annuaire_joker":
                for autre, liste in index[motif].items():
                    if autre != cle and "JOKER" in autre and all(a in ("JOKER", b) for a, b in zip(autre, cle)):
                        noms += liste
            if defaut == "noms_distincts":
                noms = sorted(set(noms))
            if len(noms) == 1 or (defaut == "au_moins_un" and noms):
                trouves[i] = noms[0]
        if defaut == "inverse":
            return {nom: i for i, nom in trouves.items()}
        return trouves

    return reidentifier


def test_reidentifier_juste_ecrit_autrement():
    _reussi("reidentifier", _reid())


@pytest.mark.parametrize("defaut,morceaux", [
    ("none_valeur", ["ne filtre RIEN"]),
    ("nan_valeur", ["NaN", "x != x"]),
    ("annuaire_joker", ["ANNUAIRE", "aucune valeur connue"]),
    ("au_moins_un", ["UN SEUL candidat"]),
    ("noms_distincts", ["homonymes"]),
    ("inverse", ["l'inverse"]),
    ("double_boucle", ["beaucoup trop lent", "dictionnaire"]),
    ("sans_controle", ["ValueError"]),
    ("modifie", ["modifié"]),
])
def test_reidentifier_erreurs_classiques(defaut, morceaux):
    _echec("reidentifier", _reid(defaut), *morceaux)


def test_reidentifier_comme_une_jointure_pandas():
    """Sans valeur effacée, l'attaque est une jointure : les lignes qui ne trouvent qu'UNE personne."""
    clients = generer_clients(3000, graine=34, manquants=0.0)
    publication = [{q: c[q] for q in QUASI_IDENTIFIANTS} for c in clients]
    annuaire = generer_annuaire(clients, graine=34)
    pub = pd.DataFrame(publication).reset_index(names="ligne")
    ann = pd.DataFrame(annuaire)
    jointure = pub.merge(ann, on=list(QUASI_IDENTIFIANTS))
    uniques = jointure.groupby("ligne").filter(lambda g: len(g) == 1)
    attendu = dict(zip(uniques["ligne"], uniques["nom"]))
    assert corrige.reidentifier(publication, annuaire, QUASI_IDENTIFIANTS) == attendu
    assert _ref_reid(publication, annuaire, list(QUASI_IDENTIFIANTS)) == attendu
    assert 0.2 < len(attendu) / len(publication) < 0.8          # une bonne part des clients retrouvés


def test_reidentifier_reste_rapide_et_le_correcteur_compte_bien():
    """Le corrigé passe sous la limite du correcteur, avec de la marge ; la double boucle non."""
    publication, annuaire = checks._grand_recoupement()
    limite = checks.LIGNES_PAR_LIGNE_207 * (len(publication) + len(annuaire))
    compte = [0]

    def local(frame, evenement, arg):
        if evenement == "line":
            compte[0] += 1
        return local

    fichier = corrige.reidentifier.__code__.co_filename
    sys.settrace(lambda frame, ev, arg: local if frame.f_code.co_filename == fichier else None)
    try:
        corrige.reidentifier(publication, annuaire, list(QUASI_IDENTIFIANTS))
    finally:
        sys.settrace(None)
    assert compte[0] < limite / 3, (compte[0], limite)


# --------------------------------------------------------------------------- Défi 208 : audit


def _audit(defaut=None):
    """Écrite autrement que le corrigé (Counter, defaultdict(set), sorted), avec UN défaut."""

    def audit(lignes, quasi_identifiants, sensible):
        quasi = list(quasi_identifiants)
        if defaut != "sans_controle" and (not lignes or not quasi or sensible in quasi):
            raise ValueError("audit impossible")

        def norme(v):
            if defaut == "nan_distincts":
                return None if v is None else v
            return None if _manquant(v) else v

        effectifs, vues = Counter(), defaultdict(set)
        for ligne in lignes:
            cle = tuple(norme(ligne.get(q)) for q in quasi)
            if defaut == "dropna" and None in cle:
                continue
            effectifs[cle] += 1
            s = ligne.get(sensible)
            vues[cle].add(s if defaut == "sensible_nan" else (None if _manquant(s) else s))
        classes = [Classe(cle, n, n if defaut == "diversite_effectif" else len(vues[cle])) for cle, n in effectifs.items()]
        if defaut == "tri_par_valeurs":
            classes = sorted(classes)
        elif defaut == "tri_decroissant":
            classes = sorted(classes, key=lambda c: (c.effectif, c.diversite), reverse=True)
        elif defaut != "sans_tri":
            classes = sorted(classes, key=lambda c: (c.effectif, c.diversite))
        k = max(c.effectif for c in classes) if defaut == "k_max" else min(c.effectif for c in classes)
        if defaut == "l_globale":
            l = len({None if _manquant(x.get(sensible)) else x.get(sensible) for x in lignes})
        else:
            l = min(c.diversite for c in classes)
        return k, l, classes

    return audit


def test_audit_juste_ecrit_autrement():
    _reussi("audit", _audit())


@pytest.mark.parametrize("defaut,morceaux", [
    ("dropna", ["dropna=False"]),
    ("nan_distincts", ["deux NaN", "AVANT de grouper"]),
    ("sensible_nan", ["colonne sensible", "deux NaN"]),
    ("diversite_effectif", ["DIFFÉRENTES"]),
    ("k_max", ["PLUS PETIT effectif"]),
    ("l_globale", ["classe par classe"]),
    ("tri_decroissant", ["CROISSANT"]),
    ("sans_tri", ["triées par effectif croissant"]),
    ("tri_par_valeurs", ["TypeError", "ne triez pas"]),
    ("sans_controle", ["ValueError"]),
])
def test_audit_erreurs_classiques(defaut, morceaux):
    _echec("audit", _audit(defaut), *morceaux)


def test_audit_comme_pandas_avec_dropna_false():
    """pandas donne le même k et le même l… à condition de garder les valeurs manquantes (dropna=False)."""
    clients = generer_clients(2000, graine=208, manquants=0.05)
    lignes = [{q: c[q] for q in (*QUASI_IDENTIFIANTS, "motif")} for c in clients]
    resultat = corrige.audit(lignes, QUASI_IDENTIFIANTS, "motif")
    df = pd.DataFrame(lignes).astype({"age": "float"})
    garde = df.groupby(list(QUASI_IDENTIFIANTS), dropna=False)
    assert garde.size().min() == resultat.k
    assert garde["motif"].nunique(dropna=False).min() == resultat.l
    assert sorted(garde.size()) == sorted(c.effectif for c in resultat.classes)
    jette = df.groupby(list(QUASI_IDENTIFIANTS))                       # dropna=True par défaut
    assert jette.size().sum() < len(lignes)                           # des clients ont disparu de l'audit


def test_audit_regroupe_toutes_les_formes_de_valeur_manquante():
    lignes = [{"age": None, "m": 1}, {"age": float("nan"), "m": 1}, {"m": 2}, {"age": 30, "m": float("nan")},
              {"age": 30, "m": None}]
    resultat = corrige.audit(lignes, ["age"], "m")
    assert resultat == (2, 1, [Classe((30,), 2, 1), Classe((None,), 3, 2)])


# --------------------------------------------------------------------------- Défi 209 : generaliser


def _gen(defaut=None):
    """Écrite autrement que le corrigé : une hauteur après l'autre, avec UN défaut."""

    def generaliser(lignes, hierarchies, k, suppression_max=0):
        attributs = list(hierarchies)
        if defaut != "sans_controle":
            if not attributs or isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(lignes):
                raise ValueError("k invalide")
            if isinstance(suppression_max, bool) or not isinstance(suppression_max, int) \
                    or not 0 <= suppression_max < len(lignes):
                raise ValueError("suppression_max invalide")
        tailles = [len(hierarchies[a]) for a in attributs]
        smax = 0 if defaut == "sans_suppression" else suppression_max

        def essayer(vecteur):
            cles = [tuple(hierarchies[a][n](x.get(a)) for a, n in zip(attributs, vecteur)) for x in lignes]
            effectifs = Counter(cles)
            petites = [e for e in effectifs.values() if (e <= k if defaut == "inferieur_ou_egal" else e < k)]
            trop = len(petites) if defaut == "classes_comptees" else sum(petites)
            if trop > smax:
                return None
            if defaut == "modifie":
                for x, cle in zip(lignes, cles):
                    x.update(zip(attributs, cle))
                return dict(zip(attributs, vecteur)), list(lignes), 0
            if defaut == "memes_objets" and not any(vecteur):
                return dict(zip(attributs, vecteur)), list(lignes), 0
            gardees = [{**x, **dict(zip(attributs, cle))} for x, cle in zip(lignes, cles)
                       if effectifs[cle] >= k or defaut == "suppression_gardee"]
            supprimees = trop if defaut == "suppression_gardee" else len(lignes) - len(gardees)
            return dict(zip(attributs, vecteur)), gardees, supprimees

        if defaut == "glouton":
            vecteur = [0] * len(attributs)
            while (resultat := essayer(vecteur)) is None:
                distinctes = [(len({hierarchies[a][vecteur[i]](x.get(a)) for x in lignes}), -i)
                              for i, a in enumerate(attributs) if vecteur[i] < tailles[i] - 1]
                vecteur[-max(distinctes)[1]] += 1
            return resultat
        if defaut == "hauteur_max":
            for vecteur in sorted(itertools.product(*(range(t) for t in tailles)), key=max):
                if (resultat := essayer(vecteur)) is not None:
                    return resultat
        for hauteur in range(sum(t - 1 for t in tailles) + 1):
            vecteurs = [v for v in itertools.product(*(range(t) for t in tailles)) if sum(v) == hauteur]
            if defaut == "dernier_a_egalite":
                vecteurs.reverse()
            elif defaut == "ordre_inverse":
                vecteurs.sort(key=lambda v: v[::-1])
            for vecteur in vecteurs:
                resultat = essayer(vecteur)
                if resultat is not None:
                    return resultat
        raise ValueError("aucune généralisation ne suffit")

    return generaliser


def test_generaliser_juste_ecrit_autrement():
    _reussi("generaliser", _gen())


@pytest.mark.parametrize("defaut,morceaux", [
    ("glouton", ["gloutonne", "TOUS les vecteurs"]),
    ("hauteur_max", ["SOMME de ses niveaux"]),
    ("dernier_a_egalite", ["PREMIER vecteur"]),
    ("ordre_inverse", ["varie le MOINS vite"]),
    ("classes_comptees", ["des LIGNES"]),
    ("inferieur_ou_egal", ["`effectif < k`"]),
    ("sans_suppression", ["suppression_max"]),
    ("suppression_gardee", ["restées dans le fichier"]),
    ("modifie", ["modifié"]),
    ("memes_objets", ["COPIES"]),
    ("sans_controle", ["ValueError"]),
])
def test_generaliser_erreurs_classiques(defaut, morceaux):
    _echec("generaliser", _gen(defaut), *morceaux)


def test_generaliser_trouve_le_minimum_de_tous_les_vecteurs_acceptables():
    """Le corrigé rend la plus petite hauteur parmi TOUS les vecteurs acceptables (recherche exhaustive), et la
    k-anonymité qu'il annonce est vraie."""
    alea = random.Random(34)
    for _ in range(40):
        lignes = [{q: c[q] for q in QUASI_IDENTIFIANTS} for c in generer_clients(alea.randint(3, 30),
                                                                                 graine=alea.randint(0, 999))]
        k, smax = alea.randint(1, 4), alea.randint(0, 2)
        if k > len(lignes) or smax >= len(lignes):
            continue
        g = corrige.generaliser(lignes, HIERARCHIES, k, smax)
        acceptables = []
        for v in itertools.product(*(range(len(h)) for h in HIERARCHIES.values())):
            niveaux = dict(zip(HIERARCHIES, v))
            effectifs = Counter(tuple(appliquer(x, HIERARCHIES, niveaux)[q] for q in QUASI_IDENTIFIANTS) for x in lignes)
            if sum(e for e in effectifs.values() if e < k) <= smax:
                acceptables.append(v)
        assert sum(g.niveaux.values()) == min(sum(v) for v in acceptables)
        assert tuple(g.niveaux.values()) == min(v for v in acceptables if sum(v) == sum(g.niveaux.values()))
        assert g.supprimees <= smax and len(g.lignes) + g.supprimees == len(lignes)
        if g.lignes:
            assert corrige.audit(g.lignes, QUASI_IDENTIFIANTS, "absente").k >= k


def test_le_glouton_monte_vraiment_trop_haut_sur_un_cas_du_correcteur():
    """Le piège nommé par le correcteur existe : Datafly ne trouve pas la plus petite hauteur partout."""
    alea = random.Random(209)
    ecarts = 0
    for _ in range(200):
        n = alea.randint(2, 25)
        lignes = [{q: c[q] for q in ("age", "sexe", "code_postal", "churn")}
                  for c in generer_clients(n, graine=alea.randint(0, 10_000), manquants=0.08)]
        attributs = alea.sample(list(HIERARCHIES), alea.randint(1, 3))
        hierarchies = {a: HIERARCHIES[a] for a in attributs}
        k, smax = alea.randint(1, min(n, 5)), alea.randint(0, min(n - 1, 4))
        juste = _ref_generaliser(lignes, hierarchies, k, smax)
        glouton = _ref_generaliser(lignes, hierarchies, k, smax, "glouton")
        assert sum(juste[0].values()) <= sum(glouton[0].values())
        ecarts += sum(juste[0].values()) < sum(glouton[0].values())
    assert ecarts >= 5


# --------------------------------------------------------------------------- Défi 210 : secret_statistique


def _secret(defaut=None):
    """Écrite autrement que le corrigé (compteurs par ligne et par colonne, for … else), avec UN défaut."""

    def secret_statistique(tableau, seuil):
        if defaut != "sans_controle":
            if isinstance(seuil, bool) or not isinstance(seuil, int) or seuil < 1:
                raise ValueError("seuil")
            if len(tableau) < 2 or len(tableau[0]) < 2 or len({len(x) for x in tableau}) != 1:
                raise ValueError("forme")
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for x in tableau for v in x):
                raise ValueError("comptage")
        n, m = len(tableau), len(tableau[0])
        if defaut == "inferieur_ou_egal":
            cache = [[0 < v <= seuil for v in x] for x in tableau]
        elif defaut == "zeros":
            cache = [[v < seuil for v in x] for x in tableau]
        else:
            cache = [[0 < v < seuil for v in x] for x in tableau]
        rangees = [[(i, j) for j in range(m)] for i in range(n)]
        if defaut != "lignes_seules":
            rangees += [[(i, j) for i in range(n)] for j in range(m)]
        def completer(rangee):
            visibles = [(i, j) for i, j in rangee if not cache[i][j]]
            if defaut == "plus_grande":
                i, j = max(visibles, key=lambda p: tableau[p[0]][p[1]])
            elif defaut == "premiere_visible":
                i, j = visibles[0]
            elif defaut == "egalite_derniere":
                i, j = min(reversed(visibles), key=lambda p: tableau[p[0]][p[1]])
            else:
                i, j = min(visibles, key=lambda p: tableau[p[0]][p[1]])
            cache[i][j] = True

        def une_seule(rangee):
            return [cache[i][j] for i, j in rangee].count(True) == 1

        if defaut in ("une_passe", "sans_recommencer"):
            while True:                               # on parcourt tout, sans revenir au début après chaque case
                bouge = False
                for rangee in rangees:
                    if une_seule(rangee):
                        completer(rangee)
                        bouge = True
                if not bouge or defaut == "une_passe":
                    break
        else:
            while True:
                for rangee in rangees:
                    if une_seule(rangee):
                        completer(rangee)
                        break                         # et on recommence depuis la première ligne
                else:
                    break
        if defaut == "entiers":
            return [[int(c) for c in x] for x in cache]
        if defaut == "modifie":
            tableau[0][0] += 0 if cache[0][0] else 1
        return cache

    return secret_statistique


def test_secret_statistique_juste_ecrit_autrement():
    _reussi("secret_statistique", _secret())


@pytest.mark.parametrize("defaut,morceaux", [
    ("lignes_seules", ["COLONNES", "Retrouvée par soustraction"]),
    ("une_passe", ["une seule passe"]),
    ("sans_recommencer", ["RECOMMENCEZ"]),
    ("inferieur_ou_egal", ["`1 <= x < seuil`"]),
    ("zeros", ["case à 0"]),
    ("plus_grande", ["PLUS PETITE"]),
    ("premiere_visible", ["pas la première venue"]),
    ("egalite_derniere", ["PREMIÈRE case"]),
    ("entiers", ["booléens"]),
    ("modifie", ["modifié"]),
    ("sans_controle", ["ValueError"]),
])
def test_secret_statistique_erreurs_classiques(defaut, morceaux):
    _echec("secret_statistique", _secret(defaut), *morceaux)


def test_le_masque_du_corrige_ne_se_resout_pas_par_soustraction():
    """Sur 500 tableaux : toutes les petites cases sont cachées, et aucune rangée n'a UNE seule case cachée."""
    alea = random.Random(34)
    for _ in range(500):
        n, m = alea.randint(2, 7), alea.randint(2, 7)
        tableau = [[alea.choice([0, 1, 2]) if alea.random() < 0.3 else alea.randint(3, 99) for _ in range(m)]
                   for _ in range(n)]
        cache = corrige.secret_statistique(tableau, 3)
        assert all(cache[i][j] for i in range(n) for j in range(m) if 1 <= tableau[i][j] < 3)
        assert checks._attaque(cache, tableau) == ""
        assert cache == _ref_secret(tableau, 3)


def test_la_regle_des_lignes_seules_se_fait_vraiment_attraper():
    tableau = [[40, 2, 9], [30, 50, 60], [70, 80, 1]]
    masque = _ref_secret(tableau, 3, "lignes_seules")
    assert "Retrouvée par soustraction" in checks._attaque(masque, tableau)


# --------------------------------------------------------------------------- Défi 211 : Budget


def _budget(defaut=None):
    """Écrite autrement que le corrigé (une comptabilité en Fraction, une fonction de bruit à deux branches)."""
    import random as module_random

    class Budget:
        partage = []                                    # un attribut de CLASSE (utilisé par le défaut « partage »)

        def __init__(self, epsilon_total, tirage):
            if defaut != "sans_controle" and (isinstance(epsilon_total, bool) or not isinstance(epsilon_total, (int, float))
                                              or not 0 < epsilon_total < math.inf):
                raise ValueError("epsilon_total")
            self.total = Fraction(str(float(epsilon_total)))
            self.depenses = Budget.partage if defaut == "partage" else []
            self.tirage = tirage

        def _reste(self):
            if defaut == "float_exact":
                return float(self.total) - sum(float(e) for e in self.depenses)
            return self.total - sum(self.depenses, Fraction(0))

        if defaut == "restant_methode":
            def restant(self):
                return float(self._reste())
        else:
            @property
            def restant(self):
                return max(0.0, float(self._reste()))

        def _payer(self, epsilon):
            if defaut != "epsilon_non_verifie" and (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
                                                    or not 0 < epsilon < math.inf):
                raise ValueError("epsilon")
            if defaut == "tire_avant_verifier":
                self.u = self.tirage()                # tiré AVANT de savoir si la question passe
            cout = float(epsilon) if defaut == "float_exact" else Fraction(str(epsilon))
            if defaut == "paye_avant_verifier":
                self.depenses.append(cout)
                if self._reste() < 0:
                    raise BudgetEpuise("plus de budget")
                return
            if cout > self._reste():
                raise (ValueError if defaut == "valueerror_budget" else BudgetEpuise)("plus de budget")
            self.depenses.append(cout)

        def _bruit(self, b):
            if defaut == "tire_avant_verifier":
                u = self.u
            else:
                u = module_random.random() if defaut == "random" else self.tirage()
            if defaut == "sans_bruit":
                return 0.0
            if defaut == "deux_tirages":
                self.tirage()
            if defaut == "echelle_epsilon":
                b = 1 / b
            if defaut == "exponentielle":
                return -b * math.log(u)
            if defaut == "sans_signe":
                return -b * math.log(1 - 2 * abs(u - 0.5))
            bruit = b * math.log(2 * u) if u < 0.5 else -b * math.log(2 - 2 * u)
            return -bruit if defaut == "signe_inverse" else bruit

        def compter(self, lignes, condition, epsilon):
            self._payer(epsilon)
            resultat = sum(bool(condition(x)) for x in lignes) + self._bruit(1 / epsilon)
            return float(round(resultat)) if defaut == "arrondi" else resultat

        def sommer(self, lignes, colonne, plafond, epsilon):
            if defaut != "sans_controle" and (isinstance(plafond, bool) or not isinstance(plafond, (int, float))
                                              or not 0 < plafond < math.inf):
                raise ValueError("plafond")
            self._payer(epsilon)
            total = 0.0
            for x in lignes:
                v = x.get(colonne)
                if defaut == "none_plante":
                    v = float("nan") if v is None else v
                    total += min(max(v, 0), plafond)
                    continue
                if v is None or (v != v and defaut != "nan_somme"):
                    continue
                if defaut == "sans_plafond":
                    total += v
                elif defaut == "negatifs":
                    total += min(v, plafond)
                else:
                    total += min(max(v, 0), plafond)
            echelle = 1 / epsilon if defaut == "plafond_echelle" else plafond / epsilon
            return total + self._bruit(echelle)

    return Budget


def test_budget_juste_ecrit_autrement():
    _reussi("Budget", _budget())


@pytest.mark.parametrize("defaut,morceaux", [
    ("echelle_epsilon", ["1 / epsilon", "GRAND"]),
    ("sans_signe", ["toujours positif"]),
    ("signe_inverse", ["signe est inversé"]),
    ("exponentielle", ["EXPONENTIEL"]),
    ("sans_bruit", ["bruitée", "Karim"]),
    ("arrondi", ["non arrondi"]),
    ("random", ["aucun tirage", "module random"]),
    ("deux_tirages", ["UN tirage par question"]),
    ("float_exact", ["0.30000000000000004"]),
    ("paye_avant_verifier", ["ne coûte rien"]),
    ("tire_avant_verifier", ["ne tire rien"]),
    ("valueerror_budget", ["BudgetEpuise"]),
    ("sans_plafond", ["[0, plafond]", "2 500 €"]),
    ("negatifs", ["NÉGATIVES"]),
    ("plafond_echelle", ["SENSIBILITÉ"]),
    ("nan_somme", ["NaN"]),
    ("none_plante", ["NaN"]),
    ("restant_methode", ["@property"]),
    ("partage", ["attribut de CLASSE", "partagé"]),
    ("epsilon_non_verifie", ["ValueError"]),
    ("sans_controle", ["ValueError"]),
])
def test_budget_erreurs_classiques(defaut, morceaux):
    _echec("Budget", _budget(defaut), *morceaux)


def test_le_bruit_du_corrige_suit_la_loi_de_laplace():
    """F(bruit(u)) = u pour la fonction de répartition de Laplace, et E|X| = b sur 20 000 tirages."""
    b = 2.0

    def repartition(x):
        return 0.5 * math.exp(x / b) if x < 0 else 1 - 0.5 * math.exp(-x / b)

    for u in (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999):
        budget = corrige.Budget(1.0, tirages_fixes([u]))
        bruit = budget.compter([], bool, 1 / b)                # vrai résultat 0 : la réponse EST le bruit
        assert math.isclose(repartition(bruit), u, rel_tol=1e-9, abs_tol=1e-12)
    budget = corrige.Budget(1e9, tirage_hasard(211))
    bruits = [budget.compter([], bool, 1 / b) for _ in range(20_000)]
    assert abs(statistics.fmean(bruits)) < 0.1
    assert abs(statistics.fmean(abs(x) for x in bruits) - b) < 0.1
    assert abs(statistics.pvariance(bruits) - 2 * b * b) < 0.5


def test_le_budget_s_epuise_sans_tirer_ni_depenser():
    budget = corrige.Budget(0.3, tirages_fixes([0.5, 0.5, 0.5]))
    for _ in range(3):
        assert budget.compter([1, 2], bool, 0.1) == 2.0
    assert budget.restant == 0.0
    with pytest.raises(BudgetEpuise):
        budget.compter([1, 2], bool, 0.1)                    # tirages_fixes n'a plus rien : aucun tirage fait
    assert 0.1 + 0.1 + 0.1 > 0.3                                 # le piège annoncé existe


# --------------------------------------------------------------------------- Outils communs et exemples


def test_les_exemples_racontent_l_histoire_annoncee():
    clients = clients_exemple()
    export = corrige.preparer_export(clients, COLONNES_ETUDE, CLE_EXEMPLE)
    assert not any(set(ligne) & IDENTIFIANTS_DIRECTS for ligne in export)
    retrouves = corrige.reidentifier(export, annuaire_exemple(), QUASI_IDENTIFIANTS)
    assert len(retrouves) == 13 and all(clients[i]["nom"] == nom for i, nom in retrouves.items())   # tous justes
    assert sum(export[i]["motif"] == "impayé" for i in retrouves) == 5      # les 5 impayés ont retrouvé leur nom
    assert 18 in retrouves and export[18]["age"] is None                    # Claire : âge effacé, retrouvée quand même
    g = corrige.generaliser(export, HIERARCHIES, 3)
    annuaire = [appliquer(p, HIERARCHIES, g.niveaux) for p in annuaire_exemple()]
    assert corrige.reidentifier(g.lignes, annuaire, QUASI_IDENTIFIANTS) == {}
    resultat = corrige.audit(g.lignes, QUASI_IDENTIFIANTS, "motif")
    homogene = resultat.classes[0]
    assert (resultat.k, resultat.l) == (3, 1) and homogene.valeurs[2] == "6912*"
    assert {export[i]["motif"] for i in range(len(export)) if export[i]["code_postal"] == "69120"} == {"impayé"}
    g2 = corrige.generaliser(export, HIERARCHIES, 2, suppression_max=1)
    assert sum(g2.niveaux.values()) < sum(g.niveaux.values()) and g2.supprimees == 1
    masque = corrige.secret_statistique(comptages_exemple(), 3)
    assert sum(map(sum, masque)) == 4 and checks._attaque(masque, comptages_exemple()) == ""


def test_les_outils_communs_sont_reproductibles_et_independants():
    assert clients_exemple() == clients_exemple() and clients_exemple() is not clients_exemple()
    a, b = generer_clients(300, graine=3), generer_clients(300, graine=3)
    assert a == b and a != generer_clients(300, graine=4)
    assert len({c["customer_id"] for c in a}) == 300
    assert any(c["age"] is None for c in generer_clients(2000, graine=1))
    assert generer_annuaire(a, graine=1) == generer_annuaire(a, graine=1)
    tirage = tirage_hasard(5)
    assert all(0 < tirage() < 1 for _ in range(1000))
    with pytest.raises(ValueError):
        tirages_fixes([0.0])
    for nom, niveaux in HIERARCHIES.items():
        assert niveaux[0](None) is None and niveaux[0](float("nan")) is None and niveaux[-1](None) == "*"
    assert [f(47) for f in HIERARCHIES["age"]] == [47, "45-49", "40-49", "40-59", "*"]
    assert [f("69120") for f in HIERARCHIES["code_postal"]] == ["69120", "6912*", "691**", "69***", "*"]


def test_les_references_rejouent_les_bonnes_erreurs():
    lignes = [{"age": 30, "m": "a"}, {"age": None, "m": "b"}]
    assert _ref_audit(lignes, ["age"], "m", "dropna")[2] == [((30,), 1, 1)]
    assert _ref_secret([[10, 2], [20, 30]], 3, "lignes_seules") == [[True, True], [False, False]]
    publication = [{"age": None, "sexe": "M"}]
    annuaire = [{"nom": "Ben", "age": 34, "sexe": "M"}]
    assert _ref_reid(publication, annuaire, ["age", "sexe"]) == {0: "Ben"}
    assert _ref_reid(publication, annuaire, ["age", "sexe"], "none_valeur") == {}


# --------------------------------------------------------------------------- La série


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(206, 212))
    assert any(s is SERIE for s in series())            # enregistrée dans defis/verif.py


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_34 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(confidentialite)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_34 import *` (dans les exemples) doit donner les fonctions du corrigé
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
    clients = clients_exemple()
    tableau = comptages_exemple()
    copies = copy.deepcopy((clients, tableau))
    export = corrige.preparer_export(clients, COLONNES_ETUDE, CLE_EXEMPLE)
    corrige.reidentifier(export, annuaire_exemple(), QUASI_IDENTIFIANTS)
    corrige.audit(clients, QUASI_IDENTIFIANTS, "motif")
    corrige.generaliser(clients, HIERARCHIES, 3, 1)
    corrige.secret_statistique(tableau, 5)
    corrige.Budget(1.0, tirage_hasard(1)).sommer(clients, "facture", 100, 0.5)
    assert (clients, tableau) == copies

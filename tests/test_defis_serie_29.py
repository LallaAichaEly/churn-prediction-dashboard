"""Série 29 (comparer et fusionner) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 176`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées ;
- le corrigé fait comme les vrais outils : difflib (diff unifié), GNU patch, `git merge-file --diff3` ;
- les exemples des docstrings, les outils communs et le corrigé sont justes.
"""

from __future__ import annotations

import copy
import difflib
import doctest
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.corrige_serie_29 as corrige  # noqa: E402
from defis.checks_serie_29 import SERIE  # noqa: E402
from defis.verif import A_FAIRE, ECHEC, ERREUR, REUSSI, evaluer, evaluer_defi  # noqa: E402
from defis.versions import (  # noqa: E402
    BASE,
    EUX,
    NOUS,
    SEPARATEUR,
    PatchRefuse,
    config_exemple,
    generer_versions,
    tables_exemple,
    versions_en_conflit,
    versions_exemple,
)

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}), DEFIS[nom])


def _echec(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


# --------------------------------------------------------------------------- Défi 176 : difference


def _difference(defaut=None):
    """Un script écrit autrement que le corrigé (tableau des PRÉFIXES, reconstruit depuis la fin), avec UN défaut."""

    def difference(ancien, nouveau):
        if defaut != "texte_accepte" and (isinstance(ancien, str) or isinstance(nouveau, str)):
            raise TypeError("des listes de lignes")
        a, b = list(ancien), list(nouveau)
        if defaut == "positionnel":
            ops = []
            for x, y in zip(a, b):
                ops += [(" ", x)] if x == y else [("-", x), ("+", y)]
            return ops + [("-", x) for x in a[len(b):]] + [("+", y) for y in b[len(a):]]
        if defaut == "difflib":
            ops = []
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
                if tag == "equal":
                    ops += [(" ", x) for x in a[i1:i2]]
                else:
                    ops += [("-", x) for x in a[i1:i2]] + [("+", y) for y in b[j1:j2]]
            return ops
        if defaut == "ensembles":
            return [(" ", x) for x in a if x in b] + [("-", x) for x in a if x not in b] + \
                   [("+", y) for y in b if y not in a]
        debut, fin = [], []
        if defaut not in ("sans_tronquer", "strip"):
            k = 0
            while k < len(a) and k < len(b) and a[k] == b[k]:
                k += 1
            debut = [(" ", x) for x in a[:k]]
            if defaut == "modifie_entree":
                del ancien[:k], nouveau[:k]
            a, b = a[k:], b[k:]
            while a and b and a[-1] == b[-1]:
                fin.insert(0, (" ", a.pop()))
                b.pop()
        a_cmp, b_cmp = ([x.strip() for x in a], [x.strip() for x in b]) if defaut == "strip" else (a, b)
        n, m = len(a), len(b)
        t = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                t[i][j] = t[i - 1][j - 1] + 1 if a_cmp[i - 1] == b_cmp[j - 1] else max(t[i - 1][j], t[i][j - 1])
        ops, i, j = [], n, m
        while i or j:
            if i and j and a_cmp[i - 1] == b_cmp[j - 1]:
                ops.append((" ", a[i - 1]))
                i, j = i - 1, j - 1
            elif defaut == "plus_avant_moins" and i and (not j or t[i - 1][j] >= t[i][j - 1]):
                ops.append(("-", a[i - 1]))          # en remontant, « - » d'abord : à l'endroit, « + » avant « - »
                i -= 1
            elif j and (not i or t[i][j - 1] >= t[i - 1][j]):
                ops.append(("+", b[j - 1]))
                j -= 1
            else:
                ops.append(("-", a[i - 1]))
                i -= 1
        ops.reverse()
        if defaut == "sans_return":
            return None
        return debut + ops + fin

    return difference


def _ranger(ops):
    sortie, moins, plus = [], [], []
    for op in ops:
        if op[0] == " ":
            sortie += moins + plus + [op]
            moins, plus = [], []
        else:
            (moins if op[0] == "-" else plus).append(op)
    return sortie + moins + plus


@pytest.mark.parametrize("defaut,mots", [
    ("positionnel", ("position par position", "sous-suite commune")),
    ("difflib", ("difflib", "PLUS COURT")),
    ("ensembles", ("redonner", "ordre")),
    ("plus_avant_moins", ("« - » écrit juste après un « + »",)),
    ("strip", ("DIFFÉRENTES", "strip()")),
    ("sans_tronquer", ("3 000 lignes", "début et la fin communs")),
    ("texte_accepte", ("TypeError", "LETTRE par")),
    ("modifie_entree", ("a modifié",)),
    ("sans_return", ("return",)),
])
def test_difference_erreurs_classiques(defaut, mots):
    _echec("difference", _difference(defaut), *mots)


def test_difference_sans_defaut_un_autre_script_minimal_est_accepte():
    _reussi("difference", _difference())
    a, b = ["a", "a", "b"], ["b", "b", "a"]
    assert _difference()(a, b) != corrige.difference(a, b)    # deux scripts aussi courts : les deux sont justes


def test_difference_des_listes_de_listes_sont_acceptees():
    _reussi("difference", lambda a, b: [list(op) for op in corrige.difference(a, b)])


def test_difference_une_recursion_sans_memoire_est_arretee():
    def difference(ancien, nouveau):
        if isinstance(ancien, str) or isinstance(nouveau, str):
            raise TypeError

        def lcs(i, j):
            if i == len(ancien) or j == len(nouveau):
                return 0
            if ancien[i] == nouveau[j]:
                return 1 + lcs(i + 1, j + 1)
            return max(lcs(i + 1, j), lcs(i, j + 1))

        ops, i, j = [], 0, 0
        while i < len(ancien) and j < len(nouveau):
            if ancien[i] == nouveau[j]:
                ops.append((" ", ancien[i]))
                i, j = i + 1, j + 1
            elif lcs(i + 1, j) >= lcs(i, j + 1):
                ops.append(("-", ancien[i]))
                i += 1
            else:
                ops.append(("+", nouveau[j]))
                j += 1
        return _ranger(ops + [("-", x) for x in ancien[i:]] + [("+", y) for y in nouveau[j:]])

    _echec("difference", difference, "trop lent")


# --------------------------------------------------------------------------- Défi 177 : diff_unifie


def _unifie(defaut=None):
    """Un diff unifié écrit autrement (un bloc ouvert puis fermé au fil des opérations), avec UN défaut."""

    def plage(debut, longueur):
        if defaut == "zero":
            debut -= 1
        if longueur == 0 and defaut != "vide":
            return f"{debut - 1},0"
        if longueur == 1 and defaut != "virgule_un":
            return f"{debut}"
        return f"{debut},{longueur}"

    def diff_unifie(operations, nom="fichier", contexte=3):
        if contexte < 0 and defaut != "contexte_negatif":
            raise ValueError("contexte")
        ops = list(operations)
        changes = [k for k, (op, _) in enumerate(ops) if op != " "]
        if not changes:
            return ["--- a/" + nom, "+++ b/" + nom] if defaut == "entetes_seuls" else []
        seuil = {"seuil_strict": 2 * contexte - 1, "seuil_large": 2 * contexte + 1}.get(defaut, 2 * contexte)
        blocs, courant = [], [changes[0]]
        for k in changes[1:]:
            if k - courant[-1] - 1 > seuil:
                blocs.append(courant)
                courant = []
            courant.append(k)
        blocs.append(courant)
        sortie = [] if defaut == "sans_entete" else ["--- a/" + nom, "+++ b/" + nom]
        for bloc in blocs:
            debut = bloc[0] - contexte if defaut == "negatif" else max(0, bloc[0] - contexte)
            fin = min(len(ops), bloc[-1] + 1 + contexte)
            morceau = ops[debut:fin]
            debut = max(debut, 0)
            ligne_a = 1 + sum(op != "+" for op, _ in ops[:debut])
            ligne_b = 1 + sum(op != "-" for op, _ in ops[:debut])
            n_a = sum(op != "+" for op, _ in morceau)
            n_b = sum(op != "-" for op, _ in morceau)
            sortie.append(f"@@ -{plage(ligne_a, n_a)} +{plage(ligne_b, n_b)} @@")
            for op, ligne in morceau:
                sortie.append(ligne if (defaut == "sans_signe" and op == " ") else op + ligne)
        if defaut == "texte":
            return "\n".join(sortie)
        if defaut == "sans_return":
            return None
        return sortie

    return diff_unifie


@pytest.mark.parametrize("defaut,mots", [
    ("virgule_un", ("« -5 »", "« ,1 »")),
    ("zero", ("à partir de 1",)),
    ("vide", ("ligne d'AVANT", "-5,0")),
    ("seuil_strict", ("EXACTEMENT 2 × contexte", "<= 2 * contexte")),
    ("seuil_large", ("DEUX blocs",)),
    ("negatif", ("max(0, premier - contexte)", "FIN de la liste")),
    ("sans_entete", ("en-têtes", "a/")),
    ("sans_signe", ("une espace",)),
    ("entetes_seuls", ("n'écrit RIEN",)),
    ("texte", ("LISTE de lignes",)),
    ("contexte_negatif", ("ValueError",)),
    ("sans_return", ("return",)),
])
def test_diff_unifie_erreurs_classiques(defaut, mots):
    _echec("diff_unifie", _unifie(defaut), *mots)


def test_diff_unifie_sans_defaut_est_accepte():
    _reussi("diff_unifie", _unifie())


def test_diff_unifie_qui_modifie_les_operations_est_refuse():
    def diff_unifie(operations, nom="fichier", contexte=3):
        resultat = corrige.diff_unifie(operations, nom, contexte)
        operations.clear()
        return resultat

    _echec("diff_unifie", diff_unifie, "a modifié")


def test_le_diff_unifie_du_corrige_est_celui_de_difflib():
    """Quand les opérations viennent de difflib, le corrigé écrit EXACTEMENT ce qu'écrit difflib.unified_diff."""
    import random

    alea = random.Random(29)
    for _ in range(500):
        a = [f"l{i}" for i in range(alea.randrange(0, 30))]
        b = [x + " !" if alea.random() < 0.1 else x for x in a if alea.random() > 0.1]
        if alea.random() < 0.5:
            b.insert(alea.randrange(len(b) + 1), "insérée")
        ops = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            ops += [(" ", x) for x in a[i1:i2]] if tag == "equal" else \
                [("-", x) for x in a[i1:i2]] + [("+", y) for y in b[j1:j2]]
        contexte = alea.randrange(0, 5)
        attendu = list(difflib.unified_diff(a, b, "a/f", "b/f", n=contexte, lineterm=""))
        assert corrige.diff_unifie(ops, nom="f", contexte=contexte) == attendu


# --------------------------------------------------------------------------- Défi 178 : appliquer_patch

_EN_TETE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*")


def _patch(defaut=None):
    """Un patch appliqué autrement (sur une copie qu'on modifie bloc après bloc), avec UN défaut."""
    motif = (r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@.*" if defaut == "virgule_obligatoire"
             else r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@" if defaut == "sans_section"
             else _EN_TETE.pattern)

    def lire(patch):
        blocs, k = [], 0
        while k < len(patch) and not patch[k].startswith("@@"):
            k += 1
        while k < len(patch):
            m = re.fullmatch(motif, patch[k])
            if m is None:
                raise PatchRefuse("en-tête")
            a, b, d = int(m[1]), int(m[2] or 1), int(m[4] or 1)
            k += 1
            attendues, nouvelles = [], []
            if defaut in ("sans_compter", "entete_par_prefixe"):
                while k < len(patch) and not patch[k].startswith("@@"):
                    ligne = patch[k]
                    k += 1
                    if defaut == "entete_par_prefixe" and ligne.startswith(("---", "+++")):
                        continue
                    if ligne[:1] in " -":
                        attendues.append(ligne[1:])
                    if ligne[:1] in " +":
                        nouvelles.append(ligne[1:])
                b = len(attendues)
            else:
                ra, rn = b, d
                while ra > 0 or rn > 0:
                    if k >= len(patch):
                        raise PatchRefuse("tronqué")
                    op, texte = patch[k][:1], patch[k][1:]
                    if op not in (" ", "-", "+"):
                        raise PatchRefuse("ligne")
                    if op in " -":
                        attendues.append(texte)
                        ra -= 1
                    if op in " +":
                        nouvelles.append(texte)
                        rn -= 1
                    if ra < 0 or rn < 0:
                        raise PatchRefuse("compte")
                    k += 1
            blocs.append((a, b, attendues, nouvelles))
        return blocs

    def appliquer_patch(lignes, patch):
        blocs = lire(patch)
        fichier = lignes if defaut == "en_place" else list(lignes)
        ajout = 0           # lignes ajoutées (ou retirées) par les blocs déjà appliqués à `fichier`
        decalage, fin_precedent = 0, 0
        for numero, (a, b, attendues, nouvelles) in enumerate(blocs):
            nominal = a - 1 if (b > 0 or defaut == "zero_moins_un") else a
            prevu = nominal + (0 if defaut == "sans_report" else decalage)
            if defaut == "sans_verification":
                trouve = prevu
            else:
                ecarts = [0] if defaut == "sans_recherche" else [0] + [e for d in range(1, len(lignes) + 1)
                                                                       for e in ((-d, d) if defaut == "moins_dabord"
                                                                                 else (d, -d))]
                trouve = next((prevu + e for e in ecarts
                               if fin_precedent <= prevu + e <= len(lignes) - b
                               and lignes[prevu + e:prevu + e + b] == attendues), None)
                if trouve is None and defaut == "sans_refus":
                    trouve = prevu
                if trouve is None:
                    if defaut == "partiel" and numero > 0:
                        return fichier
                    if defaut == "valueerror":
                        raise ValueError("bloc introuvable")
                    raise PatchRefuse("bloc introuvable")
            decalage = trouve - nominal
            fichier[trouve + ajout:trouve + ajout + b] = nouvelles
            ajout += len(nouvelles) - b
            fin_precedent = trouve + b
        if defaut == "meme_liste" and not blocs:
            return lignes
        return fichier

    return appliquer_patch


@pytest.mark.parametrize("defaut,mots", [
    ("sans_verification", ("3 lignes de plus en haut",)),
    ("sans_refus", ("devrait lever PatchRefuse", "AVANT de le modifier")),
    ("sans_recherche", ("3 lignes de plus en haut", "+1, -1")),
    ("virgule_obligatoire", ("« -5 » sans virgule",)),
    ("sans_section", ("nom de la fonction",)),
    ("zero_moins_un", ("« -5,0 »", "indice 5")),
    ("entete_par_prefixe", ("« --- total par contrat »", "COMPTEZ")),
    ("sans_report", ("SON numéro + 4",)),
    ("moins_dabord", ("PLUS BAS", "+1 avant -1")),
    ("sans_compter", ("n'en a que 2", "PatchRefuse")),
    ("partiel", ("devrait lever PatchRefuse", "Tout ou rien")),
    ("en_place", ("a modifié le fichier",)),
    ("valueerror", ("levez PatchRefuse",)),
    ("meme_liste", ("NOUVELLE liste",)),
])
def test_appliquer_patch_erreurs_classiques(defaut, mots):
    _echec("appliquer_patch", _patch(defaut), *mots)


def test_appliquer_patch_sans_defaut_est_accepte():
    _reussi("appliquer_patch", _patch())


@pytest.mark.skipif(shutil.which("patch") is None, reason="GNU patch n'est pas installé")
def test_le_corrigé_applique_comme_gnu_patch(tmp_path):
    """Partout où GNU patch accepte le patch (avec --fuzz=0), le corrigé donne le même fichier."""
    import random

    alea = random.Random(178)
    compares = 0
    for _ in range(150):
        a = [f"l{i}" for i in range(alea.randrange(5, 50))]
        b = [x + " !" if alea.random() < 0.1 else x for x in a]
        if a == b:
            continue
        patch = corrige.diff_unifie(corrige.difference(a, b), contexte=alea.randrange(1, 4))
        cible = [f"pre {i}" for i in range(alea.randrange(0, 4))] + a
        (tmp_path / "f").write_text("".join(x + "\n" for x in cible))
        (tmp_path / "p").write_text("".join(x + "\n" for x in patch))
        r = subprocess.run(["patch", "-s", "--fuzz=0", "-o", str(tmp_path / "sortie"), str(tmp_path / "f"),
                            str(tmp_path / "p")], capture_output=True, text=True)
        if r.returncode == 0:   # GNU patch « ancre » en plus au début du fichier un bloc au contexte incomplet
            assert corrige.appliquer_patch(cible, patch) == (tmp_path / "sortie").read_text().splitlines()
            compares += 1
    assert compares > 80


# --------------------------------------------------------------------------- Défi 179 : fusion_trois_voies


def _fusion(defaut=None):
    """Une fusion écrite autrement (par blocs de modifications en coordonnées de base), avec UN défaut."""
    marques = ("<<<<<<< HEAD", "||||||| base", "=======", ">>>>>>> autre") if defaut == "etiquettes" \
        else (NOUS, BASE, SEPARATEUR, EUX)

    def blocs(base, autre):
        """[(debut, fin, nouvelles_lignes)] : base[debut:fin] remplacé par nouvelles_lignes."""
        sortie, i, courant = [], 0, None
        for op, ligne in corrige.difference(base, autre):
            if op == " ":
                if courant:
                    sortie.append(tuple(courant))
                    courant = None
                i += 1
                continue
            if courant is None:
                courant = [i, i, []]
            if op == "-":
                i += 1
                courant[1] = i
            else:
                courant[2].append(ligne)
        if courant:
            sortie.append(tuple(courant))
        return [(d, f, tuple(lignes)) for d, f, lignes in sortie]

    def fusion_trois_voies(base, nous, eux):
        if defaut == "deux_voies":
            sortie = []
            for op, ligne in corrige.difference(nous, eux):
                sortie.append(ligne)
            return sortie, 0
        bn, be = blocs(base, nous), blocs(base, eux)
        tous = sorted([(d, f, n, "n") for d, f, n in bn] + [(d, f, n, "e") for d, f, n in be])
        groupes = []    # des blocs qui se chevauchent ou se TOUCHENT (comme Git) forment un même groupe
        for d, f, n, cote in tous:
            if groupes and (d < groupes[-1][1] or (d == groupes[-1][1] and defaut != "voisines")):
                gd, gf, membres = groupes[-1]
                groupes[-1] = (gd, max(gf, f), membres + [(d, f, n, cote)])
            else:
                groupes.append((d, f, [(d, f, n, cote)]))
        sortie, position, conflits = [], 0, 0
        for gd, gf, membres in groupes:
            sortie += base[position:gd]

            def version(cote):
                lignes, i = [], gd
                for d, f, n, c in membres:
                    if c == cote:
                        lignes += base[i:d] + list(n)
                        i = f
                return lignes + base[i:gf]

            n, e, b = version("n"), version("e"), base[gd:gf]
            if n == b:
                sortie += e
            elif e == b or (n == e and defaut != "faux_conflit"):
                sortie += n
            elif defaut == "silencieux":
                sortie += n
            else:
                sortie += [marques[0], *n] + ([] if defaut == "sans_base" else [marques[1], *b]) + \
                          [marques[2], *e, marques[3]]
                conflits += 1
            position = gf
        sortie += base[position:]
        if defaut == "compte_marqueurs":
            conflits = sum(x.startswith(("<<<<<<<", "=======", ">>>>>>>")) for x in sortie)
        if defaut == "modifie_entree":
            base.clear()
        return sortie, conflits

    return fusion_trois_voies


@pytest.mark.parametrize("defaut,mots", [
    ("silencieux", ("ne montre aucun conflit", "CHOISIT")),
    ("faux_conflit", ("n'en est pas un", "EXACTEMENT la même modification")),
    ("sans_base", ("version de BASE", "diff3")),
    ("etiquettes", ("les marqueurs sont exactement",)),
    ("deux_voies", ("BASE", "supprimée")),
    ("voisines", ("se touchent",)),
    ("compte_marqueurs", ("nombre de conflits",)),
    ("modifie_entree", ("a modifié",)),
])
def test_fusion_erreurs_classiques(defaut, mots):
    _echec("fusion_trois_voies", _fusion(defaut), *mots)


def test_fusion_sans_defaut_par_blocs_est_acceptee():
    _reussi("fusion_trois_voies", _fusion())


def test_fusion_qui_renvoie_une_liste_seule_est_expliquee():
    _echec("fusion_trois_voies", lambda b, n, e: corrige.fusion_trois_voies(b, n, e)[0], "paire")


def test_fusion_ecrite_sans_difference_demande_le_defi_176():
    import defis.serie_29 as eleve

    def fusion_trois_voies(base, nous, eux):
        eleve.difference(base, nous)
        return [], 0

    _echec("fusion_trois_voies", fusion_trois_voies, "défi 176")


@pytest.mark.skipif(shutil.which("git") is None, reason="git n'est pas installé")
def test_le_corrige_fusionne_comme_git_merge_file(tmp_path):
    for graine in range(150):
        base, nous, eux = generer_versions(5 + graine % 35, graine=graine, taux=(0.05, 0.15, 0.3)[graine % 3])
        for nom, lignes in (("base", base), ("nous", nous), ("eux", eux)):
            (tmp_path / nom).write_text("".join(x + "\n" for x in lignes))
        r = subprocess.run(["git", "merge-file", "-p", "--diff3", "-L", "nous", "-L", "base", "-L", "eux",
                            str(tmp_path / "nous"), str(tmp_path / "base"), str(tmp_path / "eux")],
                           capture_output=True, text=True)
        lignes, conflits = corrige.fusion_trois_voies(base, nous, eux)
        assert (lignes, conflits) == (r.stdout.splitlines(), r.returncode), graine


# --------------------------------------------------------------------------- Défi 180 : comparer_tables


def _tables(defaut=None):
    """Une comparaison écrite autrement (boucle sur les colonnes, puis sur les clés), avec UN défaut."""

    def nombre(v):
        return isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))

    def egales(x, y):
        if defaut == "nan":
            return x == y
        if pd.isna(x) or pd.isna(y):
            return bool(pd.isna(x) and pd.isna(y))
        if nombre(x) and nombre(y):
            if defaut == "types_stricts" and type(x) is not type(y):
                return False
            return x == y if defaut == "flottants_stricts" else math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)
        if defaut == "en_texte":
            return str(x) == str(y)
        return not (nombre(x) or nombre(y)) and x == y

    def comparer_tables(avant, apres, cle):
        if defaut == "inplace":
            avant["_source"] = "avant"          # « marquer » les lignes… dans la table de l'appelant
        for t in (avant, apres):
            if cle not in t.columns:
                raise KeyError(cle)
            if defaut != "doublons" and (t[cle].duplicated().any() or t[cle].isna().any()):
                raise ValueError("clé en double ou absente")
        a, b = avant.set_index(cle), apres.set_index(cle)
        if defaut == "doublons":
            a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
        colonnes = sorted(set(a.columns) | set(b.columns)) if defaut == "toutes_colonnes" \
            else [c for c in a.columns if c in b.columns]
        if defaut == "position":
            paires = [(a.index[p], a.iloc[p], b.iloc[p]) for p in range(min(len(a), len(b)))]
        else:
            paires = [(k, a.loc[k], b.loc[k]) for k in sorted(set(a.index) & set(b.index))]
        modifies = {}
        for k, ligne_a, ligne_b in paires:
            changements = {c: (ligne_a.get(c), ligne_b.get(c)) for c in colonnes
                           if not egales(ligne_a.get(c), ligne_b.get(c))}
            if changements or defaut == "vides_inclus":
                modifies[k] = changements
        ordre = list if defaut == "non_trie" else sorted
        return {"ajoutes": ordre([k for k in b.index if k not in a.index]),
                "supprimes": ordre([k for k in a.index if k not in b.index]),
                "modifies": modifies,
                "colonnes_ajoutees": ordre([c for c in b.columns if c not in a.columns]),
                "colonnes_supprimees": ordre([c for c in a.columns if c not in b.columns])}

    return comparer_tables


@pytest.mark.parametrize("defaut,mots", [
    ("nan", ("NaN == NaN",)),
    ("flottants_stricts", ("0.1 + 0.2", "isclose")),
    ("en_texte", ("Bourg-en-Bresse", "jamais égal à un nombre")),
    ("types_stricts", ("3 et 3.0",)),
    ("position", ("CLÉ", "position")),
    ("doublons", ("ValueError", ".duplicated()")),
    ("inplace", ("a modifié la table",)),
    ("toutes_colonnes", ("COMMUNES",)),
    ("non_trie", ("Triez",)),
    ("vides_inclus", ("NON VIDE",)),
])
def test_comparer_tables_erreurs_classiques(defaut, mots):
    _echec("comparer_tables", _tables(defaut), *mots)


def test_comparer_tables_par_jointure_est_acceptee():
    _reussi("comparer_tables", _tables())


def test_comparer_tables_qui_renvoie_des_scalaires_numpy_est_acceptee():
    def comparer_tables(avant, apres, cle):
        r = corrige.comparer_tables(avant, apres, cle)
        r["modifies"] = {k: {c: tuple(np.float64(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                                      for v in paire) for c, paire in d.items()} for k, d in r["modifies"].items()}
        return r

    _reussi("comparer_tables", comparer_tables)


# --------------------------------------------------------------------------- Défi 181 : fusion_config

_RIEN = object()


def _config(defaut=None):
    """Une fusion de configuration écrite autrement (une pile de sections à traiter), avec UN défaut."""

    def identiques(a, b):
        if defaut == "egalite_python":
            return a == b
        if a is _RIEN or b is _RIEN:
            return a is b
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return set(a) == set(b) and all(identiques(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(identiques(x, y) for x, y in zip(a, b))
        return a == b

    def fusion_config(base, nous, eux):
        if defaut != "pas_typeerror" and not all(isinstance(x, dict) for x in (base, nous, eux)):
            raise TypeError("dict")
        if not isinstance(nous, dict):
            return {}, []
        racine, conflits = {}, []
        pile = [(base, nous, eux, racine, "")]
        while pile:
            b0, n0, e0, cible, chemin = pile.pop()
            b0 = b0 if isinstance(b0, dict) else {}
            for cle in sorted(set(b0) | set(n0) | set(e0)):
                if defaut == "get_none":
                    b, n, e = b0.get(cle), n0.get(cle), e0.get(cle)
                    rien = None
                else:
                    b, n, e = b0.get(cle, _RIEN), n0.get(cle, _RIEN), e0.get(cle, _RIEN)
                    rien = _RIEN
                ici = cle if defaut == "chemin_court" else (f"{chemin}.{cle}" if chemin else cle)
                if identiques(n, e) or identiques(e, b):
                    choix = n
                elif identiques(n, b):
                    choix = e
                elif isinstance(n, dict) and isinstance(e, dict) and defaut != "sans_recursion":
                    cible[cle] = {}
                    pile.append((b, n, e, cible[cle], ici))
                    continue
                else:
                    if defaut != "eux_gagne":
                        conflits.append(ici)
                    choix = e if defaut == "eux_gagne" else n
                if choix is not rien:
                    cible[cle] = choix if defaut == "partage" else copy.deepcopy(choix)
        if defaut == "modifie_entree":
            nous.clear()
        return racine, (conflits if defaut == "non_trie" else sorted(conflits))

    return fusion_config


@pytest.mark.parametrize("defaut,mots", [
    ("get_none", ("None", "dict.get")),
    ("egalite_python", ("True == 1",)),
    ("sans_recursion", ("récursion",)),
    ("eux_gagne", ("conflits [] au lieu de ['relance.budget_euros']",)),
    ("chemin_court", ("avec des points",)),
    ("partage", ("PARTAGE", "deepcopy")),
    ("modifie_entree", ("a modifié la configuration « nous »",)),
    ("pas_typeerror", ("TypeError",)),
])
def test_fusion_config_erreurs_classiques(defaut, mots):
    _echec("fusion_config", _config(defaut), *mots)


def test_fusion_config_avec_une_pile_est_acceptee():
    _reussi("fusion_config", _config())


def test_fusion_config_non_triee_est_expliquee():
    def fusion_config(base, nous, eux):
        fusion, conflits = corrige.fusion_config(base, nous, eux)
        return fusion, sorted(conflits, reverse=True)

    _echec("fusion_config", fusion_config, "triez")


# --------------------------------------------------------------------------- outils communs, corrigé, docstrings


def test_les_tests_du_corrige_passent():
    tests = [f for nom, f in list(vars(corrige).items()) if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6


def test_le_corrige_reussit_tout():
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6


def test_un_plantage_donne_un_message_lisible():
    def plante(*args, **kwargs):
        return 1 / 0

    assert _evaluer("difference", plante).statut == ECHEC
    assert "ZeroDivisionError" in _evaluer("difference", plante).message
    assert _evaluer("difference", None).statut == ERREUR


def test_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.serie_29 as eleve
    import defis.versions as versions

    for nom in ("BASE", "EUX", "NOUS", "SEPARATEUR", "PatchRefuse", "config_exemple", "generer_versions", "montrer",
                "tables_exemple", "versions_en_conflit", "versions_exemple"):
        assert getattr(eleve, nom) is getattr(versions, nom), nom


def test_les_exemples_sont_neufs_a_chaque_appel():
    a, b = versions_exemple(), versions_exemple()
    a[0].append("modifié")
    assert b[0] != a[0] and versions_exemple()[0] == b[0]
    c1, c2 = config_exemple(), config_exemple()
    c1[0]["modele"]["seuil"] = 0.9
    assert c2[0]["modele"]["seuil"] == 0.35
    t1, _ = tables_exemple()
    t1.loc[0, "contrat"] = "?"
    assert tables_exemple()[0].loc[0, "contrat"] == "Mensuel"


def test_generer_versions_est_reproductible_et_sans_ambiguite():
    for graine in range(30):
        base, nous, eux = generer_versions(60, graine=graine, taux=0.2)
        assert (base, nous, eux) == generer_versions(60, graine=graine, taux=0.2)
        assert len(set(base)) == len(base) and len(set(nous)) == len(nous) and len(set(eux)) == len(eux)
        for autre in (nous, eux):   # aucune ligne de base déplacée : l'ordre des lignes gardées est celui de base
            gardees = [x for x in autre if x in set(base)]
            assert gardees == [x for x in base if x in set(gardees)]
    vus = [generer_versions(80, graine=g, taux=0.2) for g in range(10)]
    assert any(set(n) & set(e) - set(b) for b, n, e in vus)          # des modifications identiques des deux côtés


def test_les_exemples_contiennent_les_pieges_annonces():
    base, nous, eux = versions_exemple()
    assert corrige.fusion_trois_voies(base, nous, eux)[1] == 0
    assert corrige.fusion_trois_voies(*versions_en_conflit())[1] == 1
    avant, apres = tables_exemple()
    assert list(avant["client_id"]) != list(apres["client_id"])                 # un autre ordre
    assert apres.loc[apres["client_id"] == "C003", "facture"].iloc[0] != 0.3     # 0.1 + 0.2
    assert avant["anciennete_mois"].dtype.kind == "i" and apres["anciennete_mois"].dtype.kind == "f"
    assert avant["appels_support"].isna().sum() == 1 and apres["appels_support"].isna().sum() == 1
    b, n, e = config_exemple()
    assert n["relance"]["budget_euros"] != e["relance"]["budget_euros"] != b["relance"]["budget_euros"]


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    """Les exemples de defis/serie_29.py et defis/versions.py, rejoués avec le corrigé."""
    import defis.serie_29 as eleve
    import defis.versions as versions

    tests = doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(versions)
    for nom in DEFIS:  # `from defis.serie_29 import *` (dans les exemples) doit donner les fonctions du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 40 and resultat.failed == 0

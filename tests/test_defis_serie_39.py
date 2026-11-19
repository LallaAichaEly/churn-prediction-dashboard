"""Série 39 (livrer sans tout casser) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 236`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (une version sans expression régulière, une
  résolution sans récursion, `int(hexdigest()[:16], 16)`, un drapeau évalué par une classe…) ;
- les références du correcteur tiennent face à des vérifications indépendantes : l'ordre d'exemple de semver.org,
  un ordre total (antisymétrique et transitif), une recherche exhaustive des combinaisons de paquets, un seau
  identique d'un lancement de Python à l'autre (et `hash()` qui, lui, change) ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (`max(tags)` choisit 2.9.4, la rupture cachée dans un pied impose 3.0.0, l'appli refuse la 3.0.0, la
  résolution doit revenir en arrière, et le nouveau score n'est servi qu'à 52 conseillers).
"""

from __future__ import annotations

import copy
import doctest
import hashlib
import itertools
import os
import random
import subprocess
import sys
import types
from functools import cmp_to_key
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import defis.checks_serie_39 as checks  # noqa: E402
import defis.corrige_serie_39 as corrige  # noqa: E402
import defis.livraison as livraison  # noqa: E402
from defis.checks_serie_39 import (  # noqa: E402
    SERIE,
    _explication,
    _index_long,
    _ref_affecter,
    _ref_comparer,
    _ref_evaluer,
    _ref_prochaine,
    _ref_resoudre,
    _ref_satisfait,
    _seau,
)
from defis.livraison import (  # noqa: E402
    PALIERS,
    Conflit,
    Decision,
    conseillers_exemple,
    drapeaux_exemple,
    generer_versions,
    historique_exemple,
    index_exemple,
    tags_exemple,
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


def _avec(fonction, **remplacements):
    """Une copie de `fonction` (et des fonctions du corrigé qu'elle appelle) qui voit d'autres fonctions globales :
    exactement ce que fait un élève dont l'un des défis précédents est faux."""
    espace = dict(vars(corrige))
    for nom, valeur in vars(corrige).items():
        if isinstance(valeur, types.FunctionType) and valeur.__module__ == corrige.__name__:
            espace[nom] = types.FunctionType(valeur.__code__, espace, nom, valeur.__defaults__, valeur.__closure__)
    espace.update(remplacements)
    return espace[fonction]


# --------------------------------------------------------------------------- Défi 236 : comparer_versions


def _comparer_bogue(bug):
    def comparer_versions(a, b):
        corrige._lire(a), corrige._lire(b)
        return _ref_comparer(a, b, bug)
    return comparer_versions


@pytest.mark.parametrize("bug,morceau", [
    ("texte", "TEXTES"),
    ("sans_pre", "ignorez la pré-version"),
    ("pre_apres", "APRÈS sa version"),
    ("pre_texte", "comme UN texte"),
    ("ids_texte", "NOMBRES"),
    ("numerique_apres", "11.4.3"),
    ("court_apres", "COURTE"),
    ("build", "métadonnées"),
])
def test_comparer_versions_nomme_l_erreur(bug, morceau):
    _echec("comparer_versions", _comparer_bogue(bug), morceau)


def test_comparer_versions_texte_brut():
    _echec("comparer_versions", lambda a, b: (a > b) - (a < b), "TEXTES", "2.10.1")


def test_comparer_versions_ecrit_par_un_eleve():
    def avec_split(a, b):
        """Sans expression régulière : des split et des contrôles à la main."""
        def lire(v):
            if not isinstance(v, str):
                raise TypeError(v)
            v = v.split("+", 1)
            if len(v) == 2 and (not v[1] or any(not x or not all(c.isascii() and (c.isalnum() or c == "-") for c in x)
                                                for x in v[1].split("."))):
                raise ValueError(v)
            noyau, _, pre = v[0].partition("-")
            if _ and not pre:
                raise ValueError(v)
            nombres = noyau.split(".")
            if len(nombres) != 3 or any(not n.isascii() or not n.isdigit() or (len(n) > 1 and n[0] == "0")
                                        for n in nombres):
                raise ValueError(v)
            ids = pre.split(".") if pre else []
            for x in ids:
                if not x or not all(c.isascii() and (c.isalnum() or c == "-") for c in x):
                    raise ValueError(v)
                if x.isdigit() and len(x) > 1 and x[0] == "0":
                    raise ValueError(v)
            cle = [(0, int(x), "") if x.isdigit() else (1, 0, x) for x in ids]
            return [int(n) for n in nombres] + [not ids, cle]
        x, y = lire(a), lire(b)
        return -1 if x < y else (1 if x > y else 0)

    _reussi("comparer_versions", avec_split)
    _reussi("comparer_versions", corrige.comparer_versions)


@pytest.mark.parametrize("version,morceau", [
    ("1.2.3\n", "fullmatch"), ("١.٢.٣", "ASCII"), ("01.2.3", "zéro"), ("1.2.3-01", "NUMÉRIQUE"), ("v1.2.3", "Git"),
])
def test_comparer_versions_trop_tolerant(version, morceau):
    import re

    def tolerant(a, b):
        for v in (a, b):
            if not isinstance(v, str):
                raise TypeError(v)
            if v != version:
                corrige._lire(v)
        return corrige.comparer_versions(*(("0.0.0" if v == version else v) for v in (a, b)))

    message = _echec("comparer_versions", tolerant, "devrait lever ValueError")
    assert morceau in message, message
    assert re.match(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$", "1.2.3\n")   # le piège de « $ » existe bien
    assert re.fullmatch(r"\d", "١") and int("١") == 1 and "²".isdigit()        # et celui de \d aussi


def test_comparer_versions_qui_renvoie_un_bool_ou_une_difference():
    _echec("comparer_versions", lambda a, b: corrige.comparer_versions(a, b) > 0, "-1, 0 ou 1")
    _echec("comparer_versions", lambda a, b: 2 * corrige.comparer_versions(a, b), "-1, 0 ou 1")


def test_reference_comparer_est_un_ordre_total():
    versions = generer_versions(60, graine=7) + checks._CHAINE_SEMVER + tags_exemple()
    for a, b in itertools.product(versions, repeat=2):
        assert _ref_comparer(a, b) == -_ref_comparer(b, a) == corrige.comparer_versions(a, b)
    rng = random.Random(0)
    for _ in range(3000):
        a, b, c = rng.sample(versions, 3)
        if _ref_comparer(a, b) <= 0 and _ref_comparer(b, c) <= 0:
            assert _ref_comparer(a, c) <= 0
    assert sorted(reversed(checks._CHAINE_SEMVER), key=cmp_to_key(_ref_comparer)) == checks._CHAINE_SEMVER


# --------------------------------------------------------------------------- Défi 237 : prochaine_version


def _prochaine_bogue(bug):
    def prochaine_version(actuelle, messages):
        corrige._lire(actuelle)
        for m in messages:
            if not isinstance(m, str):
                raise TypeError(m)
        return _ref_prochaine(actuelle, messages, bug)
    return prochaine_version


@pytest.mark.parametrize("bug,morceau", [
    ("cumul", "CHAQUE commit"),
    ("sans_remise", "repartent de 0"),
    ("pied", "CORPS"),
    ("exclamation", "« ! »"),
    ("partout", "DÉBUT d'une ligne"),
    ("minuscule", "majuscules"),
    ("casse", "casse"),
    ("prefixe", "startswith"),
    ("espace", "trop tolérante"),
    ("zero", "0.y.z"),
    ("pre", "pré-version"),
    ("meme", "None"),
    ("build", "build"),
])
def test_prochaine_version_nomme_l_erreur(bug, morceau):
    _echec("prochaine_version", _prochaine_bogue(bug), morceau)


def test_prochaine_version_ecrit_par_un_eleve():
    import re

    def autrement(actuelle, messages):
        m, n, c, pre = corrige._lire(actuelle)
        niveau = 0
        for message in messages:
            if not isinstance(message, str):
                raise TypeError(message)
            premiere, *corps = message.replace("\r\n", "\n").split("\n")
            t = re.match(r"^(?P<type>[a-zA-Z]+)(\((?P<portee>[^()\n\r]+)\))?(?P<rupture>!)?: (?P<sujet>\S.*)\Z", premiere)
            if not t:
                continue
            if t["rupture"] or any(re.match(r"BREAKING[ -]CHANGE: ", ligne) for ligne in corps):
                niveau = 3
            elif t["type"].lower() == "feat":
                niveau = max(niveau, 2)
            elif t["type"].lower() in ("fix", "perf"):
                niveau = max(niveau, 1)
        if not niveau:
            return None
        if niveau == 3 and m == 0:
            niveau = 2
        base = [m, n, c]
        i = 3 - niveau                        # 0 : majeur, 1 : mineur, 2 : correctif
        if pre and not any(base[i + 1:]):     # une pré-version de ce qu'on publierait : sa version finale
            return "{}.{}.{}".format(*base)
        base[i] += 1
        base[i + 1:] = [0] * (2 - i)
        return "{}.{}.{}".format(*base)

    _reussi("prochaine_version", autrement)


def test_prochaine_version_qui_modifie_les_messages():
    def prochaine_version(actuelle, messages):
        resultat = corrige.prochaine_version(actuelle, messages)
        messages.sort()
        return resultat

    _echec("prochaine_version", prochaine_version, "modifié")


# --------------------------------------------------------------------------- Défi 238 : satisfait


def _satisfait_bogue(bug):
    def satisfait(version, contrainte):
        corrige._lire(version)
        if not isinstance(contrainte, str):
            raise TypeError(contrainte)
        for alternative in contrainte.split("||"):
            corrige._comparateurs(alternative)
        return _ref_satisfait(version, contrainte, bug)
    return satisfait


@pytest.mark.parametrize("bug,morceau", [
    ("texte", "textes"),
    ("caret_zero", "0.3.0"),
    ("caret_sans_borne", "s'arrête à 2.0.0"),
    ("borne_incluse", "EXCLUE"),
    ("tilde_caret", "correctifs"),
    ("pre_toujours", "refuse 2.2.0-rc.1"),
    ("pre_jamais", "EST acceptée"),
    ("pre_global", "MÊME alternative"),
    ("pre_sans_tuple", "MÊMES"),
    ("etoile_pre", "« * »"),
    ("sans_moins_zero", "2.0.0-0"),
    ("et_ou", "lie plus fort"),
])
def test_satisfait_nomme_l_erreur(bug, morceau):
    _echec("satisfait", _satisfait_bogue(bug), morceau)


def test_satisfait_ecrit_par_un_eleve():
    """Les intervalles de npm : chaque comparaison devient [bas, haut[, puis on teste."""
    def par_intervalles(version, contrainte):
        if not isinstance(contrainte, str):
            raise TypeError(contrainte)
        cle = corrige._cle(version)
        pre = corrige._lire(version)[3]
        alternatives = []
        for alternative in contrainte.split("||"):
            bornes, ecrites = [], []
            for morceau in alternative.split(","):
                morceau = morceau.strip()
                if morceau == "*":
                    continue
                op = next((o for o in (">=", "<=", ">", "<", "=", "^", "~") if morceau.startswith(o)), "")
                v = morceau[len(op):].strip()
                corrige._lire(v)
                ecrites.append(v)
                if op in ("^", "~"):
                    bornes += [(">=", v), ("<", corrige._borne(op, v))]
                else:
                    bornes.append((op or "=", v))
            alternatives.append((bornes, ecrites))
        for bornes, ecrites in alternatives:
            tests = {">=": lambda a, b: a >= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
                     "<=": lambda a, b: a <= b, "=": lambda a, b: a == b}
            if all(tests[op](cle, corrige._cle(v)) for op, v in bornes):
                if not pre or any(corrige._lire(v)[3] and corrige._lire(v)[:3] == corrige._lire(version)[:3]
                                  for v in ecrites):
                    return True
        return False

    _reussi("satisfait", par_intervalles)


def test_satisfait_signale_un_comparer_faux():
    faux = _avec("satisfait", comparer_versions=_comparer_bogue("sans_pre"))
    _echec("satisfait", faux, "comparer_versions (défi 236) ne passe pas encore")


def test_satisfait_qui_ne_lit_pas_toute_la_contrainte():
    def paresseux(version, contrainte):
        if not isinstance(contrainte, str) or not isinstance(version, str):
            raise TypeError
        for alternative in contrainte.split("||"):
            if corrige.satisfait(version, alternative):
                return True
        return False

    _echec("satisfait", paresseux, "faute de frappe doit se voir tout de suite")


# --------------------------------------------------------------------------- Défi 239 : resoudre


def _resoudre_bogue(bug):
    def resoudre(index, demandes):
        return _ref_resoudre(copy.deepcopy(index), demandes, bug)
    return resoudre


@pytest.mark.parametrize("bug,morceau", [
    ("texte", "TEXTES"),
    ("ancienne", "ANCIENNE"),
    ("ordre_index", "Triez-les"),
    ("sans_retour", "arrière"),
    ("racine_seule", "AUSSI des contraintes"),
    ("sans_verif_choisis", "DÉJÀ CHOISIES"),
    ("pile", "BOUT de la file"),
    ("trie", "DÉCIDÉS"),
    ("keyerror", "KeyError"),
])
def test_resoudre_nomme_l_erreur(bug, morceau):
    _echec("resoudre", _resoudre_bogue(bug), morceau)


def test_resoudre_sans_recursion():
    """Une pile explicite d'itérateurs au lieu d'une fonction récursive : même ordre, même réponse."""
    def resoudre(index, demandes):
        def versions_possibles(paquet, contraintes, choisis):
            for version in sorted(index.get(paquet, {}), key=cmp_to_key(corrige.comparer_versions), reverse=True):
                if all(corrige.satisfait(version, c) for c in contraintes.get(paquet, [])):
                    avec = {**choisis, paquet: version}
                    if all(d not in avec or corrige.satisfait(avec[d], c) for d, c in index[paquet][version].items()):
                        yield version

        depart = ({}, {p: [c] for p, c in demandes.items()}, list(demandes))
        pile = [(depart, None)]
        while pile:
            (choisis, contraintes, file), iterateur = pile[-1]
            if not file:
                return choisis
            paquet = file[0]
            if iterateur is None:
                iterateur = versions_possibles(paquet, contraintes, choisis)
                pile[-1] = ((choisis, contraintes, file), iterateur)
            version = next(iterateur, None)
            if version is None:
                pile.pop()
                continue
            deps = index[paquet][version]
            avec = {**choisis, paquet: version}
            nouvelles = {**contraintes, **{d: contraintes.get(d, []) + [c] for d, c in deps.items()}}
            suite = file[1:] + [d for d in deps if d not in avec and d not in file[1:]]
            pile.append(((avec, nouvelles, suite), None))
        raise Conflit("aucune combinaison")

    _reussi("resoudre", resoudre)


def test_resoudre_sans_elaguer_est_trop_lent():
    """Choisir des versions complètes et ne vérifier les dépendances qu'à la fin : juste, mais exponentiel."""
    def resoudre(index, demandes):
        def valable(choisis):
            return all(corrige.satisfait(choisis[d], c) for p, v in choisis.items()
                       for d, c in index[p][v].items() if d in choisis)

        def decider(choisis, contraintes, file):
            if not file:
                return choisis if valable(choisis) else None
            paquet, reste = file[0], file[1:]
            for version in sorted(index.get(paquet, {}), key=cmp_to_key(corrige.comparer_versions), reverse=True):
                if all(corrige.satisfait(version, c) for c in contraintes[paquet]):
                    deps = index[paquet][version]
                    avec = {**choisis, paquet: version}
                    nouvelles = {d: contraintes.get(d, []) + [c] for d, c in deps.items()}
                    solution = decider(avec, {**contraintes, **nouvelles},
                                       reste + [d for d in deps if d not in avec and d not in reste])
                    if solution is not None:
                        return solution
            return None

        solution = decider({}, {p: [c] for p, c in demandes.items()}, list(demandes))
        if solution is None:
            raise Conflit("aucune")
        return solution

    _echec("resoudre", resoudre, "beaucoup trop lent", "12 puissance 60")


def test_resoudre_signale_un_satisfait_faux():
    faux = _avec("resoudre", satisfait=_satisfait_bogue("caret_sans_borne"))
    _echec("resoudre", faux, "satisfait (défi 238) ne passe pas encore")


def test_resoudre_qui_modifie_l_index():
    def resoudre(index, demandes):
        index.setdefault("cache", {})   # « juste un cache », dans l'index de l'appelant
        return corrige.resoudre(index, demandes)

    _echec("resoudre", resoudre, "modifié")


def _tous_les_choix(index, demandes):
    """Toutes les combinaisons valables, par recherche exhaustive (chaque paquet : absent, ou l'une de ses versions)."""
    paquets = list(index)
    for choix in itertools.product(*[[None, *index[p]] for p in paquets]):
        obtenu = {p: v for p, v in zip(paquets, choix) if v is not None}
        if all(p in index for p in demandes) and not _explication(index, demandes, obtenu):
            yield obtenu


def test_reference_resoudre_comme_une_recherche_exhaustive():
    rng = random.Random(5)
    vus = {"conflit": 0, "solution": 0}
    for _ in range(300):
        index, demandes = checks._index_alea(rng)
        if sum(len(v) for v in index.values()) > 13:
            continue
        valables = list(_tous_les_choix(index, demandes))
        try:
            solution = _ref_resoudre(index, demandes)
        except Conflit:
            assert valables == []
            vus["conflit"] += 1
            continue
        vus["solution"] += 1
        assert solution in valables
        assert solution == corrige.resoudre(index, demandes)
        assert list(solution.items()) == list(corrige.resoudre(index, demandes).items())
    assert vus["conflit"] > 10 and vus["solution"] > 30


def test_l_index_long_est_vraiment_long_pour_une_recherche_naive():
    index, demandes = _index_long()
    solution = corrige.resoudre(index, demandes)
    assert solution["base"] == "1.0.0" and all(solution[f"p{i:02d}"] == "1.0.0" for i in range(60))
    assert len(solution) == 61


# --------------------------------------------------------------------------- Défi 240 : affecter


def _affecter_bogue(bug):
    def affecter(experience, identifiant, poids):
        try:
            corrige.affecter(experience, identifiant, poids)
        except (TypeError, ValueError):
            raise
        return _ref_affecter(experience, identifiant, poids, bug)
    return affecter


@pytest.mark.parametrize("bug,morceau", [
    ("hash", "SALÉ"),
    ("hash_tuple", "SALÉ"),
    ("sans_experience", "MÊME seau"),
    ("sans_separateur", "« : »"),
    ("hex32", "HEXADÉCIMAUX"),
    ("petit_bout", "\"big\""),
    ("tout", "PREMIERS"),
    ("md5", "SHA-256"),
    ("modulo_total", "TOTAL des poids"),
    ("inferieur_egal", "strictement"),
    ("trie", "ordre du dict"),
])
def test_affecter_nomme_l_erreur(bug, morceau):
    _echec("affecter", _affecter_bogue(bug), morceau)


def test_affecter_ecrit_par_un_eleve():
    from fractions import Fraction

    def avec_hexdigest(experience, identifiant, poids):
        corrige.affecter(experience, identifiant, poids)   # les mêmes contrôles
        seau = int(hashlib.sha256(f"{experience}:{identifiant}".encode()).hexdigest()[:16], 16) % PALIERS
        total, cumul = sum(poids.values()), 0
        for variante, p in poids.items():
            cumul += p
            if Fraction(seau, PALIERS) < Fraction(cumul, total):
                return variante

    _reussi("affecter", avec_hexdigest)


def test_les_frontieres_sont_vraiment_atteintes():
    ids = checks._identifiants_pile("nouveau_score", {0, 2500, 5000, 7500})
    assert [_seau("nouveau_score", i) for i in ids] == [0, 2500, 5000, 7500]
    pile = ids[1]   # seau 2500 : 2500 × 100 == 10 000 × 25, donc PAS « on »
    assert corrige.affecter("nouveau_score", pile, {"on": 25, "off": 75}) == "off"
    assert _ref_affecter("nouveau_score", pile, {"on": 25, "off": 75}, "inferieur_egal") == "on"
    assert corrige.affecter("nouveau_score", ids[0], {"a": 0, "b": 5, "c": 0, "d": 5}) == "b"


_SCRIPT = """
import sys
sys.path.insert(0, {racine!r})
from defis.corrige_serie_39 import affecter
ids = [f"U{{i:04d}}" for i in range(1, 200)]
print([affecter("nouveau_score", i, {{"on": 25, "off": 75}}) for i in ids].count("on"),
      [hash("nouveau_score:" + i) % 10000 for i in ids[:5]])
"""


def test_affecter_est_le_meme_d_un_lancement_a_l_autre_et_pas_hash():
    sorties = []
    for graine in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": graine}
        sortie = subprocess.run([sys.executable, "-c", _SCRIPT.format(racine=str(ROOT))], capture_output=True,
                                text=True, env=env, check=True).stdout
        sorties.append(sortie.split(" ", 1))
    assert sorties[0][0] == sorties[1][0]           # sha256 : le même seau partout
    assert sorties[0][1] != sorties[1][1]           # hash() : un autre seau à chaque lancement


def test_affecter_est_proportionnel_independant_et_monotone():
    ids = [f"C{i:06d}" for i in range(40_000)]
    a = [corrige.affecter("a", i, {"on": 25, "off": 75}) == "on" for i in ids]
    b = [corrige.affecter("b", i, {"on": 25, "off": 75}) == "on" for i in ids]
    assert abs(sum(a) / len(ids) - 0.25) < 0.01
    assert abs(sum(x and y for x, y in zip(a, b)) / len(ids) - 0.0625) < 0.01   # indépendants
    dix = [corrige.affecter("a", i, {"on": 10, "off": 90}) == "on" for i in ids]
    assert all(y for x, y in zip(dix, a) if x)                                   # 10 % → 25 % n'enlève personne


# --------------------------------------------------------------------------- Défi 241 : evaluer


def _evaluer_bogue(bug):
    def evaluer(drapeaux, nom, utilisateur):
        corrige.evaluer(drapeaux, nom, utilisateur)   # les mêmes contrôles (et les mêmes erreurs)
        return _ref_evaluer(drapeaux, nom, utilisateur, bug)
    return evaluer


@pytest.mark.parametrize("bug,morceau", [
    ("cibles_avant_coupure", "bouton d'arrêt"),
    ("sans_prerequis", "prérequis"),
    ("absent_pas_dans", "ABSENT"),
    ("version_texte", "\"4.10.2\" < \"4.2.0\""),
    ("ou", "TOUTES"),
    ("derniere", "PREMIÈRE"),
    ("zero", "start=1"),
    ("coupe_off", "en dur"),
    ("cibles_apres_regles", "AVANT les règles"),
])
def test_evaluer_nomme_l_erreur(bug, morceau):
    _echec("evaluer", _evaluer_bogue(bug), morceau)


def test_evaluer_par_une_classe():
    class Evaluateur:
        def __init__(self, drapeaux, utilisateur):
            self.drapeaux, self.utilisateur, self.en_cours = drapeaux, utilisateur, []

        def condition(self, attribut, operateur, operande):
            valeur = self.utilisateur.get(attribut)
            if valeur is None:
                return False
            if operateur == "dans":
                return valeur in operande
            if operateur == "pas_dans":
                return valeur not in operande
            try:
                c = corrige.comparer_versions(valeur, operande)
            except ValueError:
                return False
            return (c >= 0) == (operateur == "version>=")

        def servir(self, nom, servie):
            return corrige.affecter(nom, self.utilisateur["id"], servie) if isinstance(servie, dict) else servie

        def __call__(self, nom):
            if nom in self.en_cours or nom not in self.drapeaux:
                raise ValueError(nom)
            d = self.drapeaux[nom]
            corrige._valider(d)
            if not d["actif"]:
                return Decision(d["coupe"], "coupé")
            self.en_cours.append(nom)
            try:
                for autre, attendue in d.get("prerequis", {}).items():
                    if self(autre).valeur != attendue:
                        return Decision(d["coupe"], f"prérequis {autre}")
            finally:
                self.en_cours.pop()
            for valeur, ids in d.get("cibles", {}).items():
                if self.utilisateur["id"] in ids:
                    return Decision(valeur, "ciblé")
            for n, regle in enumerate(d.get("regles", []), 1):
                if all(self.condition(*c) for c in regle["si"]):
                    return Decision(self.servir(nom, regle["alors"]), f"règle {n}")
            return Decision(self.servir(nom, d["defaut"]), "défaut")

    def evaluer(drapeaux, nom, utilisateur):
        if not isinstance(utilisateur.get("id"), str):
            raise TypeError("id")
        return Evaluateur(drapeaux, utilisateur)(nom)

    _reussi("evaluer", evaluer)


def test_evaluer_signale_un_affecter_ou_un_comparer_faux():
    faux = _avec("evaluer", affecter=_affecter_bogue("sans_experience"))
    _echec("evaluer", faux, "votre affecter ne passe pas encore")
    faux = _avec("evaluer", comparer_versions=_comparer_bogue("sans_pre"))
    _echec("evaluer", faux, "votre comparer_versions ne passe pas encore")


def test_evaluer_qui_plante_sur_une_version_illisible():
    faux = _avec("evaluer", comparer_versions=lambda a, b: corrige.comparer_versions(a, b))
    vrai = faux.__globals__["_condition"]

    def _condition(utilisateur, attribut, operateur, operande):
        if operateur.startswith("version"):
            c = corrige.comparer_versions(utilisateur[attribut], operande)   # « 4.10 » lève ValueError
            return c >= 0 if operateur == "version>=" else c < 0
        return vrai(utilisateur, attribut, operateur, operande)

    faux.__globals__["_condition"] = _condition
    _echec("evaluer", faux, "lève")


def test_evaluer_qui_boucle_sur_un_cycle():
    faux = _avec("evaluer")
    vrai = faux.__globals__["_evaluer"]

    def _evaluer(drapeaux, nom, utilisateur, identifiant, chemin):
        return vrai(drapeaux, nom, utilisateur, identifiant, ())   # le chemin oublié : plus de détection

    faux.__globals__["_evaluer"] = _evaluer
    _echec("evaluer", faux, "CYCLE")


# --------------------------------------------------------------------------- les outils communs et l'histoire


def test_les_outils_communs():
    tags = tags_exemple()
    assert tags == sorted(tags) and tags is not tags_exemple() and len(set(tags)) == len(tags) == 24
    assert all(corrige._lire(t) for t in tags)
    assert repr(generer_versions(30, graine=3)) == repr(generer_versions(30, graine=3))
    assert all(corrige._lire(v) for v in generer_versions(300, graine=1))
    conseillers = conseillers_exemple()
    assert len(conseillers) == 480 and [c["id"] for c in conseillers] == [f"U{i:04d}" for i in range(1, 481)]
    assert sum("role" not in c for c in conseillers) > 5
    assert {c["appli"] for c in conseillers} >= {None, "4.10", "4.10.2", "4.2.0-beta.3"}
    assert drapeaux_exemple() == drapeaux_exemple() and drapeaux_exemple() is not drapeaux_exemple()
    for versions in index_exemple().values():
        assert all(corrige._lire(v) for v in versions)


def test_l_histoire_tient():
    tags = tags_exemple()
    assert max(tags) == "2.9.4"                                                   # l'erreur du 3 novembre
    stables = [t for t in tags if corrige._lire(t)[3] == ()]
    assert max(stables, key=cmp_to_key(corrige.comparer_versions)) == "2.10.1"
    historique = historique_exemple()
    assert corrige.prochaine_version("2.10.1", historique) == "3.0.0"
    assert _ref_prochaine("2.10.1", historique, "pied") == "2.11.0"              # le numéro que Théo avait choisi
    assert not corrige.satisfait("3.0.0", "^2.9.0") and corrige.satisfait("2.11.0", "^2.9.0")
    demandes = {"churn-modele": "^3.0.0", "graphiques": "^1.2.0", "journal": "*"}
    with pytest.raises(Conflit):
        _ref_resoudre(index_exemple(), demandes, "sans_retour")                  # sans retour arrière : impossible
    assert corrige.resoudre(index_exemple(), demandes)["graphiques"] == "1.3.1"   # pas 1.4.0
    drapeaux, conseillers = drapeaux_exemple(), conseillers_exemple()
    decisions = [corrige.evaluer(drapeaux, "nouveau_score", c) for c in conseillers]
    servis = [c for c, d in zip(conseillers, decisions) if d.valeur == "on"]
    assert len(servis) == 52
    assert all(corrige.comparer_versions(c["appli"], "4.2.0") >= 0 for c in servis)   # jamais l'appli 4.1
    texte = [_ref_evaluer(drapeaux, "nouveau_score", c, "version_texte") for c in conseillers]
    assert sum(d.valeur == "on" for d in texte) < 40                         # « 4.10.2 » < « 4.2.0 » en texte
    assert all(corrige.evaluer(drapeaux, "export_csv", c) == Decision("off", "coupé") for c in conseillers)


# --------------------------------------------------------------------------- le correcteur lui-même


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(236, 242))
    assert any(s is SERIE for s in series())            # enregistrée dans defis/verif.py


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_39 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(livraison)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_39 import *` (dans les exemples) doit donner les fonctions du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 25 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    index, drapeaux, conseillers, messages = index_exemple(), drapeaux_exemple(), conseillers_exemple(), historique_exemple()
    copies = copy.deepcopy((index, drapeaux, conseillers, messages))
    corrige.prochaine_version("2.10.1", messages)
    corrige.resoudre(index, {"churn-modele": "^3.0.0", "graphiques": "^1.2.0", "journal": "*"})
    for c in conseillers:
        for nom in drapeaux:
            corrige.evaluer(drapeaux, nom, c)
    poids = {"on": 25, "off": 75}
    corrige.affecter("x", "U1", poids)
    assert (index, drapeaux, conseillers, messages) == copies and poids == {"on": 25, "off": 75}


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(ValueError):
        corrige.comparer_versions("1.2.3\n", "1.2.3")
    with pytest.raises(ValueError):
        corrige.satisfait("1.5.0", "^1.0.0 || =>2.0.0")
    with pytest.raises(TypeError):
        corrige.affecter("e", "U1", {"a": True})
    with pytest.raises(ValueError, match="cycle"):
        corrige.evaluer({"x": {"actif": True, "coupe": "off", "defaut": "on", "prerequis": {"x": "on"}}}, "x",
                        {"id": "U1"})
    with pytest.raises(TypeError):
        corrige.evaluer({"x": {"actif": "false", "coupe": "off", "defaut": "on"}}, "x", {"id": "U1"})
    with pytest.raises(Conflit):
        corrige.resoudre({"a": {"1.0.0": {}}}, {"a": "^2.0.0"})

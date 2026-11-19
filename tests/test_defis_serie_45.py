"""Série 45 (lire ce que disent les clients) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 272`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (une découpe par expression régulière `[^\\W_]+`,
  des propositions coupées avec `re.split`, un entraînement avec `defaultdict` et des différences de logarithmes,
  des probabilités avec `scipy.special.logsumexp` ou NumPy, des mots révélateurs triés en entier, un classifieur en
  ligne avec des `defaultdict`) ;
- les solutions trop lentes (recompter chaque mot du vocabulaire dans tous les documents, recalculer tout le modèle à
  chaque prédiction) sont arrêtées et expliquées ;
- les références du correcteur tiennent face à des vérifications indépendantes : le Bayes naïf multinomial de
  scikit-learn (`MultinomialNB`, ses log-probabilités et ses `predict_proba`), un calcul à la main en fractions, et un
  classifieur en ligne qui retombe exactement sur l'entraînement complet après des milliers d'apprentissages et
  d'oublis ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (la règle « résili » : 26 % de précision et 21 % de rappel ; le Bayes naïf : 77 % et 65 % avec les mots seuls,
  74 % et 77 % avec les négations ; « Je ne suis pas satisfait. » à 51 % sans les négations, 99,6 % avec).
"""

from __future__ import annotations

import doctest
import math
import random
import re
import sys
import types
import unicodedata
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import defis.corrige_serie_45 as corrige  # noqa: E402
import defis.verbatims as verbatims  # noqa: E402
from defis.checks_serie_45 import (  # noqa: E402
    _BUGS_CLASSIFIEUR,
    _BUGS_ENTRAINER,
    _BUGS_MOTS,
    _BUGS_NEGATIONS,
    _BUGS_PROBABILITES,
    _BUGS_REVELATEURS,
    SERIE,
    _cas_272,
    _cas_273,
    _cas_274,
    _cas_275,
    _cas_276,
    _jouer,
    _modeles_egaux,
    _ref_entrainer,
    _ref_mots,
    _ref_negations,
    _ref_probabilites,
    _ref_revelateurs,
    _ref_sorties,
    _RefClassifieur,
    _scenarios_277,
    _sorties_egales,
)
from defis.verbatims import (  # noqa: E402
    FINS_DE_PORTEE,
    LETTRES_LIEES,
    NEGATIONS,
    PONCTUATION,
    PREFIXE,
    Modele,
    bilan,
    commentaires_exemple,
    generer_commentaires,
    nouveaux_exemple,
    regle_actuelle,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

# Attention : ce module ne doit PAS définir de noms `mots`, `avec_negations`, `entrainer` ou `probabilites` au niveau
# du module : pendant les défis 273 et 277, le correcteur prête les siens au module des fonctions vérifiées (ici, ce
# fichier de tests).

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet, **autres):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}, **autres), DEFIS[nom])


def _echec(nom, objet, *morceaux):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message
    return resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


def _avec_les_gardes(nom, calcul):
    """Une version boguée qui garde les vérifications du corrigé (les erreurs attendues), pour isoler LE bug."""
    def fonction(*args, **kwargs):
        getattr(corrige, nom)(*args, **kwargs)        # lève les mêmes erreurs que le corrigé
        return calcul(*args, **kwargs)
    fonction.__name__ = nom
    return fonction


def _classe_boguee(bug):
    class ClassifieurEnLigne(_RefClassifieur):
        def __init__(self, alpha=1.0):
            super().__init__(alpha, bug)
    return ClassifieurEnLigne


# --------------------------------------------------------------------------- Défi 272 : mots


@pytest.mark.parametrize("bug,morceau", [
    ("dernier_mot", "DERNIER mot"),
    ("split", "`split()`"),
    ("w", "tiret bas"),
    ("az", "autres alphabets"),
    ("isalpha", "efface les chiffres"),
    ("ascii", "« uvre »"),
    ("nfd", "NFKD"),
    ("accents", "combining"),
    ("ligatures", "LETTRES_LIEES"),
    ("sans_casse", "casefold"),
    ("ensemble", "RÉPÉTITIONS"),
])
def test_mots_nomme_l_erreur(bug, morceau):
    _echec("mots", _avec_les_gardes("mots", lambda texte: _ref_mots(texte, bug)), morceau)


def test_mots_qui_rend_un_ensemble():
    _echec("mots", _avec_les_gardes("mots", lambda texte: set(_ref_mots(texte))), "ensemble", "liste")


def test_mots_sans_verification_du_type():
    def fonction(texte):
        return _ref_mots(str(texte)) if texte is not None else []

    _echec("mots", fonction, "TypeError")


def test_mots_accepte_une_expression_reguliere_sans_tiret_bas():
    def fonction(texte):
        corrige.mots(texte)
        decompose = unicodedata.normalize("NFKD", texte.casefold())
        sans = "".join(c for c in decompose if not unicodedata.combining(c))
        for lettre, remplacement in LETTRES_LIEES.items():
            sans = sans.replace(lettre, remplacement)
        return re.findall(r"[^\W_]+", sans)              # \w sans le tiret bas : exactement isalnum

    _reussi("mots", fonction)


def test_mots_accepte_lower_et_une_boucle_par_indices():
    def fonction(texte):
        corrige.mots(texte)
        t = unicodedata.normalize("NFKD", texte.lower())
        t = "".join(c for c in t if unicodedata.combining(c) == 0).translate(str.maketrans(LETTRES_LIEES))
        resultat, debut = [], None
        for i, c in enumerate(t + " "):
            if c.isalnum() and debut is None:
                debut = i
            elif not c.isalnum() and debut is not None:
                resultat.append(t[debut:i])
                debut = None
        return resultat

    _reussi("mots", fonction)


# --------------------------------------------------------------------------- Défi 273 : avec_negations


@pytest.mark.parametrize("bug,morceau", [
    ("fenetre_1", "mot SUIVANT"),
    ("bascule", "« ne … pas » est UNE négation"),
    ("tout_le_texte", "traverse la ponctuation"),
    ("point_seulement", "qu'aux points"),
    ("sans_fin_de_portee", "FINS_DE_PORTEE"),
    ("negations_marquees", "« non_pas »"),
    ("fin_marquee", "gardé tel quel"),
    ("plus", "« plus » n'ouvre pas"),
    ("sans_n", "« n' »"),
])
def test_negations_nomme_l_erreur(bug, morceau):
    _echec("avec_negations", _avec_les_gardes("avec_negations", lambda texte: _ref_negations(texte, bug)), morceau)


def test_negations_avec_une_autre_decoupe():
    def fonction(texte):
        return [m.lower() for m in re.findall(r"\w+", texte)]

    _echec("avec_negations", _avec_les_gardes("avec_negations", fonction), "au lieu de")


def test_negations_accepte_re_split():
    def fonction(texte):
        corrige.avec_negations(texte)
        resultat = []
        for proposition in re.split("[" + re.escape(PONCTUATION) + "]", texte):
            nie = False
            for mot in corrige.mots(proposition):
                if mot in NEGATIONS:
                    nie = True
                    resultat.append(mot)
                elif mot in FINS_DE_PORTEE:
                    nie = False
                    resultat.append(mot)
                else:
                    resultat.append(f"{PREFIXE}{mot}" if nie else mot)
        return resultat

    _reussi("avec_negations", fonction)


def test_negations_utilise_les_mots_pretes(monkeypatch):
    """Pendant le 273, le correcteur prête SA fonction `mots` : un 272 à faire ne fait pas échouer un 273 juste."""
    module = types.ModuleType("faux_module_273")
    exec(  # le fichier d'un·e apprenant·e : `mots` pas encore écrite, `avec_negations` s'en sert
        "from defis.verbatims import NEGATIONS, FINS_DE_PORTEE, PONCTUATION, PREFIXE\n"
        "def mots(texte):\n"
        "    raise NotImplementedError\n"
        "def avec_negations(texte):\n"
        "    if not isinstance(texte, str):\n"
        "        raise TypeError('un texte')\n"
        "    for c in PONCTUATION:\n"
        "        texte = texte.replace(c, '.')\n"
        "    resultat = []\n"
        "    for proposition in texte.split('.'):\n"
        "        nie = False\n"
        "        for mot in mots(proposition):\n"
        "            if mot in NEGATIONS or mot in FINS_DE_PORTEE:\n"
        "                nie = mot in NEGATIONS\n"
        "                resultat.append(mot)\n"
        "            else:\n"
        "                resultat.append(PREFIXE + mot if nie else mot)\n"
        "    return resultat\n", module.__dict__)
    monkeypatch.setitem(sys.modules, "faux_module_273", module)
    _reussi("avec_negations", module.avec_negations)
    with pytest.raises(NotImplementedError):            # la vôtre est remise en place
        module.mots("")


# --------------------------------------------------------------------------- Défi 274 : entrainer


@pytest.mark.parametrize("bug,morceau", [
    ("probas", "logarithmes"),
    ("log10", "`math.log10`"),
    ("a_priori_uniforme", "DOCUMENTS de cette classe"),
    ("a_priori_mots", "un long commentaire ne compte pas plus"),
    ("presence", "OCCURRENCES"),
    ("distincts", "DISTINCTS"),
    ("v_classe", "TOUT le vocabulaire"),
    ("alpha_sans_v", "alpha × taille du vocabulaire au dénominateur"),
    ("v_sans_alpha", "c'est juste quand alpha vaut 1"),
    ("classes_ordre", "`sorted`"),
])
def test_entrainer_nomme_l_erreur(bug, morceau):
    calcul = _avec_les_gardes("entrainer", lambda d, e, alpha=1.0: _ref_entrainer(d, e, alpha, bug))
    _echec("entrainer", calcul, morceau)


def test_entrainer_sans_lissage():
    def fonction(documents, etiquettes, alpha=1.0):
        return _ref_entrainer(documents, etiquettes, 0.0)       # alpha ignoré : log(0) pour un mot absent

    _echec("entrainer", _avec_les_gardes("entrainer", fonction), "ValueError", "lissage")


def test_entrainer_avec_le_vocabulaire_de_chaque_classe():
    def fonction(documents, etiquettes, alpha=1.0):
        modele = _ref_entrainer(documents, etiquettes, alpha)
        comptes = defaultdict(set)
        for document, etiquette in zip(documents, etiquettes):
            comptes[etiquette].update(document)
        return Modele(modele.classes, modele.log_a_priori,
                      {c: {m: v for m, v in d.items() if m in comptes[c]} for c, d in modele.log_proba.items()})

    _echec("entrainer", _avec_les_gardes("entrainer", fonction), "il manque", "TOUS les mots du vocabulaire")


def test_entrainer_qui_prend_le_logarithme_d_un_denominateur_nul():
    def fonction(documents, etiquettes, alpha=1.0):
        modele = _ref_entrainer(documents, etiquettes, alpha)
        for c in modele.classes:
            total = sum(len(d) for d, e in zip(documents, etiquettes) if e == c)
            math.log(total + alpha * len(modele.log_proba[c]))          # calculé même sans aucun mot
        return modele

    _echec("entrainer", _avec_les_gardes("entrainer", fonction), "le vocabulaire est VIDE")


def test_entrainer_qui_rend_un_dict():
    def fonction(documents, etiquettes, alpha=1.0):
        modele = _ref_entrainer(documents, etiquettes, alpha)
        return {"classes": modele.classes, "log_a_priori": modele.log_a_priori, "log_proba": modele.log_proba}

    _echec("entrainer", _avec_les_gardes("entrainer", fonction), "rendez un `Modele(")


def test_entrainer_qui_modifie_ses_arguments():
    def fonction(documents, etiquettes, alpha=1.0):
        modele = _ref_entrainer(documents, etiquettes, alpha)
        if isinstance(documents, list):
            documents.sort(key=len)
        return modele

    _echec("entrainer", _avec_les_gardes("entrainer", fonction), "MODIFIÉ")


def test_entrainer_qui_recompte_chaque_mot_est_arrete():
    def fonction(documents, etiquettes, alpha=1.0):
        corrige.entrainer(documents, etiquettes, alpha)
        classes = sorted(set(etiquettes))
        vocabulaire = sorted({m for d in documents for m in d})
        n = len(documents)
        a_priori, log_proba = {}, {}
        for c in classes:
            docs = [d for d, e in zip(documents, etiquettes) if e == c]
            a_priori[c] = math.log(len(docs) / n)
            total = sum(len(d) for d in docs)
            log_proba[c] = {}
            for mot in vocabulaire:                     # chaque mot : tous les documents, encore et encore
                compte = 0
                for d in docs:
                    compte += d.count(mot)
                log_proba[c][mot] = math.log((compte + alpha) / (total + alpha * len(vocabulaire)))
        return Modele(classes, a_priori, log_proba)

    _echec("entrainer", fonction, "beaucoup trop lent", "Counter")


def test_entrainer_accepte_defaultdict_et_des_differences_de_logarithmes():
    def fonction(documents, etiquettes, alpha=1.0):
        corrige.entrainer(documents, etiquettes, alpha)
        par_classe = defaultdict(lambda: defaultdict(int))
        documents_par_classe = defaultdict(int)
        vocabulaire = set()
        for document, etiquette in zip(documents, etiquettes):
            documents_par_classe[etiquette] += 1
            for mot in document:
                par_classe[etiquette][mot] += 1
                vocabulaire.add(mot)
        classes = sorted(documents_par_classe)
        log_proba = {c: {} for c in classes}
        for c in classes if vocabulaire else []:            # sans aucun mot, rien à calculer
            log_denominateur = math.log(sum(par_classe[c].values()) + alpha * len(vocabulaire))
            log_proba[c] = {m: math.log(par_classe[c][m] + alpha) - log_denominateur for m in vocabulaire}
        log_n = math.log(len(documents))
        return Modele(classes, {c: math.log(documents_par_classe[c]) - log_n for c in classes}, log_proba)

    _reussi("entrainer", fonction)


# --------------------------------------------------------------------------- Défi 275 : probabilites


@pytest.mark.parametrize("bug,morceau", [
    ("sans_a_priori", "probabilité a priori"),
    ("ensemble", "`set(mots)`"),
    ("moyenne", "MOYENNE"),
    ("scores", "SCORES"),
    ("non_normalise", "SOMME"),
])
def test_probabilites_nomme_l_erreur(bug, morceau):
    _echec("probabilites", _avec_les_gardes("probabilites", lambda m, x: _ref_probabilites(m, x, bug)), morceau)


def _scores(modele, mots):
    vocabulaire = modele.log_proba[modele.classes[0]]
    return {c: modele.log_a_priori[c] + sum(modele.log_proba[c][m] for m in mots if m in vocabulaire)
            for c in modele.classes}


def test_probabilites_sans_retrancher_le_maximum():
    def fonction(modele, mots):
        exps = {c: math.exp(s) for c, s in _scores(modele, mots).items()}
        total = sum(exps.values())
        return {c: e / total for c, e in exps.items()}

    _echec("probabilites", _avec_les_gardes("probabilites", fonction), "ZeroDivisionError", "PLUS GRAND des scores")


def test_probabilites_par_un_produit():
    def fonction(modele, mots):
        vocabulaire = modele.log_proba[modele.classes[0]]
        produits = {}
        for c in modele.classes:
            p = math.exp(modele.log_a_priori[c])
            for m in mots:
                if m in vocabulaire:
                    p *= math.exp(modele.log_proba[c][m])
            produits[c] = p
        total = sum(produits.values())
        return {c: p / total for c, p in produits.items()}

    _echec("probabilites", _avec_les_gardes("probabilites", fonction), "ZeroDivisionError", "LOGARITHMES")


def test_probabilites_par_un_produit_qui_se_rabat_sur_l_a_priori():
    def fonction(modele, mots):
        vocabulaire = modele.log_proba[modele.classes[0]]
        produits = {c: math.exp(modele.log_a_priori[c]) for c in modele.classes}
        for m in mots:
            if m in vocabulaire:
                for c in modele.classes:
                    produits[c] *= math.exp(modele.log_proba[c][m])
        total = sum(produits.values())
        if total == 0:                                       # « au pire, l'a priori »
            return {c: math.exp(modele.log_a_priori[c]) for c in modele.classes}
        return {c: p / total for c, p in produits.items()}

    _echec("probabilites", _avec_les_gardes("probabilites", fonction), "Sur un long commentaire", "LOGARITHMES")


def test_probabilites_en_retranchant_le_minimum():
    def fonction(modele, mots):
        scores = _scores(modele, mots)
        plus_petit = min(scores.values())
        exps = {c: math.exp(s - plus_petit) for c, s in scores.items()}
        total = sum(exps.values())
        return {c: e / total for c, e in exps.items()}

    _echec("probabilites", _avec_les_gardes("probabilites", fonction), "OverflowError", "pas le plus petit")


def test_probabilites_qui_plante_sur_un_mot_inconnu():
    def fonction(modele, mots):
        scores = {c: modele.log_a_priori[c] + sum(modele.log_proba[c][m] for m in mots) for c in modele.classes}
        plus_grand = max(scores.values())
        exps = {c: math.exp(s - plus_grand) for c, s in scores.items()}
        return {c: e / sum(exps.values()) for c, e in exps.items()}

    _echec("probabilites", _avec_les_gardes("probabilites", fonction), "KeyError", "ignorez-le")


def test_probabilites_dans_le_desordre():
    def fonction(modele, mots):
        p = _ref_probabilites(modele, mots)
        return dict(reversed(list(p.items())))

    _echec("probabilites", _avec_les_gardes("probabilites", fonction), "l'ordre de modele.classes")


def test_probabilites_qui_lit_le_texte_caractere_par_caractere():
    def fonction(modele, mots):
        return _ref_probabilites(modele, list(mots))

    _echec("probabilites", fonction, "probabilites(modele, 'prix')", "TypeError")


def test_probabilites_accepte_logsumexp_de_scipy():
    scipy_special = pytest.importorskip("scipy.special")

    def fonction(modele, mots):
        corrige.probabilites(modele, mots)                  # les mêmes erreurs
        scores = _scores(modele, mots)
        log_total = scipy_special.logsumexp(list(scores.values()))
        return {c: math.exp(s - log_total) for c, s in scores.items()}

    _reussi("probabilites", fonction)


def test_probabilites_accepte_numpy():
    np = pytest.importorskip("numpy")

    def fonction(modele, mots):
        corrige.probabilites(modele, mots)
        index = {m: i for i, m in enumerate(modele.log_proba[modele.classes[0]])}
        comptes = np.zeros(len(index))
        for m in mots:
            if m in index:
                comptes[index[m]] += 1
        table = np.array([[modele.log_proba[c][m] for m in index] for c in modele.classes]).reshape(len(modele.classes),
                                                                                                    len(index))
        scores = np.array([modele.log_a_priori[c] for c in modele.classes]) + table @ comptes
        exps = np.exp(scores - scores.max())
        return dict(zip(modele.classes, (exps / exps.sum()).tolist()))

    _reussi("probabilites", fonction)


# --------------------------------------------------------------------------- Défi 276 : mots_revelateurs


@pytest.mark.parametrize("bug,morceau", [
    ("frequence", "fréquents partout"),
    ("moyenne", "MOYENNE des autres"),
    ("a_l_envers", "CROISSANT"),
    ("egalites_inverse", "INVERSE"),
    ("egalites_ordre", "ne doit pas dépendre"),
    ("sans_filtre", "strictement positifs"),
    ("rapport", "RAPPORT"),
])
def test_revelateurs_nomme_l_erreur(bug, morceau):
    calcul = _avec_les_gardes("mots_revelateurs", lambda m, c, k=10: _ref_revelateurs(m, c, k, bug))
    _echec("mots_revelateurs", calcul, morceau)


def test_revelateurs_par_frequence_dans_l_ordre_du_dict():
    def fonction(modele, classe, k=10):
        return sorted(modele.log_proba[classe].items(), key=lambda paire: -paire[1])[:k]

    _echec("mots_revelateurs", _avec_les_gardes("mots_revelateurs", fonction), "fréquents partout")


def test_revelateurs_qui_rend_des_listes():
    def fonction(modele, classe, k=10):
        return [list(paire) for paire in _ref_revelateurs(modele, classe, k)]

    _echec("mots_revelateurs", _avec_les_gardes("mots_revelateurs", fonction), "tuple (mot, écart)")


def test_revelateurs_sans_les_verifications():
    def fonction(modele, classe, k=10):
        return _ref_revelateurs(modele, classe, k) if classe in modele.classes and len(modele.classes) > 1 else []

    _echec("mots_revelateurs", fonction, "ValueError")


def test_revelateurs_accepte_un_tri_complet():
    def fonction(modele, classe, k=10):
        corrige.mots_revelateurs(modele, classe, k)
        ecarts = []
        for mot in sorted(modele.log_proba[classe]):
            meilleur_ailleurs = max(modele.log_proba[c][mot] for c in modele.classes if c != classe)
            if modele.log_proba[classe][mot] > meilleur_ailleurs:
                ecarts.append((mot, modele.log_proba[classe][mot] - meilleur_ailleurs))
        ecarts.sort(key=lambda paire: paire[1], reverse=True)       # stable : l'ordre alphabétique reste
        return ecarts[:k]

    _reussi("mots_revelateurs", fonction)


def test_la_reference_des_revelateurs_se_calcule_a_la_main():
    modele = _cas_276()[0][0]                       # l'exemple de la docstring
    # P(cher | part) = 2/14, P(cher | reste) = 1/14 ; P(prix | part) = 3/14, P(prix | reste) = 2/14
    attendu = [("cher", math.log(2)), ("rio", math.log(2)), ("prix", math.log(Fraction(3, 2)))]
    obtenu = _ref_revelateurs(modele, "part")
    assert [m for m, _ in obtenu] == [m for m, _ in attendu]
    assert all(math.isclose(e, a, rel_tol=1e-12) for (_, e), (_, a) in zip(obtenu, attendu))
    assert _ref_revelateurs(modele, "reste", 1)[0][0] == "merci"


# --------------------------------------------------------------------------- Défi 277 : ClassifieurEnLigne


@pytest.mark.parametrize("bug,morceau", [
    ("vocabulaire_garde", "del self._vocabulaire[mot]"),
    ("classe_gardee", "disparaît"),
    ("non_atomique", "Vérifiez TOUS les mots d'abord"),
    ("sans_verification", "ce qui n'a jamais été appris"),
    ("totaux_garde", "N(c)"),
    ("cache", "cache"),
    ("len_mots", "DOCUMENTS"),
])
def test_classifieur_nomme_l_erreur(bug, morceau):
    _echec("ClassifieurEnLigne", _classe_boguee(bug), morceau)


def test_classifieur_dont_les_comptes_sont_partages():
    class ClassifieurEnLigne(corrige.ClassifieurEnLigne):
        _documents_communs = Counter()

        def __init__(self, alpha=1.0):
            super().__init__(alpha)
            self._documents = self._documents_communs

    _echec("ClassifieurEnLigne", ClassifieurEnLigne, "partagent")


def test_classifieur_dont_le_modele_est_une_vue():
    class ClassifieurEnLigne(corrige.ClassifieurEnLigne):
        def modele(self):
            photo = super().modele()
            self._vues = getattr(self, "_vues", []) + [photo]
            return photo

        def apprendre(self, mots, etiquette):
            super().apprendre(mots, etiquette)
            for vue in getattr(self, "_vues", []):          # « mettre à jour » les modèles déjà rendus
                vue.log_proba.clear()

    _echec("ClassifieurEnLigne", ClassifieurEnLigne, "photographie")


def test_classifieur_qui_recalcule_tout_a_chaque_prediction_est_arrete():
    class ClassifieurEnLigne(corrige.ClassifieurEnLigne):
        def probabilites(self, mots):
            corrige.ClassifieurEnLigne.probabilites(self, mots)       # les mêmes erreurs
            v = len(self._vocabulaire)
            n = sum(self._documents.values())
            table = {}
            for c in sorted(self._documents):                     # toutes les log-probabilités, à chaque appel
                table[c] = {}
                for mot in self._vocabulaire:
                    table[c][mot] = math.log((self._comptes[c][mot] + self._alpha) / (self._totaux[c] + self._alpha * v))
            scores = {c: math.log(self._documents[c] / n) + sum(table[c][m] for m in mots if m in table[c])
                      for c in table}
            plus_grand = max(scores.values())
            exps = {c: math.exp(s - plus_grand) for c, s in scores.items()}
            return {c: e / sum(exps.values()) for c, e in exps.items()}

    _echec("ClassifieurEnLigne", ClassifieurEnLigne, "beaucoup trop lent", "comptes à jour")


def test_classifieur_qui_ne_verifie_pas_avant_de_modifier():
    class ClassifieurEnLigne(corrige.ClassifieurEnLigne):
        def apprendre(self, mots, etiquette):
            if isinstance(mots, str) and isinstance(etiquette, str):
                self._documents[etiquette] += 1         # le texte au lieu des mots : compté, PUIS refusé
            corrige.ClassifieurEnLigne.apprendre(self, mots, etiquette)

    _echec("ClassifieurEnLigne", ClassifieurEnLigne, "une opération refusée ne change rien")


def test_classifieur_accepte_des_defaultdict():
    class ClassifieurEnLigne:
        def __init__(self, alpha=1.0):
            corrige.ClassifieurEnLigne(alpha)
            self.alpha = alpha
            self.docs = defaultdict(int)
            self.mots_par_classe = defaultdict(lambda: defaultdict(int))
            self.total = defaultdict(int)
            self.vocab = defaultdict(int)

        def apprendre(self, mots, etiquette):
            corrige.ClassifieurEnLigne().apprendre(mots, etiquette)
            self.docs[etiquette] += 1
            for m in mots:
                self.mots_par_classe[etiquette][m] += 1
                self.vocab[m] += 1
            self.total[etiquette] += len(mots)

        def oublier(self, mots, etiquette):
            corrige.ClassifieurEnLigne().apprendre(mots, etiquette)        # mêmes TypeError
            besoin = Counter(mots)
            if self.docs.get(etiquette, 0) == 0 or any(self.mots_par_classe[etiquette].get(m, 0) < n
                                                        for m, n in besoin.items()):
                raise ValueError("pas appris")
            for m, n in besoin.items():
                self.mots_par_classe[etiquette][m] -= n
                self.vocab[m] -= n
                if self.vocab[m] == 0:
                    del self.vocab[m]
            self.total[etiquette] -= len(mots)
            self.docs[etiquette] -= 1
            if self.docs[etiquette] == 0:
                del self.docs[etiquette]
                self.mots_par_classe.pop(etiquette, None)       # une classe de documents vides n'a aucun mot
                self.total.pop(etiquette, None)

        def probabilites(self, mots):
            if isinstance(mots, str) or any(not isinstance(m, str) for m in mots):
                raise TypeError("une liste de mots")
            if not self.docs:
                raise ValueError("vide")
            v, n = len(self.vocab), sum(self.docs.values())
            scores = {}
            for c in sorted(self.docs):
                scores[c] = math.log(self.docs[c]) - math.log(n)
                for m in mots:
                    if m in self.vocab:
                        scores[c] += math.log(self.mots_par_classe[c].get(m, 0) + self.alpha)
                        scores[c] -= math.log(self.total[c] + self.alpha * v)
            plus_grand = max(scores.values())
            exps = {c: math.exp(s - plus_grand) for c, s in scores.items()}
            total = sum(exps.values())
            return {c: e / total for c, e in exps.items()}

        def modele(self):
            if not self.docs:
                raise ValueError("vide")
            documents = []
            etiquettes = []
            for c in self.docs:                     # un document « résumé » par classe ne suffit pas : on recalcule
                documents.append([m for m, n in self.mots_par_classe[c].items() for _ in range(n)])
                etiquettes.append(c)
            modele = corrige.entrainer(documents, etiquettes, self.alpha)
            n = sum(self.docs.values())
            return Modele(modele.classes, {c: math.log(self.docs[c] / n) for c in modele.classes}, modele.log_proba)

        def __len__(self):
            return sum(self.docs.values())

    _reussi("ClassifieurEnLigne", ClassifieurEnLigne)


def test_la_reference_en_ligne_retombe_sur_l_entrainement_complet():
    """Après des milliers d'apprentissages et d'oublis, le modèle en ligne est celui de l'entraînement complet."""
    alea = random.Random(4577)
    clf = _RefClassifieur(0.7)
    restants = []
    for texte, etiquette in generer_commentaires(3000, graine=4578):
        document = _ref_negations(texte)
        clf.apprendre(document, etiquette)
        restants.append((document, etiquette))
        if alea.random() < 0.3:
            document, etiquette = restants.pop(alea.randrange(len(restants)))
            clf.oublier(list(reversed(document)), etiquette)
    attendu = _ref_entrainer([d for d, _ in restants], [e for _, e in restants], 0.7)
    assert _modeles_egaux(clf.modele(), attendu) and len(clf) == len(restants)
    for texte, _ in nouveaux_exemple()[:200]:
        document = _ref_negations(texte)
        assert _sorties_egales(clf.probabilites(document), _ref_probabilites(attendu, document))
    assert _modeles_egaux(corrige.entrainer([d for d, _ in restants], [e for _, e in restants], 0.7), attendu)


# --------------------------------------------------------------------------- les références face à scikit-learn


def test_la_reference_est_le_bayes_naif_de_scikit_learn():
    np = pytest.importorskip("numpy")
    naive_bayes = pytest.importorskip("sklearn.naive_bayes")
    texte_sklearn = pytest.importorskip("sklearn.feature_extraction.text")

    anciens = commentaires_exemple()
    documents = [_ref_negations(t) for t, _ in anciens]
    etiquettes = [e for _, e in anciens]
    for alpha in (1.0, 0.3):
        modele = _ref_entrainer(documents, etiquettes, alpha)
        vectoriseur = texte_sklearn.CountVectorizer(analyzer=lambda document: document)
        x = vectoriseur.fit_transform(documents)
        nb = naive_bayes.MultinomialNB(alpha=alpha).fit(x, etiquettes)
        assert list(nb.classes_) == modele.classes
        for i, c in enumerate(nb.classes_):
            assert math.isclose(nb.class_log_prior_[i], modele.log_a_priori[c], rel_tol=1e-12)
            for mot, j in vectoriseur.vocabulary_.items():
                assert math.isclose(nb.feature_log_prob_[i, j], modele.log_proba[c][mot], rel_tol=1e-9)
        nouveaux = [_ref_negations(t) for t, _ in nouveaux_exemple()] + [["rio"] * 50, [], ["inconnu"]]
        probas = nb.predict_proba(vectoriseur.transform(nouveaux))
        for document, ligne in zip(nouveaux, probas):
            p = _ref_probabilites(modele, document)
            assert np.allclose([p[c] for c in nb.classes_], ligne, atol=1e-9)


def test_la_reference_des_probabilites_se_calcule_a_la_main():
    modele = _cas_275()[0][0]                   # l'exemple de la docstring
    part = Fraction(1, 3) * Fraction(1, 2) * Fraction(1, 3)
    reste = Fraction(2, 3) * Fraction(1, 3) * Fraction(1, 6)
    p = _ref_probabilites(modele, ["prix", "cher"])
    assert math.isclose(p["part"], float(part / (part + reste))) and math.isclose(p["part"], 0.6)
    assert math.isclose(_ref_entrainer([["a", "a", "b"], ["b"], ["c"]], ["x", "x", "y"]).log_proba["x"]["a"],
                        math.log(3 / 7))


# --------------------------------------------------------------------------- outils, histoire, corrigé


def test_les_outils_communs():
    anciens, nouveaux = commentaires_exemple(), nouveaux_exemple()
    assert len(anciens) == 4000 and len(nouveaux) == 1000
    assert commentaires_exemple() is not anciens and commentaires_exemple() == anciens
    assert {e for _, e in anciens + nouveaux} == {"part", "reste"}
    textes = [t for t, _ in anciens]
    assert any("ﬁ" in t for t in textes) and any("ᵉ" in t for t in textes) and any("’" in t for t in textes)
    assert any(t.isupper() for t in textes) and any("resilier" in t for t in textes)       # sans accent
    assert generer_commentaires(50, graine=3) == generer_commentaires(50, graine=3)
    assert regle_actuelle("RÉSILIATION") == "part" and regle_actuelle("Je pars") == "reste"
    assert bilan(["part"], ["reste"]) == {"signales": 1, "trouves": 0, "precision": 0.0, "rappel": None}
    assert {"ne", "n", "pas"} <= NEGATIONS and "plus" not in NEGATIONS and "mais" in FINS_DE_PORTEE
    assert "," in PONCTUATION and "\n" in PONCTUATION and "'" not in PONCTUATION and PREFIXE == "non_"
    assert verbatims.Modele is Modele


def test_l_histoire_tient():
    anciens, nouveaux = commentaires_exemple(), nouveaux_exemple()
    vraies = [e for _, e in nouveaux]
    assert round(sum(e == "part" for _, e in anciens) / len(anciens), 2) == 0.2
    regle = bilan([regle_actuelle(t) for t, _ in nouveaux], vraies)
    assert regle == {"signales": 180, "trouves": 46, "precision": 0.256, "rappel": 0.207}
    assert sum(v == "part" for v in vraies) - regle["trouves"] == 176          # « 176 sont partis sans être vus »

    def classe(modele, decouper, texte):
        p = corrige.probabilites(modele, decouper(texte))
        return max(p, key=p.get)

    sans = corrige.entrainer([corrige.mots(t) for t, _ in anciens], [e for _, e in anciens])
    avec = corrige.entrainer([corrige.avec_negations(t) for t, _ in anciens], [e for _, e in anciens])
    bilan_sans = bilan([classe(sans, corrige.mots, t) for t, _ in nouveaux], vraies)
    bilan_avec = bilan([classe(avec, corrige.avec_negations, t) for t, _ in nouveaux], vraies)
    assert bilan_sans == {"signales": 189, "trouves": 145, "precision": 0.767, "rappel": 0.653}
    assert bilan_avec == {"signales": 231, "trouves": 171, "precision": 0.74, "rappel": 0.77}
    assert bilan_avec["rappel"] > 0.75 > bilan_sans["rappel"] > 0.6 and bilan_sans["rappel"] > 3 * regle["rappel"]

    texte = "Je ne suis pas satisfait."
    assert round(corrige.probabilites(sans, corrige.mots(texte))["part"], 2) == 0.51
    assert round(corrige.probabilites(avec, corrige.avec_negations(texte))["part"], 3) == 0.996
    texte = "Pas question de résilier, je suis bien ici."
    assert regle_actuelle(texte) == "part" and corrige.probabilites(avec, corrige.avec_negations(texte))["part"] < 0.01
    revelateurs = [m for m, _ in corrige.mots_revelateurs(avec, "part", 8)]
    assert revelateurs[:3] == ["code", "demande", "rio"] and "non_content" in revelateurs

    clf = corrige.ClassifieurEnLigne()                      # apprendre chaque soir = tout réentraîner
    for t, e in anciens:
        clf.apprendre(corrige.avec_negations(t), e)
    assert _modeles_egaux(clf.modele(), avec)
    clf.oublier(corrige.avec_negations(anciens[36][0]), anciens[36][1])        # le droit à l'effacement
    sans_lui = anciens[:36] + anciens[37:]
    assert _modeles_egaux(clf.modele(), corrige.entrainer([corrige.avec_negations(t) for t, _ in sans_lui],
                                                          [e for _, e in sans_lui]))


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    serie = next(s for s in series() if s.code == "45")
    assert [r.statut for r in evaluer(serie)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(serie, corrige=True)] == [REUSSI] * 6


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_45 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(verbatims)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_45 import *` (dans les exemples) doit donner les objets du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 45 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    documents = [["prix", "cher"], ("merci",)]
    etiquettes = ["part", "reste"]
    copies = repr((documents, etiquettes))
    modele = corrige.entrainer(documents, etiquettes)
    copie_modele = repr(modele)
    mots_ = ["prix", "inconnu"]
    corrige.probabilites(modele, mots_)
    corrige.mots_revelateurs(modele, "part")
    clf = corrige.ClassifieurEnLigne()
    clf.apprendre(mots_, "part")
    clf.oublier(mots_, "part")
    assert repr((documents, etiquettes)) == copies and repr(modele) == copie_modele and mots_ == ["prix", "inconnu"]


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(ValueError):                    # la raison d'être du lissage
        math.log(0)
    assert 0.5 ** 1100 == 0.0                          # la raison d'être des logarithmes
    with pytest.raises(TypeError):
        corrige.mots(None)
    with pytest.raises(TypeError):
        corrige.avec_negations(["je"])
    with pytest.raises(TypeError):
        corrige.entrainer(["je résilie"], ["part"])
    with pytest.raises(ValueError):
        corrige.entrainer([["a"]], ["part"], alpha=0)
    with pytest.raises(TypeError):
        corrige.probabilites(corrige.entrainer([["a"]], ["part"]), "a")
    with pytest.raises(ValueError):
        corrige.mots_revelateurs(corrige.entrainer([["a"]], ["part"]), "part")
    clf = corrige.ClassifieurEnLigne()
    with pytest.raises(ValueError):
        clf.probabilites([])
    with pytest.raises(ValueError):
        clf.oublier(["a"], "part")


def test_les_bugs_annonces_existent_tous():
    """Chaque erreur classique décrite par le correcteur a un test ci-dessus (et inversement)."""
    assert set(_BUGS_MOTS) == {"dernier_mot", "split", "w", "az", "isalpha", "ascii", "nfd", "accents", "ligatures",
                               "sans_casse", "ensemble"}
    assert set(_BUGS_NEGATIONS) == {"fenetre_1", "bascule", "tout_le_texte", "point_seulement", "sans_fin_de_portee",
                                    "negations_marquees", "fin_marquee", "plus", "sans_n"}
    assert set(_BUGS_ENTRAINER) == {"probas", "log10", "a_priori_uniforme", "a_priori_mots", "presence", "distincts",
                                    "v_classe", "alpha_sans_v", "v_sans_alpha", "classes_ordre"}
    assert set(_BUGS_PROBABILITES) == {"sans_a_priori", "ensemble", "moyenne", "scores", "non_normalise"}
    assert set(_BUGS_REVELATEURS) == {"frequence", "moyenne", "a_l_envers", "egalites_inverse", "egalites_ordre",
                                      "sans_filtre", "rapport"}
    assert set(_BUGS_CLASSIFIEUR) == {"vocabulaire_garde", "classe_gardee", "non_atomique", "sans_verification",
                                      "totaux_garde", "cache", "len_mots"}


def test_chaque_bug_change_au_moins_un_cas():
    petits_272 = _cas_272()[:23]
    for bug in _BUGS_MOTS:
        assert any(_ref_mots(t, bug) != _ref_mots(t) for t in petits_272), bug
    petits_273 = _cas_273()[:23]
    for bug in _BUGS_NEGATIONS:
        assert any(_ref_negations(t, bug) != _ref_negations(t) for t in petits_273), bug
    for bug in _BUGS_ENTRAINER:
        assert any(not _modeles_egaux(_ref_entrainer(d, e, bug=bug, **k), _ref_entrainer(d, e, **k))
                   for d, e, k in _cas_274()[:-2]), bug
    for bug in _BUGS_PROBABILITES:
        assert any(not _sorties_egales(_ref_probabilites(m, x, bug), _ref_probabilites(m, x))
                   for m, x in _cas_275()[:10]), bug
    for bug in _BUGS_REVELATEURS:
        assert any(_ref_revelateurs(m, c, k, bug) != _ref_revelateurs(m, c, k) for m, c, k in _cas_276()), bug
    for bug in _BUGS_CLASSIFIEUR:
        assert any(not all(_sorties_egales(o, a) for o, a in zip(_ref_sorties(x, ops, bug), _ref_sorties(x, ops)))
                   for x, ops in _scenarios_277()[:6]), bug


def test_les_scenarios_en_ligne_sont_valables():
    """Les scénarios tirés au hasard du défi 277 n'oublient que ce qui a été appris, sauf les oublis faits pour être
    refusés (un mot jamais appris, une étiquette inconnue) : ceux-là sont toujours refusés."""
    for alpha, operations in _scenarios_277()[6:]:
        sorties = _jouer(_RefClassifieur(alpha), operations)
        oublis = [(op, s) for op, s in zip(operations, sorties) if op[0] == "oublier"]
        assert len(oublis) > 50
        for op, sortie in oublis:
            refuse = isinstance(sortie, tuple) and sortie[:2] == ("!", "ValueError")
            assert refuse == ("jamais_vu" in op[1] or op[2] == "inconnue"), op

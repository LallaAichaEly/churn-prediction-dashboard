"""Série 45 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_45.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours un mot coupé
en deux par un accent (défi 272), une négation qui en annule une autre (défi 273), un mot jamais vu dans une classe
qui fait `math.log(0)` (défi 274), un produit de probabilités qui vaut 0.0 sur un long commentaire (défi 275), un mot
« révélateur » qui n'est qu'un mot fréquent (défi 276), ou un mot oublié qui reste dans le vocabulaire (défi 277).

Fil rouge de la série : un mot ne veut rien dire seul. « Résilier » est dans « je vais résilier » ET dans « je ne
vais pas résilier » ; « le » est partout ; et un mot qu'on n'a jamais vu dans une classe n'y est pas impossible.

1. découper en MOTS, c'est d'abord ramener à une même écriture ce que les clients tapent de cent façons
   (majuscules, accents, « ﬁ », « 3ᵉ », apostrophes courbes) : casefold, NFKD, sans les accents ;
2. une NÉGATION change le sens des mots qui la suivent, jusqu'à la fin de la proposition : on les MARQUE
   (« non_resilier »), et deux mots de négation ne s'annulent pas (« ne … pas ») ;
3. ENTRAÎNER un Bayes naïf, c'est COMPTER : les documents de chaque classe, les mots de chaque classe ; le lissage
   de Laplace (+ alpha) évite qu'un mot absent d'une classe la rende impossible ;
4. PRÉDIRE, c'est additionner des LOGARITHMES (un produit de 300 probabilités vaut 0.0 en float), puis revenir aux
   probabilités en retranchant le plus grand score avant `exp` (« log-sum-exp ») ;
5. un mot RÉVÉLATEUR n'est pas un mot fréquent : c'est un mot plus probable dans la classe que dans TOUTE autre ;
6. un modèle EN LIGNE garde ses comptes à jour, et ne recalcule que ce qu'on lui demande : apprendre, oublier
   (le droit à l'effacement) et prédire coûtent le nombre de mots du commentaire, pas la taille du vocabulaire.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import heapq  # noqa: F401
import math
import unicodedata
from collections import Counter

from defis.verbatims import (  # noqa: F401
    FINS_DE_PORTEE,
    LETTRES_LIEES,
    NEGATIONS,
    PARTI,
    PONCTUATION,
    PREFIXE,
    RESTE,
    Modele,
    bilan,
    commentaires_exemple,
    generer_commentaires,
    nouveaux_exemple,
    regle_actuelle,
)

_SANS_LIGATURES = str.maketrans(LETTRES_LIEES)


def _texte(valeur, nom: str = "texte") -> str:
    if not isinstance(valeur, str):
        raise TypeError(f"{nom} doit être un texte (str), pas {type(valeur).__name__}")
    return valeur


def mots(texte) -> list[str]:
    """Défi 272 — Les mots d'un commentaire, écrits d'une seule façon."""
    # casefold : les minuscules « sans exception » ; NFKD : « ﬁ » → « fi », « 3ᵉ » → « 3e », « é » → « e » + accent
    decompose = unicodedata.normalize("NFKD", _texte(texte).casefold())
    lettres = "".join(c for c in decompose if not unicodedata.combining(c)).translate(_SANS_LIGATURES)
    resultat, mot = [], []
    for c in lettres:
        if c.isalnum():                      # une lettre ou un chiffre, de n'importe quel alphabet (pas « _ »)
            mot.append(c)
        elif mot:
            resultat.append("".join(mot))
            mot = []
    if mot:                                  # le dernier mot, s'il n'est suivi d'aucun séparateur
        resultat.append("".join(mot))
    return resultat


def avec_negations(texte) -> list[str]:
    """Défi 273 — Les mots, ceux qui suivent une négation marqués (« non_resilier »), proposition par proposition."""
    _texte(texte)
    propositions, courante = [], []
    for c in texte:
        if c in PONCTUATION:
            propositions.append("".join(courante))
            courante = []
        else:
            courante.append(c)
    propositions.append("".join(courante))

    resultat = []
    for proposition in propositions:
        nie = False                          # une nouvelle proposition commence sans négation
        for mot in mots(proposition):
            if mot in NEGATIONS:
                nie = True                   # « ne … pas » : le second mot ne referme rien
                resultat.append(mot)
            elif mot in FINS_DE_PORTEE:
                nie = False
                resultat.append(mot)
            else:
                resultat.append(PREFIXE + mot if nie else mot)
    return resultat


def _alpha(alpha) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha doit être un nombre, pas {type(alpha).__name__}")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"alpha doit être un nombre > 0 et fini, pas {alpha!r}")
    return alpha


def _mots_d_un_document(document, nom: str = "un document") -> list[str]:
    if isinstance(document, str) or not isinstance(document, (list, tuple)):
        raise TypeError(f"{nom} est une liste de mots (pas {type(document).__name__})")
    for mot in document:
        if not isinstance(mot, str):
            raise TypeError(f"un mot est un texte, pas {type(mot).__name__}")
    return document


def _log_proba(comptes: Counter, total: int, vocabulaire, alpha: float) -> dict:
    denominateur = total + alpha * len(vocabulaire)
    return {mot: math.log((comptes[mot] + alpha) / denominateur) for mot in vocabulaire}


def entrainer(documents, etiquettes, alpha=1.0) -> Modele:
    """Défi 274 — Compter, lisser, prendre les logarithmes : le Bayes naïf multinomial."""
    alpha = _alpha(alpha)
    if not isinstance(documents, (list, tuple)) or not isinstance(etiquettes, (list, tuple)):
        raise TypeError("documents et etiquettes sont des listes")
    if len(documents) != len(etiquettes):
        raise ValueError(f"{len(documents)} documents mais {len(etiquettes)} étiquettes")
    if not documents:
        raise ValueError("aucun document : rien à apprendre")
    for etiquette in etiquettes:
        if not isinstance(etiquette, str):
            raise TypeError(f"une étiquette est un texte, pas {type(etiquette).__name__}")

    nb_documents = Counter(etiquettes)
    comptes = {classe: Counter() for classe in nb_documents}
    for document, etiquette in zip(documents, etiquettes):
        comptes[etiquette].update(_mots_d_un_document(document))     # les OCCURRENCES, répétitions comprises

    classes = sorted(nb_documents)
    vocabulaire = sorted(set().union(*comptes.values()))             # TOUS les mots, toutes classes confondues
    return Modele(
        classes=classes,
        log_a_priori={c: math.log(nb_documents[c] / len(documents)) for c in classes},
        log_proba={c: _log_proba(comptes[c], sum(comptes[c].values()), vocabulaire, alpha) for c in classes},
    )


def _normaliser(scores: dict) -> dict:
    """exp(score) / somme, sans débordement : on retranche le plus grand score (« log-sum-exp »)."""
    plus_grand = max(scores.values())
    exps = {c: math.exp(s - plus_grand) for c, s in scores.items()}   # le plus grand vaut exp(0) = 1
    total = sum(exps.values())
    return {c: e / total for c, e in exps.items()}


def probabilites(modele, mots) -> dict:
    """Défi 275 — La probabilité de chaque classe pour une liste de mots, sans que le produit tombe à 0.0."""
    _mots_d_un_document(mots, "mots")
    vocabulaire = modele.log_proba[modele.classes[0]]
    connus = [mot for mot in mots if mot in vocabulaire]               # un mot jamais vu ne dit rien : on l'ignore
    scores = {c: modele.log_a_priori[c] + sum(modele.log_proba[c][mot] for mot in connus) for c in modele.classes}
    return _normaliser(scores)


def mots_revelateurs(modele, classe, k=10) -> list[tuple[str, float]]:
    """Défi 276 — Les `k` mots qui distinguent le mieux `classe` de TOUTES les autres classes."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k doit être un int, pas {type(k).__name__}")
    if k < 0:
        raise ValueError("k doit être >= 0")
    if classe not in modele.classes:
        raise ValueError(f"{classe!r} n'est pas une classe du modèle ({modele.classes})")
    autres = [c for c in modele.classes if c != classe]
    if not autres:
        raise ValueError("une seule classe : il n'y a rien à comparer")
    ecarts = []
    for mot, log_p in modele.log_proba[classe].items():
        ecart = log_p - max(modele.log_proba[c][mot] for c in autres)  # face à la classe CONCURRENTE la plus forte
        if ecart > 0:
            ecarts.append((mot, ecart))
    return heapq.nsmallest(k, ecarts, key=lambda paire: (-paire[1], paire[0]))


class ClassifieurEnLigne:
    """Défi 277 — Un Bayes naïf qui apprend, oublie et prédit sans jamais tout recompter."""

    def __init__(self, alpha=1.0) -> None:
        self._alpha = _alpha(alpha)
        self._documents = Counter()          # classe → nombre de documents
        self._comptes = {}                   # classe → Counter(mot → occurrences)
        self._totaux = Counter()             # classe → nombre total d'occurrences
        self._vocabulaire = Counter()        # mot → occurrences toutes classes confondues (> 0 : dans le vocabulaire)

    def apprendre(self, mots, etiquette) -> None:
        _mots_d_un_document(mots, "mots")
        _texte(etiquette, "etiquette")
        self._documents[etiquette] += 1
        self._comptes.setdefault(etiquette, Counter()).update(mots)
        self._totaux[etiquette] += len(mots)
        self._vocabulaire.update(mots)

    def oublier(self, mots, etiquette) -> None:
        _mots_d_un_document(mots, "mots")
        _texte(etiquette, "etiquette")
        a_retirer = Counter(mots)
        comptes = self._comptes.get(etiquette)
        # tout vérifier AVANT de modifier quoi que ce soit : une opération refusée ne change rien
        if comptes is None or any(comptes[mot] < n for mot, n in a_retirer.items()):
            raise ValueError(f"ce document n'a pas été appris avec l'étiquette {etiquette!r}")
        for mot, n in a_retirer.items():
            comptes[mot] -= n
            if comptes[mot] == 0:
                del comptes[mot]
            self._vocabulaire[mot] -= n
            if self._vocabulaire[mot] == 0:
                del self._vocabulaire[mot]   # plus personne ne l'a écrit : il sort du vocabulaire
        self._totaux[etiquette] -= len(mots)
        self._documents[etiquette] -= 1
        if self._documents[etiquette] == 0:  # plus aucun document : la classe disparaît
            del self._documents[etiquette], self._comptes[etiquette], self._totaux[etiquette]

    def probabilites(self, mots) -> dict:
        _mots_d_un_document(mots, "mots")
        if not self._documents:
            raise ValueError("rien n'a été appris")
        connus = [mot for mot in mots if mot in self._vocabulaire]
        n, v = sum(self._documents.values()), len(self._vocabulaire)
        scores = {}
        for classe in sorted(self._documents):
            comptes, denominateur = self._comptes[classe], self._totaux[classe] + self._alpha * v
            scores[classe] = math.log(self._documents[classe] / n) + sum(
                math.log((comptes[mot] + self._alpha) / denominateur) for mot in connus)
        return _normaliser(scores)

    def modele(self) -> Modele:
        if not self._documents:
            raise ValueError("rien n'a été appris")
        n, classes, vocabulaire = sum(self._documents.values()), sorted(self._documents), sorted(self._vocabulaire)
        return Modele(
            classes=classes,
            log_a_priori={c: math.log(self._documents[c] / n) for c in classes},
            log_proba={c: _log_proba(self._comptes[c], self._totaux[c], vocabulaire, self._alpha) for c in classes},
        )

    def __len__(self) -> int:
        return sum(self._documents.values())


# ---------------------------------------------------------------------------- des tests, comme dans tests/


def test_mots():
    assert mots("Je RÉSILIE l’offre ﬁbre !!! 😡") == ["je", "resilie", "l", "offre", "fibre"]
    assert mots("C'est la 3ᵉ coupure, 29,99 €") == ["c", "est", "la", "3e", "coupure", "29", "99"]
    assert mots("mot_de_passe") == ["mot", "de", "passe"] and mots("") == []


def test_avec_negations():
    assert avec_negations("Je n'ai jamais eu de coupure.") == ["je", "n", "non_ai", "jamais", "non_eu", "non_de",
                                                                "non_coupure"]
    assert avec_negations("Aucun souci, mais c'est plus cher") == ["aucun", "non_souci", "mais", "c", "est", "plus",
                                                                    "cher"]


def test_entrainer_et_probabilites():
    modele = entrainer([["a", "a", "b"], ["b"], ["c"]], ["x", "x", "y"])
    assert modele.classes == ["x", "y"]
    assert math.isclose(modele.log_proba["x"]["a"], math.log(3 / 7))          # (2 + 1) / (4 + 1 × 3)
    p = probabilites(modele, ["a"] * 2000)                                     # le produit vaudrait 0.0
    assert p["x"] > 0.999 and math.isclose(sum(p.values()), 1)


def test_mots_revelateurs():
    modele = entrainer([["prix", "le"], ["prix", "le"], ["merci", "le"], ["merci", "le"]], ["part", "part", "reste", "reste"])
    assert [m for m, _ in mots_revelateurs(modele, "part")] == ["prix"]          # « le » : partout, il ne révèle rien


def test_classifieur_en_ligne():
    clf = ClassifieurEnLigne()
    clf.apprendre(["a", "b"], "x")
    clf.apprendre(["c"], "y")
    clf.oublier(["c"], "y")
    assert len(clf) == 1 and clf.modele().classes == ["x"] and set(clf.modele().log_proba["x"]) == {"a", "b"}

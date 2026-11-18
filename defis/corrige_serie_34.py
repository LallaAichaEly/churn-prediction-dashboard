"""Série 34 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_34.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours une valeur
manquante (un NaN qui n'est égal à rien, un `groupby` qui oublie les lignes incomplètes, une valeur effacée qu'on
croit protectrice), ou un nombre à virgule qui fait refuser une question à laquelle on avait droit.

Fil rouge de la série : enlever le nom ne suffit pas. Tout le reste en découle.

1. on n'envoie que les colonnes DEMANDÉES (une liste blanche), et l'identifiant devient un pseudonyme calculé
   avec une clé (HMAC) : sans clé, un pseudonyme se retrouve en essayant tous les identifiants possibles ;
2. l'âge, le sexe et le code postal, anodins un par un, désignent souvent UNE personne : un fichier de
   prospection suffit pour remettre un nom sur la ligne. Une valeur effacée n'efface pas les autres ;
3. on mesure ce risque (le k du k-anonymat : chaque ligne doit ressembler à au moins k − 1 autres) et celui
   d'une classe où tout le monde a la même donnée sensible (le l de la l-diversité) ;
4. on le réduit en rendant les quasi-identifiants moins précis, le MOINS possible, et en retirant quelques
   lignes trop singulières plutôt que de flouter tout le monde ;
5. un tableau de comptages a ses propres fuites : une petite case cachée se retrouve par soustraction si les
   totaux sont publiés ;
6. la confidentialité différentielle répond autrement : chaque réponse reçoit un bruit calibré, et le total des
   questions est plafonné par un budget.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import math
from collections import Counter

from defis.confidentialite import (  # noqa: F401
    CLE_EXEMPLE,
    COLONNES_ETUDE,
    COMMUNES,
    HIERARCHIES,
    IDENTIFIANTS_DIRECTS,
    MOTIFS,
    QUASI_IDENTIFIANTS,
    Audit,
    BudgetEpuise,
    Classe,
    Generalisation,
    annuaire_exemple,
    appliquer,
    clients_exemple,
    comptages_exemple,
    generer_annuaire,
    generer_clients,
    tirage_hasard,
    tirages_fixes,
)


def _manquant(x) -> bool:
    """None ou NaN (NaN est le seul nombre différent de lui-même)."""
    return x is None or (isinstance(x, float) and x != x)


def _valeur(x):
    """Une valeur manquante, quelle que soit sa forme, devient None : deux NaN ne sont pas égaux, deux None si."""
    return None if _manquant(x) else x


def _entier(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# ============================================================================ Défi 206 : préparer l'export


def preparer_export(lignes: list[dict], colonnes: list[str], cle: bytes) -> list[dict]:
    """Une ligne par client : {"pseudo": HMAC de son identifiant, puis SEULEMENT les colonnes demandées}."""
    if not isinstance(cle, (bytes, bytearray)) or len(cle) < 16:
        raise ValueError("la clé doit faire au moins 16 octets (des bytes) : une clé courte se devine")
    colonnes = list(colonnes)
    interdites = sorted(set(colonnes) & IDENTIFIANTS_DIRECTS)
    if interdites:
        raise ValueError(f"{interdites} : un identifiant direct ne part jamais, même si on le demande")
    export = []
    for i, ligne in enumerate(lignes):
        identifiant = ligne.get("customer_id")
        if not isinstance(identifiant, str) or not identifiant.strip():
            raise ValueError(f"ligne {i} : pas d'identifiant client, impossible de lui donner un pseudonyme")
        # Le même client doit recevoir le même pseudonyme dans tous les exports : on normalise d'abord.
        message = identifiant.strip().upper().encode("utf-8")
        sortie = {"pseudo": hmac.new(cle, message, hashlib.sha256).hexdigest()}
        for colonne in colonnes:             # une LISTE BLANCHE : une colonne ajoutée demain au CRM ne part pas
            sortie[colonne] = ligne.get(colonne)
        export.append(sortie)
    return export


# ============================================================================ Défi 207 : l'attaque par recoupement


def reidentifier(publication: list[dict], annuaire: list[dict], quasi_identifiants: list[str]) -> dict[int, str]:
    """{numéro de ligne publiée: nom} pour chaque ligne qui ne correspond qu'à UNE personne de l'annuaire."""
    quasi = list(quasi_identifiants)
    if not quasi:
        raise ValueError("aucun quasi-identifiant : on ne peut rien recouper")
    # Un index par MOTIF de valeurs connues : une ligne dont l'âge est effacé se compare à l'annuaire sur le sexe
    # et le code postal seulement. Il y a au plus 2 ** len(quasi) motifs, et chacun ne se construit qu'une fois.
    index: dict[tuple, dict[tuple, list]] = {}
    trouves = {}
    for i, ligne in enumerate(publication):
        connus = tuple(not _manquant(ligne.get(q)) for q in quasi)
        if connus not in index:
            compte: dict[tuple, list] = {}
            for personne in annuaire:
                cle = tuple(_valeur(personne.get(q)) for q, connu in zip(quasi, connus) if connu)
                if cle in compte:
                    compte[cle][0] += 1                    # deux homonymes parfaits : deux candidats
                else:
                    compte[cle] = [1, personne.get("nom")]
            index[connus] = compte
        cle = tuple(ligne.get(q) for q, connu in zip(quasi, connus) if connu)
        candidats = index[connus].get(cle)
        if candidats is not None and candidats[0] == 1:   # un seul candidat : l'attaquant a un nom
            trouves[i] = candidats[1]
    return trouves


# ============================================================================ Défi 208 : mesurer le risque


def audit(lignes: list[dict], quasi_identifiants: list[str], sensible: str) -> Audit:
    """Audit(k, l, classes) : le k-anonymat et la l-diversité d'un fichier, classe par classe."""
    quasi = list(quasi_identifiants)
    if not quasi:
        raise ValueError("aucun quasi-identifiant : rien à auditer")
    if sensible in quasi:
        raise ValueError(f"« {sensible} » ne peut pas être à la fois quasi-identifiant et attribut sensible")
    groupes: dict[tuple, list] = {}             # valeurs -> [effectif, valeurs sensibles vues]
    for ligne in lignes:
        # Une valeur manquante est une valeur comme une autre pour l'attaquant (« celui dont l'âge manque ») : on
        # la GARDE, sous une seule forme. Le `groupby` de pandas, lui, jette ces lignes par défaut (dropna=True).
        valeurs = tuple(_valeur(ligne.get(q)) for q in quasi)
        groupe = groupes.setdefault(valeurs, [0, set()])
        groupe[0] += 1
        groupe[1].add(_valeur(ligne.get(sensible)))
    if not groupes:
        raise ValueError("aucune ligne : rien à auditer")
    classes = [Classe(valeurs, effectif, len(vues)) for valeurs, (effectif, vues) in groupes.items()]
    # Tri STABLE : à égalité d'effectif et de diversité, l'ordre de première apparition est conservé.
    classes.sort(key=lambda c: (c.effectif, c.diversite))
    return Audit(min(c.effectif for c in classes), min(c.diversite for c in classes), classes)


# ============================================================================ Défi 209 : généraliser


def generaliser(lignes: list[dict], hierarchies: dict, k: int, suppression_max: int = 0) -> Generalisation:
    """La généralisation la MOINS haute qui rend le fichier k-anonyme, en retirant au plus `suppression_max` lignes."""
    lignes = list(lignes)
    attributs = list(hierarchies)
    if not attributs:
        raise ValueError("aucune hiérarchie : rien à généraliser")
    if not _entier(k) or not 1 <= k <= len(lignes):
        raise ValueError(f"k = {k!r} : il faut un entier entre 1 et le nombre de lignes ({len(lignes)})")
    if not _entier(suppression_max) or not 0 <= suppression_max < len(lignes):
        raise ValueError(f"suppression_max = {suppression_max!r} : un entier >= 0, plus petit que le nombre de lignes")
    hauteurs = [range(len(hierarchies[a])) for a in attributs]
    # Tous les vecteurs de niveaux, par hauteur totale croissante. `product` les donne dans l'ordre
    # lexicographique et `sorted` est STABLE : à hauteur égale, le premier dans cet ordre passe devant.
    for vecteur in sorted(itertools.product(*hauteurs), key=sum):
        cles = [tuple(hierarchies[a][n](ligne.get(a)) for a, n in zip(attributs, vecteur)) for ligne in lignes]
        effectifs = Counter(cles)
        trop_rares = sum(1 for cle in cles if effectifs[cle] < k)     # des LIGNES, pas des classes
        if trop_rares > suppression_max:
            continue
        gardees = []
        for ligne, cle in zip(lignes, cles):
            if effectifs[cle] >= k:
                copie = dict(ligne)                                     # jamais la ligne reçue elle-même
                copie.update(zip(attributs, cle))
                gardees.append(copie)
        return Generalisation(dict(zip(attributs, vecteur)), gardees, len(lignes) - len(gardees))
    raise ValueError("aucune généralisation ne suffit : le dernier niveau de chaque hiérarchie devrait tout réunir")


# ============================================================================ Défi 210 : le secret statistique


def secret_statistique(tableau: list[list[int]], seuil: int) -> list[list[bool]]:
    """Le masque des cases à cacher : les petites cases, puis de quoi empêcher de les retrouver par soustraction."""
    if not _entier(seuil) or seuil < 1:
        raise ValueError(f"seuil = {seuil!r} : un entier >= 1")
    valeurs = [list(ligne) for ligne in tableau]
    if len(valeurs) < 2 or any(len(ligne) != len(valeurs[0]) for ligne in valeurs) or len(valeurs[0]) < 2:
        raise ValueError("il faut un tableau rectangulaire d'au moins 2 lignes et 2 colonnes")
    if any(not _entier(x) or x < 0 for ligne in valeurs for x in ligne):
        raise ValueError("des comptages : des entiers >= 0")
    n, m = len(valeurs), len(valeurs[0])
    # 1) Le secret PRIMAIRE : une case de 1 à seuil − 1 personnes les expose. Une case à 0 n'expose personne.
    cache = [[1 <= x < seuil for x in ligne] for ligne in valeurs]
    # 2) Le secret SECONDAIRE : une ligne (ou une colonne) qui n'a qu'UNE case cachée la trahit, puisque son
    #    total est publié. On en cache une deuxième, la plus petite, et on recommence depuis le début : cacher dans
    #    une colonne peut laisser une LIGNE avec une seule case cachée.
    while True:
        rangees = [[(i, j) for j in range(m)] for i in range(n)] + [[(i, j) for i in range(n)] for j in range(m)]
        rangee = next((r for r in rangees if sum(cache[i][j] for i, j in r) == 1), None)
        if rangee is None:
            return cache
        visibles = [(valeurs[i][j], position, (i, j)) for position, (i, j) in enumerate(rangee) if not cache[i][j]]
        _, _, (i, j) = min(visibles)            # la plus petite ; à égalité, la première de la rangée
        cache[i][j] = True


# ============================================================================ Défi 211 : le budget de confidentialité


def _positif_fini(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0 < x < math.inf   # refuse aussi NaN


class Budget:
    """Répondre à des comptages et à des sommes avec un bruit de Laplace, sans dépasser `epsilon_total`."""

    def __init__(self, epsilon_total: float, tirage) -> None:
        if not _positif_fini(epsilon_total):
            raise ValueError(f"epsilon_total = {epsilon_total!r} : un nombre > 0")
        self._total = epsilon_total
        self._depense = 0.0
        self._tirage = tirage

    @property
    def restant(self) -> float:
        return max(0.0, self._total - self._depense)

    def _depenser(self, epsilon: float) -> None:
        if not _positif_fini(epsilon):
            raise ValueError(f"epsilon = {epsilon!r} : un nombre > 0")
        # À 1e-9 près : 0.1 + 0.1 + 0.1 vaut 0.30000000000000004, et un budget de 0.3 doit permettre trois questions.
        if self._depense + epsilon > self._total + 1e-9:
            raise BudgetEpuise(f"il reste {self.restant:.3g} de budget, la question en demande {epsilon}")
        self._depense += epsilon               # on paie AVANT de répondre ; une question refusée ne coûte rien

    def _bruit(self, echelle: float) -> float:
        u = self._tirage() - 0.5                # un seul tirage par réponse, uniforme dans ]−0,5 ; 0,5[
        return -echelle * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))

    def compter(self, lignes, condition, epsilon: float) -> float:
        self._depenser(epsilon)
        vrai = sum(1 for ligne in lignes if condition(ligne))
        # Un client de plus ou de moins change un comptage d'au plus 1 (la SENSIBILITÉ) : bruit d'échelle 1 / epsilon.
        return float(vrai + self._bruit(1 / epsilon))

    def sommer(self, lignes, colonne: str, plafond: float, epsilon: float) -> float:
        if not _positif_fini(plafond):
            raise ValueError(f"plafond = {plafond!r} : un nombre > 0")
        self._depenser(epsilon)
        # Chaque valeur est ramenée dans [0, plafond] : un seul client ne peut plus changer la somme de plus de
        # `plafond`. Sans plafond, la facture d'un seul client se lirait dans le résultat.
        vrai = sum(min(max(ligne.get(colonne), 0), plafond) for ligne in lignes if not _manquant(ligne.get(colonne)))
        return float(vrai + self._bruit(plafond / epsilon))


# ============================================================================ Tests (comme dans tests/)


def test_preparer_export():
    export = preparer_export(clients_exemple(), ["age", "churn"], CLE_EXEMPLE)
    assert set(export[0]) == {"pseudo", "age", "churn"} and len(export[0]["pseudo"]) == 64
    autre = preparer_export([{"customer_id": " c00412 "}], [], CLE_EXEMPLE)
    assert autre[0]["pseudo"] == export[0]["pseudo"]                  # même client, même pseudonyme
    assert preparer_export([{"customer_id": "C00412"}], [], b"une autre cle de 32 octets !!!!!")[0] != export[0]


def test_reidentifier():
    publication = [{"age": 34, "sexe": "F"}, {"age": None, "sexe": "M"}, {"age": 50, "sexe": "F"}]
    annuaire = [{"nom": "A", "age": 34, "sexe": "F"}, {"nom": "B", "age": 34, "sexe": "M"},
                {"nom": "C", "age": 50, "sexe": "F"}, {"nom": "D", "age": 50, "sexe": "F"}]
    assert reidentifier(publication, annuaire, ["age", "sexe"]) == {0: "A", 1: "B"}


def test_audit():
    lignes = [{"age": 30, "motif": "prix"}, {"age": 30, "motif": "prix"}, {"age": None, "motif": "impayé"},
              {"age": float("nan"), "motif": None}]
    resultat = audit(lignes, ["age"], "motif")
    assert (resultat.k, resultat.l) == (2, 1)
    assert resultat.classes == [Classe((30,), 2, 1), Classe((None,), 2, 2)]


def test_generaliser():
    lignes = [{"age": a, "churn": 1} for a in (31, 33, 36, 52)]
    g = generaliser(lignes, {"age": HIERARCHIES["age"]}, 2, suppression_max=1)
    assert (g.niveaux, g.supprimees) == ({"age": 2}, 1)                     # 52 ans retiré : tranches de 10 ans
    assert g.lignes == [{"age": "30-39", "churn": 1}] * 3 and lignes[0]["age"] == 31
    assert generaliser(lignes, {"age": HIERARCHIES["age"]}, 4).niveaux == {"age": 4}


def test_secret_statistique():
    assert secret_statistique([[10, 2], [20, 30]], 3) == [[True, True], [True, True]]
    assert secret_statistique([[10, 20], [30, 40]], 3) == [[False, False], [False, False]]


def test_budget():
    budget = Budget(0.3, tirages_fixes([0.5, 0.75, 0.5]))
    assert budget.compter([1, 2, 3], lambda x: x > 1, 0.1) == 2.0              # u = 0,5 : aucun bruit
    assert math.isclose(budget.compter([1, 2, 3], lambda x: x > 1, 0.1), 2 + 10 * math.log(2))
    budget.compter([], bool, 0.1)                                              # 0.1 + 0.1 + 0.1 : permis
    try:
        budget.compter([], bool, 0.1)
    except BudgetEpuise:
        pass
    else:
        raise AssertionError("le budget aurait dû être épuisé")

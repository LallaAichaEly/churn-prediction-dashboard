"""Série 30 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_30.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours un
`str.split()` à la place de `mots()`, une liste lue en entier alors qu'une dichotomie suffisait, ou un
décalage de positions après une insertion dans un texte.

Fil rouge de la série : un moteur de recherche travaille AVANT qu'on lui pose la question.

1. le même découpage partout : l'index et la requête passent par `mots()`. Deux découpages différents, et
   « Résilier, » ne retrouve jamais « résilier » ;
2. on prépare une fois (l'index : mot → identifiants TRIÉS), pour ne plus jamais relire les notes. Une
   requête ne lit alors qu'une petite partie de l'index : la liste la plus courte, et quelques cases des
   autres trouvées par dichotomie ;
3. le texte d'origine est sacré : on cherche sur la forme normalisée (sans accents, en minuscules), mais
   on montre le texte TEL QU'IL EST ÉCRIT, grâce aux positions (début, fin) de chaque mot ;
4. « pertinent » se définit, et se calcule : BM25 récompense un mot RARE (idf), qui revient souvent
   (sans que la 10e fois compte autant que la 1re), dans une note COURTE.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import bisect
import heapq
import math

from defis.recherche import (  # noqa: F401
    Comptee,
    generer_notes,
    generer_vocabulaire,
    index_frequences,
    mots,
    mots_et_positions,
    normaliser,
    notes_exemple,
)

# ============================================================================ Défi 182 : index_inverse


def index_inverse(notes: dict[int, str]) -> dict[str, list[int]]:
    """{mot: identifiants triés des notes qui le contiennent, chacun une fois}."""
    index: dict[str, list[int]] = {}
    # Parcourir les notes dans l'ordre des identifiants : chaque liste se remplit DÉJÀ triée, sans
    # jamais rien trier ni chercher dedans (un `if identifiant not in liste` relit toute la liste).
    for identifiant in sorted(notes):
        texte = notes[identifiant]
        if not isinstance(texte, str):        # None, NaN : une case vide du fichier, pas une erreur
            continue
        for mot in set(mots(texte)):          # set : « fibre » deux fois dans la note → une seule entrée
            index.setdefault(mot, []).append(identifiant)
    return index                              # un vrai dict : lire un mot absent lève KeyError


# ============================================================================ Défi 183 : rechercher


def rechercher(index: dict[str, list[int]], requete: str) -> list[int]:
    """Les identifiants (triés) des notes qui contiennent TOUS les mots de la requête."""
    termes = set(mots(requete))               # le même découpage que l'index ; un mot répété compte une fois
    if not termes:
        return []
    listes = []
    for terme in termes:
        liste = index.get(terme)
        if not liste:                         # un mot que personne n'a écrit : la réponse est déjà connue
            return []
        listes.append(liste)

    # La plus COURTE d'abord : chacun de ses identifiants est un candidat, cherché par dichotomie dans les
    # autres. 3 candidats dans une liste d'un million : 3 × 20 cases lues, au lieu d'un million.
    listes.sort(key=len)                      # trie la liste DES listes (la nôtre), pas les listes de l'index
    plus_courte, autres = listes[0], listes[1:]
    depart = [0] * len(autres)                # les candidats croissent : on ne cherche jamais en arrière
    resultat = []
    for identifiant in plus_courte:
        for k, liste in enumerate(autres):
            position = bisect.bisect_left(liste, identifiant, depart[k])
            depart[k] = position
            if position == len(liste) or liste[position] != identifiant:
                break
        else:                                 # aucun `break` : présent dans toutes les listes
            resultat.append(identifiant)
    return resultat                           # une NOUVELLE liste, jamais celle de l'index


# ============================================================================ Défi 184 : completer


def completer(vocabulaire: list[str], frequences: dict[str, int], prefixe: str, k: int = 5) -> list[str]:
    """Les k mots du vocabulaire (trié) qui commencent par `prefixe`, les plus fréquents d'abord."""
    debut_de_mot = normaliser(prefixe).strip()       # « Rési » se tape, « resi » se cherche
    if k <= 0:
        return []
    # Le vocabulaire est TRIÉ : tous les mots qui commencent par « resi » sont côte à côte, à partir de
    # l'endroit où « resi » lui-même serait rangé. bisect_LEFT, pour garder le mot « resi » s'il existe.
    i = bisect.bisect_left(vocabulaire, debut_de_mot)
    candidats = []
    while i < len(vocabulaire):
        mot = vocabulaire[i]
        if not mot.startswith(debut_de_mot):  # le premier mot qui ne commence plus par le préfixe : fini
            break
        candidats.append(mot)
        i += 1
    # Les k meilleurs : la fréquence décroissante, puis l'ordre alphabétique (et pas reverse=True, qui
    # inverserait AUSSI l'ordre alphabétique des ex aequo).
    return heapq.nsmallest(k, candidats, key=lambda mot: (-frequences.get(mot, 0), mot))


# ============================================================================ Défi 185 : surligner


def surligner(texte: str, requete: str, avant: str = "[", apres: str = "]") -> str:
    """Le texte d'origine, chaque mot de la requête encadré par `avant` et `apres`."""
    termes = set(mots(requete))
    # On construit un NOUVEAU texte, morceau par morceau, de gauche à droite : les positions données par
    # mots_et_positions restent justes (insérer dans `texte` les décalerait toutes).
    morceaux, recopie_jusqua = [], 0
    for mot, debut, fin in mots_et_positions(texte):
        if mot in termes:
            morceaux += [texte[recopie_jusqua:debut], avant, texte[debut:fin], apres]
            recopie_jusqua = fin
    morceaux.append(texte[recopie_jusqua:])
    return "".join(morceaux)


# ============================================================================ Défi 186 : classer (BM25)


def classer(requete: str, frequences: dict[str, dict[int, int]], longueurs: dict[int, int], k: int = 10,
            k1: float = 1.2, b: float = 0.75) -> list[tuple[int, float]]:
    """Les k notes les plus pertinentes selon BM25 : [(identifiant, score)], du meilleur au moins bon."""
    if not k1 >= 0:                           # `k1 < 0` laisserait passer NaN (toute comparaison à NaN est fausse)
        raise ValueError(f"k1 doit être un nombre >= 0, pas {k1!r}")
    if not 0 <= b <= 1:
        raise ValueError(f"b doit être entre 0 et 1, pas {b!r}")
    termes = list(dict.fromkeys(mots(requete)))      # sans doublon, dans l'ordre de la requête
    n = len(longueurs)
    if k <= 0 or n == 0:
        return []
    moyenne = sum(longueurs.values()) / n             # sur TOUTES les notes, vides comprises
    if moyenne == 0:
        return []

    scores: dict[int, float] = {}
    for terme in termes:
        notes = frequences.get(terme)
        if not notes:                         # un mot que personne n'a écrit n'apporte rien
            continue
        df = len(notes)
        # L'idf de BM25 « + 1 » (celui de Lucene et d'Elasticsearch) : toujours > 0. Sans le « 1 + », un mot
        # présent dans plus de la moitié des notes aurait un idf NÉGATIF : le contenir ferait baisser le score.
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for identifiant, tf in notes.items():
            # tf / (tf + k1 · …) : la 1re occurrence compte beaucoup, la 10e presque plus (saturation).
            # (1 - b + b · dl / moyenne) : une note plus longue que la moyenne est un peu pénalisée.
            normalisation = k1 * (1 - b + b * longueurs[identifiant] / moyenne)
            scores[identifiant] = scores.get(identifiant, 0.0) + idf * tf * (k1 + 1) / (tf + normalisation)
    # On ne trie que les CANDIDATS (les notes qui contiennent au moins un mot), et seulement les k premiers.
    return heapq.nsmallest(k, scores.items(), key=lambda paire: (-paire[1], paire[0]))


# ============================================================================ Défi 187 : meilleur_extrait


def meilleur_extrait(texte: str, requete: str) -> tuple[int, int] | None:
    """(début, fin) du plus court passage qui contient tous les mots de la requête présents dans le texte."""
    jetons = mots_et_positions(texte)
    presents = set(mots(requete)) & {mot for mot, _, _ in jetons}
    if not presents:
        return None

    # Fenêtre glissante : `droite` avance d'un mot à chaque tour ; dès que la fenêtre contient tout, on
    # avance `gauche` tant qu'elle contient encore tout. Chaque mot entre une fois et sort une fois : O(n).
    dans_la_fenetre = {mot: 0 for mot in presents}
    manquants = len(presents)
    meilleur = None
    gauche = 0
    for droite, (mot, _, _) in enumerate(jetons):
        if mot in dans_la_fenetre:
            dans_la_fenetre[mot] += 1
            if dans_la_fenetre[mot] == 1:
                manquants -= 1
        while manquants == 0:
            if meilleur is None or droite - gauche < meilleur[1] - meilleur[0]:   # « < » : à égalité, la 1re
                meilleur = (gauche, droite)
            sortant = jetons[gauche][0]
            if sortant in dans_la_fenetre:
                dans_la_fenetre[sortant] -= 1
                if dans_la_fenetre[sortant] == 0:
                    manquants += 1
            gauche += 1
    premier, dernier = meilleur
    return jetons[premier][1], jetons[dernier][2]    # des positions dans le TEXTE, pas des numéros de mots


# ============================================================================ Des tests, comme dans tests/


def test_l_index_ne_garde_chaque_note_qu_une_fois_et_dans_l_ordre():
    index = index_inverse({30: "fibre, FIBRE", 4: "Fibre ?", 9: None})
    assert index == {"fibre": [4, 30]}


def test_rechercher_ne_lit_presque_rien_de_la_liste_longue():
    index = {"rare": Comptee([5, 70_000, 123_456]), "client": Comptee(range(0, 200_000))}
    assert rechercher(index, "Client RARE") == [5, 70_000, 123_456]
    assert index["client"].lues < 100
    assert rechercher(index, "rare absent") == [] and rechercher(index, "?!") == []


def test_completer_garde_le_prefixe_lui_meme_et_departage_par_ordre_alphabetique():
    vocabulaire = ["fibre", "fibres", "fidelite", "fil"]
    frequences = {"fibre": 3, "fibres": 3, "fidelite": 9, "fil": 1}
    assert completer(vocabulaire, frequences, "FI", k=3) == ["fidelite", "fibre", "fibres"]
    assert completer(vocabulaire, frequences, "fibre") == ["fibre", "fibres"]


def test_surligner_garde_le_texte_d_origine():
    assert surligner("Résilier ? RÉSILIER, résiliation.", "resilier") == "[Résilier] ? [RÉSILIER], résiliation."
    assert surligner("rien à voir", "") == "rien à voir"


def test_bm25_prefere_le_mot_rare_et_la_note_courte():
    frequences, longueurs = index_frequences({1: "client fibre", 2: "client", 3: "client fibre " + "mot " * 20})
    classement = classer("client fibre", frequences, longueurs)
    assert [identifiant for identifiant, _ in classement] == [1, 3, 2]


def test_le_meilleur_extrait_est_le_plus_court_puis_le_premier():
    texte = "fibre a b c resilier d fibre resilier"
    debut, fin = meilleur_extrait(texte, "résilier fibre")
    assert texte[debut:fin] == "fibre resilier"
    assert meilleur_extrait("aucun rapport", "fibre") is None

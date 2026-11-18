"""Série 30 — Chercher dans les notes : index inversé, recherche, autocomplétion, surlignage, BM25, extrait.

🔎 VOUS VOULEZ DÉVELOPPER ? LA BARRE DE RECHERCHE EST DANS TOUTES LES APPLICATIONS. ICI, VOUS L'ÉCRIVEZ.

Après chaque appel, le conseiller écrit une note libre : « Client mécontent : la FIBRE coupe tous les
soirs. Menace de résilier. » Il y en a des centaines de milliers. Awa, de l'équipe de rétention, veut taper
« résilier fibre » et voir, en un clin d'œil : les notes qui contiennent les deux mots, les plus parlantes
d'abord, avec le passage qui répond, surligné. Et pendant qu'elle tape « rési », la barre lui propose
« resilier » et « resiliation ». Relire toutes les notes à chaque touche prendrait des secondes : un
moteur de recherche (Elasticsearch, la recherche de votre éditeur, celle d'un site) prépare un INDEX une
fois, puis répond en ne lisant presque rien. Cette série vous fait écrire ses six pièces :

    182 index_inverse     l'index : pour chaque mot, les identifiants TRIÉS des notes qui le contiennent
    183 rechercher        les notes qui contiennent TOUS les mots, sans relire une liste d'un million
    184 completer         proposer les mots qui commencent par « rési » pendant qu'on tape
    185 surligner         montrer les mots trouvés dans le texte d'origine (accents et majuscules gardés)
    186 classer           les notes les plus pertinentes d'abord : la formule BM25
    187 meilleur_extrait  le plus court passage qui contient tous les mots (fenêtre glissante)

Prérequis : la série 00 (listes, dict, tri). La recherche dichotomique (`bisect`, défi 112 de la série 18)
et la normalisation du texte (série 20) l'éclairent, mais tout ce qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- on ne cherche pas dans le texte brut : « Résilier, », « RÉSILIER » et « résilier » sont le même mot.
  `mots(texte)` (déjà importée) découpe et normalise : minuscules, sans accents, sans ponctuation. L'index
  ET la requête passent par elle ; deux découpages différents, et la recherche rate des notes ;
- un INDEX INVERSÉ, c'est le dictionnaire d'un livre à l'envers : {mot: [identifiants des notes]}. Les
  identifiants y sont TRIÉS, chacun une fois : c'est ce qui permet ensuite de chercher par dichotomie ;
- chercher deux mots, c'est INTERSECTER deux listes triées. Si l'une a 3 identifiants et l'autre un
  million, on ne lit pas le million : on cherche chacun des 3 dans la grande, par dichotomie (`bisect`,
  une vingtaine de cases lues pour un million) ;
- dans une liste TRIÉE de mots, tous ceux qui commencent par « resi » sont côte à côte. On saute au
  premier avec `bisect_left`, puis on avance tant que ça commence par « resi » ;
- montrer, c'est l'inverse de chercher : on retrouve les mots par leur forme normalisée, mais on affiche
  le texte TEL QU'IL EST ÉCRIT. `mots_et_positions(texte)` donne, pour chaque mot, sa position (début, fin)
  dans le texte d'origine ;
- « pertinent » se calcule : un mot RARE vaut plus qu'un mot qu'on trouve partout (« client »), un mot qui
  revient compte plus (mais la 10e fois moins que la 1re), et une note courte qui le contient est plus
  centrée sur lui qu'un long compte rendu. C'est la formule BM25, celle d'Elasticsearch par défaut.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_30 import *
    >>> notes = notes_exemple()
    >>> notes[17]
    'Client mécontent : la FIBRE coupe tous les soirs. Menace de résilier.'
    >>> mots(notes[17])
    ['client', 'mecontent', 'la', 'fibre', 'coupe', 'tous', 'les', 'soirs', 'menace', 'de', 'resilier']

Quand les six défis sont réussis, vous avez un petit moteur de recherche :

    >>> index = index_inverse(notes)
    >>> rechercher(index, "Résilier la FIBRE")                     # toutes les notes qui ont les trois mots
    [17, 30]
    >>> completer(sorted(index), {mot: len(ids) for mot, ids in index.items()}, "rési")
    ['resilier', 'resiliation']
    >>> frequences, longueurs = index_frequences(notes)
    >>> [identifiant for identifiant, score in classer("résilier fibre", frequences, longueurs, k=3)]
    [30, 17, 25]
    >>> debut, fin = meilleur_extrait(notes[30], "résilier fibre")
    >>> surligner(notes[30][debut:fin], "résilier fibre")
    '[résilier] la [fibre]'

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les six défis sont indépendants (le
   correcteur fabrique lui-même l'index dont le 183 a besoin) ; l'ordre proposé est le plus progressif.
2. Lancez `python scripts/defis.py --defi 182` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_30.py`.

Les outils communs (`notes_exemple`, `normaliser`, `mots`, `mots_et_positions`, `index_frequences`,
`Comptee`, `generer_notes`, `generer_vocabulaire`) sont dans `defis/recherche.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import bisect  # noqa: F401 — bisect.bisect_left(liste_triee, x, lo) : où ranger x, en ~20 lectures pour un million
import heapq  # noqa: F401 — heapq.nsmallest(k, valeurs, key=…) : les k plus petits, sans tout trier
import math  # noqa: F401 — math.log(x) : le logarithme NATUREL (ln) ; math.log10, c'est autre chose

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

# ============================================================================ L'index


def index_inverse(notes: dict[int, str]) -> dict[str, list[int]]:
    """Défi 182 🟢 — L'index inversé : {mot: identifiants des notes qui contiennent ce mot}.

        >>> index = index_inverse(notes_exemple())
        >>> index["fibre"]                       # « fibre » est deux fois dans la note 30 : elle compte une fois
        [17, 25, 30]
        >>> index["resilier"]                    # « résilier », « Résilier ? » et « RÉSILIER » : le même mot
        [3, 11, 17, 19, 30]
        >>> sorted(index_inverse({7: "Fibre, fibre !", 2: "La fibre.", 9: None}).items())
        [('fibre', [2, 7]), ('la', [2])]

    Les règles :

    - les mots sont ceux que donne `mots(texte)` (minuscules, sans accents, sans ponctuation), et AUCUN
      autre découpage : la requête passera par la même fonction ;
    - chaque liste est TRIÉE par identifiant croissant, et chaque identifiant n'y est qu'une fois (même si
      le mot revient dans la note). Les notes arrivent dans le désordre : un export ne trie rien ;
    - une note qui n'est pas un texte (None, NaN : une case vide du fichier) ne donne aucun mot, sans
      planter ; une note vide ou sans mot (« ?! ») non plus ;
    - le résultat est un vrai `dict`, avec des `list` : lire un mot absent doit lever KeyError. Un
      `defaultdict` ajouterait une entrée vide à chaque recherche ratée, et l'index grossirait tout seul ;
    - ne modifiez pas `notes`.

    Ce que vous apprenez : la structure de données au cœur de tous les moteurs de recherche, et une idée
    qui sert partout : préparer UNE fois ce qui servira à chaque question.

    Indice : parcourez `sorted(notes)` (les identifiants, dans l'ordre) : chaque liste se remplit alors
    déjà triée. `set(mots(texte))` donne chaque mot une fois ; `index.setdefault(mot, []).append(i)`.
    """
    raise NotImplementedError


# ============================================================================ Chercher


def rechercher(index: dict[str, list[int]], requete: str) -> list[int]:
    """Défi 183 🟡 — Les identifiants (triés) des notes qui contiennent TOUS les mots de la requête.

        >>> index = index_inverse(notes_exemple())
        >>> rechercher(index, "résilier FIBRE")
        [17, 30]
        >>> rechercher(index, "fibre résiliation")            # aucune note n'a les deux
        []
        >>> rechercher(index, "fibre wifi"), rechercher(index, "?!")   # « wifi » n'est nulle part ; aucun mot
        ([], [])
        >>> grande = Comptee(range(0, 2_000_000, 2))            # « client » : un million de notes
        >>> rapide = {"client": grande, "concurrent": Comptee([14, 1_500_001, 1_999_998])}
        >>> rechercher(rapide, "client concurrent"), grande.lues < 200
        ([14, 1999998], True)

    Les règles :

    - la requête est découpée par `mots(requete)`, comme les notes ; un mot répété ne compte qu'une fois ;
    - un mot absent de l'index → [] (pas de KeyError) ; une requête sans aucun mot → [] ;
    - le résultat est une NOUVELLE liste, triée, sans doublon (jamais une liste de l'index elle-même :
      l'appelant pourrait la modifier) ; ne modifiez pas l'index ni ses listes ;
    - ne lisez jamais une liste longue en entier quand une courte suffit : le correcteur vous donne des
      `Comptee` (des listes qui comptent les cases lues). Avec 3 identifiants d'un côté et un million de
      l'autre, il accepte quelques centaines de lectures, pas un million. Un `set(liste)`, un `x in liste`
      ou un parcours des deux listes en parallèle lisent tout.

    Ce que vous apprenez : intersecter des listes triées (la requête ET de tous les moteurs), choisir par
    où commencer (la plus courte), et la dichotomie, qui lit 20 cases sur un million.

    Indice : les listes des mots, triées par `len` (gratuit) ; chaque identifiant de la plus courte est un
    candidat. Pour chaque autre liste : `p = bisect.bisect_left(liste, candidat, depart)` puis
    `p < len(liste) and liste[p] == candidat`. Les candidats croissent : gardez `depart = p` pour la suite.
    """
    raise NotImplementedError


# ============================================================================ Compléter pendant qu'on tape


def completer(vocabulaire: list[str], frequences: dict[str, int], prefixe: str, k: int = 5) -> list[str]:
    """Défi 184 🟡 — Les k mots du vocabulaire qui commencent par `prefixe`, les plus fréquents d'abord.

    `vocabulaire` est la liste TRIÉE des mots de l'index (normalisés, sans doublon) ; `frequences[mot]` est
    le nombre de notes qui contiennent le mot.

        >>> index = index_inverse(notes_exemple())
        >>> vocabulaire, frequences = sorted(index), {mot: len(ids) for mot, ids in index.items()}
        >>> completer(vocabulaire, frequences, "Rési")
        ['resilier', 'resiliation']
        >>> frequences["commercial"], frequences["coupe"], frequences["coupure"], frequences["courrier"]
        (2, 1, 1, 1)
        >>> completer(vocabulaire, frequences, "co", k=3)      # à égalité, l'ordre alphabétique
        ['commercial', 'coupe', 'coupure']
        >>> completer(vocabulaire, frequences, "wifi"), completer(vocabulaire, frequences, "co", k=0)
        ([], [])

    Les règles :

    - le préfixe est normalisé comme les mots (`normaliser`), et débarrassé des espaces autour : « Rési »
      cherche « resi ». Le préfixe vide propose les mots les plus fréquents de tout le vocabulaire ;
    - un mot égal au préfixe lui-même commence par le préfixe : « fibre » est proposé quand on a tapé
      « fibre » ;
    - au plus k mots, par fréquence DÉCROISSANTE, puis par ordre alphabétique en cas d'égalité ; k <= 0 → [] ;
    - le vocabulaire peut compter des centaines de milliers de mots : le correcteur vous donne un
      `Comptee`, et n'accepte qu'une dichotomie (une vingtaine de lectures) plus les mots qui commencent
      vraiment par le préfixe. Parcourir tout le vocabulaire, même avec `startswith`, est refusé ;
    - ne modifiez ni le vocabulaire ni les fréquences.

    Ce que vous apprenez : la recherche par préfixe dans une liste triée (ce que fait une barre de recherche
    à chaque touche, et un index B-tree pour `LIKE 'resi%'` en SQL), la fin d'une plage, et départager des
    ex aequo sans tout inverser.

    Indice : `i = bisect.bisect_left(vocabulaire, p)`, puis avancez tant que `vocabulaire[i].startswith(p)`.
    Puis `heapq.nsmallest(k, candidats, key=lambda mot: (-frequences[mot], mot))` : le « - » inverse la
    fréquence, et SEULEMENT elle (`reverse=True` inverserait aussi l'ordre alphabétique).
    """
    raise NotImplementedError


# ============================================================================ Montrer ce qu'on a trouvé


def surligner(texte: str, requete: str, avant: str = "[", apres: str = "]") -> str:
    """Défi 185 🟢 — Le texte d'origine, où chaque mot de la requête est encadré par `avant` et `apres`.

        >>> notes = notes_exemple()
        >>> surligner(notes[30], "fibre résilier")
        'Troisième appel pour la [fibre] : coupure, technicien, coupure. Client à bout, veut [résilier] la [fibre] et le mobile.'
        >>> surligner("Résiliation ? Non : RÉSILIER, puis résilié.", "resilier", "<mark>", "</mark>")
        'Résiliation ? Non : <mark>RÉSILIER</mark>, puis résilié.'
        >>> surligner("Demande de remise", "de")                 # « de » dans « Demande » n'est pas le mot « de »
        'Demande [de] remise'

    Les règles :

    - un mot du texte est surligné quand sa forme normalisée (celle de `mots_et_positions`) est l'un des
      mots de la requête (découpée par `mots`). Un mot ENTIER : « fibre » ne surligne pas « fibres » ;
    - tout le reste du texte est recopié À L'IDENTIQUE : majuscules, accents, ponctuation, espaces. Le mot
      surligné aussi : on montre « RÉSILIER » tel qu'il est écrit, pas « resilier » ;
    - chaque mot trouvé est encadré séparément, même s'il touche le suivant : « [fibre] [coupée] » ;
    - une requête sans aucun mot → le texte inchangé ; un texte qui n'est pas un str → TypeError.

    Ce que vous apprenez : séparer la forme qu'on CHERCHE de celle qu'on MONTRE, travailler avec des
    positions dans un texte, et pourquoi on ne modifie pas un texte qu'on est en train de parcourir.

    Indice : `str.replace` remplace AUSSI à l'intérieur des mots et ignore les accents. Construisez un
    nouveau texte morceau par morceau : pour chaque (mot, debut, fin) trouvé, ajoutez
    `texte[recopie_jusqua:debut]`, `avant`, `texte[debut:fin]`, `apres`, puis `recopie_jusqua = fin`.
    """
    raise NotImplementedError


# ============================================================================ Les meilleures notes d'abord


def classer(requete: str, frequences: dict[str, dict[int, int]], longueurs: dict[int, int], k: int = 10,
            k1: float = 1.2, b: float = 0.75) -> list[tuple[int, float]]:
    """Défi 186 🔴 — Les k notes les plus pertinentes selon BM25 : [(identifiant, score)], meilleure d'abord.

    `frequences[mot]` = {identifiant: nombre de fois où le mot est dans la note} ; `longueurs[identifiant]`
    = nombre de mots de la note, pour TOUTES les notes. `index_frequences(notes)` vous les donne.

        >>> frequences, longueurs = index_frequences(notes_exemple())
        >>> [(i, round(score, 3)) for i, score in classer("résilier fibre", frequences, longueurs)]
        [(30, 1.654), (17, 1.622), (25, 1.292), (11, 0.64), (19, 0.612), (3, 0.563)]
        >>> classer("résilier fibre", frequences, longueurs, k=2) == classer("fibre résilier", frequences, longueurs)[:2]
        True
        >>> classer("wifi", frequences, longueurs), classer("", frequences, longueurs)
        ([], [])

    La note 30 est la plus longue, mais elle dit deux fois « fibre » ; la 25 n'a pas « résilier », mais
    elle est courte et « fibre » est plus rare que « résilier ».

    La formule (BM25, avec l'idf de Lucene et d'Elasticsearch) :

        N         = len(longueurs), le nombre de notes (les vides comprises)
        moyenne   = la longueur moyenne des notes (somme des longueurs / N)
        df        = le nombre de notes qui contiennent le mot : len(frequences[mot])
        idf       = ln(1 + (N - df + 0.5) / (df + 0.5))                        math.log, pas math.log10
        score     = la somme, sur chaque mot DISTINCT de la requête que la note contient, de
                    idf × tf × (k1 + 1) / (tf + k1 × (1 - b + b × longueur / moyenne))
                    où tf = frequences[mot][identifiant] et longueur = longueurs[identifiant]

    Les règles :

    - la requête est découpée par `mots(requete)` ; un mot répété ne compte qu'une fois, un mot absent
      n'apporte rien ;
    - seules les notes qui contiennent au moins un mot de la requête sont classées ;
    - au plus k paires (identifiant, score), par score DÉCROISSANT, puis par identifiant croissant en cas
      d'égalité ; k <= 0, aucune note, ou aucun mot trouvé → [] ;
    - k1 < 0 (ou NaN), b hors de [0, 1] (ou NaN) → ValueError. Ne modifiez pas les dict reçus.

    Ce que vous apprenez : ce que « pertinent » veut dire pour un moteur de recherche, traduire une formule
    en code sans en perdre un morceau, et le piège de l'idf classique, ln((N - df + 0.5) / (df + 0.5)), qui
    devient NÉGATIF pour un mot présent dans plus de la moitié des notes.

    Indice : `termes = list(dict.fromkeys(mots(requete)))` (sans doublon). Un dict `scores` {identifiant:
    score}, rempli en parcourant SEULEMENT `frequences[terme]` pour chaque terme. Puis
    `heapq.nsmallest(k, scores.items(), key=lambda p: (-p[1], p[0]))`.
    """
    raise NotImplementedError


# ============================================================================ Le passage qui répond


def meilleur_extrait(texte: str, requete: str) -> tuple[int, int] | None:
    """Défi 187 🔴 — (début, fin) du plus court passage du texte qui contient tous les mots de la requête.

        >>> notes = notes_exemple()
        >>> debut, fin = meilleur_extrait(notes[30], "fibre résilier")
        >>> debut, fin, notes[30][debut:fin]          # « fibre … résilier » (9 mots) perd contre 3 mots
        (82, 99, 'résilier la fibre')
        >>> debut, fin = meilleur_extrait(notes[17], "résilier fibre prix")   # « prix » n'y est pas : ignoré
        >>> notes[17][debut:fin]
        'FIBRE coupe tous les soirs. Menace de résilier'
        >>> meilleur_extrait(notes[17], "wifi"), meilleur_extrait(notes[17], "")
        (None, None)

    Les règles :

    - les mots de la requête viennent de `mots(requete)` (un mot répété compte une fois) ; ceux du texte de
      `mots_et_positions(texte)`. Le passage doit contenir chaque mot de la requête PRÉSENT dans le texte ;
      les mots absents sont ignorés. Aucun mot de la requête dans le texte (ou aucun mot dans la requête)
      → None ;
    - « plus court » se compte en MOTS, du premier mot du passage au dernier ; à égalité, le premier dans
      le texte ;
    - le résultat est (début, fin), des positions dans le TEXTE : `texte[début:fin]` va du début du premier
      mot du passage à la fin du dernier (pas des numéros de mots) ;
    - le texte peut avoir 20 000 mots : le correcteur compte les lignes exécutées dans votre fichier.
      Essayer chaque début possible et avancer jusqu'à tout trouver, c'est des millions de lignes : refusé.

    Ce que vous apprenez : la FENÊTRE GLISSANTE à deux indices (« minimum window substring », un classique
    d'entretien), qui transforme O(n²) en O(n), et ce que fait la ligne grise sous chaque résultat d'un
    moteur de recherche.

    Indice : `dans_la_fenetre` = {mot: 0 pour chaque mot présent}, `manquants` = leur nombre. `droite` avance
    d'un mot à chaque tour (compte + 1 ; s'il passe à 1, `manquants -= 1`). Puis, TANT QUE `manquants == 0` :
    notez la fenêtre si elle est plus courte, retirez le mot de `gauche` (compte - 1 ; s'il tombe à 0,
    `manquants += 1`) et avancez `gauche`.
    """
    raise NotImplementedError

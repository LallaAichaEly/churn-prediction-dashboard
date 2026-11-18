"""Série 29 — Comparer et fusionner : diff, patch, fusion à trois voies, tables, configurations.

🔀 VOUS VOULEZ DÉVELOPPER ? VOUS TAPEREZ `git diff` ET `git merge` CENT FOIS PAR SEMAINE. ICI, VOUS LES ÉCRIVEZ.

Inès et Malik partent du même `score_risque` (le dernier commit commun, la « base »). Inès ajoute une
docstring et passe le seuil d'appels au support à 4 ; Malik ajoute une règle pour la fibre et plafonne le
score. Git montre leurs modifications (`git diff`), les transporte (un patch), puis les réunit
(`git merge`) sans rien demander… sauf quand les deux ont touché la même ligne. Pendant ce temps, le
nouveau code de préparation des données doit produire EXACTEMENT la même table que l'ancien, et la
configuration du modèle a, elle aussi, été modifiée sur deux branches. Cette série vous fait écrire les
six outils qui comparent et fusionnent :

    176 difference          le plus court script de modifications (la plus longue sous-suite commune)
    177 diff_unifie         l'afficher comme `git diff` : « @@ -5,3 +5,4 @@ », contexte, blocs
    178 appliquer_patch     l'appliquer à un fichier qui a un peu bougé, ou le refuser EN ENTIER
    179 fusion_trois_voies  `git merge` : base, nous, eux… et les conflits <<<<<<< ||||||| ======= >>>>>>>
    180 comparer_tables     le nouveau code donne-t-il la même table ? (NaN, 0.1 + 0.2, "01000" et 1000)
    181 fusion_config       fusionner deux versions d'une configuration JSON, clé par clé

Prérequis : la série 00, les listes et les dict. La distance d'édition (défi 164, série 27) est la cousine
du défi 176 : même tableau, autre question. Faites l'atelier Git avant ou après : vous y verrez ces
outils au travail (`python scripts/atelier_git.py`).

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un fichier, ici, est une LISTE de lignes (sans « \\n ») : `texte.splitlines()` la donne. Les lignes sont
  numérotées à partir de 1 (dans un diff, un éditeur, un message d'erreur), les listes à partir de 0 ;
- comparer position par position ne marche pas : une ligne ajoutée en haut décale toutes les autres. On
  cherche la PLUS LONGUE SOUS-SUITE COMMUNE (des lignes présentes dans les deux versions, dans le même
  ordre) : tout le reste est supprimé (« - ») ou ajouté (« + »). Le script le plus court garde le plus de
  lignes ;
- un patch ne transporte pas que des numéros de ligne : il transporte du CONTEXTE (les lignes autour). C'est
  ce qui permet de l'appliquer à un fichier qui a un peu bougé, et de REFUSER quand le fichier ne
  ressemble plus à ce qu'il attendait ;
- une fusion a besoin de TROIS versions. Avec deux seulement, « Malik a supprimé la ligne » et « Inès l'a
  ajoutée » se ressemblent trait pour trait. On compare chaque côté à la base : si un seul côté a changé
  un morceau, on prend sa version ; si les deux l'ont changé pareil, aussi ; sinon c'est un CONFLIT, qu'on
  montre au lieu de choisir ;
- « égal » se définit : pour une table, NaN et NaN, 0.1 + 0.2 et 0.3, 12 et 12.0 sont égaux ; "01000" et
  1000 ne le sont pas. Pour une configuration, True et 1, 1 et 1.0, une clé à None et une clé absente ne
  le sont pas non plus.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_29 import *
    >>> base, nous, eux = versions_exemple()
    >>> print(montrer(base))
      1 | def score_risque(client):
      2 |     score = 0
      3 |     if client["contrat"] == "Mensuel":
      4 |         score += 2
      5 |     if client["appels_support"] >= 3:
      6 |         score += 2
      7 |     if client["anciennete_mois"] < 6:
      8 |         score += 1
      9 |     return score

Quand les six défis sont réussis, vous avez un petit Git :

    >>> patch = diff_unifie(difference(base, nous), nom="score.py")      # git diff
    >>> appliquer_patch(base, patch) == nous                              # git apply
    True
    >>> fusion, conflits = fusion_trois_voies(base, nous, eux)            # git merge
    >>> conflits, len(fusion)
    (0, 12)
    >>> fusion[-1]
    '    return min(score, 5)'

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites le 176 avant le 177 et le 179 (ils
   s'en servent) ; les défis 180 et 181 sont indépendants.
2. Lancez `python scripts/defis.py --defi 176` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_29.py`.

Les outils communs (`versions_exemple`, `versions_en_conflit`, `config_exemple`, `tables_exemple`,
`generer_versions`, `montrer`, `PatchRefuse`, `NOUS`, `BASE`, `SEPARATEUR`, `EUX`) sont dans
`defis/versions.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import copy  # noqa: F401 — copy.deepcopy(d) : une copie qui ne partage rien, même les dict imbriqués
import math  # noqa: F401 — math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
import numbers  # noqa: F401 — isinstance(x, numbers.Number) : int, float, et les nombres de NumPy
import re  # noqa: F401 — re.fullmatch(motif, texte) : le motif doit couvrir TOUT le texte

import pandas as pd  # noqa: F401 — pd.isna(x) : None, NaN, NaT

from defis.versions import (  # noqa: F401
    BASE,
    EUX,
    NOUS,
    SEPARATEUR,
    PatchRefuse,
    config_exemple,
    generer_versions,
    montrer,
    tables_exemple,
    versions_en_conflit,
    versions_exemple,
)

# ============================================================================ Le plus court script


def difference(ancien: list[str], nouveau: list[str]) -> list[tuple[str, str]]:
    """Défi 176 🟡 — Le plus court script de modifications qui transforme `ancien` en `nouveau`.

    Le résultat est une liste de paires (opération, ligne), dans l'ordre du fichier :
    (" ", ligne) la ligne est gardée, ("-", ligne) elle est supprimée, ("+", ligne) elle est ajoutée.

        >>> difference(["a", "b", "c"], ["a", "B", "c"])
        [(' ', 'a'), ('-', 'b'), ('+', 'B'), (' ', 'c')]
        >>> difference(["}", "", "}"], ["}", "x", "", "}"])         # les lignes répétées restent à leur place
        [(' ', '}'), ('+', 'x'), (' ', ''), (' ', '}')]
        >>> base, nous, eux = versions_exemple()
        >>> [o for o in difference(base, nous) if o[0] != " "][1:]     # Inès : le seuil 3 devient 4
        [('-', '    if client["appels_support"] >= 3:'), ('+', '    if client["appels_support"] >= 4:')]
        >>> sum(op != " " for op, _ in difference(base, ["# titre"] + base))  # UNE ligne ajoutée, pas 10 changements
        1

    Les règles :

    - les lignes « » et « - », dans l'ordre, redonnent `ancien` ; les lignes « » et « + » redonnent `nouveau` ;
    - le script est le plus COURT possible : il garde autant de lignes que la plus longue sous-suite
      commune (s'il y a plusieurs scripts aussi courts, n'importe lequel convient) ;
    - entre deux lignes gardées, toutes les « - » d'abord, puis toutes les « + » (comme git diff) ;
    - les lignes se comparent telles quelles : « x = 1 » et « x = 1 » suivi d'une espace sont différentes ;
    - `ancien` ou `nouveau` qui est un texte (str) → TypeError : il serait comparé lettre par lettre ;
    - 3 000 lignes dont une seule change : votre fonction doit répondre aussitôt (le correcteur compte les
      lignes exécutées) ; et ne modifiez pas les listes reçues.

    Ce que vous apprenez : la plus longue sous-suite commune (programmation dynamique, la cousine de la
    distance d'édition), reconstruire une solution à partir du tableau, et réduire un problème avant de le
    résoudre (ce que fait git diff).

    Indice : 1) avancez `debut` tant que ancien[debut] == nouveau[debut], puis reculez `fin` de la même façon
    par la fin : il ne reste qu'un petit milieu `a`, `b`. 2) `t[i][j]` = la longueur de la plus longue
    sous-suite commune de a[i:] et b[j:] : `t[i][j] = t[i+1][j+1] + 1` si a[i] == b[j], sinon
    `max(t[i+1][j], t[i][j+1])`, rempli de i = len(a) - 1 à 0. 3) Partez de i = j = 0 : lignes égales → " " ;
    sinon « - » si t[i+1][j] >= t[i][j+1], « + » sinon. 4) Rangez chaque bloc : `sorted(bloc, key=…)` est
    STABLE, il garde l'ordre des lignes.
    """
    raise NotImplementedError


# ============================================================================ Le format de git diff


def diff_unifie(operations: list[tuple[str, str]], nom: str = "fichier", contexte: int = 3) -> list[str]:
    """Défi 177 🟢 — Les lignes du « diff unifié » (le format de `git diff`) qui correspond aux opérations.

        >>> ops = difference(["a", "b", "c", "d"], ["a", "B", "c", "d"])
        >>> print("\\n".join(diff_unifie(ops, nom="lettres.txt", contexte=1)))
        --- a/lettres.txt
        +++ b/lettres.txt
        @@ -1,3 +1,3 @@
         a
        -b
        +B
         c
        >>> diff_unifie(difference(["a", "b"], ["a", "b"]))           # rien n'a changé : git diff n'écrit rien
        []
        >>> diff_unifie(difference(["a", "b", "c"], ["a", "B", "c"]), contexte=0)[2]   # UNE ligne : pas de « ,1 »
        '@@ -2 +2 @@'
        >>> diff_unifie(difference([], ["x"]))[2]                    # rien dans l'ancien : « APRÈS la ligne 0 »
        '@@ -0,0 +1 @@'

    Les règles :

    - deux lignes d'en-tête : « --- a/{nom} » (l'ancien fichier), puis « +++ b/{nom} » (le nouveau) ;
    - puis des BLOCS. Un bloc montre une ou plusieurs modifications, avec jusqu'à `contexte` lignes gardées
      avant et après (moins au début ou à la fin du fichier). Deux modifications séparées par AU PLUS
      2 × contexte lignes gardées vont dans le même bloc (leurs contextes se toucheraient) ;
    - chaque bloc commence par « @@ -a,b +c,d @@ » : il commence à la ligne a de l'ancien fichier (numérotée
      à partir de 1) et en montre b (le contexte et les « - ») ; c et d, pareil pour le nouveau (le contexte
      et les « + ») ;
    - une plage d'UNE ligne s'écrit sans « ,1 » : « -5 » ; une plage VIDE donne le numéro de la ligne d'AVANT
      (« -5,0 » : rien, juste après la ligne 5 ; « -0,0 » : au tout début) ;
    - chaque ligne du bloc est l'opération suivie de la ligne : " a", "-b", "+B" ;
    - aucune modification → [] ; contexte négatif → ValueError ; ne modifiez pas la liste reçue.

    Ce que vous apprenez : lire un `git diff` sans hésiter (c'est ce qu'on relit dans chaque revue de code),
    et les deux pièges éternels des numéros de ligne : commencer à 1, et l'indice négatif qui, en Python,
    repart de la fin (`ops[-2:5]` est vide).

    Indice : notez d'abord, pour chaque opération, le numéro de la prochaine ligne de l'ancien et du nouveau
    fichier (deux compteurs qui partent de 1 : une « - » n'avance que l'ancien, une « + » que le nouveau).
    Puis les indices des changements, regroupés quand `k - dernier - 1 <= 2 * contexte`. Chaque bloc va de
    `max(0, premier - contexte)` à `min(len(ops), dernier + 1 + contexte)`.
    """
    raise NotImplementedError


# ============================================================================ Appliquer un patch


def appliquer_patch(lignes: list[str], patch: list[str]) -> list[str]:
    """Défi 178 🟡 — Le fichier `lignes` après application du `patch` (un diff unifié, comme au défi 177).

        >>> base, nous, eux = versions_exemple()
        >>> patch = diff_unifie(difference(base, nous), nom="score.py")
        >>> appliquer_patch(base, patch) == nous
        True
        >>> deplace = ["# relu par l'équipe"] + base            # le fichier a bougé : une ligne de plus en haut
        >>> appliquer_patch(deplace, patch) == ["# relu par l'équipe"] + nous
        True
        >>> try:
        ...     appliquer_patch(base[:4] + base[5:], patch)        # la ligne 5 a disparu : le patch ne colle plus
        ... except PatchRefuse:
        ...     print("refusé, et le fichier n'a pas été touché")
        refusé, et le fichier n'a pas été touché

    Lire le patch :

    - tout ce qui précède le premier « @@ » est un en-tête, ignoré (« --- a/… », « +++ b/… », « diff --git… ») ;
    - « @@ -a,b +c,d @@ » ouvre un bloc ; ce qui suit le second « @@ » est ignoré (git y écrit le nom de la
      fonction). « -5 » sans virgule veut dire « -5,1 » ;
    - le bloc contient EXACTEMENT b lignes « » ou « - » et d lignes « » ou « + » : COMPTEZ-les. Une ligne
      supprimée « -- total » (un commentaire SQL) s'écrit « --- total » : ce n'est pas un en-tête ;
    - un en-tête mal formé, une ligne qui ne commence ni par « », ni par « - », ni par « + », un bloc qui a
      plus ou moins de lignes qu'annoncé → PatchRefuse.

    Appliquer chaque bloc, dans l'ordre :

    - ses lignes « » et « - » (les lignes ATTENDUES) doivent se trouver telles quelles dans le fichier. Les
      numéros parlent de l'ANCIEN fichier : le bloc devrait commencer à l'indice a - 1 (ou a, si b == 0 :
      « -5,0 », c'est APRÈS la ligne 5) ;
    - si elles n'y sont pas, cherchez-les autour : écart +1, -1, +2, -2… (plus bas d'abord, comme GNU patch),
      jamais avant la fin du bloc précédent ni après la fin du fichier ;
    - l'écart trouvé pour un bloc s'ajoute à la position prévue des blocs suivants (le fichier a bougé pour
      eux aussi) ;
    - puis les lignes attendues sont remplacées par les lignes « » et « + » ;
    - si un bloc est introuvable → PatchRefuse, et RIEN n'est appliqué (même pas les blocs d'avant). Ne
      modifiez jamais la liste reçue : renvoyez une nouvelle liste.

    Ce que vous apprenez : lire un format texte en COMPTANT plutôt qu'en devinant, un traitement « tout ou
    rien », et pourquoi un patch transporte du contexte : `git apply`, `git cherry-pick` et `git rebase`
    reposent tous là-dessus.

    Indice : deux temps. D'abord lire TOUS les blocs (sans toucher au fichier) en une liste de
    (a, b, attendues, nouvelles), avec `re.fullmatch(r"@@ -(\\d+)(?:,(\\d+))? \\+(\\d+)(?:,(\\d+))? @@.*", ligne)`.
    Puis construire `sortie` : `lues` (lignes de l'ancien fichier déjà recopiées), `decalage` (0 au départ),
    et pour chaque bloc `sortie += lignes[lues:position] + nouvelles ; lues = position + b`.
    """
    raise NotImplementedError


# ============================================================================ Fusionner trois versions


def fusion_trois_voies(base: list[str], nous: list[str], eux: list[str]) -> tuple[list[str], int]:
    """Défi 179 🔴 — (lignes fusionnées, nombre de conflits), comme `git merge-file --diff3 nous base eux`.

        >>> fusion, conflits = fusion_trois_voies(*versions_exemple())
        >>> conflits
        0
        >>> print(montrer(fusion[4:]))
          1 |         score += 2
          2 |     if client["appels_support"] >= 4:
          3 |         score += 2
          4 |     if client["anciennete_mois"] < 6:
          5 |         score += 1
          6 |     if client["internet"] == "Fibre":
          7 |         score += 1
          8 |     return min(score, 5)
        >>> fusion, conflits = fusion_trois_voies(*versions_en_conflit())
        >>> conflits
        1
        >>> print("\\n".join(fusion[4:10]))
        <<<<<<< nous
            if client["appels_support"] >= 4:
        ||||||| base
            if client["appels_support"] >= 3:
        =======
            if client["appels_support"] >= 5:
        >>> fusion[10], fusion[11]                          # la fin du conflit, puis la suite du fichier
        ('>>>>>>> eux', '        score += 2')

    La méthode (l'algorithme « diff3 » que Git utilise) :

    1. `difference(base, nous)` dit quelles lignes de base Inès a GARDÉES, et où elles sont dans `nous` ;
       pareil pour `difference(base, eux)` ;
    2. une ligne de base gardée des DEUX côtés est un point d'appui : elle va dans la fusion ;
    3. entre deux points d'appui, on regarde ce que chacun a fait du morceau de base qui les sépare
       (b = base[i:i2], n = nous[j:j2], e = eux[k:k2]) :
       - n == b : seul Malik y a touché (ou personne) → on prend e ;
       - e == b : seule Inès y a touché → on prend n ;
       - n == e : les deux ont fait la MÊME modification → on la prend une fois ;
       - sinon c'est un CONFLIT : on écrit NOUS, les lignes de n, BASE, celles de b, SEPARATEUR, celles de e,
         puis EUX (les constantes importées en haut du fichier), et on compte un conflit ;
    4. on continue après le point d'appui, jusqu'au bout des trois fichiers.

    Les règles :

    - deux modifications VOISINES (aucune ligne gardée des deux côtés entre elles) sont un conflit, comme
      dans Git : rien ne garantit qu'elles aient un sens ensemble ;
    - supprimer d'un côté et modifier de l'autre est un conflit ;
    - ne modifiez pas les listes reçues.

    Le correcteur n'utilise que des versions où personne ne DÉPLACE de lignes et où chaque ligne ajoutée est
    nouvelle : la plus longue sous-suite commune y est unique, donc la bonne fusion aussi (c'est celle de Git).

    Ce que vous apprenez : ce que fait vraiment `git merge`, pourquoi il a besoin de l'ancêtre commun, et
    lire un conflit avec sa section « base » (`git config --global merge.conflictStyle diff3`).

    Indice : transformez chaque `difference` en dict {indice dans base: indice dans l'autre version} (deux
    compteurs i et j : « » les avance tous les deux, « - » seulement i, « + » seulement j). Puis une boucle
    `while True` : cherchez le prochain i2 >= i présent dans les DEUX dict (ou len(base)), traitez le morceau,
    puis `i, j, k = i2 + 1, …`.
    """
    raise NotImplementedError


# ============================================================================ Comparer deux tables


def comparer_tables(avant: pd.DataFrame, apres: pd.DataFrame, cle: str) -> dict:
    """Défi 180 🟢 — Ce qui a changé entre deux tables, alignées par la colonne `cle`.

    Vous réécrivez la préparation des données (plus rapide, plus lisible). Avant de la mettre en production,
    on vérifie qu'elle produit LA MÊME table que l'ancienne : un « test de non-régression des données ».

        >>> avant, apres = tables_exemple()
        >>> r = comparer_tables(avant, apres, cle="client_id")
        >>> r["modifies"]
        {'C005': {'contrat': ('Mensuel', 'Annuel')}}
        >>> r["ajoutes"], r["supprimes"]
        (['C009'], ['C006'])
        >>> r["colonnes_ajoutees"], r["colonnes_supprimees"]
        (['score_risque'], ['segment'])

    Le résultat est un dict avec exactement ces cinq clés :

    - "ajoutes" / "supprimes" : les clés présentes seulement dans `apres` / seulement dans `avant` (triées) ;
    - "modifies" : {clé: {colonne: (valeur avant, valeur après)}}, seulement pour les clés présentes des deux
      côtés, seulement les colonnes COMMUNES, et seulement les clés qui ont vraiment changé ;
    - "colonnes_ajoutees" / "colonnes_supprimees" : les colonnes présentes d'un seul côté (triées).

    Deux valeurs sont ÉGALES :

    - si elles sont toutes deux absentes (None, NaN, NaT : `pd.isna`). Attention, `nan == nan` vaut False ;
    - si ce sont deux nombres proches : `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)` (0.1 + 0.2 et 0.3,
      12 et 12.0, un entier NumPy et un int Python) ;
    - sinon si `a == b`, à condition que ce ne soit pas un texte comparé à un nombre : "01000" et 1000 sont
      DIFFÉRENTS (le code postal a perdu son zéro : une vraie régression).

    Et :

    - les lignes s'alignent par la clé, pas par leur position (le nouveau code peut trier autrement) ;
    - une clé en double ou absente (NaN) dans une table → ValueError ; une colonne `cle` qui manque →
      KeyError (ou ValueError) ;
    - ne modifiez pas les tables reçues (pas de `inplace=True`).

    Ce que vous apprenez : écrire un test de non-régression des données, définir l'égalité (NaN, flottants,
    types), et aligner deux tables par une clé plutôt que par la position.

    Indice : `a = avant.set_index(cle)` (une NOUVELLE table), `a.index.duplicated().any()`,
    `set(a.index) & set(b.index)`, puis `a.at[k, colonne]` pour lire une case.
    """
    raise NotImplementedError


# ============================================================================ Fusionner deux configurations


def fusion_config(base: dict, nous: dict, eux: dict) -> tuple[dict, list[str]]:
    """Défi 181 🔴 — Fusionner trois versions d'une configuration : (configuration fusionnée, conflits).

    Git fusionne des LIGNES ; un fichier JSON se fusionne mieux clé par clé. C'est la même idée qu'au
    défi 179, appliquée à des dict imbriqués.

        >>> base, nous, eux = config_exemple()
        >>> fusion, conflits = fusion_config(base, nous, eux)
        >>> conflits
        ['relance.budget_euros']
        >>> fusion["modele"]
        {'nom': 'logistic', 'seuil': 0.4, 'variables_derivees': True}
        >>> fusion["relance"]
        {'canal': 'email', 'budget_euros': 6000, 'exclure': ['Deux ans']}
        >>> fusion["surveillance"]
        {'psi_max': 0.25}

    Pour chaque clé présente dans au moins une des trois versions (b, n, e : sa valeur dans base, nous, eux,
    ou « absente ») :

    - n et e identiques → on la prend (si elles sont absentes toutes les deux, la clé disparaît) ;
    - n identique à b → seul « eux » a changé : on prend e (absente : la clé disparaît) ;
    - e identique à b → on prend n ;
    - n et e sont deux dict → on fusionne CES dict de la même façon (base : b si c'est un dict, sinon {}) ;
    - sinon c'est un conflit : on garde n (ou son absence) et on ajoute le chemin de la clé aux conflits,
      avec des points : "relance.budget_euros".

    « Identiques » veut dire même type ET même valeur, jusque dans les listes : True n'est pas 1, 1 n'est pas
    1.0. Une clé ABSENTE n'est pas une clé à None (null en JSON). Une liste se compare en entier.

    Et :

    - la liste des conflits est triée ;
    - la configuration fusionnée ne partage aucun dict ni aucune liste avec les versions reçues (copiez ce
      que vous reprenez : `copy.deepcopy`), et les versions reçues ne sont pas modifiées ;
    - une version qui n'est pas un dict → TypeError.

    Ce que vous apprenez : la récursion sur des données imbriquées, la différence entre « absent » et
    « vide » (le piège de `dict.get`), et l'égalité de Python (`True == 1`) qui n'est pas toujours la bonne.

    Indice : `ABSENTE = object()` (un objet unique, qui n'est égal qu'à lui-même), puis
    `base.get(cle, ABSENTE)`. Une fonction récursive `_fusion(b, n, e, chemin, conflits)` qui remplit la
    liste `conflits` reçue.
    """
    raise NotImplementedError

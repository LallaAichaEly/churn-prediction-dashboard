"""Série 27 — Retrouver le même client : comparer, bloquer, décider, regrouper, fusionner, mesurer.

🪞 VOUS VOULEZ DÉVELOPPER ? UN JOUR, ON VOUS DIRA : « ON A 300 000 CLIENTS… OU PEUT-ÊTRE 220 000. »

Le fichier clients réunit le site, les boutiques et le support. Hélène Dupré y est TROIS fois : « Hélène
Dupré » sur le site, « DUPRE Helene » en boutique, « Helene Dupre » au support. Sa fiche boutique est
résiliée : pour le modèle, Hélène est partie (un faux churn), alors qu'elle est toujours cliente sur le site.
Son ancienneté vient de la fiche la plus récente : 2 ans au lieu de 7. Et le marketing lui enverra trois
fois le même SMS, alors qu'elle les a refusés au support. Avant tout modèle, il faut retrouver les fiches
d'une même personne. C'est la « résolution d'entités » (ou dédoublonnage), et cette série vous la fait écrire :

    164 distance_edition       « Dupont » et « Dupond » : à combien de fautes de frappe ?
    165 regrouper              A = B et B = C, donc A = C : les groupes (union-find), même sur 200 000 fiches
    166 paires_candidates      ne JAMAIS comparer tout le monde à tout le monde (le blocage)
    167 score_correspondance   même personne ? accord, désaccord… et valeur absente, trois cas différents
    168 fiche_de_reference     une seule fiche par client, sans perdre un refus de SMS
    169 qualite_dedoublonnage  précision et rappel… en PAIRES, sans fabriquer les paires

Prérequis : la série 00, les dictionnaires et les `set` (défi 45). La série 20 (le texte : accents,
téléphones) éclaire les normalisations ; la série 09 (complexité) les défis 165 et 166.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- comparer deux fiches donne trois réponses, pas deux : D'ACCORD (le même e-mail), en DÉSACCORD (deux
  e-mails différents), ou RIEN À COMPARER (l'une des deux n'a pas d'e-mail). Deux fiches sans e-mail n'ont
  PAS le même e-mail. Une valeur absente s'écrit None, "", "   ", ou NaN (une case vide lue par pandas) ;
  et NaN est sournois : `bool(float("nan"))` vaut True, et `nan == nan` vaut False. `pd.isna(x)` le voit ;
- à l'inverse, False et 0 ne sont PAS des valeurs absentes : « a refusé les SMS » est une réponse ;
- on ne compare jamais toutes les paires : 300 000 fiches, c'est 45 milliards de paires. On ne compare
  que les fiches qui partagent une CLÉ DE BLOCAGE (le même e-mail, le même téléphone, le même nom dans le
  même code postal). Et une clé ABSENTE ne réunit personne : sinon, le « bloc » des fiches sans téléphone
  en réunirait 100 000 ;
- « se ressembler » n'est pas transitif (F02 ressemble à F01, F01 à F03, mais F02 pas assez à F03), « être
  le même client » l'est : on décide paire par paire, puis on REGROUPE (union-find) ;
- une erreur de dédoublonnage n'en vaut pas une autre : fusionner deux personnes (la mère et le fils) est
  bien plus grave que laisser deux fiches d'une même personne. D'où deux mesures, précision et rappel.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_27 import *
    >>> fiches = fiches_exemple()
    >>> len(fiches), len(VERITE_EXEMPLE)              # 14 fiches, 8 vrais clients
    (14, 8)
    >>> fiches["F01"]["nom"], fiches["F02"]["nom"], fiches["F03"]["nom"]
    ('Hélène Dupré', 'DUPRE Helene', 'Helene Dupre')
    >>> normaliser_nom("DUPRE Helene"), normaliser_telephone("+33 6 12 34 56 78")
    ('dupre helene', '0612345678')
    >>> [cle(fiches["F02"]) for cle in CLES]           # les trois clés de blocage du défi 166
    [None, '0612345678', 'dupre helene|75011']

Quand les six défis sont réussis, ils s'assemblent en un dédoublonnage complet :

    >>> paires, trop_gros = paires_candidates(fiches)
    >>> retenues = [(a, b) for a, b in sorted(paires) if score_correspondance(fiches[a], fiches[b]) >= SEUIL]
    >>> groupes = regrouper(fiches, retenues)
    >>> groupes[0], len(groupes)
    (['F01', 'F02', 'F03'], 8)
    >>> qualite_dedoublonnage(groupes, VERITE_EXEMPLE)["f1"]
    1.0

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites le 164 avant le 167 (le score s'en
   sert) ; les autres défis sont indépendants.
2. Lancez `python scripts/defis.py --defi 164` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_27.py`.

Les outils communs (`fiches_exemple`, `VERITE_EXEMPLE`, `generer_fiches`, `normaliser_nom`,
`normaliser_email`, `normaliser_telephone`, `cle_email`, `cle_telephone`, `cle_nom_code_postal`, `CLES`,
`SEUIL`) sont dans `defis/doublons.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import itertools  # noqa: F401 — itertools.combinations(liste, 2) : toutes les paires, chacune une fois
from collections import Counter  # noqa: F401 — Counter(...) compte les répétitions
from datetime import date  # noqa: F401

import pandas as pd  # noqa: F401 — pd.isna(x) : None, NaN, NaT… (mais pas "", à vérifier à part)

from defis.doublons import (  # noqa: F401
    CLES,
    SEUIL,
    VERITE_EXEMPLE,
    cle_email,
    cle_nom_code_postal,
    cle_telephone,
    fiches_exemple,
    generer_fiches,
    normaliser_email,
    normaliser_nom,
    normaliser_telephone,
)

# ============================================================================ La distance d'édition


def distance_edition(a: str, b: str) -> int:
    """Défi 164 🟢 — Le nombre minimal de fautes de frappe pour passer de `a` à `b` (distance de Levenshtein).

    Une « faute », c'est insérer une lettre, en supprimer une, ou en remplacer une par une autre :

        >>> distance_edition("Dupont", "Dupond")          # un remplacement
        1
        >>> distance_edition("chat", "chats"), distance_edition("kitten", "sitting")
        (1, 3)
        >>> distance_edition("", "abc"), distance_edition("Hélène", "Helene")
        (3, 2)
        >>> distance_edition("ab", "ba")                  # une inversion compte DEUX remplacements
        2

    La méthode (la programmation dynamique) : un tableau `d` où `d[i][j]` est la distance entre les i
    premières lettres de `a` et les j premières de `b`.

    - `d[0][j] = j` (passer de "" à j lettres : j insertions) et `d[i][0] = i` (i suppressions) ;
    - sinon `d[i][j]` est le plus petit de :
      `d[i - 1][j] + 1` (supprimer a[i - 1]), `d[i][j - 1] + 1` (insérer b[j - 1]),
      `d[i - 1][j - 1] + (0 si a[i - 1] == b[j - 1] sinon 1)` (remplacer, gratuit si c'est la même lettre) ;
    - la réponse est `d[len(a)][len(b)]`.

    Les règles :

    - les lettres sont comparées telles quelles : « é » et « e », « D » et « d » sont différentes (on
      normalise AVANT, avec `normaliser_nom`, quand on le veut) ;
    - `a` ou `b` qui n'est pas un texte (None, un nombre, NaN) → TypeError : une case vide n'est pas "" ;
    - deux textes de 1 500 lettres (un commentaire, une adresse recopiée) doivent marcher : pas de
      RecursionError, et pas une récursion sans mémoire, qui ferait 3^n appels.

    Ce que vous apprenez : la programmation dynamique (un tableau rempli dans l'ordre, chaque case calculée
    une fois à partir de cases déjà connues), et pourquoi la récursion « évidente » ne passe pas l'échelle.

    Indice : `precedente = list(range(len(b) + 1))` (la ligne 0), puis pour chaque lettre de a, une
    `courante = [i]` remplie de gauche à droite avec `min(...)`. Deux lignes suffisent.
    """
    raise NotImplementedError


# ============================================================================ Regrouper (union-find)


def regrouper(identifiants, paires) -> list[list]:
    """Défi 165 🟢 — Les groupes de fiches d'un même client, à partir des paires « même client ».

    F01 = F02 et F01 = F03, donc F01, F02 et F03 sont UN client, même si personne n'a comparé F02 à F03 :

        >>> regrouper(["F01", "F02", "F03", "F04"], [("F02", "F01"), ("F01", "F03")])
        [['F01', 'F02', 'F03'], ['F04']]
        >>> regrouper(["C", "B", "A", "D", "E"], [("A", "B"), ("D", "C"), ("B", "D")])
        [['A', 'B', 'C', 'D'], ['E']]

    Les règles :

    - `identifiants` : toutes les fiches (une liste, les clés d'un dictionnaire, un générateur… lus une
      seule fois). Une fiche qu'aucune paire ne cite est un client à elle seule : `['F04']` ;
    - `paires` : des couples `(a, b)` (une liste ou un générateur, lus une fois), dans n'importe quel ordre ;
      `(a, a)` ne change rien ;
    - le résultat : une liste de listes, chaque groupe trié, et les groupes triés (donc par leur premier
      élément) : le même résultat quel que soit l'ordre des paires ;
    - un identifiant présent deux fois dans `identifiants` → ValueError (une clé cassée n'est pas un
      doublon à fusionner) ; une paire qui cite une fiche inconnue → ValueError qui la nomme ;
    - 200 000 fiches reliées en une seule chaîne (F1 = F2, F2 = F3, …), dans un sens ou dans l'autre :
      quelques dixièmes de seconde, et pas de RecursionError.

    La méthode (union-find) : un dictionnaire `parent` où chaque fiche pointe vers une autre fiche de son
    groupe ; la « racine » d'un groupe pointe vers elle-même. Pour savoir si deux fiches sont ensemble, on
    compare leurs racines ; pour réunir deux groupes, la racine de l'un pointe vers celle de l'autre.

    Ce que vous apprenez : une structure de données qui rend la transitivité presque gratuite, la
    complexité qui se cache dans « renuméroter tout le groupe », la pile de Python (environ 1 000 appels).

    Indice : `parent = {i: i for i in identifiants}` (en vérifiant les doublons) ; une fonction
    `racine(x)` avec une boucle `while parent[x] != x` (pas de récursion) ; pour une paire,
    `parent[racine(b)] = racine(a)`. Pour que les chemins restent courts : le plus PETIT groupe rejoint le
    plus grand (un dictionnaire `taille`), et/ou chaque fiche traversée pointe ensuite directement sur la
    racine (la compression de chemin). À la fin : un dictionnaire racine → liste de fiches.
    """
    raise NotImplementedError


# ============================================================================ Le blocage


def paires_candidates(fiches: dict, cles=CLES, taille_max: int = 50) -> tuple[set, list]:
    """Défi 166 🟡 — Les seules paires qui valent la peine d'être comparées, et les blocs trop gros.

    300 000 fiches font 45 milliards de paires : on ne les comparera jamais toutes. Une CLÉ DE BLOCAGE
    est une fonction qui donne, pour une fiche, une valeur (l'e-mail normalisé, le téléphone…) ; deux
    fiches qui reçoivent la même valeur d'une MÊME clé forment un bloc, et seules les paires d'un même bloc
    sont proposées.

        >>> paires, trop_gros = paires_candidates(fiches_exemple())
        >>> len(paires), ("F01", "F02") in paires, ("F02", "F01") in paires, trop_gros
        (11, True, False, [])
        >>> paires, trop_gros = paires_candidates(fiches_exemple(), taille_max=2)
        >>> len(paires), trop_gros
        (8, [(2, 'dupre helene|75011', 3), (2, 'marie petit|75015', 3)])

    Les règles :

    - `fiches` : un dictionnaire `{identifiant: fiche}` ; `cles` : une liste de fonctions `cle(fiche)` ;
    - chaque clé est appelée UNE fois par fiche (n appels par clé, pas un par paire) ;
    - une clé qui renvoie None ou "" (la fiche n'a pas d'e-mail…) ne met la fiche dans AUCUN bloc ;
    - deux fiches ne sont candidates que si UNE MÊME clé leur donne la même valeur : « 75011 » pour la
      première clé et « 75011 » pour la deuxième, ce n'est pas un bloc ;
    - un bloc de plus de `taille_max` fiches (le numéro du centre d'appels saisi pour 400 clients) ne
      donne AUCUNE paire, mais il est signalé : `(position de la clé dans cles, valeur, nombre de fiches)` ;
    - le résultat : `(paires, trop_gros)`. `paires` est un `set` de tuples `(a, b)` avec `a < b` (chaque
      paire une seule fois, même si deux clés la proposent) ; `trop_gros` est une liste triée du plus gros
      bloc au plus petit, puis par position de clé, puis par valeur ;
    - `taille_max` qui n'est pas un entier d'au moins 2 → ValueError.

    Ce que vous apprenez : regrouper avec un dictionnaire de listes plutôt que comparer chaque fiche à chaque
    autre (n × k au lieu de n²), la valeur absente qui fabrique un bloc géant, ne rien jeter en silence.

    Indice : pour CHAQUE clé, `blocs = {}` puis `blocs.setdefault(valeur, []).append(ident)` ; ensuite
    `itertools.combinations(sorted(membres), 2)` donne les paires d'un bloc, déjà dans l'ordre (petit, grand).
    """
    raise NotImplementedError


# ============================================================================ Le score de correspondance


def score_correspondance(a: dict, b: dict) -> int:
    """Défi 167 🟡 — Deux fiches sont-elles la même personne ? Un score : à partir de SEUIL (5), oui.

    Chaque champ apporte des points quand les deux fiches sont d'accord, en retire quand elles se
    contredisent, et ne compte PAS quand l'une des deux ne le connaît pas :

        champ           les deux le connaissent et…             accord   désaccord
        email           normaliser_email identiques               +4        -1
        telephone       normaliser_telephone identiques           +3        -1
        naissance       textes identiques (sans espaces autour)   +3        -8
        nom             mots normalisés TRIÉS identiques           +3
                        sinon, distance_edition(…) <= 2           +1
                        sinon                                               -4
        code_postal     textes identiques (sans espaces autour)   +1         0

        >>> f = fiches_exemple()
        >>> score_correspondance(f["F01"], f["F02"])     # téléphone +3, nom à l'envers +3, code postal +1
        7
        >>> score_correspondance(f["F04"], f["F05"])     # e-mail +4, téléphone +3, naissance -8, nom -4, cp +1
        -4
        >>> score_correspondance(f["F09"], f["F11"])     # la naissance de F11 est un NaN : elle ne compte pas
        7
        >>> score_correspondance({"email": None}, {"email": None}), score_correspondance({}, {"nom": "Léa"})
        (0, 0)

    Les règles :

    - « connaître » un champ : la clé existe dans la fiche ET la valeur n'est ni None, ni NaN, ni un texte
      vide ou fait d'espaces. Pour l'e-mail, le téléphone et le nom, c'est « la normalisation ne donne pas
      "" » : `normaliser_telephone("06 12")` vaut "" (un numéro illisible ne contredit rien) ;
    - le nom : `" ".join(sorted(normaliser_nom(nom).split()))`, pour que « DUPRE Helene » et « Hélène
      Dupré » soient le même nom ; la distance d'édition (défi 164) se calcule sur ces mots triés ;
    - naissance et code postal : `str(valeur).strip()`, comparés tels quels ;
    - le score est le même dans les deux sens (`score(a, b) == score(b, a)`), et les fiches ne sont pas
      modifiées ; le résultat est un `int`.

    Ce que vous apprenez : trois cas et pas deux (accord, désaccord, absence), le NaN qui passe un `if`,
    normaliser avant de comparer, des poids qui disent qu'une contradiction pèse plus qu'un accord.

    Indice : pour chaque champ, `va, vb = normaliser_xxx(a.get("xxx")), normaliser_xxx(b.get("xxx"))` puis
    `if va and vb: score += 4 if va == vb else -1`. Pour naissance et code postal, écrivez une petite
    fonction `_texte(v)` qui renvoie "" si `v is None` ou si `pd.isna(v)` (à n'appeler que si v n'est pas
    un texte), sinon `str(v).strip()`.
    """
    raise NotImplementedError


# ============================================================================ La fiche de référence


def fiche_de_reference(fiches) -> dict:
    """Défi 168 🟡 — Une seule fiche par client, construite à partir de toutes ses fiches.

    Hélène a trois fiches (F01, F02, F03). Sa fiche de référence prend, champ par champ, la valeur la plus
    RÉCENTE qui soit connue, sauf pour trois champs qui ont leur propre règle :

        >>> f = fiches_exemple()
        >>> helene = fiche_de_reference([{"id": i, **f[i]} for i in ("F01", "F02", "F03")])
        >>> helene["id"], helene["anciens_ids"], helene["maj"]
        ('F01', ['F01', 'F02', 'F03'], datetime.date(2024, 3, 2))
        >>> helene["inscription"], helene["resilie"], helene["optin_sms"]       # 2016 (F02), pas 2021 (F03)
        (datetime.date(2016, 9, 14), False, False)
        >>> helene["telephone"], helene["email"], helene["naissance"]          # F03 n'a pas de téléphone
        ('06 12 34 56 78', ' Helene.Dupre@Exemple.fr', '1984-03-12')

    Les règles :

    - `fiches` : une liste (ou un itérable) de dictionnaires, les fiches d'UN client. Chacune a un "id"
      et une date "maj" (sa dernière mise à jour, un `datetime.date`) ; sinon, ou si la liste est vide →
      ValueError ;
    - "id" : le plus petit identifiant ; "anciens_ids" : tous les identifiants, triés ;
    - "maj" : la date la plus récente ;
    - "inscription" : la PLUS ANCIENNE date d'inscription connue (l'ancienneté compte depuis le premier
      contact), None si aucune fiche ne la connaît ;
    - "resilie" : True seulement si TOUTES les fiches sont résiliées (True ou 1 ; une fiche où l'information
      manque n'est pas résiliée), un bool : quitter la boutique en restant cliente sur le site n'est pas partir ;
    - tout autre champ (présent dans au moins une fiche) : la valeur de la fiche la plus récemment mise à
      jour où ce champ est CONNU (ni absent, ni None, ni NaN, ni un texte vide ou d'espaces), recopiée
      telle quelle ; à date égale, la fiche qui vient en DERNIER dans la liste (importée après) ; None si
      aucune fiche ne le connaît ;
    - False et 0 sont des valeurs CONNUES : `optin_sms = False` veut dire « a refusé les SMS » ;
    - les fiches reçues ne sont pas modifiées : le résultat est un NOUVEAU dictionnaire.

    Ce que vous apprenez : des règles de « survie » champ par champ (pas « la fiche la plus récente
    gagne tout »), le tri stable et ses ex aequo, False n'est pas None, les conséquences d'une fusion sur
    le modèle (ancienneté, faux churn) et sur la loi (un refus de SMS).

    Indice : `ordre = sorted(range(len(fiches)), key=lambda i: (fiches[i]["maj"], i))` range les positions
    de la plus ancienne à la plus récente (ex aequo : l'ordre de la liste) ; parcourez `reversed(ordre)`
    et gardez la première valeur connue. Attention à `sorted(..., reverse=True)` : les ex aequo y gardent
    leur ordre d'origine.
    """
    raise NotImplementedError


# ============================================================================ La qualité d'un dédoublonnage


def qualite_dedoublonnage(predits, verite) -> dict:
    """Défi 169 🔴 — Votre dédoublonnage est-il bon ? Précision et rappel, comptés en PAIRES de fiches.

    Une paire de fiches est « prédite » si vos groupes les mettent ensemble, « vraie » si ce sont
    vraiment la même personne, « juste » si les deux à la fois :

        >>> verite = [["A", "B", "C"], ["D"]]
        >>> q = qualite_dedoublonnage([["A", "B"], ["C"], ["D"]], verite)
        >>> q["paires_predites"], q["paires_vraies"], q["paires_justes"]
        (1, 3, 1)
        >>> q["precision"], round(q["rappel"], 3), q["f1"]
        (1.0, 0.333, 0.5)
        >>> sorted(q)
        ['f1', 'paires_justes', 'paires_predites', 'paires_vraies', 'precision', 'rappel']
        >>> tout = qualite_dedoublonnage([["A", "B", "C", "D"]], verite)    # tout fusionner : rappel parfait…
        >>> tout["paires_predites"], tout["precision"], tout["rappel"]
        (6, 0.5, 1.0)

    Les règles :

    - `predits`, `verite` : deux regroupements des MÊMES fiches, chacun une liste de groupes (listes, tuples
      ou sets d'identifiants), comme la sortie du défi 165 ;
    - un groupe de k fiches contient k(k - 1)/2 paires (un groupe d'une fiche : 0) ;
    - "paires_predites", "paires_vraies", "paires_justes" : trois `int` ;
    - "precision" = justes / prédites (1.0 s'il n'y a aucune paire prédite : aucune fusion n'est fausse),
      "rappel" = justes / vraies (1.0 s'il n'y a aucune vraie paire), "f1" = 2 × P × R / (P + R) (la
      moyenne HARMONIQUE ; 0.0 si P + R = 0) : trois `float` ;
    - un identifiant dans deux groupes (ou deux fois dans un groupe) → ValueError ; une fiche présente d'un
      côté et pas de l'autre → ValueError ;
    - trois groupes de 20 000 fiches doivent marcher : ils contiennent 600 millions de paires, qu'on ne
      peut PAS fabriquer une par une.

    Ce que vous apprenez : la précision et le rappel sur autre chose qu'un classifieur, compter des paires
    sans les énumérer (le tableau croisé), lire les deux erreurs (tout fusionner, ne rien fusionner).

    Indice : un dictionnaire `{fiche: numéro de son groupe}` de chaque côté. Les paires justes sont les
    paires de fiches qui ont le MÊME couple (groupe prédit, vrai groupe) : `Counter((p[f], v[f]) for f in
    p)`, puis k(k - 1)/2 pour chaque compte. Utilisez `//` pour rester en entiers.
    """
    raise NotImplementedError

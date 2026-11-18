"""Série 25 — Écrire un petit langage : les règles de ciblage du marketing, du texte à la requête SQL.

🗣️ VOUS VOULEZ DÉVELOPPER ? UN JOUR, ON VOUS DEMANDERA : « ON PEUT ÉCRIRE NOS RÈGLES NOUS-MÊMES ? »

Le service marketing veut choisir lui-même les clients à appeler, sans attendre un développeur :

    contrat = "mensuel" et anciennete < 6 et (appels_support >= 3 ou facture > 80)

La mauvaise idée, c'est `eval()` (n'importe qui exécute n'importe quoi sur le serveur) ou une suite de
`split(" et ")` qui casse au premier texte contenant « et ». La bonne, c'est ce que font Python, SQL,
Excel, les filtres de Jira ou de Gmail : DÉCOUPER le texte en jetons, en construire un ARBRE, puis
travailler sur l'arbre. Cette série vous fait écrire toute la chaîne :

    152 decouper   le texte → des jetons (l'« analyseur lexical »)
    153 analyser   les jetons → un arbre, avec la priorité de « et » sur « ou », et des erreurs qui
                   montrent l'endroit exact (l'« analyseur syntaxique », par descente récursive)
    154 evaluer    l'arbre + un client → vrai, faux… ou INCONNU (l'interpréteur, la logique de SQL)
    155 vers_texte l'arbre → le texte, avec juste les parenthèses nécessaires (l'aller-retour)
    156 verifier_regle  toutes les erreurs de colonnes et de types AVANT de lancer la campagne
    157 vers_sql   l'arbre → une requête SQL paramétrée, sans injection possible (le compilateur)

Prérequis : la série 00, les dictionnaires et les tuples. La série 10 (NULL en SQL) et la série 13
(injection SQL) éclairent les défis 154 et 157. Pas de pandas, pas de `re` obligatoire.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un JETON est un morceau qui a un sens : `anciennete`, `<`, `6`. On les lit de gauche à droite avec UN
  indice `i` qui avance ; on ne raccourcit jamais le texte (`texte = texte[1:]` recopie tout le reste à
  chaque jeton : 100 000 caractères, 5 milliards de copies) ;
- on prend toujours le jeton le plus LONG possible : `<=` est un opérateur, pas `<` puis `=` ; `etat` est
  un nom, pas le mot-clé `et` suivi de `at` ;
- l'ARBRE ne garde que la structure (les parenthèses ont disparu, elles ont servi à la construire) ;
- la PRIORITÉ : `non` s'applique à la condition qui le suit, `et` passe avant `ou` (comme × avant + :
  `a ou b et c` se lit `a ou (b et c)`). Une fonction par niveau de priorité, chacune appelant la suivante :
  c'est la « descente récursive », la méthode de la plupart des vrais langages ;
- la LOGIQUE À TROIS VALEURS : un client dont on ne connaît pas la région n'est ni « dans le Nord » ni
  « hors du Nord » : la condition vaut INCONNU (None). C'est le NULL de SQL (série 10) : faux ET inconnu
  donne faux, vrai OU inconnu donne vrai, NON inconnu reste inconnu.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_25 import *
    >>> regles_exemple()["nouveaux_mensuels"]
    'contrat = "mensuel" et anciennete < 6'
    >>> clients_exemple()[0]["contrat"]
    'mensuel'
    >>> erreur = ErreurSyntaxe("caractère inattendu '#'", 9)
    >>> print(souligner("contrat =# 3", erreur))
    contrat =# 3
             ^ caractère inattendu '#'
    >>> isinstance(erreur, ValueError)
    True

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites 152 avant 153 (votre `analyser`
   se sert de votre `decouper`). Les défis 154 à 157 reçoivent un arbre tout fait : vous pouvez les faire
   sans avoir réussi 153 (mais 155 et 156 se testent plus facilement avec).
2. Lancez `python scripts/defis.py --defi 152` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_25.py`.

Les outils communs (`Jeton`, `ErreurSyntaxe`, `souligner`, `MOTS_CLES`, `BOOLEENS`, `OPERATEURS`,
`clients_exemple`, `schema_clients`, `regles_exemple`, `generer_clients`, `generer_arbre`, `base_clients`,
`afficher_arbre`) sont dans `defis/langage.py`, déjà importés ci-dessous. `eval`, `exec` et `ast` sont
INTERDITS dans vos réponses (c'est précisément ce qu'on ne veut pas faire avec un texte venu d'ailleurs).

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import difflib  # noqa: F401 — difflib.get_close_matches("ancienete", ["anciennete", "facture"], n=1)
import math  # noqa: F401 — math.isnan, math.isfinite
import numbers  # noqa: F401 — isinstance(x, numbers.Real) : int, float, et les nombres de numpy

from defis.langage import (  # noqa: F401
    BOOLEENS,
    MOTS_CLES,
    OPERATEURS,
    ErreurSyntaxe,
    Jeton,
    afficher_arbre,
    base_clients,
    clients_exemple,
    generer_arbre,
    generer_clients,
    regles_exemple,
    schema_clients,
    souligner,
)

# ============================================================================ Les jetons


def decouper(texte: str) -> list[Jeton]:
    """Défi 152 🟢 — Découper une règle en jetons (l'analyseur lexical).

    Renvoie la liste des jetons de `texte`, de gauche à droite, chacun avec la POSITION (l'indice, à
    partir de 0) de son premier caractère, et termine TOUJOURS par `Jeton("FIN", None, len(texte))`.

        >>> decouper("anciennete <= 6")
        [Jeton(type='NOM', valeur='anciennete', position=0), Jeton(type='OP', valeur='<=', position=11), \
Jeton(type='NOMBRE', valeur=6, position=14), Jeton(type='FIN', valeur=None, position=15)]
        >>> [j.type for j in decouper('region dans ("Nord", "Est") ET fibre = vrai')]
        ['NOM', 'DANS', '(', 'TEXTE', ',', 'TEXTE', ')', 'ET', 'NOM', 'OP', 'BOOLEEN', 'FIN']

    Les règles :

    - les espaces (et tout ce que `str.isspace()` reconnaît : tabulation, retour à la ligne, espace
      insécable) séparent les jetons et ne produisent rien ;
    - `(`, `)` et `,` : un jeton dont le type et la valeur sont le caractère lui-même ;
    - les opérateurs `=`, `!=`, `<`, `<=`, `>`, `>=` : type "OP". Le plus LONG d'abord (`<=` est UN jeton).
      Les habitués d'autres langages écrivent `==` ou `<>` : acceptés, avec la valeur `"="` ou `"!="` ;
    - un NOMBRE : des chiffres 0 à 9, éventuellement un point suivi de chiffres, éventuellement un `-`
      collé devant (`-3`). Valeur `int` sans point (`6` → 6, pas 6.0), `float` avec (`79.5` → 79.5) ;
    - un TEXTE : entre guillemets droits, doubles ou simples (`"mensuel"`, `'mensuel'`). La valeur est
      le texte SANS les guillemets. Une barre oblique inverse protège le caractère suivant : `"a\\"b"`
      donne `a"b`, `"a\\\\b"` donne `a\\b`. Un texte jamais fermé lève `ErreurSyntaxe` à la position du
      guillemet OUVRANT (c'est lui qu'il faut aller voir), avec le mot « guillemet » dans le message ;
    - un MOT : une lettre (`str.isalpha()`, accents compris) ou `_`, puis des lettres, chiffres ou `_`.
      En minuscules, `et`, `ou`, `non`, `dans` sont des mots-clés (types "ET", "OU", "NON", "DANS", valeur
      le mot en minuscules : `ET` et `Et` aussi), `vrai` et `faux` des "BOOLEEN" (valeur True, False) ;
      tout autre mot est un "NOM", gardé tel qu'écrit ;
    - les guillemets typographiques (“ ” « » ‘ ’ „), qui arrivent avec un copier-coller depuis Word,
      lèvent `ErreurSyntaxe` avec le mot « guillemet » dans le message ;
    - tout autre caractère (`#`, `!` seul, `.` seul, `+`…) lève `ErreurSyntaxe` à SA position ;
    - un `texte` qui n'est pas un `str` lève `TypeError` ;
    - 200 000 caractères en moins d'une seconde.

    Ce que vous apprenez : lire un texte caractère par caractère avec un indice, la règle du « plus long
    jeton », distinguer mots-clés et noms, signaler une erreur À L'ENDROIT où elle est.

    Indice : `i = 0`, puis `while i < len(texte):` et un `if / elif` selon le caractère `c = texte[i]`.
    Pour un mot : `j = i + 1`, `while j < len(texte) and (texte[j].isalnum() or texte[j] == "_"): j += 1`,
    le mot est `texte[i:j]`, puis `i = j`. Pour un opérateur : regardez `texte[i:i + 2]` d'abord.
    """
    raise NotImplementedError


# ============================================================================ L'arbre


def analyser(texte: str) -> tuple:
    """Défi 153 🔴 — Construire l'arbre d'une règle (l'analyseur syntaxique, par descente récursive).

    Renvoie l'arbre décrit dans `defis/langage.py` : `(op, colonne, valeur)`, `("dans", colonne, (v1, …))`,
    `("non", x)`, `("et", x, y, …)`, `("ou", x, y, …)`. Utilisez VOTRE `decouper`.

        >>> analyser('contrat = "mensuel" et anciennete < 6')
        ('et', ('=', 'contrat', 'mensuel'), ('<', 'anciennete', 6))
        >>> analyser('a = 1 ou b = 2 et c = 3')          # « et » passe avant « ou »
        ('ou', ('=', 'a', 1), ('et', ('=', 'b', 2), ('=', 'c', 3)))
        >>> analyser('non a = 1 et b = 2')                # « non » ne prend que la condition qui suit
        ('et', ('non', ('=', 'a', 1)), ('=', 'b', 2))
        >>> analyser('a = 1 et b = 2 et c = 3')           # trois conditions, UN seul « et »
        ('et', ('=', 'a', 1), ('=', 'b', 2), ('=', 'c', 3))
        >>> analyser('a = 1 et (b = 2 et c = 3)')         # les parenthèses ÉCRITES restent visibles
        ('et', ('=', 'a', 1), ('et', ('=', 'b', 2), ('=', 'c', 3)))

    La grammaire (un niveau de priorité par ligne, du plus faible au plus fort) :

        regle       := ou FIN
        ou          := et   ( "ou" et )*        → ("ou", …) s'il y en a au moins deux, sinon l'unique enfant
        et          := non  ( "et" non )*       → pareil avec "et"
        non         := "non" non  |  primaire   → ("non", …)
        primaire    := "(" ou ")"  |  NOM OP valeur  |  NOM "dans" "(" valeur ( "," valeur )* ")"
        valeur      := NOMBRE | TEXTE | BOOLEEN

    Les erreurs : toujours `ErreurSyntaxe(message, position)`, à la position du PREMIER jeton qui ne va
    pas (la fin du texte compte comme un jeton, à la position `len(texte)`), avec ces exceptions et ces
    mots dans le message :

    - règle vide (ou que des espaces) : position `len(texte)`, le mot « vide » ;
    - parenthèse ouverte et jamais fermée (on arrive à la fin) : position de CETTE parenthèse ouvrante,
      le mot « parenthèse » (Python fait pareil depuis la version 3.10 : « '(' was never closed ») ;
    - `contrat = mensuel` (un NOM là où il faut une valeur) : le mot « guillemet » (on a oublié de mettre
      le texte entre guillemets) ;
    - `6 > anciennete` (une valeur là où il faut une colonne) : le mot « colonne » ;
    - quelque chose EN TROP après une règle complète (`a = 1 b = 2`, `a = 1)`) : c'est une erreur, pas
      un détail à ignorer — sinon la moitié de la règle disparaîtrait en silence ;
    - plus de 100 niveaux imbriqués (chaque `(` ouverte et chaque `non` en cours compte pour un niveau) :
      le mot « imbriqu », à la position du jeton qui dépasse. Une règle tapée par un humain n'en a jamais
      autant ; un texte malveillant si, et il ne doit pas faire planter le serveur (RecursionError) ;
    - les erreurs de `decouper` (ErreurSyntaxe, TypeError) passent telles quelles.

    En revanche, 5 000 conditions reliées par « ou » à la suite ne sont PAS imbriquées : elles doivent
    marcher (une boucle `while`, pas un appel récursif par « ou »).

    Ce que vous apprenez : la descente récursive, la priorité et l'associativité, regarder le jeton
    suivant sans le consommer, des messages d'erreur qui aident, borner une récursion sur une entrée
    venue de l'extérieur.

    Indice : une liste `jetons = decouper(texte)` et un indice `pos` ; deux petites fonctions internes
    `voir()` (le jeton courant) et `avancer()` (le renvoie et passe au suivant), avec `nonlocal pos`.
    Puis une fonction par ligne de la grammaire : `ou()` appelle `et()` dans une boucle, `et()` appelle
    `non()`, `non()` s'appelle elle-même ou appelle `primaire()`, et `primaire()` rappelle `ou()` pour
    l'intérieur d'une parenthèse. À la fin, vérifiez que `voir().type == "FIN"`.
    """
    raise NotImplementedError


# ============================================================================ L'interpréteur


def evaluer(arbre: tuple, client) -> bool | None:
    """Défi 154 🟡 — Ce client est-il ciblé ? Vrai, faux… ou INCONNU (la logique à trois valeurs de SQL).

    Renvoie True, False ou None (inconnu) pour `client`, un dictionnaire {colonne: valeur}.

        >>> client = clients_exemple()[3]                        # le client 4 : région inconnue (None)
        >>> evaluer(("=", "region", "Nord"), client) is None
        True
        >>> evaluer(("non", ("=", "region", "Nord")), client) is None   # NON inconnu = inconnu
        True
        >>> evaluer(("et", ("=", "contrat", "annuel"), ("=", "region", "Nord")), client)   # faux ET inconnu
        False
        >>> evaluer(("ou", ("=", "contrat", "mensuel"), ("=", "region", "Nord")), client)  # vrai OU inconnu
        True

    Les règles :

    - une valeur INCONNUE chez le client, `None` ou NaN (une case vide lue par pandas : `float("nan")`,
      `numpy.nan`), rend la comparaison inconnue : None, quel que soit l'opérateur (même `!=` : NaN != 6
      vaut True en Python, mais on ne sait PAS si la facture est différente de 6) ;
    - une colonne ABSENTE du client lève `KeyError` (c'est une faute de frappe dans la règle, pas une
      donnée manquante : avec `client.get(colonne)`, la règle ne ciblerait plus personne, en silence) ;
    - `("dans", colonne, valeurs)` vaut True si la valeur du client est égale à l'une des valeurs ;
    - "et" : False si au moins une condition est False ; sinon None si au moins une est None ; sinon
      True. "ou" : True si au moins une est True ; sinon None si au moins une est None ; sinon False.
      "non" : None reste None, sinon le contraire ;
    - COURT-CIRCUIT, comme `and` et `or` en Python : "et" évalue ses conditions dans l'ordre et s'arrête
      à la première False (les suivantes ne sont pas lues du tout) ; "ou" s'arrête à la première True ;
    - on ne compare que deux valeurs du même GENRE : deux nombres (int ou float, numpy compris), deux
      textes, ou deux booléens. Sinon `TypeError` avec le nom de la colonne. Attention : pour Python,
      True est un nombre (`True == 1`), et `"6" == 6` vaut False sans erreur. Sur deux booléens, seuls
      `=` et `!=` ont un sens (TypeError pour `<`…) ;
    - le résultat est exactement True, False ou None : avec des valeurs numpy (un client lu dans un
      DataFrame), `np.float64(3) < 6` donne `np.True_`, qui n'est pas `True` (`is True` vaut False) ;
    - un nœud inconnu (`("xor", …)`) lève `ValueError`.

    Ce que vous apprenez : parcourir un arbre récursivement (un interpréteur), la logique à trois valeurs
    de SQL, le court-circuit, NaN et les booléens qui se font passer pour des nombres.

    Indice : `noeud = arbre[0]`, puis un cas par sorte de nœud. Pour "et" : `resultat = True`, puis pour
    chaque enfant `r = evaluer(enfant, client)` ; `if r is False: return False` ; `if r is None: resultat
    = None`. Écrivez deux petites fonctions : `_inconnu(v)` (None ou NaN) et `_genre(v)` ("booleen" testé
    AVANT "nombre", puisque `isinstance(True, int)` vaut True).
    """
    raise NotImplementedError


# ============================================================================ L'aller-retour


def vers_texte(arbre: tuple) -> str:
    """Défi 155 🟡 — Réécrire l'arbre en texte, avec seulement les parenthèses nécessaires.

    Le tableau de bord affiche la règle « propre » au marketing, et on l'enregistre sous cette forme. La
    propriété à garantir : `analyser(vers_texte(arbre)) == arbre` pour TOUT arbre valide (l'aller-retour).

        >>> vers_texte(("ou", ("=", "a", 1), ("et", ("=", "b", 2), ("=", "c", 3))))
        'a = 1 ou b = 2 et c = 3'
        >>> vers_texte(("et", ("=", "a", 1), ("ou", ("=", "b", 2), ("=", "c", 3))))
        'a = 1 et (b = 2 ou c = 3)'
        >>> vers_texte(("et", ("=", "a", 1), ("et", ("=", "b", 2), ("=", "c", 3))))
        'a = 1 et (b = 2 et c = 3)'
        >>> vers_texte(("non", ("dans", "region", ("Nord", 'Le "Sud"'))))
        'non region dans ("Nord", "Le \\\\"Sud\\\\"")'

    Le format (le correcteur compare au caractère près) :

    - une comparaison : `colonne op valeur`, une espace de chaque côté de l'opérateur ;
    - "dans" : `colonne dans (v1, v2)`, les valeurs séparées par une virgule et une espace ;
    - "et" / "ou" : les conditions séparées par ` et ` / ` ou ` ; "non" : `non ` puis la condition ;
    - des parenthèses SEULEMENT là où l'arbre l'exige : un "ou" dans un "et" ; un "et" dans un "et" (ou
      un "ou" dans un "ou" : relu sans parenthèses, il ne ferait plus qu'un seul nœud) ; un "et" ou un
      "ou" sous un "non". Jamais autour de toute la règle, jamais autour d'une comparaison ;
    - les valeurs : True et False s'écrivent `vrai` et `faux` (pas `True`, que l'analyseur lirait comme
      un nom de colonne, ni `1`) ; un int avec `str` ; un float avec `repr` (`0.1 + 0.2` s'écrit
      `0.30000000000000004` : repr donne le texte le plus court qui redonne EXACTEMENT le même float,
      `round` ou `:.2f` changeraient la règle) ; un texte entre guillemets doubles, en mettant une barre
      oblique inverse devant chaque `\\` et chaque `"` (et rien d'autre) ;
    - ce qui ne se relirait pas lève `ValueError` : un float infini ou NaN, un nom de colonne qui n'est
      pas un mot (`"date début"`, `""`, `"2024"`) ou qui est un mot-clé (`"et"`, `"Vrai"`), un "et"/"ou"
      à moins de deux conditions, un "dans" sans valeur, un nœud inconnu.

    Le correcteur n'utilise que des floats que repr écrit sans exposant (pas de `1e-07`).

    Ce que vous apprenez : la priorité vue de l'autre côté (quand faut-il des parenthèses ?), échapper un
    texte, l'aller-retour comme test (un « test par propriété », série 11), `repr` des floats.

    Indice : une fonction récursive. Pour un "et"/"ou", écrivez chaque enfant, et entourez-le de
    parenthèses si c'est un "et" ou un "ou" dont la priorité est inférieure OU ÉGALE à celle du parent
    (`{"ou": 1, "et": 2}`). Pour échapper : `.replace("\\\\", "\\\\\\\\")` D'ABORD, puis les guillemets.
    """
    raise NotImplementedError


# ============================================================================ Le vérificateur de types


def verifier_regle(arbre: tuple, schema: dict) -> list[str]:
    """Défi 156 🟡 — Trouver TOUTES les erreurs d'une règle avant de lancer la campagne.

    `schema` donne le genre de chaque colonne : "nombre", "texte" ou "booleen" (voir `schema_clients()`).
    Renvoie la liste des problèmes, un message par condition fautive, dans l'ordre de lecture de la
    règle (de gauche à droite) ; une liste vide si tout va bien. On n'évalue rien : on regarde l'arbre.

        >>> verifier_regle(("<", "anciennete", 6), schema_clients())
        []
        >>> for probleme in verifier_regle(("et", ("<", "ancienete", 6), ("=", "facture", "80"),
        ...                                       (">", "fibre", False)), schema_clients()):
        ...     print(probleme)
        colonne « ancienete » inconnue : vouliez-vous dire « anciennete » ?
        « facture » est un nombre, mais elle est comparée à "80", qui est un texte
        « fibre » est un booléen : « > » n'a pas de sens (utilisez = ou !=)

    Les trois problèmes à trouver, et ce que le message doit contenir (majuscules et accents ignorés) :

    - une colonne absente du schéma : son nom et « inconnue ». Si une colonne du schéma ne diffère que
      par les majuscules (`Anciennete`), proposez-la ; sinon la plus proche selon
      `difflib.get_close_matches(colonne, list(schema), n=1)`, s'il y en a une : « vouliez-vous dire »
      et son nom. S'il n'y en a aucune, pas de « vouliez-vous dire » (une suggestion absurde égare) ;
    - une valeur d'un autre genre que la colonne (pour "dans", la première valeur fautive suffit) : le
      nom de la colonne, son genre (« nombre », « texte » ou « booléen ») et celui de la valeur. Rappel :
      `True` n'est pas un nombre ici, et `6.5` comme `6` sont des nombres ;
    - un `<`, `<=`, `>` ou `>=` sur une colonne booléenne : son nom, « booléen » et l'opérateur.

    Un seul message par condition, même si elle a plusieurs défauts : la colonne inconnue d'abord (on ne
    regarde pas plus loin), sinon le genre de la valeur, sinon l'opérateur. Le schéma et l'arbre ne sont
    pas modifiés ; un nœud inconnu lève `ValueError`.

    Ce que vous apprenez : vérifier AVANT d'exécuter (c'est ce que font mypy et les compilateurs), donner
    TOUTES les erreurs d'un coup plutôt que la première (série 14), des suggestions qui font gagner du
    temps (`difflib`, la bibliothèque standard les fait pour vous).

    Indice : un parcours de l'arbre (récursif, ou avec une pile) qui ajoute à une liste `problemes` au
    lieu de lever une exception. Pour le genre d'une valeur, réutilisez votre `_genre` du défi 154.
    """
    raise NotImplementedError


# ============================================================================ Le compilateur


def vers_sql(arbre: tuple) -> tuple[str, list]:
    """Défi 157 🔴 — Traduire la règle en requête SQL paramétrée (le compilateur).

    Les clients sont dans une base SQL : plutôt que de les charger tous en Python pour les filtrer un
    par un, on fait filtrer la base. Renvoie `(sql, params)` : la condition à mettre après `WHERE`, avec
    un `?` à la place de chaque valeur, et la liste des valeurs dans l'ordre des `?`.

        >>> vers_sql(("et", ("=", "contrat", "mensuel"), ("dans", "region", ("Nord", "Est"))))
        ('("contrat" = ? AND "region" IN (?, ?))', ['mensuel', 'Nord', 'Est'])
        >>> sql, params = vers_sql(analyser(regles_exemple()["nouveaux_mensuels"]))  # avec votre analyser
        >>> base_clients(clients_exemple()).execute(f"SELECT id FROM clients WHERE {sql}", params).fetchall()
        [(1,), (4,), (6,)]

    Les règles :

    - AUCUNE valeur dans le texte SQL, jamais : `region = 'Nord'` construit avec une f-string, c'est
      l'injection SQL de la série 13 (une région `x' OR 1=1 --` cible toute la base) ;
    - les noms de colonnes, eux, ne peuvent pas être des `?` : écrivez-les entre guillemets DOUBLES, en
      doublant chaque `"` qu'ils contiennent (`prix "TTC"` → `"prix ""TTC""\"`). Sans guillemets, une
      colonne qui s'appelle `order` ou `date fin` donne une erreur de syntaxe SQL ;
    - `=` `<` `<=` `>` `>=` s'écrivent pareil en SQL, `!=` s'écrit `<>` (ou `!=`, que SQLite accepte) ;
      "dans" → `IN (?, ?, …)`, "et" → `AND`, "ou" → `OR`, "non" → `NOT` ;
    - la requête doit sélectionner EXACTEMENT les clients pour lesquels `evaluer` vaut True. Le correcteur
      le vérifie sur des centaines de règles et de clients, avec des NULL. Bonne nouvelle : la logique à
      trois valeurs du défi 154 est celle de SQL, il n'y a rien à ajouter… si les parenthèses sont justes
      (en SQL aussi, AND passe avant OR) ;
    - un "et"/"ou" à moins de deux conditions, un "dans" sans valeur, un nœud inconnu : `ValueError`.

    Ce que vous apprenez : compiler un arbre vers un autre langage, la séparation code / données (les
    requêtes paramétrées), citer un identifiant SQL, vérifier un compilateur en comparant ses résultats à
    ceux de l'interpréteur.

    Indice : une fonction récursive interne qui renvoie le texte SQL d'un nœud et ajoute ses valeurs à
    une liste `params` partagée (déclarée dans `vers_sql`). Le plus sûr : TOUJOURS des parenthèses autour
    d'un AND / OR et derrière NOT. `", ".join("?" * 3)` donne `'?, ?, ?'`.
    """
    raise NotImplementedError

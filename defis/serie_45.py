"""Série 45 — Lire ce que disent les clients : mots, négations, Bayes naïf, probabilités, mots révélateurs, en continu.

💬 VOUS VOULEZ DÉVELOPPER ? UN JOUR, VOTRE CODE LIRA DU TEXTE ÉCRIT PAR DES HUMAINS : DES AVIS, DES TICKETS, DES
COURRIELS, DES MESSAGES DE CHAT. ET UN HUMAIN N'ÉCRIT JAMAIS DEUX FOIS UN MOT DE LA MÊME FAÇON. ICI, VOUS ÉCRIVEZ CE QUE
FONT LES FILTRES ANTI-SPAM (LE BAYES NAÏF EST NÉ LÀ), LES OUTILS D'ANALYSE DES AVIS CLIENTS ET LES MOTEURS DE
CLASSEMENT DES TICKETS DE SUPPORT : DÉCOUPER, COMPTER, LISSER, ADDITIONNER DES LOGARITHMES, EXPLIQUER, APPRENDRE EN
CONTINU.

Chaque mois, l'enquête de satisfaction se termine par une question ouverte : « Avez-vous quelque chose à nous dire ? ».
Aujourd'hui, une seule règle exploite les réponses : un commentaire qui contient « résili » déclenche l'appel d'un
conseiller fidélisation. Le mois dernier, elle a signalé 180 clients ; 134 sont restés, et beaucoup avaient écrit
« Pas question de résilier ». Pendant ce temps, 176 clients sont partis sans que la règle les voie : ils avaient écrit
« J'ai demandé mon code RIO » ou « Je pars chez un concurrent ». Inès, chargée de la relation client, a les 4 000
réponses des six derniers mois, et elle sait qui est parti. Elle écrit les six outils de cette série :

    272 mots                 les mots d'un commentaire, écrits d'une seule façon (« RÉSILIE », « resilie », « résilie »)
    273 avec_negations       les mots qui suivent une négation, marqués : « je ne vais pas non_resilier »
    274 entrainer            le Bayes naïf : compter les mots de chaque classe, lisser, prendre les logarithmes
    275 probabilites         la probabilité que le client parte, sans que le produit tombe à 0.0
    276 mots_revelateurs     les mots qui distinguent le mieux ceux qui partent : de quoi parler en comité
    277 ClassifieurEnLigne   apprendre chaque soir, oublier un client qui le demande, prédire, sans tout recompter

Prérequis : la série 00, les `dict` et `collections.Counter`. La série 20 (le texte et l'Unicode) éclaire le défi 272,
la série 23 (le modèle sans scikit-learn) les défis 274 et 275, et la série 34 (les données personnelles) le défi 277.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un modèle ne lit pas des phrases : il compte des MOTS. Avant de compter, il faut que « RÉSILIE », « résilie » et
  « resilie » soient le même mot (défi 272) ;
- compter les mots perd leur ORDRE : « je vais résilier » et « je ne vais pas résilier » ont presque les mêmes mots.
  Le remède classique : marquer les mots qui suivent une négation (« non_resilier »), jusqu'à la fin de la
  proposition (défi 273) ;
- le BAYES NAÏF suppose que les mots d'un commentaire sont indépendants une fois la classe connue (c'est faux, d'où
  « naïf », et pourtant ça marche). P(classe | mots) est alors proportionnel à P(classe) × P(mot 1 | classe) ×
  P(mot 2 | classe) × … Entraîner, c'est estimer ces probabilités en COMPTANT (défi 274) ;
- un produit de 300 probabilités plus petites que 0,01 vaut 0.0 en float (le plus petit float positif est environ
  5e-324) : on additionne leurs LOGARITHMES, puis on revient aux probabilités sans déborder (défi 275) ;
- les défis 273 et 277 utilisent d'autres défis : pendant leur vérification, le correcteur prête sa propre `mots`
  (et ses `entrainer` et `probabilites`) à votre fichier : chaque défi est noté seul, mais faites le 272 d'abord.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_45 import *
    >>> anciens = commentaires_exemple()            # 4 000 couples (texte, "part" ou "reste")
    >>> anciens[10]
    ('Pas question de résilier, je suis bien ici.', 'reste')
    >>> nouveaux = nouveaux_exemple()               # les 1 000 réponses du mois suivant
    >>> vraies = [etiquette for _, etiquette in nouveaux]
    >>> bilan([regle_actuelle(texte) for texte, _ in nouveaux], vraies)       # la règle « résili » d'aujourd'hui
    {'signales': 180, 'trouves': 46, 'precision': 0.256, 'rappel': 0.207}

Quand les six défis sont réussis, Inès peut répondre au comité :

    >>> sans = entrainer([mots(texte) for texte, _ in anciens], [e for _, e in anciens])
    >>> modele = entrainer([avec_negations(texte) for texte, _ in anciens], [e for _, e in anciens])
    >>> def classe(modele, decouper, texte):
    ...     p = probabilites(modele, decouper(texte))
    ...     return max(p, key=p.get)
    >>> bilan([classe(sans, mots, texte) for texte, _ in nouveaux], vraies)
    {'signales': 189, 'trouves': 145, 'precision': 0.767, 'rappel': 0.653}
    >>> bilan([classe(modele, avec_negations, texte) for texte, _ in nouveaux], vraies)
    {'signales': 231, 'trouves': 171, 'precision': 0.74, 'rappel': 0.77}
    >>> texte = "Je ne suis pas satisfait."
    >>> round(probabilites(sans, mots(texte))["part"], 2), round(probabilites(modele, avec_negations(texte))["part"], 3)
    (0.51, 0.996)
    >>> [mot for mot, ecart in mots_revelateurs(modele, "part", 8)]
    ['code', 'demande', 'rio', 'des', 'engagement', 'fin', 'resilie', 'non_content']

La règle « résili » se trompe sept fois sur dix, et ne voit qu'un partant sur cinq. Le Bayes naïf, écrit en une
centaine de lignes, en trouve deux sur trois avec les mots seuls ; avec les négations, plus de trois sur quatre, pour à
peu près la même précision. Sans les négations, « Je ne suis pas satisfait. » est un pile ou face (51 %) : le modèle
voit « satisfait ». Avec, il y voit « non_satisfait ». Et le comité apprend ce que disent ceux qui partent : le code
RIO (qu'on demande pour garder son numéro chez un concurrent), la fin de l'engagement, « je ne suis plus content ».

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites le 272 d'abord (le 273 s'en sert) ; les
   autres sont indépendants, et l'ordre proposé est le plus progressif.
2. Lancez `python scripts/defis.py --defi 272` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_45.py`.

Les outils communs (`commentaires_exemple`, `nouveaux_exemple`, `regle_actuelle`, `bilan`, `generer_commentaires`,
`Modele`, `NEGATIONS`, `FINS_DE_PORTEE`, `PONCTUATION`, `PREFIXE` et `LETTRES_LIEES`) sont dans `defis/verbatims.py`,
déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import heapq  # noqa: F401 — heapq.nsmallest(k, elements, key=...) : les k premiers sans tout trier
import math  # noqa: F401 — math.log, math.exp, math.isfinite
import unicodedata  # noqa: F401 — unicodedata.normalize("NFKD", texte), unicodedata.combining(c)
from collections import Counter  # noqa: F401 — Counter(mots) compte les occurrences ; .update(mots) en ajoute

from defis.verbatims import (  # noqa: F401
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

# ============================================================================ Lire


def mots(texte) -> list[str]:
    """Défi 272 🟢 — Les mots d'un commentaire, écrits d'une seule façon : sans majuscules, sans accents.

        >>> mots("Je RÉSILIE l’offre ﬁbre !!! 😡")
        ['je', 'resilie', 'l', 'offre', 'fibre']
        >>> mots("C'est la 3ᵉ coupure ce mois-ci : 29,99 € pour ÇA ?")
        ['c', 'est', 'la', '3e', 'coupure', 'ce', 'mois', 'ci', '29', '99', 'pour', 'ca']
        >>> mots("Œuvre d'art, mot_de_passe")
        ['oeuvre', 'd', 'art', 'mot', 'de', 'passe']
        >>> mots("Merci à محمد du service client"), mots("!!!")
        (['merci', 'a', 'محمد', 'du', 'service', 'client'], [])

    Ici : le client a tapé « RÉSILIE » en majuscules, une apostrophe courbe (celle des téléphones), et « ﬁbre » avec la
    ligature « ﬁ » (un seul caractère, U+FB01) d'un texte copié depuis un PDF. Pour le modèle, « résilie »,
    « RESILIE » et « resilie » doivent être UN mot, et « ﬁbre » doit être « fibre ». « 3ᵉ » (le « ᵉ » en exposant de
    la typographie française) devient « 3e », comme l'écrivent ceux qui n'ont pas de « ᵉ » sur leur clavier. Les
    émojis et la ponctuation séparent les mots, et ne sont pas des mots. Le prénom écrit en arabe reste un mot.

    Les règles :

    - `texte` est un `str` : autre chose (None, des octets, un nombre) → TypeError ;
    - mettez le texte en minuscules avec `casefold()` (les minuscules « sans exception » : `"ß".casefold()` donne
      "ss"), puis décomposez-le avec `unicodedata.normalize("NFKD", ...)` : chaque lettre accentuée devient la lettre
      suivie de son accent (« é » → « e » + « ◌́ »), et les formes « de présentation » reprennent leur forme simple
      (« ﬁ » → « fi », « ᵉ » → « e », « ４ » → « 4 ») ;
    - retirez les accents : ce sont les caractères dont `unicodedata.combining(c)` n'est pas nul ;
    - certaines lettres ne se décomposent pas : remplacez « œ » et « æ » grâce à `LETTRES_LIEES` (« œ » → « oe ») ;
    - un MOT est une suite, la plus longue possible, de caractères pour lesquels `c.isalnum()` est vrai : des
      lettres et des chiffres, de n'importe quel alphabet. Tout le reste sépare les mots (espaces, ponctuation,
      apostrophes, traits d'union, émojis) et disparaît, y compris le tiret bas « _ » : il est réservé aux marques
      de négation du défi 273 (« non_resilier »), qu'un client ne doit pas pouvoir écrire lui-même ;
    - rendez la liste des mots dans l'ordre du texte, AVEC les répétitions (le modèle compte les occurrences) ;
      [] s'il n'y en a aucun.

    Les pièges : `texte.split()` garde « cher! » collé à son point d'exclamation ; `re.findall(r"\\w+", ...)` garde
    « mot_de_passe » en un seul mot (pour `\\w`, « _ » est une lettre) ; `[a-z0-9]+` efface les autres alphabets ;
    `isalpha()` efface les chiffres (« 4g » deviendrait « g ») ; `.encode("ascii", "ignore")` efface « œ » (« œuvre »
    → « uvre ») et le prénom en arabe ; NFD au lieu de NFKD garde « 3ᵉ » ; et `sorted(set(...))` perd l'ordre et les
    répétitions.

    Ce que vous apprenez : à « normaliser » un texte avant de le comparer ou de le compter (ce que font tous les
    moteurs de recherche et tous les modèles de texte), la décomposition Unicode NFKD, et à découper avec une règle
    écrite (un mot = une suite de `isalnum`) plutôt qu'avec une expression régulière qui en cache une autre.

    Indice : `decompose = unicodedata.normalize("NFKD", texte.casefold())` ; `"".join(c for c in decompose if not
    unicodedata.combining(c))` ; `.translate(str.maketrans(LETTRES_LIEES))` ; puis parcourez les caractères en
    construisant le mot courant, et rangez-le quand arrive un séparateur (et à la fin du texte).
    """
    raise NotImplementedError


def avec_negations(texte) -> list[str]:
    """Défi 273 🟡 — Les mots du texte, où chaque mot qui suit une négation devient « non_mot ».

        >>> avec_negations("Je ne vais pas résilier, le débit est excellent.")
        ['je', 'ne', 'non_vais', 'pas', 'non_resilier', 'le', 'debit', 'est', 'excellent']
        >>> avec_negations("Je n'ai jamais eu de coupure, mais c'est plus cher qu'avant.")
        ['je', 'n', 'non_ai', 'jamais', 'non_eu', 'non_de', 'non_coupure', 'mais', 'c', 'est', 'plus', 'cher', 'qu', 'avant']
        >>> avec_negations("Sans fibre pas de télétravail. Je résilie !")
        ['sans', 'non_fibre', 'pas', 'non_de', 'non_teletravail', 'je', 'resilie']

    Ici : dans « Je ne vais pas résilier », « résilier » est nié : il devient « non_resilier », un AUTRE mot pour le
    modèle, que les clients qui restent écrivent souvent. La négation commence à « ne » (en français, le verbe est
    ENTRE « ne » et « pas » : « vais » est nié aussi) ; la virgule la referme : « le débit est excellent » n'est pas
    nié. Dans le deuxième exemple, « n' » (devenu « n » au défi 272) ouvre la négation, et « mais » la referme avant la
    fin de la proposition. « plus » n'ouvre rien : « c'est plus cher » n'est pas une négation (et « je ne veux plus »
    est nié grâce à « ne »).

    Les règles :

    - `texte` est un `str` (TypeError sinon) ;
    - coupez le texte en PROPOSITIONS : chaque caractère de `PONCTUATION` (« . , ; : ! ? … ( ) » et le retour à la
      ligne) termine une proposition (et disparaît) ;
    - dans chaque proposition, prenez les mots avec `mots(proposition)` (votre défi 272 : pendant la vérification, le
      correcteur prête le sien), puis parcourez-les, en partant « sans négation » au début de CHAQUE proposition :
        • un mot de `NEGATIONS` (« ne », « n », « pas », « jamais », « rien », « aucun », « aucune », « sans », « ni »)
          est gardé tel quel, et ouvre la négation, même si elle est déjà ouverte : « ne … pas » ne s'annulent PAS
          (ce n'est pas une double négation, c'est une négation en deux morceaux) ;
        • un mot de `FINS_DE_PORTEE` (« mais », « car », « donc », « sauf », « pourtant », « cependant ») est gardé tel
          quel, et referme la négation ;
        • tout autre mot devient `PREFIXE + mot` (« non_ » + mot) si la négation est ouverte, et reste tel quel sinon ;
    - rendez tous les mots, proposition après proposition, dans une seule liste ; [] s'il n'y en a aucun.

    Ce que vous apprenez : la « portée » d'une négation (la technique de Das et Chen, 2001, reprise dans tous les
    cours d'analyse de sentiment), à garder un ÉTAT en parcourant une liste (une variable « nié » qui s'ouvre et se
    referme), à découper un texte en deux étapes (propositions, puis mots), et à réutiliser une fonction déjà testée.

    Indice : une boucle sur les caractères qui range la proposition courante dès qu'arrive un `c in PONCTUATION` (sans
    oublier la dernière) ; puis, pour chaque proposition, `nie = False` et une boucle sur `mots(proposition)`.
    """
    raise NotImplementedError


# ============================================================================ Apprendre


def entrainer(documents, etiquettes, alpha=1.0) -> Modele:
    """Défi 274 🟡 — Le Bayes naïf : compter les mots de chaque classe, lisser, prendre les logarithmes.

        >>> modele = entrainer([["prix", "cher", "prix"], ["merci"], ["prix", "merci"]], ["part", "reste", "reste"])
        >>> modele.classes
        ['part', 'reste']
        >>> {c: round(math.exp(v), 4) for c, v in modele.log_a_priori.items()}
        {'part': 0.3333, 'reste': 0.6667}
        >>> {mot: round(math.exp(v), 4) for mot, v in sorted(modele.log_proba["part"].items())}
        {'cher': 0.3333, 'merci': 0.1667, 'prix': 0.5}
        >>> {mot: round(math.exp(v), 4) for mot, v in sorted(modele.log_proba["reste"].items())}
        {'cher': 0.1667, 'merci': 0.5, 'prix': 0.3333}

    Ici : trois commentaires déjà découpés en mots, un client parti et deux restés. P(part) = 1/3. Le vocabulaire a
    trois mots : « cher », « merci », « prix ». Chez les partis, on compte 3 mots : « prix » deux fois, « cher » une
    fois, « merci » jamais. Sans lissage, P(merci | part) vaudrait 0 : un seul « merci » suffirait à rendre le départ
    IMPOSSIBLE, quels que soient les autres mots. Avec le lissage de Laplace (alpha = 1), on ajoute 1 à chaque compte :
    P(prix | part) = (2 + 1) / (3 + 1 × 3) = 0,5, P(cher | part) = (1 + 1) / 6, P(merci | part) = (0 + 1) / 6.

    Les règles :

    - `documents` est une liste (ou un tuple) de documents ; un DOCUMENT est une liste (ou un tuple) de mots (des
      `str`), comme ceux que rendent `mots` et `avec_negations`. Un document qui est un `str` (le texte au lieu de
      ses mots), un mot qui n'est pas un `str` → TypeError ;
    - `etiquettes` est une liste (ou un tuple) de `str` (TypeError sinon), une par document : pas la même longueur
      que `documents` → ValueError ; aucun document → ValueError (il n'y a rien à apprendre) ;
    - `alpha` est un nombre > 0 et fini (pas un bool) : TypeError ou ValueError. alpha = 0, c'est l'absence de
      lissage, et un `math.log(0)` qui attend son heure ;
    - rendez un `Modele(classes=..., log_a_priori=..., log_proba=...)` (la classe est dans `defis/verbatims.py`) :
        • `classes` : la liste des étiquettes DISTINCTES, dans l'ordre croissant (`sorted`) ;
        • `log_a_priori[c]` = log(nombre de documents de c / nombre de documents) ;
        • le VOCABULAIRE est l'ensemble des mots vus dans TOUS les documents, toutes classes confondues ; `n(c, mot)`
          est le nombre d'OCCURRENCES de `mot` dans les documents de c (un mot répété compte autant de fois), et
          N(c) la somme des n(c, mot) : le nombre total de mots des documents de c (répétitions comprises) ;
        • `log_proba[c]` est un dict qui contient TOUS les mots du vocabulaire (même ceux que c n'a jamais vus), avec
          log_proba[c][mot] = log((n(c, mot) + alpha) / (N(c) + alpha × taille du vocabulaire)) ;
        • « log », c'est le logarithme NÉPÉRIEN : `math.log` (pas `math.log10`) ;
    - un document vide compte pour les probabilités a priori, et n'ajoute aucun mot ; les arguments ne sont pas
      modifiés ;
    - le correcteur passe 30 000 documents et un vocabulaire de 20 000 mots, et compte les lignes exécutées : compter
      chaque mot du vocabulaire en reparcourant tous les documents est beaucoup trop lent. Parcourez les documents UNE
      fois, avec un `Counter` par classe.

    Ce que vous apprenez : qu'entraîner un modèle peut se résumer à COMPTER (le Bayes naïf multinomial de scikit-learn,
    `MultinomialNB`, fait exactement ce calcul : le correcteur le vérifie), le lissage de Laplace (ce qu'on n'a jamais
    vu n'est pas impossible), et pourquoi on range des logarithmes plutôt que des probabilités.

    Indice : `nb_documents = Counter(etiquettes)` ; `comptes = {c: Counter() for c in nb_documents}` ; `for document,
    etiquette in zip(documents, etiquettes): comptes[etiquette].update(document)` ; `vocabulaire =
    set().union(*comptes.values())` ; `total = sum(comptes[c].values())`.
    """
    raise NotImplementedError


def probabilites(modele, mots) -> dict:
    """Défi 275 🟡 — La probabilité de chaque classe pour une liste de mots, même quand elle est très longue.

        >>> modele = entrainer([["prix", "cher", "prix"], ["merci"], ["prix", "merci"]], ["part", "reste", "reste"])
        >>> {c: round(p, 4) for c, p in probabilites(modele, ["prix", "cher"]).items()}
        {'part': 0.6, 'reste': 0.4}
        >>> {c: round(p, 4) for c, p in probabilites(modele, ["prix", "inconnu", "prix"]).items()}
        {'part': 0.5294, 'reste': 0.4706}
        >>> {c: round(p, 4) for c, p in probabilites(modele, []).items()}
        {'part': 0.3333, 'reste': 0.6667}
        >>> 0.5 ** 1100                                         # 1 100 mots de probabilité 0,5 : le produit…
        0.0
        >>> {c: round(p, 4) for c, p in probabilites(modele, ["prix"] * 1100).items()}
        {'part': 1.0, 'reste': 0.0}

    Ici : pour « prix cher », le score de « part » est P(part) × P(prix | part) × P(cher | part) = 1/3 × 1/2 × 1/3 =
    1/18, celui de « reste » 2/3 × 1/3 × 1/6 = 1/27. On les ramène à une somme de 1 : 0,6 et 0,4. Le mot « inconnu »,
    jamais vu à l'entraînement, ne dit rien : on l'ignore ; « prix » répété compte deux fois. Sans aucun mot, il reste
    les probabilités a priori. Un courrier de réclamation de 1 100 mots ferait un produit de 1 100 probabilités : 0.0
    en float, pour TOUTES les classes, et 0 / 0 ensuite. En logarithmes, rien ne déborde.

    Les règles :

    - `modele` est un `Modele` comme ceux du défi 274 (le correcteur fabrique les siens) ; `mots` est une liste (ou un
      tuple) de `str`. Un `str` à la place de la liste (le texte au lieu de ses mots : « prix » serait lu « p », « r »,
      « i », « x »), ou un mot qui n'est pas un `str` → TypeError ;
    - pour chaque classe c, le SCORE vaut log_a_priori[c] + la somme des log_proba[c][mot] pour chaque mot de `mots`
      qui est dans le vocabulaire (un mot répété compte autant de fois qu'il apparaît ; un mot hors du vocabulaire est
      ignoré : il n'a de log_proba dans aucune classe) ;
    - la probabilité de c vaut exp(score de c) / la somme des exp(score) de toutes les classes. Ne calculez pas
      `math.exp(score)` directement : avec 1 100 mots, le score vaut environ -800, et `math.exp(-800)` vaut 0.0.
      Retranchez d'abord le PLUS GRAND des scores : la plus grande exponentielle vaut alors exp(0) = 1, les autres
      sont entre 0 et 1, et le quotient ne change pas (c'est l'astuce « log-sum-exp ») ;
    - rendez un dict {classe: probabilité} (des float), avec les classes dans l'ordre de `modele.classes`. Les
      probabilités font 1 au total ; une classe très improbable peut valoir 0.0.

    Ce que vous apprenez : à calculer en LOGARITHMES quand on multiplie beaucoup de petits nombres (les modèles de
    langue, les modèles de Markov cachés, la vraisemblance de toute la statistique le font), l'astuce « log-sum-exp »
    (`scipy.special.softmax` la fait pour vous), et à ignorer ce que le modèle n'a jamais vu plutôt que de planter.

    Indice : `vocabulaire = modele.log_proba[modele.classes[0]]` (tous les log_proba ont les mêmes mots) ; `scores =
    {c: modele.log_a_priori[c] + sum(...) for c in modele.classes}` ; `plus_grand = max(scores.values())` ;
    `math.exp(s - plus_grand)`.
    """
    raise NotImplementedError


# ============================================================================ Expliquer et durer


def mots_revelateurs(modele, classe, k=10) -> list[tuple[str, float]]:
    """Défi 276 🟡 — Les `k` mots qui distinguent le mieux `classe` de TOUTES les autres classes.

        >>> modele = entrainer([["le", "prix", "le", "rio"], ["le", "prix", "le", "cher"],
        ...                     ["le", "merci", "le", "top"], ["le", "merci", "le", "prix"]],
        ...                    ["part", "part", "reste", "reste"])
        >>> [(mot, round(ecart, 3)) for mot, ecart in mots_revelateurs(modele, "part")]
        [('cher', 0.693), ('rio', 0.693), ('prix', 0.405)]
        >>> [(mot, round(ecart, 3)) for mot, ecart in mots_revelateurs(modele, "reste", k=1)]
        [('merci', 1.099)]

    Ici : « le » est le mot le plus fréquent chez les partis (4 mots sur 8), mais il l'est autant chez les restés : il
    ne révèle rien. Pour chaque mot, on regarde de combien il est plus probable dans la classe que dans la classe
    CONCURRENTE la plus forte, en logarithme : P(rio | part) = (1 + 1) / (8 + 6) = 1/7 et P(rio | reste) = 1/14, écart
    log(1/7) - log(1/14) = log(2) = 0,693. « cher » a le même écart : à égalité, l'ordre alphabétique. « prix », écrit
    deux fois chez les partis mais aussi une fois chez les restés, révèle moins : log(3/14) - log(2/14) = 0,405.
    « rio » n'a été écrit qu'une fois : le lissage du défi 274 tempère les mots rares (sans lui, un mot vu une seule
    fois, et jamais ailleurs, aurait un écart infini).

    Les règles :

    - `modele` est un `Modele` comme ceux du défi 274 ; `classe` doit être l'une de `modele.classes` (ValueError
      sinon), et le modèle doit avoir au moins deux classes (ValueError sinon : il n'y a rien à comparer) ; `k` est un
      int >= 0 (pas un bool) : TypeError ou ValueError ;
    - pour chaque mot du vocabulaire, l'ÉCART vaut log_proba[classe][mot] - le PLUS GRAND des log_proba[c][mot] pour
      les AUTRES classes c (avec trois classes ou plus, un mot ne révèle une classe que s'il la distingue de chacune
      des autres, pas « en moyenne ») ;
    - gardez les mots dont l'écart est strictement POSITIF (un mot au moins aussi probable ailleurs ne révèle rien) ;
    - rangez-les par écart DÉCROISSANT ; à écart égal, par ordre alphabétique du mot (l'ordre de `sorted` sur les
      textes), quel que soit l'ordre des mots dans `modele.log_proba` ;
    - rendez les `k` premiers, sous la forme d'une liste de tuples (mot, écart), l'écart étant la différence des
      logarithmes (un float, pas le rapport des probabilités) ; moins de `k` s'il n'y en a pas assez ; [] si k vaut 0.

    Ce que vous apprenez : à expliquer un modèle avec ses propres paramètres (le « log-odds ratio » des analyses de
    texte), qu'un mot fréquent n'est pas un mot révélateur, à départager les égalités pour que la réponse ne dépende
    pas de l'ordre d'un dict, et les k premiers d'une liste sans tout trier (`heapq.nsmallest`).

    Indice : `autres = [c for c in modele.classes if c != classe]` ; `ecart = log_p - max(modele.log_proba[c][mot] for
    c in autres)` ; `heapq.nsmallest(k, ecarts, key=lambda paire: (-paire[1], paire[0]))` (ou `sorted(...)[:k]`).
    """
    raise NotImplementedError


class ClassifieurEnLigne:
    """Défi 277 🔴 — Un Bayes naïf qui apprend chaque soir, oublie un client qui le demande, et prédit sans tout recompter.

        >>> clf = ClassifieurEnLigne()                       # alpha = 1 par défaut
        >>> clf.apprendre(["prix", "cher", "prix"], "part")
        >>> clf.apprendre(["merci"], "reste")
        >>> clf.apprendre(["prix", "merci"], "reste")
        >>> {c: round(p, 4) for c, p in clf.probabilites(["prix", "cher"]).items()}      # comme au défi 275
        {'part': 0.6, 'reste': 0.4}
        >>> clf.oublier(["prix", "cher", "prix"], "part")    # le client demande l'effacement de ses données
        >>> len(clf), clf.modele().classes, sorted(clf.modele().log_proba["reste"])
        (2, ['reste'], ['merci', 'prix'])
        >>> clf.oublier(["cher"], "reste")                   # « cher » n'a jamais été appris comme « reste »
        Traceback (most recent call last):
          ...
        ValueError: ce document n'a pas été appris avec l'étiquette 'reste'

    Ici : chaque soir, les commentaires dont on connaît maintenant l'issue (le client est-il parti dans les deux
    mois ?) s'ajoutent au modèle, sans réentraîner sur les six derniers mois. Et quand un client exerce son droit à
    l'effacement (RGPD, article 17), son commentaire doit SORTIR du modèle : ensuite, tout se passe comme s'il n'avait
    jamais été appris. Ici, le seul client parti s'en va : la classe « part » disparaît, et « cher », que lui seul
    avait écrit, sort du vocabulaire.

    Ce qu'il faut écrire :

    - `__init__(alpha=1.0)` : un classifieur vide ; `alpha` comme au défi 274 (un nombre > 0 et fini, pas un bool) ;
    - `apprendre(mots, etiquette)` : ajoute un document (une liste ou un tuple de `str`) d'étiquette `etiquette` (un
      `str`) ;
    - `oublier(mots, etiquette)` : retire un document appris avec cette étiquette ;
    - `probabilites(mots)` : le dict {classe: probabilité} du défi 275, pour le modèle tel qu'il est maintenant ;
    - `modele()` : un `Modele` égal à celui que rendrait `entrainer` (défi 274) sur les documents appris et pas
      oubliés, avec le même alpha ;
    - `len(clf)` (`__len__`) : le nombre de documents appris et pas oubliés.

    Les règles :

    - après n'importe quelle suite d'`apprendre` et d'`oublier`, `probabilites` et `modele` donnent EXACTEMENT ce que
      donneraient `probabilites(entrainer(documents, etiquettes, alpha), mots)` et `entrainer(documents, etiquettes,
      alpha)` sur les documents restants (à un milliardième près) ;
    - une classe dont le dernier document est oublié DISPARAÎT ; un mot que plus aucun document restant ne contient
      SORT du vocabulaire (la taille du vocabulaire, au dénominateur, diminue) ;
    - `oublier` lève ValueError si l'étiquette n'a aucun document, ou si un mot de `mots` y apparaît PLUS de fois que
      dans les documents appris avec cette étiquette (on ne peut pas oublier ce qu'on n'a pas appris) ; et alors RIEN ne
      change : vérifiez tout AVANT de modifier quoi que ce soit ;
    - `probabilites` et `modele` sur un classifieur vide → ValueError (rien n'a été appris) ; un `mots` qui est un
      `str`, un mot ou une étiquette qui n'est pas un `str` → TypeError ;
    - `modele()` rend une photographie : les `apprendre` et `oublier` suivants ne la modifient pas. Deux classifieurs ne
      partagent rien ;
    - le correcteur apprend 20 000 documents (un vocabulaire de 40 000 mots), en oublie 4 000 et fait 10 000
      prédictions, et compte les lignes exécutées : `apprendre`, `oublier` et `probabilites` doivent coûter de l'ordre
      du nombre de mots du document (plus le nombre de classes), PAS de la taille du vocabulaire. Recalculer tous les
      log_proba (ou appeler `entrainer`) à chaque prédiction est beaucoup trop lent. Gardez les COMPTES à jour (les
      documents par classe, les occurrences de chaque mot par classe, le total de mots par classe, et le nombre
      d'occurrences de chaque mot toutes classes confondues, pour savoir quand il sort du vocabulaire) et calculez les
      log-probabilités des seuls mots demandés.

    Ce que vous apprenez : l'apprentissage EN LIGNE (`partial_fit` dans scikit-learn), qu'un modèle « par comptage » se
    met à jour dans les deux sens (c'est ce qui rend le droit à l'effacement possible sans tout réentraîner), à garder
    des comptes cohérents (un mot à zéro doit vraiment disparaître), et à vérifier une opération avant de l'appliquer.

    Indice : dans `__init__`, `self._documents = Counter()`, `self._comptes = {}` (classe → Counter), `self._totaux =
    Counter()` et `self._vocabulaire = Counter()` (mot → occurrences, toutes classes) ; dans `oublier`, `a_retirer =
    Counter(mots)`, vérifiez, puis retirez (et `del` ce qui tombe à zéro) ; `probabilites` reprend le calcul du défi 275
    avec log((n(c, mot) + alpha) / (N(c) + alpha × len(self._vocabulaire))).
    """

    def __init__(self, alpha=1.0) -> None:
        raise NotImplementedError

    def apprendre(self, mots, etiquette) -> None:
        raise NotImplementedError

    def oublier(self, mots, etiquette) -> None:
        raise NotImplementedError

    def probabilites(self, mots) -> dict:
        raise NotImplementedError

    def modele(self) -> Modele:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

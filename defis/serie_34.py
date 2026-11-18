"""Série 34 — Partager sans exposer : pseudonymiser, recouper, auditer, généraliser, secret statistique, bruit.

🕵️ VOUS VOULEZ DÉVELOPPER ? DES DONNÉES PERSONNELLES : ICI, VOUS LES PROTÉGEZ, ET VOUS PROUVEZ QUE ÇA TIENT.

Le laboratoire d'économie de l'université veut étudier le churn sur de vraies données ; la direction veut publier
les départs par commune dans son rapport annuel. Inès, la déléguée à la protection des données (DPO), doit signer
avant que quoi que ce soit sorte. Sa question n'est pas « avez-vous enlevé les noms ? » mais « quelqu'un peut-il
les remettre ? ». Cette série vous fait écrire les six pièces de sa réponse :

    206 preparer_export       ne sortir que les colonnes demandées, l'identifiant remplacé par un pseudonyme
    207 reidentifier          l'attaque : remettre un nom sur une ligne grâce à un fichier de prospection
    208 audit                 mesurer le risque : k-anonymat et l-diversité
    209 generaliser           rendre le fichier moins précis, le moins possible, jusqu'à ce qu'il tienne
    210 secret_statistique    publier un tableau de comptages sans qu'une petite case se retrouve par soustraction
    211 Budget                répondre aux questions avec un bruit calibré, dans la limite d'un budget

Prérequis : la série 00 (listes, dictionnaires, boucles) et une classe (défi 48) pour le défi 211. Les valeurs
manquantes de la série 27 (le NaN qui passe un `if`) l'éclairent. Le défi 85 (série 13) vous a fait écrire un HMAC
pour signer un lien : ici, le même outil sert à pseudonymiser.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un IDENTIFIANT DIRECT désigne une personne à lui seul : nom, e-mail, téléphone, numéro client. Il ne sort jamais ;
- un QUASI-IDENTIFIANT est anodin seul (l'âge, le sexe, le code postal) mais, ensemble, ils désignent souvent UNE
  personne : aux États-Unis, la date de naissance, le sexe et le code postal suffisent pour 87 % de la population
  (L. Sweeney, 2000). Un ATTRIBUT SENSIBLE est ce qu'on ne veut pas voir relié à un nom : ici, le motif de départ
  « impayé » ;
- PSEUDONYMISER, c'est remplacer l'identifiant par un code. Pour le RGPD, une donnée pseudonymisée reste une donnée
  personnelle (considérant 26) : la suite de la série montre pourquoi. ANONYMISER, c'est rendre la ré-identification
  impossible en pratique, et ça se MESURE ;
- une ATTAQUE PAR RECOUPEMENT relie le fichier publié à un fichier qui a des noms (liste électorale, fichier de
  prospection, réseau social) par les quasi-identifiants. Une ligne qui n'a qu'un seul candidat est ré-identifiée ;
- un fichier est K-ANONYME si chaque combinaison de quasi-identifiants y est partagée par au moins k lignes :
  l'attaquant hésite toujours entre au moins k personnes. Il est L-DIVERS si, dans chacun de ces groupes, l'attribut
  sensible prend au moins l valeurs différentes. Sinon, pas besoin de savoir QUI : tout le groupe a la même ;
- un tableau de comptages publié avec ses totaux se résout comme un système d'équations : cacher UNE case d'une
  ligne ne sert à rien quand le total de la ligne est publié ;
- la CONFIDENTIALITÉ DIFFÉRENTIELLE ajoute à chaque réponse un bruit tiré au hasard, calibré pour que la réponse
  change à peine qu'un client donné soit dans les données ou non. Chaque réponse coûte un « epsilon » ; les coûts
  s'additionnent, et on s'arrête quand le budget est épuisé.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_34 import *
    >>> clients = clients_exemple()
    >>> clients[9]["nom"], clients[9]["age"], clients[9]["sexe"], clients[9]["code_postal"], clients[9]["motif"]
    ('Karim Mansouri', 64, 'M', '69120', 'impayé')

Quand les six défis sont réussis, vous pouvez répondre à Inès :

    >>> export = preparer_export(clients, COLONNES_ETUDE, CLE_EXEMPLE)       # ni nom, ni e-mail, ni numéro
    >>> sorted(export[9])
    ['age', 'anciennete', 'churn', 'code_postal', 'contrat', 'motif', 'pseudo', 'sexe']
    >>> retrouves = reidentifier(export, annuaire_exemple(), QUASI_IDENTIFIANTS)
    >>> len(retrouves), retrouves[9], export[9]["motif"]       # 13 clients sur 20 ont retrouvé leur nom
    (13, 'Karim Mansouri', 'impayé')
    >>> audit(export, QUASI_IDENTIFIANTS, "motif")[:2]         # k = 1 : des lignes uniques
    (1, 1)
    >>> g = generaliser(export, HIERARCHIES, 3)
    >>> g.niveaux, g.supprimees                                # sur 20 lignes, l'âge et le sexe y passent
    ({'age': 4, 'sexe': 1, 'code_postal': 1}, 0)
    >>> annuaire = [appliquer(p, HIERARCHIES, g.niveaux) for p in annuaire_exemple()]   # l'attaquant s'adapte
    >>> reidentifier(g.lignes, annuaire, QUASI_IDENTIFIANTS)
    {}
    >>> resultat = audit(g.lignes, QUASI_IDENTIFIANTS, "motif")
    >>> resultat.k, resultat.l, resultat.classes[0]
    (3, 1, Classe(valeurs=('*', '*', '6912*'), effectif=3, diversite=1))

Plus personne n'a de nom… mais les trois clients de Vaulx-en-Velin (6912*) sont tous partis pour « impayé » : qui
sait que son voisin est client sait pourquoi il est parti. Le fichier est 3-anonyme, pas 2-divers. Inès refuse le
fichier ligne à ligne, et propose des chiffres agrégés :

    >>> secret_statistique(comptages_exemple(), 3)[3]     # Vaulx-en-Velin : 1 déménagement à cacher… et les 4 impayés
    [False, False, True, True]
    >>> budget = Budget(1.0, tirage_hasard(40))
    >>> round(budget.compter(clients, lambda c: c["motif"] == "impayé", 0.5), 1)    # 5 en vrai
    5.3
    >>> round(budget.sommer(clients, "facture", 150, 0.5), 1)                        # 1 335 € en vrai
    1349.7
    >>> budget.restant
    0.0

Le laboratoire a ses deux réponses, et personne ne peut dire si Karim est dedans. Le hasard a été clément : le bruit
d'un comptage est de l'ordre de 1 / epsilon = 2 clients, celui de la somme de plafond / epsilon = 300 €. Sur 20
clients, c'est beaucoup ; sur les 5 000 clients de `data/customers.csv`, c'est déjà peu. La confidentialité
différentielle est faite pour les grands nombres.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les défis sont indépendants ; l'ordre proposé est
   celui de l'histoire.
2. Lancez `python scripts/defis.py --defi 206` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_34.py`.

Les outils communs (`clients_exemple`, `annuaire_exemple`, `comptages_exemple`, `HIERARCHIES`, `appliquer`,
`QUASI_IDENTIFIANTS`, `IDENTIFIANTS_DIRECTS`, `COLONNES_ETUDE`, `CLE_EXEMPLE`, `Classe`, `Audit`, `Generalisation`,
`BudgetEpuise`, `tirages_fixes`, `tirage_hasard`, `generer_clients`, `generer_annuaire`) sont dans
`defis/confidentialite.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import hashlib  # noqa: F401 — hashlib.sha256 : la fonction de hachage du HMAC
import hmac  # noqa: F401 — hmac.new(cle, message, hashlib.sha256).hexdigest()
import itertools  # noqa: F401 — itertools.product : toutes les combinaisons de niveaux (défi 209)
import math  # noqa: F401 — math.log, math.copysign, math.isnan

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

# ============================================================================ Ce qui sort, et sous quel nom


def preparer_export(lignes: list[dict], colonnes: list[str], cle: bytes) -> list[dict]:
    """Défi 206 🟢 — Le fichier qui part au laboratoire : un pseudonyme, puis SEULEMENT les colonnes demandées.

        >>> export = preparer_export(clients_exemple(), ["age", "churn"], CLE_EXEMPLE)
        >>> export[0]                                                    # doctest: +ELLIPSIS
        {'pseudo': '704b7184d689...', 'age': 34, 'churn': 1}
        >>> len(export[0]["pseudo"]), export[0]["pseudo"] != export[1]["pseudo"]
        (64, True)
        >>> preparer_export([{"customer_id": " c00412 "}], [], CLE_EXEMPLE) == [{"pseudo": export[0]["pseudo"]}]
        True

    Le dernier exemple : « c00412 » avec des espaces, tel que l'écrit l'outil du support, est le même client que
    « C00412 » dans le CRM. Il reçoit le même pseudonyme : le laboratoire pourra relier ses exports d'un mois sur
    l'autre, sans jamais savoir qui est derrière.

    Les règles :

    - chaque ligne de sortie est un NOUVEAU dictionnaire : la clé "pseudo", puis chaque colonne de `colonnes`, avec
      sa valeur (None si la ligne ne l'a pas). Aucune autre colonne : c'est une LISTE BLANCHE. Retirer seulement les
      colonnes interdites (une liste noire) laisserait partir la colonne « note_conseiller » ajoutée au CRM la
      semaine dernière, avec le numéro de téléphone qu'un client a dicté. C'est la minimisation du RGPD ;
    - pseudo = `hmac.new(cle, identifiant.encode("utf-8"), hashlib.sha256).hexdigest()`, les 64 caractères, où
      identifiant = `ligne["customer_id"].strip().upper()` (le même client, écrit de deux façons, doit avoir le même
      pseudonyme) ;
    - la clé reste chez vous : ce sont des bytes, au moins 16 octets, sinon ValueError. Un SHA-256 SANS clé ne
      protège rien : il n'existe qu'un million d'identifiants de la forme « C » + 6 chiffres, et un attaquant les
      essaie tous en une seconde. `sha256(cle + identifiant)` n'est pas un HMAC (relisez le défi 85) ;
    - gardez les 64 caractères : tronqué à 8, un pseudonyme n'a que 4 milliards de valeurs possibles, et il y a
      déjà une chance sur deux que deux clients parmi 77 000 reçoivent le même (le paradoxe des anniversaires) ;
    - une colonne de `colonnes` qui est un identifiant direct (IDENTIFIANTS_DIRECTS : customer_id, nom, email,
      telephone) → ValueError, même si le laboratoire la demande ; une ligne sans customer_id (absent, None, pas un
      texte, ou vide) → ValueError ;
    - ne modifiez pas `lignes`.

    Ce que vous apprenez : la minimisation (on n'envoie que ce qui est demandé, par liste blanche), la
    pseudonymisation par HMAC et pourquoi un hachage sans clé n'en est pas une, et normaliser un identifiant avant
    de le transformer.

    Indice : `identifiant = ligne.get("customer_id")`, vérifiez que c'est un texte non vide, puis
    `{"pseudo": hmac.new(cle, identifiant.strip().upper().encode("utf-8"), hashlib.sha256).hexdigest()}` et une
    boucle sur `colonnes` avec `ligne.get(colonne)`.
    """
    raise NotImplementedError


def reidentifier(publication: list[dict], annuaire: list[dict], quasi_identifiants: list[str]) -> dict[int, str]:
    """Défi 207 🟡 — L'attaque par recoupement : {numéro de ligne publiée: nom} quand UN SEUL candidat correspond.

        >>> publication = [{"age": 34, "sexe": "F", "motif": "prix"}, {"age": None, "sexe": "M", "motif": "impayé"},
        ...                {"age": 50, "sexe": "F", "motif": None}]
        >>> annuaire = [{"nom": "Ana", "age": 34, "sexe": "F"}, {"nom": "Ben", "age": 34, "sexe": "M"},
        ...             {"nom": "Cléo", "age": 50, "sexe": "F"}, {"nom": "Dina", "age": 50, "sexe": "F"}]
        >>> reidentifier(publication, annuaire, ["age", "sexe"])
        {0: 'Ana', 1: 'Ben'}

    Ici : la ligne 0 (34 ans, F) n'a qu'une candidate, Ana. L'âge de la ligne 1 a été effacé : on la compare sur le
    sexe seulement, et Ben est le seul homme de l'annuaire. Il est parti pour impayé, et l'attaquant le sait. La
    ligne 2 a deux candidates (Cléo et Dina) : l'attaquant ne peut pas trancher.

    Les règles :

    - pour chaque ligne publiée (numérotée à partir de 0, dans l'ordre), ses CANDIDATS sont les personnes de
      l'annuaire qui ont exactement les mêmes valeurs (==) pour chaque quasi-identifiant CONNU de la ligne ;
    - une valeur effacée dans la publication (None, NaN, ou colonne absente) ne filtre rien : la personne peut avoir
      n'importe quelle valeur. Effacer une valeur ne protège donc pas la ligne : les autres suffisent souvent ;
    - dans l'annuaire, une valeur absente (None, NaN ou colonne absente) n'est égale à aucune valeur connue ;
    - un seul candidat → la ligne est ré-identifiée : resultat[numéro] = le "nom" du candidat. Zéro ou plusieurs
      candidats → la ligne n'apparaît pas dans le résultat. Deux lignes de l'annuaire sont deux personnes, même si
      elles portent le même nom ;
    - quasi_identifiants vide → ValueError ;
    - ne modifiez ni `publication` ni `annuaire` ;
    - 10 000 lignes publiées face à 10 000 personnes : le correcteur compte les lignes exécutées. Comparer chaque
      ligne à chaque personne, c'est 100 millions de comparaisons. Construisez un INDEX (un dictionnaire) qui, pour
      chaque combinaison de valeurs, donne le nombre de personnes et le nom de la première. Il en faut un par MOTIF
      de quasi-identifiants connus (tous connus ; l'âge effacé ; …) : au plus 2 ** len(quasi_identifiants), et
      chacun ne se construit qu'une fois.

    Ce que vous apprenez : l'attaque qui a fait naître le k-anonymat (en 1997, L. Sweeney a retrouvé le dossier
    médical du gouverneur du Massachusetts dans un fichier « anonyme », avec la liste électorale de Cambridge),
    une jointure par dictionnaire au lieu d'une double boucle, et pourquoi une valeur manquante se traite à part.

    Indice : `connus = tuple(not manquant(ligne.get(q)) for q in quasi_identifiants)`, avec
    `manquant = lambda x: x is None or x != x` (NaN est le seul nombre différent de lui-même). Si ce motif n'a pas
    encore son index, parcourez l'annuaire une fois et comptez, pour chaque clé
    `tuple(personne.get(q) for q, c in zip(quasi_identifiants, connus) if c)`, les personnes et le premier nom.
    """
    raise NotImplementedError


# ============================================================================ Mesurer, puis réduire le risque


def audit(lignes: list[dict], quasi_identifiants: list[str], sensible: str) -> Audit:
    """Défi 208 🟢 — Audit(k, l, classes) : à combien d'autres chaque ligne ressemble, et ce que son groupe trahit.

        >>> lignes = [{"age": 30, "motif": "prix"}, {"age": 41, "motif": "prix"}, {"age": None, "motif": "impayé"},
        ...           {"age": 30, "motif": "prix"}, {"age": 41, "motif": None}, {"age": float("nan"), "motif": None},
        ...           {"age": 41, "motif": "prix"}]
        >>> audit(lignes, ["age"], "motif")
        Audit(k=2, l=1, classes=[Classe(valeurs=(30,), effectif=2, diversite=1),
                                 Classe(valeurs=(None,), effectif=2, diversite=2),
                                 Classe(valeurs=(41,), effectif=3, diversite=2)])

    Ici : les deux clients de 30 ans sont partis pour « prix » : leur classe est 2-anonyme mais trahit leur motif
    (diversité 1). Les deux âges manquants (None et NaN) forment UNE classe. Le fichier est 2-anonyme et 1-divers.

    Les règles :

    - une CLASSE regroupe les lignes qui ont les mêmes valeurs pour tous les `quasi_identifiants` ; `valeurs` est
      ce tuple, dans l'ordre de `quasi_identifiants` ;
    - une valeur manquante (None, NaN, colonne absente) s'écrit None dans `valeurs`, et toutes les lignes où elle
      manque vont dans la MÊME classe. Ne les jetez pas : « le client dont l'âge manque » est un signe distinctif
      comme un autre. C'est le piège de pandas : `groupby` les jette par défaut (`dropna=False` les garde), et le
      fichier paraît plus sûr qu'il n'est. Et deux NaN ne sont pas égaux entre eux : convertissez-les en None AVANT
      de grouper ;
    - effectif = le nombre de lignes de la classe ; diversite = le nombre de valeurs DIFFÉRENTES de la colonne
      `sensible` dans la classe (une valeur manquante, None ou NaN, compte comme une valeur : None) ;
    - k = le plus petit effectif ; l = la plus petite diversité (pas forcément dans la même classe) ;
    - classes = la liste des Classe, triée par effectif croissant puis par diversité croissante (les plus exposées
      d'abord) ; à égalité, dans l'ordre où chaque classe apparaît pour la première fois dans `lignes`. `sorted` et
      `list.sort` sont STABLES : triez la liste construite dans cet ordre. Ne triez pas par `valeurs` : comparer
      None à un nombre lève TypeError ;
    - aucune ligne, `quasi_identifiants` vide, ou `sensible` parmi les quasi-identifiants → ValueError ;
    - le résultat est un Audit (ou un tuple (k, l, classes)) ; ne modifiez pas `lignes`.

    Ce que vous apprenez : le k-anonymat et la l-diversité (l'attaque par homogénéité : un groupe où tout le monde a
    la même donnée sensible la livre sans qu'on sache qui est qui), le `groupby` qui perd des lignes en silence
    (exercice 1.2), et le tri stable.

    Indice : `groupes = {}` ; pour chaque ligne, `cle = tuple(None if manquant(v) else v for v in …)`, puis
    `groupes.setdefault(cle, [0, set()])` : on ajoute 1 à l'effectif et la valeur sensible au set. Un dict garde
    l'ordre d'insertion : c'est l'ordre de première apparition.
    """
    raise NotImplementedError


def generaliser(lignes: list[dict], hierarchies: dict, k: int, suppression_max: int = 0) -> Generalisation:
    """Défi 209 🔴 — La généralisation la MOINS haute qui rend le fichier k-anonyme : Generalisation(…).

        >>> lignes = [{"age": 31, "sexe": "F"}, {"age": 33, "sexe": "M"}, {"age": 36, "sexe": "F"},
        ...           {"age": 52, "sexe": "M"}]
        >>> hierarchies = {"age": HIERARCHIES["age"], "sexe": HIERARCHIES["sexe"]}
        >>> generaliser(lignes, hierarchies, 2)
        Generalisation(niveaux={'age': 4, 'sexe': 0},
                       lignes=[{'age': '*', 'sexe': 'F'}, {'age': '*', 'sexe': 'M'}, {'age': '*', 'sexe': 'F'},
                               {'age': '*', 'sexe': 'M'}],
                       supprimees=0)
        >>> generaliser(lignes, hierarchies, 2, suppression_max=1)
        Generalisation(niveaux={'age': 2, 'sexe': 1},
                       lignes=[{'age': '30-39', 'sexe': '*'}, {'age': '30-39', 'sexe': '*'}, {'age': '30-39', 'sexe': '*'}],
                       supprimees=1)

    Ici : sans suppression, il faut masquer tout l'âge (niveau 4) pour que chaque ligne ait un double. En acceptant
    de retirer UNE ligne (le client de 52 ans), il suffit de tranches de 10 ans et de masquer le sexe : hauteur
    2 + 1 = 3 au lieu de 4 + 0 = 4.

    Les règles :

    - `hierarchies` = {quasi-identifiant: [f0, f1, …]} ; fn(valeur) donne la valeur généralisée au niveau n. f0 est
      la valeur exacte, le dernier niveau met tout le monde ensemble (« * »). L'ordre des clés est celui des
      quasi-identifiants. Appelez-les sur `ligne.get(q)` : elles savent traiter une valeur manquante ;
    - un VECTEUR de niveaux donne un niveau par quasi-identifiant ; sa HAUTEUR est la somme des niveaux. La
      généralisation est GLOBALE : le même niveau pour toutes les lignes (pas seulement pour celles qui posent
      problème) ;
    - un vecteur est ACCEPTABLE si, une fois toutes les lignes généralisées, les lignes dont la classe (mêmes valeurs
      généralisées) compte moins de k lignes sont au plus `suppression_max` : on compte des LIGNES, pas des classes.
      Ces lignes-là sont retirées du fichier (« supprimées ») ;
    - on choisit le vecteur acceptable de plus petite hauteur ; à hauteur égale, le premier dans l'ordre
      lexicographique (celui de `itertools.product`). Il faut essayer TOUS les vecteurs d'une hauteur avant de
      passer à la suivante : généraliser pas à pas l'attribut qui a le plus de valeurs différentes (l'approche
      gloutonne de Datafly) monte souvent plus haut que nécessaire ;
    - niveaux = {quasi-identifiant: niveau choisi} ; lignes = pour chaque ligne gardée, dans l'ordre, une COPIE où
      chaque quasi-identifiant est remplacé par sa valeur généralisée (les autres colonnes intactes) ; supprimees =
      le nombre de lignes retirées ;
    - `hierarchies` vide → ValueError ; k doit être un entier entre 1 et len(lignes), suppression_max un entier
      >= 0 et < len(lignes), sinon ValueError (True n'est pas un entier ici) ; si aucun vecteur n'est acceptable
      (une hiérarchie dont le dernier niveau ne réunit pas tout), ValueError ;
    - ne modifiez pas `lignes`.

    Ce que vous apprenez : la généralisation globale et la suppression (l'algorithme de P. Samarati, 2001, base des
    outils comme ARX), parcourir toutes les combinaisons avec `itertools.product`, un tri stable par hauteur, et le
    vrai prix de l'anonymat : sur un petit fichier, il faut sacrifier beaucoup.

    Indice : `vecteurs = sorted(itertools.product(*(range(len(h)) for h in hierarchies.values())), key=sum)` donne
    les vecteurs par hauteur, et dans l'ordre lexicographique à hauteur égale. Pour chacun : les clés généralisées de
    chaque ligne, un `collections.Counter` des clés, et le nombre de lignes dont la clé compte moins de k.
    """
    raise NotImplementedError


# ============================================================================ Publier des chiffres


def secret_statistique(tableau: list[list[int]], seuil: int) -> list[list[bool]]:
    """Défi 210 🟡 — Le masque des cases à ne pas publier : les petites, et de quoi empêcher de les recalculer.

        >>> secret_statistique([[12, 2, 30], [8, 15, 4], [20, 9, 11]], 3)
        [[True, True, False], [True, False, True], [False, True, True]]

    Le tableau est publié avec le total de chaque ligne et de chaque colonne. Ici, seul le 2 est à cacher (moins de
    3 personnes). Mais sa ligne publie son total (44) : avec 12 et 30, on retrouve 2. On cache donc aussi 12, la plus
    petite case visible de la ligne. Mais la colonne de 12 n'a alors qu'une case cachée : on cache 8, et ainsi de
    suite. Un seul 2 à protéger, et six cases finissent cachées : chaque ligne et chaque colonne en a deux, et
    plus aucune ne se retrouve par soustraction.

    Les règles :

    - `tableau` : une liste de lignes de comptages (des entiers >= 0), toutes de même longueur ;
    - le secret PRIMAIRE : une case qui vaut de 1 à seuil − 1 est cachée. Une case à 0 n'expose personne : elle reste
      visible ;
    - le secret SECONDAIRE : tant qu'une ligne ou une colonne contient EXACTEMENT UNE case cachée, prenez la PREMIÈRE
      (les lignes de haut en bas d'abord, puis les colonnes de gauche à droite), cachez-y la plus petite case encore
      visible (à égalité, la première de la ligne ou de la colonne), puis RECOMMENCEZ depuis la première ligne : une
      case cachée pour une colonne peut laisser une LIGNE avec une seule case cachée ;
    - une ligne ou une colonne avec 0 ou au moins 2 cases cachées est sûre (pour ce correcteur) ;
    - le résultat est une NOUVELLE liste de listes de booléens (True = cachée), de la forme du tableau ;
    - moins de 2 lignes ou de 2 colonnes, lignes de longueurs différentes, comptage qui n'est pas un entier >= 0
      (True et 2.0 non plus), seuil qui n'est pas un entier >= 1 → ValueError ; ne modifiez pas `tableau`.

    Ce que vous apprenez : le secret statistique (le seuil dépend du domaine : 3 est courant en statistique
    d'entreprise, 11 pour les données de santé du SNDS), le secret secondaire et ses cascades, et une boucle qui
    tourne jusqu'à ce que plus rien ne bouge (un « point fixe »). Les outils professionnels (τ-ARGUS) vont plus loin
    : ils vérifient par programmation linéaire que chaque case cachée garde une fourchette assez large.

    Indice : `cache = [[1 <= x < seuil for x in ligne] for ligne in tableau]`, puis une boucle `while True:` qui
    cherche la première rangée à une seule case cachée (une rangée = la liste des positions (i, j) d'une ligne ou
    d'une colonne), cache sa plus petite case visible avec `min` sur des tuples (valeur, position, (i, j)), et
    s'arrête quand il n'y en a plus.
    """
    raise NotImplementedError


class Budget:
    """Défi 211 🔴 — Répondre aux questions du laboratoire avec un bruit de Laplace, sans dépasser le budget.

        >>> budget = Budget(1.0, tirages_fixes([0.5, 0.75]))
        >>> budget.compter(clients_exemple(), lambda c: c["motif"] == "impayé", 0.5)    # u = 0,5 : bruit nul
        5.0
        >>> round(budget.compter(clients_exemple(), lambda c: c["churn"] == 1, 0.25), 3)   # 11 + 4 × ln(2)
        13.773
        >>> budget.restant
        0.25
        >>> budget.compter(clients_exemple(), lambda c: True, 0.5)     # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        BudgetEpuise: il reste 0.25 de budget, la question en demande 0.5

    Ici : le 1er tirage vaut 0,5 et donne un bruit nul, 5 clients sont partis pour impayé. Le 2e vaut 0,75 : avec
    epsilon = 0,25, l'échelle du bruit est 1 / 0,25 = 4, et le bruit 4 × ln(2) ≈ 2,77. La 3e question coûterait 0,5,
    il n'en reste que 0,25 : refusée, sans rien tirer ni rien dépenser.

    Les règles :

    - `Budget(epsilon_total, tirage)` : epsilon_total est un nombre > 0 (ni NaN, ni l'infini, ni True), sinon
      ValueError. `tirage` est une fonction sans argument qui renvoie un nombre uniforme dans ]0, 1[ : c'est votre
      SEULE source de hasard (pas le module `random` : le correcteur doit pouvoir rejouer vos tirages) ;
    - `compter(lignes, condition, epsilon)` → le nombre de lignes pour lesquelles `condition(ligne)` est vraie, PLUS
      un bruit de Laplace d'échelle b = 1 / epsilon. Un client de plus ou de moins change un comptage d'au plus 1
      (sa SENSIBILITÉ) : plus epsilon est petit, plus le bruit est grand ;
    - `sommer(lignes, colonne, plafond, epsilon)` → la somme des valeurs de `colonne`, chacune ramenée dans
      [0, plafond] (une valeur manquante, None, NaN ou absente, compte 0), PLUS un bruit d'échelle plafond / epsilon.
      Sans plafond, un seul client à 10 000 € changerait la somme de 10 000 € : sa facture se lirait dans la
      réponse. plafond doit être un nombre > 0, sinon ValueError ;
    - le bruit : UN tirage u = tirage(), puis  bruit = −b × signe(u − 0,5) × ln(1 − 2 × |u − 0,5|),  où signe vaut
      +1 si u >= 0,5 et −1 sinon (`math.copysign(1, u - 0.5)`). C'est la réciproque de la fonction de répartition
      de Laplace : un nombre uniforme devient un nombre de Laplace. Le résultat est un float, non arrondi ;
    - chaque question acceptée coûte son epsilon. Si elle ferait dépasser le budget → BudgetEpuise, AVANT tout
      tirage et sans rien dépenser. Comparez à 1e-9 près : 0.1 + 0.1 + 0.1 vaut 0.30000000000000004, et un budget de
      0,3 doit permettre trois questions à 0,1 (ou tenez les comptes avec `fractions.Fraction(str(epsilon))`) ;
    - epsilon <= 0, NaN, infini ou pas un nombre → ValueError, sans rien tirer ni rien dépenser ;
    - `restant` : un attribut (ou une `@property`) qui vaut ce qu'il reste du budget, jamais négatif.

    Ce que vous apprenez : la confidentialité différentielle (C. Dwork, 2006 ; utilisée par le recensement américain
    de 2020), la sensibilité d'une question et pourquoi on plafonne une somme, transformer un nombre uniforme en un
    nombre d'une autre loi (la méthode d'inversion), la composition (les epsilons s'additionnent), et un budget en
    virgule flottante qui ne vous vole pas une question.

    Indice : gardez `self._total`, `self._depense = 0.0` et `self._tirage`. Une méthode `_payer(epsilon)` vérifie
    epsilon puis le budget, et ajoute epsilon à la dépense ; une méthode `_bruit(echelle)` fait UN tirage. `compter`
    et `sommer` appellent `_payer`, calculent la vraie réponse, et ajoutent `_bruit`.
    """

    def __init__(self, epsilon_total: float, tirage) -> None:
        raise NotImplementedError

    @property
    def restant(self) -> float:
        raise NotImplementedError

    def compter(self, lignes, condition, epsilon: float) -> float:
        raise NotImplementedError

    def sommer(self, lignes, colonne: str, plafond: float, epsilon: float) -> float:
        raise NotImplementedError

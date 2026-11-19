"""Série 38 — Traiter tout le monde de la même façon : mesurer, comparer, standardiser, chercher les indices, tester, corriger.

⚖️  VOUS VOULEZ DÉVELOPPER ? UN JOUR, VOTRE CODE DÉCIDERA POUR DES GENS. ICI, VOUS APPRENEZ À VÉRIFIER QU'IL LES
TRAITE ÉQUITABLEMENT.

Depuis novembre, le modèle de churn décide qui reçoit l'offre de fidélité (−20 % pendant six mois) : tout client
dont le score atteint 0.5 la reçoit. Mme Roux (C00194), 71 ans, a écrit au service client : « Mon voisin a reçu la
remise, pas moi. Est-ce parce que je suis âgée ? » Elle est partie en décembre. Le service juridique demande un
audit chiffré avant la prochaine campagne, et Inès, qui a construit le modèle, doit y répondre. Le modèle est
« bon » en moyenne (son AUC n'a pas bougé). Mais est-il bon pour TOUT LE MONDE ? Cette série vous fait écrire
les six mesures d'un audit d'équité :

    230 taux_par_groupe          qui reçoit l'offre, et qui le modèle repère parmi ceux qui partent, groupe par groupe
    231 quatre_cinquiemes        la règle des 4/5 : un groupe est-il nettement moins bien servi que le mieux servi ?
    232 taux_standardises        l'écart vient-il de l'âge, ou des contrats que chaque âge choisit ? (Simpson)
    233 proxies                  retirer la colonne « âge » suffit-il ? Ce que les autres colonnes en disent
    234 testing                  le test de situation : la même cliente, avec 40 ans, aurait-elle eu l'offre ?
    235 seuils_egalite_chances   ce qu'il faudrait pour repérer 60 % des départs dans chaque groupe

Prérequis : la série 00 (fonctions, listes, dictionnaires, boucles). La série 03 (précision, rappel, seuil), la
série 26 (comparer des taux sans se tromper) et la série 34 (les valeurs absentes, NaN) l'éclairent, mais tout ce
qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- le modèle SÉLECTIONNE un client (lui envoie l'offre) quand score >= seuil. Parmi les sélectionnés, certains
  sont vraiment partis ensuite (VRAIS POSITIFS), d'autres sont restés (FAUX POSITIFS) ;
- quatre taux, chacun avec SON dénominateur : le taux de SÉLECTION (sélectionnés / effectif du groupe), le taux
  de VRAIS POSITIFS ou RAPPEL (vrais positifs / partis : « parmi ceux qui partent, combien le modèle en
  repère-t-il ? »), le taux de FAUX POSITIFS (faux positifs / restés) et la PRÉCISION (vrais positifs /
  sélectionnés). Un taux dont le dénominateur est nul vaut None (« pas mesurable »), pas 0.0 (« mesuré, et nul ») ;
- un attribut PROTÉGÉ (l'âge, ici) est une caractéristique sur laquelle on ne doit pas traiter les gens
  différemment sans une raison solide. Mesurer l'équité, c'est comparer les taux d'un groupe à l'autre :
  l'« impact disproportionné » compare les taux de sélection, l'« égalité des chances » les taux de vrais positifs ;
- une valeur ABSENTE (le client n'a pas donné son âge) est None, ou NaN quand les données viennent de pandas.
  `float("nan") == float("nan")` est False : deux NaN ne sont pas la même clé de dict. `x != x` n'est vrai que
  pour un NaN ;
- `fractions.Fraction(2, 3)` est un nombre EXACT ; `Fraction("0.55")` vaut exactement 11/20. Les floats, eux,
  arrondissent : `0.55 * 100` vaut 55.00000000000001, et (20 / 30) / (25 / 30) vaut 0.7999999999999999.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_38 import *
    >>> clients = clients_exemple()
    >>> len(clients), sum(c["score"] >= SEUIL for c in clients), clients[193]["age"], clients[193]["score"]
    (6000, 1813, '65+', 0.413)
    >>> modele([clients[193], {**clients[193], "age": "30-44"}])       # Mme Roux, avec 40 ans
    [0.413, 0.587]

Quand les six défis sont réussis, Inès peut répondre au service juridique :

    >>> groupes = taux_par_groupe(clients, "age")
    >>> {g: round(x.taux_selection, 2) for g, x in groupes.items()}
    {'30-44': 0.35, '65+': 0.11, '18-29': 0.42, '45-64': 0.29, None: 0.35}
    >>> {g: round(x.tvp, 2) for g, x in groupes.items()}              # parmi ceux qui sont partis
    {'30-44': 0.55, '65+': 0.26, '18-29': 0.63, '45-64': 0.52, None: 0.65}
    >>> quatre_cinquiemes(groupes).signales
    ['65+', '45-64']
    >>> {g: round(t, 2) for g, t in taux_standardises(clients, "age", "contrat").standardise.items()}
    {'30-44': 0.32, '65+': 0.19, '18-29': 0.31, '45-64': 0.32, None: 0.36}
    >>> [(p.colonne, round(p.v, 2)) for p in proxies(clients, "age", COLONNES)][:2]
    [('paiement', 0.28), ('contrat', 0.2)]
    >>> [(b.testee, b.gagnent) for b in testing(modele, clients, "age", AGES) if b.origine == "65+"]
    [('18-29', 98), ('30-44', 98), ('45-64', 98)]
    >>> {g: r.seuil for g, r in seuils_egalite_chances(clients, "age", 0.6).items()}
    {'30-44': 0.463, '65+': 0.214, '18-29': 0.562, '45-64': 0.45, None: 0.562}

Les 65 ans et plus reçoivent l'offre presque quatre fois moins souvent que les 18-29 ans, et le modèle ne repère
qu'un sur quatre de ceux qui partent (contre plus d'un sur deux ailleurs). L'écart des 45-64 ans s'explique par
leurs contrats (standardisé, il disparaît) ; celui des 65 ans et plus, non. Retirer la colonne « âge » ne
suffirait pas (le paiement par chèque la trahit), et le test de situation le prouve : 98 clients de 65 ans et
plus, dont Mme Roux, auraient eu l'offre avec n'importe quel autre âge. La cause est trouvée : le modèle a appris
en 2024 que les seniors partaient peu, et ce n'est plus vrai. Son AUC chez les 65 ans et plus (0.78) vaut pourtant
celle des autres : il les CLASSE bien entre eux, mais les place tous trop bas. Aucune métrique moyenne ne l'aurait vu.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. L'ordre proposé est le plus progressif : le 231
   lit les `Groupe` du défi 230, mais le correcteur lui donne les siens (vous pouvez le faire avant).
2. Lancez `python scripts/defis.py --defi 230` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_38.py`.

Les outils communs (`clients_exemple`, `generer_clients`, `modele`, `selection`, `SEUIL`, `AGES`, `ZONES`,
`CONTRATS`, `PAIEMENTS`, `INTERNETS`, `COLONNES`, et les résultats `Groupe`, `Impact`, `Standardisation`, `Proxy`,
`Bascule`, `Reglage`) sont dans `defis/equite.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import math  # noqa: F401 — math.isnan, math.sqrt
from collections import Counter  # noqa: F401 — Counter(zip(a, b)) compte des paires
from fractions import Fraction  # noqa: F401 — Fraction(2, 3), Fraction("0.55") : des nombres exacts

from defis.equite import (  # noqa: F401
    AGES,
    COLONNES,
    CONTRATS,
    INTERNETS,
    PAIEMENTS,
    SEUIL,
    ZONES,
    Bascule,
    Groupe,
    Impact,
    Proxy,
    Reglage,
    Standardisation,
    clients_exemple,
    generer_clients,
    modele,
    selection,
)

# ============================================================================ Mesurer


def taux_par_groupe(clients: list[dict], colonne: str, seuil: float = SEUIL) -> dict:
    """Défi 230 🟢 — {valeur de `colonne`: Groupe(...)} : ce que le modèle a décidé pour chaque groupe.

        >>> clients = [{"age": "65+", "score": 0.31, "parti": 1}, {"age": "65+", "score": 0.52, "parti": 0},
        ...            {"age": "30-44", "score": 0.5, "parti": 1}, {"age": "30-44", "score": 0.12, "parti": 0},
        ...            {"age": None, "score": 0.8, "parti": 1}, {"age": float("nan"), "score": 0.2, "parti": 1}]
        >>> r = taux_par_groupe(clients, "age")
        >>> list(r)
        ['65+', '30-44', None]
        >>> r["65+"]
        Groupe(effectif=2, selectionnes=1, positifs=1, vrais_positifs=0, faux_positifs=1, taux_selection=0.5, tvp=0.0, tfp=1.0, precision=0.0)
        >>> r["30-44"].taux_selection, r["30-44"].tvp       # 0.5 >= 0.5 : sélectionné
        (0.5, 1.0)
        >>> r[None].effectif, r[None].tvp, r[None].tfp     # deux partis, aucun resté : pas de taux de faux positifs
        (2, 0.5, None)

    Ici : la cliente de 65 ans et plus qui est partie avait un score de 0.31, elle n'a rien reçu (tvp = 0.0) ;
    l'offre est allée à l'autre, qui serait resté de toute façon (tfp = 1.0). Le client sans âge et celle dont
    l'âge vaut NaN forment ENSEMBLE le groupe None : sans lui, deux clients disparaîtraient de l'audit.

    Les règles :

    - un groupe par valeur de `client[colonne]`, dans l'ordre où chaque valeur apparaît pour la première fois ;
    - une valeur ABSENTE (clé manquante, None ou NaN) forme le groupe None. Tous les absents ensemble : deux NaN
      ne sont pas égaux, et `pandas.groupby` jette les absents par défaut (`dropna=True`) ;
    - un client est SÉLECTIONNÉ si son score >= seuil (0.5 tout rond reçoit l'offre). Il est POSITIF s'il est
      parti (parti == 1) ;
    - chaque groupe donne Groupe(effectif, selectionnes, positifs, vrais_positifs, faux_positifs,
      taux_selection, tvp, tfp, precision), avec : taux_selection = selectionnes / effectif ; tvp = vrais_positifs
      / positifs ; tfp = faux_positifs / (effectif − positifs) ; precision = vrais_positifs / selectionnes. Les
      taux sont des float, ou None quand le dénominateur est nul ;
    - erreurs : seuil qui n'est pas un nombre (ou un bool) → TypeError, hors de [0, 1] → ValueError ; un score qui
      n'est pas un nombre (ou un bool) → TypeError, NaN ou hors de [0, 1] → ValueError ; parti qui ne vaut ni 0
      ni 1 → ValueError ;
    - aucun client → {} ; ne modifiez pas `clients`.

    Ce que vous apprenez : le bon dénominateur pour chaque taux (le rappel se compte parmi ceux qui PARTENT,
    pas parmi ceux qu'on sélectionne), et ne perdre personne en route.

    Indice : `valeur is None or (isinstance(valeur, float) and math.isnan(valeur))` ; un dict {groupe: [cinq
    compteurs]} rempli avec `setdefault` en une passe ; True + 1 vaut 2.
    """
    raise NotImplementedError


def quatre_cinquiemes(groupes: dict, min_effectif: int = 30) -> Impact:
    """Défi 231 🟢 — Impact(reference, ratios, signales, ignores) : la règle des quatre cinquièmes.

        >>> quatre_cinquiemes({"A": selection(25, 30), "B": selection(19, 30), "C": selection(3, 3)})
        Impact(reference='A', ratios={'A': 1.0, 'B': 0.76}, signales=['B'], ignores=['C'])
        >>> quatre_cinquiemes({"A": selection(25, 30), "B": selection(20, 30)})    # 4/5 tout juste : pas signalé
        Impact(reference='A', ratios={'A': 1.0, 'B': 0.8}, signales=[], ignores=[])
        >>> impact = quatre_cinquiemes(taux_par_groupe(clients_exemple(), "age"))
        >>> impact.reference, impact.signales, round(impact.ratios["65+"], 2)
        ('18-29', ['65+', '45-64'], 0.26)

    Ici (1er exemple) : le groupe A est le mieux servi (25 sur 30). B reçoit l'offre 19 fois sur 30 : 0.76 fois
    le taux de A, moins de 4/5 : signalé. C a 3 sélectionnés sur 3 (100 %), mais trois clients ne prouvent rien :
    il est mis de côté, et surtout il ne devient pas la référence (sinon A serait signalé à son tour). 2e exemple :
    20 / 30 contre 25 / 30, c'est 4/5 tout juste, donc PAS « moins de 4/5 », même si le float calcule
    (20 / 30) / (25 / 30) = 0.7999999999999999.

    Les règles (la « règle des quatre cinquièmes » des contrôleurs américains de l'emploi, reprise par les audits
    d'algorithmes : un repère pour repérer, pas un verdict) :

    - `groupes` est un dict {groupe: Groupe} comme celui du défi 230 ; seuls `effectif` et `selectionnes` servent ;
    - un groupe d'effectif < min_effectif n'est PAS jugé : il va dans `ignores` (dans l'ordre du dict) ;
    - la référence est le groupe jugé au plus fort taux de sélection (le premier dans l'ordre du dict à égalité) ;
      aucun groupe jugé → Impact(None, {}, [], ignores) ;
    - ratios : pour chaque groupe jugé, dans l'ordre du dict, son taux divisé par celui de la référence (un
      float ; la référence vaut 1.0). Si la référence est à 0 (personne n'a rien reçu), chaque ratio vaut None et
      personne n'est signalé ;
    - signales : les groupes jugés dont le ratio est STRICTEMENT inférieur à 4/5, dans l'ordre du dict. Décidez en
      ENTIERS : s / n < 4/5 × s_ref / n_ref s'écrit 5 × s × n_ref < 4 × s_ref × n ;
    - erreurs : min_effectif qui n'est pas un int (ou un bool) → TypeError, < 1 → ValueError ; un groupe dont
      les sélectionnés sont négatifs ou dépassent l'effectif → ValueError ; ne modifiez pas `groupes`.

    Ce que vous apprenez : comparer chaque groupe au MIEUX servi (pas à la moyenne), mettre de côté les groupes
    trop petits pour conclure, et prendre une décision à une limite sans laisser un arrondi décider pour vous.

    Indice : comparer s1 / n1 > s2 / n2 sans diviser : s1 × n2 > s2 × n1 ; `Fraction(s * n_ref, n * s_ref)`
    donne le ratio exact (et `float(...)` le ratio à renvoyer).
    """
    raise NotImplementedError


def taux_standardises(clients: list[dict], colonne: str, strate: str, seuil: float = SEUIL) -> Standardisation:
    """Défi 232 🟡 — Standardisation(brut, standardise, strates, exclues) : comparer à composition égale.

        >>> def lot(g, s, n, sel):    # n clients du groupe g dans la strate s, dont sel reçoivent l'offre
        ...     return [{"g": g, "s": s, "score": 0.9}] * sel + [{"g": g, "s": s, "score": 0.1}] * (n - sel)
        >>> clients = lot("A", "x", 30, 27) + lot("A", "y", 10, 1) + lot("B", "x", 10, 10) + lot("B", "y", 30, 6)
        >>> r = taux_standardises(clients, "g", "s")
        >>> r.brut, r.standardise
        ({'A': 0.7, 'B': 0.4}, {'A': 0.5, 'B': 0.6})
        >>> taux_standardises(clients + lot("B", "z", 5, 5), "g", "s").exclues     # z n'existe que chez B
        ['z']
        >>> r = taux_standardises(clients_exemple(), "age", "contrat")
        >>> {g: round(t, 2) for g, t in r.standardise.items()}, r.strates
        ({'30-44': 0.32, '65+': 0.19, '18-29': 0.31, '45-64': 0.32, None: 0.36}, ['Annuel', 'Mensuel', 'Biannuel'])

    Ici (1er exemple) : B est mieux servi que A dans CHAQUE strate (100 % contre 90 % en x, 20 % contre 10 % en
    y), et pourtant moins bien servi au total (40 % contre 70 %) : la plupart des clients de B sont dans la strate
    y, où l'offre est rare. C'est le PARADOXE DE SIMPSON. Standardiser, c'est donner aux deux groupes la même
    composition (celle de toute la clientèle : 40 clients en x et 40 en y, donc moitié-moitié) : A vaut
    0.5 × 0.9 + 0.5 × 0.1 = 0.5 et B 0.5 × 1.0 + 0.5 × 0.2 = 0.6.

    Les règles (la « standardisation directe » des épidémiologistes) :

    - groupes (selon `colonne`) et strates (selon `strate`) dans l'ordre d'apparition ; une valeur absente (clé
      manquante, None ou NaN) forme le groupe ou la strate None ; sélectionné = score >= seuil ;
    - brut : {groupe: sélectionnés / effectif du groupe} ;
    - les strates COMMUNES sont celles où chaque groupe a au moins un client (dans l'ordre d'apparition) ; les
      autres sont `exclues` (dans l'ordre d'apparition) : on ne peut pas y comparer tout le monde ;
    - standardise : {groupe: Σ poids(s) × taux du groupe dans s} sur les strates communes, où poids(s) = (clients
      de TOUS les groupes dans s) / (clients de toutes les strates communes). Les poids font 1 ;
    - aucun client → Standardisation({}, {}, [], []) ; aucune strate commune → ValueError ; colonne == strate →
      ValueError ; seuil et scores comme au défi 230 (`parti` ne sert pas ici) ; ne modifiez pas `clients`.
    - le correcteur vous passe 120 000 clients : comptez tout en UNE passe (pas une passe par groupe et par
      strate).

    Ce que vous apprenez : séparer l'effet d'un attribut de celui de la composition des groupes, et pourquoi un
    écart brut n'accuse personne tant qu'on ne l'a pas standardisé (ni ne l'innocente, d'ailleurs).

    Indice : `Counter()` indexé par (groupe, strate) pour les effectifs et les sélectionnés, un `Counter` par
    strate ; un dict sert d'ensemble ordonné (`d.setdefault(cle, None)`).
    """
    raise NotImplementedError


# ============================================================================ Chercher la cause


def proxies(clients: list[dict], protegee: str, colonnes) -> list[Proxy]:
    """Défi 233 🟡 — [Proxy(colonne, v, modalites)] : ce que chaque colonne dit de l'attribut protégé.

        >>> clients = [{"a": 1, "x": x, "y": y} for x, y in (("u", "p"), ("u", "q"), ("u", "p"), ("v", "q"))]
        >>> clients += [{"a": 2, "x": x, "y": y} for x, y in (("v", "p"), ("v", "q"), ("v", "p"), ("u", "q"))]
        >>> proxies(clients, "a", ["y", "x"])
        [Proxy(colonne='x', v=0.5, modalites=2), Proxy(colonne='y', v=0.0, modalites=2)]
        >>> [(p.colonne, round(p.v, 2)) for p in proxies(clients_exemple(), "age", COLONNES)]
        [('paiement', 0.28), ('contrat', 0.2), ('internet', 0.2), ('zone', 0.15), ('appels', 0.03)]

    Ici (1er exemple) : `y` ne dit rien de `a` (chaque valeur de a a autant de p que de q) : V = 0. `x` en dit
    beaucoup sans tout dire : trois fois sur quatre, « u » va avec a = 1 et « v » avec a = 2 : V = 0.5 (le
    tableau croisé [[3, 1], [1, 3]] donne χ² = 2, et racine(2 / (8 × 1)) = 0.5). Sur les clients de novembre, le moyen de
    paiement trahit l'âge plus que tout le reste (44 % des 65 ans et plus paient par chèque, 2 % des 18-29 ans) :
    retirer la colonne « âge » du modèle ne suffirait pas, il retrouverait l'âge par le chèque.

    Les règles (le V de Cramér, sans correction) :

    - pour chaque colonne, on croise ses valeurs avec celles de `protegee` : n_ij = nombre de clients qui ont la
      valeur i de `protegee` et la valeur j de la colonne ; a_i et b_j sont les totaux de ligne et de colonne, n
      le nombre de clients. Une valeur absente (clé manquante, None ou NaN) est une valeur comme une autre (None) ;
    - χ² = Σ (n_ij − e_ij)² / e_ij avec e_ij = a_i × b_j / n, sur TOUTES les cases du tableau croisé, y compris
      celles où personne n'est (n_ij = 0, mais e_ij > 0 compte quand même) ;
    - V = racine(χ² / (n × (k − 1))) avec k = min(nombre de valeurs de `protegee`, nombre de valeurs de la
      colonne). Si k vaut 1 (une seule valeur d'un côté), V = 0.0 ;
    - le résultat : un Proxy(colonne, v, modalites) par colonne, `modalites` = le nombre de valeurs différentes de
      la colonne, trié du V le plus FORT au plus faible ; à égalité, dans l'ordre de `colonnes` ;
    - erreurs : `protegee` dans `colonnes`, ou une colonne demandée deux fois → ValueError ; aucun client →
      ValueError ; ne modifiez pas `clients`. Le correcteur vous passe 100 000 clients : comptez chaque tableau
      croisé en une passe (pas une passe par case).

    Ce que vous apprenez : l'« équité par ignorance » ne marche pas (les autres colonnes portent l'information),
    mesurer un lien entre deux variables qualitatives, et l'erreur classique des cases vides oubliées.

    Indice : `Counter(zip(valeurs_protegee, valeurs_colonne))` ; développé, χ² = n × (Σ n_ij² / (a_i × b_j) − 1),
    où seules les cases non vides apparaissent dans la somme (en float, un χ² nul peut alors sortir à −1e-16 :
    `max(chi2, 0.0)` avant la racine, ou des `Fraction`) ; `sorted(..., key=lambda p: -p.v)` est stable.
    """
    raise NotImplementedError


def testing(modele, clients: list[dict], colonne: str, valeurs, seuil: float = SEUIL) -> list[Bascule]:
    """Défi 234 🟡 — [Bascule(origine, testee, testes, gagnent, perdent)] : le test de situation.

        >>> clients = [{"age": "65+", "contrat": "Mensuel", "appels": 2},
        ...            {"age": "30-44", "contrat": "Mensuel", "appels": 2}, {"age": None, "contrat": "Annuel"}]
        >>> modele(clients)
        [0.401, 0.574, 0.109]
        >>> testing(modele, clients, "age", ["30-44", "65+"])
        [Bascule(origine='30-44', testee='65+', testes=1, gagnent=0, perdent=1), Bascule(origine='65+', testee='30-44', testes=1, gagnent=1, perdent=0)]
        >>> [b for b in testing(modele, clients_exemple(), "age", AGES) if b.origine == "65+"][0]
        Bascule(origine='65+', testee='18-29', testes=1105, gagnent=98, perdent=0)

    Ici (1er exemple) : le même client (contrat mensuel, deux appels au support) a 0.574 à 40 ans et 0.401 à
    65 ans et plus. Le premier perdrait l'offre s'il avait 65 ans ; le second la gagnerait s'il en avait 40. Le
    troisième n'a pas d'âge connu : il n'est pas testé.

    Les règles (le « testing », ou test de situation : deux dossiers identiques sauf UN détail) :

    - les clients testés sont ceux dont la valeur de `colonne` (absente → None) est dans `valeurs` ;
    - `modele` reçoit une LISTE de clients (des dict) et renvoie une liste de scores, un par client, dans l'ordre.
      Décision d'origine : modele(clients testés) >= seuil. Recalculez-la : le champ "score" peut dater d'une
      ancienne version du modèle ;
    - pour chaque valeur testée t, chaque client testé dont la valeur d'origine o est différente de t est copié
      avec `colonne` = t (rien d'autre ne change), et la copie est scorée ;
    - le résultat : une Bascule(o, t, testes, gagnent, perdent) par paire o != t, o dans l'ordre de `valeurs` puis
      t dans l'ordre de `valeurs` ; testes = le nombre de clients testés d'origine o ; gagnent = ceux qui n'ont
      pas l'offre avec o mais l'auraient avec t ; perdent = l'inverse ;
    - le modèle est derrière une API : appelez-le PAR LOTS, au plus len(valeurs) + 1 fois en tout (une fois pour
      les originaux, une fois par valeur testée), jamais une fois par client. S'il renvoie un nombre de scores
      différent du nombre de clients envoyés → ValueError ;
    - ne modifiez JAMAIS un client, même le temps d'un appel (`client[colonne] = t` puis remettre l'ancienne
      valeur : une exception au milieu, et le client est faux pour toujours). Envoyez des COPIES ;
    - erreurs : moins de deux valeurs, ou deux fois la même → ValueError ; seuil comme au défi 230.

    Ce que vous apprenez : prouver qu'un attribut change une décision « toutes choses égales par ailleurs »,
    appeler un service coûteux par lots, et ne jamais toucher aux données qu'on vous prête.

    Indice : `{**client, colonne: t}` est une copie avec une seule clé changée ; gardez les POSITIONS des clients
    envoyés dans chaque lot pour retrouver la décision d'origine de chacun (`zip(positions, scores)`).
    """
    raise NotImplementedError


# ============================================================================ Corriger


def seuils_egalite_chances(clients: list[dict], colonne: str, cible: float) -> dict:
    """Défi 235 🔴 — {groupe: Reglage(seuil, tvp, tfp, taux_selection)} : un seuil par groupe pour l'égalité des chances.

        >>> clients = [{"g": "A", "score": s / 1000, "parti": 1} for s in range(100, 200)]   # 100 partis
        >>> clients += [{"g": "A", "score": 0.15, "parti": 0}, {"g": "B", "score": 0.3, "parti": 0}]
        >>> seuils_egalite_chances(clients, "g", 0.55)
        {'A': Reglage(seuil=0.145, tvp=0.55, tfp=1.0, taux_selection=0.5544554455445545), 'B': None}
        >>> {g: r.seuil for g, r in seuils_egalite_chances(clients_exemple(), "age", 0.6).items()}
        {'30-44': 0.463, '65+': 0.214, '18-29': 0.562, '45-64': 0.45, None: 0.562}

    Ici (1er exemple) : pour repérer 55 % des 100 partis du groupe A, il en faut 55. Leurs scores vont de 0.199 à
    0.100 ; le 55e plus haut est 0.145. Avec ce seuil, 55 partis sur 100 sont repérés (tvp = 0.55), et le seul
    resté (0.15) reçoit l'offre aussi (tfp = 1.0). Avec 0.146, on n'en repérerait que 54. Le groupe B n'a aucun
    parti : on ne peut pas y mesurer le rappel (None). 2e exemple : pour repérer 60 % des départs dans chaque
    groupe, il faudrait envoyer l'offre aux 65 ans et plus dès 0.214, et aux 18-29 ans à partir de 0.562.

    Les règles (le post-traitement de Hardt, Price et Srebro, 2016, « Equality of opportunity ») :

    - un groupe par valeur de `colonne`, dans l'ordre d'apparition, l'absence formant le groupe None (comme au
      défi 230) ;
    - pour chaque groupe, le seuil est le PLUS HAUT score t d'un client parti tel que (partis du groupe avec score
      >= t) / (partis du groupe) >= cible ;
    - « >= cible » se juge EXACTEMENT, avec la cible telle qu'elle est écrite : 0.55 veut dire 55/100, et
      55 partis sur 100 l'atteignent. En float, 0.55 × 100 vaut 55.00000000000001 : `math.ceil` en ferait 56, un
      parti de trop, et un seuil trop bas. `Fraction(str(0.55))` vaut exactement 11/20 ; `Fraction(0.55)`, lui,
      vaut 0.55000000000000004440… (le float tel qu'il est stocké), et se trompe aussi ;
    - avec ce seuil (score >= t), Reglage(seuil, tvp, tfp, taux_selection) sur TOUT le groupe : tvp = partis
      sélectionnés / partis ; tfp = restés sélectionnés / restés (None s'il n'y a aucun resté) ; taux_selection =
      sélectionnés / effectif. Un groupe sans aucun parti → None ;
    - cible : un nombre (int, float ou Fraction ; pas un bool, pas un texte → TypeError) dans ]0, 1] (sinon, ou
      NaN → ValueError) ; scores et parti comme au défi 230 ; ne modifiez pas `clients` ;
    - le correcteur vous passe un groupe de 60 000 partis : essayer chaque seuil possible et recompter les partis
      à chaque fois est bien trop lent. Triez une fois.

    Ce que vous apprenez : ce que coûte l'égalité des chances (des seuils DIFFÉRENTS selon le groupe : c'est
    traiter les gens différemment exprès, ce que le droit n'autorise pas toujours ; on le calcule pour MESURER
    l'écart, et souvent mieux vaut corriger la cause et réentraîner), et compter exactement près d'une limite.

    Indice : le plus petit k tel que k / P >= cible vaut ceil(cible × P), calculé en entiers avec
    `-(-numerateur * P // denominateur)` ; le seuil est alors le k-ième plus haut score de parti
    (`sorted(..., reverse=True)[k - 1]`).
    """
    raise NotImplementedError

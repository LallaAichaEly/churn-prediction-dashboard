"""Série 31 — Proposer la bonne offre : cooccurrences, règles, recommandation, évaluation, diversité.

🛒 VOUS VOULEZ DÉVELOPPER ? « LES CLIENTS QUI ONT PRIS CECI ONT AUSSI PRIS… » : ICI, VOUS L'ÉCRIVEZ.

Un client qui a plusieurs options part moins : dans `_churn_logit` (src/churn/data.py), chaque produit en
plus retire 0,25 au logit du churn. Nadia, de l'équipe marketing, veut donc proposer à chaque client
« l'offre suivante » : celle qu'il a le plus de chances de prendre. Proposer « mobile » à tout le monde ne
sert à rien (70 % l'ont déjà, et les autres n'en veulent pas). Il faut apprendre des 300 000 paniers ce qui
va ENSEMBLE, recommander client par client, puis prouver, chiffres à l'appui, que les recommandations
auraient été justes. Cette série vous fait écrire les six pièces d'un moteur de recommandation :

    188 cooccurrences          pour chaque paire d'options, le nombre de clients qui ont les deux
    189 regles_association     « si tv alors sport » : support, confiance et lift (et le piège de la popularité)
    190 recommander            les k options à proposer à UN client, avec un plan B quand on ne sait rien
    191 decouper_dans_le_temps le passé pour apprendre, l'avenir pour noter : jamais l'inverse
    192 mesurer                précision@k, rappel@k et NDCG@k : les recommandations étaient-elles justes ?
    193 diversifier            trois options presque identiques n'en valent pas une : la méthode MMR

Prérequis : la série 00 (listes, dict, set, tri avec `key`). La précision et le rappel (série 01) et la
validation dans le temps (série 21) l'éclairent, mais tout ce qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un PANIER, c'est la liste des options d'un client. Un vrai export a des défauts : une option écrite deux
  fois (export rejoué), un client sans option ([]), une case vide (None). On compte des CLIENTS, pas des
  lignes : `set(panier)` partout ;
- tout part d'un comptage : pour chaque paire d'options, chez combien de clients on les trouve ENSEMBLE
  (la cooccurrence). Un client à 3 options donne 3 paires ; il n'y a jamais besoin d'essayer toutes les
  paires d'un catalogue de 5 000 options ;
- « 75 % des clients qui ont tv ont aussi mobile » (la confiance) n'apprend rien si 64 % de TOUS les
  clients ont mobile. Le LIFT compare au hasard : 1 = aucun lien, 3 = trois fois plus souvent ensemble
  que si les clients choisissaient au hasard. Mais un lift calculé sur 1 client est une coïncidence :
  on exige un SUPPORT minimal (un nombre de clients) ;
- recommander à un client : additionner, pour chaque option qu'il n'a pas, sa similarité avec chacune
  de celles qu'il a, et garder les meilleures. Un client dont on ne sait rien reçoit les plus populaires ;
- évaluer : on fait comme si on était le 1er juin. On apprend sur ce qui s'est passé AVANT, on recommande,
  puis on regarde ce que chaque client a VRAIMENT pris ensuite. Couper au hasard, c'est laisser le modèle
  voir l'avenir ;
- une bonne liste est juste ET variée : « tv », « box2 » et « tv 4K » sont presque la même proposition.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_31 import *
    >>> paniers = paniers_exemple()
    >>> paniers["C13"], paniers["C08"]                       # « tv » deux fois : l'export a été rejoué
    (['mobile', 'tv', 'sport', 'box2'], ['mobile', 'tv', 'tv'])

Quand les six défis sont réussis, vous avez un petit moteur de recommandation :

    >>> sorted(cooccurrences(paniers)["tv"].items())           # chez combien de clients « tv » va avec…
    [('box2', 2), ('mobile', 3), ('sport', 3)]
    >>> regles_association(paniers, support_min=2)[0]
    Regle(si='antivirus', alors='cloud', support=2, confiance=1.0, lift=7.0)
    >>> sim = similarites_cosinus(paniers)
    >>> recommander(paniers["C01"], sim, populaires(paniers), k=2)   # C01 a mobile, tv et sport
    ['box2', 'gaming']
    >>> historique, a_deviner = decouper_dans_le_temps(abonnements_exemple(), dt.date(2026, 6, 1))
    >>> a_deviner
    {'C02': {'musique'}, 'C01': {'tv'}, 'C03': {'gaming'}}
    >>> recommandations = {"C01": ["sport", "tv", "box2"], "C02": ["gaming", "musique"]}
    >>> {nom: round(valeur, 4) for nom, valeur in mesurer(recommandations, a_deviner, k=3).items()}
    {'precision': 0.2222, 'rappel': 0.6667, 'ndcg': 0.4206}

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les six défis sont indépendants (le
   correcteur fournit lui-même les similarités dont les défis 190 et 193 ont besoin) ; le 189 peut appeler
   votre 188 ; l'ordre proposé est le plus progressif.
2. Lancez `python scripts/defis.py --defi 188` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_31.py`.

Les outils communs (`paniers_exemple`, `abonnements_exemple`, `populaires`, `similarites_cosinus`,
`Regle`, `SimilaritesComptees`, `generer_paniers`, `generer_abonnements`) sont dans `defis/offres.py`,
déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import datetime as dt
import math  # noqa: F401 — math.log2(x) : le logarithme en base 2 ; math.inf : plus grand que tout
from collections import Counter  # noqa: F401 — Counter() : un dict qui compte ; .update(un_set) ajoute 1 par élément

from defis.offres import (  # noqa: F401
    CATALOGUE,
    Regle,
    SimilaritesComptees,
    abonnements_exemple,
    generer_abonnements,
    generer_paniers,
    paniers_exemple,
    populaires,
    similarites_cosinus,
)

# ============================================================================ Compter ce qui va ensemble


def cooccurrences(paniers: dict[str, list[str] | None]) -> dict[str, dict[str, int]]:
    """Défi 188 🟢 — {a: {b: nombre de clients qui ont a ET b}} pour toutes les paires d'options.

        >>> cooc = cooccurrences(paniers_exemple())
        >>> sorted(cooc["tv"].items())             # C08 a « tv » deux fois : il ne compte qu'une fois
        [('box2', 2), ('mobile', 3), ('sport', 3)]
        >>> cooc["tv"]["sport"] == cooc["sport"]["tv"]
        True
        >>> cooc["presse"], "presse" in cooc["presse"]       # détenue, mais jamais avec une autre
        ({}, False)
        >>> cooccurrences({"A": ["tv", "mobile", "tv"], "B": ["mobile", "tv"], "C": None, "D": []})
        {'mobile': {'tv': 2}, 'tv': {'mobile': 2}}

    Les règles :

    - un client compte UNE fois par paire, même si l'export a écrit une option deux fois ;
    - le résultat est SYMÉTRIQUE : `cooc[a][b] == cooc[b][a]`, et ne contient jamais de paire (a, a) ;
    - toute option détenue par au moins un client est une clé, même si elle n'a aucune voisine (sa valeur
      est alors {}) : le défi 190 lira `cooc[option]` pour chaque option d'un client ;
    - seules les paires vues sont stockées (pas de 0) ; le résultat est un vrai `dict` de vrais `dict` (un
      `defaultdict` ajouterait une case vide à chaque paire qu'on CONSULTE sans l'avoir vue) ;
    - un panier None (case vide) ou [] ne donne rien, sans planter ; ne modifiez pas `paniers` ;
    - le catalogue peut compter des milliers d'options, et les clients des centaines de milliers : le
      correcteur compte les lignes exécutées. Essayer toutes les paires du catalogue, ou comparer les
      clients deux à deux, est refusé. Un client à k options donne k(k-1)/2 paires : c'est tout.

    Ce que vous apprenez : la matrice de cooccurrence, la base des « vous aimerez aussi » (et des
    co-occurrences de mots en traitement du langage), et compter ce qui EXISTE plutôt que tout essayer.

    Indice : pour chaque panier non vide, `offres = sorted(set(panier))`, puis
    `for i, a in enumerate(offres): for b in offres[i + 1:]:` et ajoutez 1 dans les DEUX sens
    (`cooc.setdefault(a, {})`, puis `cooc[a][b] = cooc[a].get(b, 0) + 1`).
    """
    raise NotImplementedError


def regles_association(paniers: dict[str, list[str] | None], support_min: int = 2) -> list[Regle]:
    """Défi 189 🟡 — Les règles « si → alors » vues chez au moins `support_min` clients, les plus fortes d'abord.

        >>> regles = regles_association(paniers_exemple(), support_min=2)
        >>> regles[0]
        Regle(si='antivirus', alors='cloud', support=2, confiance=1.0, lift=7.0)
        >>> par_paire = {(r.si, r.alors): r for r in regles}
        >>> r = par_paire["tv", "mobile"]                  # 75 % des clients tv ont mobile…
        >>> r.confiance, round(r.lift, 3)                  # … mais 64 % de TOUS les clients l'ont : lien faible
        (0.75, 1.167)
        >>> r = par_paire["musique", "gaming"]             # moins de confiance, mais un vrai lien
        >>> round(r.confiance, 3), round(r.lift, 3)
        (0.667, 3.111)
        >>> round(par_paire["mobile", "tv"].confiance, 3)  # la confiance dépend du SENS de la règle
        0.333
        >>> ("assurance", "cloud") in par_paire            # lift 7… sur UN seul client : écartée
        False
        >>> len(regles), len(regles_association(paniers_exemple(), support_min=1))
        (18, 28)

    Les définitions, avec N le nombre de clients, n(a) le nombre de clients qui ont a, n(a, b) ceux qui ont
    les deux (un client compte une fois, même si l'export répète une option) :

    - `support` = n(si, alors), un NOMBRE de clients (pas une proportion) ;
    - `confiance` = n(si, alors) / n(si) : parmi les clients qui ont `si`, la part qui a aussi `alors` ;
    - `lift` = n(si, alors) × N / (n(si) × n(alors)) : combien de fois plus souvent que le hasard. N compte
      TOUS les clients de `paniers`, y compris ceux sans option ([] ou None) : ils auraient pu acheter ;

    Les règles :

    - chaque paire donne DEUX règles, une dans chaque sens (a → b et b → a) : même support, même lift,
      confiances différentes. Jamais de règle a → a ;
    - seules les règles de support >= `support_min` sont gardées (« au moins ») ;
    - `support_min` doit être un entier >= 1, sinon ValueError (une règle vue chez zéro client n'existe pas) ;
    - triées par lift DÉCROISSANT, puis confiance DÉCROISSANTE, puis `si` et `alors` par ordre alphabétique ;
    - chaque règle est un `Regle(si, alors, support, confiance, lift)` (un tuple nommé, déjà importé) ; ne
      modifiez pas `paniers`.

    Ce que vous apprenez : l'analyse du panier de la ménagère (« market basket analysis »), le piège de la
    popularité (la confiance seule recommande à tous ce que tout le monde a déjà), et pourquoi une mesure
    impressionnante calculée sur un client ne prouve rien.

    Indice : `n = len(paniers)` ; `detenteurs = Counter()` puis `detenteurs.update(set(panier or ()))` pour
    chaque panier. Parcourez `cooccurrences(paniers)` (votre défi 188) : chaque `cooc[si][alors]` est un
    support. Triez avec `key=lambda r: (-r.lift, -r.confiance, r.si, r.alors)`, sans `reverse=True`.
    """
    raise NotImplementedError


# ============================================================================ Recommander


def recommander(panier: list[str] | None, voisins: dict[str, dict[str, float]], populaires: list[str],
                k: int = 3) -> list[str]:
    """Défi 190 🟡 — Les k options à proposer à un client : les plus proches des siennes, puis les populaires.

    `voisins[a][b]` est la similarité (positive) entre les options a et b (par exemple
    `similarites_cosinus(paniers)`) ; une paire absente vaut 0. `populaires` est la liste des options de la
    plus répandue à la moins répandue.

        >>> paniers = paniers_exemple()
        >>> sim, pop = similarites_cosinus(paniers), populaires(paniers)
        >>> recommander(["mobile", "tv", "sport"], sim, pop, k=4)     # gaming et musique à égalité
        ['box2', 'gaming', 'musique', 'antivirus']
        >>> recommander(["tv", "sport"], sim, pop, k=3)                # gaming : le plan B (populaire)
        ['box2', 'mobile', 'gaming']
        >>> recommander([], sim, pop, k=3), recommander(None, sim, pop, k=2)   # on ne sait rien
        (['mobile', 'tv', 'gaming'], ['mobile', 'tv'])
        >>> recommander(["tv", "tv", "mobile"], sim, pop, k=3)         # « tv » deux fois compte une fois
        ['sport', 'box2', 'gaming']

    Les règles :

    - le score d'une option que le client N'A PAS est la SOMME de ses similarités avec chacune des options
      qu'il a (chaque option comptée une fois, même si l'export l'a écrite deux fois) ;
    - une option que le client a déjà n'est JAMAIS proposée ; une option qu'il a mais qui n'a pas de ligne
      dans `voisins` (une option toute nouvelle) n'apporte rien, sans planter ;
    - d'abord les options de score > 0, par score DÉCROISSANT, puis par ordre alphabétique à égalité ; un
      score nul n'est pas une raison de proposer ;
    - s'il en manque pour arriver à k, complétez avec `populaires`, dans l'ordre, sans doublon et sans ce
      que le client a déjà. Le plan B vient APRÈS ce qu'on sait du client, jamais avant ;
    - au plus k options (moins s'il n'y en a pas assez) ; k = 0 → [] ; k < 0 → ValueError ;
    - ne modifiez ni `panier`, ni `voisins`, ni `populaires`.

    Ce que vous apprenez : le filtrage collaboratif « objet à objet » (celui d'Amazon depuis 2003), le
    démarrage à froid (« cold start ») et la règle d'or d'un système de recommandation en production : il
    répond TOUJOURS quelque chose de sensé.

    Indice : `detenues = set(panier or ())`. Pour chaque `offre` de `sorted(detenues)`, parcourez
    `voisins.get(offre, {}).items()` et ajoutez la similarité au score des options non détenues. Triez avec
    `key=lambda o: (-scores[o], o)` les options de score > 0, gardez k, puis complétez avec `populaires`.
    """
    raise NotImplementedError


# ============================================================================ Évaluer sans tricher


def decouper_dans_le_temps(abonnements: list[tuple[str, str, dt.date]], coupure: dt.date
                           ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Défi 191 🟢 — (historique, a_deviner) : le passé pour apprendre, les NOUVELLES options pour noter.

    `abonnements` est le journal des souscriptions : des tuples (client, option, date), DANS LE DÉSORDRE.

        >>> historique, a_deviner = decouper_dans_le_temps(abonnements_exemple(), dt.date(2026, 6, 1))
        >>> historique
        {'C01': {'mobile'}, 'C02': {'mobile'}, 'C03': set(), 'C04': {'tv'}}
        >>> a_deviner                       # C01 reprend « mobile » après la coupure : rien de nouveau
        {'C02': {'musique'}, 'C01': {'tv'}, 'C03': {'gaming'}}
        >>> "C04" in a_deviner              # rien pris après la coupure : rien à deviner
        False

    Les règles :

    - `historique[client]` : l'ensemble (set) des options souscrites STRICTEMENT AVANT la coupure. Tout
      client du journal y figure, même s'il n'a rien souscrit avant (set() : un client « froid », à qui il
      faudra quand même recommander quelque chose) ;
    - `a_deviner[client]` : l'ensemble des options souscrites le jour de la coupure ou après, que le client
      n'avait PAS avant (un réabonnement n'est pas une découverte). Seuls les clients qui ont au moins une
      option nouvelle y figurent ;
    - le journal est dans le désordre, avec des lignes en double : le résultat ne doit pas en dépendre. Une
      souscription d'avant la coupure peut apparaître APRÈS celle d'après dans la liste ;
    - un journal vide → ({}, {}) ; ne modifiez pas `abonnements` (pas de `.sort()` dessus).

    Ce que vous apprenez : la validation temporelle d'un système de recommandation. Un découpage au hasard
    laisse le modèle apprendre sur l'avenir et promet des résultats qu'il n'aura jamais en production ; noter
    les options déjà détenues le récompense pour proposer ce que le client a déjà.

    Indice : un premier passage remplit `historique` (setdefault pour CHAQUE client, add si `date < coupure`)
    et `apres` (sinon). Ensuite seulement, pour chaque client de `apres` : `apres[client] - historique[client]`.
    """
    raise NotImplementedError


def mesurer(recommandations: dict[str, list[str]], a_deviner: dict[str, set[str]], k: int = 3) -> dict[str, float]:
    """Défi 192 🔴 — {"precision": …, "rappel": …, "ndcg": …} : les moyennes des trois mesures @k.

    `recommandations[client]` est la liste ordonnée des options proposées ; `a_deviner[client]` l'ensemble
    (non vide) de celles qu'il a vraiment prises ensuite (le défi 191).

        >>> a_deviner = {"C01": {"tv"}, "C02": {"musique"}, "C03": {"gaming"}}
        >>> recommandations = {"C01": ["sport", "tv", "box2"], "C02": ["gaming", "musique"], "C04": ["sport"]}
        >>> scores = mesurer(recommandations, a_deviner, k=3)
        >>> {nom: round(valeur, 4) for nom, valeur in scores.items()}
        {'precision': 0.2222, 'rappel': 0.6667, 'ndcg': 0.4206}
        >>> mesurer({"A": ["tv", "box2"]}, {"A": {"tv", "box2"}}, k=2)
        {'precision': 1.0, 'rappel': 1.0, 'ndcg': 1.0}

    Ici, C01 : « tv » trouvée au rang 2 ; C02 : « musique » au rang 2 ; C03 : rien proposé, il compte 0 ;
    C04 n'avait rien à deviner : ignoré. Pour UN client, avec `touches` le nombre d'options des k premières
    proposées qui sont dans sa cible :

    - précision@k = touches / k (sur k, même si la liste est plus courte : proposer moins n'est pas
      récompensé) ;
    - rappel@k = touches / taille de la cible ;
    - NDCG@k = DCG / DCG idéal, où DCG = somme, pour chaque touche au rang r (r = 1 pour la première
      proposée), de 1 / log2(r + 1). Le DCG idéal est celui de la meilleure liste possible : des touches aux
      rangs 1, 2, …, jusqu'à min(k, taille de la cible). Une touche en tête vaut 1 ; au rang 2, 0,63.

    Les règles :

    - la moyenne porte sur TOUS les clients de `a_deviner` : un client sans recommandation (absent, ou liste
      vide) compte 0 pour les trois mesures. Les clients de `recommandations` absents de `a_deviner` sont
      ignorés ;
    - seules les k premières options proposées comptent ; une option proposée deux fois ne touche qu'une
      fois (la deuxième occupe une place pour rien) ;
    - k doit être un entier >= 1, et `a_deviner` non vide, sinon ValueError ;
    - le résultat est un dict de trois `float`, exactement ces trois clés ; ne modifiez rien.

    Ce que vous apprenez : les mesures d'un système de recommandation (et d'un moteur de recherche), pourquoi
    l'ORDRE compte (NDCG), et l'erreur qui embellit tous les tableaux de bord : oublier les clients pour qui
    on n'a rien proposé. (La base du logarithme ne change pas le NDCG : elle se simplifie dans la division.)

    Indice : pour chaque client de `a_deviner` : `liste = recommandations.get(client) or []`, puis
    `for rang, offre in enumerate(liste[:k], start=1)`, avec un set `vues` pour ne compter une option qu'une
    fois. Le DCG idéal : `sum(1 / math.log2(r + 1) for r in range(1, min(k, len(cible)) + 1))`.
    """
    raise NotImplementedError


# ============================================================================ Varier les propositions


def diversifier(candidats: dict[str, float], similarites: dict[str, dict[str, float]], k: int = 3,
                lam: float = 0.7) -> list[str]:
    """Défi 193 🔴 — Les k options choisies une à une : bien notées, mais pas trop proches des déjà choisies.

    `candidats[option]` est la note de l'option (son score de recommandation) ; `similarites[a][b]` est la
    similarité, entre 0 et 1, entre une candidate a et une option b déjà choisie (absente : 0 ; une
    candidate peut ne pas avoir de ligne du tout).

        >>> candidats = {"tv": 0.9, "box2": 0.85, "sport": 0.8, "musique": 0.5}
        >>> similarites = {"tv": {"box2": 0.9, "sport": 0.6}, "box2": {"tv": 0.9, "sport": 0.5},
        ...                "sport": {"tv": 0.6, "box2": 0.5}}
        >>> diversifier(candidats, similarites, k=3, lam=0.7)     # box2 ressemble trop à tv
        ['tv', 'sport', 'musique']
        >>> diversifier(candidats, similarites, k=3, lam=1.0)     # lam = 1 : seule la note compte
        ['tv', 'box2', 'sport']

    La méthode MMR (« maximal marginal relevance ») : à chaque tour, parmi les candidates pas encore
    choisies, prenez celle qui maximise

        valeur = lam × note − (1 − lam) × (sa similarité la PLUS FORTE avec une option déjà choisie)

    (au premier tour, rien n'est choisi : la pénalité vaut 0). Ici, au 2e tour : box2 vaut
    0,7 × 0,85 − 0,3 × 0,9 = 0,325 ; sport 0,7 × 0,8 − 0,3 × 0,6 = 0,38 ; musique 0,35. C'est sport.

    Les règles :

    - à valeur égale, la meilleure note d'abord, puis l'ordre alphabétique ;
    - au plus k options, chacune une fois ; k = 0 → [] ; k plus grand que le nombre de candidates → toutes ;
    - `lam` doit être entre 0 et 1 (inclus), sinon ValueError (NaN aussi) ; k < 0 → ValueError ;
    - ne modifiez ni `candidats` ni `similarites` ;
    - avec 3 000 candidates et k = 40, le correcteur vous donne des `SimilaritesComptees`, qui comptent les
      similarités lues. Relire à chaque tour la similarité de chaque candidate avec TOUTES les options déjà
      choisies, c'est 40 fois trop : gardez pour chaque candidate sa pénalité la plus forte jusqu'ici, et
      mettez-la à jour avec la SEULE option qu'on vient de choisir.

    Ce que vous apprenez : l'algorithme glouton de diversification des moteurs de recherche et des fils
    d'actualité, le compromis pertinence / diversité réglé par un seul paramètre, et la mise à jour
    incrémentale, qui transforme O(k² × n) en O(k × n).

    Indice : `restantes = sorted(candidats, key=lambda o: (-candidats[o], o))` et
    `penalite = dict.fromkeys(restantes, 0.0)`. À chaque tour, gardez la PREMIÈRE candidate de valeur
    strictement plus grande (« > ») : l'ordre de `restantes` règle les égalités. Après le choix `c` :
    `penalite[o] = max(penalite[o], similarites.get(o, {}).get(c, 0.0))` pour chaque restante `o`.
    """
    raise NotImplementedError

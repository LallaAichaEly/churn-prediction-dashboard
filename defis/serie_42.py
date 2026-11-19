"""Série 42 — Ce que vaut un client : transitions, projection, durée de vie, valeur actualisée, rentabilité, lissage.

💎 VOUS VOULEZ DÉVELOPPER ? UN JOUR, ON VOUS DEMANDERA « COMBIEN PEUT-ON DÉPENSER POUR GARDER CE CLIENT ? ».
LA RÉPONSE DU TABLEUR (MARGE / TAUX DE CHURN) EST FAUSSE POUR PRESQUE TOUT LE MONDE. ICI, VOUS ÉCRIVEZ CE QUE
FONT LES ÉQUIPES FINANCE ET DATA DES OPÉRATEURS, DES BANQUES ET DES ABONNEMENTS : UNE CHAÎNE DE MARKOV.

Le directeur marketing veut offrir une remise de 120 € à chaque client « à risque ». La finance a répondu avec la
formule de son tableur : « un client rapporte 23,50 € de marge par mois et 3,2 % des clients partent chaque
mois, donc un client vaut 23,50 / 0,032 = 734 € ; 120 €, c'est donné ». Inès, contrôleuse de gestion, n'est pas
convaincue : un client à risque ne rapporte pas la marge moyenne (9 € : il appelle le support et obtient des
gestes commerciaux), il part bien plus vite que les autres, il revient parfois à la normale, et un euro touché
dans dix ans ne vaut pas un euro aujourd'hui. Elle décrit chaque client, chaque mois, par un ÉTAT (« Fidèle »,
« Actif », « À risque », « Parti »), et écrit les six outils de cette série :

    254 matrice_transition    compter les passages d'un état à l'autre, d'un mois au SUIVANT, dans des relevés en désordre
    255 projection            ce que deviennent 1 000 clients, mois après mois (un vecteur ligne × la matrice)
    256 duree_de_vie          combien de mois un client reste-t-il, selon son état ? (un système linéaire)
    257 valeur_client         ce qu'il rapportera, en euros d'aujourd'hui (actualisé au taux mensuel équivalent)
    258 delai_rentabilite     au bout de combien de mois un client recruté a-t-il remboursé ce qu'il a coûté ?
    259 matrices_par_segment  30 clients Pro, aucun départ vu : lisser vers toute la clientèle (personne n'est éternel)

Prérequis : la série 00 et les dict de dict (`matrice["Actif"]["Parti"]`). La cohorte de la série 21 (défi 131),
NumPy (série 23) et l'uplift de la série 35 l'éclairent, mais tout ce qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- une MATRICE DE TRANSITION dit, pour chaque état de départ, où sont les clients le mois suivant :
  `matrice["Actif"]["À risque"] == 0.05` veut dire que 5 % des clients actifs ce mois-ci sont à risque le mois
  prochain. Chaque LIGNE fait 100 % (un client est forcément quelque part le mois suivant) ; les colonnes, non ;
- « Parti » est ABSORBANT : on n'en sort pas (sa ligne vaut 100 % sur lui-même). Les autres états sont
  TRANSITOIRES : tôt ou tard, chaque client les quitte pour de bon ;
- le mois suivant d'une cohorte : `vers[j] = Σ_i clients[i] × matrice[i][j]` (les clients qui étaient en i,
  multipliés par la part qui passe de i à j, additionnés sur tous les i). C'est un VECTEUR LIGNE multiplié par la
  matrice ; à l'envers (`Σ_j matrice[i][j] × clients[j]`), on calcule autre chose : la valeur, défi 257 ;
- l'ACTUALISATION : avec un coût du capital de 10 % par an, 110 € dans un an valent 100 € aujourd'hui. Le taux
  MENSUEL équivalent est `(1 + 0.10) ** (1 / 12) - 1` = 0,797 % (pas 10 % / 12 = 0,833 %, qui ferait 10,47 % sur
  un an). Un euro touché dans t mois vaut `1 / (1 + r) ** t` euro aujourd'hui ;
- les matrices que reçoivent les défis 255 à 258 sont COMPLÈTES (chaque ligne a une valeur pour chaque état) et
  leurs lignes font 100 % ; `afficher(matrice)` (defis/valeur.py) les montre en pourcentages : pour déboguer.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_42 import *
    >>> releves = releves_exemple()
    >>> len(releves), releves[:2]                  # (client, mois, état) : 48 434 relevés, dans le désordre
    (48434, [('C02024', '2026-01', 'Fidèle'), ('C01860', '2026-01', 'Actif')])
    >>> matrice = matrice_exemple()                # celle de l'an dernier : pour avancer sans attendre le 254
    >>> matrice["À risque"]
    {'Fidèle': 0.0, 'Actif': 0.25, 'À risque': 0.55, 'Parti': 0.2}

Quand les six défis sont réussis, Inès peut répondre au directeur marketing :

    >>> print(afficher(matrice_transition(releves)))          # cette année : proche de l'an dernier
                 Fidèle    Actif À risque    Parti
    Fidèle        95.2%     4.0%     0.7%     0.2%
    Actif          3.2%    89.5%     5.5%     1.8%
    À risque       0.0%    24.4%    52.8%    22.8%
    Parti          0.0%     0.0%     0.0%   100.0%
    >>> {etat: round(n) for etat, n in projection({"Actif": 1000}, matrice, 12)[12].items()}
    {'Fidèle': 178, 'Actif': 441, 'À risque': 58, 'Parti': 323}
    >>> {etat: round(t, 1) for etat, t in duree_de_vie(matrice).items()}    # en mois ; le tableur dit 31 pour tous
    {'Fidèle': 54.0, 'Actif': 37.8, 'À risque': 23.2}
    >>> valeur = valeur_client(matrice, MARGES, TAUX_ANNUEL, 120)          # sur dix ans, en euros d'aujourd'hui
    >>> {etat: round(v) for etat, v in valeur.items()}
    {'Fidèle': 1013, 'Actif': 657, 'À risque': 378, 'Parti': 0}
    >>> round(valeur["Actif"] - valeur["À risque"])          # ce que rapporte un client ramené à « Actif »
    279
    >>> delai_rentabilite(matrice, MARGES, "Actif", 180, TAUX_ANNUEL)    # un client recruté 180 € : 10 mois
    10
    >>> lissees = matrices_par_segment(releves, segments_exemple(), force=50)
    >>> {etat: round(t) for etat, t in duree_de_vie(lissees["Pro"]).items()}   # sans lissage : infini
    {'Fidèle': 369, 'Actif': 325, 'À risque': 170}

Le tableur disait 734 € pour tout le monde. Un client fidèle vaut 1 013 € sur dix ans, un client à risque 378 €.
La remise de 120 € ne « sauve » pas 734 € : si elle ramène un client à risque à l'état « Actif », elle rapporte
657 - 378 = 279 €. Elle n'est rentable que si elle convainc plus de 120 / 279 = 43 % des clients à qui on la
donne : c'est le chiffre qu'il faut aller mesurer (avec un groupe témoin, série 35 et projet 3), au lieu de
débattre. Et les 30 clients Pro, chez qui personne n'est jamais parti, ne sont pas éternels : lissés vers
l'ensemble de la clientèle, ils restent quand même 27 ans en moyenne (325 mois), et valent bien plus.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les six défis sont indépendants (les défis 255 à
   258 partent de `matrice_exemple()`) ; l'ordre proposé est le plus progressif (le 259 reprend le comptage du 254,
   le 258 cumule ce que le 257 calcule à rebours).
2. Lancez `python scripts/defis.py --defi 254` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_42.py`.

Les outils communs (`releves_exemple`, `segments_exemple`, `matrice_exemple`, `afficher`, `generer_releves`,
`ETATS`, `PARTI`, `OFFRES`, `MARGES`, `TAUX_ANNUEL` et `MOIS_PERDU`) sont dans `defis/valeur.py`, déjà importés
ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import math  # noqa: F401 — math.isfinite, math.isclose

from defis.valeur import (  # noqa: F401
    ETATS,
    MARGES,
    MOIS_PERDU,
    OFFRES,
    PARTI,
    TAUX_ANNUEL,
    afficher,
    generer_releves,
    matrice_exemple,
    releves_exemple,
    segments_exemple,
)

# ============================================================================ Estimer


def matrice_transition(releves, etats=ETATS) -> dict[str, dict[str, float]]:
    """Défi 254 🟢 — {depuis: {vers: part}} : la matrice de transition, estimée sur des relevés mensuels.

        >>> releves = [("C1", "2025-11", "Actif"), ("C1", "2025-12", "Actif"), ("C1", "2026-01", "À risque"),
        ...            ("C2", "2025-12", "Actif"), ("C2", "2026-02", "Parti")]       # C2 : janvier manque
        >>> matrice = matrice_transition(releves)
        >>> matrice["Actif"]
        {'Fidèle': 0.0, 'Actif': 0.5, 'À risque': 0.5, 'Parti': 0.0}
        >>> matrice["Parti"]                      # jamais quitté : on y reste
        {'Fidèle': 0.0, 'Actif': 0.0, 'À risque': 0.0, 'Parti': 1.0}

    Ici : C1 passe d'Actif (novembre) à Actif (décembre), puis d'Actif (décembre) à À risque (janvier) : deux
    passages depuis « Actif », un vers chaque état, donc 50 % et 50 %. Décembre 2025 et janvier 2026 se SUIVENT.
    C2 n'a pas de relevé en janvier : on ne sait pas s'il est parti en janvier ou en février, et on ne compte RIEN
    entre décembre et février. Aucun client n'a été vu quitter « Parti » (ni « Fidèle », ni « À risque ») : sur ces
    lignes, on reste où l'on est (100 % sur la diagonale). Sur les 48 434 relevés de l'exemple, on retrouve à peu
    près la matrice de l'an dernier.

    Les règles :

    - `releves` est un itérable de tuples (client, mois, état), dans n'importe quel ordre (le correcteur le
      mélange, et vous le passe sous forme de flux qui ne se relit pas : rangez ce qu'il vous faut en une lecture) ;
      un mois s'écrit « AAAA-MM » (« 2025-12 ») ;
    - un PASSAGE de i vers j : un même client, dans l'état i un mois, dans l'état j le mois SUIVANT (après
      décembre 2025 vient janvier 2026). Deux relevés d'un même client séparés par un mois manquant ne font pas de
      passage ; deux relevés de clients différents non plus ;
    - matrice[i][j] = (passages de i vers j) / (passages depuis i). Un état jamais quitté (aucun passage depuis
      lui) : 1.0 vers lui-même, 0.0 ailleurs. Chaque ligne contient TOUS les états de `etats`, dans leur ordre, et
      la matrice a une ligne par état de `etats`, dans leur ordre ; les valeurs sont des float ;
    - un même relevé chargé deux fois (même client, même mois, même état) ne compte qu'une fois ; deux relevés du
      même client et du même mois qui disent deux états différents → ValueError (lequel croire ?) ;
    - un état qui n'est pas dans `etats`, ou un mois illisible (« 2025-13 », « 2025-1 », « 202512 », 202512) →
      ValueError.

    Ce que vous apprenez : estimer des probabilités en COMPTANT, regrouper des lignes par client (un dict de
    dict), et le piège des dates écrites en texte : `int("2025-12".replace("-", "")) + 1` donne 202513, pas
    202601. Le numéro d'un mois `annee * 12 + mois - 1` se suit, lui, d'un mois à l'autre, décembre compris.

    Indice : `par_client.setdefault(client, {})[numero] = etat` (numero = annee * 12 + mois - 1) ; puis, pour chaque
    client, `suivant = mois_du_client.get(numero + 1)` : pas besoin de trier. Ensuite `comptes[i][j] += 1`, et
    chaque ligne divisée par `sum(comptes[i].values())`.
    """
    raise NotImplementedError


# ============================================================================ Projeter


def projection(depart: dict, matrice: dict, mois: int) -> list[dict[str, float]]:
    """Défi 255 🟡 — [mois 0, mois 1, …, mois `mois`] : ce que devient une cohorte, mois après mois.

        >>> etapes = projection({"Actif": 1000}, matrice_exemple(), 12)
        >>> len(etapes), etapes[0]
        (13, {'Fidèle': 0.0, 'Actif': 1000.0, 'À risque': 0.0, 'Parti': 0.0})
        >>> {etat: round(n, 1) for etat, n in etapes[1].items()}
        {'Fidèle': 30.0, 'Actif': 900.0, 'À risque': 50.0, 'Parti': 20.0}
        >>> {etat: round(n) for etat, n in etapes[12].items()}
        {'Fidèle': 178, 'Actif': 441, 'À risque': 58, 'Parti': 323}

    Ici : 1 000 clients actifs en janvier. En février, 3 % deviennent fidèles (30), 90 % restent actifs (900), 5 %
    passent à risque (50) et 2 % partent (20). En mars, il faut aussi compter ce que deviennent les 30 fidèles et
    les 50 à risque : « À risque » reçoit 0,8 % des fidèles, 5 % des actifs et 55 % des à-risque. Au bout d'un an,
    323 des 1 000 clients sont partis, et 58 sont à risque : ce sont eux que vise la remise du marketing.

    Les règles :

    - `depart` : {état: nombre de clients} ; un état absent compte 0 client. Un état qui n'est pas dans la
      matrice, ou un nombre négatif → ValueError ; un nombre qui n'est pas un int ou un float (un bool, un
      texte) → TypeError ;
    - la réponse est une liste de `mois` + 1 dict : l'élément 0 est le départ, complété par des 0.0 ; l'élément
      t + 1 se calcule à partir de l'élément t : `vers[j] = Σ_i effectifs[i] × matrice[i][j]` ; chaque dict a tous
      les états de la matrice, dans son ordre ; des nombres de clients (des float), PAS arrondis (arrondir chaque
      mois fausse le résultat d'un an) et PAS des parts ;
    - chaque élément est un NOUVEAU dict (si vous modifiez le même dict mois après mois, toute la liste montre le
      dernier mois) ; `depart` n'est pas modifié ;
    - `mois` : un int >= 0 (sinon TypeError ou ValueError) ; 0 → [le départ complété].

    Ce que vous apprenez : le produit d'un vecteur LIGNE par une matrice (celui de `numpy` s'écrit
    `effectifs @ P`, et `P @ effectifs` est le piège), la récurrence « le mois suivant à partir du précédent »
    (O(mois × états²), au lieu de recalculer P puissance t à chaque mois), et à PRÉVOIR une table de rétention
    (celle du défi 131 se mesure, celle-ci se calcule avant que les mois passent).

    Indice : `courant = {e: float(depart.get(e, 0)) for e in matrice}` ; dans la boucle, `suivant =
    dict.fromkeys(matrice, 0.0)`, puis `for i in matrice: for j, p in matrice[i].items(): suivant[j] +=
    courant[i] * p` ; `resultat.append(suivant)` et `courant = suivant`.
    """
    raise NotImplementedError


# ============================================================================ Combien de temps ?


def duree_de_vie(matrice: dict, absorbants=(PARTI,)) -> dict[str, float]:
    """Défi 256 🟡 — {état: mois} : combien de mois, en moyenne, un client reste client, selon son état de départ.

        >>> duree_de_vie({"Actif": {"Actif": 0.9, "Parti": 0.1}, "Parti": {"Actif": 0.0, "Parti": 1.0}})["Actif"]
        10.000000000000002
        >>> {etat: round(t, 1) for etat, t in duree_de_vie(matrice_exemple()).items()}
        {'Fidèle': 54.0, 'Actif': 37.8, 'À risque': 23.2}

    Ici (1er exemple) : un client actif part avec une chance sur 10 chaque mois : il reste 10 mois en moyenne, le
    mois en cours compris (un client qui part à coup sûr le mois prochain compte 1 mois). Avec plusieurs états, on
    ne peut plus diviser 1 par le taux de départ : un client à risque part à 20 % par mois (1 / 0,20 = 5 mois),
    mais un sur quatre redevient actif chaque mois, et vit alors bien plus longtemps. Il reste 23,2 mois en moyenne.

    Les règles :

    - la durée de vie t(i) d'un état transitoire i : le mois en cours, plus la durée de vie de l'état où il sera le
      mois suivant, en moyenne : `t(i) = 1 + Σ_j matrice[i][j] × t(j)`, la somme portant sur les états j
      TRANSITOIRES (un client parti n'a plus de mois à vivre). Ce sont n équations à n inconnues : (I − Q) · t = 1,
      où Q est la matrice limitée aux états transitoires ;
    - `absorbants` : un itérable d'états (un texte seul, comme "Parti", désigne UN état : `set("Parti")`, ce sont
      cinq lettres). Chacun doit être dans la matrice, et absorbant (matrice[a][a] == 1) : sinon ValueError ;
    - la réponse a une clé par état TRANSITOIRE (pas les absorbants), dans l'ordre de la matrice ; des float ;
    - si, depuis un état transitoire, AUCUN état absorbant n'est atteignable (deux états qui se renvoient les
      clients sans jamais les laisser partir), sa durée de vie est infinie → ValueError, qui nomme ces états ;
    - le correcteur vous passe des clients qui restent 10 000 mois en moyenne : une boucle « mois après mois »
      arrêtée à 1 000 mois, ou qui attend que le résultat ne bouge plus, est fausse ou trop lente. Résolvez le
      système : `numpy.linalg.solve(a, b)` (a est la liste des lignes de I − Q, b la liste de 1), ou l'élimination
      de Gauss écrite à la main.

    Ce que vous apprenez : qu'une moyenne qui « dépend d'elle-même » s'écrit en équations (la même idée que
    l'espérance d'un lancer de dés jusqu'au premier six), la « matrice fondamentale » (I − Q)⁻¹ des chaînes
    absorbantes, et pourquoi 1 / taux de churn ne marche que s'il n'y a qu'un état.

    Indice : `transitoires = [e for e in matrice if e not in absorbants]` ; `a = [[(1.0 if i == j else 0.0) -
    matrice[i].get(j, 0.0) for j in transitoires] for i in transitoires]` ; pour l'infini, partez des absorbants et
    remontez : un état « peut partir » s'il a une probabilité > 0 vers un état qui peut partir.
    """
    raise NotImplementedError


# ============================================================================ Combien d'euros ?


def valeur_client(matrice: dict, marges: dict, taux_annuel: float, horizon: int) -> dict[str, float]:
    """Défi 257 🟡 — {état: euros} : la marge qu'un client rapportera sur `horizon` mois, en euros d'aujourd'hui.

        >>> {etat: round(v) for etat, v in valeur_client(matrice_exemple(), MARGES, TAUX_ANNUEL, 120).items()}
        {'Fidèle': 1013, 'Actif': 657, 'À risque': 378, 'Parti': 0}
        >>> valeur_client({"A": {"A": 1.0}}, {"A": 100}, 0.0, 12)          # 12 mois à 100 €, sans actualiser
        {'A': 1200.0}
        >>> round(valeur_client({"A": {"A": 1.0}}, {"A": 100}, 0.10, 12)["A"], 2)
        1149.14

    Ici (3e exemple) : un client qui rapporte 100 € par mois pendant 12 mois, à 10 % par an. Le mois en cours n'est
    pas actualisé (100 €), le suivant vaut 100 / 1,10 ** (1 / 12) = 99,21 €, le 12e 100 / 1,10 ** (11 / 12) =
    91,63 € : 1 149,14 € en tout (avec 10 % / 12 par mois, on trouverait 1 146,93 €). Avec la matrice : un client
    actif rapporte 22 € ce mois-ci, puis, le mois suivant, la valeur de l'état où il sera, en moyenne.

    Les règles :

    - `marges` : {état: marge mensuelle en euros} ; un état absent rapporte 0 ; un état qui n'est pas dans la
      matrice → ValueError ; une marge qui n'est pas un int ou un float → TypeError ;
    - le taux mensuel est r = (1 + taux_annuel) ** (1 / 12) - 1 ; `taux_annuel` est un int ou un float >= 0
      (sinon TypeError ou ValueError) ;
    - V(i) = Σ_{t = 0}^{horizon − 1} (marge moyenne au mois t, pour un client parti de i) / (1 + r) ** t : le mois
      0 (le mois en cours) n'est PAS actualisé ; `horizon` est un int >= 0 (0 → que des 0.0) ;
    - la réponse a une clé par état de la matrice (les absorbants compris), dans son ordre ; des float ;
    - à rebours, c'est une seule boucle : sur 0 mois, V = 0 ; sur h mois, `V(i) = marge(i) + Σ_j matrice[i][j] ×
      V_{h−1}(j) / (1 + r)` (ce mois-ci, plus ce que vaudra, sur h − 1 mois, l'état du mois prochain, actualisé
      d'un mois). Le correcteur demande 1 200 mois : recalculer chaque mois à partir de zéro est trop lent.

    Ce que vous apprenez : l'actualisation (le « discounted cash flow » de la finance), le taux mensuel
    ÉQUIVALENT, la récurrence à rebours (la programmation dynamique de Bellman : la valeur aujourd'hui = ce qu'on
    gagne maintenant + la valeur de demain, actualisée), et que cette fois la matrice multiplie un vecteur COLONNE
    (`P @ V`) : on regarde où l'on VA, pas d'où l'on vient.

    Indice : `facteur = 1 / (1 + r)` ; `v = dict.fromkeys(matrice, 0.0)` ; `for _ in range(horizon): v = {i:
    m[i] + facteur * sum(p * v[j] for j, p in matrice[i].items()) for i in matrice}`.
    """
    raise NotImplementedError


def delai_rentabilite(matrice: dict, marges: dict, etat: str, cout: float, taux_annuel: float,
                      horizon_max: int = 120) -> int | None:
    """Défi 258 🔴 — Le nombre de mois au bout duquel un client parti de `etat` a rapporté `cout` (actualisé), ou None.

        >>> delai_rentabilite(matrice_exemple(), MARGES, "Actif", 180, TAUX_ANNUEL)
        10
        >>> delai_rentabilite({"A": {"A": 1.0}}, {"A": 10}, "A", 30, 0.0)        # 10 + 10 + 10 >= 30 : 3 mois
        3
        >>> print(delai_rentabilite(matrice_exemple(), MARGES, "À risque", 400, TAUX_ANNUEL))
        None

    Ici (1er exemple) : recruter un client coûte 180 € (publicité, box offerte) ; il arrive « Actif ». Le mois 0, il
    rapporte 22 € ; le mois 1, 3 % de 30 € + 90 % de 22 € + 5 % de 9 € (il a pu changer d'état, ou partir),
    actualisé d'un mois ; et ainsi de suite. Le cumul atteint 180 € pendant le 10e mois : le client recruté est
    rentable au bout de 10 mois. Un client à risque, lui, ne rapportera jamais 400 € : même sur 1 000 ans, il vaut
    388 €. Ce n'est pas « un très grand nombre de mois », c'est None.

    Les règles :

    - au mois t (t = 0, 1, 2…), la cohorte d'UN client parti de `etat` (100 % dans `etat` au mois 0, puis la
      projection du défi 255) rapporte Σ_i part(i) × marge(i), actualisé de `1 / (1 + r) ** t`, le mois 0 n'étant
      pas actualisé, avec le r du défi 257 ;
    - la réponse est le plus petit n (en mois : 1 si le mois 0 suffit) tel que la somme des mois 0 à n − 1 est
      >= `cout` (« au moins » : un cumul égal au coût suffit) ; `cout` == 0 → 0 ; jamais atteint dans les
      `horizon_max` premiers mois → None (pas `horizon_max`, qui serait une vraie réponse) ;
    - `etat` doit être dans la matrice (sinon ValueError) ; `cout` est un int ou un float >= 0, `horizon_max` un
      int >= 0, `marges` et `taux_annuel` suivent les règles du défi 257 (sinon TypeError ou ValueError) ;
    - une seule passe : le correcteur demande 2 400 mois pour un coût jamais atteint, et compte les lignes
      exécutées. Appeler `valeur_client(…, horizon=n)` pour n = 1, 2, 3… refait tout le calcul à chaque mois.

    Ce que vous apprenez : le délai de récupération (« CAC payback », l'indicateur que suivent tous les services
    d'abonnement), à combiner la projection (vers l'avant) et l'actualisation, une boucle qui s'arrête dès qu'elle
    a sa réponse, et à distinguer « pas trouvé » (None) d'une réponse.

    Indice : `repartition = dict.fromkeys(matrice, 0.0)`, puis `repartition[etat] = 1.0` ; `cumul, actualisation =
    0.0, 1.0` ; dans la boucle : `cumul += actualisation * sum(...)`, `if cumul >= cout: return t + 1`,
    `actualisation /= 1 + r`, et la répartition du mois suivant (celle du défi 255).
    """
    raise NotImplementedError


# ============================================================================ Les petits segments


def matrices_par_segment(releves, segments: dict, etats=ETATS, force: float = 50) -> dict[str, dict]:
    """Défi 259 🔴 — {segment: matrice} : une matrice par segment, LISSÉE vers celle de toute la clientèle.

        >>> releves = [("A", "2026-01", "Actif"), ("A", "2026-02", "Parti"), ("B", "2026-01", "Actif"),
        ...            ("B", "2026-02", "Actif"), ("C", "2026-01", "Actif"), ("C", "2026-02", "Actif"),
        ...            ("D", "2026-01", "Actif"), ("D", "2026-02", "Actif")]
        >>> segments = {"A": "Essentiel", "B": "Pro", "C": "Essentiel", "D": "Essentiel"}
        >>> matrices_par_segment(releves, segments, etats=("Actif", "Parti"), force=4)["Pro"]["Actif"]
        {'Actif': 0.8, 'Parti': 0.2}

    Ici : toute la clientèle compte 4 passages depuis « Actif », dont 1 départ : 75 % / 25 %. Le segment Pro n'a
    vu qu'un passage (Actif → Actif) : brut, il dirait 100 % / 0 % (« les clients Pro ne partent jamais »). Lissé
    avec `force` = 4, on lui prête 4 passages répartis comme toute la clientèle (3 vers Actif, 1 vers Parti) :
    (1 + 3) / (1 + 4) = 80 % et (0 + 1) / (1 + 4) = 20 %. Sur les relevés de l'exemple, les 30 clients Pro n'ont
    jamais été vus partir, ni même passer à risque : sans lissage, `duree_de_vie` lève ValueError (infini) ; avec
    `force=50`, un client Pro actif reste 325 mois.

    Les règles :

    - les passages se comptent comme au défi 254 (mêmes règles, mêmes erreurs : mois consécutifs d'un même client,
      doublons exacts comptés une fois, relevés contradictoires, états inconnus et mois illisibles → ValueError) ;
    - la matrice GLOBALE est celle du défi 254 sur TOUS les passages (de tous les segments ensemble) ;
    - pour un segment s et un état i, avec n_s(i, j) les passages du segment de i vers j et n_s(i) leur total :
      `P_s[i][j] = (n_s(i, j) + force × P_globale[i][j]) / (n_s(i) + force)`. Si n_s(i) + force vaut 0 (rien vu,
      rien emprunté), la ligne est celle de la matrice globale ;
    - `segments` : {client: segment} (un dict, sinon TypeError) ; un client des relevés absent de `segments` →
      ValueError. La réponse a une clé par segment présent dans `segments` (même sans aucun relevé : sa matrice
      est alors la globale), et chaque matrice a toutes ses lignes et colonnes, dans l'ordre de `etats` ;
    - `force` : un int ou un float >= 0 (sinon TypeError ou ValueError) ; 0 → les matrices brutes de chaque segment.

    Ce que vous apprenez : qu'une estimation sur peu de données est extrême (0 départ vu ≠ 0 % de départs), le
    lissage « bayésien empirique » (des pseudo-observations empruntées à la population, d'autant moins influentes
    que le segment a de données : 50 passages prêtés pèsent beaucoup face à 10, presque rien face à 10 000), et
    pourquoi la moyenne des deux matrices (50 / 50) n'est pas la même chose.

    Indice : réutilisez votre comptage du défi 254 en gardant le client de chaque passage ; `comptes = {s: {i:
    dict.fromkeys(etats, 0) for i in etats} for s in segments.values()}` ; puis la formule ligne par ligne.
    """
    raise NotImplementedError

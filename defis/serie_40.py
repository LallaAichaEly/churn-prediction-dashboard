"""Série 40 — Compter sans tout retenir : filtre de Bloom, éléments fréquents, Count-Min, HyperLogLog, hachage cohérent.

🧮 VOUS VOULEZ DÉVELOPPER ? UN JOUR, VOS DONNÉES NE TIENDRONT PLUS DANS UN `set`. ICI, VOUS ÉCRIVEZ LES STRUCTURES
QUE REDIS, CASSANDRA, BIGQUERY OU LES NAVIGATEURS UTILISENT POUR RÉPONDRE « À PEU PRÈS », AVEC UNE ERREUR CONNUE,
DANS QUELQUES KILO-OCTETS.

Le 12 novembre, à 19 h, la campagne de rétention s'est arrêtée net : le service d'envoi des SMS gardait dans un
`set` les 40 millions de numéros contactés depuis le début du mois (pour ne jamais écrire deux fois au même
client), et ce `set` a fini par dépasser les 3 Go de mémoire de son serveur. Le même soir, le tableau de bord
annonçait 11 242 « visiteurs uniques » de la page de résiliation pour la semaine : il additionnait les visiteurs
de chaque jour, et un client venu lundi puis jeudi comptait deux fois (ils n'étaient que 7 981). Et le service de
score, passé de 4 à 5 serveurs avec `hash(client) % 5`, a vu 80 % de ses clients changer de serveur : tous ses
caches sont repartis de zéro, et les réponses ont mis six fois plus de temps pendant une heure. Inès, développeuse
de la plateforme d'envoi, reprend tout. Cette série vous fait écrire ses six outils :

    242 dimensionner   la taille d'un filtre de Bloom pour n clés et un taux d'erreur donné
    243 FiltreBloom    « ce numéro a-t-il déjà reçu un SMS ? » en 14 Ko au lieu de 3 Go : jamais de doublon
    244 frequents      les numéros qui appellent le plus le support, avec k - 1 compteurs (Misra-Gries)
    245 CompteurMin    combien de fois CE numéro a-t-il appelé ? Jamais moins que la vérité (Count-Min)
    246 HyperLogLog    combien de visiteurs DIFFÉRENTS cette semaine ? 16 Ko, 1 % d'erreur, et la semaine
                       se calcule à partir des jours
    247 Anneau         quel serveur calcule le score de ce client ? Ajouter un serveur ne déplace que sa part

Prérequis : la série 00 et une classe (défi 48). Les `set` du défi 45, la dichotomie (`bisect`, défi 112) et le
seau SHA-256 du défi 240 l'éclairent, mais tout ce qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- une structure PROBABILISTE garde un résumé au lieu des données. Elle peut se tromper, mais d'une façon prévue
  à l'avance : le filtre de Bloom ne se trompe que dans un sens (il peut dire « déjà vu » à tort, jamais « pas vu »
  à tort), Count-Min ne fait que surestimer, HyperLogLog se trompe d'environ 1,04 / √m ;
- toutes reposent sur une fonction de hachage : un texte → un grand entier qui a l'air tiré au hasard, mais qui
  est toujours le même pour le même texte. `empreinte(texte, graine)` (defis/esquisses.py) vous la donne : 64
  bits tirés de SHA-256. Changer la graine donne une AUTRE fonction de hachage, indépendante de la première.
  Pas `hash()` : il change à chaque lancement de Python (série 39, défi 240) ;
- `x >> 60` garde les 4 bits de gauche d'un entier de 64 bits ; `x & ((1 << 60) - 1)` garde les 60 bits de
  droite ; `x.bit_length()` est le nombre de bits qu'il faut pour écrire x (0 pour 0, 3 pour 5 = 0b101) ;
  `octet.bit_count()` compte ses bits à 1 ;
- un `bytearray(n)` est une suite de n octets (entiers de 0 à 255) modifiable : 8 bits chacun, sans les ~28
  octets que coûte chaque entier Python dans une liste ;
- ces résumés se FUSIONNENT : trois serveurs, ou sept jours, donnent un seul résumé, sans revoir une seule donnée
  (comme la fusion de la série 28, défi 175).

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_40 import *
    >>> empreinte("0612345678"), empreinte("0612345678", 1)            # deux fonctions de hachage
    (8160095906645286627, 15777807589842033787)
    >>> {serveur: len(numeros) for serveur, numeros in envois_exemple().items()}
    {'sms-1': 5000, 'sms-2': 4000, 'sms-3': 3000}

Quand les six défis sont réussis, Inès peut relancer la campagne, et le reste :

    >>> envois, campagne = envois_exemple(), campagne_exemple()
    >>> d = dimensionner(12_000, 0.01)
    >>> d.bits, d.k, d.octets                                  # 14 Ko pour 12 000 numéros
    (115021, 7, 14378)
    >>> filtres = []
    >>> for numeros in envois.values():                        # chaque serveur remplit son filtre
    ...     filtre = FiltreBloom(d.bits, d.k)
    ...     for numero in numeros:
    ...         filtre.ajouter(numero)
    ...     filtres.append(filtre)
    >>> deja = filtres[0].union(filtres[1]).union(filtres[2])
    >>> sum(numero in deja for numero in campagne)              # 1 000 déjà contactés, et 35 de trop
    1035
    >>> list(frequents(appels_exemple(), 100))[:4]              # tout ce qui dépasse 600 appels sur 60 000
    ['0700000000', '0611111111', '0622222222', '0633333333']
    >>> cm = CompteurMin.pour_erreur(0.001, 0.01)
    >>> for numero in appels_exemple():
    ...     cm.ajouter(numero)
    >>> cm.estimer("0700000000"), cm.estimer("0600000000")      # vrais comptes : 2 400 et 0
    (2405, 4)
    >>> jours = []
    >>> for visiteurs in visites_exemple().values():
    ...     h = HyperLogLog(14)
    ...     for v in visiteurs:
    ...         h.ajouter(v)
    ...     jours.append(h)
    >>> semaine = jours[0]
    >>> for h in jours[1:]:
    ...     semaine = semaine.fusionner(h)
    >>> round(semaine.estimation())                             # 7 981 en vrai ; 11 242 en additionnant
    7947
    >>> anneau = Anneau(SERVEURS)
    >>> avant = {c: anneau.serveur_pour(c) for c in clients_exemple()}
    >>> anneau.ajouter("score-5")
    >>> deplaces = [c for c, s in avant.items() if anneau.serveur_pour(c) != s]
    >>> len(deplaces), {anneau.serveur_pour(c) for c in deplaces}     # avec « % 5 » : 16 028 déplacés
    (3714, {'score-5'})

Le filtre de Bloom bloque les 1 000 numéros déjà contactés (aucun doublon, c'est garanti) et 35 numéros de trop
(1,2 % des 3 000 nouveaux : le taux d'erreur choisi), pour 14 Ko au lieu d'un `set` de plusieurs Mo ; à 40
millions de numéros, 72 Mo au lieu de 3 Go. Les quatre gros appelants sont trouvés avec 99 compteurs, l'estimation
de Count-Min ne dépasse jamais la vérité de plus de 60 appels (0,1 % de 60 000), la semaine se compte à 0,4 % près
à partir de sept résumés de 16 Ko, et l'arrivée du cinquième serveur ne déplace que 18,6 % des clients, tous vers
lui.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Les six défis sont indépendants ; l'ordre proposé
   est le plus progressif (le 242 donne la taille du filtre du 243).
2. Lancez `python scripts/defis.py --defi 242` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_40.py`.

Les outils communs (`empreinte`, `Dimension`, `taux_theorique`, `envois_exemple`, `campagne_exemple`,
`appels_exemple`, `visites_exemple`, `clients_exemple`, `SERVEURS`, `JOURS`, `generer_numeros`, `generer_appels`
et `generer_visiteurs`) sont dans `defis/esquisses.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import bisect  # noqa: F401 — bisect.bisect_left(liste_triee, x) : la première place où x pourrait s'insérer
import math  # noqa: F401 — math.log (logarithme NATUREL), math.ceil, math.exp, math.e

from defis.esquisses import (  # noqa: F401
    JOURS,
    SERVEURS,
    Dimension,
    appels_exemple,
    campagne_exemple,
    clients_exemple,
    empreinte,
    envois_exemple,
    generer_appels,
    generer_numeros,
    generer_visiteurs,
    taux_theorique,
    visites_exemple,
)

# ============================================================================ Ne jamais écrire deux fois


def dimensionner(n: int, taux: float) -> Dimension:
    """Défi 242 🟢 — Dimension(bits, k, taux_reel) : la taille d'un filtre de Bloom pour `n` clés et un `taux` de
    faux positifs.

        >>> d = dimensionner(1000, 0.01)
        >>> d.bits, d.k, round(d.taux_reel, 5)
        (9586, 7, 0.01003)
        >>> d = dimensionner(40_000_000, 0.001)
        >>> d.bits, d.k, d.octets                 # 72 Mo pour 40 millions de numéros, au lieu de plus de 3 Go
        (575103503, 10, 71887938)
        >>> dimensionner(10, 0.9)[:2]             # la formule donne k = 0,2 : on garde au moins une position
        (3, 1)

    Ici : pour retenir 1 000 numéros en acceptant 1 % de faux positifs, il faut 9 586 bits (1,2 Ko) et 7 positions
    par numéro. Le taux réel (1,003 %) dépasse un peu 1 % parce que k a été arrondi à un entier.

    Les règles (les formules classiques du filtre de Bloom) :

    - bits = ⌈-n · ln(taux) / (ln 2)²⌉ : le logarithme NATUREL (`math.log`), et arrondi VERS LE HAUT (`math.ceil`) ;
    - k = (bits / n) · ln 2, arrondi à l'entier le plus proche (`round`), et AU MOINS 1 ;
    - taux_reel = (1 - e^(-k · n / bits))^k, calculé avec CES bits et CE k (c'est `taux_theorique(bits, k, n)`) ;
    - la réponse est un `Dimension(bits, k, taux_reel)` (de defis/esquisses.py) ; `d.octets` en donne la taille ;
    - erreurs : n qui n'est pas un int (un float, un texte, un bool) → TypeError ; n < 1 → ValueError ; taux qui
      n'est pas un nombre (un texte, None, un bool) → TypeError ; taux qui n'est pas STRICTEMENT entre 0 et 1
      (0, 1, -0.1, NaN, l'infini) → ValueError.

    Ce que vous apprenez : dimensionner AVANT de coder (la mémoire se calcule), et qu'un taux d'erreur se choisit :
    diviser le taux par 10 ne coûte que 4,8 bits de plus par clé.

    Indice : `math.ceil(-n * math.log(taux) / math.log(2) ** 2)` ; `0 < taux < 1` est faux pour NaN, ce qui le
    refuse d'office.
    """
    raise NotImplementedError


class FiltreBloom:
    """Défi 243 🟡 — « Ce numéro a-t-il déjà reçu un SMS ? », sans garder les numéros : un tableau de bits.

        >>> f = FiltreBloom(20, 3)
        >>> f.positions("0612345678"), f.positions("0699999999")
        ([7, 14, 1], [19, 10, 1])
        >>> f.ajouter("0612345678")
        >>> f.octets                              # les bits 1 et 7 de l'octet 0, le bit 6 de l'octet 1
        bytearray(b'\\x82@\\x00')
        >>> "0612345678" in f, "0699999999" in f, f.remplissage()
        (True, False, 0.15)

    Ici : le filtre a 20 bits (rangés dans 3 octets) et met 3 bits à 1 par numéro. « 0612345678 » allume les bits
    7, 14 et 1. « 0699999999 » demande les bits 19, 10 et 1 : le bit 1 est allumé (par l'autre numéro), mais pas
    le 19 : il n'a jamais été ajouté, c'est CERTAIN. Si ses trois bits avaient été allumés par d'autres numéros, le
    filtre aurait répondu « présent » à tort : un faux positif (un client n'aurait pas reçu son SMS). L'inverse
    (dire « absent » pour un numéro ajouté) est impossible : ses bits ont été allumés, et rien ne les éteint.

    Ce qu'il faut écrire :

    - `__init__(bits, k)` : un filtre vide ; trois attributs publics : `bits`, `k`, et `octets`, un `bytearray`
      de ⌈bits / 8⌉ octets à zéro ;
    - `positions(cle)` : la LISTE des k positions de la clé (des entiers de 0 à bits - 1), dans l'ordre i = 0, 1,
      …, k - 1, doublons compris ;
    - `ajouter(cle)` : met à 1 les k bits de la clé ;
    - `cle in filtre` (la méthode `__contains__`) : True si les k bits sont TOUS à 1 ;
    - `remplissage()` : la part des bits à 1, entre 0 et 1 ;
    - `union(autre)` : un NOUVEAU filtre qui contient les clés des deux (sans modifier ni l'un ni l'autre).

    Les règles :

    - les positions (le « double hachage » de Kirsch et Mitzenmacher, 2006) : avec h1 = `empreinte(cle, 0)` et
      h2 = `empreinte(cle, 1)`, la position numéro i vaut (h1 + i · h2) modulo bits ;
    - la position p est le bit numéro p % 8 de l'octet p // 8, en partant du bit de POIDS FAIBLE (le bit j vaut
      `1 << j`) : la position 9 est le bit 1 de l'octet 1 ;
    - aucune clé n'est gardée (le correcteur fouille vos attributs) : seulement les bits ;
    - `remplissage()` divise par `bits`, pas par 8 fois le nombre d'octets (les bits de bourrage du dernier octet
      ne font pas partie du filtre) ;
    - `union` : même `bits` et même `k` exigés (sinon ValueError) ; un `autre` qui n'est pas un filtre → TypeError.
      Chaque octet du résultat est le OU (`|`) des deux octets ;
    - erreurs : bits ou k qui n'est pas un int (ou un bool) → TypeError, < 1 → ValueError ; une clé qui n'est pas
      un texte (`612345678` n'est pas « 0612345678 ») → TypeError, pour `positions`, `ajouter` et `in`, et le
      filtre n'a pas changé ;
    - deux filtres ne partagent rien.

    Ce que vous apprenez : manipuler des bits (`|=`, `<<`, `>>`, `&`), répondre « peut-être » ou « sûrement pas »
    au lieu de « oui » ou « non », et choisir le sens de l'erreur : mieux vaut oublier un SMS que l'envoyer deux
    fois.

    Indice : `self.octets[p // 8] |= 1 << (p % 8)` pour allumer, `self.octets[p // 8] >> (p % 8) & 1` pour lire ;
    `all(...)` ; `bytearray(a | b for a, b in zip(...))` pour l'union.
    """

    def __init__(self, bits: int, k: int) -> None:
        raise NotImplementedError

    def positions(self, cle: str) -> list[int]:
        raise NotImplementedError

    def ajouter(self, cle: str) -> None:
        raise NotImplementedError

    def __contains__(self, cle: str) -> bool:
        raise NotImplementedError

    def remplissage(self) -> float:
        raise NotImplementedError

    def union(self, autre: FiltreBloom) -> FiltreBloom:
        raise NotImplementedError


# ============================================================================ Repérer les gros appelants


def frequents(flux, k: int) -> dict:
    """Défi 244 🟡 — {élément: compteur} : les candidats « fréquents » du flux, avec AU PLUS k - 1 compteurs.

        >>> frequents("aabacad", 2)               # « a » est là 4 fois sur 7 : plus de la moitié
        {'a': 1}
        >>> frequents(["x", "y", "x", "z", "x", "y", "x"], 3)
        {'x': 3, 'y': 1}
        >>> resultat = frequents(appels_exemple(), 100)
        >>> len(resultat), list(resultat.items())[:4]
        (65, [('0700000000', 1833), ('0611111111', 933), ('0622222222', 335), ('0633333333', 136)])

    Ici (2e exemple, k = 3, donc 2 compteurs) : x → {x: 1}, y → {x: 1, y: 1}, x → {x: 2, y: 1}. Puis z arrive
    et il n'y a plus de place : on retire 1 à TOUS les compteurs ({x: 1}, y tombe à 0 et disparaît), et z n'entre
    pas. Puis x → {x: 2}, y → {x: 2, y: 1}, x → {x: 3, y: 1}.

    La garantie (Misra et Gries, 1982) : sur n éléments, tout élément vu PLUS de n / k fois est dans le résultat,
    et son compteur vaut au moins (vrai compte - n / k), jamais plus que le vrai compte. Sur les 60 000 appels du
    mois, avec k = 100 : tout numéro qui a appelé plus de 600 fois est trouvé, avec 99 compteurs au plus, au lieu
    d'un dictionnaire de 15 611 numéros. Le robot (2 400 appels) a un compteur de 1 833 : entre 1 800 et 2 400.

    Les règles :

    - `flux` est n'importe quel itérable (une liste, un texte, un générateur…) : lisez-le UNE seule fois (un
      générateur relu est vide, en silence) ; ses éléments sont hashables ;
    - pour chaque élément x : s'il a un compteur, +1 ; sinon, s'il y a moins de k - 1 compteurs, il en reçoit un
      qui vaut 1 ; sinon, on retire 1 à TOUS les compteurs, on supprime ceux qui tombent à 0, et x n'entre PAS ;
    - la réponse est un NOUVEAU dict des compteurs restants, du plus grand au plus petit (à égalité, l'ordre est
      libre) ;
    - erreurs : k qui n'est pas un int (ou un bool) → TypeError ; k < 2 → ValueError (k = 1 ne laisserait aucun
      compteur), avant même de lire le flux.

    Ce que vous apprenez : l'algorithme de vote de Boyer-Moore (« majority element », k = 2) et sa généralisation,
    deux classiques d'entretien, et qu'un résultat APPROCHÉ peut s'accompagner d'une garantie EXACTE.

    Indice : `for cle in list(compteurs):` pour décrémenter (on ne supprime pas dans un dict qu'on parcourt) ;
    `dict(sorted(compteurs.items(), key=lambda p: p[1], reverse=True))`.
    """
    raise NotImplementedError


class CompteurMin:
    """Défi 245 🟡 — Combien de fois CE numéro a-t-il appelé ? Une estimation qui ne sous-estime jamais (Count-Min).

        >>> cm = CompteurMin(5, 3)
        >>> cm.ajouter("0611111111", 4)
        >>> cm.ajouter("0622222222")
        >>> cm.colonnes("0611111111"), cm.colonnes("0622222222")
        ([1, 1, 0], [3, 2, 2])
        >>> cm.tableau
        [[0, 4, 0, 1, 0], [0, 4, 1, 0, 0], [4, 0, 1, 0, 0]]
        >>> cm.estimer("0611111111"), cm.estimer("0600000009"), cm.total
        (4, 1, 5)
        >>> grand = CompteurMin.pour_erreur(0.001, 0.01)
        >>> grand.largeur, grand.profondeur
        (2719, 5)

    Ici : le tableau a 3 lignes de 5 cases. Chaque numéro a UNE colonne par ligne, et `ajouter` y ajoute son
    compte dans chaque ligne. « 0600000009 » n'a jamais appelé, mais ses colonnes [1, 2, 2] tombent sur des cases
    remplies par d'autres (4, 1 et 1) : le minimum, 1, est une surestimation. Une estimation ne peut jamais être
    EN DESSOUS de la vérité : chaque case contient le vrai compte, plus celui des numéros qui la partagent.

    Ce qu'il faut écrire :

    - `__init__(largeur, profondeur)` : un tableau de `profondeur` lignes de `largeur` zéros (une liste de
      listes), et quatre attributs publics : `largeur`, `profondeur`, `tableau` et `total` (la somme de tous
      les comptes ajoutés, 0 au départ) ;
    - `CompteurMin.pour_erreur(epsilon, delta)` (une `@classmethod`) : un compteur de largeur ⌈e / ε⌉ et de
      profondeur ⌈ln(1 / δ)⌉ (e = `math.e`, ln = `math.log`). Il garantit : estimation <= vrai compte +
      ε · total, avec une probabilité d'au moins 1 - δ ;
    - `colonnes(cle)` : la liste des colonnes de la clé, une par ligne : pour la ligne i, `empreinte(cle, i)`
      modulo `largeur` (une graine DIFFÉRENTE par ligne) ;
    - `ajouter(cle, n=1)` : ajoute n à la case de la clé dans chaque ligne, et à `total` ;
    - `estimer(cle)` : le MINIMUM des cases de la clé.

    Les règles :

    - chaque ligne est une liste DIFFÉRENTE : `[[0] * largeur] * profondeur` répète la même liste (le correcteur
      le voit) ;
    - n est un int >= 0 : un compte négatif casserait la garantie (TypeError pour un float ou un bool, ValueError
      pour un négatif) ; une clé qui n'est pas un texte → TypeError ; dans tous ces cas, rien n'a changé ;
    - largeur ou profondeur qui n'est pas un int → TypeError, < 1 → ValueError ; epsilon ou delta qui n'est pas
      un nombre → TypeError, pas strictement entre 0 et 1 → ValueError ;
    - deux compteurs ne partagent rien.

    Ce que vous apprenez : pourquoi le MINIMUM (chaque ligne se trompe vers le haut, la moins polluée gagne), la
    liste de listes qui n'en est qu'une (le piège de `*` sur une liste), et une `@classmethod` qui construit un
    objet à partir de ce qu'on veut garantir plutôt que de sa taille.

    Indice : `[[0] * largeur for _ in range(profondeur)]` ; `cls(largeur, profondeur)` dans la classmethod ;
    validez la clé ET n avant de toucher au tableau.
    """

    def __init__(self, largeur: int, profondeur: int) -> None:
        raise NotImplementedError

    @classmethod
    def pour_erreur(cls, epsilon: float, delta: float) -> CompteurMin:
        raise NotImplementedError

    def colonnes(self, cle: str) -> list[int]:
        raise NotImplementedError

    def ajouter(self, cle: str, n: int = 1) -> None:
        raise NotImplementedError

    def estimer(self, cle: str) -> int:
        raise NotImplementedError


# ============================================================================ Compter les visiteurs différents


class HyperLogLog:
    """Défi 246 🔴 — Combien de visiteurs DIFFÉRENTS ? Une estimation à ~1 % près dans 2**precision octets.

        >>> h = HyperLogLog(4)                    # 2**4 = 16 registres
        >>> h.ajouter("V000001")
        >>> x = empreinte("V000001")
        >>> f"{x:064b}"[:12], x >> 60             # les 4 bits de gauche : le registre 6
        ('011001011011', 6)
        >>> list(h.registres)                     # après « 0110 », « 01… » : le premier 1 est au rang 2
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> for v in ["V000002", "V000003", "V000001"]:
        ...     h.ajouter(v)
        >>> list(h.registres), round(h.estimation(), 3)
        ([0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0], 3.322)

    L'idée (Flajolet, Fusy, Gandouet et Meunier, 2007) : une empreinte a l'air tirée au hasard. Qu'elle commence
    par « 1 » arrive une fois sur 2, par « 01 » une fois sur 4, par « 0000001 » une fois sur 128. Si le plus long
    début en zéros qu'on ait vu atteint le rang r, on a probablement vu de l'ordre de 2**r visiteurs différents,
    et revoir le MÊME visiteur ne change rien (même empreinte). Pour ne pas dépendre d'un seul coup de chance, on
    répartit les visiteurs entre m = 2**precision registres et on fait une moyenne (harmonique).

    Ici : V000001 tombe dans le registre 6 (« 0110 »), et le reste de son empreinte commence par « 01 » : rang 2.
    V000002 tombe dans le registre 10 (rang 2), V000003 dans le 1 (rang 1), et V000001 revu ne change rien. Avec
    13 registres vides sur 16, l'estimation passe par le comptage des vides : 16 · ln(16 / 13) = 3,322.

    Ce qu'il faut écrire :

    - `__init__(precision=14)` : deux attributs publics, `precision` (p) et `registres`, m = 2**p registres à 0
      (un `bytearray(m)` : 16 Ko pour p = 14 ; une liste d'entiers est acceptée) ;
    - `ajouter(cle)` ;
    - `estimation()` : le nombre estimé de clés différentes (un float) ;
    - `fusionner(autre)` : un NOUVEL HyperLogLog, comme si un seul avait vu les clés des deux.

    Les règles (HyperLogLog, avec une empreinte de 64 bits) :

    - ajouter : x = `empreinte(cle)` ; j = les p bits de GAUCHE de x (`x >> (64 - p)`) ; w = les 64 - p bits
      restants ; rang = la position du premier 1 de w écrit sur 64 - p bits, en partant de la gauche et en
      comptant à partir de 1 (w = 0 → 64 - p + 1). Le registre j garde le MAXIMUM de ses rangs ;
    - estimation : avec m registres de valeurs r_j, E = α · m² / Σ 2^(-r_j), où α vaut 0.673 pour m = 16, 0.697
      pour m = 32, 0.709 pour m = 64, et 0.7213 / (1 + 1.079 / m) au-delà. Si E <= 2.5 · m et qu'il reste V > 0
      registres à 0, la réponse est m · ln(m / V) (le « linear counting », bien plus juste pour peu de clés) ;
      sinon, E ;
    - fusionner : le MAXIMUM registre par registre (pas la somme : un visiteur venu deux jours compterait deux
      fois) ; même precision exigée (sinon ValueError) ; un `autre` qui n'est pas un HyperLogLog → TypeError ;
      ni `self` ni `autre` ne changent ;
    - aucune clé n'est gardée : seulement les registres ;
    - erreurs : precision qui n'est pas un int → TypeError, hors de 4 à 16 → ValueError ; une clé qui n'est pas un
      texte → TypeError ;
    - deux objets ne partagent rien.

    Ce que vous apprenez : la structure derrière `PFCOUNT` de Redis et `APPROX_COUNT_DISTINCT` de BigQuery, les
    opérations sur les bits, la moyenne harmonique, et pourquoi un résumé qui se fusionne par un maximum permet de
    compter la semaine (ou le mois, ou trois pays) à partir des jours.

    Indice : `rang = (64 - p) - w.bit_length() + 1` ; `bin(w)` ne convient pas tel quel (il supprime les zéros de
    tête) ; `self.registres.count(0)` compte les registres vides ; `2.0 ** -r`.
    """

    def __init__(self, precision: int = 14) -> None:
        raise NotImplementedError

    def ajouter(self, cle: str) -> None:
        raise NotImplementedError

    def estimation(self) -> float:
        raise NotImplementedError

    def fusionner(self, autre: HyperLogLog) -> HyperLogLog:
        raise NotImplementedError


# ============================================================================ Répartir sans tout déplacer


class Anneau:
    """Défi 247 🔴 — Quel serveur calcule le score de ce client ? Le hachage cohérent : ajouter un serveur ne
    déplace que sa part.

        >>> anneau = Anneau(["score-1", "score-2", "score-3", "score-4"])
        >>> len(anneau), anneau.serveur_pour("C000001"), anneau.serveur_pour("C000002")
        (4, 'score-3', 'score-1')
        >>> anneau.serveur_pour("score-2#7")      # une clé posée PILE sur un point de score-2
        'score-2'
        >>> anneau.ajouter("score-5")
        >>> anneau.retirer("score-1")
        >>> len(anneau), anneau.serveur_pour("C000002")
        (4, 'score-5')

    L'idée (Karger et al., 1997 ; Amazon Dynamo, Cassandra, les répartiteurs de charge) : on place les serveurs
    sur un cercle de 2**64 positions, chacun en `repliques` points (des « nœuds virtuels »). Un client va au
    premier point qu'on rencontre en tournant à partir de SA position. Ajouter un serveur pose de nouveaux points :
    seuls les clients situés juste avant eux changent de serveur, et ils vont tous chez le nouveau. Avec
    `empreinte(client) % n`, passer de 4 à 5 serveurs déplace 4 clients sur 5 (16 028 sur 20 000).

    Ce qu'il faut écrire :

    - `__init__(serveurs=(), repliques=100)` : un anneau avec ces serveurs (n'importe quel itérable de noms) ;
    - `ajouter(serveur)` et `retirer(serveur)` ;
    - `serveur_pour(cle)` : le serveur du client `cle` ;
    - `len(anneau)` (`__len__`) : le nombre de serveurs.

    Les règles :

    - le point numéro i (de 0 à repliques - 1) du serveur s est à la position `empreinte(f"{s}#{i}")` ; la clé est
      à la position `empreinte(cle)` ;
    - `serveur_pour(cle)` : le serveur du premier point dont la position est SUPÉRIEURE OU ÉGALE à celle de la clé
      (une clé pile sur un point lui appartient) ; s'il n'y en a pas, le tout premier point (on a fait le tour du
      cercle) ;
    - le correcteur pose 200 serveurs de 100 points (20 000 points) et demande 20 000 clients : parcourir tous les
      points à chaque client (400 millions de comparaisons) est arrêté. Gardez les positions TRIÉES et cherchez
      par dichotomie (`bisect`) ;
    - `retirer` enlève tous les points du serveur, et seulement eux ;
    - erreurs : repliques qui n'est pas un int → TypeError, < 1 → ValueError ; un nom de serveur qui n'est pas un
      texte → TypeError, vide → ValueError, déjà présent (à l'ajout, y compris en double dans `serveurs`) ou
      absent (au retrait) → ValueError ; une clé qui n'est pas un texte → TypeError ; `serveur_pour` sur un
      anneau sans serveur → ValueError ;
    - deux anneaux ne partagent rien.

    Ce que vous apprenez : répartir des données entre machines sans tout déplacer à chaque changement (la base
    des caches et des bases distribuées), la recherche par dichotomie, et le modulo qui ferme le cercle.

    Indice : une liste triée de (position, serveur) et la liste des seules positions ; `bisect.bisect_left` (pas
    `bisect_right`, qui sauterait un point pile sur la clé) ; `i % len(points)` pour faire le tour.
    """

    def __init__(self, serveurs=(), repliques: int = 100) -> None:
        raise NotImplementedError

    def ajouter(self, serveur: str) -> None:
        raise NotImplementedError

    def retirer(self, serveur: str) -> None:
        raise NotImplementedError

    def serveur_pour(self, cle: str) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

"""Série 37 — Facturer au centime près : prorata, répartition, TVA, jours fériés, relances, rapprochement.

🧾 VOUS VOULEZ DÉVELOPPER ? UN JOUR, VOTRE CODE MANIPULERA DE L'ARGENT. ICI, VOUS APPRENEZ À NE PAS EN PERDRE.

Nadia vient d'arriver au service facturation. Deux nouvelles l'attendent. La comptable a trouvé des écarts de
quelques centimes sur près d'une facture sur deux : l'ancien code calculait en float, arrondissait avec `round()`
et la TVA ligne par ligne (145,33 € au lieu de 145,36 € pour la facture de C04217). Et l'équipe data a remarqué
qu'une partie des départs du mois n'en étaient pas : un prélèvement refusé (carte expirée, compte vide un 28 du
mois), que personne n'a relancé, puis l'abonnement coupé. C'est le CHURN INVOLONTAIRE : le client voulait rester.
Cette série vous fait écrire les six calculs d'une facturation qui tombe juste et qui rattrape les paiements
manqués :

    224 prorata          ce que coûte un changement de formule le 17 février (qui n'a pas 30 jours)
    225 repartir         étaler 10 € de remise sur six lignes sans perdre ni inventer un centime
    226 facturer         la facture de novembre : HT, remise, TVA par taux, TTC
    227 jours_feries     les onze jours fériés de n'importe quelle année (Pâques compris)
    228 relances         quand représenter un prélèvement refusé, et quand suspendre
    229 rapprocher       quel virement paie quelle facture, sur un relevé bancaire en désordre

Prérequis : la série 00 (fonctions, listes, dictionnaires, boucles). Les défis 44 (`datetime`) et 51 (les
expressions régulières) et la série 17 (le `Montant` en centimes, `Decimal`) l'éclairent, mais tout ce qu'il
faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un montant s'écrit en NOMBRE ENTIER DE CENTIMES : 29,99 € → 2999. Un float ne sait pas écrire 0.1, ni 1.005
  (`1.005 * 100` vaut 100.49999999999999) : il perd des centimes sans prévenir ;
- un prix unitaire peut avoir plus de deux décimales (un SMS à 0.045 €). Il arrive en TEXTE, et se lit avec
  `Decimal("0.045")`, qui est exact. Attention : `Decimal("12,99")` lève `decimal.InvalidOperation`, qui n'est PAS
  une ValueError ;
- `round()` arrondit les demis au nombre PAIR : `round(498.5)` vaut 498, `round(499.5)` vaut 500 (l'« arrondi du
  banquier »). Sur une facture, un demi-centime s'arrondit vers le HAUT : en entiers, a / b arrondi au plus
  proche s'écrit `(2 * a + b) // (2 * b)` ; avec Decimal, `x.quantize(Decimal(1), rounding=ROUND_HALF_UP)` ;
- la TVA française a quatre taux : 20 % (normal : le forfait, les SMS), 10 % (la télévision payante), 5,5 %
  (les livres, même audio) et 2,1 % (la presse). Une facture donne, pour CHAQUE taux, la base HT et la TVA ;
- une date se manipule avec `datetime.date` : `date(2026, 3, 1) - date(2026, 2, 1)` est un `timedelta` dont
  `.days` vaut 28 ; `jour + timedelta(days=1)` est le lendemain ; `jour.weekday()` vaut 0 le lundi… 5 le samedi
  et 6 le dimanche ;
- un JOUR OUVRÉ est un jour de semaine (lundi à vendredi) qui n'est pas férié : la banque ne prélève que ces
  jours-là ;
- RAPPROCHER (ou « lettrer ») un relevé bancaire, c'est dire quel paiement règle quelle facture. Ce qui n'a
  été payé par personne reste dû ; ce qui a été payé en trop est un CRÉDIT du client.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_37 import *
    >>> lignes_exemple()[1]
    {'libelle': 'SMS hors forfait', 'quantite': 1234, 'prix_unitaire': '0.045', 'tva': '20'}
    >>> euros(2999), DELAIS, GRACE
    ('29,99 €', (1, 3, 7, 14), 7)

Quand les six défis sont réussis, Nadia peut répondre à la comptable et à l'équipe data :

    >>> from datetime import date
    >>> debut, fin, changement = date(2026, 2, 1), date(2026, 3, 1), date(2026, 2, 17)
    >>> prorata(2999, debut, fin, changement) - prorata(1999, debut, fin, changement)   # Essentiel → Premium
    428
    >>> facture = facturer(lignes_exemple(), remise=1000)
    >>> euros(facture.total_ht), euros(facture.total_tva), euros(facture.total_ttc)
    ('114,02 €', '19,61 €', '133,63 €')
    >>> [(t.taux, euros(t.tva)) for t in facture.par_taux]
    [('20', '18,12 €'), ('10', '1,15 €'), ('5.5', '0,15 €'), ('2.1', '0,19 €')]
    >>> len(jours_feries(2026)), jours_feries(2026)[4]                 # l'Ascension
    (11, datetime.date(2026, 5, 14))
    >>> r = relances(date(2026, 11, 5))                                # les prélèvements refusés du 5 novembre
    >>> [j.day for j in r.essais], r.suspension
    ([6, 9, 12, 19], datetime.date(2026, 11, 26))
    >>> lettrage = rapprocher(creances_exemple(), paiements_exemple())
    >>> len(lettrage.restes), euros(sum(lettrage.restes.values()))
    (303, '9 285,01 €')
    >>> len(lettrage.credits), euros(sum(lettrage.credits.values())), len(lettrage.non_affectes)
    (52, '140,17 €', 15)

Le changement de formule coûte 4,28 € au client (avec « 30 jours », on lui aurait facturé 4,00 €). La facture de
novembre tombe juste au centime sur les quatre taux. Et sur 6 035 factures, 303 restent dues, dont 181 factures de
novembre que personne n'a payées du tout : des prélèvements refusés le 5. Relancés le 6, le 9, le 12 et le 19
novembre, ce sont autant de clients qui ne partiront pas sans l'avoir voulu.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. L'ordre proposé est le plus progressif : le 226
   appelle votre `repartir` (défi 225), et le 228 votre `jours_feries` (défi 227).
2. Lancez `python scripts/defis.py --defi 224` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_37.py`.

Les outils communs (`lignes_exemple`, `creances_exemple`, `paiements_exemple`, `generer_lignes`,
`generer_creances`, `generer_paiements`, `euros`, `TAUX_TVA`, `DELAIS`, `GRACE`, et les données et résultats
`TotalTaux`, `Facture`, `Relances`, `Creance`, `Paiement`, `Affectation`, `Rapprochement`) sont dans
`defis/facturation.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import re  # noqa: F401 — re.compile, re.IGNORECASE, motif.findall(texte)
from datetime import date, datetime, timedelta  # noqa: F401
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation  # noqa: F401

from defis.facturation import (  # noqa: F401
    DELAIS,
    GRACE,
    TAUX_TVA,
    Affectation,
    Creance,
    Facture,
    Paiement,
    Rapprochement,
    Relances,
    TotalTaux,
    creances_exemple,
    euros,
    generer_creances,
    generer_lignes,
    generer_paiements,
    lignes_exemple,
    paiements_exemple,
)

# ============================================================================ Facturer


def prorata(montant: int, debut: date, fin: date, jour: date) -> int:
    """Défi 224 🟢 — La part de `montant` (en centimes) qui correspond aux jours de `jour` à `fin`.

        >>> from datetime import date
        >>> prorata(2999, date(2026, 2, 1), date(2026, 3, 1), date(2026, 2, 17))   # 12 jours sur 28
        1285
        >>> prorata(2999, date(2028, 2, 1), date(2028, 3, 1), date(2028, 2, 17))   # 13 jours sur 29 (bissextile)
        1344
        >>> prorata(997, date(2026, 4, 1), date(2026, 5, 1), date(2026, 4, 16))    # 498.5 centimes → 499
        499
        >>> prorata(2999, date(2026, 2, 1), date(2026, 3, 1), date(2026, 3, 1))    # plus aucun jour
        0

    Ici (1er exemple) : C04217 paie 29,99 € par mois pour la période du 1er février au 1er mars. Il résilie (ou
    change de formule) le 17 février : il reste 12 jours (du 17 au 28 inclus) sur les 28 de février. Sa part vaut
    2999 × 12 / 28 = 1285,29 centimes, donc 1285. En changeant de formule, on lui rend le prorata de l'ancienne et
    on lui facture celui de la nouvelle.

    Les règles :

    - la période va de `debut` (inclus) à `fin` (EXCLU : c'est le premier jour de la période suivante). Elle
      compte (fin - debut).days jours : 28 en février 2026, 29 en février 2028, 365 ou 366 pour un an ;
    - il reste (fin - jour).days jours. Le résultat est montant × restants / jours de la période, arrondi au
      centime le plus proche, les demis vers le HAUT (498.5 → 499), renvoyé en int ;
    - montant est un int >= 0 (en centimes). Un float ou un bool → TypeError ; un montant négatif → ValueError ;
    - debut, fin et jour sont des `datetime.date`. Un `datetime` (avec une heure) ou un texte → TypeError.
      Attention : un datetime EST une date pour `isinstance` ; testez-le d'abord ;
    - debut doit être avant fin, et jour entre debut et fin (les deux compris) : sinon ValueError.

    Ce que vous apprenez : compter des jours avec les vrais calendriers, décider d'un arrondi et le tenir, et
    rester en ENTIERS (montant / periode * restants en float donne 499.49999… au lieu de 499.5).

    Indice : `(fin - debut).days`, puis `(2 * montant * restants + periode) // (2 * periode)`.
    """
    raise NotImplementedError


def repartir(total: int, poids: list[int]) -> list[int]:
    """Défi 225 🟡 — Répartir `total` centimes au prorata des `poids`, sans perdre ni inventer un centime.

        >>> repartir(1000, [1, 1, 1])
        [334, 333, 333]
        >>> repartir(1000, [3999, 5553, 1249, 999, 300, 302])        # 10 € de remise sur les six lignes
        [322, 448, 101, 81, 24, 24]
        >>> repartir(100, [0, 1, 0, 2])                              # un poids nul ne reçoit rien
        [0, 33, 0, 67]
        >>> repartir(-1000, [1, 1, 1])                               # le symétrique de 1000
        [-334, -333, -333]
        >>> repartir(3, [3, 13, 11, 3])                              # égalité de restes : le premier
        [1, 1, 1, 0]

    Ici (1er exemple) : chaque part exacte vaut 333,33… On donne à chacun la partie entière (333), il manque
    1000 − 999 = 1 centime, qu'on donne à la part dont le RESTE (la partie après la virgule) est le plus grand.
    Les trois sont à égalité : le premier le prend. Avec `round()` sur chaque part, la somme ferait 999.

    Les règles (la « méthode du plus fort reste ») :

    - la part exacte de chaque poids p vaut total × p / somme des poids. Chaque part reçoit d'abord sa partie
      ENTIÈRE ; les centimes qui manquent pour faire `total` vont, un chacun, aux parts dont le reste est le plus
      grand ; à égalité de reste, la première dans la liste passe avant ;
    - le résultat est une NOUVELLE liste d'int, de la même longueur que `poids`, dont la somme vaut exactement
      `total`, et où chaque part vaut l'arrondi inférieur ou supérieur de sa part exacte ;
    - un `total` négatif se répartit comme |total|, avec le signe moins (−1000 → [−334, −333, −333]) ;
    - total et chaque poids sont des int (float ou bool → TypeError) ; un poids négatif → ValueError ; aucun
      poids, ou des poids tous nuls → ValueError ;
    - ne modifiez pas `poids`. Le correcteur répartit aussi un montant sur 100 000 poids : chercher le plus grand
      reste avec `max()` une fois par centime à donner est bien trop lent.

    Ce que vous apprenez : la répartition qui tombe juste (remises, frais de port, budget par région, sièges
    d'une assemblée), et comparer des restes ENTIERS : en float, 3 × 13 / 30 − 1 vaut 0.30000000000000004, un
    peu plus que 3 × 3 / 30 = 0.3, et l'égalité disparaît.

    Indice : `q, r = divmod(abs(total) * p, somme)` donne la partie entière et le reste (entier) d'un coup ;
    puis triez les positions par reste décroissant (`sorted(..., key=lambda i: -restes[i])` garde l'ordre des
    ex aequo) et ajoutez 1 aux premières.
    """
    raise NotImplementedError


def facturer(lignes: list[dict], remise: int = 0) -> Facture:
    """Défi 226 🟡 — Facture(lignes, par_taux, total_ht, total_tva, total_ttc), en centimes.

        >>> f = facturer([{"libelle": "Forfait", "quantite": 1, "prix_unitaire": "29.99", "tva": "20"},
        ...               {"libelle": "SMS", "quantite": 1234, "prix_unitaire": "0.045", "tva": "20"},
        ...               {"libelle": "Livre audio", "quantite": 1, "prix_unitaire": "3.00", "tva": "5.5"}])
        >>> f.lignes, f.total_ht, f.total_tva, f.total_ttc
        ([2999, 5553, 300], 8852, 1727, 10579)
        >>> f.par_taux
        [TotalTaux(taux='20', ht=8552, tva=1710), TotalTaux(taux='5.5', ht=300, tva=17)]
        >>> facturer([{"libelle": "x", "quantite": 3, "prix_unitaire": "1.005", "tva": "20"}]).lignes
        [302]
        >>> facture = facturer(lignes_exemple(), remise=1000)
        >>> facture.lignes, euros(facture.total_ttc)
        ([3677, 5105, 1148, 918, 276, 278], '133,63 €')

    Ici (1er exemple) : 1 234 SMS à 0,045 € font 55,53 € tout rond (5553). Les deux lignes à 20 % font 85,52 € HT :
    leur TVA vaut 17,104 €, donc 1710 centimes. Le livre audio (5,5 %) : 300 × 5,5 / 100 = 16,5 centimes, arrondis
    à 17 (`round(16.5)` donnerait 16). 2e exemple : 3 × 1,005 = 3,015 € → 302 centimes (en float : 301).

    Les règles :

    - chaque ligne est un dict : "libelle", "quantite" (int >= 1), "prix_unitaire" (HT, en euros, écrit en TEXTE
      avec un point : "0.045") et "tva" (un des textes de TAUX_TVA : "20", "10", "5.5", "2.1") ;
    - le HT d'une ligne = quantite × prix_unitaire, en centimes, arrondi au plus proche (demis vers le haut) ;
    - la `remise` (int de centimes, 0 par défaut) se RÉPARTIT sur les lignes au prorata de leur HT avec votre
      `repartir` (défi 225), et se déduit de chaque ligne AVANT la TVA ; `lignes` du résultat donne le HT de
      chaque ligne remise déduite, dans l'ordre reçu ;
    - pour chaque taux utilisé : ht = la somme des lignes à ce taux, tva = ht × taux / 100, arrondie UNE fois au
      centime (demis vers le haut). Pas ligne par ligne : trois lignes à 12,37 € donnent 7,42 € de TVA, pas
      3 × 2,47 = 7,41 € ;
    - `par_taux` est une liste de TotalTaux(taux, ht, tva), du taux le plus FORT au plus faible (en nombre : en
      texte, "5.5" > "20") ; total_ht et total_tva sont les sommes de `par_taux`, total_ttc leur somme ;
    - erreurs : quantité ou remise qui n'est pas un int (float, bool) → TypeError ; prix en float → TypeError
      (il a peut-être déjà perdu des centimes) ; quantité < 1, prix illisible ("12,99", "abc"), négatif ou non
      fini ("NaN", "Infinity"), taux inconnu ("19.6", ou le nombre 20), remise négative ou plus grande que le
      total HT → ValueError ;
    - ne modifiez pas `lignes`.

    Ce que vous apprenez : `Decimal` pour lire un prix sans le déformer, où se placent les arrondis d'une facture
    (une fois par ligne pour le HT, une fois par taux pour la TVA), et pourquoi une remise se répartit.

    Indice : `int((Decimal(prix) * quantite * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))` ; un dict
    {taux: ht} ; `sorted(..., key=lambda t: Decimal(t), reverse=True)`.
    """
    raise NotImplementedError


# ============================================================================ Relancer


def jours_feries(annee: int, alsace_moselle: bool = False) -> list[date]:
    """Défi 227 🟢 — Les jours fériés de l'année en France, triés, sans doublon.

        >>> [f"{j:%d/%m}" for j in jours_feries(2026)]
        ['01/01', '06/04', '01/05', '08/05', '14/05', '25/05', '14/07', '15/08', '01/11', '11/11', '25/12']
        >>> len(jours_feries(2008)), date(2008, 5, 1) in jours_feries(2008)   # Ascension le 1er mai
        (10, True)
        >>> len(jours_feries(2026, alsace_moselle=True))
        13

    Ici (1er exemple) : Pâques 2026 tombe le dimanche 5 avril, donc le lundi de Pâques le 6, l'Ascension le jeudi
    14 mai et le lundi de Pentecôte le 25 mai. Les huit autres jours ont une date fixe.

    Les règles :

    - les huit jours fixes : 1er janvier, 1er mai, 8 mai, 14 juillet, 15 août, 1er novembre, 11 novembre,
      25 décembre ;
    - les trois jours qui dépendent de Pâques : le lundi de Pâques (Pâques + 1 jour), l'Ascension (Pâques +
      39 jours, un jeudi) et le lundi de Pentecôte (Pâques + 50 jours : la Pentecôte est le dimanche Pâques + 49) ;
    - avec alsace_moselle=True, deux jours en plus : le Vendredi saint (Pâques − 2 jours) et le 26 décembre ;
    - le dimanche de Pâques se calcule ainsi (l'algorithme de Meeus, Jones et Butcher, que des entiers) :

          a = annee % 19          b, c = divmod(annee, 100)      d, e = divmod(b, 4)
          f = (b + 8) // 25       g = (b - f + 1) // 3           h = (19 * a + b - d - g + 15) % 30
          i, k = divmod(c, 4)     l = (32 + 2 * e + 2 * i - h - k) % 7
          m = (a + 11 * h + 22 * l) // 451
          mois, jour = divmod(h + l - 7 * m + 114, 31)   →   Pâques = date(annee, mois, jour + 1)

    - le résultat est une NOUVELLE liste de `date` à chaque appel (si vous gardez un cache, renvoyez-en une copie),
      triée, SANS DOUBLON (en 2008, l'Ascension tombait le 1er mai : 10 jours fériés seulement) ;
    - annee est un int (float, texte ou bool → TypeError), entre 1583 (le calendrier grégorien) et 9999 (sinon
      ValueError).

    Ce que vous apprenez : recopier une formule sans la casser (`//` et pas `/`), des dates calculées les unes à
    partir des autres avec `timedelta`, et un `set` pour ne compter qu'une fois.

    Indice : une fonction `paques(annee)`, puis un ensemble de dates et `sorted(...)`.
    """
    raise NotImplementedError


def relances(echec: date, delais=DELAIS, grace: int = GRACE) -> Relances:
    """Défi 228 🟡 — Relances(essais, suspension) : quand représenter un prélèvement refusé le jour `echec`.

        >>> from datetime import date
        >>> r = relances(date(2026, 4, 30))          # un jeudi, veille du 1er mai
        >>> [f"{j:%d/%m}" for j in r.essais], f"{r.suspension:%d/%m}"
        (['04/05', '05/05', '07/05', '15/05'], '22/05')
        >>> relances(date(2026, 12, 31)).essais[0]   # le 1er janvier 2027 est férié AUSSI
        datetime.date(2027, 1, 4)
        >>> relances(date(2026, 11, 13), delais=(1, 3), grace=0)
        Relances(essais=[datetime.date(2026, 11, 16), datetime.date(2026, 11, 17)], suspension=datetime.date(2026, 11, 17))

    Ici (1er exemple) : le prélèvement du jeudi 30 avril est refusé. 1er essai prévu le 1er mai (J+1) : férié,
    puis le week-end, donc le lundi 4. 2e essai prévu le dimanche 3 (J+3) : le premier jour ouvré APRÈS le 4, le
    mardi 5. 3e essai le jeudi 7 (J+7). 4e prévu le 14 (J+14) : c'est l'Ascension, donc le vendredi 15. Si tout
    échoue, l'abonnement est suspendu 7 jours après le dernier essai : le vendredi 22.

    Les règles :

    - un jour OUVRÉ est un lundi, mardi, mercredi, jeudi ou vendredi qui n'est pas férié. Utilisez votre
      `jours_feries` (défi 227) avec l'année du jour TESTÉ (un refus le 31 décembre se relance en janvier, et le
      1er janvier de l'année SUIVANTE est férié) ;
    - l'essai n° k est prévu à echec + delais[k] jours (comptés depuis le REFUS, jamais depuis l'essai d'avant).
      Il a lieu le premier jour ouvré à partir de cette date prévue ET strictement après l'essai précédent ;
      jamais deux essais le même jour ;
    - la suspension a lieu le premier jour ouvré à partir du dernier essai + `grace` jours (ou de echec + grace
      s'il n'y a aucun délai) ;
    - echec est une `datetime.date` (un datetime ou un texte → TypeError) ; chaque délai est un int (sinon
      TypeError) >= 1, et les délais sont strictement croissants (sinon ValueError) ; grace est un int (sinon
      TypeError) >= 0 (sinon ValueError) ;
    - ne modifiez pas `delais`.

    Ce que vous apprenez : une règle métier sur un calendrier (week-ends, fériés, changement d'année), et
    pourquoi on recalcule chaque date depuis l'événement d'origine plutôt que d'enchaîner des décalages.

    Indice : une fonction `ouvre(jour)` (`jour.weekday() >= 5` pour le week-end ; un dict {année: set des
    fériés} pour ne calculer chaque année qu'une fois) et une boucle `while not ouvre(jour): jour += timedelta(1)`.
    """
    raise NotImplementedError


def rapprocher(creances: list[Creance], paiements: list[Paiement]) -> Rapprochement:
    """Défi 229 🔴 — Rapprochement(affectations, restes, credits, non_affectes) : qui a payé quelle facture.

        >>> from datetime import date
        >>> creances = [Creance("FAC-2026-00001", "C1", 2000, date(2026, 10, 5)),
        ...             Creance("FAC-2026-00002", "C1", 3000, date(2026, 11, 5)),
        ...             Creance("FAC-2026-00003", "C2", 1500, date(2026, 11, 5))]
        >>> paiements = [Paiement("B3", date(2026, 11, 20), "C1", 4000, "VIR SOLDE"),
        ...              Paiement("B1", date(2026, 11, 6), "C1", 3000, "VIR FAC-2026-00002"),
        ...              Paiement("B2", date(2026, 11, 9), None, 1500, "VIR INST 0042")]
        >>> r = rapprocher(creances, paiements)
        >>> r.affectations
        [Affectation(paiement='B1', numero='FAC-2026-00002', montant=3000), Affectation(paiement='B3', numero='FAC-2026-00001', montant=2000)]
        >>> r.restes, r.credits, r.non_affectes
        ({'FAC-2026-00003': 1500}, {'C1': 2000}, ['B2'])

    Ici : le relevé arrive dans le désordre ; on traite B1 (6 novembre) d'abord. Il cite FAC-2026-00002, qui est
    bien à C1 : elle est soldée. B2 vient d'on ne sait qui : non affecté (C2 doit toujours ses 15 €, même si le
    montant correspond). B3 ne cite rien : il paie la plus vieille facture de C1 encore due (2000), et les 20 €
    restants deviennent un crédit de C1.

    Les règles :

    - on traite les paiements dans l'ordre de leur `jour` ; le même jour, dans l'ordre reçu ;
    - un paiement dont le client est None, ou n'a AUCUNE facture, est non affecté : son id va dans
      `non_affectes`, et rien d'autre ;
    - sinon, le paiement règle d'abord les factures que son libellé CITE : un numéro est « FAC- », 4 chiffres,
      « - », 5 chiffres, en majuscules ou en minuscules, qui n'est pas collé à une autre lettre ou un autre
      chiffre (« FAC-2026-000127 » est une faute de frappe : il ne cite PAS FAC-2026-00012). Dans l'ordre de leur
      apparition, chaque numéro une seule fois, et seulement si la facture est à CE client et pas encore soldée ;
    - ce qui reste du paiement règle ensuite les factures encore dues du client, de la plus ancienne ÉCHÉANCE à la
      plus récente (à échéance égale, par numéro) ;
    - chaque facture reçoit au plus ce qu'elle doit encore ; chaque règlement ajoute une
      Affectation(id du paiement, numéro, montant) ; ce qui reste à la fin est un CRÉDIT du client (les crédits
      d'un même client s'additionnent ; on ne s'en sert pas pour payer d'autres factures) ;
    - le résultat : `affectations` dans l'ordre où on les a faites ; `restes` {numéro: reste dû > 0} dans l'ordre
      des créances ; `credits` {client: crédit > 0} dans l'ordre où chaque crédit est apparu ; `non_affectes`
      dans l'ordre de traitement ;
    - les montants sont des int > 0 (float ou bool → TypeError, sinon ValueError) ; un numéro de facture ou un id
      de paiement en double → ValueError ;
    - ne modifiez ni `creances` ni `paiements` (pas de `.sort()` sur la liste reçue). Le correcteur rapproche
      20 000 factures et autant de paiements : parcourir toutes les factures pour chaque paiement est bien trop
      lent.

    Ce que vous apprenez : le lettrage, ce que fait chaque nuit le logiciel de comptabilité (ou l'outil de
    paiement), un index par client et un curseur au lieu d'une recherche, et une expression régulière avec ses
    frontières (`\\b`).

    Indice : `motif = re.compile(r"\\bFAC-\\d{4}-\\d{5}\\b", re.IGNORECASE)` ; un dict {numéro: reste dû} ; un
    dict {client: numéros triés par (échéance, numéro)} ; `sorted(paiements, key=lambda p: p.jour)` est stable.
    """
    raise NotImplementedError

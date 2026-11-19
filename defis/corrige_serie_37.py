"""Série 37 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_37.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours un float
caché (`montant / periode * restants`, `float("1.005")`), un `round()` qui arrondit au pair, ou une liste
parcourue en entier pour chaque paiement.

Fil rouge de la série : de l'argent, ça se compte en ENTIERS de centimes, et chaque arrondi est une décision
écrite noir sur blanc (où, quand, dans quel sens). Un centime perdu sur une facture, c'est 30 000 centimes sur
le mois, et une comptable qui ne fait plus confiance au système.

1. un prorata se calcule sur les VRAIS jours de la période (février en a 28 ou 29), la fin exclue, et s'arrondit
   au plus proche avec les demis vers le haut, en entiers : (2 × a + b) // (2 × b) ;
2. répartir un montant sans perdre de centime : la partie entière de chaque part, puis les centimes qui manquent
   aux plus grands RESTES (entiers, pas des floats), le premier à égalité ;
3. une facture : HT de chaque ligne arrondi au centime avec `Decimal` (le prix est du TEXTE), remise répartie
   sur les lignes AVANT la TVA, puis la TVA calculée une fois par TAUX, et jamais avec `round()` ;
4. les jours fériés : Pâques par l'algorithme (que des `//` et des `%`), l'Ascension 39 jours après, le lundi
   de Pentecôte 50 jours après, et un ensemble, car en 2008 l'Ascension tombait le 1er mai ;
5. les relances : chaque essai au premier jour ouvré à partir de sa date prévue ET après le précédent, avec les
   fériés de l'année du JOUR testé (un refus le 31 décembre se relance en janvier) ;
6. le rapprochement : les paiements dans l'ordre des DATES, les numéros cités d'abord (s'ils sont bien à ce
   client), puis les plus vieilles factures, le reste en crédit ; un index par client, jamais un parcours de
   toutes les factures pour chaque paiement.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import heapq
import re
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

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

UN_JOUR = timedelta(days=1)


def _entier(x, nom: str) -> int:
    """Un vrai int : ni float (déjà arrondi), ni bool (True vaut 1 en Python)."""
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"{nom} doit être un entier, pas {type(x).__name__} ({x!r})")
    return x


def _jour(x, nom: str) -> date:
    """Une date sans heure : un datetime EST une date pour isinstance, d'où le premier test."""
    if isinstance(x, datetime) or not isinstance(x, date):
        raise TypeError(f"{nom} doit être une datetime.date, pas {type(x).__name__} ({x!r})")
    return x


# ============================================================================ Défi 224


def prorata(montant: int, debut: date, fin: date, jour: date) -> int:
    for nom, valeur in (("debut", debut), ("fin", fin), ("jour", jour)):
        _jour(valeur, nom)
    if _entier(montant, "montant") < 0:
        raise ValueError(f"montant négatif : {montant}")
    if not debut < fin:
        raise ValueError(f"période vide ou à l'envers : du {debut} au {fin}")
    if not debut <= jour <= fin:
        raise ValueError(f"le {jour} n'est pas dans la période du {debut} au {fin}")
    periode = (fin - debut).days  # les VRAIS jours : 28, 29, 30, 31, 365 ou 366
    restants = (fin - jour).days  # `fin` est exclue : c'est le premier jour de la période suivante
    # montant × restants / periode, arrondi au plus proche (les demis vers le haut), sans jamais quitter les entiers.
    # (round() arrondirait 498.5 à 498 ; montant / periode * restants en float donne 499.49999… pour 499.5.)
    return (2 * montant * restants + periode) // (2 * periode)


# ============================================================================ Défi 225


def repartir(total: int, poids: list[int]) -> list[int]:
    _entier(total, "total")
    poids = list(poids)
    for p in poids:
        if _entier(p, "chaque poids") < 0:
            raise ValueError(f"poids négatif : {p}")
    somme = sum(poids)
    if somme == 0:
        raise ValueError("aucun poids, ou des poids tous nuls : on ne peut rien répartir")
    signe, a_repartir = (-1 if total < 0 else 1), abs(total)
    parts, restes = [], []
    for p in poids:
        q, r = divmod(a_repartir * p, somme)  # la part exacte vaut q + r / somme : r est un RESTE entier
        parts.append(q)
        restes.append(r)
    manque = a_repartir - sum(parts)  # toujours entre 0 et len(poids) - 1
    # Les `manque` plus grands restes, le premier à égalité : un tri (ou un tas), pas un max() par centime.
    for i in heapq.nsmallest(manque, range(len(poids)), key=lambda i: (-restes[i], i)):
        parts[i] += 1
    return [signe * x for x in parts]


# ============================================================================ Défi 226

_CENT = Decimal(100)


def _au_centime(x: Decimal) -> int:
    """Arrondi au plus proche, les demis vers le haut (ROUND_HALF_UP) : 12.5 → 13. round(12.5) donnerait 12."""
    return int(x.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _prix(texte) -> Decimal:
    if isinstance(texte, float):
        raise TypeError(f"prix_unitaire en float ({texte!r}) : il a peut-être déjà perdu des centimes, "
                        "passez une chaîne comme '1.005'")
    if not isinstance(texte, str):
        raise TypeError(f"prix_unitaire doit être une chaîne, pas {type(texte).__name__}")
    try:
        prix = Decimal(texte.strip())
    except InvalidOperation:  # ce n'est PAS une ValueError : il faut la convertir soi-même
        raise ValueError(f"prix_unitaire illisible : {texte!r} (un point, pas une virgule)") from None
    if not prix.is_finite() or prix < 0:
        raise ValueError(f"prix_unitaire impossible : {texte!r}")
    return prix


def facturer(lignes: list[dict], remise: int = 0) -> Facture:
    if _entier(remise, "remise") < 0:
        raise ValueError(f"remise négative : {remise}")
    bruts, taux = [], []
    for ligne in lignes:
        quantite = _entier(ligne["quantite"], "quantite")
        if quantite < 1:
            raise ValueError(f"quantité impossible : {quantite}")
        prix = _prix(ligne["prix_unitaire"])
        if ligne["tva"] not in TAUX_TVA:
            raise ValueError(f"taux de TVA inconnu : {ligne['tva']!r} (attendu : un de {TAUX_TVA})")
        bruts.append(_au_centime(prix * quantite * _CENT))  # le HT de la ligne, arrondi UNE fois
        taux.append(ligne["tva"])
    if remise > sum(bruts):
        raise ValueError(f"remise de {remise} centimes sur une facture de {sum(bruts)} centimes")
    # La remise baisse la base de chaque taux : on la répartit sur les lignes, au prorata de leur HT, AVANT la TVA.
    parts = repartir(remise, bruts) if remise else [0] * len(bruts)
    nets = [b - p for b, p in zip(bruts, parts)]
    ht_par_taux: dict[str, int] = {}
    for t, net in zip(taux, nets):
        ht_par_taux[t] = ht_par_taux.get(t, 0) + net
    par_taux = [TotalTaux(t, ht, _au_centime(Decimal(ht) * Decimal(t) / _CENT))  # Decimal("5.5") est exact
                for t, ht in sorted(ht_par_taux.items(), key=lambda kv: Decimal(kv[0]), reverse=True)]
    total_ht = sum(t.ht for t in par_taux)
    total_tva = sum(t.tva for t in par_taux)
    return Facture(nets, par_taux, total_ht, total_tva, total_ht + total_tva)


# ============================================================================ Défi 227


def paques(annee: int) -> date:
    """Le dimanche de Pâques (algorithme « anonyme » de Meeus, Jones et Butcher, calendrier grégorien)."""
    a = annee % 19
    b, c = divmod(annee, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    mois, jour = divmod(h + l - 7 * m + 114, 31)
    return date(annee, mois, jour + 1)


def jours_feries(annee: int, alsace_moselle: bool = False) -> list[date]:
    if not 1583 <= _entier(annee, "annee") <= 9999:
        raise ValueError(f"année hors du calendrier grégorien : {annee}")
    p = paques(annee)
    jours = {date(annee, 1, 1), p + timedelta(days=1), date(annee, 5, 1), date(annee, 5, 8),
             p + timedelta(days=39),  # Ascension : le jeudi, 39 jours après le dimanche de Pâques
             p + timedelta(days=50),  # lundi de Pentecôte : Pentecôte = Pâques + 49 (un dimanche), + 1
             date(annee, 7, 14), date(annee, 8, 15), date(annee, 11, 1), date(annee, 11, 11), date(annee, 12, 25)}
    if alsace_moselle:
        jours |= {p - timedelta(days=2), date(annee, 12, 26)}  # Vendredi saint et Saint-Étienne
    return sorted(jours)  # un ENSEMBLE d'abord : en 2008, l'Ascension tombait le 1er mai


# ============================================================================ Défi 228


def relances(echec: date, delais=DELAIS, grace: int = GRACE) -> Relances:
    _jour(echec, "echec")
    delais = list(delais)
    for d in delais:
        if _entier(d, "chaque délai") < 1:
            raise ValueError(f"délai impossible : {d} (au moins 1 jour)")
    if any(b <= a for a, b in zip(delais, delais[1:])):
        raise ValueError(f"délais non strictement croissants : {delais}")
    if _entier(grace, "grace") < 0:
        raise ValueError(f"grâce négative : {grace}")
    feries: dict[int, set[date]] = {}  # un calcul par ANNÉE rencontrée, pas par jour testé

    def ouvre(jour: date) -> bool:
        if jour.weekday() >= 5:  # 5 = samedi, 6 = dimanche
            return False
        if jour.year not in feries:
            feries[jour.year] = set(jours_feries(jour.year))
        return jour not in feries[jour.year]

    def premier_ouvre(jour: date) -> date:
        while not ouvre(jour):
            jour += UN_JOUR
        return jour

    essais: list[date] = []
    for d in delais:
        prevu = echec + timedelta(days=d)  # toujours compté depuis le REFUS, pas depuis l'essai décalé
        if essais:
            prevu = max(prevu, essais[-1] + UN_JOUR)  # jamais deux essais le même jour
        essais.append(premier_ouvre(prevu))
    suspension = premier_ouvre((essais[-1] if essais else echec) + timedelta(days=grace))
    return Relances(essais, suspension)


# ============================================================================ Défi 229

_NUMERO = re.compile(r"\bFAC-\d{4}-\d{5}\b", re.IGNORECASE)  # \b : « FAC-2026-000127 » ne cite PAS FAC-2026-00012


def rapprocher(creances: list[Creance], paiements: list[Paiement]) -> Rapprochement:
    reste: dict[str, int] = {}
    proprietaire: dict[str, str] = {}
    par_client: dict[str, list[Creance]] = {}
    for c in creances:
        if _entier(c.montant, "montant d'une facture") <= 0:
            raise ValueError(f"facture {c.numero} de montant {c.montant}")
        if c.numero in reste:
            raise ValueError(f"numéro de facture en double : {c.numero}")
        reste[c.numero] = c.montant
        proprietaire[c.numero] = c.client
        par_client.setdefault(c.client, []).append(c)
    # La file de chaque client, de la plus vieille échéance à la plus récente (puis par numéro), et un curseur :
    # tout ce qui est avant le curseur est soldé. Chaque facture n'est donc passée qu'une fois en tout.
    files = {client: [c.numero for c in sorted(liste, key=lambda c: (c.echeance, c.numero))]
             for client, liste in par_client.items()}
    curseur = dict.fromkeys(files, 0)
    vus: set[str] = set()
    for p in paiements:
        if _entier(p.montant, "montant d'un paiement") <= 0:
            raise ValueError(f"paiement {p.id} de montant {p.montant}")
        if p.id in vus:
            raise ValueError(f"identifiant de paiement en double : {p.id}")
        vus.add(p.id)

    affectations: list[Affectation] = []
    credits: dict[str, int] = {}
    non_affectes: list[str] = []

    def payer(paiement: str, numero: str, disponible: int) -> int:
        montant = min(reste[numero], disponible)
        reste[numero] -= montant
        affectations.append(Affectation(paiement, numero, montant))
        return disponible - montant

    for p in sorted(paiements, key=lambda p: p.jour):  # sorted est STABLE : même jour → ordre reçu
        if p.client not in files:
            non_affectes.append(p.id)
            continue
        disponible = p.montant
        cites = dict.fromkeys(m.upper() for m in _NUMERO.findall(p.libelle or ""))  # sans doublon, dans l'ordre
        for numero in cites:
            if disponible and proprietaire.get(numero) == p.client and reste[numero] > 0:
                disponible = payer(p.id, numero, disponible)
        file, i = files[p.client], curseur[p.client]
        while disponible and i < len(file):
            if reste[file[i]] == 0:
                i += 1
                continue
            disponible = payer(p.id, file[i], disponible)
        curseur[p.client] = i
        if disponible:
            credits[p.client] = credits.get(p.client, 0) + disponible
    return Rapprochement(affectations, {n: r for n, r in reste.items() if r > 0}, credits, non_affectes)


# ============================================================================ Des tests comme dans tests/


def test_fevrier_n_a_pas_trente_jours():
    assert prorata(2999, date(2026, 2, 1), date(2026, 3, 1), date(2026, 2, 17)) == 1285   # 12 jours sur 28
    assert prorata(2999, date(2028, 2, 1), date(2028, 3, 1), date(2028, 2, 17)) == 1344   # 13 jours sur 29


def test_un_demi_centime_s_arrondit_vers_le_haut():
    assert prorata(997, date(2026, 4, 1), date(2026, 5, 1), date(2026, 4, 16)) == 499     # 498.5 → 499


def test_la_repartition_tombe_juste():
    assert repartir(1000, [1, 1, 1]) == [334, 333, 333]
    assert repartir(-1000, [1, 1, 1]) == [-334, -333, -333]
    assert repartir(3, [3, 13, 11, 3]) == [1, 1, 1, 0]   # en float, le 2e reste (0.30000000000000004) gagnerait


def test_la_tva_se_calcule_par_taux():
    trois = [{"libelle": "x", "quantite": 1, "prix_unitaire": "12.37", "tva": "20"}] * 3
    assert facturer(trois).total_tva == 742   # ligne par ligne : 3 × 247 = 741


def test_l_ascension_peut_tomber_le_1er_mai():
    assert len(jours_feries(2008)) == 10 and date(2008, 5, 1) in jours_feries(2008)


def test_un_refus_le_31_decembre_se_relance_en_janvier():
    assert relances(date(2026, 12, 31)).essais[0] == date(2027, 1, 4)   # le 1er janvier 2027 est férié


def test_un_numero_d_un_autre_client_ne_paie_rien():
    creances = [Creance("FAC-2026-00001", "A", 1000, date(2026, 11, 5)),
                Creance("FAC-2026-00002", "B", 1000, date(2026, 11, 5))]
    r = rapprocher(creances, [Paiement("P1", date(2026, 11, 6), "A", 1000, "VIR FAC-2026-00002")])
    assert r.restes == {"FAC-2026-00002": 1000}

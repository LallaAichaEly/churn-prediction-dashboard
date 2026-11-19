"""Série 38 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_38.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours un
dénominateur (le groupe, pas toute la clientèle ; les partis, pas les sélectionnés), une valeur absente qui
disparaît sans bruit, ou une comparaison en float tout près d'une limite (4/5, 0.55 × 100).

Fil rouge de la série : un modèle peut être juste « en moyenne » et injuste pour un groupe. On ne le voit qu'en
mesurant groupe par groupe, avec le bon dénominateur, sans perdre personne en route, et en se demandant à chaque
fois si l'écart vient de l'attribut lui-même ou de ce qui l'accompagne.

1. les taux d'un groupe se calculent sur CE groupe : la sélection sur son effectif, le rappel (tvp) sur ses
   partis, les faux positifs sur ses restés, la précision sur ses sélectionnés ; et None quand on ne peut pas
   diviser, car 0.0 voudrait dire « mesuré, et nul » ;
2. la règle des quatre cinquièmes compare chaque groupe au MIEUX traité, les petits groupes mis de côté, et
   la limite 4/5 se teste en ENTIERS : 5 × s × n_ref < 4 × s_ref × n ;
3. un écart brut peut venir de la composition des groupes (paradoxe de Simpson) : on le standardise en donnant
   à chaque groupe la répartition par strate de toute la clientèle, sur les strates communes ;
4. retirer la colonne « âge » ne retire pas l'âge : le V de Cramér mesure ce que chaque autre colonne en dit,
   en comptant AUSSI les cases vides du tableau croisé ;
5. le test de situation : le même client, une seule chose changée, et des COPIES envoyées au modèle par lots ;
6. l'égalité des chances : pour chaque groupe, le plus haut seuil qui atteint le rappel visé, en comptant
   exactement (`Fraction(str(0.55))` × 100 vaut 55 ; `0.55 * 100` vaut 55.00000000000001).

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from numbers import Real

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

# ============================================================================ outils communs aux six défis


def _absent(valeur) -> bool:
    """None, ou NaN (un NaN n'est égal à rien, pas même à lui-même : `valeur != valeur`)."""
    return valeur is None or (isinstance(valeur, float) and math.isnan(valeur))


def _cle(client: dict, colonne: str):
    """La valeur de `colonne` pour ce client ; None si elle est absente (clé manquante, None ou NaN)."""
    valeur = client.get(colonne)
    return None if _absent(valeur) else valeur


def _nombre(valeur, nom: str) -> None:
    if isinstance(valeur, bool) or not isinstance(valeur, Real):
        raise TypeError(f"{nom} doit être un nombre, pas {type(valeur).__name__}")


def _seuil(seuil) -> None:
    _nombre(seuil, "seuil")
    if not 0 <= seuil <= 1:  # NaN échoue aussi à cette comparaison
        raise ValueError(f"seuil hors de [0, 1] : {seuil!r}")


def _score(client: dict) -> float:
    score = client["score"]
    _nombre(score, "score")
    if not 0 <= score <= 1:
        raise ValueError(f"score hors de [0, 1] : {score!r}")
    return score


def _parti(client: dict) -> bool:
    parti = client["parti"]
    if parti not in (0, 1):
        raise ValueError(f"parti vaut 0 ou 1, pas {parti!r}")
    return parti == 1


# ============================================================================ Mesurer


def taux_par_groupe(clients: list[dict], colonne: str, seuil: float = SEUIL) -> dict:
    """Défi 230 — Un Groupe par valeur de `colonne`, dans l'ordre d'apparition, l'absence formant le groupe None."""
    _seuil(seuil)
    comptes: dict = {}  # groupe → [effectif, sélectionnés, positifs, vrais positifs, faux positifs]
    for client in clients:
        selectionne, parti = _score(client) >= seuil, _parti(client)
        c = comptes.setdefault(_cle(client, colonne), [0, 0, 0, 0, 0])
        c[0] += 1
        c[1] += selectionne
        c[2] += parti
        c[3] += selectionne and parti
        c[4] += selectionne and not parti
    return {groupe: _groupe(*c) for groupe, c in comptes.items()}


def _groupe(effectif: int, selectionnes: int, positifs: int, vp: int, fp: int) -> Groupe:
    negatifs = effectif - positifs
    return Groupe(effectif, selectionnes, positifs, vp, fp,
                  taux_selection=selectionnes / effectif,
                  tvp=vp / positifs if positifs else None,          # le rappel : parmi ceux qui sont PARTIS
                  tfp=fp / negatifs if negatifs else None,          # parmi ceux qui sont RESTÉS
                  precision=vp / selectionnes if selectionnes else None)


def quatre_cinquiemes(groupes: dict, min_effectif: int = 30) -> Impact:
    """Défi 231 — La règle des quatre cinquièmes, comparée en entiers."""
    if isinstance(min_effectif, bool) or not isinstance(min_effectif, int):
        raise TypeError("min_effectif doit être un int")
    if min_effectif < 1:
        raise ValueError("min_effectif doit valoir au moins 1")
    juges, ignores = [], []
    for nom, groupe in groupes.items():
        s, n = groupe.selectionnes, groupe.effectif
        if not 0 <= s <= n:
            raise ValueError(f"groupe {nom!r} : {s} sélectionnés sur {n}")
        (juges if n >= min_effectif else ignores).append(nom)
    if not juges:
        return Impact(None, {}, [], ignores)
    reference = juges[0]
    for nom in juges[1:]:  # s / n > s_ref / n_ref, sans diviser ; à égalité, le premier reste
        if groupes[nom].selectionnes * groupes[reference].effectif > \
                groupes[reference].selectionnes * groupes[nom].effectif:
            reference = nom
    s_ref, n_ref = groupes[reference].selectionnes, groupes[reference].effectif
    if s_ref == 0:  # personne n'a rien reçu : aucun rapport ne veut rien dire
        return Impact(reference, {nom: None for nom in juges}, [], ignores)
    ratios = {nom: Fraction(groupes[nom].selectionnes * n_ref, groupes[nom].effectif * s_ref) for nom in juges}
    signales = [nom for nom in juges if ratios[nom] < Fraction(4, 5)]
    return Impact(reference, {nom: float(r) for nom, r in ratios.items()}, signales, ignores)


def taux_standardises(clients: list[dict], colonne: str, strate: str, seuil: float = SEUIL) -> Standardisation:
    """Défi 232 — Taux de sélection bruts, et standardisés sur la répartition par strate de toute la clientèle."""
    if colonne == strate:
        raise ValueError("on ne standardise pas un groupe par lui-même")
    _seuil(seuil)
    effectifs, selections, par_strate = Counter(), Counter(), Counter()
    groupes: dict = {}  # des dict servent d'ensembles ORDONNÉS (ordre d'apparition)
    strates: dict = {}
    for client in clients:  # UNE passe : tout se compte au passage
        g, s = _cle(client, colonne), _cle(client, strate)
        groupes.setdefault(g, None)
        strates.setdefault(s, None)
        effectifs[g, s] += 1
        selections[g, s] += _score(client) >= seuil
        par_strate[s] += 1
    if not groupes:
        return Standardisation({}, {}, [], [])
    communes = [s for s in strates if all(effectifs[g, s] for g in groupes)]
    exclues = [s for s in strates if not all(effectifs[g, s] for g in groupes)]
    if not communes:
        raise ValueError("aucune strate commune à tous les groupes : rien de comparable")
    total = sum(par_strate[s] for s in communes)
    brut = {g: sum(selections[g, s] for s in strates) / sum(effectifs[g, s] for s in strates) for g in groupes}
    standardise = {g: float(sum(Fraction(par_strate[s], total) * Fraction(selections[g, s], effectifs[g, s])
                                for s in communes))
                   for g in groupes}
    return Standardisation(brut, standardise, communes, exclues)


# ============================================================================ Chercher la cause


def proxies(clients: list[dict], protegee: str, colonnes) -> list[Proxy]:
    """Défi 233 — Le V de Cramér entre `protegee` et chaque colonne, du plus fort au plus faible."""
    colonnes = list(colonnes)
    if protegee in colonnes:
        raise ValueError(f"{protegee!r} ne peut pas être son propre indice")
    if len(set(colonnes)) != len(colonnes):
        raise ValueError("une colonne est demandée deux fois")
    if not clients:
        raise ValueError("aucun client : rien à mesurer")
    attribut = [_cle(client, protegee) for client in clients]
    a = Counter(attribut)
    resultats = []
    for colonne in colonnes:
        valeurs = [_cle(client, colonne) for client in clients]
        b = Counter(valeurs)
        croise = Counter(zip(attribut, valeurs))
        k = min(len(a), len(b))
        if k == 1:
            v = 0.0  # une seule valeur d'un côté : la colonne ne peut rien révéler
        else:
            # χ² = Σ (n_ij − e_ij)² / e_ij sur TOUTES les cases, e_ij = a_i × b_j / n. Développé, c'est
            # n × (Σ n_ij² / (a_i × b_j) − 1), où seules les cases non vides comptent : les cases vides sont
            # dans le « − 1 ». En fractions, aucun arrondi avant la racine.
            somme = sum(Fraction(m * m, a[i] * b[j]) for (i, j), m in croise.items())
            v = math.sqrt((somme - 1) / (k - 1))
        resultats.append(Proxy(colonne, v, len(b)))
    return sorted(resultats, key=lambda p: -p.v)  # sorted est stable : à égalité, l'ordre de `colonnes`


def testing(modele, clients: list[dict], colonne: str, valeurs, seuil: float = SEUIL) -> list[Bascule]:
    """Défi 234 — Le test de situation : le même client, avec une autre valeur de `colonne`, est-il traité pareil ?"""
    _seuil(seuil)
    valeurs = list(valeurs)
    if len(valeurs) < 2 or any(v in valeurs[:i] for i, v in enumerate(valeurs)):
        raise ValueError("il faut au moins deux valeurs, toutes différentes")
    testes = [client for client in clients if _cle(client, colonne) in valeurs]
    origines = [_cle(client, colonne) for client in testes]
    avant = [score >= seuil for score in _scorer(modele, testes)]  # le modèle d'aujourd'hui, pas le champ "score"
    comptes = {(o, t): [0, 0, 0] for o in valeurs for t in valeurs if o != t}
    for testee in valeurs:
        positions = [i for i, o in enumerate(origines) if o != testee]
        copies = [{**testes[i], colonne: testee} for i in positions]  # des COPIES : l'original ne bouge jamais
        for i, score in zip(positions, _scorer(modele, copies)):    # chaque score revient à SA position
            c = comptes[origines[i], testee]
            apres = score >= seuil
            c[0] += 1
            c[1] += apres and not avant[i]
            c[2] += avant[i] and not apres
    return [Bascule(o, t, *c) for (o, t), c in comptes.items()]


def _scorer(modele, lot: list[dict]) -> list:
    """Un appel au modèle pour tout le lot (aucun pour un lot vide), et autant de scores que de clients."""
    if not lot:
        return []
    scores = list(modele(lot))
    if len(scores) != len(lot):
        raise ValueError(f"le modèle a renvoyé {len(scores)} scores pour {len(lot)} clients")
    return scores


# ============================================================================ Corriger


def _cible(cible) -> Fraction:
    """La cible telle qu'elle est ÉCRITE : 0.55 vaut exactement 11/20 (Fraction(0.55) vaudrait 0.5500000000000000444…)."""
    _nombre(cible, "cible")
    try:
        exacte = Fraction(str(cible)) if isinstance(cible, float) else Fraction(cible)
    except ValueError:  # Fraction("nan"), Fraction("inf")
        raise ValueError(f"cible illisible : {cible!r}") from None
    if not 0 < exacte <= 1:
        raise ValueError(f"cible hors de ]0, 1] : {cible!r}")
    return exacte


def seuils_egalite_chances(clients: list[dict], colonne: str, cible: float) -> dict:
    """Défi 235 — Pour chaque groupe, le plus haut seuil qui donne un taux de vrais positifs >= cible."""
    cible = _cible(cible)
    scores: dict = {}  # groupe → (scores des partis, scores des restés)
    for client in clients:
        score, parti = _score(client), _parti(client)
        scores.setdefault(_cle(client, colonne), ([], []))[0 if parti else 1].append(score)
    reglages = {}
    for groupe, (partis, restes) in scores.items():
        if not partis:
            reglages[groupe] = None  # personne n'est parti : le rappel n'existe pas
            continue
        p = len(partis)
        k = -(-cible.numerator * p // cible.denominator)  # le plus petit k tel que k / p >= cible : ceil, en entiers
        seuil = sorted(partis, reverse=True)[k - 1]       # le k-ième plus haut score d'un parti
        vp = sum(1 for s in partis if s >= seuil)         # les ex aequo du k-ième passent avec lui
        fp = sum(1 for s in restes if s >= seuil)
        reglages[groupe] = Reglage(seuil, vp / p, fp / len(restes) if restes else None, (vp + fp) / (p + len(restes)))
    return reglages


# ============================================================================ des tests, comme dans tests/


def test_taux_par_groupe_garde_les_absents_et_divise_par_le_groupe():
    clients = [{"g": "A", "score": 0.5, "parti": 1}, {"g": "A", "score": 0.2, "parti": 0},
               {"g": float("nan"), "score": 0.9, "parti": 0}, {"score": 0.1, "parti": 1}]
    resultat = taux_par_groupe(clients, "g")
    assert list(resultat) == ["A", None]
    assert resultat["A"] == Groupe(2, 1, 1, 1, 0, 0.5, 1.0, 0.0, 1.0)   # 0.5 >= 0.5 : sélectionné
    assert resultat[None] == Groupe(2, 1, 1, 0, 1, 0.5, 0.0, 1.0, 0.0)


def test_quatre_cinquiemes_a_la_limite_exacte():
    groupes = {"A": selection(25, 30), "B": selection(20, 30)}
    assert (20 / 30) / (25 / 30) < 0.8                     # le float se trompe…
    assert quatre_cinquiemes(groupes).signales == []       # …4/5 tout juste n'est pas « moins de 4/5 »


def test_taux_standardises_retrouve_simpson():
    def lot(g, s, n, sel):
        return [{"g": g, "s": s, "score": 0.9}] * sel + [{"g": g, "s": s, "score": 0.1}] * (n - sel)

    r = taux_standardises(lot("A", "x", 30, 27) + lot("A", "y", 10, 1) + lot("B", "x", 10, 10) + lot("B", "y", 30, 6),
                          "g", "s")
    assert r.brut == {"A": 0.7, "B": 0.4}                   # B est moins bien servi au total…
    assert r.standardise == {"A": 0.5, "B": 0.6}            # …mais mieux dans CHAQUE strate


def test_proxies_compte_les_cases_vides():
    clients = [{"a": 1, "x": "u"}, {"a": 1, "x": "u"}, {"a": 2, "x": "v"}, {"a": 2, "x": "v"}, {"a": 3, "x": "v"}]
    (proxy,) = proxies(clients, "a", ["x"])
    assert math.isclose(proxy.v, 1.0)                       # x dit tout de a (mais a ne dit pas tout de x)


def test_testing_ne_touche_pas_aux_clients():
    clients = [{"age": "65+", "contrat": "Mensuel", "appels": 2}, {"age": "30-44", "contrat": "Mensuel", "appels": 2}]
    avant = [dict(c) for c in clients]
    assert testing(modele, clients, "age", ["30-44", "65+"]) == [Bascule("30-44", "65+", 1, 0, 1),
                                                                 Bascule("65+", "30-44", 1, 1, 0)]
    assert clients == avant


def test_seuils_egalite_chances_compte_exactement():
    clients = [{"g": "A", "score": s / 1000, "parti": 1} for s in range(100, 200)]
    assert math.ceil(0.55 * 100) == 56                     # le float compte un parti de trop…
    assert seuils_egalite_chances(clients, "g", 0.55)["A"] == Reglage(0.145, 0.55, None, 0.55)

"""Série 31 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_31.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours une
option écrite deux fois et comptée deux fois, une moyenne calculée sur les seuls clients qui ont reçu une
recommandation, ou un `reverse=True` qui renverse aussi l'ordre alphabétique des ex aequo.

Fil rouge de la série : recommander, c'est compter, puis se méfier de ce qu'on a compté.

1. on compte des CLIENTS, pas des lignes : une option écrite deux fois par l'export compte une fois
   (`set(panier)`), partout ;
2. ce qui est populaire n'est pas pertinent : « mobile » est chez tout le monde, la confiance seule le
   recommanderait à tous. Le lift compare au hasard ; un support minimal écarte les coïncidences ;
3. on ne propose jamais ce que le client a déjà, et quand on ne sait rien, on a un plan B (les plus
   populaires) placé APRÈS ce qu'on sait ;
4. on évalue comme en production : on coupe dans le TEMPS, on ne devine que ce qui est NOUVEAU, et la
   moyenne porte sur TOUS les clients à deviner, y compris ceux pour qui on n'a rien proposé ;
5. trois fois la même idée n'en font pas trois : une bonne liste est variée (MMR), et on peut le calculer
   sans tout relire à chaque tour.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter

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

# ============================================================================ Défi 188 : cooccurrences


def cooccurrences(paniers: dict[str, list[str] | None]) -> dict[str, dict[str, int]]:
    """{a: {b: nombre de clients qui ont a ET b}}, symétrique, sans paire (a, a)."""
    cooc: dict[str, dict[str, int]] = {}
    for panier in paniers.values():
        if not panier:                         # None (case vide) ou [] : rien à compter
            continue
        offres = sorted(set(panier))           # set : « tv » écrite deux fois compte une fois
        for a in offres:
            cooc.setdefault(a, {})             # une option seule dans son panier est quand même une clé
        # Les paires DE CE PANIER seulement : k options → k(k-1)/2 paires. On ne parcourt jamais le
        # catalogue (des milliers d'options) ni les autres clients.
        for i, a in enumerate(offres):
            for b in offres[i + 1:]:           # b après a : chaque paire une fois, jamais (a, a)
                cooc[a][b] = cooc[a].get(b, 0) + 1
                cooc[b][a] = cooc[b].get(a, 0) + 1   # et dans les deux sens : cooc[b][a] == cooc[a][b]
    return cooc


# ============================================================================ Défi 189 : regles_association


def regles_association(paniers: dict[str, list[str] | None], support_min: int = 2) -> list[Regle]:
    """Les règles « si → alors » vues chez au moins `support_min` clients, les plus fortes d'abord."""
    if isinstance(support_min, bool) or not isinstance(support_min, int) or support_min < 1:
        raise ValueError(f"support_min doit être un entier ≥ 1, pas {support_min!r}")
    n = len(paniers)                           # TOUS les clients : ceux sans option auraient pu acheter
    detenteurs = Counter()
    for panier in paniers.values():
        detenteurs.update(set(panier or ()))
    regles = []
    for si, voisines in cooccurrences(paniers).items():
        for alors, support in voisines.items():
            if support < support_min:          # « au moins » : ≥, donc on écarte seulement <
                continue
            confiance = support / detenteurs[si]                       # P(alors | si) : on divise par `si`
            lift = support * n / (detenteurs[si] * detenteurs[alors])  # une seule division : a→b et b→a
            regles.append(Regle(si, alors, support, confiance, lift))  # ont EXACTEMENT le même lift
    # Pas de reverse=True : il renverserait aussi l'ordre alphabétique des ex aequo.
    regles.sort(key=lambda r: (-r.lift, -r.confiance, r.si, r.alors))
    return regles


# ============================================================================ Défi 190 : recommander


def recommander(panier: list[str] | None, voisins: dict[str, dict[str, float]], populaires: list[str],
                k: int = 3) -> list[str]:
    """Les `k` options à proposer : les plus proches de celles du client, puis les plus populaires."""
    if k < 0:
        raise ValueError(f"k doit être positif ou nul, pas {k}")
    detenues = set(panier or ())
    scores: dict[str, float] = {}
    for offre in sorted(detenues):             # sorted : le même ordre d'addition à chaque exécution
        for autre, similarite in voisins.get(offre, {}).items():   # .get : une option nouvelle n'a pas de ligne
            if autre not in detenues:          # jamais ce que le client a déjà
                scores[autre] = scores.get(autre, 0.0) + similarite
    proposees = sorted((o for o, s in scores.items() if s > 0), key=lambda o: (-scores[o], o))[:k]
    deja = set(proposees)
    for offre in populaires:                   # le plan B, APRÈS ce qu'on sait du client
        if len(proposees) >= k:
            break
        if offre not in detenues and offre not in deja:
            proposees.append(offre)
            deja.add(offre)
    return proposees


# ============================================================================ Défi 191 : decouper_dans_le_temps


def decouper_dans_le_temps(abonnements: list[tuple[str, str, dt.date]], coupure: dt.date
                           ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(historique avant la coupure, options NOUVELLES prises à partir de la coupure)."""
    historique: dict[str, set[str]] = {}
    apres: dict[str, set[str]] = {}
    for client, offre, date in abonnements:
        historique.setdefault(client, set())   # chaque client existe, même sans passé (client « froid »)
        if date < coupure:                     # strictement avant ; le jour de la coupure est « après »
            historique[client].add(offre)
        else:
            apres.setdefault(client, set()).add(offre)
    # On ne décide qu'à la FIN ce qui est nouveau : les lignes arrivent dans le désordre, et une
    # souscription d'avant la coupure peut être lue après celle d'après.
    a_deviner = {}
    for client, offres in apres.items():
        nouvelles = offres - historique[client]
        if nouvelles:                          # rien de nouveau : rien à deviner, le client n'est pas noté
            a_deviner[client] = nouvelles
    return historique, a_deviner


# ============================================================================ Défi 192 : mesurer


def mesurer(recommandations: dict[str, list[str]], a_deviner: dict[str, set[str]], k: int = 3) -> dict[str, float]:
    """Précision@k, rappel@k et NDCG@k, en moyenne sur TOUS les clients à deviner."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k doit être un entier ≥ 1, pas {k!r}")
    if not a_deviner:
        raise ValueError("aucun client à deviner : la moyenne n'existe pas")
    total_precision = total_rappel = total_ndcg = 0.0
    for client, cible in a_deviner.items():
        liste = recommandations.get(client) or []   # rien proposé : 0 partout, mais le client COMPTE
        vues: set[str] = set()
        touches, dcg = 0, 0.0
        for rang, offre in enumerate(liste[:k], start=1):   # seules les k premières sont montrées
            if offre in cible and offre not in vues:         # proposer deux fois « tv » ne marque qu'une fois
                touches += 1
                dcg += 1 / math.log2(rang + 1)               # rang 1 → 1/log2(2) = 1
            vues.add(offre)
        # L'idéal : toutes les bonnes réponses en tête, mais on ne peut pas en montrer plus de k.
        ideal = sum(1 / math.log2(rang + 1) for rang in range(1, min(k, len(cible)) + 1))
        total_precision += touches / k               # sur k : proposer moins n'est pas récompensé
        total_rappel += touches / len(cible)
        total_ndcg += dcg / ideal
    m = len(a_deviner)
    return {"precision": total_precision / m, "rappel": total_rappel / m, "ndcg": total_ndcg / m}


# ============================================================================ Défi 193 : diversifier


def diversifier(candidats: dict[str, float], similarites: dict[str, dict[str, float]], k: int = 3,
                lam: float = 0.7) -> list[str]:
    """Les `k` options choisies une à une : bonne note, mais pas trop proches de celles déjà choisies (MMR)."""
    if not 0 <= lam <= 1:                      # écrit ainsi, NaN est refusé aussi (toute comparaison est fausse)
        raise ValueError(f"lam doit être entre 0 et 1, pas {lam!r}")
    if k < 0:
        raise ValueError(f"k doit être positif ou nul, pas {k}")
    # Rangées une fois pour toutes par note décroissante puis nom : en gardant la PREMIÈRE meilleure valeur
    # rencontrée (« > » strict), les ex aequo sont départagés comme demandé, sans rien trier à chaque tour.
    restantes = sorted(candidats, key=lambda o: (-candidats[o], o))
    penalite = dict.fromkeys(restantes, 0.0)   # pour chaque candidate : sa similarité la PLUS FORTE avec le choix
    choix: list[str] = []
    while restantes and len(choix) < k:
        meilleure, valeur = None, -math.inf
        for offre in restantes:
            v = lam * candidats[offre] - (1 - lam) * penalite[offre]
            if v > valeur:
                meilleure, valeur = offre, v
        choix.append(meilleure)
        restantes.remove(meilleure)
        # Le seul changement de ce tour, c'est `meilleure` : on met à jour le maximum de chaque candidate
        # avec CETTE option seulement. Relire toutes les options déjà choisies coûterait k fois plus.
        for offre in restantes:
            ligne = similarites.get(offre)
            if ligne is not None:
                penalite[offre] = max(penalite[offre], ligne.get(meilleure, 0.0))
    return choix


# ============================================================================ Des tests, comme dans tests/


def test_une_option_ecrite_deux_fois_compte_une_fois():
    cooc = cooccurrences({"A": ["tv", "mobile", "tv"], "B": ["mobile", "tv"], "C": ["presse"], "D": None})
    assert cooc == {"mobile": {"tv": 2}, "tv": {"mobile": 2}, "presse": {}}


def test_le_lift_corrige_la_popularite():
    regles = {(r.si, r.alors): r for r in regles_association(paniers_exemple(), support_min=2)}
    assert regles["tv", "mobile"].confiance > regles["musique", "gaming"].confiance   # « mobile » est partout…
    assert regles["tv", "mobile"].lift < regles["musique", "gaming"].lift             # … le lift le voit


def test_on_ne_propose_jamais_ce_que_le_client_a_deja():
    voisins = {"tv": {"sport": 0.5, "box2": 0.75}, "sport": {"tv": 0.5}}
    assert recommander(["tv", "sport"], voisins, ["mobile", "tv"], k=2) == ["box2", "mobile"]


def test_le_jour_de_la_coupure_est_a_deviner_et_un_reabonnement_non():
    historique, a_deviner = decouper_dans_le_temps(abonnements_exemple(), dt.date(2026, 6, 1))
    assert a_deviner["C01"] == {"tv"} and historique["C03"] == set()


def test_un_client_sans_recommandation_compte_zero():
    assert mesurer({}, {"A": {"tv"}}, k=3) == {"precision": 0.0, "rappel": 0.0, "ndcg": 0.0}


def test_diversifier_sans_diversite_redonne_le_classement():
    candidats = {"tv": 0.9, "box2": 0.85, "sport": 0.8}
    assert diversifier(candidats, {"tv": {"box2": 0.9}}, k=3, lam=1.0) == ["tv", "box2", "sport"]

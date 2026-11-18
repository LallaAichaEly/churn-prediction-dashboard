"""Série 35 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_35.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours un
signe (départs évités = témoin − offre, et pas l'inverse), des ex aequo traités un par un dans la courbe de
Qini, ou un modèle jugé sur les clients qui ont servi à l'entraîner.

Fil rouge de la série : le score de churn dit QUI va partir ; il ne dit pas QUI on peut retenir. Pour le
savoir, il faut comparer, à profil égal, ceux qui ont reçu l'offre et ceux qui ne l'ont pas reçue.

1. sur un segment, l'effet de l'offre est une DIFFÉRENCE de deux taux : celui du témoin moins celui de
   l'offre. Sans témoin, pas d'effet mesurable (None, pas 0) ;
2. pour prédire cet effet client par client, on estime le risque AVEC et SANS offre, chacun dans sa cellule,
   en tirant les petites cellules vers le taux de leur groupe : c'est le T-learner ;
3. quand la chance d'être contacté n'est pas la même pour tous, la différence brute ment : on repondère
   chaque client par l'inverse de sa chance (l'issue transformée), dont la moyenne est l'effet ;
4. un classement se juge à sa courbe de Qini : les départs évités si l'on contacte les k premiers. Les
   ex aequo entrent ensemble, sinon la courbe dépend d'un ordre que personne n'a choisi ;
5. on n'appelle que les clients dont l'effet attendu rapporte plus que l'appel ne coûte : jamais les chiens
   qui dorment, même s'il reste du budget ;
6. un modèle se juge sur des clients qu'il n'a pas vus : sur les autres, il se flatte.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import math

from defis.ciblage import (  # noqa: F401
    COUT_CONTACT,
    VARIABLES,
    Effet,
    Plan,
    Qini,
    campagne_exemple,
    campagne_stratifiee_exemple,
    generer_campagne,
    prospects_exemple,
    risque_vrai,
    uplift_vrai,
)


def _manquant(x) -> bool:
    """None ou NaN (NaN est le seul nombre différent de lui-même)."""
    return x is None or x != x


def _zero_un(client: dict, cle: str, i: int) -> int:
    """La valeur 0 ou 1 de client[cle] (True et False acceptés) ; toute autre valeur est une erreur de saisie."""
    valeur = client[cle]
    if valeur is True or valeur is False or (isinstance(valeur, int) and valeur in (0, 1)):
        return int(valeur)
    raise ValueError(f"client n°{i} : {cle} vaut {valeur!r}, 0 ou 1 attendu")


# ============================================================================ Défi 212 : effet par segment


def effet_par_segment(clients: list[dict], variable: str) -> dict:
    """{valeur de `variable`: Effet(n_offre, n_temoin, taux_offre, taux_temoin, uplift)}, dans l'ordre d'apparition."""
    # Quatre compteurs par segment : [clients avec offre, départs avec offre, témoins, départs témoins].
    # Un dict garde l'ordre d'insertion : l'ordre d'apparition vient tout seul.
    compteurs: dict = {}
    for i, client in enumerate(clients):
        offre, parti = _zero_un(client, "offre", i), _zero_un(client, "parti", i)
        c = compteurs.setdefault(client[variable], [0, 0, 0, 0])
        if offre:
            c[0] += 1
            c[1] += parti
        else:
            c[2] += 1
            c[3] += parti
    effets = {}
    for valeur, (n_offre, d_offre, n_temoin, d_temoin) in compteurs.items():
        # Un groupe vide n'a pas de taux : None, et surtout pas 0.0 (« personne n'est parti » serait faux).
        taux_offre = d_offre / n_offre if n_offre else None
        taux_temoin = d_temoin / n_temoin if n_temoin else None
        # Départs évités par client contacté : ce qui se passe SANS l'offre, moins ce qui se passe AVEC.
        uplift = None if taux_offre is None or taux_temoin is None else taux_temoin - taux_offre
        effets[valeur] = Effet(n_offre, n_temoin, taux_offre, taux_temoin, uplift)
    return effets


# ============================================================================ Défi 213 : T-learner


def t_learner(historique: list[dict], nouveaux: list[dict], variables, m: float = 20) -> list[float]:
    """L'uplift prédit de chaque nouveau client : risque lissé sans offre − risque lissé avec offre, dans sa cellule."""
    if isinstance(m, bool) or not isinstance(m, (int, float)) or not m > 0 or m == math.inf:
        raise ValueError(f"m doit être un nombre > 0 (fini), pas {m!r}")
    variables = tuple(variables)
    # Une passe : pour chaque groupe (1 = offre, 0 = témoin), les clients et les départs par cellule.
    n = {0: {}, 1: {}}
    d = {0: {}, 1: {}}
    total_n, total_d = [0, 0], [0, 0]
    for i, client in enumerate(historique):
        g, parti = _zero_un(client, "offre", i), _zero_un(client, "parti", i)
        cellule = tuple(client[v] for v in variables)
        n[g][cellule] = n[g].get(cellule, 0) + 1
        d[g][cellule] = d[g].get(cellule, 0) + parti
        total_n[g] += 1
        total_d[g] += parti
    if total_n[0] == 0 or total_n[1] == 0:
        raise ValueError("il faut des clients AVEC et SANS offre dans l'historique pour estimer un effet")
    # Chaque groupe est tiré vers SON taux global : l'a priori des contactés n'est pas celui des témoins.
    a_priori = (total_d[0] / total_n[0], total_d[1] / total_n[1])

    def risque(g: int, cellule: tuple) -> float:
        # Cellule jamais vue dans ce groupe : n = d = 0, et la formule donne l'a priori. Pas de cas à part.
        return (d[g].get(cellule, 0) + m * a_priori[g]) / (n[g].get(cellule, 0) + m)

    predictions = []
    for client in nouveaux:            # on ne lit QUE les variables : les nouveaux n'ont ni offre ni départ
        cellule = tuple(client[v] for v in variables)
        predictions.append(risque(0, cellule) - risque(1, cellule))
    return predictions


# ============================================================================ Défi 214 : issue transformée


def issue_transformee(clients: list[dict]) -> list[float]:
    """Pour chaque client : parti × ((1 − offre) / (1 − p) − offre / p), avec p = sa probabilité d'avoir reçu l'offre."""
    issues = []
    for i, client in enumerate(clients):
        offre, parti = _zero_un(client, "offre", i), _zero_un(client, "parti", i)
        p = client["p_offre"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 < p < 1:   # refuse aussi NaN
            raise ValueError(f"client n°{i} : p_offre vaut {p!r}, il faut 0 < p < 1 (sinon, pas de comparaison)")
        # parti × ((1 − offre) / (1 − p) − offre / p), écrit cas par cas (et sans fabriquer un -0.0) :
        # un témoin parti compte +1 / (1 − p) : il « représente » 1 / (1 − p) clients comme lui ;
        # un appelé parti compte −1 / p ; un client resté compte 0 : seuls les départs portent l'information.
        if not parti:
            issues.append(0.0)
        elif offre:
            issues.append(-1 / p)
        else:
            issues.append(1 / (1 - p))
    return issues


# ============================================================================ Défi 215 : courbe de Qini


def courbe_qini(uplifts: list[float], clients: list[dict]) -> Qini:
    """Qini(points, qini) : les départs évités estimés si l'on contacte les k premiers selon `uplifts`."""
    n = len(clients)
    if len(uplifts) != n:
        raise ValueError(f"{len(uplifts)} prédictions pour {n} clients")
    if n == 0:
        raise ValueError("aucun client : pas de courbe")
    for i, u in enumerate(uplifts):
        if _manquant(u):
            raise ValueError(f"la prédiction n°{i} est manquante ({u!r}) : un NaN se trie n'importe où")
    groupes = [_zero_un(c, "offre", i) for i, c in enumerate(clients)]
    partis = [_zero_un(c, "parti", i) for i, c in enumerate(clients)]
    ordre = sorted(range(n), key=lambda i: -uplifts[i])          # O(n log n), le plus prometteur d'abord
    points = [(0, 0.0)]
    n_t = n_c = d_t = d_c = 0
    for position, i in enumerate(ordre):
        if groupes[i]:
            n_t += 1
            d_t += partis[i]
        else:
            n_c += 1
            d_c += partis[i]
        dernier_du_groupe = position == n - 1 or uplifts[ordre[position + 1]] != uplifts[i]
        if dernier_du_groupe:            # les ex aequo entrent ENSEMBLE : un seul point après le dernier
            # Les témoins du haut de la liste disent combien de contactés seraient partis sans l'offre.
            gain = d_c * n_t / n_c - d_t if n_c else 0.0
            points.append((position + 1, gain))
    courbe = sum((k2 - k1) * (g1 + g2) / 2 for (k1, g1), (k2, g2) in zip(points, points[1:]))
    hasard = n * points[-1][1] / 2      # contacter au hasard : la diagonale de (0, 0) à (n, gain total)
    return Qini(points, (courbe - hasard) / n)


# ============================================================================ Défi 216 : plan de contact


def plan_de_contact(clients: list[dict], uplifts: list[float], cout: int, budget: int) -> Plan:
    """Plan(clients, gain, cout) : les clients qui rapportent le plus (uplift × valeur − coût > 0), dans le budget."""
    for nom, x, minimum in (("cout", cout, 1), ("budget", budget, 0)):
        if isinstance(x, bool) or not isinstance(x, int) or x < minimum:
            raise ValueError(f"{nom} doit être un nombre entier d'euros >= {minimum}, pas {x!r}")
    if len(uplifts) != len(clients):
        raise ValueError(f"{len(uplifts)} prédictions pour {len(clients)} clients")
    candidats = []
    for i, (client, u) in enumerate(zip(clients, uplifts)):
        valeur = client["valeur"]
        if _manquant(u) or _manquant(valeur):
            raise ValueError(f"client n°{i} : uplift ou valeur manquant")
        gain = u * valeur - cout
        if gain > 0:                    # un chien qui dort (u < 0) ou un appel qui coûte plus qu'il ne rapporte : non
            candidats.append((-gain, client["customer_id"], gain))
    candidats.sort()                    # gain décroissant, puis customer_id croissant
    choisis = candidats[: budget // cout]
    return Plan([cid for _, cid, _ in choisis], sum(g for _, _, g in choisis), len(choisis) * cout)


# ============================================================================ Défi 217 : prédictions hors pli


def predictions_hors_pli(historique: list[dict], apprendre, k: int = 5) -> list[float]:
    """La prédiction de chaque client par un modèle entraîné SANS lui (le pli du client n°i est i % k)."""
    n = len(historique)
    if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= n:
        raise ValueError(f"k doit être un entier entre 2 et {n} (le nombre de clients), pas {k!r}")
    predictions: list = [None] * n
    for pli in range(k):
        entrainement = [c for i, c in enumerate(historique) if i % k != pli]
        indices = list(range(pli, n, k))
        # Des COPIES sans la réponse : `del client["parti"]` modifierait l'historique de l'appelant.
        a_predire = [{cle: v for cle, v in historique[i].items() if cle not in ("offre", "parti")} for i in indices]
        resultat = list(apprendre(entrainement, a_predire))
        if len(resultat) != len(indices):
            raise ValueError(f"le pli {pli} : {len(resultat)} prédictions pour {len(indices)} clients")
        for i, u in zip(indices, resultat):
            predictions[i] = u          # à SA place : la prédiction n°i est celle du client n°i
    return predictions


# ============================================================================ Tests (comme dans tests/)


def _client(cid, offre, parti, **autres):
    return {"customer_id": cid, "contrat": "mensuel", "offre": offre, "parti": parti, "p_offre": 0.5, **autres}


def test_effet_par_segment():
    clients = [_client("A", 1, 0), _client("B", 1, 1), _client("C", 0, 1), _client("D", 0, 1),
               _client("E", 1, 0, contrat="annuel")]
    effets = effet_par_segment(clients, "contrat")
    assert effets["mensuel"] == Effet(2, 2, 0.5, 1.0, 0.5)
    assert effets["annuel"] == Effet(1, 0, 0.0, None, None)          # pas de témoin : pas d'effet


def test_t_learner_tire_les_petites_cellules_vers_leur_groupe():
    historique = [_client("A", 1, 0), _client("B", 1, 0), _client("C", 0, 1), _client("D", 0, 0)]
    # a priori : 0 départ sur 2 avec offre, 1 sur 2 sans. Cellule inconnue : 0.5 − 0.0.
    assert t_learner(historique, [{"contrat": "biennal"}], ["contrat"], m=4) == [0.5]
    assert t_learner(historique, [], ["contrat"]) == []


def test_issue_transformee():
    clients = [_client("A", 0, 1, p_offre=0.8), _client("B", 1, 1, p_offre=0.8), _client("C", 1, 0, p_offre=0.8)]
    assert [round(x, 9) for x in issue_transformee(clients)] == [5.0, -1.25, 0.0]


def test_courbe_qini_et_ex_aequo():
    clients = [_client("A", 1, 0), _client("B", 0, 1), _client("C", 1, 1), _client("D", 0, 0)]
    q = courbe_qini([0.3, 0.3, 0.1, 0.1], clients)
    assert q.points == [(0, 0.0), (2, 1.0), (4, 0.0)]                # deux points, pas quatre
    assert math.isclose(q.qini, 0.5)


def test_plan_de_contact_n_appelle_pas_les_chiens_qui_dorment():
    clients = [{"customer_id": c, "valeur": 1000} for c in "ABC"]
    plan = plan_de_contact(clients, [0.10, -0.05, 0.02], cout=30, budget=1000)
    assert plan == Plan(["A"], 70.0, 30)


def test_predictions_hors_pli_ne_montre_jamais_la_reponse():
    vus = []

    def apprendre(entrainement, a_predire):
        vus.append(({c["customer_id"] for c in entrainement}, [sorted(c) for c in a_predire]))
        return [0.0] * len(a_predire)

    historique = [_client(c, i % 2, 0) for i, c in enumerate("ABCDE")]
    assert predictions_hors_pli(historique, apprendre, k=2) == [0.0] * 5
    assert vus[0][0] == {"B", "D"} and vus[1][0] == {"A", "C", "E"}
    assert all("parti" not in cles and "offre" not in cles for _, a_predire in vus for cles in a_predire)
    assert historique[0]["parti"] == 0                                # l'historique n'a pas été touché

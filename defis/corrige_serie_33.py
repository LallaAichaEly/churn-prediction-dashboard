"""Série 33 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_33.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours une
prévision qui a vu le mois qu'elle devait prévoir (`ajustees[t]` calculé APRÈS avoir lissé `historique[t]`,
ou la méthode appelée avec `serie[:t + 1]`), ou le niveau écrasé avant d'avoir calculé la tendance (Holt).

Fil rouge de la série : une prévision ne connaît QUE le passé. Tout le reste en découle.

1. la méthode la plus simple (le même mois l'an dernier) est la référence à battre : sans elle, « 30 départs
   d'erreur » ne veut rien dire ;
2. une seule erreur ne suffit pas : la MAE dit de combien on se trompe, la RMSE si on se trompe parfois
   BEAUCOUP, le biais dans quel SENS (et donc comment corriger), la MAPE en pourcentage, sauf si un mois vaut 0 ;
3. un lissage se calcule en UNE passe : la prévision du mois t s'écrit AVANT de regarder le mois t, puis le
   niveau (et la tendance) se mettent à jour ;
4. pour juger une méthode, on la rejoue dans le passé, origine après origine, en ne lui montrant que ce
   qu'on savait alors : c'est la seule évaluation honnête d'une prévision ;
5. une prévision seule ne suffit pas pour décider : on y ajoute la marge que les erreurs PASSÉES, dans le
   bon sens et au bon horizon, rendent nécessaire.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import math

from defis.previsions import (  # noqa: F401
    DOSSIERS_PAR_CONSEILLER,
    Erreurs,
    Essai,
    Lissage,
    Marge,
    departs_exemple,
    generer_serie,
    mois_exemple,
)


def _manquant(x) -> bool:
    """None ou NaN (NaN est le seul nombre différent de lui-même)."""
    return x is None or x != x


def _valeurs(serie, nom: str) -> list:
    """Une COPIE de la série, après avoir refusé tout trou : un NaN contaminerait en silence tous les calculs."""
    valeurs = list(serie)
    for i, x in enumerate(valeurs):
        if _manquant(x):
            raise ValueError(f"{nom}[{i}] est manquant ({x!r}) : comblez le trou avant de prévoir")
    return valeurs


# ============================================================================ Défi 200 : prévision naïve


def prevision_naive(historique: list[float], horizon: int, saison: int = 1) -> list[float]:
    """Les `horizon` mois suivants : chacun reprend le même mois du DERNIER cycle de `saison` mois."""
    if horizon < 0:
        raise ValueError(f"horizon = {horizon} : on ne prévoit pas un nombre négatif de mois")
    if saison < 1:
        raise ValueError(f"saison = {saison} : un cycle dure au moins 1 mois")
    valeurs = _valeurs(historique, "historique")
    if len(valeurs) < saison:
        raise ValueError(f"{len(valeurs)} mois connus : il en faut au moins un cycle complet ({saison})")
    cycle = valeurs[len(valeurs) - saison:]           # le dernier cycle, dans l'ordre du calendrier
    # Le mois h (0, 1, …) qui suit l'historique tombe à la place h % saison du cycle : au-delà d'un cycle, on
    # recommence. Avec saison = 1, le cycle est le dernier mois : « le mois prochain comme ce mois-ci ».
    return [float(cycle[h % saison]) for h in range(horizon)]


# ============================================================================ Défi 201 : erreurs


def erreurs(reels: list[float], prevus: list[float]) -> Erreurs:
    """Erreurs(mae, rmse, biais, mape) des prévisions `prevus` face à ce qui s'est passé, `reels`."""
    reels, prevus = _valeurs(reels, "reels"), _valeurs(prevus, "prevus")
    if len(reels) != len(prevus):                     # zip s'arrêterait au plus court sans rien dire
        raise ValueError(f"{len(reels)} réels pour {len(prevus)} prévisions")
    if not reels:
        raise ValueError("aucune prévision à juger : une moyenne de rien n'existe pas")
    n = len(reels)
    ecarts = [p - r for r, p in zip(reels, prevus)]  # > 0 : on avait prévu trop
    mae = sum(abs(e) for e in ecarts) / n
    rmse = math.sqrt(sum(e * e for e in ecarts) / n)  # la moyenne des carrés, PUIS la racine
    biais = sum(ecarts) / n                           # sans valeur absolue : les erreurs se compensent, exprès
    if any(r == 0 for r in reels):
        mape = None                                   # « 3 départs prévus, 0 réel » : 3 / 0, l'infini
    else:
        mape = sum(abs(e) / abs(r) for e, r in zip(ecarts, reels)) / n
    return Erreurs(mae, rmse, biais, mape)


# ============================================================================ Défi 202 : lissage exponentiel


def lissage_exponentiel(historique: list[float], alpha: float, horizon: int) -> Lissage:
    """Lissage exponentiel simple : un niveau qui suit les données, d'autant plus vite que `alpha` est grand."""
    if not 0 < alpha <= 1:                            # écrit ainsi, NaN est refusé lui aussi
        raise ValueError(f"alpha = {alpha!r} : il faut 0 < alpha <= 1")
    if horizon < 0:
        raise ValueError(f"horizon = {horizon} : on ne prévoit pas un nombre négatif de mois")
    valeurs = _valeurs(historique, "historique")
    if not valeurs:
        raise ValueError("aucun mois connu : rien à lisser")
    niveau = float(valeurs[0])                        # le seul point de départ qui ne connaît pas l'avenir
    ajustees: list[float | None] = [None]             # le mois 0 n'a pas de passé : pas de prévision
    for y in valeurs[1:]:
        ajustees.append(niveau)                       # 1) la prévision, AVANT de regarder y
        niveau = alpha * y + (1 - alpha) * niveau     # 2) puis on apprend de y
    return Lissage(ajustees, [niveau] * horizon, niveau, 0.0)


# ============================================================================ Défi 203 : Holt


def holt(historique: list[float], alpha: float, beta: float, horizon: int) -> Lissage:
    """La méthode de Holt : un niveau ET une tendance lissés ; la prévision prolonge la tendance."""
    for nom, valeur in (("alpha", alpha), ("beta", beta)):
        if not 0 < valeur <= 1:
            raise ValueError(f"{nom} = {valeur!r} : il faut 0 < {nom} <= 1")
    if horizon < 0:
        raise ValueError(f"horizon = {horizon} : on ne prévoit pas un nombre négatif de mois")
    valeurs = _valeurs(historique, "historique")
    if len(valeurs) < 2:
        raise ValueError(f"{len(valeurs)} mois connus : il en faut 2 pour voir une tendance")
    # Le départ utilise les mois 0 ET 1 : on ne peut donc prévoir honnêtement aucun des deux.
    niveau, tendance = float(valeurs[1]), float(valeurs[1] - valeurs[0])
    ajustees: list[float | None] = [None, None]
    for y in valeurs[2:]:
        prevu = niveau + tendance                     # 1) la prévision, avant de voir y
        ajustees.append(prevu)
        ancien = niveau                               # 2) GARDER l'ancien niveau : la tendance en a besoin
        niveau = alpha * y + (1 - alpha) * prevu
        tendance = beta * (niveau - ancien) + (1 - beta) * tendance
    previsions = [niveau + h * tendance for h in range(1, horizon + 1)]   # h mois plus tard : h tendances
    return Lissage(ajustees, previsions, niveau, tendance)


# ============================================================================ Défi 204 : validation glissante


def validation_glissante(serie: list[float], methode, debut: int, horizon: int = 1) -> list[Essai]:
    """Rejoue `methode` à chaque origine, avec le seul passé connu alors, et garde sa prévision à `horizon` mois."""
    if horizon < 1:
        raise ValueError(f"horizon = {horizon} : on prévoit au moins le mois suivant")
    if debut < 1:
        raise ValueError(f"debut = {debut} : une méthode a besoin d'au moins un mois de passé")
    valeurs = list(serie)
    derniere = len(valeurs) - horizon                 # la dernière origine dont la cible est connue
    if debut > derniere:
        raise ValueError(f"{len(valeurs)} mois : aucune origine entre {debut} et {derniere}, rien à tester")
    essais = []
    for origine in range(debut, derniere + 1):
        prevus = methode(valeurs[:origine], horizon)  # UN appel, et une copie qui s'arrête AVANT la cible
        cible = origine + horizon - 1
        essais.append(Essai(origine, cible, prevus[horizon - 1], valeurs[cible]))
    return essais


# ============================================================================ Défi 205 : marge de sécurité


def marge_de_securite(serie: list[float], methode, debut: int, horizon: int, niveau: int = 90) -> Marge:
    """La prévision à `horizon` mois, plus la marge qui aurait suffi dans `niveau` % des essais passés."""
    if isinstance(niveau, bool) or not isinstance(niveau, int) or not 1 <= niveau <= 99:
        raise ValueError(f"niveau = {niveau!r} : un pourcentage ENTIER entre 1 et 99 (90, pas 0.9)")
    essais = validation_glissante(serie, methode, debut, horizon)
    # Dans le sens qui coûte : réel − prévu > 0, ce sont des départs que personne n'avait prévus.
    ecarts = sorted(e.reel - e.prevu for e in essais)
    n = len(ecarts)
    # Le plus petit k tel que k / n >= niveau / 100 : un plafond, calculé en ENTIERS. En float,
    # 28 / 100 * 25 vaut 7.000000000000001, et son plafond 8 au lieu de 7.
    k = (niveau * n + 99) // 100
    marge = ecarts[k - 1]
    prevision = methode(list(serie), horizon)[horizon - 1]   # cette fois, TOUT le passé est connu
    return Marge(prevision, marge, prevision + marge)


# ============================================================================ Tests (comme dans tests/)


def test_prevision_naive():
    assert prevision_naive([10, 20, 30, 40, 50, 60], 6, saison=4) == [30.0, 40.0, 50.0, 60.0, 30.0, 40.0]
    assert prevision_naive([1, 2, 3], 2) == [3.0, 3.0]
    assert prevision_naive([1, 2, 3], 0) == []


def test_erreurs():
    e = erreurs([10, 20], [13, 16])
    assert (e.mae, e.biais) == (3.5, -0.5) and math.isclose(e.rmse, math.sqrt(12.5))
    assert erreurs([0, 5], [1, 5]).mape is None


def test_lissage_et_holt_sur_une_droite():
    droite = [100.0 + 5 * t for t in range(10)]
    assert holt(droite, 0.3, 0.2, 2).previsions == [150.0, 155.0]          # une droite se prolonge exactement
    assert lissage_exponentiel(droite, 1.0, 1).previsions == [145.0]        # alpha = 1 : la prévision naïve
    assert lissage_exponentiel([4.0], 0.5, 2) == Lissage([None], [4.0, 4.0], 4.0, 0.0)


def test_validation_glissante_ne_montre_que_le_passe():
    vus = []

    def methode(passe, horizon):
        vus.append(len(passe))
        return [passe[-1]] * horizon

    essais = validation_glissante([1.0, 2.0, 3.0, 4.0, 5.0], methode, 2, horizon=2)
    assert vus == [2, 3] and [(e.origine, e.cible) for e in essais] == [(2, 3), (3, 4)]
    assert [e.prevu for e in essais] == [2.0, 3.0] and [e.reel for e in essais] == [4.0, 5.0]


def test_marge_de_securite():
    def naive(passe, horizon):
        return prevision_naive(passe, horizon, saison=12)

    m = marge_de_securite(departs_exemple(), naive, 24, 3, 90)
    assert (m.prevision, m.marge, m.borne) == (452.0, 40.0, 492.0)
    assert math.ceil(m.borne / DOSSIERS_PAR_CONSEILLER) == 13               # un conseiller de plus que sans marge

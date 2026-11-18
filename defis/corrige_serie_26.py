"""Série 26 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_26.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours
un `replace=False`, un `/ 2` oublié, un `<` à la place d'un `<=`, ou un `ddof`.

Fil rouge de la série : un chiffre mesuré sur des clients n'est qu'UNE réalisation du hasard. Avant de
l'annoncer, on se demande « de combien aurait-il bougé avec d'autres clients ? ». Quatre habitudes :

1. on simule plutôt que de se souvenir d'une formule : rééchantillonner (bootstrap) ou mélanger les
   étiquettes (permutation) marche pour n'importe quelle statistique, et se vérifie sur de petits cas ;
2. on fixe le nombre de clients AVANT l'expérience (puissance), et on corrige quand on fait beaucoup de
   tests (Benjamini-Hochberg) : sinon le hasard finit toujours par « trouver » quelque chose ;
3. quand on compare deux modèles ou deux groupes sur les MÊMES clients, on garde les paires ensemble :
   c'est gratuit et souvent bien plus précis ;
4. le hasard d'une fonction vient d'un générateur à elle (`np.random.default_rng(graine)`) : même graine,
   même résultat, et aucun effet sur le reste du programme.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import math
import numbers
from statistics import NormalDist

import numpy as np

from defis.incertitude import (  # noqa: F401
    EFFET_REEL,
    attentes_support,
    auc,
    campagne_exemple,
    notes_deux_scripts,
    p_valeur_exacte,
    p_valeurs_segments,
    puissance_simulee,
    scores_deux_modeles,
)

# ============================================================================ Le bootstrap


def _tableau(valeurs, nom: str = "valeurs") -> np.ndarray:
    """Une liste, un tableau ou une Series → un tableau de float, lu par POSITION, sans NaN, non vide."""
    x = np.asarray(valeurs, dtype=float)  # une Series devient ses valeurs : plus d'étiquettes qui gênent
    if x.ndim != 1 or len(x) == 0:
        raise ValueError(f"{nom} : il faut au moins une valeur (une liste ou un tableau à une dimension)")
    if np.isnan(x).any():
        raise ValueError(f"{nom} : valeurs manquantes (NaN). Retirez-les d'abord, et dites combien.")
    return x


def _verifier_confiance(confiance) -> None:
    if not 0 < confiance < 1:  # `not` : un NaN (toutes ses comparaisons sont fausses) est refusé aussi
        raise ValueError(f"confiance = {confiance!r} : un nombre entre 0 et 1 (0.95, pas 95)")


def intervalle_bootstrap(valeurs, statistique=np.mean, n_tirages: int = 2000, confiance: float = 0.95,
                         graine: int = 0) -> tuple[float, float]:
    x = _tableau(valeurs)
    _verifier_confiance(confiance)
    rng = np.random.default_rng(graine)  # LE générateur de cette fonction : ni np.random.seed, ni état global
    n = len(x)
    stats = np.empty(n_tirages)
    for t in range(n_tirages):
        # n positions tirées AVEC remise : certains clients reviennent deux fois, d'autres pas du tout. Sans
        # remise (une permutation), chaque tirage contiendrait exactement les mêmes clients : même moyenne,
        # même médiane, un intervalle de largeur nulle.
        stats[t] = statistique(x[rng.integers(0, n, size=n)])
    # 95 % au milieu : on laisse 2,5 % de chaque côté. `np.quantile` attend des proportions (0.025), pas des
    # pourcentages (`np.percentile` attend 2.5).
    bas, haut = np.quantile(stats, [(1 - confiance) / 2, (1 + confiance) / 2])
    return float(bas), float(haut)
    # Version vectorisée, quand la statistique accepte `axis=` : x[rng.integers(0, n, (n_tirages, n))] donne
    # TOUS les tirages d'un coup (une ligne par tirage), puis np.median(..., axis=1). Plus rapide, mais
    # n_tirages × n valeurs en mémoire, et une `lambda` quelconque ne saurait pas quoi faire de `axis`.


# ============================================================================ Le test de permutation


def p_valeur_permutation(a, b, n_permutations: int = 10_000, graine: int = 0) -> float:
    a, b = _tableau(a, "a"), _tableau(b, "b")
    rng = np.random.default_rng(graine)
    tous = np.concatenate([a, b])  # un NOUVEAU tableau : le mélanger ne touche pas aux listes reçues
    k = len(a)
    observe = abs(b.mean() - a.mean())  # bilatéral : un écart dans un sens ou dans l'autre compte
    extremes = 0
    for _ in range(n_permutations):
        # Si les étiquettes « ancien » / « nouveau » ne changent rien (l'hypothèse « aucun effet »), n'importe
        # quel partage des MÊMES notes en un groupe de len(a) et un groupe de len(b) était aussi probable.
        melange = rng.permutation(tous)
        ecart = abs(melange[k:].mean() - melange[:k].mean())  # couper à len(a), pas au milieu
        # `>=` : un écart ÉGAL est aussi extrême (avec `>`, des notes toutes égales donneraient 1 / (n + 1) au lieu de 1)
        extremes += bool(ecart >= observe)
    # +1 en haut et en bas : le partage observé est l'un des partages possibles. Jamais p = 0 : avec 999
    # permutations, on ne peut pas affirmer mieux que 1 / 1000.
    return float((extremes + 1) / (n_permutations + 1))


# ============================================================================ La taille d'échantillon


def taille_echantillon(p_temoin: float, baisse: float, alpha: float = 0.05, puissance: float = 0.80) -> int:
    for nom, valeur in (("p_temoin", p_temoin), ("baisse", baisse), ("alpha", alpha), ("puissance", puissance)):
        if isinstance(valeur, bool) or not isinstance(valeur, numbers.Real) or not 0 < valeur < 1:
            raise ValueError(f"{nom} = {valeur!r} : une proportion strictement entre 0 et 1 "
                             f"(5 points de baisse s'écrivent 0.05)")
    if baisse >= p_temoin:
        raise ValueError(f"une baisse de {baisse} points sur un churn de {p_temoin} donnerait un churn négatif ou nul")
    p_offre = p_temoin - baisse
    p_moyen = (p_temoin + p_offre) / 2
    loi = NormalDist()
    # Bilatéral : l'offre pourrait aussi AUGMENTER le churn, on veut le voir. Donc 2,5 % de chaque côté.
    z_alpha = loi.inv_cdf(1 - alpha / 2)   # 1,96 pour 5 %
    z_beta = loi.inv_cdf(puissance)        # 0,84 pour 80 % (et pas 0,8 : c'est un quantile, pas une proportion)
    # Sous « aucun effet », les deux groupes partagent le même churn (p_moyen) ; sous « l'offre marche »,
    # chacun a le sien. D'où les deux racines.
    racine = (z_alpha * math.sqrt(2 * p_moyen * (1 - p_moyen))
              + z_beta * math.sqrt(p_temoin * (1 - p_temoin) + p_offre * (1 - p_offre)))
    # Arrondi VERS LE HAUT : 1 250,3 clients ne suffisent pas, il en faut 1 251. `math.ceil` renvoie un int.
    return math.ceil(racine ** 2 / baisse ** 2)
    # Vérification par simulation : puissance_simulee(taille_echantillon(0.30, 0.05), 0.30, 0.25) ≈ 0.80.


# ============================================================================ Benjamini-Hochberg


def benjamini_hochberg(p_valeurs, q: float = 0.05) -> list[bool]:
    _verifier_confiance(q)  # même contrainte : strictement entre 0 et 1
    p = np.asarray(p_valeurs, dtype=float)  # une Series → ses valeurs, par position ; la liste reçue reste intacte
    if p.ndim != 1:
        raise ValueError("p_valeurs : une liste ou un tableau à une dimension")
    if np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_valeurs : chaque p-valeur doit être un nombre entre 0 et 1")
    m = len(p)
    if m == 0:
        return []
    ordre = np.argsort(p, kind="stable")      # positions de la plus petite à la plus grande : O(m log m)
    seuils = q * np.arange(1, m + 1) / m      # k/m × q pour k = 1..m
    passe = p[ordre] <= seuils                # `<=` : une p-valeur pile sur son seuil est retenue
    if not passe.any():
        return [False] * m
    # Le PLUS GRAND k qui passe, même si des plus petits ont échoué : c'est une procédure « montante ».
    # S'arrêter au premier échec (procédure « descendante ») rejetterait moins, sans raison.
    k = int(np.flatnonzero(passe)[-1]) + 1
    decouvertes = np.zeros(m, dtype=bool)
    decouvertes[ordre[:k]] = True             # on remet chaque décision à la position de SA p-valeur
    return decouvertes.tolist()               # des bool Python, dans l'ordre reçu


# ============================================================================ Comparer deux modèles


def comparer_auc(y, proba_a, proba_b, n_tirages: int = 1000, confiance: float = 0.95, graine: int = 0) -> dict:
    y = np.asarray(y)
    a, b = _tableau(proba_a, "proba_a"), _tableau(proba_b, "proba_b")
    if not len(y) == len(a) == len(b):
        raise ValueError(f"longueurs différentes : y {len(y)}, proba_a {len(a)}, proba_b {len(b)}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y ne doit contenir que des 0 et des 1")
    if y.min() == y.max():
        raise ValueError("y ne contient qu'une seule classe : l'AUC n'existe pas")
    _verifier_confiance(confiance)
    ecart = auc(y, b) - auc(y, a)  # sur TOUT l'échantillon : c'est LA mesure ; le bootstrap dit seulement sa précision
    rng = np.random.default_rng(graine)
    n, ecarts, ignores = len(y), [], 0
    for _ in range(n_tirages):
        i = rng.integers(0, n, size=n)
        yi = y[i]
        if yi.min() == yi.max():  # que des 0 (ou que des 1) : pas d'AUC. On l'écarte, et on le DIT.
            ignores += 1
            continue
        # LES MÊMES clients pour les deux modèles (tirage APPARIÉ) : un client difficile pénalise A et B dans le
        # même tirage, et l'écart ne garde que ce qui les distingue vraiment. Tirer A et B séparément ajouterait
        # tout le bruit des clients, deux fois.
        ecarts.append(auc(yi, b[i]) - auc(yi, a[i]))
    if not ecarts:
        raise ValueError("aucun tirage ne contient les deux classes : trop peu de clients partis (ou restés)")
    bas, haut = np.quantile(ecarts, [(1 - confiance) / 2, (1 + confiance) / 2])
    return {"ecart": float(ecart), "bas": float(bas), "haut": float(haut), "tirages_ignores": ignores}


# ============================================================================ CUPED


def _estimer(v: np.ndarray, traite: np.ndarray) -> tuple[float, float]:
    """Écart des moyennes (traités - témoins) et son erreur type (variances séparées, ddof=1)."""
    t, c = v[traite], v[~traite]
    return float(t.mean() - c.mean()), math.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))


def effet_cuped(y, x_avant, traite) -> dict:
    y = _tableau(y, "y")
    x = np.array(x_avant, dtype=float)  # np.array COPIE : remplir ses NaN ne touchera pas le tableau reçu
    groupes = np.asarray(traite)
    if not len(y) == len(x) == len(groupes):
        raise ValueError(f"longueurs différentes : y {len(y)}, x_avant {len(x)}, traite {len(groupes)}")
    if not all(isinstance(v, (bool, np.bool_, numbers.Integral)) and v in (0, 1) for v in groupes.tolist()):
        raise ValueError("traite : True / False (ou 1 / 0) pour chaque client")
    groupes = groupes.astype(bool)
    if groupes.sum() < 2 or (~groupes).sum() < 2:
        raise ValueError("il faut au moins 2 clients dans chaque groupe pour estimer une variance")
    # Les clients sans score (nouveaux) : on leur donne la moyenne, donc une correction NULLE. Les retirer
    # casserait le tirage au sort (on ne compare plus les mêmes populations).
    connus = ~np.isnan(x)
    x[~connus] = x[connus].mean() if connus.any() else 0.0
    if x.max() == x.min():  # constant (ou rien de connu) : x n'apprend rien sur y
        theta = 0.0
    else:
        # θ sur TOUS les clients, traités et témoins ensemble, avec le MÊME ddof en haut et en bas.
        # (np.cov a ddof=1 par défaut, np.var ddof=0 : les mélanger multiplie θ par n / (n - 1).)
        theta = float(np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1))
    # Centré sur la moyenne de TOUS : le score moyen est le même dans les deux groupes (tirage au sort), donc la
    # correction ne déplace pas l'effet, elle retire seulement la part de y que le score expliquait déjà.
    # Centrer chaque groupe sur SA moyenne annulerait la correction : on retrouverait l'effet brut.
    y_ajuste = y - theta * (x - x.mean())
    effet, erreur = _estimer(y_ajuste, groupes)
    effet_brut, erreur_brute = _estimer(y, groupes)
    return {"effet": effet, "erreur_type": erreur, "effet_brut": effet_brut, "erreur_type_brute": erreur_brute,
            "theta": theta}


# ============================================================================ Tests (à la pytest)


def test_le_bootstrap_retrouve_l_erreur_type_de_la_moyenne():
    x = np.random.default_rng(1).normal(50, 10, 400)
    bas, haut = intervalle_bootstrap(x, n_tirages=3000)
    attendu = 2 * 1.96 * x.std() / math.sqrt(len(x))  # la formule classique, quand elle existe
    assert abs((haut - bas) / attendu - 1) < 0.1
    assert intervalle_bootstrap(x, graine=3) == intervalle_bootstrap(list(x), graine=3)


def test_la_p_valeur_de_permutation_est_proche_de_l_exacte():
    a, b = [12, 15, 9], [20, 18, 25, 17]
    assert abs(p_valeur_permutation(a, b, 20_000) - p_valeur_exacte(a, b)) < 0.01
    assert p_valeur_permutation(range(20), range(100, 120), 999) == 1 / 1000  # jamais 0
    assert p_valeur_permutation([5.0] * 8, [5.0] * 10, 99) == 1.0  # `>=` : rien n'est plus extrême qu'un écart nul


def test_la_taille_donne_bien_80_pourcent_de_puissance():
    n = taille_echantillon(0.30, 0.05)
    assert n == 1251
    assert abs(puissance_simulee(n, 0.30, 0.25, n_experiences=20_000) - 0.80) < 0.015
    assert taille_echantillon(0.30, 0.05, puissance=0.90) > n > taille_echantillon(0.30, 0.10)


def test_benjamini_hochberg_est_une_procedure_montante():
    assert benjamini_hochberg([0.20, 0.036, 0.01, 0.03]) == [False, True, True, True]
    assert benjamini_hochberg([0.05, 0.025, 0.0125, 0.04]) == [True] * 4  # pile sur le seuil : retenu
    segments = p_valeurs_segments()
    retenus = [s for s, r in zip(segments, benjamini_hochberg(list(segments.values()))) if r]
    assert retenus == ["mensuel_cheque", "fibre_mensuel", "jeunes_fibre"]


def test_la_comparaison_appariee_est_plus_precise():
    df = scores_deux_modeles()
    r = comparer_auc(df["churn"], df["champion"], df["challenger"], n_tirages=500)
    assert r["bas"] < r["ecart"] < r["haut"] and r["tirages_ignores"] == 0
    seul = intervalle_bootstrap(np.arange(len(df)), lambda i: auc(df["churn"].to_numpy()[i.astype(int)],
                                                                   df["champion"].to_numpy()[i.astype(int)]), 300)
    assert (r["haut"] - r["bas"]) < (seul[1] - seul[0]) / 2  # l'écart apparié est bien plus précis qu'une AUC seule


def test_cuped_garde_l_effet_et_reduit_l_erreur():
    df = campagne_exemple()
    r = effet_cuped(df["churn"], df["score_avant"], df["offre"])
    assert abs(r["effet_brut"] - (df[df.offre].churn.mean() - df[~df.offre].churn.mean())) < 1e-12
    assert r["erreur_type"] < 0.92 * r["erreur_type_brute"]
    assert abs(r["effet"] - EFFET_REEL) < 3 * r["erreur_type"]
    constant = effet_cuped(df["churn"], np.full(len(df), 0.5), df["offre"])
    assert constant["theta"] == 0 and constant["effet"] == constant["effet_brut"]

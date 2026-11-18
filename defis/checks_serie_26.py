"""Vérifications automatiques de la série 26 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur se comporte comme un·e statisticien·ne qui relit votre analyse avant qu'elle parte à la
direction :

- il ne peut pas comparer vos intervalles au centième près (votre hasard n'est pas le sien) : il vérifie
  ce que le hasard ne change PAS. La largeur d'un intervalle bootstrap de la moyenne doit retrouver la
  formule classique 2 × 1,96 × écart type / √n ; un test de permutation sur deux groupes complètement
  séparés doit donner exactement 1 / (n_permutations + 1) ; sur de petits groupes, votre p-valeur doit être
  proche de la p-valeur EXACTE (tous les partages possibles, essayés un par un) ;
- quand ça ne va pas, il rejoue les erreurs classiques (tirage SANS remise, générateur recréé dans la
  boucle, intervalle à 90 % au lieu de 95 %, test unilatéral, `>` au lieu de `>=`, Bonferroni au lieu de
  Benjamini-Hochberg, procédure qui s'arrête au premier échec, tirages séparés pour les deux modèles,
  `np.cov` et `np.var` qui n'ont pas le même ddof, chaque groupe centré sur sa propre moyenne…) pour vous
  dire LAQUELLE vous avez écrite ;
- au défi 160, il simule des milliers de tests A/B avec VOTRE nombre de clients, pour vous dire quelle
  puissance vous auriez vraiment eue ;
- il vérifie que votre fonction ne touche ni au hasard global (`np.random.seed`), ni aux données reçues
  (un `.sort()` ou des NaN remplis « en place » chez l'appelant), et qu'elle lit une Series par POSITION.
"""

from __future__ import annotations

import math
import numbers
import random
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from .incertitude import (
    attentes_support,
    auc,
    campagne_exemple,
    notes_deux_scripts,
    p_valeur_exacte,
    p_valeurs_segments,
    puissance_simulee,
    scores_deux_modeles,
)
from .verif import Defi, Serie

# ---------------------------------------------------------------------------- outils


def _decrire(exc: BaseException) -> str:
    texte = str(exc).split("\n")[0]
    if len(texte) > 160:
        texte = texte[:159] + "…"
    return f"{type(exc).__name__}({texte!r})" if texte else type(exc).__name__


def _court(valeur, n: int = 160) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


def _f(x: float, chiffres: int = 4) -> str:
    return f"{x:.{chiffres}f}"


class _BoucleSansFin(BaseException):
    """Levée dans VOTRE fonction quand elle exécute trop de lignes (BaseException : `except Exception` ne l'attrape pas)."""


def _trace(f, appel, lignes_max: int | None):
    """Le résultat de `appel()` ; arrête net au-delà de `lignes_max` lignes exécutées dans le fichier de f."""
    code = getattr(f, "__code__", None)
    if code is None or lignes_max is None:
        return appel()
    fichier, compte = code.co_filename, [0]

    def local(frame, evenement, arg):
        if evenement == "line":
            compte[0] += 1
            if compte[0] > lignes_max:
                raise _BoucleSansFin
        return local

    def global_(frame, evenement, arg):
        return local if frame.f_code.co_filename == fichier else None

    precedent = sys.gettrace()
    sys.settrace(global_)
    try:
        return appel()
    finally:
        sys.settrace(precedent)


_CONSEIL_HASARD = ("Créez le générateur de la fonction au début : `rng = np.random.default_rng(graine)`, puis "
                   "`rng.integers(...)` ou `rng.permutation(...)`. `np.random.seed`, `np.random.choice`, "
                   "`np.random.shuffle` ou `random.shuffle` changent le hasard de TOUT le programme (le code des "
                   "autres compris), et `np.random.default_rng()` sans graine donne un résultat différent à chaque "
                   "appel.")
_CONSEIL_LENT = ("Votre fonction a été arrêtée : elle exécute trop de lignes. Une boucle dans une boucle sur 20 000 "
                 "valeurs, c'est 400 millions de tours : triez UNE fois (`np.argsort` ou `sorted(range(m), "
                 "key=...)`), puis parcourez le résultat une seule fois.")


def _etat_hasard():
    etat = np.random.get_state()
    return etat[0], etat[1].copy(), etat[2], etat[3], etat[4], random.getstate()


def _meme_hasard(a, b) -> bool:
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def _appel(f, args: tuple, kwargs: dict | None, quoi: str, conseil: str = "", lignes_max: int | None = None):
    """f(*args, **kwargs), en transformant toute erreur en message lisible, et en surveillant le hasard global."""
    kwargs = kwargs or {}
    avant = _etat_hasard()
    try:
        resultat = _trace(f, lambda: f(*args, **kwargs), lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil or _CONSEIL_LENT}") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None
    if not _meme_hasard(avant, _etat_hasard()):
        raise AssertionError(f"{quoi} modifie le hasard GLOBAL (`np.random` ou `random`). {_CONSEIL_HASARD}")
    return resultat


def _doit_lever(f, args: tuple, kwargs: dict | None, quoi: str, conseil: str = "") -> None:
    """Exige que f(*args, **kwargs) lève ValueError."""
    kwargs = kwargs or {}
    try:
        resultat = f(*args, **kwargs)
    except NotImplementedError:
        raise
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de ValueError. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever ValueError, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _reel(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _intact(avant, apres) -> bool:
    if isinstance(avant, np.ndarray):
        return isinstance(apres, np.ndarray) and avant.shape == apres.shape and np.array_equal(
            avant, apres, equal_nan=avant.dtype.kind == "f")
    if isinstance(avant, pd.Series):
        return avant.equals(apres) and avant.index.equals(apres.index)
    return avant == apres


def _copie(x):
    return x.copy() if isinstance(x, (np.ndarray, pd.Series)) else list(x)


# ---------------------------------------------------------------------------- références


def _ref_intervalle(x, statistique, n_tirages: int, confiance: float, graine: int) -> tuple[float, float]:
    rng = np.random.default_rng(graine)
    n = len(x)
    stats = [statistique(x[rng.integers(0, n, size=n)]) for _ in range(n_tirages)]
    bas, haut = np.quantile(stats, [(1 - confiance) / 2, (1 + confiance) / 2])
    return float(bas), float(haut)


def _ref_permutation(a, b, n_permutations: int, graine: int, coupe: int | None = None,
                     bilateral: bool = True) -> float:
    rng = np.random.default_rng(graine)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    tous = np.concatenate([a, b])
    k = len(a) if coupe is None else coupe
    observe = b.mean() - a.mean()
    extremes = 0
    for _ in range(n_permutations):
        m = rng.permutation(tous)
        ecart = m[k:].mean() - m[:k].mean()
        extremes += bool(abs(ecart) >= abs(observe) if bilateral else ecart >= observe)
    return (extremes + 1) / (n_permutations + 1)


def _n_brut(p1: float, p2: float, z_alpha: float, z_beta: float) -> float:
    p_moyen = (p1 + p2) / 2
    racine = (z_alpha * math.sqrt(2 * p_moyen * (1 - p_moyen))
              + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
    return racine ** 2 / (p1 - p2) ** 2


def _ref_taille(p: float, baisse: float, alpha: float, puissance: float) -> int:
    z = NormalDist().inv_cdf
    return math.ceil(_n_brut(p, p - baisse, z(1 - alpha / 2), z(puissance)))


def _variantes_taille(p: float, baisse: float, alpha: float, puissance: float) -> list[tuple[int, str]]:
    """Les réponses des erreurs classiques, chacune avec son explication."""
    z = NormalDist().inv_cdf
    za, zb = z(1 - alpha / 2), z(puissance)
    exact = _n_brut(p, p - baisse, za, zb)
    p2 = p - baisse
    p_moyen = (p + p2) / 2
    return [
        (math.ceil(_n_brut(p, p2, z(1 - alpha), zb)),
         f"Vous avez pris z_alpha = NormalDist().inv_cdf(1 - alpha) = {z(1 - alpha):.2f} : c'est un test UNILATÉRAL. "
         f"Le test est bilatéral (l'offre pourrait aussi faire MONTER le churn, on veut le voir) : "
         f"inv_cdf(1 - alpha / 2) = {za:.2f}."),
        (math.ceil(_n_brut(p, p2, za, z(1 - puissance))),
         f"z_beta = NormalDist().inv_cdf(1 - puissance) vaut {z(1 - puissance):.2f} (NÉGATIF) : c'est "
         f"inv_cdf(puissance) = {zb:.2f}."),
        (math.ceil(_n_brut(p, p2, za, puissance)),
         f"Vous avez utilisé la puissance ({puissance}) comme un z : il faut son QUANTILE, "
         f"NormalDist().inv_cdf({puissance}) = {zb:.2f}."),
        (math.ceil(_n_brut(p, p2, alpha / 2, zb)) if alpha / 2 > 0 else -1,
         f"Vous avez utilisé alpha / 2 ({alpha / 2}) comme un z : il faut son quantile, "
         f"NormalDist().inv_cdf(1 - alpha / 2) = {za:.2f}."),
        (round(exact),
         f"Vous arrondissez au plus proche : {exact:.2f} clients ne suffisent pas, il en faut {math.ceil(exact)}. "
         f"Arrondissez VERS LE HAUT avec `math.ceil`."),
        (int(exact),
         f"`int(...)` coupe les décimales (c'est un arrondi vers le BAS) : {exact:.2f} clients ne suffisent pas. "
         f"Arrondissez vers le haut avec `math.ceil`."),
        (2 * math.ceil(exact),
         "C'est le nombre de clients PAR GROUPE qu'on demande (vous avez renvoyé le total des deux groupes)."),
        (math.ceil((za + zb) ** 2 * 2 * p_moyen * (1 - p_moyen) / baisse ** 2),
         "Vous avez utilisé la formule simplifiée (z_alpha + z_beta)² × 2 p_moyen (1 - p_moyen) / baisse² : elle "
         "suppose la même variance quand l'offre marche et quand elle ne marche pas. Prenez celle de l'énoncé : "
         "√(2 p_moyen (1 - p_moyen)) va avec z_alpha, √(p_temoin(1 - p_temoin) + p_offre(1 - p_offre)) avec z_beta."),
        (math.ceil((za + zb) ** 2 * (p * (1 - p) + p2 * (1 - p2)) / baisse ** 2),
         "Vous avez utilisé la formule simplifiée (z_alpha + z_beta)² × (p_t(1 - p_t) + p_o(1 - p_o)) / baisse² : "
         "sous « aucun effet », les deux groupes partagent le même churn, p_moyen. Prenez celle de l'énoncé : "
         "√(2 p_moyen (1 - p_moyen)) va avec z_alpha, √(p_temoin(1 - p_temoin) + p_offre(1 - p_offre)) avec z_beta."),
        (math.ceil(_n_brut(p, p + baisse, za, zb)) if p + baisse < 1 else -1,
         "p_offre = p_temoin - baisse : l'offre doit faire BAISSER le churn (vous avez écrit p_temoin + baisse)."),
        (math.ceil(_n_brut(p, p * (1 - baisse), za, zb)),
         f"La baisse est en POINTS : {p:.0%} - {baisse * 100:.0f} points = {p2:.0%}, pas {p:.0%} × (1 - {baisse}) = "
         f"{p * (1 - baisse):.1%} (une baisse RELATIVE)."),
    ]


def _ref_bh(p, q: float, mode: str = "juste") -> list[bool]:
    p = [float(v) for v in p]
    m = len(p)
    if mode == "bonferroni":
        return [v <= q / m for v in p]
    if mode == "brut":
        return [v <= q for v in p]
    ordre = sorted(range(m), key=lambda i: p[i])
    k = 0
    for rang, i in enumerate(ordre, start=1):
        seuil = q * rang / m
        passe = p[i] < seuil if mode == "strict" else p[i] <= seuil
        if passe:
            k = rang
        elif mode == "descendante":
            break
    decisions = [False] * m
    for i in ordre[:k]:
        decisions[i] = True
    if mode == "trie":
        return [decisions[i] for i in ordre]
    return decisions


_EXPLICATIONS_BH = {
    "descendante": ("vous vous arrêtez à la PREMIÈRE p-valeur qui dépasse son seuil. Il faut le PLUS GRAND k tel que "
                    "p(k) <= k / m × q, même si une p-valeur plus petite a échoué : toutes celles avant k sont "
                    "retenues (la procédure est « montante »)."),
    "bonferroni": ("c'est la correction de Bonferroni (p <= q / m) : elle protège, mais rejette presque tout. "
                   "Benjamini-Hochberg compare la k-ième plus petite p-valeur à k / m × q, un seuil qui monte avec k."),
    "brut": ("vous comparez chaque p-valeur à q sans correction : c'est exactement le piège des comparaisons "
             "multiples (sur 20 segments sans aucun effet, on en « trouve » un en moyenne)."),
    "strict": ("une p-valeur PILE sur son seuil (p(k) = k / m × q) doit être retenue : `<=`, pas `<`."),
    "trie": ("vos décisions sont dans l'ordre des p-valeurs TRIÉES. Il faut les remettre dans l'ordre REÇU : la "
             "i-ème décision correspond à la i-ème p-valeur (`decisions[ordre[:k]] = True`)."),
}


def _ref_comparer(y, a, b, n_tirages: int, confiance: float, graine: int, apparie: bool = True) -> dict:
    rng = np.random.default_rng(graine)
    n, ecarts, ignores = len(y), [], 0
    for _ in range(n_tirages):
        i = rng.integers(0, n, size=n)
        j = i if apparie else rng.integers(0, n, size=n)
        if y[i].min() == y[i].max() or y[j].min() == y[j].max():
            ignores += 1
            continue
        ecarts.append(auc(y[j], b[j]) - auc(y[i], a[i]))
    bas, haut = np.quantile(ecarts, [(1 - confiance) / 2, (1 + confiance) / 2])
    return {"ecart": auc(y, b) - auc(y, a), "bas": float(bas), "haut": float(haut), "tirages_ignores": ignores}


def _ref_cuped(y, x, t, mode: str = "juste") -> dict:
    y, x, t = np.asarray(y, dtype=float), np.array(x, dtype=float), np.asarray(t).astype(bool)
    if mode == "nan_retires":
        garde = ~np.isnan(x)
        y, x, t = y[garde], x[garde], t[garde]
    connus = ~np.isnan(x)
    if mode == "nan_zero":
        x[~connus] = 0.0
    else:
        x[~connus] = x[connus].mean() if connus.any() else 0.0
    base = (~t) if mode == "theta_temoins" else np.ones(len(x), dtype=bool)
    xb, yb = x[base], y[base]
    if xb.max() == xb.min():
        theta = 0.0
    elif mode == "theta_y":
        theta = float(np.cov(xb, yb, ddof=1)[0, 1] / yb.var(ddof=1))
    elif mode == "ddof":
        theta = float(np.cov(xb, yb)[0, 1] / xb.var())
    elif mode == "ddof_inverse":
        theta = float(np.cov(xb, yb, ddof=0)[0, 1] / xb.var(ddof=1))
    else:
        theta = float(np.cov(xb, yb, ddof=1)[0, 1] / xb.var(ddof=1))
    if mode == "centrage_groupe":
        centre = np.where(t, x[t].mean(), x[~t].mean())
    else:
        centre = x.mean()
    y_ajuste = y - theta * (x - centre)

    def estimer(v):
        vt, vc = v[t], v[~t]
        effet = vt.mean() - vc.mean()
        if mode == "signe":
            effet = -effet
        if mode == "se_ddof0":
            erreur = math.sqrt(vt.var() / len(vt) + vc.var() / len(vc))
        elif mode == "se_commune":
            erreur = math.sqrt(v.var(ddof=1) * (1 / len(vt) + 1 / len(vc)))
        else:
            erreur = math.sqrt(vt.var(ddof=1) / len(vt) + vc.var(ddof=1) / len(vc))
        return float(effet), erreur

    effet, erreur = estimer(y_ajuste)
    effet_brut, erreur_brute = estimer(y)
    return {"effet": effet, "erreur_type": erreur, "effet_brut": effet_brut, "erreur_type_brute": erreur_brute,
            "theta": theta}


_EXPLICATIONS_CUPED = {
    "signe": "l'effet est traités MOINS témoins (négatif quand l'offre fait baisser le churn), pas l'inverse.",
    "nan_retires": ("vous retirez les clients sans score. Ils ont été tirés au sort comme les autres : les retirer "
                    "change les groupes comparés (et l'effet brut). Gardez-les en leur donnant la MOYENNE des "
                    "scores connus : leur correction sera nulle."),
    "nan_zero": ("vous donnez un score de 0 aux clients sans score : c'est affirmer qu'ils ne partiront pas. "
                 "Donnez-leur la MOYENNE des scores connus : leur correction sera nulle."),
    "theta_y": "θ = cov(x, y) / var(x) : c'est la variance de x (le score) en bas, pas celle de y.",
    "ddof": ("`np.cov` divise par n - 1 (ddof=1) et `x.var()` par n (ddof=0) : votre θ est multiplié par "
             "n / (n - 1). Prenez le même ddof en haut et en bas : `np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)`."),
    "ddof_inverse": ("vous divisez la covariance par n (ddof=0) et la variance par n - 1 (ddof=1) : prenez le même "
                     "ddof en haut et en bas : `np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)`."),
    "theta_temoins": ("θ se calcule sur TOUS les clients, traités et témoins ensemble (le score a été calculé avant "
                      "le tirage au sort : il se comporte pareil dans les deux groupes), pas sur les témoins seuls."),
    "centrage_groupe": ("vous centrez chaque groupe sur SA moyenne de x : la correction s'annule exactement dans "
                        "l'écart des moyennes, et vous retrouvez l'effet brut. Centrez tout le monde sur la moyenne "
                        "de x de TOUS les clients."),
    "se_ddof0": "les variances de l'erreur type se calculent avec ddof=1 (`v.var(ddof=1)`), pas ddof=0.",
    "se_commune": ("l'erreur type utilise la variance de CHAQUE groupe : √(var_traités / n_traités + "
                   "var_témoins / n_témoins), pas une variance commune."),
}


# ---------------------------------------------------------------------------- Défi 158


def _paire(r, quoi: str) -> tuple[float, float]:
    if not isinstance(r, (tuple, list)) or len(r) != 2:
        raise AssertionError(f"{quoi} doit renvoyer un tuple `(bas, haut)`, pas {_court(r)}. Écrivez "
                             f"`bas, haut = np.quantile(...)` puis `return float(bas), float(haut)`.")
    if not all(_reel(v) for v in r) or not all(math.isfinite(v) for v in r):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : `bas` et `haut` doivent être deux nombres (float), sans "
                             f"NaN. `bas, haut = np.quantile(resultats, [...])` puis `return float(bas), float(haut)`.")
    bas, haut = float(r[0]), float(r[1])
    if bas > haut:
        raise AssertionError(f"{quoi} renvoie bas = {bas} > haut = {haut} : les quantiles sont dans le mauvais ordre "
                             f"((1 - confiance) / 2 d'abord, puis (1 + confiance) / 2).")
    return bas, haut


_NORMALES = np.random.default_rng(2026).normal(50, 10, 400).round(2)
_Z95 = NormalDist().inv_cdf(0.975)
_CONSEIL_QUANTILE = ("`np.quantile` attend des proportions entre 0 et 1 (0.025 et 0.975) ; c'est `np.percentile` qui "
                     "attend des pourcentages (2.5 et 97.5).")
_CONSEIL_TABLEAU = ("`statistique` doit recevoir un TABLEAU NumPy (une liste ne sait pas faire `x > 15`) : tirez "
                    "dans `x = np.asarray(valeurs, dtype=float)`, avec `x[positions]`.")


def verifier_intervalle_bootstrap(f) -> None:
    attentes = attentes_support()
    quoi = "intervalle_bootstrap(attentes_support(), np.median, n_tirages=500)"
    r = _paire(_appel(f, (attentes, np.median), {"n_tirages": 500}, quoi, _CONSEIL_QUANTILE), quoi)
    if r[0] == r[1]:
        raise AssertionError(
            f"{quoi} renvoie un intervalle de largeur nulle {r} : tous vos tirages donnent la même médiane. Trois causes "
            f"possibles : vous tirez SANS remise (une permutation contient toujours les mêmes clients, donc la même "
            f"médiane), vous créez `rng = np.random.default_rng(graine)` DANS la boucle (chaque tirage repart de la "
            f"même graine : le même tirage 500 fois), ou vous calculez la statistique une seule fois. Tirez des "
            f"positions AVEC remise : `rng.integers(0, n, size=n)`.")
    if _paire(_appel(f, (attentes, np.median), {"n_tirages": 500}, quoi), quoi) != r:
        raise AssertionError(f"Deux appels de {quoi} avec la même graine donnent deux intervalles différents. "
                             f"{_CONSEIL_HASARD}")
    autres = {_paire(_appel(f, (attentes, np.mean), {"n_tirages": 200, "graine": g},
                            f"intervalle_bootstrap(attentes_support(), n_tirages=200, graine={g})"), quoi)
              for g in range(1, 5)}
    if len(autres) == 1:
        raise AssertionError("Avec les graines 1, 2, 3 et 4, intervalle_bootstrap(attentes_support(), n_tirages=200) "
                             f"donne toujours {autres.pop()} : la graine n'est pas utilisée. {_CONSEIL_HASARD}")

    # la largeur, comparée à la formule classique de la moyenne
    x = _NORMALES.copy()
    quoi = "intervalle_bootstrap(x, n_tirages=2000) sur 400 valeurs d'écart type 10"
    bas, haut = _paire(_appel(f, (x,), {"n_tirages": 2000}, quoi), quoi)
    if not _intact(_NORMALES, x):
        raise AssertionError(f"{quoi} a modifié le tableau reçu (mélangé ou trié sur place ?). Ne touchez jamais aux "
                             f"données de l'appelant : tirez des POSITIONS, `x[rng.integers(0, n, size=n)]`.")
    theorie = 2 * _Z95 * x.std() / math.sqrt(len(x))
    moyenne = float(x.mean())
    rapport = (haut - bas) / theorie
    if abs((bas + haut) / 2 - moyenne) > 0.25 * theorie:
        raise AssertionError(f"{quoi} renvoie ({_f(bas)}, {_f(haut)}), qui n'est pas centré sur la moyenne "
                             f"({_f(moyenne)}). Chaque tirage doit contenir des valeurs de `valeurs`, et "
                             f"`statistique` doit être appliquée à chaque tirage.")
    if not 0.88 <= rapport <= 1.12:
        if 0.75 < rapport < 0.88:
            cause = ("C'est la largeur d'un intervalle à 90 % : les quantiles sont (1 - confiance) / 2 = 0.025 et "
                     "(1 + confiance) / 2 = 0.975, pas 0.05 et 0.95.")
        else:
            cause = ("Chaque tirage doit contenir AUTANT de valeurs que l'échantillon (n = 400, et pas n_tirages), "
                     "tirées avec remise, et l'intervalle va du quantile 2,5 % au quantile 97,5 % des n_tirages "
                     "résultats (pas moyenne ± une erreur type).")
        raise AssertionError(f"{quoi} renvoie ({_f(bas)}, {_f(haut)}), de largeur {_f(haut - bas)}. Pour une "
                             f"moyenne, la formule classique donne 2 × 1,96 × écart type / √n = {_f(theorie)} : votre "
                             f"intervalle est {rapport:.2f} fois trop {'large' if rapport > 1 else 'étroit'}. {cause}")
    quoi80 = "intervalle_bootstrap(x, n_tirages=2000, confiance=0.80)"
    bas80, haut80 = _paire(_appel(f, (x,), {"n_tirages": 2000, "confiance": 0.80}, quoi80), quoi80)
    rapport80 = (haut80 - bas80) / (haut - bas)
    if not 0.55 <= rapport80 <= 0.75:
        raise AssertionError(f"{quoi80} renvoie ({_f(bas80)}, {_f(haut80)}) : un intervalle à 80 % doit être environ "
                             f"0,65 fois aussi large que celui à 95 % ({_f(bas)}, {_f(haut)}), il l'est {rapport80:.2f} "
                             f"fois. Utilisez le paramètre `confiance` : quantiles (1 - confiance) / 2 et "
                             f"(1 + confiance) / 2.")

    # la statistique demandée, et un tableau pour elle
    quoi = "intervalle_bootstrap(attentes_support(), np.median, n_tirages=2000)"
    bas, haut = _paire(_appel(f, (attentes, np.median), {"n_tirages": 2000}, quoi), quoi)
    if not (5.0 <= bas <= 6.5 and 9.0 <= haut <= 10.5):
        moyenne = _ref_intervalle(attentes, np.mean, 2000, 0.95, 0)
        cause = (" C'est l'intervalle de la MOYENNE : appelez `statistique(tirage)`, pas `np.mean(tirage)`."
                 if abs(bas - moyenne[0]) < 1.5 and abs(haut - moyenne[1]) < 1.5 else "")
        raise AssertionError(f"{quoi} renvoie ({bas}, {haut}) : l'intervalle de la médiane devrait être proche de "
                             f"(6.0, 9.5) (à un demi-point près, selon le hasard des tirages).{cause}")
    quoi = "intervalle_bootstrap(liste, lambda x: np.mean(x > 15), n_tirages=300) (une LISTE de 41 attentes)"
    liste = attentes.tolist()
    bas, haut = _paire(_appel(f, (liste, lambda v: float(np.mean(v > 15))), {"n_tirages": 300}, quoi,
                              _CONSEIL_TABLEAU), quoi)
    if not (0.03 <= bas <= 0.16 and 0.28 <= haut <= 0.42):
        raise AssertionError(f"{quoi} renvoie ({bas}, {haut}) : 9 attentes sur 41 (22 %) dépassent 15 minutes, "
                             f"l'intervalle devrait être proche de (0.10, 0.34).")
    if liste != attentes.tolist():
        raise AssertionError(f"{quoi} a modifié la liste reçue. Ne touchez jamais aux données de l'appelant.")

    # le type reçu ne change rien
    quoi = "intervalle_bootstrap(serie, n_tirages=300, graine=7) avec une Series d'index 1000 à 1040"
    serie = pd.Series(attentes, index=range(1000, 1041))
    r_serie = _paire(_appel(f, (serie,), {"n_tirages": 300, "graine": 7}, quoi,
                            "Lisez une Series par POSITION : `x = np.asarray(valeurs, dtype=float)` d'abord "
                            "(`serie[positions]` cherche des ÉTIQUETTES 0, 1, 2… qui n'existent pas)."), quoi)
    r_tableau = _paire(_appel(f, (attentes,), {"n_tirages": 300, "graine": 7}, quoi), quoi)
    r_liste = _paire(_appel(f, (attentes.tolist(),), {"n_tirages": 300, "graine": 7}, quoi), quoi)
    if not (np.allclose(r_serie, r_tableau, rtol=1e-12) and np.allclose(r_liste, r_tableau, rtol=1e-12)):
        raise AssertionError(f"Avec la même graine, une Series donne {r_serie}, une liste {r_liste} et un tableau "
                             f"{r_tableau} : le résultat ne doit pas dépendre du type reçu. Convertissez d'abord : "
                             f"`x = np.asarray(valeurs, dtype=float)`.")

    _doit_lever(f, ([],), None, "intervalle_bootstrap([])", "Sans aucune valeur, il n'y a rien à rééchantillonner.")
    _doit_lever(f, ([7.0, float("nan"), 12.5],), None, "intervalle_bootstrap([7.0, nan, 12.5])",
                "Un NaN rendrait la moitié des tirages NaN, sans prévenir : refusez-le (`np.isnan(x).any()`).")
    _doit_lever(f, (attentes,), {"confiance": 95}, "intervalle_bootstrap(attentes, confiance=95)",
                "La confiance est une proportion : 0.95, pas 95. Vérifiez `0 < confiance < 1`.")
    _doit_lever(f, (attentes,), {"confiance": 1.0}, "intervalle_bootstrap(attentes, confiance=1.0)",
                "Vérifiez `0 < confiance < 1` (strictement).")


# ---------------------------------------------------------------------------- Défi 159


def _p(r, quoi: str) -> float:
    if not _reel(r) or not 0 < r <= 1:
        raise AssertionError(f"{quoi} doit renvoyer une p-valeur (un float entre 0 exclu et 1 inclus), pas "
                             f"{_court(r)}.")
    return float(r)


def verifier_p_valeur_permutation(f) -> None:
    ancien, nouveau = notes_deux_scripts()
    separes_a, separes_b = list(range(20)), list(range(100, 120))
    for a, b, sens in ((separes_a, separes_b, "a < b"), (separes_b, separes_a, "a > b")):
        quoi = f"p_valeur_permutation(a, b, n_permutations=999) avec deux groupes complètement séparés ({sens})"
        r = _appel(f, (a, b), {"n_permutations": 999}, quoi)
        if r == 0:
            raise AssertionError(f"{quoi} renvoie 0. Aucun mélange n'atteint l'écart observé… mais le partage observé "
                                 f"est l'un des partages possibles : la p-valeur vaut (compte + 1) / (n_permutations + "
                                 f"1) = 1 / 1000, jamais 0. Avec 999 mélanges, on ne peut pas affirmer mieux.")
        p = _p(r, quoi)
        if abs(p - 1 / 1000) < 1e-12:
            continue
        if abs(p - 1 / 999) < 1e-12:
            raise AssertionError(f"{quoi} renvoie 1/999 : divisez par n_permutations + 1 = 1000 (le partage observé "
                                 f"compte aussi en bas).")
        if p > 0.5 and sens == "a > b":
            raise AssertionError(f"{quoi} renvoie {p} alors que le premier groupe est bien plus HAUT : votre test ne "
                                 f"voit qu'un sens. Le test est bilatéral : comparez des écarts en VALEUR ABSOLUE, "
                                 f"`abs(moyenne(b') - moyenne(a')) >= abs(moyenne(b) - moyenne(a))`.")
        if p > 0.5:
            raise AssertionError(f"{quoi} renvoie {p} : presque tous vos mélanges atteignent l'écart observé. Mélangez "
                                 f"toutes les valeurs ENSEMBLE (`rng.permutation(np.concatenate([a, b]))`) avant de "
                                 f"couper (mélanger chaque groupe séparément ne change jamais sa moyenne), et "
                                 f"comparez des écarts en VALEUR ABSOLUE.")
        raise AssertionError(f"{quoi} renvoie {p} au lieu de 1/1000 = 0.001 : aucun des 999 mélanges n'atteint l'écart "
                             f"observé, donc p = (0 + 1) / (999 + 1).")

    quoi = "p_valeur_permutation([5.0] * 8, [5.0] * 10, n_permutations=99)"
    p = _p(_appel(f, ([5.0] * 8, [5.0] * 10), {"n_permutations": 99}, quoi), quoi)
    if abs(p - 1) > 1e-12:
        cause = (" Vous comptez les écarts STRICTEMENT plus grands (`>`) : un écart ÉGAL à l'écart observé est aussi "
                 "extrême, il faut `>=`." if abs(p - 1 / 100) < 1e-12 else "")
        raise AssertionError(f"{quoi} renvoie {p} au lieu de 1.0 : toutes les notes sont égales, chaque mélange donne "
                             f"un écart de 0, aussi grand que l'écart observé (0).{cause}")

    for a, b in (([12, 15, 9], [20, 18, 25, 17]), ([3, 9, 4], [6, 8, 10, 7, 12, 5, 9, 11])):
        quoi = f"p_valeur_permutation({a}, {b}, n_permutations=20000)"
        p = _p(_appel(f, (a, b), {"n_permutations": 20000}, quoi), quoi)
        exacte = p_valeur_exacte(a, b)
        if abs(p - exacte) > 0.012:
            partages = math.comb(len(a) + len(b), len(a))
            milieu = _ref_permutation(a, b, 20000, 0, coupe=(len(a) + len(b)) // 2)
            unilaterale = _ref_permutation(a, b, 20000, 0, bilateral=False)
            if abs(p - milieu) < 0.012 and abs(milieu - exacte) > 0.03:
                cause = (f" Vous coupez le mélange au MILIEU : le premier groupe doit garder sa taille, len(a) = "
                         f"{len(a)} (`melange[:len(a)]` et `melange[len(a):]`).")
            elif abs(p - unilaterale) < 0.012 and abs(unilaterale - exacte) > 0.03:
                cause = " Votre test ne regarde qu'un sens : comparez des écarts en VALEUR ABSOLUE (bilatéral)."
            else:
                cause = ""
            raise AssertionError(f"{quoi} renvoie {p}, alors que la p-valeur exacte vaut {exacte:.4f} (les {partages} "
                                 f"partages possibles, tous essayés : `p_valeur_exacte(a, b)`). Avec 20 000 mélanges, "
                                 f"on devrait en être à moins de 0,01.{cause}")

    quoi = "p_valeur_permutation(ancien, nouveau, n_permutations=2000) avec notes_deux_scripts()"
    p = _p(_appel(f, (ancien, nouveau), {"n_permutations": 2000}, quoi), quoi)
    if not 0.04 <= p <= 0.11:
        raise AssertionError(f"{quoi} renvoie {p} : avec ces notes, la p-valeur est d'environ 0,074 (un écart de "
                             f"1,19 point qu'un partage au hasard atteint environ 7 fois sur 100).")

    quoi = "p_valeur_permutation(ancien, nouveau, n_permutations=300, graine=7)"
    p1 = _appel(f, (ancien, nouveau), {"n_permutations": 300, "graine": 7}, quoi)
    if _appel(f, (ancien, nouveau), {"n_permutations": 300, "graine": 7}, quoi) != p1:
        raise AssertionError(f"Deux appels de {quoi} donnent deux p-valeurs différentes. {_CONSEIL_HASARD}")
    if len({_appel(f, (ancien, nouveau), {"n_permutations": 300, "graine": g}, quoi) for g in range(1, 6)}) == 1:
        raise AssertionError(f"Avec les graines 1 à 5, {quoi.replace('graine=7', 'graine=g')} donne toujours la même "
                             f"p-valeur : la graine n'est pas utilisée. {_CONSEIL_HASARD}")

    a, b = np.array(ancien, dtype=float), np.array(nouveau, dtype=float)
    a0, b0 = a.copy(), b.copy()
    quoi = "p_valeur_permutation(np.array(ancien), np.array(nouveau), n_permutations=300)"
    _appel(f, (a, b), {"n_permutations": 300}, quoi)
    if not (_intact(a0, a) and _intact(b0, b)):
        raise AssertionError(f"{quoi} a modifié les tableaux reçus (mélangés sur place ?). Mélangez un NOUVEAU tableau "
                             f"(`np.concatenate([a, b])` en crée un), jamais les données de l'appelant.")
    quoi = "p_valeur_permutation(Series(ancien, index=50..63), Series(nouveau, index=10..21), n_permutations=2000)"
    p = _p(_appel(f, (pd.Series(ancien, index=range(50, 64)), pd.Series(nouveau, index=range(10, 22))),
                  {"n_permutations": 2000}, quoi, "Lisez une Series par POSITION : `np.asarray(a, dtype=float)`."), quoi)
    if not 0.04 <= p <= 0.11:
        raise AssertionError(f"{quoi} renvoie {p} au lieu d'environ 0,074 : lisez une Series par POSITION "
                             f"(`np.asarray(a, dtype=float)`), son index ne compte pas.")

    _doit_lever(f, ([], [1.0, 2.0]), None, "p_valeur_permutation([], [1.0, 2.0])",
                "Un groupe vide n'a pas de moyenne.")
    _doit_lever(f, ([6.0, float("nan")], [7.0, 8.0]), None, "p_valeur_permutation([6.0, nan], [7.0, 8.0])",
                "Un NaN rend toutes les moyennes NaN et toutes les comparaisons fausses : refusez-le.")


# ---------------------------------------------------------------------------- Défi 160

_CAS_TAILLE = [(0.30, 0.05, 0.05, 0.80), (0.30, 0.10, 0.05, 0.80), (0.50, 0.10, 0.01, 0.90),
               (0.10, 0.02, 0.05, 0.80), (0.25, 0.05, 0.10, 0.80), (0.08, 0.03, 0.05, 0.95)]


def verifier_taille_echantillon(f) -> None:
    for p, baisse, alpha, puissance in _CAS_TAILLE:
        quoi = f"taille_echantillon({p}, {baisse}, alpha={alpha}, puissance={puissance})"
        r = _appel(f, (p, baisse), {"alpha": alpha, "puissance": puissance}, quoi)
        attendu = _ref_taille(p, baisse, alpha, puissance)
        if _reel(r) and not isinstance(r, numbers.Integral):
            precis = f" ({r} au lieu de {attendu})" if r != attendu else ""
            raise AssertionError(f"{quoi} renvoie un {type(r).__name__}{precis} : un nombre de clients est un ENTIER. "
                                 f"`math.ceil(...)` renvoie directement un int (`np.ceil` renvoie un float).")
        if not isinstance(r, numbers.Integral) or isinstance(r, (bool, np.bool_)):
            raise AssertionError(f"{quoi} doit renvoyer un int, pas {_court(r)}.")
        if r == attendu:
            continue
        explication = next((texte for valeur, texte in _variantes_taille(p, baisse, alpha, puissance)
                            if valeur == r and valeur != attendu), "")
        puissance_reelle = (f" Avec votre n = {r} par groupe, sur 4 000 tests A/B simulés, l'offre n'aurait été "
                            f"détectée que dans {puissance_simulee(int(r), p, p - baisse, alpha):.0%} des cas "
                            f"(il en faut {puissance:.0%})." if 1 <= r < attendu else "")
        if r > attendu and r <= 10_000_000:
            puissance_reelle = (f" Avec votre n = {r} par groupe, vous recruteriez {r - attendu} clients de trop "
                                f"par groupe.")
        raise AssertionError(f"{quoi} renvoie {r} au lieu de {attendu}.{puissance_reelle} {explication}".rstrip())

    conseil_points = "La baisse est une proportion : 5 points s'écrivent 0.05."
    _doit_lever(f, (0.30, 5), None, "taille_echantillon(0.30, 5)", conseil_points)
    _doit_lever(f, (30, 0.05), None, "taille_echantillon(30, 0.05)", "Le churn est une proportion : 30 % = 0.30.")
    _doit_lever(f, (0.30, 0.30), None, "taille_echantillon(0.30, 0.30)",
                "Une baisse de 30 points sur un churn de 30 % donnerait un churn nul : refusez-la.")
    _doit_lever(f, (0.30, 0.40), None, "taille_echantillon(0.30, 0.40)",
                "Une baisse plus grande que le churn donnerait un churn négatif : refusez-la.")
    _doit_lever(f, (0.30, 0), None, "taille_echantillon(0.30, 0)",
                "Une baisse nulle ne se détecte avec aucun nombre de clients (division par zéro) : refusez-la.")
    _doit_lever(f, (0.30, -0.05), None, "taille_echantillon(0.30, -0.05)",
                "La baisse est un nombre strictement positif.")
    _doit_lever(f, (0.30, 0.05), {"alpha": 0}, "taille_echantillon(0.30, 0.05, alpha=0)",
                "alpha est strictement entre 0 et 1.")
    _doit_lever(f, (0.30, 0.05), {"puissance": 80}, "taille_echantillon(0.30, 0.05, puissance=80)",
                "La puissance est une proportion : 80 % = 0.80.")
    _doit_lever(f, (float("nan"), 0.05), None, "taille_echantillon(nan, 0.05)",
                "Un NaN passe à travers `p < 0 or p > 1` (toutes ses comparaisons sont fausses) : écrivez "
                "`not 0 < p < 1`.")


# ---------------------------------------------------------------------------- Défi 161


def _decisions(r, m: int, quoi: str) -> list[bool]:
    if isinstance(r, np.ndarray):
        raise AssertionError(f"{quoi} renvoie un tableau NumPy : renvoyez une LISTE de booléens (`decisions.tolist()`, "
                             f"qui donne aussi des bool Python).")
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} doit renvoyer une liste de booléens, pas {_court(r)}.")
    if len(r) != m:
        raise AssertionError(f"{quoi} renvoie {len(r)} décisions pour {m} p-valeurs : une décision par p-valeur, "
                             f"dans l'ordre reçu.")
    if not all(isinstance(v, (bool, np.bool_)) for v in r):
        raise AssertionError(f"{quoi} renvoie {_court(r)} : chaque décision est True ou False.")
    return [bool(v) for v in r]


def _verifier_bh(f, p, q: float | None, quoi: str, lignes_max: int | None = 500_000) -> None:
    kwargs = {} if q is None else {"q": q}
    q = 0.05 if q is None else q
    r = _decisions(_appel(f, (p,), kwargs, quoi, lignes_max=lignes_max), len(p), quoi)
    attendu = _ref_bh(p, q)
    if r == attendu:
        return
    cause = next((texte for mode, texte in _EXPLICATIONS_BH.items() if _ref_bh(p, q, mode) == r), "")
    if not cause and q != 0.05 and r == _ref_bh(p, 0.05):
        cause = f"vous utilisez q = 0.05 au lieu du paramètre `q` reçu ({q})."
    montrer = _court(list(p) if not isinstance(p, pd.Series) else p.tolist())
    raise AssertionError(f"{quoi} avec p = {montrer} renvoie {_court(r)} au lieu de {_court(attendu)}"
                         + (f" : {cause}" if cause else "."))


def verifier_benjamini_hochberg(f) -> None:
    segments = list(p_valeurs_segments().values())
    _verifier_bh(f, segments, None, "benjamini_hochberg(list(p_valeurs_segments().values()))")
    for p in ([0.20, 0.036, 0.01, 0.03], [0.05, 0.025, 0.0125, 0.04], [0.01, 0.5, 0.01], [0.03], [0.06],
              [0.9, 0.0001, 0.3, 0.02, 0.6, 0.011, 0.04]):
        _verifier_bh(f, p, None, f"benjamini_hochberg({p})")
    _verifier_bh(f, [0.04, 0.07, 0.6, 0.02], 0.10, "benjamini_hochberg([0.04, 0.07, 0.6, 0.02], q=0.10)")
    r = _appel(f, ([],), None, "benjamini_hochberg([])")
    if r != []:
        raise AssertionError(f"benjamini_hochberg([]) renvoie {_court(r)} au lieu de [] : aucune p-valeur, aucune "
                             f"décision.")

    alea = random.Random(26)
    for essai in range(300):
        m = alea.randint(1, 40)
        p = [round(alea.uniform(0, 0.02) if alea.random() < 0.3 else alea.random(), 3) for _ in range(m)]
        q = alea.choice([None, 0.05, 0.10, 0.20])
        _verifier_bh(f, p, q, f"benjamini_hochberg(p{'' if q is None else f', q={q}'}) (essai n° {essai + 1})")

    liste = [0.20, 0.036, 0.01, 0.03]
    avant = list(liste)
    _appel(f, (liste,), None, "benjamini_hochberg(liste)")
    if liste != avant:
        raise AssertionError(f"benjamini_hochberg(liste) a modifié la liste reçue : {avant} est devenue {liste} "
                             f"(un `.sort()` ?). Triez une COPIE (`sorted(...)`, `np.argsort`) : l'appelant a "
                             f"encore besoin de ses p-valeurs dans son ordre.")
    for nom, p in (("un tuple", (0.20, 0.036, 0.01, 0.03)), ("un tableau NumPy", np.array([0.20, 0.036, 0.01, 0.03])),
                   ("une Series d'index [100, 3, 57, 8]", pd.Series([0.20, 0.036, 0.01, 0.03], index=[100, 3, 57, 8]))):
        _verifier_bh(f, p, None, f"benjamini_hochberg({nom})")

    for p, raison in (([0.1, float("nan")], "un NaN"), ([0.1, 1.2], "1.2"), ([-0.01, 0.2], "-0.01")):
        _doit_lever(f, (p,), None, f"benjamini_hochberg({p})",
                    f"Une p-valeur est une probabilité entre 0 et 1 : {raison} signale une erreur en amont.")
    _doit_lever(f, ([0.01, 0.2],), {"q": 5}, "benjamini_hochberg([0.01, 0.2], q=5)",
                "q est une proportion strictement entre 0 et 1 (0.05, pas 5).")
    _doit_lever(f, ([0.01, 0.2],), {"q": 0}, "benjamini_hochberg([0.01, 0.2], q=0)",
                "q est une proportion strictement entre 0 et 1.")

    gros = np.random.default_rng(161).uniform(0, 1, 20_000)
    gros[:300] = np.random.default_rng(162).uniform(0, 1e-5, 300)
    _verifier_bh(f, gros.tolist(), None, "benjamini_hochberg(20 000 p-valeurs)", lignes_max=1_000_000)


# ---------------------------------------------------------------------------- Défi 162

_CLES_AUC = {"ecart", "bas", "haut", "tirages_ignores"}
_CONSEIL_UNE_CLASSE = ("Un tirage peut ne contenir AUCUN client parti (ou aucun resté) : son AUC n'existe pas. Testez "
                       "`y[i].min() == y[i].max()` AVANT de la calculer : écartez ce tirage (`continue`) et comptez-le "
                       "dans `tirages_ignores`.")


def _resultat_auc(r, quoi: str) -> dict:
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} doit renvoyer un dictionnaire {{'ecart', 'bas', 'haut', 'tirages_ignores'}}, pas "
                             f"{_court(r)}.")
    if set(r) != _CLES_AUC:
        manquent, en_trop = _CLES_AUC - set(r), set(r) - _CLES_AUC
        raise AssertionError(f"{quoi} renvoie les clés {sorted(r)} : il faut exactement {sorted(_CLES_AUC)}"
                             + (f" (manquent : {sorted(manquent)})" if manquent else "")
                             + (f" (en trop : {sorted(en_trop)})" if en_trop else "") + ".")
    for cle in ("ecart", "bas", "haut"):
        if _reel(r[cle]) and math.isnan(r[cle]):
            raise AssertionError(f"{quoi} : r[{cle!r}] vaut NaN. {_CONSEIL_UNE_CLASSE} (Les versions récentes de "
                                 f"`roc_auc_score` ne lèvent plus d'erreur dans ce cas : elles renvoient NaN, avec un "
                                 f"simple avertissement, et un seul NaN rend tous les quantiles NaN.)")
        if not _reel(r[cle]) or not math.isfinite(r[cle]):
            raise AssertionError(f"{quoi} : r[{cle!r}] vaut {_court(r[cle])}, il faut un nombre (float) sans NaN.")
    if not isinstance(r["tirages_ignores"], numbers.Integral) or isinstance(r["tirages_ignores"], (bool, np.bool_)):
        raise AssertionError(f"{quoi} : r['tirages_ignores'] vaut {_court(r['tirages_ignores'])}, il faut un entier "
                             f"(le NOMBRE de tirages écartés).")
    if r["bas"] > r["haut"]:
        raise AssertionError(f"{quoi} renvoie bas = {r['bas']} > haut = {r['haut']} : quantiles dans le mauvais ordre.")
    return r


def verifier_comparer_auc(f) -> None:
    df = scores_deux_modeles()
    y, a, b = (df[c].to_numpy().copy() for c in ("churn", "champion", "challenger"))  # modifiables (pandas 3)
    copies = (y.copy(), a.copy(), b.copy())
    # Un seul gros appel (1 500 tirages) sert à presque tout : avec `roc_auc_score`, chaque AUC coûte environ 2 ms.
    quoi = "comparer_auc(y, champion, challenger, n_tirages=1500) sur scores_deux_modeles()"
    r = _resultat_auc(_appel(f, (y, a, b), {"n_tirages": 1500}, quoi, _CONSEIL_QUANTILE), quoi)
    if not all(_intact(c, v) for c, v in zip(copies, (y, a, b))):
        raise AssertionError(f"{quoi} a modifié un des tableaux reçus. Tirez des POSITIONS (`y[i]`, `a[i]`) : cela crée "
                             f"de nouveaux tableaux.")
    ecart = auc(y, b) - auc(y, a)
    if abs(r["ecart"] - ecart) > 1e-9:
        if abs(r["ecart"] + ecart) < 1e-9:
            cause = "C'est AUC(b) - AUC(a) : positif quand b (le challenger) fait mieux."
        elif abs(r["ecart"] - ecart) < 0.01:
            cause = ("L'écart se mesure sur TOUS les clients, une seule fois : c'est LA mesure. Le bootstrap sert "
                     "seulement à dire sa précision (ce n'est pas la moyenne des tirages).")
        else:
            cause = "L'écart est AUC(y, proba_b) - AUC(y, proba_a), sur tous les clients."
        raise AssertionError(f"{quoi} : r['ecart'] vaut {r['ecart']:.6f} au lieu de {ecart:.6f}. {cause}")
    if r["tirages_ignores"] != 0:
        raise AssertionError(f"{quoi} : r['tirages_ignores'] vaut {r['tirages_ignores']}, alors qu'un tirage de 600 "
                             f"clients contient toujours des partis et des restés. N'écartez que les tirages d'une "
                             f"seule classe.")

    ref = _ref_comparer(y, a, b, 3000, 0.95, 99)
    largeur = ref["haut"] - ref["bas"]
    sa_largeur = r["haut"] - r["bas"]
    if sa_largeur < 0.05 * largeur:
        raise AssertionError(f"{quoi} renvoie un intervalle de largeur presque nulle ({r['bas']:.5f}, {r['haut']:.5f}) : "
                             f"tous vos tirages se ressemblent. Tirez des positions AVEC remise "
                             f"(`rng.integers(0, n, size=n)`), un NOUVEAU tirage à chaque tour, avec un `rng` créé "
                             f"une seule fois, hors de la boucle.")
    if sa_largeur > 1.8 * largeur:
        raise AssertionError(f"{quoi} renvoie ({r['bas']:.4f}, {r['haut']:.4f}), un intervalle {sa_largeur / largeur:.1f} "
                             f"fois trop large (attendu environ ({ref['bas']:.4f}, {ref['haut']:.4f})). Vous tirez "
                             f"sans doute des clients DIFFÉRENTS pour les deux modèles : tirez UNE fois les positions "
                             f"`i`, puis calculez AUC(y[i], b[i]) - AUC(y[i], a[i]). Les deux modèles se trompent "
                             f"souvent sur les mêmes clients ; garder les paires retire ce bruit commun.")
    # Avec 1 500 tirages, la largeur d'une version juste varie de ±6 % d'une écriture à l'autre (5 écarts types
    # avant 0,88) ; à 90 %, elle perd environ 18 %.
    if sa_largeur < 0.88 * largeur or abs(r["bas"] - ref["bas"]) > 0.2 * largeur \
            or abs(r["haut"] - ref["haut"]) > 0.2 * largeur:
        cause = (" Il est trop étroit : c'est la largeur d'un intervalle à 90 %. Les quantiles sont (1 - confiance) / 2 "
                 "= 0.025 et (1 + confiance) / 2 = 0.975." if 0.7 < sa_largeur / largeur < 0.88 else "")
        raise AssertionError(f"{quoi} renvoie ({r['bas']:.4f}, {r['haut']:.4f}) au lieu d'environ ({ref['bas']:.4f}, "
                             f"{ref['haut']:.4f}) (à {0.2 * largeur:.4f} près, selon le hasard des tirages).{cause}")
    quoi80 = "comparer_auc(y, champion, challenger, n_tirages=300, confiance=0.80)"
    r80 = _resultat_auc(_appel(f, (y, a, b), {"n_tirages": 300, "confiance": 0.80}, quoi80), quoi80)
    if not 0.5 <= (r80["haut"] - r80["bas"]) / sa_largeur <= 0.8:
        raise AssertionError(f"{quoi80} renvoie ({r80['bas']:.4f}, {r80['haut']:.4f}), presque aussi large qu'à 95 % "
                             f"({r['bas']:.4f}, {r['haut']:.4f}) : utilisez le paramètre `confiance`.")

    # un seul client parti sur 15 : un tirage sur trois n'en contient aucun
    y_petit = np.array([1] + [0] * 14)
    a_petit = np.linspace(0.1, 0.8, 15)
    b_petit = np.r_[0.9, np.linspace(0.1, 0.8, 14)]
    quoi = "comparer_auc(y, a, b, n_tirages=300) avec 1 client parti sur 15"
    r = _resultat_auc(_appel(f, (y_petit, a_petit, b_petit), {"n_tirages": 300}, quoi, _CONSEIL_UNE_CLASSE), quoi)
    if not 60 <= r["tirages_ignores"] <= 150:
        cause = (" Vous ne comptez pas les tirages écartés." if r["tirages_ignores"] == 0 else "")
        raise AssertionError(f"{quoi} : r['tirages_ignores'] vaut {r['tirages_ignores']}. Un tirage de 15 clients avec "
                             f"remise ne contient aucun client parti une fois sur trois environ ((14/15)¹⁵ ≈ 35 %) : "
                             f"environ 105 tirages sur 300 doivent être écartés ET comptés.{cause} "
                             f"{_CONSEIL_UNE_CLASSE}")

    quoi = "comparer_auc(y, champion, challenger, n_tirages=50, graine=5)"
    r1 = _appel(f, (y, a, b), {"n_tirages": 50, "graine": 5}, quoi)
    if _appel(f, (y, a, b), {"n_tirages": 50, "graine": 5}, quoi) != r1:
        raise AssertionError(f"Deux appels de {quoi} donnent deux résultats différents. {_CONSEIL_HASARD}")
    if len({_appel(f, (y, a, b), {"n_tirages": 30, "graine": g}, quoi)["bas"] for g in range(1, 4)}) == 1:
        raise AssertionError(f"Avec les graines 1, 2 et 3, {quoi.replace('graine=5', 'graine=g')} donne toujours le "
                             f"même intervalle : la graine n'est pas utilisée. {_CONSEIL_HASARD}")
    quoi = "comparer_auc(Series y d'index 1000 à 1599, Series champion, Series challenger, n_tirages=50, graine=5)"
    y_serie = pd.Series(y, index=range(1000, 1600))
    r_serie = _resultat_auc(_appel(f, (y_serie, df["champion"], df["challenger"]), {"n_tirages": 50, "graine": 5},
                                   quoi, "Lisez les Series par POSITION : `np.asarray(y)` d'abord (`y[i]` cherche des "
                                         "ÉTIQUETTES)."), quoi)
    if not all(math.isclose(r_serie[c], r1[c], rel_tol=1e-9, abs_tol=1e-12) for c in ("ecart", "bas", "haut")):
        raise AssertionError(f"Avec la même graine, des Series donnent {r_serie} et des tableaux {r1} : lisez les "
                             f"Series par POSITION (`np.asarray`), leur index ne compte pas.")

    for args, quoi, conseil in (
            ((y[:-1], a, b), "comparer_auc(y sans son dernier client, a, b)", "Les trois doivent avoir la même longueur."),
            ((np.r_[y[:-1], 2], a, b), "comparer_auc(y avec un 2, a, b)", "y ne contient que des 0 et des 1."),
            ((np.zeros(600, dtype=int), a, b), "comparer_auc(y sans aucun départ, a, b)",
             "Avec une seule classe, l'AUC n'existe pas, même sur tout l'échantillon."),
            ((y, np.r_[a[:-1], np.nan], b), "comparer_auc(y, a avec un NaN, b)", "Refusez les NaN dans les probabilités.")):
        _doit_lever(f, args, {"n_tirages": 50}, quoi, conseil)
    _doit_lever(f, (y, a, b), {"n_tirages": 50, "confiance": 95}, "comparer_auc(y, a, b, confiance=95)",
                "La confiance est une proportion : 0.95, pas 95.")


# ---------------------------------------------------------------------------- Défi 163

_CLES_CUPED = {"effet", "erreur_type", "effet_brut", "erreur_type_brute", "theta"}
_ORDRE_CUPED = ("effet_brut", "erreur_type_brute", "theta", "effet", "erreur_type")


_NAN_SANS_SCORE = ("Les clients sans score (NaN dans x_avant) rendent NaN toute moyenne qui les touche : donnez-leur "
                   "d'abord la moyenne des scores CONNUS (`x[np.isnan(x)] = x[~np.isnan(x)].mean()`, sur une COPIE).")


def _resultat_cuped(r, quoi: str, conseil_nan: str = _NAN_SANS_SCORE) -> dict:
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} doit renvoyer un dictionnaire, pas {_court(r)}.")
    if set(r) != _CLES_CUPED:
        raise AssertionError(f"{quoi} renvoie les clés {sorted(r)} : il faut exactement {sorted(_CLES_CUPED)}.")
    for cle in _ORDRE_CUPED:
        v = r[cle]
        if not _reel(v):
            raise AssertionError(f"{quoi} : r[{cle!r}] vaut {_court(v)}, il faut un nombre (float).")
        if math.isnan(v):
            raise AssertionError(f"{quoi} : r[{cle!r}] vaut NaN. {conseil_nan}")
    return r


def _proche(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _verifier_cuped(f, y, x, t, quoi: str) -> dict:
    r = _resultat_cuped(_appel(f, (y, x, t), None, quoi), quoi)
    attendu = _ref_cuped(y, x, t)
    for cle in _ORDRE_CUPED:
        if _proche(r[cle], attendu[cle]):
            continue
        cause = ""
        for mode, texte in _EXPLICATIONS_CUPED.items():
            variante = _ref_cuped(y, x, t, mode)
            if not _proche(variante[cle], attendu[cle]) and _proche(r[cle], variante[cle]):
                cause = texte
                break
        if not cause and cle == "effet" and _proche(r["effet"], r["effet_brut"]):
            cause = _EXPLICATIONS_CUPED["centrage_groupe"]
        raise AssertionError(f"{quoi} : r[{cle!r}] vaut {r[cle]:.6g} au lieu de {attendu[cle]:.6g}"
                             + (f" : {cause}" if cause else "."))
    return r


def verifier_effet_cuped(f) -> None:
    df = campagne_exemple()
    y, x, t = (df[c].to_numpy().copy() for c in ("churn", "score_avant", "offre"))  # modifiables (pandas 3)
    copies = (y.copy(), x.copy(), t.copy())
    _verifier_cuped(f, y, x, t, "effet_cuped(churn, score_avant, offre) sur campagne_exemple()")
    if not all(_intact(c, v) for c, v in zip(copies, (y, x, t))):
        raise AssertionError("effet_cuped(churn, score_avant, offre) a modifié un des tableaux reçus : remplir les NaN "
                             "« en place » change les données de l'appelant (ses nouveaux clients ont soudain un "
                             "score !). Travaillez sur une COPIE : `x = np.array(x_avant, dtype=float)` (np.array "
                             "copie ; np.asarray peut renvoyer le MÊME tableau).")

    # 12 clients : un ddof oublié se voit (n / (n - 1) = 1,09)
    y12 = [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    x12 = [0.8, 0.2, np.nan, 0.6, 0.7, 0.1, np.nan, 0.3, 0.2, 0.5, 0.9, 0.4]
    t12 = [True, False, True, False, True, False, True, False, True, False, True, False]
    _verifier_cuped(f, y12, x12, t12, f"effet_cuped({y12}, {x12}, {t12})".replace("nan", "NaN"))
    _verifier_cuped(f, y12, x12, [int(v) for v in t12], "effet_cuped(y, x, traite en 1 / 0)")

    quoi = "effet_cuped(churn, [0.5] * n, offre) (un score constant)"
    conseil = ("Un score constant n'apprend rien : θ = 0. Testez `x.max() == x.min()` AVANT de diviser par sa "
               "variance, qui vaut 0.")
    r = _resultat_cuped(_appel(f, (y, np.full(len(y), 0.5), t), None, quoi, conseil), quoi, conseil)
    if r["theta"] != 0 or not _proche(r["effet"], r["effet_brut"]):
        raise AssertionError(f"{quoi} renvoie θ = {r['theta']} : un score constant n'explique rien, θ = 0 et l'effet "
                             f"corrigé est l'effet brut.")
    quoi = "effet_cuped(churn, [NaN] * n, offre) (aucun score connu)"
    conseil = ("Aucun score connu : rien à corriger, θ = 0. La moyenne d'un tableau VIDE de scores connus vaut NaN "
               "(avec un RuntimeWarning) : traitez ce cas à part.")
    r = _resultat_cuped(_appel(f, (y, np.full(len(y), np.nan), t), None, quoi, conseil), quoi, conseil)
    if r["theta"] != 0 or not _proche(r["effet"], r["effet_brut"]):
        raise AssertionError(f"{quoi} renvoie θ = {r['theta']} : sans aucun score connu, θ = 0.")

    quoi = "effet_cuped(Series churn d'index n - 1, …, 1, 0, Series score d'index 0, 1, …, n - 1, liste offre)"
    y_serie = pd.Series(y, index=np.arange(len(y))[::-1])
    r_serie = _resultat_cuped(_appel(f, (y_serie, pd.Series(x), t.tolist()), None, quoi,
                                     "Lisez les Series par POSITION : `np.asarray(y, dtype=float)`."), quoi)
    attendu = _ref_cuped(y, x, t)
    if not all(_proche(r_serie[c], attendu[c]) for c in _ORDRE_CUPED):
        raise AssertionError(f"{quoi} ne donne pas le même résultat que les tableaux ({r_serie['effet']:.6g} au lieu "
                             f"de {attendu['effet']:.6g}) : pandas aligne deux Series sur leurs ÉTIQUETTES. Lisez tout "
                             f"par POSITION : `np.asarray(...)` d'abord.")

    n = len(y)
    for args, quoi, conseil in (
            ((y[:-1], x, t), "effet_cuped(y sans son dernier client, x, traite)", "Les trois doivent avoir la même longueur."),
            ((np.r_[np.nan, y[1:]], x, t), "effet_cuped(y avec un NaN, x, traite)",
             "y doit être connu pour tous les clients du test."),
            ((y, x, np.where(t, "oui", "non")), "effet_cuped(y, x, ['oui', 'non', …])",
             "traite : True / False ou 1 / 0 (« non » est une chaîne non vide, donc VRAIE pour Python !)."),
            ((y, x, np.where(t, 2, 0)), "effet_cuped(y, x, [2, 0, …])", "traite : True / False ou 1 / 0."),
            ((y, x, np.arange(n) == 0), "effet_cuped(y, x, un seul client traité)",
             "Avec un seul client dans un groupe, sa variance (ddof=1) n'existe pas : au moins 2 par groupe.")):
        _doit_lever(f, args, None, quoi, conseil)


# ---------------------------------------------------------------------------- la série

SERIE = Serie(
    code="26",
    titre="Conclure sans se tromper : bootstrap, permutation, puissance, tests multiples, CUPED",
    module_eleve="defis.serie_26",
    module_corrige="defis.corrige_serie_26",
    defis=(
        Defi(158, "intervalle_bootstrap", "🟢", "—", verifier_intervalle_bootstrap),
        Defi(159, "p_valeur_permutation", "🟢", "—", verifier_p_valeur_permutation),
        Defi(160, "taille_echantillon", "🟡", "—", verifier_taille_echantillon),
        Defi(161, "benjamini_hochberg", "🟡", "—", verifier_benjamini_hochberg),
        Defi(162, "comparer_auc", "🟡", "3.4", verifier_comparer_auc),
        Defi(163, "effet_cuped", "🔴", "—", verifier_effet_cuped),
    ),
)

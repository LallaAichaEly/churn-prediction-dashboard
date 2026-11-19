"""Vérifications automatiques de la série 38 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est le service juridique, qui veut des chiffres qu'on peut défendre, et une association de
consommateurs, qui refait vos calculs :

- partout, il choisit des clients où chaque erreur classique se voit : un score à 0.5 tout rond, un groupe sans
  aucun parti (le rappel n'existe pas), des âges absents écrits None ou NaN (deux NaN ne sont pas égaux), un
  rapport de 4/5 tout juste que le float calcule 0.7999999999999999, un petit groupe à 100 % qui ne doit pas
  servir de référence, un paradoxe de Simpson, des cases vides dans un tableau croisé, une cible de 55 % sur
  100 partis (0.55 × 100 vaut 55.00000000000001) ;
- quand votre résultat est faux, il rejoue les erreurs classiques (> au lieu de >=, le mauvais dénominateur,
  rappel et précision échangés, 0.0 au lieu de None, absents jetés, comparaison à la moyenne au lieu du mieux
  servi, poids de chaque groupe au lieu de ceux de la clientèle, χ² sans les cases vides, V sans racine,
  décision lue dans le vieux champ "score", scores décalés d'une position, `math.ceil(cible * P)`…) pour nommer
  la vôtre ;
- au défi 234, il vous prête un modèle ESPION, qui compte vos appels et regarde ce que vous lui envoyez : un
  appel par client, ou un client modifié (même le temps d'un appel), se voient ;
- aux défis 232, 233 et 235, il compte les lignes exécutées dans votre fichier sur 100 000 clients ou plus :
  une passe par groupe et par strate, une passe par case du tableau croisé, ou un recomptage pour chaque seuil
  essayé se voient, quelle que soit la machine ;
- et il vérifie que vos fonctions ne modifient pas ce qu'elles reçoivent.
"""

from __future__ import annotations

import copy
import math
import random
import sys
from collections import Counter
from fractions import Fraction
from numbers import Real

from .equite import (AGES, COLONNES, SEUIL, Bascule, Groupe, Impact, Proxy, Reglage, Standardisation,
                     clients_exemple, generer_clients, modele, selection)
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


class _BoucleSansFin(BaseException):
    """Levée dans VOTRE fonction quand elle travaille trop (BaseException : `except Exception` ne l'attrape pas)."""


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


_CONSEIL_LENT = "Votre fonction a été arrêtée : elle exécute beaucoup trop de lignes (une boucle dans une boucle ?)."


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = 2_000_000, conseil_lent: str = ""):
    """f(*args), en transformant toute erreur en message lisible."""
    try:
        return _trace(f, lambda: f(*args), lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu=ValueError) -> None:
    """Exige que f(*args) lève `attendu`."""
    try:
        resultat = _trace(f, lambda: f(*args), 2_000_000)
    except NotImplementedError:
        raise
    except attendu:
        return
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {_CONSEIL_LENT}") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _cliche(clients: list[dict]) -> list[dict]:
    """Une copie de chaque client (les valeurs, NaN compris, sont les MÊMES objets : la comparaison reste juste)."""
    return [dict(c) for c in clients]


def _intact(avant, apres, quoi: str, nom: str) -> None:
    if avant != apres:
        raise AssertionError(f"{quoi} a modifié {nom} reçu(s). Une fonction ne touche pas à ce qu'on lui prête : "
                             "lisez les valeurs, comptez à côté, construisez de NOUVEAUX objets.")


def _pas_none(resultat, quoi: str) -> None:
    if resultat is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")


def _nombre(x) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool)


def _proche(obtenu, attendu) -> bool:
    """Deux taux égaux à l'arrondi près ; None n'est égal qu'à None (et 0.0 n'est pas None)."""
    if attendu is None or obtenu is None:
        return obtenu is None and attendu is None
    return _nombre(obtenu) and math.isclose(obtenu, attendu, rel_tol=1e-9, abs_tol=1e-12)


def _memes_dicts(obtenu, attendu, egal) -> bool:
    """Mêmes clés, dans le même ordre, et des valeurs égales selon `egal`."""
    return (isinstance(obtenu, dict) and list(obtenu) == list(attendu)
            and all(egal(obtenu[k], attendu[k]) for k in attendu))


def _absent(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _cle(client: dict, colonne: str):
    valeur = client.get(colonne)
    return None if _absent(valeur) else valeur


def _champs_qui_different(obtenu: tuple, attendu: tuple, egal) -> list[int]:
    return [i for i in range(len(attendu)) if not egal(i, obtenu[i], attendu[i])]


def _bug_qui_explique(obtenu: tuple, attendu: tuple, rejouer, bugs: dict, egal) -> str:
    """Le conseil du premier bug dont le rejeu donne VOS valeurs sur tous les champs où vous vous trompez."""
    champs = _champs_qui_different(obtenu, attendu, egal)
    for bug, conseil in bugs.items():
        try:
            faux = rejouer(bug)
        except Exception:
            continue
        if all(egal(i, obtenu[i], faux[i]) for i in champs):
            return conseil
    return ""


# ============================================================================ défi 230 : taux_par_groupe


def _ref_taux(clients: list[dict], colonne: str, seuil: float = SEUIL, bug: str = "") -> dict:
    """La bonne réponse ; `bug` rejoue une erreur classique."""
    comptes: dict = {}
    for client in clients:
        valeur = client.get(colonne)
        if bug == "sans_absents" and _absent(valeur):
            continue
        cle = valeur if bug == "nan_a_part" else (None if _absent(valeur) else valeur)
        score = client["score"]
        selectionne = score > seuil if bug == "strict" else score >= seuil
        parti = client["parti"] == 1
        c = comptes.setdefault(cle, [0, 0, 0, 0, 0])
        c[0] += 1
        c[1] += selectionne
        c[2] += parti
        c[3] += selectionne and parti
        c[4] += selectionne and not parti
    if bug == "tri":
        comptes = dict(sorted(comptes.items(), key=lambda kv: (kv[0] is None, str(kv[0]))))
    vide = 0.0 if bug == "zero" else None
    resultat = {}
    for groupe, (n, s, p, vp, fp) in comptes.items():
        negatifs = n - p
        taux = s / (len(clients) if bug == "population" else n)
        tvp = vp / p if p else vide
        tfp = (fp / n if bug == "tfp_effectif" else fp / negatifs) if negatifs else vide
        precision = vp / s if s else vide
        if bug == "rappel_precision":
            tvp, precision = precision, tvp
        resultat[groupe] = Groupe(n, s, p, vp, fp, taux, tvp, tfp, precision)
    return resultat


_BUGS_TAUX = {
    "strict": "Vous sélectionnez les clients dont le score est STRICTEMENT supérieur au seuil : l'offre part dès "
              "score >= seuil (un score de 0.5 tout rond la reçoit).",
    "nan_a_part": "Les âges NaN ne rejoignent pas le groupe None : `float('nan') == float('nan')` est False, donc chaque "
                  "NaN devient sa propre clé de dict. Une valeur absente, None OU NaN, va dans le groupe None "
                  "(`isinstance(v, float) and math.isnan(v)`).",
    "sans_absents": "Des clients ont disparu : ceux dont la valeur est absente (clé manquante, None ou NaN) forment le "
                    "groupe None. On ne perd personne dans un audit (`groupby` jette les absents par défaut : "
                    "`dropna=False`).",
    "population": "Votre taux de sélection divise par le nombre TOTAL de clients. Chaque taux se calcule dans SON "
                  "groupe : sélectionnés du groupe / effectif du groupe.",
    "rappel_precision": "Vous avez échangé le taux de vrais positifs (le RAPPEL : vrais positifs / PARTIS) et la "
                        "précision (vrais positifs / SÉLECTIONNÉS). Le rappel répond à « parmi ceux qui partent, "
                        "combien le modèle en repère-t-il ? ».",
    "tfp_effectif": "Votre taux de faux positifs divise par l'effectif du groupe. Il se calcule parmi ceux qui sont "
                    "RESTÉS : faux positifs / (effectif − positifs).",
    "zero": "Un taux dont le dénominateur est nul vaut None (« pas mesurable »), pas 0.0 (« mesuré, et nul ») : un "
            "groupe où personne n'est parti n'a pas un rappel de 0 %, il n'a pas de rappel du tout.",
    "tri": "Les groupes doivent apparaître dans l'ordre où chaque valeur apparaît pour la première fois dans les "
           "clients, pas triés.",
}

_CHAMPS_GROUPE = Groupe._fields


def _meme_groupe(obtenu, attendu: Groupe) -> bool:
    if not isinstance(obtenu, tuple) or len(obtenu) != len(attendu):
        return False
    for i, (o, a) in enumerate(zip(obtenu, attendu)):
        if i < 5:
            if type(o) is not int or o != a:
                return False
        elif not _proche(o, a):
            return False
    return True


def _un_taux(f, clients: list[dict], colonne: str, seuil: float, quoi: str) -> None:
    attendu = _ref_taux(clients, colonne, seuil)
    avant = _cliche(clients)
    obtenu = _appel(f, (clients, colonne, seuil), quoi)
    _pas_none(obtenu, quoi)
    _intact(avant, clients, quoi, "les clients")
    if not isinstance(obtenu, dict):
        raise AssertionError(f"{quoi} renvoie {type(obtenu).__name__} ; attendu un dict {{groupe: Groupe(...)}}.")
    if _memes_dicts(obtenu, attendu, _meme_groupe):
        return
    for bug, conseil in _BUGS_TAUX.items():
        if _memes_dicts(obtenu, _ref_taux(clients, colonne, seuil, bug), _meme_groupe):
            raise AssertionError(f"{quoi} : {conseil}")
    if list(obtenu) != list(attendu):
        raise AssertionError(f"{quoi} donne les groupes {_court(list(obtenu), 100)} au lieu de "
                             f"{_court(list(attendu), 100)} (dans l'ordre d'apparition, les absents dans None).")
    for groupe, a in attendu.items():
        o = obtenu[groupe]
        if not isinstance(o, tuple) or len(o) != len(a):
            raise AssertionError(f"{quoi} : le groupe {groupe!r} vaut {_court(o)} ; attendu un Groupe(...) de neuf "
                                 "valeurs (le tuple nommé de defis/equite.py).")
        for i, nom in enumerate(_CHAMPS_GROUPE):
            if not _meme_groupe(o[: i + 1] + a[i + 1:], a):
                type_ = "" if i >= 5 or type(o[i]) is int else f" ({type(o[i]).__name__} ; attendu un int)"
                raise AssertionError(f"{quoi} : pour le groupe {groupe!r}, {nom} vaut {_court(o[i])}{type_} au lieu "
                                     f"de {a[i]!r}.")


def _c(groupe, score, parti, colonne="age", **autres) -> dict:
    return {colonne: groupe, "score": score, "parti": parti, **autres}


def _sans_cle(score, parti) -> dict:
    return {"score": score, "parti": parti}


def verifier_taux_par_groupe(f) -> None:
    def nan():
        return float("nan")  # un NOUVEL objet à chaque fois, comme dans un vrai fichier

    cas = [
        ("l'exemple de la docstring", [_c("65+", 0.31, 1), _c("65+", 0.52, 0), _c("30-44", 0.5, 1),
                                       _c("30-44", 0.12, 0), _c(None, 0.8, 1), _c(nan(), 0.2, 1)]),
        ("des scores à 0.5 tout rond", [_c("A", 0.5, 1), _c("A", 0.5, 0), _c("A", 0.49, 1), _c("B", 0.5, 0)]),
        ("un groupe sans parti, un sans sélectionné, un sans resté",
         [_c("A", 0.9, 0), _c("A", 0.1, 0), _c("B", 0.1, 1), _c("B", 0.2, 0), _c("C", 0.7, 1), _c("C", 0.3, 1)]),
        ("des groupes de tailles différentes", [_c("A", 0.6, 1)] * 3 + [_c("B", 0.6, 0), _c("B", 0.1, 1)]),
        ("des âges absents de trois façons", [_c("A", 0.6, 1), _sans_cle(0.7, 0), _c(nan(), 0.2, 1),
                                              _c(None, 0.55, 1), _c(nan(), 0.9, 0), _c("A", 0.4, 0)]),
        ("des groupes qui n'arrivent pas dans l'ordre alphabétique",
         [_c("65+", 0.2, 1), _c("18-29", 0.8, 1), _c("45-64", 0.6, 0), _c("18-29", 0.3, 0)]),
        ("des parti en booléens", [_c("A", 0.7, True), _c("A", 0.2, False)]),
        ("aucun client", []),
    ]
    for nom, clients in cas:
        _un_taux(f, clients, "age", SEUIL, f"taux_par_groupe(<{nom}>, 'age')")
    alea = random.Random(230)
    for n in range(25):
        clients = generer_clients(alea.randint(1, 400), graine=alea.randrange(10**6))
        colonne = alea.choice(("age", "zone", "contrat", "paiement", "appels"))
        seuil = alea.choice((0.5, 0.3, 0.7, round(alea.random(), 3), 0.0, 1.0))
        _un_taux(f, clients, colonne, seuil, f"taux_par_groupe(generer_clients(…), {colonne!r}, {seuil}) "
                                             f"(tirage n° {n})")
    _un_taux(f, clients_exemple(), "age", SEUIL, "taux_par_groupe(clients_exemple(), 'age')")
    base = [_c("A", 0.6, 1)]
    for clients, seuil, attendu, conseil in (
            ([_c("A", 1.2, 1)], SEUIL, ValueError, "un score est une probabilité, entre 0 et 1."),
            ([_c("A", -0.1, 0)], SEUIL, ValueError, "un score est une probabilité, entre 0 et 1."),
            ([_c("A", float("nan"), 0)], SEUIL, ValueError, "un score NaN n'est pas un score (NaN >= 0.5 est False : "
                                                            "il passerait inaperçu)."),
            ([_c("A", "0.6", 1)], SEUIL, TypeError, "un texte n'est pas un score."),
            ([_c("A", True, 1)], SEUIL, TypeError, "True n'est pas un score (même si True >= 0.5)."),
            ([_c("A", 0.6, 2)], SEUIL, ValueError, "parti vaut 0 ou 1."),
            ([_c("A", 0.6, None)], SEUIL, ValueError, "parti vaut 0 ou 1."),
            (base, 1.5, ValueError, "le seuil est une probabilité, entre 0 et 1."),
            (base, float("nan"), ValueError, "un seuil NaN ne sélectionnerait personne sans prévenir."),
            (base, "0.5", TypeError, "un texte n'est pas un seuil."),
            (base, True, TypeError, "True n'est pas un seuil.")):
        _doit_lever(f, (clients, "age", seuil), f"taux_par_groupe({_court(clients, 80)}, 'age', {seuil!r})",
                    conseil, attendu)


# ============================================================================ défi 231 : quatre_cinquiemes


def _ref_quatre(groupes: dict, min_effectif: int = 30, bug: str = "") -> Impact:
    """La bonne réponse, en fractions ; `bug` rejoue une erreur classique."""
    juges, ignores = [], []
    for nom, groupe in groupes.items():
        (juges if bug == "petits" or groupe.effectif >= min_effectif else ignores).append(nom)
    if not juges:
        return Impact(None, {}, [], ignores)
    taux = {nom: Fraction(groupes[nom].selectionnes, groupes[nom].effectif) for nom in juges}
    if bug == "ref_compte":
        reference = max(juges, key=lambda nom: groupes[nom].selectionnes)
    elif bug == "ref_premier":
        reference = juges[0]
    else:
        reference = juges[0]
        for nom in juges[1:]:
            if taux[nom] > taux[reference] or (bug == "ref_derniere" and taux[nom] == taux[reference]):
                reference = nom
    base = taux[reference]
    if bug == "moyenne":
        base = Fraction(sum(groupes[n].selectionnes for n in juges), sum(groupes[n].effectif for n in juges))
    if base == 0:
        return Impact(reference, {nom: None for nom in juges}, [], ignores)
    if bug == "inverse":
        ratios = {nom: (base / taux[nom] if taux[nom] else None) for nom in juges}
        signales = [nom for nom in juges if ratios[nom] is None or ratios[nom] > Fraction(5, 4)]
    else:
        ratios = {nom: taux[nom] / base for nom in juges}
        if bug == "float":
            s_ref, n_ref = groupes[reference].selectionnes, groupes[reference].effectif
            signales = [nom for nom in juges
                        if (groupes[nom].selectionnes / groupes[nom].effectif) / (s_ref / n_ref) < 0.8]
        elif bug == "large":
            signales = [nom for nom in juges if ratios[nom] <= Fraction(4, 5)]
        else:
            signales = [nom for nom in juges if ratios[nom] < Fraction(4, 5)]
    return Impact(reference, {nom: (None if r is None else float(r)) for nom, r in ratios.items()}, signales, ignores)


_BUGS_QUATRE = {
    "petits": "Les groupes d'effectif < min_effectif ne sont pas jugés : ils vont dans `ignores`, et ne peuvent pas "
              "servir de référence (3 sélectionnés sur 3, c'est 100 %, mais trois clients ne prouvent rien).",
    "ref_compte": "Votre référence est le groupe qui a le plus de SÉLECTIONNÉS. C'est le groupe au plus fort TAUX de "
                  "sélection (sélectionnés / effectif) : un grand groupe a mécaniquement plus de sélectionnés.",
    "ref_premier": "Votre référence est le premier groupe du dict. C'est le groupe jugé au plus fort taux de sélection.",
    "ref_derniere": "À égalité de taux, la référence est le PREMIER groupe dans l'ordre du dict (comparez avec > "
                    "strict, pas >=).",
    "moyenne": "Vous comparez chaque groupe au taux MOYEN (tous groupes confondus). La règle des 4/5 compare chaque "
               "groupe au groupe le MIEUX servi : la moyenne contient déjà le groupe désavantagé, et adoucit l'écart.",
    "inverse": "Vos ratios sont à l'envers : c'est le taux du groupe divisé par celui de la référence (la référence "
               "vaut 1.0, les autres moins).",
    "float": "Vous comparez des floats à la limite : (20 / 30) / (25 / 30) vaut 0.7999999999999999 en float, alors "
             "que le rapport exact vaut 4/5 tout juste, qui n'est PAS « moins de 4/5 ». Décidez en entiers : "
             "5 × s × n_ref < 4 × s_ref × n (ou avec `fractions.Fraction`).",
    "large": "Un groupe est signalé si son ratio est STRICTEMENT inférieur à 4/5 : 4/5 tout juste n'est pas signalé "
             "(< et pas <=).",
}


def _egal_impact(i: int, obtenu, attendu) -> bool:
    if i == 1:
        return _memes_dicts(obtenu, attendu, _proche)
    if i == 0:
        return obtenu == attendu and type(obtenu) is type(attendu)
    return isinstance(obtenu, list) and obtenu == attendu


_CHAMPS_IMPACT = Impact._fields


def _un_quatre(f, groupes: dict, min_effectif: int, quoi: str) -> None:
    attendu = _ref_quatre(groupes, min_effectif)
    avant = copy.deepcopy(groupes)
    obtenu = _appel(f, (groupes, min_effectif), quoi)
    _pas_none(obtenu, quoi)
    _intact(avant, groupes, quoi, "les groupes")
    if not isinstance(obtenu, tuple) or len(obtenu) != 4:
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} ; attendu Impact(reference, ratios, signales, ignores).")
    champs = _champs_qui_different(obtenu, attendu, _egal_impact)
    if not champs:
        return
    conseil = _bug_qui_explique(obtenu, attendu, lambda bug: _ref_quatre(groupes, min_effectif, bug), _BUGS_QUATRE,
                                _egal_impact)
    i = champs[0]
    detail = (f"{quoi} : {_CHAMPS_IMPACT[i]} vaut {_court(obtenu[i], 140)} au lieu de {_court(attendu[i], 140)} "
              f"(sélectionnés / effectif : {_court({n: (g.selectionnes, g.effectif) for n, g in groupes.items()}, 160)}).")
    raise AssertionError(f"{detail} {conseil}".rstrip())


def verifier_quatre_cinquiemes(f) -> None:
    s = selection
    cas = [
        ("l'exemple de la docstring", {"A": s(25, 30), "B": s(19, 30), "C": s(3, 3)}, 30),
        ("4/5 tout juste, que le float calcule juste", {"A": s(24, 36), "B": s(16, 30)}, 30),
        ("4/5 tout juste, que le float calcule 0.7999999999999999", {"A": s(25, 30), "B": s(20, 30)}, 30),
        ("4/5 tout juste, encore", {"A": s(3, 30), "B": s(4, 50), "C": s(30, 45)}, 30),
        ("deux groupes ex aequo en tête", {"A": s(15, 60), "B": s(10, 40), "C": s(5, 40)}, 30),
        ("un grand groupe moins bien servi", {"A": s(30, 100), "B": s(20, 40), "C": s(9, 40)}, 30),
        ("le mieux servi n'est pas le premier", {"A": s(10, 50), "B": s(30, 50), "C": s(25, 50)}, 30),
        ("un groupe sans aucun sélectionné", {"A": s(0, 40), "B": s(12, 40)}, 30),
        ("personne n'a rien reçu", {"A": s(0, 40), "B": s(0, 35)}, 30),
        ("aucun groupe assez grand", {"A": s(1, 3), None: s(2, 2)}, 30),
        ("aucun groupe", {}, 30),
        ("un groupe None", {None: s(30, 40), "A": s(10, 40)}, 30),
        ("min_effectif = 1", {"A": s(1, 1), "B": s(4, 5)}, 1),
    ]
    for nom, groupes, min_effectif in cas:
        _un_quatre(f, groupes, min_effectif, f"quatre_cinquiemes(<{nom}>, {min_effectif}) "
                                             f"({_court({k: (g.selectionnes, g.effectif) for k, g in groupes.items()}, 90)})")
    exemple = clients_exemple()
    for colonne in ("age", "zone", "contrat", "paiement"):
        for min_effectif in (30, 200):
            _un_quatre(f, _ref_taux(exemple, colonne), min_effectif,
                       f"quatre_cinquiemes(taux_par_groupe(clients_exemple(), {colonne!r}), {min_effectif})")
    alea = random.Random(231)
    for n in range(200):
        groupes = {}
        for k in range(alea.randint(1, 6)):
            effectif = alea.choice((alea.randint(1, 40), alea.randint(30, 60), alea.randint(1, 2000)))
            groupes[alea.choice(("G", "H", "K")) + str(k)] = s(alea.randint(0, effectif), effectif)
        _un_quatre(f, groupes, alea.choice((1, 10, 30)), f"quatre_cinquiemes(<tirage n° {n}>)")
    for groupes, min_effectif, attendu, conseil in (
            ({"A": s(5, 40)}, 0, ValueError, "min_effectif vaut au moins 1."),
            ({"A": s(5, 40)}, 2.5, TypeError, "min_effectif est un int."),
            ({"A": s(5, 40)}, True, TypeError, "True n'est pas un effectif."),
            ({"A": s(50, 40)}, 30, ValueError, "50 sélectionnés sur 40 clients : impossible."),
            ({"A": s(-1, 40)}, 30, ValueError, "un nombre de sélectionnés négatif : impossible.")):
        _doit_lever(f, (groupes, min_effectif), f"quatre_cinquiemes({_court(groupes, 60)}, {min_effectif!r})",
                    conseil, attendu)


# ============================================================================ défi 232 : taux_standardises


def _ref_standard(clients: list[dict], colonne: str, strate: str, seuil: float = SEUIL, bug: str = "") -> Standardisation:
    """La bonne réponse, en fractions ; `bug` rejoue une erreur classique."""
    effectifs, selections, par_strate = Counter(), Counter(), Counter()
    groupes: dict = {}
    strates: dict = {}
    for client in clients:
        gv, sv = client.get(colonne), client.get(strate)
        if bug == "sans_absents" and (_absent(gv) or _absent(sv)):
            continue
        g, s = (None if _absent(gv) else gv), (None if _absent(sv) else sv)
        groupes.setdefault(g, None)
        strates.setdefault(s, None)
        effectifs[g, s] += 1
        selections[g, s] += client["score"] >= seuil
        par_strate[s] += 1
    if not groupes:
        return Standardisation({}, {}, [], [])
    communes = [s for s in strates if all(effectifs[g, s] for g in groupes)]
    exclues = [s for s in strates if s not in communes]
    if bug == "toutes_strates":
        communes, exclues = list(strates), []
    if not communes:
        raise ValueError("aucune strate commune")
    brut = {g: Fraction(sum(selections[g, s] for s in strates), sum(effectifs[g, s] for s in strates)) for g in groupes}

    def taux(g, s):
        return Fraction(selections[g, s], effectifs[g, s]) if effectifs[g, s] else Fraction(0)

    standardise = {}
    for g in groupes:
        if bug == "poids_egaux":
            poids = {s: Fraction(1, len(communes)) for s in communes}
        elif bug == "poids_du_groupe":
            total = sum(effectifs[g, s] for s in communes)
            poids = {s: Fraction(effectifs[g, s], total) for s in communes}
        elif bug == "poids_toutes":
            total = sum(par_strate.values())
            poids = {s: Fraction(par_strate[s], total) for s in communes}
        else:
            total = sum(par_strate[s] for s in communes)
            poids = {s: Fraction(par_strate[s], total) for s in communes}
        standardise[g] = sum(poids[s] * taux(g, s) for s in communes)
    return Standardisation({g: float(t) for g, t in brut.items()}, {g: float(t) for g, t in standardise.items()},
                           communes, exclues)


_BUGS_STANDARD = {
    "poids_du_groupe": "Vous pondérez chaque groupe par SA PROPRE répartition entre les strates : vous retrouvez son "
                       "taux brut. Standardiser, c'est donner à TOUS les groupes la même répartition, celle de toute "
                       "la clientèle : poids(s) = clients de tous les groupes dans s / clients des strates communes.",
    "poids_egaux": "Vous faites la moyenne simple des taux par strate (chaque strate pèse pareil). Chaque strate pèse "
                   "ce qu'elle pèse dans toute la clientèle : poids(s) = clients de s / clients des strates communes.",
    "toutes_strates": "Une strate où un groupe n'a personne ne peut pas servir à comparer : on ne connaît pas le taux "
                      "de ce groupe dans cette strate (ce n'est pas 0). Gardez les strates COMMUNES à tous les groupes, "
                      "et mettez les autres dans `exclues`.",
    "poids_toutes": "Vos poids sont calculés sur TOUTES les strates, y compris les exclues : ils ne font plus 1. "
                    "Divisez par le nombre de clients des seules strates communes.",
    "sans_absents": "Des clients ont disparu : une valeur absente (clé manquante, None ou NaN), de groupe ou de strate, "
                    "forme le groupe ou la strate None. On ne perd personne.",
}


def _egal_standard(i: int, obtenu, attendu) -> bool:
    if i < 2:
        return _memes_dicts(obtenu, attendu, _proche)
    return isinstance(obtenu, list) and obtenu == attendu


_CHAMPS_STANDARD = Standardisation._fields


def _un_standard(f, clients, colonne, strate, seuil, quoi, lignes_max: int | None = 2_000_000) -> None:
    attendu = _ref_standard(clients, colonne, strate, seuil)
    avant = _cliche(clients) if len(clients) < 20_000 else None
    obtenu = _appel(f, (clients, colonne, strate, seuil), quoi, lignes_max=lignes_max,
                    conseil_lent="Votre fonction a été arrêtée : refaire une passe sur tous les clients pour chaque "
                                 "groupe et chaque strate est bien trop lent. Comptez tout en UNE passe, dans des "
                                 "Counter indexés par (groupe, strate).")
    _pas_none(obtenu, quoi)
    if avant is not None:
        _intact(avant, clients, quoi, "les clients")
    if not isinstance(obtenu, tuple) or len(obtenu) != 4:
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} ; attendu Standardisation(brut, standardise, strates, "
                             "exclues).")
    champs = _champs_qui_different(obtenu, attendu, _egal_standard)
    if not champs:
        return

    def rejouer(bug):
        return _ref_standard(clients, colonne, strate, seuil, bug)

    conseil = _bug_qui_explique(obtenu, attendu, rejouer, _BUGS_STANDARD, _egal_standard)
    i = champs[0]
    raise AssertionError(f"{quoi} : {_CHAMPS_STANDARD[i]} vaut {_court(obtenu[i], 160)} au lieu de "
                         f"{_court(attendu[i], 160)}. {conseil}".rstrip())


def _lot(g, s, n: int, sel: int) -> list[dict]:
    return [{"g": g, "s": s, "score": 0.9}] * sel + [{"g": g, "s": s, "score": 0.1}] * (n - sel)


_GRANDS: dict = {}


def _grands_clients() -> list[dict]:
    """120 000 clients, générés une seule fois (ils servent aux défis 232 et 233)."""
    if "clients" not in _GRANDS:
        _GRANDS["clients"] = generer_clients(120_000, graine=38)
    return _GRANDS["clients"]


def verifier_taux_standardises(f) -> None:
    nan = float("nan")
    simpson = _lot("A", "x", 30, 27) + _lot("A", "y", 10, 1) + _lot("B", "x", 10, 10) + _lot("B", "y", 30, 6)
    cas = [
        ("le paradoxe de Simpson de la docstring", simpson),
        ("une strate qui n'existe que dans un groupe", simpson + _lot("B", "z", 5, 5)),
        ("des strates de tailles très différentes", _lot("A", "x", 90, 60) + _lot("A", "y", 10, 2)
         + _lot("B", "x", 20, 18) + _lot("B", "y", 80, 8) + _lot("C", "x", 5, 1) + _lot("C", "y", 5, 5)),
        ("des groupes et des strates absents", _lot("A", "x", 4, 3) + _lot(None, "x", 3, 1) + _lot(nan, "y", 2, 2)
         + _lot("A", None, 3, 0) + _lot("A", "y", 2, 1) + _lot(None, None, 2, 1) + [{"s": "x", "score": 0.7}]
         + [{"g": "A", "score": 0.6}] + _lot("A", nan, 1, 1)),
        ("un seul groupe", _lot("A", "x", 7, 3) + _lot("A", "y", 3, 3)),
        ("des scores à 0.5 tout rond", [{"g": "A", "s": "x", "score": 0.5}, {"g": "B", "s": "x", "score": 0.49},
                                        {"g": "A", "s": "y", "score": 0.2}, {"g": "B", "s": "y", "score": 0.5}]),
        ("aucun client", []),
    ]
    for nom, clients in cas:
        _un_standard(f, clients, "g", "s", SEUIL, f"taux_standardises(<{nom}>, 'g', 's')")
    alea = random.Random(232)
    for n in range(25):
        clients = generer_clients(alea.randint(20, 600), graine=alea.randrange(10**6))
        colonne, strate = alea.sample(("age", "zone", "contrat", "paiement", "internet"), 2)
        seuil = alea.choice((0.5, 0.3, round(alea.random(), 2)))
        try:
            _ref_standard(clients, colonne, strate, seuil)
        except ValueError:
            continue
        _un_standard(f, clients, colonne, strate, seuil,
                     f"taux_standardises(generer_clients(…), {colonne!r}, {strate!r}, {seuil}) (tirage n° {n})")
    exemple = clients_exemple()
    for strate in ("contrat", "paiement", "zone"):
        _un_standard(f, exemple, "age", strate, SEUIL, f"taux_standardises(clients_exemple(), 'age', {strate!r})")
    _un_standard(f, _grands_clients(), "age", "appels", SEUIL,
                 "taux_standardises(<120 000 clients>, 'age', 'appels')", lignes_max=7_000_000)
    for args, attendu, conseil in (
            ((simpson, "g", "g"), ValueError, "on ne standardise pas un groupe par lui-même."),
            ((_lot("A", "x", 3, 1) + _lot("B", "y", 3, 2), "g", "s"), ValueError,
             "aucune strate n'a des clients des deux groupes : rien n'est comparable."),
            (([{"g": "A", "s": "x", "score": 2.0}], "g", "s"), ValueError, "un score est entre 0 et 1."),
            (([{"g": "A", "s": "x", "score": "0.9"}], "g", "s"), TypeError, "un texte n'est pas un score."),
            ((simpson, "g", "s", -0.5), ValueError, "le seuil est entre 0 et 1."),
            ((simpson, "g", "s", None), TypeError, "None n'est pas un seuil.")):
        _doit_lever(f, args, f"taux_standardises(<{len(args[0])} clients>, {', '.join(map(repr, args[1:]))})",
                    conseil, attendu)


# ============================================================================ défi 233 : proxies


def _ref_proxies(clients: list[dict], protegee: str, colonnes, bug: str = "") -> list[Proxy]:
    """La bonne réponse (V² exact en fractions) ; `bug` rejoue une erreur classique."""
    resultats = []
    for colonne in colonnes:
        paires = []
        for client in clients:
            a, b = client.get(protegee), client.get(colonne)
            if bug == "sans_absents" and (_absent(a) or _absent(b)):
                continue
            paires.append((None if _absent(a) else a, None if _absent(b) else b))
        n = len(paires)
        lignes, cols, croise = Counter(p[0] for p in paires), Counter(p[1] for p in paires), Counter(paires)
        k = max(len(lignes), len(cols)) if bug == "k_max" else min(len(lignes), len(cols))
        if k <= 1 or n == 0:
            v2 = Fraction(0)
        else:
            if bug == "cases_observees":
                chi2 = Fraction(0)
                for (i, j), m in croise.items():
                    e = Fraction(lignes[i] * cols[j], n)
                    chi2 += (m - e) ** 2 / e
            else:
                chi2 = n * (sum(Fraction(m * m, lignes[i] * cols[j]) for (i, j), m in croise.items()) - 1)
            v2 = chi2 / n if bug == "phi" else chi2 / (n * (k - 1))
        v = float(v2) if bug == "sans_racine" else math.sqrt(v2)
        resultats.append((v2, Proxy(colonne, v, len(cols))))
    if bug == "croissant":
        resultats.sort(key=lambda r: r[0])
    elif bug != "sans_tri":
        resultats.sort(key=lambda r: -r[0])
    return [p for _, p in resultats]


_BUGS_PROXIES = {
    "cases_observees": "Votre χ² ne compte que les cases du tableau croisé où il y a quelqu'un (les clés de votre "
                       "Counter). Une case VIDE compte aussi : n_ij = 0 mais e_ij = a_i × b_j / n > 0, donc elle "
                       "ajoute e_ij au χ². Ou bien développez : χ² = n × (Σ n_ij² / (a_i × b_j) − 1), où les cases "
                       "vides sont dans le « − 1 ».",
    "sans_racine": "Vous renvoyez V² : le V de Cramér est la RACINE de χ² / (n × (k − 1)).",
    "k_max": "k est le plus PETIT des deux nombres de valeurs (min, pas max) : V = racine(χ² / (n × (min(r, c) − 1))).",
    "phi": "Vous oubliez de diviser par (k − 1) : racine(χ² / n) est le « phi », qui peut dépasser 1 quand les deux "
           "colonnes ont plus de deux valeurs. V = racine(χ² / (n × (k − 1))).",
    "sans_absents": "Des clients ont disparu : une valeur absente (clé manquante, None ou NaN) est une valeur comme une "
                    "autre (None). « Âge non renseigné » est une information, et l'oublier change le tableau croisé.",
    "croissant": "Triez du V le plus FORT au plus faible : les colonnes les plus suspectes d'abord.",
    "sans_tri": "Triez les Proxy du V le plus fort au plus faible (à égalité, dans l'ordre de `colonnes`).",
}


def _meme_proxy(obtenu, attendu: Proxy) -> bool:
    return (isinstance(obtenu, tuple) and len(obtenu) == 3 and obtenu[0] == attendu[0]
            and _proche(obtenu[1], attendu[1]) and type(obtenu[2]) is int and obtenu[2] == attendu[2])


def _memes_proxies(obtenu, attendu: list[Proxy]) -> bool:
    return (isinstance(obtenu, list) and len(obtenu) == len(attendu)
            and all(_meme_proxy(o, a) for o, a in zip(obtenu, attendu)))


def _ordre_ok(obtenu: list, attendu: list[Proxy], colonnes: list) -> bool:
    """Du V le plus fort au plus faible (à un arrondi près), et à égalité EXACTE de vos V, dans l'ordre de `colonnes`."""
    vrai = {p.colonne: p.v for p in attendu}
    rang = {c: i for i, c in enumerate(colonnes)}
    for avant, apres in zip(obtenu, obtenu[1:]):
        if vrai[avant[0]] < vrai[apres[0]] - 1e-12:
            return False
        if avant[1] == apres[1] and rang[avant[0]] > rang[apres[0]]:
            return False
    return True


_CONSEIL_RACINE = ("Si c'est « math domain error » : en float, un χ² nul peut sortir à −1e-16, et la racine d'un "
                   "négatif n'existe pas. Prenez max(χ², 0.0) avant la racine, ou calculez en Fraction.")


def _un_proxies(f, clients, protegee, colonnes, quoi, lignes_max: int | None = 2_000_000) -> None:
    attendu = _ref_proxies(clients, protegee, colonnes)
    avant = _cliche(clients) if len(clients) < 20_000 else None
    colonnes_avant = list(colonnes)
    obtenu = _appel(f, (clients, protegee, colonnes), quoi, _CONSEIL_RACINE, lignes_max=lignes_max,
                    conseil_lent="Votre fonction a été arrêtée : refaire une passe sur tous les clients pour chaque "
                                 "case du tableau croisé est bien trop lent. Comptez le tableau en une passe : "
                                 "`Counter(zip(valeurs_protegee, valeurs_colonne))`.")
    _pas_none(obtenu, quoi)
    if avant is not None:
        _intact(avant, clients, quoi, "les clients")
    _intact(colonnes_avant, list(colonnes), quoi, "les colonnes")
    if _memes_proxies(obtenu, attendu):
        return
    forme = isinstance(obtenu, list) and all(isinstance(p, tuple) and len(p) == 3 for p in obtenu)
    par_colonne = {p[0]: p for p in obtenu} if forme else {}
    valeurs_ok = (forme and len(obtenu) == len(attendu) and set(par_colonne) == {p.colonne for p in attendu}
                  and all(_meme_proxy(par_colonne[a.colonne], a) for a in attendu))
    if valeurs_ok and _ordre_ok(obtenu, attendu, list(colonnes)):
        return  # vos V ont un arrondi d'écart avec les nôtres, et votre tri est juste pour VOS valeurs
    for bug, conseil in _BUGS_PROXIES.items():
        if _memes_proxies(obtenu, _ref_proxies(clients, protegee, colonnes, bug)):
            raise AssertionError(f"{quoi} : {conseil}")
    if not forme:
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} ; attendu une liste de Proxy(colonne, v, modalites).")
    for a in attendu:
        o = par_colonne.get(a.colonne)
        if o is None:
            raise AssertionError(f"{quoi} : la colonne {a.colonne!r} manque (un Proxy par colonne demandée).")
        if not _meme_proxy(o, a):
            raise AssertionError(f"{quoi} : pour {a.colonne!r}, vous donnez {_court(tuple(o))} au lieu de {a!r}.")
    if len(obtenu) != len(attendu):
        raise AssertionError(f"{quoi} : {len(obtenu)} Proxy au lieu de {len(attendu)} (un par colonne demandée).")
    raise AssertionError(f"{quoi} : l'ordre est {[p[0] for p in obtenu]} au lieu de {[p.colonne for p in attendu]}. "
                         "Du V le plus fort au plus faible ; à égalité, dans l'ordre de `colonnes` (un tri stable).")


def verifier_proxies(f) -> None:
    nan = float("nan")

    def cl(**valeurs):
        return dict(valeurs)

    doc = ([cl(a=1, x=x, y=y) for x, y in (("u", "p"), ("u", "q"), ("u", "p"), ("v", "q"))]
           + [cl(a=2, x=x, y=y) for x, y in (("v", "p"), ("v", "q"), ("v", "p"), ("u", "q"))])
    vides = [cl(a=1, x="u"), cl(a=1, x="u"), cl(a=2, x="v"), cl(a=2, x="v"), cl(a=3, x="v")]
    grand = [cl(a=a, x=x, w=w) for a, x, w in ((1, "u", 0), (1, "u", 1), (1, "v", 2), (2, "v", 0), (2, "t", 1),
                                               (3, "t", 2), (3, "t", 0), (3, "u", 1), (4, "s", 2), (4, "s", 0),
                                               (2, "s", 1), (4, "u", 2))]
    absents = [cl(a=1, x="u"), cl(a=nan, x="u"), cl(x="v"), cl(a=2, x=None), cl(a=2, x=nan), cl(a=1, x="v"),
               cl(a=None, x="u"), cl(a=2, x="v")]
    cas = [
        ("l'exemple de la docstring", doc, "a", ["y", "x"]),
        ("un tableau croisé avec des cases vides", vides, "a", ["x"]),
        ("des colonnes à plusieurs valeurs", grand, "a", ["x", "w"]),
        ("une colonne à une seule valeur", [cl(a=i % 3, x="u", y=i % 2) for i in range(12)], "a", ["x", "y"]),
        ("des valeurs absentes", absents, "a", ["x"]),
        ("deux colonnes identiques", [{**c, "x2": c["x"]} for c in doc], "a", ["x2", "y", "x"]),
        ("aucune colonne", doc, "a", []),
    ]
    for nom, clients, protegee, colonnes in cas:
        _un_proxies(f, clients, protegee, colonnes, f"proxies(<{nom}>, {protegee!r}, {colonnes!r})")
    alea = random.Random(233)
    for n in range(20):
        clients = generer_clients(alea.randint(1, 500), graine=alea.randrange(10**6))
        protegee = alea.choice(("age", "zone", "contrat"))
        colonnes = [c for c in ("age", "zone", "contrat", "paiement", "internet", "appels") if c != protegee]
        alea.shuffle(colonnes)
        _un_proxies(f, clients, protegee, colonnes, f"proxies(generer_clients(…), {protegee!r}, {colonnes}) "
                                                    f"(tirage n° {n})")
    _un_proxies(f, clients_exemple(), "age", COLONNES, f"proxies(clients_exemple(), 'age', {COLONNES})")
    _un_proxies(f, _grands_clients()[:100_000], "age", COLONNES, "proxies(<100 000 clients>, 'age', COLONNES)",
                lignes_max=10_000_000)
    for args, conseil in (
            ((doc, "a", ["x", "a"]), "l'attribut protégé ne peut pas être son propre indice."),
            ((doc, "a", ["x", "x"]), "une colonne demandée deux fois."),
            (([], "a", ["x"]), "aucun client : rien à mesurer.")):
        _doit_lever(f, args, f"proxies(<{len(args[0])} clients>, {args[1]!r}, {args[2]!r})", conseil, ValueError)


# ============================================================================ défi 234 : testing


class _Espion:
    """Un modèle qui compte ses appels et vérifie qu'on ne lui envoie jamais un client modifié."""

    def __init__(self, fonction, originaux: list[dict]):
        self.fonction = fonction
        self.appels = 0
        self.modifies: list[dict] = []
        self.originaux = {id(c): (c, dict(c)) for c in originaux}

    def __call__(self, lot):
        self.appels += 1
        lot = list(lot)
        for client in lot:
            original = self.originaux.get(id(client))
            if original is not None and original[0] is client and client != original[1]:
                self.modifies.append(dict(client))
        return self.fonction(lot)


def _modele_test(lot: list[dict]) -> list[float]:
    """Un petit modèle lisible : la base du client, −0.2 à 65 ans et plus, +0.1 de 18 à 29 ans."""
    effet = {"65+": -0.2, "18-29": 0.1}
    return [round(min(max(c.get("base", 0.5) + effet.get(c.get("age"), 0.0) + 0.05 * (c.get("zone") == "Rurale"),
                          0.0), 1.0), 3) for c in lot]


def _ref_testing(fonction, clients: list[dict], colonne: str, valeurs, seuil: float = SEUIL,
                 bug: str = "") -> list[Bascule]:
    """La bonne réponse ; `bug` rejoue une erreur classique."""
    valeurs = list(valeurs)
    testes = [c for c in clients if _cle(c, colonne) in valeurs]
    origines = [_cle(c, colonne) for c in testes]
    if bug == "score_stocke":
        avant = [c["score"] >= seuil for c in testes]
    else:
        avant = [s >= seuil for s in fonction(testes)] if testes else []
    avec_soi = bug == "avec_soi"
    comptes = {(o, t): [0, 0, 0] for o in valeurs for t in valeurs if o != t or avec_soi}
    for t in valeurs:
        positions = [i for i, o in enumerate(origines) if o != t or avec_soi]
        copies = [{**testes[i], colonne: t} for i in positions]
        scores = fonction(copies) if copies else []
        for j, (i, score) in enumerate(zip(positions, scores)):
            initial = avant[j] if bug == "decalage" else avant[i]
            apres = score >= seuil
            c = comptes[origines[i], t]
            c[0] += 1
            gagne, perd = apres and not initial, initial and not apres
            if bug == "sens":
                gagne, perd = perd, gagne
            c[1] += gagne
            c[2] += perd
    return [Bascule(o, t, *c) for (o, t), c in comptes.items()]


_BUGS_TESTING = {
    "score_stocke": "Vous lisez la décision d'origine dans le champ \"score\" du client. Il peut dater d'une ancienne "
                    "version du modèle : recalculez-la avec `modele(clients testés)`, comme pour les copies (sinon, "
                    "vous comparez deux modèles différents, pas deux âges).",
    "decalage": "Vos scores sont décalés : le lot d'une valeur testée ne contient pas tous les clients (ceux qui ont "
                "déjà cette valeur n'y sont pas), donc la j-ième copie n'est pas le j-ième client testé. Gardez la "
                "POSITION de chaque client envoyé (`zip(positions, scores)`).",
    "sens": "Vous avez échangé `gagnent` et `perdent` : gagnent = pas d'offre avec la vraie valeur, l'offre avec la "
            "valeur testée.",
    "avec_soi": "Ne testez pas une valeur contre elle-même : une Bascule par paire (origine, testee) avec origine "
                "DIFFÉRENTE de testee.",
}


def _memes_bascules(obtenu, attendu: list[Bascule]) -> bool:
    return (isinstance(obtenu, list) and len(obtenu) == len(attendu)
            and all(isinstance(o, tuple) and tuple(o) == tuple(a) and all(type(x) is int for x in tuple(o)[2:])
                    for o, a in zip(obtenu, attendu)))


def _un_testing(f, fonction, clients, colonne, valeurs, seuil, quoi) -> None:
    attendu = _ref_testing(fonction, clients, colonne, valeurs, seuil)
    avant = _cliche(clients)
    valeurs_avant = list(valeurs)
    espion = _Espion(fonction, clients)
    obtenu = _appel(f, (espion, clients, colonne, valeurs, seuil), quoi)
    _pas_none(obtenu, quoi)
    if espion.modifies:
        raise AssertionError(f"{quoi} : le modèle a reçu un client ORIGINAL modifié ({_court(espion.modifies[0], 120)}). "
                             "Même si vous remettez l'ancienne valeur après l'appel, une exception au milieu (le "
                             "modèle est une API, elle peut tomber) laisserait le client faux pour toujours. Envoyez "
                             "des COPIES : `{**client, colonne: valeur}`.")
    _intact(avant, clients, quoi, "les clients")
    _intact(valeurs_avant, list(valeurs), quoi, "les valeurs")
    maximum = len(valeurs) + 1
    if espion.appels > maximum:
        raise AssertionError(f"{quoi} appelle le modèle {espion.appels} fois pour {len(clients)} clients et "
                             f"{len(valeurs)} valeurs ; au plus {maximum} (un lot pour les originaux, puis un lot par "
                             "valeur testée). Le modèle est derrière une API : un appel par client, c'est un aller-"
                             "retour réseau par client.")
    if _memes_bascules(obtenu, attendu):
        return
    for bug, conseil in _BUGS_TESTING.items():
        if _memes_bascules(obtenu, _ref_testing(fonction, clients, colonne, valeurs, seuil, bug)):
            raise AssertionError(f"{quoi} : {conseil}")
    if not isinstance(obtenu, list):
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} ; attendu une liste de Bascule.")
    if [tuple(b)[:2] for b in obtenu if isinstance(b, tuple)] != [b[:2] for b in attendu]:
        raise AssertionError(f"{quoi} : les paires (origine, testee) sont {_court([tuple(b)[:2] for b in obtenu], 140)} "
                             f"au lieu de {_court([b[:2] for b in attendu], 140)} (origine dans l'ordre de `valeurs`, "
                             "puis testee dans l'ordre de `valeurs`, sans origine == testee).")
    for o, a in zip(obtenu, attendu):
        if not isinstance(o, tuple) or tuple(o) != tuple(a):
            raise AssertionError(f"{quoi} : vous donnez {_court(o)} au lieu de {a!r}.")
    raise AssertionError(f"{quoi} : testes, gagnent et perdent sont des int.")


def _client_test(age, base, score=None, **autres) -> dict:
    client = {"age": age, "base": base, **autres}
    client["score"] = _modele_test([client])[0] if score is None else score
    return client


def verifier_testing(f) -> None:
    t = _client_test
    cas = [
        ("l'exemple de la docstring", modele,
         [{"age": "65+", "contrat": "Mensuel", "appels": 2, "score": 0.401},
          {"age": "30-44", "contrat": "Mensuel", "appels": 2, "score": 0.574},
          {"age": None, "contrat": "Annuel", "score": 0.109}], ["30-44", "65+"]),
        ("des clients près du seuil", _modele_test,
         [t("65+", 0.65), t("30-44", 0.55), t("18-29", 0.45), t("65+", 0.75), t("30-44", 0.45), t("18-29", 0.35),
          t("45-64", 0.5)], ["18-29", "30-44", "65+"]),
        ("un champ \"score\" d'un ancien modèle", _modele_test,
         [t("65+", 0.6, score=0.9), t("30-44", 0.45, score=0.1), t("18-29", 0.42, score=0.8), t("65+", 0.68, score=0.2)],
         ["18-29", "30-44", "65+"]),
        ("des âges absents ou hors des valeurs testées", _modele_test,
         [t(None, 0.45), t(float("nan"), 0.6), t("45-64", 0.55), {"base": 0.5, "score": 0.5}, t("65+", 0.62),
          t("18-29", 0.41)], ["65+", "18-29"]),
        ("personne à tester", _modele_test, [t("45-64", 0.55), t(None, 0.6)], ["65+", "18-29"]),
        ("aucun client", _modele_test, [], ["65+", "18-29"]),
    ]
    for nom, fonction, clients, valeurs in cas:
        _un_testing(f, fonction, clients, "age", valeurs, SEUIL, f"testing(modele, <{nom}>, 'age', {valeurs})")
    alea = random.Random(234)
    for n in range(20):
        clients = [t(alea.choice(AGES + (None,)), round(alea.uniform(0.2, 0.8), 2),
                     zone=alea.choice(("Urbaine", "Rurale"))) for _ in range(alea.randint(1, 60))]
        colonne, valeurs = alea.choice((("age", alea.sample(AGES, alea.randint(2, 4))),
                                        ("zone", ["Urbaine", "Rurale"]), ("zone", ["Rurale", "Urbaine"])))
        seuil = alea.choice((0.5, 0.4, 0.6))
        _un_testing(f, _modele_test, clients, colonne, valeurs, seuil,
                    f"testing(modele, <tirage n° {n}>, {colonne!r}, {valeurs}, {seuil})")
    exemple = clients_exemple()
    _un_testing(f, modele, exemple, "age", list(AGES), SEUIL, "testing(modele, clients_exemple(), 'age', AGES)")
    _un_testing(f, modele, exemple[:2000], "age", ["65+", "30-44"], 0.4,
                "testing(modele, clients_exemple()[:2000], 'age', ['65+', '30-44'], 0.4)")
    deux = [t("65+", 0.6), t("30-44", 0.6)]

    def bavard(lot):
        return _modele_test(lot)[:-1] if lot else []

    for args, attendu, conseil in (
            ((_modele_test, deux, "age", ["65+"]), ValueError, "il faut au moins deux valeurs à comparer."),
            ((_modele_test, deux, "age", ["65+", "30-44", "65+"]), ValueError, "une valeur demandée deux fois."),
            ((bavard, deux, "age", ["65+", "30-44"]), ValueError,
             "le modèle renvoie un score de moins que de clients envoyés : impossible de savoir lequel manque."),
            ((_modele_test, deux, "age", ["65+", "30-44"], 1.5), ValueError, "le seuil est entre 0 et 1."),
            ((_modele_test, deux, "age", ["65+", "30-44"], "0.5"), TypeError, "un texte n'est pas un seuil.")):
        _doit_lever(f, args, f"testing({getattr(args[0], '__name__', 'modele')}, <2 clients>, "
                             f"{', '.join(map(repr, args[2:]))})", conseil, attendu)


# ============================================================================ défi 235 : seuils_egalite_chances


def _ref_seuils(clients: list[dict], colonne: str, cible, bug: str = "") -> dict:
    """La bonne réponse ; `bug` rejoue une erreur classique."""
    exacte = Fraction(str(cible)) if isinstance(cible, float) else Fraction(cible)
    scores: dict = {}
    for client in clients:
        valeur = client.get(colonne)
        if bug == "sans_absents" and _absent(valeur):
            continue
        cle = None if _absent(valeur) else valeur
        scores.setdefault(cle, ([], []))[0 if client["parti"] == 1 else 1].append(client["score"])

    def k_de(p: int) -> int:
        if bug == "float":
            return math.ceil(float(cible) * p)
        if bug == "fraction_float":
            return math.ceil(Fraction(float(cible)) * p)
        return -(-exacte.numerator * p // exacte.denominator)

    def seuil_de(partis: list) -> float:
        tries = sorted(partis, reverse=True)
        if bug == "division_float":  # vp / P en float, comparé à la cible telle quelle (une Fraction, par exemple)
            for i, score in enumerate(tries):
                if (i + 1 == len(tries) or tries[i + 1] != score) and (i + 1) / len(tries) >= cible:
                    return score
        k = k_de(len(tries))
        if bug == "decale" and k < len(tries):
            return tries[k]
        return tries[k - 1]

    tous = [s for partis, _ in scores.values() for s in partis]
    reglages = {}
    for groupe, (partis, restes) in scores.items():
        if not partis:
            reglages[groupe] = None
            continue
        seuil = seuil_de(tous if bug == "global" else partis)
        if bug == "strict":
            vp, fp = sum(s > seuil for s in partis), sum(s > seuil for s in restes)
        else:
            vp, fp = sum(s >= seuil for s in partis), sum(s >= seuil for s in restes)
        n = len(partis) + len(restes)
        if bug == "tfp_effectif":
            tfp = fp / n
        else:
            tfp = fp / len(restes) if restes else None
        reglages[groupe] = Reglage(seuil, vp / len(partis), tfp, (vp + fp) / n)
    return reglages


_BUGS_SEUILS = {
    "float": "Vous comptez en float : 0.55 × 100 vaut 55.00000000000001, et `math.ceil` en fait 56. Il faut 55 partis "
             "sur 100, pas 56 : votre seuil est un cran trop bas. Lisez la cible telle qu'elle est écrite, "
             "`Fraction(str(cible))`, et calculez k en entiers : `-(-numerateur * P // denominateur)`.",
    "fraction_float": "`Fraction(cible)` sur un float donne le float tel qu'il est stocké : Fraction(0.1) vaut "
                      "0.1000000000000000055511151231257827…, un peu PLUS que 1/10, et ceil(Fraction(0.1) × 20) vaut 3 "
                      "au lieu de 2. Passez par le texte : `Fraction(str(cible))` vaut exactement 1/10.",
    "division_float": "Vous comparez vp / P, un float, à la cible : avec une cible Fraction(2, 3), 2 / 3 vaut "
                      "0.6666666666666666 en float, un peu MOINS que 2/3, et 2 partis sur 3 ne « suffisent » plus. "
                      "Comparez en entiers (vp × dénominateur >= numérateur × P) ou en fractions (Fraction(vp, P)).",
    "decale": "Votre seuil est un cran trop bas : le k-ième plus haut score est `tries[k - 1]` (les listes commencent "
              "à 0), pas `tries[k]`.",
    "strict": "Avec le seuil t, un client est sélectionné si score >= t (pas > t) : les partis qui ont exactement le "
              "score du seuil sont repérés.",
    "global": "Vous calculez UN seul seuil pour tout le monde. L'égalité des chances demande un seuil PAR GROUPE : "
              "pour chacun, le plus haut seuil qui repère au moins `cible` de SES partis.",
    "tfp_effectif": "Votre taux de faux positifs divise par l'effectif du groupe. Il se calcule parmi ceux qui sont "
                    "RESTÉS : restés sélectionnés / restés.",
    "sans_absents": "Des clients ont disparu : une valeur absente (clé manquante, None ou NaN) forme le groupe None.",
}


def _meme_reglage(obtenu, attendu) -> bool:
    if attendu is None:
        return obtenu is None
    return (isinstance(obtenu, tuple) and len(obtenu) == 4
            and all(_proche(o, a) for o, a in zip(obtenu, attendu)))


def _un_seuils(f, clients, colonne, cible, quoi, lignes_max: int | None = 2_000_000) -> None:
    attendu = _ref_seuils(clients, colonne, cible)
    avant = _cliche(clients) if len(clients) < 20_000 else None
    obtenu = _appel(f, (clients, colonne, cible), quoi, lignes_max=lignes_max,
                    conseil_lent="Votre fonction a été arrêtée : essayer chaque seuil possible et recompter les partis "
                                 "à chaque fois fait P × P opérations. Triez les scores des partis UNE fois : le seuil "
                                 "est le k-ième plus haut, avec k = ceil(cible × P) calculé en entiers.")
    _pas_none(obtenu, quoi)
    if avant is not None:
        _intact(avant, clients, quoi, "les clients")
    if not isinstance(obtenu, dict):
        raise AssertionError(f"{quoi} renvoie {type(obtenu).__name__} ; attendu un dict {{groupe: Reglage(...)}}.")
    if _memes_dicts(obtenu, attendu, _meme_reglage):
        return
    for bug, conseil in _BUGS_SEUILS.items():
        if _memes_dicts(obtenu, _ref_seuils(clients, colonne, cible, bug), _meme_reglage):
            raise AssertionError(f"{quoi} : {conseil}")
    if list(obtenu) != list(attendu):
        raise AssertionError(f"{quoi} donne les groupes {_court(list(obtenu), 100)} au lieu de "
                             f"{_court(list(attendu), 100)} (dans l'ordre d'apparition, les absents dans None).")
    for groupe, a in attendu.items():
        if not _meme_reglage(obtenu[groupe], a):
            raise AssertionError(f"{quoi} : pour le groupe {groupe!r}, vous donnez {_court(obtenu[groupe], 140)} au "
                                 f"lieu de {a!r}.")


def _p(g, score, parti=1) -> dict:
    return {"g": g, "score": score, "parti": parti}


def verifier_seuils_egalite_chances(f) -> None:
    doc = ([_p("A", s / 1000) for s in range(100, 200)] + [_p("A", 0.15, 0), _p("B", 0.3, 0)])
    cas = [
        ("un petit groupe", [_p("A", 0.9), _p("A", 0.7), _p("A", 0.4), _p("A", 0.2), _p("A", 0.8, 0), _p("A", 0.3, 0)],
         0.5),
        ("l'exemple de la docstring", doc, 0.55),
        ("une cible de 10 % sur 20 partis", [_p("A", s / 100) for s in range(20, 40)] + [_p("A", 0.385, 0)], 0.1),
        ("des ex aequo au seuil", [_p("A", s) for s in (0.9, 0.8, 0.8, 0.8, 0.5, 0.2)] + [_p("A", 0.8, 0)], 0.4),
        ("deux groupes très différents", [_p("A", s / 10) for s in range(1, 10)] + [_p("B", s / 100) for s in range(1, 9)]
         + [_p("A", 0.55, 0), _p("B", 0.05, 0), _p("B", 0.01, 0)], 0.5),
        ("un groupe sans resté, un sans parti", [_p("A", 0.3), _p("A", 0.6), _p("B", 0.9, 0)], 0.5),
        ("une cible de 100 %", [_p("A", 0.7), _p("A", 0.2), _p("A", 0.1, 0), _p("A", 0.25, 0)], 1),
        ("une cible en fraction", [_p("A", 0.7), _p("A", 0.5), _p("A", 0.3), _p("A", 0.6, 0)], Fraction(2, 3)),
        ("des groupes absents", [_p(None, 0.4), _p(float("nan"), 0.6), {"score": 0.5, "parti": 1}, _p("A", 0.7),
                                 _p(float("nan"), 0.45, 0)], 0.5),
        ("aucun client", [], 0.5),
    ]
    for nom, clients, cible in cas:
        _un_seuils(f, clients, "g", cible, f"seuils_egalite_chances(<{nom}>, 'g', {cible!r})")
    alea = random.Random(235)
    for n in range(30):
        clients = generer_clients(alea.randint(1, 500), graine=alea.randrange(10**6))
        colonne = alea.choice(("age", "zone", "contrat"))
        cible = alea.choice((0.55, 0.6, 0.75, 0.9, 0.1, 0.2, 1, round(alea.uniform(0.05, 1), 2)))
        _un_seuils(f, clients, colonne, cible, f"seuils_egalite_chances(generer_clients(…), {colonne!r}, {cible}) "
                                               f"(tirage n° {n})")
    for cible in (0.6, 0.55, 0.9):
        _un_seuils(f, clients_exemple(), "age", cible, f"seuils_egalite_chances(clients_exemple(), 'age', {cible})")
    alea = random.Random(2350)
    grand = [_p("A", round(alea.random(), 4)) for _ in range(60_000)] + [_p("A", round(alea.random(), 4), 0)
                                                                         for _ in range(20_000)]
    _un_seuils(f, grand, "g", 0.55, "seuils_egalite_chances(<un groupe de 60 000 partis>, 'g', 0.55)",
               lignes_max=5_000_000)
    base = [_p("A", 0.6), _p("A", 0.3, 0)]
    for clients, cible, attendu, conseil in (
            (base, 0, ValueError, "une cible de 0 ne demande rien : elle est dans ]0, 1]."),
            (base, 1.2, ValueError, "on ne repère pas 120 % des partis."),
            (base, -0.5, ValueError, "une cible négative ne veut rien dire."),
            (base, float("nan"), ValueError, "une cible NaN ne veut rien dire."),
            (base, float("inf"), ValueError, "une cible infinie ne veut rien dire."),
            (base, "0.6", TypeError, "un texte n'est pas une cible."),
            (base, True, TypeError, "True n'est pas une cible (même si True == 1)."),
            ([_p("A", 1.5)], 0.5, ValueError, "un score est entre 0 et 1."),
            ([_p("A", 0.5, parti=3)], 0.5, ValueError, "parti vaut 0 ou 1.")):
        _doit_lever(f, (clients, "g", cible), f"seuils_egalite_chances({_court(clients, 70)}, 'g', {cible!r})",
                    conseil, attendu)


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"taux_par_groupe": 230, "quatre_cinquiemes": 231, "taux_standardises": 232, "proxies": 233,
            "testing": 234, "seuils_egalite_chances": 235}

SERIE = Serie(
    code="38",
    titre="Traiter tout le monde de la même façon : taux par groupe, règle des 4/5, Simpson, proxies, testing, "
          "égalité des chances",
    module_eleve="defis.serie_38",
    module_corrige="defis.corrige_serie_38",
    defis=(
        Defi(230, "taux_par_groupe", "🟢", "—", verifier_taux_par_groupe),
        Defi(231, "quatre_cinquiemes", "🟢", "—", verifier_quatre_cinquiemes),
        Defi(232, "taux_standardises", "🟡", "—", verifier_taux_standardises),
        Defi(233, "proxies", "🟡", "—", verifier_proxies),
        Defi(234, "testing", "🟡", "—", verifier_testing),
        Defi(235, "seuils_egalite_chances", "🔴", "—", verifier_seuils_egalite_chances),
    ),
)

"""Vérifications automatiques de la série 35 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est Yasmine, qui prépare la campagne de décembre et doit convaincre la directrice qu'appeler
les plus forts scores de churn est une mauvaise idée, et le serveur qui compte ce que chaque calcul lui coûte :

- aux défis 213 et 215, il compte les lignes exécutées dans votre fichier sur 20 000 clients : recompter
  l'historique pour chaque nouveau client, ou les k premiers pour chaque k, se voit quelle que soit la machine ;
- au défi 213, les nouveaux clients n'ont ni `offre` ni `parti` : les lire se voit tout de suite ;
- au défi 217, il vous donne un modèle ESPION : il note avec quels clients vous l'entraînez et ce que vous
  lui montrez des clients à prédire. Un client prédit par un modèle qui l'a vu, une réponse laissée dans les
  clients à prédire, des plis en blocs au lieu de i % k, des prédictions rangées pli par pli se voient ;
- partout, il rejoue les erreurs classiques (uplift pris dans le mauvais sens, taux calculé sur tout le
  segment, zéro inventé pour un groupe vide, a priori commun aux deux groupes, risque renvoyé au lieu de
  l'effet, p = 0.5 pour tout le monde, ex aequo séparés, courbe d'uplift au lieu de Qini, aire sans la droite
  du hasard, plan par le score, chiens qui dorment appelés pour remplir le budget, gain brut…) pour nommer la
  vôtre ;
- et il vérifie que vos fonctions ne modifient pas ce qu'elles reçoivent.

Les taux et les gains se comparent à 1e-9 près : il ne vous reproche jamais un arrondi.
"""

from __future__ import annotations

import copy
import math
import numbers
import random
import sys

from .ciblage import (VARIABLES, campagne_exemple, campagne_stratifiee_exemple, generer_campagne, prospects_exemple,
                      uplift_vrai)
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
_REPONSES = ("offre", "parti", "p_offre")


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = 2_000_000, conseil_lent: str = "",
           kwargs: dict | None = None, nouveaux: bool = False):
    """f(*args, **kwargs), en transformant toute erreur en message lisible."""
    kwargs = kwargs or {}
    try:
        return _trace(f, lambda: f(*args, **kwargs), lignes_max)
    except NotImplementedError as exc:
        _pas_encore(f, exc)
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except KeyError as exc:
        if nouveaux and exc.args and exc.args[0] in ("offre", "parti"):
            raise AssertionError(f"{quoi} lit `{exc.args[0]}` chez un NOUVEAU client, qui n'en a pas : il n'a encore été "
                                 "ni appelé ni observé. Ne lisez que ses variables (`client[v] for v in variables`) ; "
                                 "`offre` et `parti` ne se lisent que dans l'historique.") from None
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None


def _pas_encore(f, exc: NotImplementedError) -> None:
    """Votre fonction est écrite, mais elle appelle une fonction qui ne l'est pas encore."""
    cadres = []
    tb = exc.__traceback__
    while tb is not None:
        cadres.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    nom = getattr(f, "__name__", "")
    if nom and nom in cadres and cadres[-1] != nom:
        numero = _NUMEROS.get(cadres[-1])
        defi = f" (défi {numero})" if numero else ""
        raise AssertionError(f"`{nom}` appelle `{cadres[-1]}`, qui n'est pas encore écrite (raise "
                             f"NotImplementedError) : faites d'abord ce défi-là{defi}.") from None


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu=ValueError, kwargs: dict | None = None) -> None:
    """Exige que f(*args, **kwargs) lève `attendu` (une classe d'exception, ou un tuple de classes)."""
    noms = " ou ".join(c.__name__ for c in attendu) if isinstance(attendu, tuple) else attendu.__name__
    try:
        resultat = _trace(f, lambda: f(*args, **(kwargs or {})), 2_000_000)
    except NotImplementedError as exc:
        _pas_encore(f, exc)
        raise
    except attendu:
        return
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {_CONSEIL_LENT}") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {noms}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {noms}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _intact(avant, apres, quoi: str, nom: str) -> None:
    if not _egaux_nan(avant, apres):
        raise AssertionError(f"{quoi} a modifié {nom} reçu(s) (avant : {_court(avant, 100)}, après : "
                             f"{_court(apres, 100)}). Une fonction ne touche pas à ce qu'on lui prête : travaillez sur "
                             "une copie, ou construisez un nouvel objet.")


def _egaux_nan(a, b) -> bool:
    """a == b, en considérant deux NaN comme égaux (dans des listes et des dicts imbriqués)."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_egaux_nan(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_egaux_nan(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return type(a) is type(b) and a == b


def _pas_none(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")


def _nombre(x) -> bool:
    """Un nombre réel, pas un bool (les entiers et flottants de NumPy et pandas sont acceptés)."""
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _proche(x, y, tol: float = 1e-9) -> bool:
    return _nombre(x) and _nombre(y) and math.isclose(x, y, rel_tol=tol, abs_tol=tol)


def _meme(x, y) -> bool:
    """Deux valeurs égales : None et None, ou deux nombres à 1e-9 près."""
    return (x is None and y is None) or _proche(x, y)


def _memes(r, attendu) -> bool:
    try:
        return len(r) == len(attendu) and all(_meme(x, y) for x, y in zip(r, attendu))
    except TypeError:
        return False


def _fmt(x) -> str:
    return f"{x:.6g}" if _nombre(x) else repr(x)


def _campagne(rng: random.Random, n: int, stratifiee: bool = False, booleens: bool = False,
              manquants: bool = False) -> list[dict]:
    """Une petite campagne tirée au hasard ; parfois offre/parti en bool, parfois des variables manquantes."""
    clients = generer_campagne(n, graine=rng.randint(0, 10**6), p_offre=rng.choice([0.2, 0.5, 0.8]),
                               stratifiee=stratifiee, prefixe=rng.choice("CXK"))
    for c in clients:
        if booleens and rng.random() < 0.5:
            c["offre"], c["parti"] = bool(c["offre"]), bool(c["parti"])
        if manquants and rng.random() < 0.15:
            c[rng.choice(VARIABLES)] = None
    return clients


def _sans_reponse(clients: list[dict]) -> list[dict]:
    return [{k: v for k, v in c.items() if k not in _REPONSES} for c in clients]


_ERREURS_SAISIE = (2, -1, "oui", "1", None, 0.5)

# ---------------------------------------------------------------------------- défi 212 : effet par segment


def _ref_effets(clients, variable: str, variante: str = "") -> dict:
    compteurs: dict = {}
    for c in clients:
        x = compteurs.setdefault(c[variable], [0, 0, 0, 0])
        offre = int(c["offre"]) if variante != "groupes_inverses" else 1 - int(c["offre"])
        if offre:
            x[0] += 1
            x[1] += int(c["parti"])
        else:
            x[2] += 1
            x[3] += int(c["parti"])
    tous_temoins = sum(x[2] for x in compteurs.values())
    tous_departs_temoins = sum(x[3] for x in compteurs.values())
    effets = {}
    for valeur, (n_o, d_o, n_t, d_t) in compteurs.items():
        if variante == "taux_segment":
            taux_o = d_o / (n_o + n_t) if n_o else None
            taux_t = d_t / (n_o + n_t) if n_t else None
        elif variante == "zero":
            taux_o = d_o / n_o if n_o else 0.0
            taux_t = d_t / n_t if n_t else 0.0
        else:
            taux_o = d_o / n_o if n_o else None
            taux_t = d_t / n_t if n_t else None
        if variante == "temoin_global":
            taux_t = tous_departs_temoins / tous_temoins if tous_temoins else None
        if taux_o is None or taux_t is None:
            uplift = None
        elif variante == "signe":
            uplift = taux_o - taux_t
        elif variante == "relatif":
            uplift = (taux_t - taux_o) / taux_t if taux_t else None
        else:
            uplift = taux_t - taux_o
        effets[valeur] = (n_o, n_t, taux_o, taux_t, uplift)
    return effets


_VARIANTES_EFFETS = [
    ("signe", "l'uplift, ce sont les départs ÉVITÉS : taux_temoin − taux_offre (sans l'offre, moins avec). Vous "
              "calculez l'inverse : une offre qui retient paraîtrait faire partir."),
    ("taux_segment", "chaque taux se calcule dans SON groupe : départs des appelés / n_offre, départs des témoins / "
                     "n_temoin. Vous divisez par tout le segment (n_offre + n_temoin)."),
    ("temoin_global", "le taux témoin d'un segment se calcule avec les témoins DE CE SEGMENT, pas avec tous les témoins "
                      "de la campagne : sinon on compare des mensuels appelés à un témoin moyen."),
    ("zero", "un groupe vide n'a pas de taux : None (et l'uplift aussi), pas 0.0. Zéro voudrait dire « personne n'est "
             "parti », ce que personne n'a observé."),
    ("relatif", "l'uplift est une DIFFÉRENCE de taux (taux_temoin − taux_offre), pas une variation relative."),
    ("groupes_inverses", "offre == 1, ce sont les clients APPELÉS ; offre == 0, le groupe témoin. Vous avez inversé "
                         "les deux groupes."),
]

_NOMS_EFFET = ("n_offre", "n_temoin", "taux_offre", "taux_temoin", "uplift")


def _un_effet(f, clients, variable: str, quoi: str) -> None:
    avant = copy.deepcopy(clients)
    r = _appel(f, (clients, variable), quoi)
    _intact(avant, clients, quoi, "les clients")
    _pas_none(r, quoi)
    attendu = _ref_effets(clients, variable)
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : un dict {{valeur du segment: Effet(...)}} est "
                             "attendu.")
    if set(r) != set(attendu):
        manque, trop = [k for k in attendu if k not in r], [k for k in r if k not in attendu]
        if None in manque:
            raise AssertionError(f"{quoi} n'a pas de segment None : une valeur manquante forme un segment comme un autre "
                                 "(ses clients ne disparaissent pas du calcul).")
        raise AssertionError(f"{quoi} : segments {_court(list(r), 100)} au lieu de {_court(list(attendu), 100)}"
                             f"{' (il manque ' + _court(manque, 60) + ')' if manque else ''}"
                             f"{' (en trop : ' + _court(trop, 60) + ')' if trop else ''}.")
    if list(r) != list(attendu):
        raise AssertionError(f"{quoi} range les segments dans l'ordre {_court(list(r), 100)} au lieu de "
                             f"{_court(list(attendu), 100)} : l'ordre où chaque segment APPARAÎT pour la première fois "
                             "dans les clients (un dict garde l'ordre d'insertion ; ne le triez pas).")
    for valeur, e in r.items():
        if not isinstance(e, tuple) or len(e) != 5:
            raise AssertionError(f"Dans {quoi}, le segment {valeur!r} vaut {_court(e, 100)} : un Effet(n_offre, n_temoin, "
                                 "taux_offre, taux_temoin, uplift) (ou un tuple de 5 valeurs dans cet ordre) est attendu.")
    if all(_memes(r[v], attendu[v]) for v in attendu):
        return
    for variante, message in _VARIANTES_EFFETS:
        autre = _ref_effets(clients, variable, variante)
        if all(_memes(r[v], autre[v]) for v in attendu) and not all(_memes(autre[v], attendu[v]) for v in attendu):
            valeur = next(v for v in attendu if not _memes(autre[v], attendu[v]))
            raise AssertionError(f"Dans {quoi}, le segment {valeur!r} vaut {_court(tuple(r[valeur]), 120)} au lieu de "
                                 f"{_court(attendu[valeur], 120)} : {message}")
    valeur = next(v for v in attendu if not _memes(r[v], attendu[v]))
    i = next(i for i in range(5) if not _meme(r[valeur][i], attendu[valeur][i]))
    raise AssertionError(f"Dans {quoi}, pour le segment {valeur!r}, {_NOMS_EFFET[i]} vaut {_fmt(r[valeur][i])} au lieu "
                         f"de {_fmt(attendu[valeur][i])}.")


def verifier_effet_par_segment(f) -> None:
    doc = [{"contrat": "mensuel", "offre": 1, "parti": 0}, {"contrat": "mensuel", "offre": 1, "parti": 1},
           {"contrat": "mensuel", "offre": 0, "parti": 1}, {"contrat": "annuel", "offre": 1, "parti": 0}]
    _un_effet(f, doc, "contrat", "effet_par_segment(l'exemple de la docstring, \"contrat\")")
    campagne = campagne_exemple()
    for variable in VARIABLES + ("p_offre",):
        _un_effet(f, campagne, variable, f"effet_par_segment(campagne_exemple(), {variable!r})")
    petits = [
        [],
        [{"s": "a", "offre": 0, "parti": 1}, {"s": "a", "offre": 0, "parti": 0}],         # aucun appelé
        [{"s": None, "offre": 1, "parti": 1}, {"s": "b", "offre": 0, "parti": 0}, {"s": None, "offre": 0, "parti": 0}],
        [{"s": 1, "offre": True, "parti": False}, {"s": 1, "offre": False, "parti": True}],
        [{"s": "b", "offre": 1, "parti": 0}, {"s": "a", "offre": 0, "parti": 1}, {"s": "b", "offre": 0, "parti": 1},
         {"s": "a", "offre": 1, "parti": 1}, {"s": "c", "offre": 1, "parti": 0}],
    ]
    for clients in petits:
        _un_effet(f, clients, "s", f"effet_par_segment({_court(clients, 90)}, 's')")
    for cle in ("offre", "parti"):
        for mauvais in _ERREURS_SAISIE:
            clients = [{"s": "a", "offre": 1, "parti": 0}, {"s": "a", "offre": 0, "parti": 1}]
            clients[1][cle] = mauvais
            _doit_lever(f, (clients, "s"), f"effet_par_segment(…, 's') avec {cle} = {mauvais!r}",
                        "`offre` et `parti` valent 0 ou 1 (True et False acceptés) : toute autre valeur est une erreur "
                        "de saisie → ValueError (ne la comptez ni comme un départ, ni comme un témoin).")
    alea = random.Random(212)
    for essai in range(150):
        clients = _campagne(alea, alea.randint(1, 60), stratifiee=alea.random() < 0.3,
                            booleens=alea.random() < 0.3, manquants=alea.random() < 0.3)
        variable = alea.choice(VARIABLES)
        _un_effet(f, clients, variable, f"effet_par_segment(<{len(clients)} clients tirés au hasard>, {variable!r}) "
                                        f"(n°{essai})")


# ---------------------------------------------------------------------------- défi 213 : T-learner


def _ref_t_learner(historique, nouveaux, variables, m, variante: str = "") -> list[float]:
    variables = tuple(variables)
    n = {0: {}, 1: {}}
    d = {0: {}, 1: {}}
    tn, td = [0, 0], [0, 0]
    for c in historique:
        g, y = int(c["offre"]), int(c["parti"])
        cell = tuple(c[v] for v in variables)
        n[g][cell] = n[g].get(cell, 0) + 1
        d[g][cell] = d[g].get(cell, 0) + y
        tn[g] += 1
        td[g] += y
    prior = [td[0] / tn[0], td[1] / tn[1]]
    if variante == "a_priori_commun":
        prior = [(td[0] + td[1]) / (tn[0] + tn[1])] * 2
    elif variante == "a_priori_inverse":
        prior = [prior[1], prior[0]]

    def risque(g, cell):
        ng, dg = n[g].get(cell, 0), d[g].get(cell, 0)
        if variante == "sans_a_priori":
            return dg / (ng + m)
        if variante == "brut":
            return dg / ng if ng else prior[g]
        return (dg + m * prior[g]) / (ng + m)

    def risque_global(cell):
        ng = n[0].get(cell, 0) + n[1].get(cell, 0)
        dg = d[0].get(cell, 0) + d[1].get(cell, 0)
        return (dg + m * (td[0] + td[1]) / (tn[0] + tn[1])) / (ng + m)

    resultat = []
    for c in nouveaux:
        cell = tuple(c[v] for v in variables)
        if variante == "signe":
            resultat.append(risque(1, cell) - risque(0, cell))
        elif variante == "risque_temoin":
            resultat.append(risque(0, cell))
        elif variante == "risque_global":
            resultat.append(risque_global(cell))
        else:
            resultat.append(risque(0, cell) - risque(1, cell))
    return resultat


_VARIANTES_T = [
    ("signe", "l'uplift, c'est le risque SANS offre moins le risque AVEC (des départs évités). Vous calculez "
              "l'inverse : les persuadables passeraient en bas de la liste."),
    ("a_priori_commun", "chaque groupe est tiré vers SON taux global : P_offre pour le modèle des appelés, P_temoin "
                        "pour celui des témoins. Avec un a priori commun, l'effet moyen de l'offre s'efface dans "
                        "toutes les petites cellules."),
    ("a_priori_inverse", "le modèle des appelés est tiré vers le taux des APPELÉS, celui des témoins vers le taux des "
                         "TÉMOINS : vous avez croisé les deux a priori."),
    ("sans_a_priori", "le lissage ajoute m × P_g au numérateur ET m au dénominateur : (d + m × P_g) / (n + m). Sans "
                      "m × P_g en haut, chaque petite cellule est tirée vers 0, pas vers le taux de son groupe."),
    ("brut", "les taux sont LISSÉS : (d + m × P_g) / (n + m), même pour une cellule bien remplie. Vous prenez le taux "
             "brut d / n (m ne sert que pour les cellules vides)."),
    ("risque_temoin", "vous renvoyez le RISQUE de départ sans offre (ce que fait le score de churn), pas l'effet de "
                      "l'offre : il faut lui soustraire le risque AVEC offre."),
    ("risque_global", "vous renvoyez UN risque, appelés et témoins confondus : c'est un modèle de churn, il ne dit "
                      "rien de l'effet de l'offre. Il faut deux modèles (un par groupe) et leur différence."),
]


def _un_t(f, historique, nouveaux, variables, m, quoi: str, lignes_max: int | None = 2_000_000) -> None:
    avant = copy.deepcopy((historique, nouveaux))
    args = (historique, nouveaux, variables) if m is None else (historique, nouveaux, variables, m)
    r = _appel(f, args, quoi, lignes_max=lignes_max, nouveaux=True,
               conseil_lent="Comptez une fois par cellule (un dict par groupe), puis répondez à chaque nouveau client "
                            "par une lecture : relire l'historique pour chacun, c'est des centaines de millions de "
                            "lignes.")
    _intact(avant[0], historique, quoi, "l'historique")
    _intact(avant[1], nouveaux, quoi, "les nouveaux clients")
    _pas_none(r, quoi)
    attendu = _ref_t_learner(historique, nouveaux, variables, 20 if m is None else m)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list de {len(nouveaux)} uplifts est attendue.")
    if len(r) != len(nouveaux):
        raise AssertionError(f"{quoi} renvoie {len(r)} prédictions pour {len(nouveaux)} nouveaux clients : une par "
                             "client, dans le même ordre.")
    for x in r:
        if not _nombre(x):
            raise AssertionError(f"{quoi} renvoie {_court(x, 60)} parmi ses prédictions : des nombres (float) sont "
                                 "attendus.")
    if _memes(r, attendu):
        return
    for variante, message in _VARIANTES_T:
        autre = _ref_t_learner(historique, nouveaux, variables, 20 if m is None else m, variante)
        if _memes(r, autre) and not _memes(autre, attendu):
            i = next(i for i in range(len(r)) if not _proche(autre[i], attendu[i]))
            raise AssertionError(f"Dans {quoi}, le nouveau client n°{i} reçoit {_fmt(r[i])} au lieu de "
                                 f"{_fmt(attendu[i])} : {message}")
    if m is None and _memes(r, _ref_t_learner(historique, nouveaux, variables, 1)):
        raise AssertionError(f"{quoi} (sans m) ne donne pas le résultat de m = 20 : la valeur par défaut de m est 20.")
    i = next(i for i in range(len(r)) if not _proche(r[i], attendu[i]))
    cell = tuple(nouveaux[i][v] for v in variables)
    raise AssertionError(f"Dans {quoi}, le nouveau client n°{i} (cellule {_court(cell, 60)}) reçoit {_fmt(r[i])} au "
                         f"lieu de {_fmt(attendu[i])} : risque_temoin − risque_offre, chacun égal à "
                         "(d_g(c) + m × P_g) / (n_g(c) + m).")


_SEPARE_213 = [{"contrat": "b", "offre": 0, "parti": 1}, {"contrat": "b", "offre": 1, "parti": 1},
               {"contrat": "b", "offre": 1, "parti": 0}, {"contrat": "a", "offre": 1, "parti": 0},
               {"contrat": "a", "offre": 1, "parti": 0}]        # chaque erreur classique y donne un résultat différent


def verifier_t_learner(f) -> None:
    _un_t(f, copy.deepcopy(_SEPARE_213), [{"contrat": "a"}, {"contrat": "b"}, {"contrat": "c"}], ["contrat"], 2,
          f"t_learner({_court(_SEPARE_213, 200)}, [{{'contrat': 'a'}}, {{'contrat': 'b'}}, {{'contrat': 'c'}}], "
          "['contrat'], m=2)")
    historique = [{"contrat": "mensuel", "offre": 1, "parti": 0}, {"contrat": "mensuel", "offre": 0, "parti": 1},
                  {"contrat": "annuel", "offre": 1, "parti": 0}, {"contrat": "annuel", "offre": 0, "parti": 0}]
    _un_t(f, historique, [{"contrat": "mensuel"}, {"contrat": "biennal"}], ["contrat"], 2,
          "t_learner(l'exemple de la docstring, m=2)")
    campagne, prospects = campagne_exemple(), prospects_exemple()
    profils = [{"contrat": c} for c in ("mensuel", "annuel", "biennal")]
    _un_t(f, campagne, profils, ["contrat"], 20, "t_learner(campagne_exemple(), [mensuel, annuel, biennal], "
                                                 "['contrat'], m=20)")
    _un_t(f, campagne, prospects, VARIABLES, 20, "t_learner(campagne_exemple(), prospects_exemple(), VARIABLES, 20)")
    _un_t(f, campagne, prospects, VARIABLES, None, "t_learner(campagne_exemple(), prospects_exemple(), VARIABLES)")
    _un_t(f, campagne, prospects[:50], ("appels", "contrat"), 3.5, "t_learner(campagne, prospects[:50], "
                                                                  "('appels', 'contrat'), m=3.5)")
    _un_t(f, campagne, prospects[:20], (), 20, "t_learner(campagne, prospects[:20], (), m=20) (aucune variable : "
                                               "une seule cellule)")
    _un_t(f, campagne, [], VARIABLES, 20, "t_learner(campagne, [], VARIABLES, 20)")
    for m in (0, -5, None, "20", True, float("nan")):
        _doit_lever(f, (historique, [{"contrat": "mensuel"}], ["contrat"], m), f"t_learner(…, m={m!r})",
                    "m est un nombre (int ou float, pas un bool) > 0 : sinon ValueError. Écrivez `not m > 0`, qui "
                    "refuse aussi NaN.")
    for clients, pourquoi in (([{"contrat": "a", "offre": 1, "parti": 0}], "aucun témoin"),
                              ([{"contrat": "a", "offre": 0, "parti": 1}], "aucun appelé"),
                              ([], "historique vide"),
                              ([{"contrat": "a", "offre": 1, "parti": 0}, {"contrat": "a", "offre": 0, "parti": 2}],
                               "parti = 2"),
                              ([{"contrat": "a", "offre": "oui", "parti": 0}, {"contrat": "a", "offre": 0, "parti": 1}],
                               "offre = 'oui'")):
        _doit_lever(f, (clients, [{"contrat": "a"}], ["contrat"], 20), f"t_learner({_court(clients, 90)}, …)",
                    f"Ici : {pourquoi}. Sans appelé ou sans témoin, pas d'effet mesurable ; et `offre` / `parti` "
                    "valent 0 ou 1 : sinon ValueError.")
    alea = random.Random(213)
    for essai in range(120):
        historique = _campagne(alea, alea.randint(2, 80), stratifiee=alea.random() < 0.3,
                               booleens=alea.random() < 0.3, manquants=alea.random() < 0.3)
        if len({int(c["offre"]) for c in historique}) < 2:
            historique[0]["offre"], historique[1]["offre"] = 1, 0
        nouveaux = _sans_reponse(_campagne(alea, alea.randint(0, 30), manquants=alea.random() < 0.3))
        variables = alea.choice([VARIABLES, ("contrat",), ("appels", "anciennete"), ("contrat", "appels")])
        m = alea.choice([1, 2, 5, 20, 0.5, 100])
        _un_t(f, historique, nouveaux, variables, m, f"t_learner(<{len(historique)} clients>, <{len(nouveaux)} nouveaux>, "
                                                     f"{variables}, m={m}) (tiré au hasard n°{essai})")
    grand = generer_campagne(20_000, graine=2130)
    nouveaux = _sans_reponse(generer_campagne(20_000, graine=2131))
    _un_t(f, grand, nouveaux, VARIABLES, 20, "t_learner(<20 000 clients>, <20 000 nouveaux>, VARIABLES, 20)",
          lignes_max=60 * 40_000)


# ---------------------------------------------------------------------------- défi 214 : issue transformée


def _ref_issue(clients, variante: str = "") -> list[float]:
    part = sum(int(c["offre"]) for c in clients) / len(clients) if clients else 0.5
    resultat = []
    for c in clients:
        t, y, p = int(c["offre"]), int(c["parti"]), c["p_offre"]
        if variante == "demi":
            p = 0.5
        elif variante == "part_globale":
            p = part
        if variante == "retenu":
            y = 1 - y
        if not y or (variante == "part_globale" and not 0 < p < 1):
            resultat.append(0.0)
        elif variante == "sans_un_moins_p":
            resultat.append(-1 / p if t else 1.0)
        elif variante == "inverse":
            resultat.append(-1 / (1 - p) if t else 1 / p)
        elif variante == "signe":
            resultat.append(1 / p if t else -1 / (1 - p))
        else:
            resultat.append(-1 / p if t else 1 / (1 - p))
    return resultat


_VARIANTES_ISSUE = [
    ("signe", "on compte des départs ÉVITÉS : un témoin parti fait MONTER l'effet de l'offre (+1 / (1 − p)), un appelé "
              "parti le fait baisser (−1 / p). Vos signes sont inversés."),
    ("retenu", "c'est `parti` qui porte l'information : un client RESTÉ vaut 0. Vous utilisez 1 − parti (les clients "
               "restés), ce qui inverse le sens de l'effet."),
    ("demi", "chaque client a SA probabilité d'avoir été appelé, client[\"p_offre\"] : avec 0.5 pour tout le monde, une "
             "campagne tirée à 80/20 ou selon le score donne un effet faux."),
    ("part_globale", "chaque client a SA probabilité d'avoir été appelé, client[\"p_offre\"] : pas la part d'appelés de "
                     "toute la liste. Quand p dépend du score, c'est justement ce qui corrige le biais."),
    ("sans_un_moins_p", "un témoin parti vaut 1 / (1 − p), pas 1 : il représente 1 / (1 − p) clients comme lui, comme "
                        "un appelé parti en représente 1 / p."),
    ("inverse", "un appelé avait la chance p d'être appelé : il compte −1 / p ; un témoin avait la chance 1 − p d'être "
                "témoin : il compte 1 / (1 − p). Vous avez croisé p et 1 − p."),
]


def _une_issue(f, clients, quoi: str) -> None:
    avant = copy.deepcopy(clients)
    r = _appel(f, (clients,), quoi)
    _intact(avant, clients, quoi, "les clients")
    _pas_none(r, quoi)
    attendu = _ref_issue(clients)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list d'un nombre par client est attendue.")
    if len(r) != len(clients):
        raise AssertionError(f"{quoi} renvoie {len(r)} valeurs pour {len(clients)} clients : une par client.")
    for x in r:
        if not _nombre(x):
            raise AssertionError(f"{quoi} renvoie {_court(x, 60)} parmi ses valeurs : des nombres (float) sont attendus.")
    if _memes(r, attendu):
        return
    for variante, message in _VARIANTES_ISSUE:
        autre = _ref_issue(clients, variante)
        if _memes(r, autre) and not _memes(autre, attendu):
            i = next(i for i in range(len(r)) if not _proche(autre[i], attendu[i]))
            raise AssertionError(f"Dans {quoi}, le client n°{i} ({_court(clients[i], 90)}) vaut {_fmt(r[i])} au lieu de "
                                 f"{_fmt(attendu[i])} : {message}")
    i = next(i for i in range(len(r)) if not _proche(r[i], attendu[i]))
    raise AssertionError(f"Dans {quoi}, le client n°{i} ({_court(clients[i], 90)}) vaut {_fmt(r[i])} au lieu de "
                         f"{_fmt(attendu[i])} : parti × ((1 − offre) / (1 − p) − offre / p).")


_SEPARE_214 = [{"offre": 1, "parti": 0, "p_offre": 0.8}, {"offre": 0, "parti": 0, "p_offre": 0.75},
               {"offre": 0, "parti": 1, "p_offre": 0.5}, {"offre": 0, "parti": 1, "p_offre": 0.2},
               {"offre": 1, "parti": 1, "p_offre": 0.25}]      # chaque erreur classique y donne un résultat différent


def verifier_issue_transformee(f) -> None:
    _une_issue(f, copy.deepcopy(_SEPARE_214), f"issue_transformee({_court(_SEPARE_214, 200)})")
    doc = [{"offre": 0, "parti": 1, "p_offre": 0.5}, {"offre": 1, "parti": 1, "p_offre": 0.5},
           {"offre": 1, "parti": 0, "p_offre": 0.5}, {"offre": 0, "parti": 1, "p_offre": 0.75}]
    _une_issue(f, doc, "issue_transformee(l'exemple de la docstring)")
    _une_issue(f, [], "issue_transformee([])")
    _une_issue(f, campagne_stratifiee_exemple(), "issue_transformee(campagne_stratifiee_exemple())")
    _une_issue(f, campagne_exemple(), "issue_transformee(campagne_exemple())")
    _une_issue(f, [{"offre": True, "parti": True, "p_offre": 0.2}, {"offre": False, "parti": True, "p_offre": 0.9},
                   {"offre": 1, "parti": 1, "p_offre": 1 / 3}], "issue_transformee(offre et parti en bool, p variés)")
    for p in (0, 1, 0.0, 1.0, 1.5, -0.2, None, float("nan"), "0.5"):
        _doit_lever(f, ([{"offre": 1, "parti": 1, "p_offre": 0.5}, {"offre": 0, "parti": 1, "p_offre": p}],),
                    f"issue_transformee(…) avec p_offre = {p!r}",
                    "Il faut 0 < p < 1 : p = 0 ou 1, c'est un client qu'on ne pouvait pas comparer, et un p manquant "
                    "(None, NaN) ou qui n'est pas un nombre → ValueError. `not 0 < p < 1` refuse aussi NaN.")
    for cle in ("offre", "parti"):
        for mauvais in (2, "oui", None):
            client = {"offre": 1, "parti": 0, "p_offre": 0.5, cle: mauvais}
            _doit_lever(f, ([client],), f"issue_transformee([{client}])",
                        "`offre` et `parti` valent 0 ou 1 (True et False acceptés), sinon ValueError.")
    alea = random.Random(214)
    for essai in range(150):
        clients = _campagne(alea, alea.randint(1, 40), stratifiee=alea.random() < 0.5, booleens=alea.random() < 0.3)
        if alea.random() < 0.5:
            for c in clients:
                c["p_offre"] = alea.choice([0.1, 0.25, 0.3, 0.5, 0.7, 0.8, 0.95, 1 / 3])
        _une_issue(f, clients, f"issue_transformee(<{len(clients)} clients tirés au hasard>) (n°{essai})")


# ---------------------------------------------------------------------------- défi 215 : courbe de Qini


def _ref_qini(uplifts, clients, variante: str = "") -> tuple:
    n = len(clients)
    signe = 1 if variante == "croissant" else -1
    ordre = sorted(range(n), key=lambda i: signe * uplifts[i])
    total_t = sum(int(c["offre"]) for c in clients)
    total_c = n - total_t
    points = [] if variante == "sans_origine" else [(0, 0.0)]
    n_t = n_c = d_t = d_c = 0
    for position, i in enumerate(ordre):
        if int(clients[i]["offre"]):
            n_t += 1
            d_t += int(clients[i]["parti"])
        else:
            n_c += 1
            d_c += int(clients[i]["parti"])
        dernier = position == n - 1 or uplifts[ordre[position + 1]] != uplifts[i]
        if not dernier and variante != "ex_aequo":
            continue
        if variante == "ratio_global":
            gain = d_c * total_t / total_c - d_t if total_c else 0.0
        elif variante == "sans_ratio":
            gain = float(d_c - d_t)
        elif variante == "taux":
            gain = (d_c / n_c - d_t / n_t) * (position + 1) if n_c and n_t else 0.0
        elif variante == "nc_zero" and not n_c:
            gain = float(-d_t)
        elif variante == "signe":
            gain = d_t - d_c * n_t / n_c if n_c else 0.0
        else:
            gain = d_c * n_t / n_c - d_t if n_c else 0.0
        points.append((position + 1, gain))
    avec_origine = points if points and points[0] == (0, 0.0) else [(0, 0.0)] + points
    courbe = sum((k2 - k1) * (g1 + g2) / 2 for (k1, g1), (k2, g2) in zip(avec_origine, avec_origine[1:]))
    hasard = n * avec_origine[-1][1] / 2
    if variante == "qini_brut":
        qini = courbe / n
    elif variante == "qini_non_normalise":
        qini = courbe - hasard
    else:
        qini = (courbe - hasard) / n
    return points, qini


_VARIANTES_QINI = [
    ("ex_aequo", "des clients EX AEQUO (même prédiction) entrent ENSEMBLE : un seul point, après le dernier d'entre eux. "
                 "Point par point, la courbe dépend de l'ordre de la liste, que personne n'a choisi, et un modèle qui "
                 "prédit la même chose pour tous aurait l'air bon."),
    ("croissant", "on range les clients par prédiction DÉCROISSANTE : les plus prometteurs d'abord."),
    ("sans_origine", "la courbe commence au point (0, 0.0) : avant d'appeler qui que ce soit, aucun départ évité."),
    ("ratio_global", "le taux de départ sans offre se lit chez les témoins DES k PREMIERS : D_C × N_T / N_C avec les "
                     "compteurs du haut de la liste, pas avec les totaux de toute la campagne."),
    ("sans_ratio", "il y a moins de témoins que d'appelés : on ramène les départs des témoins à la taille du groupe "
                   "appelé, D_C × N_T / N_C, avant de soustraire D_T."),
    ("taux", "vous tracez la courbe d'UPLIFT, (D_C / N_C − D_T / N_T) × k. La courbe de Qini compte les départs évités "
             "chez les APPELÉS du haut de la liste : D_C × N_T / N_C − D_T."),
    ("nc_zero", "tant qu'aucun témoin n'est dans les k premiers (N_C = 0), on ne peut rien estimer : gain = 0.0, pas "
                "−D_T."),
    ("signe", "le gain compte les départs ÉVITÉS : ce qu'auraient fait les appelés sans l'appel (D_C × N_T / N_C) MOINS "
              "ce qu'ils ont fait (D_T). Vous calculez l'inverse."),
    ("qini_brut", "qini est l'aire AU-DESSUS de la droite du hasard : retirez l'aire sous la diagonale de (0, 0) à "
                  "(n, gain total), n × gain / 2, avant de diviser par n."),
    ("qini_non_normalise", "qini = (aire sous la courbe − aire du hasard) / n : vous oubliez de diviser par n."),
]


def _une_qini(f, uplifts, clients, quoi: str, lignes_max: int | None = 2_000_000) -> None:
    avant = copy.deepcopy((uplifts, clients))
    r = _appel(f, (uplifts, clients), quoi, lignes_max=lignes_max,
               conseil_lent="Un tri, puis UNE passe en cumulant N_T, D_T, N_C, D_C : recompter les k premiers pour "
                            "chaque k, c'est des centaines de millions de lignes sur 20 000 clients.")
    _intact(avant[0], uplifts, quoi, "les prédictions")
    _intact(avant[1], clients, quoi, "les clients")
    _pas_none(r, quoi)
    points, qini = _ref_qini(uplifts, clients)
    if not isinstance(r, tuple) or len(r) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : un Qini(points, qini) (ou un tuple (points, qini)) est "
                             "attendu.")
    rp, rq = r
    if not isinstance(rp, list):
        raise AssertionError(f"{quoi} : points vaut {_court(rp, 100)}, une list de tuples (k, gain) est attendue.")
    for pt in rp:
        if (not isinstance(pt, (tuple, list)) or len(pt) != 2 or not _nombre(pt[1])
                or not isinstance(pt[0], numbers.Integral) or isinstance(pt[0], bool)):
            raise AssertionError(f"{quoi} : le point {_court(pt, 60)} n'est pas un tuple (k, gain), avec k entier.")
    if not _nombre(rq):
        raise AssertionError(f"{quoi} : qini vaut {_court(rq, 60)}, un nombre est attendu.")

    def memes_points(a, b):
        return len(a) == len(b) and all(x[0] == y[0] and _proche(x[1], y[1]) for x, y in zip(a, b))

    if memes_points(rp, points) and _proche(rq, qini):
        return
    for variante, message in _VARIANTES_QINI:
        vp, vq = _ref_qini(uplifts, clients, variante)
        if memes_points(rp, vp) and _proche(rq, vq) and not (memes_points(vp, points) and _proche(vq, qini)):
            raise AssertionError(f"Dans {quoi} : {message}")
    if not memes_points(rp, points):
        if [k for k, _ in rp] != [k for k, _ in points]:
            raise AssertionError(f"Dans {quoi}, les k des points sont {_court([k for k, _ in rp], 90)} au lieu de "
                                 f"{_court([k for k, _ in points], 90)} : (0, 0.0), puis un point après le dernier "
                                 "client de chaque valeur distincte de prédiction, de la plus haute à la plus basse.")
        j = next(j for j in range(len(points)) if not _proche(rp[j][1], points[j][1]))
        raise AssertionError(f"Dans {quoi}, le point k = {points[j][0]} a un gain de {_fmt(rp[j][1])} au lieu de "
                             f"{_fmt(points[j][1])} : D_C × N_T / N_C − D_T, avec les appelés et les témoins des k "
                             "premiers (0.0 si N_C = 0).")
    raise AssertionError(f"Dans {quoi}, les points sont justes mais qini vaut {_fmt(rq)} au lieu de {_fmt(qini)} : "
                         "(somme des trapèzes (k2 − k1) × (g1 + g2) / 2 − n × gain total / 2) / n.")


_SEPARE_215 = ([0.3, 0.1, 0.4, 0.1, 0.1, 0.2],
               [{"offre": 0, "parti": 1}, {"offre": 1, "parti": 0}, {"offre": 1, "parti": 1}, {"offre": 0, "parti": 1},
                {"offre": 0, "parti": 1}, {"offre": 0, "parti": 0}])   # chaque erreur classique y donne autre chose


def verifier_courbe_qini(f) -> None:
    uplifts, clients = copy.deepcopy(_SEPARE_215)
    _une_qini(f, uplifts, clients, f"courbe_qini({uplifts}, {_court(clients, 200)})")
    doc = [{"offre": 1, "parti": 0}, {"offre": 0, "parti": 1}, {"offre": 1, "parti": 1}, {"offre": 0, "parti": 0}]
    _une_qini(f, [0.3, 0.3, 0.1, 0.1], doc, "courbe_qini([0.3, 0.3, 0.1, 0.1], l'exemple de la docstring)")
    _une_qini(f, [0.1, 0.3, 0.2, 0.0], doc, "courbe_qini([0.1, 0.3, 0.2, 0.0], l'exemple de la docstring)")
    _une_qini(f, [0.5, 0.5, 0.5, 0.5], doc, "courbe_qini([0.5] * 4, …) (tout le monde ex aequo)")
    _une_qini(f, [3, 2, 1, 0], [{"offre": 1, "parti": 1}, {"offre": 1, "parti": 0}, {"offre": 0, "parti": 1},
                                {"offre": 0, "parti": 0}], "courbe_qini([3, 2, 1, 0], <deux appelés puis deux témoins>)")
    _une_qini(f, [1.0], [{"offre": 0, "parti": 1}], "courbe_qini([1.0], <un seul témoin>)")
    campagne = campagne_exemple()
    _une_qini(f, [c["score"] for c in campagne], campagne, "courbe_qini(<les scores>, campagne_exemple())")
    _une_qini(f, [uplift_vrai(c) for c in campagne], campagne, "courbe_qini(<l'uplift vrai>, campagne_exemple())")
    _une_qini(f, [-c["score"] for c in campagne], campagne, "courbe_qini(<l'opposé des scores>, campagne_exemple())")
    for uplifts, clients, pourquoi in (([0.1], doc, "moins de prédictions que de clients"),
                                       ([0.1] * 5, doc, "plus de prédictions que de clients"),
                                       ([], [], "aucun client"),
                                       ([0.1, float("nan"), 0.2, 0.0], doc, "une prédiction NaN"),
                                       ([0.1, None, 0.2, 0.0], doc, "une prédiction None"),
                                       ([0.1, 0.2], [{"offre": 2, "parti": 0}, {"offre": 0, "parti": 1}], "offre = 2"),
                                       ([0.1, 0.2], [{"offre": 1, "parti": "oui"}, {"offre": 0, "parti": 1}],
                                        "parti = 'oui'")):
        _doit_lever(f, (uplifts, clients), f"courbe_qini({_court(uplifts, 40)}, {_court(clients, 60)})",
                    f"Ici : {pourquoi}. Des longueurs différentes, aucun client, une prédiction manquante (un NaN se "
                    "range n'importe où dans un tri) ou `offre` / `parti` qui ne vaut pas 0 ou 1 → ValueError.")
    alea = random.Random(215)
    for essai in range(150):
        clients = _campagne(alea, alea.randint(1, 50), booleens=alea.random() < 0.3)
        genre = alea.random()
        if genre < 0.4:
            uplifts = [alea.choice([-0.1, 0.0, 0.05, 0.1, 0.2]) for _ in clients]        # beaucoup d'ex aequo
        elif genre < 0.7:
            uplifts = [alea.randint(-16, 16) / 16 for _ in clients]
        else:
            uplifts = [alea.random() for _ in clients]
        _une_qini(f, uplifts, clients, f"courbe_qini(<{len(clients)} prédictions>, <{len(clients)} clients>) (tiré au "
                                       f"hasard n°{essai})")
    grand = generer_campagne(20_000, graine=2150)
    _une_qini(f, [c["score"] for c in grand], grand, "courbe_qini(<20 000 scores>, <20 000 clients>)",
              lignes_max=60 * 20_000)
    _une_qini(f, [uplift_vrai(c) for c in grand], grand, "courbe_qini(<20 000 uplifts, 5 valeurs distinctes>, "
                                                         "<20 000 clients>)", lignes_max=60 * 20_000)


# ---------------------------------------------------------------------------- défi 216 : plan de contact


def _ref_plan(clients, uplifts, cout: int, budget: int, variante: str = "") -> tuple:
    gains = [u * c["valeur"] - cout for c, u in zip(clients, uplifts)]
    lignes = [(g, c["customer_id"], u, c.get("score")) for c, u, g in zip(clients, uplifts, gains)]
    if variante == "par_score":
        lignes.sort(key=lambda t: (-t[3], t[1]))
    elif variante == "par_uplift":
        lignes = [t for t in lignes if t[0] > 0]
        lignes.sort(key=lambda t: (-t[2], t[1]))
    elif variante == "remplir":
        lignes.sort(key=lambda t: (-t[0], t[1]))
    elif variante == "zero_inclus":
        lignes = sorted((t for t in lignes if t[0] >= 0), key=lambda t: (-t[0], t[1]))
    elif variante == "egalite_inverse":
        lignes = sorted((t for t in lignes if t[0] > 0), key=lambda t: t[1], reverse=True)
        lignes.sort(key=lambda t: -t[0])                # tri stable : à gain égal, l'ordre décroissant reste
    else:
        lignes = sorted((t for t in lignes if t[0] > 0), key=lambda t: (-t[0], t[1]))
    choisis = lignes[: budget // cout]
    gain = sum(t[0] for t in choisis)
    if variante == "brut":
        gain = sum(t[0] + cout for t in choisis)
    depense = budget if variante == "cout_budget" else len(choisis) * cout
    return [t[1] for t in choisis], gain, depense


_VARIANTES_PLAN = [
    ("par_score", "vous choisissez les clients au plus fort SCORE de churn : ce sont surtout des causes perdues. On "
                  "appelle ceux dont l'appel RAPPORTE le plus : uplift × valeur − coût."),
    ("par_uplift", "on classe par GAIN attendu (uplift × valeur − coût), pas par uplift seul : un uplift de 0.10 sur un "
                   "client à 1 200 € rapporte plus qu'un uplift de 0.12 sur un client à 500 €."),
    ("remplir", "on n'appelle que les clients au gain strictement positif. Remplir le budget avec des chiens qui "
                "dorment (uplift < 0) ou des appels qui coûtent plus qu'ils ne rapportent, c'est PERDRE de l'argent."),
    ("zero_inclus", "un appel au gain nul coûte du temps de conseiller pour rien : seulement les gains > 0 (strictement)."),
    ("egalite_inverse", "à gain égal, customer_id CROISSANT (« A » avant « B »)."),
    ("brut", "le gain du plan est NET : la somme des uplift × valeur − coût des clients choisis (le coût déjà déduit)."),
    ("cout_budget", "cout est ce que coûtent les appels du plan, len(clients) × cout, pas tout le budget."),
]


def _un_plan(f, clients, uplifts, cout, budget, quoi: str) -> None:
    avant = copy.deepcopy((clients, uplifts))
    r = _appel(f, (clients, uplifts, cout, budget), quoi)
    _intact(avant[0], clients, quoi, "les clients")
    _intact(avant[1], uplifts, quoi, "les prédictions")
    _pas_none(r, quoi)
    ids, gain, depense = _ref_plan(clients, uplifts, cout, budget)
    if not isinstance(r, tuple) or len(r) != 3:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : un Plan(clients, gain, cout) (ou un tuple de 3 valeurs "
                             "dans cet ordre) est attendu.")
    rc, rg, rd = r
    if not isinstance(rc, list):
        raise AssertionError(f"{quoi} : clients vaut {_court(rc, 80)}, une list de customer_id est attendue.")
    if not _nombre(rg) or not _nombre(rd):
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : gain et cout doivent être des nombres.")
    if rc == ids and _proche(rg, gain, 1e-9) and rd == depense:
        return
    for variante, message in _VARIANTES_PLAN:
        vc, vg, vd = _ref_plan(clients, uplifts, cout, budget, variante)
        if rc == vc and _proche(rg, vg) and rd == vd and not (vc == ids and _proche(vg, gain) and vd == depense):
            raise AssertionError(f"Dans {quoi} : {message}")
    if rc != ids:
        if set(rc) == set(ids):
            raise AssertionError(f"Dans {quoi}, les bons clients sont choisis, dans le mauvais ordre "
                                 f"({_court(rc, 80)} au lieu de {_court(ids, 80)}) : gain décroissant, puis customer_id "
                                 "croissant.")
        if len(rc) != len(ids):
            raise AssertionError(f"Dans {quoi}, {len(rc)} clients appelés au lieu de {len(ids)} : au plus budget // cout "
                                 f"= {budget // cout} appels, et seulement ceux dont uplift × valeur − coût > 0.")
        raise AssertionError(f"Dans {quoi}, clients vaut {_court(rc, 80)} au lieu de {_court(ids, 80)}.")
    if not _proche(rg, gain):
        raise AssertionError(f"Dans {quoi}, gain vaut {_fmt(rg)} au lieu de {_fmt(gain)} : la somme des "
                             "uplift × valeur − coût des clients choisis.")
    raise AssertionError(f"Dans {quoi}, cout vaut {_fmt(rd)} au lieu de {depense} : len(clients) × cout.")


def verifier_plan_de_contact(f) -> None:
    doc = [{"customer_id": "A", "valeur": 1000, "score": 0.2}, {"customer_id": "B", "valeur": 1000, "score": 0.9},
           {"customer_id": "C", "valeur": 200, "score": 0.5}, {"customer_id": "D", "valeur": 1000, "score": 0.1}]
    u = [0.10, -0.05, 0.20, 0.04]
    for budget in (1000, 59, 60, 0, 29, 30, 90, 120):
        _un_plan(f, doc, u, 30, budget, f"plan_de_contact(l'exemple de la docstring, cout=30, budget={budget})")
    _un_plan(f, doc, [0.10, 0.12, 0.30, 0.06], 30, 60, "plan_de_contact(<A : 0.10 × 1000, B : 0.12 × 1000, …>, 30, 60)")
    _un_plan(f, doc, [0.05, 0.05, 0.25, 0.05], 20, 1000, "plan_de_contact(<trois gains de 30 € ex aequo>, 20, 1000)")
    _un_plan(f, doc, [0.03, 0.01, 0.15, -0.2], 30, 1000, "plan_de_contact(<des gains nuls ou négatifs>, 30, 1000)")
    campagne, prospects = campagne_exemple(), prospects_exemple()
    predits = _ref_t_learner(campagne, prospects, VARIABLES, 20)
    for budget in (9000, 30_000, 6000, 31):
        _un_plan(f, prospects, predits, 30, budget, f"plan_de_contact(prospects_exemple(), <uplifts du T-learner>, 30, "
                                                    f"{budget})")
    _un_plan(f, [], [], 30, 1000, "plan_de_contact([], [], 30, 1000)")
    for cout, budget in ((0, 100), (-30, 100), (30.0, 100), (True, 100), ("30", 100), (30, -1), (30, 100.5),
                         (30, None), (30, True)):
        _doit_lever(f, (doc, u, cout, budget), f"plan_de_contact(…, cout={cout!r}, budget={budget!r})",
                    "cout est un entier d'euros >= 1 et budget un entier d'euros >= 0 (ni bool, ni float, ni texte) : "
                    "sinon ValueError.")
    for uplifts, clients, pourquoi in (([0.1], doc, "moins de prédictions que de clients"),
                                       ([0.1] * 5, doc, "plus de prédictions que de clients"),
                                       ([0.1, None, 0.2, 0.0], doc, "un uplift None"),
                                       ([0.1, float("nan"), 0.2, 0.0], doc, "un uplift NaN"),
                                       (u, doc[:3] + [{"customer_id": "D", "valeur": None}], "une valeur None"),
                                       (u, doc[:3] + [{"customer_id": "D", "valeur": float("nan")}], "une valeur NaN")):
        _doit_lever(f, (clients, uplifts, 30, 1000), f"plan_de_contact(…) ({pourquoi})",
                    f"Ici : {pourquoi}. Des longueurs différentes (`zip` tronquerait sans rien dire) ou une valeur "
                    "manquante → ValueError.")
    alea = random.Random(216)
    for essai in range(150):
        clients = _sans_reponse(_campagne(alea, alea.randint(0, 40)))
        if alea.random() < 0.5:
            for c in clients:
                c["valeur"] = alea.choice([300, 600, 1000])
        uplifts = [alea.choice([-0.1, -0.02, 0.0, 0.03, 0.05, 0.1, 0.2]) if alea.random() < 0.6
                   else alea.randint(-32, 64) / 256 for _ in clients]
        cout = alea.choice([10, 20, 30, 50])
        budget = alea.choice([0, cout - 1, cout, 3 * cout + 1, 10 * cout, 10_000])
        _un_plan(f, clients, uplifts, cout, budget, f"plan_de_contact(<{len(clients)} clients>, <uplifts>, {cout}, "
                                                    f"{budget}) (tiré au hasard n°{essai})")


# ---------------------------------------------------------------------------- défi 217 : prédictions hors pli


def _valeur_espion(cid: str) -> float:
    return sum(ord(x) * (i + 1) for i, x in enumerate(cid)) / 1000


class _Apprenti:
    """Un modèle ESPION : il note avec quoi on l'entraîne et ce qu'on lui montre, et prédit des valeurs reconnaissables."""

    def __init__(self, manque: int = 0):
        self.appels: list[tuple[list, list, list]] = []
        self.manque = manque

    def __call__(self, entrainement, a_predire):
        self.appels.append(([c.get("customer_id") for c in entrainement],
                            [sorted(c) for c in entrainement],
                            [dict(c) for c in a_predire]))
        resultat = [_valeur_espion(str(c.get("customer_id"))) + len(entrainement) / 10_000 for c in a_predire]
        return resultat[: len(resultat) - self.manque] if self.manque else resultat


def _ref_hors_pli(historique, k: int) -> list[float]:
    n = len(historique)
    return [_valeur_espion(c["customer_id"]) + (n - len(range(i % k, n, k))) / 10_000 for i, c in enumerate(historique)]


def _un_hors_pli(f, historique, k, quoi: str) -> None:
    avant = copy.deepcopy(historique)
    espion = _Apprenti()
    args = (historique, espion) if k is None else (historique, espion, k)
    r = _appel(f, args, quoi)
    _intact(avant, historique, quoi, "l'historique")
    _pas_none(r, quoi)
    k = 5 if k is None else k
    n = len(historique)
    ids = [c["customer_id"] for c in historique]
    par_id = {c["customer_id"]: c for c in historique}
    plis = [[ids[i] for i in range(f_, n, k)] for f_ in range(k)]
    if len(espion.appels) != k:
        raise AssertionError(f"{quoi} appelle `apprendre` {len(espion.appels)} fois au lieu de {k} : un appel par pli, "
                             "un pli = les clients n°i tels que i % k == f.")
    vus = []
    for entrainement, cles_entrainement, a_predire in espion.appels:
        predits = [c.get("customer_id") for c in a_predire]
        for c in a_predire:
            if "parti" in c or "offre" in c:
                raise AssertionError(f"{quoi} montre au modèle des clients à prédire qui contiennent encore "
                                     f"{'`parti`' if 'parti' in c else '`offre`'} : un modèle qui peut lire la réponse "
                                     "finit toujours par la lire. Copiez chaque client SANS \"offre\" ni \"parti\".")
            if c.get("customer_id") not in par_id or len(c) != len(par_id[c["customer_id"]]) - 2:
                raise AssertionError(f"{quoi} montre au modèle le client à prédire {_court(c, 100)} : on ne retire QUE "
                                     "\"offre\" et \"parti\", le modèle a besoin de toutes ses variables.")
        if any(not {"offre", "parti"} <= set(cles) for cles in cles_entrainement):
            raise AssertionError(f"{quoi} entraîne le modèle sur des clients sans `offre` ou sans `parti` : on ne retire "
                                 "la réponse que des clients À PRÉDIRE. Sans elle, le modèle n'a rien à apprendre.")
        commun = set(predits) & set(entrainement)
        if commun:
            raise AssertionError(f"{quoi} entraîne le modèle avec des clients qu'il doit ensuite prédire (par exemple "
                                 f"{sorted(commun)[0]!r}) : c'est exactement la triche que la validation hors pli évite. "
                                 "Entraînement = les clients dont le pli n'est PAS f.")
        if predits not in plis:
            if sorted(predits) in [sorted(p) for p in plis]:
                raise AssertionError(f"{quoi} donne les clients d'un pli dans le désordre : gardez l'ordre de "
                                     "l'historique.")
            raise AssertionError(f"{quoi} donne à prédire {_court(predits, 80)}, qui n'est pas un pli : le pli du client "
                                 f"n°i est i % k (ici {k}). Avec k = {k}, le pli 0 contient les clients n°0, {k}, "
                                 f"{2 * k}…, pas un bloc de clients consécutifs.")
        if sorted(entrainement) != sorted(set(ids) - set(predits)) or len(entrainement) != n - len(predits):
            raise AssertionError(f"{quoi} entraîne le modèle avec {len(entrainement)} clients au lieu de "
                                 f"{n - len(predits)} : tous les clients des AUTRES plis, une fois chacun.")
        if entrainement != [cid for cid in ids if cid not in set(predits)]:
            raise AssertionError(f"{quoi} entraîne le modèle avec les bons clients, mais dans le désordre : gardez "
                                 "l'ordre de l'historique.")
        vus.append(predits)
    if sorted(map(tuple, vus)) != sorted(map(tuple, plis)):
        raise AssertionError(f"{quoi} prédit deux fois le même pli et oublie un autre : chaque pli f de 0 à k − 1, une "
                             "fois.")
    attendu = _ref_hors_pli(historique, k)
    if not isinstance(r, list) or len(r) != n:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : une list de {n} prédictions (une par client de "
                             "l'historique) est attendue.")
    if _memes(r, attendu):
        return
    concatene = [x for p in range(k) for x in (attendu[i] for i in range(p, n, k))]
    if _memes(r, concatene):
        raise AssertionError(f"{quoi} renvoie les prédictions pli par pli, mises bout à bout : la prédiction n°i doit "
                             "être celle du client n°i de l'historique. Rangez chacune à SA place.")
    i = next(i for i in range(n) if not _meme(r[i], attendu[i]))
    raise AssertionError(f"Dans {quoi}, la prédiction n°{i} vaut {_fmt(r[i])} au lieu de {_fmt(attendu[i])} (ce que "
                         "le modèle du pli de ce client a prédit pour lui).")


def verifier_predictions_hors_pli(f) -> None:
    campagne = campagne_exemple()
    cas = [(campagne[:5], 2), (campagne[:10], 5), (campagne[:10], None), (campagne[:7], 7), (campagne[:6], 3),
           (campagne[:200], 5), (campagne[:11], 4)]
    for historique, k in cas:
        appel = "" if k is None else f", k={k}"
        _un_hors_pli(f, historique, k, f"predictions_hors_pli(campagne_exemple()[:{len(historique)}], apprendre{appel})")
    for k in (1, 0, -2, 6, 2.0, True, "3", None):
        _doit_lever(f, (campagne[:5], _Apprenti(), k), f"predictions_hors_pli(<5 clients>, apprendre, k={k!r})",
                    "k est un int entre 2 et le nombre de clients : sinon ValueError (k = 1 : le modèle n'aurait aucun "
                    "client pour s'entraîner ; k > n : des plis vides).")
    _doit_lever(f, (campagne[:6], _Apprenti(manque=1), 2), "predictions_hors_pli(<6 clients>, <un modèle qui oublie un "
                                                         "client>, k=2)",
                "Si `apprendre` ne renvoie pas une prédiction par client à prédire, ValueError : `zip` tronquerait sans "
                "rien dire, et un client n'aurait pas de prédiction.")
    apprendre = lambda e, p: _ref_t_learner(e, p, VARIABLES, 20)  # noqa: E731
    n = len(campagne)
    attendu = [None] * n
    for pli in range(5):
        indices = range(pli, n, 5)
        entrainement = [c for i, c in enumerate(campagne) if i % 5 != pli]
        predits = apprendre(entrainement, _sans_reponse([campagne[i] for i in indices]))
        for i, u in zip(indices, predits):
            attendu[i] = u
    quoi = "predictions_hors_pli(campagne_exemple(), <le T-learner du corrigé>, k=5)"
    avant = copy.deepcopy(campagne)
    r = _appel(f, (campagne, apprendre, 5), quoi)
    _intact(avant, campagne, quoi, "l'historique")
    if not _memes(r, attendu):
        raise AssertionError(f"{quoi} ne donne pas les mêmes prédictions que le corrigé : relisez les messages sur les "
                             "petits historiques.")
    alea = random.Random(217)
    for essai in range(100):
        n = alea.randint(2, 40)
        historique = _campagne(alea, n, manquants=alea.random() < 0.3)
        _un_hors_pli(f, historique, alea.randint(2, n), f"predictions_hors_pli(<{n} clients tirés au hasard>, apprendre, "
                                                        f"k) (n°{essai})")


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"effet_par_segment": 212, "t_learner": 213, "issue_transformee": 214, "courbe_qini": 215,
            "plan_de_contact": 216, "predictions_hors_pli": 217}

SERIE = Serie(
    code="35",
    titre="Cibler ceux qu'on peut convaincre : effet par segment, T-learner, issue transformée, Qini, plan, hors pli",
    module_eleve="defis.serie_35",
    module_corrige="defis.corrige_serie_35",
    defis=(
        Defi(212, "effet_par_segment", "🟢", "1.2", verifier_effet_par_segment),
        Defi(213, "t_learner", "🟡", "—", verifier_t_learner),
        Defi(214, "issue_transformee", "🟡", "—", verifier_issue_transformee),
        Defi(215, "courbe_qini", "🔴", "4.4", verifier_courbe_qini),
        Defi(216, "plan_de_contact", "🟡", "4.3", verifier_plan_de_contact),
        Defi(217, "predictions_hors_pli", "🔴", "3.4", verifier_predictions_hors_pli),
    ),
)

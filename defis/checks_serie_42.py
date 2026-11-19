"""Vérifications automatiques de la série 42 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est le contrôle de gestion, qui refait vos calculs avant de les présenter au comité de
direction, et l'entrepôt de données, qui vous envoie les relevés tels qu'ils sont vraiment.

- il compare vos matrices, vos projections, vos durées et vos valeurs à celles d'une version de référence, à un
  milliardième près (les règles sont fixées pour qu'il n'y ait qu'une bonne réponse) ;
- quand votre résultat est faux, il rejoue les erreurs classiques (des passages comptés entre deux clients ou
  par-dessus un mois manquant, décembre et janvier qui ne se suivent pas, un doublon compté deux fois, une ligne
  divisée par la colonne, la matrice prise dans le mauvais sens, le mois 0 oublié, 1 / taux de départ, le mois en
  cours oublié, une boucle arrêtée à 1 000 mois, 10 % / 12, le mois 0 actualisé, `>` au lieu de `>=`, la moyenne
  des deux matrices au lieu du lissage…) pour nommer la vôtre ;
- il vous passe les relevés sous forme de flux qui ne se lisent qu'une fois (`UneSeuleFois`), dans le désordre,
  avec un mois perdu et des doublons ; il choisit des matrices où chaque erreur se voit (des clients qui vivent
  10 000 mois, des coûts couverts au centime près), et compte les lignes exécutées dans votre fichier quand un
  calcul refait à chaque mois (ou une boucle qui attend que le résultat ne bouge plus) serait trop lent.
"""

from __future__ import annotations

import random
import sys

from .flux import UneSeuleFois
from .valeur import (
    ETATS,
    MARGES,
    PARTI,
    TAUX_ANNUEL,
    generer_releves,
    matrice_exemple,
    releves_exemple,
    segments_exemple,
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


class _BoucleSansFin(BaseException):
    """Levée dans VOTRE code quand il travaille trop (BaseException : `except Exception` ne l'attrape pas)."""


def _fichier(objet) -> str | None:
    code = getattr(objet, "__code__", None) or getattr(getattr(objet, "__func__", None), "__code__", None)
    if code is not None:
        return code.co_filename
    module = sys.modules.get(getattr(objet, "__module__", None) or "")
    return getattr(module, "__file__", None)


def _trace(fichier: str | None, appel, lignes_max: int | None):
    """Le résultat de `appel()` ; arrête net au-delà de `lignes_max` lignes exécutées dans `fichier`."""
    if fichier is None or lignes_max is None:
        return appel()
    compte = [0]

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


_CONSEIL_LENT = "Votre code a été arrêté : il exécute beaucoup trop de lignes (une boucle qui ne s'arrête pas ?)."


def _conseil_exception(exc: BaseException) -> str:
    texte = str(exc)
    if isinstance(exc, RuntimeError) and "déjà été lu" in texte:
        return ("Vous lisez les relevés DEUX fois. Un générateur relu ne lève pas d'erreur : il est VIDE, et votre "
                "deuxième boucle ne verrait rien, en silence. Une seule boucle `for client, mois, etat in releves:` "
                "qui range tout dans un dict, puis vous travaillez sur le dict.")
    if isinstance(exc, TypeError) and "UneSeuleFois" in texte:
        return ("Les relevés arrivent sous forme de flux : pas de `len(releves)`, pas de `releves[i]`, pas de "
                "`sorted(releves, …)` suivi d'une deuxième lecture. Rangez-les d'abord, en UNE boucle, dans un dict "
                "par client.")
    if isinstance(exc, ZeroDivisionError):
        return ("Une division par zéro : une ligne sans aucun passage (un état jamais quitté, comme « Parti », qui n'a "
                "plus de relevé après le départ) a un total de 0. Sa règle : 1.0 sur la diagonale, 0.0 ailleurs.")
    if type(exc).__name__ == "LinAlgError":
        return ("NumPy dit que le système n'a pas de solution : depuis certains états, on ne part jamais (durée de vie "
                "infinie). Vérifiez que chaque état transitoire peut atteindre un état absorbant AVANT de résoudre, "
                "et levez une ValueError qui nomme les états bloqués.")
    if isinstance(exc, KeyError):
        return ("Une clé manque : un état absent de `depart` ou de `marges` compte 0 (`dico.get(etat, 0)`), et la "
                "réponse a une clé pour CHAQUE état de la matrice (ou de `etats`).")
    return ""


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = None, conseil_lent: str = "",
           kwargs: dict | None = None, division: str = ""):
    """f(*args), en transformant toute erreur en message lisible (NotImplementedError : le défi est à faire).

    `division` : le conseil à donner, pour CE défi, si votre code divise par zéro.
    """
    kwargs = kwargs or {}
    try:
        return _trace(_fichier(f) if lignes_max else None, lambda: f(*args, **kwargs), lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError. {conseil}".rstrip()) from None
    except ZeroDivisionError as exc:
        conseil = division or _conseil_exception(exc) or conseil
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {_conseil_exception(exc) or conseil}".rstrip()) from None


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu: type = ValueError, kwargs: dict | None = None):
    """Exige que f(*args) lève `attendu` (et rien d'autre). Renvoie l'exception, pour lire son message."""
    kwargs = kwargs or {}
    try:
        resultat = f(*args, **kwargs)
    except NotImplementedError:
        raise
    except attendu as exc:
        return exc
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendu.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendu.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _premier_bug(obtenus, rejouer, bugs: dict, egal) -> str:
    """Le conseil du premier bug dont le rejeu donne EXACTEMENT vos réponses."""
    for bug, conseil in bugs.items():
        try:
            faux = rejouer(bug)
        except Exception:
            continue
        if egal(faux, obtenus):
            return conseil
    return ""


def _nombre(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _proche(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is b
    return _nombre(a) and abs(a - b) <= tol * max(1.0, abs(b))


def _dicts_proches(a, b, tol: float = 1e-9) -> bool:
    return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
            and all(_proche(a[k], b[k], tol) for k in b))


def _matrices_proches(a, b, tol: float = 1e-9) -> bool:
    return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
            and all(_dicts_proches(a[k], b[k], tol) for k in b))


def _listes(egal):
    return lambda a, b: len(a) == len(b) and all(egal(x, y) for x, y in zip(a, b))


def _ecart(obtenu: dict, attendu: dict, tol: float = 1e-9) -> str:
    """La première case qui diffère, pour la montrer dans le message."""
    for k, v in attendu.items():
        if isinstance(v, dict):
            for j, w in v.items():
                x = obtenu.get(k, {}).get(j) if isinstance(obtenu.get(k), dict) else None
                if not _proche(x, w, tol):
                    return f"[{k!r}][{j!r}] vaut {_court(x, 40)} au lieu de {w!r}"
        elif not _proche(obtenu.get(k), v, tol):
            return f"[{k!r}] vaut {_court(obtenu.get(k), 40)} au lieu de {v!r}"
    return ""


def _complete(etats, absorbants=(), alea: random.Random | None = None, zeros: float = 0.3) -> dict:
    """Une matrice aléatoire complète ; chaque état transitoire a au moins 2 % de chances d'être absorbé."""
    alea = alea or random.Random(0)
    matrice = {}
    for i in etats:
        if i in absorbants:
            matrice[i] = {j: float(i == j) for j in etats}
            continue
        poids = {j: (0.0 if alea.random() < zeros else alea.random()) for j in etats}
        if absorbants:
            poids[alea.choice(list(absorbants))] += 0.05 + 0.3 * alea.random()
        elif not any(poids.values()):
            poids[i] = 1.0
        total = sum(poids.values())
        matrice[i] = {j: poids[j] / total for j in etats}
    return matrice


# ============================================================================ défi 254 : matrice_transition


def _numero(mois) -> int:
    if not isinstance(mois, str) or len(mois) != 7 or mois[4] != "-" or not (mois[:4] + mois[5:]).isdigit():
        raise ValueError(f"mois illisible : {mois!r}")
    annee, numero = int(mois[:4]), int(mois[5:])
    if not 1 <= numero <= 12:
        raise ValueError(f"mois illisible : {mois!r}")
    return annee * 12 + numero - 1


def _ref_passages(releves, etats, bug: str = "") -> list[tuple]:
    """[(client, depuis, vers)] ; `bug` rejoue une erreur classique de comptage."""
    releves = list(releves)
    connus = set(etats)
    for client, mois, etat in releves:
        if etat not in connus:
            raise ValueError(f"état inconnu : {etat!r}")
        _numero(mois)
    if bug == "doublons":            # une jointure de la table avec elle-même : un doublon compte deux fois
        par_mois: dict = {}
        for client, mois, etat in releves:
            par_mois.setdefault((client, _numero(mois)), []).append(etat)
        return [(c, i, j) for (c, n), depuis in par_mois.items() for i in depuis for j in par_mois.get((c, n + 1), ())]
    uniques = list(dict.fromkeys(releves))
    vus: dict = {}
    for client, mois, etat in uniques:
        deja = vus.setdefault((client, mois), etat)
        if deja != etat:
            raise ValueError("relevés contradictoires")
    if bug == "ordre":               # dans l'ordre d'arrivée, sans trier
        par_client: dict = {}
        for client, mois, etat in uniques:
            par_client.setdefault(client, []).append((_numero(mois), etat))
        return [(c, a[1], b[1]) for c, suite in par_client.items() for a, b in zip(suite, suite[1:])
                if b[0] == a[0] + 1]
    if bug == "frontiere":           # triés par (client, mois), mais sans regarder si le client change
        tries = sorted(uniques, key=lambda r: (str(r[0]), _numero(r[1])))
        return [(a[0], a[2], b[2]) for a, b in zip(tries, tries[1:]) if _numero(b[1]) == _numero(a[1]) + 1]
    par_client = {}
    for client, mois, etat in uniques:
        par_client.setdefault(client, []).append((mois, etat))
    passages = []
    for client, suite in par_client.items():
        suite.sort(key=lambda r: _numero(r[0]))
        for (m1, e1), (m2, e2) in zip(suite, suite[1:]):
            if bug == "trous":
                ok = True
            elif bug == "decembre":
                ok = int(m2.replace("-", "")) == int(m1.replace("-", "")) + 1
            else:
                ok = _numero(m2) == _numero(m1) + 1
            if ok:
                passages.append((client, e1, e2))
    return passages


def _ref_compter(passages, etats) -> dict:
    comptes = {i: dict.fromkeys(etats, 0) for i in etats}
    for _, depuis, vers in passages:
        comptes[depuis][vers] += 1
    return comptes


def _ref_normaliser(comptes, etats, bug: str = "", releves=None) -> dict:
    if bug == "colonnes":
        totaux = {j: sum(comptes[i][j] for i in etats) for j in etats}
        return {i: {j: comptes[i][j] / totaux[j] if totaux[j] else float(i == j) for j in etats} for i in etats}
    if bug == "total":
        total = sum(sum(ligne.values()) for ligne in comptes.values())
        return {i: {j: comptes[i][j] / total if total else float(i == j) for j in etats} for i in etats}
    if bug == "occurrences":         # divisé par le nombre de relevés de l'état, pas par ses passages
        occurrences = {i: 0 for i in etats}
        for _, _, etat in dict.fromkeys(releves):
            occurrences[etat] += 1
        return {i: {j: comptes[i][j] / occurrences[i] if occurrences[i] else float(i == j) for j in etats}
                for i in etats}
    matrice = {}
    for i in etats:
        total = sum(comptes[i].values())
        if total:
            matrice[i] = {j: comptes[i][j] / total for j in etats}
        elif bug == "vide":
            matrice[i] = dict.fromkeys(etats, 0.0)
        elif bug == "uniforme":
            matrice[i] = dict.fromkeys(etats, 1 / len(etats))
        else:
            matrice[i] = {j: float(i == j) for j in etats}
    return matrice


_BUGS_COMPTAGE = ("doublons", "ordre", "frontiere", "trous", "decembre")


def _ref_matrice(releves, etats=ETATS, bug: str = "") -> dict:
    releves = list(releves)
    etats = list(etats)
    passages = _ref_passages(releves, etats, bug if bug in _BUGS_COMPTAGE else "")
    return _ref_normaliser(_ref_compter(passages, etats), etats, "" if bug in _BUGS_COMPTAGE else bug, releves)


_BUGS_MATRICE = {
    "frontiere": "Vous comptez un passage entre le DERNIER relevé d'un client et le PREMIER du client suivant (après un "
                 "tri par client puis par mois, deux lignes voisines ne sont pas forcément le même client). Un passage, "
                 "c'est UN client, deux mois consécutifs : regroupez par client d'abord (un dict {client: {mois: état}}).",
    "trous": "Vous comptez un passage entre deux relevés voisins d'un client même quand un mois MANQUE entre eux (la panne "
             "d'août : juillet → septembre). On ne sait pas ce qui s'est passé en août : ne comptez que les mois qui se "
             "SUIVENT (`mois_du_client.get(numero + 1)`).",
    "decembre": "Décembre et janvier ne se suivent pas chez vous : `int('2026-01'.replace('-', ''))` vaut 202601, et "
                "202512 + 1 = 202513. Numérotez les mois de façon continue : `annee * 12 + mois - 1` (décembre 2025 → "
                "24311, janvier 2026 → 24312).",
    "ordre": "Vous prenez les relevés de chaque client dans l'ordre où ils ARRIVENT : ils sortent d'une requête sans "
             "ORDER BY, dans le désordre. Rangez-les par mois (un dict {numéro du mois: état} par client), puis "
             "cherchez le mois suivant.",
    "doublons": "Un relevé chargé DEUX fois compte deux fois chez vous (c'est ce que fait une jointure de la table avec "
                "elle-même) : la ligne de son état penche vers ce qu'il est devenu. Un doublon exact ne compte qu'une "
                "fois (un dict {mois: état} par client le règle tout seul).",
    "colonnes": "Vous divisez par le total de chaque COLONNE : c'est chaque LIGNE qui fait 100 % (un client actif ce "
                "mois-ci est forcément quelque part le mois suivant). matrice[i][j] = passages(i → j) / passages "
                "depuis i.",
    "total": "Vous divisez par le nombre TOTAL de passages : la matrice entière fait 100 %, au lieu de chaque ligne. "
             "matrice[i][j] = passages(i → j) / (passages DEPUIS i).",
    "occurrences": "Vous divisez par le nombre de RELEVÉS de l'état i, pas par le nombre de passages depuis i : le "
                   "dernier relevé d'un client (et celui qui précède un mois manquant) n'a pas de mois suivant, il ne "
                   "doit pas compter. Vos lignes font moins de 100 %.",
    "vide": "Un état jamais quitté (aucun passage depuis lui, comme « Parti ») donne chez vous une ligne de zéros. Un "
            "client ne disparaît pas : sa règle est 1.0 vers lui-même, 0.0 ailleurs.",
    "uniforme": "Un état jamais quitté (aucun passage depuis lui) donne chez vous une ligne répartie au hasard (1 / "
                "nombre d'états) : un client parti redeviendrait fidèle ! Sa règle : 1.0 vers lui-même, 0.0 ailleurs.",
}


def _cas_254() -> list[tuple[list, dict]]:
    cas = [
        ([("C1", "2025-11", "Actif"), ("C1", "2025-12", "Actif"), ("C1", "2026-01", "À risque"),
          ("C2", "2025-12", "Actif"), ("C2", "2026-02", "Parti")], ETATS),
        ([("A", "2026-01", "Actif"), ("A", "2025-12", "Fidèle"), ("B", "2025-12", "Actif"),
          ("B", "2026-01", "Actif")], ETATS),                                         # décembre → janvier
        ([("A", "2025-07", "Actif"), ("A", "2025-09", "Parti"), ("B", "2025-07", "Actif"),
          ("B", "2025-08", "Actif")], ETATS),                                         # un trou en août
        ([("A", "2026-01", "Fidèle"), ("B", "2026-02", "À risque"), ("B", "2026-03", "Parti"),
          ("C", "2026-01", "Fidèle"), ("C", "2026-02", "Fidèle")], ETATS),            # A puis B : pas un passage
        ([("A", "2026-01", "Actif"), ("A", "2026-01", "Actif"), ("A", "2026-02", "Actif"),
          ("B", "2026-01", "Actif"), ("B", "2026-02", "Parti")], ETATS),              # un doublon exact
        ([("A", "2026-02", "Actif"), ("A", "2026-01", "Fidèle"), ("B", "2026-03", "Actif"),
          ("B", "2026-02", "Actif"), ("B", "2026-01", "À risque")], ETATS),           # dans le désordre
        ([], ETATS),
        ([("X", "2025-12", "Mensuel"), ("X", "2026-01", "Annuel"), ("Y", "2025-12", "Mensuel"),
          ("Y", "2026-01", "Résilié"), ("Z", "2025-11", "Annuel"), ("Z", "2025-12", "Annuel")],
         ("Mensuel", "Annuel", "Résilié")),
        ([(7, "2024-12", "b"), (7, "2025-01", "a"), (8, "2024-12", "a"), (8, "2025-01", "a"), (8, "2025-02", "b")],
         ["a", "b"]),                                                                  # des clients numérotés
    ]
    for graine in range(6):
        releves, _ = generer_releves(40 + 10 * graine, graine=graine, debut="2025-06", fin="2026-03",
                                     manquants=0.08, doublons=0.05)
        cas.append((releves, ETATS))
    return [(releves, {"etats": etats}) for releves, etats in cas]


def _verifier_forme_matrice(r, etats, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie {type(r).__name__} : attendu un dict {{depuis: {{vers: part}}}}.")
    if set(r) != set(etats):
        manque = [e for e in etats if e not in r]
        raise AssertionError(f"{quoi} a pour lignes {_court(list(r), 100)} : attendu une ligne par état de `etats` "
                             f"({_court(list(etats), 100)})" + (f" ; il manque {manque} (un état jamais vu a quand même "
                                                                 "sa ligne)." if manque else "."))
    for i, ligne in r.items():
        if not isinstance(ligne, dict) or set(ligne) != set(etats):
            raise AssertionError(f"{quoi} : la ligne {i!r} vaut {_court(ligne, 100)} : chaque ligne est un dict avec "
                                 "TOUS les états de `etats` (0.0 pour ceux qu'on n'atteint pas).")
        for j, p in ligne.items():
            if not _nombre(p):
                raise AssertionError(f"{quoi} : [{i!r}][{j!r}] vaut {_court(p, 40)} : attendu un nombre (float).")
    if list(r) != list(etats) or any(list(ligne) != list(etats) for ligne in r.values()):
        raise AssertionError(f"{quoi} : les états ne sont pas dans l'ordre de `etats` ({_court(list(etats), 100)}). "
                             "Construisez lignes et colonnes en parcourant `etats`.")


def _quoi(nom: str, releves, kwargs: dict) -> str:
    options = "".join(f", {k}={v!r}" for k, v in kwargs.items() if k != "segments" and not (k == "etats" and v == ETATS))
    return f"{nom}(<{len(releves)} relevés>{', segments' if 'segments' in kwargs else ''}{options})"


def _verifier_releves(f, cas, forme, ref, bugs, nom: str, egal, ecart) -> None:
    """Appelle f sur chaque cas (relevés passés en flux), vérifie la forme, puis compare à la référence."""
    obtenus = []
    for releves, kwargs in cas:
        quoi = _quoi(nom, releves, kwargs)
        r = _appel(f, (UneSeuleFois(releves),), quoi, kwargs=kwargs)
        forme(r, kwargs, quoi)
        obtenus.append(r)
    for (releves, kwargs), r in zip(cas, obtenus):
        attendu = ref(releves, kwargs, "")
        if not egal(r, attendu):
            conseil = _premier_bug(obtenus, lambda bug: [ref(rel, kw, bug) for rel, kw in cas], bugs, _listes(egal))
            montre = sorted(releves, key=lambda x: (str(x[0]), str(x[1])))
            raise AssertionError(f"{_quoi(nom, releves, kwargs)}, sur les relevés {_court(montre, 260)} : "
                                 f"{ecart(r, attendu)}. {conseil}".rstrip())


def _segments_proches(a, b) -> bool:
    return isinstance(a, dict) and a.keys() == b.keys() and all(_matrices_proches(a[k], b[k]) for k in b)


def _ecart_segments(r, attendu) -> str:
    for segment, matrice in attendu.items():
        if not _matrices_proches(r.get(segment), matrice):
            return f"pour le segment {segment!r}, {_ecart(r.get(segment) or {}, matrice)}"
    return ""


def _lignes_releves(n: int) -> int:
    return 60 * n + 20_000


def verifier_matrice_transition(f) -> None:
    _verifier_releves(f, _cas_254(), lambda r, kw, quoi: _verifier_forme_matrice(r, kw.get("etats", ETATS), quoi),
                      lambda releves, kw, bug: _ref_matrice(releves, kw.get("etats", ETATS), bug), _BUGS_MATRICE,
                      "matrice_transition", _matrices_proches, _ecart)

    # ------------------------------------------------------------------ les relevés de l'entrepôt, en flux
    releves = releves_exemple()
    r = _appel(f, (UneSeuleFois(releves),), "matrice_transition(un flux des 48 434 relevés de l'exemple)",
               lignes_max=_lignes_releves(len(releves)),
               conseil_lent="Pour chaque relevé, chercher le mois suivant dans TOUS les relevés fait 48 434 × 48 434 "
                            "comparaisons. Rangez-les d'abord dans un dict {client: {mois: état}} : la recherche du mois "
                            "suivant devient `mois_du_client.get(numero + 1)`.")
    _verifier_forme_matrice(r, ETATS, "matrice_transition(les relevés de l'exemple)")
    attendu = _ref_matrice(releves)
    if not _matrices_proches(r, attendu):
        conseil = _premier_bug(r, lambda bug: _ref_matrice(releves, ETATS, bug), _BUGS_MATRICE, _matrices_proches)
        raise AssertionError(f"Sur les relevés de l'exemple, matrice_transition donne {_ecart(r, attendu)}. "
                             f"{conseil}".rstrip())

    # ------------------------------------------------------------------ ce qu'il faut refuser
    base = [("A", "2026-01", "Actif"), ("A", "2026-02", "Fidèle")]
    for releves, conseil in (
            (base + [("B", "2026-01", "Suspendu")], "« Suspendu » n'est pas dans `etats` : ValueError (une matrice "
                                                    "qui l'oublierait en silence perdrait ces clients)."),
            (base + [("B", "2025-13", "Actif")], "« 2025-13 » n'est pas un mois."),
            (base + [("B", "2025-00", "Actif")], "« 2025-00 » n'est pas un mois."),
            (base + [("B", "2025-1", "Actif")], "« 2025-1 » n'est pas au format AAAA-MM."),
            (base + [("B", "202512", "Actif")], "« 202512 » n'est pas au format AAAA-MM."),
            (base + [("B", 202512, "Actif")], "202512 (un int) n'est pas un mois « AAAA-MM »."),
            (base + [("B", "2025/12", "Actif")], "« 2025/12 » n'est pas au format AAAA-MM."),
            (base + [("A", "2026-01", "À risque")], "A est « Actif » ET « À risque » en janvier 2026 : lequel croire ? "
                                                    "Deux relevés contradictoires → ValueError (un doublon EXACT, lui, "
                                                    "est accepté).")):
        _doit_lever(f, (releves,), f"matrice_transition({_court(releves, 120)})", conseil, ValueError)


# ============================================================================ défi 255 : projection


def _ref_projection(depart, matrice, mois, bug: str = "") -> list[dict]:
    etats = list(matrice)
    courant = {e: float(depart.get(e, 0)) for e in etats}
    if bug == "proportions":
        total = sum(courant.values()) or 1.0
        courant = {e: n / total for e, n in courant.items()}
    resultat = [] if bug == "sans_depart" else [dict(courant)]
    for _ in range(mois + (1 if bug == "un_de_trop" else 0)):
        suivant = dict.fromkeys(etats, 0.0)
        for i in etats:
            for j in etats:
                if bug == "transpose":
                    suivant[i] += matrice[i][j] * courant[j]
                else:
                    suivant[j] += courant[i] * matrice[i][j]
        if bug == "arrondi":
            suivant = {e: float(round(n)) for e, n in suivant.items()}
        courant = suivant
        resultat.append(dict(courant))
    return resultat


_BUGS_PROJECTION = {
    "transpose": "Votre matrice est prise dans le mauvais sens : vous calculez `Σ_j matrice[i][j] × effectifs[j]` (la "
                 "matrice fois un vecteur COLONNE). Le mois suivant, l'état j REÇOIT des clients de chaque état i : "
                 "`vers[j] += effectifs[i] * matrice[i][j]`. Avec NumPy : `effectifs @ P`, pas `P @ effectifs`.",
    "sans_depart": "Il manque le mois 0 : la liste commence par le DÉPART (complété par des 0.0), puis les `mois` mois "
                   "suivants. `mois` + 1 éléments.",
    "un_de_trop": "Vous faites un mois de trop : la liste a `mois` + 1 éléments, du mois 0 (le départ) au mois `mois`.",
    "proportions": "Vous rendez des PARTS (qui font 1 au total) : la règle demande des nombres de CLIENTS. Partez de "
                   "`depart` tel quel (1 000 clients restent 1 000 clients répartis autrement).",
    "arrondi": "Vous arrondissez chaque mois à un nombre entier de clients : ces arrondis s'accumulent, et au bout d'un "
               "an le résultat est faux (5 % de 3 clients, c'est 0,15 client, pas 0). Gardez des float ; arrondissez à "
               "l'affichage seulement.",
}


def _cas_255() -> list[tuple]:
    me = matrice_exemple()
    cas = [
        ({"Actif": 1000}, me, 12),
        ({"Fidèle": 3000, "Actif": 5500, "À risque": 1500}, me, 24),
        ({"À risque": 7}, me, 5),
        ({}, me, 3),
        ({"Actif": 10, "Parti": 4}, me, 0),
        ({"Actif": 2.5, "Fidèle": 0.5}, me, 7),
        ({"Mensuel": 600, "Annuel": 400},
         {"Mensuel": {"Mensuel": 0.8, "Annuel": 0.15, "Résilié": 0.05},
          "Annuel": {"Mensuel": 0.02, "Annuel": 0.96, "Résilié": 0.02},
          "Résilié": {"Mensuel": 0.0, "Annuel": 0.0, "Résilié": 1.0}}, 6),
    ]
    alea = random.Random(255)
    for k in range(12):
        etats = [f"E{n}" for n in range(alea.randrange(2, 6))]
        matrice = _complete(etats, alea=alea)
        depart = {e: alea.randrange(0, 50) for e in etats if alea.random() < 0.7}
        cas.append((depart, matrice, alea.randrange(1, 15)))
    return cas


def _verifier_forme_projection(r, matrice, mois: int, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie {type(r).__name__} : attendu une LISTE de dict (un par mois).")
    for t, etape in enumerate(r):
        if not isinstance(etape, dict) or set(etape) != set(matrice):
            raise AssertionError(f"{quoi} : l'élément {t} vaut {_court(etape, 100)} : chaque mois est un dict avec "
                                 f"TOUS les états de la matrice ({_court(list(matrice), 80)}), 0.0 compris (le départ "
                                 "aussi : complétez-le).")
        if not all(_nombre(n) for n in etape.values()):
            raise AssertionError(f"{quoi} : l'élément {t} vaut {_court(etape, 100)} : attendu des nombres de clients.")
        if list(etape) != list(matrice):
            raise AssertionError(f"{quoi} : les états du mois {t} ne sont pas dans l'ordre de la matrice.")


def verifier_projection(f) -> None:
    cas = _cas_255()
    obtenus = []
    for depart, matrice, mois in cas:
        quoi = f"projection({_court(depart, 70)}, <matrice {_court(list(matrice), 50)}>, {mois})"
        copie = dict(depart)
        r = _appel(f, (depart, matrice, mois), quoi)
        _verifier_forme_projection(r, matrice, mois, quoi)
        if depart != copie or list(depart) != list(copie):
            raise AssertionError(f"{quoi} a modifié `depart` : il doit rester intact (partez d'une copie).")
        if any(etape is depart for etape in r):
            raise AssertionError(f"{quoi} renvoie `depart` lui-même comme mois 0 : si l'appelant modifie ce mois, il "
                                 "modifie son départ. Le mois 0 est une COPIE complétée par des 0.0.")
        if len({id(etape) for etape in r}) != len(r):
            raise AssertionError(f"{quoi} : plusieurs éléments de votre liste sont le MÊME dict (vous l'ajoutez, puis "
                                 "vous le modifiez le mois suivant : toute la liste montre le dernier mois). Créez un "
                                 "NOUVEAU dict à chaque mois (`suivant = dict.fromkeys(matrice, 0.0)`).")
        obtenus.append(r)
    egal = _listes(_dicts_proches)
    for (depart, matrice, mois), r in zip(cas, obtenus):
        attendu = _ref_projection(depart, matrice, mois)
        if not egal(r, attendu):
            conseil = _premier_bug(obtenus, lambda bug: [_ref_projection(*c, bug) for c in cas], _BUGS_PROJECTION,
                                   _listes(egal))
            quoi = f"projection({_court(depart, 70)}, <matrice {_court(list(matrice), 50)}>, {mois})"
            if len(r) != len(attendu):
                raise AssertionError(f"{quoi} renvoie {len(r)} mois au lieu de {len(attendu)} (du mois 0 au mois "
                                     f"{mois}). {conseil}".rstrip())
            t = next(t for t, (a, b) in enumerate(zip(r, attendu)) if not _dicts_proches(a, b))
            raise AssertionError(f"{quoi} : au mois {t}, {_ecart(r[t], attendu[t])}. {conseil}".rstrip())

    # ------------------------------------------------------------------ trois ans… puis 250 ans
    me = matrice_exemple()
    r = _appel(f, ({"Actif": 1000}, me, 3000), "projection({'Actif': 1000}, matrice_exemple(), 3000)",
               lignes_max=1_500_000,
               conseil_lent="Chaque mois se calcule à partir du mois PRÉCÉDENT (une étape), pas en recalculant depuis "
                            "le départ (t étapes au mois t : 4,5 millions d'étapes pour 3 000 mois).")
    if len(r) != 3001 or not _dicts_proches(r[-1], _ref_projection({"Actif": 1000}, me, 3000)[-1], 1e-6):
        raise AssertionError("Sur 3 000 mois, le dernier mois de votre projection est faux.")

    # ------------------------------------------------------------------ ce qu'il faut refuser
    for args, attendu, conseil in (
            (({"Suspendu": 5}, me, 3), ValueError, "« Suspendu » n'est pas un état de la matrice."),
            (({"Actif": -5}, me, 3), ValueError, "Un nombre de clients ne peut pas être négatif."),
            (({"Actif": 5}, me, -1), ValueError, "Un nombre de mois négatif n'a pas de sens."),
            (({"Actif": 5}, me, 2.0), TypeError, "`mois` est un int (range(2.0) n'existe pas)."),
            (({"Actif": 5}, me, True), TypeError, "True n'est pas un nombre de mois (c'est un bool)."),
            (({"Actif": "5"}, me, 3), TypeError, "« 5 » est un texte, pas un nombre de clients."),
            (({"Actif": True}, me, 3), TypeError, "True n'est pas un nombre de clients.")):
        _doit_lever(f, args, f"projection({_court(args[0], 40)}, matrice_exemple(), {args[2]!r})", conseil, attendu)


# ============================================================================ défi 256 : duree_de_vie


def _resoudre(a, b):
    n = len(b)
    lignes = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda k: abs(lignes[k][col]))
        lignes[col], lignes[pivot] = lignes[pivot], lignes[col]
        for k in range(col + 1, n):
            facteur = lignes[k][col] / lignes[col][col]
            if facteur:
                for c in range(col, n + 1):
                    lignes[k][c] -= facteur * lignes[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (lignes[i][n] - sum(lignes[i][c] * x[c] for c in range(i + 1, n))) / lignes[i][i]
    return x


def _ref_duree(matrice, absorbants=(PARTI,), bug: str = "") -> dict:
    if isinstance(absorbants, str):
        absorbants = (absorbants,)
    absorbants = set(absorbants)
    transitoires = [e for e in matrice if e not in absorbants]
    if bug == "geometrique":
        return {i: 1 / sum(matrice[i][a] for a in absorbants) for i in transitoires}
    if bug == "tronque":
        t = dict.fromkeys(transitoires, 0.0)
        for _ in range(1000):
            t = {i: 1 + sum(matrice[i][j] * t[j] for j in transitoires) for i in transitoires}
        return t
    if bug == "transpose":
        a = [[float(i == j) - matrice[j][i] for j in transitoires] for i in transitoires]
    else:
        a = [[float(i == j) - matrice[i][j] for j in transitoires] for i in transitoires]
    t = dict(zip(transitoires, _resoudre(a, [1.0] * len(transitoires))))
    if bug == "sans_mois_courant":
        return {i: v - 1 for i, v in t.items()}
    if bug == "avec_absorbants":
        return {e: t.get(e, 0.0) for e in matrice}
    return t


_BUGS_DUREE = {
    "geometrique": "Vous calculez 1 / (taux de départ de l'état) : c'est juste quand il n'y a qu'UN état. Ici, un client "
                   "à risque part à 20 % par mois (5 mois), mais un sur quatre redevient actif et vit alors bien plus "
                   "longtemps : sa durée dépend de celle des autres états. Écrivez t(i) = 1 + Σ_j P[i][j] × t(j) pour "
                   "tous les états transitoires, et résolvez ce système.",
    "sans_mois_courant": "Vos durées sont trop courtes de 1 : vous oubliez le mois EN COURS. Un client qui part à coup "
                         "sûr le mois prochain compte 1 mois (pas 0) : t(i) = 1 + Σ_j P[i][j] × t(j), soit "
                         "(I − Q) · t = 1.",
    "transpose": "Votre système est pris dans le mauvais sens : la ligne de l'état i contient P[i][j] (où VA un client "
                 "parti de i), pas P[j][i]. a[i][j] = (1 si i == j sinon 0) − matrice[i][j].",
    "tronque": "Vous additionnez mois après mois, et vous vous arrêtez trop tôt (au bout de 1 000 mois) : un client qui "
               "part avec une chance sur 10 000 par mois reste 10 000 mois en moyenne. Ne bouclez pas : résolvez le "
               "système (I − Q) · t = 1 (`numpy.linalg.solve`, ou Gauss).",
    "avec_absorbants": "Votre réponse contient les états ABSORBANTS : elle n'a une clé que pour les états transitoires "
                       "(un client parti n'a pas de durée de vie à venir).",
}


def _cas_256() -> list[tuple]:
    me = matrice_exemple()
    cas = [
        (me, (PARTI,)),
        ({"Actif": {"Actif": 0.9, "Parti": 0.1}, "Parti": {"Actif": 0.0, "Parti": 1.0}}, ("Parti",)),
        ({"A": {"A": 0.0, "P": 1.0}, "P": {"A": 0.0, "P": 1.0}}, ("P",)),
        ({"A": {"A": 0.0, "B": 1.0, "P": 0.0}, "B": {"A": 0.0, "B": 0.5, "P": 0.5},
          "P": {"A": 0.0, "B": 0.0, "P": 1.0}}, ("P",)),
        ({"Résilié": {"Résilié": 1.0, "Actif": 0.0, "Impayé": 0.0, "Relance": 0.0},
          "Actif": {"Résilié": 0.01, "Actif": 0.95, "Impayé": 0.0, "Relance": 0.04},
          "Impayé": {"Résilié": 0.0, "Actif": 0.0, "Impayé": 1.0, "Relance": 0.0},
          "Relance": {"Résilié": 0.05, "Actif": 0.6, "Impayé": 0.25, "Relance": 0.1}}, ("Résilié", "Impayé")),
        (me, "Parti"),                                       # un texte seul : UN état
        ({"Fidèle": {"Fidèle": 0.9999, "Parti": 0.0001}, "Parti": {"Fidèle": 0.0, "Parti": 1.0}}, [PARTI]),
        ({"Parti": {"Parti": 1.0, "Actif": 0.0, "Fidèle": 0.0},
          "Actif": {"Parti": 0.02, "Actif": 0.9, "Fidèle": 0.08},
          "Fidèle": {"Parti": 0.0001, "Actif": 0.05, "Fidèle": 0.9499}}, {PARTI}),
    ]
    alea = random.Random(256)
    for k in range(10):
        etats = [f"E{n}" for n in range(alea.randrange(2, 9))]
        absorbants = tuple(alea.sample(etats[1:], alea.randrange(1, min(3, len(etats)))))
        cas.append((_complete(etats, absorbants, alea), absorbants))
    return cas


def _verifier_forme_duree(r, matrice, absorbants, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie {type(r).__name__} : attendu un dict {{état: mois}}.")
    abs_ = {absorbants} if isinstance(absorbants, str) else set(absorbants)
    transitoires = [e for e in matrice if e not in abs_]
    if any(e in abs_ for e in r) and set(r) - abs_ == set(transitoires):
        raise AssertionError(f"{quoi} contient les états absorbants {sorted(set(r) & abs_)} : la réponse n'a une clé "
                             "que pour les états TRANSITOIRES (un client parti n'a plus de durée de vie).")
    if set(r) != set(transitoires):
        raise AssertionError(f"{quoi} a pour clés {_court(list(r), 100)} au lieu des états transitoires "
                             f"{transitoires}." + (" Un texte seul (« Parti ») désigne UN état : `set('Parti')` donne "
                                                    "cinq lettres." if isinstance(absorbants, str) else ""))
    if not all(_nombre(v) for v in r.values()):
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : attendu des nombres de mois (float).")


def verifier_duree_de_vie(f) -> None:
    cas = _cas_256()
    obtenus = []
    lent = ("Une boucle « mois après mois » qui attend que le résultat ne bouge plus fait des centaines de milliers de "
            "tours pour un client qui reste 10 000 mois. Résolvez le système (I − Q) · t = 1 en une fois : "
            "`numpy.linalg.solve(a, b)`, ou l'élimination de Gauss.")
    for matrice, absorbants in cas:
        quoi = f"duree_de_vie(<matrice {_court(list(matrice), 60)}>, {absorbants!r})"
        conseil = ("Un texte seul (« Parti ») désigne UN état absorbant : `set('Parti')` donne cinq lettres "
                   "({'P', 'a', 'r', 't', 'i'}). Écrivez `if isinstance(absorbants, str): absorbants = (absorbants,)`."
                   if isinstance(absorbants, str) else "")
        r = _appel(f, (matrice, absorbants), quoi, conseil, lignes_max=200_000, conseil_lent=lent,
                   division="Une division par zéro : depuis cet état, on ne part pas DIRECTEMENT (probabilité 0 vers "
                            "l'absorbant), mais on part en passant par un autre état. 1 / taux de départ ne marche "
                            "qu'avec un seul état : écrivez t(i) = 1 + Σ_j P[i][j] × t(j) pour tous les états "
                            "transitoires, et résolvez ce système.")
        _verifier_forme_duree(r, matrice, absorbants, quoi)
        obtenus.append(r)
    egal = (lambda a, b: _dicts_proches(a, b, 1e-7))
    for (matrice, absorbants), r in zip(cas, obtenus):
        attendu = _ref_duree(matrice, absorbants)
        if not egal(r, attendu):
            conseil = _premier_bug(obtenus, lambda bug: [_ref_duree(m, a, bug) for m, a in cas], _BUGS_DUREE,
                                   _listes(egal))
            raise AssertionError(f"duree_de_vie(<matrice {_court(list(matrice), 60)}>, {absorbants!r}) : "
                                 f"{_ecart(r, attendu, 1e-7)}. {conseil}".rstrip())

    # ------------------------------------------------------------------ ce qu'il faut refuser
    me = matrice_exemple()
    for args, conseil in (
            ((me, ("Suspendu",)), "« Suspendu » n'est pas un état de la matrice."),
            ((me, ("Actif",)), "« Actif » n'est pas absorbant : on le quitte (matrice['Actif']['Actif'] vaut 0.9, pas "
                               "1). Vérifiez que chaque absorbant a 1.0 sur lui-même."),
            ((me, ("Parti", "À risque")), "« À risque » n'est pas absorbant.")):
        _doit_lever(f, args, f"duree_de_vie(matrice_exemple(), {args[1]!r})", conseil, ValueError)
    bloquee = {"Fidèle": {"Fidèle": 0.95, "Actif": 0.05, "À risque": 0.0, "Parti": 0.0},
               "Actif": {"Fidèle": 0.06, "Actif": 0.94, "À risque": 0.0, "Parti": 0.0},
               "À risque": {"Fidèle": 0.0, "Actif": 0.3, "À risque": 0.5, "Parti": 0.2},
               "Parti": {"Fidèle": 0.0, "Actif": 0.0, "À risque": 0.0, "Parti": 1.0}}
    exc = _doit_lever(f, (bloquee, (PARTI,)), "duree_de_vie(<les clients Pro, sans lissage>)",
                      "Depuis « Fidèle » et « Actif », on ne passe jamais à risque, donc on ne part jamais : durée "
                      "INFINIE. Un résultat énorme ou négatif (le système n'a pas de solution, NumPy le dit parfois, "
                      "parfois non) serait faux : vérifiez que chaque état peut atteindre un absorbant.", ValueError)
    if "Fidèle" not in str(exc) or "Actif" not in str(exc):
        raise AssertionError(f"Votre ValueError dit {str(exc)!r} : elle doit NOMMER les états d'où l'on ne part jamais "
                             "(« Fidèle » et « Actif »), pour qu'on sache quoi corriger.")


# ============================================================================ défi 257 : valeur_client


def _ref_taux(taux_annuel: float, bug: str = "") -> float:
    if bug == "taux_simple":
        return taux_annuel / 12
    if bug == "annuel":
        return taux_annuel
    if bug == "sans_actualisation":
        return 0.0
    return (1 + taux_annuel) ** (1 / 12) - 1


def _ref_valeur(matrice, marges, taux_annuel, horizon, bug: str = "") -> dict:
    etats = list(matrice)
    m = {e: float(marges.get(e, 0.0)) for e in etats}
    if bug == "tableur":
        absorbants = [a for a in etats if matrice[a][a] == 1.0]
        return {i: m[i] / sum(matrice[i][a] for a in absorbants) for i in etats}
    facteur = 1 / (1 + _ref_taux(taux_annuel, bug))
    tours = horizon + {"horizon_plus_un": 1, "horizon_moins_un": -1}.get(bug, 0)
    v = dict.fromkeys(etats, 0.0)
    for _ in range(max(tours, 0)):
        if bug == "transpose":
            v = {i: m[i] + facteur * sum(matrice[j][i] * v[j] for j in etats) for i in etats}
        elif bug == "actualise_mois0":
            v = {i: facteur * (m[i] + sum(matrice[i][j] * v[j] for j in etats)) for i in etats}
        else:
            v = {i: m[i] + facteur * sum(matrice[i][j] * v[j] for j in etats) for i in etats}
    return v


_BUGS_VALEUR = {
    "taux_simple": "Votre taux mensuel est taux_annuel / 12 : 10 % / 12 = 0,833 % par mois font 10,47 % sur un an. Le "
                   "taux mensuel ÉQUIVALENT est (1 + taux_annuel) ** (1 / 12) - 1 (0,797 % pour 10 %).",
    "annuel": "Vous actualisez chaque MOIS au taux ANNUEL (10 % par mois, c'est 214 % par an) : passez au taux mensuel "
              "équivalent, (1 + taux_annuel) ** (1 / 12) - 1.",
    "sans_actualisation": "Vous n'actualisez pas : un euro touché dans dix ans compte comme un euro aujourd'hui. Chaque "
                          "mois futur est divisé par (1 + r) une fois de plus (r : le taux mensuel équivalent).",
    "actualise_mois0": "Vous actualisez aussi le mois EN COURS : la marge de ce mois-ci est touchée maintenant, elle "
                       "n'est pas divisée par (1 + r). V(i) = marge(i) + Σ_j P[i][j] × V(j) / (1 + r).",
    "horizon_plus_un": "Vous comptez un mois de trop : `horizon` mois, c'est les mois 0 à horizon − 1 (12 mois à 100 €, "
                       "sans actualiser, font 1 200 €).",
    "horizon_moins_un": "Vous comptez un mois de moins : `horizon` mois, c'est les mois 0 à horizon − 1 (12 mois à "
                        "100 €, sans actualiser, font 1 200 €).",
    "transpose": "Votre matrice est prise dans le mauvais sens : la valeur de i dépend des états où un client de i VA "
                 "(`matrice[i][j]`, la ligne de i), pas de ceux d'où viennent les clients de i (`matrice[j][i]`).",
    "tableur": "Vous appliquez la formule du tableur (marge / taux de départ) : elle ignore les changements d'état, "
               "l'horizon et l'actualisation. La valeur se calcule à rebours, mois par mois : V = marge + P · V / (1 + r).",
}


def _cas_257() -> list[tuple]:
    me = matrice_exemple()
    seul = {"A": {"A": 1.0}}
    cas = [
        (me, MARGES, TAUX_ANNUEL, 120),
        (seul, {"A": 100}, 0.10, 12),
        (seul, {"A": 100}, 0.0, 12),
        (me, MARGES, 0.0, 36),
        (me, MARGES, 0.10, 0),
        (me, MARGES, 0.10, 1),
        (me, {"Actif": 22}, 0.05, 24),
        (me, {"Fidèle": 30, "Actif": 22.5, "À risque": 9, "Parti": -3}, 0.2, 60),
        ({"Mensuel": {"Mensuel": 0.8, "Annuel": 0.15, "Résilié": 0.05},
          "Annuel": {"Mensuel": 0.02, "Annuel": 0.96, "Résilié": 0.02},
          "Résilié": {"Mensuel": 0.0, "Annuel": 0.0, "Résilié": 1.0}}, {"Mensuel": 18, "Annuel": 25}, 0.08, 48),
    ]
    alea = random.Random(257)
    for k in range(10):
        etats = [f"E{n}" for n in range(alea.randrange(2, 6))]
        matrice = _complete(etats, alea=alea)
        marges = {e: alea.randrange(-5, 40) for e in etats if alea.random() < 0.8}
        cas.append((matrice, marges, alea.choice((0, 0.03, 0.1, 0.25)), alea.randrange(1, 40)))
    return cas


def _verifier_forme_valeur(r, matrice, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, dict) or set(r) != set(matrice):
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : attendu un dict avec une clé par état de la matrice "
                             f"({_court(list(matrice), 80)}), les absorbants compris.")
    if not all(_nombre(v) for v in r.values()):
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : attendu des montants en euros (float).")


def verifier_valeur_client(f) -> None:
    cas = _cas_257()
    obtenus = []
    for matrice, marges, taux, horizon in cas:
        quoi = f"valeur_client(<matrice {_court(list(matrice), 50)}>, {_court(marges, 70)}, {taux!r}, {horizon!r})"
        copie = dict(marges)
        r = _appel(f, (matrice, marges, taux, horizon), quoi,
                   division="Une division par zéro : vous divisez par un taux de départ qui vaut 0 pour certains états. "
                            "La formule du tableur (marge / taux de départ) ne vaut qu'avec un seul état et sans "
                            "horizon : calculez à rebours, V = marge + P · V / (1 + r), sans aucune division par un "
                            "taux de départ.")
        _verifier_forme_valeur(r, matrice, quoi)
        if marges != copie:
            raise AssertionError(f"{quoi} a modifié `marges` : il doit rester intact.")
        obtenus.append(r)
    for (matrice, marges, taux, horizon), r in zip(cas, obtenus):
        attendu = _ref_valeur(matrice, marges, taux, horizon)
        if not _dicts_proches(r, attendu):
            conseil = _premier_bug(obtenus, lambda bug: [_ref_valeur(*c, bug) for c in cas], _BUGS_VALEUR,
                                   _listes(_dicts_proches))
            raise AssertionError(f"valeur_client(<matrice {_court(list(matrice), 50)}>, {_court(marges, 70)}, {taux!r}, "
                                 f"{horizon!r}) : {_ecart(r, attendu)}. {conseil}".rstrip())

    # ------------------------------------------------------------------ cent ans
    me = matrice_exemple()
    r = _appel(f, (me, MARGES, TAUX_ANNUEL, 1200), "valeur_client(matrice_exemple(), MARGES, TAUX_ANNUEL, 1200)",
               lignes_max=1_000_000,
               conseil_lent="Pour 1 200 mois, recalculer la répartition du mois t à partir du mois 0 (t étapes) fait "
                            "720 000 étapes. À rebours, c'est une seule boucle de 1 200 tours : V = marge + P · V / (1 + r).")
    if not _dicts_proches(r, _ref_valeur(me, MARGES, TAUX_ANNUEL, 1200), 1e-7):
        raise AssertionError("Sur 1 200 mois, valeur_client donne un résultat faux.")

    # ------------------------------------------------------------------ ce qu'il faut refuser
    for args, attendu, conseil in (
            ((me, {"Suspendu": 10}, 0.1, 12), ValueError, "« Suspendu » n'est pas un état de la matrice."),
            ((me, {"Actif": "22"}, 0.1, 12), TypeError, "« 22 » est un texte, pas une marge."),
            ((me, MARGES, -0.1, 12), ValueError, "Un taux d'actualisation négatif est refusé."),
            ((me, MARGES, "10 %", 12), TypeError, "Le taux est un nombre (0.10), pas un texte."),
            ((me, MARGES, True, 12), TypeError, "True n'est pas un taux."),
            ((me, MARGES, 0.1, -1), ValueError, "Un horizon négatif n'a pas de sens."),
            ((me, MARGES, 0.1, 12.0), TypeError, "L'horizon est un nombre ENTIER de mois (un int)."),
            ((me, MARGES, 0.1, True), TypeError, "True n'est pas un horizon.")):
        _doit_lever(f, args, f"valeur_client(matrice_exemple(), {_court(args[1], 60)}, {args[2]!r}, {args[3]!r})",
                    conseil, attendu)


# ============================================================================ défi 258 : delai_rentabilite


def _ref_cumuls(matrice, marges, etat, taux_annuel, mois: int, bug: str = "") -> list[float]:
    """Le cumul actualisé à la fin des mois 0, 1, …, mois − 1."""
    etats = list(matrice)
    m = {e: float(marges.get(e, 0.0)) for e in etats}
    facteur = 1 / (1 + _ref_taux(taux_annuel, bug))
    repartition = dict.fromkeys(etats, 0.0)
    repartition[etat] = 1.0
    cumul, actualisation, cumuls = 0.0, facteur if bug == "actualise_mois0" else 1.0, []
    for _ in range(mois):
        cumul += actualisation * sum(repartition[e] * m[e] for e in etats)
        cumuls.append(cumul)
        actualisation *= facteur
        if bug != "cohorte_figee":
            suivant = dict.fromkeys(etats, 0.0)
            for i in etats:
                for j in etats:
                    suivant[j] += repartition[i] * matrice[i][j]
            repartition = suivant
    return cumuls


def _ref_delai(matrice, marges, etat, cout, taux_annuel, horizon_max=120, bug: str = ""):
    if cout == 0:
        return 0
    for t, cumul in enumerate(_ref_cumuls(matrice, marges, etat, taux_annuel, horizon_max, bug)):
        if cumul > cout if bug == "strict" else cumul >= cout:
            return t if bug == "zero_base" else t + 1
    return horizon_max if bug == "horizon" else None


_BUGS_DELAI = {
    "strict": "Vous attendez que le cumul DÉPASSE le coût (`>`) : un cumul ÉGAL au coût suffit, le client a remboursé ce "
              "qu'il a coûté (`>=`). 10 + 10 + 10 = 30 : un coût de 30 € est couvert au bout de 3 mois.",
    "zero_base": "Votre réponse est trop petite de 1 : c'est un nombre de MOIS (1 si le mois 0 suffit), pas l'indice du "
                 "dernier mois. Renvoyez t + 1.",
    "horizon": "Quand le coût n'est jamais couvert, vous renvoyez `horizon_max` : c'est une vraie réponse (« couvert "
               "au bout de 120 mois »), donc un mensonge. Jamais couvert → None.",
    "sans_actualisation": "Vous n'actualisez pas : chaque mois futur doit être divisé par (1 + r) une fois de plus que "
                          "le précédent (r : le taux mensuel équivalent, (1 + taux_annuel) ** (1 / 12) - 1).",
    "actualise_mois0": "Vous actualisez aussi le mois 0 : la marge de ce mois-ci est touchée maintenant (facteur 1), "
                       "celle du mois t est divisée par (1 + r) ** t.",
    "taux_simple": "Votre taux mensuel est taux_annuel / 12 : le taux mensuel ÉQUIVALENT est (1 + taux_annuel) ** "
                   "(1 / 12) - 1 (0,797 % pour 10 %, pas 0,833 %).",
    "annuel": "Vous actualisez chaque MOIS au taux ANNUEL : passez au taux mensuel équivalent, (1 + taux_annuel) ** "
              "(1 / 12) - 1.",
    "cohorte_figee": "Vous comptez la marge de l'état de DÉPART tous les mois : le client change d'état (il peut devenir "
                     "fidèle, passer à risque, ou partir). Faites avancer la répartition d'un mois à chaque tour, comme "
                     "au défi 255.",
}


def _cas_258() -> list[tuple]:
    me = matrice_exemple()
    seul = {"A": {"A": 1.0}}
    moitie = {"A": {"A": 0.5, "B": 0.5}, "B": {"A": 0.0, "B": 1.0}}
    cas = [
        (me, MARGES, "Actif", 180, TAUX_ANNUEL, 120),
        (seul, {"A": 10}, "A", 30, 0.0, 120),                  # 10 + 10 + 10 = 30 : pile
        (seul, {"A": 10}, "A", 30.5, 0.0, 120),
        (moitie, {"A": 8}, "A", 12, 0.0, 120),                 # 8 + 4 = 12 : pile
        (moitie, {"A": 8}, "A", 15.75, 0.0, 120),              # 8 + 4 + 2 + 1 + 0,5 + 0,25 : pile au 6e mois
        (me, MARGES, "Fidèle", 0, TAUX_ANNUEL, 120),
        (me, MARGES, "À risque", 400, TAUX_ANNUEL, 120),
        (seul, {"A": 10}, "A", 120, 0.0, 12),                  # couvert au dernier mois permis
        (seul, {"A": 10}, "A", 130, 0.0, 12),                  # jamais dans les 12 mois
        (me, MARGES, "À risque", 250, TAUX_ANNUEL, 120),
        (me, MARGES, "Fidèle", 500, 0.25, 120),
        (me, {"Actif": 22}, "Actif", 100, 0.05, 60),
        ({"Mensuel": {"Mensuel": 0.8, "Annuel": 0.15, "Résilié": 0.05},
          "Annuel": {"Mensuel": 0.02, "Annuel": 0.96, "Résilié": 0.02},
          "Résilié": {"Mensuel": 0.0, "Annuel": 0.0, "Résilié": 1.0}}, {"Mensuel": 18, "Annuel": 25}, "Mensuel", 150,
         0.08, 120),
    ]
    alea = random.Random(258)
    while len(cas) < 30:
        etats = [f"E{n}" for n in range(alea.randrange(2, 6))]
        matrice = _complete(etats, alea=alea)
        marges = {e: alea.randrange(1, 40) for e in etats}
        taux = alea.choice((0.03, 0.1, 0.25))
        etat = alea.choice(etats)
        cumuls = _ref_cumuls(matrice, marges, etat, taux, 60)
        cout = round(cumuls[alea.randrange(0, 60)] * alea.uniform(0.98, 1.02), 2)
        # aucun cumul à moins d'un millionième du coût : pas d'arrondi qui départage
        if all(abs(c - cout) > 1e-6 * cout for c in cumuls):
            cas.append((matrice, marges, etat, cout, taux, 60))
    return cas


def verifier_delai_rentabilite(f) -> None:
    cas = _cas_258()
    obtenus = []
    for matrice, marges, etat, cout, taux, horizon_max in cas:
        quoi = (f"delai_rentabilite(<matrice {_court(list(matrice), 50)}>, {_court(marges, 60)}, {etat!r}, {cout!r}, "
                f"{taux!r}, horizon_max={horizon_max})")
        r = _appel(f, (matrice, marges, etat, cout, taux), quoi, kwargs={"horizon_max": horizon_max})
        if r is not None and (not _nombre(r) or isinstance(r, float)):
            raise AssertionError(f"{quoi} renvoie {_court(r)} : attendu un nombre ENTIER de mois (un int), ou None.")
        obtenus.append(r)
    for c, r in zip(cas, obtenus):
        attendu = _ref_delai(*c)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_delai(*x, bug=bug) for x in cas], _BUGS_DELAI,
                                   lambda a, b: a == b)
            matrice, marges, etat, cout, taux, horizon_max = c
            raise AssertionError(f"delai_rentabilite(<matrice {_court(list(matrice), 50)}>, {_court(marges, 60)}, "
                                 f"{etat!r}, {cout!r}, {taux!r}, horizon_max={horizon_max}) renvoie {r!r} au lieu de "
                                 f"{attendu!r}. {conseil}".rstrip())

    # ------------------------------------------------------------------ deux cents ans pour rien
    me = matrice_exemple()
    r = _appel(f, (me, MARGES, "À risque", 400, TAUX_ANNUEL), "delai_rentabilite(matrice_exemple(), MARGES, 'À risque', "
               "400, TAUX_ANNUEL, horizon_max=2400)", kwargs={"horizon_max": 2400}, lignes_max=800_000,
               conseil_lent="Vous refaites tout le calcul pour chaque mois (valeur_client(…, horizon=n) pour n = 1, 2, "
                            "3…) : 2 400 mois, c'est près de 3 millions d'étapes. Une seule passe suffit : la répartition "
                            "avance d'un mois, le cumul s'allonge d'un mois.")
    if r is not None:
        raise AssertionError(f"Un client à risque ne rapporte jamais 400 € (388 € au mieux) : delai_rentabilite renvoie "
                             f"{r!r} au lieu de None sur 2 400 mois.")

    # ------------------------------------------------------------------ ce qu'il faut refuser
    for args, kwargs, attendu, conseil in (
            ((me, MARGES, "Suspendu", 100, 0.1), {}, ValueError, "« Suspendu » n'est pas un état de la matrice."),
            ((me, MARGES, "Actif", -5, 0.1), {}, ValueError, "Un coût négatif est refusé."),
            ((me, MARGES, "Actif", "180", 0.1), {}, TypeError, "Le coût est un nombre, pas un texte."),
            ((me, MARGES, "Actif", 180, -0.1), {}, ValueError, "Un taux négatif est refusé."),
            ((me, {"Suspendu": 3}, "Actif", 180, 0.1), {}, ValueError, "« Suspendu » n'est pas un état de la matrice."),
            ((me, MARGES, "Actif", 180, 0.1), {"horizon_max": -1}, ValueError, "Un horizon négatif n'a pas de sens."),
            ((me, MARGES, "Actif", 180, 0.1), {"horizon_max": 12.0}, TypeError, "horizon_max est un int.")):
        _doit_lever(f, args, f"delai_rentabilite(matrice_exemple(), …, {args[2]!r}, {args[3]!r}, {args[4]!r}"
                             f"{', horizon_max=' + repr(kwargs['horizon_max']) if kwargs else ''})", conseil, attendu,
                    kwargs=kwargs)


# ============================================================================ défi 259 : matrices_par_segment


def _ref_segments(releves, segments, etats=ETATS, force=50, bug: str = "") -> dict:
    etats = list(etats)
    releves = list(releves)
    if any(client not in segments for client, _, _ in releves):
        raise ValueError("client sans segment")
    passages = _ref_passages(releves, etats)
    globale = _ref_normaliser(_ref_compter(passages, etats), etats)
    par_segment = {s: [] for s in segments.values()}
    for p in passages:
        par_segment[segments[p[0]]].append(p)
    if bug == "globale_moyenne":
        brutes = {s: _ref_normaliser(_ref_compter(ps, etats), etats) for s, ps in par_segment.items()}
        globale = {i: {j: sum(b[i][j] for b in brutes.values()) / len(brutes) for j in etats} for i in etats}
    resultat = {}
    for segment, siens in par_segment.items():
        comptes = _ref_compter(siens, etats)
        matrice = {}
        for i in etats:
            total = sum(comptes[i].values())
            if bug == "sans_lissage":
                matrice[i] = ({j: comptes[i][j] / total for j in etats} if total else dict(globale[i]))
            elif bug == "moyenne":
                brute = {j: comptes[i][j] / total for j in etats} if total else dict(globale[i])
                matrice[i] = {j: (brute[j] + globale[i][j]) / 2 for j in etats}
            elif bug == "uniforme":
                matrice[i] = ({j: (comptes[i][j] + force / len(etats)) / (total + force) for j in etats}
                              if total + force else dict(globale[i]))
            elif bug == "denominateur":
                matrice[i] = ({j: (comptes[i][j] + force * globale[i][j]) / total for j in etats}
                              if total else dict(globale[i]))
            elif total + force == 0:
                matrice[i] = {j: float(i == j) for j in etats} if bug == "identite" else dict(globale[i])
            else:
                matrice[i] = {j: (comptes[i][j] + force * globale[i][j]) / (total + force) for j in etats}
        resultat[segment] = matrice
    return resultat


_BUGS_SEGMENTS = {
    "sans_lissage": "Vos matrices ne sont pas lissées : `force` ne change rien chez vous. On prête à chaque ligne `force` "
                    "passages répartis comme toute la clientèle : (n_s(i, j) + force × P_globale[i][j]) / (n_s(i) + force).",
    "moyenne": "Vous faites la MOYENNE de la matrice du segment et de la matrice globale (50 / 50) : un segment de 10 000 "
               "passages pèserait autant qu'un segment de 3. Le lissage ajoute `force` pseudo-passages aux VRAIS "
               "passages : (n_s(i, j) + force × P_globale[i][j]) / (n_s(i) + force). Plus le segment a de données, moins "
               "l'emprunt compte.",
    "uniforme": "Vous prêtez les `force` passages répartis ÉGALEMENT entre les états (force / nombre d'états chacun) : un "
                "client parti redeviendrait fidèle ! Prêtez-les comme se comporte toute la clientèle : force × "
                "P_globale[i][j].",
    "denominateur": "Vos lignes font plus de 100 % : vous ajoutez `force` pseudo-passages en haut, mais pas en bas. "
                    "Divisez par n_s(i) + force.",
    "globale_moyenne": "Votre matrice globale est la moyenne des matrices des segments : le segment Pro (30 clients) y "
                       "pèse autant que le segment Essentiel (2 396 clients). La matrice globale se compte sur TOUS les "
                       "passages, ensemble (celle du défi 254).",
    "identite": "Avec force = 0, un état jamais quitté DANS LE SEGMENT donne chez vous 1.0 sur la diagonale : la règle "
                "dit que si n_s(i) + force vaut 0 (rien vu, rien emprunté), la ligne est celle de la matrice GLOBALE.",
}


def _cas_259() -> list[tuple]:
    exemple = [("A", "2026-01", "Actif"), ("A", "2026-02", "Parti"), ("B", "2026-01", "Actif"),
               ("B", "2026-02", "Actif"), ("C", "2026-01", "Actif"), ("C", "2026-02", "Actif"),
               ("D", "2026-01", "Actif"), ("D", "2026-02", "Actif")]
    seg = {"A": "Essentiel", "B": "Pro", "C": "Essentiel", "D": "Essentiel"}
    cas = [
        (exemple, {"segments": seg, "etats": ("Actif", "Parti"), "force": 4}),
        (exemple, {"segments": seg, "etats": ("Actif", "Parti"), "force": 0}),
        (exemple, {"segments": {**seg, "E": "Confort"}, "etats": ("Actif", "Parti"), "force": 2}),   # Confort : vide
        (exemple + [("E", "2026-01", "Parti")], {"segments": {**seg, "E": "Confort"}, "etats": ("Actif", "Parti"),
                                                 "force": 0}),
        (exemple, {"segments": seg, "etats": ("Actif", "Parti"), "force": 2.5}),
    ]
    for graine in range(5):
        releves, segments = generer_releves(60 + 20 * graine, graine=100 + graine, debut="2025-06", fin="2026-03",
                                            manquants=0.05, doublons=0.03)
        for client in list(segments)[:: 7]:                    # quelques clients de plus en Pro
            segments[client] = "Pro"
        cas.append((releves, {"segments": segments, "etats": ETATS, "force": (0, 1, 10, 50, 7.5)[graine]}))
    return cas


def _verifier_forme_segments(r, segments, etats, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie {type(r).__name__} : attendu un dict {{segment: matrice}}.")
    attendus = set(segments.values())
    if set(r) != attendus:
        manque = sorted(attendus - set(r))
        raise AssertionError(f"{quoi} a pour segments {sorted(r)} au lieu de {sorted(attendus)}"
                             + (f" : il manque {manque} (un segment sans relevé a quand même sa matrice : la globale)."
                                if manque else "."))
    for segment, matrice in r.items():
        _verifier_forme_matrice(matrice, etats, f"{quoi}[{segment!r}]")


def verifier_matrices_par_segment(f) -> None:
    cas = _cas_259()
    _verifier_releves(f, cas, lambda r, kw, quoi: _verifier_forme_segments(r, kw["segments"], kw["etats"], quoi),
                      lambda releves, kw, bug: _ref_segments(releves, kw["segments"], kw["etats"], kw["force"], bug),
                      _BUGS_SEGMENTS, "matrices_par_segment", _segments_proches, _ecart_segments)

    # ------------------------------------------------------------------ toute la clientèle, et les 30 clients Pro
    releves, segments = releves_exemple(), segments_exemple()
    r = _appel(f, (UneSeuleFois(releves), segments), "matrices_par_segment(un flux des 48 434 relevés, "
               "segments_exemple(), force=50)", kwargs={"force": 50}, lignes_max=_lignes_releves(len(releves)) + 50_000,
               conseil_lent="Rangez les relevés une seule fois dans un dict par client, puis comptez : chercher le mois "
                            "suivant (ou le segment) dans toute la liste, à chaque relevé, est bien trop lent.")
    _verifier_forme_segments(r, segments, ETATS, "matrices_par_segment(les relevés de l'exemple)")
    attendu = _ref_segments(releves, segments, ETATS, 50)
    if not _segments_proches(r, attendu):
        conseil = _premier_bug(r, lambda bug: _ref_segments(releves, segments, ETATS, 50, bug), _BUGS_SEGMENTS,
                               _segments_proches)
        raise AssertionError(f"Sur les relevés de l'exemple (force=50), matrices_par_segment donne "
                             f"{_ecart_segments(r, attendu)}. {conseil}".rstrip())

    # ------------------------------------------------------------------ ce qu'il faut refuser
    base = [("A", "2026-01", "Actif"), ("A", "2026-02", "Fidèle"), ("B", "2026-01", "Actif")]
    seg = {"A": "Pro", "B": "Essentiel"}
    for args, kwargs, attendu, conseil in (
            ((base + [("C", "2026-01", "Actif")], seg), {}, ValueError,
             "Le client C n'a pas de segment : on ne sait pas où compter ses passages → ValueError."),
            ((base, seg), {"force": -1}, ValueError, "Une force négative retirerait des passages : refusé."),
            ((base, seg), {"force": "50"}, TypeError, "`force` est un nombre, pas un texte."),
            ((base, seg), {"force": True}, TypeError, "True n'est pas une force."),
            ((base, [("A", "Pro")]), {}, TypeError, "`segments` est un dict {client: segment}."),
            ((base + [("B", "2026-01", "Parti")], seg), {}, ValueError, "B est « Actif » ET « Parti » en janvier 2026."),
            ((base + [("B", "2026-13", "Actif")], seg), {}, ValueError, "« 2026-13 » n'est pas un mois."),
            ((base + [("B", "2026-02", "Suspendu")], seg), {}, ValueError, "« Suspendu » n'est pas dans `etats`.")):
        _doit_lever(f, args, f"matrices_par_segment({_court(args[0], 90)}, {_court(args[1], 50)}"
                             f"{', force=' + repr(kwargs['force']) if kwargs else ''})", conseil, attendu, kwargs=kwargs)


# ---------------------------------------------------------------------------- la série

SERIE = Serie(
    code="42",
    titre="Ce que vaut un client : transitions, projection, durée de vie, valeur actualisée, rentabilité, lissage",
    module_eleve="defis.serie_42",
    module_corrige="defis.corrige_serie_42",
    defis=(
        Defi(254, "matrice_transition", "🟢", "—", verifier_matrice_transition),
        Defi(255, "projection", "🟡", "—", verifier_projection),
        Defi(256, "duree_de_vie", "🟡", "—", verifier_duree_de_vie),
        Defi(257, "valeur_client", "🟡", "—", verifier_valeur_client),
        Defi(258, "delai_rentabilite", "🔴", "—", verifier_delai_rentabilite),
        Defi(259, "matrices_par_segment", "🔴", "—", verifier_matrices_par_segment),
    ),
)

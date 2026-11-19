"""Vérifications automatiques de la série 47 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est le réseau, qui livre les messages en retard, deux fois ou jamais, et l'horloge du serveur, qui
recule quand on la remet à l'heure ; et c'est l'équipe de la facturation, qui compte vos allers-retours.

- il compare vos réponses à celles d'une version de référence, sur des cas choisis (une réplique absente d'un vecteur,
  un « é » écrit en deux caractères, deux fiches égales écrites dans un autre ordre, une feuille de l'arbre où une
  seule fiche diffère, un message arrivé avant celui qui le précède, une heure qui recule, une estampille venue du
  futur, une vieille copie qui ferait revenir un client effacé…) et sur des milliers de cas tirés au hasard ;
- quand votre résultat est faux, il rejoue les erreurs classiques (les sommes des vecteurs, les répliques communes
  seulement, `json.dumps` sans `sort_keys`, Unicode non normalisé, toutes les feuilles demandées, les messages
  appliqués à leur arrivée, l'heure du serveur prise pour une estampille, la fiche la plus récente gardée en entier,
  l'effacement sans pierre tombale…) pour nommer la vôtre ;
- il vérifie que tout ce qui est mal formé est refusé par une erreur, que rien de ce que vous recevez n'est modifié,
  compte les allers-retours réseau du défi 286 et les lignes exécutées dans votre fichier quand une solution naïve
  serait trop lente.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
import unicodedata
from datetime import datetime, timezone
from decimal import Decimal

from .synchro import APRES, AVANT, CONCURRENTES, DERIVE_MAX_MS, EGALES, Replique, parc_exemple
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
    if isinstance(exc, KeyError):
        return ("Une clé absente (KeyError) : une réplique absente d'un vecteur vaut 0, un client absent d'un état "
                "vaut {\"champs\": {}, \"efface\": None} : lisez avec `.get(...)`.")
    if isinstance(exc, TypeError) and "unhashable" in str(exc):
        return ("TypeError « unhashable » : une liste ou un dict ne peut pas entrer dans un `set` ni servir de clé de "
                "dict.")
    if isinstance(exc, TypeError) and "not supported between" in str(exc):
        return ("Vous comparez deux choses qui ne se comparent pas (`None` et un tuple ?) : un effacement absent vaut "
                "None, écartez-le avant de prendre le plus grand.")
    if isinstance(exc, AttributeError):
        return "Un attribut manque : l'avez-vous bien créé dans `__init__` (avec `self.`) ?"
    return ""


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = None, conseil_lent: str = "",
           kwargs: dict | None = None, fichier: str | None = None, au_lieu: dict | None = None):
    """f(*args), en transformant toute erreur en message lisible (NotImplementedError : le défi est à faire)."""
    kwargs = kwargs or {}
    try:
        return _trace((fichier or _fichier(f)) if lignes_max else None, lambda: f(*args, **kwargs), lignes_max)
    except NotImplementedError:
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError. {conseil}".rstrip()) from None
    except Exception as exc:
        propre = next((texte for genre, texte in (au_lieu or {}).items() if isinstance(exc, genre)), "")
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {propre or _conseil_exception(exc) or conseil}".rstrip()) \
            from None


def _doit_lever(f, args: tuple, quoi: str, conseil: str = "", attendu=ValueError, kwargs: dict | None = None):
    """Exige que f(*args) lève `attendu` (un type ou un tuple de types). Renvoie l'exception, pour lire son message."""
    kwargs = kwargs or {}
    noms = " ou ".join(t.__name__ for t in (attendu if isinstance(attendu, tuple) else (attendu,)))
    try:
        resultat = f(*args, **kwargs)
    except NotImplementedError:
        raise
    except attendu as exc:
        return exc
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError au lieu de {noms}. {conseil}".rstrip()) from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {noms}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {noms}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


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


def _sans_modification(avant, apres, quoi: str, conseil: str = "") -> None:
    if avant != apres:
        raise AssertionError(f"Votre code a MODIFIÉ {quoi} : on lit ce qu'on reçoit, on ne le change pas. "
                             f"{conseil}".rstrip())


def _entier(valeur, quoi: str) -> int:
    if isinstance(valeur, bool) or not isinstance(valeur, int):
        raise TypeError(f"{quoi} doit être un entier")
    if valeur < 0:
        raise ValueError(f"{quoi} est négatif")
    return valeur


def _texte(valeur, quoi: str) -> str:
    if not isinstance(valeur, str):
        raise TypeError(f"{quoi} doit être un texte")
    if not valeur:
        raise ValueError(f"{quoi} est vide")
    return valeur


# ============================================================================ défi 284 : comparer_versions


def _ref_versions(a, b, bug: str = "") -> str:
    """La comparaison de référence ; `bug` rejoue une erreur classique (après les mêmes vérifications)."""
    for vecteur in (a, b):
        if not isinstance(vecteur, dict):
            raise TypeError("un vecteur est un dict")
        for replique, compteur in vecteur.items():
            _texte(replique, "une réplique")
            _entier(compteur, "un compteur")
    if bug == "somme":
        sa, sb = sum(a.values()), sum(b.values())
        return EGALES if sa == sb else AVANT if sa < sb else APRES
    repliques = a.keys() & b.keys() if bug == "communes" else a.keys() | b.keys()
    if bug == "strict":
        a_dans_b = all(a.get(r, 0) < b.get(r, 0) for r in repliques)
        b_dans_a = all(b.get(r, 0) < a.get(r, 0) for r in repliques)
        if a == b:
            return EGALES
    else:
        a_dans_b = all(a.get(r, 0) <= b.get(r, 0) for r in repliques)
        b_dans_a = all(b.get(r, 0) <= a.get(r, 0) for r in repliques)
    if bug == "egalite":
        if a == b:
            return EGALES
        return AVANT if a_dans_b else APRES if b_dans_a else CONCURRENTES
    if a_dans_b and b_dans_a:
        return EGALES
    if bug == "un_sens":
        return AVANT if a_dans_b else APRES
    if bug == "inverse":
        return APRES if a_dans_b else AVANT if b_dans_a else CONCURRENTES
    return AVANT if a_dans_b else APRES if b_dans_a else CONCURRENTES


_BUGS_VERSIONS = {
    "somme": "Vous comparez les SOMMES des compteurs : {\"crm\": 3, \"appli\": 1} (4) et {\"crm\": 2, \"appli\": 4} (6) "
             "deviennent « avant », alors que chacune a vu quelque chose que l'autre n'a pas vu (concurrentes). "
             "Comparez réplique par réplique.",
    "communes": "Vous ne regardez que les répliques présentes dans les DEUX vecteurs : {\"crm\": 1} et {\"appli\": 1} "
                "deviennent « egales ». Parcourez `a.keys() | b.keys()`, et lisez une réplique absente comme 0.",
    "egalite": "Vous testez l'égalité avec `a == b` : {\"crm\": 2, \"appli\": 0} et {\"crm\": 2} ne sont plus "
               "« egales ». Une réplique absente vaut 0 : deux vecteurs sont égaux quand chacun est inclus dans "
               "l'autre.",
    "un_sens": "Vous ne trouvez jamais de concurrentes : « inclus dans b ? sinon : après ». Il faut DEUX tests "
               "(a inclus dans b, b inclus dans a) : si aucun ne tient, elles sont concurrentes.",
    "inverse": "Vous inversez « avant » et « apres » : `a` est AVANT `b` quand `b` a tout vu de `a`, et plus (chaque "
               "compteur de `a` est inférieur ou égal à celui de `b`).",
    "strict": "Vous comparez avec `<` au lieu de `<=` : {\"crm\": 2, \"appli\": 1} et {\"crm\": 3, \"appli\": 1} "
              "deviennent concurrentes, alors que la seconde a tout vu de la première. Un compteur égal ne "
              "contredit pas l'inclusion.",
}


def _cas_284() -> list[tuple[dict, dict]]:
    cas = [({"crm": 2, "appli": 1}, {"crm": 3, "appli": 1}), ({"crm": 3, "appli": 1}, {"crm": 2, "appli": 1}),
           ({"crm": 2, "appli": 0}, {"crm": 2}), ({"crm": 2}, {"crm": 2, "appli": 0}),
           ({"crm": 3, "appli": 1}, {"crm": 2, "appli": 4}), ({"crm": 1}, {"appli": 1}), ({}, {}),
           ({}, {"crm": 0}), ({}, {"crm": 1}), ({"crm": 1}, {}), ({"crm": 1, "appli": 1}, {"crm": 1, "appli": 1}),
           ({"crm": 5, "appli": 0}, {"crm": 5, "factu": 0}), ({"crm": 1, "appli": 2}, {"crm": 2, "appli": 1}),
           ({"crm": 0, "appli": 3}, {"appli": 3, "factu": 1}), ({"a": 10}, {"a": 9, "b": 1})]
    alea = random.Random(2841)
    repliques = ["crm", "appli", "factu"]
    for _ in range(600):
        a = {r: alea.randint(0, 3) for r in repliques if alea.random() < 0.7}
        b = {r: alea.randint(0, 3) for r in repliques if alea.random() < 0.7}
        if alea.random() < 0.3:                       # b voit tout de a, et parfois plus
            b = {r: a.get(r, 0) + alea.choice([0, 0, 1]) for r in set(a) | set(b)}
        cas.append((a, b))
    return cas


def verifier_comparer_versions(f) -> None:
    cas = _cas_284()
    obtenus = []
    for a, b in cas:
        copies = copy.deepcopy((a, b))
        quoi = f"comparer_versions({a!r}, {b!r})"
        r = _appel(f, (a, b), quoi)
        _sans_modification(copies, (a, b), "un vecteur reçu",
                           "`a.setdefault(r, 0)` ajoute la réplique au vecteur : lisez avec `a.get(r, 0)`.")
        if r not in (AVANT, APRES, EGALES, CONCURRENTES):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} : rendez l'une des constantes AVANT, APRES, EGALES ou "
                                 "CONCURRENTES (les textes \"avant\", \"apres\", \"egales\", \"concurrentes\").")
        obtenus.append(r)
    for (a, b), r in zip(cas, obtenus):
        attendu = _ref_versions(a, b)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_versions(x, y, bug) for x, y in cas], _BUGS_VERSIONS,
                                   lambda x, y: x == y)
            raise AssertionError(f"comparer_versions({a!r}, {b!r}) renvoie {r!r} au lieu de {attendu!r}. "
                                 f"{conseil or 'Pour chaque réplique de a OU de b (absente : 0), comparez les deux compteurs.'}")

    for args, attendu, conseil in (
            (([("crm", 1)], {}), TypeError, "Un vecteur est un dict, pas une liste de couples."),
            (({}, None), TypeError, "Vérifiez b aussi."),
            (({1: 2}, {}), TypeError, "Une réplique est un texte."),
            (({"": 1}, {}), ValueError, "Une réplique vide n'en est pas une."),
            (({"crm": -1}, {}), ValueError, "Un compteur négatif n'a pas de sens."),
            (({"crm": 1}, {"crm": -1}), ValueError, "Vérifiez b aussi, en entier."),
            (({"crm": 1.0}, {}), TypeError, "Un compteur est un entier (1.0 est un float)."),
            (({"crm": "2"}, {}), TypeError, "Un compteur est un entier, pas un texte."),
            (({"crm": True}, {}), TypeError, "`True` est un `int` en Python, mais pas un compteur : écartez les "
                                             "`bool` (`isinstance(n, bool)`)."),
            (({"crm": 5}, {"appli": None}), TypeError,
             "Vérifiez TOUT avant de comparer : b a un compteur qui vaut None.")):
        _doit_lever(f, args, f"comparer_versions{_court(args, 70)}", conseil, attendu)


# ============================================================================ défi 285 : empreinte


def _ref_canonique(valeur, chemin: str, bug: str = "", profondeur: int = 0):
    if valeur is None or isinstance(valeur, (bool, int)):
        return valeur
    if isinstance(valeur, float):
        if not math.isfinite(valeur):
            raise ValueError(f"{chemin} n'est pas fini")
        return valeur
    if isinstance(valeur, str):
        if bug == "sans_nfc" or (bug == "nfc_surface" and profondeur > 1):
            return valeur
        return unicodedata.normalize("NFD" if bug == "nfd" else "NFC", valeur)
    if isinstance(valeur, (list, tuple)):
        return [_ref_canonique(v, f"{chemin}[{i}]", bug, profondeur + 1) for i, v in enumerate(valeur)]
    if isinstance(valeur, dict):
        copie = {}
        for cle, v in valeur.items():
            if not isinstance(cle, str):
                raise TypeError(f"{chemin} a une clé qui n'est pas un texte")
            normale = cle if bug in ("sans_nfc", "cles_brutes") or (bug == "nfc_surface" and profondeur > 0) else \
                unicodedata.normalize("NFD" if bug == "nfd" else "NFC", cle)
            if normale in copie:
                raise ValueError(f"{chemin} a deux clés identiques une fois normalisées")
            copie[normale] = _ref_canonique(v, f"{chemin}[{normale!r}]", bug, profondeur + 1)
        return copie
    raise TypeError(f"{chemin} est un {type(valeur).__name__}")


def _ref_empreinte(fiche, bug: str = "") -> str:
    if not isinstance(fiche, dict):
        raise TypeError("une fiche est un dict")
    canonique = _ref_canonique(fiche, "la fiche", bug)
    options = {"sort_keys": True, "ensure_ascii": False, "separators": (",", ":"), "allow_nan": False}
    if bug == "str":
        return hashlib.sha256(str(fiche).encode("utf-8")).hexdigest()
    if bug == "nfc_apres":
        texte = json.dumps(_ref_canonique(fiche, "la fiche", "sans_nfc"), **options)
        return hashlib.sha256(unicodedata.normalize("NFC", texte).encode("utf-8")).hexdigest()
    if bug == "sans_tri":
        options["sort_keys"] = False
    if bug == "tri_surface":
        canonique, options["sort_keys"] = dict(sorted(canonique.items())), False
    if bug == "separateurs":
        del options["separators"]
    if bug == "ascii":
        options["ensure_ascii"] = True
    return hashlib.sha256(json.dumps(canonique, **options).encode("utf-8")).hexdigest()


_BUGS_EMPREINTE = {
    "sans_tri": "Vous écrivez les clés dans l'ordre où elles ont été ajoutées : {\"offre\": …, \"sms\": …} et "
                "{\"sms\": …, \"offre\": …}, la même fiche, ont deux empreintes. `json.dumps(..., sort_keys=True)`.",
    "tri_surface": "Vous triez les clés du PREMIER niveau seulement (`dict(sorted(fiche.items()))`) : celles des dict "
                   "imbriqués restent dans leur ordre d'arrivée. `sort_keys=True` trie à toute profondeur.",
    "sans_nfc": "Vous ne normalisez pas l'Unicode : « Orléans » écrit par le CRM (« é » en un caractère) et envoyé par "
                "l'iPhone (« e » + accent combinant) n'ont pas la même empreinte. `unicodedata.normalize(\"NFC\", "
                "texte)` pour CHAQUE texte, clés comprises.",
    "nfd": "Vous normalisez en NFD (les accents séparés) : c'est stable, mais ce n'est pas la forme du protocole. "
           "Normalisez en NFC (la forme composée, celle du web).",
    "cles_brutes": "Vous normalisez les valeurs, pas les CLÉS : un champ nommé « numéro » (en NFD) ne donne pas la même "
                   "empreinte. Normalisez aussi chaque clé de chaque dict.",
    "nfc_surface": "Vous ne normalisez que le premier niveau de la fiche : les textes des listes et des dict imbriqués "
                   "gardent leur écriture d'origine. Faites-le à toute profondeur (une fonction récursive).",
    "nfc_apres": "Vous normalisez le texte JSON APRÈS l'avoir écrit : les clés ont été triées avant d'être normalisées "
                 "(« e » + accent combinant passe avant « f », « é » passe après). Normalisez chaque texte AVANT "
                 "`json.dumps`.",
    "str": "Vous hachez `str(fiche)` : c'est l'écriture de Python (des apostrophes, `True`, l'ordre d'insertion), pas un "
           "format d'échange que l'autre système sait reproduire. Écrivez la fiche avec `json.dumps(...)` et les "
           "options données.",
    "separateurs": "Vous gardez les séparateurs par défaut de `json.dumps` (\", \" et \": \") : votre empreinte est "
                   "stable, mais ce n'est pas celle que calcule l'autre système. `separators=(\",\", \":\")`.",
    "ascii": "Vous écrivez les accents en `\\u00e9` (`ensure_ascii=True`, la valeur par défaut) : l'autre système écrit "
             "« é » en UTF-8. `ensure_ascii=False`, puis `.encode(\"utf-8\")`.",
}

_NFD = unicodedata.normalize("NFD", "é")


def _cas_285() -> list:
    orleans = unicodedata.normalize("NFD", "Orléans")
    cas = [{}, {"a": 1}, {"offre": "Fibre 1G", "sms": True}, {"sms": True, "offre": "Fibre 1G"},
           {"ville": "Orléans"}, {"ville": orleans}, {"n": 1}, {"n": 1.0}, {"n": True}, {"n": "1"}, {"n": None},
           {"n": 0.1}, {"n": -0.0}, {"n": 1e100}, {"n": 12345678901234567890}, {"liste": [3, 1, 2]},
           {"liste": (3, 1, 2)}, {"z": {"b": 1, "a": 2}, "a": [{"y": 1, "x": "é"}]},
           {"adresse": {"ville": orleans, "rue": "rue de l'Été"}},
           {"rues": ["quai de l'" + unicodedata.normalize("NFD", "Écluse"), "rue des Lilas"]},
           {"num" + _NFD + "ro": "06"}, {"numéro": "06"}, {"f": 1, _NFD: 2}, {"f": {_NFD + "t" + _NFD: "x", "a": 1}}, {"émoji": "😀 Ça marche"}, {"texte": "a\"b\\c\n\td"},
           {"vide": "", "liste_vide": [], "dict_vide": {}}, {"B": 1, "a": 2, "_": 3, "É": 4, "e": 5},
           {"profond": {"x": {"y": {"z": ["Noël", {"ü": "ﬁ"}]}}}},
           {"consentement_sms": False, "consentement_email": True, "email": "c0042@exemple.fr"}]
    alea = random.Random(2851)
    textes = ["a", "é", _NFD, "Zoé", "Zo" + _NFD, "ﬁ", "😀", "x y", " ", "Ω"]
    for _ in range(120):
        fiche = {}
        for _ in range(alea.randint(0, 5)):
            cle = alea.choice(textes) + alea.choice(["", "1", "_x"])
            if unicodedata.normalize("NFC", cle) in {unicodedata.normalize("NFC", k) for k in fiche}:
                continue
            fiche[cle] = alea.choice([alea.randint(-5, 5), alea.random(), alea.choice(textes), None,
                                      alea.random() < 0.5, [alea.choice(textes), 2], {alea.choice(["b", "a"]): 1}])
        cas.append(fiche)
    return cas


def verifier_empreinte(f) -> None:
    cas = _cas_285()
    obtenus = []
    for fiche in cas:
        copie = copy.deepcopy(fiche)
        quoi = f"empreinte({_court(fiche, 80)})"
        r = _appel(f, (fiche,), quoi)
        _sans_modification(copie, fiche, "la fiche", "Construisez une COPIE canonique, sans toucher à la fiche reçue.")
        if isinstance(r, int) and not isinstance(r, bool):
            raise AssertionError(f"{quoi} renvoie un entier ({r}) : `hash()` change à chaque lancement de Python "
                                 "(PYTHONHASHSEED) et n'existe pas dans les autres langages. Rendez "
                                 "`hashlib.sha256(octets).hexdigest()`.")
        if not isinstance(r, str):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) : rendez un texte, "
                                 "`hashlib.sha256(octets).hexdigest()`.")
        if len(r) == 32:
            raise AssertionError(f"{quoi} renvoie 32 caractères : c'est MD5, cassé depuis 2004. Utilisez "
                                 "`hashlib.sha256`.")
        if len(r) != 64 or any(c not in "0123456789abcdef" for c in r):
            raise AssertionError(f"{quoi} renvoie {r!r} : une empreinte SHA-256 s'écrit en 64 caractères "
                                 "hexadécimaux en minuscules (`.hexdigest()`, pas `.digest()`).")
        obtenus.append(r)
    for fiche, r in zip(cas, obtenus):
        attendu = _ref_empreinte(fiche)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_empreinte(x, bug) for x in cas], _BUGS_EMPREINTE,
                                   lambda x, y: x == y)
            texte = json.dumps(_ref_canonique(fiche, "la fiche"), sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"))
            raise AssertionError(f"empreinte({_court(fiche, 100)}) renvoie {r[:12]}… au lieu de {attendu[:12]}…. "
                                 f"{conseil or 'Le texte à hacher, en UTF-8, est : ' + _court(texte, 120)}")

    for args, attendu, conseil in (
            (([("a", 1)],), TypeError, "Une fiche est un dict."),
            ((None,), TypeError, "None n'est pas une fiche."),
            (({"rappels": {"lundi", "jeudi"}},), TypeError, "Un set n'a pas d'ordre : pas d'écriture canonique."),
            (({"vu_le": datetime(2026, 10, 1, tzinfo=timezone.utc)},), TypeError,
             "Une date a dix écritures possibles : refusez-la (la fiche doit l'écrire elle-même, en texte)."),
            (({"photo": b"\x89PNG"},), TypeError, "Des octets ne sont pas du texte."),
            (({"montant": Decimal("19.90")},), TypeError, "Un Decimal n'est pas un type JSON."),
            (({1: "a"},), TypeError, "`json.dumps` écrirait la clé 1 comme \"1\" : {1: x} et {\"1\": x} auraient la "
                                     "même empreinte. Les clés sont des textes."),
            (({"adresse": {("rue", 1): "x"}},), TypeError, "Les clés des dict IMBRIQUÉS aussi sont des textes."),
            (({"x": [1, {2: "b"}]},), TypeError, "Les clés des dict rangés dans une liste aussi."),
            (({"x": float("nan")},), ValueError, "NaN n'a pas d'écriture JSON (et n'est pas égal à lui-même)."),
            (({"x": [float("inf")]},), ValueError, "L'infini n'a pas d'écriture JSON, même dans une liste."),
            (({"é": 1, _NFD: 2},), ValueError, "Ces deux clés s'écrivent pareil une fois normalisées en NFC : la "
                                                "fiche est ambiguë.")):
        _doit_lever(f, args, f"empreinte{_court(args, 70)}", conseil, attendu)


# ============================================================================ défi 286 : differences


def _ref_differences(locale, distante, bug: str = "") -> dict:
    if locale.profondeur != distante.profondeur:
        raise ValueError("profondeurs différentes")
    ecarts: dict[str, str] = {}
    if bug not in ("partout", "toutes_les_feuilles") and locale.empreinte("") == distante.empreinte(""):
        return ecarts
    if bug == "toutes_les_feuilles":
        a_voir = ["".join(p) for p in _prefixes(locale.profondeur)]
    else:
        a_voir = [""]
    while a_voir:
        prefixe = a_voir.pop()
        if len(prefixe) == locale.profondeur:
            ici, la_bas = locale.feuille(prefixe), distante.feuille(prefixe)
            cles = la_bas.keys() if bug == "un_sens" else ici.keys() | la_bas.keys()
            for identifiant in cles:
                if identifiant not in la_bas:
                    if bug != "sans_absents":
                        ecarts[identifiant] = "distante" if bug == "inverse" else "locale"
                elif identifiant not in ici:
                    if bug != "sans_absents":
                        ecarts[identifiant] = "locale" if bug == "inverse" else "distante"
                elif ici[identifiant] != la_bas[identifiant]:
                    ecarts[identifiant] = "modifiee"
        else:
            ici, la_bas = locale.enfants(prefixe), distante.enfants(prefixe)
            a_voir.extend(e for e in ici if bug == "partout" or ici[e] != la_bas[e])
    return ecarts


def _prefixes(profondeur: int):
    if profondeur == 0:
        return [""]
    return [p + c for p in _prefixes(profondeur - 1) for c in "0123456789abcdef"]


_BUGS_DIFFERENCES = {
    "un_sens": "Vous ne parcourez que les fiches de la DISTANTE : celles qui n'existent que dans la locale disparaissent. "
               "Dans une feuille, parcourez `ici.keys() | la_bas.keys()`.",
    "sans_absents": "Vous ne trouvez que les fiches MODIFIÉES : une fiche présente d'un seul côté (un client pas encore "
                    "facturé, un compte clos) diffère aussi. Marquez-la \"locale\" ou \"distante\".",
    "inverse": "Vous inversez \"locale\" et \"distante\" : \"locale\" veut dire « la fiche n'existe que dans `locale` ».",
}


def _cas_286() -> list[tuple[str, dict, dict, int]]:
    """(nom, empreintes locales, empreintes distantes, profondeur)."""
    alea = random.Random(2861)

    def parc(n: int, prefixe: str = "K") -> dict:
        return {f"{prefixe}{i:05d}": format(alea.getrandbits(64), "016x") for i in range(n)}

    def abimer(base: dict, modifiees: int, seulement_ici: int, seulement_la_bas: int) -> tuple[dict, dict]:
        ici, la_bas = dict(base), dict(base)
        ids = alea.sample(sorted(base), modifiees + seulement_ici + seulement_la_bas)
        for i in ids[:modifiees]:
            la_bas[i] = format(alea.getrandbits(64), "016x")
        for i in ids[modifiees:modifiees + seulement_ici]:
            del la_bas[i]
        for i in ids[modifiees + seulement_ici:]:
            del ici[i]
        return ici, la_bas

    petit = {"C00001": "a1", "C00002": "b2", "C00003": "c3"}
    cas = [("deux répliques identiques", petit, dict(petit), 2),
           ("l'exemple de la docstring", petit, {"C00001": "a1", "C00002": "XX", "C00004": "d4"}, 2),
           ("deux répliques vides", {}, {}, 3),
           ("une locale vide", {}, parc(40), 2),
           ("une distante vide", parc(40), {}, 2),
           ("une profondeur 0 (la racine est une feuille)", *abimer(parc(30), 2, 1, 1), 0),
           ("une profondeur 1", *abimer(parc(200), 3, 2, 2), 1)]
    base = parc(3000)
    cas.append(("3 000 fiches identiques", base, dict(base), 3))
    for n, (m, i, la) in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1), (5, 4, 3), (20, 0, 0), (0, 15, 15)]):
        cas.append((f"3 000 fiches, {m} modifiées, {i} seulement locales, {la} seulement distantes",
                    *abimer(base, m, i, la), 3 if n % 2 == 0 else 2))
    meme_feuille = {k: v for k, v in base.items()}
    voisins = sorted(base, key=lambda k: hashlib.sha256(k.encode()).hexdigest())[:6]   # dans la même feuille
    autre = dict(meme_feuille)
    for k in voisins[:3]:
        autre[k] = "modifiee"
    cas.append(("des différences toutes dans la même feuille", meme_feuille, autre, 2))
    crm, facturation = parc_exemple()
    cas.append(("les 48 000 fiches du CRM et de la facturation", crm, facturation, 3))
    return cas


def _repliques(ici: dict, la_bas: dict, profondeur: int) -> tuple[Replique, Replique]:
    return Replique(ici, profondeur, nom="locale"), Replique(la_bas, profondeur, nom="distante")


def verifier_differences(f) -> None:
    for nom, ici, la_bas, profondeur in _cas_286():
        reference = _repliques(ici, la_bas, profondeur)
        attendu = _ref_differences(*reference)
        appels, transferees = reference[1].appels, reference[1].fiches_transferees
        locale, distante = _repliques(ici, la_bas, profondeur)
        quoi = f"differences(locale, distante) ({nom})"
        r = _appel(f, (locale, distante), quoi, lignes_max=3_000_000,
                   au_lieu={ValueError: "Une `Replique` refuse `enfants` sur une feuille et `feuille` sur un nœud : "
                                        "une feuille est un préfixe de `profondeur` chiffres (et si `profondeur` vaut 0, "
                                        "la racine \"\" est une feuille)."})
        if not isinstance(r, dict):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} : rendez un dict {{identifiant: \"locale\", "
                                 "\"distante\" ou \"modifiee\"}}.")
        if r != attendu:
            faux = {k: v for k, v in r.items() if attendu.get(k) != v}
            manquants = {k: v for k, v in attendu.items() if k not in r}
            conseil = _premier_bug(r, lambda bug: _ref_differences(*_repliques(ici, la_bas, profondeur), bug),
                                   _BUGS_DIFFERENCES, lambda x, y: x == y)
            detail = (f"{len(attendu)} fiches diffèrent ; vous en trouvez {len(r)}"
                      + (f", dont celles-ci à tort ou mal nommées : {_court(faux, 100)}" if faux else "")
                      + (f" ; il manque : {_court(manquants, 100)}" if manquants else ""))
            raise AssertionError(f"{quoi} : {detail}. {conseil}".rstrip())
        if distante.appels > appels:
            raise AssertionError(
                f"{quoi} trouve les bonnes fiches, mais pose {distante.appels} questions à la distante, quand {appels} "
                "suffisent. " + ("Deux racines égales : tout est égal, une question suffit." if not attendu else
                                 "Ne descendez QUE dans les enfants dont les empreintes diffèrent, et ne demandez "
                                 "`feuille` que pour une feuille qui diffère : chaque question est un aller-retour "
                                 "réseau (vous visitez des sous-arbres identiques)."))
        if distante.fiches_transferees > transferees:
            raise AssertionError(f"{quoi} trouve les bonnes fiches, mais se fait envoyer {distante.fiches_transferees} "
                                 f"fiches, quand {transferees} suffisent : ne demandez `feuille` que pour une feuille "
                                 "dont les empreintes diffèrent.")

    a, b = _repliques({"x": "1"}, {"x": "1"}, 2), Replique({"x": "1"}, 3, nom="distante")
    _doit_lever(f, (a[0], b), "differences(une réplique de profondeur 2, une de profondeur 3)",
                "Deux arbres de profondeurs différentes ne se comparent pas : vérifiez `profondeur` d'abord.")


# ============================================================================ défi 287 : Recepteur


class _RefRecepteur:
    """Le récepteur de référence ; `bug` rejoue une erreur classique."""

    def __init__(self, bug: str = "") -> None:
        self.bug = bug
        self.prochain: dict[str, int] = {}
        self.attente: dict[str, dict[int, dict]] = {}

    def recevoir(self, message) -> list[dict]:
        if not isinstance(message, dict):
            raise TypeError("un message est un dict")
        if "source" not in message or "numero" not in message:
            raise ValueError("message incomplet")
        source = _texte(message["source"], "la source")
        numero = _entier(message["numero"], "le numéro")
        if numero < 1:
            raise ValueError("les numéros commencent à 1")
        if self.bug == "arrivee":
            return [message]
        cle = "*" if self.bug == "global" else source
        prochain = self.prochain.get(cle, 0 if self.bug == "zero" else 1)
        if numero < prochain:
            return [message] if self.bug == "doublons" else []
        attente = self.attente.get(cle, {})
        if numero in attente:
            if attente[numero] != message:
                if self.bug == "conflit_ignore":
                    return []
                if self.bug == "conflit_remplace":
                    attente[numero] = message
                    return []
                raise ValueError("deux messages différents pour un même numéro")
            return []
        if numero > prochain:
            if self.bug == "saute":
                self.prochain[cle] = numero + 1
                return [message]
            self.attente.setdefault(cle, {})[numero] = message
            return []
        appliques = [message]
        prochain += 1
        while prochain in attente:
            appliques.append(attente.pop(prochain))
            prochain += 1
            if self.bug == "un_seul":
                break
        if self.bug == "sans_vidage":
            appliques, prochain = [message], numero + 1
        self.prochain[cle] = prochain
        if cle in self.attente and not attente:
            del self.attente[cle]
        return appliques

    def en_attente(self) -> int:
        return sum(len(a) for a in self.attente.values())

    def manquants(self) -> dict[str, list[int]]:
        resultat = {}
        for source, attente in self.attente.items():
            debut = self.prochain.get(source, 0 if self.bug == "zero" else 1)
            if self.bug == "manquants_tous":
                resultat[source] = list(range(debut, max(attente)))
            else:
                resultat[source] = [n for n in range(debut, max(attente)) if n not in attente]
        if self.bug == "manquants_vides":
            for source in self.prochain:
                resultat.setdefault(source, [])
        return resultat


_BUGS_RECEPTEUR = {
    "doublons": "Un message déjà appliqué revient (le téléphone l'a renvoyé, faute d'accusé de réception) et vous "
                "l'appliquez à nouveau : il écrase ce qui a suivi. Un numéro plus petit que celui qu'on attend est un "
                "doublon : rendez [].",
    "arrivee": "Vous appliquez chaque message à son ARRIVÉE : l'ordre du réseau n'est pas celui du téléphone (un message "
               "en retard écrase le suivant), et les doublons repassent. Gardez, par source, le numéro attendu, et "
               "mettez de côté ceux qui arrivent trop tôt.",
    "global": "Vous avez UN compteur pour toutes les sources : chaque téléphone numérote SES messages à partir de 1. "
              "Un dict {source: numéro attendu}.",
    "zero": "Vous attendez le numéro 0 : les numéros commencent à 1 (`self.prochain.get(source, 1)`).",
    "un_seul": "Quand le message attendu arrive, vous ne libérez qu'UN des messages qui attendaient derrière lui : "
               "continuez tant que les numéros se suivent (`while prochain in attente:`).",
    "sans_vidage": "Quand le message attendu arrive, vous oubliez ceux qui attendaient derrière lui : ils ne seront "
                   "jamais appliqués. Rendez [ce message, puis les suivants en attente, tant que les numéros se "
                   "suivent].",
    "saute": "Vous SAUTEZ le trou : le n° 5 est appliqué alors que le n° 4 n'est pas arrivé, et quand le 4 arrive, il "
             "passe pour un doublon (il est perdu). Un message qui arrive trop tôt attend son tour.",
    "conflit_ignore": "Deux messages DIFFÉRENTS portent le même numéro, et vous gardez le premier sans rien dire : c'est "
                      "le signe d'un téléphone (ou d'un serveur) cassé. Levez ValueError.",
    "conflit_remplace": "Deux messages DIFFÉRENTS portent le même numéro, et vous remplacez le premier sans rien dire : "
                        "levez ValueError, sans rien changer.",
    "manquants_tous": "Dans `manquants()`, vous listez TOUS les numéros entre l'attendu et le plus grand en attente, "
                      "y compris ceux qui sont déjà arrivés : seuls ceux qui ne sont pas là manquent.",
    "manquants_vides": "Dans `manquants()`, une source sans message en attente n'a rien qui manque : elle n'y figure "
                       "pas (pas de liste vide).",
}


def _m(source: str, numero, **reste) -> dict:
    return {"source": source, "numero": numero, "champ": "consentement_sms", "valeur": numero % 2 == 0, **reste}


def _scenarios_287() -> list[list[tuple]]:
    """Des suites d'opérations : ("recevoir", message), ("en_attente",) ou ("manquants",)."""
    r, n, q = "recevoir", "en_attente", "manquants"
    a, b = "tel-001", "tel-002"
    scenarios = [
        [(r, _m(a, 1)), (r, _m(a, 2)), (r, _m(a, 3)), (n,), (q,)],
        [(r, _m(a, 2)), (q,), (n,), (r, _m(a, 1)), (n,), (q,), (r, _m(a, 3))],
        [(r, _m(a, 1)), (r, _m(a, 1)), (r, _m(a, 2)), (r, _m(a, 1)), (r, _m(a, 2))],
        [(r, _m(a, 1)), (r, _m(a, 1, valeur="autre")), (r, _m(a, 2))],      # un doublon tardif, même différent : ignoré
        [(r, _m(a, 4)), (r, _m(a, 3)), (r, _m(a, 2)), (q,), (n,), (r, _m(a, 1)), (n,), (q,)],
        [(r, _m(a, 3)), (r, _m(a, 5)), (q,), (r, _m(a, 1)), (q,), (r, _m(a, 2)), (q,), (r, _m(a, 4)), (q,), (n,)],
        [(r, _m(a, 1)), (r, _m(b, 1)), (r, _m(b, 2)), (r, _m(a, 2)), (r, _m(b, 4)), (q,), (r, _m(a, 4)), (q,),
         (r, _m(b, 3)), (q,), (n,)],
        [(r, _m(a, 2)), (r, _m(a, 2)), (n,), (r, _m(a, 2, valeur="autre")), (n,), (q,), (r, _m(a, 1)), (n,)],
        [(r, _m(a, 3)), (r, _m(a, 2)), (r, _m(a, 1)), (r, _m(a, 2)), (r, _m(a, 3)), (n,), (q,)],
        [(r, _m(b, 2)), (r, _m(a, 1)), (q,), (r, _m(a, 3)), (q,)],
    ]
    alea = random.Random(2871)
    for graine in range(40):
        operations = []
        attendus = {f"tel-{i}": 0 for i in range(3)}
        messages = []
        for source in attendus:
            for numero in range(1, alea.randint(2, 12)):
                messages.append(_m(source, numero, envoye_le=graine))
        livraison = []
        for message in messages:
            livraison.append((alea.random() * 10, message))
            if alea.random() < 0.2:
                livraison.append((alea.random() * 10, message))
        livraison.sort(key=lambda x: x[0])
        for _t, message in livraison:
            operations.append((r, message))
            if alea.random() < 0.3:
                operations.append((alea.choice([n, q]),))
        operations += [(n,), (q,)]
        scenarios.append(operations)
    return scenarios


def _jouer_287(creer, operations) -> list:
    recepteur = creer()
    sorties = []
    for op in operations:
        try:
            if op[0] == "recevoir":
                sortie = recepteur.recevoir(op[1])
            elif op[0] == "en_attente":
                sortie = recepteur.en_attente()
            else:
                sortie = recepteur.manquants()
        except (ValueError, TypeError) as exc:
            sortie = ("erreur", type(exc).__name__)
        sorties.append(sortie)
    return sorties


def _decrire_ops_287(operations, jusqu_a: int) -> str:
    morceaux = []
    for op in operations[: jusqu_a + 1]:
        if op[0] == "recevoir":
            extra = "" if op[1].get("valeur") == (op[1]["numero"] % 2 == 0) else " (contenu différent)"
            morceaux.append(f"recevoir({op[1]['source']} n° {op[1]['numero']}{extra})")
        else:
            morceaux.append(f"{op[0]}()")
    texte = " ; ".join(morceaux)
    return texte if len(texte) <= 400 else "… " + texte[-400:]


def _decrire_sortie_287(sortie) -> str:
    if isinstance(sortie, tuple) and sortie[:1] == ("erreur",):
        return f"lève {sortie[1]}"
    if isinstance(sortie, list) and all(isinstance(m, dict) and "numero" in m for m in sortie):
        return "rend " + repr([f"{m['source']} n° {m['numero']}" for m in sortie])
    return f"rend {_court(sortie, 120)}"


def _creer(classe, quoi: str):
    return _appel(classe, (), quoi)


def verifier_recepteur(classe) -> None:
    # Deux récepteurs : chacun son état (vérifié d'abord : un état partagé fausserait tout le reste).
    un, autre = _creer(classe, "Recepteur()"), _creer(classe, "Recepteur()")
    m1, m2 = _m("tel-001", 1), _m("tel-001", 2)
    _appel(un.recevoir, (m2,), "recevoir(tel-001 n° 2)")
    sortie = _appel(autre.recevoir, (m1,), "recevoir(tel-001 n° 1)")
    if isinstance(sortie, list) and any(m == m2 for m in sortie):
        raise AssertionError("Deux récepteurs se mélangent : le n° 2, reçu par un premier récepteur, ressort d'un "
                             "second, qui n'a reçu que le n° 1. Rangez l'état dans `self` (dans `__init__`), pas dans "
                             "un attribut de CLASSE (`attente = {}` écrit sous `class Recepteur:` est partagé par tous "
                             "les récepteurs).")
    messages = [_m("tel-002", 2), _m("tel-002", 1)]
    copies = copy.deepcopy(messages)
    recepteur = _creer(classe, "Recepteur()")
    for message in messages:
        _appel(recepteur.recevoir, (message,), "recevoir(un message)")
    _sans_modification(copies, messages, "un message reçu")

    for operations in _scenarios_287():
        attendu = _jouer_287(_RefRecepteur, operations)
        obtenu = _appel(lambda: _jouer_287(classe, operations), (), "votre Recepteur",
                        au_lieu={KeyError: "Une clé absente (KeyError) : une source NOUVELLE attend le numéro 1 "
                                           "(`self.prochain.get(source, 1)`)."})
        for i, (o, a) in enumerate(zip(obtenu, attendu)):
            if o != a:
                conseil = _premier_bug(obtenu, lambda bug: _jouer_287(lambda: _RefRecepteur(bug), operations),
                                       _BUGS_RECEPTEUR, lambda x, y: x == y)
                op = operations[i]
                raise AssertionError(f"Après {_decrire_ops_287(operations, i)}, votre {op[0]}() "
                                     f"{_decrire_sortie_287(o)} au lieu de {_decrire_sortie_287(a).removeprefix('rend ')}"
                                     f". {conseil}".rstrip())

    # 30 000 messages d'une même source, du dernier au premier.
    recepteur = _creer(classe, "Recepteur()")
    messages = [{"source": "tel-lent", "numero": n} for n in range(30_000, 0, -1)]

    def tout_recevoir():
        sorties = [recepteur.recevoir(m) for m in messages]
        return sorties[-1], recepteur.en_attente(), recepteur.manquants()

    dernier, attente, manquants = _appel(
        tout_recevoir, (), "Recevoir 30 000 messages arrivés du dernier au premier", lignes_max=3_000_000,
        fichier=_fichier(classe.recevoir),
        conseil_lent="Pour chaque message, vous parcourez ceux qui attendent : 30 000 × 30 000 opérations. Rangez-les "
                     "dans un dict {numéro: message} par source : `prochain in attente` et `attente.pop(prochain)` "
                     "sont immédiats.")
    if [m["numero"] for m in dernier] != list(range(1, 30_001)) or attente != 0 or manquants != {}:
        raise AssertionError("Après 30 000 messages arrivés du dernier au premier, l'arrivée du n° 1 doit rendre les "
                             f"30 000 messages dans l'ordre (vous en rendez {len(dernier)}), et plus rien ne doit "
                             "attendre.")

    recepteur = _creer(classe, "Recepteur()")
    for args, attendu, conseil in (
            (([("source", "tel-001")],), TypeError, "Un message est un dict."),
            (({"numero": 1},), ValueError, "Un message sans \"source\" : ValueError."),
            (({"source": "tel-001"},), ValueError, "Un message sans \"numero\" : ValueError."),
            (({"source": "", "numero": 1},), ValueError, "Une source vide n'en est pas une."),
            (({"source": 7, "numero": 1},), TypeError, "Une source est un texte."),
            (({"source": "tel-001", "numero": "1"},), TypeError, "Un numéro est un entier, pas un texte."),
            (({"source": "tel-001", "numero": 1.0},), TypeError, "Un numéro est un entier (1.0 est un float)."),
            (({"source": "tel-001", "numero": True},), TypeError, "`True` n'est pas un numéro."),
            (({"source": "tel-001", "numero": 0},), ValueError, "Les numéros commencent à 1."),
            (({"source": "tel-001", "numero": -3},), ValueError, "Les numéros commencent à 1.")):
        _doit_lever(recepteur.recevoir, args, f"recevoir({_court(args[0], 60)})", conseil, attendu)


# ============================================================================ défi 288 : HorlogeHybride


def _estampille(valeur) -> tuple[int, int, str]:
    if not isinstance(valeur, tuple):
        raise TypeError("une estampille est un tuple")
    if len(valeur) != 3:
        raise ValueError("une estampille a trois éléments")
    return _entier(valeur[0], "l"), _entier(valeur[1], "c"), _texte(valeur[2], "la réplique")


class _RefHorloge:
    """L'horloge hybride de référence ; `bug` rejoue une erreur classique."""

    def __init__(self, replique, physique, derive_max=DERIVE_MAX_MS, bug: str = "") -> None:
        self.replique = _texte(replique, "la réplique")
        if not callable(physique):
            raise TypeError("physique doit être appelable")
        self.derive_max = _entier(derive_max, "derive_max")
        self.physique, self.bug = physique, bug
        self.l = self.c = 0

    def _pt(self) -> int:
        return _entier(self.physique(), "l'heure physique")

    def estampiller(self) -> tuple:
        pt = self._pt()
        if self.bug == "physique":
            return (pt, 0, self.replique)
        if self.bug == "lamport":
            self.l += 1
            return (self.l, 0, self.replique)
        if self.bug == "sans_compteur":
            self.l = max(self.l, pt)
            return (self.l, 0, self.replique)
        if self.bug == "suit_le_physique":
            self.c = self.c + 1 if pt == self.l else 0
            self.l = pt
            return (self.l, self.c, self.replique)
        return self._local(pt)

    def recevoir(self, estampille) -> tuple:
        lm, cm, _ = _estampille(estampille)
        pt = self._pt()
        trop = lm - pt >= self.derive_max if self.bug == "derive_inclusive" else lm - pt > self.derive_max
        if trop and self.bug not in ("sans_derive", "derive_apres"):
            raise ValueError("trop d'avance")
        if self.bug == "physique":
            resultat = (pt, 0, self.replique)
        elif self.bug == "ignore_recu":
            resultat = self._local(pt)
        elif self.bug == "suit_le_physique":
            self.c = self.c + 1 if pt == self.l else 0
            self.l = pt
            resultat = (self.l, self.c, self.replique)
        elif self.bug == "lamport":
            self.l = max(self.l, lm) + 1
            resultat = (self.l, 0, self.replique)
        elif self.bug == "sans_compteur":
            self.l = max(self.l, lm, pt)
            resultat = (self.l, 0, self.replique)
        else:
            nouveau = max(self.l, lm, pt)
            if nouveau == self.l == lm:
                c = self.c + 1 if self.bug == "max_oublie" else max(self.c, cm) + 1
            elif nouveau == self.l:
                c = self.c + 1
            elif nouveau == lm:
                c = 0 if self.bug == "compteur_zero" else cm + 1
            else:
                c = 0
            self.l, self.c = nouveau, c
            resultat = (self.l, self.c, self.replique)
        if trop and self.bug == "derive_apres":
            raise ValueError("trop d'avance")
        return resultat

    def _local(self, pt: int) -> tuple:
        if pt > self.l:
            self.l, self.c = pt, 0
        else:
            self.c += 1
        return (self.l, self.c, self.replique)


_BUGS_HORLOGE = {
    "physique": "Vous rendez l'heure du serveur, (pt, 0, réplique) : elle RECULE quand on la remet à l'heure (la "
                "correction paraît plus ancienne que la faute), et deux modifications de la même milliseconde ont la "
                "même estampille. Gardez `l` (le plus grand temps vu) et un compteur `c`.",
    "lamport": "Vous avez écrit une horloge de Lamport (`l + 1` à chaque fois) : l'ordre est bon, mais l'estampille ne "
               "dit plus l'heure (elle doit suivre l'heure du serveur quand celle-ci avance : si pt > l, l = pt).",
    "sans_compteur": "Votre compteur `c` reste à 0 : quand l'heure n'avance pas (même milliseconde, ou horloge qui "
                     "recule), deux estampilles sont ÉGALES. Si pt <= l, augmentez `c`.",
    "suit_le_physique": "Votre `l` suit l'heure du serveur même quand elle RECULE : `l` est le plus grand temps VU, il ne "
                        "redescend jamais (si pt <= l, on garde l et on augmente c).",
    "max_oublie": "Quand votre `l` et `lm` sont égaux et l'emportent, il faut c = max(c, cm) + 1 : avec c + 1 seulement, "
                  "l'estampille rendue peut être plus petite que celle reçue.",
    "ignore_recu": "Votre `recevoir` ignore l'estampille reçue : ce qu'on écrit après avoir reçu l'état de l'appli "
                   "paraît plus ancien que lui. Prenez `nouveau = max(l, lm, pt)`.",
    "compteur_zero": "Quand l'estampille reçue l'emporte (nouveau == lm), vous remettez `c` à 0 : si `cm` > 0, vous "
                     "rendez une estampille PLUS PETITE que celle reçue. Dans ce cas, c = cm + 1.",
    "sans_derive": "Vous acceptez une estampille qui a plus de `derive_max` ms d'avance : un téléphone réglé en 2031 "
                   "ferait perdre au CRM toutes ses modifications suivantes. Levez ValueError.",
    "derive_apres": "Vous levez bien ValueError pour une estampille venue du futur, mais APRÈS avoir changé `l` ou `c` : "
                    "le mal est fait. Vérifiez la dérive avant de toucher à l'état.",
    "derive_inclusive": "Une avance d'exactement `derive_max` ms est permise : on refuse seulement quand "
                        "lm - pt > derive_max (strictement).",
}


def _scenarios_288() -> list[tuple[str, list]]:
    """(nom, opérations) ; une opération : ("e", pt) ou ("r", pt, estampille), sur l'horloge "crm" (derive_max 60 000)."""
    d = DERIVE_MAX_MS
    scenarios = [
        ("l'exemple de la docstring", [("e", 1000), ("e", 1000), ("e", 880), ("e", 880), ("r", 880, (1200, 5, "appli")),
                                       ("e", 1500), ("r", 1500, (1500 + d + 1, 0, "telephone")), ("e", 1500)]),
        ("l'heure avance", [("e", 1000), ("e", 1001), ("e", 1500), ("e", 9000)]),
        ("la même milliseconde", [("e", 1000)] * 5),
        ("l'heure recule de deux minutes", [("e", 200_000), ("e", 200_010), ("e", 80_020), ("e", 80_030),
                                            ("e", 200_009), ("e", 200_011), ("e", 200_012)]),
        ("l'estampille reçue l'emporte", [("e", 1000), ("r", 1000, (1500, 7, "appli")), ("e", 1000), ("e", 1600)]),
        ("même temps des deux côtés", [("e", 1000), ("e", 1000), ("r", 1000, (1000, 5, "appli")),
                                       ("r", 900, (1000, 0, "appli")), ("e", 900)]),
        ("l'heure du serveur l'emporte", [("e", 1000), ("r", 5000, (1200, 9, "appli")), ("e", 5000)]),
        ("mon temps l'emporte", [("e", 9000), ("r", 9000, (1200, 9, "appli")), ("r", 100, (8000, 3, "appli"))]),
        ("une estampille venue du futur", [("e", 1000), ("r", 1000, (1000 + d + 1, 0, "telephone")),
                                           ("e", 1000), ("e", 1001)]),
        ("une avance d'exactement derive_max", [("e", 1000), ("r", 1000, (1000 + d, 2, "appli")), ("e", 1000)]),
        ("une avance juste au-dessus", [("e", 5000), ("e", 5000), ("r", 5000, (5000 + d + 1, 0, "x")), ("e", 5000)]),
        ("le départ à zéro", [("e", 0), ("e", 0), ("r", 0, (0, 0, "appli"))]),
        ("recevoir d'abord", [("r", 1000, (1000, 3, "appli")), ("e", 1000), ("e", 999)]),
    ]
    alea = random.Random(2881)
    for graine in range(30):
        operations, pt = [], alea.randint(1_000, 50_000)
        lointain = pt
        for _ in range(alea.randint(20, 120)):
            tirage = alea.random()
            if tirage < 0.05:
                pt = max(0, pt - alea.randint(1, 150_000))           # la mise à l'heure recule
            elif tirage < 0.5:
                pt += alea.choice([0, 0, 1, alea.randint(1, 3000)])
            lointain = max(lointain, pt)
            if alea.random() < 0.6:
                operations.append(("e", pt))
            else:
                lm = max(0, pt + alea.choice([-5000, -1, 0, 0, 1, 1000, d - 1, d, d + 1, 2 * d,
                                              alea.randint(-200, 200)]))
                if alea.random() < 0.3:
                    lm = lointain
                operations.append(("r", pt, (lm, alea.choice([0, 0, 1, 2, 7]), alea.choice(["appli", "factu"]))))
        scenarios.append((f"une suite tirée au hasard (graine {graine})", operations))
    return scenarios


def _jouer_288(creer, operations) -> list:
    heure = [0]
    horloge = creer("crm", lambda: heure[0])
    sorties = []
    for op in operations:
        heure[0] = op[1]
        try:
            sortie = horloge.estampiller() if op[0] == "e" else horloge.recevoir(op[2])
        except ValueError:
            sortie = "ValueError"
        sorties.append(sortie)
    return sorties


def _decrire_op_288(op) -> str:
    return f"estampiller() à {op[1]} ms" if op[0] == "e" else f"recevoir({op[2]}) à {op[1]} ms"


def verifier_horloge_hybride(classe) -> None:
    # Deux horloges : chacune son état (vérifié d'abord : un état partagé fausserait tout le reste).
    heure = [1000]
    une = _creer_horloge(classe, "crm", heure)
    premiere = _appel(une.estampiller, (), "estampiller()")
    _appel(une.estampiller, (), "estampiller()")
    autre = _creer_horloge(classe, "appli", heure)
    seconde = _appel(autre.estampiller, (), "estampiller()")
    if isinstance(premiere, tuple) and isinstance(seconde, tuple) and premiere[:2] != seconde[:2]:
        raise AssertionError(f"Deux horloges se mélangent : une NOUVELLE horloge, à la même heure, rend {seconde!r} alors "
                             f"que la première avait commencé par {premiere!r}. Rangez `l` et `c` dans `self` (dans "
                             "`__init__`), pas dans des attributs de classe.")

    for nom, operations in _scenarios_288():
        attendu = _jouer_288(_RefHorloge, operations)
        obtenu = _appel(lambda: _jouer_288(classe, operations), (), f"votre HorlogeHybride ({nom})")
        for i, (o, a) in enumerate(zip(obtenu, attendu)):
            if o != a:
                conseil = _premier_bug(obtenu, lambda bug: _jouer_288(
                    lambda r, p: _RefHorloge(r, p, bug=bug), operations), _BUGS_HORLOGE, lambda x, y: x == y)
                avant = ("… ; " if i > 3 else "") + " ; ".join(
                    _decrire_op_288(op) + f" → {s}" for op, s in zip(operations[max(0, i - 3):i], attendu[max(0, i - 3):i]))
                if not conseil:
                    if o == "ValueError":
                        conseil = "Vous levez ValueError à tort : on refuse seulement quand lm - pt > derive_max."
                    elif a == "ValueError":
                        conseil = "Il fallait lever ValueError : l'estampille reçue a plus de derive_max ms d'avance."
                    elif isinstance(o, tuple) and len(o) == 3 and i and isinstance(attendu[i - 1], tuple) and \
                            o <= attendu[i - 1]:
                        conseil = "Vos estampilles doivent être strictement croissantes."
                    else:
                        conseil = ("Relisez les quatre cas de `recevoir`, comparés à l'ANCIEN `l`."
                                   if operations[i][0] == "r" else "Si pt > l : l = pt et c = 0 ; sinon c = c + 1.")
                raise AssertionError(f"{nom} : {('après ' + avant + ', ') if avant else ''}{_decrire_op_288(operations[i])}"
                                     f" donne {o!r} au lieu de {a!r}. {conseil}".rstrip())

    heure = [1000]
    horloge = _creer_horloge(classe, "crm", heure)
    for args, attendu, conseil in (
            (([1000, 0, "appli"],), TypeError, "Une estampille est un tuple (une liste venue d'un JSON se convertit "
                                               "avec `tuple(...)` avant)."),
            (((1000, 0),), ValueError, "Une estampille a trois éléments (l, c, réplique)."),
            ((("1000", 0, "appli"),), TypeError, "l est un entier."),
            (((1000.0, 0, "appli"),), TypeError, "l est un entier (1000.0 est un float)."),
            (((1000, True, "appli"),), TypeError, "c est un entier, pas un booléen."),
            (((-1, 0, "appli"),), ValueError, "l ne peut pas être négatif."),
            (((1000, 0, ""),), ValueError, "La réplique est un texte non vide."),
            (((1000, 0, 7),), TypeError, "La réplique est un texte.")):
        _doit_lever(horloge.recevoir, args, f"recevoir({_court(args[0], 40)})", conseil, attendu)
    for args, attendu, conseil in (
            (("", lambda: 0), ValueError, "Une réplique vide n'en est pas une."),
            ((7, lambda: 0), TypeError, "Une réplique est un texte."),
            (("crm", 1000), TypeError, "`physique` est une FONCTION qui rend l'heure (lambda: ...), pas une heure."),
            (("crm", lambda: 0, -1), ValueError, "derive_max ne peut pas être négatif."),
            (("crm", lambda: 0, 1.5), TypeError, "derive_max est un entier.")):
        _doit_lever(classe, args, f"HorlogeHybride{_court(args, 60)}", conseil, attendu)
    flottante = _appel(classe, ("crm", lambda: 1000.5), "HorlogeHybride(\"crm\", lambda: 1000.5)")
    _doit_lever(flottante.estampiller, (), "estampiller() quand physique() rend 1000.5", "L'heure physique est un "
                "nombre ENTIER de millisecondes (`time.time() * 1000` est un float : `time.time_ns() // 1_000_000`).",
                TypeError)


def _creer_horloge(classe, replique, heure):
    return _appel(classe, (replique, lambda: heure[0]), f"HorlogeHybride({replique!r}, physique)")


# ============================================================================ défi 289 : fusionner


_SANS = {"champs": {}, "efface": None}


def _verifier_etat(etat) -> None:
    if not isinstance(etat, dict):
        raise TypeError("un état est un dict")
    for client, fiche in etat.items():
        _texte(client, "un client")
        if not isinstance(fiche, dict):
            raise TypeError("une fiche est un dict")
        if set(fiche) != {"champs", "efface"}:
            raise ValueError("une fiche a deux clés")
        if not isinstance(fiche["champs"], dict):
            raise TypeError("les champs sont un dict")
        if fiche["efface"] is not None:
            _estampille(fiche["efface"])
        for champ, valeur in fiche["champs"].items():
            _texte(champ, "un champ")
            if not isinstance(valeur, tuple):
                raise TypeError("(valeur, estampille)")
            if len(valeur) != 2:
                raise ValueError("(valeur, estampille)")
            _estampille(valeur[1])


def _ref_fusionner(a, b, bug: str = "") -> dict:
    _verifier_etat(a)
    _verifier_etat(b)
    fusion = {}
    for client in sorted(a.keys() | b.keys()):
        fa, fb = a.get(client, _SANS), b.get(client, _SANS)
        if bug == "b_gagne":
            fiche = fb if client in b else fa
            fusion[client] = {"champs": dict(fiche["champs"]), "efface": fiche["efface"]}
            continue
        if bug == "fiche_entiere":
            def plus_recente(fiche):
                dates = [e for _v, e in fiche["champs"].values()] + ([fiche["efface"]] if fiche["efface"] else [])
                return max(dates) if dates else (-1, -1, "")
            fiche = fa if plus_recente(fa) >= plus_recente(fb) else fb
            fusion[client] = {"champs": dict(fiche["champs"]), "efface": fiche["efface"]}
            continue
        effacements = [e for e in (fa["efface"], fb["efface"]) if e is not None]
        efface = (min if bug == "efface_min" else max)(effacements) if effacements else None
        champs = {}
        for champ in sorted(fa["champs"].keys() | fb["champs"].keys()):
            versions = []
            for fiche in (fa, fb):
                v = fiche["champs"].get(champ)
                if v is not None and bug == "filtre_local" and fiche["efface"] is not None and v[1] < fiche["efface"]:
                    v = None
                if v is not None:
                    versions.append(v)
            if not versions:
                continue
            if len(versions) == 2 and versions[0][1] == versions[1][1] and versions[0][0] != versions[1][0] \
                    and bug != "egalite_silencieuse":
                raise ValueError("même estampille, deux valeurs")
            if bug == "temps_seul":
                valeur, estampille = max(versions, key=lambda v: v[1][0])
            else:
                valeur, estampille = max(versions, key=lambda v: v[1])
            if bug in ("sans_filtre", "filtre_local") or efface is None or estampille > efface:
                champs[champ] = (valeur, estampille)
        if bug == "sans_pierre_tombale" and efface is not None and not champs:
            continue
        fusion[client] = {"champs": champs, "efface": efface}
    return fusion


_BUGS_FUSIONNER = {
    "fiche_entiere": "Vous gardez la fiche la plus récente EN ENTIER : c'est le script d'aujourd'hui (le refus des SMS "
                     "saisi dans l'appli le matin disparaît quand un conseiller change l'offre l'après-midi). "
                     "Fusionnez CHAMP par champ.",
    "b_gagne": "Vous laissez `b` gagner (`{**a, **b}`) : la fusion dépend de l'ordre des arguments "
               "(fusionner(a, b) != fusionner(b, a)), et les deux copies ne convergent pas. Pour chaque champ, la plus "
               "grande estampille gagne, d'où qu'elle vienne.",
    "sans_pierre_tombale": "Vous retirez du résultat le client effacé : sans pierre tombale, la copie suivante qui a "
                           "encore sa fiche le fait revenir (les 41 clients effacés d'octobre). Gardez "
                           "{\"champs\": {}, \"efface\": estampille}.",
    "sans_filtre": "Vous gardez la pierre tombale, mais pas l'effacement : les champs écrits AVANT lui restent. Ne "
                   "gardez que ceux dont l'estampille est plus grande que \"efface\".",
    "filtre_local": "Chaque côté n'applique que SON effacement à SES champs : les champs plus anciens qui viennent de "
                    "l'AUTRE copie (celle qui n'a pas encore vu l'effacement) reviennent. Prenez le plus grand "
                    "effacement des deux côtés, puis filtrez TOUS les champs.",
    "efface_min": "Vous gardez le plus PETIT des deux effacements : prenez le plus grand (le plus récent).",
    "temps_seul": "Vous ne comparez que `l` : deux écritures de la même milliseconde (c ou réplique différents) "
                  "donnent un résultat qui dépend de l'ordre des arguments. Comparez les estampilles ENTIÈRES "
                  "(des tuples : `max(versions, key=lambda v: v[1])`).",
    "egalite_silencieuse": "Deux valeurs différentes portent la même estampille, et vous en gardez une sans rien dire : "
                           "une estampille date UNE écriture, c'est le signe d'une horloge cassée. Levez ValueError.",
}


def _cas_289() -> list[tuple[str, dict, dict]]:
    def fiche(champs=None, efface=None) -> dict:
        return {"champs": dict(champs or {}), "efface": efface}

    imp, crm, app = (50, 0, "import"), (100, 0, "crm"), (90, 0, "appli")
    cas = [
        ("deux champs modifiés de chaque côté",
         {"C1": fiche({"offre": ("Fibre", crm), "sms": (True, imp)})},
         {"C1": fiche({"offre": ("Essentiel", imp), "sms": (False, app)})}),
        ("le même champ des deux côtés", {"C1": fiche({"offre": ("Fibre", crm)})},
         {"C1": fiche({"offre": ("Mobile", app)})}),
        ("un client d'un seul côté", {"C1": fiche({"offre": ("Fibre", crm)})}, {"C2": fiche({"sms": (False, app)})}),
        ("deux états vides", {}, {}),
        ("un état vide", {}, {"C1": fiche({"sms": (False, app)}, (40, 0, "crm"))}),
        ("un effacement face à une vieille copie", {"C1": fiche({}, (120, 0, "crm"))},
         {"C1": fiche({"offre": ("Fibre", crm), "sms": (False, app)})}),
        ("une vieille copie face à un effacement", {"C1": fiche({"offre": ("Fibre", crm), "sms": (False, app)})},
         {"C1": fiche({}, (120, 0, "crm"))}),
        ("une réinscription après l'effacement", {"C1": fiche({}, (120, 0, "crm"))},
         {"C1": fiche({"email": ("x@exemple.fr", (130, 0, "appli")), "sms": (True, app)})}),
        ("deux effacements", {"C1": fiche({"a": (1, (110, 0, "crm"))}, (100, 0, "crm"))},
         {"C1": fiche({"b": (2, (105, 0, "appli"))}, (120, 0, "appli"))}),
        ("un effacement dans chaque copie, des champs entre les deux",
         {"C1": fiche({"a": (1, (115, 0, "crm"))}, (100, 0, "crm"))},
         {"C1": fiche({"b": (2, (130, 0, "appli"))}, (120, 0, "appli"))}),
        ("la même milliseconde, des compteurs différents", {"C1": fiche({"a": ("x", (100, 2, "crm"))})},
         {"C1": fiche({"a": ("y", (100, 1, "crm"))})}),
        ("la même milliseconde, le même compteur, deux répliques", {"C1": fiche({"a": ("x", (100, 0, "appli"))})},
         {"C1": fiche({"a": ("y", (100, 0, "crm"))})}),
        ("la même estampille, la même valeur", {"C1": fiche({"a": ("x", crm)})}, {"C1": fiche({"a": ("x", crm)})}),
        ("des valeurs de tous les types", {"C1": fiche({"a": (None, crm), "b": ([1, 2], app), "c": ({"k": 1}, imp)})},
         {"C1": fiche({"a": (0, imp), "d": (3.5, app)})}),
    ]
    alea = random.Random(2891)
    clients, champs, repliques = ["C1", "C2", "C3", "C4"], ["a", "b", "c"], ["crm", "appli", "factu"]

    def estampille() -> tuple:
        return (alea.randint(1, 8), alea.randint(0, 2), alea.choice(repliques))

    def etat() -> dict:
        resultat = {}
        for client in clients:
            if alea.random() < 0.25:
                continue
            efface = estampille() if alea.random() < 0.3 else None
            fiche_ = {}
            for champ in champs:
                if alea.random() < 0.6:
                    e = estampille()
                    if e != efface:
                        fiche_[champ] = (f"{champ}{e[0]}{e[1]}{e[2]}", e)   # une valeur par estampille : pas de conflit
            resultat[client] = {"champs": fiche_, "efface": efface}
        return resultat

    for i in range(500):
        cas.append((f"deux états tirés au hasard (n° {i})", etat(), etat()))
    return cas


def _resume_289(a, b) -> str:
    return f"a = {_court(a, 220)}, b = {_court(b, 220)}"


def _partage(resultat, etat) -> bool:
    for client, fiche in resultat.items():
        source = etat.get(client)
        if isinstance(source, dict) and isinstance(fiche, dict):
            if fiche is source or fiche.get("champs") is source.get("champs"):
                return True
    return False


def verifier_fusionner(f) -> None:
    cas = _cas_289()
    obtenus = []
    for nom, a, b in cas:
        copies = copy.deepcopy((a, b))
        quoi = f"fusionner(a, b) ({nom})"
        r = _appel(f, (a, b), quoi)
        _sans_modification(copies, (a, b), "un des états reçus",
                           "`dict(a)` ou `a.copy()` copie le dict du dessus, pas les fiches : construisez de NOUVEAUX "
                           "dict pour chaque fiche.")
        if not isinstance(r, dict):
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} : rendez un état, un dict {{client: fiche}}.")
        obtenus.append(r)
    for (nom, a, b), r in zip(cas, obtenus):
        quoi = f"fusionner(a, b) ({nom})"
        attendu = _ref_fusionner(a, b)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_fusionner(x, y, bug) for _n, x, y in cas],
                                   _BUGS_FUSIONNER, lambda x, y: x == y)
            if not conseil:
                inverse = _appel(f, (b, a), f"fusionner(b, a) ({nom})")
                if inverse != r:
                    conseil = ("fusionner(a, b) et fusionner(b, a) donnent deux résultats différents : votre fusion "
                               "dépend de l'ordre des copies, elles ne convergeront pas.")
                else:
                    conseil = ("Pour chaque champ : la plus grande estampille ; \"efface\" : le plus grand effacement ; "
                               "puis seulement les champs écrits après lui.")
            raise AssertionError(f"{quoi} : avec {_resume_289(a, b)}, vous rendez {_court(r, 260)} au lieu de "
                                 f"{_court(attendu, 260)}. {conseil}")
        if _partage(r, a) or _partage(r, b):
            raise AssertionError(f"{quoi} : votre résultat PARTAGE des dict avec a ou b (une fiche, ou ses \"champs\") : "
                                 "les deux copies ne sont plus indépendantes, modifier l'une modifierait l'autre. "
                                 "Construisez un nouveau {\"champs\": {...}, \"efface\": ...} pour chaque client.")

    # Les propriétés qui font converger les copies, sur des triplets.
    alea = random.Random(2892)
    etats = [x for _n, x, _y in cas[-300:]]
    for _ in range(60):
        x, y, z = alea.sample(etats, 3)
        gauche = _appel(f, (_appel(f, (x, y), "fusionner(x, y)"), z), "fusionner(fusionner(x, y), z)")
        droite = _appel(f, (x, _appel(f, (y, z), "fusionner(y, z)")), "fusionner(x, fusionner(y, z))")
        if gauche != droite:
            raise AssertionError(f"fusionner(fusionner(x, y), z) != fusionner(x, fusionner(y, z)) avec x = {_court(x, 150)}, "
                                 f"y = {_court(y, 150)}, z = {_court(z, 150)} : l'ordre des échanges change le "
                                 "résultat.")

    fiche = {"champs": {}, "efface": None}
    estampille = (100, 0, "crm")
    for args, attendu, conseil in (
            (([], {}), TypeError, "Un état est un dict."),
            (({}, None), TypeError, "Vérifiez b aussi."),
            (({"C1": [("a", 1)]}, {}), TypeError, "Une fiche est un dict."),
            (({"C1": {"champs": {}}}, {}), ValueError, "Une fiche a exactement deux clés : \"champs\" et \"efface\"."),
            (({"C1": {**fiche, "maj": 1}}, {}), ValueError, "Une fiche a exactement deux clés."),
            (({"C1": {"champs": [], "efface": None}}, {}), TypeError, "\"champs\" est un dict."),
            (({"C1": {"champs": {"a": ["x", estampille]}, "efface": None}}, {}), TypeError,
             "Un champ vaut un tuple (valeur, estampille)."),
            (({"C1": {"champs": {"a": ("x", estampille, 1)}, "efface": None}}, {}), ValueError,
             "Un champ vaut un tuple de DEUX éléments."),
            (({"C1": {"champs": {"a": ("x", [100, 0, "crm"])}, "efface": None}}, {}), TypeError,
             "Une estampille est un tuple."),
            (({"C1": {"champs": {"a": ("x", (100, 0))}, "efface": None}}, {}), ValueError,
             "Une estampille a trois éléments."),
            (({"C1": {"champs": {"a": ("x", (-1, 0, "crm"))}, "efface": None}}, {}), ValueError,
             "l ne peut pas être négatif."),
            (({"C1": {"champs": {"a": ("x", (1.5, 0, "crm"))}, "efface": None}}, {}), TypeError, "l est un entier."),
            (({"C1": {"champs": {}, "efface": (100, 0, "")}}, {}), ValueError,
             "L'estampille de l'effacement aussi se vérifie."),
            (({"C1": {"champs": {}, "efface": "hier"}}, {}), TypeError, "\"efface\" est None ou une estampille."),
            (({1: fiche}, {}), TypeError, "Un client est un texte."),
            (({"C1": fiche}, {"C2": {"champs": {"a": ("x", (1, 0, 5))}, "efface": None}}), TypeError,
             "Vérifiez b EN ENTIER aussi, même les clients que a n'a pas."),
            (({"C1": {"champs": {"a": ("x", estampille)}, "efface": None}},
              {"C1": {"champs": {"a": ("y", estampille)}, "efface": None}}), ValueError,
             "Deux valeurs différentes portent la même estampille : une horloge est cassée, levez ValueError.")):
        _doit_lever(f, args, f"fusionner{_court(args, 90)}", conseil, attendu)


SERIE = Serie(
    code="47",
    titre="Garder deux systèmes d'accord : versions, empreintes, arbre de Merkle, messages en ordre, horloge hybride, "
          "fusion",
    module_eleve="defis.serie_47",
    module_corrige="defis.corrige_serie_47",
    defis=(
        Defi(284, "comparer_versions", "🟢", "—", verifier_comparer_versions),
        Defi(285, "empreinte", "🟢", "—", verifier_empreinte),
        Defi(286, "differences", "🟡", "—", verifier_differences),
        Defi(287, "Recepteur", "🟡", "—", verifier_recepteur),
        Defi(288, "HorlogeHybride", "🔴", "—", verifier_horloge_hybride),
        Defi(289, "fusionner", "🔴", "—", verifier_fusionner),
    ),
)

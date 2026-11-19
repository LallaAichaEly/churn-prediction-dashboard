"""Vérifications automatiques de la série 46 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est l'auditeur de sécurité, qui essaie d'entrer par chaque porte que votre code laisse entrouverte,
et le comité des accès, qui ne retirera des droits que si vos listes sont exactes.

- il compare vos réponses à celles d'une version de référence, sur des cas choisis (« client » contre
  « clients:lire », « gestes:invalider » contre « gestes:valider », un refus placé APRÈS une autorisation, une liste de
  rôles vide, un cycle d'héritage, une entrée du journal tombée pile au début de la période, un délégant rétrogradé…)
  et sur l'annuaire, les règles et les 172 000 accès du journal de l'exemple ;
- quand votre résultat est faux, il rejoue les erreurs classiques (`startswith`, `fnmatch`, `zip`, une sous-chaîne, un
  seul étage d'héritage, l'héritage à l'envers, la première règle qui gagne, l'autorisation par défaut, `any` au lieu
  de `all`, [] pris pour « tout le monde », les rôles directs, la période oubliée, le conflit cherché rôle par rôle,
  la délégation recopiée ou jamais vérifiée…) pour nommer la vôtre ;
- il vérifie que tout ce qui est mal formé est REFUSÉ (une erreur, jamais un accès), que rien n'est modifié, et compte
  les lignes exécutées dans votre fichier quand une solution naïve serait trop lente ;
- pendant les défis 280 à 283, il prête sa propre `permission_couvre` (et ses `roles_effectifs` et `decider`) à votre
  fichier : chaque défi est noté seul.
"""

from __future__ import annotations

import copy
import fnmatch
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from .acces import (
    AUTORISER,
    CONDITIONS,
    INCOMPATIBLES,
    JOKER,
    LISTE,
    REFUSER,
    REVUE_LE,
    SEPARATEUR,
    Decision,
    heritage_exemple,
    journal_exemple,
    regles_exemple,
    utilisateurs_exemple,
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

_CLE_ABSENTE = ("Une clé absente (KeyError) : \"roles\" et \"si\" sont FACULTATIFS dans une règle (sans \"roles\", "
                "elle vise tout le monde ; sans \"si\", elle n'a pas de condition) : lisez-les avec "
                "`regle.get(\"si\", [])` ou `\"roles\" in regle`.")
_PAS_HACHABLE = ("TypeError « unhashable » : une liste ne peut pas entrer dans un `set` ni servir de clé de dict. "
                 "Transformez-la (`tuple(...)`), ou rangez ses éléments un par un.")


def _conseil_exception(exc: BaseException) -> str:
    if isinstance(exc, KeyError):
        return _CLE_ABSENTE
    if isinstance(exc, TypeError) and "unhashable" in str(exc):
        return _PAS_HACHABLE
    if isinstance(exc, TypeError) and ("offset-naive" in str(exc) or "offset-aware" in str(exc)):
        return ("Vous comparez une date sans fuseau à une date avec fuseau : vérifiez `maintenant.tzinfo` au début, et "
                "levez ValueError si elle n'en a pas.")
    if isinstance(exc, IndexError):
        return "Un indice sort de la liste : vérifiez vos bornes (un motif et une permission n'ont pas forcément la " \
               "même longueur)."
    return ""


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = None, conseil_lent: str = "",
           kwargs: dict | None = None, fichier: str | None = None, au_lieu: dict | None = None):
    """f(*args), en transformant toute erreur en message lisible (NotImplementedError : le défi est à faire).

    `au_lieu` : {type d'exception: conseil} propres à un défi, qui remplacent les conseils généraux.
    """
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


class _Pretes:
    """Remplace des fonctions dans le module de `f` le temps d'une vérification, puis remet les vôtres."""

    def __init__(self, f, remplacements: dict) -> None:
        self.module = sys.modules.get(getattr(f, "__module__", None) or "")
        self.remplacements, self.avant = remplacements, {}

    def __enter__(self):
        if self.module is not None:
            for nom, objet in self.remplacements.items():
                if hasattr(self.module, nom):
                    self.avant[nom] = getattr(self.module, nom)
                    setattr(self.module, nom, objet)
        return self

    def __exit__(self, *exc) -> None:
        for nom, objet in self.avant.items():
            setattr(self.module, nom, objet)


def _instant(valeur, nom: str) -> datetime:
    if not isinstance(valeur, datetime):
        raise TypeError(f"{nom} doit être un datetime, pas {type(valeur).__name__}")
    if valeur.tzinfo is None or valeur.utcoffset() is None:
        raise ValueError(f"{nom} n'a pas de fuseau horaire")
    return valeur


def _sans_modification(avant, apres, quoi: str) -> None:
    if avant != apres:
        raise AssertionError(f"Votre fonction a MODIFIÉ {quoi} : on lit une configuration (ou un journal), on ne la "
                             "change pas. Travaillez sur vos propres listes et ensembles (un `set()` neuf, "
                             "`list(...)`, `{**dico, ...}`), sans `.append`, `.extend`, `+=` ni affectation sur ce que "
                             "vous recevez.")


_COLLECTIONS = (list, tuple, set, frozenset)

# ============================================================================ défi 278 : permission_couvre


def _segments(texte, nom: str) -> list[str]:
    if not isinstance(texte, str):
        raise TypeError(f"{nom} doit être un texte (str), pas {type(texte).__name__}")
    segments = texte.split(SEPARATEUR)
    if "" in segments:
        raise ValueError(f"{nom} {texte!r} : segment vide")
    return segments


def _ref_couvre(motif, permission, bug: str = "") -> bool:
    """La couverture de référence ; `bug` rejoue une erreur classique (après les mêmes vérifications)."""
    morceaux, demandes = _segments(motif, "le motif"), _segments(permission, "la permission")
    if any(JOKER in s or LISTE in s for s in demandes):
        raise ValueError(f"la permission {permission!r} doit être précise")
    for segment in morceaux:
        if segment != JOKER and any(c == "" or JOKER in c for c in segment.split(LISTE)):
            raise ValueError(f"le motif {motif!r} est mal écrit")
    if bug == "startswith":
        return permission.startswith(motif.removesuffix(JOKER))
    if bug == "fnmatch":
        return fnmatch.fnmatchcase(permission, motif)
    if bug == "zip":
        return all(m == JOKER or p in m.split(LISTE) for m, p in zip(morceaux, demandes))
    if bug == "longueur" and len(morceaux) != len(demandes):
        return False
    for i in range(max(len(morceaux), len(demandes))):
        segment = morceaux[i] if i < len(morceaux) else JOKER
        if segment == JOKER:
            continue
        if i >= len(demandes):
            return False
        if bug == "liste":
            ok = demandes[i] == segment
        elif bug == "sous_chaine":
            ok = demandes[i] in segment
        else:
            ok = demandes[i] in segment.split(LISTE)
        if not ok:
            return False
    return True


_BUGS_COUVRE = {
    "startswith": "Vous testez un DÉBUT de texte (`permission.startswith(...)`) : « client » couvre alors "
                  "« clients:lire », et « clients » couvre « clientsvip:lire ». C'est la faille d'aujourd'hui "
                  "(`regle_actuelle`). Coupez motif et permission en segments (`split(SEPARATEUR)`) et comparez des "
                  "segments ENTIERS.",
    "fnmatch": "`fnmatch` laisse « * » avaler plusieurs segments (« *:exporter » couvre « clients:lire:exporter ») et ne "
               "connaît ni les listes « lire,modifier », ni les segments sous-entendus (« clients » ne couvre plus "
               "« clients:lire »). Comparez segment par segment : « * » vaut UN segment.",
    "zip": "`zip(motif, permission)` s'arrête au plus court : un motif plus PRÉCIS que la permission la couvre alors "
           "(« clients:lire:telephone » couvre « clients:lire » : le droit de lire un numéro donne toute la fiche). Les "
           "segments en trop du motif doivent valoir « * », sinon il ne couvre pas.",
    "longueur": "Vous exigez la même longueur : « clients » ne couvre plus « clients:lire », et « clients:lire:* » ne "
                "couvre plus « clients:lire ». Un motif plus court vaut « * » pour ses segments manquants ; un motif "
                "plus long couvre si ses segments en trop sont des « * ».",
    "liste": "« clients:lire,modifier » est une LISTE de valeurs : il couvre « clients:lire » et « clients:modifier ». "
             "Coupez chaque segment du motif avec `segment.split(LISTE)` et cherchez la valeur demandée dedans.",
    "sous_chaine": "`valeur in segment` est un test de SOUS-CHAÎNE quand `segment` est un texte : « valider » est dans "
                   "« invalider », « lire » dans « relire ». Cherchez la valeur dans la LISTE `segment.split(LISTE)`.",
}

_MOTIFS_278 = ["*", "clients", "client", "clients:*", "clients:lire", "clients:li", "clients:lire:*",
               "clients:lire:telephone", "clients:lire,modifier", "clients:modifier,lire", "clients:lire,exporter:*",
               "*:lire", "*:exporter", "*:*", "*:*:*", "gestes:invalider", "gestes:valider", "gestes",
               "gestes:valider,invalider", "rapports:*", "modeles:*:prod", "clients:lire:telephone,email",
               "clients:*:telephone", "Clients:lire", "clientsvip", "*:lire:*:*", "clients:relire"]
_PERMISSIONS_278 = ["clients:lire", "clients:lire:telephone", "clients:lire:email", "clients:modifier",
                    "clients:exporter", "clients:lire:exporter", "clients", "clientsvip:lire", "gestes:valider",
                    "gestes:proposer", "gestes:invalider", "rapports:lire", "modeles:deployer:prod",
                    "modeles:deployer", "c", "lire", "x:y:z:w", "clients:Lire"]


def _cas_278() -> list[tuple[str, str]]:
    cas = [(m, p) for m in _MOTIFS_278 for p in _PERMISSIONS_278]
    alea = random.Random(2781)
    valeurs = ["a", "b", "ab", "ba"]
    for _ in range(1500):
        permission = SEPARATEUR.join(alea.choice(valeurs) for _ in range(alea.randint(1, 4)))
        segments = []
        for _ in range(alea.randint(1, 4)):
            tirage = alea.random()
            segments.append(JOKER if tirage < 0.25 else LISTE.join(alea.sample(valeurs, alea.randint(2, 3)))
                            if tirage < 0.45 else alea.choice(valeurs))
        cas.append((SEPARATEUR.join(segments), permission))
    return cas


def verifier_permission_couvre(f) -> None:
    cas = _cas_278()
    obtenus = []
    for motif, permission in cas:
        quoi = f"permission_couvre({motif!r}, {permission!r})"
        r = _appel(f, (motif, permission), quoi)
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié un `return` (ou le `return False` de la fin) ?")
        if type(r) is not bool:
            raise AssertionError(f"{quoi} renvoie {_court(r, 60)} ({type(r).__name__}) au lieu de True ou False : rendez "
                                 "un vrai booléen.")
        obtenus.append(r)
    for (motif, permission), r in zip(cas, obtenus):
        attendu = _ref_couvre(motif, permission)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_couvre(m, p, bug) for m, p in cas], _BUGS_COUVRE,
                                   lambda a, b: a == b)
            if not conseil:
                conseil = (("Un motif plus LONG que la permission ne la couvre que si ses segments en trop valent "
                            "« * »." if len(motif.split(SEPARATEUR)) > len(permission.split(SEPARATEUR)) else
                            "Comparez position par position : le segment de la permission doit être l'une des valeurs "
                            "du segment du motif (ou « * »).") if r else
                           "Un motif plus COURT couvre aussi ce qui est en dessous : ses segments manquants valent "
                           "« * ».")
            raise AssertionError(f"permission_couvre({motif!r}, {permission!r}) renvoie {r} au lieu de {attendu}. "
                                 f"{conseil}".rstrip())

    for args, attendu, conseil in (
            (("", "clients:lire"), ValueError, "Un motif vide ne couvre rien : il vient d'une colonne vide, refusez-le "
                                               "(aujourd'hui, `startswith(\"\")` donne TOUT au support informatique)."),
            (("clients:", "clients:lire"), ValueError, "« clients: » a un segment vide."),
            (("clients::lire", "clients:lire"), ValueError, "« clients::lire » a un segment vide."),
            ((":lire", "clients:lire"), ValueError, "« :lire » a un segment vide."),
            (("clients:lire", ""), ValueError, "Une permission vide n'en est pas une."),
            (("clients:lire", "clients:"), ValueError, "« clients: » a un segment vide."),
            (("clients:lire", "clients:*"), ValueError, "On demande une action PRÉCISE : « * » n'est permis que dans "
                                                        "un motif."),
            (("*", "*"), ValueError, "On demande une action PRÉCISE : « * » n'est permis que dans un motif."),
            (("clients:lire", "clients:lire,modifier"), ValueError, "On demande UNE action : une liste n'est permise "
                                                                    "que dans un motif."),
            (("clients:li*", "clients:lire"), ValueError, "« li* » n'est pas un motif : « * » est un segment ENTIER."),
            (("clients:lire,", "clients:lire"), ValueError, "« lire, » contient une valeur vide."),
            (("clients:,lire", "clients:lire"), ValueError, "« ,lire » contient une valeur vide."),
            (("clients:*,lire", "clients:lire"), ValueError, "« * » ne se met pas dans une liste."),
            (("rapports:li*", "clients:lire"), ValueError,
             "Le premier segment ne correspond pas, mais le motif est mal écrit : vérifiez TOUT le motif AVANT de "
             "comparer (une configuration fausse doit se voir, même quand la réponse serait « non »)."),
            (("rapports", "clients:*"), ValueError,
             "Le premier segment ne correspond pas, mais la permission demandée n'est pas précise : vérifiez-la EN "
             "ENTIER avant de comparer."),
            ((None, "clients:lire"), TypeError, "None n'est pas un motif."),
            (("clients:lire", None), TypeError, "None n'est pas une permission."),
            ((["clients", "lire"], "clients:lire"), TypeError, "Une liste n'est pas un motif : passez le texte."),
            ((b"clients", "clients:lire"), TypeError, "Des octets ne sont pas un texte.")):
        _doit_lever(f, args, f"permission_couvre{_court(args, 60)}", conseil, attendu)


# ============================================================================ défi 279 : roles_effectifs


def _ref_roles(noms, heritage, bug: str = "") -> frozenset:
    """Les rôles effectifs de référence ; `bug` rejoue une erreur classique (après les mêmes vérifications)."""
    if isinstance(noms, str) or not isinstance(noms, _COLLECTIONS):
        raise TypeError(f"noms doit être une liste de rôles, pas {type(noms).__name__}")
    if not isinstance(heritage, dict):
        raise TypeError(f"heritage doit être un dict, pas {type(heritage).__name__}")
    for nom in noms:
        if not isinstance(nom, str):
            raise TypeError(f"un nom de rôle est un texte, pas {type(nom).__name__}")
    graphe = heritage
    if bug == "sens_inverse":
        graphe = {role: [] for role in heritage}
        for role, parents in heritage.items():
            for parent in parents:
                graphe.setdefault(parent, []).append(role)
    if bug == "sans_depart":
        a_voir = [parent for nom in noms if nom in heritage for parent in heritage[nom]]
        for nom in noms:
            if nom not in heritage:
                raise ValueError(f"rôle inconnu : {nom!r}")
    else:
        a_voir = list(noms)
    vus: set[str] = set()
    while a_voir:
        role = a_voir.pop()
        if role in vus:
            continue
        if role not in heritage:
            raise ValueError(f"rôle inconnu : {role!r}")
        parents = heritage[role]
        if isinstance(parents, str) or not isinstance(parents, (list, tuple)):
            raise TypeError(f"les parents de {role!r} doivent être une liste")
        vus.add(role)
        if bug == "un_etage" and role not in noms:
            continue
        a_voir.extend(graphe.get(role, []))
    return frozenset(vus)


_BUGS_ROLES = {
    "un_etage": "Vous ne suivez qu'UN étage d'héritage : un manager d'agence a bien le rôle de conseiller senior, mais "
                "pas celui de conseiller (le parent de son parent). Recommencez avec les parents de chaque rôle ajouté, "
                "jusqu'à ce qu'il n'y ait plus rien de nouveau (une pile, ou une file).",
    "sens_inverse": "Vous suivez l'héritage à l'ENVERS : `heritage[role]` donne les rôles dont `role` hérite (ses "
                    "PARENTS), pas ceux qui héritent de lui. Avec votre version, un conseiller deviendrait manager.",
    "sans_depart": "Les rôles de DÉPART manquent : un manager d'agence a aussi le rôle « manager_agence » lui-même. "
                   "Commencez avec `a_voir = list(noms)`, pas avec leurs parents.",
}


def _graphe_aleatoire(graine: int, n: int, cycles: bool) -> dict:
    alea = random.Random(graine)
    roles = [f"r{i:02d}" for i in range(n)]
    heritage = {}
    for i, role in enumerate(roles):
        candidats = roles if cycles else roles[i + 1:]
        heritage[role] = alea.sample(candidats, min(len(candidats), alea.randint(0, 3)))
    return heritage


def _cas_279() -> list[tuple]:
    heritage = heritage_exemple()
    roles = sorted(heritage)
    cas = [([role], heritage) for role in roles]
    cas += [(["stagiaire", "analyste"], heritage), (("conseiller", "conseiller"), heritage),
            ({"direction", "dpo"}, heritage), (frozenset({"data_scientist", "manager_agence"}), heritage),
            (roles, heritage), ([], heritage)]
    cycle = {"a": ["b"], "b": ["a", "c"], "c": []}
    cas += [(["a"], cycle), (["b"], cycle), (["c"], cycle), (["x"], {"x": ["x"]}),
            (["haut"], {"haut": ["g", "d"], "g": ["bas"], "d": ["bas"], "bas": []})]
    for graine, n, cycles in ((2791, 30, False), (2792, 30, True), (2793, 60, True)):
        graphe = _graphe_aleatoire(graine, n, cycles)
        alea = random.Random(graine + 10)
        cas += [(alea.sample(sorted(graphe), alea.randint(1, 3)), graphe) for _ in range(25)]
    return cas


def _losanges(etages: int = 60, largeur: int = 30) -> dict:
    """Chaque rôle d'un étage hérite de 2 ou 3 rôles de l'étage suivant : des chemins par milliards, peu de rôles."""
    alea = random.Random(2794)
    heritage = {"sommet": [f"e0_{j}" for j in range(largeur)]}
    for e in range(etages):
        for j in range(largeur):
            heritage[f"e{e}_{j}"] = ([] if e == etages - 1 else
                                     [f"e{e + 1}_{k}" for k in alea.sample(range(largeur), alea.randint(2, 3))])
    return heritage


def _verifier_resultat_roles(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, (set, frozenset)):
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} ({type(r).__name__}) au lieu d'un ensemble : rendez un "
                             "`frozenset` (ou un `set`). Un ensemble n'a ni ordre ni doublon, comme des rôles.")
    for role in r:
        if not isinstance(role, str):
            raise AssertionError(f"{quoi} contient {role!r} ({type(role).__name__}) : un rôle est un texte.")


def verifier_roles_effectifs(f) -> None:
    cas = _cas_279()
    obtenus = []
    for noms, heritage in cas:
        quoi = f"roles_effectifs({_court(noms, 60)}, heritage)"
        avant = copy.deepcopy((noms, heritage))
        r = _appel(f, (noms, heritage), quoi, "Un cycle d'héritage (« a » hérite de « b », qui hérite de « a ») vous "
                   "fait tourner en rond : gardez l'ensemble des rôles déjà vus, et ne repartez jamais d'un rôle vu.")
        _verifier_resultat_roles(r, quoi)
        _sans_modification(avant, (noms, heritage), "les noms ou le dict `heritage`")
        obtenus.append(r)
    for (noms, heritage), r in zip(cas, obtenus):
        attendu = _ref_roles(noms, heritage)
        if set(r) != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_roles(n, h, bug) for n, h in cas], _BUGS_ROLES,
                                   lambda a, b: len(a) == len(b) and all(set(x) == set(y) for x, y in zip(a, b)))
            if not conseil:
                conseil = ("Il vous manque des rôles : suivez les parents de CHAQUE rôle ajouté, à n'importe quelle "
                           "profondeur." if set(r) < attendu else
                           "Vous avez des rôles en trop : seuls les rôles de départ et leurs ancêtres (les parents, les "
                           "parents des parents…) sont effectifs.")
            graphe = "heritage_exemple()" if heritage == heritage_exemple() else _court(heritage, 120)
            raise AssertionError(f"roles_effectifs({_court(noms, 80)}, {graphe}) renvoie {sorted(r)} au lieu de "
                                 f"{sorted(attendu)}. {conseil}".rstrip())

    losanges = _losanges()
    conseil = ("Le correcteur compte les lignes exécutées dans votre fichier : vous repassez par les mêmes rôles une fois "
               "par CHEMIN qui y mène (deux parents qui ont les mêmes deux parents, sur 60 étages, c'est plus de 2 ** 60 "
               "chemins). Gardez l'ensemble des rôles déjà vus (`vus`), et ne développez jamais deux fois le même rôle.")
    r = _appel(f, (["sommet"], losanges), "roles_effectifs(['sommet'], heritage) (1 801 rôles sur 60 étages)",
               conseil, lignes_max=60 * (len(losanges) + sum(map(len, losanges.values()))), conseil_lent=conseil)
    _verifier_resultat_roles(r, "roles_effectifs(['sommet'], heritage)")
    if set(r) != _ref_roles(["sommet"], losanges):
        raise AssertionError(f"Sur 1 801 rôles en losanges, roles_effectifs(['sommet'], heritage) renvoie {len(r)} rôles "
                             f"au lieu de {len(losanges)} (alors que les petits cas passent).")

    for args, attendu, conseil in (
            (("manager_agence", heritage_exemple()), TypeError,
             "Un texte seul n'est pas une liste de rôles : `for nom in \"manager_agence\"` parcourt ses LETTRES. "
             "Vérifiez `isinstance(noms, str)` d'abord."),
            ((None, heritage_exemple()), TypeError, "None n'est pas une liste de rôles."),
            (([1], heritage_exemple()), TypeError, "Un nom de rôle est un texte."),
            ((["conseiller"], [("conseiller", [])]), TypeError, "heritage est un dict."),
            ((["manger_agence"], heritage_exemple()), ValueError,
             "Un rôle inconnu (une faute de frappe dans la configuration) se refuse : ValueError. L'ignorer cache "
             "l'erreur, et le jour où quelqu'un crée ce rôle, il reçoit des droits que personne n'a décidés."),
            ((["a"], {"a": ["b"]}), ValueError,
             "« b » est un parent de « a », mais n'est pas dans `heritage` : rôle inconnu, ValueError."),
            ((["a"], {"a": "conseiller", "conseiller": []}), TypeError,
             "Les parents de « a » sont un texte, pas une liste : `extend(\"conseiller\")` ajouterait ses LETTRES. "
             "TypeError.")):
        _doit_lever(f, args, f"roles_effectifs{_court(args, 70)}", conseil, attendu)


# ============================================================================ défi 280 : decider


def _verifier_regles(regles, permission, couvre=None) -> list[bool]:
    couvre = couvre or _ref_couvre
    if not isinstance(regles, (list, tuple)):
        raise TypeError(f"regles doit être une liste, pas {type(regles).__name__}")
    ids, couvertures = set(), []
    for regle in regles:
        if not isinstance(regle, dict):
            raise TypeError("une règle est un dict")
        for cle in ("id", "effet", "permission"):
            if cle not in regle:
                raise ValueError(f"une règle n'a pas de {cle!r}")
        if regle["id"] in ids:
            raise ValueError(f"deux règles s'appellent {regle['id']!r}")
        ids.add(regle["id"])
        if regle["effet"] not in (AUTORISER, REFUSER):
            raise ValueError(f"effet inconnu {regle['effet']!r}")
        if "roles" in regle and (isinstance(regle["roles"], str) or not isinstance(regle["roles"], _COLLECTIONS)):
            raise TypeError("roles doit être une liste")
        si = regle.get("si", [])
        if isinstance(si, str) or not isinstance(si, (list, tuple)):
            raise TypeError("si doit être une liste")
        for condition in si:
            if condition not in CONDITIONS:
                raise ValueError(f"condition inconnue {condition!r}")
        couvertures.append(couvre(regle["permission"], permission))
    return couvertures


def _condition(nom: str, utilisateur: dict, client: dict, bug: str) -> bool:
    if bug == "erreur_avalee":
        try:
            return CONDITIONS[nom](utilisateur, client)
        except KeyError:
            return False
    return CONDITIONS[nom](utilisateur, client)


def _ref_decider(utilisateur, permission, client, regles, bug: str = "") -> Decision:
    """La décision de référence ; `bug` rejoue une erreur classique (après les mêmes vérifications)."""
    if not isinstance(utilisateur, dict) or not isinstance(client, dict):
        raise TypeError("utilisateur et client sont des dict")
    roles = utilisateur["roles"]
    if isinstance(roles, str) or not isinstance(roles, _COLLECTIONS):
        raise TypeError("les rôles de l'utilisateur sont une liste")
    couvertures = _verifier_regles(regles, permission)
    roles = set(roles)
    refus, autorisations = [], []
    for regle, couvre in zip(regles, couvertures):
        if bug == "egalite":
            couvre = regle["permission"] == permission
        if not couvre:
            continue
        if bug == "roles_vides_tous":
            if regle.get("roles") and not roles & set(regle["roles"]):
                continue
        elif bug != "sans_roles" and "roles" in regle and not roles & set(regle["roles"]):
            continue
        si = regle.get("si", [])
        if bug == "une_condition":
            ok = not si or any(_condition(nom, utilisateur, client, bug) for nom in si)
        else:
            ok = bug == "sans_conditions" or all(_condition(nom, utilisateur, client, bug) for nom in si)
        if not ok:
            continue
        if regle["effet"] == REFUSER:
            if not bug:
                return Decision(False, regle["id"])
            refus.append(regle["id"])
        else:
            autorisations.append(regle["id"])
        if bug == "premiere":
            return Decision(regle["effet"] == AUTORISER, regle["id"])
    if bug == "dernier_id":
        refus, autorisations = refus[::-1], autorisations[::-1]
    if bug == "autorisation_gagne" and autorisations:
        return Decision(True, autorisations[0])
    if refus:
        return Decision(False, refus[0])
    if autorisations:
        return Decision(True, autorisations[0])
    return Decision(bug == "defaut_autorise", None)


_BUGS_DECIDER = {
    "premiere": "La PREMIÈRE règle qui s'applique décide chez vous (comme dans un pare-feu) : une autorisation placée "
                "avant un refus l'emporte, et l'ordre des règles change la décision. Ici, un refus qui s'applique "
                "l'emporte TOUJOURS : regardez toutes les règles, puis décidez.",
    "autorisation_gagne": "Chez vous, une autorisation l'emporte sur un refus : la fiche d'un salarié devient lisible "
                          "dès qu'une règle générale l'autorise. C'est l'inverse : si un REFUS s'applique, c'est refusé.",
    "defaut_autorise": "Quand aucune règle ne s'applique, vous AUTORISEZ. Le défaut est le REFUS : "
                       "`Decision(False, None)`. Un oubli dans la configuration doit fermer une porte, pas l'ouvrir.",
    "sans_conditions": "Vous ignorez les conditions (\"si\") : la règle « clients de MON agence » s'applique à tous les "
                       "clients. Une règle ne s'applique que si TOUTES ses conditions sont vraies : "
                       "`CONDITIONS[nom](utilisateur, client)`.",
    "une_condition": "Une seule condition vraie suffit chez vous (`any`) : il les faut TOUTES (`all`). « meme_agence » "
                     "ET « meme_region », ce n'est pas l'un ou l'autre.",
    "roles_vides_tous": "Une règle avec \"roles\": [] vise PERSONNE, mais votre `if not regle.get(\"roles\")` la "
                        "traite comme une règle sans \"roles\" (tout le monde). Testez la PRÉSENCE de la clé "
                        "(`\"roles\" in regle`), pas sa « vérité ».",
    "sans_roles": "Vous ignorez les rôles des règles : un stagiaire profite de la règle des managers. Une règle avec "
                  "\"roles\" ne s'applique qu'à quelqu'un qui a au moins un de ces rôles (`set(roles) & "
                  "set(regle[\"roles\"])`).",
    "egalite": "Vous comparez le motif à la permission avec `==` : « clients:* » ou « clients:lire,modifier » ne "
               "couvrent alors rien. Utilisez `permission_couvre(regle[\"permission\"], permission)` (défi 278 : le "
               "correcteur vous prête la sienne).",
    "dernier_id": "La décision est juste, mais pas la règle citée : c'est l'id du PREMIER refus (ou de la PREMIÈRE "
                  "autorisation) dans l'ordre de la liste, celui qu'on écrira dans le journal.",
}


def _effectifs(roles) -> set:
    return set(_ref_roles(roles, heritage_exemple()))


_UTILISATEURS_280 = [
    {"roles": _effectifs(["manager_agence"]), "agence": "Lille", "region": "Nord"},
    {"roles": sorted(_effectifs(["conseiller"])), "agence": "Lille", "region": "Nord"},
    {"roles": tuple(_effectifs(["stagiaire"])), "agence": "Lille", "region": "Nord"},
    {"roles": frozenset(_effectifs(["directeur_regional"])), "agence": "Lille", "region": "Nord"},
    {"roles": _effectifs(["analyste"]), "agence": "Siège", "region": "Siège"},
    {"roles": _effectifs(["data_scientist"]), "agence": "Siège", "region": "Siège"},
    {"roles": _effectifs(["dpo"]), "agence": "Siège", "region": "Siège"},
    {"roles": _effectifs(["prestataire_sms"]), "agence": "Siège", "region": "Siège"},
    {"roles": _effectifs(["admin_technique"]), "agence": "Siège", "region": "Siège"},
    {"roles": _effectifs(["direction"]), "agence": "Siège", "region": "Siège"},
    {"roles": _effectifs(["conseiller", "analyste"]), "agence": "Rouen", "region": "Nord"},
    {"roles": [], "agence": "Lille", "region": "Nord"},
    {"roles": _effectifs(["manager_agence", "dpo"]), "agence": "Rouen", "region": "Nord"},
]
_PERMISSIONS_280 = ["clients:lire", "clients:lire:telephone", "clients:modifier", "clients:exporter",
                    "gestes:proposer", "gestes:valider", "rapports:lire", "modeles:deployer", "clientsvip:lire"]
_CLIENTS_280 = [
    {"agence": "Lille", "region": "Nord", "salarie": False, "consentement_sms": True},
    {"agence": "Lille", "region": "Nord", "salarie": True, "consentement_sms": True},
    {"agence": "Lille", "region": "Nord", "salarie": False, "consentement_sms": False},
    {"agence": "Rouen", "region": "Nord", "salarie": False, "consentement_sms": True},
    {"agence": "Marseille", "region": "Sud", "salarie": True, "consentement_sms": False},
    {"agence": "Siège", "region": "Siège", "salarie": False, "consentement_sms": True},
]


def _regles_280() -> list[list[dict]]:
    pieges = [
        {"id": "a1", "effet": AUTORISER, "permission": "clients:lire", "roles": []},
        {"id": "a2", "effet": AUTORISER, "permission": "clients", "roles": ["conseiller"],
         "si": ["meme_agence", "meme_region"]},
        {"id": "a3", "effet": AUTORISER, "permission": "rapports:lire"},
        {"id": "d1", "effet": REFUSER, "permission": "clients:exporter", "si": ["autre_region"]},
        {"id": "a4", "effet": AUTORISER, "permission": "clients:exporter,modifier", "roles": ["manager_agence"]},
        {"id": "d2", "effet": REFUSER, "permission": "rapports:*", "roles": ["stagiaire"]},
        {"id": "a5", "effet": AUTORISER, "permission": "gestes:valider,proposer", "roles": ["manager_agence", "dpo"],
         "si": ["meme_agence"]},
        {"id": "d3", "effet": REFUSER, "permission": "gestes:valider", "roles": ["dpo"]},
        {"id": "a6", "effet": AUTORISER, "permission": "*:lire", "roles": ["analyste"],
         "si": ["autre_agence", "client_salarie"]},
        {"id": "a7", "effet": AUTORISER, "permission": "clients:lire:*", "roles": ("prestataire_sms", "dpo")},
        {"id": "d4", "effet": REFUSER, "permission": "clients:lire:telephone", "si": ["sans_consentement_sms"],
         "roles": []},
        {"id": "a8", "effet": AUTORISER, "permission": "modeles:*", "roles": {"admin_technique"}, "si": []},
    ]
    exemple = regles_exemple()
    return [exemple, exemple[::-1], pieges, pieges[::-1], tuple(exemple[6:] + pieges[:6])]


def _cas_280() -> list[tuple]:
    """Des copies neuves à chaque appel : un code qui modifie ce qu'il reçoit n'abîme pas les cas suivants."""
    utilisateurs, clients = copy.deepcopy(_UTILISATEURS_280), copy.deepcopy(_CLIENTS_280)
    return [(u, p, c, regles) for regles in _regles_280() for u in utilisateurs for p in _PERMISSIONS_280
            for c in clients]


def _decrire_cas_280(utilisateur, permission, client, regles) -> str:
    if regles == regles_exemple():
        nom = "regles_exemple()"
    elif regles == regles_exemple()[::-1]:
        nom = "regles_exemple()[::-1] (les mêmes règles, dans l'ordre inverse)"
    else:
        nom = "regles = " + _court(regles, 300) + " (toutes les règles : `_regles_280()` dans defis/checks_serie_46.py)"
    roles = sorted(utilisateur["roles"])
    return (f"decider({{'roles': {roles}, 'agence': {utilisateur['agence']!r}, 'region': {utilisateur['region']!r}}}, "
            f"{permission!r}, {client!r}, regles), avec {nom}")


def verifier_decider(f) -> None:
    with _Pretes(f, {"permission_couvre": _ref_couvre}):
        _verifier_decider(f)


def _verifier_decider(f) -> None:
    cas = _cas_280()
    obtenus = []
    for utilisateur, permission, client, regles in cas:
        quoi = _decrire_cas_280(utilisateur, permission, client, regles)
        r = _appel(f, (utilisateur, permission, client, regles), quoi,
                   au_lieu={KeyError: "Une clé absente (KeyError) : \"roles\" et \"si\" sont FACULTATIFS dans une règle "
                                      "(sans \"roles\", elle vise tout le monde ; sans \"si\", elle n'a pas de "
                                      "condition)."})
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, Decision):
            raise AssertionError(f"{quoi} renvoie {_court(r, 80)} ({type(r).__name__}) au lieu d'une Decision : rendez "
                                 "`Decision(True, id)`, `Decision(False, id)` ou `Decision(False, None)` (la classe "
                                 "est dans defis/acces.py).")
        if type(r.autorise) is not bool:
            raise AssertionError(f"{quoi} renvoie {r!r} : `autorise` doit valoir True ou False.")
        obtenus.append(r)
    for (utilisateur, permission, client, regles), r in zip(cas, obtenus):
        attendu = _ref_decider(utilisateur, permission, client, regles)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_decider(*x, bug=bug) for x in cas], _BUGS_DECIDER,
                                   lambda a, b: a == b)
            if not conseil and r.delegation is not None:
                conseil = "`delegation` reste None au défi 280 (elle sert au défi 283)."
            raise AssertionError(f"{_decrire_cas_280(utilisateur, permission, client, regles)} renvoie {r!r} au lieu "
                                 f"de {attendu!r}. {conseil}".rstrip())

    # ------------------------------------------------------------------ rien n'est modifié
    utilisateur, client, regles = copy.deepcopy((_UTILISATEURS_280[0], _CLIENTS_280[1], _regles_280()[2]))
    avant = copy.deepcopy((utilisateur, client, regles))
    for permission in _PERMISSIONS_280:
        _appel(f, (utilisateur, permission, client, regles), f"decider(..., {permission!r}, ...)")
    _sans_modification(avant, (utilisateur, client, regles), "l'utilisateur, le client ou les règles")

    # ------------------------------------------------------------------ une condition qu'on ne sait pas évaluer
    manager = _UTILISATEURS_280[0]
    for client, conseil in (
            ({"agence": "Lille", "region": "Nord", "consentement_sms": True},
             "Ce client n'a pas d'attribut « salarie » : la condition « client_salarie » de R13 (un REFUS) ne peut pas "
             "être évaluée. Si vous la tenez pour fausse, le refus saute et la fiche s'ouvre. Laissez remonter la "
             "KeyError (ne l'attrapez pas)."),
            ({"region": "Nord", "salarie": False, "consentement_sms": True},
             "Ce client n'a pas d'attribut « agence » : la condition « meme_agence » ne peut pas être évaluée. Laissez "
             "remonter la KeyError.")):
        r = None
        try:
            r = f(manager, "clients:lire", client, regles_exemple())
        except NotImplementedError:
            raise
        except (KeyError, ValueError):
            continue
        except Exception as exc:
            raise AssertionError(f"decider(manager de Lille, 'clients:lire', {client!r}, regles_exemple()) lève "
                                 f"{_decrire(exc)} au lieu de KeyError. {conseil}") from None
        raise AssertionError(f"decider(manager de Lille, 'clients:lire', {client!r}, regles_exemple()) renvoie {r!r} au "
                             f"lieu de lever KeyError. {conseil}")

    # ------------------------------------------------------------------ une configuration mal écrite est refusée
    lille = {"agence": "Lille", "region": "Nord", "salarie": False, "consentement_sms": True}
    refus_d_abord = {"id": "X", "effet": REFUSER, "permission": "*"}
    for regles, attendu, conseil in (
            ([{"id": "A", "effet": "refuse", "permission": "clients"}], ValueError,
             "« refuse » n'est pas un effet (AUTORISER ou REFUSER) : une faute de frappe ne doit pas devenir une "
             "autorisation (ni être ignorée)."),
            ([{"id": "A", "effet": "Autoriser", "permission": "clients"}], ValueError,
             "« Autoriser » (avec une majuscule) n'est pas AUTORISER."),
            ([{"id": "A", "permission": "clients"}], ValueError, "Une règle sans \"effet\" est mal écrite."),
            ([{"effet": AUTORISER, "permission": "clients"}], ValueError, "Une règle sans \"id\" est mal écrite."),
            ([{"id": "A", "effet": AUTORISER}], ValueError, "Une règle sans \"permission\" est mal écrite."),
            ([{"id": "A", "effet": AUTORISER, "permission": "clients"},
              {"id": "A", "effet": REFUSER, "permission": "rapports"}], ValueError,
             "Deux règles ont l'id « A » : le journal ne saurait plus laquelle a décidé."),
            ([{"id": "A", "effet": AUTORISER, "permission": "clients", "si": ["meme_agense"]}], ValueError,
             "« meme_agense » n'est pas une condition de CONDITIONS."),
            ([{"id": "A", "effet": AUTORISER, "permission": "clients::lire"}], ValueError,
             "« clients::lire » n'est pas un motif valide (défi 278)."),
            ([{"id": "A", "effet": AUTORISER, "permission": "clients", "roles": "conseiller"}], TypeError,
             "\"roles\" est un texte au lieu d'une liste : `set(\"conseiller\")` donne ses LETTRES."),
            ([{"id": "A", "effet": AUTORISER, "permission": "clients", "si": "meme_agence"}], TypeError,
             "\"si\" est un texte au lieu d'une liste."),
            ([refus_d_abord, {"id": "B", "effet": "refuse", "permission": "clients"}], ValueError,
             "La règle X suffit à refuser, mais la règle B est mal écrite : vérifiez TOUTES les règles AVANT de "
             "décider (une erreur de configuration doit se voir tout de suite, pas le jour où X change)."),
            ([refus_d_abord, {"id": "B", "effet": AUTORISER, "permission": "clients", "si": ["inconnue"]}], ValueError,
             "La règle X suffit à refuser, mais la condition de la règle B n'existe pas : vérifiez TOUTES les règles "
             "AVANT de décider."),
            ({"id": "A", "effet": AUTORISER, "permission": "clients"}, TypeError,
             "Les règles sont une LISTE de dict, pas un dict.")):
        _doit_lever(f, (manager, "clients:lire", lille, regles), f"decider(manager, 'clients:lire', client, "
                    f"{_court(regles, 140)})", conseil, attendu)
    for args, attendu, conseil in (
            (({"roles": "manager_agence", "agence": "Lille", "region": "Nord"}, "clients:lire", lille, regles_exemple()),
             TypeError, "Les rôles de l'utilisateur sont un texte : `set(\"manager_agence\")` donne ses LETTRES."),
            ((manager, "clients:*", lille, regles_exemple()), ValueError,
             "On demande une permission PRÉCISE (défi 278)."),
            ((manager, "clients:lire", None, regles_exemple()), TypeError, "Le client est un dict.")):
        _doit_lever(f, args, f"decider{_court(args, 90)}", conseil, attendu)


# ============================================================================ défi 281 : droits_dormants


def _ref_dormants(utilisateurs, heritage, regles, journal, maintenant, jours=90, bug: str = "") -> dict:
    """Les droits dormants de référence ; `bug` rejoue une erreur classique (après les mêmes vérifications)."""
    _instant(maintenant, "maintenant")
    if isinstance(jours, bool) or not isinstance(jours, int):
        raise TypeError("jours est un entier")
    if jours <= 0:
        raise ValueError("jours doit être > 0")
    debut = maintenant - timedelta(days=jours)
    utilises, permissions = set(), defaultdict(set)
    for entree in journal:
        if entree["regle"] is None:
            continue
        date = entree["date"]
        if bug == "sans_periode":
            dedans = True
        elif bug == "debut_exclu":
            dedans = debut < date <= maintenant
        elif bug == "futur_compte":
            dedans = debut <= date
        else:
            dedans = debut <= date <= maintenant
        if dedans:
            utilises.add((entree["utilisateur"], entree["regle"]))
            if bug == "par_permission":
                permissions[entree["utilisateur"]].add(entree["permission"])
    resultat = {}
    for utilisateur in utilisateurs:
        identifiant = utilisateur["identifiant"]
        roles = set(utilisateur["roles"]) if bug == "sans_heritage" else _ref_roles(utilisateur["roles"], heritage)
        ids = []
        for regle in regles:
            if regle["effet"] != AUTORISER and bug != "refus_inclus":
                continue
            if bug == "roles_vides_tous":
                vise = not regle.get("roles") or roles & set(regle["roles"])
            else:
                vise = "roles" not in regle or roles & set(regle["roles"])
            if not vise:
                continue
            if bug == "par_permission":
                utilise = any(_ref_couvre(regle["permission"], p) for p in permissions[identifiant])
            else:
                utilise = (identifiant, regle["id"]) in utilises
            if not utilise:
                ids.append(regle["id"])
        if bug != "non_tries":
            ids.sort()
        if ids or bug == "vides_inclus":
            resultat[identifiant] = ids
    return resultat


_BUGS_DORMANTS = {
    "sans_heritage": "Vous prenez les rôles DIRECTS des comptes : un manager d'agence n'aurait pas les droits d'un "
                     "conseiller, et les droits hérités (les plus oubliés) n'apparaissent jamais dans la revue. "
                     "Utilisez `roles_effectifs(u[\"roles\"], heritage)` (le correcteur vous prête la sienne).",
    "refus_inclus": "Un refus n'est pas un droit : on ne « retire » pas un refus faute de s'en être servi. Ne gardez "
                    "que les règles AUTORISER.",
    "sans_periode": "Vous comptez TOUT le journal : un export d'il y a 200 jours n'est pas un usage récent. Seules les "
                    "entrées datées de `maintenant - timedelta(days=jours)` à `maintenant` comptent.",
    "debut_exclu": "Une entrée tombée PILE au début de la période (`maintenant - timedelta(days=jours)`) compte : les "
                   "bornes sont COMPRISES (`debut <= date <= maintenant`), et votre `debut < date` l'oublie.",
    "futur_compte": "Une entrée datée APRÈS `maintenant` ne compte pas (on fait la revue « au 2 novembre » d'un journal "
                    "qui continue) : `debut <= date <= maintenant`.",
    "par_permission": "Vous jugez l'usage par la PERMISSION (« il a lu des fiches ») au lieu de la RÈGLE : un compte qui "
                      "lit grâce à une règle n'utilise pas les AUTRES règles qui donnent la même lecture. Cherchez le "
                      "couple (utilisateur, id de la règle) dans le journal.",
    "vides_inclus": "Ne gardez que les comptes qui ont AU MOINS un droit dormant : pas d'entrée avec une liste vide.",
    "roles_vides_tous": "Une règle avec \"roles\": [] ne vise PERSONNE : ce n'est le droit de personne. Votre "
                        "`not regle.get(\"roles\")` la traite comme une règle sans \"roles\" (tout le monde). Testez "
                        "`\"roles\" in regle`.",
    "non_tries": "Les ids de chaque compte doivent être triés (`sorted`), pas dans l'ordre des règles.",
}


def _cas_281() -> list[tuple]:
    """(utilisateurs, heritage, regles, journal, maintenant, jours) ; le dernier est l'exemple complet."""
    heritage = {"conseiller": [], "manager": ["conseiller"], "stagiaire": ["conseiller"], "audit": []}
    regles = [
        {"id": "R9", "effet": AUTORISER, "permission": "clients:lire", "roles": ["conseiller"], "si": ["meme_agence"]},
        {"id": "R2", "effet": AUTORISER, "permission": "clients:exporter", "roles": ["manager"]},
        {"id": "R5", "effet": AUTORISER, "permission": "clients", "roles": ["audit"]},
        {"id": "R1", "effet": REFUSER, "permission": "clients:exporter", "roles": ["stagiaire"]},
        {"id": "R7", "effet": AUTORISER, "permission": "rapports:lire", "roles": []},
        {"id": "R3", "effet": AUTORISER, "permission": "rapports:lire"},
    ]
    utilisateurs = [{"identifiant": "m1", "roles": ["manager"]}, {"identifiant": "c1", "roles": ("conseiller",)},
                    {"identifiant": "s1", "roles": ["stagiaire"]}, {"identifiant": "a1", "roles": ["audit"]},
                    {"identifiant": "x1", "roles": []}, {"identifiant": "m2", "roles": ["manager", "audit"]}]
    m = REVUE_LE
    debut = m - timedelta(days=90)

    def e(jours, qui, permission, regle):
        return {"date": m - timedelta(days=jours) if isinstance(jours, int) else jours, "utilisateur": qui,
                "permission": permission, "regle": regle, "objet": "C0001"}

    journal = [
        e(1, "m1", "clients:lire", "R9"), e(debut, "m1", "clients:exporter", "R2"), e(0, "m1", "rapports:lire", "R3"),
        e(-1, "c1", "clients:lire", "R9"), e(40, "c1", "rapports:lire", "R3"),
        e(100, "a1", "clients:modifier", "R5"), e(2, "a1", "rapports:lire", "R3"),
        e(5, "s1", "clients:exporter", "R1"), e(5, "s1", "clients:exporter", None), e(3, "s1", "rapports:lire", "R3"),
        e(10, "m2", "clients:lire", "R5"), e(11, "m2", "clients:exporter", "R2"), e(12, "m2", "rapports:lire", "R3"),
        e(13, "x1", "rapports:lire", "R3"), e(debut - timedelta(seconds=1), "x1", "rapports:lire", "R7"),
        e(20, "fantome", "clients:lire", "R9"),
    ]
    paris = timezone(timedelta(hours=1))
    autre = [dict(entree, date=entree["date"].astimezone(paris)) for entree in journal]
    cas = [(utilisateurs, heritage, regles, journal, m, 90),
           (utilisateurs, heritage, regles, autre, m.astimezone(paris), 90),
           (utilisateurs, heritage, regles[::-1], journal, m, 30),
           (utilisateurs, heritage, regles, journal, m - timedelta(days=12), 1),
           (utilisateurs[:0], heritage, regles, journal, m, 90),
           (utilisateurs, heritage, regles, [], m, 90)]
    exemple = [{"identifiant": "u1", "roles": ["manager"]}, {"identifiant": "u2", "roles": ["conseiller"]},
               {"identifiant": "u3", "roles": ["conseiller"]}]
    regles_doc = [{"id": "L", "effet": AUTORISER, "permission": "clients:lire", "roles": ["conseiller"]},
                  {"id": "E", "effet": AUTORISER, "permission": "clients:exporter", "roles": ["manager"]},
                  {"id": "S", "effet": REFUSER, "permission": "clients", "roles": ["conseiller"],
                   "si": ["client_salarie"]}]
    journal_doc = [{"date": m - timedelta(days=3), "utilisateur": "u2", "regle": "L", "permission": "clients:lire"},
                   {"date": m - timedelta(days=200), "utilisateur": "u1", "regle": "E",
                    "permission": "clients:exporter"},
                   {"date": m - timedelta(days=1), "utilisateur": "u1", "regle": "L", "permission": "clients:lire"}]
    cas.append((exemple, {"conseiller": [], "manager": ["conseiller"]}, regles_doc, journal_doc, m, 90))
    cas.append((utilisateurs_exemple(), heritage_exemple(), regles_exemple(), journal_exemple(), REVUE_LE, 90))
    return cas


def _decrire_cas_281(cas) -> str:
    utilisateurs, heritage, regles, journal, maintenant, jours = cas
    if len(journal) > 1000:
        return ("droits_dormants(utilisateurs_exemple(), heritage_exemple(), regles_exemple(), journal_exemple(), "
                "REVUE_LE)")
    return (f"droits_dormants(utilisateurs, heritage, regles, journal, {maintenant.isoformat()}, {jours}), avec "
            f"utilisateurs = {_court(utilisateurs, 200)}, heritage = {heritage!r}, regles = {_court(regles, 420)}, et un "
            f"journal de {len(journal)} entrées (le cas complet : `_cas_281()` dans defis/checks_serie_46.py)")


def _verifier_resultat_dormants(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} ({type(r).__name__}) au lieu d'un dict {{identifiant: "
                             "[ids]}.")
    for cle, ids in r.items():
        if not isinstance(ids, list):
            raise AssertionError(f"{quoi} associe à {cle!r} {_court(ids, 60)} ({type(ids).__name__}) au lieu d'une "
                                 "liste d'ids triés.")


def verifier_droits_dormants(f) -> None:
    with _Pretes(f, {"roles_effectifs": _ref_roles, "permission_couvre": _ref_couvre}):
        _verifier_dormants(f)


def _verifier_dormants(f) -> None:
    cas = _cas_281()
    conseil_lent = ("Le correcteur compte les lignes exécutées dans votre fichier : vous relisez le journal (172 000 "
                    "entrées) pour chaque compte ou pour chaque droit, soit des dizaines de millions d'étapes. Lisez-le "
                    "UNE fois et rangez dans un `set` les couples (utilisateur, règle) utilisés pendant la période ; "
                    "ensuite, chaque test `(identifiant, id) in utilises` est immédiat.")
    obtenus = []
    for x in cas:
        quoi = _decrire_cas_281(x)
        avant = copy.deepcopy(x[:4]) if len(x[3]) < 1000 else None
        gros = len(x[3]) > 1000
        r = _appel(f, x, quoi, lignes_max=8 * len(x[3]) + 400 * len(x[0]) + 2000 if gros else None,
                   conseil_lent=conseil_lent)
        _verifier_resultat_dormants(r, quoi)
        if avant is not None:
            _sans_modification(avant, x[:4], "les comptes, l'héritage, les règles ou le journal")
        obtenus.append(r)
    for x, r in zip(cas, obtenus):
        attendu = _ref_dormants(*x)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_dormants(*y, bug=bug) for y in cas], _BUGS_DORMANTS,
                                   lambda a, b: a == b)
            ecarts = [cle for cle in sorted(set(r) | set(attendu), key=str) if r.get(cle) != attendu.get(cle)][:3]
            detail = " ; ".join(f"pour {cle!r} : {_court(r[cle], 80) if cle in r else 'absent'} au lieu de "
                                f"{_court(attendu[cle], 80) if cle in attendu else 'absent'}" for cle in ecarts)
            raise AssertionError(f"{_decrire_cas_281(x)} est faux ({detail}). {conseil}".rstrip())

    # ------------------------------------------------------------------ un journal qu'on ne lit qu'une fois
    utilisateurs, heritage, regles, journal, maintenant, jours = cas[0]
    r = _appel(f, (utilisateurs, heritage, regles, iter(journal), maintenant, jours),
               "droits_dormants(..., iter(journal), ...)")
    if r != _ref_dormants(*cas[0]):
        raise AssertionError("Avec un journal qu'on ne peut lire qu'UNE fois (un itérateur, comme un fichier lu ligne à "
                             "ligne ou une requête à une base), votre résultat change : vous le parcourez plusieurs "
                             "fois, et la deuxième lecture est vide. Lisez-le une seule fois, au début.")

    # ------------------------------------------------------------------ erreurs
    naif = REVUE_LE.replace(tzinfo=None)
    for args, attendu, conseil in (
            ((naif, 90), ValueError, "Une date sans fuseau ne se compare pas aux dates du journal (en UTC) : ValueError."),
            (("2026-11-02", 90), TypeError, "Un texte n'est pas une date : TypeError."),
            ((REVUE_LE, 0), ValueError, "Une période de 0 jour n'a pas de sens : ValueError."),
            ((REVUE_LE, -30), ValueError, "Une période négative n'a pas de sens : ValueError."),
            ((REVUE_LE, True), TypeError, "True n'est pas un nombre de jours."),
            ((REVUE_LE, 1.5), TypeError, "Un nombre de jours est un entier.")):
        _doit_lever(f, (utilisateurs, heritage, regles, journal) + args,
                    f"droits_dormants(..., {_court(args[0], 40)}, {args[1]!r})", conseil, attendu)


# ============================================================================ défi 282 : conflits


def _vise(regle: dict, roles, bug: str = "") -> bool:
    if bug == "roles_vides_tous":
        return not regle.get("roles") or bool(set(roles) & set(regle["roles"]))
    return "roles" not in regle or bool(set(roles) & set(regle["roles"]))


def _ref_a_le_droit(roles, permission: str, regles, bug: str = "") -> bool:
    accorde = False
    for regle in regles:
        if not _vise(regle, roles, bug):
            continue
        couvre = (regle["permission"] == permission if bug == "egalite"
                  else _ref_couvre(regle["permission"], permission))
        if not couvre:
            continue
        if regle["effet"] == REFUSER:
            if bug == "refus_ignores":
                continue
            if not regle.get("si") or bug == "refus_conditionnels":
                return False
        elif bug != "conditions_exigees" or not regle.get("si"):
            accorde = True
    return accorde


def _ref_conflits(utilisateurs, heritage, regles, incompatibles, bug: str = "") -> list:
    """Les conflits de référence ; `bug` rejoue une erreur classique (après les mêmes vérifications)."""
    paires = []
    for paire in incompatibles:
        if isinstance(paire, str) or not isinstance(paire, (list, tuple)):
            raise TypeError("une incompatibilité est un couple")
        if len(paire) != 2:
            raise ValueError("une incompatibilité relie deux permissions")
        paires.append(tuple(paire))
    roles_de = {}
    for utilisateur in utilisateurs:
        roles_de[utilisateur["identifiant"]] = (set(utilisateur["roles"]) if bug == "sans_heritage" else
                                                _ref_roles(utilisateur["roles"], heritage))

    def conflit(roles, premiere, seconde) -> bool:
        if bug == "par_role":
            return any(_ref_a_le_droit({r}, premiere, regles) and _ref_a_le_droit({r}, seconde, regles) for r in roles)
        return _ref_a_le_droit(roles, premiere, regles, bug) and _ref_a_le_droit(roles, seconde, regles, bug)

    if bug == "paires_d_abord":
        return [(u["identifiant"], p, s) for p, s in paires for u in utilisateurs
                if conflit(roles_de[u["identifiant"]], p, s)]
    return [(u["identifiant"], p, s) for u in utilisateurs for p, s in paires
            if conflit(roles_de[u["identifiant"]], p, s)]


_BUGS_CONFLITS = {
    "par_role": "Vous cherchez le conflit rôle par rôle : aucun rôle ne permet, SEUL, de proposer et de valider. C'est "
                "le CUMUL qui est interdit (un manager valide par « manager_agence » et propose par « conseiller », "
                "dont il hérite). Jugez chaque permission avec TOUS les rôles effectifs du compte ensemble.",
    "egalite": "Vous comparez le motif d'une règle à la permission avec `==` : « gestes:* » ou "
               "« gestes:valider,proposer » donnent pourtant les deux. Utilisez `permission_couvre` (le correcteur vous "
               "prête la sienne).",
    "sans_heritage": "Vous prenez les rôles DIRECTS : un manager n'a alors pas le droit de proposer (il le tient de "
                     "« conseiller », par héritage). Utilisez `roles_effectifs(u[\"roles\"], heritage)`.",
    "refus_conditionnels": "Un refus AVEC conditions ne retire le droit que pour certains clients (les salariés, ceux "
                           "d'une autre agence) : pour les autres, le droit existe encore, et le cumul aussi. Seul un "
                           "refus SANS condition retire le droit.",
    "refus_ignores": "Vous ignorez les refus : un refus SANS condition (sans \"si\", ou \"si\": []) retire le droit "
                     "pour TOUS les clients, et avec lui le conflit.",
    "conditions_exigees": "Une autorisation SOUS condition (« les clients de mon agence ») donne quand même le droit, "
                          "pour ces clients-là : il compte.",
    "roles_vides_tous": "Une règle avec \"roles\": [] ne vise PERSONNE, mais votre `not regle.get(\"roles\")` la traite "
                        "comme une règle sans \"roles\" (tout le monde). Testez `\"roles\" in regle`.",
    "paires_d_abord": "Les conflits sont justes, mais pas leur ordre : les comptes dans l'ordre de `utilisateurs`, et "
                      "pour chaque compte, les couples dans l'ordre de `incompatibles` (la boucle sur les comptes à "
                      "l'extérieur).",
}


def _cas_282() -> list[tuple]:
    heritage = {"conseiller": [], "manager": ["conseiller"], "stagiaire": [], "ds": [], "chef_ds": ["ds"],
                "fourretout": [], "vide": [], "relecteur": []}
    regles = [
        {"id": "P", "effet": AUTORISER, "permission": "gestes:proposer", "roles": ["conseiller"], "si": ["meme_agence"]},
        {"id": "V", "effet": AUTORISER, "permission": "gestes:valider", "roles": ["manager"]},
        {"id": "T", "effet": AUTORISER, "permission": "gestes:*", "roles": ["stagiaire"]},
        {"id": "N", "effet": REFUSER, "permission": "gestes:valider", "roles": ["stagiaire"], "si": []},
        {"id": "E", "effet": AUTORISER, "permission": "modeles:entrainer", "roles": ["ds"]},
        {"id": "D", "effet": AUTORISER, "permission": "modeles:deployer", "roles": ["chef_ds"]},
        {"id": "X", "effet": REFUSER, "permission": "modeles:deployer", "roles": ["chef_ds"], "si": ["client_salarie"]},
        {"id": "F", "effet": AUTORISER, "permission": "gestes:valider,proposer", "roles": ["fourretout"]},
        {"id": "Z", "effet": AUTORISER, "permission": "modeles", "roles": []},
        {"id": "R", "effet": AUTORISER, "permission": "gestes:valider", "roles": ["relecteur"], "si": ["meme_agence"]},
        {"id": "Q", "effet": REFUSER, "permission": "gestes:valider", "si": ["autre_agence"]},
    ]
    identifiants = [("u1", ["manager"]), ("u2", ["conseiller"]), ("u3", ["stagiaire"]), ("u4", ["chef_ds"]),
                    ("u5", ["ds"]), ("u6", ["fourretout"]), ("u7", ("conseiller", "ds")), ("u8", ["vide"]),
                    ("u9", ["conseiller", "relecteur"]), ("u10", ["conseiller", "chef_ds"]), ("u11", [])]
    utilisateurs = [{"identifiant": i, "roles": r, "agence": "Lille", "region": "Nord"} for i, r in identifiants]
    paires = INCOMPATIBLES + [["gestes:proposer", "modeles:deployer"]]
    exemple = utilisateurs_exemple()
    corrige = {**heritage_exemple(), "support_informatique": []}
    return [(utilisateurs, heritage, regles, paires),
            (utilisateurs[::-1], heritage, regles[::-1], paires[::-1]),
            (utilisateurs, heritage, regles, []),
            (exemple, heritage_exemple(), regles_exemple(), INCOMPATIBLES),
            (exemple, corrige, regles_exemple(), INCOMPATIBLES)]


def _decrire_cas_282(cas) -> str:
    utilisateurs, heritage, regles, paires = cas
    if len(utilisateurs) > 100:
        nom = "heritage_exemple()" if heritage == heritage_exemple() else "{**heritage_exemple(), " \
                                                                          "'support_informatique': []}"
        return f"conflits(utilisateurs_exemple(), {nom}, regles_exemple(), INCOMPATIBLES)"
    return (f"conflits(utilisateurs, heritage, regles, {paires!r}), avec utilisateurs = {_court(utilisateurs, 260)}, "
            f"heritage = {heritage!r} et regles = {_court(regles, 500)} (le cas complet : `_cas_282()` dans "
            f"defis/checks_serie_46.py)")


def verifier_conflits(f) -> None:
    with _Pretes(f, {"roles_effectifs": _ref_roles, "permission_couvre": _ref_couvre}):
        _verifier_conflits(f)


def _verifier_conflits(f) -> None:
    cas = _cas_282()
    obtenus = []
    for x in cas:
        quoi = _decrire_cas_282(x)
        avant = copy.deepcopy(x)
        r = _appel(f, x, quoi)
        _sans_modification(avant, x, "les comptes, l'héritage, les règles ou les incompatibilités")
        if r is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not isinstance(r, list):
            raise AssertionError(f"{quoi} renvoie {_court(r, 80)} ({type(r).__name__}) au lieu d'une liste de tuples "
                                 "(identifiant, permission 1, permission 2), dans l'ordre.")
        for element in r:
            if not isinstance(element, tuple) or len(element) != 3:
                raise AssertionError(f"{quoi} contient {_court(element, 80)} : chaque conflit est un TUPLE (identifiant, "
                                     "permission 1, permission 2).")
        obtenus.append(r)
    for x, r in zip(cas, obtenus):
        attendu = _ref_conflits(*x)
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_conflits(*y, bug=bug) for y in cas], _BUGS_CONFLITS,
                                   lambda a, b: a == b)
            manquants = [c for c in attendu if c not in r][:3]
            en_trop = [c for c in r if c not in attendu][:3]
            detail = (f"il manque {manquants}" if manquants else "") + (" ; " if manquants and en_trop else "") + \
                     (f"en trop : {en_trop}" if en_trop else "")
            raise AssertionError(f"{_decrire_cas_282(x)} est faux ({detail or 'mauvais ordre'}). {conseil}".rstrip())

    utilisateurs, heritage, regles, _ = cas[0]
    for paires, attendu, conseil in (
            (["gestes:proposer"], TypeError, "Chaque incompatibilité est un COUPLE (un tuple), pas un texte."),
            ([("gestes:proposer", "gestes:valider", "modeles:deployer")], ValueError,
             "Une incompatibilité relie exactement DEUX permissions."),
            ([("gestes:proposer",)], ValueError, "Une incompatibilité relie exactement DEUX permissions."),
            ([None], TypeError, "None n'est pas un couple.")):
        _doit_lever(f, (utilisateurs, heritage, regles, paires), f"conflits(..., {paires!r})", conseil, attendu)


# ============================================================================ défi 283 : Delegations


class _RefDelegations:
    """La référence ; `bug` rejoue une erreur classique."""

    def __init__(self, annuaire, heritage, regles, bug: str = "") -> None:
        if not isinstance(annuaire, dict) or not isinstance(heritage, dict):
            raise TypeError("annuaire et heritage sont des dict")
        if not isinstance(regles, (list, tuple)):
            raise TypeError("regles est une liste")
        if bug == "annuaire_copie":
            annuaire = copy.deepcopy(annuaire)
        self._annuaire, self._heritage, self._regles, self._bug = annuaire, heritage, regles, bug
        self._delegations: dict[int, tuple] = {}
        self._revoquees: set[int] = set()

    def ajouter(self, delegant, delegue, motif, debut, fin) -> int:
        for identifiant in (delegant, delegue):
            if not isinstance(identifiant, str):
                raise TypeError("un identifiant est un texte")
            if identifiant not in self._annuaire:
                raise ValueError(f"compte inconnu : {identifiant!r}")
        if delegant == delegue:
            raise ValueError("on ne se délègue rien à soi-même")
        _ref_couvre(motif, "x")
        _instant(debut, "debut")
        _instant(fin, "fin")
        if not debut < fin:
            raise ValueError("debut doit précéder fin")
        numero = len(self._delegations) + 1
        photo = copy.deepcopy(self._compte(delegant)) if self._bug == "photo" else None
        self._delegations[numero] = (delegant, delegue, motif, debut, fin, photo)
        return numero

    def revoquer(self, numero) -> None:
        if isinstance(numero, bool) or not isinstance(numero, int):
            raise TypeError("un numéro est un entier")
        if numero not in self._delegations or numero in self._revoquees:
            raise ValueError(f"pas de délégation active n°{numero}")
        self._revoquees.add(numero)

    def _compte(self, identifiant) -> dict:
        compte = self._annuaire[identifiant]
        return {**compte, "roles": _ref_roles(compte["roles"], self._heritage)}

    def decider(self, identifiant, permission, client, maintenant, _deja: frozenset = frozenset()) -> Decision:
        bug = self._bug
        if not isinstance(identifiant, str):
            raise TypeError("un identifiant est un texte")
        if identifiant not in self._annuaire:
            raise ValueError(f"compte inconnu : {identifiant!r}")
        _instant(maintenant, "maintenant")
        propre = _ref_decider(self._compte(identifiant), permission, client, self._regles)
        if propre.autorise or (propre.regle is not None and bug != "refus_contourne"):
            return propre
        ordre = sorted(self._delegations.items(), reverse=bug == "derniere")
        for numero, (delegant, delegue, motif, debut, fin, photo) in ordre:
            if (numero in self._revoquees and bug != "revocation_ignoree") or delegue != identifiant:
                continue
            if bug == "fin_incluse":
                en_cours = debut <= maintenant <= fin
            elif bug == "debut_ignore":
                en_cours = maintenant < fin
            else:
                en_cours = debut <= maintenant < fin
            if not en_cours or not _ref_couvre(motif, permission):
                continue
            if bug == "sans_controle":
                return Decision(True, None, numero)
            if bug == "photo":
                compte = photo
            elif delegant not in self._annuaire and bug != "delegant_parti":
                continue
            else:
                compte = self._compte(delegant)
            if bug == "attributs_du_delegue":
                compte = {**self._annuaire[identifiant], "roles": compte["roles"]}
            if bug == "chaine" and delegant not in _deja:
                pour_lui = self.decider(delegant, permission, client, maintenant, _deja | {identifiant})
            else:
                pour_lui = _ref_decider(compte, permission, client, self._regles)
            if pour_lui.autorise:
                return Decision(True, pour_lui.regle, numero)
        return propre


_BUGS_DELEGATIONS = {
    "photo": "Vous avez recopié le compte (ou les rôles) du délégant au moment d'`ajouter` : quand il est rétrogradé, "
             "ou quand il quitte l'entreprise, son délégué garde ses anciens droits. Gardez seulement son IDENTIFIANT, "
             "et relisez son compte dans l'annuaire à chaque `decider`.",
    "sans_controle": "Vous accordez le motif délégué sans demander au délégant s'il a ce droit, LUI, pour ce client : "
                     "un conseiller pourrait déléguer « * » et en faire un administrateur. Appelez `decider` pour le "
                     "délégant (ses rôles effectifs, son agence) : la délégation ne vaut que s'il est autorisé.",
    "attributs_du_delegue": "Vous jugez le délégant avec les attributs du DÉLÉGUÉ (son agence) : la délégation de la "
                            "manager de Lille ouvre alors les clients de l'agence du délégué, pas ceux de Lille. Passez à "
                            "`decider` le compte du délégant tel qu'il est dans l'annuaire, avec ses rôles effectifs.",
    "chaine": "Vous suivez aussi les délégations que le délégant a lui-même reçues : un droit emprunté se re-prête, de "
              "proche en proche. Pour le délégant, appelez le `decider` du défi 280 (ses propres règles), pas votre "
              "méthode `decider` (qui ajoute ses délégations).",
    "fin_incluse": "Une délégation « jusqu'au 17 août » s'arrête le 17 août à 0 h : la fin est EXCLUE (`debut <= "
                   "maintenant < fin`).",
    "debut_ignore": "Une délégation qui commence le 3 août ne vaut rien le 2 : vérifiez aussi le début (`debut <= "
                    "maintenant`).",
    "refus_contourne": "Un REFUS explicite du délégué (une règle REFUSER qui s'applique à lui, `regle` n'est pas None) "
                       "ne se contourne pas par une délégation : rendez sa décision propre telle quelle. Seul le refus "
                       "PAR DÉFAUT (`regle` None) laisse chercher une délégation.",
    "revocation_ignoree": "Une délégation révoquée ne vaut plus rien, tout de suite : gardez les numéros révoqués et "
                          "sautez-les.",
    "derniere": "Plusieurs délégations conviennent : prenez la PREMIÈRE dans l'ordre des numéros (la plus ancienne), "
                "pas la dernière.",
    "delegant_parti": "Un délégant qui a quitté l'annuaire ne délègue plus rien : sautez sa délégation au lieu de "
                      "planter (`if delegant not in self._annuaire: continue`).",
    "annuaire_copie": "Vous avez recopié l'annuaire dans __init__ (`copy.deepcopy`, `dict(annuaire)`, une liste…) : un "
                      "changement de rôle ou un départ ne se voit plus. Gardez l'annuaire REÇU (`self._annuaire = "
                      "annuaire`) et relisez-y les comptes à chaque `decider`.",
}

_AOUT = datetime(2026, 8, 3, tzinfo=timezone.utc)
_LILLE = {"agence": "Lille", "region": "Nord", "salarie": False, "consentement_sms": True}
_AMIENS = {"agence": "Amiens", "region": "Nord", "salarie": False, "consentement_sms": True}


def _scenarios_283() -> list[list[tuple]]:
    """Chaque scénario : une liste d'opérations, jouées sur un annuaire neuf tiré de `utilisateurs_exemple()`.

    ("ajouter", delegant, delegue, motif, debut, fin), ("revoquer", n), ("decider", id, permission, client, instant),
    ("roles", id, nouveaux rôles) et ("partir", id) : ces deux dernières modifient l'annuaire vivant.
    """
    j = timedelta(days=1)
    histoire = [
        ("ajouter", "u001", "u024", "gestes:valider", _AOUT, _AOUT + 14 * j),
        ("decider", "u024", "gestes:valider", _LILLE, _AOUT + 2 * j),
        ("decider", "u024", "gestes:valider", _LILLE, _AOUT),
        ("decider", "u024", "gestes:valider", _LILLE, _AOUT + 14 * j),
        ("decider", "u024", "gestes:valider", _LILLE, _AOUT - timedelta(seconds=1)),
        ("decider", "u024", "gestes:valider", _AMIENS, _AOUT + 2 * j),
        ("decider", "u024", "clients:exporter", _LILLE, _AOUT + 2 * j),
        ("decider", "u002", "gestes:valider", _LILLE, _AOUT + 2 * j),
        ("decider", "u001", "gestes:valider", _LILLE, _AOUT + 2 * j),
        ("roles", "u001", ["conseiller"]),
        ("decider", "u024", "gestes:valider", _LILLE, _AOUT + 3 * j),
        ("roles", "u001", ["manager_agence"]),
        ("decider", "u024", "gestes:valider", _LILLE, _AOUT + 4 * j),
        ("revoquer", 1),
        ("decider", "u024", "gestes:valider", _LILLE, _AOUT + 5 * j),
        ("revoquer", 1),
    ]
    refus = [
        ("ajouter", "u001", "u021", "clients", _AOUT, _AOUT + 7 * j),
        ("decider", "u021", "clients:exporter", _LILLE, _AOUT + j),
        ("decider", "u021", "clients:lire", {**_LILLE, "salarie": True}, _AOUT + j),
        ("decider", "u021", "clients:lire", _LILLE, _AOUT + j),
        ("ajouter", "u005", "u006", "*", _AOUT, _AOUT + 7 * j),
        ("decider", "u006", "clients:exporter", _LILLE, _AOUT + j),
        ("decider", "u006", "gestes:valider", _LILLE, _AOUT + j),
        ("decider", "u006", "clients:modifier", _LILLE, _AOUT + j),
        ("ajouter", "u001", "u001", "gestes", _AOUT, _AOUT + j),
        ("ajouter", "u001", "inconnu", "gestes", _AOUT, _AOUT + j),
        ("ajouter", "u001", "u006", "gestes", _AOUT + j, _AOUT),
        ("ajouter", "u001", "u006", "gestes::valider", _AOUT, _AOUT + j),
        ("revoquer", 7),
        ("decider", "personne", "gestes:valider", _LILLE, _AOUT),
    ]
    chaines = [
        ("ajouter", "u001", "u002", "gestes", _AOUT, _AOUT + 10 * j),
        ("ajouter", "u002", "u003", "gestes", _AOUT, _AOUT + 10 * j),
        ("decider", "u002", "gestes:valider", _LILLE, _AOUT + j),
        ("decider", "u003", "gestes:valider", _LILLE, _AOUT + j),
        ("ajouter", "u003", "u002", "gestes:valider", _AOUT, _AOUT + 10 * j),
        ("decider", "u003", "gestes:valider", _LILLE, _AOUT + j),
        ("ajouter", "u065", "u024", "gestes:valider", _AOUT, _AOUT + 10 * j),
        ("ajouter", "u001", "u024", "gestes", _AOUT, _AOUT + 10 * j),
        ("decider", "u024", "gestes:valider", _LILLE, _AOUT + j),
        ("decider", "u024", "gestes:valider", _AMIENS, _AOUT + j),
        ("partir", "u065"),
        ("decider", "u024", "gestes:valider", _LILLE, _AOUT + j),
        ("partir", "u001"),
        ("decider", "u024", "gestes:valider", _LILLE, _AOUT + j),
        ("decider", "u002", "gestes:valider", _LILLE, _AOUT + j),
    ]
    scenarios = [histoire, refus, chaines]
    par_agence = defaultdict(list)
    for u in utilisateurs_exemple():
        par_agence[u["agence"]].append(u["identifiant"])
    chefs = ["u001", "u023", "u065"]                       # les managers de Lille et d'Amiens, le directeur du Nord
    groupe = chefs + par_agence["Lille"][1:6] + par_agence["Amiens"][1:6] + par_agence["Siège"][:2]
    motifs = ["gestes:valider", "gestes", "clients:exporter", "clients", "*", "rapports:lire", "clients:lire,modifier"]
    permissions = ["gestes:valider", "gestes:proposer", "clients:exporter", "clients:lire", "clients:modifier",
                   "rapports:lire", "modeles:deployer"]
    clients = [_LILLE, _AMIENS, {**_LILLE, "salarie": True}, {"agence": "Metz", "region": "Est", "salarie": False,
                                                              "consentement_sms": False}]
    roles_possibles = [["conseiller"], ["manager_agence"], ["directeur_regional"], ["stagiaire"],
                       ["conseiller_senior"]]
    agence_de = {u["identifiant"]: u["agence"] for u in utilisateurs_exemple()}
    client_de = {"Lille": _LILLE, "Amiens": _AMIENS, "Siège": _LILLE}
    for graine in (2831, 2832, 2833):
        alea = random.Random(graine)
        ops, ajoutees = [], []
        for _ in range(240):
            tirage = alea.random()
            if tirage < 0.2:
                delegant = alea.choice(chefs) if alea.random() < 0.6 else alea.choice(groupe)
                delegue = alea.choice([g for g in groupe if g != delegant])
                debut = _AOUT + timedelta(hours=alea.randrange(0, 240))
                ajoutees.append((delegant, delegue, alea.choice(motifs), debut,
                                 debut + timedelta(hours=alea.randrange(1, 200))))
                ops.append(("ajouter", *ajoutees[-1]))
            elif tirage < 0.25 and ajoutees:
                ops.append(("revoquer", alea.randint(1, len(ajoutees))))
            elif tirage < 0.28:
                ops.append(("roles", alea.choice(groupe), alea.choice(roles_possibles)))
            elif tirage < 0.6 and ajoutees:                # le délégué d'une délégation, pendant (ou au bord de) sa durée
                delegant, delegue, motif, debut, fin = alea.choice(ajoutees)
                couvertes = [p for p in permissions if _ref_couvre(motif, p)] or permissions
                instant = alea.choice([debut, fin, debut + (fin - debut) * alea.random()])
                ops.append(("decider", delegue, alea.choice(couvertes), client_de[agence_de[delegant]], instant))
            else:
                ops.append(("decider", alea.choice(groupe), alea.choice(permissions), alea.choice(clients),
                            _AOUT + timedelta(hours=alea.randrange(-24, 300))))
        scenarios.append(ops)
    return scenarios


def _jouer(objet, annuaire: dict, operations) -> list:
    """Joue les opérations ; une sortie par opération (ce que renvoie la méthode, ou l'exception levée)."""
    sorties = []
    for op in operations:
        try:
            if op[0] == "roles":
                annuaire[op[1]]["roles"] = list(op[2])
                sorties.append(None)
            elif op[0] == "partir":
                del annuaire[op[1]]
                sorties.append(None)
            else:
                sorties.append(getattr(objet, op[0])(*copy.deepcopy(op[1:])))
        except NotImplementedError:
            raise
        except _BoucleSansFin:
            raise
        except Exception as exc:
            sorties.append(("!", type(exc).__name__, _decrire(exc)))
    return sorties


def _annuaire() -> dict:
    return {u["identifiant"]: u for u in utilisateurs_exemple()}


def _ref_sorties(operations, bug: str = "") -> list:
    annuaire = _annuaire()
    return _jouer(_RefDelegations(annuaire, heritage_exemple(), regles_exemple(), bug), annuaire, operations)


def _sorties_egales(a, b) -> bool:
    if isinstance(b, tuple) and b[:1] == ("!",):
        return isinstance(a, tuple) and a[:2] == b[:2]
    return type(a) is type(b) and a == b


def _decrire_op(op) -> str:
    if op[0] == "roles":
        return f"annuaire[{op[1]!r}][\"roles\"] = {list(op[2])!r}"
    if op[0] == "partir":
        return f"del annuaire[{op[1]!r}]"
    morceaux = []
    for x in op[1:]:
        if isinstance(x, datetime):
            morceaux.append(f"<{x.isoformat()}>")
        elif isinstance(x, dict):
            morceaux.append(f"<client, agence {x['agence']}{', salarié' if x['salarie'] else ''}>")
        else:
            morceaux.append(repr(x))
    return f"d.{op[0]}({', '.join(morceaux)})"


def _decrire_ops(operations, jusqu_a: int) -> str:
    return ("… " if jusqu_a > 5 else "") + " ; ".join(_decrire_op(op) for op in operations[max(0, jusqu_a - 5): jusqu_a + 1])


def _decrire_sortie(s, attendu: bool = False) -> str:
    if isinstance(s, tuple) and s[:1] == ("!",):
        return f"lever {s[1]}" if attendu else f"lève {s[2]}"
    return f"{'renvoyer' if attendu else 'renvoie'} {_court(s, 120)}"


def verifier_delegations(classe) -> None:
    with _Pretes(classe, {"permission_couvre": _ref_couvre, "roles_effectifs": _ref_roles, "decider": _ref_decider}):
        _verifier_delegations(classe)


def _creer(classe, annuaire):
    return _appel(classe, (annuaire, heritage_exemple(), regles_exemple()),
                  "Delegations(annuaire, heritage_exemple(), regles_exemple())")


def _verifier_delegations(classe) -> None:
    # ------------------------------------------------------------------ deux objets ne partagent rien
    annuaire = _annuaire()
    a, b = _creer(classe, annuaire), _creer(classe, annuaire)
    numero = _appel(a.ajouter, ("u001", "u024", "gestes", _AOUT, _AOUT + timedelta(days=3)),
                    "a.ajouter('u001', 'u024', 'gestes', ...)")
    if numero != 1:
        raise AssertionError(f"La première délégation d'un objet `Delegations` doit porter le numéro 1, pas {numero!r}.")
    if _appel(b.decider, ("u024", "gestes:valider", _LILLE, _AOUT + timedelta(days=1)), "b.decider(...)").autorise:
        raise AssertionError("Deux objets `Delegations` partagent leurs délégations : une délégation ajoutée à `a` vaut "
                             "aussi dans `b`. Créez le dict des délégations dans __init__ (pas comme attribut de la "
                             "classe, ni comme valeur par défaut d'un argument).")

    # ------------------------------------------------------------------ les scénarios
    scenarios = _scenarios_283()
    obtenus, annuaires = [], []
    for operations in scenarios:
        annuaire = _annuaire()
        obtenus.append(_jouer(_creer(classe, annuaire), annuaire, operations))
        annuaires.append(annuaire)
    attendus = [_ref_sorties(ops) for ops in scenarios]
    for operations, o_tout, a_tout in zip(scenarios, obtenus, attendus):
        i = next((k for k, (o, a) in enumerate(zip(o_tout, a_tout)) if not _sorties_egales(o, a)), None)
        if i is None:
            continue
        o, a = o_tout[i], a_tout[i]
        conseil = _premier_bug(obtenus, lambda bug: [_ref_sorties(ops, bug) for ops in scenarios], _BUGS_DELEGATIONS,
                               lambda x, y: len(x) == len(y) and all(
                                   len(p) == len(q) and all(_sorties_egales(s, t) for s, t in zip(p, q))
                                   for p, q in zip(x, y)))
        if not conseil and isinstance(a, Decision) and not isinstance(o, (Decision, tuple)):
            conseil = "Rendez une `Decision` (la classe est dans defis/acces.py)."
        parti_avant = any(op[0] == "partir" for op in operations[:i])
        if not conseil and isinstance(o, tuple) and o[1] == "KeyError" and isinstance(a, Decision) and parti_avant:
            conseil = _BUGS_DELEGATIONS["delegant_parti"]
        if not conseil and isinstance(a, tuple) and a[1] == "ValueError" and operations[i][0] == "decider":
            conseil = ("Un compte qui n'est pas dans l'annuaire → ValueError (vérifiez `identifiant in self._annuaire` "
                       "AVANT de lire son compte).")
        if not conseil and isinstance(o, Decision) and isinstance(a, Decision) and o.autorise == a.autorise:
            if a.regle is None and a.delegation is None and o.regle is not None:
                conseil = ("Sans délégation qui autorise, rendez la décision PROPRE du compte (ici, le refus par "
                           "défaut : `regle` None) : le refus du délégant ne remonte pas.")
            else:
                conseil = ("La décision est juste, mais pas ses détails : `regle` est la règle qui autorise le "
                           "DÉLÉGANT, et `delegation` le numéro de la délégation utilisée (None sans délégation).")
        raise AssertionError(f"Avec d = Delegations(annuaire, heritage_exemple(), regles_exemple()) et l'annuaire de "
                             f"l'exemple : {_decrire_ops(operations, i)} : la dernière opération "
                             f"{_decrire_sortie(o)} au lieu de {_decrire_sortie(a, True)}. {conseil}".rstrip())

    # ------------------------------------------------------------------ l'annuaire n'est pas modifié
    for operations, annuaire in zip(scenarios, annuaires):
        attendu = _annuaire()
        _jouer(_RefDelegations(attendu, heritage_exemple(), regles_exemple()), attendu, operations)
        if annuaire != attendu:
            cle = next(k for k in sorted(set(annuaire) | set(attendu)) if annuaire.get(k) != attendu.get(k))
            raise AssertionError(f"Votre classe a MODIFIÉ l'annuaire : le compte {cle!r} vaut {_court(annuaire.get(cle))} "
                                 f"au lieu de {_court(attendu.get(cle))}. Pour passer les rôles effectifs à `decider`, "
                                 "faites une COPIE du compte (`{**compte, \"roles\": roles_effectifs(...)}`), sans "
                                 "toucher à celui de l'annuaire.")

    # ------------------------------------------------------------------ erreurs
    for args, attendu, conseil in (
            (([], heritage_exemple(), regles_exemple()), TypeError, "L'annuaire est un dict {identifiant: compte}."),):
        _doit_lever(classe, args, "Delegations([], heritage_exemple(), regles_exemple())", conseil, attendu)
    d = _creer(classe, _annuaire())
    naif = _AOUT.replace(tzinfo=None)
    for methode, args, attendu, conseil in (
            ("ajouter", ("u001", "u024", "gestes", naif, _AOUT + timedelta(days=1)), ValueError,
             "`debut` n'a pas de fuseau horaire : ValueError."),
            ("ajouter", ("u001", "u024", "gestes", _AOUT, naif + timedelta(days=1)), ValueError,
             "`fin` n'a pas de fuseau horaire : ValueError."),
            ("ajouter", ("u001", "u024", "gestes", "2026-08-03", _AOUT), TypeError, "Un texte n'est pas une date."),
            ("ajouter", ("u001", "u024", "gestes", _AOUT, _AOUT), ValueError,
             "Une délégation qui finit quand elle commence ne vaut rien : `debut < fin`, ValueError."),
            ("ajouter", ("u001", "u024", "", _AOUT, _AOUT + timedelta(days=1)), ValueError,
             "Un motif vide est mal écrit (défi 278)."),
            ("ajouter", ("u001", "u024", None, _AOUT, _AOUT + timedelta(days=1)), TypeError,
             "None n'est pas un motif (défi 278)."),
            ("ajouter", ("u001", 24, "gestes", _AOUT, _AOUT + timedelta(days=1)), TypeError,
             "Un identifiant est un texte."),
            ("ajouter", ("u001", "u999", "gestes", _AOUT, _AOUT + timedelta(days=1)), ValueError,
             "u999 n'est pas dans l'annuaire."),
            ("ajouter", ("u024", "u024", "gestes", _AOUT, _AOUT + timedelta(days=1)), ValueError,
             "On ne se délègue rien à soi-même."),
            ("revoquer", (1,), ValueError, "Il n'y a pas encore de délégation n°1."),
            ("revoquer", (True,), TypeError, "True n'est pas un numéro."),
            ("revoquer", ("1",), TypeError, "Un numéro est un entier."),
            ("decider", ("u999", "gestes:valider", _LILLE, _AOUT), ValueError, "u999 n'est pas dans l'annuaire."),
            ("decider", ("u024", "gestes:valider", _LILLE, naif), ValueError,
             "`maintenant` n'a pas de fuseau horaire : ValueError."),
            ("decider", ("u024", "gestes:valider", _LILLE, "2026-08-03"), TypeError, "Un texte n'est pas une date.")):
        _doit_lever(getattr(d, methode), args, f"d.{methode}{_court(args, 90)}", conseil, attendu)
    numero = _appel(d.ajouter, ("u001", "u024", "gestes", _AOUT, _AOUT + timedelta(days=1)), "d.ajouter(...)")
    _appel(d.revoquer, (numero,), f"d.revoquer({numero})")
    _doit_lever(d.revoquer, (numero,), f"d.revoquer({numero}) une deuxième fois",
                "Une délégation déjà révoquée ne se révoque pas deux fois : ValueError.")


# ---------------------------------------------------------------------------- la série

SERIE = Serie(
    code="46",
    titre="Qui a le droit de voir quoi : permissions, rôles, décisions, droits dormants, cumuls, délégations",
    module_eleve="defis.serie_46",
    module_corrige="defis.corrige_serie_46",
    defis=(
        Defi(278, "permission_couvre", "🟢", "—", verifier_permission_couvre),
        Defi(279, "roles_effectifs", "🟢", "—", verifier_roles_effectifs),
        Defi(280, "decider", "🟡", "—", verifier_decider),
        Defi(281, "droits_dormants", "🟡", "—", verifier_droits_dormants),
        Defi(282, "conflits", "🟡", "—", verifier_conflits),
        Defi(283, "Delegations", "🔴", "—", verifier_delegations),
    ),
)

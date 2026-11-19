"""Vérifications automatiques de la série 39 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est le script de déploiement, qui ne pardonne aucune approximation, et l'équipe d'astreinte,
qui veut pouvoir tout couper en une seconde :

- partout, il choisit des cas où chaque erreur classique se voit : 2.10 contre 2.9, une pré-version contre sa
  version, beta.11 contre beta.2, un « BREAKING CHANGE » caché dans le pied d'un message (et un autre écrit dans
  la première ligne, qui ne compte pas), `^0.2.3` contre 0.3.0, une rc qu'on n'a pas demandée, un paquet qui
  oblige à revenir en arrière, un conseiller dont le seau tombe PILE sur une frontière, un rôle absent ;
- quand votre résultat est faux, il rejoue les erreurs classiques (comparaison de textes, pré-version après sa
  version, identifiants comparés en texte, versions cumulées, correctif pas remis à zéro, `startswith("fix")`,
  `^` sans le cas 0.x, pré-versions toujours acceptées, ET et OU inversés, pas de retour arrière, `hash()`,
  nom de l'expérience oublié dans le hachage, `<=` à la frontière, ciblage avant le bouton d'arrêt, attribut
  absent qui remplit « pas_dans », première règle comptée 0…) pour nommer la vôtre ;
- aux défis 237, 238, 239 et 241, si votre résultat est faux parce qu'un défi précédent (236, 238 ou 240) ne
  passe pas encore, il vous le dit ;
- au défi 239, il compte les lignes exécutées dans votre fichier sur 60 paquets de 12 versions : essayer les
  combinaisons sans écarter tout de suite une version qui contredit un choix déjà fait se voit ;
- et il vérifie que vos fonctions ne modifient pas ce qu'elles reçoivent.
"""

from __future__ import annotations

import copy
import hashlib
import random
import re
import sys
from functools import cmp_to_key

from .livraison import (OPERATEURS, PALIERS, Conflit, Decision, conseillers_exemple, drapeaux_exemple,
                        generer_conseillers, generer_versions, historique_exemple, index_exemple, tags_exemple)
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
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError : une fonction s'appelle elle-même sans fin. "
                             f"{conseil}".rstrip()) from None
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


def _pas_none(resultat, quoi: str) -> None:
    if resultat is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")


def _intact(avant, apres, quoi: str, nom: str) -> None:
    if avant != apres:
        raise AssertionError(f"{quoi} a modifié {nom} reçu(s). Une fonction ne touche pas à ce qu'on lui prête : "
                             "lisez les valeurs, construisez de NOUVEAUX objets.")


def _fonction_eleve(f, nom: str):
    """La fonction `nom` du fichier de l'élève (celui où f est écrite), ou None."""
    fonction = getattr(f, "__globals__", {}).get(nom)
    return fonction if callable(fonction) else None


def _premier_bug(obtenus: list, rejouer, bugs: dict) -> str:
    """Le conseil du premier bug dont le rejeu donne EXACTEMENT vos réponses sur toute la batterie."""
    for bug, conseil in bugs.items():
        try:
            faux = rejouer(bug)
        except Exception:
            continue
        if faux == obtenus:
            return conseil
    return ""


# ============================================================================ défi 236 : comparer_versions

_IDENT = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
                     rf"(?:-({_IDENT}(?:\.{_IDENT})*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?")


def _lire(version: str) -> tuple:
    """(majeur, mineur, correctif, pré-version en texte ou None, build ou None) ; ValueError si invalide."""
    if not isinstance(version, str):
        raise TypeError("pas un texte")
    t = _SEMVER.fullmatch(version)
    if t is None:
        raise ValueError(f"version invalide : {version!r}")
    return int(t[1]), int(t[2]), int(t[3]), t[4], t[5]


def _cmp(x, y) -> int:
    return (x > y) - (x < y)


def _ref_comparer(a: str, b: str, bug: str = "") -> int:
    """La bonne réponse ; `bug` rejoue une erreur classique."""
    if bug == "texte":
        return _cmp(a, b)
    ma, na, pa, pre_a, build_a = _lire(a)
    mb, nb, pb, pre_b, build_b = _lire(b)
    c = _cmp((ma, na, pa), (mb, nb, pb))
    if c or bug == "sans_pre":
        return c
    if pre_a is None or pre_b is None:
        if pre_a is None and pre_b is None:
            return _cmp(build_a or "", build_b or "") if bug == "build" else 0
        c = 1 if pre_a is None else -1
        return -c if bug == "pre_apres" else c
    if bug == "pre_texte":
        return _cmp(pre_a, pre_b)
    ids_a, ids_b = pre_a.split("."), pre_b.split(".")
    for x, y in zip(ids_a, ids_b):
        num_x, num_y = x.isdigit(), y.isdigit()
        if bug == "ids_texte":
            c = _cmp(x, y)
        elif num_x and num_y:
            c = _cmp(int(x), int(y))
        elif num_x != num_y:
            c = -1 if num_x else 1
            if bug == "numerique_apres":
                c = -c
        else:
            c = _cmp(x, y)
        if c:
            return c
    c = _cmp(len(ids_a), len(ids_b))
    if c:
        return -c if bug == "court_apres" else c
    return _cmp(build_a or "", build_b or "") if bug == "build" else 0


_BUGS_COMPARER = {
    "texte": "Vous comparez les versions comme des TEXTES : \"2.10.1\" < \"2.9.4\" parce que \"1\" < \"9\" (l'erreur du "
             "3 novembre). Découpez en nombres : (2, 10, 1) > (2, 9, 4).",
    "sans_pre": "Vous ignorez la pré-version : 1.0.0-rc.1 et 1.0.0 n'ont pas le même rang. La pré-version passe AVANT "
                "la version qu'elle prépare.",
    "pre_apres": "Pour vous, une pré-version passe APRÈS sa version (un tuple plus long est plus grand ?) : c'est "
                 "l'inverse, 1.0.0-rc.1 < 1.0.0. Sans pré-version, une version passe après toutes ses pré-versions.",
    "pre_texte": "Vous comparez la pré-version comme UN texte : \"beta.11\" < \"beta.2\" en texte. Découpez-la aux "
                 "points et comparez identifiant par identifiant : 11 et 2 sont des nombres.",
    "ids_texte": "Vous comparez les identifiants de pré-version en texte, même quand ils sont numériques : 11 > 2, "
                 "mais \"11\" < \"2\". Deux identifiants faits de chiffres se comparent en NOMBRES.",
    "numerique_apres": "Pour vous, un identifiant numérique passe après un identifiant alphanumérique : c'est "
                       "l'inverse (1.0.0-9 < 1.0.0-alpha), règle 11.4.3 de SemVer.",
    "court_apres": "Quand tous les identifiants communs sont égaux, la pré-version la plus COURTE passe avant : "
                   "1.0.0-alpha < 1.0.0-alpha.1.",
    "build": "Les métadonnées de build (après « + ») ne comptent pas dans l'ordre : 1.0.0+build.1 et 1.0.0+build.2 "
             "ont le même rang (0).",
}

_CHAINE_SEMVER = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11",
                  "1.0.0-rc.1", "1.0.0"]

_PAIRES_236 = [
    ("2.10.1", "2.9.4"), ("1.0.0-rc.1", "1.0.0"), ("1.0.0", "1.0.0-rc.1"), ("1.0.0-beta.11", "1.0.0-beta.2"),
    ("1.0.0-alpha", "1.0.0-alpha.1"), ("1.0.0-alpha.1", "1.0.0-alpha"), ("1.0.0-9", "1.0.0-alpha"),
    ("1.0.0-alpha", "1.0.0-9"), ("1.0.0+build.1", "1.0.0+build.2"), ("1.0.0-rc.1+b", "1.0.0-rc.1+a"),
    ("1.0.0-10", "1.0.0-9"), ("1.0.0-a.b", "1.0.0-a-b"), ("1.0.0-Z", "1.0.0-a"), ("10.0.0", "9.99.99"),
    ("0.0.1", "0.0.0"), ("1.2.3", "1.2.3"), ("1.2.3-x.7.z.92", "1.2.3-x.7.z.92"), ("1.0.0-0", "1.0.0-00a"),
    ("2.0.0", "11.0.0"), ("1.11.0", "1.2.0"), ("1.0.0-rc.1", "1.0.0-rc.1.0"), ("3.0.0-rc.1", "2.99.99"),
]


def _paires_alea(graine: int) -> list[tuple[str, str]]:
    versions = generer_versions(36, graine=graine)
    return [(a, b) for a in versions for b in versions]


def _batterie_236() -> list[tuple[str, str]]:
    paires = list(_PAIRES_236)
    paires += [(a, b) for a in _CHAINE_SEMVER for b in _CHAINE_SEMVER]
    tags = tags_exemple()
    paires += [(a, b) for a in tags for b in tags]
    for graine in (236, 2360):
        paires += _paires_alea(graine)
    return paires


def _valeur_236(resultat, quoi: str) -> int:
    if type(resultat) is not int or resultat not in (-1, 0, 1):
        raise AssertionError(f"{quoi} renvoie {_court(resultat)} ({type(resultat).__name__}) ; attendu -1, 0 ou 1 (un "
                             "int, pas un bool). Avec deux clés : `(cle_a > cle_b) - (cle_a < cle_b)`.")
    return resultat


def _comparer_juste(f) -> str:
    """'' si votre comparer_versions donne les bonnes réponses sur la batterie, sinon le premier écart."""
    for a, b in _PAIRES_236 + [(x, y) for x in _CHAINE_SEMVER for y in _CHAINE_SEMVER]:
        try:
            obtenu = f(a, b)
        except Exception as exc:
            return f"comparer_versions({a!r}, {b!r}) lève {_decrire(exc)}"
        if obtenu != _ref_comparer(a, b):
            return f"comparer_versions({a!r}, {b!r}) renvoie {obtenu!r} au lieu de {_ref_comparer(a, b)}"
    return ""


_INVALIDES_236 = [
    ("1.2", "il faut trois nombres : MAJEUR.MINEUR.CORRECTIF."),
    ("1.2.3.4", "il faut exactement trois nombres."),
    ("v1.2.3", "« v » n'est pas une version (c'est un préfixe de tag Git : retirez-le AVANT d'appeler)."),
    ("01.2.3", "pas de zéro inutile en tête (« 01 »)."),
    ("1.02.3", "pas de zéro inutile en tête (« 02 »)."),
    ("1.2.3-01", "un identifiant de pré-version NUMÉRIQUE n'a pas de zéro inutile en tête (« 01 »)."),
    ("1.2.3-", "une pré-version n'est pas vide."),
    ("1.2.3-a..b", "un identifiant de pré-version n'est pas vide (« a..b »)."),
    ("1.2.3+", "des métadonnées de build ne sont pas vides."),
    ("1.2.3+a..b", "un identifiant de build n'est pas vide."),
    ("1.2.3\n", "un « \\n » final n'appartient pas à la version : avec `re.match` et « $ », il passe (« $ » accepte "
                "une fin de ligne). Utilisez `re.fullmatch`, ou « \\Z »."),
    ("١.٢.٣", "« ١.٢.٣ » s'écrit avec des chiffres arabes : `\\d` et `isdigit()` les acceptent (et `int(\"١\")` vaut 1). "
              "SemVer ne veut que les chiffres ASCII 0 à 9 : écrivez [0-9]."),
    ("1.2.3-²", "« ² » n'est pas un identifiant valable (`\"²\".isdigit()` est True !). Seuls [0-9A-Za-z-] sont permis."),
    ("1.2.3-bêta", "« ê » n'est pas permis dans une pré-version : seuls [0-9A-Za-z-]."),
    ("1.2.3-beta_1", "« _ » n'est pas permis dans une pré-version : seuls [0-9A-Za-z-]."),
    (" 1.2.3", "une espace n'appartient pas à la version."),
    ("1.2.3 ", "une espace n'appartient pas à la version."),
    ("-1.2.3", "un nombre de version n'est pas négatif."),
    ("1.2.3+a+b", "un seul « + »."),
    ("", "une version vide n'est pas une version."),
    ("1..3", "il faut trois nombres."),
]

_VALIDES_236 = ["0.0.0", "1.2.3-01a", "1.2.3-0", "1.2.3+001", "1.2.3--", "1.2.3-x-y.--z", "10.20.30",
                "1.0.0-alpha+001", "1.0.0+21AF26D3----117B344092BD", "99999999999999999999.0.0"]


def verifier_comparer_versions(f) -> None:
    paires = _batterie_236()
    obtenus = []
    for a, b in paires:
        quoi = f"comparer_versions({a!r}, {b!r})"
        obtenus.append(_valeur_236(_appel(f, (a, b), quoi), quoi))
    attendus = [_ref_comparer(a, b) for a, b in paires]
    if obtenus != attendus:
        conseil = _premier_bug(obtenus, lambda bug: [_ref_comparer(a, b, bug) for a, b in paires], _BUGS_COMPARER)
        i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        a, b = paires[i]
        raise AssertionError(f"comparer_versions({a!r}, {b!r}) renvoie {obtenus[i]} au lieu de {attendus[i]}. "
                             f"{conseil}".rstrip())
    ordre = sorted(reversed(_CHAINE_SEMVER), key=cmp_to_key(f))
    if ordre != _CHAINE_SEMVER:
        raise AssertionError(f"Triées avec votre fonction, les versions d'exemple de semver.org donnent {ordre} au lieu "
                             f"de {_CHAINE_SEMVER}.")
    for version in _VALIDES_236:
        quoi = f"comparer_versions({version!r}, {version!r})"
        if _valeur_236(_appel(f, (version, version), quoi, f"{version!r} est une version SemVer VALABLE."), quoi) != 0:
            raise AssertionError(f"{quoi} devrait renvoyer 0 : une version a le même rang qu'elle-même.")
    for version, conseil in _INVALIDES_236:
        _doit_lever(f, (version, "1.0.0"), f"comparer_versions({version!r}, '1.0.0')", conseil)
        _doit_lever(f, ("1.0.0", version), f"comparer_versions('1.0.0', {version!r})", conseil)
    for valeur in (None, 1.2, b"1.2.3", (1, 2, 3)):
        _doit_lever(f, (valeur, "1.0.0"), f"comparer_versions({valeur!r}, '1.0.0')", "une version est un texte.",
                    TypeError)


# ============================================================================ défi 237 : prochaine_version

_ENTETE = re.compile(r"([A-Za-z]+)(?:\(([^()\r\n]+)\))?(!)?: (\S.*)")
_ENTETE_SOUPLE = re.compile(r"\s*([A-Za-z]+)(?:\(([^()\r\n]*)\))?(!)?\s*:\s*(.*)")


def _niveau(message: str, bug: str = "") -> int:
    if bug == "prefixe":
        bas = message.lower()
        if bas.startswith("feat"):
            niveau = 2
        elif bas.startswith(("fix", "perf")):
            niveau = 1
        else:
            niveau = 0
        return 3 if "BREAKING CHANGE: " in message.split("\n", 1)[-1] and "\n" in message else niveau
    lignes = [ligne.rstrip("\r") for ligne in message.split("\n")]
    entete = (_ENTETE_SOUPLE if bug == "espace" else _ENTETE).fullmatch(lignes[0])
    if entete is None:
        return 0
    corps = lignes[1:]
    pieds = ("BREAKING CHANGE: ", "BREAKING-CHANGE: ")
    if bug == "partout":
        rupture_pied = "BREAKING CHANGE" in message
    elif bug == "minuscule":
        rupture_pied = any(ligne.upper().startswith(pieds) for ligne in corps)
    else:
        rupture_pied = bug != "pied" and any(ligne.startswith(pieds) for ligne in corps)
    rupture = (entete[3] == "!" and bug != "exclamation") or rupture_pied
    if rupture:
        return 3
    type_ = entete[1] if bug == "casse" else entete[1].lower()
    return {"feat": 2, "fix": 1, "perf": 1}.get(type_, 0)


def _ref_prochaine(actuelle: str, messages, bug: str = "") -> str | None:
    """La bonne réponse ; `bug` rejoue une erreur classique."""
    majeur, mineur, correctif, pre, build = _lire(actuelle)
    niveaux = [_niveau(m, bug) for m in messages]
    niveau = max(niveaux, default=0)
    if niveau == 0:
        return actuelle if bug == "meme" else None
    if niveau == 3 and majeur == 0 and bug != "zero":
        niveau = 2
    if bug == "pre":
        pre = None
    suffixe = f"+{build}" if bug == "build" and build else ""
    if bug == "cumul":
        for n in niveaux:
            if n == 3:
                majeur, mineur, correctif = majeur + 1, 0, 0
            elif n == 2:
                mineur, correctif = mineur + 1, 0
            elif n == 1:
                correctif += 1
        return f"{majeur}.{mineur}.{correctif}"
    if niveau == 3:
        if pre and mineur == 0 and correctif == 0:
            return f"{majeur}.0.0{suffixe}"
        return f"{majeur + 1}.{mineur if bug == 'sans_remise' else 0}.{correctif if bug == 'sans_remise' else 0}{suffixe}"
    if niveau == 2:
        if pre and correctif == 0:
            return f"{majeur}.{mineur}.0{suffixe}"
        return f"{majeur}.{mineur + 1}.{correctif if bug == 'sans_remise' else 0}{suffixe}"
    return f"{majeur}.{mineur}.{correctif if pre else correctif + 1}{suffixe}"


_BUGS_PROCHAINE = {
    "cumul": "Vous augmentez le numéro pour CHAQUE commit : un `feat` ou dix donnent UNE version mineure (1.4.2 → "
             "1.5.0), et les `fix` qui les accompagnent n'ajoutent rien. Prenez le niveau le plus fort, puis calculez "
             "UNE fois.",
    "sans_remise": "Les numéros à droite de celui qu'on augmente repartent de 0 : 1.4.2 + feat → 1.5.0 (pas 1.5.2), "
                   "1.4.2 + rupture → 2.0.0.",
    "pied": "Vous ne lisez pas le CORPS du message : une ligne « BREAKING CHANGE: … » annonce une rupture (version "
            "majeure), même dans un `fix` ou un `refactor`. C'est le piège de la 2.11.0 de Théo.",
    "exclamation": "Un « ! » juste avant les deux-points (« feat!: », « refactor(api)!: ») annonce une rupture : "
                   "version majeure.",
    "partout": "Vous cherchez « BREAKING CHANGE » n'importe où dans le message. Il ne compte qu'au DÉBUT d'une ligne "
               "du corps (pas dans la première ligne, pas au milieu d'une phrase), suivi de « : ».",
    "minuscule": "« breaking change: » en minuscules ne compte pas : le pied s'écrit « BREAKING CHANGE: » (ou "
                 "« BREAKING-CHANGE: »), en majuscules.",
    "casse": "Le type ne dépend pas de la casse : « Feat: » est un feat, « FIX(api): » un fix (Conventional Commits, "
             "règle 15). Comparez `type.lower()`.",
    "prefixe": "Vous testez le début du message (`startswith(\"fix\")`…) : « fixup! fix: … », « feature: … » ou "
               "« fixes #12 » ne sont pas des commits conventionnels. Vérifiez TOUTE la première ligne : "
               "`re.fullmatch(r\"([A-Za-z]+)(?:\\(([^()\\r\\n]+)\\))?(!)?: (\\S.*)\", ligne)`.",
    "espace": "Votre lecture de la première ligne est trop tolérante : « fix : x », « feat:x » ou « feat(): x » ne "
              "sont pas conventionnels (deux-points collés au type ou à la portée, UNE espace après, une portée non "
              "vide).",
    "zero": "En 0.y.z, une rupture ne fait monter que le MINEUR (0.4.1 → 0.5.0) : avant 1.0.0, tout peut changer, "
            "et passer en 1.0.0 est une décision, pas un effet de bord.",
    "pre": "Après une pré-version, on publie la version qu'elle préparait : 3.0.0-rc.2 + fix → 3.0.0 (pas 3.0.1), "
           "3.1.0-rc.1 + feat → 3.1.0. Relisez les règles de calcul.",
    "meme": "Quand aucun commit ne publie rien, renvoyez None : renvoyer la version actuelle ferait republier (ou "
            "re-étiqueter) la même version.",
    "build": "Les métadonnées de build (« +build.7 ») de la version actuelle ne passent pas dans la suivante.",
}

_CAS_237 = [
    ("1.4.2", ["feat: export CSV", "feat(api): pagination", "fix: arrondi"]),
    ("1.4.2", ["fix: arrondi", "refactor: tout\n\nBREAKING CHANGE: le champ score disparaît"]),
    ("1.4.2", ["docs: README", "Merge branch 'main'", "fixup! fix: arrondi"]),
    ("1.4.2", ["feat: a", "feat: b", "feat: c"]),
    ("1.4.2", ["fix: a", "perf: b"]),
    ("2.3.4", ["feat: x"]),
    ("2.3.4", ["fix: x\n\nBREAKING CHANGE: y"]),
    ("2.3.4", ["fix: x\n\nBREAKING-CHANGE: y"]),
    ("2.3.4", ["refactor!: x"]),
    ("2.3.4", ["feat(api)!: x"]),
    ("2.3.4", ["docs: expliquer BREAKING CHANGE: rien ne casse"]),
    ("2.3.4", ["fix: x\n\nvoir la note BREAKING CHANGE: de la 2.0"]),
    ("2.3.4", ["fix: x\n\nbreaking change: y"]),
    ("2.3.4", ["Feat: x"]),
    ("2.3.4", ["FIX(api): x"]),
    ("2.3.4", ["fixup! fix: x", "feature: y", "fixes #12"]),
    ("2.3.4", ["fix : x", "feat:y", "feat(): z", " feat: w"]),
    ("0.4.1", ["feat!: x"]),
    ("0.4.1", ["fix: x\n\nBREAKING CHANGE: y"]),
    ("3.0.0-rc.2", ["fix: x"]),
    ("3.0.0-rc.2", ["feat: x"]),
    ("3.1.0-rc.1", ["feat: x"]),
    ("3.1.2-rc.1", ["feat: x"]),
    ("3.1.0-beta", ["feat!: x"]),
    ("3.0.0-rc.2", ["feat!: x"]),
    ("2.3.4", ["chore: a", "docs: b"]),
    ("2.3.4", []),
    ("2.3.4+build.7", ["fix: x"]),
    ("2.3.4+build.7", ["feat!: x"]),
    ("2.3.4", ["fix: x\r\n\r\nBREAKING CHANGE: y"]),
    ("2.3.4", ["feat: x\n\nLe corps.\nBREAKING CHANGE: tout"]),
    ("9.9.9", ["fix: x", "feat: y"]),
    ("2.10.1", historique_exemple()),
]

_ENTETES_ALEA = ["feat: a", "fix: b", "perf: c", "docs: d", "chore(ci): e", "Feat(api): f", "FIX: g", "refactor!: h",
                 "feat(x)!: i", "fixup! fix: j", "feature: k", "fix : l", "feat:m", "WIP", "Merge branch 'x'",
                 "test: n", "feat(): o", "style: p", "fixes #3"]
_CORPS_ALEA = ["", "\n\nUn corps.", "\n\nBREAKING CHANGE: q", "\n\nbreaking change: r", "\n\nBREAKING-CHANGE: s",
               "\n\nvoir BREAKING CHANGE: t", "\r\n\r\nBREAKING CHANGE: u", "\n\nRefs: #4"]


def _cas_237_alea(graine: int) -> list[tuple[str, list[str]]]:
    rng = random.Random(graine)
    cas = []
    for _ in range(120):
        actuelle = rng.choice(["0.1.0", "0.4.1", "1.0.0", "1.4.2", "2.10.1", "3.0.0-rc.2", "3.1.0-beta.1",
                               "3.1.4-rc.1", "0.3.0-alpha", "2.3.4+build.7", "10.0.0"])
        messages = [rng.choice(_ENTETES_ALEA) + rng.choice(_CORPS_ALEA) for _ in range(rng.randint(0, 5))]
        cas.append((actuelle, messages))
    return cas


def verifier_prochaine_version(f) -> None:
    cas = _CAS_237 + _cas_237_alea(237)
    obtenus = []
    for actuelle, messages in cas:
        quoi = f"prochaine_version({actuelle!r}, {_court(messages, 110)})"
        avant = list(messages)
        resultat = _appel(f, (actuelle, messages), quoi)
        _intact(avant, messages, quoi, "les messages")
        if resultat is not None and not isinstance(resultat, str):
            raise AssertionError(f"{quoi} renvoie {_court(resultat)} ({type(resultat).__name__}) ; attendu un texte "
                                 "(« 1.5.0 ») ou None.")
        obtenus.append(resultat)
    attendus = [_ref_prochaine(a, m) for a, m in cas]
    if obtenus != attendus:
        conseil = _premier_bug(obtenus, lambda bug: [_ref_prochaine(a, m, bug) for a, m in cas], _BUGS_PROCHAINE)
        i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        actuelle, messages = cas[i]
        raise AssertionError(f"prochaine_version({actuelle!r}, {_court(messages, 140)}) renvoie {obtenus[i]!r} au lieu "
                             f"de {attendus[i]!r}. {conseil}".rstrip())
    for actuelle, messages, attendu, conseil in (
            ("2.3", ["fix: x"], ValueError, "« 2.3 » n'est pas une version SemVer."),
            ("v2.3.4", ["fix: x"], ValueError, "« v2.3.4 » n'est pas une version SemVer (le « v » est celui du tag)."),
            ("2.3.4", ["docs: x", None], TypeError, "un message de commit est un texte."),
            ("2.3.4", [b"fix: x"], TypeError, "des octets ne sont pas un texte : décodez-les d'abord."),
            (None, ["fix: x"], TypeError, "la version actuelle est un texte.")):
        _doit_lever(f, (actuelle, messages), f"prochaine_version({actuelle!r}, {messages!r})", conseil, attendu)


# ============================================================================ défi 238 : satisfait

_OPS = (">=", "<=", ">", "<", "=", "^", "~")


def _comparateurs(alternative: str, bug: str = "") -> list:
    resultat = []
    for morceau in alternative.split(","):
        morceau = morceau.strip()
        if morceau == "*":
            resultat.append(("*", None))
            continue
        op = next((o for o in _OPS if morceau.startswith(o)), "=")
        version = morceau[len(op):].strip() if morceau.startswith(op) else morceau
        _lire(version)
        resultat.append((op, version))
    return resultat


def _borne(op: str, version: str, bug: str = "") -> str:
    majeur, mineur, correctif, _, _ = _lire(version)
    fin = "" if bug == "sans_moins_zero" else "-0"
    if op == "~" and bug != "tilde_caret":
        return f"{majeur}.{mineur + 1}.0{fin}"
    if majeur > 0 or bug == "caret_zero":
        return f"{majeur + 1}.0.0{fin}"
    if mineur > 0:
        return f"0.{mineur + 1}.0{fin}"
    return f"0.0.{correctif + 1}{fin}"


def _respecte(version: str, op: str, cible, bug: str = "") -> bool:
    comparer = (lambda a, b: _ref_comparer(a, b, "texte")) if bug == "texte" else _ref_comparer
    if op == "*":
        return True
    c = comparer(version, cible)
    if op in ("^", "~"):
        if bug == "caret_sans_borne":
            return c >= 0
        if bug == "borne_incluse":   # « <= 2.0.0 » au lieu de « < 2.0.0-0 »
            return c >= 0 and comparer(version, _borne(op, cible, "sans_moins_zero")) <= 0
        return c >= 0 and comparer(version, _borne(op, cible, bug)) < 0
    return {"=": c == 0, ">=": c >= 0, "<=": c <= 0, ">": c > 0, "<": c < 0}[op]


def _pre_permise(version: str, comparateurs: list, bug: str = "") -> bool:
    majeur, mineur, correctif, pre, _ = _lire(version)
    if pre is None or bug == "pre_toujours":
        return True
    if bug == "pre_jamais":
        return False
    for op, v in comparateurs:
        if v is None:
            if bug == "etoile_pre":
                return True
            continue
        m, n, c, p, _ = _lire(v)
        if p is not None and (bug == "pre_sans_tuple" or (m, n, c) == (majeur, mineur, correctif)):
            return True
    return False


def _ref_satisfait(version: str, contrainte: str, bug: str = "") -> bool:
    """La bonne réponse ; `bug` rejoue une erreur classique."""
    _lire(version)
    if bug == "et_ou":   # (A || B), C : des OU dans des ET
        groupes = [[_comparateurs(a) for a in g.split("||")] for g in contrainte.split(",")]
        tous = [c for g in groupes for alt in g for c in alt]
        return (all(any(all(_respecte(version, o, v) for o, v in alt) for alt in g) for g in groupes)
                and _pre_permise(version, tous))
    alternatives = [_comparateurs(a, bug) for a in contrainte.split("||")]
    tous = [c for alt in alternatives for c in alt]
    for comparateurs in alternatives:
        if all(_respecte(version, op, v, bug) for op, v in comparateurs) and \
                _pre_permise(version, tous if bug == "pre_global" else comparateurs, bug):
            return True
    return False


_BUGS_SATISFAIT = {
    "texte": "Vous comparez les versions comme des textes (\"2.10.0\" < \"2.9.0\") : servez-vous de l'ordre du défi 236.",
    "caret_zero": "`^0.2.3` ne va pas jusqu'à 1.0.0 : en 0.y.z, chaque version mineure peut tout casser, donc `^0.2.3` "
                  "s'arrête à 0.3.0 (exclu), et `^0.0.3` à 0.0.4.",
    "caret_sans_borne": "`^1.2.3` n'est pas « >= 1.2.3 » : il s'arrête à 2.0.0 (exclu). C'est tout son intérêt : "
                        "une version majeure casse ceux qui s'en servent.",
    "borne_incluse": "La borne de `^` et `~` est EXCLUE : `^1.2.3` refuse 2.0.0 (et toutes ses pré-versions : la "
                     "borne est 2.0.0-0).",
    "tilde_caret": "`~1.2.3` n'accepte que des correctifs : >= 1.2.3 et < 1.3.0. C'est `^` qui accepte 1.3.0.",
    "pre_toujours": "Une pré-version n'est acceptée que si l'alternative en ÉCRIT une du même M.m.c : « >=2.0.0 » "
                    "refuse 2.2.0-rc.1 (sinon, `npm install` vous installerait des versions pas finies).",
    "pre_jamais": "Une pré-version EST acceptée quand l'alternative en écrit une du même M.m.c : 2.2.0-rc.1 respecte "
                  "« >=2.2.0-rc.0 ».",
    "pre_global": "La pré-version doit être écrite dans la MÊME alternative : dans « ^1.0.0 || =2.0.0-rc.1 », 1.5.0-rc.1 "
                  "ne remplit pas la première (qui n'écrit aucune pré-version), ni la seconde.",
    "pre_sans_tuple": "La pré-version écrite doit avoir les MÊMES majeur, mineur et correctif : « >=1.2.0-rc.0 » "
                      "accepte 1.2.0-rc.5, mais pas 1.9.0-beta.1.",
    "etoile_pre": "« * » n'écrit aucune pré-version : il ne laisse donc passer aucune pré-version.",
    "sans_moins_zero": "La borne de `^1.2.3` est 2.0.0-0, pas 2.0.0 : sinon 2.0.0-rc.1 (plus petite que 2.0.0) "
                       "passerait la borne. « -0 » est la plus petite pré-version possible.",
    "et_ou": "« , » (ET) lie plus fort que « || » (OU) : « A || B, C » veut dire A OU (B ET C). Découpez d'abord sur "
             "« || », puis chaque alternative sur « , ».",
}

_CAS_238 = [
    ("1.9.0", "^1.2.3"), ("2.0.0", "^1.2.3"), ("1.2.3", "^1.2.3"), ("1.2.2", "^1.2.3"), ("2.0.0-rc.1", "^1.2.3"),
    ("0.2.9", "^0.2.3"), ("0.3.0", "^0.2.3"), ("0.2.2", "^0.2.3"), ("0.0.3", "^0.0.3"), ("0.0.4", "^0.0.3"),
    ("1.2.9", "~1.2.3"), ("1.3.0", "~1.2.3"), ("0.2.9", "~0.2.3"), ("0.3.0", "~0.2.3"),
    ("2.4.1", ">=2.0.0, <3.0.0"), ("3.0.0", ">=2.0.0, <3.0.0"), ("2.0.0", ">2.0.0"), ("2.0.0", "<=2.0.0"),
    ("3.1.0", "^1.0.0 || ^3.0.0"), ("2.1.0", "^1.0.0 || ^3.0.0"),
    ("1.5.0", "^1.0.0 || ^3.0.0, >=3.2.0"), ("3.1.0", "^1.0.0 || ^3.0.0, >=3.2.0"),
    ("2.2.0-rc.1", ">=2.0.0"), ("2.2.0-rc.1", ">=2.2.0-rc.0"), ("2.2.0-rc.1", "*"), ("2.2.0", "*"),
    ("1.9.0-beta.1", ">=1.2.0-rc.0"), ("1.5.0-rc.1", "^1.0.0 || =2.0.0-rc.1"),
    ("2.0.0-rc.1", "^1.2.3, >=2.0.0-rc.0"), ("2.0.0-rc.1", "~2.0.0-rc.0"), ("2.0.0-rc.0", "~2.0.0-rc.1"),
    ("1.0.0+build.5", "=1.0.0"), ("1.0.0", "1.0.0"), ("1.0.0", " =  1.0.0 "), ("1.0.1", ">= 1.0.0 , < 1.1.0"),
    ("2.10.0", ">2.9.0"), ("2.10.0", "<2.9.0"), ("10.0.0", "^9.0.0"), ("2.10.0", "~2.9.0"), ("3.0.0", "^2.9.0"),
    ("2.11.0", "^2.9.0"), ("0.10.0", "^0.9.0"), ("1.2.3-beta.11", ">1.2.3-beta.2"),
    ("1.2.3-beta.2", ">1.2.3-beta.11"), ("0.0.0", "*"), ("1.0.0", "* || ^2.0.0"), ("1.0.0", "<1.0.0 || >1.0.0"),
]


def _cas_238_alea(graine: int) -> list[tuple[str, str]]:
    rng = random.Random(graine)
    versions = generer_versions(80, graine=graine) + ["0.2.3", "0.3.0", "1.2.3", "2.0.0", "2.0.0-rc.1"]
    cas = []
    for _ in range(400):
        alternatives = []
        for _ in range(rng.choice((1, 1, 2, 3))):
            morceaux = []
            for _ in range(rng.choice((1, 1, 2))):
                op = rng.choice(("^", "~", ">=", ">", "<", "<=", "=", "", "*"))
                morceaux.append("*" if op == "*" else op + rng.choice(versions).split("+")[0])
            alternatives.append(", ".join(morceaux))
        cas.append((rng.choice(versions), " || ".join(alternatives)))
    return cas


def _satisfait_juste(f) -> str:
    """'' si votre satisfait est juste sur les cas écrits à la main, sinon le premier écart."""
    for version, contrainte in _CAS_238:
        try:
            obtenu = f(version, contrainte)
        except Exception as exc:
            return f"satisfait({version!r}, {contrainte!r}) lève {_decrire(exc)}"
        if obtenu is not _ref_satisfait(version, contrainte):
            return f"satisfait({version!r}, {contrainte!r}) renvoie {obtenu!r} au lieu de {_ref_satisfait(version, contrainte)}"
    return ""


def verifier_satisfait(f) -> None:
    cas = _CAS_238 + _cas_238_alea(238)
    obtenus = []
    for version, contrainte in cas:
        quoi = f"satisfait({version!r}, {contrainte!r})"
        resultat = _appel(f, (version, contrainte), quoi)
        if type(resultat) is not bool:
            raise AssertionError(f"{quoi} renvoie {_court(resultat)} ({type(resultat).__name__}) ; attendu True ou False.")
        obtenus.append(resultat)
    attendus = [_ref_satisfait(v, c) for v, c in cas]
    if obtenus != attendus:
        conseil = _premier_bug(obtenus, lambda bug: [_ref_satisfait(v, c, bug) for v, c in cas], _BUGS_SATISFAIT)
        i = next(i for i, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        version, contrainte = cas[i]
        if not conseil:
            ecart = _comparer_juste(_fonction_eleve(f, "comparer_versions") or (lambda a, b: _ref_comparer(a, b)))
            if ecart:
                conseil = (f"Votre comparer_versions (défi 236) ne passe pas encore ({ecart}) : réussissez-le d'abord, "
                           "satisfait s'en sert.")
        raise AssertionError(f"satisfait({version!r}, {contrainte!r}) renvoie {obtenus[i]} au lieu de {attendus[i]}. "
                             f"{conseil}".rstrip())
    for version, contrainte, conseil in (
            ("1.0.0", "", "une contrainte vide ne veut rien dire."),
            ("1.0.0", "   ", "une contrainte vide ne veut rien dire."),
            ("1.0.0", ">=1.0.0,", "un morceau vide (après la virgule) est une faute de frappe : refusez-la."),
            ("1.0.0", "|| ^1.0.0", "une alternative vide est une faute de frappe : refusez-la."),
            ("1.0.0", "=>1.0.0", "« => » n'est pas un opérateur (c'est « >= »)."),
            ("1.0.0", "^1.2", "une version incomplète (« 1.2 ») n'est pas permise ici."),
            ("1.0.0", "^^1.0.0", "« ^^ » n'est pas un opérateur."),
            ("1.5.0", "^1.0.0 || =>2.0.0", "la seconde alternative est fausse, même si la première suffit : une "
                                           "faute de frappe doit se voir tout de suite, pas le jour où on l'atteint."),
            ("1.2", "^1.0.0", "« 1.2 » n'est pas une version SemVer."),
            ("1.0.0", "1.0.0 1.1.0", "deux versions sans virgule ni « || » : refusez.")):
        _doit_lever(f, (version, contrainte), f"satisfait({version!r}, {contrainte!r})", conseil)
    for version, contrainte in ((None, "^1.0.0"), ("1.0.0", None), ("1.0.0", ["^1.0.0"])):
        _doit_lever(f, (version, contrainte), f"satisfait({version!r}, {contrainte!r})",
                    "une version et une contrainte sont des textes.", TypeError)


# ============================================================================ défi 239 : resoudre


def _ref_resoudre(index: dict, demandes: dict, bug: str = "") -> dict:
    """La bonne réponse (Conflit si aucune) ; `bug` rejoue une erreur classique."""
    racine = {p: [c] for p, c in demandes.items()}

    def ordre(paquet):
        versions = list(index.get(paquet, {}))
        if bug == "ordre_index":
            return versions
        if bug == "texte":
            return sorted(versions, reverse=True)
        return sorted(versions, key=cmp_to_key(_ref_comparer), reverse=bug != "ancienne")

    def decider(choisis, contraintes, file):
        if not file:
            return choisis
        paquet, reste = file[0], file[1:]
        if bug == "keyerror":
            index[paquet]
        for version in ordre(paquet):
            connues = racine.get(paquet, []) if bug == "racine_seule" else contraintes[paquet]
            if not all(_ref_satisfait(version, c) for c in connues):
                continue
            avec = {**choisis, paquet: version}
            dependances = index[paquet][version]
            if bug != "sans_verif_choisis" and any(d in avec and not _ref_satisfait(avec[d], c)
                                                   for d, c in dependances.items()):
                continue
            nouvelles = {d: contraintes.get(d, []) + [c] for d, c in dependances.items()}
            ajout = [d for d in dependances if d not in avec and d not in reste]
            suite = ajout + reste if bug == "pile" else reste + ajout
            solution = decider(avec, {**contraintes, **nouvelles}, suite)
            if solution is not None or bug == "sans_retour":
                return solution
        return None

    solution = decider({}, {p: list(c) for p, c in racine.items()}, list(demandes))
    if solution is None:
        raise Conflit("aucune combinaison")
    return dict(sorted(solution.items())) if bug == "trie" else solution


_BUGS_RESOUDRE = {
    "texte": "Vous triez les versions comme des TEXTES (\"2.9.0\" > \"2.10.0\") : triez avec "
             "`key=cmp_to_key(comparer_versions)`.",
    "ancienne": "Vous essayez les versions de la plus ANCIENNE à la plus récente : un gestionnaire de paquets essaie "
                "d'abord la plus récente (`reverse=True`).",
    "ordre_index": "Vous essayez les versions dans l'ordre où l'index les écrit : rien ne garantit que cet ordre soit "
                   "celui des versions. Triez-les, de la plus récente à la plus ancienne (ordre du défi 236).",
    "sans_retour": "Vous abandonnez au premier échec : quand aucune version d'un paquet ne convient, il faut REVENIR "
                   "sur le choix précédent et essayer sa version suivante (retour arrière).",
    "racine_seule": "Vous ne vérifiez que la contrainte de `demandes` : les versions déjà choisies posent AUSSI des "
                    "contraintes sur leurs dépendances (graphiques 1.4.0 demande numerique ^1.9.0).",
    "sans_verif_choisis": "Quand vous choisissez une version, vérifiez que ses dépendances DÉJÀ CHOISIES respectent "
                          "ses contraintes : sinon vous installez deux versions incompatibles.",
    "pile": "Vous décidez les nouvelles dépendances AVANT les paquets déjà en attente (une pile) : elles vont au BOUT "
            "de la file (`reste + ajout`).",
    "trie": "Le dict doit suivre l'ordre où les paquets ont été DÉCIDÉS (l'ordre de la file), pas l'ordre alphabétique.",
    "keyerror": "Un paquet absent de l'index n'a simplement aucune version : on revient en arrière (et Conflit si "
                "rien d'autre ne marche), sans lever KeyError.",
}


def _explication(index: dict, demandes: dict, obtenu: dict) -> str:
    """Pourquoi votre solution n'en est pas une, ou '' si elle est valable."""
    for paquet, version in obtenu.items():
        if version not in index.get(paquet, {}):
            return f"{paquet} {version} n'existe pas dans l'index."
    for paquet, contrainte in demandes.items():
        if paquet not in obtenu:
            return f"le paquet demandé {paquet} manque."
        if not _ref_satisfait(obtenu[paquet], contrainte):
            return f"{paquet} {obtenu[paquet]} ne respecte pas la demande {contrainte!r}."
    necessaires = set(demandes)
    for paquet, version in obtenu.items():
        for dependance, contrainte in index[paquet][version].items():
            necessaires.add(dependance)
            if dependance not in obtenu:
                return f"{paquet} {version} demande {dependance}, qui manque."
            if not _ref_satisfait(obtenu[dependance], contrainte):
                return (f"{paquet} {version} demande {dependance} {contrainte!r}, mais vous avez choisi "
                        f"{dependance} {obtenu[dependance]}.")
    inutiles = [p for p in obtenu if p not in necessaires]
    if inutiles:
        return f"{inutiles} ne sont demandés par personne : n'installez que le nécessaire."
    return ""


def _un_resoudre(f, index: dict, demandes: dict, quoi: str, lignes_max: int | None = 2_000_000,
                 conseil_lent: str = "") -> None:
    try:
        attendu = _ref_resoudre(index, demandes)
    except Conflit:
        attendu = None
    avant_index, avant_demandes = copy.deepcopy(index), copy.deepcopy(demandes)
    if attendu is None:
        try:
            resultat = _trace(f, lambda: f(index, demandes), lignes_max)
        except NotImplementedError:
            raise
        except Conflit:
            _intact(avant_index, index, quoi, "l'index")
            return
        except _BoucleSansFin:
            raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
        except RecursionError:
            raise AssertionError(f"{quoi} lève RecursionError au lieu de Conflit.") from None
        except Exception as exc:
            conseil = _BUGS_RESOUDRE["keyerror"] if isinstance(exc, KeyError) else ""
            raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de Conflit (defis/livraison.py). "
                                 f"{conseil}".rstrip()) from None
        raise AssertionError(f"{quoi} devrait lever Conflit (aucune combinaison ne convient), mais renvoie "
                             f"{_court(resultat)}. {_explication(index, demandes, resultat) if isinstance(resultat, dict) else ''}".rstrip())
    try:
        obtenu = _appel(f, (index, demandes), quoi, lignes_max=lignes_max, conseil_lent=conseil_lent)
    except AssertionError as exc:
        if " lève KeyError" in str(exc):
            raise AssertionError(f"{exc} {_BUGS_RESOUDRE['keyerror']}") from None
        if " lève Conflit" in str(exc):
            raise AssertionError(f"{exc} Pourtant, une combinaison existe : {_court(attendu, 200)}. Revenez-vous bien "
                                 "en arrière quand un paquet n'a plus de version possible ?") from None
        raise
    _pas_none(obtenu, quoi)
    _intact(avant_index, index, quoi, "l'index")
    _intact(avant_demandes, demandes, quoi, "les demandes")
    if not isinstance(obtenu, dict):
        raise AssertionError(f"{quoi} renvoie {_court(obtenu)} ; attendu un dict {{paquet: version}}.")
    if list(obtenu.items()) == list(attendu.items()):
        return
    for bug, conseil in _BUGS_RESOUDRE.items():
        try:
            faux = _ref_resoudre(copy.deepcopy(index), demandes, bug)
        except (Conflit, KeyError):
            continue
        if list(faux.items()) == list(obtenu.items()):
            raise AssertionError(f"{quoi} renvoie {_court(obtenu, 200)} au lieu de {_court(attendu, 200)}. {conseil}")
    explication = _explication(index, demandes, obtenu)
    if explication:
        raise AssertionError(f"{quoi} renvoie {_court(obtenu, 200)}, qui ne tient pas : {explication}")
    if obtenu == attendu:
        raise AssertionError(f"{quoi} renvoie les bonnes versions, mais pas dans l'ordre où les paquets ont été "
                             f"décidés : {list(obtenu)} au lieu de {list(attendu)}.")
    plus_recent = next((p for p in attendu if p in obtenu and _ref_comparer(attendu[p], obtenu[p]) > 0), None)
    detail = (f" Par exemple, {plus_recent} pouvait être en {attendu[plus_recent]} (vous avez {obtenu[plus_recent]})."
              if plus_recent else "")
    raise AssertionError(f"{quoi} renvoie {_court(obtenu, 200)} : c'est une combinaison valable, mais pas celle qu'on "
                         f"attend, {_court(attendu, 200)}.{detail} Décidez les paquets dans l'ordre d'une FILE et "
                         "essayez chaque fois la version la plus récente d'abord.")


_POOL_VERSIONS = ["0.3.0", "0.3.5", "0.4.0", "1.0.0", "1.2.0", "1.10.0", "1.9.3", "2.0.0-rc.1", "2.0.0", "2.1.0",
                  "3.1.4", "1.2.0-beta.2"]
_POOL_CONTRAINTES = ["*", "^1.0.0", "~1.2.0", ">=1.2.0", "<2.0.0", "^0.3.0", ">=2.0.0-rc.0", "^1.0.0 || ^2.0.0",
                     "=1.10.0", ">=1.9.0, <3.0.0", "^2.0.0", "~0.3.0", ">1.2.0-beta.1"]


def _index_alea(rng: random.Random) -> tuple[dict, dict]:
    noms = [f"p{i}" for i in range(rng.randint(2, 6))]
    index = {}
    for nom in noms:
        versions = rng.sample(_POOL_VERSIONS, rng.randint(1, 5))
        autres = [n for n in noms if n != nom] + (["fantome"] if rng.random() < 0.15 else [])
        index[nom] = {v: {d: rng.choice(_POOL_CONTRAINTES) for d in rng.sample(autres, rng.randint(0, min(2, len(autres))))}
                      for v in versions}
    demandes = {n: rng.choice(_POOL_CONTRAINTES) for n in rng.sample(noms, rng.randint(1, min(3, len(noms))))}
    return index, demandes


def _index_long(n: int = 60, versions: int = 12) -> tuple[dict, dict]:
    """60 paquets en chaîne : seule la plus ANCIENNE version de chacun accepte la base 1.x qu'on impose."""
    index = {"base": {f"{k}.0.0": {} for k in range(1, versions + 1)}}
    for i in range(n):
        suivant = {f"p{i + 1:02d}": "*"} if i + 1 < n else {}
        index[f"p{i:02d}"] = {f"{k}.0.0": {"base": "^1.0.0" if k == 1 else "^2.0.0", **suivant}
                              for k in range(versions, 0, -1)}
    return index, {"base": "^1.0.0", "p00": "*"}


def verifier_resoudre(f) -> None:
    petit = {"app": {"2.0.0": {"lib": "^2.0.0"}, "1.0.0": {"lib": "^1.0.0"}}, "lib": {"2.1.0": {}, "1.4.0": {}},
             "outil": {"1.1.0": {"lib": "^1.0.0"}}}
    cas = [
        ("l'ordre des paquets décidés", {"z": {"1.0.0": {"m": "*"}}, "a": {"1.0.0": {}}, "m": {"1.0.0": {}}},
         {"z": "*", "a": "*"}),
        ("le 1er exemple de la docstring", petit, {"app": "*"}),
        ("le 2e exemple de la docstring", petit, {"app": "*", "outil": "^1.0.0"}),
        ("le 3e exemple de la docstring", petit, {"app": "^2.0.0", "outil": "*"}),
        ("des versions à deux chiffres", {"a": {"1.10.0": {}, "1.9.0": {}, "1.2.0": {}}}, {"a": "^1.0.0"}),
        ("des versions écrites dans le désordre", {"a": {"2.0.0": {}, "3.0.0": {}, "1.0.0": {}}}, {"a": "*"}),
        ("une rc à ne pas installer", {"a": {"2.0.0-rc.1": {}, "1.9.0": {}}}, {"a": "*"}),
        ("une rc demandée", {"a": {"2.0.0-rc.1": {}, "1.9.0": {}}}, {"a": ">=2.0.0-rc.0"}),
        ("un paquet inconnu", {"a": {"1.0.0": {}}}, {"a": "*", "b": "*"}),
        ("une dépendance inconnue, puis une version sans elle",
         {"a": {"2.0.0": {"absent": "*"}, "1.0.0": {}}}, {"a": "*"}),
        ("une contrainte posée APRÈS le choix", {"a": {"1.0.0": {"c": "^1.0.0"}}, "b": {"1.0.0": {"c": "^2.0.0"}},
                                                  "c": {"2.0.0": {}, "1.0.0": {}}}, {"a": "*", "b": "*"}),
        ("une dépendance déjà choisie trop récente",
         {"c": {"2.0.0": {}, "1.0.0": {}}, "a": {"2.0.0": {"c": "^1.0.0"}, "1.0.0": {}}}, {"c": "*", "a": "*"}),
        ("une dépendance demandée par deux paquets", {"a": {"1.0.0": {"c": ">=1.0.0"}}, "b": {"1.0.0": {"c": "<2.0.0"}},
                                                       "c": {"2.0.0": {}, "1.5.0": {}, "0.9.0": {}}},
         {"a": "*", "b": "*"}),
        ("l'ordre de la file", {"a": {"1.0.0": {"x": "*", "y": "*"}}, "b": {"1.0.0": {"z": "*"}}, "x": {"1.0.0": {}},
                                "y": {"1.0.0": {}}, "z": {"1.0.0": {"x": "*"}}}, {"a": "*", "b": "*"}),
        ("une file où l'ordre change la réponse",
         {"a": {"1.0.0": {"c": "*"}}, "b": {"1.0.0": {"d": "*"}}, "c": {"2.0.0": {"e": "^2.0.0"}, "1.0.0": {}},
          "d": {"2.0.0": {"e": "^1.0.0"}, "1.0.0": {}}, "e": {"2.0.0": {}, "1.0.0": {}}}, {"a": "*", "b": "*"}),
        ("deux paquets qui dépendent l'un de l'autre", {"a": {"1.0.0": {"b": "^1.0.0"}}, "b": {"1.0.0": {"a": "^1.0.0"}}},
         {"a": "*"}),
        ("le service de score", index_exemple(), {"churn-modele": "^3.0.0", "graphiques": "^1.2.0", "journal": "*"}),
        ("le service de score, impossible", index_exemple(), {"churn-modele": "^3.2.0", "numerique": "^1.9.0"}),
        ("aucune demande", {"a": {"1.0.0": {}}}, {}),
    ]
    for nom, index, demandes in cas:
        _un_resoudre_avec_satisfait(f, copy.deepcopy(index), dict(demandes), f"resoudre(<{nom}>, {_court(demandes, 90)})")
    rng = random.Random(239)
    for n in range(150):
        index, demandes = _index_alea(rng)
        _un_resoudre_avec_satisfait(f, index, demandes, f"resoudre(<index tiré au hasard n° {n} : {_court(index, 160)}>, "
                                                        f"{demandes!r})")
    index, demandes = _index_long()
    _un_resoudre(f, index, demandes, "resoudre(<60 paquets de 12 versions>, {'base': '^1.0.0', 'p00': '*'})",
                 lignes_max=4_000_000,
                 conseil_lent="Écartez une version dès qu'elle contredit un choix déjà fait (ici, la base 1.x) au lieu "
                              "d'essayer des combinaisons complètes : sur 60 paquets de 12 versions, il y en a "
                              "12 puissance 60.")


def _un_resoudre_avec_satisfait(f, index, demandes, quoi) -> None:
    try:
        _un_resoudre(f, index, demandes, quoi)
    except AssertionError as exc:
        satisfait_eleve = _fonction_eleve(f, "satisfait")
        ecart = _satisfait_juste(satisfait_eleve) if satisfait_eleve else ""
        if ecart:
            raise AssertionError(f"{exc} Votre satisfait (défi 238) ne passe pas encore ({ecart}) : réussissez-le "
                                 "d'abord, resoudre s'en sert.") from None
        raise


# ============================================================================ défi 240 : affecter


def _seau(experience: str, identifiant: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{experience}:{identifiant}".encode("utf-8")).digest()[:8], "big") % PALIERS


def _ref_affecter(experience: str, identifiant: str, poids: dict, bug: str = "") -> str:
    """La bonne réponse ; `bug` rejoue une erreur classique."""
    texte = f"{experience}:{identifiant}"
    if bug == "hash":
        seau = hash(texte) % PALIERS
    elif bug == "hash_tuple":
        seau = hash((experience, identifiant)) % PALIERS
    elif bug == "sans_experience":
        seau = int.from_bytes(hashlib.sha256(identifiant.encode("utf-8")).digest()[:8], "big") % PALIERS
    elif bug == "sans_separateur":
        seau = int.from_bytes(hashlib.sha256((experience + identifiant).encode("utf-8")).digest()[:8], "big") % PALIERS
    elif bug == "hex32":
        seau = int(hashlib.sha256(texte.encode("utf-8")).hexdigest()[:8], 16) % PALIERS
    elif bug == "petit_bout":
        seau = int.from_bytes(hashlib.sha256(texte.encode("utf-8")).digest()[:8], "little") % PALIERS
    elif bug == "tout":
        seau = int.from_bytes(hashlib.sha256(texte.encode("utf-8")).digest(), "big") % PALIERS
    elif bug == "md5":
        seau = int.from_bytes(hashlib.md5(texte.encode("utf-8")).digest()[:8], "big") % PALIERS
    else:
        seau = _seau(experience, identifiant)
    items = sorted(poids.items()) if bug == "trie" else list(poids.items())
    total = sum(poids.values())
    if bug == "modulo_total":
        n = int.from_bytes(hashlib.sha256(texte.encode("utf-8")).digest()[:8], "big") % total
        cumul = 0
        for variante, p in items:
            cumul += p
            if n < cumul:
                return variante
    cumul = 0
    for variante, p in items:
        cumul += p
        if (seau * total <= PALIERS * cumul) if bug == "inferieur_egal" else (seau * total < PALIERS * cumul):
            return variante
    raise AssertionError("inatteignable")


_BUGS_AFFECTER = {
    "hash": "Vous vous servez de `hash()` : il est SALÉ à chaque lancement de Python (PYTHONHASHSEED). Votre réponse "
            "change à chaque redémarrage du serveur, et d'un serveur à l'autre : un conseiller verrait le nouveau "
            "score, puis plus, puis de nouveau. Prenez `hashlib.sha256`.",
    "hash_tuple": "Vous vous servez de `hash((experience, identifiant))` : `hash()` est SALÉ à chaque lancement de "
                  "Python. Prenez `hashlib.sha256(f\"{experience}:{identifiant}\".encode(\"utf-8\"))`.",
    "sans_experience": "Vous ne hachez que l'identifiant : chaque conseiller tombe dans le MÊME seau pour toutes les "
                       "expériences, donc les mêmes 10 % essuient tous les plâtres. Hachez "
                       "f\"{experience}:{identifiant}\".",
    "sans_separateur": "Vous collez l'expérience et l'identifiant sans « : » : (\"ab\", \"c\") et (\"a\", \"bc\") "
                       "donneraient le même seau. Hachez f\"{experience}:{identifiant}\".",
    "hex32": "Vous lisez 8 caractères HEXADÉCIMAUX (32 bits) : la règle dit les 8 premiers OCTETS du `digest()` "
             "(64 bits), lus avec `int.from_bytes(…, \"big\")`.",
    "petit_bout": "Lisez les octets avec `int.from_bytes(…, \"big\")` (l'octet de poids fort en premier), pas "
                  "\"little\".",
    "tout": "Ne lisez que les 8 PREMIERS octets du `digest()` (`digest()[:8]`), pas les 32.",
    "md5": "La règle dit SHA-256 (`hashlib.sha256`), pas MD5.",
    "modulo_total": "Vous prenez le seau modulo le TOTAL des poids : quand les poids changent (10 % → 25 %), tout le "
                    "monde change de seau, et des conseillers qui voyaient « on » ne le voient plus. Le seau est modulo "
                    "PALIERS, toujours le même ; seules les frontières bougent : seau × total < PALIERS × cumul.",
    "inferieur_egal": "À la frontière, la variante est la SUIVANTE : la règle est seau × total < PALIERS × cumul "
                      "(strictement). Avec `<=`, le seau 2500 de 25 % / 75 % tomberait dans « on », et une variante de "
                      "poids 0 pourrait être choisie.",
    "trie": "Les variantes se prennent dans l'ordre du dict, pas triées par nom.",
}

_POIDS_240 = [{"on": 25, "off": 75}, {"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": 1, "d": 1},
              {"off": 90, "on": 10}, {"a": 0, "b": 5, "c": 0, "d": 5}, {"z": 3, "a": 1}, {"seul": 7},
              {"x": 10**6, "y": 1}, {"a": 2500, "b": 2500, "c": 5000}]


def _identifiants_pile(experience: str, seaux: set) -> list[str]:
    """Des identifiants dont le seau tombe PILE sur une frontière (trouvés en essayant, une fois pour toutes)."""
    trouves, i = {}, 0
    while len(trouves) < len(seaux):
        identifiant = f"P{i}"
        s = _seau(experience, identifiant)
        if s in seaux and s not in trouves:
            trouves[s] = identifiant
        i += 1
    return [trouves[s] for s in sorted(trouves)]


_PILE: dict = {}


def _cas_240() -> list[tuple[str, str, dict]]:
    if "cas" not in _PILE:
        cas = []
        ids = [f"U{i:04d}" for i in range(1, 481)] + ["Zoé", "🚀", "", "a:b", "C00194"]
        for experience in ("nouveau_score", "export_csv", ""):
            for poids in _POIDS_240:
                cas += [(experience, i, poids) for i in ids[:: (1 if poids is _POIDS_240[0] else 7)]]
        for identifiant in _identifiants_pile("nouveau_score", {0, 2500, 5000, 7500}):
            for poids in _POIDS_240:
                cas.append(("nouveau_score", identifiant, poids))
        _PILE["cas"] = cas
    return _PILE["cas"]


def _affecter_juste(f) -> str:
    for experience, identifiant, poids in _cas_240()[:400]:
        try:
            obtenu = f(experience, identifiant, poids)
        except Exception as exc:
            return f"affecter({experience!r}, {identifiant!r}, {poids!r}) lève {_decrire(exc)}"
        if obtenu != _ref_affecter(experience, identifiant, poids):
            return f"affecter({experience!r}, {identifiant!r}, {poids!r}) renvoie {obtenu!r}"
    return ""


def verifier_affecter(f) -> None:
    cas = _cas_240()
    obtenus = []
    for experience, identifiant, poids in cas:
        quoi = f"affecter({experience!r}, {identifiant!r}, {poids!r})"
        avant = dict(poids)
        obtenus.append(_appel(f, (experience, identifiant, poids), quoi))
        _intact(avant, poids, quoi, "les poids")
    attendus = [_ref_affecter(e, i, p) for e, i, p in cas]
    if obtenus != attendus:
        conseil = _premier_bug(obtenus, lambda bug: [_ref_affecter(e, i, p, bug) for e, i, p in cas], _BUGS_AFFECTER)
        k = next(k for k, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
        experience, identifiant, poids = cas[k]
        raise AssertionError(f"affecter({experience!r}, {identifiant!r}, {poids!r}) renvoie {obtenus[k]!r} au lieu de "
                             f"{attendus[k]!r} (son seau est {_seau(experience, identifiant)}). {conseil}".rstrip())
    for args, attendu, conseil in (
            (("e", "U1", {}), ValueError, "sans variante, il n'y a rien à servir."),
            (("e", "U1", {"a": 0, "b": 0}), ValueError, "un total nul ne permet de choisir aucune variante."),
            (("e", "U1", {"a": -1, "b": 2}), ValueError, "un poids négatif n'a pas de sens."),
            (("e", "U1", {"a": 0.5, "b": 0.5}), TypeError, "un poids est un ENTIER (0.5 se compte mal : 50 et 50)."),
            (("e", "U1", {"a": True, "b": 1}), TypeError, "True n'est pas un poids (même si True == 1)."),
            (("e", "U1", {"a": "25", "b": "75"}), TypeError, "\"25\" est un texte, pas un poids."),
            (("e", "U1", [("a", 1)]), TypeError, "poids est un dict {variante: poids}."),
            (("e", 42, {"a": 1}), TypeError, "l'identifiant est un texte : 42 et \"42\" doivent-ils tomber dans le même "
                                             "seau ? Refusez plutôt que de deviner."),
            ((None, "U1", {"a": 1}), TypeError, "le nom de l'expérience est un texte.")):
        _doit_lever(f, args, f"affecter{_court(args)}", conseil, attendu)


# ============================================================================ défi 241 : evaluer


def _condition(utilisateur: dict, attribut: str, operateur: str, operande, bug: str = "") -> bool:
    valeur = utilisateur.get(attribut)
    if valeur is None and not (bug == "absent_pas_dans" and operateur == "pas_dans"):
        return False
    if operateur == "dans":
        return valeur in operande
    if operateur == "pas_dans":
        return valeur not in operande
    if bug == "version_texte":
        c = _cmp(valeur, operande)
    else:
        try:
            c = _ref_comparer(valeur, operande)
        except (TypeError, ValueError):
            return False
    return c >= 0 if operateur == "version>=" else c < 0


def _valider_drapeau(drapeau) -> None:
    if not isinstance(drapeau, dict):
        raise TypeError("un drapeau est un dict")
    if type(drapeau.get("actif")) is not bool:
        raise TypeError("actif")
    if not isinstance(drapeau.get("coupe"), str):
        raise TypeError("coupe")
    if "defaut" not in drapeau:
        raise ValueError("defaut")
    servies = [drapeau["defaut"]] + [r["alors"] for r in drapeau.get("regles", [])]
    for servie in servies:
        if isinstance(servie, dict):
            if not servie:
                raise ValueError("poids vides")
            if any(type(p) is not int for p in servie.values()):
                raise TypeError("poids")
            if any(p < 0 for p in servie.values()) or sum(servie.values()) == 0:
                raise ValueError("poids")
        elif not isinstance(servie, str):
            raise TypeError("valeur servie")
    for regle in drapeau.get("regles", []):
        for attribut, operateur, operande in regle["si"]:
            if operateur not in OPERATEURS:
                raise ValueError("opérateur")
            if operateur in ("dans", "pas_dans") and not isinstance(operande, (list, tuple, set, frozenset)):
                raise TypeError("opérande")
            if operateur.startswith("version"):
                _lire(operande)


def _ref_evaluer(drapeaux: dict, nom: str, utilisateur: dict, bug: str = "", chemin: tuple = ()) -> Decision:
    """La bonne réponse ; `bug` rejoue une erreur classique."""
    identifiant = utilisateur.get("id") if isinstance(utilisateur, dict) else None
    if not isinstance(identifiant, str):
        raise TypeError("id")
    if nom in chemin:
        raise ValueError("cycle")
    if nom not in drapeaux:
        raise ValueError("inconnu")
    d = drapeaux[nom]
    _valider_drapeau(d)
    coupe = "off" if bug == "coupe_off" else d["coupe"]

    def cibles():
        for valeur, ids in d.get("cibles", {}).items():
            if identifiant in ids:
                return Decision(valeur, "ciblé")
        return None

    def servir(servie):
        return _ref_affecter(nom, identifiant, servie) if isinstance(servie, dict) else servie

    if bug == "cibles_avant_coupure" and cibles():
        return cibles()
    if not d["actif"]:
        return Decision(coupe, "coupé")
    if bug != "sans_prerequis":
        for autre, attendue in d.get("prerequis", {}).items():
            if _ref_evaluer(drapeaux, autre, utilisateur, bug, chemin + (nom,)).valeur != attendue:
                return Decision(coupe, f"prérequis {autre}")
    if bug != "cibles_apres_regles" and cibles():
        return cibles()
    regles = list(enumerate(d.get("regles", []), start=0 if bug == "zero" else 1))
    if bug == "derniere":
        regles.reverse()
    tous = any if bug == "ou" else all
    for numero, regle in regles:
        if tous(_condition(utilisateur, *c, bug=bug) for c in regle["si"]) if regle["si"] else True:
            return Decision(servir(regle["alors"]), f"règle {numero}")
    if bug == "cibles_apres_regles" and cibles():
        return cibles()
    return Decision(servir(d["defaut"]), "défaut")


_BUGS_EVALUER = {
    "cibles_avant_coupure": "Le bouton d'arrêt (\"actif\": False) l'emporte sur TOUT, ciblage compris : quand on "
                            "coupe à 9 h 02 parce que le centre d'appels est aveugle, les testeurs aussi doivent "
                            "repasser sur l'ancien écran.",
    "sans_prerequis": "Vous ignorez les prérequis : si « fiche_v2 » ne vaut pas « on » pour ce conseiller, le nouveau "
                      "score (qui s'affiche DANS la fiche v2) ne peut pas lui être servi → Decision(coupe, "
                      "\"prérequis fiche_v2\").",
    "absent_pas_dans": "Un attribut ABSENT ne remplit aucune condition, pas même « pas_dans » : un conseiller sans rôle "
                       "connu n'est peut-être pas « pas stagiaire ». Quand on ne sait pas, on ne promet rien.",
    "version_texte": "Vous comparez les versions d'appli comme des textes : \"4.10.2\" < \"4.2.0\" en texte, et les "
                     "conseillers qui ont l'appli la plus récente seraient exclus. Servez-vous de comparer_versions.",
    "ou": "Une règle ne s'applique que si TOUTES ses conditions sont remplies (`all`), pas une seule (`any`).",
    "derniere": "C'est la PREMIÈRE règle remplie qui s'applique, pas la dernière : arrêtez-vous dès qu'une règle "
                "est remplie.",
    "zero": "Les règles se numérotent à partir de 1 (« règle 1 » est la première) : `enumerate(regles, start=1)`.",
    "coupe_off": "Quand le drapeau est coupé ou qu'un prérequis manque, servez sa valeur \"coupe\", pas \"off\" écrit "
                 "en dur : un drapeau peut servir \"ancien\" / \"nouveau\", ou \"v1\" / \"v2\".",
    "cibles_apres_regles": "Le ciblage passe AVANT les règles : un testeur désigné par son identifiant reçoit sa valeur "
                           "même s'il remplit une règle.",
}


def _cas_241_mains() -> list[tuple[str, dict, str, dict]]:
    """Des petits cas où chaque erreur classique change la réponse."""
    base = {
        "a": {"actif": False, "coupe": "ancien", "cibles": {"nouveau": ["T1"]}, "defaut": "nouveau"},
        "b": {"actif": True, "coupe": "ancien", "prerequis": {"c": "on"}, "cibles": {"nouveau": ["T1"]},
              "defaut": "nouveau"},
        "c": {"actif": True, "coupe": "off", "regles": [{"si": [["appli", "version>=", "4.2.0"]], "alors": "on"}],
              "defaut": "off"},
        "d": {"actif": True, "coupe": "off",
              "regles": [{"si": [["role", "pas_dans", ["stagiaire"]]], "alors": "on"}], "defaut": "off"},
        "e": {"actif": True, "coupe": "off", "cibles": {"off": ["T1"]},
              "regles": [{"si": [["site", "dans", ["Lyon"]], ["role", "dans", ["superviseur"]]], "alors": "un"},
                         {"si": [["site", "dans", ["Lyon"]]], "alors": "deux"},
                         {"si": [], "alors": "trois"}],
              "defaut": "jamais"},
        "f": {"actif": True, "coupe": "off", "regles": [{"si": [["appli", "version<", "4.2.0"]], "alors": "vieux"}],
              "defaut": {"on": 50, "off": 50}},
        "g": {"actif": True, "coupe": "off", "prerequis": {"a": "nouveau"}, "defaut": "on"},
    }
    utilisateurs = [
        {"id": "T1", "site": "Lyon", "role": "superviseur", "appli": "4.10.2"},
        {"id": "U1", "site": "Lyon", "role": "superviseur", "appli": "4.10.2"},
        {"id": "U2", "site": "Lyon", "role": "conseiller", "appli": "4.1.0"},
        {"id": "U3", "site": "Nantes", "appli": "4.2.0"},
        {"id": "U4", "site": "Lille", "role": "stagiaire", "appli": None},
        {"id": "U5", "site": "Lyon", "role": None, "appli": "4.10"},
        {"id": "U6", "role": "conseiller", "appli": "4.2.0-beta.3"},
        {"id": "U7", "site": "Rennes", "role": "superviseur", "appli": "4.9.1"},
    ]
    return [(f"{_court({nom: base[nom]}, 230)}", base, nom, u) for nom in base for u in utilisateurs]


def _drapeaux_alea(rng: random.Random) -> dict:
    noms = [f"d{i}" for i in range(rng.randint(1, 5))]
    drapeaux = {}
    valeurs = ("on", "off", "v1", "v2")
    for i, nom in enumerate(noms):
        d = {"actif": rng.random() < 0.85, "coupe": rng.choice(valeurs)}
        if i and rng.random() < 0.5:   # des prérequis vers les drapeaux PRÉCÉDENTS : pas de cycle
            d["prerequis"] = {autre: rng.choice(valeurs) for autre in rng.sample(noms[:i], rng.randint(1, i))}
        if rng.random() < 0.5:
            d["cibles"] = {rng.choice(valeurs): [f"U{rng.randint(1, 60):04d}" for _ in range(rng.randint(1, 6))]
                           for _ in range(rng.randint(1, 2))}
        regles = []
        for _ in range(rng.randint(0, 3)):
            conditions = []
            for _ in range(rng.randint(0, 3)):
                attribut = rng.choice(("site", "role", "equipe", "appli", "appli", "pays"))
                if attribut == "appli":
                    conditions.append([attribut, rng.choice(("version>=", "version<")),
                                       rng.choice(("4.2.0", "4.10.0", "4.1.3", "4.2.0-beta.1"))])
                else:
                    pool = {"site": ["Lyon", "Nantes", "Lille", "Rennes"], "role": ["conseiller", "stagiaire",
                                                                                    "superviseur"],
                            "equipe": ["rétention", "facturation", "technique"], "pays": ["FR", "MA"]}[attribut]
                    conditions.append([attribut, rng.choice(("dans", "pas_dans")), rng.sample(pool, rng.randint(1, 2))])
            alors = rng.choice(valeurs) if rng.random() < 0.6 else {"on": rng.randint(0, 5), "off": rng.randint(1, 5)}
            regles.append({"si": conditions, "alors": alors})
        if regles:
            d["regles"] = regles
        d["defaut"] = rng.choice(valeurs) if rng.random() < 0.7 else {"v1": rng.randint(1, 3), "v2": rng.randint(0, 3)}
        drapeaux[nom] = d
    return drapeaux


def _un_lot_evaluer(f, cas: list, quoi_lot: str) -> None:
    obtenus = []
    for nom_cas, drapeaux, nom, utilisateur in cas:
        quoi = f"evaluer(<{nom_cas}>, {nom!r}, {_court(utilisateur, 110)})"
        avant_d, avant_u = copy.deepcopy(drapeaux), copy.deepcopy(utilisateur)
        resultat = _appel(f, (drapeaux, nom, utilisateur), quoi)
        _pas_none(resultat, quoi)
        _intact(avant_d, drapeaux, quoi, "les drapeaux")
        _intact(avant_u, utilisateur, quoi, "l'utilisateur")
        if not isinstance(resultat, tuple) or len(resultat) != 2:
            raise AssertionError(f"{quoi} renvoie {_court(resultat)} ; attendu Decision(valeur, raison) (defis/livraison.py).")
        obtenus.append(tuple(resultat))
    attendus = [tuple(_ref_evaluer(d, n, u)) for _, d, n, u in cas]
    if obtenus == attendus:
        return
    conseil = _premier_bug(obtenus, lambda bug: [tuple(_ref_evaluer(d, n, u, bug)) for _, d, n, u in cas],
                           _BUGS_EVALUER)
    k = next(k for k, (o, a) in enumerate(zip(obtenus, attendus)) if o != a)
    nom_cas, drapeaux, nom, utilisateur = cas[k]
    if not conseil:
        prealables = []
        for nom_fonction, juste in (("comparer_versions", _comparer_juste), ("affecter", _affecter_juste)):
            fonction = _fonction_eleve(f, nom_fonction)
            ecart = juste(fonction) if fonction else ""
            if ecart:
                prealables.append(f"votre {nom_fonction} ne passe pas encore ({ecart})")
        if prealables:
            conseil = ("Attention : " + " ; ".join(prealables) + ". evaluer s'en sert : réussissez d'abord les défis "
                       "236 et 240.")
    raise AssertionError(f"evaluer(<{nom_cas}>, {nom!r}, {_court(utilisateur, 120)}) renvoie {obtenus[k]!r} au lieu de "
                         f"{attendus[k]!r} ({quoi_lot}). {conseil}".rstrip())


def verifier_evaluer(f) -> None:
    _un_lot_evaluer(f, _cas_241_mains(), "petits cas écrits à la main")
    drapeaux = drapeaux_exemple()
    cas = [("drapeaux_exemple()", drapeaux, nom, c) for nom in drapeaux for c in conseillers_exemple()]
    _un_lot_evaluer(f, cas, "les drapeaux et les conseillers d'exemple")
    rng = random.Random(241)
    cas = []
    for n in range(60):
        drapeaux = _drapeaux_alea(rng)
        conseillers = generer_conseillers(30, graine=n)
        for c in conseillers[::3]:
            if rng.random() < 0.3:
                c["pays"] = rng.choice(("FR", "MA"))
        cas += [(f"drapeaux tirés au hasard n° {n} : {_court(drapeaux, 200)}", drapeaux, nom, c)
                for nom in drapeaux for c in conseillers]
    _un_lot_evaluer(f, cas, "drapeaux tirés au hasard")
    ok = {"actif": True, "coupe": "off", "defaut": "on"}
    u = {"id": "U1", "site": "Lyon", "appli": "4.2.0"}
    for drapeaux, nom, utilisateur, attendu, conseil in (
            ({"x": {**ok, "actif": "false"}}, "x", u, TypeError,
             "\"false\" est un TEXTE, et un texte non vide est vrai pour un `if` : le drapeau serait allumé. Exigez "
             "un bool (`type(actif) is bool`)."),
            ({"x": {**ok, "actif": 0}}, "x", u, TypeError, "actif vaut True ou False, pas 0."),
            ({"x": {"actif": True, "defaut": "on"}}, "x", u, TypeError, "il faut une valeur \"coupe\" (un texte)."),
            ({"x": {"actif": True, "coupe": "off"}}, "x", u, ValueError, "il faut une valeur \"defaut\"."),
            ({"x": {**ok, "regles": [{"si": [["appli", "version>", "4.2.0"]], "alors": "on"}]}}, "x", u, ValueError,
             "« version> » n'est pas dans OPERATEURS."),
            ({"x": {**ok, "regles": [{"si": [["site", "dans", "Lyon"]], "alors": "on"}]}}, "x",
             {"id": "U1", "site": "Ly"}, TypeError, "« dans » attend une LISTE : avec un texte, \"Ly\" in \"Lyon\" est "
                                                    "vrai, et le conseiller de « Ly » passerait."),
            ({"x": {**ok, "regles": [{"si": [], "alors": "on"}, {"si": [["appli", "version>=", "4.2"]], "alors": "on"}]}},
             "x", u, ValueError, "« 4.2 » n'est pas une version : la règle 2 est fausse, même si la règle 1 suffit. "
                                 "Validez tout le drapeau AVANT de servir."),
            ({"x": {**ok, "defaut": {"on": 0, "off": 0}}}, "x", u, ValueError, "des poids tous nuls (défi 240)."),
            ({"x": {**ok, "regles": [{"si": [], "alors": {"on": 0.5, "off": 0.5}}]}}, "x", u, TypeError,
             "un poids est un entier (défi 240)."),
            ({"x": ok}, "y", u, ValueError, "un drapeau inconnu : ValueError, pas KeyError."),
            ({"x": {**ok, "prerequis": {"y": "on"}}}, "x", u, ValueError, "un prérequis inconnu : ValueError."),
            ({"x": {**ok, "prerequis": {"y": "on"}}, "y": {**ok, "prerequis": {"x": "on"}}}, "x", u, ValueError,
             "x demande y qui demande x : un CYCLE. Gardez le chemin des drapeaux en cours d'évaluation et levez "
             "ValueError quand un nom y est déjà (sinon : RecursionError, ou pire, une boucle sans fin)."),
            ({"x": {**ok, "prerequis": {"x": "on"}}}, "x", u, ValueError, "x se demande lui-même : un cycle."),
            ({"x": ok}, "x", {"site": "Lyon"}, TypeError, "un utilisateur sans \"id\" ne peut pas être réparti."),
            ({"x": ok}, "x", {"id": 7}, TypeError, "l'id est un texte (défi 240).")):
        _doit_lever(f, (drapeaux, nom, utilisateur), f"evaluer({_court(drapeaux, 140)}, {nom!r}, {utilisateur!r})",
                    conseil, attendu)


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"comparer_versions": 236, "prochaine_version": 237, "satisfait": 238, "resoudre": 239, "affecter": 240,
            "evaluer": 241}

SERIE = Serie(
    code="39",
    titre="Livrer sans tout casser : comparer des versions, numéroter, contraindre, résoudre, déployer peu à peu",
    module_eleve="defis.serie_39",
    module_corrige="defis.corrige_serie_39",
    defis=(
        Defi(236, "comparer_versions", "🟢", "—", verifier_comparer_versions),
        Defi(237, "prochaine_version", "🟢", "—", verifier_prochaine_version),
        Defi(238, "satisfait", "🟡", "—", verifier_satisfait),
        Defi(239, "resoudre", "🔴", "—", verifier_resoudre),
        Defi(240, "affecter", "🟡", "—", verifier_affecter),
        Defi(241, "evaluer", "🔴", "—", verifier_evaluer),
    ),
)

"""Vérifications automatiques de la série 45 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est le comité de direction, qui ne changera la règle « résili » que si vos chiffres tiennent, et
l'enquête de satisfaction, qui vous envoie les commentaires tels que les clients les tapent vraiment.

- il compare vos mots, vos modèles, vos probabilités et vos mots révélateurs à ceux d'une version de référence (à un
  milliardième près pour les nombres : les règles sont fixées pour qu'il n'y ait qu'une bonne réponse), sur des cas
  choisis et sur les 5 000 commentaires de l'exemple ;
- quand votre résultat est faux, il rejoue les erreurs classiques (`split()` au lieu d'une vraie découpe, `\\w` qui
  garde « _ », NFD au lieu de NFKD, le dernier mot oublié, la négation qui ne dure qu'un mot ou qui s'annule, « plus »
  pris pour une négation, le vocabulaire de la seule classe, les mots distincts comptés au lieu des occurrences, la
  probabilité a priori uniforme, `math.log10`, le mot répété compté une fois, l'écart à la moyenne des autres classes,
  le mot oublié resté dans le vocabulaire, l'oubli à moitié fait…) pour nommer la vôtre ;
- il choisit des cas où chaque erreur se voit (« 3ᵉ », « ﬁbre », un prénom en arabe, « ne … pas », un commentaire de
  3 000 mots, des écarts à égalité, trois classes, le dernier document d'une classe oublié), et compte les lignes
  exécutées dans votre fichier quand une solution naïve serait trop lente ;
- pendant les défis 273 et 277, il prête sa propre `mots` (et ses `entrainer` et `probabilites`) à votre fichier :
  chaque défi est noté seul.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
import re
import sys
import unicodedata
from collections import Counter

from .verbatims import (
    FINS_DE_PORTEE,
    LETTRES_LIEES,
    NEGATIONS,
    PONCTUATION,
    PREFIXE,
    Modele,
    commentaires_exemple,
    generer_commentaires,
    nouveaux_exemple,
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

_LOG_ZERO = ("`math.log(0)` : vous prenez le logarithme d'une probabilité nulle. Sans lissage, un mot jamais vu dans une "
             "classe y a une probabilité de 0 : ajoutez alpha à chaque compte, (n(c, mot) + alpha) / (N(c) + alpha × "
             "taille du vocabulaire). Et une classe sans aucun document n'a plus rien à faire dans le modèle.")
_SOUS_DEPASSEMENT = ("Une division par zéro : sur un long commentaire, `math.exp(score)` vaut 0.0 pour TOUTES les classes "
                     "(un score de -800 est trop petit pour un float), ou un produit de probabilités tombe à 0.0, et la "
                     "somme aussi. Additionnez des LOGARITHMES, puis retranchez le PLUS GRAND des scores avant "
                     "`math.exp` : la plus grande exponentielle vaut 1, et la somme ne peut plus être nulle.")
_DEBORDEMENT = ("Un nombre a débordé (`math.exp` d'un nombre trop grand) : retranchez le PLUS GRAND des scores avant "
                "`math.exp`, pas le plus petit. Ainsi chaque différence est <= 0 et chaque exponentielle entre 0 et 1.")
_MOT_INCONNU = ("Un mot qui n'est pas dans le vocabulaire (jamais vu à l'entraînement) n'a pas de log_proba : il ne dit "
                "rien sur la classe, ignorez-le (`if mot in vocabulaire`), au lieu de planter.")


def _conseil_exception(exc: BaseException) -> str:
    texte = str(exc)
    if isinstance(exc, ValueError) and "math domain" in texte:
        return _LOG_ZERO
    if isinstance(exc, ZeroDivisionError):
        return _SOUS_DEPASSEMENT
    if isinstance(exc, OverflowError):
        return _DEBORDEMENT
    if isinstance(exc, KeyError):
        return _MOT_INCONNU
    if isinstance(exc, IndexError):
        return "Un indice sort de la liste : vérifiez vos bornes."
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


class _Echec:
    """Ce que votre code a fait d'anormal sur un cas (une exception, un mauvais type), déjà dit en français."""

    def __init__(self, message: str) -> None:
        self.message = message


def _diagnostic(sorties, cas, rejouer, bugs: dict, egal) -> str:
    """Le conseil du premier bug dont le rejeu donne EXACTEMENT vos réponses, sur TOUS les cas où votre code a répondu
    (pas seulement le premier faux : deux erreurs peuvent donner la même réponse sur un cas, et pas sur les autres)."""
    for bug, conseil in bugs.items():
        pareil = True
        for x, o in zip(cas, sorties):
            if isinstance(o, _Echec):
                continue
            try:
                faux = rejouer(x, bug)
            except Exception:
                pareil = False
                break
            if not egal(o, faux):
                pareil = False
                break
        if pareil:
            return conseil
    return ""


def _nombre(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _proche(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return _nombre(a) and _nombre(b) and abs(a - b) <= tol * max(1.0, abs(b))


def _proche_proba(a, b) -> bool:
    """Deux probabilités égales à 1e-7 près : une somme de 3 000 logarithmes, faite dans un autre ordre (ou avec
    `math.fsum`), peut différer de 1e-8 ; les erreurs classiques, elles, changent bien plus."""
    return _nombre(a) and _nombre(b) and abs(a - b) <= 1e-7


def _listes(egal):
    return lambda a, b: len(a) == len(b) and all(egal(x, y) for x, y in zip(a, b))


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


def _liste_de_textes(r, quoi: str, attendu) -> None:
    """Exige une liste de str ; explique sinon."""
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if isinstance(r, (set, frozenset)):
        raise AssertionError(f"{quoi} renvoie un ensemble ({_court(r, 80)}) : un ensemble perd l'ORDRE et les "
                             "RÉPÉTITIONS des mots, que le modèle compte. Rendez une liste.")
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} ({type(r).__name__}) au lieu d'une liste de mots, comme "
                             f"{_court(attendu, 80)}.")
    for mot in r:
        if not isinstance(mot, str):
            raise AssertionError(f"{quoi} contient {mot!r} ({type(mot).__name__}) : un mot est un texte (str).")


# ============================================================================ défi 272 : mots

_SANS_LIGATURES = str.maketrans(LETTRES_LIEES)


def _normaliser(texte: str, bug: str = "") -> str:
    t = texte if bug == "sans_casse" else texte.casefold()
    if bug == "nfd":
        t = unicodedata.normalize("NFD", t)
    elif bug != "accents":
        t = unicodedata.normalize("NFKD", t)
    if bug == "ascii":
        return t.encode("ascii", "ignore").decode("ascii")
    if bug != "accents":
        t = "".join(c for c in t if not unicodedata.combining(c))
    return t if bug == "ligatures" else t.translate(_SANS_LIGATURES)


def _ref_mots(texte, bug: str = "") -> list[str]:
    """Les mots de référence ; `bug` rejoue une erreur classique."""
    if not isinstance(texte, str):
        raise TypeError(f"texte doit être un str, pas {type(texte).__name__}")
    t = _normaliser(texte, bug)
    if bug == "split":
        return t.split()
    if bug == "w":
        return re.findall(r"\w+", t)
    if bug == "az":
        return re.findall(r"[a-z0-9]+", t)
    garder = str.isalpha if bug == "isalpha" else str.isalnum
    resultat, mot = [], []
    for c in t:
        if garder(c):
            mot.append(c)
        elif mot:
            resultat.append("".join(mot))
            mot = []
    if mot and bug != "dernier_mot":
        resultat.append("".join(mot))
    return sorted(set(resultat)) if bug == "ensemble" else resultat


_BUGS_MOTS = {
    "dernier_mot": "Le DERNIER mot disparaît quand rien ne le suit (« je résilie » donne seulement ['je']) : votre boucle "
                   "range un mot quand arrive un séparateur ; après la boucle, rangez aussi le mot en cours, s'il n'est "
                   "pas vide.",
    "split": "`split()` ne coupe qu'aux espaces : « cher! », « l'offre » et « mois-ci » restent collés à leur ponctuation. "
             "Un mot est une suite de caractères pour lesquels `c.isalnum()` est vrai ; TOUT le reste sépare les mots.",
    "w": "`\\w` (dans une expression régulière) prend le tiret bas « _ » pour une lettre : « mot_de_passe » reste un mot. "
         "Le tiret bas est réservé aux marques de négation du défi 273 (« non_resilier ») : coupez sur tout ce qui "
         "n'est pas `c.isalnum()`.",
    "az": "`[a-z0-9]` efface les lettres des autres alphabets : le prénom « محمد » disparaît. Rien ne doit disparaître : un "
          "mot est une suite de caractères pour lesquels `c.isalnum()` est vrai, quel que soit l'alphabet.",
    "isalpha": "`isalpha()` efface les chiffres : « 4G » devient « g », « 29,99 » disparaît. Un mot est fait de lettres ET "
               "de chiffres : `c.isalnum()`.",
    "ascii": "`.encode(\"ascii\", \"ignore\")` efface ce qui n'est pas ASCII au lieu de le transcrire : « œuvre » devient "
             "« uvre », et le prénom en arabe disparaît. Retirez seulement les accents (`unicodedata.combining(c)`), "
             "remplacez « œ » et « æ » avec `LETTRES_LIEES`, et gardez les autres lettres.",
    "nfd": "NFD sépare les accents, mais garde les formes « de présentation » : « 3ᵉ » reste « 3ᵉ » (et pas « 3e »), "
           "« ４Ｇ » (pleine chasse) reste « ４ｇ ». NFKD (le K de « compatibilité ») les ramène à leur forme simple : "
           "`unicodedata.normalize(\"NFKD\", ...)`.",
    "accents": "Les accents sont restés : « résilie » et « resilie » sont deux mots pour votre modèle, alors que 30 % des "
               "clients tapent sans accents. Décomposez (`unicodedata.normalize(\"NFKD\", ...)`), puis écartez les "
               "caractères dont `unicodedata.combining(c)` n'est pas nul.",
    "ligatures": "« œ » et « æ » ne se décomposent pas : « cœur » reste « cœur », alors que le client qui tape « coeur » "
                 "écrit le même mot. Remplacez-les avec `LETTRES_LIEES` : `.translate(str.maketrans(LETTRES_LIEES))`.",
    "sans_casse": "Les majuscules sont restées : « RÉSILIE » et « résilie » sont deux mots pour votre modèle. Commencez "
                  "par `texte.casefold()`.",
    "ensemble": "Vous rendez les mots distincts et triés (`sorted(set(...))`) : le modèle a besoin de l'ORDRE (pour les "
                "négations du défi 273) et des RÉPÉTITIONS (il compte les occurrences). Rendez tous les mots, dans "
                "l'ordre du texte.",
}


def _cas_272() -> list[str]:
    cas = [
        "Je RÉSILIE l’offre ﬁbre !!! 😡",
        "C'est la 3ᵉ coupure ce mois-ci : 29,99 € pour ÇA ?",
        "Œuvre d'art, mot_de_passe",
        "Merci à محمد du service client",
        "je résilie",
        "resilie RÉSILIE Résilie résilie",
        "résilier sans accent précomposé",
        "ＲＩＯ et ４Ｇ",
        "Ex æquo, ma sœur et mon cœur",
        "cher!cher?cher",
        "l'offre l’offre lʼoffre",
        "tél. 06 12 34 56 78",
        "4G 5G wifi_6",
        "tab\tnouvelle\nligne",
        "Ma box 📦➡️🗑️ point",
        "İstanbul et ǅemal",
        "m² et ½ tarif",
        "!!!",
        "",
        "   \n ",
        "’",
        "a",
        "Fin",
    ]
    return cas + [texte for texte, _ in commentaires_exemple() + nouveaux_exemple()]


def _indice_mots(obtenu: list, attendu: list) -> str:
    """Un indice tiré de la différence, quand aucune erreur classique ne correspond."""
    if any("_" in mot for mot in obtenu):
        return _BUGS_MOTS["w"]
    if any(not mot.isalnum() for mot in obtenu):
        return "Un de vos mots contient autre chose que des lettres et des chiffres : " + _BUGS_MOTS["split"]
    if any(unicodedata.combining(c) or c != c.casefold() for mot in obtenu for c in mot):
        return _BUGS_MOTS["accents"]
    if len(obtenu) < len(attendu):
        return "Il vous manque des mots : rien ne doit disparaître, sauf les séparateurs."
    return ""


def verifier_mots(f) -> None:
    cas = _cas_272()
    obtenus, attendus = [], [_ref_mots(texte) for texte in cas]
    for texte, attendu in zip(cas, attendus):
        quoi = f"mots({_court(texte, 70)})"
        r = _appel(f, (texte,), quoi)
        _liste_de_textes(r, quoi, attendu)
        obtenus.append(r)
    for texte, r, attendu in zip(cas, obtenus, attendus):
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_mots(t, bug) for t in cas], _BUGS_MOTS,
                                   lambda a, b: a == b) or _indice_mots(r, attendu)
            raise AssertionError(f"mots({_court(texte, 70)}) renvoie {_court(r, 200)} au lieu de {_court(attendu, 200)}. "
                                 f"{conseil}".rstrip())

    for args, conseil in (((None,), "None n'est pas un texte."),
                          ((b"resilier",), "Des octets ne sont pas un texte : il faut d'abord les décoder (`.decode()`)."),
                          ((42,), "42 n'est pas un texte."),
                          ((["je", "resilie"],), "Une liste n'est pas un texte.")):
        _doit_lever(f, args, f"mots{_court(args, 40)}", conseil, TypeError)


# ============================================================================ défi 273 : avec_negations


def _propositions(texte: str, bug: str = "") -> list[str]:
    if bug == "tout_le_texte":
        return [texte]
    if bug == "point_seulement":
        return texte.split(".")
    propositions, courante = [], []
    for c in texte:
        if c in PONCTUATION:
            propositions.append("".join(courante))
            courante = []
        else:
            courante.append(c)
    propositions.append("".join(courante))
    return propositions


def _ref_negations(texte, bug: str = "") -> list[str]:
    """Les mots avec négations de référence ; `bug` rejoue une erreur classique."""
    if not isinstance(texte, str):
        raise TypeError(f"texte doit être un str, pas {type(texte).__name__}")
    negations = NEGATIONS | {"plus"} if bug == "plus" else NEGATIONS - {"n"} if bug == "sans_n" else NEGATIONS
    resultat = []
    for proposition in _propositions(texte, bug):
        nie, reste = False, 0
        for mot in _ref_mots(proposition):
            if mot in negations:
                resultat.append(PREFIXE + mot if bug == "negations_marquees" and nie else mot)
                nie = (not nie) if bug == "bascule" else True
                reste = 1
            elif mot in FINS_DE_PORTEE and bug != "sans_fin_de_portee":
                resultat.append(PREFIXE + mot if bug == "fin_marquee" and nie else mot)
                nie, reste = False, 0
            elif bug == "fenetre_1":
                resultat.append(PREFIXE + mot if reste else mot)
                reste = 0
            else:
                resultat.append(PREFIXE + mot if nie else mot)
    return resultat


_BUGS_NEGATIONS = {
    "fenetre_1": "Votre négation ne marque que le mot SUIVANT : dans « je n'ai jamais eu de coupure », « coupure » est nié "
                 "aussi. La négation reste ouverte jusqu'à la fin de la proposition (ou jusqu'à « mais », « car »…).",
    "bascule": "Deux mots de négation s'annulent chez vous : « je ne vais pas résilier » donne « resilier » sans marque. "
               "En français, « ne … pas » est UNE négation en deux morceaux : un mot de NEGATIONS ouvre la négation "
               "(`nie = True`), même si elle est déjà ouverte ; il ne la referme jamais.",
    "tout_le_texte": "Votre négation traverse la ponctuation : dans « Pas de coupure, le débit est excellent », « excellent » "
                     "n'est pas nié. Coupez d'abord le texte en propositions (à chaque caractère de PONCTUATION), et "
                     "repartez « sans négation » au début de CHACUNE.",
    "point_seulement": "Vous ne coupez les propositions qu'aux points : une virgule, un point d'exclamation, un retour à la "
                       "ligne terminent aussi une proposition. Coupez à chaque caractère de PONCTUATION (« . , ; : ! ? … ( "
                       ") » et « \\n »).",
    "sans_fin_de_portee": "« mais » (et « car », « donc », « sauf », « pourtant », « cependant ») referme la négation : "
                          "« je ne résilie pas mais le débit est lent » ne nie pas « débit ». Utilisez FINS_DE_PORTEE.",
    "negations_marquees": "Un mot de négation n'est jamais marqué : « je ne vais pas » donne « pas », pas « non_pas ». "
                          "Seuls les AUTRES mots reçoivent le préfixe.",
    "fin_marquee": "Le mot qui referme la négation (« mais ») n'est pas marqué : il est gardé tel quel, puis la négation se "
                   "referme.",
    "plus": "« plus » n'ouvre pas de négation : « c'est plus cher » veut dire « plus cher », pas « pas cher ». Il n'est "
            "pas dans NEGATIONS (et « je ne veux plus » est nié grâce à « ne »).",
    "sans_n": "L'élision « n' » (« je n'ai jamais ») devient le mot « n » au défi 272 : il ouvre la négation, comme « ne ». "
              "Utilisez la constante NEGATIONS, qui le contient.",
}


def _cas_273() -> list[str]:
    cas = [
        "Je ne vais pas résilier, le débit est excellent.",
        "Je n'ai jamais eu de coupure, mais c'est plus cher qu'avant.",
        "Sans fibre pas de télétravail. Je résilie !",
        "Je ne résilie pas mais le débit est lent",
        "C'est plus cher qu'avant et je ne veux plus de cette box",
        "Pas de coupure, le débit est excellent",
        "Aucun souci ; rien à redire ! Je recommande",
        "Je ne suis pas satisfait (vraiment pas) et je pars",
        "Pas de réseau\nJe pars chez un concurrent",
        "Je n’ai rien reçu… je résilie",
        "ni la box ni le wifi ne marchent donc je résilie",
        "Personne ne répond, je suis à bout",
        "pas pas pas résilier",
        "Je ne veux pas résilier car je suis satisfait sauf pour le prix",
        "Le technicien n'est jamais venu: j'attends",
        "Aucune coupure (sauf hier) mais pas cher",
        "Je n'ai pas dit que je ne partirais pas",
        "JE NE RÉSILIE PAS.",
        "Rien ne marche mais rien",
        "jamais",
        "mais",
        "...",
        "",
    ]
    return cas + [texte for texte, _ in commentaires_exemple() + nouveaux_exemple()]


def verifier_avec_negations(f) -> None:
    with _Pretes(f, {"mots": _ref_mots}):
        _verifier_negations(f)


def _verifier_negations(f) -> None:
    cas = _cas_273()
    obtenus, attendus = [], [_ref_negations(texte) for texte in cas]
    for texte, attendu in zip(cas, attendus):
        quoi = f"avec_negations({_court(texte, 70)})"
        r = _appel(f, (texte,), quoi)
        _liste_de_textes(r, quoi, attendu)
        obtenus.append(r)
    for texte, r, attendu in zip(cas, obtenus, attendus):
        if r != attendu:
            conseil = _premier_bug(obtenus, lambda bug: [_ref_negations(t, bug) for t in cas], _BUGS_NEGATIONS,
                                   lambda a, b: a == b)
            if not conseil and [m.removeprefix(PREFIXE) for m in r] != _ref_mots(texte):
                conseil = ("Sans les préfixes, vos mots ne sont pas ceux de `mots(texte)` : prenez les mots de chaque "
                           "proposition avec `mots(proposition)` (le correcteur vous prête la sienne).")
            raise AssertionError(f"avec_negations({_court(texte, 70)}) renvoie {_court(r, 220)} au lieu de "
                                 f"{_court(attendu, 220)}. {conseil}".rstrip())

    for args, conseil in (((None,), "None n'est pas un texte."), ((["je", "ne"],), "Une liste n'est pas un texte."),
                          ((42,), "42 n'est pas un texte.")):
        _doit_lever(f, args, f"avec_negations{_court(args, 40)}", conseil, TypeError)


# ============================================================================ défi 274 : entrainer


def _ref_entrainer(documents, etiquettes, alpha=1.0, bug: str = "") -> Modele:
    """Le modèle de référence ; `bug` rejoue une erreur classique."""
    nb_documents = Counter(etiquettes)
    comptes = {c: Counter() for c in nb_documents}
    for document, etiquette in zip(documents, etiquettes):
        comptes[etiquette].update(set(document) if bug == "presence" else document)
    classes = list(dict.fromkeys(etiquettes)) if bug == "classes_ordre" else sorted(nb_documents)
    vocabulaire = sorted(set().union(*comptes.values()))
    log = math.log10 if bug == "log10" else (lambda x: x) if bug == "probas" else math.log
    tous_les_mots = sum(sum(c.values()) for c in comptes.values())
    a_priori = {}
    for c in classes:
        if bug == "a_priori_uniforme":
            a_priori[c] = log(1 / len(classes))
        elif bug == "a_priori_mots":
            a_priori[c] = log(sum(comptes[c].values()) / tous_les_mots)
        else:
            a_priori[c] = log(nb_documents[c] / len(documents))
    log_proba = {}
    for c in classes:
        total = len(comptes[c]) if bug == "distincts" else sum(comptes[c].values())
        v = len(comptes[c]) if bug == "v_classe" else len(vocabulaire)
        if bug == "alpha_sans_v":
            denominateur = total + alpha
        elif bug == "v_sans_alpha":
            denominateur = total + v
        else:
            denominateur = total + alpha * v
        log_proba[c] = {mot: log((comptes[c][mot] + alpha) / denominateur) for mot in vocabulaire}
    return Modele(classes=classes, log_a_priori=a_priori, log_proba=log_proba)


_BUGS_ENTRAINER = {
    "probas": "Vous rangez des PROBABILITÉS, pas leurs logarithmes : log_a_priori et log_proba contiennent des "
              "math.log(...). Un produit de 300 probabilités vaut 0.0 en float ; une somme de 300 logarithmes, non.",
    "log10": "Vous prenez le logarithme décimal (`math.log10`) : « log », ici, c'est le logarithme NÉPÉRIEN, `math.log` "
             "(celui de `math.exp`, au défi 275).",
    "a_priori_uniforme": "Vos probabilités a priori sont toutes égales (1 / nombre de classes) : P(classe) est la part des "
                         "DOCUMENTS de cette classe. 20 % des clients partent : P(part) = 0,2, pas 0,5.",
    "a_priori_mots": "Vos probabilités a priori comptent des MOTS : P(classe) est la part des DOCUMENTS de la classe "
                     "(un long commentaire ne compte pas plus qu'un court). `Counter(etiquettes)`.",
    "presence": "Vous comptez les DOCUMENTS qui contiennent un mot, pas ses OCCURRENCES : un mot répété dans un "
                "commentaire compte autant de fois qu'il apparaît (c'est le Bayes naïf « multinomial »). "
                "`comptes[etiquette].update(document)`, sans `set`.",
    "distincts": "Votre N(c) compte les mots DISTINCTS de la classe : c'est le nombre TOTAL de mots de ses documents, "
                 "répétitions comprises (`sum(comptes[c].values())`).",
    "v_classe": "Au dénominateur, la taille du vocabulaire est celle de TOUT le vocabulaire (les mots de toutes les "
                "classes), pas celle des mots vus dans la classe : N(c) + alpha × len(vocabulaire). Sinon, les "
                "probabilités d'une classe ne font plus 1.",
    "alpha_sans_v": "Votre dénominateur est N(c) + alpha : on ajoute alpha à CHAQUE mot du vocabulaire au numérateur, "
                    "donc alpha × taille du vocabulaire au dénominateur, pour que les probabilités fassent 1.",
    "v_sans_alpha": "Votre dénominateur est N(c) + taille du vocabulaire : c'est juste quand alpha vaut 1, pas sinon. "
                    "N(c) + alpha × taille du vocabulaire.",
    "classes_ordre": "Vos classes sont dans l'ordre où elles apparaissent : `classes` est rangée dans l'ordre croissant "
                     "(`sorted`), pour que le modèle ne dépende pas de l'ordre des documents.",
}


def _modeles_egaux(a, b) -> bool:
    try:
        return (list(a.classes) == list(b.classes) and set(a.log_a_priori) == set(b.log_a_priori)
                and all(_proche(a.log_a_priori[c], b.log_a_priori[c]) for c in b.log_a_priori)
                and set(a.log_proba) == set(b.log_proba)
                and all(set(a.log_proba[c]) == set(b.log_proba[c])
                        and all(_proche(a.log_proba[c][m], v) for m, v in b.log_proba[c].items())
                        for c in b.log_proba))
    except Exception:
        return False


def _ecart_modeles(obtenu, attendu, quoi: str) -> str:
    """La PREMIÈRE différence entre deux modèles, dite en français ("" s'ils sont égaux)."""
    for attribut in ("classes", "log_a_priori", "log_proba"):
        if not hasattr(obtenu, attribut):
            if isinstance(obtenu, dict):
                return (f"{quoi} renvoie un dict : rendez un `Modele(classes=..., log_a_priori=..., log_proba=...)` (la "
                        "classe est dans defis/verbatims.py, déjà importée).")
            return (f"{quoi} renvoie {_court(obtenu, 80)} ({type(obtenu).__name__}), qui n'a pas d'attribut `{attribut}` : "
                    "rendez un `Modele(classes=..., log_a_priori=..., log_proba=...)`.")
    if not isinstance(obtenu.classes, list):
        return f"{quoi}.classes vaut {_court(obtenu.classes, 80)} : une LISTE (`sorted(...)` en rend une)."
    if obtenu.classes != attendu.classes:
        if sorted(obtenu.classes, key=repr) == sorted(attendu.classes, key=repr):
            return f"{quoi}.classes vaut {obtenu.classes!r} au lieu de {attendu.classes!r}. {_BUGS_ENTRAINER['classes_ordre']}"
        return f"{quoi}.classes vaut {_court(obtenu.classes, 80)} au lieu de {_court(attendu.classes, 80)}."
    for nom in ("log_a_priori", "log_proba"):
        dico = getattr(obtenu, nom)
        if not isinstance(dico, dict) or set(dico) != set(attendu.classes):
            return (f"{quoi}.{nom} vaut {_court(dico, 80)} : un dict qui a une clé par classe "
                    f"({_court(attendu.classes, 60)}).")
    for c in attendu.classes:
        if not _proche(obtenu.log_a_priori[c], attendu.log_a_priori[c]):
            return (f"{quoi}.log_a_priori[{c!r}] vaut {obtenu.log_a_priori[c]!r} au lieu de "
                    f"{attendu.log_a_priori[c]!r}.")
    for c in attendu.classes:
        dico, ref = obtenu.log_proba[c], attendu.log_proba[c]
        if not isinstance(dico, dict):
            return f"{quoi}.log_proba[{c!r}] vaut {_court(dico, 80)} : un dict {{mot: log-probabilité}}."
        manquants = sorted(set(ref) - set(dico))
        if manquants:
            return (f"Dans {quoi}.log_proba[{c!r}], il manque {_court(manquants[:5], 80)} ({len(manquants)} mot(s)) : "
                    "chaque classe a une log-probabilité pour TOUS les mots du vocabulaire, même ceux qu'elle n'a jamais "
                    "vus (c'est à ça que sert le lissage). Le vocabulaire, ce sont les mots de tous les documents.")
        en_trop = sorted(set(dico) - set(ref), key=repr)
        if en_trop:
            return (f"{quoi}.log_proba[{c!r}] contient {_court(en_trop[:5], 80)}, qui n'est dans aucun document : le "
                    "vocabulaire, ce sont les mots vus à l'entraînement, et seulement eux.")
        for mot, v in ref.items():
            if not _proche(dico[mot], v):
                return f"{quoi}.log_proba[{c!r}][{mot!r}] vaut {dico[mot]!r} au lieu de {v!r}."
    return ""


def _logs_positifs(modele) -> bool:
    """Le modèle contient-il un « logarithme de probabilité » strictement positif (donc impossible) ?"""
    valeurs = list(getattr(modele, "log_a_priori", {}).values()) if isinstance(getattr(modele, "log_a_priori", None),
                                                                                dict) else []
    log_proba = getattr(modele, "log_proba", None)
    if isinstance(log_proba, dict):
        valeurs += [v for d in log_proba.values() if isinstance(d, dict) for v in d.values()]
    return any(_nombre(v) and v > 0 for v in valeurs)


def _decrire_appel_entrainer(documents, etiquettes, kwargs) -> str:
    extra = "".join(f", {k}={v!r}" for k, v in kwargs.items())
    return f"entrainer({_court(documents, 70)}, {_court(etiquettes, 50)}{extra})"


def _corpus_mots(n: int, graine: int, classes=("part", "reste")) -> tuple[list, list]:
    """Des documents (avec négations) et des étiquettes, tirés de `generer_commentaires`, sur d'autres classes au
    besoin."""
    alea = random.Random(graine)
    documents, etiquettes = [], []
    for texte, etiquette in generer_commentaires(n, graine=graine):
        documents.append(_ref_negations(texte))
        etiquettes.append(etiquette if len(classes) == 2 else alea.choice(classes))
    return documents, etiquettes


def _cas_274() -> list[tuple]:
    """(documents, etiquettes, kwargs) : la dernière famille contient des classes sans aucun mot."""
    cas = [
        ([["prix", "cher", "prix"], ["merci"], ["prix", "merci"]], ["part", "reste", "reste"], {}),
        ([["prix", "cher", "prix"], ["merci"], ["prix", "merci"]], ["part", "reste", "reste"], {"alpha": 0.5}),
        ([["a", "a", "a", "b"], ["b", "c"], ["c", "c", "d", "e", "e"], ["a"], ["f"]],
         ["reste", "part", "reste", "inconnu", "part"], {"alpha": 2.5}),
        ((("rio", "rio", "rio", "rio"), ("merci",), ("merci", "top"), ("rio", "code")),
         ("part", "reste", "reste", "part"), {"alpha": 1}),
        ([["x"] * 6 + ["y"], ["y", "z"], ["z"]], ["b", "a", "a"], {"alpha": 0.01}),
        ([["un", "seul", "document"]], ["reste"], {}),
        ([["le", "prix", "le", "rio"], ["le", "prix", "le", "cher"], ["le", "merci", "le", "top"],
          ["le", "merci", "le", "prix"]], ["part", "part", "reste", "reste"], {"alpha": 3}),
    ]
    for n, graine, classes, alpha in ((300, 2741, ("part", "reste"), 1.0), (400, 2742, ("prix", "reseau", "service"), 0.3)):
        documents, etiquettes = _corpus_mots(n, graine, classes)
        cas.append((documents, etiquettes, {"alpha": alpha}))
    # des documents vides : une classe peut n'avoir aucun mot
    cas.append(([[], ["a"], [], ["a", "b"]], ["vide", "part", "vide", "reste"], {}))
    cas.append(([[], []], ["x", "x"], {}))
    return cas


def _gros_corpus(n_documents: int, taille_vocabulaire: int, graine: int, classes=("a", "b", "c")) -> tuple[list, list]:
    """Des documents de mots « m123 », de fréquences inégales (loi de Zipf), pour mesurer la vitesse."""
    alea = random.Random(graine)
    vocabulaire = [f"m{k}" for k in range(taille_vocabulaire)]
    cumul = list(itertools.accumulate(1 / (k + 1) for k in range(taille_vocabulaire)))   # calculé une seule fois
    documents = [alea.choices(vocabulaire, cum_weights=cumul, k=alea.randint(3, 20)) for _ in range(n_documents)]
    # chaque mot du vocabulaire apparaît au moins une fois
    for k in range(0, taille_vocabulaire, 20):
        documents.append(vocabulaire[k:k + 20])
    etiquettes = [alea.choice(classes) for _ in documents]
    return documents, etiquettes


_VOCABULAIRE_VIDE = ("Ici, tous les documents sont vides : le vocabulaire est VIDE, N(c) + alpha × 0 vaut 0, et "
                     "`math.log(0)` plante. Il n'y a aucun mot à qui donner une probabilité : log_proba[c] vaut {}. Ne "
                     "calculez (ou ne prenez le logarithme de) ce dénominateur que pour les mots du vocabulaire.")


def _un_entrainement(f, documents, etiquettes, kwargs):
    """Votre modèle pour un cas, ou un _Echec (exception, None, arguments modifiés)."""
    copies = (repr(documents), repr(etiquettes))
    quoi = _decrire_appel_entrainer(documents, etiquettes, kwargs)
    vide = not any(documents)
    try:
        r = _appel(f, (documents, etiquettes), quoi, kwargs=kwargs, au_lieu={ValueError: _VOCABULAIRE_VIDE} if vide else None)
    except AssertionError as exc:
        return _Echec(str(exc))
    if r is None:
        return _Echec(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if (repr(documents), repr(etiquettes)) != copies:
        return _Echec(f"{quoi} a MODIFIÉ ses arguments : `documents` et `etiquettes` doivent rester intacts.")
    return r


def verifier_entrainer(f) -> None:
    cas = _cas_274()
    sorties = [_un_entrainement(f, d, e, k) for d, e, k in cas]
    for (documents, etiquettes, kwargs), r in zip(cas, sorties):
        if isinstance(r, _Echec):
            raise AssertionError(r.message)
        ecart = _ecart_modeles(r, _ref_entrainer(documents, etiquettes, **kwargs),
                               _decrire_appel_entrainer(documents, etiquettes, kwargs))
        if ecart:
            conseil = _diagnostic(sorties, cas, lambda x, bug: _ref_entrainer(x[0], x[1], bug=bug, **x[2]),
                                  _BUGS_ENTRAINER, _modeles_egaux)
            if not conseil and _logs_positifs(r):
                conseil = "Un logarithme de probabilité est toujours <= 0 (une probabilité est <= 1)."
            raise AssertionError(f"{ecart} {conseil}".rstrip())

    # ------------------------------------------------------------------ erreurs
    for args, kwargs, attendu, conseil in (
            ((["je résilie"], ["part"]), {}, TypeError, "Un document est une LISTE de mots : passez `mots(texte)` (ou "
                                                         "`avec_negations(texte)`), pas le texte."),
            (([["je", 1]], ["part"]), {}, TypeError, "Un mot est un texte (str)."),
            (("je resilie", ["part"]), {}, TypeError, "`documents` est une liste de documents, pas un texte."),
            (([["a"], ["b"]], "pr"), {}, TypeError, "`etiquettes` est une liste d'étiquettes, pas un texte."),
            (([["a"]], [1]), {}, TypeError, "Une étiquette est un texte (str)."),
            (([["a"], ["b"]], ["part"]), {}, ValueError, "Deux documents, une étiquette : il en faut une par document."),
            (([], []), {}, ValueError, "Aucun document : il n'y a rien à apprendre."),
            (([["a"]], ["part"]), {"alpha": 0}, ValueError, "alpha = 0, c'est l'absence de lissage : un `math.log(0)` "
                                                            "qui attend son heure."),
            (([["a"]], ["part"]), {"alpha": -1}, ValueError, "Un alpha négatif n'a pas de sens."),
            (([["a"]], ["part"]), {"alpha": float("nan")}, ValueError, "NaN n'est pas un alpha."),
            (([["a"]], ["part"]), {"alpha": float("inf")}, ValueError, "Un alpha infini effacerait tous les comptes."),
            (([["a"]], ["part"]), {"alpha": True}, TypeError, "True n'est pas un alpha (`isinstance(True, int)` est "
                                                              "vrai : excluez bool)."),
            (([["a"]], ["part"]), {"alpha": "1"}, TypeError, "« 1 » est un texte.")):
        _doit_lever(f, args, _decrire_appel_entrainer(*args, kwargs), conseil, attendu, kwargs=kwargs)

    # ------------------------------------------------------------------ 30 000 documents, 20 000 mots
    documents, etiquettes = _gros_corpus(30_000, 20_000, 2743)
    jetons = sum(len(d) for d in documents)
    attendu = _ref_entrainer(documents, etiquettes)
    conseil = ("Le correcteur compte les lignes exécutées dans votre fichier : pour chaque mot du vocabulaire, "
               "reparcourir tous les documents (ou appeler `document.count(mot)`), c'est des centaines de millions "
               "d'étapes. Parcourez les documents UNE fois, en comptant avec un `Counter` par classe "
               "(`comptes[etiquette].update(document)`), puis calculez les log-probabilités mot par mot.")
    quoi = f"entrainer({len(documents)} documents, {len(attendu.log_proba['a'])} mots de vocabulaire)"
    r = _appel(f, (documents, etiquettes), quoi, lignes_max=8 * (jetons + 3 * 20_000) + 10_000, conseil_lent=conseil)
    ecart = _ecart_modeles(r, attendu, quoi)
    if ecart:
        raise AssertionError(f"{ecart} (alors que les petits cas passent).")


# ============================================================================ défi 275 : probabilites


def _ref_probabilites(modele, mots, bug: str = "") -> dict:
    """Les probabilités de référence ; `bug` rejoue une erreur classique."""
    if isinstance(mots, str) or not isinstance(mots, (list, tuple)):
        raise TypeError("mots est une liste de mots")
    if any(not isinstance(m, str) for m in mots):
        raise TypeError("un mot est un texte")
    vocabulaire = modele.log_proba[modele.classes[0]]
    connus = [m for m in (set(mots) if bug == "ensemble" else mots) if m in vocabulaire]
    scores = {}
    for c in modele.classes:
        somme = math.fsum(modele.log_proba[c][m] for m in connus)
        if bug == "moyenne":
            somme = somme / len(connus) if connus else 0.0
        scores[c] = somme + (0.0 if bug == "sans_a_priori" else modele.log_a_priori[c])
    if bug == "scores":
        return scores
    plus_grand = max(scores.values())
    exps = {c: math.exp(s - plus_grand) for c, s in scores.items()}
    if bug == "non_normalise":
        return exps
    total = sum(exps.values())
    return {c: e / total for c, e in exps.items()}


_BUGS_PROBABILITES = {
    "sans_a_priori": "Vous oubliez la probabilité a priori : le score de c commence à log_a_priori[c]. Sans elle, un "
                     "commentaire sans mot connu donnerait 50 % de départs, alors que 20 % des clients partent.",
    "ensemble": "Un mot répété ne compte qu'une fois chez vous (`set(mots)`) : « prix prix prix » compte trois fois "
                "log_proba[c][« prix »]. Parcourez `mots` tel quel.",
    "moyenne": "Vous faites la MOYENNE des log-probabilités des mots : c'est leur SOMME (le logarithme d'un produit). "
               "Un long commentaire plein de mots de départ est PLUS sûr qu'un court.",
    "scores": "Vous rendez les SCORES (des logarithmes, négatifs) : il faut des probabilités, entre 0 et 1, qui font 1 au "
              "total : exp(score - plus grand score), divisé par la somme de ces exponentielles.",
    "non_normalise": "Vos probabilités ne font pas 1 au total : divisez chaque exp(score - plus grand) par leur SOMME.",
}


def _modele_exemple() -> Modele:
    anciens = commentaires_exemple()
    return _ref_entrainer([_ref_negations(t) for t, _ in anciens], [e for _, e in anciens])


def _cas_275() -> list[tuple]:
    """(modele, mots) : les cas courts d'abord, puis les longs commentaires."""
    petit = _ref_entrainer([["prix", "cher", "prix"], ["merci"], ["prix", "merci"]], ["part", "reste", "reste"])
    trois = _ref_entrainer(*_corpus_mots(400, 2751, ("prix", "reseau", "service")), alpha=0.5)
    seul = _ref_entrainer([["a", "b"], ["b"]], ["reste", "reste"])
    vide = _ref_entrainer([[], []], ["part", "reste"])
    exemple = _modele_exemple()
    cas = [(petit, ["prix", "cher"]), (petit, ["prix", "inconnu", "prix"]), (petit, []), (petit, ("merci", "merci")),
           (petit, ["inconnu", "absent"]), (petit, ["cher"] * 3 + ["merci"]), (seul, ["a"]), (seul, ["z"]),
           (vide, ["a"]), (vide, [])]
    cas += [(trois, _ref_negations(t)) for t, _ in generer_commentaires(30, graine=2752)]
    cas += [(exemple, _ref_negations(t)) for t, _ in nouveaux_exemple()[:300]]
    cas += [(exemple, _ref_negations("Je ne vais pas résilier, le débit est excellent.") * 3),
            (exemple, ["rio", "merci", "rio", "inconnu"] * 5)]
    # les longs : un produit de probabilités, ou exp(score), tombent à 0.0
    long_texte = [m for t, _ in nouveaux_exemple()[:150] for m in _ref_negations(t)]
    cas += [(petit, ["prix"] * 1100), (exemple, long_texte), (exemple, ["rio"] * 3000), (trois, long_texte * 2)]
    return cas


def _probas_egales(a, b) -> bool:
    """Deux dicts {classe: nombre} égaux : mêmes classes, mêmes valeurs (des probabilités à 1e-7 près, et des
    nombres plus grands, comme les scores, à un milliardième près en valeur relative)."""
    return (isinstance(a, dict) and set(a) == set(b)
            and all(_proche_proba(a[k], b[k]) if abs(b[k]) <= 1 else _proche(a[k], b[k]) for k in b))


def _decrire_mots(mots) -> str:
    return _court(mots, 60) if len(mots) <= 12 else f"[… {len(mots)} mots …]"


def _une_prediction(f, modele, mots):
    """Vos probabilités pour un cas, ou un _Echec (exception, None, pas un dict de nombres sur les bonnes classes)."""
    quoi = f"probabilites(modele à {len(modele.classes)} classe(s), {_decrire_mots(mots)})"
    try:
        r = _appel(f, (modele, mots), quoi)
    except AssertionError as exc:
        return _Echec(str(exc))
    if r is None:
        return _Echec(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, dict):
        return _Echec(f"{quoi} renvoie {_court(r, 80)} ({type(r).__name__}) au lieu d'un dict {{classe: probabilité}}.")
    if set(r) != set(modele.classes):
        return _Echec(f"{quoi} a pour clés {_court(sorted(r, key=repr), 80)} au lieu des classes du modèle, "
                      f"{modele.classes!r}.")
    for c, p in r.items():
        if not _nombre(p) or math.isnan(p):
            return _Echec(f"{quoi}[{c!r}] vaut {p!r} : une probabilité est un nombre entre 0 et 1."
                          + (f" {_SOUS_DEPASSEMENT}" if _nombre(p) else ""))
    return r


def verifier_probabilites(f) -> None:
    cas = _cas_275()
    sorties = [_une_prediction(f, modele, mots) for modele, mots in cas]
    for (modele, mots), r in zip(cas, sorties):
        if isinstance(r, _Echec):
            raise AssertionError(r.message)
        quoi = f"probabilites(modele à {len(modele.classes)} classe(s), {_decrire_mots(mots)})"
        attendu = _ref_probabilites(modele, mots)
        if not all(_proche_proba(r[c], attendu[c]) for c in attendu):
            conseil = _diagnostic(sorties, cas, lambda x, bug: _ref_probabilites(x[0], x[1], bug), _BUGS_PROBABILITES,
                                  _probas_egales)
            if not conseil and len(mots) > 500:
                conseil = _SOUS_DEPASSEMENT.replace("Une division par zéro : sur", "Sur")
            if not conseil and not _proche(math.fsum(r.values()), 1.0):
                conseil = _BUGS_PROBABILITES["non_normalise"]
            c = next(k for k in attendu if not _proche_proba(r[k], attendu[k]))
            raise AssertionError(f"{quoi}[{c!r}] vaut {r[c]!r} au lieu de {attendu[c]!r}. {conseil}".rstrip())
        if list(r) != list(modele.classes):
            raise AssertionError(f"{quoi} range ses clés dans l'ordre {list(r)!r} : rangez-les dans l'ordre de "
                                 f"modele.classes, {modele.classes!r} (parcourez `modele.classes`).")

    petit = cas[0][0]
    for args, conseil in (((petit, "prix"), "Passez la LISTE des mots (`mots(texte)`), pas le texte : « prix » serait lu "
                                            "« p », « r », « i », « x »."),
                          ((petit, ["prix", 3]), "Un mot est un texte (str)."),
                          ((petit, None), "None n'est pas une liste de mots.")):
        _doit_lever(f, args, f"probabilites(modele, {args[1]!r})", conseil, TypeError)


# ============================================================================ défi 276 : mots_revelateurs


def _ref_revelateurs(modele, classe, k=10, bug: str = "") -> list[tuple[str, float]]:
    """Les mots révélateurs de référence ; `bug` rejoue une erreur classique."""
    autres = [c for c in modele.classes if c != classe]
    paires = []
    for mot, log_p in modele.log_proba[classe].items():
        if bug == "frequence":
            ecart = log_p
        elif bug == "moyenne":
            ecart = log_p - sum(modele.log_proba[c][mot] for c in autres) / len(autres)
        else:
            ecart = log_p - max(modele.log_proba[c][mot] for c in autres)
        if ecart > 0 or bug in ("sans_filtre", "frequence"):
            paires.append((mot, math.exp(ecart) if bug == "rapport" else ecart))
    if bug == "a_l_envers":
        paires.sort(key=lambda p: (p[1], p[0]))
    elif bug == "egalites_inverse":
        paires.sort(key=lambda p: (p[1], p[0]), reverse=True)
    elif bug == "egalites_ordre":
        paires.sort(key=lambda p: -p[1])
    else:
        paires.sort(key=lambda p: (-p[1], p[0]))
    return paires[:k]


_BUGS_REVELATEURS = {
    "frequence": "Vous classez les mots par leur probabilité DANS la classe : ce sont les mots fréquents partout (« le », "
                 "« de », « je »). Un mot révèle une classe s'il y est PLUS probable qu'ailleurs : l'écart log_proba[classe]"
                 "[mot] - max(log_proba[autre][mot]).",
    "moyenne": "Vous comparez la classe à la MOYENNE des autres : avec trois classes ou plus, un mot très fréquent dans "
               "une seule autre classe passerait pour révélateur. Comparez à la classe concurrente la plus forte : "
               "`max(modele.log_proba[c][mot] for c in autres)`.",
    "a_l_envers": "Vos mots sont rangés par écart CROISSANT : le plus révélateur (le plus grand écart) d'abord.",
    "egalites_inverse": "À écart égal, vos mots sont dans l'ordre alphabétique INVERSE : la clé de tri est (-écart, mot), "
                        "sans `reverse=True`.",
    "egalites_ordre": "À écart égal, vos mots restent dans l'ordre du dict `log_proba` : la réponse ne doit pas dépendre "
                      "de cet ordre. Départagez par ordre alphabétique : la clé de tri est (-écart, mot).",
    "sans_filtre": "Vous gardez des mots dont l'écart est négatif ou nul : un mot au moins aussi probable ailleurs ne "
                   "révèle rien. Ne gardez que les écarts strictement positifs (`if ecart > 0`).",
    "rapport": "Vous rendez le RAPPORT des probabilités (exp de l'écart) : rendez l'écart des LOGARITHMES, "
               "log_proba[classe][mot] - max(...), sans `math.exp`.",
}


def _modele_melange(graine: int) -> Modele:
    """Un modèle dont le vocabulaire est rangé dans le désordre, avec des écarts à égalité."""
    documents = [["le", "prix", "le", "rio"], ["le", "prix", "le", "cher"], ["le", "merci", "le", "top"],
                 ["le", "merci", "le", "prix"], ["zut", "abus", "abus"], ["zut", "abus"]]
    modele = _ref_entrainer(documents, ["part", "part", "reste", "reste", "part", "part"])
    alea = random.Random(graine)
    ordre = list(modele.log_proba["part"])
    alea.shuffle(ordre)
    if ordre == sorted(ordre):
        ordre.reverse()
    return Modele(classes=modele.classes, log_a_priori=modele.log_a_priori,
                  log_proba={c: {m: modele.log_proba[c][m] for m in ordre} for c in modele.classes})


def _cas_276() -> list[tuple]:
    """(modele, classe, k)."""
    docstring = _ref_entrainer([["le", "prix", "le", "rio"], ["le", "prix", "le", "cher"], ["le", "merci", "le", "top"],
                                ["le", "merci", "le", "prix"]], ["part", "part", "reste", "reste"])
    melange = _modele_melange(2761)
    # trois classes : « box » est très fréquent dans « reseau », un peu dans « prix », rare dans « service »
    trois = _ref_entrainer([["box", "box", "box", "coupure"], ["box", "tarif", "tarif"], ["box", "tarif"],
                            ["attente", "conseiller"], ["attente", "box"]],
                           ["reseau", "prix", "prix", "service", "service"])
    pareils = _ref_entrainer([["a", "b"], ["a", "b"]], ["part", "reste"])
    corpus3 = _ref_entrainer(*_corpus_mots(400, 2762, ("prix", "reseau", "service")), alpha=0.5)
    exemple = _modele_exemple()
    return [(docstring, "part", 10), (docstring, "reste", 1), (melange, "part", 10), (melange, "reste", 10),
            (melange, "part", 2), (trois, "prix", 10), (trois, "reseau", 10), (trois, "service", 3),
            (pareils, "part", 10), (docstring, "part", 0), (corpus3, "prix", 15), (corpus3, "service", 15),
            (exemple, "part", 10), (exemple, "reste", 10), (exemple, "part", 1000)]


def _des_revelateurs(f, modele, classe, k):
    """Vos mots révélateurs pour un cas, ou un _Echec (exception, None, pas une liste de tuples (mot, écart))."""
    quoi = f"mots_revelateurs(modele à {len(modele.log_proba[classe])} mots, {classe!r}, k={k})"
    try:
        r = _appel(f, (modele, classe), quoi, kwargs={"k": k})
    except AssertionError as exc:
        return _Echec(str(exc))
    if r is None:
        return _Echec(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(r, list):
        return _Echec(f"{quoi} renvoie {_court(r, 80)} ({type(r).__name__}) au lieu d'une liste de tuples (mot, écart).")
    for paire in r:
        if not isinstance(paire, tuple) or len(paire) != 2 or not isinstance(paire[0], str) or not _nombre(paire[1]):
            return _Echec(f"{quoi} contient {_court(paire, 60)} : chaque élément est un tuple (mot, écart), un str et "
                          "un float.")
    return r


def _paires_egales(a, b) -> bool:
    return len(a) == len(b) and all(x[0] == y[0] and _proche(x[1], y[1]) for x, y in zip(a, b))


def verifier_mots_revelateurs(f) -> None:
    cas = _cas_276()
    sorties = [_des_revelateurs(f, modele, classe, k) for modele, classe, k in cas]
    for (modele, classe, k), r in zip(cas, sorties):
        if isinstance(r, _Echec):
            raise AssertionError(r.message)
        quoi = f"mots_revelateurs(modele à {len(modele.log_proba[classe])} mots, {classe!r}, k={k})"
        attendu = _ref_revelateurs(modele, classe, k)
        if not _paires_egales(r, attendu):
            conseil = _diagnostic(sorties, cas, lambda x, bug: _ref_revelateurs(x[0], x[1], x[2], bug), _BUGS_REVELATEURS,
                                  _paires_egales)
            if not conseil and r and all(_proche(e, modele.log_proba[classe].get(m, math.nan)) for m, e in r):
                conseil = _BUGS_REVELATEURS["frequence"]           # des log-probabilités, pas des écarts
            elif not conseil and any(e <= 0 for _, e in r):
                conseil = _BUGS_REVELATEURS["sans_filtre"]
            raise AssertionError(f"{quoi} renvoie {_court(r, 200)} au lieu de {_court(attendu, 200)}. {conseil}".rstrip())

    docstring = cas[0][0]
    seul = _ref_entrainer([["a"]], ["reste"])
    for args, kwargs, attendu, conseil in (
            ((docstring, "inconnue"), {}, ValueError, "« inconnue » n'est pas une classe du modèle."),
            ((seul, "reste"), {}, ValueError, "Une seule classe : il n'y a rien à comparer."),
            ((docstring, "part"), {"k": -1}, ValueError, "k négatif n'a pas de sens."),
            ((docstring, "part"), {"k": True}, TypeError, "True n'est pas un nombre de mots."),
            ((docstring, "part"), {"k": 2.0}, TypeError, "k est un int."),
            ((docstring, "part"), {"k": "3"}, TypeError, "« 3 » est un texte.")):
        _doit_lever(f, args, f"mots_revelateurs(modele, {args[1]!r}{''.join(f', {a}={b!r}' for a, b in kwargs.items())})",
                    conseil, attendu, kwargs=kwargs)


# ============================================================================ défi 277 : ClassifieurEnLigne


def _verifier_mots(mots) -> None:
    if isinstance(mots, str) or not isinstance(mots, (list, tuple)):
        raise TypeError("mots est une liste de mots")
    if any(not isinstance(m, str) for m in mots):
        raise TypeError("un mot est un texte")


class _RefClassifieur:
    """Le classifieur de référence (des comptes tenus à jour) ; `bug` rejoue une erreur classique."""

    def __init__(self, alpha=1.0, bug: str = "") -> None:
        if not _nombre(alpha):
            raise TypeError("alpha est un nombre")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha > 0")
        self.alpha, self.bug = alpha, bug
        self.documents, self.comptes, self.totaux, self.vocabulaire = Counter(), {}, Counter(), Counter()
        self.cache = None

    def apprendre(self, mots, etiquette) -> None:
        _verifier_mots(mots)
        if not isinstance(etiquette, str):
            raise TypeError("une étiquette est un texte")
        self.documents[etiquette] += 1
        self.comptes.setdefault(etiquette, Counter()).update(mots)
        self.totaux[etiquette] += len(mots)
        self.vocabulaire.update(mots)

    def _retirer_mot(self, comptes: Counter, mot: str, n: int) -> None:
        comptes[mot] -= n
        if comptes[mot] == 0:
            del comptes[mot]
        self.vocabulaire[mot] -= n
        if self.vocabulaire[mot] == 0 and self.bug != "vocabulaire_garde":
            del self.vocabulaire[mot]

    def oublier(self, mots, etiquette) -> None:
        _verifier_mots(mots)
        if not isinstance(etiquette, str):
            raise TypeError("une étiquette est un texte")
        a_retirer = Counter(mots)
        if self.bug == "sans_verification":
            comptes = self.comptes.setdefault(etiquette, Counter())
            for mot, n in a_retirer.items():
                comptes[mot] -= n
                self.vocabulaire[mot] -= n
        else:
            comptes = self.comptes.get(etiquette)
            if comptes is None:
                raise ValueError("étiquette inconnue")
            if self.bug == "non_atomique":
                for mot, n in a_retirer.items():
                    if comptes[mot] < n:
                        raise ValueError("pas appris")
                    self._retirer_mot(comptes, mot, n)
            else:
                if any(comptes[mot] < n for mot, n in a_retirer.items()):
                    raise ValueError("pas appris")
                for mot, n in a_retirer.items():
                    self._retirer_mot(comptes, mot, n)
        if self.bug != "totaux_garde":
            self.totaux[etiquette] -= len(mots)
        self.documents[etiquette] -= 1
        if self.documents[etiquette] <= 0 and self.bug not in ("classe_gardee", "sans_verification"):
            del self.documents[etiquette], self.comptes[etiquette], self.totaux[etiquette]

    def _modele(self) -> Modele:
        if not self.documents:
            raise ValueError("rien n'a été appris")
        n = sum(self.documents.values())
        classes, vocabulaire = sorted(self.documents), sorted(self.vocabulaire)
        log_proba = {}
        for c in classes:
            denominateur = self.totaux[c] + self.alpha * len(vocabulaire)
            log_proba[c] = {m: math.log((self.comptes[c][m] + self.alpha) / denominateur) for m in vocabulaire}
        return Modele(classes=classes, log_a_priori={c: math.log(self.documents[c] / n) for c in classes},
                      log_proba=log_proba)

    def probabilites(self, mots) -> dict:
        _verifier_mots(mots)
        if self.bug == "cache":
            if self.cache is None:
                self.cache = self._modele()
            return _ref_probabilites(self.cache, mots)
        if not self.documents:
            raise ValueError("rien n'a été appris")
        connus = [m for m in mots if m in self.vocabulaire]
        n, v = sum(self.documents.values()), len(self.vocabulaire)
        scores = {}
        for c in sorted(self.documents):
            denominateur = self.totaux[c] + self.alpha * v
            scores[c] = math.log(self.documents[c] / n) + math.fsum(
                math.log((self.comptes[c][m] + self.alpha) / denominateur) for m in connus)
        plus_grand = max(scores.values())
        exps = {c: math.exp(s - plus_grand) for c, s in scores.items()}
        total = sum(exps.values())
        return {c: e / total for c, e in exps.items()}

    def modele(self) -> Modele:
        return self._modele()

    def __len__(self) -> int:
        return sum(self.totaux.values()) if self.bug == "len_mots" else sum(self.documents.values())


_BUGS_CLASSIFIEUR = {
    "vocabulaire_garde": "Un mot que plus aucun document ne contient reste dans votre vocabulaire : après l'oubli, il "
                         "compte encore dans la taille du vocabulaire (au dénominateur) et apparaît dans `modele()`. "
                         "Quand son compte (toutes classes) tombe à zéro, retirez-le (`del self._vocabulaire[mot]`).",
    "classe_gardee": "Une classe dont le dernier document est oublié reste chez vous, avec 0 document : log(0 / n) pour "
                     "sa probabilité a priori. Quand une classe n'a plus de document, elle disparaît (ses comptes "
                     "aussi).",
    "non_atomique": "Un `oublier` refusé a quand même modifié vos comptes : vous retirez les mots un par un, et vous "
                    "découvrez en cours de route qu'un mot n'avait pas été appris. Vérifiez TOUS les mots d'abord "
                    "(`any(comptes[mot] < n for mot, n in a_retirer.items())`), et ne modifiez rien si l'un manque.",
    "sans_verification": "Votre `oublier` accepte d'oublier ce qui n'a jamais été appris : les comptes deviennent "
                         "négatifs. Si l'étiquette n'a aucun document, ou si un mot apparaît plus de fois que dans les "
                         "comptes de cette étiquette : ValueError, et rien ne change.",
    "totaux_garde": "Après un oubli, le nombre total de mots de la classe, N(c), n'a pas diminué : retirez `len(mots)` de "
                    "son total, comme vous retirez chaque mot de ses comptes.",
    "cache": "Vos probabilités ne bougent plus après le premier appel : vous gardez un modèle calculé une fois (un "
             "« cache ») sans le mettre à jour quand on apprend ou qu'on oublie. Calculez à partir des comptes ACTUELS.",
    "len_mots": "`len(clf)` compte des MOTS : c'est le nombre de DOCUMENTS appris (et pas oubliés).",
}


def _jouer(clf, operations) -> list:
    """Joue les opérations ; une sortie par opération (ce que renvoie la méthode, ou l'exception levée)."""
    sorties = []
    for op in operations:
        try:
            if op[0] == "len":
                sorties.append(len(clf))
            elif op[0] == "modele":
                sorties.append(copy.deepcopy(clf.modele()))      # une copie : le défaut « vue » a son propre test
            else:
                sorties.append(copy.deepcopy(getattr(clf, op[0])(*op[1:])))
        except NotImplementedError:
            raise
        except _BoucleSansFin:
            raise
        except Exception as exc:
            sorties.append(("!", type(exc).__name__, _decrire(exc)))
    return sorties


def _sorties_egales(a, b) -> bool:
    if isinstance(b, tuple) and b[:1] == ("!",):
        return isinstance(a, tuple) and a[:2] == b[:2]
    if isinstance(b, Modele):
        return _modeles_egaux(a, b)
    if isinstance(b, dict):
        return _probas_egales(a, b)
    return type(a) is type(b) and a == b


def _scenarios_277() -> list[tuple]:
    """(alpha, opérations) ; alpha None : le classifieur est créé SANS argument."""
    exemple = (None, [("apprendre", ["prix", "cher", "prix"], "part"), ("apprendre", ["merci"], "reste"),
                      ("apprendre", ["prix", "merci"], "reste"), ("probabilites", ["prix", "cher"]),
                      ("oublier", ["prix", "cher", "prix"], "part"), ("len",), ("modele",),
                      ("oublier", ["cher"], "reste"), ("probabilites", ["prix", "cher"])])
    effacement = (1.0, [("apprendre", ["a", "b"], "x"), ("apprendre", ["a", "c"], "y"), ("apprendre", ["d"], "y"),
                        ("probabilites", ["a", "d"]), ("oublier", ["d"], "y"), ("probabilites", ["a", "d"]),
                        ("probabilites", ["c", "c", "b"]), ("modele",), ("len",)])
    disparition = (1.0, [("apprendre", ["a"], "x"), ("apprendre", ["b"], "y"), ("oublier", ["b"], "y"),
                         ("probabilites", ["a"]), ("modele",), ("len",), ("oublier", ["a"], "x"), ("len",),
                         ("probabilites", ["a"]), ("modele",), ("apprendre", ("a", "a"), "z"), ("probabilites", ["a"]),
                         ("modele",)])
    refus = (2.0, [("apprendre", ["a", "a", "b"], "x"), ("apprendre", ["c"], "y"), ("oublier", ["a", "z"], "x"),
                   ("oublier", ["a", "a", "a"], "x"), ("oublier", ["c"], "x"), ("oublier", ["a"], "w"),
                   ("probabilites", ["a", "b", "c"]), ("modele",), ("len",), ("oublier", ["b", "a", "a"], "x"),
                   ("modele",)])
    cache = (0.5, [("apprendre", ["a"], "x"), ("apprendre", ["b"], "y"), ("probabilites", ["a"]),
                   ("apprendre", ["a", "a"], "y"), ("probabilites", ["a"]), ("oublier", ["b"], "y"),
                   ("probabilites", ["a", "b"]), ("len",)])
    vide = (1.0, [("len",), ("probabilites", []), ("modele",), ("oublier", [], "x"), ("apprendre", [], "x"),
                  ("probabilites", ["a"]), ("modele",), ("len",)])
    scenarios = [exemple, effacement, disparition, refus, cache, vide]
    for graine, alpha in ((2771, 1.0), (2772, 0.5), (2773, 2.5)):
        alea = random.Random(graine)
        mots_possibles = [f"m{k}" for k in range(25)]
        ops, appris = [], []
        for _ in range(300):
            tirage = alea.random()
            if tirage < 0.45 or not appris:
                doc = alea.choices(mots_possibles, k=alea.randint(0, 6))
                etiquette = alea.choice(["part", "reste", "doute"])
                ops.append(("apprendre", doc, etiquette))
                appris.append((doc, etiquette))
            elif tirage < 0.65:
                doc, etiquette = appris.pop(alea.randrange(len(appris)))
                melange = list(doc)
                alea.shuffle(melange)
                ops.append(("oublier", melange, etiquette))
            elif tirage < 0.7:                      # jamais appris : toujours refusé
                ops.append(("oublier", alea.choices(mots_possibles, k=2) + ["jamais_vu"],
                            alea.choice(["part", "reste", "doute"])))
            elif tirage < 0.72:
                ops.append(("oublier", [], "inconnue"))
            elif tirage < 0.9:
                ops.append(("probabilites", alea.choices(mots_possibles + ["inconnu"], k=alea.randint(0, 8))))
            elif tirage < 0.95:
                ops.append(("modele",))
            else:
                ops.append(("len",))
        scenarios.append((alpha, ops))
    return scenarios


def _decrire_ops(operations, jusqu_a: int) -> str:
    morceaux = []
    for op in operations[max(0, jusqu_a - 4): jusqu_a + 1]:
        if op[0] in ("len", "modele"):
            morceaux.append("len(clf)" if op[0] == "len" else "clf.modele()")
        else:
            morceaux.append(f"clf.{op[0]}({', '.join(_court(x, 50) for x in op[1:])})")
    return ("… " if jusqu_a > 4 else "") + " ; ".join(morceaux)


def _creer(classe, alpha):
    return _appel(classe, () if alpha is None else (alpha,),
                  "ClassifieurEnLigne()" if alpha is None else f"ClassifieurEnLigne({alpha!r})")


def _decrire_sortie(s) -> str:
    """Ce que VOTRE code a fait, pour une opération."""
    if isinstance(s, tuple) and s[:1] == ("!",):
        return f"lève {s[2]}"
    if isinstance(s, Modele) and s.classes and isinstance(s.log_proba, dict) and s.classes[0] in s.log_proba:
        return f"renvoie un Modele de classes {s.classes!r} et {len(s.log_proba[s.classes[0]])} mot(s)"
    return f"renvoie {_court(s, 120)}"


def _decrire_attendu(s) -> str:
    """Ce que la référence fait, pour la même opération."""
    if isinstance(s, tuple) and s[:1] == ("!",):
        return f"lever {s[1]}"
    if isinstance(s, Modele):
        return f"renvoyer un Modele de classes {s.classes!r} et {len(s.log_proba[s.classes[0]])} mot(s)"
    return f"renvoyer {_court(s, 120)}"


def _ref_sorties(alpha, operations, bug: str = "") -> list:
    return _jouer(_RefClassifieur(1.0 if alpha is None else alpha, bug), operations)


def verifier_classifieur(classe) -> None:
    with _Pretes(classe, {"mots": _ref_mots, "avec_negations": _ref_negations, "entrainer": _ref_entrainer,
                          "probabilites": _ref_probabilites}):
        _verifier_classifieur(classe)


def _verifier_classifieur(classe) -> None:
    # ------------------------------------------------------------------ deux classifieurs ne partagent rien
    a, b = _creer(classe, 1.0), _creer(classe, 1.0)
    _appel(a.apprendre, (["prix"], "part"), "a.apprendre(['prix'], 'part')")
    if _appel(len, (b,), "len(b)") != 0:
        raise AssertionError("Deux classifieurs partagent leurs comptes : après a.apprendre(['prix'], 'part'), un AUTRE "
                             "classifieur b n'est plus vide. Créez les comptes dans __init__ (pas comme attributs de la "
                             "classe, et pas comme valeur par défaut d'un argument).")

    # ------------------------------------------------------------------ les scénarios
    scenarios = _scenarios_277()
    obtenus = [_jouer(_creer(classe, alpha), ops) for alpha, ops in scenarios]
    attendus = [_ref_sorties(alpha, ops) for alpha, ops in scenarios]
    for (alpha, operations), o_tout, a_tout in zip(scenarios, obtenus, attendus):
        i = next((k for k, (o, a) in enumerate(zip(o_tout, a_tout)) if not _sorties_egales(o, a)), None)
        if i is None:
            continue
        nom = "ClassifieurEnLigne()" if alpha is None else f"ClassifieurEnLigne({alpha!r})"
        contexte = f"Avec clf = {nom} : {_decrire_ops(operations, i)}"
        o, a = o_tout[i], a_tout[i]
        conseil = _premier_bug(obtenus, lambda bug: [_ref_sorties(x, ops, bug) for x, ops in scenarios],
                               _BUGS_CLASSIFIEUR, _listes(_listes(_sorties_egales)))
        if not conseil and isinstance(o, tuple) and o[:1] == ("!",):
            conseil = next((texte for genre, texte in ((ValueError, _LOG_ZERO), (ZeroDivisionError, _SOUS_DEPASSEMENT),
                                                       (OverflowError, _DEBORDEMENT), (KeyError, _MOT_INCONNU))
                            if o[1] == genre.__name__ and (genre is not ValueError or "math domain" in o[2])
                            and (genre is not KeyError or operations[i][0] == "probabilites")), "")
            if not conseil and o[1] == "KeyError":
                conseil = ("Une classe (ou un mot) manque dans un de vos dicts : une classe qui n'a appris que des "
                           "documents VIDES n'a aucun mot, mais elle existe ; utilisez `.get(cle, valeur)`, "
                           "`.pop(cle, None)` ou un `Counter`.")
        if not conseil and isinstance(a, Modele) and isinstance(o, Modele):
            conseil = _ecart_modeles(o, a, "clf.modele()")
        raise AssertionError(f"{contexte} : la dernière opération {_decrire_sortie(o)} au lieu de "
                             f"{_decrire_attendu(a)}. {conseil}".rstrip())

    # ------------------------------------------------------------------ une photographie, pas une vue
    clf = _creer(classe, 1.0)
    _appel(clf.apprendre, (["prix", "cher"], "part"), "clf.apprendre(['prix', 'cher'], 'part')")
    _appel(clf.apprendre, (["merci"], "reste"), "clf.apprendre(['merci'], 'reste')")
    photo = _appel(clf.modele, (), "clf.modele()")
    attendu = _RefClassifieur(1.0)
    attendu.apprendre(["prix", "cher"], "part")
    attendu.apprendre(["merci"], "reste")
    attendu = attendu.modele()
    _appel(clf.apprendre, (["prix", "rio", "rio"], "part"), "clf.apprendre(['prix', 'rio', 'rio'], 'part')")
    _appel(clf.oublier, (["merci"], "reste"), "clf.oublier(['merci'], 'reste')")
    if not _modeles_egaux(photo, attendu):
        raise AssertionError("Un modèle obtenu par clf.modele() a changé APRÈS coup, quand on a appris et oublié d'autres "
                             "documents : `modele()` rend une photographie. Construisez de NOUVEAUX dicts (et une "
                             "nouvelle liste de classes), au lieu de rendre vos comptes ou des vues sur eux.")

    # ------------------------------------------------------------------ erreurs (et une opération refusée ne change rien)
    for args, attendu_exc, conseil in (
            ((0,), ValueError, "alpha = 0, c'est l'absence de lissage."),
            ((-1,), ValueError, "Un alpha négatif n'a pas de sens."),
            ((float("nan"),), ValueError, "NaN n'est pas un alpha."),
            ((float("inf"),), ValueError, "Un alpha infini effacerait tous les comptes."),
            ((True,), TypeError, "True n'est pas un alpha."),
            (("1",), TypeError, "« 1 » est un texte.")):
        _doit_lever(classe, args, f"ClassifieurEnLigne{_court(args, 30)}", conseil, attendu_exc)
    clf = _creer(classe, 1.0)
    _doit_lever(clf.probabilites, (["a"],), "clf.probabilites(['a']) sur un classifieur vide",
                "Rien n'a été appris : ValueError.")
    _doit_lever(clf.modele, (), "clf.modele() sur un classifieur vide", "Rien n'a été appris : ValueError.")
    _appel(clf.apprendre, (["a", "b"], "x"), "clf.apprendre(['a', 'b'], 'x')")
    for methode, args, attendu_exc, conseil in (
            ("apprendre", ("je résilie", "x"), TypeError, "Passez la LISTE des mots, pas le texte."),
            ("apprendre", (["a", 1], "x"), TypeError, "Un mot est un texte (str)."),
            ("apprendre", (["a"], 1), TypeError, "Une étiquette est un texte (str)."),
            ("oublier", ("ab", "x"), TypeError, "Passez la LISTE des mots, pas le texte."),
            ("oublier", (["a"], None), TypeError, "Une étiquette est un texte (str)."),
            ("probabilites", ("ab",), TypeError, "Passez la LISTE des mots, pas le texte."),
            ("probabilites", ([None],), TypeError, "Un mot est un texte (str).")):
        _doit_lever(getattr(clf, methode), args, f"clf.{methode}{_court(args, 40)}", conseil, attendu_exc)
    if _appel(len, (clf,), "len(clf)") != 1 or not _sorties_egales(
            _appel(clf.modele, (), "clf.modele()"), _ref_entrainer([["a", "b"]], ["x"])):
        raise AssertionError("Après des opérations refusées (TypeError), le classifieur a changé : une opération refusée "
                             "ne change rien. Vérifiez les arguments AVANT de modifier vos comptes.")

    # ------------------------------------------------------------------ 8 000 documents, 2 000 oublis, 4 000 prédictions
    alea = random.Random(2779)
    documents, etiquettes = _gros_corpus(8_000, 20_000, 2778)
    operations, appris = [], []
    for k, (document, etiquette) in enumerate(zip(documents, etiquettes)):
        operations.append(("apprendre", document, etiquette))
        appris.append((document, etiquette))
        if k % 4 == 3:
            document, etiquette = appris.pop(alea.randrange(len(appris)))
            operations.append(("oublier", document, etiquette))
        if k % 2 == 1:
            operations.append(("probabilites", alea.choices(document + ["inconnu", f"m{alea.randrange(30_000)}"], k=10)))
    operations += [("len",), ("modele",)]
    jetons = sum(len(op[1]) for op in operations if op[0] != "len" and op[0] != "modele")
    attendus = _jouer(_RefClassifieur(1.0), operations)
    gros = _creer(classe, 1.0)
    conseil = ("Le correcteur compte les lignes exécutées dans votre fichier : recalculer les log-probabilités de TOUT le "
               "vocabulaire (ou appeler `entrainer`) à chaque prédiction, ou parcourir tous les documents appris à chaque "
               "oubli, c'est des centaines de millions d'étapes. Gardez les comptes à jour (documents par classe, "
               "occurrences de chaque mot par classe, total par classe, occurrences de chaque mot toutes classes), et ne "
               "calculez que les log-probabilités des mots demandés.")
    obtenus = _appel(_jouer, (gros, operations[:-1]), f"{len(operations) - 1} opérations sur 8 000 documents",
                     lignes_max=25 * jetons + 40 * len(operations), conseil_lent=conseil, fichier=_fichier(classe))
    obtenus.append(_appel(gros.modele, (), "clf.modele() après 8 000 documents"))
    i = next((k for k, (o, a) in enumerate(zip(obtenus, attendus)) if not _sorties_egales(o, a)), None)
    if i is not None:
        raise AssertionError(f"Sur {len(operations)} opérations (8 000 documents), l'opération {i} "
                             f"({_decrire_ops(operations, i).split(' ; ')[-1]}) {_decrire_sortie(obtenus[i])} au lieu de "
                             f"{_decrire_attendu(attendus[i])} (alors que les petits scénarios passent).")


# ---------------------------------------------------------------------------- la série

SERIE = Serie(
    code="45",
    titre="Lire ce que disent les clients : mots, négations, Bayes naïf, probabilités, mots révélateurs, en continu",
    module_eleve="defis.serie_45",
    module_corrige="defis.corrige_serie_45",
    defis=(
        Defi(272, "mots", "🟢", "—", verifier_mots),
        Defi(273, "avec_negations", "🟡", "—", verifier_avec_negations),
        Defi(274, "entrainer", "🟡", "—", verifier_entrainer),
        Defi(275, "probabilites", "🟡", "—", verifier_probabilites),
        Defi(276, "mots_revelateurs", "🟡", "—", verifier_mots_revelateurs),
        Defi(277, "ClassifieurEnLigne", "🔴", "—", verifier_classifieur),
    ),
)

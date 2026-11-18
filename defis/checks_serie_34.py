"""Vérifications automatiques de la série 34 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est Inès, la déléguée à la protection des données, et l'attaquant qu'elle imagine toujours :

- au défi 206, il retourne vos pseudonymes (un SHA-256 sans clé se retrouve en essayant tous les identifiants),
  ajoute au CRM une colonne que personne n'a demandée, écrit le même client de deux façons, et change la clé ;
- au défi 207, il compte les lignes exécutées dans votre fichier sur 10 000 lignes face à 10 000 personnes : une
  double boucle se voit, quelle que soit la machine ;
- aux défis 207 et 208, il glisse des valeurs manquantes sous toutes leurs formes (None, NaN, colonne absente) et
  des homonymes parfaits ;
- au défi 210, il recalcule par soustraction toute case cachée seule dans sa ligne ou dans sa colonne ;
- au défi 211, il vous donne une source de hasard qui compte ses tirages : il rejoue chaque bruit, et voit une
  question refusée qui tire ou qui dépense ;
- partout, il rejoue les erreurs classiques pour nommer la vôtre, et vérifie que vos fonctions ne modifient pas ce
  qu'elles reçoivent.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import itertools
import math
import random
import sys
from collections import Counter

from .confidentialite import (
    CLE_EXEMPLE,
    COLONNES_ETUDE,
    HIERARCHIES,
    IDENTIFIANTS_DIRECTS,
    QUASI_IDENTIFIANTS,
    BudgetEpuise,
    annuaire_exemple,
    appliquer,
    clients_exemple,
    comptages_exemple,
    exact,
    generer_annuaire,
    generer_clients,
    masquer,
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
    """Levée dans VOTRE fonction quand elle travaille trop (BaseException : `except Exception` ne l'attrape pas)."""


def _trace(f, appel, lignes_max: int | None):
    """Le résultat de `appel()` ; arrête net au-delà de `lignes_max` lignes exécutées dans le fichier de f."""
    code = getattr(f, "__code__", None) or getattr(getattr(f, "__init__", None), "__code__", None)
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


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = 3_000_000, conseil_lent: str = "",
           kwargs: dict | None = None):
    """f(*args, **kwargs), en transformant toute erreur en message lisible."""
    kwargs = kwargs or {}
    try:
        return _trace(f, lambda: f(*args, **kwargs), lignes_max)
    except NotImplementedError as exc:
        _pas_encore(f, exc)
        raise
    except _BoucleSansFin:
        raise AssertionError(f"{quoi} est beaucoup trop lent. {conseil_lent or _CONSEIL_LENT}") from None
    except RecursionError:
        raise AssertionError(f"{quoi} lève RecursionError : remplacez la récursion par une boucle. {conseil}".rstrip()) \
            from None
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
        resultat = _trace(f, lambda: f(*args, **(kwargs or {})), 3_000_000)
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


def _egaux_nan(a, b) -> bool:
    """a == b, en considérant deux NaN comme égaux (dans des listes, tuples et dictionnaires aussi)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_egaux_nan(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_egaux_nan(a[c], b[c]) for c in a)
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


def _intact(avant, apres, quoi: str, nom: str) -> None:
    if not _egaux_nan(avant, apres):
        raise AssertionError(f"{quoi} a modifié {nom} reçu(s) (avant : {_court(avant, 100)}, après : "
                             f"{_court(apres, 100)}). Une fonction ne touche pas à ce qu'on lui prête : travaillez sur "
                             "une copie, ou construisez de nouveaux objets.")


def _pas_none(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")


def _nombre(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _proche(x, y, tol: float = 1e-9) -> bool:
    return _nombre(x) and _nombre(y) and math.isclose(x, y, rel_tol=tol, abs_tol=tol)


def _manquant(x) -> bool:
    return x is None or (isinstance(x, float) and x != x)


def _valeur(x):
    return None if _manquant(x) else x


def _nan() -> float:
    """Un NaN NEUF : deux NaN distincts ne sont égaux ni par ==, ni par `is` (comme ceux d'un fichier lu)."""
    return float("nan") * 1.0


# ---------------------------------------------------------------------------- défi 206 : préparer l'export


def _normaliser(identifiant: str) -> str:
    return identifiant.strip().upper()


def _hmac(cle: bytes, texte: str, algo=hashlib.sha256, codage: str = "utf-8") -> str:
    return hmac.new(cle, texte.encode(codage), algo).hexdigest()


def _ref_pseudo(cle: bytes, identifiant: str) -> str:
    return _hmac(cle, _normaliser(identifiant))


_ATTAQUE = ("il n'existe qu'un million d'identifiants « C » + 6 chiffres : un attaquant les essaie tous en une seconde "
            "et retrouve tout le monde")

_VARIANTES_PSEUDO = [
    (lambda cle, i: hashlib.sha256(_normaliser(i).encode("utf-8")).hexdigest(),
     "c'est sha256({n!r}), SANS clé : " + _ATTAQUE + ". Utilisez `hmac.new(cle, …, hashlib.sha256)`."),
    (lambda cle, i: hashlib.sha256(i.encode("utf-8")).hexdigest(),
     "c'est sha256({i!r}), SANS clé (et sans normaliser) : " + _ATTAQUE + ". Utilisez `hmac.new(cle, …, hashlib.sha256)`."),
    (lambda cle, i: hashlib.sha256(cle + _normaliser(i).encode("utf-8")).hexdigest(),
     "c'est sha256(cle + identifiant) : une construction « maison », pas un HMAC (elle permet l'attaque par extension "
     "de longueur, relisez le défi 85). Utilisez `hmac.new(cle, identifiant, hashlib.sha256)`."),
    (lambda cle, i: hashlib.sha256(_normaliser(i).encode("utf-8") + cle).hexdigest(),
     "c'est sha256(identifiant + cle) : une construction « maison », pas un HMAC. Utilisez "
     "`hmac.new(cle, identifiant, hashlib.sha256)`, qui a été conçu (et prouvé) pour ça."),
    (lambda cle, i: _hmac(CLE_EXEMPLE, _normaliser(i)),
     "il est calculé avec CLE_EXEMPLE au lieu de la clé reçue en argument : chaque destinataire a sa propre clé."),
    (lambda cle, i: _hmac(cle, i),
     "c'est le HMAC de l'identifiant tel quel, {i!r}. Normalisez-le d'abord (`.strip().upper()`) : sinon le même "
     "client, écrit « c00412 » par le support et « C00412 » par le CRM, reçoit deux pseudonymes."),
    (lambda cle, i: _hmac(cle, i.strip()),
     "l'identifiant {i!r} est débarrassé de ses espaces mais pas mis en MAJUSCULES : `.strip().upper()`."),
    (lambda cle, i: _hmac(cle, i.upper()),
     "l'identifiant {i!r} est en majuscules mais garde ses espaces : `.strip().upper()`."),
    (lambda cle, i: _hmac(cle, i.strip().lower()),
     "la normalisation est `.strip().upper()` (en MAJUSCULES, comme dans le CRM), pas `.lower()`."),
    (lambda cle, i: _hmac(cle, _normaliser(i), hashlib.md5),
     "c'est un HMAC-MD5 : MD5 est cassé depuis 2004. Utilisez `hashlib.sha256`."),
    (lambda cle, i: _hmac(cle, _normaliser(i), hashlib.sha1),
     "c'est un HMAC-SHA1 : utilisez `hashlib.sha256`, comme demandé."),
    (lambda cle, i: _hmac(cle, _normaliser(i), hashlib.sha512),
     "c'est un HMAC-SHA512 : la règle est `hashlib.sha256` (64 caractères hexadécimaux)."),
    (lambda cle, i: _hmac(cle, _normaliser(i), codage="latin-1"),
     "l'identifiant {n!r} est encodé en latin-1 : encodez en UTF-8 (`.encode(\"utf-8\")`), sinon un identifiant "
     "accentué change de pseudonyme selon l'ordinateur."),
]


def _verifier_pseudo(p, cle: bytes, identifiant: str, quoi: str) -> None:
    attendu = _ref_pseudo(cle, identifiant)
    if isinstance(p, (bytes, bytearray)):
        if bytes(p).hex() == attendu:
            raise AssertionError(f"Dans {quoi}, le pseudonyme est des bytes ({_court(p, 40)}) : prenez `.hexdigest()`, "
                                 "pas `.digest()`, pour un texte de 64 caractères qu'on peut écrire dans un CSV.")
        raise AssertionError(f"Dans {quoi}, le pseudonyme est des bytes : un texte (`.hexdigest()`) est attendu.")
    if not isinstance(p, str):
        raise AssertionError(f"Dans {quoi}, le pseudonyme vaut {_court(p, 60)} : un texte de 64 caractères "
                             "hexadécimaux est attendu (`.hexdigest()`). Pas `hash()` : il change à chaque lancement de "
                             "Python.")
    if p == attendu:
        return
    n = _normaliser(identifiant)
    for variante, message in _VARIANTES_PSEUDO:
        if p == variante(cle, identifiant):
            raise AssertionError(f"Dans {quoi}, le pseudonyme de {identifiant!r} " + message.format(i=identifiant, n=n))
    if 0 < len(p) < 64 and attendu.startswith(p):
        raise AssertionError(f"Dans {quoi}, le pseudonyme fait {len(p)} caractères : gardez les 64. Tronqué, il prend "
                             "beaucoup moins de valeurs, et deux clients finissent par recevoir le même (le paradoxe "
                             "des anniversaires).")
    raise AssertionError(f"Dans {quoi}, le pseudonyme de {identifiant!r} vaut {_court(p, 30)} au lieu de "
                         f"{attendu[:16]}… : hmac.new(cle, {n!r}.encode(\"utf-8\"), hashlib.sha256).hexdigest().")


def _un_export(f, lignes, colonnes, cle, quoi: str) -> list:
    copies = copy.deepcopy((lignes, colonnes))
    r = _appel(f, (lignes, colonnes, cle), quoi)
    _intact(copies[0], lignes, quoi, "les lignes")
    _intact(copies[1], colonnes, quoi, "les colonnes")
    _pas_none(r, quoi)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list de dictionnaires est attendue.")
    if len(r) != len(lignes):
        raise AssertionError(f"{quoi} renvoie {len(r)} lignes au lieu de {len(lignes)} : une par client, dans l'ordre.")
    attendues = ["pseudo", *colonnes]
    for numero, (sortie, ligne) in enumerate(zip(r, lignes)):
        ici = f"{quoi}, ligne {numero}"
        if not isinstance(sortie, dict):
            raise AssertionError(f"Dans {ici}, on trouve {_court(sortie, 60)} : un dictionnaire est attendu.")
        if sortie is ligne:
            raise AssertionError(f"Dans {ici}, la ligne renvoyée est la ligne reçue elle-même : construisez un NOUVEAU "
                                 "dictionnaire.")
        en_trop = [c for c in sortie if c not in attendues]
        directs = [c for c in en_trop if c in IDENTIFIANTS_DIRECTS]
        if directs:
            raise AssertionError(f"Dans {ici}, l'identifiant direct {directs[0]!r} est parti au laboratoire : il ne sort "
                                 "JAMAIS. Ne gardez que \"pseudo\" et les colonnes demandées.")
        if en_trop:
            raise AssertionError(f"Dans {ici}, la colonne {en_trop[0]!r} est partie alors que personne ne l'a demandée : "
                                 "c'est une LISTE BLANCHE (seulement \"pseudo\" et `colonnes`), pas une liste noire. Une "
                                 "colonne ajoutée demain au CRM (une note du conseiller, avec un numéro de téléphone) ne "
                                 "doit pas partir sans qu'on l'ait décidé.")
        manquantes = [c for c in attendues if c not in sortie]
        if manquantes:
            raise AssertionError(f"Dans {ici}, la colonne {manquantes[0]!r} manque : chaque colonne demandée est "
                                 "présente (avec None si la ligne ne l'a pas).")
        for colonne in colonnes:
            if not _egaux_nan(sortie[colonne], ligne.get(colonne)):
                raise AssertionError(f"Dans {ici}, {colonne!r} vaut {_court(sortie[colonne], 40)} au lieu de "
                                     f"{_court(ligne.get(colonne), 40)} (la valeur de la ligne, None si elle ne l'a pas).")
        _verifier_pseudo(sortie["pseudo"], cle, ligne["customer_id"], ici)
    return r


_CONSEIL_206 = ("La clé : des bytes d'au moins 16 octets. Une colonne demandée qui est un identifiant direct (customer_id, "
                "nom, email, telephone), ou une ligne sans customer_id texte non vide → ValueError.")


def verifier_preparer_export(f) -> None:
    clients = clients_exemple()
    cle_courte = b"seize octets pil"                                  # exactement 16 : permis
    note = "rappeler Mme Haddad au 06 12 34 56 78, elle hésite"
    pieges = [{"customer_id": " c00412 ", "age": 34, "note_conseiller": note},
              {"customer_id": "C00412\t", "age": 34, "churn": 1},
              {"customer_id": "client-élodie", "age": 27, "nom": "Élodie Martin", "email": "elodie@exemple.fr"},
              {"customer_id": "C00418", "churn": float("nan")}]
    cas = [
        (clients, list(COLONNES_ETUDE), CLE_EXEMPLE),
        (clients, ["age", "churn"], CLE_EXEMPLE),
        (clients, ["motif", "age"], cle_courte),
        (clients, [], b"une autre cle, celle de l'INSEE"),
        (pieges, ["age", "churn"], CLE_EXEMPLE),
        (pieges, ["age"], b"\x00\xff" * 12),
        ([], ["age"], CLE_EXEMPLE),
    ]
    for lignes, colonnes, cle in cas:
        _un_export(f, lignes, colonnes, cle, f"preparer_export({_court(lignes, 50)}, {colonnes}, {_court(cle, 30)})")
    # La même ligne, deux clés : deux pseudonymes. Deux destinataires ne doivent pas pouvoir croiser leurs fichiers.
    a = _un_export(f, clients[:3], [], CLE_EXEMPLE, "preparer_export(clients[:3], [], CLE_EXEMPLE)")
    b = _un_export(f, clients[:3], [], b"la cle d'un autre destinataire", "preparer_export(clients[:3], [], autre_cle)")
    if a[0]["pseudo"] == b[0]["pseudo"]:
        raise AssertionError("preparer_export donne le même pseudonyme avec deux clés différentes : il doit dépendre de "
                             "la clé (sinon, à quoi sert-elle ?).")
    for cle, pourquoi in ((b"", "une clé vide"), (b"trop courte", "une clé de 11 octets"),
                          ("une clé en texte, pas en bytes", "une clé en texte (str), pas en bytes"),
                          (None, "pas de clé")):
        _doit_lever(f, (clients[:2], ["age"], cle), f"preparer_export(clients[:2], ['age'], {cle!r})",
                    f"Ici : {pourquoi}. " + _CONSEIL_206)
    for colonne in sorted(IDENTIFIANTS_DIRECTS):
        _doit_lever(f, (clients[:2], ["age", colonne], CLE_EXEMPLE),
                    f"preparer_export(clients[:2], ['age', {colonne!r}], CLE_EXEMPLE)",
                    f"Ici : {colonne!r} est un identifiant direct, demandé par erreur. " + _CONSEIL_206)
    for ligne in ({"age": 30}, {"customer_id": None, "age": 30}, {"customer_id": "", "age": 30},
                  {"customer_id": "   ", "age": 30}, {"customer_id": 412, "age": 30}):
        _doit_lever(f, ([clients[0], ligne], ["age"], CLE_EXEMPLE),
                    f"preparer_export([clients[0], {ligne}], ['age'], CLE_EXEMPLE)",
                    "Ici : une ligne sans identifiant utilisable. " + _CONSEIL_206)
    alea = random.Random(206)
    autres = ["facture", "note_conseiller", "colonne_absente"]
    for essai in range(120):
        lignes = generer_clients(alea.randint(1, 12), graine=alea.randint(0, 10_000))
        for ligne in lignes:
            forme = alea.random()
            if forme < 0.2:
                ligne["customer_id"] = ligne["customer_id"].lower()
            elif forme < 0.35:
                ligne["customer_id"] = f"  {ligne['customer_id']} "
            if alea.random() < 0.3:
                ligne["note_conseiller"] = note
        colonnes = alea.sample(list(COLONNES_ETUDE) + autres, alea.randint(0, 6))
        cle = bytes(alea.randrange(256) for _ in range(alea.randint(16, 48)))
        _un_export(f, lignes, colonnes, cle, f"preparer_export({_court(lignes, 50)}, {colonnes}, une clé de "
                                             f"{len(cle)} octets) (tiré au hasard n°{essai})")


# ---------------------------------------------------------------------------- défi 207 : réidentifier


def _ref_reid(publication, annuaire, quasi, variante: str = "") -> dict:
    """La référence (un index par motif de valeurs connues) et, en variante, les erreurs classiques."""
    if variante:
        return _ref_reid_naive(publication, annuaire, quasi, variante)
    index: dict = {}
    trouves = {}
    for i, ligne in enumerate(publication):
        connus = tuple(not _manquant(ligne.get(q)) for q in quasi)
        if connus not in index:
            compte: dict = {}
            for personne in annuaire:
                cle = tuple(_valeur(personne.get(q)) for q, c in zip(quasi, connus) if c)
                compte.setdefault(cle, []).append(personne.get("nom"))
            index[connus] = compte
        noms = index[connus].get(tuple(ligne.get(q) for q, c in zip(quasi, connus) if c), [])
        if len(noms) == 1:
            trouves[i] = noms[0]
    return trouves


def _ref_reid_naive(publication, annuaire, quasi, variante: str) -> dict:
    trouves = {}
    for i, ligne in enumerate(publication):
        candidats = []
        for personne in annuaire:
            ok = True
            for q in quasi:
                v, w = ligne.get(q), personne.get(q)
                if variante == "none_valeur":                     # None compté comme une valeur à faire correspondre
                    if _valeur(v) != _valeur(w):
                        ok = False
                elif variante == "nan_valeur":                    # seul `is None` est testé : NaN reste une valeur
                    if v is not None and v != w:
                        ok = False
                elif variante == "annuaire_joker":                # une valeur absente de l'annuaire va avec tout
                    if not _manquant(v) and not _manquant(w) and v != w:
                        ok = False
                elif not _manquant(v) and v != w:
                    ok = False
            if ok:
                candidats.append(personne.get("nom"))
        if variante == "au_moins_un" and candidats:
            trouves[i] = candidats[0]
        elif variante == "noms_distincts" and len(set(candidats)) == 1:
            trouves[i] = candidats[0]
        elif variante not in ("au_moins_un", "noms_distincts") and len(candidats) == 1:
            trouves[i] = candidats[0]
    return trouves


_VARIANTES_REID = [
    ("none_valeur", "une valeur effacée (None) dans la publication ne filtre RIEN : la personne peut avoir n'importe "
                    "quelle valeur. Vous exigez None des deux côtés : votre audit croit la ligne protégée alors que les "
                    "autres valeurs suffisent à la reconnaître."),
    ("nan_valeur", "un NaN est une valeur effacée, comme None : il ne filtre rien. Testé avec `is None` seulement, NaN "
                   "reste une valeur, n'est égal à rien (pas même à lui-même), et votre audit sous-estime le risque. "
                   "Testez `x is None or x != x`."),
    ("annuaire_joker", "dans l'ANNUAIRE, une valeur absente n'est égale à aucune valeur connue : l'attaquant ne sait pas "
                       "si la personne a 34 ans, il ne peut pas la retenir comme candidate sur ce critère."),
    ("noms_distincts", "deux lignes de l'annuaire sont deux personnes, même avec le même nom (des homonymes) : comptez "
                       "les personnes, pas les noms différents."),
    ("au_moins_un", "une ligne n'est ré-identifiée que si elle a UN SEUL candidat : avec deux, l'attaquant ne peut pas "
                    "trancher. Vous prenez le premier venu."),
]

_CONSEIL_REID_LENT = ("Ne comparez pas chaque ligne à chaque personne : construisez un dictionnaire {valeurs: [nombre de "
                      "personnes, premier nom]}, un par motif de valeurs connues, une seule fois chacun.")


def _une_reid(f, publication, annuaire, quasi, quoi: str, lignes_max: int | None = 3_000_000, verifier_intact=True,
              attendu=None) -> None:
    copies = copy.deepcopy((publication, annuaire, quasi)) if verifier_intact else None
    r = _appel(f, (publication, annuaire, quasi), quoi, lignes_max=lignes_max, conseil_lent=_CONSEIL_REID_LENT)
    if copies is not None:
        _intact(copies[0], publication, quoi, "la publication")
        _intact(copies[1], annuaire, quoi, "l'annuaire")
    _pas_none(r, quoi)
    if attendu is None:
        attendu = _ref_reid(publication, annuaire, quasi)
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : un dict {{numéro de ligne publiée: nom}} est "
                             "attendu.")
    if r and all(isinstance(c, str) for c in r) and all(_nombre(v) for v in r.values()):
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : c'est l'inverse, {{numéro de ligne: nom}}.")
    if r == attendu:
        return
    if verifier_intact:
        for variante, message in _VARIANTES_REID:
            autre = _ref_reid(publication, annuaire, quasi, variante)
            if r == autre and autre != attendu:
                raise AssertionError(f"{quoi} : {message}")
    en_trop = sorted(set(r) - set(attendu), key=repr)
    oublies = sorted(set(attendu) - set(r))
    if en_trop:
        i = en_trop[0]
        raise AssertionError(f"Dans {quoi}, la ligne {i!r} est attribuée à {r[i]!r}, mais elle n'a pas UN SEUL candidat "
                             "dans l'annuaire (mêmes valeurs pour chaque quasi-identifiant connu de la ligne).")
    if oublies:
        i = oublies[0]
        raise AssertionError(f"Dans {quoi}, la ligne {i} ({_court(publication[i], 80)}) n'a qu'un candidat, "
                             f"{attendu[i]!r}, et vous ne la ré-identifiez pas.")
    i = next(i for i in attendu if r[i] != attendu[i])
    raise AssertionError(f"Dans {quoi}, la ligne {i} est attribuée à {r[i]!r} au lieu de {attendu[i]!r}.")


def verifier_reidentifier(f) -> None:
    publication = [{"age": 34, "sexe": "F", "motif": "prix"}, {"age": None, "sexe": "M", "motif": "impayé"},
                   {"age": 50, "sexe": "F", "motif": None}]
    annuaire = [{"nom": "Ana", "age": 34, "sexe": "F"}, {"nom": "Ben", "age": 34, "sexe": "M"},
                {"nom": "Cléo", "age": 50, "sexe": "F"}, {"nom": "Dina", "age": 50, "sexe": "F"}]
    export = [{q: c[q] for q in ("age", "sexe", "code_postal", "motif")} for c in clients_exemple()]
    niveaux = {"age": 4, "sexe": 1, "code_postal": 1}
    floute = [appliquer(c, HIERARCHIES, niveaux) for c in export]
    annuaire_floute = [appliquer(p, HIERARCHIES, niveaux) for p in annuaire_exemple()]
    cas = [
        (publication, annuaire, ["age", "sexe"]),
        (publication, annuaire, ["sexe", "age"]),
        (export, annuaire_exemple(), list(QUASI_IDENTIFIANTS)),
        (floute, annuaire_floute, list(QUASI_IDENTIFIANTS)),
        ([{"age": 34, "sexe": "F"}], [{"nom": "Ana", "age": 34, "sexe": "F"}, {"nom": "Ana", "age": 34, "sexe": "F"}],
         ["age", "sexe"]),                                                      # deux homonymes parfaits
        ([{"age": 40, "sexe": "F"}, {"age": None, "sexe": "F"}, {"sexe": "M"}],
         [{"nom": "Zoé", "age": None, "sexe": "F"}, {"nom": "Lou", "sexe": "M"}], ["age", "sexe"]),
        ([{"age": _nan(), "sexe": "F"}, {"age": 40, "sexe": "F"}],
         [{"nom": "Zoé", "age": _nan(), "sexe": "F"}, {"nom": "Iris", "age": 40, "sexe": "F"}], ["age", "sexe"]),
        ([{"age": None, "sexe": None}], [{"nom": "Seul", "age": 70, "sexe": "M"}], ["age", "sexe"]),   # tout effacé
        ([], annuaire, ["age"]),
        (publication, [], ["age", "sexe"]),
    ]
    for pub, ann, quasi in cas:
        _une_reid(f, pub, ann, quasi, f"reidentifier({_court(pub, 50)}, {_court(ann, 40)}, {quasi})")
    _doit_lever(f, (publication, annuaire, []), "reidentifier(publication, annuaire, [])",
                "Sans quasi-identifiant, il n'y a rien à recouper → ValueError.")
    alea = random.Random(207)
    for essai in range(200):
        quasi = alea.sample(["age", "sexe", "cp"], alea.randint(1, 3))
        domaines = {"age": [30, 31, 32], "sexe": ["F", "M"], "cp": ["69003", "69100"]}

        def valeur(q, absent_possible=True):
            tirage = alea.random()
            if tirage < 0.1:
                return None
            if tirage < 0.15:
                return _nan()
            if tirage < 0.2 and absent_possible:
                return "ABSENT"
            return alea.choice(domaines[q])

        def ligne(nom=None):
            d = {} if nom is None else {"nom": nom}
            for q in ("age", "sexe", "cp"):
                v = valeur(q)
                if v != "ABSENT":
                    d[q] = v
            return d

        pub = [ligne() for _ in range(alea.randint(0, 8))]
        ann = [ligne(alea.choice(["Ana", "Ben", "Cléo", "Dina", "Éli"])) for _ in range(alea.randint(0, 10))]
        _une_reid(f, pub, ann, quasi, f"reidentifier({_court(pub, 60)}, {_court(ann, 60)}, {quasi}) (tiré au hasard "
                                      f"n°{essai})")
    pub, ann = _grand_recoupement()
    _une_reid(f, pub, ann, list(QUASI_IDENTIFIANTS), f"reidentifier({len(pub)} lignes, {len(ann)} personnes, "
              f"{list(QUASI_IDENTIFIANTS)})", lignes_max=LIGNES_PAR_LIGNE_207 * (len(pub) + len(ann)),
              verifier_intact=False)


LIGNES_PAR_LIGNE_207 = 150      # une double boucle en exécute des milliers ; un index, une vingtaine au plus


def _grand_recoupement() -> tuple:
    """10 000 lignes publiées (quelques valeurs manquantes) face à un annuaire d'environ 10 000 personnes."""
    clients = generer_clients(10_000, graine=207)
    publication = [{q: c[q] for q in ("age", "sexe", "code_postal", "motif")} for c in clients]
    return publication, generer_annuaire(clients, graine=207, part=0.5, autres=5_000)


# ---------------------------------------------------------------------------- défi 208 : audit


def _ref_audit(lignes, quasi, sensible, variante: str = "") -> tuple:
    groupes: dict = {}
    for numero, ligne in enumerate(lignes):
        brutes = [ligne.get(q) for q in quasi]
        if variante == "dropna" and any(_manquant(v) for v in brutes):
            continue
        if variante == "nan_distincts":
            valeurs = tuple(("NaN", numero) if (isinstance(v, float) and v != v) else v for v in brutes)
        else:
            valeurs = tuple(_valeur(v) for v in brutes)
        s = ligne.get(sensible)
        if variante == "sensible_nan" and isinstance(s, float) and s != s:
            s = ("NaN", numero)
        else:
            s = _valeur(s)
        groupe = groupes.setdefault(valeurs, [0, set()])
        groupe[0] += 1
        if not (variante == "sensible_sans_manquant" and s is None):
            groupe[1].add(s)
    classes = []
    for valeurs, (effectif, vues) in groupes.items():
        if variante == "nan_distincts":
            valeurs = tuple(_nan() if isinstance(v, tuple) else v for v in valeurs)
        diversite = effectif if variante == "diversite_effectif" else len(vues)
        classes.append((valeurs, effectif, diversite))
    if not classes:
        return 0, 0, []
    if variante == "tri_decroissant":
        classes.sort(key=lambda c: (c[1], c[2]), reverse=True)
    elif variante == "tri_effectif_seul":
        classes.sort(key=lambda c: c[1])
    elif variante != "sans_tri":
        classes.sort(key=lambda c: (c[1], c[2]))
    k = min(c[1] for c in classes)
    l = min(c[2] for c in classes)
    if variante == "k_max":
        k = max(c[1] for c in classes)
    elif variante == "l_max":
        l = max(c[2] for c in classes)
    elif variante == "l_globale":
        l = len({_valeur(ligne.get(sensible)) for ligne in lignes})
    elif variante == "nombre_de_classes":
        k = len(classes)
    return k, l, classes


_VARIANTES_AUDIT = [
    ("dropna", "les lignes où un quasi-identifiant manque ont disparu de votre audit (c'est ce que fait `groupby` de "
               "pandas par défaut). Elles forment une classe comme les autres, avec None : « le client dont l'âge "
               "manque » est souvent seul de son espèce. Avec pandas : `groupby(…, dropna=False)`."),
    ("nan_distincts", "deux NaN ne sont pas égaux entre eux : chaque ligne où l'âge vaut NaN forme sa propre classe. "
                      "Remplacez toute valeur manquante (None ou NaN) par None AVANT de grouper."),
    ("sensible_nan", "dans la colonne sensible aussi, deux NaN ne sont pas égaux : chacun compte comme une valeur "
                     "différente et gonfle la diversité. Remplacez NaN par None avant de compter."),
    ("sensible_sans_manquant", "une valeur sensible manquante compte comme une valeur (None) dans la diversité."),
    ("diversite_effectif", "diversite = le nombre de valeurs DIFFÉRENTES de l'attribut sensible dans la classe (un "
                           "set), pas le nombre de lignes."),
    ("k_max", "k est le PLUS PETIT effectif : le fichier n'est pas plus sûr que sa ligne la plus exposée."),
    ("l_max", "l est la PLUS PETITE diversité : une seule classe homogène suffit à trahir ses membres."),
    ("l_globale", "l se mesure classe par classe (la plus petite diversité), pas sur tout le fichier : un fichier peut "
                  "contenir cinq motifs et une classe où tout le monde a le même."),
    ("nombre_de_classes", "k est le plus petit EFFECTIF de classe, pas le nombre de classes."),
    ("tri_decroissant", "les classes se trient par effectif CROISSANT puis diversité croissante : les plus exposées "
                        "d'abord."),
    ("sans_tri", "les classes doivent être triées par effectif croissant, puis par diversité croissante (à égalité, "
                 "dans l'ordre de première apparition)."),
    ("tri_effectif_seul", "à effectif égal, les classes se trient par diversité croissante : `key=lambda c: (c.effectif, "
                          "c.diversite)`."),
]


def _memes_classes(r, attendu) -> bool:
    """Deux listes de classes égales, dans le même ordre (deux NaN comptent comme égaux)."""
    if len(r) != len(attendu):
        return False
    return all(_egaux_nan(tuple(c[0]), tuple(a[0])) and c[1] == a[1] and c[2] == a[2] for c, a in zip(r, attendu))


def _un_audit(f, lignes, quasi, sensible, quoi: str) -> None:
    copies = copy.deepcopy(lignes)
    r = _appel(f, (lignes, quasi, sensible), quoi, "Une valeur manquante, None ou NaN, devient None : ne triez pas sur "
                                                    "les valeurs (comparer None à un nombre lève TypeError).")
    _intact(copies, lignes, quoi, "les lignes")
    _pas_none(r, quoi)
    k, l, classes = _ref_audit(lignes, quasi, sensible)
    if not isinstance(r, tuple) or len(r) != 3:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : un Audit(k, l, classes) (ou un tuple de 3 valeurs) est "
                             "attendu.")
    rk, rl, rc = r
    if not isinstance(rc, list) or not all(isinstance(c, tuple) and len(c) == 3 for c in rc):
        raise AssertionError(f"Dans {quoi}, classes vaut {_court(rc, 100)} : une list de Classe(valeurs, effectif, "
                             "diversite) est attendue.")
    for c in rc:
        if not isinstance(c[0], tuple):
            raise AssertionError(f"Dans {quoi}, une classe a pour valeurs {_court(c[0], 60)} : un tuple des valeurs des "
                                 "quasi-identifiants, dans leur ordre, est attendu.")
    if rk == k and rl == l and _memes_classes(rc, classes):
        return
    for variante, message in _VARIANTES_AUDIT:
        ak, al, ac = _ref_audit(lignes, quasi, sensible, variante)
        if rk == ak and rl == al and _memes_classes(rc, ac) and not (ak == k and al == l and _memes_classes(ac, classes)):
            raise AssertionError(f"{quoi} : {message}")
    if rk != k:
        raise AssertionError(f"Dans {quoi}, k vaut {_court(rk, 30)} au lieu de {k} (le plus petit effectif de classe).")
    if rl != l:
        raise AssertionError(f"Dans {quoi}, l vaut {_court(rl, 30)} au lieu de {l} (la plus petite diversité de classe).")
    attendues = {c[0]: c for c in classes}
    for c in rc:
        valeurs = tuple(_valeur(v) for v in c[0])
        if valeurs not in attendues:
            raise AssertionError(f"Dans {quoi}, la classe {_court(c[0], 60)} n'existe pas (les valeurs des "
                                 f"quasi-identifiants {quasi}, None si elles manquent).")
        a = attendues[valeurs]
        if (c[1], c[2]) != (a[1], a[2]):
            raise AssertionError(f"Dans {quoi}, la classe {_court(a[0], 60)} a effectif={_court(c[1], 20)}, "
                                 f"diversite={_court(c[2], 20)} au lieu de effectif={a[1]}, diversite={a[2]}.")
    if len(rc) != len(classes):
        raise AssertionError(f"Dans {quoi}, il y a {len(rc)} classes au lieu de {len(classes)} : une par combinaison "
                             "de valeurs (None si elles manquent), aucune oubliée, aucune en double.")
    raise AssertionError(f"Dans {quoi}, les classes sont dans le désordre : par effectif croissant, puis diversité "
                         "croissante, et à égalité dans l'ordre de première apparition (un tri STABLE).")


def verifier_audit(f) -> None:
    exemple = [{"age": 30, "motif": "prix"}, {"age": 41, "motif": "prix"}, {"age": None, "motif": "impayé"},
               {"age": 30, "motif": "prix"}, {"age": 41, "motif": None}, {"age": _nan(), "motif": None},
               {"age": 41, "motif": "prix"}]
    export = [{q: c[q] for q in ("age", "sexe", "code_postal", "motif")} for c in clients_exemple()]
    floute = [appliquer(c, HIERARCHIES, {"age": 4, "sexe": 1, "code_postal": 1}) for c in export]
    cas = [
        (exemple, ["age"], "motif"),
        (export, list(QUASI_IDENTIFIANTS), "motif"),
        (floute, list(QUASI_IDENTIFIANTS), "motif"),
        (floute, ["code_postal"], "motif"),
        ([{"age": 30, "motif": "prix"}], ["age"], "motif"),
        ([{"age": 30, "sexe": "F", "motif": "prix"}, {"age": 30, "motif": "prix"}, {"sexe": "F", "motif": _nan()},
          {"age": _nan(), "sexe": "F", "motif": _nan()}, {"age": 30, "sexe": None, "motif": "impayé"}],
         ["age", "sexe"], "motif"),
        ([{"x": 1, "s": "a"}, {"x": 2, "s": "a"}, {"x": 1, "s": "b"}, {"x": 2, "s": "a"}, {"x": 3, "s": "a"},
          {"x": 3, "s": "b"}], ["x"], "s"),
    ]
    for lignes, quasi, sensible in cas:
        _un_audit(f, lignes, quasi, sensible, f"audit({_court(lignes, 60)}, {quasi}, {sensible!r})")
    for lignes, quasi, sensible, pourquoi in (([], ["age"], "motif", "aucune ligne"),
                                             (exemple, [], "motif", "aucun quasi-identifiant"),
                                             (exemple, ["age", "motif"], "motif", "l'attribut sensible est aussi "
                                                                                 "quasi-identifiant")):
        _doit_lever(f, (lignes, quasi, sensible), f"audit({_court(lignes, 40)}, {quasi}, {sensible!r})",
                    f"Ici : {pourquoi} → ValueError.")
    alea = random.Random(208)
    for essai in range(200):
        quasi = alea.sample(["age", "sexe", "cp"], alea.randint(1, 3))
        domaines = {"age": [30, 31, 45], "sexe": ["F", "M"], "cp": ["69003", "69100"], "motif": ["prix", "impayé", "aucun"]}
        lignes = []
        for _ in range(alea.randint(1, 12)):
            ligne = {}
            for colonne in ("age", "sexe", "cp", "motif"):
                tirage = alea.random()
                if tirage < 0.08:
                    ligne[colonne] = None
                elif tirage < 0.16:
                    ligne[colonne] = _nan()
                elif tirage < 0.2:
                    continue
                else:
                    ligne[colonne] = alea.choice(domaines[colonne])
            lignes.append(ligne)
        _un_audit(f, lignes, quasi, "motif", f"audit({_court(lignes, 70)}, {quasi}, 'motif') (tiré au hasard n°{essai})")


# ---------------------------------------------------------------------------- défi 209 : généraliser


def _vecteurs(hierarchies, variante: str = "") -> list:
    attributs = list(hierarchies)
    if variante == "ordre_inverse":                 # lexicographique en partant du DERNIER attribut
        tous = [tuple(reversed(v)) for v in itertools.product(*(range(len(hierarchies[a])) for a in reversed(attributs)))]
    else:
        tous = list(itertools.product(*(range(len(hierarchies[a])) for a in attributs)))
    if variante == "dernier_a_egalite":
        return sorted(tous, key=lambda v: (sum(v), [-x for x in v]))
    if variante == "hauteur_max":
        return sorted(tous, key=max)
    return sorted(tous, key=sum)


def _essayer(lignes, hierarchies, vecteur, k: int, smax: int, variante: str = ""):
    attributs = list(hierarchies)
    cles = [tuple(hierarchies[a][n](ligne.get(a)) for a, n in zip(attributs, vecteur)) for ligne in lignes]
    effectifs = Counter(cles)
    rare = (lambda e: e <= k) if variante == "inferieur_ou_egal" else (lambda e: e < k)
    if variante == "classes_comptees":
        trop = sum(1 for e in effectifs.values() if rare(e))
    else:
        trop = sum(1 for c in cles if rare(effectifs[c]))
    if trop > (0 if variante == "sans_suppression" else smax):
        return None
    gardees = []
    for ligne, cle in zip(lignes, cles):
        if not rare(effectifs[cle]) or variante == "suppression_gardee":
            copie = dict(ligne)
            copie.update(zip(attributs, cle))
            gardees.append(copie)
    if variante == "suppression_gardee":                # comptées comme supprimées… et restées dans le fichier
        return dict(zip(attributs, vecteur)), gardees, trop
    return dict(zip(attributs, vecteur)), gardees, len(lignes) - len(gardees)


def _ref_generaliser(lignes, hierarchies, k: int, smax: int, variante: str = ""):
    if variante == "glouton":
        return _ref_glouton(lignes, hierarchies, k, smax)
    for vecteur in _vecteurs(hierarchies, variante):
        resultat = _essayer(lignes, hierarchies, vecteur, k, smax, variante)
        if resultat is not None:
            return resultat
    return None


def _ref_glouton(lignes, hierarchies, k: int, smax: int):
    """Datafly : monter d'un niveau l'attribut qui a le plus de valeurs différentes, jusqu'à ce que ça passe."""
    attributs = list(hierarchies)
    vecteur = [0] * len(attributs)
    while True:
        resultat = _essayer(lignes, hierarchies, tuple(vecteur), k, smax)
        if resultat is not None:
            return resultat
        choix = [(len({hierarchies[a][vecteur[i]](ligne.get(a)) for ligne in lignes}), -i, i)
                 for i, a in enumerate(attributs) if vecteur[i] < len(hierarchies[a]) - 1]
        if not choix:
            return None
        vecteur[max(choix)[2]] += 1


_VARIANTES_GENERALISER = [
    ("glouton", "monter pas à pas l'attribut qui a le plus de valeurs différentes (l'approche gloutonne de Datafly) ne "
                "donne pas la généralisation la moins haute. Essayez TOUS les vecteurs de niveaux, par hauteur (somme "
                "des niveaux) croissante."),
    ("hauteur_max", "la hauteur d'un vecteur est la SOMME de ses niveaux, pas le plus grand : (2, 2, 2) monte de 6 "
                    "crans en tout, (3, 0, 0) de 3."),
    ("dernier_a_egalite", "à hauteur égale, on garde le PREMIER vecteur acceptable dans l'ordre lexicographique, celui de "
                          "`itertools.product` (le premier quasi-identifiant varie le MOINS vite). `sorted(…, key=sum)` "
                          "est stable : il garde cet ordre."),
    ("ordre_inverse", "à hauteur égale, on garde le PREMIER vecteur acceptable dans l'ordre lexicographique, celui de "
                      "`itertools.product` : le premier quasi-identifiant (la première clé de `hierarchies`) varie le "
                      "MOINS vite."),
    ("classes_comptees", "suppression_max compte des LIGNES : une classe de 2 lignes trop rares, ce sont 2 lignes à "
                         "retirer, pas une."),
    ("inferieur_ou_egal", "une classe de k lignes est k-anonyme : seules les classes de MOINS de k lignes posent problème "
                          "(`effectif < k`)."),
    ("sans_suppression", "vous n'utilisez pas `suppression_max` : retirer quelques lignes trop singulières évite souvent "
                         "de généraliser tout le monde."),
    ("suppression_gardee", "les lignes des classes trop petites sont comptées comme supprimées… mais elles sont restées "
                           "dans le fichier. Retirez-les de `lignes`."),
]


def _un_generaliser(f, lignes, hierarchies, k, smax, quoi: str) -> None:
    copies = copy.deepcopy(lignes)
    args = (lignes, hierarchies, k) if smax is None else (lignes, hierarchies, k, smax)
    r = _appel(f, args, quoi, "Pour chaque vecteur : hierarchies[q][niveau](ligne.get(q)) pour chaque quasi-identifiant.")
    _intact(copies, lignes, quoi, "les lignes")
    _pas_none(r, quoi)
    s = 0 if smax is None else smax
    attendu = _ref_generaliser(lignes, hierarchies, k, s)
    if not isinstance(r, tuple) or len(r) != 3:
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : une Generalisation(niveaux, lignes, supprimees) (ou un "
                             "tuple de 3 valeurs) est attendue.")
    niveaux, sorties, supprimees = r
    if not isinstance(niveaux, dict):
        raise AssertionError(f"Dans {quoi}, niveaux vaut {_court(niveaux, 60)} : un dict {{quasi-identifiant: niveau}} "
                             "est attendu.")
    if not isinstance(sorties, list) or not all(isinstance(x, dict) for x in sorties):
        raise AssertionError(f"Dans {quoi}, lignes vaut {_court(sorties, 80)} : une list de dictionnaires est attendue.")
    for x in sorties:
        if any(x is ligne for ligne in lignes):
            raise AssertionError(f"Dans {quoi}, une ligne renvoyée est une ligne reçue : renvoyez des COPIES "
                                 "(`dict(ligne)`), même quand rien ne change.")
    if (niveaux, supprimees) == (attendu[0], attendu[2]) and _egaux_nan(sorties, attendu[1]):
        return
    for variante, message in _VARIANTES_GENERALISER:
        autre = _ref_generaliser(lignes, hierarchies, k, s, variante)
        if autre is None:
            continue
        if (niveaux, supprimees) == (autre[0], autre[2]) and _egaux_nan(sorties, autre[1]) and not (
                autre[0] == attendu[0] and autre[2] == attendu[2] and _egaux_nan(autre[1], attendu[1])):
            if niveaux == attendu[0]:
                raise AssertionError(f"{quoi} : {message}")
            raise AssertionError(f"{quoi} donne les niveaux {niveaux} au lieu de {attendu[0]} : {message}")
    if niveaux != attendu[0]:
        hauteur = sum(attendu[0].values())
        raise AssertionError(f"Dans {quoi}, niveaux vaut {_court(niveaux, 80)} au lieu de {attendu[0]} (hauteur "
                             f"{hauteur}) : le vecteur acceptable de plus petite hauteur, le premier dans l'ordre de "
                             "`itertools.product` à hauteur égale.")
    if supprimees != attendu[2]:
        raise AssertionError(f"Dans {quoi}, supprimees vaut {_court(supprimees, 20)} au lieu de {attendu[2]} : le nombre "
                             f"de lignes dont la classe compte moins de k = {k} lignes.")
    if len(sorties) != len(attendu[1]):
        raise AssertionError(f"Dans {quoi}, il reste {len(sorties)} lignes au lieu de {len(attendu[1])} (toutes, sauf les "
                             "lignes supprimées).")
    i = next(i for i in range(len(sorties)) if not _egaux_nan(sorties[i], attendu[1][i]))
    raise AssertionError(f"Dans {quoi}, la ligne gardée n°{i} vaut {_court(sorties[i], 100)} au lieu de "
                         f"{_court(attendu[1][i], 100)} : une copie de la ligne, chaque quasi-identifiant remplacé par sa "
                         "valeur généralisée, les autres colonnes intactes, dans l'ordre.")


def _dizaine(x):
    return None if _manquant(x) else x // 10


_SANS_SOMMET = {"x": [exact, _dizaine]}                       # le dernier niveau ne réunit pas tout le monde


def verifier_generaliser(f) -> None:
    quatre = [{"age": 31, "sexe": "F"}, {"age": 33, "sexe": "M"}, {"age": 36, "sexe": "F"}, {"age": 52, "sexe": "M"}]
    deux = {"age": HIERARCHIES["age"], "sexe": HIERARCHIES["sexe"]}
    export = [{q: c[q] for q in ("age", "sexe", "code_postal", "motif")} for c in clients_exemple()]
    # (0, 1) et (1, 0) passent tous les deux : (0, 1) d'abord. (Le glouton, lui, tombe juste : b a plus de valeurs.)
    egalite = [{"a": a, "b": b} for a in (1, 2) for b in "xyz"]
    h_egalite = {"a": [exact, masquer], "b": [exact, masquer]}
    h_trois = {"a": [exact, masquer], "b": [exact, masquer], "c": [exact, masquer]}
    egalite_trois = [{"a": a, "b": b, "c": c} for a, b, c in ((1, 1, 1), (1, 2, 1), (2, 1, 2), (2, 2, 2))]
    sept = [{q: c[q] for q in ("age", "sexe", "code_postal")} for c in generer_clients(7, graine=279267, manquants=0)]
    quatre_bis = [{q: c[q] for q in ("age", "sexe", "code_postal")}
                  for c in generer_clients(4, graine=553259, manquants=0)]
    cas = [
        (egalite, h_egalite, 2, 0),
        (egalite_trois, h_trois, 2, 0),            # le glouton monte à (1, 1, 0) ; (0, 1, 0) suffisait
        (sept, HIERARCHIES, 2, 2),                 # le glouton monte trop haut
        (quatre_bis, HIERARCHIES, 2, 1),           # le plus grand niveau n'est pas la hauteur
        (quatre, deux, 2, None),
        (quatre, deux, 2, 1),
        (quatre, {"sexe": HIERARCHIES["sexe"], "age": HIERARCHIES["age"]}, 2, 1),
        (export, HIERARCHIES, 3, None),
        (export, HIERARCHIES, 2, 1),
        (export, HIERARCHIES, 1, 0),
        (export, HIERARCHIES, 20, None),
        ([{"x": 3, "y": 1}, {"x": 7, "y": 1}, {"x": 13, "y": 2}], _SANS_SOMMET | {"y": [exact, masquer]}, 2, 1),
        ([{"age": None, "sexe": "F"}, {"age": _nan(), "sexe": "F"}, {"sexe": "M"}, {"age": 40, "sexe": "M"}],
         deux, 2, 0),
    ]
    for lignes, hierarchies, k, smax in cas:
        appel = "" if smax is None else f", suppression_max={smax}"
        _un_generaliser(f, lignes, hierarchies, k, smax,
                        f"generaliser({_court(lignes, 50)}, hiérarchies {list(hierarchies)}, k={k}{appel})")
    for lignes, hierarchies, k, smax, pourquoi in (
            (quatre, deux, 0, 0, "k = 0"), (quatre, deux, 5, 0, "k > nombre de lignes"),
            (quatre, deux, 2.0, 0, "k n'est pas un entier"), (quatre, deux, True, 0, "k = True"),
            (quatre, deux, 2, -1, "suppression_max < 0"), (quatre, deux, 2, 4, "suppression_max = nombre de lignes"),
            (quatre, deux, 2, True, "suppression_max = True"), (quatre, {}, 2, 0, "aucune hiérarchie"),
            ([], deux, 1, 0, "aucune ligne"),
            ([{"x": 3}, {"x": 25}], _SANS_SOMMET, 2, 0, "aucun vecteur acceptable (le dernier niveau ne réunit pas "
                                                        "tout)")):
        _doit_lever(f, (lignes, hierarchies, k, smax), f"generaliser({_court(lignes, 40)}, hiérarchies "
                    f"{list(hierarchies)}, k={k!r}, suppression_max={smax!r})", f"Ici : {pourquoi} → ValueError.")
    alea = random.Random(209)
    for essai in range(200):
        n = alea.randint(2, 25)
        lignes = [{q: c[q] for q in ("age", "sexe", "code_postal", "churn")}
                  for c in generer_clients(n, graine=alea.randint(0, 10_000), manquants=0.08)]
        attributs = alea.sample(list(HIERARCHIES), alea.randint(1, 3))
        hierarchies = {a: HIERARCHIES[a] for a in attributs}
        k = alea.randint(1, min(n, 5))
        smax = alea.randint(0, min(n - 1, 4))
        _un_generaliser(f, lignes, hierarchies, k, smax, f"generaliser({_court(lignes, 60)}, hiérarchies {attributs}, "
                                                         f"k={k}, suppression_max={smax}) (tiré au hasard n°{essai})")
    grand = [{q: c[q] for q in ("age", "sexe", "code_postal")} for c in generer_clients(600, graine=209)]
    _un_generaliser(f, grand, HIERARCHIES, 10, 15, "generaliser(600 clients, HIERARCHIES, k=10, suppression_max=15)")


# ---------------------------------------------------------------------------- défi 210 : secret statistique


def _rangees(n: int, m: int, lignes: bool = True, colonnes: bool = True) -> list:
    rangees = [[(i, j) for j in range(m)] for i in range(n)] if lignes else []
    return rangees + ([[(i, j) for i in range(n)] for j in range(m)] if colonnes else [])


def _ref_secret(tableau, seuil: int, variante: str = "") -> list:
    n, m = len(tableau), len(tableau[0])
    if variante == "inferieur_ou_egal":
        cache = [[1 <= x <= seuil for x in ligne] for ligne in tableau]
    elif variante == "zeros":
        cache = [[x < seuil for x in ligne] for ligne in tableau]
    else:
        cache = [[1 <= x < seuil for x in ligne] for ligne in tableau]
    rangees = _rangees(n, m, colonnes=variante != "lignes_seules")

    def complement(rangee):
        visibles = [(tableau[i][j], p, (i, j)) for p, (i, j) in enumerate(rangee) if not cache[i][j]]
        if variante == "plus_grande":
            return max(visibles, key=lambda v: (v[0], -v[1]))[2]
        if variante == "premiere_visible":
            return visibles[0][2]
        if variante == "egalite_derniere":
            return min(visibles, key=lambda v: (v[0], -v[1]))[2]
        return min(visibles)[2]

    if variante in ("une_passe", "sans_recommencer"):
        while True:
            change = False
            for rangee in rangees:
                if sum(cache[i][j] for i, j in rangee) == 1:
                    i, j = complement(rangee)
                    cache[i][j] = change = True
            if not change or variante == "une_passe":
                return cache
    while True:
        rangee = next((r for r in rangees if sum(cache[i][j] for i, j in r) == 1), None)
        if rangee is None:
            return cache
        i, j = complement(rangee)
        cache[i][j] = True


_VARIANTES_SECRET = [
    ("lignes_seules", "les totaux des COLONNES sont publiés aussi : une colonne qui n'a qu'une case cachée la trahit. "
                      "Traitez les lignes, puis les colonnes."),
    ("une_passe", "une seule passe ne suffit pas : cacher une case pour une colonne peut laisser une LIGNE avec une "
                  "seule case cachée. Recommencez jusqu'à ce que plus aucune ligne ni colonne n'en ait une seule."),
    ("sans_recommencer", "après chaque case cachée, RECOMMENCEZ depuis la première ligne : l'ordre des cases cachées "
                         "change le résultat, et la règle fixe celui-là."),
    ("inferieur_ou_egal", "le secret primaire cache les cases de 1 à seuil − 1 (`1 <= x < seuil`) : une case qui vaut "
                          "le seuil est publiable."),
    ("zeros", "une case à 0 n'expose personne : elle reste visible au secret primaire (`1 <= x < seuil`)."),
    ("plus_grande", "on cache la PLUS PETITE case visible de la rangée : cacher 52 pour protéger 2 fait perdre "
                    "beaucoup d'information pour la même protection."),
    ("premiere_visible", "on cache la plus PETITE case visible de la rangée, pas la première venue."),
    ("egalite_derniere", "à égalité de valeur, on cache la PREMIÈRE case de la rangée (la plus à gauche dans une ligne, "
                         "la plus haute dans une colonne)."),
]


def _attaque(cache, tableau) -> str:
    """Une case cachée seule dans sa rangée, recalculée par soustraction (le message de l'attaquant), ou ''."""
    n, m = len(tableau), len(tableau[0])
    for numero, rangee in enumerate(_rangees(n, m)):
        cachees = [(i, j) for i, j in rangee if cache[i][j]]
        if len(cachees) == 1:
            i, j = cachees[0]
            total = sum(tableau[a][b] for a, b in rangee)
            visibles = total - tableau[i][j]
            genre = f"la ligne {i}" if numero < n else f"la colonne {j}"
            return (f"{genre} n'a qu'une case cachée, ({i}, {j}) : son total publié ({total}) moins ses cases visibles "
                    f"({visibles}) donne {tableau[i][j]}. Retrouvée par soustraction.")
    return ""


def _un_secret(f, tableau, seuil: int, quoi: str) -> None:
    copies = copy.deepcopy(tableau)
    r = _appel(f, (tableau, seuil), quoi)
    _intact(copies, tableau, quoi, "le tableau")
    _pas_none(r, quoi)
    attendu = _ref_secret(tableau, seuil)
    n, m = len(tableau), len(tableau[0])
    if not isinstance(r, list) or len(r) != n or not all(isinstance(x, list) and len(x) == m for x in r):
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : une list de {n} lignes de {m} booléens est attendue.")
    if r is tableau or any(x is y for x, y in zip(r, tableau)):
        raise AssertionError(f"{quoi} renvoie (une ligne du) tableau reçu : construisez une NOUVELLE liste de listes.")
    if not all(isinstance(x, bool) for ligne in r for x in ligne):
        raise AssertionError(f"{quoi} renvoie {_court(r, 100)} : des booléens (True = cachée) sont attendus.")
    if r == attendu:
        return
    for i in range(n):
        for j in range(m):
            if 1 <= tableau[i][j] < seuil and not r[i][j]:
                raise AssertionError(f"Dans {quoi}, la case ({i}, {j}) vaut {tableau[i][j]} (moins de {seuil}) et reste "
                                     "visible : c'est le secret PRIMAIRE, `1 <= x < seuil`.")
    for variante, message in _VARIANTES_SECRET:
        autre = _ref_secret(tableau, seuil, variante)
        if variante == "une_passe" and not _attaque(autre, tableau):
            continue                                    # une passe a suffi ici : l'erreur est ailleurs
        if r == autre and autre != attendu:
            fuite = _attaque(r, tableau)
            raise AssertionError(f"{quoi} : {message}" + (f" Ici, {fuite}" if fuite else ""))
    fuite = _attaque(r, tableau)
    if fuite:
        raise AssertionError(f"Dans {quoi}, {fuite} Tant qu'une ligne ou une colonne a une seule case cachée, cachez-y "
                             "la plus petite case visible.")
    i, j = next((i, j) for i in range(n) for j in range(m) if r[i][j] != attendu[i][j])
    raise AssertionError(f"Dans {quoi}, la case ({i}, {j}) est {'cachée' if r[i][j] else 'visible'} au lieu d'être "
                         f"{'cachée' if attendu[i][j] else 'visible'} (attendu : {attendu}). Rappel de l'ordre : la "
                         "première rangée à une seule case cachée (lignes, puis colonnes), sa plus petite case visible "
                         "(la première à égalité), et on recommence du début.")


def verifier_secret_statistique(f) -> None:
    cas = [
        ([[12, 2, 30], [8, 15, 4], [20, 9, 11]], 3),
        (comptages_exemple(), 3),
        (comptages_exemple(), 5),
        (comptages_exemple(), 11),
        (comptages_exemple(), 1),                                         # seuil 1 : rien à cacher
        ([[10, 20], [30, 40]], 3),
        ([[10, 2], [20, 30]], 3),
        ([[0, 0, 5], [7, 0, 9]], 3),                                       # des zéros : rien d'exposé
        ([[5, 1, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]], 3),                   # égalités : la première
        ([[40, 2, 9], [30, 50, 60], [70, 80, 1]], 3),                      # deux secrets primaires
        ([[3, 50], [50, 3]], 3),                                           # le seuil lui-même est publiable
        ([[1, 100, 100], [100, 100, 100], [100, 100, 2]], 3),
        ([[9, 1, 8], [7, 6, 5]], 2),
    ]
    for tableau, seuil in cas:
        _un_secret(f, tableau, seuil, f"secret_statistique({_court(tableau, 70)}, {seuil})")
    for tableau, seuil, pourquoi in (
            ([[1, 2, 3]], 3, "une seule ligne"), ([[1], [2]], 3, "une seule colonne"), ([], 3, "aucune ligne"),
            ([[1, 2], [3]], 3, "lignes de longueurs différentes"), ([[1, -2], [3, 4]], 3, "un comptage négatif"),
            ([[1, 2.0], [3, 4]], 3, "un comptage qui n'est pas un entier"), ([[1, True], [3, 4]], 3, "True"),
            ([[1, 2], [3, 4]], 0, "seuil 0"), ([[1, 2], [3, 4]], 2.5, "seuil non entier"),
            ([[1, 2], [3, 4]], True, "seuil = True")):
        _doit_lever(f, (tableau, seuil), f"secret_statistique({tableau}, {seuil!r})",
                    f"Ici : {pourquoi}. Il faut au moins 2 lignes et 2 colonnes de même longueur, des entiers >= 0, et "
                    "un seuil entier >= 1 → sinon ValueError.")
    alea = random.Random(210)
    for essai in range(300):
        n, m = alea.randint(2, 6), alea.randint(2, 6)
        tableau = [[alea.choice([0, 1, 2, 3, 4]) if alea.random() < 0.3 else alea.randint(5, 60) for _ in range(m)]
                   for _ in range(n)]
        seuil = alea.choice([2, 3, 3, 5, 11])
        _un_secret(f, tableau, seuil, f"secret_statistique({_court(tableau, 70)}, {seuil}) (tiré au hasard n°{essai})")


# ---------------------------------------------------------------------------- défi 211 : le budget


class _Tirage:
    """Une source de hasard qui renvoie les valeurs prévues ET compte ses tirages."""

    def __init__(self, valeurs):
        self.valeurs = list(valeurs)
        self.tires = 0

    def __call__(self) -> float:
        if self.tires >= len(self.valeurs):
            raise RuntimeError("le correcteur n'avait prévu qu'un tirage par question acceptée")
        self.tires += 1
        return self.valeurs[self.tires - 1]


def _laplace(echelle: float, u: float, variante: str = "") -> float:
    c = u - 0.5
    signe = math.copysign(1.0, c)
    if variante == "sans_signe":
        return -echelle * math.log(1 - 2 * abs(c))
    if variante == "signe_inverse":
        return echelle * signe * math.log(1 - 2 * abs(c))
    if variante == "exponentielle":
        return -echelle * math.log(u)
    if variante == "log10":
        return -echelle * signe * math.log10(1 - 2 * abs(c))
    return -echelle * signe * math.log(1 - 2 * abs(c))


_VARIANTES_BRUIT = [
    ("echelle_epsilon", "l'échelle du bruit est 1 / epsilon (sensibilité / epsilon), pas epsilon : plus epsilon est "
                        "PETIT (plus de confidentialité), plus le bruit doit être GRAND."),
    ("echelle_un", "l'échelle du bruit dépend d'epsilon : b = 1 / epsilon pour un comptage, plafond / epsilon pour une "
                   "somme."),
    ("echelle_plafond_oublie", "pour une somme, l'échelle est plafond / epsilon : un client peut changer la somme de "
                               "`plafond`, pas de 1 (c'est la SENSIBILITÉ de la question)."),
    ("sans_signe", "le bruit est −b × signe(u − 0,5) × ln(1 − 2|u − 0,5|) : il est négatif quand u < 0,5. Sans le "
                   "signe, il est toujours positif, et toutes les réponses sont trop hautes."),
    ("signe_inverse", "le bruit est −b × signe(u − 0,5) × ln(1 − 2|u − 0,5|) : le signe est inversé (avec u > 0,5, le "
                      "bruit est positif)."),
    ("exponentielle", "−b × ln(u) est un bruit EXPONENTIEL (toujours positif) : celui de Laplace est "
                      "−b × signe(u − 0,5) × ln(1 − 2|u − 0,5|)."),
    ("log10", "ln est le logarithme NATUREL : `math.log`, pas `math.log10`."),
    ("sans_bruit", "la réponse doit être bruitée : vrai résultat + bruit de Laplace. Sans bruit, la différence entre "
                   "deux questions (« tous les clients » et « tous sauf Karim ») révèle Karim."),
    ("arrondi", "la réponse est un float, non arrondi : vrai résultat + bruit."),
]


def _attendu_reponse(vrai: float, echelle: float, u: float) -> float:
    return vrai + _laplace(echelle, u)


def _pieges_somme(lignes, colonne: str, plafond: float) -> list:
    """Les vrais résultats qu'on obtient en oubliant une partie du plafonnement."""
    valeurs = [ligne.get(colonne) for ligne in lignes if not _manquant(ligne.get(colonne))]
    return [
        (sum(valeurs), "chaque valeur doit être ramenée dans [0, plafond] AVANT de sommer : sans plafond, une facture de "
                       "2 500 € se lit dans la réponse (le bruit, calibré pour `plafond`, ne la cache pas)."),
        (sum(min(v, plafond) for v in valeurs), "les valeurs NÉGATIVES aussi se ramènent dans [0, plafond] : "
                                                "`min(max(v, 0), plafond)`. Un avoir de −20 € compte 0."),
        (sum(max(v, 0) for v in valeurs), "les valeurs au-dessus du plafond se ramènent au plafond : "
                                          "`min(max(v, 0), plafond)`."),
    ]


def _nommer_bruit(resultat, vrai: float, epsilon: float, sensibilite: float, u: float, quoi: str,
                  pieges: list = ()) -> None:
    attendu = vrai + _laplace(sensibilite / epsilon, u)
    if _proche(resultat, attendu):
        return
    if isinstance(resultat, float) and resultat != resultat:
        raise AssertionError(f"{quoi} renvoie NaN : une valeur manquante (NaN) a contaminé le calcul. Une valeur "
                             "manquante (None, NaN, colonne absente) compte 0.")
    for autre_vrai, message in pieges:
        bruite = autre_vrai + _laplace(sensibilite / epsilon, u)
        if _proche(resultat, bruite) and not _proche(bruite, attendu):
            raise AssertionError(f"{quoi} renvoie {resultat!r} au lieu de {attendu!r} : {message}")
    autres = {
        "echelle_epsilon": vrai + _laplace(sensibilite * epsilon, u),
        "echelle_un": vrai + _laplace(1.0, u),
        "echelle_plafond_oublie": vrai + _laplace(1 / epsilon, u),
        "sans_signe": vrai + _laplace(sensibilite / epsilon, u, "sans_signe"),
        "signe_inverse": vrai + _laplace(sensibilite / epsilon, u, "signe_inverse"),
        "exponentielle": vrai + _laplace(sensibilite / epsilon, u, "exponentielle"),
        "log10": vrai + _laplace(sensibilite / epsilon, u, "log10"),
        "sans_bruit": vrai,
        "arrondi": round(attendu),
    }
    for variante, message in _VARIANTES_BRUIT:
        if variante == "echelle_plafond_oublie" and sensibilite == 1:
            continue
        if variante == "echelle_un" and sensibilite / epsilon == 1:
            continue
        if _proche(resultat, autres[variante]) and not _proche(autres[variante], attendu):
            raise AssertionError(f"{quoi} renvoie {resultat!r} au lieu de {attendu!r} : {message}")
    raise AssertionError(f"{quoi} renvoie {_court(resultat, 40)} au lieu de {attendu!r} = {vrai!r} (le vrai résultat) "
                         f"+ un bruit d'échelle {sensibilite / epsilon!r} tiré avec u = {u}.")


def _restant(budget, quoi: str) -> float:
    try:
        valeur = budget.restant
    except NotImplementedError:
        raise
    except AttributeError:
        raise AssertionError(f"{quoi} : l'objet n'a pas d'attribut `restant` (ce qu'il reste du budget).") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} : lire `restant` lève {_decrire(exc)}.") from None
    if callable(valeur):
        raise AssertionError(f"{quoi} : `restant` est une méthode. C'est un attribut, ou une `@property` (on écrit "
                             "`budget.restant`, sans parenthèses).")
    if not _nombre(valeur):
        raise AssertionError(f"{quoi} : `restant` vaut {_court(valeur, 40)}, un nombre est attendu.")
    return valeur


def _creer(cls, total, tirage, quoi: str):
    budget = _appel(cls, (total, tirage), quoi, lignes_max=None)
    return budget


def _question(budget, methode: str, args: tuple, quoi: str):
    fonction = getattr(budget, methode, None)
    if fonction is None:
        raise AssertionError(f"{quoi} : l'objet n'a pas de méthode `{methode}`.")
    return _appel(fonction, args, quoi, lignes_max=None)


def _doit_refuser(budget, methode: str, args: tuple, tirage: _Tirage, avant: float, quoi: str, attendu=BudgetEpuise,
                  pourquoi: str = "") -> None:
    tires = tirage.tires
    nom = attendu.__name__
    try:
        resultat = getattr(budget, methode)(*args)
    except NotImplementedError:
        raise
    except attendu:
        pass
    except BudgetEpuise:
        raise AssertionError(f"{quoi} lève BudgetEpuise au lieu de ValueError : {pourquoi} C'est une ERREUR de l'appelant "
                             "(ValueError), pas un budget épuisé.") from None
    except RuntimeError as exc:
        if "correcteur" in str(exc):
            raise AssertionError(f"{quoi} fait un tirage alors que la question doit être refusée : une question refusée "
                                 "ne tire rien (vérifiez AVANT de tirer).") from None
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {nom}. {pourquoi}".rstrip()) from None
    except ValueError as exc:
        if attendu is BudgetEpuise:
            raise AssertionError(f"{quoi} lève ValueError({str(exc)[:80]!r}) au lieu de BudgetEpuise : la question est "
                                 "correcte, c'est le budget qui manque. Levez l'exception BudgetEpuise (importée de "
                                 "defis.confidentialite).") from None
        raise
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {nom}. {pourquoi}".rstrip()) from None
    else:
        raise AssertionError(f"{quoi} devrait lever {nom}, mais renvoie {_court(resultat, 40)}. {pourquoi}".rstrip())
    if tirage.tires != tires:
        raise AssertionError(f"{quoi} lève bien {nom}, mais a d'abord fait un tirage : une question refusée ne tire rien "
                             "(vérifiez AVANT de tirer).")
    apres = _restant(budget, quoi)
    if not _proche(apres, avant):
        raise AssertionError(f"{quoi} lève bien {nom}, mais le budget restant est passé de {avant!r} à {apres!r} : une "
                             "question refusée ne coûte rien (vérifiez AVANT de dépenser).")


def _rejouer(cls, operations, total: float, quoi: str) -> None:
    """Joue une suite de questions ; chacune : ('compter' | 'sommer', args, epsilon, vrai, sensibilite, u ou None)."""
    us = [op[5] for op in operations if op[5] is not None]
    tirage = _Tirage(us)
    budget = _creer(cls, total, tirage, f"{quoi} : Budget({total}, tirage)")
    depense = 0.0
    if not _proche(_restant(budget, quoi), total):
        indice = (" Un budget tout neuf a déjà dépensé : la dépense est-elle rangée dans un attribut de CLASSE, partagé "
                  "par tous les budgets ? Rangez-la dans `self`, dans __init__." if _restant(budget, quoi) < total else "")
        raise AssertionError(f"{quoi} : juste après la création, `restant` vaut {_restant(budget, quoi)!r} au lieu de "
                             f"{total!r}.{indice}")
    for numero, (methode, args, epsilon, vrai, sensibilite, u) in enumerate(operations):
        ici = f"{quoi}, question n°{numero} ({methode}, epsilon = {epsilon})"
        avant = _restant(budget, ici)
        if u is None:
            _doit_refuser(budget, methode, args, tirage, avant, ici)
            continue
        tires = tirage.tires
        try:
            r = getattr(budget, methode)(*args)
        except NotImplementedError:
            raise
        except BudgetEpuise:
            if depense + epsilon > total - 1e-6 and depense + epsilon <= total + 1e-9:
                raise AssertionError(f"{ici} lève BudgetEpuise alors qu'il reste {total - depense:.6g} : vous comparez des "
                                     "nombres à virgule exactement. 0.1 + 0.1 + 0.1 vaut 0.30000000000000004 et dépasse "
                                     "0.3 ; comparez à 1e-9 près (`depense + epsilon > total + 1e-9`), ou comptez avec "
                                     "`fractions.Fraction(str(epsilon))`.") from None
            raise AssertionError(f"{ici} lève BudgetEpuise alors qu'il reste {total - depense:.6g} de budget, assez pour "
                                 "cette question.") from None
        except RuntimeError as exc:
            if "correcteur" in str(exc):
                raise AssertionError(f"{ici} fait plusieurs tirages : UN tirage par question acceptée.") from None
            raise AssertionError(f"{ici} lève {_decrire(exc)}.") from None
        except Exception as exc:
            indice = " Une valeur manquante (None, NaN, colonne absente) compte 0." if methode == "sommer" else ""
            raise AssertionError(f"{ici} lève {_decrire(exc)}.{indice}") from None
        if tirage.tires == tires:
            raise AssertionError(f"{ici} ne fait aucun tirage avec la fonction `tirage` reçue : c'est votre seule source "
                                 "de hasard (pas le module random), pour que le bruit soit rejouable.")
        if tirage.tires > tires + 1:
            raise AssertionError(f"{ici} fait {tirage.tires - tires} tirages : UN tirage par question acceptée.")
        _pas_none(r, ici)
        if not _nombre(r):
            raise AssertionError(f"{ici} renvoie {_court(r, 40)} : un nombre (float) est attendu.")
        pieges = _pieges_somme(args[0], args[1], args[2]) if methode == "sommer" else []
        _nommer_bruit(r, vrai, epsilon, sensibilite, u, ici, pieges)
        depense += epsilon
        reste = _restant(budget, ici)
        if not _proche(reste, max(0.0, total - depense)) and not math.isclose(reste, max(0.0, total - depense),
                                                                               abs_tol=1e-9):
            raise AssertionError(f"{ici} : après la question, `restant` vaut {reste!r} au lieu de "
                                 f"{max(0.0, total - depense)!r} (le budget moins tous les epsilons déjà dépensés).")


def _clients_211() -> list:
    return [{"churn": 1, "motif": "impayé", "facture": 64.5}, {"churn": 0, "motif": None, "facture": 89.0},
            {"churn": 1, "motif": "prix", "facture": 2500.0}, {"churn": 1, "motif": "impayé", "facture": -20.0},
            {"churn": 0, "motif": None, "facture": None}, {"churn": 0, "motif": None, "facture": _nan()},
            {"churn": 1, "motif": "impayé"}, {"churn": 0, "motif": None, "facture": 150}]


def verifier_Budget(cls) -> None:
    clients = _clients_211()
    partis = sum(c["churn"] for c in clients)                     # 4
    impayes = sum(c["motif"] == "impayé" for c in clients)        # 3

    def somme(plafond):
        return sum(min(max(c["facture"], 0), plafond) for c in clients if not _manquant(c.get("facture")))

    est_parti = lambda c: c["churn"] == 1             # noqa: E731
    est_impaye = lambda c: c["motif"] == "impayé"     # noqa: E731
    scenarios = [
        ("le budget de l'exemple", 1.0, [("compter", (clients, est_impaye, 0.5), 0.5, impayes, 1, 0.5),
                                         ("compter", (clients, est_parti, 0.25), 0.25, partis, 1, 0.75),
                                         ("compter", (clients, est_parti, 0.5), 0.5, None, None, None)]),
        ("des bruits des deux côtés", 2.0, [("compter", (clients, est_parti, 0.5), 0.5, partis, 1, 0.25),
                                            ("compter", (clients, est_parti, 0.5), 0.5, partis, 1, 0.9),
                                            ("compter", (clients, est_impaye, 1.0), 1.0, impayes, 1, 0.1)]),
        ("trois questions à 0,1 sur un budget de 0,3", 0.3, [
            ("compter", (clients, est_parti, 0.1), 0.1, partis, 1, 0.6),
            ("compter", (clients, est_parti, 0.1), 0.1, partis, 1, 0.3),
            ("compter", (clients, est_parti, 0.1), 0.1, partis, 1, 0.8),
            ("compter", (clients, est_parti, 0.1), 0.1, None, None, None)]),
        ("dix questions à 0,1 sur un budget de 1", 1.0,
         [("compter", (clients, est_parti, 0.1), 0.1, partis, 1, (i + 1) / 11) for i in range(10)]
         + [("compter", (clients, est_parti, 0.05), 0.05, None, None, None)]),
        ("une question trop chère, puis une qui passe", 1.0, [
            ("compter", (clients, est_parti, 0.75), 0.75, partis, 1, 0.7),
            ("compter", (clients, est_parti, 0.5), 0.5, None, None, None),
            ("compter", (clients, est_parti, 0.25), 0.25, partis, 1, 0.2)]),
        ("des sommes plafonnées", 3.0, [
            ("sommer", (clients, "facture", 150, 0.5), 0.5, somme(150), 150, 0.75),
            ("sommer", (clients, "facture", 100, 1.0), 1.0, somme(100), 100, 0.3),
            ("sommer", (clients, "facture", 3000.0, 1.0), 1.0, somme(3000.0), 3000.0, 0.5),
            ("compter", (clients, est_impaye, 0.5), 0.5, impayes, 1, 0.95),
            ("sommer", (clients, "facture", 150, 0.25), 0.25, None, None, None)]),
        ("une somme sur une colonne absente", 1.0, [("sommer", (clients, "remise", 50, 1.0), 1.0, 0, 50, 0.6)]),
    ]
    for nom, total, operations in scenarios:
        _rejouer(cls, operations, total, f"Budget : {nom}")
    for total in (0, -1.0, _nan(), math.inf, True, "1.0", None):
        _doit_lever(cls, (total, _Tirage([])), f"Budget({total!r}, tirage)",
                    "epsilon_total doit être un nombre > 0 (ni NaN, ni l'infini, ni un booléen) → sinon ValueError.")
    for epsilon in (0, -0.5, _nan(), math.inf, True, "0.5"):
        tirage = _Tirage([])
        budget = _creer(cls, 1.0, tirage, "Budget(1.0, tirage)")
        _doit_refuser(budget, "compter", (clients, est_parti, epsilon), tirage, _restant(budget, "Budget(1.0, tirage)"),
                      f"Budget(1.0, tirage).compter(clients, condition, {epsilon!r})", ValueError,
                      "epsilon doit être un nombre > 0 (ni NaN, ni l'infini, ni un booléen).")
    for plafond in (0, -10, _nan(), "100"):
        tirage = _Tirage([])
        budget = _creer(cls, 1.0, tirage, "Budget(1.0, tirage)")
        _doit_refuser(budget, "sommer", (clients, "facture", plafond, 0.5), tirage,
                      _restant(budget, "Budget(1.0, tirage)"),
                      f"Budget(1.0, tirage).sommer(clients, 'facture', {plafond!r}, 0.5)", ValueError,
                      "plafond doit être un nombre > 0.")
    # Deux budgets ne partagent rien (un attribut de CLASSE serait partagé par tous les objets).
    t1, t2 = _Tirage([0.5]), _Tirage([])
    b1 = _creer(cls, 1.0, t1, "Budget(1.0, tirage)")
    b2 = _creer(cls, 1.0, t2, "Budget(1.0, autre_tirage)")
    _question(b1, "compter", (clients, est_parti, 0.75), "b1.compter(clients, condition, 0.75)")
    if not _proche(_restant(b2, "b2"), 1.0):
        raise AssertionError("Deux budgets partagent leur dépense : après une question posée au PREMIER, le second n'a "
                             f"plus que {_restant(b2, 'b2')!r}. Rangez la dépense dans `self` (dans __init__), pas dans un "
                             "attribut de classe.")
    alea = random.Random(211)
    for essai in range(120):
        total = alea.choice([0.3, 0.5, 1.0, 1.5, 2.0])
        depense, operations = 0.0, []
        for _ in range(alea.randint(1, 8)):
            epsilon = alea.choice([0.05, 0.1, 0.2, 0.25, 0.5, 1.0])
            refusee = depense + epsilon > total + 1e-9
            u = None if refusee else alea.randint(1, 1023) / 1024
            if alea.random() < 0.6:
                vrai = partis if alea.random() < 0.5 else impayes
                condition = est_parti if vrai == partis else est_impaye
                operations.append(("compter", (clients, condition, epsilon), epsilon, vrai, 1, u))
            else:
                plafond = alea.choice([50, 100, 150.0, 1000])
                operations.append(("sommer", (clients, "facture", plafond, epsilon), epsilon, somme(plafond), plafond, u))
            if not refusee:
                depense += epsilon
        _rejouer(cls, operations, total, f"Budget : suite de questions tirée au hasard n°{essai} (budget {total})")


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"preparer_export": 206, "reidentifier": 207, "audit": 208, "generaliser": 209, "secret_statistique": 210,
            "Budget": 211}

SERIE = Serie(
    code="34",
    titre="Partager sans exposer : pseudonymiser, recouper, auditer, généraliser, secret statistique, bruit",
    module_eleve="defis.serie_34",
    module_corrige="defis.corrige_serie_34",
    defis=(
        Defi(206, "preparer_export", "🟢", "—", verifier_preparer_export),
        Defi(207, "reidentifier", "🟡", "—", verifier_reidentifier),
        Defi(208, "audit", "🟢", "1.2", verifier_audit),
        Defi(209, "generaliser", "🔴", "—", verifier_generaliser),
        Defi(210, "secret_statistique", "🟡", "—", verifier_secret_statistique),
        Defi(211, "Budget", "🔴", "—", verifier_Budget),
    ),
)

"""Vérifications automatiques de la série 30 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Ici, le correcteur est Awa, de l'équipe de rétention, devant la barre de recherche, et le serveur qui
compte ce que chaque recherche lui coûte :

- aux défis 183 et 184, il vous donne des `Comptee` (des listes qui comptent les cases lues) de centaines
  de milliers d'identifiants ou de mots. Il connaît le nombre de lectures qu'une dichotomie demande, et
  refuse un code qui lit tout, quelle que soit la vitesse de la machine ;
- au défi 187, il compte les lignes exécutées dans votre fichier sur un texte de 20 000 mots : essayer
  chaque début de passage se voit ;
- partout, il rejoue les erreurs classiques (`str.split()` au lieu de `mots()`, liste non triée, doublon,
  `defaultdict`, union au lieu d'intersection, `bisect_right`, fin de plage « préfixe + z »,
  `reverse=True` et ses ex aequo, `str.replace`, positions décalées, `math.log10`, idf négatif, moyenne
  des seules notes non vides, tf compté une fois, numéros de mots au lieu de positions…) pour nommer la
  vôtre ;
- et il vérifie que vos fonctions ne modifient pas ce qu'elles reçoivent.
"""

from __future__ import annotations

import bisect
import collections
import copy
import math
import random
import sys

from .recherche import (
    Comptee,
    generer_notes,
    generer_vocabulaire,
    index_frequences,
    mots,
    mots_et_positions,
    normaliser,
    notes_exemple,
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


def _appel(f, args: tuple, quoi: str, conseil: str = "", lignes_max: int | None = 2_000_000, conseil_lent: str = "",
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
        raise AssertionError(f"{quoi} lève RecursionError : Python s'arrête vers 1 000 appels imbriqués. Remplacez "
                             f"la récursion par une boucle. {conseil}".rstrip()) from None
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
    if avant != apres:
        raise AssertionError(f"{quoi} a modifié {nom} reçu (avant : {_court(avant, 100)}, après : {_court(apres, 100)}). "
                             "Une fonction ne touche pas à ce qu'on lui prête : travaillez sur une copie, ou "
                             "construisez un nouvel objet.")


def _pas_none(r, quoi: str) -> None:
    if r is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")


# ---------------------------------------------------------------------------- défi 182 : index_inverse


def _ref_index(notes: dict) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for identifiant in sorted(notes):
        texte = notes[identifiant]
        if isinstance(texte, str):
            for mot in set(mots(texte)):
                index.setdefault(mot, []).append(identifiant)
    return index


def _comparer_index(r, attendu: dict, notes: dict, quoi: str) -> None:
    _pas_none(r, quoi)
    if isinstance(r, collections.defaultdict):
        raise AssertionError(f"{quoi} renvoie un defaultdict. Lire un mot absent (index['wifi']) y AJOUTE une entrée "
                             "vide : l'index grossirait à chaque recherche ratée, et « 'wifi' in index » deviendrait "
                             "vrai. Renvoyez un vrai dict : `return dict(index)`.")
    if not isinstance(r, dict):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : un dict {{mot: [identifiants]}} est attendu.")
    for mot, ids in r.items():
        if not isinstance(ids, list):
            conseil = (" Un set n'a pas d'ordre : on ne pourra pas y chercher par dichotomie (défi 183)."
                       if isinstance(ids, (set, frozenset)) else "")
            raise AssertionError(f"Dans {quoi}, la valeur du mot {mot!r} est un {type(ids).__name__} ({_court(ids, 60)}) : "
                                 f"une list d'identifiants triés est attendue.{conseil}")
    en_trop = [m for m in r if m not in attendu]
    if en_trop:
        mot = en_trop[0]
        if mot in ("none", "nan", "None", "NaN", "<NA>", "na"):
            raise AssertionError(f"{quoi} contient le mot {mot!r} : str(None) vaut 'None' et str(nan) vaut 'nan'. Une note "
                                 "qui n'est pas un texte (case vide d'un CSV) ne contient AUCUN mot : sautez-la "
                                 "(`if not isinstance(texte, str): continue`).")
        if not isinstance(mot, str) or mots(mot) != [mot]:
            raise AssertionError(f"{quoi} contient la clé {mot!r}, qui n'est pas un mot de `mots(texte)` : vous découpez "
                                 "vous-même (split, lower…). Utilisez mots(texte) : minuscules, sans accents, sans "
                                 "ponctuation. La requête passera par la même fonction : la clé "
                                 f"{mot!r} ne serait jamais trouvée.")
        raise AssertionError(f"{quoi} contient le mot {mot!r}, qui n'est dans aucune note.")
    manquants = [m for m in attendu if m not in r]
    if manquants:
        mot = manquants[0]
        raise AssertionError(f"{quoi} n'a pas le mot {mot!r} (présent dans les notes {attendu[mot]}). Prenez TOUS les "
                             "mots de `mots(texte)`, pour chaque note.")
    for mot, ids in attendu.items():
        recu = r[mot]
        if recu == ids:
            continue
        doublons = [i for i, n in collections.Counter(recu).items() if n > 1]
        if doublons:
            raise AssertionError(f"Dans {quoi}, la note {doublons[0]} apparaît {recu.count(doublons[0])} fois pour le mot "
                                 f"{mot!r} ({_court(recu, 80)}) : un mot répété dans une note ne la compte qu'une fois "
                                 "(set(mots(texte))).")
        if sorted(recu) == ids:
            raise AssertionError(f"Dans {quoi}, le mot {mot!r} donne {_court(recu, 80)} : pas trié. Les notes arrivent "
                                 "dans le désordre ; parcourez-les par identifiant croissant (`for i in sorted(notes)`), "
                                 "ou triez chaque liste à la fin.")
        raise AssertionError(f"Dans {quoi}, le mot {mot!r} donne {_court(recu, 80)} au lieu de {_court(ids, 80)}.")


def _un_index(f, notes: dict, quoi: str, conseil: str = "") -> None:
    avant = copy.deepcopy(notes)
    r = _appel(f, (notes,), quoi, conseil)
    _intact(avant, notes, quoi, "le dict des notes")
    _comparer_index(r, _ref_index(notes), notes, quoi)


def verifier_index_inverse(f) -> None:
    _un_index(f, notes_exemple(), "index_inverse(notes_exemple())")
    _un_index(f, {5: "fibre Fibre FIBRE !", 2: "fibre"}, 'index_inverse({5: "fibre Fibre FIBRE !", 2: "fibre"})')
    _un_index(f, {30: "Résilier.", 4: "RÉSILIER ?", 12: "résilier, vite"},
              'index_inverse({30: "Résilier.", 4: "RÉSILIER ?", 12: "résilier, vite"})')
    _un_index(f, {}, "index_inverse({})")
    _un_index(f, {1: "", 2: "?!", 3: "   "}, 'index_inverse({1: "", 2: "?!", 3: "   "})')
    _un_index(f, {3: "fibre", 1: None, 2: float("nan"), 4: "Fibre en panne"},
              'index_inverse({3: "fibre", 1: None, 2: nan, 4: "Fibre en panne"})',
              "Une note None ou NaN (une case vide d'un CSV) ne doit pas faire planter : elle ne contient aucun mot. "
              "Sautez-la : `if not isinstance(texte, str): continue`.")
    for graine in (182, 1182):
        _un_index(f, generer_notes(1500, graine=graine), f"index_inverse(generer_notes(1500, graine={graine}))")


# ---------------------------------------------------------------------------- défi 183 : rechercher


def _ref_rechercher(index: dict, requete: str) -> list[int]:
    termes = set(mots(requete))
    if not termes or any(t not in index for t in termes):
        return []
    return sorted(set.intersection(*(set(_valeurs(index[t])) for t in termes)))


def _valeurs(liste) -> list:
    return list(liste._valeurs) if isinstance(liste, Comptee) else list(liste)


def _budget_rechercher(tailles: list[int]) -> int:
    """Ce qu'une dichotomie lit : la plus courte en entier, puis ~log2(n) + 2 cases par candidat et par liste."""
    if not tailles:
        return 30
    s = sorted(tailles)
    return 3 * (s[0] + s[0] * sum(math.ceil(math.log2(t + 1)) + 2 for t in s[1:])) + 30


def _une_recherche(f, index: dict, requete: str, quoi: str) -> list:
    avant = {m: (id(v), _valeurs(v)) for m, v in index.items()}
    r = _appel(f, (index, requete), quoi,
               "Un mot absent de l'index → [] : utilisez index.get(mot) (ou `mot in index`) plutôt que index[mot]."
               if requete and any(t not in index for t in mots(requete)) else
               "Les listes de l'index se lisent mais ne se modifient pas (ni .sort(), ni .pop()) : pour commencer "
               "par la plus courte, triez la liste DES listes : sorted(listes, key=len).")
    apres = {m: (id(v), _valeurs(v)) for m, v in index.items()}
    if apres != avant:
        raise AssertionError(f"{quoi} a modifié l'index reçu (une liste remplacée, triée ou vidée). L'index sert à "
                             "TOUTES les recherches suivantes : lisez-le sans y toucher.")
    _pas_none(r, quoi)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list d'identifiants triés est attendue.")
    if any(r is v for v in index.values()):
        raise AssertionError(f"{quoi} renvoie la liste de l'index ELLE-MÊME : l'appelant qui la modifierait abîmerait "
                             "l'index. Renvoyez une nouvelle liste (list(…)).")
    attendu = _ref_rechercher(index, requete)
    if r == attendu:
        return r
    termes = set(mots(requete))
    if not termes:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : une requête sans aucun mot → [] (rien n'a été demandé).")
    if len(set(r)) < len(r):
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)}, avec des doublons : chaque note une seule fois.")
    if sorted(r) == attendu:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : pas trié. Les identifiants vont par ordre croissant.")
    union = sorted(set().union(*(set(_valeurs(index.get(t, []))) for t in termes)))
    if sorted(r) == union and union != attendu:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)} : c'est l'UNION (les notes "
                             "qui ont AU MOINS un mot). Il faut celles qui ont TOUS les mots : l'intersection.")
    mots_split = set(requete.lower().split())
    if mots_split != termes:
        avec_split = ([] if not mots_split or any(t not in index for t in mots_split)
                      else sorted(set.intersection(*(set(_valeurs(index[t])) for t in mots_split))))
        if sorted(r) == avec_split:
            raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)} : découpez la requête "
                                 "avec mots(requete), comme les notes. « FIBRE » ou « résilier, » ne sont pas des mots "
                                 "de l'index.")
    raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)}.")


def _recherche_comptee(f, listes: dict[str, list[int]], requete: str, quoi: str) -> None:
    index = {mot: Comptee(ids) for mot, ids in listes.items()}
    _une_recherche(f, index, requete, quoi)
    presents = [index[t] for t in set(mots(requete)) if t in index]
    if len(presents) < len(set(mots(requete))):
        presents = []                          # un mot absent : la réponse est [] sans rien lire
    lues = sum(liste.lues for liste in index.values())
    budget = _budget_rechercher([len(liste) for liste in presents])
    if lues <= budget:
        return
    mot, liste = max(index.items(), key=lambda paire: paire[1].lues)
    if liste.lues > len(liste):
        detail = f"dont {liste.lues} dans la liste de {mot!r}, qui n'a que {len(liste)} identifiants"
    else:
        detail = f"dont {liste.lues} sur {len(liste)} dans la liste de {mot!r}"
    if not presents:
        conseil = ("Un mot de la requête est absent de l'index : la réponse est [] AVANT de lire quoi que ce soit. "
                   "Vérifiez d'abord que chaque mot est dans l'index.")
    elif liste.lues > 2 * len(liste):
        conseil = ("Vous RELISEZ la même liste à chaque tour : un `x in liste` dans une boucle la parcourt depuis le "
                   "début, à chaque fois. Partez de la liste la plus COURTE, et cherchez chacun de ses identifiants dans "
                   "les autres par dichotomie : bisect.bisect_left(liste, candidat, depart).")
    elif liste.lues >= 0.9 * len(liste) and len(liste) > min(len(p) for p in presents):
        conseil = ("Vous lisez TOUTE une liste longue : un set(), list() ou un parcours des deux listes en parallèle "
                   "lisent chaque case. Partez de la liste la plus COURTE, et cherchez chacun de ses identifiants dans "
                   "les autres par dichotomie : bisect.bisect_left(liste, candidat, depart).")
    else:
        conseil = ("Partez de la liste la plus COURTE (len() est gratuit) : chacun de ses identifiants est un candidat, "
                   "cherché dans les autres par dichotomie (bisect.bisect_left).")
    raise AssertionError(f"{quoi} a lu {lues} cases de l'index ({detail}), alors que {budget} suffisent. {conseil}")


def verifier_rechercher(f) -> None:
    index = _ref_index(notes_exemple())
    for requete in ("résilier FIBRE", "fibre", "Client résilier mobile", "RÉSILIER, client.", "fibre résiliation",
                    "fibre wifi", "wifi", "?!", "", "fibre fibre FIBRE", "la fibre"):
        _une_recherche(f, index, requete, f"rechercher(index_inverse(notes_exemple()), {requete!r})")

    alea = random.Random(183)
    notes = generer_notes(1500, graine=183)
    index = _ref_index(notes)
    frequents = sorted(index, key=lambda m: -len(index[m]))
    vocabulaire = frequents[:30] + alea.sample(frequents[30:], 30)
    for _ in range(60):
        requete = " ".join(alea.choice(vocabulaire).upper() if alea.random() < 0.2 else alea.choice(vocabulaire)
                           for _ in range(alea.randint(1, 4)))
        _une_recherche(f, index, requete, f"rechercher(index_inverse(generer_notes(1500, graine=183)), {requete!r})")

    grand = list(range(0, 400_000, 2))
    _recherche_comptee(f, {"client": grand, "concurrent": [14, 77_777, 150_000, 300_001, 399_998]},
                       "client concurrent", "rechercher({'client': 200 000 identifiants, 'concurrent': 5}, "
                                            "'client concurrent')")
    _recherche_comptee(f, {"client": grand, "concurrent": [14, 77_777, 150_000, 300_001, 399_998]},
                       "concurrent client", "rechercher({'client': 200 000 identifiants, 'concurrent': 5}, "
                                            "'concurrent client')")
    _recherche_comptee(f, {"client": grand, "fibre": list(range(0, 400_000, 3)), "resilier": list(range(0, 400_000, 4)),
                           "concurrent": list(range(7, 400_000, 4_007))},
                       "client fibre résilier concurrent",
                       "rechercher(index de 4 listes : 200 000, 133 334, 100 000 et 100 identifiants, "
                       "'client fibre résilier concurrent')")
    _recherche_comptee(f, {"client": grand, "fibre": list(range(0, 400_000, 3))}, "client wifi fibre",
                       "rechercher({'client': 200 000 identifiants, 'fibre': 133 334}, 'client wifi fibre')")
    a = sorted(alea.sample(range(40_000), 5_000))
    b = sorted(alea.sample(range(40_000), 5_000))
    _recherche_comptee(f, {"fibre": a, "mobile": b}, "fibre mobile",
                       "rechercher({'fibre': 5 000 identifiants, 'mobile': 5 000}, 'fibre mobile')")


# ---------------------------------------------------------------------------- défi 184 : completer


def _commencent_par(vocabulaire, p: str) -> list[str]:
    """Les mots du vocabulaire TRIÉ qui commencent par p (tests/test_defis_serie_30.py le compare à un parcours complet)."""
    valeurs = _valeurs(vocabulaire)
    i = bisect.bisect_left(valeurs, p)
    j = i
    while j < len(valeurs) and valeurs[j].startswith(p):
        j += 1
    return valeurs[i:j]


def _ref_completer(vocabulaire, frequences: dict, prefixe: str, k: int = 5) -> list[str]:
    p = normaliser(prefixe).strip()
    if k <= 0:
        return []
    return sorted(_commencent_par(vocabulaire, p), key=lambda m: (-frequences.get(m, 0), m))[:k]


def _une_completion(f, vocabulaire, frequences: dict, prefixe: str, k, quoi: str, intact: bool = True) -> list:
    avant = (_valeurs(vocabulaire), copy.deepcopy(frequences)) if intact else None
    r = _appel(f, (vocabulaire, frequences, prefixe, k), quoi)
    if intact:
        _intact(avant, (_valeurs(vocabulaire), frequences), quoi, "le vocabulaire ou les fréquences")
    _pas_none(r, quoi)
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list de mots est attendue.")
    attendu = _ref_completer(vocabulaire, frequences, prefixe, k)
    if r == attendu:
        return r
    p = normaliser(prefixe).strip()
    tous = sorted(_commencent_par(vocabulaire, p), key=lambda m: (-frequences.get(m, 0), m))
    if k <= 0:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : k <= 0 → []. Attention, liste[:-2] n'est pas vide : "
                             "elle enlève les deux derniers.")
    if len(r) > k:
        raise AssertionError(f"{quoi} renvoie {len(r)} mots : au plus k = {k}.")
    if p and not r and attendu:
        if prefixe != p and not _commencent_par(vocabulaire, prefixe):
            raise AssertionError(f"{quoi} ne trouve rien, alors que {_court(attendu, 80)} commencent par {p!r}. Normalisez "
                                 f"le préfixe comme les mots (normaliser(prefixe).strip()) : le vocabulaire est en "
                                 "minuscules, sans accents.")
    if p in _valeurs(vocabulaire) and p in attendu and p not in r:
        raise AssertionError(f"{quoi} oublie le mot {p!r}, égal au préfixe lui-même : il commence bien par {p!r}. "
                             "bisect_right saute le mot égal ; prenez bisect.bisect_left.")
    if r == sorted(tous, key=lambda m: (frequences.get(m, 0), m))[:k]:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)} : les plus FRÉQUENTS "
                             "d'abord (fréquence décroissante).")
    inverse = sorted(tous, key=lambda m: (frequences.get(m, 0), m), reverse=True)[:k]
    if r == inverse:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)} : avec reverse=True, les ex "
                             "aequo sortent dans l'ordre alphabétique INVERSE. Ne retournez que la fréquence : "
                             "key=lambda mot: (-frequences[mot], mot).")
    manques = [m for m in tous if m not in r and m in attendu]
    if manques:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)} : le mot {manques[0]!r} "
                             f"commence pourtant par {p!r}. La fin de la plage n'est pas « préfixe + 'z' » (« raz » et "
                             "« razzia » viennent après « raz »), ni « préfixe + '~' » (« cœur » vient après « c~ ») : "
                             "avancez tant que mot.startswith(prefixe).")
    raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de {_court(attendu, 80)}.")


def verifier_completer(f) -> None:
    index = _ref_index(notes_exemple())
    vocabulaire, frequences = sorted(index), {mot: len(ids) for mot, ids in index.items()}
    for prefixe, k in (("rési", 5), ("Rési", 5), ("co", 3), ("CO", 10), ("  rési ", 5), ("c", 5), ("wifi", 5),
                       ("fibre", 5), ("", 4), ("co", 0), ("co", -2), ("RÉSILIATION", 5)):
        _une_completion(f, vocabulaire, frequences, prefixe, k,
                        f"completer(vocabulaire de notes_exemple(), frequences, {prefixe!r}, k={k})")

    mots_choisis = {"ra": 1, "rabais": 4, "rappel": 9, "raz": 7, "razzia": 8, "rb": 50, "q": 50, "c": 2,
                    "cable": 3, "client": 6, "cœur": 12, "czar": 5, "d": 1}
    vocabulaire = sorted(mots_choisis)
    for prefixe, k in (("ra", 3), ("ra", 10), ("c", 2), ("c", 10), ("raz", 5), ("r", 2)):
        _une_completion(f, vocabulaire, mots_choisis, prefixe, k, f"completer({vocabulaire}, …, {prefixe!r}, k={k})")

    grand, frequences = generer_vocabulaire(100_000, graine=184)
    copie = dict(frequences)
    alea = random.Random(184)
    prefixes = [grand[alea.randrange(len(grand))][:n] for n in (2, 4, 6, 8)] + ["zzz", "ba", "tra"]
    for prefixe in prefixes:
        vocabulaire = Comptee(grand)           # en lecture seule : il ne peut pas être modifié
        quoi = f"completer(generer_vocabulaire(100_000, graine=184), …, {prefixe!r}, k=5)"
        _une_completion(f, vocabulaire, frequences, prefixe, 5, quoi, intact=False)
        _intact(copie, frequences, quoi, "le dict des fréquences")
        trouves = len(_commencent_par(grand, prefixe))
        budget = 3 * (math.ceil(math.log2(len(grand) + 1)) + 2 + trouves) + 20
        if vocabulaire.lues > budget:
            raise AssertionError(
                f"{quoi} a lu {vocabulaire.lues} mots sur {len(grand)} pour {trouves} mots qui commencent par {prefixe!r} "
                f"(une dichotomie et ces {trouves} mots : {budget} lectures au plus). Le vocabulaire est TRIÉ : sautez "
                "directement au premier candidat avec bisect.bisect_left(vocabulaire, prefixe), puis arrêtez-vous au "
                "premier mot qui ne commence plus par le préfixe.")


# ---------------------------------------------------------------------------- défi 185 : surligner


def _ref_surligner(texte: str, requete: str, avant: str = "[", apres: str = "]") -> str:
    termes = set(mots(requete))
    morceaux, lu = [], 0
    for mot, debut, fin in mots_et_positions(texte):
        if mot in termes:
            morceaux += [texte[lu:debut], avant, texte[debut:fin], apres]
            lu = fin
    return "".join(morceaux) + texte[lu:]


def _un_surlignage(f, texte: str, requete: str, quoi: str, avant: str | None = None, apres: str | None = None) -> None:
    args = (texte, requete) if avant is None else (texte, requete, avant, apres)
    r = _appel(f, args, quoi)
    _pas_none(r, quoi)
    avant, apres = ("[", "]") if avant is None else (avant, apres)
    attendu = _ref_surligner(texte, requete, avant, apres)
    if r == attendu:
        return
    if not isinstance(r, str):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : le texte surligné (un str) est attendu.")
    termes = set(mots(requete))
    if not termes:
        raise AssertionError(f"{quoi} renvoie {r!r} : une requête sans aucun mot laisse le texte inchangé ({texte!r}).")
    if normaliser(r) == normaliser(attendu) or r.lower() == attendu.lower():
        raise AssertionError(f"{quoi} renvoie {r!r} au lieu de {attendu!r} : recopiez le texte D'ORIGINE, tel qu'il est "
                             "écrit (texte[debut:fin]), pas sa forme normalisée.")
    if len(r) == len(attendu) and r.count(avant) == attendu.count(avant):
        raise AssertionError(f"{quoi} renvoie {r!r} au lieu de {attendu!r} : les marques sont décalées. Insérer dans le "
                             "texte décale toutes les positions qui suivent. Construisez un nouveau texte morceau par "
                             "morceau, de gauche à droite (ou insérez en partant de la FIN).")
    if "\u0301" in texte:
        raise AssertionError(f"{quoi} renvoie {r!r} au lieu de {attendu!r}. Ce texte écrit « é » en DEUX caractères "
                             "(« e » suivi de l'accent U+0301) : une regex \\w+ coupe le mot en deux. Les positions de "
                             "mots_et_positions(texte), elles, comptent l'accent dans le mot.")
    if avant and r.count(avant) > attendu.count(avant):
        raise AssertionError(f"{quoi} renvoie {r!r} au lieu de {attendu!r} : vous surlignez aussi l'intérieur d'un mot. "
                             "str.replace remplace PARTOUT (« de » dans « Demande », « fibre » dans « fibres ») : "
                             "surlignez les mots ENTIERS que donne mots_et_positions(texte).")
    if avant and r.count(avant) < attendu.count(avant):
        raise AssertionError(f"{quoi} renvoie {r!r} au lieu de {attendu!r} : il manque des mots surlignés. Un mot est "
                             "surligné quand sa forme NORMALISÉE (sans accents, en minuscules) est un mot de la requête. "
                             "str.replace ou une regex comparent le texte BRUT, où « RÉSILIER » n'est pas « resilier » : "
                             "parcourez mots_et_positions(texte).")
    raise AssertionError(f"{quoi} renvoie {r!r} au lieu de {attendu!r}.")


def verifier_surligner(f) -> None:
    notes = notes_exemple()
    cas = [
        (notes[30], "fibre résilier"),
        (notes[17], "FIBRE, résilier"),
        ("Demande de remise", "de"),
        ("fibres et fibre", "fibre"),
        ("La FIBRE, la fibre ; la Fibre !", "fibre"),
        ("fibre coupée", "fibre coupée"),
        ("😡 FIBRE en panne 😡", "fibre panne"),
        (notes[8], "l'offre"),
        ("Rien à signaler.", ""),
        ("Rien à signaler.", "?!"),
        ("", "fibre"),
        ("Résilier vite, résilier !", "résilier"),
    ]
    for texte, requete in cas:
        _un_surlignage(f, texte, requete, f"surligner({texte!r}, {requete!r})")
    for texte, requete, avant, apres in (("Résiliation ? Non : RÉSILIER, puis résilié.", "resilier", "<mark>", "</mark>"),
                                          (notes[30], "la fibre coupure", "<b>", "</b>"),
                                          ("abc abc abc", "abc", "", "")):
        _un_surlignage(f, texte, requete, f"surligner({texte!r}, {requete!r}, {avant!r}, {apres!r})", avant, apres)
    _doit_lever(f, (None, "fibre"), "surligner(None, 'fibre')", "Un texte qui n'est pas un str → TypeError "
                "(mots_et_positions la lève pour vous).", attendu=TypeError)

    alea = random.Random(185)
    for identifiant, texte in list(generer_notes(200, graine=185).items())[:120]:
        candidats = mots(texte) or ["fibre"]
        requete = " ".join(alea.choice(candidats).upper() for _ in range(alea.randint(1, 3)))
        _un_surlignage(f, texte, requete, f"surligner(generer_notes(200, graine=185)[{identifiant}], {requete!r})",
                       "<<", ">>")


# ---------------------------------------------------------------------------- défi 186 : classer (BM25)


def _ref_classer(requete: str, frequences: dict, longueurs: dict, k: int = 10, k1: float = 1.2, b: float = 0.75,
                 variante: str | None = None) -> list[tuple[int, float]]:
    if variante == "split":
        termes = list(dict.fromkeys(requete.lower().split()))
    elif variante == "doublons":
        termes = mots(requete)
    else:
        termes = list(dict.fromkeys(mots(requete)))
    n = len(longueurs)
    if k <= 0 or n == 0:
        return []
    candidats = {i for t in termes for i in frequences.get(t, {})}
    if variante == "moyenne_non_vides":
        non_vides = [x for x in longueurs.values() if x > 0]
        moyenne = sum(non_vides) / len(non_vides) if non_vides else 0
    elif variante == "moyenne_candidats":
        moyenne = sum(longueurs[i] for i in candidats) / len(candidats) if candidats else 0
    else:
        moyenne = sum(longueurs.values()) / n
    if moyenne == 0:
        return []
    scores: dict[int, float] = {}
    if variante == "tous":
        scores = {i: 0.0 for i in longueurs}
    for terme in termes:
        notes = frequences.get(terme)
        if not notes:
            continue
        df = len(notes)
        if variante == "log10":
            idf = math.log10(1 + (n - df + 0.5) / (df + 0.5))
        elif variante == "idf_classique":
            idf = math.log((n - df + 0.5) / (df + 0.5))
        elif variante == "idf_tfidf":
            idf = math.log(n / df)
        else:
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, tf in notes.items():
            if variante == "presence":
                tf = 1
            dl = longueurs[i]
            rapport = moyenne / dl if variante == "longueur_inversee" else dl / moyenne
            if variante == "sans_longueur":
                rapport = 1.0
            numerateur = idf * tf * (1 if variante == "sans_k1_plus_1" else k1 + 1)
            scores[i] = scores.get(i, 0.0) + numerateur / (tf + k1 * (1 - b + b * rapport))
    cle = (lambda p: (-p[1], -p[0])) if variante == "egalites_inverse" else (lambda p: (-p[1], p[0]))
    return sorted(scores.items(), key=cle)[:k]


_VARIANTES_BM25 = {
    "log10": "Vous utilisez math.log10 : l'idf de BM25 est un logarithme NATUREL, math.log.",
    "idf_classique": "Vous utilisez l'idf classique ln((N - df + 0.5) / (df + 0.5)) : pour un mot présent dans plus de "
                     "la moitié des notes (« client »), il est NÉGATIF, et contenir le mot fait BAISSER le score. "
                     "L'idf de Lucene ajoute 1 dans le logarithme : ln(1 + (N - df + 0.5) / (df + 0.5)).",
    "idf_tfidf": "Vous utilisez l'idf de TF-IDF, ln(N / df) : celui de BM25 est ln(1 + (N - df + 0.5) / (df + 0.5)).",
    "moyenne_non_vides": "Votre longueur moyenne ignore les notes vides : elle se calcule sur TOUTES les notes, vides "
                         "comprises (sum(longueurs.values()) / len(longueurs)).",
    "moyenne_candidats": "Votre longueur moyenne n'est calculée que sur les notes qui contiennent un mot de la requête : "
                         "c'est la moyenne de TOUTES les notes (sum(longueurs.values()) / len(longueurs)).",
    "longueur_inversee": "La longueur est à l'envers : c'est longueur / moyenne (une note plus LONGUE que la moyenne est "
                         "un peu pénalisée), pas moyenne / longueur.",
    "sans_longueur": "La longueur des notes n'est pas prise en compte : le dénominateur est "
                     "tf + k1 × (1 - b + b × longueur / moyenne).",
    "presence": "Vous comptez chaque mot une seule fois par note (tf = 1) : tf est le NOMBRE de fois où le mot est dans "
                "la note, frequences[mot][identifiant].",
    "sans_k1_plus_1": "Il manque le facteur (k1 + 1) au numérateur : l'ordre est le bon, mais ce ne sont pas les scores "
                      "de la formule demandée (un seuil de pertinence fixé sur ces scores serait faux).",
    "doublons": "Un mot répété dans la requête (« client client ») compte plusieurs fois chez vous : prenez les mots "
                "DISTINCTS, list(dict.fromkeys(mots(requete))).",
    "egalites_inverse": "À score égal, rangez par identifiant CROISSANT : key=lambda p: (-p[1], p[0]). reverse=True "
                        "inverserait aussi l'ordre des identifiants.",
    "split": "Découpez la requête avec mots(requete) : « RÉSILIER, » n'est pas un mot de l'index.",
    "tous": "Seules les notes qui contiennent au moins un mot de la requête sont classées (pas les autres, avec un "
            "score de 0) : ne parcourez que frequences[mot], pas toutes les notes.",
}


def _meme_classement(r, attendu) -> bool:
    if not isinstance(r, list) or len(r) != len(attendu):
        return False
    for paire, (i, s) in zip(r, attendu):
        try:
            ri, rs = paire
        except (TypeError, ValueError):
            return False
        if ri != i or not isinstance(rs, (int, float)) or not math.isclose(rs, s, rel_tol=1e-9, abs_tol=1e-12):
            return False
    return True


def _un_classement(f, requete: str, frequences: dict, longueurs: dict, quoi: str, **parametres) -> None:
    avant = copy.deepcopy((frequences, longueurs))
    r = _appel(f, (requete, frequences, longueurs), quoi, kwargs=parametres)
    _intact(avant, (frequences, longueurs), quoi, "les dict des fréquences ou des longueurs")
    _pas_none(r, quoi)
    attendu = _ref_classer(requete, frequences, longueurs, **parametres)
    if _meme_classement(r, attendu):
        return
    if not isinstance(r, list):
        raise AssertionError(f"{quoi} renvoie un {type(r).__name__} : une list de paires (identifiant, score) est attendue.")
    for paire in r:
        if not isinstance(paire, (tuple, list)) or len(paire) != 2:
            raise AssertionError(f"{quoi} contient {_court(paire, 60)} : chaque élément est une paire (identifiant, score).")
    for variante, message in _VARIANTES_BM25.items():
        autre = _ref_classer(requete, frequences, longueurs, variante=variante, **parametres)
        if not _meme_classement(autre, attendu) and _meme_classement(r, autre):
            raise AssertionError(f"{quoi} renvoie {_court(r, 120)} au lieu de {_court(attendu, 120)}. {message}")
    k = parametres.get("k", 10)
    if len(r) > max(k, 0):
        raise AssertionError(f"{quoi} renvoie {len(r)} notes : au plus k = {k}.")
    if attendu and not r:
        raise AssertionError(f"{quoi} renvoie [] au lieu de {_court(attendu, 120)}.")
    if not attendu and r:
        raise AssertionError(f"{quoi} renvoie {_court(r, 120)} au lieu de [] (aucun mot de la requête dans les notes, "
                             "k <= 0, ou aucune note).")
    tous = _ref_classer(requete, frequences, longueurs, **{**parametres, "k": 10**9})
    scores = dict(tous)
    for i, s in r:
        if i not in scores:
            raise AssertionError(f"{quoi} classe la note {i!r}, qui ne contient aucun mot de la requête.")
        if not isinstance(s, (int, float)) or not math.isclose(s, scores[i], rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(f"{quoi} donne à la note {i!r} le score {s!r} au lieu de {scores[i]!r}. Relisez la "
                                 "formule morceau par morceau (idf, tf, k1 + 1, la longueur et la moyenne).")
    raise AssertionError(f"{quoi} renvoie {_court(r, 120)} au lieu de {_court(attendu, 120)} : les scores sont justes, "
                         "mais pas l'ordre ou la sélection. Gardez les k MEILLEURS, par score décroissant, puis par "
                         "identifiant croissant à égalité : heapq.nsmallest(k, scores.items(), "
                         "key=lambda p: (-p[1], p[0])).")


def verifier_classer(f) -> None:
    frequences, longueurs = index_frequences(notes_exemple())
    for requete, parametres in (("résilier fibre", {}), ("fibre résilier", {"k": 2}), ("client", {}),
                                ("RÉSILIER, la FIBRE !", {}), ("client client fibre", {}), ("wifi", {}), ("", {}),
                                ("?!", {}), ("résilier fibre", {"k": 50}), ("résilier fibre", {"k": 0}),
                                ("résilier fibre", {"k": -1}), ("coupure fibre", {"k1": 2.0, "b": 0.3})):
        reglages = "".join(f", {cle}={valeur!r}" for cle, valeur in parametres.items())
        _un_classement(f, requete, frequences, longueurs, f"classer({requete!r}, *index_frequences(notes_exemple())"
                                                          f"{reglages})", **parametres)

    egaux = {3: "fibre en panne", 2: "fibre en panne", 1: "panne fibre en", 4: "client"}
    frequences_egaux, longueurs_egaux = index_frequences(egaux)
    _un_classement(f, "fibre", frequences_egaux, longueurs_egaux,
                   "classer('fibre', *index_frequences({3: 'fibre en panne', 2: 'fibre en panne', "
                   "1: 'panne fibre en', 4: 'client'}))")
    _un_classement(f, "fibre", {}, {}, "classer('fibre', {}, {})")
    _un_classement(f, "fibre", {}, {1: 0, 2: 0}, "classer('fibre', {}, {1: 0, 2: 0}) (que des notes vides)")

    notes = generer_notes(1500, graine=186)
    frequences, longueurs = index_frequences(notes)
    for requete, parametres in (("client fibre", {}), ("résilier fibre prix", {"k": 5}), ("client", {"k": 20}),
                                ("Client, CLIENT ; client", {}), ("fibre débit concurrent", {"k1": 2.0, "b": 0.0}),
                                ("fibre débit concurrent", {"k1": 0.0}), ("facture retard", {"b": 1.0, "k": 30}),
                                ("le la de client", {"k": 15})):
        reglages = "".join(f", {cle}={valeur!r}" for cle, valeur in parametres.items())
        _un_classement(f, requete, frequences, longueurs,
                       f"classer({requete!r}, *index_frequences(generer_notes(1500, graine=186)){reglages})",
                       **parametres)

    frequences, longueurs = index_frequences(notes_exemple())
    for parametres in ({"k1": -0.5}, {"k1": float("nan")}, {"b": 1.5}, {"b": -0.1}, {"b": float("nan")}):
        reglages = ", ".join(f"{cle}={valeur!r}" for cle, valeur in parametres.items())
        _doit_lever(f, ("fibre", frequences, longueurs), f"classer('fibre', …, {reglages})",
                    "k1 doit être >= 0 et b entre 0 et 1. Attention : toute comparaison avec NaN est fausse, donc "
                    "`k1 < 0` laisse passer NaN ; écrivez `not k1 >= 0`.", kwargs=parametres)


# ---------------------------------------------------------------------------- défi 187 : meilleur_extrait


def _ref_fenetre(texte: str, requete: str, mesure: str = "mots", egalite: str = "premiere"):
    """(premier mot, dernier mot) du plus court passage, ou None. Écrit en O(n²) exprès : c'est la RÉFÉRENCE."""
    jetons = mots_et_positions(texte)
    presents = set(mots(requete)) & {m for m, _, _ in jetons}
    if not presents:
        return None
    meilleur, taille_meilleure = None, None
    for i, (mot, _, _) in enumerate(jetons):
        if mot not in presents:
            continue
        vus = set()
        for j in range(i, len(jetons)):
            if jetons[j][0] in presents:
                vus.add(jetons[j][0])
            if len(vus) == len(presents):
                taille = j - i if mesure == "mots" else jetons[j][2] - jetons[i][1]
                if (taille_meilleure is None or taille < taille_meilleure
                        or (egalite == "derniere" and taille == taille_meilleure)):
                    meilleur, taille_meilleure = (i, j), taille
                break
    return meilleur


def _ref_extrait(texte: str, requete: str, **options):
    fenetre = _ref_fenetre(texte, requete, **options)
    if fenetre is None:
        return None
    jetons = mots_et_positions(texte)
    return jetons[fenetre[0]][1], jetons[fenetre[1]][2]


def _un_extrait(f, texte: str, requete: str, quoi: str, lignes_max: int | None = 2_000_000, attendu=...) -> None:
    r = _appel(f, (texte, requete), quoi, lignes_max=lignes_max,
               conseil_lent="Votre fonction a été arrêtée : elle exécute beaucoup trop de lignes. Essayer chaque début "
                            "de passage, puis avancer jusqu'à tout trouver, c'est O(n²). Une fenêtre glissante fait "
                            "entrer chaque mot une fois (à droite) et sortir une fois (à gauche) : O(n).")
    if attendu is ...:
        attendu = _ref_extrait(texte, requete)
    if r == attendu and (r is None or isinstance(r, tuple)):
        return
    if attendu is None:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} au lieu de None : aucun mot de la requête n'est dans le "
                             "texte (ou la requête n'a aucun mot).")
    if r is None:
        raise AssertionError(f"{quoi} renvoie None au lieu de {attendu} ({texte[attendu[0]:attendu[1]]!r}). Les mots de "
                             "la requête ABSENTS du texte sont ignorés : le passage contient ceux qui sont présents.")
    if isinstance(r, list) and tuple(r) == attendu:
        raise AssertionError(f"{quoi} renvoie une liste {r} : un tuple (début, fin) est attendu.")
    if not isinstance(r, tuple) or len(r) != 2:
        raise AssertionError(f"{quoi} renvoie {_court(r, 80)} : un tuple (début, fin) est attendu.")
    premier, dernier = _ref_fenetre(texte, requete)
    if r in ((premier, dernier), (premier, dernier + 1)):
        raise AssertionError(f"{quoi} renvoie {r} au lieu de {attendu} : ce sont des numéros de MOTS. Le résultat est une "
                             "position dans le TEXTE : le début du premier mot et la fin du dernier, que donne "
                             "mots_et_positions(texte).")
    if r == (attendu[0], attendu[1] - 1):
        raise AssertionError(f"{quoi} renvoie {r} au lieu de {attendu} : la fin est EXCLUE, comme dans texte[début:fin] "
                             f"({texte[attendu[0]:attendu[1]]!r}) : c'est la fin du dernier mot, pas son dernier caractère.")
    passage = texte[r[0]:r[1]] if all(isinstance(x, int) for x in r) else "?"
    if r == _ref_extrait(texte, requete, mesure="caracteres"):
        raise AssertionError(f"{quoi} renvoie {r} ({passage!r}) au lieu de {attendu} ({texte[attendu[0]:attendu[1]]!r}) : "
                             "« plus court » se compte en MOTS, pas en caractères.")
    if r == _ref_extrait(texte, requete, egalite="derniere"):
        raise AssertionError(f"{quoi} renvoie {r} ({passage!r}) au lieu de {attendu} ({texte[attendu[0]:attendu[1]]!r}) : "
                             "à longueur égale, le PREMIER passage du texte. Ne remplacez le meilleur que par un passage "
                             "strictement plus court (« < », pas « <= »).")
    raise AssertionError(f"{quoi} renvoie {r} ({passage!r}) au lieu de {attendu} ({texte[attendu[0]:attendu[1]]!r}).")


def verifier_meilleur_extrait(f) -> None:
    notes = notes_exemple()
    cas = [
        (notes[30], "fibre résilier"),
        (notes[17], "résilier fibre prix"),
        (notes[17], "wifi"),
        (notes[17], ""),
        (notes[17], "?!"),
        ("", "fibre"),
        ("fibre ok fibre", "fibre"),
        ("a fibre resilier b resilier fibre", "fibre resilier"),
        ("fibre fibre resilier", "FIBRE fibre résilier"),
        ("fibre aaaaaaaaaaaaaaaaaaaa resilier x y fibre", "fibre resilier"),
        ("Résilier la fibre ? Oui, résilier.", "résilier fibre"),
        ("fibre a b c d resilier fibre", "résilier fibre"),
        ("x fibre y résilier z fibre w", "résilier fibre absent"),
        (notes[30], "la fibre coupure client"),
    ]
    for texte, requete in cas:
        _un_extrait(f, texte, requete, f"meilleur_extrait({texte!r}, {requete!r})")

    alea = random.Random(187)
    for essai in range(300):
        texte = " ".join(alea.choice("abcdef") for _ in range(alea.randint(0, 25)))
        requete = " ".join(alea.sample("abcdeg", alea.randint(1, 4)))
        _un_extrait(f, texte, requete, f"meilleur_extrait({texte!r}, {requete!r}) (tiré au hasard n°{essai})")

    remplissage = [alea.choice(["client", "appel", "la", "le", "de", "pour", "mois"]) for _ in range(20_000)]
    for position, mot in ((150, "fibre"), (9_000, "resilier"), (9_500, "concurrent"), (12_000, "fibre"),
                          (12_050, "concurrent"), (19_990, "resilier"), (19_995, "fibre")):
        remplissage[position] = mot
    texte = " ".join(remplissage)
    _un_extrait(f, texte, "fibre résilier concurrent", "meilleur_extrait(un texte de 20 000 mots, "
                "'fibre résilier concurrent')", lignes_max=60 * len(remplissage) + 5_000)


# ---------------------------------------------------------------------------- la série

_NUMEROS = {"index_inverse": 182, "rechercher": 183, "completer": 184, "surligner": 185, "classer": 186,
            "meilleur_extrait": 187}

SERIE = Serie(
    code="30",
    titre="Chercher dans les notes : index inversé, recherche, autocomplétion, surlignage, BM25, extrait",
    module_eleve="defis.serie_30",
    module_corrige="defis.corrige_serie_30",
    defis=(
        Defi(182, "index_inverse", "🟢", "—", verifier_index_inverse),
        Defi(183, "rechercher", "🟡", "—", verifier_rechercher),
        Defi(184, "completer", "🟡", "—", verifier_completer),
        Defi(185, "surligner", "🟢", "—", verifier_surligner),
        Defi(186, "classer", "🔴", "—", verifier_classer),
        Defi(187, "meilleur_extrait", "🔴", "—", verifier_meilleur_extrait),
    ),
)

"""Vérifications automatiques de la série 19 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Toute la série repose sur une règle : pour celui qui utilise le code, RIEN ne doit changer. Le
correcteur vérifie donc surtout des comportements « d'avant » :

- au défi 116, il photographie de vieilles fonctions qui modifient leurs arguments, renvoient une liste
  qu'elles allongent ensuite, ou lèvent des exceptions, et appuie sur Ctrl+C ;
- au défi 117, il appelle `remise` comme le font les 40 anciens appels, et regarde QUELLE ligne
  l'avertissement désigne ;
- au défi 118, il compare votre fonction à l'ancien script (sa logique, recopiée ici) sur des centaines
  de fichiers tirés au hasard, et reconnaît les écarts classiques (seuil, 30 jours, majuscule, budget,
  date du jour) ;
- au défi 120, il compare votre adaptateur à `PasserelleAncienne`, numéro par numéro, et fait tomber le
  nouvel opérateur en panne de toutes les façons prévues… et d'une imprévue ;
- au défi 121, il donne à `en_ombre` des fonctions qui plantent, modifient leurs arguments ou
  s'interrompent, et vérifie que l'utilisateur ne voit jamais la différence.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import io
import json
import random
import warnings
from datetime import date, timedelta

from .heritage import (CLIENTS_EXEMPLE, JOUR_EXEMPLE, LIGNES_EXEMPLE, PREFERENCES_V1_EXEMPLE,
                       PREFERENCES_V2_EXEMPLE, ClientMessagerie, ErreurMessagerie, PasserelleAncienne, Reponse,
                       ancien_score, nouveau_score, remise_ancienne)
from .verif import Defi, Serie


def _court(valeur, n: int = 180) -> str:
    texte = repr(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


def _decrire(exc: BaseException) -> str:
    return f"{type(exc).__name__}({str(exc)!r})" if str(exc) else type(exc).__name__


def _leve(appel, attendue, quoi: str, conseil: str = "") -> BaseException:
    """Exige que `appel()` lève `attendue` ; renvoie l'exception. NotImplementedError remonte (« à faire »)."""
    try:
        resultat = appel()
    except NotImplementedError:
        raise
    except attendue as exc:
        return exc
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de {attendue.__name__}. {conseil}".rstrip()) from None
    raise AssertionError(f"{quoi} devrait lever {attendue.__name__}, mais renvoie {_court(resultat)}. {conseil}".rstrip())


def _appel(appel, quoi: str, conseil: str = ""):
    """Appelle `appel()` ; une exception inattendue devient un message qui dit QUEL appel a planté."""
    try:
        return appel()
    except NotImplementedError:
        raise
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None


# ============================================================================ Défi 116 : photographier


def verifier_photographier(photographier) -> None:
    def double(x):
        return x * 2

    photo = photographier(double, [(1,), (2,)])
    if photo is None:
        raise AssertionError("photographier renvoie None : avez-vous oublié le `return` de la liste ?")
    if not isinstance(photo, list):
        raise AssertionError(f"photographier doit renvoyer une LISTE de paires, pas {type(photo).__name__}.")
    attendu = [((1,), ("ok", 2)), ((2,), ("ok", 4))]
    if photo != attendu:
        conseil = ""
        if len(photo) == 2 and all(isinstance(p, list) for p in photo):
            conseil = " Chaque élément est une paire en TUPLE : `(arguments, resultat)`, pas une liste."
        elif len(photo) == 2 and all(isinstance(p, tuple) and len(p) == 2 for p in photo):
            if photo[0][1] == 2:
                conseil = (" Le résultat se note `(\"ok\", valeur)` : la valeur seule ne dit pas si c'est une valeur ou "
                           "une erreur.")
            elif photo[0][0] == [1]:
                conseil = " `arguments` doit rester le tuple reçu, `(1,)`, pas une liste."
        raise AssertionError(f"photographier(double, [(1,), (2,)]) renvoie {_court(photo)} au lieu de {attendu}.{conseil}")

    # ------------------------------------------------------------------ un seul appel par cas, un seul parcours
    appels = []

    def espion(x):
        appels.append(x)
        return x

    photo = photographier(espion, ((i,) for i in range(4)))
    if not appels and photo == []:
        raise AssertionError("Avec un GÉNÉRATEUR de 4 cas, la photo est vide et la fonction n'a jamais été appelée : un "
                             "générateur ne se parcourt qu'une fois (pas de `len(list(cas))` avant la boucle).")
    if appels != [0, 1, 2, 3]:
        raise AssertionError(f"Sur 4 cas, la fonction a été appelée avec {appels} : une seule fois par cas, dans l'ordre "
                             "(certaines vieilles fonctions envoient un e-mail à chaque appel).")
    if len(photo) != 4:
        raise AssertionError(f"Avec un GÉNÉRATEUR de 4 cas, la photo a {len(photo)} éléments au lieu de 4 : un générateur "
                             "ne se parcourt qu'une fois (pas de `len(list(cas))` avant la boucle).")
    if photographier(espion, []) != []:
        raise AssertionError("Sans aucun cas, la photo doit être une liste vide.")

    # ------------------------------------------------------------------ les exceptions font partie du comportement
    def fragile(client):
        return 1 / client["n"]

    cas = [({"n": 2},), ({"n": 0},), ({},)]
    quoi = "Avec une fonction qui lève ZeroDivisionError puis KeyError"
    try:
        photo = photographier(fragile, cas)
    except NotImplementedError:
        raise
    except Exception as exc:
        raise AssertionError(f"{quoi}, photographier laisse remonter {_decrire(exc)} : une exception fait partie du "
                             "comportement, on la NOTE (`except Exception as exc:`) et on passe au cas suivant.") from None
    attendu = [(({"n": 2},), ("ok", 0.5)), (({"n": 0},), ("erreur", "ZeroDivisionError", "division by zero")),
               (({},), ("erreur", "KeyError", "'n'"))]
    if photo != attendu:
        conseil = ""
        if len(photo) == 3 and isinstance(photo[1], tuple) and len(photo[1]) == 2:
            note = photo[1][1]
            garde_l_objet = isinstance(note, tuple) and any(isinstance(x, BaseException) for x in note)
            if isinstance(note, BaseException) or garde_l_objet:
                conseil = (" Vous gardez l'OBJET exception : notez `(\"erreur\", type(exc).__name__, str(exc))`. Deux "
                           "exceptions identiques ne sont jamais égales avec ==, et deux photos du même code différeraient.")
            elif isinstance(note, tuple) and any(isinstance(x, type) for x in note):
                conseil = " Notez le NOM du type, un texte : `type(exc).__name__`, pas la classe elle-même."
            elif isinstance(note, tuple) and len(note) == 3 and note[2] != "division by zero":
                conseil = " Le message est `str(exc)`."
        raise AssertionError(f"{quoi}, photographier renvoie {_court(photo, 300)} au lieu de {attendu}.{conseil}")
    if photographier(fragile, cas) != photo:
        raise AssertionError(f"{quoi}, deux photos du MÊME code ne sont pas égales : la photo ne servirait à rien. "
                             "Ne gardez que des textes et des valeurs, pas d'objet exception.")

    # ------------------------------------------------------------------ une fonction qui modifie ses arguments
    cas = [(dict(c),) for c in CLIENTS_EXEMPLE]
    avant = copy.deepcopy(cas)
    photo = photographier(ancien_score, cas)
    quoi = "Avec ancien_score, qui MODIFIE le client reçu (« 89,90 » devient 89.9)"
    if cas != avant:
        conseil = (" `copy.copy` ou `tuple(arguments)` ne copient que le tuple : le dictionnaire dedans est le MÊME objet. "
                   "Il faut `copy.deepcopy(arguments)`.")
        raise AssertionError(f"{quoi}, vos cas ont été modifiés : le premier client a maintenant charges = "
                             f"{cas[0][0]['charges']!r}. Appelez la fonction sur une COPIE PROFONDE des arguments.{conseil}")
    if [p[0] for p in photo] != avant:
        raise AssertionError(f"{quoi}, la photo montre les arguments APRÈS l'appel ({_court(photo[0][0])}) : notez une "
                             "copie des arguments AVANT d'appeler la fonction (on veut savoir ce qu'on lui a DONNÉ).")
    attendu = [("ok", 0.95), ("ok", 0.35), ("ok", 0.85), ("erreur", "KeyError", "'charges'"),
               ("erreur", "ValueError", "charges négatives : -10.0")]
    if [p[1] for p in photo] != attendu:
        raise AssertionError(f"{quoi}, les résultats sont {_court([p[1] for p in photo], 300)} au lieu de {attendu}.")

    # ------------------------------------------------------------------ une valeur renvoyée modifiée plus tard
    historique: list = []

    def vieille(x):
        historique.append(x)
        return historique          # renvoie TOUJOURS la même liste, qu'elle allonge à chaque appel

    photo = photographier(vieille, [(1,), (2,), (3,)])
    attendu = [((1,), ("ok", [1])), ((2,), ("ok", [1, 2])), ((3,), ("ok", [1, 2, 3]))]
    if photo != attendu:
        conseil = ""
        if photo and photo[0][1] == ("ok", [1, 2, 3]):
            conseil = (" La fonction renvoie toujours la MÊME liste et l'allonge ensuite : votre photo du 1er appel a "
                       "changé après coup. Copiez la valeur renvoyée : `copy.deepcopy(valeur)`.")
        raise AssertionError(f"Avec une fonction qui renvoie une liste qu'elle allonge à chaque appel, la photo est "
                             f"{_court(photo, 300)} au lieu de {attendu}.{conseil}")

    # ------------------------------------------------------------------ Ctrl+C arrête tout
    def interrompue(x):
        raise KeyboardInterrupt

    try:
        photo = photographier(interrompue, [(1,)])
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError(f"Quand l'utilisateur appuie sur Ctrl+C (KeyboardInterrupt) pendant un cas, photographier "
                             f"renvoie {_court(photo)} : le programme doit s'arrêter. Attrapez `Exception`, pas tout "
                             "(`except:` et `except BaseException` attrapent aussi Ctrl+C).")


# ============================================================================ Défi 117 : remise


def _avertissements(appel):
    """Exécute `appel()` en notant les avertissements et ce qui est affiché. Renvoie (résultat, avertissements, sortie)."""
    sortie = io.StringIO()
    with warnings.catch_warnings(record=True) as notes, contextlib.redirect_stdout(sortie):
        warnings.simplefilter("always")
        resultat = appel()
    return resultat, list(notes), sortie.getvalue()


def _proche(a, b) -> bool:
    return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(a - b) < 0.0051


def verifier_remise(remise) -> None:
    parametres = inspect.signature(remise).parameters
    if "pourcentage" not in parametres or parametres["pourcentage"].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise AssertionError("`pourcentage` doit être un paramètre NOMMÉ seulement : gardez la signature "
                             "`remise(montant, taux=None, *, pourcentage=None)`.")
    noms = list(parametres)
    if noms[:2] != ["montant", "taux"] or parametres["taux"].kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
        raise AssertionError("`taux` doit rester le 2e paramètre, positionnel ou nommé : les anciens appels écrivent "
                             "`remise(100, 0.2)` comme `remise(100, taux=0.2)`.")

    # ------------------------------------------------------------------ la nouvelle façon
    for montant, pourcentage in ((100, 20), (59.99, 15), (80, 25), (19.9, 33.3), (0, 50), (100, 100), (100, 0)):
        quoi = f"remise({montant}, pourcentage={pourcentage})"
        attendu = remise_ancienne(montant, pourcentage / 100)
        conseil = (" 0 est une remise valide : testez `pourcentage is None`, pas `if pourcentage:`."
                   if pourcentage == 0 else "")
        resultat, notes, _ = _appel(lambda: _avertissements(lambda: remise(montant, pourcentage=pourcentage)), quoi,
                                    conseil)
        if resultat is None:
            raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
        if not _proche(resultat, attendu):
            raise AssertionError(f"{quoi} renvoie {resultat!r} au lieu de {attendu!r} (le prix APRÈS remise, arrondi au "
                                 "centime).")
        if notes:
            raise AssertionError(f"{quoi} émet un avertissement ({notes[0].category.__name__}: {notes[0].message}) : la "
                                 "NOUVELLE façon d'appeler ne doit rien signaler.")

    # ------------------------------------------------------------------ l'ancienne façon, par position et par nom
    for appel, quoi, montant, taux in ((lambda: remise(100, 0.2), "remise(100, 0.2)", 100, 0.2),
                                       (lambda: remise(100, taux=0.2), "remise(100, taux=0.2)", 100, 0.2),
                                       (lambda: remise(59.99, 0.15), "remise(59.99, 0.15)", 59.99, 0.15),
                                       (lambda: remise(80, 0), "remise(80, 0)", 80, 0)):
        conseil = " 0 est un taux valide : testez `taux is None`, pas `if taux:`." if taux == 0 else ""
        resultat, notes, sortie = _appel(lambda: _avertissements(appel), quoi, conseil)
        attendu = remise_ancienne(montant, taux)
        if not _proche(resultat, attendu):
            raise AssertionError(f"{quoi} renvoie {resultat!r} au lieu de {attendu!r} : l'ancien appel doit donner "
                                 f"EXACTEMENT le résultat de remise_ancienne({montant}, {taux}).{conseil}")
        if not notes:
            conseil = (" Vous l'affichez avec print : un print ne se filtre pas, ne se transforme pas en erreur dans les "
                       "tests et passe inaperçu dans les journaux. Utilisez `warnings.warn(...)`." if sortie.strip() else "")
            raise AssertionError(f"{quoi} marche, mais ne prévient personne : émettez un DeprecationWarning, sinon "
                                 f"l'ancienne façon ne disparaîtra jamais.{conseil}")
        if len(notes) > 1:
            raise AssertionError(f"{quoi} émet {len(notes)} avertissements : un seul par appel.")
        note = notes[0]
        if not issubclass(note.category, DeprecationWarning):
            raise AssertionError(f"{quoi} émet un {note.category.__name__} : la catégorie qui veut dire « cette façon "
                                 "d'appeler va disparaître » est DeprecationWarning (`warnings.warn(message, "
                                 "DeprecationWarning, stacklevel=2)`).")
        if "pourcentage" not in str(note.message):
            raise AssertionError(f"L'avertissement de {quoi} dit « {note.message} » : dites quoi écrire à la place (le "
                                 "message doit nommer `pourcentage`).")
        if not note.filename.replace("\\", "/").endswith("checks_serie_19.py"):
            raise AssertionError(f"L'avertissement de {quoi} désigne {note.filename.replace(chr(92), '/').split('/')[-1]}, "
                                 f"ligne {note.lineno} : c'est-à-dire votre fonction, pas la ligne qui l'APPELLE, celle "
                                 "qu'il faut corriger. Ajoutez `stacklevel=2` à warnings.warn (3 si vous l'appelez depuis "
                                 "une petite fonction intermédiaire). Sans lui, Python CACHE même l'avertissement par "
                                 "défaut.")

    # ------------------------------------------------------------------ ce qui doit être refusé
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _leve(lambda: remise(100, 20), ValueError, "remise(100, 20) (quelqu'un a cru écrire 20 %)",
              "Comme avant, un taux hors de [0, 1] est refusé : sinon on offre 2 000 % de remise.")
        for valeur in (120, -5):
            exc = _leve(lambda: remise(100, pourcentage=valeur), ValueError, f"remise(100, pourcentage={valeur})")
            if "pourcentage" not in str(exc):
                raise AssertionError(f"Le message de la ValueError de remise(100, pourcentage={valeur}) est « {exc} » : "
                                     "nommez le paramètre fautif, `pourcentage`.")
        for appel, quoi in ((lambda: remise(100), "remise(100) (ni taux, ni pourcentage)"),
                            (lambda: remise(100, 0.2, pourcentage=20), "remise(100, 0.2, pourcentage=20) (les deux)")):
            exc = _leve(appel, TypeError, quoi, "C'est un mauvais APPEL, comme un argument oublié : TypeError.")
            if "taux" not in str(exc) or "pourcentage" not in str(exc):
                raise AssertionError(f"Le message de la TypeError de {quoi} est « {exc} » : il doit nommer `taux` et "
                                     "`pourcentage`, pour qu'on comprenne tout de suite quoi corriger.")

    # ------------------------------------------------------------------ des montants au hasard, les deux façons
    rng = random.Random(117)
    for _ in range(300):
        montant, points = round(rng.uniform(0, 500), 2), rng.choice([0, 5, 10, 12.5, 15, 20, 25, 33, 50, 70, 100])
        attendu = remise_ancienne(montant, points / 100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nouvelle, ancienne = remise(montant, pourcentage=points), remise(montant, points / 100)
        if not (_proche(nouvelle, attendu) and _proche(ancienne, attendu)):
            raise AssertionError(f"remise({montant}, pourcentage={points}) renvoie {nouvelle!r} et remise({montant}, "
                                 f"{points / 100}) renvoie {ancienne!r}, au lieu de {attendu!r} tous les deux. Une seule "
                                 "formule pour les deux façons : `remise_ancienne(montant, pourcentage / 100)`.")


# ============================================================================ Défi 118 : decider_relances

CONTRATS = ("Mensuel", "Annuel", "Biannuel")


def _reference_relances(lignes, jour, budget=50, *, seuil_strict=False, trente_exclus=False, consentement_souple=False,
                        sans_tri_id=False, budget_avant_filtres=False):
    """La logique de l'ancien script, recopiée ; les options reproduisent les écarts classiques (pour les messages)."""
    candidats = [(float(ligne["score"].replace(",", ".")), ligne) for ligne in lignes]
    if sans_tri_id:
        candidats.sort(key=lambda c: -c[0])
    else:
        candidats.sort(key=lambda c: (-c[0], c[1]["client_id"]))
    if budget_avant_filtres:
        candidats = candidats[:budget]
    relances = []
    for score, ligne in candidats:
        if score <= 0.6 if seuil_strict else score < 0.6:
            continue
        consentement = ligne["consentement"].strip().lower() if consentement_souple else ligne["consentement"]
        if consentement != "oui":
            continue
        if ligne["dernier_contact"]:
            jours = (jour - date.fromisoformat(ligne["dernier_contact"])).days
            if jours <= 30 if trente_exclus else jours < 30:
                continue
        if len(relances) >= budget:
            break
        relances.append((ligne["client_id"], "ENGAGEMENT" if ligne["contrat"] == "Mensuel" else "FIDELITE"))
    return relances


_ECARTS_RELANCES = (
    ({"seuil_strict": True}, "Un client au score de 0,6 PILE doit être relancé : l'ancien code fait `if score < 0.6: "
                             "continue` (il ne saute que les scores strictement inférieurs)."),
    ({"trente_exclus": True}, "Un client contacté il y a 30 jours PILE doit être relancé : l'ancien code ne saute que "
                              "`(aujourd_hui - dernier).days < 30`."),
    ({"consentement_souple": True}, "Vous relancez des clients dont le consentement est écrit « Oui » ou « oui » avec "
                                    "des espaces : l'ancien code les refuse (`!= \"oui\"`). C'est peut-être un bogue, "
                                    "mais le corriger ici change qui reçoit des SMS au milieu d'un changement censé ne "
                                    "rien changer. Gardez le comportement, et notez le bogue dans un ticket à part."),
    ({"sans_tri_id": True}, "À score égal, l'ancien code trie par client_id : `sort(key=lambda c: (-c[\"score\"], "
                            "c[\"client_id\"]))`."),
    ({"budget_avant_filtres": True}, "Vous appliquez le budget AVANT les filtres (les 50 meilleurs scores, puis on "
                                     "filtre) : l'ancien code relance les 50 premiers clients QUI PASSENT les filtres. "
                                     "Le budget se vérifie juste avant d'ajouter une relance."),
)


def _ligne_au_hasard(rng: random.Random, client_id: str, jour: date) -> dict[str, str]:
    score = rng.choice([round(rng.random(), 2), round(rng.uniform(0.55, 0.99), 2), 0.6, 0.6, 0.59, 0.82])
    ecart = rng.choice([None, None, 0, 1, 15, 29, 30, 30, 31, 90, 400, -5])
    return {
        "client_id": client_id,
        "telephone": "06" + "".join(rng.choice("0123456789") for _ in range(8)),
        "score": f"{score}".replace(".", ","),
        "contrat": rng.choice(CONTRATS),
        "consentement": rng.choice(["oui"] * 7 + ["non", "Oui", " oui"]),
        "dernier_contact": "" if ecart is None else (jour - timedelta(days=ecart)).isoformat(),
    }


def _fichier_au_hasard(rng: random.Random, jour: date, n: int) -> list[dict[str, str]]:
    ids = [f"C{i:04d}" for i in rng.sample(range(10_000), n)]
    return [_ligne_au_hasard(rng, client_id, jour) for client_id in ids]


def _expliquer_ecart(obtenu, lignes, jour, budget) -> str:
    attendu = _reference_relances(lignes, jour, budget)
    for options, conseil in _ECARTS_RELANCES:
        if obtenu == _reference_relances(lignes, jour, budget, **options) != attendu:
            return conseil
    if obtenu == _reference_relances(lignes, date.today(), budget) != attendu:
        return (f"Vos décisions correspondent à la date d'AUJOURD'HUI ({date.today()}), pas à celle reçue ({jour}) : "
                "n'appelez jamais date.today() dans le cœur, utilisez le paramètre `aujourd_hui`.")
    par_id = {ligne["client_id"]: ligne for ligne in lignes}
    for i, (a, b) in enumerate(zip(obtenu, attendu)):
        if a != b:
            ligne = par_id.get(b[0])
            return (f"Première différence en position {i} : vous avez {a!r}, l'ancien script aurait relancé {b!r} "
                    f"(la ligne : {_court(ligne, 260)}).")
    if len(obtenu) < len(attendu):
        return (f"Il vous manque {len(attendu) - len(obtenu)} relance(s) à la fin, à partir de {attendu[len(obtenu)]!r} "
                f"(la ligne : {_court(par_id.get(attendu[len(obtenu)][0]), 260)}).")
    return f"Vous avez {len(obtenu) - len(attendu)} relance(s) en trop à la fin, à partir de {obtenu[len(attendu)]!r}."


def _verifier_un_fichier(decider, lignes, jour, budget, quoi):
    copie = copy.deepcopy(lignes)
    sortie = io.StringIO()
    with contextlib.redirect_stdout(sortie):
        obtenu = _appel(lambda: decider(copie, jour, budget), quoi)
    if sortie.getvalue():
        raise AssertionError(f"{quoi}, votre fonction affiche « {sortie.getvalue().strip()[:80]} » : le cœur ne fait pas "
                             "de print. Il RENVOIE sa décision, et c'est la coquille qui l'affiche si elle veut.")
    if copie != lignes:
        modifiee = next(a for a, b in zip(copie, lignes) if a != b)
        raise AssertionError(f"{quoi}, votre fonction a modifié les lignes reçues (par exemple {_court(modifiee)}) : "
                             "l'ancien script le faisait, le cœur ne doit pas. Calculez le score dans une variable, ou "
                             "une nouvelle liste.")
    if obtenu is None:
        raise AssertionError(f"{quoi}, votre fonction renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(obtenu, list) or not all(isinstance(r, tuple) and len(r) == 2 for r in obtenu):
        raise AssertionError(f"{quoi}, votre fonction renvoie {_court(obtenu)} : une LISTE de tuples "
                             "(client_id, offre) était attendue.")
    attendu = _reference_relances(lignes, jour, budget)
    if obtenu != attendu:
        raise AssertionError(f"{quoi}, vous décidez {_court(obtenu, 240)} au lieu de {_court(attendu, 240)}. "
                             + _expliquer_ecart(obtenu, lignes, jour, budget))
    return obtenu


def verifier_decider_relances(decider) -> None:
    # ------------------------------------------------------------------ les cas limites, un par un
    jour = date(2025, 6, 2)
    limites = [
        {"client_id": "L1", "score": "0,6", "consentement": "oui", "dernier_contact": "", "contrat": "Annuel"},
        {"client_id": "L2", "score": "0,8", "consentement": "oui", "dernier_contact": "2025-05-03", "contrat": "Mensuel"},
        {"client_id": "L3", "score": "0,8", "consentement": "Oui", "dernier_contact": "", "contrat": "Mensuel"},
        {"client_id": "L4", "score": "0,8", "consentement": "oui", "dernier_contact": "2025-06-10", "contrat": "Annuel"},
        {"client_id": "L5", "score": "0,59", "consentement": "oui", "dernier_contact": "", "contrat": "Mensuel"},
        {"client_id": "L6", "score": "0,8", "consentement": "oui", "dernier_contact": "2025-05-04", "contrat": "Biannuel"},
    ]
    for ligne in limites:
        ligne["telephone"] = "0600000000"
    explications = {
        "L1": "score 0,6 pile", "L2": "dernier contact il y a 30 jours pile", "L3": "consentement « Oui »",
        "L4": "dernier contact dans le FUTUR (erreur de saisie : l'ancien code ne le relance pas)",
        "L5": "score 0,59", "L6": "dernier contact il y a 29 jours",
    }
    for ligne in limites:
        _verifier_un_fichier(decider, [ligne], jour, 50, f"Pour un client seul, {explications[ligne['client_id']]}")

    quoi = "Sur LIGNES_EXEMPLE au JOUR_EXEMPLE"
    premiere = _verifier_un_fichier(decider, LIGNES_EXEMPLE, JOUR_EXEMPLE, 50, quoi)
    for numero in range(2, 16):
        seconde = _appel(lambda: decider(copy.deepcopy(LIGNES_EXEMPLE), JOUR_EXEMPLE, 50), quoi)
        if seconde != premiere:
            break
    if seconde != premiere:
        raise AssertionError(f"{quoi}, l'appel n°{numero} renvoie {_court(seconde)} au lieu de {_court(premiere)} "
                             "comme le 1er : votre fonction garde un état d'un appel à l'autre (un compteur global, "
                             "comme le budget de l'ancien script ?). Le budget appartient à UN appel : une variable locale.")
    _verifier_un_fichier(decider, LIGNES_EXEMPLE, JOUR_EXEMPLE, 2, "Sur LIGNES_EXEMPLE avec budget=2")
    _verifier_un_fichier(decider, LIGNES_EXEMPLE, JOUR_EXEMPLE, 0, "Sur LIGNES_EXEMPLE avec budget=0")
    _verifier_un_fichier(decider, [], JOUR_EXEMPLE, 50, "Sur un fichier vide")

    # ------------------------------------------------------------------ des centaines de fichiers tirés au hasard
    rng = random.Random(118)
    for _ in range(300):
        jour = date(2025, 1, 1) + timedelta(days=rng.randrange(365))
        lignes = _fichier_au_hasard(rng, jour, rng.randint(1, 40))
        budget = rng.choice([50, 50, 3, 5, 10, 1])
        _verifier_un_fichier(decider, lignes, jour, budget,
                             f"Sur un fichier tiré au hasard ({len(lignes)} clients, jour {jour}, budget {budget})")


# ============================================================================ Défi 119 : lire_preferences


def _prefs(canaux, heure=10, langue="fr"):
    return {"version": 2, "canaux": canaux, "heure": heure, "langue": langue}


_PREFERENCES_JUSTES = (
    (PREFERENCES_V1_EXEMPLE, _prefs(["sms"], 18), "le format 1 de l'exemple"),
    ('{"sms": false, "email": true}', _prefs(["email"]), "un format 1 sans heure"),
    ('{"sms": true, "email": true, "heure": 0}', _prefs(["email", "sms"], 0), "un format 1 à minuit (heure 0)"),
    ('{"version": 1, "sms": true}', _prefs(["sms"]), "un format 1 qui écrit \"version\": 1"),
    ("{}", _prefs([]), "un objet vide (format 1, rien de coché)"),
    (PREFERENCES_V2_EXEMPLE, _prefs(["courrier", "sms"], 9, "en"), "le format 2 de l'exemple"),
    ('{"version": 2, "canaux": ["sms", "sms", "email"]}', _prefs(["email", "sms"]), "un format 2 avec un doublon"),
    ('{"version": 2, "canaux": [], "heure": 23, "fuseau": "Europe/Paris"}', _prefs([], 23),
     "un format 2 avec une clé inconnue (« fuseau »)"),
    ('{"version": 2}', _prefs([]), "un format 2 réduit à sa version"),
    (None, _prefs([]), "None (client créé avant les préférences)"),
    ("", _prefs([]), "une chaîne vide"),
    ("   ", _prefs([]), "une chaîne d'espaces"),
)

_PREFERENCES_REFUSEES = (
    ('{"version": 3, "canaux": ["sms"]}', "une version 3 (écrite par une version plus récente de l'application)", "3",
     "Ne devinez pas : refusez toute version que vous ne connaissez pas."),
    ('{"version": "2", "canaux": ["sms"]}', "une version \"2\" en texte", None, "\"2\" n'est pas 2 : vérifiez le type."),
    ('{"version": null}', "une version null", None, "Une version absente vaut 1 ; une version null n'est pas absente."),
    ('{"version": true, "sms": true}', "une version true", None,
     "En Python, True == 1 : `version == 1` accepte True. Vérifiez le type exact (`type(version) is int`)."),
    ('{"version": 2, "canaux": ["sms", "pigeon"]}', "un canal « pigeon »", "pigeon", "Le message doit nommer le canal."),
    ('{"version": 2, "canaux": "sms"}', "des canaux en texte au lieu d'une liste", None,
     "Sinon, \"sms\" devient la liste ['m', 's', 's']."),
    ('{"heure": 24}', "une heure 24", None, "L'heure va de 0 à 23."),
    ('{"heure": -1}', "une heure -1", None, ""),
    ('{"heure": true}', "une heure true", None,
     "En Python, True est un entier (isinstance(True, int) vaut True) : vérifiez `type(heure) is int`."),
    ('{"heure": "18"}', "une heure en texte", None, ""),
    ('{"version": 2, "heure": 12.5}', "une heure 12.5", None, ""),
    ("[1, 2]", "du JSON qui n'est pas un objet (une liste)", None, "Vérifiez `isinstance(donnees, dict)`."),
    ('"sms"', "du JSON qui n'est pas un objet (un texte)", None, ""),
    ("pas du json", "un texte qui n'est pas du JSON", None, "json.JSONDecodeError hérite de ValueError : laissez-la "
                                                            "remonter."),
    ('{"sms": true', "du JSON tronqué", None, ""),
)


def verifier_lire_preferences(lire) -> None:
    for texte, attendu, quoi in _PREFERENCES_JUSTES:
        conseil = ""
        if texte is None:
            conseil = "Testez `texte is None` AVANT json.loads (qui refuse None)."
        elif not texte.strip():
            conseil = ("Une chaîne vide ou faite d'espaces donne les préférences par défaut : testez "
                       "`not texte.strip()` AVANT json.loads.")
        elif '"version"' not in texte:
            conseil = ("Le format 1 n'a pas de clé \"version\" : `donnees[\"version\"]` lève KeyError, "
                       "`donnees.get(\"version\", 1)` donne 1.")
        obtenu = _appel(lambda: lire(texte), f"lire_preferences({texte!r}) ({quoi})", conseil)
        if obtenu is None:
            raise AssertionError(f"lire_preferences({texte!r}) renvoie None : avez-vous oublié le `return` ?")
        if obtenu != attendu:
            conseil = ""
            if isinstance(obtenu, dict):
                if set(obtenu) - set(attendu):
                    conseil = (f" Le résultat ne contient que les 4 clés du format 2 : les clés inconnues "
                               f"({', '.join(sorted(set(obtenu) - set(attendu)))}) sont ignorées.")
                elif obtenu.get("heure") == 10 and attendu["heure"] == 0:
                    conseil = (" 0 (minuit) est une heure valide : `donnees.get(\"heure\") or 10` la remplace par 10. "
                               "Utilisez `donnees.get(\"heure\", 10)`.")
                elif not isinstance(obtenu.get("canaux"), list):
                    conseil = " `canaux` doit être une LISTE triée : `sorted(set(canaux))`."
                elif sorted(set(obtenu.get("canaux", []))) == attendu["canaux"]:
                    conseil = " `canaux` est triée, sans doublon : `sorted(set(canaux))`."
                elif obtenu.get("version") != 2:
                    conseil = " Le résultat est TOUJOURS au format 2 : \"version\": 2."
            raise AssertionError(f"lire_preferences({texte!r}) ({quoi}) renvoie {_court(obtenu)} au lieu de "
                                 f"{attendu}.{conseil}")
        relu = _appel(lambda: lire(json.dumps(obtenu)), f"lire_preferences(json.dumps({obtenu}))")
        if relu != obtenu:
            raise AssertionError(f"Relire ce qu'on vient d'écrire change le résultat : lire_preferences({texte!r}) donne "
                                 f"{obtenu}, mais relire son json.dumps donne {_court(relu)}.")

    # ------------------------------------------------------------------ le défaut partagé
    defaut = lire(None)
    defaut["canaux"].append("sms")
    defaut["heure"] = 3
    for texte in (None, ""):
        nouveau = lire(texte)
        if nouveau != _prefs([]):
            raise AssertionError(f"Après qu'un appelant a modifié le résultat de lire_preferences(None) "
                                 f"(prefs[\"canaux\"].append(\"sms\")), lire_preferences({texte!r}) renvoie {nouveau} : "
                                 "tous les clients suivants recevraient des SMS sans l'avoir accepté ! Renvoyez un "
                                 "NOUVEAU dictionnaire (et une nouvelle liste) à chaque appel, pas une constante du module.")

    # ------------------------------------------------------------------ ce qu'on refuse
    for texte, quoi, mot, conseil in _PREFERENCES_REFUSEES:
        exc = _leve(lambda: lire(texte), ValueError, f"lire_preferences({texte!r}) ({quoi})", conseil)
        if mot and mot not in str(exc):
            raise AssertionError(f"La ValueError de lire_preferences({texte!r}) dit « {exc} » : nommez « {mot} » dans le "
                                 "message, pour qu'on sache quoi chercher.")


# ============================================================================ Défi 120 : AdaptateurSMS

_FORMATS_VALIDES = (
    ("06 12 34 56 78", "+33612345678"), ("0612345678", "+33612345678"), ("06.12.34.56.78", "+33612345678"),
    ("06-12-34-56-78", "+33612345678"), ("+33 6 12 34 56 78", "+33612345678"), ("+33612345678", "+33612345678"),
    ("0033 6 12 34 56 78", "+33612345678"), ("07 98 76 54 32", "+33798765432"), ("01 23 45 67 89", "+33123456789"),
)
_FORMATS_INVALIDES = ("06 12 34", "", "abc", "06123456789", "+33 0 12 34 56 78", "00 12 34 56 78", "6 12 34 56 78",
                      "06 12 34 56 7a")


class _CreateQuiPlante:
    """Un faux kit dont `create` lève une erreur qui n'est pas une ErreurMessagerie (un bogue du kit)."""

    def __init__(self) -> None:
        self.appels = 0
        self.messages = self

    def create(self, **_kwargs):
        self.appels += 1
        raise TypeError("create() : bogue interne du kit")


def verifier_AdaptateurSMS(classe) -> None:
    client = ClientMessagerie("cle_test")
    adaptateur = classe(client)
    if client.appels:
        raise AssertionError("Créer l'adaptateur a déjà envoyé un SMS : __init__ garde le client et l'expéditeur, rien de "
                             "plus.")
    resultat = _appel(lambda: adaptateur.envoyer_sms("06 12 34 56 78", "Bonjour"),
                      'envoyer_sms("06 12 34 56 78", "Bonjour")')
    if resultat is not True:
        conseil = ""
        if isinstance(resultat, Reponse):
            conseil = (" Vous renvoyez la Reponse du nouvel opérateur : l'ancienne interface renvoie un BOOLÉEN. "
                       "`return reponse.statut == \"accepte\"`.")
        elif client.appels and client.appels[-1]["destinataire"] != "+33612345678":
            conseil = (f" Le nouvel opérateur a reçu destinataire={client.appels[-1]['destinataire']!r} : il veut le format "
                       "international, « +33612345678 ».")
        elif not client.appels:
            conseil = " Le nouvel opérateur n'a même pas été appelé : `self.client.messages.create(...)`."
        raise AssertionError(f'envoyer_sms("06 12 34 56 78", "Bonjour") renvoie {_court(resultat)} au lieu de '
                             f'True.{conseil}')
    if client.envoyes != [("+33612345678", "Bonjour", "RETENTION", 1)]:
        raise AssertionError(f"Après un envoi, le nouvel opérateur a envoyé {client.envoyes} au lieu de "
                             "[('+33612345678', 'Bonjour', 'RETENTION', 1)] (destinataire, contenu, expéditeur, SMS "
                             "facturés).")

    autre_client = ClientMessagerie("cle_autre")
    autre = classe(autre_client, expediteur="FIDELITE")
    _appel(lambda: autre.envoyer_sms("0798765432", "Merci"), "Avec expediteur=\"FIDELITE\", envoyer_sms(...)")
    if len(client.envoyes) != 1 or not autre_client.envoyes:
        raise AssertionError("Un 2e adaptateur, sur un AUTRE client, a envoyé par le client du 1er : gardez le client dans "
                             "l'OBJET (self.client), pas dans un attribut de classe.")
    if [e[2] for e in autre_client.envoyes] != ["FIDELITE"]:
        raise AssertionError(f"AdaptateurSMS(client, expediteur=\"FIDELITE\") envoie avec l'expéditeur "
                             f"{[e[2] for e in autre_client.envoyes]} : passez `expediteur=self.expediteur` à create.")

    # ------------------------------------------------------------------ les numéros : comme l'ancien
    for numero, international in _FORMATS_VALIDES:
        client = ClientMessagerie("cle_test")
        resultat = _appel(lambda: classe(client).envoyer_sms(numero, "Bonjour"), f"envoyer_sms({numero!r}, \"Bonjour\")")
        if resultat is not True or [a["destinataire"] for a in client.appels] != [international]:
            recu = [a["destinataire"] for a in client.appels]
            raise AssertionError(f"envoyer_sms({numero!r}, \"Bonjour\") renvoie {resultat!r} et le nouvel opérateur a reçu "
                                 f"{recu} : l'ancien opérateur acceptait ce numéro, il faut l'envoyer à {international!r}. "
                                 "Retirez espaces, points et tirets, puis convertissez 0X…, +33X… et 0033X… en +33X….")
    for numero in _FORMATS_INVALIDES:
        client = ClientMessagerie("cle_test")
        resultat = _appel(lambda: classe(client).envoyer_sms(numero, "Bonjour"), f"envoyer_sms({numero!r}, \"Bonjour\")",
                          "Un numéro mal formé n'est pas une erreur du programme : l'ancien opérateur renvoyait False.")
        if resultat is not False or client.envoyes:
            raise AssertionError(f"envoyer_sms({numero!r}, \"Bonjour\") renvoie {resultat!r} (SMS envoyés : "
                                 f"{client.envoyes}) : c'est un numéro mal formé, l'ancien opérateur renvoyait False.")

    # ------------------------------------------------------------------ « pas parti » : False
    for quoi, client, numero in (
        ("Pour un client sur liste STOP (statut « refuse »)", ClientMessagerie("cle_test", refuses=("+33699999999",)),
         "06 99 99 99 99"),
        ("Quand l'opérateur est en panne passagère (« INDISPONIBLE »)", ClientMessagerie("cle_test", panne="INDISPONIBLE"),
         "0612345678"),
        ("Quand l'opérateur répond « NUMERO_INVALIDE »", ClientMessagerie("cle_test", panne="NUMERO_INVALIDE"),
         "0612345678"),
    ):
        resultat = _appel(lambda: classe(client).envoyer_sms(numero, "Bonjour"), f"{quoi}, envoyer_sms",
                          "L'ancien opérateur renvoyait False : le SMS n'est pas parti, l'appelant passe au suivant.")
        if resultat is not False:
            conseil = " Regardez `reponse.statut` : « refuse » veut dire que rien n'est parti." if "STOP" in quoi else ""
            raise AssertionError(f"{quoi}, envoyer_sms renvoie {_court(resultat)} au lieu de False.{conseil}")
        if len(client.appels) != 1:
            raise AssertionError(f"{quoi}, create a été appelé {len(client.appels)} fois pour UN SMS : pas de nouvel essai "
                                 "caché dans l'adaptateur (les reprises, c'est le défi 88, à un autre étage).")

    # ------------------------------------------------------------------ les vraies erreurs remontent
    client = ClientMessagerie("perimee_2023")
    quoi = "Quand le nouvel opérateur refuse la clé d'API (« CLE_INVALIDE »)"
    exc = _leve(lambda: classe(client).envoyer_sms("0612345678", "Bonjour"), PermissionError, f"{quoi}, envoyer_sms",
                "L'ancien opérateur levait PermissionError : c'est un bogue de configuration, il faut le voir.")
    if not isinstance(exc.__cause__, ErreurMessagerie):
        raise AssertionError(f"{quoi}, la PermissionError n'a pas l'erreur d'origine comme cause : "
                             "`raise PermissionError(...) from exc` garde l'ErreurMessagerie dans la traceback.")

    client = ClientMessagerie("cle_test", panne="QUOTA_DEPASSE")
    quoi = "Quand le nouvel opérateur répond « QUOTA_DEPASSE » (un code que l'ancien contrat ne prévoit pas)"
    exc = _leve(lambda: classe(client).envoyer_sms("0612345678", "Bonjour"), ErreurMessagerie, f"{quoi}, envoyer_sms",
                "Toute erreur non prévue remonte TELLE QUELLE (`raise` seul) : la transformer en False ou en autre chose "
                "cache un vrai problème.")
    if exc.code != "QUOTA_DEPASSE":
        raise AssertionError(f"{quoi}, l'erreur qui remonte a le code {exc.code!r} : relancez la même, avec `raise` seul.")

    kit = _CreateQuiPlante()
    _leve(lambda: classe(kit).envoyer_sms("0612345678", "Bonjour"), TypeError,
          "Quand le kit du nouvel opérateur a un bogue (TypeError dans create), envoyer_sms",
          "N'attrapez que ErreurMessagerie : un `except Exception` transformerait les bogues en « SMS pas parti ».")

    # ------------------------------------------------------------------ le texte : vérifié AVANT d'envoyer
    for texte, quoi in (("x" * 161, "un texte de 161 caractères"), ("", "un texte vide")):
        client = ClientMessagerie("cle_test")
        conseil = (" Le nouvel opérateur, lui, l'accepte et facture 2 SMS : chaque relance un peu longue coûterait le "
                   "double, en silence." if texte else "")
        _leve(lambda: classe(client).envoyer_sms("0612345678", texte), ValueError, f"envoyer_sms avec {quoi}",
              f"L'ancien opérateur levait ValueError : c'est un bogue de l'appelant.{conseil}")
        if client.appels:
            raise AssertionError(f"Avec {quoi}, le nouvel opérateur a été appelé avant la ValueError : vérifiez le texte "
                                 "AVANT d'appeler create (rien ne doit partir).")
    client = ClientMessagerie("cle_test")
    if _appel(lambda: classe(client).envoyer_sms("0612345678", "x" * 160), "envoyer_sms avec 160 caractères") is not True:
        raise AssertionError("Un texte de 160 caractères pile est accepté par l'ancien opérateur : il doit partir.")

    # ------------------------------------------------------------------ côte à côte avec l'ancien, au hasard
    rng = random.Random(120)
    stop = ("0611111111", "0622222222")
    ancienne = PasserelleAncienne(stop=stop)
    client = ClientMessagerie("cle_test", refuses=("+33611111111", "+33622222222"))
    adaptateur = classe(client)
    numeros = [n for n, _ in _FORMATS_VALIDES] + list(_FORMATS_INVALIDES) + list(stop) + ["06 11 11 11 11"]
    for _ in range(200):
        numero = rng.choice(numeros)
        texte = rng.choice(["Bonjour", "Votre offre fidélité : -20 % pendant 6 mois. STOP au 36180", "y" * 160])
        attendu, obtenu = ancienne.envoyer_sms(numero, texte), adaptateur.envoyer_sms(numero, texte)
        if obtenu != attendu:
            raise AssertionError(f"envoyer_sms({numero!r}, {texte[:20]!r}…) renvoie {obtenu!r}, l'ancien opérateur "
                                 f"renvoyait {attendu!r}.")
    if len(client.appels) > 200:
        raise AssertionError(f"200 envois ont fait {len(client.appels)} appels à create : un seul par SMS.")


# ============================================================================ Défi 121 : en_ombre


class _Tirages:
    """Une fausse fonction `tirage` qui renvoie des valeurs choisies, et compte ses appels."""

    def __init__(self, valeurs) -> None:
        self.valeurs = list(valeurs)
        self.appels = 0

    def __call__(self) -> float:
        self.appels += 1
        if not self.valeurs:
            raise AssertionError("tirage() a été appelé plus d'une fois par appel : tirez UNE fois, au début.")
        return self.valeurs.pop(0)


def _noter(fonction, *args, **kwargs):
    try:
        return ("ok", fonction(*args, **kwargs)), None
    except Exception as exc:
        return ("erreur", type(exc).__name__, str(exc)), exc


def verifier_en_ombre(en_ombre) -> None:
    differences: list = []
    score = en_ombre(ancien_score, nouveau_score, differences.append)
    if score is None or not callable(score):
        raise AssertionError(f"en_ombre(...) renvoie {_court(score)} : elle doit renvoyer une FONCTION (définie à "
                             "l'intérieur, puis `return fonction`).")
    if getattr(score, "stats", None) != {"appels": 0, "comparaisons": 0, "differences": 0}:
        raise AssertionError(f"Juste après en_ombre(...), la fonction renvoyée a stats = "
                             f"{_court(getattr(score, 'stats', None))} "
                             "au lieu de {'appels': 0, 'comparaisons': 0, 'differences': 0} : `fonction.stats = stats` "
                             "avant le return.")
    if getattr(score, "__name__", None) != "ancien_score" or score.__doc__ != ancien_score.__doc__:
        raise AssertionError(f"La fonction renvoyée s'appelle {getattr(score, '__name__', None)!r} : décorez-la de "
                             "`@functools.wraps(ancienne)` pour qu'elle garde le nom et la docstring de l'ancienne.")

    # ------------------------------------------------------------------ chacun ses stats
    a = en_ombre(lambda: 1, lambda: 1, print)
    b = en_ombre(lambda: 1, lambda: 2, lambda d: None)
    a()
    a()
    if b.stats != {"appels": 0, "comparaisons": 0, "differences": 0}:
        raise AssertionError(f"Deux fonctions en_ombre partagent leurs stats ({b.stats} pour celle qu'on n'a jamais "
                             "appelée) : créez le dictionnaire DANS en_ombre, pas au niveau du module ni en valeur par "
                             "défaut d'un paramètre.")

    # ------------------------------------------------------------------ une différence de valeur
    client = {"client_id": "C006", "charges": "50"}
    quoi = 'score({"client_id": "C006", "charges": "50"})'
    resultat = _appel(lambda: score(client), quoi)
    if resultat != 0.62:
        raise AssertionError(f"{quoi} renvoie {_court(resultat)} au lieu de 0.62 : l'utilisateur reçoit TOUJOURS le "
                             "résultat de l'ANCIENNE version.")
    if client["charges"] != 50.0 or not isinstance(client["charges"], float):
        raise AssertionError(f"Après {quoi}, le client de l'appelant n'a pas été converti comme ancien_score le fait "
                             "(charges en float) : l'ancienne reçoit les VRAIS arguments, pour que rien ne change pour "
                             "l'appelant. Ce sont les copies qui servent à la comparaison et au journal.")
    attendu = [{"arguments": (({"client_id": "C006", "charges": "50"},), {}), "ancienne": ("ok", 0.62),
                "nouvelle": ("ok", 0.37)}]
    if differences != attendu:
        conseil = ""
        if differences and isinstance(differences[0], dict) and "arguments" in differences[0]:
            notes = differences[0]["arguments"]
            if notes != attendu[0]["arguments"] and repr(notes).find("50.0") >= 0:
                conseil = (" Le journal montre les arguments APRÈS l'ancienne, qui a converti les charges : copiez-les "
                           "(copy.deepcopy) AVANT de l'appeler.")
            elif notes != attendu[0]["arguments"]:
                conseil = (" \"arguments\" vaut le couple (args, kwargs) : un tuple des arguments positionnels, puis un "
                           "dictionnaire.")
        elif not differences:
            conseil = " Le journal est vide : la nouvelle version n'a-t-elle pas été appelée, ou pas comparée ?"
        raise AssertionError(f"Après {quoi}, le journal contient {_court(differences, 300)} au lieu de {attendu}.{conseil}")
    if score.stats != {"appels": 1, "comparaisons": 1, "differences": 1}:
        raise AssertionError(f"Après un appel comparé et différent, stats vaut {score.stats} au lieu de "
                             "{'appels': 1, 'comparaisons': 1, 'differences': 1}.")

    # ------------------------------------------------------------------ pas de différence
    resultat = _appel(lambda: score({"client_id": "C002", "charges": 45.0, "contrat": "Annuel"}), "score(C002)")
    if resultat != 0.35 or len(differences) != 1:
        raise AssertionError(f"Pour un client où les deux versions répondent 0.35, score renvoie {resultat!r} et le "
                             f"journal a {len(differences) - 1} nouvelle(s) entrée(s) : il ne faut noter que les "
                             "DIFFÉRENCES.")

    # ------------------------------------------------------------------ l'ancienne plante
    quoi = "Pour un client sans charges (l'ancienne lève KeyError, la nouvelle ValueError)"
    try:
        score({"client_id": "C004", "contrat": "Biannuel"})
    except KeyError:
        pass
    except Exception as exc:
        conseil = (" C'est l'erreur de la NOUVELLE version : elle ne doit JAMAIS gêner l'utilisateur. Notez-la "
                   "(`except Exception`) et continuez." if str(exc) == "charges manquantes" else "")
        raise AssertionError(f"{quoi}, l'appelant reçoit {_decrire(exc)} : il doit recevoir l'erreur de l'ANCIENNE, la "
                             f"même, comme avant.{conseil}") from None
    else:
        raise AssertionError(f"{quoi}, l'appelant ne reçoit aucune erreur : l'ancienne a levé KeyError, elle doit remonter.")
    if len(differences) != 2 or differences[-1].get("ancienne") != ("erreur", "KeyError", "'charges'") or \
            differences[-1].get("nouvelle") != ("erreur", "ValueError", "charges manquantes"):
        raise AssertionError(f"{quoi}, le journal contient {_court(differences[1:], 300)} : la comparaison doit avoir lieu "
                             "MÊME quand l'ancienne plante (c'est peut-être la nouvelle qui a raison). Attendu : "
                             "ancienne = ('erreur', 'KeyError', \"'charges'\"), nouvelle = ('erreur', 'ValueError', "
                             "'charges manquantes').")
    try:
        score({"client_id": "C005", "charges": -10})
    except ValueError:
        pass
    if len(differences) != 2:
        raise AssertionError("Quand les deux versions lèvent ValueError (avec ou sans le même message), ce n'est pas une "
                             "différence : on compare le TYPE des erreurs.")
    if score.stats != {"appels": 4, "comparaisons": 4, "differences": 2}:
        raise AssertionError(f"Après 4 appels comparés dont 2 différents, stats vaut {score.stats}.")

    erreur = ValueError("tarif inconnu")

    def ancienne_qui_plante(x):
        raise erreur

    journal: list = []
    f = en_ombre(ancienne_qui_plante, lambda x: x, journal.append)
    try:
        f(3)
    except ValueError as exc:
        if exc is not erreur:
            raise AssertionError("Quand l'ancienne lève une exception, l'appelant doit recevoir CETTE exception-là "
                                 "(`raise exc`), pas une nouvelle fabriquée avec le même message.") from None
    if not journal or journal[0].get("nouvelle") != ("ok", 3):
        raise AssertionError(f"Quand l'ancienne plante et que la nouvelle répond 3, le journal contient {_court(journal)}.")

    # ------------------------------------------------------------------ la nouvelle plante : l'utilisateur ne voit rien
    journal = []
    f = en_ombre(lambda x: x + 1, lambda x: x / 0, journal.append)
    quoi = "Quand la nouvelle version plante (ZeroDivisionError)"
    resultat = _appel(lambda: f(1), f"{quoi}, f(1)",
                      "La nouvelle version ne doit JAMAIS gêner l'utilisateur : notez son erreur, renvoyez l'ancien "
                      "résultat.")
    if resultat != 2:
        raise AssertionError(f"{quoi}, f(1) renvoie {_court(resultat)} au lieu de 2 (le résultat de l'ancienne).")
    if journal != [{"arguments": ((1,), {}), "ancienne": ("ok", 2), "nouvelle": ("erreur", "ZeroDivisionError",
                                                                                 "division by zero")}]:
        raise AssertionError(f"{quoi}, le journal contient {_court(journal, 300)} : la nouvelle a planté, c'est une "
                             "différence à noter, `(\"erreur\", \"ZeroDivisionError\", \"division by zero\")`.")

    def interrompue(x):
        raise KeyboardInterrupt

    try:
        en_ombre(lambda x: x, interrompue, lambda d: None)(1)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Quand l'utilisateur appuie sur Ctrl+C pendant la nouvelle version, le programme doit "
                             "s'arrêter : `except Exception`, pas `except:` ni `except BaseException`.")

    # ------------------------------------------------------------------ la nouvelle modifie ses arguments
    def nouvelle_qui_abime(client):
        client["vu"] = True
        return 0

    journal = []
    f = en_ombre(lambda client: 1, nouvelle_qui_abime, journal.append)
    client = {"id": "C1"}
    f(client)
    if client != {"id": "C1"}:
        raise AssertionError("La nouvelle version a modifié le client de l'APPELANT : donnez-lui une copie à elle "
                             "(copy.deepcopy).")
    if journal and journal[0]["arguments"] != (({"id": "C1"},), {}):
        raise AssertionError(f"Le journal montre les arguments {journal[0]['arguments']} : la nouvelle les a modifiés après "
                             "coup. Donnez-lui une copie, et gardez une AUTRE copie pour le journal.")

    # ------------------------------------------------------------------ arguments nommés, egal
    journal = []
    f = en_ombre(lambda x, bonus=0: x + bonus, lambda x, bonus=0: x + 2 * bonus, journal.append)
    if _appel(lambda: f(1, bonus=2), "f(1, bonus=2)") != 3 or \
            journal != [{"arguments": ((1,), {"bonus": 2}), "ancienne": ("ok", 3), "nouvelle": ("ok", 5)}]:
        raise AssertionError(f"Avec un argument nommé, f(1, bonus=2) et le journal {_court(journal, 300)} : passez "
                             "`*args, **kwargs` aux deux versions, et notez \"arguments\": (args, kwargs).")
    journal = []
    proches = en_ombre(lambda x: x, lambda x: x + 0.02, journal.append, egal=lambda a, b: abs(a - b) < 0.05)
    proches(0.30)
    if journal:
        raise AssertionError("Avec egal=lambda a, b: abs(a - b) < 0.05, 0.30 et 0.32 sont égaux : comparez les valeurs avec "
                             "`egal(a, b)`, pas avec ==.")

    # ------------------------------------------------------------------ proportion et tirage
    _leve(lambda: en_ombre(lambda: 1, lambda: 1, print, proportion=1.5), ValueError, "en_ombre(..., proportion=1.5)",
          "Vérifiez proportion dès en_ombre(...), pas au premier appel.")
    _leve(lambda: en_ombre(lambda: 1, lambda: 1, print, proportion=-0.1), ValueError, "en_ombre(..., proportion=-0.1)")
    appels_nouvelle = []

    def nouvelle_comptee(x):
        appels_nouvelle.append(x)
        return -x

    tirages = _Tirages([0.05, 0.5, 0.09, 0.95])
    journal = []
    f = en_ombre(lambda x: x, nouvelle_comptee, journal.append, proportion=0.1, tirage=tirages)
    resultats = [_appel(lambda: f(i), f"Avec proportion=0.1, f({i})") for i in (1, 2, 3, 4)]
    if appels_nouvelle != [1, 3]:
        raise AssertionError(f"Avec proportion=0.1 et des tirages 0.05, 0.5, 0.09, 0.95, la nouvelle a été appelée pour "
                             f"{appels_nouvelle} au lieu de [1, 3] : comparez quand `tirage() < proportion`, et seulement "
                             "là (la nouvelle coûte cher).")
    if resultats != [1, 2, 3, 4] or f.stats != {"appels": 4, "comparaisons": 2, "differences": 2}:
        raise AssertionError(f"Avec proportion=0.1, résultats {resultats} et stats {f.stats} au lieu de [1, 2, 3, 4] et "
                             "{'appels': 4, 'comparaisons': 2, 'differences': 2}.")
    appels_nouvelle.clear()
    jamais = en_ombre(lambda x: x, nouvelle_comptee, journal.append, proportion=0.0)
    for i in range(20):
        jamais(i)
    if appels_nouvelle:
        raise AssertionError("Avec proportion=0.0, la nouvelle version ne doit jamais être appelée.")

    # ------------------------------------------------------------------ 300 clients : comme l'ancienne, toujours
    rng = random.Random(121)
    journal = []
    score = en_ombre(ancien_score, nouveau_score, journal.append)
    attendues = 0
    for i in range(300):
        client = {"client_id": f"C{i:03d}", "charges": rng.choice([rng.uniform(0, 150), "79,90", -1, 200])}
        if rng.random() < 0.8:
            client["contrat"] = rng.choice(CONTRATS)
        if rng.random() < 0.1:
            del client["charges"]
        client["appels"] = rng.randint(0, 6)
        direct, exc_direct = _noter(ancien_score, copy.deepcopy(client))
        nouveau, _ = _noter(nouveau_score, copy.deepcopy(client))
        via, exc_via = _noter(score, client)
        if via[:2] != direct[:2]:
            raise AssertionError(f"Pour le client {_court(client)}, l'appelant obtient {via} au lieu de {direct} : il doit "
                                 "TOUJOURS obtenir ce que donnerait l'ancienne.")
        if (direct[0] == "ok") != (nouveau[0] == "ok") or (direct[0] == "ok" and direct[1] != nouveau[1]) or \
                (direct[0] == "erreur" and direct[1] != nouveau[1]):
            attendues += 1
    if score.stats != {"appels": 300, "comparaisons": 300, "differences": attendues} or len(journal) != attendues:
        raise AssertionError(f"Sur 300 clients, stats vaut {score.stats} et le journal a {len(journal)} entrées, au lieu de "
                             f"{attendues} différences.")


SERIE = Serie(
    code="19",
    titre="Faire évoluer du code existant : figer, déprécier, extraire, migrer, adapter, comparer",
    module_eleve="defis.serie_19",
    module_corrige="defis.corrige_serie_19",
    defis=(
        Defi(116, "photographier", "🟢", "—", verifier_photographier),
        Defi(117, "remise", "🟢", "—", verifier_remise),
        Defi(118, "decider_relances", "🟡", "—", verifier_decider_relances),
        Defi(119, "lire_preferences", "🟡", "—", verifier_lire_preferences),
        Defi(120, "AdaptateurSMS", "🟡", "—", verifier_AdaptateurSMS),
        Defi(121, "en_ombre", "🔴", "—", verifier_en_ombre),
    ),
)

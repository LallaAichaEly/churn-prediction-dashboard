"""Vérifications automatiques de la série 09 (algorithmes de l'entretien technique).

Deux familles de messages :
- les erreurs de LOGIQUE classiques (un geste compté deux fois, une pile vide qu'on dépile, une
  fenêtre d'un jour de trop, une période contenue qui raccourcit tout, un tas qui garde les plus
  petits, un parcours en profondeur qui donne de fausses distances) ;
- les solutions JUSTES mais qui ne passent pas à l'échelle. Pour les repérer sans attendre des
  heures, le correcteur utilise des « espions » (une liste qui compte combien de fois on la lit,
  des scores qui comptent combien d'entre eux sont en mémoire en même temps) et des chronomètres
  sur des données de taille moyenne.
"""

from __future__ import annotations

import math
import random
import time
from datetime import date, timedelta

import numpy as np

from .verif import Defi, Serie


def _pas_none(resultat, quoi: str):
    if resultat is None:
        raise AssertionError(f"{quoi} : la fonction a retourné None. Avez-vous oublié `return` ?")
    return resultat


def _chronometrer(appel, seuil: float):
    """(résultat, durée) : le meilleur de deux essais, pour ne pas pénaliser une machine occupée."""
    duree, resultat = math.inf, None
    for _ in range(2):
        debut = time.perf_counter()
        resultat = appel()
        duree = min(duree, time.perf_counter() - debut)
        if duree < seuil:
            break
    return resultat, duree


def _milliers(n: int) -> str:
    """12500 → « 12 500 » (séparateur de milliers à la française)."""
    return f"{n:,}".replace(",", " ")


def _entier(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


# ============================================================================ Défi 56 : two sum

_OFFRES = [
    # (couts, budget, cas)
    ([40, 10, 25, 15, 30], 55, "l'exemple de la docstring"),
    ([20, 35, 20], 40, "deux gestes différents au même prix"),
    ([20, 35], 40, "un seul geste à 20 €, qu'on n'offre pas deux fois"),
    ([10, 25, 50], 60, "une réponse qui commence à la position 0"),
    ([5, 15, 25], 100, "aucune paire possible"),
    ([], 50, "un catalogue vide"),
    ([50], 50, "un seul geste"),
    ([30, 70, 45, 55, 100, 0], 100, "plusieurs réponses possibles, dont un geste gratuit"),
]


def _paires_valides(couts: list[int], budget: int) -> set[tuple[int, int]]:
    n = len(couts)
    return {(i, j) for i in range(n) for j in range(i + 1, n) if couts[i] + couts[j] == budget}


class _CatalogueEspion(list):
    """Une liste qui compte combien de fois on la parcourt, la lit case par case ou y cherche."""

    def __init__(self, valeurs):
        super().__init__(valeurs)
        self.parcours = self.lectures = self.tranches = self.recherches = 0

    def __iter__(self):
        self.parcours += 1
        return super().__iter__()

    def __getitem__(self, cle):
        if isinstance(cle, slice):
            self.tranches += 1
        else:
            self.lectures += 1
        return super().__getitem__(cle)

    def __contains__(self, valeur):
        self.recherches += 1
        return super().__contains__(valeur)

    def index(self, *args):
        self.recherches += 1
        return super().index(*args)

    def count(self, valeur):
        self.recherches += 1
        return super().count(valeur)


def _verifier_cas_offres(f, couts: list[int], budget: int, cas: str) -> None:
    copie = list(couts)
    appel = f"deux_offres({couts}, {budget})"
    obtenu = f(copie, budget)
    if copie != couts:
        raise AssertionError(
            f"Sur {cas}, votre fonction a modifié `couts` ({couts} est devenu {copie} : sort, remove, pop ?). Les "
            "positions renvoyées ne correspondraient plus au catalogue de l'appelant. Ne touchez pas à la liste reçue."
        )
    valides = _paires_valides(couts, budget)
    paire = tuple(obtenu) if isinstance(obtenu, (tuple, list)) else obtenu

    if not valides:
        if obtenu is None:
            return
        if isinstance(paire, tuple) and len(paire) == 2 and paire[0] == paire[1]:
            raise AssertionError(
                f"{appel} renvoie {obtenu!r} : c'est DEUX FOIS le même geste ({cas}). Cherchez le complément dans "
                "`vus` AVANT d'y enregistrer le geste courant : ainsi un geste ne peut pas se compléter lui-même."
            )
        raise AssertionError(
            f"{appel} renvoie {obtenu!r} au lieu de None ({cas}). « Pas de réponse », c'est None : pas -1, pas False, "
            "pas une liste vide. Une convention unique évite à l'appelant de deviner : `if paire is None:`."
        )

    exemple = min(valides)
    if obtenu is None:
        indice = ""
        if all(0 in p for p in valides):
            indice = (" La seule réponse utilise la position 0, or 0 est « faux » en Python : `if vus.get(complement):` "
                      "ne voit pas le geste n°0. Testez `if complement in vus:`.")
        raise AssertionError(f"{appel} renvoie None alors que {exemple} convient ({cas}).{indice}")
    if not (isinstance(paire, tuple) and len(paire) == 2 and all(_entier(x) for x in paire)):
        raise AssertionError(
            f"{appel} renvoie {obtenu!r} : attendu un tuple de deux POSITIONS entières (i, j), par exemple {exemple}."
        )
    i, j = paire
    if (i, j) in valides:
        return
    if (j, i) in valides:
        raise AssertionError(
            f"{appel} renvoie {obtenu!r} : les bons gestes, mais attendu i < j, soit ({j}, {i}). Le geste déjà vu "
            "(dans `vus`) a la plus petite position : renvoyez (vus[complement], j)."
        )
    if i == j:
        raise AssertionError(
            f"{appel} renvoie {obtenu!r} : c'est DEUX FOIS le geste n°{i} ({cas}). Cherchez le complément dans `vus` "
            "AVANT d'y enregistrer le geste courant."
        )
    if sorted(paire) in [sorted((couts[a], couts[b])) for a, b in valides]:
        raise AssertionError(
            f"{appel} renvoie {obtenu!r} : ce sont les COÛTS des gestes, pas leurs POSITIONS dans la liste (attendu par "
            f"exemple {exemple}). `for position, cout in enumerate(couts):` donne les deux."
        )
    n = len(couts)
    if 0 <= i < n and 0 <= j < n:
        raise AssertionError(
            f"{appel} renvoie {obtenu!r}, mais couts[{i}] + couts[{j}] = {couts[i] + couts[j]}, pas {budget} ({cas}). "
            f"Attendu par exemple {exemple}."
        )
    raise AssertionError(f"{appel} renvoie {obtenu!r} : ces positions n'existent pas dans un catalogue de {n} gestes.")


def verifier_deux_offres(f) -> None:
    for couts, budget, cas in _OFFRES:
        _verifier_cas_offres(f, couts, budget, cas)

    # Le pire cas (aucune paire) sur une liste espion : une boucle dans une boucle s'y voit.
    n = 400
    espion = _CatalogueEspion(range(0, 2 * n, 2))  # que des coûts pairs…
    resultat = f(espion, 2 * n + 1)  # … et un budget impair : aucune paire possible
    if resultat is not None:
        raise AssertionError(f"Avec {n} gestes de coût pair et un budget impair, aucune paire n'est possible : attendu None.")
    if espion.parcours > 3 or espion.lectures > 3 * n or espion.tranches > 3 or espion.recherches > 3:
        detail = []
        if espion.parcours > 3:
            detail.append(f"parcouru la liste {espion.parcours} fois")
        if espion.lectures > 3 * n:
            detail.append(f"lu {_milliers(espion.lectures)} cases (couts[i])")
        if espion.tranches > 3:
            detail.append(f"découpé {espion.tranches} tranches (couts[i + 1:])")
        if espion.recherches > 3:
            detail.append(f"cherché {espion.recherches} fois dans la liste (in, .index, .count)")
        raise AssertionError(
            f"Les réponses sont justes, mais pour {n} gestes votre fonction a {', '.join(detail)} : c'est une boucle "
            f"dans une boucle, O(n²). {n} gestes font {_milliers(n * (n - 1) // 2)} paires ; 1 million de gestes, 500 milliards "
            "(des heures). Chercher dans une LISTE la parcourt aussi en entier. Un dictionnaire `vus` (coût → position) "
            "répond en O(1) à « ai-je déjà vu le complément ? » : une seule boucle suffit."
        )

    # Et en vrai : 8 000 gestes, sans réponse (le pire cas).
    grand = list(range(0, 16_000, 2))
    random.Random(56).shuffle(grand)
    resultat, duree = _chronometrer(lambda: f(list(grand), 16_001), seuil=0.2)
    if resultat is not None:
        raise AssertionError("Avec 8 000 gestes de coût pair et un budget impair, attendu None.")
    if duree >= 0.2:
        raise AssertionError(
            f"Les réponses sont justes, mais 8 000 gestes prennent {duree:.1f} s (un dictionnaire : environ 0,002 s). "
            "Vous testez encore toutes les paires, O(n²) : une seule boucle et un dictionnaire `vus` suffisent."
        )


# ============================================================================ Défi 57 : pile

_FORMULES = [
    # (formule, attendu, cas, explication si votre réponse est fausse)
    ("log(1 + [total_charges] / {tenure_months})", True, "l'exemple de la docstring",
     "Les trois sortes de délimiteurs y sont bien appariées ; les autres caractères s'ignorent."),
    ("", True, "une formule vide", "Rien d'ouvert, rien à fermer : c'est valide."),
    ("tenure_months * 12", True, "une formule sans délimiteur", "Rien d'ouvert, rien à fermer : c'est valide."),
    ("{[()()]}", True, "des délimiteurs imbriqués", "Chaque fermant ferme le dernier ouvrant encore ouvert, du bon type."),
    ("(support_calls * 2", False, "une parenthèse jamais fermée",
     "À la fin, tout ce qui a été ouvert doit être fermé : la pile doit être VIDE (`return not pile`, pas `return True`)."),
    ("((a)", False, "une parenthèse ouverte de trop",
     "À la fin, la pile doit être VIDE : `return not pile`, pas `return True`."),
    ("[monthly_charges)", False, "un crochet fermé par une parenthèse",
     "En dépilant, comparez le sommet à l'ouvrant ATTENDU : PAIRES[\")\"] vaut \"(\", or le sommet est \"[\"."),
    (")a + b(", False, "une parenthèse fermée avant d'être ouverte",
     "Compter les ouvrants et les fermants ne suffit pas, l'ORDRE compte : un fermant qui arrive sur une pile vide "
     "rend la formule fausse immédiatement."),
    ("(a + [b) - c]", False, "des paires qui se croisent",
     "Compter type par type ne suffit pas : « ( [ ) ] » a un ouvrant et un fermant de chaque type. Il faut une PILE : "
     "un fermant doit correspondre au DERNIER ouvrant encore ouvert."),
    ("a + b)", False, "un fermant de trop", "Un fermant sans ouvrant correspondant : la pile est vide, c'est faux."),
    ("}", False, "un fermant seul", "Un fermant sur une pile vide : c'est faux."),
]


def _appeler_formule(f, formule: str, cas: str):
    affiche = formule if len(formule) < 60 else formule[:25] + "…" + formule[-25:]
    appel = f"parenthesage_valide({affiche!r})"
    try:
        return f(formule), appel
    except NotImplementedError:
        raise
    except IndexError as exc:
        raise AssertionError(
            f"{appel} plante (IndexError: {exc}) sur {cas} : vous dépilez (pile.pop() ou pile[-1]) une pile VIDE. "
            "Un fermant sans ouvrant rend la formule fausse : testez `if not pile: return False` avant de dépiler."
        ) from None
    except RecursionError:
        raise AssertionError(
            f"{appel} plante (RecursionError) sur {cas} : votre fonction s'appelle elle-même une fois par niveau, et "
            "Python s'arrête vers 1 000 niveaux. Une PILE explicite (une liste avec append / pop) n'a pas cette limite."
        ) from None


def _verifier_booleen(obtenu, appel: str) -> None:
    if obtenu is None:
        raise AssertionError(f"{appel} : la fonction a retourné None. Avez-vous oublié `return` (sur tous les chemins) ?")
    if not isinstance(obtenu, (bool, np.bool_)):
        raise AssertionError(
            f"{appel} renvoie {obtenu!r} ({type(obtenu).__name__}) : renvoyez un vrai booléen, True ou False "
            "(par exemple `return not pile`, et non `return pile`)."
        )


def verifier_parenthesage_valide(f) -> None:
    for formule, attendu, cas, explication in _FORMULES:
        obtenu, appel = _appeler_formule(f, formule, cas)
        _verifier_booleen(obtenu, appel)
        if bool(obtenu) != attendu:
            raise AssertionError(f"{appel} renvoie {obtenu} au lieu de {attendu} ({cas}). {explication}")

    profonds = [
        ("(" * 3000 + "[x]" + ")" * 3000, True, "3 000 niveaux d'imbrication"),
        ("([" * 2000 + "])" * 2000, True, "4 000 niveaux alternant ( et ["),
        ("{" * 3000 + "}" * 2999, False, "3 000 accolades ouvertes et une fermée de moins"),
    ]
    for formule, attendu, cas in profonds:
        obtenu, appel = _appeler_formule(f, formule, cas)
        _verifier_booleen(obtenu, appel)
        if bool(obtenu) != attendu:
            raise AssertionError(f"{appel} renvoie {obtenu} au lieu de {attendu} ({cas}).")


# ============================================================================ Défi 58 : fenêtre glissante

_APPELS = [
    # (jours, largeur, cas)
    ([1, 2, 2, 9, 3, 15, 8, 10], 7, "l'exemple de la docstring"),
    ([], 7, "aucun appel"),
    ([42], 7, "un seul appel"),
    ([1, 8], 7, "deux appels à 7 jours d'écart, jamais dans les mêmes 7 jours"),
    ([1, 7], 7, "deux appels à 6 jours d'écart, dans les mêmes 7 jours"),
    ([5, 3, 3, 4, 3], 1, "largeur = 1 : le jour le plus chargé"),
    ([10, 10, 10, 10], 3, "quatre appels le même jour"),
    ([30, 1, 29, 2, 28, 31, 3], 7, "des appels dans le désordre"),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, "un appel par jour"),
    ([100, 1, 50, 2, 51, 52, 3, 53, 4], 4, "deux pics, le second plus fort"),
]


def _pic_force_brute(jours: list[int], largeur: int) -> int:
    """La référence lisible (O(n²), suffisant pour les petits cas)."""
    if largeur < 1:
        return 0
    return max((sum(d <= x <= d + largeur - 1 for x in jours) for d in set(jours)), default=0)


def _pic_rapide(jours: list[int], largeur: int) -> int:
    t, meilleur, gauche = sorted(jours), 0, 0
    for droite, jour in enumerate(t):
        while jour - t[gauche] >= largeur:
            gauche += 1
        meilleur = max(meilleur, droite - gauche + 1)
    return meilleur


def _verifier_cas_appels(f, jours: list[int], largeur: int, cas: str) -> None:
    copie = list(jours)
    appel = f"pic_appels({jours}, largeur={largeur})"
    obtenu = _pas_none(f(copie, largeur), appel)
    if copie != jours:
        raise AssertionError(
            f"Sur {cas}, votre fonction a modifié `jours` (devenu {copie}) : jours.sort() trie la liste DE L'APPELANT, "
            "qui l'affiche peut-être dans l'ordre d'arrivée. sorted(jours) renvoie une copie triée."
        )
    if not _entier(obtenu):
        raise AssertionError(f"{appel} renvoie {obtenu!r} ({type(obtenu).__name__}) : attendu un nombre entier d'appels.")
    attendu = _pic_force_brute(jours, largeur)
    if obtenu == attendu:
        return
    # D'abord le tri : un code juste sur la liste triée n'a pas d'erreur de fenêtre.
    if jours != sorted(jours):
        try:
            sur_trie = f(sorted(jours), largeur)
        except Exception:  # noqa: BLE001 — simple sonde
            sur_trie = None
        if sur_trie == attendu:
            raise AssertionError(
                f"{appel} renvoie {obtenu} au lieu de {attendu} ({cas}), alors que la même liste TRIÉE donne la bonne "
                "réponse. La fenêtre glissante suppose des jours dans l'ordre : commencez par t = sorted(jours)."
            )
    if jours and obtenu == _pic_force_brute(jours, largeur + 1):
        raise AssertionError(
            f"{appel} renvoie {obtenu} au lieu de {attendu} ({cas}) : votre fenêtre couvre {largeur + 1} jours. "
            f"{largeur} jours consécutifs à partir du jour d, c'est d à d + {largeur - 1} : deux appels sont dans la même "
            f"fenêtre si leur écart est au plus {largeur - 1}. La condition pour retirer l'appel de gauche est "
            "`t[droite] - t[gauche] >= largeur` (et non `>`)."
        )
    if jours and obtenu == _pic_force_brute(jours, largeur - 1):
        raise AssertionError(
            f"{appel} renvoie {obtenu} au lieu de {attendu} ({cas}) : votre fenêtre ne couvre que {largeur - 1} jours. "
            f"Du jour d au jour d + {largeur - 1} INCLUS, cela fait bien {largeur} jours : retirez l'appel de gauche "
            "quand `t[droite] - t[gauche] >= largeur`."
        )
    raise AssertionError(f"{appel} renvoie {obtenu} au lieu de {attendu} ({cas}).")


def verifier_pic_appels(f) -> None:
    for jours, largeur, cas in _APPELS:
        _verifier_cas_appels(f, jours, largeur, cas)

    # Et en vrai : 8 000 appels concentrés sur 10 jours (des milliers dans chaque fenêtre).
    hasard = random.Random(58)
    jours = [hasard.randrange(10) for _ in range(8000)]
    attendu = _pic_rapide(jours, 7)
    obtenu, duree = _chronometrer(lambda: f(list(jours), 7), seuil=0.25)
    if obtenu != attendu:
        raise AssertionError(f"Sur 8 000 appels répartis sur 10 jours, le pic sur 7 jours vaut {attendu}, pas {obtenu}.")
    if duree >= 0.25:
        raise AssertionError(
            f"Les réponses sont justes, mais 8 000 appels prennent {duree:.1f} s (deux pointeurs : environ 0,003 s). "
            "Vous recomptez toute la fenêtre pour chaque appel : O(n²). Sur la liste triée, faites seulement AVANCER "
            "`gauche` quand l'appel de gauche sort de la fenêtre : chaque appel entre une fois et sort une fois."
        )


# ============================================================================ Défi 59 : intervalles


def _j(mois: int, jour: int, annee: int = 2024) -> date:
    return date(annee, mois, jour)


_PERIODES = [
    # (periodes, attendu, cas, explication si faux)
    ([(_j(3, 1), _j(4, 30)), (_j(1, 1), _j(1, 31)), (_j(2, 1), _j(3, 31)), (_j(6, 1), _j(6, 30))],
     [(_j(1, 1), _j(4, 30)), (_j(6, 1), _j(6, 30))], "l'exemple de la docstring", ""),
    ([], [], "aucune période", "Sans période, renvoyez une liste vide."),
    ([(_j(5, 1), _j(8, 31))], [(_j(5, 1), _j(8, 31))], "une seule période", ""),
    ([(_j(1, 1), _j(1, 31)), (_j(2, 1), _j(2, 29))], [(_j(1, 1), _j(2, 29))], "un renouvellement le lendemain",
     "Un contrat qui finit le 31 janvier et un autre qui commence le 1er février : le client n'a jamais été sans "
     "abonnement, c'est une seule période. Fusionnez quand `debut <= fin + timedelta(days=1)`."),
    ([(date(2023, 12, 1), date(2023, 12, 31)), (_j(1, 1), _j(3, 31))], [(date(2023, 12, 1), _j(3, 31))],
     "un renouvellement au 1er janvier",
     "Le lendemain du 31 décembre 2023 est le 1er janvier 2024 : calculez-le avec `fin + timedelta(days=1)`, qui gère "
     "les fins de mois et d'année (pas en ajoutant 1 au numéro du jour)."),
    ([(_j(1, 1), _j(1, 30)), (_j(2, 1), _j(2, 29))], [(_j(1, 1), _j(1, 30)), (_j(2, 1), _j(2, 29))],
     "une interruption d'un seul jour (le 31 janvier)",
     "Le 31 janvier, le client n'avait aucun contrat : ce sont deux périodes. Seul le LENDEMAIN de la fin (pas le "
     "surlendemain) prolonge la période."),
    ([(_j(1, 1), _j(1, 15)), (_j(1, 15), _j(1, 31))], [(_j(1, 1), _j(1, 31))], "deux périodes qui partagent un jour",
     "Elles se touchent le 15 janvier : une seule période."),
    ([(_j(1, 1), _j(12, 31)), (_j(3, 1), _j(3, 31)), (_j(5, 1), _j(5, 31))], [(_j(1, 1), _j(12, 31))],
     "des périodes entièrement contenues dans une autre",
     "Le contrat de mars est CONTENU dans celui de l'année : la fin de la période en cours reste le 31 décembre. "
     "Prenez `max(fin_en_cours, fin)`, pas simplement la fin de la dernière période lue."),
    ([(_j(4, 1), _j(4, 30)), (_j(4, 1), _j(4, 30))], [(_j(4, 1), _j(4, 30))], "une saisie en double", ""),
    ([(_j(6, 1), _j(6, 30)), (_j(1, 1), _j(1, 31)), (_j(7, 1), _j(7, 31))],
     [(_j(1, 1), _j(1, 31)), (_j(6, 1), _j(7, 31))], "des contrats dans le désordre", ""),
    ([(_j(3, 2), _j(3, 2)), (_j(3, 1), _j(3, 1)), (_j(3, 4), _j(3, 4))],
     [(_j(3, 1), _j(3, 2)), (_j(3, 4), _j(3, 4))], "des contrats d'un seul jour", ""),
]


def _normaliser_periodes(obtenu):
    if not isinstance(obtenu, list):
        return None
    try:
        return [tuple(p) for p in obtenu]
    except TypeError:
        return None


def verifier_fusionner_periodes(f) -> None:
    for periodes, attendu, cas, explication in _PERIODES:
        copie = list(periodes)
        appel = "fusionner_periodes(...)"
        obtenu = _pas_none(f(copie), appel)
        if copie != periodes:
            raise AssertionError(
                f"Sur {cas}, votre fonction a modifié la liste `periodes` (periodes.sort() ?). Le CRM ou l'appelant en a "
                "encore besoin telle quelle : parcourez sorted(periodes), une copie triée."
            )
        lignes = _normaliser_periodes(obtenu)
        if lignes is None:
            raise AssertionError(f"Sur {cas}, attendu une liste de tuples (debut, fin), obtenu {type(obtenu).__name__}.")
        if lignes == attendu:
            continue
        if any(len(p) != 2 for p in lignes):
            raise AssertionError(f"Sur {cas}, chaque période doit être un couple (debut, fin) : obtenu {lignes}.")
        if any(type(x) is not date for p in lignes for x in p):
            types = sorted({type(x).__name__ for p in lignes for x in p})
            raise AssertionError(
                f"Sur {cas}, les bornes sont de type {types} : renvoyez les objets `date` reçus (sans les convertir)."
            )
        if sorted(lignes) == attendu:
            raise AssertionError(
                f"Sur {cas}, les bonnes périodes, mais pas triées par date de début : {lignes}. Parcourez sorted(periodes)."
            )
        if periodes != sorted(periodes):
            try:
                sur_trie = _normaliser_periodes(f(sorted(periodes)))
            except Exception:  # noqa: BLE001 — simple sonde
                sur_trie = None
            if sur_trie == attendu:
                raise AssertionError(
                    f"Sur {cas}, résultat faux ({_afficher(lignes)}), alors que les mêmes contrats TRIÉS donnent la bonne réponse. "
                    "La fusion suppose que chaque période commence après la précédente : commencez par sorted(periodes)."
                )
        raise AssertionError(
            f"Sur {cas}, attendu {_afficher(attendu)}, obtenu {_afficher(lignes)}. "
            f"{explication or _diagnostic_periodes(lignes)}".rstrip()
        )


def _diagnostic_periodes(lignes) -> str:
    """Cherche dans VOTRE résultat deux périodes qui auraient dû être fusionnées."""
    for (_, fin), (debut, _) in zip(lignes, lignes[1:]):
        if debut == fin + timedelta(days=1):
            return (f"Une période finit le {fin:%d/%m/%Y} et la suivante commence le LENDEMAIN : le client n'a jamais été "
                    "sans abonnement, c'est une seule période. Fusionnez quand `debut <= fin + timedelta(days=1)` "
                    "(timedelta gère les fins de mois et d'année, pas un calcul sur le numéro du jour).")
        if debut <= fin:
            return (f"Une période finit le {fin:%d/%m/%Y} et la suivante commence le {debut:%d/%m/%Y}, avant ou le même "
                    "jour : elles se chevauchent et doivent être fusionnées (`<=`, pas `<`).")
    return ""


def _afficher(periodes) -> str:
    return "[" + ", ".join(f"({d:%d/%m/%Y} → {f:%d/%m/%Y})" for d, f in periodes) + "]"


# ============================================================================ Défi 60 : tas


class _Compteur:
    """Combien de scores sont en mémoire en même temps (vivants), au maximum."""

    vivants = 0
    maximum = 0

    @classmethod
    def remettre_a_zero(cls) -> None:
        cls.vivants = cls.maximum = 0


class _Score(tuple):
    """Un tuple (customer_id, proba) ordinaire… qui se signale à sa création et à sa destruction."""

    __slots__ = ()

    def __new__(cls, client: str, proba: float):
        score = super().__new__(cls, (client, proba))
        _Compteur.vivants += 1
        _Compteur.maximum = max(_Compteur.maximum, _Compteur.vivants)
        return score

    def __del__(self):
        try:
            _Compteur.vivants -= 1
        except Exception:  # noqa: BLE001 — à la fermeture de Python, le module peut déjà être détruit
            pass


def _flux(scores):
    """Un vrai générateur : un seul passage, pas de len(), pas d'indexation."""
    for client, proba in scores:
        yield _Score(client, proba)


def _top_reference(scores, k: int):
    return sorted(scores, key=lambda s: s[1], reverse=True)[:k] if k > 0 else []  # sorted est stable


_TOPS = [
    # (scores, k, cas)
    ([("C1", 0.2), ("C2", 0.9), ("C3", 0.5), ("C4", 0.9)], 3, "l'exemple de la docstring"),
    ([("C9", 0.7), ("C10", 0.7), ("C2", 0.1), ("C11", 0.7), ("C3", 0.65)], 2,
     "trois clients à égalité, arrivés dans l'ordre C9, C10 puis C11"),
    ([("C5", 0.3), ("C6", 0.8), ("C7", 0.6)], 10, "moins de k clients dans le flux"),
    ([("C5", 0.3), ("C6", 0.8)], 0, "k = 0"),
    ([], 3, "un flux vide"),
    ([(f"C{i}", round(0.1 * (i % 7), 1)) for i in range(30)], 5, "30 clients et beaucoup d'égalités"),
]


def _appeler_top(f, scores, k: int, cas: str):
    try:
        return f(_flux(scores), k)
    except NotImplementedError:
        raise
    except TypeError as exc:
        if "generator" in str(exc) or "len()" in str(exc):
            raise AssertionError(
                f"Sur {cas}, votre fonction plante (TypeError: {exc}) : `flux` est un GÉNÉRATEUR. Il n'a pas de len() et "
                "ne s'indexe pas (flux[0]) : on ne peut que le parcourir, une fois, avec `for client, proba in flux:`."
            ) from None
        raise


def _verifier_cas_top(f, scores, k: int, cas: str) -> None:
    obtenu = _pas_none(_appeler_top(f, scores, k, cas), "top_k_flux")
    attendu = _top_reference(scores, k)
    if not isinstance(obtenu, list):
        raise AssertionError(f"Sur {cas}, attendu une liste, obtenu {type(obtenu).__name__}.")
    lignes = [tuple(x) if isinstance(x, (tuple, list)) else x for x in obtenu]
    if lignes == attendu:
        return
    if any(not (isinstance(x, tuple) and len(x) == 2) for x in lignes):
        raise AssertionError(
            f"Sur {cas}, chaque élément doit être le tuple d'origine (customer_id, proba) : obtenu {lignes[:3]}… "
            "Si votre tas contient (proba, position, client), reconstruisez (client, proba) à la fin."
        )
    if lignes and all(isinstance(x[0], float) and isinstance(x[1], str) for x in lignes):
        raise AssertionError(
            f"Sur {cas}, les éléments sont dans l'ordre (proba, customer_id) : renvoyez (customer_id, proba), comme dans le flux."
        )
    if not attendu:
        raise AssertionError(f"Sur {cas}, attendu une liste vide, obtenu {lignes}.")
    if len(scores) > len(attendu) and k > 0 and len(lignes) == 0:
        raise AssertionError(
            f"Sur {cas}, liste vide au lieu de {attendu}. Un générateur ne se parcourt qu'UNE fois : si vous l'avez déjà "
            "parcouru (un max(), un list(), une première boucle…), le second passage ne voit plus rien."
        )
    plus_petits = sorted(scores, key=lambda s: s[1])[:k]
    if k < len(scores) and sorted(lignes) == sorted(plus_petits) and sorted(lignes) != sorted(attendu):
        raise AssertionError(
            f"Sur {cas}, vous renvoyez les k PLUS FAIBLES probabilités ({lignes}). heapq est un tas MIN : son sommet "
            "tas[0] est le plus PETIT. C'est voulu : dans un tas des k meilleurs, le sommet est le moins bon des "
            "meilleurs, celui qu'un nouveau venu plus fort doit remplacer (et pas l'inverse)."
        )
    if len(lignes) != len(attendu):
        raise AssertionError(f"Sur {cas}, {len(lignes)} clients au lieu de {len(attendu)} (k = {k}) : attendu {attendu}.")
    if sorted(lignes) == sorted(attendu):
        if [s[1] for s in lignes] != [s[1] for s in attendu]:
            raise AssertionError(
                f"Sur {cas}, les bons clients, mais pas triés par probabilité DÉCROISSANTE : {lignes}. Un tas n'est pas "
                "une liste triée (seul tas[0] est garanti) : triez-le à la fin, du plus risqué au moins risqué."
            )
        raise AssertionError(
            f"Sur {cas}, à égalité de probabilité, l'ordre doit être celui d'ARRIVÉE dans le flux : attendu {attendu}, "
            f"obtenu {lignes}. Départager par l'identifiant ne marche pas (en texte, « C10 » < « C9 »), ni par la "
            "position croissante (avec reverse=True, la position la plus GRANDE passe devant). Mettez l'OPPOSÉ de la "
            "position dans vos tuples, (proba, -position, client) : à proba égale, le premier arrivé est le plus grand."
        )
    probas_seuil = attendu[-1][1]
    if [s[1] for s in lignes] == [s[1] for s in attendu]:
        raise AssertionError(
            f"Sur {cas}, à égalité de probabilité ({probas_seuil}), c'est le client arrivé le PREMIER qui doit être gardé : "
            f"attendu {attendu}, obtenu {lignes}. Avec des tuples (proba, position, client), un nouveau venu à égalité bat "
            "l'ancien ; avec (proba, -position, client), il ne le bat pas."
        )
    raise AssertionError(f"Sur {cas}, attendu {attendu}, obtenu {lignes}.")


def verifier_top_k_flux(f) -> None:
    for scores, k, cas in _TOPS:
        _verifier_cas_top(f, scores, k, cas)

    # Et en vrai : 100 000 scores. Les _Score comptent combien d'entre eux sont en mémoire à la fois.
    hasard = random.Random(60)
    scores = [(f"C{i:06d}", round(hasard.random(), 3)) for i in range(100_000)]
    k = 5
    _Compteur.remettre_a_zero()
    debut = time.perf_counter()
    obtenu = _pas_none(_appeler_top(f, scores, k, "100 000 scores"), "top_k_flux")
    duree = time.perf_counter() - debut
    maximum = _Compteur.maximum
    lignes = [tuple(x) for x in obtenu]
    del obtenu
    attendu = _top_reference(scores, k)
    if lignes != attendu:
        raise AssertionError(f"Sur 100 000 scores avec k = {k}, attendu {attendu}, obtenu {lignes}.")
    if maximum > 10 * k + 100:
        raise AssertionError(
            f"Le résultat est juste, mais {_milliers(maximum)} scores ont été en mémoire EN MÊME TEMPS pour en garder {k}. "
            "list(flux) ou sorted(flux) chargent tout le flux : avec les 50 millions de clients de la nuit, la machine "
            "manque de mémoire. Gardez seulement un tas de k éléments (heapq), et laissez partir les autres."
        )
    if duree >= 2.0:
        raise AssertionError(
            f"Le résultat est juste, mais 100 000 scores prennent {duree:.1f} s. Retriez-vous votre liste des k meilleurs "
            "à chaque nouveau score ? Avec un tas, chaque score coûte O(log k)."
        )


# ============================================================================ Défi 61 : graphe


def _bfs_reference(parrainages, partis, distance_max: int) -> dict[str, int]:
    voisins: dict[str, list[str]] = {}
    for a, b in parrainages:
        voisins.setdefault(a, []).append(b)
        voisins.setdefault(b, []).append(a)
    distance = {p: 0 for p in partis}
    file = list(distance)
    for client in file:  # une liste parcourue en avançant : une file sans pop(0)
        if distance[client] < distance_max:
            for v in voisins.get(client, ()):
                if v not in distance:
                    distance[v] = distance[client] + 1
                    file.append(v)
    return {c: d for c, d in distance.items() if d > 0}


_RESEAUX = [
    # (parrainages, partis, distance_max, cas, explication si faux)
    ([("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")], ["A", "F"], 2, "l'exemple de la docstring", ""),
    ([("B", "A"), ("C", "B")], ["A"], 2, "un client parti qui est le FILLEUL",
     "Le lien de parrainage vaut dans les DEUX sens : ajoutez b aux voisins de a ET a aux voisins de b."),
    ([("P1", "X1"), ("X1", "X2"), ("X2", "X3"), ("X3", "P2")], ["P1", "P2"], 3, "une chaîne entre deux clients partis",
     "C'est le client parti le PLUS PROCHE qui compte : X1 est à 1 lien de P1, X3 à 1 lien de P2. Il faut UN seul "
     "parcours en LARGEUR depuis TOUS les partis à la fois (tous dans la file au départ, à distance 0), avec une "
     "vraie file : deque.popleft() sert le PREMIER arrivé, alors que liste.pop() prend le DERNIER (c'est une pile, un "
     "parcours en profondeur). Un parcours par client parti, qui écrase les distances, donne aussi ce résultat."),
    ([("P", "A"), ("A", "B"), ("B", "C"), ("P", "C"), ("C", "D")], ["P"], 3, "deux chemins de longueurs différentes",
     "C est voisin DIRECT de P (distance 1), même si on peut aussi l'atteindre par P → A → B → C. Un parcours en "
     "PROFONDEUR (récursif, ou une pile avec pop()) suit le premier chemin trouvé, pas le plus court. Le parcours en "
     "LARGEUR (une file : deque.popleft()) traite tous les clients à distance 1, puis 2… : la première fois qu'on "
     "atteint un client, c'est par le plus court chemin."),
    ([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")], {"A"}, 3, "un cycle (A → B → C → A)",
     "Le réseau a des cycles : retenez les clients déjà atteints (le dictionnaire `distance` sert d'ensemble « déjà "
     "vus ») et ne les remettez jamais dans la file."),
    ([("A", "B"), ("B", "C")], ["A", "B"], 2, "deux clients partis voisins",
     "Un client parti n'est pas « exposé », il est déjà parti : il n'apparaît jamais dans le résultat."),
    ([("A", "B"), ("B", "C"), ("C", "D")], ["A"], 1, "distance_max = 1",
     "Avec distance_max = 1, seuls les voisins DIRECTS des partis comptent."),
    ([("A", "B")], {"Z"}, 2, "un client parti absent du réseau", "Z n'a aucun lien : personne n'est exposé."),
    ([], ["A"], 2, "aucun parrainage", "Sans lien, personne n'est exposé : renvoyez un dictionnaire vide."),
    ([("A", "B"), ("A", "B"), ("B", "C")], ["A"], 2, "un lien enregistré deux fois", ""),
]


def _appeler_reseau(f, parrainages, partis, distance_max: int, cas: str):
    copie_liens = list(parrainages)
    copie_partis = type(partis)(partis)
    try:
        obtenu = f(copie_liens, copie_partis, distance_max)
    except NotImplementedError:
        raise
    except RecursionError:
        raise AssertionError(
            f"Sur {cas}, votre fonction plante (RecursionError) : un parcours récursif s'appelle une fois par client de "
            "la chaîne, et Python s'arrête vers 1 000 niveaux. Le parcours en largeur avec une file (collections.deque) "
            "n'a pas cette limite, et il donne en plus les plus COURTES distances."
        ) from None
    if copie_liens != list(parrainages) or copie_partis != partis:
        raise AssertionError(
            f"Sur {cas}, votre fonction a modifié `parrainages` ou `partis` (un append, un remove, un add ?). Travaillez "
            "sur vos propres structures (voisins, distance, file)."
        )
    return obtenu


def _verifier_reseau(f, parrainages, partis, distance_max: int, cas: str, explication: str) -> None:
    obtenu = _pas_none(_appeler_reseau(f, parrainages, partis, distance_max, cas), "clients_exposes")
    attendu = _bfs_reference(parrainages, partis, distance_max)
    if not isinstance(obtenu, dict):
        raise AssertionError(f"Sur {cas}, attendu un dictionnaire {{client: distance}}, obtenu {type(obtenu).__name__}.")
    if obtenu == attendu:
        return
    partis_presents = sorted(c for c in obtenu if c in set(partis))
    if partis_presents:
        raise AssertionError(
            f"Sur {cas}, les clients partis {partis_presents} sont dans le résultat : un client parti n'est pas exposé, il "
            "est déjà parti. Retirez les distances 0 à la fin, et ne donnez jamais de distance à un parti."
        )
    trop_loin = sorted(c for c, d in obtenu.items() if _entier(d) and d > distance_max)
    if trop_loin:
        raise AssertionError(
            f"Sur {cas}, {trop_loin} sont à plus de distance_max = {distance_max} liens : arrêtez d'explorer à partir "
            "d'un client dont la distance vaut déjà distance_max."
        )
    if not explication and _sens_unique(f):
        explication = ("Votre parcours ne suit les liens que du parrain vers le filleul : or un client parti peut être le "
                       "FILLEUL. Le lien vaut dans les DEUX sens : ajoutez b aux voisins de a ET a aux voisins de b.")
    raise AssertionError(
        f"Sur {cas}, attendu {dict(sorted(attendu.items()))}, obtenu {dict(sorted(obtenu.items()))}. {explication}".rstrip()
    )


def _sens_unique(f) -> bool:
    """Sonde : votre fonction voit-elle le parrain d'un filleul parti ?"""
    try:
        return f([("PARRAIN", "FILLEUL")], ["FILLEUL"], 1) == {}
    except Exception:  # noqa: BLE001 — simple sonde
        return False


def verifier_clients_exposes(f) -> None:
    for parrainages, partis, distance_max, cas, explication in _RESEAUX:
        _verifier_reseau(f, parrainages, partis, distance_max, cas, explication)

    # Une longue chaîne de parrainages : 3 000 clients, chacun parrainé par le précédent.
    chaine = [(f"C{i}", f"C{i + 1}") for i in range(3000)]
    _verifier_reseau(f, chaine, ["C0"], 5000, "une chaîne de 3 000 parrainages", "")

    # Et en vrai : 4 000 clients, 8 000 liens, 5 clients partis, tout le réseau à portée.
    hasard = random.Random(61)
    clients = [f"C{i:05d}" for i in range(4000)]
    liens = [(clients[i], clients[i + 1]) for i in range(len(clients) - 1)]  # un réseau connexe…
    liens += [tuple(hasard.sample(clients, 2)) for _ in range(8000 - len(liens))]  # … plus des liens au hasard
    partis = hasard.sample(clients, 5)
    attendu = _bfs_reference(liens, partis, 50)
    obtenu, duree = _chronometrer(lambda: f(list(liens), list(partis), 50), seuil=0.3)
    if obtenu != attendu:
        if not isinstance(obtenu, dict):
            raise AssertionError(f"Sur un réseau de 4 000 clients, attendu un dictionnaire, obtenu {type(obtenu).__name__}.")
        faux = next(((c, obtenu.get(c), d) for c, d in attendu.items() if obtenu.get(c) != d), None)
        detail = f" (par exemple {faux[0]} : {faux[1]} au lieu de {faux[2]})" if faux else ""
        trop_long = any(_entier(obtenu.get(c)) and obtenu[c] > d for c, d in attendu.items())
        indice = (" Certaines distances sont trop LONGUES : votre parcours atteint des clients par un chemin qui n'est pas "
                  "le plus court. C'est le signe d'un parcours en profondeur (une pile : liste.pop()) : utilisez une FILE, "
                  "collections.deque et popleft(), pour traiter les clients par distance croissante.") if trop_long else ""
        raise AssertionError(f"Sur un réseau de 4 000 clients et 8 000 liens, les distances sont fausses{detail}.{indice}")
    if duree >= 0.3:
        raise AssertionError(
            f"Les distances sont justes, mais 4 000 clients et 8 000 liens prennent {duree:.1f} s (un parcours en largeur : "
            "environ 0,01 s). Reparcourez-vous `parrainages` pour trouver les voisins de CHAQUE client (O(V × E)) ? "
            "Construisez une fois le dictionnaire `voisins`. Autres suspects : `file.pop(0)` (O(n), deque.popleft() est "
            "en O(1)) et `if voisin in deja_vus` sur une LISTE (un set ou un dict répond en O(1))."
        )


# ============================================================================ La série

SERIE = Serie(
    code="09",
    titre="Penser en algorithmes : les classiques de l'entretien technique",
    module_eleve="defis.serie_09",
    module_corrige="defis.corrige_serie_09",
    defis=(
        Defi(56, "deux_offres", "🟢", "—", verifier_deux_offres),
        Defi(57, "parenthesage_valide", "🟢", "—", verifier_parenthesage_valide),
        Defi(58, "pic_appels", "🟡", "—", verifier_pic_appels),
        Defi(59, "fusionner_periodes", "🟡", "—", verifier_fusionner_periodes),
        Defi(60, "top_k_flux", "🟡", "—", verifier_top_k_flux),
        Defi(61, "clients_exposes", "🔴", "—", verifier_clients_exposes),
    ),
)


"""Vérifications automatiques du projet 7 (le rapport du mois).

Le correcteur place des abonnés PILE sur les frontières du mois (arrivé le 1er, parti le 1er, parti le
dernier jour, parti la veille), essaie décembre et février 2028, demande « 2026-1 » et « 2026-10-01 »,
fait écrire 0,125 €, 2,675 € et -0,001 €, coupe la courbe par des mois inconnus, lit votre SVG avec un
vrai lecteur XML et votre page avec un vrai lecteur HTML, glisse un client nommé `<script>…</script>`
dans le rapport, relit votre code à la recherche d'un fichier ouvert sans encodage, et rejoue 13
versions boguées d'`indicateurs_du_mois` pour vous dire LAQUELLE vous avez écrite.

Lire ce fichier est permis (les cas testés sont des indices), mais essayez d'abord seul·e.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import math
import os
import re
import tempfile
import textwrap
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import fields
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType

from .checks_reentrainement import _appel, _leve_value_error
from .checks_retention import _appelle, _lancer, _pas_none
from .exemples_rapport import CHEMIN_EXEMPLE
from .rapport_corrige import formats as ref_formats
from .rapport_corrige import graphique as ref_graphique
from .rapport_corrige.cli import construire_rapport as ref_construire_rapport
from .rapport_corrige.indicateurs import indicateurs_du_mois as ref_indicateurs_du_mois
from .rapport_corrige.indicateurs import par_segment as ref_par_segment
from .rapport_corrige.modeles import (DEPARTS_AFFICHES, ECHELLE_PAR_DEFAUT, ESPACE_MILLIERS, ESPACE_UNITE, HAUTEUR,
                                      LARGEUR, MARGE_BAS, MARGE_DROITE, MARGE_GAUCHE, MARGE_HAUT, MOIS_HISTORIQUE,
                                      SVG_NS, TIRET, Indicateurs)
from .rapport_corrige.periodes import decaler as ref_decaler
from .rapport_corrige.periodes import fin_du_mois as ref_fin_du_mois
from .verif import Etape, Projet

# ---------------------------------------------------------------------------- outils


def _modeles(module: ModuleType) -> ModuleType:
    """Le `modeles.py` du MÊME paquet que le module testé (le vôtre ou celui du corrigé)."""
    return importlib.import_module(f"{module.__package__}.modeles")


def _voisin(module: ModuleType, nom: str) -> ModuleType:
    """Un autre fichier du MÊME paquet (vos étapes précédentes, ou celles du corrigé)."""
    return importlib.import_module(f"{module.__package__}.{nom}")


@contextlib.contextmanager
def _dossier_temporaire():
    with tempfile.TemporaryDirectory(prefix="projet_rapport_") as dossier:
        yield Path(dossier)


def _jour(texte: str | None) -> date | None:
    return date.fromisoformat(texte) if texte else None


def _fabrique(M: ModuleType):
    """Un fabricant d'abonnés avec la classe `Abonne` du paquet testé."""
    def abonne(client: str, debut: str, fin: str | None = None, mensualite: float = 50.0,
               segment: str = "Particulier", nom: str | None = None):
        return M.Abonne(client, nom if nom is not None else f"Client {client}", segment, _jour(debut), _jour(fin),
                        mensualite)
    return abonne


_CHAMPS = tuple(f.name for f in fields(Indicateurs))


def _valeurs(indicateurs, quoi: str) -> tuple:
    """Les champs d'un Indicateurs (le vôtre ou celui du corrigé), comparables entre eux."""
    indicateurs = _pas_none(indicateurs, quoi)
    manquants = [c for c in _CHAMPS if not hasattr(indicateurs, c)]
    if manquants:
        raise AssertionError(f"{quoi} doit renvoyer un `Indicateurs(...)` (modeles.py), pas "
                             f"{type(indicateurs).__name__} (il manque {', '.join(manquants)}).")
    return tuple(getattr(indicateurs, c) for c in _CHAMPS)


def _egaux(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return type(a) is type(b) and a == b if isinstance(a, date) or isinstance(b, date) else a == b


def _memes_valeurs(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and all(_egaux(x, y) for x, y in zip(a, b))


def _montrer_abonnes(abonnes) -> str:
    return ", ".join(f"{a.client} ({a.debut:%d/%m/%Y} → {a.fin:%d/%m/%Y})" if a.fin else
                     f"{a.client} ({a.debut:%d/%m/%Y} → toujours là)" for a in abonnes)


# ============================================================================ Étape 1 : les mois


def verifier_periodes(module: ModuleType) -> None:
    lire, fin, decaler, nom = module.lire_mois, module.fin_du_mois, module.decaler, module.nom_du_mois

    for texte, attendu in (("2026-10", date(2026, 10, 1)), ("2024-01", date(2024, 1, 1)),
                           ("1999-12", date(1999, 12, 1)), ("  2026-03 ", date(2026, 3, 1))):
        obtenu = _pas_none(_appel(lambda: lire(texte), f"lire_mois({texte!r})"), f"lire_mois({texte!r})")
        if isinstance(obtenu, datetime):
            raise AssertionError(f"lire_mois({texte!r}) renvoie un datetime ({obtenu!r}) : renvoyez une date "
                                 f"(`.date()`, ou `date(annee, mois, 1)`).")
        if obtenu != attendu:
            raise AssertionError(f"lire_mois({texte!r}) renvoie {obtenu!r} au lieu de {attendu!r}"
                                 + (" (les espaces autour sont tolérés : `.strip()`)." if texte != texte.strip() else "."))
    conseils = {
        "2026-1": "« 2026-1 » est accepté, alors qu'il faut DEUX chiffres pour le mois (rapport_2026-1.html se "
                  "rangerait après rapport_2026-10.html). `strptime(\"%Y-%m\")` l'accepte : vérifiez la forme avec "
                  "`re.fullmatch(r\"\\d{4}-\\d{2}\", texte)`.",
        "2026-10-01": "« 2026-10-01 » est accepté : `re.match` ne vérifie que le DÉBUT du texte. `re.fullmatch` "
                      "vérifie le texte entier.",
        "2026-13": "il n'y a pas de 13e mois.",
        "2026-00": "le mois va de 01 à 12 : « 00 » n'est pas un mois.",
        "10/2026": "« 10/2026 » n'est pas au format AAAA-MM.",
        "octobre": "« octobre » n'est pas au format AAAA-MM.",
        "": "un texte vide n'est pas un mois.",
        "2026-1O": "« 2026-1O » contient la lettre O, pas un zéro.",
    }
    for texte, conseil in conseils.items():
        _leve_value_error(lambda: lire(texte), f"lire_mois({texte!r})", conseil)

    for jour, attendu, conseil in (
        (date(2026, 10, 15), date(2026, 10, 31), "un jour au milieu du mois désigne tout le mois."),
        (date(2026, 4, 1), date(2026, 4, 30), "avril a 30 jours."),
        (date(2026, 2, 10), date(2026, 2, 28), "février 2026 a 28 jours."),
        (date(2028, 2, 1), date(2028, 2, 29), "2028 est bissextile : février a 29 jours. `calendar.monthrange` le sait."),
        (date(2100, 2, 1), date(2100, 2, 28), "2100 n'est PAS bissextile (divisible par 100 mais pas par 400). "
                                              "`calendar.monthrange` le sait."),
        (date(2026, 1, 31), date(2026, 1, 31), "le 31 janvier est déjà le dernier jour de janvier."),
        (date(2026, 12, 1), date(2026, 12, 31), "en décembre, le mois suivant est en JANVIER de l'année d'après."),
    ):
        try:
            obtenu = _pas_none(fin(jour), f"fin_du_mois({jour!r})")
        except (NotImplementedError, AssertionError):
            raise
        except ValueError as exc:
            raise AssertionError(f"fin_du_mois({jour!r}) lève ValueError ({exc}) : {conseil} Utilisez "
                                 f"`calendar.monthrange(annee, mois)[1]`, le nombre de jours du mois.") from None
        if obtenu != attendu or isinstance(obtenu, datetime):
            raise AssertionError(f"fin_du_mois({jour!r}) renvoie {obtenu!r} au lieu de {attendu!r} : {conseil}")

    for jour, n, attendu, conseil in (
        (date(2026, 10, 15), 0, date(2026, 10, 1), "avec n = 0, le 1er du MÊME mois (on renvoie toujours un 1er)."),
        (date(2026, 10, 1), 1, date(2026, 11, 1), ""),
        (date(2026, 10, 31), 1, date(2026, 11, 1), "le 31 octobre plus un mois donne le 1er novembre (il n'y a pas "
                                                   "de 31 novembre : renvoyez toujours le 1er)."),
        (date(2026, 1, 1), -1, date(2025, 12, 1), "le mois d'avant janvier est décembre de l'année PRÉCÉDENTE."),
        (date(2026, 12, 1), 1, date(2027, 1, 1), "le mois d'après décembre est janvier de l'année SUIVANTE."),
        (date(2026, 3, 1), 12, date(2027, 3, 1), "un mois ne fait pas 30 jours : 12 × 30 jours après le 1er mars "
                                                 "2026, on est le 24 février 2027. Comptez en mois, pas en jours."),
        (date(2026, 1, 15), -13, date(2024, 12, 1), "13 mois avant janvier 2026, c'est décembre 2024."),
        (date(2026, 10, 1), -24, date(2024, 10, 1), ""),
        (date(2026, 7, 1), -7, date(2025, 12, 1), "`//` et `%` de Python gèrent les nombres négatifs : "
                                                  "(2026 * 12 + 6 - 7) // 12 vaut 2025."),
    ):
        obtenu = _pas_none(_appel(lambda: decaler(jour, n), f"decaler({jour!r}, {n})"), f"decaler({jour!r}, {n})")
        if obtenu != attendu or isinstance(obtenu, datetime):
            if not conseil and getattr(obtenu, "day", None) != 1:
                conseil = "renvoyez toujours le 1er du mois."
            if getattr(obtenu, "day", None) == 1 and (obtenu.year, obtenu.month) != (attendu.year, attendu.month):
                conseil += (" Comptez en MOIS, pas en jours : `rang = annee * 12 + mois - 1 + n`, puis "
                            "`date(rang // 12, rang % 12 + 1, 1)`.")
            raise AssertionError(f"decaler({jour!r}, {n}) renvoie {obtenu!r} au lieu de {attendu!r}"
                                 + (f" : {conseil.strip()}" if conseil else "."))

    if _appelle(nom, "setlocale"):
        raise AssertionError("nom_du_mois appelle `locale.setlocale` : il change la langue de TOUT le programme, et "
                             "lève locale.Error sur un serveur où le français n'est pas installé. Utilisez MOIS_FR "
                             "(modeles.py).")
    for jour, attendu in ((date(2026, 10, 1), "octobre 2026"), (date(2026, 8, 15), "août 2026"),
                          (date(2026, 1, 1), "janvier 2026"), (date(2025, 12, 1), "décembre 2025"),
                          (date(2026, 2, 28), "février 2026")):
        try:
            obtenu = _pas_none(nom(jour), f"nom_du_mois({jour!r})")
        except (NotImplementedError, AssertionError):
            raise
        except IndexError:
            raise AssertionError(f"nom_du_mois({jour!r}) lève IndexError : janvier est le mois 1 mais l'élément 0 de "
                                 f"MOIS_FR, décembre le mois 12 mais l'élément 11.") from None
        if obtenu == attendu:
            continue
        if not isinstance(obtenu, str):
            raise AssertionError(f"nom_du_mois({jour!r}) doit renvoyer un texte, pas {type(obtenu).__name__}.")
        if obtenu == jour.strftime("%B %Y") or re.search(r"[A-Z][a-z]+ \d{4}", obtenu) and obtenu[0] != attendu[0].upper():
            raise AssertionError(f"nom_du_mois({jour!r}) renvoie {obtenu!r} : `strftime(\"%B\")` écrit le mois dans la "
                                 f"langue de l'ordinateur. Utilisez MOIS_FR (modeles.py).")
        if obtenu.lower() == attendu:
            raise AssertionError(f"nom_du_mois({jour!r}) renvoie {obtenu!r} au lieu de {attendu!r} : en français, les "
                                 f"noms des mois s'écrivent en minuscules.")
        if obtenu.split(" ")[0] in [m for m in _modeles(module).MOIS_FR]:
            raise AssertionError(f"nom_du_mois({jour!r}) renvoie {obtenu!r} au lieu de {attendu!r} : décalage d'un "
                                 f"mois ? Janvier est le mois 1 mais l'élément 0 de MOIS_FR.")
        raise AssertionError(f"nom_du_mois({jour!r}) renvoie {obtenu!r} au lieu de {attendu!r}.")


# ============================================================================ Étape 2 : les chiffres du mois


def _variante(*, denominateur_fin=False, fin_le_1er_absent=False, debut_le_1er_present=False,
              departs_sans_borne_basse=False, departs_sans_dernier_jour=False, taux_tous_departs=False,
              taux_zero=False, revenu_actifs_seulement=False, arrivees_sans_borne_basse=False,
              actifs_fin_sans_arrivees=False, decembre_plante=False, actifs_fin_dernier_jour=False,
              mois_tel_quel=False, classe=Indicateurs) -> Callable:
    """`indicateurs_du_mois`, juste par défaut ; chaque option y glisse UNE erreur classique."""
    def indicateurs_du_mois(abonnes, mois):
        premier = mois if mois_tel_quel else mois.replace(day=1)
        if decembre_plante:
            dernier = date(premier.year, premier.month + 1, 1) - timedelta(days=1)
        else:
            dernier = ref_fin_du_mois(premier)
        actifs_debut = arrivees = departs = actifs_fin = partis_actifs = 0
        revenu = 0.0
        for a in abonnes:
            if fin_le_1er_absent:
                actif = a.debut < premier and (a.fin is None or a.fin > premier)
            elif debut_le_1er_present:
                actif = a.debut <= premier and (a.fin is None or a.fin >= premier)
            else:
                actif = a.debut < premier and (a.fin is None or a.fin >= premier)
            if departs_sans_borne_basse:
                part = a.fin is not None and a.fin <= dernier
            elif departs_sans_dernier_jour:
                part = a.fin is not None and premier <= a.fin < dernier
            else:
                part = a.fin is not None and premier <= a.fin <= dernier
            actifs_debut += actif
            arrivees += (a.debut <= dernier) if arrivees_sans_borne_basse else (premier <= a.debut <= dernier)
            if actifs_fin_dernier_jour:
                actifs_fin += a.debut <= dernier and (a.fin is None or a.fin >= dernier)
            else:
                actifs_fin += a.debut <= dernier and (a.fin is None or a.fin > dernier)
            if part:
                departs += 1
                partis_actifs += actif
                if actif or not revenu_actifs_seulement:
                    revenu += a.mensualite
        if actifs_fin_sans_arrivees:
            actifs_fin = actifs_debut - departs
        base = actifs_fin if denominateur_fin else actifs_debut
        numerateur = departs if taux_tous_departs else partis_actifs
        taux = numerateur / base if base else (0.0 if taux_zero else None)
        return classe(premier, actifs_debut, arrivees, departs, actifs_fin, taux, revenu)
    return indicateurs_du_mois


#: (options, ce que fait la version boguée (étape 7), le conseil quand c'est la vôtre (étape 2))
_MUTANTS_INDICATEURS = (
    (dict(denominateur_fin=True), "calcule le taux sur les abonnés de FIN de mois",
     "Le taux se calcule sur les abonnés du DÉBUT du mois (actifs_debut), pas sur ceux de la fin : le "
     "dénominateur ne doit pas dépendre de ce qui se passe pendant le mois."),
    (dict(fin_le_1er_absent=True), "oublie dans actifs_debut un client parti LE 1er du mois",
     "Un client parti le 1er était encore abonné le 1er (`fin` est le DERNIER jour d'abonnement, inclus) : il fait "
     "partie des actifs_debut, et c'est un départ du mois. Comparez avec `fin >= premier`, pas `>`."),
    (dict(debut_le_1er_present=True), "compte dans actifs_debut un client arrivé LE 1er du mois",
     "Un client arrivé le 1er n'était pas là « au début » du mois : c'est une arrivée. `debut < premier`, pas `<=`."),
    (dict(departs_sans_borne_basse=True), "compte comme départs du mois des clients partis AVANT le mois",
     "Un départ du mois a sa `fin` ENTRE le 1er et le dernier jour : vous avez oublié la borne basse "
     "(`premier <= fin <= dernier`), et un client parti en septembre compte dans les départs d'octobre."),
    (dict(departs_sans_dernier_jour=True), "oublie les départs du DERNIER jour du mois",
     "Le dernier jour fait partie du mois : `fin <= dernier`, pas `<`. Un client parti le 31 octobre est un "
     "départ d'octobre."),
    (dict(taux_tous_departs=True), "met au numérateur du taux les clients arrivés ET partis pendant le mois",
     "Le taux ne compte que les départs des abonnés du DÉBUT du mois : un client arrivé le 3 et parti le 20 "
     "n'est pas au dénominateur, il ne peut pas être au numérateur (sinon le taux peut dépasser 100 %)."),
    (dict(taux_zero=True), "renvoie un taux de 0.0 quand personne n'était abonné au début du mois",
     "Sans abonné au début du mois, le taux n'existe pas : renvoyez None. 0.0 voudrait dire « personne n'est "
     "parti », et le rapport afficherait « 0,0 % » au lieu de « — »."),
    (dict(revenu_actifs_seulement=True), "ne compte dans le revenu perdu que les départs des abonnés du début",
     "Le revenu perdu compte TOUS les départs du mois, y compris un client arrivé puis parti pendant le mois : "
     "sa mensualité est bel et bien perdue."),
    (dict(arrivees_sans_borne_basse=True), "compte comme arrivées des clients arrivés AVANT le mois",
     "Une arrivée du mois a son `debut` ENTRE le 1er et le dernier jour : `premier <= debut <= dernier`."),
    (dict(actifs_fin_sans_arrivees=True), "calcule actifs_fin sans compter les arrivées du mois",
     "actifs_fin = actifs_debut + arrivees - departs : vous avez oublié les arrivées. Le plus sûr est de compter "
     "directement ceux qui sont encore là après le dernier jour, puis de vérifier l'égalité."),
    (dict(decembre_plante=True), "plante en décembre",
     "En décembre, `date(annee, mois + 1, 1)` demande un mois 13 et lève ValueError. Utilisez `fin_du_mois` "
     "(étape 1)."),
    (dict(actifs_fin_dernier_jour=True), "compte dans actifs_fin un client parti LE dernier jour du mois",
     "Un client parti le dernier jour n'est plus là à la fin du mois : `fin > dernier`, pas `>=`."),
    (dict(mois_tel_quel=True), "ne ramène pas le mois reçu au 1er (avec le 15 octobre, elle ne compte qu'une "
                               "moitié d'octobre)",
     "`mois` peut être n'importe quel jour du mois : partez du 1er (`decaler(mois, 0)`), pour les calculs ET "
     "pour `Indicateurs.mois`."),
)


def _scenarios(M: ModuleType) -> list[tuple[str, list, date]]:
    """(description, abonnés, mois) : chaque règle, et chaque version boguée, s'y voit au moins une fois."""
    A = _fabrique(M)
    frontieres = [
        A("A", "2026-01-10"), A("B", "2026-02-01", "2026-10-01", 30.0), A("C", "2026-03-05", "2026-10-31", 40.0),
        A("D", "2026-10-01"), A("E", "2026-10-03", "2026-10-20", 25.0), A("F", "2025-06-01", "2026-09-30", 99.0),
        A("G", "2026-11-01"), A("H", "2026-04-01", "2026-11-01"), A("I", "2026-09-30", "2026-10-15", 12.5),
    ]
    return [
        ("octobre 2026, avec des abonnés sur toutes les frontières", frontieres, date(2026, 10, 1)),
        ("le même octobre, demandé avec le 15 octobre", frontieres, date(2026, 10, 15)),
        ("un mois où personne n'était abonné au début", [A("N", "2026-10-05"), A("O", "2026-10-06", "2026-10-09")],
         date(2026, 10, 1)),
        ("décembre 2026", [A("X", "2026-01-01", "2026-12-31", 20.0), A("Y", "2026-12-31"),
                           A("Z", "2025-05-05", "2027-01-01")], date(2026, 12, 1)),
        ("février 2028 (bissextile)", [A("P", "2027-05-01", "2028-02-29"), A("Q", "2028-02-29"),
                                       A("R", "2027-01-01", "2028-02-28")], date(2028, 2, 1)),
        ("une liste vide", [], date(2026, 10, 1)),
    ]


def _signature(fonction, scenarios) -> list:
    """Ce que renvoie (ou lève) `fonction` sur chaque scénario."""
    resultats = []
    for _, abonnes, mois in scenarios:
        try:
            resultats.append(_valeurs(fonction(list(abonnes), mois), "indicateurs_du_mois"))
        except (NotImplementedError, AssertionError):
            raise
        except Exception as exc:  # noqa: BLE001
            resultats.append(type(exc))
    return resultats


def _meme_signature(a: list, b: list) -> bool:
    return len(a) == len(b) and all(x is y if isinstance(x, type) or isinstance(y, type) else _memes_valeurs(x, y)
                                    for x, y in zip(a, b))


def verifier_indicateurs(module: ModuleType) -> None:
    M = _modeles(module)
    fonction = module.indicateurs_du_mois
    scenarios = _scenarios(M)

    for description, abonnes, mois in scenarios:
        copie = list(abonnes)
        try:
            obtenu = _valeurs(fonction(abonnes, mois), f"indicateurs_du_mois({description})")
        except (NotImplementedError, AssertionError):
            raise
        except Exception:  # noqa: BLE001 — on cherche d'abord si c'est une erreur classique
            obtenu = None
        if abonnes != copie:
            raise AssertionError(f"indicateurs_du_mois a MODIFIÉ la liste reçue ({description}) : ne la triez pas, ne "
                                 f"la videz pas. Elle sert ensuite aux autres mois et aux segments.")
        attendu = _valeurs(ref_indicateurs_du_mois(abonnes, mois), "le corrigé")
        if obtenu is not None and _memes_valeurs(obtenu, attendu):
            continue
        votre = _signature(fonction, scenarios)
        for options, _, conseil in _MUTANTS_INDICATEURS:
            if _meme_signature(votre, _signature(_variante(**options), scenarios)):
                raise AssertionError(conseil)
        if obtenu is None:
            fonction(list(abonnes), mois)  # on laisse remonter l'erreur : le correcteur montre la ligne
        ecarts = [f"{c} vaut {o!r} au lieu de {a!r}" for c, o, a in zip(_CHAMPS, obtenu, attendu) if not _egaux(o, a)]
        raise AssertionError(f"Pour {description} ({mois:%d/%m/%Y}) : {' ; '.join(ecarts)}. Les abonnés : "
                             f"{_montrer_abonnes(abonnes) or 'aucun'}. Calculez chaque indicateur à la main avec la "
                             f"règle de la docstring.")

    exemple = M.lire_abonnes(CHEMIN_EXEMPLE)
    for mois in (date(2026, 10, 1), date(2024, 1, 1), date(2025, 12, 1)):
        obtenu = _valeurs(fonction(exemple, mois), "indicateurs_du_mois")
        attendu = _valeurs(ref_indicateurs_du_mois(exemple, mois), "le corrigé")
        if not _memes_valeurs(obtenu, attendu):
            ecarts = [f"{c} vaut {o!r} au lieu de {a!r}" for c, o, a in zip(_CHAMPS, obtenu, attendu) if not _egaux(o, a)]
            raise AssertionError(f"Sur projets/exemple_abonnes.csv, pour {mois:%m/%Y} : {' ; '.join(ecarts)}.")

    _verifier_par_segment(module, M)


def _verifier_par_segment(module: ModuleType, M: ModuleType) -> None:
    A = _fabrique(M)
    abonnes = [
        A("S1", "2025-01-01", "2026-10-10", segment="Senior"), A("P1", "2026-02-01", segment="Pro & TPE"),
        A("S2", "2024-03-01", segment="Senior"), A("A1", "2020-01-01", "2023-01-01", segment="Ancien"),
        A("P2", "2026-01-15", "2026-10-31", 99.0, segment="Pro & TPE"), A("X1", "2026-10-05", segment="Particulier"),
        A("X2", "2025-10-05", "2026-10-01", 20.0, segment="Particulier"),
    ]
    octobre = date(2026, 10, 1)
    attendu = ref_par_segment(abonnes, octobre)
    obtenu = _pas_none(module.par_segment(list(abonnes), octobre), "par_segment")
    if not isinstance(obtenu, dict):
        raise AssertionError(f"par_segment doit renvoyer un dict {{segment: Indicateurs}}, pas {type(obtenu).__name__}.")
    if set(obtenu) != set(attendu):
        manque = sorted(set(attendu) - set(obtenu))
        raise AssertionError(f"par_segment renvoie les segments {list(obtenu)} au lieu de {list(attendu)}"
                             + (f" : il manque {manque}. Un segment présent dans les abonnés figure dans le résultat, "
                                f"même si aucun de ses abonnés n'est actif ce mois-là (« Ancien »)." if manque else "."))
    if list(obtenu) != list(attendu):
        raise AssertionError(f"par_segment renvoie les segments dans l'ordre {list(obtenu)} au lieu de {list(attendu)} "
                             f"(l'ordre de `sorted`) : le tableau du rapport doit toujours se lire dans le même ordre.")
    for segment in attendu:
        o, a = _valeurs(obtenu[segment], f"par_segment[{segment!r}]"), _valeurs(attendu[segment], "le corrigé")
        if not _memes_valeurs(o, a):
            raise AssertionError(f"par_segment : les chiffres du segment {segment!r} ne sont pas ceux "
                                 f"qu'indicateurs_du_mois donne pour ses seuls abonnés ({o} au lieu de {a}). "
                                 f"Réutilisez indicateurs_du_mois sur les abonnés de chaque segment.")
    depuis_generateur = _pas_none(module.par_segment((a for a in abonnes), octobre), "par_segment(générateur)")
    if list(depuis_generateur) != list(attendu) or any(
            not _memes_valeurs(_valeurs(depuis_generateur[s], "par_segment"), _valeurs(attendu[s], "le corrigé"))
            for s in attendu):
        raise AssertionError("par_segment donne un autre résultat quand `abonnes` est un générateur : un générateur "
                             "est vide au deuxième passage. Regroupez d'abord les abonnés par segment, en UN passage.")


# ============================================================================ Étape 3 : écrire les nombres


def _visible(texte) -> str:
    """repr() montre les espaces spéciales : '1\\u202f234,50\\xa0€'."""
    return repr(texte)


def _diagnostic_format(obtenu: str, attendu: str) -> str:
    if not isinstance(obtenu, str):
        return f"renvoyez un texte (str), pas {type(obtenu).__name__}."
    if "None" in obtenu or "nan" in obtenu:
        return f"une valeur inconnue (None) s'écrit TIRET ({TIRET!r})."
    if obtenu.replace(" ", ESPACE_MILLIERS) == attendu:
        return ("les milliers se séparent par ESPACE_MILLIERS (« \\u202f », une espace fine insécable), pas par une "
                "espace ordinaire.")
    if obtenu.replace(" ", ESPACE_UNITE) == attendu or obtenu.replace(ESPACE_MILLIERS, ESPACE_UNITE) == attendu:
        return "avant l'unité (€, %, pt, ou le mot), ESPACE_UNITE (« \\xa0 », une espace insécable)."
    if obtenu.replace(" ", "") == attendu.replace(ESPACE_MILLIERS, "").replace(ESPACE_UNITE, ""):
        return "il manque les espaces : ESPACE_MILLIERS entre les milliers, ESPACE_UNITE avant l'unité."
    if "." in obtenu and "," not in obtenu:
        return "en français, la virgule sépare les décimales : « 12,5 », pas « 12.5 »."
    if obtenu.startswith(("-0,", "+0,")) and attendu.lstrip("+-") == obtenu.lstrip("+-") and attendu[0] not in "+-":
        return ("un résultat arrondi à zéro s'écrit sans signe : personne ne lit « -0,0 ». Testez le signe sur la "
                "valeur ARRONDIE (`f\"{x:+.1f}\"` écrit « +0.0 » et « -0.0 »).")
    return ""


def _comparer_format(fonction, nom: str, args: tuple, attendu: str, conseil: str = "") -> None:
    quoi = f"{nom}({', '.join(repr(a) for a in args)})"
    obtenu = _pas_none(_appel(lambda: fonction(*args), quoi), quoi)
    if obtenu == attendu:
        return
    pourquoi = _diagnostic_format(obtenu, attendu) or conseil
    raise AssertionError(f"{quoi} renvoie {_visible(obtenu)} au lieu de {_visible(attendu)}"
                         + (f" : {pourquoi}" if pourquoi else "."))


def verifier_formats(module: ModuleType) -> None:
    arrondi = ("arrondissez à la moitié SUPÉRIEURE le nombre tel qu'il s'écrit : "
               "`Decimal(str(x)).quantize(Decimal(\"0.01\"), rounding=ROUND_HALF_UP)`. `round()` arrondit au pair "
               "(0,125 → 0,12), et `Decimal(x)` ou `f\"{x:.2f}\"` voient la valeur binaire du float (2,675 → 2,67).")
    for montant, conseil in ((1234.5, ""), (0, ""), (12.0, ""), (7821.59, ""), (0.125, arrondi), (2.675, arrondi),
                             (1.005, arrondi), (999.995, "arrondissez AVANT de séparer les milliers : 999,995 € fait "
                                                         "1 000,00 €."),
                             (1234567.891, ""), (-1234.5, "un montant négatif garde son signe « - »."),
                             (-0.001, ""), (None, "")):
        _comparer_format(module.euros, "euros", (montant,), ref_formats.euros(montant), conseil)

    pour_cent = ("multipliez par 100 en Decimal (`Decimal(str(part)) * 100`) : en float, 0.0295 * 100 vaut "
                 "2.9499999999999997, et l'on écrirait « 2,9 % » au lieu de « 3,0 % ».")
    for args, conseil in (((0.1234,), ""), ((0.03,), ""), ((1,), ""), ((0,), ""), ((0.0295,), pour_cent),
                          ((0.0185,), pour_cent), ((0.12345, 2), pour_cent), ((0.5, 0), "avec 0 décimale, pas de "
                                                                                       "virgule : « 50 % »."),
                          ((0.0004, 0), ""), ((-0.0001,), ""), ((12.3456,), "les milliers se séparent aussi dans un "
                                                                           "pourcentage."),
                          ((None,), "")):
        _comparer_format(module.pourcentage, "pourcentage", args, ref_formats.pourcentage(*args), conseil)

    en_decimal = ("calculez l'écart en Decimal à partir des taux exacts : en float, 0.1135 - 0.1 vaut "
                  "0.013499999999999998, et l'on écrirait +1,3 pt au lieu de +1,4 pt.")
    for args, conseil in (((0.02, 0.03), "c'est un écart en POINTS : (3 % - 2 %) × 100 = +1,0 pt. Pas la variation "
                                         "relative (+50 %)."),
                          ((0.020, 0.032), ""), ((0.032, 0.020), "un écart négatif s'écrit avec « - »."),
                          ((0.1, 0.1135), en_decimal), ((0.0185, 0.0), en_decimal),
                          ((0.05, 0.0501), "un écart nul APRÈS arrondi s'écrit sans signe : « 0,0 pt »."),
                          ((0.05, 0.0499), ""), ((0.0, 0.0), ""), ((0.2, 0.3), ""),
                          ((None, 0.03), "si l'un des taux est None, il n'y a rien à comparer : TIRET."),
                          ((0.03, None), "si l'un des taux est None, il n'y a rien à comparer : TIRET.")):
        attendu = ref_formats.ecart_points(*args)
        obtenu = _pas_none(_appel(lambda: module.ecart_points(*args), f"ecart_points{args}"), f"ecart_points{args}")
        if isinstance(obtenu, str) and obtenu.endswith("pts"):
            raise AssertionError(f"ecart_points{args} renvoie {_visible(obtenu)} : l'abréviation « pt » ne prend pas de "
                                 f"« s ».")
        _comparer_format(module.ecart_points, "ecart_points", args, attendu, conseil)

    for n, mot, conseil in ((0, "client", "en français, 0 est au SINGULIER : « 0 client »."), (1, "client", ""),
                            (2, "client", "le pluriel commence à 2."), (89, "départ", ""),
                            (1234, "client", ""), (1000000, "client", ""),
                            (3, "mois", "un mot qui finit déjà par s, x ou z ne change pas : « 3 mois »."),
                            (2, "prix", "un mot qui finit déjà par s, x ou z ne change pas : « 2 prix »."),
                            (2, "nez", "un mot qui finit déjà par s, x ou z ne change pas : « 2 nez »."),
                            (1, "départ", "")):
        _comparer_format(module.nombre, "nombre", (n, mot), ref_formats.nombre(n, mot), conseil)


# ============================================================================ Étape 4 : la courbe en SVG


def verifier_echelle(module: ModuleType) -> None:
    for maximum in (0.03, 0.05, 0.0201, 0.024, 0.026, 0.011, 0.12, 0.3, 0.25, 0.7, 1.0, 7, 250, 0.000123, 5e-06,
                    0.0, -1.0):
        attendu = ref_graphique.echelle_haute(maximum)
        quoi = f"echelle_haute({maximum!r})"
        try:
            obtenu = _pas_none(module.echelle_haute(maximum), quoi)
        except (NotImplementedError, AssertionError):
            raise
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise AssertionError(f"{quoi} lève {type(exc).__name__} ({exc}) : si le maximum n'est pas strictement "
                                 f"positif, renvoyez ECHELLE_PAR_DEFAUT (`math.log10(0)` n'existe pas).") from None
        try:
            juste = math.isclose(float(obtenu), attendu, rel_tol=1e-9)
        except (TypeError, ValueError):
            juste = False
        if juste:
            continue
        if maximum in (0.05, 0.25, 1.0, 5e-06) and isinstance(obtenu, (int, float)) and obtenu > attendu:
            raise AssertionError(f"{quoi} renvoie {obtenu!r} au lieu de {attendu!r} : {maximum!r} est déjà un nombre rond, "
                                 f"on le garde (`>=`, pas `>`)" + (". Et fabriquez vos candidats sans erreur d'arrondi : "
                                 "5 * 10 ** -6 vaut 4.9999999999999996e-06, plus petit que 5e-06 ; "
                                 "float(Decimal(5).scaleb(-6)) vaut exactement 5e-06." if maximum == 5e-06 else "."))
        if maximum in (0.0201, 0.024) and math.isclose(float(obtenu), 0.05):
            raise AssertionError(f"{quoi} renvoie {obtenu!r} au lieu de {attendu!r} : 2,5 fait partie des nombres ronds "
                                 f"(1, 2, 2,5 et 5).")
        raise AssertionError(f"{quoi} renvoie {obtenu!r} au lieu de {attendu!r} : le plus petit nombre de la forme 1, 2, "
                             f"2,5 ou 5 × 10^k qui est supérieur ou égal au maximum.")


def _lire_svg(svg, quoi: str) -> ET.Element:
    svg = _pas_none(svg, quoi)
    if not isinstance(svg, str):
        raise AssertionError(f"{quoi} doit renvoyer le SVG sous forme de texte (str), pas {type(svg).__name__}.")
    try:
        racine = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise AssertionError(f"{quoi} n'est pas du XML bien formé ({exc}). Un « & » ou un « < » non échappé dans le "
                             f"titre ou une étiquette ? Passez tout texte venu de l'extérieur dans `html.escape`. "
                             f"Une balise pas fermée (`<line ... />`) ?") from None
    if racine.tag == "svg":
        raise AssertionError(f"{quoi} : la balise <svg> doit déclarer `xmlns=\"{SVG_NS}\"`. Sans elle, le navigateur "
                             f"ne sait pas que c'est un dessin.")
    if racine.tag != f"{{{SVG_NS}}}svg":
        raise AssertionError(f"{quoi} doit commencer par la balise <svg ...>, pas <{racine.tag}>.")
    return racine


def _nombres(texte: str | None) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", texte or "")]


def _points(polyline: ET.Element) -> list[tuple[float, float]]:
    valeurs = _nombres(polyline.get("points"))
    return list(zip(valeurs[0::2], valeurs[1::2]))


def _proches(a: tuple, b: tuple, tol: float = 0.11) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def _texte_xml(element: ET.Element) -> str:
    return "".join(element.itertext())


def _attendu_courbe(valeurs, largeur: int, hauteur: int):
    haut = ref_graphique.echelle_haute(max((v for v in valeurs if v is not None), default=0.0))
    gauche, droite, sommet, bas = MARGE_GAUCHE, largeur - MARGE_DROITE, MARGE_HAUT, hauteur - MARGE_BAS
    n = len(valeurs)
    x = [gauche + (droite - gauche) * (i / (n - 1) if n > 1 else 0.5) for i in range(n)]
    y = [None if v is None else bas - (bas - sommet) * v / haut for v in valeurs]
    suites, suite = [], []
    for i, v in enumerate([*valeurs, None]):
        if v is not None:
            suite.append((x[i], y[i]))
        else:
            if len(suite) >= 2:
                suites.append(suite)
            suite = []
    return haut, x, y, suites


def _verifier_un_dessin(module: ModuleType, etiquettes, valeurs, titre: str, largeur: int = LARGEUR,
                        hauteur: int = HAUTEUR, quoi: str = "courbe_svg(...)") -> None:
    pourcentage = _voisin(module, "formats").pourcentage
    if (largeur, hauteur) == (LARGEUR, HAUTEUR):
        svg = module.courbe_svg(list(etiquettes), list(valeurs), titre)
    else:
        svg = module.courbe_svg(list(etiquettes), list(valeurs), titre, largeur, hauteur)
    racine = _lire_svg(svg, quoi)
    ns = f"{{{SVG_NS}}}"
    vue = re.split(r"[\s,]+", (racine.get("viewBox") or "").strip())
    if vue != ["0", "0", str(largeur), str(hauteur)]:
        raise AssertionError(f"{quoi} : viewBox vaut {racine.get('viewBox')!r} au lieu de \"0 0 {largeur} {hauteur}\""
                             + (" (utilisez les paramètres `largeur` et `hauteur`, pas les constantes)."
                                if (largeur, hauteur) != (LARGEUR, HAUTEUR) else "."))
    enfants = list(racine)
    if not enfants or enfants[0].tag != f"{ns}title":
        raise AssertionError(f"{quoi} : le premier enfant de <svg> doit être <title>…</title> (ce que lit un lecteur "
                             f"d'écran).")
    if _texte_xml(enfants[0]) != titre:
        lu = _texte_xml(enfants[0])
        raise AssertionError(f"{quoi} : le <title> se lit {lu!r} au lieu de {titre!r}"
                             + (" : vous avez échappé deux fois (« &amp;amp; »)." if "&amp;" in lu or "&lt;" in lu else "."))

    haut, xs, ys, suites = _attendu_courbe(valeurs, largeur, hauteur)
    cercles = racine.findall(f".//{ns}circle")
    connus = [i for i, v in enumerate(valeurs) if v is not None]
    if len(cercles) != len(connus):
        raise AssertionError(f"{quoi} : {len(cercles)} <circle> au lieu de {len(connus)}, un par valeur CONNUE (pas de "
                             f"point pour un mois inconnu, None).")
    obtenus = sorted(((_nombres(c.get("cx"))[:1] + _nombres(c.get("cy"))[:1], c) for c in cercles),
                     key=lambda paire: paire[0])
    if any(len(xy) != 2 for xy, _ in obtenus):
        raise AssertionError(f"{quoi} : chaque <circle> a besoin de `cx` et `cy`, en pixels (des nombres).")
    for (xy, cercle), i in zip(obtenus, connus):
        attendu = (xs[i], ys[i])
        if _proches(tuple(xy), attendu):
            titre_point = cercle.find(f"{ns}title")
            texte = f"{etiquettes[i]} : {pourcentage(valeurs[i])}"
            if titre_point is None or _texte_xml(titre_point) != texte:
                lu = None if titre_point is None else _texte_xml(titre_point)
                raise AssertionError(f"{quoi} : le point de {etiquettes[i]!r} doit contenir <title>{texte}</title> "
                                     f"(l'infobulle), pas {lu!r}.")
            continue
        bas = hauteur - MARGE_BAS
        if abs(xy[0] - xs[i]) <= 0.11 and abs(xy[1] - (MARGE_HAUT + (bas - ys[i]))) <= 0.11:
            raise AssertionError(f"{quoi} : le point de {etiquettes[i]!r} est à y = {xy[1]} au lieu de {ys[i]:.1f}. En SVG, "
                                 f"y grandit vers le BAS : une valeur plus grande doit avoir un y plus PETIT "
                                 f"(y = (hauteur - MARGE_BAS) - v / haut × (hauteur - MARGE_BAS - MARGE_HAUT)).")
        if len(valeurs) > 1 and abs(xy[0] - (MARGE_GAUCHE + (largeur - MARGE_GAUCHE - MARGE_DROITE) * i / len(valeurs))) <= 0.11:
            raise AssertionError(f"{quoi} : le point de {etiquettes[i]!r} est à x = {xy[0]} au lieu de {xs[i]:.1f} : divisez "
                                 f"la largeur par n - 1 intervalles (le dernier point touche le bord droit), pas par n.")
        raise AssertionError(f"{quoi} : le point de {etiquettes[i]!r} (valeur {valeurs[i]!r}) est en {tuple(xy)} au lieu "
                             f"de ({xs[i]:.1f}, {ys[i]:.1f}). L'échelle monte jusqu'à echelle_haute({max(valeurs[j] for j in connus)!r}) "
                             f"= {haut!r}.")

    lignes = [_points(p) for p in racine.findall(f".//{ns}polyline")]
    if len(lignes) != len(suites) or not all(len(a) == len(b) and all(_proches(p, q) for p, q in zip(a, b))
                                             for a, b in zip(sorted(lignes), sorted(suites))):
        tous = [(xs[i], ys[i]) for i in connus]
        if len(lignes) == 1 and len(lignes[0]) == len(valeurs) and len(connus) < len(valeurs):
            raise AssertionError(f"{quoi} : une seule <polyline> passe par les {len(valeurs)} mois, y compris les mois "
                                 f"inconnus (None) : un mois inconnu COUPE la courbe, il ne la fait pas tomber à zéro.")
        if len(lignes) == 1 and len(lignes[0]) == len(tous) and len(suites) != 1:
            raise AssertionError(f"{quoi} : une seule <polyline> relie tous les points connus, par-dessus les mois "
                                 f"inconnus : un mois inconnu (None) doit COUPER la courbe. Attendu : {len(suites)} "
                                 f"polyline(s).")
        if any(len(ligne) == 1 for ligne in lignes):
            raise AssertionError(f"{quoi} : une <polyline> d'un seul point ne dessine rien. Une suite d'un seul mois "
                                 f"connu n'a pas de polyline : son cercle suffit.")
        raise AssertionError(f"{quoi} : {len(lignes)} <polyline> de {[len(ligne) for ligne in lignes]} points au lieu de "
                             f"{len(suites)} de {[len(s) for s in suites]} points (une par suite d'au moins deux mois "
                             f"connus consécutifs, avec les mêmes coordonnées que les cercles).")

    textes = [_texte_xml(t) for t in racine.iter(f"{ns}text")]
    graduations = [pourcentage(v) for v in (0.0, haut / 2, haut)]
    for attendu in [*etiquettes, *graduations]:
        if attendu not in textes:
            raise AssertionError(f"{quoi} : aucun <text> ne contient {attendu!r}. Il faut une étiquette par mois et les "
                                 f"graduations {graduations}"
                                 + (" (avec html.escape : sinon « & » casse le dessin)." if "&" in attendu else "."))
    if len(racine.findall(f".//{ns}line")) < 3:
        raise AssertionError(f"{quoi} : il faut au moins trois <line> horizontales, pour 0, haut / 2 et haut.")
    if re.search(r"\d\.\d{3,}", svg):
        trop = re.search(r"\d+\.\d{3,}", svg).group(0)
        raise AssertionError(f"{quoi} : des coordonnées comme {trop} alourdissent le fichier pour rien : une décimale "
                             f"suffit (`f\"{{x:.1f}}\"`).")


def verifier_graphique(module: ModuleType) -> None:
    verifier_echelle(module)
    etiquettes = ["nov. 25", "déc. 25", "janv. 26", "févr. 26", "mars 26", "avr. 26", "mai 26", "juin 26",
                  "juil. 26", "août 26", "sept. 26", "R&D <oct.>"]
    valeurs = [0.016, 0.02, 0.019, None, 0.021, None, 0.022, 0.017, 0.016, 0.018, 0.02, 0.03]
    titre = "Taux de churn — Pro & TPE <test>"
    _verifier_un_dessin(module, etiquettes, valeurs, titre, quoi="courbe_svg (12 mois, dont deux inconnus)")
    _verifier_un_dessin(module, etiquettes[:5], [0.2, 0.25, 0.1, 0.45, 0.3], "Cinq mois", 400, 200,
                        quoi="courbe_svg(..., largeur=400, hauteur=200)")
    try:
        _verifier_un_dessin(module, ["oct. 26"], [0.03], "Un seul mois", quoi="courbe_svg (un seul mois)")
        _verifier_un_dessin(module, ["a", "b", "c"], [None, None, None], "Rien de connu",
                            quoi="courbe_svg (aucune valeur connue)")
        _verifier_un_dessin(module, ["a", "b"], [0.0, 0.0], "Que des zéros", quoi="courbe_svg (que des zéros)")
    except ZeroDivisionError as exc:
        raise AssertionError(f"courbe_svg lève ZeroDivisionError ({exc}) : avec un seul mois, n - 1 vaut 0 (placez le "
                             f"point au milieu) ; avec que des zéros ou que des None, l'échelle vient de "
                             f"echelle_haute (étape 4), qui ne renvoie jamais 0.") from None
    except ValueError as exc:
        if "max()" in str(exc) or "empty" in str(exc):
            raise AssertionError(f"courbe_svg lève ValueError ({exc}) : sans aucune valeur connue, `max` d'une liste vide "
                                 f"plante. `max(..., default=0.0)`.") from None
        raise
    _leve_value_error(lambda: module.courbe_svg(["a", "b"], [0.1], "t"), "courbe_svg(['a', 'b'], [0.1], ...)",
                      "Deux étiquettes pour une valeur : une erreur dans l'appel, à signaler tout de suite.")


# ============================================================================ Étape 5 : la page HTML

_VIDES = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_EXTERNE = re.compile(r"^\s*(https?:|//)", re.IGNORECASE)
_URL_EXTERNE = re.compile(r"url\(\s*['\"]?\s*(https?:|//)|@import", re.IGNORECASE)


def _propre(texte: str) -> str:
    """Comme un navigateur : les blancs ASCII consécutifs comptent pour une espace (les insécables restent)."""
    return re.sub(r"[ \t\r\n]+", " ", texte).strip(" ")


class _Lecteur(HTMLParser):
    """Lit la page comme un navigateur : balises, textes, indicateurs, tableaux."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.doctype: str | None = None
        self.balises: list[tuple[str, dict]] = []
        self.pile: list[list] = []
        self.indicateurs: dict[str, str] = {}
        self.tableaux: dict[str, list[list[list[str]]]] = {}
        self.titre: list[str] = []
        self.h1: list[str] = []
        self.style: list[str] = []
        self.texte: list[str] = []
        self._tableau: str | None = None
        self._cellule: list[str] | None = None

    def handle_decl(self, decl):
        self.doctype = self.doctype or decl

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        self.balises.append((tag, a))
        if tag == "table" and "data-tableau" in a:
            self._tableau = a["data-tableau"]
            self.tableaux.setdefault(self._tableau, [])
        elif self._tableau is not None and tag == "tr":
            self.tableaux[self._tableau].append([])
        elif self._tableau is not None and tag == "td" and self.tableaux[self._tableau]:
            self._cellule = []
            self.tableaux[self._tableau][-1].append(self._cellule)
        if tag not in _VIDES:
            self.pile.append([tag, a, [] if "data-indicateur" in a else None])

    def handle_startendtag(self, tag, attrs):
        self.balises.append((tag, {k: (v or "") for k, v in attrs}))

    def handle_endtag(self, tag):
        if tag == "td":
            self._cellule = None
        if tag == "table":
            self._tableau = None
        for i in range(len(self.pile) - 1, -1, -1):
            if self.pile[i][0] == tag:
                for _, a, morceaux in self.pile[i:]:
                    if morceaux is not None:
                        self.indicateurs.setdefault(a["data-indicateur"], _propre("".join(morceaux)))
                del self.pile[i:]
                break

    def handle_data(self, data):
        ouvertes = [t for t, _, _ in self.pile]
        self.texte.append(data)
        for _, _, morceaux in self.pile:
            if morceaux is not None:
                morceaux.append(data)
        if self._cellule is not None:
            self._cellule.append(data)
        if ouvertes and ouvertes[-1] == "title" and "svg" not in ouvertes:
            self.titre.append(data)
        if "h1" in ouvertes:
            self.h1.append(data)
        if ouvertes and ouvertes[-1] == "style":
            self.style.append(data)

    def lignes(self, tableau: str) -> list[list[str]]:
        return [[_propre("".join(c)) for c in ligne] for ligne in self.tableaux.get(tableau, []) if ligne]


def _lire_page(page, quoi: str) -> _Lecteur:
    page = _pas_none(page, quoi)
    if not isinstance(page, str):
        raise AssertionError(f"{quoi} doit renvoyer la page sous forme de texte (str), pas {type(page).__name__}.")
    lecteur = _Lecteur()
    lecteur.feed(page)
    lecteur.close()
    return lecteur


def _rapport_de_test(M: ModuleType, avec_departs: bool = True, precedent_vide: bool = False):
    I, A = M.Indicateurs, _fabrique(M)
    mois = date(2026, 10, 1)
    taux = [0.016, 0.0195, None, 0.0171, 0.02, 0.0176, 0.0213, 0.0228, 0.022, 0.017, 0.0184, 0.0313]
    historique = tuple(I(ref_decaler(mois, k - 11), 0 if t is None else 2500 + 10 * k, 80, 0 if t is None else 40 + k,
                         2540 + 10 * k, t, 1000.0 + k) for k, t in enumerate(taux))
    actuel = I(mois, 2844, 83, 89, 2838, 0.0313, 7821.59)
    precedent = I(date(2026, 9, 1), 0, 0, 0, 0, None, 0.0) if precedent_vide else \
        I(date(2026, 9, 1), 2826, 70, 52, 2844, 0.0184, 2287.55)
    segments = {"Particulier": I(mois, 1623, 40, 36, 1627, 0.0222, 1100.0),
                "Pro & TPE": I(mois, 742, 20, 48, 714, 0.0647, 6500.5),
                "Seniors <i>VIP</i>": I(mois, 0, 3, 0, 3, None, 0.0)}
    departs = (
        A("C00909", "2021-06-29", "2026-10-07", 249.0, "Pro & TPE", nom="Durand & Fils"),
        A("C02801", "2025-02-03", "2026-10-14", 249.0, "Pro & TPE", nom='<script>alert("x")</script>'),
        A("C00938", "2021-07-28", "2026-10-30", 179.0, "Pro & TPE", nom='Café "Chez Zoé"'),
        A("C01000", "2024-01-01", "2026-10-01", 1234.5, "Pro & TPE", nom="R&eacute;mi <b>Gras</b>"),
        A("C01001", "2023-03-03", "2026-10-31", 19.99, "Particulier", nom="O'Brien"),
    ) if avec_departs else ()
    return M.Rapport(mois=mois, actuel=actuel, precedent=precedent, segments=segments, historique=historique,
                     departs=departs)


def _attendus_page(module: ModuleType, rapport) -> dict:
    F, G = _voisin(module, "formats"), _voisin(module, "graphique")
    P = _voisin(module, "periodes")
    a, p = rapport.actuel, rapport.precedent
    return {
        "titre": f"Rapport churn — {P.nom_du_mois(rapport.mois)}",
        "indicateurs": {
            "actifs_debut": F.nombre(a.actifs_debut, "client"),
            "departs": F.nombre(a.departs, "départ"),
            "taux_churn": F.pourcentage(a.taux_churn),
            "ecart_taux": F.ecart_points(p.taux_churn, a.taux_churn),
            "revenu_perdu": F.euros(a.revenu_perdu),
        },
        "segments": [[s, F.nombre(i.actifs_debut, "client"), F.nombre(i.departs, "départ"), F.pourcentage(i.taux_churn)]
                     for s, i in rapport.segments.items()],
        "departs": [[d.client, d.nom, d.segment, F.euros(d.mensualite), f"{d.fin:%d/%m/%Y}"] for d in rapport.departs],
        "courbe": G.courbe_svg([_modeles(module).etiquette_courte(i.mois) for i in rapport.historique],
                               [i.taux_churn for i in rapport.historique],
                               f"Taux de churn mensuel, de {P.nom_du_mois(rapport.historique[0].mois)} à "
                               f"{P.nom_du_mois(rapport.mois)}"),
    }


def _verifier_securite(lecteur: _Lecteur, page: str) -> None:
    scripts = [a for t, a in lecteur.balises if t == "script"]
    if scripts:
        if 'alert("x")' in "".join(lecteur.texte) or "alert(\"x\")" in page:
            raise AssertionError("FAILLE XSS : la page contient une balise <script>. Le nom d'un client "
                                 "(`<script>alert(\"x\")</script>`) est entré dans la page sans `html.escape` : le "
                                 "navigateur l'EXÉCUTE à l'ouverture du rapport. Échappez tout texte qui vient des "
                                 "données.")
        raise AssertionError("La page contient un <script> : les logiciels de courriel les retirent, et un rapport n'en a "
                             "pas besoin.")
    for tag, a in lecteur.balises:
        if tag == "link" and "stylesheet" in a.get("rel", "").lower():
            raise AssertionError("La page charge une feuille de style (<link rel=\"stylesheet\">) : mettez le style dans "
                                 "un <style> de la page, pour qu'elle s'ouvre sans réseau.")
        for attribut in ("src", "href") if tag == "link" else ("src",):  # un lien <a href> se clique, il ne se charge pas
            if _EXTERNE.match(a.get(attribut, "")):
                raise AssertionError(f"La page télécharge une ressource extérieure (<{tag} {attribut}=\"{a[attribut]}\">) : "
                                     f"le rapport doit s'ouvrir sans réseau, dans dix ans, tel qu'il était.")
        if _URL_EXTERNE.search(a.get("style", "")):
            raise AssertionError(f"Un attribut style de <{tag}> télécharge une ressource extérieure (`url(http…)` ou "
                                 f"`@import`) : tout doit être dans la page.")
    if _URL_EXTERNE.search("".join(lecteur.style)):
        raise AssertionError("Le <style> télécharge une ressource extérieure (`@import` ou `url(http…)`, une police "
                             "Google par exemple) : tout doit être dans la page.")


def _verifier_une_page(module: ModuleType, rapport, quoi: str) -> None:
    page = module.page_html(rapport)  # d'abord : une page pas encore écrite est « à faire », pas « en échec »
    attendus = _attendus_page(module, rapport)
    lecteur = _lire_page(page, quoi)
    _verifier_securite(lecteur, page)

    if not page.lstrip().lower().startswith("<!doctype html>"):
        raise AssertionError(f"{quoi} doit commencer par `<!doctype html>` : sans lui, le navigateur passe en mode "
                             f"« quirks » et la mise en page change.")
    if not any(t == "html" and a.get("lang", "").lower() == "fr" for t, a in lecteur.balises):
        raise AssertionError(f"{quoi} : il faut `<html lang=\"fr\">` (la langue sert aux lecteurs d'écran et à la "
                             f"coupure des mots).")
    if not any(t == "meta" and a.get("charset", "").lower() == "utf-8" for t, a in lecteur.balises):
        raise AssertionError(f"{quoi} : il faut `<meta charset=\"utf-8\">`, sinon « é » et « € » peuvent s'afficher "
                             f"« Ã© » et « â‚¬ ».")
    titre = attendus["titre"]
    if _propre("".join(lecteur.titre)) != titre:
        raise AssertionError(f"{quoi} : <title> contient {_propre(''.join(lecteur.titre))!r} au lieu de {titre!r}.")
    if _propre("".join(lecteur.h1)) != titre:
        raise AssertionError(f"{quoi} : il faut un <h1> qui dit {titre!r} (il contient "
                             f"{_propre(''.join(lecteur.h1))!r}).")

    courbe = attendus["courbe"]
    if courbe not in page:
        import html as html_module
        if html_module.escape(courbe) in page:
            raise AssertionError(f"{quoi} : la courbe a été échappée : le navigateur afficherait le CODE du dessin au "
                                 f"lieu du dessin. Elle est déjà du SVG : insérez-la telle quelle.")
        raise AssertionError(f"{quoi} : la page doit contenir, telle quelle, la courbe renvoyée par votre "
                             f"`courbe_svg` (étape 4) avec les étiquettes `etiquette_courte(i.mois)`, les valeurs "
                             f"`i.taux_churn` des mois de `rapport.historique` et le titre de la docstring.")

    for nom, attendu in attendus["indicateurs"].items():
        if nom not in lecteur.indicateurs:
            raise AssertionError(f"{quoi} : aucun élément avec `data-indicateur=\"{nom}\"` "
                                 f"(il doit contenir {attendu!r}).")
        if lecteur.indicateurs[nom] != attendu:
            raise AssertionError(f"{quoi} : l'élément `data-indicateur=\"{nom}\"` contient "
                                 f"{lecteur.indicateurs[nom]!r} au lieu de {attendu!r} (EXACTEMENT ce texte : le reste "
                                 f"de la phrase va hors de l'élément).")

    lignes = lecteur.lignes("segments")
    if "segments" not in lecteur.tableaux:
        raise AssertionError(f"{quoi} : aucun `<table data-tableau=\"segments\">`.")
    if lignes != attendus["segments"]:
        en_clair = [s for s in rapport.segments if "<" in s]
        if any(t in ("i", "b") for t, _ in lecteur.balises) and en_clair:
            raise AssertionError(f"{quoi} : le segment {en_clair[0]!r} n'est pas échappé : ses balises <i> sont "
                                 f"devenues de la mise en forme. `html.escape` sur tout texte qui vient des données.")
        raise AssertionError(f"{quoi} : le tableau des segments contient {lignes} au lieu de {attendus['segments']} "
                             f"(une ligne de <td> par segment ; les en-têtes en <th>).")

    if rapport.departs:
        if "departs" not in lecteur.tableaux:
            raise AssertionError(f"{quoi} : aucun `<table data-tableau=\"departs\">` alors qu'il y a des départs.")
        lignes = lecteur.lignes("departs")
        if lignes != attendus["departs"]:
            noms = [ligne[1] for ligne in lignes if len(ligne) > 1]
            if any("&amp;" in n or "&lt;" in n or "&quot;" in n for n in noms):
                raise AssertionError(f"{quoi} : un nom est échappé DEUX fois (on lit {[n for n in noms if '&amp;' in n or '&lt;' in n or '&quot;' in n][0]!r} "
                                     f"à l'écran) : n'appelez `html.escape` qu'une fois, au moment d'écrire le HTML.")
            if "Rémi Gras" in noms or "R&eacute;mi Gras" in noms or any("Rémi" in n for n in noms):
                raise AssertionError(f"{quoi} : le nom {'R&eacute;mi <b>Gras</b>'!r} n'est pas échappé : le navigateur a "
                                     f"lu « &eacute; » comme un « é » et <b> comme du gras. `html.escape` sur tout "
                                     f"texte qui vient des données.")
            raise AssertionError(f"{quoi} : le tableau des départs contient {lignes} au lieu de {attendus['departs']} "
                                 f"(une ligne de 5 <td> par départ, dans l'ordre reçu).")
    else:
        if "departs" in lecteur.tableaux:
            raise AssertionError(f"{quoi} : sans aucun départ, pas de tableau des départs (un tableau vide), mais la "
                                 f"phrase « Aucun départ ce mois-ci. ».")
        if "Aucun départ ce mois-ci." not in _propre("".join(lecteur.texte)):
            raise AssertionError(f"{quoi} : sans aucun départ, la page doit dire « Aucun départ ce mois-ci. ».")

    if module.page_html(rapport) != page:
        raise AssertionError(f"{quoi} : deux appels avec le même rapport donnent deux pages différentes (une heure de "
                             f"génération ? du hasard ?). La page ne doit dépendre QUE du rapport.")


def verifier_page(module: ModuleType) -> None:
    M = _modeles(module)
    _verifier_une_page(module, _rapport_de_test(M), "page_html (octobre 2026)")
    _verifier_une_page(module, _rapport_de_test(M, avec_departs=False), "page_html (un mois sans départ)")
    _verifier_une_page(module, _rapport_de_test(M, precedent_vide=True),
                       "page_html (un mois précédent sans abonnés, donc sans taux)")


# ============================================================================ Étape 6 : le programme


def _abonnes_cli(M: ModuleType) -> list:
    """Des abonnés avec plus de 10 départs en octobre, des mensualités égales, et des données jusqu'au 31/10."""
    A = _fabrique(M)
    abonnes = [A(f"C0{k:04d}", f"2025-{1 + k % 12:02d}-0{1 + k % 9}", segment=("Particulier", "Pro & TPE")[k % 2],
                 mensualite=20.0 + k) for k in range(1, 40)]
    # Les départs arrivent dans l'ordre DÉCROISSANT des identifiants : un tri qui ne départage pas les
    # mensualités égales par identifiant garde cet ordre, et se voit.
    abonnes += [A(f"C1{k:04d}", "2025-06-15", f"2026-10-{1 + 2 * k:02d}", mensualite=(99.0, 49.5, 99.0, 12.0)[k % 4],
                  segment=("Senior", "Pro & TPE")[k % 2], nom=("Durand & Fils", "Zoé <Zaza>")[k % 2])
                for k in reversed(range(15))]
    abonnes += [A("C20001", "2026-09-03", "2026-09-30", 15.0), A("C20002", "2026-10-10", "2026-10-11", 99.0),
                A("C20003", "2025-11-20", "2026-08-31", 33.0)]
    return abonnes


def _ecrire_abonnes(M: ModuleType, chemin: Path, abonnes) -> Path:
    import csv
    with chemin.open("w", encoding="utf-8", newline="") as f:
        ecrivain = csv.writer(f, lineterminator="\n")
        ecrivain.writerow(M.COLONNES_ABONNES)
        for a in abonnes:
            ecrivain.writerow((a.client, a.nom, a.segment, a.debut.isoformat(), a.fin.isoformat() if a.fin else "",
                               f"{a.mensualite:.2f}"))
    return chemin


def _verifier_construire_rapport(module: ModuleType, M: ModuleType) -> None:
    for quoi, abonnes, mois in (("des abonnés de test, octobre 2026", _abonnes_cli(M), date(2026, 10, 1)),
                                ("les mêmes, demandés avec le 20 octobre", _abonnes_cli(M), date(2026, 10, 20)),
                                ("projets/exemple_abonnes.csv, janvier 2026", M.lire_abonnes(CHEMIN_EXEMPLE),
                                 date(2026, 1, 1))):
        attendu = ref_construire_rapport(abonnes, mois)
        obtenu = _pas_none(module.construire_rapport(list(abonnes), mois), "construire_rapport")
        for champ in ("mois", "actuel", "precedent", "segments", "historique", "departs"):
            if not hasattr(obtenu, champ):
                raise AssertionError(f"construire_rapport doit renvoyer un `Rapport(...)` (modeles.py), pas "
                                     f"{type(obtenu).__name__}.")
        if obtenu.mois != attendu.mois:
            raise AssertionError(f"construire_rapport ({quoi}) : Rapport.mois vaut {obtenu.mois!r} au lieu de "
                                 f"{attendu.mois!r} (le 1er du mois).")
        for champ, nom in (("actuel", "le mois du rapport"), ("precedent", "le mois d'AVANT")):
            if not _memes_valeurs(_valeurs(getattr(obtenu, champ), champ), _valeurs(getattr(attendu, champ), champ)):
                raise AssertionError(f"construire_rapport ({quoi}) : `{champ}` doit être indicateurs_du_mois pour {nom} "
                                     f"({getattr(attendu, champ).mois:%m/%Y}).")
        if list(obtenu.segments) != list(attendu.segments) or any(
                not _memes_valeurs(_valeurs(obtenu.segments[s], "segments"), _valeurs(attendu.segments[s], "segments"))
                for s in attendu.segments):
            raise AssertionError(f"construire_rapport ({quoi}) : `segments` doit être par_segment du mois.")
        historique = tuple(obtenu.historique)
        mois_obtenus = [getattr(i, "mois", None) for i in historique]
        mois_attendus = [i.mois for i in attendu.historique]
        if mois_obtenus != mois_attendus:
            if mois_obtenus == mois_attendus[::-1]:
                raise AssertionError(f"construire_rapport ({quoi}) : l'historique est à l'envers : du plus ANCIEN au "
                                     f"plus récent, le mois du rapport en dernier (la courbe se lit de gauche à droite).")
            raise AssertionError(f"construire_rapport ({quoi}) : l'historique couvre {[f'{m:%m/%Y}' for m in mois_obtenus if m]} "
                                 f"au lieu des {MOIS_HISTORIQUE} derniers mois, le mois du rapport inclus "
                                 f"({mois_attendus[0]:%m/%Y} à {mois_attendus[-1]:%m/%Y}).")
        if any(not _memes_valeurs(_valeurs(o, "historique"), _valeurs(a, "historique"))
               for o, a in zip(historique, attendu.historique)):
            raise AssertionError(f"construire_rapport ({quoi}) : chaque mois de l'historique doit être "
                                 f"indicateurs_du_mois pour ce mois-là.")
        clients = [getattr(d, "client", d) for d in obtenu.departs]
        attendus = [d.client for d in attendu.departs]
        if clients != attendus:
            tous = sorted((d for d in abonnes if d.fin and attendu.mois <= d.fin <= ref_fin_du_mois(attendu.mois)),
                          key=lambda d: (-d.mensualite, d.client))
            if len(clients) > DEPARTS_AFFICHES:
                raise AssertionError(f"construire_rapport ({quoi}) : {len(clients)} départs dans le rapport : gardez les "
                                     f"DEPARTS_AFFICHES ({DEPARTS_AFFICHES}) premiers.")
            par_client = {d.client: d.mensualite for d in abonnes}
            if sorted(clients) == sorted(attendus) or [par_client.get(c) for c in clients] == \
                    [d.mensualite for d in attendu.departs]:
                raise AssertionError(f"construire_rapport ({quoi}) : les départs sont dans l'ordre {clients} au lieu de "
                                     f"{attendus} : mensualité DÉCROISSANTE, puis identifiant croissant à mensualité "
                                     f"égale (`key=lambda a: (-a.mensualite, a.client)`).")
            if set(clients) <= {d.client for d in tous}:
                raise AssertionError(f"construire_rapport ({quoi}) : les départs {clients} ne sont pas les "
                                     f"{DEPARTS_AFFICHES} plus grosses mensualités ({attendus}).")
            raise AssertionError(f"construire_rapport ({quoi}) : `departs` contient {clients} au lieu de {attendus} "
                                 f"(les abonnés partis PENDANT le mois, même règle qu'à l'étape 2).")
        if not isinstance(obtenu.departs, tuple) or not isinstance(obtenu.historique, tuple):
            raise AssertionError(f"construire_rapport ({quoi}) : `historique` et `departs` sont des tuples (le Rapport "
                                 f"est figé : personne ne doit pouvoir y ajouter un mois après coup).")
    generateur = _pas_none(module.construire_rapport((a for a in _abonnes_cli(M)), date(2026, 10, 1)),
                           "construire_rapport(générateur)")
    if [getattr(d, "client", d) for d in generateur.departs] != [d.client for d in ref_construire_rapport(
            _abonnes_cli(M), date(2026, 10, 1)).departs] or getattr(generateur.actuel, "actifs_debut", None) != \
            ref_construire_rapport(_abonnes_cli(M), date(2026, 10, 1)).actuel.actifs_debut:
        raise AssertionError("construire_rapport donne un autre résultat quand `abonnes` est un générateur : il est vide "
                             "après le premier passage. Commencez par `abonnes = list(abonnes)`.")


def _sans_encodage(module: ModuleType) -> list[str]:
    """Les appels open / write_text / read_text du fichier qui n'ont pas d'`encoding=`."""
    try:
        arbre = ast.parse(textwrap.dedent(inspect.getsource(module)))
    except (OSError, TypeError, SyntaxError):
        return []
    fautes = []
    for noeud in ast.walk(arbre):
        if not isinstance(noeud, ast.Call):
            continue
        nom = noeud.func.id if isinstance(noeud.func, ast.Name) else getattr(noeud.func, "attr", None)
        if nom not in ("open", "write_text", "read_text"):
            continue
        if any(k.arg == "encoding" for k in noeud.keywords):
            continue
        position_mode = 1 if isinstance(noeud.func, ast.Name) else 0
        modes = [noeud.args[position_mode]] if nom == "open" and len(noeud.args) > position_mode else []
        modes += [k.value for k in noeud.keywords if k.arg == "mode"]
        if any(isinstance(m, ast.Constant) and isinstance(m.value, str) and "b" in m.value for m in modes):
            continue
        positions = {"write_text": 2, "read_text": 1, "open": 4 if isinstance(noeud.func, ast.Name) else 3}
        if len(noeud.args) >= positions[nom]:
            continue  # encodage donné par position
        fautes.append(f"`{ast.unparse(noeud)}` (ligne {noeud.lineno})")
    return fautes


def _executer(main, argv: list[str], quoi: str) -> tuple[int, str, str]:
    try:
        return _lancer(main, argv)
    except (NotImplementedError, AssertionError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"`{quoi}` lève {type(exc).__name__} ({exc}) : une erreur prévue s'affiche en UNE phrase "
                             f"sur stderr, avec le code 1, jamais en traceback.") from None


def _commande(argv: list[str]) -> str:
    return "python -m projets.rapport " + " ".join(argv)


def _attendre_erreur(main, argv: list[str], dossier_sortie: Path, pourquoi: str, codes=(1,)) -> None:
    quoi = _commande(argv)
    code, out, err = _executer(main, argv, quoi)
    if code == 0 or code not in codes:
        raise AssertionError(f"`{quoi}` renvoie le code {code!r} au lieu de {codes[0]} : {pourquoi}")
    if not err.strip():
        raise AssertionError(f"`{quoi}` : le message d'erreur doit aller sur stderr (`print(..., file=sys.stderr)`), "
                             f"pas sur stdout.")
    if "Traceback" in err:
        raise AssertionError(f"`{quoi}` affiche un traceback : une erreur prévue s'explique en UNE phrase.")
    if dossier_sortie.exists() and any(dossier_sortie.iterdir()):
        raise AssertionError(f"`{quoi}` a quand même écrit un rapport : en cas d'erreur, aucun fichier.")


def verifier_cli(module: ModuleType) -> None:
    M = _modeles(module)
    _verifier_construire_rapport(module, M)

    fautes = _sans_encodage(module)
    if fautes:
        raise AssertionError(f"Dans cli.py, {fautes[0]} n'a pas d'`encoding=`. Python prend alors l'encodage du "
                             f"système : cp1252 sous Windows, qui ne connaît pas l'espace fine insécable "
                             f"(UnicodeEncodeError) et écrit « é » autrement que le navigateur ne le lit. Écrivez "
                             f"`encoding=\"utf-8\"`.")

    main = module.main
    with _dossier_temporaire() as dossier:
        abonnes = _abonnes_cli(M)
        fichier = _ecrire_abonnes(M, dossier / "abonnes.csv", abonnes)
        sortie = dossier / "sorties" / "rapports"
        argv = ["--abonnes", str(fichier), "--mois", "2026-10", "--sortie", str(sortie), "--date-du-jour", "2026-11-19"]
        code, out, err = _executer(main, argv, _commande(argv))
        if code is None:
            raise AssertionError(f"`{_commande(argv)}` : main renvoie None. Terminez par `return 0`.")
        if code != 0:
            raise AssertionError(f"`{_commande(argv)}` renvoie le code {code!r} au lieu de 0. stderr : {err.strip()[:300]!r}")
        chemin = sortie / "rapport_2026-10.html"
        if not chemin.exists():
            ecrits = [str(p.relative_to(dossier)) for p in dossier.rglob("*.html")]
            raise AssertionError(f"`{_commande(argv)}` devait écrire {chemin.relative_to(dossier)} (dans le dossier "
                                 f"--sortie, créé s'il le faut), mais on trouve {ecrits or 'aucun fichier HTML'}.")
        try:
            contenu = chemin.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise AssertionError("Le rapport écrit n'est pas en UTF-8 : précisez `encoding=\"utf-8\"`.") from None
        attendu = module.page_html(module.construire_rapport(M.lire_abonnes(fichier), date(2026, 10, 1)))
        if contenu.replace("\r\n", "\n") != attendu.replace("\r\n", "\n"):
            raise AssertionError("Le fichier écrit n'est pas exactement page_html(construire_rapport(abonnés, mois)).")
        if str(chemin) not in out:
            raise AssertionError(f"Affichez le chemin du rapport écrit ({chemin}) : c'est le fichier qu'on va ouvrir.")
        if ESPACE_MILLIERS in out or ESPACE_UNITE in out:
            raise AssertionError("Le résumé affiché dans le terminal contient des espaces insécables : une console "
                                 "Windows redirigée vers un fichier écrit en cp1252, et print() planterait sur l'espace "
                                 "fine. Remplacez ESPACE_MILLIERS et ESPACE_UNITE par des espaces ordinaires.")
        rapport = ref_construire_rapport(abonnes, date(2026, 10, 1))
        taux = ref_formats.pourcentage(rapport.actuel.taux_churn).replace(ESPACE_UNITE, " ")
        ecart = ref_formats.ecart_points(rapport.precedent.taux_churn, rapport.actuel.taux_churn).replace(ESPACE_UNITE, " ")
        for morceau, nom in ((taux, "le taux de churn"), (ecart, "l'écart avec le mois d'avant"),
                             (str(rapport.actuel.actifs_debut), "le nombre de clients au début du mois")):
            if morceau not in out:
                raise AssertionError(f"Le résumé affiché doit donner {nom} ({morceau!r}). On lit : {out.strip()[:300]!r}")
        if err.strip():
            raise AssertionError(f"Quand tout va bien, rien sur stderr (on y lit {err.strip()[:200]!r}).")

        code, _, _ = _executer(main, argv, _commande(argv))
        if code != 0 or chemin.read_text(encoding="utf-8") != contenu:
            raise AssertionError("Relancer la même commande doit réécrire le même rapport, et renvoyer 0.")

        precedent = Path.cwd()
        os.chdir(dossier)
        try:
            argv_defaut = ["--abonnes", str(fichier), "--mois", "2026-10", "--date-du-jour", "2026-11-19"]
            code, _, err = _executer(main, argv_defaut, _commande(argv_defaut))
        finally:
            os.chdir(precedent)
        if code != 0 or not (dossier / "rapports" / "rapport_2026-10.html").exists():
            raise AssertionError("Sans --sortie, le rapport s'écrit dans le dossier `rapports` (créé s'il le faut).")

        limite = dossier / "limite"
        argv_limite = ["--abonnes", str(fichier), "--mois", "2026-10", "--sortie", str(limite), "--date-du-jour",
                       "2026-11-01"]
        code, _, err = _executer(main, argv_limite, _commande(argv_limite))
        if code != 0:
            raise AssertionError(f"`{_commande(argv_limite)}` doit fonctionner : le 1er novembre, octobre est terminé "
                                 f"(stderr : {err.strip()[:200]!r}).")

        vide = dossier / "rien"
        erreurs = (
            (["--mois", "2026-10", "--date-du-jour", "2026-10-31"],
             "le 31 octobre, octobre n'est pas encore terminé : refusez (fin_du_mois(mois) doit être AVANT la date du "
             "jour).", (1,), fichier),
            (["--mois", "2026-11", "--date-du-jour", "2026-11-19"],
             "novembre n'est pas terminé le 19 novembre : un rapport fait trop tôt oublie les départs des derniers "
             "jours.", (1,), fichier),
            (["--mois", "2026-13", "--date-du-jour", "2026-11-19"], "« 2026-13 » n'est pas un mois (lire_mois lève "
                                                                    "ValueError).", (1, 2), fichier),
            (["--mois", "2026-10", "--date-du-jour", "19/11/2026"], "la date du jour s'écrit AAAA-MM-JJ.", (1, 2),
             fichier),
            (["--mois", "2027-03", "--date-du-jour", "2027-06-01"], "aucune date du fichier n'est en mars 2027 ou "
                                                                    "après : le fichier s'arrête avant ce mois.",
             (1,), fichier),
            (["--mois", "2026-10", "--date-du-jour", "2026-11-19"], "le fichier n'existe pas (FileNotFoundError).",
             (1,), dossier / "introuvable.csv"),
        )
        for options, pourquoi, codes, source in erreurs:
            _attendre_erreur(main, ["--abonnes", str(source), *options, "--sortie", str(vide)], vide, pourquoi, codes)
        faux = dossier / "faux.csv"
        faux.write_text("id;nom\nC1;x\n", encoding="utf-8")
        _attendre_erreur(main, ["--abonnes", str(faux), "--mois", "2026-10", "--date-du-jour", "2026-11-19",
                                "--sortie", str(vide)], vide,
                         "le fichier n'a pas les bonnes colonnes (lire_abonnes lève ValueError).")


# ============================================================================ Étape 7 : le test


def _dans_vos_classes(fonction, M: ModuleType):
    """La même fonction, mais qui renvoie VOTRE classe Indicateurs (pour qu'un `==` de votre test fonctionne)."""
    def indicateurs_du_mois(abonnes, mois):
        resultat = fonction(abonnes, mois)
        return M.Indicateurs(**{c: getattr(resultat, c) for c in _CHAMPS})
    return indicateurs_du_mois


def verifier_tester_indicateurs(module: ModuleType) -> None:
    M = _modeles(module)
    tester = module.tester_indicateurs_du_mois
    try:
        tester(_dans_vos_classes(_variante(), M))
    except NotImplementedError:
        raise
    except AssertionError as exc:
        raise AssertionError(f"Votre test ÉCHOUE sur la version JUSTE d'indicateurs_du_mois "
                             f"({str(exc) or 'assert sans message'}). Relisez les règles de l'étape 2 et recalculez vos "
                             f"cas à la main (le 1er, le dernier jour, la veille, le lendemain).") from None
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"Votre test plante sur la version JUSTE : {type(exc).__name__}: {exc}. Fabriquez vos "
                             f"abonnés avec `_abonne(...)`.") from None

    survivants = []
    for options, quoi, _ in _MUTANTS_INDICATEURS:
        try:
            tester(_dans_vos_classes(_variante(**options), M))
        except Exception:  # noqa: BLE001 — n'importe quelle erreur : le test a vu le bug
            continue
        survivants.append(quoi)
    if survivants:
        if len(survivants) == len(_MUTANTS_INDICATEURS):
            raise AssertionError("Votre test passe sur TOUTES les versions boguées : il ne vérifie rien. Écrivez des "
                                 "`assert` sur actifs_debut, arrivees, departs, actifs_fin, taux_churn et revenu_perdu.")
        raise AssertionError(f"{len(_MUTANTS_INDICATEURS) - len(survivants)}/{len(_MUTANTS_INDICATEURS)} bugs attrapés. "
                             f"Votre test laisse passer une version qui {survivants[0]}. Ajoutez un cas où cette "
                             f"version et la bonne donnent des chiffres différents.")


# ============================================================================ Le projet

PROJET = Projet(
    numero=7,
    code="rapport",
    titre="Le rapport du mois",
    paquet_eleve="projets.rapport",
    paquet_corrige="projets.rapport_corrige",
    commande="python -m projets.rapport --abonnes projets/exemple_abonnes.csv --mois 2026-10",
    etapes=(
        Etape(1, "Les mois du calendrier", "🟢", "periodes", ("lire_mois", "fin_du_mois", "decaler", "nom_du_mois"),
              verifier_periodes),
        Etape(2, "Les chiffres du mois : le bon dénominateur", "🟡", "indicateurs",
              ("indicateurs_du_mois", "par_segment"), verifier_indicateurs),
        Etape(3, "Écrire les nombres comme un humain", "🟡", "formats",
              ("euros", "pourcentage", "ecart_points", "nombre"), verifier_formats),
        Etape(4, "La courbe des 12 derniers mois, en SVG", "🔴", "graphique", ("echelle_haute", "courbe_svg"),
              verifier_graphique),
        Etape(5, "La page HTML, autonome et sans injection", "🔴", "page", ("page_html",), verifier_page),
        Etape(6, "Le programme du début de mois", "🟡", "cli", ("construire_rapport", "main"), verifier_cli),
        Etape(7, "Écrire le test qui protège indicateurs_du_mois", "🔴", "mes_tests", ("tester_indicateurs_du_mois",),
              verifier_tester_indicateurs),
    ),
)

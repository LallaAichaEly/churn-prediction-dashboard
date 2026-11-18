"""Vérifications automatiques du projet 6 (l'entrepôt des clients).

Le correcteur écrit ses propres exports, avec un piège à la fois : une BOM, des fins de ligne Windows,
une ligne vide, un compte de test, une date du 31 février, un « nan » qui passe pour un nombre, une
ligne JSON coupée, un identifiant écrit sans guillemets, une mise à jour arrivée avant la version
qu'elle remplace. Il recharge un vieux fichier par-dessus un récent, simule un disque plein au milieu
d'un chargement, place des tickets PILE sur les frontières de la fenêtre, et rejoue 13 versions
boguées classiques de `construire_table` pour vous dire LAQUELLE vous avez écrite.

Lire ce fichier est permis (les cas testés sont des indices), mais essayez d'abord seul·e.
"""

from __future__ import annotations

import contextlib
import csv
import importlib
import json
import sqlite3
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pandas as pd

from .checks_reentrainement import _appel, _leve_value_error
from .checks_retention import _lancer, _pas_none
from .entrepot_corrige.fichier import charger_fichier as ref_charger_fichier
from .entrepot_corrige.modeles import (COLONNES_CRM, COLONNES_TABLE, FENETRE_JOURS, HORIZON_JOURS, mois_entre,
                                       ouvrir_base as ref_ouvrir_base)
from .entrepot_corrige.table import construire_table as ref_construire_table
from .exemples_entrepot import PHOTO_EXEMPLE, generer_exports
from .verif import Etape, Projet

# ---------------------------------------------------------------------------- outils


def _modeles(module: ModuleType) -> ModuleType:
    """Le `modeles.py` du MÊME paquet que le module testé (le vôtre ou celui du corrigé)."""
    return importlib.import_module(f"{module.__package__}.modeles")


@contextlib.contextmanager
def _dossier_temporaire():
    with tempfile.TemporaryDirectory(prefix="projet_entrepot_") as dossier:
        yield Path(dossier)


@contextlib.contextmanager
def _remplacer(module: ModuleType, nom: str, par, source: str):
    """Remplace `nom` dans le module testé ET dans le fichier où il est défini (`source`), puis remet tout."""
    autre = importlib.import_module(f"{module.__package__}.{source}")
    anciens = [(m, getattr(m, nom)) for m in {id(module): module, id(autre): autre}.values() if hasattr(m, nom)]
    for m, _ in anciens:
        setattr(m, nom, par)
    try:
        yield
    finally:
        for m, ancien in anciens:
            setattr(m, nom, ancien)


def _ecrire(dossier: Path, nom: str, lignes: list[str], bom: bool = True, crlf: bool = False) -> Path:
    fin = "\r\n" if crlf else "\n"
    contenu = (b"\xef\xbb\xbf" if bom else b"") + (fin.join(lignes) + fin).encode("utf-8")
    chemin = dossier / nom
    chemin.write_bytes(contenu)
    return chemin


def _lecture(resultat, quoi: str) -> tuple[list, list]:
    resultat = _pas_none(resultat, quoi)
    if not hasattr(resultat, "lignes") or not hasattr(resultat, "rejets"):
        raise AssertionError(f"{quoi} doit renvoyer `Lecture(lignes=..., rejets=...)` (modeles.py), pas "
                             f"{type(resultat).__name__}.")
    return list(resultat.lignes), list(resultat.rejets)


def _montrer(rejets: list) -> str:
    return "[" + ", ".join(f"ligne {getattr(r, 'ligne', '?')} : {getattr(r, 'raison', r)!r}" for r in rejets[:3]) + \
        (", …]" if len(rejets) > 3 else "]")


# ============================================================================ Étape 1 : lire le CRM

_ENTETE = ";".join(COLONNES_CRM)
_L1 = "C00001;14/03/2024;;Mensuel;Prélèvement;Fibre;Non;Non;2;79,90;2026-01-04T17:42:00"
_L2 = "C00002;01/02/2023;15/07/2026;Annuel;Chèque;ADSL;Oui;Oui;1;45,5;2026-06-02T09:00:05"
_L3 = "C00003 ; 30/06/2026 ;; Biannuel ;Carte bancaire;Aucun;Non;Non; 4 ;25,00;2026-06-30T23:59:59"
_ATTENDUS_CRM = (
    ("C00001", date(2024, 3, 14), None, "Mensuel", "Prélèvement", "Fibre", "Non", "Non", 2, 79.9,
     datetime(2026, 1, 4, 17, 42)),
    ("C00002", date(2023, 2, 1), date(2026, 7, 15), "Annuel", "Chèque", "ADSL", "Oui", "Oui", 1, 45.5,
     datetime(2026, 6, 2, 9, 0, 5)),
    ("C00003", date(2026, 6, 30), None, "Biannuel", "Carte bancaire", "Aucun", "Non", "Non", 4, 25.0,
     datetime(2026, 6, 30, 23, 59, 59)),
)


def _champ(ligne: str, numero: int, valeur: str) -> str:
    champs = ligne.split(";")
    champs[numero] = valeur
    return ";".join(champs)


#: (ligne piégée, mot que la raison doit citer, pourquoi la ligne doit être écartée)
_PIEGES_CRM = (
    (_L1.rsplit(";", 1)[0], "champs", "elle n'a que 10 champs (le dernier `;` manque)"),
    (_L1 + ";", "champs", "elle a 12 champs (un `;` en trop)"),
    (_champ(_L1, 0, "TEST01"), "client", "« TEST01 » ne respecte pas FORMAT_CLIENT (un compte de test)"),
    (_champ(_L1, 0, "C1"), "client", "« C1 » ne respecte pas FORMAT_CLIENT (C suivi de 5 chiffres)"),
    (_champ(_L1, 1, "31/02/2026"), "debut", "le 31 février n'existe pas (`strptime` le refuse)"),
    (_champ(_L1, 1, "2026-02-03"), "debut", "« 2026-02-03 » n'est pas au format JJ/MM/AAAA"),
    (_champ(_L1, 1, ""), "debut", "la date de début est obligatoire"),
    (_champ(_L1, 2, "01/01/2020"), "fin", "la fin (01/01/2020) est avant le début (14/03/2024)"),
    (_champ(_L1, 2, "bientôt"), "fin", "« bientôt » n'est pas une date"),
    (_champ(_L1, 3, "Mensual"), "contrat", "« Mensual » n'est pas une valeur de MODALITES"),
    (_champ(_L1, 5, "fibre"), "internet", "« fibre » (minuscule) n'est pas une valeur de MODALITES"),
    (_champ(_L1, 4, ""), "paiement", "le moyen de paiement est vide"),
    (_champ(_L1, 7, "Peut-être"), "senior", "« Peut-être » n'est pas Oui ou Non"),
    (_champ(_L1, 8, "0"), "produits", "0 produit : il en faut au moins 1"),
    (_champ(_L1, 8, "2.5"), "produits", "2.5 n'est pas un entier"),
    (_champ(_L1, 8, "deux"), "produits", "« deux » n'est pas écrit avec des chiffres"),
    (_champ(_L1, 9, "N/A"), "mensualite", "« N/A » n'est pas un montant"),
    (_champ(_L1, 9, "nan"), "mensualite", "« nan » n'est pas un montant (mais `float('nan')` ne lève pas "
                                          "d'erreur : `math.isfinite`)"),
    (_champ(_L1, 9, "inf"), "mensualite", "« inf » n'est pas un montant (`math.isfinite`)"),
    (_champ(_L1, 9, "-5,00"), "mensualite", "une mensualité négative n'a pas de sens"),
    (_champ(_L1, 9, "0,00"), "mensualite", "la mensualité doit être STRICTEMENT positive"),
    (_champ(_L1, 10, "hier"), "maj", "« hier » n'est pas une date et heure ISO"),
)


def _comparer_contrat(obtenu, attendu: tuple, ou: str) -> None:
    noms = ("client", "debut", "fin", "contrat", "paiement", "internet", "assistance", "senior", "produits",
            "mensualite", "maj")
    for nom, valeur in zip(noms, attendu):
        lu = getattr(obtenu, nom, "<absent>")
        if nom in ("debut", "fin") and valeur is not None:
            if type(lu) is datetime:  # noqa: E721 — datetime est une sous-classe de date
                raise AssertionError(f"{ou}, `{nom}` : un datetime ({lu!r}) au lieu d'une date. "
                                     f"`datetime.strptime(...).date()` : on veut un jour, pas un instant.")
            if type(lu) is not date:  # noqa: E721
                raise AssertionError(f"{ou}, `{nom}` : attendu {valeur!r} (un objet date), obtenu {lu!r}.")
        if nom == "fin" and valeur is None and lu is not None:
            raise AssertionError(f"{ou}, `fin` est vide dans le fichier : attendu None, obtenu {lu!r}.")
        if nom == "produits" and (type(lu) is not int):  # noqa: E721
            raise AssertionError(f"{ou}, `produits` : attendu l'entier {valeur}, obtenu {lu!r} "
                                 f"({type(lu).__name__}).")
        if nom == "mensualite":
            if not isinstance(lu, float) or abs(lu - valeur) > 1e-9:
                raise AssertionError(f"{ou}, `mensualite` : attendu {valeur!r} (un float), obtenu {lu!r}. "
                                     f"« 79,90 » s'écrit avec une virgule : remplacez-la par un point avant float().")
            continue
        if nom == "maj" and type(lu) is not datetime:  # noqa: E721
            raise AssertionError(f"{ou}, `maj` : attendu {valeur!r} (un datetime), obtenu {lu!r}.")
        if lu != valeur:
            conseil = " Retirez les espaces autour de chaque valeur (`strip`)." if isinstance(lu, str) and \
                lu.strip() == valeur else ""
            raise AssertionError(f"{ou}, `{nom}` : attendu {valeur!r}, obtenu {lu!r}.{conseil}")


def _lire_contrats(module, chemin: Path, quoi: str) -> tuple[list, list]:
    try:
        resultat = module.lire_contrats(chemin)
    except (NotImplementedError, AssertionError):
        raise
    except (ValueError, KeyError) as exc:
        if "﻿" in str(exc) or "ufeff" in repr(exc):
            raise AssertionError(f"Sur {quoi}, la première colonne s'appelle '\\ufeffclient' : c'est la BOM écrite "
                                 f"par Excel. Ouvrez le fichier avec encoding=\"utf-8-sig\".") from None
        raise AssertionError(f"Sur {quoi}, lire_contrats lève {type(exc).__name__}: {exc}") from None
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"Sur {quoi}, lire_contrats lève {type(exc).__name__}: {exc}") from None
    return _lecture(resultat, "lire_contrats")


def verifier_lire_contrats(module: ModuleType) -> None:
    with _dossier_temporaire() as dossier:
        # 1. Un fichier sain, tel qu'Excel l'écrit : BOM, \r\n, lignes vides à la fin.
        chemin = _ecrire(dossier, "crm_2026-07-01.csv", [_ENTETE, _L1, _L2, _L3, "", ""], crlf=True)
        lignes, rejets = _lire_contrats(module, chemin, "un fichier sain (BOM, fins de ligne Windows)")
        if rejets:
            vides = [r for r in rejets if getattr(r, "ligne", None) in (5, 6)]
            if vides:
                raise AssertionError("Les lignes vides de la fin du fichier sont devenues des rejets : ignorez-les "
                                     "(ce ne sont ni des contrats ni des erreurs).")
            raise AssertionError(f"Un fichier sain donne des rejets : {_montrer(rejets)}")
        if len(lignes) != 3:
            raise AssertionError(f"Un fichier sain de 3 contrats (suivis de 2 lignes vides) donne {len(lignes)} "
                                 f"contrat(s) au lieu de 3.")
        for numero, (obtenu, attendu) in enumerate(zip(lignes, _ATTENDUS_CRM), start=2):
            _comparer_contrat(obtenu, attendu, f"Ligne {numero} du fichier sain")

        # Le même, sans BOM et avec des fins de ligne \n : doit aussi se lire.
        chemin = _ecrire(dossier, "crm_2026-07-02.csv", [_ENTETE, _L1, _L2], bom=False)
        lignes, rejets = _lire_contrats(module, chemin, "un fichier sans BOM")
        if len(lignes) != 2 or rejets:
            raise AssertionError(f"Sans BOM, un fichier de 2 contrats donne {len(lignes)} contrat(s) et "
                                 f"{len(rejets)} rejet(s). utf-8-sig lit aussi les fichiers sans BOM : ne retirez "
                                 f"pas le premier caractère « à la main ».")

        # 2. Chaque piège, seul entre deux bonnes lignes.
        for piege, mot, pourquoi in _PIEGES_CRM:
            chemin = _ecrire(dossier, "crm_piege.csv", [_ENTETE, _L1, piege, _L2])
            try:
                resultat = module.lire_contrats(chemin)
            except (NotImplementedError, AssertionError):
                raise
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(f"La ligne « {piege} » fait planter lire_contrats ({type(exc).__name__}: {exc}). "
                                     f"Une ligne abîmée ne doit pas arrêter la lecture : mettez-la dans `rejets` "
                                     f"(try/except ValueError) et continuez.") from None
            lignes, rejets = _lecture(resultat, "lire_contrats")
            if len(lignes) == 3 and not rejets:
                raise AssertionError(f"La ligne « {piege} » a été acceptée, alors que {pourquoi}.")
            if len(rejets) != 1 or len(lignes) != 2:
                raise AssertionError(f"Avec la ligne « {piege} » entre deux bonnes lignes : {len(lignes)} contrat(s) "
                                     f"et {len(rejets)} rejet(s), attendu 2 et 1.")
            rejet = rejets[0]
            if getattr(rejet, "ligne", None) != 3:
                raise AssertionError(f"Le rejet de la ligne « {piege} » porte le numéro {getattr(rejet, 'ligne', None)!r} "
                                     f"au lieu de 3 : l'en-tête est la ligne 1 (`lecteur.line_num`).")
            if getattr(rejet, "fichier", None) != "crm_piege.csv":
                raise AssertionError(f"Rejet.fichier vaut {getattr(rejet, 'fichier', None)!r} : attendu le nom du "
                                     f"fichier sans le dossier, 'crm_piege.csv' (`Path(chemin).name`).")
            raison = str(getattr(rejet, "raison", ""))
            if mot not in raison:
                raise AssertionError(f"La ligne « {piege} » est bien écartée, mais sa raison ({raison!r}) ne cite pas "
                                     f"« {mot} » : l'équipe du CRM ne saura pas quoi corriger.")

        # 3. Tous les pièges ensemble, avec des lignes vides au milieu : l'ordre et les numéros.
        contenu = [_ENTETE, _L1, "", *[p for p, _, _ in _PIEGES_CRM[:6]], "", _L2, _PIEGES_CRM[9][0], _L3]
        chemin = _ecrire(dossier, "crm_tout.csv", contenu, crlf=True)
        lignes, rejets = _lire_contrats(module, chemin, "un fichier qui mélange bonnes et mauvaises lignes")
        numeros = [getattr(r, "ligne", None) for r in rejets]
        if [getattr(c, "client", None) for c in lignes] != ["C00001", "C00002", "C00003"] or numeros != \
                [4, 5, 6, 7, 8, 9, 12]:
            raise AssertionError(f"Sur un fichier qui mélange 3 bonnes lignes, 7 mauvaises et des lignes vides : "
                                 f"contrats {[getattr(c, 'client', None) for c in lignes]}, rejets aux lignes "
                                 f"{numeros} ; attendu les contrats C00001, C00002, C00003 et les rejets aux lignes "
                                 f"[4, 5, 6, 7, 8, 9, 12] (les lignes vides comptent dans la numérotation).")

        # 4. Un en-tête inattendu : ce n'est pas le bon fichier.
        for entete, quoi in ((_ENTETE.replace(";", ","), "séparé par des virgules"),
                             ("client;debut;fin;contrat", "auquel il manque des colonnes"),
                             (";".join(reversed(COLONNES_CRM)), "aux colonnes dans le désordre")):
            chemin = _ecrire(dossier, "crm_entete.csv", [entete, _L1])
            _leve_value_error(lambda: module.lire_contrats(chemin), f"Un fichier à l'en-tête {quoi}",
                              "Comparez l'en-tête à COLONNES_CRM avant de lire la suite.")

        chemin = _ecrire(dossier, "crm_vide.csv", [_ENTETE])
        lignes, rejets = _lire_contrats(module, chemin, "un fichier qui n'a que l'en-tête")
        if lignes or rejets:
            raise AssertionError("Un fichier qui n'a que l'en-tête doit donner Lecture(lignes=[], rejets=[]).")


# ============================================================================ Étape 2 : lire le support


def _jt(ticket="T000001", client="C00001", ouvert="2026-02-14T10:03:00", maj=None, **autres) -> str:
    objet = {"ticket": ticket, "client": client, "ouvert": ouvert, "maj": maj or ouvert, **autres}
    return json.dumps(objet, ensure_ascii=False)


_BON_A = _jt("T000101", "C00101", "2026-03-01T08:00:00", statut="ouvert")
_BON_B = _jt("T000102", "C00102", "2026-03-02T08:00:00", statut="ouvert")

#: (ligne piégée, mot que la raison doit citer (None : n'importe quelle raison), pourquoi)
_PIEGES_SUPPORT = (
    ('{"ticket": "T000009", "client": "C000', "JSON", "elle a été coupée au milieu (JSON illisible)"),
    ('["T000009", "C00009", "2026-02-14T10:03:00"]', None, "c'est une liste JSON, pas un objet {...}"),
    ("42", None, "c'est un nombre, pas un objet {...}"),
    (_jt().replace(', "ouvert": "2026-02-14T10:03:00"', ""), "ouvert", "la clé `ouvert` manque"),
    (json.dumps({"ticket": "T000009", "client": 42, "ouvert": "2026-02-14T10:03:00", "maj": "2026-02-14T10:03:00"}),
     "client", "le client est le NOMBRE 42, pas un texte (`re.fullmatch` planterait : vérifiez le type d'abord)"),
    (_jt(ticket="T12"), "ticket", "« T12 » ne respecte pas FORMAT_TICKET"),
    (_jt(client="C1"), "client", "« C1 » ne respecte pas FORMAT_CLIENT"),
    (_jt(ouvert="14/02/2026 10:03"), "ouvert", "« 14/02/2026 10:03 » n'est pas une date ISO"),
    (_jt(maj="2026-02-13T09:00:00"), "maj", "la mise à jour (13/02) est avant l'ouverture (14/02)"),
    (json.dumps({"ticket": "T000009", "client": "C00009", "ouvert": "2026-02-14T10:03:00", "maj": None}), "maj",
     "`maj` vaut null (None en Python) : `fromisoformat(None)` planterait, vérifiez le type"),
)


def _lire_tickets(module, chemin: Path, quoi: str) -> tuple[list, list]:
    try:
        resultat = module.lire_tickets(chemin)
    except (NotImplementedError, AssertionError):
        raise
    except json.JSONDecodeError as exc:
        conseil = (" `json.load` lit UN document JSON ; ici chaque LIGNE est un document : `json.loads(ligne)` "
                   "dans une boucle." if "Extra data" in str(exc) else
                   " Une ligne vide n'est pas du JSON : ignorez-la (`if not ligne.strip(): continue`)."
                   if "Expecting value" in str(exc) else "")
        raise AssertionError(f"Sur {quoi}, lire_tickets lève JSONDecodeError: {exc}.{conseil}") from None
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"Sur {quoi}, lire_tickets lève {type(exc).__name__}: {exc}") from None
    return _lecture(resultat, "lire_tickets")


def verifier_lire_tickets(module: ModuleType) -> None:
    with _dossier_temporaire() as dossier:
        # 1. Un fichier sain : des clés en plus, une ligne vide, des mises à jour (dont une arrivée AVANT).
        contenu = [
            _jt("T000002", "C00002", "2026-03-01T08:00:00", statut="ouvert"),
            _jt("T000001", "C00001", "2026-02-14T10:03:00", statut="ouvert", canal="téléphone"),
            "",
            _jt("T000001", "C00001", "2026-02-14T10:03:00", "2026-02-15T08:00:00", statut="résolu"),
            _jt("T000003", "C00003", "2026-04-01T09:30:00", "2026-04-02T12:00:00", statut="clos"),
            _jt("T000003", "C00003", "2026-04-01T09:30:00", "2026-04-01T12:00:00", statut="résolu"),
        ]
        chemin = _ecrire(dossier, "support_2026-05-04.jsonl", contenu, bom=False)
        lignes, rejets = _lire_tickets(module, chemin, "un fichier sain")
        if rejets:
            if any(getattr(r, "ligne", None) == 3 for r in rejets):
                raise AssertionError("La ligne vide (ligne 3) est devenue un rejet : ignorez-la.")
            raise AssertionError(f"Un fichier sain donne des rejets : {_montrer(rejets)}")
        ids = [getattr(t, "ticket", None) for t in lignes]
        if len(lignes) > 3:
            raise AssertionError(f"Le fichier contient 3 tickets, dont 2 renvoyés après une mise à jour : vous en "
                                 f"rendez {len(lignes)} ({ids}). Un ticket ne doit apparaître qu'une fois.")
        if sorted(ids) != ["T000001", "T000002", "T000003"]:
            raise AssertionError(f"Tickets obtenus : {ids}, attendu T000001, T000002 et T000003.")
        if ids != ["T000001", "T000002", "T000003"]:
            raise AssertionError(f"Les tickets doivent être triés par identifiant : obtenu {ids}.")
        un, deux, trois = lignes
        for t in lignes:
            for cle in ("ouvert", "maj"):
                if type(getattr(t, cle, None)) is not datetime:  # noqa: E721
                    raise AssertionError(f"{t.ticket} : `{cle}` doit être un datetime, obtenu "
                                         f"{getattr(t, cle, None)!r}.")
        if un.maj != datetime(2026, 2, 15, 8, 0):
            raise AssertionError(f"T000001 a été renvoyé avec maj = 2026-02-15T08:00 : c'est cette version qu'on "
                                 f"garde, obtenu maj = {un.maj}.")
        if trois.maj != datetime(2026, 4, 2, 12, 0):
            raise AssertionError("T000003 apparaît deux fois, la version la plus récente (maj = 2026-04-02) AVANT "
                                 "l'ancienne (2026-04-01) : vous gardez la dernière LUE au lieu de la plus RÉCENTE. "
                                 "Comparez les `maj` avant de remplacer.")
        if (deux.client, deux.ouvert) != ("C00002", datetime(2026, 3, 1, 8, 0)):
            raise AssertionError(f"T000002 : attendu client C00002 ouvert le 2026-03-01T08:00, obtenu "
                                 f"{deux.client!r} ouvert le {deux.ouvert}.")

        # 2. Chaque piège, seul entre deux bonnes lignes.
        for piege, mot, pourquoi in _PIEGES_SUPPORT:
            chemin = _ecrire(dossier, "support_piege.jsonl", [_BON_A, piege, _BON_B], bom=False)
            try:
                resultat = module.lire_tickets(chemin)
            except (NotImplementedError, AssertionError):
                raise
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(f"La ligne « {piege} » fait planter lire_tickets ({type(exc).__name__}: {exc}), "
                                     f"alors que {pourquoi}. Mettez-la dans `rejets` et continuez.") from None
            lignes, rejets = _lecture(resultat, "lire_tickets")
            if len(lignes) == 3 and not rejets:
                raise AssertionError(f"La ligne « {piege} » a été acceptée, alors que {pourquoi}.")
            if len(rejets) != 1 or len(lignes) != 2:
                raise AssertionError(f"Avec la ligne « {piege} » entre deux bonnes lignes : {len(lignes)} ticket(s) "
                                     f"et {len(rejets)} rejet(s), attendu 2 et 1.")
            rejet = rejets[0]
            if getattr(rejet, "ligne", None) != 2:
                raise AssertionError(f"Le rejet de la ligne « {piege} » porte le numéro {getattr(rejet, 'ligne', None)!r} "
                                     f"au lieu de 2 : la première ligne est la ligne 1 (`enumerate(f, start=1)`).")
            if getattr(rejet, "fichier", None) != "support_piege.jsonl":
                raise AssertionError(f"Rejet.fichier vaut {getattr(rejet, 'fichier', None)!r} : attendu "
                                     f"'support_piege.jsonl' (`Path(chemin).name`).")
            raison = str(getattr(rejet, "raison", ""))
            if not raison or (mot is not None and mot not in raison):
                attendu = f"ne cite pas « {mot} »" if mot else "est vide"
                raise AssertionError(f"La ligne « {piege} » est bien écartée, mais sa raison ({raison!r}) {attendu}.")

        # 3. L'ordre des rejets et un fichier vide.
        contenu = [_BON_A, _PIEGES_SUPPORT[5][0], "", _PIEGES_SUPPORT[0][0], _BON_B, _PIEGES_SUPPORT[3][0]]
        chemin = _ecrire(dossier, "support_tout.jsonl", contenu, bom=False)
        _, rejets = _lire_tickets(module, chemin, "un fichier qui mélange bonnes et mauvaises lignes")
        if [getattr(r, "ligne", None) for r in rejets] != [2, 4, 6]:
            raise AssertionError(f"Rejets aux lignes {[getattr(r, 'ligne', None) for r in rejets]}, attendu [2, 4, 6] "
                                 f"(dans l'ordre du fichier, les lignes vides comptent dans la numérotation).")
        chemin = dossier / "support_vide.jsonl"
        chemin.write_text("", encoding="utf-8")
        lignes, rejets = _lire_tickets(module, chemin, "un fichier vide")
        if lignes or rejets:
            raise AssertionError("Un fichier vide doit donner Lecture(lignes=[], rejets=[]).")


# ============================================================================ Étape 3 : ranger dans la base


def _contrats_de_test(M: ModuleType) -> dict[str, object]:
    def c(client, debut, fin, contrat, paiement, maj):
        return M.Contrat(client=client, debut=debut, fin=fin, contrat=contrat, paiement=paiement, internet="Fibre",
                         assistance="Non", senior="Non", produits=2, mensualite=79.9, maj=maj)
    return {
        "c1": c("C00001", date(2024, 3, 14), None, "Mensuel", "Prélèvement", datetime(2026, 1, 4, 17, 42)),
        "c2": c("C00002", date(2023, 2, 1), date(2026, 7, 15), "Annuel", "Chèque", datetime(2026, 6, 2, 9, 0, 5)),
        "c3": c("C00003", date(2026, 6, 30), None, "Biannuel", "Virement", datetime(2026, 6, 30, 23, 59, 59)),
        "c1_recent": c("C00001", date(2024, 3, 14), None, "Annuel", "Carte bancaire", datetime(2026, 2, 1, 8, 0)),
        "c2_ancien": c("C00002", date(2023, 2, 1), None, "Mensuel", "Chèque", datetime(2026, 5, 1, 8, 0)),
        "c4_recent": c("C00004", date(2025, 1, 1), None, "Annuel", "Virement", datetime(2026, 3, 1, 8, 0)),
        "c4_ancien": c("C00004", date(2025, 1, 1), None, "Mensuel", "Chèque", datetime(2026, 1, 1, 8, 0)),
    }


def _lignes(connexion: sqlite3.Connection, table: str) -> list[tuple]:
    return connexion.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()  # noqa: S608 — noms fixes


def _charger(fonction, connexion, lignes: list, quoi: str) -> int:
    try:
        n = fonction(connexion, lignes)
    except (NotImplementedError, AssertionError):
        raise
    except sqlite3.IntegrityError as exc:
        raise AssertionError(f"{quoi} lève IntegrityError ({exc}) : la ligne existe déjà. Un simple INSERT ne suffit "
                             f"pas : ON CONFLICT (...) DO UPDATE ... WHERE excluded.maj > ...maj.") from None
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{quoi} lève {type(exc).__name__}: {exc}") from None
    n = _pas_none(n, quoi)
    if isinstance(n, bool) or not isinstance(n, int):
        raise AssertionError(f"{quoi} doit renvoyer un entier (le nombre de lignes changées), pas {n!r}.")
    return n


def _verifier_textes(connexion, table: str, colonnes: tuple[str, ...], instants: tuple[str, ...]) -> None:
    for colonne in colonnes:
        for (valeur,) in connexion.execute(f"SELECT {colonne} FROM {table} WHERE {colonne} IS NOT NULL"):  # noqa: S608
            if not isinstance(valeur, str):
                raise AssertionError(f"{table}.{colonne} contient {valeur!r} ({type(valeur).__name__}) : rangez les "
                                     f"dates en TEXTE ISO (`.isoformat()`).")
            if valeur in ("None", "NaT", ""):
                raise AssertionError(f"{table}.{colonne} contient le texte {valeur!r} : une date absente doit devenir "
                                     f"NULL (passez None, pas str(None)).")
            if colonne in instants:
                if " " in valeur:
                    raise AssertionError(f"{table}.{colonne} contient {valeur!r}, avec une ESPACE entre le jour et "
                                         f"l'heure : c'est le convertisseur automatique (déprécié) de sqlite3. Passez "
                                         f"`instant.isoformat(timespec=\"seconds\")`, qui écrit un T.")
                try:
                    ok = datetime.fromisoformat(valeur).isoformat(timespec="seconds") == valeur
                except ValueError:
                    ok = False
                if not ok:
                    raise AssertionError(f"{table}.{colonne} contient {valeur!r} : attendu le format "
                                         f"AAAA-MM-JJTHH:MM:SS (`isoformat(timespec=\"seconds\")`).")
            else:
                try:
                    ok = date.fromisoformat(valeur).isoformat() == valeur
                except ValueError:
                    ok = False
                if not ok:
                    raise AssertionError(f"{table}.{colonne} contient {valeur!r} : attendu un jour AAAA-MM-JJ "
                                         f"(`jour.isoformat()`).")


def verifier_charger(module: ModuleType) -> None:
    M = _modeles(module)
    t = _contrats_de_test(M)
    connexion = M.ouvrir_base(":memory:")
    try:
        quoi = "charger_contrats(connexion, [3 nouveaux contrats])"
        n = _charger(module.charger_contrats, connexion, [t["c1"], t["c2"], t["c3"]], quoi)
        if not connexion.in_transaction:
            raise AssertionError("charger_contrats a validé la transaction (commit ou `with connexion:`) : c'est "
                                 "l'appelant (étape 4) qui doit pouvoir tout annuler si la suite échoue.")
        lignes = _lignes(connexion, "contrats")
        if len(lignes) != 3:
            raise AssertionError(f"Après {quoi}, la table contrats a {len(lignes)} ligne(s) au lieu de 3.")
        _verifier_textes(connexion, "contrats", ("debut", "fin", "maj"), ("maj",))
        attendu = [("C00001", "2024-03-14", None, "Mensuel", "Prélèvement", "Fibre", "Non", "Non", 2, 79.9,
                    "2026-01-04T17:42:00"),
                   ("C00002", "2023-02-01", "2026-07-15", "Annuel", "Chèque", "Fibre", "Non", "Non", 2, 79.9,
                    "2026-06-02T09:00:05"),
                   ("C00003", "2026-06-30", None, "Biannuel", "Virement", "Fibre", "Non", "Non", 2, 79.9,
                    "2026-06-30T23:59:59")]
        if lignes != attendu:
            raise AssertionError(f"Après {quoi}, la table contrats contient {lignes[0]!r}… ; attendu {attendu[0]!r}… "
                                 f"(les colonnes dans l'ordre du SCHEMA).")
        if n != 3:
            raise AssertionError(f"{quoi} renvoie {n}, attendu 3 lignes créées.")

        quoi = "charger [C00001 plus récent, C00002 plus ancien]"
        n = _charger(module.charger_contrats, connexion, [t["c1_recent"], t["c2_ancien"]], quoi)
        par_client = {ligne[0]: ligne for ligne in _lignes(connexion, "contrats")}
        if par_client["C00002"][3] == "Mensuel":
            raise AssertionError("Une version PLUS ANCIENNE de C00002 (maj 2026-05-01) a écrasé la version en base "
                                 "(maj 2026-06-02) : c'est ce que fait INSERT OR REPLACE. Ajoutez la condition "
                                 "`WHERE excluded.maj > contrats.maj` à votre ON CONFLICT ... DO UPDATE.")
        if par_client["C00001"][3] != "Annuel":
            raise AssertionError("Une version PLUS RÉCENTE de C00001 (maj 2026-02-01) n'a pas remplacé l'ancienne : "
                                 "INSERT OR IGNORE ou DO NOTHING ne mettent jamais à jour. ON CONFLICT ... DO UPDATE.")
        if par_client["C00001"][4] != "Carte bancaire" or par_client["C00001"][10] != "2026-02-01T08:00:00":
            raise AssertionError(f"C00001 n'a été que partiellement mis à jour : {par_client['C00001']!r}. Mettez à "
                                 f"jour TOUTES les colonnes, `maj` comprise.")
        if n != 1:
            raise AssertionError(f"{quoi} renvoie {n}, attendu 1 (seul C00001 a changé) : renvoyez les lignes "
                                 f"RÉELLEMENT créées ou modifiées (`connexion.total_changes` avant et après).")

        quoi = "recharger les mêmes contrats"
        attendu = _lignes(connexion, "contrats")
        n = _charger(module.charger_contrats, connexion, [t["c1_recent"], t["c2"], t["c3"]], quoi)
        if _lignes(connexion, "contrats") != attendu:
            raise AssertionError(f"Après {quoi}, la table a changé : recharger les mêmes lignes ne doit rien changer.")
        if n != 0:
            raise AssertionError(f"{quoi} renvoie {n} au lieu de 0 : rien n'a changé. Renvoyez les lignes RÉELLEMENT "
                                 f"créées ou modifiées (`connexion.total_changes` avant et après), et n'écrasez une "
                                 f"ligne que si la nouvelle est STRICTEMENT plus récente (`>`, pas `>=`).")

        quoi = "charger [C00004 récent, puis la version ANCIENNE de C00004] dans le même appel"
        _charger(module.charger_contrats, connexion, [t["c4_recent"], t["c4_ancien"]], quoi)
        c4 = {ligne[0]: ligne for ligne in _lignes(connexion, "contrats")}.get("C00004")
        if c4 is None or c4[3] != "Annuel":
            raise AssertionError(f"Après {quoi}, C00004 vaut {c4!r} : la version la plus récente doit gagner même "
                                 f"quand les deux arrivent dans le même fichier, dans le désordre.")
        if _charger(module.charger_contrats, connexion, [], "charger_contrats(connexion, [])") != 0:
            raise AssertionError("charger_contrats(connexion, []) doit renvoyer 0.")

        # --- les tickets
        def tk(ticket, client, ouvert, maj):
            return M.Ticket(ticket=ticket, client=client, ouvert=ouvert, maj=maj)
        t1 = tk("T000001", "C00001", datetime(2026, 2, 14, 10, 3), datetime(2026, 2, 14, 10, 3))
        t2 = tk("T000002", "C00002", datetime(2026, 3, 1, 8, 0), datetime(2026, 3, 2, 9, 30))
        connexion.commit()
        quoi = "charger_tickets(connexion, [2 nouveaux tickets])"
        n = _charger(module.charger_tickets, connexion, [t1, t2], quoi)
        if not connexion.in_transaction:
            raise AssertionError("charger_tickets a validé la transaction : pas de commit ici, c'est l'étape 4 qui "
                                 "s'en charge.")
        _verifier_textes(connexion, "tickets", ("ouvert", "maj"), ("ouvert", "maj"))
        attendu = [("T000001", "C00001", "2026-02-14T10:03:00", "2026-02-14T10:03:00"),
                   ("T000002", "C00002", "2026-03-01T08:00:00", "2026-03-02T09:30:00")]
        if _lignes(connexion, "tickets") != attendu or n != 2:
            raise AssertionError(f"Après {quoi} : table tickets {_lignes(connexion, 'tickets')!r} et {n} renvoyé ; "
                                 f"attendu {attendu!r} et 2.")
        quoi = "recharger les mêmes tickets"
        if _charger(module.charger_tickets, connexion, [t1, t2], quoi) != 0 or _lignes(connexion, "tickets") != attendu:
            raise AssertionError(f"{quoi} doit renvoyer 0 et ne rien changer.")
        quoi = "charger [T000001 mis à jour, une version ancienne de T000002]"
        t1_maj = tk("T000001", "C00001", datetime(2026, 2, 14, 10, 3), datetime(2026, 2, 20, 18, 0))
        t2_vieux = tk("T000002", "C00002", datetime(2026, 3, 1, 8, 0), datetime(2026, 3, 1, 8, 0))
        n = _charger(module.charger_tickets, connexion, [t1_maj, t2_vieux], quoi)
        majs = {ligne[0]: ligne[3] for ligne in _lignes(connexion, "tickets")}
        if majs != {"T000001": "2026-02-20T18:00:00", "T000002": "2026-03-02T09:30:00"} or n != 1:
            raise AssertionError(f"Après {quoi} : maj en base {majs}, {n} renvoyé ; attendu T000001 mis à jour, "
                                 f"T000002 inchangé (sa version reçue est plus ancienne), et 1.")
    finally:
        connexion.close()


# ============================================================================ Étape 4 : charger un fichier

_CRM_C = [f"C000{k:02d};01/0{1 + k % 9}/2025;;Mensuel;Prélèvement;Fibre;Non;Non;1;39,90;2026-01-0{1 + k % 9}T08:00:00"
          for k in range(1, 40)]


def _crm(*clients: int, mauvaises: int = 0) -> list[str]:
    return [_ENTETE] + [_CRM_C[k - 1] for k in clients] + [_champ(_CRM_C[0], 9, "N/A")] * mauvaises


def _compter(base: Path, requete: str, *parametres) -> int:
    with contextlib.closing(sqlite3.connect(str(base))) as autre:
        return autre.execute(requete, parametres).fetchone()[0]


def _bilan(module, connexion, chemin: Path, quoi: str):
    bilan = _appel(lambda: module.charger_fichier(connexion, chemin), quoi)
    bilan = _pas_none(bilan, "charger_fichier")
    for attribut in ("fichier", "statut", "lignes", "rejets"):
        if not hasattr(bilan, attribut):
            raise AssertionError(f"charger_fichier doit renvoyer un `Bilan(fichier, statut, lignes, rejets)` "
                                 f"(modeles.py), pas {type(bilan).__name__}.")
    return bilan


def _attendre(bilan, M, statut: str, lignes: int, rejets: int, quoi: str, conseil: str = "") -> None:
    noms = {M.CHARGE: "CHARGE", M.DEJA_CHARGE: "DEJA_CHARGE", M.REFUSE: "REFUSE"}
    obtenu = (bilan.statut, bilan.lignes, len(list(bilan.rejets)))
    if obtenu != (statut, lignes, rejets):
        raise AssertionError(f"{quoi} : statut {noms.get(bilan.statut, repr(bilan.statut))}, {bilan.lignes} ligne(s), "
                             f"{obtenu[2]} rejet(s) ; attendu {noms[statut]}, {lignes} ligne(s), {rejets} rejet(s)."
                             f"{(' ' + conseil) if conseil else ''}")


def verifier_charger_fichier(module: ModuleType) -> None:
    M = _modeles(module)
    with _dossier_temporaire() as dossier:
        base = dossier / "entrepot.db"
        connexion = M.ouvrir_base(base)
        try:
            for nom in ("notes.txt", "clients.csv", "crm_2026-01-05.txt"):
                autre = _ecrire(dossier, nom, ["rien"])
                _leve_value_error(lambda: module.charger_fichier(connexion, autre), f"charger_fichier sur « {nom} »",
                                  "Seuls crm_*.csv et support_*.jsonl sont des exports connus.")

            # 1. Un premier fichier : chargé, validé (commit), noté dans `fichiers`.
            crm = _ecrire(dossier, "crm_2026-01-05.csv", _crm(1, 2, 3))
            quoi = "Le premier chargement de crm_2026-01-05.csv (3 contrats)"
            bilan = _bilan(module, connexion, crm, quoi)
            if bilan.fichier != "crm_2026-01-05.csv":
                raise AssertionError(f"Bilan.fichier vaut {bilan.fichier!r} : attendu le nom sans le dossier, "
                                     f"'crm_2026-01-05.csv'.")
            _attendre(bilan, M, M.CHARGE, 3, 0, quoi)
            if _compter(base, "SELECT count(*) FROM contrats") != 3:
                raise AssertionError("Après le chargement, une AUTRE connexion ne voit pas les 3 contrats : la "
                                     "transaction n'a pas été validée. `with connexion:` fait le commit à la sortie.")
            trace = connexion.execute("SELECT nom, empreinte, charge_le, lignes, rejets FROM fichiers").fetchall()
            if len(trace) != 1 or trace[0][0] != "crm_2026-01-05.csv":
                raise AssertionError(f"La table fichiers contient {trace!r} : attendu une ligne pour "
                                     f"crm_2026-01-05.csv.")
            nom, signature, charge_le, lignes, rejets = trace[0]
            if signature != M.empreinte(crm):
                raise AssertionError(f"fichiers.empreinte vaut {signature!r} : attendu `empreinte(chemin)` "
                                     f"({M.empreinte(crm)[:12]}…).")
            try:
                datetime.fromisoformat(str(charge_le))
            except ValueError:
                raise AssertionError(f"fichiers.charge_le vaut {charge_le!r} : attendu l'heure du chargement en texte "
                                     f"ISO (`datetime.now().isoformat(timespec=\"seconds\")`).") from None
            if (lignes, rejets) != (3, 0):
                raise AssertionError(f"fichiers.lignes et fichiers.rejets valent {lignes} et {rejets}, attendu 3 et 0.")

            # 2. Le même fichier, relancé : déjà chargé, et pas relu.
            relu = []

            def piege(chemin):
                relu.append(chemin)
                raise AssertionError("relu")

            with _remplacer(module, "lire_contrats", piege, "contrats"):
                try:
                    bilan = module.charger_fichier(connexion, crm)
                except AssertionError:
                    bilan = None
            if relu:
                raise AssertionError("Un fichier déjà chargé (même nom, même empreinte) a été relu : cherchez son nom "
                                     "et son empreinte dans `fichiers` AVANT de lire le fichier.")
            _attendre(_pas_none(bilan, "charger_fichier"), M, M.DEJA_CHARGE, 0, 0,
                      "Le second chargement du même crm_2026-01-05.csv")

            # 3. Un export du support.
            support = _ecrire(dossier, "support_2026-01-05.jsonl", [
                _jt("T000001", "C00001", "2026-01-02T10:00:00"), _jt("T000002", "C00002", "2026-01-03T10:00:00"),
                _jt("T000001", "C00001", "2026-01-02T10:00:00", "2026-01-04T10:00:00", statut="résolu")], bom=False)
            _attendre(_bilan(module, connexion, support, "Le chargement d'un export du support"), M, M.CHARGE, 2, 0,
                      "support_2026-01-05.jsonl (2 tickets, dont un mis à jour)")

            # 4. Même nom, contenu corrigé : rechargé, et sa ligne de `fichiers` mise à jour.
            nouveau = _champ(_CRM_C[0], 10, "2026-02-01T08:00:00").replace(";Mensuel;", ";Annuel;")
            _ecrire(dossier, "crm_2026-01-05.csv", _crm(1, 2, 3) + [nouveau, _CRM_C[3]])
            quoi = "Le rechargement de crm_2026-01-05.csv renvoyé corrigé (C00001 modifié, C00004 ajouté)"
            try:
                bilan = _bilan(module, connexion, crm, quoi)
            except AssertionError as exc:
                if "UNIQUE" in str(exc) and "fichiers" in str(exc):
                    raise AssertionError("Le fichier corrigé (même nom, autre empreinte) fait planter l'écriture dans "
                                         "`fichiers` : son nom y est déjà. Mettez sa ligne à jour (ON CONFLICT (nom) DO "
                                         "UPDATE …, ou INSERT OR REPLACE).") from None
                raise
            _attendre(bilan, M, M.CHARGE, 2, 0, quoi, "Même nom mais empreinte différente : on recharge.")
            trace = connexion.execute("SELECT empreinte FROM fichiers WHERE nom = 'crm_2026-01-05.csv'").fetchall()
            if trace != [(M.empreinte(crm),)]:
                raise AssertionError("Après le rechargement du fichier corrigé, `fichiers` doit garder UNE ligne pour ce "
                                     "nom, avec la NOUVELLE empreinte.")

            # 5. Trop de rejets : refusé, rien n'est écrit, pas même son nom.
            avant = (_compter(base, "SELECT count(*) FROM contrats"), _compter(base, "SELECT count(*) FROM fichiers"))
            refuse = _ecrire(dossier, "crm_2026-03-02.csv", _crm(*range(5, 13), mauvaises=2))
            quoi = "crm_2026-03-02.csv (8 bonnes lignes, 2 illisibles : 20 %)"
            _attendre(_bilan(module, connexion, refuse, quoi), M, M.REFUSE, 0, 2, quoi)
            if connexion.in_transaction or (_compter(base, "SELECT count(*) FROM contrats"),
                                            _compter(base, "SELECT count(*) FROM fichiers")) != avant:
                raise AssertionError("Un fichier REFUSÉ a écrit dans la base (ses lignes ou son nom dans `fichiers`) : "
                                     "refusez-le AVANT d'écrire quoi que ce soit.")

            # 6. Exactement 10 % : chargé.
            juste = _ecrire(dossier, "crm_2026-04-01.csv", _crm(*range(13, 22), mauvaises=1))
            quoi = "crm_2026-04-01.csv (9 bonnes lignes, 1 illisible : 10 % pile)"
            _attendre(_bilan(module, connexion, juste, quoi), M, M.CHARGE, 9, 1, quoi,
                      "Le fichier n'est refusé qu'AU-DELÀ de 10 % : `>`, pas `>=`.")

            # 7. Un fichier sans aucune ligne : pas de division par zéro.
            vide = _ecrire(dossier, "crm_2026-05-04.csv", [_ENTETE])
            try:
                bilan = module.charger_fichier(connexion, vide)
            except ZeroDivisionError:
                raise AssertionError("Un export sans aucune ligne fait diviser par zéro : s'il n'y a rien à lire, il n'y "
                                     "a rien à refuser (vérifiez qu'il y a des lignes avant de calculer la part de "
                                     "rejets).") from None
            except (NotImplementedError, AssertionError):
                raise
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(f"Un export sans aucune ligne fait lever {type(exc).__name__}: {exc}") from None
            _attendre(_pas_none(bilan, "charger_fichier"), M, M.CHARGE, 0, 0, "Un export qui n'a que l'en-tête")

            # 8. Une panne au milieu : rien ne doit rester, et le fichier sera repris.
            panne = _ecrire(dossier, "crm_2026-06-01.csv", _crm(22, 23, 24, 25))
            vrai = importlib.import_module(f"{module.__package__}.chargement").charger_contrats

            def disque_plein(cnx, contrats):
                vrai(cnx, list(contrats)[:2])
                raise sqlite3.OperationalError("database or disk is full (panne simulée par le correcteur)")

            avant = (_lignes(connexion, "contrats"), _lignes(connexion, "fichiers"))
            with _remplacer(module, "charger_contrats", disque_plein, "chargement"):
                try:
                    module.charger_fichier(connexion, panne)
                except sqlite3.OperationalError:
                    pass
                except (NotImplementedError, AssertionError):
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise AssertionError(f"Pendant la panne simulée, charger_fichier lève {type(exc).__name__}: {exc}"
                                         f" au lieu de laisser remonter l'erreur d'origine.") from None
                else:
                    raise AssertionError("Une panne (disque plein) au milieu du chargement a été avalée : charger_fichier "
                                         "a terminé normalement. Laissez l'exception remonter (pas de try/except autour "
                                         "du chargement).")
            if connexion.in_transaction:
                connexion.rollback()
                raise AssertionError("Après la panne, la transaction est restée ouverte avec la moitié des lignes : "
                                     "`with connexion:` fait le rollback tout seul quand une exception le traverse.")
            apres = (_lignes(connexion, "contrats"), _lignes(connexion, "fichiers"))
            if apres[1] != avant[1]:
                raise AssertionError("Après une panne au milieu du chargement, crm_2026-06-01.csv est noté dans `fichiers` "
                                     "alors que ses lignes ne sont pas toutes là : la nuit suivante, il passerait pour "
                                     "« déjà chargé ». Notez le fichier DANS la même transaction que ses lignes.")
            if apres[0] != avant[0]:
                raise AssertionError("Après une panne au milieu du chargement, la moitié des lignes de crm_2026-06-01.csv "
                                     "sont restées dans la base. Une seule transaction (`with connexion:`) autour des "
                                     "lignes ET de la ligne de `fichiers` : tout ou rien.")
            _attendre(_bilan(module, connexion, panne, "La reprise après la panne"), M, M.CHARGE, 4, 0,
                      "La reprise de crm_2026-06-01.csv après la panne")

            # 9. Un mauvais en-tête : l'erreur de lecture remonte.
            mauvais = _ecrire(dossier, "crm_2026-07-01.csv", [_ENTETE.replace(";", ","), _CRM_C[30]])
            _leve_value_error(lambda: module.charger_fichier(connexion, mauvais),
                              "charger_fichier sur un export du CRM à l'en-tête inattendu",
                              "Laissez remonter la ValueError de lire_contrats.")
        finally:
            connexion.close()


# ============================================================================ Étape 5 : la table à date

_PHOTO = date(2026, 6, 30)

#: (client, debut, fin, mensualite) — tous avec maj = 2026-12-31 : la base connaît les départs jusque-là.
_SCENARIO_CONTRATS = (
    ("C00001", "2024-01-15", None, 50.0),         # aucun ticket
    ("C00002", "2025-12-30", None, 39.99),        # un ticket le jour même, l'après-midi
    ("C00003", "2020-01-01", "2026-06-30", 50.0),  # parti le jour de la photo
    ("C00004", "2026-07-01", None, 50.0),         # arrivé le lendemain
    ("C00005", "2023-05-10", "2026-09-28", 50.0),  # part le dernier jour de l'horizon ; un ticket le lendemain
    ("C00006", "2023-05-10", "2026-09-29", 50.0),  # part le jour d'après l'horizon ; un ticket à 365 jours
    ("C00007", "2026-06-20", None, 50.0),         # dix jours d'ancienneté
    ("C00008", "2019-03-01", "2025-11-02", 50.0),  # parti depuis longtemps
    ("C00009", "2022-02-10", None, 50.0),         # un ticket à 364 jours, un vieux
    ("C00010", "2026-01-31", None, 50.0),         # 4 mois complets, mais 150 jours
    ("C00011", "2021-08-30", "2026-07-01", 50.0),  # part le lendemain de la photo
)
_SCENARIO_TICKETS = (
    ("T000001", "C00002", "2026-06-30T15:00:00"),
    ("T000002", "C00005", "2026-07-01T09:00:00"),
    ("T000003", "C00006", "2025-06-30T10:00:00"),
    ("T000004", "C00009", "2025-07-01T10:00:00"),
    ("T000005", "C00009", "2024-02-01T10:00:00"),
    ("T000006", "C99999", "2026-06-01T10:00:00"),
    ("T000007", "C00011", "2026-06-30T00:00:00"),
)


def _base_scenario(ouvrir, contrats=_SCENARIO_CONTRATS, tickets=_SCENARIO_TICKETS,
                   maj: str = "2026-12-31T00:00:00") -> sqlite3.Connection:
    connexion = ouvrir(":memory:")
    connexion.executemany("INSERT INTO contrats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
        (client, debut, fin, "Mensuel", "Prélèvement", "Fibre", "Non", "Non", 1, mensualite, maj)
        for client, debut, fin, mensualite in contrats])
    connexion.executemany("INSERT INTO tickets VALUES (?, ?, ?, ?)",
                          [(ticket, client, ouvert, ouvert) for ticket, client, ouvert in tickets])
    # Un client aux valeurs toutes différentes, pour vérifier que chaque colonne va au bon endroit.
    connexion.execute("UPDATE contrats SET contrat = 'Annuel', paiement = 'Chèque', internet = 'ADSL', "
                      "assistance = 'Oui', senior = 'Oui', produits = 3 WHERE client = 'C00009'")
    connexion.commit()
    return connexion


def _variante(jointure: str = "LEFT JOIN", compte: str = "COUNT(t.ticket)", ouverture: str = "date(t.ouvert)",
              borne_basse: str | None = ">", borne_haute: bool = True, fin: str | None = "c.fin > :photo",
              arrives: bool = True, horizon_actif: bool = True, horizon_inclus: bool = True, maturite: bool = True,
              plancher: int = 1):
    """Une version de `construire_table`, juste avec les réglages par défaut, boguée sinon."""
    def construire_table(connexion, photo, horizon=HORIZON_JOURS):
        fin_horizon = photo + timedelta(days=horizon)
        connue = connexion.execute("SELECT max(maj) FROM contrats").fetchone()[0]
        if connue is None:
            raise ValueError("base vide")
        if maturite and fin_horizon > datetime.fromisoformat(connue).date():
            raise ValueError("photo trop récente")
        jointures = ["t.client = c.client"]
        if borne_basse:
            jointures.append(f"{ouverture} {borne_basse} :depuis")
        if borne_haute:
            jointures.append(f"{ouverture} <= :photo")
        filtres = (["c.debut <= :photo"] if arrives else []) + ([f"(c.fin IS NULL OR {fin})"] if fin else [])
        requete = (f"SELECT c.client, c.debut, c.fin, c.contrat, c.paiement, c.internet, c.assistance, c.senior, "
                   f"c.produits, c.mensualite, {compte} FROM contrats AS c {jointure} tickets AS t "
                   f"ON {' AND '.join(jointures)} {'WHERE ' + ' AND '.join(filtres) if filtres else ''} "
                   f"GROUP BY c.client ORDER BY c.client")
        lignes = []
        parametres = {"photo": photo.isoformat(), "depuis": (photo - timedelta(days=FENETRE_JOURS)).isoformat()}
        for client, debut, fin_, contrat, paiement, internet, assistance, senior, produits, mensualite, n in \
                connexion.execute(requete, parametres):
            anciennete = max(plancher, mois_entre(date.fromisoformat(debut), photo))
            part = fin_ is not None and (not horizon_actif or (date.fromisoformat(fin_) <= fin_horizon if horizon_inclus
                                                               else date.fromisoformat(fin_) < fin_horizon))
            lignes.append((client, anciennete, mensualite, round(mensualite * anciennete, 2), n, produits, contrat,
                           paiement, internet, assistance, senior, int(part)))
        return pd.DataFrame(lignes, columns=list(COLONNES_TABLE)).astype(
            {"tenure_months": "int64", "support_calls": "int64", "num_products": "int64", "churn": "int64"})
    return construire_table


#: (version boguée, ce qu'elle fait, le conseil)
_MUTANTS_TABLE = (
    (_variante(jointure="JOIN"), "oublie les clients qui n'ont aucun ticket dans la fenêtre (un JOIN au lieu d'un "
     "LEFT JOIN, ou les conditions sur les tickets placées dans le WHERE au lieu du ON)",
     "Un LEFT JOIN garde chaque client, même sans ticket, à condition que les conditions sur t.ouvert soient dans le "
     "ON : dans le WHERE, elles éliminent les lignes de NULL."),
    (_variante(compte="COUNT(*)"), "compte 1 ticket aux clients qui n'en ont aucun (COUNT(*))",
     "COUNT(*) compte la ligne de NULL du LEFT JOIN : COUNT(t.ticket) ne compte que les vrais tickets."),
    (_variante(ouverture="t.ouvert"), "compare les dates en TEXTE : le ticket ouvert le jour de la photo à 15 h est "
     "oublié, et celui d'il y a 365 jours à 10 h est compté",
     "\"2026-06-30T15:00:00\" <= \"2026-06-30\" est faux en texte : comparez date(t.ouvert)."),
    (_variante(borne_haute=False), "compte les tickets ouverts APRÈS la photo (une fuite du futur)",
     "Un ticket ouvert le lendemain de la photo n'existait pas encore ce jour-là : date(t.ouvert) <= photo."),
    (_variante(borne_basse=None), "compte tous les tickets, même ceux d'il y a deux ans",
     "Seuls les tickets des 365 derniers jours comptent : date(t.ouvert) > photo − 365 jours."),
    (_variante(borne_basse=">="), "compte le ticket ouvert 365 jours pile avant la photo",
     "La fenêtre est photo − 365 jours < jour d'ouverture : `>`, pas `>=`."),
    (_variante(fin="c.fin >= :photo"), "garde le client parti le jour même de la photo",
     "Parti le jour de la photo, il n'est plus client ce jour-là : fin > photo."),
    (_variante(fin=None), "garde les clients partis bien avant la photo",
     "Seuls les clients encore là le jour de la photo : fin IS NULL OR fin > photo."),
    (_variante(arrives=False), "garde les clients arrivés après la photo",
     "Un client arrivé après la photo n'existait pas encore : debut <= photo."),
    (_variante(horizon_actif=False), "compte comme partis les clients qui partent APRÈS l'horizon",
     "churn = 1 seulement si fin <= photo + horizon jours."),
    (_variante(horizon_inclus=False), "oublie le client qui part le dernier jour de l'horizon",
     "Le dernier jour de l'horizon est compris : fin <= photo + horizon, pas `<`."),
    (_variante(maturite=False), "accepte une photo trop récente, dont on ne connaît pas encore tous les départs",
     "Levez ValueError si le jour de la dernière `maj` est avant photo + horizon jours."),
    (_variante(plancher=0), "donne 0 mois d'ancienneté à un client arrivé il y a dix jours",
     "tenure_months = max(1, mois_entre(debut, photo))."),
)


def _resultat(fonction, connexion, photo, horizon=HORIZON_JOURS):
    """Le résultat de `fonction`, ou l'exception levée (pour comparer une version à une autre)."""
    try:
        table = fonction(connexion, photo, horizon)
    except (NotImplementedError, AssertionError):
        raise
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__
    if not isinstance(table, pd.DataFrame):
        return repr(table)
    return table.reset_index(drop=True).to_dict("list")


def _comparer_tables(obtenu: pd.DataFrame, attendu: pd.DataFrame, quoi: str) -> None:
    if list(obtenu.columns) != list(COLONNES_TABLE):
        raise AssertionError(f"{quoi} : colonnes {list(obtenu.columns)}, attendu exactement {list(COLONNES_TABLE)} "
                             f"(COLONNES_TABLE), dans cet ordre.")
    clients_o, clients_a = list(obtenu["customer_id"]), list(attendu["customer_id"])
    if len(set(clients_o)) != len(clients_o):
        raise AssertionError(f"{quoi} : certains clients apparaissent plusieurs fois. Un GROUP BY c.client regroupe "
                             f"leurs tickets en une ligne.")
    if set(clients_o) != set(clients_a):
        en_trop, manquants = sorted(set(clients_o) - set(clients_a)), sorted(set(clients_a) - set(clients_o))
        raise AssertionError(f"{quoi} : {len(clients_o)} clients au lieu de {len(clients_a)}"
                             f"{'. En trop : ' + ', '.join(en_trop[:5]) if en_trop else ''}"
                             f"{'. Manquants : ' + ', '.join(manquants[:5]) if manquants else ''}.")
    if clients_o != clients_a:
        raise AssertionError(f"{quoi} : les clients doivent être triés par customer_id.")
    if not isinstance(obtenu.index, pd.RangeIndex) or obtenu.index.start != 0:
        raise AssertionError(f"{quoi} : l'index doit être 0, 1, 2… (`reset_index(drop=True)`).")
    for colonne in ("tenure_months", "support_calls", "num_products", "churn"):
        if not pd.api.types.is_integer_dtype(obtenu[colonne]):
            raise AssertionError(f"{quoi} : la colonne {colonne} est de type {obtenu[colonne].dtype}, attendu des "
                                 f"entiers.")
    for colonne in COLONNES_TABLE[1:]:
        for client, o, a in zip(clients_a, obtenu[colonne], attendu[colonne]):
            if (abs(float(o) - float(a)) > 1e-6) if colonne in ("monthly_charges", "total_charges") else o != a:
                raise AssertionError(f"{quoi} : {client}, {colonne} vaut {o!r}, attendu {a!r}.")


def verifier_construire_table(module: ModuleType) -> None:
    M = _modeles(module)
    construire = module.construire_table

    with contextlib.closing(M.ouvrir_base(":memory:")) as vide:
        _leve_value_error(lambda: construire(vide, _PHOTO), "construire_table sur une base sans contrat",
                          "Rien à exporter : dites-le.")

    connexion = _base_scenario(M.ouvrir_base)
    try:
        avant = connexion.total_changes
        table = _appel(lambda: construire(connexion, _PHOTO), "construire_table(base, 30/06/2026)")
        if not isinstance(_pas_none(table, "construire_table"), pd.DataFrame):
            raise AssertionError(f"construire_table doit renvoyer un DataFrame, pas {type(table).__name__}.")
        if connexion.total_changes != avant:
            raise AssertionError("construire_table a modifié la base : elle ne doit que la lire.")

        reference = _resultat(ref_construire_table, connexion, _PHOTO)
        obtenu = _resultat(construire, connexion, _PHOTO)
        if obtenu != reference:
            for mutant, quoi, conseil in _MUTANTS_TABLE:
                if _resultat(mutant, connexion, _PHOTO) == obtenu:
                    raise AssertionError(f"Votre construire_table {quoi}. {conseil}")
            _comparer_tables(table, ref_construire_table(connexion, _PHOTO), "Le 30/06/2026")

        # L'horizon est un paramètre : il doit servir, pour les départs ET pour la maturité.
        table = _appel(lambda: construire(connexion, _PHOTO, 30), "construire_table(base, 30/06/2026, horizon=30)")
        _comparer_tables(table, ref_construire_table(connexion, _PHOTO, 30), "Avec horizon=30")
        _appel(lambda: construire(connexion, date(2026, 10, 2)),
               "construire_table au 02/10/2026 (02/10 + 90 jours = 31/12, le jour de la dernière maj : c'est connu)")
        for photo, horizon in ((date(2026, 10, 3), HORIZON_JOURS), (date(2026, 12, 2), 30)):
            _leve_value_error(lambda: construire(connexion, photo, horizon),
                              f"construire_table au {photo:%d/%m/%Y} avec horizon={horizon} (la base ne connaît les "
                              f"départs que jusqu'au 31/12/2026)",
                              "On ne connaît pas encore tous les départs de l'horizon : la photo est trop récente.")
        _appel(lambda: construire(connexion, date(2026, 12, 1), 30), "construire_table au 01/12/2026 avec horizon=30")
    finally:
        connexion.close()

    # Pour finir, les vrais exports d'exemple : votre table doit être exactement celle de la référence.
    with _dossier_temporaire() as dossier:
        base = _base_exemple(dossier)
        with contextlib.closing(sqlite3.connect(str(base))) as cnx:
            table = _appel(lambda: construire(cnx, PHOTO_EXEMPLE), "construire_table sur les exports d'exemple")
            _comparer_tables(table, ref_construire_table(cnx, PHOTO_EXEMPLE), "Sur les exports d'exemple, au 30/06/2026")


def _base_exemple(dossier: Path) -> Path:
    """Charge les exports d'exemple avec le CORRIGÉ dans `dossier/reference.db` (indépendant de vos étapes 1 à 4)."""
    base = dossier / "reference.db"
    with contextlib.closing(ref_ouvrir_base(base)) as connexion:
        for chemin in sorted(generer_exports(dossier / "exports")):
            ref_charger_fichier(connexion, chemin)
    return base


# ============================================================================ Étape 6 : le programme


def _commande(argv: list[str]) -> str:
    return "python -m projets.entrepot " + " ".join(argv)


def _contenu(base: Path) -> tuple:
    with contextlib.closing(sqlite3.connect(str(base))) as connexion:
        return (_lignes(connexion, "contrats"), _lignes(connexion, "tickets"),
                [ligne[:2] for ligne in _lignes(connexion, "fichiers")])


def _executer(main, argv: list[str]) -> tuple[int, str, str]:
    code, sortie, erreurs = _lancer(main, argv)
    if "Traceback" in sortie + erreurs:
        raise AssertionError(f"`{_commande(argv)}` affiche un traceback : une erreur attendue s'affiche en une phrase.")
    return code, sortie, erreurs


def verifier_cli(module: ModuleType) -> None:
    main = module.main
    with _dossier_temporaire() as dossier:
        exports = dossier / "exports"
        fichiers = sorted(p.name for p in generer_exports(exports))
        (exports / "notes.txt").write_text("à lire avant de lancer\n", encoding="utf-8")
        (exports / "crm_brouillon.xlsx").write_bytes(b"PK\x03\x04")

        # La référence : les mêmes exports chargés par le corrigé.
        reference = dossier / "reference.db"
        with contextlib.closing(ref_ouvrir_base(reference)) as connexion:
            rejets_attendus = []
            for nom in fichiers:
                rejets_attendus += [(r.fichier, r.ligne) for r in ref_charger_fichier(connexion, exports / nom).rejets]
        attendu = _contenu(reference)

        # 1. Premier chargement.
        base, rejets = dossier / "entrepot.db", dossier / "rejets.csv"
        argv = ["charger", "--dossier", str(exports), "--base", str(base), "--rejets", str(rejets)]
        code, sortie, erreurs = _executer(main, argv)
        if code != 0:
            raise AssertionError(f"`{_commande(argv)}` renvoie {code} (stderr : {erreurs.strip()[:200]!r}) : tous les "
                                 f"exports d'exemple se chargent, attendu 0.")
        if not base.is_file():
            raise AssertionError(f"`{_commande(argv)}` n'a pas créé la base {base.name}.")
        obtenu = _contenu(base)
        if obtenu != attendu:
            noms = ("contrats", "tickets", "fichiers")
            quoi = next(n for n, o, a in zip(noms, obtenu, attendu) if o != a)
            raise AssertionError(f"Après `{_commande(argv)}`, la table {quoi} ne contient pas ce qu'elle devrait "
                                 f"({len(obtenu[noms.index(quoi)])} lignes au lieu de {len(attendu[noms.index(quoi)])}). "
                                 f"Chargez chaque crm_*.csv et support_*.jsonl du dossier, triés par nom, avec "
                                 f"charger_fichier ; les autres fichiers sont laissés de côté.")
        absents = [nom for nom in fichiers if nom not in sortie]
        if absents:
            raise AssertionError(f"La sortie de `charger` ne cite pas {absents[0]} : une ligne par fichier, avec son nom.")
        if "chargé" not in sortie:
            raise AssertionError("La sortie de `charger` doit dire, pour chaque fichier, « chargé », « déjà chargé » ou "
                                 "« refusé ».")
        if not rejets.is_file():
            raise AssertionError(f"Les exports d'exemple ont {len(rejets_attendus)} lignes écartées : écrivez-les dans le "
                                 f"fichier --rejets ({rejets.name}).")
        with rejets.open(encoding="utf-8", newline="") as f:
            lecteur = csv.DictReader(f)
            if lecteur.fieldnames != ["fichier", "ligne", "raison"]:
                raise AssertionError(f"{rejets.name} a l'en-tête {lecteur.fieldnames!r}, attendu fichier,ligne,raison "
                                     f"(module csv, séparateur virgule).")
            lus = [(r["fichier"], int(r["ligne"])) for r in lecteur]
        if sorted(lus) != sorted(rejets_attendus):
            raise AssertionError(f"{rejets.name} contient {len(lus)} rejet(s), attendu {len(rejets_attendus)} : "
                                 f"{sorted(rejets_attendus)[:3]}…")

        # 2. On relance : tout est déjà chargé, rien ne change, pas de fichier de rejets.
        rejets.unlink()
        code, sortie, erreurs = _executer(main, argv)
        if code != 0 or _contenu(base) != attendu:
            raise AssertionError(f"Relancer `{_commande(argv)}` sur le même dossier doit renvoyer 0 et ne rien changer "
                                 f"(code {code}).")
        lignes_fichiers = [ligne for ligne in sortie.splitlines() if any(nom in ligne for nom in fichiers)]
        if len(lignes_fichiers) < len(fichiers) or not all("déjà" in ligne for ligne in lignes_fichiers):
            raise AssertionError("En relançant `charger`, chaque fichier doit être annoncé « déjà chargé ».")
        if rejets.exists():
            raise AssertionError(f"Aucune ligne n'a été écartée lors du second lancement : n'écrivez pas {rejets.name}.")

        # 3. Un fichier trop abîmé et un fichier à l'en-tête inattendu : refusés, code 1, base inchangée.
        _ecrire(exports, "crm_2026-10-15.csv", _crm(1, 2, 3, mauvaises=2))
        _ecrire(exports, "crm_2026-10-20.csv", [_ENTETE.replace(";", ","), _CRM_C[0]])
        code, sortie, erreurs = _executer(main, argv)
        if code != 1:
            raise AssertionError(f"Deux fichiers sont refusés (crm_2026-10-15.csv : 40 % de rejets ; crm_2026-10-20.csv : "
                                 f"mauvais en-tête) : `charger` doit renvoyer 1, il renvoie {code}.")
        for nom in ("crm_2026-10-15.csv", "crm_2026-10-20.csv"):
            if not any(nom in ligne and "refus" in ligne for ligne in (sortie + erreurs).splitlines()):
                raise AssertionError(f"La sortie de `charger` doit dire que {nom} est « refusé ».")
        if _contenu(base) != attendu:
            raise AssertionError("Des fichiers refusés ont modifié la base.")
        if not rejets.is_file():
            raise AssertionError(f"Les 2 lignes illisibles de crm_2026-10-15.csv (refusé) doivent être écrites dans "
                                 f"{rejets.name} : c'est là que l'équipe du CRM verra quoi corriger.")
        (exports / "crm_2026-10-15.csv").unlink()
        (exports / "crm_2026-10-20.csv").unlink()

        # 4. Un dossier introuvable : pas de base vide créée par erreur.
        nouvelle = dossier / "jamais.db"
        argv_absent = ["charger", "--dossier", str(dossier / "absent"), "--base", str(nouvelle)]
        code, sortie, erreurs = _executer(main, argv_absent)
        if code != 1 or not erreurs.strip():
            raise AssertionError(f"`{_commande(argv_absent)}` : un dossier introuvable s'annonce sur stderr avec le "
                                 f"code 1 (obtenu {code}).")
        if nouvelle.exists():
            raise AssertionError("Avec un dossier introuvable, `charger` a créé une base vide : vérifiez le dossier "
                                 "AVANT ouvrir_base.")

        # 5. Exporter la table du projet 1.
        sortie_csv = dossier / "sortie" / "customers.csv"
        argv = ["exporter", "--base", str(base), "--photo", "2026-06-30", "--sortie", str(sortie_csv)]
        code, sortie, erreurs = _executer(main, argv)
        if code != 0:
            raise AssertionError(f"`{_commande(argv)}` renvoie {code} (stderr : {erreurs.strip()[:200]!r}), attendu 0.")
        if not sortie_csv.is_file():
            raise AssertionError(f"`{_commande(argv)}` n'a pas écrit {sortie_csv} (créez son dossier s'il n'existe pas : "
                                 f"`Path(...).parent.mkdir(parents=True, exist_ok=True)`).")
        lue = pd.read_csv(sortie_csv)
        with contextlib.closing(sqlite3.connect(str(base))) as connexion:
            reference_table = ref_construire_table(connexion, PHOTO_EXEMPLE)
        if list(lue.columns) == ["Unnamed: 0", *COLONNES_TABLE]:
            raise AssertionError("Le CSV exporté commence par une colonne d'index : `to_csv(..., index=False)`.")
        _comparer_tables(lue, reference_table, f"Le fichier écrit par `{_commande(argv)}`")
        if str(len(reference_table)) not in sortie:
            raise AssertionError(f"La sortie de `exporter` doit dire combien de clients ont été exportés "
                                 f"({len(reference_table)}).")

        # 6. Les erreurs de `exporter`.
        for argv, quoi in (
            (["exporter", "--base", str(base), "--photo", "2026-07-04", "--sortie", str(dossier / "trop.csv")],
             "une photo trop récente (2026-07-04 + 90 jours > dernière maj connue)"),
            (["exporter", "--base", str(dossier / "absente.db"), "--photo", "2026-06-30", "--sortie",
              str(dossier / "trop.csv")], "une base introuvable"),
            (["exporter", "--base", str(base), "--photo", "30/06/2026", "--sortie", str(dossier / "trop.csv")],
             "une photo au format JJ/MM/AAAA"),
        ):
            code, sortie, erreurs = _executer(main, argv)
            if code == 0 or not erreurs.strip():
                raise AssertionError(f"`{_commande(argv)}` ({quoi}) : attendu un message sur stderr et un code non nul, "
                                     f"obtenu le code {code}.")
            if (dossier / "trop.csv").exists():
                raise AssertionError(f"`{_commande(argv)}` ({quoi}) a quand même écrit le fichier de sortie.")
        if (dossier / "absente.db").exists():
            raise AssertionError("`exporter` sur une base introuvable a créé une base vide : vérifiez qu'elle existe "
                                 "AVANT ouvrir_base.")


# ============================================================================ Étape 7 : tester construire_table


def verifier_tester_construire_table(module: ModuleType) -> None:
    tester = module.tester_construire_table
    try:
        tester(ref_construire_table)
    except NotImplementedError:
        raise
    except AssertionError as exc:
        raise AssertionError(f"Votre test ÉCHOUE sur la version JUSTE de construire_table "
                             f"({str(exc) or 'assert sans message'}). Relisez les règles de l'étape 5 et recalculez vos "
                             f"cas à la main (mois_entre, 365 jours, photo + horizon).") from None
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"Votre test plante sur la version JUSTE : {type(exc).__name__}: {exc}. Fabriquez vos "
                             f"données avec `_base([_contrat(...)], [_ticket(...)])`.") from None

    survivants = []
    for mutant, quoi, conseil in _MUTANTS_TABLE:
        try:
            tester(mutant)
        except Exception:  # noqa: BLE001 — n'importe quelle erreur : le test a vu le bug
            continue
        survivants.append((quoi, conseil))
    if survivants:
        if len(survivants) == len(_MUTANTS_TABLE):
            raise AssertionError("Votre test passe sur TOUTES les versions boguées : il ne vérifie rien. Écrivez des "
                                 "`assert` sur les clients présents, leurs support_calls, leur churn…")
        quoi, conseil = survivants[0]
        raise AssertionError(f"{len(_MUTANTS_TABLE) - len(survivants)}/{len(_MUTANTS_TABLE)} bugs attrapés. Votre test "
                             f"laisse passer une version qui {quoi}. Ajoutez un cas où cette version et la bonne donnent "
                             f"des tables différentes.")


# ============================================================================ Le projet

PROJET = Projet(
    numero=6,
    code="entrepot",
    titre="L'entrepôt des clients",
    paquet_eleve="projets.entrepot",
    paquet_corrige="projets.entrepot_corrige",
    commande="python -m projets.entrepot charger --dossier projets/exemple_exports --base entrepot.db",
    etapes=(
        Etape(1, "Lire l'export du CRM", "🟢", "contrats", ("lire_contrats",), verifier_lire_contrats),
        Etape(2, "Lire l'export du support", "🟡", "tickets", ("lire_tickets",), verifier_lire_tickets),
        Etape(3, "Ranger dans la base : la version la plus récente gagne", "🟡", "chargement",
              ("charger_contrats", "charger_tickets"), verifier_charger),
        Etape(4, "Charger un fichier une fois, en entier ou pas du tout", "🔴", "fichier", ("charger_fichier",),
              verifier_charger_fichier),
        Etape(5, "La table du projet 1, à la date de la photo", "🔴", "table", ("construire_table",),
              verifier_construire_table),
        Etape(6, "Le programme de la nuit : charger et exporter", "🟡", "cli", ("main",), verifier_cli),
        Etape(7, "Écrire le test qui protège construire_table", "🔴", "mes_tests", ("tester_construire_table",),
              verifier_tester_construire_table),
    ),
)

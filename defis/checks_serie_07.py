"""Vérifications automatiques de la série 07 (écrire du code réutilisable).

Les messages visent les pièges qui coûtent le plus cher une fois le code en service : compter
30 jours par mois, chercher dans une liste au lieu d'un set, lire un CSV Excel avec les réglages
anglais, tout charger en mémoire « pour faire simple », partager un attribut entre tous les
objets d'une classe, oublier qu'un résultat peut valoir 0 ou None.
"""

from __future__ import annotations

import calendar
import itertools
import math
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .verif import Defi, Serie


def _pas_none(resultat, quoi: str):
    if resultat is None:
        raise AssertionError(f"{quoi} : la fonction a retourné None. Avez-vous oublié `return` ?")
    return resultat


# ============================================================================ Défi 44 : dates

_ANCIENNETES = [
    # (début, fin, attendu)
    ("2024-01-15", "2024-03-14", 1),
    ("2024-01-15", "2024-03-15", 2),
    (date(2024, 1, 15), date(2024, 1, 15), 0),
    (date(2024, 5, 1), "2024-07-01", 2),
    ("2023-11-20", "2024-02-19", 2),
    ("2023-11-20", "2024-02-20", 3),
    ("2020-06-10", "2024-06-09", 47),
    (" 2024-01-15 ", "2024-02-15\n", 1),
    ("2024-01-31", "2024-02-29", 1),
    ("2023-01-31", "2023-02-28", 1),
    ("2024-01-31", "2024-02-28", 0),
    ("2024-03-31", "2024-04-30", 1),
    ("2024-03-31", "2024-04-29", 0),
    ("2024-02-29", "2025-02-28", 12),
    ("2024-01-30", "2024-03-01", 1),
]


def _jour(valeur) -> date:
    return valeur if isinstance(valeur, date) else date.fromisoformat(valeur.strip())


def _compte_en_jours(f) -> bool:
    """La fonction divise-t-elle un nombre de jours par ~30 ? On le voit sur des cas qui trahissent l'approximation."""
    sondes = [(date(2024, 1, 1), date(2024, 1, 31), 0), (date(2024, 2, 1), date(2024, 3, 1), 1),
              (date(2023, 1, 1), date(2023, 12, 31), 11)]
    for debut, fin, juste in sondes:
        jours = (fin - debut).days
        estimations = {jours // 30, round(jours / 30), int(jours / 30.44), round(jours / 30.44), jours // 31}
        try:
            obtenu = f(debut, fin)
        except Exception:  # noqa: BLE001 — simple sonde : le vrai diagnostic est ailleurs
            return False
        if obtenu != juste and obtenu in estimations:
            return True
    return False


def _diagnostic_anciennete(f, obtenu, attendu: int, debut: date, fin: date, appel: str) -> str:
    calendaires = (fin.year - debut.year) * 12 + (fin.month - debut.month)
    fin_de_mois = fin.day == calendar.monthrange(fin.year, fin.month)[1]
    if _compte_en_jours(f):
        return (f"{appel} renvoie {obtenu} au lieu de {attendu}. Diviser un nombre de jours par 30 (ou 30,44) "
                "donne une approximation : un mois a 28, 29, 30 ou 31 jours (du 01/02 au 01/03, il y a 1 mois "
                "mais 29 jours). Comparez les années, les mois et les jours des deux dates.")
    if obtenu == calendaires - 1 and fin_de_mois and fin.day < debut.day:
        return (f"{appel} renvoie {obtenu} au lieu de {attendu}. Cas de la FIN DE MOIS : le {fin:%d/%m} est le "
                f"dernier jour du mois, il ne peut pas y avoir de « {debut.day} ». Un abonnement du {debut:%d/%m} "
                "est donc renouvelé ce jour-là : le mois est complet. calendar.monthrange(annee, mois)[1] donne "
                "le nombre de jours d'un mois.")
    if obtenu == calendaires and fin.day < debut.day:
        return (f"{appel} renvoie {obtenu} au lieu de {attendu} : vous comptez les mois du CALENDRIER, pas les "
                f"mois COMPLETS. Du {debut:%d/%m} au {fin:%d/%m}, le dernier mois n'est pas terminé (le "
                f"{debut.day} n'est pas encore atteint) : retirez 1 dans ce cas.")
    if obtenu == fin.month - debut.month - (fin.day < debut.day) and fin.year != debut.year:
        return (f"{appel} renvoie {obtenu} au lieu de {attendu} : les années comptent aussi. Un mois = "
                "(fin.year - debut.year) * 12 + (fin.month - debut.month), puis la correction du jour.")
    return f"{appel} renvoie {obtenu!r} au lieu de {attendu}."


def verifier_anciennete_en_mois(f) -> None:
    for debut, fin, attendu in _ANCIENNETES:
        appel = f"anciennete_en_mois({debut!r}, {fin!r})"
        try:
            obtenu = f(debut, fin)
        except NotImplementedError:
            raise
        except (TypeError, AttributeError) as exc:
            raise AssertionError(
                f"{appel} plante ({type(exc).__name__}: {exc}). Les dates arrivent soit en objet `date`, soit en "
                "texte (une colonne de CSV) : convertissez le texte avec date.fromisoformat(texte.strip()) et "
                "laissez les objets date tels quels (isinstance(valeur, date))."
            ) from None
        except ValueError as exc:
            indice = ""
            if any(isinstance(v, str) and v != v.strip() for v in (debut, fin)):
                indice = (" Le texte est entouré d'espaces (ou d'un retour à la ligne), fréquent quand on copie depuis "
                          "un tableur : nettoyez-le avec .strip() avant date.fromisoformat.")
            raise AssertionError(f"{appel} lève ValueError({exc}) alors que les dates sont valides.{indice}") from None
        _pas_none(obtenu, appel)
        if isinstance(obtenu, bool) or not isinstance(obtenu, (int, np.integer)):
            if isinstance(obtenu, float) and obtenu == attendu:
                raise AssertionError(f"{appel} renvoie {obtenu!r} : un nombre de mois complets est un ENTIER (int).")
            raise AssertionError(
                f"{appel} : attendu un int ({attendu}), obtenu {type(obtenu).__name__} ({obtenu!r})."
            )
        if obtenu != attendu:
            raise AssertionError(_diagnostic_anciennete(f, obtenu, attendu, _jour(debut), _jour(fin), appel))

    for debut, fin, pourquoi in [
        ("2024-03-01", "2024-02-01", "la fin est AVANT le début"),
        ("15/01/2024", "2024-03-01", "le format n'est pas AAAA-MM-JJ"),
        ("2024-02-30", "2024-03-01", "le 30 février n'existe pas"),
    ]:
        appel = f"anciennete_en_mois({debut!r}, {fin!r})"
        try:
            obtenu = f(debut, fin)
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{appel} lève {type(exc).__name__} : attendu une ValueError ({pourquoi}).") from None
        if debut == "2024-03-01":
            raise AssertionError(
                f"{appel} renvoie {obtenu!r}. Une fin avant le début, c'est une donnée incohérente (dates "
                "inversées dans le CRM ?) : levez une ValueError pour qu'elle se VOIE, au lieu de renvoyer "
                "une ancienneté négative ou nulle qui partirait dans le modèle."
            )
        if debut == "15/01/2024":
            raise AssertionError(
                f"{appel} renvoie {obtenu!r} : ne devinez pas le format d'une date. Le 03/04/2024, c'est le "
                "3 avril en France et le 4 mars aux États-Unis. Acceptez seulement AAAA-MM-JJ (ISO) et "
                "laissez date.fromisoformat lever sa ValueError."
            )
        raise AssertionError(f"{appel} renvoie {obtenu!r} : attendu une ValueError ({pourquoi}).")


# ============================================================================ Défi 45 : complexité


class _ListeEspion(list):
    """Une liste qui compte les recherches (`in`, .index, .count) et les parcours (for)."""

    def __init__(self, *args):
        super().__init__(*args)
        self.recherches = 0
        self.parcours = 0

    def __contains__(self, x):
        self.recherches += 1
        return super().__contains__(x)

    def index(self, *args):
        self.recherches += 1
        return super().index(*args)

    def count(self, x):
        self.recherches += 1
        return super().count(x)

    def __iter__(self):
        self.parcours += 1
        return super().__iter__()


_A_RISQUE = [
    # (mois dernier, ce mois, attendu, description)
    (["C1", "C2", "C3", "C4"], ["C9", "C3", "C1", "C7"], ["C3", "C1"], "l'exemple de la docstring"),
    (["C1", "C2"], ["C2", "C1", "C2", "C1"], ["C2", "C1"], "un client listé deux fois ce mois-ci"),
    (["C1", "C1", "C2"], ["C1"], ["C1"], "un client listé deux fois le mois dernier"),
    (["C5", "C6"], ["C7", "C8"], [], "aucun client en commun"),
    ([], ["C1"], [], "le mois dernier vide"),
    (["C1"], [], [], "ce mois-ci vide"),
]


def verifier_clients_toujours_a_risque(f) -> None:
    for avant, maintenant, attendu, cas in _A_RISQUE:
        a, b = _ListeEspion(avant), _ListeEspion(maintenant)
        obtenu = _pas_none(f(a, b), "clients_toujours_a_risque")
        assert list(a) == avant and list(b) == maintenant, (
            f"Sur {cas}, votre fonction a modifié une des listes reçues (remove, pop, sort…). Travaillez sur "
            "des copies ou de nouvelles structures : l'appelant a peut-être encore besoin de ses listes."
        )
        if isinstance(obtenu, (set, frozenset)):
            raise AssertionError(
                f"Sur {cas}, vous renvoyez un set : il n'a pas d'ORDRE, et l'équipe rétention traite la liste dans "
                "l'ordre de ce mois-ci (les plus risqués d'abord). Utilisez un set pour CHERCHER, une liste pour "
                "le résultat."
            )
        assert isinstance(obtenu, list), f"Sur {cas}, attendu une liste, obtenu {type(obtenu).__name__}"
        if obtenu != attendu:
            if sorted(obtenu) == sorted(attendu):
                raise AssertionError(
                    f"Sur {cas} : {obtenu} au lieu de {attendu}. Les bons clients, mais pas dans l'ORDRE de "
                    "`ce_mois`. Parcourez `ce_mois` (pas `mois_dernier`, ni un set) pour construire le résultat."
                )
            if len(obtenu) > len(set(obtenu)):
                raise AssertionError(
                    f"Sur {cas} : {obtenu} au lieu de {attendu}. Un client ne doit apparaître qu'UNE fois : retenez "
                    "ceux déjà ajoutés dans un set `deja_vus`."
                )
            raise AssertionError(f"Sur {cas} : {obtenu} au lieu de {attendu}.")

    # Les recherches dans une LISTE : c'est ce qui rend le code quadratique.
    avant = _ListeEspion(f"C{i:03d}" for i in range(0, 400, 2))
    maintenant = _ListeEspion(f"C{i:03d}" for i in range(0, 400, 3))
    f(avant, maintenant)
    for liste, nom in ((avant, "mois_dernier"), (maintenant, "ce_mois")):
        if liste.recherches > 3:
            raise AssertionError(
                f"Le résultat est juste, mais votre fonction a cherché {liste.recherches} fois dans la liste "
                f"`{nom}` (`x in {nom}`, .index, .count) pour 200 clients. Chaque recherche dans une liste la "
                "parcourt en entier : avec 200 000 clients chaque mois, cela fait des dizaines de milliards de "
                f"comparaisons (des heures). Convertissez d'abord : `{nom}_set = set({nom})`, puis cherchez dans le set."
            )
        if liste.parcours > 3:
            raise AssertionError(
                f"Le résultat est juste, mais votre fonction a parcouru la liste `{nom}` {liste.parcours} fois : une "
                "boucle dans une boucle (ou un set(…) recréé à chaque tour). Construisez le set UNE fois, AVANT la boucle."
            )

    # Et en vrai : 20 000 clients, tous encore à risque.
    ids = [f"C{i:06d}" for i in range(20_000)]
    duree = math.inf
    for _ in range(2):  # le meilleur de deux essais, pour ne pas pénaliser une machine occupée
        debut = time.perf_counter()
        obtenu = f(list(ids), list(ids))
        duree = min(duree, time.perf_counter() - debut)
        if duree < 0.5:
            break
    assert obtenu == ids, "Avec 20 000 clients tous encore à risque, le résultat doit être la liste complète, dans l'ordre."
    if duree >= 0.5:
        raise AssertionError(
            f"Le résultat est juste, mais 20 000 clients prennent {duree:.1f} s (un set répond en quelques "
            "millisecondes). Cherchez-vous dans votre propre liste résultat (`if x not in resultat`) ? C'est aussi "
            "une recherche dans une liste, qui grandit à chaque ajout : utilisez un set `deja_vus` à côté."
        )


# ============================================================================ Défi 46 : export Excel

_EXPORT = (
    "id;Contrat;Facture mensuelle ;Appels support;Paiement\n"
    "00123;Mensuel;89,90;3;Chèque\n"
    "00456;Annuel;45,50;0;Carte bancaire\n"
    "01789;Mensuel;120,00;5;Prélèvement\n"
)
_COLONNES_EXPORT = ["id", "Contrat", "Facture mensuelle", "Appels support", "Paiement"]
_VARIANTES = [
    # (description, encodage, fin de ligne, type du chemin)
    ("un fichier UTF-8", "utf-8", "\n", str),
    ("un fichier UTF-8 avec BOM (Excel « CSV UTF-8 »)", "utf-8-sig", "\r\n", Path),
    ("un fichier Windows-1252 (Excel « CSV (séparateur : point-virgule) »)", "cp1252", "\r\n", str),
]


def _verifier_export(df, cas: str) -> None:
    _pas_none(df, "lire_export_excel")
    assert isinstance(df, pd.DataFrame), f"Sur {cas}, attendu un DataFrame, obtenu {type(df).__name__}"
    colonnes = [str(c) for c in df.columns]
    if len(colonnes) == 1 and ";" in colonnes[0]:
        raise AssertionError(
            f"Sur {cas}, tout est dans UNE colonne ({colonnes[0][:40]!r}…). Excel en français sépare les colonnes "
            "par des POINTS-VIRGULES, car la virgule sert aux décimales : read_csv(…, sep=\";\")."
        )
    if colonnes and colonnes[0].startswith("﻿"):
        raise AssertionError(
            f"Sur {cas}, la première colonne s'appelle {colonnes[0]!r} : le caractère invisible \\ufeff est le "
            "BOM qu'Excel place au début des fichiers « CSV UTF-8 ». Lisez avec encoding=\"utf-8-sig\" : il "
            "retire le BOM s'il est là et lit l'UTF-8 normal sinon. Sans ça, df[\"id\"] lève une KeyError."
        )
    if colonnes != _COLONNES_EXPORT:
        if [c.strip() for c in colonnes] == _COLONNES_EXPORT:
            raise AssertionError(
                f"Sur {cas}, colonnes {colonnes} : l'en-tête « Facture mensuelle » a été tapé avec un espace en "
                "trop à la fin. Nettoyez les noms : df.columns = df.columns.str.strip()."
            )
        raise AssertionError(f"Sur {cas}, colonnes {colonnes} au lieu de {_COLONNES_EXPORT}.")
    paiements = [str(p) for p in df["Paiement"]]
    if any("Ã" in p for p in paiements):
        raise AssertionError(
            f"Sur {cas}, « Chèque » est devenu {paiements[0]!r} : de l'UTF-8 lu comme du Windows-1252 (ou latin-1). "
            "cp1252 décode n'importe quoi SANS planter, donc il ne doit servir qu'en DERNIER recours : essayez "
            "utf-8-sig, et seulement s'il lève UnicodeDecodeError, relisez en cp1252."
        )
    if paiements != ["Chèque", "Carte bancaire", "Prélèvement"]:
        raise AssertionError(f"Sur {cas}, la colonne Paiement vaut {paiements} au lieu de ['Chèque', 'Carte bancaire', 'Prélèvement'].")
    factures = df["Facture mensuelle"]
    if not pd.api.types.is_numeric_dtype(factures):
        raise AssertionError(
            f"Sur {cas}, « Facture mensuelle » est du texte ({list(factures)}) : on ne peut pas en faire la "
            "moyenne. En français, 89,90 s'écrit avec une virgule : read_csv(…, decimal=\",\")."
        )
    assert np.allclose(factures.to_numpy(dtype=float), [89.9, 45.5, 120.0]), (
        f"Sur {cas}, « Facture mensuelle » vaut {list(factures)} au lieu de [89.9, 45.5, 120.0]."
    )
    assert pd.api.types.is_numeric_dtype(df["Appels support"]) and list(df["Appels support"]) == [3, 0, 5], (
        f"Sur {cas}, « Appels support » vaut {list(df['Appels support'])} au lieu des nombres [3, 0, 5]."
    )
    ids = list(df["id"])
    if ids != ["00123", "00456", "01789"]:
        if [str(i) for i in ids] == ["123", "456", "1789"]:
            raise AssertionError(
                f"Sur {cas}, les identifiants sont devenus {ids} : pandas a vu des chiffres et en a fait des "
                "NOMBRES, en perdant les zéros du début. « 00123 » ne correspond plus à aucun client du CRM ! Un "
                "identifiant est un TEXTE (on ne l'additionne jamais) : read_csv(…, dtype={\"id\": str})."
            )
        raise AssertionError(f"Sur {cas}, la colonne id vaut {ids} au lieu de ['00123', '00456', '01789'].")
    assert isinstance(df.index, pd.RangeIndex), (
        f"Sur {cas}, l'index vaut {list(df.index)} : gardez l'index par défaut 0, 1, 2… (pas d'index_col)."
    )


def verifier_lire_export_excel(f) -> None:
    with tempfile.TemporaryDirectory() as dossier:
        for i, (cas, encodage, fin_de_ligne, type_chemin) in enumerate(_VARIANTES):
            chemin = Path(dossier) / f"export_{i}.csv"
            chemin.write_bytes(_EXPORT.replace("\n", fin_de_ligne).encode(encodage))
            try:
                df = f(type_chemin(chemin))
            except NotImplementedError:
                raise
            except UnicodeDecodeError as exc:
                raise AssertionError(
                    f"Sur {cas}, votre fonction plante : {exc.__class__.__name__} ({exc.reason} à l'octet "
                    f"{exc.start}). Ce fichier n'est pas en UTF-8 : c'est le Windows-1252 des anciennes versions "
                    "d'Excel (« è » y est l'octet 0xe8). Essayez utf-8-sig, et si UnicodeDecodeError, relisez avec "
                    "encoding=\"cp1252\" (try / except UnicodeDecodeError)."
                ) from None
            except (TypeError, AttributeError) as exc:
                raise AssertionError(
                    f"Sur {cas} (chemin de type {type_chemin.__name__}), votre fonction plante "
                    f"({type(exc).__name__}: {exc}). Acceptez un str comme un Path : pd.read_csv sait lire les deux."
                ) from None
            _verifier_export(df, cas)


# ============================================================================ Défi 47 : générateurs


class _Source:
    """Un itérateur de 0 à n-1 qui compte combien d'éléments on lui a demandés."""

    def __init__(self, n: int) -> None:
        self._interne = iter(range(n))
        self.lus = 0

    def __iter__(self):
        return self

    def __next__(self):
        valeur = next(self._interne)
        self.lus += 1
        return valeur


def _lots(f, elements, taille, maximum: int = 50) -> list:
    """Récupère au plus `maximum` lots (protège d'un générateur infini)."""
    return list(itertools.islice(iter(f(elements, taille)), maximum))


def verifier_par_lots(f) -> None:
    cas = [
        (["a", "b", "c", "d", "e"], 2, [["a", "b"], ["c", "d"], ["e"]], "l'exemple de la docstring"),
        (list(range(6)), 3, [[0, 1, 2], [3, 4, 5]], "6 éléments par lots de 3"),
        ([1, 2, 3], 10, [[1, 2, 3]], "un lot plus grand que la liste"),
        ([1, 2, 3], 1, [[1], [2], [3]], "des lots de 1"),
        ([], 3, [], "une liste vide"),
    ]
    for elements, taille, attendu, nom in cas:
        res = f(list(elements), taille)
        _pas_none(res, f"par_lots({elements!r}, {taille})")
        lots = _lots(f, list(elements), taille)
        if lots and len(lots) == len(attendu) and all(isinstance(lot, tuple) for lot in lots):
            raise AssertionError(f"Sur {nom}, les lots sont des tuples ({lots[0]!r}) : renvoyez des listes (list(…)).")
        if len(lots) == len(attendu) > 1 and len({id(lot) for lot in lots}) < len(lots):
            raise AssertionError(
                f"Sur {nom}, vous obtenez {lots} au lieu de {attendu} : tous les lots sont la MÊME liste. Après "
                "`yield lot`, `lot.clear()` vide aussi la liste déjà donnée à l'appelant. Créez une nouvelle "
                "liste à chaque fois : `lot = []`."
            )
        if lots != attendu:
            if lots == attendu[:-1]:
                raise AssertionError(
                    f"Sur {nom}, il manque le dernier lot : {lots} au lieu de {attendu}. Après la boucle, s'il reste "
                    "des éléments (même moins que `taille`), il faut encore les donner."
                )
            if lots[:-1] == attendu and lots[-1:] == [[]]:
                raise AssertionError(
                    f"Sur {nom}, le dernier lot est VIDE : {lots}. Ne donnez le reste que s'il contient quelque "
                    "chose (`if lot:`) : un lot vide ferait un appel inutile au modèle (ou à l'API)."
                )
            raise AssertionError(f"Sur {nom}, vous obtenez {lots} au lieu de {attendu}.")

    try:
        lots = _lots(f, (x * x for x in range(5)), 2)
    except TypeError as exc:
        raise AssertionError(
            f"Avec un GÉNÉRATEUR en entrée, votre fonction plante (TypeError: {exc}). Un générateur n'a ni len() ni "
            "indices [i:j] : il se lit une seule fois, dans l'ordre. Parcourez `elements` avec une boucle for, "
            "qui marche sur tout ce qui est itérable (liste, range, fichier, générateur, df.itertuples()…)."
        ) from None
    assert lots == [[0, 1], [4, 9], [16]], f"Avec le générateur des carrés 0, 1, 4, 9, 16 : {lots} au lieu de [[0, 1], [4, 9], [16]]"

    source = _Source(1000)
    res = f(source, 3)
    premier = next(iter(res))
    if source.lus > 4:
        raise AssertionError(
            f"Pour donner le PREMIER lot de 3, votre fonction a lu {source.lus} éléments sur 1000 : elle charge tout "
            "en mémoire avant de commencer. Avec un fichier de 50 millions de lignes, l'ordinateur sature. Utilisez "
            "`yield lot` au lieu de construire une liste de lots : la fonction devient un générateur, qui ne lit "
            "que ce dont on a besoin, au moment où on le demande."
        )
    assert premier == [0, 1, 2], f"Premier lot : {premier} au lieu de [0, 1, 2]"

    for taille in (0, -2):
        try:
            lots = _lots(f, [1, 2, 3], taille, maximum=10)
        except ValueError:
            continue
        raise AssertionError(
            f"par_lots([1, 2, 3], {taille}) donne {lots[:3]}{'…' if len(lots) > 3 else ''} au lieu de lever une "
            "ValueError. Une taille < 1 n'a pas de sens (et peut faire tourner une boucle à l'infini) : "
            "vérifiez-la en tout premier, `if taille < 1: raise ValueError(…)`."
        )


# ============================================================================ Défi 48 : classes


def _nouveau(cls, *args):
    try:
        return cls(*args)
    except TypeError as exc:
        raise AssertionError(
            f"JournalPredictions({', '.join(map(repr, args))}) plante (TypeError: {exc}). Le constructeur s'écrit "
            "`def __init__(self, seuil=0.5):` : n'oubliez ni `self`, ni la valeur par défaut."
        ) from None


def _longueur(journal) -> int:
    try:
        return len(journal)
    except TypeError as exc:
        raise AssertionError(
            f"len(journal) plante ({exc}). Pour que len() marche sur vos objets, définissez la méthode spéciale "
            "`def __len__(self): return …`."
        ) from None


def verifier_journal_predictions(cls) -> None:
    journal = _nouveau(cls)
    assert getattr(journal, "seuil", None) == 0.5, (
        "JournalPredictions() doit avoir un attribut `seuil` valant 0.5 par défaut : dans __init__, `self.seuil = seuil`."
    )
    assert _longueur(journal) == 0, "Un journal tout neuf doit être vide : len(journal) == 0"
    assert _pas_none(journal.a_risque(), "a_risque()") == [], "Un journal vide n'a aucun client à risque : attendu []"
    try:
        taux = journal.taux_alerte()
    except ZeroDivisionError:
        raise AssertionError(
            "taux_alerte() plante sur un journal vide (ZeroDivisionError) : sans client, renvoyez 0.0."
        ) from None
    assert taux == 0.0, f"taux_alerte() d'un journal vide : {taux!r} au lieu de 0.0"

    # Des scores qui arrivent dans le désordre, avec une égalité.
    for client, proba in [("C4", 0.9), ("C2", 0.2), ("C1", 0.9), ("C3", 0.5), ("C5", 0.7)]:
        journal.ajouter(client, proba)
    assert _longueur(journal) == 5, f"Après 5 clients ajoutés, len(journal) vaut {len(journal)} au lieu de 5"
    alertes = _pas_none(journal.a_risque(), "a_risque()")
    attendu = ["C1", "C4", "C5", "C3"]
    if alertes != attendu:
        if "C3" not in alertes and [c for c in attendu if c != "C3"] == alertes:
            raise AssertionError(
                f"a_risque() renvoie {alertes} : il manque C3, dont la probabilité vaut EXACTEMENT le seuil (0.5). "
                "Le seuil est inclus : proba >= seuil (comme partout dans le projet)."
            )
        if alertes == ["C4", "C1", "C5", "C3"]:
            raise AssertionError(
                f"a_risque() renvoie {alertes} au lieu de {attendu} : C1 et C4 ont la même probabilité (0.9). En cas "
                "d'égalité, triez par identifiant pour que le résultat ne dépende pas de l'ordre d'arrivée : "
                "key=lambda c: (-proba, client)."
            )
        if sorted(alertes) == sorted(attendu):
            raise AssertionError(
                f"a_risque() renvoie {alertes} au lieu de {attendu} : triez par probabilité DÉCROISSANTE (le plus "
                "risqué en premier), puis par identifiant."
            )
        raise AssertionError(f"a_risque() renvoie {alertes} au lieu de {attendu} (scores : C4 0.9, C2 0.2, C1 0.9, C3 0.5, C5 0.7).")
    taux = journal.taux_alerte()
    assert isinstance(taux, (int, float)) and math.isclose(taux, 0.8), (
        f"taux_alerte() vaut {taux!r} au lieu de 0.8 (4 clients à risque sur 5). C'est une proportion, entre 0 et 1."
    )

    # Le score d'un client change : le nouveau REMPLACE l'ancien.
    journal.ajouter("C2", 0.95)
    journal.ajouter("C1", 0.1)
    if _longueur(journal) != 5:
        raise AssertionError(
            f"Après avoir mis à jour les scores de C2 et C1, len(journal) vaut {len(journal)} au lieu de 5. Un client "
            "déjà présent ne compte qu'une fois et son nouveau score remplace l'ancien : un dictionnaire "
            "{client: proba} fait exactement ça (`self.scores[client_id] = proba`)."
        )
    alertes = journal.a_risque()
    assert alertes == ["C2", "C4", "C5", "C3"], (
        f"Après C2 → 0.95 et C1 → 0.1, a_risque() renvoie {alertes} au lieu de ['C2', 'C4', 'C5', 'C3'] : c'est le "
        "DERNIER score de chaque client qui compte (pas le plus haut, pas le premier)."
    )

    # Des probabilités impossibles sont refusées… sans rien enregistrer.
    for proba in (1.2, -0.1, float("nan")):
        try:
            journal.ajouter("C9", proba)
        except ValueError:
            pass
        else:
            indice = (" Attention : `nan < 0` et `nan > 1` sont tous deux faux ; écrivez `if not 0 <= proba <= 1`."
                      if math.isnan(proba) else "")
            raise AssertionError(f"ajouter('C9', {proba}) doit lever une ValueError : une probabilité est entre 0 et 1.{indice}")
        assert _longueur(journal) == 5, (
            f"ajouter('C9', {proba}) a levé une ValueError, mais C9 a quand même été enregistré : vérifiez AVANT d'écrire."
        )

    # Le seuil peut changer après coup : les alertes doivent suivre.
    journal.seuil = 0.6
    alertes = journal.a_risque()
    assert alertes == ["C2", "C4", "C5"], (
        f"Après `journal.seuil = 0.6`, a_risque() renvoie {alertes} au lieu de ['C2', 'C4', 'C5']. Ne figez rien au "
        "moment de l'ajout : stockez les scores, et calculez les alertes À LA DEMANDE avec self.seuil."
    )
    assert _nouveau(cls, 0.3).seuil == 0.3, "JournalPredictions(0.3).seuil doit valoir 0.3"

    # Deux journaux (deux mois, deux modèles…) sont indépendants.
    janvier, fevrier = _nouveau(cls), _nouveau(cls)
    janvier.ajouter("X1", 0.8)
    if _longueur(fevrier) != 0:
        raise AssertionError(
            "Un client ajouté au journal de janvier apparaît aussi dans celui de février ! Le dictionnaire est un "
            "attribut de CLASSE (écrit sous `class JournalPredictions:`), partagé par TOUS les objets. Créez-le "
            "dans __init__ : `self.scores = {}`, et chaque journal aura le sien."
        )

    for seuil in (1.5, -0.1):
        try:
            cls(seuil)
        except ValueError:
            continue
        raise AssertionError(f"JournalPredictions({seuil}) doit lever une ValueError : un seuil est entre 0 et 1.")


# ============================================================================ Défi 49 : décorateurs


def verifier_avec_cache(f) -> None:
    appels: list = []

    def score_client(client_id, seuil=0.5):
        """Probabilité de churn d'un client (lente : interroge le modèle)."""
        appels.append((client_id, seuil))
        return f"{client_id!r}@{seuil}"

    score = f(score_client)
    _pas_none(score, "avec_cache(score_client)")
    assert callable(score), f"avec_cache doit renvoyer une FONCTION, obtenu {type(score).__name__}"

    assert score("C1") == "'C1'@0.5", f"score('C1') renvoie {score('C1')!r} au lieu du résultat de la fonction d'origine"
    score("C1")
    if len(appels) != 1:
        if score is score_client:
            raise AssertionError(
                "avec_cache renvoie la fonction d'origine telle quelle : rien n'est mis en cache. Définissez une "
                "fonction `enveloppe(*args, **kwargs)` à l'intérieur, qui regarde le cache avant d'appeler, et renvoyez-la."
            )
        raise AssertionError(
            f"Après deux appels score('C1'), la fonction d'origine a été appelée {len(appels)} fois au lieu de 1 : "
            "le deuxième appel doit réutiliser le résultat mis en cache."
        )
    score("C2")
    assert len(appels) == 2, "score('C2') n'a jamais été calculé : des arguments différents demandent un nouveau calcul."

    stats = getattr(score, "statistiques", None)
    if stats is None:
        raise AssertionError(
            "La fonction renvoyée n'a pas de méthode statistiques(). (En vrai, @functools.cache fait le reste de ce "
            "défi pour vous, mais ici on le recode pour comprendre.) Ajoutez-la à l'enveloppe avant de la renvoyer : "
            "`enveloppe.statistiques = lambda: {…}`."
        )
    valeurs = stats()
    assert valeurs == {"calculs": 2, "reutilisations": 1}, (
        f"Après score('C1'), score('C1'), score('C2') : statistiques() renvoie {valeurs!r} au lieu de "
        "{'calculs': 2, 'reutilisations': 1}. Comptez dans une variable de l'enveloppe (dictionnaire, ou `nonlocal`)."
    )

    # Les arguments nommés font partie de la clé.
    a, b = score("C1", seuil=0.3), score("C1", seuil=0.7)
    if a == b:
        raise AssertionError(
            f"score('C1', seuil=0.3) et score('C1', seuil=0.7) renvoient tous deux {a!r} : la clé du cache oublie les "
            "arguments NOMMÉS. Utilisez une clé qui contient args ET kwargs : (args, tuple(sorted(kwargs.items())))."
        )
    score("C1", seuil=0.3)
    assert len(appels) == 4, "Le deuxième score('C1', seuil=0.3) doit venir du cache."

    # 1 et "1" ne sont pas le même client.
    un, texte = score(1), score("1")
    if un == texte:
        raise AssertionError(
            "score(1) et score('1') renvoient le même résultat : la clé du cache confond l'entier 1 et le texte '1' "
            "(une clé fabriquée avec str() ou join ?). Utilisez directement le tuple des arguments comme clé de "
            "dictionnaire : il garde le type de chaque valeur."
        )

    # Deux fonctions décorées ont chacune leur cache.
    plus_un = f(lambda x: x + 1)
    fois_dix = f(lambda x: x * 10)
    plus_un(2)
    if fois_dix(2) != 20:
        raise AssertionError(
            f"fois_dix(2) renvoie {fois_dix(2)!r} au lieu de 20 : il a reçu le résultat de plus_un(2). Le cache est "
            "PARTAGÉ entre toutes les fonctions décorées (variable globale, ou argument par défaut `cache={}` créé "
            "une seule fois). Créez le dictionnaire DANS avec_cache : un nouveau à chaque décoration."
        )

    # Un résultat « faux » (0.0, None) reste un vrai résultat, à mettre en cache.
    for resultat in (0.0, None):
        calculs: list = []

        def modele(client_id, _r=resultat, _c=calculs):
            _c.append(client_id)
            return _r

        rapide = f(modele)
        rapide("C7")
        obtenu = rapide("C7")
        assert obtenu == resultat, f"Le cache renvoie {obtenu!r} au lieu de {resultat!r}"
        if len(calculs) != 1:
            raise AssertionError(
                f"Une fonction qui renvoie {resultat!r} est recalculée à chaque appel : `if cache.get(cle):` (ou "
                f"`if cache.get(cle) is not None`) considère {resultat!r} comme « absent ». Un client de probabilité "
                "0.0 existe ! Testez la présence de la CLÉ : `if cle in cache:`."
            )

    # Une erreur n'est pas un résultat.
    essais: list = []

    def fragile(x):
        essais.append(x)
        if len(essais) == 1:
            raise ConnectionError("API du modèle indisponible")
        return x * 2

    robuste = f(fragile)
    try:
        robuste(3)
    except ConnectionError:
        pass
    else:
        raise AssertionError(
            "Quand la fonction d'origine lève une erreur, l'enveloppe doit la laisser passer (pas de try/except qui "
            "l'avale) : l'appelant doit savoir que le calcul a échoué."
        )
    obtenu = robuste(3)
    assert obtenu == 6, (
        f"Après un premier appel en erreur, robuste(3) renvoie {obtenu!r} au lieu de 6 : une erreur ne doit pas être "
        "mise en cache. Ne remplissez le cache qu'APRÈS un calcul réussi."
    )

    nom, doc = getattr(score, "__name__", None), getattr(score, "__doc__", None)
    if nom != "score_client" or doc != score_client.__doc__:
        raise AssertionError(
            f"Tout fonctionne, mais la fonction décorée s'appelle {nom!r} au lieu de 'score_client' et a perdu sa "
            "documentation : help(), les traces d'erreur et les rapports de tests deviennent illisibles. Décorez "
            "l'enveloppe avec @functools.wraps(fonction)."
        )


SERIE = Serie(
    code="07",
    titre="Écrire du code réutilisable : dates, fichiers, générateurs, classes",
    module_eleve="defis.serie_07",
    module_corrige="defis.corrige_serie_07",
    defis=(
        Defi(44, "anciennete_en_mois", "🟢", "—", verifier_anciennete_en_mois),
        Defi(45, "clients_toujours_a_risque", "🟢", "—", verifier_clients_toujours_a_risque),
        Defi(46, "lire_export_excel", "🟡", "—", verifier_lire_export_excel),
        Defi(47, "par_lots", "🟡", "6.4", verifier_par_lots),
        Defi(48, "JournalPredictions", "🟡", "—", verifier_journal_predictions),
        Defi(49, "avec_cache", "🔴", "—", verifier_avec_cache),
    ),
)

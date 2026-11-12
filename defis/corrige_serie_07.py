"""Série 07 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_07.py`.

Si votre version passe les vérifications, elle est juste, même si elle ne ressemble pas à
celle-ci. Comparez : notez dans JOURNAL.md l'écriture ou le réflexe que vous ne connaissiez pas.
"""

from __future__ import annotations

import calendar
import functools
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd

# ============================================================================ Dates


def _en_date(valeur: date | str) -> date:
    if isinstance(valeur, date):
        return valeur
    # fromisoformat lève une ValueError sur "15/01/2024" : on refuse de DEVINER le format
    # (le 03/04, c'est le 3 avril en France et le 4 mars aux États-Unis).
    return date.fromisoformat(valeur.strip())


def anciennete_en_mois(debut: date | str, fin: date | str) -> int:
    debut, fin = _en_date(debut), _en_date(fin)
    if fin < debut:
        raise ValueError(f"La fin ({fin}) est avant le début ({debut}).")

    mois = (fin.year - debut.year) * 12 + (fin.month - debut.month)
    # Le dernier mois n'est complet que si on a atteint le même numéro de jour…
    dernier_jour = calendar.monthrange(fin.year, fin.month)[1]
    # … ou la fin d'un mois trop court pour l'avoir (31 janvier → 29 février = 1 mois).
    if fin.day < debut.day and fin.day != dernier_jour:
        mois -= 1
    return mois


# ============================================================================ Complexité


def clients_toujours_a_risque(mois_dernier: list[str], ce_mois: list[str]) -> list[str]:
    # Un set répond à « x est-il dedans ? » en temps constant, quelle que soit sa taille.
    # `x in une_liste`, lui, parcourt la liste : 200 000 × 200 000 comparaisons.
    avant = set(mois_dernier)
    deja_vus: set[str] = set()   # même raison : `x not in resultat` serait lent
    resultat = []
    for client in ce_mois:       # on parcourt `ce_mois` pour garder SON ordre
        if client in avant and client not in deja_vus:
            resultat.append(client)
            deja_vus.add(client)
    return resultat

    # Variante en une ligne (dict garde l'ordre d'insertion depuis Python 3.7) :
    #     avant = set(mois_dernier)
    #     return list(dict.fromkeys(c for c in ce_mois if c in avant))
    # À éviter : list(set(ce_mois) & set(mois_dernier)) — rapide, mais l'ordre est perdu.


# ============================================================================ Fichiers


def lire_export_excel(chemin: str | Path) -> pd.DataFrame:
    # utf-8-sig lit l'UTF-8 avec OU sans BOM : on l'essaie d'abord. cp1252 (Windows) décode
    # presque n'importe quels octets, sans jamais planter : il doit donc passer en DERNIER,
    # sinon un fichier UTF-8 serait lu de travers (« Chèque » → « ChÃ¨que ») sans erreur.
    for encodage in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(chemin, sep=";", decimal=",", encoding=encodage, dtype={"id": str})
            break
        except UnicodeDecodeError:
            continue
    df.columns = df.columns.str.strip()   # « Contrat » et « Contrat  » sont deux colonnes différentes
    return df


# ============================================================================ Générateurs


def par_lots(elements: Iterable, taille: int) -> Iterator[list]:
    if taille < 1:
        raise ValueError(f"La taille d'un lot doit être ≥ 1 (reçu : {taille!r}).")
    lot = []
    for element in elements:      # `for` marche sur TOUT itérable : liste, range, générateur, fichier…
        lot.append(element)
        if len(lot) == taille:
            yield lot             # le code s'arrête ici jusqu'à ce qu'on demande le lot suivant
            lot = []              # une NOUVELLE liste (lot.clear() viderait aussi celle qu'on vient de donner)
    if lot:                       # le dernier lot, incomplet… mais jamais vide
        yield lot

    # Python 3.12+ : itertools.batched(elements, taille) fait la même chose (avec des tuples).
    # Remarque : ici, la ValueError ne part qu'au premier `next()`, car le corps d'un générateur
    # ne s'exécute qu'à ce moment-là. Pour la lever dès l'appel, on sépare la vérification :
    #     def par_lots(elements, taille):
    #         if taille < 1: raise ValueError(...)
    #         return _par_lots(elements, taille)   # _par_lots contient les yield


# ============================================================================ Classes


class JournalPredictions:
    def __init__(self, seuil: float = 0.5) -> None:
        if not 0 <= seuil <= 1:
            raise ValueError(f"Le seuil doit être entre 0 et 1 (reçu : {seuil!r}).")
        self.seuil = seuil
        # Créé DANS __init__ : chaque journal a son propre dictionnaire. Écrit au niveau de la
        # classe (`scores = {}` sous `class …:`), il serait partagé par TOUS les journaux.
        self.scores: dict[str, float] = {}

    def ajouter(self, client_id: str, proba: float) -> None:
        if not 0 <= proba <= 1:   # attrape aussi nan : `0 <= nan` est False
            raise ValueError(f"Probabilité invalide pour {client_id!r} : {proba!r}")
        self.scores[client_id] = proba   # un dictionnaire : le nouveau score remplace l'ancien

    def __len__(self) -> int:
        return len(self.scores)

    def a_risque(self) -> list[str]:
        # Calculé à la demande, avec le seuil ACTUEL : si on le change, le résultat suit.
        alertes = [(proba, client) for client, proba in self.scores.items() if proba >= self.seuil]
        alertes.sort(key=lambda pc: (-pc[0], pc[1]))   # proba décroissante, puis id croissant
        return [client for _, client in alertes]

    def taux_alerte(self) -> float:
        if not self.scores:
            return 0.0
        return len(self.a_risque()) / len(self.scores)


# ============================================================================ Décorateurs


def avec_cache(fonction: Callable) -> Callable:
    # Ces variables vivent dans la « fermeture » (closure) de `enveloppe` : une par appel à
    # avec_cache, donc un cache par fonction décorée.
    cache: dict = {}
    stats = {"calculs": 0, "reutilisations": 0}

    @functools.wraps(fonction)    # recopie __name__, __doc__… : le tableau de bord et help() restent lisibles
    def enveloppe(*args, **kwargs):
        cle = (args, tuple(sorted(kwargs.items())))
        if cle in cache:          # PAS `if cache.get(cle):` — un score de 0.0 ou None est un vrai résultat
            stats["reutilisations"] += 1
            return cache[cle]
        stats["calculs"] += 1
        resultat = fonction(*args, **kwargs)   # si ça plante, rien n'est mis en cache
        cache[cle] = resultat
        return resultat

    enveloppe.statistiques = lambda: dict(stats)   # une copie : l'appelant ne peut pas fausser les compteurs
    return enveloppe

    # En vrai : @functools.cache (ou @functools.lru_cache(maxsize=…) pour borner la mémoire), et
    # dans Streamlit @st.cache_data — voir app.py. Ils font la même chose, en mieux testé.

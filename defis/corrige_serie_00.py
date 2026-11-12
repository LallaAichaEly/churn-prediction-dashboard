"""Série 00 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_00.py`.

Si votre version passe les vérifications, elle est juste, même si elle ne ressemble pas à
celle-ci. Comparez les deux : il y a presque toujours plusieurs bonnes façons d'écrire un
programme. Notez dans JOURNAL.md l'écriture que vous ne connaissiez pas.
"""

from __future__ import annotations

import pandas as pd

# Table de correspondance du défi 37 (la même que dans serie_00.py).
SYNONYMES = {
    "mensuel": "Mensuel",
    "mois": "Mensuel",
    "annuel": "Annuel",
    "1 an": "Annuel",
    "biannuel": "Biannuel",
    "bi-annuel": "Biannuel",
    "2 ans": "Biannuel",
}

# ============================================================================ Bases de Python


def taux_en_pourcentage(partis: int, total: int) -> float:
    # Le cas particulier d'abord : sans lui, 0 / 0 lève ZeroDivisionError.
    if total == 0:
        return 0.0
    # `return` RENVOIE la valeur à qui a appelé la fonction ; `print` ne fait que l'afficher.
    return round(100 * partis / total, 1)


def segment_client(anciennete_mois: int) -> str:
    if anciennete_mois < 0:
        raise ValueError(f"Ancienneté négative impossible : {anciennete_mois}")
    # Python teste les conditions dans l'ordre et s'arrête à la première vraie : arrivé au
    # `elif`, on sait déjà que anciennete_mois >= 6. Inutile d'écrire `6 <= anciennete_mois < 24`.
    if anciennete_mois < 6:
        return "nouveau"
    elif anciennete_mois < 24:
        return "installé"
    else:
        return "fidèle"


def compter_par_contrat(contrats: list[str]) -> dict[str, int]:
    compte: dict[str, int] = {}  # créé UNE fois, AVANT la boucle
    for contrat in contrats:
        # .get(clé, 0) renvoie 0 si la clé n'existe pas encore : pas de KeyError.
        compte[contrat] = compte.get(contrat, 0) + 1
    return compte
    # Version « bibliothèque standard » : from collections import Counter ; return dict(Counter(contrats))


def clients_a_risque(clients: list[dict], seuil: float = 0.5) -> list[str]:
    # 1) filtrer : une compréhension de liste se lit « les c de clients tels que … »
    retenus = [c for c in clients if c["proba"] >= seuil]
    # 2) trier : sorted() crée une NOUVELLE liste (list.sort() modifierait celle de l'appelant).
    #    key= dit sur quoi trier ; reverse=True met les plus grandes probabilités en premier.
    retenus = sorted(retenus, key=lambda c: c["proba"], reverse=True)
    # 3) ne garder que les identifiants
    return [c["id"] for c in retenus]


def facture_moyenne(factures: list[float]) -> float:
    # Les trois bugs du code d'origine :
    #   - `range(1, len(factures))` sautait la première facture (les indices commencent à 0) ;
    #   - `total = factures[i]` remplaçait le total au lieu de l'augmenter (`+=`) ;
    #   - une liste vide provoquait une division par zéro.
    if len(factures) == 0:
        return 0.0
    total = 0.0
    for facture in factures:  # parcourir les valeurs directement évite les erreurs d'indice
        total += facture
    return total / len(factures)
    # En une ligne : return sum(factures) / len(factures) if factures else 0.0


# ============================================================================ Premiers pas en pandas


def filtrer_clients(df: pd.DataFrame, contrat: str, anciennete_max: int) -> pd.DataFrame:
    # Chaque condition donne une Series de True/False (un « masque »).
    # On les combine avec & (et) — pas `and` — et CHAQUE condition est entre parenthèses,
    # car & est prioritaire sur == et <= en Python.
    masque = (df["contract"] == contrat) & (df["tenure_months"] <= anciennete_max)
    return df[masque]
    # Alternative lisible : df.query("contract == @contrat and tenure_months <= @anciennete_max")


def taux_par_groupe_a_la_main(df: pd.DataFrame, colonne: str) -> dict:
    taux = {}
    for modalite in df[colonne].unique():               # 1) les valeurs distinctes
        groupe = df[df[colonne] == modalite]             # 2) les lignes de ce groupe
        taux[modalite] = float(groupe["churn"].mean())   # 3) la moyenne de 0/1 = la proportion de 1
    return taux
    # C'est exactement ce que fait, en une ligne et bien plus vite :
    #     df.groupby(colonne)["churn"].mean().to_dict()
    # « split → apply → combine » : découper en groupes, calculer, rassembler.


def nettoyer_contrats(contrats: pd.Series) -> pd.Series:
    # 1) normaliser le texte : espaces autour, majuscules/minuscules
    cles = contrats.str.strip().str.lower()
    # 2) traduire avec le dictionnaire ; ce qui n'y est pas devient NaN
    propres = cles.map(SYNONYMES)
    # 3) une valeur est INCONNUE si elle existait (non manquante) mais n'a pas été traduite.
    #    Piège : `propres.isna()` seul compterait aussi les vraies valeurs manquantes.
    inconnues = contrats[propres.isna() & contrats.notna()]
    if len(inconnues) > 0:
        raise ValueError(f"Contrats inconnus : {sorted(inconnues.unique())}")
    return propres  # même index que l'entrée : .str et .map le conservent

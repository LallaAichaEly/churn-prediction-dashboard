"""Étape 1 (corrigé) — Refuser un lot abîmé AVANT d'entraîner quoi que ce soit."""

from __future__ import annotations

import pandas as pd

from .modeles import CATEGORIELLES, CIBLE, COLONNES, IDENTIFIANT, MODALITES, MOIS, NUMERIQUES

#: AAAA-MM, mois de 01 à 12 : "2026-9", "2026/09" ou "09-2026" sont refusés.
_FORMAT_MOIS = r"\d{4}-(0[1-9]|1[0-2])"


def problemes(lots: pd.DataFrame) -> list[str]:
    manquantes = [c for c in COLONNES if c not in lots.columns]
    if manquantes:
        # Inutile d'aller plus loin : toutes les règles suivantes lisent ces colonnes.
        return [f"colonnes manquantes : {', '.join(manquantes)}"]

    trouves: list[str] = []

    # 1. Les cases vides, colonne par colonne (une seule phrase par colonne, avec le nombre).
    for colonne in COLONNES:
        vides = int(lots[colonne].isna().sum())
        if vides:
            trouves.append(f"{colonne} : {vides} valeur(s) manquante(s)")

    # 2. Le mois : un texte AAAA-MM. `str.fullmatch` refuse aussi « 2026-09 » suivi d'un espace.
    mois = lots[MOIS].dropna().astype(str)
    mal_ecrits = mois[~mois.str.fullmatch(_FORMAT_MOIS)]
    if len(mal_ecrits):
        trouves.append(f"{MOIS} : {len(mal_ecrits)} valeur(s) au mauvais format (ex. {mal_ecrits.iloc[0]!r}), "
                       f"attendu AAAA-MM")

    # 3. La cible : 0 ou 1. `to_numeric(errors="coerce")` transforme « oui » en NaN, qui n'est pas dans {0, 1}.
    cible = lots[CIBLE].dropna()
    hors_cible = cible[~pd.to_numeric(cible, errors="coerce").isin([0, 1])]
    if len(hors_cible):
        trouves.append(f"{CIBLE} : {len(hors_cible)} valeur(s) autre(s) que 0 ou 1 (ex. {hors_cible.iloc[0]!r})")

    # 4. Les nombres : lisibles (« 45,90 » ne l'est pas) et jamais négatifs.
    for colonne in NUMERIQUES:
        brutes = lots[colonne].dropna()
        valeurs = pd.to_numeric(brutes, errors="coerce")
        illisibles = brutes[valeurs.isna()]
        if len(illisibles):
            trouves.append(f"{colonne} : {len(illisibles)} valeur(s) non numérique(s) (ex. {illisibles.iloc[0]!r})")
        negatives = int((valeurs < 0).sum())
        if negatives:
            trouves.append(f"{colonne} : {negatives} valeur(s) négative(s)")

    # 5. Les catégories : une faute de frappe (« Mensual ») serait IGNORÉE en silence par le modèle
    # (OneHotEncoder(handle_unknown="ignore")) : le client serait scoré comme s'il n'avait pas de contrat.
    for colonne in CATEGORIELLES:
        valeurs = lots[colonne].dropna()
        inconnues = valeurs[~valeurs.isin(MODALITES[colonne])]
        if len(inconnues):
            exemples = ", ".join(repr(v) for v in sorted(set(map(str, inconnues)))[:3])
            trouves.append(f"{colonne} : {len(inconnues)} valeur(s) inconnue(s) ({exemples})")

    # 6. Un client deux fois dans la même photo : l'export a été concaténé deux fois.
    doublons = lots[lots.duplicated([MOIS, IDENTIFIANT], keep="first")]
    if len(doublons):
        exemple = doublons.iloc[0]
        trouves.append(f"{len(doublons)} client(s) en double dans le même mois "
                       f"(ex. {exemple[IDENTIFIANT]} en {exemple[MOIS]})")

    # 7. Un mois où personne n'est parti (ou tout le monde) : la colonne churn n'a pas été remplie.
    lisibles = lots.assign(_cible=pd.to_numeric(lots[CIBLE], errors="coerce")).dropna(subset=[MOIS, "_cible"])
    for un_mois, taux in lisibles.groupby(MOIS)["_cible"].mean().items():
        if taux in (0.0, 1.0):
            n = int((lisibles[MOIS] == un_mois).sum())
            trouves.append(f"mois {un_mois} : {taux:.0%} de départs sur {n} clients, étiquettes manquantes ?")

    return trouves

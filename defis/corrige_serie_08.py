"""Série 08 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_08.py`.

Si votre version passe les vérifications, elle est juste, même si elle ne ressemble pas à
celle-ci. Comparez : notez dans JOURNAL.md l'écriture ou le réflexe que vous ne connaissiez pas.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date

import numpy as np
import pandas as pd

# ============================================================================ Texte

# \b = « frontière de mot ». Sans elle, « AC00123 » (un code agence) et « C001234 » (une
# ancienne numérotation à 6 chiffres) donneraient de faux identifiants.
# On compile la regex une fois pour toutes, au chargement du module.
_IDENTIFIANT = re.compile(r"\bC\d{5}\b", flags=re.IGNORECASE)


def extraire_identifiants(texte: str | None) -> list[str]:
    # Une note vide arrive en None, ou en NaN (un float !) quand on fait df["note"].apply(...).
    if not isinstance(texte, str):
        return []
    trouves = (identifiant.upper() for identifiant in _IDENTIFIANT.findall(texte))
    # dict.fromkeys garde l'ordre d'insertion et ignore les doublons : un dédoublonnage
    # « stable » en une ligne (un set perdrait l'ordre).
    return list(dict.fromkeys(trouves))


# ============================================================================ Jointures


def ajouter_total_factures(clients: pd.DataFrame, factures: pd.DataFrame) -> pd.DataFrame:
    # 1) Agréger D'ABORD : une ligne par client. Joindre avant d'agréger dupliquerait chaque
    #    client autant de fois qu'il a de factures.
    totaux = factures.groupby("customer_id")["montant"].sum()
    # 2) `map` cherche chaque identifiant dans `totaux` : l'index et l'ordre de `clients` sont
    #    conservés, les factures de clients inconnus sont ignorées, les clients sans facture
    #    reçoivent NaN, que l'on remplace par 0 (ils n'ont rien payé).
    #    `assign` renvoie une copie : `clients` n'est pas modifié.
    return clients.assign(total_factures=clients["customer_id"].map(totaux).fillna(0.0).astype(float))

    # Version avec merge, tout aussi juste :
    #   resultat = clients.merge(totaux.rename("total_factures"), how="left",
    #                            left_on="customer_id", right_index=True, validate="many_to_one")
    #   resultat["total_factures"] = resultat["total_factures"].fillna(0.0)
    #   resultat.index = clients.index      # merge renumérote l'index : on remet le bon
    # `validate="many_to_one"` fait planter la jointure si `totaux` avait un client en double :
    # une assurance gratuite contre la duplication silencieuse de lignes.


# ============================================================================ JSON

COLONNES_API = ["customer_id", "contrat", "nb_factures", "facture_moyenne", "nb_options"]


def aplatir_reponse_api(reponse: dict | str) -> pd.DataFrame:
    if isinstance(reponse, str):  # le texte brut de la réponse HTTP
        reponse = json.loads(reponse)

    lignes = []
    # `.get(cle) or []` couvre les trois cas : clé absente, valeur null (None), liste vide.
    for client in reponse.get("clients") or []:
        contrat = client.get("contrat") or {}
        factures = client.get("factures") or []
        montants = [facture["montant"] for facture in factures]
        lignes.append({
            "customer_id": client["id"],
            "contrat": contrat.get("type"),
            "nb_factures": len(factures),
            # La moyenne de rien n'existe pas : NaN (« on ne sait pas »), surtout pas 0 €.
            "facture_moyenne": sum(montants) / len(montants) if montants else math.nan,
            "nb_options": len(client.get("options") or []),
        })
    # `columns=` garantit les colonnes même quand la page est vide.
    return pd.DataFrame(lignes, columns=COLONNES_API)

    # Pour aller plus loin : pd.json_normalize(reponse["clients"]) aplatit les dictionnaires
    # imbriqués automatiquement (colonnes « contrat.type »…), mais pas les listes : il faut
    # quand même compter les factures et les options soi-même.


# ============================================================================ Vectorisation


def niveau_risque_lent(df: pd.DataFrame) -> pd.Series:
    """La version d'origine, juste mais lente (identique à celle de `defis/serie_08.py`)."""
    niveaux = []
    for _, client in df.iterrows():
        points = 0
        if client["contract"] == "Mensuel":
            points += 3
        if client["support_calls"] >= 3:
            points += 2
        if client["tenure_months"] < 6:
            points += 1
        if client["tech_support"] == "Oui":
            points -= 2
        if points >= 4:
            niveaux.append("élevé")
        elif points >= 2:
            niveaux.append("moyen")
        else:
            niveaux.append("faible")
    return pd.Series(niveaux, index=df.index, name="niveau_risque")


def niveau_risque(df: pd.DataFrame) -> pd.Series:
    # Chaque condition devient une colonne de True/False calculée d'un coup ; multipliée par
    # un entier, True vaut 1 et False vaut 0. Une seule addition pour 200 000 clients.
    points = (
        3 * (df["contract"] == "Mensuel")
        + 2 * (df["support_calls"] >= 3)
        + 1 * (df["tenure_months"] < 6)
        - 2 * (df["tech_support"] == "Oui")
    )
    # np.select = un if / elif / else vectorisé : la PREMIÈRE condition vraie l'emporte.
    niveaux = np.select([points >= 4, points >= 2], ["élevé", "moyen"], default="faible")
    return pd.Series(niveaux, index=df.index, name="niveau_risque")


# ============================================================================ Mise à jour


def mettre_a_jour(table: pd.DataFrame, nouveautes: pd.DataFrame, cle: str = "customer_id") -> pd.DataFrame:
    # Garde-fou : si la référence est déjà corrompue, on s'arrête AVANT d'aggraver les choses.
    doublons = table.loc[table[cle].duplicated(), cle]
    if not doublons.empty:
        raise ValueError(f"La table de référence contient des clés en double : {doublons.unique()[:5].tolist()}")
    if nouveautes.empty:  # une nuit sans modification
        return table.reset_index(drop=True)

    # concat EMPILE l'ancienne table et les nouveautés (dans l'ordre des colonnes de `table`) ;
    # drop_duplicates(subset=cle, keep="last") garde, pour chaque client, la DERNIÈRE ligne :
    # celle des nouveautés si elle existe, et parmi elles la plus récente de la journée.
    # Contrairement à DataFrame.update, une valeur devenue vide (email retiré) remplace bien
    # l'ancienne. Et rejouer la même mise à jour redonne la même table : c'est idempotent.
    tout = pd.concat([table, nouveautes[table.columns]], ignore_index=True)
    return tout.drop_duplicates(subset=cle, keep="last").reset_index(drop=True)


# ============================================================================ Événements


def variables_evenements(
    evenements: pd.DataFrame,
    clients: list[str],
    date_reference: str | date,
    fenetre_jours: int = 30,
) -> pd.DataFrame:
    reference = pd.Timestamp(date_reference)
    # `assign` travaille sur une copie : la colonne « date » de l'appelant reste intacte.
    ev = evenements.assign(date=pd.to_datetime(evenements["date"]))

    # 1) LA règle d'or : on ne garde que ce qu'on savait AU MOMENT de la prédiction, c'est-à-dire
    #    strictement avant le jour de référence. Tout le reste est une fuite temporelle.
    ev = ev[ev["date"] < reference]
    debut_fenetre = reference - pd.Timedelta(days=fenetre_jours)

    appels = ev[(ev["type"] == "appel_support") & (ev["date"] >= debut_fenetre)].groupby("customer_id").size()
    reclamations = ev[ev["type"] == "reclamation"].groupby("customer_id").size()
    dernier = ev.groupby("customer_id")["date"].max()

    # 2) Une ligne par client à scorer, dans l'ordre demandé, même sans aucun événement :
    #    `reindex` aligne sur cette liste (et ignore les clients qui n'y sont pas).
    index = pd.Index(clients, name="customer_id")
    return pd.DataFrame({
        "appels_recents": appels.reindex(index, fill_value=0).astype(int),
        "reclamations": reclamations.reindex(index, fill_value=0).astype(int),
        # Aucun événement : NaN (« jamais vu »), pas 0 (« vu aujourd'hui »).
        "jours_depuis_dernier": (reference - dernier.reindex(index)).dt.days.astype(float),
    }, index=index)

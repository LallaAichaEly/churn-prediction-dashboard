"""Série 08 — Assembler les données : texte, jointures, JSON, mises à jour, événements.

🔗  APRÈS LA SÉRIE 07 : LES DONNÉES N'ARRIVENT JAMAIS EN UN SEUL TABLEAU PROPRE.

Dans le projet, `data/customers.csv` est déjà prêt. En entreprise, il faut le CONSTRUIRE : les notes
du support sont du texte libre, les factures sont dans une autre table, le CRM répond en JSON,
une mise à jour arrive chaque nuit, les appels au support sont un journal d'événements datés. Cette
série entraîne les gestes qui transforment ces sources en une table fiable pour le modèle, et les
erreurs silencieuses qui vont avec : une jointure qui duplique des clients, un champ JSON absent
qui fait planter le script, une mise à jour qui garde l'ancienne version, un événement « du futur »
qui rend le modèle faussement excellent.

Prérequis : la série 00 (surtout `groupby`). La série 07 est conseillée avant (dates, `set`).

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py --defi 50` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, essayez dans un terminal Python, puis seulement en dernier
   recours ouvrez `defis/corrige_serie_08.py`.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import json  # défi 52
import math  # défi 52
import re  # défi 50
from datetime import date  # défi 55

import numpy as np  # défi 53
import pandas as pd

# ============================================================================ Texte


def extraire_identifiants(texte: str | None) -> list[str]:
    """Défi 50 🟢 — Retrouver les clients cités dans une note du support (expressions régulières).

    Les conseillers écrivent des notes en texte libre. Pour relier chaque note à un client, il faut
    y retrouver les identifiants, de la forme « C » suivie d'exactement 5 chiffres :

        extraire_identifiants("Le client C00123 a rappelé, voir aussi c00456.")  → ["C00123", "C00456"]

    Règles :
    - le « c » peut être en minuscule : renvoyez toujours la forme en MAJUSCULE (« C00456 ») ;
    - un identifiant est un MOT entier : rien dans « AC00123 » (un code agence) ni dans « C001234 »
      (l'ancienne numérotation à 6 chiffres) ;
    - la ponctuation collée ne gêne pas : « (C00123) », « C00123, » ou « C00123/C00124 » comptent ;
    - chaque client une seule fois, dans l'ordre de PREMIÈRE apparition ;
    - renvoyez une liste (vide s'il n'y a rien) ; une note vide arrive en "", en None, ou en NaN
      (une cellule vide de pandas, quand on fait `df["note"].apply(extraire_identifiants)`).

    Ce que vous apprenez : les expressions régulières (module `re`), l'outil universel pour
    extraire une information d'un texte : `re.findall`, `\\d{5}`, la frontière de mot `\\b`,
    l'option `re.IGNORECASE`. Et pourquoi `texte.split()` ne suffit pas.

    Indice : `re.findall(r"\\bC\\d{5}\\b", texte, flags=re.IGNORECASE)` fait presque tout. Pour
    dédoublonner en gardant l'ordre : `list(dict.fromkeys(liste))`. Pour la note vide :
    `isinstance(texte, str)`.
    """
    raise NotImplementedError


# ============================================================================ Jointures


def ajouter_total_factures(clients: pd.DataFrame, factures: pd.DataFrame) -> pd.DataFrame:
    """Défi 51 🟢 — Joindre deux tables sans dupliquer ni perdre de clients.

    `clients` a une ligne par client ; `factures` une ligne par facture (colonnes `customer_id`
    et `montant`). Ajoutez à `clients` une colonne `total_factures` : la somme de ses factures.

        clients                         factures                   résultat
           customer_id contract            customer_id  montant       customer_id contract total_factures
        0  C1          Mensuel             C1           50.0       0  C1          Mensuel          80.0
        1  C2          Annuel              C1           30.0       1  C2          Annuel            0.0
        2  C3          Mensuel             C3           20.0       2  C3          Mensuel          20.0
                                           C9           99.0

    Renvoyez un NOUVEAU DataFrame qui :
    - a exactement une ligne par client de `clients`, dans le même ordre, avec le même INDEX
      (après un filtrage, l'index ne vaut pas 0, 1, 2… : gardez-le tel quel) ;
    - garde toutes les colonnes de `clients` et ajoute `total_factures` (un float) ;
    - met 0.0 aux clients sans facture (C2), et ignore les factures de clients inconnus (C9) ;
    - ne modifie pas `clients` ni `factures`.

    Ce que vous apprenez : la jointure, l'opération la plus fréquente… et la plus traîtresse.
    `pd.merge` fait par défaut une jointure INTERNE (les clients sans facture disparaissent), joindre
    AVANT d'agréger duplique chaque client autant de fois qu'il a de factures, et `merge` renumérote
    l'index. Aucune de ces erreurs ne plante : on s'en rend compte (ou pas) des semaines plus tard.

    Indice : agrégez d'abord, `factures.groupby("customer_id")["montant"].sum()` (une ligne par
    client). Puis `clients["customer_id"].map(totaux)` va chercher le total de chaque client en
    gardant l'index de `clients`. Il reste les NaN (`.fillna(0.0)`) et la copie (`clients.assign(...)`).
    """
    raise NotImplementedError


# ============================================================================ JSON


def aplatir_reponse_api(reponse: dict | str) -> pd.DataFrame:
    """Défi 52 🟡 — Transformer la réponse JSON d'une API en tableau, sans planter sur un champ vide.

    Le CRM répond à `GET /clients?page=1` par un JSON imbriqué. Les champs vides sont tantôt
    ABSENTS, tantôt `null` (None en Python) :

        {"page": 1,
         "clients": [
           {"id": "C00001",
            "contrat": {"type": "Mensuel", "debut": "2023-01-15"},
            "factures": [{"mois": "2024-01", "montant": 89.9}, {"mois": "2024-02", "montant": 92.1}],
            "options": ["fibre", "tv"]},
           {"id": "C00002", "contrat": null, "factures": [], "options": null},
           {"id": "C00003"}
         ]}

    Renvoyez un DataFrame avec une ligne par client (dans l'ordre de l'API) et exactement ces colonnes :

        customer_id  contrat  nb_factures  facture_moyenne  nb_options
        C00001       Mensuel  2            91.0             2
        C00002       None     0            NaN              0
        C00003       None     0            NaN              0

    - `contrat` : le `type` du contrat, None si le contrat est absent ou null ;
    - `nb_factures`, `nb_options` : le nombre d'éléments, 0 si la liste est absente, null ou vide ;
    - `facture_moyenne` : la moyenne des `montant`, NaN (et non 0) s'il n'y a aucune facture ;
    - `reponse` est soit un dictionnaire, soit le TEXTE brut de la réponse HTTP (une chaîne JSON) ;
    - sans clé "clients" (ou une liste vide), renvoyez un DataFrame vide qui a quand même ces colonnes ;
    - ne modifiez pas `reponse`. L'identifiant `id` est toujours présent.

    Ce que vous apprenez : parcourir des données imbriquées (dictionnaires de listes de
    dictionnaires), `json.loads`, et surtout `dict.get` avec une valeur par défaut. Une API ne
    garantit presque jamais que tous les champs sont là : un script qui fait `client["contrat"]["type"]`
    marche sur le premier client et plante au millième.

    Indice : `client.get("contrat") or {}` donne {} que la clé soit absente OU null. Construisez une
    liste de dictionnaires (un par client), puis `pd.DataFrame(lignes, columns=[...])`.
    `math.nan` est un NaN.
    """
    raise NotImplementedError


# ============================================================================ Vectorisation

COLONNES_RISQUE = ["contract", "support_calls", "tenure_months", "tech_support"]


def niveau_risque_lent(df: pd.DataFrame) -> pd.Series:
    """Le code actuel de l'équipe (à NE PAS modifier) : juste, mais ~5 s pour 200 000 clients.

    Une version vectorisée met 0,04 s. Et le vrai barème compte 40 règles et tourne chaque nuit.
    """
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
    """Défi 53 🟡 — Vectoriser : même résultat que `niveau_risque_lent`, cent fois plus vite.

    Lisez `niveau_risque_lent` juste au-dessus : un barème de points (contrat mensuel +3, 3 appels
    ou plus +2, moins de 6 mois d'ancienneté +1, support technique −2), puis un niveau « élevé »
    (4 points ou plus), « moyen » (2 ou 3) ou « faible ». Il est juste, mais il traite les clients
    UN PAR UN avec `iterrows` : avec le vrai barème, le traitement de nuit dépasse son créneau.

    Écrivez `niveau_risque(df)` qui :
    - renvoie EXACTEMENT la même Series que `niveau_risque_lent(df)` (mêmes valeurs, même index) ;
    - ne parcourt pas les lignes une par une (ni `iterrows`, ni `itertuples`, ni `apply(axis=1)`,
      ni boucle `for` sur les lignes) : 50 000 clients en moins d'une demi-seconde ;
    - ne modifie pas `df` (pas de colonne ajoutée).

    Ce que vous apprenez : la vectorisation, LE réflexe de performance avec pandas et numpy. Une
    comparaison sur une colonne entière (`df["support_calls"] >= 3`) est exécutée en C, d'un coup,
    au lieu de 50 000 tours de boucle Python. True compte pour 1 dans un calcul.

    Indice : `points = 3 * (df["contract"] == "Mensuel") + 2 * (df["support_calls"] >= 3) + ...`
    Puis `np.select([points >= 4, points >= 2], ["élevé", "moyen"], default="faible")` fait le
    if / elif / else d'un coup. Emballez le résultat : `pd.Series(..., index=df.index)`.
    """
    raise NotImplementedError


# ============================================================================ Mise à jour


def mettre_a_jour(table: pd.DataFrame, nouveautes: pd.DataFrame, cle: str = "customer_id") -> pd.DataFrame:
    """Défi 54 🟡 — Appliquer la mise à jour de la nuit (un « upsert ») sans dupliquer ni perdre.

    Chaque nuit, le CRM envoie les clients créés ou modifiés dans la journée (`nouveautes`, mêmes
    colonnes que `table`, éventuellement dans un autre ordre). Renvoyez la table à jour :

    - un client présent dans les deux : sa ligne est REMPLACÉE par la nouvelle version, y compris
      quand une valeur devient vide (un client qui retire son email : la nouvelle valeur None doit
      remplacer l'ancienne, c'est une obligation légale) ;
    - un nouveau client : il est ajouté ;
    - un client absent des nouveautés : il reste tel quel ;
    - si un client apparaît plusieurs fois dans `nouveautes` (modifié deux fois dans la journée),
      c'est la DERNIÈRE ligne qui compte ;
    - à la fin, une seule ligne par client, les colonnes dans l'ordre de `table`, un index
      renuméroté 0, 1, 2… ; l'ordre des lignes est libre ;
    - rejouer deux fois la même mise à jour (le traitement relancé après une panne) donne la même
      table qu'une seule fois ;
    - levez une ValueError si `table` contient déjà une clé en double (la référence est corrompue :
      il faut le voir tout de suite, pas l'aggraver) ;
    - ne modifiez ni `table` ni `nouveautes`. `cle` est le nom de la colonne identifiant.

    Ce que vous apprenez : l'« upsert » (update + insert), le pain quotidien du data engineering, et
    l'IDEMPOTENCE : un traitement qu'on peut relancer sans risque. Ainsi que les pièges de
    `pd.concat` (il empile, il ne remplace pas), de `drop_duplicates` (sur quelles colonnes ?
    quelle ligne garder ?) et de `DataFrame.update` (qui ignore les valeurs vides).

    Indice : `pd.concat([table, nouveautes[table.columns]], ignore_index=True)` met l'ancienne
    version AVANT la nouvelle. Il reste à ne garder que la dernière ligne de chaque client :
    regardez les paramètres `subset` et `keep` de `drop_duplicates`. Le garde-fou :
    `table[cle].duplicated().any()`.
    """
    raise NotImplementedError


# ============================================================================ Événements


def variables_evenements(
    evenements: pd.DataFrame,
    clients: list[str],
    date_reference: str | date,
    fenetre_jours: int = 30,
) -> pd.DataFrame:
    """Défi 55 🔴 — Fabriquer des variables à partir d'un journal d'événements, sans fuite temporelle.

    `evenements` est le journal du CRM : une ligne par événement, colonnes `customer_id`, `date`
    (texte « AAAA-MM-JJ » ou dates pandas) et `type` (« appel_support », « reclamation »,
    « connexion »…). Le modèle tourne le matin du `date_reference` (texte « AAAA-MM-JJ » ou `date`)
    pour les clients de la liste `clients`. Renvoyez un DataFrame :

    - indexé par les identifiants de `clients`, dans cet ordre : une ligne par client, même s'il n'a
      aucun événement (et aucune ligne pour les autres clients du journal) ;
    - avec exactement ces trois colonnes :
        `appels_recents` (int)        nombre d'« appel_support » dans les `fenetre_jours` jours qui
                                      précèdent : du jour `date_reference - fenetre_jours` INCLUS
                                      à la veille de `date_reference` incluse ;
        `reclamations` (int)          nombre de « reclamation » dans tout l'historique ;
        `jours_depuis_dernier` (float) jours entre le dernier événement (tout type) et
                                      `date_reference` ; NaN si le client n'a aucun événement.
    - RÈGLE D'OR : un événement daté du `date_reference` ou après N'EXISTE PAS (le matin de la
      prédiction, on ne le connaît pas encore). Il ne compte nulle part.
    - ne modifiez pas `evenements`.

    Exemple, pour une référence au 15/03/2024 et une fenêtre de 30 jours (du 14/02 au 14/03 inclus) :
    un appel le 14/02 compte, un appel le 13/02 ne compte pas, un appel le 15/03 ne compte pas.

    Ce que vous apprenez : le feature engineering sur des données datées (filtres de dates,
    `groupby(...).size()`, `reindex`) et la FUITE TEMPORELLE, l'erreur la plus coûteuse en
    prédiction de churn : compter les appels « de la semaine du départ » rend le modèle excellent
    au test… et inutile en production, où ces appels n'ont pas encore eu lieu.

    Indice : `reference = pd.Timestamp(date_reference)`, et
    `ev = evenements.assign(date=pd.to_datetime(evenements["date"]))` (une copie, dates converties).
    Filtrez d'abord `ev["date"] < reference`. La fenêtre commence à
    `reference - pd.Timedelta(days=fenetre_jours)`. Comptez avec `groupby("customer_id").size()`,
    puis `.reindex(clients, fill_value=0)` pour les clients sans événement. Le dernier événement :
    `groupby("customer_id")["date"].max()` ; `(reference - dernier).dt.days` donne les jours.
    """
    raise NotImplementedError

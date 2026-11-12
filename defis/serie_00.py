"""Série 00 — Remise à niveau : bases de Python et premiers pas en pandas.

👋 VOUS DÉBUTEZ, OU VOUS NE VOUS SENTEZ PAS ENCORE « À NIVEAU » ? COMMENCEZ ICI.

Pas besoin de savoir déjà faire de la data science : ces 8 défis entraînent les briques
que tout le reste du projet utilise (fonctions, conditions, boucles, listes, dictionnaires,
puis filtrer et regrouper un tableau pandas). Comptez 20 à 40 minutes par défi au début :
c'est normal, et c'est exactement là qu'on progresse.

✍️  C'EST VOTRE FICHIER. Mode d'emploi :

1. Remplacez chaque `raise NotImplementedError` par votre code (le défi 34 est différent :
   le code est déjà écrit, mais il contient des bugs à trouver).
2. Lancez `python scripts/defis.py --defi 30` (le numéro du défi) : ✅ si c'est juste, sinon
   ❌ avec un message qui dit QUOI corriger. Lisez-le en entier : il est écrit pour vous aider.
3. Corrigez, relancez, recommencez. Faire 5 essais avant d'y arriver, c'est le métier.
4. Bloqué·e plus de 20 minutes ? Testez votre fonction à la main (voir ci-dessous), relisez
   l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_00.py`.

Pour essayer une fonction à la main, ouvrez un terminal Python À LA RACINE du projet :

    $ python
    >>> from defis.serie_00 import taux_en_pourcentage
    >>> taux_en_pourcentage(35, 100)
    35.0

(Après chaque modification du fichier, quittez Python avec exit() et relancez-le.)

Niveaux : 🟢 débutant · 🟡 intermédiaire.
"""

from __future__ import annotations

import pandas as pd

# Table de correspondance du défi 37 : clé = texte nettoyé, valeur = nom officiel du contrat.
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
    """Défi 30 🟢 — Votre première fonction : un taux de churn en pourcentage.

    Renvoie le pourcentage de clients partis, arrondi à 1 décimale.
    S'il n'y a aucun client (total == 0), renvoie 0.0 au lieu de planter.

    Exemples :
        taux_en_pourcentage(35, 100)  → 35.0
        taux_en_pourcentage(1, 3)     → 33.3
        taux_en_pourcentage(0, 0)     → 0.0

    Ce que vous apprenez : écrire une fonction, la différence entre `print` et `return`,
    traiter un cas particulier AVANT qu'il ne provoque une erreur.

    Rappel de syntaxe :
        if condition:
            return valeur        # la fonction s'arrête ici et renvoie `valeur`
        round(3.14159, 2)        # → 3.14
    """
    raise NotImplementedError


def segment_client(anciennete_mois: int) -> str:
    """Défi 31 🟢 — Classer un client selon son ancienneté.

    - moins de 6 mois          → "nouveau"
    - de 6 à 23 mois           → "installé"
    - 24 mois ou plus          → "fidèle"
    - ancienneté négative      → lever une erreur : raise ValueError("…votre message…")

    Ce que vous apprenez : if / elif / else, et surtout les BORNES (6 est-il « nouveau » ?).
    Dans ce projet, les nouveaux clients sont ceux qui partent le plus : ce découpage sert vraiment.

    Rappel : Python teste les conditions dans l'ordre et s'arrête à la première vraie.
        if x < 0:
            ...
        elif x < 10:
            ...
        else:
            ...
    """
    raise NotImplementedError


def compter_par_contrat(contrats: list[str]) -> dict[str, int]:
    """Défi 32 🟢 — Compter combien de clients ont chaque type de contrat.

    Exemple :
        compter_par_contrat(["Mensuel", "Annuel", "Mensuel"])  → {"Mensuel": 2, "Annuel": 1}
        compter_par_contrat([])                                → {}

    Ce que vous apprenez : la boucle `for` et le dictionnaire, la structure la plus utilisée
    en Python. (C'est ce que fait `df["contract"].value_counts()` en pandas.)

    Rappel :
        compte = {}                         # un dictionnaire vide
        compte["Mensuel"] = 1               # ajouter / remplacer une valeur
        compte.get("Annuel", 0)             # la valeur, ou 0 si la clé n'existe pas
        for contrat in contrats:            # parcourir une liste
            ...
    """
    raise NotImplementedError


def clients_a_risque(clients: list[dict], seuil: float = 0.5) -> list[str]:
    """Défi 33 🟢 — La liste d'appels de l'équipe de rétention.

    `clients` est une liste de dictionnaires, par exemple :
        [{"id": "C001", "proba": 0.30}, {"id": "C002", "proba": 0.91}, {"id": "C004", "proba": 0.72}]

    Renvoie les IDENTIFIANTS des clients dont la probabilité de départ est >= `seuil`,
    du plus risqué au moins risqué :
        clients_a_risque(clients)  → ["C002", "C004"]

    Ce que vous apprenez : filtrer une liste, trier selon un critère, ne pas modifier les
    données qu'on vous confie.

    Rappel :
        [x for x in liste if condition]                  # filtrer (compréhension de liste)
        sorted(liste, key=lambda c: c["proba"])          # trier sur la clé "proba" (croissant)
        sorted(..., reverse=True)                        # décroissant
    """
    raise NotImplementedError


def facture_moyenne(factures: list[float]) -> float:
    """Défi 34 🟢 — 🐞 DÉBOGAGE : ce code est déjà écrit… mais il est faux.

    Il doit renvoyer la moyenne des factures, et 0.0 si la liste est vide :
        facture_moyenne([10.0, 20.0, 60.0])  → 30.0
        facture_moyenne([])                  → 0.0

    Il contient 3 bugs. Lancez `python scripts/defis.py --defi 34` : le message pointe le
    premier. Corrigez-le, relancez, passez au suivant.

    Ce que vous apprenez : lire un message d'erreur, et déboguer méthodiquement. Un·e
    développeur·se passe plus de temps à lire et corriger du code qu'à en écrire du neuf.

    Astuce : ajoutez temporairement `print(i, total)` dans la boucle pour VOIR ce qui se passe,
    puis retirez-le une fois le bug trouvé.
    """
    total = 0
    for i in range(1, len(factures)):
        total = factures[i]
    return total / len(factures)


# ============================================================================ Premiers pas en pandas
#
# Un DataFrame est un tableau : une ligne par client, une colonne par information.
# Pour en avoir un sous la main dans un terminal Python :
#     >>> import sys; sys.path.insert(0, "src")
#     >>> from churn.data import generate_customers
#     >>> df = generate_customers(1000)
#     >>> df.head()                      # les 5 premières lignes
#     >>> df["contract"]                 # une colonne (une Series)
#     >>> df["tenure_months"] <= 6       # une Series de True / False


def filtrer_clients(df: pd.DataFrame, contrat: str, anciennete_max: int) -> pd.DataFrame:
    """Défi 35 🟢 — Garder les lignes qui vérifient DEUX conditions.

    Renvoie les clients dont le contrat (colonne "contract") vaut `contrat` ET dont
    l'ancienneté (colonne "tenure_months") est inférieure ou égale à `anciennete_max`.

    Exemple : filtrer_clients(df, "Mensuel", 6) → les clients au contrat mensuel arrivés depuis
    6 mois au plus, le groupe le plus à risque du projet.

    Ce que vous apprenez : le filtrage par masque booléen, LA compétence pandas de base.

    Rappel :
        df[df["contract"] == "Mensuel"]                      # une condition
        df[(condition_1) & (condition_2)]                    # deux conditions : & et des parenthèses
    """
    raise NotImplementedError


def taux_par_groupe_a_la_main(df: pd.DataFrame, colonne: str) -> dict:
    """Défi 36 🟡 (Parcours 1.2) — Comprendre `groupby` en le refaisant à la main.

    Renvoie un dictionnaire {modalité: taux de churn}, par exemple pour colonne="contract" :
        {"Mensuel": 0.52, "Annuel": 0.18, "Biannuel": 0.07}

    ⚠️ SANS utiliser groupby : une boucle sur les valeurs distinctes de la colonne, un filtre,
    une moyenne. Le taux de churn d'un groupe est la moyenne de sa colonne "churn" (0 ou 1).

    Une fois ✅, comparez avec `df.groupby(colonne)["churn"].mean()` : une ligne au lieu de
    cinq. Vous saurez désormais ce qu'elle fait pour vous (découper → calculer → rassembler).

    Rappel :
        df[colonne].unique()         # les valeurs distinctes
        float(serie.mean())          # la moyenne, en nombre Python ordinaire
    """
    raise NotImplementedError


def nettoyer_contrats(contrats: pd.Series) -> pd.Series:
    """Défi 37 🟡 — Nettoyer des données saisies à la main (comme dans la vraie vie).

    Un export du service client contient des contrats écrits n'importe comment :
        [" mensuel", "MENSUEL", "Annuel ", "1 an", "Bi-Annuel", "2 ans", "mois", None]

    Renvoie une Series avec les noms officiels ("Mensuel", "Annuel", "Biannuel"), en :
    1. enlevant les espaces au début et à la fin, et en passant tout en minuscules ;
    2. traduisant avec le dictionnaire SYNONYMES (en haut de ce fichier) ;
    3. gardant les valeurs manquantes (None / NaN) manquantes, sans erreur ;
    4. levant une ValueError qui CITE les valeurs inconnues (ex. "trimestriel") : une valeur
       qu'on ne sait pas traduire ne doit pas disparaître en silence.

    Le résultat garde le même index que `contrats`, et `contrats` n'est pas modifiée.

    Rappel :
        serie.str.strip()            # enlève les espaces (sur chaque valeur d'une Series)
        serie.str.lower()            # minuscules
        serie.map(dictionnaire)      # traduit ; ce qui n'est pas dans le dictionnaire → NaN
        serie.isna() / serie.notna() # masques des valeurs manquantes / présentes
    """
    raise NotImplementedError

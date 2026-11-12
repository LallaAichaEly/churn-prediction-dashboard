"""Série 07 — Écrire du code réutilisable : dates, fichiers, générateurs, classes.

🧱  APRÈS LA SÉRIE 06 : PASSER DU SCRIPT AU CODE QU'UNE ÉQUIPE RÉUTILISE.

La série 06 vous a appris à lire une erreur, tester et refactoriser. Celle-ci entraîne ce qu'on
vous demandera dès votre première vraie tâche : des dates venues d'un CRM, un fichier envoyé par
le marketing depuis Excel, une liste de 200 000 clients qui rend le script interminable, un
fichier trop gros pour la mémoire, un objet qui garde un état, un calcul lent à ne pas refaire.
Chaque défi correspond à une notion de Python qu'on croise partout : `datetime`, `set`,
encodages, `yield`, `class`, décorateurs.

Prérequis : la série 00. La série 06 est conseillée avant (surtout le défi 40).

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py --defi 44` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, essayez dans un terminal Python, puis seulement en dernier
   recours ouvrez `defis/corrige_serie_07.py`.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import calendar  # défi 44
import functools  # défi 49
from datetime import date  # défi 44
from pathlib import Path  # défi 46
from typing import Callable, Iterable, Iterator

import pandas as pd

# ============================================================================ Dates


def anciennete_en_mois(debut: date | str, fin: date | str) -> int:
    """Défi 44 🟢 — Calculer une ancienneté en mois COMPLETS à partir de deux dates.

    Le modèle attend `tenure_months`, mais le CRM ne donne que la date de souscription. Il faut
    donc compter les mois complets entre `debut` (souscription) et `fin` (aujourd'hui) :

        anciennete_en_mois("2024-01-15", "2024-03-14")  → 1   (le 15 mars n'est pas encore atteint)
        anciennete_en_mois("2024-01-15", "2024-03-15")  → 2
        anciennete_en_mois("2023-11-20", "2024-02-20")  → 3   (on change d'année)
        anciennete_en_mois("2024-01-31", "2024-02-29")  → 1   (fin de mois : voir ci-dessous)
        anciennete_en_mois("2024-01-31", "2024-02-28")  → 0

    Règles :
    - chaque date est soit un objet `date`, soit un texte « AAAA-MM-JJ » (une colonne de CSV),
      éventuellement entouré d'espaces ;
    - un mois est complet quand on atteint le même numéro de jour que `debut`… ou, si ce jour
      n'existe pas dans le mois de `fin` (un 31 en février), le DERNIER jour de ce mois ;
    - renvoyez un entier (int) ;
    - levez une ValueError si `fin` est avant `debut`, ou si un texte n'est pas une vraie date
      au format AAAA-MM-JJ (« 15/01/2024 », « 2024-02-30 »).

    Ce que vous apprenez : le module `datetime`, et pourquoi on ne divise pas un nombre de jours
    par 30 (un mois a 28, 29, 30 ou 31 jours).

    Indice : `date.fromisoformat(texte.strip())` convertit le texte (et lève ValueError s'il est
    invalide). Nombre de mois du calendrier : (fin.year - debut.year) * 12 + (fin.month - debut.month).
    Retirez 1 si le dernier mois n'est pas complet. `calendar.monthrange(annee, mois)[1]` donne le
    nombre de jours d'un mois.
    """
    raise NotImplementedError


# ============================================================================ Complexité


def clients_toujours_a_risque(mois_dernier: list[str], ce_mois: list[str]) -> list[str]:
    """Défi 45 🟢 — Le bon conteneur : passer de 3 heures à 30 millisecondes.

    Le mois dernier, une campagne de rétention a ciblé les clients à risque. Ce mois-ci, lesquels
    sont ENCORE à risque ? Renvoyez les identifiants présents dans les deux listes :

        clients_toujours_a_risque(["C1", "C2", "C3", "C4"], ["C9", "C3", "C1", "C7"])  → ["C3", "C1"]

    - dans l'ordre de `ce_mois` (l'équipe rétention traite la liste dans cet ordre) ;
    - chaque client une seule fois, même s'il apparaît plusieurs fois dans une liste ;
    - sans modifier les listes reçues ;
    - et RAPIDEMENT : en vrai, chaque liste compte 200 000 clients.

    Ce que vous apprenez : la complexité d'un algorithme. `x in une_liste` parcourt toute la liste ;
    dans une boucle de 200 000 tours, cela fait 200 000 × 200 000 = 40 milliards de comparaisons.
    `x in un_set` répond presque instantanément, quelle que soit la taille du set. C'est la cause
    n° 1 des scripts « qui marchaient sur l'échantillon » et ne finissent jamais en production.

    Indice : convertissez `mois_dernier` en set UNE fois, avant la boucle. Pour ne pas ajouter un
    client deux fois, retenez ceux déjà ajoutés… dans un autre set (pas dans la liste résultat).
    """
    raise NotImplementedError


# ============================================================================ Fichiers


def lire_export_excel(chemin: str | Path) -> pd.DataFrame:
    """Défi 46 🟡 — Lire le fichier que le marketing a exporté depuis Excel (version française).

    Ouvert dans un éditeur de texte, le fichier ressemble à ceci :

        id;Contrat;Facture mensuelle ;Appels support;Paiement
        00123;Mensuel;89,90;3;Chèque
        00456;Annuel;45,50;0;Carte bancaire

    Avec `pd.read_csv(chemin)` tout court, tout est faux. Renvoyez un DataFrame où :
    - chaque colonne est séparée (Excel en français utilise « ; », car « , » sert aux décimales) ;
    - « Facture mensuelle » et « Appels support » sont des NOMBRES (89.9, 3) ;
    - les noms de colonnes n'ont pas d'espace autour (« Facture mensuelle », sans l'espace final) ;
    - `id` reste du TEXTE : « 00123 » ne doit pas devenir 123 ;
    - les accents sont justes (« Chèque », « Prélèvement »), quel que soit l'encodage du fichier.
      Selon la version d'Excel, il est en UTF-8, en UTF-8 avec BOM (un caractère invisible au
      début), ou en Windows-1252 (cp1252).

    `chemin` peut être un texte ou un `Path`.

    Ce que vous apprenez : les options de `pd.read_csv` (`sep`, `decimal`, `encoding`, `dtype`)
    et ce qu'est un encodage. Lire correctement un fichier « venu d'ailleurs » est la première
    tâche de presque tout projet de données.

    Indice : `encoding="utf-8-sig"` lit l'UTF-8 avec ou sans BOM. Le Windows-1252 provoque alors
    une `UnicodeDecodeError` : attrapez-la et relisez avec `encoding="cp1252"`. Pourquoi pas
    cp1252 d'abord ? Essayez : `"Chèque".encode("utf-8").decode("cp1252")`.
    """
    raise NotImplementedError


# ============================================================================ Générateurs


def par_lots(elements: Iterable, taille: int) -> Iterator[list]:
    """Défi 47 🟡 (Parcours 6.4) — Traiter un flux par lots, sans tout charger en mémoire.

    Pour noter 50 millions de lignes, on envoie les clients au modèle (ou à l'API) par paquets
    de 1 000 : assez gros pour être efficace, assez petit pour tenir en mémoire.

        list(par_lots(["a", "b", "c", "d", "e"], 2))  → [["a", "b"], ["c", "d"], ["e"]]
        list(par_lots([], 3))                         → []

    - chaque lot est une liste de `taille` éléments ; le dernier peut être plus petit, mais
      jamais vide ;
    - `elements` peut être n'importe quel itérable : liste, range, générateur, lignes d'un
      fichier ouvert, `df.itertuples()`… Certains n'ont ni len() ni indices [i:j] ;
    - la fonction est un GÉNÉRATEUR (`yield`) : pour donner le premier lot, elle ne lit que les
      premiers éléments, pas tout le flux ;
    - une `taille` inférieure à 1 lève une ValueError.

    Ce que vous apprenez : `yield`, la différence entre une liste (tout en mémoire) et un
    générateur (un élément à la fois, à la demande), et le protocole d'itération de Python.

    Indice :
        lot = []
        for element in elements:
            lot.append(element)
            if len(lot) == taille:
                yield lot
                ...                # repartez d'une NOUVELLE liste
        ...                        # et le reste ?
    """
    raise NotImplementedError


# ============================================================================ Classes


class JournalPredictions:
    """Défi 48 🟡 — Une classe qui garde un état : le journal des scores du mois.

    L'API calcule des scores toute la journée. On veut garder le DERNIER score de chaque client
    et sortir à tout moment la liste des alertes :

        journal = JournalPredictions(seuil=0.5)
        journal.ajouter("C4", 0.9)
        journal.ajouter("C2", 0.2)
        journal.ajouter("C1", 0.9)
        len(journal)            → 3
        journal.a_risque()      → ["C1", "C4"]       (0.9 et 0.9 : égalité → ordre des id)
        journal.taux_alerte()   → 0.666…             (2 clients à risque sur 3)
        journal.ajouter("C1", 0.1)                   (son nouveau score REMPLACE l'ancien)
        journal.a_risque()      → ["C4"]

    Complétez les méthodes ci-dessous. Attendus :
    - `seuil` (0.5 par défaut) est gardé dans l'attribut `self.seuil` ; un seuil hors de [0, 1]
      lève une ValueError ;
    - `ajouter` refuse (ValueError) une probabilité hors de [0, 1], y compris nan, et n'enregistre
      alors RIEN ;
    - `a_risque()` : les clients dont la probabilité est ≥ seuil, de la plus forte à la plus
      faible, à égalité par identifiant ;
    - si on change `journal.seuil` après coup, `a_risque()` et `taux_alerte()` en tiennent compte ;
    - `taux_alerte()` : la part des clients à risque, entre 0 et 1 (0.0 si le journal est vide) ;
    - deux journaux (janvier, février) sont totalement indépendants.

    Ce que vous apprenez : `__init__` et `self`, la différence entre un attribut d'OBJET et un
    attribut de CLASSE (partagé par tous les objets : un bug célèbre), les méthodes spéciales
    comme `__len__`, et « stocker les données, calculer le reste à la demande ».

    Indice : un dictionnaire `{client: proba}` créé dans `__init__`. Pour le tri :
    `sorted(…, key=lambda paire: (-paire[1], paire[0]))`.
    """

    def __init__(self, seuil: float = 0.5) -> None:
        raise NotImplementedError

    def ajouter(self, client_id: str, proba: float) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def a_risque(self) -> list[str]:
        raise NotImplementedError

    def taux_alerte(self) -> float:
        raise NotImplementedError


# ============================================================================ Décorateurs


def avec_cache(fonction: Callable) -> Callable:
    """Défi 49 🔴 — Écrire un décorateur : ne jamais calculer deux fois la même chose.

    Interroger le modèle prend du temps, et le tableau de bord redemande souvent les mêmes
    clients. Écrivez un décorateur qui mémorise les résultats :

        @avec_cache
        def score_client(client_id, seuil=0.5):
            ...                                   # lent

        score_client("C1")              # calcule
        score_client("C1")              # réutilise le résultat, sans appeler la fonction
        score_client("C2")              # calcule
        score_client.statistiques()     → {"calculs": 2, "reutilisations": 1}

    `avec_cache(fonction)` renvoie une NOUVELLE fonction qui :
    - renvoie la même chose que `fonction` ;
    - n'appelle `fonction` qu'une fois par combinaison d'arguments (positionnels ET nommés :
      `score_client("C1", seuil=0.3)` et `score_client("C1", seuil=0.7)` sont deux calculs) ;
    - met aussi en cache les résultats qui valent 0.0 ou None (un client à 0 % existe) ;
    - ne met PAS en cache une erreur : si `fonction` lève une exception, elle remonte à
      l'appelant, et l'appel suivant recalcule ;
    - a une méthode `statistiques()` qui renvoie {"calculs": …, "reutilisations": …} ;
    - garde le nom et la documentation de `fonction` (`__name__`, `__doc__`).
    Deux fonctions décorées ont chacune leur propre cache.

    Ce que vous apprenez : une fonction est un objet comme un autre (on peut la passer, la
    renvoyer, lui ajouter un attribut), les fonctions imbriquées et leur « fermeture » (closure),
    `*args, **kwargs`, et `functools.wraps`. C'est exactement ce que fait `@st.cache_data` dans
    `app.py` (et `@functools.cache` dans la bibliothèque standard, que vous utiliserez en vrai).

    Indice :
        def avec_cache(fonction):
            cache = {}                                  # un par fonction décorée
            @functools.wraps(fonction)
            def enveloppe(*args, **kwargs):
                cle = (args, tuple(sorted(kwargs.items())))
                ...
            enveloppe.statistiques = ...
            return enveloppe
    """
    raise NotImplementedError

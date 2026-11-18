"""Étape 3 🟡 — Ranger les lignes dans la base : la version la plus récente gagne, toujours.

Vérifier : python scripts/projet.py --projet 6 --etape 3
"""

from __future__ import annotations

import sqlite3

from .modeles import Contrat, Ticket  # noqa: F401


def charger_contrats(connexion: sqlite3.Connection, contrats: list[Contrat]) -> int:
    """Range des contrats dans la table `contrats` et renvoie le nombre de lignes créées ou modifiées.

    Lisez d'abord `SCHEMA` dans `modeles.py`. Pour chaque contrat :

    - s'il n'est pas encore dans la base : on l'ajoute ;
    - s'il y est déjà avec une `maj` PLUS ANCIENNE : on remplace toutes ses colonnes ;
    - s'il y est déjà avec une `maj` plus récente ou égale : on ne touche à rien. Un vieux fichier
      rechargé par erreur ne doit jamais remettre un contrat dans un état ancien (c'est le défaut de
      `INSERT OR REPLACE`, qui remplace toujours). Et recharger les mêmes lignes ne change rien.

    SQLite le fait en UNE requête, un « upsert » :

        INSERT INTO contrats (client, …, maj) VALUES (?, …, ?)
        ON CONFLICT (client) DO UPDATE SET debut = excluded.debut, …, maj = excluded.maj
        WHERE excluded.maj > contrats.maj

    (`excluded` désigne la ligne qu'on essayait d'insérer). Avec `connexion.executemany(requete,
    liste_de_tuples)`, une seule requête pour toutes les lignes, et des `?` pour les valeurs : jamais
    de f-string dans une requête SQL.

    Les dates vont en base en TEXTE ISO : `jour.isoformat()` → "2026-06-30" et
    `instant.isoformat(timespec="seconds")` → "2026-06-30T15:04:00". Ne laissez pas sqlite3 convertir
    seul les objets `date` et `datetime` : il écrirait "2026-06-30 15:04:00" (une espace au lieu du
    T), qui ne se compare plus correctement aux autres textes, et ce convertisseur est déprécié
    depuis Python 3.12. `fin` à `None` devient NULL tout seul.

    Le nombre renvoyé : les lignes réellement créées ou modifiées. `connexion.total_changes` compte
    les lignes modifiées depuis l'ouverture de la connexion : regardez-le avant et après.

    Ne validez PAS la transaction (pas de `commit()`, pas de `with connexion:`) : c'est l'appelant
    (l'étape 4) qui décide quand elle se termine, pour pouvoir tout annuler si la suite échoue.

    Ce que vous apprenez : l'upsert `ON CONFLICT … DO UPDATE … WHERE`, `executemany` et les requêtes
    paramétrées, ranger des dates dans SQLite, rendre une écriture rejouable sans danger
    (idempotente), savoir à qui appartient une transaction.
    """
    raise NotImplementedError


def charger_tickets(connexion: sqlite3.Connection, tickets: list[Ticket]) -> int:
    """La même chose pour la table `tickets` (clé : `ticket`) : ajoutée si nouvelle, remplacée seulement
    par une version de `maj` plus récente. Renvoie le nombre de lignes créées ou modifiées, sans
    `commit()`.
    """
    raise NotImplementedError

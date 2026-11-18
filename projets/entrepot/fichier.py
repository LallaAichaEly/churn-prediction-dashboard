"""Étape 4 🔴 — Charger un fichier une fois, en entier ou pas du tout.

Vérifier : python scripts/projet.py --projet 6 --etape 4
"""

from __future__ import annotations

import sqlite3
from datetime import datetime  # noqa: F401 — vous en aurez besoin
from pathlib import Path

from .chargement import charger_contrats, charger_tickets  # noqa: F401
from .contrats import lire_contrats  # noqa: F401
from .modeles import CHARGE, DEJA_CHARGE, REFUSE, SEUIL_REJETS, Bilan, empreinte  # noqa: F401
from .tickets import lire_tickets  # noqa: F401


def charger_fichier(connexion: sqlite3.Connection, chemin: str | Path) -> Bilan:
    """Charge UN export dans la base et renvoie son `Bilan(fichier, statut, lignes, rejets)`.

    Le programme tourne chaque nuit sur le dossier des exports, qui garde tous les fichiers des mois
    passés. Il doit donc pouvoir être relancé sans risque : un fichier déjà chargé ne l'est pas deux
    fois, et une panne au milieu (disque plein, coupure) ne laisse jamais la base à moitié remplie.

    Dans cet ordre :

    1. Le nom dit ce qu'est le fichier : `crm_*.csv` → `lire_contrats` puis `charger_contrats` ;
       `support_*.jsonl` → `lire_tickets` puis `charger_tickets`. Tout autre nom : `ValueError`.
    2. `empreinte(chemin)` (fournie dans `modeles.py`). Si la table `fichiers` a déjà une ligne avec
       ce NOM et cette EMPREINTE : renvoyez `Bilan(nom, DEJA_CHARGE, 0, [])` sans même relire le
       fichier. Même nom mais empreinte différente : le fichier a été corrigé et renvoyé, on le
       recharge (l'upsert de l'étape 3 garde la version la plus récente de chaque ligne).
    3. Lisez-le. Une `ValueError` de la lecture (mauvais en-tête) remonte telle quelle.
    4. Si la part de lignes écartées dépasse `SEUIL_REJETS` (rejets / (lignes + rejets) > 10 %) :
       renvoyez `Bilan(nom, REFUSE, 0, rejets)` sans RIEN écrire. Un fichier aussi abîmé cache
       sûrement un problème plus grave (mauvais séparateur, colonnes décalées) : mieux vaut attendre
       sa correction. Exactement 10 % passe. Un fichier sans aucune ligne n'est pas une division par
       zéro : il se charge, avec 0 ligne.
    5. Dans UNE SEULE transaction : rangez les lignes (étape 3) ET notez le fichier dans la table
       `fichiers` (nom, empreinte, `charge_le` = maintenant en texte ISO, nombre de lignes changées,
       nombre de rejets ; si le nom y est déjà, mettez sa ligne à jour). Le plus simple :

           with connexion:      # commit à la sortie du bloc… ou rollback si une exception survient
               ...

       Si quoi que ce soit échoue au milieu, rien ne doit rester : ni la moitié des lignes, ni le nom
       du fichier dans `fichiers` (sinon, la nuit suivante, il passerait pour « déjà chargé » et ses
       lignes manquantes ne viendraient jamais). L'exception remonte : ne l'attrapez pas.
    6. Renvoyez `Bilan(nom, CHARGE, <lignes changées>, rejets)`.

    `nom` est le nom du fichier, sans le dossier (`Path(chemin).name`).

    Ce que vous apprenez : rendre un traitement rejouable (idempotence) grâce à une empreinte, une
    transaction qui fait tout ou rien, un seuil de qualité qui arrête la chaîne, l'ordre des
    opérations quand chaque ligne peut échouer.
    """
    raise NotImplementedError

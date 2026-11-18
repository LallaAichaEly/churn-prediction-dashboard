"""Étape 2 🟡 — Lire l'export du support : une ligne JSON par ticket, et chaque ticket une seule fois.

Vérifier : python scripts/projet.py --projet 6 --etape 2
"""

from __future__ import annotations

import json  # noqa: F401 — vous en aurez besoin
import re  # noqa: F401
from datetime import datetime  # noqa: F401
from pathlib import Path

from .modeles import CHAMPS_TICKET, FORMAT_CLIENT, FORMAT_TICKET, Lecture, Rejet, Ticket  # noqa: F401


def lire_tickets(chemin: str | Path) -> Lecture:
    """Lit un export du support (`support_AAAA-MM-JJ.jsonl`) et renvoie `Lecture(lignes=[Ticket…], rejets=[Rejet…])`.

    Le format JSON Lines met UN objet JSON par ligne (lisez `CHAMPS_TICKET` dans `modeles.py`). On le
    lit donc ligne par ligne, avec `json.loads` sur chaque ligne, et pas avec `json.load` sur tout
    le fichier (ce n'est pas UN document JSON). Ouvrez-le avec `encoding="utf-8"` (« résolu » a un
    accent) et numérotez les lignes à partir de 1, lignes vides comprises (`enumerate(f, start=1)`).

    Une ligne vide est ignorée. Une ligne devient un `Ticket` si :

    1. c'est du JSON valide. L'export est parfois coupé au milieu de la dernière ligne :
       `json.JSONDecodeError` ne doit pas arrêter la lecture. Rejet : « JSON illisible (...) » ;
    2. c'est un objet `{...}`, pas une liste ni un nombre (`isinstance(objet, dict)`) ;
    3. les quatre clés de `CHAMPS_TICKET` sont présentes (les autres clés, comme `statut`, sont
       ignorées) : la raison cite la ou les clés manquantes ;
    4. `ticket` et `client` sont des TEXTES qui respectent `FORMAT_TICKET` et `FORMAT_CLIENT`
       (attention : `re.fullmatch` plante sur le nombre 42, vérifiez d'abord le type) ;
    5. `ouvert` et `maj` sont des dates et heures ISO (`datetime.fromisoformat`), et `maj` n'est
       pas avant `ouvert`.

    Comme à l'étape 1, chaque rejet est un `Rejet(fichier, ligne, raison)` dont la raison commence
    par le nom de la clé fautive (« client : identifiant illisible 42 »).

    Le piège du format : un ticket est RENVOYÉ à chaque mise à jour, avec la même clé `ticket` et une
    `maj` plus récente. Ne gardez qu'UN `Ticket` par identifiant : celui dont la `maj` est la plus
    récente, PAS le dernier lu (les lignes n'arrivent pas toujours dans l'ordre). Un dictionnaire
    `{identifiant: ticket}` fait très bien l'affaire. Renvoyez les tickets triés par identifiant, et
    les rejets dans l'ordre du fichier.

    Ce que vous apprenez : le format JSON Lines, `json.loads` et ses erreurs, vérifier le type de ce
    qu'on reçoit, dédoublonner en gardant la version la plus récente.
    """
    raise NotImplementedError

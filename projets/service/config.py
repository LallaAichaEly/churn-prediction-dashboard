"""Étape 1 🟢 — Lire la configuration dans les variables d'environnement, sans jamais montrer un secret.

Vérifier : python scripts/projet.py --projet 4 --etape 1
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401 — vous en aurez besoin
from typing import Mapping

from .modeles import (  # noqa: F401
    BASE_PAR_DEFAUT,
    LONGUEUR_CLE_MIN,
    TAILLE_PAGE_MAX,
    TAILLE_PAGE_PAR_DEFAUT,
    Config,
    ConfigInvalide,
)


def lire_config(env: Mapping[str, str]) -> Config:
    """Lit la configuration du service dans `env` et renvoie un `Config` (modeles.py).

    En production, on appellera `lire_config(os.environ)` une seule fois, au démarrage (étape 6).
    Ici, lisez UNIQUEMENT `env`, jamais `os.environ` : c'est ce qui rend la fonction testable avec
    un simple dict. Une variable absente, vide ou faite d'espaces compte comme absente.

    Les trois variables :

    - `SERVICE_CLES` (obligatoire) : un conseiller par entrée, séparées par des virgules, sous la
      forme `nom=clé`. Exemple : `"alice=cle-alice-0123456789, bruno=YWJjZGVmZ2hpamtsbW5vcA=="`.
      - Retirez les espaces autour des noms et des clés ; ignorez les entrées vides (« alice=…, »).
      - Coupez chaque entrée au PREMIER « = » seulement : une clé en base64 se termine souvent
        par « == » (cherchez `str.partition`).
      - Sont des erreurs : une entrée sans « = », un nom vide ou qui contient une espace, un
        conseiller présent deux fois, une clé de moins de `LONGUEUR_CLE_MIN` caractères, la même
        clé pour deux conseillers (on ne saurait pas qui appelle), aucun conseiller du tout.
      - Le résultat va dans `cles` : un dict {conseiller: clé}.
    - `SERVICE_BASE` : le chemin de la base, `BASE_PAR_DEFAUT` par défaut. Un `Path`, pas un texte.
    - `SERVICE_TAILLE_PAGE` : la taille des pages de l'API, un ENTIER entre 1 et `TAILLE_PAGE_MAX`,
      `TAILLE_PAGE_PAR_DEFAUT` par défaut. « 2.5 » ou « vingt » sont des erreurs.

    Les erreurs :

    1. Ne vous arrêtez pas à la première : rassemblez-les dans une liste, puis levez UNE fois
       `ConfigInvalide("Configuration invalide : " + " ; ".join(erreurs))`. L'équipe corrige tout
       en un seul redémarrage.
    2. Chaque message nomme la variable à corriger (« SERVICE_TAILLE_PAGE : entier entre 1 et 100
       attendu, reçu 'vingt' »).
    3. JAMAIS une clé dans un message : il finira dans un journal, une capture d'écran, un ticket.
       Nommez le conseiller (« la clé de 'alice' est trop courte »), ou la position de l'entrée.

    Ce que vous apprenez : configuration par variables d'environnement, conversions prudentes,
    rassembler les erreurs, hygiène des secrets. (C'est le défi 86 de la série 14, en vrai.)
    """
    raise NotImplementedError

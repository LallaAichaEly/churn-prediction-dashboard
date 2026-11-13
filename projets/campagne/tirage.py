"""Étape 1 🟢 — Tirer le groupe témoin au hasard… de façon reproductible.

Vérifier : python scripts/projet.py --projet 3 --etape 1
"""

from __future__ import annotations

import hashlib  # noqa: F401 — vous en aurez besoin

from .modeles import GROUPE_OFFRE, GROUPE_TEMOIN, PART_TEMOIN  # noqa: F401


def tirage(customer_id: str, graine: str) -> float:
    """Renvoie un nombre « au hasard » entre 0 (inclus) et 1 (exclu), TOUJOURS le même pour un
    même client et une même graine.

    La recette, à suivre exactement (d'autres programmes de l'entreprise doivent pouvoir refaire
    le même tirage, et le correcteur compare vos nombres aux siens) :

    1. Fabriquez le texte `"{graine}:{customer_id}"`, par exemple "2026-11:C10042".
    2. Calculez son empreinte SHA-256 (`hashlib.sha256`) à partir de ses octets en UTF-8
       (`texte.encode("utf-8")`), sous forme hexadécimale (`.hexdigest()` : 64 caractères 0-9a-f).
    3. Prenez les 16 PREMIERS caractères (8 octets) et lisez-les comme un entier en base 16
       (`int(..., 16)`) : un entier entre 0 et 2**64 − 1.
    4. Divisez par 2**64.

    Pourquoi pas `random.random()` ? Relancer le programme changerait tous les groupes : un client
    contacté hier deviendrait témoin aujourd'hui. Pourquoi pas `random.seed(graine)` puis
    `random.shuffle(clients)` ? Ajouter UN client au plan décalerait tout le monde. Ici, le groupe
    d'un client ne dépend que de lui et de la graine : c'est ainsi que les grandes plateformes
    répartissent leurs utilisateurs dans les tests A/B.

    Ce que vous apprenez : une fonction de hachage, les octets contre le texte (`encode`), la base 16,
    la reproductibilité.

    Piège : la fonction `hash()` de Python semble faire l'affaire, mais pour un texte elle change à
    chaque lancement du programme (cherchez « PYTHONHASHSEED »). Le correcteur le vérifie.
    """
    raise NotImplementedError


def groupe(customer_id: str, graine: str, part_temoin: float = PART_TEMOIN) -> str:
    """Renvoie le groupe du client : `GROUPE_TEMOIN` si `tirage(customer_id, graine) < part_temoin`,
    sinon `GROUPE_OFFRE`.

    1. Si `part_temoin` n'est pas entre 0 et 1 (bornes incluses), levez `ValueError`.
       Pensez à `float("nan")` : `nan < 0` et `nan > 1` sont tous les deux faux…
    2. Avec `part_temoin=0.2`, environ 20 % des clients sont témoins ; avec 0, aucun ; avec 1, tous.
    3. Renvoyez les constantes de `modeles.py` (`GROUPE_TEMOIN`, `GROUPE_OFFRE`), pas un texte
       retapé à la main : une faute de frappe (« témoin » avec un accent) serait refusée par la base.

    Ce que vous apprenez : réutiliser sa propre fonction, valider un paramètre, les constantes.
    """
    raise NotImplementedError

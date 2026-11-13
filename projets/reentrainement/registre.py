"""Étape 5 🔴 — Le registre des modèles : chaque version gardée, vérifiée, réversible.

Vérifier : python scripts/projet.py --projet 5 --etape 5
"""

from __future__ import annotations

import hashlib  # noqa: F401 — vous en aurez besoin
import io  # noqa: F401
import os  # noqa: F401
import tempfile  # noqa: F401
from pathlib import Path

import joblib  # noqa: F401

from .fichiers import ecrire_index, lire_index  # noqa: F401
from .modeles import Mesures, ModeleAltere, Version, nom_fichier  # noqa: F401

# Avant ce projet, le modèle vivait dans UN fichier, `models/churn_model.joblib`, écrasé à chaque
# entraînement. Le jour où un mauvais modèle est parti en production, impossible de revenir en
# arrière : l'ancien n'existait plus. Le registre garde TOUTES les versions dans un dossier
# (v001.joblib, v002.joblib…), décrites par un index (`registre.json`, voir modeles.py) que
# `lire_index` et `ecrire_index` (fichiers.py, déjà écrits) lisent et écrivent.


def enregistrer(dossier: str | Path, modele, mesures: Mesures) -> Version:
    """Ajoute `modele` au registre et renvoie sa `Version`. N'en fait PAS le champion.

    1. Créez le dossier s'il n'existe pas (`mkdir(parents=True, exist_ok=True)`).
    2. Le numéro : le plus grand numéro de l'index + 1 (1 pour le premier). Pas `len + 1` : si une
       version a été retirée de l'index, `len + 1` réutiliserait un numéro et écraserait un fichier.
       Le nom du fichier : `nom_fichier(numero)` (« v001.joblib »).
    3. Les octets du modèle : `joblib.dump(modele, tampon)` dans un `io.BytesIO()`, puis
       `tampon.getvalue()`. L'empreinte : `hashlib.sha256(contenu).hexdigest()`.
    4. Écriture ATOMIQUE (série 14, défi 90) : écrivez les octets dans un fichier temporaire du MÊME
       dossier (`tempfile.mkstemp(dir=dossier)`), puis `os.replace(temporaire, dossier / fichier)`.
       Si une erreur survient (disque plein), effacez le temporaire et laissez l'erreur remonter :
       un fichier de modèle tronqué ne doit jamais exister sous son vrai nom.
    5. L'index EN DERNIER : ajoutez `version.vers_dict()` à `index["versions"]`, puis
       `ecrire_index(dossier, index)`. Tant que l'index n'est pas écrit, la version n'existe pas.

    Ce que vous apprenez : versionner des artefacts, l'empreinte d'un fichier, l'écriture atomique,
    l'ordre des écritures quand le programme peut s'arrêter à tout moment.
    """
    raise NotImplementedError


def charger(dossier: str | Path, numero: int):
    """Renvoie le modèle de la version `numero`, après avoir vérifié qu'il n'a pas été modifié.

    1. Cherchez la version dans l'index ; inconnue → `ValueError`.
    2. Lisez les octets du fichier UNE fois (`read_bytes()`), calculez leur SHA-256, et comparez-la
       à celle de l'index. Différente → `ModeleAltere` : quelqu'un (ou un disque abîmé) a modifié
       le fichier.
    3. Seulement ensuite, chargez CES MÊMES octets : `joblib.load(io.BytesIO(contenu))`. Ne rouvrez
       pas le fichier : il pourrait avoir été échangé entre la vérification et le chargement.

    Pourquoi tant de soin ? Un fichier joblib (pickle) peut EXÉCUTER du code au moment où on le
    charge (série 13, défi 84). Le correcteur remplace un modèle par un fichier piégé : votre
    `charger` doit refuser AVANT de l'ouvrir avec joblib.

    Ce que vous apprenez : vérifier l'intégrité avant de faire confiance, « vérifier puis utiliser »
    sur les mêmes octets.
    """
    raise NotImplementedError


def promouvoir(dossier: str | Path, numero: int) -> None:
    """Fait de la version `numero` le champion : ajoute son numéro à la fin de `index["champions"]`.

    - Une version inconnue → `ValueError`.
    - Si elle est déjà le champion actuel (dernier élément), ne changez rien : pas de doublon dans
      l'historique (sinon `revenir` reviendrait… à elle-même).
    - Écrivez l'index avec `ecrire_index`.
    """
    raise NotImplementedError


def revenir(dossier: str | Path) -> Version:
    """Remet en production le champion PRÉCÉDENT et renvoie sa `Version` : c'est le bouton « annuler ».

    - Retirez le dernier élément de `index["champions"]` ; le champion est alors le nouvel avant-dernier.
    - S'il y a moins de deux champions dans l'historique, il n'y a pas de précédent : `ValueError`,
      SANS toucher à l'index.
    - Aucun fichier n'est supprimé : la version retirée reste dans `index["versions"]`, on pourra la
      promouvoir à nouveau plus tard.

    Ce que vous apprenez : une opération réversible, l'historique plutôt que l'écrasement.
    """
    raise NotImplementedError

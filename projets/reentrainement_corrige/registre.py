"""Étape 5 (corrigé) — Le registre des modèles : chaque version gardée, vérifiée, réversible."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

import joblib

from .fichiers import ecrire_index, lire_index
from .modeles import Mesures, ModeleAltere, Version, nom_fichier


def _entree(index: dict, numero: int) -> dict:
    entree = next((v for v in index["versions"] if v["numero"] == numero), None)
    if entree is None:
        connus = [v["numero"] for v in index["versions"]]
        raise ValueError(f"version {numero} inconnue (versions du registre : {connus or 'aucune'})")
    return entree


def enregistrer(dossier: str | Path, modele, mesures: Mesures) -> Version:
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    index = lire_index(dossier)
    # max + 1, pas len + 1 : si une version a été retirée à la main, len + 1 réutiliserait un numéro.
    numero = max((v["numero"] for v in index["versions"]), default=0) + 1
    fichier = nom_fichier(numero)

    # Les octets d'abord, en mémoire : l'empreinte est celle de CE QUI EST ÉCRIT, exactement.
    tampon = io.BytesIO()
    joblib.dump(modele, tampon)
    contenu = tampon.getvalue()

    # Écriture atomique : un fichier temporaire du même dossier, puis os.replace. Un disque plein au
    # milieu laisse au pire un fichier temporaire (effacé), jamais un v003.joblib tronqué.
    descripteur, temporaire = tempfile.mkstemp(dir=dossier, prefix=".modele-", suffix=".tmp")
    try:
        with os.fdopen(descripteur, "wb") as sortie:
            sortie.write(contenu)
            sortie.flush()
            os.fsync(sortie.fileno())
        os.replace(temporaire, dossier / fichier)
    except BaseException:
        Path(temporaire).unlink(missing_ok=True)
        raise

    # L'index en DERNIER : tant qu'il n'est pas écrit, la version n'existe pas pour le reste du programme.
    version = Version(numero, fichier, hashlib.sha256(contenu).hexdigest(), mesures)
    index["versions"].append(version.vers_dict())
    ecrire_index(dossier, index)
    return version


def charger(dossier: str | Path, numero: int):
    dossier = Path(dossier)
    entree = _entree(lire_index(dossier), numero)
    # On lit les octets UNE fois, on vérifie CES octets, et on charge CES octets. Vérifier le fichier
    # puis le rouvrir pour joblib.load laisserait un instant pour l'échanger entre les deux.
    contenu = (dossier / entree["fichier"]).read_bytes()
    if hashlib.sha256(contenu).hexdigest() != entree["sha256"]:
        raise ModeleAltere(f"{entree['fichier']} a été modifié depuis son enregistrement : chargement refusé")
    return joblib.load(io.BytesIO(contenu))


def promouvoir(dossier: str | Path, numero: int) -> None:
    index = lire_index(dossier)
    _entree(index, numero)                 # une version inconnue ne peut pas devenir championne
    if index["champions"] and index["champions"][-1] == numero:
        return                             # déjà championne : rien à changer, pas de doublon dans l'historique
    index["champions"].append(numero)
    ecrire_index(dossier, index)


def revenir(dossier: str | Path) -> Version:
    index = lire_index(dossier)
    if len(index["champions"]) < 2:
        raise ValueError("aucun champion précédent : impossible de revenir en arrière")
    index["champions"].pop()
    ecrire_index(dossier, index)
    return Version.depuis_dict(_entree(index, index["champions"][-1]))

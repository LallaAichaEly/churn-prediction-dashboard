"""Lire les lots, lire et écrire l'index du registre, mettre en forme (DÉJÀ ÉCRIT : à lire).

Ces fonctions ne sont pas le cœur du projet, mais elles sont de bons exemples à imiter : lecture d'un
CSV sans laisser pandas deviner les types qui comptent, écriture atomique d'un fichier JSON, nombres au
format français.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from .modeles import IDENTIFIANT, INDEX, MOIS, Mesures, Version


def lire_lots(chemin: str | Path) -> pd.DataFrame:
    """Lit le fichier des lots mensuels SANS rien vérifier : c'est le travail de `problemes` (étape 1).

    `mois` et `customer_id` sont lus comme du texte : sinon pandas lirait « 2026-09 » tel quel, mais
    un identifiant « 00042 » deviendrait le nombre 42. Une case vide devient NaN (valeur manquante) :
    `problemes` la signalera.
    """
    return pd.read_csv(chemin, dtype={MOIS: str, IDENTIFIANT: str}, encoding="utf-8-sig")


def lire_index(dossier: str | Path) -> dict:
    """L'index du registre (`registre.json`), ou un registre vide s'il n'existe pas encore."""
    chemin = Path(dossier) / INDEX
    if not chemin.exists():
        return {"versions": [], "champions": []}
    return json.loads(chemin.read_text(encoding="utf-8"))


def ecrire_index(dossier: str | Path, index: dict) -> None:
    """Écrit l'index de façon ATOMIQUE : un fichier temporaire dans le même dossier, puis `os.replace`.

    Si le programme s'arrête au milieu (coupure, disque plein), l'ancien index reste intact : on ne
    se retrouve jamais avec un `registre.json` à moitié écrit, illisible, qui ferait perdre la trace
    de tous les modèles. `os.replace` est atomique quand les deux fichiers sont sur le même disque :
    d'où le fichier temporaire dans le MÊME dossier (série 14, défi 90).
    """
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    descripteur, temporaire = tempfile.mkstemp(dir=dossier, prefix=".index-", suffix=".tmp")
    try:
        with os.fdopen(descripteur, "w", encoding="utf-8") as fichier:
            json.dump(index, fichier, ensure_ascii=False, indent=2)
            fichier.flush()
            os.fsync(fichier.fileno())
        os.replace(temporaire, dossier / INDEX)
    except BaseException:
        Path(temporaire).unlink(missing_ok=True)
        raise


def versions(dossier: str | Path) -> list[Version]:
    """Toutes les versions du registre, dans l'ordre où elles ont été enregistrées."""
    return [Version.depuis_dict(v) for v in lire_index(dossier)["versions"]]


def champion(dossier: str | Path) -> Version | None:
    """La version en production (la dernière promue), ou None si aucun modèle n'a encore été promu."""
    index = lire_index(dossier)
    if not index["champions"]:
        return None
    numero = index["champions"][-1]
    return next(Version.depuis_dict(v) for v in index["versions"] if v["numero"] == numero)


# ---------------------------------------------------------------------------- Mise en forme


def nombre(valeur: float, decimales: int = 3) -> str:
    """0.8291 → "0,829" : la virgule française."""
    return f"{valeur:.{decimales}f}".replace(".", ",").replace("-", "−")


def euros(valeur: float) -> str:
    return f"{nombre(valeur, 2)} €"


def pourcentage(valeur: float) -> str:
    return f"{nombre(100 * valeur, 1)} %"


def tableau(lignes: list[tuple[str, Mesures]]) -> str:
    """Les mesures de plusieurs modèles sur le même mois, l'une sous l'autre :

                         AUC   coût/client    alertes
        champion v001  0,781       61,20 €     38,0 %
        challenger     0,829       48,10 €     41,2 %
    """
    largeur = max([len(nom) for nom, _ in lignes] + [10])
    texte = [f"{'':<{largeur}}  {'AUC':>6}  {'coût/client':>12}  {'alertes':>8}"]
    for nom, m in lignes:
        texte.append(f"{nom:<{largeur}}  {nombre(m.auc):>6}  {euros(m.cout):>12}  {pourcentage(m.taux_alerte):>8}")
    return "\n".join(texte)

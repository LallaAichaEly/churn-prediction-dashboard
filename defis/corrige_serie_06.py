"""Série 06 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_06.py`.

Si votre version passe les vérifications, elle est juste, même si elle ne ressemble pas à
celle-ci. Comparez : notez dans JOURNAL.md l'écriture ou le réflexe que vous ne connaissiez pas.
"""

from __future__ import annotations

import argparse
import math
from typing import Callable

import pandas as pd

# ============================================================================ Lire une erreur


def analyser_traceback(texte: str) -> dict:
    lignes = [ligne for ligne in texte.splitlines() if ligne.strip()]
    dernier_appel = None
    for ligne in lignes:
        # `  File "/projet/src/churn/data.py", line 97, in validate`
        if ligne.strip().startswith('File "'):
            dernier_appel = ligne.strip()  # on écrase : à la fin, il reste le PLUS RÉCENT
    if dernier_appel is None:
        raise ValueError("Aucune ligne « File ... » : est-ce bien une traceback ?")

    fichier, reste = dernier_appel[len('File "'):].split('", line ', 1)
    numero, fonction = reste.split(", in ", 1)

    # Dernière ligne : `KeyError: 'contrat'`. On coupe au PREMIER « : » seulement, car le message
    # peut en contenir d'autres (`ValueError: could not convert string to float: 'abc'`).
    erreur, _, message = lignes[-1].strip().partition(":")
    return {
        "fichier": fichier,
        "ligne": int(numero),
        "fonction": fonction,
        "erreur": erreur,
        "message": message.strip(),
    }


# ============================================================================ Écrire un test


def taux_en_pourcentage(partis: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(100 * partis / total, 1)


def tester_taux_en_pourcentage(taux_en_pourcentage: Callable[[int, int], float]) -> None:
    # Un cas « normal » : attrape une fonction qui renvoie une proportion (0.35) au lieu d'un %.
    assert taux_en_pourcentage(35, 100) == 35.0, "35 partis sur 100 → 35.0 %"
    # Un DEUXIÈME cas normal : attrape une fonction qui renverrait toujours la même valeur.
    assert taux_en_pourcentage(1, 4) == 25.0, "1 sur 4 → 25.0 %"
    # Une valeur « pas ronde » : vérifie l'arrondi à 1 décimale (ni 33, ni 33.333…).
    assert taux_en_pourcentage(1, 3) == 33.3, "1 sur 3 → 33.3 % (arrondi à 1 décimale)"
    # Le cas limite : aucun client ne doit pas faire planter (ZeroDivisionError).
    assert taux_en_pourcentage(0, 0) == 0.0, "aucun client → 0.0, sans planter"
    # Pourquoi `==` sur des floats marche ici : la fonction PROMET un arrondi à 1 décimale, donc
    # 33.3 exactement. Pour un calcul non arrondi, on utiliserait math.isclose ou pytest.approx.


# ============================================================================ Entrées utilisateur


def lire_seuil(texte: str | None, defaut: float = 0.5) -> float:
    if texte is None or not texte.strip():
        return defaut  # rien saisi : la valeur par défaut, c'est le seul cas « silencieux »

    nettoye = texte.strip().replace(",", ".")
    pourcentage = nettoye.endswith("%")
    if pourcentage:
        nettoye = nettoye[:-1].strip()

    try:
        valeur = float(nettoye)
    except ValueError:  # l'exception PRÉCISE : un autre bug ne sera pas caché
        raise ValueError(f"Seuil illisible : {texte!r} (exemples valides : 0.4, 0,4, 40 %)") from None
    if pourcentage:
        valeur /= 100

    # `not 0 <= valeur <= 1` refuse aussi NaN (toute comparaison avec NaN est fausse)…
    # … alors que `valeur < 0 or valeur > 1` le laisserait passer. math.isfinite le dit clairement.
    if not math.isfinite(valeur) or not 0 <= valeur <= 1:
        raise ValueError(f"Le seuil doit être entre 0 et 1 (ou 0 et 100 %), reçu : {texte!r}")
    return valeur


# ============================================================================ Refactoriser


def resume_kpis_ancien(df: pd.DataFrame) -> pd.DataFrame:
    partis = df[df["churn"] == 1]
    restes = df[df["churn"] == 0]
    lignes = []
    m1 = partis["monthly_charges"].mean()
    m0 = restes["monthly_charges"].mean()
    lignes.append({"variable": "monthly_charges", "moyenne_partis": round(m1, 2),
                   "moyenne_restes": round(m0, 2), "ecart": round(m1 - m0, 2)})
    a1 = partis["tenure_months"].mean()
    a0 = restes["tenure_months"].mean()
    lignes.append({"variable": "tenure_months", "moyenne_partis": round(a1, 2),
                   "moyenne_restes": round(a0, 2), "ecart": round(a1 - a0, 2)})
    s1 = partis["support_calls"].mean()
    s0 = restes["support_calls"].mean()
    lignes.append({"variable": "support_calls", "moyenne_partis": round(s1, 2),
                   "moyenne_restes": round(s0, 2), "ecart": round(s1 - s0, 2)})
    return pd.DataFrame(lignes)


def _comparer(partis: pd.Series, restes: pd.Series, variable: str) -> dict:
    """Le bloc qui était copié 3 fois, écrit une seule fois."""
    m1, m0 = partis.mean(), restes.mean()
    # L'écart se calcule sur les moyennes NON arrondies, comme dans l'ancien code :
    # round(m1, 2) - round(m0, 2) donnerait parfois 12.340000000000003 au lieu de 12.34.
    return {"variable": variable, "moyenne_partis": round(m1, 2),
            "moyenne_restes": round(m0, 2), "ecart": round(m1 - m0, 2)}


def resume_kpis(
    df: pd.DataFrame,
    colonnes: list[str] | tuple[str, ...] = ("monthly_charges", "tenure_months", "support_calls"),
) -> pd.DataFrame:
    partis = df[df["churn"] == 1]
    restes = df[df["churn"] == 0]
    return pd.DataFrame([_comparer(partis[col], restes[col], col) for col in colonnes])


# ============================================================================ Ligne de commande


def creer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Note un fichier de clients et liste les plus à risque.")
    parser.add_argument("fichier", help="chemin du fichier CSV des clients")
    # `type=lire_seuil` : argparse lui passe le texte, et transforme sa ValueError en message
    # d'erreur clair (`argument --seuil: invalid lire_seuil value: '1.5'`). Bonus : « 0,3 » et
    # « 30 % » sont acceptés. On réutilise le code du défi 40 au lieu de le réécrire.
    parser.add_argument("--seuil", type=lire_seuil, default=0.5, help="seuil de décision entre 0 et 1 (défaut : 0.5)")
    parser.add_argument("--modele", choices=["logistic", "gboost"], default="logistic", help="modèle à utiliser")
    parser.add_argument("--top", type=int, default=10, help="nombre de clients à afficher (défaut : 10)")
    parser.add_argument("-v", "--verbose", action="store_true", help="affiche le détail des calculs")
    return parser


# ============================================================================ Trouver quand un bug est apparu


def premier_commit_fautif(commits: list[str], est_bon: Callable[[str], bool]) -> str:
    if len(commits) < 2:
        raise ValueError("Il faut au moins 2 commits : un bon (le premier) et un mauvais (le dernier).")
    # Invariant : commits[bon] est bon, commits[mauvais] est mauvais. Le premier commit mauvais
    # est donc dans ]bon, mauvais]. Chaque test coupe cet intervalle en deux.
    bon, mauvais = 0, len(commits) - 1
    while mauvais - bon > 1:  # voisins : `mauvais` est le premier mauvais
        milieu = (bon + mauvais) // 2  # strictement entre les deux, jamais testé deux fois
        if est_bon(commits[milieu]):
            bon = milieu
        else:
            mauvais = milieu
    return commits[mauvais]

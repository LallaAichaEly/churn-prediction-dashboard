"""Étape 1 (corrigé) — Lire le fichier des clients sans s'arrêter à la première ligne fausse."""

from __future__ import annotations

import csv
from pathlib import Path

from .modeles import COLONNES_REQUISES, CONTRATS, Client


def _texte(ligne: dict, colonne: str) -> str:
    # Une ligne trop courte (export coupé) donne None pour les colonnes absentes : on le traite
    # comme une case vide au lieu de planter sur None.strip().
    return (ligne.get(colonne) or "").strip()


def _nombre(ligne: dict, colonne: str, type_: type):
    texte = _texte(ligne, colonne)
    if not texte:
        raise ValueError(f"{colonne} est vide")
    try:
        return type_(texte)
    except ValueError:
        attendu = "un nombre entier" if type_ is int else "un nombre"
        raise ValueError(f"{colonne} « {texte} » n'est pas {attendu}") from None


def _client(ligne: dict) -> Client:
    """Convertit une ligne du CSV en Client. Lève ValueError, avec un message lisible, si elle est fausse."""
    identifiant = _texte(ligne, "customer_id")
    if not identifiant:
        raise ValueError("customer_id est vide")

    proba = _nombre(ligne, "proba_churn", float)
    # Écrit ainsi, le test refuse aussi `nan` : toute comparaison avec nan est fausse,
    # alors que `proba < 0 or proba > 1` laisserait passer nan.
    if not 0 <= proba <= 1:
        raise ValueError(f"proba_churn doit être entre 0 et 1, pas « {_texte(ligne, 'proba_churn')} »")

    facture = _nombre(ligne, "monthly_charges", float)
    if not facture >= 0:
        raise ValueError(f"monthly_charges ne peut pas être négatif (« {_texte(ligne, 'monthly_charges')} »)")

    anciennete = _nombre(ligne, "tenure_months", int)
    appels = _nombre(ligne, "support_calls", int)
    if anciennete < 0 or appels < 0:
        raise ValueError("tenure_months et support_calls ne peuvent pas être négatifs")

    contrat = _texte(ligne, "contract")
    if contrat not in CONTRATS:
        raise ValueError(f"contract « {contrat} » inconnu (attendu : {', '.join(CONTRATS)})")

    return Client(identifiant, proba, facture, anciennete, contrat, appels)


def lire_clients(chemin: str | Path) -> tuple[list[Client], list[str]]:
    chemin = Path(chemin)
    # utf-8-sig : lit aussi bien un UTF-8 « normal » qu'un export Excel qui commence par une
    # marque invisible (BOM) ; sans lui, la première colonne s'appellerait "﻿customer_id".
    # newline="" : demandé par la documentation du module csv (champs sur plusieurs lignes, Windows).
    with chemin.open(encoding="utf-8-sig", newline="") as fichier:
        lecteur = csv.DictReader(fichier)
        presentes = lecteur.fieldnames or []  # None si le fichier est vide
        manquantes = [c for c in COLONNES_REQUISES if c not in presentes]
        if manquantes:
            raise ValueError(f"{chemin.name} : colonnes manquantes : {', '.join(manquantes)}")

        clients: list[Client] = []
        erreurs: list[str] = []
        deja_vus: set[str] = set()
        for ligne in lecteur:
            numero = lecteur.line_num  # numéro de ligne DANS LE FICHIER (l'en-tête est la ligne 1)
            try:
                client = _client(ligne)
            except ValueError as exc:
                erreurs.append(f"ligne {numero} : {exc}")
                continue
            if client.customer_id in deja_vus:
                erreurs.append(f"ligne {numero} : customer_id {client.customer_id} en double (ligne ignorée)")
                continue
            deja_vus.add(client.customer_id)
            clients.append(client)
    return clients, erreurs

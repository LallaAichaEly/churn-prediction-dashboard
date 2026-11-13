"""Étape 1 (corrigé) — Lire la configuration dans les variables d'environnement, sans jamais montrer un secret."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .modeles import (
    BASE_PAR_DEFAUT,
    LONGUEUR_CLE_MIN,
    TAILLE_PAGE_MAX,
    TAILLE_PAGE_PAR_DEFAUT,
    Config,
    ConfigInvalide,
)


def _valeur(env: Mapping[str, str], nom: str) -> str | None:
    # « SERVICE_BASE= » (vide) dans un fichier .env compte comme absent : on prend le défaut.
    texte = (env.get(nom) or "").strip()
    return texte or None


def _lire_cles(texte: str | None, erreurs: list[str]) -> dict[str, str]:
    if texte is None:
        erreurs.append("SERVICE_CLES : obligatoire (format « alice=clé1,bruno=clé2 »)")
        return {}
    cles: dict[str, str] = {}
    for numero, entree in enumerate(texte.split(","), start=1):
        entree = entree.strip()
        if not entree:
            continue  # « alice=…, » : la virgule finale ne gêne pas
        # partition coupe au PREMIER « = » : une clé en base64 (« …YWJj== ») en contient souvent.
        # entree.split("=") la couperait en morceaux.
        nom, egal, cle = entree.partition("=")
        nom, cle = nom.strip(), cle.strip()
        # Les messages nomment le conseiller (ou la position), JAMAIS la clé : ce message finira
        # dans un journal, que bien plus de gens lisent que le fichier de secrets.
        if not egal or not nom or " " in nom:
            erreurs.append(f"SERVICE_CLES : l'entrée n°{numero} n'a pas la forme « nom=clé »")
        elif nom in cles:
            erreurs.append(f"SERVICE_CLES : le conseiller {nom!r} apparaît deux fois")
        elif len(cle) < LONGUEUR_CLE_MIN:
            erreurs.append(f"SERVICE_CLES : la clé de {nom!r} est trop courte "
                           f"({LONGUEUR_CLE_MIN} caractères au moins)")
        elif cle in cles.values():
            erreurs.append(f"SERVICE_CLES : {nom!r} a la même clé qu'un autre conseiller "
                           "(on ne saurait pas qui appelle)")
        else:
            cles[nom] = cle
    if not cles and not any("SERVICE_CLES" in e for e in erreurs):
        erreurs.append("SERVICE_CLES : aucun conseiller")
    return cles


def lire_config(env: Mapping[str, str]) -> Config:
    # On COLLECTE les erreurs : un seul message dit tout ce qu'il faut corriger avant de relancer.
    erreurs: list[str] = []

    cles = _lire_cles(_valeur(env, "SERVICE_CLES"), erreurs)

    base = Path(_valeur(env, "SERVICE_BASE") or BASE_PAR_DEFAUT)

    taille_page, texte = TAILLE_PAGE_PAR_DEFAUT, _valeur(env, "SERVICE_TAILLE_PAGE")
    if texte is not None:
        try:
            taille_page = int(texte)  # int("2.5") et int("vingt") lèvent ValueError : on ne devine pas
        except ValueError:
            taille_page = 0
        if not 1 <= taille_page <= TAILLE_PAGE_MAX:
            erreurs.append(f"SERVICE_TAILLE_PAGE : entier entre 1 et {TAILLE_PAGE_MAX} attendu, reçu {texte!r}")

    if erreurs:
        raise ConfigInvalide("Configuration invalide : " + " ; ".join(erreurs))
    return Config(base=base, cles=cles, taille_page=taille_page)

    # En production : `lire_config(os.environ)`, une seule fois, au démarrage (étape 6). Recevoir
    # `env` en paramètre plutôt que lire os.environ ici rend la fonction testable avec un simple dict.

"""Étape 1 🟢 — Lire l'export du CRM sans s'arrêter à la première ligne abîmée.

Vérifier : python scripts/projet.py --projet 6 --etape 1
"""

from __future__ import annotations

import csv  # noqa: F401 — vous en aurez besoin
import math  # noqa: F401
import re  # noqa: F401
from datetime import date, datetime  # noqa: F401
from pathlib import Path

from .modeles import COLONNES_CRM, FORMAT_CLIENT, MODALITES, Contrat, Lecture, Rejet  # noqa: F401


def lire_contrats(chemin: str | Path) -> Lecture:
    """Lit un export du CRM (`crm_AAAA-MM-JJ.csv`) et renvoie `Lecture(lignes=[Contrat…], rejets=[Rejet…])`.

    Le format est décrit dans `modeles.py` (lisez `COLONNES_CRM`). Une ligne abîmée ne doit PAS
    arrêter la lecture : on la met de côté dans `rejets`, avec sa raison, et on continue. Sinon une
    seule faute de frappe dans un fichier de 50 000 lignes bloquerait tout le traitement de la nuit.

    Le fichier :

    - s'ouvre avec `encoding="utf-8-sig"` (Excel écrit au début du fichier une marque invisible, la
      BOM ; avec "utf-8", la première colonne s'appellerait "\\ufeffclient") et `newline=""` (le module
      `csv` gère lui-même les fins de ligne \\n ou \\r\\n) ;
    - se lit avec `csv.reader(f, delimiter=";")`. `lecteur.line_num` donne le numéro de la ligne
      qu'on vient de lire (l'en-tête est la ligne 1) ;
    - commence par l'en-tête : s'il n'est pas exactement `COLONNES_CRM`, levez `ValueError` (ce n'est
      pas le bon fichier, inutile d'aller plus loin) ;
    - peut contenir des lignes vides (Excel en ajoute à la fin) : ignorez-les, ce ne sont ni des
      contrats ni des rejets.

    Une ligne devient un `Contrat` (avec les bons types : `date`, `int`, `float`, `datetime`) si :

    1. elle a exactement 11 champs. Sinon, rejet : « 10 champs au lieu de 11 ». Retirez les espaces
       autour de chaque valeur (`strip`) : « Mensuel » et « Mensuel  » sont la même valeur ;
    2. `client` respecte `FORMAT_CLIENT` (`re.fullmatch`) : « TEST01 » est un compte de test ;
    3. `debut` est une vraie date JJ/MM/AAAA : `datetime.strptime(texte, "%d/%m/%Y").date()` (le
       `.date()` compte : on veut un jour, pas un instant) ; « 31/02/2026 » ou « 2026-02-03 » sont
       refusés ;
    4. `fin` est vide (→ `None`) ou une vraie date JJ/MM/AAAA, jamais avant `debut` ;
    5. `contrat`, `paiement`, `internet`, `assistance` et `senior` prennent une des valeurs de
       `MODALITES` (« Mensual » ou « fibre » sont refusés) ;
    6. `produits` est un entier écrit avec des chiffres, au moins 1 (« 0 » ou « 2.5 » sont refusés) ;
    7. `mensualite` est un montant strictement positif écrit avec une virgule (« 79,90 » → 79.9).
       Attention : `float("nan")` et `float("inf")` ne lèvent pas d'erreur ! (`math.isfinite`) ;
    8. `maj` est une date et heure ISO : `datetime.fromisoformat`.

    Chaque rejet est un `Rejet(fichier=<nom du fichier, sans le dossier>, ligne=<numéro>,
    raison=<une phrase>)`, et la raison commence par le NOM de la colonne fautive (« mensualite :
    nombre illisible 'N/A' ») : c'est ce que lira l'équipe du CRM pour corriger. Une ligne a au plus
    un rejet. Les contrats et les rejets restent dans l'ordre du fichier.

    Une astuce qui simplifie tout : écrivez une petite fonction qui transforme UNE ligne en `Contrat`
    et lève `ValueError("colonne : explication")` au premier problème ; la boucle principale fait
    `try: … except ValueError as exc: rejets.append(Rejet(..., str(exc)))`.

    Ce que vous apprenez : lire un vrai CSV d'Excel (BOM, `;`, virgule décimale, dates françaises),
    convertir et valider chaque champ, mettre une ligne en quarantaine au lieu de tout arrêter.
    """
    raise NotImplementedError

"""Série 06 — Développer comme un·e pro : lire une erreur, tester, structurer.

🛠️  VOUS VOULEZ DÉVELOPPER, MAIS VOUS NE VOUS SENTEZ PAS ENCORE COMPÉTENT·E ?

Savoir écrire une formule ne suffit pas pour développer : il faut aussi comprendre une erreur,
prouver que son code marche, résister aux saisies bizarres, nettoyer un code qui a grandi trop
vite, offrir une ligne de commande et retrouver QUAND un bug est apparu. Ce sont les gestes de
tous les jours en entreprise, et ce qui distingue un notebook qui « marche chez moi » d'un
logiciel sur lequel une équipe peut compter.

Prérequis : la série 00 (fonctions, if, boucles). Aucune notion de machine learning.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py --defi 38` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, essayez dans un terminal Python, puis seulement en dernier
   recours ouvrez `defis/corrige_serie_06.py`.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import argparse  # défi 42
from typing import Callable

import pandas as pd

# ============================================================================ Lire une erreur


def analyser_traceback(texte: str) -> dict:
    """Défi 38 🟢 — Lire un message d'erreur Python (une « traceback »).

    Quand un programme plante, Python affiche la liste des appels en cours, puis l'erreur :

        Traceback (most recent call last):
          File "/projet/scripts/train.py", line 52, in <module>
            main()
          File "/projet/scripts/train.py", line 31, in main
            df = load_or_create(ROOT / "data" / "customers.csv")
          File "/projet/src/churn/data.py", line 118, in load_or_create
            return validate(pd.read_csv(path))
          File "/projet/src/churn/data.py", line 97, in validate
            if df["contrat"].isna().any():
               ~~^^^^^^^^^^
        KeyError: 'contrat'

    « most recent call last » : l'appel le PLUS RÉCENT, celui où l'erreur s'est produite, est
    EN BAS. C'est là qu'on regarde d'abord (ici : data.py, ligne 97, la colonne s'appelle
    "contract", pas "contrat").

    Renvoyez un dictionnaire décrivant le DERNIER appel et l'erreur :

        {"fichier": "/projet/src/churn/data.py", "ligne": 97, "fonction": "validate",
         "erreur": "KeyError", "message": "'contrat'"}

    - "ligne" est un entier (int), pas du texte ;
    - "erreur" est le nom avant le premier « : » de la dernière ligne, "message" ce qui suit
      (sans l'espace du début). Une erreur sans message (ex. `StopIteration`) a un message "" ;
    - attention : le message peut lui-même contenir des « : ».

    Ce que vous apprenez : lire une erreur en partant du BAS ; découper du texte
    (`.splitlines()`, `.strip()`, `.startswith()`, `.split(sep, 1)`, `int()`).

    Indice : une ligne d'appel commence par `File "` une fois les espaces retirés. Parcourez
    toutes les lignes et gardez la dernière qui correspond ; la ligne d'erreur, elle, est la
    dernière ligne non vide du texte.
    """
    raise NotImplementedError


# ============================================================================ Écrire un test


def taux_en_pourcentage(partis: int, total: int) -> float:
    """La fonction du défi 30, déjà écrite (et juste). C'est elle que le défi 39 va tester.

    Pourcentage de clients partis, arrondi à 1 décimale ; 0.0 s'il n'y a aucun client.
    """
    if total == 0:
        return 0.0
    return round(100 * partis / total, 1)


def tester_taux_en_pourcentage(taux_en_pourcentage: Callable[[int, int], float]) -> None:
    """Défi 39 🟢 (Parcours 6.1) — Écrire un test qui attrape les bugs.

    Jusqu'ici, c'est le correcteur qui testait VOTRE code. Inversez les rôles : écrivez un test
    pour la fonction `taux_en_pourcentage` décrite juste au-dessus (pourcentage arrondi à
    1 décimale, 0.0 s'il n'y a aucun client).

    Votre fonction reçoit une version de `taux_en_pourcentage` et vérifie qu'elle est juste
    avec des `assert` :

        assert taux_en_pourcentage(35, 100) == 35.0, "35 partis sur 100 → 35 %"

    Le correcteur lance ensuite votre test :
    - sur la version CORRECTE : votre test doit passer (aucune erreur) ;
    - sur 5 versions BOGUÉES, écrites comme le ferait un·e collègue pressé·e : votre test doit
      échouer pour CHACUNE (une AssertionError ou une erreur, peu importe : un test qui casse
      a détecté le problème).

    Ce que vous apprenez : un bon test choisit ses exemples pour piéger les erreurs probables
    (cas limite, valeurs « pas rondes »). Un seul exemple ne suffit presque jamais.
    C'est exactement ce que font les fichiers `tests/test_*.py` du projet, lancés par `pytest`.

    Indice : demandez-vous « quelles erreurs pourrait-on faire en codant cette fonction ? »,
    puis écrivez un assert qui échoue pour chacune. N'appelez que la fonction reçue en paramètre.
    """
    raise NotImplementedError


# ============================================================================ Entrées utilisateur


def lire_seuil(texte: str | None, defaut: float = 0.5) -> float:
    """Défi 40 🟡 — Transformer une saisie humaine en seuil fiable (gérer les exceptions).

    Le seuil de décision (voir l'onglet Performance) arrive souvent sous forme de TEXTE : un
    champ de formulaire, une variable d'environnement, une option de ligne de commande. Les
    humains écrivent « 0,4 », « 40 % » ou rien du tout. Renvoyez un float entre 0 et 1 :

        lire_seuil("0.4")    → 0.4        lire_seuil(" 0,35 ") → 0.35   (virgule française)
        lire_seuil("40%")    → 0.4        lire_seuil("40 %")   → 0.4    (pourcentage)
        lire_seuil(None)     → 0.5        lire_seuil("   ")    → 0.5    (rien saisi → defaut)
        lire_seuil("", 0.3)  → 0.3

    Et levez une ValueError, avec un message qui CITE la saisie, si :
    - le texte n'est pas un nombre : lire_seuil("abc") → ValueError("… 'abc' …") ;
    - le seuil n'est pas entre 0 et 1 (bornes comprises) : "1.5", "-0.1", "150 %" ;
    - le seuil n'est pas un vrai nombre : float("nan") ne plante pas… mais « nan » n'est pas
      un seuil. (Testez `nan < 0` et `nan > 1` dans un terminal : surprise.)

    Ce que vous apprenez : try / except AVEC le nom de l'exception attendue. `except:` tout
    seul (ou `except Exception`) attrape aussi vos propres bugs et les cache.

    Indice :
        try:
            valeur = float(texte_nettoye)
        except ValueError:
            raise ValueError(f"Seuil illisible : {texte!r}") from None
    """
    raise NotImplementedError


# ============================================================================ Refactoriser


def resume_kpis_ancien(df: pd.DataFrame) -> pd.DataFrame:
    """Le code à nettoyer pour le défi 41 (ne le modifiez pas : il sert de référence).

    Écrit à la va-vite, il compare les clients partis et restés sur 3 variables… par
    copier-coller. Ajouter une 4e variable, c'est copier un bloc de plus, et corriger un bug,
    c'est le corriger 3 fois (et en oublier une).
    """
    partis = df[df["churn"] == 1]
    restes = df[df["churn"] == 0]
    lignes = []
    # facture mensuelle
    m1 = partis["monthly_charges"].mean()
    m0 = restes["monthly_charges"].mean()
    lignes.append({"variable": "monthly_charges", "moyenne_partis": round(m1, 2),
                   "moyenne_restes": round(m0, 2), "ecart": round(m1 - m0, 2)})
    # ancienneté
    a1 = partis["tenure_months"].mean()
    a0 = restes["tenure_months"].mean()
    lignes.append({"variable": "tenure_months", "moyenne_partis": round(a1, 2),
                   "moyenne_restes": round(a0, 2), "ecart": round(a1 - a0, 2)})
    # appels au support
    s1 = partis["support_calls"].mean()
    s0 = restes["support_calls"].mean()
    lignes.append({"variable": "support_calls", "moyenne_partis": round(s1, 2),
                   "moyenne_restes": round(s0, 2), "ecart": round(s1 - s0, 2)})
    return pd.DataFrame(lignes)


def resume_kpis(
    df: pd.DataFrame,
    colonnes: list[str] | tuple[str, ...] = ("monthly_charges", "tenure_months", "support_calls"),
) -> pd.DataFrame:
    """Défi 41 🟡 — Refactoriser : même résultat, sans copier-coller.

    Réécrivez `resume_kpis_ancien` (juste au-dessus) pour qu'elle marche avec N'IMPORTE QUELLE
    liste de colonnes numériques :

        resume_kpis(df)                                   → exactement resume_kpis_ancien(df)
        resume_kpis(df, ["num_products", "total_charges"]) → 2 lignes, dans cet ordre

    Règles du refactoring :
    - le résultat doit être IDENTIQUE à l'ancien, au centième près et dans tous les cas, y
      compris les cas bizarres (aucun client parti, valeurs manquantes…). On améliore le code,
      on ne change pas ce qu'il fait ;
    - aucun nom de colonne écrit en dur dans le corps de la fonction (la boucle s'en charge) ;
    - pas d'appel à `resume_kpis_ancien` (ce serait tricher) ;
    - `df` ne doit pas être modifié.

    Ce que vous apprenez : repérer une répétition, la remplacer par une boucle (ou une petite
    fonction), et vérifier qu'on n'a RIEN cassé en comparant l'ancien et le nouveau code sur
    les mêmes données. Les développeurs appellent ça un « test de caractérisation ».

    Indice : regardez précisément ce que l'ancien code arrondit, et à quel moment.
    """
    raise NotImplementedError


# ============================================================================ Ligne de commande


def creer_parser() -> argparse.ArgumentParser:
    """Défi 42 🟡 — Une vraie ligne de commande avec argparse.

    On veut un script `scorer.py` qui note un fichier de clients. Construisez (sans l'appeler)
    l'objet `argparse.ArgumentParser` qui comprend :

        python scorer.py clients.csv
        python scorer.py clients.csv --seuil 0.3 --modele gboost --top 5 -v

    - `fichier`        : argument positionnel, OBLIGATOIRE (le chemin du CSV) ;
    - `--seuil`        : nombre décimal entre 0 et 1, défaut 0.5. Un seuil hors de [0, 1]
                         (ex. `--seuil 1.5`) doit être refusé par argparse ;
    - `--modele`       : "logistic" ou "gboost" uniquement, défaut "logistic" ;
    - `--top`          : nombre entier, défaut 10 ;
    - `-v`/`--verbose` : interrupteur sans valeur (False par défaut, True s'il est présent) ;
    - chaque option a une aide (`help="…"`) : c'est ce qu'affiche `python scorer.py --help`.

    Le correcteur fera par exemple :
        args = creer_parser().parse_args(["clients.csv", "--seuil", "0.3"])
        args.fichier == "clients.csv" and args.seuil == 0.3 and args.top == 10

    Ce que vous apprenez : `add_argument` avec `type=`, `default=`, `choices=`,
    `action="store_true"` et `help=`. Ouvrez `scripts/train.py` : il fait déjà la même chose.

    Indice : `type=` accepte n'importe quelle fonction qui reçoit le texte et renvoie la valeur
    (ou lève une ValueError, qu'argparse transforme en message d'erreur propre). Votre
    `lire_seuil` du défi 40 en est une…
    """
    raise NotImplementedError


# ============================================================================ Trouver quand un bug est apparu


def premier_commit_fautif(commits: list[str], est_bon: Callable[[str], bool]) -> str:
    """Défi 43 🔴 — La bissection, comme `git bisect`.

    Lundi, le ROC-AUC valait 0,83. Aujourd'hui : 0,71. Entre les deux, 64 commits. Lequel a
    cassé le modèle ? Les tester un par un prendrait la journée (chaque test réentraîne le
    modèle). La bissection en teste au plus 6.

    - `commits` : identifiants des versions, du plus ancien au plus récent ;
    - le premier est BON, le dernier est MAUVAIS (on le sait déjà : inutile de les tester) ;
    - quelque part, un commit casse tout : tous ceux d'avant sont bons, lui et tous ceux
      d'après sont mauvais ;
    - `est_bon(commit)` renvoie True si ce commit est bon (c'est l'opération coûteuse).

    Renvoyez le PREMIER commit mauvais, en appelant `est_bon` au plus ⌈log2(n − 1)⌉ fois
    (n = nombre de commits ; 6 appels pour 65 commits). Avec moins de 2 commits, levez une
    ValueError. Ne modifiez pas la liste reçue.

        premier_commit_fautif(["a1", "b2", "c3", "d4", "e5"], lambda c: c in {"a1", "b2"})  → "c3"

    Ce que vous apprenez : la recherche par dichotomie, un « invariant » de boucle
    (bon ≤ … < mauvais) et les erreurs « à un près » (off-by-one), les plus fréquentes de
    toute la programmation. En vrai, git le fait pour vous :
        git bisect start ; git bisect bad HEAD ; git bisect good v1.0
        git bisect run pytest -q tests/test_churn.py

    Indice : gardez deux positions, `bon` (dernier commit connu bon) et `mauvais` (premier
    commit connu mauvais). Tant qu'elles ne sont pas voisines, testez celle du milieu et
    déplacez l'une des deux.
    """
    raise NotImplementedError

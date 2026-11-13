"""Série 14 — Tenir en production : configuration, journaux, reprises, fuseaux horaires, écriture atomique, disjoncteur.

🚦 VOUS VOULEZ DÉVELOPPER ? ÉCRIVEZ DU CODE QUI TIENT LA NUIT, QUAND PERSONNE NE REGARDE.

Sur votre ordinateur, tout se passe bien : le réseau répond, le disque n'est jamais plein, vous
êtes à Paris et vous lisez les messages dans le terminal. En production, le traitement de nuit
tourne seul à 3 h du matin, sur un serveur réglé en UTC, l'opérateur SMS est surchargé une fois
sur cent, et quand quelque chose casse, il ne reste que le journal pour comprendre.

Les six défis suivent le traitement de nuit du projet, du démarrage à l'envoi des SMS de
relance (série 12) :

    86 lire sa configuration    87 journaliser le lot    88 réessayer un appel qui échoue
    89 planifier l'envoi à l'heure de Paris    90 écrire le fichier de scores sans jamais le laisser
    à moitié écrit    91 couper un service en panne avant qu'il n'entraîne tout le reste

Le fil rouge : on ne suppose rien de l'environnement. On le REÇOIT en paramètre (la configuration,
le journal, la fonction d'attente, l'horloge) : le code devient testable, et le correcteur peut
simuler une nuit de pannes en une fraction de seconde.

Prérequis : la série 00 et le défi 40 (série 06 : les exceptions). Le défi 91 demande d'écrire une
classe (défi 48, série 07). Le défi 71 (série 11 : les faux objets) aide à comprendre l'idée.
Rien à installer : `logging`, `zoneinfo`, `tempfile` et `os` font partie de Python.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_14 import *
    >>> lire_config({"CHURN_CLE_API": "cle-equipe-data", "CHURN_MODE_TEST": "oui"})   # défi 86
    >>> scorer_lot(exemple_lot(), score_exemple, journal_console())                    # défi 87
    >>> appeler_avec_reprises(service_capricieux(echecs=2), dormir=print)               # défi 88
    >>> prochain_envoi(datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc))             # défi 89

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py --defi 86` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_14.py`.

Les outils communs (Config, ErreurTemporaire, HorlogeFactice…) sont dans `defis/production.py`,
déjà importés ci-dessous : vous n'avez pas besoin d'ouvrir ce fichier.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import logging  # noqa: F401 — défi 87
import os  # noqa: F401 — défi 90
import tempfile  # noqa: F401 — défi 90
import time
from datetime import datetime, timedelta, timezone  # noqa: F401 — défi 89
from pathlib import Path
from typing import Callable, Iterable, Mapping

from defis.production import (  # noqa: F401 — tout est utile pour essayer vos fonctions
    FAUX,
    PARIS,
    TRANSITOIRES,
    VRAIS,
    CircuitOuvert,
    Config,
    ErreurTemporaire,
    HorlogeFactice,
    exemple_lot,
    journal_console,
    score_exemple,
    service_capricieux,
)

# ============================================================================ Configuration


def lire_config(env: Mapping[str, str]) -> Config:
    """Défi 86 🟢 — Lire sa configuration dans les variables d'environnement, et tout vérifier au démarrage.

    En production, on ne modifie pas le code pour changer un réglage : le serveur fournit des
    VARIABLES D'ENVIRONNEMENT (du texte), et le programme les lit au démarrage. `env` est ce
    dictionnaire de textes (en production, on passe `os.environ`). Renvoyez une `Config` :

    | Variable           | Champ        | Règle                                                    | Défaut      |
    |--------------------|--------------|----------------------------------------------------------|-------------|
    | `CHURN_CLE_API`    | `cle_api`    | texte non vide                                           | obligatoire |
    | `CHURN_SEUIL`      | `seuil`      | nombre (float) entre 0 et 1 inclus, écrit avec un point  | 0.5         |
    | `CHURN_TAILLE_LOT` | `taille_lot` | nombre ENTIER, au moins 1                                | 500         |
    | `CHURN_MODE_TEST`  | `mode_test`  | VRAIS ou FAUX ci-dessous, en majuscules ou minuscules    | False       |

    - VRAIS = ("1", "true", "oui", "yes", "on") ; FAUX = ("0", "false", "non", "no", "off") ;
      tout autre texte (« vrai », « peut-être ») est une erreur ;
    - les espaces autour d'une valeur sont retirés ; une valeur vide (« » ou « ␣␣ ») compte comme
      ABSENTE : défaut, ou erreur pour la clé obligatoire ;
    - S'IL Y A DES ERREURS, levez UNE seule ValueError dont le message cite le nom de CHAQUE
      variable fautive, pas seulement la première : sinon on redéploie quatre fois pour quatre
      fautes de frappe ;
    - le message ne contient jamais la valeur de la clé d'API (série 13, défi 81) ;
    - lisez UNIQUEMENT `env`, jamais `os.environ` directement, et ne modifiez pas `env`.

        >>> lire_config({"CHURN_CLE_API": "cle-equipe-data", "CHURN_SEUIL": "0.35"})
        Config(seuil=0.35, taille_lot=500, mode_test=False)
        >>> lire_config({"CHURN_SEUIL": "0,35", "CHURN_MODE_TEST": "vrai"})
        ValueError: CHURN_CLE_API : obligatoire ; CHURN_SEUIL : … ; CHURN_MODE_TEST : …

    Les pièges : `bool("false")` vaut True (tout texte non vide est « vrai ») ; `int("2.5")` lève
    une ValueError mais `int(float("2.5"))` donne 2 sans prévenir ; et `float("nan")` passe le test
    `seuil < 0 or seuil > 1` (NaN n'est ni plus petit ni plus grand que rien).

    Ce que vous apprenez : « échouer tôt et clairement ». Un programme qui refuse de démarrer avec
    un message précis vaut mieux qu'un programme qui tourne toute la nuit avec un seuil absurde.

    Indice : une liste `erreurs = []`, une petite fonction qui renvoie la valeur nettoyée ou None
    (`env.get(nom)`, `.strip()`), puis un bloc par variable, avec `try: float(texte) except ValueError:`.
    Pour le seuil, `if not 0 <= seuil <= 1` refuse aussi NaN. À la fin :
    `if erreurs: raise ValueError(" ; ".join(erreurs))`.
    """
    raise NotImplementedError


# ============================================================================ Journalisation


def scorer_lot(clients: Iterable[dict], score: Callable[[dict], float], journal: logging.Logger) -> dict[str, float]:
    """Défi 87 🟢 — Journaliser un traitement de nuit plutôt qu'afficher avec print.

    Chaque nuit, le modèle note des milliers de clients. Personne ne regarde l'écran : `print` ne
    sert à rien. On écrit dans un JOURNAL (`logging`), et c'est l'application qui décide où vont
    les messages (fichier, outil de supervision) et à partir de quel niveau.

    Renvoyez le dictionnaire {identifiant: score} des clients notés, en suivant ces règles :

    - facture mensuelle absente, None ou NÉGATIVE → on ne note pas ce client (n'appelez pas
      `score`) et on écrit un AVERTISSEMENT (`journal.warning`) qui contient son identifiant.
      Une facture de 0 € est valide : le client est noté ;
    - `score(client)` plante → le lot CONTINUE avec les clients suivants, et on écrit une ERREUR
      avec la traceback (`journal.exception`, appelé dans le `except`), qui contient l'identifiant ;
    - à la fin, exactement UN message d'INFORMATION (`journal.info`) résume le lot : il contient
      le nombre de clients notés, d'ignorés et en erreur. Rien d'autre au niveau INFO ou au-dessus
      (un message par client noté, c'est `journal.debug`, ou rien) ;
    - jamais de `print`, et seulement dans le journal REÇU (pas `logging.warning(...)`, qui écrit
      dans le journal « racine ») ;
    - ne configurez pas le journal dans la fonction (pas de `logging.basicConfig`, de `addHandler`
      ni de `setLevel`) : c'est le rôle de l'application ;
    - les messages contiennent l'identifiant, jamais la ligne entière : le journal n'a pas à
      contenir l'e-mail des clients (RGPD) ;
    - attrapez `Exception`, pas tout (`except:` tout court) : Ctrl+C doit encore arrêter le lot.

        >>> scorer_lot(exemple_lot(), score_exemple, journal_console())
        10:00:00 WARNING  lot_de_nuit — client C002 ignoré : facture mensuelle invalide (None)
        10:00:00 WARNING  lot_de_nuit — client C004 ignoré : facture mensuelle invalide (-12.5)
        10:00:00 ERROR    lot_de_nuit — client C005 : le modèle a échoué
        Traceback (most recent call last): … ZeroDivisionError: float division by zero
        10:00:00 INFO     lot_de_nuit — lot terminé : 3 clients notés, 2 ignorés, 1 en erreur
        {'C001': 1.0, 'C003': 0.0, 'C006': 0.15}

    Ce que vous apprenez : les niveaux (DEBUG < INFO < WARNING < ERROR), `logger.exception` qui
    garde la traceback, et la règle d'or : le code qui travaille ÉCRIT dans un journal, le
    programme principal le CONFIGURE.

    Indice : `facture = client.get("facture_mensuelle")` puis `if facture is None or facture < 0:`
    (pas `if not facture`, qui écarte 0). Écrivez les messages ainsi :
    `journal.warning("client %s ignoré : facture invalide (%r)", identifiant, facture)` ; logging
    ne fabrique le texte que si le message est vraiment écrit.
    """
    raise NotImplementedError


# ============================================================================ Reprises


def appeler_avec_reprises(appel: Callable[[], object], *, tentatives: int = 4, base: float = 0.5,
                          plafond: float = 8.0, dormir: Callable[[float], None] = time.sleep):
    """Défi 88 🟡 — Réessayer un appel qui échoue, en attendant de plus en plus longtemps.

    L'opérateur SMS répond « surchargé, réessayez » une fois sur cent. Abandonner tout de suite,
    c'est perdre la relance ; réessayer en boucle sans attendre, c'est l'achever. La bonne pratique
    est l'attente EXPONENTIELLE : 0,5 s, puis 1 s, puis 2 s… Appelez `appel()` (sans argument) :

    - succès → renvoyez son résultat tout de suite, quel qu'il soit (même None ou 0) ;
    - erreur PASSAGÈRE (une de TRANSITOIRES : ErreurTemporaire, ConnectionError, TimeoutError) et
      il reste des tentatives → attendez avec `dormir(délai)`, puis réessayez. Avant la k-ième
      reprise (k = 0, 1, 2…), le délai vaut `base * 2**k`, sans jamais dépasser `plafond` ;
    - si l'erreur porte un attribut `attente` qui n'est pas None (le service a dit combien de temps
      attendre : l'en-tête HTTP Retry-After), attendez ce temps-là à la place, sans dépasser
      `plafond`. Une ConnectionError n'a pas d'attribut `attente` : `getattr(erreur, "attente", None)` ;
    - `tentatives` est le nombre TOTAL d'appels (4 = 1 appel + 3 reprises). Après le dernier échec,
      on n'attend plus, et on relance la DERNIÈRE erreur elle-même (`raise`), pas une nouvelle ;
    - toute autre erreur (ValueError : « numéro invalide »…) remonte tout de suite, sans attendre :
      elle ne disparaîtra pas en réessayant ;
    - `tentatives` < 1 → ValueError, sans appeler le service ;
    - n'appelez jamais `time.sleep` directement : utilisez `dormir`. Le correcteur passe une fausse
      fonction qui note les délais au lieu d'attendre.

        >>> appeler_avec_reprises(service_capricieux(echecs=2), dormir=print)
        0.5
        1.0
        'SMS envoyé'

    Ce que vous apprenez : distinguer une panne passagère d'une erreur définitive, et « injecter »
    l'attente pour tester en une milliseconde un code qui, en vrai, attend 15 secondes.

    Indice : `for numero in range(tentatives):`, un `try: return appel()` et un
    `except TRANSITOIRES as erreur:`. Dans le `except`, si c'est la dernière tentative : `raise`.
    Sinon, calculez le délai et appelez `dormir(min(plafond, delai))`.
    """
    raise NotImplementedError


# ============================================================================ Fuseaux horaires


def prochain_envoi(maintenant: datetime) -> datetime:
    """Défi 89 🟡 — Envoyer les SMS à 10 h, heure de Paris… sur un serveur réglé en UTC.

    Les SMS de relance partent à 10 h 00, heure de Paris, du lundi au vendredi. Le serveur, lui,
    vit en UTC (temps universel), comme presque tous les serveurs. Renvoyez le prochain instant
    d'envoi STRICTEMENT après `maintenant`, exprimé en UTC :

    - `maintenant` est une date « consciente » de son fuseau (`tzinfo` renseigné), dans N'IMPORTE
      QUEL fuseau : UTC, Paris, New York… Une date « naïve » (sans fuseau) lève une ValueError :
      on ne peut pas savoir de quelle heure elle parle ;
    - le jour de la semaine se lit à PARIS : vendredi 23 h 30 UTC, c'est déjà samedi à Paris ;
    - s'il est exactement 10 h 00 à Paris, l'envoi du jour est déjà parti : c'est le suivant ;
    - le résultat est en UTC (`tzinfo=timezone.utc`), à 10 h 00 min 00 s pile, heure de Paris.

        >>> prochain_envoi(datetime(2026, 10, 21, 7, 0, tzinfo=timezone.utc))    # mercredi, 9 h à Paris
        datetime(2026, 10, 21, 8, 0, tzinfo=timezone.utc)                       # 10 h à Paris (heure d'été)
        >>> prochain_envoi(datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc))   # vendredi après-midi
        datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc)                       # lundi : 10 h = 9 h UTC !

    Le piège : le changement d'heure. Le dimanche 25 octobre 2026, Paris passe de UTC+2 à UTC+1 :
    10 h à Paris, c'est 8 h UTC le vendredi et 9 h UTC le lundi suivant. Un calcul fait en UTC
    avec le décalage d'aujourd'hui (« +2 h ») envoie les SMS à 9 h toute la semaine suivante.
    Autre piège : `maintenant.replace(tzinfo=PARIS)` ne CONVERTIT pas l'heure, il change seulement
    l'étiquette. Pour convertir : `maintenant.astimezone(PARIS)`.

    Ce que vous apprenez : stocker et comparer en UTC, raisonner en heure locale seulement pour
    les règles humaines (« 10 h », « du lundi au vendredi »), et ne jamais coder un décalage à la main.

    Indice : `local = maintenant.astimezone(PARIS)`, puis un candidat
    `datetime(local.year, local.month, local.day, 10, tzinfo=PARIS)`. Tant qu'il n'est pas après
    `maintenant` ou qu'il tombe un week-end (`candidat.weekday() >= 5`), passez au jour suivant
    (`jour + timedelta(days=1)`, et refaites un candidat à 10 h). Enfin `candidat.astimezone(timezone.utc)`.
    """
    raise NotImplementedError


# ============================================================================ Écriture atomique


def ecrire_atomique(chemin: Path, lignes: Iterable[str]) -> None:
    """Défi 90 🟡 — Écrire le fichier des scores sans jamais le laisser à moitié écrit.

    Le traitement de nuit écrit `scores.csv`, que le tableau de bord lit À TOUT MOMENT. Si le
    traitement plante au milieu (bogue, disque plein, serveur redémarré), ou si le tableau de bord
    lit pendant l'écriture, il trouve un fichier coupé en deux : la moitié des clients a disparu.

    Écrivez `lignes` (des textes, chacun suivi d'un « \\n », en UTF-8) dans `chemin` de façon
    ATOMIQUE : à tout instant, `chemin` contient soit l'ancien fichier complet, soit le nouveau
    fichier complet, jamais un mélange.

    - écrivez d'abord dans un fichier TEMPORAIRE, DANS LE MÊME DOSSIER que `chemin`, puis remplacez
      `chemin` d'un coup avec `os.replace(temporaire, chemin)`. Ce renommage est atomique… à
      condition de rester sur le même disque : un fichier temporaire dans /tmp peut être ailleurs ;
    - fermez le fichier temporaire AVANT de le renommer (sinon une partie peut rester en mémoire) ;
    - le nom du temporaire doit être unique : deux traitements peuvent tourner en même temps, et
      un fichier `scores.csv.tmp` qui existe déjà appartient peut-être à quelqu'un d'autre ;
    - si une erreur survient (`lignes` peut être un générateur qui plante en route), `chemin` reste
      exactement comme avant (ou absent s'il n'existait pas), le temporaire est supprimé, et
      l'erreur remonte (`raise`) : on ne la cache pas ;
    - à aucun moment vous n'ouvrez `chemin` lui-même en écriture.

        >>> dossier = Path(tempfile.mkdtemp())
        >>> ecrire_atomique(dossier / "scores.csv", ["id;score", "C001;0.82", "C003;0.15"])
        >>> (dossier / "scores.csv").read_text(encoding="utf-8")
        'id;score\\nC001;0.82\\nC003;0.15\\n'

    Ce que vous apprenez : penser à ce qui se passe PENDANT l'écriture et en cas de plantage, et
    `try`/`except`/`raise` pour nettoyer derrière soi sans avaler l'erreur. Le même schéma sert
    pour un modèle `.joblib`, une configuration, une base SQLite exportée.

    Indice : `descripteur, nom = tempfile.mkstemp(dir=Path(chemin).parent, suffix=".tmp")` crée un
    fichier au nom unique dans le bon dossier. Puis
    `with os.fdopen(descripteur, "w", encoding="utf-8") as fichier:` pour l'écrire, et APRÈS le
    `with` : `os.replace(nom, chemin)`. Le tout dans un `try:` ; dans
    `except BaseException:`, supprimez le temporaire (`Path(nom).unlink(missing_ok=True)`), puis `raise`.
    """
    raise NotImplementedError


# ============================================================================ Disjoncteur


class Disjoncteur:
    """Défi 91 🔴 — Couper un service en panne, comme un disjoncteur électrique.

    Quand l'opérateur SMS est en panne pour de bon, réessayer chaque relance (défi 88) fait attendre
    tout le lot, et bombarde un service déjà à terre. Le disjoncteur (« circuit breaker ») s'en
    souvient : après plusieurs échecs, il refuse les appels tout de suite pendant un moment, puis
    laisse passer un appel d'essai pour voir si le service est revenu.

        fermé ──(seuil_echecs échecs D'AFFILÉE)──▶ ouvert ──(délai écoulé)──▶ semi-ouvert
          ▲                                          ▲                            │
          └────────────── l'essai réussit ───────────┼────────────────────────────┤
                                                     └────── l'essai échoue ──────┘

    `Disjoncteur(seuil_echecs=3, delai=30.0, horloge=time.monotonic)`. `horloge()` donne l'heure en
    secondes : utilisez-la, elle seule (le correcteur passe une HorlogeFactice et l'avance à la main).

    - `etat` (une propriété, `@property`) vaut « ferme », « ouvert » ou « semi_ouvert » ;
    - `appeler(fonction, *args, **kwargs)` :
      • FERMÉ : appelle `fonction(*args, **kwargs)` et renvoie son résultat. Un échec PASSAGER (une
        erreur de TRANSITOIRES) compte, et l'erreur remonte telle quelle ; un succès remet le
        compteur à zéro (ce sont des échecs D'AFFILÉE). Au `seuil_echecs`-ième échec d'affilée, le
        disjoncteur s'OUVRE (cet échec-là remonte aussi tel quel) ;
      • OUVERT : lève `CircuitOuvert` SANS appeler `fonction` ;
      • au bout de `delai` secondes après l'ouverture (`horloge() - ouverture >= delai`), il est
        SEMI-OUVERT, même si personne ne l'a appelé entre-temps : l'appel suivant est un ESSAI.
        Réussi → fermé, compteur à zéro. Échec passager → de nouveau ouvert, tout de suite (un
        seul échec suffit), et le délai repart de zéro ;
    - les AUTRES erreurs (ValueError…) remontent sans changer l'état : le service a répondu,
      c'est la demande qui était mauvaise ;
    - deux disjoncteurs (SMS, e-mail) sont totalement indépendants.

        >>> horloge = HorlogeFactice()
        >>> sms = Disjoncteur(seuil_echecs=2, delai=30, horloge=horloge)
        >>> en_panne = service_capricieux(echecs=99)
        >>> for _ in range(3):
        ...     try:
        ...         sms.appeler(en_panne)
        ...     except ErreurTemporaire as erreur:
        ...         print(type(erreur).__name__, sms.etat)
        ErreurTemporaire ferme
        ErreurTemporaire ouvert
        CircuitOuvert ouvert
        >>> horloge.avancer(30); sms.etat
        'semi_ouvert'

    Ce que vous apprenez : une machine à états (des états, des transitions, rien d'autre), une
    propriété calculée à la demande, et une idée centrale des systèmes distribués : un service en
    panne ne doit pas faire tomber ceux qui l'appellent.

    Indice : gardez deux attributs dans `__init__` : le nombre d'échecs d'affilée et l'instant
    d'ouverture (`None` quand c'est fermé). L'état se CALCULE à partir d'eux et de `horloge()`.
    Dans `appeler`, regardez l'état AVANT l'appel, puis `try`/`except TRANSITOIRES:` … `raise`.
    """

    def __init__(self, seuil_echecs: int = 3, delai: float = 30.0, horloge: Callable[[], float] = time.monotonic) -> None:
        raise NotImplementedError

    @property
    def etat(self) -> str:
        raise NotImplementedError

    def appeler(self, fonction: Callable, *args, **kwargs):
        raise NotImplementedError

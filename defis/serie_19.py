"""Série 19 — Faire évoluer du code existant : figer, déprécier, extraire, migrer, adapter, comparer.

🏗️ VOUS VOULEZ DÉVELOPPER ? EN ENTREPRISE, ON ÉCRIT RAREMENT SUR UNE PAGE BLANCHE.

Premier jour dans l'équipe rétention. Le code a quatre ans, il n'a presque pas de tests, et il fait
tourner l'entreprise : le score de risque, les remises, le script qui envoie les SMS de relance chaque
nuit. Votre première mission : passer au nouvel opérateur SMS (l'ancien ferme le mois prochain),
nettoyer le calcul des remises, et rendre le script de nuit testable. Sans rien casser.

On ne réécrit pas tout d'un coup. On avance par petits pas, chacun SÛR :

    116 figer le comportement actuel avant d'y toucher     117 renommer un paramètre sans casser 40 appels
    118 extraire la décision du script de nuit             119 lire l'ancien ET le nouveau format de données
    120 brancher le nouvel opérateur derrière l'ancienne interface
    121 faire tourner la nouvelle version dans l'ombre de l'ancienne, et compter les différences

Prérequis : la série 00, le défi 40 (`try` / `except`) et une classe (défi 48). Les défis 117 et 120
prolongent les séries 06 et 14 ; le défi 121 réutilise ce que vous écrivez au défi 116.
Rien à installer : tout fait partie de Python.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- un TEST DE CARACTÉRISATION ne dit pas ce que le code DEVRAIT faire, mais ce qu'il FAIT aujourd'hui, bizarreries
  comprises. On le pose AVANT de toucher au code : tant qu'il passe, on n'a rien changé ;
- une modification de comportement et un nettoyage ne vont JAMAIS dans le même changement. Si l'ancien
  code a un bogue, on le note dans un ticket, on fait le nettoyage à l'identique, puis on corrige le
  bogue dans un changement séparé, qu'on peut relire, tester et annuler seul ;
- une fonction appelée à 40 endroits ne change pas de signature du jour au lendemain : on accepte
  l'ancienne ET la nouvelle façon de l'appeler, l'ancienne prévient (`DeprecationWarning`), et on la
  retire quand plus personne ne l'utilise ;
- un « cœur pur » ne lit pas de fichier, n'affiche rien, n'envoie rien et ne regarde pas l'heure : il
  reçoit des données et RENVOIE une décision. Le reste (lire, envoyer, afficher) est une fine coquille
  autour. Le cœur se teste en une ligne ; la coquille n'a presque plus rien à tester ;
- les données écrites hier ne disparaissent pas quand le format change : le code qui LIT doit accepter
  l'ancien format aussi longtemps qu'il en reste, et refuser franchement ce qu'il ne connaît pas ;
- un ADAPTATEUR est une petite classe qui présente l'ancienne interface et parle la nouvelle : les 40
  appels ne changent pas, une seule classe sait que l'opérateur a changé ;
- avant de remplacer une fonction en production, on peut faire tourner la nouvelle « dans l'ombre » :
  l'utilisateur reçoit toujours l'ancien résultat, et on note chaque cas où la nouvelle aurait répondu
  autre chose. On bascule quand il n'y a plus de différence.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_19 import *
    >>> photo = photographier(ancien_score, [(c,) for c in CLIENTS_EXEMPLE])    # défi 116
    >>> photographier(nouveau_score, [(c,) for c in CLIENTS_EXEMPLE]) == photo  # la réécriture fait-elle pareil ?
    False
    >>> decider_relances(LIGNES_EXEMPLE, JOUR_EXEMPLE)                          # défi 118

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code (pour le défi 120, la classe entière).
2. Lancez `python scripts/defis.py --defi 116` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_19.py`.

Le vieux code (`ancien_score`, `nouveau_score`, `remise_ancienne`, `ancien_traitement_relances`,
`PasserelleAncienne`) et le kit du nouvel opérateur (`ClientMessagerie`, `ErreurMessagerie`) sont dans
`defis/heritage.py`, déjà importés ci-dessous. Lisez-le comme on lit le code d'une équipe qu'on rejoint.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import copy  # noqa: F401 — utile aux défis 116 et 121
import functools  # noqa: F401 — utile au défi 121
import json  # noqa: F401 — utile au défi 119
import operator
import random
import warnings  # noqa: F401 — utile au défi 117
from datetime import date
from typing import Callable, Iterable

from defis.heritage import (  # noqa: F401 — tout est utile pour essayer vos fonctions
    CANAUX,
    CLIENTS_EXEMPLE,
    JOUR_EXEMPLE,
    LIGNES_EXEMPLE,
    PREFERENCES_V1_EXEMPLE,
    PREFERENCES_V2_EXEMPLE,
    ClientMessagerie,
    ErreurMessagerie,
    OperateurFactice,
    PasserelleAncienne,
    Reponse,
    ancien_score,
    ancien_traitement_relances,
    ecrire_lignes,
    nouveau_score,
    remise_ancienne,
)

# ============================================================================ Avant de toucher


def photographier(fonction: Callable, cas: Iterable[tuple]) -> list[tuple[tuple, tuple]]:
    """Défi 116 🟢 — Figer le comportement actuel avant d'y toucher (test de caractérisation).

    On vous demande de nettoyer `ancien_score`. Personne ne sait exactement ce qu'il fait, et il n'a
    aucun test. Avant de changer une ligne, on PHOTOGRAPHIE son comportement sur des cas variés :
    après le nettoyage, la nouvelle photo doit être identique à l'ancienne.

    `cas` est une liste de tuples d'arguments : `(client,)` veut dire « appeler fonction(client) ».
    Appelez la fonction une fois par cas et renvoyez, dans l'ordre, une liste de paires
    `(arguments, resultat)` :

    - `arguments` : les arguments tels qu'ils étaient AVANT l'appel ;
    - `resultat` : `("ok", valeur renvoyée)`, ou `("erreur", "NomDeLException", "message")` si la
      fonction a levé une exception. Une exception fait partie du comportement : on la note.

        >>> photo = photographier(ancien_score, [(c,) for c in CLIENTS_EXEMPLE])
        >>> for arguments, resultat in photo:
        ...     print(arguments[0]["client_id"], arguments[0].get("charges"), resultat)
        C001 89,90 ('ok', 0.95)
        C002 45.0 ('ok', 0.35)
        C003 130 ('ok', 0.85)
        C004 None ('erreur', 'KeyError', "'charges'")
        C005 -10 ('erreur', 'ValueError', 'charges négatives : -10.0')
        >>> CLIENTS_EXEMPLE[0]["charges"]                  # vos cas sont intacts
        '89,90'
        >>> photographier(nouveau_score, [(c,) for c in CLIENTS_EXEMPLE]) == photo
        False

    La réécriture `nouveau_score` ne fait donc PAS pareil (comparez les deux photos ligne à ligne pour
    savoir où). Les règles, que le correcteur essaie toutes :

    - `ancien_score` MODIFIE le client qu'il reçoit (« 89,90 » devient 89.9). Appelez la fonction sur une
      COPIE profonde des arguments (`copy.deepcopy`) : la photo montre les arguments d'avant l'appel, et
      les cas de l'appelant ne sont pas abîmés (sinon, le 2e passage photographie d'autres entrées) ;
    - copiez aussi la valeur renvoyée : certaines vieilles fonctions renvoient une liste qu'elles
      allongent encore aux appels suivants, et la photo changerait après coup ;
    - notez le NOM de l'exception (`type(exc).__name__`) et son message (`str(exc)`), pas l'objet
      exception lui-même : deux exceptions identiques ne sont jamais égales avec `==`, et deux photos du
      même comportement ne seraient pas égales ;
    - `except Exception`, pas `except:` ni `except BaseException` : si l'utilisateur appuie sur Ctrl+C
      (KeyboardInterrupt), le programme doit s'arrêter, pas le noter comme un résultat ;
    - une seule exécution par cas : certaines vieilles fonctions envoient un e-mail à chaque appel ;
    - `cas` peut être un générateur : ne le parcourez qu'une fois.

    Ce que vous apprenez : le test de caractérisation (« golden master »), et pourquoi copier les
    données mutables.

    Indice : une boucle `for arguments in cas`, `copie = copy.deepcopy(arguments)`, puis dans un `try`,
    `valeur = fonction(*copy.deepcopy(arguments))` ; ajoutez `(copie, ("ok", copy.deepcopy(valeur)))`, ou
    dans le `except Exception as exc:`, `(copie, ("erreur", type(exc).__name__, str(exc)))`.
    """
    raise NotImplementedError


def remise(montant: float, taux: float | None = None, *, pourcentage: float | None = None) -> float:
    """Défi 117 🟢 — Renommer un paramètre sans casser les 40 appels existants (`DeprecationWarning`).

    `remise_ancienne(montant, taux)` attend une FRACTION : 0.2 pour 20 %. Chaque mois, quelqu'un écrit
    `remise_ancienne(100, 20)` et le programme plante, ou pire, quelqu'un « corrige » en divisant par 100
    au mauvais endroit. L'équipe décide : désormais, on écrit `remise(montant, pourcentage=20)`.

    Mais `remise(montant, 0.2)` est écrit à 40 endroits, dans trois dépôts. On ne casse rien : on
    accepte l'ancienne façon, on PRÉVIENT ceux qui l'utilisent, et on la retirera dans six mois.

        >>> remise(100, pourcentage=20)
        80.0
        >>> remise(100, 0.2)                     # l'ancien appel marche encore, avec un avertissement
        80.0
        <stdin>:1: DeprecationWarning: remise(montant, taux) est dépréciée : écrivez remise(montant, pourcentage=20)

    Les règles :

    - avec `pourcentage` (entre 0 et 100) : le prix après remise, arrondi au centime, SANS avertissement ;
    - avec `taux` (l'ancienne façon, par position `remise(100, 0.2)` ou par nom `remise(100, taux=0.2)`) :
      exactement le résultat de `remise_ancienne(montant, taux)`, ET un avertissement
      `warnings.warn(message, DeprecationWarning, stacklevel=2)` dont le message nomme `pourcentage`
      (dites aux gens quoi écrire à la place) ;
    - `pourcentage` hors de [0, 100] : ValueError qui nomme `pourcentage` ; `taux` hors de [0, 1] :
      ValueError, comme avant (`remise(100, 20)` doit toujours planter, pas offrir 2 000 % de remise) ;
    - les deux à la fois, ou aucun des deux : TypeError, avec un message qui nomme `taux` et `pourcentage` ;
    - `pourcentage` est un paramètre NOMMÉ seulement (le `*` de la signature, déjà en place : gardez-le) :
      `remise(100, None, 20)` serait illisible.

    Les pièges, que le correcteur essaie tous :

    - 0 est une remise valide (`taux=0`, `pourcentage=0`) : `if taux:` le prend pour « absent ». Testez
      `taux is None` ;
    - `stacklevel=2` fait pointer l'avertissement vers la ligne qui APPELLE `remise`, celle qu'il faut
      corriger. Sans lui, il pointe vers votre fonction, et Python le CACHE : par défaut, un
      DeprecationWarning ne s'affiche que s'il désigne du code du script principal ou du terminal ;
    - un `print` n'est pas un avertissement : on ne peut ni le filtrer, ni en faire une erreur dans les
      tests (`python -W error::DeprecationWarning`), ni le voir dans pytest.

    Ce que vous apprenez : faire évoluer une interface publique sans casser ceux qui l'utilisent.

    Indice : validez d'abord les combinaisons (`if (taux is None) == (pourcentage is None): raise TypeError`),
    puis, si `taux` est donné : `warnings.warn(...)` et `return remise_ancienne(montant, taux)`. Sinon :
    vérifiez `pourcentage` et `return remise_ancienne(montant, pourcentage / 100)`. Une seule formule.
    """
    raise NotImplementedError


# ============================================================================ Rendre testable


def decider_relances(lignes: list[dict[str, str]], aujourd_hui: date, budget: int = 50) -> list[tuple[str, str]]:
    """Défi 118 🟡 — Extraire la décision du script de nuit : un cœur pur, sans fichier ni date du jour.

    Lisez `ancien_traitement_relances` dans `defis/heritage.py`. Il lit un fichier, regarde la date du
    jour, DÉCIDE qui relancer, envoie les SMS et affiche un compte rendu. Pour tester la décision, il
    faudrait un vrai fichier, un faux opérateur, et attendre le bon jour. Et son « budget du jour » est
    un compteur GLOBAL : lancé deux fois dans le même programme, il n'envoie rien la deuxième fois.

    Écrivez le CŒUR de ce script : la décision seule. Vous recevez les lignes du fichier (des
    dictionnaires de TEXTE, tels que `csv.DictReader` les lit) et la date du jour en paramètre ; vous
    renvoyez la liste des relances, `(client_id, offre)`, dans l'ordre où le script les enverrait.

        >>> decider_relances(LIGNES_EXEMPLE, JOUR_EXEMPLE)
        [('C02', 'FIDELITE'), ('C01', 'ENGAGEMENT'), ('C08', 'FIDELITE'), ('C03', 'ENGAGEMENT')]
        >>> decider_relances(LIGNES_EXEMPLE, JOUR_EXEMPLE, budget=2)
        [('C02', 'FIDELITE'), ('C01', 'ENGAGEMENT')]

    La règle d'or : EXACTEMENT les décisions de l'ancien script, sur n'importe quel fichier. Lisez son
    code ligne à ligne : le seuil du score, le consentement, les 30 jours depuis le dernier contact,
    l'ordre, le budget, le choix de l'offre. Le correcteur compare votre fonction à l'ancien script sur
    des centaines de fichiers tirés au hasard, et sur les cas limites (un score pile au seuil, un contact
    il y a pile 30 jours…).

    Et ce qui rend la fonction PURE :

    - aucune lecture de fichier, aucun `print`, aucun envoi : on reçoit des données, on renvoie une
      décision ;
    - jamais `date.today()` : la date arrive en paramètre. Le correcteur vous donne une date de 2025 ;
    - aucun état global : appelée deux fois avec les mêmes lignes, la fonction renvoie deux fois la même
      chose. Le budget s'applique à UN appel ;
    - les lignes reçues ne sont pas modifiées (l'ancien script remplaçait le texte « 0,82 » par 0.82
      dans chaque ligne : l'appelant qui relit ses lignes après coup aurait une surprise).

    Le piège le plus tentant : « améliorer » en passant. L'ancien script refuse un consentement écrit
    « Oui » avec une majuscule (`!= "oui"`). C'est probablement un bogue, mais le corriger ICI enverrait
    des SMS à des clients qui n'en recevaient pas, au milieu d'un changement censé ne rien changer. On le
    note dans un ticket, et on le corrige dans un changement séparé, relu pour lui-même.

    Ce que vous apprenez : séparer la décision (pure, testable) des effets (fichiers, réseau, horloge),
    et refactoriser à comportement identique.

    Indice : construisez une NOUVELLE liste de candidats `(score, client_id, ligne)` avec
    `float(ligne["score"].replace(",", "."))`, triez-la par `(-score, client_id)`, puis gardez les
    lignes qui passent les trois filtres de l'ancien code, dans l'ordre, jusqu'à `budget`.
    """
    raise NotImplementedError


# ============================================================================ Changer de format


def lire_preferences(texte: str | None) -> dict:
    """Défi 119 🟡 — Lire l'ancien ET le nouveau format, refuser ce qu'on ne connaît pas.

    Les préférences de contact des clients sont stockées en JSON dans la base. Jusqu'en 2024,
    l'application écrivait le format 1 ; depuis, elle écrit le format 2. Les 1,2 million de lignes
    écrites avant sont toujours là, et personne ne veut toutes les réécrire un vendredi soir.

        format 1 (sans "version")      {"sms": true, "email": false, "heure": 18}
        format 2                       {"version": 2, "canaux": ["sms", "courrier"], "heure": 9, "langue": "en"}

    Écrivez la fonction qui lit les DEUX et renvoie toujours un dictionnaire au format 2 complet :

        >>> lire_preferences(PREFERENCES_V1_EXEMPLE)
        {'version': 2, 'canaux': ['sms'], 'heure': 18, 'langue': 'fr'}
        >>> lire_preferences(PREFERENCES_V2_EXEMPLE)
        {'version': 2, 'canaux': ['courrier', 'sms'], 'heure': 9, 'langue': 'en'}
        >>> lire_preferences(None)                     # client créé avant les préférences
        {'version': 2, 'canaux': [], 'heure': 10, 'langue': 'fr'}

    Les règles :

    - `None`, une chaîne vide ou faite d'espaces : les préférences par défaut ci-dessus (aucun canal :
      sans consentement, on ne contacte pas) ;
    - format 1 : pas de clé "version" (certaines lignes ont `"version": 1`, c'est le même format).
      `"sms"` et `"email"` sont des booléens, absents = false. Ils deviennent la liste `canaux` ;
    - format 2 : `canaux` est une liste de canaux parmi `CANAUX` ("courrier", "email", "sms"). Un canal
      inconnu : ValueError qui le nomme ;
    - dans les deux formats : `canaux` triée et sans doublon ; `heure` un entier de 0 à 23 (10 si absente),
      sinon ValueError ; `langue` « fr » si absente ;
    - une version que vous ne connaissez pas (3, "2", null…) : ValueError qui la nomme. NE DEVINEZ PAS :
      une ligne écrite par une version plus récente de l'application doit être lue par une version plus
      récente du code, pas à moitié comprise ;
    - une clé inconnue (« fuseau », ajoutée par une autre équipe) est ignorée : elle ne fait pas planter, et
      elle n'apparaît pas dans le résultat ;
    - un texte qui n'est pas du JSON, ou du JSON qui n'est pas un objet (`[1, 2]`) : ValueError.

    Les pièges, que le correcteur essaie tous :

    - le dictionnaire par défaut : si vous renvoyez une constante du module, l'appelant qui fait
      `prefs["canaux"].append("sms")` donne le canal SMS à TOUS les clients suivants. Renvoyez un nouveau
      dictionnaire (et une nouvelle liste) à chaque appel ;
    - `True` est un entier en Python (`isinstance(True, int)` vaut True) : `"heure": true` doit être refusé ;
    - relire ce qu'on vient d'écrire doit redonner la même chose :
      `lire_preferences(json.dumps(lire_preferences(t))) == lire_preferences(t)`.

    Ce que vous apprenez : faire évoluer un format de données sans réécrire le passé (lecteur tolérant,
    versions explicites), et pourquoi refuser vaut mieux que deviner.

    Indice : `donnees = json.loads(texte)` ; vérifiez `isinstance(donnees, dict)`. Puis
    `version = donnees.get("version", 1)` : si c'est 1, `canaux = [c for c in ("email", "sms") if donnees.get(c)]` ;
    si c'est 2, `canaux = donnees.get("canaux", [])` à vérifier ; sinon ValueError. Terminez par
    `{"version": 2, "canaux": sorted(set(canaux)), "heure": heure, "langue": donnees.get("langue", "fr")}`.
    Attention, `True == 1` : `version is True` doit être refusé.
    """
    raise NotImplementedError


# ============================================================================ Changer de fournisseur


class AdaptateurSMS:
    """Défi 120 🟡 — Brancher le nouvel opérateur derrière l'ancienne interface (le patron « adaptateur »).

    L'ancien opérateur ferme. Quarante endroits du code font `passerelle.envoyer_sms(numero, texte)` et
    comptent sur sa réponse True / False. Le nouvel opérateur a un kit tout différent :
    `client.messages.create(destinataire=…, contenu=…, expediteur=…)`, des numéros au format +33…, une
    `Reponse` au lieu d'un booléen, et une `ErreurMessagerie` avec un `code` pour chaque problème.

    Plutôt que de modifier les 40 appels, écrivez UNE classe qui présente l'ancienne interface et parle
    au nouvel opérateur. Le jour de la bascule, on remplace `PasserelleAncienne()` par
    `AdaptateurSMS(ClientMessagerie(cle))` à un seul endroit.

        >>> client = ClientMessagerie("cle_test_123")
        >>> sms = AdaptateurSMS(client)
        >>> sms.envoyer_sms("06 12 34 56 78", "Votre offre vous attend")
        True
        >>> client.envoyes                       # (destinataire, contenu, expéditeur, SMS facturés)
        [('+33612345678', 'Votre offre vous attend', 'RETENTION', 1)]
        >>> sms.envoyer_sms("06 12 34", "Votre offre vous attend")         # numéro mal formé
        False

    `AdaptateurSMS(client, expediteur="RETENTION")` garde le client et l'expéditeur, sans rien envoyer.
    `envoyer_sms(numero, texte)` respecte À LA LETTRE le contrat de `PasserelleAncienne` (lisez sa
    docstring dans `defis/heritage.py`) :

    - les numéros français s'écrivent comme on veut (« 06 12 34 56 78 », « 06.12.34.56.78 »,
      « +33 6 12 34 56 78 », « 0033612345678 »…) : convertissez-les en « +33612345678 ». Un numéro mal
      formé renvoie False ;
    - True si le SMS est parti (statut « accepte »), False s'il n'est pas parti : liste STOP (statut
      « refuse »), numéro invalide (« NUMERO_INVALIDE »), panne passagère (« INDISPONIBLE ») ;
    - un texte vide ou de plus de 160 caractères : ValueError, et RIEN n'est envoyé. Le nouvel opérateur,
      lui, accepte 1 600 caractères et facture un SMS par tranche de 160 : sans cette vérification,
      chaque relance un peu longue serait facturée deux fois, en silence ;
    - clé d'API refusée (« CLE_INVALIDE ») : PermissionError, comme l'ancien. Gardez l'erreur d'origine
      comme cause (`raise PermissionError(...) from exc`) : c'est elle qui dit ce qui s'est passé ;
    - toute AUTRE erreur (« QUOTA_DEPASSE », un code apparu l'an prochain, un bogue) remonte telle quelle.
      Un adaptateur qui transforme tout en False cache les vrais problèmes : 10 000 clients
      « injoignables », et personne ne sait pourquoi ;
    - un seul appel à `create` par SMS : pas de nouvel essai caché ici (les reprises, c'est le défi 88).

    Ce que vous apprenez : le patron adaptateur, traduire des erreurs sans en perdre le sens, et
    respecter un contrat qu'on n'a pas écrit.

    Indice : dans `__init__`, `self.client = client` et `self.expediteur = expediteur`. Dans
    `envoyer_sms` : vérifiez le texte, convertissez le numéro (retirez espaces, points et tirets avec
    `re.sub`, puis trois formes possibles : 0X…, +33X…, 0033X…), appelez `create` dans un `try`, puis
    `except ErreurMessagerie as exc:` et regardez `exc.code` ; tout code non prévu : `raise`.
    """

    def __init__(self, client: ClientMessagerie, expediteur: str = "RETENTION") -> None:
        raise NotImplementedError

    def envoyer_sms(self, numero: str, texte: str) -> bool:
        raise NotImplementedError


# ============================================================================ Basculer sans risque


def en_ombre(ancienne: Callable, nouvelle: Callable, journal: Callable[[dict], object], *,
             proportion: float = 1.0, tirage: Callable[[], float] = random.random,
             egal: Callable[[object, object], bool] = operator.eq) -> Callable:
    """Défi 121 🔴 — Faire tourner la nouvelle version dans l'ombre de l'ancienne, et compter les différences.

    Votre collègue a réécrit le score (`nouveau_score`). Les tests passent, mais les tests ne couvrent
    pas les 400 000 clients réels. Plutôt que de basculer un lundi matin et de croiser les doigts, on fait
    tourner les DEUX en production pendant deux semaines : l'utilisateur reçoit toujours le résultat de
    l'ancienne, la nouvelle tourne « dans l'ombre », et chaque fois qu'elle aurait répondu autre chose,
    on le note. On bascule quand le journal reste vide.

    `en_ombre(ancienne, nouvelle, journal)` renvoie une NOUVELLE fonction qui s'appelle comme `ancienne` :

        >>> differences = []
        >>> score = en_ombre(ancien_score, nouveau_score, differences.append)
        >>> score({"client_id": "C006", "charges": 50})        # l'utilisateur reçoit l'ANCIEN résultat
        0.62
        >>> differences
        [{'arguments': (({'client_id': 'C006', 'charges': 50},), {}), 'ancienne': ('ok', 0.62), 'nouvelle': ('ok', 0.37)}]
        >>> score.stats
        {'appels': 1, 'comparaisons': 1, 'differences': 1}

    À chaque appel `f(*args, **kwargs)` :

    1. on décide si cet appel est comparé : `tirage() < proportion` (avec `proportion=0.1`, un appel sur
       dix : la nouvelle version coûte cher, on ne la lance pas à chaque fois). `proportion` hors de
       [0, 1] : ValueError, dès `en_ombre(...)` ;
    2. on appelle `ancienne(*args, **kwargs)` avec les VRAIS arguments (l'appelant compte peut-être sur ce
       qu'elle leur fait). Son résultat est renvoyé, ou son exception relevée, TELLE QUELLE : pour
       l'utilisateur, rien ne change ;
    3. si l'appel est comparé, on appelle aussi `nouvelle` avec les MÊMES arguments, et on compare les
       deux comportements, notés comme au défi 116 : `("ok", valeur)` ou `("erreur", "NomDuType", "message")`.
       Deux valeurs sont différentes si `egal(a, b)` est faux ; deux erreurs, si leur TYPE diffère ; une
       valeur et une erreur, toujours. En cas de différence, appelez `journal({"arguments": (args, kwargs),
       "ancienne": ..., "nouvelle": ...})` ;
    4. `f.stats` compte les appels, les comparaisons et les différences, pour CETTE fonction-là.

    Les pièges, que le correcteur essaie tous :

    - la nouvelle version ne doit JAMAIS gêner l'utilisateur : si elle plante, on note la différence et on
      renvoie le résultat de l'ancienne. (Mais un Ctrl+C, KeyboardInterrupt, s'arrête : `except Exception`) ;
    - si l'ANCIENNE plante, la comparaison a quand même lieu (c'est peut-être la nouvelle qui a raison),
      puis l'exception de l'ancienne remonte, la même ;
    - `ancien_score` MODIFIE le client qu'il reçoit : la nouvelle version recevrait des arguments déjà
      transformés, et le journal montrerait autre chose que ce qu'a envoyé l'appelant. Faites une copie
      profonde des arguments AVANT d'appeler l'ancienne, et donnez à la nouvelle une copie à elle ;
    - avec `functools.wraps(ancienne)`, la nouvelle fonction garde le nom et la docstring de l'ancienne
      (les journaux et les messages d'erreur restent lisibles) ;
    - deux fonctions `en_ombre` ont chacune leurs `stats` : pas de dictionnaire global ;
    - quand l'appel n'est pas comparé, `nouvelle` n'est pas appelée du tout.

    Ce que vous apprenez : basculer d'une version à l'autre en production sans risque (le « shadow
    testing », celui de la bibliothèque Scientist de GitHub).

    Indice : validez `proportion`, créez `stats = {"appels": 0, "comparaisons": 0, "differences": 0}`, puis
    une fonction intérieure décorée de `@functools.wraps(ancienne)`. Une petite fonction `executer(f, args,
    kwargs)` qui renvoie le comportement noté évite d'écrire deux fois le try / except. Pour relever
    l'exception de l'ancienne, gardez l'objet exception lui-même à côté de son comportement noté.
    N'oubliez pas `fonction.stats = stats` avant de la renvoyer.
    """
    raise NotImplementedError

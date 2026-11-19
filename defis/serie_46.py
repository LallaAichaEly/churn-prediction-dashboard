"""Série 46 — Qui a le droit de voir quoi : permissions, rôles, décisions, droits dormants, cumuls, délégations.

🔑 VOUS VOULEZ DÉVELOPPER ? TOUTE APPLICATION QU'ON OUVRE À PLUS D'UNE PERSONNE DOIT RÉPONDRE, À CHAQUE CLIC : « CETTE
PERSONNE A-T-ELLE LE DROIT DE FAIRE ÇA, SUR CETTE DONNÉE-LÀ ? ». ICI, VOUS ÉCRIVEZ CE QUE FONT LES SYSTÈMES DE
PERMISSIONS (CEUX DE DJANGO, D'APACHE SHIRO, LES « POLICIES » D'AWS IAM, LES GROUPES IMBRIQUÉS D'UN ANNUAIRE) ET LES
REVUES D'ACCÈS QU'EXIGENT LES AUDITEURS : COMPARER, HÉRITER, DÉCIDER, MESURER, SÉPARER, DÉLÉGUER. UNE ERREUR ICI NE FAIT
PAS PLANTER LE PROGRAMME : ELLE OUVRE UNE PORTE, EN SILENCE.

Le tableau de bord du churn s'ouvre aux 300 salariés de l'opérateur. Aujourd'hui, l'accès tient en une ligne
(`regle_actuelle`) : une permission est accordée si elle COMMENCE par l'un des « droits » du compte. Résultat : 36
comptes peuvent exporter le fichier des 1 200 clients, les 300 comptes peuvent lire la fiche des 39 clients qui
travaillent chez l'opérateur, et les deux comptes du prestataire SMS lisent les numéros des 360 clients qui ont refusé
les SMS. Les 7 comptes du support informatique, eux, peuvent TOUT faire : leur colonne « droits » était vide à l'import
de l'annuaire, et `permission.startswith("")` est toujours vrai. Nadia, développeuse de l'équipe du tableau de bord,
remplace cette ligne par un vrai modèle d'accès (des rôles qui héritent les uns des autres, et 14 règles qui autorisent
ou refusent, sous conditions). Elle écrit les six outils de cette série :

    278 permission_couvre    le droit « clients:lire,modifier » couvre-t-il la permission « clients:modifier » ?
    279 roles_effectifs      un manager d'agence EST aussi un conseiller senior, qui EST aussi un conseiller
    280 decider              accordé ou refusé, et par quelle règle : par défaut on refuse, un refus l'emporte
    281 droits_dormants      les droits qu'on a sans s'en être servi depuis 90 jours : ceux qu'on peut retirer
    282 conflits             ceux qui peuvent proposer ET valider un geste commercial (la séparation des tâches)
    283 Delegations          valider les gestes de son manager pendant ses congés, sans son mot de passe

Prérequis : la série 00, les `dict` et les `set`. La série 13 (le code sûr) éclaire les défis 278 et 280, la série
24 (les graphes) le défi 279, la série 14 (les fuseaux horaires) les défis 281 et 283.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- une PERMISSION s'écrit en segments séparés par « : » : « clients:lire:telephone » (lire le numéro de téléphone
  d'un client). Un DROIT est un MOTIF de permissions : « clients:* » (toute action sur les clients), « clients »
  (pareil, et tout ce qui est en dessous), « clients:lire,modifier » (l'une ou l'autre) (défi 278) ;
- un RÔLE regroupe des personnes (« conseiller », « manager_agence ») et peut HÉRITER d'autres rôles : tout ce qui
  vaut pour un conseiller vaut aussi pour un manager. Les rôles « effectifs » d'un compte sont ses rôles et tous ceux
  dont ils héritent (défi 279) ;
- une RÈGLE dit qui (des rôles), quoi (un motif), sous quelles conditions (le client est de la même agence, il est
  salarié…), et si elle AUTORISE ou REFUSE. La décision suit trois principes, ceux d'AWS IAM et de XACML : par défaut
  on refuse ; un refus qui s'applique l'emporte sur toute autorisation ; l'ordre des règles ne change pas la décision
  (défi 280) ;
- le MOINDRE PRIVILÈGE : chacun n'a que les droits dont il se sert. Les droits dormants se retirent (défi 281), et
  certains cumuls sont interdits, même à quelqu'un de confiance (défi 282) ;
- les défis 280 à 283 utilisent les défis précédents : pendant leur vérification, le correcteur prête sa propre
  `permission_couvre` (et ses `roles_effectifs` et `decider`) à votre fichier : chaque défi est noté seul, mais faites
  le 278 d'abord.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_46 import *
    >>> utilisateurs, clients = utilisateurs_exemple(), clients_exemple()
    >>> aujourd_hui = lambda utilisateur, permission, client: regle_actuelle(utilisateur, permission)
    >>> exposition(aujourd_hui, utilisateurs, clients, "clients:exporter")       # le contrôle d'aujourd'hui
    {'comptes': 36, 'couples': 43200}
    >>> salaries = [c for c in clients if c["salarie"]]
    >>> exposition(aujourd_hui, utilisateurs, salaries, "clients:lire")
    {'comptes': 300, 'couples': 11700}

Quand les six défis sont réussis, Nadia peut répondre à l'auditeur :

    >>> heritage, regles = heritage_exemple(), regles_exemple()
    >>> def nouveau(heritage):
    ...     effectifs = {u["identifiant"]: {**u, "roles": roles_effectifs(u["roles"], heritage)} for u in utilisateurs}
    ...     return lambda u, permission, client: decider(effectifs[u["identifiant"]], permission, client, regles).autorise
    >>> exposition(nouveau(heritage), utilisateurs, clients, "clients:exporter")
    {'comptes': 27, 'couples': 14745}
    >>> exposition(nouveau(heritage), utilisateurs, salaries, "clients:lire")
    {'comptes': 11, 'couples': 429}
    >>> dormants = droits_dormants(utilisateurs, heritage, regles, journal_exemple(), REVUE_LE)
    >>> len(dormants), sum(len(ids) for ids in dormants.values()), dormants["u300"]
    (48, 85, ['R11'])
    >>> len(conflits(utilisateurs, heritage, regles, INCOMPATIBLES))
    45
    >>> corrige = {**heritage, "support_informatique": []}     # le support n'hérite plus de l'équipe technique
    >>> exposition(nouveau(corrige), utilisateurs, clients, "clients:exporter")
    {'comptes': 20, 'couples': 6345}
    >>> len(conflits(utilisateurs, corrige, regles, INCOMPATIBLES))
    31

Avec les règles, l'export passe de 36 comptes qui voient tout à 27 comptes, et chaque manager n'exporte plus que les
clients de son agence ; la fiche d'un salarié n'est plus lisible que par 11 comptes au lieu de 300. Mais 7 de ces
comptes sont ceux du support informatique : le modèle a hérité de l'erreur de l'annuaire (« support_informatique »
hérite d'« admin_technique », donc de la règle R11, « * »). La revue des droits dormants le montre : en 90 jours, aucun
des 7 ne s'en est servi. On retire cet héritage : 20 comptes peuvent exporter, et 4 lire la fiche d'un salarié (le DPO
et les 3 administrateurs). Restent 31 cumuls interdits, dont les 12 managers d'agence, qui proposent ET valident les
gestes commerciaux, parce qu'un manager EST un conseiller : l'héritage donne trop, et c'est la revue qui le dit.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites le 278 d'abord (les autres s'en servent), puis
   le 279 ; l'ordre proposé est le plus progressif.
2. Lancez `python scripts/defis.py --defi 278` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_46.py`.

Les outils communs (`utilisateurs_exemple`, `heritage_exemple`, `regles_exemple`, `clients_exemple`, `journal_exemple`,
`regle_actuelle`, `exposition`, `Decision`, `SEPARATEUR`, `JOKER`, `LISTE`, `AUTORISER`, `REFUSER`, `CONDITIONS`,
`INCOMPATIBLES` et `REVUE_LE`) sont dans `defis/acces.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone  # noqa: F401 — datetime(2026, 8, 3, tzinfo=timezone.utc)

from defis.acces import (  # noqa: F401
    AUTORISER,
    CONDITIONS,
    INCOMPATIBLES,
    JOKER,
    LISTE,
    REFUSER,
    REVUE_LE,
    SEPARATEUR,
    Decision,
    clients_exemple,
    exposition,
    heritage_exemple,
    journal_exemple,
    regle_actuelle,
    regles_exemple,
    utilisateurs_exemple,
)

# ============================================================================ Comparer et hériter


def permission_couvre(motif, permission) -> bool:
    """Défi 278 🟢 — Le motif d'un droit couvre-t-il la permission demandée ?

        >>> permission_couvre("clients:lire", "clients:lire")
        True
        >>> permission_couvre("clients", "clients:lire:telephone"), permission_couvre("clients:*", "clients:exporter")
        (True, True)
        >>> permission_couvre("clients:lire,modifier", "clients:modifier"), permission_couvre("*:lire", "rapports:lire")
        (True, True)
        >>> permission_couvre("client", "clients:lire"), permission_couvre("clients:lire:telephone", "clients:lire")
        (False, False)
        >>> permission_couvre("gestes:invalider", "gestes:valider"), permission_couvre("*:exporter", "clients:lire:exporter")
        (False, False)
        >>> permission_couvre("", "clients:lire")
        Traceback (most recent call last):
          ...
        ValueError: le motif '' : segment vide

    Ici : « clients » couvre tout ce qui concerne les clients, jusqu'au numéro de téléphone ; « clients:lire,modifier »
    couvre la lecture et la modification, pas l'export. Mais « client » (un « s » oublié dans un fichier de
    configuration) ne couvre rien : un segment se compare EN ENTIER, ce n'est pas un début de texte. Et le droit de
    lire un seul champ (« clients:lire:telephone ») ne donne pas le droit de lire toute la fiche (« clients:lire ») :
    un motif plus PRÉCIS ne couvre jamais une permission plus LARGE. « gestes:invalider » ne contient pas le droit de
    valider, même si « valider » est écrit dedans. Un motif vide ne couvre rien : il est refusé, franchement.

    Les règles :

    - `motif` et `permission` sont des `str` (TypeError sinon). On les coupe en SEGMENTS à chaque `SEPARATEUR` (« : »)
      ; un segment vide (« », « clients: », « clients::lire ») → ValueError ;
    - la permission demandée est PRÉCISE : aucun de ses segments ne contient `JOKER` (« * ») ni `LISTE` (« , »)
      (ValueError sinon : on demande une action, pas une famille d'actions) ;
    - dans le motif, un segment vaut soit `JOKER` tout seul (n'importe quelle valeur), soit une ou plusieurs valeurs
      séparées par `LISTE` (« lire,modifier » vaut « lire » ou « modifier »). Une valeur vide (« lire, ») ou qui
      contient « * » (« li* », « *,lire ») → ValueError ;
    - on compare position par position : le segment i de la permission doit être l'une des valeurs du segment i du
      motif (ou le motif a « * » à cette position). Comparez des segments ENTIERS (égalité), jamais des débuts de texte
      ni des sous-chaînes ;
    - un motif plus COURT que la permission la couvre si ses segments correspondent : ses segments manquants valent
      « * » (« clients » couvre « clients:lire:telephone ») ;
    - un motif plus LONG que la permission ne la couvre que si tous ses segments en trop valent « * »
      (« clients:lire:* » couvre « clients:lire », mais « clients:lire:telephone » ne la couvre pas) ;
    - rendez True ou False (un vrai `bool`).

    Les pièges : `permission.startswith(motif)` laisse « client » couvrir « clients:lire » (et « » couvrir tout :
    c'est la faille d'aujourd'hui) ; `fnmatch` laisse « * » avaler plusieurs segments (« *:exporter » couvrirait
    « clients:lire:exporter ») ; `zip(motif, permission)` s'arrête au plus court, et laisse un motif plus précis couvrir
    une permission plus large ; `valeur in "lire,modifier"` est un test de SOUS-CHAÎNE (« valider » est dans
    « invalider »).

    Ce que vous apprenez : à comparer des identifiants structurés morceau par morceau (le « WildcardPermission »
    d'Apache Shiro, les scopes OAuth, les chemins d'un routeur web), qu'un test de préfixe sur du texte est presque
    toujours une faille, et à refuser une entrée mal formée plutôt que de deviner.

    Indice : `morceaux = motif.split(SEPARATEUR)` ; une boucle `for i in range(max(len(morceaux), len(demandes)))` ;
    pour le segment i du motif, « * » si `i >= len(morceaux)` ; et `demandes[i] in segment.split(LISTE)` (une LISTE de
    valeurs, pas une sous-chaîne).
    """
    raise NotImplementedError


def roles_effectifs(noms, heritage) -> frozenset:
    """Défi 279 🟢 — Les rôles donnés, et tous ceux dont ils héritent, directement ou non.

        >>> heritage = heritage_exemple()
        >>> sorted(roles_effectifs(["manager_agence"], heritage))
        ['conseiller', 'conseiller_senior', 'manager_agence']
        >>> sorted(roles_effectifs(["direction"], heritage))
        ['analyste', 'conseiller', 'conseiller_senior', 'directeur_regional', 'direction', 'manager_agence']
        >>> roles_effectifs(["conseiller"], heritage)
        frozenset({'conseiller'})
        >>> sorted(roles_effectifs(["a"], {"a": ["b"], "b": ["a"]}))       # deux rôles qui héritent l'un de l'autre
        ['a', 'b']
        >>> roles_effectifs(["manger_agence"], heritage)
        Traceback (most recent call last):
          ...
        ValueError: rôle inconnu : 'manger_agence'

    Ici : `heritage["manager_agence"]` vaut ["conseiller_senior"], et `heritage["conseiller_senior"]` vaut
    ["conseiller"] : un manager d'agence a donc aussi les rôles de conseiller senior et de conseiller, et toutes les
    règles qui visent un conseiller le visent. La direction hérite de deux rôles, et de leurs parents. L'héritage va
    vers les PARENTS seulement : un conseiller n'est pas manager. Deux rôles qui héritent l'un de l'autre (une erreur
    de configuration, mais elle arrive) ont les mêmes rôles effectifs, sans boucle sans fin. Un nom mal tapé est
    refusé : une faute de frappe dans la configuration doit se voir le jour où on la fait.

    Les règles :

    - `noms` est une liste, un tuple ou un ensemble (`set`, `frozenset`) de noms de rôles (des `str`). Un `str` seul
      (« manager_agence » au lieu de ["manager_agence"]), autre chose, ou un nom qui n'est pas un `str` → TypeError ;
    - `heritage` est un `dict` {rôle: liste (ou tuple) des rôles dont il hérite directement} (TypeError sinon) ;
    - rendez un `frozenset` (ou un `set`) : les rôles de `noms`, plus tous ceux dont ils héritent, directement ou par
      l'intermédiaire d'autres rôles, à n'importe quelle profondeur ;
    - un rôle de `noms`, ou un parent rencontré en chemin, qui n'est pas une clé de `heritage` → ValueError (un rôle
      inconnu n'est ni ignoré, ni accepté) ; la liste des parents d'un rôle rencontré qui n'est pas une liste ni un
      tuple → TypeError ;
    - les cycles sont permis et ne bouclent pas ; `heritage` n'est pas modifié ;
    - le correcteur passe un annuaire de 1 801 rôles en « losanges » sur 60 étages (chaque rôle hérite de deux ou trois
      rôles de l'étage suivant, que d'autres rôles partagent) et compte les lignes exécutées : suivre chaque CHEMIN au
      lieu de chaque RÔLE, c'est plus de 2 ** 60 chemins. Gardez l'ensemble des rôles déjà vus, et ne repartez jamais
      d'un rôle vu.

    Les pièges : ne suivre qu'un étage (`heritage[role]` sans recommencer) ; suivre l'héritage dans le mauvais sens
    (les rôles qui héritent de « conseiller » : un conseiller deviendrait manager) ; oublier les rôles de départ ;
    ajouter les parents à la liste de `heritage` elle-même (vous modifiez la configuration) ; une fonction récursive
    sans ensemble des rôles vus (RecursionError sur un cycle, et des heures sur les losanges).

    Ce que vous apprenez : le parcours d'un graphe avec un ensemble des sommets déjà vus (en largeur, en profondeur ou
    avec une pile : le résultat est le même), ce que font les groupes imbriqués d'un annuaire (Active Directory, LDAP)
    et l'héritage des rôles de tous les frameworks web, et à ne pas modifier ce qu'on lit.

    Indice : `vus = set()` ; `a_voir = list(noms)` ; `while a_voir: role = a_voir.pop()` ; si `role in vus`,
    `continue` ; vérifiez `role in heritage` ; `vus.add(role)` ; `a_voir.extend(heritage[role])`.
    """
    raise NotImplementedError


# ============================================================================ Décider


def decider(utilisateur, permission, client, regles) -> Decision:
    """Défi 280 🟡 — Accès accordé ou refusé, et la règle qui en décide.

        >>> regles = regles_exemple()
        >>> manager = {"roles": {"manager_agence", "conseiller_senior", "conseiller"}, "agence": "Lille", "region": "Nord"}
        >>> client = {"agence": "Lille", "region": "Nord", "salarie": False, "consentement_sms": True}
        >>> decider(manager, "clients:exporter", client, regles)
        Decision(autorise=True, regle='R04', delegation=None)
        >>> decider(manager, "clients:exporter", {**client, "agence": "Rouen"}, regles)
        Decision(autorise=False, regle=None, delegation=None)
        >>> decider(manager, "clients:lire", {**client, "salarie": True}, regles)
        Decision(autorise=False, regle='R13', delegation=None)
        >>> sms = {"roles": {"prestataire_sms"}, "agence": "Siège", "region": "Siège"}
        >>> decider(sms, "clients:lire:telephone", client, regles).regle
        'R10'
        >>> decider(sms, "clients:lire:telephone", {**client, "consentement_sms": False}, regles).regle
        'R12'

    Ici : la manager de Lille (ses rôles effectifs, défi 279, sont déjà calculés) exporte les clients de son agence :
    la règle R04 l'autorise, sous la condition « meme_agence ». Pour un client de Rouen, aucune règle ne s'applique :
    c'est un refus PAR DÉFAUT (regle=None). Pour un client salarié de l'opérateur, R01 l'autoriserait à lire la
    fiche, mais R13 la refuse aux conseillers (et un manager EST un conseiller) quand le client est salarié : un REFUS
    l'emporte, même s'il vient après l'autorisation dans la liste. Le prestataire SMS lit les numéros (R10), sauf
    ceux des clients qui ont refusé les SMS (R12).

    Les règles :

    - `utilisateur` et `client` sont des `dict` (TypeError sinon) ; `utilisateur["roles"]` contient ses rôles
      EFFECTIFS (héritage compris, comme les rend `roles_effectifs`) : une liste, un tuple ou un ensemble (un `str`
      → TypeError) ; `permission` est une permission précise (défi 278) ;
    - `regles` est une liste (ou un tuple) de `dict` (TypeError sinon). Chaque règle a :
        • "id" (son nom), "effet" (`AUTORISER` ou `REFUSER`) et "permission" (un motif, défi 278) ;
        • "roles" (facultatif) : la liste des rôles visés. Sans "roles", la règle vise tout le monde ; avec
          "roles": [], elle ne vise PERSONNE (une liste vide n'est pas « pas de liste ») ;
        • "si" (facultatif) : la liste des NOMS de conditions, qui doivent TOUTES être vraies (« et », pas « ou ») ;
          chaque nom est une clé de `CONDITIONS` et `CONDITIONS[nom](utilisateur, client)` rend True ou False ;
    - vérifiez TOUTES les règles AVANT de décider (une faute dans la 14e règle doit se voir même si la 3e suffit) :
      une clé "id", "effet" ou "permission" absente, deux règles avec le même "id", un effet inconnu (« refuse »,
      « Autoriser »), une condition inconnue, un motif mal écrit → ValueError ; "roles" doit être une liste, un tuple
      ou un ensemble, et "si" une liste ou un tuple : un `str` (ou autre chose) à leur place → TypeError ;
    - une règle S'APPLIQUE si son motif couvre la permission (`permission_couvre`, défi 278 : le correcteur vous
      prête la sienne), si elle vise au moins un des rôles de l'utilisateur, et si toutes ses conditions sont vraies.
      Évaluez les conditions en DERNIER (seulement si le motif et les rôles correspondent) ; une condition qui lit un
      attribut absent lève KeyError : laissez-la remonter. Ne la remplacez jamais par « condition fausse » : un refus
      qui ne s'applique plus, c'est une porte ouverte ;
    - la décision :
        • si au moins une règle REFUSER s'applique : `Decision(False, id)`, avec l'id du PREMIER de ces refus dans la
          liste ;
        • sinon, si au moins une règle AUTORISER s'applique : `Decision(True, id)`, avec l'id de la PREMIÈRE de ces
          autorisations ;
        • sinon : `Decision(False, None)` (le refus par défaut) ;
    - rien n'est modifié (ni l'utilisateur, ni le client, ni les règles).

    Les pièges : la première règle qui s'applique décide (comme un pare-feu) : alors l'ordre des règles change la
    décision, et une autorisation placée avant un refus l'annule ; autoriser par défaut ; une condition sur deux qui
    suffit (`any` au lieu de `all`) ; `if not regle.get("roles")` qui traite [] comme « tout le monde » ; comparer le
    motif à la permission avec `==` ; un `try`/`except KeyError` qui transforme une condition impossible à évaluer en
    « fausse » ; ne vérifier que les règles qu'on a lues avant de décider.

    Ce que vous apprenez : l'algorithme de décision des politiques d'accès (« deny overrides » de XACML, l'évaluation
    des policies d'AWS IAM : refus explicite, autorisation explicite, refus implicite), le principe « fail closed »
    (dans le doute, on refuse, ou on lève une erreur), et à valider une configuration entière avant de s'en servir.

    Indice : une première boucle qui vérifie chaque règle et calcule `permission_couvre(regle["permission"],
    permission)` ; une seconde qui garde le premier refus et la première autorisation qui s'appliquent ; `set(roles)
    & set(regle["roles"])` ; `all(CONDITIONS[nom](utilisateur, client) for nom in regle.get("si", []))`.
    """
    raise NotImplementedError


# ============================================================================ Revoir les accès


def droits_dormants(utilisateurs, heritage, regles, journal, maintenant, jours=90) -> dict:
    """Défi 281 🟡 — Pour chaque compte, les autorisations qu'il a sans s'en être servi depuis `jours` jours.

        >>> regles = [{"id": "L", "effet": AUTORISER, "permission": "clients:lire", "roles": ["conseiller"]},
        ...           {"id": "E", "effet": AUTORISER, "permission": "clients:exporter", "roles": ["manager"]},
        ...           {"id": "S", "effet": REFUSER, "permission": "clients", "roles": ["conseiller"], "si": ["client_salarie"]}]
        >>> heritage = {"conseiller": [], "manager": ["conseiller"]}
        >>> utilisateurs = [{"identifiant": "u1", "roles": ["manager"]}, {"identifiant": "u2", "roles": ["conseiller"]},
        ...                 {"identifiant": "u3", "roles": ["conseiller"]}]
        >>> journal = [{"date": REVUE_LE - timedelta(days=3), "utilisateur": "u2", "regle": "L"},
        ...            {"date": REVUE_LE - timedelta(days=200), "utilisateur": "u1", "regle": "E"},
        ...            {"date": REVUE_LE - timedelta(days=1), "utilisateur": "u1", "regle": "L"}]
        >>> droits_dormants(utilisateurs, heritage, regles, journal, REVUE_LE)
        {'u1': ['E'], 'u3': ['L']}
        >>> dormants = droits_dormants(utilisateurs_exemple(), heritage_exemple(), regles_exemple(), journal_exemple(),
        ...                            REVUE_LE)
        >>> len(dormants), dormants["u300"], dormants["u023"], dormants["u065"]
        (48, ['R11'], ['R04'], ['R01', 'R02', 'R03', 'R04'])

    Ici : u1 est manager, donc aussi conseiller (défi 279) : il a les autorisations L et E. Il a lu des fiches hier
    (L), mais son dernier export (E) date de 200 jours : E dort. u2 s'est servi de L il y a 3 jours. u3 n'a jamais rien
    fait : L dort. S est un refus : ce n'est pas un droit, il ne dort jamais. Sur l'annuaire de l'exemple, 48 comptes
    ont au moins un droit dormant : u300 (support informatique) a hérité de R11 (« * ») sans jamais s'en servir, le
    manager d'Amiens (u023) n'a pas exporté depuis l'été, et le directeur régional du Nord (u065) a hérité des droits
    d'un manager d'agence dont il ne se sert plus.

    Les règles :

    - `utilisateurs` : une liste de `dict` {"identifiant", "roles" (les rôles DIRECTS), …} ; `heritage` comme au défi
      279 (utilisez `roles_effectifs` : le correcteur vous prête la sienne) ; `regles` comme au défi 280 ;
    - `journal` : une liste (ou n'importe quel itérable) d'entrées {"date" (datetime avec fuseau), "utilisateur" (un
      identifiant), "regle" (l'id de la règle qui a autorisé l'accès, ou None pour un accès refusé), …} ;
    - `maintenant` : un `datetime` (TypeError sinon) AVEC fuseau horaire (ValueError sinon : on ne compare pas une
      heure sans fuseau aux dates du journal) ; `jours` : un `int` > 0 (pas un bool) : TypeError ou ValueError ;
    - les DROITS d'un compte sont les règles AUTORISER qui le visent par ses rôles EFFECTIFS (sans "roles" : tout le
      monde ; "roles": [] : personne), quelles que soient leurs conditions (le droit existe, même s'il ne vaut que
      pour les clients de son agence). Les règles REFUSER ne sont pas des droits ;
    - un droit est UTILISÉ si le journal contient une entrée de ce compte avec l'id de cette règle, datée de
      `maintenant - timedelta(days=jours)` à `maintenant`, BORNES COMPRISES. Une entrée plus ancienne ne compte pas ;
      une entrée datée APRÈS `maintenant` non plus (on peut faire la revue « au 1er novembre » d'un journal qui
      continue) ;
    - rendez un `dict` {identifiant: liste des ids de ses droits dormants, triés (`sorted`)}, avec SEULEMENT les comptes
      qui en ont au moins un ;
    - rien n'est modifié ;
    - le correcteur passe les 172 000 entrées du journal de l'exemple et les 300 comptes, et compte les lignes
      exécutées : relire le journal pour chaque compte (ou pour chaque droit) est beaucoup trop lent. Lisez le journal
      UNE fois, et rangez dans un `set` les couples (utilisateur, règle) utilisés pendant la période.

    Les pièges : les rôles directs au lieu des rôles effectifs (le directeur régional n'aurait aucun droit de
    manager) ; compter les refus comme des droits ; oublier la période (un export d'il y a 200 jours n'est pas un usage
    récent) ; `debut < date` qui oublie l'entrée tombée pile au début de la période ; compter les entrées postérieures à
    `maintenant` ; juger l'usage par la PERMISSION (« il a lu des fiches ») au lieu de la RÈGLE (« grâce à quelle
    règle ») ; garder les comptes sans droit dormant, avec une liste vide.

    Ce que vous apprenez : la « revue des accès » que demandent les auditeurs (ISO 27001, SOC 2) et le principe du
    moindre privilège, à joindre deux sources (la configuration et le journal) en lisant la plus grosse une seule fois,
    et à borner une période dans le temps sans se tromper d'un jour.

    Indice : `debut = maintenant - timedelta(days=jours)` ; `utilises = {(e["utilisateur"], e["regle"]) for e in
    journal if e["regle"] is not None and debut <= e["date"] <= maintenant}` ; puis, pour chaque compte, `roles =
    roles_effectifs(u["roles"], heritage)` et les AUTORISER qui le visent et dont (identifiant, id) n'est pas utilisé.
    """
    raise NotImplementedError


def conflits(utilisateurs, heritage, regles, incompatibles) -> list[tuple[str, str, str]]:
    """Défi 282 🟡 — Les comptes qui cumulent deux permissions incompatibles (la séparation des tâches).

        >>> regles = [{"id": "P", "effet": AUTORISER, "permission": "gestes:proposer", "roles": ["conseiller"],
        ...            "si": ["meme_agence"]},
        ...           {"id": "V", "effet": AUTORISER, "permission": "gestes:valider", "roles": ["manager"]},
        ...           {"id": "T", "effet": AUTORISER, "permission": "gestes:*", "roles": ["stagiaire"]},
        ...           {"id": "N", "effet": REFUSER, "permission": "gestes:valider", "roles": ["stagiaire"]}]
        >>> heritage = {"conseiller": [], "manager": ["conseiller"], "stagiaire": []}
        >>> utilisateurs = [{"identifiant": "u1", "roles": ["manager"]}, {"identifiant": "u2", "roles": ["conseiller"]},
        ...                 {"identifiant": "u3", "roles": ["stagiaire"]}]
        >>> conflits(utilisateurs, heritage, regles, [("gestes:proposer", "gestes:valider")])
        [('u1', 'gestes:proposer', 'gestes:valider')]
        >>> conflits(utilisateurs_exemple()[:2], heritage_exemple(), regles_exemple(), INCOMPATIBLES)
        [('u001', 'gestes:proposer', 'gestes:valider')]

    Ici : un geste commercial (un avoir, un mois offert) est proposé par un conseiller et validé par un manager. Si la
    même personne peut faire les deux, personne ne relit ses gestes : c'est la SÉPARATION DES TÂCHES (le « principe
    des quatre yeux »). u1 est manager : V lui permet de valider, et comme un manager EST un conseiller (défi 279), P
    lui permet de proposer. Aucune règle ne lui donne les deux : c'est le CUMUL de ses rôles qui est interdit. P ne
    vaut que pour les clients de son agence, mais le droit existe : la condition ne l'efface pas. u3 aurait tout par
    « gestes:* », mais un refus SANS condition (N) lui retire la validation pour tous les clients : pas de conflit.

    Les règles :

    - `utilisateurs`, `heritage` et `regles` comme au défi 281 (utilisez `roles_effectifs` et `permission_couvre` : le
      correcteur vous prête les siennes) ; `incompatibles` : une liste de couples (permission 1, permission 2) de
      permissions précises. Un couple qui est un `str` ou qui n'est pas un tuple ni une liste → TypeError ; un couple
      qui n'a pas exactement deux permissions → ValueError ;
    - un compte A LE DROIT d'une permission (au moins pour un client) si, avec ses rôles EFFECTIFS :
        • au moins une règle AUTORISER le vise (par ses rôles, comme au défi 281) et son motif couvre la permission,
          quelles que soient ses conditions ;
        • et aucune règle REFUSER SANS condition (sans "si", ou avec "si": []) ne le vise avec un motif qui couvre la
          permission. Un refus AVEC conditions ne retire le droit que pour certains clients : le droit existe encore ;
    - un compte est en CONFLIT pour un couple s'il a le droit des DEUX permissions, en comptant TOUS ses rôles
      effectifs ENSEMBLE (pas rôle par rôle) ;
    - rendez la liste des (identifiant, permission 1, permission 2), les comptes dans l'ordre de `utilisateurs`, et pour
      chaque compte les couples dans l'ordre de `incompatibles` ; [] s'il n'y en a aucun. Rien n'est modifié.

    Les pièges : chercher le conflit rôle par rôle (aucun rôle n'est toxique seul : c'est le cumul qui l'est, et il
    n'apparaît qu'avec tous les rôles effectifs ensemble) ; comparer les permissions avec `==` (« gestes:* » donne les
    deux) ; les rôles directs au lieu des rôles effectifs ; oublier qu'un refus sans condition retire le droit ; ou, à
    l'inverse, laisser un refus SOUS condition le retirer pour tous ; ne compter que les autorisations sans condition.

    Ce que vous apprenez : la séparation des tâches (« segregation of duties ») des auditeurs financiers, que les
    risques naissent souvent du CUMUL de droits légitimes, et à raisonner sur ce qu'un compte peut faire « pour au
    moins un client » sans énumérer les clients.

    Indice : une fonction `a_le_droit(roles, permission, regles)` qui parcourt les règles qui visent ces rôles et dont
    le motif couvre la permission : un REFUSER sans "si" → False tout de suite ; un AUTORISER → retenez-le ; puis, pour
    chaque compte, `roles = roles_effectifs(u["roles"], heritage)` une seule fois, et chaque couple.
    """
    raise NotImplementedError


# ============================================================================ Déléguer


class Delegations:
    """Défi 283 🔴 — Des délégations datées et révocables, qui ne donnent jamais plus que ce que le délégant peut.

        >>> annuaire = {u["identifiant"]: u for u in utilisateurs_exemple()}
        >>> delegations = Delegations(annuaire, heritage_exemple(), regles_exemple())
        >>> aout = datetime(2026, 8, 3, tzinfo=timezone.utc)
        >>> delegations.ajouter("u001", "u002", "gestes:valider", aout, aout + timedelta(days=14))
        1
        >>> client = {"agence": "Lille", "region": "Nord", "salarie": False, "consentement_sms": True}
        >>> delegations.decider("u002", "gestes:valider", client, aout + timedelta(days=2))
        Decision(autorise=True, regle='R03', delegation=1)
        >>> delegations.decider("u002", "gestes:valider", client, aout + timedelta(days=14))   # c'est fini
        Decision(autorise=False, regle=None, delegation=None)
        >>> delegations.decider("u002", "gestes:valider", {**client, "agence": "Rouen"}, aout + timedelta(days=2))
        Decision(autorise=False, regle=None, delegation=None)
        >>> delegations.decider("u002", "clients:exporter", client, aout + timedelta(days=2))    # pas délégué
        Decision(autorise=False, regle=None, delegation=None)
        >>> annuaire["u001"]["roles"] = ["conseiller"]          # le 5 août, u001 n'est plus manager
        >>> delegations.decider("u002", "gestes:valider", client, aout + timedelta(days=2))
        Decision(autorise=False, regle=None, delegation=None)

    Ici : en août, la manager de Lille (u001) part deux semaines. Aujourd'hui, elle donnerait son mot de passe à un
    conseiller senior (u002) : il aurait TOUS ses droits, et le journal dirait que c'est elle qui a tout fait. Avec une
    délégation, u002 valide les gestes commerciaux (et seulement eux) des clients de Lille (l'agence de la DÉLÉGANTE,
    puisque c'est son droit qu'il emprunte), du 3 août 0 h au 17 août 0 h (exclu). Le journal dira « R03, délégation
    n°1 ». Et quand u001 cesse d'être manager, la délégation ne donne plus rien : on ne délègue que ce qu'on a ENCORE.

    Ce qu'il faut écrire :

    - `__init__(annuaire, heritage, regles)` : `annuaire` est un `dict` {identifiant: compte}, où chaque compte est un
      `dict` avec "roles" (ses rôles DIRECTS) et ses attributs ("agence", "region"…) ; `heritage` comme au défi 279 ;
      `regles` comme au défi 280. Ne COPIEZ PAS l'annuaire : c'est l'annuaire vivant (un départ, un changement de rôle
      s'y voient aussitôt) ;
    - `ajouter(delegant, delegue, motif, debut, fin)` : le compte `delegant` prête, de `debut` (compris) à `fin`
      (exclue), les permissions couvertes par `motif` au compte `delegue`. Rend le NUMÉRO de la délégation : 1, puis 2,
      3… dans l'ordre des ajouts ;
    - `revoquer(numero)` : la délégation ne vaut plus rien, tout de suite ;
    - `decider(identifiant, permission, client, maintenant)` : la `Decision` (défi 280) pour ce compte, à cet instant,
      délégations comprises.

    Les règles :

    - `ajouter` : deux identifiants (`str`, TypeError sinon) présents dans l'annuaire, et différents (ValueError sinon :
      on ne se délègue rien à soi-même) ; un motif valide (défi 278 : TypeError ou ValueError) ; `debut` et `fin` des
      `datetime` (TypeError) avec fuseau (ValueError), et `debut < fin` (ValueError). On n'y vérifie PAS les droits du
      délégant : ils peuvent changer d'ici là, ils sont vérifiés à chaque `decider` ;
    - `revoquer` : un numéro (`int`, pas un bool : TypeError) d'une délégation qui existe et n'est pas déjà révoquée
      (ValueError sinon) ;
    - `decider` : `identifiant` présent dans l'annuaire (ValueError sinon) ; `maintenant` un `datetime` avec fuseau
      (TypeError, ValueError). Puis :
        1. la décision PROPRE du compte : `decider(compte, permission, client, regles)` (défi 280 : le correcteur vous
           prête le sien, et ses `roles_effectifs` et `permission_couvre`), avec ses rôles EFFECTIFS (le compte de
           l'annuaire a ses rôles directs : passez une COPIE avec "roles" remplacés par `roles_effectifs(...)`, sans
           modifier l'annuaire). Si elle autorise, ou si elle refuse PAR UNE RÈGLE (un refus explicite, `regle` n'est
           pas None), rendez-la telle quelle : une délégation ne lève jamais un refus explicite ;
        2. sinon (refus par défaut), parcourez les délégations dans l'ordre de leurs numéros, et gardez celles qui ne
           sont pas révoquées, dont le délégué est ce compte, qui sont en cours (`debut <= maintenant < fin`), dont le
           motif couvre la permission, et dont le délégant est ENCORE dans l'annuaire (un délégant parti ne délègue
           plus rien : pas de KeyError) ;
        3. pour chacune, la décision du DÉLÉGANT lui-même, MAINTENANT : `decider` avec SES rôles effectifs d'aujourd'hui
           et SES attributs (son agence), pour ce client, sans ses propres délégations reçues (pas de chaîne : on ne
           re-délègue pas ce qu'on a emprunté). La première qui l'autorise donne `Decision(True, sa_regle, numero)` ;
        4. si aucune ne l'autorise : la décision propre (le refus par défaut) ;
    - deux objets `Delegations` ne partagent rien.

    Les pièges : copier les rôles du délégant au moment d'`ajouter` (il est rétrogradé le 5 août, et son adjoint garde
    ses droits) ; accorder le motif délégué sans demander au délégant s'il l'a (on déléguerait « * » sans l'avoir) ;
    juger le délégant avec l'agence du DÉLÉGUÉ ; suivre les délégations du délégant (une chaîne de prêts) ; une `fin`
    comprise, ou un `debut` oublié ; une délégation qui contourne un refus explicite ; une révocation oubliée ; un
    délégant parti de l'annuaire qui fait planter `decider`.

    Ce que vous apprenez : la délégation sans partage de mot de passe (les « assume role » d'AWS, l'emprunt d'identité
    tracé des logiciels de gestion), à ne jamais RECOPIER un droit mais à le REDEMANDER au moment où on s'en sert (la
    révocation devient immédiate), et les intervalles de temps semi-ouverts [début, fin).

    Indice : `self._delegations = {}` (numéro → (delegant, delegue, motif, debut, fin)) et `self._revoquees = set()` ;
    `compte = {**self._annuaire[identifiant], "roles": roles_effectifs(self._annuaire[identifiant]["roles"],
    self._heritage)}` ; `for numero, (delegant, ...) in sorted(self._delegations.items())`.
    """

    def __init__(self, annuaire, heritage, regles) -> None:
        raise NotImplementedError

    def ajouter(self, delegant, delegue, motif, debut, fin) -> int:
        raise NotImplementedError

    def revoquer(self, numero) -> None:
        raise NotImplementedError

    def decider(self, identifiant, permission, client, maintenant) -> Decision:
        raise NotImplementedError

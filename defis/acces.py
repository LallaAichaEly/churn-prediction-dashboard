"""Outils communs de la série 46 : l'annuaire, les rôles, les règles d'accès, les clients et le journal des accès.

Le tableau de bord du churn s'ouvre aux 300 salariés de l'opérateur : des conseillers dans 12 agences, leurs managers,
les directeurs régionaux, les analystes et les data scientists du siège, la direction, le délégué à la protection des
données (DPO), les comptes du prestataire qui envoie les SMS, l'équipe technique. Aujourd'hui, l'accès est contrôlé par
une seule ligne (`regle_actuelle`) : chaque compte a une liste de « droits », et une permission est accordée si elle
COMMENCE par l'un d'eux. Nadia, développeuse de l'équipe du tableau de bord, remplace cette ligne par un vrai modèle
d'accès. Ce module lui fournit :

- `utilisateurs_exemple()` : les 300 comptes, des dict {"identifiant", "roles", "agence", "region", "droits"} ;
  "roles" donne les rôles DIRECTS du compte (l'héritage n'est pas résolu), "droits" les droits d'aujourd'hui ;
- `heritage_exemple()` : le dict {rôle: [rôles dont il hérite]} (un manager d'agence EST aussi un conseiller senior,
  qui EST aussi un conseiller) ;
- `regles_exemple()` : les 14 règles du nouveau modèle, des dict {"id", "effet", "permission", "roles", "si"} ;
- `clients_exemple()` : 1 200 clients, des dict {"identifiant", "agence", "region", "salarie", "consentement_sms"} ;
- `journal_exemple()` : les 120 derniers jours d'accès, des dict {"date", "utilisateur", "permission", "regle",
  "objet"}, jusqu'à `REVUE_LE` ;
- `regle_actuelle(utilisateur, permission)` : le contrôle d'aujourd'hui ;
- `exposition(peut, utilisateurs, clients, permission)` : combien de comptes peuvent `permission` sur au moins un client,
  et combien de couples (compte, client) cela fait ;
- `Decision` : ce que renvoie une décision d'accès (défis 280 et 283) ;
- les constantes `SEPARATEUR`, `JOKER`, `LISTE` (défi 278), `AUTORISER`, `REFUSER`, `CONDITIONS` (défi 280),
  `INCOMPATIBLES` (défi 282) et `REVUE_LE` (défi 281).

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> utilisateurs = utilisateurs_exemple()
    >>> len(utilisateurs), utilisateurs[0]
    (300, {'identifiant': 'u001', 'roles': ['manager_agence'], 'agence': 'Lille', 'region': 'Nord', 'droits': ['clients', 'gestes', 'rapports']})
    >>> heritage_exemple()["manager_agence"], heritage_exemple()["direction"]
    (['conseiller_senior'], ['directeur_regional', 'analyste'])
    >>> regles_exemple()[0]
    {'id': 'R01', 'effet': 'autoriser', 'permission': 'clients:lire,modifier', 'roles': ['conseiller'], 'si': ['meme_agence']}
    >>> clients_exemple()[0]
    {'identifiant': 'C0001', 'agence': 'Lille', 'region': 'Nord', 'salarie': False, 'consentement_sms': False}
    >>> regle_actuelle(utilisateurs[0], "clients:exporter"), regle_actuelle(utilisateurs[0], "modeles:deployer")
    (True, False)
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Défi 278 : une permission s'écrit en SEGMENTS séparés par « : » (« clients:lire:telephone ») ; dans un MOTIF, le
# segment « * » vaut n'importe quel segment, et « lire,modifier » vaut « lire » ou « modifier ».
SEPARATEUR = ":"
JOKER = "*"
LISTE = ","

# Défi 280 : l'effet d'une règle.
AUTORISER, REFUSER = "autoriser", "refuser"

# Défi 280 : les conditions qu'une règle peut poser (dans sa liste "si"), par leur nom. Chacune reçoit l'utilisateur et
# le client, et lit leurs attributs : un attribut absent lève KeyError (on ne décide pas sans savoir).
CONDITIONS = {
    "meme_agence": lambda utilisateur, client: utilisateur["agence"] == client["agence"],
    "autre_agence": lambda utilisateur, client: utilisateur["agence"] != client["agence"],
    "meme_region": lambda utilisateur, client: utilisateur["region"] == client["region"],
    "autre_region": lambda utilisateur, client: utilisateur["region"] != client["region"],
    "client_salarie": lambda utilisateur, client: bool(client["salarie"]),
    "sans_consentement_sms": lambda utilisateur, client: not client["consentement_sms"],
}

# Défi 282 : les couples de permissions qu'une même personne ne doit pas cumuler (la séparation des tâches) : celui
# qui propose un geste commercial ne le valide pas ; celui qui entraîne un modèle ne le met pas en production.
INCOMPATIBLES = [("gestes:proposer", "gestes:valider"), ("modeles:entrainer", "modeles:deployer")]

# Défi 281 : le jour de la revue des accès (le journal s'arrête là).
REVUE_LE = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Decision:
    """Une décision d'accès (défis 280 et 283).

    - `autorise` : True si l'accès est accordé ;
    - `regle` : l'identifiant de la règle qui a décidé ("R04"), ou None quand aucune règle ne s'applique (refus par
      défaut) ;
    - `delegation` : le numéro de la délégation qui a permis l'accès (défi 283), None sinon.
    """

    autorise: bool
    regle: str | None = None
    delegation: int | None = None


# ---------------------------------------------------------------------------- l'organisation

REGIONS = {
    "Nord": ["Lille", "Amiens", "Rouen"],
    "Est": ["Strasbourg", "Metz", "Dijon"],
    "Sud": ["Marseille", "Montpellier", "Toulouse"],
    "Ouest": ["Nantes", "Rennes", "Bordeaux"],
}
AGENCES = [agence for agences in REGIONS.values() for agence in agences]
SIEGE = "Siège"

_HERITAGE = {
    "conseiller": [],
    "stagiaire": ["conseiller"],
    "conseiller_senior": ["conseiller"],
    "manager_agence": ["conseiller_senior"],
    "directeur_regional": ["manager_agence"],
    "direction": ["directeur_regional", "analyste"],
    "analyste": [],
    "data_scientist": ["analyste"],
    "dpo": [],
    "prestataire_sms": [],
    "admin_technique": [],
    "support_informatique": ["admin_technique"],
}

_REGLES = [
    ("R01", AUTORISER, "clients:lire,modifier", ["conseiller"], ["meme_agence"]),
    ("R02", AUTORISER, "gestes:proposer", ["conseiller"], ["meme_agence"]),
    ("R03", AUTORISER, "gestes:valider", ["manager_agence"], ["meme_agence"]),
    ("R04", AUTORISER, "clients:exporter", ["manager_agence"], ["meme_agence"]),
    ("R05", AUTORISER, "clients:lire", ["directeur_regional"], ["meme_region"]),
    ("R06", AUTORISER, "rapports:*", ["manager_agence", "analyste"], []),
    ("R07", AUTORISER, "clients:lire", ["analyste"], []),
    ("R08", AUTORISER, "modeles:*", ["data_scientist"], []),
    ("R09", AUTORISER, "clients:*", ["dpo"], []),
    ("R10", AUTORISER, "clients:lire:telephone", ["prestataire_sms"], []),
    ("R11", AUTORISER, "*", ["admin_technique"], []),
    ("R12", REFUSER, "clients:lire:telephone", ["prestataire_sms"], ["sans_consentement_sms"]),
    ("R13", REFUSER, "clients:*", ["conseiller", "analyste"], ["client_salarie"]),
    ("R14", REFUSER, "clients:exporter,modifier", ["stagiaire"], []),
]

# Les droits d'aujourd'hui (`regle_actuelle`), selon le premier rôle du compte. Le "" des comptes du support vient d'un
# import de l'annuaire où la colonne était vide : `permission.startswith("")` est toujours vrai.
_DROITS = {
    "conseiller": ["clients:lire", "clients:modifier", "gestes:proposer"],
    "stagiaire": ["clients:lire"],
    "conseiller_senior": ["clients:lire", "clients:modifier", "gestes"],
    "manager_agence": ["clients", "gestes", "rapports"],
    "directeur_regional": ["clients", "gestes", "rapports"],
    "direction": ["clients", "gestes", "rapports"],
    "analyste": ["clients:lire", "rapports"],
    "data_scientist": ["clients", "modeles", "rapports"],
    "dpo": ["clients"],
    "prestataire_sms": ["clients:lire"],
    "admin_technique": ["clients", "gestes", "rapports", "modeles"],
    "support_informatique": [""],
}


def heritage_exemple() -> dict[str, list[str]]:
    """{rôle: [rôles dont il hérite directement]} : 12 rôles.

        >>> sorted(heritage_exemple())[:4]
        ['admin_technique', 'analyste', 'conseiller', 'conseiller_senior']
        >>> heritage_exemple()["support_informatique"]
        ['admin_technique']
    """
    return {role: list(parents) for role, parents in _HERITAGE.items()}


def regles_exemple() -> list[dict]:
    """Les 14 règles du nouveau modèle d'accès (11 autorisations, 3 refus), dans l'ordre du fichier de configuration.

        >>> [r["id"] for r in regles_exemple() if r["effet"] == REFUSER]
        ['R12', 'R13', 'R14']
        >>> regles_exemple()[12]
        {'id': 'R13', 'effet': 'refuser', 'permission': 'clients:*', 'roles': ['conseiller', 'analyste'], 'si': ['client_salarie']}
    """
    return [{"id": id_, "effet": effet, "permission": permission, "roles": list(roles), "si": list(si)}
            for id_, effet, permission, roles, si in _REGLES]


@functools.lru_cache(maxsize=1)
def _annuaire() -> tuple:
    alea = random.Random(4601)
    comptes = []

    def ajouter(roles, agence, region):
        comptes.append((tuple(roles), agence, region))

    for region, agences in REGIONS.items():
        for agence in agences:
            ajouter(["manager_agence"], agence, region)
            for _ in range(3):
                ajouter(["conseiller_senior"], agence, region)
            for _ in range(16):
                ajouter(["conseiller"], agence, region)
            for _ in range(2 if agence in ("Lille", "Marseille", "Nantes", "Strasbourg", "Toulouse", "Rennes") else 1):
                ajouter(["stagiaire"], agence, region)
        ajouter(["directeur_regional"], agences[0], region)
    for roles, nombre in ((["direction"], 3), (["analyste"], 16), (["data_scientist"], 6), (["dpo"], 1),
                          (["prestataire_sms"], 2), (["admin_technique"], 3), (["support_informatique"], 7)):
        for _ in range(nombre):
            ajouter(roles, SIEGE, SIEGE)
    # Cinq conseillers travaillent aussi pour l'équipe d'analyse (deux rôles directs).
    conseillers = [i for i, (roles, _, _) in enumerate(comptes) if roles == ("conseiller",)]
    for i in alea.sample(conseillers, 5):
        roles, agence, region = comptes[i]
        comptes[i] = (roles + ("analyste",), agence, region)
    assert len(comptes) == 300
    return tuple((f"u{i:03d}", roles, agence, region) for i, (roles, agence, region) in enumerate(comptes, start=1))


def utilisateurs_exemple() -> list[dict]:
    """Les 300 comptes : {"identifiant", "roles" (directs), "agence", "region", "droits" (ceux d'aujourd'hui)}.

        >>> utilisateurs = utilisateurs_exemple()
        >>> [u["identifiant"] for u in utilisateurs if "analyste" in u["roles"] and "conseiller" in u["roles"]]
        ['u058', 'u062', 'u179', 'u180', 'u236']
        >>> utilisateurs[-1]
        {'identifiant': 'u300', 'roles': ['support_informatique'], 'agence': 'Siège', 'region': 'Siège', 'droits': ['']}
    """
    return [{"identifiant": identifiant, "roles": list(roles), "agence": agence, "region": region,
             "droits": list(_DROITS[roles[0]])}
            for identifiant, roles, agence, region in _annuaire()]


@functools.lru_cache(maxsize=1)
def _clients() -> tuple:
    alea = random.Random(4602)
    clients = []
    for agence in AGENCES:
        region = next(r for r, agences in REGIONS.items() if agence in agences)
        for _ in range(100):
            clients.append((agence, region, alea.random() < 0.03, alea.random() < 0.7))
    return tuple((f"C{i:04d}", *c) for i, c in enumerate(clients, start=1))


def clients_exemple() -> list[dict]:
    """1 200 clients (100 par agence) : {"identifiant", "agence", "region", "salarie", "consentement_sms"}.

    "salarie" : le client travaille aussi chez l'opérateur (sa fiche est protégée) ; "consentement_sms" : il accepte
    de recevoir des SMS commerciaux.

        >>> clients = clients_exemple()
        >>> len(clients), sum(c["salarie"] for c in clients), sum(not c["consentement_sms"] for c in clients)
        (1200, 39, 360)
    """
    return [{"identifiant": identifiant, "agence": agence, "region": region, "salarie": salarie,
             "consentement_sms": consentement}
            for identifiant, agence, region, salarie, consentement in _clients()]


# ---------------------------------------------------------------------------- le journal des accès

# Les habitudes de chaque rôle : (permission, règle, accès par jour ouvré en moyenne). Un rôle hérite des habitudes de
# ses parents seulement quand elles sont listées ici : un directeur régional ne valide plus de gestes commerciaux.
_HABITUDES = {
    "conseiller": [("clients:lire", "R01", 6.0), ("clients:modifier", "R01", 1.5), ("gestes:proposer", "R02", 0.4)],
    "stagiaire": [("clients:lire", "R01", 4.0)],
    "conseiller_senior": [("clients:lire", "R01", 5.0), ("clients:modifier", "R01", 1.5),
                          ("gestes:proposer", "R02", 0.5)],
    "manager_agence": [("clients:lire", "R01", 2.0), ("gestes:proposer", "R02", 0.1), ("gestes:valider", "R03", 1.0),
                       ("rapports:lire", "R06", 1.0), ("clients:exporter", "R04", 0.03)],
    "directeur_regional": [("clients:lire", "R05", 0.5), ("rapports:lire", "R06", 1.0)],
    "direction": [("rapports:lire", "R06", 0.6)],
    "analyste": [("rapports:lire", "R06", 3.0), ("rapports:exporter", "R06", 0.5), ("clients:lire", "R07", 0.05)],
    "data_scientist": [("modeles:entrainer", "R08", 2.0), ("modeles:deployer", "R08", 0.1),
                       ("rapports:lire", "R06", 0.5)],
    "dpo": [("clients:lire", "R09", 0.3), ("clients:exporter", "R09", 0.05)],
    "prestataire_sms": [("clients:lire:telephone", "R10", 40.0)],
    "admin_technique": [("modeles:deployer", "R11", 0.2), ("clients:modifier", "R11", 0.1)],
    "support_informatique": [],
}
_JOURS = 120


@functools.lru_cache(maxsize=1)
def _journal() -> tuple:
    alea = random.Random(4603)
    debut = REVUE_LE - timedelta(days=_JOURS)
    clients_par_agence: dict[str, list[str]] = {}
    for identifiant, agence, *_ in _clients():
        clients_par_agence.setdefault(agence, []).append(identifiant)
    tous = [identifiant for identifiant, *_ in _clients()]
    entrees = []
    gestes = 0
    absents = set(alea.sample(range(300), 9))                  # en congé longue durée : aucun accès
    for i, (identifiant, roles, agence, _) in enumerate(_annuaire()):
        if i in absents:
            continue
        habitudes = [h for role in roles for h in _HABITUDES[role]]
        if identifiant in ("u023", "u088", "u197"):            # trois managers n'ont plus exporté depuis l'été
            habitudes = [h for h in habitudes if h[1] != "R04"]
            entrees.append((debut + timedelta(days=4 + i % 20, hours=9), identifiant, "clients:exporter", "R04",
                            agence))
        for jour in range(_JOURS):
            date = debut + timedelta(days=jour)
            if date.weekday() >= 5:
                continue
            for permission, regle, par_jour in habitudes:
                nombre = int(par_jour) + (alea.random() < par_jour - int(par_jour))
                for _ in range(nombre):
                    instant = date + timedelta(hours=7, seconds=alea.randrange(11 * 3600))
                    if permission.startswith("gestes"):
                        gestes += permission == "gestes:proposer"
                        objet = f"G{alea.randrange(1, 4000):04d}"
                    elif permission.startswith("clients"):
                        pool = clients_par_agence.get(agence, tous)
                        objet = alea.choice(pool) if regle != "R04" else agence
                    else:
                        objet = permission.split(SEPARATEUR)[0]
                    entrees.append((instant, identifiant, permission, regle, objet))
            if alea.random() < 0.02:                              # une tentative refusée (aucune règle)
                entrees.append((date + timedelta(hours=10), identifiant, "clients:exporter", None, agence))
    entrees.sort(key=lambda e: (e[0], e[1]))
    return tuple(entrees)


def journal_exemple() -> list[dict]:
    """Les accès des 120 jours qui précèdent `REVUE_LE`, du plus ancien au plus récent.

    Chaque entrée : {"date" (datetime avec fuseau, UTC), "utilisateur", "permission", "regle" (la règle qui a autorisé,
    None pour un accès refusé), "objet" (le client, le geste, l'agence ou le rapport concerné)}.

        >>> journal = journal_exemple()
        >>> len(journal) > 100_000, journal[0]["date"] >= REVUE_LE - timedelta(days=120), journal[-1]["date"] < REVUE_LE
        (True, True, True)
        >>> sorted(journal[0])
        ['date', 'objet', 'permission', 'regle', 'utilisateur']
    """
    return [{"date": date, "utilisateur": utilisateur, "permission": permission, "regle": regle, "objet": objet}
            for date, utilisateur, permission, regle, objet in _journal()]


# ---------------------------------------------------------------------------- aujourd'hui, et ce qu'on y voit


def regle_actuelle(utilisateur: dict, permission: str) -> bool:
    """Le contrôle d'aujourd'hui : la permission est accordée si elle COMMENCE par l'un des droits du compte.

    Il ne regarde ni le client (son agence, s'il est salarié, s'il accepte les SMS), ni les segments : le droit
    "clients" accorde "clients:exporter" sur les 1 200 clients, et le droit "" (une colonne vide à l'import de
    l'annuaire) accorde TOUT.

        >>> regle_actuelle({"droits": ["clients:lire"]}, "clients:lire:telephone")
        True
        >>> regle_actuelle({"droits": ["gestes"]}, "gestes:valider"), regle_actuelle({"droits": [""]}, "modeles:deployer")
        (True, True)
    """
    return any(permission.startswith(droit) for droit in utilisateur["droits"])


def exposition(peut, utilisateurs, clients, permission: str) -> dict:
    """Combien de comptes peuvent `permission` sur au moins un client, et combien de couples (compte, client) au total.

    `peut(utilisateur, permission, client)` répond True ou False. Une décision ne dépend que des ATTRIBUTS du client
    (son agence, sa région, s'il est salarié, s'il accepte les SMS), pas de son identifiant : `peut` n'est appelée
    qu'une fois par compte et par « profil » de client (48 profils pour les 1 200 clients de l'exemple).

        >>> utilisateurs, clients = utilisateurs_exemple()[:3], clients_exemple()[:10]
        >>> exposition(lambda u, p, c: regle_actuelle(u, p), utilisateurs, clients, "clients:exporter")
        {'comptes': 1, 'couples': 10}
    """
    profils: dict[tuple, list] = {}
    for client in clients:
        cle = tuple(sorted((k, v) for k, v in client.items() if k != "identifiant"))
        profils.setdefault(cle, [client, 0])[1] += 1
    comptes = couples = 0
    for utilisateur in utilisateurs:
        n = sum(nombre for client, nombre in profils.values() if peut(utilisateur, permission, client))
        comptes += n > 0
        couples += n
    return {"comptes": comptes, "couples": couples}

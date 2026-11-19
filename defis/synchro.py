"""Outils communs de la série 47 : deux systèmes qui gardent chacun une copie des fiches clients, et ce qui les relie.

Les conseillers modifient les fiches dans le CRM ; les clients modifient les leurs dans l'appli mobile (leur adresse,
leur e-mail, et surtout leurs consentements : « j'accepte les SMS », « je refuse les e-mails »). Chaque système garde
sa copie, et chaque nuit, à 2 h, un script (`synchro_actuelle`) les remet d'accord. Ce module fournit à Malik,
développeur de l'équipe « données clients », de quoi mesurer ce que ce script perd, et de quoi tester ce qui le
remplace :

- `evenements_exemple()` : 30 jours de modifications (octobre 2026) sur les 1 200 clients, des dict {"t", "replique",
  "client", "action", "champ", "valeur"} rangés par instant réel "t" (en millisecondes depuis 1970) ;
- `heure_physique(replique, t)` : ce qu'affiche l'horloge du serveur de `replique` à l'instant réel `t`. Celle du CRM
  avançait de deux minutes ; le 13 octobre à 10 h 14, le service de mise à l'heure (NTP) l'a corrigée d'un coup :
  elle a RECULÉ de deux minutes ;
- `synchro_actuelle(evenements)` : ce que fait le script d'aujourd'hui, nuit après nuit ; renvoie la « vue » finale du
  CRM, {client: {champ: valeur}} ;
- `rejouer(evenements, Horloge, fusionner)` : la même période avec le nouveau système (défis 288 et 289) ; renvoie
  l'état final du CRM, {client: {"champs": {champ: (valeur, estampille)}, "efface": estampille ou None}} ;
- `vue(etat)` : la vue {client: {champ: valeur}} d'un tel état ; `bilan(vue)` : ce qui est faux dans une vue, comparée
  à ce que les gens ont VRAIMENT fait (le dernier changement de chaque champ, et les effacements demandés) ;
- `versions_exemple()` : les vecteurs de versions des fiches que le script a dû départager, nuit par nuit (défi 284) ;
- `HorlogePhysique` : la première idée, des estampilles prises à l'heure du serveur (défi 288) ;
- `Replique` et `parc_exemple()` : les 48 000 fiches du CRM et de la facturation, et leurs arbres d'empreintes
  (défi 286) ;
- `livraisons_exemple()`, `consentements(messages)` et `ecarts_consentement(etat)` : les messages envoyés par les
  téléphones des clients, tels qu'ils arrivent au serveur de l'appli (défi 287) ;
- les constantes `AVANT`, `APRES`, `EGALES`, `CONCURRENTES` (défi 284), `REPLIQUES`, `CHAMPS`, `PROFONDEUR`,
  `CHIFFRES`, `VIDE` (défi 286) et `DERIVE_MAX_MS` (défi 288).

Chaque fonction renvoie de NOUVEAUX objets à chaque appel : modifiez-les sans crainte.

    >>> evenements = evenements_exemple()
    >>> len(evenements), evenements[0]
    (4751, {'t': 1790838337257, 'replique': 'appli', 'client': 'C0676', 'action': 'modifier', 'champ': 'email', 'valeur': 'c0676.7153@exemple.fr'})
    >>> datetime.fromtimestamp(evenements[0]["t"] / 1000, timezone.utc)
    datetime.datetime(2026, 10, 1, 7, 5, 37, 257000, tzinfo=datetime.timezone.utc)
    >>> bilan(synchro_actuelle(evenements))
    {'champs_faux': 347, 'ressuscites': 41, 'sms_malgre_refus': 186}
"""

from __future__ import annotations

import copy
import functools
import hashlib
import random
from datetime import datetime, timezone

# Défi 284 : ce que `comparer_versions(a, b)` peut répondre.
AVANT, APRES, EGALES, CONCURRENTES = "avant", "apres", "egales", "concurrentes"

# Les deux systèmes qui gardent une copie des fiches, et les champs d'une fiche.
REPLIQUES = ("crm", "appli")
CHAMPS = ("email", "telephone", "adresse", "offre", "consentement_sms", "consentement_email")

# Défi 286 : l'arbre d'empreintes a PROFONDEUR étages sous la racine ; chaque nœud a 16 enfants, un par chiffre
# hexadécimal ; VIDE est l'empreinte d'une feuille (ou d'un sous-arbre) sans aucune fiche.
PROFONDEUR = 3
CHIFFRES = "0123456789abcdef"
VIDE = hashlib.sha256(b"").hexdigest()

# Défi 288 : une estampille reçue qui a plus de DERIVE_MAX_MS millisecondes d'avance sur l'horloge locale est refusée.
DERIVE_MAX_MS = 60_000

JOUR_MS = 86_400_000
HEURE_MS = 3_600_000
DEBUT = datetime(2026, 10, 1, tzinfo=timezone.utc)
DEBUT_MS = int(DEBUT.timestamp() * 1000)
JOURS = 30
NUITS = tuple(DEBUT_MS + n * JOUR_MS + 2 * HEURE_MS for n in range(1, JOURS + 1))   # la synchro de 2 h du matin
RECUL_NTP = DEBUT_MS + 12 * JOUR_MS + 10 * HEURE_MS + 14 * 60_000                   # le 13 octobre, 10 h 14
AVANCE_CRM_MS = 120_000
ESTAMPILLE_IMPORT = (DEBUT_MS - JOUR_MS, 0, "import")       # la fiche de départ, importée la veille dans les deux


def heure_physique(replique: str, t: int) -> int:
    """L'heure (en ms) qu'affiche le serveur de `replique` à l'instant réel `t`.

    Le serveur du CRM avançait de deux minutes ; à `RECUL_NTP`, la mise à l'heure l'a ramené d'un coup à l'heure
    juste : son horloge a reculé. Celle de l'appli est juste.

        >>> heure_physique("crm", RECUL_NTP - 1) - heure_physique("crm", RECUL_NTP)
        119999
        >>> heure_physique("appli", RECUL_NTP)  == RECUL_NTP
        True
    """
    if replique == "crm" and t < RECUL_NTP:
        return t + AVANCE_CRM_MS
    return t


# ---------------------------------------------------------------------------- les fiches et les modifications

_RUES = ["rue des Lilas", "avenue Jean Jaurès", "place du Marché", "rue de la Gare", "boulevard Pasteur",
         "impasse des Tilleuls", "rue Victor Hugo", "chemin des Vignes", "allée des Peupliers", "quai de Loire"]
_VILLES = ["Lille", "Amiens", "Rouen", "Caen", "Nantes", "Rennes", "Tours", "Orléans", "Dijon", "Lyon", "Grenoble",
           "Marseille"]
_OFFRES = ["Essentiel", "Fibre 1G", "Fibre 2G", "Mobile 20 Go", "Mobile 100 Go", "Duo Fibre + Mobile"]
_CHAMPS_CRM = ("email", "telephone", "adresse", "offre")
_CHAMPS_APPLI = ("consentement_sms", "consentement_email", "consentement_sms", "consentement_email", "email", "adresse")
_EFFACES = 41
_INVERSIONS = 7          # des corrections faites de part et d'autre du recul de l'horloge du CRM
CLIENTS = tuple(f"C{n:04d}" for n in range(1, 1201))


def _adresse(alea: random.Random) -> str:
    return f"{alea.randint(1, 120)} {alea.choice(_RUES)}, {alea.choice(_VILLES)}"


def _nouvelle_valeur(alea: random.Random, client: str, champ: str, avant):
    if champ.startswith("consentement"):
        return not avant
    if champ == "email":
        return f"{client.lower()}.{alea.randrange(10_000):04d}@exemple.fr"
    if champ == "telephone":
        return f"06{alea.randrange(10 ** 8):08d}"
    if champ == "adresse":
        return _adresse(alea)
    return alea.choice([o for o in _OFFRES if o != avant])


@functools.lru_cache(maxsize=1)
def _donnees() -> tuple:
    """(fiches de départ, événements, effacés) : tirés une fois, toujours les mêmes."""
    alea = random.Random(4701)
    initiales = {}
    for client in CLIENTS:
        initiales[client] = {"email": f"{client.lower()}@exemple.fr", "telephone": f"06{alea.randrange(10 ** 8):08d}",
                             "adresse": _adresse(alea), "offre": alea.choice(_OFFRES),
                             "consentement_sms": alea.random() < 0.6, "consentement_email": alea.random() < 0.5}

    def instant() -> int:
        jour = alea.randrange(JOURS)
        return DEBUT_MS + jour * JOUR_MS + 7 * HEURE_MS + alea.randrange(15 * HEURE_MS)

    def loin_du_recul(t: int) -> bool:
        return abs(t - RECUL_NTP) > 10 * 60_000

    # 1. Les instants des modifications : deux modifications d'un même champ d'un même client sont séparées d'au moins
    #    une heure (sauf les corrections, juste après, dans le même système).
    prevues: list[tuple[int, str, str, str]] = []           # (t, replique, client, champ)
    par_champ: dict[tuple[str, str], list[int]] = {}
    for client in CLIENTS:
        for replique, champs in (("crm", _CHAMPS_CRM), ("appli", _CHAMPS_APPLI)):
            for _ in range(alea.choice([0, 0, 1, 1, 1, 2, 2, 3, 4])):
                champ = alea.choice(champs)
                for _essai in range(20):
                    t = instant()
                    deja = par_champ.setdefault((client, champ), [])
                    if loin_du_recul(t) and all(abs(t - u) >= HEURE_MS for u in deja):
                        deja.append(t)
                        prevues.append((t, replique, client, champ))
                        if replique == "crm" and alea.random() < 0.1:          # une faute de frappe, corrigée
                            prevues.append((t + alea.randint(10_000, 90_000), replique, client, champ))
                        break

    # 1 bis. La campagne de rétention : le matin, le client reçoit le SMS de la campagne et change son consentement aux
    #    SMS dans l'appli ; quelques heures plus tard, le même jour, un conseiller l'appelle et change son offre dans le
    #    CRM.
    for client in alea.sample(CLIENTS, 420):
        for _essai in range(20):
            t = instant()
            u = t + alea.randint(HEURE_MS, 6 * HEURE_MS)
            if (u - DEBUT_MS) % JOUR_MS >= 22 * HEURE_MS or not (loin_du_recul(t) and loin_du_recul(u)):
                continue
            offre, sms = par_champ.setdefault((client, "offre"), []), par_champ.setdefault((client, "consentement_sms"), [])
            if all(abs(t - v) >= HEURE_MS for v in sms) and all(abs(u - v) >= HEURE_MS for v in offre):
                sms.append(t)
                offre.append(u)
                prevues += [(t, "appli", client, "consentement_sms"), (u, "crm", client, "offre")]
                break

    # 2. Les effacements demandés (RGPD) : plus rien n'arrive à ces clients dans l'heure qui précède, ni après.
    effaces = sorted(alea.sample(CLIENTS, _EFFACES))
    effacements = {}
    for client in effaces:
        t = DEBUT_MS + alea.randint(5, 25) * JOUR_MS + 7 * HEURE_MS + alea.randrange(15 * HEURE_MS)
        effacements[client] = (t, "crm" if alea.random() < 2 / 3 else "appli")
    prevues = [p for p in prevues if p[2] not in effacements or p[0] < effacements[p[2]][0] - HEURE_MS]

    # 3. Les corrections faites juste avant ET juste après le recul de l'horloge du CRM (moins de deux minutes en
    #    tout) : c'est le seul champ modifié de ces clients pendant tout le mois.
    candidats = [c for c in CLIENTS if c not in effacements]
    for client in alea.sample(candidats, _INVERSIONS):
        champ = alea.choice(_CHAMPS_CRM)
        prevues = [p for p in prevues if (p[2], p[3]) != (client, champ)]
        avant = alea.randint(20_000, 90_000)
        apres = alea.randint(5_000, 110_000 - avant)
        prevues += [(RECUL_NTP - avant - alea.randrange(1000), "crm", client, champ),
                    (RECUL_NTP + apres + alea.randrange(1000), "crm", client, champ)]

    # 4. Les valeurs, dans l'ordre du temps (un consentement s'inverse ; une correction change la valeur).
    courantes = copy.deepcopy(initiales)
    evenements = []
    for t, replique, client, champ in sorted(prevues):
        valeur = _nouvelle_valeur(alea, client, champ, courantes[client][champ])
        courantes[client][champ] = valeur
        evenements.append((t, replique, client, "modifier", champ, valeur))
    evenements += [(t, replique, client, "effacer", None, None) for client, (t, replique) in effacements.items()]
    evenements.sort()
    return initiales, tuple(evenements), tuple(effaces)


def fiches_initiales() -> dict[str, dict]:
    """Les 1 200 fiches au 1er octobre, identiques dans les deux systèmes : {client: {champ: valeur}}.

        >>> fiches_initiales()["C0001"]
        {'email': 'c0001@exemple.fr', 'telephone': '0653844756', 'adresse': '38 impasse des Tilleuls, Rennes', 'offre': 'Mobile 20 Go', 'consentement_sms': True, 'consentement_email': False}
    """
    return copy.deepcopy(_donnees()[0])


def evenements_exemple() -> list[dict]:
    """Les modifications d'octobre 2026, rangées par instant réel "t" (ms).

    {"t", "replique" ("crm" ou "appli"), "client", "action" ("modifier" ou "effacer"), "champ", "valeur"} ; pour un
    effacement (le client a demandé qu'on efface ses données), "champ" et "valeur" valent None.

        >>> evenements = evenements_exemple()
        >>> sum(e["action"] == "effacer" for e in evenements), {e["replique"] for e in evenements} == set(REPLIQUES)
        (41, True)
    """
    return [{"t": t, "replique": r, "client": c, "action": a, "champ": ch, "valeur": v}
            for t, r, c, a, ch, v in _donnees()[1]]


def _verite() -> tuple[dict, set]:
    """Ce que les gens ont VRAIMENT fait : la dernière valeur de chaque champ, et les clients effacés."""
    initiales, evenements, effaces = _donnees()
    fiches = copy.deepcopy(initiales)
    for _t, _r, client, action, champ, valeur in evenements:
        if action == "modifier":
            fiches[client][champ] = valeur
    for client in effaces:
        del fiches[client]
    return fiches, set(effaces)


# ---------------------------------------------------------------------------- le script d'aujourd'hui


def synchro_actuelle(evenements) -> dict[str, dict]:
    """Le script d'aujourd'hui, rejoué nuit après nuit. Renvoie la vue finale du CRM, {client: {champ: valeur}}.

    Chaque système applique ses modifications à sa copie et note, pour chaque fiche, `modifie_le` : l'heure de SON
    serveur à la dernière modification. Un effacement retire la fiche de la copie. Chaque nuit, à 2 h, pour chaque
    client :
    - une fiche présente d'un seul côté est recopiée de l'autre (« elle manque, je la remets ») ;
    - deux fiches différentes : la plus récente selon `modifie_le` est recopiée EN ENTIER sur l'autre.

    Trois pertes en découlent : un client qui a changé son consentement dans l'appli le matin perd ce changement si un
    conseiller modifie son adresse dans le CRM l'après-midi (la fiche du CRM est plus récente, elle gagne en entier) ;
    un client effacé d'un côté revient de l'autre la nuit suivante ; et l'heure du serveur n'est pas l'heure.
    """
    initiales, _evenements, _effaces = _donnees()
    copies = {r: {c: {"fiche": dict(f), "modifie_le": ESTAMPILLE_IMPORT[0]} for c, f in initiales.items()}
              for r in REPLIQUES}

    def nuit() -> None:
        crm, appli = copies["crm"], copies["appli"]
        for client in sorted(crm.keys() | appli.keys()):
            a, b = crm.get(client), appli.get(client)
            if a is None:
                crm[client] = copy.deepcopy(b)
            elif b is None:
                appli[client] = copy.deepcopy(a)
            elif a["fiche"] != b["fiche"]:
                gagnante = a if a["modifie_le"] >= b["modifie_le"] else b
                crm[client], appli[client] = copy.deepcopy(gagnante), copy.deepcopy(gagnante)

    nuits = list(NUITS)
    for e in sorted(evenements, key=lambda e: e["t"]):
        while nuits and nuits[0] <= e["t"]:
            nuit()
            nuits.pop(0)
        copie = copies[e["replique"]]
        if e["action"] == "effacer":
            copie.pop(e["client"], None)
        else:
            fiche = copie.setdefault(e["client"], {"fiche": {}, "modifie_le": 0})
            fiche["fiche"][e["champ"]] = e["valeur"]
            fiche["modifie_le"] = heure_physique(e["replique"], e["t"])
    for _ in nuits:
        nuit()
    return {client: dict(fiche["fiche"]) for client, fiche in copies["crm"].items()}


def bilan(vue) -> dict:
    """Ce qui est faux dans une vue {client: {champ: valeur}} (celle du CRM à la fin du mois).

    - "champs_faux" : les champs des clients NON effacés dont la valeur n'est pas la dernière que quelqu'un a saisie
      (une valeur absente compte comme fausse) ;
    - "ressuscites" : les clients effacés à leur demande qui ont encore une fiche (même partielle) ;
    - "sms_malgre_refus" : les clients qui ont REFUSÉ les SMS en dernier, mais que la vue donne d'accord (ils
      recevront les SMS de la prochaine campagne).

        >>> bilan(vue({}))["champs_faux"]                      # une vue vide : tous les champs sont faux
        6954
    """
    verite, effaces = _verite()
    faux = sum(vue.get(client, {}).get(champ, _ABSENT) != valeur
               for client, fiche in verite.items() for champ, valeur in fiche.items())
    ressuscites = sum(1 for client in effaces if vue.get(client))
    sms = sum(1 for client, fiche in verite.items()
              if fiche["consentement_sms"] is False and vue.get(client, {}).get("consentement_sms") is True)
    return {"champs_faux": faux, "ressuscites": ressuscites, "sms_malgre_refus": sms}


_ABSENT = object()


# ---------------------------------------------------------------------------- le nouveau système


def _etat_initial() -> dict:
    return {client: {"champs": {champ: (valeur, ESTAMPILLE_IMPORT) for champ, valeur in fiche.items()}, "efface": None}
            for client, fiche in _donnees()[0].items()}


def _plus_grande(etat: dict):
    estampilles = [est for fiche in etat.values() for _v, est in fiche["champs"].values()]
    estampilles += [fiche["efface"] for fiche in etat.values() if fiche["efface"] is not None]
    return max(estampilles)


def rejouer(evenements, Horloge, fusionner) -> dict:
    """Le mois d'octobre avec le nouveau système. Renvoie l'état final du CRM.

    `Horloge(replique, physique)` est une classe d'horloge (`HorlogeHybride`, défi 288, ou `HorlogePhysique`) ;
    `fusionner(a, b)` est la fusion de deux états (défi 289). Chaque système a son horloge, dont l'heure physique est
    `heure_physique(replique, t)`. Comme dans Cassandra ou Riak, TOUTE écriture passe par la fusion :
    - une modification est estampillée (`horloge.estampiller()`), puis fusionnée dans la copie du système :
      fusionner({client: fiche}, {client: {"champs": {champ: (valeur, estampille)}, "efface": None}}) ;
    - un effacement aussi, avec {"champs": {}, "efface": estampille} ;
    - chaque nuit, à 2 h, les deux copies sont fusionnées, et chaque horloge reçoit (`horloge.recevoir(...)`) la plus
      grande estampille de l'autre copie.
    """
    copies = {r: _etat_initial() for r in REPLIQUES}
    maintenant = [DEBUT_MS]
    horloges = {r: Horloge(r, functools.partial(lambda r: heure_physique(r, maintenant[0]), r)) for r in REPLIQUES}

    def nuit(t: int) -> None:
        maintenant[0] = t
        crm, appli = copies["crm"], copies["appli"]
        plus_grandes = {"crm": _plus_grande(appli), "appli": _plus_grande(crm)}
        fusion = fusionner(crm, appli)
        copies["crm"], copies["appli"] = fusion, copy.deepcopy(fusion)
        for r in REPLIQUES:
            horloges[r].recevoir(plus_grandes[r])

    nuits = list(NUITS)
    for e in sorted(evenements, key=lambda e: e["t"]):
        while nuits and nuits[0] <= e["t"]:
            nuit(nuits.pop(0))
        maintenant[0] = e["t"]
        replique, client = e["replique"], e["client"]
        estampille = horloges[replique].estampiller()
        if e["action"] == "effacer":
            delta = {"champs": {}, "efface": estampille}
        else:
            delta = {"champs": {e["champ"]: (e["valeur"], estampille)}, "efface": None}
        copie = copies[replique]
        locale = {client: copie[client]} if client in copie else {}
        copie[client] = fusionner(locale, {client: delta})[client]
    for t in nuits:
        nuit(t)
    return copies["crm"]


def vue(etat) -> dict[str, dict]:
    """La vue {client: {champ: valeur}} d'un état {client: {"champs": {champ: (valeur, estampille)}, "efface": …}}.

    Un client sans aucun champ (effacé) n'apparaît pas.

        >>> vue({"C1": {"champs": {"offre": ("Essentiel", (5, 0, "crm"))}, "efface": None},
        ...      "C2": {"champs": {}, "efface": (9, 0, "crm")}})
        {'C1': {'offre': 'Essentiel'}}
    """
    return {client: {champ: valeur for champ, (valeur, _est) in fiche["champs"].items()}
            for client, fiche in etat.items() if fiche["champs"]}


class HorlogePhysique:
    """La première idée : estampiller avec l'heure du serveur, (heure, 0, réplique). Même interface que le défi 288.

    Elle recule quand le serveur recule : une correction faite juste après le recul reçoit une estampille PLUS PETITE
    que la faute qu'elle corrige, et la fusion garde la faute.

        >>> horloge = HorlogePhysique("crm", lambda: 1_000)
        >>> horloge.estampiller(), horloge.recevoir((5_000, 3, "appli"))
        ((1000, 0, 'crm'), (1000, 0, 'crm'))
    """

    def __init__(self, replique: str, physique, derive_max: int = DERIVE_MAX_MS) -> None:
        self.replique, self._physique = replique, physique

    def estampiller(self) -> tuple[int, int, str]:
        return (self._physique(), 0, self.replique)

    def recevoir(self, estampille) -> tuple[int, int, str]:
        return self.estampiller()


@functools.lru_cache(maxsize=1)
def _versions() -> tuple:
    """Les vecteurs de versions des deux copies, la nuit, pour chaque fiche modifiée depuis la nuit précédente."""
    _initiales, evenements, _effaces = _donnees()
    vecteurs = {r: {c: {} for c in CLIENTS} for r in REPLIQUES}
    touches: set[str] = set()
    resultat = []
    nuits = list(NUITS)

    def nuit(numero: int) -> None:
        for client in sorted(touches):
            a, b = vecteurs["crm"][client], vecteurs["appli"][client]
            resultat.append((numero, client, dict(a), dict(b)))
            fusion = {r: max(a.get(r, 0), b.get(r, 0)) for r in a.keys() | b.keys()}
            vecteurs["crm"][client], vecteurs["appli"][client] = dict(fusion), dict(fusion)
        touches.clear()

    numero = 0
    for t, replique, client, _action, _champ, _valeur in evenements:
        while nuits and nuits[0] <= t:
            nuits.pop(0)
            numero += 1
            nuit(numero)
        vecteur = vecteurs[replique][client]
        vecteur[replique] = vecteur.get(replique, 0) + 1
        touches.add(client)
    while nuits:
        nuits.pop(0)
        numero += 1
        nuit(numero)
    return tuple(resultat)


def versions_exemple() -> list[tuple[int, str, dict, dict]]:
    """(nuit, client, vecteur du CRM, vecteur de l'appli) : chaque fiche modifiée depuis la nuit précédente.

    Un vecteur de versions compte, pour chaque système, combien de modifications faites DANS ce système la copie a
    déjà vues : {"crm": 3, "appli": 1}. Une modification dans le CRM ajoute 1 à "crm" dans la copie du CRM ; la nuit,
    les deux copies prennent, système par système, le plus grand des deux compteurs.

        >>> versions = versions_exemple()
        >>> len(versions), versions[0]
        (3948, (1, 'C0007', {'crm': 1}, {}))
    """
    return [(n, c, dict(a), dict(b)) for n, c, a, b in _versions()]


# ---------------------------------------------------------------------------- défi 286 : l'arbre d'empreintes


def seau(identifiant: str, profondeur: int = PROFONDEUR) -> str:
    """La feuille d'un client : les `profondeur` premiers chiffres hexadécimaux du SHA-256 de son identifiant.

        >>> seau("C00001")
        'd7c'
    """
    return hashlib.sha256(identifiant.encode("utf-8")).hexdigest()[:profondeur]


class Replique:
    """Une copie des fiches, vue À TRAVERS LE RÉSEAU : chaque question coûte un aller-retour.

    `Replique(empreintes)` reçoit {identifiant: empreinte de la fiche} et range les fiches dans un arbre : les feuilles
    sont les préfixes de `profondeur` chiffres hexadécimaux (« 9c6 ») et chaque client est dans la feuille `seau(id)`.
    L'empreinte d'une feuille résume ses fiches ; celle d'un nœud résume celles de ses 16 enfants ; un sous-arbre sans
    fiche a l'empreinte `VIDE`. Deux copies qui ont les mêmes fiches ont donc la même empreinte racine, et deux
    sous-arbres d'empreintes égales ont (sauf collision de SHA-256) les mêmes fiches.

    - `empreinte(prefixe)` : l'empreinte du nœud `prefixe` ("" est la racine) ;
    - `enfants(prefixe)` : {prefixe + chiffre: empreinte} pour les 16 enfants d'un nœud (pas d'une feuille) ;
    - `feuille(prefixe)` : {identifiant: empreinte} des fiches d'une feuille (len(prefixe) == profondeur) ;
    - `appels` compte les questions posées, `fiches_transferees` les fiches reçues par `feuille`.

        >>> copie = Replique({"C00001": "a" * 64, "C00002": "b" * 64}, profondeur=1)
        >>> seau("C00001", 1), seau("C00002", 1)
        ('d', '9')
        >>> copie.empreinte("") != VIDE, copie.enfants("")["0"] == VIDE, copie.feuille("d")
        (True, True, {'C00001': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'})
        >>> copie.appels, copie.fiches_transferees
        (3, 1)
    """

    def __init__(self, empreintes: dict, profondeur: int = PROFONDEUR, nom: str = "") -> None:
        if not isinstance(empreintes, dict):
            raise TypeError("une Replique se construit à partir d'un dict {identifiant: empreinte}")
        if type(profondeur) is not int or profondeur < 0:
            raise ValueError("la profondeur est un entier positif ou nul")
        self.profondeur, self.nom = profondeur, nom
        self.appels = self.fiches_transferees = 0
        self._feuilles: dict[str, dict] = {}
        for identifiant, empreinte in empreintes.items():
            self._feuilles.setdefault(seau(identifiant, profondeur), {})[identifiant] = empreinte
        self._occupes = {f[:k] for f in self._feuilles for k in range(profondeur + 1)}
        self._noeuds: dict[str, str] = {}
        self._calculer("")

    def _calculer(self, prefixe: str) -> str:
        if prefixe not in self._occupes:
            return VIDE
        if len(prefixe) == self.profondeur:
            lignes = "\n".join(f"{i}\t{e}" for i, e in sorted(self._feuilles[prefixe].items()))
            empreinte = hashlib.sha256(lignes.encode("utf-8")).hexdigest()
        else:      # 16 empreintes de 64 caractères, toujours dans le même ordre : les mettre bout à bout est sans ambiguïté
            empreinte = hashlib.sha256("".join(self._calculer(prefixe + c) for c in CHIFFRES).encode()).hexdigest()
        self._noeuds[prefixe] = empreinte
        return empreinte

    def _prefixe(self, prefixe, longueurs: str) -> str:
        if not isinstance(prefixe, str):
            raise TypeError(f"un préfixe est un texte, pas {type(prefixe).__name__}")
        if any(c not in CHIFFRES for c in prefixe) or len(prefixe) > self.profondeur:
            raise ValueError(f"{prefixe!r} n'est pas un nœud de l'arbre (profondeur {self.profondeur})")
        if longueurs == "noeud" and len(prefixe) == self.profondeur:
            raise ValueError(f"{prefixe!r} est une FEUILLE : elle n'a pas d'enfants, demandez `feuille({prefixe!r})`")
        if longueurs == "feuille" and len(prefixe) != self.profondeur:
            raise ValueError(f"{prefixe!r} n'est pas une feuille (une feuille a {self.profondeur} chiffres) : "
                             f"demandez `enfants({prefixe!r})`")
        return prefixe

    def empreinte(self, prefixe: str = "") -> str:
        self.appels += 1
        return self._noeuds.get(self._prefixe(prefixe, "tout"), VIDE)

    def enfants(self, prefixe: str = "") -> dict[str, str]:
        self.appels += 1
        prefixe = self._prefixe(prefixe, "noeud")
        return {prefixe + c: self._noeuds.get(prefixe + c, VIDE) for c in CHIFFRES}

    def feuille(self, prefixe: str) -> dict[str, str]:
        self.appels += 1
        fiches = dict(self._feuilles.get(self._prefixe(prefixe, "feuille"), {}))
        self.fiches_transferees += len(fiches)
        return fiches

    def __repr__(self) -> str:
        return f"Replique({self.nom or 'sans nom'}, profondeur={self.profondeur}, appels={self.appels})"


@functools.lru_cache(maxsize=1)
def _parc() -> tuple:
    alea = random.Random(4786)
    crm = {f"C{n:05d}": format(alea.getrandbits(256), "064x") for n in range(1, 48_001)}
    facturation = dict(crm)
    ids = sorted(crm)
    tires = alea.sample(ids, 37)
    for identifiant in tires[:14]:                       # modifiées d'un côté, pas encore de l'autre
        facturation[identifiant] = format(alea.getrandbits(256), "064x")
    for identifiant in tires[14:27]:                     # des fiches que la facturation n'a pas encore
        del facturation[identifiant]
    for identifiant in tires[27:]:                       # des comptes clos dans le CRM, encore facturés
        del crm[identifiant]
    return tuple(crm.items()), tuple(facturation.items())


def parc_exemple() -> tuple[dict, dict]:
    """Les empreintes des 48 000 fiches du CRM et de la facturation (anciens clients compris) : (crm, facturation).

    Chaque nuit, le script d'aujourd'hui demande à la facturation chaque fiche, une par une : 48 000 allers-retours,
    pour une poignée de fiches qui diffèrent.

        >>> crm, facturation = parc_exemple()
        >>> len(crm), len(facturation), sum(crm[i] != facturation.get(i) for i in crm)
        (47990, 47987, 27)
    """
    crm, facturation = _parc()
    return dict(crm), dict(facturation)


# ---------------------------------------------------------------------------- défi 287 : les messages de l'appli


@functools.lru_cache(maxsize=1)
def _livraisons() -> tuple:
    alea = random.Random(4787)
    envoyes, arrivees, perdus = [], [], {}
    for n in range(1, 301):
        source, client = f"tel-{n:03d}", f"C{n:04d}"
        t = DEBUT_MS + 7 * HEURE_MS + alea.randrange(HEURE_MS)
        valeurs = {"consentement_sms": alea.random() < 0.6, "consentement_email": alea.random() < 0.5}
        for numero in range(1, alea.randint(2, 40) + 1):
            t += alea.randint(5_000, 40 * 60_000)
            champ = alea.choice(("consentement_sms", "consentement_email"))
            valeurs[champ] = not valeurs[champ]
            message = {"source": source, "numero": numero, "client": client, "champ": champ,
                       "valeur": valeurs[champ], "envoye_le": t}
            envoyes.append(message)
            retard = alea.expovariate(1 / 3_000) if alea.random() < 0.85 else alea.randint(60_000, 30 * 60_000)
            arrivees.append((t + int(retard), message))
            if alea.random() < 0.08:                                    # pas d'accusé de réception : renvoyé
                arrivees.append((t + int(retard) + alea.randint(30_000, 15 * 60_000), message))
    # Quatre téléphones perdent un message en route (un tunnel, un avion) ; le soir, revenus sur le Wi-Fi, ils renvoient
    # tout ce qui suit leur dernier accusé de réception.
    derniers: dict[str, int] = {}
    for message in envoyes:
        derniers[message["source"]] = message["numero"]
    for source in alea.sample(sorted(s for s, n in derniers.items() if n >= 3), 4):
        perdus[source] = alea.randint(2, min(derniers[source], 30) - 1)
    arrivees = [(t, m) for t, m in arrivees if perdus.get(m["source"]) != m["numero"]]
    arrivees.sort(key=lambda a: (a[0], a[1]["source"], a[1]["numero"]))
    soir = max(t for t, _m in arrivees) + 60_000
    renvois = [m for m in envoyes if m["source"] in perdus and m["numero"] >= perdus[m["source"]]]
    journee = tuple(m for _t, m in arrivees)
    return journee + tuple(renvois), len(journee), tuple(envoyes), soir


def livraisons_exemple() -> list[dict]:
    """Les messages des 300 téléphones, dans l'ordre où ils ARRIVENT au serveur de l'appli (une journée, puis le soir).

    Un message : {"source" (le téléphone), "numero" (1, 2, 3… dans l'ordre où le téléphone les a écrits), "client",
    "champ", "valeur", "envoye_le"}. Le réseau mobile en livre certains en retard (donc dans le désordre), en livre
    d'autres deux fois (le téléphone, sans accusé de réception, renvoie), et en perd quelques-uns : le soir, les
    téléphones concernés renvoient tout ce qui suit leur dernier accusé de réception (les `RENVOIS` derniers messages).

        >>> livraisons = livraisons_exemple()
        >>> len(livraisons), livraisons[0]["source"], len(livraisons) - RENVOIS
        (6893, 'tel-104', 6861)
    """
    return [dict(m) for m in _livraisons()[0]]


def consentements(messages) -> dict[str, dict]:
    """{client: {champ: valeur}} en appliquant les messages dans l'ordre donné (le dernier appliqué l'emporte).

        >>> consentements([{"client": "C1", "champ": "consentement_sms", "valeur": True},
        ...                {"client": "C1", "champ": "consentement_sms", "valeur": False}])
        {'C1': {'consentement_sms': False}}
    """
    etat: dict[str, dict] = {}
    for message in messages:
        etat.setdefault(message["client"], {})[message["champ"]] = message["valeur"]
    return etat


def ecarts_consentement(etat) -> int:
    """Le nombre de clients dont un consentement, dans `etat`, n'est pas le DERNIER que le client a choisi.

        >>> ecarts_consentement({})          # aucun message appliqué : tous les clients sont faux
        300
    """
    attendu = consentements(_livraisons()[2])
    return sum(etat.get(client, {}) != valeurs for client, valeurs in attendu.items())


RENVOIS = len(_livraisons()[0]) - _livraisons()[1]

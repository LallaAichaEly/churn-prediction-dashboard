"""Série 46 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_46.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours un
`startswith` qui laisse « client » couvrir « clients:lire » (défi 278), un héritage suivi dans le mauvais sens ou
sans ensemble des rôles déjà vus (défi 279), une règle qui gagne parce qu'elle est la première alors qu'un refus la
suit (défi 280), un journal relu pour chaque compte (défi 281), une incompatibilité cherchée rôle par rôle (défi 282),
ou une délégation qui donne plus que ce que le délégant a encore (défi 283).

Fil rouge de la série : dans le doute, on refuse. Un motif mal écrit, un rôle inconnu, une règle mal formée, un
attribut manquant, une date sans fuseau : chacun lève une erreur au lieu d'accorder un accès « par défaut ».

1. une permission se compare SEGMENT par SEGMENT : « client » n'est pas un début de « clients:lire », et un motif
   plus précis (« clients:lire:telephone ») ne couvre pas une permission plus large (« clients:lire ») ;
2. les rôles hérités se suivent vers les PARENTS, avec un ensemble des rôles déjà vus : un cycle ne boucle pas, et
   un losange (deux chemins vers le même rôle) ne se parcourt qu'une fois ;
3. une décision d'accès : par défaut on REFUSE ; un refus qui s'applique l'emporte sur toute autorisation, quel que
   soit l'ordre des règles ; et toutes les règles sont vérifiées avant de décider ;
4. un droit DORMANT est un droit qu'on a sans s'en servir : on lit le journal UNE fois pour savoir qui a utilisé
   quoi, puis on compare ;
5. la séparation des tâches se juge sur TOUS les rôles d'une personne à la fois : aucun rôle n'est toxique seul,
   c'est leur cumul qui l'est ;
6. une délégation ne donne jamais plus que ce que le délégant peut, LUI, au moment où on s'en sert : on ne
   recopie pas ses droits, on les redemande.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

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

_COLLECTIONS = (list, tuple, set, frozenset)

# ============================================================================ Défi 278 : permission_couvre


def _segments(texte, nom: str) -> list[str]:
    if not isinstance(texte, str):
        raise TypeError(f"{nom} doit être un texte (str), pas {type(texte).__name__}")
    segments = texte.split(SEPARATEUR)
    if "" in segments:
        raise ValueError(f"{nom} {texte!r} : segment vide")
    return segments


def permission_couvre(motif, permission) -> bool:
    """Le motif d'un droit couvre-t-il la permission demandée ?"""
    morceaux = _segments(motif, "le motif")
    demandes = _segments(permission, "la permission")
    for segment in demandes:                   # on demande UNE action précise : ni joker, ni liste
        if JOKER in segment or LISTE in segment:
            raise ValueError(f"la permission {permission!r} doit être précise (ni « * » ni « , »)")
    valeurs = []                               # pour chaque segment du motif : None (joker) ou les valeurs permises
    for segment in morceaux:
        if segment == JOKER:
            valeurs.append(None)
            continue
        choix = segment.split(LISTE)
        if any(c == "" or JOKER in c for c in choix):
            raise ValueError(f"le motif {motif!r} est mal écrit ({segment!r})")
        valeurs.append(set(choix))
    # Un motif plus COURT couvre aussi les sous-permissions (ses segments manquants valent « * ») ; un motif plus LONG
    # ne couvre que si tous ses segments en trop sont des jokers (« clients:lire:* » couvre « clients:lire »).
    for i in range(max(len(valeurs), len(demandes))):
        permises = valeurs[i] if i < len(valeurs) else None
        if permises is None:
            continue
        if i >= len(demandes) or demandes[i] not in permises:
            return False
    return True


# ============================================================================ Défi 279 : roles_effectifs


def roles_effectifs(noms, heritage) -> frozenset:
    """Les rôles donnés, et tous ceux dont ils héritent, directement ou non."""
    if isinstance(noms, str) or not isinstance(noms, _COLLECTIONS):
        raise TypeError(f"noms doit être une liste de rôles, pas {type(noms).__name__}")
    if not isinstance(heritage, dict):
        raise TypeError(f"heritage doit être un dict, pas {type(heritage).__name__}")
    for nom in noms:
        if not isinstance(nom, str):
            raise TypeError(f"un nom de rôle est un texte, pas {type(nom).__name__}")
    vus: set[str] = set()
    a_voir = list(noms)                        # une pile : pas de récursion, pas de RecursionError
    while a_voir:
        role = a_voir.pop()
        if role in vus:                        # déjà vu : un cycle ou un losange s'arrête là
            continue
        if role not in heritage:
            raise ValueError(f"rôle inconnu : {role!r}")
        parents = heritage[role]
        if isinstance(parents, str) or not isinstance(parents, (list, tuple)):
            raise TypeError(f"les parents de {role!r} doivent être une liste, pas {type(parents).__name__}")
        vus.add(role)
        a_voir.extend(parents)                 # vers les PARENTS : ce dont le rôle hérite
    return frozenset(vus)


# ============================================================================ Défi 280 : decider


def _verifier_regles(regles, permission) -> list[bool]:
    """Vérifie TOUTES les règles (avant de décider) ; rend, pour chacune, si son motif couvre la permission."""
    if not isinstance(regles, (list, tuple)):
        raise TypeError(f"regles doit être une liste, pas {type(regles).__name__}")
    ids, couvertures = set(), []
    for regle in regles:
        if not isinstance(regle, dict):
            raise TypeError(f"une règle est un dict, pas {type(regle).__name__}")
        for cle in ("id", "effet", "permission"):
            if cle not in regle:
                raise ValueError(f"une règle n'a pas de {cle!r} : {regle!r}")
        if regle["id"] in ids:
            raise ValueError(f"deux règles s'appellent {regle['id']!r}")
        ids.add(regle["id"])
        if regle["effet"] not in (AUTORISER, REFUSER):
            raise ValueError(f"règle {regle['id']} : effet inconnu {regle['effet']!r}")
        if "roles" in regle and (isinstance(regle["roles"], str) or not isinstance(regle["roles"], _COLLECTIONS)):
            raise TypeError(f"règle {regle['id']} : roles doit être une liste")
        si = regle.get("si", [])
        if isinstance(si, str) or not isinstance(si, (list, tuple)):
            raise TypeError(f"règle {regle['id']} : si doit être une liste")
        for condition in si:
            if condition not in CONDITIONS:
                raise ValueError(f"règle {regle['id']} : condition inconnue {condition!r}")
        couvertures.append(permission_couvre(regle["permission"], permission))
    return couvertures


def decider(utilisateur, permission, client, regles) -> Decision:
    """Accès accordé ou refusé, et la règle qui en décide."""
    if not isinstance(utilisateur, dict) or not isinstance(client, dict):
        raise TypeError("utilisateur et client sont des dict")
    roles = utilisateur["roles"]
    if isinstance(roles, str) or not isinstance(roles, _COLLECTIONS):
        raise TypeError("les rôles de l'utilisateur sont une liste")
    couvertures = _verifier_regles(regles, permission)
    roles = set(roles)
    autorisation = None
    for regle, couvre in zip(regles, couvertures):
        if not couvre:
            continue
        if "roles" in regle and not roles & set(regle["roles"]):   # [] : ne vise personne
            continue
        # Les conditions en dernier : une condition lit un attribut ; s'il manque, KeyError remonte (on ne décide pas
        # sans savoir, et surtout pas « condition fausse », qui ferait sauter un refus).
        if not all(CONDITIONS[condition](utilisateur, client) for condition in regle.get("si", [])):
            continue
        if regle["effet"] == REFUSER:
            return Decision(False, regle["id"])           # un refus l'emporte, où qu'il soit dans la liste
        if autorisation is None:
            autorisation = regle["id"]                    # la PREMIÈRE autorisation, si aucun refus ne suit
    return Decision(autorisation is not None, autorisation)


# ============================================================================ Défi 281 : droits_dormants


def _instant(valeur, nom: str) -> datetime:
    if not isinstance(valeur, datetime):
        raise TypeError(f"{nom} doit être un datetime, pas {type(valeur).__name__}")
    if valeur.tzinfo is None or valeur.utcoffset() is None:
        raise ValueError(f"{nom} n'a pas de fuseau horaire : impossible de le comparer au journal")
    return valeur


def _vise(regle: dict, roles) -> bool:
    """La règle vise-t-elle quelqu'un qui a ces rôles (héritage compris) ? Sans "roles" : tout le monde."""
    return "roles" not in regle or bool(set(roles) & set(regle["roles"]))


def droits_dormants(utilisateurs, heritage, regles, journal, maintenant, jours=90) -> dict:
    """{identifiant: [règles d'autorisation qu'il a sans s'en être servi depuis `jours` jours]}."""
    _instant(maintenant, "maintenant")
    if isinstance(jours, bool) or not isinstance(jours, int):
        raise TypeError("jours est un nombre entier de jours")
    if jours <= 0:
        raise ValueError("jours doit être > 0")
    debut = maintenant - timedelta(days=jours)
    utilises = set()                           # (utilisateur, règle) : le journal n'est lu qu'UNE fois
    for entree in journal:
        if entree["regle"] is not None and debut <= entree["date"] <= maintenant:
            utilises.add((entree["utilisateur"], entree["regle"]))
    autorisations = [regle for regle in regles if regle["effet"] == AUTORISER]
    dormants = {}
    for utilisateur in utilisateurs:
        roles = roles_effectifs(utilisateur["roles"], heritage)
        ids = sorted(regle["id"] for regle in autorisations
                     if _vise(regle, roles) and (utilisateur["identifiant"], regle["id"]) not in utilises)
        if ids:
            dormants[utilisateur["identifiant"]] = ids
    return dormants


# ============================================================================ Défi 282 : conflits


def _a_le_droit(roles, permission: str, regles) -> bool:
    """Au moins pour un client : une autorisation le vise et couvre la permission, et aucun refus SANS condition."""
    accorde = False
    for regle in regles:
        if not _vise(regle, roles) or not permission_couvre(regle["permission"], permission):
            continue
        if regle["effet"] == REFUSER:
            if not regle.get("si"):
                return False                   # un refus sans condition retire le droit pour tous les clients
        else:
            accorde = True                     # une autorisation, même sous condition : le droit existe
    return accorde


def conflits(utilisateurs, heritage, regles, incompatibles) -> list[tuple[str, str, str]]:
    """[(identifiant, permission 1, permission 2)] pour chaque compte qui cumule deux permissions incompatibles."""
    paires = []
    for paire in incompatibles:
        if isinstance(paire, str) or not isinstance(paire, (list, tuple)):
            raise TypeError("chaque incompatibilité est un couple (permission, permission)")
        if len(paire) != 2:
            raise ValueError(f"une incompatibilité relie DEUX permissions : {paire!r}")
        paires.append(tuple(paire))
    resultat = []
    for utilisateur in utilisateurs:
        roles = roles_effectifs(utilisateur["roles"], heritage)      # TOUS ses rôles à la fois
        for premiere, seconde in paires:
            if _a_le_droit(roles, premiere, regles) and _a_le_droit(roles, seconde, regles):
                resultat.append((utilisateur["identifiant"], premiere, seconde))
    return resultat


# ============================================================================ Défi 283 : Delegations


class Delegations:
    """Des délégations datées, révocables, qui ne donnent jamais plus que ce que le délégant peut lui-même."""

    def __init__(self, annuaire, heritage, regles) -> None:
        if not isinstance(annuaire, dict) or not isinstance(heritage, dict):
            raise TypeError("annuaire et heritage sont des dict")
        if not isinstance(regles, (list, tuple)):
            raise TypeError("regles est une liste")
        self._annuaire = annuaire              # PAS de copie : l'annuaire vivant fait foi (départs, changements)
        self._heritage = heritage
        self._regles = regles
        self._delegations: dict[int, tuple] = {}
        self._revoquees: set[int] = set()

    def ajouter(self, delegant, delegue, motif, debut, fin) -> int:
        for identifiant in (delegant, delegue):
            if not isinstance(identifiant, str):
                raise TypeError("un identifiant est un texte")
            if identifiant not in self._annuaire:
                raise ValueError(f"compte inconnu : {identifiant!r}")
        if delegant == delegue:
            raise ValueError("on ne se délègue pas un droit à soi-même")
        permission_couvre(motif, "x")          # vérifie le motif (TypeError, ValueError)
        _instant(debut, "debut")
        _instant(fin, "fin")
        if not debut < fin:
            raise ValueError("la délégation doit commencer avant de finir")
        numero = len(self._delegations) + 1
        self._delegations[numero] = (delegant, delegue, motif, debut, fin)   # l'IDENTIFIANT, pas une copie du compte
        return numero

    def revoquer(self, numero) -> None:
        if isinstance(numero, bool) or not isinstance(numero, int):
            raise TypeError("le numéro d'une délégation est un entier")
        if numero not in self._delegations or numero in self._revoquees:
            raise ValueError(f"pas de délégation active n°{numero}")
        self._revoquees.add(numero)

    def _compte(self, identifiant) -> dict:
        utilisateur = self._annuaire[identifiant]
        return {**utilisateur, "roles": roles_effectifs(utilisateur["roles"], self._heritage)}

    def decider(self, identifiant, permission, client, maintenant) -> Decision:
        if not isinstance(identifiant, str):
            raise TypeError("un identifiant est un texte")
        if identifiant not in self._annuaire:
            raise ValueError(f"compte inconnu : {identifiant!r}")
        _instant(maintenant, "maintenant")
        propre = decider(self._compte(identifiant), permission, client, self._regles)
        if propre.autorise or propre.regle is not None:   # accordé, ou refusé EXPRÈS : une délégation n'y change rien
            return propre
        for numero, (delegant, delegue, motif, debut, fin) in sorted(self._delegations.items()):
            if numero in self._revoquees or delegue != identifiant or not debut <= maintenant < fin:
                continue
            if not permission_couvre(motif, permission) or delegant not in self._annuaire:
                continue                       # un délégant parti de l'annuaire ne délègue plus rien
            # Ce que le délégant peut LUI, maintenant, pour CE client (son agence, ses rôles du jour), sans ses propres
            # délégations reçues : pas de chaîne.
            pour_lui = decider(self._compte(delegant), permission, client, self._regles)
            if pour_lui.autorise:
                return Decision(True, pour_lui.regle, numero)
        return propre


# ---------------------------------------------------------------------------- des tests, comme dans tests/


def test_permission_couvre():
    assert permission_couvre("clients", "clients:lire:telephone")
    assert permission_couvre("clients:lire,modifier", "clients:modifier")
    assert not permission_couvre("client", "clients:lire")                    # pas un début de texte
    assert not permission_couvre("clients:lire:telephone", "clients:lire")    # plus précis ne couvre pas plus large
    assert not permission_couvre("gestes:invalider", "gestes:valider")        # pas une sous-chaîne


def test_roles_effectifs():
    heritage = {"a": ["b"], "b": ["a", "c"], "c": []}                           # un cycle : pas de boucle sans fin
    assert roles_effectifs(["a"], heritage) == {"a", "b", "c"}
    assert roles_effectifs(["c"], heritage) == {"c"}                            # vers les parents seulement


def test_decider():
    regles = [{"id": "A", "effet": AUTORISER, "permission": "clients"},
              {"id": "B", "effet": REFUSER, "permission": "clients:exporter", "si": ["client_salarie"]}]
    utilisateur, client = {"roles": []}, {"salarie": True}
    assert decider(utilisateur, "clients:exporter", client, regles) == Decision(False, "B")
    assert decider(utilisateur, "clients:lire", client, regles) == Decision(True, "A")
    assert decider(utilisateur, "rapports:lire", client, regles) == Decision(False, None)


def test_droits_dormants():
    regles = [{"id": "A", "effet": AUTORISER, "permission": "x", "roles": ["r"]}]
    journal = [{"date": REVUE_LE - timedelta(days=100), "utilisateur": "u1", "regle": "A"}]
    utilisateurs = [{"identifiant": "u1", "roles": ["r"]}]
    assert droits_dormants(utilisateurs, {"r": []}, regles, journal, REVUE_LE) == {"u1": ["A"]}


def test_conflits():
    heritage = {"proposeur": [], "valideur": []}
    regles = [{"id": "P", "effet": AUTORISER, "permission": "gestes:proposer", "roles": ["proposeur"]},
              {"id": "V", "effet": AUTORISER, "permission": "gestes:valider", "roles": ["valideur"]}]
    utilisateurs = [{"identifiant": "u1", "roles": ["proposeur", "valideur"]}]
    assert conflits(utilisateurs, heritage, regles, INCOMPATIBLES) == [("u1", "gestes:proposer", "gestes:valider")]


def test_delegations():
    annuaire = {"chef": {"roles": ["chef"], "agence": "A"}, "adjoint": {"roles": [], "agence": "A"}}
    regles = [{"id": "R", "effet": AUTORISER, "permission": "gestes:valider", "roles": ["chef"], "si": ["meme_agence"]}]
    delegations = Delegations(annuaire, {"chef": []}, regles)
    numero = delegations.ajouter("chef", "adjoint", "gestes", REVUE_LE, REVUE_LE + timedelta(days=14))
    client = {"agence": "A"}
    assert delegations.decider("adjoint", "gestes:valider", client, REVUE_LE) == Decision(True, "R", numero)
    annuaire["chef"]["roles"] = []                                             # le chef perd son rôle
    assert delegations.decider("adjoint", "gestes:valider", client, REVUE_LE) == Decision(False, None)

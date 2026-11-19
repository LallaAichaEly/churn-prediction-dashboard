"""Série 46 (qui a le droit de voir quoi) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 278`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (une couverture par expression régulière, un
  parcours en largeur avec `deque` ou récursif avec un ensemble des rôles vus, une décision en deux listes, un journal
  rangé dans un dict de dernières utilisations, des conflits par compréhension, des délégations rangées en liste) ;
- les solutions trop lentes (relire le journal pour chaque compte, suivre chaque chemin d'héritage) sont arrêtées et
  expliquées, et tout ce qui est mal formé (un motif vide, un rôle inconnu, une règle mal écrite, une date sans fuseau)
  est REFUSÉ par une erreur, jamais par un accès ;
- les références du correcteur tiennent face à des vérifications indépendantes : une traduction des motifs en
  expressions régulières, un parcours en largeur, pandas sur les 172 000 accès du journal, une recherche exhaustive
  sur les clients de l'exemple ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient (36
  comptes exportent les 1 200 clients aujourd'hui, 27 avec les règles, 20 une fois l'héritage du support retiré ; 11
  comptes lisent la fiche d'un salarié au lieu de 300 ; 85 droits dormants sur 48 comptes ; 45 cumuls interdits, 31
  ensuite).
"""

from __future__ import annotations

import doctest
import re
import sys
import types
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import defis.acces as acces  # noqa: E402
import defis.corrige_serie_46 as corrige  # noqa: E402
from defis.acces import (  # noqa: E402
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
from defis.checks_serie_46 import (  # noqa: E402
    _BUGS_CONFLITS,
    _BUGS_COUVRE,
    _BUGS_DECIDER,
    _BUGS_DELEGATIONS,
    _BUGS_DORMANTS,
    _BUGS_ROLES,
    _AOUT,
    _LILLE,
    SERIE,
    _annuaire,
    _cas_278,
    _cas_279,
    _cas_280,
    _cas_281,
    _cas_282,
    _jouer,
    _ref_conflits,
    _ref_couvre,
    _ref_decider,
    _ref_dormants,
    _ref_roles,
    _ref_sorties,
    _RefDelegations,
    _scenarios_283,
    _sorties_egales,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

# Attention : ce module ne doit PAS définir de noms `permission_couvre`, `roles_effectifs` ou `decider` au niveau du
# module : pendant les défis 280 à 283, le correcteur prête les siens au module des fonctions vérifiées (ici, ce
# fichier de tests).

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}), DEFIS[nom])


def _echec(nom, objet, *morceaux):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message
    return resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


def _avec_les_gardes(nom, calcul):
    """Une version boguée qui garde les vérifications du corrigé (les erreurs attendues), pour isoler LE bug."""
    def fonction(*args, **kwargs):
        getattr(corrige, nom)(*args, **kwargs)        # lève les mêmes erreurs que le corrigé
        return calcul(*args, **kwargs)
    fonction.__name__ = nom
    return fonction


def _classe_boguee(bug):
    class Delegations(_RefDelegations):
        def __init__(self, annuaire, heritage, regles):
            super().__init__(annuaire, heritage, regles, bug)
    return Delegations


# --------------------------------------------------------------------------- Défi 278 : permission_couvre


@pytest.mark.parametrize("bug,morceau", [
    ("startswith", "DÉBUT de texte"),
    ("fnmatch", "`fnmatch`"),
    ("zip", "`zip(motif, permission)`"),
    ("longueur", "même longueur"),
    ("liste", "LISTE de valeurs"),
    ("sous_chaine", "SOUS-CHAÎNE"),
])
def test_couvre_nomme_l_erreur(bug, morceau):
    _echec("permission_couvre", _avec_les_gardes("permission_couvre", lambda m, p: _ref_couvre(m, p, bug)), morceau)


def test_couvre_qui_oublie_le_return_false():
    def fonction(motif, permission):
        return True if corrige.permission_couvre(motif, permission) else None

    _echec("permission_couvre", fonction, "None", "return")


def test_couvre_qui_rend_un_entier():
    _echec("permission_couvre", lambda m, p: int(corrige.permission_couvre(m, p)), "vrai booléen")


def test_couvre_qui_accepte_un_motif_vide():
    def fonction(motif, permission):
        if motif == "":
            return True                                   # la faille d'aujourd'hui
        return corrige.permission_couvre(motif, permission)

    _echec("permission_couvre", fonction, "Un motif vide", "ValueError")


def test_couvre_qui_accepte_un_joker_dans_la_demande():
    def fonction(motif, permission):
        if permission.endswith(":*"):
            return motif == permission
        return corrige.permission_couvre(motif, permission)

    _echec("permission_couvre", fonction, "PRÉCISE")


def test_couvre_accepte_une_expression_reguliere():
    def fonction(motif, permission):
        corrige.permission_couvre(motif, permission)      # les mêmes erreurs
        morceaux = []
        for segment in motif.split(SEPARATEUR):
            morceaux.append("[^:]+" if segment == JOKER else "(?:" + "|".join(map(re.escape, segment.split(LISTE))) + ")")
        while morceaux and morceaux[-1] == "[^:]+":       # « clients:lire:* » couvre « clients:lire »
            morceaux.pop()
        return re.fullmatch(":".join(morceaux) + ("(?::.*)?" if morceaux else ".*"), permission) is not None

    _reussi("permission_couvre", fonction)


def test_couvre_accepte_une_version_recursive():
    def fonction(motif, permission):
        corrige.permission_couvre(motif, permission)

        def couvre(m, p):
            if not m:
                return True
            if not p:
                return all(s == JOKER for s in m)
            return (m[0] == JOKER or p[0] in m[0].split(LISTE)) and couvre(m[1:], p[1:])

        return couvre(motif.split(SEPARATEUR), permission.split(SEPARATEUR))

    _reussi("permission_couvre", fonction)


# --------------------------------------------------------------------------- Défi 279 : roles_effectifs


@pytest.mark.parametrize("bug,morceau", [
    ("un_etage", "UN étage"),
    ("sens_inverse", "à l'ENVERS"),
    ("sans_depart", "DÉPART"),
])
def test_roles_nomme_l_erreur(bug, morceau):
    _echec("roles_effectifs", _avec_les_gardes("roles_effectifs", lambda n, h: _ref_roles(n, h, bug)), morceau)


def test_roles_qui_rend_une_liste():
    _echec("roles_effectifs", lambda n, h: sorted(corrige.roles_effectifs(n, h)), "ensemble")


def test_roles_sans_ensemble_des_vus_est_arrete():
    def fonction(noms, heritage):
        corrige.roles_effectifs(noms, heritage)

        def monter(role):
            resultat = {role}
            for parent in heritage[role]:
                resultat |= monter(parent)
            return resultat

        return frozenset().union(*map(monter, noms))

    _echec("roles_effectifs", fonction, "RecursionError")


def test_roles_qui_suit_chaque_chemin_est_arrete():
    def fonction(noms, heritage):
        corrige.roles_effectifs(noms, heritage)
        vus, pile = set(), [(n, frozenset()) for n in noms]
        while pile:                                         # un chemin à la fois (et pas de boucle sur un cycle)
            role, chemin = pile.pop()
            vus.add(role)
            pile.extend((p, chemin | {role}) for p in heritage[role] if p not in chemin | {role})
        return frozenset(vus)

    _echec("roles_effectifs", fonction, "beaucoup trop lent", "2 ** 60")


def test_roles_qui_modifie_l_heritage():
    def fonction(noms, heritage):
        resultat = corrige.roles_effectifs(noms, heritage)
        for nom in noms:
            heritage[nom].extend(sorted(resultat - {nom} - set(heritage[nom])))   # « un cache » dans la configuration
        return resultat

    _echec("roles_effectifs", fonction, "MODIFIÉ")


def test_roles_qui_ignore_un_role_inconnu():
    def fonction(noms, heritage):
        if isinstance(noms, str) or not isinstance(noms, (list, tuple, set, frozenset)):
            raise TypeError("une liste de rôles")
        return corrige.roles_effectifs([n for n in noms if not isinstance(n, str) or n in heritage], heritage)

    _echec("roles_effectifs", fonction, "faute de frappe")


def test_roles_qui_parcourt_les_lettres_d_un_texte():
    def fonction(noms, heritage):
        if isinstance(noms, str):
            noms = list(noms)
        return corrige.roles_effectifs(noms, heritage)

    _echec("roles_effectifs", fonction, "LETTRES")


def test_roles_accepte_un_parcours_en_largeur():
    def fonction(noms, heritage):
        corrige.roles_effectifs(noms, heritage)
        vus, file = set(noms), deque(noms)
        while file:
            for parent in heritage[file.popleft()]:
                if parent not in vus:
                    vus.add(parent)
                    file.append(parent)
        return vus

    _reussi("roles_effectifs", fonction)


def test_roles_accepte_une_recursion_avec_les_vus():
    def fonction(noms, heritage):
        corrige.roles_effectifs(noms, heritage)
        vus = set()

        def visiter(role):
            if role not in vus:
                vus.add(role)
                for parent in heritage[role]:
                    visiter(parent)

        for nom in noms:
            visiter(nom)
        return frozenset(vus)

    _reussi("roles_effectifs", fonction)


# --------------------------------------------------------------------------- Défi 280 : decider


@pytest.mark.parametrize("bug,morceau", [
    ("premiere", "PREMIÈRE règle"),
    ("autorisation_gagne", "une autorisation l'emporte"),
    ("defaut_autorise", "vous AUTORISEZ"),
    ("sans_conditions", "ignorez les conditions"),
    ("une_condition", "(`any`)"),
    ("roles_vides_tous", "vise PERSONNE"),
    ("sans_roles", "ignorez les rôles"),
    ("egalite", "`==`"),
    ("dernier_id", "PREMIER refus"),
])
def test_decider_nomme_l_erreur(bug, morceau):
    _echec("decider", _avec_les_gardes("decider", lambda *args: _ref_decider(*args, bug=bug)), morceau)


def test_decider_qui_avale_l_attribut_manquant():
    def fonction(*args):                                  # les vérifications de la référence, sans celles du corrigé
        return _ref_decider(*args, bug="erreur_avalee")    # (qui lèverait la KeyError attendue)

    _echec("decider", fonction, "KeyError", "refus saute")


def test_decider_qui_rend_un_booleen():
    _echec("decider", lambda *args: corrige.decider(*args).autorise, "au lieu d'une Decision")


def test_decider_qui_ne_verifie_que_ce_qu_il_lit():
    def fonction(utilisateur, permission, client, regles):
        for regle in regles:                                # une seule boucle : vérifier, puis décider, en marchant
            if regle.get("effet") not in (AUTORISER, REFUSER):
                raise ValueError("effet inconnu")
            if regle["effet"] == REFUSER and corrige.permission_couvre(regle["permission"], permission) \
                    and "roles" not in regle and not regle.get("si"):
                return Decision(False, regle["id"])
        return corrige.decider(utilisateur, permission, client, regles)

    _echec("decider", fonction, "TOUTES les règles AVANT")


def test_decider_qui_modifie_l_utilisateur():
    def fonction(utilisateur, permission, client, regles):
        if not isinstance(utilisateur["roles"], str):
            utilisateur["roles"] = sorted(utilisateur["roles"])      # « pour avoir une liste triée »
        return corrige.decider(utilisateur, permission, client, regles)

    _echec("decider", fonction, "MODIFIÉ")


def test_decider_accepte_deux_listes_et_utilise_la_couverture_pretee(monkeypatch):
    module = types.ModuleType("faux_module_280")
    exec(  # le fichier d'un·e apprenant·e : `permission_couvre` pas encore écrite, `decider` s'en sert
        "from defis.acces import AUTORISER, REFUSER, CONDITIONS, Decision\n"
        "def permission_couvre(motif, permission):\n"
        "    raise NotImplementedError\n"
        "def decider(utilisateur, permission, client, regles):\n"
        "    if not isinstance(utilisateur, dict) or not isinstance(client, dict):\n"
        "        raise TypeError('des dict')\n"
        "    if isinstance(utilisateur['roles'], str) or not isinstance(regles, (list, tuple)):\n"
        "        raise TypeError('des listes')\n"
        "    vus, couvre = set(), []\n"
        "    for r in regles:\n"
        "        if not isinstance(r, dict) or any(c not in r for c in ('id', 'effet', 'permission')):\n"
        "            raise ValueError('règle mal écrite')\n"
        "        if r['id'] in vus or r['effet'] not in (AUTORISER, REFUSER):\n"
        "            raise ValueError('règle mal écrite')\n"
        "        vus.add(r['id'])\n"
        "        for cle in ('roles', 'si'):\n"
        "            if cle in r and (isinstance(r[cle], str) or not isinstance(r[cle], (list, tuple, set, frozenset))):\n"
        "                raise TypeError(cle)\n"
        "        if any(c not in CONDITIONS for c in r.get('si', [])):\n"
        "            raise ValueError('condition inconnue')\n"
        "        couvre.append(permission_couvre(r['permission'], permission))\n"
        "    mes_roles = set(utilisateur['roles'])\n"
        "    applicables = [r for r, c in zip(regles, couvre) if c\n"
        "                   and ('roles' not in r or any(x in mes_roles for x in r['roles']))\n"
        "                   and all(CONDITIONS[s](utilisateur, client) for s in r.get('si', []))]\n"
        "    refus = [r['id'] for r in applicables if r['effet'] == REFUSER]\n"
        "    oui = [r['id'] for r in applicables if r['effet'] == AUTORISER]\n"
        "    if refus:\n"
        "        return Decision(False, refus[0])\n"
        "    return Decision(True, oui[0]) if oui else Decision(False)\n", module.__dict__)
    monkeypatch.setitem(sys.modules, "faux_module_280", module)
    _reussi("decider", module.decider)
    with pytest.raises(NotImplementedError):            # la vôtre est remise en place
        module.permission_couvre("a", "a")


# --------------------------------------------------------------------------- Défi 281 : droits_dormants


@pytest.mark.parametrize("bug,morceau", [
    ("sans_heritage", "rôles DIRECTS"),
    ("refus_inclus", "Un refus n'est pas un droit"),
    ("sans_periode", "TOUT le journal"),
    ("debut_exclu", "PILE au début"),
    ("futur_compte", "APRÈS `maintenant`"),
    ("par_permission", "par la PERMISSION"),
    ("vides_inclus", "AU MOINS un droit dormant"),
    ("roles_vides_tous", "ne vise PERSONNE"),
    ("non_tries", "triés"),
])
def test_dormants_nomme_l_erreur(bug, morceau):
    _echec("droits_dormants", _avec_les_gardes("droits_dormants", lambda *args, **kw: _ref_dormants(*args, **kw, bug=bug)),
           morceau)


def test_dormants_qui_relit_le_journal_pour_chaque_compte_est_arrete():
    def fonction(utilisateurs, heritage, regles, journal, maintenant, jours=90):
        corrige.droits_dormants(utilisateurs, heritage, regles, journal, maintenant, jours)
        debut = maintenant - timedelta(days=jours)
        resultat = {}
        for u in utilisateurs:
            roles = corrige.roles_effectifs(u["roles"], heritage)
            utilises = set()
            for e in journal:                               # tout le journal, pour CHAQUE compte
                if e["utilisateur"] == u["identifiant"] and e["regle"] and debut <= e["date"] <= maintenant:
                    utilises.add(e["regle"])
            ids = sorted(r["id"] for r in regles if r["effet"] == AUTORISER
                         and ("roles" not in r or roles & set(r["roles"])) and r["id"] not in utilises)
            if ids:
                resultat[u["identifiant"]] = ids
        return resultat

    _echec("droits_dormants", fonction, "beaucoup trop lent", "UNE fois")


def test_dormants_qui_lit_le_journal_deux_fois():
    def fonction(utilisateurs, heritage, regles, journal, maintenant, jours=90):
        debut = maintenant - timedelta(days=jours)
        n = sum(1 for _ in journal)                         # « combien d'entrées ? » : la première lecture…
        utilises = {(e["utilisateur"], e["regle"]) for e in journal if debut <= e["date"] <= maintenant} if n else set()
        resultat = {}
        for u in utilisateurs:
            roles = corrige.roles_effectifs(u["roles"], heritage)
            ids = sorted(r["id"] for r in regles if r["effet"] == AUTORISER
                         and ("roles" not in r or roles & set(r["roles"])) and (u["identifiant"], r["id"]) not in utilises)
            if ids:
                resultat[u["identifiant"]] = ids
        return resultat

    _echec("droits_dormants", _avec_les_gardes("droits_dormants", fonction), "UNE fois")


def test_dormants_sans_verifier_le_fuseau():
    def fonction(utilisateurs, heritage, regles, journal, maintenant, jours=90):
        if maintenant.tzinfo is None:
            maintenant = maintenant.replace(tzinfo=timezone.utc)
        return corrige.droits_dormants(utilisateurs, heritage, regles, journal, maintenant, jours)

    _echec("droits_dormants", fonction, "fuseau")


def test_dormants_accepte_un_dict_des_dernieres_utilisations():
    def fonction(utilisateurs, heritage, regles, journal, maintenant, jours=90):
        corrige.droits_dormants(utilisateurs, heritage, regles, [], maintenant, jours)    # les mêmes erreurs
        derniere = {}
        for e in journal:
            if e["regle"] is not None and e["date"] <= maintenant:
                cle = (e["utilisateur"], e["regle"])
                if cle not in derniere or e["date"] > derniere[cle]:
                    derniere[cle] = e["date"]
        limite = maintenant - timedelta(days=jours)
        resultat = {}
        for u in utilisateurs:
            roles = corrige.roles_effectifs(u["roles"], heritage)
            ids = [r["id"] for r in regles if r["effet"] == AUTORISER and ("roles" not in r or roles & set(r["roles"]))
                   and derniere.get((u["identifiant"], r["id"]), limite - timedelta(seconds=1)) < limite]
            if ids:
                resultat[u["identifiant"]] = sorted(ids)
        return resultat

    _reussi("droits_dormants", fonction)


# --------------------------------------------------------------------------- Défi 282 : conflits


@pytest.mark.parametrize("bug,morceau", [
    ("par_role", "rôle par rôle"),
    ("egalite", "`==`"),
    ("sans_heritage", "rôles DIRECTS"),
    ("refus_conditionnels", "AVEC conditions"),
    ("refus_ignores", "ignorez les refus"),
    ("conditions_exigees", "SOUS condition"),
    ("roles_vides_tous", "ne vise PERSONNE"),
    ("paires_d_abord", "ordre"),
])
def test_conflits_nomme_l_erreur(bug, morceau):
    _echec("conflits", _avec_les_gardes("conflits", lambda *args: _ref_conflits(*args, bug=bug)), morceau)


def test_conflits_qui_rend_des_listes():
    _echec("conflits", lambda *args: [list(c) for c in corrige.conflits(*args)], "TUPLE")


def test_conflits_accepte_une_comprehension():
    def fonction(utilisateurs, heritage, regles, incompatibles):
        corrige.conflits(utilisateurs, heritage, regles, incompatibles)

        def peut(roles, permission):
            vises = [r for r in regles if ("roles" not in r or roles & set(r["roles"]))
                     and corrige.permission_couvre(r["permission"], permission)]
            return (any(r["effet"] == AUTORISER for r in vises)
                    and not any(r["effet"] == REFUSER and not r.get("si") for r in vises))

        effectifs = [(u["identifiant"], corrige.roles_effectifs(u["roles"], heritage)) for u in utilisateurs]
        return [(i, a, b) for i, roles in effectifs for a, b in incompatibles if peut(roles, a) and peut(roles, b)]

    _reussi("conflits", fonction)


# --------------------------------------------------------------------------- Défi 283 : Delegations


@pytest.mark.parametrize("bug,morceau", [
    ("photo", "recopié le compte"),
    ("sans_controle", "sans demander au délégant"),
    ("attributs_du_delegue", "attributs du DÉLÉGUÉ"),
    ("chaine", "reçues"),
    ("fin_incluse", "fin est EXCLUE"),
    ("debut_ignore", "vérifiez aussi le début"),
    ("refus_contourne", "REFUS explicite"),
    ("revocation_ignoree", "révoquée"),
    ("derniere", "PREMIÈRE"),
    ("delegant_parti", "quitté l'annuaire"),
    ("annuaire_copie", "recopié l'annuaire"),
])
def test_delegations_nomme_l_erreur(bug, morceau):
    _echec("Delegations", _classe_boguee(bug), morceau)


def test_delegations_partagees_entre_objets():
    class Delegations(_RefDelegations):
        _commun: dict = {}

        def __init__(self, annuaire, heritage, regles):
            super().__init__(annuaire, heritage, regles)
            self._delegations = Delegations._commun          # un attribut de CLASSE : partagé

    _echec("Delegations", Delegations, "partagent")


def test_delegations_qui_modifie_l_annuaire():
    class Delegations(_RefDelegations):
        def _compte(self, identifiant):
            compte = self._annuaire[identifiant]
            compte["roles"] = sorted(_ref_roles(compte["roles"], self._heritage))   # écrit dans l'annuaire
            return compte

    _echec("Delegations", Delegations, "MODIFIÉ l'annuaire")


def test_delegations_qui_copie_l_annuaire():
    class Delegations(_RefDelegations):
        def __init__(self, annuaire, heritage, regles):
            import copy
            super().__init__(copy.deepcopy(annuaire), heritage, regles)

    _echec("Delegations", Delegations, "recopié l'annuaire")


def test_delegations_sans_numero_a_partir_de_1():
    class Delegations(_RefDelegations):
        def ajouter(self, *args):
            return super().ajouter(*args) - 1

    _echec("Delegations", Delegations, "numéro 1")


def test_delegations_acceptent_une_liste_et_le_decider_prete(monkeypatch):
    module = types.ModuleType("faux_module_283")
    exec(  # le fichier d'un·e apprenant·e : les défis 278 à 280 pas encore écrits, `Delegations` s'en sert
        "from datetime import datetime\n"
        "from defis.acces import Decision\n"
        "def permission_couvre(motif, permission):\n"
        "    raise NotImplementedError\n"
        "def roles_effectifs(noms, heritage):\n"
        "    raise NotImplementedError\n"
        "def decider(utilisateur, permission, client, regles):\n"
        "    raise NotImplementedError\n"
        "def _date(x):\n"
        "    if not isinstance(x, datetime):\n"
        "        raise TypeError('une date')\n"
        "    if x.utcoffset() is None:\n"
        "        raise ValueError('sans fuseau')\n"
        "class Delegations:\n"
        "    def __init__(self, annuaire, heritage, regles):\n"
        "        if not isinstance(annuaire, dict):\n"
        "            raise TypeError('un dict')\n"
        "        self.annuaire, self.heritage, self.regles, self.liste = annuaire, heritage, regles, []\n"
        "    def ajouter(self, delegant, delegue, motif, debut, fin):\n"
        "        if not isinstance(delegant, str) or not isinstance(delegue, str):\n"
        "            raise TypeError('des identifiants')\n"
        "        if delegant not in self.annuaire or delegue not in self.annuaire or delegant == delegue:\n"
        "            raise ValueError('comptes')\n"
        "        permission_couvre(motif, 'x')\n"
        "        _date(debut); _date(fin)\n"
        "        if fin <= debut:\n"
        "            raise ValueError('dates')\n"
        "        self.liste.append({'de': delegant, 'a': delegue, 'motif': motif, 'debut': debut, 'fin': fin,\n"
        "                           'active': True})\n"
        "        return len(self.liste)\n"
        "    def revoquer(self, numero):\n"
        "        if type(numero) is not int:\n"
        "            raise TypeError('un entier')\n"
        "        if not 1 <= numero <= len(self.liste) or not self.liste[numero - 1]['active']:\n"
        "            raise ValueError('inconnue')\n"
        "        self.liste[numero - 1]['active'] = False\n"
        "    def _vu(self, identifiant):\n"
        "        compte = dict(self.annuaire[identifiant])\n"
        "        compte['roles'] = roles_effectifs(compte['roles'], self.heritage)\n"
        "        return compte\n"
        "    def decider(self, identifiant, permission, client, maintenant):\n"
        "        if identifiant not in self.annuaire:\n"
        "            raise ValueError('inconnu')\n"
        "        _date(maintenant)\n"
        "        propre = decider(self._vu(identifiant), permission, client, self.regles)\n"
        "        if propre.autorise or propre.regle:\n"
        "            return propre\n"
        "        for numero, d in enumerate(self.liste, start=1):\n"
        "            if d['active'] and d['a'] == identifiant and d['debut'] <= maintenant < d['fin'] \\\n"
        "                    and d['de'] in self.annuaire and permission_couvre(d['motif'], permission):\n"
        "                pour_lui = decider(self._vu(d['de']), permission, client, self.regles)\n"
        "                if pour_lui.autorise:\n"
        "                    return Decision(True, pour_lui.regle, numero)\n"
        "        return propre\n", module.__dict__)
    monkeypatch.setitem(sys.modules, "faux_module_283", module)
    _reussi("Delegations", module.Delegations)
    with pytest.raises(NotImplementedError):
        module.decider({}, "x", {}, [])


# --------------------------------------------------------------------------- les références tiennent


def _couvre_par_regex(motif: str, permission: str) -> bool:
    """Une traduction indépendante : chaque segment du motif devient une expression régulière."""
    morceaux = motif.split(SEPARATEUR)
    while len(morceaux) > 1 and morceaux[-1] == JOKER:
        morceaux.pop()
    if morceaux == [JOKER]:
        return True
    motifs = ["[^:]+" if s == JOKER else "(?:" + "|".join(map(re.escape, s.split(LISTE))) + ")" for s in morceaux]
    return re.fullmatch(":".join(motifs) + "(?::[^:]+)*", permission) is not None


def test_la_reference_de_couverture_est_une_expression_reguliere():
    for motif, permission in _cas_278():
        assert _ref_couvre(motif, permission) == _couvre_par_regex(motif, permission), (motif, permission)
        assert corrige.permission_couvre(motif, permission) == _ref_couvre(motif, permission)


def test_la_reference_des_roles_est_un_parcours_en_largeur():
    for noms, heritage in _cas_279():
        vus, file = set(noms), deque(noms)
        while file:
            for parent in heritage[file.popleft()]:
                if parent not in vus:
                    vus.add(parent)
                    file.append(parent)
        assert _ref_roles(noms, heritage) == vus == corrige.roles_effectifs(noms, heritage)


def test_la_reference_de_decision_suit_les_trois_principes():
    """Refus explicite > autorisation explicite > refus implicite, et l'ordre des règles ne change pas la décision."""
    for utilisateur, permission, client, regles in _cas_280():
        decision = _ref_decider(utilisateur, permission, client, regles)
        assert decision == corrige.decider(utilisateur, permission, client, regles)
        a_l_envers = _ref_decider(utilisateur, permission, client, list(regles)[::-1])
        assert a_l_envers.autorise == decision.autorise
        applicables = [r for r in regles if corrige.permission_couvre(r["permission"], permission)
                       and ("roles" not in r or set(r["roles"]) & set(utilisateur["roles"]))
                       and all(CONDITIONS[c](utilisateur, client) for c in r.get("si", []))]
        effets = {r["effet"] for r in applicables}
        assert decision.autorise == (effets == {AUTORISER})


def test_la_reference_des_droits_dormants_est_celle_de_pandas():
    journal = pd.DataFrame(journal_exemple())
    utilisateurs, heritage, regles = utilisateurs_exemple(), heritage_exemple(), regles_exemple()
    periode = journal[(journal["date"] >= REVUE_LE - pd.Timedelta(days=90)) & (journal["date"] <= REVUE_LE)]
    utilises = set(periode.dropna(subset=["regle"]).groupby(["utilisateur", "regle"]).size().index)
    droits = pd.DataFrame([(u["identifiant"], r["id"]) for u in utilisateurs for r in regles
                           if r["effet"] == AUTORISER and set(r["roles"]) & _ref_roles(u["roles"], heritage)],
                          columns=["utilisateur", "regle"])
    dormants = droits[[(u, r) not in utilises for u, r in zip(droits["utilisateur"], droits["regle"])]]
    attendu = {u: sorted(g["regle"]) for u, g in dormants.groupby("utilisateur")}
    assert _ref_dormants(utilisateurs, heritage, regles, journal_exemple(), REVUE_LE) == attendu
    assert corrige.droits_dormants(utilisateurs, heritage, regles, journal_exemple(), REVUE_LE) == attendu


def test_la_reference_des_conflits_face_aux_clients_de_l_exemple():
    """« Au moins pour un client » se juge sur les règles, sans énumérer les clients : sur les 1 200 clients de
    l'exemple, la recherche exhaustive trouve les mêmes conflits, sauf la direction, dont l'agence (« Siège ») n'a
    aucun client AUJOURD'HUI. La revue la signale quand même : il suffit d'un client au siège."""
    utilisateurs, heritage, regles, clients = utilisateurs_exemple(), heritage_exemple(), regles_exemple(), clients_exemple()
    profils = list({tuple(sorted((k, v) for k, v in c.items() if k != "identifiant")): c for c in clients}.values())
    exhaustif = []
    for u in utilisateurs:
        compte = {**u, "roles": _ref_roles(u["roles"], heritage)}
        for a, b in INCOMPATIBLES:
            if all(any(_ref_decider(compte, p, c, regles).autorise for c in profils) for p in (a, b)):
                exhaustif.append((u["identifiant"], a, b))
    reference = _ref_conflits(utilisateurs, heritage, regles, INCOMPATIBLES)
    assert reference == corrige.conflits(utilisateurs, heritage, regles, INCOMPATIBLES)
    assert set(exhaustif) <= set(reference)
    direction = {u["identifiant"] for u in utilisateurs if u["roles"] == ["direction"]}
    assert {c[0] for c in set(reference) - set(exhaustif)} == direction and len(direction) == 3


def test_la_reference_des_delegations_est_celle_du_corrige():
    for operations in _scenarios_283():
        annuaire = _annuaire()
        obtenus = _jouer(corrige.Delegations(annuaire, heritage_exemple(), regles_exemple()), annuaire, operations)
        assert all(_sorties_egales(o, a) for o, a in zip(obtenus, _ref_sorties(operations)))


def test_les_scenarios_de_delegation_utilisent_vraiment_des_delegations():
    for operations in _scenarios_283()[3:]:
        sorties = _ref_sorties(operations)
        par_delegation = [s for s in sorties if isinstance(s, Decision) and s.delegation is not None]
        refus = [s for s in sorties if isinstance(s, Decision) and not s.autorise]
        assert len(par_delegation) >= 5 and len(refus) >= 20


# --------------------------------------------------------------------------- outils communs, histoire, corrigé


def test_les_outils_communs():
    utilisateurs, heritage, regles = utilisateurs_exemple(), heritage_exemple(), regles_exemple()
    assert len(utilisateurs) == 300 and len({u["identifiant"] for u in utilisateurs}) == 300
    assert utilisateurs_exemple() is not utilisateurs and utilisateurs_exemple() == utilisateurs
    assert all(role in heritage for u in utilisateurs for role in u["roles"])
    assert all(role in heritage for r in regles for role in r["roles"])
    assert all(parent in heritage for parents in heritage.values() for parent in parents)
    assert all(c in CONDITIONS for r in regles for c in r["si"]) and len({r["id"] for r in regles}) == 14
    clients = clients_exemple()
    assert len(clients) == 1200 and {c["agence"] for c in clients} == set(acces.AGENCES)
    journal = journal_exemple()
    assert all(e["date"].tzinfo is not None for e in journal[:1000])
    assert journal == sorted(journal, key=lambda e: (e["date"], e["utilisateur"]))
    assert {e["regle"] for e in journal} <= {r["id"] for r in regles if r["effet"] == AUTORISER} | {None}
    assert REVUE_LE - timedelta(days=120) <= journal[0]["date"] and journal[-1]["date"] < REVUE_LE
    assert regle_actuelle({"droits": [""]}, "modeles:deployer") and not regle_actuelle({"droits": ["rapports"]}, "x")
    assert exposition(lambda u, p, c: c["salarie"], [{}], clients, "x") == {"comptes": 1, "couples": 39}
    assert Decision(True, "R01") == Decision(True, "R01", None) and INCOMPATIBLES[0] == ("gestes:proposer",
                                                                                         "gestes:valider")
    with pytest.raises(KeyError):                       # un attribut absent : on ne décide pas sans savoir
        CONDITIONS["client_salarie"]({}, {"agence": "Lille"})


def test_l_histoire_tient():
    utilisateurs, clients, heritage, regles = utilisateurs_exemple(), clients_exemple(), heritage_exemple(), regles_exemple()
    salaries = [c for c in clients if c["salarie"]]
    sans_sms = [c for c in clients if not c["consentement_sms"]]
    aujourd_hui = lambda u, p, c: regle_actuelle(u, p)  # noqa: E731

    def nouveau(h):
        effectifs = {u["identifiant"]: {**u, "roles": corrige.roles_effectifs(u["roles"], h)} for u in utilisateurs}
        return lambda u, p, c: corrige.decider(effectifs[u["identifiant"]], p, c, regles).autorise

    assert exposition(aujourd_hui, utilisateurs, clients, "clients:exporter") == {"comptes": 36, "couples": 43200}
    assert exposition(aujourd_hui, utilisateurs, salaries, "clients:lire") == {"comptes": 300, "couples": 11700}
    prestataire = [u for u in utilisateurs if "prestataire_sms" in u["roles"]]
    assert exposition(aujourd_hui, prestataire, sans_sms, "clients:lire:telephone") == {"comptes": 2, "couples": 720}
    support = [u for u in utilisateurs if "support_informatique" in u["roles"]]
    assert len(support) == 7 and all(regle_actuelle(u, "modeles:deployer") for u in support)

    assert exposition(nouveau(heritage), utilisateurs, clients, "clients:exporter") == {"comptes": 27, "couples": 14745}
    assert exposition(nouveau(heritage), utilisateurs, salaries, "clients:lire") == {"comptes": 11, "couples": 429}
    assert exposition(nouveau(heritage), prestataire, sans_sms, "clients:lire:telephone") == {"comptes": 0, "couples": 0}
    dormants = corrige.droits_dormants(utilisateurs, heritage, regles, journal_exemple(), REVUE_LE)
    assert (len(dormants), sum(map(len, dormants.values()))) == (48, 85)
    assert all(dormants[u["identifiant"]] == ["R11"] for u in support)
    managers_sans_export = [u["identifiant"] for u in utilisateurs if u["roles"] == ["manager_agence"]
                            and "R04" in dormants.get(u["identifiant"], [])]
    assert managers_sans_export == ["u023", "u088", "u197", "u219"]
    conflits = corrige.conflits(utilisateurs, heritage, regles, INCOMPATIBLES)
    assert len(conflits) == 45
    managers = {u["identifiant"] for u in utilisateurs if u["roles"] == ["manager_agence"]}
    assert managers <= {i for i, a, _ in conflits if a == "gestes:proposer"} and len(managers) == 12

    corrige_ = {**heritage, "support_informatique": []}
    assert exposition(nouveau(corrige_), utilisateurs, clients, "clients:exporter") == {"comptes": 20, "couples": 6345}
    assert exposition(nouveau(corrige_), utilisateurs, salaries, "clients:lire") == {"comptes": 4, "couples": 156}
    assert len(corrige.conflits(utilisateurs, corrige_, regles, INCOMPATIBLES)) == 31

    annuaire = {u["identifiant"]: u for u in utilisateurs}       # la manager de Lille part en congés
    delegations = corrige.Delegations(annuaire, heritage, regles)
    n = delegations.ajouter("u001", "u002", "gestes:valider", _AOUT, _AOUT + timedelta(days=14))
    assert delegations.decider("u002", "gestes:valider", _LILLE, _AOUT + timedelta(days=1)) == Decision(True, "R03", n)
    assert not delegations.decider("u002", "clients:exporter", _LILLE, _AOUT + timedelta(days=1)).autorise


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    serie = next(s for s in series() if s.code == "46")
    assert [r.statut for r in evaluer(serie)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(serie, corrige=True)] == [REUSSI] * 6


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_46 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(acces)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_46 import *` (dans les exemples) doit donner les objets du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 38 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    utilisateurs, heritage, regles = utilisateurs_exemple()[:40], heritage_exemple(), regles_exemple()
    journal = journal_exemple()[:2000]
    copies = repr((utilisateurs, heritage, regles, journal))
    corrige.roles_effectifs(["direction"], heritage)
    for u in utilisateurs[:5]:
        corrige.decider({**u, "roles": corrige.roles_effectifs(u["roles"], heritage)}, "clients:lire", _LILLE, regles)
    corrige.droits_dormants(utilisateurs, heritage, regles, journal, REVUE_LE)
    corrige.conflits(utilisateurs, heritage, regles, INCOMPATIBLES)
    annuaire = {u["identifiant"]: u for u in utilisateurs}
    d = corrige.Delegations(annuaire, heritage, regles)
    d.ajouter("u001", "u002", "gestes", _AOUT, _AOUT + timedelta(days=1))
    d.decider("u002", "gestes:valider", _LILLE, _AOUT)
    assert repr((utilisateurs, heritage, regles, journal)) == copies


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(ValueError):
        corrige.permission_couvre("", "clients:lire")
    with pytest.raises(ValueError):
        corrige.permission_couvre("clients", "clients:*")
    with pytest.raises(TypeError):
        corrige.roles_effectifs("conseiller", heritage_exemple())
    with pytest.raises(ValueError):
        corrige.roles_effectifs(["inconnu"], heritage_exemple())
    with pytest.raises(ValueError):
        corrige.decider({"roles": []}, "x", {}, [{"id": "a", "effet": "deny", "permission": "x"}])
    with pytest.raises(KeyError):
        corrige.decider({"roles": ["conseiller"], "agence": "Lille"}, "clients:lire", {}, regles_exemple())
    with pytest.raises(ValueError):
        corrige.droits_dormants([], {}, [], [], datetime(2026, 11, 2))
    with pytest.raises(ValueError):
        corrige.conflits([], {}, [], [("a", "b", "c")])
    d = corrige.Delegations({"a": {"roles": []}, "b": {"roles": []}}, {}, [])
    with pytest.raises(ValueError):
        d.ajouter("a", "a", "x", _AOUT, _AOUT + timedelta(days=1))
    with pytest.raises(ValueError):
        d.revoquer(1)


def test_les_bugs_annonces_existent_tous():
    """Chaque erreur classique décrite par le correcteur a un test ci-dessus (et inversement)."""
    assert set(_BUGS_COUVRE) == {"startswith", "fnmatch", "zip", "longueur", "liste", "sous_chaine"}
    assert set(_BUGS_ROLES) == {"un_etage", "sens_inverse", "sans_depart"}
    assert set(_BUGS_DECIDER) == {"premiere", "autorisation_gagne", "defaut_autorise", "sans_conditions",
                                  "une_condition", "roles_vides_tous", "sans_roles", "egalite", "dernier_id"}
    assert set(_BUGS_DORMANTS) == {"sans_heritage", "refus_inclus", "sans_periode", "debut_exclu", "futur_compte",
                                   "par_permission", "vides_inclus", "roles_vides_tous", "non_tries"}
    assert set(_BUGS_CONFLITS) == {"par_role", "egalite", "sans_heritage", "refus_conditionnels", "refus_ignores",
                                   "conditions_exigees", "roles_vides_tous", "paires_d_abord"}
    assert set(_BUGS_DELEGATIONS) == {"photo", "sans_controle", "attributs_du_delegue", "chaine", "fin_incluse",
                                      "debut_ignore", "refus_contourne", "revocation_ignoree", "derniere",
                                      "delegant_parti", "annuaire_copie"}


def test_chaque_bug_change_au_moins_un_cas():
    cas_278 = _cas_278()[:486]                          # les cas écrits à la main, sans les cas tirés au hasard
    for bug in _BUGS_COUVRE:
        assert any(_ref_couvre(m, p, bug) != _ref_couvre(m, p) for m, p in cas_278), bug
    for bug in _BUGS_ROLES:
        assert any(_ref_roles(n, h, bug) != _ref_roles(n, h) for n, h in _cas_279()), bug
    for bug in _BUGS_DECIDER:
        assert any(_ref_decider(*x, bug=bug) != _ref_decider(*x) for x in _cas_280()), bug
    petits_281 = _cas_281()[:-1]
    for bug in _BUGS_DORMANTS:
        assert any(_ref_dormants(*x, bug=bug) != _ref_dormants(*x) for x in petits_281), bug
    petits_282 = _cas_282()[:3]
    for bug in _BUGS_CONFLITS:
        assert any(_ref_conflits(*x, bug=bug) != _ref_conflits(*x) for x in petits_282), bug
    for bug in _BUGS_DELEGATIONS:
        assert any(not all(_sorties_egales(o, a) for o, a in zip(_ref_sorties(ops, bug), _ref_sorties(ops)))
                   for ops in _scenarios_283()[:3]), bug

"""Série 47 (garder deux systèmes d'accord) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 284`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (des vecteurs vus comme des ensembles d'événements,
  un `match` pour l'écriture canonique, une descente récursive dans l'arbre, une salle d'attente en tas, une horloge
  qui garde un couple (l, c), une fusion par copie profonde) ;
- les solutions trop coûteuses (toutes les feuilles demandées, une liste parcourue à chaque message) sont arrêtées et
  expliquées, et tout ce qui est mal formé (un compteur négatif, un `set` dans une fiche, une estampille en liste, une
  fiche à trois clés) est refusé par une erreur ;
- les références du correcteur tiennent face à des vérifications indépendantes : des ensembles d'événements, des textes
  JSON écrits à la main, une comparaison directe des deux parcs, un modèle de la livraison dans l'ordre, les propriétés
  de l'horloge hybride, et la convergence de trois copies qui s'échangent leurs états dans un ordre quelconque ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (347 champs faux, 41 clients ressuscités et 186 SMS malgré un refus aujourd'hui ; 7 corrections perdues avec l'heure
  du serveur, aucune avec l'horloge hybride ; 37 fiches trouvées en 82 allers-retours ; 23 consentements faux à
  l'arrivée, aucun avec le récepteur).
"""

from __future__ import annotations

import copy
import doctest
import hashlib
import heapq
import json
import math
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import defis.corrige_serie_47 as corrige  # noqa: E402
import defis.synchro as synchro  # noqa: E402
from defis.checks_serie_47 import (  # noqa: E402
    _BUGS_DIFFERENCES,
    _BUGS_EMPREINTE,
    _BUGS_FUSIONNER,
    _BUGS_HORLOGE,
    _BUGS_RECEPTEUR,
    _BUGS_VERSIONS,
    SERIE,
    _cas_284,
    _cas_285,
    _cas_286,
    _cas_289,
    _jouer_287,
    _jouer_288,
    _ref_differences,
    _ref_empreinte,
    _ref_fusionner,
    _ref_versions,
    _RefHorloge,
    _RefRecepteur,
    _repliques,
    _scenarios_287,
    _scenarios_288,
)
from defis.synchro import (  # noqa: E402
    APRES,
    AVANT,
    CONCURRENTES,
    DERIVE_MAX_MS,
    EGALES,
    RECUL_NTP,
    RENVOIS,
    VIDE,
    HorlogePhysique,
    Replique,
    bilan,
    consentements,
    ecarts_consentement,
    evenements_exemple,
    heure_physique,
    livraisons_exemple,
    parc_exemple,
    rejouer,
    synchro_actuelle,
    versions_exemple,
    vue,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

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


def _debut(conseil: str) -> str:
    return conseil[:50]


# ============================================================================ 284 comparer_versions


@pytest.mark.parametrize("bug", list(_BUGS_VERSIONS))
def test_versions_nomme_l_erreur(bug):
    _echec("comparer_versions", lambda a, b: _ref_versions(a, b, bug), _debut(_BUGS_VERSIONS[bug]))


def test_versions_qui_modifie_le_vecteur():
    def avec_setdefault(a, b):
        for r in a.keys() | b.keys():
            a.setdefault(r, 0)
            b.setdefault(r, 0)
        return corrige.comparer_versions(a, b)

    _echec("comparer_versions", avec_setdefault, "MODIFIÉ", "a.get(r, 0)")


def test_versions_qui_rend_autre_chose():
    _echec("comparer_versions", lambda a, b: corrige.comparer_versions(a, b).upper(), "l'une des constantes")


def test_versions_qui_lit_une_replique_absente_avec_des_crochets():
    def crochets(a, b):
        corrige.comparer_versions(a, b)
        a_dans_b = all(a[r] <= b[r] for r in a.keys() | b.keys())
        b_dans_a = all(b[r] <= a[r] for r in a.keys() | b.keys())
        return EGALES if a_dans_b and b_dans_a else AVANT if a_dans_b else APRES if b_dans_a else CONCURRENTES

    _echec("comparer_versions", crochets, "KeyError", ".get(")


def test_versions_qui_accepte_un_booleen():
    def sans_bool(a, b):
        if isinstance(a, dict) and isinstance(b, dict) and any(n is True for n in [*a.values(), *b.values()]):
            return EGALES
        return corrige.comparer_versions(a, b)

    _echec("comparer_versions", sans_bool, "`True` est un `int`")


def _versions_par_ensembles(a, b):
    corrige.comparer_versions(a, b)                     # les mêmes vérifications
    ea = {(r, i) for r, n in a.items() for i in range(1, n + 1)}
    eb = {(r, i) for r, n in b.items() for i in range(1, n + 1)}
    return EGALES if ea == eb else AVANT if ea < eb else APRES if ea > eb else CONCURRENTES


def test_versions_accepte_des_ensembles_d_evenements():
    _reussi("comparer_versions", _versions_par_ensembles)


def test_la_reference_des_versions_est_celle_des_ensembles_d_evenements():
    for a, b in _cas_284():
        assert _ref_versions(a, b) == _versions_par_ensembles(a, b) == corrige.comparer_versions(a, b)


# ============================================================================ 285 empreinte


@pytest.mark.parametrize("bug", list(_BUGS_EMPREINTE))
def test_empreinte_nomme_l_erreur(bug):
    _echec("empreinte", lambda fiche: _ref_empreinte(fiche, bug), _debut(_BUGS_EMPREINTE[bug]))


def test_empreinte_avec_hash():
    _echec("empreinte", lambda fiche: hash(json.dumps(fiche, sort_keys=True)), "PYTHONHASHSEED")


def test_empreinte_en_md5():
    def md5(fiche):
        corrige.empreinte(fiche)
        texte = json.dumps(corrige._canonique(fiche, "x"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.md5(texte.encode()).hexdigest()

    _echec("empreinte", md5, "MD5")


def test_empreinte_en_octets():
    _echec("empreinte", lambda fiche: bytes.fromhex(corrige.empreinte(fiche)), "rendez un texte")


def test_empreinte_en_majuscules():
    _echec("empreinte", lambda fiche: corrige.empreinte(fiche).upper(), "minuscules")


def test_empreinte_qui_normalise_la_fiche_sur_place():
    def sur_place(fiche):
        resultat = corrige.empreinte(fiche)
        for cle, valeur in list(fiche.items()):
            if isinstance(valeur, str):
                fiche[cle] = unicodedata.normalize("NFC", valeur)
        return resultat

    _echec("empreinte", sur_place, "MODIFIÉ")


@pytest.mark.parametrize("fiche,morceau", [
    ({"rappels": {"lundi", "jeudi"}}, "Un set n'a pas d'ordre"),
    ({1: "a"}, "même empreinte"),
    ({"x": float("nan")}, "NaN"),
    ({"é": 1, unicodedata.normalize("NFD", "é"): 2}, "ambiguë"),
])
def test_empreinte_qui_accepte_l_inacceptable(fiche, morceau):
    def laxiste(f):
        if f == fiche or (isinstance(f, dict) and repr(f) == repr(fiche)):
            return "0" * 64
        return corrige.empreinte(f)

    _echec("empreinte", laxiste, morceau)


def _empreinte_match(fiche):
    if not isinstance(fiche, dict):
        raise TypeError("une fiche est un dict")

    def canon(valeur):
        match valeur:
            case None | bool() | int():
                return valeur
            case float() if math.isfinite(valeur):
                return valeur
            case float():
                raise ValueError("pas fini")
            case str():
                return unicodedata.normalize("NFC", valeur)
            case list() | tuple():
                return [canon(v) for v in valeur]
            case dict():
                if not all(isinstance(k, str) for k in valeur):
                    raise TypeError("clé")
                copie = {unicodedata.normalize("NFC", k): canon(v) for k, v in valeur.items()}
                if len(copie) != len(valeur):
                    raise ValueError("deux clés identiques")
                return copie
            case _:
                raise TypeError(type(valeur).__name__)

    texte = json.dumps(canon(fiche), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def test_empreinte_accepte_un_match():
    _reussi("empreinte", _empreinte_match)


def test_la_reference_d_empreinte_face_a_des_textes_ecrits_a_la_main():
    nfd = unicodedata.normalize("NFD", "é")
    for fiche, texte in (({}, "{}"), ({"b": [1, 2.5, None, True], "a": "é"}, '{"a":"é","b":[1,2.5,null,true]}'),
                         ({"ville": "Orl" + nfd + "ans", "n": (1, 2)}, '{"n":[1,2],"ville":"Orléans"}'),
                         ({"z": {"y": 1, "x": {"w": "😀"}}}, '{"z":{"x":{"w":"😀"},"y":1}}'),
                         ({"f": 1, nfd: 2}, '{"f":1,"é":2}')):
        attendu = hashlib.sha256(texte.encode("utf-8")).hexdigest()
        assert _ref_empreinte(fiche) == corrige.empreinte(fiche) == _empreinte_match(fiche) == attendu
    for fiche in _cas_285():
        assert _ref_empreinte(fiche) == corrige.empreinte(fiche)


# ============================================================================ 286 differences


@pytest.mark.parametrize("bug", list(_BUGS_DIFFERENCES))
def test_differences_nomme_l_erreur(bug):
    _echec("differences", lambda a, b: _ref_differences(a, b, bug), _debut(_BUGS_DIFFERENCES[bug]))


def test_differences_qui_demande_toutes_les_feuilles():
    _echec("differences", lambda a, b: _ref_differences(a, b, "toutes_les_feuilles"), "questions à la distante",
           "une question suffit")


def test_differences_qui_descend_partout():
    def partout_sauf_racine(locale, distante):
        if locale.empreinte("") == distante.empreinte(""):
            return {}
        return _ref_differences(locale, distante, "partout")

    _echec("differences", partout_sauf_racine, "questions à la distante", "sous-arbres identiques")


def test_differences_qui_rapatrie_toutes_les_fiches_d_une_feuille_identique():
    def feuilles_en_trop(locale, distante):
        resultat = corrige.differences(locale, distante)
        if resultat and locale.profondeur == 3:
            distante.feuille("000")                    # une feuille de plus, pour rien
        return resultat

    _echec("differences", feuilles_en_trop, "questions à la distante")


def test_differences_sans_verifier_les_profondeurs():
    def sans_verifier(locale, distante):
        try:
            return corrige.differences(locale, distante)
        except ValueError:
            return {}

    _echec("differences", sans_verifier, "profondeur")


def test_differences_qui_oublie_la_profondeur_zero():
    def toujours_des_enfants(locale, distante):
        locale.enfants("")                              # une question à la locale : gratuite, mais pas en profondeur 0
        return corrige.differences(locale, distante)

    _echec("differences", toujours_des_enfants, "la racine \"\" est une feuille")


def _differences_recursives(locale, distante):
    if locale.profondeur != distante.profondeur:
        raise ValueError("profondeurs différentes")
    ecarts = {}

    def descendre(prefixe):
        if len(prefixe) == locale.profondeur:
            ici, la_bas = locale.feuille(prefixe), distante.feuille(prefixe)
            ecarts.update({i: "locale" for i in set(ici) - set(la_bas)})
            ecarts.update({i: "distante" for i in set(la_bas) - set(ici)})
            ecarts.update({i: "modifiee" for i in set(ici) & set(la_bas) if ici[i] != la_bas[i]})
            return
        ici, la_bas = locale.enfants(prefixe), distante.enfants(prefixe)
        for enfant in ici:
            if ici[enfant] != la_bas[enfant]:
                descendre(enfant)

    if locale.empreinte("") != distante.empreinte(""):
        descendre("")
    return ecarts


def test_differences_accepte_une_descente_recursive():
    _reussi("differences", _differences_recursives)


def test_la_reference_des_differences_est_une_comparaison_directe():
    for _nom, ici, la_bas, profondeur in _cas_286():
        direct = {i: "locale" for i in ici.keys() - la_bas.keys()}
        direct |= {i: "distante" for i in la_bas.keys() - ici.keys()}
        direct |= {i: "modifiee" for i in ici.keys() & la_bas.keys() if ici[i] != la_bas[i]}
        assert _ref_differences(*_repliques(ici, la_bas, profondeur)) == direct
        assert corrige.differences(*_repliques(ici, la_bas, profondeur)) == direct


def test_les_cas_de_l_arbre_couvrent_ce_qu_ils_annoncent():
    cas = {nom: (ici, la_bas, p) for nom, ici, la_bas, p in _cas_286()}
    ici, la_bas, p = cas["des différences toutes dans la même feuille"]
    feuilles = {synchro.seau(i, p) for i in ici if ici[i] != la_bas[i]}
    assert len(feuilles) == 1 and len([i for i in ici if ici[i] != la_bas[i]]) == 3
    assert any(p == 0 for _i, _l, p in cas.values()) and any(p == 1 for _i, _l, p in cas.values())


# ============================================================================ 287 Recepteur


def _classe_recepteur(bug):
    return type("Recepteur", (_RefRecepteur,), {"__init__": lambda self: _RefRecepteur.__init__(self, bug)})


@pytest.mark.parametrize("bug", list(_BUGS_RECEPTEUR))
def test_recepteur_nomme_l_erreur(bug):
    _echec("Recepteur", _classe_recepteur(bug), _debut(_BUGS_RECEPTEUR[bug]))


def test_recepteur_partage_entre_objets():
    class Partage(corrige.Recepteur):
        _prochain: dict = {}
        _attente: dict = {}

        def __init__(self):
            pass

    _echec("Recepteur", Partage, "attribut de CLASSE")


class _RecepteurEnListe:
    """Juste, mais les messages qui attendent sont dans une liste parcourue à chaque fois."""

    def __init__(self):
        self.prochain, self.attente = {}, []

    def recevoir(self, message):
        corrige.Recepteur().recevoir(message)           # les mêmes vérifications (un récepteur jetable)
        source, numero = message["source"], message["numero"]
        prochain = self.prochain.get(source, 1)
        if numero < prochain:
            return []
        for m in self.attente:
            if m["source"] == source and m["numero"] == numero:
                if m != message:
                    raise ValueError("conflit")
                return []
        if numero > prochain:
            self.attente.append(message)
            return []
        appliques = [message]
        prochain += 1
        trouve = True
        while trouve:
            trouve = False
            for m in self.attente:
                if m["source"] == source and m["numero"] == prochain:
                    self.attente.remove(m)
                    appliques.append(m)
                    prochain += 1
                    trouve = True
                    break
        self.prochain[source] = prochain
        return appliques

    def en_attente(self):
        return len(self.attente)

    def manquants(self):
        resultat = {}
        for m in self.attente:
            resultat.setdefault(m["source"], set()).add(m["numero"])
        return {s: [n for n in range(self.prochain.get(s, 1), max(ns)) if n not in ns] for s, ns in resultat.items()}


def test_recepteur_en_liste_est_arrete():
    _echec("Recepteur", _RecepteurEnListe, "beaucoup trop lent", "dict {numéro: message}")


def test_recepteur_qui_modifie_le_message():
    class Marque(corrige.Recepteur):
        def recevoir(self, message):
            resultat = super().recevoir(message)
            message["recu"] = True
            return resultat

    _echec("Recepteur", Marque, "MODIFIÉ")


class _RecepteurEnTas:
    """Une autre écriture juste : un tas (heapq) par source, et un dict pour retrouver les doublons."""

    def __init__(self):
        self.prochain, self.tas, self.par_numero = {}, {}, {}

    def recevoir(self, message):
        corrige.Recepteur().recevoir(message)
        source, numero = message["source"], message["numero"]
        prochain = self.prochain.get(source, 1)
        if numero < prochain:
            return []
        deja = self.par_numero.get((source, numero))
        if deja is not None:
            if deja != message:
                raise ValueError("conflit")
            return []
        heapq.heappush(self.tas.setdefault(source, []), numero)
        self.par_numero[(source, numero)] = message
        appliques = []
        tas = self.tas[source]
        while tas and tas[0] == prochain:
            heapq.heappop(tas)
            appliques.append(self.par_numero.pop((source, prochain)))
            prochain += 1
        self.prochain[source] = prochain
        if not tas:
            del self.tas[source]
        return appliques

    def en_attente(self):
        return len(self.par_numero)

    def manquants(self):
        return {s: sorted(set(range(self.prochain.get(s, 1), max(t))) - set(t)) for s, t in self.tas.items()}


def test_recepteur_accepte_un_tas():
    _reussi("Recepteur", _RecepteurEnTas)


def test_la_reference_du_recepteur_livre_chaque_message_une_fois_dans_l_ordre():
    alea = random.Random(47)
    for _ in range(30):
        envoyes = [{"source": f"t{s}", "numero": n, "x": alea.random()} for s in range(4)
                   for n in range(1, alea.randint(1, 30))]
        livres = envoyes + alea.sample(envoyes, len(envoyes) // 3)
        alea.shuffle(livres)
        recepteur = _RefRecepteur()
        appliques = [m for message in livres for m in recepteur.recevoir(message)]
        par_source = {}
        for message in appliques:
            par_source.setdefault(message["source"], []).append(message["numero"])
        assert sorted(appliques, key=lambda m: (m["source"], m["numero"])) == \
            sorted(envoyes, key=lambda m: (m["source"], m["numero"]))
        assert all(numeros == list(range(1, len(numeros) + 1)) for numeros in par_source.values())
        assert recepteur.en_attente() == 0 and recepteur.manquants() == {}
    for operations in _scenarios_287():
        assert _jouer_287(_RefRecepteur, operations) == _jouer_287(corrige.Recepteur, operations)


# ============================================================================ 288 HorlogeHybride


@pytest.mark.parametrize("bug", list(_BUGS_HORLOGE))
def test_horloge_nomme_l_erreur(bug):
    _echec("HorlogeHybride", lambda r, p, d=DERIVE_MAX_MS: _RefHorloge(r, p, d, bug), _debut(_BUGS_HORLOGE[bug]))


def test_horloge_partagee_entre_objets():
    class Partagee:
        l = c = 0                                       # des attributs de CLASSE : une seule horloge pour toutes

        def __init__(self, replique, physique, derive_max=DERIVE_MAX_MS):
            corrige.HorlogeHybride(replique, physique, derive_max)
            self.replique, self.physique = replique, physique

        def estampiller(self):
            pt = self.physique()
            if pt > Partagee.l:
                Partagee.l, Partagee.c = pt, 0
            else:
                Partagee.c += 1
            return (Partagee.l, Partagee.c, self.replique)

        def recevoir(self, estampille):
            return self.estampiller()

    _echec("HorlogeHybride", Partagee, "attributs de classe")


def test_horloge_qui_accepte_une_heure_flottante():
    class Flottante(corrige.HorlogeHybride):
        def _maintenant(self):
            return int(self._physique())

    _echec("HorlogeHybride", Flottante, "time.time_ns()")


def test_horloge_qui_accepte_une_liste():
    class Liste(corrige.HorlogeHybride):
        def recevoir(self, estampille):
            return super().recevoir(tuple(estampille))

    _echec("HorlogeHybride", Liste, "tuple(...)")


class _HorlogeEnCouple:
    """Une autre écriture juste : l'état est un couple (l, c), et le compteur est le plus grand des candidats."""

    def __init__(self, replique, physique, derive_max=DERIVE_MAX_MS):
        corrige.HorlogeHybride(replique, physique, derive_max)       # les mêmes vérifications
        self.replique, self.physique, self.derive_max, self.etat = replique, physique, derive_max, (0, 0)

    def _pt(self):
        pt = self.physique()
        if isinstance(pt, bool) or not isinstance(pt, int):
            raise TypeError("des millisecondes entières")
        return pt

    def estampiller(self):
        pt = self._pt()
        l, c = self.etat
        self.etat = (pt, 0) if pt > l else (l, c + 1)
        return (*self.etat, self.replique)

    def recevoir(self, estampille):
        corrige._estampille(estampille)
        lm, cm, _ = estampille
        pt = self._pt()
        if lm - pt > self.derive_max:
            raise ValueError("trop d'avance")
        l, c = self.etat
        nouveau = max(l, lm, pt)
        compteur = max(cc for ll, cc in ((l, c + 1), (lm, cm + 1), (pt, 0)) if ll == nouveau)
        self.etat = (nouveau, compteur)
        return (nouveau, compteur, self.replique)


def test_horloge_accepte_un_couple():
    _reussi("HorlogeHybride", _HorlogeEnCouple)


def test_la_reference_de_l_horloge_tient_ses_promesses():
    for _nom, operations in _scenarios_288():
        heure = [0]
        horloge = _RefHorloge("crm", lambda: heure[0])
        precedente = None
        for op in operations:
            heure[0] = op[1]
            try:
                estampille = horloge.estampiller() if op[0] == "e" else horloge.recevoir(op[2])
            except ValueError:
                assert op[0] == "r" and op[2][0] - op[1] > DERIVE_MAX_MS
                continue
            assert precedente is None or estampille > precedente           # strictement croissantes
            assert estampille[0] >= op[1]                                  # jamais sous l'heure physique
            if op[0] == "r":
                assert estampille[:2] > op[2][:2]                          # après ce qu'on a reçu
            precedente = estampille
        assert _jouer_288(_RefHorloge, operations) == _jouer_288(corrige.HorlogeHybride, operations)


# ============================================================================ 289 fusionner


@pytest.mark.parametrize("bug", list(_BUGS_FUSIONNER))
def test_fusionner_nomme_l_erreur(bug):
    _echec("fusionner", lambda a, b: _ref_fusionner(a, b, bug), _debut(_BUGS_FUSIONNER[bug]))


def test_fusionner_par_copie_superficielle():
    def superficielle(a, b):
        resultat = corrige.fusionner(a, b)
        copie = dict(a)
        for client, fiche in resultat.items():
            if client in copie:
                copie[client]["champs"].update(fiche["champs"])
        return resultat

    _echec("fusionner", superficielle, "MODIFIÉ", "NOUVEAUX")


def test_fusionner_qui_partage_les_fiches():
    def partage(a, b):
        resultat = corrige.fusionner(a, b)
        for client, fiche in a.items():
            if resultat.get(client) == fiche:
                resultat[client] = fiche
        return resultat

    _echec("fusionner", partage, "PARTAGE")


def test_fusionner_qui_depend_de_l_ordre_sans_bug_connu():
    def premier_arrive(a, b):
        resultat = corrige.fusionner(a, b)
        for client in resultat:
            for champ, (valeur, estampille) in list(resultat[client]["champs"].items()):
                if client in a and champ in a[client]["champs"] and estampille[2] == "factu":
                    resultat[client]["champs"][champ] = a[client]["champs"][champ]
        return resultat

    _echec("fusionner", premier_arrive, "fusionner(b, a)")


def _fusion_par_copie(a, b):
    corrige.fusionner(a, b)                             # les mêmes vérifications
    resultat = copy.deepcopy(a)
    for client, fiche in b.items():
        cible = resultat.setdefault(client, {"champs": {}, "efface": None})
        if fiche["efface"] is not None and (cible["efface"] is None or fiche["efface"] > cible["efface"]):
            cible["efface"] = fiche["efface"]
        for champ, (valeur, estampille) in fiche["champs"].items():
            if champ not in cible["champs"] or estampille > cible["champs"][champ][1]:
                cible["champs"][champ] = (valeur, estampille)
    for fiche in resultat.values():
        if fiche["efface"] is not None:
            fiche["champs"] = {k: v for k, v in fiche["champs"].items() if v[1] > fiche["efface"]}
    return resultat


def test_fusionner_accepte_une_copie_profonde():
    _reussi("fusionner", _fusion_par_copie)


def test_la_reference_de_fusion_converge():
    etats = [x for _nom, x, _y in _cas_289()[-200:]]
    alea = random.Random(4789)
    for _ in range(100):
        x, y, z = alea.sample(etats, 3)
        assert _ref_fusionner(x, y) == _ref_fusionner(y, x)
        assert _ref_fusionner(_ref_fusionner(x, y), z) == _ref_fusionner(x, _ref_fusionner(y, z))
        propre = _ref_fusionner(x, {})
        assert _ref_fusionner(propre, propre) == propre
    for _ in range(30):                                 # trois copies qui s'échangent leurs états au hasard
        copies = alea.sample(etats, 3)
        for _echange in range(12):
            i, j = alea.sample(range(3), 2)
            copies[j] = _ref_fusionner(copies[j], copies[i])
        for i in range(3):
            for j in range(3):
                copies[j] = _ref_fusionner(copies[j], copies[i])
        assert copies[0] == copies[1] == copies[2]
    for _nom, a, b in _cas_289():
        assert _ref_fusionner(a, b) == corrige.fusionner(a, b) == _fusion_par_copie(a, b)


# ============================================================================ les outils communs et l'histoire


def test_les_outils_communs():
    evenements = evenements_exemple()
    assert evenements == sorted(evenements, key=lambda e: e["t"]) and evenements_exemple() is not evenements
    assert {e["action"] for e in evenements} == {"modifier", "effacer"}
    assert heure_physique("crm", RECUL_NTP - 1) > heure_physique("crm", RECUL_NTP)
    assert heure_physique("appli", 5) == 5
    assert bilan(vue({}))["champs_faux"] == 1159 * 6
    assert vue({"C1": {"champs": {}, "efface": (1, 0, "crm")}}) == {}
    replique = Replique({}, profondeur=2)
    assert replique.empreinte("") == VIDE and set(replique.enfants("")) == {f"{c}" for c in "0123456789abcdef"}
    with pytest.raises(ValueError):
        replique.enfants("00")
    with pytest.raises(ValueError):
        replique.feuille("0")
    with pytest.raises(ValueError):
        replique.empreinte("g")
    with pytest.raises(TypeError):
        replique.empreinte(0)
    assert replique.appels == 6                         # les questions refusées comptent aussi
    assert HorlogePhysique("crm", lambda: 7).estampiller() == (7, 0, "crm")
    assert consentements([]) == {} and ecarts_consentement({}) == 300
    livraisons = livraisons_exemple()
    assert all(isinstance(m["numero"], int) and m["numero"] >= 1 for m in livraisons)


def test_l_histoire_tient():
    evenements = evenements_exemple()
    assert len(evenements) == 4751
    assert bilan(synchro_actuelle(evenements)) == {"champs_faux": 347, "ressuscites": 41, "sms_malgre_refus": 186}

    versions = Counter(corrige.comparer_versions(a, b) for _n, _c, a, b in versions_exemple())
    assert versions == {APRES: 1750, AVANT: 1703, CONCURRENTES: 495} and sum(versions.values()) == 3948

    physique = rejouer(evenements, HorlogePhysique, corrige.fusionner)
    assert bilan(vue(physique)) == {"champs_faux": 7, "ressuscites": 0, "sms_malgre_refus": 0}
    hybride = rejouer(evenements, corrige.HorlogeHybride, corrige.fusionner)
    assert bilan(vue(hybride)) == {"champs_faux": 0, "ressuscites": 0, "sms_malgre_refus": 0}
    # Les 7 champs perdus avec l'heure du serveur sont ceux modifiés deux fois autour du recul de l'horloge du CRM.
    autour = {(e["client"], e["champ"]) for e in evenements if abs(e["t"] - RECUL_NTP) < 120_000}
    perdus = {(c, ch) for c, fiche in vue(hybride).items() for ch, v in fiche.items() if vue(physique)[c][ch] != v}
    assert len(perdus) == 7 and perdus <= autour

    crm, facturation = parc_exemple()
    locale, distante = Replique(crm), Replique(facturation)
    ecarts = corrige.differences(locale, distante)
    assert Counter(ecarts.values()) == {"modifiee": 14, "locale": 13, "distante": 10}
    assert (distante.appels, distante.fiches_transferees) == (82, 467)

    livraisons = livraisons_exemple()
    assert (len(livraisons), RENVOIS, ecarts_consentement(consentements(livraisons))) == (6893, 32, 23)
    recepteur = corrige.Recepteur()
    appliques = [m for message in livraisons[:-RENVOIS] for m in recepteur.recevoir(message)]
    assert (len(appliques), recepteur.en_attente(), len(recepteur.manquants())) == (6329, 28, 4)
    appliques += [m for message in livraisons[-RENVOIS:] for m in recepteur.recevoir(message)]
    assert (len(appliques), recepteur.en_attente(), ecarts_consentement(consentements(appliques))) == (6361, 0, 0)
    assert len({(m["source"], m["numero"]) for m in livraisons}) == 6361          # chacun une fois, et une seule


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    serie = next(s for s in series() if s.code == "47")
    assert [r.statut for r in evaluer(serie)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(serie, corrige=True)] == [REUSSI] * 6


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_47 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(synchro)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_47 import *` (dans les exemples) doit donner les objets du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 45 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    a, b = _cas_289()[0][1], _cas_289()[0][2]
    vecteurs = ({"crm": 1}, {"appli": 2})
    fiche = {"ville": unicodedata.normalize("NFD", "Orléans"), "n": [1, {"b": 2, "a": 1}]}
    message = {"source": "t", "numero": 2}
    copies = copy.deepcopy((a, b, vecteurs, fiche, message))
    corrige.fusionner(a, b)
    corrige.comparer_versions(*vecteurs)
    corrige.empreinte(fiche)
    corrige.Recepteur().recevoir(message)
    assert (a, b, vecteurs, fiche, message) == copies


def test_les_erreurs_attendues_du_corrige():
    with pytest.raises(ValueError):
        corrige.comparer_versions({"crm": -1}, {})
    with pytest.raises(TypeError):
        corrige.empreinte({"x": {1, 2}})
    with pytest.raises(ValueError):
        corrige.differences(Replique({}, 1), Replique({}, 2))
    recepteur = corrige.Recepteur()
    recepteur.recevoir({"source": "t", "numero": 3, "v": 1})
    with pytest.raises(ValueError):
        recepteur.recevoir({"source": "t", "numero": 3, "v": 2})
    assert recepteur.en_attente() == 1                  # rien n'a changé
    horloge = corrige.HorlogeHybride("crm", lambda: 1000)
    with pytest.raises(ValueError):
        horloge.recevoir((1000 + DERIVE_MAX_MS + 1, 0, "x"))
    assert horloge.estampiller() == (1000, 0, "crm")    # rien n'a changé
    with pytest.raises(ValueError):
        corrige.fusionner({"C": {"champs": {"a": (1, (1, 0, "x"))}, "efface": None}},
                          {"C": {"champs": {"a": (2, (1, 0, "x"))}, "efface": None}})


def test_les_bugs_annonces_existent_tous():
    """Chaque erreur classique décrite par le correcteur a un test ci-dessus (et inversement)."""
    assert set(_BUGS_VERSIONS) == {"somme", "communes", "egalite", "un_sens", "inverse", "strict"}
    assert set(_BUGS_EMPREINTE) == {"sans_tri", "tri_surface", "sans_nfc", "nfd", "cles_brutes", "nfc_surface",
                                    "nfc_apres", "str", "separateurs", "ascii"}
    assert set(_BUGS_DIFFERENCES) == {"un_sens", "sans_absents", "inverse"}
    assert set(_BUGS_RECEPTEUR) == {"doublons", "arrivee", "global", "zero", "un_seul", "sans_vidage", "saute",
                                    "conflit_ignore", "conflit_remplace", "manquants_tous", "manquants_vides"}
    assert set(_BUGS_HORLOGE) == {"physique", "lamport", "sans_compteur", "suit_le_physique", "max_oublie",
                                  "ignore_recu", "compteur_zero", "sans_derive", "derive_apres", "derive_inclusive"}
    assert set(_BUGS_FUSIONNER) == {"fiche_entiere", "b_gagne", "sans_pierre_tombale", "sans_filtre", "filtre_local",
                                    "efface_min", "temps_seul", "egalite_silencieuse"}


def test_chaque_bug_change_au_moins_un_cas():
    cas_284 = _cas_284()[:15]                           # les cas écrits à la main
    for bug in _BUGS_VERSIONS:
        assert any(_ref_versions(a, b, bug) != _ref_versions(a, b) for a, b in cas_284), bug
    cas_285 = _cas_285()[:32]
    for bug in _BUGS_EMPREINTE:
        assert any(_ref_empreinte(f, bug) != _ref_empreinte(f) for f in cas_285), bug
    for bug in _BUGS_DIFFERENCES:
        assert any(_ref_differences(*_repliques(i, la, p), bug) != _ref_differences(*_repliques(i, la, p))
                   for _n, i, la, p in _cas_286()[:7]), bug
    for bug in _BUGS_RECEPTEUR:
        assert any(_jouer_287(_classe_recepteur(bug), ops) != _jouer_287(_RefRecepteur, ops)
                   for ops in _scenarios_287()[:10]), bug
    for bug in _BUGS_HORLOGE:
        assert any(_jouer_288(lambda r, p: _RefHorloge(r, p, bug=bug), ops) != _jouer_288(_RefHorloge, ops)
                   for _n, ops in _scenarios_288()[:13]), bug
    for bug in _BUGS_FUSIONNER:
        def rejeu(a, b, bug=bug):
            try:
                return _ref_fusionner(a, b, bug)
            except ValueError:
                return "ValueError"

        def reference(a, b):
            try:
                return _ref_fusionner(a, b)
            except ValueError:
                return "ValueError"

        cas = _cas_289()[:14] + [("conflit", {"C": {"champs": {"a": (1, (1, 0, "x"))}, "efface": None}},
                                  {"C": {"champs": {"a": (2, (1, 0, "x"))}, "efface": None}})]
        assert any(rejeu(a, b) != reference(a, b) for _n, a, b in cas), bug

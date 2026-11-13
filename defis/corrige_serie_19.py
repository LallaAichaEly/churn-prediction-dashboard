"""Série 19 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_19.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la situation à laquelle vous n'aviez pas pensé.

Fil rouge de la série : on fait évoluer un code qui tourne, par petits pas SÛRS. Trois réflexes :

1. savoir ce que le code fait AUJOURD'HUI avant d'y toucher (photo, lecture attentive, comparaison
   dans l'ombre), bizarreries comprises ;
2. ne jamais casser ceux qui dépendent de vous : l'ancienne façon d'appeler, l'ancien format de
   données, l'ancienne interface continuent de marcher pendant la transition ;
3. séparer les changements : un nettoyage ne change pas le comportement, un correctif de bogue va dans
   un changement à part. Chacun se relit, se teste et s'annule seul.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import copy
import functools
import json
import operator
import random
import re
import warnings
from datetime import date
from typing import Callable, Iterable

from defis.heritage import (  # noqa: F401
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


def _comportement(fonction: Callable, args: tuple, kwargs: dict | None = None) -> tuple[tuple, BaseException | None]:
    """Appelle la fonction et note ce qu'elle a fait : ("ok", valeur) ou ("erreur", type, message).

    Renvoie aussi l'exception elle-même (ou None) : le défi 121 doit pouvoir la relever.
    """
    try:
        valeur = fonction(*args, **(kwargs or {}))
    except Exception as exc:            # pas BaseException : Ctrl+C arrête le programme, il ne se « note » pas
        return ("erreur", type(exc).__name__, str(exc)), exc
    return ("ok", copy.deepcopy(valeur)), None   # copie : la fonction pourrait modifier sa valeur plus tard


def photographier(fonction: Callable, cas: Iterable[tuple]) -> list[tuple[tuple, tuple]]:
    """Défi 116 — la photo du comportement actuel (voir l'énoncé dans serie_19.py)."""
    photo = []
    for arguments in cas:                               # un seul parcours : `cas` peut être un générateur
        avant = copy.deepcopy(arguments)                # ce que l'appelant a donné, AVANT l'appel
        resultat, _ = _comportement(fonction, copy.deepcopy(arguments))   # la fonction abîme sa copie, pas `cas`
        photo.append((avant, resultat))
    return photo
    # Pour s'en servir dans un test (tests/test_score.py) : on enregistre la photo une fois, en JSON
    # ou avec pickle, puis on vérifie `photographier(score, CAS) == photo_enregistree` à chaque changement.
    # Les cas : de vrais clients anonymisés ET les cas limites (vide, négatif, texte, absent, énorme).


def remise(montant: float, taux: float | None = None, *, pourcentage: float | None = None) -> float:
    """Défi 117 — renommer un paramètre sans casser les appels existants."""
    if taux is not None and pourcentage is not None:     # `is not None` : 0 est une remise valide
        raise TypeError("remise() : donnez `pourcentage` (ou l'ancien `taux`), pas les deux")
    if taux is None and pourcentage is None:
        raise TypeError("remise() : il manque `pourcentage` (ou l'ancien `taux`)")
    if taux is not None:
        warnings.warn(f"remise(montant, taux) est dépréciée : écrivez remise(montant, pourcentage={taux * 100:g})",
                      DeprecationWarning, stacklevel=2)   # 2 : la ligne de l'APPELANT, celle qu'il faut corriger
        return remise_ancienne(montant, taux)
    if not 0 <= pourcentage <= 100:
        raise ValueError(f"pourcentage doit être entre 0 et 100 (reçu : {pourcentage})")
    return remise_ancienne(montant, pourcentage / 100)   # UNE formule : les deux façons ne peuvent pas diverger
    # Dans six mois : `grep -rn "remise(.*, 0\." .` ne trouve plus rien, les tests tournent avec
    # `-W error::DeprecationWarning`… et on retire `taux`. Notez la date de retrait dans le message ou le
    # CHANGELOG : un avertissement sans échéance est ignoré pour toujours.


# ============================================================================ Rendre testable


def decider_relances(lignes: list[dict[str, str]], aujourd_hui: date, budget: int = 50) -> list[tuple[str, str]]:
    """Défi 118 — le cœur pur du script de nuit : mêmes décisions, sans fichier, sans horloge, sans état."""
    candidats = []
    for ligne in lignes:
        score = float(ligne["score"].replace(",", "."))       # une NOUVELLE valeur : la ligne reste intacte
        candidats.append((-score, ligne["client_id"], ligne))
    candidats.sort(key=lambda c: (c[0], c[1]))                # même ordre que l'ancien : score décroissant, puis id

    relances = []
    for moins_score, client_id, ligne in candidats:
        if -moins_score < 0.6:                                # « < 0.6 : continue » → 0.6 pile est relancé
            continue
        if ligne["consentement"] != "oui":                    # « Oui » refusé, comme avant (ticket ouvert à part)
            continue
        if ligne["dernier_contact"]:
            if (aujourd_hui - date.fromisoformat(ligne["dernier_contact"])).days < 30:
                continue                                      # 30 jours pile : relancé ; date future : jamais
        if len(relances) >= budget:                           # le budget appartient à CET appel
            break
        relances.append((client_id, "ENGAGEMENT" if ligne["contrat"] == "Mensuel" else "FIDELITE"))
    return relances


def traiter_relances(chemin, operateur, aujourd_hui: date | None = None, budget: int = 50) -> list[tuple[str, str]]:
    """La nouvelle COQUILLE du script de nuit : lire, décider (le cœur), envoyer. Plus rien à tester ici."""
    import csv

    with open(chemin, encoding="utf-8", newline="") as fichier:
        lignes = list(csv.DictReader(fichier, delimiter=";"))
    telephones = {ligne["client_id"]: ligne["telephone"] for ligne in lignes}
    relances = decider_relances(lignes, aujourd_hui or date.today(), budget)
    for client_id, offre in relances:
        operateur.envoyer(telephones[client_id], offre)
    return relances


# ============================================================================ Changer de format

_DEFAUT_HEURE = 10


def lire_preferences(texte: str | None) -> dict:
    """Défi 119 — lire le format 1 et le format 2, rendre toujours un format 2 complet."""
    if texte is None or not texte.strip():
        # Un NOUVEAU dictionnaire à chaque appel (jamais une constante du module qu'un appelant modifierait).
        return {"version": 2, "canaux": [], "heure": _DEFAUT_HEURE, "langue": "fr"}
    donnees = json.loads(texte)                 # json.JSONDecodeError hérite de ValueError
    if not isinstance(donnees, dict):
        raise ValueError(f"préférences : un objet JSON était attendu, pas {type(donnees).__name__}")

    version = donnees.get("version", 1)          # pas de clé : les lignes d'avant 2024, format 1
    if type(version) is not int or version not in (1, 2):   # True == 1 et "2" != 2 : on vérifie le TYPE exact
        raise ValueError(f"préférences : version {version!r} inconnue (ce code lit les versions 1 et 2)")

    if version == 1:
        canaux = []
        for canal in ("email", "sms"):
            valeur = donnees.get(canal, False)
            if not isinstance(valeur, bool):     # « "false" » serait VRAI : un consentement mal lu, c'est grave
                raise ValueError(f"préférences (format 1) : {canal} doit valoir true ou false, pas {valeur!r}")
            if valeur:
                canaux.append(canal)
    else:
        canaux = donnees.get("canaux", [])
        if not isinstance(canaux, list):
            raise ValueError(f"préférences : canaux doit être une liste, pas {canaux!r}")
        for canal in canaux:
            if canal not in CANAUX:
                raise ValueError(f"préférences : canal {canal!r} inconnu (connus : {', '.join(CANAUX)})")

    heure = donnees.get("heure", _DEFAUT_HEURE)
    if type(heure) is not int or not 0 <= heure <= 23:      # type exact : true n'est pas une heure
        raise ValueError(f"préférences : heure doit être un entier de 0 à 23, pas {heure!r}")

    # On ne recopie QUE les clés connues : une clé inconnue est tolérée, pas propagée.
    return {"version": 2, "canaux": sorted(set(canaux)), "heure": heure, "langue": donnees.get("langue", "fr")}
    # Et l'écriture ? On écrit toujours le format le plus récent : json.dumps(lire_preferences(ancien)).
    # Les lignes au format 1 disparaissent ainsi peu à peu, à chaque mise à jour, sans grande migration.


# ============================================================================ Changer de fournisseur


def _numero_international(numero: str) -> str | None:
    brut = re.sub(r"[ .\-]", "", numero)
    for motif, prefixe in ((r"0([1-9]\d{8})", "+33"), (r"\+33([1-9]\d{8})", "+33"), (r"0033([1-9]\d{8})", "+33")):
        trouve = re.fullmatch(motif, brut)
        if trouve:
            return prefixe + trouve.group(1)
    return None


class AdaptateurSMS:
    """Défi 120 — l'ancienne interface (`envoyer_sms -> bool`), le nouvel opérateur derrière."""

    # Les codes qui veulent dire « ce SMS n'est pas parti, passez au suivant », comme le False de l'ancien.
    ECHECS_D_ENVOI = frozenset({"NUMERO_INVALIDE", "INDISPONIBLE"})

    def __init__(self, client: ClientMessagerie, expediteur: str = "RETENTION") -> None:
        self.client = client
        self.expediteur = expediteur

    def envoyer_sms(self, numero: str, texte: str) -> bool:
        if not texte or len(texte) > 160:       # AVANT tout envoi : le nouvel opérateur facturerait 2 SMS
            raise ValueError(f"le texte doit faire de 1 à 160 caractères (reçu : {len(texte)})")
        destinataire = _numero_international(numero)
        if destinataire is None:
            return False
        try:
            reponse = self.client.messages.create(destinataire=destinataire, contenu=texte,
                                                  expediteur=self.expediteur)
        except ErreurMessagerie as exc:
            if exc.code in self.ECHECS_D_ENVOI:
                return False
            if exc.code == "CLE_INVALIDE":
                raise PermissionError("clé d'API refusée par l'opérateur SMS") from exc   # la cause reste visible
            raise                               # QUOTA_DEPASSE, un code inconnu… : on ne cache pas ce qu'on ignore
        return reponse.statut == "accepte"


# ============================================================================ Basculer sans risque


def en_ombre(ancienne: Callable, nouvelle: Callable, journal: Callable[[dict], object], *,
             proportion: float = 1.0, tirage: Callable[[], float] = random.random,
             egal: Callable[[object, object], bool] = operator.eq) -> Callable:
    """Défi 121 — l'ancienne répond, la nouvelle tourne dans l'ombre, les différences sont notées."""
    if not 0 <= proportion <= 1:
        raise ValueError(f"proportion doit être entre 0 et 1 (reçu : {proportion})")
    stats = {"appels": 0, "comparaisons": 0, "differences": 0}   # créé à CHAQUE appel d'en_ombre : un par fonction

    def different(a: tuple, b: tuple) -> bool:
        if a[0] == "ok" and b[0] == "ok":
            return not egal(a[1], b[1])
        if a[0] == "erreur" and b[0] == "erreur":
            return a[1] != b[1]                 # même type d'erreur : même comportement (le message peut changer)
        return True

    @functools.wraps(ancienne)
    def fonction(*args, **kwargs):
        stats["appels"] += 1
        comparer = tirage() < proportion
        if comparer:
            avant = copy.deepcopy((args, kwargs))            # AVANT l'ancienne, qui peut abîmer ses arguments
        ancien, exc_ancienne = _comportement(ancienne, args, kwargs)
        if comparer:
            nouveau, _ = _comportement(nouvelle, *copy.deepcopy(avant))   # sa copie à elle
            stats["comparaisons"] += 1
            if different(ancien, nouveau):
                stats["differences"] += 1
                journal({"arguments": avant, "ancienne": ancien, "nouvelle": nouveau})
        if exc_ancienne is not None:
            raise exc_ancienne                  # la MÊME exception, pour l'appelant rien ne change
        return ancien[1]

    fonction.stats = stats
    return fonction
    # Note : _comportement renvoie une COPIE de la valeur de l'ancienne. Si l'appelant comptait recevoir
    # l'objet lui-même (pour le modifier, par exemple), renvoyez plutôt la valeur d'origine et gardez la
    # copie pour la comparaison. Pour un score (un float), ça ne change rien.


# ============================================================================ Tests (comme dans tests/)


def test_la_photo_est_stable_et_ne_touche_pas_aux_cas():
    cas = [(dict(c),) for c in CLIENTS_EXEMPLE]
    photo = photographier(ancien_score, cas)
    assert photo == photographier(ancien_score, cas)          # deux photos du même code sont égales
    assert cas[0][0]["charges"] == "89,90"                     # la fonction n'a abîmé qu'une copie
    assert photo[3][1] == ("erreur", "KeyError", "'charges'")
    assert photographier(nouveau_score, cas) != photo         # la réécriture change le comportement : on le SAIT


def test_l_ancien_appel_de_remise_marche_et_previent():
    with warnings.catch_warnings(record=True) as avertissements:
        warnings.simplefilter("always")
        assert remise(80, 0.25) == remise(80, pourcentage=25) == 60.0
        assert remise(80, taux=0) == 80.0                      # 0 n'est pas « absent »
    assert [a.category for a in avertissements] == [DeprecationWarning, DeprecationWarning]
    assert avertissements[0].filename == __file__              # stacklevel=2 : la ligne de l'appelant
    for appel, erreur in ((lambda: remise(100, 20), ValueError), (lambda: remise(100), TypeError),
                          (lambda: remise(100, 0.2, pourcentage=20), TypeError)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                appel()
        except erreur:
            pass
        else:
            raise AssertionError(f"{erreur.__name__} attendue")


def test_la_decision_ne_depend_que_de_ses_entrees():
    lignes = copy.deepcopy(LIGNES_EXEMPLE)
    premiere = decider_relances(lignes, JOUR_EXEMPLE)
    assert premiere == decider_relances(lignes, JOUR_EXEMPLE)   # pas de compteur global
    assert lignes == LIGNES_EXEMPLE                             # lignes intactes
    assert ("C03", "ENGAGEMENT") in premiere                    # score 0,6 pile, contact il y a 30 jours pile
    assert all(client_id != "C07" for client_id, _ in premiere)  # « Oui » : comme l'ancien script


def test_la_nouvelle_coquille_envoie_ce_que_le_coeur_decide():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as dossier:
        chemin = ecrire_lignes(LIGNES_EXEMPLE, Path(dossier) / "relances.csv")
        operateur = OperateurFactice()
        relances = traiter_relances(chemin, operateur, JOUR_EXEMPLE)
    assert operateur.envois == [("0622222222", "FIDELITE"), ("0611111111", "ENGAGEMENT"),
                                ("0688888888", "FIDELITE"), ("0633333333", "ENGAGEMENT")]
    assert relances == decider_relances(LIGNES_EXEMPLE, JOUR_EXEMPLE)


def test_les_deux_formats_de_preferences_se_relisent_a_l_identique():
    for texte in (PREFERENCES_V1_EXEMPLE, PREFERENCES_V2_EXEMPLE, None, '{"version": 1}', '{"canaux": []}'):
        prefs = lire_preferences(texte)
        assert lire_preferences(json.dumps(prefs)) == prefs
    defaut = lire_preferences(None)
    defaut["canaux"].append("sms")
    assert lire_preferences("") == {"version": 2, "canaux": [], "heure": 10, "langue": "fr"}
    for refuse in ('{"version": 3}', '{"version": true}', '{"version": 2, "canaux": ["pigeon"]}', "[1, 2]",
                   '{"heure": true}', '{"heure": 24}', "pas du json"):
        try:
            lire_preferences(refuse)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{refuse} aurait dû être refusé")


def test_l_adaptateur_respecte_le_contrat_de_l_ancienne_passerelle():
    ancienne = PasserelleAncienne(stop=("0699999999",))
    client = ClientMessagerie("cle_test", refuses=("+33699999999",))
    nouvelle = AdaptateurSMS(client)
    for numero in ("06 12 34 56 78", "+33 6 12 34 56 78", "0033612345678", "06.99.99.99.99", "06 12", "abc"):
        assert nouvelle.envoyer_sms(numero, "Bonjour") == ancienne.envoyer_sms(numero, "Bonjour"), numero
    for client, attendue in ((ClientMessagerie("mauvaise_cle"), PermissionError),
                             (ClientMessagerie("cle_test", panne="QUOTA_DEPASSE"), ErreurMessagerie)):
        try:
            AdaptateurSMS(client).envoyer_sms("0612345678", "Bonjour")
        except attendue:
            pass
        else:
            raise AssertionError(f"{attendue.__name__} attendue")
    assert AdaptateurSMS(ClientMessagerie("cle_test", panne="INDISPONIBLE")).envoyer_sms("0612345678", "Bonjour") is False


def test_l_ombre_ne_change_rien_pour_l_utilisateur_et_note_les_differences():
    differences = []
    score = en_ombre(ancien_score, nouveau_score, differences.append)
    clients = [dict(c) for c in CLIENTS_EXEMPLE] + [{"client_id": "C006", "charges": 50}]
    for client in clients:
        try:
            attendu = ("ok", ancien_score(copy.deepcopy(client)))
        except Exception as exc:
            attendu = ("erreur", type(exc).__name__)
        try:
            obtenu = ("ok", score(client))
        except Exception as exc:
            obtenu = ("erreur", type(exc).__name__)
        assert obtenu == attendu                               # l'utilisateur voit l'ancien comportement
    assert score.stats == {"appels": 6, "comparaisons": 6, "differences": 2}   # C004 (KeyError/ValueError) et C006
    assert [d["arguments"][0][0]["client_id"] for d in differences] == ["C004", "C006"]
    assert score.__name__ == "ancien_score"

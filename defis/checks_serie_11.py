"""Vérifications automatiques de la série 11 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Chaque défi de cette série vous demande d'écrire un TEST. Le correcteur le lance :
- sur la version juste de la fonction, et parfois sur une seconde version juste écrite autrement :
  votre test doit passer sur toutes (sinon il est faux, ou « fragile ») ;
- sur des versions boguées (des « mutants ») : votre test doit échouer sur chacune.

Les fonctions ci-dessous sont des COPIES des fonctions fournies dans `defis/serie_11.py`. Si vous
modifiez par erreur la fonction fournie dans votre fichier, la correction ne change pas.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from .verif import Defi, Serie


def _lancer(test: Callable, *args) -> BaseException | None:
    """Lance le test de l'élève. Renvoie l'exception qui l'a fait échouer, ou None s'il passe.

    On attrape BaseException car l'échec de `pytest.raises` (« DID NOT RAISE ») n'est PAS une
    Exception : pytest le fait dériver directement de BaseException.
    """
    try:
        test(*args)
    except (NotImplementedError, KeyboardInterrupt):
        raise
    except BaseException as exc:  # noqa: BLE001 — AssertionError, plantage, échec de pytest.raises…
        return exc
    return None


def _decrire(exc: BaseException) -> str:
    texte = " ".join(str(exc).split())
    if type(exc).__name__ == "Failed":  # pytest.raises : l'erreur attendue n'est pas arrivée
        return f"pytest.raises a échoué : {texte}"
    if isinstance(exc, AssertionError):
        return f"assert échoué : {texte}" if texte else "un assert sans message a échoué"
    return f"{type(exc).__name__}: {texte}"


def _verifier_mutants(test: Callable, mutants: list, tout_passe: str, args: Callable[[], tuple] = tuple) -> None:
    """Le test doit échouer sur chaque mutant ; sinon, on explique le premier bug qui passe."""
    survivants = []
    for mutant, defaut, conseil in mutants:
        if _lancer(test, mutant, *args()) is None:
            survivants.append((defaut, conseil))
    if not survivants:
        return
    if len(survivants) == len(mutants):
        raise AssertionError(f"Votre test passe sur TOUTES les versions boguées : il ne vérifie presque rien. {tout_passe}")
    defaut, conseil = survivants[0]
    raise AssertionError(
        f"{len(mutants) - len(survivants)}/{len(mutants)} bugs détectés. Votre test laisse passer une version "
        f"qui {defaut}. {conseil}"
    )


def _verifier_test(test: Callable, juste: Callable, autre_juste: Callable | None, mutants: list, *,
                   si_echec_juste: str, si_fragile: str, tout_passe: str) -> None:
    exc = _lancer(test, juste)
    if exc is not None:
        if type(exc).__name__ == "Failed":  # pytest.raises n'a rien vu venir
            si_echec_juste = ("Votre `pytest.raises` attend une erreur que la fonction correcte ne lève pas : l'entrée "
                              "que vous lui donnez est-elle vraiment invalide d'après la docstring ?")
        raise AssertionError(
            f"Votre test ÉCHOUE sur la version CORRECTE de la fonction ({_decrire(exc)}). Un test qui crie au "
            f"loup est pire que pas de test. {si_echec_juste}"
        )
    if autre_juste is not None:
        exc = _lancer(test, autre_juste)
        if exc is not None:
            raise AssertionError(
                f"Votre test passe sur la fonction fournie, mais ÉCHOUE sur une autre version JUSTE "
                f"({_decrire(exc)}). C'est un test fragile : il bloquerait un refactoring légitime. {si_fragile}"
            )
    _verifier_mutants(test, mutants, tout_passe)


# ============================================================================ Défi 68 : frontières


def _segment(mois: int) -> str:
    if mois < 6:
        return "nouveau"
    if mois < 24:
        return "installé"
    return "fidèle"


def _segment_6_inclus(mois: int) -> str:
    if mois <= 6:
        return "nouveau"
    if mois < 24:
        return "installé"
    return "fidèle"


def _segment_frontiere_5(mois: int) -> str:
    if mois < 5:
        return "nouveau"
    if mois < 24:
        return "installé"
    return "fidèle"


def _segment_24_inclus(mois: int) -> str:
    if mois < 6:
        return "nouveau"
    if mois <= 24:
        return "installé"
    return "fidèle"


def _segment_frontiere_23(mois: int) -> str:
    if mois < 6:
        return "nouveau"
    if mois < 23:
        return "installé"
    return "fidèle"


def _segment_zero_oublie(mois: int) -> str:
    if mois and mois < 6:  # 0 est « faux » en Python : un client de 0 mois saute la première branche
        return "nouveau"
    if mois < 24:
        return "installé"
    return "fidèle"


_MUTANTS_SEGMENT = [
    (_segment_6_inclus, "range 6 mois dans « nouveau » (`mois <= 6` au lieu de `mois < 6`)",
     "Testez la PREMIÈRE valeur de la zone suivante : 6 mois."),
    (_segment_frontiere_5, "range 5 mois dans « installé » (frontière décalée à 5)",
     "Testez la DERNIÈRE valeur de la zone « nouveau » : 5 mois."),
    (_segment_24_inclus, "range 24 mois dans « installé » (`mois <= 24` au lieu de `mois < 24`)",
     "Il y a DEUX frontières : testez aussi chaque côté de la seconde, à commencer par 24."),
    (_segment_frontiere_23, "range 23 mois dans « fidèle » (frontière décalée à 23)",
     "Testez la dernière valeur de la zone « installé » : 23 mois."),
    (_segment_zero_oublie, "range un client de 0 mois dans « installé » (`if mois and mois < 6` : 0 est « faux » en Python)",
     "Testez la plus petite valeur possible : 0 mois, un client inscrit ce mois-ci."),
]

MAX_CAS = 6


def verifier_cas_limites_anciennete(f) -> None:
    cas = f()
    assert cas is not None, "Votre fonction ne renvoie rien : avez-vous oublié le `return` ?"
    assert isinstance(cas, (list, tuple)), f"Attendu une liste de tuples (mois, attendu), obtenu {type(cas).__name__}."
    assert len(cas) > 0, "Votre liste est vide : proposez au moins un cas."
    for element in cas:
        assert isinstance(element, (tuple, list)) and len(element) == 2, (
            f"Chaque cas est un tuple (mois, segment attendu), comme (3, 'nouveau'). Obtenu : {element!r}."
        )
        mois, attendu = element
        assert isinstance(mois, int) and not isinstance(mois, bool), f"Dans {element!r}, le nombre de mois doit être un entier."
        assert mois >= 0, f"Dans {element!r} : une ancienneté négative n'existe pas, restez à 0 mois ou plus."
        assert _segment(mois) == attendu, (
            f"Le cas {element!r} est faux : segment_anciennete({mois}) vaut {_segment(mois)!r}. Relisez la "
            "docstring : « 0 à 5 mois », c'est 5 compris ; « 6 à 23 mois », 6 et 23 compris."
        )
    assert len(cas) <= MAX_CAS, (
        f"Vous proposez {len(cas)} cas : au plus {MAX_CAS}. Essayer toutes les valeurs ne passe pas à l'échelle "
        "(et serait impossible pour un prix ou une date). Gardez seulement les valeurs de chaque côté des frontières."
    )

    survivants = [(defaut, conseil) for mutant, defaut, conseil in _MUTANTS_SEGMENT
                  if all(mutant(mois) == attendu for mois, attendu in cas)]
    if survivants:
        defaut, conseil = survivants[0]
        raise AssertionError(
            f"{len(_MUTANTS_SEGMENT) - len(survivants)}/{len(_MUTANTS_SEGMENT)} bugs détectés. Une version qui "
            f"{defaut} passe tous vos cas. {conseil}"
        )


# ============================================================================ Défi 69 : flottants


def _taux(partis: int, total: int) -> float:
    if total == 0:
        return 0.0
    return partis / total


def _taux_par_les_restants(partis: int, total: int) -> float:
    """Juste, mais calculée autrement : le dernier chiffre peut différer (1 sur 3 → 0.33333333333333337)."""
    if total == 0:
        return 0.0
    restants = total - partis
    return 1 - restants / total


def _taux_pourcentage(partis: int, total: int) -> float:
    if total == 0:
        return 0.0
    return 100 * partis / total


def _taux_arrondi(partis: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(partis / total, 2)


def _taux_divise_par_restes(partis: int, total: int) -> float:
    if total == 0:
        return 0.0
    return partis / (total - partis)


def _taux_plante_si_vide(partis: int, total: int) -> float:
    return partis / total


def _taux_retention(partis: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (total - partis) / total


_MUTANTS_TAUX = [
    (_taux_pourcentage, "renvoie un POURCENTAGE (25.0) au lieu d'une proportion (0.25)",
     "Comparez un cas normal à la proportion attendue : taux_churn(1, 4) == pytest.approx(0.25)."),
    (_taux_arrondi, "arrondit à 2 décimales (0.33 au lieu de 0.333…)",
     "Choisissez un résultat qui a beaucoup de décimales (1 sur 3) et gardez la tolérance par défaut de "
     "pytest.approx : avec une tolérance large (abs=0.01) ou en arrondissant vous-même, un arrondi passe inaperçu."),
    (_taux_divise_par_restes, "divise par les clients RESTÉS au lieu du total (1 parti, 3 restés : 0.33 au lieu de 0.25)",
     "Un cas normal dont vous connaissez le résultat suffit, par exemple 1 sur 4."),
    (_taux_plante_si_vide, "plante (ZeroDivisionError) quand il n'y a aucun client",
     "Testez le cas limite : taux_churn(0, 0) doit valoir 0.0."),
    (_taux_retention, "renvoie le taux de RÉTENTION (clients restés) au lieu du taux de départ",
     "Évitez les exemples symétriques comme 1 sur 2 (0.5 dans les deux cas) : prenez 1 sur 4."),
]


def verifier_tester_taux_churn(test) -> None:
    _verifier_test(
        test, _taux, _taux_par_les_restants, _MUTANTS_TAUX,
        si_echec_juste=(
            "Comparez-vous avec `==` ? Un flottant calculé n'est presque jamais égal au chiffre près à celui que vous "
            "avez écrit : taux_churn(1, 3) vaut 0.3333333333333333, pas 0.333. Utilisez pytest.approx(1 / 3). Sinon, "
            "vérifiez vos valeurs attendues : une proportion entre 0 et 1, pas un pourcentage."
        ),
        si_fragile=(
            "Cette version calcule 1 - restants / total : pour 1 sur 3, elle donne 0.33333333333333337 au lieu de "
            "0.3333333333333333. Votre test compare au dernier chiffre près (avec ==) : utilisez "
            "`taux_churn(1, 3) == pytest.approx(1 / 3)` ou math.isclose."
        ),
        tout_passe="Comparez le résultat à la valeur attendue : `assert taux_churn(1, 4) == pytest.approx(0.25)`.",
    )


# ============================================================================ Défi 70 : erreurs

_CONTRATS = {"mensuel": "Mensuel", "1 an": "1 an", "2 ans": "2 ans"}


def _contrat(texte: str) -> str:
    cle = texte.strip().lower()
    if cle not in _CONTRATS:
        raise ValueError(f"Contrat inconnu : {texte!r} (attendu : Mensuel, 1 an ou 2 ans)")
    return _CONTRATS[cle]


def _contrat_autre_message(texte: str) -> str:
    """Juste aussi : le message cite toujours la saisie, mais il est formulé autrement."""
    cle = texte.strip().lower()
    if cle in _CONTRATS:
        return _CONTRATS[cle]
    raise ValueError(f"{texte!r} n'est pas un type de contrat : choisissez Mensuel, 1 an ou 2 ans")


def _contrat_renvoie_none(texte: str):
    return _CONTRATS.get(texte.strip().lower())


def _contrat_keyerror(texte: str) -> str:
    return _CONTRATS[texte.strip().lower()]


def _contrat_message_vague(texte: str) -> str:
    cle = texte.strip().lower()
    if cle not in _CONTRATS:
        raise ValueError("Contrat inconnu")
    return _CONTRATS[cle]


def _contrat_refuse_2_ans(texte: str) -> str:
    contrats = {"mensuel": "Mensuel", "1 an": "1 an", "2 an": "2 ans"}  # faute de frappe dans la clé
    cle = texte.strip().lower()
    if cle not in contrats:
        raise ValueError(f"Contrat inconnu : {texte!r} (attendu : Mensuel, 1 an ou 2 ans)")
    return contrats[cle]


def _contrat_sans_strip(texte: str) -> str:
    cle = texte.lower()
    if cle not in _CONTRATS:
        raise ValueError(f"Contrat inconnu : {texte!r} (attendu : Mensuel, 1 an ou 2 ans)")
    return _CONTRATS[cle]


_MUTANTS_CONTRAT = [
    (_contrat_renvoie_none, "renvoie None au lieu de lever une erreur pour « Mensual »",
     "Vérifiez qu'une saisie inconnue LÈVE une erreur : `with pytest.raises(ValueError): lire_contrat('Mensual')`. "
     "Un try/except seul ne détecte pas ce bug : si l'erreur n'arrive pas, il ne se passe rien."),
    (_contrat_keyerror, "lève une KeyError au lieu d'une ValueError",
     "pytest.raises(Exception) accepte n'importe quelle erreur, même celle d'un bug : demandez précisément ValueError."),
    (_contrat_message_vague, "lève une ValueError dont le message ne dit pas ce qui a été saisi (« Contrat inconnu »)",
     "La docstring promet que le message CITE la saisie : `pytest.raises(ValueError, match='Mensual')`."),
    (_contrat_refuse_2_ans, "refuse « 2 ans », pourtant valide (faute de frappe dans son dictionnaire)",
     "Testez CHAQUE valeur valide, pas seulement une : 'mensuel', '1 an' et '2 ans'."),
    (_contrat_sans_strip, "refuse « mensuel » quand une espace traîne autour (copier-coller depuis Excel)",
     "Testez une saisie valide mais mal tapée : lire_contrat('  MENSUEL ') doit donner 'Mensuel'."),
]


def verifier_tester_lire_contrat(test) -> None:
    _verifier_test(
        test, _contrat, _contrat_autre_message, _MUTANTS_CONTRAT,
        si_echec_juste=(
            "Vérifiez vos valeurs attendues (lire_contrat('mensuel') vaut 'Mensuel', avec une majuscule) et le type "
            "d'erreur attendu (ValueError). Pour une erreur attendue, utilisez `with pytest.raises(ValueError):` : "
            "sans lui, l'erreur levée fait échouer votre propre test."
        ),
        si_fragile=(
            "Son message est formulé autrement : « 'Mensual' n'est pas un type de contrat : choisissez… ». La docstring "
            "promet seulement que le message CITE la saisie : `match='Mensual'`, pas la phrase entière."
        ),
        tout_passe="Commencez par `assert lire_contrat('mensuel') == 'Mensuel'`, puis testez le chemin d'erreur avec pytest.raises.",
    )


# ============================================================================ Défi 71 : faux objets


def _relancer(clients, envoyer, seuil=0.5) -> int:
    deja, parties = set(), 0
    for client in clients:
        customer_id = client["customer_id"]
        if client["proba"] < seuil or customer_id in deja:
            continue
        deja.add(customer_id)
        try:
            envoyer(customer_id, f"Bonjour, profitez de -20 % sur votre prochain mois ({customer_id}).")
        except ConnectionError:
            continue
        parties += 1
    return parties


def _relancer_autre_message(clients, envoyer, seuil=0.5) -> int:
    """Juste aussi : le marketing a changé le texte (et la boucle est écrite autrement)."""
    a_relancer = []
    for client in clients:
        if client["proba"] >= seuil and client["customer_id"] not in a_relancer:
            a_relancer.append(client["customer_id"])
    parties = 0
    for customer_id in a_relancer:
        try:
            envoyer(customer_id, "Offre fidélité : votre prochain mois à -20 %, rien à faire pour en profiter !")
            parties += 1
        except ConnectionError:
            pass
    return parties


def _relancer_tout_le_monde(clients, envoyer, seuil=0.5) -> int:
    deja, parties = set(), 0
    for client in clients:
        customer_id = client["customer_id"]
        if customer_id in deja:
            continue
        deja.add(customer_id)
        try:
            envoyer(customer_id, "Bonjour, profitez de -20 % sur votre prochain mois.")
        except ConnectionError:
            continue
        parties += 1
    return parties


def _relancer_seuil_exclu(clients, envoyer, seuil=0.5) -> int:
    deja, parties = set(), 0
    for client in clients:
        customer_id = client["customer_id"]
        if client["proba"] <= seuil or customer_id in deja:
            continue
        deja.add(customer_id)
        try:
            envoyer(customer_id, "Bonjour, profitez de -20 % sur votre prochain mois.")
        except ConnectionError:
            continue
        parties += 1
    return parties


def _relancer_doublons(clients, envoyer, seuil=0.5) -> int:
    parties = 0
    for client in clients:
        if client["proba"] < seuil:
            continue
        try:
            envoyer(client["customer_id"], "Bonjour, profitez de -20 % sur votre prochain mois.")
        except ConnectionError:
            continue
        parties += 1
    return parties


def _relancer_abandonne(clients, envoyer, seuil=0.5) -> int:
    deja, parties = set(), 0
    for client in clients:
        customer_id = client["customer_id"]
        if client["proba"] < seuil or customer_id in deja:
            continue
        deja.add(customer_id)
        try:
            envoyer(customer_id, "Bonjour, profitez de -20 % sur votre prochain mois.")
        except ConnectionError:
            break  # abandonne tous les clients suivants
        parties += 1
    return parties


def _relancer_compte_les_echecs(clients, envoyer, seuil=0.5) -> int:
    deja, parties = set(), 0
    for client in clients:
        customer_id = client["customer_id"]
        if client["proba"] < seuil or customer_id in deja:
            continue
        deja.add(customer_id)
        parties += 1  # compté AVANT de savoir si l'envoi a réussi
        try:
            envoyer(customer_id, "Bonjour, profitez de -20 % sur votre prochain mois.")
        except ConnectionError:
            continue
    return parties


def _relancer_seuil_fige(clients, envoyer, seuil=0.5) -> int:
    return _relancer(clients, envoyer, 0.5)  # le paramètre est accepté… puis ignoré


_MUTANTS_RELANCE = [
    (_relancer_tout_le_monde, "envoie une offre à TOUS les clients, même à ceux qui ne risquent rien",
     "Mettez dans la liste un client sous le seuil et vérifiez qu'il n'a rien reçu (comparez la liste des envois)."),
    (_relancer_seuil_exclu, "ne relance pas un client dont la proba vaut EXACTEMENT le seuil (> au lieu de >=)",
     "Ajoutez un client pile au seuil : {'customer_id': 'C9', 'proba': 0.5}."),
    (_relancer_doublons, "envoie deux offres à un client présent deux fois dans la liste",
     "Mettez un même client deux fois dans la liste et vérifiez qu'il n'apparaît qu'une fois dans vos envois."),
    (_relancer_abandonne, "abandonne tous les clients suivants à la première panne",
     "Simulez une panne pour un client AU MILIEU de la liste (un faux `envoyer` qui lève ConnectionError pour lui) "
     "et vérifiez que les clients d'après reçoivent quand même leur offre."),
    (_relancer_compte_les_echecs, "compte comme « parties » les offres dont l'envoi a échoué",
     "Avec une panne simulée, vérifiez aussi le NOMBRE renvoyé : l'offre en échec ne compte pas."),
    (_relancer_seuil_fige, "ignore le paramètre `seuil` (utilise toujours 0.5)",
     "Appelez aussi la fonction avec un autre seuil, par exemple seuil=0.8 : un paramètre jamais testé est un bug en attente."),
]


def verifier_tester_relancer_clients(test) -> None:
    _verifier_test(
        test, _relancer, _relancer_autre_message, _MUTANTS_RELANCE,
        si_echec_juste=(
            "Relisez les règles : relance si proba >= seuil (pile au seuil compris), une seule fois par client, et une "
            "panne (ConnectionError) ne compte pas mais n'arrête pas la boucle. Si vous réutilisez la même liste "
            "d'envois pour deux appels, videz-la entre les deux (envois.clear())."
        ),
        si_fragile=(
            "Cette version envoie un autre texte : « Offre fidélité : votre prochain mois à -20 %… ». La docstring dit "
            "que le texte est libre : vérifiez À QUI on écrit et combien de fois, pas la formulation du message."
        ),
        tout_passe="Passez un faux `envoyer` qui note les customer_id reçus, puis comparez cette liste à celle attendue.",
    )


# ============================================================================ Défi 72 : fichiers

_NOM_RAPPORT = "clients_a_risque.csv"


def _exporter(scores: pd.DataFrame, dossier: Path, seuil: float = 0.5) -> Path:
    chemin = Path(dossier) / _NOM_RAPPORT
    if chemin.exists():
        raise FileExistsError(f"Le rapport existe déjà, il n'a pas été écrasé : {chemin}")
    a_risque = scores.loc[scores["proba"] >= seuil, ["customer_id", "proba"]]
    a_risque.sort_values("proba", ascending=False).to_csv(chemin, sep=";", index=False)
    return chemin


def _exporter_dossier_courant(scores: pd.DataFrame, dossier: Path, seuil: float = 0.5) -> Path:
    return _exporter(scores, Path("."), seuil)  # oublie `dossier` : écrit dans le dossier courant


def _exporter_avec_index(scores: pd.DataFrame, dossier: Path, seuil: float = 0.5) -> Path:
    chemin = Path(dossier) / _NOM_RAPPORT
    if chemin.exists():
        raise FileExistsError(chemin)
    a_risque = scores.loc[scores["proba"] >= seuil, ["customer_id", "proba"]]
    a_risque.sort_values("proba", ascending=False).to_csv(chemin, sep=";")
    return chemin


def _exporter_virgules(scores: pd.DataFrame, dossier: Path, seuil: float = 0.5) -> Path:
    chemin = Path(dossier) / _NOM_RAPPORT
    if chemin.exists():
        raise FileExistsError(chemin)
    a_risque = scores.loc[scores["proba"] >= seuil, ["customer_id", "proba"]]
    a_risque.sort_values("proba", ascending=False).to_csv(chemin, index=False)
    return chemin


def _exporter_non_trie(scores: pd.DataFrame, dossier: Path, seuil: float = 0.5) -> Path:
    chemin = Path(dossier) / _NOM_RAPPORT
    if chemin.exists():
        raise FileExistsError(chemin)
    scores.loc[scores["proba"] >= seuil, ["customer_id", "proba"]].to_csv(chemin, sep=";", index=False)
    return chemin


def _exporter_ecrase(scores: pd.DataFrame, dossier: Path, seuil: float = 0.5) -> Path:
    chemin = Path(dossier) / _NOM_RAPPORT
    a_risque = scores.loc[scores["proba"] >= seuil, ["customer_id", "proba"]]
    a_risque.sort_values("proba", ascending=False).to_csv(chemin, sep=";", index=False)
    return chemin


_MUTANTS_EXPORT = [
    (_exporter_dossier_courant, "écrit le fichier dans le dossier COURANT au lieu de `dossier`",
     "Relisez le fichier à l'endroit promis, `dossier / 'clients_a_risque.csv'`, et pas seulement via le chemin "
     "renvoyé (ou vérifiez que ce chemin est bien dans `dossier`)."),
    (_exporter_avec_index, "écrit aussi l'index de pandas (une colonne « Unnamed: 0 » apparaît dans Excel)",
     "Vérifiez la liste exacte des colonnes : `list(rapport.columns) == ['customer_id', 'proba']`."),
    (_exporter_virgules, "sépare les colonnes par des virgules : Excel en français met tout dans une seule colonne",
     "Relisez avec `sep=';'` et vérifiez la liste exacte des colonnes."),
    (_exporter_non_trie, "ne trie pas les clients du plus risqué au moins risqué",
     "Donnez des scores dans le désordre (0.6, 0.9, 0.7) et vérifiez l'ordre des customer_id dans le fichier."),
    (_exporter_ecrase, "écrase sans prévenir un rapport déjà présent",
     "Appelez la fonction deux fois avec le même dossier : la seconde fois doit lever FileExistsError "
     "(`with pytest.raises(FileExistsError): …`)."),
]


def _dans_un_bac_a_sable(test: Callable, exporter: Callable) -> tuple[BaseException | None, list[str]]:
    """Lance le test avec un `dossier` vide, depuis un dossier courant vide lui aussi.

    Renvoie l'échec éventuel et les fichiers laissés dans le dossier courant : votre dépôt n'est
    jamais touché, même si un test (ou un mutant) écrit au mauvais endroit.
    """
    ici = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="defi72_courant_") as courant, \
            tempfile.TemporaryDirectory(prefix="defi72_dossier_") as dossier:
        os.chdir(courant)
        try:
            exc = _lancer(test, exporter, Path(dossier))
        finally:
            os.chdir(ici)
        return exc, sorted(os.listdir(courant))


def verifier_tester_exporter_rapport(test) -> None:
    exc, laisses = _dans_un_bac_a_sable(test, _exporter)
    if exc is not None:
        conseil = (
            "Relisez-vous le fichier avec `sep=';'` ? Sans lui, pandas voit une seule colonne « customer_id;proba ». "
            "Vérifiez aussi vos valeurs attendues : seulement les clients avec proba >= seuil, pile au seuil compris."
        )
        if isinstance(exc, FileExistsError):
            conseil = ("La fonction refuse d'écraser un rapport : pour l'appeler une seconde fois, entourez l'appel de "
                       "`with pytest.raises(FileExistsError):`.")
        raise AssertionError(
            f"Votre test ÉCHOUE sur la version CORRECTE de la fonction ({_decrire(exc)}). {conseil}"
        )
    if laisses:
        raise AssertionError(
            f"Votre test a laissé {', '.join(laisses)} dans le dossier courant (en vrai : dans votre dépôt). "
            "N'écrivez que dans le `dossier` reçu en paramètre, comme avec `tmp_path` dans pytest."
        )

    survivants = []
    for mutant, defaut, conseil in _MUTANTS_EXPORT:
        exc, _ = _dans_un_bac_a_sable(test, mutant)
        if exc is None:
            survivants.append((defaut, conseil))
    if survivants:
        if len(survivants) == len(_MUTANTS_EXPORT):
            raise AssertionError(
                "Votre test passe sur TOUTES les versions boguées : il ne vérifie presque rien. Relisez le fichier écrit "
                "(`pd.read_csv(dossier / 'clients_a_risque.csv', sep=';')`) et comparez-le à ce que promet la docstring."
            )
        defaut, conseil = survivants[0]
        raise AssertionError(
            f"{len(_MUTANTS_EXPORT) - len(survivants)}/{len(_MUTANTS_EXPORT)} bugs détectés. Votre test laisse passer "
            f"une version qui {defaut}. {conseil}"
        )


# ============================================================================ Défi 73 : propriétés


def _repartir(clients: list, n_conseillers: int) -> list[list]:
    if n_conseillers < 1:
        raise ValueError(f"Il faut au moins un conseiller, reçu : {n_conseillers}")
    taille, reste = divmod(len(clients), n_conseillers)
    groupes, debut = [], 0
    for i in range(n_conseillers):
        fin = debut + taille + (1 if i < reste else 0)
        groupes.append(clients[debut:fin])
        debut = fin
    return groupes


def _repartir_bonus_a_la_fin(clients: list, n_conseillers: int) -> list[list]:
    """Juste aussi : ce sont les DERNIERS conseillers qui reçoivent un client de plus."""
    if n_conseillers < 1:
        raise ValueError(f"Il faut au moins un conseiller, reçu : {n_conseillers}")
    taille, reste = divmod(len(clients), n_conseillers)
    groupes, debut = [], 0
    for i in range(n_conseillers):
        fin = debut + taille + (1 if i >= n_conseillers - reste else 0)
        groupes.append(clients[debut:fin])
        debut = fin
    return groupes


def _repartir_perd_le_reste(clients: list, n_conseillers: int) -> list[list]:
    taille = len(clients) // n_conseillers
    return [clients[i * taille:(i + 1) * taille] for i in range(n_conseillers)]


def _repartir_reste_au_dernier(clients: list, n_conseillers: int) -> list[list]:
    taille = len(clients) // n_conseillers
    groupes = [clients[i * taille:(i + 1) * taille] for i in range(n_conseillers - 1)]
    groupes.append(clients[(n_conseillers - 1) * taille:])
    return groupes


def _repartir_sans_listes_vides(clients: list, n_conseillers: int) -> list[list]:
    return [groupe for groupe in _repartir(clients, n_conseillers) if groupe]


def _repartir_vide_l_entree(clients: list, n_conseillers: int) -> list[list]:
    taille, reste = divmod(len(clients), n_conseillers)
    return [[clients.pop(0) for _ in range(taille + (1 if i < reste else 0))] for i in range(n_conseillers)]


def _repartir_limite_100(clients: list, n_conseillers: int) -> list[list]:
    return _repartir(clients[:100], n_conseillers)  # une limite oubliée d'un ancien système


def _repartir_a_tour_de_role(clients: list, n_conseillers: int) -> list[list]:
    return [clients[i::n_conseillers] for i in range(n_conseillers)]


_MUTANTS_REPARTIR = [
    (_repartir_perd_le_reste, "perd des clients quand la division ne tombe pas juste (10 clients pour 3 conseillers : 9 répartis)",
     "Vérifiez la garantie 2 : recollées bout à bout, les listes redonnent tous les clients "
     "(`[c for g in groupes for c in g] == avant`)."),
    (_repartir_reste_au_dernier, "donne tout le reste au dernier conseiller (11 clients pour 4 : 2, 2, 2 et 5)",
     "Vérifiez la garantie 3, `max(tailles) - min(tailles) <= 1`, sur des cas où le reste de la division vaut au "
     "moins 2. « 10 clients pour 3 » ne le révèle pas : faites varier les tailles."),
    (_repartir_sans_listes_vides, "oublie les conseillers sans client quand il y a moins de clients que de conseillers "
     "(2 clients pour 5 conseillers : 2 listes au lieu de 5)",
     "Générez aussi des cas avec plus de conseillers que de clients, et avec 0 client."),
    (_repartir_vide_l_entree, "vide la liste reçue (elle retire les clients avec `pop`)",
     "Vérifiez la garantie 4 : gardez une copie avant l'appel (`avant = list(clients)`), puis `assert clients == avant`."),
    (_repartir_limite_100, "ignore les clients au-delà du 100e (une limite oubliée d'un ancien système)",
     "Faites varier le nombre de clients jusqu'à quelques centaines : un test qui ne génère que de petites listes "
     "ne voit pas ce bug."),
    (_repartir_a_tour_de_role, "distribue les clients à tour de rôle (C0 au 1er, C1 au 2e…) au lieu de blocs consécutifs",
     "La garantie 2 porte aussi sur l'ORDRE : recollées bout à bout, les listes doivent redonner exactement `clients`. "
     "Comparer des ensembles (set) ou des longueurs ne suffit pas."),
]

DUREE_MAX = 5.0  # secondes : une suite de tests doit rester rapide pour qu'on la lance souvent


def verifier_tester_repartir(test) -> None:
    debut = time.perf_counter()
    exc = _lancer(test, _repartir)
    duree = time.perf_counter() - debut
    if exc is not None:
        raise AssertionError(
            f"Votre test ÉCHOUE sur la version CORRECTE de la fonction ({_decrire(exc)}). Un test qui crie au loup est "
            "pire que pas de test. Relisez les 4 garanties. Pour la n° 2, comparez à une COPIE faite avant l'appel. "
            "Pour la n° 3, comparez la plus grande et la plus petite taille."
        )
    if duree > DUREE_MAX:
        raise AssertionError(
            f"Votre test prend {duree:.1f} s ({math.ceil(DUREE_MAX)} s au plus) : une suite de tests lente finit par ne plus être "
            "lancée. Quelques centaines de cas tirés au hasard suffisent."
        )
    exc = _lancer(test, _repartir_bonus_a_la_fin)
    if exc is not None:
        raise AssertionError(
            f"Votre test passe sur la fonction fournie, mais ÉCHOUE sur une autre version JUSTE ({_decrire(exc)}). Elle "
            "donne le client en plus aux DERNIERS conseillers (10 clients pour 3 : 3, 3 et 4). La docstring ne précise pas "
            "qui le reçoit : ne testez que les 4 garanties promises, pas un résultat exact."
        )
    _verifier_mutants(
        test, _MUTANTS_REPARTIR,
        tout_passe="Générez des cas au hasard (random.Random(0)) et vérifiez chacune des 4 garanties avec un assert.",
    )


SERIE = Serie(
    code="11",
    titre="Tester comme un·e pro : frontières, flottants, erreurs, faux objets, fichiers, propriétés",
    module_eleve="defis.serie_11",
    module_corrige="defis.corrige_serie_11",
    defis=(
        Defi(68, "cas_limites_anciennete", "🟢", "6.1", verifier_cas_limites_anciennete),
        Defi(69, "tester_taux_churn", "🟢", "6.1", verifier_tester_taux_churn),
        Defi(70, "tester_lire_contrat", "🟡", "—", verifier_tester_lire_contrat),
        Defi(71, "tester_relancer_clients", "🟡", "—", verifier_tester_relancer_clients),
        Defi(72, "tester_exporter_rapport", "🟡", "—", verifier_tester_exporter_rapport),
        Defi(73, "tester_repartir", "🔴", "—", verifier_tester_repartir),
    ),
)

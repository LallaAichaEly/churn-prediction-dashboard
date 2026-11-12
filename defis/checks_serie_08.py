"""Vérifications automatiques de la série 08 (assembler les données).

Les messages visent les erreurs SILENCIEUSES de l'assemblage de données, celles qui ne plantent
pas mais faussent le modèle : une regex sans frontière de mot, une jointure interne qui perd des
clients ou une jointure avant agrégation qui les duplique, un champ JSON absent ou null, une boucle
ligne à ligne, un `concat` qui empile au lieu de remplacer, un événement « du futur » compté dans
les variables.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import re
import time
from datetime import date

import numpy as np
import pandas as pd

from .verif import Defi, Serie


def _pas_none(resultat, quoi: str):
    if resultat is None:
        raise AssertionError(f"{quoi} : la fonction a retourné None. Avez-vous oublié `return` ?")
    return resultat


def _vide(valeur) -> bool:
    """None, NaN, NaT, pd.NA : une valeur absente, quelle que soit sa forme."""
    if valeur is None or valeur is pd.NaT or valeur is pd.NA:
        return True
    return isinstance(valeur, (float, np.floating)) and math.isnan(valeur)


# ============================================================================ Défi 50 : regex

_NOTES = [
    # (texte, attendu, cas)
    ("Le client C00123 a rappelé.", ["C00123"], "une note simple"),
    ("Dossiers C00123 C00456 et C00789", ["C00123", "C00456", "C00789"], "plusieurs identifiants"),
    ("Le client C00123 a rappelé, voir aussi c00456.", ["C00123", "C00456"], "l'exemple de la docstring"),
    ("Dossiers C00123, C00456 et (C00789).", ["C00123", "C00456", "C00789"], "de la ponctuation collée"),
    ("Transfert C00001/C00002 ;C00003", ["C00001", "C00002", "C00003"], "des identifiants séparés par « / » et « ; »"),
    ("Client C00456 transféré depuis C00123 ; C00456 rappellera.", ["C00456", "C00123"],
     "un client cité deux fois"),
    ("c00042 a appelé. C00042 rappellera demain.", ["C00042"], "le même client en minuscule puis en majuscule"),
    ("Code agence AC00123, client C00456.", ["C00456"], "un code agence « AC00123 »"),
    ("Ancien numéro C001234, nouveau C00042.", ["C00042"], "l'ancienne numérotation à 6 chiffres « C001234 »"),
    ("Aucun identifiant ici : 00123, C123, C 00123.", [], "une note sans identifiant"),
    ("Rappel C00321\nPuis C00654", ["C00321", "C00654"], "une note sur deux lignes"),
]
_NOTES_VIDES = [("", "une note vide"), (None, "une note None"), (math.nan, "une cellule vide de pandas (NaN)")]


def _sonde(f, texte: str):
    """Appelle votre fonction sur un petit texte révélateur (None si elle plante)."""
    try:
        return f(texte)
    except Exception:  # noqa: BLE001 — simple sonde : le vrai diagnostic est ailleurs
        return None


def _diagnostic_note(f, texte: str, obtenu: list, attendu: list[str], appel: str, cas: str) -> str:
    base = f"{appel} renvoie {obtenu!r} au lieu de {attendu!r} ({cas})"
    if any(isinstance(x, tuple) for x in obtenu):
        return (f"{base}. La liste contient des tuples : votre regex a des parenthèses (groupes), et re.findall "
                "renvoie alors le contenu des groupes. Retirez les parenthèses, ou écrivez (?:...) pour un groupe "
                "qui ne capture pas.")
    if any(not isinstance(x, str) for x in obtenu):
        return f"{base}. La liste doit contenir des textes comme « C00123 »."
    if any(x != x.upper() for x in obtenu):
        return f"{base}. Renvoyez la forme en MAJUSCULE : « c00456 » devient « C00456 » (.upper())."
    if len(obtenu) != len(set(obtenu)) and set(obtenu) == set(attendu):
        return (f"{base}. Chaque client une seule fois : dédoublonnez en gardant l'ordre, par exemple "
                "list(dict.fromkeys(liste)) (un set, lui, perdrait l'ordre).")
    if len(obtenu) == len(set(obtenu)) and set(obtenu) == set(attendu):
        return (f"{base}. Les bons identifiants, mais pas dans l'ordre de PREMIÈRE apparition dans la note "
                "(avez-vous trié, ou dédoublonné avec un set ?).")
    en_trop = [x for x in obtenu if x not in attendu]
    if en_trop:
        e = en_trop[0]
        if re.search(rf"\w{e}|{e}\w", texte, flags=re.IGNORECASE):
            return (f"{base}. « {e} » est un morceau d'un mot plus long (« AC00123 » est un code agence, « C001234 » "
                    "une ancienne numérotation). Encadrez le motif de frontières de mot : r\"\\bC\\d{5}\\b\".")
        return f"{base}. « {e} » n'est pas un identifiant : exactement « C » puis 5 chiffres, collés."
    manquants = [x for x in attendu if x not in obtenu]
    if manquants:
        m = manquants[0]
        # On SONDE votre fonction sur des textes minimaux pour trouver la vraie cause.
        if not re.search(m, texte) and _sonde(f, f"note {m.lower()} fin") != [m]:
            return (f"{base}. « {m.lower()} » est écrit en minuscule dans la note : ajoutez flags=re.IGNORECASE "
                    "(ou cherchez [Cc]), puis mettez le résultat en majuscule.")
        if _sonde(f, f"Dossiers ({m}), {m}.") != [m]:
            return (f"{base}. Il manque {m}, collé à de la ponctuation. Découper avec texte.split() laisse « C00123, » "
                    "ou « C00001/C00002 » en un seul morceau : une regex (re.findall) trouve le motif où qu'il soit.")
        return f"{base}. Il manque {manquants}."
    return base + "."


def verifier_extraire_identifiants(f) -> None:
    for texte, attendu, cas in _NOTES:
        appel = f"extraire_identifiants({texte!r})"
        obtenu = _pas_none(f(texte), appel)
        if isinstance(obtenu, str):
            raise AssertionError(
                f"{appel} renvoie le texte {obtenu!r} : renvoyez une LISTE, même pour un seul identifiant. "
                "re.findall renvoie toutes les correspondances (re.search s'arrête à la première)."
            )
        if isinstance(obtenu, (set, frozenset)):
            raise AssertionError(
                f"{appel} renvoie un set : il n'a pas d'ordre, or on veut l'ordre d'apparition dans la note. "
                "Dédoublonnez avec list(dict.fromkeys(liste))."
            )
        if not isinstance(obtenu, list):
            raise AssertionError(f"{appel} : attendu une liste, obtenu {type(obtenu).__name__} ({obtenu!r}).")
        if obtenu != attendu:
            raise AssertionError(_diagnostic_note(f, texte, obtenu, attendu, appel, cas))

    for texte, cas in _NOTES_VIDES:
        appel = f"extraire_identifiants({texte!r})"
        try:
            obtenu = f(texte)
        except NotImplementedError:
            raise
        except (TypeError, AttributeError) as exc:
            raise AssertionError(
                f"{appel} plante ({type(exc).__name__}: {exc}) sur {cas}. Dans un DataFrame, une note vide arrive en "
                "None ou en NaN (un float !) : commencez par `if not isinstance(texte, str): return []`."
            ) from None
        if obtenu != []:
            raise AssertionError(f"{appel} renvoie {obtenu!r} au lieu de [] ({cas}).")


# ============================================================================ Défi 51 : jointure


def _cas_factures():
    """(description, clients, factures, totaux attendus dans l'ordre de clients)."""
    clients = pd.DataFrame({"customer_id": ["C1", "C2", "C3"], "contract": ["Mensuel", "Annuel", "Mensuel"]})
    factures = pd.DataFrame({"customer_id": ["C1", "C1", "C3", "C9"], "montant": [50.0, 30.0, 20.0, 99.0]})
    yield "l'exemple de la docstring", clients, factures, [80.0, 0.0, 20.0]

    # Des clients FILTRÉS : index 10, 42, 7 (pas 0, 1, 2), ordre quelconque, une colonne de plus.
    clients = pd.DataFrame(
        {"customer_id": ["C3", "C1", "C2"], "contract": ["Mensuel", "Annuel", "Mensuel"],
         "monthly_charges": [30.0, 70.0, 50.0]},
        index=[10, 42, 7],
    )
    factures = pd.DataFrame({"customer_id": ["C3", "C1", "C3", "C1", "C3"], "montant": [20.0, 50.0, 20.0, -50.0, 20.0]})
    yield "des clients filtrés (index 10, 42, 7) et un avoir de -50 €", clients, factures, [60.0, 0.0, 0.0]

    clients = pd.DataFrame({"customer_id": ["C1", "C2"], "contract": ["Annuel", "Mensuel"]})
    factures = pd.DataFrame({"customer_id": pd.Series([], dtype=object), "montant": pd.Series([], dtype=float)})
    yield "aucune facture ce mois-ci", clients, factures, [0.0, 0.0]


def _verifier_cas_factures(f, cas, clients, factures, attendu) -> None:
    avant_c, avant_f = clients.copy(), factures.copy()
    obtenu = _pas_none(f(clients, factures), "ajouter_total_factures")
    if not (clients.equals(avant_c) and factures.equals(avant_f)):
        raise AssertionError(
            f"Sur {cas}, votre fonction a modifié `clients` ou `factures` (colonne ajoutée sur place ?). Renvoyez un "
            "NOUVEAU DataFrame : clients.assign(total_factures=...) ou clients.copy()."
        )
    if not isinstance(obtenu, pd.DataFrame):
        raise AssertionError(f"Sur {cas}, attendu un DataFrame, obtenu {type(obtenu).__name__}.")
    if "total_factures" not in obtenu.columns:
        raise AssertionError(
            f"Sur {cas}, pas de colonne `total_factures` (colonnes : {list(obtenu.columns)}). Après un merge, la colonne "
            "s'appelle souvent « montant » : renommez-la (.rename)."
        )
    manquantes = [c for c in clients.columns if c not in obtenu.columns]
    if manquantes:
        raise AssertionError(f"Sur {cas}, les colonnes {manquantes} de `clients` ont disparu : gardez-les toutes.")

    ids = list(obtenu["customer_id"])
    if "C9" in ids:
        raise AssertionError(
            f"Sur {cas}, le client inconnu C9 (une facture sans client) a été ajouté : vous faites une jointure "
            "externe (how=\"outer\" ou \"right\"). Seuls les clients de `clients` comptent : how=\"left\"."
        )
    if len(obtenu) > len(clients):
        doublons = sorted({i for i in ids if ids.count(i) > 1})
        raise AssertionError(
            f"Sur {cas}, {len(obtenu)} lignes pour {len(clients)} clients : {doublons} apparaissent plusieurs fois, une "
            "fois par FACTURE. Vous avez joint avant d'agréger : faites d'abord "
            "factures.groupby(\"customer_id\")[\"montant\"].sum() (une ligne par client), puis la jointure. "
            "Astuce pro : merge(..., validate=\"many_to_one\") plante au lieu de dupliquer en silence."
        )
    if len(obtenu) < len(clients):
        perdus = [c for c in clients["customer_id"] if c not in ids]
        raise AssertionError(
            f"Sur {cas}, {perdus} ont disparu : ce sont les clients SANS facture. pd.merge fait par défaut une jointure "
            "INTERNE (how=\"inner\") qui ne garde que les clients présents des deux côtés. Utilisez how=\"left\" "
            "(ou .map), puis remplacez les NaN par 0."
        )
    if ids != list(clients["customer_id"]):
        raise AssertionError(
            f"Sur {cas}, les clients ne sont plus dans l'ordre de `clients` ({ids} au lieu de "
            f"{list(clients['customer_id'])}). Avez-vous trié, ou mis les factures à gauche de la jointure ?"
        )
    if list(obtenu.index) != list(clients.index):
        raise AssertionError(
            f"Sur {cas}, l'index vaut {list(obtenu.index)} au lieu de {list(clients.index)} : merge RENUMÉROTE l'index. "
            "Ces clients viennent d'un filtrage, et leur index sert à les retrouver ailleurs (prédictions, autre "
            "table). Gardez-le : resultat.index = clients.index, ou utilisez clients[\"customer_id\"].map(totaux)."
        )
    totaux = list(obtenu["total_factures"])
    if any(_vide(t) for t in totaux):
        raise AssertionError(
            f"Sur {cas}, total_factures contient des NaN ({totaux}) pour les clients sans facture : ils n'ont rien payé, "
            "leur total vaut 0.0 (.fillna(0.0))."
        )
    for client, t, a in zip(ids, totaux, attendu):
        if not math.isclose(float(t), a, abs_tol=1e-9):
            nb = int((factures["customer_id"] == client).sum())
            indice = " (c'est le NOMBRE de factures : utilisez .sum(), pas .count() ni .size())" if t == nb != a else ""
            raise AssertionError(f"Sur {cas}, total_factures de {client} vaut {t} au lieu de {a}{indice}.")


def verifier_ajouter_total_factures(f) -> None:
    for cas, clients, factures, attendu in _cas_factures():
        _verifier_cas_factures(f, cas, clients, factures, attendu)


# ============================================================================ Défi 52 : JSON

_COLONNES_API = ["customer_id", "contrat", "nb_factures", "facture_moyenne", "nb_options"]

_CLIENT_COMPLET = {
    "id": "C00001",
    "contrat": {"type": "Mensuel", "debut": "2023-01-15"},
    "factures": [{"mois": "2024-01", "montant": 89.9}, {"mois": "2024-02", "montant": 92.1}],
    "options": ["fibre", "tv"],
}
_CLIENTS_INCOMPLETS = [
    {"id": "C00002", "contrat": None, "factures": [], "options": None},
    {"id": "C00003"},
    {"id": "C00004", "contrat": {"debut": "2024-01-01"}, "factures": None, "options": []},
    {"id": "C00005", "contrat": {"type": "Annuel"}, "factures": [{"mois": "2024-02", "montant": 40.0}]},
]
# (customer_id, contrat, nb_factures, facture_moyenne, nb_options)
_LIGNE_COMPLETE = ("C00001", "Mensuel", 2, 91.0, 2)
_LIGNES_INCOMPLETES = [
    ("C00002", None, 0, math.nan, 0),
    ("C00003", None, 0, math.nan, 0),
    ("C00004", None, 0, math.nan, 0),
    ("C00005", "Annuel", 1, 40.0, 0),
]


def _cas_api():
    """(description, entrée, lignes attendues)."""
    yield "un client complet", {"page": 1, "clients": [_CLIENT_COMPLET]}, [_LIGNE_COMPLETE]
    page = {"page": 1, "clients": [_CLIENT_COMPLET, *_CLIENTS_INCOMPLETS]}
    yield "des champs absents ou null", page, [_LIGNE_COMPLETE, *_LIGNES_INCOMPLETES]
    yield "le TEXTE brut de la réponse (une chaîne JSON)", json.dumps(page), [_LIGNE_COMPLETE, *_LIGNES_INCOMPLETES]
    yield "une page vide ({\"clients\": []})", {"page": 9, "clients": []}, []
    yield "une réponse sans clé \"clients\"", {"page": 9, "erreur": "page inexistante"}, []


def _appeler_api(f, entree, cas: str):
    texte = isinstance(entree, str)
    try:
        return f(entree)
    except NotImplementedError:
        raise
    except (AttributeError, TypeError) as exc:
        if texte:
            raise AssertionError(
                f"Sur {cas}, votre fonction plante ({type(exc).__name__}: {exc}). La réponse arrive ici sous forme de "
                "TEXTE : convertissez-la d'abord en dictionnaire avec json.loads(reponse) si isinstance(reponse, str)."
            ) from None
        if "NoneType" in str(exc):
            raise AssertionError(
                f"Sur {cas}, votre fonction plante ({type(exc).__name__}: {exc}). Un champ vaut null (None) : "
                "client.get(\"contrat\") renvoie None, et None.get(...) ou len(None) plantent. Écrivez "
                "client.get(\"contrat\") or {} et client.get(\"factures\") or [] : on obtient {} ou [] que la clé "
                "soit absente OU null."
            ) from None
        if "JSON object must be" in str(exc):
            raise AssertionError(
                f"Sur {cas}, json.loads plante ({exc}) : ici, la réponse est DÉJÀ un dictionnaire. N'appelez json.loads "
                "que si isinstance(reponse, str)."
            ) from None
        raise AssertionError(f"Sur {cas}, votre fonction plante : {type(exc).__name__}: {exc}") from None
    except KeyError as exc:
        if exc.args and exc.args[0] == "clients":
            raise AssertionError(
                f"Sur {cas}, KeyError: 'clients'. Une réponse sans clients n'a pas de clé \"clients\" : "
                "reponse.get(\"clients\") or [] renvoie une liste vide au lieu de planter."
            ) from None
        raise AssertionError(
            f"Sur {cas}, KeyError: {exc}. Le client C00003 n'a que son id : les autres clés sont ABSENTES. "
            "dict[\"cle\"] plante sur une clé absente, dict.get(\"cle\") renvoie None (ou la valeur par défaut "
            "donnée en second argument)."
        ) from None
    except (ZeroDivisionError, ValueError) as exc:
        if isinstance(exc, ZeroDivisionError) or type(exc).__name__ == "StatisticsError":
            raise AssertionError(
                f"Sur {cas}, votre fonction plante ({type(exc).__name__}: {exc}) : la moyenne d'une liste VIDE de "
                "factures. Elle n'existe pas : renvoyez math.nan quand il n'y a aucune facture."
            ) from None
        raise AssertionError(f"Sur {cas}, votre fonction plante : {type(exc).__name__}: {exc}") from None


def _verifier_ligne_api(ligne: dict, attendue: tuple, cas: str) -> None:
    cid, contrat, nb_f, moyenne, nb_o = attendue
    if contrat is None:
        if isinstance(ligne["contrat"], dict):
            raise AssertionError(f"Sur {cas}, `contrat` de {cid} est un dictionnaire : gardez seulement son \"type\".")
        if not _vide(ligne["contrat"]):
            raise AssertionError(
                f"Sur {cas}, `contrat` de {cid} vaut {ligne['contrat']!r} au lieu de None : son contrat est absent, null "
                "ou sans type. Une valeur absente se note None (pas \"\" ni \"None\"), pour que pandas la voie (isna)."
            )
    elif ligne["contrat"] != contrat:
        raise AssertionError(f"Sur {cas}, `contrat` de {cid} vaut {ligne['contrat']!r} au lieu de {contrat!r}.")

    for colonne, attendu in (("nb_factures", nb_f), ("nb_options", nb_o)):
        valeur = ligne[colonne]
        if _vide(valeur):
            raise AssertionError(
                f"Sur {cas}, `{colonne}` de {cid} est vide au lieu de {attendu} : une liste absente ou null contient "
                "0 élément."
            )
        if valeur != attendu:
            raise AssertionError(f"Sur {cas}, `{colonne}` de {cid} vaut {valeur!r} au lieu de {attendu}.")

    valeur = ligne["facture_moyenne"]
    if math.isnan(moyenne):
        if not _vide(valeur):
            raise AssertionError(
                f"Sur {cas}, `facture_moyenne` de {cid} vaut {valeur!r} au lieu de NaN : ce client n'a AUCUNE facture. "
                "0 € voudrait dire « il paie 0 € par mois » et ferait baisser la moyenne de tout le fichier ; NaN dit "
                "« on ne sait pas » et pandas l'ignore dans les calculs (math.nan)."
            )
    elif _vide(valeur) or not isinstance(valeur, (int, float, np.number)) or not math.isclose(valeur, moyenne):
        raise AssertionError(f"Sur {cas}, `facture_moyenne` de {cid} vaut {valeur!r} au lieu de {moyenne}.")


def verifier_aplatir_reponse_api(f) -> None:
    for cas, entree, attendues in _cas_api():
        avant = copy.deepcopy(entree)
        obtenu = _pas_none(_appeler_api(f, entree, cas), "aplatir_reponse_api")
        if entree != avant:
            raise AssertionError(
                f"Sur {cas}, votre fonction a modifié la réponse reçue (pop, del, affectation…). Lisez-la sans la "
                "changer : l'appelant peut en avoir encore besoin (pour la journaliser, par exemple)."
            )
        if not isinstance(obtenu, pd.DataFrame):
            raise AssertionError(f"Sur {cas}, attendu un DataFrame, obtenu {type(obtenu).__name__}.")
        if list(obtenu.columns) != _COLONNES_API:
            indice = ""
            if not attendues and len(obtenu.columns) == 0:
                indice = (" Avec une liste vide, pd.DataFrame([]) n'a aucune colonne : précisez-les, "
                          "pd.DataFrame(lignes, columns=[...]). Le code qui utilise le tableau ensuite ne plantera pas.")
            elif "id" in obtenu.columns:
                indice = " La colonne « id » de l'API doit s'appeler « customer_id », comme dans le reste du projet."
            raise AssertionError(f"Sur {cas}, colonnes {list(obtenu.columns)} au lieu de {_COLONNES_API}.{indice}")
        if len(obtenu) != len(attendues):
            raise AssertionError(f"Sur {cas}, {len(obtenu)} lignes au lieu de {len(attendues)} (une par client).")
        ids = list(obtenu["customer_id"])
        if ids != [a[0] for a in attendues]:
            raise AssertionError(f"Sur {cas}, customer_id vaut {ids} au lieu de {[a[0] for a in attendues]} (ordre de l'API).")
        for ligne, attendue in zip(obtenu.to_dict("records"), attendues):
            _verifier_ligne_api(ligne, attendue, cas)


# ============================================================================ Défi 53 : vectorisation

_NIVEAUX = {"élevé", "moyen", "faible"}


def _points(df: pd.DataFrame) -> pd.Series:
    return (3 * (df["contract"] == "Mensuel") + 2 * (df["support_calls"] >= 3)
            + 1 * (df["tenure_months"] < 6) - 2 * (df["tech_support"] == "Oui"))


def _reference_lente(df: pd.DataFrame) -> pd.Series:
    """Copie privée de `niveau_risque_lent` : on compare à CETTE version, même si la vôtre a été modifiée."""
    niveaux = []
    for _, client in df.iterrows():
        points = (3 * (client["contract"] == "Mensuel") + 2 * (client["support_calls"] >= 3)
                  + (client["tenure_months"] < 6) - 2 * (client["tech_support"] == "Oui"))
        niveaux.append("élevé" if points >= 4 else "moyen" if points >= 2 else "faible")
    return pd.Series(niveaux, index=df.index, name="niveau_risque")


def _clients_risque() -> list[tuple[str, pd.DataFrame]]:
    combinaisons = pd.DataFrame(
        list(itertools.product(["Mensuel", "Annuel", "Biannuel"], [2, 3, 4], [5, 6], ["Oui", "Non"])),
        columns=["contract", "support_calls", "tenure_months", "tech_support"],
    )
    combinaisons.index = [f"C{i:03d}" for i in range(100, 100 + len(combinaisons))]  # un index qui n'est pas 0, 1, 2…
    from churn.data import generate_customers

    hasard = generate_customers(600, seed=8)
    hasard = hasard[hasard["senior"] == "Non"]  # un filtrage : l'index a des trous
    return [("toutes les combinaisons de règles (seuils 3 appels et 6 mois compris)", combinaisons),
            ("600 clients du générateur, filtrés", hasard)]


class _ParcoursLigneALigne(Exception):
    """Levée par la table espion quand le code la parcourt ligne par ligne."""


class _TableEspion(pd.DataFrame):
    """Un DataFrame normal, sauf qu'il refuse d'être parcouru ligne par ligne."""

    @property
    def _constructor(self):
        return _TableEspion

    def iterrows(self):
        raise _ParcoursLigneALigne("iterrows")

    def itertuples(self, index=True, name="Pandas"):
        raise _ParcoursLigneALigne("itertuples")

    def apply(self, func, axis=0, *args, **kwargs):
        if axis in (1, "columns"):
            raise _ParcoursLigneALigne("apply(axis=1)")
        return super().apply(func, axis, *args, **kwargs)


def _diagnostic_niveaux(df: pd.DataFrame, obtenu: pd.Series, attendu: pd.Series, cas: str) -> str:
    faux = [i for i, (o, a) in enumerate(zip(obtenu, attendu)) if o != a]
    points = _points(df)
    ligne = df.iloc[faux[0]]
    p = int(points.iloc[faux[0]])
    client = ", ".join(f"{c}={ligne[c]!r}" if isinstance(ligne[c], str) else f"{c}={ligne[c]}"
                       for c in ["contract", "support_calls", "tenure_months", "tech_support"])
    base = (f"Sur {cas}, {len(faux)} client(s) mal classé(s). Par exemple ({client}) : {obtenu.iloc[faux[0]]!r} au lieu "
            f"de {attendu.iloc[faux[0]]!r} ({p} point{'s' if abs(p) > 1 else ''})")
    if "élevé" in set(attendu) and "élevé" not in set(obtenu) and "moyen" in set(obtenu):
        return (f"{base}. Aucun client n'est « élevé » : np.select garde la PREMIÈRE condition vraie, et un client à "
                "4 points ou plus a AUSSI « 2 points ou plus ». Mettez la condition la plus stricte en premier : "
                "[points >= 4, points >= 2].")
    lignes_fausses = df.iloc[faux]
    if (lignes_fausses["support_calls"] == 3).all():
        return f"{base}. Tous les clients mal classés ont exactement 3 appels : « 3 appels ou plus », c'est >= 3."
    if (lignes_fausses["tenure_months"] == 6).all():
        return f"{base}. Tous les clients mal classés ont exactement 6 mois : « moins de 6 mois », c'est < 6 (6 ne compte pas)."
    if (lignes_fausses["tech_support"] == "Oui").all():
        return f"{base}. Tous les clients mal classés ont le support technique : il RETIRE 2 points."
    if points.iloc[faux].isin([2, 4]).all():
        return f"{base}. Tous les clients mal classés sont pile sur un seuil (2 ou 4 points) : « 4 ou plus », c'est >= 4."
    return f"{base}. Refaites le calcul de ses points à la main et comparez à votre code, règle par règle."


def _verifier_niveaux(f, df: pd.DataFrame, cas: str) -> None:
    avant = df.copy()
    obtenu = _pas_none(f(df), "niveau_risque")
    if not df.equals(avant) or list(df.columns) != list(avant.columns):
        raise AssertionError(
            f"Sur {cas}, votre fonction a modifié `df` (une colonne « points » ajoutée ?). Gardez les calculs "
            "intermédiaires dans des variables : points = 3 * (...) + ..."
        )
    if isinstance(obtenu, pd.DataFrame):
        raise AssertionError(f"Sur {cas}, attendu une Series (une seule colonne de niveaux), obtenu un DataFrame.")
    if isinstance(obtenu, (np.ndarray, list)):
        raise AssertionError(
            f"Sur {cas}, attendu une Series, obtenu {type(obtenu).__name__}. np.select renvoie un tableau numpy, sans "
            "index : emballez-le, pd.Series(niveaux, index=df.index)."
        )
    if not isinstance(obtenu, pd.Series):
        raise AssertionError(f"Sur {cas}, attendu une Series, obtenu {type(obtenu).__name__}.")
    if len(obtenu) != len(df):
        raise AssertionError(f"Sur {cas}, {len(obtenu)} niveaux pour {len(df)} clients.")
    inconnus = set(obtenu) - _NIVEAUX
    if inconnus:
        raise AssertionError(
            f"Sur {cas}, valeurs inattendues {sorted(map(str, inconnus))} : les niveaux s'écrivent exactement "
            "« élevé » (avec les accents), « moyen » et « faible », comme dans niveau_risque_lent."
        )
    attendu = _reference_lente(df)
    if list(obtenu) != list(attendu):
        raise AssertionError(_diagnostic_niveaux(df, obtenu, attendu, cas))
    if list(obtenu.index) != list(df.index):
        raise AssertionError(
            f"Sur {cas}, les niveaux sont justes mais l'index vaut {list(obtenu.index)[:4]}… au lieu de "
            f"{list(df.index)[:4]}… : pd.Series(niveaux) sans index= renumérote 0, 1, 2… Ajoutez index=df.index, "
            "sinon les niveaux seront collés aux MAUVAIS clients au moment de les rapprocher."
        )


def verifier_niveau_risque(f) -> None:
    jeux = _clients_risque()
    for cas, df in jeux:
        _verifier_niveaux(f, df, cas)

    # Les lignes une par une : la table espion refuse d'être parcourue ainsi.
    cas, df = jeux[0]
    try:
        f(_TableEspion(df.copy()))
    except _ParcoursLigneALigne as exc:
        raise AssertionError(
            f"Le résultat est juste, mais votre fonction parcourt les clients un par un ({exc.args[0]}). C'est "
            "exactement ce qui rend niveau_risque_lent trop lent. Calculez chaque règle sur la COLONNE entière : "
            "(df[\"support_calls\"] >= 3) donne d'un coup une colonne de True/False, et 2 * True vaut 2."
        ) from None

    # Et en vrai : 50 000 clients.
    from churn.data import generate_customers

    grand = generate_customers(50_000, seed=11)
    duree = math.inf
    for _ in range(2):  # le meilleur de deux essais, pour ne pas pénaliser une machine occupée
        debut = time.perf_counter()
        obtenu = f(grand)
        duree = min(duree, time.perf_counter() - debut)
        if duree < 0.5:
            break
    attendu = np.select([_points(grand) >= 4, _points(grand) >= 2], ["élevé", "moyen"], default="faible")
    assert list(obtenu) == list(attendu), "Sur 50 000 clients, les niveaux diffèrent de ceux de niveau_risque_lent."
    if duree >= 0.5:
        raise AssertionError(
            f"Le résultat est juste, mais 50 000 clients prennent {duree:.1f} s (une version vectorisée : environ "
            "0,01 s). Une boucle `for i in range(len(df))` avec df.loc / df.iloc traite encore les clients un par un : "
            "calculez chaque règle sur la colonne entière."
        )


# ============================================================================ Défi 54 : upsert


def _table(cle: str = "customer_id") -> pd.DataFrame:
    return pd.DataFrame({
        cle: ["C1", "C2", "C3"],
        "contract": ["Mensuel", "Annuel", "Mensuel"],
        "monthly_charges": [50.0, 70.0, 30.0],
        "email": ["a@exemple.fr", "b@exemple.fr", "c@exemple.fr"],
    })


def _nouveautes(cle: str = "customer_id") -> pd.DataFrame:
    return pd.DataFrame({
        cle: ["C2", "C4", "C3", "C1"],
        "contract": ["Mensuel", "Biannuel", "Mensuel", "Mensuel"],
        "monthly_charges": [75.0, 90.0, 30.0, 50.0],
        # C3 a retiré son email ; C1 est renvoyé à l'identique (le CRM l'a « touché » sans rien changer).
        "email": ["b@exemple.fr", "d@exemple.fr", None, "a@exemple.fr"],
    })


_ATTENDU_UPSERT = {
    "C1": ("Mensuel", 50.0, "a@exemple.fr"),
    "C2": ("Mensuel", 75.0, "b@exemple.fr"),
    "C3": ("Mensuel", 30.0, None),
    "C4": ("Biannuel", 90.0, "d@exemple.fr"),
}


def _cas_upsert():
    """(description, table, nouveautes, clé, contenu attendu {clé: (contract, monthly_charges, email)})."""
    yield ("l'exemple : C2 modifié, C4 nouveau, C3 retire son email, C1 renvoyé à l'identique", _table(), _nouveautes(),
           "customer_id",
           _ATTENDU_UPSERT)
    deux_fois = pd.DataFrame({
        "email": ["a@exemple.fr", "a.nouvelle@exemple.fr"],
        "customer_id": ["C1", "C1"],
        "monthly_charges": [55.0, 60.0],
        "contract": ["Mensuel", "Annuel"],
    })
    yield ("C1 modifié deux fois dans la journée (colonnes dans un autre ordre)", _table(), deux_fois, "customer_id",
           {"C1": ("Annuel", 60.0, "a.nouvelle@exemple.fr"), "C2": ("Annuel", 70.0, "b@exemple.fr"),
            "C3": ("Mensuel", 30.0, "c@exemple.fr")})
    yield ("une nuit sans aucune modification", _table(), _nouveautes().iloc[0:0], "customer_id",
           {"C1": ("Mensuel", 50.0, "a@exemple.fr"), "C2": ("Annuel", 70.0, "b@exemple.fr"),
            "C3": ("Mensuel", 30.0, "c@exemple.fr")})
    yield "une clé nommée « id » (paramètre cle)", _table("id"), _nouveautes("id"), "id", _ATTENDU_UPSERT


def _normaliser(valeur):
    if _vide(valeur):
        return None
    return round(float(valeur), 6) if isinstance(valeur, (int, float, np.number)) else valeur


def _contenu(df: pd.DataFrame, cle: str) -> dict:
    return {
        ligne[cle]: tuple(_normaliser(ligne[c]) for c in ("contract", "monthly_charges", "email"))
        for ligne in pd.DataFrame(df).to_dict("records")
    }


def _verifier_upsert(f, cas, table, nouveautes, cle, attendu) -> pd.DataFrame:
    avant_t, avant_n = table.copy(), nouveautes.copy()
    try:
        obtenu = f(table, nouveautes, cle=cle)
    except KeyError as exc:
        if cle != "customer_id" and "customer_id" in str(exc):
            raise AssertionError(
                f"Sur {cas}, KeyError {exc} : la colonne identifiant s'appelle ici « {cle} ». Utilisez le paramètre "
                "`cle` partout, au lieu d'écrire \"customer_id\" en dur."
            ) from None
        raise
    _pas_none(obtenu, "mettre_a_jour")
    if not (table.equals(avant_t) and nouveautes.equals(avant_n)):
        raise AssertionError(
            f"Sur {cas}, votre fonction a modifié `table` ou `nouveautes` (affectation, inplace=True…). Si le "
            "traitement plante à mi-chemin, l'ancienne table doit être intacte : travaillez sur une copie."
        )
    if not isinstance(obtenu, pd.DataFrame):
        raise AssertionError(f"Sur {cas}, attendu un DataFrame, obtenu {type(obtenu).__name__}.")
    if list(obtenu.columns) != list(table.columns):
        indice = ""
        if "index" in obtenu.columns:
            indice = " Une colonne « index » est apparue : reset_index(drop=True) jette l'ancien index au lieu d'en faire une colonne."
        elif set(obtenu.columns) == set(table.columns):
            indice = " Remettez les colonnes dans l'ordre de `table` : nouveautes[table.columns]."
        raise AssertionError(f"Sur {cas}, colonnes {list(obtenu.columns)} au lieu de {list(table.columns)}.{indice}")

    cles = list(obtenu[cle])
    doublons = sorted({c for c in cles if cles.count(c) > 1})
    if doublons:
        if obtenu.duplicated().any():
            raise AssertionError(
                f"Sur {cas}, {doublons} apparaissent plusieurs fois. pd.concat EMPILE les tables, il ne remplace rien : "
                "chaque client modifié se retrouve avec son ancienne ET sa nouvelle ligne. Ne gardez qu'une ligne par "
                "client : drop_duplicates(subset=cle, keep=...)."
            )
        raise AssertionError(
            f"Sur {cas}, {doublons} apparaissent plusieurs fois. drop_duplicates() sans `subset` ne retire que les "
            "lignes IDENTIQUES en tout point ; un client modifié diffère de son ancienne version, donc les deux "
            "restent. Comparez seulement la clé : drop_duplicates(subset=cle, ...)."
        )
    manquants = [c for c in attendu if c not in cles]
    if manquants:
        if all(c not in set(table[cle]) for c in manquants):
            raise AssertionError(
                f"Sur {cas}, le nouveau client {manquants} n'a pas été ajouté. DataFrame.update ne fait que mettre à "
                "jour les lignes EXISTANTES : un upsert doit aussi insérer les nouvelles."
            )
        raise AssertionError(
            f"Sur {cas}, {manquants} ont disparu : un client absent des nouveautés n'a pas changé, il doit rester tel "
            "quel (la mise à jour de la nuit ne contient pas toute la base)."
        )
    if len(cles) != len(attendu):
        raise AssertionError(f"Sur {cas}, clients {sorted(cles)} au lieu de {sorted(attendu)}.")

    contenu = _contenu(obtenu, cle)
    for client, valeurs in attendu.items():
        if contenu[client] == valeurs:
            continue
        avant = _contenu(table, cle).get(client)
        lignes_nouvelles = nouveautes[nouveautes[cle] == client]
        # (Vérifié en premier : avec DataFrame.update, C2 est juste mais C3 garde son ancien email.)
        if valeurs[2] is None and contenu[client][2] is not None:
            raise AssertionError(
                f"Sur {cas}, l'email de {client} vaut encore {contenu[client][2]!r} au lieu d'être vide : le client l'a "
                "retiré. DataFrame.update et combine_first IGNORENT les valeurs vides de la mise à jour (ils les "
                "considèrent comme « pas d'information ») : remplacez la LIGNE entière."
            )
        if contenu[client] == avant and len(lignes_nouvelles) > 0:
            raise AssertionError(
                f"Sur {cas}, {client} a gardé son ANCIENNE version {avant} au lieu de {valeurs}. Avec "
                "drop_duplicates, keep=\"first\" (le défaut) garde la première ligne rencontrée, donc l'ancienne : "
                "gardez la dernière, keep=\"last\" (les nouveautés sont placées APRÈS la table)."
            )
        if len(lignes_nouvelles) > 1:
            raise AssertionError(
                f"Sur {cas}, {client} vaut {contenu[client]} au lieu de {valeurs} : il a été modifié deux fois dans la "
                "journée, c'est la DERNIÈRE ligne des nouveautés qui compte."
            )
        raise AssertionError(f"Sur {cas}, {client} vaut {contenu[client]} au lieu de {valeurs}.")

    if list(obtenu.index) != list(range(len(obtenu))):
        raise AssertionError(
            f"Sur {cas}, l'index vaut {list(obtenu.index)} au lieu de 0, 1, 2… : après concat, deux lignes peuvent "
            "porter le même index, et df.loc[2] renverrait alors deux clients. pd.concat(..., ignore_index=True) "
            "ou .reset_index(drop=True)."
        )
    return obtenu


def verifier_mettre_a_jour(f) -> None:
    for cas, table, nouveautes, cle, attendu in _cas_upsert():
        _verifier_upsert(f, cas, table, nouveautes, cle, attendu)

    # Idempotence : le traitement relancé après une panne ne doit rien changer.
    une_fois = f(_table(), _nouveautes())
    deux_fois = f(une_fois.copy(), _nouveautes())
    if len(deux_fois) != len(une_fois) or _contenu(deux_fois, "customer_id") != _contenu(une_fois, "customer_id"):
        raise AssertionError(
            f"Rejouer la même mise à jour change la table ({len(une_fois)} puis {len(deux_fois)} lignes). Un traitement "
            "de nuit est souvent relancé après une panne : appliqué deux fois, il doit donner le même résultat "
            "(il doit être idempotent)."
        )

    corrompue = pd.concat([_table(), _table().iloc[[1]]], ignore_index=True)  # C2 en double
    try:
        f(corrompue, _nouveautes())
    except ValueError:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"Avec une table où C2 est en double, votre fonction lève {type(exc).__name__} : attendu une ValueError."
        ) from None
    raise AssertionError(
        "Avec une table de référence où C2 apparaît deux fois, votre fonction ne dit rien. La référence est corrompue : "
        "levez une ValueError (if table[cle].duplicated().any(): raise ValueError(...)) pour que quelqu'un enquête, "
        "au lieu de continuer à produire des chiffres faux."
    )


# ============================================================================ Défi 55 : événements

# Le journal du CRM. Référence : le 15/03/2024 au matin.
_JOURNAL = [
    ("C1", "2024-03-14", "appel_support"),   # la veille : compte
    ("C1", "2024-02-14", "appel_support"),   # J-30 pile : compte (fenêtre de 30 jours)
    ("C1", "2024-02-13", "appel_support"),   # J-31 : hors fenêtre
    ("C1", "2024-03-10", "connexion"),
    ("C1", "2023-12-01", "reclamation"),     # ancienne, mais les réclamations comptent sur tout l'historique
    ("C2", "2024-03-15", "appel_support"),   # le jour J : pas encore connu le matin
    ("C2", "2024-03-20", "appel_support"),   # dans le FUTUR
    ("C2", "2024-04-01", "reclamation"),     # dans le FUTUR
    ("C2", "2024-01-05", "connexion"),       # dernier événement connu de C2 : 70 jours avant
    ("C5", "2024-03-01", "appel_support"),   # C5 n'est pas à scorer
    ("C4", "2024-03-01", "reclamation"),
    ("C4", "2024-02-01", "reclamation"),
    # C3 : aucun événement
]
_COLONNES_EVT = ["appels_recents", "reclamations", "jours_depuis_dernier"]


def _journal(dates_pandas: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(_JOURNAL, columns=["customer_id", "date", "type"])
    if dates_pandas:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _cas_evenements():
    """(description, journal, clients, date_reference, fenetre, attendu {client: (appels, réclamations, jours)})."""
    yield ("référence au 15/03/2024, fenêtre de 30 jours", _journal(), ["C1", "C2", "C3", "C4"], "2024-03-15", 30,
           {"C1": (2, 1, 1.0), "C2": (0, 0, 70.0), "C3": (0, 0, math.nan), "C4": (0, 2, 14.0)})
    yield ("une fenêtre de 7 jours, une référence de type date et des dates pandas", _journal(dates_pandas=True),
           ["C1", "C2", "C3", "C4"], date(2024, 3, 15), 7,
           {"C1": (1, 1, 1.0), "C2": (0, 0, 70.0), "C3": (0, 0, math.nan), "C4": (0, 2, 14.0)})
    yield ("une autre liste de clients, dans un autre ordre", _journal(), ["C4", "C3", "C1"], "2024-03-15", 30,
           {"C4": (0, 2, 14.0), "C3": (0, 0, math.nan), "C1": (2, 1, 1.0)})


_FUITE = (
    "C'est une FUITE TEMPORELLE : le matin du 15/03, ces événements n'ont pas encore eu lieu. Le modèle entraîné "
    "ainsi paraît excellent au test (il « voit » les appels de la semaine du départ) et s'effondre en production. "
    "Filtrez d'abord : ev[ev[\"date\"] < reference]."
)
_JOUR_J = (
    "Un événement daté du jour de référence lui-même n'est pas connu le matin de la prédiction : gardez les dates "
    "STRICTEMENT avant (<, pas <=)."
)
# (fenêtre, colonne, client, valeur obtenue) → explication
_PIEGES = {
    (30, "appels_recents", "C2", 2): "C2 a un appel le 15/03 (jour J) et un le 20/03 (futur). " + _FUITE,
    (30, "appels_recents", "C2", 1): "C2 a un appel le 15/03, le jour J. " + _JOUR_J,
    (30, "appels_recents", "C1", 1): (
        "C1 a des appels le 14/03, le 14/02 et le 13/02. Le 14/02 est pile à 30 jours du 15/03 : la fenêtre "
        "l'INCLUT (date >= reference - pd.Timedelta(days=fenetre_jours))."),
    (30, "appels_recents", "C1", 3): (
        "C1 a des appels le 14/03, le 14/02 et le 13/02 : le 13/02 est à 31 jours, hors de la fenêtre. Les appels "
        "« récents » ne comptent que sur les `fenetre_jours` derniers jours."),
    (7, "appels_recents", "C1", 2): (
        "Avec une fenêtre de 7 jours, seul l'appel du 14/03 compte (celui du 14/02 est à 30 jours). Utilisez le "
        "paramètre `fenetre_jours` au lieu d'écrire 30 en dur."),
    (30, "reclamations", "C2", 1): "C2 a une réclamation le 01/04, après la référence. " + _FUITE,
    (7, "reclamations", "C2", 1): "C2 a une réclamation le 01/04, après la référence. " + _FUITE,
    (30, "reclamations", "C1", 0): (
        "C1 a une réclamation le 01/12/2023. Les réclamations comptent sur TOUT l'historique connu, pas seulement "
        "dans la fenêtre : n'appliquez la fenêtre qu'aux appels."),
    (7, "reclamations", "C1", 0): (
        "C1 a une réclamation le 01/12/2023. Les réclamations comptent sur TOUT l'historique connu, pas seulement "
        "dans la fenêtre : n'appliquez la fenêtre qu'aux appels."),
    (30, "jours_depuis_dernier", "C2", -17.0): "Le « dernier » événement de C2 que vous prenez est celui du 01/04. " + _FUITE,
    (7, "jours_depuis_dernier", "C2", -17.0): "Le « dernier » événement de C2 que vous prenez est celui du 01/04. " + _FUITE,
    (30, "jours_depuis_dernier", "C2", 0.0): "Le « dernier » événement de C2 que vous prenez est celui du 15/03. " + _JOUR_J,
    (7, "jours_depuis_dernier", "C2", 0.0): "Le « dernier » événement de C2 que vous prenez est celui du 15/03. " + _JOUR_J,
}


def _affiche(valeur) -> str:
    """Une valeur lisible dans un message : 2 plutôt que np.int64(2), NaN plutôt que nan."""
    if _vide(valeur):
        return "NaN"
    if isinstance(valeur, np.generic):
        valeur = valeur.item()
    return repr(valeur)


def _egal(obtenu, attendu) -> bool:
    if isinstance(attendu, float) and math.isnan(attendu):
        return _vide(obtenu)
    return not _vide(obtenu) and math.isclose(float(obtenu), attendu)


def _verifier_cas_evenements(f, cas, journal, clients, reference, fenetre, attendu) -> None:
    avant = journal.copy()
    try:
        obtenu = f(journal, list(clients), reference, fenetre_jours=fenetre)
    except TypeError as exc:
        raise AssertionError(
            f"Sur {cas}, votre fonction plante (TypeError: {exc}). On ne compare pas un texte « 2024-03-15 » à une "
            "date, ni une date à un texte : convertissez tout en dates pandas, pd.Timestamp(date_reference) et "
            "pd.to_datetime(evenements[\"date\"])."
        ) from None
    _pas_none(obtenu, "variables_evenements")
    if not journal.equals(avant):
        raise AssertionError(
            f"Sur {cas}, votre fonction a modifié `evenements` (la colonne date convertie sur place ?). Travaillez sur "
            "une copie : ev = evenements.assign(date=pd.to_datetime(evenements[\"date\"]))."
        )
    if not isinstance(obtenu, pd.DataFrame):
        raise AssertionError(f"Sur {cas}, attendu un DataFrame, obtenu {type(obtenu).__name__}.")
    if "customer_id" in obtenu.columns:
        raise AssertionError(
            f"Sur {cas}, customer_id est une colonne : il doit être l'INDEX (une ligne = un client, que l'on retrouve "
            "par df.loc[\"C1\"]). Utilisez .set_index(\"customer_id\") ou construisez le DataFrame avec index=clients."
        )
    if list(obtenu.columns) != _COLONNES_EVT:
        raise AssertionError(f"Sur {cas}, colonnes {list(obtenu.columns)} au lieu de {_COLONNES_EVT}.")
    index = list(obtenu.index)
    if index != list(clients):
        en_trop = [c for c in index if c not in clients]
        manquants = [c for c in clients if c not in index]
        if en_trop:
            raise AssertionError(
                f"Sur {cas}, {en_trop} ne sont pas dans `clients` : le journal contient d'autres clients, mais on ne "
                "calcule les variables que pour ceux à scorer. .reindex(clients) ne garde que ceux-là."
            )
        if manquants:
            raise AssertionError(
                f"Sur {cas}, {manquants} n'ont pas de ligne : sans aucun événement, ils disparaissent du groupby. Le "
                "modèle doit pourtant les scorer : .reindex(clients, fill_value=0) les remet, avec 0 appel."
            )
        raise AssertionError(f"Sur {cas}, lignes dans l'ordre {index} au lieu de {list(clients)} : .reindex(clients).")

    for colonne in _COLONNES_EVT:
        for client in clients:
            valeur, voulu = obtenu.at[client, colonne], attendu[client][_COLONNES_EVT.index(colonne)]
            if _egal(valeur, voulu):
                continue
            if colonne != "jours_depuis_dernier" and _vide(valeur):
                raise AssertionError(
                    f"Sur {cas}, `{colonne}` de {client} est vide (NaN) au lieu de {voulu} : un client sans événement "
                    "en a 0. .reindex(clients, fill_value=0), ou .fillna(0)."
                )
            if colonne == "jours_depuis_dernier" and client == "C3":
                raise AssertionError(
                    f"Sur {cas}, `jours_depuis_dernier` de C3 vaut {_affiche(valeur)} au lieu de NaN : C3 n'a AUCUN événement. "
                    "0 voudrait dire « vu aujourd'hui », le contraire de la réalité : laissez NaN."
                )
            cle = (fenetre, colonne, client, float(valeur)) if not _vide(valeur) else None
            explication = _PIEGES.get(cle, "")
            raise AssertionError(
                f"Sur {cas}, `{colonne}` de {client} vaut {_affiche(valeur)} au lieu de {_affiche(voulu)}. "
                f"{explication}".rstrip()
            )

    for colonne in ("appels_recents", "reclamations"):
        if not pd.api.types.is_integer_dtype(obtenu[colonne]):
            raise AssertionError(
                f"Sur {cas}, `{colonne}` est de type {obtenu[colonne].dtype} : un nombre d'événements est un entier "
                "(.astype(int)). Les NaN remplacés après coup laissent souvent des float."
            )


def verifier_variables_evenements(f) -> None:
    for cas, journal, clients, reference, fenetre, attendu in _cas_evenements():
        _verifier_cas_evenements(f, cas, journal, clients, reference, fenetre, attendu)


# ============================================================================ La série

SERIE = Serie(
    code="08",
    titre="Assembler les données : texte, jointures, JSON, mises à jour, événements",
    module_eleve="defis.serie_08",
    module_corrige="defis.corrige_serie_08",
    defis=(
        Defi(50, "extraire_identifiants", "🟢", "—", verifier_extraire_identifiants),
        Defi(51, "ajouter_total_factures", "🟢", "—", verifier_ajouter_total_factures),
        Defi(52, "aplatir_reponse_api", "🟡", "—", verifier_aplatir_reponse_api),
        Defi(53, "niveau_risque", "🟡", "—", verifier_niveau_risque),
        Defi(54, "mettre_a_jour", "🟡", "—", verifier_mettre_a_jour),
        Defi(55, "variables_evenements", "🔴", "—", verifier_variables_evenements),
    ),
)

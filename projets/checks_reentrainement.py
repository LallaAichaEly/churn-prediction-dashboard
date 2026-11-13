"""Vérifications automatiques du projet 5 (réentraîner le modèle sans casser la production).

Le correcteur fabrique ses propres lots mensuels (avec une dérive : un concurrent attire les clients
fibre à partir d'un certain mois), y glisse des pièges un par un (un mois mal écrit, « Mensual », une
virgule française, un client en double, un mois sans aucun départ), interroge vos fonctions avec de
faux modèles qui refusent de voir la colonne `churn`, simule un disque plein au milieu d'un
enregistrement, échange un fichier de modèle contre un fichier piégé, et rejoue les versions boguées
classiques de `decider` pour vous dire LAQUELLE vous avez écrite.

Lire ce fichier est permis (les cas testés sont des indices), mais essayez d'abord seul·e.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import pickle
import warnings
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from .checks_retention import _lancer, _pas_none, _proche
from .reentrainement_corrige import decision as ref_decision
from .reentrainement_corrige import decoupage as ref_decoupage
from .reentrainement_corrige import evaluation as ref_evaluation
from .reentrainement_corrige import fichiers as ref_fichiers
from .reentrainement_corrige.entrainement import entrainer as ref_entrainer
from .reentrainement_corrige import registre as ref_registre
from .reentrainement_corrige.modeles import (CIBLE, COLONNES, IDENTIFIANT, MODALITES, MOIS, REGLES, VARIABLES, Decision,
                                              Mesures)
from .verif import Etape, Projet

# ---------------------------------------------------------------------------- outils


def _modeles(module: ModuleType) -> ModuleType:
    """Le `modeles.py` du MÊME paquet que le module testé (le vôtre ou celui du corrigé)."""
    return importlib.import_module(f"{module.__package__}.modeles")


@contextlib.contextmanager
def _dossier_temporaire():
    with tempfile.TemporaryDirectory(prefix="projet_reentrainement_") as dossier:
        yield Path(dossier)


def generer_lots(graine: int = 2026, debut: str = "2025-10", nb_mois: int = 12, clients: int = 300,
                 derive_depuis: str = "2026-04", force: float = 2.0) -> pd.DataFrame:
    """Des lots mensuels fictifs, reproductibles : `projets/exemple_lots.csv` vaut `generer_lots()`.

    Les clients suivent la « vérité cachée » du projet 1 (`src/churn/data.py`). À partir de
    `derive_depuis`, un concurrent attire les clients fibre (sa force monte en trois mois jusqu'à
    `force`) : c'est une dérive du CONCEPT, le lien entre les variables et le départ change, et un
    modèle entraîné avant la dérive se trompe de plus en plus.
    """
    rng = np.random.default_rng(graine)
    morceaux, numero = [], 0
    premier, derive = pd.Period(debut, "M"), pd.Period(derive_depuis, "M")
    for k in range(nb_mois):
        mois = premier + k
        concurrence = min(force, force * ((mois - derive).n + 1) / 3) if mois >= derive else 0.0
        contrat = rng.choice(MODALITES["contract"], clients, p=[0.55, 0.25, 0.20])
        echelle = np.select([contrat == "Mensuel", contrat == "Annuel"], [14.0, 30.0], 45.0)
        anciennete = np.clip(rng.exponential(echelle), 1, 72).round().astype(int)
        internet = rng.choice(MODALITES["internet_service"], clients, p=[0.45, 0.35, 0.20])
        base = np.select([internet == "Fibre", internet == "ADSL"], [80.0, 55.0], 25.0)
        prix = np.clip(base + rng.normal(0, 12, clients), 15, 130).round(2)
        lot = pd.DataFrame({
            MOIS: str(mois),
            IDENTIFIANT: [f"C{numero + i:05d}" for i in range(1, clients + 1)],
            "tenure_months": anciennete,
            "monthly_charges": prix,
            "total_charges": (prix * anciennete * rng.uniform(0.95, 1.05, clients)).round(2),
            "support_calls": rng.poisson(1.3, clients),
            "num_products": rng.integers(1, 5, clients),
            "contract": contrat,
            "payment_method": rng.choice(MODALITES["payment_method"], clients, p=[0.35, 0.30, 0.20, 0.15]),
            "internet_service": internet,
            "tech_support": rng.choice(MODALITES["tech_support"], clients, p=[0.35, 0.65]),
            "senior": rng.choice(MODALITES["senior"], clients, p=[0.18, 0.82]),
        })
        numero += clients
        logit = (-1.2 + 1.6 * (lot.contract == "Mensuel") - 0.9 * (lot.contract == "Biannuel")
                 - 0.035 * lot.tenure_months + 0.018 * (lot.monthly_charges - 65) + 0.45 * lot.support_calls
                 + 0.7 * (lot.payment_method == "Chèque") + 0.5 * (lot.internet_service == "Fibre")
                 - 0.8 * (lot.tech_support == "Oui") - 0.25 * (lot.num_products - 1) + 0.3 * (lot.senior == "Oui")
                 + concurrence * (lot.internet_service == "Fibre") - 0.6 * concurrence * (lot.contract == "Mensuel")
                 + rng.normal(0, 0.6, clients))
        lot[CIBLE] = (rng.uniform(size=clients) < 1 / (1 + np.exp(-logit.to_numpy(float)))).astype(int)
        morceaux.append(lot)
    return pd.concat(morceaux, ignore_index=True)[list(COLONNES)]


def _appel(fonction, quoi: str):
    """Appelle fonction() ; une exception inattendue devient un message qui dit QUEL appel a planté."""
    try:
        return fonction()
    except (NotImplementedError, AssertionError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{quoi} lève {type(exc).__name__}: {exc}") from None


def _leve_value_error(fonction, quoi: str, conseil: str) -> None:
    try:
        resultat = fonction()
    except ValueError:
        return
    except (NotImplementedError, AssertionError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{quoi} lève {type(exc).__name__} ({exc}) : levez ValueError. {conseil}") from None
    raise AssertionError(f"{quoi} devrait lever ValueError, mais renvoie {type(resultat).__name__}. {conseil}")


# ============================================================================ Étape 1 : problemes


def _avec(lots: pd.DataFrame, **cases) -> pd.DataFrame:
    """Une copie de `lots` où l'on écrit des valeurs : _avec(lots, contract={3: "Mensual"})."""
    copie = lots.copy()
    for colonne, valeurs in cases.items():
        if any(isinstance(v, str) for v in valeurs.values()) and copie[colonne].dtype.kind in "biuf":
            copie[colonne] = copie[colonne].astype(object)
        elif any(isinstance(v, float) for v in valeurs.values()) and copie[colonne].dtype.kind in "biu":
            copie[colonne] = copie[colonne].astype(float)
        for ligne, valeur in valeurs.items():
            copie.loc[ligne, colonne] = valeur
    return copie


def _problemes(f, lots: pd.DataFrame, cas: str) -> list[str]:
    avant = lots.copy()
    trouves = _appel(lambda: f(lots), f"problemes({cas})")
    _pas_none(trouves, "problemes")
    if not isinstance(trouves, list) or not all(isinstance(t, str) for t in trouves):
        raise AssertionError(f"problemes({cas}) doit renvoyer une LISTE de phrases (str), pas {trouves!r:.120}.")
    if not lots.equals(avant) or list(lots.columns) != list(avant.columns):
        raise AssertionError(f"problemes({cas}) a MODIFIÉ le lot reçu. Vérifier, ce n'est pas réparer : travaillez sur "
                             f"des colonnes lues (`lots[colonne]`), sans `inplace=True` ni nouvelle colonne.")
    return trouves


def _mentionne(trouves: list[str], *mots: str) -> bool:
    return any(all(mot in t for mot in mots) for t in trouves)


#: (cas, modifications, mots qui doivent apparaître dans UNE phrase, conseil si rien n'est trouvé)
_PIEGES = (
    ("2 cases vides dans monthly_charges", {"monthly_charges": {4: np.nan, 9: np.nan}}, ("monthly_charges", "2"),
     "Comptez les cases vides de chaque colonne avec `lots[colonne].isna().sum()`, et dites combien."),
    ("un mois écrit « 2026-9 »", {MOIS: {5: "2026-9", 6: "2026-9"}, CIBLE: {5: 0, 6: 1}}, (MOIS,),
     "Le mois doit avoir la forme AAAA-MM : `lots[MOIS].str.fullmatch(r\"\\d{4}-(0[1-9]|1[0-2])\")`."),
    ("un mois écrit « 2026-13 »", {MOIS: {5: "2026-13", 6: "2026-13"}, CIBLE: {5: 0, 6: 1}}, (MOIS,),
     "Il n'y a pas de 13ᵉ mois : le motif doit n'accepter que 01 à 12."),
    ("un churn qui vaut 2", {CIBLE: {7: 2}}, (CIBLE,),
     "La cible ne peut valoir que 0 ou 1 : `pd.to_numeric(..., errors=\"coerce\").isin([0, 1])`."),
    ("un churn qui vaut « oui »", {CIBLE: {7: "oui"}}, (CIBLE,),
     "Un texte dans la colonne churn : `pd.to_numeric(colonne, errors=\"coerce\")` le transforme en NaN, qui n'est "
     "pas dans {0, 1}."),
    ("une facture écrite « 45,90 » (virgule française, export Excel)", {"monthly_charges": {3: "45,90"}},
     ("monthly_charges",),
     "Une valeur illisible comme nombre : `pd.to_numeric(colonne, errors=\"coerce\")` donne NaN là où le texte n'est "
     "pas un nombre."),
    ("un nombre d'appels au support négatif", {"support_calls": {6: -1}}, ("support_calls",),
     "Les colonnes numériques ne peuvent pas être négatives."),
    ("un contrat « Mensual » (faute de frappe)", {"contract": {2: "Mensual"}}, ("contract",),
     "Comparez chaque colonne catégorielle à MODALITES : `~lots[colonne].isin(MODALITES[colonne])`. Sans ce "
     "contrôle, le modèle IGNORE la valeur inconnue en silence (`handle_unknown=\"ignore\"`)."),
    ("un service internet « fibre » en minuscules", {"internet_service": {2: "fibre"}}, ("internet_service",),
     "« fibre » n'est pas « Fibre » : les modalités se comparent exactement."),
)


def verifier_problemes(module: ModuleType) -> None:
    f = module.problemes
    propre = generer_lots(graine=51, debut="2026-01", nb_mois=3, clients=40)

    trouves = _problemes(f, propre, "un lot sans défaut")
    if trouves:
        raise AssertionError(f"Sur un lot sans aucun défaut, problemes renvoie {trouves[:2]} : il devrait renvoyer une "
                             f"liste VIDE. Un contrôle trop sévère fait refuser tous les lots, et on finit par le couper.")

    # Les colonnes manquantes : une seule phrase, et on s'arrête là.
    sans = propre.drop(columns=[CIBLE, "contract"])
    trouves = _problemes(f, sans, "un lot sans les colonnes churn et contract")
    if not _mentionne(trouves, CIBLE, "contract"):
        raise AssertionError(f"Il manque les colonnes churn et contract : problemes doit le dire dans UNE phrase qui cite "
                             f"les deux, obtenu {trouves!r:.200}. Comparez COLONNES et `lots.columns`.")
    if len(trouves) != 1:
        raise AssertionError(f"Il manque deux colonnes et problemes renvoie {len(trouves)} phrases : quand des colonnes "
                             f"manquent, renvoyez tout de suite UNE phrase (les autres règles liraient ces colonnes).")

    # Cinquante cases vides : UNE phrase pour la colonne, pas cinquante.
    trouves = _problemes(f, _avec(propre, tech_support={i: np.nan for i in range(50)}),
                         "50 cases vides dans tech_support")
    sur_colonne = [t for t in trouves if "tech_support" in t]
    if len(sur_colonne) > 2:
        raise AssertionError(f"50 cases vides dans tech_support donnent {len(sur_colonne)} phrases : écrivez UNE phrase "
                             f"par colonne avec le nombre (« tech_support : 50 valeur(s) manquante(s) »), sinon le "
                             f"vrai problème se perd dans une liste de 500 lignes.")
    if not any("50" in t for t in sur_colonne):
        raise AssertionError(f"50 cases vides dans tech_support : la phrase doit dire combien (50), obtenu "
                             f"{sur_colonne!r:.200}.")

    # Un piège à la fois.
    for cas, cases, mots, conseil in _PIEGES:
        trouves = _problemes(f, _avec(propre, **cases), cas)
        if not trouves:
            raise AssertionError(f"Lot avec {cas} : problemes ne trouve rien. {conseil}")
        if not _mentionne(trouves, *mots):
            raise AssertionError(f"Lot avec {cas} : aucune phrase ne cite {' et '.join(repr(m) for m in mots)} "
                                 f"({trouves!r:.200}). Dites QUELLE colonne pose problème. {conseil}")

    # Un mois vide : NaN ne doit pas faire planter la vérification du format.
    trouves = _problemes(f, _avec(propre, **{MOIS: {11: np.nan}}), "une case vide dans la colonne mois")
    if not _mentionne(trouves, MOIS):
        raise AssertionError(f"Une case vide dans la colonne mois : problemes doit le signaler, obtenu {trouves!r:.200}.")

    # Les doublons : (mois, client), pas le client seul.
    double = pd.concat([propre, propre.iloc[[8]]], ignore_index=True)
    client = propre.iloc[8][IDENTIFIANT]
    trouves = _problemes(f, double, f"le client {client} deux fois dans le même mois")
    if not _mentionne(trouves, client):
        raise AssertionError(f"Le client {client} apparaît deux fois dans le même mois : problemes doit le signaler en "
                             f"citant un exemple ({client}), obtenu {trouves!r:.200}. Pensez à "
                             f"`lots.duplicated([MOIS, IDENTIFIANT])`.")
    deux_mois = propre.copy()
    premier_mois, deuxieme = sorted(deux_mois[MOIS].unique())[:2]
    a = deux_mois.index[deux_mois[MOIS] == premier_mois][0]
    b = deux_mois.index[deux_mois[MOIS] == deuxieme][0]
    deux_mois.loc[b, IDENTIFIANT] = deux_mois.loc[a, IDENTIFIANT]
    trouves = _problemes(f, deux_mois, "un même client présent dans deux mois différents")
    if trouves:
        raise AssertionError(f"Un client présent en {premier_mois} ET en {deuxieme} n'est PAS un doublon : chaque mois est "
                             f"une nouvelle photo des clients. problemes renvoie pourtant {trouves!r:.200}. Cherchez "
                             f"les doublons sur le couple (mois, customer_id), pas sur customer_id seul.")

    # Un mois sans aucun départ : la colonne churn n'a pas été remplie.
    vide = propre.copy()
    dernier = sorted(vide[MOIS].unique())[-1]
    vide.loc[vide[MOIS] == dernier, CIBLE] = 0
    trouves = _problemes(f, vide, f"un mois ({dernier}) où personne n'est parti")
    if not _mentionne(trouves, dernier):
        raise AssertionError(f"En {dernier}, churn vaut 0 pour TOUS les clients : c'est l'incident de mars (l'export "
                             f"avait oublié de remplir la colonne). problemes doit citer ce mois, obtenu "
                             f"{trouves!r:.200}. Calculez le taux de départ par mois "
                             f"(`groupby(MOIS)[CIBLE].mean()`) et signalez 0 % et 100 %.")

    # Tous les problèmes à la fois.
    plusieurs = _avec(propre, contract={2: "Mensual"}, support_calls={6: -1}, **{MOIS: {5: "2026-9"}})
    trouves = _problemes(f, plusieurs, "un lot avec 3 défauts (mois, contract, support_calls)")
    manque = [m for m in (MOIS, "contract", "support_calls") if not _mentionne(trouves, m)]
    if manque:
        raise AssertionError(f"Un lot avec trois défauts (mois mal écrit, « Mensual », appels négatifs) : problemes ne "
                             f"signale pas {', '.join(manque)} ({trouves!r:.200}). Ne vous arrêtez pas au premier "
                             f"problème : rassemblez-les TOUS dans la liste, pour tout corriger en une fois.")


# ============================================================================ Étape 2 : decouper


def _mois_de(df) -> list[str]:
    return sorted(set(map(str, df[MOIS])))


def _decouper(f, lots, *args, cas: str):
    resultat = _appel(lambda: f(lots, *args), f"decouper(lots, {', '.join(map(repr, args))})  ({cas})")
    _pas_none(resultat, "decouper")
    if not (isinstance(resultat, tuple) and len(resultat) == 2
            and all(isinstance(r, pd.DataFrame) for r in resultat)):
        raise AssertionError(f"decouper doit renvoyer un tuple de deux DataFrame (train, test), pas "
                             f"{type(resultat).__name__}.")
    return resultat


def verifier_decouper(module: ModuleType) -> None:
    f = module.decouper
    m = _modeles(module)
    lots = generer_lots(graine=52, debut="2025-01", nb_mois=16, clients=12)
    melange = lots.sample(frac=1.0, random_state=0)  # les lignes ne sont pas dans l'ordre des mois

    train, test = _decouper(f, melange, "2026-01", 12, cas="lignes mélangées")
    if _mois_de(test) == [f"2025-{i:02d}" for i in range(1, 13)] and _mois_de(train) == ["2026-01"]:
        raise AssertionError("decouper renvoie (test, train) : l'ordre attendu est (train, test).")
    if _mois_de(test) != ["2026-01"]:
        raise AssertionError(f"decouper(lots, '2026-01') : test doit contenir le seul mois 2026-01, pas {_mois_de(test)}.")
    attendus = [f"2025-{i:02d}" for i in range(1, 13)]
    if "2026-01" in _mois_de(train):
        raise AssertionError("decouper(lots, '2026-01') : le mois de test se retrouve AUSSI dans train. Un modèle évalué "
                             "sur des clients qu'il a vus pendant l'entraînement paraît meilleur qu'il n'est : train "
                             "s'arrête au mois d'AVANT (`< mois_test`, pas `<=`).")
    futurs = [x for x in _mois_de(train) if x > "2026-01"]
    if futurs:
        raise AssertionError(f"decouper(lots, '2026-01') : train contient {futurs[0]}, un mois APRÈS le mois de test. Le "
                             f"jour où l'on entraîne pour tester 2026-01, ces mois n'existent pas encore : c'est une fuite "
                             f"temporelle. Gardez seulement les mois < mois_test.")
    if _mois_de(train) != attendus:
        obtenus = _mois_de(train)
        raise AssertionError(f"decouper(lots, '2026-01', historique=12) : train doit couvrir les 12 mois de 2025-01 à "
                             f"2025-12, obtenu {len(obtenus)} mois ({obtenus[0] if obtenus else '—'} → "
                             f"{obtenus[-1] if obtenus else '—'}). Le premier mois gardé est `Period(mois_test) - "
                             f"historique`, inclus.")
    if sorted(train[IDENTIFIANT]) != sorted(lots.loc[lots[MOIS].isin(attendus), IDENTIFIANT]):
        raise AssertionError("decouper : train n'a pas exactement les lignes des 12 mois d'historique (lignes perdues ou "
                             "en double).")
    if list(train.columns) != list(lots.columns) or list(test.columns) != list(lots.columns):
        raise AssertionError("decouper doit garder toutes les colonnes du lot (mois, customer_id, variables, churn).")

    # À cheval sur deux années, et la valeur par défaut.
    train, test = _decouper(f, lots, "2026-02", 3, cas="historique de 3 mois")
    if _mois_de(train) != ["2025-11", "2025-12", "2026-01"]:
        raise AssertionError(f"decouper(lots, '2026-02', historique=3) : train doit couvrir 2025-11, 2025-12 et 2026-01, "
                             f"obtenu {_mois_de(train)}. Comptez en mois avec `pd.Period('2026-02', freq='M') - 3`, "
                             f"qui passe correctement d'une année à l'autre.")
    train, _ = _appel(lambda: f(lots, "2026-04"), "decouper(lots, '2026-04')  (historique par défaut)")
    if len(_mois_de(train)) != m.HISTORIQUE:
        raise AssertionError(f"decouper(lots, '2026-04') sans historique : la valeur par défaut est HISTORIQUE "
                             f"({m.HISTORIQUE} mois), obtenu {len(_mois_de(train))} mois.")
    train, _ = _decouper(f, lots, "2025-04", 12, cas="moins d'historique que demandé")
    if _mois_de(train) != ["2025-01", "2025-02", "2025-03"]:
        raise AssertionError(f"decouper(lots, '2025-04', 12) : seuls 3 mois existent avant 2025-04 ; train doit les "
                             f"garder tous (2025-01 à 2025-03), obtenu {_mois_de(train)}.")

    # L'appelant peut modifier train sans abîmer les lots.
    avant = lots.copy()
    train, test = _decouper(f, lots, "2025-06", 3, cas="copies")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train[CIBLE] = 9
        test[CIBLE] = 9
    if not lots.equals(avant):
        raise AssertionError("Modifier train ou test a modifié les lots d'origine : renvoyez des copies "
                             "(`lots[masque].copy()`).")

    # Les refus.
    _leve_value_error(lambda: f(lots, "2027-01", 12), "decouper(lots, '2027-01')  (aucune ligne ce mois-là)",
                      "Sans ligne pour le mois de test, il n'y a rien à mesurer.")
    _leve_value_error(lambda: f(lots, "2025-01", 12), "decouper(lots, '2025-01')  (le tout premier mois)",
                      "Aucun mois avant 2025-01 : train serait vide, et on ne peut pas entraîner sur rien.")
    for mauvais in ("2026-1", "2026-13", "01-2026", "2026/01"):
        _leve_value_error(lambda: f(lots, mauvais, 12), f"decouper(lots, {mauvais!r})",
                          "Vérifiez le format AAAA-MM du mois de test avant de vous en servir.")
    for mauvais in (0, -3, True, 1.5):
        _leve_value_error(lambda: f(lots, "2026-01", mauvais), f"decouper(lots, '2026-01', historique={mauvais!r})",
                          "historique est un nombre ENTIER de mois, au moins 1. `type(historique) is not int` refuse "
                          "aussi True et 1.5.")


# ============================================================================ Étape 3 : mesurer


class _FauxModele:
    """Un modèle qui renvoie des probabilités choisies d'avance (une par ligne, selon l'index du test)."""

    def __init__(self, proba: pd.Series, classes=(0, 1)):
        self.proba = proba
        self.classes_ = np.array(classes)
        self.colonnes_vues: list[str] = []

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            raise AssertionError("mesurer passe au modèle autre chose qu'un DataFrame : donnez-lui `test[list(VARIABLES)]`, "
                                 "un DataFrame dont les colonnes portent leur nom.")
        self.colonnes_vues = list(X.columns)
        if CIBLE in X.columns:
            raise AssertionError("mesurer donne au modèle la colonne churn, c'est-à-dire LA RÉPONSE. Un vrai modèle "
                                 "l'ignorerait peut-être, mais d'autres l'utiliseraient : donnez-lui seulement "
                                 "`test[list(VARIABLES)]`.")
        if list(X.columns) != list(VARIABLES):
            raise AssertionError(f"mesurer donne au modèle les colonnes {list(X.columns)} : donnez-lui exactement "
                                 f"`test[list(VARIABLES)]` (ni le mois, ni l'identifiant).")
        p1 = self.proba.loc[X.index].to_numpy(float)
        return np.column_stack([p1, 1 - p1]) if self.classes_[0] == 1 else np.column_stack([1 - p1, p1])

    def predict(self, X):
        raise AssertionError("mesurer appelle predict, qui décide au seuil 0,5 : utilisez predict_proba, puis comparez "
                             "les probabilités au seuil (SEUIL par défaut, ou celui passé en paramètre).")


def _test_mesure(graine: int = 53, n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    test = generer_lots(graine=graine, debut="2026-05", nb_mois=1, clients=n)
    test.index = [f"L{i}" for i in range(100, 100 + n)]  # un index qui ne commence pas à 0
    rng = np.random.default_rng(graine)
    proba = pd.Series(rng.uniform(0, 1, n).round(3), index=test.index)
    proba.iloc[:4] = 0.4  # pile au seuil
    return test, proba


def _comparer_mesures(obtenu, attendu: Mesures, quoi: str, seuil: float) -> None:
    for champ in ("mois", "clients"):
        if getattr(obtenu, champ) != getattr(attendu, champ):
            raise AssertionError(f"{quoi} : {champ} vaut {getattr(obtenu, champ)!r} au lieu de {getattr(attendu, champ)!r}.")
    if _proche(obtenu.auc, 1 - attendu.auc) and not _proche(attendu.auc, 0.5):
        raise AssertionError(f"{quoi} : l'AUC vaut {obtenu.auc:.3f} au lieu de {attendu.auc:.3f}, soit 1 − la bonne "
                             f"valeur : vous lisez la probabilité de RESTER. La colonne de la classe 1 est "
                             f"`list(modele.classes_).index(1)`, pas forcément la deuxième.")
    if not _proche(obtenu.auc, attendu.auc):
        raise AssertionError(f"{quoi} : l'AUC vaut {obtenu.auc!r} au lieu de {attendu.auc:.4f} "
                             f"(`roc_auc_score(y, proba)`).")
    if not _proche(obtenu.taux_alerte, attendu.taux_alerte):
        raise AssertionError(f"{quoi} : taux_alerte vaut {obtenu.taux_alerte!r} au lieu de {attendu.taux_alerte:.4f}. "
                             f"On alerte quand la probabilité est >= {seuil} (les clients pile au seuil sont alertés).")
    if not _proche(obtenu.cout, attendu.cout):
        if _proche(obtenu.cout, attendu.cout * attendu.clients):
            raise AssertionError(f"{quoi} : cout vaut {obtenu.cout:.2f}, le coût TOTAL ; on veut le coût PAR CLIENT "
                                 f"({attendu.cout:.2f} €), pour comparer des mois de tailles différentes.")
        raise AssertionError(f"{quoi} : cout vaut {obtenu.cout!r} au lieu de {attendu.cout:.4f} € par client : "
                             f"(COUT_FN × partants non alertés + COUT_FP × restés alertés) / nombre de clients.")


def verifier_mesurer(module: ModuleType) -> None:
    f = module.mesurer
    m = _modeles(module)
    test, proba = _test_mesure()
    avant = test.copy()

    modele = _FauxModele(proba)
    obtenu = _pas_none(_appel(lambda: f(modele, test), "mesurer(modele, test)"), "mesurer")
    if not isinstance(obtenu, m.Mesures):
        raise AssertionError(f"mesurer doit renvoyer un objet Mesures (modeles.py), pas {type(obtenu).__name__}.")
    if not test.equals(avant):
        raise AssertionError("mesurer a MODIFIÉ le DataFrame de test : ne lui ajoutez pas de colonne.")
    attendu = ref_evaluation.mesurer(_FauxModele(proba), test)
    _comparer_mesures(obtenu, attendu, "mesurer(modele, test)", 0.4)

    inverse = _FauxModele(proba, classes=(1, 0))
    obtenu = _appel(lambda: f(inverse, test), "mesurer(modele dont classes_ vaut [1, 0], test)")
    _comparer_mesures(obtenu, attendu, "mesurer(modele dont classes_ vaut [1, 0], test)", 0.4)

    obtenu = _appel(lambda: f(_FauxModele(proba), test, 0.7), "mesurer(modele, test, seuil=0.7)")
    _comparer_mesures(obtenu, ref_evaluation.mesurer(_FauxModele(proba), test, 0.7), "mesurer(modele, test, seuil=0.7)",
                      0.7)

    # Un vrai modèle, sur un vrai mois.
    lots = generer_lots(graine=54, debut="2026-01", nb_mois=4, clients=80)
    train, vrai_test = ref_decoupage.decouper(lots, "2026-04")
    vrai = ref_entrainer(train)
    obtenu = _appel(lambda: f(vrai, vrai_test), "mesurer(un vrai modèle, test)")
    _comparer_mesures(obtenu, ref_evaluation.mesurer(vrai, vrai_test), "mesurer(un vrai modèle, test)", 0.4)

    # Les refus.
    deux_mois = pd.concat([test, generer_lots(graine=55, debut="2026-06", nb_mois=1, clients=10)])
    proba_deux = pd.concat([proba, pd.Series(0.5, index=deux_mois.index[len(test):])])
    _leve_value_error(lambda: f(_FauxModele(proba_deux), deux_mois), "mesurer(modele, un test sur DEUX mois)",
                      "Le test doit couvrir un seul mois : `test[MOIS].unique()` doit avoir un élément.")
    une_classe = test.assign(**{CIBLE: 0})
    _leve_value_error(lambda: f(_FauxModele(proba), une_classe), "mesurer(modele, un test où personne n'est parti)",
                      "Avec une seule classe, l'AUC n'est pas définie : vérifiez qu'il y a des 0 ET des 1.")


# ============================================================================ Étape 4 : decider

_TOL = 1e-9


def _variante_decider(defaut: str):
    """Une version boguée classique de decider (même contrat, UN défaut)."""

    def decider(challenger, champion, regles=REGLES):
        if defaut == "regles_en_dur":
            regles = REGLES
        if champion is not None and champion.mois != challenger.mois and defaut != "mois_ignore":
            raise ValueError("mois différents")
        tol = 0.0 if defaut == "sans_tolerance" else _TOL
        raisons, ok = [], True
        if challenger.auc < regles.auc_min - tol and defaut != "plancher_oublie" and not (
                defaut == "plancher_sans_champion" and champion is None):
            ok = False
            raisons.append("plancher")
            if defaut == "premiere_raison":
                return Decision(False, tuple(raisons))
        if champion is None:
            return Decision(ok, tuple(raisons) or ("premier modèle",))
        if defaut == "gain_relatif":
            gain = challenger.auc / champion.auc - 1
        else:
            gain = challenger.auc - champion.auc
        minimum = 0.0 if defaut == "gain_min_oublie" else regles.gain_auc_min
        refuse_gain = gain <= minimum if defaut == "gain_min_oublie" else gain < minimum - tol
        if refuse_gain:
            ok = False
            raisons.append("gain")
            if defaut == "premiere_raison":
                return Decision(False, tuple(raisons))
        else:
            raisons.append("gain ok")
        hausse = champion.cout - challenger.cout if defaut == "cout_inverse" else challenger.cout - champion.cout
        tolerance = 0.0 if defaut == "tolerance_cout_oubliee" else regles.hausse_cout_max
        if defaut != "cout_oublie" and hausse > tolerance + tol:
            ok = False
            raisons.append("coût")
        else:
            raisons.append("coût ok")
        return Decision(ok, tuple(raisons))

    return decider


#: (défaut, ce que fait la version boguée, conseil)
_DEFAUTS_DECIDER = (
    ("sans_tolerance", "refuse un gain d'exactement 0,01 (0,82 contre 0,81), parce que 0.82 - 0.81 vaut "
                       "0.009999999999999898 en Python",
     "Comparez avec une petite marge : `gain < regles.gain_auc_min - 1e-9` pour refuser."),
    ("plancher_sans_champion", "promeut le premier modèle même sous le plancher (AUC 0,65 sans champion)",
     "Le plancher `auc_min` s'applique AUSSI quand il n'y a pas encore de champion : un modèle inutilisable ne part "
     "jamais en production."),
    ("plancher_oublie", "promeut un modèle sous le plancher (AUC 0,69 contre un champion à 0,60)",
     "Vérifiez `challenger.auc >= regles.auc_min` dans tous les cas, même quand le challenger bat le champion."),
    ("gain_relatif", "calcule un gain RELATIF (0,758 contre 0,75 : +1,07 %, alors que l'écart n'est que de +0,008)",
     "Le gain est l'écart ABSOLU d'AUC : `challenger.auc - champion.auc`."),
    ("gain_min_oublie", "promeut dès que le challenger est un peu meilleur (+0,008), sans exiger gain_auc_min",
     "Un gain minuscule peut venir du hasard du mois de test : exigez `gain >= regles.gain_auc_min`."),
    ("cout_oublie", "ne regarde pas le coût : un challenger plus cher (60 € contre 50 € par client) est promu",
     "Refusez quand `challenger.cout - champion.cout > regles.hausse_cout_max`."),
    ("cout_inverse", "inverse le sens du coût : refuse le challenger MOINS cher et promeut le plus cher",
     "La hausse de coût est `challenger.cout - champion.cout` (positive quand le challenger coûte plus)."),
    ("tolerance_cout_oubliee", "refuse toute hausse de coût, même permise par `hausse_cout_max` (+4 € avec une "
                               "tolérance de 5 €)",
     "Comparez la hausse à `regles.hausse_cout_max`, pas à 0."),
    ("regles_en_dur", "ignore les règles reçues en paramètre et applique toujours REGLES",
     "Lisez les seuils dans le paramètre `regles` : `regles.auc_min`, `regles.gain_auc_min`, `regles.hausse_cout_max`."),
    ("premiere_raison", "s'arrête à la première règle enfreinte (une seule raison quand trois règles refusent)",
     "Examinez TOUTES les règles et ajoutez une raison pour chacune : l'historique doit dire tout ce qui n'allait pas."),
    ("mois_ignore", "compare un champion et un challenger mesurés sur deux mois différents",
     "Levez ValueError quand `champion.mois != challenger.mois` : deux examens différents ne se comparent pas."),
)

_MUTANTS_DECIDER = tuple((_variante_decider(d), quoi, conseil) for d, quoi, conseil in _DEFAUTS_DECIDER)


def _scenarios(m: ModuleType) -> list[tuple[str, tuple]]:
    def mes(auc, cout=50.0, mois="2026-09"):
        return m.Mesures(mois=mois, clients=500, auc=auc, cout=cout, taux_alerte=0.3)

    souples = m.Regles(auc_min=0.70, gain_auc_min=0.0, hausse_cout_max=5.0)
    strictes = m.Regles(auc_min=0.90, gain_auc_min=0.02, hausse_cout_max=0.0)
    return [
        ("challenger 0,75, pas de champion", (mes(0.75), None)),
        ("challenger 0,65, pas de champion", (mes(0.65), None)),
        ("challenger 0,69, champion 0,60", (mes(0.69), mes(0.60))),
        ("challenger 0,80, champion 0,75", (mes(0.80), mes(0.75))),
        ("challenger 0,758, champion 0,75", (mes(0.758), mes(0.75))),
        ("challenger 0,82, champion 0,81 (gain d'exactement 0,01)", (mes(0.82), mes(0.81))),
        ("challenger 0,85 à 60 €, champion 0,80 à 50 €", (mes(0.85, 60.0), mes(0.80, 50.0))),
        ("challenger 0,85 à 40 €, champion 0,80 à 50 €", (mes(0.85, 40.0), mes(0.80, 50.0))),
        ("challenger 0,80 à 54 €, champion 0,80 à 50 €, règles souples (gain 0, tolérance 5 €)",
         (mes(0.80, 54.0), mes(0.80, 50.0), souples)),
        ("challenger 0,85, pas de champion, plancher à 0,90", (mes(0.85), None, strictes)),
        ("challenger 0,95, champion 0,94, gain exigé 0,02", (mes(0.95), mes(0.94), strictes)),
        ("challenger 0,65 à 60 €, champion 0,66 à 50 €", (mes(0.65, 60.0), mes(0.66, 50.0))),
        ("challenger sur 2026-09, champion sur 2026-08", (mes(0.80), mes(0.75, mois="2026-08"))),
    ]


def _issue(f, args) -> object:
    """Ce que décide f : True / False, "ValueError", ou le nom d'une autre exception."""
    try:
        d = f(*args)
    except (NotImplementedError, AssertionError):
        raise
    except ValueError:
        return "ValueError"
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__
    return getattr(d, "promouvoir", d)


def _issue_mutant(f, args) -> object:
    try:
        return _issue(f, args)
    except AssertionError:  # pragma: no cover — les variantes ne lèvent pas AssertionError
        return "AssertionError"


def _dire(issue) -> str:
    return {True: "promouvoir", False: "refuser"}.get(issue, f"lever {issue}")


def verifier_decider(module: ModuleType) -> None:
    f = module.decider
    m = _modeles(module)
    scenarios = _scenarios(m)

    attendues = [_issue(ref_decision.decider, args) for _, args in scenarios]
    obtenues = [_issue(f, args) for _, args in scenarios]
    if obtenues != attendues:
        for mutant, quoi, conseil in _MUTANTS_DECIDER:
            if [_issue_mutant(mutant, args) for _, args in scenarios] == obtenues:
                raise AssertionError(f"Votre decider {quoi}. {conseil}")
        i = next(i for i, (o, a) in enumerate(zip(obtenues, attendues)) if o != a)
        raise AssertionError(f"{scenarios[i][0]} : decider devrait {_dire(attendues[i])}, mais choisit de "
                             f"{_dire(obtenues[i])}. Relisez les règles dans la docstring, dans l'ordre.")

    for (quoi, args), attendue in zip(scenarios, attendues):
        if attendue == "ValueError":
            continue
        d = f(*args)
        if not isinstance(d, m.Decision):
            raise AssertionError(f"decider doit renvoyer un objet Decision (modeles.py), pas {type(d).__name__}.")
        if not isinstance(d.promouvoir, bool):
            raise AssertionError(f"Decision.promouvoir doit être True ou False, pas {d.promouvoir!r}.")
        raisons = d.raisons
        if not isinstance(raisons, tuple) or not raisons or not all(isinstance(r, str) and r.strip() for r in raisons):
            raise AssertionError(f"{quoi} : Decision.raisons vaut {raisons!r:.150}. Il faut un TUPLE d'au moins une "
                                 f"phrase, même quand le modèle est promu (« pourquoi ? » est la première question "
                                 f"qu'on pose devant l'historique). Rassemblez une liste, puis `tuple(raisons)`.")
    trois = f(*scenarios[11][1])
    if len(trois.raisons) < 3:
        raise AssertionError(f"challenger 0,65 à 60 €, champion 0,66 à 50 € : trois règles refusent (plancher, gain et "
                             f"coût), mais raisons n'en donne que {len(trois.raisons)} ({trois.raisons}). Examinez "
                             f"toutes les règles, une raison par règle.")


# ============================================================================ Étape 5 : le registre

_DECLENCHE: list[str] = []


def _declencher() -> str:
    """Appelée si quelqu'un charge le fichier piégé : c'est exactement ce que le registre doit empêcher."""
    _DECLENCHE.append("chargé !")
    return "piège"


class _Piege:
    def __reduce__(self):
        return (_declencher, ())


class _DisquePlein:
    """Un objet qu'on ne peut pas enregistrer : pickle échoue APRÈS avoir écrit un gros tableau."""

    def __reduce__(self):
        raise OSError(28, "No space left on device (disque plein simulé)")


def _mesures_exemple(m: ModuleType, auc: float = 0.8, mois: str = "2026-03") -> Mesures:
    return m.Mesures(mois=mois, clients=300, auc=auc, cout=50.0, taux_alerte=0.3)


def _index(dossier: Path) -> dict:
    return json.loads((dossier / "registre.json").read_text(encoding="utf-8"))


def _fichiers(dossier: Path) -> set[str]:
    return {p.name for p in dossier.iterdir()}


def verifier_registre(module: ModuleType) -> None:
    m = _modeles(module)
    enregistrer, charger, promouvoir, revenir = module.enregistrer, module.charger, module.promouvoir, module.revenir

    with _dossier_temporaire() as racine:
        dossier = racine / "registre" / "modeles"  # n'existe pas encore
        modele_1 = {"poids": [0.5, -1.2, 3.0], "nom": "modèle 1"}
        v1 = _pas_none(_appel(lambda: enregistrer(dossier, modele_1, _mesures_exemple(m, 0.81)),
                              "enregistrer(dossier qui n'existe pas encore, modele, mesures)"), "enregistrer")
        if not isinstance(v1, m.Version):
            raise AssertionError(f"enregistrer doit renvoyer un objet Version (modeles.py), pas {type(v1).__name__}.")
        if (v1.numero, v1.fichier) != (1, "v001.joblib"):
            raise AssertionError(f"La première version doit être le numéro 1, fichier « v001.joblib » (nom_fichier), "
                                 f"obtenu {v1.numero!r} et {v1.fichier!r}.")
        chemin = dossier / v1.fichier
        if not chemin.exists():
            raise AssertionError(f"enregistrer n'a pas écrit {chemin} : le modèle doit être dans le dossier du registre.")
        if v1.sha256 != hashlib.sha256(chemin.read_bytes()).hexdigest():
            raise AssertionError("Version.sha256 n'est pas l'empreinte SHA-256 du fichier écrit : calculez-la sur les "
                                 "octets exacts du fichier (`hashlib.sha256(contenu).hexdigest()`).")
        if not (dossier / m.INDEX).exists():
            raise AssertionError("enregistrer n'a pas écrit l'index (registre.json) : utilisez `ecrire_index`.")
        index = _index(dossier)
        if [v["numero"] for v in index.get("versions", [])] != [1] or index["versions"][0]["sha256"] != v1.sha256:
            raise AssertionError(f"Après un enregistrement, l'index doit contenir la version 1 et son empreinte, obtenu "
                                 f"{index!r:.200}. Ajoutez `version.vers_dict()` à index['versions'].")
        if index["versions"][0]["mesures"]["auc"] != 0.81:
            raise AssertionError("L'index ne garde pas les mesures de la version : utilisez `version.vers_dict()`.")
        if index.get("champions"):
            raise AssertionError("enregistrer a promu la version : enregistrer n'est PAS promouvoir. Un modèle refusé "
                                 "est enregistré quand même, pour garder la trace de ce qui a été essayé.")

        v2 = _appel(lambda: enregistrer(dossier, {"nom": "modèle 2"}, _mesures_exemple(m, 0.83)),
                    "enregistrer (deuxième version)")
        if v2.numero != 2 or not (dossier / "v002.joblib").exists():
            raise AssertionError(f"La deuxième version doit porter le numéro 2 (v002.joblib), obtenu {v2.numero!r}.")

        # Charger : le même objet, et l'empreinte vérifiée.
        recharge = _appel(lambda: charger(dossier, 1), "charger(dossier, 1)")
        if recharge != modele_1:
            raise AssertionError(f"charger(dossier, 1) renvoie {recharge!r:.100} au lieu du modèle enregistré.")
        _leve_value_error(lambda: charger(dossier, 7), "charger(dossier, 7)  (version inconnue)",
                          "Cherchez la version dans l'index ; si elle n'y est pas, levez ValueError.")
        contenu = chemin.read_bytes()
        chemin.write_bytes(contenu[:-1] + bytes([contenu[-1] ^ 1]))  # un seul bit changé
        try:
            charger(dossier, 1)
        except m.ModeleAltere:
            pass
        except (NotImplementedError, AssertionError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"v001.joblib a été modifié (un bit) : charger doit lever ModeleAltere, pas "
                                 f"{type(exc).__name__} ({exc}). Comparez l'empreinte des octets lus à celle de "
                                 f"l'index AVANT de charger.") from None
        else:
            raise AssertionError("v001.joblib a été modifié (un bit), et charger l'a chargé quand même : comparez "
                                 "`hashlib.sha256(contenu).hexdigest()` à l'empreinte de l'index, et levez "
                                 "ModeleAltere si elles diffèrent.")
        _DECLENCHE.clear()
        chemin.write_bytes(pickle.dumps(_Piege()))
        try:
            charger(dossier, 1)
        except m.ModeleAltere:
            pass
        except (NotImplementedError, AssertionError):
            raise
        except Exception:  # noqa: BLE001
            pass
        if _DECLENCHE:
            raise AssertionError("v001.joblib a été remplacé par un fichier piégé, et charger l'a OUVERT avec joblib.load "
                                 "avant de refuser : un fichier pickle exécute du code dès son chargement (série 13, défi "
                                 "84). Vérifiez l'empreinte d'ABORD, puis chargez ces mêmes octets "
                                 "(`joblib.load(io.BytesIO(contenu))`).")
        chemin.write_bytes(contenu)

        # Disque plein au milieu de l'écriture : ni fichier tronqué, ni entrée dans l'index.
        avant_fichiers, avant_index = _fichiers(dossier), _index(dossier)
        gros = [np.arange(200_000, dtype=float), _DisquePlein()]
        try:
            enregistrer(dossier, gros, _mesures_exemple(m, 0.9))
        except (NotImplementedError, AssertionError):
            raise
        except Exception:  # noqa: BLE001 — l'erreur doit remonter : c'est attendu
            pass
        else:
            raise AssertionError("Le disque est plein au milieu de l'enregistrement, et enregistrer ne lève aucune "
                                 "erreur : ne l'attrapez pas, l'appelant doit savoir que rien n'a été enregistré.")
        restes = sorted(_fichiers(dossier) - avant_fichiers)
        if restes:
            raise AssertionError(f"Le disque est plein au milieu de l'enregistrement, et il reste {restes} dans le "
                                 f"registre. Un fichier de modèle tronqué ne doit jamais exister sous son vrai nom : "
                                 f"écrivez les octets dans un fichier temporaire du même dossier, puis `os.replace` ; en "
                                 f"cas d'erreur, effacez le temporaire.")
        if _index(dossier) != avant_index:
            raise AssertionError("Le disque est plein au milieu de l'enregistrement, et l'index a quand même changé : "
                                 "écrivez l'index EN DERNIER, une fois le fichier du modèle en place.")

        # Numéroter avec max + 1 : une version retirée à la main ne doit pas faire réutiliser un numéro.
        index = _index(dossier)
        index["versions"] = [v for v in index["versions"] if v["numero"] != 1]
        (dossier / "registre.json").write_text(json.dumps(index), encoding="utf-8")
        v3 = _appel(lambda: enregistrer(dossier, {"nom": "modèle 3"}, _mesures_exemple(m)), "enregistrer")
        if v3.numero != 3:
            raise AssertionError(f"La version 1 a été retirée de l'index ; il reste la version 2. La suivante doit être la "
                                 f"3 (le plus grand numéro + 1), obtenu {v3.numero}. Avec `len + 1`, on réutiliserait le "
                                 f"numéro 2 et on écraserait v002.joblib.")

    # Promouvoir et revenir.
    with _dossier_temporaire() as dossier:
        for i in range(1, 4):
            ref_registre.enregistrer(dossier, {"i": i}, _mesures_exemple(m, 0.8 + i / 100))
        _leve_value_error(lambda: promouvoir(dossier, 9), "promouvoir(dossier, 9)  (version inconnue)",
                          "On ne peut promouvoir qu'une version enregistrée.")
        _leve_value_error(lambda: revenir(dossier), "revenir(dossier)  (aucun champion)",
                          "Sans champion précédent, il n'y a pas d'arrière où revenir.")
        _appel(lambda: promouvoir(dossier, 1), "promouvoir(dossier, 1)")
        if _index(dossier)["champions"] != [1]:
            raise AssertionError(f"Après promouvoir(dossier, 1), champions doit valoir [1], obtenu "
                                 f"{_index(dossier)['champions']}.")
        _leve_value_error(lambda: revenir(dossier), "revenir(dossier)  (un seul champion jusqu'ici)",
                          "Avec un seul champion dans l'historique, il n'y a pas de précédent : levez ValueError, et "
                          "ne touchez pas à l'index.")
        if _index(dossier)["champions"] != [1]:
            raise AssertionError("revenir a échoué (aucun champion précédent), mais l'index a changé : vérifiez AVANT de "
                                 "modifier.")
        promouvoir(dossier, 1)
        if _index(dossier)["champions"] != [1]:
            raise AssertionError(f"Promouvoir deux fois la version 1 donne champions = {_index(dossier)['champions']} : "
                                 f"si elle est déjà championne, ne changez rien (sinon `revenir` reviendrait… à elle-même).")
        promouvoir(dossier, 3)
        if _index(dossier)["champions"] != [1, 3] or ref_fichiers.champion(dossier).numero != 3:
            raise AssertionError(f"Après promouvoir 1 puis 3, champions doit valoir [1, 3] (l'historique des promotions), "
                                 f"obtenu {_index(dossier)['champions']}.")
        retour = _pas_none(_appel(lambda: revenir(dossier), "revenir(dossier)"), "revenir")
        if not isinstance(retour, m.Version) or retour.numero != 1:
            raise AssertionError(f"revenir doit renvoyer la Version redevenue championne (la 1), obtenu {retour!r:.120}.")
        if _index(dossier)["champions"] != [1] or len(_index(dossier)["versions"]) != 3:
            raise AssertionError(f"Après revenir, champions doit valoir [1] et les 3 versions doivent rester "
                                 f"enregistrées, obtenu {_index(dossier)!r:.200}. Revenir en arrière ne supprime aucun "
                                 f"modèle.")


# ============================================================================ Étape 6 : le programme


def _ecrire_lots(dossier: Path, lots: pd.DataFrame, nom: str = "lots.csv") -> str:
    chemin = dossier / nom
    lots.to_csv(chemin, index=False, encoding="utf-8")
    return str(chemin)


def _commande(argv: list[str]) -> str:
    return "python -m projets.reentrainement " + " ".join(argv)


def _lancer_sans_exception(main, argv: list[str]) -> tuple[int, str, str]:
    try:
        return _lancer(main, argv)
    except (NotImplementedError, AssertionError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"`{_commande(argv)}` plante avec {type(exc).__name__}: {exc}. Attrapez les erreurs "
                             f"attendues (ValueError, ModeleAltere, OSError) dans main, affichez une phrase sur "
                             f"stderr et renvoyez 1.") from None


def _attendu_entrainer(lots: pd.DataFrame, mois: str, registre_ref: Path) -> tuple[bool, int]:
    """Ce que doit donner `entrainer` : (promu ?, numéro du champion après), calculé avec le corrigé."""
    train, test = ref_decoupage.decouper(lots, mois)
    modele = ref_entrainer(train)
    mesures = ref_evaluation.mesurer(modele, test)
    actuel = ref_fichiers.champion(registre_ref)
    champion = ref_evaluation.mesurer(ref_registre.charger(registre_ref, actuel.numero), test) if actuel else None
    decision = ref_decision.decider(mesures, champion)
    version = ref_registre.enregistrer(registre_ref, modele, mesures)
    if decision.promouvoir:
        ref_registre.promouvoir(registre_ref, version.numero)
    return decision.promouvoir, ref_fichiers.champion(registre_ref).numero


def verifier_cli(module: ModuleType) -> None:
    main = module.main
    lots = generer_lots(graine=2026, debut="2025-10", nb_mois=12, clients=150)
    scenario = (("2026-03", "le premier modèle"), ("2026-04", "un mois plus tard, sans dérive"),
                ("2026-09", "après la dérive (un concurrent attire la fibre)"))

    with _dossier_temporaire() as dossier:
        chemin_lots = _ecrire_lots(dossier, lots)
        registre, registre_ref = dossier / "registre", dossier / "reference"

        argv = ["historique", "--registre", str(registre)]
        code, sortie, erreurs = _lancer_sans_exception(main, argv)
        if code != 0:
            raise AssertionError(f"`{_commande(argv)}` sur un registre vide renvoie {code} : dites simplement que le "
                                 f"registre est vide, et renvoyez 0. {erreurs.strip()[:150]}")

        for numero, (mois, quoi) in enumerate(scenario, start=1):
            argv = ["entrainer", "--lots", chemin_lots, "--mois", mois, "--registre", str(registre)]
            code, sortie, erreurs = _lancer_sans_exception(main, argv)
            promu, champion_attendu = _attendu_entrainer(lots, mois, registre_ref)
            if code != 0:
                indice = ""
                if "mois" in erreurs and numero > 1:
                    indice = (" Le champion a-t-il été RE-MESURÉ sur le mois de test ? Ses mesures enregistrées datent "
                              "de son propre mois : decider refuse (à raison) de comparer deux mois différents. "
                              "Chargez-le (`charger`) et appelez `mesurer(champion, test)`.")
                raise AssertionError(f"`{_commande(argv)}` ({quoi}) renvoie {code} au lieu de 0. "
                                     f"{erreurs.strip()[:200]}{indice}")
            if not (registre / "registre.json").exists():
                raise AssertionError(f"`{_commande(argv)}` n'a rien écrit dans le registre : enregistrez le challenger.")
            index = _index(registre)
            if len(index["versions"]) != numero:
                raise AssertionError(f"Après {numero} entraînement(s), le registre doit contenir {numero} version(s), "
                                     f"obtenu {len(index['versions'])}. Enregistrez CHAQUE challenger, promu ou non : "
                                     f"l'historique garde la trace de ce qui a été refusé et pourquoi.")
            if not index["champions"] or index["champions"][-1] != champion_attendu:
                actuel = index["champions"][-1] if index["champions"] else None
                raise AssertionError(
                    f"`{_commande(argv)}` ({quoi}) : le champion devrait être la version {champion_attendu} "
                    f"(challenger {'promu' if promu else 'refusé'}), obtenu {actuel}. Mesurez le challenger ET le "
                    f"champion sur le même mois de test, puis suivez `decider` : promouvoir seulement s'il le dit.")
            if f"v{numero:03d}" not in sortie:
                raise AssertionError(f"`{_commande(argv)}` : la sortie doit nommer la version enregistrée (« v{numero:03d} »), "
                                     f"obtenu :\n{sortie[-300:]}")

        # historique
        argv = ["historique", "--registre", str(registre)]
        code, sortie, _ = _lancer_sans_exception(main, argv)
        if code != 0 or not all(f"v{i:03d}" in sortie for i in (1, 2, 3)):
            raise AssertionError(f"`{_commande(argv)}` doit renvoyer 0 et lister v001, v002 et v003, obtenu "
                                 f"(code {code}) :\n{sortie[-300:]}")
        ligne_champion = next((l for l in sortie.splitlines() if "champion" in l.lower()), "")
        if f"v{_index(registre)['champions'][-1]:03d}" not in ligne_champion:
            raise AssertionError(f"`{_commande(argv)}` doit signaler le champion actuel sur sa ligne (le mot « champion » "
                                 f"et son numéro), obtenu :\n{sortie[-300:]}")

        # revenir
        champions = _index(registre)["champions"]  # [1, 3] : v002 a été refusée
        argv = ["revenir", "--registre", str(registre)]
        if len(champions) >= 2:
            code, sortie, erreurs = _lancer_sans_exception(main, argv)
            if code != 0 or _index(registre)["champions"] != champions[:-1]:
                raise AssertionError(f"`{_commande(argv)}` doit remettre en production le champion précédent "
                                     f"(v{champions[-2]:03d}) et renvoyer 0, obtenu le code {code}. {erreurs.strip()[:150]}")
            if f"v{champions[-2]:03d}" not in sortie:
                raise AssertionError(f"`{_commande(argv)}` doit dire quelle version est revenue en production "
                                     f"(v{champions[-2]:03d}), obtenu : {sortie.strip()[:200]}")
        code, sortie, erreurs = _lancer_sans_exception(main, argv)
        if code != 1 or not erreurs.strip():
            raise AssertionError(f"`{_commande(argv)}` sans champion précédent doit afficher une phrase sur stderr et "
                                 f"renvoyer 1, obtenu le code {code} (stderr : {erreurs.strip()[:150]!r}).")

        # Un lot abîmé : rien n'est entraîné ni écrit.
        abime = _avec(lots, contract={3: "Mensual"})
        chemin_abime = _ecrire_lots(dossier, abime, "abime.csv")
        avant = (_fichiers(registre), _index(registre))
        argv = ["entrainer", "--lots", chemin_abime, "--mois", "2026-09", "--registre", str(registre)]
        code, sortie, erreurs = _lancer_sans_exception(main, argv)
        if code != 1:
            raise AssertionError(f"`{_commande(argv)}` sur un lot qui contient « Mensual » doit refuser le lot et renvoyer "
                                 f"1, obtenu {code}. Appelez `problemes(lots)` AVANT tout entraînement.")
        if "contract" not in sortie + erreurs:
            raise AssertionError(f"`{_commande(argv)}` refuse le lot sans dire pourquoi : affichez chaque problème "
                                 f"(ici, la colonne contract), de préférence sur stderr.")
        if (_fichiers(registre), _index(registre)) != avant:
            raise AssertionError(f"`{_commande(argv)}` a refusé le lot, mais a quand même modifié le registre : rien ne "
                                 f"doit être entraîné ni écrit avec un lot abîmé.")

        # Un mois absent : une phrase, pas une traceback.
        argv = ["entrainer", "--lots", chemin_lots, "--mois", "2030-01", "--registre", str(registre)]
        code, _, erreurs = _lancer_sans_exception(main, argv)
        if code != 1 or "2030-01" not in erreurs:
            raise AssertionError(f"`{_commande(argv)}` (un mois absent des lots) doit afficher sur stderr une phrase qui "
                                 f"cite 2030-01 et renvoyer 1, obtenu le code {code} (stderr : {erreurs.strip()[:150]!r}).")
        if (_fichiers(registre), _index(registre)) != avant:
            raise AssertionError(f"`{_commande(argv)}` a échoué mais a modifié le registre.")


# ============================================================================ Étape 7 : tester decider


def verifier_tester_decider(module: ModuleType) -> None:
    tester = module.tester_decider
    try:
        tester(ref_decision.decider)
    except NotImplementedError:
        raise
    except AssertionError as exc:
        raise AssertionError(f"Votre test ÉCHOUE sur la version JUSTE de decider ({str(exc) or 'assert sans message'}). "
                             f"Relisez les règles de l'étape 4 et recalculez vos cas à la main.") from None
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"Votre test plante sur la version JUSTE : {type(exc).__name__}: {exc}. Fabriquez vos "
                             f"mesures avec `_mesures(auc, cout, mois)` et vos règles avec `Regles(...)`.") from None

    survivants = []
    for mutant, quoi, conseil in _MUTANTS_DECIDER:
        try:
            tester(mutant)
        except Exception:  # noqa: BLE001 — n'importe quelle erreur : le test a vu le bug
            continue
        survivants.append((quoi, conseil))
    if survivants:
        if len(survivants) == len(_MUTANTS_DECIDER):
            raise AssertionError("Votre test passe sur TOUTES les versions boguées : il ne vérifie rien. Écrivez des "
                                 "`assert decider(...).promouvoir` et `assert not decider(...).promouvoir` sur des cas "
                                 "choisis.")
        quoi, conseil = survivants[0]
        raise AssertionError(f"{len(_MUTANTS_DECIDER) - len(survivants)}/{len(_MUTANTS_DECIDER)} bugs attrapés. Votre "
                             f"test laisse passer une version qui {quoi}. Ajoutez un cas où cette version et la bonne "
                             f"donnent des réponses différentes.")


# ============================================================================ Le projet

PROJET = Projet(
    numero=5,
    code="reentrainement",
    titre="Réentraîner le modèle sans casser la production",
    paquet_eleve="projets.reentrainement",
    paquet_corrige="projets.reentrainement_corrige",
    commande="python -m projets.reentrainement entrainer --lots projets/exemple_lots.csv --mois 2026-03 "
             "--registre registre_modeles",
    etapes=(
        Etape(1, "Refuser un lot abîmé", "🟢", "lots", ("problemes",), verifier_problemes),
        Etape(2, "Couper dans le temps", "🟡", "decoupage", ("decouper",), verifier_decouper),
        Etape(3, "Mesurer un modèle sur un mois", "🟡", "evaluation", ("mesurer",), verifier_mesurer),
        Etape(4, "Décider : champion ou challenger ?", "🔴", "decision", ("decider",), verifier_decider),
        Etape(5, "Le registre des modèles", "🔴", "registre", ("enregistrer", "charger", "promouvoir", "revenir"),
              verifier_registre),
        Etape(6, "Le programme à trois commandes", "🟡", "cli", ("main",), verifier_cli),
        Etape(7, "Écrire le test qui protège decider", "🔴", "mes_tests", ("tester_decider",), verifier_tester_decider),
    ),
)

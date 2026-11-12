"""Vérifications automatiques de la série 04 (modéliser et industrialiser).

Lire ces tests est aussi un bon exercice : ils montrent les cas limites auxquels penser.
"""

from __future__ import annotations

import copy
import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from .checks_serie_01 import _proche
from .checks_serie_02 import _dataframe
from .checks_serie_03 import _colonnes, _pipelines_entraines
from .verif import Defi, Serie


def _jamais_none(obtenu, quoi: str):
    assert obtenu is not None, f"{quoi} : la fonction a retourné None. Avez-vous oublié `return` ?"
    return obtenu


# ======================================================================= Module 3 — Modéliser


def _poids(f, y, quoi: str) -> dict:
    poids = _jamais_none(f(y), quoi)
    assert isinstance(poids, dict), f"{quoi} : attendu un dict {{classe: poids}}, obtenu {type(poids).__name__}"
    return poids


def verifier_poids_classes(f) -> None:
    poids = _poids(f, [0, 0, 0, 1], "poids_classes([0, 0, 0, 1])")
    assert set(poids) == {0, 1}, f"Les clés doivent être les classes 0 et 1, obtenu {sorted(poids)}"
    w0, w1 = float(poids[0]), float(poids[1])
    if w0 >= w1:
        raise AssertionError(
            f"Poids obtenus {{0: {w0:.3f}, 1: {w1:.3f}}} : la classe MAJORITAIRE (0, 3 exemples sur 4) pèse "
            "plus que la rare. C'est l'inverse : le poids est inversement proportionnel à l'effectif, n / (k × effectif)."
        )
    if math.isclose(w0, 4 / 3) and math.isclose(w1, 4.0):
        raise AssertionError(
            "Obtenu {0: 1.333, 1: 4.0} au lieu de {0: 0.667, 1: 2.0} : le rapport entre les classes est bon, mais "
            "divisez aussi par le nombre de classes k. Avec la bonne formule, Σ poids × effectif vaut n (ici 4), "
            "comme si chaque exemple pesait 1 en moyenne."
        )
    _proche(w0, 4 / (2 * 3), "poids de la classe 0 pour y=[0, 0, 0, 1] (4 / (2 × 3))")
    _proche(w1, 4 / (2 * 1), "poids de la classe 1 pour y=[0, 0, 0, 1] (4 / (2 × 1))")

    poids = _poids(f, pd.Series([1, 0, 0, 0], index=[10, 11, 12, 13]), "poids_classes sur une pd.Series")
    _proche(poids[1], 2.0, "poids de la classe 1 quand y est une pd.Series dont l'index ne commence pas à 0")

    trois = _poids(f, np.array([0, 0, 0, 1, 1, 2]), "poids_classes avec 3 classes")
    assert set(trois) == {0, 1, 2}, f"3 classes dans y : attendu les clés [0, 1, 2], obtenu {sorted(trois)}"
    if math.isclose(float(trois[2]), 3.0):
        raise AssertionError(
            "Avec 3 classes, le poids de la classe 2 vaut 6 / (3 × 1) = 2, obtenu 3 : k (le nombre de classes) "
            "se lit dans les données, len(np.unique(y)), il ne vaut pas toujours 2."
        )
    for classe, attendu in {0: 6 / 9, 1: 1.0, 2: 2.0}.items():
        _proche(trois[classe], attendu, f"poids de la classe {classe} avec 3 classes")

    from sklearn.utils.class_weight import compute_class_weight

    from churn.data import TARGET, generate_customers

    y = generate_customers(n=2000, seed=11)[TARGET]
    poids = _poids(f, y, "poids_classes sur le vrai jeu de données")
    attendus = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    for classe, attendu in zip((0, 1), attendus):
        _proche(poids[classe], attendu, f"poids de la classe {classe} sur 2 000 clients (comparé à compute_class_weight)")
    total = {c: float(poids[c]) * int((y == c).sum()) for c in (0, 1)}
    _proche(total[0], total[1], "poids total des restés contre poids total des partis (doivent être égaux)")


@lru_cache(maxsize=4)
def _cv_attendue(n: int, seed_donnees: int, valeurs_C: tuple, n_plis: int, seed: int, entrainement: bool = False):
    """AUC par pli pour chaque C, sur les plis d'évaluation (ou d'entraînement si demandé)."""
    from sklearn.model_selection import StratifiedKFold, cross_validate

    from churn.data import FEATURES, TARGET, generate_customers
    from churn.model import build_pipeline

    df = generate_customers(n=n, seed=seed_donnees)
    cv = StratifiedKFold(n_splits=n_plis, shuffle=True, random_state=seed)
    lignes = []
    for C in valeurs_C:
        res = cross_validate(build_pipeline("logistic").set_params(clf__C=C), df[FEATURES], df[TARGET],
                             cv=cv, scoring="roc_auc", return_train_score=entrainement)
        scores = res["train_score" if entrainement else "test_score"]
        lignes.append({"C": float(C), "auc_moyenne": scores.mean(), "auc_ecart_type": scores.std()})
    return pd.DataFrame(lignes)


def _choix(table: pd.DataFrame, n_plis: int, tolerance: str = "erreur", sens: str = "min") -> float:
    meilleur = table.loc[table["auc_moyenne"].idxmax()]
    marge = meilleur["auc_ecart_type"] / (math.sqrt(n_plis) if tolerance == "erreur" else 1)
    ok = table.loc[table["auc_moyenne"] >= meilleur["auc_moyenne"] - marge, "C"]
    return float(ok.min() if sens == "min" else ok.max())


def _appel_choisir_C(f, n, seed_donnees, valeurs_C, n_plis, seed):
    from churn.data import FEATURES, TARGET, generate_customers

    df = generate_customers(n=n, seed=seed_donnees)
    res = _jamais_none(f(df[FEATURES], df[TARGET], list(valeurs_C), n_plis=n_plis, seed=seed), "choisir_C")
    assert isinstance(res, tuple) and len(res) == 2, (
        f"choisir_C doit renvoyer un couple (table, C_choisi), obtenu {type(res).__name__}"
    )
    table, choisi = res
    table = _dataframe(table, "choisir_C : la table (1er élément du couple)")
    _colonnes(table, ["C", "auc_moyenne", "auc_ecart_type"], "choisir_C")
    return table, _jamais_none(choisi, "choisir_C : le C choisi (2e élément du couple)")


def verifier_choisir_C(f) -> None:
    valeurs, n_plis = (0.001, 0.01, 0.1, 1.0, 10.0), 5
    table, choisi = _appel_choisir_C(f, 2000, 11, valeurs, n_plis, 0)
    attendue = _cv_attendue(2000, 11, valeurs, n_plis, 0)
    assert len(table) == len(valeurs), f"{len(valeurs)} valeurs de C : attendu {len(valeurs)} lignes, obtenu {len(table)}"
    assert [float(c) for c in table["C"]] == list(valeurs), (
        f"Une ligne par C dans l'ordre de valeurs_C {list(valeurs)}, obtenu {list(table['C'])}"
    )
    if table["auc_moyenne"].nunique() == 1:
        raise AssertionError(
            "Toutes les valeurs de C donnent la même AUC : C n'est pas appliqué. "
            "build_pipeline(\"logistic\").set_params(clf__C=C) règle l'étape \"clf\" du Pipeline."
        )
    if not np.allclose(table["auc_moyenne"], attendue["auc_moyenne"], atol=1e-9):
        sur_le_train = _cv_attendue(2000, 11, valeurs, n_plis, 0, entrainement=True)
        if np.allclose(table["auc_moyenne"], sur_le_train["auc_moyenne"], atol=1e-6):
            raise AssertionError(
                "Vos AUC sont celles des plis d'ENTRAÎNEMENT : le modèle est noté sur des clients qu'il a déjà vus, "
                "et plus C est grand, plus il a l'air bon. Notez-le sur le pli mis de côté (cross_val_score le fait)."
            )
        raise AssertionError(
            f"AUC moyennes attendues {np.round(attendue['auc_moyenne'], 4).tolist()}, obtenu "
            f"{np.round(table['auc_moyenne'].astype(float), 4).tolist()}. Utilisez-vous les MÊMES plis pour chaque C : "
            "StratifiedKFold(n_splits=n_plis, shuffle=True, random_state=seed), scoring=\"roc_auc\" ?"
        )
    if np.allclose(table["auc_ecart_type"], attendue["auc_ecart_type"] * math.sqrt(n_plis / (n_plis - 1)), atol=1e-9):
        raise AssertionError(
            "Écart-type calculé avec ddof=1 (pd.Series.std) : utilisez np.std (ddof=0), comme au défi 9"
        )
    assert np.allclose(table["auc_ecart_type"], attendue["auc_ecart_type"], atol=1e-9), (
        f"Écarts-types attendus {np.round(attendue['auc_ecart_type'], 4).tolist()}, obtenu "
        f"{np.round(table['auc_ecart_type'].astype(float), 4).tolist()} (np.std des AUC des {n_plis} plis)"
    )

    bon = _choix(attendue, n_plis)
    pieges = {
        float(attendue.loc[attendue["auc_moyenne"].idxmax(), "C"]): (
            "Vous prenez le C au meilleur score brut. Ses quelques millièmes d'avance sont bien plus petits que "
            "l'erreur standard : c'est du bruit. Prenez le PLUS PETIT C dont la moyenne est >= meilleure − "
            "écart-type / √n_plis."
        ),
        _choix(attendue, n_plis, sens="max"): (
            "Vous prenez le plus GRAND C dans la tolérance. Plus C est petit, plus la régularisation est forte : "
            "le modèle le plus simple est celui du plus PETIT C."
        ),
        _choix(attendue, n_plis, tolerance="ecart"): (
            "Vous tolérez un écart-type entier. L'incertitude sur une MOYENNE de n_plis plis est l'erreur standard, "
            "écart-type / √n_plis : sinon on accepte un modèle nettement moins bon."
        ),
    }
    assert len(set(pieges) | {bon}) == 4  # garde-fou du correcteur : chaque erreur donne un C différent
    if float(choisi) in pieges:
        raise AssertionError(f"C choisi : {float(choisi)} au lieu de {bon}. " + pieges[float(choisi)])
    _proche(choisi, bon, "C choisi par la règle de l'erreur standard")

    # Valeurs de C dans le désordre, 3 plis : le « plus petit C » est une valeur, pas une position.
    desordre = (1.0, 0.003, 10.0, 0.03)
    table, choisi = _appel_choisir_C(f, 1000, 7, desordre, 3, 1)
    attendue = _cv_attendue(1000, 7, desordre, 3, 1)
    assert [float(c) for c in table["C"]] == list(desordre), (
        f"Gardez l'ordre de valeurs_C {list(desordre)}, obtenu {list(table['C'])}"
    )
    assert np.allclose(table["auc_moyenne"], attendue["auc_moyenne"], atol=1e-9), (
        "AUC incorrectes avec n_plis=3 et seed=1 : utilisez bien les paramètres n_plis et seed"
    )
    _proche(choisi, _choix(attendue, 3), f"C choisi parmi {list(desordre)} (dans le désordre) avec 3 plis")


# ================================================================ Module 6 — Industrialiser


def verifier_versions_compatibles(f) -> None:
    def attendu(sauvegardee, installee, valeur, explication):
        obtenu = _jamais_none(f(sauvegardee, installee), f"versions_compatibles({sauvegardee!r}, {installee!r})")
        assert bool(obtenu) is valeur, (
            f"versions_compatibles({sauvegardee!r}, {installee!r}) : attendu {valeur}, obtenu {obtenu!r}. {explication}"
        )

    attendu("1.5.0", "1.5.0", True, "Même version.")
    attendu("1.5.2", "1.5.0", True,
            "Seul le 3e nombre (correctif) diffère : il ne change pas le format des modèles. Comparez MAJEUR et MINEUR.")
    attendu("1.4.2", "1.5.0", False, "Le nombre mineur diffère (4 contre 5).")
    attendu("2.5.0", "1.5.0", False, "Le nombre majeur diffère.")
    attendu("1.10.1", "1.1.0", False,
            "\"1.10\"[:3] vaut \"1.1\" et float(\"1.10\") vaut 1.1 : une version n'est pas un nombre décimal. "
            "Découpez sur les points et comparez des entiers.")
    attendu("1.1.0", "1.10.1", False, "1.1 et 1.10 sont deux versions mineures différentes.")
    attendu("1.6", "1.6.1", True, "\"1.6\" (sans correctif) et \"1.6.1\" ont le même majeur et le même mineur.")

    for vide in (None, ""):
        try:
            obtenu = f(vide, "1.5.0")
        except Exception as exc:
            raise AssertionError(
                f"versions_compatibles({vide!r}, \"1.5.0\") plante ({type(exc).__name__}). Un ancien modèle n'a pas de "
                "version notée : renvoyez False (on ne peut rien garantir) au lieu de faire planter le chargement."
            ) from None
        assert obtenu is not None, "versions_compatibles : la fonction a retourné None. Avez-vous oublié `return` ?"
        assert bool(obtenu) is False, f"versions_compatibles({vide!r}, \"1.5.0\") : version inconnue, attendu False"

    import sklearn

    attendu(sklearn.__version__, sklearn.__version__, True, "La version installée est compatible avec elle-même.")


class _CompteurAppels:
    """Enveloppe un pipeline et compte les appels à predict_proba."""

    def __init__(self, pipeline):
        self._pipeline, self.appels = pipeline, 0

    def predict_proba(self, X):
        self.appels += 1
        return self._pipeline.predict_proba(X)

    def __getitem__(self, cle):
        return self._pipeline[cle]

    def __getattr__(self, nom):
        return getattr(self._pipeline, nom)


def _clients_json(n: int = 6, seed: int = 21) -> list[dict]:
    """Des clients comme les enverrait une API : clés dans le désordre, un identifiant en plus."""
    from churn.data import FEATURES, ID_COL, generate_customers

    df = generate_customers(n=n, seed=seed)[[ID_COL, *FEATURES]]
    clients = json.loads(df.to_json(orient="records"))  # des types Python, comme un vrai JSON reçu
    return [dict(reversed(list(c.items()))) if i % 2 else c for i, c in enumerate(clients)]


def verifier_predire_lot(f) -> None:
    from churn.data import FEATURES

    pipeline, _ = _pipelines_entraines(False)
    clients = _clients_json()
    probas = pipeline.predict_proba(pd.DataFrame(clients)[FEATURES])[:, 1]
    seuil = float(probas[2])  # le client n°2 est PILE au seuil
    avant = copy.deepcopy(clients)

    compteur = _CompteurAppels(pipeline)
    res = _jamais_none(f(compteur, clients, seuil=seuil), "predire_lot")
    assert isinstance(res, list), f"predire_lot doit renvoyer une liste, obtenu {type(res).__name__}"
    assert clients == avant, (
        "Les dictionnaires reçus ont été modifiés (clé retirée ou ajoutée ?). Ils appartiennent à l'appelant : "
        "travaillez sur un DataFrame, pd.DataFrame(clients) crée une copie."
    )
    assert len(res) == len(clients), f"{len(clients)} clients : attendu {len(clients)} prédictions, obtenu {len(res)}"
    if compteur.appels > 1:
        raise AssertionError(
            f"predict_proba appelé {compteur.appels} fois pour {len(clients)} clients. Vectorisez : construisez UN "
            "DataFrame avec tous les clients et appelez predict_proba UNE fois (des dizaines de fois plus rapide)."
        )
    assert all(isinstance(r, dict) and set(r) == {"churn_probability", "at_risk"} for r in res), (
        f"Chaque prédiction est un dict avec exactement les clés churn_probability et at_risk, obtenu {res[0]!r}"
    )
    try:
        json.dumps(res)
    except TypeError as exc:
        raise AssertionError(
            f"Le résultat ne passe pas dans json.dumps ({exc}) : une API ne pourrait pas le renvoyer. "
            "Convertissez les valeurs numpy en types Python : float(p) et bool(p >= seuil)."
        ) from None
    assert all(isinstance(r["at_risk"], bool) for r in res), "at_risk doit être un bool Python (True / False)"
    for i, (r, p) in enumerate(zip(res, probas)):
        _proche(r["churn_probability"], p, f"probabilité du client n°{i} (les clients sont-ils dans le même ordre ?)")
    if float(res[2]["churn_probability"]) >= seuil and res[2]["at_risk"] is not True:
        raise AssertionError(
            "Le client n°2 a une probabilité égale au seuil : il est à risque (proba >= seuil, pas >), comme dans l'API."
        )
    assert [r["at_risk"] for r in res] == [float(r["churn_probability"]) >= seuil for r in res], (
        f"at_risk incorrect pour certains clients (seuil {seuil:.4f}) : at_risk = churn_probability >= seuil"
    )

    try:
        vide = f(_CompteurAppels(pipeline), [], seuil=0.5)
    except Exception as exc:
        raise AssertionError(
            f"Une liste vide fait planter predire_lot ({type(exc).__name__}). Renvoyez [] sans appeler le modèle : "
            "scikit-learn refuse un tableau de 0 ligne."
        ) from None
    assert vide == [], f"Une liste vide doit renvoyer [], obtenu {vide!r}"

    for variable, valeur in (("contract", "absente"), ("senior", None)):
        incomplets = _clients_json(n=3, seed=22)
        if valeur == "absente":
            del incomplets[1][variable]
        else:
            incomplets[1][variable] = None
        decrit = f"le client n°1 n'a pas de « {variable} »" + (" (valeur None, un null JSON)" if valeur is None else "")
        try:
            obtenu = f(pipeline, incomplets, seuil=0.5)
        except ValueError as exc:
            assert variable in str(exc), (
                f"Quand {decrit}, la ValueError doit NOMMER la variable manquante pour qu'on corrige vite. "
                f"Message obtenu : {exc!r}"
            )
            continue
        except Exception as exc:
            raise AssertionError(
                f"Quand {decrit}, attendu une ValueError qui nomme « {variable} », obtenu {type(exc).__name__}: {exc}"
            ) from None
        raise AssertionError(
            f"Quand {decrit}, la fonction a renvoyé {obtenu!r} sans erreur : une prédiction calculée sur un client "
            "incomplet, fausse et silencieuse. Vérifiez que chaque variable de FEATURES est présente et non None, "
            "sinon `raise ValueError(...)` en la nommant."
        )


# ===================================================================== Module 7 — Surveillance


def _rapport(psis: dict[str, float]) -> pd.DataFrame:
    from churn.monitoring import status

    rapport = pd.DataFrame({"feature": list(psis), "psi": list(psis.values())})
    rapport["statut"] = rapport["psi"].map(status)
    return rapport


def _alerte(f, rapport: pd.DataFrame, quoi: str, **kwargs) -> tuple[int, list[str]]:
    avant = rapport.copy()
    res = _jamais_none(f(rapport, **kwargs), quoi)
    if not rapport.equals(avant):
        raise AssertionError(
            f"{quoi} : le rapport reçu a été modifié (tri sur place, colonne ajoutée…). Il appartient à l'appelant, "
            "qui peut encore en avoir besoin : sort_values(...) sans inplace renvoie une copie triée."
        )
    assert isinstance(res, tuple) and len(res) == 2, f"{quoi} : attendu un couple (code, variables), obtenu {res!r}"
    code, variables = res
    assert int(code) in (0, 1), f"{quoi} : le code de sortie vaut 0 ou 1, obtenu {code!r}"
    assert isinstance(variables, list), (
        f"{quoi} : variables doit être une liste Python, obtenu {type(variables).__name__} (pensez à .tolist())"
    )
    return int(code), [str(v) for v in variables]


def verifier_alerte_derive(f) -> None:
    rapport = _rapport({                 # volontairement pas trié
        "support_calls": 0.02,
        "contract": 0.25,                # pile au seuil : déjà une dérive
        "monthly_charges": 0.41,
        "tenure_months": 0.12,
        "payment_method": float("nan"),  # PSI non mesuré
        "internet_service": 0.31,
    })
    code, variables = _alerte(f, rapport, "alerte_derive(rapport)")
    attendues = ["monthly_charges", "internet_service", "contract", "payment_method"]
    if "contract" not in variables:
        raise AssertionError(
            "« contract » a un PSI de 0,25 pile : c'est déjà une dérive (psi >= seuil, pas >). "
            "C'est la règle de churn.monitoring.status."
        )
    if "payment_method" not in variables:
        raise AssertionError(
            "« payment_method » a un PSI manquant (NaN) : on n'a pas pu le mesurer, c'est une alerte. "
            "Attention, NaN >= 0.25 vaut False : ajoutez `| psi.isna()` au masque."
        )
    assert sorted(variables) == sorted(attendues), f"Variables en alerte attendues {attendues}, obtenu {variables}"
    assert variables == attendues, (
        f"Ordre attendu {attendues} (de la plus dérivée à la moins dérivée, PSI manquants à la fin), obtenu {variables}. "
        "sort_values(\"psi\", ascending=False, na_position=\"last\")."
    )
    assert code == 1, f"Des variables dérivent : le code de sortie doit valoir 1, obtenu {code}"

    _, plus_strict = _alerte(f, rapport, "alerte_derive(rapport, seuil=0.10)", seuil=0.10)
    if plus_strict == attendues:
        raise AssertionError(
            "Avec seuil=0.10, « tenure_months » (PSI 0,12) devrait aussi être en alerte : le paramètre seuil est "
            "ignoré. N'utilisez pas la colonne statut, calculée pour 0,25 seulement."
        )
    assert plus_strict == ["monthly_charges", "internet_service", "contract", "tenure_months", "payment_method"], (
        f"Avec seuil=0.10 : variables attendues monthly_charges, internet_service, contract, tenure_months, "
        f"payment_method, obtenu {plus_strict}"
    )

    code, variables = _alerte(f, _rapport({"contract": 0.04, "senior": 0.249, "tenure_months": 0.1}),
                              "alerte_derive sur un rapport stable")
    assert variables == [] and code == 0, (
        f"Aucun PSI >= 0,25 ni manquant : attendu (0, []), obtenu ({code}, {variables})"
    )

    from churn.data import generate_customers
    from churn.monitoring import drift_report, simulate_batch

    vrai = drift_report(generate_customers(n=2000, seed=5), simulate_batch(n=2000, seed=6, price_increase=0.2))
    code, variables = _alerte(f, vrai.sample(frac=1, random_state=0), "alerte_derive sur un vrai rapport mélangé")
    attendu = vrai.loc[vrai["psi"] >= 0.25, "feature"].tolist()
    assert "monthly_charges" in attendu  # garde-fou du correcteur : une hausse de 20 % fait bien dériver la facture
    assert (code, variables) == (1, attendu), (
        f"Hausse des prix de 20 % : attendu (1, {attendu}), obtenu ({code}, {variables})"
    )


SERIE = Serie(
    code="04",
    titre="Modéliser et industrialiser",
    module_eleve="defis.serie_04",
    module_corrige="defis.corrige_serie_04",
    defis=(
        Defi(20, "poids_classes", "🟢", "3.6", verifier_poids_classes),
        Defi(21, "choisir_C", "🔴", "3.5", verifier_choisir_C),
        Defi(22, "versions_compatibles", "🟢", "6.2", verifier_versions_compatibles),
        Defi(23, "predire_lot", "🟡", "6.4", verifier_predire_lot),
        Defi(24, "alerte_derive", "🟢", "7.6", verifier_alerte_derive),
    ),
)

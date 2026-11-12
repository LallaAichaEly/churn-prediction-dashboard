"""Vérifications automatiques de la série 05 (prouver l'impact : SQL, incertitude, tests A/B, survie).

Lire ces tests est aussi un bon exercice : ils montrent les cas limites auxquels penser.
"""

from __future__ import annotations

import math
import sqlite3
from statistics import NormalDist

import numpy as np
import pandas as pd

from .checks_serie_01 import _proche
from .checks_serie_02 import _dataframe
from .checks_serie_03 import _colonnes
from .checks_serie_04 import _jamais_none
from .verif import Defi, Serie

# =========================================================================== SQL


def _base(**tables: pd.DataFrame) -> sqlite3.Connection:
    """Une base SQLite en mémoire : rien n'est écrit sur le disque."""
    connexion = sqlite3.connect(":memory:")
    for nom, table in tables.items():
        table.to_sql(nom, connexion, index=False)
    return connexion


def _sql(f, quoi: str) -> str:
    sql = _jamais_none(f(), quoi)
    assert isinstance(sql, str), f"{quoi} doit renvoyer le TEXTE de la requête (une str), obtenu {type(sql).__name__}"
    assert "select" in sql.lower(), f"{quoi} : la requête doit contenir un SELECT, obtenu {sql!r}"
    return sql


def _executer(sql: str, connexion: sqlite3.Connection, quoi: str, params: dict | None = None) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, connexion, params=params)
    except Exception as exc:
        cause = exc.__cause__ or exc  # pandas enveloppe l'erreur SQLite : on garde le message court
        message = str(cause)
        conseil = ""
        if "has no name" in message:
            conseil = " Utilisez le paramètre NOMMÉ :k, pas un « ? » anonyme."
        elif "did not supply a value" in message:
            conseil = " Le seul paramètre fourni s'appelle :k."
        elif "misuse of window function" in message:
            conseil = (" WHERE est évalué AVANT les fonctions de fenêtre : calculez le rang dans une sous-requête "
                       "(WITH classement AS (...)), puis filtrez dessus.")
        elif "no such table" in message:
            conseil = " Les tables s'appellent `clients` (défi 25) et `scores` (défi 26)."
        elif "no such function" in message and "row_number" in message.lower():
            conseil = f" Votre SQLite ({sqlite3.sqlite_version}) est trop ancien : il faut 3.25 ou plus."
        raise AssertionError(f"{quoi} : la requête plante dans SQLite ({type(cause).__name__}: {message}).{conseil}") from None


def verifier_requete_churn_par_contrat(f) -> None:
    from churn.data import generate_customers

    sql = _sql(f, "requete_churn_par_contrat()")
    jouet = generate_customers(n=7, seed=0).assign(
        contract=["A", "B", "A", "C", "A", "B", "C"], churn=[1, 0, 1, 1, 0, 0, 0]
    )
    res = _executer(sql, _base(clients=jouet), "requete_churn_par_contrat")
    _colonnes(res, ["contract", "clients", "taux_churn"],
              "requete_churn_par_contrat (nommez les colonnes calculées avec AS)")
    if len(res) == 1:
        raise AssertionError(
            "Une seule ligne pour 3 contrats : sans GROUP BY contract, COUNT et AVG résument toute la table."
        )
    assert sorted(res["contract"]) == ["A", "B", "C"], f"Une ligne par contrat attendue (A, B, C), obtenu {list(res['contract'])}"
    par_contrat = res.set_index("contract")
    assert [int(par_contrat.loc[c, "clients"]) for c in "ABC"] == [3, 2, 2], (
        f"Nombre de clients par contrat attendu A=3, B=2, C=2, obtenu {par_contrat['clients'].to_dict()} "
        "(COUNT(*) compte les lignes du groupe)"
    )
    taux = par_contrat["taux_churn"].astype(float)
    if math.isclose(taux["A"], 0.0) and math.isclose(taux["C"], 0.0):
        raise AssertionError(
            "Taux de churn de A = 0 alors que 2 clients sur 3 sont partis : c'est la division ENTIÈRE de SQL "
            "(2 / 3 vaut 0 entre deux entiers). Écrivez AVG(churn) ou 1.0 * SUM(churn) / COUNT(*)."
        )
    for contrat, attendu in {"A": 2 / 3, "B": 0.0, "C": 0.5}.items():
        _proche(taux[contrat], attendu, f"taux de churn du contrat {contrat}")
    assert list(res["contract"]) == ["A", "C", "B"], (
        f"Ordre attendu A (67 %), C (50 %), B (0 %), obtenu {list(res['contract'])} : ORDER BY taux_churn DESC"
    )

    vrai = generate_customers(n=2000, seed=3)
    res = _executer(sql, _base(clients=vrai), "requete_churn_par_contrat sur 2 000 clients")
    attendu = (vrai.groupby("contract")["churn"].agg(clients="size", taux_churn="mean")
               .sort_values("taux_churn", ascending=False).reset_index())
    assert list(res["contract"]) == list(attendu["contract"]), (
        f"Sur 2 000 clients : ordre attendu {list(attendu['contract'])}, obtenu {list(res['contract'])}"
    )
    assert np.allclose(res["taux_churn"].astype(float), attendu["taux_churn"]), (
        f"Sur 2 000 clients : taux attendus {np.round(attendu['taux_churn'], 4).tolist()}, "
        f"obtenu {np.round(res['taux_churn'].astype(float), 4).tolist()} (comparé à pandas groupby)"
    )


def _scores_jouet() -> pd.DataFrame:
    # Insérés dans le désordre : l'ordre de la table ne doit jamais décider du résultat.
    lignes = [
        ("C00013", "C", 0.70), ("A00001", "A", 0.80), ("C00012", "C", 0.70), ("B00002", "B", 0.50),
        ("A00004", "A", 0.90), ("C00015", "C", 0.20), ("A00002", "A", 0.95), ("C00011", "C", 0.70),
        ("B00001", "B", 0.30), ("C00014", "C", 0.90), ("A00003", "A", 0.60),
    ]
    return pd.DataFrame(lignes, columns=["customer_id", "contract", "proba_churn"])


def _top_attendu(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    trie = scores.sort_values(["proba_churn", "customer_id"], ascending=[False, True])
    top = trie.groupby("contract").head(k).copy()
    top["rang"] = top.groupby("contract").cumcount() + 1
    return top.sort_values(["contract", "rang"])[["contract", "rang", "customer_id", "proba_churn"]].reset_index(drop=True)


def _comparer_top(res: pd.DataFrame, attendu: pd.DataFrame, quoi: str) -> None:
    assert len(res) == len(attendu), f"{quoi} : {len(attendu)} lignes attendues, obtenu {len(res)}"
    assert list(res["contract"]) == list(attendu["contract"]) and list(res["rang"].astype(int)) == list(attendu["rang"]), (
        f"{quoi} : lignes dans le désordre (contrats {list(res['contract'])}, rangs {list(res['rang'])}). "
        "Terminez par ORDER BY contract, rang."
    )
    assert list(res["customer_id"]) == list(attendu["customer_id"]), (
        f"{quoi} : clients attendus {list(attendu['customer_id'])}, obtenu {list(res['customer_id'])}"
    )
    assert np.allclose(res["proba_churn"].astype(float), attendu["proba_churn"]), f"{quoi} : proba_churn incorrectes"


def verifier_requete_top_risque(f) -> None:
    sql = _sql(f, "requete_top_risque()")
    scores = _scores_jouet()
    connexion = _base(scores=scores)
    res = _executer(sql, connexion, "requete_top_risque avec k=2", params={"k": 2})
    _colonnes(res, ["contract", "rang", "customer_id", "proba_churn"], "requete_top_risque")
    if len(res) == 2:
        raise AssertionError(
            "2 lignes pour k=2 et 3 contrats : un LIMIT s'applique à TOUT le résultat, pas à chaque contrat. "
            "Classez à l'intérieur de chaque contrat : ROW_NUMBER() OVER (PARTITION BY contract ORDER BY ...)."
        )
    par_contrat = res.groupby("contract").size().to_dict()
    if par_contrat.get("C", 0) > 2:
        raise AssertionError(
            f"Le contrat C a {par_contrat['C']} lignes pour k=2 : trois clients sont ex aequo à 0,70, et RANK ou "
            "DENSE_RANK leur donne le même rang. L'équipe ne peut appeler que k clients : ROW_NUMBER() numérote "
            "1, 2, 3… sans ex aequo."
        )
    assert par_contrat == {"A": 2, "B": 2, "C": 2}, (
        f"Lignes par contrat attendues A=2, B=2, C=2 pour k=2, obtenu {par_contrat}"
    )
    attendu = _top_attendu(scores, 2)
    a = res[res["contract"] == "A"]
    if set(a["customer_id"]) == {"A00001", "A00003"}:
        raise AssertionError(
            "Contrat A : vous avez pris les clients les MOINS risqués (0,80 et 0,60). Le rang 1 doit être la plus "
            "forte probabilité : ORDER BY proba_churn DESC dans le OVER (...)."
        )
    assert sorted(res["rang"].astype(int).tolist()) == [1, 1, 1, 2, 2, 2], (
        f"Rangs attendus 1 et 2 dans chaque contrat, obtenu {res['rang'].tolist()}"
    )
    c2 = res[(res["contract"] == "C") & (res["rang"].astype(int) == 2)]["customer_id"].tolist()
    if c2 and c2[0] != "C00011":
        raise AssertionError(
            f"Contrat C, rang 2 : obtenu {c2[0]}, attendu C00011. Trois clients sont ex aequo à 0,70 : sans "
            "départage, le choix dépend de l'ordre de la table et change d'une exécution à l'autre. Ajoutez "
            "customer_id ASC au ORDER BY de la fenêtre."
        )
    _comparer_top(res, attendu, "requete_top_risque avec k=2")

    un = _executer(sql, connexion, "requete_top_risque avec k=1", params={"k": 1})
    trois = _executer(sql, connexion, "requete_top_risque avec k=3", params={"k": 3})
    if len(un) == len(trois):
        raise AssertionError(
            "Même nombre de lignes pour k=1 et k=3 : la valeur de k est écrite en dur dans la requête. "
            "Utilisez le paramètre nommé :k (WHERE rang <= :k)."
        )
    _comparer_top(un, _top_attendu(scores, 1), "requete_top_risque avec k=1")
    _comparer_top(trois, _top_attendu(scores, 3), "requete_top_risque avec k=3 (B n'a que 2 clients : tous les deux)")

    from churn.data import generate_customers

    vrais = generate_customers(n=2000, seed=8)[["customer_id", "contract"]].sample(frac=1, random_state=1)
    vrais["proba_churn"] = np.random.default_rng(2).uniform(size=len(vrais)).round(2)  # beaucoup d'ex aequo
    res = _executer(sql, _base(scores=vrais), "requete_top_risque sur 2 000 clients", params={"k": 10})
    _comparer_top(res, _top_attendu(vrais, 10), "requete_top_risque sur 2 000 clients avec k=10")


# =========================================================================== Incertitude


def _z(confiance: float) -> float:
    return NormalDist().inv_cdf(1 - (1 - confiance) / 2)


def _wilson(departs: int, clients: int, z: float) -> tuple[float, float]:
    p, n = departs / clients, clients
    d = 1 + z**2 / n
    centre, demi = (p + z**2 / (2 * n)) / d, z / d * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, centre - demi), min(1.0, centre + demi)


def _wald(departs: int, clients: int, z: float) -> tuple[float, float]:
    p = departs / clients
    demi = z * math.sqrt(p * (1 - p) / clients)
    return p - demi, p + demi


def _meme(a, b, tol: float = 1e-6) -> bool:
    return all(math.isclose(float(x), float(y), abs_tol=tol) for x, y in zip(a, b))


def _intervalle(f, *args, **kwargs) -> tuple[float, float]:
    quoi = f"intervalle_wilson({', '.join([*map(repr, args), *(f'{k}={v!r}' for k, v in kwargs.items())])})"
    res = _jamais_none(f(*args, **kwargs), quoi)
    assert isinstance(res, tuple) and len(res) == 2, f"{quoi} : attendu un couple (bas, haut), obtenu {res!r}"
    bas, haut = (float(v) for v in res)
    assert bas <= haut, f"{quoi} : la borne basse ({bas:.4f}) dépasse la borne haute ({haut:.4f})"
    return bas, haut


def verifier_intervalle_wilson(f) -> None:
    z95 = _z(0.95)
    obtenu = _intervalle(f, 5, 10)
    if _meme(obtenu, _wilson(5, 10, NormalDist().inv_cdf(0.95)), 1e-4):
        raise AssertionError(
            "Vous utilisez z = 1,645, le quantile UNILATÉRAL. Un intervalle à 95 % laisse 2,5 % de chaque côté : "
            "z = NormalDist().inv_cdf(1 - (1 - confiance) / 2) ≈ 1,96."
        )
    if _meme(obtenu, _wald(5, 10, z95), 1e-4):
        raise AssertionError(
            "C'est l'intervalle de Wald, p ± z·√(p(1−p)/n). Il est juste ici mais faux aux extrêmes "
            "(0 départ, petits effectifs) : codez la formule de Wilson de la docstring."
        )
    attendu = _wilson(5, 10, z95)
    _proche(obtenu[0], attendu[0], "borne basse pour 5 départs sur 10 clients", tol=1e-4)
    _proche(obtenu[1], attendu[1], "borne haute pour 5 départs sur 10 clients", tol=1e-4)

    bas, haut = _intervalle(f, 0, 12)
    if haut <= 1e-9:
        raise AssertionError(
            "0 départ sur 12 clients donne [0 ; 0] : un intervalle de largeur nulle prétend une certitude "
            "impossible avec 12 clients (c'est le défaut de Wald). Wilson donne environ [0 ; 0,24]."
        )
    _proche(haut, _wilson(0, 12, z95)[1], "borne haute pour 0 départ sur 12 clients", tol=1e-4)
    _proche(bas, 0.0, "borne basse pour 0 départ sur 12 clients", tol=1e-9)

    bas, haut = _intervalle(f, 1, 10)
    assert bas >= 0, (
        f"1 départ sur 10 donne une borne basse négative ({bas:.3f}) : un taux de churn ne peut pas être "
        "négatif. Wilson reste toujours dans [0 ; 1]."
    )
    _proche(bas, _wilson(1, 10, z95)[0], "borne basse pour 1 départ sur 10 clients", tol=1e-4)

    bas, haut = _intervalle(f, 3, 3)
    assert haut <= 1 + 1e-9, f"3 départs sur 3 : borne haute {haut:.6f} > 1, impossible pour une proportion"
    _proche(bas, _wilson(3, 3, z95)[0], "borne basse pour 3 départs sur 3 clients", tol=1e-4)

    a_90 = _intervalle(f, 5, 10, confiance=0.90)
    if _meme(a_90, attendu, 1e-4):
        raise AssertionError(
            "Même intervalle à 90 % qu'à 95 % : le paramètre confiance est ignoré (z = 1,96 écrit en dur ?)."
        )
    if _meme(a_90, _wilson(5, 10, NormalDist().inv_cdf(0.90)), 1e-4):
        raise AssertionError(
            "À 90 %, vous utilisez z = inv_cdf(0,90) ≈ 1,28, le quantile UNILATÉRAL. Il faut 5 % de chaque côté : "
            "inv_cdf(1 - (1 - confiance) / 2) ≈ 1,645."
        )
    attendu_90 = _wilson(5, 10, _z(0.90))
    _proche(a_90[0], attendu_90[0], "borne basse à 90 % pour 5 sur 10", tol=1e-4)
    assert a_90[1] - a_90[0] < attendu[1] - attendu[0], "Un intervalle à 90 % doit être plus ÉTROIT qu'à 95 %"

    for departs, clients, pourquoi in ((0, 0, "aucun client"), (5, 3, "plus de départs que de clients"),
                                       (-1, 10, "départs négatifs")):
        try:
            res = f(departs, clients)
        except ValueError:
            continue
        except Exception as exc:
            raise AssertionError(
                f"intervalle_wilson({departs}, {clients}) ({pourquoi}) : attendu une ValueError claire, obtenu "
                f"{type(exc).__name__}: {exc}"
            ) from None
        raise AssertionError(
            f"intervalle_wilson({departs}, {clients}) ({pourquoi}) renvoie {res!r} au lieu de lever une ValueError : "
            "des entrées impossibles doivent être refusées, pas transformées en chiffre."
        )

    # La leçon : à taux égal, l'incertitude fond avec la taille du segment.
    petit, grand = _intervalle(f, 6, 10), _intervalle(f, 600, 1000)
    assert grand[1] - grand[0] < (petit[1] - petit[0]) / 5, (
        "Avec 100 fois plus de clients, l'intervalle doit être environ 10 fois plus étroit (√100)"
    )


def _p_bilaterale(z: float) -> float:
    return 2 * (1 - NormalDist().cdf(abs(z)))


def _ab_attendu(x1: int, n1: int, x2: int, n2: int, commune: bool = True) -> dict:
    p1, p2 = x1 / n1, x2 / n2
    if commune:
        p = (x1 + x2) / (n1 + n2)
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    else:
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = (p2 - p1) / se
    return {"ecart": p2 - p1, "z": z, "p_value": _p_bilaterale(z)}


def _ab(f, *args) -> dict:
    quoi = f"comparer_ab{args}"
    res = _jamais_none(f(*args), quoi)
    assert isinstance(res, dict), f"{quoi} : attendu un dict, obtenu {type(res).__name__}"
    assert set(res) == {"ecart", "z", "p_value"}, f"{quoi} : clés attendues ecart, z, p_value, obtenu {sorted(res)}"
    for cle, valeur in res.items():
        assert not (isinstance(valeur, float) and math.isnan(valeur)), (
            f"{quoi} : {cle} vaut NaN. Si l'erreur standard est nulle, renvoyez z = 0.0 et p_value = 1.0."
        )
    return {k: float(v) for k, v in res.items()}


def _variance_commune(res: dict, *args: int) -> None:
    attendu, separees = _ab_attendu(*args), _ab_attendu(*args, commune=False)
    if math.isclose(res["z"], separees["z"], rel_tol=1e-6):
        raise AssertionError(
            f"z = {res['z']:.4f} au lieu de {attendu['z']:.4f} pour {args} : vous estimez la variance de chaque groupe "
            "séparément. Le test suppose « l'offre ne change rien » (hypothèse nulle) : les deux groupes ont alors le "
            "MÊME taux, estimé sur les deux groupes réunis (variance commune)."
        )


def verifier_comparer_ab(f) -> None:
    res = _ab(f, 120, 400, 96, 400)
    attendu = _ab_attendu(120, 400, 96, 400)
    if math.isclose(res["ecart"], 0.06, abs_tol=1e-9):
        raise AssertionError(
            "ecart vaut +0,06 alors que l'offre fait BAISSER le churn (30 % → 24 %) : ecart = taux offre − taux "
            "témoin, négatif quand l'offre marche."
        )
    _proche(res["ecart"], -0.06, "ecart pour 30 % (témoin) contre 24 % (offre)")
    assert res["z"] < 0, f"z doit avoir le signe de l'écart (négatif ici), obtenu {res['z']:.3f}"
    if res["p_value"] > 1:
        raise AssertionError(
            f"p_value = {res['p_value']:.3f} > 1 : impossible pour une probabilité. Avec z négatif, "
            "1 − Φ(z) est proche de 1 : utilisez la valeur absolue, 2 × (1 − Φ(|z|))."
        )
    if math.isclose(res["p_value"], attendu["p_value"] / 2, rel_tol=1e-4):
        raise AssertionError(
            f"p_value = {res['p_value']:.4f}, la moitié de la bonne valeur : c'est la p-value UNILATÉRALE. On ne "
            "sait pas à l'avance si l'offre aide ou nuit (une offre maladroite peut AUGMENTER le churn) : "
            "test bilatéral, p = 2 × (1 − Φ(|z|))."
        )
    _variance_commune(res, 120, 400, 96, 400)
    _proche(res["z"], attendu["z"], "z pour 120/400 contre 96/400", tol=1e-4)
    _proche(res["p_value"], attendu["p_value"], "p_value pour 120/400 contre 96/400", tol=1e-4)

    # Groupes de tailles différentes : c'est là que la variance commune change vraiment z.
    res = _ab(f, 50, 200, 300, 1500)
    attendu = _ab_attendu(50, 200, 300, 1500)
    _variance_commune(res, 50, 200, 300, 1500)
    _proche(res["z"], attendu["z"], "z pour 50/200 contre 300/1500", tol=1e-4)
    _proche(res["p_value"], attendu["p_value"], "p_value pour 50/200 contre 300/1500", tol=1e-4)

    res = _ab(f, 300, 1000, 240, 1000)
    _proche(res["p_value"], _ab_attendu(300, 1000, 240, 1000)["p_value"], "p_value pour 300/1000 contre 240/1000",
            tol=1e-4)

    try:
        res = _ab(f, 0, 100, 0, 100)
    except ZeroDivisionError:
        raise AssertionError(
            "comparer_ab(0, 100, 0, 100) plante (division par zéro) : aucun départ nulle part, l'erreur standard "
            "est nulle. Renvoyez z = 0.0 et p_value = 1.0 (aucune différence observable)."
        ) from None
    assert (res["z"], res["p_value"]) == (0.0, 1.0), (
        f"Aucun départ dans les deux groupes : attendu z = 0.0 et p_value = 1.0, obtenu {res}"
    )

    for args in ((0, 0, 10, 100), (10, 100, 0, 0)):
        try:
            res = f(*args)
        except ValueError:
            continue
        except Exception as exc:
            raise AssertionError(
                f"comparer_ab{args} (un groupe vide) : attendu une ValueError claire, obtenu {type(exc).__name__}"
            ) from None
        raise AssertionError(f"comparer_ab{args} : un groupe vide doit lever une ValueError, obtenu {res!r}")

    from scipy.stats import chi2_contingency

    rng = np.random.default_rng(5)
    for _ in range(5):
        n1, n2 = (int(v) for v in rng.integers(50, 3000, size=2))
        x1, x2 = int(rng.integers(1, n1)), int(rng.integers(1, n2))
        res = _ab(f, x1, n1, x2, n2)
        p_scipy = chi2_contingency([[x1, n1 - x1], [x2, n2 - x2]], correction=False)[1]
        _proche(res["p_value"], p_scipy, f"p_value pour {x1}/{n1} contre {x2}/{n2} (comparée au χ² de scipy)", tol=1e-6)


# =========================================================================== Survie


def _km(duree, parti, a_risque_strict: bool = False, tous_partis: bool = False, sans_censures: bool = False,
        naif: bool = False) -> pd.DataFrame:
    """Kaplan-Meier de référence, et ses variantes fausses (pour reconnaître les erreurs classiques)."""
    duree, parti = np.asarray(duree), np.asarray(parti) == 1
    if sans_censures:
        duree, parti = duree[parti], parti[parti]
    lignes, s, cumul = [], 1.0, 0
    for mois in sorted(set(duree[parti].tolist())):
        a_risque = int((duree > mois).sum() if a_risque_strict else (duree >= mois).sum())
        departs = int((duree == mois).sum() if tous_partis else ((duree == mois) & parti).sum())
        cumul += departs
        s = 1 - cumul / len(duree) if naif else (s * (1 - departs / a_risque) if a_risque else float("nan"))
        lignes.append({"mois": int(mois), "a_risque": a_risque, "departs": departs, "survie": s})
    return pd.DataFrame(lignes, columns=["mois", "a_risque", "departs", "survie"])


def _courbe(f, df: pd.DataFrame, quoi: str) -> pd.DataFrame:
    avant = df.copy()
    try:
        obtenu = f(df)
    except ZeroDivisionError:
        raise AssertionError(
            f"{quoi} : division par zéro, donc a_risque vaut 0 à un mois où quelqu'un part. Impossible : le client "
            "qui part au mois t était encore client au début du mois t. a_risque compte tenure_months >= t."
        ) from None
    res = _dataframe(obtenu, quoi)
    if not df.equals(avant):
        raise AssertionError(f"{quoi} : le DataFrame reçu a été modifié. Il appartient à l'appelant.")
    _colonnes(res, ["mois", "a_risque", "departs", "survie"], quoi)
    return res.reset_index(drop=True)


def _egal(colonne: pd.Series, valeurs) -> bool:
    return len(colonne) == len(valeurs) and np.allclose(colonne.astype(float), np.asarray(valeurs, dtype=float))


def verifier_courbe_survie(f) -> None:
    jouet = pd.DataFrame(
        {"customer_id": list("FBDACE"), "tenure_months": [5, 2, 3, 1, 4, 2], "churn": [1, 0, 1, 1, 0, 1]},
        index=[10, 11, 12, 13, 14, 15],
    )  # dans le désordre, index qui ne commence pas à 0
    res = _courbe(f, jouet, "courbe_survie sur 6 clients")
    mois = [int(m) for m in res["mois"]]
    if 4 in mois:
        raise AssertionError(
            f"Mois obtenus {mois} : le mois 4 n'a aucun départ (le client à 4 mois est encore là). Une ligne par "
            "mois où au moins un client est PARTI : la courbe ne bouge qu'à ces mois-là."
        )
    assert mois == [1, 2, 3, 5], f"Mois attendus [1, 2, 3, 5] (ceux où quelqu'un part, croissants), obtenu {mois}"

    ref = _km(jouet["tenure_months"], jouet["churn"])
    if _egal(res["a_risque"], _km(jouet["tenure_months"], jouet["churn"], sans_censures=True)["a_risque"]):
        raise AssertionError(
            f"a_risque = {res['a_risque'].tolist()} : vous ne comptez que les clients partis. Un client encore "
            "présent à 4 mois (censuré) était bien là aux mois 1, 2 et 3 : il compte dans a_risque jusqu'à son "
            "ancienneté. Sinon la courbe finit toujours à 0 % : « tout le monde part », ce qui est faux."
        )
    if _egal(res["a_risque"], _km(jouet["tenure_months"], jouet["churn"], a_risque_strict=True)["a_risque"]):
        raise AssertionError(
            f"a_risque = {res['a_risque'].tolist()} au lieu de {ref['a_risque'].tolist()} : un client qui part au "
            "mois t était encore client au début du mois t. a_risque compte tenure_months >= t, pas > t."
        )
    assert _egal(res["a_risque"], ref["a_risque"]), (
        f"a_risque attendu {ref['a_risque'].tolist()}, obtenu {res['a_risque'].tolist()} "
        "(clients avec tenure_months >= mois)"
    )
    if _egal(res["departs"], _km(jouet["tenure_months"], jouet["churn"], tous_partis=True)["departs"]):
        raise AssertionError(
            f"departs = {res['departs'].tolist()} : au mois 2, un des deux clients est encore là (churn = 0). Un "
            "client censuré n'est PAS un départ : comptez tenure_months == mois ET churn == 1."
        )
    assert _egal(res["departs"], ref["departs"]), (
        f"departs attendu {ref['departs'].tolist()}, obtenu {res['departs'].tolist()}"
    )
    if _egal(res["survie"], _km(jouet["tenure_months"], jouet["churn"], naif=True)["survie"]):
        raise AssertionError(
            f"survie = {np.round(res['survie'].astype(float), 3).tolist()} : c'est 1 − (départs cumulés / tous les "
            "clients). Cette formule ignore la censure : un client arrivé il y a 2 mois ne peut pas encore être "
            "parti au mois 3. Kaplan-Meier divise par les clients ENCORE à risque à chaque mois, puis multiplie."
        )
    if _egal(res["survie"], 1 - ref["departs"] / ref["a_risque"]):
        raise AssertionError(
            "survie = 1 − departs / a_risque, mois par mois : c'est la probabilité de passer CE mois. La survie "
            "jusqu'au mois t est le PRODUIT de ces probabilités depuis le début (cumprod)."
        )
    assert _egal(res["survie"], ref["survie"]), (
        f"survie attendue {np.round(ref['survie'], 4).tolist()}, obtenu {np.round(res['survie'].astype(float), 4).tolist()}"
    )

    personne = _courbe(f, jouet.assign(churn=0), "courbe_survie quand personne n'est parti")
    assert len(personne) == 0, f"Personne n'est parti : la courbe n'a aucune ligne, obtenu {len(personne)}"

    from churn.data import generate_customers

    vrai = generate_customers(n=3000, seed=4)
    res = _courbe(f, vrai, "courbe_survie sur 3 000 clients")
    ref = _km(vrai["tenure_months"], vrai["churn"])
    assert len(res) == len(ref), f"Sur 3 000 clients : {len(ref)} mois avec départ attendus, obtenu {len(res)}"
    assert _egal(res["a_risque"], ref["a_risque"]) and _egal(res["departs"], ref["departs"]), (
        "Sur 3 000 clients : a_risque ou departs incorrects"
    )
    assert np.allclose(res["survie"].astype(float), ref["survie"], atol=1e-9), "Sur 3 000 clients : survie incorrecte"
    assert res["survie"].is_monotonic_decreasing, "Une courbe de survie ne remonte jamais"


SERIE = Serie(
    code="05",
    titre="Prouver l'impact : SQL, incertitude, tests A/B, survie",
    module_eleve="defis.serie_05",
    module_corrige="defis.corrige_serie_05",
    defis=(
        Defi(25, "requete_churn_par_contrat", "🟢", "1.2", verifier_requete_churn_par_contrat),
        Defi(26, "requete_top_risque", "🟡", "—", verifier_requete_top_risque),
        Defi(27, "intervalle_wilson", "🟡", "—", verifier_intervalle_wilson),
        Defi(28, "comparer_ab", "🟡", "—", verifier_comparer_ab),
        Defi(29, "courbe_survie", "🔴", "—", verifier_courbe_survie),
    ),
)

"""Vérifications automatiques de la série 10 (SQL pour l'analyse).

Votre requête est exécutée sur de petites tables écrites à la main, où chaque piège classique donne
un résultat reconnaissable (un client sans appel qui disparaît, un chiffre d'affaires multiplié par
trois, un mois de décembre qui ne précède plus janvier…), puis sur une base de 600 clients, comparée
à un calcul indépendant fait avec pandas.

Lire ces tables est un bon exercice : elles montrent les cas limites auxquels un·e analyste pense
AVANT d'envoyer un chiffre à sa direction.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta
from functools import cache

import pandas as pd

from .donnees_sql import charger, generer_tables
from .verif import Defi, Serie

COLONNES = {
    "clients": ["customer_id", "contract", "region", "signup_date", "churn_date"],
    "factures": ["invoice_id", "customer_id", "invoice_date", "amount"],
    "appels": ["call_id", "customer_id", "call_date", "motif"],
}

# ============================================================================ Outils communs


def _sql(f, quoi: str) -> str:
    sql = f()
    if sql is None:
        raise AssertionError(f"{quoi} : la fonction a retourné None. Avez-vous oublié `return` ?")
    if not isinstance(sql, str):
        raise AssertionError(f"{quoi} doit renvoyer le TEXTE de la requête (une str), obtenu {type(sql).__name__}.")
    if "select" not in sql.lower():
        raise AssertionError(f"{quoi} : la requête doit contenir un SELECT, obtenu {sql!r}.")
    return sql


def _tables(**tables: list[tuple]) -> dict[str, pd.DataFrame]:
    """Les trois tables ; celles qu'on ne donne pas existent mais sont vides."""
    return {nom: pd.DataFrame(tables.get(nom, []), columns=colonnes) for nom, colonnes in COLONNES.items()}


def _conseil(message: str, params: dict | None) -> str:
    bas = message.lower()
    if "no such table" in bas:
        return " Les tables s'appellent `clients`, `factures` et `appels` (python -m defis.donnees_sql pour les voir)."
    if "no such column" in bas:
        colonnes = " ; ".join(f"{t}({', '.join(c)})" for t, c in COLONNES.items())
        return (f" Les colonnes existantes : {colonnes}. Une colonne calculée (AS …) n'existe pas encore dans le WHERE "
                "de la même requête : répétez l'expression ou passez par une CTE.")
    if "ambiguous column" in bas:
        return " Deux tables ont cette colonne : préfixez-la par l'alias de sa table (c.customer_id, a.customer_id)."
    if "has no name" in bas or "binding" in bas:
        noms = ", ".join(f":{p}" for p in params or {})
        return f" Utilisez le paramètre NOMMÉ ({noms}), pas un « ? » anonyme."
    if "did not supply a value" in bas or "you did not supply" in bas:
        noms = ", ".join(f":{p}" for p in params or {}) or "aucun"
        return f" Paramètres fournis par le correcteur : {noms}."
    if "misuse of aggregate" in bas:
        return (" Un agrégat (COUNT, SUM, MAX…) n'a pas sa place dans WHERE, évalué ligne par ligne AVANT le GROUP BY : "
                "filtrez avec HAVING, ou calculez l'agrégat dans une CTE.")
    if "misuse of window function" in bas:
        return (" WHERE et GROUP BY sont évalués AVANT les fonctions de fenêtre : calculez la fenêtre dans une CTE "
                "(WITH … AS (…)), puis filtrez ou regroupez dessus.")
    if "no such function" in bas and any(n in bas for n in ("lag", "lead", "row_number", "rank")):
        return f" Votre SQLite ({sqlite3.sqlite_version}) est trop ancien pour les fonctions de fenêtre : il faut 3.25 ou plus."
    if "syntax error" in bas:
        return " Regardez juste AVANT le mot cité : une virgule en trop avant FROM, une parenthèse non fermée ?"
    return ""


def _executer(sql: str, tables: dict[str, pd.DataFrame], quoi: str, params: dict | None = None) -> pd.DataFrame:
    connexion = charger(tables)
    try:
        return pd.read_sql_query(sql, connexion, params=params)
    except Exception as exc:
        cause = exc.__cause__ or exc  # pandas enveloppe l'erreur SQLite : on garde le message court
        message = str(cause)
        raise AssertionError(
            f"{quoi} : la requête plante dans SQLite ({type(cause).__name__}: {message}).{_conseil(message, params)}"
        ) from None
    finally:
        connexion.close()


def _colonnes(res: pd.DataFrame, attendues: list[str], quoi: str) -> None:
    if list(res.columns) != attendues:
        raise AssertionError(
            f"{quoi} : colonnes attendues {attendues}, obtenu {list(res.columns)}. Nommez chaque colonne calculée "
            "avec AS, dans cet ordre."
        )


def _nul(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _meme(x, y, tol: float = 1e-6) -> bool:
    if _nul(y):
        return _nul(x)
    if _nul(x):
        return False
    if isinstance(y, str) or isinstance(x, str):
        return x == y
    try:
        return math.isclose(float(x), float(y), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def _lisible(v):
    if _nul(v):
        return None
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 2)
    if hasattr(v, "item"):  # entier ou flottant numpy
        return _lisible(v.item())
    return v


def _lignes(res: pd.DataFrame) -> list[tuple]:
    return [tuple(_lisible(v) for v in ligne) for ligne in res.itertuples(index=False)]


def _egal(res: pd.DataFrame, attendu: list[tuple]) -> bool:
    return len(res) == len(attendu) and all(
        len(l) == len(a) and all(_meme(x, y) for x, y in zip(l, a))
        for l, a in zip(res.itertuples(index=False), attendu)
    )


def _apercu(lignes: list[tuple], n: int = 6) -> str:
    lignes = [tuple(_lisible(v) for v in l) for l in lignes]
    suite = f" … (+{len(lignes) - n} lignes)" if len(lignes) > n else ""
    return f"{lignes[:n]}{suite}"


def _par(res: pd.DataFrame, cle: str) -> dict:
    """{valeur de la clé: {colonne: valeur}} (la dernière ligne gagne en cas de doublon)."""
    return {ligne[cle]: ligne for ligne in res.to_dict("records")}


def _comparer(res: pd.DataFrame, attendu: list[tuple], quoi: str, tri: str) -> None:
    """Dernier filet : même contenu, puis même ordre."""
    if _egal(res, attendu):
        return
    if len(res) == len(attendu):
        cle = lambda l: tuple(str(_lisible(v)) for v in l)  # noqa: E731
        trie_res = sorted((tuple(l) for l in res.itertuples(index=False)), key=cle)
        trie_att = sorted(attendu, key=cle)
        if all(all(_meme(x, y) for x, y in zip(l, a)) for l, a in zip(trie_res, trie_att)):
            raise AssertionError(f"{quoi} : les bonnes lignes, mais pas dans le bon ordre. {tri}")
    raise AssertionError(f"{quoi} : attendu {_apercu(attendu)}, obtenu {_apercu(_lignes(res))}.")


@cache
def _grande_base() -> dict[str, pd.DataFrame]:
    """600 clients, générés une seule fois pour tous les défis."""
    return generer_tables(n_clients=600, seed=21)


# ============================================================================ Défi 62 : NULL

_CLIENTS_62 = [
    ("B3", "Mensuel", "Sud", "2025-01-01", None),
    ("A1", "Mensuel", "Nord", "2025-01-10", "2026-01-10"),
    ("X1", "Mensuel", None, "2025-01-01", "2026-01-05"),
    ("E1", "Annuel", "Est", "2025-01-01", None),
    ("B1", "Mensuel", "Sud", "2025-01-01", "2026-02-01"),
    ("A2", "Mensuel", "Nord", "2025-02-01", None),
    ("X2", "Annuel", None, "2025-01-01", None),
    ("B2", "Annuel", "Sud", "2025-01-01", "2026-03-01"),
    ("A3", "Annuel", "Nord", "2025-03-01", None),
    ("B4", "Biannuel", "Sud", "2025-01-01", None),
]
_ATTENDU_62 = [("Inconnue", 2, 1, 0.5), ("Sud", 4, 2, 0.5), ("Nord", 3, 1, 1 / 3), ("Est", 1, 0, 0.0)]


def _reference_62(tables: dict[str, pd.DataFrame]) -> list[tuple]:
    c = tables["clients"].assign(region=lambda d: d["region"].fillna("Inconnue"))
    g = c.groupby("region").agg(clients=("customer_id", "size"), partis=("churn_date", "count")).reset_index()
    g["taux_churn"] = g["partis"] / g["clients"]
    g = g.sort_values(["taux_churn", "region"], ascending=[False, True])
    return list(g.itertuples(index=False, name=None))


def verifier_requete_churn_par_region(f) -> None:
    quoi = "requete_churn_par_region"
    sql = _sql(f, f"{quoi}()")
    res = _executer(sql, _tables(clients=_CLIENTS_62), quoi)
    _colonnes(res, ["region", "clients", "partis", "taux_churn"], quoi)
    if any(_nul(r) for r in res["region"]):
        raise AssertionError(
            f"{quoi} : une ligne a une région vide (NULL) : ce sont les 2 clients dont la région est inconnue. Un "
            "tableau de bord afficherait une case vide que personne ne comprend : COALESCE(region, 'Inconnue')."
        )
    regions = set(res["region"])
    if regions == {"Sud", "Nord", "Est"}:
        raise AssertionError(
            f"{quoi} : les clients sans région ont disparu. Ne les filtrez pas : 8 % de la base, c'est trop pour "
            "être ignoré en silence. Regroupez-les sous 'Inconnue' avec COALESCE(region, 'Inconnue')."
        )
    if regions != {"Inconnue", "Sud", "Nord", "Est"} or len(res) != 4:
        raise AssertionError(
            f"{quoi} : attendu une ligne par région (Est, Inconnue, Nord, Sud), obtenu {_apercu(_lignes(res))}. "
            "Un GROUP BY sur la région ?"
        )
    lignes = _par(res, "region")
    attendu = {r[0]: r for r in _ATTENDU_62}
    if any(not _meme(lignes[r]["clients"], attendu[r][1]) for r in attendu):
        raise AssertionError(
            f"{quoi} : nombre de clients attendu {({r: a[1] for r, a in attendu.items()})}, obtenu "
            f"{({r: _lisible(l['clients']) for r, l in lignes.items()})}. COUNT(*) compte les lignes du groupe."
        )
    partis = {r: _lisible(l["partis"]) for r, l in lignes.items()}
    if all(_meme(lignes[r]["partis"], attendu[r][1]) for r in attendu):
        raise AssertionError(
            f"{quoi} : partis = clients dans chaque région, comme si tout le monde était parti. COUNT(x) compte les "
            "valeurs NON NULL de x : COUNT(*) compte toutes les lignes, et COUNT(churn_date IS NOT NULL) aussi, "
            "car le résultat d'une comparaison vaut 0 ou 1, et 0 n'est pas NULL. Écrivez COUNT(churn_date) "
            "ou SUM(churn_date IS NOT NULL)."
        )
    if all(_meme(v, 0) for v in partis.values()):
        raise AssertionError(
            f"{quoi} : 0 parti partout, alors que 4 clients sur 10 ont une date de départ. `churn_date = NULL` "
            "n'est jamais vrai : NULL veut dire « inconnu », et « inconnu = inconnu » est encore inconnu. Testez "
            "avec IS NULL / IS NOT NULL, ou comptez les valeurs présentes avec COUNT(churn_date)."
        )
    if any(not _meme(lignes[r]["partis"], attendu[r][2]) for r in attendu):
        raise AssertionError(
            f"{quoi} : partis attendus {({r: a[2] for r, a in attendu.items()})}, obtenu {partis}. Un client est parti "
            "quand churn_date n'est PAS NULL."
        )
    taux = {r: _lisible(l["taux_churn"]) for r, l in lignes.items()}
    if all(_meme(lignes[r]["taux_churn"], 1 - attendu[r][3]) for r in attendu):
        raise AssertionError(
            f"{quoi} : taux obtenus {taux} : c'est la part des clients RESTÉS. Le taux de churn est la part des partis."
        )
    if all(_meme(lignes[r]["taux_churn"], 0) for r in attendu):
        raise AssertionError(
            f"{quoi} : taux de churn nul partout ({taux}) : c'est la division ENTIÈRE de SQL (1 / 2 vaut 0 entre deux "
            "entiers). Écrivez 1.0 * COUNT(churn_date) / COUNT(*), ou AVG(churn_date IS NOT NULL)."
        )
    if any(not _meme(lignes[r]["taux_churn"], attendu[r][3]) for r in attendu):
        raise AssertionError(
            f"{quoi} : taux attendus {({r: round(a[3], 4) for r, a in attendu.items()})}, obtenu {taux} "
            "(partis / clients de la région)."
        )
    _comparer(res, _ATTENDU_62, quoi,
              "Triez par taux de churn décroissant, puis par région pour départager Inconnue et Sud (tous deux à "
              "50 %) : ORDER BY taux_churn DESC, region. Sans départage, l'ordre des ex aequo peut changer d'une "
              "exécution à l'autre.")

    grande = _grande_base()
    res = _executer(sql, grande, f"{quoi} sur 600 clients")
    _comparer(res, _reference_62(grande), f"{quoi} sur 600 clients", "ORDER BY taux_churn DESC, region.")


# ============================================================================ Défi 63 : anti-jointure

_CLIENTS_63 = [
    ("D", "Annuel", "Sud", "2025-01-01", None),
    ("B", "Mensuel", "Nord", "2025-01-01", None),
    ("A", "Mensuel", "Nord", "2025-01-01", None),
    ("E", "Mensuel", "Est", "2025-01-01", "2026-02-01"),
    ("C", "Annuel", "Est", "2025-01-01", "2026-03-01"),
]
_APPELS_63 = [
    (1, "B", "2026-01-10", "Panne"),
    (2, None, "2026-01-12", "Offre"),       # un numéro inconnu : customer_id NULL
    (3, "B", "2026-02-03", "Facturation"),
    (4, "E", "2026-01-20", "Résiliation"),
]


def _reference_63(tables: dict[str, pd.DataFrame]) -> list[tuple]:
    c = tables["clients"]
    ont_appele = set(tables["appels"]["customer_id"].dropna())
    actifs = c[c["churn_date"].isna() & ~c["customer_id"].isin(ont_appele)]
    return [(cid,) for cid in sorted(actifs["customer_id"])]


def verifier_requete_actifs_sans_appel(f) -> None:
    quoi = "requete_actifs_sans_appel"
    sql = _sql(f, f"{quoi}()")
    res = _executer(sql, _tables(clients=_CLIENTS_63, appels=_APPELS_63), quoi)
    _colonnes(res, ["customer_id"], quoi)
    ids = list(res["customer_id"])
    bas = " ".join(sql.lower().split())
    if not ids:
        if "not in" in bas:
            sans_anonyme = _executer(sql, _tables(clients=_CLIENTS_63, appels=[a for a in _APPELS_63 if a[1]]), quoi)
            if list(sans_anonyme["customer_id"]):
                raise AssertionError(
                    f"{quoi} : résultat VIDE, à cause d'un seul appel anonyme (customer_id NULL). `x NOT IN (…, NULL)` "
                    "ne vaut jamais vrai : « x est-il différent d'une valeur inconnue ? », réponse inconnue. Sans "
                    "l'appel anonyme, votre requête marche… jusqu'au premier appel anonyme en production. Utilisez "
                    "NOT EXISTS (SELECT 1 FROM appels a WHERE a.customer_id = c.customer_id), ou un LEFT JOIN suivi "
                    "de WHERE a.call_id IS NULL."
                )
        if "= null" in bas or "=null" in bas:
            raise AssertionError(f"{quoi} : résultat vide. `= NULL` n'est jamais vrai : écrivez IS NULL.")
        raise AssertionError(f"{quoi} : résultat vide, alors que A et D sont toujours clients et n'ont jamais appelé.")
    if any(_nul(i) for i in ids):
        raise AssertionError(f"{quoi} : une ligne a un customer_id NULL : partez de la table `clients`, pas de `appels`.")
    partis = sorted({i for i in ids if i in {"C", "E"}})
    if partis:
        raise AssertionError(
            f"{quoi} : {partis} sont partis (churn_date renseignée) : on ne fait pas d'enquête de satisfaction auprès "
            "d'anciens clients. Ajoutez la condition c.churn_date IS NULL."
        )
    if "B" in ids:
        raise AssertionError(
            f"{quoi} : B a appelé le support deux fois, il ne doit pas apparaître (obtenu {ids}). Avec un LEFT JOIN, "
            "les clients SANS appel sont ceux dont le côté droit est resté vide : gardez WHERE a.call_id IS NULL."
        )
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{quoi} : des clients apparaissent plusieurs fois ({ids}) : une ligne par client.")
    _comparer(res, [("A",), ("D",)], quoi, "Terminez par ORDER BY customer_id.")

    grande = _grande_base()
    res = _executer(sql, grande, f"{quoi} sur 600 clients")
    _comparer(res, _reference_63(grande), f"{quoi} sur 600 clients", "ORDER BY customer_id.")


# ============================================================================ Défi 64 : LEFT JOIN, ON ou WHERE

_D1, _D2 = "2026-06-30", "2026-03-31"
_CLIENTS_64 = [
    ("K01", "Mensuel", "Nord", "2025-01-01", None),
    ("K02", "Mensuel", "Nord", "2025-01-01", None),          # aucun appel
    ("K03", "Annuel", "Sud", "2025-01-01", None),            # un appel, trop ancien
    ("K04", "Annuel", "Sud", "2025-01-01", None),            # un appel le jour J, un à J - 90
    ("K05", "Mensuel", "Est", "2025-01-01", "2026-05-20"),   # parti entre les deux dates
    ("K06", "Mensuel", "Est", "2026-07-15", None),           # pas encore client
    ("K07", "Annuel", "Ouest", "2025-01-01", "2026-06-30"),  # parti LE jour de l'extraction
    ("K08", "Mensuel", "Ouest", "2025-01-01", "2026-05-01"),
    ("K09", "Mensuel", "Nord", "2026-04-10", "2026-06-01"),  # arrivé après _D2, parti avant _D1
    ("K10", "Biannuel", "Nord", "2025-01-01", None),         # un appel à J - 91
]
_APPELS_64 = [
    (1, "K01", "2026-06-15", "Panne"), (2, "K01", "2026-05-01", "Offre"), (3, "K01", "2026-01-01", "Panne"),
    (4, "K03", "2025-12-01", "Facturation"),
    (5, "K04", "2026-06-30", "Panne"), (6, "K04", "2026-04-01", "Facturation"),
    (7, "K10", "2026-03-31", "Offre"),
    (8, "K05", "2026-05-10", "Résiliation"),
    (9, "K07", "2026-02-14", "Panne"),
    (10, "K08", "2026-03-01", "Facturation"),
    (11, "K06", "2026-07-20", "Offre"),
    (12, "K09", "2026-04-20", "Panne"),
    (13, None, "2026-06-01", "Offre"),
]
_ATTENDU_64 = {
    _D1: [("K01", 2), ("K04", 1), ("K02", 0), ("K03", 0), ("K10", 0)],
    _D2: [("K01", 1), ("K07", 1), ("K08", 1), ("K02", 0), ("K03", 0), ("K04", 0), ("K05", 0), ("K10", 0)],
}


def _reference_64(tables: dict[str, pd.DataFrame], jour: str) -> list[tuple]:
    c, a = tables["clients"], tables["appels"]
    actifs = c[(c["signup_date"] <= jour) & (c["churn_date"].isna() | (c["churn_date"] > jour))]
    debut = (date.fromisoformat(jour) - timedelta(days=90)).isoformat()
    recents = a[(a["call_date"] >= debut) & (a["call_date"] < jour)]
    nb = recents.groupby("customer_id").size()
    out = pd.DataFrame({"customer_id": actifs["customer_id"]})
    out["appels_90j"] = out["customer_id"].map(nb).fillna(0).astype(int)
    out = out.sort_values(["appels_90j", "customer_id"], ascending=[False, True])
    return list(out.itertuples(index=False, name=None))


def _verifier_64_a_une_date(res: pd.DataFrame, jour: str, quoi: str) -> None:
    attendu = dict(_ATTENDU_64[jour])
    obtenu = {r["customer_id"]: r["appels_90j"] for r in res.to_dict("records")}
    ids = list(res["customer_id"])
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{quoi} : des clients apparaissent plusieurs fois ({ids}) : GROUP BY c.customer_id.")
    sans_appel = sorted(c for c, n in attendu.items() if n == 0 and c not in obtenu)
    if sans_appel and len(sans_appel) >= 2:
        raise AssertionError(
            f"{quoi} : les clients sans appel récent ont disparu ({sans_appel}), alors que « 0 appel » est une "
            "information, souvent la plus importante. Une condition sur la table de DROITE d'un LEFT JOIN placée "
            "dans le WHERE élimine les lignes où elle vaut NULL : le LEFT JOIN redevient un INNER JOIN. Mettez les "
            "conditions de date dans le ON : LEFT JOIN appels a ON a.customer_id = c.customer_id AND a.call_date …"
        )
    if jour == _D1 and "K06" in obtenu:
        raise AssertionError(
            f"{quoi} : K06 s'inscrit le 2026-07-15, après la date d'extraction : il n'était pas encore client. "
            "Ajoutez c.signup_date <= :date_extraction."
        )
    if jour == _D1 and "K07" in obtenu:
        raise AssertionError(
            f"{quoi} : K07 est parti le jour même de l'extraction ({_D1}) : il n'est plus client ce jour-là. "
            "Un client est actif si churn_date est NULL OU strictement après :date_extraction."
        )
    if jour == _D2 and "K09" in obtenu:
        raise AssertionError(
            f"{quoi} : avec :date_extraction = {_D2}, K09 (inscrit le 2026-04-10) apparaît. C'est la priorité des "
            "opérateurs : `A AND B OR C` se lit `(A AND B) OR C`, comme 2 × 3 + 4. Mettez des parenthèses : "
            "signup_date <= :date_extraction AND (churn_date IS NULL OR churn_date > :date_extraction)."
        )
    partis = sorted(c for c in obtenu if c not in attendu)
    if partis:
        raise AssertionError(
            f"{quoi} : {partis} ne sont pas clients au {jour} (partis avant, ou pas encore arrivés). Un client actif "
            "à la date d'extraction : signup_date <= :date_extraction AND (churn_date IS NULL OR churn_date > "
            ":date_extraction)."
        )
    manquants = sorted(c for c in attendu if c not in obtenu)
    if manquants:
        if jour == _D2 and {"K05", "K07", "K08"} & set(manquants):
            raise AssertionError(
                f"{quoi} : avec :date_extraction = {_D2}, il manque {manquants}. Ils sont partis APRÈS cette date : "
                "ce jour-là, ils étaient encore clients. Le modèle est réentraîné sur des « photos » passées de la "
                "base : `churn_date IS NULL` décrit aujourd'hui, pas la date demandée. Écrivez (churn_date IS NULL "
                "OR churn_date > :date_extraction)."
            )
        raise AssertionError(f"{quoi} : il manque les clients {manquants} au {jour}.")
    faux = {c: (_lisible(obtenu[c]), attendu[c]) for c in attendu if not _meme(obtenu[c], attendu[c])}
    if faux:
        if _meme(obtenu.get("K02"), 1):
            raise AssertionError(
                f"{quoi} : K02 n'a jamais appelé, mais votre requête lui compte 1 appel. Après un LEFT JOIN, un client "
                "sans appel garde UNE ligne, complétée par des NULL : COUNT(*) la compte. COUNT(a.call_id) ne compte "
                "que les vrais appels."
            )
        if jour == _D1 and _meme(obtenu.get("K04"), 2):
            raise AssertionError(
                f"{quoi} : K04 a 2 appels au lieu de 1 : l'appel du {_D1}, jour de l'extraction, est compté. Le jour J "
                "n'est pas terminé quand on extrait la base : la fenêtre s'arrête la veille, a.call_date < "
                ":date_extraction (ni <=, ni absence de borne haute)."
            )
        if jour == _D1 and _meme(obtenu.get("K04"), 0):
            raise AssertionError(
                f"{quoi} : K04 a 0 appel au lieu de 1 : son appel du 2026-04-01, exactement 90 jours avant, doit "
                "compter. Bornez avec a.call_date >= date(:date_extraction, '-90 days') (>=, pas >)."
            )
        if jour == _D1 and _meme(obtenu.get("K10"), 1):
            raise AssertionError(
                f"{quoi} : K10 a 1 appel au lieu de 0 : son appel du 2026-03-31 est 91 jours avant l'extraction. "
                "« 3 mois » n'est pas « 90 jours » (date(…, '-3 months') recule au 2026-03-30) : "
                "date(:date_extraction, '-90 days')."
            )
        if jour == _D2 and _meme(obtenu.get("K05"), 1):
            raise AssertionError(
                f"{quoi} : avec :date_extraction = {_D2}, K05 a 1 appel : celui du 2026-05-10, dans le FUTUR de la "
                "date d'extraction. Sans borne haute (a.call_date < :date_extraction), le modèle apprend avec des "
                "informations qu'il n'aura jamais en production : c'est la fuite temporelle du défi 55."
            )
        raise AssertionError(f"{quoi} au {jour} : appels (obtenu, attendu) incorrects pour {faux}.")
    _comparer(res, _ATTENDU_64[jour], f"{quoi} au {jour}",
              "Les clients qui appellent le plus d'abord, puis par identifiant : ORDER BY appels_90j DESC, customer_id.")


def verifier_requete_appels_recents(f) -> None:
    quoi = "requete_appels_recents"
    sql = _sql(f, f"{quoi}()")
    tables = _tables(clients=_CLIENTS_64, appels=_APPELS_64)
    res1 = _executer(sql, tables, quoi, params={"date_extraction": _D1})
    _colonnes(res1, ["customer_id", "appels_90j"], quoi)
    _verifier_64_a_une_date(res1, _D1, quoi)

    res2 = _executer(sql, tables, quoi, params={"date_extraction": _D2})
    if _egal(res2, _ATTENDU_64[_D1]):
        raise AssertionError(
            f"{quoi} : avec :date_extraction = {_D2}, le résultat est celui du {_D1}. La date est-elle écrite en dur "
            "dans la requête ? Utilisez le paramètre :date_extraction partout (actifs ET fenêtre d'appels)."
        )
    _verifier_64_a_une_date(res2, _D2, quoi)

    grande = _grande_base()
    for jour in ("2026-06-30", "2025-11-15"):
        res = _executer(sql, grande, f"{quoi} sur 600 clients", params={"date_extraction": jour})
        _comparer(res, _reference_64(grande, jour), f"{quoi} sur 600 clients au {jour}",
                  "ORDER BY appels_90j DESC, customer_id.")


# ============================================================================ Défi 65 : explosion des jointures

_CLIENTS_65 = [
    ("P3", "Annuel", "Sud", "2025-01-01", None),
    ("P1", "Mensuel", "Nord", "2025-01-01", None),
    ("P4", "Biannuel", "Est", "2025-01-01", None),   # ni facture ni appel
    ("P2", "Mensuel", "Nord", "2025-01-01", "2026-01-01"),
    ("P5", "Annuel", "Sud", "2025-01-01", None),     # des factures, aucun appel
]
_FACTURES_65 = [
    (1, "P1", "2026-01-05", 10.0), (2, "P1", "2026-02-05", 10.0), (3, "P1", "2026-03-05", 10.0),
    (4, "P2", "2025-12-01", 20.0),
    (5, "P3", "2026-01-10", 15.0), (6, "P3", "2026-02-10", 15.0),
    (7, "P5", "2026-03-01", 5.0),
]
_APPELS_65 = [
    (1, "P1", "2026-01-15", "Panne"), (2, "P1", "2026-02-20", "Facturation"),
    (3, "P3", "2026-02-11", "Offre"),
    (4, None, "2026-02-12", "Offre"),
]
_ATTENDU_65 = [("Annuel", 2, 35.0, 1), ("Biannuel", 1, 0.0, 0), ("Mensuel", 2, 50.0, 2)]


def _reference_65(tables: dict[str, pd.DataFrame]) -> list[tuple]:
    c, fa, ap = tables["clients"], tables["factures"], tables["appels"]
    ca = c["customer_id"].map(fa.groupby("customer_id")["amount"].sum()).fillna(0.0)
    nb = c["customer_id"].map(ap.groupby("customer_id").size()).fillna(0)
    out = (c.assign(ca=ca, nb=nb).groupby("contract")
           .agg(clients=("customer_id", "size"), chiffre_affaires=("ca", "sum"), appels=("nb", "sum"))
           .reset_index().sort_values("contract"))
    return list(out.itertuples(index=False, name=None))


def verifier_requete_bilan_par_contrat(f) -> None:
    quoi = "requete_bilan_par_contrat"
    sql = _sql(f, f"{quoi}()")
    res = _executer(sql, _tables(clients=_CLIENTS_65, factures=_FACTURES_65, appels=_APPELS_65), quoi)
    _colonnes(res, ["contract", "clients", "chiffre_affaires", "appels"], quoi)
    lignes = _par(res, "contract")
    if "Biannuel" not in lignes:
        raise AssertionError(
            f"{quoi} : le contrat Biannuel a disparu. Son seul client, P4, n'a ni facture ni appel : une jointure "
            "interne (JOIN) le supprime. Partez de `clients` et utilisez des LEFT JOIN."
        )
    if set(lignes) != {"Annuel", "Biannuel", "Mensuel"} or len(res) != 3:
        raise AssertionError(f"{quoi} : attendu une ligne par contrat, obtenu {_apercu(_lignes(res))}.")
    mensuel, bi = lignes["Mensuel"], lignes["Biannuel"]
    ca_gonfle, appels_gonfles = _meme(mensuel["chiffre_affaires"], 80), _meme(mensuel["appels"], 6)
    if ca_gonfle or appels_gonfles:
        symptomes = []
        if ca_gonfle:
            symptomes.append("80 € de chiffre d'affaires au lieu de 50 € (chaque facture de P1 comptée 2 fois)")
        if appels_gonfles:
            symptomes.append("6 appels au lieu de 2 (chaque appel de P1 compté 3 fois)")
        raise AssertionError(
            f"{quoi} : Mensuel obtient {' et '.join(symptomes)}. P1 a 3 factures et 2 appels : joindre les DEUX "
            "tables à clients donne 3 × 2 = 6 lignes pour P1. C'est l'explosion des jointures (fan-out), et "
            "DISTINCT ne répare pas une SOMME. Agrégez d'abord chaque table PAR CLIENT dans une CTE "
            "(WITH ca_client AS (SELECT customer_id, SUM(amount) … GROUP BY customer_id)), puis joignez."
        )
    if _meme(mensuel["clients"], 7) or _meme(lignes["Annuel"]["clients"], 3):
        raise AssertionError(
            f"{quoi} : {_lisible(mensuel['clients'])} clients Mensuel au lieu de 2 : COUNT(*) compte les lignes de la "
            "jointure, pas les clients. Agrégez les factures et les appels par client AVANT de joindre."
        )
    if _nul(bi["chiffre_affaires"]) or _nul(bi["appels"]):
        raise AssertionError(
            f"{quoi} : Biannuel affiche {_lisible(bi['chiffre_affaires'])} € et {_lisible(bi['appels'])} appels. "
            "SUM de rien vaut NULL en SQL, pas 0 : un tableau de bord afficherait une case vide. "
            "COALESCE(SUM(…), 0)."
        )
    if _meme(lignes["Annuel"]["appels"], 2) or _meme(lignes["Annuel"]["appels"], 3):
        raise AssertionError(
            f"{quoi} : Annuel compte {_lisible(lignes['Annuel']['appels'])} appels au lieu de 1 : P5 n'a aucun appel, "
            "mais une ligne vide après le LEFT JOIN. Comptez les appels réels (COUNT(*) dans la CTE des appels, "
            "puis SUM), pas les lignes."
        )
    _comparer(res, _ATTENDU_65, quoi, "Terminez par ORDER BY contract.")

    grande = _grande_base()
    res = _executer(sql, grande, f"{quoi} sur 600 clients")
    _comparer(res, _reference_65(grande), f"{quoi} sur 600 clients", "ORDER BY contract.")


# ============================================================================ Défi 66 : fonctions de fenêtre

# Des montants ENTIERS, pour démasquer la division entière ; insérés dans le désordre.
_FACTURES_66 = [
    (1, "P1", "2026-02-03", 250), (2, "P2", "2026-01-05", 100), (3, "P1", "2026-01-20", 200),
    (4, "P3", "2026-03-10", 350), (5, "P2", "2026-02-28", 150), (6, "P1", "2026-04-01", 420),
]
_ATTENDU_66 = [
    ("2026-01", 300, None, 300),
    ("2026-02", 400, 100 * 100 / 300, 700),
    ("2026-03", 350, -50 * 100 / 400, 1050),
    ("2026-04", 420, 70 * 100 / 350, 1470),
]


def _reference_66(tables: dict[str, pd.DataFrame]) -> list[tuple]:
    fa = tables["factures"]
    m = fa.groupby(fa["invoice_date"].str[:7])["amount"].sum().sort_index()
    out = pd.DataFrame({"mois": m.index, "chiffre_affaires": m.values})
    out["evolution_pct"] = out["chiffre_affaires"].pct_change() * 100
    out["ca_cumule"] = out["chiffre_affaires"].cumsum()
    return [(r.mois, r.chiffre_affaires, None if pd.isna(r.evolution_pct) else r.evolution_pct, r.ca_cumule)
            for r in out.itertuples(index=False)]


def verifier_requete_evolution_mensuelle(f) -> None:
    quoi = "requete_evolution_mensuelle"
    sql = _sql(f, f"{quoi}()")
    res = _executer(sql, _tables(factures=_FACTURES_66), quoi)
    _colonnes(res, ["mois", "chiffre_affaires", "evolution_pct", "ca_cumule"], quoi)
    if len(res) == len(_FACTURES_66):
        raise AssertionError(
            f"{quoi} : une ligne par facture (ou par jour) au lieu d'une par mois. Regroupez sur le mois : "
            "strftime('%Y-%m', invoice_date) AS mois … GROUP BY mois."
        )
    mois = [str(m) for m in res["mois"]]
    if sorted(mois) != [a[0] for a in _ATTENDU_66]:
        raise AssertionError(
            f"{quoi} : mois attendus {[a[0] for a in _ATTENDU_66]} (format 'AAAA-MM'), obtenu {mois}. "
            "strftime('%Y-%m', invoice_date) ; strftime('%m', …) seul confondrait janvier 2025 et janvier 2026."
        )
    lignes = _par(res, "mois")
    if any(not _meme(lignes[m]["chiffre_affaires"], ca) for m, ca, *_ in _ATTENDU_66):
        raise AssertionError(
            f"{quoi} : chiffre d'affaires attendu {[a[1] for a in _ATTENDU_66]}, obtenu "
            f"{[_lisible(lignes[a[0]]['chiffre_affaires']) for a in _ATTENDU_66]} (SUM(amount) par mois)."
        )
    if mois != sorted(mois):
        raise AssertionError(f"{quoi} : mois dans le désordre ({mois}) : terminez par ORDER BY mois.")

    evo = [lignes[a[0]]["evolution_pct"] for a in _ATTENDU_66]
    ref = [a[2] for a in _ATTENDU_66]
    lisible = [_lisible(e) for e in evo]
    if not _nul(evo[0]):
        if _nul(evo[-1]):
            raise AssertionError(
                f"{quoi} : évolutions {lisible} : vous comparez chaque mois au mois SUIVANT (le dernier n'a pas de "
                "voisin). LAG lit la ligne précédente dans l'ordre de la fenêtre : OVER (ORDER BY mois), croissant ; "
                "LEAD lirait la suivante."
            )
        raise AssertionError(
            f"{quoi} : le premier mois a une évolution de {lisible[0]} au lieu de NULL. Il n'a pas de mois précédent : "
            "« +0 % » affirmerait une stabilité que personne n'a mesurée. Laissez LAG renvoyer NULL (pas de "
            "COALESCE, pas de valeur par défaut)."
        )
    if all(_meme(e, r / 100) for e, r in zip(evo[1:], ref[1:])):
        raise AssertionError(
            f"{quoi} : évolutions {lisible} : ce sont des proportions. La colonne s'appelle evolution_PCT : "
            "multipliez par 100 (+33,3 et non +0,333)."
        )
    if all(not _nul(e) and float(e).is_integer() for e in evo[1:]) and not all(_meme(e, r) for e, r in zip(evo[1:], ref[1:])):
        raise AssertionError(
            f"{quoi} : évolutions {lisible} au lieu de {[_lisible(r) for r in ref]} : des nombres entiers, signe de la "
            "division ENTIÈRE (les montants sont des entiers, et 10000 / 300 vaut 33 en SQL). Commencez le calcul "
            "par 100.0 * (…) pour passer en décimal."
        )
    if all(_meme(e, 100 * (a[1] - p[1]) / a[1]) for e, a, p in zip(evo[1:], _ATTENDU_66[1:], _ATTENDU_66)):
        raise AssertionError(
            f"{quoi} : évolutions {lisible} : vous divisez par le mois COURANT. Une évolution se rapporte au point de "
            "départ : (ca - ca_précédent) / ca_précédent."
        )
    if any(not _meme(e, r) for e, r in zip(evo, ref)):
        raise AssertionError(
            f"{quoi} : évolutions attendues {[_lisible(r) for r in ref]} %, obtenu {lisible}. "
            "100.0 * (ca - LAG(ca) OVER (ORDER BY mois)) / LAG(ca) OVER (ORDER BY mois)."
        )
    cumul = [_lisible(lignes[a[0]]["ca_cumule"]) for a in _ATTENDU_66]
    if len(set(cumul)) == 1:
        raise AssertionError(
            f"{quoi} : ca_cumule vaut {cumul[0]} chaque mois : c'est le total général. Sans ORDER BY dans OVER (), "
            "la fenêtre est toute la table. SUM(ca) OVER (ORDER BY mois) cumule du premier mois au mois courant."
        )
    if any(not _meme(c, a[3]) for c, a in zip(cumul, _ATTENDU_66)):
        raise AssertionError(
            f"{quoi} : ca_cumule attendu {[a[3] for a in _ATTENDU_66]}, obtenu {cumul}. SUM(ca) OVER (ORDER BY mois), "
            "sur le chiffre d'affaires MENSUEL (après le GROUP BY)."
        )
    _comparer(res, _ATTENDU_66, quoi, "ORDER BY mois.")

    grande = _grande_base()
    res = _executer(sql, grande, f"{quoi} sur 600 clients")
    _comparer(res, _reference_66(grande), f"{quoi} sur 600 clients", "ORDER BY mois.")


# ============================================================================ Défi 67 : îlots et trous

_APPELS_67 = [
    # S1 : nov., déc., janv. (2 appels), févr. → 4 mois d'affilée, à cheval sur deux années
    (1, "S1", "2025-11-03", "Panne"), (2, "S1", "2025-12-15", "Panne"), (3, "S1", "2026-01-02", "Panne"),
    (4, "S1", "2026-01-20", "Facturation"), (5, "S1", "2026-02-07", "Panne"),
    # S2 : janv., mars, avr. → 3 mois différents, mais au plus 2 consécutifs
    (6, "S2", "2026-01-10", "Offre"), (7, "S2", "2026-03-10", "Panne"), (8, "S2", "2026-04-10", "Panne"),
    # S3 : trois appels, un seul mois
    (9, "S3", "2026-05-01", "Panne"), (10, "S3", "2026-05-02", "Panne"), (11, "S3", "2026-05-03", "Panne"),
    # S4 : juin, juil., août 2025
    (12, "S4", "2025-06-01", "Facturation"), (13, "S4", "2025-07-01", "Panne"), (14, "S4", "2025-08-01", "Panne"),
    # S5 : déc. 2024 puis janv. 2025
    (15, "S5", "2024-12-05", "Panne"), (16, "S5", "2025-01-05", "Panne"),
    # Des numéros inconnus, trois mois d'affilée
    (17, None, "2026-01-01", "Offre"), (18, None, "2026-02-01", "Offre"), (19, None, "2026-03-01", "Offre"),
]
_SERIES_67 = {"S1": 4, "S4": 3, "S2": 2, "S5": 2, "S3": 1}


def _attendu_67(mois_min: int) -> list[tuple]:
    garde = [(c, n) for c, n in _SERIES_67.items() if n >= mois_min]
    return sorted(garde, key=lambda x: (-x[1], x[0]))


def _reference_67(tables: dict[str, pd.DataFrame], mois_min: int) -> list[tuple]:
    a = tables["appels"].dropna(subset=["customer_id"])
    meilleurs = {}
    for client, dates in a.groupby("customer_id")["call_date"]:
        nums = sorted({int(d[:4]) * 12 + int(d[5:7]) for d in dates})
        meilleur = courant = 1
        for avant, apres in zip(nums, nums[1:]):
            courant = courant + 1 if apres == avant + 1 else 1
            meilleur = max(meilleur, courant)
        meilleurs[client] = meilleur
    return sorted(((c, n) for c, n in meilleurs.items() if n >= mois_min), key=lambda x: (-x[1], x[0]))


def verifier_requete_series_appels(f) -> None:
    quoi = "requete_series_appels"
    sql = _sql(f, f"{quoi}()")
    tables = _tables(appels=_APPELS_67)
    res = _executer(sql, tables, quoi, params={"mois_min": 1})
    _colonnes(res, ["customer_id", "plus_longue_serie"], quoi)
    if any(_nul(c) for c in res["customer_id"]):
        raise AssertionError(
            f"{quoi} : une ligne a un customer_id NULL : ce sont les appels de numéros inconnus, regroupés comme s'ils "
            "venaient d'un même client. Écartez-les : WHERE customer_id IS NOT NULL."
        )
    ids = list(res["customer_id"])
    if len(ids) != len(set(ids)):
        raise AssertionError(
            f"{quoi} : des clients apparaissent plusieurs fois ({ids}) : une ligne par client, avec la plus longue de "
            "ses séries (MAX de la longueur des îlots, GROUP BY customer_id)."
        )
    res2 = _executer(sql, tables, quoi, params={"mois_min": 2})
    res3 = _executer(sql, tables, quoi, params={"mois_min": 3})
    if set(res["customer_id"]) == set(res2["customer_id"]) == set(res3["customer_id"]):
        raise AssertionError(
            f"{quoi} : avec :mois_min = 1, 2 ou 3, les mêmes clients ({sorted(set(ids))}). Filtrez avec HAVING "
            "MAX(longueur) >= :mois_min (HAVING filtre APRÈS le GROUP BY ; WHERE, avant), sans écrire la valeur en dur."
        )
    obtenu = {r["customer_id"]: r["plus_longue_serie"] for r in res.to_dict("records")}
    if set(obtenu) != set(_SERIES_67):
        raise AssertionError(
            f"{quoi} : avec :mois_min = 1, attendu les clients {sorted(_SERIES_67)}, obtenu {sorted(obtenu)}."
        )
    lisible = {c: _lisible(n) for c, n in sorted(obtenu.items())}
    if _meme(obtenu["S1"], 5) or _meme(obtenu["S3"], 3):
        raise AssertionError(
            f"{quoi} : séries obtenues {lisible} : vous comptez des APPELS (S3 a appelé 3 fois, mais en un seul mois). "
            "Il faut compter des MOIS consécutifs."
        )
    if _meme(obtenu["S2"], 3) and _meme(obtenu["S1"], 4):
        raise AssertionError(
            f"{quoi} : S2 obtient 3, alors qu'il a appelé en janvier, mars et avril : 3 mois DIFFÉRENTS, mais au plus "
            "2 CONSÉCUTIFS (février manque). Compter les mois distincts ne suffit pas : numérotez les mois, puis "
            "regroupez ceux dont « num_mois - ROW_NUMBER() » est identique (méthode des îlots)."
        )
    if _meme(obtenu["S1"], 2) and _meme(obtenu["S5"], 1):
        raise AssertionError(
            f"{quoi} : séries obtenues {lisible} : décembre et janvier ne se suivent plus (S5 : déc. 2024 puis "
            "janv. 2025 donne 1 au lieu de 2). Le mois seul (12 puis 1) recommence chaque année : numérotez avec "
            "année × 12 + mois, par exemple CAST(strftime('%Y', call_date) AS INTEGER) * 12 + "
            "CAST(strftime('%m', call_date) AS INTEGER)."
        )
    if _meme(obtenu["S1"], 3):
        raise AssertionError(
            f"{quoi} : S1 obtient 3 au lieu de 4 : il a appelé DEUX fois en janvier. Sans SELECT DISTINCT sur "
            "(client, mois), ces deux appels prennent deux rangs, et « num_mois - rang » change au milieu de la "
            "série. Dédoublonnez les mois avant ROW_NUMBER()."
        )
    faux = {c: (_lisible(obtenu[c]), n) for c, n in _SERIES_67.items() if not _meme(obtenu[c], n)}
    if faux:
        raise AssertionError(f"{quoi} : plus longues séries (obtenu, attendu) incorrectes pour {faux}.")
    _comparer(res, _attendu_67(1), f"{quoi} avec :mois_min = 1",
              "Les plus longues séries d'abord, puis par identifiant : ORDER BY plus_longue_serie DESC, customer_id.")

    _comparer(res2, _attendu_67(2), f"{quoi} avec :mois_min = 2", "ORDER BY plus_longue_serie DESC, customer_id.")
    _comparer(res3, _attendu_67(3), f"{quoi} avec :mois_min = 3", "ORDER BY plus_longue_serie DESC, customer_id.")

    grande = _grande_base()
    for mois_min in (2, 3):
        res = _executer(sql, grande, f"{quoi} sur 600 clients", params={"mois_min": mois_min})
        _comparer(res, _reference_67(grande, mois_min), f"{quoi} sur 600 clients avec :mois_min = {mois_min}",
                  "ORDER BY plus_longue_serie DESC, customer_id.")


# ============================================================================ La série

SERIE = Serie(
    code="10",
    titre="SQL pour l'analyse : NULL, jointures, fenêtres",
    module_eleve="defis.serie_10",
    module_corrige="defis.corrige_serie_10",
    defis=(
        Defi(62, "requete_churn_par_region", "🟢", "1.2", verifier_requete_churn_par_region),
        Defi(63, "requete_actifs_sans_appel", "🟢", "—", verifier_requete_actifs_sans_appel),
        Defi(64, "requete_appels_recents", "🟡", "—", verifier_requete_appels_recents),
        Defi(65, "requete_bilan_par_contrat", "🟡", "—", verifier_requete_bilan_par_contrat),
        Defi(66, "requete_evolution_mensuelle", "🟡", "—", verifier_requete_evolution_mensuelle),
        Defi(67, "requete_series_appels", "🔴", "—", verifier_requete_series_appels),
    ),
)

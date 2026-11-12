"""Vérifications automatiques de la série 00 (remise à niveau).

Les messages visent les erreurs que font TOUS les débutants : `print` au lieu de `return`,
`<=` au lieu de `<`, `and` au lieu de `&`… Les reconnaître vite, c'est déjà progresser.
"""

from __future__ import annotations

import ast
import inspect
import io
import math
import textwrap
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from .verif import Defi, Serie


def _appeler(f, *args, quoi: str, **kwargs):
    """Appelle f et transforme les erreurs de débutant les plus courantes en conseils."""
    sortie = io.StringIO()
    try:
        with redirect_stdout(sortie):
            resultat = f(*args, **kwargs)
    except ZeroDivisionError:
        raise AssertionError(
            f"{quoi} : division par zéro (ZeroDivisionError). Traitez ce cas AVANT de diviser, "
            "avec un `if ... : return ...` en début de fonction."
        ) from None
    if resultat is None:
        if sortie.getvalue().strip():
            raise AssertionError(
                f"{quoi} : la fonction AFFICHE « {sortie.getvalue().strip()[:60]} » avec print() mais ne "
                "RENVOIE rien. print() montre une valeur à l'écran ; return la donne au programme. "
                "Remplacez print(...) par return ..."
            )
        raise AssertionError(f"{quoi} : la fonction a retourné None. Avez-vous oublié `return` ?")
    return resultat


def _nombre(obtenu, quoi: str) -> float:
    assert not isinstance(obtenu, str), (
        f"{quoi} : vous renvoyez du texte ({obtenu!r}). Renvoyez un nombre : on pourra toujours "
        "l'afficher joliment plus tard, mais on ne calcule pas avec du texte."
    )
    assert isinstance(obtenu, (int, float, np.integer, np.floating)), (
        f"{quoi} : attendu un nombre, obtenu {type(obtenu).__name__}"
    )
    return float(obtenu)


def _proche(obtenu, attendu: float, quoi: str, tol: float = 1e-9) -> None:
    valeur = _nombre(obtenu, quoi)
    assert math.isclose(valeur, attendu, rel_tol=tol, abs_tol=tol), f"{quoi} : attendu {attendu}, obtenu {valeur}"


# ============================================================================ Bases de Python


def verifier_taux_en_pourcentage(f) -> None:
    res = _nombre(_appeler(f, 35, 100, quoi="taux_en_pourcentage(35, 100)"), "taux_en_pourcentage(35, 100)")
    if math.isclose(res, 0.35):
        raise AssertionError(
            "taux_en_pourcentage(35, 100) vaut 0.35 : c'est une proportion. Un POURCENTAGE se "
            "multiplie par 100 → 35.0"
        )
    _proche(res, 35.0, "taux_en_pourcentage(35, 100)")

    tiers = _nombre(_appeler(f, 1, 3, quoi="taux_en_pourcentage(1, 3)"), "taux_en_pourcentage(1, 3)")
    if tiers == 33.0:
        raise AssertionError("taux_en_pourcentage(1, 3) vaut 33.0 au lieu de 33.3 : // est la division ENTIÈRE. Utilisez /")
    if math.isclose(tiers, 100 / 3, rel_tol=1e-6):
        raise AssertionError("taux_en_pourcentage(1, 3) vaut 33.333… : arrondissez à 1 décimale avec round(x, 1) → 33.3")
    _proche(tiers, 33.3, "taux_en_pourcentage(1, 3)")
    _proche(_appeler(f, 2, 3, quoi="taux_en_pourcentage(2, 3)"), 66.7, "taux_en_pourcentage(2, 3) (arrondi au plus proche)")
    _proche(_appeler(f, 0, 0, quoi="taux_en_pourcentage(0, 0)"), 0.0, "taux_en_pourcentage(0, 0) (aucun client)")


def verifier_segment_client(f) -> None:
    attendus = {0: "nouveau", 5: "nouveau", 6: "installé", 12: "installé", 23: "installé", 24: "fidèle", 100: "fidèle"}
    for mois, attendu in attendus.items():
        obtenu = _appeler(f, mois, quoi=f"segment_client({mois})")
        if isinstance(obtenu, str) and obtenu != attendu and obtenu.strip().lower().replace("é", "e") == attendu.replace("é", "e"):
            raise AssertionError(
                f"segment_client({mois}) renvoie {obtenu!r} au lieu de {attendu!r} : presque ! "
                "Le texte doit être EXACTEMENT celui demandé (minuscules, accent, sans espace). "
                "Pour un programme, « installe » et « installé » sont deux mots différents."
            )
        if mois in (6, 24) and obtenu != attendu:
            raise AssertionError(
                f"segment_client({mois}) renvoie {obtenu!r} au lieu de {attendu!r}. {mois} mois n'est PAS "
                f"strictement inférieur à {mois} : vérifiez vos bornes (< ou <= ?)."
            )
        assert obtenu == attendu, f"segment_client({mois}) : attendu {attendu!r}, obtenu {obtenu!r}"
    try:
        f(-1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "segment_client(-1) devrait lever une erreur : une ancienneté négative est impossible. "
            "Écrivez `raise ValueError(\"...\")` : mieux vaut planter clairement que classer -1 en « nouveau »."
        )


def verifier_compter_par_contrat(f) -> None:
    contrats = ["Mensuel", "Annuel", "Mensuel", "Biannuel", "Mensuel"]
    copie = list(contrats)
    try:
        res = _appeler(f, contrats, quoi="compter_par_contrat")
    except KeyError as exc:
        raise AssertionError(
            f"KeyError {exc} : vous ajoutez 1 à une clé qui n'existe pas encore dans le dictionnaire. "
            "Utilisez compte.get(contrat, 0) + 1, ou testez d'abord `if contrat in compte:`."
        ) from None
    assert contrats == copie, "Vous avez modifié la liste reçue : une fonction de calcul ne doit pas toucher à ses entrées"
    assert isinstance(res, dict), f"Attendu un dictionnaire {{contrat: nombre}}, obtenu {type(res).__name__}"
    attendu = {"Mensuel": 3, "Annuel": 1, "Biannuel": 1}
    if set(res) == set(attendu) and all(v == 1 for v in res.values()):
        raise AssertionError(
            "Chaque contrat est compté 1 fois au lieu de 3 pour « Mensuel » : votre dictionnaire est-il "
            "recréé à chaque tour de boucle ? Créez-le une seule fois, AVANT le `for`."
        )
    assert dict(res) == attendu, f"Attendu {attendu}, obtenu {dict(res)}"
    vide = _appeler(f, [], quoi="compter_par_contrat([])")
    assert dict(vide) == {}, f"Pour une liste vide, retournez un dictionnaire vide {{}}, obtenu {vide!r}"


def verifier_clients_a_risque(f) -> None:
    clients = [
        {"id": "C001", "proba": 0.30},
        {"id": "C002", "proba": 0.91},
        {"id": "C003", "proba": 0.50},
        {"id": "C004", "proba": 0.72},
    ]
    copie = [dict(c) for c in clients]
    res = _appeler(f, clients, quoi="clients_a_risque")
    assert clients == copie, (
        "La liste reçue a été modifiée (triée ou vidée ?). `liste.sort()` trie EN PLACE la liste de "
        "l'appelant ; `sorted(liste, ...)` crée une nouvelle liste. Utilisez sorted()."
    )
    assert isinstance(res, list), f"Attendu une liste d'identifiants, obtenu {type(res).__name__}"
    assert all(isinstance(x, str) for x in res), (
        f"Renvoyez seulement les identifiants (['C002', ...]), pas les dictionnaires complets — obtenu {res[:2]}"
    )
    if sorted(res) == ["C002", "C004"]:
        raise AssertionError(
            "C003 a une probabilité de 0.50, PILE au seuil : il est à risque (proba >= seuil). "
            "Utilisez >= et non >."
        )
    assert sorted(res) == ["C002", "C003", "C004"], f"Attendu les clients C002, C003, C004 (proba >= 0.5), obtenu {res}"
    if res == ["C003", "C004", "C002"]:
        raise AssertionError("Ordre inversé : les PLUS risqués d'abord. Ajoutez reverse=True à sorted().")
    assert res == ["C002", "C004", "C003"], (
        f"Attendu ['C002', 'C004', 'C003'] (du plus risqué au moins risqué), obtenu {res}. "
        "Triez sur la probabilité : sorted(..., key=lambda c: c['proba'], reverse=True)"
    )
    assert _appeler(f, clients, seuil=0.8, quoi="clients_a_risque(seuil=0.8)") == ["C002"], (
        "Avec seuil=0.8, seul C002 est à risque : utilisez-vous bien le paramètre `seuil` (et pas 0.5 en dur) ?"
    )


def verifier_facture_moyenne(f) -> None:
    # Les cas sont choisis pour isoler les bugs un par un, du premier au dernier.
    une = _nombre(_appeler(f, [40.0], quoi="facture_moyenne([40.0])"), "facture_moyenne([40.0])")
    if une != 40.0:
        raise AssertionError(
            f"🐞 Avec une seule facture [40.0], la moyenne vaut 40.0 — obtenu {une}. La première facture "
            "est-elle bien lue ? En Python, le premier élément d'une liste a l'indice 0 : regardez où commence range()."
        )
    trois = _nombre(_appeler(f, [10.0, 20.0, 60.0], quoi="facture_moyenne([10, 20, 60])"), "facture_moyenne([10, 20, 60])")
    if math.isclose(trois, 20.0):
        raise AssertionError(
            "🐞 La moyenne de [10, 20, 60] vaut 30.0 — obtenu 20.0 = 60 / 3 : `total` ne garde que la "
            "DERNIÈRE facture. Pour additionner, il faut `total += ...` (ou total = total + ...), pas `total = ...`."
        )
    _proche(trois, 30.0, "🐞 facture_moyenne([10, 20, 60])")
    _proche(_appeler(f, [], quoi="🐞 facture_moyenne([]) (liste vide)"), 0.0, "🐞 facture_moyenne([]) (liste vide)")


# ============================================================================ Premiers pas en pandas


def _petit_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "contract": ["Mensuel", "Mensuel", "Annuel", "Mensuel", "Biannuel", "Mensuel"],
            "tenure_months": [3, 12, 4, 6, 2, 30],
            "churn": [1, 0, 0, 1, 0, 0],
        }
    )


def verifier_filtrer_clients(f) -> None:
    df = _petit_df()
    avant = df.copy()
    try:
        res = _appeler(f, df, "Mensuel", 6, quoi="filtrer_clients")
    except ValueError as exc:
        if "ambiguous" in str(exc):
            raise AssertionError(
                "« The truth value of a Series is ambiguous » : en pandas, on combine deux conditions avec "
                "& (et) ou | (ou), jamais `and` / `or`. Exemple : (df['a'] == x) & (df['b'] <= y)"
            ) from None
        raise
    except TypeError:
        raise AssertionError(
            "TypeError en combinant les conditions : mettez CHAQUE condition entre parenthèses. "
            "& est calculé avant == et <= : (df['contract'] == contrat) & (df['tenure_months'] <= anciennete_max)"
        ) from None
    pd.testing.assert_frame_equal(df, avant, obj="df d'entrée (il ne doit pas changer)")
    assert isinstance(res, pd.DataFrame), f"Attendu un DataFrame (les lignes retenues), obtenu {type(res).__name__}"
    ids = list(res["customer_id"])
    if ids == ["C1"]:
        raise AssertionError("C4 a exactement 6 mois d'ancienneté : il doit être gardé (<= anciennete_max, pas <).")
    assert ids == ["C1", "C4"], (
        f"Contrat Mensuel ET ancienneté <= 6 → attendu ['C1', 'C4'], obtenu {ids}. Les DEUX conditions doivent être vraies."
    )
    autre = list(_appeler(f, df, "Annuel", 100, quoi="filtrer_clients(df, 'Annuel', 100)")["customer_id"])
    assert autre == ["C3"], f"filtrer_clients(df, 'Annuel', 100) : attendu ['C3'], obtenu {autre}. Utilisez-vous `contrat` ?"


def verifier_taux_par_groupe_a_la_main(f) -> None:
    try:  # on lit le CODE de la fonction (pas sa docstring ni les commentaires, qui peuvent citer groupby)
        arbre = ast.parse(textwrap.dedent(inspect.getsource(f)))
        appels = {n.attr for n in ast.walk(arbre) if isinstance(n, ast.Attribute)}
    except (OSError, TypeError, SyntaxError):
        appels = set()
    if "groupby" in appels:
        raise AssertionError(
            "Ce défi se fait SANS groupby : le but est de comprendre ce que groupby fait pour vous. "
            "Boucle sur df[colonne].unique(), filtre, moyenne. (groupby servira juste après, pour comparer.)"
        )
    df = _petit_df()
    res = _appeler(f, df, "contract", quoi="taux_par_groupe_a_la_main")
    assert isinstance(res, dict), f"Attendu un dictionnaire {{modalité: taux}}, obtenu {type(res).__name__}"
    assert set(res) == {"Mensuel", "Annuel", "Biannuel"}, f"Clés attendues : Mensuel, Annuel, Biannuel — obtenu {list(res)}"
    for cle, attendu in {"Mensuel": 0.5, "Annuel": 0.0, "Biannuel": 0.0}.items():
        _proche(res[cle], attendu, f"taux de churn du groupe {cle}")
    from churn.data import generate_customers

    clients = generate_customers(n=800, seed=3)
    vrai = _appeler(f, clients, "internet_service", quoi="taux_par_groupe_a_la_main sur 800 clients")
    reference = clients.groupby("internet_service")["churn"].mean()
    for modalite, attendu in reference.items():
        _proche(vrai.get(modalite), attendu, f"taux de churn de « {modalite} » (800 clients)")


def verifier_nettoyer_contrats(f) -> None:
    sale = pd.Series(
        [" mensuel", "MENSUEL", "Annuel ", "1 an", "Bi-Annuel", "2 ans", "mois", None],
        index=[10, 11, 12, 13, 14, 15, 16, 17],
    )
    avant = sale.copy()
    try:
        res = _appeler(f, sale, quoi="nettoyer_contrats")
    except ValueError as exc:
        if "none" in str(exc).lower() or "nan" in str(exc).lower():
            cause = ("Une valeur MANQUANTE (None/NaN) n'est pas une valeur inconnue : elle doit rester "
                     "manquante, sans erreur. Une valeur est inconnue si elle existait mais n'a pas été traduite.")
        else:
            cause = ("Ces valeurs sont pourtant toutes dans SYNONYMES une fois nettoyées : enlevez les espaces "
                     "(.str.strip()) ET passez en minuscules (.str.lower()) AVANT de traduire.")
        raise AssertionError(f"ValueError levée à tort ({exc}). {cause}") from None
    except AttributeError as exc:
        raise AssertionError(
            f"AttributeError ({exc}) : sur une Series, les méthodes de texte passent par `.str` : "
            "contrats.str.strip().str.lower()"
        ) from None
    pd.testing.assert_series_equal(sale, avant, obj="Series d'entrée (elle ne doit pas changer)")
    assert isinstance(res, pd.Series), f"Attendu une pd.Series, obtenu {type(res).__name__}"
    assert res.index.equals(sale.index), "Le résultat doit garder le même index que l'entrée (pas de reset_index)"
    attendu = ["Mensuel", "Mensuel", "Annuel", "Annuel", "Biannuel", "Biannuel", "Mensuel"]
    obtenu = list(res.iloc[:7])
    if obtenu != attendu:
        i = next(i for i, (o, a) in enumerate(zip(obtenu, attendu)) if o != a)
        raise AssertionError(
            f"Attendu {attendu}, obtenu {obtenu}. Par exemple {sale.iloc[i]!r} devrait devenir {attendu[i]!r}. "
            "Avez-vous enlevé les espaces (.str.strip()) ET mis en minuscules (.str.lower()) avant d'utiliser SYNONYMES ?"
        )
    assert pd.isna(res.iloc[7]), f"La valeur manquante doit rester manquante (NaN), obtenu {res.iloc[7]!r}"

    try:
        f(pd.Series(["Mensuel", "trimestriel", "annuel", "Trimestriel"]))
    except ValueError as exc:
        assert "trimestriel" in str(exc).lower(), (
            f"Le message d'erreur doit citer la valeur inconnue pour aider à corriger — obtenu : {exc}"
        )
    else:
        raise AssertionError(
            "« trimestriel » n'est pas dans SYNONYMES : levez une ValueError qui la cite. Un contrat inconnu "
            "transformé en silence en NaN, c'est un client qui disparaît des statistiques sans que personne ne le sache."
        )


SERIE = Serie(
    code="00",
    titre="Remise à niveau : bases de Python et premiers pas en pandas",
    module_eleve="defis.serie_00",
    module_corrige="defis.corrige_serie_00",
    defis=(
        Defi(30, "taux_en_pourcentage", "🟢", "—", verifier_taux_en_pourcentage),
        Defi(31, "segment_client", "🟢", "—", verifier_segment_client),
        Defi(32, "compter_par_contrat", "🟢", "—", verifier_compter_par_contrat),
        Defi(33, "clients_a_risque", "🟢", "—", verifier_clients_a_risque),
        Defi(34, "facture_moyenne", "🟢", "—", verifier_facture_moyenne),
        Defi(35, "filtrer_clients", "🟢", "—", verifier_filtrer_clients),
        Defi(36, "taux_par_groupe_a_la_main", "🟡", "1.2", verifier_taux_par_groupe_a_la_main),
        Defi(37, "nettoyer_contrats", "🟡", "—", verifier_nettoyer_contrats),
    ),
)

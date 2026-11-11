"""Vérifications automatiques de la série 02 (explorer les données).

Lire ces tests est aussi un bon exercice : ils montrent les cas limites auxquels penser.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .checks_serie_01 import _proche
from .verif import Defi, Serie

COLONNES = ["contract", "payment_method"]


@lru_cache(maxsize=1)
def _clients_cache() -> pd.DataFrame:
    from churn.data import generate_customers

    return generate_customers(n=3000, seed=7)


def _clients() -> pd.DataFrame:
    return _clients_cache().copy()  # une copie : votre code ne peut pas abîmer les données des autres tests


def _dataframe(obtenu, quoi: str) -> pd.DataFrame:
    assert obtenu is not None, f"{quoi} : la fonction a retourné None. Avez-vous oublié `return` ?"
    if isinstance(obtenu, pd.Series) and obtenu.dtype == bool:
        raise AssertionError(f"{quoi} : vous avez retourné le masque booléen. Retournez les lignes : df[masque]")
    assert isinstance(obtenu, pd.DataFrame), f"{quoi} : attendu un pd.DataFrame, obtenu {type(obtenu).__name__}"
    return obtenu


# =========================================================================== Module 1 — Données


def verifier_lift(f) -> None:
    df = pd.DataFrame({"contract": ["A", "A", "B", "B"], "churn": [1, 1, 0, 0]})
    _proche(f(df, "contract", "A"), 2.0, "lift de A (taux du groupe 100 % ÷ taux global 50 %)")
    _proche(f(df, "contract", "B"), 0.0, "lift de B (personne ne part dans ce groupe)")

    df = pd.DataFrame({"contract": list("AAABBBBB"), "churn": [1, 1, 0, 1, 0, 0, 0, 0]})
    obtenu = f(df, "contract", "A")
    if obtenu is not None and np.isclose(float(obtenu), (2 / 3) / (1 / 5)):
        raise AssertionError(
            "Vous divisez par le taux des AUTRES clients. Le lift se calcule par rapport au taux "
            "GLOBAL (tous les clients, groupe compris)."
        )
    _proche(obtenu, (2 / 3) / (3 / 8), "lift de A (2/3 du groupe partent, 3/8 au total)")

    clients = _clients()
    assert float(f(clients, "contract", "Mensuel")) > 1.2, "Sur les vraies données, le contrat mensuel devrait avoir un lift > 1,2"
    assert float(f(clients, "tech_support", "Oui")) < 1, "Le support technique retient les clients : lift < 1 attendu"

    try:
        resultat = f(clients, "contract", "Hebdomadaire")
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"Modalité inexistante : attendu une ValueError, obtenu {type(exc).__name__}. "
            "Testez si le groupe est vide et levez `raise ValueError(...)` vous-même."
        ) from None
    raise AssertionError(
        f"Modalité inexistante (« Hebdomadaire ») : la fonction a retourné {resultat!r} au lieu de lever "
        "une ValueError. Un NaN silencieux finit toujours dans un tableau de bord…"
    )


def verifier_factures_hors_bornes(f) -> None:
    abime = _clients()
    abime.loc[10, "monthly_charges"] = 600.0
    abime.loc[20, "monthly_charges"] = -5.0
    abime.loc[30, "monthly_charges"] = 500.0  # pile sur la borne haute : accepté
    abime.loc[40, "monthly_charges"] = 0.0    # pile sur la borne basse : accepté
    abime.loc[50, "monthly_charges"] = np.nan
    avant = abime.copy()

    suspectes = _dataframe(f(abime), "factures_hors_bornes")
    pd.testing.assert_frame_equal(abime, avant, obj="df d'entrée (il ne doit pas changer)")
    trouvees = sorted(suspectes.index)
    if 30 in trouvees or 40 in trouvees:
        raise AssertionError(
            "0 € et 500 € pile sont acceptés : utilisez `<` et `>`, pas `<=` et `>=`. "
            f"Lignes signalées : {trouvees}"
        )
    if 50 in trouvees:
        raise AssertionError(
            "La ligne 50 a une facture manquante (NaN) : ce n'est pas une facture hors bornes. "
            "Attention, `~serie.between(...)` compte NaN comme hors bornes."
        )
    assert trouvees == [10, 20], f"Attendu les lignes [10, 20] (600 € et -5 €), obtenu {trouvees}"
    assert list(suspectes.columns) == list(abime.columns), "Gardez toutes les colonnes des lignes suspectes"

    clients = _clients()
    perso = _dataframe(f(clients, minimum=20, maximum=120), "factures_hors_bornes(minimum=20, maximum=120)")
    attendues = clients[(clients["monthly_charges"] < 20) | (clients["monthly_charges"] > 120)]
    assert len(attendues) > 0  # garde-fou du correcteur lui-même
    assert sorted(perso.index) == sorted(attendues.index), (
        f"Avec minimum=20 et maximum=120 : {len(attendues)} lignes attendues, {len(perso)} obtenues. "
        "Utilisez-vous bien les paramètres `minimum` et `maximum` ?"
    )


def verifier_churn_par_tranche_anciennete(f) -> None:
    petit = pd.DataFrame({"tenure_months": [1, 6, 7, 12, 13, 72], "churn": [1, 1, 0, 1, 0, 0]})
    resume = _dataframe(f(petit, bornes=(0, 6, 12, 72)), "churn_par_tranche_anciennete")
    assert list(resume.columns) == ["tranche", "clients", "taux_churn"], (
        f"Colonnes attendues ['tranche', 'clients', 'taux_churn'], obtenu {list(resume.columns)}. "
        "Pensez à reset_index() pour transformer l'index en colonne."
    )
    tranches = [str(t) for t in resume["tranche"]]
    assert tranches == ["1-6", "7-12", "13-72"], f"Étiquettes attendues ['1-6', '7-12', '13-72'], obtenu {tranches}"
    clients_par_tranche = [int(c) for c in resume["clients"]]
    assert clients_par_tranche == [2, 2, 2], (
        f"Attendu 2 clients par tranche, obtenu {clients_par_tranche}. Un client de 6 mois est dans "
        "« 1-6 » et un de 7 mois dans « 7-12 » : pd.cut inclut la borne droite (right=True)."
    )
    for attendu, obtenu, nom in zip([1.0, 0.5, 0.0], resume["taux_churn"], tranches):
        _proche(obtenu, attendu, f"taux de churn de la tranche {nom}")

    clients = _clients()
    complet = _dataframe(f(clients), "churn_par_tranche_anciennete (bornes par défaut)")
    tranches = [str(t) for t in complet["tranche"]]
    assert tranches == ["1-6", "7-12", "13-24", "25-48", "49-72"], f"Tranches par défaut inattendues : {tranches}"
    assert int(complet["clients"].sum()) == len(clients), (
        f"{int(complet['clients'].sum())} clients répartis sur {len(clients)} : chaque client doit être "
        "dans exactement une tranche. Vérifiez que les clients de 1 et 72 mois sont bien comptés."
    )
    assert complet["taux_churn"].iloc[0] > 2 * complet["taux_churn"].iloc[-1], (
        "Les nouveaux clients devraient partir bien plus que les anciens : vérifiez l'ordre des tranches"
    )


# ============================================================================== 🔴 Défi final


def verifier_segments_a_risque(f) -> None:
    colonnes_attendues = [*COLONNES, "clients", "taux_churn"]
    piege = pd.DataFrame({
        "contract": ["Rare"] * 3 + ["Courant"] * 100 + ["Limite"] * 50,
        "churn": [1] * 3 + [1] * 40 + [0] * 60 + [1] * 10 + [0] * 40,
    })
    res = _dataframe(f(piege, ["contract"], min_clients=50), "segments_a_risque")
    contrats = list(res["contract"]) if "contract" in res.columns else []
    if "Rare" in contrats:
        raise AssertionError(
            "Le segment « Rare » (3 clients, 100 % de churn) est dans le résultat : excluez les segments "
            "de moins de `min_clients` clients. Un taux mesuré sur 3 personnes ne veut rien dire."
        )
    assert "Limite" in contrats, (
        "Le segment « Limite » a exactement 50 clients avec min_clients=50 : il doit être conservé (>=, pas >)"
    )
    assert contrats == ["Courant", "Limite"], f"Attendu ['Courant', 'Limite'] (40 % puis 20 %), obtenu {contrats}"

    clients = _clients()
    segments = _dataframe(f(clients, COLONNES), "segments_a_risque(clients, ['contract', 'payment_method'])")
    assert list(segments.columns) == colonnes_attendues, (
        f"Colonnes attendues {colonnes_attendues}, obtenu {list(segments.columns)}"
    )
    assert len(segments) == 5, f"top=5 par défaut : attendu 5 segments, obtenu {len(segments)}"
    assert list(segments.index) == [0, 1, 2, 3, 4], "L'index doit être 0, 1, 2… : pensez à reset_index(drop=True)"
    assert segments["taux_churn"].is_monotonic_decreasing, "Triez du taux de churn le plus élevé au plus faible"

    premier = segments.iloc[0]
    groupe = clients[(clients["contract"] == premier["contract"])
                     & (clients["payment_method"] == premier["payment_method"])]
    assert int(premier["clients"]) == len(groupe), (
        f"Le segment {premier['contract']} × {premier['payment_method']} compte {len(groupe)} clients, "
        f"obtenu {premier['clients']}"
    )
    _proche(premier["taux_churn"], groupe["churn"].mean(), "taux de churn du premier segment")

    large = _dataframe(f(clients, COLONNES, min_clients=100, top=10), "segments_a_risque(min_clients=100, top=10)")
    assert (large["clients"] >= 100).all(), "Le paramètre `min_clients` est ignoré"
    assert large.iloc[0]["contract"] == "Mensuel", "Le segment le plus à risque devrait être un contrat mensuel"

    vide = _dataframe(f(clients, COLONNES, min_clients=10**6), "segments_a_risque(min_clients=1 000 000)")
    assert vide.empty, "Aucun segment n'a un million de clients : le résultat doit être vide"
    assert list(vide.columns) == colonnes_attendues, "Même vide, le résultat doit garder ses colonnes"


SERIE = Serie(
    code="02",
    titre="Explorer les données",
    module_eleve="defis.serie_02",
    module_corrige="defis.corrige_serie_02",
    defis=(
        Defi(11, "lift", "🟡", "—", verifier_lift),
        Defi(12, "factures_hors_bornes", "🟡", "1.3", verifier_factures_hors_bornes),
        Defi(13, "churn_par_tranche_anciennete", "🟡", "—", verifier_churn_par_tranche_anciennete),
        Defi(14, "segments_a_risque", "🔴", "—", verifier_segments_a_risque),
    ),
)

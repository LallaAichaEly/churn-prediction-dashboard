"""Garantit que chaque défi est faisable et que la correction automatique est fiable.

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py`) ; ils
vérifient le corrigé et le moteur de correction.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defis.verif import A_FAIRE, ECHEC, ERREUR, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

TOUS_LES_DEFIS = [(s, d) for s in series() for d in s.defis]


@pytest.mark.parametrize("serie,defi", TOUS_LES_DEFIS, ids=lambda x: getattr(x, "fonction", None) or x.code)
def test_le_corrige_reussit(serie, defi):
    import importlib

    resultat = evaluer_defi(importlib.import_module(serie.module_corrige), defi)
    assert resultat.statut == REUSSI, resultat.message


@pytest.mark.parametrize("serie", series(), ids=lambda s: s.code)
def test_chaque_fonction_existe_dans_le_fichier_eleve(serie):
    """Le fichier à compléter doit proposer toutes les fonctions (même si non codées)."""
    assert all(r.statut != ERREUR or "introuvable" not in r.message for r in evaluer(serie))


def _defi(nom):
    return next(d for _, d in TOUS_LES_DEFIS if d.fonction == nom)


def test_une_fonction_non_codee_est_a_faire():
    def pas_encore(*_args, **_kwargs):
        raise NotImplementedError

    module = SimpleNamespace(__name__="faux", taux_churn_par=pas_encore)
    assert evaluer_defi(module, _defi("taux_churn_par")).statut == A_FAIRE


def test_la_fuite_de_donnees_est_detectee():
    """Erreur la plus instructive de la série : standardiser avec les stats de train + test."""

    def avec_fuite(train, test):
        tout = pd.concat([train, test])
        m, s = tout.mean(), tout.std(ddof=0)
        return (train - m) / s, (test - m) / s

    resultat = evaluer_defi(SimpleNamespace(__name__="faux", standardiser=avec_fuite), _defi("standardiser"))
    assert resultat.statut == ECHEC and "fuite" in resultat.message


def test_oublier_le_return_donne_un_message_clair():
    module = SimpleNamespace(__name__="faux", cout_total=lambda *a, **k: None)
    resultat = evaluer_defi(module, _defi("cout_total"))
    assert resultat.statut == ECHEC and "return" in resultat.message


def test_un_plantage_est_signale_comme_erreur():
    module = SimpleNamespace(__name__="faux", lift_top=lambda *a, **k: 1 / 0)
    assert evaluer_defi(module, _defi("lift_top")).statut == ERREUR


# --------------------------------------------------------------------------- Série 02 : erreurs classiques


def _evaluer(nom, fonction):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: fonction}), _defi(nom))


def test_un_nan_silencieux_est_refuse():
    """lift() doit lever une ValueError sur une modalité inconnue, pas renvoyer NaN."""

    def sans_garde_fou(df, colonne, modalite):
        return float(df.loc[df[colonne] == modalite, "churn"].mean() / df["churn"].mean())

    resultat = _evaluer("lift", sans_garde_fou)
    assert resultat.statut == ECHEC and "ValueError" in resultat.message


def test_between_compte_les_nan_comme_hors_bornes():
    def avec_between(df, minimum=0.0, maximum=500.0):
        return df[~df["monthly_charges"].between(minimum, maximum)]

    resultat = _evaluer("factures_hors_bornes", avec_between)
    assert resultat.statut == ECHEC and "NaN" in resultat.message


def test_les_bornes_sont_acceptees():
    def bornes_exclues(df, minimum=0.0, maximum=500.0):
        return df[(df["monthly_charges"] <= minimum) | (df["monthly_charges"] >= maximum)]

    resultat = _evaluer("factures_hors_bornes", bornes_exclues)
    assert resultat.statut == ECHEC and "pile" in resultat.message


def test_les_tranches_incluent_la_borne_droite():
    from defis.corrige_serie_02 import churn_par_tranche_anciennete

    def borne_gauche(df, bornes=(0, 6, 12, 24, 48, 72)):
        decale = df.assign(tenure_months=df["tenure_months"] + 1)  # simule right=False
        return churn_par_tranche_anciennete(decale, bornes)

    resultat = _evaluer("churn_par_tranche_anciennete", borne_gauche)
    assert resultat.statut == ECHEC and "right=True" in resultat.message


def test_un_petit_segment_ne_doit_pas_gagner():
    """La leçon du défi 14 : sans filtre sur l'effectif, un segment de 3 clients prend la tête."""

    def sans_filtre(df, colonnes, min_clients=50, top=5):
        segments = df.groupby(colonnes)["churn"].agg(clients="size", taux_churn="mean").reset_index()
        return segments.sort_values("taux_churn", ascending=False).head(top).reset_index(drop=True)

    resultat = _evaluer("segments_a_risque", sans_filtre)
    assert resultat.statut == ECHEC and "min_clients" in resultat.message


def test_les_numeros_de_defis_sont_uniques_et_consecutifs():
    """`python scripts/defis.py --defi N` doit désigner un seul défi, sans trou dans la numérotation."""
    numeros = [d.numero for _, d in TOUS_LES_DEFIS]
    assert numeros == list(range(1, len(numeros) + 1))


# --------------------------------------------------------------------------- Série 03 : erreurs classiques


def test_le_seuil_cible_les_probas_egales():
    from defis.corrige_serie_03 import precision_rappel_par_seuil

    def strict(y_true, proba, seuils):
        decale = [s + 1e-9 for s in seuils]  # simule proba > seuil
        return precision_rappel_par_seuil(y_true, proba, decale).assign(seuil=list(seuils))

    resultat = _evaluer("precision_rappel_par_seuil", strict)
    assert resultat.statut == ECHEC and ">=" in resultat.message


def test_une_precision_non_definie_vaut_nan():
    """La leçon du défi 15 : un 0 inventé fausse la courbe précision-rappel."""
    from defis.corrige_serie_03 import precision_rappel_par_seuil

    def zero_par_defaut(y_true, proba, seuils):
        return precision_rappel_par_seuil(y_true, proba, seuils).fillna({"precision": 0.0})

    resultat = _evaluer("precision_rappel_par_seuil", zero_par_defaut)
    assert resultat.statut == ECHEC and "NaN" in resultat.message


def _calibration(y_true, proba, n_tranches, tranche):
    import numpy as np

    df = pd.DataFrame({"tranche": tranche, "p": np.asarray(proba, dtype=float), "y": np.asarray(y_true)})
    return (df.groupby("tranche", observed=True)
            .agg(clients=("y", "size"), proba_moyenne=("p", "mean"), taux_reel=("y", "mean")).reset_index())


def test_une_proba_de_1_reste_dans_la_derniere_tranche():
    import numpy as np

    def sans_borne(y_true, proba, n_tranches=10):
        return _calibration(y_true, proba, n_tranches, (np.asarray(proba) * n_tranches).astype(int))

    resultat = _evaluer("table_calibration", sans_borne)
    assert resultat.statut == ECHEC and "n'existe pas" in resultat.message


def test_les_tranches_incluent_leur_borne_gauche():
    import numpy as np

    def borne_droite(y_true, proba, n_tranches=10):
        bornes = np.linspace(0, 1, n_tranches + 1)
        tranche = pd.cut(np.asarray(proba), bornes, labels=False, include_lowest=True)  # ]a, b]
        return _calibration(y_true, proba, n_tranches, tranche)

    resultat = _evaluer("table_calibration", borne_droite)
    assert resultat.statut == ECHEC and "right=False" in resultat.message


def test_la_calibration_utilise_les_probas_predites():
    from defis.corrige_serie_03 import table_calibration

    def milieu(y_true, proba, n_tranches=10):
        table = table_calibration(y_true, proba, n_tranches)
        return table.assign(proba_moyenne=(table["tranche"] + 0.5) / n_tranches)

    resultat = _evaluer("table_calibration", milieu)
    assert resultat.statut == ECHEC and "MILIEU" in resultat.message


def test_les_ex_aequo_comptent_pour_un_demi():
    import numpy as np

    def sans_ex_aequo(y_true, proba):
        y_true, proba = np.asarray(y_true), np.asarray(proba)
        partis, restes = proba[y_true == 1], proba[y_true == 0]
        return float((partis[:, None] > restes[None, :]).mean())

    resultat = _evaluer("auc_par_paires", sans_ex_aequo)
    assert resultat.statut == ECHEC and "½" in resultat.message


def test_l_auc_compare_dans_le_bon_sens():
    from defis.corrige_serie_03 import auc_par_paires

    resultat = _evaluer("auc_par_paires", lambda y_true, proba: 1 - auc_par_paires(y_true, proba))
    assert resultat.statut == ECHEC and "RESTÉ" in resultat.message


def test_les_contributions_se_trient_en_valeur_absolue():
    """La leçon du défi 18 : ce qui retient un client compte autant que ce qui le pousse à partir."""
    from defis.corrige_serie_03 import contributions_client

    def tri_signe(pipeline, client, top=3):
        toutes = contributions_client(pipeline, client, top=None)
        trie = toutes.sort_values("contribution", ascending=False)
        return (trie if top is None else trie.head(top)).reset_index(drop=True)

    resultat = _evaluer("contributions_client", tri_signe)
    assert resultat.statut == ECHEC and "valeur absolue" in resultat.message


def test_les_contributions_utilisent_les_valeurs_transformees():
    import numpy as np

    def valeurs_brutes(pipeline, client, top=3):
        noms = pipeline.named_steps["prep"].get_feature_names_out()
        brut = np.ones(len(noms))  # tout sauf les valeurs transformées
        contrib = pd.Series(brut * pipeline[-1].coef_[0], index=noms)
        contrib = contrib.sort_values(key=abs, ascending=False)
        contrib = contrib if top is None else contrib.head(top)
        return contrib.rename_axis("variable").reset_index(name="contribution")

    resultat = _evaluer("contributions_client", valeurs_brutes)
    assert resultat.statut == ECHEC and "TRANSFORMÉES" in resultat.message


def test_les_contributions_marchent_avec_les_variables_derivees():
    from defis.corrige_serie_03 import contributions_client

    def seulement_prep(pipeline, client, top=3):
        prep = pipeline.named_steps["prep"]
        prep.transform(client)  # plante si une étape de variables dérivées précède "prep"
        return contributions_client(pipeline, client, top)

    resultat = _evaluer("contributions_client", seulement_prep)
    assert resultat.statut == ECHEC and "pipeline[:-1]" in resultat.message


def test_une_tranche_vide_ne_rend_pas_le_psi_infini():
    import numpy as np

    def sans_plancher(ref, act):
        ref, act = np.asarray(ref, dtype=float), np.asarray(act, dtype=float)
        ref, act = ref / ref.sum(), act / act.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum((act - ref) * np.log(act / ref)))

    resultat = _evaluer("psi_parts", sans_plancher)
    assert resultat.statut == ECHEC and "0,0001" in resultat.message


def test_le_psi_utilise_le_logarithme_naturel():
    import numpy as np

    def log10(ref, act):
        ref, act = np.asarray(ref, dtype=float), np.asarray(act, dtype=float)
        return float(np.sum((act - ref) * np.log10(act / ref)))

    resultat = _evaluer("psi_parts", log10)
    assert resultat.statut == ECHEC and "NATUREL" in resultat.message


def test_le_psi_accepte_des_effectifs():
    import numpy as np

    def sans_normaliser(ref, act):
        ref, act = np.clip(np.asarray(ref, dtype=float), 1e-4, None), np.clip(np.asarray(act, dtype=float), 1e-4, None)
        return float(np.sum((act - ref) * np.log(act / ref)))

    resultat = _evaluer("psi_parts", sans_normaliser)
    assert resultat.statut == ECHEC and "somme" in resultat.message

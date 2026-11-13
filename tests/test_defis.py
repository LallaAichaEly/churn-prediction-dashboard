"""Garantit que chaque défi est faisable et que la correction automatique est fiable.

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py`) ; ils
vérifient le corrigé et le moteur de correction.
"""

import random
import sys
import tempfile
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
    """`python scripts/defis.py --defi N` doit désigner un seul défi, sans trou dans la numérotation.

    (La série 00, ajoutée après coup, est affichée en premier : on vérifie donc l'ensemble des
    numéros, pas leur ordre d'affichage.)
    """
    numeros = [d.numero for _, d in TOUS_LES_DEFIS]
    assert sorted(numeros) == list(range(1, len(numeros) + 1))


def test_la_remise_a_niveau_est_proposee_en_premier():
    """Un·e débutant·e qui lance `python scripts/defis.py` doit tomber sur la série 00."""
    assert series()[0].code == "00"


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


# --------------------------------------------------------------------------- Série 04 : erreurs classiques


def test_la_classe_rare_recoit_le_plus_grand_poids():
    import numpy as np

    def a_l_envers(y):
        classes, effectifs = np.unique(np.asarray(y), return_counts=True)
        return {c.item(): e / effectifs.sum() for c, e in zip(classes, effectifs)}

    resultat = _evaluer("poids_classes", a_l_envers)
    assert resultat.statut == ECHEC and "MAJORITAIRE" in resultat.message


def test_les_poids_se_divisent_par_le_nombre_de_classes():
    import numpy as np

    def sans_k(y):
        classes, effectifs = np.unique(np.asarray(y), return_counts=True)
        return {c.item(): effectifs.sum() / e for c, e in zip(classes, effectifs)}

    resultat = _evaluer("poids_classes", sans_k)
    assert resultat.statut == ECHEC and "nombre de classes" in resultat.message


def test_le_nombre_de_classes_se_lit_dans_les_donnees():
    import numpy as np

    def deux_classes(y):
        classes, effectifs = np.unique(np.asarray(y), return_counts=True)
        return {c.item(): effectifs.sum() / (2 * e) for c, e in zip(classes, effectifs)}

    resultat = _evaluer("poids_classes", deux_classes)
    assert resultat.statut == ECHEC and "np.unique" in resultat.message


def _choisir_C_avec(regle):
    """Un choisir_C dont la table est juste mais qui choisit C avec une autre règle."""
    from defis.corrige_serie_04 import choisir_C

    def variante(X, y, valeurs_C, n_plis=5, seed=0):
        table, _ = choisir_C(X, y, valeurs_C, n_plis=n_plis, seed=seed)
        return table, regle(table, n_plis)

    return variante


def test_choisir_C_ne_prend_pas_le_meilleur_score_brut():
    """La leçon du défi 21 : quelques millièmes d'AUC d'avance, c'est du bruit."""
    regle = _choisir_C_avec(lambda t, k: float(t.loc[t["auc_moyenne"].idxmax(), "C"]))
    resultat = _evaluer("choisir_C", regle)
    assert resultat.statut == ECHEC and "meilleur score brut" in resultat.message


def test_choisir_C_prend_le_modele_le_plus_simple():
    import math

    def plus_grand(t, k):
        meilleur = t.loc[t["auc_moyenne"].idxmax()]
        ok = t["auc_moyenne"] >= meilleur["auc_moyenne"] - meilleur["auc_ecart_type"] / math.sqrt(k)
        return float(t.loc[ok, "C"].max())

    resultat = _evaluer("choisir_C", _choisir_C_avec(plus_grand))
    assert resultat.statut == ECHEC and "plus PETIT C" in resultat.message


def test_choisir_C_tolere_une_erreur_standard_pas_un_ecart_type():
    def ecart_type_entier(t, k):
        meilleur = t.loc[t["auc_moyenne"].idxmax()]
        return float(t.loc[t["auc_moyenne"] >= meilleur["auc_moyenne"] - meilleur["auc_ecart_type"], "C"].min())

    resultat = _evaluer("choisir_C", _choisir_C_avec(ecart_type_entier))
    assert resultat.statut == ECHEC and "√n_plis" in resultat.message


def test_choisir_C_detecte_une_evaluation_sur_les_plis_d_entrainement():
    from sklearn.model_selection import StratifiedKFold, cross_validate

    from churn.model import build_pipeline

    def note_sur_le_train(X, y, valeurs_C, n_plis=5, seed=0):
        cv = StratifiedKFold(n_splits=n_plis, shuffle=True, random_state=seed)
        lignes = []
        for C in valeurs_C:
            s = cross_validate(build_pipeline("logistic").set_params(clf__C=C), X, y, cv=cv,
                               scoring="roc_auc", return_train_score=True)["train_score"]
            lignes.append({"C": C, "auc_moyenne": s.mean(), "auc_ecart_type": s.std()})
        return pd.DataFrame(lignes), float(min(valeurs_C))

    resultat = _evaluer("choisir_C", note_sur_le_train)
    assert resultat.statut == ECHEC and "ENTRAÎNEMENT" in resultat.message


def test_choisir_C_detecte_un_C_non_applique():
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from churn.model import build_pipeline

    def oublie_C(X, y, valeurs_C, n_plis=5, seed=0):
        cv = StratifiedKFold(n_splits=n_plis, shuffle=True, random_state=seed)
        s = cross_val_score(build_pipeline("logistic"), X, y, cv=cv, scoring="roc_auc")
        table = pd.DataFrame({"C": list(valeurs_C), "auc_moyenne": s.mean(), "auc_ecart_type": s.std()})
        return table, float(min(valeurs_C))

    resultat = _evaluer("choisir_C", oublie_C)
    assert resultat.statut == ECHEC and "set_params" in resultat.message


def test_un_correctif_ne_rend_pas_un_modele_incompatible():
    resultat = _evaluer("versions_compatibles", lambda sauvegardee, installee: sauvegardee == installee)
    assert resultat.statut == ECHEC and "correctif" in resultat.message


def test_une_version_n_est_pas_un_nombre_decimal():
    def trois_caracteres(sauvegardee, installee):
        return bool(sauvegardee) and sauvegardee[:3] == installee[:3]

    resultat = _evaluer("versions_compatibles", trois_caracteres)
    assert resultat.statut == ECHEC and "pas un nombre décimal" in resultat.message


def test_un_modele_sans_version_ne_fait_pas_planter():
    def sans_garde_fou(sauvegardee, installee):
        return sauvegardee.split(".")[:2] == installee.split(".")[:2]

    resultat = _evaluer("versions_compatibles", sans_garde_fou)
    assert resultat.statut == ECHEC and "ancien modèle" in resultat.message


def test_predire_lot_appelle_le_modele_une_seule_fois():
    from defis.corrige_serie_04 import predire_lot

    def client_par_client(pipeline, clients, seuil=0.5):
        return [r for c in clients for r in predire_lot(pipeline, [c], seuil)]

    resultat = _evaluer("predire_lot", client_par_client)
    assert resultat.statut == ECHEC and "Vectorisez" in resultat.message


def test_predire_lot_renvoie_des_types_json():
    import numpy as np

    from churn.data import FEATURES

    def types_numpy(pipeline, clients, seuil=0.5):
        if not clients:
            return []
        probas = pipeline.predict_proba(pd.DataFrame(clients)[FEATURES])[:, 1]
        return [{"churn_probability": p, "at_risk": np.bool_(p >= seuil)} for p in probas]

    resultat = _evaluer("predire_lot", types_numpy)
    assert resultat.statut == ECHEC and "json.dumps" in resultat.message


def test_predire_lot_cible_les_probas_egales_au_seuil():
    from defis.corrige_serie_04 import predire_lot

    def strict(pipeline, clients, seuil=0.5):
        return [{**r, "at_risk": r["churn_probability"] > seuil} for r in predire_lot(pipeline, clients, seuil)]

    resultat = _evaluer("predire_lot", strict)
    assert resultat.statut == ECHEC and ">=" in resultat.message


def test_predire_lot_accepte_une_liste_vide():
    from churn.data import FEATURES

    def sans_garde_fou(pipeline, clients, seuil=0.5):
        probas = pipeline.predict_proba(pd.DataFrame(clients, columns=FEATURES))[:, 1]
        return [{"churn_probability": float(p), "at_risk": bool(p >= seuil)} for p in probas]

    resultat = _evaluer("predire_lot", sans_garde_fou)
    assert resultat.statut == ECHEC and "liste vide" in resultat.message


def test_predire_lot_refuse_un_client_incomplet():
    """La leçon du défi 23 : sans contrôle, un client sans contrat est prédit sans erreur."""
    from churn.data import FEATURES

    def sans_controle(pipeline, clients, seuil=0.5):
        if not clients:
            return []
        probas = pipeline.predict_proba(pd.DataFrame(clients).reindex(columns=FEATURES))[:, 1]
        return [{"churn_probability": float(p), "at_risk": bool(p >= seuil)} for p in probas]

    resultat = _evaluer("predire_lot", sans_controle)
    assert resultat.statut == ECHEC and "silencieuse" in resultat.message


def test_predire_lot_ne_modifie_pas_les_clients_recus():
    from defis.corrige_serie_04 import predire_lot

    def retire_l_identifiant(pipeline, clients, seuil=0.5):
        for c in clients:
            c.pop("customer_id", None)
        return predire_lot(pipeline, clients, seuil)

    resultat = _evaluer("predire_lot", retire_l_identifiant)
    assert resultat.statut == ECHEC and "appelant" in resultat.message


def _alerte_avec(masque, trier=True):
    def variante(rapport, seuil=0.25):
        trie = rapport.sort_values("psi", ascending=False, na_position="last") if trier else rapport
        variables = trie.loc[masque(trie, seuil), "feature"].tolist()
        return int(bool(variables)), variables

    return variante


def test_un_psi_de_0_25_pile_est_une_derive():
    resultat = _evaluer("alerte_derive", _alerte_avec(lambda t, s: (t["psi"] > s) | t["psi"].isna()))
    assert resultat.statut == ECHEC and ">=" in resultat.message


def test_un_psi_manquant_declenche_l_alerte():
    """La leçon du défi 24 : NaN >= 0.25 vaut False, et l'absence de mesure passerait pour un feu vert."""
    resultat = _evaluer("alerte_derive", _alerte_avec(lambda t, s: t["psi"] >= s))
    assert resultat.statut == ECHEC and "NaN" in resultat.message


def test_les_alertes_sont_triees():
    masque = lambda t, s: (t["psi"] >= s) | t["psi"].isna()  # noqa: E731
    resultat = _evaluer("alerte_derive", _alerte_avec(masque, trier=False))
    assert resultat.statut == ECHEC and "na_position" in resultat.message


def test_l_alerte_utilise_le_seuil_et_pas_le_statut():
    resultat = _evaluer("alerte_derive", _alerte_avec(lambda t, s: t["statut"] == "dérive"))
    assert resultat.statut == ECHEC and "seuil est ignoré" in resultat.message


def test_l_alerte_ne_modifie_pas_le_rapport():
    from defis.corrige_serie_04 import alerte_derive

    def ajoute_une_colonne(rapport, seuil=0.25):
        rapport["alerte"] = (rapport["psi"] >= seuil) | rapport["psi"].isna()  # écrit dans le DataFrame reçu
        return alerte_derive(rapport, seuil)

    resultat = _evaluer("alerte_derive", ajoute_une_colonne)
    assert resultat.statut == ECHEC and "modifié" in resultat.message


# --------------------------------------------------------------------------- Série 05 : erreurs classiques


def _sql_corrige(nom):
    import defis.corrige_serie_05 as corrige

    return getattr(corrige, nom)()


def test_sql_la_division_entiere_est_detectee():
    """La leçon du défi 25 : en SQL, 2 / 3 vaut 0 entre deux entiers."""
    sql = _sql_corrige("requete_churn_par_contrat").replace("AVG(churn)", "SUM(churn) / COUNT(*)")
    resultat = _evaluer("requete_churn_par_contrat", lambda: sql)
    assert resultat.statut == ECHEC and "ENTIÈRE" in resultat.message


def test_sql_sans_group_by():
    sql = "SELECT contract, COUNT(*) AS clients, AVG(churn) AS taux_churn FROM clients"
    resultat = _evaluer("requete_churn_par_contrat", lambda: sql)
    assert resultat.statut == ECHEC and "GROUP BY" in resultat.message


def test_sql_la_division_decimale_est_acceptee():
    sql = _sql_corrige("requete_churn_par_contrat").replace("AVG(churn)", "1.0 * SUM(churn) / COUNT(*)")
    assert _evaluer("requete_churn_par_contrat", lambda: sql).statut == REUSSI


def test_top_risque_un_limit_global_ne_suffit_pas():
    sql = "SELECT contract, 1 AS rang, customer_id, proba_churn FROM scores ORDER BY proba_churn DESC LIMIT :k"
    resultat = _evaluer("requete_top_risque", lambda: sql)
    assert resultat.statut == ECHEC and "PARTITION BY" in resultat.message


def test_top_risque_rank_donne_trop_de_clients():
    sql = _sql_corrige("requete_top_risque").replace("ROW_NUMBER()", "RANK()").replace(", customer_id ASC", "")
    resultat = _evaluer("requete_top_risque", lambda: sql)
    assert resultat.statut == ECHEC and "ROW_NUMBER" in resultat.message


def test_top_risque_les_ex_aequo_sont_departages():
    sql = _sql_corrige("requete_top_risque").replace(", customer_id ASC", "")
    resultat = _evaluer("requete_top_risque", lambda: sql)
    assert resultat.statut == ECHEC and "customer_id ASC" in resultat.message


def test_top_risque_k_n_est_pas_ecrit_en_dur():
    sql = _sql_corrige("requete_top_risque").replace(":k", "2")
    resultat = _evaluer("requete_top_risque", lambda: sql)
    assert resultat.statut == ECHEC and ":k" in resultat.message


def test_top_risque_explique_le_where_sur_une_fenetre():
    fenetre = "ROW_NUMBER() OVER (PARTITION BY contract ORDER BY proba_churn DESC, customer_id)"
    sql = f"SELECT contract, {fenetre} AS rang, customer_id, proba_churn FROM scores WHERE {fenetre} <= :k"
    resultat = _evaluer("requete_top_risque", lambda: sql)
    assert resultat.statut == ECHEC and "WITH" in resultat.message


def _wilson_avec(z_de):
    import math

    def variante(departs, clients, confiance=0.95):
        if clients <= 0 or not 0 <= departs <= clients:
            raise ValueError("entrées impossibles")
        z, p, n = z_de(confiance), departs / clients, clients
        d = 1 + z**2 / n
        centre, demi = (p + z**2 / (2 * n)) / d, z / d * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        return max(0.0, centre - demi), min(1.0, centre + demi)

    return variante


def test_wilson_refuse_l_intervalle_de_wald():
    import math
    from statistics import NormalDist

    def wald(departs, clients, confiance=0.95):
        z, p = NormalDist().inv_cdf(1 - (1 - confiance) / 2), departs / clients
        demi = z * math.sqrt(p * (1 - p) / clients)
        return p - demi, p + demi

    resultat = _evaluer("intervalle_wilson", wald)
    assert resultat.statut == ECHEC and "Wald" in resultat.message


def test_wilson_est_bilateral():
    from statistics import NormalDist

    resultat = _evaluer("intervalle_wilson", _wilson_avec(lambda c: NormalDist().inv_cdf(c)))
    assert resultat.statut == ECHEC and "UNILATÉRAL" in resultat.message


def test_wilson_utilise_le_niveau_de_confiance():
    resultat = _evaluer("intervalle_wilson", _wilson_avec(lambda c: 1.959963984540054))
    assert resultat.statut == ECHEC and "confiance est ignoré" in resultat.message


def _ab_avec(signe=1, commune=True, bilateral=True, valeur_absolue=True):
    import math
    from statistics import NormalDist

    def variante(partis_temoin, n_temoin, partis_offre, n_offre):
        p1, p2 = partis_temoin / n_temoin, partis_offre / n_offre
        p = (partis_temoin + partis_offre) / (n_temoin + n_offre)
        se = (math.sqrt(p * (1 - p) * (1 / n_temoin + 1 / n_offre)) if commune
              else math.sqrt(p1 * (1 - p1) / n_temoin + p2 * (1 - p2) / n_offre))
        ecart = signe * (p2 - p1)
        z = ecart / se
        queue = 1 - NormalDist().cdf(abs(z) if valeur_absolue else z)
        return {"ecart": ecart, "z": z, "p_value": (2 if bilateral else 1) * queue}

    return variante


def test_ab_l_ecart_est_offre_moins_temoin():
    resultat = _evaluer("comparer_ab", _ab_avec(signe=-1))
    assert resultat.statut == ECHEC and "offre − taux témoin" in resultat.message


def test_ab_utilise_la_variance_commune():
    resultat = _evaluer("comparer_ab", _ab_avec(commune=False))
    assert resultat.statut == ECHEC and "variance commune" in resultat.message


def test_ab_est_bilateral():
    """La leçon du défi 28 : en unilatéral, p = 0,028 ferait croire à une preuve qui n'en est pas une."""
    resultat = _evaluer("comparer_ab", _ab_avec(bilateral=False))
    assert resultat.statut == ECHEC and "UNILATÉRALE" in resultat.message


def test_ab_prend_la_valeur_absolue_de_z():
    resultat = _evaluer("comparer_ab", _ab_avec(valeur_absolue=False))
    assert resultat.statut == ECHEC and "|z|" in resultat.message


def test_ab_sans_aucun_depart_ne_plante_pas():
    resultat = _evaluer("comparer_ab", _ab_avec())  # pas de garde-fou quand l'erreur standard est nulle
    assert resultat.statut == ECHEC and "division par zéro" in resultat.message


def _survie_avec(**variante):
    from defis.checks_serie_05 import _km

    return lambda df: _km(df["tenure_months"], df["churn"], **variante)


def test_survie_les_censures_restent_a_risque():
    """La leçon du défi 29 : oublier les clients encore présents fait croire que tout le monde part."""
    resultat = _evaluer("courbe_survie", _survie_avec(sans_censures=True))
    assert resultat.statut == ECHEC and "censuré" in resultat.message


def test_survie_un_censure_n_est_pas_un_depart():
    resultat = _evaluer("courbe_survie", _survie_avec(tous_partis=True))
    assert resultat.statut == ECHEC and "n'est PAS un départ" in resultat.message


def test_survie_la_formule_naive_est_refusee():
    resultat = _evaluer("courbe_survie", _survie_avec(naif=True))
    assert resultat.statut == ECHEC and "ENCORE à risque" in resultat.message


def test_survie_est_un_produit_cumule():
    from defis.corrige_serie_05 import courbe_survie

    def mois_par_mois(df):
        table = courbe_survie(df)
        return table.assign(survie=1 - table["departs"] / table["a_risque"])

    resultat = _evaluer("courbe_survie", mois_par_mois)
    assert resultat.statut == ECHEC and "PRODUIT" in resultat.message


def test_survie_une_ecriture_vectorisee_est_acceptee():
    def avec_pandas(df):
        partis = df[df["churn"] == 1].groupby("tenure_months").size()
        a_risque = [int((df["tenure_months"] >= m).sum()) for m in partis.index]
        table = pd.DataFrame({"mois": partis.index, "a_risque": a_risque, "departs": partis.values})
        return table.assign(survie=(1 - table["departs"] / table["a_risque"]).cumprod())

    assert _evaluer("courbe_survie", avec_pandas).statut == REUSSI


# --------------------------------------------------------------------------- Série 00 : erreurs de débutant


def test_print_au_lieu_de_return_est_explique():
    def avec_print(partis, total):
        print(round(100 * partis / total, 1))

    resultat = _evaluer("taux_en_pourcentage", avec_print)
    assert resultat.statut == ECHEC and "print()" in resultat.message and "return" in resultat.message


def test_la_division_par_zero_donne_un_conseil():
    resultat = _evaluer("taux_en_pourcentage", lambda partis, total: round(100 * partis / total, 1))
    assert resultat.statut == ECHEC and "zéro" in resultat.message


def test_une_borne_mal_placee_est_signalee():
    def inferieur_ou_egal(mois):
        if mois < 0:
            raise ValueError("négatif")
        return "nouveau" if mois <= 6 else ("installé" if mois < 24 else "fidèle")

    resultat = _evaluer("segment_client", inferieur_ou_egal)
    assert resultat.statut == ECHEC and "bornes" in resultat.message


def test_un_accent_oublie_est_signale():
    def sans_accent(mois):
        if mois < 0:
            raise ValueError("négatif")
        return "nouveau" if mois < 6 else ("installe" if mois < 24 else "fidèle")

    resultat = _evaluer("segment_client", sans_accent)
    assert resultat.statut == ECHEC and "EXACTEMENT" in resultat.message


def test_le_dictionnaire_recree_dans_la_boucle_est_detecte():
    def recree(contrats):
        for c in contrats:
            compte = {}
            compte[c] = compte.get(c, 0) + 1
        return compte if contrats else {}

    def toujours_un(contrats):
        return {c: 1 for c in contrats}

    assert _evaluer("compter_par_contrat", toujours_un).statut == ECHEC
    assert "AVANT le `for`" in _evaluer("compter_par_contrat", toujours_un).message
    assert _evaluer("compter_par_contrat", recree).statut == ECHEC


def test_la_keyerror_du_compteur_est_expliquee():
    def sans_get(contrats):
        compte = {}
        for c in contrats:
            compte[c] += 1
        return compte

    resultat = _evaluer("compter_par_contrat", sans_get)
    assert resultat.statut == ECHEC and ".get(" in resultat.message


def test_trier_en_place_la_liste_recue_est_refuse():
    def tri_en_place(clients, seuil=0.5):
        clients.sort(key=lambda c: c["proba"], reverse=True)
        return [c["id"] for c in clients if c["proba"] >= seuil]

    resultat = _evaluer("clients_a_risque", tri_en_place)
    assert resultat.statut == ECHEC and "sorted()" in resultat.message


def test_le_seuil_strict_est_signale_pour_les_clients_a_risque():
    def strict(clients, seuil=0.5):
        return [c["id"] for c in sorted(clients, key=lambda c: c["proba"], reverse=True) if c["proba"] > seuil]

    resultat = _evaluer("clients_a_risque", strict)
    assert resultat.statut == ECHEC and ">=" in resultat.message


def test_le_code_bogue_du_defi_34_echoue_bug_par_bug():
    """Le défi 34 part d'un code faux : chaque bug corrigé doit faire apparaître le suivant."""

    def trois_bugs(factures):
        total = 0
        for i in range(1, len(factures)):
            total = factures[i]
        return total / len(factures)

    def deux_bugs(factures):
        total = 0
        for i in range(len(factures)):
            total = factures[i]
        return total / len(factures)

    def un_bug(factures):
        return sum(factures) / len(factures)

    assert "range()" in _evaluer("facture_moyenne", trois_bugs).message
    assert "total +=" in _evaluer("facture_moyenne", deux_bugs).message
    assert "zéro" in _evaluer("facture_moyenne", un_bug).message


def test_and_au_lieu_de_et_commercial_est_explique():
    def avec_and(df, contrat, anciennete_max):
        return df[(df["contract"] == contrat) and (df["tenure_months"] <= anciennete_max)]

    resultat = _evaluer("filtrer_clients", avec_and)
    assert resultat.statut == ECHEC and "&" in resultat.message


def test_les_parentheses_oubliees_sont_expliquees():
    def sans_parentheses(df, contrat, anciennete_max):
        return df[df["contract"] == contrat & df["tenure_months"] <= anciennete_max]

    resultat = _evaluer("filtrer_clients", sans_parentheses)
    assert resultat.statut == ECHEC and "parenthèses" in resultat.message


def test_groupby_est_interdit_au_defi_36_mais_pas_dans_la_docstring():
    def avec_groupby(df, colonne):
        return df.groupby(colonne)["churn"].mean().to_dict()

    def boucle(df, colonne):
        """Comme df.groupby(colonne)["churn"].mean(), mais à la main."""
        return {m: float(df.loc[df[colonne] == m, "churn"].mean()) for m in df[colonne].unique()}

    resultat = _evaluer("taux_par_groupe_a_la_main", avec_groupby)
    assert resultat.statut == ECHEC and "SANS groupby" in resultat.message
    assert _evaluer("taux_par_groupe_a_la_main", boucle).statut == REUSSI


def test_une_valeur_manquante_n_est_pas_une_valeur_inconnue():
    """La leçon du défi 37 : `propres.isna()` seul confond « manquant » et « inconnu »."""
    from defis.corrige_serie_00 import SYNONYMES

    def confond(contrats):
        propres = contrats.str.strip().str.lower().map(SYNONYMES)
        if propres.isna().any():
            raise ValueError(f"Contrats inconnus : {list(contrats[propres.isna()])}")
        return propres

    resultat = _evaluer("nettoyer_contrats", confond)
    assert resultat.statut == ECHEC and "MANQUANTE" in resultat.message


def test_oublier_de_normaliser_le_texte_est_explique():
    from defis.corrige_serie_00 import SYNONYMES

    def sans_normaliser(contrats):
        propres = contrats.map(SYNONYMES)
        inconnues = contrats[propres.isna() & contrats.notna()]
        if len(inconnues):
            raise ValueError(f"Contrats inconnus : {sorted(inconnues.unique())}")
        return propres

    resultat = _evaluer("nettoyer_contrats", sans_normaliser)
    assert resultat.statut == ECHEC and ".str.strip()" in resultat.message


def test_une_valeur_inconnue_ne_disparait_pas_en_silence():
    from defis.corrige_serie_00 import SYNONYMES

    resultat = _evaluer("nettoyer_contrats", lambda contrats: contrats.str.strip().str.lower().map(SYNONYMES))
    assert resultat.statut == ECHEC and "trimestriel" in resultat.message


# --------------------------------------------------------------------------- Série 06 : développer comme un·e pro


def test_traceback_le_premier_appel_n_est_pas_celui_qui_plante():
    import defis.corrige_serie_06 as corrige

    def premier_appel(texte):
        res = corrige.analyser_traceback(texte)
        premier = next(ligne.strip() for ligne in texte.splitlines() if ligne.strip().startswith('File "'))
        res["fichier"] = premier[len('File "'):].split('"', 1)[0]
        return res

    resultat = _evaluer("analyser_traceback", premier_appel)
    assert resultat.statut == ECHEC and "most recent call last" in resultat.message


def test_traceback_un_message_avec_deux_points_reste_entier():
    import defis.corrige_serie_06 as corrige

    def coupe_partout(texte):
        res = corrige.analyser_traceback(texte)
        res["message"] = res["message"].split(":")[0].strip()
        return res

    resultat = _evaluer("analyser_traceback", coupe_partout)
    assert resultat.statut == ECHEC and "PREMIER" in resultat.message


def test_traceback_la_ligne_est_un_entier():
    import defis.corrige_serie_06 as corrige

    resultat = _evaluer("analyser_traceback", lambda t: {**corrige.analyser_traceback(t), "ligne": "97"})
    assert resultat.statut == ECHEC and "int(" in resultat.message


def test_traceback_une_erreur_sans_message_ne_fait_pas_planter_la_verification():
    import defis.corrige_serie_06 as corrige

    def deballe(texte):
        res = corrige.analyser_traceback(texte)
        derniere = [ligne for ligne in texte.splitlines() if ligne.strip()][-1]
        res["erreur"], res["message"] = derniere.split(": ", 1)
        return res

    resultat = _evaluer("analyser_traceback", deballe)
    assert resultat.statut == ECHEC and "AUCUN message" in resultat.message


def test_traceback_chainee_on_lit_la_derniere_erreur():
    import defis.corrige_serie_06 as corrige

    def premiere_erreur(texte):
        res = corrige.analyser_traceback(texte)
        if "During handling" in texte:
            res["erreur"] = "FileNotFoundError"
        return res

    resultat = _evaluer("analyser_traceback", premiere_erreur)
    assert resultat.statut == ECHEC and "TOUT EN BAS" in resultat.message


def test_un_test_avec_un_seul_exemple_ne_suffit_pas():
    def un_seul_exemple(taux_en_pourcentage):
        assert taux_en_pourcentage(35, 100) == 35.0

    resultat = _evaluer("tester_taux_en_pourcentage", un_seul_exemple)
    assert resultat.statut == ECHEC and "aucun client" in resultat.message


def test_un_test_sans_assert_ne_verifie_rien():
    resultat = _evaluer("tester_taux_en_pourcentage", lambda taux_en_pourcentage: taux_en_pourcentage(35, 100))
    assert resultat.statut == ECHEC and "ne vérifie rien" in resultat.message


def test_un_test_qui_echoue_sur_le_bon_code_est_refuse():
    def valeur_fausse(taux_en_pourcentage):
        assert taux_en_pourcentage(1, 3) == 33.33

    resultat = _evaluer("tester_taux_en_pourcentage", valeur_fausse)
    assert resultat.statut == ECHEC and "CORRECTE" in resultat.message


def test_un_test_approximatif_laisse_passer_l_arrondi_oublie():
    import math

    def approximatif(f):
        assert f(35, 100) == 35.0 and f(1, 4) == 25.0 and f(0, 0) == 0.0
        assert math.isclose(f(1, 3), 33.3, abs_tol=0.05)

    resultat = _evaluer("tester_taux_en_pourcentage", approximatif)
    assert resultat.statut == ECHEC and "4/5" in resultat.message and "arrondi" in resultat.message


def _seuil_presque(modification):
    """Le corrigé de lire_seuil, avec une erreur de débutant·e injectée."""
    import defis.corrige_serie_06 as corrige

    def lire_seuil(texte, defaut=0.5):
        return modification(texte, defaut, corrige.lire_seuil)

    return lire_seuil


def test_lire_seuil_nan_passe_entre_les_bornes():
    def sans_nan(texte, defaut, juste):
        return float("nan") if texte == "nan" else juste(texte, defaut)

    resultat = _evaluer("lire_seuil", _seuil_presque(sans_nan))
    assert resultat.statut == ECHEC and "nan < 0" in resultat.message


def test_lire_seuil_une_saisie_invalide_ne_devient_pas_le_defaut():
    def silencieux(texte, defaut, juste):
        try:
            return juste(texte, defaut)
        except ValueError:
            if texte == "abc":
                return defaut
            raise

    resultat = _evaluer("lire_seuil", _seuil_presque(silencieux))
    assert resultat.statut == ECHEC and "en silence" in resultat.message


def test_lire_seuil_le_message_cite_la_saisie():
    def message_vague(texte, defaut, juste):
        try:
            return juste(texte, defaut)
        except ValueError:
            raise ValueError("seuil invalide") from None

    resultat = _evaluer("lire_seuil", _seuil_presque(message_vague))
    assert resultat.statut == ECHEC and "ne cite pas la saisie" in resultat.message


def test_lire_seuil_virgule_francaise():
    def sans_virgule(texte, defaut, juste):
        if texte and "," in texte:
            return float(texte)
        return juste(texte, defaut)

    resultat = _evaluer("lire_seuil", _seuil_presque(sans_virgule))
    assert resultat.statut == ECHEC and "virgule" in resultat.message


def test_lire_seuil_refuse_except_exception():
    def lire_seuil(texte, defaut=0.5):
        if texte is None or not texte.strip():
            return defaut
        nettoye = texte.strip().replace(",", ".")
        pourcentage = nettoye.endswith("%")
        try:
            valeur = float(nettoye.rstrip("%"))
        except Exception:
            raise ValueError(f"Seuil illisible : {texte!r}") from None
        valeur = valeur / 100 if pourcentage else valeur
        if not 0 <= valeur <= 1:
            raise ValueError(f"Hors bornes : {texte!r}")
        return valeur

    resultat = _evaluer("lire_seuil", lire_seuil)
    assert resultat.statut == ECHEC and "except ValueError" in resultat.message


_KPIS = ("monthly_charges", "tenure_months", "support_calls")


def test_refactoring_l_ecart_se_calcule_avant_l_arrondi():
    def arrondi_trop_tot(df, colonnes=_KPIS):
        partis, restes = df[df["churn"] == 1], df[df["churn"] == 0]
        lignes = []
        for col in colonnes:
            m1, m0 = round(partis[col].mean(), 2), round(restes[col].mean(), 2)
            lignes.append({"variable": col, "moyenne_partis": m1, "moyenne_restes": m0, "ecart": round(m1 - m0, 2)})
        return pd.DataFrame(lignes)

    resultat = _evaluer("resume_kpis", arrondi_trop_tot)
    assert resultat.statut == ECHEC and "ARRONDIES" in resultat.message


def test_refactoring_groupby_sans_client_parti():
    def avec_groupby(df, colonnes=_KPIS):
        moyennes = df.groupby("churn")[list(colonnes)].mean()
        lignes = []
        for col in colonnes:
            m1, m0 = moyennes.loc[1, col], moyennes.loc[0, col]
            lignes.append({"variable": col, "moyenne_partis": round(m1, 2), "moyenne_restes": round(m0, 2),
                           "ecart": round(m1 - m0, 2)})
        return pd.DataFrame(lignes)

    resultat = _evaluer("resume_kpis", avec_groupby)
    assert resultat.statut == ECHEC and "AUCUN client" in resultat.message


def test_refactoring_sans_colonne_en_dur():
    from defis.corrige_serie_06 import resume_kpis

    def en_dur(df, colonnes=_KPIS):
        if "monthly_charges" not in df:
            raise KeyError("monthly_charges")
        return resume_kpis(df, colonnes)

    resultat = _evaluer("resume_kpis", en_dur)
    assert resultat.statut == ECHEC and "en dur" in resultat.message


def test_refactoring_qui_appelle_l_ancien_code_est_refuse():
    import defis.corrige_serie_06 as corrige
    from defis.corrige_serie_06 import resume_kpis_ancien

    def triche(df, colonnes=_KPIS):
        if tuple(colonnes) == _KPIS:
            return resume_kpis_ancien(df)
        return corrige.resume_kpis(df, colonnes)

    resultat = _evaluer("resume_kpis", triche)
    assert resultat.statut == ECHEC and "ancien" in resultat.message


def _parser(seuil_type=None, aide="aide"):
    import argparse

    from defis.corrige_serie_06 import lire_seuil

    parser = argparse.ArgumentParser()
    parser.add_argument("fichier", help=aide)
    parser.add_argument("--seuil", type=seuil_type or lire_seuil, default=0.5, help=aide)
    parser.add_argument("--modele", choices=["logistic", "gboost"], default="logistic", help=aide)
    parser.add_argument("--top", type=int, default=10, help=aide)
    parser.add_argument("-v", "--verbose", action="store_true", help=aide)
    return parser


def test_parser_type_float_ne_verifie_pas_les_bornes():
    resultat = _evaluer("creer_parser", lambda: _parser(seuil_type=float))
    assert resultat.statut == ECHEC and "1.5" in resultat.message and "lire_seuil" in resultat.message


def test_parser_sans_type_donne_du_texte():
    resultat = _evaluer("creer_parser", lambda: _parser(seuil_type=str))
    assert resultat.statut == ECHEC and "TEXTE" in resultat.message


def test_parser_chaque_option_a_une_aide():
    resultat = _evaluer("creer_parser", lambda: _parser(aide=None))
    assert resultat.statut == ECHEC and "help=" in resultat.message


def test_parser_une_ligne_invalide_ne_ferme_pas_le_correcteur():
    """argparse appelle sys.exit() sur une erreur : la vérification doit l'intercepter (sinon tout s'arrête)."""
    import argparse

    def fichier_en_option():
        parser = argparse.ArgumentParser()
        parser.add_argument("--fichier", required=True)
        return parser

    resultat = _evaluer("creer_parser", fichier_en_option)
    assert resultat.statut == ECHEC and "positionnel" in resultat.message


def test_parser_avec_lire_seuil_non_codee_renvoie_au_defi_40():
    def pas_encore(texte):
        raise NotImplementedError

    resultat = _evaluer("creer_parser", lambda: _parser(seuil_type=pas_encore))
    assert resultat.statut == ECHEC and "défi 40" in resultat.message


def _bissection(milieu, reponse="mauvais", debut=0):
    def premier_commit_fautif(commits, est_bon):
        if len(commits) < 2:
            raise ValueError("au moins deux commits")
        bon, mauvais = debut, len(commits) - 1
        while mauvais - bon > 1:
            m = milieu(bon, mauvais)
            if est_bon(commits[m]):
                bon = m
            else:
                mauvais = m
        return commits[mauvais if reponse == "mauvais" else bon]

    return premier_commit_fautif


def test_bissection_a_un_pres():
    resultat = _evaluer("premier_commit_fautif", _bissection(lambda b, m: (b + m) // 2, reponse="bon"))
    assert resultat.statut == ECHEC and "à un près" in resultat.message


def test_bissection_lineaire_trop_d_appels():
    resultat = _evaluer("premier_commit_fautif", _bissection(lambda b, m: b + 1))
    assert resultat.statut == ECHEC and "un par un" in resultat.message


def test_bissection_boucle_infinie_detectee():
    """Une boucle qui ne rétrécit jamais l'intervalle doit donner un message, pas bloquer le correcteur."""

    def bloque(commits, est_bon):
        if len(commits) < 2:
            raise ValueError("au moins deux commits")
        bon, mauvais = 0, len(commits) - 1
        while mauvais - bon > 1:
            m = (bon + mauvais) // 2
            if not est_bon(commits[m]):
                mauvais = m
        return commits[mauvais]

    resultat = _evaluer("premier_commit_fautif", bloque)
    assert resultat.statut == ECHEC and "boucle infinie" in resultat.message


def test_bissection_ne_reteste_pas_les_extremites():
    def lower_bound(commits, est_bon):
        if len(commits) < 2:
            raise ValueError("au moins deux commits")
        lo, hi = 0, len(commits) - 1
        while lo < hi:
            m = (lo + hi) // 2
            if est_bon(commits[m]):
                lo = m + 1
            else:
                hi = m
        return commits[lo]

    resultat = _evaluer("premier_commit_fautif", lower_bound)
    assert resultat.statut == ECHEC and "premier ou le dernier" in resultat.message


def test_bissection_trop_peu_de_commits():
    resultat = _evaluer("premier_commit_fautif", lambda commits, est_bon: commits[-1])
    assert resultat.statut == ECHEC and "ValueError" in resultat.message


# --------------------------------------------------------------------------- Série 07 : code réutilisable


def _mois_sans_fin_de_mois(debut, fin):
    from defis.corrige_serie_07 import _en_date

    debut, fin = _en_date(debut), _en_date(fin)
    if fin < debut:
        raise ValueError("fin avant début")
    return (fin.year - debut.year) * 12 + (fin.month - debut.month) - (fin.day < debut.day)


def test_anciennete_mois_du_calendrier_au_lieu_de_mois_complets():
    from defis.corrige_serie_07 import _en_date

    def calendaires(debut, fin):
        debut, fin = _en_date(debut), _en_date(fin)
        if fin < debut:
            raise ValueError("fin avant début")
        return (fin.year - debut.year) * 12 + (fin.month - debut.month)

    resultat = _evaluer("anciennete_en_mois", calendaires)
    assert resultat.statut == ECHEC and "COMPLETS" in resultat.message


def test_anciennete_un_mois_n_a_pas_30_jours():
    from defis.corrige_serie_07 import _en_date

    resultat = _evaluer("anciennete_en_mois", lambda d, f: (_en_date(f) - _en_date(d)).days // 30)
    assert resultat.statut == ECHEC and "30" in resultat.message


def test_anciennete_fin_de_mois():
    resultat = _evaluer("anciennete_en_mois", _mois_sans_fin_de_mois)
    assert resultat.statut == ECHEC and "FIN DE MOIS" in resultat.message


def test_anciennete_espaces_autour_de_la_date():
    from datetime import date

    from defis.corrige_serie_07 import anciennete_en_mois

    def sans_strip(debut, fin):
        debut = debut if isinstance(debut, date) else date.fromisoformat(debut)
        fin = fin if isinstance(fin, date) else date.fromisoformat(fin)
        return anciennete_en_mois(debut, fin)

    resultat = _evaluer("anciennete_en_mois", sans_strip)
    assert resultat.statut == ECHEC and ".strip()" in resultat.message


def test_anciennete_negative_refusee():
    from defis.corrige_serie_07 import _en_date, anciennete_en_mois

    def negative(debut, fin):
        if _en_date(fin) < _en_date(debut):
            return -1
        return anciennete_en_mois(debut, fin)

    resultat = _evaluer("anciennete_en_mois", negative)
    assert resultat.statut == ECHEC and "incohérente" in resultat.message


def test_anciennete_ne_devine_pas_le_format():
    from datetime import datetime

    from defis.corrige_serie_07 import anciennete_en_mois

    def devine(debut, fin):
        if isinstance(debut, str) and "/" in debut:
            debut = datetime.strptime(debut, "%d/%m/%Y").date()
        return anciennete_en_mois(debut, fin)

    resultat = _evaluer("anciennete_en_mois", devine)
    assert resultat.statut == ECHEC and "devinez" in resultat.message


def test_toujours_a_risque_recherche_dans_une_liste():
    def dans_la_liste(mois_dernier, ce_mois):
        return list(dict.fromkeys(c for c in ce_mois if c in mois_dernier))

    resultat = _evaluer("clients_toujours_a_risque", dans_la_liste)
    assert resultat.statut == ECHEC and "set(" in resultat.message


def test_toujours_a_risque_set_recree_a_chaque_tour():
    def set_dans_la_boucle(mois_dernier, ce_mois):
        return list(dict.fromkeys(c for c in ce_mois if c in set(mois_dernier)))

    resultat = _evaluer("clients_toujours_a_risque", set_dans_la_boucle)
    assert resultat.statut == ECHEC and "AVANT la boucle" in resultat.message


def test_toujours_a_risque_doublons_cherches_dans_le_resultat():
    def doublons_lents(mois_dernier, ce_mois):
        avant, resultat = set(mois_dernier), []
        for client in ce_mois:
            if client in avant and client not in resultat:
                resultat.append(client)
        return resultat

    resultat = _evaluer("clients_toujours_a_risque", doublons_lents)
    assert resultat.statut == ECHEC and "deja_vus" in resultat.message


def test_toujours_a_risque_ordre_et_doublons():
    def ordre_du_mois_dernier(mois_dernier, ce_mois):
        maintenant = set(ce_mois)
        return list(dict.fromkeys(c for c in mois_dernier if c in maintenant))

    def avec_doublons(mois_dernier, ce_mois):
        avant = set(mois_dernier)
        return [c for c in ce_mois if c in avant]

    assert "ORDRE" in _evaluer("clients_toujours_a_risque", ordre_du_mois_dernier).message
    assert "UNE fois" in _evaluer("clients_toujours_a_risque", avec_doublons).message
    assert "set" in _evaluer("clients_toujours_a_risque", lambda a, b: set(a) & set(b)).message


def _export(**options):
    """Une version de lire_export_excel avec des options de read_csv choisies (et rien d'autre)."""

    def lire(chemin):
        df = pd.read_csv(chemin, **options)
        df.columns = df.columns.str.strip()
        return df

    return lire


@pytest.mark.parametrize("options,attendu", [
    ({}, "POINTS-VIRGULES"),
    ({"sep": ";", "decimal": ",", "encoding": "cp1252", "dtype": {"id": str}}, "ChÃ¨que"),
    ({"sep": ";", "decimal": ",", "encoding": "utf-8-sig", "dtype": {"id": str}}, "cp1252"),
], ids=["sans_options", "toujours_cp1252", "sans_repli_cp1252"])
def test_export_excel_options_manquantes(options, attendu):
    resultat = _evaluer("lire_export_excel", _export(**options))
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def _export_presque(**retirer):
    """Le corrigé de lire_export_excel, privé d'une étape."""

    def lire(chemin):
        options = {"sep": ";", "decimal": ",", "dtype": {"id": str}}
        for cle in retirer:
            options.pop(cle, None)
        for encodage in (retirer.get("encodage") or "utf-8-sig", "cp1252"):
            try:
                df = pd.read_csv(chemin, encoding=encodage, **options)
                break
            except UnicodeDecodeError:
                continue
        if "strip" not in retirer:
            df.columns = df.columns.str.strip()
        return df

    return lire


@pytest.mark.parametrize("retirer,attendu", [
    ({"strip": True}, ".str.strip()"),
    ({"decimal": True}, "decimal"),
    ({"dtype": True}, "zéros"),
], ids=["noms_non_nettoyes", "virgule_decimale", "identifiants_numeriques"])
def test_export_excel_etape_oubliee(retirer, attendu):
    resultat = _evaluer("lire_export_excel", _export_presque(**retirer))
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_par_lots_qui_charge_tout_en_memoire():
    from defis.corrige_serie_07 import par_lots

    resultat = _evaluer("par_lots", lambda elements, taille: list(par_lots(elements, taille)))
    assert resultat.statut == ECHEC and "yield" in resultat.message


def test_par_lots_avec_des_indices():
    def tranches(elements, taille):
        if taille < 1:
            raise ValueError(taille)
        for i in range(0, len(elements), taille):
            yield elements[i:i + taille]

    resultat = _evaluer("par_lots", tranches)
    assert resultat.statut == ECHEC and "GÉNÉRATEUR" in resultat.message


def test_par_lots_reutilise_la_meme_liste():
    def meme_liste(elements, taille):
        lot = []
        for element in elements:
            lot.append(element)
            if len(lot) == taille:
                yield lot
                lot.clear()
        if lot:
            yield lot

    resultat = _evaluer("par_lots", meme_liste)
    assert resultat.statut == ECHEC and "MÊME liste" in resultat.message


def test_par_lots_dernier_lot_oublie_ou_vide():
    def sans_reste(elements, taille):
        lot = []
        for element in elements:
            lot.append(element)
            if len(lot) == taille:
                yield lot
                lot = []

    def reste_vide(elements, taille):
        lot = []
        for element in elements:
            lot.append(element)
            if len(lot) == taille:
                yield lot
                lot = []
        yield lot

    assert "dernier lot" in _evaluer("par_lots", sans_reste).message
    assert "VIDE" in _evaluer("par_lots", reste_vide).message


def test_par_lots_taille_nulle_sans_boucle_infinie():
    import itertools

    def sans_garde(elements, taille):
        it = iter(elements)
        while True:
            lot = list(itertools.islice(it, max(taille, 0)))
            if not lot and taille > 0:
                return
            yield lot

    resultat = _evaluer("par_lots", sans_garde)
    assert resultat.statut == ECHEC and "ValueError" in resultat.message


def _journal(**changements):
    """Une sous-classe du corrigé de JournalPredictions, avec un défaut injecté."""
    from defis.corrige_serie_07 import JournalPredictions

    return type("JournalPredictions", (JournalPredictions,), changements)


def test_journal_attribut_de_classe_partage():
    from defis.corrige_serie_07 import JournalPredictions

    partage = {}

    def init(self, seuil=0.5):
        JournalPredictions.__init__(self, seuil)
        self.scores = partage

    resultat = _evaluer("JournalPredictions", _journal(__init__=init))
    assert resultat.statut == ECHEC and "CLASSE" in resultat.message


def test_journal_seuil_inclus_et_egalites():
    def strict(self):
        return [c for c, p in sorted(self.scores.items(), key=lambda cp: (-cp[1], cp[0])) if p > self.seuil]

    def sans_departage(self):
        return [c for c, p in sorted(self.scores.items(), key=lambda cp: -cp[1]) if p >= self.seuil]

    assert "EXACTEMENT" in _evaluer("JournalPredictions", _journal(a_risque=strict)).message
    assert "égalité" in _evaluer("JournalPredictions", _journal(a_risque=sans_departage)).message


def test_journal_un_client_mis_a_jour_ne_compte_qu_une_fois():
    from defis.corrige_serie_07 import JournalPredictions

    def init(self, seuil=0.5):
        JournalPredictions.__init__(self, seuil)
        self.historique = []

    def ajouter(self, client_id, proba):
        JournalPredictions.ajouter(self, client_id, proba)
        self.historique.append(client_id)

    journal = _journal(__init__=init, ajouter=ajouter, __len__=lambda self: len(self.historique))
    resultat = _evaluer("JournalPredictions", journal)
    assert resultat.statut == ECHEC and "dictionnaire" in resultat.message


def test_journal_nan_et_seuil_fige():
    from defis.corrige_serie_07 import JournalPredictions

    def accepte_nan(self, client_id, proba):
        if proba < 0 or proba > 1:
            raise ValueError(proba)
        self.scores[client_id] = proba

    def init(self, seuil=0.5):
        JournalPredictions.__init__(self, seuil)
        self._seuil_initial = seuil

    def seuil_fige(self):
        return [c for c, p in sorted(self.scores.items(), key=lambda cp: (-cp[1], cp[0])) if p >= self._seuil_initial]

    assert "nan" in _evaluer("JournalPredictions", _journal(ajouter=accepte_nan)).message
    assert "À LA DEMANDE" in _evaluer("JournalPredictions", _journal(__init__=init, a_risque=seuil_fige)).message


def test_journal_vide_et_len():
    def division(self):
        return len(self.a_risque()) / len(self.scores)

    from defis.corrige_serie_07 import JournalPredictions

    assert "0.0" in _evaluer("JournalPredictions", _journal(taux_alerte=division)).message
    methodes = {nom: v for nom, v in vars(JournalPredictions).items() if nom != "__len__" and not nom.startswith("__dict")}
    sans_len = type("JournalPredictions", (), methodes)
    assert "__len__" in _evaluer("JournalPredictions", sans_len).message


def _cache_presque(cle=None, present=None, cache_global=None, wraps=True, avale=False):
    """Une version d'avec_cache avec une erreur de débutant·e injectée."""
    import functools

    def avec_cache(fonction):
        cache = {} if cache_global is None else cache_global
        stats = {"calculs": 0, "reutilisations": 0}

        def enveloppe(*args, **kwargs):
            k = cle(args, kwargs) if cle else (args, tuple(sorted(kwargs.items())))
            if (present(cache, k) if present else k in cache):
                stats["reutilisations"] += 1
                return cache[k]
            stats["calculs"] += 1
            try:
                cache[k] = fonction(*args, **kwargs)
            except Exception:
                if not avale:
                    raise
                cache[k] = None
            return cache[k]

        if wraps:
            enveloppe = functools.wraps(fonction)(enveloppe)
        enveloppe.statistiques = lambda: dict(stats)
        return enveloppe

    return avec_cache


@pytest.mark.parametrize("erreur,attendu", [
    ({"cle": lambda args, kwargs: args}, "NOMMÉS"),
    ({"cle": lambda args, kwargs: "-".join(map(str, args)) + str(sorted(kwargs.items()))}, "confond"),
    ({"present": lambda cache, k: cache.get(k)}, "cle in cache"),
    ({"cache_global": {}}, "PARTAGÉ"),
    ({"wraps": False}, "functools.wraps"),
    ({"avale": True}, "laisser passer"),
], ids=["kwargs_oublies", "cle_texte", "resultat_faux", "cache_partage", "sans_wraps", "erreur_avalee"])
def test_avec_cache_erreurs_classiques(erreur, attendu):
    resultat = _evaluer("avec_cache", _cache_presque(**erreur))
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_avec_cache_sans_cache_ou_sans_statistiques():
    import functools

    assert "telle quelle" in _evaluer("avec_cache", lambda fonction: fonction).message
    assert "statistiques()" in _evaluer("avec_cache", functools.cache).message


# --------------------------------------------------------------------------- Série 08 : assembler les données


def _regex(motif, flags=0, unique=True):
    """Une version de extraire_identifiants construite à partir d'une regex (pour tester les pièges)."""
    import re

    def extraire(texte):
        if not isinstance(texte, str):
            return []
        trouves = [m.upper() for m in re.findall(motif, texte, flags)]
        return list(dict.fromkeys(trouves)) if unique else trouves

    return extraire


@pytest.mark.parametrize("fonction,attendu", [
    (_regex(r"\bC\d{5}\b"), "minuscule"),
    (_regex(r"C\d{5}", __import__("re").I), "frontières de mot"),
    (_regex(r"\bC\d{5}", __import__("re").I), "frontières de mot"),
    (_regex(r"\bC\d{5}\b", __import__("re").I, unique=False), "une seule fois"),
    (lambda t: sorted({m.upper() for m in __import__("re").findall(r"\bC\d{5}\b", t, 2)}) if isinstance(t, str) else [],
     "ordre"),
    (lambda t: [m.upper() for m in t.split() if __import__("re").fullmatch(r"[Cc]\d{5}", m)] if isinstance(t, str) else [],
     "ponctuation"),
    (lambda t: __import__("re").findall(r"\b(C)(\d{5})\b", t), "tuples"),
    (lambda t: __import__("re").findall(r"\bC\d{5}\b", t, 2)[0], "LISTE"),
    (lambda t: list(dict.fromkeys(m.upper() for m in __import__("re").findall(r"\bC\d{5}\b", t, 2))), "isinstance"),
], ids=["sans_ignorecase", "sans_frontiere", "sans_frontiere_finale", "doublons", "trie", "split", "groupes",
        "premier_seulement", "note_vide"])
def test_extraire_identifiants_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("extraire_identifiants", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def _totaux(f, how="left", rang="apres", nan=0.0, index=False):
    def ajouter(clients, factures):
        if rang == "avant":
            resultat = clients.merge(factures, on="customer_id", how=how).rename(columns={"montant": "total_factures"})
        else:
            totaux = getattr(factures.groupby("customer_id")["montant"], f)().rename("total_factures").reset_index()
            resultat = clients.merge(totaux, on="customer_id", how=how)
        if nan is not None:
            resultat["total_factures"] = resultat["total_factures"].fillna(nan)
        if index:
            resultat.index = clients.index
        return resultat

    return ajouter


@pytest.mark.parametrize("fonction,attendu", [
    (_totaux("sum", rang="avant"), "joint avant d'agréger"),
    (_totaux("sum", how="inner"), "INTERNE"),
    (_totaux("sum", how="outer"), "externe"),
    (_totaux("sum", nan=None, index=True), "0.0"),
    (_totaux("sum"), "RENUMÉROTE"),
    (_totaux("count", index=True), ".sum()"),
], ids=["jointure_avant_agregation", "jointure_interne", "jointure_externe", "nan_non_remplaces", "index_perdu",
        "count_au_lieu_de_sum"])
def test_ajouter_total_factures_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("ajouter_total_factures", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_ajouter_total_factures_sur_place():
    from defis.corrige_serie_08 import ajouter_total_factures

    def sur_place(clients, factures):
        clients["total_factures"] = ajouter_total_factures(clients, factures)["total_factures"]
        return clients

    resultat = _evaluer("ajouter_total_factures", sur_place)
    assert resultat.statut == ECHEC and "NOUVEAU DataFrame" in resultat.message


def _api(texte=True, ou_vide=True, moyenne_vide="nan", colonnes=True, get_clients=True):
    """Une version de aplatir_reponse_api avec une seule erreur à la fois."""
    import json
    import math

    import defis.corrige_serie_08 as corrige

    def aplatir(reponse):
        if texte and isinstance(reponse, str):
            reponse = json.loads(reponse)
        lignes = []
        for client in (reponse.get("clients") or []) if get_clients else reponse["clients"]:
            contrat = (client.get("contrat") or {}) if ou_vide else client.get("contrat", {})
            factures = (client.get("factures") or []) if ou_vide else client.get("factures", [])
            montants = [f["montant"] for f in factures]
            if montants or moyenne_vide == "division":
                moyenne = sum(montants) / len(montants)
            else:
                moyenne = 0.0 if moyenne_vide == "zero" else math.nan
            lignes.append({"customer_id": client["id"], "contrat": contrat.get("type"), "nb_factures": len(factures),
                           "facture_moyenne": moyenne, "nb_options": len(client.get("options") or [])})
        return pd.DataFrame(lignes, columns=corrige.COLONNES_API if colonnes else None)

    return aplatir


@pytest.mark.parametrize("fonction,attendu", [
    (_api(ou_vide=False), "or {}"),
    (_api(moyenne_vide="division"), "liste VIDE"),
    (_api(moyenne_vide="zero"), "NaN"),
    (_api(texte=False), "json.loads"),
    (_api(colonnes=False), "columns=[...]"),
    (_api(get_clients=False), "reponse.get(\"clients\")"),
], ids=["null_non_gere", "moyenne_liste_vide", "moyenne_zero", "texte_json", "page_vide_sans_colonnes",
        "sans_cle_clients"])
def test_aplatir_reponse_api_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("aplatir_reponse_api", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_aplatir_reponse_api_cle_absente():
    def crochets(reponse):
        lignes = []
        for c in reponse["clients"]:
            contrat, factures = c["contrat"] or {}, c["factures"] or []  # C00003 n'a que son id : KeyError
            montants = [f["montant"] for f in factures]
            lignes.append({"customer_id": c["id"], "contrat": contrat.get("type"), "nb_factures": len(factures),
                           "facture_moyenne": sum(montants) / len(montants) if montants else float("nan"),
                           "nb_options": len(c["options"] or [])})
        return pd.DataFrame(lignes)

    resultat = _evaluer("aplatir_reponse_api", crochets)
    assert resultat.statut == ECHEC and "dict.get" in resultat.message, resultat.message


def _bareme(appels=">=", anciennete="<", support=-2, elevé=">=", ordre="strict_d_abord"):
    import numpy as np

    ops = {">=": lambda a, b: a >= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}

    def niveau(df):
        points = (3 * (df["contract"] == "Mensuel") + 2 * ops[appels](df["support_calls"], 3)
                  + 1 * ops[anciennete](df["tenure_months"], 6) + support * (df["tech_support"] == "Oui"))
        conditions = [ops[elevé](points, 4), points >= 2]
        choix = ["élevé", "moyen"]
        if ordre != "strict_d_abord":
            conditions, choix = conditions[::-1], choix[::-1]
        return pd.Series(np.select(conditions, choix, default="faible"), index=df.index)

    return niveau


@pytest.mark.parametrize("fonction,attendu", [
    (_bareme(appels=">"), ">= 3"),
    (_bareme(anciennete="<="), "< 6"),
    (_bareme(support=2), "RETIRE"),
    (_bareme(elevé=">"), ">= 4"),
    (_bareme(ordre="inverse"), "PREMIÈRE condition"),
], ids=["seuil_appels", "seuil_anciennete", "signe_support", "seuil_niveau", "select_inverse"])
def test_niveau_risque_erreurs_de_bareme(fonction, attendu):
    resultat = _evaluer("niveau_risque", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_niveau_risque_ligne_a_ligne_est_refuse():
    from defis.corrige_serie_08 import niveau_risque_lent

    for lent in (niveau_risque_lent,
                 lambda df: df.apply(lambda c: niveau_risque_lent(c.to_frame().T).iloc[0], axis=1)):
        resultat = _evaluer("niveau_risque", lent)
        assert resultat.statut == ECHEC and "un par un" in resultat.message, resultat.message


def test_niveau_risque_sans_index_ou_en_tableau():
    from defis.corrige_serie_08 import niveau_risque

    assert "index=df.index" in _evaluer("niveau_risque", lambda df: pd.Series(niveau_risque(df).to_numpy())).message
    assert "pd.Series" in _evaluer("niveau_risque", lambda df: niveau_risque(df).to_numpy()).message


def _upsert(dedoublonner=True, subset=True, keep="last", ignore_index=True, garde_fou=True):
    def mettre_a_jour(table, nouveautes, cle="customer_id"):
        if garde_fou and table[cle].duplicated().any():
            raise ValueError("doublons")
        tout = pd.concat([table, nouveautes[table.columns]], ignore_index=ignore_index)
        if dedoublonner:
            tout = tout.drop_duplicates(subset=cle if subset else None, keep=keep)
        return tout.reset_index(drop=True) if ignore_index else tout

    return mettre_a_jour


def _avec_update(table, nouveautes, cle="customer_id"):
    ancienne = table.set_index(cle)
    nouvelle = nouveautes.drop_duplicates(cle, keep="last").set_index(cle)[ancienne.columns]
    ancienne.update(nouvelle)
    return pd.concat([ancienne, nouvelle[~nouvelle.index.isin(ancienne.index)]]).reset_index()


@pytest.mark.parametrize("fonction,attendu", [
    (_upsert(dedoublonner=False), "EMPILE"),
    (_upsert(subset=False), "sans `subset`"),
    (_upsert(keep="first"), "keep=\"last\""),
    (_upsert(ignore_index=False), "ignore_index=True"),
    (_upsert(garde_fou=False), "ValueError"),
    (_avec_update, "IGNORENT les valeurs vides"),
], ids=["concat_seul", "sans_subset", "garde_l_ancienne", "index_duplique", "sans_garde_fou", "update_ignore_none"])
def test_mettre_a_jour_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("mettre_a_jour", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_mettre_a_jour_avec_la_cle_en_dur():
    from defis.corrige_serie_08 import mettre_a_jour

    resultat = _evaluer("mettre_a_jour", lambda table, nouveautes, cle="customer_id": mettre_a_jour(table, nouveautes))
    assert resultat.statut == ECHEC and "paramètre `cle`" in resultat.message, resultat.message


def _evenements(filtre="<", debut=">=", fenetre_fixe=None, fenetre_reclamations=False, reindex=True, jours_vides=None):
    """Une version de variables_evenements avec une seule erreur à la fois."""

    def variables(evenements, clients, date_reference, fenetre_jours=30):
        reference = pd.Timestamp(date_reference)
        ev = evenements.assign(date=pd.to_datetime(evenements["date"]))
        if filtre == "<":
            ev = ev[ev["date"] < reference]
        elif filtre == "<=":
            ev = ev[ev["date"] <= reference]
        borne = reference - pd.Timedelta(days=fenetre_fixe or fenetre_jours)
        recent = ev["date"] >= borne if debut == ">=" else ev["date"] > borne
        appels = ev[(ev["type"] == "appel_support") & recent].groupby("customer_id").size()
        masque = ev["type"] == "reclamation"
        reclamations = ev[masque & recent if fenetre_reclamations else masque].groupby("customer_id").size()
        dernier = ev.groupby("customer_id")["date"].max()
        index = pd.Index(clients) if reindex else dernier.index
        resultat = pd.DataFrame({
            "appels_recents": appels.reindex(index, fill_value=0).astype(int),
            "reclamations": reclamations.reindex(index, fill_value=0).astype(int),
            "jours_depuis_dernier": (reference - dernier.reindex(index)).dt.days.astype(float),
        }, index=index)
        if jours_vides is not None:
            resultat["jours_depuis_dernier"] = resultat["jours_depuis_dernier"].fillna(jours_vides)
        return resultat

    return variables


@pytest.mark.parametrize("fonction,attendu", [
    (_evenements(filtre=None), "FUITE TEMPORELLE"),
    (_evenements(filtre="<="), "STRICTEMENT avant"),
    (_evenements(debut=">"), "l'INCLUT"),
    (_evenements(fenetre_fixe=30), "fenetre_jours"),
    (_evenements(fenetre_reclamations=True), "TOUT l'historique"),
    (_evenements(reindex=False), "reindex"),
    (_evenements(jours_vides=0.0), "laissez NaN"),
], ids=["fuite_futur", "jour_j_inclus", "borne_de_fenetre_exclue", "fenetre_en_dur", "reclamations_fenetrees",
        "sans_reindex", "jours_zero_au_lieu_de_nan"])
def test_variables_evenements_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("variables_evenements", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_variables_evenements_ne_modifie_pas_le_journal():
    from defis.corrige_serie_08 import variables_evenements

    def sur_place(evenements, clients, date_reference, fenetre_jours=30):
        evenements["date"] = pd.to_datetime(evenements["date"])
        return variables_evenements(evenements, clients, date_reference, fenetre_jours)

    resultat = _evaluer("variables_evenements", sur_place)
    assert resultat.statut == ECHEC and "copie" in resultat.message, resultat.message


def test_variables_evenements_dates_texte_comparees_a_une_date():
    from defis.corrige_serie_08 import variables_evenements

    def sans_conversion(evenements, clients, date_reference, fenetre_jours=30):
        connus = evenements[evenements["date"] < date_reference]  # marche par hasard sur des textes ISO
        return variables_evenements(connus, clients, date_reference, fenetre_jours)

    resultat = _evaluer("variables_evenements", sans_conversion)
    assert resultat.statut == ECHEC and "pd.Timestamp" in resultat.message, resultat.message


# --------------------------------------------------------------------------- Série 09 : algorithmes d'entretien


def _deux_offres(get=False, enregistrer_avant=False, sans_reponse=None, valeurs=False, inverse=False, tri=False):
    """Une version de deux_offres avec une seule erreur à la fois."""

    def deux_offres(couts, budget):
        if tri:
            couts.sort()
        vus = {}
        for j, cout in enumerate(couts):
            complement = budget - cout
            if enregistrer_avant:
                vus.setdefault(cout, j)
            if (vus.get(complement) if get else complement in vus):
                i = vus[complement]
                if valeurs:
                    return complement, cout
                return (j, i) if inverse else (i, j)
            vus.setdefault(cout, j)
        return sans_reponse

    return deux_offres


def _toutes_les_paires(couts, budget):
    for i in range(len(couts)):
        for j in range(i + 1, len(couts)):
            if couts[i] + couts[j] == budget:
                return i, j
    return None


@pytest.mark.parametrize("fonction,attendu", [
    (_deux_offres(get=True), "position 0"),
    (_deux_offres(enregistrer_avant=True), "DEUX FOIS"),
    (_deux_offres(sans_reponse=-1), "au lieu de None"),
    (_deux_offres(valeurs=True), "COÛTS"),
    (_deux_offres(inverse=True), "i < j"),
    (_deux_offres(tri=True), "modifié `couts`"),
    (_toutes_les_paires, "boucle dans une boucle"),
    (lambda c, b: next(((i, c.index(b - x, i + 1)) for i, x in enumerate(c) if b - x in c[i + 1:]), None),
     "boucle dans une boucle"),
    (lambda c, b: _toutes_les_paires(list(c), b), "8 000 gestes"),
], ids=["get_position_zero", "se_complete_lui_meme", "moins_un", "couts_au_lieu_des_positions", "ordre_inverse",
        "tri_sur_place", "force_brute", "tranches", "force_brute_sur_copie"])
def test_deux_offres_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("deux_offres", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


_FERMANTS = {")": "(", "]": "[", "}": "{"}


def _pile(vide=True, type_=True, fin=True, booleen=True):
    """Une version de parenthesage_valide avec une seule erreur à la fois."""

    def valide(formule):
        pile = []
        for c in formule:
            if c in "([{":
                pile.append(c)
            elif c in _FERMANTS:
                if vide and not pile:
                    return False
                sommet = pile.pop()
                if type_ and sommet != _FERMANTS[c]:
                    return False
        if not booleen:
            return pile
        return not pile if fin else True

    return valide


def _recursif(formule):
    s = "".join(c for c in formule if c in "()[]{}")

    def lire(i, fermant):
        while i < len(s):
            if s[i] in "([{":
                i = lire(i + 1, {"(": ")", "[": "]", "{": "}"}[s[i]])
                if i is None:
                    return None
            else:
                return i + 1 if s[i] == fermant else None
        return i if fermant is None else None

    return lire(0, None) == len(s)


@pytest.mark.parametrize("fonction,attendu", [
    (lambda f: all(f.count(o) == f.count(c) for c, o in _FERMANTS.items()), "l'ORDRE compte"),
    (_pile(vide=False), "pile VIDE"),
    (_pile(type_=False), "ouvrant ATTENDU"),
    (_pile(fin=False), "return not pile"),
    (_pile(booleen=False), "vrai booléen"),
    (_recursif, "RecursionError"),
], ids=["compter", "pile_vide", "sans_type", "sans_verification_finale", "renvoie_la_pile", "recursif"])
def test_parenthesage_valide_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("parenthesage_valide", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_parenthesage_valide_par_remplacements_est_accepte():
    """Juste, même si O(n²) : on n'exige pas la pile, on l'enseigne."""

    def remplacer(formule):
        s = "".join(c for c in formule if c in "()[]{}")
        while (plus_court := s.replace("()", "").replace("[]", "").replace("{}", "")) != s:
            s = plus_court
        return not s

    assert _evaluer("parenthesage_valide", remplacer).statut == REUSSI


def _fenetre(tri="copie", condition=">="):
    """Une version de pic_appels avec une seule erreur à la fois."""

    def pic(jours, largeur=7):
        if tri == "sur_place":
            jours.sort()
            t = jours
        else:
            t = sorted(jours) if tri == "copie" else list(jours)
        meilleur = gauche = 0
        for droite, jour in enumerate(t):
            ecart = largeur if condition != ">=-1" else largeur - 1
            while (jour - t[gauche] > ecart) if condition == ">" else (jour - t[gauche] >= ecart):
                gauche += 1
            meilleur = max(meilleur, droite - gauche + 1)
        return meilleur

    return pic


def _recompter(jours, largeur=7):
    return max((sum(1 for x in jours if d <= x < d + largeur) for d in jours), default=0)


@pytest.mark.parametrize("fonction,attendu", [
    (_fenetre(condition=">"), "couvre 8 jours"),
    (_fenetre(condition=">=-1"), "ne couvre que 6 jours"),
    (_fenetre(tri="aucun"), "TRIÉE"),
    (_fenetre(tri="sur_place"), "sorted(jours)"),
    (_recompter, "O(n²)"),
], ids=["fenetre_trop_large", "fenetre_trop_courte", "sans_tri", "tri_sur_place", "recompte_tout"])
def test_pic_appels_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("pic_appels", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def _fusion(tri="copie", lendemain=True, maximum=True, inclus=True):
    """Une version de fusionner_periodes avec une seule erreur à la fois."""
    from datetime import timedelta

    def fusionner(periodes):
        if tri == "sur_place":
            periodes.sort()
            liste = periodes
        else:
            liste = sorted(periodes) if tri == "copie" else list(periodes)
        resultat = []
        for debut, fin in liste:
            if resultat:
                limite = resultat[-1][1] + (timedelta(days=1) if lendemain else timedelta(0))
                if (debut <= limite) if inclus else (debut < limite):
                    resultat[-1] = (resultat[-1][0], max(resultat[-1][1], fin) if maximum else fin)
                    continue
            resultat.append((debut, fin))
        return resultat

    return fusionner


def _numero_du_jour_plus_un(periodes):
    from datetime import date

    resultat = []
    for debut, fin in sorted(periodes):
        if resultat:
            derniere = resultat[-1][1]
            try:
                lendemain = date(derniere.year, derniere.month, derniere.day + 1)
            except ValueError:  # le 32 janvier n'existe pas…
                lendemain = derniere
            if debut <= lendemain:
                resultat[-1] = (resultat[-1][0], max(derniere, fin))
                continue
        resultat.append((debut, fin))
    return resultat


@pytest.mark.parametrize("fonction,attendu", [
    (_fusion(tri="aucun"), "TRIÉS"),
    (_fusion(tri="sur_place"), "sorted(periodes)"),
    (_fusion(lendemain=False), "LENDEMAIN"),
    (_fusion(lendemain=False, inclus=False), "LENDEMAIN"),
    (_fusion(maximum=False), "max(fin_en_cours, fin)"),
    (_numero_du_jour_plus_un, "timedelta"),
], ids=["sans_tri", "tri_sur_place", "sans_lendemain", "chevauchement_strict", "periode_contenue", "jour_plus_un"])
def test_fusionner_periodes_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("fusionner_periodes", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_fusionner_periodes_accepte_des_listes():
    from defis.corrige_serie_09 import fusionner_periodes

    resultat = _evaluer("fusionner_periodes", lambda p: [list(x) for x in fusionner_periodes(p)])
    assert resultat.statut == REUSSI, resultat.message


def _tas(cle="-position", trier=True):
    """Une version manuelle de top_k_flux avec une seule erreur à la fois."""
    import heapq

    def top(flux, k):
        tas = []
        for position, (client, proba) in enumerate(flux):
            if k <= 0:
                break
            element = {"-position": (proba, -position, client), "position": (proba, position, client),
                       "client": (proba, client)}[cle]
            if len(tas) < k:
                heapq.heappush(tas, element)
            elif element > tas[0]:
                heapq.heapreplace(tas, element)
        ordre = sorted(tas, reverse=True) if trier else tas
        return [(e[-1], e[0]) for e in ordre]

    return top


def _deux_passages(flux, k):
    import heapq

    max((proba for _, proba in flux), default=None)  # le générateur est maintenant épuisé
    return heapq.nlargest(k, flux, key=lambda s: s[1])


@pytest.mark.parametrize("fonction,attendu", [
    (lambda flux, k: sorted(flux, key=lambda s: s[1], reverse=True)[:k], "EN MÊME TEMPS"),
    (lambda flux, k: sorted(flux, key=lambda s: s[1], reverse=True)[:k] if len(flux) else [], "GÉNÉRATEUR"),
    (_tas(cle="client"), "ordre doit être celui d'ARRIVÉE"),
    (_tas(cle="position"), "-position"),
    (_tas(trier=False), "DÉCROISSANTE"),
    (lambda flux, k: __import__("heapq").nsmallest(k, flux, key=lambda s: s[1]), "PLUS FAIBLES"),
    (_deux_passages, "qu'UNE fois"),
    (lambda flux, k: [(p, c) for c, p in __import__("heapq").nlargest(k, flux, key=lambda s: s[1])], "(customer_id, proba)"),
], ids=["tout_trier", "len_du_generateur", "egalite_par_identifiant", "egalite_par_position", "tas_non_trie",
        "les_plus_petits", "deux_passages", "tuple_inverse"])
def test_top_k_flux_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("top_k_flux", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_top_k_flux_nlargest_est_accepte():
    import heapq

    resultat = _evaluer("top_k_flux", lambda flux, k: heapq.nlargest(k, flux, key=lambda s: s[1]))
    assert resultat.statut == REUSSI, resultat.message


def _parcours(deux_sens=True, multi_sources=True, file=True, exclure_partis=True, balayer_liens=False):
    """Une version de clients_exposes avec une seule erreur à la fois."""
    from collections import deque

    def exposes(parrainages, partis, distance_max=2):
        voisins = {}
        for a, b in parrainages:
            voisins.setdefault(a, []).append(b)
            if deux_sens:
                voisins.setdefault(b, []).append(a)

        def voisins_de(client):
            if balayer_liens:  # reparcourt tous les liens pour chaque client : O(V × E)
                return [b for a, b in parrainages if a == client] + [a for a, b in parrainages if b == client]
            return voisins.get(client, [])

        def bfs(sources):
            distance = {s: 0 for s in sources}
            attente = deque(sources)
            while attente:
                client = attente.popleft() if file else attente.pop()
                if distance[client] < distance_max:
                    for v in voisins_de(client):
                        if v not in distance:
                            distance[v] = distance[client] + 1
                            attente.append(v)
            return distance

        if multi_sources:
            distance = bfs(list(partis))
        else:
            distance = {}
            for p in partis:
                distance.update(bfs([p]))
        if not exclure_partis:
            return {c: d or 1 for c, d in distance.items() if d > 0 or any(c in lien for lien in parrainages)}
        return {c: d for c, d in distance.items() if d > 0 and c not in set(partis)}

    return exposes


def _profondeur_recursive(parrainages, partis, distance_max=2):
    voisins = {}
    for a, b in parrainages:
        voisins.setdefault(a, []).append(b)
        voisins.setdefault(b, []).append(a)
    distance = {p: 0 for p in partis}

    def visiter(client):
        if distance[client] < distance_max:
            for v in voisins.get(client, []):
                if v not in distance:
                    distance[v] = distance[client] + 1
                    visiter(v)

    for p in list(partis):
        visiter(p)
    return {c: d for c, d in distance.items() if d > 0}


@pytest.mark.parametrize("fonction,attendu", [
    (_parcours(deux_sens=False), "DEUX sens"),
    (_parcours(multi_sources=False), "PLUS PROCHE"),
    (_parcours(file=False), "LARGEUR"),
    (_parcours(exclure_partis=False), "déjà parti"),
    (_parcours(balayer_liens=True), "O(V × E)"),
    (_profondeur_recursive, "LARGEUR"),
], ids=["sens_unique", "un_parcours_par_parti", "pile_au_lieu_de_file", "partis_inclus", "voisins_recalcules",
        "profondeur_recursive"])
def test_clients_exposes_erreurs_classiques(fonction, attendu):
    resultat = _evaluer("clients_exposes", fonction)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_clients_exposes_recursion_sur_une_longue_chaine():
    """Une profondeur récursive qui garde le minimum : juste… jusqu'à la chaîne de 3 000 parrainages."""

    def profondeur_minimale(parrainages, partis, distance_max=2):
        voisins = {}
        for a, b in parrainages:
            voisins.setdefault(a, []).append(b)
            voisins.setdefault(b, []).append(a)
        distance = {}

        def visiter(client, d):
            if d > distance_max or distance.get(client, d + 1) <= d:
                return
            distance[client] = d
            for v in voisins.get(client, []):
                visiter(v, d + 1)

        for p in partis:
            visiter(p, 0)
        return {c: d for c, d in distance.items() if d > 0}

    resultat = _evaluer("clients_exposes", profondeur_minimale)
    assert resultat.statut == ECHEC and "RecursionError" in resultat.message, resultat.message


# --------------------------------------------------------------------------- Série 10 : SQL pour l'analyse


def _sql_10(nom, *remplacements):
    """La requête du corrigé de la série 10, avec une seule erreur introduite à la fois."""
    import defis.corrige_serie_10 as corrige

    sql = getattr(corrige, nom)()
    for avant, apres in remplacements:
        assert avant in sql, f"{avant!r} introuvable dans le corrigé de {nom}"
        sql = sql.replace(avant, apres)
    return sql


def _verdict_10(nom, sql):
    return _evaluer(nom, lambda: sql)


_BUGS_SERIE_10 = [
    # Défi 62 : NULL
    ("requete_churn_par_region", _sql_10("requete_churn_par_region",
                                         ("COUNT(churn_date)", "COUNT(churn_date IS NOT NULL)")), "0 n'est pas NULL"),
    ("requete_churn_par_region", _sql_10("requete_churn_par_region", ("COUNT(churn_date)", "COUNT(*)")),
     "0 n'est pas NULL"),
    ("requete_churn_par_region",
     "SELECT COALESCE(region, 'Inconnue') AS region, COUNT(*) AS clients, "
     "SUM(CASE WHEN churn_date <> NULL THEN 1 ELSE 0 END) AS partis, "
     "AVG(CASE WHEN churn_date <> NULL THEN 1.0 ELSE 0 END) AS taux_churn "
     "FROM clients GROUP BY 1 ORDER BY taux_churn DESC, region", "IS NULL"),
    ("requete_churn_par_region", _sql_10("requete_churn_par_region",
                                         ("COALESCE(region, 'Inconnue')  AS region", "region AS region"),
                                         ("GROUP BY COALESCE(region, 'Inconnue')", "GROUP BY region")), "COALESCE"),
    ("requete_churn_par_region", _sql_10("requete_churn_par_region",
                                         ("FROM clients\n", "FROM clients WHERE region IS NOT NULL\n")), "disparu"),
    ("requete_churn_par_region", _sql_10("requete_churn_par_region",
                                         ("AVG(churn_date IS NOT NULL)", "COUNT(churn_date) / COUNT(*)")),
     "division ENTIÈRE"),
    ("requete_churn_par_region", _sql_10("requete_churn_par_region",
                                         ("AVG(churn_date IS NOT NULL)", "AVG(churn_date IS NULL)")), "RESTÉS"),
    ("requete_churn_par_region", _sql_10("requete_churn_par_region", (", region ", ", region DESC ")),
     "départager"),
    # Défi 63 : anti-jointure
    ("requete_actifs_sans_appel",
     "SELECT customer_id FROM clients WHERE churn_date IS NULL "
     "AND customer_id NOT IN (SELECT customer_id FROM appels) ORDER BY customer_id", "NOT EXISTS"),
    ("requete_actifs_sans_appel", _sql_10("requete_actifs_sans_appel", ("c.churn_date IS NULL", "1 = 1")),
     "churn_date IS NULL"),
    ("requete_actifs_sans_appel", _sql_10("requete_actifs_sans_appel", ("AND a.call_id IS NULL", "")),
     "B a appelé"),
    ("requete_actifs_sans_appel", _sql_10("requete_actifs_sans_appel", ("c.churn_date IS NULL", "c.churn_date = NULL")),
     "IS NULL"),
    # Défi 64 : LEFT JOIN, ON ou WHERE
    ("requete_appels_recents", _sql_10("requete_appels_recents",
                                       ("ON a.customer_id = c.customer_id\n              AND",
                                        "ON a.customer_id = c.customer_id\n        WHERE"),
                                       ("WHERE c.signup_date", "AND c.signup_date")), "dans le ON"),
    ("requete_appels_recents", _sql_10("requete_appels_recents", ("COUNT(a.call_id)", "COUNT(*)")), "COUNT(a.call_id)"),
    ("requete_appels_recents", _sql_10("requete_appels_recents",
                                       ("a.call_date <  :date_extraction", "a.call_date <= :date_extraction")),
     "jour de l'extraction"),
    ("requete_appels_recents", _sql_10("requete_appels_recents", ("AND a.call_date <  :date_extraction", "")),
     "jour de l'extraction"),
    ("requete_appels_recents", _sql_10("requete_appels_recents", (">= date(", "> date(")), "(>=, pas >)"),
    ("requete_appels_recents", _sql_10("requete_appels_recents", ("'-90 days'", "'-3 months'")), "« 3 mois »"),
    ("requete_appels_recents", _sql_10("requete_appels_recents",
                                       ("AND (c.churn_date IS NULL OR c.churn_date > :date_extraction)",
                                        "AND c.churn_date IS NULL OR c.churn_date > :date_extraction")),
     "priorité des opérateurs"),
    ("requete_appels_recents", _sql_10("requete_appels_recents",
                                       ("AND (c.churn_date IS NULL OR c.churn_date > :date_extraction)",
                                        "AND c.churn_date IS NULL")), "« photos » passées"),
    ("requete_appels_recents", _sql_10("requete_appels_recents", ("c.signup_date <= :date_extraction", "1 = 1")),
     "pas encore client"),
    ("requete_appels_recents", _sql_10("requete_appels_recents",
                                       ("c.churn_date > :date_extraction", "c.churn_date >= :date_extraction")),
     "strictement après"),
    ("requete_appels_recents", _sql_10("requete_appels_recents", (":date_extraction", "'2026-06-30'")), "en dur"),
    ("requete_appels_recents", _sql_10("requete_appels_recents", (":date_extraction", "?")), "paramètre NOMMÉ"),
    ("requete_appels_recents", _sql_10("requete_appels_recents", ("appels_90j DESC, c.customer_id", "c.customer_id")),
     "ORDER BY appels_90j DESC"),
    # Défi 65 : explosion des jointures
    ("requete_bilan_par_contrat",
     "SELECT c.contract, COUNT(*) AS clients, COALESCE(SUM(f.amount), 0) AS chiffre_affaires, "
     "COUNT(a.call_id) AS appels FROM clients c LEFT JOIN factures f ON f.customer_id = c.customer_id "
     "LEFT JOIN appels a ON a.customer_id = c.customer_id GROUP BY c.contract ORDER BY c.contract", "fan-out"),
    ("requete_bilan_par_contrat",
     "SELECT c.contract, COUNT(DISTINCT c.customer_id) AS clients, COALESCE(SUM(f.amount), 0) AS chiffre_affaires, "
     "COUNT(DISTINCT a.call_id) AS appels FROM clients c LEFT JOIN factures f ON f.customer_id = c.customer_id "
     "LEFT JOIN appels a ON a.customer_id = c.customer_id GROUP BY c.contract ORDER BY c.contract",
     "DISTINCT ne répare pas"),
    ("requete_bilan_par_contrat", _sql_10("requete_bilan_par_contrat", ("LEFT JOIN ca_client", "JOIN ca_client")),
     "Biannuel a disparu"),
    ("requete_bilan_par_contrat", _sql_10("requete_bilan_par_contrat",
                                          ("COALESCE(SUM(f.montant), 0)", "SUM(f.montant)")), "COALESCE(SUM(…), 0)"),
    ("requete_bilan_par_contrat", _sql_10("requete_bilan_par_contrat", ("ORDER BY c.contract", "ORDER BY clients DESC")),
     "ORDER BY contract"),
    # Défi 66 : fonctions de fenêtre
    ("requete_evolution_mensuelle", _sql_10("requete_evolution_mensuelle",
                                            ("strftime('%Y-%m', invoice_date)", "invoice_date")), "une par mois"),
    ("requete_evolution_mensuelle", _sql_10("requete_evolution_mensuelle",
                                            ("strftime('%Y-%m', invoice_date)", "strftime('%m', invoice_date)")),
     "'AAAA-MM'"),
    ("requete_evolution_mensuelle", _sql_10("requete_evolution_mensuelle", ("100.0 *", "100 *")), "division ENTIÈRE"),
    ("requete_evolution_mensuelle", _sql_10("requete_evolution_mensuelle", ("100.0 *", "1.0 *")), "proportions"),
    ("requete_evolution_mensuelle", _sql_10("requete_evolution_mensuelle",
                                            ("(ORDER BY mois)", "(ORDER BY mois DESC)")), "mois SUIVANT"),
    ("requete_evolution_mensuelle", _sql_10("requete_evolution_mensuelle",
                                            ("100.0 * (", "COALESCE(100.0 * ("),
                                            ("OVER chrono          AS evolution_pct",
                                             "OVER chrono, 0)      AS evolution_pct")), "au lieu de NULL"),
    ("requete_evolution_mensuelle", _sql_10("requete_evolution_mensuelle",
                                            ("/ LAG(chiffre_affaires) OVER chrono ", "/ chiffre_affaires ")),
     "mois COURANT"),
    ("requete_evolution_mensuelle", _sql_10("requete_evolution_mensuelle",
                                            ("SUM(chiffre_affaires) OVER chrono", "SUM(chiffre_affaires) OVER ()")),
     "total général"),
    # Défi 67 : îlots et trous
    ("requete_series_appels", _sql_10("requete_series_appels", ("SELECT DISTINCT", "SELECT")), "DISTINCT"),
    ("requete_series_appels", _sql_10("requete_series_appels",
                                      ("CAST(strftime('%Y', call_date) AS INTEGER) * 12\n                 + ", "")),
     "année × 12 + mois"),
    ("requete_series_appels", _sql_10("requete_series_appels", ("WHERE customer_id IS NOT NULL", "")),
     "numéros inconnus"),
    ("requete_series_appels", _sql_10("requete_series_appels", (":mois_min", "2")), "HAVING"),
    ("requete_series_appels", _sql_10("requete_series_appels", ("HAVING MAX(longueur) >= :mois_min", "")), "HAVING"),
    ("requete_series_appels",
     "SELECT customer_id, COUNT(DISTINCT strftime('%Y-%m', call_date)) AS plus_longue_serie FROM appels "
     "WHERE customer_id IS NOT NULL GROUP BY customer_id HAVING plus_longue_serie >= :mois_min "
     "ORDER BY plus_longue_serie DESC, customer_id", "CONSÉCUTIFS"),
    ("requete_series_appels",
     "SELECT customer_id, COUNT(*) AS plus_longue_serie FROM appels WHERE customer_id IS NOT NULL "
     "GROUP BY customer_id HAVING COUNT(*) >= :mois_min ORDER BY plus_longue_serie DESC, customer_id", "APPELS"),
    ("requete_series_appels",
     "SELECT customer_id, COUNT(*) AS plus_longue_serie FROM appels WHERE COUNT(*) >= :mois_min GROUP BY customer_id",
     "HAVING"),
]


@pytest.mark.parametrize("nom,sql,attendu", _BUGS_SERIE_10,
                         ids=[f"{n.removeprefix('requete_')}-{i}" for i, (n, _, _) in enumerate(_BUGS_SERIE_10)])
def test_serie_10_chaque_piege_sql_a_son_diagnostic(nom, sql, attendu):
    resultat = _verdict_10(nom, sql)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


@pytest.mark.parametrize("nom,sql", [
    ("requete_actifs_sans_appel",
     "SELECT customer_id FROM clients c WHERE churn_date IS NULL "
     "AND NOT EXISTS (SELECT 1 FROM appels a WHERE a.customer_id = c.customer_id) ORDER BY customer_id"),
    ("requete_bilan_par_contrat",
     "SELECT c.contract, COUNT(*) AS clients, "
     "COALESCE(SUM((SELECT SUM(amount) FROM factures f WHERE f.customer_id = c.customer_id)), 0) AS chiffre_affaires, "
     "COALESCE(SUM((SELECT COUNT(*) FROM appels a WHERE a.customer_id = c.customer_id)), 0) AS appels "
     "FROM clients c GROUP BY c.contract ORDER BY c.contract"),
    ("requete_churn_par_region", _sql_10("requete_churn_par_region",
                                         ("GROUP BY COALESCE(region, 'Inconnue')", "GROUP BY region"))),
], ids=["not_exists", "sous_requetes_correlees", "group_by_alias"])
def test_serie_10_une_autre_ecriture_juste_est_acceptee(nom, sql):
    """En SQL, il y a souvent plusieurs écritures justes : le correcteur juge le résultat, pas le style."""
    resultat = _verdict_10(nom, sql)
    assert resultat.statut == REUSSI, resultat.message


def test_serie_10_la_base_d_exemple_est_reproductible_et_piegee():
    """La base de démonstration doit contenir les pièges dont parlent les docstrings."""
    from defis.donnees_sql import generer_tables

    a, b = generer_tables(n_clients=50, seed=3), generer_tables(n_clients=50, seed=3)
    for nom in a:
        pd.testing.assert_frame_equal(a[nom], b[nom])
    tables = generer_tables()
    assert tables["clients"]["region"].isna().any(), "il faut des régions inconnues (défi 62)"
    assert tables["appels"]["customer_id"].isna().any(), "il faut des appels anonymes (défis 63 et 67)"
    assert tables["clients"]["churn_date"].notna().any() and tables["clients"]["churn_date"].isna().any()
    premiere = tables["factures"].groupby("customer_id")["invoice_date"].min()
    inscription = tables["clients"].set_index("customer_id")["signup_date"]
    assert (premiere >= inscription.loc[premiere.index]).all(), "aucune facture avant l'inscription"


# --------------------------------------------------------------------------- Série 11 : tester comme un·e pro


def _segment_ref(mois):
    return "nouveau" if mois < 6 else "installé" if mois < 24 else "fidèle"


@pytest.mark.parametrize("cas,attendu", [
    ([(3, "nouveau"), (12, "installé"), (40, "fidèle")], "6 mois"),          # milieux de zone : aucune frontière
    ([(5, "nouveau"), (6, "installé"), (30, "fidèle")], "DEUX frontières"),  # la seconde frontière est oubliée
    ([(5, "nouveau"), (6, "installé"), (23, "installé"), (24, "fidèle")], "0 mois"),
    ([(m, _segment_ref(m)) for m in range(73)], "au plus 6"),               # tout essayer n'est pas tester
    ([(6, "nouveau")], "est faux"),
], ids=["milieux", "une_frontiere", "sans_zero", "exhaustif", "cas_faux"])
def test_serie_11_frontieres(cas, attendu):
    resultat = _evaluer("cas_limites_anciennete", lambda: cas)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_serie_11_egalite_exacte_entre_flottants_est_fragile():
    """`== 1 / 3` passe sur la fonction fournie mais casse sur une version juste calculée autrement."""

    def egalite_exacte(taux_churn):
        assert taux_churn(1, 4) == 0.25
        assert taux_churn(1, 3) == 1 / 3
        assert taux_churn(0, 0) == 0.0

    resultat = _evaluer("tester_taux_churn", egalite_exacte)
    assert resultat.statut == ECHEC and "JUSTE" in resultat.message and "pytest.approx" in resultat.message


def test_serie_11_tolerance_trop_large_laisse_passer_l_arrondi():
    def tolerance_large(taux_churn):
        assert taux_churn(1, 4) == pytest.approx(0.25)
        assert taux_churn(1, 3) == pytest.approx(0.33, abs=0.01)
        assert taux_churn(0, 0) == 0.0

    resultat = _evaluer("tester_taux_churn", tolerance_large)
    assert resultat.statut == ECHEC and "arrondit à 2 décimales" in resultat.message


def _contrat_valides(lire_contrat):
    assert lire_contrat("mensuel") == "Mensuel"
    assert lire_contrat("  MENSUEL ") == "Mensuel"
    assert lire_contrat("1 an") == "1 an"
    assert lire_contrat("2 ans") == "2 ans"


def test_serie_11_un_try_except_ne_detecte_pas_l_absence_d_erreur():
    def try_except(lire_contrat):
        _contrat_valides(lire_contrat)
        try:
            lire_contrat("Mensual")
        except ValueError:
            pass

    resultat = _evaluer("tester_lire_contrat", try_except)
    assert resultat.statut == ECHEC and "renvoie None" in resultat.message


def test_serie_11_raises_exception_est_trop_large():
    def trop_large(lire_contrat):
        _contrat_valides(lire_contrat)
        with pytest.raises(Exception):  # sans match : toute erreur est acceptée
            lire_contrat("Mensual")

    resultat = _evaluer("tester_lire_contrat", trop_large)
    assert resultat.statut == ECHEC and "KeyError" in resultat.message


def test_serie_11_verifier_toute_la_phrase_du_message_est_fragile():
    def phrase_entiere(lire_contrat):
        _contrat_valides(lire_contrat)
        with pytest.raises(ValueError, match="Contrat inconnu"):
            lire_contrat("Mensual")

    resultat = _evaluer("tester_lire_contrat", phrase_entiere)
    assert resultat.statut == ECHEC and "JUSTE" in resultat.message and "match='Mensual'" in resultat.message


def test_serie_11_un_raises_qui_ne_voit_rien_venir_est_un_echec_pas_un_plantage():
    """L'échec de pytest.raises dérive de BaseException : il ne doit pas faire planter le correcteur."""

    def attend_une_erreur_sur_une_saisie_valide(lire_contrat):
        with pytest.raises(ValueError):
            lire_contrat("mensuel")

    resultat = _evaluer("tester_lire_contrat", attend_une_erreur_sur_une_saisie_valide)
    assert resultat.statut == ECHEC and "vraiment invalide" in resultat.message


def test_serie_11_verifier_le_texte_du_message_envoye_est_fragile():
    def texte_exact(relancer_clients):
        envois = []
        relancer_clients([{"customer_id": "C1", "proba": 0.9}], lambda cid, msg: envois.append((cid, msg)))
        assert envois == [("C1", "Bonjour, profitez de -20 % sur votre prochain mois (C1).")]

    resultat = _evaluer("tester_relancer_clients", texte_exact)
    assert resultat.statut == ECHEC and "texte est libre" in resultat.message


def test_serie_11_sans_panne_simulee_l_abandon_passe_inapercu():
    def sans_panne(relancer_clients):
        envois = []
        clients = [{"customer_id": "C1", "proba": 0.9}, {"customer_id": "C2", "proba": 0.1},
                   {"customer_id": "C3", "proba": 0.5}, {"customer_id": "C1", "proba": 0.9}]
        assert relancer_clients(clients, lambda cid, msg: envois.append(cid)) == 2
        assert envois == ["C1", "C3"]
        envois.clear()
        assert relancer_clients(clients, lambda cid, msg: envois.append(cid), seuil=0.8) == 1

    resultat = _evaluer("tester_relancer_clients", sans_panne)
    assert resultat.statut == ECHEC and "ConnectionError" in resultat.message


def test_serie_11_un_test_qui_ecrit_hors_du_dossier_temporaire_est_refuse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # même si le correcteur se trompait, rien n'atterrirait dans le dépôt

    def ecrit_ici(exporter_rapport, dossier):
        exporter_rapport(pd.DataFrame({"customer_id": ["C1"], "proba": [0.9]}), Path("."))

    resultat = _evaluer("tester_exporter_rapport", ecrit_ici)
    assert resultat.statut == ECHEC and "dossier courant" in resultat.message
    assert list(tmp_path.iterdir()) == [], "le correcteur doit lancer le test dans un dossier jetable"


def test_serie_11_relire_sans_le_bon_separateur_est_explique():
    def sans_sep(exporter_rapport, dossier):
        exporter_rapport(pd.DataFrame({"customer_id": ["C1", "C2"], "proba": [0.9, 0.7]}), dossier)
        assert list(pd.read_csv(dossier / "clients_a_risque.csv").columns) == ["customer_id", "proba"]

    resultat = _evaluer("tester_exporter_rapport", sans_sep)
    assert resultat.statut == ECHEC and "sep=';'" in resultat.message


def test_serie_11_relire_le_chemin_renvoye_ne_suffit_pas():
    def via_le_chemin_renvoye(exporter_rapport, dossier):
        scores = pd.DataFrame({"customer_id": ["C1", "C2", "C3"], "proba": [0.6, 0.9, 0.7]})
        rapport = pd.read_csv(exporter_rapport(scores, dossier), sep=";")
        assert list(rapport.columns) == ["customer_id", "proba"]
        assert list(rapport["customer_id"]) == ["C2", "C3", "C1"]
        with pytest.raises(FileExistsError):
            exporter_rapport(scores, dossier)

    resultat = _evaluer("tester_exporter_rapport", via_le_chemin_renvoye)
    assert resultat.statut == ECHEC and "dossier COURANT" in resultat.message


def _proprietes(repartir, max_clients=250, ordre=True, copie=True):
    hasard = random.Random(0)
    for _ in range(300):
        clients = [f"C{i}" for i in range(hasard.randint(0, max_clients))]
        n, avant = hasard.randint(1, 15), list(clients)
        groupes = repartir(clients, n)
        assert len(groupes) == n
        recolles = [c for g in groupes for c in g]
        assert (recolles == avant) if ordre else (sorted(recolles) == sorted(avant))
        tailles = [len(g) for g in groupes]
        assert max(tailles) - min(tailles) <= 1
        if copie:
            assert clients == avant


@pytest.mark.parametrize("test,attendu", [
    (lambda repartir: _proprietes(repartir, max_clients=40), "100e"),
    (lambda repartir: _proprietes(repartir, ordre=False), "tour de rôle"),
    (lambda repartir: _proprietes(repartir, copie=False), "garantie 4"),
    (lambda repartir: [repartir([f"C{i}" for i in range(10)], 3)], "TOUTES"),  # n'affirme rien
], ids=["petites_listes", "ensembles", "sans_copie", "aucun_assert"])
def test_serie_11_chaque_propriete_oubliee_laisse_passer_son_bug(test, attendu):
    resultat = _evaluer("tester_repartir", test)
    assert resultat.statut == ECHEC and attendu in resultat.message, resultat.message


def test_serie_11_un_resultat_exact_non_promis_est_fragile():
    def resultat_exact(repartir):
        assert repartir(list("abcdefghij"), 3) == [list("abcd"), list("efg"), list("hij")]

    resultat = _evaluer("tester_repartir", resultat_exact)
    assert resultat.statut == ECHEC and "DERNIERS conseillers" in resultat.message


def test_serie_11_les_fonctions_fournies_sont_celles_du_correcteur():
    """L'élève essaie ses tests sur les fonctions de serie_11.py : elles doivent se comporter comme les références."""
    import defis.checks_serie_11 as checks
    import defis.corrige_serie_11 as corrige
    import defis.serie_11 as eleve

    assert [eleve.segment_anciennete(m) for m in range(80)] == [checks._segment(m) for m in range(80)]
    corrige.tester_taux_churn(eleve.taux_churn)
    corrige.tester_lire_contrat(eleve.lire_contrat)
    corrige.tester_relancer_clients(eleve.relancer_clients)
    corrige.tester_repartir(eleve.repartir)
    with tempfile.TemporaryDirectory() as dossier:
        corrige.tester_exporter_rapport(eleve.exporter_rapport, Path(dossier))


# --------------------------------------------------------------------------- Série 12 : construire une API web
#
# Chaque « mutant » ci-dessous est une version du corrigé avec UNE erreur classique. Le correcteur doit la
# refuser (❌) avec un message qui nomme l'erreur et sa correction.

from typing import Literal  # noqa: E402

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

_Contrat12 = Literal["Mensuel", "Annuel", "Biannuel"]


def _echec_12(nom, fonction, *mots):
    resultat = _evaluer(nom, fonction)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


def _lecture(plante=False, null=False, code=404, detail=True, copie=False, retour=True):
    def routes_lecture(clients):
        base = dict(clients) if copie else clients
        routeur = APIRouter()

        @routeur.get("/clients/{client_id}")
        def lire(client_id: str):
            if plante:
                return base[client_id]
            if null:
                return base.get(client_id)
            if client_id not in base:
                raise HTTPException(code, f"Client {client_id} introuvable" if detail else "Introuvable")
            return base[client_id]

        return routeur if retour else None
    return routes_lecture


@pytest.mark.parametrize("mutant,mots", [
    (_lecture(plante=True), ("500", "KeyError", "HTTPException(status_code=404")),
    (_lecture(null=True), ("null",)),
    (_lecture(code=400), ("400",)),
    (_lecture(detail=False), ("citer l'identifiant",)),
    (_lecture(copie=True), ("copie",)),
    (_lecture(retour=False), ("return routeur",)),
], ids=["keyerror", "null", "400", "detail_muet", "copie", "sans_return"])
def test_serie_12_lecture_erreurs_classiques(mutant, mots):
    _echec_12("routes_lecture", mutant, *mots)


def test_serie_12_une_mauvaise_adresse_est_signalee():
    def mauvais_chemin(clients):
        routeur = APIRouter()

        @routeur.get("/client/{client_id}")
        def lire(client_id: str):
            return clients[client_id]
        return routeur

    def mauvaise_methode(clients):
        routeur = APIRouter()

        @routeur.post("/clients/{client_id}")
        def lire(client_id: str):
            return clients[client_id]
        return routeur

    _echec_12("routes_lecture", mauvais_chemin, "introuvable", "/client/{client_id}")
    _echec_12("routes_lecture", mauvaise_methode, "@routeur.get")


def _recherche(trier=True, limite_defaut=20, strict=False, couper_avant=False, contrat_requis=False,
               bornes=True, le_100=True, vide_404=False):
    def routes_recherche(clients):
        routeur = APIRouter()
        q_appels = Query(0, ge=0) if bornes else 0
        q_limite = Query(limite_defaut, ge=1, le=100) if le_100 else Query(limite_defaut, ge=1, lt=100)
        if not bornes:
            q_limite = limite_defaut

        def chercher(contrat, min_appels, limite):
            tous = sorted(clients.values(), key=lambda c: c["id"]) if trier else list(clients.values())
            if couper_avant:
                tous = tous[:limite]
            trouves = [c for c in tous if (contrat is None or c["contrat"] == contrat)
                       and (c["appels_support"] > min_appels if strict else c["appels_support"] >= min_appels)]
            if vide_404 and not trouves:
                raise HTTPException(404, "aucun client")
            return trouves if couper_avant else trouves[:limite]

        if contrat_requis:
            @routeur.get("/clients")
            def avec_contrat(contrat: str, min_appels: int = q_appels, limite: int = q_limite):
                return chercher(contrat, min_appels, limite)
        else:
            @routeur.get("/clients")
            def sans_contrat(contrat: str | None = None, min_appels: int = q_appels, limite: int = q_limite):
                return chercher(contrat, min_appels, limite)
        return routeur
    return routes_recherche


@pytest.mark.parametrize("mutant,mots", [
    (_recherche(trier=False), ("Triez",)),
    (_recherche(limite_defaut=1000), ("20 par défaut",)),
    (_recherche(strict=True), ("`>=`",)),
    (_recherche(couper_avant=True), ("Filtrez d'abord",)),
    (_recherche(contrat_requis=True), ("valeur par défaut",)),
    (_recherche(bornes=False), ("ge=0",)),
    (_recherche(le_100=False), ("le=100",)),
    (_recherche(vide_404=True), ("pas une erreur",)),
], ids=["non_trie", "sans_limite", "strict", "coupe_avant", "contrat_requis", "sans_bornes", "lt_100", "vide_404"])
def test_serie_12_recherche_erreurs_classiques(mutant, mots):
    _echec_12("routes_recherche", mutant, *mots)


class _ClientJuste(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^C\d{3}$")
    anciennete_mois: int = Field(ge=0)
    facture_mensuelle: float = Field(gt=0, le=500)
    contrat: _Contrat12
    appels_support: int = Field(default=0, ge=0)


class _ClientSansAncre(_ClientJuste):
    id: str = Field(pattern=r"C\d{3}")


class _ClientGe0(_ClientJuste):
    facture_mensuelle: float = Field(ge=0, le=500)


class _ClientLt500(_ClientJuste):
    facture_mensuelle: float = Field(gt=0, lt=500)


class _ClientAppelsRequis(_ClientJuste):
    appels_support: int = Field(ge=0)


class _ClientExtrasIgnores(_ClientJuste):
    model_config = ConfigDict(extra="ignore")


def _creation(modele=_ClientJuste, code=201, location=True, ecrase=False, objet=False):
    def routes_creation(clients):
        routeur = APIRouter()

        @routeur.post("/clients", status_code=code)
        def creer(nouveau: modele, response: Response):
            if nouveau.id in clients and not ecrase:
                raise HTTPException(409, "existe déjà")
            clients[nouveau.id] = nouveau if objet else nouveau.model_dump()
            if location:
                response.headers["Location"] = f"/clients/{nouveau.id}"
            return nouveau.model_dump()
        return routeur
    return routes_creation


@pytest.mark.parametrize("mutant,mots", [
    (_creation(code=200), ("201",)),
    (_creation(location=False), ("Location",)),
    (_creation(ecrase=True), ("ÉCRASÉ",)),
    (_creation(objet=True), ("dictionnaire ordinaire",)),
    (_creation(modele=_ClientSansAncre), ("C0420", "ancrez")),
    (_creation(modele=_ClientGe0), ("gt=0",)),
    (_creation(modele=_ClientLt500), ("le=500",)),
    (_creation(modele=_ClientAppelsRequis), ("FACULTATIF",)),
    (_creation(modele=_ClientExtrasIgnores), ("apels_support", "extra=")),
], ids=["code_200", "sans_location", "ecrase", "objet_pydantic", "motif_sans_ancre", "ge_0", "lt_500",
        "appels_requis", "extras_ignores"])
def test_serie_12_creation_erreurs_classiques(mutant, mots):
    _echec_12("routes_creation", mutant, *mots)


class _ModifJuste(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anciennete_mois: int = Field(default=None, ge=0)
    facture_mensuelle: float = Field(default=None, gt=0, le=500)
    contrat: _Contrat12 = None
    appels_support: int = Field(default=None, ge=0)


class _ModifOptionnel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anciennete_mois: int | None = Field(default=None, ge=0)
    facture_mensuelle: float | None = Field(default=None, gt=0, le=500)
    contrat: _Contrat12 | None = None
    appels_support: int | None = Field(default=None, ge=0)


class _ModifExtrasIgnores(_ModifJuste):
    model_config = ConfigDict(extra="ignore")


def _modification(modele=_ModifJuste, mode="exclude_unset", plante_inconnu=False):
    def routes_modification(clients):
        routeur = APIRouter()

        @routeur.patch("/clients/{client_id}")
        def modifier(client_id: str, modification: modele):
            if client_id not in clients and not plante_inconnu:
                raise HTTPException(404, "introuvable")
            client = clients[client_id]
            if mode == "exclude_unset":
                changements = modification.model_dump(exclude_unset=True)
            elif mode == "tout":
                changements = modification.model_dump()
            elif mode == "si_vrai":
                changements = {k: v for k, v in modification.model_dump().items() if v}
            elif mode == "pas_none":
                changements = {k: v for k, v in modification.model_dump().items() if v is not None}
            clients[client_id] = {**client, **changements}
            return clients[client_id]
        return routeur
    return routes_modification


def _modification_champ_par_champ(clients):
    """Valide « à la main », champ par champ, en écrivant au fur et à mesure : un refus arrive trop tard."""
    routeur = APIRouter()

    @routeur.patch("/clients/{client_id}")
    def modifier(client_id: str, corps: dict):
        if client_id not in clients:
            raise HTTPException(404, "introuvable")
        for champ, valeur in corps.items():
            if champ not in ("anciennete_mois", "facture_mensuelle", "contrat", "appels_support") or valeur is None:
                raise HTTPException(422, f"champ {champ} refusé")
            if champ == "facture_mensuelle" and not 0 < valeur <= 500:
                raise HTTPException(422, "facture invalide")
            if champ in ("anciennete_mois", "appels_support") and valeur < 0:
                raise HTTPException(422, f"{champ} négatif")
            if champ == "contrat" and valeur not in ("Mensuel", "Annuel", "Biannuel"):
                raise HTTPException(422, "contrat invalide")
            clients[client_id][champ] = valeur
        return clients[client_id]
    return routeur


@pytest.mark.parametrize("mutant,mots", [
    (_modification(mode="tout", modele=_ModifOptionnel), ("exclude_unset",)),
    (_modification(mode="si_vrai"), ("0 est une VALEUR",)),
    (_modification(mode="pas_none", modele=_ModifOptionnel), ("null",)),
    (_modification(modele=_ModifOptionnel), ("null",)),
    (_modification(modele=_ModifExtrasIgnores), ("id", "extra")),
    (_modification(plante_inconnu=True), ("500", "404")),
    (_modification_champ_par_champ, ("Tout ou rien",)),
], ids=["model_dump_complet", "if_valeur", "ignore_null", "accepte_null", "extras_ignores", "inconnu_plante",
        "champ_par_champ"])
def test_serie_12_modification_erreurs_classiques(mutant, mots):
    _echec_12("routes_modification", mutant, *mots)


def _admin(proteger_delete=True, inverser=False, existence_avant=False, copier_cles=False, via_query=False,
           code_delete=204):
    def routes_admin(clients, cles_valides):
        cles = set(cles_valides) if copier_cles else cles_valides

        def controler(cle):
            if cle is None:
                raise HTTPException(403 if inverser else 401, "clé manquante")
            if cle not in cles:
                raise HTTPException(401 if inverser else 403, "accès refusé")

        if via_query:
            def verifier_cle(x_api_key: str | None = None):
                controler(x_api_key)
        else:
            def verifier_cle(x_api_key: str | None = Header(default=None)):
                controler(x_api_key)

        routeur = APIRouter(prefix="/admin")

        @routeur.get("/clients", dependencies=[Depends(verifier_cle)])
        def tous():
            return sorted(clients.values(), key=lambda c: c["id"])

        @routeur.delete("/clients/{client_id}", status_code=code_delete)
        def supprimer(client_id: str, x_api_key: str | None = Header(default=None)):
            if existence_avant and client_id not in clients:
                raise HTTPException(404, "introuvable")
            if proteger_delete:
                controler(x_api_key)
            if client_id not in clients:
                raise HTTPException(404, "introuvable")
            del clients[client_id]
        return routeur
    return routes_admin


@pytest.mark.parametrize("mutant,mots", [
    (_admin(proteger_delete=False), ("SUPPRIMÉ",)),
    (_admin(inverser=True), ("inversés",)),
    (_admin(existence_avant=True), ("intrus",)),
    (_admin(copier_cles=True), ("RÉVOQUÉE",)),
    (_admin(via_query=True), ("Header",)),
    (_admin(code_delete=200), ("204",)),
], ids=["delete_ouvert", "401_403_inverses", "existence_avant", "cles_copiees", "cle_dans_l_adresse", "delete_200"])
def test_serie_12_admin_erreurs_classiques(mutant, mots):
    _echec_12("routes_admin", mutant, *mots)


class _Demande12(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_id: str
    offre: Literal["remise_10", "mois_offert", "appel_conseiller"]


_MEMOIRE_GLOBALE_12: dict = {}


def _relances(memoriser=True, rejeu_409=False, comparer_corps=True, retenir_avant=False, attraper=True,
              globale=False, cle_corps=False, meme_id=False, anti_doublon_contenu=False):
    def routes_relances(clients, envoyer):
        routeur = APIRouter()
        deja_vu = _MEMOIRE_GLOBALE_12 if globale else {}
        compteur = [0]

        @routeur.post("/relances", status_code=201)
        def relancer(demande: _Demande12, idempotency_key: str | None = Header(default=None)):
            if not idempotency_key:
                raise HTTPException(400, "Idempotency-Key obligatoire")
            corps = demande.model_dump()
            cle = str(sorted(corps.items())) if cle_corps else idempotency_key
            if memoriser and cle in deja_vu:
                ancien_corps, reponse = deja_vu[cle]
                if rejeu_409:
                    raise HTTPException(409, "déjà traitée")
                if comparer_corps and corps != ancien_corps:
                    raise HTTPException(422, "clé déjà utilisée")
                return reponse
            if anti_doublon_contenu:  # « par sécurité », une seule relance par contenu, quelle que soit la clé
                for ancien_corps, reponse in deja_vu.values():
                    if ancien_corps == corps:
                        return reponse
            if demande.client_id not in clients:
                raise HTTPException(404, "client inconnu")
            if retenir_avant:
                deja_vu[cle] = (corps, None)
            if attraper:
                try:
                    envoyer(demande.client_id, demande.offre)
                except Exception:
                    raise HTTPException(503, "réessayez") from None
            else:
                envoyer(demande.client_id, demande.offre)
            compteur[0] += 1
            reponse = {"relance_id": "R" if meme_id else f"R{compteur[0]}", **corps}
            if memoriser:
                deja_vu[cle] = (corps, reponse)
            return reponse
        return routeur
    return routes_relances


@pytest.mark.parametrize("mutant,mots", [
    (_relances(memoriser=False), ("DEUXIÈME SMS",)),
    (_relances(rejeu_409=True), ("MÊME réponse",)),
    (_relances(comparer_corps=False), ("corps différent",)),
    (_relances(retenir_avant=True), ("AVANT l'envoi",)),
    (_relances(attraper=False), ("503", "ConnectionError")),
    (_relances(cle_corps=True), ("AUTRE offre",)),
    (_relances(anti_doublon_contenu=True), ("que sur la clé",)),
    (_relances(meme_id=True), ("identifiant unique",)),
], ids=["sans_memoire", "rejeu_409", "corps_non_compare", "retenue_avant_envoi", "panne_non_attrapee",
        "cle_ignoree", "anti_doublon_par_contenu", "meme_relance_id"])
def test_serie_12_relances_erreurs_classiques(mutant, mots):
    _echec_12("routes_relances", mutant, *mots)


def test_serie_12_une_memoire_globale_est_detectee():
    _MEMOIRE_GLOBALE_12.clear()
    _echec_12("routes_relances", _relances(globale=True), "partagée", "DANS `routes_relances`")


def test_serie_12_les_tests_du_corrige_passent():
    """Les tests « comme dans tests/ » en fin de corrigé sont de vrais tests : ils doivent passer."""
    import defis.corrige_serie_12 as corrige

    tests = [f for nom, f in vars(corrige).items() if nom.startswith("test_")]
    assert len(tests) >= 5
    for test in tests:
        test()


def test_serie_12_les_outils_du_fichier_eleve_sont_ceux_du_correcteur():
    """L'élève essaie ses routes sur exemple_clients() : la base doit être celle du correcteur, et neuve à chaque appel."""
    import defis.checks_serie_12 as checks
    import defis.serie_12 as eleve

    assert eleve.exemple_clients() == checks._BASE
    base = eleve.exemple_clients()
    base["C001"]["contrat"] = "Annuel"
    assert eleve.exemple_clients()["C001"]["contrat"] == "Mensuel"
    assert eleve.CONTRATS == checks._CONTRATS
    client = eleve.essayer(_lecture()(eleve.exemple_clients()))
    assert client.get("/clients/C001").json()["id"] == "C001"


# --------------------------------------------------------------------------- Série 13 : écrire du code sûr

import hashlib  # noqa: E402
import hmac  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import sqlite3  # noqa: E402

_MOTS_13 = ("password", "mot_de_passe", "secret", "token", "api_key", "authorization")
_DEBUTS_13 = ("=", "+", "-", "@", "\t", "\r")


def _echec_13(nom, fonction, *mots):
    resultat = _evaluer(nom, fonction)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


# Défi 80 : injection SQL


def _sql_fstring(connexion, nom, contrat=None):
    return [ligne[0] for ligne in connexion.execute(f"SELECT id FROM clients WHERE nom = '{nom}' ORDER BY id")]


def _sql_echappe(connexion, nom, contrat=None):
    nom = nom.replace("'", "''")
    sql = f"SELECT id FROM clients WHERE nom = '{nom}'"
    if contrat is not None:
        sql += f" AND contrat = '{contrat}'"
    return [ligne[0] for ligne in connexion.execute(sql + " ORDER BY id")]


def _sql_sans_virgule(connexion, nom, contrat=None):
    return [ligne[0] for ligne in connexion.execute("SELECT id FROM clients WHERE nom = ? ORDER BY id", (nom))]


def _sql_param(ordre=True, like=False, contrat_toujours=False, contrat_colle=False, tuples=False, retour=True):
    def clients_par_nom(connexion, nom, contrat=None):
        sql = f"SELECT id FROM clients WHERE nom {'LIKE' if like else '='} ?"
        valeurs = [nom]
        if contrat_colle and contrat is not None:
            sql += f" AND contrat = '{contrat}'"
        elif contrat_toujours or contrat is not None:
            sql += " AND contrat = ?"
            valeurs.append(contrat)
        lignes = connexion.execute(sql + (" ORDER BY id" if ordre else ""), valeurs).fetchall()
        if not retour:
            return None
        return lignes if tuples else [ligne[0] for ligne in lignes]
    return clients_par_nom


@pytest.mark.parametrize("mutant,mots", [
    (_sql_fstring, ("toutes lettres", "TOUTE la table", "?")),
    (_sql_echappe, ("toutes lettres", "doublez")),
    (_sql_sans_virgule, ("(nom,)", "virgule")),
    (_sql_param(ordre=False), ("ORDER BY id",)),
    (_sql_param(like=True), ("LIKE", "EXACT")),
    (_sql_param(contrat_toujours=True), ("contrat = NULL",)),
    (_sql_param(contrat_colle=True), ("contrat est collé",)),
    (_sql_param(tuples=True), ("ligne[0]",)),
    (_sql_param(retour=False), ("return",)),
], ids=["fstring", "echappement_maison", "tuple_sans_virgule", "sans_tri", "like", "contrat_null", "contrat_colle",
        "tuples", "sans_return"])
def test_serie_13_injection_sql_erreurs_classiques(mutant, mots):
    _echec_13("clients_par_nom", mutant, *mots)


def test_serie_13_injection_sql_autres_ecritures_justes_acceptees():
    """pandas, un curseur, `with connexion`, row_factory, paramètres nommés : tout ce qui est juste passe."""

    def avec_pandas(connexion, nom, contrat=None):
        sql, valeurs = "SELECT id FROM clients WHERE nom = :nom", {"nom": nom}
        if contrat is not None:
            sql, valeurs = sql + " AND contrat = :contrat", {**valeurs, "contrat": contrat}
        return pd.read_sql_query(sql + " ORDER BY id", connexion, params=valeurs)["id"].tolist()

    def avec_curseur(connexion, nom, contrat=None):
        connexion.row_factory = sqlite3.Row
        with connexion:
            curseur = connexion.cursor()
            curseur.execute("SELECT id FROM clients WHERE nom = ? AND (? IS NULL OR contrat = ?) ORDER BY id",
                            (nom, contrat, contrat))
            return [ligne["id"] for ligne in curseur.fetchall()]

    for juste in (avec_pandas, avec_curseur):
        resultat = _evaluer("clients_par_nom", juste)
        assert resultat.statut == REUSSI, resultat.message


# Défi 81 : secrets dans les journaux


def _masquer(sur_place=False, minuscules=True, tirets=True, contient=True, recursion=True, listes=True,
             dict_sensible_masque=True, types=True, supprimer=False, mots=_MOTS_13):
    def sensible(cle):
        nom = str(cle)
        nom = nom.lower() if minuscules else nom
        nom = nom.replace("-", "_") if tirets else nom
        return any(m in nom for m in mots) if contient else nom in mots

    def valeur(v):
        if recursion and isinstance(v, dict):
            return masquer_secrets(v)
        if listes and isinstance(v, list):
            return [valeur(x) for x in v]
        return v if types else str(v)

    def masquer_secrets(donnees):
        if sur_place:
            for cle in list(donnees):
                if sensible(cle):
                    donnees[cle] = "***"
            return donnees
        resultat = {}
        for cle, v in donnees.items():
            if not dict_sensible_masque and isinstance(v, dict):
                resultat[cle] = masquer_secrets(v)
            elif sensible(cle):
                if not supprimer:
                    resultat[cle] = "***"
            else:
                resultat[cle] = valeur(v)
        return resultat
    return masquer_secrets


@pytest.mark.parametrize("mutant,mots", [
    (_masquer(sur_place=True), ("MODIFIÉ", "NOUVEAU dictionnaire")),
    (_masquer(minuscules=False), ("Authorization", "lower()")),
    (_masquer(contient=False), ("access_token", "CONTIENT")),
    (_masquer(tirets=False), ("X-API-Key", "replace")),
    (_masquer(recursion=False), ("headers", "récursion")),
    (_masquer(listes=False), ("corps.clients[0].token", "LISTE")),
    (_masquer(dict_sensible_masque=False), ("corps.secret", "ENTIÈRE")),
    (_masquer(types=False), ("TYPE", "0.8")),
    (_masquer(supprimer=True), ("disparu",)),
    (_masquer(mots=_MOTS_13 + ("a",)), ("contrat", "pas des secrets")),
], ids=["sur_place", "casse", "egalite", "tirets", "sans_recursion", "listes", "dict_sensible", "types", "supprimer",
        "trop"])
def test_serie_13_masquer_secrets_erreurs_classiques(mutant, mots):
    _echec_13("masquer_secrets", mutant, *mots)


def test_serie_13_masquer_secrets_copie_superficielle_detectee():
    def copie_puis_modifie(donnees):
        import copy as _copy
        resultat = _copy.copy(donnees)
        for cle, v in resultat.items():
            if isinstance(v, dict):
                for k in v:
                    if any(m in k.lower().replace("-", "_") for m in _MOTS_13):
                        v[k] = "***"
            elif any(m in cle.lower().replace("-", "_") for m in _MOTS_13):
                resultat[cle] = "***"
        return resultat

    _echec_13("masquer_secrets", copie_puis_modifie, "MODIFIÉ", "imbriqués")


# Défi 82 : traversée de répertoire


def _export_13(resoudre=True, resoudre_dossier=True, mode="relative_to", refuser_dossier=True, refus_points=False,
            composants=False, erreur=ValueError, abspath=False, none_si_refus=False):
    def chemin_export(dossier, nom_fichier):
        if refus_points and ".." in nom_fichier:
            raise erreur("..")
        if composants and ".." in Path(nom_fichier).parts:
            raise erreur("..")
        racine = Path(dossier).resolve() if resoudre_dossier else Path(dossier)
        if abspath:
            chemin = Path(os.path.abspath(os.path.join(racine, nom_fichier)))
        else:
            chemin = (racine / nom_fichier).resolve() if resoudre else racine / nom_fichier
        refus = refuser_dossier and chemin == racine
        if mode == "relative_to":
            refus = refus or not chemin.is_relative_to(racine)
        elif mode == "startswith":
            refus = refus or not str(chemin).startswith(str(racine))
        if refus:
            if none_si_refus:
                return None
            raise erreur(nom_fichier)
        return chemin
    return chemin_export


@pytest.mark.parametrize("mutant,mots", [
    (_export_13(resoudre=False, mode=None), ("ACCEPTÉ", "« .. » remonte")),
    (_export_13(refus_points=True), ("rapport..v2.csv", "noms honnêtes")),
    (_export_13(resoudre=False, mode=None, composants=True), ("absolu REMPLACE",)),
    (_export_13(mode="startswith"), ("exports_pirate", "is_relative_to")),
    (_export_13(refuser_dossier=False), ("LUI-MÊME", "chemin == racine")),
    (_export_13(abspath=True), ("lien symbolique", "resolve()")),
    (_export_13(resoudre_dossier=False), ("RELATIF", "AUSSI le dossier")),
    (_export_13(erreur=PermissionError), ("PermissionError", "ValueError")),
    (_export_13(none_si_refus=True), ("None", "ValueError")),
], ids=["sans_controle", "points_dans_le_texte", "chemin_absolu", "prefixe_texte", "le_dossier_lui_meme", "lien",
        "dossier_relatif", "mauvaise_exception", "none"])
def test_serie_13_chemin_export_erreurs_classiques(mutant, mots):
    _echec_13("chemin_export", mutant, *mots)


# Défi 83 : injection de formules


def _formules(debuts=_DEBUTS_13, toutes_colonnes=False, sans_isinstance=False, sur_place=False, contient=False,
              retirer=False, texte_manquant=False, deja=False, reset_index=False):
    def cellule(v):
        if sans_isinstance:
            texte = str(v)
        elif isinstance(v, str):
            texte = v
        else:
            return v
        if texte_manquant and texte in ("None", "nan"):
            return texte
        if sans_isinstance and v is None:
            return v
        dangereux = any(d in texte for d in debuts) if contient else texte.startswith(debuts + (("'",) if deja else ()))
        if not dangereux:
            return v
        return texte.lstrip("".join(_DEBUTS_13)) if retirer else "'" + texte

    def neutraliser_formules(df):
        resultat = df if sur_place else df.copy()
        for colonne in resultat.columns:
            texte = resultat[colonne].dtype == object or pd.api.types.is_string_dtype(resultat[colonne].dtype)
            if toutes_colonnes or texte:
                serie = resultat[colonne].astype(object) if texte_manquant else resultat[colonne]
                if texte_manquant:
                    serie = serie.map(lambda v: str(v) if v is None or (isinstance(v, float) and v != v) else v)
                resultat[colonne] = serie.map(cellule)
        return resultat.reset_index(drop=True) if reset_index else resultat
    return neutraliser_formules


@pytest.mark.parametrize("mutant,mots", [
    (_formules(debuts=("=",)), ("'+'", "DEBUTS_DANGEREUX")),
    (_formules(debuts=("=", "+", "-", "@")), ("'\\t'", "'\\r'")),
    (_formules(toutes_colonnes=True, sans_isinstance=True), ("colonne numérique", "-12.5")),
    (_formules(sans_isinstance=True), ("-7", "isinstance(valeur, str)")),
    (_formules(sur_place=True), ("REÇU", "df.copy()")),
    (_formules(contient=True), ("remise=10%", "PREMIER caractère")),
    (_formules(retirer=True), ("DEVANT", "-Dupont")),
    (_formules(texte_manquant=True), ("manquante",)),
    (_formules(deja=True), ("déjà du texte",)),
    (_formules(reset_index=True), ("même index",)),
], ids=["egal_seulement", "sans_tab_cr", "nombres_convertis", "nombres_dans_le_texte", "sur_place", "contient",
        "retirer", "manquants", "double_apostrophe", "index"])
def test_serie_13_neutraliser_formules_erreurs_classiques(mutant, mots):
    _echec_13("neutraliser_formules", mutant, *mots)


# Défi 84 : désérialisation


def _modele(verifier=True, avant=True, deux_lectures=False, nettoyer=True, hacher_nom=False, exception=ValueError,
            none_si_faux=False, avec_joblib=False):
    def charger_modele_verifie(chemin, empreinte_attendue):
        octets = Path(chemin).read_bytes()
        objet = None if avant else pickle.loads(octets)
        attendu = empreinte_attendue.strip().lower() if nettoyer else empreinte_attendue
        empreinte = hashlib.sha256(str(chemin).encode() if hacher_nom else octets).hexdigest()
        if verifier and empreinte != attendu:
            if none_si_faux:
                return None
            raise exception("empreinte inattendue")
        if not avant:
            return objet
        if avec_joblib:
            import joblib
            return joblib.load(chemin)
        if deux_lectures:
            with open(chemin, "rb") as fichier:
                return pickle.load(fichier)
        return pickle.loads(octets)
    return charger_modele_verifie


@pytest.mark.parametrize("mutant,mots", [
    (_modele(avant=False), ("EXÉCUTÉ DU CODE", "AVANT")),
    (_modele(verifier=False), ("EXÉCUTÉ DU CODE",)),
    (_modele(deux_lectures=True), ("2 fois", "UNE seule fois")),
    (_modele(avec_joblib=True), ("2 fois", "pickle.loads(octets)")),
    (_modele(nettoyer=False), ("MAJUSCULES", "strip().lower()")),
    (_modele(hacher_nom=True), ("INTACT", "OCTETS")),
    (_modele(exception=RuntimeError), ("RuntimeError", "ValueError")),
    (_modele(none_si_faux=True), ("None", "exception")),
], ids=["charge_avant", "sans_verification", "deux_lectures", "joblib_relit", "empreinte_brute", "hache_le_nom",
        "runtime_error", "none"])
def test_serie_13_charger_modele_erreurs_classiques(mutant, mots):
    import defis.checks_serie_13 as checks

    checks._PIEGES.clear()
    _echec_13("charger_modele_verifie", mutant, *mots)


def test_serie_13_le_piege_est_inoffensif_et_l_open_est_restaure():
    import builtins
    import io

    import defis.checks_serie_13 as checks

    ouvrir_avant, io_avant = builtins.open, io.open
    _evaluer("charger_modele_verifie", _modele(deux_lectures=True))
    assert builtins.open is ouvrir_avant and io.open is io_avant
    checks._PIEGES.clear()
    assert pickle.loads(pickle.dumps(checks._ObjetPiege())) == {"modele": "piégé"}
    assert checks._PIEGES == ["le fichier piégé a exécuté du code pendant son chargement"]
    checks._PIEGES.clear()


# Défi 85 : lien signé


def _signataire(sep=".", signer_exp=True, compare="octets", construction="hmac", verifier_sig=True, strict=True,
                cle_aleatoire=False, controle_longueur=True, url="hex", inferieur_strict=True, secret_utilise=True):
    import base64
    import secrets as _secrets

    class Signataire:
        def __init__(self, secret):
            if controle_longueur and len(secret) < 16:
                raise ValueError("court")
            self._secret = _secrets.token_bytes(32) if cle_aleatoire else secret

        def _sig(self, message):
            if not secret_utilise:
                brut = hashlib.sha256(message.encode()).digest()
            elif construction == "hmac":
                brut = hmac.new(self._secret, message.encode(), hashlib.sha256).digest()
            else:
                brut = hashlib.sha256(self._secret + message.encode()).digest()
            return base64.b64encode(brut).decode() if url == "b64" else brut.hex()

        def _message(self, client_id, expiration):
            return f"{client_id}{sep}{expiration}" if signer_exp else client_id

        def signer(self, client_id, expiration):
            return f"{client_id}.{expiration}.{self._sig(self._message(client_id, expiration))}"

        def verifier(self, jeton, maintenant):
            morceaux = jeton.split(".")
            if strict:
                client_id, expiration, signature = morceaux
            else:
                client_id, expiration, signature = morceaux[0], morceaux[1], morceaux[2]
            attendue = self._sig(self._message(client_id, expiration))
            if verifier_sig:
                a, b = (signature, attendue) if compare == "texte" else (signature.encode(), attendue.encode())
                if not hmac.compare_digest(a, b):
                    raise ValueError("signature")
            if (int(expiration) <= maintenant) if inferieur_strict else (int(expiration) < maintenant):
                raise ValueError("expiré")
            return client_id

    return Signataire


_BASE_EGAL = _signataire()


class _SignataireEgal(_BASE_EGAL):
    """Tout est juste, sauf la comparaison finale, faite avec == (le correcteur lit le code, pas ce texte)."""

    def verifier(self, jeton, maintenant):
        client_id, expiration, signature = jeton.split(".")
        if signature != self._sig(f"{client_id}.{expiration}"):  # et non compare_digest
            raise ValueError("signature")
        if int(expiration) <= maintenant:
            raise ValueError("expiré")
        return client_id


@pytest.mark.parametrize("mutant,mots", [
    (_signataire(signer_exp=False), ("APRÈS la date", "prolongé")),
    (_signataire(sep=""), ("SÉPARÉS", "C0011")),
    (_signataire(construction="maison"), ("hash(secret + message)", "HMAC")),
    (_SignataireEgal, ("compare_digest", "temps")),
    (_signataire(verifier_sig=False), ("AUTRE secret",)),
    (_signataire(strict=False), ("IndexError", "ValueError")),
    (_signataire(compare="texte"), ("non ASCII", "octets")),
    (_signataire(cle_aleatoire=True), ("NOUVEL objet", "redémarrage")),
    (_signataire(controle_longueur=False), ("court", "16")),
    (_signataire(url="b64"), ("adresse web", "urlsafe")),
    (_signataire(inferieur_strict=False), ("seconde même",)),
    (_signataire(secret_utilise=False), ("ne dépend pas du secret",)),
], ids=["expiration_non_signee", "sans_separateur", "construction_maison", "egalite", "signature_ignoree",
        "index_error", "compare_digest_texte", "cle_aleatoire", "cle_courte", "base64_standard", "borne",
        "sans_secret"])
def test_serie_13_signataire_erreurs_classiques(mutant, mots):
    _echec_13("Signataire", mutant, *mots)


def test_serie_13_compare_digest_dans_une_docstring_ne_compte_pas():
    """Le correcteur lit le CODE : un commentaire ou une docstring qui cite compare_digest ne suffit pas."""
    import defis.checks_serie_13 as checks

    assert not checks._utilise_compare_digest(_SignataireEgal)
    assert checks._utilise_compare_digest(_signataire())


def test_serie_13_les_tests_du_corrige_passent():
    import defis.corrige_serie_13 as corrige

    tests = [f for nom, f in vars(corrige).items() if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_13_les_outils_du_fichier_eleve_sont_ceux_du_correcteur():
    import defis.checks_serie_13 as checks
    import defis.corrige_serie_13 as corrige
    import defis.serie_13 as eleve

    for module in (eleve, corrige):
        assert module.MOTS_SENSIBLES == checks.MOTS_SENSIBLES
        assert module.MASQUE == checks.MASQUE
        assert module.DEBUTS_DANGEREUX == checks.DEBUTS_DANGEREUX
        assert module.base_exemple().execute("SELECT * FROM clients").fetchall() == \
            checks._base().execute("SELECT * FROM clients").fetchall()
    exports = eleve.bac_a_sable()
    assert (exports / "equipe").is_dir() and (exports.parent / "exports_pirate").is_dir()
    chemin, empreinte = eleve.modele_exemple(exports)
    assert corrige.charger_modele_verifie(chemin, empreinte)["modele"] == "logistique"
    assert corrige.neutraliser_formules(eleve.exemple_export())["nom"].tolist()[2:] == ["'+33 6 12 34 56 78", "'-Dupont"]


def test_serie_13_autres_ecritures_justes_acceptees():
    """Le correcteur juge le comportement : des solutions justes écrites autrement doivent passer."""
    import base64
    import copy as _copy
    import io as _io
    import json
    import re as _re

    def masquer_par_copie_profonde(donnees):
        resultat = _copy.deepcopy(donnees)
        a_visiter = [resultat]
        while a_visiter:
            objet = a_visiter.pop()
            elements = objet.items() if isinstance(objet, dict) else enumerate(objet)
            for cle, valeur in list(elements):
                if isinstance(objet, dict) and any(m in str(cle).lower().replace("-", "_") for m in _MOTS_13):
                    objet[cle] = "***"
                elif isinstance(valeur, (dict, list)):
                    a_visiter.append(valeur)
        return resultat

    def export_avec_os_path(dossier, nom_fichier):
        racine = os.path.realpath(dossier)
        chemin = os.path.realpath(os.path.join(racine, nom_fichier))
        if chemin == racine or os.path.commonpath([racine, chemin]) != racine:
            raise ValueError(nom_fichier)
        return chemin  # un str : accepté aussi

    def formules_avec_dataframe_map(df):
        def cellule(v):
            return "'" + v if isinstance(v, str) and v[:1] in _DEBUTS_13 and v else v
        texte = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        resultat = df.copy()
        resultat[texte] = resultat[texte].map(cellule)
        return resultat

    def modele_avec_file_digest(chemin, empreinte_attendue):
        with open(chemin, "rb") as fichier:
            empreinte = hashlib.file_digest(fichier, "sha256").hexdigest()
            if not hmac.compare_digest(empreinte, empreinte_attendue.strip().lower()):
                raise ValueError("empreinte")
            fichier.seek(0)
            return pickle.load(fichier)

    def modele_avec_joblib(chemin, empreinte_attendue):
        import joblib
        octets = Path(chemin).read_bytes()
        if hashlib.sha256(octets).hexdigest() != empreinte_attendue.strip().lower():
            raise ValueError("empreinte")
        return joblib.load(_io.BytesIO(octets))

    class SignataireJson:
        """Charge utile JSON en base64 « URL », signature HMAC sur les octets décodés (malléable, mais sûre)."""

        def __init__(self, secret):
            if len(secret) < 16:
                raise ValueError("court")
            self._secret = secret

        def signer(self, client_id, expiration):
            charge = base64.urlsafe_b64encode(json.dumps([client_id, expiration]).encode()).decode().rstrip("=")
            return f"{charge}.{self._signature(charge.encode() + b'===')}"

        def _signature(self, charge_b64):
            octets = base64.urlsafe_b64decode(charge_b64)
            return base64.urlsafe_b64encode(hmac.new(self._secret, octets, hashlib.sha256).digest()).decode().rstrip("=")

        def verifier(self, jeton, maintenant):
            if not _re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{43}", jeton):
                raise ValueError("forme")
            charge, signature = jeton.split(".")
            if not hmac.compare_digest(signature.encode(), self._signature(charge.encode() + b"===").encode()):
                raise ValueError("signature")
            client_id, expiration = json.loads(base64.urlsafe_b64decode(charge + "==="))
            if expiration <= maintenant:
                raise ValueError("expiré")
            return client_id

    for nom, juste in [("masquer_secrets", masquer_par_copie_profonde), ("chemin_export", export_avec_os_path),
                       ("neutraliser_formules", formules_avec_dataframe_map),
                       ("charger_modele_verifie", modele_avec_file_digest),
                       ("charger_modele_verifie", modele_avec_joblib), ("Signataire", SignataireJson)]:
        resultat = _evaluer(nom, juste)
        assert resultat.statut == REUSSI, (nom, juste.__name__, resultat.message)


# --------------------------------------------------------------------------- Série 14 : tenir en production

import contextlib  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

from defis.production import (  # noqa: E402
    FAUX,
    PARIS,
    TRANSITOIRES,
    VRAIS,
    CircuitOuvert,
    Config,
    ErreurTemporaire,
)


def _echec_14(nom, fonction, *mots):
    resultat = _evaluer(nom, fonction)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


# Défi 86 : configuration


def _config(environ=False, bool_naif=False, premiere=False, nan_ok=False, int_float=False, vide_absent=True,
            modifie=False, fuite=False, minuscules=True, strip=True, min_lot=1, crochets=False):
    def lire_config(env):
        source = dict(os.environ) | dict(env) if environ else env
        if modifie:
            env.setdefault("CHURN_SEUIL", "0.5")
        erreurs = []

        def valeur(nom):
            texte = source[nom] if crochets else source.get(nom)
            if texte is None:
                return None
            texte = texte.strip() if strip else texte
            return texte if (texte.strip() or not vide_absent) else None

        def erreur(message):
            erreurs.append(message)
            if premiere:
                raise ValueError(message)

        cle = valeur("CHURN_CLE_API")
        if cle is None:
            erreur("CHURN_CLE_API : obligatoire")
        seuil, texte = 0.5, valeur("CHURN_SEUIL")
        if texte is not None:
            try:
                seuil = float(texte)
            except ValueError:
                seuil = -1.0
            if (seuil < 0 or seuil > 1) if nan_ok else not 0 <= seuil <= 1:
                erreur(f"CHURN_SEUIL invalide : {texte!r}")
        taille, texte = 500, valeur("CHURN_TAILLE_LOT")
        if texte is not None:
            try:
                taille = int(float(texte)) if int_float else int(texte)
            except ValueError:
                taille = -1
            if taille < min_lot:
                erreur(f"CHURN_TAILLE_LOT invalide : {texte!r}")
        mode, texte = False, valeur("CHURN_MODE_TEST")
        if texte is not None:
            texte = texte.lower() if minuscules else texte
            if bool_naif:
                mode = bool(texte)
            elif texte in VRAIS:
                mode = True
            elif texte not in FAUX:
                erreur(f"CHURN_MODE_TEST invalide : {texte!r}")
        if erreurs:
            raise ValueError(" ; ".join(erreurs) + (f" (env = {dict(env)})" if fuite else ""))
        return Config(cle_api=cle, seuil=seuil, taille_lot=taille, mode_test=mode)

    return lire_config


@pytest.mark.parametrize("mutant,mots", [
    (_config(environ=True), ("os.environ",)),
    (_config(bool_naif=True), ("bool(",)),
    (_config(premiere=True), ("TOUTES",)),
    (_config(nan_ok=True), ("NaN",)),
    (_config(int_float=True), ("int(float",)),
    (_config(vide_absent=False), ("ABSENTE",)),
    (_config(modifie=True), ("MODIFIE",)),
    (_config(fuite=True), ("clé d'API en clair",)),
    (_config(minuscules=False), (".lower()",)),
    (_config(strip=False), (".strip()",)),
    (_config(min_lot=0), ("REFUSÉE", "au moins 1")),
    (_config(crochets=True), ("env.get",)),
])
def test_serie_14_config_erreurs_classiques(mutant, mots):
    _echec_14("lire_config", mutant, *mots)


def test_serie_14_config_os_environ_est_restaure():
    import defis.checks_serie_14 as checks

    avant = dict(os.environ)
    _evaluer("lire_config", _config(environ=True))
    _evaluer("lire_config", _config(crochets=True))
    assert dict(os.environ) == avant
    assert not set(checks._PIEGES_ENVIRON) & (set(os.environ) - set(avant))


# Défi 87 : journalisation


def _lot(imprimer=False, basic=False, gestionnaire=False, niveau=False, racine=False, try_except=True, zero=False,
         score_avant=False, niveau_ignore="warning", traceback=True, info_par_client=False, resume=True, ligne=False,
         except_nu=False, retour=True):
    def scorer_lot(clients, score, journal):
        if basic:
            logging.basicConfig(level=logging.INFO)
        if gestionnaire:
            journal.addHandler(logging.NullHandler())
        if niveau:
            journal.setLevel(logging.INFO)
        cible = logging if racine else journal
        scores, ignores, erreurs = {}, 0, 0
        for client in clients:
            identifiant, facture = client["id"], client.get("facture_mensuelle")
            if score_avant:
                with contextlib.suppress(Exception):
                    score(client)
            invalide = (not facture and facture != 0) if not zero else not facture
            if invalide or (facture is not None and facture < 0):
                if imprimer:
                    print("ignoré", identifiant)
                quoi = client if ligne else identifiant
                getattr(cible, niveau_ignore)("client %s ignoré", quoi)
                ignores += 1
                continue
            if not try_except:
                scores[identifiant] = score(client)
            else:
                try:
                    scores[identifiant] = score(client)
                except (BaseException if except_nu else Exception):
                    if traceback:
                        cible.exception("client %s : échec", identifiant)
                    else:
                        cible.error("client %s : échec", identifiant)
                    erreurs += 1
                    continue
            if info_par_client:
                cible.info("client %s noté", identifiant)
        if resume:
            cible.info("lot terminé : %d notés, %d ignorés, %d en erreur", len(scores), ignores, erreurs)
        return scores if retour else None

    return scorer_lot


@pytest.mark.parametrize("mutant,mots", [
    (_lot(imprimer=True), ("print",)),
    (_lot(basic=True), ("basicConfig",)),
    (_lot(gestionnaire=True), ("addHandler",)),
    (_lot(niveau=True), ("setLevel",)),
    (_lot(racine=True), ("journal racine",)),
    (_lot(try_except=False), ("try:",)),
    (_lot(zero=True), ("A103", "if not facture")),
    (_lot(score_avant=True), ("AVANT",)),
    (_lot(niveau_ignore="error"), ("WARNING",)),
    (_lot(niveau_ignore="info"), ("WARNING",)),
    (_lot(traceback=False), ("traceback",)),
    (_lot(info_par_client=True), ("journal.debug",)),
    (_lot(resume=False), ("résumé",)),
    (_lot(ligne=True), ("e-mail",)),
    (_lot(except_nu=True), ("Ctrl+C",)),
    (_lot(retour=False), ("return",)),
])
def test_serie_14_journal_erreurs_classiques(mutant, mots):
    _echec_14("scorer_lot", mutant, *mots)


def test_serie_14_le_journal_est_remis_en_etat():
    racine = logging.getLogger()
    avant = (list(racine.handlers), racine.level, logging.basicConfig)
    for mutant in (_lot(basic=True), _lot(gestionnaire=True), _lot(niveau=True), _lot(racine=True)):
        _evaluer("scorer_lot", mutant)
    assert (list(racine.handlers), racine.level, logging.basicConfig) == avant


# Défi 88 : reprises


def _reprises(reessayer=True, premier_exposant=0, doubler=True, total=True, relancer="derniere", attendre_fin=False,
              plafonner=True, attraper=TRANSITOIRES, attente_si="none", getattr_=True, vrai_sleep=False,
              resultat_faux=False, verifier_tentatives=True):
    def appeler_avec_reprises(appel, *, tentatives=4, base=0.5, plafond=8.0, dormir=time.sleep):
        if verifier_tentatives and tentatives < 1:
            raise ValueError("tentatives")
        premiere = None
        nombre = tentatives if total else tentatives + 1
        for numero in range(nombre):
            try:
                resultat = appel()
                if resultat_faux and not resultat:
                    raise ErreurTemporaire("réponse vide")
                return resultat
            except attraper as erreur:
                premiere = premiere or erreur
                derniere = numero == nombre - 1
                if not reessayer or (derniere and not attendre_fin):
                    if relancer == "derniere":
                        raise
                    if relancer == "premiere":
                        raise premiere from None
                    if relancer == "nouvelle":
                        raise RuntimeError("échec après plusieurs tentatives") from erreur
                    return None
                attente = erreur.attente if not getattr_ else getattr(erreur, "attente", None)
                connu = attente if attente_si == "none" else (attente or None)
                exposant = numero + premier_exposant if doubler else 0
                delai = connu if connu is not None else base * 2 ** exposant
                (time.sleep if vrai_sleep else dormir)(min(plafond, delai) if plafonner else delai)
                if derniere:
                    if relancer == "derniere":
                        raise
                    return None

    return appeler_avec_reprises


@pytest.mark.parametrize("mutant,mots", [
    (_reprises(reessayer=False), ("abandonne",)),
    (_reprises(premier_exposant=1), ("base * 2**0",)),
    (_reprises(doubler=False), ("DOUBLER",)),
    (_reprises(total=False), ("TOTAL",)),
    (_reprises(relancer=None), ("croira",)),
    (_reprises(relancer="nouvelle"), ("nouvelle erreur", "raise")),
    (_reprises(relancer="premiere"), ("PREMIÈRE",)),
    (_reprises(attendre_fin=True), ("DERNIER échec",)),
    (_reprises(plafonner=False), ("plafond",)),
    (_reprises(attraper=Exception), ("TRANSITOIRES",)),
    (_reprises(attraper=ErreurTemporaire), ("TRANSITOIRES",)),
    (_reprises(attente_si="vrai"), ("is not None",)),
    (_reprises(getattr_=False), ("getattr",)),
    (_reprises(vrai_sleep=True), ("dormir",)),
    (_reprises(resultat_faux=True), ("EXCEPTION",)),
    (_reprises(verifier_tentatives=False), ("tentatives=0",)),
])
def test_serie_14_reprises_erreurs_classiques(mutant, mots):
    _echec_14("appeler_avec_reprises", mutant, *mots)


def test_serie_14_le_vrai_sleep_est_restaure():
    sleep = time.sleep
    _evaluer("appeler_avec_reprises", _reprises(vrai_sleep=True))
    assert time.sleep is sleep


# Défi 89 : fuseaux horaires


def _envoi(naive_ok=False, convertir="astimezone", decalage_fixe=False, strict=True, sortie="utc",
           garder_minutes=False, week_end=True):
    """Le raisonnement « à l'horloge murale » : l'heure qu'il est, puis aujourd'hui ou demain."""
    def prochain_envoi(maintenant):
        if not naive_ok and maintenant.tzinfo is None:
            raise ValueError("naïve")
        if convertir == "astimezone":
            local = maintenant.astimezone(PARIS)
        elif convertir == "replace":
            local = maintenant.replace(tzinfo=PARIS)
        else:
            local = maintenant
        heure = (local.hour, local.minute, local.second, local.microsecond)
        jour = local.date()
        if heure >= (10, 0, 0, 0) if strict else heure > (10, 0, 0, 0):
            jour += timedelta(days=1)
        while week_end and jour.weekday() >= 5:
            jour += timedelta(days=1)
        if decalage_fixe:  # calcule en UTC, avec le décalage d'AUJOURD'HUI
            candidat = (datetime(jour.year, jour.month, jour.day, 10) - local.utcoffset()).replace(tzinfo=timezone.utc)
        elif garder_minutes:
            candidat = local.replace(year=jour.year, month=jour.month, day=jour.day, hour=10)
        else:
            candidat = datetime(jour.year, jour.month, jour.day, 10, tzinfo=PARIS)
        if sortie == "paris":
            return candidat.astimezone(PARIS)
        if sortie == "naive":
            return candidat.astimezone(timezone.utc).replace(tzinfo=None)
        return candidat.astimezone(timezone.utc)

    return prochain_envoi


@pytest.mark.parametrize("mutant,mots", [
    (_envoi(naive_ok=True), ("NAÏVE",)),
    (_envoi(convertir="replace"), ("astimezone(PARIS)",)),
    (_envoi(convertir="aucune"), ("astimezone(PARIS)",)),
    (_envoi(decalage_fixe=True), ("CHANGEMENT D'HEURE",)),
    (_envoi(strict=False), ("10 h pile",)),
    (_envoi(sortie="paris"), ("UTC",)),
    (_envoi(sortie="naive"), ("NAÏVE",)),
    (_envoi(garder_minutes=True), ("pile",)),
    (_envoi(week_end=False), ("week-end",)),
])
def test_serie_14_fuseaux_erreurs_classiques(mutant, mots):
    _echec_14("prochain_envoi", mutant, *mots)


# Défi 90 : écriture atomique


def _atomique(direct=False, ailleurs=False, nom_fixe=False, nettoyer=True, relancer="meme", fermer_avant=True,
              tout_en_memoire=False, fin_de_ligne=True, copier=False):
    def ecrire_atomique(chemin, lignes):
        chemin = Path(chemin)
        if tout_en_memoire:
            texte = "".join(f"{ligne}\n" for ligne in lignes)
            chemin.write_text(texte, encoding="utf-8")
            return
        if direct:
            with open(chemin, "w", encoding="utf-8") as fichier:
                for ligne in lignes:
                    fichier.write(f"{ligne}\n")
            return
        if nom_fixe:
            temporaire = chemin.with_name(chemin.name + ".tmp")
            descripteur = os.open(temporaire, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        else:
            descripteur, nom = tempfile.mkstemp(dir=None if ailleurs else chemin.parent)
            temporaire = Path(nom)
        try:
            fichier = os.fdopen(descripteur, "w", encoding="utf-8")
            for ligne in lignes:
                fichier.write(f"{ligne}\n" if fin_de_ligne else ligne)
            if not fermer_avant:
                os.replace(temporaire, chemin)
                fichier.close()
                return
            fichier.close()
            if copier:
                shutil.copyfile(temporaire, chemin)
                temporaire.unlink()
            else:
                os.replace(temporaire, chemin)
        except BaseException:
            with contextlib.suppress(NameError):
                fichier.close()
            if nettoyer:
                temporaire.unlink(missing_ok=True)
            if relancer == "meme":
                raise
            if relancer == "nouvelle":
                raise OSError("écriture impossible") from None

    return ecrire_atomique


@pytest.mark.parametrize("mutant,mots", [
    (_atomique(direct=True), ("TEMPORAIRE",)),
    (_atomique(tout_en_memoire=True), ("TEMPORAIRE",)),
    (_atomique(copier=True), ("TEMPORAIRE",)),
    (_atomique(ailleurs=True), ("mkstemp(dir=",)),
    (_atomique(nom_fixe=True), ("unique",)),
    (_atomique(nettoyer=False), ("supprimez-le",)),
    (_atomique(relancer=None), ("relancez",)),
    (_atomique(relancer="nouvelle"), ("simple `raise`",)),
    (_atomique(fermer_avant=False), ("APRÈS l'avoir fermé",)),
    (_atomique(fin_de_ligne=False), ("\\n",)),
])
def test_serie_14_ecriture_atomique_erreurs_classiques(mutant, mots):
    _echec_14("ecrire_atomique", mutant, *mots)


def test_serie_14_les_espions_d_ecriture_sont_retires():
    import builtins
    import io

    avant = (builtins.open, io.open, os.replace, os.rename)
    for mutant in (_atomique(), _atomique(direct=True), _atomique(nettoyer=False)):
        _evaluer("ecrire_atomique", mutant)
    assert (builtins.open, io.open, os.replace, os.rename) == avant


# Défi 91 : disjoncteur


def _disjoncteur(propriete=True, remise_a_zero=True, decalage=0, circuit_a_l_ouverture=False, appeler_si_ouvert=False,
                 strict=False, vraie_horloge=False, calcul_paresseux=True, essai_unique=True, relancer_delai=True,
                 attraper=TRANSITOIRES, en_dur=False, partage=False, avaler=False, transmettre=True):
    class Disjoncteur:
        _commun = {"echecs": 0, "ouverture": None}

        def __init__(self, seuil_echecs=3, delai=30.0, horloge=time.monotonic):
            self.seuil = 3 if en_dur else seuil_echecs
            self.delai = 30.0 if en_dur else delai
            self.horloge = time.monotonic if vraie_horloge else horloge
            self.s = Disjoncteur._commun if partage else {"echecs": 0, "ouverture": None}
            self._memo = "ferme"

        def _calculer(self):
            if self.s["ouverture"] is None:
                return "ferme"
            ecoule = self.horloge() - self.s["ouverture"]
            return "semi_ouvert" if (ecoule > self.delai if strict else ecoule >= self.delai) else "ouvert"

        def _etat(self):
            if calcul_paresseux:
                return self._calculer()
            return self._memo

        def appeler(self, fonction, *args, **kwargs):
            etat = self._calculer()
            if etat == "ouvert" and not appeler_si_ouvert:
                self._memo = "ouvert"
                raise CircuitOuvert("ouvert")
            try:
                resultat = fonction(*args, **kwargs) if transmettre else fonction()
            except attraper:
                self.s["echecs"] += 1
                if (etat == "semi_ouvert" and essai_unique) or self.s["echecs"] >= self.seuil + decalage:
                    if relancer_delai or self.s["ouverture"] is None:
                        self.s["ouverture"] = self.horloge()
                    if not essai_unique:
                        self.s["echecs"] = 0  # « on repart de zéro »… et il faudra 3 échecs pour rouvrir
                    self._memo = "ouvert"
                    if circuit_a_l_ouverture:
                        raise CircuitOuvert("vient de s'ouvrir") from None
                if avaler:
                    return None
                raise
            if remise_a_zero or etat == "semi_ouvert":
                self.s["echecs"] = 0
            self.s["ouverture"] = None
            self._memo = "ferme"
            return resultat

    if propriete:
        Disjoncteur.etat = property(Disjoncteur._etat)
    else:
        Disjoncteur.etat = Disjoncteur._etat
    return Disjoncteur


@pytest.mark.parametrize("mutant,mots", [
    (_disjoncteur(propriete=False), ("@property",)),
    (_disjoncteur(remise_a_zero=False), ("D'AFFILÉE",)),
    (_disjoncteur(decalage=-1), ("pas avant",)),
    (_disjoncteur(decalage=1), ("seuil_echecs",)),
    (_disjoncteur(circuit_a_l_ouverture=True), ("OUVRE",)),
    (_disjoncteur(appeler_si_ouvert=True), ("appelle quand même",)),
    (_disjoncteur(strict=True), (">= delai",)),
    (_disjoncteur(vraie_horloge=True), ("horloge REÇUE",)),
    (_disjoncteur(calcul_paresseux=False), ("propriété `etat`",)),
    (_disjoncteur(essai_unique=False), ("Un seul échec suffit",)),
    (_disjoncteur(relancer_delai=False), ("repart de zéro",)),
    (_disjoncteur(attraper=Exception), ("TRANSITOIRES",)),
    (_disjoncteur(attraper=ErreurTemporaire), ("TRANSITOIRES",)),
    (_disjoncteur(en_dur=True), ("en dur",)),
    (_disjoncteur(partage=True), ("CLASSE",)),
    (_disjoncteur(avaler=True), ("raise",)),
    (_disjoncteur(transmettre=False), ("*args, **kwargs",)),
])
def test_serie_14_disjoncteur_erreurs_classiques(mutant, mots):
    _echec_14("Disjoncteur", mutant, *mots)


# Le corrigé, le fichier élève, et d'autres solutions justes


@pytest.mark.parametrize("nom,fabrique", [("lire_config", _config), ("scorer_lot", _lot),
                                          ("appeler_avec_reprises", _reprises), ("prochain_envoi", _envoi),
                                          ("ecrire_atomique", _atomique), ("Disjoncteur", _disjoncteur)])
def test_serie_14_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Garde-fou : chaque version boguée ci-dessus ne diffère du juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_serie_14_les_tests_du_corrige_passent():
    import defis.corrige_serie_14 as corrige

    tests = [f for nom, f in vars(corrige).items() if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_14_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.production as production
    import defis.serie_14 as eleve

    for nom in ("Config", "ErreurTemporaire", "CircuitOuvert", "TRANSITOIRES", "HorlogeFactice", "PARIS", "VRAIS", "FAUX",
                "exemple_lot", "score_exemple", "journal_console", "service_capricieux"):
        assert getattr(eleve, nom) is getattr(production, nom), nom
    assert issubclass(CircuitOuvert, ErreurTemporaire)
    assert "cle" not in repr(Config(cle_api="cle-secrete"))


def test_serie_14_autres_ecritures_justes_acceptees():
    """Le correcteur juge le comportement : des solutions justes écrites autrement doivent passer."""
    def config_par_table(env):
        def booleen(texte):
            if texte.lower() not in VRAIS + FAUX:
                raise ValueError(texte)
            return texte.lower() in VRAIS

        def borne(conversion, minimum, maximum=math.inf):
            def convertir(texte):
                valeur = conversion(texte)
                if not minimum <= valeur <= maximum:
                    raise ValueError(texte)
                return valeur
            return convertir

        regles = {"cle_api": ("CHURN_CLE_API", str, None), "seuil": ("CHURN_SEUIL", borne(float, 0, 1), 0.5),
                  "taille_lot": ("CHURN_TAILLE_LOT", borne(int, 1), 500), "mode_test": ("CHURN_MODE_TEST", booleen, False)}
        valeurs, fautes = {}, []
        for champ, (variable, conversion, defaut) in regles.items():
            texte = (env.get(variable) or "").strip()
            if not texte:
                if defaut is None:
                    fautes.append(f"{variable} manquante")
                valeurs[champ] = defaut
                continue
            try:
                valeurs[champ] = conversion(texte)
            except ValueError:
                fautes.append(f"{variable} invalide")
        if fautes:
            raise ValueError(", ".join(fautes))
        return Config(**valeurs)

    def lot_avec_error_exc_info(clients, score, journal):
        notes, comptes = {}, {"ignorés": 0, "en erreur": 0}
        for client in clients:
            facture = client.get("facture_mensuelle")
            if not isinstance(facture, (int, float)) or facture < 0:
                journal.warning(f"Client {client['id']} : facture invalide, client ignoré")
                comptes["ignorés"] += 1
                continue
            try:
                notes[client["id"]] = score(client)
            except Exception:
                journal.error(f"Client {client['id']} : erreur du modèle", exc_info=True)
                comptes["en erreur"] += 1
        journal.info(f"Fin du lot : notés={len(notes)} ignorés={comptes['ignorés']} erreurs={comptes['en erreur']}")
        return notes

    def reprises_en_while(appel, *, tentatives=4, base=0.5, plafond=8.0, dormir=time.sleep):
        if tentatives <= 0:
            raise ValueError("au moins une tentative")
        essai = 0
        while True:
            try:
                return appel()
            except TRANSITOIRES as erreur:
                essai += 1
                if essai >= tentatives:
                    raise
                demande = getattr(erreur, "attente", None)
                dormir(min(plafond, base * 2 ** (essai - 1) if demande is None else demande))

    def envoi_par_liste(maintenant):
        if maintenant.utcoffset() is None:
            raise ValueError("fuseau manquant")
        aujourd_hui = maintenant.astimezone(PARIS).date()
        jours = (aujourd_hui + timedelta(days=n) for n in range(8))
        candidats = (datetime.combine(jour, datetime.min.time().replace(hour=10), tzinfo=PARIS) for jour in jours)
        return min(c for c in candidats if c.isoweekday() <= 5 and c > maintenant).astimezone(timezone.utc)

    def atomique_named_temporary(chemin, lignes):
        chemin = Path(chemin)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=chemin.parent, delete=False,
                                         suffix=".partiel") as fichier:
            temporaire = Path(fichier.name)
        try:
            with temporaire.open("w", encoding="utf-8") as fichier:
                fichier.writelines(f"{ligne}\n" for ligne in lignes)
            temporaire.replace(chemin)
        except BaseException:
            temporaire.unlink(missing_ok=True)
            raise

    class DisjoncteurAEtatStocke:
        def __init__(self, seuil_echecs=3, delai=30.0, horloge=time.monotonic):
            self.seuil_echecs, self.delai, self.horloge = seuil_echecs, delai, horloge
            self._etat, self._echecs, self._depuis = "ferme", 0, 0.0

        @property
        def etat(self):
            if self._etat == "ouvert" and self.horloge() - self._depuis >= self.delai:
                self._etat = "semi_ouvert"
            return self._etat

        def _ouvrir(self):
            self._etat, self._depuis = "ouvert", self.horloge()

        def appeler(self, fonction, *args, **kwargs):
            if self.etat == "ouvert":
                raise CircuitOuvert("patientez")
            essai = self.etat == "semi_ouvert"
            try:
                resultat = fonction(*args, **kwargs)
            except TRANSITOIRES:
                self._echecs += 1
                if essai or self._echecs >= self.seuil_echecs:
                    self._ouvrir()
                raise
            self._etat, self._echecs = "ferme", 0
            return resultat

    for nom, juste in [("lire_config", config_par_table), ("scorer_lot", lot_avec_error_exc_info),
                       ("appeler_avec_reprises", reprises_en_while), ("prochain_envoi", envoi_par_liste),
                       ("ecrire_atomique", atomique_named_temporary), ("Disjoncteur", DisjoncteurAEtatStocke)]:
        resultat = _evaluer(nom, juste)
        assert resultat.statut == REUSSI, (nom, juste.__name__, resultat.message)


# --------------------------------------------------------------------------- Série 15 : enregistrer sans rien perdre

from defis.persistance import BAIL, STATUTS, ConflitDeVersion  # noqa: E402


def _echec_15(nom, fonction, *mots):
    resultat = _evaluer(nom, fonction)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


# Défi 92 : transaction


def _offre(budget_dabord=False, commit_entre=False, sans_rollback=False, sans_commit=False, avale=False,
           strict=False, sans_montant=False, sans_campagne=False, fstring=False, ferme=False, mauvais_retour=False,
           sans_where=False):
    def accorder_offre(conn, client_id, campagne, montant):
        try:
            if not sans_montant and not montant > 0:
                raise ValueError("montant invalide")
            if fstring:
                ligne = conn.execute(f"SELECT reste FROM budget WHERE campagne = '{campagne}'").fetchone()
            else:
                ligne = conn.execute("SELECT reste FROM budget WHERE campagne = ?", (campagne,)).fetchone()
            if ligne is None:
                if not sans_campagne:
                    raise ValueError("campagne inconnue")
            elif (ligne[0] <= montant) if strict else (ligne[0] < montant):
                raise ValueError("budget insuffisant")

            def inserer():
                return conn.execute("INSERT INTO offres (client_id, campagne, montant) VALUES (?, ?, ?)",
                                    (client_id, campagne, montant))

            def debiter():
                if sans_where:
                    conn.execute("UPDATE budget SET reste = reste - ?", (montant,))
                else:
                    conn.execute("UPDATE budget SET reste = reste - ? WHERE campagne = ?", (montant, campagne))

            if budget_dabord:
                debiter()
                if commit_entre:
                    conn.commit()
                curseur = inserer()
            else:
                curseur = inserer()
                if commit_entre:
                    conn.commit()
                debiter()
            if not sans_commit:
                conn.commit()
            return True if mauvais_retour else curseur.lastrowid
        except Exception:
            if not sans_rollback:
                conn.rollback()
            if avale:
                return None
            raise
        finally:
            if ferme:
                conn.close()

    return accorder_offre


@pytest.mark.parametrize("mutant,mots", [
    (_offre(commit_entre=True), ("ENTRE les deux écritures", "printemps")),
    (_offre(commit_entre=True, budget_dabord=True), ("ENTRE les deux écritures",)),
    (_offre(sans_rollback=True), ("transaction OUVERTE", "rollback")),
    (_offre(sans_rollback=True, budget_dabord=True), ("transaction OUVERTE", "la base contiendrait", "budget")),
    (_offre(sans_commit=True), ("ne sont pas validées", "commit")),
    (_offre(avale=True), ("aurait dû lever une erreur", "ne cachez pas")),
    (_offre(strict=True), ("EXACTEMENT", "reste < montant")),
    (_offre(sans_montant=True), ("montant nul", "montant > 0")),
    (_offre(sans_campagne=True), ("n'existe pas", "AUCUNE ligne")),
    (_offre(fstring=True), ("apostrophe", "?")),
    (_offre(ferme=True), ("FERME la connexion",)),
    (_offre(mauvais_retour=True), ("lastrowid",)),
    (_offre(sans_where=True), ("AUTRES campagnes", "WHERE campagne = ?")),
])
def test_serie_15_offre_erreurs_classiques(mutant, mots):
    _echec_15("accorder_offre", mutant, *mots)


# Défi 93 : pagination


def _pages(ge=False, sans_order=False, tout_lire=False, suivant_plein=False, sans_validation=False,
           validation_tardive=False, fstring=False, cles=("id", "nom", "score"), suivant_premier=False):
    def page_clients(conn, apres=None, taille=20):
        if not sans_validation and not validation_tardive and not 1 <= taille <= 100:
            raise ValueError("taille")
        ordre = "" if sans_order else " ORDER BY id"
        limite = "" if tout_lire else " LIMIT ?"
        n = taille if suivant_plein else taille + 1
        params = () if tout_lire else (n,)
        if apres is None:
            lignes = conn.execute(f"SELECT id, nom, score FROM clients{ordre}{limite}", params).fetchall()
        elif fstring:
            lignes = conn.execute(f"SELECT id, nom, score FROM clients WHERE id > '{apres}'{ordre}{limite}", params).fetchall()
        else:
            op = ">=" if ge else ">"
            lignes = conn.execute(f"SELECT id, nom, score FROM clients WHERE id {op} ?{ordre}{limite}", (apres, *params)).fetchall()
        if validation_tardive and not 1 <= taille <= 100:
            raise ValueError("taille")
        page = [dict(zip(cles, ligne)) for ligne in lignes[:taille]]
        if suivant_plein:
            suivant = lignes[-1][0] if len(lignes) == taille else None
        elif suivant_premier:
            suivant = lignes[taille][0] if len(lignes) > taille else None
        else:
            suivant = lignes[taille - 1][0] if len(lignes) > taille else None
        return page, suivant

    return page_clients


@pytest.mark.parametrize("mutant,mots", [
    (_pages(ge=True), ("deux fois", "id > ?")),
    (_pages(sans_order=True), ("ORDER BY id",)),
    (_pages(tout_lire=True), ("toute la table", "LIMIT")),
    (_pages(suivant_plein=True), ("page suivante vide", "LIMIT taille + 1")),
    (_pages(sans_validation=True), ("ValueError", "de 1 à 100")),
    (_pages(validation_tardive=True), ("APRÈS avoir interrogé",)),
    (_pages(fstring=True), ("apostrophe", "?")),
    (_pages(cles=("identifiant", "nom", "score")), ("exactement les clés",)),
    (_pages(suivant_premier=True), ("DERNIER client de la page",)),
])
def test_serie_15_pages_erreurs_classiques(mutant, mots):
    _echec_15("page_clients", mutant, *mots)


# Défi 94 : N+1


def _factures_15(n_plus_1=False, sans_vides=False, deux_lectures=False, doublons=False, sans_tri=False,
                 fstring=False, cles=("mois", "montant")):
    def factures_par_client(conn, ids):
        if deux_lectures:
            marques = ", ".join("?" for _ in ids)
            ids = list(ids)
        else:
            ids = list(ids) if doublons else list(dict.fromkeys(ids))
            marques = ", ".join("?" * len(ids))
        resultat = {} if sans_vides else {i: [] for i in ids}
        ordre = "" if sans_tri else " ORDER BY client_id, mois"
        if n_plus_1:
            lignes = [(i, m, x) for i in ids
                      for m, x in conn.execute(f"SELECT mois, montant FROM factures WHERE client_id = ?{ordre.replace('client_id, ', '')}", (i,))]
        elif fstring:
            liste = ", ".join(f"'{i}'" for i in ids)
            lignes = conn.execute(f"SELECT client_id, mois, montant FROM factures WHERE client_id IN ({liste}){ordre}").fetchall()
        else:
            lignes = conn.execute(f"SELECT client_id, mois, montant FROM factures WHERE client_id IN ({marques}){ordre}", ids).fetchall()
        par_client = {}
        for client_id, mois, montant in lignes:
            par_client.setdefault(client_id, []).append(dict(zip(cles, (mois, montant))))
        if doublons:  # on range en parcourant les id demandés… doublons compris
            for i in ids:
                resultat.setdefault(i, []).extend(par_client.get(i, []))
        else:
            for client_id, factures in par_client.items():
                resultat.setdefault(client_id, []).extend(factures)
        return resultat

    return factures_par_client


@pytest.mark.parametrize("mutant,mots", [
    (_factures_15(n_plus_1=True), ("N+1", "IN (?, ?")),
    (_factures_15(sans_vides=True), ("aucune facture", "{id: [] for id in ids}")),
    (_factures_15(deux_lectures=True), ("GÉNÉRATEUR", "list(dict.fromkeys(ids))")),
    (_factures_15(doublons=True), ("deux fois", "dict.fromkeys")),
    (_factures_15(sans_tri=True), ("triées par mois",)),
    (_factures_15(fstring=True), ("apostrophe", "?")),
    (_factures_15(cles=("mois", "total")), ('{"mois": …, "montant": …}',)),
])
def test_serie_15_factures_erreurs_classiques(mutant, mots):
    _echec_15("factures_par_client", mutant, *mots)


# Défi 95 : migrations


def _migrer_15(executescript=False, sans_begin=False, version_a_la_fin=False, version_apres_commit=False,
               rejoue=False, avale=False, continue_apres=False, sans_futur=False, une_transaction=False,
               compte_total=False, sans_rollback=False, toujours_begin_sans_commit=False):
    def migrer(conn, migrations):
        version = 0 if rejoue else conn.execute("PRAGMA user_version").fetchone()[0]
        if not sans_futur and version > len(migrations):
            raise ValueError("base plus récente que le programme")
        appliquees, erreur = 0, None
        if une_transaction:
            conn.execute("BEGIN")
        for i in range(version, len(migrations)):
            try:
                if executescript:
                    conn.executescript(";\n".join(migrations[i]) + f";\nPRAGMA user_version = {i + 1};")
                    appliquees += 1
                    continue
                if not sans_begin and not une_transaction:
                    conn.execute("BEGIN")
                for instruction in migrations[i]:
                    conn.execute(instruction)
                if version_apres_commit:
                    conn.commit()
                if not version_a_la_fin:
                    conn.execute(f"PRAGMA user_version = {i + 1}")
                if not une_transaction and not toujours_begin_sans_commit:
                    conn.commit()
                appliquees += 1
            except Exception as exc:
                if not sans_rollback:
                    conn.rollback()
                if continue_apres:
                    erreur = erreur or exc
                    continue
                if avale:
                    return appliquees
                if version_a_la_fin and appliquees:
                    conn.execute(f"PRAGMA user_version = {version + appliquees}")
                raise
        if une_transaction:
            conn.commit()
        if version_a_la_fin and appliquees:
            conn.execute(f"PRAGMA user_version = {len(migrations)}")
        if erreur:
            raise erreur
        return len(migrations) if compte_total else appliquees

    return migrer


@pytest.mark.parametrize("mutant,mots", [
    (_migrer_15(executescript=True), ("executescript", "une par une")),
    (_migrer_15(sans_begin=True), ("À MOITIÉ", "BEGIN")),
    (_migrer_15(version_a_la_fin=True), ("MÊME commit",)),
    (_migrer_15(version_apres_commit=True), ("MÊME commit", "PRAGMA")),
    (_migrer_15(rejoue=True), ("DÉJÀ appliquées", "range(version, len(migrations))")),
    (_migrer_15(avale=True), ("sans lever d'erreur", "raise")),
    (_migrer_15(continue_apres=True), ("appliqué m4", "première erreur")),
    (_migrer_15(sans_futur=True), ("ValueError", "plus récent")),
    (_migrer_15(une_transaction=True), ("annulées aussi", "SA transaction")),
    (_migrer_15(compte_total=True), ("NOMBRE",)),
    (_migrer_15(sans_rollback=True), ("transaction OUVERTE",)),
    (_migrer_15(toujours_begin_sans_commit=True), ("transaction",)),
])
def test_serie_15_migrations_erreurs_classiques(mutant, mots):
    _echec_15("migrer", mutant, *mots)


# Défi 96 : verrou optimiste


def _statut_15(lire_puis_ecrire=False, sans_increment=False, conflit_pour_inconnu=False, sans_commit=False,
               fstring=False, sans_validation=False, sans_rowcount=False, keyerror_pour_conflit=False, retour=None):
    def changer_statut(conn, client_id, statut, version_lue):
        if not sans_validation and statut not in STATUTS:
            raise ValueError("statut inconnu")
        increment = "" if sans_increment else ", version = version + 1"
        if lire_puis_ecrire:
            ligne = conn.execute("SELECT version FROM clients WHERE id = ?", (client_id,)).fetchone()
            if ligne is None:
                raise KeyError(client_id)
            if ligne[0] != version_lue:
                raise ConflitDeVersion("fiche périmée")
            conn.execute(f"UPDATE clients SET statut = ?{increment} WHERE id = ?", (statut, client_id))
            conn.commit()
            return version_lue + 1
        if fstring:
            curseur = conn.execute(f"UPDATE clients SET statut = '{statut}'{increment} "
                                   f"WHERE id = '{client_id}' AND version = {version_lue}")
        else:
            curseur = conn.execute(f"UPDATE clients SET statut = ?{increment} WHERE id = ? AND version = ?",
                                   (statut, client_id, version_lue))
        if not sans_commit:
            conn.commit()
        if curseur.rowcount >= 1 or sans_rowcount:
            return retour if retour is not None else version_lue + 1
        if conflit_pour_inconnu:
            raise ConflitDeVersion("conflit")
        if keyerror_pour_conflit:
            raise KeyError(client_id)
        if conn.execute("SELECT 1 FROM clients WHERE id = ?", (client_id,)).fetchone() is None:
            raise KeyError(client_id)
        raise ConflitDeVersion("conflit")

    return changer_statut


@pytest.mark.parametrize("mutant,mots", [
    (_statut_15(lire_puis_ecrire=True), ("une milliseconde trop tôt", "DANS l'UPDATE")),
    (_statut_15(sans_increment=True), ("version = version + 1",)),
    (_statut_15(conflit_pour_inconnu=True), ("KeyError", "n'existe pas")),
    (_statut_15(keyerror_pour_conflit=True), ("ConflitDeVersion",)),
    (_statut_15(sans_commit=True), ("ne sont pas validées",)),
    (_statut_15(fstring=True), ("TOUTES les fiches", "?")),
    (_statut_15(sans_validation=True), ("perdu", "STATUTS")),
    (_statut_15(sans_rowcount=True), ("rowcount",)),
    (_statut_15(retour="2"), ("NOUVELLE version",)),
])
def test_serie_15_statut_erreurs_classiques(mutant, mots):
    _echec_15("changer_statut", mutant, *mots)


# Défi 97 : file de tâches


def _tache_15(sans_recheck=False, abandonne=False, sans_bail=False, bail_strict=False, ordre="priorite DESC, id",
              sans_essais=False, prend_faites=False, en_cours_toujours=False, update_tout=False, sans_commit=False,
              fstring=False, horloge=False, recheck_a_faire=False):
    import time as _time

    def prendre_tache(conn, travailleur, maintenant, bail=BAIL):
        comparaison = "<" if bail_strict else "<="
        if sans_bail:
            dispo = "(statut = 'a_faire' AND (? + ? IS NOT NULL))"
        elif prend_faites:
            dispo = f"(statut IN ('a_faire', 'faite') OR (statut = 'en_cours' AND pris_a + ? {comparaison} ?))"
        elif en_cours_toujours:
            dispo = "(statut IN ('a_faire', 'en_cours') AND (? + ? IS NOT NULL))"
        else:
            dispo = f"(statut = 'a_faire' OR (statut = 'en_cours' AND pris_a + ? {comparaison} ?))"
        recheck = "statut = 'a_faire' AND (? + ? IS NOT NULL)" if recheck_a_faire else dispo
        pris_a = _time.time() if horloge else maintenant
        essais = "essais" if sans_essais else "essais + 1"
        while True:
            ligne = conn.execute(f"SELECT id FROM taches WHERE {dispo} ORDER BY {ordre} LIMIT 1", (bail, maintenant)).fetchone()
            if ligne is None:
                return None
            if update_tout:
                curseur = conn.execute(f"UPDATE taches SET statut = 'en_cours', travailleur = ?, pris_a = ?, essais = {essais} "
                                       "WHERE statut = 'a_faire'", (travailleur, pris_a))
            elif fstring:
                curseur = conn.execute(f"UPDATE taches SET statut = 'en_cours', travailleur = '{travailleur}', pris_a = {pris_a}, "
                                       f"essais = {essais} WHERE id = {ligne[0]}")
            elif sans_recheck:
                curseur = conn.execute(f"UPDATE taches SET statut = 'en_cours', travailleur = ?, pris_a = ?, essais = {essais} "
                                       "WHERE id = ?", (travailleur, pris_a, ligne[0]))
            else:
                curseur = conn.execute(f"UPDATE taches SET statut = 'en_cours', travailleur = ?, pris_a = ?, essais = {essais} "
                                       f"WHERE id = ? AND {recheck}", (travailleur, pris_a, ligne[0], bail, maintenant))
            if not sans_commit:
                conn.commit()
            if curseur.rowcount >= 1:
                client_id, n = conn.execute("SELECT client_id, essais FROM taches WHERE id = ?", (ligne[0],)).fetchone()
                return {"id": ligne[0], "client_id": client_id, "essais": n}
            if abandonne:
                return None

    return prendre_tache


@pytest.mark.parametrize("mutant,mots", [
    (_tache_15(sans_recheck=True), ("deux SMS", "RÉPÉTER")),
    (_tache_15(abandonne=True), ("il reste", "recommencez")),
    (_tache_15(sans_bail=True), ("tombé en panne",)),
    (_tache_15(bail_strict=True), ("PILE", "<=")),
    (_tache_15(ordre="id"), ("priorite DESC, id",)),
    (_tache_15(ordre="priorite DESC, id DESC"), ("priorite DESC, id",)),
    (_tache_15(ordre="priorite, id"), ("priorite DESC, id",)),
    (_tache_15(sans_essais=True), ("essais = essais + 1",)),
    (_tache_15(prend_faites=True), ("jamais reprise",)),
    (_tache_15(en_cours_toujours=True), ("bail court encore",)),
    (_tache_15(update_tout=True), ("AUTRES tâches", "WHERE id = ?")),
    (_tache_15(sans_commit=True), ("ne sont pas validées",)),
    (_tache_15(fstring=True), ("apostrophe", "?")),
    (_tache_15(horloge=True), ("time.time()",)),
    (_tache_15(recheck_a_faire=True), ("boucle", "exactement la même")),
])
def test_serie_15_taches_erreurs_classiques(mutant, mots):
    _echec_15("prendre_tache", mutant, *mots)


@pytest.mark.parametrize("nom,fabrique", [("accorder_offre", _offre), ("page_clients", _pages),
                                          ("factures_par_client", _factures_15), ("migrer", _migrer_15),
                                          ("changer_statut", _statut_15), ("prendre_tache", _tache_15)])
def test_serie_15_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Garde-fou : chaque version boguée ci-dessus ne diffère du juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_serie_15_les_tests_du_corrige_passent():
    import defis.corrige_serie_15 as corrige

    tests = [f for nom, f in vars(corrige).items() if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_15_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.persistance as persistance
    import defis.serie_15 as eleve

    for nom in ("BAIL", "STATUTS", "ConflitDeVersion", "MIGRATIONS_EXEMPLE", "MIGRATION_CASSEE", "afficher",
                "base_exemple", "deux_connexions"):
        assert getattr(eleve, nom) is getattr(persistance, nom), nom


def test_serie_15_les_exemples_des_docstrings_sont_justes(capsys):
    """Les exemples de defis/serie_15.py et de defis/persistance.py, rejoués avec le corrigé."""
    from defis.corrige_serie_15 import (MIGRATION_CASSEE, MIGRATIONS_EXEMPLE, accorder_offre, afficher, base_exemple,
                                        changer_statut, factures_par_client, migrer, page_clients, prendre_tache)

    conn = base_exemple()
    afficher(conn, "SELECT * FROM budget")
    assert capsys.readouterr().out.splitlines() == [
        "campagne     | reste", "-------------+------", "automne      | 500.0", "fidelite     | 120.0",
        "noël d'hiver | 200.0", "(3 lignes)"]
    assert accorder_offre(conn, "C001", "automne", 50) == 2
    with pytest.raises(ValueError, match="120.0 € restants, 500 € demandés"):
        accorder_offre(conn, "C004", "fidelite", 500)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed: offres.client_id"):
        accorder_offre(conn, "C002", "automne", 20)
    page, suivant = page_clients(conn, taille=3)
    assert page[0] == {"id": "C001", "nom": "Léa Martin", "score": 0.82} and suivant == "C003"
    assert [c["id"] for c in page_clients(conn, apres="C006", taille=3)[0]] == ["C007", "C008"]
    assert page_clients(conn, apres="C006", taille=3)[1] is None
    assert factures_par_client(conn, ["C001", "C004"]) == {
        "C001": [{"mois": "2026-07", "montant": 79.9}, {"mois": "2026-08", "montant": 89.9},
                 {"mois": "2026-09", "montant": 89.9}],
        "C004": []}
    assert changer_statut(conn, "C001", "contacte", version_lue=1) == 2
    with pytest.raises(ConflitDeVersion, match="rechargez la fiche"):
        changer_statut(conn, "C001", "parti", version_lue=1)
    pris = [prendre_tache(conn, "poste-4", maintenant=1000.0) for _ in range(5)]
    assert pris == [{"id": 2, "client_id": "C004", "essais": 1}, {"id": 3, "client_id": "C006", "essais": 1},
                    {"id": 6, "client_id": "C007", "essais": 2}, {"id": 1, "client_id": "C001", "essais": 1}, None]

    conn.close()

    vide = sqlite3.connect(":memory:")
    assert migrer(vide, MIGRATIONS_EXEMPLE) == 3 and migrer(vide, MIGRATIONS_EXEMPLE) == 0
    with pytest.raises(sqlite3.OperationalError, match="no such table: table_qui_n_existe_pas"):
        migrer(vide, MIGRATIONS_EXEMPLE + [MIGRATION_CASSEE])
    assert vide.execute("PRAGMA user_version").fetchone() == (3,)
    vide.close()


def test_serie_15_le_correcteur_ne_laisse_aucun_fichier():
    import defis.corrige_serie_15 as corrige
    from defis.checks_serie_15 import SERIE

    dossier = Path(tempfile.gettempdir())
    avant = set(dossier.glob("correcteur-serie15-*"))
    for defi in SERIE.defis:
        assert evaluer_defi(corrige, defi).statut == REUSSI
    assert set(dossier.glob("correcteur-serie15-*")) == avant


def test_serie_15_autres_ecritures_justes_acceptees():
    """Le correcteur juge le comportement : des solutions justes écrites autrement doivent passer."""

    def offre_debit_atomique(conn, client_id, campagne, montant):
        # try/except explicite, et la base vérifie elle-même le budget dans l'UPDATE.
        if montant <= 0 or math.isnan(montant):
            raise ValueError("montant")
        try:
            debit = conn.execute("UPDATE budget SET reste = reste - ? WHERE campagne = ? AND reste >= ?",
                                 (montant, campagne, montant))
            if debit.rowcount == 0:
                existe = conn.execute("SELECT 1 FROM budget WHERE campagne = ?", (campagne,)).fetchone()
                raise ValueError("budget insuffisant" if existe else "campagne inconnue")
            identifiant = conn.execute("INSERT INTO offres (client_id, campagne, montant) VALUES (?, ?, ?)",
                                       (client_id, campagne, montant)).lastrowid
            conn.commit()
            return identifiant
        except BaseException:
            conn.rollback()
            raise

    def offre_begin_immediate(conn, client_id, campagne, montant):
        if not montant > 0:
            raise ValueError("montant")
        conn.execute("BEGIN IMMEDIATE")
        try:
            reste = conn.execute("SELECT reste FROM budget WHERE campagne = :c", {"c": campagne}).fetchone()
            if reste is None or reste[0] < montant:
                raise ValueError("refusé")
            conn.execute("UPDATE budget SET reste = reste - :m WHERE campagne = :c", {"m": montant, "c": campagne})
            curseur = conn.cursor()
            curseur.execute("INSERT INTO offres (client_id, campagne, montant) VALUES (?, ?, ?)", (client_id, campagne, montant))
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return curseur.lastrowid

    def pages_requete_unique(conn, apres=None, taille=20):
        if not isinstance(taille, int) or not 0 < taille <= 100:
            raise ValueError("taille")
        curseur = conn.cursor()
        curseur.row_factory = sqlite3.Row
        curseur.execute("SELECT id, nom, score FROM clients WHERE :apres IS NULL OR id > :apres ORDER BY id LIMIT :n",
                        {"apres": apres, "n": taille + 1})
        lignes = curseur.fetchmany(taille + 1)
        page = [dict(ligne) for ligne in lignes[:taille]]
        return page, (page[-1]["id"] if len(lignes) > taille else None)

    def pages_avec_count(conn, apres=None, taille=20):
        if taille < 1 or taille > 100:
            raise ValueError("taille")
        borne = apres if apres is not None else ""
        page = [{"id": i, "nom": n, "score": s} for i, n, s in conn.execute(
            "SELECT id, nom, score FROM clients WHERE id > ? ORDER BY id LIMIT ?", (borne, taille))]
        if not page:
            return [], None
        (reste,) = conn.execute("SELECT COUNT(*) FROM clients WHERE id > ?", (page[-1]["id"],)).fetchone()
        return page, page[-1]["id"] if reste else None

    def factures_json_each(conn, ids):
        import json

        ids = list(dict.fromkeys(ids))
        resultat = {i: [] for i in ids}
        for client_id, mois, montant in conn.execute(
                "SELECT client_id, mois, montant FROM factures WHERE client_id IN (SELECT value FROM json_each(?))",
                (json.dumps(ids),)):
            resultat[client_id].append({"mois": mois, "montant": montant})
        for factures in resultat.values():
            factures.sort(key=lambda f: f["mois"])  # trier en Python plutôt qu'en SQL
        return resultat

    def factures_par_paquets(conn, ids):
        ids, resultat = list(ids), {}
        for i in ids:
            resultat.setdefault(i, [])
        uniques = list(resultat)
        for debut in range(0, len(uniques), 300):  # paquets de 300 : 2 requêtes pour 500 clients
            paquet = uniques[debut:debut + 300]
            requete = f"SELECT client_id, mois, montant FROM factures WHERE client_id IN ({','.join('?' for _ in paquet)}) ORDER BY mois"
            for client_id, mois, montant in conn.execute(requete, paquet).fetchall():
                resultat[client_id].append({"mois": mois, "montant": montant})
        return resultat

    def factures_pandas(conn, ids):
        ids = list(dict.fromkeys(ids))
        if not ids:
            return {}
        table = pd.read_sql_query(
            f"SELECT client_id, mois, montant FROM factures WHERE client_id IN ({', '.join(['?'] * len(ids))})",
            conn, params=ids).sort_values(["client_id", "mois"])
        groupes = {cle: g[["mois", "montant"]].to_dict("records") for cle, g in table.groupby("client_id")}
        return {i: [{"mois": f["mois"], "montant": float(f["montant"])} for f in groupes.get(i, [])] for i in ids}

    def migrer_autocommit(conn, migrations):
        # Le mode « autocommit » de sqlite3 : Python n'ouvre plus aucune transaction, on écrit BEGIN / COMMIT.
        ancien, conn.isolation_level = conn.isolation_level, None
        try:
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            if version > len(migrations):
                raise ValueError("base plus récente")
            for numero, instructions in enumerate(migrations[version:], start=version + 1):
                conn.execute("BEGIN")
                try:
                    conn.execute("PRAGMA user_version = %d" % numero)  # au début : c'est la même transaction
                    for instruction in instructions:
                        conn.execute(instruction)
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            return len(migrations) - version
        finally:
            conn.isolation_level = ancien

    def migrer_with_begin(conn, migrations):
        version = conn.execute("pragma user_version").fetchone()[0]
        if version > len(migrations):
            raise ValueError("trop récente")
        faites = 0
        for i in range(version, len(migrations)):
            with conn:
                conn.execute("BEGIN")
                for instruction in migrations[i]:
                    conn.execute(instruction)
                conn.execute(f"pragma user_version={i + 1}")
            faites += 1
        return faites

    def migrer_script_encadre(conn, migrations):
        # executescript est correct SI le script ouvre et ferme lui-même sa transaction.
        (version,) = conn.execute("PRAGMA user_version").fetchone()
        if version > len(migrations):
            raise ValueError("trop récente")
        for i in range(version, len(migrations)):
            script = "BEGIN;\n" + ";\n".join(migrations[i]) + f";\nPRAGMA user_version = {i + 1};\nCOMMIT;"
            try:
                conn.executescript(script)
            except BaseException:
                conn.rollback()
                raise
        return len(migrations) - version

    def statut_verrou_pessimiste(conn, client_id, statut, version_lue):
        if statut not in STATUTS:
            raise ValueError(statut)
        conn.execute("BEGIN IMMEDIATE")  # personne ne peut écrire avant notre COMMIT
        try:
            ligne = conn.execute("SELECT version FROM clients WHERE id = ?", (client_id,)).fetchone()
            if ligne is None:
                raise KeyError(client_id)
            if ligne[0] != version_lue:
                raise ConflitDeVersion("rechargez")
            conn.execute("UPDATE clients SET statut = ?, version = ? WHERE id = ?", (statut, version_lue + 1, client_id))
        except BaseException:
            conn.rollback()
            raise
        conn.commit()
        return version_lue + 1

    def statut_existence_puis_update(conn, client_id, statut, version_lue):
        if statut not in STATUTS:
            raise ValueError(statut)
        if conn.execute("SELECT COUNT(*) FROM clients WHERE id = ?", (client_id,)).fetchone()[0] == 0:
            raise KeyError(client_id)
        with conn:
            modifiees = conn.execute("UPDATE clients SET version = version + 1, statut = ? WHERE version = ? AND id = ?",
                                     (statut, version_lue, client_id)).rowcount
        if not modifiees:
            raise ConflitDeVersion("conflit")
        return version_lue + 1

    def tache_verrou_pessimiste(conn, travailleur, maintenant, bail=BAIL):
        conn.execute("BEGIN IMMEDIATE")
        try:
            ligne = conn.execute("SELECT id, client_id, essais FROM taches WHERE statut = 'a_faire' "
                                 "OR (statut = 'en_cours' AND pris_a <= ? - ?) ORDER BY priorite DESC, id LIMIT 1",
                                 (maintenant, bail)).fetchone()
            if ligne is not None:
                conn.execute("UPDATE taches SET statut = 'en_cours', travailleur = ?, pris_a = ?, essais = ? WHERE id = ?",
                             (travailleur, maintenant, ligne[2] + 1, ligne[0]))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return None if ligne is None else {"id": ligne[0], "client_id": ligne[1], "essais": ligne[2] + 1}

    def tache_returning(conn, travailleur, maintenant, bail=BAIL):
        with conn:
            ligne = conn.execute(
                "UPDATE taches SET statut = 'en_cours', travailleur = ?, pris_a = ?, essais = essais + 1 "
                "WHERE id = (SELECT id FROM taches WHERE statut = 'a_faire' OR (statut = 'en_cours' AND pris_a + ? <= ?) "
                "ORDER BY priorite DESC, id LIMIT 1) RETURNING id, client_id, essais",
                (travailleur, maintenant, bail, maintenant)).fetchone()
        return None if ligne is None else dict(zip(("id", "client_id", "essais"), ligne))

    justes = [("accorder_offre", offre_debit_atomique), ("accorder_offre", offre_begin_immediate),
              ("page_clients", pages_requete_unique), ("page_clients", pages_avec_count),
              ("factures_par_client", factures_json_each), ("factures_par_client", factures_par_paquets),
              ("factures_par_client", factures_pandas),
              ("migrer", migrer_autocommit), ("migrer", migrer_with_begin), ("migrer", migrer_script_encadre),
              ("changer_statut", statut_verrou_pessimiste), ("changer_statut", statut_existence_puis_update),
              ("prendre_tache", tache_verrou_pessimiste)]
    if sqlite3.sqlite_version_info >= (3, 35):  # RETURNING
        justes.append(("prendre_tache", tache_returning))
    for nom, juste in justes:
        resultat = _evaluer(nom, juste)
        assert resultat.statut == REUSSI, (nom, juste.__name__, resultat.message)


def test_serie_15_une_panne_ne_valide_pas_une_migration_qu_un_redemarrage_aurait_annulee():
    """Une migration qui n'attrape que sqlite3.Error laisse passer la panne sans rollback : au redémarrage,
    SQLite annule la transaction ouverte. Le correcteur ne doit pas la valider à sa place."""

    def migrer_except_etroit(conn, migrations):
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > len(migrations):
            raise ValueError("trop récente")
        for i in range(version, len(migrations)):
            conn.execute("BEGIN")
            try:
                for instruction in migrations[i]:
                    conn.execute(instruction)
                conn.execute(f"PRAGMA user_version = {i + 1}")
                conn.commit()
            except sqlite3.Error:
                conn.rollback()
                raise
        return len(migrations) - version

    assert _evaluer("migrer", migrer_except_etroit).statut == REUSSI


# --------------------------------------------------------------------------- Série 16 : asyncio

import asyncio  # noqa: E402

from defis.concurrence import (  # noqa: E402
    SCORES_EXEMPLE,
    Interblocage,
    NumeroInvalide,
    OperateurSMS,
    Pages,
    ServeurAuth,
    ServiceIndisponible,
    ServiceScores,
    executer,
)


def _echec_16(nom, fonction, *mots):
    resultat = _evaluer(nom, fonction)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


# Défi 98 : gather


def _paralleles(sequentiel=False, doublons=True, arrivee=False, sans_await=False, avale=False, dormir=False,
                synchrone=False, asyncio_run=False):
    async def scores_en_parallele(service, client_ids):
        uniques = list(dict.fromkeys(client_ids)) if doublons else list(client_ids)
        if dormir:
            time.sleep(0.01)
        if sans_await:
            return {c: service.score(c) for c in uniques}
        try:
            if sequentiel:
                return {c: await service.score(c) for c in uniques}
            if arrivee:
                async def avec_id(c):
                    return c, await service.score(c)
                return dict([await t for t in asyncio.as_completed([avec_id(c) for c in uniques])])
            scores = await asyncio.gather(*(service.score(c) for c in uniques))
            return dict(zip(uniques, scores))
        except KeyError:
            if avale:
                return {}
            raise

    if synchrone:
        def scores_synchrones(service, client_ids):
            if asyncio_run:
                return asyncio.run(scores_en_parallele(service, client_ids))
            return {}
        return scores_synchrones
    return scores_en_parallele


@pytest.mark.parametrize("mutant,mots", [
    (_paralleles(sequentiel=True), ("1,6 s", "l'un APRÈS l'autre", "gather")),
    (_paralleles(doublons=False), ("C003 a été demandé 2 fois", "dict.fromkeys")),
    (_paralleles(arrivee=True), ("dans l'ordre de la liste reçue", "as_completed")),
    (_paralleles(sans_await=True), ("Il manque un `await`",)),
    (_paralleles(avale=True), ("Laissez l'erreur remonter",)),
    (_paralleles(dormir=True), ("time.sleep()", "BLOQUE toute la boucle", "asyncio.sleep")),
    (_paralleles(synchrone=True), ("au lieu d'une coroutine", "async def")),
    (_paralleles(synchrone=True, asyncio_run=True), ("n'appelez pas asyncio.run()",)),
])
def test_serie_16_gather_erreurs_classiques(mutant, mots):
    _echec_16("scores_en_parallele", mutant, *mots)


# Défi 99 : délai maximal


def _delai(mode="wait_for", attrape=TimeoutError, ou=False, valide=True):
    async def score_ou_defaut(service, client_id, delai, defaut=None):
        if valide and not delai > 0:
            raise ValueError("délai")
        try:
            if mode == "wait_for":
                score = await asyncio.wait_for(service.score(client_id), timeout=delai)
            elif mode == "timeout":
                async with asyncio.timeout(delai):
                    score = await service.score(client_id)
            elif mode == "direct":
                score = await service.score(client_id)
            elif mode in ("wait", "wait_annule"):
                tache = asyncio.create_task(service.score(client_id))
                finies, _ = await asyncio.wait({tache}, timeout=delai)
                if not finies:
                    if mode == "wait_annule":
                        tache.cancel()
                    raise TimeoutError
                score = tache.result()
            elif mode == "dormir_puis_regarder":
                tache = asyncio.create_task(service.score(client_id))
                await asyncio.sleep(delai)
                if not tache.done():
                    tache.cancel()
                    raise TimeoutError
                score = tache.result()
            return score or defaut if ou else score
        except attrape:
            return defaut

    return score_ou_defaut


@pytest.mark.parametrize("mutant,mots", [
    (_delai(mode="direct"), ("a attendu la réponse lente", "wait_for")),
    (_delai(mode="wait"), ("tourne ENCORE en arrière-plan", "tache.cancel()")),
    (_delai(mode="dormir_puis_regarder"), ("attend toujours la fin du délai",)),
    (_delai(attrape=Exception), ("SEUL le dépassement du délai", "pas Exception")),
    (_delai(attrape=KeyError), ("laisse remonter TimeoutError",)),
    (_delai(ou=True), ("0.0 est un VRAI score", "score or defaut")),
    (_delai(valide=False), ("delai=0", "ValueError")),
])
def test_serie_16_delai_erreurs_classiques(mutant, mots):
    _echec_16("score_ou_defaut", mutant, *mots)


def test_serie_16_delai_autres_ecritures_justes_acceptees():
    for juste in (_delai(mode="timeout"), _delai(mode="wait_annule"), _delai(attrape=asyncio.TimeoutError)):
        assert _evaluer("score_ou_defaut", juste).statut == REUSSI


def test_serie_16_le_vrai_sleep_est_restaure():
    vrai = time.sleep
    _evaluer("scores_en_parallele", _paralleles(dormir=True))
    assert time.sleep is vrai


# Défi 100 : Semaphore


def _campagne(limite="semaphore", erreurs="return_exceptions", dedoublonner=True, garder="exception", valide=True,
              ordre=True):
    async def envoyer_campagne(operateur, clients, texte, max_simultanes=10):
        if valide and max_simultanes < 1:
            raise ValueError("max_simultanes")
        uniques = list(dict.fromkeys(clients)) if dedoublonner else list(clients)
        places = asyncio.Semaphore(10 if limite == "dix" else max(max_simultanes, 0))
        envoyes, echecs = {}, {}

        def noter(client_id, resultat):
            if isinstance(resultat, Exception):
                echecs[client_id] = resultat if garder == "exception" else str(resultat)
            else:
                envoyes[client_id] = resultat

        async def envoyer_un(client_id):
            if limite is None:
                resultat = await operateur.envoyer(client_id, texte)
            else:
                async with places:
                    resultat = await operateur.envoyer(client_id, texte)
            return resultat

        async def envoyer_et_noter(client_id):
            try:
                noter(client_id, await envoyer_un(client_id))
            except Exception as exc:
                noter(client_id, exc)

        if limite == "sequentiel":
            resultats = []
            for client_id in uniques:
                try:
                    resultats.append(await operateur.envoyer(client_id, texte))
                except Exception as exc:
                    resultats.append(exc)
        elif limite == "paquets":
            resultats = []
            for i in range(0, len(uniques), max_simultanes):
                paquet = uniques[i:i + max_simultanes]
                resultats += await asyncio.gather(*(operateur.envoyer(c, texte) for c in paquet),
                                                  return_exceptions=True)
        elif not ordre:
            await asyncio.gather(*(envoyer_et_noter(c) for c in uniques))
            return envoyes, echecs
        else:
            resultats = await asyncio.gather(*(envoyer_un(c) for c in uniques),
                                             return_exceptions=erreurs == "return_exceptions")
        for client_id, resultat in zip(uniques, resultats):
            noter(client_id, resultat)
        return envoyes, echecs

    return envoyer_campagne


@pytest.mark.parametrize("mutant,mots", [
    (_campagne(limite=None), ("refusé", "Surcharge", "asyncio.Semaphore(max_simultanes)")),
    (_campagne(limite="sequentiel"), ("un par un",)),
    (_campagne(limite="paquets"), ("PAQUETS", "le plus lent")),
    (_campagne(limite="dix"), ("max_simultanes=3", "Respectez le paramètre")),
    (_campagne(erreurs="propager"), ("arrêté TOUTE la campagne", "return_exceptions=True")),
    (_campagne(dedoublonner=False), ("C001 a reçu 2 SMS", "dict.fromkeys")),
    (_campagne(garder="texte"), ("Gardez l'EXCEPTION",)),
    (_campagne(ordre=False), ("dans l'ordre de la liste `clients`",)),
    (_campagne(valide=False), ("max_simultanes=0", "ValueError")),
])
def test_serie_16_campagne_erreurs_classiques(mutant, mots):
    _echec_16("envoyer_campagne", mutant, *mots)


# Défi 101 : Lock


def _cache(verrou=True, double=True, marge_fixe=None, horloge="boucle", verrou_de_classe=False, sans_finally=False,
           avec_marge=True, garder=True, couple=False, avale=False, expiration_avant=False, synchrone=False):
    def maintenant():
        return asyncio.get_running_loop().time() if horloge == "boucle" else time.monotonic()

    class CacheJeton:
        _verrou_commun = asyncio.Lock() if verrou_de_classe else None

        def __init__(self, serveur, marge=60.0):
            self.serveur = serveur
            self.marge = marge if marge_fixe is None else marge_fixe
            self._jeton, self._expire_a = None, float("-inf")
            self._verrou = self._verrou_commun if verrou_de_classe else asyncio.Lock()

        def _valable(self):
            limite = self._expire_a - (self.marge if avec_marge else 0.0)
            if expiration_avant:
                return maintenant() < limite
            return garder and self._jeton is not None and maintenant() < limite

        async def _renouveler(self):
            debut = maintenant()
            if expiration_avant:
                self._expire_a = debut + 3600.0
            try:
                jeton, duree = await self.serveur.nouveau_jeton()
            except ServiceIndisponible:
                if avale:
                    return None
                raise
            self._jeton, self._expire_a = (jeton, duree) if couple else jeton, debut + duree

        async def _jeton_async(self):
            if self._valable():
                return self._jeton
            if not verrou:
                await self._renouveler()
                return self._jeton
            if sans_finally:
                await self._verrou.acquire()
                if not double or not self._valable():
                    await self._renouveler()
                self._verrou.release()
                return self._jeton
            async with self._verrou:
                if not double or not self._valable():
                    await self._renouveler()
                return self._jeton

        if synchrone:
            def jeton(self):
                return "jeton-1"
        else:
            async def jeton(self):
                return await self._jeton_async()

    return CacheJeton


def _cache_tache_partagee():
    """Toutes les tâches attendent la MÊME demande (une tâche partagée)… qui reste en échec pour toujours."""

    class CacheJeton:
        def __init__(self, serveur, marge=60.0):
            self.serveur, self.marge = serveur, marge
            self._demande, self._expire_a = None, float("-inf")

        async def _demander(self):
            debut = asyncio.get_running_loop().time()
            jeton, duree = await self.serveur.nouveau_jeton()
            self._expire_a = debut + duree
            return jeton

        async def jeton(self):
            perime = asyncio.get_running_loop().time() >= self._expire_a - self.marge
            if self._demande is None or (perime and self._demande.done() and self._demande.exception() is None):
                self._demande = asyncio.ensure_future(self._demander())
            return await asyncio.shield(self._demande)

    return CacheJeton


@pytest.mark.parametrize("mutant,mots", [
    (_cache(verrou=False), ("demandes au lieu d'une", "asyncio.Lock()")),
    (_cache(double=False), ("l'une APRÈS l'autre", "double vérification")),
    (_cache(marge_fixe=60.0), ("pas 60 écrit en dur",)),
    (_cache(horloge="monotonic"), ("t = 3700 s", "asyncio.get_running_loop().time()")),
    (_cache(verrou_de_classe=True), ("CLASSE",)),
    (_cache(sans_finally=True), ("le verrou n'a jamais été rendu", "async with")),
    (_cache(avec_marge=False), ("t = 3550 s", "MAINTENANT")),
    (_cache(garder=False), ("demandes",)),
    (_cache(couple=True), ("ne renvoyez que le jeton",)),
    (_cache(avale=True), ("Laissez l'erreur remonter",)),
    (_cache(expiration_avant=True), ("49 appels ont reçu None", "AVANT d'avoir la réponse")),
    (_cache_tache_partagee(), ("rien ne doit rester en cache", "Future")),
    (_cache(synchrone=True), ("async def jeton(self)",)),
])
def test_serie_16_jeton_erreurs_classiques(mutant, mots):
    _echec_16("CacheJeton", mutant, *mots)


# Défi 102 : le premier qui répond


def _premier(mode="as_completed", annuler=True, tous_echouent="relancer", valide=True):
    async def premier_qui_repond(serveurs, client_id):
        if valide and not serveurs:
            raise ValueError("aucun serveur")
        if mode == "sequentiel":
            derniere = None
            for serveur in serveurs:
                try:
                    return await serveur.score(client_id)
                except Exception as exc:
                    derniere = exc
            raise derniere
        if mode == "gather":
            resultats = await asyncio.gather(*(s.score(client_id) for s in serveurs), return_exceptions=True)
            bons = [r for r in resultats if not isinstance(r, Exception)]
            if bons:
                return bons[0]
            raise resultats[0]
        if mode == "wait_coroutines":
            finies, _ = await asyncio.wait([s.score(client_id) for s in serveurs],
                                           return_when=asyncio.FIRST_COMPLETED)
            return finies.pop().result()
        taches = [asyncio.create_task(s.score(client_id)) for s in serveurs]
        try:
            if mode == "liste":
                derniere = None
                for tache in taches:
                    try:
                        return await tache
                    except Exception as exc:
                        derniere = exc
                raise derniere
            if mode == "premiere_finie":
                finies, _ = await asyncio.wait(taches, return_when=asyncio.FIRST_COMPLETED)
                return finies.pop().result()
            if mode == "wait_boucle":
                en_attente, derniere = set(taches), None
                while en_attente:
                    finies, en_attente = await asyncio.wait(en_attente, return_when=asyncio.FIRST_COMPLETED)
                    for tache in finies:
                        if tache.exception() is None:
                            return tache.result()
                        derniere = tache.exception()
                raise derniere
            derniere = None
            for prochaine in asyncio.as_completed(taches):
                try:
                    return await prochaine
                except Exception as exc:
                    derniere = exc
            if tous_echouent == "none":
                return None
            if tous_echouent == "autre":
                raise RuntimeError("tous les serveurs sont en panne")
            raise derniere
        finally:
            if annuler:
                for tache in taches:
                    tache.cancel()

    return premier_qui_repond


@pytest.mark.parametrize("mutant,mots", [
    (_premier(mode="gather"), ("attend les 3 réponses", "as_completed")),
    (_premier(mode="sequentiel"), ("n'ont pas été interrogés", "en même temps")),
    (_premier(mode="liste"), ("premier serveur DE LA LISTE",)),
    (_premier(annuler=False), ("travaille encore pour rien", "tache.cancel()")),
    (_premier(mode="premiere_finie"), ("Une erreur n'est pas une réponse",)),
    (_premier(tous_echouent="none"), ("Levez une erreur",)),
    (_premier(tous_echouent="autre"), ("Levez l'une des erreurs reçues",)),
    (_premier(valide=False), ("liste de serveurs vide", "ValueError")),
    (_premier(mode="wait_coroutines"), ("TÂCHES", "asyncio.create_task")),
])
def test_serie_16_premier_erreurs_classiques(mutant, mots):
    _echec_16("premier_qui_repond", mutant, *mots)


def test_serie_16_premier_variante_asyncio_wait_acceptee():
    assert _evaluer("premier_qui_repond", _premier(mode="wait_boucle")).statut == REUSSI


# Défi 103 : Queue et travailleurs


def _scorer(mode="taskgroup", borne=True, fin="sentinelles", travailleurs_en_plus=0, valide=True):
    async def scorer_tout(pages, service, nb_travailleurs=4, max_en_attente=20):
        if valide and (nb_travailleurs < 1 or max_en_attente < 1):
            raise ValueError("paramètres")
        scores = {}
        if mode == "tout_lire":
            ids = [c async for page in pages for c in page]
            places = asyncio.Semaphore(nb_travailleurs)

            async def un(c):
                async with places:
                    scores[c] = await service.score(c)
            async with asyncio.TaskGroup() as groupe:
                for c in ids:
                    groupe.create_task(un(c))
            return scores
        if mode == "paquets":
            async for page in pages:
                for i in range(0, len(page), nb_travailleurs):
                    paquet = page[i:i + nb_travailleurs]
                    for c, s in zip(paquet, await asyncio.gather(*(service.score(c) for c in paquet))):
                        scores[c] = s
            return scores
        if mode == "semaphore_en_lisant":  # une autre écriture juste : une place obtenue AVANT de prendre le client
            places = asyncio.Semaphore(nb_travailleurs)

            async def un(c):
                try:
                    scores[c] = await service.score(c)
                finally:
                    places.release()
            async with asyncio.TaskGroup() as groupe:
                async for page in pages:
                    for c in page:
                        await places.acquire()
                        groupe.create_task(un(c))
            return scores
        file = asyncio.Queue(maxsize=max_en_attente if borne else 0)
        FIN = object()
        n = nb_travailleurs + travailleurs_en_plus

        async def lire(sentinelles=fin == "sentinelles"):
            async for page in pages:
                for c in page:
                    await file.put(c)
            if sentinelles:
                for _ in range(n):
                    await file.put(FIN)

        async def travailleur():
            while True:
                c = await file.get()
                try:
                    if c is FIN:
                        return
                    scores[c] = await service.score(c)
                finally:
                    file.task_done()

        if mode == "taskgroup":
            async with asyncio.TaskGroup() as groupe:
                groupe.create_task(lire())
                for _ in range(n):
                    groupe.create_task(travailleur())
            return scores
        if mode == "join":
            async def travailleur_join():  # task_done() seulement après un succès : le piège
                while True:
                    c = await file.get()
                    scores[c] = await service.score(c)
                    file.task_done()
            taches = [asyncio.create_task(travailleur_join()) for _ in range(n)]
            await lire(sentinelles=False)
            await file.join()
            for tache in taches:
                tache.cancel()
            return scores
        if mode == "join_dans_taskgroup":
            async with asyncio.TaskGroup() as groupe:
                taches = [groupe.create_task(travailleur()) for _ in range(n)]
                await lire(sentinelles=False)
                await file.join()
                for tache in taches:
                    tache.cancel()
            return scores
        if mode == "gather":
            await asyncio.gather(lire(), *(travailleur() for _ in range(n)))
            return scores

    return scorer_tout


@pytest.mark.parametrize("mutant,mots", [
    (_scorer(borne=False), ("clients déjà lus attendaient", "maxsize")),
    (_scorer(mode="tout_lire"), ("clients déjà lus attendaient",)),
    (_scorer(mode="paquets"), ("Des travailleurs restent sans rien faire",)),
    (_scorer(fin="aucune"), ("ne se termine JAMAIS", "signal de fin")),
    (_scorer(mode="join"), ("Quand le service lève ServiceIndisponible pour C013", "ne se termine JAMAIS", "TaskGroup")),
    (_scorer(mode="gather"), ("tournent encore derrière vous", "TaskGroup")),
    (_scorer(travailleurs_en_plus=1), ("jusqu'à 5 appels", "nb_travailleurs=4")),
    (_scorer(valide=False), ("nb_travailleurs=0", "ValueError")),
])
def test_serie_16_scorer_tout_erreurs_classiques(mutant, mots):
    _echec_16("scorer_tout", mutant, *mots)


def test_serie_16_scorer_tout_autres_ecritures_justes_acceptees():
    for mode in ("semaphore_en_lisant", "join_dans_taskgroup"):
        resultat = _evaluer("scorer_tout", _scorer(mode=mode))
        assert resultat.statut == REUSSI, (mode, resultat.message)


@pytest.mark.parametrize("nom,fabrique", [("scores_en_parallele", _paralleles), ("score_ou_defaut", _delai),
                                          ("envoyer_campagne", _campagne), ("CacheJeton", _cache),
                                          ("premier_qui_repond", _premier), ("scorer_tout", _scorer)])
def test_serie_16_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Garde-fou : chaque version boguée ci-dessus ne diffère du juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_serie_16_les_tests_du_corrige_passent():
    import defis.corrige_serie_16 as corrige

    tests = [f for nom, f in vars(corrige).items() if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_16_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.concurrence as concurrence
    import defis.serie_16 as eleve

    for nom in ("SCORES_EXEMPLE", "Interblocage", "NumeroInvalide", "OperateurSMS", "Pages", "ServeurAuth",
                "ServiceIndisponible", "ServiceScores", "Surcharge", "essayer", "service_exemple"):
        assert getattr(eleve, nom) is getattr(concurrence, nom), nom


def test_serie_16_les_exemples_des_docstrings_sont_justes(capsys):
    """Les exemples de defis/serie_16.py et de defis/concurrence.py, rejoués avec le corrigé."""
    from defis.corrige_serie_16 import (CacheJeton, envoyer_campagne, essayer, premier_qui_repond, score_ou_defaut,
                                        scorer_tout, scores_en_parallele, service_exemple)

    duree = "⏱️  {} s (horloge virtuelle : rien n'a vraiment attendu)"

    def affiche(*durees):
        assert capsys.readouterr().out.splitlines() == [duree.format(d) for d in durees]

    service = service_exemple()
    assert essayer(service.score("C001")) == 0.82
    assert essayer(scores_en_parallele(service, ["C001", "C002", "C003"])) == {"C001": 0.82, "C002": 0.15,
                                                                               "C003": 0.47}
    affiche("0,2", "0,2")
    assert essayer(ServiceScores(latence=0.2).score("C005")) == 0.0
    affiche("0,2")

    service = service_exemple()
    assert essayer(scores_en_parallele(service, ["C001", "C004", "C001"])) == {"C001": 0.82, "C004": 0.91}
    assert (len(service.appels), service.max_en_cours) == (2, 2)
    assert essayer(score_ou_defaut(service_exemple(latence=0.2), "C001", delai=1.0)) == 0.82
    assert essayer(score_ou_defaut(service_exemple(latence=30.0), "C001", delai=1.0, defaut=0.5)) == 0.5
    affiche("0,2", "0,2", "1")

    operateur = OperateurSMS(max_simultanes=3, invalides={"C002"})
    envoyes, echecs = essayer(envoyer_campagne(operateur, list(SCORES_EXEMPLE), "Offre fidélité", 3))
    assert repr(echecs) == "{'C002': NumeroInvalide('numéro de téléphone invalide pour C002')}"
    assert (len(envoyes), operateur.max_en_cours, operateur.refus) == (7, 3, [])
    affiche("0,6")

    serveur = ServeurAuth(latence=0.5)

    async def cinquante_appels():
        cache = CacheJeton(serveur)
        return await asyncio.gather(*(cache.jeton() for _ in range(50)))

    assert set(essayer(cinquante_appels())) == {"jeton-1"} and len(serveur.demandes) == 1
    affiche("0,5")

    paris = ServiceScores(latence=0.3, nom="paris")
    francfort = ServiceScores(latence=0.1, nom="francfort")
    dublin = ServiceScores(latence=0.5, nom="dublin")
    assert essayer(premier_qui_repond([paris, francfort, dublin], "C004")) == 0.91
    affiche("0,1")

    ids = [f"C{n:03d}" for n in range(1, 21)]
    service = ServiceScores({c: 0.5 for c in ids}, latence=0.2)
    pages = Pages(ids, taille=5, service=service)
    scores = essayer(scorer_tout(pages, service, nb_travailleurs=4, max_en_attente=5))
    assert (len(scores), service.max_en_cours, max(avance for _, avance in pages.demandes)) == (20, 4, 5)
    affiche("1,1")

    pages = Pages(["C001", "C002", "C003", "C004", "C005"], taille=2)

    async def tout_lire():
        return [page async for page in pages]

    assert essayer(tout_lire()) == [["C001", "C002"], ["C003", "C004"], ["C005"]]
    affiche("0,4")


def test_serie_16_essayer_signale_les_taches_oubliees_et_les_interblocages(capsys):
    from defis.concurrence import essayer

    lent = ServiceScores(latence=5.0)

    async def oublie_une_tache():
        asyncio.create_task(lent.score("C001"))
        await asyncio.sleep(0.1)
        return "fini"

    assert essayer(oublie_une_tache()) == "fini"
    sortie = capsys.readouterr().out
    assert "0,1 s" in sortie and "1 tâche(s) tournaient encore" in sortie
    assert lent.annules == ["C001"]

    async def attend_pour_toujours():
        await asyncio.Queue().get()

    with pytest.raises(Interblocage):
        essayer(attend_pour_toujours())
    assert "Interblocage" in capsys.readouterr().out


def test_serie_16_l_horloge_virtuelle_ne_touche_pas_a_la_boucle_courante():
    """Le correcteur crée ses propres boucles : il ne doit ni en laisser une ouverte, ni en installer une."""
    for mutant in (_paralleles(sans_await=True), _paralleles(synchrone=True, asyncio_run=True)):
        _evaluer("scores_en_parallele", mutant)
    _evaluer("scorer_tout", _scorer(mode="gather"))
    _evaluer("scorer_tout", _scorer(mode="join"))
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()
    resultat, duree = executer(asyncio.sleep(3600, result="ok"))
    assert (resultat, duree) == ("ok", 3600.0)

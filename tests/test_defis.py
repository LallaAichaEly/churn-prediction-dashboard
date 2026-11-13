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


# --------------------------------------------------------------------------- Série 17 : concevoir ses propres types

import re  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from decimal import Decimal, InvalidOperation  # noqa: E402
from enum import Enum  # noqa: E402
from functools import total_ordering  # noqa: E402

from defis.conception import CLIENTS_EXEMPLE as CLIENTS_17, CONTRATS as CONTRATS_17, SCORES_EXEMPLE as SCORES_17  # noqa: E402
from defis.conception import Passerelle, PasserelleIndisponible  # noqa: E402


def _echec_17(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


# Défi 104 : dataclass


def _client_17(defaut=None):
    @dataclass(frozen=defaut != "non_gele")
    class Client:
        client_id: str
        anciennete_mois: int
        facture_mensuelle: float
        contrat: str = "Mensuel"
        options: tuple = ()

        def __post_init__(self):
            def refuser(erreur, champ, texte):
                erreur = ValueError if defaut == "valueerror" else erreur
                raise erreur(texte if defaut == "sans_champ" else f"{champ} : {texte}")

            cherche = re.match if defaut == "match" else re.fullmatch
            if not isinstance(self.client_id, str):
                refuser(TypeError, "client_id", "texte attendu")
            if not cherche(r"C\d+", self.client_id):
                refuser(ValueError, "client_id", "format C001")
            entier = isinstance(self.anciennete_mois, int)
            if defaut != "bool":
                entier = entier and not isinstance(self.anciennete_mois, bool)
            if not entier:
                refuser(TypeError, "anciennete_mois", "entier attendu")
            if self.anciennete_mois < 0:
                refuser(ValueError, "anciennete_mois", "négative")
            if isinstance(self.facture_mensuelle, bool) or not isinstance(self.facture_mensuelle, (int, float)):
                refuser(TypeError, "facture_mensuelle", "nombre attendu")
            if defaut == "nan":
                if self.facture_mensuelle < 0:
                    refuser(ValueError, "facture_mensuelle", "négative")
            elif not math.isfinite(self.facture_mensuelle) or self.facture_mensuelle < 0:
                refuser(ValueError, "facture_mensuelle", "nombre fini positif attendu")
            if self.contrat not in CONTRATS_17:
                refuser(ValueError, "contrat", "inconnu")
            if isinstance(self.options, str) and defaut != "texte_decoupe":
                refuser(TypeError, "options", "collection attendue")
            if not all(isinstance(o, str) for o in self.options):
                refuser(TypeError, "options", "textes attendus")
            if defaut != "liste":
                object.__setattr__(self, "options", tuple(self.options))

    return Client


@pytest.mark.parametrize("mutant,mots", [
    (_client_17("non_gele"), ("frozen=True",)),
    (_client_17("bool"), ("anciennete_mois=True", "bool hérite de int")),
    (_client_17("nan"), ("nan", "math.isfinite")),
    (_client_17("liste"), ("gardez un TUPLE", "object.__setattr__")),
    (_client_17("texte_decoupe"), ("découpé en lettres",)),
    (_client_17("match"), ("C001 '", "re.fullmatch")),
    (_client_17("sans_champ"), ("NOMMER le champ",)),
    (_client_17("valueerror"), ("au lieu de TypeError", "le TYPE ne va pas")),
])
def test_serie_17_client_erreurs_classiques(mutant, mots):
    _echec_17("Client", mutant, *mots)


def test_serie_17_client_doit_etre_une_dataclass():
    class Client:
        def __init__(self, client_id, anciennete_mois, facture_mensuelle, contrat="Mensuel", options=()):
            self.client_id, self.anciennete_mois = client_id, anciennete_mois

    _echec_17("Client", Client, "@dataclass(frozen=True)")


# Défi 105 : Enum


def _contrat_17(defaut=None, base=Enum):
    class Contrat(base):
        MENSUEL = "Mensuel"
        ANNUEL = "Annuel"
        BIANNUEL = "Biannuel"
        if defaut == "membre_dict":
            _ENGAGEMENTS = {"Mensuel": 1, "Annuel": 12, "Biannuel": 24}

        @classmethod
        def depuis_texte(cls, texte):
            if isinstance(texte, cls):
                return texte
            if not isinstance(texte, str) and defaut != "none":
                raise TypeError("texte attendu")
            if defaut == "sans_casse":
                try:
                    return cls(texte)
                except ValueError:
                    raise ValueError("contrat inconnu (Mensuel, Annuel, Biannuel)") from None
            cle = texte.strip().casefold()
            for membre in cls:
                valeur = membre.value.casefold()
                trouve = valeur.startswith(cle) if defaut == "startswith" and cle else valeur == cle
                if trouve:
                    return membre.value if defaut == "texte" else membre
            raise ValueError("contrat inconnu" if defaut == "message" else
                             f"contrat inconnu : {texte!r} (attendu : Mensuel, Annuel, Biannuel)")

        if defaut == "methode":
            def engagement_mois(self):
                return {"Mensuel": 1, "Annuel": 12, "Biannuel": 24}[self.value]
        else:
            @property
            def engagement_mois(self):
                return {"Mensuel": 1, "Annuel": 12, "Biannuel": 24}[self.value]

        if defaut != "sans_str":
            def __str__(self):
                return self.value

    return Contrat


@pytest.mark.parametrize("mutant,mots", [
    (_contrat_17("membre_dict"), ("4 : MENSUEL, ANNUEL, BIANNUEL, _ENGAGEMENTS", "devient un membre")),
    (_contrat_17("texte"), ("renvoyez le MEMBRE",)),
    (_contrat_17("startswith"), ("'Mens'", "pas de startswith")),
    (_contrat_17("sans_casse"), ("'mensuel'", "casefold")),
    (_contrat_17("methode"), ("@property",)),
    (_contrat_17("sans_str"), ("Contrat.MENSUEL", "__str__")),
    (_contrat_17("message"), ("valeurs acceptées",)),
    (_contrat_17("none"), ("depuis_texte(None)", "isinstance(texte, str)")),
])
def test_serie_17_contrat_erreurs_classiques(mutant, mots):
    _echec_17("Contrat", mutant, *mots)


def test_serie_17_contrat_strenum_accepte():
    from enum import StrEnum

    assert _evaluer("Contrat", _contrat_17(base=StrEnum)).statut == REUSSI


# Défi 106 : objet valeur


def _montant_17(defaut=None):
    from defis.corrige_serie_17 import Montant as Juste

    @total_ordering
    @dataclass(frozen=True, order=defaut == "order")
    class Montant:
        centimes: int
        devise: str = "EUR"

        __post_init__ = Juste.__post_init__
        _meme_devise = Juste._meme_devise

        @classmethod
        def depuis_euros(cls, euros, devise="EUR"):
            if defaut == "float":
                texte = str(euros).replace(" ", "").replace(" ", "").replace(" ", "").replace(",", ".")
                return cls(int(float(texte) * 100), devise)
            if defaut == "accepte_float" and isinstance(euros, float):
                euros = Decimal(str(euros))
            if isinstance(euros, bool) or not isinstance(euros, (str, int, Decimal)):
                raise TypeError("type refusé")
            if isinstance(euros, str):
                texte = euros.strip()
                espaces = (" ",) if defaut == "insecable" else (" ", " ", " ")
                for espace in espaces:
                    texte = texte.replace(espace, "")
                if defaut == "invalid_op":
                    valeur = Decimal(texte.replace(",", "."))
                else:
                    try:
                        valeur = Decimal(texte.replace(",", "."))
                    except InvalidOperation:
                        raise ValueError("illisible") from None
            else:
                valeur = Decimal(euros)
            if defaut != "nan" and not valeur.is_finite():
                raise ValueError("illisible")
            centimes = valeur * 100
            if centimes != centimes.to_integral_value():
                raise ValueError("2 décimales")
            return cls(int(centimes), devise)

        def __add__(self, autre):
            if not isinstance(autre, Montant):
                if defaut == "add_valueerror":
                    raise ValueError("pas un montant")
                return NotImplemented
            if defaut == "devises_typeerror" and self.devise != autre.devise:
                raise TypeError(f"{self.devise} et {autre.devise}")
            self._meme_devise(autre, "additionner")
            return Montant(self.centimes + autre.centimes, self.devise)

        if defaut == "radd_tout":
            def __radd__(self, autre):
                return self if isinstance(autre, (int, float)) else NotImplemented
        elif defaut != "sans_radd":
            __radd__ = Juste.__radd__

        def __mul__(self, n):
            if defaut == "mul_float" and isinstance(n, float):
                return Montant(int(self.centimes * n), self.devise)
            if isinstance(n, bool) or not isinstance(n, int):
                return NotImplemented
            return Montant(self.centimes * n, self.devise)

        if defaut != "sans_rmul":
            __rmul__ = __mul__

        if defaut != "order":
            def __lt__(self, autre):
                if not isinstance(autre, Montant):
                    return NotImplemented
                self._meme_devise(autre, "comparer")
                return self.centimes < autre.centimes

        def __str__(self):
            if defaut == "floor":
                euros, cts = divmod(self.centimes, 100)
                return f"{euros:,},{cts:02d} €".replace(",", " ", euros.bit_length() > 9).replace(" ,", ",")
            if defaut == "point":
                euros, cts = divmod(abs(self.centimes), 100)
                return f"{'-' if self.centimes < 0 else ''}{euros:,}.{cts:02d} €".replace(",", " ")
            return Juste.__str__(self)

    return Montant


@pytest.mark.parametrize("mutant,mots", [
    (_montant_17("order"), ("Montant(100, 'USD') < Montant(200)", "order=True")),
    (_montant_17("float"), ("1998", "Decimal")),
    (_montant_17("accepte_float"), ("depuis_euros(19.99)", "Un float n'est pas exact", "1998")),
    (_montant_17("invalid_op"), ("InvalidOperation", "PAS une ValueError")),
    (_montant_17("nan"), ("'Infinity'", "is_finite")),
    (_montant_17("insecable"), ("INSÉCABLES",)),
    (_montant_17("floor"), ("-5 // 100",)),
    (_montant_17("point"), ("19.99 €", "la virgule")),
    (_montant_17("sans_radd"), ("sum(", "__radd__(0)")),
    (_montant_17("radd_tout"), ("1 + Montant(100)", "__radd__ n'accepte que le 0")),
    (_montant_17("sans_rmul"), ("3 * Montant(1999)", "__rmul__")),
    (_montant_17("mul_float"), ("Montant(100) * 1.5", "un ENTIER")),
    (_montant_17("devises_typeerror"), ("au lieu de ValueError",)),
    (_montant_17("add_valueerror"), ("Montant(100) + 1", "NotImplemented")),
])
def test_serie_17_montant_erreurs_classiques(mutant, mots):
    _echec_17("Montant", mutant, *mots)


# Défi 107 : gestionnaire de contexte


def _campagne_17(defaut=None):
    if defaut in ("classe", "exit_true", "ouvre_tot"):
        class Campagne:
            def __init__(self, passerelle, nom):
                self.passerelle, self.nom = passerelle, nom
                if defaut == "ouvre_tot":
                    self.lot = passerelle.ouvrir(nom)

            def __enter__(self):
                if defaut != "ouvre_tot":
                    self.lot = self.passerelle.ouvrir(self.nom)
                return self.lot

            def __exit__(self, type_erreur, erreur, trace):
                try:
                    if type_erreur is None:
                        self.passerelle.valider(self.lot)
                    else:
                        self.passerelle.annuler(self.lot)
                finally:
                    self.passerelle.fermer(self.lot)
                return defaut == "exit_true"

        return Campagne

    @contextmanager
    def campagne(passerelle, nom):
        if defaut == "sans_try":
            lot = passerelle.ouvrir(nom)
            yield lot
            passerelle.valider(lot)
            passerelle.fermer(lot)
            return
        if defaut == "try_avant_ouvrir":
            lot = None
            try:
                lot = passerelle.ouvrir(nom)
                yield lot
            except BaseException:
                passerelle.annuler(lot)
                raise
            else:
                passerelle.valider(lot)
            finally:
                passerelle.fermer(lot)
            return
        lot = passerelle.ouvrir(nom)
        attrape = Exception if defaut == "exception" else BaseException
        try:
            yield lot
            if defaut == "valider_dans_try":
                passerelle.valider(lot)
        except attrape as exc:
            passerelle.annuler(lot)
            if defaut == "remplace":
                raise RuntimeError("campagne annulée") from exc
            if defaut != "avale":
                raise
        else:
            if defaut not in ("valider_dans_try", "valider_finally"):
                passerelle.valider(lot)
        finally:
            if defaut == "valider_finally":
                passerelle.valider(lot)
            passerelle.fermer(lot)

    return campagne


@pytest.mark.parametrize("mutant,mots", [
    (_campagne_17("sans_try"), ("ni annulé ni fermé", "AU NIVEAU DU `yield`")),
    (_campagne_17("exception"), ("Ctrl+C", "except BaseException")),
    (_campagne_17("avale"), ("AVALÉE", "`raise`")),
    (_campagne_17("exit_true"), ("AVALÉE", "__exit__ doit renvoyer False")),
    (_campagne_17("valider_finally"), ("VALIDÉ", "`else:`")),
    (_campagne_17("try_avant_ouvrir"), ("ouvrir() lève PasserelleIndisponible", "AVANT le try")),
    (_campagne_17("valider_dans_try"), ("déjà refusé", "`else:` du try")),
    (_campagne_17("remplace"), ("RuntimeError", "la MÊME")),
    (_campagne_17("ouvre_tot"), ("AVANT d'entrer",)),
])
def test_serie_17_campagne_erreurs_classiques(mutant, mots):
    _echec_17("campagne", mutant, *mots)


def test_serie_17_campagne_en_classe_acceptee():
    assert _evaluer("campagne", _campagne_17("classe")).statut == REUSSI


def test_serie_17_campagne_le_correcteur_attrape_le_ctrl_c():
    """Un KeyboardInterrupt lancé par le correcteur ne doit jamais s'échapper, même si le code de l'élève le laisse passer."""
    for defaut in ("sans_try", "exception", "remplace"):
        assert _evaluer("campagne", _campagne_17(defaut)).statut == ECHEC


# Défi 108 : collection


def _portefeuille_17(defaut=None):
    class Portefeuille:
        def __init__(self, paires=()):
            self._scores = [] if defaut == "liste_lente" else {}
            self._position = 0
            for client_id, score in paires:
                self.ajouter(client_id, score)

        def ajouter(self, client_id, score):
            if defaut == "liste_lente":
                if any(c == client_id for c, _ in self._scores):
                    raise ValueError("doublon")
                self._scores.append((client_id, score))
                return
            if client_id in self._scores and defaut != "doublon_remplace":
                raise ValueError("doublon")
            self._scores[client_id] = score
            if defaut == "iter_vue":
                tries = sorted(self._scores.items(), key=lambda p: (-p[1], p[0]))
                self._scores.clear()
                self._scores.update(tries)

        def _tries(self):
            paires = self._scores if defaut == "liste_lente" else self._scores.items()
            return sorted(paires, key=lambda p: (p[1], p[0]) if defaut == "croissant" else (-p[1], p[0]))

        def __len__(self):
            return len(self._scores)

        if defaut != "sans_contains":
            def __contains__(self, client_id):
                if defaut == "liste_lente":
                    return any(c == client_id for c, _ in self._scores)
                return client_id in self._scores

        def __getitem__(self, client_id):
            if defaut == "liste_lente":
                for c, s in self._scores:
                    if c == client_id:
                        return s
                raise KeyError(client_id)
            return self._scores.get(client_id) if defaut == "getitem_none" else self._scores[client_id]

        if defaut in ("iter_self", "iter_self_reset"):
            def __iter__(self):
                if defaut == "iter_self_reset" or not hasattr(self, "_liste"):
                    self._liste, self._position = self._tries(), 0
                return self

            def __next__(self):
                if self._position >= len(self._liste):
                    raise StopIteration
                self._position += 1
                return self._liste[self._position - 1]
        elif defaut == "iter_vue":
            def __iter__(self):
                return iter(self._scores.items())
        elif defaut == "listes":
            def __iter__(self):
                return iter([list(p) for p in self._tries()])
        else:
            def __iter__(self):
                yield from self._tries()

        def au_dessus(self, seuil):
            if defaut == "au_dessus_liste":
                return [(c, s) for c, s in self._tries() if s >= seuil]
            if defaut == "au_dessus_partage":
                nouveau = type(self)()
                nouveau._scores = self._scores
                for client_id in [c for c, s in self._scores.items() if s < seuil]:
                    del self._scores[client_id]
                return nouveau
            garde = (lambda s: s > seuil) if defaut == "seuil_strict" else (lambda s: s >= seuil)
            return type(self)((c, s) for c, s in self._tries() if garde(s))

        def __repr__(self):
            n = len(self)
            if defaut == "repr_pluriel" and n:
                return f"Portefeuille({n} clients)"
            return f"Portefeuille({'vide' if n == 0 else f'{n} client' + ('s' if n > 1 else '')})"

    return Portefeuille


@pytest.mark.parametrize("mutant,mots", [
    (_portefeuille_17("sans_contains"), ("Écrivez __contains__",)),
    (_portefeuille_17("getitem_none"), ("portefeuille[\"C999\"]", "KeyError")),
    (_portefeuille_17("croissant"), ("du plus risqué au moins risqué",)),
    (_portefeuille_17("listes"), ("couple (tuple)",)),
    (_portefeuille_17("iter_self"), ("DEUXIÈME parcours", "NOUVEL itérateur")),
    (_portefeuille_17("iter_self_reset"), ("partagent le même itérateur", "un neuf")),
    (_portefeuille_17("iter_vue"), ("pendant un parcours", "Parcourez une copie")),
    (_portefeuille_17("doublon_remplace"), ("on ne remplace pas",)),
    (_portefeuille_17("au_dessus_liste"), ("renvoyez un Portefeuille",)),
    (_portefeuille_17("au_dessus_partage"), ("indépendants",)),
    (_portefeuille_17("seuil_strict"), ("le seuil est inclus",)),
    (_portefeuille_17("repr_pluriel"), ("Portefeuille(1 client)",)),
    (_portefeuille_17("liste_lente"), ("O(n)", "dictionnaire")),
])
def test_serie_17_portefeuille_erreurs_classiques(mutant, mots):
    _echec_17("Portefeuille", mutant, *mots)


# Défi 109 : opérateurs


def _regle_17(defaut=None):
    class Regle:
        def __init__(self, nom, condition):
            if (not isinstance(nom, str) or not nom.strip()) and defaut != "nom_vide":
                raise ValueError("nom")
            if not callable(condition):
                raise TypeError("condition")
            self.nom, self._condition, self._et = nom, condition, []

        def __call__(self, client):
            resultat = self._condition(client)
            if defaut == "modifie":
                resultat = resultat and all(r(client) for r in self._et)
            return resultat if defaut == "brut" else bool(resultat)

        def __and__(self, autre):
            if not isinstance(autre, Regle) and not (defaut == "accepte_tout" and callable(autre)):
                return NotImplemented
            if defaut == "modifie":
                self._et.append(autre)
                self.nom = f"({self.nom} & {autre})"
                return self
            if defaut == "sans_court_circuit":
                return Regle(f"({self} & {autre})", lambda c: self(c) & autre(c))
            if defaut == "double":
                return Regle(f"({self} & {autre})", lambda c: bool(self(c)) and autre(c) if self(c) else False)
            texte = f"{self} & {autre}" if defaut == "parentheses" else f"({self} & {autre})"
            return Regle(texte, lambda c: self(c) and autre(c))

        def __or__(self, autre):
            if not isinstance(autre, Regle) and not (defaut == "accepte_tout" and callable(autre)):
                return NotImplemented
            texte = f"{self} | {autre}" if defaut == "parentheses" else f"({self} | {autre})"
            return Regle(texte, lambda c: self(c) or autre(c))

        def __invert__(self):
            return Regle(f"~{self}", lambda c: not self(c))

        def __str__(self):
            return self.nom

        if defaut != "sans_bool":
            def __bool__(self):
                raise TypeError("valeur de vérité ambiguë" if defaut == "bool_message" else "utilisez & | ~")

    return Regle


@pytest.mark.parametrize("mutant,mots", [
    (_regle_17("modifie"), ("NOUVELLE Regle",)),
    (_regle_17("sans_court_circuit"), ("`a(c) & b(c)` évalue toujours les deux",)),
    (_regle_17("double"), ("évalué une règle 2 fois",)),
    (_regle_17("brut"), ("VRAI booléen",)),
    (_regle_17("sans_bool"), ("bool(mensuel)", "Écrivez __bool__")),
    (_regle_17("bool_message"), ("doit dire d'utiliser & | ~",)),
    (_regle_17("parentheses"), ("'mensuel & récent'", "une paire de parenthèses")),
    (_regle_17("accepte_tout"), ("mensuel | (lambda c: True)", "NotImplemented")),
    (_regle_17("nom_vide"), ("Regle('', ...)",)),
])
def test_serie_17_regle_erreurs_classiques(mutant, mots):
    _echec_17("Regle", mutant, *mots)


def test_serie_17_regle_en_sous_classes_acceptee():
    """Une autre écriture juste : une sous-classe par opérateur, au lieu d'une lambda."""
    from defis.corrige_serie_17 import Regle as Base

    class Regle(Base):
        def __and__(self, autre):
            return Et(self, autre) if isinstance(autre, Regle) else NotImplemented

        def __or__(self, autre):
            return Ou(self, autre) if isinstance(autre, Regle) else NotImplemented

        def __invert__(self):
            return Non(self)

    class Et(Regle):
        def __init__(self, a, b):
            super().__init__(f"({a} & {b})", lambda c: a(c) and b(c))

    class Ou(Regle):
        def __init__(self, a, b):
            super().__init__(f"({a} | {b})", lambda c: a(c) or b(c))

    class Non(Regle):
        def __init__(self, a):
            super().__init__(f"~{a}", lambda c: not a(c))

    resultat = _evaluer("Regle", Regle)
    assert resultat.statut == REUSSI, resultat.message


@pytest.mark.parametrize("nom,fabrique", [("Client", _client_17), ("Contrat", _contrat_17), ("Montant", _montant_17),
                                          ("campagne", _campagne_17), ("Portefeuille", _portefeuille_17), ("Regle", _regle_17)])
def test_serie_17_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Garde-fou : chaque version boguée ci-dessus ne diffère du juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_serie_17_les_tests_du_corrige_passent():
    import defis.corrige_serie_17 as corrige

    tests = [f for nom, f in vars(corrige).items() if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_17_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.conception as conception
    import defis.serie_17 as eleve

    for nom in ("CLIENTS_EXEMPLE", "CONTRATS", "SCORES_EXEMPLE", "Lot", "Passerelle", "PasserelleIndisponible"):
        assert getattr(eleve, nom) is getattr(conception, nom), nom


def test_serie_17_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    import defis.serie_17 as eleve
    from defis.checks_serie_17 import SERIE

    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert list(eleve.Contrat) == []   # les membres sont à écrire


def test_serie_17_les_exemples_des_docstrings_sont_justes():
    """Les exemples de defis/serie_17.py, rejoués avec le corrigé."""
    from dataclasses import FrozenInstanceError

    from defis.corrige_serie_17 import Client, Contrat, Montant, Portefeuille, Regle, campagne

    client = Client("C001", 3, 49.9, options=["tv"])
    assert repr(Client("C001", 3, 49.9)) == ("Client(client_id='C001', anciennete_mois=3, facture_mensuelle=49.9, "
                                             "contrat='Mensuel', options=())")
    with pytest.raises(ValueError, match="^anciennete_mois ne peut pas être négative : -3$"):
        Client("C001", -3, 49.9)
    assert client.options == ("tv",)
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'anciennete_mois'"):
        client.anciennete_mois = 4
    assert client == Client("C001", 3, 49.9, "Mensuel", ("tv",))

    assert repr(Contrat.depuis_texte(" mensuel ")) == "<Contrat.MENSUEL: 'Mensuel'>"
    with pytest.raises(ValueError, match=re.escape("contrat inconnu : 'Bi-annuel' (attendu : Mensuel, Annuel, Biannuel)")):
        Contrat.depuis_texte("Bi-annuel")
    assert (Contrat.ANNUEL.engagement_mois, f"Contrat {Contrat.ANNUEL}") == (12, "Contrat Annuel")
    assert [c.value for c in Contrat] == ["Mensuel", "Annuel", "Biannuel"]

    assert repr(Montant.depuis_euros("1 234,5")) == "Montant(centimes=123450, devise='EUR')"
    assert " ".join(map(str, (sum([Montant(1999), Montant(1)]), Montant(-5), 3 * Montant(1999, "USD")))) == \
        "20,00 € -0,05 € 59,97 USD"
    with pytest.raises(ValueError, match="^impossible de comparer USD et EUR sans taux de change$"):
        Montant(100, "USD") < Montant(200)

    passerelle = Passerelle()
    with campagne(passerelle, "Fidélité") as lot:
        lot.ajouter("C001")
        lot.ajouter("C002")
    assert passerelle.journal == ["ouvrir Fidélité", "ajouter C001", "ajouter C002", "valider", "fermer"]
    assert passerelle.envoyes == ["C001", "C002"]

    p = Portefeuille(SCORES_17.items())
    assert (repr(p), len(p), "C002" in p, p["C002"]) == ("Portefeuille(5 clients)", 5, True, 0.15)
    assert list(p)[:3] == [("C004", 0.91), ("C001", 0.82), ("C003", 0.47)]
    assert repr(p.au_dessus(0.8)) == "Portefeuille(2 clients)"

    mensuel = Regle("mensuel", lambda c: c["contrat"] == "Mensuel")
    recent = Regle("récent", lambda c: c["anciennete_mois"] < 6)
    mecontent = Regle("mécontent", lambda c: c["appels_support"] >= 3)
    cible = mensuel & (recent | mecontent)
    assert (str(cible), str(~cible)) == ("(mensuel & (récent | mécontent))", "~(mensuel & (récent | mécontent))")
    assert [c["client_id"] for c in CLIENTS_17 if cible(c)] == ["C001", "C002"]
    with pytest.raises(TypeError, match=re.escape("la règle 'mensuel' n'est ni vraie ni fausse : combinez avec & | ~ "
                                                  "(pas and, or, not)")):
        mensuel and recent  # noqa: B018


def test_serie_17_la_passerelle_refuse_une_panne_inconnue():
    with pytest.raises(ValueError):
        Passerelle(panne="fermer")
    with pytest.raises(PasserelleIndisponible):
        Passerelle(panne="ouvrir").ouvrir("X")


def test_un_fichier_qui_ne_se_charge_pas_n_empeche_pas_les_autres_series(tmp_path, monkeypatch):
    """Une erreur au chargement (syntaxe, ou `options: list = []` dans une dataclass) ne doit pas tout faire planter :
    la série concernée s'affiche en 💥 avec la ligne fautive, et les autres séries restent corrigées."""
    import dataclasses

    from defis.checks_serie_17 import SERIE

    (tmp_path / "eleve_casse.py").write_text(
        "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Client:\n    options: list = []\n",
        encoding="utf-8")
    (tmp_path / "eleve_syntaxe.py").write_text("def f(:\n    pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for module, mots in (("eleve_casse", ("ValueError", "default_factory", "(ligne ")),
                         ("eleve_syntaxe", ("SyntaxError", "(ligne 1)"))):
        resultats = evaluer(dataclasses.replace(SERIE, module_eleve=module))
        assert [r.statut for r in resultats] == [ERREUR] * len(SERIE.defis)
        for mot in (f"{module}.py ne se charge pas", *mots):
            assert mot in resultats[0].message, resultats[0].message
    assert all(r.statut == A_FAIRE for r in evaluer(SERIE))  # le vrai fichier élève, lui, se charge toujours


# --------------------------------------------------------------------------- Série 18 : rendre le code rapide

import bisect  # noqa: E402
import cProfile  # noqa: E402
import csv  # noqa: E402
import functools  # noqa: E402
import pstats  # noqa: E402
import statistics  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from datetime import date as d18  # noqa: E402

from defis.performance import HorlogeFactice as Horloge18  # noqa: E402
from defis.performance import TARIFS_EXEMPLE as TARIFS_18  # noqa: E402
from defis.performance import clients_a_rappeler_lent, ecrire_factures, exemple_rappels  # noqa: E402


def _echec_18(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


# Défi 110 : chronométrer


def _chronometrer_18(defaut=None):
    def chronometrer(fonction, repetitions=5, horloge=time.time if defaut == "time_time" else time.perf_counter):
        if repetitions < 1 and defaut != "sans_validation":
            raise ValueError(f"repetitions doit valoir au moins 1 (reçu : {repetitions})")
        lire = time.perf_counter if defaut == "horloge_reelle" else horloge
        durees = []
        for _ in range(1 if defaut == "premiere" else repetitions):
            debut = lire()
            try:
                fonction()
            except Exception:
                if defaut != "avale":
                    raise
            durees.append(lire() - debut)
        if defaut == "moyenne":
            return sum(durees) / len(durees)
        if defaut == "total":
            return sum(durees)
        if defaut == "max":
            return max(durees)
        return min(durees)

    return chronometrer


@pytest.mark.parametrize("mutant,mots", [
    (_chronometrer_18("moyenne"), ("MOYENNE", "min(durees)")),
    (_chronometrer_18("total"), ("TOTALE",)),
    (_chronometrer_18("max"), ("plus LONGUE",)),
    (_chronometrer_18("premiere"), ("appelée 1 fois au lieu de 5",)),
    (_chronometrer_18("horloge_reelle"), ("vraie horloge", "horloge()")),
    (_chronometrer_18("avale"), ("devrait lever Panne", "avale")),
    (_chronometrer_18("sans_validation"), ("doit nommer `repetitions`", "min() arg is an empty sequence")),
    (_chronometrer_18("time_time"), ("time.time par défaut", "perf_counter")),
])
def test_serie_18_chronometrer_erreurs_classiques(mutant, mots):
    _echec_18("chronometrer", mutant, *mots)


def test_serie_18_chronometrer_qui_lit_l_horloge_une_fois_par_tour_est_accepte():
    def chronometrer(fonction, repetitions=5, horloge=time.perf_counter):
        if repetitions < 1:
            raise ValueError("repetitions doit valoir au moins 1")
        durees, avant = [], horloge()
        for _ in range(repetitions):
            fonction()
            apres = horloge()
            durees.append(apres - avant)
            avant = apres
        return min(durees)

    assert _evaluer("chronometrer", chronometrer).statut == REUSSI


# Défi 111 : profiler


def _point_chaud_18(defaut=None):
    def point_chaud(programme):
        if defaut == "deux_fois":
            programme()
        profileur = cProfile.Profile()
        if defaut == "sans_finally":
            profileur.enable()
            programme()
            profileur.disable()
        elif defaut == "avale":
            try:
                profileur.runcall(programme)
            except Exception:
                return "erreur"
        else:
            profileur.runcall(programme)
        stats = pstats.Stats(profileur).stats
        colonne = {"cumtime": 3, "ncalls": 1}.get(defaut, 2)
        cle = max(stats, key=lambda c: stats[c][colonne])
        return cle if defaut == "cle" else cle[2]

    return point_chaud


@pytest.mark.parametrize("mutant,mots", [
    (_point_chaud_18("cumtime"), ("« programme_rapport »", "CUMULÉ", "tottime")),
    (_point_chaud_18("ncalls"), ("« nettoyer_ligne »", "APPELÉE", "ncalls")),
    (_point_chaud_18("cle"), ("seulement le NOM", "élément 2")),
    (_point_chaud_18("deux_fois"), ("2 fois",)),
    (_point_chaud_18("avale"), ("devrait lever RuntimeError",)),
    (_point_chaud_18("sans_finally"), ("resté ACTIF", "finally")),
])
def test_serie_18_point_chaud_erreurs_classiques(mutant, mots):
    _echec_18("point_chaud", mutant, *mots)


def test_serie_18_un_profileur_oublie_est_arrete_par_le_correcteur():
    """Sinon tout le reste des tests serait profilé (et, depuis Python 3.12, plus aucun profileur ne démarrerait)."""
    from defis.checks_serie_18 import _profileurs_actifs

    _echec_18("point_chaud", _point_chaud_18("sans_finally"), "resté ACTIF")
    assert not _profileurs_actifs()
    assert _evaluer("point_chaud", _point_chaud_18()).statut == REUSSI


def test_serie_18_point_chaud_avec_sort_stats_est_accepte():
    def point_chaud(programme):
        profileur = cProfile.Profile()
        profileur.enable()
        try:
            programme()
        finally:
            profileur.disable()
        stats = pstats.Stats(profileur).sort_stats("tottime")
        return stats.fcn_list[0][2]

    assert _evaluer("point_chaud", point_chaud).statut == REUSSI


# Défi 112 : bisect


def _prix_18(defaut=None):
    def prix_en_vigueur(changements, dates):
        if not changements:
            raise ValueError("historique vide")
        if defaut == "en_place":
            changements.sort()
            historique = changements
        else:
            historique = sorted(changements, key=lambda c: c[0])
        jours = [j for j, _ in historique]
        prix = [p for _, p in historique]
        if defaut != "sans_doublon" and len(set(jours)) != len(jours):
            raise ValueError("deux tarifs le même jour")
        resultat = []
        for jour in (sorted(dates) if defaut == "ordre" else dates):
            if defaut == "naif":
                i = sum(1 for j in jours if j <= jour)
            elif defaut == "left":
                i = bisect.bisect_left(jours, jour)
            else:
                i = bisect.bisect_right(jours, jour)
            if i == 0 and defaut != "moins_un":
                raise ValueError("date trop ancienne" if defaut == "sans_date" else f"aucun tarif le {jour}")
            resultat.append(prix[i - 1])
        return resultat

    return prix_en_vigueur


@pytest.mark.parametrize("mutant,mots", [
    (_prix_18("left"), ("ANCIEN prix", "bisect_right")),
    (_prix_18("moins_un"), ("devrait lever ValueError", "l'indice -1 est le DERNIER")),
    (_prix_18("sans_date"), ("2023-12-31",)),
    (_prix_18("en_place"), ("MODIFIÉ", "sorted(liste)")),
    (_prix_18("sans_doublon"), ("le 1er juin 2024", "devrait lever ValueError")),
    (_prix_18("ordre"), ("dans l'ordre de `dates`",)),
    (_prix_18("naif"), ("1 500 000 comparaisons", "le correcteur l'a arrêtée", "bisect.bisect_right")),
])
def test_serie_18_prix_erreurs_classiques(mutant, mots):
    _echec_18("prix_en_vigueur", mutant, *mots)


def test_serie_18_prix_avec_des_ordinaux_et_bisect_key_est_accepte():
    def prix_en_vigueur(changements, dates):
        if not changements:
            raise ValueError("vide")
        historique = sorted((j.toordinal(), p) for j, p in changements)
        if any(a[0] == b[0] for a, b in zip(historique, historique[1:])):
            raise ValueError("doublon")
        resultat = []
        for jour in dates:
            i = bisect.bisect_right(historique, jour.toordinal(), key=lambda c: c[0])
            if i == 0:
                raise ValueError(f"pas de tarif le {jour.isoformat()}")
            resultat.append(historique[i - 1][1])
        return resultat

    assert _evaluer("prix_en_vigueur", prix_en_vigueur).statut == REUSSI


def test_serie_18_les_dates_espionnes_comptent_et_se_comparent_comme_des_dates():
    from defis.checks_serie_18 import _Compteur, _Jour

    _Jour.compteur = compteur = _Compteur()
    try:
        a, b = _Jour(2024, 1, 1), _Jour(2024, 7, 1)
        assert a < b and b >= a and a != b and d18(2024, 1, 1) == a and sorted([b, a]) == [a, b]
        assert bisect.bisect_right([a, b], _Jour(2024, 7, 1)) == 2
        assert compteur.total >= 6 and len({a, d18(2024, 1, 1)}) == 1
    finally:
        _Jour.compteur = None


# Défi 113 : cache borné et périmé


def _cache_18(defaut=None):
    def cache_lru(taille_max, duree, horloge=time.monotonic):
        if defaut != "validation" and (taille_max < 1 or duree <= 0):
            raise ValueError("taille_max et duree doivent être positifs")
        partage = OrderedDict()

        def decorateur(fonction):
            memoire = partage if defaut == "partage" else OrderedDict()
            valeurs, ordre = {}, []
            compteurs = {"calculs": 0, "reutilisations": 0}

            @functools.wraps(fonction)
            def enveloppe(*args):
                maintenant = time.monotonic() if defaut == "horloge" else horloge()
                if defaut == "fuite":
                    ordre.append(args)
                present = bool(valeurs.get(args)) if defaut == "get" else args in memoire
                if present:
                    calcule_a, resultat = memoire[args]
                    age = maintenant - calcule_a
                    if (age <= duree) if defaut == "strict" else (age < duree):
                        if defaut != "fifo":
                            memoire.move_to_end(args)
                        if defaut == "rajeunit":
                            memoire[args] = (maintenant, resultat)
                        compteurs["reutilisations"] += 1
                        return resultat
                    if defaut != "evince":
                        del memoire[args]
                if defaut == "evince" and len(memoire) >= taille_max:
                    valeurs.pop(memoire.popitem(last=False)[0], None)
                try:
                    resultat = fonction(*args)
                except Exception:
                    if defaut == "exception":
                        memoire[args] = (maintenant, None)
                        return None
                    raise
                compteurs["calculs"] += 1
                memoire[args] = (maintenant, resultat)
                valeurs[args] = resultat
                memoire.move_to_end(args)
                if defaut not in ("infini", "evince") and len(memoire) > taille_max:
                    valeurs.pop(memoire.popitem(last=False)[0], None)
                return resultat

            enveloppe.cache_info = lambda: {**compteurs, "taille": len(memoire)}
            if defaut == "wraps":
                del enveloppe.__wrapped__
                enveloppe.__name__, enveloppe.__doc__ = "enveloppe", None
            return enveloppe

        return decorateur

    return cache_lru


@pytest.mark.parametrize("mutant,mots", [
    (_cache_18("validation"), ("cache_lru(0, 60) devrait lever ValueError",)),
    (_cache_18("partage"), ("MÊME `memo", "DANS decorateur(fonction)")),
    (_cache_18("horloge"), ("horloge(), le paramètre",)),
    (_cache_18("fuite"), ("retient encore", "`ordre` qui ne fait que grandir")),
    (_cache_18("get"), ("None", "if cle in memoire")),
    (_cache_18("strict"), ("60 secondes PILE", "âge >= duree")),
    (_cache_18("fifo"), ("moins récemment UTILISÉ", "move_to_end")),
    (_cache_18("rajeunit"), ("ne le rajeunit pas",)),
    (_cache_18("evince"), ("B (calculé il y a 52 s", "DÉPASSE taille_max")),
    (_cache_18("exception"), ("devrait lever Panne",)),
    (_cache_18("infini"), ("'taille': 3",)),
    (_cache_18("wraps"), ("functools.wraps",)),
])
def test_serie_18_cache_erreurs_classiques(mutant, mots):
    _echec_18("cache_lru", mutant, *mots)


def test_serie_18_cache_avec_un_dict_ordinaire_est_accepte():
    """Depuis Python 3.7, un dict garde l'ordre d'insertion : pop + réinsertion fait le même travail qu'OrderedDict."""
    def cache_lru(taille_max, duree, horloge=time.monotonic):
        if taille_max < 1 or duree <= 0:
            raise ValueError("taille_max >= 1 et duree > 0")

        def decorateur(fonction):
            memoire, stats = {}, {"calculs": 0, "reutilisations": 0}

            @functools.wraps(fonction)
            def enveloppe(*args):
                maintenant = horloge()
                if args in memoire:
                    instant, valeur = memoire.pop(args)
                    if maintenant - instant < duree:
                        memoire[args] = (instant, valeur)
                        stats["reutilisations"] += 1
                        return valeur
                valeur = fonction(*args)
                stats["calculs"] += 1
                memoire[args] = (maintenant, valeur)
                if len(memoire) > taille_max:
                    del memoire[next(iter(memoire))]
                return valeur

            enveloppe.cache_info = lambda: dict(stats, taille=len(memoire))
            return enveloppe

        return decorateur

    assert _evaluer("cache_lru", cache_lru).statut == REUSSI


# Défi 114 : lire en flux


def _totaux_18(defaut=None):
    def totaux_par_contrat(chemin):
        if defaut == "tout_pandas":
            df = pd.read_csv(chemin).dropna(subset=["montant"])
            groupes = df.groupby("contrat")["montant"]
            return {c: (int(n), round(float(s), 2)) for c, n, s in zip(groupes.count().index, groupes.count(), groupes.sum())}
        nombres, sommes = {}, {}

        def ajouter(contrat, montant):
            if montant or defaut == "annulees":
                nombres[contrat] = nombres.get(contrat, 0) + 1
                sommes[contrat] = sommes.get(contrat, 0.0) + float(montant or 0)

        if defaut == "split":
            with open(chemin, encoding="utf-8") as fichier:
                entete = next(fichier).strip().split(",")
                for ligne in fichier:
                    if ligne.strip():
                        champs = dict(zip(entete, ligne.strip().split(",")))
                        ajouter(champs["contrat"], champs["montant"])
        else:
            fichier = open(chemin, newline="", encoding="utf-8")
            lignes = fichier.readlines() if defaut == "readlines" else fichier
            if defaut == "position":
                lecteur = csv.reader(lignes)
                next(lecteur)
                for champs in lecteur:
                    if champs:
                        ajouter(champs[1], champs[3])
            else:
                for ligne in csv.DictReader(lignes):
                    ajouter(ligne["contrat"], ligne["montant"])
            if defaut != "ouvert":
                fichier.close()
        return {c: (nombres[c], round(sommes[c], 2)) for c in nombres}

    return totaux_par_contrat


@pytest.mark.parametrize("mutant,mots", [
    (_totaux_18("tout_pandas"), ("La mémoire grandit avec le fichier", "chunksize")),
    (_totaux_18("readlines"), ("La mémoire grandit avec le fichier",)),
    (_totaux_18("split"), ("lève ValueError", "module csv")),
    (_totaux_18("position"), ("autre ordre", "par leur NOM")),
    (_totaux_18("ouvert"), ("OUVERT", "with open")),
    (_totaux_18("annulees"), ("factures annulées", "size()")),
])
def test_serie_18_totaux_erreurs_classiques(mutant, mots):
    _echec_18("totaux_par_contrat", mutant, *mots)


def test_serie_18_totaux_avec_pandas_par_paquets_est_accepte():
    def totaux_par_contrat(chemin):
        nombres, sommes = Counter(), Counter()
        for paquet in pd.read_csv(chemin, chunksize=10_000, usecols=["contrat", "montant"]):
            paquet = paquet.dropna(subset=["montant"])
            for contrat, montants in paquet.groupby("contrat")["montant"]:
                nombres[contrat] += len(montants)
                sommes[contrat] += float(montants.sum())
        return {c: (nombres[c], round(sommes[c], 2)) for c in nombres}

    resultat = _evaluer("totaux_par_contrat", totaux_par_contrat)
    assert resultat.statut == REUSSI, resultat.message


def test_serie_18_le_fichier_de_factures_a_tous_les_pieges_annonces(tmp_path):
    chemin = ecrire_factures(tmp_path / "f.csv", lignes=2_000, graine=1)
    texte = chemin.read_text(encoding="utf-8")
    assert texte.startswith("client_id,contrat,offre,montant\n") and texte.endswith("\n\n")
    assert '"Fibre, TV"' in texte and "\r" not in texte
    lignes = list(csv.DictReader(texte.splitlines()))
    assert len(lignes) == 2_000
    assert 10 < sum(not ligne["montant"] for ligne in lignes) < 100
    autre = ecrire_factures(tmp_path / "g.csv", lignes=2_000, graine=1, colonnes=("montant", "contrat", "offre", "client_id"))
    assert [dict(sorted(d.items())) for d in csv.DictReader(autre.read_text(encoding="utf-8").splitlines())] == \
        [dict(sorted(d.items())) for d in lignes]


# Défi 115 : optimiser sans rien changer


def _rappel_18(defaut=None):
    def clients_a_rappeler(clients, appels, deja_contactes, seuil_appels=3):
        if not clients and defaut != "vide":
            return []
        if defaut == "dict":
            nb = {}
            for appel in appels:
                if appel["motif"] != "information":
                    nb[appel["client_id"]] = nb.get(appel["client_id"], 0) + 1
        else:
            nb = Counter(a["client_id"] for a in appels if defaut == "information" or a["motif"] != "information")
        if defaut == "mean":
            moyenne = statistics.mean(c["facture"] for c in clients)
        elif defaut == "moyenne_filtree":
            gardes = [c for c in clients if c["client_id"] not in set(deja_contactes)]
            moyenne = sum(c["facture"] for c in gardes) / len(gardes) if gardes else 0
        else:
            moyenne = sum(c["facture"] for c in clients) / len(clients)
        exclus = deja_contactes if defaut == "lent_deja" else set(deja_contactes)
        resultat, vus = [], set()
        for client in clients:
            if defaut == "lent_moyenne":
                moyenne = sum(c["facture"] for c in clients) / len(clients)
            client_id = client["client_id"]
            if client_id in exclus or (defaut == "doublon" and client_id in vus):
                continue
            vus.add(client_id)
            if defaut == "lent_appels":
                n = sum(1 for a in appels if a["client_id"] == client_id and a["motif"] != "information")
            else:
                n = nb[client_id]
            assez = client["facture"] > moyenne if defaut == "strict" else client["facture"] >= moyenne
            if n >= seuil_appels and assez:
                resultat.append([client_id, n] if defaut == "listes" else (client_id, n))
        if defaut == "tri_id":
            resultat.sort(key=lambda paire: (-paire[1], paire[0]))
        else:
            resultat.sort(key=lambda paire: paire[1], reverse=True)
        if defaut == "modifie":
            deja_contactes.clear()
        return resultat

    return clients_a_rappeler


@pytest.mark.parametrize("mutant,mots", [
    (_rappel_18("vide"), ("divise par zéro",)),
    (_rappel_18("dict"), ("KeyError", "nb.get(client_id, 0)")),
    (_rappel_18("information"), ("« information » ne comptent pas",)),
    (_rappel_18("strict"), ("pile à la moyenne", "(>=)")),
    (_rappel_18("mean"), ("0.10000000000000002", "statistics.mean")),
    (_rappel_18("moyenne_filtree"), ("TOUS les clients", "66,67")),
    (_rappel_18("doublon"), ("deux fois C1",)),
    (_rappel_18("listes"), ("TUPLES",)),
    (_rappel_18("tri_id"), ("tri est stable", "Ne triez pas en plus par identifiant")),
    (_rappel_18("modifie"), ("MODIFIÉ une des listes",)),
    (_rappel_18("lent_appels"), ("le correcteur l'a arrêtée", "« appels »", "Counter")),
    (_rappel_18("lent_moyenne"), ("le correcteur l'a arrêtée", "« clients »", "avant la boucle")),
    (_rappel_18("lent_deja"), ("le correcteur l'a arrêtée", "« deja_contactes »", "set")),
    (clients_a_rappeler_lent, ("le correcteur l'a arrêtée",)),
])
def test_serie_18_rappel_erreurs_classiques(mutant, mots):
    _echec_18("clients_a_rappeler", mutant, *mots)


def test_serie_18_un_except_exception_n_arrete_pas_le_correcteur():
    """Le budget dépassé est une BaseException : un `except Exception` ne peut pas l'avaler."""
    def clients_a_rappeler(clients, appels, deja_contactes, seuil_appels=3):
        try:
            return clients_a_rappeler_lent(clients, appels, deja_contactes, seuil_appels)
        except Exception:
            return []

    _echec_18("clients_a_rappeler", clients_a_rappeler, "le correcteur l'a arrêtée")


def test_serie_18_la_reference_rapide_du_correcteur_est_juste():
    from defis.checks_serie_18 import _CAS_LIMITES, _reference_rapide, _tirer_cas

    for _, clients, appels, deja, seuil, _ in _CAS_LIMITES:
        assert _reference_rapide(clients, appels, deja, seuil) == clients_a_rappeler_lent(clients, appels, deja, seuil)
    rng = random.Random(2026)
    for _ in range(1_000):
        clients, appels, deja, seuil = _tirer_cas(rng)
        assert _reference_rapide(clients, appels, deja, seuil) == clients_a_rappeler_lent(clients, appels, deja, seuil)
    clients, appels, deja = exemple_rappels(n_clients=300, graine=4)
    assert _reference_rapide(clients, appels, deja) == clients_a_rappeler_lent(clients, appels, deja)


def test_serie_18_les_listes_espionnes_se_comportent_comme_des_listes():
    from defis.checks_serie_18 import _BudgetDepasse, _Compteur, _espionner

    compteur = _Compteur(budget=50)
    lignes = _espionner(compteur, "appels", [{"client_id": "C1", "motif": "panne"}] * 3, True)
    assert lignes == [{"client_id": "C1", "motif": "panne"}] * 3 and len(lignes) == 3
    assert [ligne["motif"] for ligne in lignes] == ["panne"] * 3
    assert compteur.par_categorie == {"appels": 6}
    ids = _espionner(compteur, "deja_contactes", ["C1", "C2"], False)
    assert "C2" in ids and set(ids) == {"C1", "C2"} and compteur.par_categorie["deja_contactes"] == 4
    with pytest.raises(_BudgetDepasse):
        for _ in range(30):
            "C3" in ids  # noqa: B015
    assert not issubclass(_BudgetDepasse, Exception)


# Toute la série


@pytest.mark.parametrize("nom,fabrique", [("chronometrer", _chronometrer_18), ("point_chaud", _point_chaud_18),
                                          ("prix_en_vigueur", _prix_18), ("cache_lru", _cache_18),
                                          ("totaux_par_contrat", _totaux_18), ("clients_a_rappeler", _rappel_18)])
def test_serie_18_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Garde-fou : chaque version boguée ci-dessus ne diffère du juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_serie_18_les_tests_du_corrige_passent():
    import defis.corrige_serie_18 as corrige

    tests = [f for nom, f in vars(corrige).items() if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_18_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.performance as performance
    import defis.serie_18 as eleve

    for nom in ("CONTRATS", "TARIFS_EXEMPLE", "HorlogeFactice", "clients_a_rappeler_lent", "ecrire_factures",
                "exemple_rappels", "programme_exemple"):
        assert getattr(eleve, nom) is getattr(performance, nom), nom


def test_serie_18_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    from defis.checks_serie_18 import SERIE

    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6


def test_serie_18_les_exemples_des_docstrings_sont_justes(tmp_path, capsys):
    """Les exemples de defis/serie_18.py, rejoués avec le corrigé."""
    from defis.corrige_serie_18 import (cache_lru, chronometrer, clients_a_rappeler, point_chaud, prix_en_vigueur,
                                        programme_exemple, totaux_par_contrat)

    assert point_chaud(programme_exemple) == "calculer_scores"
    assert 0 < chronometrer(programme_exemple) < 1

    assert repr(TARIFS_18) == ("[(datetime.date(2025, 1, 1), 34.9), (datetime.date(2024, 1, 1), 29.9), "
                               "(datetime.date(2024, 7, 1), 32.9)]")
    assert prix_en_vigueur(TARIFS_18, [d18(2024, 3, 15), d18(2024, 7, 1), d18(2026, 1, 1)]) == [29.9, 32.9, 34.9]

    horloge = Horloge18()

    @cache_lru(taille_max=2, duree=60, horloge=horloge)
    def score(client_id):
        print("calcul", client_id)
        return 0.5

    assert [score(c) for c in ("C1", "C1", "C2")] == [0.5] * 3
    assert capsys.readouterr().out == "calcul C1\ncalcul C2\n"
    assert [score(c) for c in ("C1", "C3")] == [0.5] * 2
    assert capsys.readouterr().out == "calcul C3\n"
    assert score.cache_info() == {"calculs": 3, "reutilisations": 2, "taille": 2}
    horloge.avancer(60)
    assert score("C1") == 0.5 and capsys.readouterr().out == "calcul C1\n"

    chemin = ecrire_factures(tmp_path / "factures.csv")
    assert round(chemin.stat().st_size / 1e6, 1) == 2.8
    assert repr(totaux_par_contrat(chemin)) == ("{'Biannuel': (19512, 1413713.44), 'Mensuel': (54010, 3914899.63), "
                                                "'Annuel': (24509, 1772371.59)}")

    clients, appels, deja = exemple_rappels()
    assert (len(clients), len(appels)) == (2_000, 20_000)
    assert clients_a_rappeler(clients, appels, deja) == clients_a_rappeler_lent(clients, appels, deja)


# --------------------------------------------------------------------------- Série 19 : faire évoluer du code existant

import copy  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import operator  # noqa: E402
import warnings  # noqa: E402
from datetime import date as d19  # noqa: E402

from defis.heritage import CANAUX as CANAUX_19  # noqa: E402
from defis.heritage import CLIENTS_EXEMPLE as CLIENTS_19  # noqa: E402
from defis.heritage import JOUR_EXEMPLE as JOUR_19  # noqa: E402
from defis.heritage import LIGNES_EXEMPLE as LIGNES_19  # noqa: E402
from defis.heritage import ClientMessagerie, ErreurMessagerie, ancien_score, nouveau_score, remise_ancienne  # noqa: E402


def _echec_19(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


# Défi 116 : photographier


def _photographier_19(defaut=None):
    def photographier(fonction, cas):
        if defaut == "deux_parcours":
            nombre = len(list(cas))  # noqa: F841 — vide le générateur avant la boucle
        photo = []
        for arguments in cas:
            avant = copy.deepcopy(arguments)
            if defaut == "copie_superficielle":
                appel = copy.copy(arguments)
            elif defaut == "sans_copie":
                appel = arguments
            else:
                appel = copy.deepcopy(arguments)
            if defaut == "photo_apres":
                avant = appel
            try:
                if defaut == "deux_appels":
                    fonction(*copy.deepcopy(arguments))
                valeur = fonction(*appel)
            except (BaseException if defaut == "tout_attraper" else Exception) as exc:
                if defaut == "laisse_remonter":
                    raise
                if defaut == "objet_exception":
                    note = ("erreur", exc)
                elif defaut == "classe":
                    note = ("erreur", type(exc), str(exc))
                else:
                    note = ("erreur", type(exc).__name__, str(exc))
            else:
                note = ("ok", valeur if defaut == "valeur_sans_copie" else copy.deepcopy(valeur))
                if defaut == "valeur_seule":
                    note = valeur
            photo.append([avant, note] if defaut == "listes" else (avant, note))
        return photo

    return photographier


@pytest.mark.parametrize("mutant,mots", [
    (_photographier_19("valeur_seule"), ('("ok", valeur)',)),
    (_photographier_19("listes"), ("TUPLE",)),
    (_photographier_19("laisse_remonter"), ("on la NOTE",)),
    (_photographier_19("objet_exception"), ("OBJET exception", "type(exc).__name__")),
    (_photographier_19("classe"), ("NOM du type",)),
    (_photographier_19("copie_superficielle"), ("vos cas ont été modifiés", "copy.deepcopy(arguments)")),
    (_photographier_19("sans_copie"), ("vos cas ont été modifiés",)),
    (_photographier_19("photo_apres"), ("APRÈS l'appel",)),
    (_photographier_19("valeur_sans_copie"), ("copy.deepcopy(valeur)",)),
    (_photographier_19("deux_appels"), ("une seule fois par cas",)),
    (_photographier_19("deux_parcours"), ("GÉNÉRATEUR",)),
    (_photographier_19("tout_attraper"), ("Ctrl+C", "Exception")),
])
def test_serie_19_photographier_erreurs_classiques(mutant, mots):
    _echec_19("photographier", mutant, *mots)


def test_serie_19_photographier_qui_garde_les_arguments_d_origine_est_accepte():
    def photographier(fonction, cas):
        photo = []
        for arguments in cas:
            try:
                note = ("ok", copy.deepcopy(fonction(*copy.deepcopy(arguments))))
            except Exception as exc:
                note = ("erreur", type(exc).__name__, str(exc))
            photo.append((arguments, note))       # la fonction n'a touché qu'une copie : l'original est intact
        return photo

    assert _evaluer("photographier", photographier).statut == REUSSI


# Défi 117 : remise


def _remise_19(defaut=None):
    def calcul(montant, taux, pourcentage):
        taux_donne = bool(taux) if defaut == "si_taux" else taux is not None
        pourcentage_donne = bool(pourcentage) if defaut == "si_pourcentage" else pourcentage is not None
        if taux_donne and pourcentage_donne:
            if defaut != "pourcentage_gagne":
                raise TypeError("remise() : donnez `pourcentage` (ou l'ancien `taux`), pas les deux")
            taux_donne = False
        if not taux_donne and not pourcentage_donne and defaut != "typeerror_muet":
            raise TypeError("remise() : il manque `pourcentage` (ou l'ancien `taux`)")
        if taux_donne:
            message = ("remise(montant, taux) est dépréciée" if defaut == "message_vague"
                       else f"remise(montant, taux) est dépréciée : écrivez remise(montant, pourcentage={taux * 100:g})")
            if defaut == "print":
                print(message)
            elif defaut != "sans_avertissement":
                categorie = UserWarning if defaut == "user_warning" else DeprecationWarning
                for _ in range(2 if defaut == "deux_avertissements" else 1):
                    warnings.warn(message, categorie, stacklevel=1 if defaut == "stacklevel_1" else 3)
            if defaut == "taux_non_valide":
                return round(montant * (1 - taux), 2)
            return remise_ancienne(montant, taux)
        if not 0 <= pourcentage <= 100 and defaut != "pourcentage_non_valide":
            raise ValueError(f"pourcentage doit être entre 0 et 100 (reçu : {pourcentage})")
        if defaut == "formule_fausse":
            return round(montant * pourcentage / 100, 2)
        return remise_ancienne(montant, pourcentage / 100)

    if defaut == "sans_etoile":
        def remise(montant, taux=None, pourcentage=None):
            return calcul(montant, taux, pourcentage)
    else:
        def remise(montant, taux=None, *, pourcentage=None):
            return calcul(montant, taux, pourcentage)
    return remise


@pytest.mark.parametrize("mutant,mots", [
    (_remise_19("sans_etoile"), ("NOMMÉ",)),
    (_remise_19("sans_avertissement"), ("ne prévient personne",)),
    (_remise_19("print"), ("print", "warnings.warn")),
    (_remise_19("user_warning"), ("UserWarning", "DeprecationWarning")),
    (_remise_19("stacklevel_1"), ("stacklevel=2", "test_defis.py")),
    (_remise_19("message_vague"), ("nommer `pourcentage`",)),
    (_remise_19("deux_avertissements"), ("un seul par appel",)),
    (_remise_19("si_taux"), ("taux is None",)),
    (_remise_19("si_pourcentage"), ("pourcentage is None",)),
    (_remise_19("taux_non_valide"), ("2 000 %",)),
    (_remise_19("pourcentage_non_valide"), ("pourcentage=120", "nommez le paramètre fautif")),
    (_remise_19("typeerror_muet"), ("nommer `taux` et `pourcentage`",)),
    (_remise_19("pourcentage_gagne"), ("les deux", "devrait lever TypeError")),
    (_remise_19("formule_fausse"), ("APRÈS remise",)),
])
def test_serie_19_remise_erreurs_classiques(mutant, mots):
    _echec_19("remise", mutant, *mots)


# Défi 118 : decider_relances


def _decider_19(defaut=None):
    deja = [0]

    def decider_relances(lignes, aujourd_hui, budget=50):
        jour = d19.today() if defaut == "today" else aujourd_hui
        if defaut == "mute":
            for ligne in lignes:
                ligne["score"] = float(str(ligne["score"]).replace(",", "."))
            candidats = [(-ligne["score"], ligne["client_id"], ligne) for ligne in lignes]
        else:
            candidats = [(-float(ligne["score"].replace(",", ".")), ligne["client_id"], ligne) for ligne in lignes]
        candidats.sort(key=(lambda c: c[0]) if defaut == "sans_tri_id" else (lambda c: (c[0], c[1])))
        if defaut == "budget_avant":
            candidats = candidats[:budget]
        relances = []
        for moins_score, client_id, ligne in candidats:
            if (-moins_score <= 0.6) if defaut == "seuil_strict" else (-moins_score < 0.6):
                continue
            consentement = ligne["consentement"].strip().lower() if defaut == "consentement_souple" else ligne["consentement"]
            if consentement != "oui":
                continue
            if ligne["dernier_contact"]:
                jours = (jour - d19.fromisoformat(ligne["dernier_contact"])).days
                if (jours <= 30) if defaut == "trente_exclus" else (jours < 30):
                    continue
            if (deja[0] if defaut == "global" else len(relances)) >= budget:
                break
            deja[0] += 1
            if defaut == "offre_par_contrat":
                offre = "FIDELITE" if ligne["contrat"] == "Annuel" else "ENGAGEMENT"
            else:
                offre = "ENGAGEMENT" if ligne["contrat"] == "Mensuel" else "FIDELITE"
            if defaut == "print":
                print(f"relance {client_id} : {offre}")
            relances.append([client_id, offre] if defaut == "listes" else (client_id, offre))
        return relances

    return decider_relances


@pytest.mark.parametrize("mutant,mots", [
    (_decider_19("mute"), ("modifié les lignes",)),
    (_decider_19("global"), ("garde un état",)),
    (_decider_19("print"), ("print",)),
    (_decider_19("today"), ("AUJOURD'HUI", "aujourd_hui")),
    (_decider_19("seuil_strict"), ("0,6 PILE",)),
    (_decider_19("trente_exclus"), ("30 jours PILE",)),
    (_decider_19("consentement_souple"), ("« Oui »", "ticket")),
    (_decider_19("sans_tri_id"), ("trie par client_id",)),
    (_decider_19("budget_avant"), ("AVANT les filtres",)),
    (_decider_19("listes"), ("LISTE de tuples",)),
    (_decider_19("offre_par_contrat"), ("Première différence",)),
])
def test_serie_19_decider_erreurs_classiques(mutant, mots):
    _echec_19("decider_relances", mutant, *mots)


def test_serie_19_la_reference_du_correcteur_est_l_ancien_script(tmp_path, monkeypatch):
    """Le correcteur recopie la logique de l'ancien script : on le vérifie en faisant tourner le VRAI vieux script."""
    import defis.heritage as heritage
    from defis.checks_serie_19 import _fichier_au_hasard, _reference_relances

    class FausseDate(d19):
        @classmethod
        def today(cls):
            return cls(2025, 4, 1)

    monkeypatch.setattr(heritage, "date", FausseDate)
    rng = random.Random(7)
    for essai in range(40):
        lignes = _fichier_au_hasard(rng, d19(2025, 4, 1), rng.randint(1, 30))
        chemin = heritage.ecrire_lignes(lignes, tmp_path / f"relances_{essai}.csv")
        operateur = heritage.OperateurFactice()
        monkeypatch.setattr(heritage, "_ENVOYES_AUJOURDHUI", 0)
        heritage.ancien_traitement_relances(chemin, operateur)
        par_telephone = {ligne["telephone"]: ligne["client_id"] for ligne in lignes}
        envoyes = [(par_telephone[t], offre) for t, offre in operateur.envois]
        assert envoyes == _reference_relances(lignes, d19(2025, 4, 1))


# Défi 119 : lire_preferences


def _lire_19(defaut=None):
    constante = {"version": 2, "canaux": [], "heure": 10, "langue": "fr"}

    def lire_preferences(texte):
        if defaut != "sans_none" and texte is None or (defaut not in ("sans_none", "vide_plante")
                                                       and texte is not None and not texte.strip()):
            return constante if defaut == "defaut_partage" else {"version": 2, "canaux": [], "heure": 10, "langue": "fr"}
        donnees = json.loads(texte)
        if not isinstance(donnees, dict) and defaut != "liste_attrapee":
            raise ValueError("objet JSON attendu")
        version = donnees["version"] if defaut == "version_obligatoire" else donnees.get("version", 1)
        if defaut == "true_version":
            connue = version in (1, 2) and not isinstance(version, str)
        elif defaut == "devine":
            connue, version = True, (1 if version == 1 else 2)
        else:
            connue = type(version) is int and version in (1, 2)
        if not connue:
            raise ValueError(f"version {version!r} inconnue")
        if version == 1:
            canaux = [c for c in ("email", "sms") if donnees.get(c)]
        else:
            canaux = donnees.get("canaux", [])
            if not isinstance(canaux, list):
                raise ValueError("canaux doit être une liste")
            if defaut == "canal_inconnu_ignore":
                canaux = [c for c in canaux if c in CANAUX_19]
            for canal in canaux:
                if canal not in CANAUX_19:
                    raise ValueError("canal inconnu" if defaut == "message_canal" else f"canal {canal!r} inconnu")
        heure = (donnees.get("heure") or 10) if defaut == "heure_ou" else donnees.get("heure", 10)
        valide = isinstance(heure, int) if defaut == "true_heure" else type(heure) is int
        if not valide or not 0 <= heure <= 23:
            raise ValueError(f"heure {heure!r} invalide")
        resultat = {"version": 2, "canaux": list(canaux) if defaut == "pas_trie" else sorted(set(canaux)),
                    "heure": heure, "langue": donnees.get("langue", "fr")}
        if defaut == "garde_inconnues":
            resultat = {**donnees, **resultat}
        return resultat

    return lire_preferences


@pytest.mark.parametrize("mutant,mots", [
    (_lire_19("sans_none"), ("texte is None",)),
    (_lire_19("vide_plante"), ("not texte.strip()",)),
    (_lire_19("version_obligatoire"), ("KeyError", 'donnees.get("version", 1)')),
    (_lire_19("defaut_partage"), ("NOUVEAU dictionnaire",)),
    (_lire_19("heure_ou"), ("0 (minuit)",)),
    (_lire_19("garde_inconnues"), ("clés inconnues",)),
    (_lire_19("devine"), ("version 3", "Ne devinez pas")),
    (_lire_19("true_version"), ("True == 1",)),
    (_lire_19("true_heure"), ("True est un entier",)),
    (_lire_19("canal_inconnu_ignore"), ("pigeon", "devrait lever ValueError")),
    (_lire_19("message_canal"), ("nommez « pigeon »",)),
    (_lire_19("pas_trie"), ("sorted(set(canaux))",)),
    (_lire_19("liste_attrapee"), ("AttributeError", "isinstance(donnees, dict)")),
])
def test_serie_19_preferences_erreurs_classiques(mutant, mots):
    _echec_19("lire_preferences", mutant, *mots)


# Défi 120 : AdaptateurSMS


def _adaptateur_19(defaut=None):
    import re as re_19

    def international(numero):
        brut = re_19.sub(r"[ .\-]", "", numero)
        formes = [(r"0([1-9]\d{8})", "+33"), (r"\+33([1-9]\d{8})", "+33")]
        if defaut != "sans_0033":
            formes.append((r"0033([1-9]\d{8})", "+33"))
        for motif, prefixe in formes:
            trouve = re_19.fullmatch(motif, brut)
            if trouve:
                return prefixe + trouve.group(1)
        return None

    class AdaptateurSMS:
        client_partage = None

        def __init__(self, client, expediteur="RETENTION"):
            if defaut == "attribut_classe":
                if AdaptateurSMS.client_partage is None:
                    AdaptateurSMS.client_partage = client
            else:
                self.client = client
            self.expediteur = "RETENTION" if defaut == "sans_expediteur" else expediteur

        def envoyer_sms(self, numero, texte):
            client = AdaptateurSMS.client_partage if defaut == "attribut_classe" else self.client
            if defaut != "sans_controle_texte" and defaut != "controle_apres" and (not texte or len(texte) > 160):
                raise ValueError("texte de 1 à 160 caractères")
            destinataire = numero if defaut == "numero_brut" else international(numero)
            if destinataire is None:
                return False
            essais = 3 if defaut == "retry" else 1
            for essai in range(essais):
                try:
                    reponse = client.messages.create(destinataire=destinataire, contenu=texte,
                                                      expediteur=self.expediteur)
                    break
                except (Exception if defaut in ("tout_false", "exception_large") else ErreurMessagerie) as exc:
                    code = getattr(exc, "code", None)
                    if defaut == "tout_false":
                        return False
                    if code == "INDISPONIBLE" and essai < essais - 1:
                        continue
                    if code in ("NUMERO_INVALIDE", "INDISPONIBLE"):
                        return False
                    if code == "CLE_INVALIDE":
                        if defaut == "sans_from":
                            raise PermissionError("clé refusée") from None
                        raise PermissionError("clé refusée") from exc
                    if defaut == "quota_false" or (defaut == "exception_large" and not isinstance(exc, ErreurMessagerie)):
                        return False
                    raise
            if defaut == "controle_apres" and (not texte or len(texte) > 160):
                raise ValueError("texte de 1 à 160 caractères")
            if defaut == "reponse":
                return reponse
            if defaut == "stop_true":
                return True
            return reponse.statut == "accepte"

    return AdaptateurSMS


@pytest.mark.parametrize("mutant,mots", [
    (_adaptateur_19("reponse"), ("BOOLÉEN",)),
    (_adaptateur_19("numero_brut"), ("format international",)),
    (_adaptateur_19("sans_0033"), ("0033 6 12 34 56 78", "+33612345678")),
    (_adaptateur_19("sans_expediteur"), ("expediteur=self.expediteur",)),
    (_adaptateur_19("attribut_classe"), ("attribut de classe",)),
    (_adaptateur_19("stop_true"), ("STOP", "reponse.statut")),
    (_adaptateur_19("retry"), ("nouvel essai",)),
    (_adaptateur_19("tout_false"), ("CLE_INVALIDE", "devrait lever PermissionError")),
    (_adaptateur_19("sans_from"), ("from exc",)),
    (_adaptateur_19("quota_false"), ("QUOTA_DEPASSE", "TELLE QUELLE")),
    (_adaptateur_19("exception_large"), ("bogue", "except Exception")),
    (_adaptateur_19("sans_controle_texte"), ("161 caractères", "facture 2 SMS")),
    (_adaptateur_19("controle_apres"), ("AVANT d'appeler create",)),
])
def test_serie_19_adaptateur_erreurs_classiques(mutant, mots):
    _echec_19("AdaptateurSMS", mutant, *mots)


# Défi 121 : en_ombre


def _ombre_19(defaut=None):
    stats_partagees = {"appels": 0, "comparaisons": 0, "differences": 0}

    def noter(fonction, args, kwargs, attraper=Exception):
        try:
            return ("ok", fonction(*args, **kwargs)), None
        except attraper as exc:
            return ("erreur", type(exc).__name__, str(exc)), exc

    def en_ombre(ancienne, nouvelle, journal, *, proportion=1.0, tirage=random.random, egal=operator.eq):
        if defaut != "proportion_tardive" and not 0 <= proportion <= 1:
            raise ValueError("proportion entre 0 et 1")
        stats = stats_partagees if defaut == "stats_globales" else {"appels": 0, "comparaisons": 0, "differences": 0}
        comparer_egal = operator.eq if defaut == "egal_ignore" else egal

        def fonction(*args, **kwargs):
            if defaut == "proportion_tardive" and not 0 <= proportion <= 1:
                raise ValueError("proportion entre 0 et 1")
            stats["appels"] += 1
            comparer = True if defaut == "tirage_ignore" else tirage() < proportion
            if comparer and defaut != "copie_apres":
                avant = copy.deepcopy((args, kwargs))
            if defaut == "ancienne_sur_copie":
                ancien, exc_ancienne = noter(ancienne, *copy.deepcopy((args, kwargs)))
            elif defaut == "pas_compare_si_ancienne_plante":
                ancien, exc_ancienne = ("ok", ancienne(*args, **kwargs)), None
            else:
                ancien, exc_ancienne = noter(ancienne, args, kwargs)
            if comparer:
                if defaut == "copie_apres":
                    avant = copy.deepcopy((args, kwargs))
                if defaut == "nouvelle_sur_originaux":
                    pour_nouvelle = (args, kwargs)
                elif defaut == "nouvelle_sur_memes":
                    pour_nouvelle = avant
                else:
                    pour_nouvelle = copy.deepcopy(avant)
                if defaut == "nouvelle_remonte":
                    nouveau = ("ok", nouvelle(*pour_nouvelle[0], **pour_nouvelle[1]))
                else:
                    attraper = BaseException if defaut == "tout_attraper" else Exception
                    nouveau, _ = noter(nouvelle, *pour_nouvelle, attraper=attraper)
                stats["comparaisons"] += 1
                if ancien[0] == nouveau[0] == "ok":
                    different = not comparer_egal(ancien[1], nouveau[1])
                elif ancien[0] == nouveau[0] == "erreur":
                    different = ancien[1] != nouveau[1]
                else:
                    different = True
                if different:
                    stats["differences"] += 1
                    journal({"arguments": avant, "ancienne": ancien, "nouvelle": nouveau})
            if exc_ancienne is not None:
                if defaut == "nouvelle_exception":
                    raise type(exc_ancienne)(*exc_ancienne.args)
                raise exc_ancienne
            return nouveau[1] if defaut == "renvoie_nouvelle" and comparer and nouveau[0] == "ok" else ancien[1]

        if defaut != "sans_wraps":
            fonction = functools.wraps(ancienne)(fonction)
        if defaut != "sans_stats":
            fonction.stats = stats
        return fonction

    return en_ombre


@pytest.mark.parametrize("mutant,mots", [
    (_ombre_19("sans_wraps"), ("functools.wraps",)),
    (_ombre_19("sans_stats"), ("fonction.stats",)),
    (_ombre_19("stats_globales"), ("partagent leurs stats",)),
    (_ombre_19("renvoie_nouvelle"), ("ANCIENNE version",)),
    (_ombre_19("ancienne_sur_copie"), ("VRAIS arguments",)),
    (_ombre_19("copie_apres"), ("AVANT de l'appeler",)),
    (_ombre_19("pas_compare_si_ancienne_plante"), ("MÊME quand l'ancienne plante",)),
    (_ombre_19("nouvelle_exception"), ("CETTE exception-là",)),
    (_ombre_19("nouvelle_remonte"), ("JAMAIS gêner",)),
    (_ombre_19("tout_attraper"), ("Ctrl+C",)),
    (_ombre_19("nouvelle_sur_originaux"), ("copie à elle",)),
    (_ombre_19("nouvelle_sur_memes"), ("AUTRE copie",)),
    (_ombre_19("egal_ignore"), ("egal(a, b)",)),
    (_ombre_19("proportion_tardive"), ("dès en_ombre",)),
    (_ombre_19("tirage_ignore"), ("tirage() < proportion",)),
])
def test_serie_19_ombre_erreurs_classiques(mutant, mots):
    _echec_19("en_ombre", mutant, *mots)


@pytest.mark.parametrize("nom,fabrique", [("photographier", _photographier_19), ("remise", _remise_19),
                                          ("decider_relances", _decider_19), ("lire_preferences", _lire_19),
                                          ("AdaptateurSMS", _adaptateur_19), ("en_ombre", _ombre_19)])
def test_serie_19_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Garde-fou : chaque version boguée ci-dessus ne diffère du juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_serie_19_les_tests_du_corrige_passent():
    import defis.corrige_serie_19 as corrige

    tests = [f for nom, f in list(vars(corrige).items()) if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_19_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.heritage as heritage
    import defis.serie_19 as eleve

    for nom in ("CANAUX", "CLIENTS_EXEMPLE", "JOUR_EXEMPLE", "LIGNES_EXEMPLE", "PREFERENCES_V1_EXEMPLE",
                "PREFERENCES_V2_EXEMPLE", "ClientMessagerie", "ErreurMessagerie", "OperateurFactice",
                "PasserelleAncienne", "Reponse", "ancien_score", "ancien_traitement_relances", "ecrire_lignes",
                "nouveau_score", "remise_ancienne"):
        assert getattr(eleve, nom) is getattr(heritage, nom), nom


def test_serie_19_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    from defis.checks_serie_19 import SERIE

    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6


def test_serie_19_le_correcteur_ne_touche_pas_aux_donnees_d_exemple():
    from defis.checks_serie_19 import SERIE

    clients, lignes = copy.deepcopy(CLIENTS_19), copy.deepcopy(LIGNES_19)
    import defis.corrige_serie_19 as corrige

    for defi in SERIE.defis:
        assert evaluer_defi(corrige, defi).statut == REUSSI
    assert CLIENTS_19 == clients and LIGNES_19 == lignes


def test_serie_19_les_exemples_des_docstrings_sont_justes():
    """Les exemples de defis/serie_19.py et defis/heritage.py, rejoués avec le corrigé."""
    from defis.corrige_serie_19 import (AdaptateurSMS, PasserelleAncienne, decider_relances, en_ombre, lire_preferences,
                                        photographier, remise)
    from defis.heritage import PREFERENCES_V1_EXEMPLE, PREFERENCES_V2_EXEMPLE

    assert ancien_score({"client_id": "C002", "charges": 45.0, "contrat": "Annuel", "appels": 0}) == 0.35
    assert remise_ancienne(100, 0.2) == 80.0
    assert PasserelleAncienne().envoyer_sms("06 12 34 56 78", "Votre offre vous attend") is True

    photo = photographier(ancien_score, [(c,) for c in CLIENTS_19])
    lignes = [f"{a[0]['client_id']} {a[0].get('charges')} {r}" for a, r in photo]
    assert lignes == ["C001 89,90 ('ok', 0.95)", "C002 45.0 ('ok', 0.35)", "C003 130 ('ok', 0.85)",
                      "C004 None ('erreur', 'KeyError', \"'charges'\")",
                      "C005 -10 ('erreur', 'ValueError', 'charges négatives : -10.0')"]
    assert CLIENTS_19[0]["charges"] == "89,90"
    assert (photographier(nouveau_score, [(c,) for c in CLIENTS_19]) == photo) is False

    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter("always")
        assert remise(100, pourcentage=20) == 80.0
        assert remise(100, 0.2) == 80.0
    assert [str(n.message) for n in notes] == [
        "remise(montant, taux) est dépréciée : écrivez remise(montant, pourcentage=20)"]

    assert decider_relances(LIGNES_19, JOUR_19) == [("C02", "FIDELITE"), ("C01", "ENGAGEMENT"), ("C08", "FIDELITE"),
                                                    ("C03", "ENGAGEMENT")]
    assert decider_relances(LIGNES_19, JOUR_19, budget=2) == [("C02", "FIDELITE"), ("C01", "ENGAGEMENT")]

    assert lire_preferences(PREFERENCES_V1_EXEMPLE) == {"version": 2, "canaux": ["sms"], "heure": 18, "langue": "fr"}
    assert lire_preferences(PREFERENCES_V2_EXEMPLE) == {"version": 2, "canaux": ["courrier", "sms"], "heure": 9,
                                                        "langue": "en"}
    assert lire_preferences(None) == {"version": 2, "canaux": [], "heure": 10, "langue": "fr"}

    client = ClientMessagerie("cle_test_123")
    sms = AdaptateurSMS(client)
    assert sms.envoyer_sms("06 12 34 56 78", "Votre offre vous attend") is True
    assert client.envoyes == [("+33612345678", "Votre offre vous attend", "RETENTION", 1)]
    assert sms.envoyer_sms("06 12 34", "Votre offre vous attend") is False

    differences = []
    score = en_ombre(ancien_score, nouveau_score, differences.append)
    assert score({"client_id": "C006", "charges": 50}) == 0.62
    assert repr(differences) == ("[{'arguments': (({'client_id': 'C006', 'charges': 50},), {}), "
                                 "'ancienne': ('ok', 0.62), 'nouvelle': ('ok', 0.37)}]")
    assert score.stats == {"appels": 1, "comparaisons": 1, "differences": 1}


def test_serie_19_le_vieux_script_de_relance_tourne_avec_un_faux_operateur(tmp_path, capsys, monkeypatch):
    """Le vieux code de heritage.py fonctionne vraiment (et garde son défaut : le budget global)."""
    import defis.heritage as heritage

    class FausseDate(d19):
        @classmethod
        def today(cls):
            return cls(2025, 3, 10)

    monkeypatch.setattr(heritage, "date", FausseDate)
    monkeypatch.setattr(heritage, "_ENVOYES_AUJOURDHUI", 48)
    chemin = heritage.ecrire_lignes(LIGNES_19, tmp_path / "relances.csv")
    operateur = heritage.OperateurFactice()
    heritage.ancien_traitement_relances(chemin, operateur)
    assert operateur.envois == [("0622222222", "FIDELITE"), ("0611111111", "ENGAGEMENT")]   # 48 + 2 = 50 : budget
    assert capsys.readouterr().out.splitlines()[-1] == "budget du jour atteint"


def test_serie_19_autres_ecritures_justes_acceptees():
    """Des solutions justes qui ne ressemblent pas au corrigé : le correcteur ne doit pas les refuser."""
    import re as re_19

    class AdaptateurSMS:                       # chiffres extraits, et un dictionnaire de traduction des codes
        TRADUCTION = {"NUMERO_INVALIDE": False, "INDISPONIBLE": False}

        def __init__(self, client, expediteur="RETENTION"):
            self.client, self.expediteur = client, expediteur

        def envoyer_sms(self, numero, texte):
            if not 1 <= len(texte) <= 160:
                raise ValueError("texte de 1 à 160 caractères")
            brut = re_19.sub(r"[ .\-]", "", numero)
            if brut.startswith("0033"):
                brut = "+" + brut[2:]
            elif brut.startswith("0") and not brut.startswith("00"):
                brut = "+33" + brut[1:]
            if not re_19.fullmatch(r"\+33[1-9]\d{8}", brut):
                return False
            try:
                return self.client.messages.create(destinataire=brut, contenu=texte,
                                                   expediteur=self.expediteur).statut == "accepte"
            except ErreurMessagerie as exc:
                if exc.code == "CLE_INVALIDE":
                    raise PermissionError(str(exc)) from exc
                if exc.code in self.TRADUCTION:
                    return self.TRADUCTION[exc.code]
                raise

    assert _evaluer("AdaptateurSMS", AdaptateurSMS).statut == REUSSI

    class _Ombre:                              # une classe avec __call__ au lieu d'une fonction intérieure
        def __init__(self, ancienne, nouvelle, journal, proportion, tirage, egal):
            functools.update_wrapper(self, ancienne)
            self.ancienne, self.nouvelle, self.journal = ancienne, nouvelle, journal
            self.proportion, self.tirage, self.egal = proportion, tirage, egal
            self.stats = {"appels": 0, "comparaisons": 0, "differences": 0}

        @staticmethod
        def _noter(f, args, kwargs):
            try:
                return ("ok", f(*args, **kwargs)), None
            except Exception as exc:
                return ("erreur", type(exc).__name__, str(exc)), exc

        def __call__(self, *args, **kwargs):
            self.stats["appels"] += 1
            if self.tirage() >= self.proportion:
                return self.ancienne(*args, **kwargs)
            avant = copy.deepcopy((args, kwargs))
            ancien, erreur = self._noter(self.ancienne, args, kwargs)
            nouveau, _ = self._noter(self.nouvelle, *copy.deepcopy(avant))
            self.stats["comparaisons"] += 1
            if ancien[0] != nouveau[0] or (ancien[0] == "ok" and not self.egal(ancien[1], nouveau[1])) or \
                    (ancien[0] == "erreur" and ancien[1] != nouveau[1]):
                self.stats["differences"] += 1
                self.journal({"arguments": avant, "ancienne": ancien, "nouvelle": nouveau})
            if erreur is not None:
                raise erreur
            return ancien[1]

    def en_ombre(ancienne, nouvelle, journal, *, proportion=1.0, tirage=random.random, egal=operator.eq):
        if not 0 <= proportion <= 1:
            raise ValueError("proportion entre 0 et 1")
        return _Ombre(ancienne, nouvelle, journal, proportion, tirage, egal)

    assert _evaluer("en_ombre", en_ombre).statut == REUSSI

    def lire_preferences(texte):               # une table de lecteurs par version
        if texte is None or texte.strip() == "":
            donnees = {}
        else:
            donnees = json.loads(texte)
        if type(donnees) is not dict:
            raise ValueError("objet JSON attendu")
        lecteurs = {1: lambda d: [c for c in ("email", "sms") if d.get(c) is True], 2: lambda d: d.get("canaux", [])}
        version = donnees.get("version", 1)
        if isinstance(version, bool) or version not in lecteurs:
            raise ValueError(f"version {version!r} inconnue")
        canaux = lecteurs[version](donnees)
        if not isinstance(canaux, list) or any(c not in CANAUX_19 for c in canaux):
            raise ValueError(f"canaux invalides : {canaux!r}")
        heure = donnees.get("heure", 10)
        if type(heure) is not int or heure not in range(24):
            raise ValueError(f"heure invalide : {heure!r}")
        return {"version": 2, "canaux": sorted(set(canaux)), "heure": heure, "langue": donnees.get("langue", "fr")}

    assert _evaluer("lire_preferences", lire_preferences).statut == REUSSI


# --------------------------------------------------------------------------- Série 20 : le texte sans surprises

import codecs  # noqa: E402
import re as re20  # noqa: E402
import unicodedata as ud20  # noqa: E402

from defis.textes import APOSTROPHES as APOSTROPHES_20  # noqa: E402
from defis.textes import EXPORT_ABIME as EXPORT_20  # noqa: E402
from defis.textes import GSM_BASE as GSM_BASE_20  # noqa: E402
from defis.textes import GSM_EXTENSION as GSM_EXT_20  # noqa: E402
from defis.textes import LETTRES_LIEES as LETTRES_20  # noqa: E402
from defis.textes import MESSAGES_EXEMPLE as MESSAGES_20  # noqa: E402
from defis.textes import NOMS_EXEMPLE as NOMS_20  # noqa: E402
from defis.textes import NUMEROS_EXEMPLE as NUMEROS_20  # noqa: E402
from defis.textes import TIRETS as TIRETS_20  # noqa: E402
from defis.textes import ZWJ as ZWJ_20  # noqa: E402


def _echec_20(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


def test_serie_20_les_tables_gsm_sont_celles_de_la_norme():
    assert len(GSM_BASE_20) == len(set(GSM_BASE_20)) == 127        # 128 positions, moins le caractère d'échappement
    assert len(GSM_EXT_20) == len(set(GSM_EXT_20)) == 10
    assert not set(GSM_BASE_20) & set(GSM_EXT_20)
    assert all(c in GSM_BASE_20 for c in "@£$¥èéùìòÇØøÅåΔ_ÆæßÉ¤¡ÄÖÑÜ§¿äöñüà\n\r")
    assert not any(c in GSM_BASE_20 + GSM_EXT_20 for c in "êâîôûëïç’«»…`\t")


# Défi 122 : cle_comparaison


def _cle_comparaison_20(defaut=None):
    def cle_comparaison(texte):
        forme = {"nfc": "NFC", "decompose": "NFKD", "sans_normaliser": None}.get(defaut, "NFKC")
        if forme:
            texte = ud20.normalize(forme, texte)
        texte = texte.lower() if defaut == "lower" else texte.casefold()
        if defaut != "garde_invisibles":
            texte = "".join(c for c in texte if ud20.category(c) != "Cf")
        if forme:
            texte = ud20.normalize(forme, texte)
        if defaut == "sans_accents":
            texte = ud20.normalize("NFC", "".join(c for c in ud20.normalize("NFD", texte) if not ud20.combining(c)))
        if defaut == "oublie_return":
            return None
        if defaut == "strip_seulement":
            return texte.strip()
        if defaut == "split_espace":
            return " ".join(mot for mot in texte.split(" ") if mot)
        return " ".join(texte.split())

    return cle_comparaison


@pytest.mark.parametrize("mutant,mots", [
    (_cle_comparaison_20("sans_normaliser"), ("Mac", 'normalize("NFKC"')),
    (_cle_comparaison_20("nfc"), ("pleine chasse", "NFKC")),
    (_cle_comparaison_20("decompose"), ("DÉCOMPOSÉE",)),
    (_cle_comparaison_20("lower"), ("ß", "casefold")),
    (_cle_comparaison_20("garde_invisibles"), ("LARGEUR NULLE", "« Cf »")),
    (_cle_comparaison_20("strip_seulement"), ("UNE espace",)),
    (_cle_comparaison_20("split_espace"), ("SANS argument",)),
    (_cle_comparaison_20("sans_accents"), ("Gardez les accents",)),
    (_cle_comparaison_20("oublie_return"), ("return",)),
])
def test_serie_20_cle_comparaison_erreurs_classiques(mutant, mots):
    _echec_20("cle_comparaison", mutant, *mots)


# Défi 123 : cle_recherche


def _cle_recherche_20(defaut=None):
    from defis.corrige_serie_20 import cle_comparaison

    def cle_recherche(texte):
        if defaut == "ascii_ignore":
            return cle_comparaison(texte).encode("ascii", "ignore").decode()
        base = ud20.normalize("NFC", texte).lower() if defaut == "sans_cle_comparaison" else cle_comparaison(texte)
        forme = "NFC" if defaut == "nfc_au_lieu_de_nfd" else "NFD"
        texte = "".join(c for c in ud20.normalize(forme, base) if not ud20.combining(c))
        table = {}
        if defaut != "sans_lettres_liees":
            table.update(LETTRES_20)
        if defaut != "sans_apostrophes":
            table.update({a: "'" for a in APOSTROPHES_20})
        if defaut != "sans_tirets":
            table.update({t: "-" for t in TIRETS_20})
        return ud20.normalize("NFC", texte.translate(str.maketrans(table)))

    return cle_recherche


@pytest.mark.parametrize("mutant,mots", [
    (_cle_recherche_20("ascii_ignore"), ('encode("ascii", "ignore")',)),
    (_cle_recherche_20("sans_cle_comparaison"), ("cle_comparaison",)),
    (_cle_recherche_20("nfc_au_lieu_de_nfd"), ("NFD", "combining")),
    (_cle_recherche_20("sans_lettres_liees"), ("LETTRES_LIEES",)),
    (_cle_recherche_20("sans_apostrophes"), ("APOSTROPHES",)),
    (_cle_recherche_20("sans_tirets"), ("TIRETS",)),
])
def test_serie_20_cle_recherche_erreurs_classiques(mutant, mots):
    _echec_20("cle_recherche", mutant, *mots)


# Défi 124 : numero_e164


def _numero_20(defaut=None):
    def refuser(texte):
        if defaut == "renvoie_none":
            return None
        raise ValueError(f"numéro illisible : {texte!r}")

    def numero_e164(texte):
        if defaut == "espace_seulement":
            brut = texte.replace(" ", "").replace(".", "").replace("-", "").replace("/", "")
        else:
            brut = "".join(c for c in texte if not c.isspace() and c not in ".-/")
        if defaut == "indexe" and brut[0] == "?":
            return None
        if defaut != "sans_zero_entre_parentheses":
            for prefixe in ("+33(0)", "0033(0)"):
                if brut.startswith(prefixe):
                    brut = prefixe[:-3] + brut[len(prefixe):]
        chiffre = str.isdigit if defaut == "isdigit" else (lambda c: c in "0123456789")
        if brut.count("+") > (1 if brut.startswith("+") else 0) or any(c != "+" and not chiffre(c) for c in brut):
            return refuser(texte)
        if brut.startswith("+"):
            if not brut.startswith("+33") and defaut != "tout_indicatif":
                return refuser(texte)
            national = brut[3:]
        elif brut.startswith("0033") and defaut != "sans_0033":
            national = brut[4:]
        elif len(brut) == 10 and brut.startswith("0"):
            national = brut[1:]
        elif len(brut) == 9 and defaut != "sans_excel":
            national = brut
        else:
            return refuser(texte)
        if defaut == "accepte_zero_apres_33" and len(national) == 10 and national[0] == "0":
            national = national[1:]
        if len(national) != 9 or national[0] == "0":
            return refuser(texte)
        return ("+330" if defaut == "garde_le_zero" else "+33") + national

    return numero_e164


@pytest.mark.parametrize("mutant,mots", [
    (_numero_20("garde_le_zero"), ("E.164", "SANS son 0")),
    (_numero_20("espace_seulement"), ("INSÉCABLES", "isspace")),
    (_numero_20("sans_zero_entre_parentheses"), ("lève ValueError", "(0)")),
    (_numero_20("sans_0033"), ("0033",)),
    (_numero_20("sans_excel"), ("Excel",)),
    (_numero_20("renvoie_none"), ("devrait lever ValueError", "None")),
    (_numero_20("indexe"), ("IndexError", "au lieu de ValueError")),
    (_numero_20("accepte_zero_apres_33"), ("+33 06",)),
    (_numero_20("tout_indicatif"), ("Belgique",)),
    (_numero_20("isdigit"), ("isdigit", "0123456789")),
])
def test_serie_20_numero_erreurs_classiques(mutant, mots):
    _echec_20("numero_e164", mutant, *mots)


# Défi 125 : cout_sms


def _cout_20(defaut=None):
    gsm, extension = set(GSM_BASE_20), set(GSM_EXT_20)

    def segments(tailles, seul, par_segment):
        if sum(tailles) <= seul:
            return 1
        if defaut == "division":
            return -(-sum(tailles) // par_segment)
        nombre, place = 1, par_segment
        for taille in tailles:
            if taille > place:
                nombre, place = nombre + 1, par_segment
            place -= taille
        return nombre

    def cout_sms(message):
        if defaut == "normalise":
            message = ud20.normalize("NFC", message)
        if message == "" and defaut != "vide_accepte":
            raise ValueError("message vide")
        if all(c in gsm or c in extension for c in message):
            tailles = [1 if defaut == "euro_1" or c in gsm else 2 for c in message]
            return ("GSM7" if defaut == "nom" else "GSM-7"), segments(tailles, 160, 160 if defaut == "par_160" else 153)
        tailles = [1 if defaut == "len_utf16" or ord(c) <= 0xFFFF else 2 for c in message]
        seul, par_segment = (160, 153) if defaut == "ucs2_comme_gsm" else (70, 67)
        return "UCS-2", segments(tailles, seul, par_segment)

    return cout_sms


@pytest.mark.parametrize("mutant,mots", [
    (_cout_20("nom"), ("exactement « GSM-7 »",)),
    (_cout_20("par_160"), ("153",)),
    (_cout_20("euro_1"), ("2 unités", "`len()`")),
    (_cout_20("division"), ("JAMAIS", "place restante")),
    (_cout_20("ucs2_comme_gsm"), ("70", "pas 160")),
    (_cout_20("len_utf16"), ("UTF-16", "0xFFFF")),
    (_cout_20("normalise"), ("tel qu'il partira",)),
    (_cout_20("vide_accepte"), ("ValueError", "message vide")),
])
def test_serie_20_cout_sms_erreurs_classiques(mutant, mots):
    _echec_20("cout_sms", mutant, *mots)


# Défi 126 : reparer_mojibake


def _reparer_20(defaut=None):
    def octets(texte):
        if defaut == "latin1":
            return texte.encode("latin-1")
        if defaut == "cp1252_strict":
            return texte.encode("cp1252")
        resultat = bytearray()
        for c in texte:
            try:
                resultat += c.encode("cp1252")
            except UnicodeEncodeError:
                if ord(c) > 0xFF or (defaut == "trous_seulement" and ord(c) not in (0x81, 0x8D, 0x8F, 0x90, 0x9D)):
                    raise
                resultat.append(ord(c))
        return bytes(resultat)

    def reparer_un(texte):
        while True:
            try:
                repare = octets(texte).decode("utf-8")
            except (UnicodeDecodeError if defaut == "decode_seulement" else UnicodeError):
                return texte
            if repare == texte and defaut != "sans_arret":
                return texte
            texte = repare
            if defaut == "un_seul_tour":
                return texte

    def reparer_mojibake(texte):
        if defaut == "ignore":
            return texte.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        if defaut == "mot_a_mot":
            return " ".join(reparer_un(mot) for mot in texte.split(" "))
        return reparer_un(texte)

    return reparer_mojibake


@pytest.mark.parametrize("mutant,mots", [
    (_reparer_20("ignore"), ('errors="ignore"', "SAIN")),
    (_reparer_20("decode_seulement"), ("UnicodeEncodeError", "`UnicodeError`")),
    (_reparer_20("sans_arret"), ("ne s'arrête jamais", "CHANGE")),
    (_reparer_20("latin1"), ("pas latin-1",)),
    (_reparer_20("cp1252_strict"), ("TROUS",)),
    (_reparer_20("un_seul_tour"), ("DEUX fois",)),
    (_reparer_20("trous_seulement"), ("latin-1", "pas seulement les 5 trous")),
    (_reparer_20("mot_a_mot"), ("EN ENTIER",)),
])
def test_serie_20_reparer_erreurs_classiques(mutant, mots):
    _echec_20("reparer_mojibake", mutant, *mots)


def test_serie_20_la_garde_anti_boucle_rend_le_traceur_precedent():
    """Arrêter une boucle sans fin ne doit pas désactiver un débogueur ou un outil de couverture."""
    def traceur(frame, evenement, arg):
        return None

    precedent = sys.gettrace()
    sys.settrace(traceur)
    try:
        resultat = _evaluer("reparer_mojibake", _reparer_20("sans_arret"))
        assert sys.gettrace() is traceur
    finally:
        sys.settrace(precedent)
    assert resultat.statut == ECHEC and "ne s'arrête jamais" in resultat.message


# Défi 127 : tronquer


def _tronquer_20(defaut=None):
    def est_drapeau(c):
        return 0x1F1E6 <= ord(c) <= 0x1F1FF

    def graphemes(texte):
        if defaut == "points_de_code":
            return list(texte)
        groupes, colle = [], False
        for c in texte:
            if defaut == "drapeaux_en_chaine":
                drapeau = est_drapeau(c) and bool(groupes) and est_drapeau(groupes[-1][-1])
            elif defaut == "sans_drapeaux":
                drapeau = False
            else:
                drapeau = est_drapeau(c) and bool(groupes) and len(groupes[-1]) == 1 and est_drapeau(groupes[-1])
            marque = ud20.combining(c) != 0 if defaut == "combining" else ud20.category(c).startswith("M")
            attache = (colle and defaut != "sans_zwj") or marque or drapeau \
                or (0x1F3FB <= ord(c) <= 0x1F3FF and defaut != "sans_teinte") or (c == ZWJ_20 and defaut != "sans_zwj")
            if attache and (groupes or defaut == "debut"):
                groupes[-1] += c
            else:
                groupes.append(c)
            colle = c == ZWJ_20
        return groupes

    def tronquer(texte, limite):
        if limite < 1 and defaut != "sans_valueerror":
            raise ValueError("limite")
        morceaux = graphemes(texte)
        taille = len(texte) if defaut == "len" else len(morceaux)
        if taille <= limite and defaut != "toujours":
            return texte.rstrip() if defaut == "rstrip_toujours" else texte
        garde = "".join(morceaux[: limite if defaut == "points_en_plus" else limite - 1])
        if defaut != "sans_rstrip":
            garde = garde.rstrip()
        return garde + ("..." if defaut == "trois_points" else "…")

    return tronquer


@pytest.mark.parametrize("mutant,mots", [
    (_tronquer_20("trois_points"), ("trois points",)),
    (_tronquer_20("points_en_plus"), ("compte dans la limite",)),
    (_tronquer_20("sans_rstrip"), ("rstrip",)),
    (_tronquer_20("toujours"), ("TEL QUEL",)),
    (_tronquer_20("rstrip_toujours"), ("espaces finales",)),
    (_tronquer_20("points_de_code"), ("15 points de code",)),
    (_tronquer_20("len"), ("15 points de code",)),
    (_tronquer_20("combining"), ("U+FE0F",)),
    (_tronquer_20("sans_teinte"), ("teinte",)),
    (_tronquer_20("sans_zwj"), ("ZWJ",)),
    (_tronquer_20("sans_drapeaux"), ("drapeau",)),
    (_tronquer_20("drapeaux_en_chaine"), ("3 caractères visibles",)),
    (_tronquer_20("debut"), ("IndexError", "pas vide")),
    (_tronquer_20("sans_valueerror"), ("ValueError",)),
])
def test_serie_20_tronquer_erreurs_classiques(mutant, mots):
    _echec_20("tronquer", mutant, *mots)


@pytest.mark.parametrize("nom,fabrique", [("cle_comparaison", _cle_comparaison_20), ("cle_recherche", _cle_recherche_20),
                                          ("numero_e164", _numero_20), ("cout_sms", _cout_20),
                                          ("reparer_mojibake", _reparer_20), ("tronquer", _tronquer_20)])
def test_serie_20_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Garde-fou : chaque version boguée ci-dessus ne diffère du juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_serie_20_les_tests_du_corrige_passent():
    import defis.corrige_serie_20 as corrige

    tests = [f for nom, f in list(vars(corrige).items()) if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_20_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.serie_20 as eleve
    import defis.textes as textes

    for nom in ("APOSTROPHES", "EXPORT_ABIME", "GSM_BASE", "GSM_EXTENSION", "LETTRES_LIEES", "MESSAGES_EXEMPLE",
                "NOMS_EXEMPLE", "NUMEROS_EXEMPLE", "TIRETS", "ZWJ", "montrer"):
        assert getattr(eleve, nom) is getattr(textes, nom), nom


def test_serie_20_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    from defis.checks_serie_20 import SERIE

    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6


def test_serie_20_le_correcteur_ne_touche_pas_aux_donnees_d_exemple():
    from defis.checks_serie_20 import SERIE

    copies = [list(NOMS_20), list(NUMEROS_20), list(MESSAGES_20), list(EXPORT_20), dict(LETTRES_20)]
    import defis.corrige_serie_20 as corrige

    for defi in SERIE.defis:
        assert evaluer_defi(corrige, defi).statut == REUSSI
    assert copies == [NOMS_20, NUMEROS_20, MESSAGES_20, EXPORT_20, LETTRES_20]


def test_serie_20_les_exemples_des_docstrings_sont_justes():
    """Les exemples de defis/serie_20.py et defis/textes.py, rejoués avec le corrigé."""
    from defis.corrige_serie_20 import (cle_comparaison, cle_recherche, cout_sms, montrer, numero_e164,
                                        reparer_mojibake, tronquer)

    assert (NOMS_20[0] == NOMS_20[1], len(NOMS_20[0]), len(NOMS_20[1])) == (False, 12, 15)
    assert ("€" in GSM_BASE_20, "€" in GSM_EXT_20) == (False, True)
    assert montrer(NOMS_20[1])[:3] == ["U+0048 LATIN CAPITAL LETTER H", "U+0065 LATIN SMALL LETTER E",
                                       "U+0301 COMBINING ACUTE ACCENT"]
    assert montrer("é") == ["U+0065 LATIN SMALL LETTER E", "U+0301 COMBINING ACUTE ACCENT"]

    assert {cle_comparaison(nom) for nom in NOMS_20[:5]} == {"hélène dupré"}
    assert cle_comparaison("JÜRGEN STRASSE") == cle_comparaison("Jürgen Straße")
    assert cle_comparaison("Hélène Dupre") != cle_comparaison("Hélène Dupré")

    assert cle_recherche("Hélène Dupré") == "helene dupre"
    assert cle_recherche("Cœur d’Alène – Ørsted") == "coeur d'alene - orsted"
    assert [n for n in ["Hélène Dupré", "Anne Martin"] if cle_recherche("DUPRE") in cle_recherche(n)] == ["Hélène Dupré"]
    assert cle_recherche("Анна Ёлкина") == "анна елкина"

    assert [numero_e164(n) for n in NUMEROS_20] == ["+33612345678"] * 7
    with pytest.raises(ValueError) as erreur:
        numero_e164("06 12 34 56 7O")
    assert str(erreur.value) == "numéro illisible : '06 12 34 56 7O'"
    assert "٠٦١٢٣٤٥٦٧٨".isdigit() and "²".isdigit() and int("٣") == 3          # ce que dit la docstring

    assert [cout_sms(m) for m in MESSAGES_20] == [("GSM-7", 1), ("UCS-2", 2), ("GSM-7", 1)]
    assert MESSAGES_20[0].replace("dimanche", "l’été") == MESSAGES_20[1]

    assert [reparer_mojibake(ligne) for ligne in EXPORT_20] == [
        "Hélène Dupré", "Prix : 20 € par mois", "ÁLVAREZ Ángel", "François", "Hélène Dupré"]
    assert "é".encode("utf-8") == b"\xc3\xa9" and "é".encode("utf-8").decode("cp1252") == "Ã©"
    assert reparer_mojibake("Łódź") == "Łódź"

    famille = "👨" + ZWJ_20 + "👩" + ZWJ_20 + "👧"
    assert (len(famille), len("🇫🇷"), len(NOMS_20[1])) == (5, 2, 15)
    assert tronquer("Votre offre expire bientôt", 10) == "Votre off…"
    assert tronquer("Votre offre expire bientôt", 7) == "Votre…"
    assert repr(tronquer(f"Bravo {famille} !", 8)) == "'Bravo 👨\\u200d👩\\u200d👧…'"
    assert tronquer(NOMS_20[1], 12) == NOMS_20[1]


def test_serie_20_autres_ecritures_justes_acceptees():
    """Des solutions justes qui ne ressemblent pas au corrigé : le correcteur ne doit pas les refuser."""

    def cle_comparaison(texte):                  # casefold d'abord, invisibles d'abord, regex pour les blancs
        visibles = "".join(c for c in texte if ud20.category(c) != "Cf")
        return re20.sub(r"\s+", " ", ud20.normalize("NFKC", visibles.casefold())).strip()

    assert _evaluer("cle_comparaison", cle_comparaison).statut == REUSSI

    def cle_recherche(texte):                    # replace() en boucle, NFKD, catégorie « Mn »
        texte = cle_comparaison(texte)
        for lettre, remplacement in LETTRES_20.items():
            texte = texte.replace(lettre, remplacement)
        for apostrophe in APOSTROPHES_20:
            texte = texte.replace(apostrophe, "'")
        for tiret in TIRETS_20:
            texte = texte.replace(tiret, "-")
        return "".join(c for c in ud20.normalize("NFKD", texte) if ud20.category(c) != "Mn")

    assert _evaluer("cle_recherche", cle_recherche).statut == REUSSI

    def numero_e164(texte):                      # une seule expression régulière, chiffres ASCII
        brut = re20.sub(r"[\s./-]", "", texte)
        trouve = re20.fullmatch(r"(?:(?:\+33|0033)(?:\(0\))?|0)?([1-9][0-9]{8})", brut)
        if trouve is None:
            raise ValueError(f"numéro invalide : {texte!r}")
        return "+33" + trouve.group(1)

    assert _evaluer("numero_e164", numero_e164).statut == REUSSI

    def cout_sms(message):                       # UTF-16 pour compter, segments remplis avec une liste
        if not message:
            raise ValueError("vide")
        gsm = all(c in GSM_BASE_20 + GSM_EXT_20 for c in message)
        tailles = [(2 if c in GSM_EXT_20 else 1) if gsm else len(c.encode("utf-16-le")) // 2 for c in message]
        seul, par_segment = (160, 153) if gsm else (70, 67)
        if sum(tailles) <= seul:
            return ("GSM-7" if gsm else "UCS-2"), 1
        remplis = [0]
        for taille in tailles:
            if remplis[-1] + taille > par_segment:
                remplis.append(0)
            remplis[-1] += taille
        return ("GSM-7" if gsm else "UCS-2"), len(remplis)

    assert _evaluer("cout_sms", cout_sms).statut == REUSSI

    def _trous(erreur):                          # un gestionnaire d'erreurs d'encodage (codecs.register_error)
        caractere = erreur.object[erreur.start]
        if ord(caractere) > 0xFF:
            raise erreur
        return bytes([ord(caractere)]), erreur.start + 1

    codecs.register_error("test_serie_20_trous", _trous)

    def reparer_mojibake(texte):                 # récursif
        try:
            repare = texte.encode("cp1252", errors="test_serie_20_trous").decode("utf-8")
        except UnicodeError:
            return texte
        return texte if repare == texte else reparer_mojibake(repare)

    assert _evaluer("reparer_mojibake", reparer_mojibake).statut == REUSSI

    def tronquer(texte, limite):                 # parcours par indices, sans construire toute la liste
        if limite < 1:
            raise ValueError("limite >= 1")

        def fin_du_grapheme(i):
            debut, i = i, i + 1
            while i < len(texte):
                c, precedent = texte[i], texte[i - 1]
                drapeau = 0x1F1E6 <= ord(c) <= 0x1F1FF and i - debut == 1 and 0x1F1E6 <= ord(texte[debut]) <= 0x1F1FF
                if precedent == ZWJ_20 or c == ZWJ_20 or ud20.category(c)[0] == "M" \
                        or 0x1F3FB <= ord(c) <= 0x1F3FF or drapeau:
                    i += 1
                else:
                    break
            return i

        coupures, i = [], 0
        while i < len(texte):
            i = fin_du_grapheme(i)
            coupures.append(i)
        if len(coupures) <= limite:
            return texte
        return texte[: coupures[limite - 2] if limite > 1 else 0].rstrip() + "…"

    assert _evaluer("tronquer", tronquer).statut == REUSSI


# --------------------------------------------------------------------------- Série 21 : le temps qui passe

import datetime as dt21  # noqa: E402
import math as math21  # noqa: E402

import numpy as np21  # noqa: E402

from defis.abonnements import abonnements_exemple as abonnements_21  # noqa: E402
from defis.abonnements import dates_saisies as dates_21  # noqa: E402
from defis.abonnements import factures_exemple as factures_21  # noqa: E402
from defis.abonnements import generer_abonnements as generer_abonnements_21  # noqa: E402
from defis.abonnements import generer_factures as generer_factures_21  # noqa: E402


def _echec_21(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


# Défi 128 : lire_dates


def _lire_dates_21(defaut=None):
    def lire_dates(textes):
        if defaut == "oublie_return":
            return None
        if defaut == "modifie":
            textes[:] = textes.str.strip()
            propres = textes
        else:
            propres = textes if defaut == "sans_strip" else textes.str.strip()
        if defaut == "texte":
            return propres
        if defaut == "sans_format":
            resultat = pd.to_datetime(propres)
        elif defaut == "dayfirst":
            resultat = pd.to_datetime(propres, dayfirst=True)
        elif defaut == "coerce":
            resultat = pd.to_datetime(propres, format="%d/%m/%Y", errors="coerce")
        else:
            resultat = pd.to_datetime(propres, format="%d/%m/%Y")
        if defaut == "liste":
            return pd.Series(list(resultat))
        if defaut == "sans_nom":
            return resultat.rename(None)
        return resultat

    return lire_dates


@pytest.mark.parametrize("mutant,mots", [
    (_lire_dates_21("sans_format"), ("INVERSÉS", 'format="%d/%m/%Y"')),
    (_lire_dates_21("dayfirst"), ("dayfirst=True", "EXIGE")),
    (_lire_dates_21("coerce"), ('errors="coerce"', "NaT")),
    (_lire_dates_21("sans_strip"), (".str.strip()",)),
    (_lire_dates_21("liste"), ("[7, 3, 9]", "garde l'index")),
    (_lire_dates_21("texte"), ("DATES", "pd.to_datetime")),
    (_lire_dates_21("modifie"), ("MODIFIÉ",)),
    (_lire_dates_21("sans_nom"), ("'inscription'",)),
    (_lire_dates_21("oublie_return"), ("return",)),
])
def test_serie_21_lire_dates_erreurs_classiques(mutant, mots):
    _echec_21("lire_dates", mutant, *mots)


# Défi 129 : nouveaux_par_mois


def _nouveaux_21(defaut=None):
    def nouveaux_par_mois(debuts, premier, dernier):
        mois = pd.period_range(premier, dernier, freq="M", name="mois")
        if defaut == "dernier_exclu":
            mois = mois[:-1]
        if defaut == "dt_month":
            par_numero = debuts.dt.month.value_counts()
            return pd.Series([int(par_numero.get(m.month, 0)) for m in mois], index=mois, name="nouveaux")
        if defaut == "minuit":
            return pd.Series([int(((debuts >= m.start_time) & (debuts <= m.end_time.normalize())).sum()) for m in mois],
                             index=mois, name="nouveaux")
        if defaut == "resample":
            return debuts.dropna().to_frame("d").set_index("d").resample("MS").size().rename("nouveaux")
        comptes = debuts.dt.to_period("M").value_counts()
        if defaut == "sans_reindex":
            comptes = comptes.sort_index()
            resultat = comptes[(comptes.index >= mois[0]) & (comptes.index <= mois[-1])].rename_axis("mois")
        elif defaut == "float":
            resultat = comptes.reindex(mois).fillna(0)
        else:
            resultat = comptes.reindex(mois, fill_value=0).astype("int64")
        if defaut == "chaines":
            resultat.index = resultat.index.astype(str)
        if defaut == "oublie_return":
            return None
        return resultat if defaut == "sans_nom" else resultat.rename("nouveaux")

    return nouveaux_par_mois


@pytest.mark.parametrize("mutant,mots", [
    (_nouveaux_21("sans_reindex"), ("il manque les mois 2024-04, 2024-05", "fill_value=0")),
    (_nouveaux_21("dernier_exclu"), ("INCLUS",)),
    (_nouveaux_21("dt_month"), ("dt.month", "2023")),
    (_nouveaux_21("minuit"), ("23 h 30", "dt.to_period")),
    (_nouveaux_21("resample"), ("DATES", "Period")),
    (_nouveaux_21("float"), ("ENTIERS",)),
    (_nouveaux_21("chaines"), ("se trient mal",)),
    (_nouveaux_21("sans_nom"), ("'nouveaux'",)),
    (_nouveaux_21("oublie_return"), ("return",)),
])
def test_serie_21_nouveaux_par_mois_erreurs_classiques(mutant, mots):
    _echec_21("nouveaux_par_mois", mutant, *mots)


# Défi 130 : mouvements_mensuels


def _mouvements_21(defaut=None):
    def mouvements_mensuels(abonnements, premier, dernier):
        debut, fin = abonnements["debut"], abonnements["fin"]
        if defaut != "sans_valueerror" and (fin < debut).any():
            raise ValueError("fin avant début")
        if defaut == "modifie":
            abonnements["mois_debut"] = debut.dt.to_period("M")
        mois_debut, mois_fin = debut.dt.to_period("M"), fin.dt.to_period("M")
        lignes = []
        for m in pd.period_range(premier, dernier, freq="M"):
            if defaut == "nat_oublie":
                present = (mois_debut < m) & (mois_fin >= m)
            elif defaut == "arrive_le_1er":
                present = (debut <= m.start_time) & (mois_fin.isna() | (mois_fin >= m))
            elif defaut == "parti_le_1er":
                present = (mois_debut < m) & (fin.isna() | (fin > m.start_time))
            elif defaut == "stock_oublie":
                present = (mois_debut < m) & (mois_debut >= pd.Period(premier, "M")) & (mois_fin.isna() | (mois_fin >= m))
            else:
                present = (mois_debut < m) & (mois_fin.isna() | (mois_fin >= m))
            arrive, part = mois_debut == m, mois_fin == m
            actifs, n_arrive, n_part = int(present.sum()), int(arrive.sum()), int(part.sum())
            actifs_fin = actifs + n_arrive if defaut == "identite" else actifs + n_arrive - n_part
            if defaut == "division_zero":
                taux = int((present & part).sum()) / actifs
            elif defaut == "taux_tous_partis":
                taux = n_part / actifs if actifs else float("nan")
            elif defaut == "taux_sur_fin":
                taux = int((present & part).sum()) / actifs_fin if actifs_fin else float("nan")
            else:
                taux = int((present & part).sum()) / actifs if actifs else (0.0 if defaut == "taux_zero" else float("nan"))
            lignes.append((actifs, n_arrive, n_part, actifs_fin, taux))
        colonnes = ["actifs_debut", "nouveaux", "partis", "actifs_fin", "taux_churn"]
        table = pd.DataFrame(lignes, index=pd.period_range(premier, dernier, freq="M", name="mois"), columns=colonnes)
        if defaut == "float":
            table = table.astype("float64")
        if defaut == "ordre_colonnes":
            table = table[["nouveaux", "partis", "actifs_debut", "actifs_fin", "taux_churn"]]
        return table

    return mouvements_mensuels


@pytest.mark.parametrize("mutant,mots", [
    (_mouvements_21("nat_oublie"), ("TOUJOURS ABONNÉS", "isna()")),
    (_mouvements_21("arrive_le_1er"), ("arrivé le 1er janvier", "AVANT")),
    (_mouvements_21("parti_le_1er"), ("parti le 1er février", "APRÈS")),
    (_mouvements_21("stock_oublie"), ("stock de départ",)),
    (_mouvements_21("identite"), ("boucler",)),
    (_mouvements_21("taux_tous_partis"), ("D, arrivé le 10",)),
    (_mouvements_21("taux_sur_fin"), ("DÉBUT",)),
    (_mouvements_21("taux_zero"), ("PERSONNE", "NaN")),
    (_mouvements_21("division_zero"), ("ZeroDivisionError", "NaN")),
    (_mouvements_21("sans_valueerror"), ("ValueError",)),
    (_mouvements_21("float"), ("ENTIERS",)),
    (_mouvements_21("ordre_colonnes"), ("dans cet ordre",)),
    (_mouvements_21("modifie"), ("MODIFIÉ",)),
])
def test_serie_21_mouvements_mensuels_erreurs_classiques(mutant, mots):
    _echec_21("mouvements_mensuels", mutant, *mots)


# Défi 131 : table_retention


def _retention_21(defaut=None):
    def table_retention(abonnements, photo):
        if defaut == "photo_texte":
            dernier = pd.Period(photo[:7], "M") - 1
        elif defaut == "veille":
            dernier = (pd.Timestamp(photo) - pd.Timedelta(days=1)).to_period("M") - 1
        elif defaut == "heure_de_plus":
            dernier = (pd.Timestamp(photo) + pd.Timedelta(hours=1)).to_period("M") - 1
        else:
            dernier = pd.Timestamp(photo).to_period("M") - (0 if defaut == "mois_en_cours" else 1)
        cohorte = abonnements["debut"].dt.to_period("M")
        mois_fin = abonnements["fin"].dt.to_period("M")
        garde = cohorte <= dernier
        cohorte, mois_fin = cohorte[garde], mois_fin[garde]
        cohortes = pd.PeriodIndex(sorted(cohorte.unique()), freq="M", name="cohorte")
        ages = list(range((dernier - cohortes[0]).n + 1))
        table = pd.DataFrame(np21.nan, index=cohortes, columns=ages)
        for c in cohortes:
            fins = mois_fin[cohorte == c]
            encore = len(fins)
            for a in ages:
                m = c + a
                if m > dernier:
                    if defaut == "zero_futur":
                        table.loc[c, a] = 0.0
                        continue
                    break
                if defaut == "nat_oublie":
                    restent = fins > m
                elif defaut == "fin_incluse":
                    restent = fins.isna() | (fins >= m)
                else:
                    restent = fins.isna() | (fins > m)
                if defaut == "conditionnel":
                    table.loc[c, a] = restent.sum() / encore if encore else np21.nan
                    encore = int(restent.sum())
                else:
                    table.loc[c, a] = restent.sum() / len(fins)
        if defaut == "colonnes_mois":
            table.columns = [cohortes[0] + a for a in ages]
        if defaut == "index_texte":
            table.index = table.index.astype(str)
        return table

    return table_retention


@pytest.mark.parametrize("mutant,mots", [
    (_retention_21("zero_futur"), ("cohorte 2024-03, âge 1 : 0 au lieu de NaN", "PAS ENCORE TERMINÉS", "triangle")),
    (_retention_21("mois_en_cours"), ("colonne de trop", "MOINS 1")),
    (_retention_21("conditionnel"), ("ENTIÈRE",)),
    (_retention_21("nat_oublie"), ("NaT", "isna()")),
    (_retention_21("fin_incluse"), ("PENDANT", "APRÈS")),
    (_retention_21("veille"), ("1er avril à minuit",)),
    (_retention_21("heure_de_plus"), ("31 mars à 23 h",)),
    (_retention_21("photo_texte"), ("pd.Timestamp(photo)",)),
    (_retention_21("colonnes_mois"), ("ÂGES",)),
    (_retention_21("index_texte"), ("se trient mal",)),
])
def test_serie_21_table_retention_erreurs_classiques(mutant, mots):
    _echec_21("table_retention", mutant, *mots)


# Défi 132 : plis_temporels


def _plis_21(defaut=None):
    def plis_temporels(mois, n_plis, horizon):
        if defaut != "sans_valueerror":
            if n_plis < 1 or horizon < 1:
                raise ValueError("paramètres")
        distincts = sorted(mois.unique())
        if defaut == "calendrier":
            distincts = list(pd.period_range(distincts[0], distincts[-1], freq="M"))
        if defaut != "sans_valueerror" and len(distincts) < n_plis:
            raise ValueError("pas assez de mois")
        if defaut == "trie":
            mois = mois.sort_values(kind="stable").reset_index(drop=True)
        positions = mois.index.to_numpy() if defaut == "etiquettes" else np21.arange(len(mois))
        plis = []
        for rang, t in enumerate(distincts[-n_plis:]):
            if defaut == "fuite":
                garde = mois < t
            elif defaut == "un_de_trop":
                garde = mois < t - horizon
            elif defaut == "mois_presents":
                rang_t = len(distincts) - n_plis + rang
                garde = mois <= distincts[rang_t - horizon] if rang_t - horizon >= 0 else mois < distincts[0]
            else:
                garde = mois <= t - horizon
            test, train = (mois == t).to_numpy(), garde.to_numpy()
            if defaut != "masques":
                test, train = positions[test], positions[train]
            if defaut != "sans_valueerror" and not train.any():
                raise ValueError("pas d'entraînement")
            if defaut == "non_trie":
                train = train[::-1]
            plis.append((train, test))
        return plis[::-1] if defaut == "ordre_plis" else plis

    return plis_temporels


@pytest.mark.parametrize("mutant,mots", [
    (_plis_21("fuite"), ("FUITE", "t - horizon")),
    (_plis_21("un_de_trop"), ("l'égalité est permise",)),
    (_plis_21("etiquettes"), ("ÉTIQUETTES", "np.arange(len(mois))")),
    (_plis_21("trie"), ("TRIÉES",)),
    (_plis_21("calendrier"), ("avril et mai manquent", "PRÉSENTS")),
    (_plis_21("mois_presents"), ("avril et mai manquent", "CALENDRIER")),
    (_plis_21("masques"), ("masque de booléens",)),
    (_plis_21("non_trie"), ("pas trié",)),
    (_plis_21("ordre_plis"), ("CHRONOLOGIQUE",)),
    (_plis_21("sans_valueerror"), ("ValueError",)),
])
def test_serie_21_plis_temporels_erreurs_classiques(mutant, mots):
    _echec_21("plis_temporels", mutant, *mots)


# Défi 133 : pont_mrr


def _pont_21(defaut=None):
    def pont_mrr(factures):
        if defaut != "sans_valueerror" and (factures["montant"] <= 0).any():
            raise ValueError("montant")
        if defaut == "modifie":
            factures["mois_texte"] = factures["mois"].astype(str)
        colonnes = ["mrr", "nouveau", "reactivation", "expansion", "contraction", "perdu"]
        if defaut in ("shift_global", "shift_client"):
            # La table LONGUE, décalée d'une ligne : globalement, ou client par client (sans calendrier).
            longue = factures.groupby(["client", "mois"], as_index=False)["montant"].sum().sort_values(["client", "mois"])
            if defaut == "shift_global":
                avant = longue["montant"].shift(fill_value=0.0)
                deja = pd.Series(False, index=longue.index)
            else:
                avant = longue.groupby("client")["montant"].shift(fill_value=0.0)
                deja = longue.groupby("client").cumcount() > 0
            b = longue["montant"]
            longue = longue.assign(
                nouveau=b.where((avant == 0) & ~deja, 0.0), reactivation=b.where((avant == 0) & deja, 0.0),
                expansion=(b - avant).where((avant > 0) & (b > avant), 0.0),
                contraction=(avant - b).where((avant > 0) & (b < avant), 0.0), perdu=0.0, mrr=b)
            pont = longue.groupby("mois")[colonnes].sum()
            return pont.reindex(pd.period_range(pont.index.min(), pont.index.max(), freq="M", name="mois"),
                                fill_value=0.0)
        if defaut == "sans_somme":
            factures = factures.drop_duplicates(["client", "mois"], keep="last")
        mois = pd.period_range(factures["mois"].min(), factures["mois"].max(), freq="M", name="mois")
        grille = factures.pivot_table(index="client", columns="mois", values="montant", aggfunc="sum", fill_value=0.0)
        if defaut != "sans_trou":
            grille = grille.reindex(columns=mois, fill_value=0.0)
        avant = grille.shift(1, axis=1, fill_value=0.0)
        deja = grille.cumsum(axis=1).shift(1, axis=1, fill_value=0.0) > 0
        if defaut == "reactivation_nouveau":
            deja = deja & False
        arrive = (avant == 0) & (grille > 0)
        signe = -1.0 if defaut == "negatifs" else 1.0
        pont = pd.DataFrame({
            "mrr": grille.sum(),
            "nouveau": grille.where(arrive & ~deja, 0.0).sum(),
            "reactivation": grille.where(arrive & deja, 0.0).sum(),
            "expansion": (grille - avant).where((avant > 0) & (grille > avant), 0.0).sum(),
            "contraction": signe * (avant - grille).where((grille > 0) & (grille < avant), 0.0).sum(),
            "perdu": signe * avant.where((avant > 0) & (grille == 0), 0.0).sum(),
        })
        if defaut == "index_texte":
            pont.index = pont.index.astype(str)
        return pont[colonnes]

    return pont_mrr


@pytest.mark.parametrize("mutant,mots", [
    (_pont_21("shift_global"), ("ligne précédente", "LUI-MÊME")),
    (_pont_21("shift_client"), ("PERDU en février", "CALENDAIRE")),
    (_pont_21("sans_somme"), ("deux lignes le MÊME mois", "aggfunc")),
    (_pont_21("reactivation_nouveau"), ("réactivation", "pas un nouveau")),
    (_pont_21("sans_trou"), ("février EXISTE",)),
    (_pont_21("negatifs"), ("NÉGATIFS",)),
    (_pont_21("sans_valueerror"), ("ValueError",)),
    (_pont_21("modifie"), ("MODIFIÉ",)),
    (_pont_21("index_texte"), ("se trient mal",)),
])
def test_serie_21_pont_mrr_erreurs_classiques(mutant, mots):
    _echec_21("pont_mrr", mutant, *mots)


@pytest.mark.parametrize("nom,fabrique", [("lire_dates", _lire_dates_21), ("nouveaux_par_mois", _nouveaux_21),
                                          ("mouvements_mensuels", _mouvements_21), ("table_retention", _retention_21),
                                          ("plis_temporels", _plis_21), ("pont_mrr", _pont_21)])
def test_serie_21_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Garde-fou : chaque version boguée ci-dessus ne diffère du juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_serie_21_les_tests_du_corrige_passent():
    import defis.corrige_serie_21 as corrige

    tests = [f for nom, f in list(vars(corrige).items()) if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_21_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.abonnements as abonnements
    import defis.serie_21 as eleve

    for nom in ("abonnements_exemple", "dates_saisies", "factures_exemple", "generer_abonnements", "generer_factures"):
        assert getattr(eleve, nom) is getattr(abonnements, nom), nom


def test_serie_21_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    from defis.checks_serie_21 import SERIE

    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6


def test_serie_21_les_exemples_sont_neufs_a_chaque_appel():
    """Chaque appel renvoie une NOUVELLE table : la modifier n'abîme pas l'exemple suivant."""
    for fabrique in (abonnements_21, dates_21, factures_21):
        premiere = fabrique()
        assert fabrique() is not premiere
        premiere.iloc[0:1] = None
        assert not fabrique().iloc[:1].isna().all(axis=None)


def test_serie_21_les_generateurs_sont_reproductibles_et_realistes():
    a, b = generer_abonnements_21(300, graine=4), generer_abonnements_21(300, graine=4)
    pd.testing.assert_frame_equal(a, b)
    assert not (a["fin"] < a["debut"]).any()
    assert 0.2 < a["fin"].isna().mean() < 0.7                                # des abonnés toujours là
    assert (a["debut"].dt.day == 1).mean() > 0.15                            # des arrivées le 1er du mois
    assert (a["debut"].dt.hour == 23).any() and (a["fin"].dt.day == 1).any()  # 23 h 30, départs le 1er
    f = generer_factures_21(60, graine=4)
    pd.testing.assert_frame_equal(f, generer_factures_21(60, graine=4))
    assert f.duplicated(["client", "mois"]).any()                             # des options sur une ligne à part
    assert (f["montant"] > 0).all() and str(f["mois"].dtype) == "period[M]"


def test_serie_21_les_exemples_des_docstrings_sont_justes():
    """Les exemples de defis/serie_21.py et defis/abonnements.py, rejoués avec le corrigé."""
    import defis.serie_21 as eleve
    from defis.corrige_serie_21 import (lire_dates, mouvements_mensuels, nouveaux_par_mois, plis_temporels, pont_mrr,
                                        table_retention)

    def lignes(objet):
        return [ligne.rstrip() for ligne in str(objet).splitlines()]

    def dans_la_docstring(fonction, objet):
        doc = [ligne.strip() for ligne in fonction.__doc__.splitlines()]
        sortie = [ligne.strip() for ligne in lignes(objet)]
        debut = doc.index(sortie[0])
        assert doc[debut: debut + len(sortie)] == sortie, (doc[debut: debut + len(sortie)], sortie)

    dates = lire_dates(dates_21())
    assert (dates.loc[101], dates.loc[103]) == (pd.Timestamp("2024-01-03"), pd.Timestamp("2024-02-15"))
    with pytest.raises(ValueError, match='time data "2024-01-03" doesn\'t match format "%d/%m/%Y"'):
        lire_dates(pd.Series(["2024-01-03"]))
    dans_la_docstring(eleve.nouveaux_par_mois, nouveaux_par_mois(abonnements_21()["debut"], "2024-01", "2024-05"))
    dans_la_docstring(eleve.mouvements_mensuels, mouvements_mensuels(abonnements_21(), "2024-01", "2024-04"))
    dans_la_docstring(eleve.table_retention, table_retention(abonnements_21(), "2024-04-15").round(2))
    dans_la_docstring(eleve.pont_mrr, pont_mrr(factures_21()))
    mois = pd.Series(pd.PeriodIndex(["2024-03", "2024-01", "2024-02", "2024-03", "2024-01"], freq="M"),
                     index=[10, 20, 30, 40, 50])
    assert repr(plis_temporels(mois, n_plis=1, horizon=2)) == "[(array([1, 4]), array([0, 3]))]"
    assert repr(plis_temporels(mois, n_plis=2, horizon=1)) == \
        "[(array([1, 4]), array([2])), (array([1, 2, 4]), array([0, 3]))]"
    assert pd.to_datetime(pd.Series(["03/01/2024"]))[0] == pd.Timestamp("2024-03-01")    # le piège annoncé
    debut = abonnements_21()["debut"].dt.to_period("M")
    assert (debut[0], debut[0] + 2, (debut[7] - debut[0]).n) == (pd.Period("2023-11", "M"), pd.Period("2024-01", "M"), 4)
    import defis.abonnements as module
    dans_la_docstring(module, abonnements_21().head(3))


def test_serie_21_autres_ecritures_justes_acceptees():
    def lire_dates(textes):                     # date par date, avec le module datetime
        def une(texte):
            if not isinstance(texte, str) or not texte.strip():
                return pd.NaT
            return dt21.datetime.strptime(texte.strip(), "%d/%m/%Y")
        return pd.to_datetime(textes.map(une))

    def nouveaux_par_mois(debuts, premier, dernier):   # groupby sur (année, mois), puis un calendrier
        valides = debuts.dropna()
        comptes = valides.groupby([valides.dt.year, valides.dt.month]).size()
        mois = pd.period_range(premier, dernier, freq="M", name="mois")
        return pd.Series([int(comptes.get((m.year, m.month), 0)) for m in mois], index=mois, name="nouveaux")

    def mouvements_mensuels(abonnements, premier, dernier):   # des dates plutôt que des Period
        debut, fin = abonnements["debut"], abonnements["fin"]
        if ((fin < debut) & fin.notna()).any():
            raise ValueError("fin avant début")
        mois = pd.period_range(premier, dernier, freq="M", name="mois")
        lignes = {}
        for m in mois:
            ouverture, cloture = m.start_time, (m + 1).start_time
            present = (debut < ouverture) & ~(fin < ouverture)          # ~(NaT < x) est vrai : NaT reste
            arrive = (debut >= ouverture) & (debut < cloture)
            part = (fin >= ouverture) & (fin < cloture)
            n = int(present.sum())
            lignes[m] = {"actifs_debut": n, "nouveaux": int(arrive.sum()), "partis": int(part.sum()),
                         "actifs_fin": n + int(arrive.sum()) - int(part.sum()),
                         "taux_churn": (present & part).sum() / n if n else float("nan")}
        return pd.DataFrame.from_dict(lignes, orient="index").rename_axis("mois").set_axis(mois)

    def table_retention(abonnements, photo):   # vectorisé : l'âge du départ, en mois
        dernier = pd.Timestamp(photo).to_period("M") - 1
        table = abonnements.assign(cohorte=abonnements["debut"].dt.to_period("M"),
                                   fin_m=abonnements["fin"].dt.to_period("M"))
        table = table[table["cohorte"] <= dernier]
        premiere = table["cohorte"].min()
        ages = range((dernier - premiere).n + 1)
        age_depart = [((f - c).n if not pd.isna(f) else math21.inf) for c, f in zip(table["cohorte"], table["fin_m"])]
        table = table.assign(age_depart=age_depart)
        lignes = {}
        for c, groupe in table.groupby("cohorte"):
            lignes[c] = [(groupe["age_depart"] > a).mean() if c + a <= dernier else float("nan") for a in ages]
        resultat = pd.DataFrame.from_dict(lignes, orient="index", columns=list(ages))
        resultat.index = pd.PeriodIndex(resultat.index, freq="M", name="cohorte")
        return resultat

    def plis_temporels(mois, n_plis, horizon):   # np.flatnonzero et un tuple
        if min(n_plis, horizon) < 1 or mois.nunique() < n_plis:
            raise ValueError("paramètres")
        plis = []
        for t in sorted(set(mois))[-n_plis:]:
            train = np21.flatnonzero((mois <= t - horizon).to_numpy())
            if train.size == 0:
                raise ValueError("pas d'entraînement")
            plis.append((train, np21.flatnonzero((mois == t).to_numpy())))
        return plis

    def pont_mrr(factures):   # groupby + unstack, et des boucles sur les mois
        if (factures["montant"] <= 0).any():
            raise ValueError("montant")
        mois = pd.period_range(factures["mois"].min(), factures["mois"].max(), freq="M", name="mois")
        grille = factures.groupby(["client", "mois"])["montant"].sum().unstack(fill_value=0.0)
        grille = grille.reindex(columns=mois, fill_value=0.0)
        lignes, deja = [], pd.Series(False, index=grille.index)
        precedent = pd.Series(0.0, index=grille.index)
        for m in mois:
            b, a = grille[m], precedent
            lignes.append({"mrr": b.sum(), "nouveau": b[(a == 0) & (b > 0) & ~deja].sum(),
                           "reactivation": b[(a == 0) & (b > 0) & deja].sum(),
                           "expansion": (b - a)[(a > 0) & (b > a)].sum(),
                           "contraction": (a - b)[(b > 0) & (b < a)].sum(), "perdu": a[(a > 0) & (b == 0)].sum()})
            deja, precedent = deja | (b > 0), b
        return pd.DataFrame(lignes, index=mois).astype(float)

    for nom, fonction in [("lire_dates", lire_dates), ("nouveaux_par_mois", nouveaux_par_mois),
                          ("mouvements_mensuels", mouvements_mensuels), ("table_retention", table_retention),
                          ("plis_temporels", plis_temporels), ("pont_mrr", pont_mrr)]:
        resultat = _evaluer(nom, fonction)
        assert resultat.statut == REUSSI, (nom, resultat.message)


# --------------------------------------------------------------------------- Série 22 : briques scikit-learn

import numbers as numbers22  # noqa: E402

import numpy as np22  # noqa: E402
from sklearn.base import BaseEstimator as BaseEstimator22  # noqa: E402
from sklearn.base import ClassifierMixin as ClassifierMixin22  # noqa: E402
from sklearn.base import TransformerMixin as TransformerMixin22  # noqa: E402
from sklearn.base import clone as clone22  # noqa: E402
from sklearn.base import is_classifier as is_classifier22  # noqa: E402
from sklearn.metrics import make_scorer as make_scorer22  # noqa: E402
from sklearn.model_selection import KFold as KFold22  # noqa: E402
from sklearn.model_selection import StratifiedKFold as StratifiedKFold22  # noqa: E402
from sklearn.model_selection import cross_val_predict as cross_val_predict22  # noqa: E402
from sklearn.utils.validation import check_is_fitted as check_is_fitted22  # noqa: E402

from defis.briques import clients_exemple as clients_22  # noqa: E402
from defis.briques import generer_clients as generer_clients_22  # noqa: E402


def _echec_22(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


def _entier_22(m, minimum):
    return not isinstance(m, bool) and isinstance(m, numbers22.Integral) and m >= minimum


# Défi 134 : ImputeurMediane


def _imputeur_22(defaut=None):
    bases = (BaseEstimator22,) if defaut == "sans_mixin" else (TransformerMixin22, BaseEstimator22)

    def fit(self, X, y=None):
        if defaut == "moyenne":
            medianes = X.mean()
        elif defaut == "np_median":
            medianes = pd.Series({c: np22.median(X[c].to_numpy(dtype=float)) for c in X.columns})
        elif defaut == "commune":
            medianes = pd.Series(np22.nanmedian(X.to_numpy(dtype=float)), index=X.columns)
        elif defaut == "colonnes_trouees":
            medianes = X.loc[:, X.isna().any()].median()
        else:
            medianes = X.median()
        if medianes.isna().any() and defaut not in ("np_median", "colonne_vide_acceptee"):
            raise ValueError("colonne vide")
        if defaut == "garde_premier" and hasattr(self, "medianes_"):
            return self
        if defaut == "sans_underscore":
            self.medianes = medianes
        else:
            self.medianes_ = medianes
        return None if defaut == "sans_return" else self

    def transform(self, X):
        if defaut not in ("sans_underscore", "sans_check"):
            check_is_fitted22(self)
        medianes = self.medianes if defaut == "sans_underscore" else self.medianes_
        if defaut not in ("sans_controle_colonnes", "colonnes_trouees") and set(X.columns) != set(medianes.index):
            raise ValueError("colonnes")
        if defaut == "mediane_test":
            return X.fillna(X.median())
        if defaut == "zero":
            return X.fillna(0)
        if defaut == "modifie":
            X.fillna(medianes, inplace=True)
            return X
        resultat = X.fillna(medianes)
        return resultat.to_numpy() if defaut == "tableau" else resultat

    return type("ImputeurMediane", bases, {"fit": fit, "transform": transform})


@pytest.mark.parametrize("mutant,mots", [
    (_imputeur_22("mediane_test"), ("médiane du TEST",)),
    (_imputeur_22("moyenne"), ("MOYENNE", "300 €")),
    (_imputeur_22("zero"), ("avec 0",)),
    (_imputeur_22("np_median"), ("np.median", "X.median()")),
    (_imputeur_22("commune"), ("MÊME valeur",)),
    (_imputeur_22("colonnes_trouees"), ("'appels'", "TOUTES les colonnes")),
    (_imputeur_22("sans_return"), ("return self",)),
    (_imputeur_22("modifie"), ("MODIFIÉ",)),
    (_imputeur_22("sans_underscore"), ("FINIT PAR _",)),
    (_imputeur_22("sans_check"), ("AVANT fit", "check_is_fitted(self)")),
    (_imputeur_22("sans_controle_colonnes"), ("'facture'", "ValueError")),
    (_imputeur_22("colonne_vide_acceptee"), ("aucune valeur", "ValueError")),
    (_imputeur_22("garde_premier"), ("PREMIER train",)),
    (_imputeur_22("tableau"), ("tableau numpy",)),
    (_imputeur_22("sans_mixin"), ("TransformerMixin",)),
])
def test_serie_22_imputeur_erreurs_classiques(mutant, mots):
    _echec_22("ImputeurMediane", mutant, *mots)


# Défi 135 : RegroupeurRares


def _regroupeur_22(defaut=None):
    def __init__(self, min_effectif=10, libelle="Autre"):
        if defaut == "verif_init" and not _entier_22(min_effectif, 1):
            raise ValueError("min_effectif")
        if defaut == "autre_nom":
            self.seuil = min_effectif
        else:
            self.min_effectif = int(min_effectif) if defaut == "conversion" else min_effectif
        if defaut == "copie_privee":
            self._m = min_effectif
        self.libelle = libelle

    def __init_kwargs__(self, **parametres):
        self.min_effectif = parametres.get("min_effectif", 10)
        self.libelle = parametres.get("libelle", "Autre")

    def fit(self, X, y=None):
        m = self._m if defaut == "copie_privee" else (self.seuil if defaut == "autre_nom" else self.min_effectif)
        if defaut == "bool_accepte":
            valide = isinstance(m, numbers22.Integral) and m >= 1
        elif defaut == "int_seulement":
            valide = isinstance(m, int) and not isinstance(m, bool) and m >= 1
        else:
            valide = _entier_22(m, 1)
        if defaut not in ("sans_validation", "verif_init") and not valide:
            raise ValueError("min_effectif")
        if defaut == "cumule" and hasattr(self, "modalites_"):
            modalites = self.modalites_
        else:
            modalites = {}
        if defaut == "commun":
            communs = pd.concat([X[c] for c in X.columns]).value_counts()
        for colonne in X.columns:
            comptes = communs if defaut == "commun" else X[colonne].value_counts()
            gardees = set(comptes.index[comptes > m] if defaut == "strict" else comptes.index[comptes >= m])
            modalites[colonne] = modalites.get(colonne, set()) | gardees
        self.modalites_ = modalites
        self.m_ = m
        return self

    def transform(self, X):
        check_is_fitted22(self)
        resultat = X if defaut == "modifie" else X.copy()
        for colonne, gardees in self.modalites_.items():
            if defaut == "compte_test":
                comptes = X[colonne].value_counts()
                gardees = set(comptes.index[comptes >= self.m_])
            garde = X[colonne].isin(gardees)
            if defaut != "nan_autre":
                garde = garde | X[colonne].isna()
            resultat[colonne] = X[colonne].where(garde, "Autre" if defaut == "libelle_fixe" else self.libelle)
        return resultat

    corps = {"fit": fit, "transform": transform,
             "__init__": __init_kwargs__ if defaut == "kwargs" else __init__}
    return type("RegroupeurRares", (TransformerMixin22, BaseEstimator22), corps)


@pytest.mark.parametrize("mutant,mots", [
    (_regroupeur_22("verif_init"), ("dès la construction", "set_params")),
    (_regroupeur_22("conversion"), ("dès la construction",)),
    (_regroupeur_22("autre_nom"), ("SON nom",)),
    (_regroupeur_22("kwargs"), ("**kwargs",)),
    (_regroupeur_22("copie_privee"), ("set_params",)),
    (_regroupeur_22("nan_autre"), ("VIDE", "isna()")),
    (_regroupeur_22("strict"), ("Lyon", "`>=`")),
    (_regroupeur_22("compte_test"), ("TRANSFORM",)),
    (_regroupeur_22("commun"), ("TOUTES les colonnes",)),
    (_regroupeur_22("sans_validation"), ("min_effectif=0", "ValueError")),
    (_regroupeur_22("bool_accepte"), ("min_effectif=True", "isinstance(True, int)")),
    (_regroupeur_22("int_seulement"), ("np.int64(3)", "numbers.Integral")),
    (_regroupeur_22("cumule"), ("fit PRÉCÉDENT",)),
    (_regroupeur_22("modifie"), ("MODIFIÉ",)),
    (_regroupeur_22("libelle_fixe"), ("self.libelle",)),
])
def test_serie_22_regroupeur_erreurs_classiques(mutant, mots):
    _echec_22("RegroupeurRares", mutant, *mots)


# Défi 136 : scoreur_cout


def _scoreur_22(defaut=None):
    def scoreur_cout(cout_fn, cout_fp, seuil=0.5, positif=1):
        def verifier():
            for cout in (cout_fn, cout_fp):
                if isinstance(cout, bool) or not isinstance(cout, numbers22.Real) or not cout >= 0:
                    raise ValueError("coût")
            if not 0 <= seuil <= 1:
                raise ValueError("seuil")

        if defaut not in ("sans_validation", "validation_tardive"):
            verifier()

        def scoreur(estimateur, X, y):
            if defaut == "validation_tardive":
                verifier()
            if defaut == "fit":
                estimateur.fit(X, y)
            classes = list(estimateur.classes_)
            if defaut == "indexerror":
                colonne = int(np22.flatnonzero(np22.asarray(classes) == positif)[0])
            else:
                colonne = classes.index(positif)
            if defaut == "predict":
                alerte = np22.asarray(estimateur.predict(X)) == positif
            else:
                proba = estimateur.predict_proba(X)[:, 1 if defaut == "colonne1" else colonne]
                alerte = proba > seuil if defaut == "strict" else proba >= seuil
            reel = (y.values if defaut == "y_values" else np22.asarray(y)) == positif
            fn, fp = (cout_fp, cout_fn) if defaut == "inverse" else (cout_fn, cout_fp)
            cout = fn * np22.sum(reel & ~alerte) + fp * np22.sum(~reel & alerte)
            if defaut == "total":
                return -float(cout)
            return float(cout / len(reel)) if defaut == "signe" else -float(cout / len(reel))

        return None if defaut == "oublie_return" else scoreur

    return scoreur_cout


@pytest.mark.parametrize("mutant,mots", [
    (_scoreur_22("signe"), ("POSITIF", "MAXIMISE")),
    (_scoreur_22("total"), ("TOTAL",)),
    (_scoreur_22("strict"), ("0,30 pile", "`>=`")),
    (_scoreur_22("predict"), ("predict", "0,5")),
    (_scoreur_22("colonne1"), ("classes_ = ['part', 'reste']", "colonne 0")),
    (_scoreur_22("inverse"), ("inversés",)),
    (_scoreur_22("fit"), ("DÉJÀ entraîné",)),
    (_scoreur_22("y_values"), ("AttributeError", "np.asarray(y)")),
    (_scoreur_22("indexerror"), ("IndexError", "ValueError")),
    (_scoreur_22("sans_validation"), ("scoreur_cout(-500, 50)", "dès l'appel")),
    (_scoreur_22("validation_tardive"), ("dès l'appel",)),
    (_scoreur_22("oublie_return"), ("return scoreur",)),
])
def test_serie_22_scoreur_erreurs_classiques(mutant, mots):
    _echec_22("scoreur_cout", mutant, *mots)


# Défi 137 : EncodeurCible


def _encodeur_22(defaut=None):
    def __init__(self, lissage=10.0):
        if defaut == "verif_init" and (isinstance(lissage, bool) or not lissage >= 0):
            raise ValueError("lissage")
        self.lissage = lissage

    def fit(self, X, y=None):
        lissage = self.lissage
        if defaut not in ("verif_init", "sans_validation_lissage") and (
                isinstance(lissage, bool) or not isinstance(lissage, numbers22.Real) or not lissage >= 0):
            raise ValueError("lissage")
        if y is None and defaut != "sans_y_check":
            raise ValueError("y")
        if defaut != "par_index":
            y = np22.asarray(y, dtype=float)
            if defaut == "sans_longueur":
                y = y[: len(X)]
            elif len(y) != len(X):
                raise ValueError("longueur")
        self.moyenne_ = float(np22.mean(y))
        self.encodages_ = {}
        for colonne in X.columns:
            valeurs = X[colonne].fillna("(vide)") if defaut == "nan_modalite" else X[colonne]
            if defaut == "par_index":
                stats = y.astype(float).groupby(valeurs).agg(["sum", "count"])
            else:
                stats = pd.Series(y).groupby(valeurs.to_numpy()).agg(["sum", "count"])
            moyenne = float((stats["sum"] / stats["count"]).mean()) if defaut == "moyenne_des_taux" else self.moyenne_
            if defaut == "moyenne_des_taux":
                self.moyenne_ = moyenne
            if defaut == "brut":
                code = stats["sum"] / stats["count"]
            elif defaut == "poids_inverses":
                code = (stats["count"] * moyenne + lissage * stats["sum"] / stats["count"]) / (stats["count"] + lissage)
            else:
                code = (stats["sum"] + lissage * moyenne) / (stats["count"] + lissage)
            self.encodages_[colonne] = code
        return self

    def transform(self, X):
        check_is_fitted22(self)
        resultat = X if defaut == "modifie" else pd.DataFrame(index=X.index)
        for colonne, encodage in self.encodages_.items():
            valeurs = X[colonne].fillna("(vide)") if defaut == "nan_modalite" else X[colonne]
            code = valeurs.map(encodage).astype(float)
            if defaut == "inconnu_zero":
                code = code.fillna(0.0)
            elif defaut != "inconnu_nan":
                code = code.fillna(self.moyenne_)
            resultat[colonne] = code
        return resultat

    return type("EncodeurCible", (TransformerMixin22, BaseEstimator22),
                {"__init__": __init__, "fit": fit, "transform": transform})


@pytest.mark.parametrize("mutant,mots", [
    (_encodeur_22("brut"), ("BRUT",)),
    (_encodeur_22("poids_inverses"), ("à l'envers",)),
    (_encodeur_22("moyenne_des_taux"), ("chaque CLIENT compte autant",)),
    (_encodeur_22("inconnu_nan"), ("Quimper", "NaN", "fillna(self.moyenne_)")),
    (_encodeur_22("inconnu_zero"), ("donne 0",)),
    (_encodeur_22("nan_modalite"), ("case vide", "pas une modalité")),
    (_encodeur_22("par_index"), ("PAR POSITION",)),
    (_encodeur_22("sans_y_check"), ("sans y", "ValueError")),
    (_encodeur_22("sans_longueur"), ("une ligne de plus",)),
    (_encodeur_22("verif_init"), ("dès la construction",)),
    (_encodeur_22("sans_validation_lissage"), ("lissage=-1",)),
    (_encodeur_22("modifie"), ("MODIFIÉ",)),
])
def test_serie_22_encodeur_erreurs_classiques(mutant, mots):
    _echec_22("EncodeurCible", mutant, *mots)


# Défi 138 : EncodeurCibleHorsPli


def _hors_pli_22(defaut=None):
    from defis.corrige_serie_22 import EncodeurCible

    parent = BaseEstimator22 if defaut == "pas_heritage" else EncodeurCible

    def __init__(self, lissage=10.0, n_plis=5, graine=0):
        if defaut != "sans_super":
            self.lissage = lissage
        self.n_plis = n_plis
        self.graine = graine

    def fit_transform(self, X, y=None):
        if defaut != "sans_fit":
            self.fit(X, y)
        y = np22.asarray(y, dtype=float)
        resultat = pd.DataFrame(np22.nan, index=X.index, columns=list(X.columns), dtype=float)
        if defaut == "sans_melange":
            plis = KFold22(n_splits=self.n_plis).split(X)
        elif defaut == "stratifie":
            plis = StratifiedKFold22(n_splits=self.n_plis, shuffle=True, random_state=self.graine).split(X, y)
        else:
            graine = 0 if defaut == "graine_ignoree" else self.graine
            plis = KFold22(n_splits=self.n_plis, shuffle=True, random_state=graine).split(X)
        moyenne = float(y.mean())
        for train, test in plis:
            encodeur = EncodeurCible(lissage=self.lissage).fit(X.iloc[train], y[train])
            if defaut == "moyenne_globale":
                encodeur.moyenne_ = moyenne
                encodeur.fit(X.iloc[train], y[train])
                for colonne in X.columns:
                    stats = pd.Series(y[train]).groupby(X[colonne].iloc[train].to_numpy()).agg(["sum", "count"])
                    encodeur.encodages_[colonne] = (stats["sum"] + self.lissage * moyenne) / (stats["count"] + self.lissage)
                encodeur.moyenne_ = moyenne
            if defaut == "loc":
                resultat.loc[test] = encodeur.transform(X.iloc[test]).to_numpy()
            else:
                resultat.iloc[test] = encodeur.transform(X.iloc[test]).to_numpy()
            if defaut == "dernier_pli":
                self.encodages_, self.moyenne_ = encodeur.encodages_, encodeur.moyenne_
        return resultat

    corps = {"__init__": __init__}
    if defaut == "pas_heritage":
        corps.update({"fit": EncodeurCible.fit, "transform": EncodeurCible.transform})
    if defaut != "naif":
        corps["fit_transform"] = fit_transform
    if defaut == "pas_heritage":
        return type("EncodeurCibleHorsPli", (TransformerMixin22, BaseEstimator22), corps)
    return type("EncodeurCibleHorsPli", (parent,), corps)


@pytest.mark.parametrize("mutant,mots", [
    (_hors_pli_22("naif"), ("SA propre réponse", "Redéfinissez `fit_transform`")),
    (_hors_pli_22("moyenne_globale"), ("TOUT y",)),
    (_hors_pli_22("sans_melange"), ("pas mélangés", "shuffle=True")),
    (_hors_pli_22("stratifie"), ("stratifiés", "pas StratifiedKFold")),
    (_hors_pli_22("graine_ignoree"), ("graine=3", "self.graine")),
    (_hors_pli_22("sans_fit"), ("NotFittedError", "self.fit(X, y)")),
    (_hors_pli_22("dernier_pli"), ("TOUT le train", "dernier pli")),
    (_hors_pli_22("loc"), ("POSITIONS", "iloc")),
    (_hors_pli_22("sans_super"), ("SON nom",)),
    (_hors_pli_22("pas_heritage"), ("HÉRITER",)),
])
def test_serie_22_hors_pli_erreurs_classiques(mutant, mots):
    _echec_22("EncodeurCibleHorsPli", mutant, *mots)


def test_serie_22_hors_pli_le_leave_one_out_est_reconnu():
    from defis.checks_serie_22 import _ref_hors_pli
    from defis.corrige_serie_22 import EncodeurCible

    class EncodeurCibleHorsPli(EncodeurCible):
        def __init__(self, lissage=10.0, n_plis=5, graine=0):
            super().__init__(lissage=lissage)
            self.n_plis, self.graine = n_plis, graine

        def fit_transform(self, X, y=None):
            self.fit(X, y)
            return _ref_hors_pli(X, y, self.lissage, variante="loo")

    _echec_22("EncodeurCibleHorsPli", EncodeurCibleHorsPli, "leave-one-out")


# Défi 139 : ClassifieurSeuil


def _seuil_22(defaut=None):
    bases = (BaseEstimator22, ClassifierMixin22) if defaut == "ordre_mixins" else (ClassifierMixin22, BaseEstimator22)

    def __init__(self, estimateur, cout_fn=500.0, cout_fp=50.0, n_plis=5, graine=0):
        if defaut == "verif_init" and cout_fn < 0:
            raise ValueError("coût")
        self.estimateur, self.cout_fn, self.cout_fp, self.n_plis, self.graine = estimateur, cout_fn, cout_fp, n_plis, graine

    def fit(self, X, y):
        if defaut not in ("sans_validation", "verif_init"):
            if min(self.cout_fn, self.cout_fp) < 0 or not _entier_22(self.n_plis, 2):
                raise ValueError("paramètres")
        y = np22.asarray(y)
        self.classes_ = np22.unique(y)
        if len(self.classes_) != 2 and defaut != "sans_validation":
            raise ValueError("classes")
        if defaut == "train":
            proba = clone22(self.estimateur).fit(X, y).predict_proba(X)[:, 1]
        else:
            if defaut == "plis_par_defaut":
                plis = self.n_plis
            elif defaut == "kfold":
                plis = KFold22(n_splits=self.n_plis, shuffle=True, random_state=self.graine)
            else:
                plis = StratifiedKFold22(n_splits=self.n_plis, shuffle=True, random_state=self.graine)
            proba = cross_val_predict22(clone22(self.estimateur), X, y, cv=plis, method="predict_proba")[:, 1]
        reel = y == self.classes_[1]
        fn, fp = (self.cout_fp, self.cout_fn) if defaut == "inverse" else (self.cout_fn, self.cout_fp)
        grille = np22.arange(1, 100) / 100
        couts = np22.array([fn * np22.sum(reel & ~((proba > s) if defaut == "strict" else (proba >= s)))
                            + fp * np22.sum(~reel & ((proba > s) if defaut == "strict" else (proba >= s)))
                            for s in grille])
        if defaut == "plus_grand":
            self.seuil_ = float(grille[len(couts) - 1 - int(np22.argmin(couts[::-1]))])
        else:
            self.seuil_ = float(grille[int(np22.argmin(couts))])
        if defaut == "sans_clone":
            self.estimateur_ = self.estimateur.fit(X, y)
        elif defaut == "dernier_pli":
            train, _ = list(StratifiedKFold22(n_splits=self.n_plis, shuffle=True, random_state=self.graine)
                            .split(X, y))[-1]
            self.estimateur_ = clone22(self.estimateur).fit(X.iloc[train], y[train])
        else:
            self.estimateur_ = clone22(self.estimateur).fit(X, y)
        return self

    def predict_proba(self, X):
        if defaut != "sans_check":
            check_is_fitted22(self)
        return self.estimateur_.predict_proba(X)

    def predict(self, X):
        if defaut == "predict_05":
            check_is_fitted22(self)
            return self.estimateur_.predict(X)
        positif = self.predict_proba(X)[:, 1] >= self.seuil_
        if defaut == "etiquettes_01":
            return positif.astype(int)
        return self.classes_[positif.astype(int)]

    return type("ClassifieurSeuil", bases, {"__init__": __init__, "fit": fit, "predict_proba": predict_proba,
                                             "predict": predict})


@pytest.mark.parametrize("mutant,mots", [
    (_seuil_22("train"), ("PROPRES données", "HORS PLI")),
    (_seuil_22("plis_par_defaut"), ("pas mélangés", "StratifiedKFold")),
    (_seuil_22("kfold"), ("pas stratifiés",)),
    (_seuil_22("inverse"), ("inversés",)),
    (_seuil_22("sans_clone"), ("ENTRAÎNÉ l'estimateur reçu", "clone")),
    (_seuil_22("dernier_pli"), ("TOUT le train",)),
    (_seuil_22("predict_05"), ("seuil de 0,5", "self.seuil_")),
    (_seuil_22("etiquettes_01"), ("ÉTIQUETTES",)),
    pytest.param(_seuil_22("ordre_mixins"), ("À GAUCHE",), marks=pytest.mark.skipif(
        is_classifier22(type("Essai", (BaseEstimator22, ClassifierMixin22), {})()),
        reason="avant scikit-learn 1.6, l'ordre des mixins ne change pas is_classifier")),
    (_seuil_22("sans_validation"), ("ValueError",)),
    (_seuil_22("sans_check"), ("AVANT fit", "check_is_fitted")),
    (_seuil_22("verif_init"), ("dès la construction",)),
])
def test_serie_22_seuil_erreurs_classiques(mutant, mots):
    _echec_22("ClassifieurSeuil", mutant, *mots)


def test_serie_22_seuil_le_strict_et_l_egalite_sont_reconnus():
    """`>` au lieu de `>=` et « le plus grand seuil » ne changent le seuil que sur des égalités : on les force."""
    from defis.checks_serie_22 import _verifier_seuil

    class Factice(ClassifierMixin22, BaseEstimator22):
        def fit(self, X, y):
            self.classes_ = np22.unique(y)
            return self

        def predict_proba(self, X):
            p = np22.asarray(X["p"], dtype=float)
            return np22.column_stack([1 - p, p])

        def predict(self, X):
            return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    X = pd.DataFrame({"p": [0.2, 0.2, 0.2, 0.2, 0.6, 0.6, 0.6, 0.6, 0.9, 0.9]})
    y = np22.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    with pytest.raises(AssertionError, match="PLUS PETIT"):
        _verifier_seuil(0.6, 0.21, "fit", Factice(), X, y, cout_fn=1.0, cout_fp=1.0, n_plis=2)
    with pytest.raises(AssertionError, match="`>=`"):
        _verifier_seuil(0.2, 0.21, "fit", Factice(), X, y, cout_fn=1.0, cout_fp=1.0, n_plis=2)


@pytest.mark.parametrize("nom,fabrique", [("ImputeurMediane", _imputeur_22), ("RegroupeurRares", _regroupeur_22),
                                          ("scoreur_cout", _scoreur_22), ("EncodeurCible", _encodeur_22),
                                          ("EncodeurCibleHorsPli", _hors_pli_22), ("ClassifieurSeuil", _seuil_22)])
def test_serie_22_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Garde-fou : chaque version boguée ci-dessus ne diffère du juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_serie_22_les_tests_du_corrige_passent():
    import defis.corrige_serie_22 as corrige

    tests = [f for nom, f in list(vars(corrige).items()) if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_22_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.briques as briques
    import defis.serie_22 as eleve

    for nom in ("clients_exemple", "generer_clients", "NUMERIQUES", "CATEGORIES"):
        assert getattr(eleve, nom) is getattr(briques, nom), nom


def test_serie_22_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    from defis.checks_serie_22 import SERIE

    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6


def test_serie_22_les_exemples_sont_neufs_a_chaque_appel():
    premiere = clients_22()
    premiere.iloc[0:1] = None
    assert not clients_22().iloc[:1].isna().all(axis=None)
    assert generer_clients_22(50) is not generer_clients_22(50)


def test_serie_22_le_generateur_est_reproductible_et_realiste():
    a = generer_clients_22(800, graine=4)
    pd.testing.assert_frame_equal(a, generer_clients_22(800, graine=4))
    assert 0.2 < a["churn"].mean() < 0.5
    assert 0.02 < a["anciennete"].isna().mean() < 0.1 and 0.01 < a["ville"].isna().mean() < 0.07
    comptes = a["ville"].value_counts()
    assert comptes.iloc[0] > 100 and (comptes <= 5).sum() >= 5                   # la longue traîne
    assert list(a.index[:2]) == ["G0001", "G0002"]
    # le churn dépend vraiment de la ville : l'encodage par la cible a quelque chose à trouver
    taux = a.groupby("ville")["churn"].agg(["mean", "size"])
    assert taux.loc[taux["size"] >= 30, "mean"].std() > 0.05


def test_serie_22_l_exemple_a_tous_les_pieges_annonces():
    c = clients_22()
    assert c[["anciennete", "facture"]].isna().sum().tolist() == [2, 1] and not c["appels"].isna().any()
    assert c["facture"].mean() > 80 and c["facture"].median() == 60
    villes = c["ville"].value_counts()
    assert villes["Paris"] == 5 and villes["Lyon"] == 3 and c["ville"].isna().sum() == 1
    assert (villes == 1).sum() == 3 and c.loc[c["ville"] == "Albi", "churn"].tolist() == [1]


def test_serie_22_les_exemples_des_docstrings_sont_justes():
    """Les exemples de defis/serie_22.py et defis/briques.py, rejoués avec le corrigé."""
    import doctest

    import defis.briques as briques
    import defis.corrige_serie_22 as corrige
    import defis.serie_22 as eleve

    contexte = dict(vars(eleve))
    for nom in ("ImputeurMediane", "RegroupeurRares", "scoreur_cout", "EncodeurCible", "EncodeurCibleHorsPli",
                "ClassifieurSeuil"):
        contexte[nom] = getattr(corrige, nom)
    lanceur = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    exemples = 0
    for module in (eleve, briques):
        for test in doctest.DocTestFinder().find(module):
            test.globs = dict(contexte)
            exemples += len(test.examples)
            lanceur.run(test)
    resultats = lanceur.summarize(verbose=False)
    assert resultats.failed == 0 and exemples >= 30, resultats


def test_serie_22_autres_ecritures_justes_acceptees():
    def scoreur_cout(cout_fn, cout_fp, seuil=0.5, positif=1):    # make_scorer, comme dans le commentaire du corrigé
        if min(cout_fn, cout_fp) < 0 or cout_fn != cout_fn or cout_fp != cout_fp or not 0 <= seuil <= 1:
            raise ValueError("paramètres")

        def cout_moyen(y, proba, pos_label, cout_fn, cout_fp, seuil):
            reel, alerte = np22.asarray(y) == pos_label, proba >= seuil
            return (cout_fn * np22.sum(reel & ~alerte) + cout_fp * np22.sum(~reel & alerte)) / len(reel)

        return make_scorer22(cout_moyen, response_method="predict_proba", greater_is_better=False,
                             pos_label=positif, cout_fn=cout_fn, cout_fp=cout_fp, seuil=seuil)

    class ImputeurMediane(TransformerMixin22, BaseEstimator22):    # une boucle, et np.nanmedian
        def fit(self, X, y=None):
            self.medianes_ = {}
            for colonne in X.columns:
                valeurs = X[colonne].to_numpy(dtype=float)
                if np22.isnan(valeurs).all():
                    raise ValueError(f"{colonne} est vide")
                self.medianes_[colonne] = float(np22.nanmedian(valeurs))
            return self

        def transform(self, X):
            check_is_fitted22(self)
            if sorted(X.columns) != sorted(self.medianes_):
                raise ValueError("colonnes")
            resultat = X.copy()
            for colonne, mediane in self.medianes_.items():
                resultat[colonne] = resultat[colonne].fillna(mediane)
            return resultat

    class RegroupeurRares(TransformerMixin22, BaseEstimator22):    # Counter et des compréhensions
        def __init__(self, min_effectif=10, libelle="Autre"):
            self.min_effectif = min_effectif
            self.libelle = libelle

        def fit(self, X, y=None):
            import collections

            m = self.min_effectif
            if type(m) is bool or not isinstance(m, (int, np22.integer)) or m < 1:
                raise ValueError("min_effectif")
            self.frequentes_ = {c: {v for v, n in collections.Counter(X[c].dropna()).items() if n >= m}
                                for c in X.columns}
            return self

        def transform(self, X):
            check_is_fitted22(self)
            return X.apply(lambda col: col.map(
                lambda v: v if pd.isna(v) or v in self.frequentes_[col.name] else self.libelle))

    class EncodeurCible(TransformerMixin22, BaseEstimator22):   # une table pandas, merge et dictionnaires
        def __init__(self, lissage=10.0):
            self.lissage = lissage

        def fit(self, X, y=None):
            if y is None or isinstance(self.lissage, bool) or not self.lissage >= 0:
                raise ValueError("y ou lissage")
            y = list(y)
            if len(y) != len(X):
                raise ValueError("longueur")
            table = X.reset_index(drop=True).assign(_y=y)
            self.prior_ = sum(y) / len(y)
            self.tables_ = {}
            for colonne in X.columns:
                groupes = table.dropna(subset=[colonne]).groupby(colonne)["_y"]
                self.tables_[colonne] = ((groupes.sum() + self.lissage * self.prior_)
                                         / (groupes.count() + self.lissage)).to_dict()
            return self

        def transform(self, X):
            check_is_fitted22(self)
            return pd.DataFrame({c: [float(self.tables_[c].get(v, self.prior_)) if not pd.isna(v) else self.prior_
                                     for v in X[c]] for c in X.columns}, index=X.index)

    from defis.corrige_serie_22 import EncodeurCible as Base

    class EncodeurCibleHorsPli(Base):   # des masques booléens plutôt que iloc[test] =
        def __init__(self, lissage=10.0, n_plis=5, graine=0):
            self.lissage, self.n_plis, self.graine = lissage, n_plis, graine

        def fit_transform(self, X, y=None):
            if type(self.n_plis) is not int or self.n_plis < 2:
                raise ValueError("n_plis")
            self.fit(X, y)
            y = np22.asarray(y, dtype=float)
            morceaux = []
            for train, test in KFold22(self.n_plis, shuffle=True, random_state=self.graine).split(X):
                morceaux.append(Base(lissage=self.lissage).fit(X.iloc[train], y[train]).transform(X.iloc[test]))
            return pd.concat(morceaux).reindex(X.index)

    class ClassifieurSeuil(ClassifierMixin22, BaseEstimator22):   # en réutilisant le code du projet
        def __init__(self, estimateur, cout_fn=500.0, cout_fp=50.0, n_plis=5, graine=0):
            self.estimateur, self.cout_fn, self.cout_fp = estimateur, cout_fn, cout_fp
            self.n_plis, self.graine = n_plis, graine

        def fit(self, X, y):
            from churn.model import best_threshold_by_cost

            if self.cout_fn < 0 or self.cout_fp < 0 or type(self.n_plis) is not int or self.n_plis < 2:
                raise ValueError("paramètres")
            self.classes_ = np22.unique(y)
            if self.classes_.size != 2:
                raise ValueError("deux classes")
            plis = StratifiedKFold22(self.n_plis, shuffle=True, random_state=self.graine)
            proba = cross_val_predict22(clone22(self.estimateur), X, y, cv=plis, method="predict_proba")[:, 1]
            reel = (np22.asarray(y) == self.classes_[1]).astype(int)
            self.seuil_ = best_threshold_by_cost(reel, proba, self.cout_fn, self.cout_fp)
            self.estimateur_ = clone22(self.estimateur).fit(X, y)
            return self

        def predict_proba(self, X):
            check_is_fitted22(self, "estimateur_")
            return self.estimateur_.predict_proba(X)

        def predict(self, X):
            return np22.where(self.predict_proba(X)[:, 1] >= self.seuil_, self.classes_[1], self.classes_[0])

    for nom, objet in [("scoreur_cout", scoreur_cout), ("ImputeurMediane", ImputeurMediane),
                       ("RegroupeurRares", RegroupeurRares), ("EncodeurCible", EncodeurCible),
                       ("EncodeurCibleHorsPli", EncodeurCibleHorsPli), ("ClassifieurSeuil", ClassifieurSeuil)]:
        resultat = _evaluer(nom, objet)
        assert resultat.statut == REUSSI, (nom, resultat.message)


# --------------------------------------------------------------------------- Série 23 : ouvrir la boîte noire

import math as math23  # noqa: E402
import numbers as numbers23  # noqa: E402
import warnings as warnings23  # noqa: E402

import numpy as np23  # noqa: E402
from sklearn.base import BaseEstimator as BaseEstimator23  # noqa: E402
from sklearn.base import ClassifierMixin as ClassifierMixin23  # noqa: E402
from sklearn.utils.validation import check_is_fitted as check_is_fitted23  # noqa: E402

from defis.maison import clients_exemple as clients_23  # noqa: E402
from defis.maison import generer_clients as generer_clients_23  # noqa: E402


def _echec_23(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


def _sig_23(z):
    e = np23.exp(-np23.abs(z))
    return np23.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _entier_23(valeur, minimum):
    return not isinstance(valeur, bool) and isinstance(valeur, numbers23.Integral) and valeur >= minimum


# Défi 140 : sigmoide


def _sigmoide_23(defaut=None):
    def sigmoide(z):
        if defaut == "lent":
            tableau = np23.asarray(z, dtype=float)
            valeurs = [1 / (1 + math23.exp(-v)) if v >= 0 else math23.exp(v) / (1 + math23.exp(v)) if v < 0 else v
                       for v in tableau.ravel().tolist()]
            resultat = np23.array(valeurs, dtype=float).reshape(tableau.shape)
            return float(resultat) if resultat.ndim == 0 else resultat
        if defaut == "boucle":
            if np23.ndim(z) == 0:
                return float(_sig_23(np23.asarray(z, dtype=float)))
            return np23.array([float(_sig_23(np23.float64(v))) for v in z])
        z = np23.asarray(z, dtype=float)
        if defaut == "naif":
            resultat = 1 / (1 + np23.exp(-z))
        elif defaut == "where_naif":
            resultat = np23.where(z >= 0, 1 / (1 + np23.exp(-z)), np23.exp(z) / (1 + np23.exp(z)))
        elif defaut == "soustraction":
            e = np23.exp(-np23.abs(z))
            resultat = np23.where(z >= 0, 1 / (1 + e), 1 - 1 / (1 + e))
        elif defaut == "clip":
            resultat = 1 / (1 + np23.exp(-np23.clip(z, -500, 500)))
        elif defaut == "sur_place":
            positif = z >= 0
            np23.abs(z, out=z)
            e = np23.exp(-z)
            resultat = np23.where(positif, 1 / (1 + e), e / (1 + e))
        elif defaut == "nan_zero":
            resultat = np23.nan_to_num(_sig_23(z))
        else:
            resultat = _sig_23(z)
        if defaut == "liste":
            return resultat.tolist()
        if defaut == "toujours_tableau":
            return np23.atleast_1d(resultat)
        return float(resultat) if resultat.ndim == 0 else resultat

    return sigmoide


@pytest.mark.parametrize("mutant,mots", [
    (_sigmoide_23("naif"), ("overflow", "np.exp(-np.abs(z))")),
    (_sigmoide_23("where_naif"), ("overflow", "DEUX branches")),
    (_sigmoide_23("soustraction"), ("sigmoide(-600)", "tous les chiffres sont perdus")),
    (_sigmoide_23("clip"), ("np.clip", "-600")),
    (_sigmoide_23("sur_place"), ("MODIFIÉ",)),
    (_sigmoide_23("nan_zero"), ("NaN reste NaN",)),
    (_sigmoide_23("liste"), ("tableau numpy",)),
    (_sigmoide_23("toujours_tableau"), ("au lieu d'un nombre",)),
    (_sigmoide_23("boucle"), ("Pas de boucle",)),
    (_sigmoide_23("lent"), ("2 millions", "boucle")),
])
def test_serie_23_sigmoide_erreurs_classiques(mutant, mots):
    _echec_23("sigmoide", mutant, *mots)


# Défi 141 : perte_logistique


def _perte_23(defaut=None):
    def perte_logistique(y, z):
        if defaut == "series_alignees" and isinstance(y, pd.Series) and isinstance(z, pd.Series):
            ya, za = y, z
        else:
            ya, za = np23.asarray(y, dtype=float), np23.asarray(z, dtype=float)
        yv, zv = np23.asarray(y, dtype=float), np23.asarray(z, dtype=float)
        if yv.shape != zv.shape:
            raise ValueError("longueurs")
        if yv.size == 0 and defaut != "vide_accepte":
            raise ValueError("vide")
        if defaut != "sans_validation" and not np23.isin(yv, (0, 1)).all():
            raise ValueError("0 ou 1")
        if defaut == "naif":
            p = _sig_23(zv)
            return float(-np23.mean(yv * np23.log(p) + (1 - yv) * np23.log(1 - p)))
        if defaut == "clip":
            p = np23.clip(_sig_23(zv), 1e-15, 1 - 1e-15)
            return float(-np23.mean(yv * np23.log(p) + (1 - yv) * np23.log(1 - p)))
        if defaut == "lent":
            return sum(max(v, 0.0) + math23.log1p(math23.exp(-abs(v))) - c * v for c, v in zip(yv, zv)) / len(yv)
        par_client = np23.logaddexp(0.0, za) - ya * za
        if defaut == "somme":
            return float(np23.sum(par_client))
        if defaut == "un_terme":
            return float(np23.mean(ya * np23.logaddexp(0.0, -za)))
        if defaut == "signe":
            return -float(np23.mean(par_client))
        if defaut == "log10":
            return float(np23.mean(par_client) / np23.log(10))
        return float(np23.mean(par_client))

    return perte_logistique


@pytest.mark.parametrize("mutant,mots", [
    (_perte_23("naif"), ("divide by zero", "np.logaddexp(0, z)")),
    (_perte_23("clip"), ("np.clip", "34,5")),
    (_perte_23("somme"), ("SOMME", "MOYENNE")),
    (_perte_23("un_terme"), ("restés",)),
    (_perte_23("signe"), ("NÉGATIVE",)),
    (_perte_23("log10"), ("logarithme naturel",)),
    (_perte_23("series_alignees"), ("ÉTIQUETTE", "np.asarray")),
    (_perte_23("sans_validation"), ("ValueError", "y contient 2")),
    (_perte_23("vide_accepte"), ("ValueError", "aucun client")),
    (_perte_23("lent"), ("2 millions",)),
])
def test_serie_23_perte_erreurs_classiques(mutant, mots):
    _echec_23("perte_logistique", mutant, *mots)


# Défi 142 : gradient_logistique


def _gradient_23(defaut=None):
    def gradient_logistique(X, y, w, b, alpha=0.0):
        if defaut == "pandas":
            Xa, ya, wa = X, y, np23.asarray(w, dtype=float)
        else:
            Xa, ya, wa = np23.asarray(X, dtype=float), np23.asarray(y, dtype=float), np23.asarray(w, dtype=float)
        n = len(ya)
        z = Xa @ wa + (0.0 if defaut == "sans_b" else b)
        if defaut == "naif":
            p = 1 / (1 + np23.exp(-np23.asarray(z, dtype=float)))
        else:
            p = _sig_23(np23.asarray(z, dtype=float))
        if isinstance(z, pd.Series):
            p = pd.Series(p, index=z.index)
        erreur = p - ya
        if defaut == "signe":
            erreur = -erreur
        if defaut == "lent":
            Xa = np23.asarray(Xa, dtype=float)
            grad_w = np23.zeros(Xa.shape[1])
            for i in range(n):
                grad_w += Xa[i] * erreur[i]
            return grad_w / n + alpha * wa, float(np23.mean(erreur))
        produit = Xa @ erreur if defaut == "X_sans_T" else Xa.T @ erreur
        produit = np23.asarray(produit, dtype=float)
        if defaut == "somme":
            grad_w, grad_b = produit, float(np23.sum(erreur))
        else:
            grad_w, grad_b = produit / n, float(np23.mean(erreur))
        if defaut == "deux_alpha":
            grad_w = grad_w + 2 * alpha * wa
        elif defaut != "sans_alpha":
            grad_w = grad_w + alpha * wa
        if defaut == "b_penalise":
            grad_b += alpha * b
        return grad_w, grad_b

    return gradient_logistique


@pytest.mark.parametrize("mutant,mots", [
    (_gradient_23("somme"), ("diviser par n",)),
    (_gradient_23("signe"), ("SIGNE",)),
    (_gradient_23("sans_b"), ("X @ w + b",)),
    (_gradient_23("b_penalise"), ("ordonnée à l'origine est pénalisée",)),
    (_gradient_23("sans_alpha"), ("pénalité est oubliée",)),
    (_gradient_23("deux_alpha"), ("compte double",)),
    (_gradient_23("X_sans_T"), ("X.T @ erreur",)),
    (_gradient_23("naif"), ("overflow", "VOTRE sigmoide")),
    (_gradient_23("pandas"), ("np.asarray",)),
    (_gradient_23("lent"), ("300 000",)),
])
def test_serie_23_gradient_erreurs_classiques(mutant, mots):
    _echec_23("gradient_logistique", mutant, *mots)


# Défi 143 : RegressionLogistiqueGD


def _descente_23(defaut=None):
    from defis.corrige_serie_23 import gradient_logistique

    bases = (BaseEstimator23, ClassifierMixin23) if defaut == "mixin_a_droite" else (ClassifierMixin23, BaseEstimator23)

    def __init__(self, alpha=0.01, pas=0.5, n_iter_max=10_000, tol=1e-6):
        if defaut == "verifie_init" and pas <= 0:
            raise ValueError("pas")
        self.alpha, self.pas, self.n_iter_max, self.tol = alpha, pas, n_iter_max, tol

    def fit(self, X, y):
        if defaut != "sans_validation":
            if (isinstance(self.alpha, bool) or not self.alpha >= 0 or isinstance(self.pas, bool) or not self.pas > 0
                    or not _entier_23(self.n_iter_max, 1) or not self.tol >= 0):
                raise ValueError("paramètres")
        X = np23.asarray(X, dtype=float)
        y = np23.asarray(y)
        self.classes_ = np23.unique(y)
        if self.classes_.size != 2:
            raise ValueError("deux classes")
        positive = self.classes_[0] if defaut == "classes_inverse" else self.classes_[1]
        y01 = (y == positive).astype(float)
        if defaut == "warm_start" and hasattr(self, "coef_"):
            w, b = self.coef_.copy(), self.intercept_
        else:
            w, b = np23.zeros(X.shape[1]), 0.0
        debut = 1 if defaut == "range_1" else 0
        self.n_iter_ = self.n_iter_max
        for iteration in range(debut, self.n_iter_max):
            grad_w, grad_b = gradient_logistique(X, y01, w, b, self.alpha)
            if defaut == "b_penalise":
                grad_b += self.alpha * b
            norme = np23.sqrt(grad_w @ grad_w) if defaut == "norme_w" else np23.sqrt(grad_w @ grad_w + grad_b ** 2)
            if defaut != "sans_arret" and norme < self.tol:
                self.n_iter_ = iteration
                break
            if defaut == "montee":
                w, b = w + self.pas * grad_w, b + self.pas * grad_b
            else:
                w = w - self.pas * grad_w
                b = b if defaut == "b_fige" else b - self.pas * grad_b
        self.coef_, self.intercept_ = w, float(b)
        return None if defaut == "sans_return" else self

    def predict_proba(self, X):
        if defaut != "sans_check":
            check_is_fitted23(self)
        p = _sig_23(np23.asarray(X, dtype=float) @ self.coef_ + self.intercept_)
        return np23.column_stack([p, 1 - p] if defaut == "proba_inversee" else [1 - p, p])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        if defaut == "predict_01":
            return (p > 0.5).astype(int)
        return self.classes_[(p > 0.5).astype(int)]

    return type("RegressionLogistiqueGD", bases, {"__init__": __init__, "fit": fit, "predict_proba": predict_proba,
                                                  "predict": predict})


@pytest.mark.parametrize("mutant,mots", [
    (_descente_23("montee"), ("MONTEZ",)),
    (_descente_23("classes_inverse"), ("classes_[1]", "signe opposé")),
    (_descente_23("b_fige"), ("b = b - pas * grad_b",)),
    (_descente_23("b_penalise"), ("ordonnée à l'origine est pénalisée",)),
    (_descente_23("sans_arret"), ("ne vous arrêtez jamais",)),
    (_descente_23("norme_w"), ("w ET b",)),
    (_descente_23("range_1"), ("encore 0",)),
    (_descente_23("warm_start"), ("repart de w = 0",)),
    (_descente_23("proba_inversee"), ("inversées",)),
    (_descente_23("predict_01"), ("ÉTIQUETTES",)),
    (_descente_23("verifie_init"), ("dès la construction",)),
    (_descente_23("sans_validation"), ("Vérifiez les paramètres",)),
    (_descente_23("sans_check"), ("NotFittedError",)),
    (_descente_23("sans_return"), ("return self",)),
    (_descente_23("mixin_a_droite"), ("À GAUCHE",)),
])
def test_serie_23_descente_erreurs_classiques(mutant, mots):
    _echec_23("RegressionLogistiqueGD", mutant, *mots)


def test_serie_23_descente_le_pas_de_trop_et_les_autres_normes_sont_reconnus():
    """Les variantes du critère d'arrêt que le correcteur diagnostique, même sans fabrique de mutant dédiée."""
    from defis.checks_serie_23 import _comparer_descente, _ref_descente

    class Faux:
        def __init__(self, w, b, n_iter):
            self.coef_, self.intercept_, self.n_iter_, self.classes_ = w, b, n_iter, np23.array([0, 1])

    clients = generer_clients_23(300, graine=143)
    X = (clients[["anciennete", "facture", "appels"]] - clients[["anciennete", "facture", "appels"]].mean()).to_numpy()
    X = X / X.std(axis=0)
    y = clients["churn"].to_numpy()
    for variante, mot in (("arret_apres", "un pas de trop"), ("norme_max", "EUCLIDIENNE"), ("norme_carre", "CARRÉ")):
        w, b, n_iter = _ref_descente(X, y, alpha=0.1, tol=1e-3, variante=variante)
        with pytest.raises(AssertionError, match=mot):
            _comparer_descente(Faux(w, b, n_iter), X, y, {"alpha": 0.1, "tol": 1e-3}, "fit")


# Défi 144 : meilleure_coupure


def _sur_tableaux_tries_23(xs, ys, min_feuille):
    """Le calcul juste… à condition que xs soit vraiment trié (et ys rangé dans le même ordre)."""
    n = xs.size
    k = np23.arange(1, n)
    partis = np23.cumsum(ys)[:-1]
    q, qg, qd = ys.mean(), partis / k, (ys.sum() - partis) / (n - k)
    gains = 2 * q * (1 - q) - k / n * 2 * qg * (1 - qg) - (n - k) / n * 2 * qd * (1 - qd)
    possible = (xs[1:] != xs[:-1]) & (k >= min_feuille) & (n - k >= min_feuille)
    if not possible.any():
        return None, 0.0
    i = int(np23.argmax(np23.where(possible, gains, -np23.inf)))
    return float((xs[i] + xs[i + 1]) / 2), float(gains[i])


def _coupure_23(defaut=None):
    from defis.checks_serie_23 import _ref_coupure

    def meilleure_coupure(x, y, min_feuille=1):
        if not _entier_23(min_feuille, 1):
            raise ValueError("min_feuille")
        brut = x
        x, y = np23.asarray(x, dtype=float), np23.asarray(y, dtype=float)
        if x.ndim != 1 or x.shape != y.shape or x.size == 0 or np23.isnan(x).any() or not np23.isin(y, (0, 1)).all():
            raise ValueError("entrées")
        if defaut == "series_etiquettes" and isinstance(brut, pd.Series):
            ordre = np23.argsort(x, kind="stable")
            return _sur_tableaux_tries_23(brut[ordre].to_numpy(dtype=float), y[ordre], min_feuille)  # des ÉTIQUETTES
        if defaut == "suppose_trie":
            return _sur_tableaux_tries_23(x, y, min_feuille)
        if defaut == "sort_sur_place":
            ordre = np23.argsort(x, kind="stable")
            y = y[ordre]
            x.sort()
        if defaut == "quadratique":
            valeurs = np23.unique(x)
            meilleur = None
            for a, b in zip(valeurs[:-1], valeurs[1:]):
                seuil = (a + b) / 2
                gauche = x <= seuil
                ng = int(gauche.sum())
                if ng < min_feuille or x.size - ng < min_feuille:
                    continue
                q, qg, qd = y.mean(), y[gauche].mean(), y[~gauche].mean()
                gain = 2 * q * (1 - q) - ng / x.size * 2 * qg * (1 - qg) - (x.size - ng) / x.size * 2 * qd * (1 - qd)
                if meilleur is None or gain > meilleur[1] + 1e-12:
                    meilleur = (float(seuil), float(gain))
            return meilleur or (None, 0.0)
        resultat = _ref_coupure(x, y, min_feuille, None if defaut in ("none_nan", "leve_si_aucune", "sort_sur_place")
                                else defaut)
        if resultat[0] is None and defaut == "none_nan":
            return float("nan"), 0.0
        if resultat[0] is None and defaut == "leve_si_aucune":
            raise ValueError("aucune coupure")
        return resultat

    return meilleure_coupure


@pytest.mark.parametrize("mutant,mots", [
    (_coupure_23("entre_egales"), ("ENTRE deux valeurs égales",)),
    (_coupure_23("valeur_gauche"), ("valeur de gauche", "MILIEU")),
    (_coupure_23("valeur_droite"), ("valeur de droite", "MILIEU")),
    (_coupure_23("non_pondere"), ("PART des clients",)),
    (_coupure_23("entropie"), ("ENTROPIE", "GINI")),
    (_coupure_23("impurete"), ("GAIN",)),
    (_coupure_23("plus_grand"), ("PLUS PETIT seuil",)),
    (_coupure_23("sans_min"), ("min_feuille est ignoré",)),
    (_coupure_23("min_gauche"), ("qu'à GAUCHE",)),
    (_coupure_23("suppose_trie"), ("pas trié", "np.argsort")),
    (_coupure_23("series_etiquettes"), ("ÉTIQUETTES d'index",)),
    (_coupure_23("sort_sur_place"), ("x.sort()", "np.argsort(x)")),
    (_coupure_23("none_nan"), ("(None, 0.0)",)),
    (_coupure_23("leve_si_aucune"), ("None, 0.0",)),
    (_coupure_23("quadratique"), ("10 000", "O(n²)")),
])
def test_serie_23_coupure_erreurs_classiques(mutant, mots):
    _echec_23("meilleure_coupure", mutant, *mots)


# Défi 145 : ArbreDecision


def _arbre_23(defaut=None):
    from defis.corrige_serie_23 import meilleure_coupure

    bases = (ClassifierMixin23, BaseEstimator23)

    def __init__(self, max_profondeur=3, min_feuille=1):
        if defaut == "verifie_init" and max_profondeur is not None and max_profondeur < 1:
            raise ValueError("max_profondeur")
        self.max_profondeur, self.min_feuille = max_profondeur, min_feuille

    def fit(self, X, y):
        if defaut != "sans_validation":
            if self.max_profondeur is not None and not _entier_23(self.max_profondeur, 1):
                raise ValueError("max_profondeur")
            if not _entier_23(self.min_feuille, 1):
                raise ValueError("min_feuille")
        X = np23.asarray(X, dtype=float)
        y = np23.asarray(y)
        self.classes_ = np23.unique(y)
        if self.classes_.size != 2:
            raise ValueError("deux classes")
        positive = self.classes_[0] if defaut == "classes_inverse" else self.classes_[1]
        self.arbre_ = self._construire(X, (y == positive).astype(float), 0)

        def mesures(noeud):
            if "colonne" not in noeud:
                return 0, 1
            (pg, fg), (pd_, fd) = mesures(noeud["gauche"]), mesures(noeud["droite"])
            return 1 + max(pg, pd_), fg + fd

        self.profondeur_, self.n_feuilles_ = mesures(self.arbre_)
        if defaut == "profondeur_en_niveaux":
            self.profondeur_ += 1
        return self

    def _construire(self, X, y, profondeur):
        p = float(y.mean())
        proba = {"majoritaire": float(round(p)) if p != 0.5 else 0.0}.get(defaut, p)
        feuille = {"proba": proba, "effectif": int(y.size)}
        limite = self.max_profondeur
        if limite is not None:
            limite = limite - 1 if defaut == "profondeur_moins_un" else limite + 1 if defaut == "profondeur_plus_un" \
                else limite
            if profondeur >= limite:
                return feuille
        if defaut != "pur_coupe" and p in (0.0, 1.0):
            return feuille
        meilleure = None
        for colonne in range(X.shape[1]):
            if defaut == "min_feuille_oublie":
                seuil, gain = meilleure_coupure(X[:, colonne], y)
            else:
                seuil, gain = meilleure_coupure(X[:, colonne], y, self.min_feuille)
            if seuil is None:
                continue
            if defaut == "premiere_colonne":
                if meilleure is None:
                    meilleure = (gain, colonne, seuil)
            elif defaut == "plus_petit_gain":
                if meilleure is None or gain < meilleure[0]:
                    meilleure = (gain, colonne, seuil)
            elif meilleure is None or gain > meilleure[0] + 1e-12:
                meilleure = (gain, colonne, seuil)
        if meilleure is None or (defaut == "gain_positif" and meilleure[0] <= 0):
            return feuille
        _, colonne, seuil = meilleure
        gauche = X[:, colonne] <= seuil
        return {"colonne": colonne, "seuil": seuil,
                "gauche": self._construire(X[gauche], y[gauche], profondeur + 1),
                "droite": self._construire(X[~gauche], y[~gauche], profondeur + 1)}

    def predict_proba(self, X):
        if defaut != "sans_check":
            check_is_fitted23(self)
        X = np23.asarray(X, dtype=float)
        p = np23.empty(len(X))
        for i, ligne in enumerate(X):
            noeud = self.arbre_
            while "colonne" in noeud:
                if defaut == "strict":
                    va_a_gauche = ligne[noeud["colonne"]] < noeud["seuil"]
                else:
                    va_a_gauche = ligne[noeud["colonne"]] <= noeud["seuil"]
                noeud = noeud["gauche"] if va_a_gauche else noeud["droite"]
            p[i] = noeud["proba"]
        return np23.column_stack([p, 1 - p] if defaut == "proba_inversee" else [1 - p, p])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return self.classes_[(p >= 0.5 if defaut == "egalite_classe1" else p > 0.5).astype(int)]

    return type("ArbreDecision", bases, {"__init__": __init__, "fit": fit, "_construire": _construire,
                                         "predict_proba": predict_proba, "predict": predict})


@pytest.mark.parametrize("mutant,mots", [
    (_arbre_23("strict"), ("EXACTEMENT sur un seuil", "x <= seuil")),
    (_arbre_23("gain_positif"), ("gain est nul", "ou exclusif")),
    (_arbre_23("profondeur_moins_un"), ("une profondeur trop tôt",)),
    (_arbre_23("profondeur_plus_un"), ("profondeur maximale est atteinte",)),
    (_arbre_23("pur_coupe"), ("PUR",)),
    (_arbre_23("majoritaire"), ("MAJORITAIRE",)),
    (_arbre_23("classes_inverse"), ("classes_[0]",)),
    (_arbre_23("premiere_colonne"), ("Comparez les gains de TOUTES les colonnes",)),
    (_arbre_23("plus_petit_gain"), ("PLUS PETIT gain",)),
    (_arbre_23("min_feuille_oublie"), ("min_feuille",)),
    (_arbre_23("egalite_classe1"), ("PREMIÈRE classe", "p > 0.5")),
    (_arbre_23("proba_inversee"), ("inversées",)),
    (_arbre_23("profondeur_en_niveaux"), ("profondeur_",)),
    (_arbre_23("verifie_init"), ("dès la construction",)),
    (_arbre_23("sans_validation"), ("Vérifiez les paramètres",)),
    (_arbre_23("sans_check"), ("NotFittedError",)),
])
def test_serie_23_arbre_erreurs_classiques(mutant, mots):
    _echec_23("ArbreDecision", mutant, *mots)


@pytest.mark.parametrize("nom,fabrique", [("sigmoide", _sigmoide_23), ("perte_logistique", _perte_23),
                                          ("gradient_logistique", _gradient_23),
                                          ("RegressionLogistiqueGD", _descente_23),
                                          ("meilleure_coupure", _coupure_23), ("ArbreDecision", _arbre_23)])
def test_serie_23_sans_defaut_les_fabriques_de_mutants_sont_justes(nom, fabrique):
    """Garde-fou : chaque version boguée ci-dessus ne diffère du juste que par le défaut qu'on teste."""
    resultat = _evaluer(nom, fabrique())
    assert resultat.statut == REUSSI, resultat.message


def test_serie_23_les_tests_du_corrige_passent():
    import defis.corrige_serie_23 as corrige

    tests = [f for nom, f in list(vars(corrige).items()) if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_serie_23_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.maison as maison
    import defis.serie_23 as eleve

    for nom in ("clients_exemple", "generer_clients", "NUMERIQUES", "gradient_numerique", "afficher_arbre"):
        assert getattr(eleve, nom) is getattr(maison, nom), nom


def test_serie_23_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    from defis.checks_serie_23 import SERIE

    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6


def test_serie_23_les_exemples_sont_neufs_a_chaque_appel():
    premiere = clients_23()
    premiere.iloc[0:1] = 0
    assert clients_23().iloc[0]["anciennete"] == 2
    assert generer_clients_23(50) is not generer_clients_23(50)


def test_serie_23_le_generateur_est_reproductible_et_realiste():
    a = generer_clients_23(3000, graine=4)
    pd.testing.assert_frame_equal(a, generer_clients_23(3000, graine=4))
    assert 0.25 < a["churn"].mean() < 0.42 and not a.isna().any(axis=None)
    assert list(a.index[:2]) == ["G0001", "G0002"]
    # des valeurs exactes en float32 : scikit-learn (qui range X en float32) voit les mêmes nombres
    for colonne in ("anciennete", "facture", "appels"):
        assert (a[colonne].astype(np23.float32).astype(float) == a[colonne]).all()
    # le churn suit vraiment les colonnes : moins d'ancienneté, plus d'appels → plus de départs
    assert a.loc[a["anciennete"] < 12, "churn"].mean() > a.loc[a["anciennete"] > 48, "churn"].mean() + 0.2
    assert a.loc[a["appels"] >= 3, "churn"].mean() > a.loc[a["appels"] == 0, "churn"].mean() + 0.15


def test_serie_23_l_exemple_a_tous_les_pieges_annonces():
    from defis.corrige_serie_23 import meilleure_coupure

    c = clients_23()
    assert c["churn"].mean() == 0.5 and (c["facture"] == 60).sum() == 2
    gains = {col: meilleure_coupure(c[col], c["churn"])[1] for col in ("anciennete", "facture", "appels")}
    assert max(gains, key=gains.get) == "appels" and sorted(gains.values())[-1] - sorted(gains.values())[-2] > 0.01
    assert c.loc[c["appels"] <= 1, "churn"].sum() == 0
    assert c.loc[["C05", "C08"], "appels"].tolist() == [4, 4] and c.loc[["C05", "C08"], "churn"].tolist() == [0, 1]


def test_serie_23_les_exemples_des_docstrings_sont_justes():
    """Les exemples de defis/serie_23.py et defis/maison.py, rejoués avec le corrigé."""
    import doctest

    import defis.corrige_serie_23 as corrige
    import defis.maison as maison
    import defis.serie_23 as eleve

    contexte = dict(vars(eleve))
    for nom in ("sigmoide", "perte_logistique", "gradient_logistique", "RegressionLogistiqueGD", "meilleure_coupure",
                "ArbreDecision"):
        contexte[nom] = getattr(corrige, nom)
    lanceur = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    exemples = 0
    with warnings23.catch_warnings():
        warnings23.simplefilter("ignore", RuntimeWarning)   # np.exp(710) prévient, c'est le sujet de l'exemple
        for module in (eleve, maison):
            for test in doctest.DocTestFinder().find(module):
                test.globs = dict(contexte)
                exemples += len(test.examples)
                lanceur.run(test)
    resultats = lanceur.summarize(verbose=False)
    assert resultats.failed == 0 and exemples >= 35, resultats


def _gains_uniques_23(X, y, noeud, min_feuille, marge=1e-9):
    """Vrai si, à chaque nœud de l'arbre, la meilleure coupure n'a AUCUNE rivale à `marge` près (toutes colonnes)."""
    from defis.checks_serie_23 import _coupures

    if "colonne" not in noeud:
        return True
    gains = np23.sort(np23.concatenate([_coupures(X[:, j], y, min_feuille)[1] for j in range(X.shape[1])]))
    if gains.size > 1 and gains[-1] - gains[-2] <= marge:
        return False
    gauche = X[:, noeud["colonne"]] <= noeud["seuil"]
    return (_gains_uniques_23(X[gauche], y[gauche], noeud["gauche"], min_feuille)
            and _gains_uniques_23(X[~gauche], y[~gauche], noeud["droite"], min_feuille))


def test_serie_23_les_arbres_compares_a_scikit_learn_n_ont_aucune_egalite():
    """Le correcteur compare à DecisionTreeClassifier sur ces réglages : aucune égalité ne doit dépendre du hasard."""
    from sklearn.tree import DecisionTreeClassifier

    from defis.corrige_serie_23 import NUMERIQUES, ArbreDecision

    for graine, profondeur, m in ((5, 4, 10), (11, 5, 20)):
        clients = generer_clients_23(1500, graine=graine)
        X, y = clients[NUMERIQUES].to_numpy(dtype=float)[:1000], clients["churn"].to_numpy()[:1000]
        arbre = ArbreDecision(max_profondeur=profondeur, min_feuille=m).fit(X, y)
        assert _gains_uniques_23(X, y.astype(float), arbre.arbre_, m)
        X_test = clients[NUMERIQUES].to_numpy(dtype=float)[1000:]
        for etat in range(5):   # quel que soit le tirage de scikit-learn
            reference = DecisionTreeClassifier(max_depth=profondeur, min_samples_leaf=m, random_state=etat).fit(X, y)
            assert np23.allclose(arbre.predict_proba(X_test), reference.predict_proba(X_test))


def test_serie_23_les_cas_speciaux_de_l_arbre_sont_ceux_de_scikit_learn():
    """Le « ou exclusif » et les feuilles à 50 / 50 : scikit-learn coupe aussi quand le gain est nul."""
    from sklearn.tree import DecisionTreeClassifier

    from defis.corrige_serie_23 import ArbreDecision

    X = np23.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
    y = np23.array(["non", "oui", "oui", "non"])
    reference = DecisionTreeClassifier(random_state=0).fit(X, y)
    arbre = ArbreDecision(max_profondeur=None).fit(X, y)
    assert reference.get_depth() == arbre.profondeur_ == 2 and reference.get_n_leaves() == arbre.n_feuilles_ == 4
    X = np23.array([[1.0], [1.0], [2.0], [2.0]])
    y = np23.array(["oui", "non", "oui", "non"])
    reference = DecisionTreeClassifier(max_depth=3).fit(X, y)
    arbre = ArbreDecision().fit(X, y)
    assert list(reference.predict(X)) == list(arbre.predict(X)) == ["non"] * 4
    assert reference.get_n_leaves() == arbre.n_feuilles_ == 2


def test_serie_23_les_souches_comparees_a_scikit_learn_n_ont_aucune_egalite():
    from defis.checks_serie_23 import _coupures

    clients = generer_clients_23(800, graine=1440)
    for colonne in ("anciennete", "facture", "appels"):
        for m in (1, 40):
            gains = np23.sort(_coupures(clients[colonne], clients["churn"], m)[1])
            assert gains[-1] - gains[-2] > 1e-9, (colonne, m)
    c = clients_23()
    for colonne in ("anciennete", "facture", "appels"):
        gains = np23.sort(_coupures(c[colonne], c["churn"], 1)[1])
        assert gains[-1] - gains[-2] > 1e-9, colonne


def test_serie_23_autres_ecritures_justes_acceptees():
    from scipy.special import expit

    def sigmoide_scipy(z):                         # la fonction de scipy, écrite en C
        resultat = expit(np23.asarray(z, dtype=float))
        return float(resultat) if resultat.ndim == 0 else resultat

    def sigmoide_errstate(z):                      # la formule directe, en faisant taire le débordement
        z = np23.asarray(z, dtype=float)
        with np23.errstate(over="ignore"):
            resultat = 1.0 / (1.0 + np23.exp(-z))
        return resultat[()] if resultat.ndim == 0 else resultat

    def perte_log1p(y, z):                         # max(z, 0) + log1p(exp(-|z|)) : l'autre écriture stable
        y, z = np23.asarray(y, dtype=float), np23.asarray(z, dtype=float)
        if len(y) != len(z) or len(y) == 0 or not set(np23.unique(y)) <= {0.0, 1.0}:
            raise ValueError("entrées")
        return float(np23.mean(np23.maximum(z, 0) + np23.log1p(np23.exp(-np23.abs(z))) - y * z))

    def gradient_einsum(X, y, w, b, alpha=0.0):    # einsum et une moyenne par colonne
        X, y, w = (np23.asarray(v, dtype=float) for v in (X, y, w))
        erreur = expit(X @ w + b) - y
        return (X * erreur[:, None]).mean(axis=0) + alpha * w, np23.einsum("i->", erreur) / len(y)

    class RegressionLogistiqueGD(ClassifierMixin23, BaseEstimator23):   # coef_ à la forme de scikit-learn : (1, d)
        def __init__(self, alpha=0.01, pas=0.5, n_iter_max=10_000, tol=1e-6):
            self.alpha, self.pas, self.n_iter_max, self.tol = alpha, pas, n_iter_max, tol

        def fit(self, X, y):
            if (type(self.n_iter_max) is not int or self.n_iter_max < 1 or isinstance(self.pas, bool)
                    or not self.pas > 0 or not self.alpha >= 0 or not self.tol >= 0):
                raise ValueError("paramètres")
            X = np23.asarray(X, dtype=float)
            self.classes_, y01 = np23.unique(np23.asarray(y), return_inverse=True)
            if len(self.classes_) != 2:
                raise ValueError("deux classes")
            theta = np23.zeros(X.shape[1] + 1)                     # [b, w0, w1, …]
            X1 = np23.column_stack([np23.ones(len(X)), X])
            masque = np23.r_[0.0, np23.ones(X.shape[1])]           # b n'est pas pénalisé
            self.n_iter_ = self.n_iter_max
            for i in range(self.n_iter_max):
                gradient = X1.T @ (expit(X1 @ theta) - y01) / len(X) + self.alpha * masque * theta
                if np23.linalg.norm(gradient) < self.tol:
                    self.n_iter_ = i
                    break
                theta = theta - self.pas * gradient
            self.coef_, self.intercept_ = theta[1:].reshape(1, -1), theta[:1]
            return self

        def predict_proba(self, X):
            check_is_fitted23(self)
            p = expit(np23.asarray(X, dtype=float) @ self.coef_[0] + self.intercept_[0])
            return np23.column_stack([1 - p, p])

        def predict(self, X):
            return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]

    def coupure_python(x, y, min_feuille=1):       # trier des paires, puis une seule boucle avec des compteurs
        if isinstance(min_feuille, bool) or not isinstance(min_feuille, numbers23.Integral) or min_feuille < 1:
            raise ValueError("min_feuille")
        x, y = np23.asarray(x, dtype=float), np23.asarray(y, dtype=float)
        if x.shape != y.shape or x.size == 0 or np23.isnan(x).any() or not np23.isin(y, (0, 1)).all():
            raise ValueError("entrées")
        paires = sorted(zip(x.tolist(), y.tolist()))
        n, total, partis, meilleur = len(paires), sum(v for _, v in paires), 0.0, None

        def gini(q):
            return 1 - q * q - (1 - q) * (1 - q)

        for k in range(1, n):
            partis += paires[k - 1][1]
            if paires[k][0] == paires[k - 1][0] or k < min_feuille or n - k < min_feuille:
                continue
            gain = gini(total / n) - k / n * gini(partis / k) - (n - k) / n * gini((total - partis) / (n - k))
            if meilleur is None or gain > meilleur[1]:
                meilleur = ((paires[k - 1][0] + paires[k][0]) / 2, gain)
        return meilleur if meilleur else (None, 0.0)

    class ArbreDecision(ClassifierMixin23, BaseEstimator23):   # des indices, un `>` strict, et Gini en 1 - p² - q²
        def __init__(self, max_profondeur=3, min_feuille=1):
            self.max_profondeur, self.min_feuille = max_profondeur, min_feuille

        def fit(self, X, y):
            if not (self.max_profondeur is None or (type(self.max_profondeur) is int and self.max_profondeur >= 1)):
                raise ValueError("max_profondeur")
            if type(self.min_feuille) is not int or self.min_feuille < 1:
                raise ValueError("min_feuille")
            self.X_ = np23.asarray(X, dtype=float)
            self.classes_, inverse = np23.unique(np23.asarray(y), return_inverse=True)
            if len(self.classes_) != 2:
                raise ValueError("deux classes")
            self.y_ = inverse.astype(float)
            self.arbre_ = self._noeud(np23.arange(len(self.y_)), 0)
            self.profondeur_, self.n_feuilles_ = self._mesurer(self.arbre_)
            del self.X_, self.y_
            return self

        def _noeud(self, indices, profondeur):
            y = self.y_[indices]
            feuille = {"proba": y.sum() / len(y), "effectif": len(y)}
            if (self.max_profondeur is not None and profondeur == self.max_profondeur) or y.min() == y.max():
                return feuille
            choix = None
            for j in range(self.X_.shape[1]):
                seuil, gain = coupure_python(self.X_[indices, j], y, self.min_feuille)
                if seuil is not None and (choix is None or gain > choix[0]):
                    choix = (gain, j, seuil)
            if choix is None:
                return feuille
            _, j, seuil = choix
            gauche = self.X_[indices, j] <= seuil
            return {"colonne": j, "seuil": seuil, "gauche": self._noeud(indices[gauche], profondeur + 1),
                    "droite": self._noeud(indices[~gauche], profondeur + 1)}

        def _mesurer(self, noeud):
            if "colonne" not in noeud:
                return 0, 1
            (pg, fg), (pd_, fd) = self._mesurer(noeud["gauche"]), self._mesurer(noeud["droite"])
            return max(pg, pd_) + 1, fg + fd

        def predict_proba(self, X):
            check_is_fitted23(self, "arbre_")
            p = []
            for ligne in np23.asarray(X, dtype=float):
                noeud = self.arbre_
                while "colonne" in noeud:
                    noeud = noeud["droite"] if ligne[noeud["colonne"]] > noeud["seuil"] else noeud["gauche"]
                p.append(noeud["proba"])
            p = np23.array(p)
            return np23.column_stack([1 - p, p])

        def predict(self, X):
            return self.classes_[np23.argmax(self.predict_proba(X), axis=1)]

    for nom, objet in [("sigmoide", sigmoide_scipy), ("sigmoide", sigmoide_errstate),
                       ("perte_logistique", perte_log1p), ("gradient_logistique", gradient_einsum),
                       ("RegressionLogistiqueGD", RegressionLogistiqueGD), ("meilleure_coupure", coupure_python),
                       ("ArbreDecision", ArbreDecision)]:
        resultat = _evaluer(nom, objet)
        assert resultat.statut == REUSSI, (nom, resultat.message)

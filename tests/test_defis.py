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

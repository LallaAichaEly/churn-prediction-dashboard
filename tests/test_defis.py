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

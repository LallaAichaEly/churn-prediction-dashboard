"""Corrigé — Module 5 : interpréter et communiquer.

Usage : python solutions/module5_interpreter.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline

try:
    from ._common import load_data, title
except ImportError:  # lancé comme script
    from _common import load_data, title

from churn.data import FEATURES, TARGET
from churn.model import feature_importance, split, train

# Les « vrais » effets écrits dans `_churn_logit` (variation du logit).
TRUE_EFFECTS = {
    "tenure_months (par mois)": -0.035,
    "monthly_charges (par €)": 0.018,
    "total_charges (par €)": 0.0,
    "support_calls (par appel)": 0.45,
    "num_products (par produit)": -0.25,
    "contract : Mensuel vs Annuel": 1.6,
    "contract : Biannuel vs Annuel": -0.9,
    "payment_method : Chèque vs Prélèvement": 0.7,
    "internet_service : Fibre vs ADSL": 0.5,
    "tech_support : Oui vs Non": -0.8,
    "senior : Oui vs Non": 0.3,
}


def _transform(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    Z = pipeline[:-1].transform(X)
    return Z.toarray() if sparse.issparse(Z) else np.asarray(Z)


def column_owners(pipeline: Pipeline) -> list[str]:
    """Pour chaque colonne transformée (après one-hot), la variable d'origine dont elle vient."""
    owners: list[str] = []
    for name, transformer, cols in pipeline.named_steps["prep"].transformers_:
        if name == "num":
            owners += list(cols)
        elif name == "cat":
            owners += [col for col, cats in zip(cols, transformer.categories_) for _ in cats]
    return owners


def recovered_effects(pipeline: Pipeline) -> pd.DataFrame:
    """5.1 — Coefficients appris par la logistique, ramenés dans les unités d'origine.

    Numériques : coef / écart-type du scaler = effet par unité (par mois, par €…).
    Catégorielles : différence entre deux modalités (seules les différences ont un sens).
    """
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    names = prep.get_feature_names_out()
    coef = dict(zip(names, clf.coef_[0]))
    scaler = prep.named_transformers_["num"]
    per_unit = {col: coef[f"num__{col}"] / s for col, s in zip(scaler.feature_names_in_, scaler.scale_)}

    learned = {
        "tenure_months (par mois)": per_unit["tenure_months"],
        "monthly_charges (par €)": per_unit["monthly_charges"],
        "total_charges (par €)": per_unit["total_charges"],
        "support_calls (par appel)": per_unit["support_calls"],
        "num_products (par produit)": per_unit["num_products"],
        "contract : Mensuel vs Annuel": coef["cat__contract_Mensuel"] - coef["cat__contract_Annuel"],
        "contract : Biannuel vs Annuel": coef["cat__contract_Biannuel"] - coef["cat__contract_Annuel"],
        "payment_method : Chèque vs Prélèvement":
            coef["cat__payment_method_Chèque"] - coef["cat__payment_method_Prélèvement"],
        "internet_service : Fibre vs ADSL":
            coef["cat__internet_service_Fibre"] - coef["cat__internet_service_ADSL"],
        "tech_support : Oui vs Non": coef["cat__tech_support_Oui"] - coef["cat__tech_support_Non"],
        "senior : Oui vs Non": coef["cat__senior_Oui"] - coef["cat__senior_Non"],
    }
    return pd.DataFrame({"vrai_effet": TRUE_EFFECTS, "effet_appris": learned})


def explain(pipeline: Pipeline, client: pd.DataFrame, background: pd.DataFrame) -> pd.Series:
    """5.3 — Contribution de chaque variable au logit d'UN client, par rapport au client moyen.

    contribution_j = coef_j × (z_client_j − moyenne(z_j))   (z = variable transformée)
    La somme des contributions = logit(client) − logit(« client moyen »).
    Pour un modèle linéaire, c'est exactement la valeur SHAP (5.5) dans l'espace du logit.
    """
    coef = pipeline.named_steps["clf"].coef_[0]
    delta = _transform(pipeline, client[FEATURES])[0] - _transform(pipeline, background[FEATURES]).mean(axis=0)
    contrib = pd.Series(coef * delta).groupby(column_owners(pipeline)).sum()
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index)


def main() -> None:
    df = load_data()
    pipeline, X_test, y_test, proba, _ = train(df, "logistic")
    X_train = split(df)[0]

    title("5.1 — Le modèle retrouve-t-il les vraies causes ?")
    print(recovered_effects(pipeline).round(4).to_string())
    print("\nImportance par permutation (baisse de ROC-AUC) :")
    print(feature_importance(pipeline, X_test, y_test).round(4).to_string(index=False))

    title("5.3 — Les 3 variables qui pèsent le plus pour le client le plus à risque")
    i = int(np.argmax(proba))
    client = X_test.iloc[[i]]
    print(client.T.rename(columns=lambda _: "valeur").to_string())
    print(f"\nProbabilité de churn : {proba[i]:.1%}  (churn réel : {df.loc[client.index[0], TARGET]})")
    print("\nContributions au logit (vs client moyen) :")
    print(explain(pipeline, client, X_train).head(3).round(3).to_string())


if __name__ == "__main__":
    main()

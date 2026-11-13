"""Corrigé du module 5 — Interpréter et communiquer.

Lancer : python corriges/module_5_interpretation.py   (≈ 30 secondes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _commun import donnees, titre
from sklearn.pipeline import Pipeline

from churn.data import CATEGORICAL_FEATURES, FEATURES
from churn.model import feature_importance, split, train


def true_effect_size(df: pd.DataFrame) -> pd.Series:
    """Écart-type de chaque terme de `_churn_logit` : la « vraie » importance de chaque variable."""
    terms = {
        "contract": 1.6 * (df["contract"] == "Mensuel") - 0.9 * (df["contract"] == "Biannuel"),
        "tenure_months": -0.035 * df["tenure_months"],
        "monthly_charges": 0.018 * df["monthly_charges"],
        "support_calls": 0.45 * df["support_calls"],
        "payment_method": 0.7 * (df["payment_method"] == "Chèque"),
        "internet_service": 0.5 * (df["internet_service"] == "Fibre"),
        "tech_support": -0.8 * (df["tech_support"] == "Oui"),
        "num_products": -0.25 * df["num_products"],
        "senior": 0.3 * (df["senior"] == "Oui"),
        "total_charges": 0.0 * df["total_charges"],  # absente de la vérité !
    }
    return pd.Series({k: float(np.std(v)) for k, v in terms.items()})


def explain_logistic(pipeline: Pipeline, client: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """5.3 — Contribution de chaque variable au logit d'UN client, comparé au client moyen.

    contribution = coefficient × (valeur transformée du client − moyenne de référence)
    Positif = pousse vers le départ ; négatif = retient le client.
    Les colonnes one-hot (ex. `cat__contract_Mensuel`) sont regroupées par variable d'origine.
    """
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    x = prep.transform(client[FEATURES])
    ref = np.asarray(prep.transform(reference[FEATURES]).mean(axis=0)).ravel()
    contrib = clf.coef_[0] * (np.asarray(x).ravel() - ref)

    def origine(name: str) -> str:
        name = name.split("__", 1)[1]
        return next((c for c in CATEGORICAL_FEATURES if name.startswith(c + "_")), name)

    names = [origine(n) for n in prep.get_feature_names_out()]
    return pd.Series(contrib, index=names).groupby(level=0).sum().sort_values(key=abs, ascending=False)


def main() -> None:
    df = donnees()
    pipeline, X_test, y_test, proba, _ = train(df)

    titre("5.1 — Importance trouvée par le modèle vs « vérité cachée »")
    table = pd.DataFrame({
        "vrai_effet (σ du terme)": true_effect_size(df),
        "importance (baisse d'AUC)": feature_importance(pipeline, X_test, y_test).set_index("feature")["importance"],
    })
    table["rang_vrai"] = table.iloc[:, 0].rank(ascending=False).astype(int)
    table["rang_modèle"] = table.iloc[:, 1].rank(ascending=False).astype(int)
    print(table.sort_values("rang_vrai").round(4).to_string())
    print("→ Les grosses causes sont retrouvées dans le bon ordre. Les petites (senior, paiement, internet)")
    print("  se mélangent : leur effet est du même ordre que le bruit d'un jeu de test de 1 250 clients.")
    print("  total_charges n'a AUCUN effet réel, mais elle est corrélée à l'ancienneté × la facture :")
    print("  le modèle peut lui « prêter » un peu d'importance. Corrélation n'est pas causalité !")

    titre("5.3 — Les 3 variables qui poussent le plus la prédiction d'un client")
    X_train, *_ = split(df)
    i = int(np.argmax(proba))  # le client le plus à risque du jeu de test
    client = X_test.iloc[[i]]
    print(client.T.rename(columns=lambda _: "client").to_string())
    print(f"\nProbabilité de churn : {proba[i]:.1%}")
    for var, val in explain_logistic(pipeline, client, X_train).head(3).items():
        sens = "pousse vers le départ" if val > 0 else "retient le client"
        print(f"  {var:18s} {val:+.2f} (logit) → {sens}")


if __name__ == "__main__":
    main()

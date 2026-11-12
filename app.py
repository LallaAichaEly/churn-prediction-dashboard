"""Module 5 — Le tableau de bord (Streamlit).

Lancer :  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from churn.data import (  # noqa: E402
    CONTRACTS,
    FEATURES,
    ID_COL,
    INTERNET_SERVICES,
    PAYMENT_METHODS,
    TARGET,
    load_or_create,
    validate,
)
from churn.model import (  # noqa: E402
    MODELS,
    best_threshold_by_cost,
    evaluate,
    feature_importance,
    roc_points,
    split,
    threshold_table,
    train,
)
from churn.monitoring import PSI_DRIFT, PSI_WATCH, drift_report, psi, simulate_batch  # noqa: E402

# Couleurs : une teinte par rôle, toujours la même (bleu = principal, orange = churn/seuil).
BLUE = "#2a78d6"
ORANGE = "#eb6834"

st.set_page_config(page_title="Churn Dashboard", page_icon="📉", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_or_create(ROOT / "data" / "customers.csv")


@st.cache_resource
def get_model(model_name: str, use_engineered: bool):
    return train(get_data(), model_name, use_engineered)


@st.cache_data
def get_importance(model_name: str, use_engineered: bool) -> pd.DataFrame:
    pipeline, X_test, y_test, _, _ = get_model(model_name, use_engineered)
    return feature_importance(pipeline, X_test, y_test)


@st.cache_data
def get_batch(price_increase: float, monthly_share: float, fibre_competition: float) -> pd.DataFrame:
    return simulate_batch(price_increase=price_increase, monthly_share=monthly_share,
                          fibre_competition=fibre_competition)


def eur(amount: float) -> str:
    """Montant au format français : « 30 450 € » (et non « 30,450 € », le format anglais)."""
    return f"{amount:,.0f} €".replace(",", " ")


def bar(df: pd.DataFrame, x: str, y: str, title: str, pct: bool = True):
    fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate=f"%{{x}}<br>Taux de churn : %{{y:{'.1%' if pct else '.2f'}}}<extra></extra>")
    if pct:
        fig.update_yaxes(tickformat=".0%")
    fig.update_layout(bargap=0.35, margin=dict(t=50, b=10), xaxis_title=None, yaxis_title="Taux de churn")
    return fig


# --------------------------------------------------------------------------- Barre latérale
df = get_data()
problems = validate(df)
if problems:
    st.error("Données invalides : " + " ; ".join(problems))
    st.stop()

st.sidebar.title("⚙️ Réglages")
model_name = st.sidebar.selectbox("Modèle", list(MODELS), format_func=MODELS.get)
use_engineered = st.sidebar.checkbox("Variables dérivées (feature engineering)")
threshold = st.sidebar.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05,
                              help="Au-dessus de ce seuil, un client est considéré « à risque ».")

pipeline, X_test, y_test, proba, _ = get_model(model_name, use_engineered)
metrics = evaluate(y_test, proba, threshold)

st.title("📉 Tableau de bord — Prédiction du churn")
st.caption("Projet d'apprentissage : chaque onglet correspond à un module du fichier PARCOURS.md.")

tab_overview, tab_perf, tab_factors, tab_risk, tab_predict, tab_monitor = st.tabs(
    ["Vue d'ensemble", "Performance", "Facteurs de churn", "Clients à risque", "Prédire un client", "Surveillance"]
)

# --------------------------------------------------------------------------- Vue d'ensemble
with tab_overview:
    churned = df[df[TARGET] == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients", f"{len(df):,}".replace(",", " "))
    c2.metric("Taux de churn", f"{df[TARGET].mean():.1%}")
    c3.metric("Revenu mensuel perdu", eur(churned["monthly_charges"].sum()))
    c4.metric("Ancienneté moyenne (mois)", f"{df['tenure_months'].mean():.1f}")

    left, right = st.columns(2)
    by_contract = df.groupby("contract")[TARGET].mean().reindex(CONTRACTS).reset_index()
    left.plotly_chart(bar(by_contract, "contract", TARGET, "Taux de churn par contrat"))

    buckets = pd.cut(df["tenure_months"], [0, 6, 12, 24, 48, 72],
                     labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    by_tenure = df.groupby(buckets, observed=True)[TARGET].mean().reset_index()
    by_tenure["tenure_months"] = by_tenure["tenure_months"].astype(str)
    right.plotly_chart(bar(by_tenure, "tenure_months", TARGET, "Taux de churn par ancienneté (mois)"))

    left, right = st.columns(2)
    by_calls = df.assign(appels=df["support_calls"].clip(upper=5).astype(str).replace("5", "5+"))
    by_calls = by_calls.groupby("appels")[TARGET].mean().reset_index()
    left.plotly_chart(bar(by_calls, "appels", TARGET, "Taux de churn par nombre d'appels au support"))
    by_pay = df.groupby("payment_method")[TARGET].mean().sort_values(ascending=False).reset_index()
    right.plotly_chart(bar(by_pay, "payment_method", TARGET, "Taux de churn par moyen de paiement"))

    with st.expander("Voir les données brutes"):
        st.dataframe(df.head(200))

# --------------------------------------------------------------------------- Performance
with tab_perf:
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{metrics.roc_auc:.3f}", help="Capacité à classer un churner avant un fidèle (0,5 = hasard).")
    c[1].metric("Accuracy", f"{metrics.accuracy:.1%}")
    c[2].metric("Précision", f"{metrics.precision:.1%}", help="Parmi les clients signalés, % qui partent vraiment.")
    c[3].metric("Rappel", f"{metrics.recall:.1%}", help="Parmi les clients qui partent, % détectés.")
    c[4].metric("F1", f"{metrics.f1:.3f}")

    left, right = st.columns(2)
    roc = roc_points(y_test, proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="Modèle",
                             line=dict(color=BLUE, width=2),
                             hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard",
                             line=dict(color="#898781", width=1, dash="dash"), hoverinfo="skip"))
    fig.update_layout(title=f"Courbe ROC (AUC = {metrics.roc_auc:.3f})",
                      xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs",
                      margin=dict(t=50, b=10))
    left.plotly_chart(fig)

    cm = pd.DataFrame([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]],
                      index=["Réel : reste", "Réel : part"], columns=["Prédit : reste", "Prédit : part"])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=["#cde2fb", "#1c5cab"],
                    title=f"Matrice de confusion (seuil = {threshold:.2f})")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10))
    right.plotly_chart(fig)

    st.subheader("Le compromis précision / rappel")
    table = threshold_table(y_test, proba)
    long = table.melt(id_vars="threshold", value_vars=["precision", "recall", "f1"],
                      var_name="métrique", value_name="valeur")
    fig = px.line(long, x="threshold", y="valeur", color="métrique", markers=True,
                  color_discrete_sequence=[BLUE, ORANGE, "#1baf7a"])
    fig.add_vline(x=threshold, line_dash="dot", line_color="#898781",
                  annotation_text="seuil actuel", annotation_position="top")
    fig.update_layout(hovermode="x unified", yaxis_tickformat=".0%", margin=dict(t=30, b=10))
    st.plotly_chart(fig)

    st.subheader("Choisir le seuil avec le métier 💶")
    c1, c2 = st.columns(2)
    cost_fn = c1.number_input("Coût d'un client perdu non détecté (€)", 0.0, 10_000.0, 500.0, 50.0)
    cost_fp = c2.number_input("Coût d'une action de rétention inutile (€)", 0.0, 10_000.0, 50.0, 10.0)
    best = best_threshold_by_cost(y_test, proba, cost_fn, cost_fp)
    m_best = evaluate(y_test, proba, best)
    st.info(
        f"Seuil optimal pour ces coûts : **{best:.2f}** → précision {m_best.precision:.0%}, "
        f"rappel {m_best.recall:.0%}, coût total {eur(m_best.fn * cost_fn + m_best.fp * cost_fp)} "
        f"(contre {eur(metrics.fn * cost_fn + metrics.fp * cost_fp)} au seuil actuel)."
    )

# --------------------------------------------------------------------------- Facteurs
with tab_factors:
    st.write("**Importance par permutation** : de combien la ROC-AUC baisse quand on mélange une variable.")
    imp = get_importance(model_name, use_engineered).sort_values("importance")
    fig = px.bar(imp, x="importance", y="feature", orientation="h", error_x="std",
                 color_discrete_sequence=[BLUE], title="Variables les plus influentes")
    fig.update_traces(hovertemplate="%{y}<br>Baisse d'AUC : %{x:.4f}<extra></extra>")
    fig.update_layout(height=450, margin=dict(t=50, b=10), yaxis_title=None, xaxis_title="Baisse de ROC-AUC")
    st.plotly_chart(fig)
    st.success(
        "🧠 Exercice : ouvrez `src/churn/data.py`, fonction `_churn_logit`. Les vraies causes du "
        "churn y sont écrites. Le modèle les a-t-il retrouvées ? Lesquelles sont sous-estimées, et pourquoi ?"
    )

# --------------------------------------------------------------------------- Clients à risque
with tab_risk:
    risk = df.loc[X_test.index, [ID_COL, *FEATURES, TARGET]].copy()
    risk["proba_churn"] = proba
    risk = risk[risk["proba_churn"] >= threshold].sort_values("proba_churn", ascending=False)
    c1, c2 = st.columns(2)
    c1.metric("Clients signalés (jeu de test)", len(risk))
    c2.metric("Revenu mensuel à risque", eur(risk["monthly_charges"].sum()))
    st.dataframe(
        risk, hide_index=True,
        column_config={"proba_churn": st.column_config.ProgressColumn("Probabilité", min_value=0, max_value=1, format="%.2f")},
    )
    st.download_button("Télécharger (CSV)", risk.to_csv(index=False).encode("utf-8"),
                       "clients_a_risque.csv", "text/csv")

# --------------------------------------------------------------------------- Prédiction
with tab_predict:
    st.write("Renseignez le profil d'un client pour estimer sa probabilité de départ.")
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Ancienneté (mois)", 1, 72, 12)
        monthly = c1.slider("Facture mensuelle (€)", 15.0, 130.0, 70.0)
        calls = c1.slider("Appels au support", 0, 10, 1)
        contract = c2.selectbox("Contrat", CONTRACTS)
        payment = c2.selectbox("Moyen de paiement", PAYMENT_METHODS)
        internet = c2.selectbox("Internet", INTERNET_SERVICES)
        products = c3.slider("Nombre de produits", 1, 4, 1)
        support = c3.radio("Support technique", ["Oui", "Non"], horizontal=True)
        senior = c3.radio("Senior", ["Oui", "Non"], index=1, horizontal=True)
        submitted = st.form_submit_button("Prédire")

    if submitted:
        client = pd.DataFrame([{
            "tenure_months": tenure, "monthly_charges": monthly, "total_charges": monthly * tenure,
            "support_calls": calls, "num_products": products, "contract": contract,
            "payment_method": payment, "internet_service": internet, "tech_support": support, "senior": senior,
        }])[FEATURES]
        p = float(pipeline.predict_proba(client)[0, 1])
        label = "⚠️ À risque" if p >= threshold else "✅ Plutôt fidèle"
        st.metric("Probabilité de churn", f"{p:.1%}", label, delta_color="off")
        st.progress(p)

# --------------------------------------------------------------------------- Surveillance (module 7)
STATUS_LABELS = {"stable": "✅ stable", "à surveiller": "⚠️ à surveiller", "dérive": "🚨 dérive"}

with tab_monitor:
    st.write(
        "Six mois après la mise en production, un **nouveau lot de 2 000 clients** arrive. "
        "Changez le monde ci-dessous et observez ce que la surveillance détecte… ou pas."
    )
    c1, c2, c3 = st.columns(3)
    price_pct = c1.slider("Hausse des prix (%)", 0, 40, 0, 5, key="drift_price")
    monthly_pct = c2.slider("Part de contrats mensuels (%)", 30, 90, 55, 5, key="drift_monthly",
                            help="55 % aujourd'hui.")
    competitor = c3.toggle("Un concurrent attire les clients fibre", key="drift_concept",
                           help="Dérive du CONCEPT : les clients ne changent pas, leur comportement si.")

    batch = get_batch(price_pct / 100, monthly_pct / 100, 1.5 if competitor else 0.0)
    report = drift_report(split(df)[0], batch)
    new_proba = pipeline.predict_proba(batch[FEATURES])[:, 1]
    predicted, observed = float(new_proba.mean()), float(batch[TARGET].mean())
    new_auc = evaluate(batch[TARGET], new_proba).roc_auc
    n_drift = int((report["statut"] == "dérive").sum())

    st.markdown("**Visible tout de suite** (pas besoin de savoir qui est parti)")
    c = st.columns(3)
    c[0].metric("Variables en dérive", n_drift, help=f"PSI > {PSI_DRIFT:.2f} par rapport aux données d'entraînement.")
    c[1].metric("PSI des scores", f"{psi(proba, new_proba):.3f}",
                help="Les probabilités prédites ont-elles changé de distribution ?")
    c[2].metric("Churn prédit", f"{predicted:.1%}", f"{(predicted - proba.mean()) * 100:+.1f} pts vs référence",
                delta_color="off")

    st.markdown("**Visible des mois plus tard** (quand les vrais départs sont connus)")
    c = st.columns(3)
    c[0].metric("Churn réel", f"{observed:.1%}", f"{(observed - predicted) * 100:+.1f} pts vs prédit",
                delta_color="off")
    c[1].metric("ROC-AUC sur ce lot", f"{new_auc:.3f}", f"{new_auc - metrics.roc_auc:+.3f} vs test")

    if n_drift:
        drifted = ", ".join(f"`{f}`" for f in report.loc[report["statut"] == "dérive", "feature"])
        st.warning(f"🚨 **Dérive des entrées** : {drifted}. Cherchez la cause (décision commerciale ? "
                   "bug de collecte ?) puis vérifiez si la performance baisse avant de réentraîner.")
    if abs(observed - predicted) > 0.05:
        st.warning(f"⚠️ **Le modèle se trompe sur le niveau de churn** : {observed:.1%} réel contre "
                   f"{predicted:.1%} prédit. Si les entrées sont stables, c'est une **dérive du concept** : "
                   "réentraînez sur des données récentes.")

    plot = report.sort_values("psi")
    fig = px.bar(plot, x="psi", y="feature", orientation="h", color_discrete_sequence=[BLUE],
                 title="Stabilité de chaque variable (PSI, données d'entraînement → nouveau lot)")
    fig.update_traces(hovertemplate="%{y}<br>PSI : %{x:.3f}<extra></extra>")
    for x, label in [(PSI_WATCH, "à surveiller"), (PSI_DRIFT, "dérive")]:
        fig.add_vline(x=x, line_dash="dot", line_color="#898781", annotation_text=label,
                      annotation_position="top")
    fig.update_layout(height=420, bargap=0.35, margin=dict(t=60, b=10), yaxis_title=None, xaxis_title="PSI",
                      xaxis_range=[0, max(0.3, float(report["psi"].max()) * 1.1)])
    st.plotly_chart(fig)

    with st.expander("Tableau détaillé"):
        st.dataframe(report.assign(statut=report["statut"].map(STATUS_LABELS)), hide_index=True,
                     column_config={"psi": st.column_config.NumberColumn("PSI", format="%.3f")})
    st.success(
        "🧠 Exercice 7.4–7.5 : augmentez les prix de 20 % puis activez le concurrent fibre. Dans quel cas "
        "le PSI donne-t-il l'alerte ? Dans quel cas seule la comparaison churn prédit / réel la donne ?"
    )

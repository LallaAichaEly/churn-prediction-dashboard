# 📗 Corrigés du parcours

> ⚠️ **Essayez d'abord !** On apprend en cherchant, pas en lisant la solution. Ouvrez un
> corrigé seulement après avoir produit *votre* réponse, puis comparez.

Les exercices de code ont un corrigé **exécutable** dans `solutions/` :

```bash
python solutions/module1_donnees.py      # ~1 s
python solutions/module2_variables.py    # ~2 s
python solutions/module3_modeliser.py    # ~20 s (GridSearchCV)
python solutions/module4_evaluer.py      # ~4 s
python solutions/module5_interpreter.py  # ~3 s
```

Les chiffres ci-dessous proviennent de ces scripts (données par défaut : 5 000 clients, `seed=42`).
Les affirmations importantes sont protégées par `tests/test_solutions.py`.

---

## Module 1 — Comprendre les données

**1.1 / 1.2** — `churn_rate_by()` renvoie le taux **et l'effectif** : un taux sans effectif peut tromper.

| contrat | taux de churn | clients | | internet | taux de churn | clients |
|---|---|---|---|---|---|---|
| Mensuel | 50,9 % | 2 794 | | Fibre | 45,4 % | 2 230 |
| Annuel | 21,5 % | 1 222 | | ADSL | 28,8 % | 1 793 |
| Biannuel | 8,9 % | 984 | | Aucun | 25,0 % | 977 |

Taux global : **35,5 %**.

**1.3** — À ajouter dans `validate()` (`src/churn/data.py`) :

```python
charges = df["monthly_charges"]
if ((charges < 0) | (charges > 500)).any():
    problems.append("Facture mensuelle hors de [0, 500] €")
```

Pourquoi pas `~charges.between(0, 500)` ? Parce qu'un `NaN` serait alors compté deux fois
(déjà signalé par la règle « valeurs manquantes »). Test correspondant :

```python
def test_validate_rejects_absurd_invoice(df):
    broken = df.copy()
    broken.loc[0, "monthly_charges"] = 600
    assert any("Facture" in p for p in validate(broken))
```

**1.4** — Exemples d'hypothèses formulées *avant* l'entraînement, en lisant `_churn_logit` :
1. Le contrat sera la variable la plus importante (effet de +1,6 et 55 % des clients sont en mensuel).
2. `total_charges` n'aura presque aucun effet propre (absent de la formule, il n'est que ancienneté × prix).
3. `senior` aura un effet réel mais une faible importance (effet modeste, seulement 18 % de seniors).

→ Vérification au module 5 : les trois se confirment.

**1.5** — Esquisse pour le jeu Kaggle « Telco Customer Churn » (testée sur un échantillon au même format) :

```python
SERVICES = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
NO = ["No", "No internet service", "No phone service"]

def load_telco(path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    raw = pd.read_csv(path)
    return pd.DataFrame({
        "customer_id": raw["customerID"],
        "tenure_months": raw["tenure"],
        "monthly_charges": raw["MonthlyCharges"],
        # Piège classique : TotalCharges est du TEXTE, vide (" ") pour les clients à 0 mois.
        "total_charges": pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0),
        "num_products": (~raw[SERVICES].isin(NO)).sum(axis=1),
        "contract": raw["Contract"].map({"Month-to-month": "Mensuel", "One year": "Annuel", "Two year": "Biannuel"}),
        "payment_method": raw["PaymentMethod"].map({
            "Bank transfer (automatic)": "Prélèvement", "Credit card (automatic)": "Carte bancaire",
            "Electronic check": "Chèque électronique", "Mailed check": "Chèque",
        }),
        "internet_service": raw["InternetService"].map({"Fiber optic": "Fibre", "DSL": "ADSL", "No": "Aucun"}),
        "tech_support": raw["TechSupport"].map(lambda v: "Oui" if v == "Yes" else "Non"),
        "senior": raw["SeniorCitizen"].map({1: "Oui", 0: "Non"}),
        "churn": (raw["Churn"] == "Yes").astype(int),
    })
```

Ce qu'il reste à adapter : `support_calls` **n'existe pas** dans ce jeu (retirez-la de
`NUMERIC_FEATURES`, du graphique et du formulaire du tableau de bord), et ajoutez
« Chèque électronique » à `PAYMENT_METHODS`. Sur ces données réelles, il n'y a plus de
« vérité cachée » : c'est tout l'intérêt de l'exercice.

**Auto-évaluation** — Connaître la vraie cause permet de *vérifier* ce que le modèle apprend
(au lieu de le croire sur parole) et de développer l'intuition pour des données réelles.

---

## Module 2 — Préparer les variables

**2.1** — `StandardScaler` : transforme chaque variable numérique en `(x − moyenne) / écart-type`,
**moyenne et écart-type appris sur le train**. Indispensable pour la régression logistique
régularisée (sinon la pénalité traite inégalement des variables en mois et en euros).
`OneHotEncoder` : remplace une catégorie par une colonne 0/1 par modalité (« Mensuel » →
`contract_Mensuel=1`, les autres à 0), car un modèle ne calcule qu'avec des nombres.

**2.2** — Variables dérivées : ROC-AUC logistique 0,834 → 0,835, boosting 0,807 → 0,811.
Pratiquement aucun effet : `avg_charge_per_month` ≈ `monthly_charges` (redondante) et le vrai
churn dépend *linéairement* des variables brutes, que le modèle voit déjà.

**2.3** — `is_new_customer` (validation croisée 5 plis) : **0,8192 ± 0,0050 → 0,8191 ± 0,0052**.
Aucun gain. L'ancienneté est déjà dans le modèle et son vrai effet est linéaire (pas de rupture
à 6 mois). Leçon : une variable n'aide que si elle apporte une information que le modèle ne peut
pas déjà représenter. Et on compare en **validation croisée**, pas sur un seul split.

**2.4** — `handle_unknown="ignore"` : une modalité jamais vue (ex. « Crypto ») est encodée en
zéros au lieu de faire planter la prédiction. En production, les données évoluent !

**2.5** — Deux expériences (`solutions/module2_variables.py`) :

| Scénario | ROC-AUC test |
|---|---|
| Normalisation **avant** le split | 0,8345 (identique à l'honnête : 0,8345) |
| Code postal aléatoire, encodé par la cible sur **tout** le jeu | **0,9268** 🚨 mirage |
| … encodé sur le train et appliqué au train (naïf) | **0,6587** 🚨 modèle trompé |
| … encodé hors-pli avec `TargetEncoder` | 0,8344 ✅ |

- La fuite par le scaler existe mais est minuscule (moyenne et écart-type de 5 000 clients ≈ ceux
  de 3 750). Elle reste une **faute** : avec une variable dont la distribution dérive dans le
  temps, l'écart devient réel.
- L'encodage par la cible montre le vrai danger : un code postal **sans aucun lien** avec le
  churn fait passer l'AUC de 0,83 à 0,93. En production, ce gain disparaît… et vos décisions
  reposent sur un chiffre faux.
- Piège subtil : encoder sur le train *et appliquer au train* fait aussi fuiter la cible (de
  chaque client d'entraînement dans sa propre variable). Le modèle apprend à faire confiance à du
  bruit et s'effondre à 0,66. La solution : l'encodage **hors-pli** (`TargetEncoder` le fait
  automatiquement dans `fit_transform`).

**Auto-évaluation** — La *data leakage*, c'est utiliser à l'entraînement une information
indisponible au moment de la prédiction réelle. Un `Pipeline` la prévient car `fit` n'est appelé
que sur les données d'entraînement (y compris dans chaque pli de validation croisée).

---

## Module 3 — Modéliser

**3.1** — La régression logistique gagne : ROC-AUC 0,834 contre 0,807.

**3.2 + 3.4** — Validation croisée 5 plis (ROC-AUC), mêmes plis pour tous les modèles :

| | pli 1 | pli 2 | pli 3 | pli 4 | pli 5 | moyenne ± écart-type |
|---|---|---|---|---|---|---|
| logistique | 0,817 | 0,822 | 0,811 | 0,825 | 0,823 | **0,819 ± 0,006** |
| gradient boosting | 0,796 | 0,801 | 0,789 | 0,816 | 0,796 | 0,799 ± 0,010 |
| forêt aléatoire | 0,810 | 0,804 | 0,810 | 0,809 | 0,803 | 0,807 ± 0,003 |

L'écart logistique − boosting est **positif sur 5 plis sur 5** (de +0,009 à +0,027) : il est
systématique, pas dû au hasard du découpage. (Un test de Student classique serait trop
optimiste, car les plis partagent leurs données d'entraînement.)

**3.3** — Les données sont **générées par un modèle logistique** : le logit est une somme
d'effets linéaires (`_churn_logit`). La régression logistique est donc le modèle « bien
spécifié ». Le boosting doit approcher des droites par des escaliers et dépense sa capacité à
chercher des interactions qui n'existent pas → il apprend une partie du bruit. Sur des données
réelles, pleines de non-linéarités et d'interactions, le boosting gagne souvent : c'est pourquoi
on **teste toujours les deux**.

**3.5** — `GridSearchCV` (sur le train uniquement) choisit `max_depth=1`, `learning_rate=0.1`,
`max_iter=500`. ROC-AUC test : **0,832** contre 0,8345 pour la logistique. Presque à égalité,
sans la battre. Explication : des arbres de profondeur 1 (« souches ») font du boosting un
**modèle additif**, exactement la forme du vrai processus. Le réglage a retrouvé la structure
des données.

**3.6** — `class_weight="balanced"` (seuil 0,5) :

| | ROC-AUC | précision | rappel | F1 |
|---|---|---|---|---|
| sans pondération | 0,834 | 68,9 % | 63,4 % | 0,660 |
| `balanced` | 0,835 | 60,2 % | **80,1 %** | 0,687 |

Le rappel monte, la précision baisse, l'AUC ne bouge pas : le **classement** des clients est
identique, seules les probabilités sont décalées vers le haut. C'est presque équivalent à
baisser le seuil, mais en perdant la calibration. Préférez choisir explicitement le seuil (module 4).

**Auto-évaluation** — Un modèle simple donne une **référence** : on sait ce que la complexité
supplémentaire rapporte vraiment. Il est aussi plus rapide, plus stable et explicable.

---

## Module 4 — Évaluer et décider

**4.1** — Seuil ↑ : on signale moins de clients, plus sûrs → précision ↑, rappel ↓. Seuil ↓ : l'inverse.

**4.2** — Le modèle « personne ne part » obtient **64,6 % d'accuracy** avec un rappel de **0 %** :
il ne détecte aucun départ. L'accuracy récompense la classe majoritaire ; avec 35 % de churn,
elle ne dit presque rien de l'utilité du modèle.

**4.3** — Seuil qui minimise `FN × coût_FN + FP × coût_FP` (jeu de test, 1 250 clients) :

| Coûts (client perdu / action inutile) | seuil optimal | précision | rappel | coût optimal | coût au seuil 0,5 |
|---|---|---|---|---|---|
| 500 € / 50 € | **0,10** | 43,9 % | 98,9 % | 30 450 € | 87 350 € |
| 100 € / 100 € | **0,50** | 68,9 % | 63,4 % | 28 900 € | 28 900 € |

Formule à retenir : si le modèle est **bien calibré**, on agit dès que
`p × coût_FN > (1 − p) × coût_FP`, soit `p > coût_FP / (coût_FP + coût_FN)` →
50 / 550 = **0,09** et 100 / 200 = **0,50**. La recherche numérique retrouve la théorie.

**4.4** — Average precision **0,709** (le hasard vaut le taux de churn, 0,354). Lecture : pour
détecter 80 % des départs, 60,5 % des clients signalés partent vraiment (seuil 0,36).
Ajout dans l'onglet Performance de `app.py` (testé) :

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

prec, rec, _ = precision_recall_curve(y_test, proba)
ap = average_precision_score(y_test, proba)
fig = go.Figure(go.Scatter(x=rec, y=prec, mode="lines", name="Modèle", line=dict(color=BLUE, width=2),
                           hovertemplate="Rappel %{x:.0%}<br>Précision %{y:.0%}<extra></extra>"))
fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="#898781", annotation_text="hasard")
fig.update_layout(title=f"Courbe précision-rappel (AP = {ap:.3f})", xaxis_title="Rappel",
                  yaxis_title="Précision", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
st.plotly_chart(fig)
```

**4.5** — Calibration (10 tranches de ~125 clients) : écart absolu moyen entre probabilité
prédite et taux observé de **3,7 points** pour la logistique (Brier 0,158) contre 4,7 pour le
boosting (Brier 0,171). Avec 125 clients par tranche, le bruit d'échantillonnage est déjà de
±9 points : la logistique est donc bien calibrée. Oui, ici « 0,7 » signifie à peu près 70 %.
Si ce n'était pas le cas : `CalibratedClassifierCV`.

**Auto-évaluation** — « Rappel de 63 % » : *sur 100 clients qui vont réellement partir, le
modèle en repère 63 à l'avance ; les 37 autres partiront sans que nous les ayons vus venir.*

---

## Module 5 — Interpréter et communiquer

**5.1** — Coefficients de la logistique ramenés en unités d'origine, comparés à `_churn_logit` :

| Effet | vrai | appris |
|---|---|---|
| ancienneté (par mois) | −0,035 | −0,029 |
| facture mensuelle (par €) | +0,018 | +0,015 |
| total facturé (par €) | 0 | −0,0001 ✅ |
| appels au support (par appel) | +0,45 | +0,42 |
| produits (par produit) | −0,25 | −0,22 |
| contrat Mensuel vs Annuel | +1,6 | +1,33 |
| contrat Biannuel vs Annuel | −0,9 | −0,80 |
| Chèque vs Prélèvement | +0,7 | +0,67 |
| Fibre vs ADSL | +0,5 | +0,65 |
| support technique | −0,8 | −0,76 |
| senior | +0,3 | +0,28 |

Tout est retrouvé, avec le bon signe. Deux écarts s'expliquent :
- **Effets légèrement sous-estimés** : le bruit ajouté au logit (`rng.normal(0, 0.6)`) agit comme
  une cause cachée, ce qui rapproche les coefficients de zéro, et la régularisation L2 les tire aussi vers 0.
- **Fibre sur-estimée** (0,65 au lieu de 0,5) : la fibre coûte ~25 € de plus par mois, donc elle
  est corrélée à la facture, dont l'effet est sous-estimé. Le modèle répartit l'effet entre deux
  variables corrélées (colinéarité).

L'importance par permutation classe `contract` (0,123) loin devant, et `senior` est à ~0 alors
que son effet est réel : **importance ≠ taille d'effet**. L'importance dépend aussi de la
fréquence (18 % de seniors) et de ce que les autres variables compensent.

**5.2** — Dans la vue d'ensemble :

```python
by_internet = df.groupby("internet_service")[TARGET].mean().reindex(INTERNET_SERVICES).reset_index()
st.plotly_chart(bar(by_internet, "internet_service", TARGET, "Taux de churn par service internet"))
```

**5.3** — Pour un modèle linéaire, la contribution de la variable *j* au logit d'un client est
`coef_j × (valeur_j − moyenne_j)` (sur les variables transformées). La somme des contributions
vaut exactement `logit(client) − logit(client moyen)`. Voir `explain()` dans
`solutions/module5_interpreter.py`. Client le plus à risque du test (95,5 %) : `support_calls`
(5 appels) +1,57, `contract` (mensuel) +0,74, `tenure_months` (5 mois) +0,47.

Dans l'onglet « Prédire un client », après `st.progress(p)` (testé ; ajoutez
`sys.path.insert(0, str(ROOT))` en haut d'`app.py` pour pouvoir importer `solutions`) :

```python
from solutions.module5_interpreter import explain

if model_name == "logistic":
    st.write("**Ce qui pèse le plus dans cette prédiction** (par rapport au client moyen) :")
    for feature, value in explain(pipeline, client, X_test).head(3).items():
        sens = "augmente" if value > 0 else "diminue"
        st.write(f"- `{feature}` {sens} le risque ({value:+.2f} sur le logit)")
```

**5.4** — Dans la barre latérale, puis remplacez `df` par `view` dans le bloc `with tab_overview:` (testé) :

```python
contracts = st.sidebar.multiselect("Contrats (vue d'ensemble)", CONTRACTS, default=CONTRACTS) or CONTRACTS
view = df[df["contract"].isin(contracts)]  # aucune sélection = tous les contrats
```

N'utilisez pas `st.stop()` pour gérer une sélection vide : cela arrêterait **tous** les onglets.

**5.5** — Avec SHAP (`pip install shap`) ; on retrouve **exactement** les valeurs de 5.3 (vérifié) :

```python
import shap
from solutions.module5_interpreter import column_owners

background = pipeline[:-1].transform(X_train)  # variables transformées (après one-hot)
masker = shap.maskers.Independent(background, max_samples=len(background))
explainer = shap.LinearExplainer(pipeline.named_steps["clf"], masker)
values = explainer(pipeline[:-1].transform(client)).values[0]
shap_by_feature = pd.Series(values).groupby(column_owners(pipeline)).sum()  # regroupe le one-hot
```

Piège : sans `max_samples`, le masker ne garde que **100 lignes** de référence et les valeurs
diffèrent légèrement. Pour le boosting, utilisez `shap.TreeExplainer`.

**Auto-évaluation** — Trois actions chiffrées (sur les données d'exemple) :
1. **Convertir les contrats mensuels** (50,9 % de churn contre 21,5 % en annuel) : remise
   sur l'engagement d'un an.
2. **Traiter la cause des appels au support** : chaque appel ajoute +0,45 au logit ; au-delà
   de 3 appels, déclencher un rappel par un conseiller.
3. **Proposer le support technique** (−0,8 au logit) aux clients fibre, qui partent à 45,4 %.

---

## Module 6 — Industrialiser

**6.1** — Chaque test protège une promesse : reproductibilité des données, validation, absence
de mutation des entrées, modèle meilleur que le hasard, signal réel appris, matrice de
confusion correcte, seuil sensible aux coûts, robustesse aux modalités inconnues.

**6.2** — `test_saved_model_gives_identical_predictions` dans `tests/test_churn.py`.

**6.3** — `.github/workflows/tests.yml` : tests sous Python 3.11 et 3.12 à chaque push.

**6.4** — `api.py` (FastAPI) + `tests/test_api.py`. Points clés : validation des entrées par
Pydantic (une faute de frappe renvoie 422 au lieu d'une prédiction fausse), modèle chargé une
fois au démarrage, chemin configurable par variable d'environnement.

**Auto-évaluation (data drift)** — Comparer régulièrement la distribution des variables récentes
à celle de l'entraînement (PSI, test de Kolmogorov-Smirnov), surveiller la distribution des
probabilités prédites, puis, quand les vraies étiquettes arrivent, la ROC-AUC sur les
nouvelles données. Réentraîner si l'écart dépasse un seuil défini à l'avance.

---

## 🏁 Exemple de mini-rapport (projet final)

> **Problème.** 35,5 % des clients résilient. Objectif : identifier à l'avance les clients à
> risque pour leur proposer une action de rétention.
>
> **Données.** 5 000 clients, 10 variables (contrat, ancienneté, facturation, support…).
> Données synthétiques validées automatiquement (aucune valeur manquante ni aberrante).
>
> **Modèle retenu.** Régression logistique : ROC-AUC 0,819 ± 0,006 en validation croisée,
> meilleure que le gradient boosting (0,799) et la forêt aléatoire (0,807), bien calibrée et
> explicable client par client.
>
> **Seuil.** Avec un client perdu à 500 € et une action inutile à 50 €, le seuil optimal est
> **0,10** : on contacte 80 % des clients du test, 98,9 % des départs sont détectés.
>
> **Impact estimé.** Sur 1 250 clients, coût 30 450 € contre 87 350 € au seuil par défaut de
> 0,5, soit **−65 %**.
>
> **Limites.** (1) Données synthétiques : à refaire sur données réelles. (2) Le calcul suppose
> qu'une action de rétention sauve 100 % des clients contactés et ne coûte rien sur les vrais
> positifs ; à affiner avec un test A/B. (3) Modèle à surveiller dans le temps (data drift).

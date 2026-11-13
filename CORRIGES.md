# ✅ Corrigés du parcours

> ⚠️ **Essayez d'abord, vérifiez ensuite.** On apprend en cherchant, pas en lisant la solution.
> Règle utile : 30 minutes de recherche honnête sur un exercice avant d'ouvrir son corrigé.

Chaque module a un script exécutable dans [`corriges/`](corriges/) qui reproduit les chiffres ci-dessous :

```bash
python corriges/module_1_donnees.py
python corriges/module_2_variables.py
python corriges/module_3_modeles.py        # ≈ 20 s
python corriges/module_4_evaluation.py
python corriges/module_5_interpretation.py
```

Les chiffres viennent du jeu par défaut (5 000 clients, `seed=42`). Si vous changez les données,
vos chiffres changeront : c'est normal, comparez surtout les **tendances**.

---

## Module 1 — Comprendre les données

**1.1** — 5 000 clients, **35,5 %** de churn. Ancienneté médiane 14 mois, facture moyenne 61 €.

**1.2** — Le contrat est de loin le facteur le plus visible :

| Contrat | Churn | | Internet | Churn |
|---|---|---|---|---|
| Mensuel | 50,9 % | | Fibre | 45,4 % |
| Annuel | 21,5 % | | ADSL | 28,8 % |
| Biannuel | 8,9 % | | Aucun | 25,0 % |

**1.3** — Voir `validate_avec_facture()` dans `corriges/module_1_donnees.py` : trois lignes à ajouter à
`validate()` et un test qui casse volontairement une facture (`9999 €`) pour vérifier que la règle se déclenche.

**1.4** — Exemples d'hypothèses tirées de `_churn_logit` : (1) le contrat mensuel sera la variable n°1
(coefficient 1,6, le plus fort) ; (2) l'ancienneté protège (−0,035 par mois, soit −2,5 sur 72 mois) ;
(3) `total_charges` n'a aucun effet propre (elle n'apparaît pas dans la formule).
Le module 5 montre que ces trois hypothèses se vérifient.

**1.5** — Correspondance des colonnes du jeu Kaggle « Telco Customer Churn »
(*non testé ici : le téléchargement nécessite un compte Kaggle*) :

```python
raw = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
services = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
df = pd.DataFrame({
    "customer_id": raw["customerID"],
    "tenure_months": raw["tenure"].clip(lower=1),
    "monthly_charges": raw["MonthlyCharges"],
    # Piège : TotalCharges est du TEXTE, avec des blancs pour les nouveaux clients.
    "total_charges": pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(raw["MonthlyCharges"]),
    "num_products": (raw[services] == "Yes").sum(axis=1).clip(lower=1),
    "contract": raw["Contract"].map({"Month-to-month": "Mensuel", "One year": "Annuel", "Two year": "Biannuel"}),
    "payment_method": raw["PaymentMethod"],
    "internet_service": raw["InternetService"].map({"Fiber optic": "Fibre", "DSL": "ADSL", "No": "Aucun"}),
    "tech_support": raw["TechSupport"].map({"Yes": "Oui"}).fillna("Non"),
    "senior": raw["SeniorCitizen"].map({1: "Oui", 0: "Non"}),
    "churn": (raw["Churn"] == "Yes").astype(int),
})
```

`support_calls` n'existe pas dans ce jeu : retirez-la de `NUMERIC_FEATURES` plutôt que d'inventer des valeurs.

**Auto-évaluation** — Connaître la vraie cause permet de *vérifier* son intuition et ses outils
(importance des variables, détection de fuite). Sur des données réelles, on ne l'a jamais : on s'entraîne
donc là où on peut se corriger.

---

## Module 2 — Préparer les variables

**2.1** — `StandardScaler` centre et réduit chaque variable numérique (moyenne 0, écart-type 1) pour que
la régression logistique traite « 1 mois » et « 1 € » sur une échelle comparable. `OneHotEncoder`
transforme une catégorie (« Mensuel ») en colonnes 0/1, car un modèle ne sait multiplier que des nombres.

**2.2 / 2.3** — ROC-AUC sur le jeu de test :

| Variables | ROC-AUC |
|---|---|
| Brutes | 0,8345 |
| + `avg_charge_per_month`, `calls_per_year` | 0,8345 |
| + `is_new_customer` | 0,8346 |

**Aucun gain** — et c'est la bonne leçon : ces variables sont des combinaisons de variables déjà
présentes, et le vrai churn est généré par une formule linéaire que le modèle capte déjà entièrement.
Sur des données réelles, le feature engineering rapporte souvent beaucoup plus.

**2.4** — `handle_unknown="ignore"` encode une modalité jamais vue (ex. un nouveau moyen de paiement
« Crypto ») comme « aucune des modalités connues » au lieu de lever une erreur. Sans lui, un seul client
atypique ferait planter toute la chaîne de prédiction en production.

**2.5** — Deux expériences dans `corriges/module_2_variables.py` :

| Expérience | ROC-AUC |
|---|---|
| Scaler entraîné sur tout le jeu (fuite « douce ») | 0,8345 — identique |
| **2 000 colonnes de bruit pur**, sélection des 20 meilleures *avant* la validation croisée | **0,838** 😱 |
| Même bruit, sélection *dans* le pipeline | 0,406 (hasard) |

Avec la fuite, du bruit pur obtient le même score que notre vrai modèle ! La fuite « douce » ne change
rien ici, ce qui est justement dangereux : on prend une mauvaise habitude sans la voir, et elle explose le
jour où l'étape fuyante est plus puissante (sélection de variables, encodage par la cible, imputation…).

**Auto-évaluation** — Il y a fuite de données quand une information du jeu de test (ou du futur)
influence l'entraînement. Le `Pipeline` garantit que *chaque* étape (`fit`) n'apprend que sur les lignes
d'entraînement, y compris à l'intérieur de chaque pli de validation croisée.

---

## Module 3 — Modéliser

**3.1** — La régression logistique gagne (0,834 contre 0,807).

**3.2 / 3.4** — Validation croisée à 5 plis (mêmes plis pour tous les modèles) :

| Modèle | ROC-AUC moyenne | Écart-type | Écart avec la logistique, pli par pli |
|---|---|---|---|
| Régression logistique | **0,819** | 0,005 | — |
| Forêt aléatoire | 0,806 | 0,003 | −0,013 ± 0,007, perd 5 plis sur 5 |
| Gradient boosting | 0,799 | 0,009 | −0,020 ± 0,006, perd 5 plis sur 5 |

Comparer les modèles **pli par pli** est plus fiable que comparer deux moyennes : un écart qui va
toujours dans le même sens est un vrai signal.

**3.3** — Le churn est généré par une **formule linéaire** dans l'espace du logit (`_churn_logit`) :
c'est *exactement* la forme d'une régression logistique. Le modèle simple est « bien spécifié » ; les
arbres doivent approcher des effets linéaires par des marches d'escalier et gaspillent de la variance.

**3.5** — Meilleure grille : `max_depth=2`, `learning_rate=0.1`, `max_iter=100` → **0,812** en CV.
Mieux qu'avant (0,799), mais toujours derrière la logistique (0,819). Le gagnant de la grille utilise des
arbres de profondeur 2 : le boosting se rapproche de lui-même d'un modèle additif.

**3.6** — Au seuil 0,5 sur le jeu de test :

| `class_weight` | ROC-AUC | Précision | Rappel | Clients signalés |
|---|---|---|---|---|
| `None` | 0,834 | 68,9 % | 63,4 % | 408 |
| `"balanced"` | 0,835 | 60,2 % | 80,1 % | 590 |

L'AUC ne bouge pas (le *classement* des clients est le même) : rééquilibrer les classes revient à
déplacer le seuil. Choisir le seuil explicitement avec les coûts métier (module 4) est plus transparent.

**Auto-évaluation** — Un modèle simple donne une référence en quelques minutes, s'explique, et
révèle si les modèles complexes apportent vraiment quelque chose. Ici, ils n'apportent rien.

---

## Module 4 — Évaluer et décider

**4.1** — Seuil qui monte → on signale moins de clients : la précision monte, le rappel baisse.

**4.2** — Prédire « personne ne part » donne **64,6 % d'accuracy** et 0 % de rappel : un modèle
inutile qui a l'air correct. Notre modèle n'ajoute « que » 12 points d'accuracy, mais détecte 63 % des départs.

**4.3** — Seuils optimaux (jeu de test) :

| Client perdu | Action inutile | Seuil | Précision | Rappel |
|---|---|---|---|---|
| 500 € | 50 € | **0,10** | 44 % | 99 % |
| 100 € | 100 € | **0,50** | 69 % | 63 % |

Règle théorique pour un modèle bien calibré : seuil ≈ coût_FP / (coût_FP + coût_FN), soit
50 / 550 ≈ 0,09 et 100 / 200 = 0,50. Les seuils trouvés collent à la théorie.

**4.4** — *Average precision* = 0,709 (le hasard vaudrait 0,354, le taux de churn). Pour détecter
80 % des départs, on ne peut pas dépasser 61 % de précision. Code pour l'onglet Performance :

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, proba)
fig = px.line(x=recall, y=precision, labels={"x": "Rappel", "y": "Précision"},
              title=f"Courbe précision-rappel (AP = {average_precision_score(y_test, proba):.3f})",
              color_discrete_sequence=[BLUE])
fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="#898781", annotation_text="hasard")
fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", margin=dict(t=50, b=10))
st.plotly_chart(fig, width="stretch")
```

**4.5** — Oui, à peu près : écart absolu moyen de 3,7 points entre probabilité prédite et churn observé
(10 tranches de 125 clients, bruit d'échantillonnage ≈ ±4,5 points par tranche). La régression logistique
est naturellement bien calibrée. Refaites l'exercice avec le boosting pour comparer.

**4.6 (bonus)** — `best_threshold_by_cost` suppose que l'action de rétention est gratuite et réussit
toujours sur un vrai churner. Avec `campaign_profit()` (client = 500 €, action = 50 € par contact) :

| Succès de l'action | Seuil optimal | Clients contactés (sur 1 250) | Gain net |
|---|---|---|---|
| 100 % | 0,14 | 926 | 169 200 € |
| 50 % | 0,26 | 737 | 64 400 € |
| 30 % | 0,30 | 673 | 24 700 € |
| 10 % | — | 0 | 0 € (ne rien faire) |

La donnée qui pèse le plus sur la valeur du projet n'est pas l'AUC, mais le **taux de succès réel des
actions de rétention**, qui se mesure par un test A/B (groupe contacté contre groupe témoin).

**Auto-évaluation** — « Rappel de 63 % » : sur 100 clients qui vont partir, l'outil en repère 63 à
temps ; les 37 autres passent sous le radar.

---

## Module 5 — Interpréter et communiquer

**5.1** — Vrai effet (écart-type de chaque terme de `_churn_logit`) contre importance par permutation :

| Rang réel | Variable | Rang trouvé par le modèle |
|---|---|---|
| 1 | contract | 1 |
| 2 | tenure_months | 2 |
| 3 | support_calls | 3 |
| 4 | monthly_charges | 4 |
| 5 | tech_support | 5 |
| 6 | num_products | 6 |
| 7 | payment_method | 8 |
| 8 | internet_service | 7 |
| 9 | senior | 10 |
| 10 | total_charges (aucun effet réel) | 9 |

Les six grandes causes sont retrouvées **dans l'ordre exact**. Les petits effets se mélangent : ils sont
du même ordre que le bruit d'un jeu de test de 1 250 clients. `senior` (effet réel faible) est
sous-estimée à zéro ; `total_charges` (aucun effet réel) emprunte un peu d'importance à l'ancienneté
et à la facture, avec lesquelles elle est corrélée.

**5.2** — Dans l'onglet « Vue d'ensemble » :

```python
by_net = df.groupby("internet_service")[TARGET].mean().reindex(INTERNET_SERVICES).reset_index()
st.plotly_chart(bar(by_net, "internet_service", TARGET, "Taux de churn par service internet"),
                width="stretch")
```

**5.3** — `explain_logistic()` dans `corriges/module_5_interpretation.py` : contribution de chaque
variable = coefficient × (valeur du client − valeur moyenne), colonnes one-hot regroupées.
Pour le client le plus à risque du jeu de test (95,5 %) : appels au support **+1,57**, contrat
mensuel **+0,74**, ancienneté de 5 mois **+0,47**. Pour l'afficher dans « Prédire un client »,
copiez `explain_logistic` dans `src/churn/model.py` (avec l'import de `CATEGORICAL_FEATURES`), puis
sous `st.progress(p)` :

```python
from churn.model import explain_logistic

if model_name == "logistic":
    X_train = df.loc[df.index.difference(X_test.index), FEATURES]
    for var, val in explain_logistic(pipeline, client, X_train).head(3).items():
        st.write(f"{'🔺' if val > 0 else '🔻'} **{var}** ({val:+.2f})")
```

**5.4** — Dans la barre latérale, puis utilisez `view` au lieu de `df` dans la vue d'ensemble :

```python
contrats = st.sidebar.multiselect("Contrats affichés", CONTRACTS, default=CONTRACTS)
view = df[df["contract"].isin(contrats)]
if view.empty:
    st.warning("Sélectionnez au moins un contrat.")
    st.stop()
```

**5.5** — Pour un modèle linéaire, SHAP calcule *exactement* la contribution de l'exercice 5.3 :
β × (x − moyenne). Vous l'avez donc déjà implémenté ! Pour le boosting, utilisez `shap.TreeExplainer`
sur le classifieur et les données transformées par `pipeline.named_steps["prep"]`.

**Auto-évaluation** — Trois actions, chiffres à l'appui : (1) proposer un passage au contrat annuel
aux clients mensuels (50,9 % de churn contre 21,5 %) ; (2) traiter en priorité les clients ayant appelé
le support 3 fois ou plus (variable n°3) ; (3) soigner l'accueil des 6 premiers mois, quand le risque
est maximal. Chaque action doit être testée en A/B avant d'être généralisée (voir 4.6).

---

## Module 6 — Industrialiser

Ce module est **réalisé dans le dépôt** : lisez le code comme un exemple de référence.

| Exercice | Où |
|---|---|
| 6.1 — Rôle de chaque test | Les docstrings de `tests/` expliquent ce que chaque test protège |
| 6.2 — Modèle sauvegardé identique | `test_saved_model_gives_identical_predictions` dans `tests/test_churn.py` |
| 6.3 — GitHub Action | `.github/workflows/tests.yml` (Python 3.11 et 3.12) |
| 6.4 — API FastAPI | `api.py` + `tests/test_api.py` |
| 6.5 — Déploiement | Étapes ci-dessous (nécessite votre compte) |

**6.5 — Déployer sur Streamlit Community Cloud** : poussez le dépôt sur GitHub → sur
[share.streamlit.io](https://share.streamlit.io), « Create app » → choisissez le dépôt, la branche `main` et
le fichier `app.py` → « Deploy ». Les dépendances sont lues dans `requirements.txt` et les données sont
générées au premier lancement. Ajoutez ensuite le lien au README.

**Auto-évaluation** — Pour détecter le *data drift* : enregistrer la distribution des variables à
l'entraînement, puis comparer chaque mois les nouvelles données (test de Kolmogorov-Smirnov ou PSI par
variable) et suivre l'AUC réelle dès que les départs sont connus. Alerte → ré-entraînement.

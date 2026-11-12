# ✅ Corrigés du parcours

> ⚠️ **Essayez d'abord.** On apprend en cherchant, pas en lisant la réponse. Revenez ici
> seulement après avoir tenté l'exercice, pour comparer votre raisonnement.

Tous les chiffres ci-dessous sont **reproductibles** :

```bash
python scripts/corriges.py            # tous les modules (~20 s)
python scripts/corriges.py --module 3 # un seul module
```

Ils correspondent aux données par défaut (`seed=42`, 5 000 clients, jeu de test = 1 250 clients).
Si vous changez la graine ou les données, vos chiffres seront différents, et c'est normal.
C'est le raisonnement qui compte.

---

## Module 1 : Comprendre les données

**1.1** Taux de churn global : **35,5 %**.

```python
import pandas as pd
df = pd.read_csv("data/customers.csv")
df.describe()
df["churn"].mean()   # 0.355
```

À remarquer dans `describe()` : l'ancienneté a une **moyenne de 21,3 mois** mais une **médiane
de 14**. La distribution est asymétrique : beaucoup de clients récents, quelques très anciens.
Le premier réflexe face à un tel écart est de tracer un histogramme.

**1.2**

| Contrat | Taux de churn | Clients |
|---|---|---|
| Mensuel | **50,9 %** | 2 794 |
| Annuel | 21,5 % | 1 222 |
| Biannuel | 8,9 % | 984 |

| Internet | Taux de churn | Clients |
|---|---|---|
| Fibre | **45,4 %** | 2 230 |
| ADSL | 28,8 % | 1 793 |
| Aucun | 25,0 % | 977 |

```python
df.groupby("contract")["churn"].agg(["mean", "size"])
```

**1.3** Dans `validate()` :

```python
    if ((df["monthly_charges"] < 0) | (df["monthly_charges"] > 500)).any():
        problems.append("Facture mensuelle hors de [0, 500] €")
```

Et le test dans `tests/test_churn.py` :

```python
def test_validate_detects_absurd_bill(df):
    broken = df.copy()
    broken.loc[0, "monthly_charges"] = 9999
    assert any("Facture" in p for p in validate(broken))
```

L'API (`api.py`) applique déjà la même règle (`Field(ge=0, le=500)`) : la validation doit être
**cohérente** à l'entraînement et en production.

**1.4** Exemples d'hypothèses raisonnables, tirées de `_churn_logit` :

1. Le **contrat** sera la variable n°1 : gros coefficient (+1,6 pour Mensuel) **et** 55 % des clients concernés.
2. `senior` pèsera peu : petit coefficient (+0,3) et seulement 18 % de seniors.
3. `total_charges` n'apparaît pas dans la vérité, donc son importance devrait être quasi nulle.

Vous vérifierez ces hypothèses à l'exercice 5.1.

**1.5** Pièges classiques du jeu Telco de Kaggle :

| Telco (Kaggle) | Ce projet | Transformation |
|---|---|---|
| `tenure` | `tenure_months` | renommer |
| `MonthlyCharges` | `monthly_charges` | renommer |
| `TotalCharges` | `total_charges` | ⚠️ c'est du **texte** avec des espaces (clients à 0 mois) : `pd.to_numeric(..., errors="coerce")`, puis remplir par 0 |
| `Contract` | `contract` | `{"Month-to-month": "Mensuel", "One year": "Annuel", "Two year": "Biannuel"}` |
| `InternetService` | `internet_service` | `{"Fiber optic": "Fibre", "DSL": "ADSL", "No": "Aucun"}` |
| `TechSupport` | `tech_support` | 3 modalités (`No internet service` en plus) |
| `SeniorCitizen` | `senior` | 0/1 → `"Non"`/`"Oui"` |
| `Churn` | `churn` | `"Yes"`/`"No"` → 1/0 |

`support_calls` et `num_products` n'existent pas dans Telco. Retirez-les de `NUMERIC_FEATURES`, ou
reconstruisez `num_products` en comptant les services souscrits (`PhoneService`, `OnlineSecurity`, …).

**Auto-évaluation.** Sur des données réelles, on ne connaît jamais la vraie cause. Ici, on peut
vérifier si le modèle retrouve la vérité, et donc **calibrer son intuition** : savoir quand une
importance de variable est fiable, et quand elle est trompeuse.

---

## Module 2 : Préparer les variables

**2.1**
- `StandardScaler` transforme chaque variable en `(x − moyenne) / écart-type`. La moyenne et
  l'écart-type sont **appris sur le jeu d'entraînement**. La régression logistique converge alors
  mieux, et la régularisation traite toutes les variables à égalité.
- `OneHotEncoder` crée une colonne 0/1 par modalité : `contract = Mensuel` devient
  `[1, 0, 0]`. Un modèle ne sait manipuler que des nombres.

**2.2** ROC-AUC de **0,834 → 0,835** : un gain négligeable. Pourquoi ?
- `avg_charge_per_month` ≈ `monthly_charges`, puisque le total est généré comme
  facture × ancienneté × (0,95 à 1,05). La variable est **redondante**.
- `calls_per_year` : la vérité utilise le nombre **brut** d'appels, pas un taux.

**2.3** En validation croisée 5 plis :

| | ROC-AUC |
|---|---|
| sans `is_new_customer` | 0,8192 ± 0,0050 |
| avec `is_new_customer` | 0,8191 ± 0,0052 |

**Aucun effet.** Dans la vérité, l'ancienneté agit de façon **linéaire** (−0,035 par mois). Il n'y a
pas de rupture à 6 mois, et `tenure_months` contient déjà toute l'information.
Leçon : une variable ne sert que si elle apporte quelque chose que le modèle ne voit pas déjà.
Comparez **toujours** en validation croisée : sur un seul split, ±0,005 n'est que du bruit.

```python
def add_new(d):
    return d.assign(is_new_customer=(d["tenure_months"] < 6).astype(int))
# → FunctionTransformer(add_new) en tête du Pipeline, et "is_new_customer" ajouté aux numériques.
```

**2.4** Avec `handle_unknown="ignore"`, une modalité jamais vue (ex. `"Crypto"`) est encodée
`[0, 0, 0, 0]` au lieu de lever une erreur. La prédiction reste possible en production.
Remarquez qu'il y a **deux couches** : l'API rejette les modalités inconnues (HTTP 422), car c'est
une erreur de saisie, et le modèle, lui, reste robuste s'il en reçoit une quand même.

**2.5** Le résultat est surprenant, et c'est toute la leçon :

| Expérience | Fuite | Sans fuite |
|---|---|---|
| (a) Scaler appris sur **tout** le jeu, avant le split | AUC 0,8345 | AUC 0,8345 |
| (b) 20 « meilleures » variables choisies parmi **2 000 variables de bruit pur**, sur tout le jeu (300 clients) | AUC **0,794** 😱 | AUC 0,452 (hasard) |

- (a) Une fuite par le scaler ne transmet qu'une moyenne et un écart-type. Avec 5 000 lignes,
  c'est **invisible**. Ne concluez pas pour autant que « la fuite, ce n'est pas grave ».
- (b) Voici la fuite dangereuse. En sélectionnant les variables **avant** la validation, on choisit
  celles qui, par hasard, collent à la cible du jeu de test. On obtient 0,79 d'AUC sur des données
  **qui ne contiennent aucune information**. En production, ce modèle vaudrait 0,5.

**Auto-évaluation.** La *data leakage* consiste à utiliser, pendant l'entraînement, une
information indisponible au moment de la prédiction (souvent : des statistiques du jeu de test).
Un `Pipeline` est ré-entraîné **dans chaque pli** : chaque étape (`fit`) ne voit que les données
d'entraînement de ce pli.

---

## Module 3 : Modéliser

**3.1** La régression logistique gagne : ROC-AUC 0,834 contre 0,807 pour le boosting.

**3.2** Dans `src/churn/model.py` :

```python
from sklearn.ensemble import RandomForestClassifier

MODELS["random_forest"] = "Forêt aléatoire"
# dans build_pipeline :
    elif model_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)
```

Résultat : AUC test **0,811**, entre les deux autres modèles.

**3.3** Comparez l'AUC sur le jeu d'entraînement et sur le jeu de test :

| Modèle | AUC train | AUC test | Écart |
|---|---|---|---|
| Logistique | 0,816 | 0,834 | −0,018 |
| Gradient boosting | **0,955** | 0,807 | **+0,149** |
| Forêt aléatoire | 0,928 | 0,811 | +0,117 |

Les données sont **générées par un modèle logistique** : le logit est une somme linéaire des
variables, plus un bruit gaussien. La régression logistique est donc exactement la bonne forme
de modèle. Les arbres doivent approcher des droites par des « escaliers » et, avec 3 750 clients,
ils apprennent aussi le **bruit** : c'est le sur-apprentissage (0,955 en train, 0,807 en test).

⚠️ Ne généralisez pas : sur des données réelles avec des interactions et des effets non linéaires,
le boosting gagne souvent. **Il faut toujours tester.**

**3.4** Validation croisée 5 plis (mêmes plis pour tous les modèles) :

| Modèle | ROC-AUC | Plis |
|---|---|---|
| Logistique | **0,819 ± 0,005** | 0,817 · 0,822 · 0,811 · 0,825 · 0,822 |
| Forêt aléatoire | 0,806 ± 0,003 | 0,811 · 0,803 · 0,807 · 0,808 · 0,803 |
| Gradient boosting | 0,799 ± 0,009 | 0,796 · 0,801 · 0,789 · 0,816 · 0,796 |

- L'écart est **significatif** : la logistique bat le boosting sur **les 5 plis**, de 0,009 à
  0,026. Comme les plis sont identiques, on compare pli par pli (comparaison appariée).
- Le 0,834 du README venait d'un split **favorable**. L'estimation honnête est 0,819.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(5, shuffle=True, random_state=42)
cross_val_score(build_pipeline("logistic"), df[FEATURES], df[TARGET], cv=cv, scoring="roc_auc")
```

**3.5** Avec `GridSearchCV` sur `learning_rate`, `max_depth`, `max_iter` et `l2_regularization` :
meilleur boosting à **0,812** (`max_depth=2`, `learning_rate=0.1`, `max_iter=100`). Il ne bat
toujours pas la logistique (0,819).
Remarquez que la recherche choisit les arbres **les plus simples** : elle pousse le boosting vers
un comportement quasi linéaire. Autre point : ce 0,812 est légèrement optimiste, car il a servi à
choisir les hyperparamètres. Pour une estimation sans biais, il faut une *validation croisée imbriquée*.

**3.6**

| `class_weight` | Précision | Rappel | F1 | AUC |
|---|---|---|---|---|
| `None` | 0,689 | 0,634 | 0,660 | 0,834 |
| `"balanced"` | 0,602 | **0,801** | 0,687 | 0,835 |

Le rappel augmente et la précision baisse, mais **l'AUC ne change quasiment pas**. Pondérer les classes
pousse les probabilités vers le haut, ce qui revient à **baisser le seuil**. Le classement des
clients reste le même. Inconvénient : les probabilités ne sont plus calibrées (voir 4.5).
Pour le même effet sans cet inconvénient, gardez le modèle et **ajustez le seuil** (module 4).

**Auto-évaluation.** On commence par un modèle simple parce qu'il est rapide, interprétable et
qu'il fixe **la barre à battre**. Ici, il n'a jamais été battu.

---

## Module 4 : Évaluer et décider

**4.1** Sur les 1 250 clients de test (logistique) :

| Seuil | Précision | Rappel | Clients signalés |
|---|---|---|---|
| 0,3 | 0,578 | **0,878** | 673 |
| 0,5 | 0,689 | 0,634 | 408 |
| 0,7 | **0,804** | 0,325 | 179 |

Baisser le seuil signale plus de clients : on rate moins de départs (rappel ↑), mais on se
trompe plus souvent (précision ↓).

**4.2** Un modèle qui prédit « reste » pour tout le monde obtient **64,6 %** d'accuracy… et un
rappel de **0**, puisqu'il ne détecte aucun départ. Notre modèle (76,9 %) ne fait « que » 12 points
de mieux en accuracy, alors qu'il détecte 63 % des départs. L'accuracy masque ce qui compte.

**4.3**

| Coût client perdu / action inutile | Seuil optimal | Coût total | Coût au seuil 0,5 |
|---|---|---|---|
| 500 € / 50 € | **0,10** | 30 450 € | 87 350 € (−65 %) |
| 100 € / 100 € | **0,50** | 28 900 € | 28 900 € |

Ces seuils se retrouvent **par le calcul**. On agit sur un client si le coût attendu de
l'inaction dépasse celui de l'action : `p × C_FN > (1 − p) × C_FP`, soit
`p > C_FP / (C_FN + C_FP)`.
- 50 / (500 + 50) = **0,09** ≈ 0,10 ✔
- 100 / (100 + 100) = **0,50** ✔

Cette formule ne fonctionne que si les probabilités sont **calibrées** (voir 4.5).
Limite du modèle de coût : il suppose qu'une action sur un vrai churner le retient à coup sûr et
gratuitement. Pour aller plus loin, introduisez un taux de succès de la rétention et un coût par
action.

**4.4**

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

prec, rec, _ = precision_recall_curve(y_test, proba)
fig = go.Figure(go.Scatter(x=rec, y=prec, mode="lines", line=dict(color=BLUE, width=2),
                           hovertemplate="Rappel %{x:.2f}<br>Précision %{y:.2f}<extra></extra>"))
fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="#898781",
              annotation_text="hasard (taux de churn)")
fig.update_layout(title=f"Précision-rappel (AP = {average_precision_score(y_test, proba):.3f})",
                  xaxis_title="Rappel", yaxis_title="Précision")
st.plotly_chart(fig)
```

La ligne du hasard n'est pas à 0,5 mais au **taux de churn** (0,355) : c'est la précision obtenue
en signalant tout le monde.

**4.5** Brier score : **0,158**. Taux de churn observé par tranche de probabilité prédite :

| Proba prédite | 0,02 | 0,07 | 0,13 | 0,21 | 0,29 | 0,38 | 0,47 | 0,57 | 0,69 | 0,82 |
|---|---|---|---|---|---|---|---|---|---|---|
| Churn observé | 0,00 | 0,04 | 0,12 | 0,14 | 0,30 | 0,38 | 0,49 | 0,64 | **0,58** | 0,86 |

Le modèle est **plutôt bien calibré** : les points suivent la diagonale. La régression logistique
optimise directement la log-loss, elle est donc généralement bien calibrée. C'est ce qui rend la
formule du 4.3 utilisable. Chaque tranche ne compte qu'environ 125 clients, donc des écarts de
quelques points relèvent du bruit. La tranche « 0,69 → 0,58 » est la plus éloignée : une
probabilité de 0,7 signifie ici plutôt **60 à 70 %** de départs réels.

```python
from sklearn.calibration import CalibrationDisplay
CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, strategy="quantile")
```

**Auto-évaluation** (« rappel de 63 % » expliqué à un manager) :
*« Sur 100 clients qui vont vraiment partir, le modèle en repère 63 à l'avance. Les 37 autres
partiront sans qu'on les ait vus venir. On peut en repérer davantage, au prix de plus de fausses
alertes. »*

---

## Module 5 : Interpréter et communiquer

**5.1** Importance par permutation (logistique) comparée à la vérité de `_churn_logit` :

| Variable | Baisse d'AUC | Verdict |
|---|---|---|
| `contract` | 0,123 | ✔ retrouvée n°1 (gros effet × beaucoup de clients) |
| `tenure_months` | 0,051 | ✔ |
| `support_calls` | 0,042 | ✔ |
| `monthly_charges` | 0,023 | ✔ |
| `tech_support` | 0,020 | ✔ |
| `num_products` | 0,011 | ✔ faible effet |
| `internet_service` | 0,010 | ⚠️ **sous-estimée** |
| `payment_method` | 0,006 | ⚠️ **sous-estimée** |
| `total_charges` | 0,002 | ✔ absente de la vérité |
| `senior` | 0,000 | ⚠️ **non détectée** |

L'importance dépend du **coefficient × nombre de clients concernés** :
- `payment_method` : Chèque a un coefficient fort (+0,7), mais ne concerne que 15 % des clients.
- `senior` : petit effet (+0,3) sur 18 % des clients, donc noyé dans le bruit.
- `internet_service` : la Fibre coûte plus cher. Son effet est en partie capté par
  `monthly_charges`, car les deux variables sont **corrélées**. L'importance par permutation se
  partage mal entre variables corrélées, et c'est sa principale limite.

**5.2** Comme les autres graphiques de la vue d'ensemble :

```python
by_internet = df.groupby("internet_service")[TARGET].mean().reindex(INTERNET_SERVICES).reset_index()
st.plotly_chart(bar(by_internet, "internet_service", TARGET, "Taux de churn par service internet"))
```

**5.3** Pour une régression logistique, la contribution de chaque variable au logit vaut
`valeur transformée × coefficient` :

```python
prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
contrib = pd.Series(prep.transform(client).ravel() * clf.coef_.ravel(),
                    index=prep.get_feature_names_out())
top3 = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(3)
```

Exemple (client 3993, probabilité 0,25) :

| Variable | Contribution | Lecture |
|---|---|---|
| `contract_Mensuel` | +1,01 | pousse vers le départ |
| `tech_support_Oui` | −0,59 | le retient |
| `support_calls` | −0,55 | il appelle **moins** que la moyenne |

Les numériques étant standardisées, leur contribution se lit **par rapport au client moyen**.

**5.4**

```python
contracts = st.sidebar.multiselect("Contrats", CONTRACTS, default=CONTRACTS)
view = df[df["contract"].isin(contracts)]
# puis remplacer df par view dans l'onglet « Vue d'ensemble » (et gérer le cas view vide)
```

**5.5** Pour un modèle linéaire, la valeur SHAP d'une variable vaut exactement
`coefficient × (x − moyenne de x)`. C'est le calcul du 5.3, **centré**. Avec `shap` :

```python
import shap
X_bg = prep.transform(X_train)
explainer = shap.LinearExplainer(clf, X_bg)
values = explainer(prep.transform(client))
shap.plots.waterfall(values[0])  # dans Streamlit : st.pyplot(...)
```

L'intérêt de SHAP : la même méthode fonctionne aussi pour le boosting (`shap.TreeExplainer`).

**Auto-évaluation**, trois actions de rétention chiffrées :

1. **Convertir les contrats mensuels** : 50,9 % de churn, contre 21,5 % en annuel. Les départs
   mensuels représentent **94 374 €** de revenu mensuel. Offrez, par exemple, un mois gratuit pour
   un passage à l'annuel.
2. **Intervenir dès le 2ᵉ appel au support** : le churn passe de 30,5 % (0–1 appel) à 39,2 %
   (2 appels), puis à **51,2 %** (3 appels et plus). Déclenchez un rappel proactif par un
   conseiller senior.
3. **Proposer le support technique** : 28,4 % de churn avec, 39,2 % sans. Offrez un essai
   gratuit aux clients à risque, en priorité ceux qui ont la Fibre (45,4 % de churn).

⚠️ Corrélation n'est pas causalité. Ici, on **sait** que ces effets sont causaux (données
synthétiques). Dans la réalité, validez chaque action par un **test A/B** avant de la généraliser.

---

## Module 6 : Industrialiser

**6.1** Ce que protège chaque test :

| Test | Protège contre |
|---|---|
| `test_generation_is_reproducible` | des résultats qui changent à chaque exécution (graine oubliée) |
| `test_generated_data_is_valid` | un générateur cassé ou un taux de churn absurde |
| `test_validate_detects_problems` | des contrôles qualité qui ne contrôlent plus rien |
| `test_engineered_features_do_not_mutate_input` | une fonction qui modifie en douce le DataFrame de l'appelant |
| `test_models_beat_random` | un modèle cassé (AUC ≈ 0,5) ou des probabilités hors de [0, 1] |
| `test_model_learns_true_signal` | un modèle qui a appris l'inverse de la réalité (encodage inversé…) |
| `test_evaluate_confusion_matrix_sums` | une erreur dans le calcul des métriques |
| `test_costly_misses_lower_the_threshold` | une optimisation du seuil qui ne respecte pas la logique métier |
| `test_unknown_model_raises` | une faute de frappe qui passerait en silence |
| `test_unknown_category_is_ignored` | un plantage en production sur une nouvelle modalité |
| `test_saved_model_gives_same_predictions` | un modèle sauvegardé qui ne prédit pas comme le modèle entraîné |
| `tests/test_api.py` | une API qui accepte des entrées absurdes ou renvoie un mauvais format |
| `tests/test_app.py` | un tableau de bord qui plante sur un réglage |

**6.2** Voir `test_saved_model_gives_same_predictions` dans `tests/test_churn.py`. On compare
les **probabilités** avec `assert_allclose` plutôt que les classes : deux modèles différents
peuvent prédire les mêmes classes avec des probabilités différentes. La comparaison des
probabilités détecte donc beaucoup plus de problèmes.

**6.3** Voir `.github/workflows/tests.yml` : à chaque push, GitHub installe les dépendances sous
Python 3.11 et 3.13, lance `pytest`, puis l'entraînement complet.

**6.4** Voir `api.py`. Points clés :
- le modèle est chargé **une fois** au démarrage (`lifespan`), pas à chaque requête ;
- Pydantic valide les entrées : une modalité inconnue ou une valeur hors bornes renvoie HTTP 422 ;
- la documentation interactive est générée automatiquement sur `/docs`.

**6.5**
1. Poussez le dépôt sur GitHub.
2. Sur share.streamlit.io : *New app*, choisissez le dépôt, votre branche principale et le fichier `app.py`.
3. `requirements.txt` est détecté automatiquement. Les données sont générées au premier lancement.
4. Copiez l'URL obtenue dans le README.

**Auto-évaluation** (*data drift*) :
- **Surveiller les entrées** : comparez la distribution de chaque variable entre l'entraînement
  et le mois en cours (test de Kolmogorov-Smirnov, ou indice PSI). `validate()` est la première
  barrière.
- **Surveiller les sorties** : la part de clients prédits « à risque » change-t-elle brutalement ?
- **Surveiller la performance** : une fois les vrais départs connus (1 à 3 mois plus tard),
  recalculez rappel et précision.
- **Ré-entraîner** régulièrement, et dès qu'une alerte se déclenche, puis comparer à l'ancien
  modèle **avant** de remplacer.

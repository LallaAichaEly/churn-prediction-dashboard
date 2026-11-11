# ✅ Corrigés du parcours

> **Mode d'emploi.** Cherchez d'abord seul·e (au moins 20 minutes par exercice). Si vous bloquez,
> lisez l'**indice**. Ouvrez la **solution** seulement après avoir essayé : c'est l'effort de recherche
> qui construit la compétence, pas la lecture du corrigé.
>
> Tous les chiffres ci-dessous ont été **mesurés** sur les données par défaut (5 000 clients,
> `seed=42`, découpage 75/25) avec scikit-learn 1.9. Avec d'autres versions, ils peuvent varier
> légèrement ; les conclusions restent les mêmes. Chaque extrait de code a été exécuté et testé.

**Corrigés exécutables.** Les scripts de `solutions/` refont les mesures des modules 1 à 5 et 7 et les
affichent ; `tests/test_solutions.py` vérifie que leurs conclusions restent vraies :

```bash
python solutions/module1_donnees.py      # ~1 s
python solutions/module2_variables.py    # ~2 s  (fuites de données, encodage par la cible)
python solutions/module3_modeliser.py    # ~20 s (validation croisée, GridSearchCV)
python solutions/module4_evaluer.py      # ~4 s  (seuils, courbe PR, calibration)
python solutions/module5_interpreter.py  # ~3 s  (effets retrouvés, explication d'un client)
python solutions/module7_surveiller.py   # ~4 s  (PSI, dérive des entrées et du concept)
```

Sommaire : [Module 1](#module-1--comprendre-les-données) · [Module 2](#module-2--préparer-les-variables) ·
[Module 3](#module-3--modéliser) · [Module 4](#module-4--évaluer-et-décider) ·
[Module 5](#module-5--interpréter-et-communiquer) · [Module 6](#module-6--industrialiser) ·
[Module 7](#module-7--surveiller-le-modèle-en-production) · [Projet final](#-projet-final)

---

## Module 1 — Comprendre les données

### 1.1 — Premier contact 🟢

💡 *Indice :* `pd.read_csv`, puis `.describe()` et `.mean()` sur une colonne 0/1.

<details><summary>Solution</summary>

```python
import pandas as pd

df = pd.read_csv("data/customers.csv")
print(df.describe().round(1))
print(f"Taux de churn : {df['churn'].mean():.1%}")
```

Résultat : **35,5 %** de churn. Ancienneté moyenne 21,3 mois (1 à 72), facture moyenne 61 €
(15 à 118 €), 1,3 appel au support en moyenne (0 à 7).

👉 La moyenne d'une colonne 0/1 est une proportion : c'est le réflexe le plus utile en analyse de churn.
</details>

### 1.2 — Taux de churn par segment 🟢

💡 *Indice :* `df.groupby("colonne")["churn"].mean()`.

<details><summary>Solution</summary>

```python
print(df.groupby("contract")["churn"].mean().sort_values())
print(df.groupby("internet_service")["churn"].mean().sort_values())
```

| Contrat | Churn | | Internet | Churn |
|---|---|---|---|---|
| Biannuel | 8,9 % | | Aucun | 25,0 % |
| Annuel | 21,5 % | | ADSL | 28,8 % |
| Mensuel | **50,9 %** | | Fibre | **45,4 %** |

Un client au mois part presque 6 fois plus qu'un client sous contrat de 2 ans : c'est **le** levier métier.
</details>

### 1.3 — Une nouvelle règle de validation 🟡

💡 *Indice :* `Series.between(a, b)` renvoie un booléen par ligne ; `.all()` les agrège.

<details><summary>Solution</summary>

Dans `validate()` (`src/churn/data.py`), avant le contrôle des identifiants :

```python
    if not df["monthly_charges"].between(0, 500).all():
        problems.append("Facture mensuelle hors de l'intervalle [0, 500] €")
```

Le test, dans `tests/test_churn.py` :

```python
def test_validate_detects_absurd_monthly_charges(df):
    broken = df.copy()
    broken.loc[0, "monthly_charges"] = 9999
    assert any("Facture" in p for p in validate(broken))
```

⚠️ `between` renvoie `False` pour une valeur manquante (`NaN`) : une facture vide déclenche donc
**aussi** cette règle. Ce n'est pas grave (le message « Valeurs manquantes » apparaît également),
mais c'est le genre de détail à connaître. Pour ne compter chaque problème qu'une fois, écrivez
`(s < 0) | (s > 500)` : toute comparaison avec `NaN` vaut `False`. C'est ce que vérifie le défi 12
(`defis/serie_02.py`), et ce que fait `validate_v2` dans `solutions/module1_donnees.py`.
</details>

### 1.4 — Formuler des hypothèses 🟡

💡 *Indice :* un coefficient positif dans `_churn_logit` augmente le risque ; regardez aussi **combien
de clients** sont concernés par chaque variable.

<details><summary>Solution (exemples d'hypothèses)</summary>

1. **Le contrat sera la variable n° 1** : gros coefficient (+1,6 pour Mensuel) *et* 55 % des clients concernés.
2. **Ancienneté et appels au support seront dans le trio de tête** : petits coefficients, mais les
   valeurs varient beaucoup (1 à 72 mois → jusqu'à −2,5 sur le logit).
3. **`senior` et `payment_method` paraîtront peu importants** malgré des coefficients réels
   (+0,3 et +0,7), car peu de clients sont concernés (18 % de seniors, 15 % de paiements par chèque).

Ces hypothèses se vérifient à l'exercice 5.1. L'intérêt : apprendre que l'importance d'une variable
dépend de **l'effet × la fréquence**, pas seulement de l'effet.
</details>

### 1.5 — Passer à de vraies données 🔴

💡 *Indice :* le fichier Kaggle s'appelle `WA_Fn-UseC_-Telco-Customer-Churn.csv`. Deux colonnes du
projet n'y existent pas : `support_calls` et `num_products`.

<details><summary>Solution (esquisse — non exécutée ici, le fichier Kaggle demande un compte)</summary>

```python
raw = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
services = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
df = pd.DataFrame({
    "customer_id": raw["customerID"],
    "tenure_months": raw["tenure"],
    "monthly_charges": raw["MonthlyCharges"],
    # Piège n° 1 : TotalCharges est du texte, avec " " pour les clients de 0 mois.
    "total_charges": pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0),
    "num_products": (raw[services] == "Yes").sum(axis=1),  # remplaçant raisonnable
    "contract": raw["Contract"].map({"Month-to-month": "Mensuel", "One year": "Annuel", "Two year": "Biannuel"}),
    "payment_method": raw["PaymentMethod"],  # ⇒ mettre à jour PAYMENT_METHODS dans data.py
    "internet_service": raw["InternetService"].map({"Fiber optic": "Fibre", "DSL": "ADSL", "No": "Aucun"}),
    "tech_support": raw["TechSupport"].map({"Yes": "Oui"}).fillna("Non"),
    "senior": raw["SeniorCitizen"].map({1: "Oui", 0: "Non"}),
    "churn": (raw["Churn"] == "Yes").astype(int),
})
```

Puis retirez `support_calls` de `NUMERIC_FEATURES` (et du formulaire de `app.py`), et lancez
`pytest` : les tests vous diront tout ce qui casse. Taux de churn attendu : environ **26,5 %**
(1 869 clients sur 7 043). Piège n° 2 : 11 clients ont 0 mois d'ancienneté ; c'est pour ça que
`add_engineered_features` utilise `.clip(lower=1)`.
</details>

### Auto-évaluation

<details><summary>Pourquoi connaître la « vraie » cause du churn quand on apprend ?</summary>

Parce qu'on peut **noter le modèle sur le fond**, pas seulement sur un score. Avec de vraies données,
on ne sait jamais si le modèle a trouvé les vraies causes ou des corrélations trompeuses. Ici, on peut
comparer : c'est un entraînement de l'intuition, comme un exercice de maths dont on connaît la réponse.
</details>

---

## Module 2 — Préparer les variables

### 2.1 — `StandardScaler` et `OneHotEncoder` 🟢

<details><summary>Solution</summary>

- **`StandardScaler`** transforme chaque variable numérique en `(x − moyenne) / écart-type`. Une
  ancienneté (0–72) et une facture totale (0–8 000) deviennent comparables. La régression logistique
  en a besoin : sa régularisation pénalise tous les coefficients de la même façon, et l'optimisation
  converge mieux.
- **`OneHotEncoder`** transforme une catégorie en colonnes 0/1 : `contract = "Mensuel"` devient
  `contract_Mensuel=1, contract_Annuel=0, contract_Biannuel=0`. Un modèle ne sait manipuler que des
  nombres, et coder `Mensuel=1, Annuel=2, Biannuel=3` inventerait un ordre et des distances qui n'existent pas.
</details>

### 2.2 — Les variables dérivées changent-elles l'AUC ? 🟢

<details><summary>Solution</summary>

| Modèle | Sans | Avec |
|---|---|---|
| Logistique | 0,8345 | 0,8345 |
| Boosting | 0,807 | 0,811 |

**Presque rien ne change.** Pourquoi ? `avg_charge_per_month` ≈ `monthly_charges` (le total est
généré comme facture × ancienneté), donc aucune information nouvelle. Et `calls_per_year` ne
correspond pas à la vraie cause : `_churn_logit` utilise le nombre **brut** d'appels.

👉 Leçon : une variable dérivée n'aide que si elle exprime quelque chose que le modèle ne peut pas
déjà voir. Avec de vraies données, c'est souvent là que se gagnent les meilleurs points.
</details>

### 2.3 — `is_new_customer` 🟡

💡 *Indice :* deux lignes à modifier dans `features.py` : la fonction et la liste `ENGINEERED_FEATURES`.

<details><summary>Solution</summary>

```python
    out["is_new_customer"] = (out["tenure_months"] < 6).astype(int)
...
ENGINEERED_FEATURES = ["avg_charge_per_month", "calls_per_year", "is_new_customer"]
```

Mesure (avec les variables dérivées activées) : logistique **0,8345 → 0,8347**, boosting
**0,811 → 0,811**. L'effet est nul, et c'est logique : dans la « vérité cachée », l'ancienneté agit
de façon **linéaire** (−0,035 par mois), sans seuil particulier à 6 mois.

👉 Un résultat négatif est un vrai résultat : notez-le dans votre `JOURNAL.md`.
</details>

### 2.4 — Pourquoi `handle_unknown="ignore"` ? 🟡

<details><summary>Solution</summary>

En production, une modalité inconnue finit toujours par arriver (nouveau moyen de paiement « Crypto »,
faute de frappe…). Par défaut, `OneHotEncoder` **lève une erreur** : l'application plante pour ce
client. Avec `"ignore"`, toutes les colonnes `payment_method_*` valent 0 : le modèle prédit avec les
autres variables. Le test `test_unknown_category_is_ignored` garantit ce comportement ; si quelqu'un
retire l'option, le test échoue.

Bonne pratique complémentaire : **refuser** les valeurs inconnues à l'entrée de l'API (exercice 6.4)
et surveiller leur fréquence. Ignorer en silence évite le plantage mais peut masquer un problème.
</details>

### 2.5 — Provoquer une fuite de données 🔴

💡 *Indice :* normalisez `df[NUMERIC_FEATURES]` **avant** `train_test_split`. Puis essayez une fuite
plus grave : sélectionner des variables avant la validation croisée.

<details><summary>Solution</summary>

**Expérience 1 — normaliser avant le split :**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from churn.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

X = df[FEATURES].copy()
X[NUMERIC_FEATURES] = StandardScaler().fit_transform(X[NUMERIC_FEATURES])   # ❌ voit le test
X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES)
X_tr, X_te, y_tr, y_te = train_test_split(X, df[TARGET], test_size=0.25, stratify=df[TARGET], random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
```

Résultat : **0,8345**, exactement comme sans fuite. Surprise ? Avec 5 000 lignes, la moyenne et
l'écart-type calculés avec ou sans le test sont quasi identiques.

**Expérience 2 — une fuite qui fait mal** (sélection de variables sur du **bruit pur**) :

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

small = df.sample(300, random_state=0)
noise = pd.DataFrame(np.random.default_rng(0).normal(size=(300, 2000)))  # 2 000 colonnes aléatoires
y = small[TARGET].to_numpy()

selected = SelectKBest(f_classif, k=20).fit_transform(noise, y)          # ❌ sélection sur tout
leaky = cross_val_score(LogisticRegression(max_iter=1000), selected, y, cv=5, scoring="roc_auc")
clean = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)),
                        noise, y, cv=5, scoring="roc_auc")                # ✅ dans le pipeline
print(leaky.mean(), clean.mean())
```

Résultat : **0,82 avec fuite** contre **0,53 sans fuite**, sur des données qui ne contiennent
**aucune** information. Le modèle « fuité » a l'air bon ; en production, il ferait du hasard.

👉 Pourquoi c'est dangereux : les métriques sont trop optimistes, on déploie un modèle qui déçoit,
et on ne comprend pas pourquoi. Règle d'or : **tout ce qui apprend des données** (scaler, imputation,
sélection, encodage de la cible…) **va dans le `Pipeline`**.
</details>

### Auto-évaluation

<details><summary>Qu'est-ce que la data leakage et comment un Pipeline la prévient-il ?</summary>

C'est quand une information indisponible au moment de la prédiction (le jeu de test, le futur, la
cible elle-même) se glisse dans l'entraînement. Un `Pipeline` appelle `fit` uniquement sur les données
d'entraînement de chaque découpage (y compris dans `cross_val_score` et `GridSearchCV`), puis
seulement `transform` sur les données de test.
</details>

---

## Module 3 — Modéliser

### 3.1 — Quel modèle gagne ? 🟢

<details><summary>Solution</summary>

La **régression logistique** : ROC-AUC 0,834 contre 0,807 pour le gradient boosting, et meilleure
sur toutes les métriques. Voir 3.3 pour comprendre pourquoi.
</details>

### 3.2 — Ajouter une forêt aléatoire 🟢

<details><summary>Solution</summary>

Dans `src/churn/model.py` :

```python
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

MODELS = {
    "logistic": "Régression logistique (référence interprétable)",
    "gboost": "Gradient boosting (plus puissant, moins lisible)",
    "rf": "Forêt aléatoire (beaucoup d'arbres indépendants)",
}
...
    elif model_name == "rf":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)
```

Ajoutez aussi `"rf"` dans `@pytest.mark.parametrize("model_name", [...])` de `test_models_beat_random`.
Le tableau de bord le propose automatiquement (il lit `MODELS`).

Résultat : ROC-AUC **0,811**, rappel 56 %. Toujours derrière la logistique.
</details>

### 3.3 — Pourquoi le modèle simple gagne-t-il ? 🟡

💡 *Indice :* relisez `generate_customers`. Comment la probabilité de churn est-elle calculée à partir du logit ?

<details><summary>Solution</summary>

Les données sont **générées par une régression logistique** : un logit **linéaire** + du bruit,
puis `1 / (1 + exp(−logit))`. La régression logistique est donc le modèle *exactement adapté*
(« bien spécifié ») : elle n'a que ~20 coefficients à estimer.

Le boosting, lui, approxime chaque effet linéaire par des **marches d'escalier** (des arbres) : il
lui faut beaucoup plus de données pour une précision égale, et il apprend une partie du bruit.

On peut le vérifier : les coefficients appris retrouvent la vérité (écart entre modalités) :

| Effet | Vérité | Appris |
|---|---|---|
| Chèque vs Prélèvement | +0,70 | +0,67 |
| Support technique Oui vs Non | −0,80 | −0,76 |
| Fibre vs sans internet | +0,50 | +0,46 |
| Senior Oui vs Non | +0,30 | +0,28 |

👉 Sur de vraies données (effets non linéaires, interactions), le boosting gagne souvent. C'est
pourquoi on teste **toujours** les deux.
</details>

### 3.4 — Validation croisée 🟡

<details><summary>Solution</summary>

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from churn.model import build_pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name in ["logistic", "gboost"]:
    scores = cross_val_score(build_pipeline(name), df[FEATURES], df[TARGET], cv=cv, scoring="roc_auc")
    print(f"{name}: {scores.round(3)}  moyenne {scores.mean():.3f} ± {scores.std():.3f}")
```

| | Plis | Moyenne ± écart-type |
|---|---|---|
| Logistique | 0,817 · 0,822 · 0,811 · 0,825 · 0,822 | **0,819 ± 0,005** |
| Boosting | 0,796 · 0,801 · 0,789 · 0,816 · 0,796 | 0,799 ± 0,009 |

L'écart (0,02) est **2 à 4 fois plus grand** que la variabilité, et la logistique gagne sur **chacun**
des 5 plis : la différence est solide.

Remarque importante : le découpage unique donnait 0,834, la validation croisée 0,819. Le découpage
unique était un peu « chanceux » : c'est pour ça qu'on ne se fie jamais à un seul découpage.
</details>

### 3.5 — `GridSearchCV` 🔴

<details><summary>Solution</summary>

```python
from sklearn.model_selection import GridSearchCV
from churn.model import split

X_train, X_test, y_train, y_test = split(df)
grid = GridSearchCV(
    build_pipeline("gboost"),
    {"clf__learning_rate": [0.02, 0.05, 0.1], "clf__max_depth": [2, 3, None], "clf__max_iter": [100, 300]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
).fit(X_train, y_train)                         # ⚠️ uniquement sur le train
print(grid.best_params_, grid.best_score_)
print(roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1]))
```

Meilleurs paramètres : `learning_rate=0.05, max_depth=2, max_iter=300` → AUC de test **0,825**
(contre 0,807 par défaut). Mieux, mais **toujours sous la logistique (0,834)**.

👉 Le réglage choisit des arbres de profondeur 2, les plus simples possibles : une confirmation de
plus que la structure est additive. Notez le préfixe `clf__` : c'est la syntaxe pour régler une étape
d'un `Pipeline`.
</details>

### 3.6 — `class_weight="balanced"` 🔴

<details><summary>Solution</summary>

```python
pipe = build_pipeline("logistic").set_params(clf__class_weight="balanced").fit(X_train, y_train)
print(evaluate(y_test, pipe.predict_proba(X_test)[:, 1]))
```

| Seuil 0,5 | ROC-AUC | Précision | Rappel |
|---|---|---|---|
| Par défaut | 0,8345 | 68,9 % | 63,4 % |
| `balanced` | 0,8348 | 60,2 % | **80,1 %** |

L'AUC ne bouge pas : le **classement** des clients est le même. `balanced` pousse surtout toutes les
probabilités vers le haut, ce qui revient presque à **baisser le seuil**. Le prix à payer : les
probabilités ne sont plus fiables (voir 4.5 : moyenne prédite 46,6 % pour un churn réel de 35,5 %).

👉 Pour privilégier le rappel, préférez souvent : modèle non pondéré + seuil choisi selon les coûts (4.3).
</details>

### Auto-évaluation

<details><summary>Pourquoi commence-t-on toujours par un modèle simple ?</summary>

Il donne une **référence** (tout modèle complexe doit faire mieux, sinon il ne vaut pas sa
complexité), il est rapide, il s'explique facilement au métier, et ses erreurs révèlent souvent des
problèmes de données. Ici, il est même le meilleur.
</details>

---

## Module 4 — Évaluer et décider

### 4.1 — Bouger le seuil 🟢

<details><summary>Solution</summary>

Seuil plus **bas** → on signale plus de clients → **rappel ↑, précision ↓** (on attrape plus de
partants, mais avec plus de fausses alertes). Seuil plus **haut** → l'inverse. La courbe F1 a un
maximum intermédiaire. Il n'existe pas de « bon » seuil universel : c'est une décision métier (4.3).
</details>

### 4.2 — Pourquoi l'accuracy est-elle trompeuse ? 🟢

<details><summary>Solution</summary>

Un modèle qui prédit « reste » pour **tout le monde** obtient **64,6 %** d'accuracy sur le jeu de test,
avec un rappel de **0 %** : il ne détecte aucun partant. Notre modèle (76,9 %) ne fait « que » 12 points
de mieux en accuracy, alors qu'il détecte 63 % des départs. L'accuracy mélange les deux types
d'erreurs et favorise la classe majoritaire.
</details>

### 4.3 — Le seuil optimal selon les coûts 🟡

💡 *Indice :* un client avec une probabilité *p* coûte en moyenne *p* × 500 € si on ne fait rien,
et (1 − *p*) × 50 € si on agit pour rien.

<details><summary>Solution</summary>

| Coûts (perdu / action inutile) | Seuil optimal | Précision | Rappel | Coût | Coût au seuil 0,5 |
|---|---|---|---|---|---|
| 500 € / 50 € | **0,10** | 43,9 % | 98,9 % | 30 450 € | 87 350 € |
| 100 € / 100 € | **0,50** | 68,9 % | 63,4 % | 28 900 € | 28 900 € |

La théorie le confirme : on agit si *p* × coût_perdu > (1 − *p*) × coût_action, soit
**seuil = coût_action / (coût_perdu + coût_action)** = 50 / 550 ≈ **0,09** et 100 / 200 = **0,50**.
Cela ne marche que si les probabilités sont fiables (4.5).

Avec 500 € / 50 €, rater un client coûte 10 fois plus cher qu'une action inutile : on contacte 997
clients sur 1 250 et on divise le coût par **près de 3**.

**Aller plus loin :** la fonction `best_threshold_by_cost` suppose que l'action est gratuite pour les
vrais positifs et qu'elle retient **100 %** des clients. Modifiez-la pour inclure le coût de l'action
pour tous les clients contactés et un taux de succès (par ex. 30 %) : le seuil remonte nettement.
</details>

### 4.4 — Courbe précision-rappel 🟡

⚡ *Échauffement :* le défi 15 (`defis/serie_03.py`) calcule ces points à la main, seuil par seuil.

<details><summary>Solution</summary>

Dans l'onglet Performance de `app.py`, juste avant `st.subheader("Le compromis précision / rappel")` :

```python
    from sklearn.metrics import average_precision_score, precision_recall_curve

    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="Modèle", line=dict(color=BLUE, width=2),
                             hovertemplate="Rappel %{x:.0%}<br>Précision %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[metrics.recall], y=[metrics.precision], mode="markers", name="Seuil actuel",
                             marker=dict(color=ORANGE, size=12)))
    fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="#898781", annotation_text="hasard")
    fig.update_layout(title=f"Courbe précision-rappel (AP = {average_precision_score(y_test, proba):.3f})",
                      xaxis_title="Rappel", yaxis_title="Précision", xaxis_tickformat=".0%",
                      yaxis_tickformat=".0%", margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")
```

Average precision : **0,709** (logistique) contre 0,661 (boosting). La ligne « hasard » est au niveau
du taux de churn (35 %), pas à 0,5 comme pour la courbe ROC. Le point orange suit le curseur de seuil.
</details>

### 4.5 — Calibration 🔴

⚡ *Échauffement :* le défi 16 construit la table de calibration à la main (tranches de même largeur,
comme `calibration_curve(..., strategy="uniform")` ; la solution ci-dessous utilise des groupes de même taille).

<details><summary>Solution</summary>

```python
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Modèle",
                             line=dict(color=BLUE, width=2),
                             hovertemplate="Prédit %{x:.0%}<br>Observé %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite",
                             line=dict(color="#898781", width=1, dash="dash"), hoverinfo="skip"))
    fig.update_layout(title=f"Calibration (Brier = {brier_score_loss(y_test, proba):.3f})",
                      xaxis_title="Probabilité prédite", yaxis_title="Taux de churn observé",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
```

Résultats (5 groupes de clients de même taille) :

| Prédit → observé | Logistique | Boosting | Logistique `balanced` |
|---|---|---|---|
| Groupe 2 | 17 % → 13 % | 13 % → 16 % | 27 % → **12 %** |
| Groupe 3 | 33 % → 34 % | 31 % → 38 % | 47 % → **34 %** |
| Groupe 4 | 52 % → 56 % | 53 % → 50 % | 67 % → **57 %** |
| Brier (plus bas = mieux) | **0,158** | 0,171 | 0,171 |

**Oui, pour la logistique, 0,7 veut à peu près dire 70 %** : ses points suivent la diagonale. La
version `balanced` surestime systématiquement le risque, de 6 à 15 points selon les groupes. Si vous avez besoin de
probabilités fiables (calcul de coûts, 4.3), vérifiez toujours la calibration, et corrigez-la au
besoin avec `CalibratedClassifierCV`.
</details>

### Auto-évaluation

<details><summary>Expliquer « rappel de 63 % » à un manager</summary>

« Sur 100 clients qui vont réellement partir, le modèle en repère 63 à l'avance. Les 37 autres nous
échappent. On peut en repérer davantage, mais au prix de plus de fausses alertes : c'est à vous de
choisir l'équilibre, selon ce que coûte une action de rétention. »
</details>

---

## Module 5 — Interpréter et communiquer

### 5.1 — Le modèle retrouve-t-il la vérité ? 🟢

<details><summary>Solution</summary>

Classement par permutation (logistique) : **contrat** (0,123) ≫ ancienneté (0,051) > appels au
support (0,042) > facture mensuelle (0,023) > support technique (0,020) > nombre de produits (0,011)
> internet (0,010) > paiement (0,006) > facture totale (0,002) > senior (0,000).

- **Retrouvé** : le trio contrat / ancienneté / appels, exactement comme prévu en 1.4.
- **Sous-estimés** : `senior` (effet réel +0,3, mais seulement 18 % de clients concernés) et
  `payment_method` (seule la modalité Chèque compte, 15 % des clients).
- **Bien vu** : `total_charges` est à ~0, et il **n'est pas** dans la vérité. Il est redondant avec
  ancienneté × facture mensuelle.

👉 L'importance par permutation mesure ce que la variable apporte **au score**, pas la force de sa
relation causale. Et deux variables corrélées se « partagent » l'importance.
</details>

### 5.2 — Churn par service internet 🟢

<details><summary>Solution</summary>

À la fin de l'onglet « Vue d'ensemble », avant l'`expander` des données brutes :

```python
    by_internet = df.groupby("internet_service")[TARGET].mean().reindex(INTERNET_SERVICES).reset_index()
    st.plotly_chart(bar(by_internet, "internet_service", TARGET, "Taux de churn par service internet"),
                    width="stretch")
```

La fonction `bar` existe déjà : réutiliser plutôt que copier-coller, c'est aussi une compétence.
</details>

### 5.3 — Les 3 variables qui pèsent le plus 🟡

⚡ *Échauffement :* le défi 18 vous fait écrire cette fonction et vérifie que intercept + Σ contributions
redonne exactement le logit du modèle.

💡 *Indice :* `pipeline[:-1]` est le pipeline **sans** le classifieur : il donne les variables
transformées. Les coefficients sont dans `pipeline.named_steps["clf"].coef_`.

<details><summary>Solution</summary>

Dans l'onglet « Prédire un client », après `st.progress(p)` :

```python
        if model_name == "logistic":
            z = pipeline[:-1].transform(client)          # toutes les étapes sauf le classifieur
            z = z.toarray() if hasattr(z, "toarray") else z
            names = pipeline.named_steps["prep"].get_feature_names_out()
            contrib = pd.Series(z[0] * pipeline.named_steps["clf"].coef_[0], index=names)
            top = contrib[contrib != 0].sort_values(key=abs, ascending=False).head(3)
            st.write("**Ce qui pèse le plus dans cette prédiction :**")
            for name, value in top.items():
                sens = "⬆️ augmente" if value > 0 else "⬇️ diminue"
                st.write(f"- `{name}` {sens} le risque ({value:+.2f} en logit)")
```

Fonctionne avec et sans variables dérivées (grâce à `pipeline[:-1]`).

⚠️ **Limite à connaître** : pour le profil par défaut, l'application affiche « `senior_Non` diminue le
risque ». C'est vrai mathématiquement, mais trompeur : ce calcul compare chaque modalité à « zéro »,
pas à un client moyen. SHAP (5.5) corrige ce défaut.
</details>

### 5.4 — Filtre par contrat 🟡

<details><summary>Solution</summary>

Dans la barre latérale :

```python
selected_contracts = st.sidebar.multiselect("Contrats (vue d'ensemble)", CONTRACTS, default=CONTRACTS)
```

Au début de l'onglet « Vue d'ensemble », créez une **nouvelle** variable, puis remplacez `df` par
`view` dans tout l'onglet :

```python
with tab_overview:
    view = df[df["contract"].isin(selected_contracts or CONTRACTS)]  # liste vide = tout afficher
    churned = view[view[TARGET] == 1]
    ...
```

⚠️ **Le piège** : écrire `df = df[...]` semble marcher dans la vue d'ensemble, mais **casse l'onglet
« Clients à risque »** (`KeyError` : `df.loc[X_test.index]` cherche des clients qui ont été filtrés).
Un script Streamlit s'exécute de haut en bas : réassigner une variable globale a des effets plus loin.
Le test `tests/test_app.py` attrape exactement ce genre d'erreur.
</details>

### 5.5 — SHAP 🔴

<details><summary>Solution</summary>

`pip install shap`, puis près des autres fonctions mises en cache :

```python
def to_dense(a):
    return a.toarray() if hasattr(a, "toarray") else a


@st.cache_resource
def get_explainer(model_name: str, use_engineered: bool):
    import shap

    pipeline, X_test, *_ = get_model(model_name, use_engineered)
    background = to_dense(pipeline[:-1].transform(X_test.sample(200, random_state=0)))
    names = pipeline.named_steps["prep"].get_feature_names_out()
    return shap.Explainer(pipeline[-1], background, feature_names=names)
```

Et dans l'onglet « Prédire un client », après la prédiction :

```python
        explanation = get_explainer(model_name, use_engineered)(to_dense(pipeline[:-1].transform(client)))
        sv = pd.Series(explanation.values[0], index=explanation.feature_names)
        sv = sv[sv.abs().sort_values().index].tail(6)  # les 6 plus fortes contributions
        fig = px.bar(x=sv.values, y=sv.index, orientation="h", color=sv.values > 0,
                     color_discrete_map={True: ORANGE, False: BLUE},
                     title="Explication SHAP : écart à un client moyen (en logit)")
        fig.update_layout(showlegend=False, xaxis_title="Contribution", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
```

`shap.Explainer` choisit seul la bonne méthode : `LinearExplainer` pour la logistique,
`TreeExplainer` pour le boosting. Les contributions sont mesurées **par rapport à un client moyen** :
`contract_Mensuel` vaut +0,40 avec SHAP contre +1,01 avec la méthode naïve de 5.3, car 56 % des
clients sont déjà au mois : être au mois est « normal », donc moins distinctif.
</details>

### Auto-évaluation

<details><summary>3 actions de rétention, chiffres à l'appui</summary>

1. **Convertir les contrats mensuels** (50,9 % de churn contre 21,5 % en annuel) : remise pour un
   engagement d'un an. C'est la variable la plus importante du modèle.
2. **Accompagner les 6 premiers mois** : 50,7 % de churn, contre 6,8 % au-delà de 4 ans (graphique par ancienneté).
3. **Alerte après 3 appels au support** : chaque appel augmente le risque (+0,45 sur le logit) ;
   proposer le support technique, qui le diminue (−0,8).

Et pour chacune : estimer le coût, le nombre de clients concernés et un taux de succès, puis
**mesurer** l'effet avec un groupe témoin (test A/B).
</details>

---

## Module 6 — Industrialiser

### 6.1 — Que protège chaque test ? 🟢

<details><summary>Solution</summary>

| Test | Ce qu'il protège |
|---|---|
| `test_generation_is_reproducible` | Même `seed` → mêmes données (sinon les résultats ne sont pas comparables) |
| `test_generated_data_is_valid` | Le générateur produit des données valides avec un taux de churn réaliste |
| `test_validate_detects_problems` | `validate()` repère vraiment ancienneté négative et valeurs manquantes |
| `test_engineered_features_do_not_mutate_input` | La fonction ne modifie pas le DataFrame d'origine (bug très fréquent) |
| `test_models_beat_random` | Chaque modèle × chaque option apprend quelque chose (AUC > 0,7) |
| `test_model_learns_true_signal` | Le modèle a le bon **sens** : mensuel plus risqué que biannuel |
| `test_evaluate_confusion_matrix_sums` | Les métriques sont justes sur un cas calculé à la main |
| `test_costly_misses_lower_the_threshold` | La logique métier du seuil va dans le bon sens |
| `test_unknown_model_raises` | Une faute de frappe donne une erreur claire, pas un comportement silencieux |
| `test_unknown_category_is_ignored` | Une nouvelle modalité ne fait pas planter la prédiction |
| `test_saved_model_gives_identical_predictions` | Le modèle rechargé prédit **exactement** comme celui qu'on a évalué |
| `tests/test_api.py` | Le contrat de l'API : codes HTTP, validation des entrées (422), seuil, entraînement automatique |
| `tests/test_app.py` | Chaque onglet du tableau de bord s'affiche sans erreur, quels que soient les réglages |
| `tests/test_solutions.py` | Les affirmations des corrigés (« la fuite gonfle l'AUC »…) restent vraies |
| `tests/test_monitoring.py` | Le PSI se tait sans dérive, alerte sur la bonne variable, et ne voit pas la dérive du concept (module 7) |
</details>

### 6.2 — Tracer la version de scikit-learn 🟡

💡 *Indice :* `sklearn.__version__` ; dans `load()`, lisez le fichier `*_metrics.json` voisin
(`path.with_name(...)`) avant `joblib.load`, et utilisez `warnings.warn`.

<details><summary>Solution</summary>

Dans `src/churn/model.py` (ajoutez `import sklearn` et `import warnings` en haut) :

```python
def save(pipeline, metrics, model_dir="models", name="churn_model") -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{name}.joblib"
    joblib.dump(pipeline, path)
    info = {**asdict(metrics), "sklearn_version": sklearn.__version__}
    (model_dir / f"{name}_metrics.json").write_text(json.dumps(info, indent=2))
    return path


def load(path="models/churn_model.joblib") -> Pipeline:
    path = Path(path)
    info_path = path.with_name(f"{path.stem}_metrics.json")
    if info_path.exists():
        trained_with = json.loads(info_path.read_text()).get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(
                f"Modèle entraîné avec scikit-learn {trained_with}, version installée "
                f"{sklearn.__version__} : réentraînez-le (python scripts/train.py).",
                UserWarning,
                stacklevel=2,
            )
    return joblib.load(path)
```

Et le test (`tests/test_churn.py`, avec `import sklearn` et `import warnings`) :

```python
def test_load_warns_on_sklearn_version_mismatch(df, tmp_path):
    pipeline, *_, metrics = train(df)
    path = save(pipeline, metrics, model_dir=tmp_path)
    info_path = tmp_path / "churn_model_metrics.json"
    info = json.loads(info_path.read_text())
    assert info["sklearn_version"] == sklearn.__version__

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # même version : aucun avertissement toléré
        load(path)

    info_path.write_text(json.dumps({**info, "sklearn_version": "0.0.1"}))
    with pytest.warns(UserWarning, match="scikit-learn 0.0.1"):
        load(path)
```

👉 **Pourquoi c'est un vrai risque.** Un fichier `joblib` est un objet Python « congelé » : il
dépend du code interne de scikit-learn. Après une mise à jour, le chargement peut échouer, ou
pire, **réussir** avec des prédictions légèrement différentes de celles que vous avez évaluées.
scikit-learn émet déjà un `InconsistentVersionWarning`, mais l'écrire dans le fichier de
métriques rend l'information lisible par un humain et par l'API (`GET /health` l'affiche alors
automatiquement). En production, on fige aussi les versions (`pip freeze > requirements.lock`).
</details>

### 6.3 — Étendre l'intégration continue 🟡

<details><summary>Solution</summary>

Dans `.github/workflows/tests.yml` :

```yaml
jobs:
  tests:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Installer les dépendances
        run: pip install -r requirements.txt ruff
      - name: Lint
        run: ruff check --select E4,E7,E9,F .
      - name: Lancer les tests
        run: pytest -q
      - name: Entraîner les modèles
        run: python scripts/train.py
```

Le code actuel passe `ruff check --select E4,E7,E9,F .` sans erreur. `fail-fast: false` laisse
tourner les autres versions de Python quand l'une échoue : on voit d'un coup *où* ça casse.
Pour voir la CI échouer, cassez volontairement un test sur une branche, poussez, et observez la
croix rouge sur GitHub (et sur le badge du README).
</details>

### 6.4 — Prédire pour une liste de clients 🔴

💡 *Indice :* le corps de la requête est une `list[Customer]`. Construisez **un seul** DataFrame
et appelez `predict_proba` **une seule fois**. `Body(min_length=1, max_length=1000)` borne la taille.

<details><summary>Solution</summary>

Dans `api.py` (importez `Annotated` depuis `typing` et `Body` depuis `fastapi`) :

```python
@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(
    customers: Annotated[list[Customer], Body(min_length=1, max_length=1000)],
    request: Request,
    threshold: float = Query(0.5, ge=0, le=1, description="Seuil de décision « à risque »"),
) -> list[Prediction]:
    """Prédit pour plusieurs clients en UN SEUL appel au modèle (vectorisé)."""
    rows = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    probas = request.app.state.pipeline.predict_proba(rows)[:, 1]
    return [
        Prediction(churn_probability=round(float(p), 4), at_risk=bool(p >= threshold), threshold=threshold)
        for p in probas
    ]
```

Et ses tests (`tests/test_api.py`) :

```python
def test_batch_matches_single_predictions(client):
    r = client.post("/predict/batch", json=[RISKY, LOYAL])
    assert r.status_code == 200
    batch = [p["churn_probability"] for p in r.json()]
    single = [client.post("/predict", json=c).json()["churn_probability"] for c in (RISKY, LOYAL)]
    assert batch == single


def test_batch_rejects_empty_list_and_any_bad_row(client):
    assert client.post("/predict/batch", json=[]).status_code == 422
    assert client.post("/predict/batch", json=[RISKY, {**LOYAL, "contract": "Hebdo"}]).status_code == 422
```

👉 Trois réflexes de production : **vectoriser** (1 appel pour 1 000 clients est des dizaines de
fois plus rapide que 1 000 appels), **borner** la taille des requêtes (une liste d'un million de
clients bloquerait le serveur), et **tout refuser** si une seule ligne est invalide (une réponse
partielle silencieuse est plus dangereuse qu'une erreur claire).
</details>

### 6.5 — Déployer sur Streamlit Community Cloud 🔴

<details><summary>Solution</summary>

1. Poussez le dépôt sur GitHub (public, ou privé avec accès autorisé).
2. Sur [share.streamlit.io](https://share.streamlit.io), connectez-vous avec GitHub → **Create app**.
3. Choisissez le dépôt, la branche `main` et le fichier `app.py`. Dans les réglages avancés,
   choisissez Python 3.11 ou plus.
4. Déployez. Les dépendances viennent de `requirements.txt`, et les données sont **générées au
   premier lancement** : il n'y a rien d'autre à configurer.
5. Ajoutez le lien en haut du README, avec une capture d'écran.

Si l'application plante au démarrage, lisez les journaux (*Manage app*) : c'est presque toujours une
dépendance manquante dans `requirements.txt`.
</details>

### Auto-évaluation

<details><summary>Détecter un data drift dans 6 mois</summary>

- **Surveiller les entrées** : comparer chaque mois la distribution des variables à celle de
  l'entraînement (moyennes, part de chaque modalité, test de Kolmogorov-Smirnov, PSI) et la part de
  modalités inconnues.
- **Surveiller les sorties** : la distribution des probabilités prédites et la part de clients signalés.
- **Surveiller la performance réelle** dès que les vrais départs sont connus (AUC, rappel, calibration).
- Prévoir un **ré-entraînement** régulier, validé par les mêmes tests avant mise en ligne.
</details>

---

## Module 7 — Surveiller le modèle en production

Chiffres : `python solutions/module7_surveiller.py`. Référence = données d'entraînement
(3 750 clients) ; nouveau lot = `simulate_batch()` (2 000 clients, `seed=2026`).

### 7.1 — Hausse des prix de 20 % 🟢

💡 *Indice :* le PSI compare des **tranches** (déciles) calculées sur la référence. Combien de
clients changent de tranche quand on multiplie chaque valeur par 1,2 ?

<details><summary>Solution</summary>

| Variable | PSI | Statut |
|---|---|---|
| `monthly_charges` | **0,258** | 🚨 dérive |
| `total_charges` | 0,038 | ✅ stable |
| toutes les autres | < 0,01 | ✅ stable |

Les deux variables augmentent de 20 %, mais le PSI mesure un déplacement **par rapport à la
dispersion** :

- `monthly_charges` : déciles serrés (26, 39, 49, 56, 63, 70, 76, 83, 90 €) ; +20 % ≈ +12 €,
  soit environ **un décile entier** → beaucoup de clients changent de tranche.
- `total_charges` : déciles de plus en plus larges (93, 215, 361, 537, 755, 1 062, 1 476, 2 172,
  3 399 €), chacun ~1,5 fois le précédent. Multiplier par 1,2 ne fait franchir une frontière qu'à
  une minorité de clients.

👉 Un PSI faible ne veut pas dire « rien n'a changé » : il veut dire « peu de clients ont changé
de tranche ». Surveillez aussi les moyennes des variables métier importantes (prix !).
</details>

### 7.2 — PSI à la main 🟢

⚡ *Échauffement :* le défi 19 (`psi_parts`) code cette formule, tranche vide comprise.

💡 *Indice :* PSI = Σ (actuel − référence) × ln(actuel / référence), une fois par modalité.

<details><summary>Solution</summary>

(0,7 − 0,5) × ln(0,7 / 0,5) + (0,3 − 0,5) × ln(0,3 / 0,5)
= 0,2 × 0,336 + (−0,2) × (−0,511) = 0,067 + 0,102 = **0,1695** → ⚠️ à surveiller.

```python
from churn.monitoring import psi
psi(["a"] * 50 + ["b"] * 50, ["a"] * 70 + ["b"] * 30)  # 0.1695
```

Chaque terme est **toujours positif** (si actuel > référence, le log est positif ; sinon les deux
facteurs sont négatifs) : les écarts ne se compensent jamais. Le PSI est une version symétrique
de la divergence de Kullback-Leibler.
</details>

### 7.3 — Le bruit du PSI 🟡

💡 *Indice :* bouclez sur 20 graines, gardez à chaque fois le PSI **maximal** sur les 10 variables
(`drift_report(...)["psi"].max()`), car c'est celui qui déclencherait une alerte.

<details><summary>Solution</summary>

```python
from churn.data import generate_customers
from churn.model import split
from churn.monitoring import drift_report

reference = split(df)[0]
for n in (200, 1000, 5000):
    worst = [drift_report(reference, generate_customers(n, seed=1000 + r))["psi"].max() for r in range(20)]
    print(n, round(sum(worst) / len(worst), 3), round(max(worst), 3))
```

| Taille du lot | PSI max moyen | PSI max, pire des 20 tirages |
|---|---|---|
| 200 | 0,079 | **0,111** ⚠️ fausse alerte |
| 1 000 | 0,019 | 0,035 |
| 5 000 | 0,006 | 0,009 |

Même sans aucune dérive, le PSI n'est jamais nul : c'est le **bruit d'échantillonnage**, qui
décroît à peu près comme 1 / n. Avec 200 clients, on déclenche des alertes « à surveiller » sur du
hasard. Règle pratique : surveillez des lots d'au moins ~1 000 clients (quitte à cumuler plusieurs
semaines), et c'est pour cela que les seuils sont 0,10 / 0,25 et pas 0.
</details>

### 7.4 — Dérive ≠ dégradation 🟡

💡 *Indice :* l'AUC mesure la capacité à **séparer** les clients. Si presque tout le monde a le
même contrat, la variable la plus discriminante sépare moins… même pour un modèle parfait.

<details><summary>Solution</summary>

| Scénario | PSI max | ROC-AUC | churn prédit | churn réel |
|---|---|---|---|---|
| aucun changement | 0,018 | 0,827 | 34,1 % | 34,7 % |
| 75 % de mensuels | 0,159 (`contract`) | **0,792** | 41,6 % | 42,4 % |

| Sur le lot « 75 % de mensuels » | ROC-AUC | churn prédit | churn réel |
|---|---|---|---|
| ancien modèle | 0,792 | 41,6 % | 42,4 % |
| réentraîné sur 5 000 clients récents | 0,794 | 41,7 % | 42,4 % |

L'AUC baisse de 0,035… et pourtant le modèle **ne se trompe pas** : il prédit 41,6 % de départs
pour 42,4 % observés, et le réentraîner ne gagne rien. La baisse vient de la **population** (plus
homogène, donc plus difficile à trier), pas du modèle. Le lien entre variables et churn n'a pas
changé, et le modèle l'a bien appris.

👉 Une AUC qui baisse n'est pas toujours un modèle cassé. Avant de réentraîner, regardez
**l'écart prédit / réel** : s'il reste faible, le modèle est juste, c'est le monde qui a changé.
Côté métier, l'information est ailleurs : environ +7,5 points de churn attendus, à anticiper.
</details>

### 7.5 — Dérive du concept 🔴

💡 *Indice :* le lot « concurrent fibre » contient **exactement les mêmes clients** que le lot
« aucun changement » : seule leur décision de partir a changé.

<details><summary>Solution</summary>

| Scénario | PSI max des entrées | PSI des scores | ROC-AUC | churn prédit | churn réel |
|---|---|---|---|---|---|
| aucun changement | 0,018 | 0,018 | 0,827 | 34,1 % | 34,7 % |
| concurrent fibre | 0,018 | 0,018 | **0,839** | 34,1 % | **44,6 %** |

Le piège parfait : **aucune** alerte sur les entrées (mêmes clients), aucune sur les scores (même
modèle, mêmes clients → mêmes probabilités), et l'AUC **monte** (les clients fibre, déjà un peu
plus risqués, partent maintenant beaucoup plus : le classement reste bon). Seul indicateur qui
sonne : **10,5 points** d'écart entre churn prédit et réel. Le modèle sous-estime le risque de
tous les clients fibre et le seuil choisi au module 4 n'est plus optimal.

```python
from solutions.module7_surveiller import retrain_on_recent
retrain_on_recent(df, fibre_competition=1.5)
```

| Sur le lot « concurrent fibre » | ROC-AUC | churn prédit | churn réel |
|---|---|---|---|
| ancien modèle | 0,839 | 34,1 % | 44,6 % |
| réentraîné sur 5 000 clients récents | **0,862** | **43,5 %** | 44,6 % |

Ici, réentraîner change tout : le modèle réapprend l'effet de la fibre. Mais il fallait attendre
d'avoir les **vrais départs** : la dérive du concept ne se voit qu'avec les étiquettes.
</details>

### 7.6 — Un script de surveillance 🔴

💡 *Indice :* `sys.exit(code)` ; un code ≠ 0 fait échouer une étape de CI ou déclenche l'alerte d'un cron.

<details><summary>Solution</summary>

`scripts/monitor.py` (testé : code 0 sur un lot stable, 1 avec une hausse des prix de 20 %) :

```python
"""Compare un nouveau lot de clients aux données d'entraînement et alerte en cas de dérive.

Usage :
    python scripts/monitor.py nouveau_lot.csv
Code de sortie : 0 = stable, 1 = au moins une variable en dérive (PSI > 0,25).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from churn.data import FEATURES, load_or_create  # noqa: E402
from churn.model import split  # noqa: E402
from churn.monitoring import drift_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", type=Path, help="CSV du nouveau lot (mêmes colonnes que data/customers.csv)")
    parser.add_argument("--reference", type=Path, default=ROOT / "data" / "customers.csv")
    args = parser.parse_args(argv)

    batch = pd.read_csv(args.batch)
    missing = [c for c in FEATURES if c not in batch.columns]
    if missing:
        print(f"Colonnes manquantes dans le lot : {missing}", file=sys.stderr)
        return 2

    reference = split(load_or_create(args.reference))[0]  # les données vues à l'entraînement
    report = drift_report(reference, batch)
    print(report.round(3).to_string(index=False))

    drifted = report.loc[report["statut"] == "dérive", "feature"].tolist()
    if drifted:
        print(f"\n🚨 Dérive détectée sur : {', '.join(drifted)}")
        return 1
    print("\n✅ Aucune dérive significative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Pour l'essayer :

```bash
python -c "import sys; sys.path.insert(0, 'src'); from churn.monitoring import simulate_batch; simulate_batch(price_increase=0.2).to_csv('lot.csv', index=False)"
python scripts/monitor.py lot.csv; echo "code de sortie : $?"   # → 1
```

Trois codes distincts (0 stable, 1 dérive, 2 fichier invalide) : celui qui reçoit l'alerte sait
tout de suite s'il doit regarder les données ou le pipeline de collecte.
</details>

### Auto-évaluation

<details><summary>AUC stable, entrées stables, mais 10 points de churn en plus que prévu</summary>

C'est une **dérive du concept** (exercice 7.5) : les clients sont les mêmes, leur comportement a
changé (concurrent, nouvelle offre, crise…). Le PSI ne peut pas la voir, et l'AUC peut même rester
bonne, car elle ne mesure que le classement. Que faire :

1. **Chercher la cause** avec le métier : le taux réel par segment (contrat, internet…) montre
   où l'écart se concentre (ici : les clients fibre).
2. **Réentraîner** sur des données récentes, puis vérifier l'écart prédit / réel et l'AUC.
3. **Recalculer le seuil** (module 4) : les probabilités ont changé, le seuil optimal aussi.
4. Mettre en place un **plan de surveillance** à trois étages : entrées (PSI, chaque semaine),
   scores (PSI des probabilités, part de clients signalés), performance (écart prédit / réel et
   AUC, dès que les vrais départs sont connus).
</details>

---

## 🏁 Projet final

Pas de corrigé « tout fait » : le rapport doit être **le vôtre**. Voici une structure et les chiffres
de référence à vérifier par vous-même.

<details><summary>Structure conseillée (1 page)</summary>

1. **Problème** — 35,5 % des clients partent ; quel coût mensuel ? (onglet Vue d'ensemble)
2. **Données** — 5 000 clients, 10 variables, contrôles qualité réalisés, limites (données synthétiques).
3. **Modèle retenu** — régression logistique : AUC 0,819 ± 0,005 en validation croisée, meilleure que
   le boosting même optimisé, interprétable et bien calibrée.
4. **Seuil choisi** — avec vos hypothèses de coûts (ex. 500 € / 50 € → seuil 0,10) et pourquoi.
5. **Impact estimé** — coût total au seuil choisi contre seuil 0,5 contre « ne rien faire ».
6. **Recommandations** — 3 actions de rétention (voir l'auto-évaluation du module 5).
7. **Limites et suites** — taux de succès des actions inconnu (prévoir un test A/B), plan de
   surveillance (module 7 : PSI des entrées, écart churn prédit / réel), validation sur de vraies données (1.5).

Un bon rapport se lit en 3 minutes par un manager **et** en 10 minutes par un data scientist.
</details>

<details><summary>Exemple rédigé (à lire après avoir écrit le vôtre)</summary>

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
</details>

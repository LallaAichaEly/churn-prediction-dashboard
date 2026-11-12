# 💡 Indices — Parcours de compétences

Bloqué sur un exercice de [`PARCOURS.md`](PARCOURS.md) ? Ouvrez les indices **un par un** :

- **Indice 1** : une direction, pour réfléchir par vous-même.
- **Indice 2** : un coup de pouce concret (API, squelette de code, piège à éviter).

Il n'y a volontairement **pas de corrigé complet** : c'est en écrivant le code soi-même
qu'on progresse. Cherchez au moins 15 minutes avant d'ouvrir l'indice 2.

Pour voir où vous en êtes : `python scripts/progression.py`.

---

## Module 1 — Comprendre les données

### 1.1
<details><summary>Indice 1</summary>

Le fichier `data/customers.csv` est créé par `python scripts/train.py` (ou au premier lancement
du tableau de bord). Ensuite : `df = pd.read_csv("data/customers.csv")`.
</details>
<details><summary>Indice 2</summary>

`churn` vaut 0 ou 1 : sa **moyenne** est donc directement le taux de churn.
Essayez aussi `df.info()` et `df["contract"].value_counts(normalize=True)`.
</details>

### 1.2
<details><summary>Indice 1</summary>

Le motif à retenir : `df.groupby(<colonne>)[<cible>].mean()`.
</details>
<details><summary>Indice 2</summary>

Ajoutez `.sort_values(ascending=False)` pour classer les modalités. Variante :
`pd.crosstab(df["contract"], df["churn"], normalize="index")`.
</details>

### 1.3
<details><summary>Indice 1</summary>

Inspirez-vous de la règle sur l'ancienneté négative dans `validate()`. La méthode
`Series.between(a, b)` est faite pour ça.
</details>
<details><summary>Indice 2</summary>

```python
if (~df["monthly_charges"].between(___, ___)).any():
    problems.append("Facture mensuelle hors de [0, 500] €")
```

Pour le test : copiez la fixture `df`, mettez `900` dans une cellule de `monthly_charges`
et vérifiez que `validate()` renvoie un problème. Testez aussi une valeur négative.
</details>

### 1.4
<details><summary>Indice 1</summary>

Un coefficient positif augmente le churn, un négatif le réduit. Mais attention aux **unités** :
`-0.035` par mois d'ancienneté paraît petit… sur 72 mois, cela fait `-2.5` !
</details>
<details><summary>Indice 2</summary>

Formulez des hypothèses **vérifiables**, par exemple : « la variable X sera dans le top 3 de
l'onglet *Facteurs de churn* » ou « le taux de churn des contrats mensuels dépassera 50 % ».
</details>

### 1.5
<details><summary>Indice 1</summary>

Colonnes du jeu Telco : `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`,
`InternetService`, `TechSupport`, `SeniorCitizen`, `Churn` (`Yes`/`No`). Utilisez
`df.rename(columns={...})` puis `.map({"Yes": 1, "No": 0})` pour la cible.
</details>
<details><summary>Indice 2</summary>

Pièges classiques :
- `TotalCharges` est du **texte** avec des espaces pour les nouveaux clients →
  `pd.to_numeric(..., errors="coerce")` puis traitez les valeurs manquantes.
- `support_calls` et `num_products` n'existent pas : retirez-les de `NUMERIC_FEATURES`, ou
  reconstruisez `num_products` en comptant les services souscrits (`OnlineSecurity`, `StreamingTV`…).
- Les modalités sont en anglais (`Month-to-month`, `Two year`…) : mettez à jour `CONTRACTS`,
  `PAYMENT_METHODS`, `INTERNET_SERVICES` et le test `test_model_learns_true_signal`.
- `SeniorCitizen` vaut 0/1, pas `Oui`/`Non`.
</details>

---

## Module 2 — Préparer les variables

### 2.1
<details><summary>Indice 1</summary>

`StandardScaler` : `(x - moyenne) / écart-type`, moyenne et écart-type **appris sur le train**.
`OneHotEncoder` : une colonne 0/1 par modalité.
</details>
<details><summary>Indice 2</summary>

Regardez le résultat concret après entraînement :

```python
prep = pipeline.named_steps["prep"]
prep.get_feature_names_out()
prep.transform(X_test.head(3))
```

Question bonus : pourquoi la normalisation compte-t-elle pour la régression logistique
(régularisation, convergence) mais pas pour les arbres ?
</details>

### 2.2
<details><summary>Indice 1</summary>

En ligne de commande : comparez `python scripts/train.py` et `python scripts/train.py --engineered`.
</details>
<details><summary>Indice 2</summary>

Un écart de 0,001 est-il un vrai gain ? Relancez avec `--seed 1`, `--seed 2`… pour mesurer la
variation **naturelle**. Puis regardez comment `total_charges` est généré dans `data.py` :
`avg_charge_per_month` apporte-t-elle une information nouvelle ?
</details>

### 2.3
<details><summary>Indice 1</summary>

Deux endroits à modifier dans `features.py` : la fonction `add_engineered_features` **et** la
liste `ENGINEERED_FEATURES` (sinon le `ColumnTransformer` ignore la nouvelle colonne).
</details>
<details><summary>Indice 2</summary>

Une comparaison donne des booléens : `(out["tenure_months"] < 6).astype(int)`.
Mesurez l'effet avec `--engineered` avant / après et notez-le dans `JOURNAL.md`.
</details>

### 2.4
<details><summary>Indice 1</summary>

Que se passe-t-il en production si un client paie avec un moyen jamais vu à l'entraînement
(ex. « Crypto ») ?
</details>
<details><summary>Indice 2</summary>

Retirez temporairement `handle_unknown="ignore"` dans `features.py` et lancez
`pytest -q -k unknown`. Avec `"ignore"`, toutes les colonnes one-hot de cette variable valent 0 :
le modèle se base uniquement sur les autres variables.
</details>

### 2.5
<details><summary>Indice 1</summary>

Appliquez `StandardScaler().fit_transform(...)` sur **tout** le jeu de données avant
`train_test_split`, puis entraînez un modèle sans scaler dans le pipeline.
</details>
<details><summary>Indice 2</summary>

Avec un simple scaler sur 5 000 lignes, l'écart sera minuscule (moyenne et écart-type bougent
peu). Pour voir une fuite **spectaculaire**, ajoutez une « variable du futur » :
```python
rng = np.random.default_rng(0)
df["a_appele_pour_resilier"] = df["churn"] * (rng.uniform(size=len(df)) < 0.8)
```

N'oubliez pas de l'ajouter à `NUMERIC_FEATURES` le temps de l'expérience.
Que devient l'AUC ? Pourquoi ce modèle échouerait-il en production ?
</details>

---

## Module 3 — Modéliser

### 3.1
<details><summary>Indice 1</summary>

`train.py` choisit le meilleur modèle selon la **ROC-AUC** : regardez la première colonne du tableau.
</details>
<details><summary>Indice 2</summary>

L'écart est-il stable ? Relancez avec plusieurs `--seed` avant de conclure.
</details>

### 3.2
<details><summary>Indice 1</summary>

Deux endroits dans `model.py` : le dictionnaire `MODELS` (nom + description) et un nouveau
`elif` dans `build_pipeline`. Le tableau de bord le proposera automatiquement.
</details>
<details><summary>Indice 2</summary>

```python
from sklearn.ensemble import RandomForestClassifier
RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=seed, n_jobs=-1)
```

`min_samples_leaf` évite des feuilles trop petites (et des probabilités trop extrêmes).
Ajoutez le nouveau nom au `parametrize` de `test_models_beat_random`.
</details>

### 3.3
<details><summary>Indice 1</summary>

Relisez `_churn_logit` : quelle est la **forme** de la formule ? Comparez-la à l'équation d'une
régression logistique.
</details>
<details><summary>Indice 2</summary>

Quand la vraie relation est linéaire dans le logit, la logistique est le « bon » modèle. Le
boosting doit approcher des droites par des escaliers et paie un coût en variance avec ~3 750
lignes d'entraînement. Expérience : générez 50 000 clients (`generate_customers(n=50_000)`) —
l'écart se réduit-il ?
</details>

### 3.4
<details><summary>Indice 1</summary>

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Passez le **pipeline complet** (pas des données déjà transformées) : sinon, fuite de données.
</details>
<details><summary>Indice 2</summary>

`scores = cross_val_score(pipeline, df[FEATURES], df[TARGET], cv=cv, scoring="roc_auc")`.
Utilisez les **mêmes plis** pour les deux modèles et regardez la différence pli par pli :
le signe est-il toujours le même ?
</details>

### 3.5
<details><summary>Indice 1</summary>

Dans un pipeline, les hyperparamètres sont préfixés par le nom de l'étape :
`clf__learning_rate`, `clf__max_depth`, `clf__min_samples_leaf`, `clf__l2_regularization`.
</details>
<details><summary>Indice 2</summary>

Commencez petit (3 × 3 valeurs), `cv=5`, `scoring="roc_auc"`, puis lisez `grid.best_params_` et
`grid.best_score_`. N'évaluez sur le jeu de test **qu'une seule fois**, à la fin. Piste : des arbres
peu profonds et très régularisés se rapprochent d'un modèle linéaire…
</details>

### 3.6
<details><summary>Indice 1</summary>

`LogisticRegression(max_iter=1000, class_weight="balanced")`.
`HistGradientBoostingClassifier` accepte aussi `class_weight`.
</details>
<details><summary>Indice 2</summary>

Au seuil 0,5, comparez précision **et** rappel. Puis regardez la ROC-AUC : change-t-elle ?
Pourquoi ? (Indice dans l'indice : pondérer les classes ressemble beaucoup à… déplacer le seuil.)
</details>

---

## Module 4 — Évaluer et décider

### 4.1
<details><summary>Indice 1</summary>

Gardez un œil sur la matrice de confusion : quelle case grossit quand le seuil baisse ?
</details>
<details><summary>Indice 2</summary>

Précision = TP / (TP + FP) ; rappel = TP / (TP + FN). Reliez chaque formule aux cases de la matrice.
</details>

### 4.2
<details><summary>Indice 1</summary>

Calculez l'accuracy d'un modèle qui prédit « reste » pour tout le monde : `1 - df["churn"].mean()`.
</details>
<details><summary>Indice 2</summary>

Comparez avec l'accuracy de notre modèle. Pour le faire proprement :
`sklearn.dummy.DummyClassifier(strategy="most_frequent")`. Quel est son rappel ?
</details>

### 4.3
<details><summary>Indice 1</summary>

Section « Choisir le seuil avec le métier 💶 » de l'onglet Performance, ou en Python :
`best_threshold_by_cost(y_test, proba, cost_fn=500, cost_fp=50)`.
</details>
<details><summary>Indice 2</summary>

Théorie : on signale un client si `p × coût_FN > (1 − p) × coût_FP`, soit
`seuil = coût_FP / (coût_FN + coût_FP)`. Calculez-le pour les deux scénarios et comparez avec
l'outil. S'ils diffèrent, pourquoi ? (→ exercice 4.5.)
</details>

### 4.4
<details><summary>Indice 1</summary>

`from sklearn.metrics import precision_recall_curve` renvoie `precision, recall, thresholds`.
Attention : `thresholds` a **un élément de moins** que les deux autres.
</details>
<details><summary>Indice 2</summary>

Créez une fonction `pr_points` dans `model.py` sur le modèle de `roc_points`, puis tracez
`go.Scatter(x=recall, y=precision)`. Ajoutez une ligne horizontale au taux de churn : c'est la
précision d'un modèle au hasard.
</details>

### 4.5
<details><summary>Indice 1</summary>

`from sklearn.calibration import calibration_curve`
`prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)`.
</details>
<details><summary>Indice 2</summary>

Tracez `prob_pred` (x) contre `prob_true` (y) avec la diagonale. Sous la diagonale = modèle trop
confiant. Comparez logistique et boosting, et calculez `brier_score_loss`. Si besoin :
`CalibratedClassifierCV`.
</details>

---

## Module 5 — Interpréter et communiquer

### 5.1
<details><summary>Indice 1</summary>

Comparez des **effets**, pas des coefficients bruts : coefficient × écart typique de la variable.
</details>
<details><summary>Indice 2</summary>

L'importance par permutation se partage entre variables **corrélées**. Calculez
`df[["tenure_months", "total_charges"]].corr()` : `total_charges` n'est pas dans `_churn_logit`…
mais que contient-elle ?
</details>

### 5.2
<details><summary>Indice 1</summary>

Copiez le bloc « Taux de churn par contrat » dans `app.py` et remplacez la colonne. La fonction
`bar()` fait déjà la mise en forme.
</details>
<details><summary>Indice 2</summary>

`df.groupby("internet_service")[TARGET].mean().reindex(INTERNET_SERVICES).reset_index()`.
Interprétation : la fibre a un coefficient de +0,5… mais elle est aussi plus chère. Les deux
effets s'additionnent-ils ?
</details>

### 5.3
<details><summary>Indice 1</summary>

Pour la logistique, contribution d'une variable = coefficient × valeur transformée.
`pipeline[:-1].transform(client)` applique toutes les étapes sauf le classifieur
(y compris les variables dérivées).
</details>
<details><summary>Indice 2</summary>

```python
x = pipeline[:-1].transform(client)[0]
noms = pipeline[:-1].get_feature_names_out()
contrib = pd.Series(x * pipeline[-1].coef_[0], index=noms)
contrib.sort_values(key=abs, ascending=False).head(3)
```

Affichez le sens : positif = pousse vers le départ. Affichez-le seulement si le modèle est `logistic`.
</details>

### 5.4
<details><summary>Indice 1</summary>

`contrats = st.sidebar.multiselect("Contrat", CONTRACTS, default=CONTRACTS)`, puis une vue filtrée
`vue = df[df["contract"].isin(contrats)]` utilisée dans l'onglet Vue d'ensemble.
</details>
<details><summary>Indice 2</summary>

Ne filtrez **pas** les données d'entraînement (`get_data()`), seulement l'affichage. Gérez la
sélection vide : `st.warning(...)` puis n'affichez rien.
</details>

### 5.5
<details><summary>Indice 1</summary>

SHAP travaille sur les données **transformées** : `X_t = pipeline[:-1].transform(X_test)` et
`noms = pipeline[:-1].get_feature_names_out()`.
</details>
<details><summary>Indice 2</summary>

`explainer = shap.Explainer(pipeline[-1], X_t, feature_names=noms)` choisit l'algorithme adapté
au modèle ; `valeurs = explainer(X_t[:1])`. Dans Streamlit, le plus simple est de tracer
`valeurs.values[0]` vous-même avec un `px.bar` horizontal. Ajoutez `shap` à `requirements.txt`.
</details>

---

## Module 6 — Industrialiser

### 6.1
<details><summary>Indice 1</summary>

Pour chaque test, posez-vous la question : « quel bug ferait échouer ce test ? »
</details>
<details><summary>Indice 2</summary>

Cassez le code exprès : changez `+ 1.6` en `- 1.6` dans `_churn_logit` et lancez `pytest -q`.
Quel test le détecte ? Annulez ensuite avec `git checkout src/churn/data.py`.
</details>

### 6.2
<details><summary>Indice 1</summary>

La fixture pytest `tmp_path` fournit un dossier temporaire :
`chemin = save(pipeline, metrics, model_dir=tmp_path)`, puis `load(chemin)`.
</details>
<details><summary>Indice 2</summary>

`np.testing.assert_allclose(original.predict_proba(X), recharge.predict_proba(X))`.
Testez avec et sans variables dérivées : le `FunctionTransformer` doit rester importable.
</details>

### 6.3
<details><summary>Indice 1</summary>

Créez `.github/workflows/tests.yml` avec `on: [push, pull_request]` et les étapes :
`actions/checkout@v4`, `actions/setup-python@v5` (Python 3.11), `pip install -r requirements.txt`,
`pytest -q`.
</details>
<details><summary>Indice 2</summary>

Ajoutez un badge au README :
`![tests](https://github.com/VOTRE_COMPTE/VOTRE_DEPOT/actions/workflows/tests.yml/badge.svg)`.
</details>

### 6.4
<details><summary>Indice 1</summary>

`pip install fastapi uvicorn` ; décrivez un client avec une classe Pydantic
(`class Client(BaseModel): ...`, 10 champs) et chargez le modèle **une seule fois** au démarrage.
</details>
<details><summary>Indice 2</summary>

```python
@app.post("/predict")
def predict(client: Client):
    X = pd.DataFrame([client.model_dump()])[FEATURES]
    return {"proba_churn": float(modele.predict_proba(X)[0, 1])}
```

Lancez `uvicorn api:app --reload` et testez sur http://localhost:8000/docs. Pour un test
automatisé : `fastapi.testclient.TestClient` (nécessite `httpx`).
</details>

### 6.5
<details><summary>Indice 1</summary>

Poussez le dépôt sur GitHub, puis sur share.streamlit.io : *New app* → dépôt, branche, `app.py`.
</details>
<details><summary>Indice 2</summary>

`requirements.txt` est lu automatiquement et les données sont générées au premier lancement :
pas besoin de versionner le CSV. L'adresse se termine par `.streamlit.app` : ajoutez-la au README.
</details>

---

## 🏁 Projet final

<details><summary>Plan conseillé pour RAPPORT.md</summary>

1. **Problème** — qui part, combien cela coûte (onglet Vue d'ensemble).
2. **Données** — origine, taille, contrôles qualité, limites.
3. **Modèle retenu** — comparaison, validation croisée, pourquoi ce choix.
4. **Seuil** — coûts métier retenus, précision / rappel obtenus.
5. **Impact** — économies estimées par rapport à « ne rien faire ».
6. **Limites et suite** — dérive des données, biais, prochaines étapes.

Une page, des chiffres, une recommandation claire.
</details>

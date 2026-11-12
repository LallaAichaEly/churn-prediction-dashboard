# 🧭 Parcours de compétences — Data Science appliquée au churn

Ce parcours transforme ce dépôt en terrain d'entraînement. Chaque module :
**lire → comprendre → modifier → vérifier**. Comptez 2 à 4 heures par module.

**Comment progresser efficacement**

1. Travaillez sur une branche par module : `git checkout -b module-3-exercices`.
2. Après chaque modification, lancez `pytest -q` et `python scripts/train.py`.
3. Notez vos résultats (avant / après) dans un fichier `JOURNAL.md` : c'est votre portfolio.
4. Cochez les cases ci-dessous au fur et à mesure.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé

---

## Module 1 — Comprendre les données (`src/churn/data.py`)

**Objectifs** : connaître ses variables, vérifier la qualité, formuler des hypothèses.

- [ ] 🟢 1.1 — Chargez `data/customers.csv` avec pandas. Affichez `df.describe()` et `df["churn"].mean()`.
- [ ] 🟢 1.2 — Calculez le taux de churn par `contract` et par `internet_service` avec `groupby`.
- [ ] 🟡 1.3 — Ajoutez une règle à `validate()` : une facture mensuelle doit être entre 0 et 500 €. Écrivez le test correspondant.
- [ ] 🟡 1.4 — Lisez `_churn_logit`. Écrivez 3 hypothèses sur ce que le modèle va trouver **avant** de l'entraîner.
- [ ] 🔴 1.5 — Remplacez le générateur par le vrai jeu « Telco Customer Churn » (Kaggle). Adaptez les noms de colonnes.

**Auto-évaluation** : Pourquoi est-il utile de connaître la « vraie » cause du churn quand on apprend ?

---

## Module 2 — Préparer les variables (`src/churn/features.py`)

**Objectifs** : encodage, normalisation, pipelines, éviter la fuite de données.

- [ ] 🟢 2.1 — Expliquez avec vos mots ce que font `StandardScaler` et `OneHotEncoder`.
- [ ] 🟢 2.2 — Activez « Variables dérivées » dans le tableau de bord. L'AUC change-t-elle ?
- [ ] 🟡 2.3 — Créez une variable `is_new_customer` (ancienneté < 6 mois). Mesurez son effet.
- [ ] 🟡 2.4 — Pourquoi `handle_unknown="ignore"` ? Lisez le test `test_unknown_category_is_ignored`.
- [ ] 🔴 2.5 — Introduisez volontairement une fuite de données (normaliser **avant** le split) et mesurez la différence. Pourquoi est-ce dangereux en production ?

**Auto-évaluation** : Qu'est-ce que la *data leakage* et comment un `Pipeline` la prévient-il ?

---

## Module 3 — Modéliser (`src/churn/model.py`)

**Objectifs** : modèle de référence, comparaison, validation croisée.

- [ ] 🟢 3.1 — Lancez `python scripts/train.py`. Quel modèle gagne ?
- [ ] 🟢 3.2 — Ajoutez un `RandomForestClassifier` dans `build_pipeline` et `MODELS`.
- [ ] 🟡 3.3 — La régression logistique bat le gradient boosting. Expliquez pourquoi (indice : regardez comment les données sont générées).
- [ ] 🟡 3.4 — Remplacez le split unique par une validation croisée à 5 plis (`cross_val_score`). Les écarts sont-ils significatifs ?
- [ ] 🔴 3.5 — Optimisez les hyperparamètres du boosting avec `GridSearchCV`. Arrivez-vous à battre la logistique ?
- [ ] 🔴 3.6 — Le churn est déséquilibré (~35 %). Testez `class_weight="balanced"` : quel effet sur précision et rappel ?

**Auto-évaluation** : Pourquoi commence-t-on toujours par un modèle simple ?

---

## Module 4 — Évaluer et décider (onglet « Performance »)

**Objectifs** : choisir la bonne métrique, relier le modèle à la valeur métier.

- [ ] 🟢 4.1 — Déplacez le curseur de seuil. Décrivez ce qui arrive à la précision et au rappel.
- [ ] 🟢 4.2 — Pourquoi l'accuracy est-elle trompeuse ici ? (Indice : que vaut un modèle qui prédit « reste » pour tout le monde ?)
- [ ] 🟡 4.3 — Avec un coût de 500 € par client perdu et 50 € par action inutile, quel seuil est optimal ? Et à 100 € / 100 € ?
- [ ] 🟡 4.4 — Ajoutez une courbe précision-rappel (`precision_recall_curve`) dans l'onglet Performance.
- [ ] 🔴 4.5 — Vérifiez la **calibration** du modèle (`CalibrationDisplay`). Une probabilité de 0,7 signifie-t-elle vraiment 70 % ?

**Auto-évaluation** : Expliquez à un manager non technique ce que signifie « rappel de 63 % ».

---

## Module 5 — Interpréter et communiquer (`app.py`)

**Objectifs** : expliquer les prédictions, construire un outil utile.

- [ ] 🟢 5.1 — Dans « Facteurs de churn », comparez le classement à `_churn_logit`. Qu'est-ce qui est retrouvé ? sous-estimé ?
- [ ] 🟢 5.2 — Ajoutez un graphique « taux de churn par service internet » dans la vue d'ensemble.
- [ ] 🟡 5.3 — Dans « Prédire un client », affichez les 3 variables qui poussent le plus la prédiction (coefficients × valeurs pour la logistique).
- [ ] 🟡 5.4 — Ajoutez un filtre par contrat dans la barre latérale qui s'applique à la vue d'ensemble.
- [ ] 🔴 5.5 — Intégrez SHAP (`pip install shap`) pour des explications locales par client.

**Auto-évaluation** : Quelles 3 actions de rétention proposeriez-vous à l'entreprise, chiffres à l'appui ?

---

## Module 6 — Industrialiser (`src/churn/api.py`, `tests/`, `.github/workflows/`)

**Objectifs** : qualité logicielle, API, intégration continue, mise en ligne.

- [ ] 🟢 6.1 — Lancez `pytest -q`. Lisez chaque test et dites ce qu'il protège. Commencez par `test_saved_model_gives_same_predictions` : pourquoi est-il vital ?
- [ ] 🟢 6.2 — Lancez l'API (`uvicorn churn.api:app --app-dir src --reload`) et testez `POST /predict` depuis `http://localhost:8000/docs`. Envoyez `"contract": "Hebdo"` : pourquoi une erreur 422 vaut-elle mieux qu'une prédiction ?
- [ ] 🟡 6.3 — Ajoutez à la réponse de `/predict` les 3 variables qui poussent le plus la prédiction (cf. 5.3). Écrivez le test dans `tests/test_api.py`.
- [ ] 🟡 6.4 — Lisez `.github/workflows/tests.yml`, puis ouvrez l'onglet *Actions* du dépôt GitHub. Ajoutez une étape `ruff check .` (linter) et corrigez ce qu'il signale.
- [ ] 🔴 6.5 — Écrivez un `Dockerfile` pour l'API et lancez-la dans un conteneur (`docker run -p 8000:8000 …`).
- [ ] 🔴 6.6 — Déployez le tableau de bord sur Streamlit Community Cloud et ajoutez le lien au README.

**Auto-évaluation** : Quelle est la différence entre un test unitaire (`test_evaluate_confusion_matrix_sums`) et un test de bout en bout (`test_model_signal_is_visible_through_the_api`) ?

---

## Module 7 — Surveiller en production (`src/churn/monitoring.py`, onglet « Surveillance »)

**Objectifs** : détecter qu'un modèle se dégrade avant que le métier ne s'en plaigne.

- [ ] 🟢 7.1 — Dans l'onglet « Surveillance (drift) », augmentez les prix de 20 %. Quelles variables dérivent ? Que devient le churn moyen prédit ?
- [ ] 🟢 7.2 — Lancez `python scripts/check_drift.py --extra-calls 1`, puis `echo $?`. À quoi sert ce code de sortie dans une tâche planifiée ?
- [ ] 🟡 7.3 — Avec +20 % de prix, `total_charges` augmente aussi de 20 %… mais reste « stable ». Pourquoi ? (Indice : comparez l'écart-type des deux variables.)
- [ ] 🟡 7.4 — Calculez à la main le PSI entre une référence 50 % / 50 % et une période 70 % / 30 %. Vérifiez avec `psi_categorical`.
- [ ] 🔴 7.5 — Le drift des variables n'est pas une baisse de performance. Simulez un *concept drift* (modifiez un coefficient de `_churn_logit`, régénérez les données) et mesurez la chute d'AUC du modèle entraîné sur l'ancien jeu.
- [ ] 🔴 7.6 — Ajoutez une GitHub Action planifiée (`on: schedule`) qui lance `check_drift.py` chaque nuit.

**Auto-évaluation** : Si le PSI du *score* du modèle est élevé mais qu'aucune variable ne dérive seule, que se passe-t-il ?

---

## 🏁 Projet final

Rédigez un mini-rapport (1 page) **pour un public non technique** : problème, données, modèle retenu,
seuil choisi et pourquoi, impact financier estimé, limites. Ajoutez-le au dépôt — c'est une pièce
maîtresse pour un portfolio ou un entretien.

Point de départ : la [fiche modèle](RAPPORT_MODELE.md) générée par `python scripts/report.py`
contient tous les chiffres, mais elle est écrite pour des data scientists. Votre travail est de la
**traduire** : que doit retenir un directeur commercial en 2 minutes ? Testez aussi d'autres coûts
(`--cost-fn 300 --cost-fp 80`) : votre recommandation change-t-elle ?

## 🚀 Et après ? Projets suivants

Une fois ce parcours terminé, réutilisez la même démarche (données → modèle → décision → production)
sur un problème différent :

1. **Segmentation client** (non supervisé) : K-means sur les mêmes clients, profils de segments, churn par segment.
2. **Prévision de ventes** (séries temporelles) : saisonnalité, validation temporelle, intervalle de prévision.
3. **Classification de textes** : avis clients → sentiment, avec un vrai jeu de données public.

## 📚 Ressources

- Documentation scikit-learn — *User Guide* (pipelines, métriques, inspection)
- Documentation Streamlit — *Get started*
- Documentation FastAPI — *Tutorial - User Guide* (validation, tests, dépendances)
- *Hands-On Machine Learning* (A. Géron) — chapitres 2 et 3

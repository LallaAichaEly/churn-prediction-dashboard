# 🧭 Parcours de compétences — Data Science appliquée au churn

Ce parcours transforme ce dépôt en terrain d'entraînement. Chaque module :
**lire → comprendre → modifier → vérifier**. Comptez 2 à 4 heures par module.

**Comment progresser efficacement**

1. Travaillez sur une branche par module : `git checkout -b module-3-exercices`.
2. Après chaque modification, lancez `pytest -q` et `python scripts/train.py`.
3. Notez vos résultats (avant / après) dans un fichier `JOURNAL.md` : c'est votre portfolio.
4. Cochez les cases ci-dessous au fur et à mesure.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé · 📘 solution de référence dans le code

> 📘 **Exercices avec solution de référence.** Certains exercices ont été implémentés pour que
> le projet soit complet (API, CI, calibration…). Pour eux, l'exercice devient : **lire,
> expliquer, puis étendre** la solution. Pour vous entraîner « à vide », recréez-la sur une
> branche partie du premier commit : `git checkout -b refaire-6.4 fad2cb7`.

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
- [ ] 📘🟡 4.4 — La courbe précision-rappel est dans l'onglet Performance (`pr_points`). Pourquoi sa ligne « hasard » est-elle au niveau du taux de churn et non à 0,5 comme pour la ROC ? Placez le point orange (seuil actuel) là où le rappel atteint 80 % : quelle précision obtenez-vous ?
- [ ] 📘🔴 4.5 — Lisez `calibration_points`. Comparez la calibration de la logistique et du boosting. Puis enveloppez le boosting dans `CalibratedClassifierCV` : le score de Brier s'améliore-t-il ?

**Auto-évaluation** : Expliquez à un manager non technique ce que signifie « rappel de 63 % ».

---

## Module 5 — Interpréter et communiquer (`app.py`)

**Objectifs** : expliquer les prédictions, construire un outil utile.

- [ ] 🟢 5.1 — Dans « Facteurs de churn », comparez le classement à `_churn_logit`. Qu'est-ce qui est retrouvé ? sous-estimé ?
- [ ] 🟢 5.2 — Ajoutez un graphique « taux de churn par service internet » dans la vue d'ensemble.
- [ ] 📘🟡 5.3 — Lisez `src/churn/explain.py` (méthode « et si… ? », affichée dans « Prédire un client » et « Clients à risque »). Dans « Clients à risque », le contrat mensuel n'apparaît jamais comme cause : pourquoi ? (indice : `typical_profile`). Expliquez aussi pourquoi `tenure_months` et `total_charges` sont mal séparés, puis, pour la logistique, comparez avec coefficients × valeurs normalisées.
- [ ] 🟡 5.4 — Ajoutez un filtre par contrat dans la barre latérale qui s'applique à la vue d'ensemble.
- [ ] 🔴 5.5 — Intégrez SHAP (`pip install shap`) pour des explications locales par client.

**Auto-évaluation** : Quelles 3 actions de rétention proposeriez-vous à l'entreprise, chiffres à l'appui ?

---

## Module 6 — Industrialiser (`tests/`, déploiement)

**Objectifs** : qualité logicielle, reproductibilité, mise en ligne.

- [ ] 🟢 6.1 — Lancez `pytest -q`. Lisez chaque test et dites ce qu'il protège.
- [ ] 📘🟡 6.2 — Le test `test_saved_model_gives_same_predictions` existe. Cassez-le volontairement (ex. ré-entraînez avec une autre graine avant de comparer) pour voir le message d'erreur, puis réparez.
- [ ] 📘🟡 6.3 — La GitHub Action `.github/workflows/tests.yml` lance les tests à chaque push. Ouvrez l'onglet *Actions* du dépôt, puis ajoutez une étape de *linting* (`ruff check .`).
- [ ] 📘🔴 6.4 — L'API est dans `api.py` (`uvicorn api:app --reload`, puis http://localhost:8000/docs). Ajoutez un endpoint `GET /model-info` qui renvoie les métriques de `models/churn_model_metrics.json`, avec son test.
- [ ] 🔴 6.5 — Déployez le tableau de bord sur Streamlit Community Cloud et ajoutez le lien au README.

**Auto-évaluation** : Si les données changent dans 6 mois (*data drift*), comment le détecteriez-vous ?

---

## 🏁 Projet final

Rédigez un mini-rapport (1 page) : problème, données, modèle retenu, seuil choisi et pourquoi,
impact financier estimé, limites. Ajoutez-le au dépôt — c'est une pièce maîtresse pour un portfolio
ou un entretien.

## 📚 Ressources

- Documentation scikit-learn — *User Guide* (pipelines, métriques, inspection)
- Documentation Streamlit — *Get started*
- *Hands-On Machine Learning* (A. Géron) — chapitres 2 et 3

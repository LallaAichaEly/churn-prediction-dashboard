# 🧭 Parcours de compétences — Data Science appliquée au churn

Ce parcours transforme ce dépôt en terrain d'entraînement. Chaque module :
**lire → comprendre → modifier → vérifier**. Comptez 2 à 4 heures par module.

**Comment progresser efficacement**

1. Travaillez sur une branche par module : `git checkout -b module-3-exercices`.
2. Après chaque modification, lancez `pytest -q` et `python scripts/train.py`.
3. Notez vos résultats (avant / après) dans un fichier `JOURNAL.md` : c'est votre portfolio.
4. Cochez les cases ci-dessous au fur et à mesure.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé

> ⚡ **Entraînement corrigé automatiquement.** En parallèle des modules, faites les
> [défis](defis/README.md) : 10 fonctions courtes dans `defis/serie_01.py`, vérifiées par
> `python scripts/progression.py`. Les exercices marqués **⚡** ci-dessous ont un défi associé :
> faites le défi d'abord, puis l'exercice dans le vrai projet.

---

## Module 1 — Comprendre les données (`src/churn/data.py`)

**Objectifs** : connaître ses variables, vérifier la qualité, formuler des hypothèses.

- [ ] 🟢 1.1 — Chargez `data/customers.csv` avec pandas. Affichez `df.describe()` et `df["churn"].mean()`.
- [ ] 🟢 1.2 ⚡ — Calculez le taux de churn par `contract` et par `internet_service` avec `groupby`.
- [ ] 🟡 1.3 — Ajoutez une règle à `validate()` : une facture mensuelle doit être entre 0 et 500 €. Écrivez le test correspondant.
- [ ] 🟡 1.4 — Lisez `_churn_logit`. Écrivez 3 hypothèses sur ce que le modèle va trouver **avant** de l'entraîner.
- [ ] 🔴 1.5 — Remplacez le générateur par le vrai jeu « Telco Customer Churn » (Kaggle). Adaptez les noms de colonnes.

**Auto-évaluation** : Pourquoi est-il utile de connaître la « vraie » cause du churn quand on apprend ?

---

## Module 2 — Préparer les variables (`src/churn/features.py`)

**Objectifs** : encodage, normalisation, pipelines, éviter la fuite de données.

- [ ] 🟢 2.1 ⚡ — Expliquez avec vos mots ce que font `StandardScaler` et `OneHotEncoder`.
- [ ] 🟢 2.2 — Activez « Variables dérivées » dans le tableau de bord. L'AUC change-t-elle ?
- [ ] 🟡 2.3 ⚡ — Créez une variable `is_new_customer` (ancienneté < 6 mois). Mesurez son effet.
- [ ] 🟡 2.4 — Pourquoi `handle_unknown="ignore"` ? Lisez le test `test_unknown_category_is_ignored`.
- [ ] 🔴 2.5 ⚡ — Introduisez volontairement une fuite de données (normaliser **avant** le split) et mesurez la différence. Pourquoi est-ce dangereux en production ?

**Auto-évaluation** : Qu'est-ce que la *data leakage* et comment un `Pipeline` la prévient-il ?

---

## Module 3 — Modéliser (`src/churn/model.py`)

**Objectifs** : modèle de référence, comparaison, validation croisée.

- [ ] 🟢 3.1 — Lancez `python scripts/train.py`. Quel modèle gagne ?
- [ ] 🟢 3.2 — Ajoutez un `RandomForestClassifier` dans `build_pipeline` et `MODELS`.
- [ ] 🟡 3.3 — La régression logistique bat le gradient boosting. Expliquez pourquoi (indice : regardez comment les données sont générées).
- [ ] 🟡 3.4 ⚡ — Remplacez le split unique par une validation croisée à 5 plis (`cross_val_score`). Les écarts sont-ils significatifs ?
- [ ] 🔴 3.5 — Optimisez les hyperparamètres du boosting avec `GridSearchCV`. Arrivez-vous à battre la logistique ?
- [ ] 🔴 3.6 — Le churn est déséquilibré (~35 %). Testez `class_weight="balanced"` : quel effet sur précision et rappel ?

**Auto-évaluation** : Pourquoi commence-t-on toujours par un modèle simple ?

---

## Module 4 — Évaluer et décider (onglet « Performance »)

**Objectifs** : choisir la bonne métrique, relier le modèle à la valeur métier.

- [ ] 🟢 4.1 — Déplacez le curseur de seuil. Décrivez ce qui arrive à la précision et au rappel.
- [ ] 🟢 4.2 ⚡ — Pourquoi l'accuracy est-elle trompeuse ici ? (Indice : que vaut un modèle qui prédit « reste » pour tout le monde ?)
- [ ] 🟡 4.3 ⚡ — Avec un coût de 500 € par client perdu et 50 € par action inutile, quel seuil est optimal ? Et à 100 € / 100 € ?
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

## Module 6 — Industrialiser (`tests/`, déploiement)

**Objectifs** : qualité logicielle, reproductibilité, mise en ligne.

- [ ] 🟢 6.1 — Lancez `pytest -q`. Lisez chaque test et dites ce qu'il protège.
- [ ] 🟡 6.2 — Écrivez un test qui vérifie que le modèle sauvegardé (`models/churn_model.joblib`) donne les mêmes prédictions une fois rechargé.
- [ ] 🟡 6.3 — Ajoutez une GitHub Action qui lance les tests à chaque push.
- [ ] 🔴 6.4 — Exposez le modèle via une API FastAPI (`POST /predict`).
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

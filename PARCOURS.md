# 🧭 Parcours de compétences — Data Science appliquée au churn

Ce parcours transforme ce dépôt en terrain d'entraînement. Chaque module :
**lire → comprendre → modifier → vérifier**. Comptez 2 à 4 heures par module.

**Comment progresser efficacement**

1. Travaillez sur une branche par module : `git checkout -b module-3-exercices`.
2. Après chaque modification, lancez `pytest -q` et `python scripts/train.py`.
3. Notez vos résultats (avant / après) dans un fichier `JOURNAL.md` : c'est votre portfolio.
4. Cochez les cases ci-dessous au fur et à mesure.
5. 📘 = **exemple résolu** : la solution est déjà dans le code. Étudiez-la, puis faites l'extension proposée.

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

## Module 6 — Industrialiser (`tests/`, déploiement)

**Objectifs** : qualité logicielle, reproductibilité, mise en ligne.

- [ ] 🟢 6.1 — Lancez `pytest -q` (33 tests). Ouvrez chaque fichier de `tests/` et dites, pour chaque test, quelle erreur il empêche.
- [ ] 🟡 6.2 — 📘 `test_saved_model_gives_same_predictions` vérifie qu'un modèle rechargé prédit à l'identique.
  **Extension** : modifiez `save()` pour ne sauvegarder que `pipeline.named_steps["clf"]`. Quel test casse, et pourquoi faut-il sauvegarder le pipeline entier ? (Annulez ensuite la modification.)
- [ ] 🟡 6.3 — 📘 `.github/workflows/tests.yml` lance les tests à chaque push.
  **Extension** : poussez un commit qui casse un test, observez la croix rouge dans l'onglet *Actions*, puis réparez. Ajoutez une étape `ruff check .` (linter).
- [ ] 🔴 6.4 — 📘 `src/churn/api.py` expose `POST /predict` et `POST /predict/batch`. Lancez-la et testez-la depuis `http://localhost:8000/docs`.
  **Extension** : ajoutez une route `GET /model` qui renvoie le contenu de `models/churn_model_metrics.json`, avec son test dans `tests/test_api.py`.
- [ ] 🔴 6.5 — Déployez le tableau de bord sur Streamlit Community Cloud (voir README) et ajoutez le lien au README.
- [ ] 🔴 6.6 — Écrivez un `Dockerfile` pour l'API (`python:3.12-slim`, `pip install -r requirements.txt`, `CMD uvicorn ...`). Vérifiez avec `docker run -p 8000:8000`.
- [ ] 🔴 6.7 — Détection de dérive : écrivez `detect_drift(reference, current)` qui compare chaque variable numérique avec un test de Kolmogorov-Smirnov (`scipy.stats.ks_2samp`). Testez-la avec des données dont vous augmentez artificiellement `monthly_charges` de 20 %.

**Auto-évaluation** : Si les données changent dans 6 mois (*data drift*), comment le détecteriez-vous ?

---

## 🏁 Projet final

Rédigez un mini-rapport (1 page) : problème, données, modèle retenu, seuil choisi et pourquoi,
impact financier estimé, limites. Ajoutez-le au dépôt — c'est une pièce maîtresse pour un portfolio
ou un entretien.

📘 Un exemple complet est généré par `python scripts/report.py` → [`RAPPORT.md`](RAPPORT.md).
**Écrivez le vôtre d'abord, sans le regarder**, puis comparez : qu'avez-vous oublié ? qu'avez-vous vu
que le rapport automatique ne dit pas ?

- [ ] 🟡 F.1 — Votre rapport, puis la comparaison avec `RAPPORT.md`.
- [ ] 🟡 F.2 — Le rapport montre qu'avec une action à 50 €, « contacter tout le monde » est presque aussi bon que le modèle. Expliquez pourquoi à un manager, en 3 phrases.
- [ ] 🔴 F.3 — Améliorez le modèle de coût : une action coûte 50 € **à chaque client contacté** et ne réussit que 30 % du temps. Modifiez `best_threshold_by_cost`, ajoutez un test, régénérez le rapport. Le seuil optimal monte-t-il ou baisse-t-il ?

## 🚀 Et après ? Vos prochains projets

Ce projet terminé, enchaînez avec un sujet qui travaille de **nouvelles** compétences, en réutilisant
la même structure (`src/`, `tests/`, tableau de bord, API, rapport) :

| Projet | Nouvelles compétences |
|---|---|
| Prévision des ventes d'un magasin | Séries temporelles, validation temporelle, saisonnalité |
| Analyse des avis clients | NLP : nettoyage de texte, TF-IDF, classification de sentiments |
| Détection de fraude bancaire | Déséquilibre extrême (< 1 %), courbe précision-rappel, rééchantillonnage |
| Recommandation de produits | Filtrage collaboratif, métriques de classement (précision@k) |

## 📚 Ressources

- Documentation scikit-learn — *User Guide* (pipelines, métriques, inspection)
- Documentation Streamlit — *Get started*
- *Hands-On Machine Learning* (A. Géron) — chapitres 2 et 3

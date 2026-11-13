# 📉 Churn Prediction Dashboard

[![Tests](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml)

Un projet complet **et pédagogique** pour développer ses compétences en data science :
des données brutes jusqu'à un tableau de bord interactif et une API qui prédisent quels clients
risquent de partir (*churn*).

> 🎯 **Objectif : apprendre en construisant.**
> - [`PARCOURS.md`](PARCOURS.md) — parcours de compétences en 6 modules, exercices progressifs, checklist.
> - [`CORRIGES.md`](CORRIGES.md) — corrigés commentés, chiffres à l'appui (à ouvrir *après* avoir cherché).
> - [`RAPPORT.md`](RAPPORT.md) — exemple de rapport final, modèle pour votre portfolio.

## Ce que vous allez pratiquer

| Compétence | Où dans le code |
|---|---|
| Génération / validation de données | `src/churn/data.py` |
| Feature engineering & pipelines sans fuite de données | `src/churn/features.py` |
| Modélisation (régression logistique, gradient boosting) | `src/churn/model.py` |
| Évaluation : ROC-AUC, précision, rappel, seuil métier | `src/churn/model.py` |
| Interprétabilité (importance par permutation) | `src/churn/model.py` |
| Tableau de bord interactif | `app.py` (Streamlit + Plotly) |
| API de prédiction | `api.py` (FastAPI) |
| Tests automatisés & intégration continue | `tests/` (pytest), `.github/workflows/tests.yml` |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py          # entraîne et compare les modèles
streamlit run app.py             # tableau de bord sur http://localhost:8501
uvicorn api:app --reload         # API sur http://localhost:8000/docs
pytest -q                        # lance les 27 tests
```

Les données sont **synthétiques** et générées au premier lancement dans `data/customers.csv`
(5 000 clients, ~35 % de churn). Aucune connexion ni donnée personnelle n'est nécessaire.

## Le tableau de bord

| Onglet | Ce qu'il montre |
|---|---|
| **Vue d'ensemble** | KPIs, taux de churn par contrat, ancienneté, appels au support, paiement |
| **Performance** | ROC-AUC, matrice de confusion, compromis précision/rappel, **seuil optimal selon les coûts métier** |
| **Facteurs de churn** | Variables les plus influentes (importance par permutation) |
| **Clients à risque** | Liste triée par probabilité de départ, export CSV |
| **Prédire un client** | Formulaire → probabilité de churn d'un profil |

Dans la barre latérale : choix du modèle, activation des variables dérivées, seuil de décision.

## L'API

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.3" -H "Content-Type: application/json" -d '{
  "tenure_months": 3, "monthly_charges": 89.9, "support_calls": 4, "num_products": 1,
  "contract": "Mensuel", "payment_method": "Chèque", "internet_service": "Fibre",
  "tech_support": "Non", "senior": "Non"}'
# → {"churn_probability": 0.97, "at_risk": true, "threshold": 0.3, "model": "LogisticRegression"}
```

Les entrées sont validées (types, bornes, modalités connues) : une requête invalide reçoit une
erreur HTTP 422 explicite. `GET /health` sert de sonde de santé.

## Résultats de référence

```
          roc_auc  accuracy  precision  recall     f1     (jeu de test, seuil 0,5)
logistic    0.834     0.769      0.689   0.634  0.660
gboost      0.807     0.746      0.655   0.596  0.624
```

En validation croisée à 5 plis : logistique **0,819 ± 0,005**, forêt aléatoire 0,806, boosting 0,799
(0,812 une fois optimisé). 💡 *Pourquoi le modèle simple gagne-t-il ?* C'est l'exercice 3.3 du parcours.

## Structure

```
├── app.py                  # Tableau de bord Streamlit
├── api.py                  # API FastAPI (POST /predict, GET /health)
├── scripts/train.py        # Entraînement + comparaison en ligne de commande
├── src/churn/
│   ├── data.py             # Données synthétiques + validation
│   ├── features.py         # Préparation des variables
│   └── model.py            # Entraînement, évaluation, seuil, importance, sauvegarde
├── tests/                  # Données, modèles, API, tableau de bord, corrigés
├── corriges/               # Un script exécutable par module du parcours
├── .github/workflows/      # Intégration continue (tests à chaque push)
├── PARCOURS.md             # 🧭 Parcours de compétences et exercices
├── CORRIGES.md             # ✅ Corrigés commentés
├── RAPPORT.md              # 📄 Exemple de rapport final
└── requirements.txt
```

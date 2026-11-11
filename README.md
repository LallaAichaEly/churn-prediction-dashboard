# 📉 Churn Prediction Dashboard

Un projet complet **et pédagogique** pour développer ses compétences en data science :
des données brutes jusqu'à un tableau de bord interactif et une API qui prédisent quels
clients risquent de partir (*churn*).

> 🎯 **Objectif : apprendre en construisant.** Le fichier [`PARCOURS.md`](PARCOURS.md) est
> un parcours de compétences en 6 modules, avec des exercices progressifs et une checklist
> pour suivre votre progression. Les corrigés sont dans [`CORRIGES.md`](CORRIGES.md) :
> à ouvrir **après** avoir essayé.

## Ce que vous allez pratiquer

| Compétence | Où dans le code |
|---|---|
| Génération / validation de données | `src/churn/data.py` |
| Feature engineering & pipelines sans fuite de données | `src/churn/features.py` |
| Modélisation (régression logistique, gradient boosting) | `src/churn/model.py` |
| Évaluation : ROC-AUC, précision, rappel, seuil métier | `src/churn/model.py` |
| Interprétabilité (importance par permutation) | `src/churn/model.py` |
| Tableau de bord interactif | `app.py` (Streamlit + Plotly) |
| Mise en production : API REST | `api.py` (FastAPI) |
| Tests automatisés & intégration continue | `tests/` (pytest), `.github/workflows/tests.yml` |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py          # entraîne et compare les modèles
streamlit run app.py             # tableau de bord sur http://localhost:8501
uvicorn api:app --reload         # API sur http://localhost:8000/docs
pytest -q                        # lance les tests (~40 s)
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
curl -X POST "http://localhost:8000/predict?threshold=0.4" -H "Content-Type: application/json" -d '{
  "tenure_months": 5, "monthly_charges": 95, "total_charges": 475, "support_calls": 4,
  "num_products": 1, "contract": "Mensuel", "payment_method": "Chèque",
  "internet_service": "Fibre", "tech_support": "Non", "senior": "Non"}'
# → {"churn_probability": 0.9678, "at_risk": true, "threshold": 0.4}
```

| Route | Rôle |
|---|---|
| `GET /health` | État de l'API et métriques du modèle chargé |
| `POST /predict` | Probabilité de churn d'un client (`threshold` optionnel) |

Les entrées sont validées : une faute de frappe (`"contract": "Mensual"`) renvoie une erreur 422.
Le modèle est lu dans `models/churn_model.joblib` (ou `$CHURN_MODEL_PATH`) et entraîné
automatiquement s'il n'existe pas encore.

## Résultats de référence

```
          roc_auc  accuracy  precision  recall     f1
logistic    0.834     0.769      0.689   0.634  0.660
gboost      0.807     0.746      0.655   0.596  0.624
```

💡 *Pourquoi le modèle simple gagne-t-il ?* C'est la question de l'exercice 3.3 du parcours.

## Structure

```
├── app.py                  # Tableau de bord Streamlit
├── api.py                  # API REST FastAPI
├── scripts/train.py        # Entraînement + comparaison en ligne de commande
├── src/churn/
│   ├── data.py             # Données synthétiques + validation
│   ├── features.py         # Préparation des variables
│   └── model.py            # Entraînement, évaluation, seuil, importance
├── solutions/              # Corrigés exécutables des modules 1 à 5
├── tests/                  # Tests : cœur, API, tableau de bord, corrigés
├── .github/workflows/      # Intégration continue (pytest à chaque push)
├── PARCOURS.md             # 🧭 Parcours de compétences et exercices
├── CORRIGES.md             # 📗 Corrigés commentés + exemple de rapport final
└── requirements.txt
```

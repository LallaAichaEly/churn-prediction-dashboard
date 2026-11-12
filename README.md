# 📉 Churn Prediction Dashboard

[![Tests](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml)

Un projet complet **et pédagogique** pour développer ses compétences en data science :
des données brutes jusqu'à un tableau de bord interactif qui prédit quels clients risquent
de partir (*churn*).

> 🎯 **Objectif : apprendre en construisant.** Le fichier [`PARCOURS.md`](PARCOURS.md) est
> un parcours de compétences en 6 modules, avec des exercices progressifs et une checklist
> pour suivre votre progression. Le résultat final — modèle retenu, seuil, impact en euros —
> est dans [`RAPPORT.md`](RAPPORT.md), et l'historique dans [`JOURNAL.md`](JOURNAL.md).

## Ce que vous allez pratiquer

| Compétence | Où dans le code |
|---|---|
| Génération / validation de données | `src/churn/data.py` |
| Feature engineering & pipelines sans fuite de données | `src/churn/features.py` |
| Modélisation (régression logistique, gradient boosting) | `src/churn/model.py` |
| Évaluation : ROC-AUC, précision, rappel, seuil métier | `src/churn/model.py` |
| Calibration des probabilités, courbe précision-rappel | `src/churn/model.py` |
| Interprétabilité globale (importance par permutation) | `src/churn/model.py` |
| Interprétabilité locale (« pourquoi CE client ? ») | `src/churn/explain.py` |
| Tableau de bord interactif | `app.py` (Streamlit + Plotly) |
| API REST avec validation des entrées | `api.py` (FastAPI + Pydantic) |
| Tests automatisés + intégration continue | `tests/` (pytest), `.github/workflows/tests.yml` |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py          # entraîne et compare les modèles
streamlit run app.py             # ouvre le tableau de bord sur http://localhost:8501
uvicorn api:app --reload         # API sur http://localhost:8000/docs
pytest -q                        # lance les tests
```

Les données sont **synthétiques** et générées au premier lancement dans `data/customers.csv`
(5 000 clients, ~35 % de churn). Aucune connexion ni donnée personnelle n'est nécessaire.

## Le tableau de bord

| Onglet | Ce qu'il montre |
|---|---|
| **Vue d'ensemble** | KPIs, taux de churn par contrat, ancienneté, appels au support, paiement |
| **Performance** | ROC-AUC, matrice de confusion, compromis précision/rappel, courbe PR, **calibration**, **seuil optimal selon les coûts métier** |
| **Facteurs de churn** | Variables les plus influentes (importance par permutation) |
| **Clients à risque** | Liste triée par probabilité de départ, export CSV, **explication par client** |
| **Prédire un client** | Formulaire → probabilité de churn d'un profil + ce qui pèse dans ce score |

## L'API

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.3" -H "Content-Type: application/json" -d '{
  "tenure_months": 3, "monthly_charges": 95, "total_charges": 285, "support_calls": 4,
  "num_products": 1, "contract": "Mensuel", "payment_method": "Chèque",
  "internet_service": "Fibre", "tech_support": "Non", "senior": "Non"}'
# → {"churn_probability": 0.97…, "at_risk": true, "threshold": 0.3}
```

`POST /predict/batch` accepte une liste de clients. Une valeur invalide (contrat inconnu,
ancienneté négative…) renvoie une erreur **422** explicite au lieu d'une prédiction absurde.

Dans la barre latérale : choix du modèle, activation des variables dérivées, seuil de décision.

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
│   ├── model.py            # Entraînement, évaluation, seuil, calibration, importance
│   └── explain.py          # Explication locale d'une prédiction
├── tests/                  # test_churn.py (modèle), test_api.py (API), test_app.py (dashboard)
├── .github/workflows/      # Intégration continue (tests à chaque push)
├── PARCOURS.md             # 🧭 Parcours de compétences et exercices
├── RAPPORT.md              # 🏁 Rapport final : modèle retenu, seuil, impact financier
├── JOURNAL.md              # 📓 Journal de progression du projet
└── requirements.txt
```

# 📉 Churn Prediction Dashboard

[![Tests](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml)

Un projet complet **et pédagogique** pour développer ses compétences en data science :
des données brutes jusqu'à un tableau de bord interactif qui prédit quels clients risquent
de partir (*churn*), servi par une API web et protégé par des tests automatiques.

> 🎯 **Objectif : apprendre en construisant.** Le fichier [`PARCOURS.md`](PARCOURS.md) est
> un parcours de compétences en 6 modules, avec des exercices progressifs et une checklist
> pour suivre votre progression.

## Ce que vous allez pratiquer

| Compétence | Où dans le code |
|---|---|
| Génération / validation de données | `src/churn/data.py` |
| Feature engineering & pipelines sans fuite de données | `src/churn/features.py` |
| Modélisation (régression logistique, gradient boosting) | `src/churn/model.py` |
| Évaluation : ROC-AUC, précision, rappel, seuil métier | `src/churn/model.py` |
| Interprétabilité (importance par permutation) | `src/churn/model.py` |
| Seuil de décision choisi par validation croisée selon les coûts | `src/churn/model.py` (`tune_threshold`) |
| Tableau de bord interactif | `app.py` (Streamlit + Plotly) |
| API web de prédiction, validation des entrées | `src/churn/api.py` (FastAPI + Pydantic) |
| Tests automatisés (données, modèles, API, tableau de bord) | `tests/` (pytest) |
| Intégration continue | `.github/workflows/tests.yml` (GitHub Actions) |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py          # entraîne, compare et sauvegarde le meilleur modèle
streamlit run app.py             # ouvre le tableau de bord sur http://localhost:8501
uvicorn churn.api:app --app-dir src --reload   # API sur http://localhost:8000/docs
pytest -q                        # lance les 32 tests
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

Une fois le modèle entraîné, n'importe quelle application peut l'interroger :

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure_months": 3, "monthly_charges": 95, "total_charges": 285, "support_calls": 4,
  "num_products": 1, "contract": "Mensuel", "payment_method": "Chèque",
  "internet_service": "Fibre", "tech_support": "Non", "senior": "Non"}'
# → {"churn_probability": 0.97, "at_risk": true, "threshold": 0.13}
```

| Route | Rôle |
|---|---|
| `GET /health` | État de l'API, modèle servi, seuil et ROC-AUC |
| `POST /predict` | Probabilité de churn d'un client (`?threshold=0.3` pour forcer un seuil) |
| `POST /predict/batch` | Jusqu'à 1 000 clients en un appel |
| `GET /docs` | Documentation interactive (essayez les routes depuis le navigateur) |

Une modalité mal orthographiée (`"mensuel"`) ou une valeur aberrante est refusée avec une
erreur 422 explicite, au lieu de produire une prédiction silencieusement fausse.

## Résultats de référence

`python scripts/train.py` (coûts par défaut : 500 € par client perdu, 50 € par action inutile) :

```
          roc_auc  threshold  precision  recall     f1  coût_test_€
logistic    0.834       0.13      0.457   0.975  0.622      31150.0
gboost      0.807       0.05      0.422   0.986  0.591      32900.0
```

Au seuil standard de 0,5, la régression logistique n'aurait détecté que 63 % des départs.
Comme rater un client coûte 10 fois plus cher qu'une action inutile, le seuil choisi descend
à 0,13 : on contacte plus de clients, mais on en perd beaucoup moins. Ce seuil est choisi par
validation croisée **sur le jeu d'entraînement** ; le jeu de test ne sert qu'à l'évaluation finale.

💡 *Pourquoi le modèle simple gagne-t-il ?* C'est la question de l'exercice 3.3 du parcours.

## Structure

```
├── app.py                      # Tableau de bord Streamlit
├── scripts/train.py            # Entraînement + comparaison en ligne de commande
├── src/churn/
│   ├── data.py                 # Données synthétiques + validation
│   ├── features.py             # Préparation des variables
│   ├── model.py                # Entraînement, évaluation, seuil, importance, sauvegarde
│   └── api.py                  # API FastAPI
├── tests/
│   ├── test_churn.py           # Données, variables, modèles, seuil, sauvegarde
│   ├── test_api.py             # Routes et validation de l'API
│   └── test_app.py             # Le tableau de bord s'exécute sans erreur
├── .github/workflows/tests.yml # CI : tests à chaque push
├── PARCOURS.md                 # 🧭 Parcours de compétences et exercices
└── requirements.txt
```

## État du projet

**v0.2.0 : chaîne complète, de la donnée à l'API.** Données → modèle → seuil métier →
tableau de bord → API → tests → CI. Les exercices des modules 1 à 5 de `PARCOURS.md` restent
à faire : ils sont pour vous. Le module 6 repart désormais de l'API et de la CI existantes.

# 📉 Churn Prediction Dashboard

[![Tests](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml)

Un projet complet **et pédagogique** pour développer ses compétences en data science :
des données brutes jusqu'à un modèle **en production** — tableau de bord interactif, API REST,
surveillance du drift et intégration continue.

> 🎯 **Objectif : apprendre en construisant.** Le fichier [`PARCOURS.md`](PARCOURS.md) est
> un parcours de compétences en 7 modules, avec des exercices progressifs et une checklist
> pour suivre votre progression.

## ✅ État du projet — v1.0 terminée

| Étape | Livrable | Statut |
|---|---|:---:|
| Données | Générateur synthétique + contrôles qualité | ✅ |
| Modélisation | Logistique vs gradient boosting, seuil métier, importance des variables | ✅ |
| Tableau de bord | Streamlit, 6 onglets | ✅ |
| API REST | FastAPI : `/predict`, `/predict/batch`, `/model`, `/health` | ✅ |
| Surveillance | Drift (PSI) : script d'alerte + onglet dédié | ✅ |
| Qualité | 43 tests pytest + GitHub Actions (Python 3.11 et 3.13) | ✅ |
| Documentation | [Fiche modèle](RAPPORT_MODELE.md) générée automatiquement | ✅ |

## Ce que vous allez pratiquer

| Compétence | Où dans le code |
|---|---|
| Génération / validation de données | `src/churn/data.py` |
| Feature engineering & pipelines sans fuite de données | `src/churn/features.py` |
| Modélisation (régression logistique, gradient boosting) | `src/churn/model.py` |
| Évaluation : ROC-AUC, précision, rappel, seuil métier | `src/churn/model.py` |
| Interprétabilité (importance par permutation) | `src/churn/model.py` |
| Tableau de bord interactif | `app.py` (Streamlit + Plotly) |
| API REST et validation des entrées | `src/churn/api.py` (FastAPI + Pydantic) |
| Surveillance en production (data drift, PSI) | `src/churn/monitoring.py`, `scripts/check_drift.py` |
| Tests automatisés et intégration continue | `tests/` (pytest), `.github/workflows/tests.yml` |
| Communication des résultats | `scripts/report.py` → `RAPPORT_MODELE.md` |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py          # entraîne et compare les modèles, sauvegarde le meilleur
streamlit run app.py             # tableau de bord sur http://localhost:8501
pytest -q                        # lance les 43 tests
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
| **Surveillance (drift)** | Simulez une hausse de prix, plus d'appels au support… et voyez quelles variables dérivent |

Dans la barre latérale : choix du modèle, activation des variables dérivées, seuil de décision.

## L'API

```bash
python scripts/train.py                          # le modèle doit exister
uvicorn churn.api:app --app-dir src --reload     # documentation interactive : http://localhost:8000/docs
```

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.4" \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 4, "monthly_charges": 89.9, "total_charges": 359.6, "support_calls": 3,
       "num_products": 1, "contract": "Mensuel", "payment_method": "Chèque",
       "internet_service": "Fibre", "tech_support": "Non", "senior": "Non"}'
# {"churn_probability":0.9497,"at_risk":true,"threshold":0.4}
```

| Route | Rôle |
|---|---|
| `GET /health` | Le service répond-il ? |
| `GET /model` | Modèle servi, variables, métriques d'évaluation |
| `POST /predict` | Probabilité de churn d'un client (`?threshold=` optionnel) |
| `POST /predict/batch` | Jusqu'à 1 000 clients d'un coup : `{"customers": [...]}` |

Une valeur invalide (contrat inconnu, ancienneté négative…) renvoie une **erreur 422** explicite
au lieu d'une prédiction absurde. Chemin du modèle modifiable via `CHURN_MODEL_PATH`.

## Surveiller le drift

```bash
python scripts/check_drift.py                                  # scénario neutre → code de sortie 0
python scripts/check_drift.py --price-increase 0.2 --extra-calls 1   # dérive → code de sortie 1
python scripts/check_drift.py --current nouveaux_clients.csv  # vos propres données
```

Le **PSI** (*Population Stability Index*) compare la distribution de chaque variable à celle
des données d'entraînement : < 0,10 stable · 0,10–0,25 modérée · > 0,25 significative.

## Résultats de référence

```
          roc_auc  accuracy  precision  recall     f1
logistic    0.834     0.769      0.689   0.634  0.660
gboost      0.807     0.746      0.655   0.596  0.624
```

💡 *Pourquoi le modèle simple gagne-t-il ?* C'est la question de l'exercice 3.3 du parcours.
Le détail (seuil optimal, impact financier, facteurs de churn, limites) est dans la
[fiche modèle](RAPPORT_MODELE.md), régénérée par `python scripts/report.py`.

## Structure

```
├── app.py                    # Tableau de bord Streamlit
├── scripts/
│   ├── train.py              # Entraînement + comparaison en ligne de commande
│   ├── report.py             # Génère la fiche modèle RAPPORT_MODELE.md
│   └── check_drift.py        # Alerte de dérive (code de sortie 1 si dérive)
├── src/churn/
│   ├── data.py               # Données synthétiques + validation
│   ├── features.py           # Préparation des variables
│   ├── model.py              # Entraînement, évaluation, seuil, importance
│   ├── api.py                # API REST FastAPI
│   └── monitoring.py         # PSI et simulation de dérive
├── tests/                    # test_churn, test_api, test_monitoring, test_app
├── .github/workflows/        # Intégration continue (GitHub Actions)
├── RAPPORT_MODELE.md         # 📋 Fiche modèle (générée)
├── PARCOURS.md               # 🧭 Parcours de compétences et exercices
├── CHANGELOG.md
└── requirements.txt
```

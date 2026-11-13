# 📉 Churn Prediction Dashboard

[![Tests](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml)

Un projet complet **et pédagogique** pour développer ses compétences en data science :
des données brutes jusqu'à un modèle **en production** (tableau de bord + API) qui prédit quels
clients risquent de partir (*churn*).

> ✅ **Statut : v1.0, projet terminé.** Données → modèle → tableau de bord → API → tests → CI → rapport.
>
> 🎯 **Objectif : apprendre en construisant.** Le fichier [`PARCOURS.md`](PARCOURS.md) est
> un parcours de compétences en 6 modules, avec des exercices progressifs et une checklist
> pour suivre votre progression. Le [`RAPPORT.md`](RAPPORT.md) montre le résultat final attendu.

## Ce que vous allez pratiquer

| Compétence | Où dans le code |
|---|---|
| Génération / validation de données | `src/churn/data.py` |
| Feature engineering & pipelines sans fuite de données | `src/churn/features.py` |
| Modélisation (régression logistique, gradient boosting) | `src/churn/model.py` |
| Évaluation : ROC-AUC, précision, rappel, seuil métier | `src/churn/model.py` |
| Interprétabilité (importance par permutation) | `src/churn/model.py` |
| Tableau de bord interactif | `app.py` (Streamlit + Plotly) |
| API REST avec validation des entrées | `src/churn/api.py` (FastAPI + Pydantic) |
| Tests automatisés (modèle, API, tableau de bord) | `tests/` (pytest) |
| Intégration continue | `.github/workflows/tests.yml` (GitHub Actions) |
| Rapport reproductible et analyse coût / bénéfice | `scripts/report.py` → `RAPPORT.md` |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py          # entraîne, compare et sauvegarde le meilleur modèle
streamlit run app.py             # tableau de bord sur http://localhost:8501
pytest -q                        # lance les 33 tests
python scripts/report.py         # régénère RAPPORT.md
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
python scripts/train.py                          # crée models/churn_model.joblib
uvicorn churn.api:app --app-dir src --reload     # documentation interactive : http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure_months": 3, "monthly_charges": 95, "support_calls": 4, "num_products": 1,
  "contract": "Mensuel", "payment_method": "Chèque", "internet_service": "Fibre",
  "tech_support": "Non", "senior": "Non"}'
# → {"churn_probability": 0.9699, "at_risk": true, "threshold": 0.5}
```

| Route | Rôle |
|---|---|
| `GET /health` | État du service, modèle chargé, seuil |
| `POST /predict` | Un client → probabilité + décision |
| `POST /predict/batch` | `{"customers": [...]}` : 1 à 1 000 clients |

Une valeur invalide (contrat inconnu, ancienneté négative…) renvoie une **erreur 422** explicite.
Le seuil se règle avec la variable d'environnement `CHURN_THRESHOLD` (ex. le seuil optimal du rapport).

## Résultats de référence

```
          roc_auc  accuracy  precision  recall     f1
logistic    0.834     0.769      0.689   0.634  0.660
gboost      0.807     0.746      0.655   0.596  0.624
```

Avec un coût de 500 € par client perdu et 50 € par action inutile, le seuil optimal est **0,10**
et le ciblage économise **25 %** par rapport à la meilleure stratégie sans modèle. Détails,
analyse de sensibilité et limites dans [`RAPPORT.md`](RAPPORT.md).

💡 *Pourquoi le modèle simple gagne-t-il ?* C'est la question de l'exercice 3.3 du parcours.

## Structure

```
├── app.py                  # Tableau de bord Streamlit
├── scripts/
│   ├── train.py            # Entraînement + comparaison en ligne de commande
│   └── report.py           # Génère RAPPORT.md avec des chiffres recalculés
├── src/churn/
│   ├── data.py             # Données synthétiques + validation
│   ├── features.py         # Préparation des variables
│   ├── model.py            # Entraînement, évaluation, seuil, importance, sauvegarde
│   └── api.py              # API REST FastAPI
├── tests/                  # 33 tests : données, modèle, persistance, API, tableau de bord, rapport
├── .github/workflows/      # Intégration continue (tests à chaque push)
├── PARCOURS.md             # 🧭 Parcours de compétences et exercices
├── RAPPORT.md              # 📄 Rapport final (généré)
├── pyproject.toml          # Configuration de pytest
└── requirements.txt
```

## Mettre en ligne le tableau de bord

1. Poussez le dépôt sur GitHub (public).
2. Sur [share.streamlit.io](https://share.streamlit.io), « New app » → choisissez ce dépôt, fichier `app.py`.
3. Ajoutez le lien obtenu en haut de ce README : c'est la vitrine de votre portfolio.

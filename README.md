# 📉 Churn Prediction — de la donnée à la production

[![CI](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/ci.yml)

Une série de **projets pédagogiques** pour développer ses compétences en data science
et en ingénierie : des données brutes jusqu'à un modèle servi par une API, conteneurisé
et testé automatiquement. Le fil rouge : prédire quels clients risquent de partir (*churn*).

> 🎯 **Objectif : apprendre en construisant.** Le fichier [`PARCOURS.md`](PARCOURS.md) contient
> les exercices progressifs de chaque projet et une checklist pour suivre votre progression.

## 🗺️ Feuille de route des projets

| # | Projet | Compétences | Statut |
|---|---|---|---|
| 1 | **Modèle + tableau de bord** | pandas, scikit-learn, métriques, seuil métier, Streamlit | ✅ Terminé |
| 2 | **Mise en production** | FastAPI, Pydantic, tests d'API, Docker, GitHub Actions | ✅ Terminé |
| 3 | Segmentation et plan de rétention | clustering, valeur vie client (CLV), ROI des campagnes | 🔜 Prochain |
| 4 | Surveillance du modèle | journalisation, *data drift*, réentraînement | 🔜 À venir |

Chaque projet est *terminé* quand : le code fonctionne, il est testé, la CI est verte,
la documentation est à jour et des exercices existent dans `PARCOURS.md`.

## Ce que vous allez pratiquer

| Compétence | Où dans le code |
|---|---|
| Génération / validation de données | `src/churn/data.py` |
| Feature engineering & pipelines sans fuite de données | `src/churn/features.py` |
| Modélisation (régression logistique, gradient boosting) | `src/churn/model.py` |
| Évaluation : ROC-AUC, précision, rappel, seuil métier | `src/churn/model.py` |
| Interprétabilité (importance par permutation) | `src/churn/model.py` |
| Tableau de bord interactif | `app.py` (Streamlit + Plotly) |
| API REST, validation des entrées | `api/main.py`, `api/schemas.py` (FastAPI + Pydantic) |
| Conteneurisation | `Dockerfile` |
| Intégration continue | `.github/workflows/ci.yml` (GitHub Actions) |
| Tests automatisés | `tests/` (pytest) |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py          # entraîne et compare les modèles
streamlit run app.py             # tableau de bord → http://localhost:8501
uvicorn api.main:app --reload    # API → http://localhost:8000/docs
pytest -q                        # lance les 27 tests
```

Les données sont **synthétiques** et générées au premier lancement dans `data/customers.csv`
(5 000 clients, ~35 % de churn). Aucune connexion ni donnée personnelle n'est nécessaire.

## Projet 1 — Le tableau de bord

| Onglet | Ce qu'il montre |
|---|---|
| **Vue d'ensemble** | KPIs, taux de churn par contrat, ancienneté, appels au support, paiement |
| **Performance** | ROC-AUC, matrice de confusion, compromis précision/rappel, **seuil optimal selon les coûts métier** |
| **Facteurs de churn** | Variables les plus influentes (importance par permutation) |
| **Clients à risque** | Liste triée par probabilité de départ, export CSV |
| **Prédire un client** | Formulaire → probabilité de churn d'un profil |

Dans la barre latérale : choix du modèle, activation des variables dérivées, seuil de décision.

### Résultats de référence

```
          roc_auc  accuracy  precision  recall     f1
logistic    0.834     0.769      0.689   0.634  0.660
gboost      0.807     0.746      0.655   0.596  0.624
```

💡 *Pourquoi le modèle simple gagne-t-il ?* C'est la question de l'exercice 3.3 du parcours.

## Projet 2 — L'API en production

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/health` | Sonde de vie (Docker, load balancer) |
| `GET` | `/model` | Type de modèle, variables, seuil, métriques de test |
| `POST` | `/predict` | Probabilité de churn d'un client |
| `POST` | `/predict/batch` | Jusqu'à 1 000 clients en une requête |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure_months": 3, "monthly_charges": 89.9, "support_calls": 4, "num_products": 1,
  "contract": "Mensuel", "payment_method": "Chèque", "internet_service": "Fibre",
  "tech_support": "Non", "senior": "Non"}'
# → {"churn_probability": 0.9676, "at_risk": true, "threshold": 0.5}
```

Une entrée invalide (contrat inconnu, ancienneté négative, champ manquant ou en trop) est rejetée
avec une erreur **422** qui désigne le champ fautif. `total_charges` est facultatif.

**Configuration** par variables d'environnement : `CHURN_THRESHOLD` (seuil, défaut 0.5) et
`CHURN_MODEL_PATH` (défaut `models/churn_model.joblib` ; le modèle est entraîné s'il n'existe pas).

**Avec Docker** (le modèle est entraîné pendant la construction de l'image) :

```bash
docker build -t churn-api .
docker run --rm -p 8000:8000 -e CHURN_THRESHOLD=0.3 churn-api
```

**Intégration continue** : à chaque push, GitHub Actions lance les tests sous Python 3.11, 3.12 et 3.13,
puis construit l'image Docker et interroge l'API pour vérifier qu'elle répond.

## Structure

```
├── app.py                    # Projet 1 — Tableau de bord Streamlit
├── api/
│   ├── main.py               # Projet 2 — API FastAPI (routes, chargement du modèle)
│   └── schemas.py            #            Contrats d'entrée / sortie (Pydantic)
├── scripts/train.py          # Entraînement + comparaison en ligne de commande
├── src/churn/
│   ├── data.py               # Données synthétiques + validation
│   ├── features.py           # Préparation des variables
│   └── model.py              # Entraînement, évaluation, seuil, importance
├── tests/
│   ├── test_churn.py         # Tests du modèle
│   └── test_api.py           # Tests de l'API
├── Dockerfile                # Image de production de l'API
├── .github/workflows/ci.yml  # Intégration continue
├── PARCOURS.md               # 🧭 Exercices de chaque projet
├── requirements-api.txt      # Dépendances pour servir le modèle
└── requirements.txt          # Tout le projet (API + tableau de bord + tests)
```

# 📉 Churn Prediction Dashboard

Un projet complet **et pédagogique** pour développer ses compétences en data science :
des données brutes jusqu'à un tableau de bord interactif qui prédit quels clients risquent
de partir (*churn*).

> 🎯 **Objectif : apprendre en construisant.** Le fichier [`PARCOURS.md`](PARCOURS.md) est
> un parcours de compétences en 6 modules, avec des exercices progressifs et une checklist
> pour suivre votre progression.

## 🗺️ Les projets

Le dépôt grandit projet après projet. Chacun est **terminé et testé**, et chacun a ses
propres exercices.

| # | Projet | Compétences | Exercices | État |
|---|---|---|---|---|
| 1 | **Tableau de bord de prédiction du churn** | pandas, scikit-learn, évaluation, Streamlit | [`PARCOURS.md`](PARCOURS.md) | ✅ terminé |
| 2 | **API de prédiction** | FastAPI, Pydantic, tests d'API, Docker | [`api/README.md`](api/README.md) | ✅ terminé |
| 3 | Surveillance du modèle | journal des prédictions, *data drift*, alertes | — | 🔜 prochain |

## Ce que vous allez pratiquer

| Compétence | Où dans le code |
|---|---|
| Génération / validation de données | `src/churn/data.py` |
| Feature engineering & pipelines sans fuite de données | `src/churn/features.py` |
| Modélisation (régression logistique, gradient boosting) | `src/churn/model.py` |
| Évaluation : ROC-AUC, précision, rappel, seuil métier | `src/churn/model.py` |
| Interprétabilité (importance par permutation) | `src/churn/model.py` |
| Tableau de bord interactif | `app.py` (Streamlit + Plotly) |
| Tests automatisés | `tests/test_churn.py` (pytest) |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py          # entraîne et compare les modèles
streamlit run app.py             # Projet 1 : tableau de bord sur http://localhost:8501
uvicorn api.main:app --reload    # Projet 2 : API sur http://localhost:8000/docs
pytest -q                        # lance les tests (33)
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

## Résultats de référence

```
          roc_auc  accuracy  precision  recall     f1
logistic    0.834     0.769      0.689   0.634  0.660
gboost      0.807     0.746      0.655   0.596  0.624
```

💡 *Pourquoi le modèle simple gagne-t-il ?* C'est la question de l'exercice 3.3 du parcours.

## Structure

```
├── app.py                  # Projet 1 — tableau de bord Streamlit
├── scripts/train.py        # Entraînement + comparaison en ligne de commande
├── src/churn/
│   ├── data.py             # Données synthétiques + validation
│   ├── features.py         # Préparation des variables
│   └── model.py            # Entraînement, évaluation, seuil, importance
├── api/                    # Projet 2 — API FastAPI
│   ├── main.py             # Routes /health, /model, /predict, /predict/batch
│   ├── schemas.py          # Validation des entrées et sorties (Pydantic)
│   ├── client_example.py   # Appeler l'API depuis Python
│   └── README.md           # 🧭 Guide et exercices du Projet 2
├── tests/
│   ├── test_churn.py       # Tests du Projet 1
│   └── test_api.py         # Tests du Projet 2
├── Dockerfile              # Image de l'API (requirements-api.txt)
├── PARCOURS.md             # 🧭 Parcours de compétences et exercices du Projet 1
└── requirements.txt
```

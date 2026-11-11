# 📉 Churn Prediction Dashboard

[![Tests](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml)

Un projet complet **et pédagogique** pour développer ses compétences en data science :
des données brutes jusqu'à un tableau de bord interactif qui prédit quels clients risquent
de partir (*churn*).

> 🎯 **Objectif : apprendre en construisant.**
>
> 1. [`PARCOURS.md`](PARCOURS.md) — 6 modules, 31 exercices progressifs et une checklist.
> 2. [`CORRIGES.md`](CORRIGES.md) — un indice puis une solution testée pour chaque exercice
>    (à ouvrir **après** avoir cherché).
> 3. [`JOURNAL.md`](JOURNAL.md) — votre carnet de bord : mesures avant / après et ce que vous avez compris.

## Ce que vous allez pratiquer

| Compétence | Où dans le code |
|---|---|
| Génération / validation de données | `src/churn/data.py` |
| Feature engineering & pipelines sans fuite de données | `src/churn/features.py` |
| Modélisation (régression logistique, gradient boosting) | `src/churn/model.py` |
| Évaluation : ROC-AUC, précision, rappel, seuil métier | `src/churn/model.py` |
| Interprétabilité (importance par permutation) | `src/churn/model.py` |
| Tableau de bord interactif | `app.py` (Streamlit + Plotly) |
| Tests automatisés (modèle **et** tableau de bord) | `tests/` (pytest + `streamlit.testing`) |
| Intégration continue | `.github/workflows/tests.yml` (GitHub Actions) |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py          # entraîne et compare les modèles
streamlit run app.py             # ouvre le tableau de bord sur http://localhost:8501
pytest -q                        # lance les tests (≈ 40 s, dont le tableau de bord)
```

À chaque `git push`, GitHub Actions réinstalle le projet, entraîne les modèles et relance tous les
tests : si un exercice casse quelque chose, vous le saurez tout de suite.

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
├── app.py                  # Tableau de bord Streamlit
├── scripts/train.py        # Entraînement + comparaison en ligne de commande
├── src/churn/
│   ├── data.py             # Données synthétiques + validation
│   ├── features.py         # Préparation des variables
│   └── model.py            # Entraînement, évaluation, seuil, importance
├── tests/
│   ├── test_churn.py       # Tests des données, variables et modèles
│   └── test_app.py         # Tests du tableau de bord (sans navigateur)
├── .github/workflows/      # Intégration continue
├── PARCOURS.md             # 🧭 Parcours de compétences et exercices
├── CORRIGES.md             # ✅ Indices et solutions testées
├── JOURNAL.md              # 📓 Votre journal de progression
└── requirements.txt
```

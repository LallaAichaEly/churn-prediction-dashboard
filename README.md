# 📉 Churn Prediction Dashboard

[![Tests](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml)

Un projet complet **et pédagogique** pour développer ses compétences en data science :
des données brutes jusqu'à un tableau de bord interactif et une API qui prédisent quels
clients risquent de partir (*churn*).

> 🎯 **Objectif : apprendre en construisant.**
>
> 1. [`PARCOURS.md`](PARCOURS.md) — 7 modules, 37 exercices progressifs et une checklist ;
>    `python scripts/progression.py` vous dit où vous en êtes et quoi faire ensuite.
> 2. [`CORRIGES.md`](CORRIGES.md) — un indice puis une solution pour chaque exercice
>    (à ouvrir **après** avoir cherché), complétés par des scripts exécutables dans `solutions/`.
> 3. [`JOURNAL.md`](JOURNAL.md) — votre carnet de bord : mesures avant / après et ce que vous avez compris.
> 4. [`defis/`](defis/README.md) — ⚡ des fonctions courtes à compléter, corrigées instantanément
>    par `python scripts/defis.py` (✅ / ❌ avec l'explication de l'erreur).
>
> 👋 **Vous débutez ?** Commencez par la série 00 des défis (`defis/serie_00.py`) : les bases de Python
> et de pandas, avec un message clair à chaque erreur. Puis le module 1 du parcours.
>
> 🛠️ **Vous voulez développer, mais ne vous sentez pas encore compétent·e ?** Après la série 00, la
> série 06 (`defis/serie_06.py`, défis 38 à 43) entraîne les gestes du métier : lire une erreur, écrire
> un test qui attrape les bugs, gérer une saisie invalide, refactoriser sans rien casser, écrire une
> ligne de commande, retrouver le commit fautif comme `git bisect`. La série 07 (`defis/serie_07.py`,
> défis 44 à 49) passe ensuite du script au code réutilisable : dates, export Excel, `set` contre liste,
> générateurs, classes, décorateurs. La série 08 (`defis/serie_08.py`, défis 50 à 55) apprend à construire
> la table du modèle à partir de vraies sources : regex sur des notes, jointures, JSON d'une API,
> vectorisation, mise à jour idempotente, variables tirées d'un journal d'événements sans fuite temporelle.
> La série 09 (`defis/serie_09.py`, défis 56 à 61) prépare l'entretien technique : les classiques
> d'algorithmique (table de hachage, pile, fenêtre glissante, intervalles, tas, parcours en largeur),
> avec un correcteur qui refuse les solutions justes mais trop lentes ou trop gourmandes en mémoire.
> La série 10 (`defis/serie_10.py`, défis 62 à 67) entraîne le SQL de tous les jours : NULL,
> anti-jointure, `LEFT JOIN` (ON ou WHERE ?), explosion des jointures, fonctions de fenêtre, îlots et trous.

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
| Surveillance en production : dérive des données (PSI) | `src/churn/monitoring.py` |
| Tests automatisés (modèle, API, tableau de bord, corrigés) | `tests/` (pytest + `streamlit.testing`) |
| Intégration continue | `.github/workflows/tests.yml` (GitHub Actions) |
| SQL (GROUP BY, fonctions de fenêtre), intervalles de confiance, test A/B, courbe de survie | `defis/serie_05.py` |
| Bases de Python et de pandas, débogage (remise à niveau) | `defis/serie_00.py` |
| Gestes du développeur : lire une traceback, écrire un test, exceptions, refactoring, argparse, bissection | `defis/serie_06.py` |
| Code réutilisable : `datetime`, encodages et CSV Excel, complexité (`set`), générateurs, classes, décorateurs | `defis/serie_07.py` |
| Assembler les données : regex, jointures, JSON d'API, vectorisation, upsert idempotent, fuite temporelle | `defis/serie_08.py` |
| Algorithmique d'entretien : complexité, dictionnaire, pile, deux pointeurs, intervalles, tas (`heapq`), BFS (`deque`) | `defis/serie_09.py` |
| SQL pour l'analyse : NULL, anti-jointure, `LEFT JOIN`, fan-out, CTE, `LAG` et cumuls, îlots et trous | `defis/serie_10.py` (SQLite) |
| Entraînement corrigé automatiquement | `defis/serie_00.py` à `defis/serie_10.py` + `scripts/defis.py` |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py          # entraîne et compare les modèles
streamlit run app.py             # tableau de bord sur http://localhost:8501
uvicorn api:app --reload         # API sur http://localhost:8000/docs
pytest -q                        # lance les tests (≈ 1 min, dont le tableau de bord)
python scripts/progression.py    # 🧭 votre progression + le prochain exercice, avec un indice
python scripts/defis.py          # ⚡ vos défis auto-corrigés : ✅ / ❌ / ⬜
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
| **Surveillance** | Simule les clients de « dans six mois » (hausse des prix, mix de contrats, concurrent) : PSI par variable, churn prédit contre réel |

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
├── scripts/progression.py  # 🧭 Coach : vérifie vos exercices, propose la suite
├── scripts/defis.py        # ⚡ Correction automatique des défis
├── defis/
│   ├── serie_00.py         # 👋 Série 00 — remise à niveau Python & pandas (8 défis, à faire en premier)
│   ├── serie_01.py         # ✍️ Série 01 — les fondamentaux (10 fonctions à compléter)
│   ├── serie_02.py         # ✍️ Série 02 — explorer les données (4 fonctions)
│   ├── serie_03.py         # ✍️ Série 03 — évaluer, expliquer, surveiller (5 fonctions)
│   ├── serie_04.py         # ✍️ Série 04 — modéliser et industrialiser (5 fonctions)
│   ├── serie_05.py         # ✍️ Série 05 — prouver l'impact : SQL, incertitude, A/B, survie (5 fonctions)
│   ├── serie_06.py         # 🛠️ Série 06 — développer comme un·e pro : erreurs, tests, refactoring, CLI (6 défis)
│   ├── serie_07.py         # 🧱 Série 07 — code réutilisable : dates, Excel, set, yield, classes, décorateurs (6 défis)
│   ├── serie_08.py         # 🔗 Série 08 — assembler les données : regex, jointures, JSON, upsert, événements (6 défis)
│   ├── serie_09.py         # 🧠 Série 09 — algorithmes d'entretien : dict, pile, fenêtre, intervalles, tas, BFS (6 défis)
│   ├── serie_10.py         # 🗄️ Série 10 — SQL pour l'analyse : NULL, jointures, fenêtres, îlots (6 défis)
│   ├── donnees_sql.py      # La base SQLite de la série 10 (clients, factures, appels) + essayer()
│   ├── corrige_serie_XX.py # Corrigés commentés (après avoir essayé !)
│   └── checks_serie_XX.py  # Vérifications automatiques
├── src/churn/
│   ├── data.py             # Données synthétiques + validation
│   ├── features.py         # Préparation des variables
│   ├── model.py            # Entraînement, évaluation, seuil, importance
│   └── monitoring.py       # Surveillance : PSI, rapport de dérive, lots simulés
├── solutions/              # Corrigés exécutables des modules 1 à 5 et 7
├── tests/
│   ├── test_churn.py       # Données, variables et modèles
│   ├── test_api.py         # API
│   ├── test_app.py         # Tableau de bord (sans navigateur)
│   ├── test_monitoring.py  # Surveillance de la dérive
│   ├── test_progression.py # Le coach juge correctement les exercices
│   ├── test_defis.py       # Le corrigé des défis réussit, les erreurs classiques sont détectées
│   └── test_solutions.py   # Les affirmations des corrigés restent vraies
├── .github/workflows/      # Intégration continue (pytest à chaque push)
├── PARCOURS.md             # 🧭 Parcours de compétences et exercices
├── CORRIGES.md             # ✅ Indices, solutions et exemple de rapport final
├── JOURNAL.md              # 📓 Votre journal de progression
└── requirements.txt
```

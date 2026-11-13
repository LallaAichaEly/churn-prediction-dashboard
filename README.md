# 📉 Churn Prediction Dashboard

[![Tests](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/LallaAichaEly/churn-prediction-dashboard/actions/workflows/tests.yml)

Un projet complet **et pédagogique** pour développer ses compétences en data science :
des données brutes jusqu'à un tableau de bord interactif et une API qui prédisent quels
clients risquent de partir (*churn*).

> 🎯 **Objectif : apprendre en construisant.** Inutile de tout lire avant de commencer : faites la
> première séance ci-dessous, puis avancez d'un défi à la fois. Les scripts vous disent si c'est juste
> et, sinon, pourquoi.

## 🚀 Par où commencer — première séance (20 minutes)

1. **Installer** (5 min) : suivez le [démarrage rapide](#démarrage-rapide), puis lancez
   `python scripts/defis.py`. Presque tout est ⬜ « à faire », c'est normal. Le défi 34 commence en ❌
   parce que c'est un code à déboguer.
2. **Réussir un premier défi** (10 min) : ouvrez `defis/serie_00.py`, lisez la docstring de
   `taux_en_pourcentage` (défi 30), remplacez `raise NotImplementedError` par votre code, puis lancez
   `python scripts/defis.py --defi 30`. Corrigez et relancez jusqu'au ✅.
3. **Noter** (5 min) : écrivez une fiche dans [`JOURNAL.md`](JOURNAL.md) avec ce qui vous a bloqué·e
   et ce que vous avez compris.

Ensuite, faites **un défi par séance**, dans l'ordre qui correspond à votre objectif :

| Votre objectif | Commencez par | Puis |
|---|---|---|
| Je débute, ou je ne me sens pas encore « à niveau » | série 00 (défis 30 à 37) | série 01, puis le module 1 de [`PARCOURS.md`](PARCOURS.md) |
| Je veux développer : écrire du code fiable, en équipe | série 00, puis l'[atelier Git](atelier_git/README.md) | séries 06 → 11 (tests) → 07 → 08 → 12 (API web) → 13 (sécurité) → 14 (production) → 15 (base de données), puis le [projet guidé](projets/README.md) |
| Je prépare un entretien technique | séries 00 et 07 | série 09 (algorithmique), puis séries 05 et 10 (SQL) |
| Je vise un poste en data science | séries 00 et 01 | les modules 1 à 7 de [`PARCOURS.md`](PARCOURS.md), avec les séries 02 à 05 |

Pour aller plus loin :

- [`defis/README.md`](defis/README.md) détaille les 16 séries de défis.
- [`PARCOURS.md`](PARCOURS.md) propose des exercices en 7 modules sur le vrai projet.
  `python scripts/progression.py` suit votre avancement.
- [`CORRIGES.md`](CORRIGES.md) donne un indice puis la solution de chaque exercice. Ouvrez-le
  **après** avoir cherché.
- [`projets/README.md`](projets/README.md) décrit le projet guidé, corrigé par `python scripts/projet.py` :
  un vrai programme à construire en 7 étapes, du CSV imparfait au plan de rétention.
- [`atelier_git/README.md`](atelier_git/README.md) décrit l'atelier Git, corrigé par
  `python scripts/atelier_git.py` : 8 situations d'équipe à résoudre dans un bac à sable, du premier
  commit propre au conflit de fusion, au `git bisect` et à la branche prête pour la revue.

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
| Tester comme un·e pro : valeurs aux frontières et `parametrize`, `pytest.approx`, `pytest.raises`, faux objets (injection de dépendance), fichiers dans `tmp_path`, tests par propriétés | `defis/serie_11.py` (vos tests, jugés par mutation) |
| Construire une API web : routes et codes HTTP (200, 201, 204, 401, 403, 404, 409, 422, 503), paramètres de requête, modèles Pydantic, PATCH partiel, clé d'API avec `Depends`, idempotence (`Idempotency-Key`) | `defis/serie_12.py` (FastAPI, appelée par `TestClient`) |
| Écrire du code sûr : requêtes SQL paramétrées (injection SQL), secrets masqués dans les journaux, chemins d'export sans traversée de répertoire, export CSV sans injection de formules, modèle pickle chargé seulement si son empreinte SHA-256 est bonne, lien signé avec HMAC (`compare_digest`, expiration) | `defis/serie_13.py` (le correcteur joue l'attaquant) |
| Tenir en production : configuration par variables d'environnement, journalisation avec `logging` (niveaux, traceback, pas de `print`), reprises avec attente exponentielle, fuseaux horaires et changement d'heure (`zoneinfo`), écriture atomique d'un fichier (`os.replace`), disjoncteur (machine à états, horloge injectée) | `defis/serie_14.py` (le correcteur simule une nuit de pannes) |
| Enregistrer sans rien perdre : transactions (tout ou rien, `rollback`), pagination par curseur, requêtes N+1, migrations versionnées (`PRAGMA user_version`), verrou optimiste (`version`, `rowcount`), file de tâches à plusieurs travailleurs (bail, `RETURNING`) | `defis/serie_15.py` (SQLite ; le correcteur fait écrire un collègue entre votre SELECT et votre UPDATE) |
| Entraînement corrigé automatiquement | `defis/serie_00.py` à `defis/serie_15.py` + `scripts/defis.py` |
| Construire un programme complet : modules, CSV imparfaits, règles métier, budget, CLI, tests | `projets/retention/` (projet 2, 7 étapes) + `scripts/projet.py` |
| Git en équipe : commit propre et `.gitignore`, branches, `stash`, conflits, `revert`, `bisect`, `reflog`, `rebase` et `push --force-with-lease` | `atelier_git/` (8 scénarios dans un bac à sable) + `scripts/atelier_git.py` |

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
python scripts/projet.py         # 🏗️ votre projet guidé (projet 2), étape par étape
python scripts/atelier_git.py    # 🌿 git en équipe : 8 situations à résoudre dans un bac à sable
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
├── scripts/projet.py       # 🏗️ Correction automatique du projet guidé
├── scripts/atelier_git.py  # 🌿 Atelier Git : prépare et corrige les 8 scénarios
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
│   ├── serie_11.py         # 🧪 Série 11 — tester comme un·e pro : frontières, approx, raises, faux objets, tmp_path, propriétés (6 défis)
│   ├── serie_12.py         # 🌐 Série 12 — construire une API web : lire, chercher, créer, modifier, protéger, idempotence (6 défis)
│   ├── serie_13.py         # 🔐 Série 13 — écrire du code sûr : SQL, secrets, chemins, formules, pickle, liens signés (6 défis)
│   ├── serie_14.py         # 🚦 Série 14 — tenir en production : configuration, journaux, reprises, fuseaux, écriture atomique, disjoncteur (6 défis)
│   ├── production.py       # Outils communs de la série 14 (Config, ErreurTemporaire, HorlogeFactice…)
│   ├── serie_15.py         # 💾 Série 15 — enregistrer sans rien perdre : transaction, pages, N+1, migrations, verrou optimiste, file de tâches (6 défis)
│   ├── persistance.py      # Outils communs de la série 15 (base_exemple, deux_connexions, afficher, ConflitDeVersion…)
│   ├── donnees_sql.py      # La base SQLite de la série 10 (clients, factures, appels) + essayer()
│   ├── corrige_serie_XX.py # Corrigés commentés (après avoir essayé !)
│   └── checks_serie_XX.py  # Vérifications automatiques
├── atelier_git/            # 🌿 Atelier Git (voir atelier_git/README.md)
│   ├── scenarios.py        # Les 8 situations : mission, indices, solution, correction
│   └── atelier.py, outils.py  # Moteur : fabrique les dépôts dans ~/atelier-git, puis les inspecte
├── projets/                # 🏗️ Projets guidés (voir projets/README.md)
│   ├── retention/          # Projet 2 — VOTRE code : 7 étapes à compléter (clients.py … cli.py, mes_tests.py)
│   ├── retention_corrige/  # Projet 2 terminé : le corrigé, et la référence du correcteur
│   ├── checks_retention.py # Vérifications des 7 étapes
│   ├── verif.py            # Moteur de correction des projets
│   └── exemple_clients.csv # 200 clients scorés par le modèle (format de l'export du tableau de bord)
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
│   ├── test_projets.py     # Le projet guidé : corrigé, programme terminé, erreurs classiques détectées
│   ├── test_atelier_git.py # L'atelier Git : chaque solution réussit, les erreurs classiques sont expliquées
│   └── test_solutions.py   # Les affirmations des corrigés restent vraies
├── .github/workflows/      # Intégration continue (pytest à chaque push)
├── PARCOURS.md             # 🧭 Parcours de compétences et exercices
├── CORRIGES.md             # ✅ Indices, solutions et exemple de rapport final
├── JOURNAL.md              # 📓 Votre journal de progression
└── requirements.txt
```

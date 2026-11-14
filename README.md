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
| Je veux développer : écrire du code fiable, en équipe | série 00, puis l'[atelier Git](atelier_git/README.md) | séries 06 → 11 (tests) → 07 → 08 → 12 (API web) → 13 (sécurité) → 14 (production) → 15 (base de données) → 16 (asyncio) → 17 (concevoir ses types) → 18 (rendre le code rapide) → 19 (faire évoluer du code existant) → 20 (le texte et l'Unicode) → 23 (recoder `fit` avec NumPy) → 24 (orchestrer un traitement de nuit), puis les [projets guidés](projets/README.md) 2, 3 et 4 (le 4 assemble les séries 12 à 15 en un vrai service web) |
| Je prépare un entretien technique | séries 00 et 07 | série 09 (algorithmique), puis séries 05 et 10 (SQL), la série 23 (« codez une régression logistique sans scikit-learn ») et la série 24 (graphes : « course schedule », détecter un cycle) |
| Je vise un poste en data science | séries 00 et 01 | les modules 1 à 7 de [`PARCOURS.md`](PARCOURS.md), avec les séries 02 à 05, la série 21 (dates, cohortes, validation dans le temps, revenu mensuel) la série 22 (écrire ses propres briques scikit-learn) et la série 23 (recoder la régression logistique et l'arbre de décision), puis le [projet 3](projets/README.md#projet-3--mesurer-leffet-de-la-campagne) (test A/B de bout en bout) et le [projet 5](projets/README.md#projet-5--réentraîner-le-modèle-sans-casser-la-production) (réentraîner chaque mois : champion, challenger, registre des modèles) |

Pour aller plus loin :

- [`defis/README.md`](defis/README.md) détaille les 25 séries de défis (00 à 24).
- [`PARCOURS.md`](PARCOURS.md) propose des exercices en 7 modules sur le vrai projet.
  `python scripts/progression.py` suit votre avancement.
- [`CORRIGES.md`](CORRIGES.md) donne un indice puis la solution de chaque exercice. Ouvrez-le
  **après** avoir cherché.
- [`projets/README.md`](projets/README.md) décrit les projets guidés, corrigés par `python scripts/projet.py` :
  quatre vrais programmes à construire en 7 étapes chacun. Le projet 2 va du CSV imparfait au plan de
  rétention ; le projet 3 mesure, avec un groupe témoin et une base SQLite, si ce plan a vraiment évité
  des départs ; le projet 4 est le service web que les conseillers utilisent pour appeler les clients :
  configuration, clés d'accès, pagination, deux écritures simultanées, et des tests jugés sur 8 versions
  boguées de l'API ; le projet 5 réentraîne le modèle chaque mois sans casser la production : lots abîmés
  refusés, validation dans le temps, champion contre challenger sur le même mois, registre des versions
  vérifié par SHA-256, et retour en arrière en une commande.
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
| Faire plusieurs choses à la fois avec `asyncio` : `gather`, délai maximal et annulation (`wait_for`), limite de concurrence (`Semaphore`), situation de compétition (`Lock`, double vérification), le premier qui répond (`as_completed`), producteur et travailleurs (`Queue` bornée, `TaskGroup`) | `defis/serie_16.py` (horloge virtuelle : le correcteur mesure vos durées et voit les interblocages) |
| Concevoir ses propres types : `@dataclass(frozen=True)` et `__post_init__` (TypeError ou ValueError), `Enum` avec `@classmethod` et `@property`, objet valeur `Montant` en centimes (`Decimal`, `NotImplemented`, `__radd__` pour `sum`, `@total_ordering`), gestionnaire de contexte (`@contextmanager`, `try / except / else / finally`), protocole des collections (`__len__`, `__contains__`, `__getitem__`, `__iter__`), surcharge d'opérateurs (`&`, `|`, `~`, `__call__`, `__bool__`) | `defis/serie_17.py` (le correcteur reçoit vos classes et s'en sert comme un collègue) |
| Rendre le code rapide, preuves à l'appui : chronométrer (`perf_counter`, répéter, garder le minimum), profiler (`cProfile`, `pstats`, tottime contre cumtime), recherche dichotomique (`bisect`, liste triée une fois), cache borné et périmé (`OrderedDict`, LRU, durée de vie, décorateur à paramètres), lire un gros fichier en flux (`csv.DictReader`, `chunksize`, `tracemalloc`), optimiser sans rien changer (`Counter`, `set`, calcul sorti de la boucle, comparaison à l'ancienne version) | `defis/serie_18.py` (le correcteur compte vos comparaisons et vos lectures, et arrête une boucle dans une boucle) |
| Faire évoluer du code existant sans rien casser : test de caractérisation (photo du comportement, `copy.deepcopy`), paramètre déprécié (`warnings.warn`, `DeprecationWarning`, `stacklevel=2`), cœur pur extrait d'un script (ni fichier, ni `print`, ni `date.today()`, ni état global), lecture de deux formats de données (versions explicites, lecteur tolérant), patron adaptateur (traduire les erreurs, `raise … from exc`), nouvelle version testée dans l'ombre de l'ancienne (`functools.wraps`, journal des différences) | `defis/serie_19.py` (le correcteur compare votre code au vieux code sur des centaines de cas, et vérifie que l'utilisateur ne voit jamais la différence) |
| Manipuler du texte sans surprises : normalisation Unicode (NFC, NFKC, NFD), `casefold` contre `lower`, catégories de caractères (`unicodedata.category`, caractères invisibles « Cf »), clé de recherche sans accents (`str.maketrans`, `translate`), valider une saisie en liste blanche (`isspace`, le piège de `isdigit` et de `\d`), coût d'un SMS (alphabet GSM-7 contre UCS-2, unités UTF-16, segments), réparer un texte mal décodé (mojibake, `encode`/`decode`, `UnicodeError`), couper sans abîmer un graphème (accents combinants, ZWJ, drapeaux) | `defis/serie_20.py` (le correcteur envoie des textes avec ce que l'œil ne voit pas, et arrête une boucle sans fin) |
| Le temps dans les données : lire des dates françaises (`pd.to_datetime(format=...)`, ni `dayfirst` ni `errors="coerce"`), raisonner en mois (`Period`, `dt.to_period`, `period_range`), calendrier sans trou (`reindex(fill_value=0)`), stock et flux de clients et vrai taux de churn mensuel (NaT, dénominateur), table de rétention par cohorte (le triangle : NaN plutôt que 0), validation « walk-forward » sans fuite (délai d'étiquette, positions et non étiquettes), pont du revenu mensuel (MRR : `pivot_table`, `shift` sur les colonnes, réactivations) | `defis/serie_21.py` (le correcteur rejoue les versions boguées classiques pour dire laquelle vous avez écrite) |
| Écrire ses propres briques scikit-learn : le contrat fit / transform (état appris en `_`, `return self`, `check_is_fitted`), des paramètres qu'on peut régler (`get_params`, `set_params`, `clone`, vérifier dans `fit` et pas dans `__init__`), un scoreur métier en euros pour `cross_val_score` et `GridSearchCV` (`classes_` et colonnes de `predict_proba`, signe moins), l'encodage par la cible lissé puis hors pli (`fit_transform` ≠ `fit` puis `transform`), un méta-estimateur qui choisit son seuil sur des probabilités hors pli (`cross_val_predict`, `clone`, `ClassifierMixin` à gauche) | `defis/serie_22.py` (le correcteur branche vos briques dans un Pipeline, une validation croisée et un GridSearchCV) |
| Ouvrir la boîte noire : recoder la régression logistique et l'arbre de décision avec NumPy. Sigmoïde stable (débordement, `exp(-|z|)`, `np.where` qui calcule ses deux branches), perte logistique à partir des scores (`np.logaddexp`, pourquoi pas `np.clip`), gradient avec pénalité L2 vérifié par différences finies (`X.T @ erreur`, le « gradient check »), descente de gradient et critère d'arrêt (les mêmes coefficients que `LogisticRegression(C=1/(n·alpha))`), impureté de Gini et meilleure coupure en O(n log n) (`np.argsort`, `np.cumsum`), arbre récursif (conditions d'arrêt, le « ou exclusif », `x <= seuil`) | `defis/serie_23.py` (le correcteur transforme chaque RuntimeWarning en erreur, et compare vos coefficients et vos arbres à ceux de scikit-learn, nœud par nœud) |
| Orchestrer un traitement de nuit (un graphe de dépendances, comme Airflow ou `make`) : tri topologique de Kahn avec un tas (`heapq`, le premier prêt par ordre alphabétique), tâches à relancer en aval (parcours en largeur, `deque`, ensemble « déjà vu »), trouver un cycle (parcours en profondeur à trois états, pile explicite au lieu de la récursion), exécuter sans qu'un échec bloque ce qui n'en dépend pas (`f(**résultats)`, `Exception` et pas `BaseException`, tout valider avant de lancer), chemin critique (programmation dynamique dans l'ordre topologique), planning sur k travailleurs (simulation par événements, trois tas) | `defis/serie_24.py` (le correcteur compte les comparaisons et les lignes exécutées : un `min` dans une boucle, une exploration exponentielle des losanges ou un temps qui avance d'une unité se voient, quelle que soit la machine) |
| Entraînement corrigé automatiquement | `defis/serie_00.py` à `defis/serie_24.py` + `scripts/defis.py` |
| Construire un programme complet : modules, CSV imparfaits, règles métier, budget, CLI, tests | `projets/retention/` (projet 2, 7 étapes) + `scripts/projet.py` |
| Mesurer l'effet d'une action (test A/B) : groupe témoin reproductible (`hashlib`), base SQLite (transactions, UPSERT, clés étrangères, `GROUP BY` + `LEFT JOIN`), intervalle de confiance, CLI à sous-commandes | `projets/campagne/` (projet 3, 7 étapes) + `scripts/projet.py --projet 3` |
| Livrer un service web utilisé par une équipe : configuration par l'environnement sans fuite de secret, répartition équitable sans requêtes N+1, pagination par curseur, verrou optimiste et transaction sur deux tables, FastAPI (`Depends`, une connexion par requête, clés comparées avec `compare_digest`, 401/403/404/409/422), CLI qui démarre le serveur, tests d'API avec `TestClient` | `projets/service/` (projet 4, 7 étapes) + `scripts/projet.py --projet 4` |
| Réentraîner un modèle sans casser la production : valider un lot avant d'entraîner, validation dans le temps (`pd.Period`), AUC et coût en euros sur le même mois pour le champion et le challenger, règle de promotion (comparer des flottants), registre de modèles (écriture atomique, SHA-256 vérifiée avant `joblib.load`, retour en arrière) | `projets/reentrainement/` (projet 5, 7 étapes) + `scripts/projet.py --projet 5` |
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
python scripts/projet.py         # 🏗️ vos projets guidés (projets 2 à 5), étape par étape
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
├── scripts/projet.py       # 🏗️ Correction automatique des projets guidés (--projet 2, 3, 4 ou 5)
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
│   ├── serie_16.py         # ⚡ Série 16 — asyncio : gather, délais, Semaphore, Lock, premier qui répond, Queue et travailleurs (6 défis)
│   ├── concurrence.py      # Outils communs de la série 16 (faux services, boucle à horloge virtuelle, essayer…)
│   ├── serie_17.py         # 🧩 Série 17 — concevoir ses types : dataclass, Enum, Montant, with, collection, opérateurs (6 défis)
│   ├── conception.py       # Outils communs de la série 17 (fausse passerelle SMS, données d'exemple)
│   ├── serie_18.py         # ⏱️ Série 18 — rendre le code rapide : chronométrer, profiler, bisect, cache LRU, lecture en flux, optimiser (6 défis)
│   ├── performance.py      # Outils communs de la série 18 (programme à profiler, factures, version lente à accélérer…)
│   ├── serie_19.py         # 🏗️ Série 19 — faire évoluer du code existant : figer, déprécier, extraire, migrer, adapter, comparer (6 défis)
│   ├── heritage.py         # Le « vieux code » de la série 19 (score de 2022, script de nuit, ancien et nouvel opérateur SMS)
│   ├── serie_20.py         # 🔤 Série 20 — le texte sans surprises : comparer, chercher, normaliser, compter, réparer, couper (6 défis)
│   ├── textes.py           # Outils communs de la série 20 (alphabet des SMS, noms, numéros et export abîmé d'exemple)
│   ├── serie_21.py         # 📅 Série 21 — le temps qui passe : dates, mois, mouvements, cohortes, validation et revenu (6 défis)
│   ├── abonnements.py      # Outils communs de la série 21 (abonnements et factures d'exemple, générateurs)
│   ├── serie_22.py         # 🧰 Série 22 — écrire ses propres briques scikit-learn : imputer, régler, noter, encoder, décider (6 défis)
│   ├── briques.py          # Outils communs de la série 22 (12 clients d'exemple, générateur à longue traîne de villes)
│   ├── serie_23.py         # 🔓 Série 23 — ouvrir la boîte noire : sigmoïde, perte, gradient, descente, coupure de Gini, arbre (6 défis)
│   ├── maison.py           # Outils communs de la série 23 (12 clients d'exemple, générateur, gradient_numerique, afficher_arbre)
│   ├── serie_24.py         # 🌙 Série 24 — orchestrer le traitement de nuit : ordre, relances, cycle, exécution, chemin critique, planning (6 défis)
│   ├── nuit.py             # Outils communs de la série 24 (le traitement de nuit, ses durées, générateur de graphes, afficher_planning)
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
│   ├── campagne/           # Projet 3 — VOTRE code : 7 étapes (tirage.py … cli.py, mes_tests.py)
│   ├── campagne_corrige/   # Projet 3 terminé : le corrigé, et la référence du correcteur
│   ├── checks_campagne.py  # Vérifications des 7 étapes (bases pièges, panne simulée, requêtes comptées)
│   ├── service/            # Projet 4 — VOTRE code : 7 étapes (config.py … cli.py, mes_tests.py)
│   ├── service_corrige/    # Projet 4 terminé : le corrigé, et la référence du correcteur
│   ├── checks_service.py   # Vérifications des 7 étapes (écriture concurrente, disque plein, clés presque justes, 8 API boguées)
│   ├── reentrainement/     # Projet 5 — VOTRE code : 7 étapes (lots.py … cli.py, mes_tests.py)
│   ├── reentrainement_corrige/ # Projet 5 terminé : le corrigé, et la référence du correcteur
│   ├── checks_reentrainement.py # Vérifications des 7 étapes (lots piégés, faux modèles, disque plein, fichier piégé, 11 règles boguées)
│   ├── verif.py            # Moteur de correction des projets
│   ├── exemple_clients.csv # 200 clients scorés par le modèle (format de l'export du tableau de bord)
│   ├── exemple_plan.csv    # 2 000 clients ciblés (format du plan du projet 2) : entrée du projet 3
│   ├── exemple_resultats.csv # Qui est parti un mois plus tard (export du CRM, imparfait)
│   ├── exemple_contacts.csv # Les 1 612 clients à appeler (écrits par `preparer`, projet 3) : entrée du projet 4
│   └── exemple_lots.csv    # 12 photos mensuelles de 300 clients, avec une dérive à partir d'avril 2026 : entrée du projet 5
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
│   ├── test_projets.py     # Le projet 2 : corrigé, programme terminé, erreurs classiques détectées
│   ├── test_projet_campagne.py # Le projet 3 : corrigé, bout en bout, 40 erreurs classiques, autres solutions justes acceptées
│   ├── test_projet_service.py  # Le projet 4 : corrigé, vrai serveur uvicorn, 56 erreurs classiques reçoivent le bon message
│   ├── test_projet_reentrainement.py # Le projet 5 : corrigé, bout en bout, 55 erreurs classiques et 11 règles boguées reçoivent le bon message
│   ├── test_atelier_git.py # L'atelier Git : chaque solution réussit, les erreurs classiques sont expliquées
│   └── test_solutions.py   # Les affirmations des corrigés restent vraies
├── .github/workflows/      # Intégration continue (pytest à chaque push)
├── PARCOURS.md             # 🧭 Parcours de compétences et exercices
├── CORRIGES.md             # ✅ Indices, solutions et exemple de rapport final
├── JOURNAL.md              # 📓 Votre journal de progression
└── requirements.txt
```

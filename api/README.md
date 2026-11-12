# 🚀 Projet 2 — Une API de prédiction du churn

Dans le Projet 1, le tableau de bord charge le modèle lui-même. En entreprise, le modèle est
plutôt **servi par une API** : le CRM, le site web ou une autre équipe envoient un client en
JSON et reçoivent une probabilité de départ. Ce projet vous apprend à construire ce service,
à le tester et à le livrer dans une image Docker.

> **Prérequis** : avoir terminé (ou au moins lancé) le Projet 1, décrit dans [`PARCOURS.md`](../PARCOURS.md).

## Compétences travaillées

| Compétence | Où dans le code |
|---|---|
| API REST avec FastAPI (routes, codes HTTP, documentation automatique) | `api/main.py` |
| Validation des entrées avec Pydantic (bornes, valeurs autorisées, champs inconnus) | `api/schemas.py` |
| Charger un modèle une seule fois au démarrage (*lifespan*) | `create_app()` dans `api/main.py` |
| Rendre une application testable (*factory* `create_app`) | `api/main.py`, `tests/test_api.py` |
| Tests d'API avec `TestClient` | `tests/test_api.py` |
| Appeler une API depuis Python | `api/client_example.py` |
| Conteneurisation (cache des couches, utilisateur non-root, healthcheck) | `Dockerfile` |

## Démarrage rapide

```bash
pip install -r requirements.txt
python scripts/train.py              # 1. entraîne et sauvegarde models/churn_model.joblib
uvicorn api.main:app --reload        # 2. démarre l'API sur http://localhost:8000
```

Ouvrez **http://localhost:8000/docs** : FastAPI génère une documentation interactive où vous
pouvez tester chaque route. Dans un autre terminal :

```bash
python api/client_example.py
```
```
Modèle servi : LogisticRegression (seuil 0.5)

Client C00042 : 95.0% → À RISQUE

Et si le même client changeait de contrat ?
  Mensuel    95.0%  ██████████████████████████████████████
  Annuel     83.4%  █████████████████████████████████
  Biannuel   69.4%  ████████████████████████████
```

### Avec Docker

```bash
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api
```

Le modèle est entraîné pendant la construction : l'image fonctionne seule, sans rien d'autre.

## Les routes

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/health` | L'API est-elle en vie ? (utilisé par Docker et les orchestrateurs) |
| `GET` | `/model` | Quel modèle est servi : type, variables, seuil, date, métriques |
| `POST` | `/predict` | Probabilité de churn d'**un** client |
| `POST` | `/predict/batch` | Probabilités pour **1 à 1000** clients en un seul appel |

Exemple avec `curl` :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 4, "monthly_charges": 89.9, "support_calls": 3, "num_products": 1,
       "contract": "Mensuel", "payment_method": "Chèque", "internet_service": "Fibre",
       "tech_support": "Non", "senior": "Non"}'
```
```json
{"customer_id": null, "churn_probability": 0.949697086640614, "at_risk": true, "threshold": 0.5}
```

`total_charges` est facultatif : s'il est absent, l'API prend *facture × ancienneté*, comme
le formulaire du tableau de bord.

Le seuil de décision se règle au démarrage : `CHURN_THRESHOLD=0.3 uvicorn api.main:app`.

## Ce qu'il faut comprendre (et que les tests vérifient)

- **La validation protège le modèle.** Une ancienneté négative, un contrat `"Hebdo"` ou un
  champ mal orthographié (`"tenure"`) renvoient une erreur **422** qui dit exactement quoi
  corriger. Sans elle, le modèle renverrait une probabilité sans aucun sens.
- **L'API ne doit rien changer à la prédiction.** `test_api_matches_the_model_used_directly`
  compare la réponse de l'API au modèle appelé directement. Un ordre de colonnes inversé
  passerait inaperçu sans ce test.
- **Batch et unitaire doivent être cohérents.** Le même client a la même probabilité, qu'il
  soit envoyé seul ou dans un lot.
- **Un modèle absent doit donner une erreur claire.** Si le fichier n'existe pas, l'API refuse
  de démarrer et indique la commande à lancer.

```bash
pytest -q tests/test_api.py
```

---

## 🧭 Exercices

Mêmes règles que pour le Projet 1 : une branche par exercice, `pytest -q` après chaque
changement, et vos notes dans `JOURNAL.md`.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé

### Découvrir

- [ ] 🟢 2.1 — Dans `/docs`, testez `POST /predict` avec l'exemple. Passez `contract` à
  `"Biannuel"`, puis `tech_support` à `"Oui"`. Notez l'effet de chaque changement.
- [ ] 🟢 2.2 — Envoyez 3 requêtes invalides de votre choix. Pour chacune, trouvez la ligne de
  `api/schemas.py` qui l'a refusée.
- [ ] 🟢 2.3 — Lisez `tests/test_api.py` en entier. Pour chaque test, écrivez en une phrase
  quel bug il empêche.

### Améliorer

- [ ] 🟡 2.4 — Ajoutez un champ `risk_level` à la réponse : `"faible"` (< 0,3), `"moyen"`
  (< seuil), `"élevé"` (≥ seuil). Écrivez le test **avant** le code.
- [ ] 🟡 2.5 — Ajoutez un middleware qui mesure la durée de chaque requête et la renvoie dans
  un en-tête `X-Process-Time`. Comparez `/predict` appelé 100 fois et `/predict/batch` avec
  100 clients : lequel est le plus rapide, et pourquoi ?
- [ ] 🟡 2.6 — Construisez l'image Docker, lancez-la, puis vérifiez sa santé avec
  `docker ps` (colonne *STATUS*). Que se passe-t-il si vous changez une ligne de `api/main.py`
  et reconstruisez : quelles étapes sont rejouées ?
- [ ] 🟡 2.7 — Ajoutez une route `GET /predict/example` qui renvoie la prédiction pour
  `EXAMPLE_CUSTOMER`. Pratique pour une démonstration rapide.

### Aller plus loin

- [ ] 🔴 2.8 — Protégez les routes `/predict*` avec une clé d'API : en-tête `X-API-Key`
  vérifié par une dépendance FastAPI (`Depends`). La clé est lue dans une variable
  d'environnement. Testez l'absence de clé (401) et une mauvaise clé (403).
- [ ] 🔴 2.9 — Ajoutez `POST /explain` : pour la régression logistique, renvoyez les 3
  variables qui poussent le plus la prédiction (coefficient × valeur transformée).
  Indice : `pipeline.named_steps["prep"].get_feature_names_out()`.
- [ ] 🔴 2.10 — Branchez le tableau de bord sur l'API : l'onglet « Prédire un client » appelle
  `POST /predict` au lieu de charger le modèle. Vous avez séparé le *front* du *back*.
- [ ] 🔴 2.11 — Enregistrez chaque prédiction (entrée, sortie, date) dans une base SQLite.
  Ces journaux sont la matière première du Projet 3 : la surveillance du modèle.

### Auto-évaluation

1. Pourquoi charger le modèle au démarrage plutôt qu'à chaque requête ?
2. Quelle différence entre une erreur 422 et une erreur 500 ? Laquelle est la faute du client ?
3. Pourquoi `extra="forbid"` ? Racontez un bug réel qu'il éviterait.
4. Vous déployez un nouveau modèle et les prédictions changent pour tous les clients. Comment
   la route `/model` vous aide-t-elle à comprendre ce qui s'est passé ?

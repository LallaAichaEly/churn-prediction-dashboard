# Journal des versions

## 1.0.0 — Projet terminé

Le projet passe de « modèle dans un notebook » à « modèle en production ».

- **API REST** (`src/churn/api.py`, FastAPI) : `/health`, `/model`, `/predict`, `/predict/batch`.
  Validation stricte des entrées (erreur 422), seuil réglable, modèle rechargé après réentraînement.
- **Surveillance du drift** (`src/churn/monitoring.py`) : PSI numérique et catégoriel, simulation
  d'une nouvelle période, script d'alerte `scripts/check_drift.py` (code de sortie 1 si dérive).
- **Tableau de bord** : nouvel onglet « Surveillance (drift) ».
- **Fiche modèle** générée par `scripts/report.py` → `RAPPORT_MODELE.md` : comparaison des modèles,
  seuil optimal et impact financier, facteurs de churn, limites.
- **Tests** : 13 → 43 (API, drift, sauvegarde/rechargement du modèle, test de fumée du tableau de bord).
- **Intégration continue** : GitHub Actions sur Python 3.11 et 3.13.
- **Parcours** : module 6 revu, nouveau module 7 « Surveiller en production », pistes de projets suivants.
- Remplacement de `use_container_width` (déprécié) dans le tableau de bord.

## 0.1.0 — Première version

Données synthétiques, pipeline scikit-learn, régression logistique et gradient boosting,
seuil métier, importance par permutation, tableau de bord Streamlit à 5 onglets, 13 tests,
parcours de compétences en 6 modules.

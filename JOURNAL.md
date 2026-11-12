# 📓 Journal de progression

Ce journal retrace l'avancement du projet, session par session. Gardez ce format pour vos
propres exercices : **ce que j'ai fait → ce que j'ai mesuré → ce que j'ai appris**.
C'est exactement ce qu'on vous demandera de raconter en entretien.

---

## Session 1 — Mise en place du projet d'apprentissage

**Fait** : générateur de données synthétiques avec « vérité cachée » connue, pipeline
scikit-learn sans fuite de données, deux modèles (logistique, gradient boosting), choix du
seuil par les coûts métier, tableau de bord Streamlit en 5 onglets, 13 tests, parcours
de compétences en 6 modules (`PARCOURS.md`).

---

## Session 2 — Terminer le projet : industrialisation et explications (12/11/2026)

**Objectif** : passer d'un notebook amélioré à un projet *livrable* — un modèle qu'on peut
appeler depuis une autre application, des tests qui tournent seuls, des prédictions
qu'on peut expliquer à un manager.

### Ce qui a été ajouté

| Ajout | Fichier | Exercice du parcours |
|---|---|---|
| API REST `POST /predict`, `POST /predict/batch`, `GET /health`, validation Pydantic | `api.py` | 6.4 |
| Intégration continue (Python 3.11 et 3.12) | `.github/workflows/tests.yml` | 6.3 |
| Test « modèle rechargé = mêmes prédictions » | `tests/test_churn.py` | 6.2 |
| Test de fumée du tableau de bord (Streamlit `AppTest`) | `tests/test_app.py` | 6.x |
| Courbe précision-rappel + précision moyenne (AP) | `model.pr_points`, onglet Performance | 4.4 |
| Calibration + score de Brier | `model.calibration_points`, onglet Performance | 4.5 |
| Explication locale « et si… ? » d'une prédiction | `src/churn/explain.py`, onglets Clients à risque et Prédire | 5.3 |
| Rapport final chiffré | `RAPPORT.md` | Projet final |

### Mesuré

- **38 tests** passent en 20 s (13 → 38) : modèle, API (contrat d'entrée/sortie), tableau de bord.
- Validation croisée 5 plis : logistique **0,819 ± 0,005** vs boosting 0,799 ± 0,009 →
  l'écart est réel, la logistique reste le modèle retenu.
- Calibration : écart moyen de **3,7 points** entre probabilité prédite et taux observé ;
  score de Brier 0,158 (logistique) vs 0,171 (boosting).
- Seuil optimal à 500 € / 100 € : **0,26** → coût 52 200 € contre 93 700 € au seuil 0,5
  et 80 700 € en contactant tout le monde (détails dans `RAPPORT.md`).
- API testée en conditions réelles : profil risqué → 0,97 ; contrat « Hebdo » → erreur 422.
- Surprise instructive : l'explication « et si… ? » ne cite jamais le contrat mensuel, car
  le client typique est lui-même mensuel. Une explication est toujours *relative à une référence*.

### Appris (à retenir pour un entretien)

1. **Valider à la frontière.** L'API refuse un contrat « Hebdomadaire » ou une ancienneté
   négative (erreur 422) au lieu de renvoyer un score absurde. Le pipeline, lui, tolère une
   modalité inconnue (`handle_unknown="ignore"`) : deux niveaux de défense différents.
2. **Un test protège un contrat, pas une ligne de code.** `test_api_categories_match_training_data`
   casse si quelqu'un ajoute un type de contrat dans `data.py` sans mettre l'API à jour.
3. **Une probabilité doit être calibrée pour être utilisée en euros.** Si le modèle dit 30 %,
   il faut observer ~30 % de départs, sinon le budget de rétention est faux.
4. **Expliquer un score ≠ expliquer le modèle.** L'importance globale dit « le contrat compte » ;
   l'explication locale dit « ce client-ci part surtout à cause de ses 5 appels au support ».

### Prochaines étapes (pour vous)

- [ ] Parcourir les exercices 📘 de `PARCOURS.md` (lire → expliquer → étendre).
- [ ] Exercice 3.4 : validation croisée dans `scripts/train.py`.
- [ ] Exercice 5.5 : SHAP, et comparer avec la méthode « et si… ? ».
- [ ] Exercice 6.5 : déployer sur Streamlit Community Cloud et ajouter le lien au README.

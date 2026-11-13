# 📄 Rapport — Prédiction du churn client

> Généré par `python scripts/report.py` (graine 42). Toutes les valeurs sont recalculées à chaque
> exécution. **Données synthétiques** : ce rapport illustre la démarche, pas un cas réel.

## 1. Problème

L'opérateur perd **35,5 %** de ses clients. Acquérir un nouveau client coûte bien plus cher
que d'en garder un : l'objectif est d'**identifier à l'avance les clients susceptibles de partir** pour
leur proposer une action de rétention ciblée, plutôt que d'arroser tout le monde.

## 2. Données

- **5 000 clients**, 5 variables numériques et 5 catégorielles (contrat, paiement, services…).
- Contrôles qualité (`validate()`) : aucun problème détecté.
- Découpage stratifié : 75 % entraînement, 25 % test (1 250 clients dont 443 départs).

## 3. Modèle retenu

| Modèle | ROC-AUC (test) | ROC-AUC (validation croisée 5 plis) | Précision @0,5 | Rappel @0,5 |
|---|---|---|---|---|
| **logistic** | 0,834 | 0,819 ± 0,005 | 68,9 % | 63,4 % |
| gboost | 0,807 | 0,799 ± 0,009 | 65,5 % | 59,6 % |

**Choix : `logistic`.** Meilleure ROC-AUC, écart confirmé par la validation croisée, et le modèle
le plus simple à expliquer et à maintenir. Règle d'or : un modèle complexe doit *mériter* sa complexité.

## 4. Ce qui fait partir les clients

Importance par permutation (baisse de ROC-AUC quand la variable est mélangée) :

| Rang | Variable | Baisse d'AUC |
|---|---|---|
| 1 | Type de contrat | 0,123 |
| 2 | Ancienneté (mois) | 0,051 |
| 3 | Appels au support | 0,042 |
| 4 | Facture mensuelle | 0,022 |
| 5 | Support technique | 0,020 |

## 5. Seuil de décision

Le seuil par défaut (0,5) n'a aucune raison d'être le bon. Hypothèses de coût :
**500 €** par client perdu non détecté, **50 €** par action de rétention inutile.

| | Seuil 0,5 | Seuil optimal **0,10** |
|---|---|---|
| Clients contactés (sur 1 250) | 408 | 997 |
| Précision | 68,9 % | 43,9 % |
| Rappel (départs détectés) | 63,4 % | 98,9 % |
| Départs manqués | 162 | 5 |
| Coût total | 87 350 € | **30 450 €** |

Quand rater un départ coûte 10× plus cher qu'une action inutile, on a intérêt à
**contacter plus large** : le seuil baisse, le rappel monte, la précision baisse — et c'est voulu.

## 6. Impact financier estimé (jeu de test)

| Stratégie | Coût |
|---|---|
| Ne rien faire | 221 500 € |
| Contacter tout le monde | 40 350 € |
| **Cibler avec le modèle (seuil 0,10)** | **30 450 €** |

Par rapport à la meilleure stratégie sans modèle, le ciblage économise **9 900 €**
(25 %), soit environ **7 920 € pour 1 000 clients**.

**Lecture critique.** Avec une action aussi bon marché, « contacter tout le monde » est déjà une
stratégie correcte : le modèle apporte surtout en évitant 248 contacts inutiles
(au prix de 5 départs manqués).
Plus l'action coûte cher (remise, appel d'un conseiller), plus le ciblage devient rentable :

| Coût d'une action inutile | Seuil optimal | Clients contactés | Coût avec modèle | Meilleur coût sans modèle | Économie |
|---|---|---|---|---|---|
| 25 € | 0,06 | 1 077 | 15 850 € | 20 175 € | 21 % |
| **50 €** | 0,10 | 997 | 30 450 € | 40 350 € | 25 % |
| 100 € | 0,26 | 737 | 52 200 € | 80 700 € | 35 % |
| 200 € | 0,28 | 702 | 83 300 € | 161 400 € | 48 % |

👉 Avant de déployer, **mesurer le vrai coût d'une action** : c'est lui qui décide si le modèle vaut l'effort.

## 7. Recommandations

1. **Fidéliser les contrats mensuels** (churn 50,9 % contre 15,9 % pour les contrats
   annuels ou biannuels) : offre de passage à l'engagement annuel.
2. **Proposer le support technique** (churn 39,2 % sans, 28,4 % avec) — à valider
   par un test A/B avant généralisation.
3. **Alerte « 3 appels au support ou plus »** : 14 % des clients, churn 51,2 %
   (contre 32,9 %). Un appel de suivi proactif par un conseiller senior.
4. **Encourager le prélèvement automatique** (churn 43,0 % par chèque, 34,0 % par prélèvement).

## 8. Limites

- **Données synthétiques** : les vraies causes sont connues et le bruit est maîtrisé ; sur des données
  réelles, attendez-vous à une performance plus faible.
- **Modèle de coût simplifié** : on suppose qu'une action de rétention sauve 100 % des clients ciblés et
  on ignore son coût pour les vrais positifs. À affiner avec un taux de succès mesuré.
- **Corrélation ≠ causalité** : un client avec support technique part moins, mais *offrir* le support
  ne garantit pas le même effet. Seule une expérience contrôlée (A/B) le prouve.
- **Pas de validation temporelle** ni de suivi de la dérive des données (*data drift*) ; la
  calibration des probabilités n'est pas vérifiée (exercice 4.5 du parcours).

## 9. Reproduire

```bash
pip install -r requirements.txt
python scripts/train.py      # entraîne et sauvegarde le meilleur modèle
python scripts/report.py     # régénère ce rapport
streamlit run app.py         # explore les résultats de façon interactive
```

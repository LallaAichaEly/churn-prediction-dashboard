# 🏁 Rapport final — Prédire le churn pour cibler la rétention

> Exemple de « projet final » du parcours (1 page). Tous les chiffres sont reproductibles :
> `python scripts/train.py`, puis l'onglet *Performance* du tableau de bord.
> Données **synthétiques** (graine 42) : les conclusions illustrent la démarche, pas un vrai marché.

## 1. Problème

Un opérateur perd **35,5 %** de ses clients. Une action de rétention (appel + offre) coûte
de l'argent : la proposer à tout le monde est cher, ne la proposer à personne l'est encore
plus. **Question : à qui proposer l'offre ?**

## 2. Données

5 000 clients, 10 variables (contrat, ancienneté, facture, appels au support, paiement…).
Contrôles qualité automatiques (`validate`) : aucune valeur manquante, cible binaire,
identifiants uniques. Découpage stratifié 75 % / 25 % (1 250 clients de test, dont 443 partants).

| Segment | Taux de churn |
|---|---|
| Contrat mensuel / annuel / biannuel | **50,9 %** / 21,5 % / 8,9 % |
| Ancienneté 0-6 mois / plus de 48 mois | **50,7 %** / 6,8 % |
| 0 appel au support / 3 appels ou plus | 26,3 % / **51,2 %** |
| Sans / avec support technique | 39,2 % / 28,4 % |

## 3. Modèle retenu : régression logistique

| Modèle | ROC-AUC (validation croisée, 5 plis) | ROC-AUC test | Précision moy. (AP) | Brier ↓ |
|---|---|---|---|---|
| **Régression logistique** | **0,819 ± 0,005** | **0,834** | **0,709** | **0,158** |
| Gradient boosting | 0,799 ± 0,009 | 0,807 | 0,661 | 0,171 |

- La logistique gagne sur **tous** les critères et l'écart (2 points d'AUC) dépasse nettement la
  variabilité entre plis : il est réel. Raison : le churn est généré par une relation
  linéaire en logit ; le boosting ajoute de la complexité sans signal à capturer.
- Les variables dérivées (`avg_charge_per_month`, `calls_per_year`) n'apportent rien
  (0,819 → 0,819) : on ne les garde pas — plus simple, aussi bon.
- **Probabilités fiables** : écart moyen de 3,7 points entre prédit et observé sur 10 groupes
  (onglet *Performance → Calibration*). On peut donc raisonner en euros.

## 4. Seuil choisi : 0,26 (et pourquoi pas 0,5)

Hypothèses : un client perdu coûte **500 €** (≈ 7,5 mois de la facture moyenne d'un partant, 67 €) ;
une action coûte **100 €** (un mois offert ≈ 67 € + traitement). Coût total sur le jeu de test :

| Stratégie | Clients contactés | Partants détectés | Coût total |
|---|---|---|---|
| Ne rien faire | 0 | 0 % | 221 500 € |
| Contacter tout le monde | 1 250 | 100 % | 80 700 € |
| Modèle, seuil 0,5 (par défaut) | 408 | 63 % | 93 700 € |
| **Modèle, seuil 0,26 (optimal)** | **737** | **91 %** | **52 200 €** |

Le seuil optimal **divise le coût par 4 par rapport à l'inaction** et économise **35 %**
par rapport à « contacter tout le monde ». Le seuil par défaut 0,5 serait une erreur : il
coûte plus cher que de contacter tout le monde, car un départ raté coûte 5 fois une action.

⚠️ Le seuil dépend du rapport des coûts : à 500 € / 50 €, l'optimum descend à 0,10 et on
contacte 80 % des clients — le gain face à « tout le monde » tombe à ~10 000 €
(30 450 € contre 40 350 €). À 100 € / 100 €, il remonte à 0,50. **Le métier doit valider ces deux chiffres.**

## 5. Ce qui fait partir les clients → 3 actions de rétention

Importance par permutation (baisse d'AUC) : contrat **0,123**, ancienneté 0,051, appels au
support 0,042, facture 0,023, support technique 0,020.

1. **Convertir les contrats mensuels** (50,9 % de churn vs 21,5 % en annuel) : remise
   d'engagement ciblée sur les mensuels à risque.
2. **Traiter la cause des appels au support** : à partir de 3 appels, un client sur deux part.
   Alerte automatique au 3ᵉ appel.
3. **Soigner les 6 premiers mois** (50,7 % de churn) : parcours d'accueil + support
   technique offert, qui est associé à 11 points de churn en moins.

## 6. Limites

- **Données synthétiques** : à revalider sur données réelles (exercice 1.5).
- **Corrélation ≠ causalité** : l'effet du support technique est observé, pas prouvé.
  Avant de généraliser une action, la tester en A/B test.
- **Modèle de coût simplifié** : il suppose qu'une action sauve 100 % des partants contactés
  et ne compte pas son coût sur les vrais positifs. Avec un taux de succès réaliste
  (ex. 30 %), le seuil optimal remonte.
- **Sous-estimation des effets rares** : le paiement par chèque (15 % des clients) et le statut
  senior ont un vrai effet dans les données mais une faible importance globale.
- **Dérive** : le comportement client change. Surveiller chaque mois l'AUC et la calibration
  sur les nouveaux clients, et ré-entraîner si l'AUC passe sous 0,78.

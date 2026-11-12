# 📋 Fiche modèle — Prédiction du churn

> Générée automatiquement par `python scripts/report.py` (paquet `churn` v1.0.0). Ne pas modifier à la main : relancez le script.

## 1. Objectif

Estimer, pour chaque client, la probabilité qu'il résilie son abonnement, afin de cibler les actions de rétention (appel, remise…) sur les clients les plus à risque.

## 2. Données

- **5 000 clients** synthétiques (`data/customers.csv`), taux de churn **35,5 %**.
- Découpage stratifié : 3 750 clients pour l'entraînement, **1 250 pour le test** (443 départs).
- 10 variables : ancienneté, facture mensuelle et totale, appels au support, nombre de produits, contrat, moyen de paiement, service internet, support technique, senior.

## 3. Modèles comparés (jeu de test, seuil 0,5)

| Modèle | Variables dérivées | ROC-AUC | Précision | Rappel | F1 |
|---|:---:|---:|---:|---:|---:|
| logistic | oui | 0,835 | 68,7 % | 63,4 % | 0,660 |
| **logistic** | non | **0,834** | 68,9 % | 63,4 % | 0,660 |
| gboost | oui | 0,811 | 65,4 % | 59,4 % | 0,622 |
| gboost | non | 0,807 | 65,5 % | 59,6 % | 0,624 |

**Modèle retenu : `logistic` sans variables dérivées** — ROC-AUC 0,834. Règle : parmi les modèles à moins de 0,005 de la meilleure ROC-AUC (0,835), on garde le plus simple. C'est aussi celui que `scripts/train.py` sauvegarde et que l'API sert.

## 4. Seuil de décision et impact financier

Hypothèses : un client perdu non détecté coûte **500 €**, une action de rétention inutile coûte **50 €**, et une action sur un vrai partant le retient.

| Stratégie | Clients contactés | Départs manqués | Coût total |
|---|---:|---:|---:|
| Ne rien faire | 0 | 443 | 221 500 € |
| Contacter tout le monde | 1 250 | 0 | 40 350 € |
| Modèle, seuil 0,50 | 408 | 162 | 87 350 € |
| **Modèle, seuil 0,10 (optimal)** | **997** | **5** | **30 450 €** |

Au seuil optimal **0,10** : précision 43,9 %, rappel 98,9 %. Gain par rapport à la meilleure stratégie *sans* modèle : **9 900 €** sur 1 250 clients, soit **8 € par client**.

💡 Pour un modèle bien calibré, la théorie donne un seuil optimal ≈ coût d'une action inutile ÷ (coût action inutile + coût client perdu) = 0,09. Plus rater un départ coûte cher par rapport à une action, plus le seuil baisse — jusqu'à contacter presque tout le monde : le modèle sert alors surtout à **écarter les clients sûrs**.

## 5. Facteurs de churn (importance par permutation)

| Rang | Variable | Baisse de ROC-AUC | Vraie cause ? |
|---:|---|---:|:---:|
| 1 | `contract` | 0,1233 | ✅ |
| 2 | `tenure_months` | 0,0509 | ✅ |
| 3 | `support_calls` | 0,0419 | ✅ |
| 4 | `monthly_charges` | 0,0225 | ✅ |
| 5 | `tech_support` | 0,0197 | ✅ |
| 6 | `num_products` | 0,0114 | ✅ |

Les données étant synthétiques, on connaît les vraies causes (`_churn_logit` dans `src/churn/data.py`) : on peut vérifier que le modèle les retrouve.

## 6. Limites et risques

- **Données synthétiques** : les chiffres illustrent la méthode, pas un vrai marché.
- **Coûts simplifiés** : on suppose qu'une action retient toujours le client et on ignore son coût quand elle réussit. Cela surestime le gain : à affiner avec le métier (taux de succès réel, coût de la remise).
- **Corrélation ≠ causalité** : une variable importante n'est pas forcément un levier d'action.
- **Dérive** : le modèle se dégrade si les clients changent. Surveiller avec `python scripts/check_drift.py` (et l'onglet « Surveillance » du tableau de bord).
- **Un seul découpage train/test** : les écarts entre modèles de moins de ~0,01 d'AUC ne sont pas forcément significatifs (voir exercice 3.4, validation croisée).

## 7. Reproduire

```bash
python scripts/train.py     # entraîne et sauvegarde le modèle
python scripts/report.py --cost-fn 500 --cost-fp 50
```

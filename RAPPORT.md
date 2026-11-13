# 📄 Rapport — Prédire le churn pour cibler la rétention

*Exemple de rapport final du parcours (une page). Écrivez le vôtre avec vos propres chiffres et choix.*

## 1. Problème

Un opérateur télécom perd **35,5 %** de ses clients. Une action de rétention (appel, remise) coûte
environ **50 €** par client contacté ; un client perdu représente environ **500 €** de marge.
Question : **qui contacter**, pour que la campagne rapporte plus qu'elle ne coûte ?

## 2. Données

5 000 clients synthétiques, 10 variables (contrat, ancienneté, facture, appels au support, services…).
Contrôles qualité automatiques avant chaque entraînement (colonnes, valeurs manquantes, cible binaire,
doublons). Découpage stratifié : 75 % entraînement, 25 % test (1 250 clients, jamais vus à l'entraînement).

## 3. Modèle retenu : régression logistique

| Modèle | ROC-AUC (validation croisée 5 plis) |
|---|---|
| **Régression logistique** | **0,819 ± 0,005** |
| Forêt aléatoire | 0,806 ± 0,003 |
| Gradient boosting (optimisé par GridSearch) | 0,812 |

La logistique gagne sur chacun des 5 plis. Elle est aussi **interprétable** (contribution de chaque
variable pour chaque client) et **bien calibrée** (une probabilité de 30 % correspond à environ 30 % de
départs observés), ce qui permet de raisonner directement en euros.

**Principaux facteurs de départ** : contrat mensuel (50,9 % de churn contre 8,9 % en biannuel),
faible ancienneté, appels répétés au support, facture élevée, absence de support technique.

## 4. Seuil de décision : 0,30

Le seuil dépend du **taux de succès de l'action de rétention**, encore inconnu :

| Succès de l'action | Seuil optimal | Gain net — modèle | Gain net — contacter tout le monde |
|---|---|---|---|
| 100 % (hypothèse optimiste) | 0,14 | 169 200 € | 159 000 € |
| 50 % | 0,26 | 64 400 € | 48 250 € |
| **30 % (hypothèse prudente)** | **0,30** | **24 700 €** | 3 950 € |

*(Gains sur les 1 250 clients du jeu de test.)* Je recommande de démarrer à **0,30** : rentable
dès 30 % de succès, et c'est dans ce scénario prudent que le modèle apporte le plus par rapport à une
campagne non ciblée (**×6**).

## 5. Impact estimé

Avec un taux de succès de 30 %, environ **20 000 € de gain net pour 1 000 clients**, en contactant un
peu plus de la moitié d'entre eux (54 %). Le gain dépend surtout du taux de succès réel, plus que de
l'AUC du modèle.

## 6. Limites

- **Données synthétiques** : les chiffres illustrent la méthode, pas un marché réel.
- **Valeur client et coût d'action supposés** : à remplacer par les vrais chiffres financiers.
- **Prédire n'est pas expliquer** : le modèle dit *qui* risque de partir, pas *quelle action* le
  retiendra. Seul un test A/B mesure l'effet causal d'une campagne.
- **Variable sensible** : `senior` (âge) n'apporte quasiment rien au modèle. La retirer ne coûte
  que 0,001 d'AUC (0,8192 → 0,8185 en validation croisée) et réduit le risque de discrimination.
- **Dérive des données** : comportements et offres changent ; le modèle doit être surveillé.

## 7. Prochaines étapes

1. **Test A/B** sur 2 mois : 50 % des clients à risque contactés, 50 % en groupe témoin → taux de succès réel.
2. Recalculer le seuil avec ce taux (onglet Performance du tableau de bord).
3. Surveiller chaque mois la distribution des variables et l'AUC réelle ; ré-entraîner si elles dérivent.

---
*Reproduire les chiffres : `python corriges/module_3_modeles.py` et `python corriges/module_4_evaluation.py`.*

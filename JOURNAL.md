# 📓 Journal d'apprentissage

Une entrée par séance. Notez ce que vous avez **essayé**, ce que vous avez **mesuré** et ce que
vous avez **compris** — y compris les échecs. Dans six mois, ce journal montrera votre
progression mieux que n'importe quel CV.

> Astuce : copiez le modèle ci-dessous en haut du fichier à chaque séance.

---

## Modèle d'entrée

### AAAA-MM-JJ — Exercice X.Y : titre court

- **Objectif** : …
- **Ce que j'ai fait** : …
- **Résultat (avant → après)** : ROC-AUC … → … · précision … → … · rappel … → …
- **Ce que j'ai compris** : …
- **Question en suspens** : …

---

## Point de départ — résultats de référence

Mesurés avec `python scripts/train.py` (données synthétiques, `seed=42`, 5 000 clients,
taux de churn 35,5 %). Toutes vos expériences se comparent à ces chiffres.

| Modèle | ROC-AUC | Accuracy | Précision | Rappel | F1 |
|---|---|---|---|---|---|
| Régression logistique | 0,834 | 0,769 | 0,689 | 0,634 | 0,660 |
| Gradient boosting | 0,807 | 0,746 | 0,655 | 0,596 | 0,624 |

Mes 3 hypothèses avant de commencer (exercice 1.4) :

1. …
2. …
3. …

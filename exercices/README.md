# 🥋 Katas auto-corrigés

Un *kata* est un petit exercice qu'on répète jusqu'à le maîtriser. Ici, **vous écrivez le code
et les tests vous corrigent immédiatement** : c'est la façon la plus rapide de progresser,
parce que vous savez tout de suite si vous avez compris.

## Comment ça marche

```bash
pytest exercices -q -rs
```

```
.F.sssssssssssssss
================================ Progression ================================
✅ réussis : 2/19   ❌ à corriger : 1   ⏳ à faire : 16
```

| Symbole | Signification |
|---|---|
| `.` ✅ | Réussi |
| `F` ❌ | Votre code tourne mais le résultat est faux : lisez la ligne `AssertionError`, elle dit pourquoi |
| `s` ⏳ | Pas encore commencé (la fonction contient encore `raise NotImplementedError`) |

**Méthode conseillée, pour chaque fonction :**

1. Lisez l'énoncé (la docstring) **et** le test correspondant dans `test_module1_donnees.py`.
2. Essayez d'abord dans un terminal Python sur un petit DataFrame (voir l'en-tête du fichier).
3. Écrivez votre solution, relancez `pytest exercices -q -rs`.
4. Bloqué plus de 20 minutes ? Relisez l'indice. Toujours bloqué ? Ouvrez le corrigé
   dans `solutions/`, **fermez-le**, puis réécrivez la solution de mémoire.
5. Une fois tout en vert, notez dans [`JOURNAL.md`](../JOURNAL.md) ce que vous avez appris.

> 💡 Pour vérifier que le correcteur lui-même fonctionne : `CHURN_SOLUTIONS=1 pytest exercices -q`
> exécute les tests sur les corrigés (tout doit être vert).

## Les modules

| Module | Fichier à compléter | Compétences | Katas |
|---|---|---|---|
| 1 — Comprendre les données | `module1_donnees.py` | pandas : `groupby`, `isna`, `pd.cut`, filtres booléens, lift, segmentation | 7 fonctions · 19 tests |
| 2 — Préparer les variables | *à venir* | | |
| 3 — Modéliser | *à venir* | | |
| 4 — Évaluer et décider | *à venir* | | |

## Après le Module 1

Vous êtes prêt pour les exercices 1.3 à 1.5 de [`PARCOURS.md`](../PARCOURS.md) :
`factures_hors_bornes` est exactement la règle à ajouter dans `validate()` (exercice 1.3).

# 🎯 Défis auto-corrigés

Vous écrivez une fonction, un script vous dit tout de suite si elle est juste et, sinon,
**pourquoi** elle ne l'est pas. C'est la façon la plus rapide de passer de « j'ai lu » à
« je sais faire ».

```bash
python scripts/progression.py            # votre tableau de progression
python scripts/progression.py --defi 5   # se concentrer sur un défi
```

```
🧭 Série 01 — Les fondamentaux  (3/10 réussis)

  ✅  1 🟢 taux_churn_par           Parcours 1.2
  ❌  5 🟡 standardiser             Parcours 2.1 · 2.5
        ↳ La moyenne du train standardisé vaut -0.310 au lieu de 0. Calculez-vous la
          moyenne sur le TRAIN uniquement ? Utiliser train + test, c'est une fuite de données.
  ⬜  6 🟢 precision_rappel         Parcours —
```

✅ réussi · ❌ résultat faux (le message vous guide) · 💥 votre code a planté · ⬜ à faire

## Les séries

| Série | Fichier à compléter | Thèmes | Défis |
|---|---|---|---|
| 01 — Les fondamentaux | `defis/serie_01.py` | pandas, qualité des données, fuite de données, précision/rappel, coût métier, validation croisée, lift | 10 (🟢×5 🟡×3 🔴×2) |

## Méthode conseillée

1. **Essayez seul·e au moins 15 minutes** avant toute aide. La difficulté fait partie de l'apprentissage.
2. Si le message d'erreur ne suffit pas, lisez la vérification correspondante dans
   `defis/checks_serie_01.py` : les cas testés sont des indices.
3. Une fois ✅, ouvrez `defis/corrige_serie_01.py` et comparez. Souvent, on découvre une
   écriture plus courte ou un piège évité par chance.
4. Notez dans votre `JOURNAL.md` : ce qui vous a bloqué, ce que vous avez appris.

> 💡 Les défis 🔴 réutilisent le code du projet (`churn.model.build_pipeline`) : vous
> apprenez aussi à lire et réutiliser un code existant, compétence clé en entreprise.

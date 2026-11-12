# 🎯 Défis auto-corrigés

Vous écrivez une fonction, un script vous dit tout de suite si elle est juste et, sinon,
**pourquoi** elle ne l'est pas. C'est la façon la plus rapide de passer de « j'ai lu » à
« je sais faire ».

```bash
python scripts/defis.py            # votre tableau de progression
python scripts/defis.py --defi 5   # se concentrer sur un défi
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

> 👋 **Vous débutez en Python, ou vous ne vous sentez pas encore « à niveau » ?** Commencez par la
> **série 00** : elle part de zéro (écrire une fonction) et prépare tout le reste. Ses défis portent les
> numéros 30 à 37 parce qu'elle a été ajoutée après les autres, mais c'est bien la première à faire :
> `python scripts/defis.py --defi 30`.
>
> 🛠️ **Vous voulez développer ?** Enchaînez avec la **série 06** (défis 38 à 43) : elle ne demande que
> la série 00 et entraîne ce qui sépare un script qui « marche chez moi » d'un code fiable.
> Puis la **série 07** (défis 44 à 49) : dates, fichiers Excel, `set`, générateurs, classes et
> décorateurs, les notions de Python qu'on rencontre dès la première vraie tâche en équipe.

| Série | Fichier à compléter | Thèmes | Défis |
|---|---|---|---|
| **00 — Remise à niveau** 👋 | `defis/serie_00.py` | fonctions, `print` vs `return`, if/elif, boucles et dictionnaires, listes et tri, **débogage** d'un code faux, filtrer un DataFrame, `groupby` refait à la main, nettoyer du texte saisi à la main | 8, n° 30 à 37 (🟢×6 🟡×2) |
| 01 — Les fondamentaux | `defis/serie_01.py` | pandas, qualité des données, fuite de données, précision/rappel, coût métier, validation croisée, lift | 10 (🟢×5 🟡×3 🔴×2) |
| 02 — Explorer les données | `defis/serie_02.py` | lift d'un groupe, règle de validation, `pd.cut` et tranches, segmentation fiable (effectif minimum) | 4 (🟡×3 🔴×1) |
| 03 — Évaluer, expliquer, surveiller | `defis/serie_03.py` | courbe précision-rappel, calibration, AUC par paires, contributions d'une prédiction, PSI | 5 (🟢×1 🟡×2 🔴×2) |
| 04 — Modéliser et industrialiser | `defis/serie_04.py` | poids de classes, réglage de C par la règle de l'erreur standard, compatibilité de versions, prédiction par lot (API), alerte de dérive | 5 (🟢×3 🟡×1 🔴×1) |
| 05 — Prouver l'impact | `defis/serie_05.py` | SQL (`GROUP BY`, top k par groupe avec `ROW_NUMBER() OVER`), intervalle de confiance de Wilson, test A/B d'une offre de rétention, courbe de survie de Kaplan-Meier | 5 (🟢×1 🟡×3 🔴×1) |
| **06 — Développer comme un·e pro** 🛠️ | `defis/serie_06.py` | lire une **traceback**, **écrire un test** qui attrape 5 versions boguées, `try`/`except` précis et saisies humaines (« 0,4 », « 40 % »), **refactoriser** sans changer le résultat, ligne de commande **argparse**, **bissection** façon `git bisect` | 6, n° 38 à 43 (🟢×2 🟡×3 🔴×1) |
| **07 — Écrire du code réutilisable** 🧱 | `defis/serie_07.py` | ancienneté en mois avec **`datetime`** (fins de mois), **complexité** : `set` contre liste sur 200 000 clients, lire un **export Excel** français (`;`, virgule décimale, encodages, identifiants « 00123 »), **générateur** `yield` par lots, une **classe** avec état (attribut d'objet ou de classe ?), un **décorateur** de cache | 6, n° 44 à 49 (🟢×2 🟡×3 🔴×1) |

## Méthode conseillée

1. **Essayez seul·e au moins 15 minutes** avant toute aide. La difficulté fait partie de l'apprentissage.
2. Si le message d'erreur ne suffit pas, lisez la vérification correspondante dans
   `defis/checks_serie_XX.py` : les cas testés sont des indices.
3. Une fois ✅, ouvrez `defis/corrige_serie_XX.py` et comparez. Souvent, on découvre une
   écriture plus courte ou un piège évité par chance.
4. Notez dans votre `JOURNAL.md` : ce qui vous a bloqué, ce que vous avez appris.

> 🧱 Le défi 48 vous fait écrire une **classe** et non une fonction : le correcteur crée plusieurs
> journaux, change le seuil après coup et vérifie que deux objets ne partagent rien.

> 🧪 Le défi 39 inverse les rôles : c'est **vous** qui écrivez le test, et le correcteur vérifie qu'il
> passe sur la bonne fonction et échoue sur 5 versions boguées. C'est le meilleur entraînement pour
> écrire ensuite vos propres `tests/test_*.py`.

> 💡 Les défis 🔴 réutilisent le code du projet (`churn.model.build_pipeline`) : vous
> apprenez aussi à lire et réutiliser un code existant, compétence clé en entreprise.

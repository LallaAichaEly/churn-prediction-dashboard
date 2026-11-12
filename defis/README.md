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
> 🛠️ **Vous voulez développer ?** Faites l'[atelier Git](../atelier_git/README.md) :
> `python scripts/atelier_git.py` propose 8 situations d'équipe (branche, conflit, annulation…) à
> résoudre dans un bac à sable. Enchaînez avec la **série 06** (défis 38 à 43) : elle ne demande que
> la série 00 et entraîne ce qui sépare un script qui « marche chez moi » d'un code fiable.
> Juste après, la **série 11** (défis 68 à 73) prolonge le défi 39 : c'est vous qui écrivez les tests,
> et le correcteur les juge sur des versions boguées ET sur d'autres versions justes.
> Puis la **série 07** (défis 44 à 49) : dates, fichiers Excel, `set`, générateurs, classes et
> décorateurs, les notions de Python qu'on rencontre dès la première vraie tâche en équipe.
> Et la **série 08** (défis 50 à 55) : construire la table du modèle à partir de sources réelles
> (texte, plusieurs tables, JSON, mises à jour, journal d'événements) sans abîmer les données.
>
> 🧠 **Vous préparez un entretien technique ?** La **série 09** (défis 56 à 61) reprend les classiques
> de l'algorithmique sur des données de churn. Elle ne demande que les séries 00 et 07.
>
> 🗄️ **Votre futur poste demandera du SQL ?** La **série 10** (défis 62 à 67) entraîne ce qui sépare
> une requête qui « tourne » d'une requête juste : NULL, anti-jointure, `LEFT JOIN` (ON ou WHERE ?),
> explosion des jointures, fonctions de fenêtre et « îlots et trous ». Elle suit la série 05.

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
| **08 — Assembler les données** 🔗 | `defis/serie_08.py` | retrouver des identifiants dans des notes avec une **regex** (`\b`, `re.IGNORECASE`), **jointure** sans dupliquer ni perdre de clients (`groupby` puis `map`/`merge`), aplatir le **JSON** d'une API (champs absents ou `null`), **vectoriser** un barème (`np.select`, sans `iterrows`), **upsert** idempotent de la mise à jour de nuit (`concat` + `drop_duplicates`), variables d'un **journal d'événements** sans **fuite temporelle** | 6, n° 50 à 55 (🟢×2 🟡×3 🔴×1) |
| **09 — Penser en algorithmes** 🧠 | `defis/serie_09.py` | la méthode d'entretien en 5 étapes et les **complexités** O(1) à O(n²) ; « two sum » avec un **dictionnaire** (la position 0 est « fausse »), délimiteurs d'une formule avec une **pile**, pic d'appels par **fenêtre glissante** (deux pointeurs), **fusion d'intervalles** de dates (`timedelta`, périodes contenues), top k d'un **flux** avec un **tas** (`heapq`, mémoire O(k), égalités), clients proches d'un parti par **parcours en largeur** multi-sources (`deque`) | 6, n° 56 à 61 (🟢×2 🟡×3 🔴×1) |
| **10 — SQL pour l'analyse** 🗄️ | `defis/serie_10.py` | l'ordre d'évaluation d'une requête et la logique à trois valeurs de **NULL** (`IS NULL`, `COUNT(col)`, `COALESCE`), **anti-jointure** (le piège de `NOT IN` face à un NULL), **`LEFT JOIN` : ON ou WHERE ?** et `COUNT(*)` qui compte 1 pour « aucun appel », une variable de « photo » à une date passée (paramètre nommé, priorité AND/OR), **explosion des jointures** (fan-out) et CTE, **fonctions de fenêtre** (`LAG`, cumul `SUM() OVER`, fenêtre nommée), **îlots et trous** (mois consécutifs avec `ROW_NUMBER()`, `HAVING`) | 6, n° 62 à 67 (🟢×2 🟡×3 🔴×1) |
| **11 — Tester comme un·e pro** 🧪 | `defis/serie_11.py` | choisir ses exemples aux **frontières** (`pytest.mark.parametrize`, le piège de 0 dans un `if`), comparer des flottants avec **`pytest.approx`** (et pourquoi `1 - 2/3 != 1/3`), tester le chemin d'erreur avec **`pytest.raises`** (`match`, ni trop large ni fragile), remplacer un service d'e-mails par un **faux** (injection de dépendance, panne simulée, `unittest.mock`), tester un **fichier** dans un dossier temporaire (`tmp_path`, ne jamais écraser), **tests par propriétés** sur des centaines de cas tirés au hasard (graine fixe, Hypothesis) | 6, n° 68 à 73 (🟢×2 🟡×3 🔴×1) |

## Méthode conseillée

1. **Essayez seul·e au moins 15 minutes** avant toute aide. La difficulté fait partie de l'apprentissage.
2. Si le message d'erreur ne suffit pas, lisez la vérification correspondante dans
   `defis/checks_serie_XX.py` : les cas testés sont des indices.
3. Une fois ✅, ouvrez `defis/corrige_serie_XX.py` et comparez. Souvent, on découvre une
   écriture plus courte ou un piège évité par chance.
4. Notez dans votre `JOURNAL.md` : ce qui vous a bloqué, ce que vous avez appris.

> 🗄️ En SQL, une erreur ne fait presque jamais planter : elle donne un chiffre faux, en silence. Le
> correcteur de la série 10 exécute votre requête sur de petites tables où chaque piège se voit (un
> appel anonyme, un client sans facture, un appel le jour même de l'extraction, décembre suivi de
> janvier), puis sur 600 clients comparés à pandas. Pour essayer une requête en l'écrivant :
> `from defis.donnees_sql import essayer`, puis `essayer("SELECT …")`. Pour voir les tables :
> `python -m defis.donnees_sql`.

> 🕵️ Dans la série 09, une solution juste ne suffit pas : le correcteur compte combien de fois votre
> code relit le catalogue (défi 56), combien de scores il garde en mémoire à la fois (défi 60), et
> chronomètre les autres sur quelques milliers de lignes. Le message dit quelle structure de données
> change tout. Lisez aussi les commentaires « Complexité » du corrigé : c'est la question qui suit
> toujours « votre code marche ? » en entretien.

> ⏳ Le défi 55 est le plus important de la série 08 pour un·e data scientist : compter un événement
> postérieur à la date de prédiction rend un modèle de churn excellent au test et inutile en production.
> Le correcteur place exprès des appels « le jour J » et « dans le futur » dans le journal.

> 🧱 Le défi 48 vous fait écrire une **classe** et non une fonction : le correcteur crée plusieurs
> journaux, change le seuil après coup et vérifie que deux objets ne partagent rien.

> 🧪 Le défi 39 inverse les rôles : c'est **vous** qui écrivez le test, et le correcteur vérifie qu'il
> passe sur la bonne fonction et échoue sur 5 versions boguées. C'est le meilleur entraînement pour
> écrire ensuite vos propres `tests/test_*.py`. La série 11 pousse l'idée plus loin : le correcteur
> lance aussi votre test sur une **autre version juste**, écrite autrement. Un test qui compare des
> flottants avec `==`, recopie le texte exact d'un message ou exige un détail non promis échoue alors :
> il bloquerait un refactoring. Le défi 72 lance votre test dans un dossier jetable et vérifie qu'il
> n'écrit rien ailleurs.

> 💡 Les défis 🔴 réutilisent le code du projet (`churn.model.build_pipeline`) : vous
> apprenez aussi à lire et réutiliser un code existant, compétence clé en entreprise.

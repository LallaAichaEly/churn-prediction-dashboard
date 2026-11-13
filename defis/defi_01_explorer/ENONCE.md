# Défi 01 — Faire parler les données 🔎

**Module** : 1 — Comprendre les données · **Niveau** : 🟢 → 🟡 · **Durée** : 45 à 90 min
**Compétences** : `groupby` / `agg`, lift, piège des petits échantillons, contrôle qualité, tests

> Avant de modéliser quoi que ce soit, un·e data scientist sait répondre à :
> *« Qui part, combien, et peut-on croire ce chiffre ? »* C'est l'objet de ce défi.

## Comment faire

1. Ouvrez `exercice.py`. Quatre fonctions vous attendent, chacune avec sa consigne.
2. Codez une fonction, puis vérifiez :
   ```bash
   pytest defis/defi_01_explorer -q          # tout le défi
   pytest defis/defi_01_explorer -q -k p1    # seulement la partie 1
   ```
3. Tous les tests sont au vert ? Comparez avec `corrige.py` : les commentaires expliquent
   les choix. Une solution différente qui passe les tests est une bonne solution.

## Les 4 parties

| Partie | Fonction | Ce que vous apprenez |
|---|---|---|
| 1 🟢 | `churn_rate_by(df, column)` | Agréger proprement : `groupby` + `agg` nommé + tri |
| 2 🟢 | `lift(df, column)` | Le **lift**, la métrique que les équipes marketing comprennent tout de suite |
| 3 🟡 | `riskiest_segment(df, columns, min_clients)` | Croiser des variables… sans tomber dans le **piège des petits échantillons** |
| 4 🟡 | `check_monthly_charges(df, low, high)` | Contrôle qualité défensif (exercice 1.3 du parcours) |

## Pour aller plus loin (facultatif)

- 🟡 **Brancher votre contrôle dans le vrai code** : appelez `check_monthly_charges` depuis
  `validate()` dans `src/churn/data.py`, et ajoutez un test dans `tests/test_churn.py`.
  Vérifiez que `pytest -q` reste au vert.
- 🔴 **Fibre ou prix ?** Les clients Fibre partent plus (`churn_rate_by(df, "internet_service")`).
  Mais la fibre est aussi plus chère. Découpez `monthly_charges` en tranches avec `pd.cut`,
  puis comparez Fibre et ADSL **à prix égal**. Les deux effets existent-ils séparément ?
  Méfiez-vous des tranches où l'un des deux groupes compte très peu de clients (partie 3 !).
  (Vérifiez votre réponse dans `_churn_logit`.) Ce raisonnement s'appelle *contrôler un
  facteur de confusion* — c'est l'une des compétences les plus recherchées en entretien.

## Questions de réflexion (à noter dans votre `JOURNAL.md`)

1. Le test `test_p2_weighted_average_is_one` affirme que la moyenne des lifts, pondérée par
   la taille des groupes, vaut toujours 1. Démontrez-le en deux lignes.
2. Vous trouvez un segment à 100 % de churn… avec 2 clients. Que répondez-vous au manager
   qui veut lancer une campagne dessus ? Comment choisiriez-vous `min_clients` ?
3. Écrivez **avant** le défi 03 vos 3 hypothèses sur les variables que le modèle jugera
   les plus importantes (exercice 1.4). Vous les vérifierez plus tard.

## ✅ Défi validé quand

- [ ] `pytest defis/defi_01_explorer -q` → **16 passed**
- [ ] Vous avez répondu aux 3 questions dans `JOURNAL.md`
- [ ] Vous avez fait un commit : `git commit -m "Défi 01 : exploration des données"`

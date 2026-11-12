# 🏗️ Projets guidés : construire un vrai programme, de A à Z

Les **défis** entraînent une fonction à la fois. Le **parcours** fait modifier un projet qui existe
déjà. Il manque une marche : **partir d'un dossier presque vide et livrer un programme qui tourne**,
avec plusieurs fichiers qui s'appellent entre eux, des données imparfaites, une ligne de commande et
des tests. C'est ce qu'on attend d'un·e développeur·se dès sa première semaine, et c'est ce que les
projets guidés entraînent.

> 🛠️ **Vous voulez développer, mais ne vous sentez pas encore compétent·e ?** La compétence vient en
> terminant des projets. Celui-ci est découpé en 7 étapes d'une à deux heures, chacune corrigée
> automatiquement avec un message qui explique l'erreur. À la fin, vous avez un outil qui fonctionne,
> que vous pouvez montrer en entretien et expliquer ligne par ligne.

```bash
python scripts/projet.py              # ✅ / ❌ / ⬜ pour chaque étape, et la suivante à faire
python scripts/projet.py --etape 4    # le détail d'une étape
```

| | Projet | Ce que vous construisez | Où |
|---|---|---|---|
| 1 | Tableau de bord du churn | Le modèle qui prédit qui va partir (déjà construit : à lire, à modifier) | racine du dépôt, `PARCOURS.md` |
| **2** | **Plan de rétention** | **Le programme qui décide quoi faire de ces prédictions** | `projets/retention/` |

---

## Projet 2 — Plan de rétention

### Le besoin

Le modèle du projet 1 donne à chaque client une probabilité de partir. La responsable marketing
arrive avec une question simple : *« J'ai 1 500 € ce mois-ci. À qui j'envoie quoi ? »*

Votre programme y répond :

```console
$ python -m projets.retention --clients projets/exemple_clients.csv --budget 1500
Clients analysés  : 200 (0 ligne(s) ignorée(s))
Clients ciblés    : 30
Coût du plan      : 1 496.00 € sur un budget de 1 500.00 €
Gain attendu      : 7 447.43 € (ROI 5.0)
  - APPEL       2
  - ENGAGEMENT  23
  - SUPPORT     5
Plan écrit dans   : plan_retention.csv
```

`projets/exemple_clients.csv` contient 200 clients scorés par le modèle du projet 1, au format de
l'export « Clients à risque » du tableau de bord. Vous pouvez donc aussi lancer votre programme sur
votre propre export.

### Les offres (dans `modeles.py`, déjà écrit)

| Code | Offre | Coût | Efficacité | Réservée à |
|---|---|---|---|---|
| `APPEL` | Appel de courtoisie d'un conseiller | 8 € | 10 % | tous |
| `SUPPORT` | Rappel prioritaire par un technicien | 20 € | 35 % | au moins 3 appels au support |
| `REMISE` | Remise de 20 % pendant 3 mois | 45 € | 30 % | au moins 3 mois d'ancienneté |
| `ENGAGEMENT` | Un mois offert contre un passage à l'annuel | 60 € | 45 % | contrats mensuels |

*Efficacité* : la part des départs que l'offre évite. Les chiffres sont fictifs.

### La règle de calcul

```
perte attendue = proba_churn × facture mensuelle × 12 mois
gain attendu   = efficacité de l'offre × perte attendue − coût de l'offre
rentabilité    = gain attendu / coût de l'offre
```

Pour chaque client, on garde l'offre au plus grand gain (s'il est positif). Puis, le budget étant
limité, on finance d'abord les actions les plus **rentables**.

### L'architecture

```
clients.csv ─► clients.py ─► regles.py ──────► allocation.py ─► rapport.py ─► plan_retention.csv
               étape 1       étape 3           étape 4          étape 5       + résumé à l'écran
                               └─ valeur.py
                                  étape 2

cli.py (étape 6) enchaîne le tout ; mes_tests.py (étape 7) protège l'étape 4 ; modeles.py est fourni.
```

Un fichier par responsabilité : on peut tester, comprendre ou remplacer chaque morceau sans toucher
aux autres. C'est la différence entre un script de 300 lignes et un logiciel.

### Les étapes

| # | Fichier | Niveau | Vous construisez | Vous apprenez |
|---|---|---|---|---|
| 1 | `clients.py` | 🟢 | La lecture du CSV, qui écarte et signale les lignes fausses au lieu de planter | `csv.DictReader`, encodages (BOM d'Excel), conversions, `try`/`except`, `nan`, lignes coupées |
| 2 | `valeur.py` | 🟢 | Perte attendue et gain attendu | Traduire une règle métier, unités, paramètres par défaut, réutiliser une fonction |
| 3 | `regles.py` | 🟡 | L'éligibilité aux offres et le choix de la meilleure | Conditions booléennes, `>=` contre `>`, `max(key=...)`, départager les ex aequo, `None` |
| 4 | `allocation.py` | 🟡 | La répartition du budget | Algorithme glouton, `sorted` contre `.sort()`, `continue` contre `break`, ne pas modifier ses entrées |
| 5 | `rapport.py` | 🟢 | Le résumé chiffré et le fichier du plan | `csv.writer`, formats de nombres, créer un dossier, le cas vide |
| 6 | `cli.py` | 🟡 | La ligne de commande qui assemble tout | `argparse` (type personnalisé), stdout contre stderr, codes de sortie, erreurs prévisibles sans traceback |
| 7 | `mes_tests.py` | 🔴 | Le test qui protège l'étape 4 | Écrire un test qui attrape 7 versions boguées (*mutation testing*) |

**Prérequis** : la série 00 des défis (fonctions, conditions, boucles) ; les séries 06 et 07 aident
beaucoup (exceptions, `argparse`, classes). Aucune notion de machine learning n'est nécessaire.

### Comment travailler

1. Créez une branche : `git checkout -b projet-retention`.
2. Lisez `projets/retention/modeles.py` : ce sont les objets que tout le projet manipule.
3. Pour chaque étape, lisez la docstring (c'est le cahier des charges), codez, puis lancez
   `python scripts/projet.py --etape N`. Le correcteur fabrique des fichiers pièges (export Excel,
   ligne coupée, `nan`, doublons…) et explique ce qui ne va pas.
4. **Commitez à chaque ✅** : `git add projets/retention && git commit -m "Projet 2 : étape N"`.
   Si vous cassez quelque chose plus tard, vous saurez revenir en arrière.
5. Une fois les 7 étapes réussies, lancez votre programme pour de vrai, puis comparez votre code au
   corrigé, `projets/retention_corrige/`, et notez dans `JOURNAL.md` ce que vous feriez autrement.

> 🧩 Les étapes s'appuient les unes sur les autres, comme dans un vrai logiciel : si l'étape 3 appelle
> une fonction de l'étape 2 que vous n'avez pas encore écrite, le correcteur vous le dit.

### Pour aller plus loin (sans correcteur : c'est votre projet)

- **La règle gloutonne n'est pas toujours la meilleure.** Avec 60 € : A coûte 50 € et rapporte 200 €
  (rentabilité 4), B et C coûtent 30 € et rapportent 105 € chacune (3,5). La règle prend A seul
  (200 €) ; B + C rapportaient 210 €. C'est le *problème du sac à dos*. Écrivez
  `allouer_optimal(actions, budget)` par programmation dynamique (budget en euros entiers) et
  mesurez l'écart sur `exemple_clients.csv` : vaut-il la complexité ?
- **Une offre moins chère peut mieux valoir.** On ne garde que la meilleure offre de chaque client ;
  avec un budget serré, l'APPEL à 8 € d'un client peut battre son ENGAGEMENT à 60 €. Modifiez
  l'allocation pour choisir parmi toutes les offres (au plus une par client).
- **De vrais tests pytest.** Créez `tests/test_mon_projet.py` qui appelle votre
  `tester_allouer_budget(allouer_budget)`, puis ajoutez des tests pour `lire_clients` avec
  `tmp_path`. La CI les lancera à chaque push.
- **Brancher le modèle.** Ajoutez une option `--modele models/churn_model.joblib` qui calcule
  `proba_churn` à partir d'un fichier de clients brut, sans passer par le tableau de bord.
- **Une route d'API.** Exposez `POST /plan` dans `api.py` (projet 1) avec votre code.
- **Une vraie commande.** Ajoutez un `pyproject.toml` avec
  `[project.scripts] retention = "projets.retention.cli:main"`, puis `pip install -e .` :
  vous pourrez taper `retention --clients … --budget 1500` depuis n'importe où.

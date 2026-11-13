# 🏗️ Projets guidés : construire un vrai programme, de A à Z

Les **défis** entraînent une fonction à la fois. Le **parcours** fait modifier un projet qui existe
déjà. Il manque une marche : **partir d'un dossier presque vide et livrer un programme qui tourne**,
avec plusieurs fichiers qui s'appellent entre eux, des données imparfaites, une ligne de commande et
des tests. C'est ce qu'on attend d'un·e développeur·se dès sa première semaine, et c'est ce que les
projets guidés entraînent.

> 🛠️ **Vous voulez développer, mais ne vous sentez pas encore compétent·e ?** La compétence vient en
> terminant des projets. Chacun est découpé en 7 étapes d'une à deux heures, chacune corrigée
> automatiquement avec un message qui explique l'erreur. À la fin, vous avez un outil qui fonctionne,
> que vous pouvez montrer en entretien et expliquer ligne par ligne.

```bash
python scripts/projet.py                       # ✅ / ❌ / ⬜ pour chaque étape de chaque projet
python scripts/projet.py --projet 3            # un seul projet, et l'étape suivante à faire
python scripts/projet.py --projet 3 --etape 4  # le détail d'une étape
```

Les trois projets forment une boucle complète, celle qu'on retrouve dans toute équipe data :
**prédire → décider → vérifier que ça a marché**.

| | Projet | Ce que vous construisez | Où |
|---|---|---|---|
| 1 | Tableau de bord du churn | Le modèle qui prédit qui va partir (déjà construit : à lire, à modifier) | racine du dépôt, `PARCOURS.md` |
| **2** | **Plan de rétention** | **Le programme qui décide quoi faire de ces prédictions** | `projets/retention/` |
| **3** | **Mesurer l'effet de la campagne** | **Le programme qui prouve (ou non) que le plan a évité des départs** | `projets/campagne/` |

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
beaucoup (exceptions, `argparse`, classes), et les scénarios 1 à 3 de l'[atelier Git](../atelier_git/README.md)
pour commiter chaque étape sur sa branche. Aucune notion de machine learning n'est nécessaire.

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

---

## Projet 3 — Mesurer l'effet de la campagne

### Le besoin

Le plan du projet 2 a été envoyé. Un mois plus tard, la directrice demande : *« Ça a marché ? Combien de
départs a-t-on vraiment évités, et pour quel prix ? »*

Compter les clients contactés qui sont restés ne répond pas à la question : beaucoup seraient restés de
toute façon. La seule façon honnête de mesurer l'effet d'une action est de garder, **tiré au hasard,
un groupe témoin** qui ne reçoit rien, puis de comparer les deux groupes. C'est un test A/B, la méthode
qu'utilisent les équipes produit, marketing et médicales pour décider sur des preuves.

Votre programme gère toute la campagne, en trois commandes :

```console
$ python -m projets.campagne preparer --plan projets/exemple_plan.csv --base campagne.db --graine 2026-11
Plan lu        : 2000 clients (2000 nouveaux dans campagne.db)
Groupe offre   : 1612 clients à contacter → a_contacter.csv
Groupe témoin  : 388 clients à NE PAS contacter

$ python -m projets.campagne importer --base campagne.db --resultats projets/exemple_resultats.csv
Attention : ligne 702 : customer_id C99999 inconnu, absent de la campagne
Attention : ligne 1202 : parti « ? » illisible (attendu : 1/0, oui/non, true/false)
Attention : ligne 1802 : customer_id C00042 inconnu, absent de la campagne
Résultats enregistrés : 1912 (3 ligne(s) ignorée(s))

$ python -m projets.campagne bilan --base campagne.db
[TOUTES] 1 612 clients avec l'offre, 388 témoins (résultats connus : 1 542 et 368)
  départs : 44,3 % avec l'offre, 56,8 % sans (écart : 12,5 points ; IC 95 % : de 6,9 à 18,1)
  départs évités : ≈ 202, pour 70 331,00 € d'offres (349,03 € par départ évité)
  ✅ l'offre réduit les départs : l'écart ne s'explique pas par le hasard

[APPEL] 172 clients avec l'offre, 48 témoins (résultats connus : 167 et 46)
  départs : 60,5 % avec l'offre, 60,9 % sans (écart : 0,4 point ; IC 95 % : de -15,5 à 16,3)
  départs évités : ≈ 1, pour 1 376,00 € d'offres (2 048,53 € par départ évité)
  🤷 impossible de conclure : l'écart peut venir du hasard (il faudrait plus de clients)
…
```

`projets/exemple_plan.csv` a le format du plan écrit par le projet 2 (2 000 clients) ;
`projets/exemple_resultats.csv` est l'export du CRM un mois plus tard, avec ses imperfections. Les deux
sont simulés, avec la graine `2026-11` : utilisez-la pour retrouver les chiffres ci-dessus.

### Ce que raconte le bilan d'exemple

Lisez-le comme un·e analyste avant de coder : globalement, l'offre évite des départs (l'intervalle
exclut 0). Mais par offre, l'histoire est plus nuancée : ENGAGEMENT et SUPPORT marchent (et SUPPORT
coûte plus de trois fois moins cher par départ évité), alors que l'APPEL et la REMISE ne prouvent rien. Que
proposeriez-vous pour la prochaine campagne ? C'est exactement la question à laquelle votre programme
permet de répondre, chiffres à l'appui.

### L'architecture

```
plan_retention.csv ─► fichiers.lire_plan ─► base.py ──────────────► campagne.db ◄── resultats.py ◄── export du CRM
(projet 2)            (fourni)               étape 2                (SQLite)         étape 3
                                               └─ tirage.py (étape 1)    │
                                                                         ▼
                                    écran ◄── fichiers.formater_ligne ◄── statistiques.py ◄── mesure.py
                                               (fourni)                   étape 5             étape 4

cli.py (étape 6) : les commandes preparer, importer, bilan ; mes_tests.py (étape 7) protège l'étape 5.
modeles.py et fichiers.py sont fournis : à lire, pas à modifier.
```

Nouveauté par rapport au projet 2 : les données vivent dans une **base SQLite**, qui survit d'une
commande à l'autre. La préparation, l'import et le bilan ont lieu à des semaines d'intervalle ; la
base doit rester juste même si une commande est relancée deux fois ou plante au milieu.

### Les étapes

| # | Fichier | Niveau | Vous construisez | Vous apprenez |
|---|---|---|---|---|
| 1 | `tirage.py` | 🟢 | Le tirage du groupe témoin, reproductible | `hashlib.sha256`, octets et base 16, pourquoi ni `random` ni `hash()` |
| 2 | `base.py` | 🟡 | La base de la campagne | `sqlite3`, `PRAGMA foreign_keys`, requêtes paramétrées, `executemany`, transaction (`with conn:`), idempotence (`ON CONFLICT DO NOTHING`, et le piège de `INSERT OR IGNORE`) |
| 3 | `resultats.py` | 🟡 | L'import des résultats du CRM | UPSERT (`DO UPDATE SET … = excluded.…`), clé étrangère et `IntegrityError`, une transaction pour tout un fichier, résister à une panne |
| 4 | `mesure.py` | 🟡 | Les comptages par offre et par groupe | `GROUP BY`, `LEFT JOIN`, `COUNT(*)` contre `COUNT(colonne)`, `COALESCE`, une requête au lieu de N, regrouper des objets |
| 5 | `statistiques.py` | 🟡 | L'effet de l'offre et son intervalle de confiance | Différence de deux proportions, erreur type, `statistics.NormalDist`, « positif » n'est pas « significatif » |
| 6 | `cli.py` | 🟡 | Le programme à trois commandes | Sous-commandes `argparse`, assembler cinq modules, stdout contre stderr, codes de sortie, fermer ses connexions (`closing`) |
| 7 | `mes_tests.py` | 🔴 | Le test qui protège l'étape 5 | Tester du calcul numérique (`math.isclose`), choisir des exemples qui séparent les erreurs |

**Prérequis** : le projet 2 (ou au moins ses étapes 1 et 6) ; la série 10 (SQL) et la série 15
(transactions) des défis aident beaucoup, la série 05 (intervalles de confiance, test A/B) donne
l'intuition statistique. Les formules sont dans les docstrings : pas besoin d'être statisticien·ne.

### Comment travailler

Comme pour le projet 2 : une branche (`git checkout -b projet-campagne`), la docstring de chaque étape
comme cahier des charges, `python scripts/projet.py --projet 3 --etape N` après chaque essai, et un
commit à chaque ✅. Le correcteur fabrique ses propres bases, simule une panne disque au milieu d'un
import et compte vos requêtes SQL : ses messages disent quoi corriger.

Une fois les 7 étapes réussies, lancez les trois commandes ci-dessus, puis enchaînez les projets 2 et 3
sur vos propres données : `python -m projets.retention … --sortie plan.csv` (avec un gros budget, pour
avoir assez de clients), puis `python -m projets.campagne preparer --plan plan.csv …`.

### Pour aller plus loin (sans correcteur : c'est votre projet)

- **Combien de clients faut-il ?** Avant la campagne, calculez la taille d'échantillon nécessaire pour
  détecter un écart de 5 points avec 80 % de puissance. Ajoutez une commande `taille` qui l'affiche.
  Avec 20 % de témoins, la REMISE de l'exemple avait-elle une chance d'être jugée ?
- **Le bon prix par départ évité.** Ajoutez au bilan la valeur des clients gardés (facture × 12 mois,
  comme au projet 2) : quelles offres rapportent plus qu'elles ne coûtent, intervalle compris ?
- **Comparer les offres entre elles.** La différence entre ENGAGEMENT et SUPPORT est-elle
  significative ? Attention aux comparaisons multiples : avec 4 offres et un niveau de 95 %, la
  chance qu'au moins une paraisse « significative » par hasard dépasse 18 %.
- **Une route d'API.** Exposez `GET /campagnes/{graine}/bilan` dans `api.py` en réutilisant `mesure.py`
  et `statistiques.py`, avec la base en dépendance FastAPI (voir la série 12 des défis).
- **Des tests pytest.** `tests/test_ma_campagne.py` : appelez `tester_effet(effet)`, et testez
  `importer_resultats` avec `tmp_path` et une base `":memory:"`.

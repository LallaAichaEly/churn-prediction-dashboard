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

Les cinq projets forment une boucle complète, celle qu'on retrouve dans toute équipe data :
**prédire → décider → agir → vérifier que ça a marché → réapprendre**.

| | Projet | Ce que vous construisez | Où |
|---|---|---|---|
| 1 | Tableau de bord du churn | Le modèle qui prédit qui va partir (déjà construit : à lire, à modifier) | racine du dépôt, `PARCOURS.md` |
| **2** | **Plan de rétention** | **Le programme qui décide quoi faire de ces prédictions** | `projets/retention/` |
| **3** | **Mesurer l'effet de la campagne** | **Le programme qui prouve (ou non) que le plan a évité des départs** | `projets/campagne/` |
| **4** | **Le service des conseillers** | **Le service web avec lequel l'équipe appelle les clients du plan, à plusieurs en même temps** | `projets/service/` |
| **5** | **Réentraîner le modèle sans casser la production** | **Le programme qui, chaque mois, remplace le modèle du projet 1 seulement s'il fait mieux, et sait revenir en arrière** | `projets/reentrainement/` |

Dans le temps, le projet 4 se place au milieu du projet 3 : `preparer` tire le groupe témoin et écrit la
liste des clients à appeler, les conseillers les appellent avec le service du projet 4, puis `bilan`
mesure l'effet un mois plus tard. Les projets 2 et 3 entraînent surtout la donnée et le calcul ; le
projet 4 entraîne le métier de développeur·se : un programme qui tourne en continu et que d'autres utilisent.
Le projet 5 referme la boucle : les clients changent, le modèle du projet 1 vieillit, et il faut le
remplacer sans jamais mettre en production un modèle pire que l'ancien.

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

---

## Projet 4 — Le service des conseillers

### Le besoin

`preparer` (projet 3) a écrit la liste des 1 612 clients à appeler. Quatre conseillers vont les appeler
pendant deux semaines. Pour l'instant, ils se partagent un fichier Excel : deux conseillers appellent le
même client, un résultat saisi le matin disparaît l'après-midi parce qu'une collègue a enregistré sa
copie du fichier par-dessus, et la responsable ne sait jamais où en est l'équipe. Elle demande :
*« Il nous faut une petite application : chaque conseiller voit SA liste, note le résultat de chaque
appel, et moi je vois l'avancement en direct. »*

Vous construisez le service web derrière cette application : une API HTTP (FastAPI) sur une base
SQLite. C'est le premier projet où votre code tourne en continu et répond à d'autres programmes : la
configuration vient de l'environnement, les clés d'accès sont des secrets, et deux écritures peuvent
arriver en même temps.

```console
$ export SERVICE_CLES="alice=cle-alice-0123456789,bruno=cle-bruno-0123456789,chloe=cle-chloe-0123456789,david=cle-david-0123456789"
$ python -m projets.service importer --contacts projets/exemple_contacts.csv
Contacts lus : 1612 (1612 nouveaux, répartis entre 4 conseillers)

conseiller      a_appeler      accepte       refuse  injoignable   traités
alice                 403            0            0            0     0,0 %
bruno                 403            0            0            0     0,0 %
chloe                 403            0            0            0     0,0 %
david                 403            0            0            0     0,0 %
TOTAL                1612            0            0            0     0,0 %

$ python -m projets.service servir
Service des conseillers : http://127.0.0.1:8000/docs (4 conseillers : alice, bruno, chloe, david ; base service.db)
```

Puis, depuis un autre terminal (ou depuis la page `/docs`, qui permet d'essayer chaque route) :

```console
$ curl -s "http://127.0.0.1:8000/contacts?taille=2" -H "X-Cle-API: cle-alice-0123456789"
{"contacts":[{"customer_id":"C10001","offre":"REMISE","conseiller":"alice","statut":"a_appeler","tentatives":0,"commentaire":"","version":1},
             {"customer_id":"C10007", …}],"suivant":"C10007"}

$ curl -s -X PUT http://127.0.0.1:8000/contacts/C10001/appel -H "X-Cle-API: cle-alice-0123456789" \
       -H "Content-Type: application/json" -d '{"statut": "accepte", "version": 1}'
{"customer_id":"C10001","offre":"REMISE","conseiller":"alice","statut":"accepte","tentatives":1,"commentaire":"","version":2}

$ # la même fiche, restée ouverte sur un autre onglet en version 1 :
$ curl -s -X PUT … -d '{"statut": "refuse", "version": 1}'
{"detail":"C10001 a été modifié entre-temps (version actuelle : 2) : rechargez la fiche avant d'enregistrer","version_actuelle":2}

$ curl -s http://127.0.0.1:8000/contacts/C10002 -H "X-Cle-API: cle-alice-0123456789"
{"detail":"C10002 est suivi par un autre conseiller"}
```

Les clés de cet exemple sont faites pour être recopiées. Pour un vrai service, générez-les :
`python -c "import secrets; print(secrets.token_urlsafe(24))"`, et ne les écrivez jamais dans le dépôt.

### Les routes

| Route | Réponse | Erreurs |
|---|---|---|
| `GET /sante` | `{"statut": "ok"}` (sans clé) | |
| `GET /contacts?statut=&apres=&taille=` | une page des contacts du conseiller, et le curseur `suivant` | 401, 422 |
| `GET /contacts/{customer_id}` | la fiche | 401, 403 (pas votre client), 404 |
| `PUT /contacts/{customer_id}/appel` | la fiche à jour (corps : `statut`, `version`, `commentaire`) | 401, 403, 404, 409 (la fiche a changé), 422 |
| `GET /avancement` | le nombre de contacts de chaque conseiller dans chaque statut | 401 |

Chaque requête (sauf `/sante`) porte la clé du conseiller dans l'en-tête `X-Cle-API`.

### L'architecture

```
SERVICE_CLES, SERVICE_BASE … ─► config.py (étape 1) ─► Config
a_contacter.csv ─► fichiers.lire_contacts ─► base.py (étape 2) ─────► service.db (SQLite)
(projet 3)         (fourni)                  répartition équitable       ▲
                                                                          │
application des conseillers ──HTTP──► api.py (étape 5) ──► contacts.py (étape 3) : lire, paginer, compter
             (en-tête X-Cle-API)                       └─► appels.py (étape 4) : enregistrer (verrou optimiste)

cli.py (étape 6) : les commandes importer, avancement, servir ; mes_tests.py (étape 7) protège l'API.
modeles.py et fichiers.py sont fournis : à lire, pas à modifier.
```

### Les étapes

| # | Fichier | Niveau | Vous construisez | Vous apprenez |
|---|---|---|---|---|
| 1 | `config.py` | 🟢 | La configuration lue dans les variables d'environnement | Lire un `Mapping` plutôt que `os.environ` (testable), conversions prudentes, `str.partition`, rassembler toutes les erreurs, ne jamais mettre un secret dans un message |
| 2 | `base.py` | 🟡 | La base et la répartition équitable des contacts | `check_same_thread` (SQLite derrière un serveur web), algorithme glouton (`min` avec une clé tuple), éviter les requêtes N+1, transaction tout ou rien, idempotence |
| 3 | `contacts.py` | 🟡 | Lire une fiche, la liste page par page, l'avancement | Requête construite morceau par morceau sans injection, pagination par curseur (`LIMIT taille + 1`), `ORDER BY`, `GROUP BY` et les statuts à 0 |
| 4 | `appels.py` | 🔴 | L'enregistrement d'un appel | Verrou optimiste (`AND version = ?` et `rowcount`), une transaction sur deux tables, choisir la bonne exception (`Introuvable`, `PasVotreClient`, `Conflit`) |
| 5 | `api.py` | 🟡 | L'API web | FastAPI : `Depends`, une connexion par requête (`yield` et `finally`), `Header`, `secrets.compare_digest` sur des octets, `Query(ge=…, le=…)`, `Literal`, gestionnaires d'exceptions (401, 403, 404, 409, 422), un journal sans secret |
| 6 | `cli.py` | 🟡 | Le programme à trois commandes | Configuration lue une fois, sous-commandes, injection de dépendance (`lancer`) pour tester un programme qui démarre un serveur, échouer au démarrage plutôt qu'à la première requête |
| 7 | `mes_tests.py` | 🔴 | Les tests de l'API | `TestClient`, données de test connues, penser comme un attaquant et comme un collègue distrait : 8 versions boguées à attraper |

**Prérequis** : les projets 2 et 3 (au moins leurs étapes 6) ; les séries 12 (API web), 13 (code sûr),
14 (production) et 15 (base de données) des défis, que ce projet assemble. La série 11 aide pour l'étape 7.

### Comment travailler

Comme pour les projets 2 et 3 : une branche (`git checkout -b projet-service`), la docstring de chaque
étape comme cahier des charges, `python scripts/projet.py --projet 4 --etape N` après chaque essai, et
un commit à chaque ✅. Le correcteur fait écrire une collègue entre votre lecture et votre écriture,
simule un disque plein, compte vos requêtes et vos connexions, essaie des clés presque justes et lit
votre journal : ses messages disent quoi corriger.

Une fois les 7 étapes réussies, lancez `importer` puis `servir` comme ci-dessus, ouvrez
http://127.0.0.1:8000/docs, et jouez la scène du besoin : deux onglets sur la même fiche, une clé
recopiée avec une faute, la fiche d'un collègue. Puis comparez votre code au corrigé,
`projets/service_corrige/`, et notez dans `JOURNAL.md` ce que vous feriez autrement.

### Pour aller plus loin (sans correcteur : c'est votre projet)

- **Ne plus garder les clés en clair.** Stockez dans la base l'empreinte de chaque clé
  (`hashlib.scrypt`, avec un sel par conseiller) plutôt que la clé elle-même, et ajoutez une commande
  `cle --conseiller alice` qui en génère une (`secrets.token_urlsafe`) et ne l'affiche qu'une fois.
- **Rappeler les injoignables.** Ajoutez une date `rappeler_le` : un client injoignable revient dans la
  liste le lendemain à 10 h, heure de Paris (série 14, défi 89).
- **Boucler avec le projet 3.** Une route `GET /export` qui renvoie, en CSV, qui a accepté quelle offre.
  Attention aux commentaires qui commencent par `=` (série 13, défi 83).
- **Une vraie page.** Une page HTML minimale (un tableau et `fetch`) servie par FastAPI, pour que les
  conseillers n'aient pas besoin de curl.
- **Un conteneur.** Un `Dockerfile` qui lance `servir --hote 0.0.0.0`, la base dans un volume, et
  `SERVICE_CLES` passée au lancement du conteneur, jamais écrite dans l'image.
- **Des journaux exploitables.** Configurez `logging` dans `servir` (heure, niveau INFO) et comptez les
  401 : beaucoup de 401 en une minute, c'est quelqu'un qui essaie des clés. Répondez alors 429.
- **De vrais tests pytest.** `tests/test_mon_service.py` qui appelle `tester_service(creer_app)` : la CI
  les lancera à chaque push.

---

## Projet 5 — Réentraîner le modèle sans casser la production

### Le besoin

Le modèle du projet 1 a été entraîné une fois, au printemps. En septembre, la responsable remarque que
les conseillers appellent de plus en plus de clients qui ne partent pas, et que des clients fibre partent
sans avoir été appelés : un concurrent s'est installé dans la région. Le modèle a vieilli.

Jusqu'ici, la personne qui l'avait construit le réentraînait « à la main » dans un notebook et écrasait
`models/churn_model.joblib`. Deux incidents plus tard, la responsable ne veut plus de ça :

- en mars, l'export du CRM est arrivé avec la colonne `churn` vide, lue comme 0 partout. Le modèle
  entraîné dessus a appris que « personne ne part » et a été mis en production : pendant trois
  semaines, aucune alerte ;
- impossible de revenir au modèle d'avant : le fichier avait été écrasé.

Elle demande : *« Un programme qu'on lance chaque mois. Il refuse les données abîmées, entraîne un
nouveau modèle, le compare à celui en production sur les données les plus récentes, et ne le met en
production que s'il est vraiment meilleur. On garde toutes les versions, et on peut revenir en arrière en
une commande. »*

C'est ce qu'on appelle le cycle de vie d'un modèle (une partie du « MLOps ») : le modèle en production
est le **champion**, le modèle tout juste entraîné est le **challenger**, et le **registre** garde
toutes les versions. Le fichier `exemple_lots.csv` contient 12 « photos » mensuelles de 300 clients
(octobre 2025 à septembre 2026) ; le concurrent arrive en avril 2026.

```console
$ python -m projets.reentrainement entrainer --lots projets/exemple_lots.csv --mois 2026-03 --registre registre_modeles
Entraînement : 1500 clients (2025-10 → 2026-02), test : 300 clients (2026-03)

                    AUC   coût/client   alertes
challenger v001   0,837       53,83 €    42,0 %

✅ v001 promue championne :
   - aucun champion en production : le premier modèle utilisable est promu

$ python -m projets.reentrainement entrainer --lots projets/exemple_lots.csv --mois 2026-04 --registre registre_modeles
Entraînement : 1800 clients (2025-10 → 2026-03), test : 300 clients (2026-04)

                    AUC   coût/client   alertes
champion v001     0,828       60,83 €    41,7 %
challenger v002   0,831       62,50 €    41,3 %

❌ v002 enregistrée, mais pas promue (le champion reste v001) :
   - gain d'AUC 0,003 < 0,010 exigé
   - coût par client 1,67 € de plus, au-delà de la tolérance (0,00 €)

$ python -m projets.reentrainement entrainer --lots projets/exemple_lots.csv --mois 2026-09 --registre registre_modeles
Entraînement : 3300 clients (2025-10 → 2026-08), test : 300 clients (2026-09)

                    AUC   coût/client   alertes
champion v001     0,789       73,83 €    40,3 %
challenger v003   0,837       59,17 €    44,0 %

✅ v003 promue championne :
   - gain d'AUC 0,048 >= 0,010
   - coût par client : −14,67 € (tolérance 0,00 €)

$ python -m projets.reentrainement historique --registre registre_modeles
v001  testé sur 2026-03  AUC 0,837  coût 53,83 €/client
v002  testé sur 2026-04  AUC 0,831  coût 62,50 €/client
v003  testé sur 2026-09  AUC 0,837  coût 59,17 €/client  ← champion

$ python -m projets.reentrainement revenir --registre registre_modeles
Retour à v001 (testé sur 2026-03, AUC 0,837) : c'est le modèle en production.
```

Regardez la deuxième commande : le challenger d'avril fait à peine mieux (+0,003 d'AUC, le hasard d'un
mois de 300 clients) et coûte plus cher, il est refusé. En septembre, le vieux champion est tombé de 0,837
à 0,789 sur les clients récents : le challenger, qui a vu la dérive, le remplace.

### L'architecture

```
exemple_lots.csv ─► fichiers.lire_lots ─► lots.py (étape 1) : refuser un lot abîmé
                     (fourni)              │
                                           ▼
                                   decoupage.py (étape 2) : train = les mois d'avant, test = le mois choisi
                                           │
                       entrainement.py ◄───┤ (fourni : la recette du projet 1)
                                           ▼
                   evaluation.py (étape 3) : AUC, coût en €, taux d'alerte — challenger ET champion, même mois
                                           ▼
                   decision.py (étape 4) : promouvoir ou non, et pourquoi
                                           ▼
                   registre.py (étape 5) : v001.joblib, v002.joblib… + registre.json (empreintes, champions)

cli.py (étape 6) : les commandes entrainer, historique, revenir ; mes_tests.py (étape 7) protège decider.
modeles.py, fichiers.py et entrainement.py sont fournis : à lire, pas à modifier.
```

### Les étapes

| # | Fichier | Niveau | Vous construisez | Vous apprenez |
|---|---|---|---|---|
| 1 | `lots.py` | 🟢 | Le garde-fou qui refuse un lot abîmé | Valider avant d'entraîner : `isna`, `str.fullmatch`, `pd.to_numeric(errors="coerce")`, `isin`, `duplicated` sur deux colonnes, taux par mois ; rassembler TOUTES les erreurs |
| 2 | `decoupage.py` | 🟡 | Le découpage dans le temps | Validation temporelle (pas de `train_test_split`), `pd.Period` et `PeriodIndex`, fenêtre d'historique, fuite temporelle |
| 3 | `evaluation.py` | 🟡 | Les mesures d'un modèle sur un mois | `predict_proba` et `classes_`, ne jamais donner la cible au modèle, AUC (indépendante du seuil) contre coût en euros (dépendant du seuil) |
| 4 | `decision.py` | 🔴 | La règle de promotion | Une règle métier en fonction pure, plancher, gain absolu, comparer des flottants (`0.82 - 0.81 < 0.01`), expliquer chaque décision |
| 5 | `registre.py` | 🔴 | Le registre des modèles | Versionner, empreinte SHA-256, écriture atomique (`os.replace`), vérifier PUIS charger les mêmes octets (un pickle exécute du code), revenir en arrière sans rien effacer |
| 6 | `cli.py` | 🟡 | Le programme à trois commandes | Assembler les étapes dans le bon ordre, re-mesurer le champion sur le mois du challenger, enregistrer même ce qui est refusé, stderr et codes de sortie |
| 7 | `mes_tests.py` | 🔴 | Le test de `decider` | Choisir des cas qui séparent la bonne version de 11 versions boguées plausibles |

**Prérequis** : les projets 2 et 3 (au moins leurs étapes 6), le module 7 du parcours (la dérive), et les
défis des séries 13 (code sûr : défi 84, le pickle), 14 (production : défi 90, l'écriture atomique) et 21
(le temps dans les données : défi 132, la validation walk-forward), que ce projet assemble.

### Comment travailler

Comme pour les projets précédents : une branche (`git checkout -b projet-reentrainement`), la docstring de
chaque étape comme cahier des charges, `python scripts/projet.py --projet 5 --etape N` après chaque essai,
et un commit à chaque ✅. Le correcteur glisse un piège à la fois dans des lots fabriqués, interroge votre
`mesurer` avec de faux modèles qui refusent de voir la colonne `churn`, simule un disque plein au milieu
d'un enregistrement, remplace un modèle par un fichier piégé, et rejoue 11 versions boguées de `decider`
pour vous dire laquelle vous avez écrite.

Une fois les 7 étapes réussies, rejouez la scène du besoin : les trois `entrainer` ci-dessus, puis
abîmez une copie du fichier (videz la colonne `churn` d'un mois, écrivez « Mensual ») et vérifiez que le
lot est refusé sans que rien ne change dans le registre. Puis comparez votre code au corrigé,
`projets/reentrainement_corrige/`, et notez dans `JOURNAL.md` ce que vous feriez autrement.

### Pour aller plus loin (sans correcteur : c'est votre projet)

- **Scorer les clients avec le champion.** Une commande `scorer --clients CSV --sortie scores.csv` qui
  charge le champion (vérifié) et écrit `customer_id,proba_churn` : c'est l'entrée du plan de rétention du
  projet 2. La boucle est alors complète, de la donnée brute à l'appel du conseiller.
- **Surveiller avant d'entraîner.** Ajoutez à `entrainer` un rapport de dérive (le PSI de
  `src/churn/monitoring.py`, module 7) entre l'historique et le mois de test, et affichez les variables
  qui ont le plus bougé : c'est souvent là que se trouve l'explication d'une promotion.
- **Un challenger d'une autre famille.** Entraînez aussi un gradient boosting (`HistGradientBoostingClassifier`)
  et laissez `decider` choisir le meilleur des deux challengers. Attention : on compare TOUJOURS sur le
  même mois.
- **Plusieurs mois de test.** Un seul mois de 300 clients, c'est bruité. Mesurez le challenger sur les
  trois derniers mois en « walk-forward » (série 21, défi 132) et exigez qu'il gagne sur chacun.
- **Une fiche par version.** Écrivez à côté de chaque `vNNN.joblib` un `vNNN.md` : données
  d'entraînement (mois, nombre de clients, taux de départ), mesures, décision et raisons. C'est la « model
  card » qu'un auditeur vous demandera.
- **Planifier.** Un `cron` (ou une GitHub Action planifiée) qui lance `entrainer` le 2 de chaque mois, et
  vous envoie un message quand un challenger est refusé trois mois de suite : c'est le signe que la
  recette elle-même doit changer.

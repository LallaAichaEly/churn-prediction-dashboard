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
> Puis la **série 12** (défis 74 à 79) : écrire vous-même les routes d'une API web avec FastAPI
> (lire, chercher, créer, modifier, protéger par une clé, rejouer une requête sans envoyer deux SMS).
> Elle ne demande que la série 00 et le défi 40, et prépare l'exercice 6.4 du parcours.
> Puis la **série 13** (défis 80 à 85) : écrire du code qu'on ne peut pas détourner. Injection SQL, secrets dans
> les journaux, noms de fichiers qui remontent les dossiers, formules cachées dans un export CSV, modèle pickle
> piégé, lien de désinscription falsifié : le correcteur joue l'attaquant contre votre code. Elle ne demande que la
> série 00 et le défi 40.
> Puis la **série 14** (défis 86 à 91) : écrire du code qui tient en production, la nuit, quand personne ne
> regarde. Configuration lue dans les variables d'environnement, journal plutôt que `print`, reprises avec attente
> exponentielle, envoi à 10 h heure de Paris sur un serveur en UTC (changement d'heure compris), fichier jamais laissé
> à moitié écrit, disjoncteur qui coupe un service en panne. Elle ne demande que la série 00 et le défi 40 (et le défi
> 48 pour le dernier).
> Puis la **série 15** (défis 92 à 97) : écrire dans une base de données partagée sans rien perdre. Une offre
> accordée en deux écritures (tout ou rien), la liste des clients page par page, 500 clients chargés en une requête au
> lieu de 501, un schéma qui évolue par migrations, deux conseillers sur la même fiche, plusieurs travailleurs sur la
> même file de SMS. Elle ne demande que la série 00, un peu de SQL (série 05 ou 10) et les requêtes paramétrées du
> défi 80.
> Puis la **série 16** (défis 98 à 103) : faire plusieurs choses à la fois avec `asyncio`. Un programme passe l'essentiel
> de son temps à attendre (un score, un SMS, une page d'API) : demander 8 scores en même temps, ne jamais attendre une
> réponse plus d'une seconde, envoyer 500 SMS 10 par 10, un seul jeton d'accès pour 50 tâches pressées, garder la réponse
> du serveur le plus rapide, et monter un lecteur de pages avec 4 travailleurs qui ne déborde jamais. Elle ne demande que
> la série 00 et le défi 40 (et une classe, défi 48, pour le défi 101). C'est aussi la clé pour lire du code FastAPI.
> Puis la **série 17** (défis 104 à 109) : concevoir ses propres types, pour ne vérifier les données qu'une fois. Un
> `Client` qui refuse d'exister s'il est invalide (`@dataclass(frozen=True)`), un `Contrat` qui ne peut valoir que trois
> choses (`Enum`), un `Montant` en centimes qui ne perd jamais un centime et n'additionne pas des euros et des dollars, un
> envoi de SMS qui ne part qu'en entier (`with`), un portefeuille qu'on parcourt avec `for`, et des règles de ciblage qu'on
> combine avec `&`, `|` et `~` comme dans pandas. Elle ne demande que la série 00, le défi 40 et la classe du défi 48.
> Pour finir, la **série 18** (défis 110 à 115) : rendre un code rapide, preuves à l'appui. Chronométrer sans se tromper
> (`perf_counter`, répéter, garder le minimum), trouver la fonction lente avec `cProfile` (tottime, pas cumtime), le bon
> tarif parmi 50 000 avec `bisect` au lieu de tout parcourir, un cache qui ne grossit pas à l'infini et ne sert pas de
> vieilles réponses (LRU et durée de vie), un export de 10 Go lu avec 1 Mo de mémoire, et une fonction de 40 minutes
> ramenée à une seconde sans changer son résultat d'un octet. Elle ne demande que la série 00, les `set` du défi 45 et le
> décorateur du défi 49.
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
| **12 — Construire une API web** 🌐 | `defis/serie_12.py` | une route `GET` et son paramètre de chemin, répondre **404** plutôt que planter (500) ou renvoyer `null`, **paramètres de requête** typés et bornés (`Query(ge=…, le=…)` → 422), corps JSON validé par un **modèle Pydantic** (`pattern` ancré, `gt`/`le`, `Literal`, `extra="forbid"`), **201** et en-tête `Location`, **409** pour un doublon, **PATCH** partiel (`exclude_unset`, refuser `null`, tout ou rien), clé d'API dans un en-tête avec **`Depends`** (**401 contre 403**, contrôler l'accès avant l'existence, révocation), **idempotence** d'un `POST` avec l'en-tête `Idempotency-Key` (rejeu, panne → **503**, ne mémoriser qu'après succès) | 6, n° 74 à 79 (🟢×2 🟡×3 🔴×1) |
| **13 — Écrire du code sûr** 🔐 | `defis/serie_13.py` | **injection SQL** : requête paramétrée (`?`, `(nom,)` avec la virgule, filtre facultatif sans `= NULL`), **secrets** masqués dans les journaux (noms normalisés, récursion dans les dictionnaires et les listes, copie sans toucher l'original), **traversée de répertoire** (`resolve()`, `is_relative_to` plutôt que `startswith`, chemin absolu, lien symbolique), **injection de formules** dans un export CSV pour Excel (texte seulement, nombres et valeurs manquantes intacts), **désérialisation** : empreinte SHA-256 vérifiée AVANT `pickle.loads`, une seule lecture (TOCTOU), **lien signé** avec HMAC (expiration signée, séparateur, `compare_digest`, toute entrée mal formée → ValueError) | 6, n° 80 à 85 (🟢×2 🟡×3 🔴×1) |
| **14 — Tenir en production** 🚦 | `defis/serie_14.py` | **configuration** lue dans les variables d'environnement (`bool("false")` vaut True, NaN qui passe `seuil < 0 or seuil > 1`, toutes les erreurs d'un coup, ne lire que ce qu'on reçoit), **journalisation** avec `logging` (niveaux, `journal.exception` et sa traceback, un seul résumé INFO, pas de `print` ni de `basicConfig` dans une fonction, pas de données personnelles, `except Exception` pour laisser passer Ctrl+C), **reprises** avec attente exponentielle plafonnée (erreurs passagères seulement, `Retry-After`, relancer la dernière erreur, `dormir` injecté), **fuseaux horaires** avec `zoneinfo` (10 h à Paris en UTC, changement d'heure, `astimezone` contre `replace`, dates naïves refusées), **écriture atomique** (temporaire unique dans le même dossier, fermé puis `os.replace`, nettoyage et `raise` en cas de panne), **disjoncteur** (machine à états fermé/ouvert/semi-ouvert, horloge injectée, `@property`) | 6, n° 86 à 91 (🟢×2 🟡×3 🔴×1) |
| **15 — Enregistrer sans rien perdre** 💾 | `defis/serie_15.py` | **transaction** « tout ou rien » (deux écritures, `with conn:`, `rollback` après N'IMPORTE QUELLE erreur, jamais de transaction laissée ouverte), **pagination par curseur** (`WHERE id > ? ORDER BY id LIMIT taille + 1`, pourquoi pas `OFFSET`), le problème des **requêtes N+1** (`IN (?, ?, …)` fabriqué sans coller de valeur, générateur lu une fois), **migrations** versionnées (`PRAGMA user_version`, `BEGIN` explicite car sqlite3 n'en ouvre pas avant un `CREATE TABLE`, le piège d'`executescript`, version écrite dans la même transaction), **verrou optimiste** (`WHERE id = ? AND version = ?`, `rowcount`, conflit contre client inconnu), **file de tâches** à plusieurs travailleurs (réclamer une tâche sans course, bail expiré, `RETURNING`) | 6, n° 92 à 97 (🟢×2 🟡×3 🔴×1) |
| **16 — Faire plusieurs choses à la fois** ⚡ | `defis/serie_16.py` | **`asyncio.gather`** (lancer d'abord, attendre ensuite ; l'ordre des résultats ; doublons), **délai maximal** avec `wait_for` ou `asyncio.timeout` (l'appel lent est ANNULÉ ; seule la lenteur donne la valeur par défaut, pas une erreur ; un score de 0.0), **`Semaphore`** pour 10 envois à la fois (pas de paquets qui attendent le plus lent, `return_exceptions=True`, garder l'exception elle-même), **`Lock`** et double vérification (50 tâches, une seule demande de jeton ; expiration avec marge ; verrou rendu après une panne), **le premier qui répond** (`create_task`, `as_completed`, une erreur n'est pas une réponse, annuler les perdants), **producteur / travailleurs** (`Queue(maxsize=…)` et contre-pression, signal de fin, `TaskGroup` qui arrête tout à la première erreur au lieu d'un `join()` bloqué pour toujours) | 6, n° 98 à 103 (🟢×2 🟡×3 🔴×1) |
| **17 — Concevoir ses propres types** 🧩 | `defis/serie_17.py` | **`@dataclass(frozen=True)`** et `__post_init__` (TypeError ou ValueError, `True` est un entier, NaN qui passe `< 0`, options gardées en tuple, `object.__setattr__`), **`Enum`** (membres, `@classmethod depuis_texte`, `@property`, `__str__`, le dictionnaire qui devient un membre), **objet valeur** `Montant` (centimes entiers, `Decimal` et pas de float, `InvalidOperation` qui n'est pas une ValueError, `NotImplemented`, `__radd__` pour `sum`, `__rmul__`, `@total_ordering` et le piège d'`order=True`, `-5 // 100`), **gestionnaire de contexte** (`@contextmanager`, `try / except BaseException / else / finally`, ne rien avaler, Ctrl+C), **protocole des collections** (`__len__`, `__contains__`, `__getitem__` et KeyError, `__iter__` qui fabrique un NOUVEL itérateur, ajout pendant un parcours, O(1) sur 20 000 clients), **surcharge d'opérateurs** (`&` `|` `~` qui renvoient une NOUVELLE règle, court-circuit, `__call__`, `__bool__` qui interdit `and` / `or` / `not`) | 6, n° 104 à 109 (🟢×2 🟡×3 🔴×1) |
| **18 — Rendre le code rapide** ⏱️ | `defis/serie_18.py` | **chronométrer** (`perf_counter` et pas `time.time`, répéter et garder le MINIMUM, horloge injectée, ne rien avaler), **profiler** avec `cProfile` et `pstats` (tottime contre cumtime et ncalls, profileur arrêté même si le programme plante : `runcall`), **recherche dichotomique** avec `bisect` (trier une fois, `bisect_right` le jour même d'un changement, l'indice -1 qui renvoie le DERNIER tarif, `sorted` et pas `.sort()`), **cache borné et périmé** (`OrderedDict`, `move_to_end`, `popitem(last=False)`, l'âge part du calcul, `>=` à la seconde près, un cache par fonction, décorateur à paramètres), **lire en flux** un gros CSV (`csv.DictReader`, colonnes par leur nom, virgule entre guillemets, montants vides, `with`, `chunksize` ; pic de mémoire mesuré avec `tracemalloc`), **optimiser sans rien changer** (`Counter`, `set`, calcul sorti de la boucle mais pas quand elle ne tourne pas, tri stable, `sum / len` contre `statistics.mean`, comparaison à l'ancienne version sur 300 cas tirés au hasard) | 6, n° 110 à 115 (🟢×2 🟡×3 🔴×1) |

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

> 🌐 Dans la série 12, le correcteur appelle vos routes comme un vrai programme (`TestClient`), y compris
> avec des requêtes fautives : client inconnu, champ mal orthographié, `null`, clé volée ou révoquée,
> opérateur SMS en panne. Il vérifie le code HTTP ET l'état de la base après coup : une requête refusée
> ne doit rien changer. Pour essayer vos routes en les écrivant : `essayer(routes_lecture(exemple_clients()))`
> (voir le haut de `defis/serie_12.py`).

> 🔐 Dans la série 13, le correcteur joue l'attaquant : il envoie « ' OR '1'='1 » à votre requête, « ../ » et un
> lien symbolique à votre chemin d'export, un fichier pickle piégé à votre chargeur de modèle (son seul effet : noter
> qu'il a été exécuté), et il modifie votre jeton signé caractère par caractère. Il vérifie aussi que les cas honnêtes
> mais inhabituels passent (« D'Almeida », « rapport..v2.csv », une facture négative) : une protection qui bloque les
> vrais utilisateurs n'est pas une bonne protection. Tout reste sur votre ordinateur : n'essayez jamais ces entrées sur
> un service qui ne vous appartient pas.

> 🚦 Dans la série 14, rien n'attend pour de vrai : le correcteur vous passe une fausse fonction `dormir` qui note les
> délais, une horloge qu'il avance à la main (`HorlogeFactice`) et un journal qui retient chaque message avec son niveau.
> Il simule ainsi en une fraction de seconde une nuit de pannes : opérateur SMS surchargé, panne franche, disque plein
> au milieu de l'écriture d'un fichier (il vérifie à chaque ligne que l'ancien fichier est encore intact), passage à
> l'heure d'hiver. C'est la même technique que dans vos propres tests : on n'attend pas 30 vraies secondes pour tester
> un délai de 30 secondes.

> 💾 Dans la série 15, le correcteur ne vous passe pas tout à fait la connexion : il vous passe une connexion « espion »
> (une vraie `sqlite3.Connection`) qui note chaque requête et compte les lignes que votre code lit. Il s'en sert pour
> compter vos requêtes (défi 94), faire tomber en panne la 2e écriture d'une offre (défi 92) ou le serveur au milieu
> d'une migration (défi 95), et surtout pour faire écrire un collègue ou un autre travailleur PILE entre votre SELECT et
> votre UPDATE (défis 96 et 97). Après chaque appel, il regarde la base depuis une AUTRE connexion : c'est ce que verront
> vos collègues. Une écriture sans `commit()` y est invisible, et une transaction laissée ouverte vous sera signalée.

> ⚡ Dans la série 16, rien n'attend pour de vrai : le correcteur exécute vos coroutines sur une boucle asyncio à
> **horloge virtuelle** (quand toutes les tâches attendent, l'horloge saute au prochain réveil). Il mesure ainsi exactement
> la durée qu'aurait prise votre code, et ses faux services notent combien d'appels vous faites EN MÊME TEMPS et lesquels
> vous annulez. Il regarde aussi ce qui tourne encore APRÈS votre retour (un appel lent jamais annulé, un travailleur
> oublié), et il détecte l'**interblocage** : un programme où toutes les tâches attendent et que rien ne réveillera, qui
> en vrai resterait figé sans un message. Pour voir tout ça en écrivant : `essayer(votre_coroutine(...))` affiche la
> durée virtuelle, et prévient s'il reste des tâches oubliées.

> 🧩 Dans la série 17, le correcteur ne reçoit pas une fonction mais votre CLASSE (sauf au défi 107), et s'en sert comme un
> collègue : il crée des objets, les compare, les met dans un `set`, les additionne avec `sum`, les trie, les parcourt deux
> fois de suite, les combine avec `&`… et vérifie que chaque mauvaise utilisation est refusée avec la BONNE erreur (TypeError
> pour un mauvais type, ValueError pour une mauvaise valeur, et un message qui nomme le champ fautif). Au défi 107, il lève
> lui-même des erreurs dans votre bloc `with`, y compris un Ctrl+C simulé, et lit le journal de la fausse passerelle SMS.

> ⏱️ Dans la série 18, le correcteur ne chronomètre presque jamais : un temps dépend de la machine. Il COMPTE. Au défi 110, il
> vous passe une fausse horloge qu'il avance lui-même pendant chaque appel. Aux défis 112 et 115, il vous passe des dates, des
> clients et des listes « espions » qui comptent chaque comparaison et chaque lecture, et il arrête votre fonction dès
> qu'elle dépasse le budget d'une solution efficace : une boucle dans une boucle est arrêtée au bout de quelques clients, et
> le message dit quelle liste vous relisez. Au défi 114, il mesure le pic de mémoire (`tracemalloc`) sur un export, puis sur
> un export deux fois plus long : si la mémoire double, vous chargez tout le fichier. Au défi 115, il compare votre version
> à l'ancienne sur 300 cas tirés au hasard et sur les cas limites qu'elle gérait sans le dire (aucun client, client pile à
> la moyenne, égalités, doublons) : c'est ce qu'on fait avant de remplacer une fonction en production.

> 💡 Les défis 🔴 réutilisent le code du projet (`churn.model.build_pipeline`) : vous
> apprenez aussi à lire et réutiliser un code existant, compétence clé en entreprise.

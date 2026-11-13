"""Série 15 — Enregistrer sans rien perdre : transactions, pagination, N+1, migrations, verrou optimiste, file de tâches.

💾 VOUS VOULEZ DÉVELOPPER ? UNE APPLICATION, C'EST D'ABORD UNE BASE DE DONNÉES QU'ON NE CASSE PAS.

En série 10, vous avez LU une base pour l'analyser. Ici, vous ÉCRIVEZ dedans, comme le fait
l'application de rétention : accorder une offre, afficher la liste des clients page par page,
faire évoluer le schéma, laisser deux conseillers travailler sur la même fiche, et distribuer les
SMS de relance à plusieurs travailleurs sans en envoyer deux fois.

Sur votre ordinateur, vous êtes seul·e et rien ne plante entre deux lignes. En production, dix
personnes écrivent en même temps, le serveur redémarre au milieu d'une mise à jour, et la table
des clients passe de 50 à 2 millions de lignes. Les six défis suivent ces six situations :

    92 accorder une offre (tout ou rien)    93 lister les clients page par page
    94 charger les factures sans 500 requêtes    95 faire évoluer le schéma (migrations)
    96 deux conseillers, une seule fiche    97 plusieurs travailleurs, une seule file de SMS

Le fil rouge : la base est partagée. Chaque fonction reçoit une connexion (`conn`), y fait son
travail et la rend PROPRE : rien d'écrit à moitié, aucune transaction laissée ouverte, la connexion
toujours utilisable (ne la fermez pas : elle ne vous appartient pas).

Prérequis : la série 00, les bases du SQL (série 05 ou 10), et les requêtes paramétrées (`?`) du
défi 80 (série 13). Rien à installer : `sqlite3` fait partie de Python.

📌 Ce qu'il faut savoir du module `sqlite3` de Python (réglages par défaut, ceux du correcteur) :

- `conn.execute(sql, parametres)` renvoie un curseur ; `.fetchone()`, `.fetchall()` lisent les
  lignes (des tuples), `.rowcount` dit combien de lignes un UPDATE ou un DELETE a modifiées et
  `.lastrowid` donne l'identifiant de la ligne qu'un INSERT vient de créer ;
- avant un INSERT, un UPDATE ou un DELETE, Python ouvre une TRANSACTION tout seul. Rien n'est
  enregistré pour les autres connexions avant `conn.commit()` ; `conn.rollback()` annule tout ce
  qui a été fait depuis le début de la transaction ; `conn.in_transaction` dit s'il y en a une ouverte ;
- `with conn:` fait `commit()` si le bloc se termine normalement, et `rollback()` s'il lève une
  erreur (attention : ça ne FERME pas la connexion, et ça n'ouvre pas de transaction tout seul) ;
- en revanche, Python n'ouvre PAS de transaction avant un CREATE TABLE ou un ALTER TABLE : il faut
  l'ouvrir vous-même avec `conn.execute("BEGIN")` (défi 95).

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_15 import *
    >>> conn = base_exemple()                              # 8 clients, 3 campagnes, 6 tâches…
    >>> afficher(conn, "SELECT * FROM budget")
    >>> accorder_offre(conn, "C001", "automne", 50)        # défi 92
    >>> page_clients(conn, taille=3)                       # défi 93
    >>> lea, hugo = deux_connexions()                      # deux conseillers sur la même base (96, 97)

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py --defi 92` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_15.py`.

Les outils communs (base_exemple, afficher, ConflitDeVersion…) sont dans `defis/persistance.py`,
déjà importés ci-dessous : vous n'avez pas besoin d'ouvrir ce fichier.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import sqlite3  # noqa: F401 — sqlite3.IntegrityError, sqlite3.connect pour essayer
from typing import Iterable, Sequence

from defis.persistance import (  # noqa: F401 — tout est utile pour essayer vos fonctions
    BAIL,
    MIGRATION_CASSEE,
    MIGRATIONS_EXEMPLE,
    STATUTS,
    ConflitDeVersion,
    afficher,
    base_exemple,
    deux_connexions,
)

# ============================================================================ Transaction


def accorder_offre(conn: sqlite3.Connection, client_id: str, campagne: str, montant: float) -> int:
    """Défi 92 🟢 — Accorder une offre de rétention : deux écritures, tout ou rien.

    Accorder une offre de `montant` euros à un client, c'est DEUX écritures :

    1. ajouter une ligne dans `offres` (client_id, campagne, montant) ;
    2. retirer `montant` du budget de la campagne : `UPDATE budget SET reste = reste - ? WHERE campagne = ?`.

    Si la 1re réussit et la 2e échoue (ou l'inverse), la base MENT : une offre sans dépense, ou un
    budget dépensé pour rien. Une TRANSACTION regroupe les deux : soit les deux sont enregistrées
    (`commit`), soit aucune (`rollback`). Renvoyez l'identifiant de la nouvelle offre (`lastrowid`).

    - `montant` doit être strictement positif ; la campagne doit exister dans `budget` ; son `reste`
      doit couvrir le montant (dépenser EXACTEMENT tout le reste est permis). Sinon : ValueError ;
    - un client a au plus UNE offre (contrainte UNIQUE de la table) : SQLite lève alors une
      `sqlite3.IntegrityError`. Laissez-la remonter (ou levez une ValueError, au choix) ;
    - APRÈS N'IMPORTE QUELLE ERREUR, même une que vous n'aviez pas prévue (le correcteur fait
      tomber en panne la 1re écriture, puis la 2e), la base est exactement comme avant, AUCUNE
      transaction n'est laissée ouverte (`conn.in_transaction` vaut False) et l'erreur remonte ;
    - en cas de succès, tout est validé (`commit`) : une autre connexion voit l'offre et le budget ;
    - les valeurs passent en paramètres (`?`), jamais dans le texte SQL : il existe une campagne
      « noël d'hiver » (série 13, défi 80) ;
    - ne fermez pas `conn`.

        >>> conn = base_exemple()
        >>> accorder_offre(conn, "C001", "automne", 50)
        2
        >>> afficher(conn, "SELECT * FROM budget WHERE campagne = 'automne'")    # 500 - 50
        >>> accorder_offre(conn, "C004", "fidelite", 500)
        ValueError: budget insuffisant pour la campagne 'fidelite' : 120.0 € restants, 500 € demandés
        >>> accorder_offre(conn, "C002", "automne", 20)      # C002 a déjà une offre
        sqlite3.IntegrityError: UNIQUE constraint failed: offres.client_id

    Le piège : un `conn.commit()` ENTRE les deux écritures, ou un `except` qui remonte l'erreur
    sans `rollback()`. Dans ce dernier cas, la 1re écriture reste en attente dans la transaction
    ouverte, invisible… jusqu'au prochain `commit()` de n'importe quel autre code qui utilise la
    connexion : il enregistrera votre moitié d'offre sans le savoir.

    Ce que vous apprenez : une transaction, c'est « tout ou rien » (le A d'ACID : atomicité), et
    c'est vous qui en tracez les bords.

    Indice : `with conn:` fait tout le travail : `commit()` à la sortie normale du bloc,
    `rollback()` si une erreur en sort. Mettez-y la lecture du budget, les vérifications (`raise
    ValueError(...)` à l'intérieur du bloc annule aussi), l'INSERT et l'UPDATE. Le budget :
    `ligne = conn.execute("SELECT reste FROM budget WHERE campagne = ?", (campagne,)).fetchone()`
    (`None` si la campagne n'existe pas).
    """
    raise NotImplementedError


# ============================================================================ Pagination


def page_clients(conn: sqlite3.Connection, apres: str | None = None, taille: int = 20) -> tuple[list[dict], str | None]:
    """Défi 93 🟢 — Lister les clients page par page, sans en sauter ni en montrer deux fois.

    L'écran « Clients » de l'application (et la route `GET /clients` d'une API, série 12) n'affiche
    pas 2 millions de lignes d'un coup : il les montre par pages. Renvoyez un couple
    `(clients, suivant)` :

    - `clients` : au plus `taille` clients, TRIÉS par `id` croissant, dont l'`id` est
      STRICTEMENT plus grand que `apres` (depuis le début si `apres` vaut None). Chaque client est
      un dictionnaire `{"id": ..., "nom": ..., "score": ...}` ;
    - `suivant` : l'`id` du DERNIER client de la page s'il reste au moins un client après lui
      (l'application le renverra comme `apres` pour demander la page suivante), sinon None.
      Quand la dernière page est pleine (40 clients, pages de 20), `suivant` vaut None dès la
      2e page : pas de 3e page vide ;
    - `taille` va de 1 à 100 ; sinon ValueError, sans interroger la base (un `?taille=1000000`
      ne doit pas faire tomber le serveur) ;
    - ne lisez pas toute la table pour n'en garder que 20 lignes : la base ne doit vous renvoyer
      qu'au plus `taille + 1` lignes (le correcteur compte celles que votre code lit) ;
    - `apres` vient de l'adresse demandée par le navigateur : c'est une entrée NON FIABLE.
      Paramètre `?` obligatoire.

        >>> conn = base_exemple()
        >>> page_clients(conn, taille=3)
        ([{'id': 'C001', 'nom': 'Léa Martin', 'score': 0.82}, {'id': 'C002', …}, {'id': 'C003', …}], 'C003')
        >>> page_clients(conn, apres="C006", taille=3)
        ([{'id': 'C007', …}, {'id': 'C008', …}], None)

    Pourquoi pas `LIMIT 20 OFFSET 40` (« sauter 40 lignes ») ? Si un client de la page 1 est
    supprimé pendant que vous lisez la page 2, tout se décale d'un cran : un client n'est jamais
    montré. S'il en arrive un nouveau, un autre est montré deux fois. Et pour la page 10 000, la base
    lit et jette 200 000 lignes. Avec un CURSEUR (« les clients après C040 »), la page suivante
    reste juste quoi qu'il arrive, et l'index de la clé primaire y va directement. Le correcteur
    supprime et ajoute des clients entre deux pages pour le vérifier.

    Ce que vous apprenez : la pagination par curseur (« keyset pagination »), et l'astuce
    `LIMIT taille + 1` : demander UNE ligne de plus dit s'il existe une page suivante, sans `COUNT(*)`.

    Indice : `SELECT id, nom, score FROM clients WHERE id > ? ORDER BY id LIMIT ?` avec
    `(apres, taille + 1)`, et une autre requête sans le WHERE quand `apres` est None. Gardez les
    `taille` premières lignes ; s'il y en avait une de plus, `suivant` est l'id de la dernière gardée.
    Sans `ORDER BY`, SQLite rend les lignes dans l'ordre qui l'arrange : rien ne garantit l'ordre des id.
    """
    raise NotImplementedError


# ============================================================================ Requêtes N+1


def factures_par_client(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, list[dict]]:
    """Défi 94 🟡 — Charger les factures de 500 clients en 1 requête, pas en 501.

    La page « Clients à risque » affiche 500 clients, chacun avec ses factures. Le code naturel :

        for id in ids:
            conn.execute("SELECT mois, montant FROM factures WHERE client_id = ?", (id,))

    C'est le problème « N+1 » : 1 requête pour la liste, puis 1 par client. Sur votre ordinateur,
    500 requêtes prennent 20 ms ; sur un vrai serveur de base de données, à 1 ms d'aller-retour
    réseau chacune, la page met une demi-seconde, et c'est la cause n°1 des applications lentes.

    Renvoyez `{id: [factures]}` pour chaque id de `ids` :

    - chaque facture est un dictionnaire `{"mois": "2026-09", "montant": 89.9}`, et la liste d'un
      client est triée par `mois` croissant ;
    - CHAQUE id demandé est une clé, même s'il n'a aucune facture (liste vide `[]`) ou s'il
      n'existe pas : l'écran n'a pas à tester `if id in resultat` ;
    - un id présent deux fois dans `ids` donne une seule clé, et ses factures ne sont pas doublées ;
    - `ids` peut être une liste, un ensemble ou un GÉNÉRATEUR (qu'on ne peut parcourir qu'une
      fois) ; `ids` vide → `{}` ;
    - AU PLUS 2 requêtes SELECT, que vous demandiez 3 clients ou 500 (le correcteur les compte) ;
    - les id passent en paramètres (`?`) : un id comme « C9' OR '1'='1 » est un client inconnu
      comme un autre, avec une liste vide.

        >>> conn = base_exemple()
        >>> factures_par_client(conn, ["C001", "C004"])
        {'C001': [{'mois': '2026-07', 'montant': 79.9}, {'mois': '2026-08', 'montant': 89.9},
                  {'mois': '2026-09', 'montant': 89.9}],
         'C004': []}

    Ce que vous apprenez : penser en ENSEMBLES plutôt qu'en boucle (une requête pour tous, puis on
    range en Python), et fabriquer une liste de `?` sans jamais coller les valeurs dans le SQL.

    Indice : `ids = list(dict.fromkeys(ids))` (une seule lecture du générateur, doublons retirés,
    ordre gardé). Puis `marques = ", ".join("?" * len(ids))` et
    `f"SELECT client_id, mois, montant FROM factures WHERE client_id IN ({marques}) ORDER BY client_id, mois"`
    avec `ids` en paramètres : le f-string ne colle que des `?`, jamais une valeur. Enfin
    `resultat = {id: [] for id in ids}` et une boucle qui range chaque ligne dans la bonne liste.
    """
    raise NotImplementedError


# ============================================================================ Migrations


def migrer(conn: sqlite3.Connection, migrations: Sequence[Sequence[str]]) -> int:
    """Défi 95 🟡 — Faire évoluer le schéma de la base, une migration à la fois, sans jamais la laisser à moitié.

    Le code évolue (« on ajoute le score », « on crée la table des offres ») et la base doit
    suivre, sur votre ordinateur, sur celui de vos collègues et en production, chacune à son
    rythme. On écrit donc les changements de schéma comme une liste ordonnée de MIGRATIONS, et la
    base retient où elle en est : `PRAGMA user_version` (un entier rangé dans le fichier, 0 pour
    une base neuve). Version k = les k premières migrations sont appliquées.

    `migrations[i]` est une suite d'instructions SQL (des textes) qui fait passer la base de la
    version i à la version i + 1. Appliquez celles qui manquent et renvoyez leur NOMBRE :

    - lisez la version : `conn.execute("PRAGMA user_version").fetchone()[0]` ; appliquez, dans
      l'ordre, seulement les migrations d'indice >= version (une base à jour → 0, rien ne bouge) ;
    - chaque migration est appliquée dans SA PROPRE transaction, AVEC la nouvelle version :
      ses instructions et `PRAGMA user_version = i + 1` sont enregistrées ensemble, ou pas du tout.
      Si le serveur s'arrête entre les deux, la base ne doit jamais dire « version 1 » avec la
      table de la version 2 (au redémarrage, la migration 2 serait rejouée : « table already exists ») ;
    - si une migration échoue : ses instructions déjà passées sont ANNULÉES, les migrations
      précédentes de cet appel RESTENT appliquées, on n'essaie pas les suivantes, la version vaut
      le nombre de migrations réellement appliquées, et l'erreur remonte telle quelle ;
    - une version plus grande que `len(migrations)` → ValueError sans rien toucher : la base est
      plus récente que le code (un collègue a migré avec une version plus récente du programme) ;
    - aucune transaction ouverte à la fin, que tout se passe bien ou non.

        >>> conn = sqlite3.connect(":memory:")
        >>> migrer(conn, MIGRATIONS_EXEMPLE)
        3
        >>> migrer(conn, MIGRATIONS_EXEMPLE)                  # déjà à jour
        0
        >>> migrer(conn, MIGRATIONS_EXEMPLE + [MIGRATION_CASSEE])
        sqlite3.OperationalError: no such table: table_qui_n_existe_pas
        >>> afficher(conn, "PRAGMA user_version")             # toujours 3 ; la table journal n'existe pas

    Les deux pièges de `sqlite3` :
    - Python n'ouvre PAS de transaction avant un CREATE TABLE ou un ALTER TABLE : chaque
      instruction est enregistrée dès qu'elle s'exécute, et `rollback()` n'a rien à annuler.
      Ouvrez-la vous-même : `conn.execute("BEGIN")`. (Les CREATE et ALTER de SQLite, eux, savent
      très bien être annulés dans une transaction.) ;
    - `conn.executescript(...)` commence par valider (COMMIT) la transaction en cours, puis
      enregistre chaque instruction une par une : à éviter ici. Exécutez les instructions une par
      une avec `conn.execute`.
    `PRAGMA` n'accepte pas de paramètre `?` : écrivez `f"PRAGMA user_version = {i + 1}"`, sans
    risque puisque c'est un entier calculé par votre code (pas une saisie).

    Ce que vous apprenez : les migrations versionnées (le principe d'Alembic, de Django et de
    Flyway), et à vérifier ce que « transaction » veut dire pour VOTRE outil avant de s'y fier.

    Indice : `for i in range(version, len(migrations)):`, puis `conn.execute("BEGIN")`, un `try:`
    qui exécute chaque instruction et le PRAGMA, puis `conn.commit()` ; dans
    `except BaseException:`, `conn.rollback()` puis `raise`.
    """
    raise NotImplementedError


# ============================================================================ Verrou optimiste


def changer_statut(conn: sqlite3.Connection, client_id: str, statut: str, version_lue: int) -> int:
    """Défi 96 🟡 — Deux conseillers, une seule fiche : ne jamais écraser le travail d'un collègue.

    Léa ouvre la fiche de C001 (version 1) à 10 h 00. Hugo ouvre la même fiche à 10 h 01, appelle
    le client et enregistre « offre_envoyee ». À 10 h 05, Léa, qui avait encore l'ancienne fiche à
    l'écran, enregistre « contacte » : sans précaution, elle ÉCRASE le travail de Hugo sans que
    personne ne le sache. C'est la « mise à jour perdue ».

    Le VERROU OPTIMISTE : chaque fiche porte un numéro de `version`. On enregistre seulement si la
    version en base est encore celle qu'on a lue, et on l'augmente de 1 en même temps. Sinon, c'est
    que quelqu'un est passé entre-temps : on refuse.

    - `statut` doit être l'un de STATUTS ; sinon ValueError, sans toucher à la base ;
    - si la fiche est encore à la version `version_lue` : `statut` est enregistré, `version`
      augmente de 1, tout est validé (`commit`), et vous renvoyez la NOUVELLE version ;
    - si la version a changé : levez `ConflitDeVersion` (importée ci-dessus), sans rien modifier ;
    - si le client n'existe pas : levez une `KeyError` (ce n'est pas un conflit : il n'y a rien à
      recharger) ;
    - aucune transaction laissée ouverte, valeurs en paramètres `?` (essayez de deviner ce que
      ferait l'id « C0' OR '1'='1 » collé dans un UPDATE…).

        >>> conn = base_exemple()
        >>> changer_statut(conn, "C001", "contacte", version_lue=1)
        2
        >>> changer_statut(conn, "C001", "parti", version_lue=1)       # fiche périmée
        ConflitDeVersion: C001 a changé depuis votre lecture (version 1) : rechargez la fiche

    Le piège : lire la version (SELECT), la comparer en Python, puis écrire (UPDATE … WHERE id = ?).
    Entre votre SELECT et votre UPDATE, un collègue peut écrire : votre comparaison était juste
    une milliseconde trop tôt. Le correcteur fait justement écrire un collègue à ce moment-là.
    La vérification et l'écriture doivent être UNE SEULE instruction : la base, elle, ne laisse
    personne s'intercaler au milieu d'un UPDATE.

    Ce que vous apprenez : les « conditions de course » (race conditions), et la réponse
    classique du web : verrouiller sans rien bloquer, et laisser l'humain trancher en cas de conflit.

    Indice : `curseur = conn.execute("UPDATE clients SET statut = ?, version = version + 1 WHERE id = ? AND version = ?", ...)`.
    `curseur.rowcount` vaut 1 si la ligne a été modifiée, 0 sinon. Si 0 : un SELECT dit
    si le client existe (KeyError) ou non (ConflitDeVersion). N'oubliez ni `commit()` ni `rollback()`.
    """
    raise NotImplementedError


# ============================================================================ File de tâches


def prendre_tache(conn: sqlite3.Connection, travailleur: str, maintenant: float, bail: float = BAIL) -> dict | None:
    """Défi 97 🔴 — Plusieurs travailleurs, une seule file de SMS : chaque tâche est prise UNE fois.

    Les SMS de relance sont rangés dans la table `taches`. Pour aller plus vite, quatre
    programmes identiques (les « travailleurs ») tournent en même temps et piochent dans la file.
    Si deux d'entre eux prennent la même tâche, le client reçoit deux SMS (et se désabonne).

    Prenez UNE tâche pour `travailleur` et renvoyez `{"id": ..., "client_id": ..., "essais": ...}`
    (le nombre d'essais APRÈS votre prise), ou None s'il n'y a rien à prendre :

    - une tâche est DISPONIBLE si son `statut` vaut « a_faire », ou si elle est « en_cours » mais
      que son BAIL a expiré : `pris_a + bail <= maintenant` (le travailleur qui l'avait prise est
      sans doute tombé en panne). Une tâche « faite » n'est jamais reprise ;
    - parmi les disponibles : la plus grande `priorite`, puis, à priorité égale, le plus petit `id` ;
    - la prendre, c'est : `statut` = « en_cours », `travailleur` = vous, `pris_a` = `maintenant`,
      `essais` augmenté de 1. Cette tâche-là seulement, rien d'autre ne change ;
    - tout est validé (`commit`), aucune transaction laissée ouverte, valeurs en paramètres `?`
      (le poste « poste d'Inès » existe) ;
    - `maintenant` est un nombre de secondes : le correcteur l'avance à la main, comme l'horloge du
      défi 91. N'utilisez pas `time.time()`.

        >>> conn = base_exemple()
        >>> prendre_tache(conn, "poste-4", maintenant=1000.0)
        {'id': 2, 'client_id': 'C004', 'essais': 1}            # priorité 5, plus petit id
        >>> prendre_tache(conn, "poste-4", maintenant=1000.0)
        {'id': 3, 'client_id': 'C006', 'essais': 1}
        >>> prendre_tache(conn, "poste-4", maintenant=1000.0)  # C007 : prise à 700, 700 + 300 <= 1000
        {'id': 6, 'client_id': 'C007', 'essais': 2}
        >>> prendre_tache(conn, "poste-4", maintenant=1000.0)
        {'id': 1, 'client_id': 'C001', 'essais': 1}
        >>> prendre_tache(conn, "poste-4", maintenant=1000.0)  # C005, prise à 900 : son bail court encore
        None

    Le piège, encore une course : vous lisez la meilleure tâche disponible (SELECT), un autre
    travailleur la prend, puis vous la prenez à votre tour (UPDATE … WHERE id = ?). Deux SMS. Le
    correcteur fait passer un autre travailleur à ce moment-là. Et s'il vous a pris la meilleure,
    il en reste peut-être d'autres : renvoyer None serait faux.

    Ce que vous apprenez : le motif « réclamer » (claim) d'une file de tâches (Celery, Sidekiq et
    les files « SKIP LOCKED » de PostgreSQL règlent ce même problème), et les baux qui permettent
    à un système de survivre à la panne d'un de ses membres.

    Indice : une boucle. Cherchez la meilleure candidate (`SELECT id FROM taches WHERE <disponible>
    ORDER BY priorite DESC, id LIMIT 1`). Aucune → None. Puis prenez-la en RÉPÉTANT la condition dans
    l'UPDATE : `UPDATE taches SET … WHERE id = ? AND <disponible>`. Si `rowcount` vaut 1, elle est à
    vous : `commit()`, relisez `client_id` et `essais`, renvoyez. S'il vaut 0, quelqu'un a été plus
    rapide : recommencez la boucle. `<disponible>` : `(statut = 'a_faire' OR (statut = 'en_cours' AND pris_a + ? <= ?))`.
    """
    raise NotImplementedError

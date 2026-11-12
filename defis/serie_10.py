"""Série 10 — SQL pour l'analyse : NULL, jointures, fenêtres.

🗄️  LA COMPÉTENCE QU'ON VOUS DEMANDERA LE PREMIER JOUR.

Dans une entreprise, les données du modèle ne sont pas dans un CSV : elles sont dans une base, et on
les interroge en SQL. C'est la compétence la plus citée dans les offres data, et presque toujours
testée en entretien. La série 05 vous a fait écrire deux requêtes ; celle-ci entraîne ce qui
distingue une requête qui « tourne » d'une requête JUSTE. En SQL, une erreur ne fait presque jamais
planter : elle donne un chiffre faux, en silence, que quelqu'un présentera en réunion.

Prérequis : la série 05 (GROUP BY, ROW_NUMBER). Aucune installation : SQLite fait partie de Python.

🗂️  LA BASE (trois tables, comme dans un vrai système d'information) :

    clients   customer_id · contract · region (NULL si inconnue) · signup_date · churn_date (NULL = toujours client)
    factures  invoice_id · customer_id · invoice_date · amount
    appels    call_id · customer_id (NULL = numéro inconnu) · call_date · motif

    Dates au format texte 'AAAA-MM-JJ' : elles se comparent comme du texte ('2026-01-31' < '2026-02-01').
    date('2026-06-30', '-90 days') → '2026-04-01'     strftime('%Y-%m', '2026-06-30') → '2026-06'

🧭  L'ORDRE DANS LEQUEL SQL ÉVALUE UNE REQUÊTE (ce n'est PAS l'ordre d'écriture) :

    1. FROM / JOIN … ON     on assemble les lignes
    2. WHERE                on filtre les lignes, une par une
    3. GROUP BY             on forme les groupes
    4. HAVING               on filtre les groupes (ici, les agrégats sont permis)
    5. fonctions de fenêtre ROW_NUMBER, LAG, SUM(…) OVER (…)
    6. SELECT               on calcule les colonnes et leurs alias (AS)
    7. ORDER BY / LIMIT     on trie, on coupe

    D'où : pas d'agrégat dans WHERE, pas de fenêtre dans WHERE, et une CTE (WITH … AS) dès qu'il faut
    filtrer sur un résultat calculé.

❓  NULL, OU LA LOGIQUE À TROIS VALEURS : NULL veut dire « inconnu ». Toute comparaison avec NULL
    (=, <>, <, IN…) vaut « inconnu », et WHERE ne garde que ce qui est VRAI. Seuls IS NULL et
    IS NOT NULL répondent vrai ou faux. COUNT(x), SUM(x), AVG(x) ignorent les NULL ; COUNT(*) non.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Chaque fonction renvoie le TEXTE d'une requête (une str). Remplacez `raise NotImplementedError`.
2. Essayez-la sur la base d'exemple pendant que vous l'écrivez :
       python -m defis.donnees_sql                           # voir les tables
       >>> from defis.donnees_sql import essayer
       >>> from defis.serie_10 import requete_churn_par_region
       >>> essayer(requete_churn_par_region())
       >>> essayer(requete_appels_recents(), date_extraction="2026-06-30")
3. Lancez `python scripts/defis.py --defi 62` (le numéro du défi) : ✅ ou ❌ avec l'explication.
4. Bloqué·e ? Relisez l'indice, construisez la requête par morceaux (d'abord le FROM et les JOIN,
   regardez les lignes, puis ajoutez le reste), et seulement en dernier recours ouvrez
   `defis/corrige_serie_10.py`.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

# ============================================================================ NULL


def requete_churn_par_region() -> str:
    """Défi 62 🟢 (Parcours 1.2) — Le taux de churn par région, sans se faire piéger par NULL.

    La direction commerciale veut le taux de churn de chaque région. Un client est PARTI quand sa
    `churn_date` est renseignée (NULL = toujours client). Certains clients n'ont pas de région (NULL).

    Renvoyez une requête sur `clients` qui donne exactement les colonnes
    ["region", "clients", "partis", "taux_churn"] :

        region    clients  partis  taux_churn
        Inconnue       33      10       0.303       (sur la base d'exemple)
        Ouest         108      29       0.269
        …

    - les clients sans région forment le groupe 'Inconnue' (pas une case vide) ;
    - taux_churn = partis / clients, en décimal (0.303, pas 0 ni 30) ;
    - du taux le plus élevé au plus faible ; à égalité, par ordre alphabétique de région.

    Ce que vous apprenez : NULL n'est pas une valeur, c'est une absence. `churn_date = NULL` n'est
    jamais vrai, et COUNT(churn_date IS NOT NULL) compte AUSSI les clients restés (0 n'est pas NULL).

    Indice : COALESCE(region, 'Inconnue') remplace NULL par une valeur. COUNT(*) compte les lignes,
    COUNT(churn_date) les dates présentes. AVG(churn_date IS NOT NULL) est la moyenne d'un 0/1.
    """
    raise NotImplementedError


# ============================================================================ Anti-jointure


def requete_actifs_sans_appel() -> str:
    """Défi 63 🟢 — Les clients qui n'ont JAMAIS appelé le support.

    L'équipe marketing veut interroger les clients silencieux : ils ne se plaignent pas, mais on ne
    sait rien d'eux. Renvoyez une requête qui donne la colonne ["customer_id"] des clients :
    - toujours clients (churn_date NULL) ;
    - sans AUCUNE ligne dans `appels` ;
    - une ligne par client, triés par customer_id.

    Attention : la table `appels` contient des appels de numéros inconnus, dont le customer_id est NULL.

    Ce que vous apprenez : l'ANTI-JOINTURE (« ceux qui n'ont pas de… »), une question d'entretien
    classique. L'écriture qui vient en premier, `customer_id NOT IN (SELECT customer_id FROM appels)`,
    renvoie un résultat VIDE dès qu'un seul appel a un customer_id NULL : « x est-il différent
    d'une valeur inconnue ? », réponse inconnue, donc ligne écartée. Sans aucun message d'erreur.

    Indice, deux écritures sûres :
        … FROM clients c LEFT JOIN appels a ON a.customer_id = c.customer_id WHERE … AND a.call_id IS NULL
        … FROM clients c WHERE … AND NOT EXISTS (SELECT 1 FROM appels a WHERE a.customer_id = c.customer_id)
    """
    raise NotImplementedError


# ============================================================================ LEFT JOIN : ON ou WHERE ?


def requete_appels_recents() -> str:
    """Défi 64 🟡 — Une variable du modèle : le nombre d'appels des 90 derniers jours, 0 compris.

    Le modèle de churn est réentraîné chaque mois sur une « photo » de la base à une date passée.
    Renvoyez une requête qui, pour la date du paramètre nommé `:date_extraction`, donne les colonnes
    ["customer_id", "appels_90j"] :

    - une ligne pour CHAQUE client actif ce jour-là : inscrit au plus tard ce jour-là
      (signup_date <= :date_extraction) et pas encore parti (churn_date NULL, ou STRICTEMENT après) ;
    - appels_90j = nombre de ses appels du jour J − 90 INCLUS au jour J EXCLU :
      date(:date_extraction, '-90 days') <= call_date < :date_extraction ;
    - un client sans appel dans la fenêtre apparaît avec 0 (pour le modèle, « 0 appel » est une
      information, souvent la plus importante) ;
    - tri : appels_90j décroissant, puis customer_id.

    Le correcteur exécute votre requête à deux dates différentes :
        pd.read_sql_query(votre_requete, connexion, params={"date_extraction": "2026-06-30"})

    Ce que vous apprenez :
    - dans un LEFT JOIN, une condition sur la table de DROITE va dans le ON : placée dans le WHERE,
      elle élimine les clients sans appel (leurs colonnes d'appel sont NULL) ;
    - après un LEFT JOIN, COUNT(*) compte 1 pour un client sans appel : COUNT(a.call_id) compte 0 ;
    - `A AND B OR C` se lit `(A AND B) OR C` : mettez des parenthèses autour du OR ;
    - le jour J et le futur sont exclus : c'est la fuite temporelle du défi 55, écrite en SQL.

    Indice : FROM clients c LEFT JOIN appels a ON a.customer_id = c.customer_id AND <les dates d'appel>
             WHERE <client actif> GROUP BY c.customer_id ORDER BY …
    """
    raise NotImplementedError


# ============================================================================ Éviter l'explosion des jointures


def requete_bilan_par_contrat() -> str:
    """Défi 65 🟡 — Chiffre d'affaires et appels par contrat, sans compter deux fois.

    Pour le comité de direction : par type de contrat, le nombre de clients (actifs ou partis), le
    chiffre d'affaires total facturé et le nombre total d'appels au support. Colonnes exactes :
    ["contract", "clients", "chiffre_affaires", "appels"], triées par contract.

    - un contrat dont les clients n'ont ni facture ni appel affiche 0 et 0 (pas NULL) ;
    - les appels de numéros inconnus (customer_id NULL) ne comptent pour aucun contrat.

    Ce que vous apprenez : le piège n°1 des tableaux de bord faux. Joindre `factures` ET `appels` à
    `clients` dans la même requête multiplie les lignes : un client avec 12 factures et 3 appels
    donne 12 × 3 = 36 lignes, donc un chiffre d'affaires compté 3 fois. Aucune erreur, juste des
    millions d'euros qui n'existent pas. COUNT(DISTINCT …) répare le nombre de clients, pas les sommes.

    Indice : agrégez d'abord chaque table PAR CLIENT, puis joignez des tables d'une ligne par client :
        WITH ca_client AS (SELECT customer_id, SUM(amount) AS montant FROM factures GROUP BY customer_id),
             appels_client AS (…)
        SELECT c.contract, … FROM clients c LEFT JOIN ca_client f ON … LEFT JOIN appels_client a ON …
    Et COALESCE(SUM(…), 0) : en SQL, la somme de rien vaut NULL.
    """
    raise NotImplementedError


# ============================================================================ Fonctions de fenêtre


def requete_evolution_mensuelle() -> str:
    """Défi 66 🟡 — Le chiffre d'affaires mois par mois : évolution et cumul.

    Le rapport mensuel de la direction financière. Sur la table `factures`, renvoyez une ligne par
    mois, triées par mois, avec les colonnes ["mois", "chiffre_affaires", "evolution_pct", "ca_cumule"] :

        mois     chiffre_affaires  evolution_pct  ca_cumule
        2026-01               300           NULL        300
        2026-02               400          33.33        700      (+100 sur 300)
        2026-03               350         -12.50       1050      (−50 sur 400)

    - mois au format 'AAAA-MM' ;
    - evolution_pct = 100 × (ce mois − mois précédent) / mois précédent, en pourcentage ;
      NULL pour le premier mois (il n'a pas de précédent : « 0 % » serait une affirmation fausse) ;
    - ca_cumule = total depuis le premier mois, mois courant compris.

    Ce que vous apprenez : les FONCTIONS DE FENÊTRE, qui calculent sur d'autres lignes sans les
    regrouper. LAG(x) OVER (ORDER BY mois) lit la valeur du mois précédent ; SUM(x) OVER (ORDER BY mois)
    fait un cumul. Sans ORDER BY dans OVER (), « précédent » n'a pas de sens et le cumul devient le total.

    Indice : une CTE `mensuel` (mois, chiffre_affaires) avec strftime('%Y-%m', invoice_date) et
    GROUP BY, puis les fenêtres sur cette CTE. Et 100.0, pas 100 : si les montants sont des entiers,
    100 * 100 / 300 vaut 33 en SQL.
    """
    raise NotImplementedError


# ============================================================================ Îlots et trous


def requete_series_appels() -> str:
    """Défi 67 🔴 — Les clients qui appellent le support plusieurs mois D'AFFILÉE.

    Un client qui appelle trois mois de suite a un problème qui ne se règle pas : c'est un signal de
    départ très fort. Pour chaque client, calculez la plus longue série de mois CALENDAIRES consécutifs
    avec au moins un appel, et gardez ceux dont la série atteint le paramètre `:mois_min`. Colonnes :
    ["customer_id", "plus_longue_serie"], triées par plus_longue_serie décroissante, puis customer_id.

        appels en nov. 2025, déc. 2025, janv. 2026 (deux fois), févr. 2026  → 4
        appels en janv., mars et avr. 2026                                  → 2 (février manque)
        trois appels en mai 2026                                            → 1 (un seul mois)

    - les appels de numéros inconnus (customer_id NULL) sont ignorés ;
    - décembre et janvier de l'année suivante sont consécutifs.

    Ce que vous apprenez : le problème des « îlots et trous » (gaps and islands), LE classique des
    entretiens SQL avancés. L'astuce : numérotez les mois (année × 12 + mois), puis retranchez leur
    rang ROW_NUMBER() : les mois consécutifs d'un même client donnent la même différence.

        num_mois  24311  24312  24313  24315
        rang          1      2      3      4
        différence 24310  24310  24310  24311   → un îlot de 3 mois, puis un îlot de 1

    Indice, en trois CTE : (1) SELECT DISTINCT customer_id, <num_mois> (DISTINCT : deux appels le même
    mois ne font qu'un mois) ; (2) num_mois - ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY
    num_mois) AS ilot ; (3) COUNT(*) par (customer_id, ilot). Enfin MAX par client et
    HAVING MAX(…) >= :mois_min.
    """
    raise NotImplementedError

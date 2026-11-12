"""Série 10 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_10.py`.

Si votre requête passe les vérifications, elle est juste, même si elle ne ressemble pas à celle-ci :
en SQL, il y a presque toujours plusieurs écritures (jointure ou NOT EXISTS, sous-requête ou CTE).
Comparez surtout la LISIBILITÉ : une requête découpée en CTE (WITH … AS) se relit, se teste étape par
étape et se corrige bien plus vite qu'un bloc de 30 lignes. Notez dans JOURNAL.md le piège qui vous a
eu·e : en SQL, une erreur ne fait presque jamais planter, elle donne un résultat faux en silence.
"""

from __future__ import annotations

# ============================================================================ NULL


def requete_churn_par_region() -> str:
    return """
        SELECT COALESCE(region, 'Inconnue')  AS region,      -- NULL n'est pas une valeur : on la nomme
               COUNT(*)                      AS clients,     -- compte les LIGNES
               COUNT(churn_date)             AS partis,      -- compte les valeurs NON NULL
               AVG(churn_date IS NOT NULL)   AS taux_churn   -- moyenne d'un 0/1 = une proportion
        FROM clients
        GROUP BY COALESCE(region, 'Inconnue')
        ORDER BY taux_churn DESC, region                     -- départage les ex aequo : résultat stable
    """
    # Trois pièges en une requête :
    # - `churn_date = NULL` n'est jamais vrai (ni faux : c'est « inconnu ») → IS NULL / IS NOT NULL ;
    # - COUNT(churn_date IS NOT NULL) compte aussi les 0, car 0 n'est pas NULL → SUM(...) ou COUNT(churn_date) ;
    # - GROUP BY region fait bien un groupe des NULL, mais l'affiche vide : COALESCE lui donne un nom.
    # En pandas, groupby ÉCARTE les NaN par défaut : df.groupby("region", dropna=False) les garde.


# ============================================================================ Anti-jointure


def requete_actifs_sans_appel() -> str:
    return """
        SELECT c.customer_id
        FROM clients AS c
        LEFT JOIN appels AS a ON a.customer_id = c.customer_id
        WHERE c.churn_date IS NULL          -- toujours client
          AND a.call_id IS NULL             -- … et aucun appel n'a trouvé de correspondance
        ORDER BY c.customer_id
    """
    # Écriture équivalente, souvent plus lisible :
    #     WHERE c.churn_date IS NULL
    #       AND NOT EXISTS (SELECT 1 FROM appels AS a WHERE a.customer_id = c.customer_id)
    # Le piège : `customer_id NOT IN (SELECT customer_id FROM appels)`. Dès qu'un appel a un
    # customer_id NULL, « x NOT IN (…, NULL) » vaut « inconnu » pour TOUS les clients : résultat vide,
    # sans erreur. Il suffit d'un appel anonyme en production pour casser la requête.


# ============================================================================ LEFT JOIN : ON ou WHERE ?


def requete_appels_recents() -> str:
    return """
        SELECT c.customer_id,
               COUNT(a.call_id) AS appels_90j                   -- COUNT(*) compterait 1 pour « aucun appel »
        FROM clients AS c
        LEFT JOIN appels AS a
               ON a.customer_id = c.customer_id
              AND a.call_date >= date(:date_extraction, '-90 days')   -- conditions sur la table de DROITE :
              AND a.call_date <  :date_extraction                     -- dans le ON, pour garder les 0
        WHERE c.signup_date <= :date_extraction                       -- conditions sur la table de GAUCHE :
          AND (c.churn_date IS NULL OR c.churn_date > :date_extraction)   -- dans le WHERE (et des parenthèses !)
        GROUP BY c.customer_id
        ORDER BY appels_90j DESC, c.customer_id
    """
    # Avec `WHERE a.call_date >= …`, les clients sans appel récent ont a.call_date NULL : la condition
    # est « inconnue », la ligne disparaît, et le LEFT JOIN redevient un INNER JOIN sans prévenir.
    # Sans parenthèses, `A AND B OR C` se lit `(A AND B) OR C` : AND est prioritaire, comme × sur +.
    # Le jour J est exclu : une journée pas encore terminée donnerait des comptes partiels (défi 55).


# ============================================================================ Éviter l'explosion des jointures


def requete_bilan_par_contrat() -> str:
    return """
        WITH ca_client AS (                       -- étape 1 : UNE ligne par client pour les factures…
            SELECT customer_id, SUM(amount) AS montant
            FROM factures
            GROUP BY customer_id
        ),
        appels_client AS (                        -- … et UNE ligne par client pour les appels
            SELECT customer_id, COUNT(*) AS nombre
            FROM appels
            WHERE customer_id IS NOT NULL
            GROUP BY customer_id
        )
        SELECT c.contract,                        -- étape 2 : des jointures 1 pour 1, sans multiplication
               COUNT(*)                         AS clients,
               COALESCE(SUM(f.montant), 0)      AS chiffre_affaires,
               COALESCE(SUM(a.nombre), 0)       AS appels
        FROM clients AS c
        LEFT JOIN ca_client     AS f ON f.customer_id = c.customer_id
        LEFT JOIN appels_client AS a ON a.customer_id = c.customer_id
        GROUP BY c.contract
        ORDER BY c.contract
    """
    # Joindre directement factures ET appels à clients : un client avec 12 factures et 3 appels donne
    # 12 × 3 = 36 lignes, donc un chiffre d'affaires compté 3 fois et des appels comptés 12 fois.
    # COUNT(DISTINCT …) répare le nombre de clients, mais pas les SOMMES : il faut agréger AVANT.
    # Réflexe : après une jointure, comparez COUNT(*) au nombre de lignes attendu.


# ============================================================================ Fonctions de fenêtre


def requete_evolution_mensuelle() -> str:
    return """
        WITH mensuel AS (
            SELECT strftime('%Y-%m', invoice_date) AS mois,
                   SUM(amount)                     AS chiffre_affaires
            FROM factures
            GROUP BY mois
        )
        SELECT mois,
               chiffre_affaires,
               100.0 * (chiffre_affaires - LAG(chiffre_affaires) OVER chrono)
                     / LAG(chiffre_affaires) OVER chrono          AS evolution_pct,
               SUM(chiffre_affaires) OVER chrono                  AS ca_cumule
        FROM mensuel
        WINDOW chrono AS (ORDER BY mois)          -- une fenêtre nommée, réutilisée trois fois
        ORDER BY mois
    """
    # LAG lit la ligne PRÉCÉDENTE de la fenêtre : sans ORDER BY dans OVER, « précédente » n'a pas de sens.
    # Le premier mois n'a pas de précédent : LAG renvoie NULL, et toute l'expression vaut NULL (pas 0 :
    # « +0 % » affirmerait une stabilité que personne n'a mesurée).
    # SUM(...) OVER (ORDER BY mois) fait un cumul : la fenêtre va du premier mois au mois courant.
    # 100.0 et pas 100 : avec des montants entiers, 100 * 50 / 300 vaut 16 en SQL, pas 16,67.


# ============================================================================ Îlots et trous


def requete_series_appels() -> str:
    return """
        WITH mois_appel AS (                      -- 1. un mois numéroté, une seule fois par client
            SELECT DISTINCT
                   customer_id,
                   CAST(strftime('%Y', call_date) AS INTEGER) * 12
                 + CAST(strftime('%m', call_date) AS INTEGER)    AS num_mois
            FROM appels
            WHERE customer_id IS NOT NULL
        ),
        ilots AS (                                -- 2. mois consécutifs → même « num_mois - rang »
            SELECT customer_id,
                   num_mois - ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY num_mois) AS ilot
            FROM mois_appel
        ),
        series AS (                               -- 3. la longueur de chaque îlot
            SELECT customer_id, COUNT(*) AS longueur
            FROM ilots
            GROUP BY customer_id, ilot
        )
        SELECT customer_id,
               MAX(longueur) AS plus_longue_serie
        FROM series
        GROUP BY customer_id
        HAVING MAX(longueur) >= :mois_min         -- HAVING filtre APRÈS l'agrégation, WHERE avant
        ORDER BY plus_longue_serie DESC, customer_id
    """
    # L'astuce des « îlots » (gaps and islands) : mois 23, 24, 25, 27 et rangs 1, 2, 3, 4 donnent
    # 22, 22, 22, 23. Les mois consécutifs partagent la même différence, le « trou » la fait changer.
    # Année × 12 + mois : décembre (2025 × 12 + 12) et janvier (2026 × 12 + 1) se suivent bien.
    # Sans DISTINCT, deux appels le même mois prennent deux rangs et coupent l'îlot en deux.

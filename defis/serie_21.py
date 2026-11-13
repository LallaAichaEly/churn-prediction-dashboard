"""Série 21 — Le temps qui passe : dates, mois, mouvements, cohortes, validation et revenu.

📅 VOUS VOULEZ DÉVELOPPER ? LA MOITIÉ DES CHIFFRES QU'ON VOUS DEMANDERA SERONT « PAR MOIS ».

Le comité de direction se réunit le 15 avril 2024. La directrice veut cinq chiffres, et les cinq
dépendent du temps :

- « Combien de nouveaux clients chaque mois ? » Le fichier du formulaire écrit les dates à la française ;
- « Combien de clients avons-nous perdus, et quel est notre taux de churn mensuel ? » ;
- « Les clients arrivés en janvier restent-ils mieux que ceux de l'an dernier ? » (une table de cohortes) ;
- « Le nouveau modèle est-il vraiment meilleur ? » Il faut le tester dans le temps, sans tricher ;
- « D'où vient la baisse du revenu mensuel ? » Des départs, des clients qui paient moins, ou des deux ?

Chaque fois, l'erreur facile donne un chiffre plausible… et faux. Un taux de churn à 0 % parce que les
clients toujours abonnés ont disparu du calcul, une rétention qui s'effondre en avril parce que avril
n'est pas fini, un modèle « excellent » qui a appris sur des étiquettes qu'on ne connaissait pas encore.

    128 lire des dates françaises sans inverser jour et mois   129 compter par mois, mois vides compris
    130 les mouvements du mois et le vrai taux de churn        131 la table de rétention par cohorte
    132 valider un modèle dans le temps, sans fuite            133 le pont du revenu mensuel (MRR)

Prérequis : la série 00 et la série 01 (pandas : `groupby`, `value_counts`, filtrer avec un masque). Le
défi 132 prolonge le défi 9 (validation croisée) et le défi 55 (pas de fuite temporelle). Rien à
installer : tout est dans pandas.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- `pd.to_datetime(serie, format="%d/%m/%Y")` lit des dates à la française ; SANS `format=`, pandas
  devine, et devine à l'américaine dès qu'il le peut : « 03/01/2024 » devient le 1er MARS ;
- une date manquante s'appelle `NaT` (« Not a Time »). Comme NaN, elle n'est égale à rien, et TOUTE
  comparaison avec elle vaut False : `fin >= date` et `fin < date` sont fausses toutes les deux. Dans
  cette série, `fin` vaut NaT quand le client est toujours abonné : chaque condition sur `fin` doit dire
  ce qu'elle fait de lui (`fin.isna() | (fin >= date)`) ;
- un MOIS est une valeur : `serie.dt.to_period("M")` donne des `Period` (« 2024-01 ») qui se comparent
  (`p < q`), s'additionnent (`p + 1` est le mois suivant, même en décembre) et se soustraient
  (`(q - p).n` est le nombre de mois entre les deux). `pd.Period("2024-02", freq="M")` en fabrique un,
  `pd.period_range("2024-01", "2024-04", freq="M")` en fabrique une suite (les deux bouts inclus) ;
- `serie.dt.month` vaut 1 pour janvier 2023 ET pour janvier 2024 : ce n'est pas un mois, c'est un
  numéro de mois ;
- `serie.reindex(nouvel_index, fill_value=0)` range une série selon un index donné : les étiquettes
  absentes apparaissent (avec 0), celles en trop disparaissent. C'est ainsi qu'on fait apparaître les
  mois où rien ne s'est passé ;
- 0 et NaN ne disent pas la même chose : 0, c'est « on a regardé, il n'y a rien » ; NaN, c'est « on ne
  sait pas » (un mois pas encore terminé, un taux sans personne au dénominateur).

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_21 import *
    >>> abonnements_exemple()                         # 8 clients, un par piège (lisez sa docstring)
    >>> pd.to_datetime(pd.Series(["03/01/2024"]))[0]  # sans format= : le 1er mars !
    Timestamp('2024-03-01 00:00:00')
    >>> mois = abonnements_exemple()["debut"].dt.to_period("M")
    >>> mois[0], mois[0] + 2, (mois[7] - mois[0]).n
    (Period('2023-11', 'M'), Period('2024-01', 'M'), 4)

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py --defi 128` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_21.py`.

Les tables d'exemple et les générateurs (`abonnements_exemple`, `generer_abonnements`,
`factures_exemple`…) sont dans `defis/abonnements.py`, déjà importés ci-dessous. Chaque appel renvoie une
NOUVELLE table : vous pouvez la modifier sans rien abîmer.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import numpy as np  # noqa: F401 — utile aux défis 130 à 133
import pandas as pd

from defis.abonnements import (  # noqa: F401 — tout est utile pour essayer vos fonctions
    abonnements_exemple,
    dates_saisies,
    factures_exemple,
    generer_abonnements,
    generer_factures,
)

# ============================================================================ Lire des dates


def lire_dates(textes: pd.Series) -> pd.Series:
    """Défi 128 🟢 — Lire des dates tapées à la française, sans jamais inverser le jour et le mois.

    Le formulaire d'inscription enregistre la date de début au format JJ/MM/AAAA, tapée à la main.
    L'ancien script faisait `pd.to_datetime(textes)` : un tiers des clients de janvier est arrivé… en
    mars, octobre ou décembre.

        >>> dates = lire_dates(dates_saisies())
        >>> dates.loc[101], dates.loc[103]
        (Timestamp('2024-01-03 00:00:00'), Timestamp('2024-02-15 00:00:00'))
        >>> dates.loc[[104, 105]].isna().tolist()        # "" et None : pas de date
        [True, True]
        >>> lire_dates(pd.Series(["2024-01-03"]))
        Traceback (most recent call last):
        ValueError: time data "2024-01-03" doesn't match format "%d/%m/%Y" …

    Les règles :

    - le format est JJ/MM/AAAA : le jour d'abord, l'année sur 4 chiffres. « 3/1/2024 » (sans les zéros)
      est accepté, c'est le 3 janvier ;
    - les espaces autour de la date sont retirés (« 15/02/2024 » avec une espace de chaque côté) ;
    - une valeur manquante (None, NaN) ou un texte vide (ou fait d'espaces) donne `NaT` ;
    - TOUT le reste lève une `ValueError` : un autre format (« 2024-01-03 », « 03-01-2024 »,
      « 3/1/24 »), une date qui n'existe pas (« 31/02/2024 »), du texte (« demain »). Jamais NaT en
      silence : une date illisible, c'est une erreur dans le fichier, il faut la voir tout de suite ;
    - le résultat est une série de dates (type `datetime64`), avec le MÊME index et le même nom que
      `textes`. `textes` n'est pas modifiée.

    Ce que vous apprenez : `pd.to_datetime` et son paramètre `format=`, pourquoi `dayfirst=True` ne
    suffit pas, et pourquoi `errors="coerce"` cache les problèmes au lieu de les résoudre.

    Indice : une ligne. `textes.str.strip()` (les valeurs manquantes restent manquantes), puis
    `pd.to_datetime(..., format="%d/%m/%Y")`. Pandas lit "" comme NaT, et lève lui-même la ValueError,
    en citant le texte fautif.
    """
    raise NotImplementedError


# ============================================================================ Compter par mois


def nouveaux_par_mois(debuts: pd.Series, premier: str, dernier: str) -> pd.Series:
    """Défi 129 🟢 — Le nombre de nouveaux clients de chaque mois, mois vides compris.

    La directrice veut une courbe des inscriptions de janvier à mai. L'ancien graphique n'avait que trois
    points : avril et mai, sans aucune inscription, n'apparaissaient pas, et la courbe reliait mars à…
    rien.

        >>> nouveaux_par_mois(abonnements_exemple()["debut"], "2024-01", "2024-05")
        mois
        2024-01    3
        2024-02    1
        2024-03    2
        2024-04    0
        2024-05    0
        Freq: M, Name: nouveaux, dtype: int64

    `debuts` est une série de dates (avec parfois NaT, et parfois une heure : « 31/01 à 23 h 30 » est en
    janvier). Renvoyez une série :

    - indexée par TOUS les mois de `premier` à `dernier`, les deux inclus, dans l'ordre, sous forme de
      `Period` (pas de texte « 2024-01 », qui se trie mal : « 2024-10 » passe avant « 2024-9 ») ; l'index
      s'appelle « mois » ;
    - nommée « nouveaux », avec des ENTIERS : le nombre de dates de chaque mois. Un mois sans inscription
      vaut 0 ;
    - les dates hors de la période, et les NaT, ne comptent pas.

    Le piège : `debuts.dt.month` donne 1 pour janvier 2023 ET pour janvier 2024. Le correcteur vous
    envoie deux années de dates.

    Ce que vous apprenez : `dt.to_period("M")`, `pd.period_range`, et `reindex(..., fill_value=0)` pour
    qu'un calendrier n'ait pas de trou.

    Indice : `mois = pd.period_range(premier, dernier, freq="M", name="mois")`, puis
    `debuts.dt.to_period("M").value_counts()`, rangé selon `mois` avec `reindex` (0 pour les mois
    absents), puis `.astype("int64")` et `.rename("nouveaux")`.
    """
    raise NotImplementedError


def mouvements_mensuels(abonnements: pd.DataFrame, premier: str, dernier: str) -> pd.DataFrame:
    """Défi 130 🟡 — Les mouvements de clients de chaque mois, et le vrai taux de churn mensuel.

    L'ancien tableau de bord affichait un taux de churn de 0,4 % par mois. Le vrai était de 4 % : le
    calcul « oubliait » tous les clients toujours abonnés, car leur date de fin vaut NaT.

        >>> mouvements_mensuels(abonnements_exemple(), "2024-01", "2024-04")
                 actifs_debut  nouveaux  partis  actifs_fin  taux_churn
        mois
        2024-01             2         3       1           4        0.00
        2024-02             4         1       1           4        0.25
        2024-03             4         2       1           5        0.25
        2024-04             5         0       1           4        0.20

    `abonnements` a les colonnes `client`, `debut` et `fin` (la date de résiliation, ou NaT si le client
    est toujours abonné). Tout se raisonne en MOIS (le jour et l'heure ne comptent pas). Pour chaque mois
    m de `premier` à `dernier` (inclus) :

    - `actifs_debut` : les clients déjà là au début du mois, c'est-à-dire arrivés AVANT le mois m (un
      client arrivé le 1er du mois est un nouveau de ce mois-là, pas un client déjà là), et pas partis
      avant : leur fin tombe pendant le mois m ou après… ou n'existe pas (NaT) ;
    - `nouveaux` : les clients dont `debut` tombe dans le mois m ;
    - `partis` : les clients dont `fin` tombe dans le mois m (tous, y compris un client arrivé ET parti
      dans le mois, comme D en janvier) ;
    - `actifs_fin` = `actifs_debut` + `nouveaux` − `partis` : c'est aussi l'`actifs_debut` du mois
      suivant. Si ce n'est pas le cas, quelqu'un est compté deux fois ou oublié ;
    - `taux_churn` : parmi les `actifs_debut`, la part qui est partie pendant le mois. Un client arrivé le
      10 et parti le 25 n'était pas là au début : il n'est NI au dénominateur NI au numérateur. Quand
      personne n'était là au début, le taux n'existe pas : NaN (pas 0) ;
    - un abonnement qui finit avant d'avoir commencé (`fin` < `debut`) est une erreur : `ValueError` ;
    - le résultat a une ligne par mois (index de `Period` nommé « mois », même les mois sans aucun
      mouvement), ces cinq colonnes dans cet ordre, des entiers pour les quatre premières. `abonnements`
      n'est pas modifiée.

    Ce que vous apprenez : stock et flux (ce qu'on a, ce qui entre, ce qui sort), le piège des
    comparaisons avec NaT, et qu'un taux se calcule sur une POPULATION : ceux qui pouvaient partir.

    Indice : `mois_debut = abonnements["debut"].dt.to_period("M")`, idem pour `fin`, puis une boucle sur
    `pd.period_range(premier, dernier, freq="M")`. Pour chaque mois `m`, trois masques :
    `present = (mois_debut < m) & (mois_fin.isna() | (mois_fin >= m))`, `mois_debut == m` et
    `mois_fin == m`. Le taux : `(present & part).sum() / present.sum()`.
    """
    raise NotImplementedError


# ============================================================================ La table de rétention


def table_retention(abonnements: pd.DataFrame, photo) -> pd.DataFrame:
    """Défi 131 🟡 — La table de rétention par cohorte : qui reste, mois après mois ?

    Une COHORTE, c'est l'ensemble des clients arrivés le même mois. La table de rétention dit, pour
    chaque cohorte, la part de ses clients encore abonnés 0, 1, 2… mois plus tard. Elle répond à « les
    clients de janvier restent-ils mieux que ceux de novembre ? », ce qu'un taux global ne dit pas.

        >>> table_retention(abonnements_exemple(), "2024-04-15").round(2)
        age         0     1     2    3    4
        cohorte
        2023-11  1.00  1.00  1.00  1.0  1.0
        2023-12  1.00  1.00  0.00  0.0  NaN
        2024-01  0.67  0.67  0.33  NaN  NaN
        2024-02  1.00  1.00   NaN  NaN  NaN
        2024-03  1.00   NaN   NaN  NaN  NaN

    `photo` est la date de l'extraction des données (un texte « 2024-04-15 » ou un `pd.Timestamp`). Les
    règles :

    - un mois est TERMINÉ si son dernier jour est passé à la date de la photo : le 15 avril, mars est
      terminé, avril ne l'est pas (photo prise le 1er avril à minuit : mars est terminé ; le 31 mars à
      23 h : il ne l'est pas). Le dernier mois terminé est le mois de la photo moins 1 ;
    - les lignes : les cohortes (le mois de `debut`) dont le premier mois est terminé, dans l'ordre, en
      `Period`, index nommé « cohorte ». Les clients arrivés plus tard ne comptent pas ;
    - les colonnes : les âges 0, 1, 2… (des entiers), jusqu'à l'écart en mois entre la plus ancienne
      cohorte et le dernier mois terminé ;
    - la case (c, a) : la part des clients de la cohorte c encore abonnés à la FIN du mois c + a, c'est-à-
      dire toujours abonnés (`fin` vaut NaT) ou partis APRÈS le mois c + a. L'âge 0, c'est la fin du
      premier mois. Le dénominateur est TOUJOURS la taille de la cohorte entière (pas les clients encore
      là le mois d'avant) : la ligne ne peut que descendre ;
    - si le mois c + a n'est pas terminé, la case vaut NaN : on ne sait pas encore. Surtout pas 0 (une
      rétention de 0 %, ce serait « tout le monde est parti »). C'est ce qui donne à la table sa forme de
      triangle.

    Ce que vous apprenez : la table de cohortes (l'outil le plus utilisé pour suivre une rétention),
    l'arithmétique des mois avec `Period`, et la différence entre « 0 » et « pas encore connu ».

    Indice : `dernier = pd.Timestamp(photo).to_period("M") - 1`. Calculez `cohorte` et `mois_fin` (deux
    `dt.to_period("M")`), gardez les clients avec `cohorte <= dernier`. Les âges : `range((dernier -
    premiere).n + 1)`. Créez `pd.DataFrame(np.nan, index=..., columns=...)`, puis remplissez avec deux
    boucles (cohortes, âges), en vous arrêtant dès que `c + a > dernier`.
    """
    raise NotImplementedError


# ============================================================================ Valider dans le temps


def plis_temporels(mois: pd.Series, n_plis: int, horizon: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Défi 132 🟡 — Découper les données pour tester un modèle comme il servira : sur l'avenir.

    Chaque mois, on prend une PHOTO des clients (une ligne par client et par mois), avec une étiquette :
    « ce client part dans les `horizon` mois qui suivent ». Une validation croisée au hasard (défi 9)
    mélange les mois : le modèle s'entraîne sur juin pour prédire mars. Il paraît excellent, puis déçoit
    en production. On teste donc dans le temps : on s'entraîne sur le passé, on teste sur le mois suivant.

    Mais attention au passé « connu » : l'étiquette de la photo du mois s n'est connue qu'au début du mois
    s + `horizon` (il faut attendre de voir si le client part). Pour tester sur le mois t, on se place au
    début du mois t : on n'a le droit de s'entraîner que sur les photos s telles que s + horizon <= t.

        >>> mois = pd.Series(pd.PeriodIndex(["2024-03", "2024-01", "2024-02", "2024-03", "2024-01"], freq="M"),
        ...                  index=[10, 20, 30, 40, 50])
        >>> plis_temporels(mois, n_plis=1, horizon=2)    # test : mars ; entraînement : janvier seulement
        [(array([1, 4]), array([0, 3]))]
        >>> plis_temporels(mois, n_plis=2, horizon=1)    # février (appris sur janvier), puis mars
        [(array([1, 4]), array([2])), (array([1, 2, 4]), array([0, 3]))]

    `mois` donne le mois de la photo de chaque ligne (des `Period`), dans n'importe quel ordre, avec
    n'importe quel index. Renvoyez une liste de `n_plis` couples `(train, test)` :

    - les mois de test sont les `n_plis` DERNIERS mois présents dans les données, dans l'ordre
      chronologique (un pli par mois) ;
    - `test` : les lignes de ce mois ; `train` : les lignes dont le mois s vérifie s + horizon <= t ;
    - des POSITIONS (0 pour la première ligne, 1 pour la deuxième…), en tableaux numpy d'entiers triés,
      pas les étiquettes de l'index : on s'en sert avec `X.iloc[train]` ou `y[train]` ;
    - `ValueError` si `n_plis` ou `horizon` vaut moins de 1, s'il y a moins de `n_plis` mois dans les
      données, ou si un pli n'a aucune ligne d'entraînement (on ne peut pas tester sans avoir appris).

    Ce que vous apprenez : la validation « walk-forward » (ce que fait `TimeSeriesSplit(gap=...)` de
    scikit-learn), le délai entre une observation et son étiquette, et la différence entre position et
    étiquette d'index (`iloc` contre `loc`).

    Indice : `distincts = sorted(mois.unique())`, `positions = np.arange(len(mois))`, puis pour chaque
    mois `t` de `distincts[-n_plis:]` : `positions[(mois == t).to_numpy()]` et
    `positions[(mois <= t - horizon).to_numpy()]`.
    """
    raise NotImplementedError


# ============================================================================ Le pont du revenu mensuel


def pont_mrr(factures: pd.DataFrame) -> pd.DataFrame:
    """Défi 133 🔴 — D'où vient la variation du revenu mensuel ? (le « pont » du MRR)

    Le revenu mensuel récurrent (MRR, « Monthly Recurring Revenue ») a baissé de 5 € en avril. La
    directrice veut savoir pourquoi : des départs ? des clients qui paient moins ? Le PONT du MRR répond
    en décomposant la variation de chaque mois en cinq mouvements :

        >>> pont_mrr(factures_exemple())
                   mrr  nouveau  reactivation  expansion  contraction  perdu
        mois
        2024-01   50.0     50.0           0.0        0.0          0.0    0.0
        2024-02  110.0     50.0           0.0       10.0          0.0    0.0
        2024-03  110.0     25.0           0.0       15.0         10.0   30.0
        2024-04  105.0      0.0          20.0        0.0          0.0   25.0

    `factures` a une ligne par facture : `client`, `mois` (un `Period`) et `montant` (en euros). Les
    lignes sont dans n'importe quel ordre, et un client peut avoir plusieurs lignes le même mois (une
    option facturée à part) : elles s'additionnent. Pour chaque mois m, du premier au dernier mois des
    factures (TOUS les mois, même un mois où personne n'a été facturé), et pour chaque client, on compare
    ce qu'il paie en m (b) à ce qu'il payait le mois CALENDAIRE précédent (a, 0 s'il n'a rien payé) :

    - `mrr` : le total facturé en m ;
    - a = 0 et b > 0 : le client arrive. S'il avait déjà été facturé un jour (avant m), c'est une
      `reactivation` (b) ; sinon c'est un `nouveau` (b). Le premier mois, tout le monde est nouveau ;
    - a > 0 et b > a : `expansion` (b − a) ; a > 0 et 0 < b < a : `contraction` (a − b) ;
    - a > 0 et b = 0 : `perdu` (a) ;
    - tous les montants sont POSITIFS (on lit « perdu : 30 € », pas « −30 € »), et le pont boucle :
      mrr(m) = mrr(m − 1) + nouveau + reactivation + expansion − contraction − perdu ;
    - un montant nul ou négatif (un avoir) n'est pas un abonnement : `ValueError` ;
    - une ligne par mois (index de `Period` nommé « mois »), ces six colonnes dans cet ordre, en
      nombres décimaux. `factures` n'est pas modifiée.

    Les pièges (le correcteur les essaie tous) : `factures.sort_values(...)["montant"].shift()` compare à
    la ligne précédente, qui est celle d'un AUTRE client, ou d'un mois facturé il y a longtemps (un
    client absent en mars est comparé, en avril, à février) ; et un client qui revient n'est pas un
    nouveau client (la directrice le compte dans l'acquisition, et le budget marketing avec).

    Ce que vous apprenez : passer d'une table longue (une ligne par événement) à une GRILLE (un client par
    ligne, un mois par colonne) avec `pivot_table`, décaler dans le temps avec `shift` sur la bonne
    dimension, et vérifier un calcul par une identité comptable.

    Indice : `grille = factures.pivot_table(index="client", columns="mois", values="montant",
    aggfunc="sum", fill_value=0.0)`, puis `reindex(columns=mois_complets, fill_value=0.0)`. Le mois
    précédent : `avant = grille.shift(1, axis=1, fill_value=0.0)`. « Déjà facturé avant » :
    `grille.cumsum(axis=1).shift(1, axis=1, fill_value=0.0) > 0`. Chaque colonne du résultat est alors
    une somme de `grille.where(condition, 0.0)` (ou de `avant`, ou de leur différence).
    """
    raise NotImplementedError

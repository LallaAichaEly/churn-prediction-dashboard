"""Série 21 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_21.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md le client que vous aviez oublié de compter (ou compté deux fois).

Fil rouge de la série : le temps n'est pas une colonne comme les autres. Quatre réflexes :

1. lire les dates avec un format EXPLICITE (`format="%d/%m/%Y"`), et refuser celles qu'on ne sait pas
   lire : pandas devine sinon, et devine à l'américaine (le mois d'abord) ;
2. raisonner en MOIS avec des `Period` (`dt.to_period("M")`) : « janvier 2024 » est une valeur, qui se
   compare, s'additionne (`mois + 1`) et se range dans l'ordre, sans se soucier du 29 février ni de
   l'heure de la dernière facture ;
3. un calendrier n'a pas de trou : les mois sans événement existent et valent 0 (`reindex`,
   `fill_value=0`) ; les mois qu'on n'a PAS ENCORE vus valent NaN, pas 0 ;
4. NaT (« toujours abonné ») n'est égal à rien et toute comparaison avec lui est fausse : chaque
   condition sur une date de fin doit dire explicitement ce qu'elle fait de NaT (`fin.isna() | ...`).

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from defis.abonnements import (  # noqa: F401
    abonnements_exemple,
    dates_saisies,
    factures_exemple,
    generer_abonnements,
    generer_factures,
)

# ============================================================================ Lire des dates


def lire_dates(textes: pd.Series) -> pd.Series:
    # 1. `.str.strip()` retire les espaces autour ; une valeur manquante (None, NaN) reste manquante, et
    #    une chaîne vide (ou faite d'espaces) devient "" : pandas lit "" comme NaT.
    # 2. `format=` EXPLICITE : sans lui, pandas devine le format… à l'américaine (MM/JJ) dès que la
    #    première date le permet, et « 03/01/2024 » devient le 1er mars. `dayfirst=True` n'est qu'une
    #    préférence : il accepte encore « 2024-01-03 ». `format=` exige.
    # 3. PAS `errors="coerce"` : une date illisible deviendrait NaT, c'est-à-dire « pas de date », et
    #    le problème réapparaîtrait trois traitements plus loin, sans qu'on sache d'où il vient. La
    #    ValueError de pandas cite le texte fautif : c'est exactement ce qu'on veut.
    # `pd.to_datetime` garde l'index et le nom de la série.
    return pd.to_datetime(textes.str.strip(), format="%d/%m/%Y")


# ============================================================================ Compter par mois


def nouveaux_par_mois(debuts: pd.Series, premier: str, dernier: str) -> pd.Series:
    # Le calendrier COMPLET d'abord : c'est lui qui donne les lignes du résultat, pas les données.
    mois = pd.period_range(premier, dernier, freq="M", name="mois")
    # `dt.to_period("M")` et pas `dt.month` : le mois 1, c'est janvier 2023 ET janvier 2024.
    # `value_counts` ignore les NaT ; `reindex` ajoute les mois sans nouveau client (fill_value=0) et
    # écarte ceux qui sont hors de la période.
    comptes = debuts.dt.to_period("M").value_counts()
    return comptes.reindex(mois, fill_value=0).astype("int64").rename("nouveaux")


COLONNES_MOUVEMENTS = ["actifs_debut", "nouveaux", "partis", "actifs_fin", "taux_churn"]


def mouvements_mensuels(abonnements: pd.DataFrame, premier: str, dernier: str) -> pd.DataFrame:
    debut, fin = abonnements["debut"], abonnements["fin"]
    if (fin < debut).any():                          # NaT < date vaut False : les abonnés en cours passent
        fautifs = abonnements.loc[fin < debut, "client"].tolist()
        raise ValueError(f"résiliation avant le début de l'abonnement : {fautifs}")

    # Tout se raisonne en mois : « parti en février » ne dépend ni du jour ni de l'heure.
    mois_debut, mois_fin = debut.dt.to_period("M"), fin.dt.to_period("M")
    lignes = []
    for m in pd.period_range(premier, dernier, freq="M"):
        # Présent au début du mois m : arrivé AVANT m (strictement : un client arrivé le 1er est un
        # nouveau de m), et pas encore parti (fin dans m ou après, OU pas de fin du tout : NaT).
        # Sans `mois_fin.isna()`, tous les clients toujours abonnés disparaissent : NaT >= m vaut False.
        present = (mois_debut < m) & (mois_fin.isna() | (mois_fin >= m))
        arrive, part = mois_debut == m, mois_fin == m
        actifs = int(present.sum())
        # Le taux de churn d'une POPULATION : parmi ceux qui étaient là au début, combien sont partis.
        # Un client arrivé le 10 et parti le 25 n'était pas au dénominateur : il n'est pas au numérateur.
        # Personne au début : le taux n'existe pas (NaN), ce n'est pas 0 %.
        taux = float((present & part).sum() / actifs) if actifs else np.nan
        lignes.append((actifs, int(arrive.sum()), int(part.sum()), actifs + int(arrive.sum()) - int(part.sum()), taux))

    index = pd.period_range(premier, dernier, freq="M", name="mois")
    resultat = pd.DataFrame(lignes, index=index, columns=COLONNES_MOUVEMENTS)
    return resultat.astype({c: "int64" for c in COLONNES_MOUVEMENTS[:4]} | {"taux_churn": "float64"})


# ============================================================================ La table de rétention


def table_retention(abonnements: pd.DataFrame, photo) -> pd.DataFrame:
    # Le dernier mois TERMINÉ à la date de la photo : le mois de la photo est en cours (même le 31 à
    # 23 h), le précédent est fini (même si la photo est prise le 1er à minuit).
    dernier = pd.Timestamp(photo).to_period("M") - 1
    cohorte = abonnements["debut"].dt.to_period("M")
    mois_fin = abonnements["fin"].dt.to_period("M")
    garde = cohorte <= dernier                       # une cohorte dont le 1er mois n'est pas fini : rien à dire
    cohorte, mois_fin = cohorte[garde], mois_fin[garde]

    cohortes = pd.PeriodIndex(sorted(cohorte.unique()), freq="M", name="cohorte")
    if len(cohortes) == 0:
        return pd.DataFrame(index=cohortes, columns=pd.RangeIndex(0, name="age"), dtype="float64")
    ages = range((dernier - cohortes[0]).n + 1)      # (mars - novembre).n vaut 4 : des mois CALENDAIRES

    table = pd.DataFrame(np.nan, index=cohortes, columns=pd.Index(list(ages), name="age"))
    for c in cohortes:
        fins = mois_fin[cohorte == c]
        taille = len(fins)                           # le dénominateur reste la cohorte ENTIÈRE
        for a in ages:
            m = c + a
            if m > dernier:                          # pas encore vu : NaN (le « triangle »), surtout pas 0
                break
            # Encore là à la fin du mois m : toujours abonné (NaT), ou parti APRÈS le mois m.
            table.loc[c, a] = float((fins.isna() | (fins > m)).sum() / taille)
    return table


# ============================================================================ Valider dans le temps


def plis_temporels(mois: pd.Series, n_plis: int, horizon: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_plis < 1 or horizon < 1:
        raise ValueError(f"n_plis et horizon doivent valoir au moins 1 (reçu : {n_plis}, {horizon})")
    distincts = sorted(mois.unique())                # les mois PRÉSENTS, dans l'ordre du calendrier
    if len(distincts) < n_plis:
        raise ValueError(f"{n_plis} plis demandés, mais seulement {len(distincts)} mois dans les données")

    # Des POSITIONS (0 à n-1), pas les étiquettes de l'index : `X.iloc[train]` et `y[train]` (numpy)
    # marchent quel que soit l'index, et l'ordre des lignes n'a pas besoin d'être chronologique.
    positions = np.arange(len(mois))
    plis = []
    for t in distincts[-n_plis:]:
        test = positions[(mois == t).to_numpy()]
        # L'étiquette de la photo du mois s (« part dans les `horizon` mois ») n'est connue qu'au début
        # du mois s + horizon. Au début du mois t, on n'a le droit qu'aux photos s avec s + horizon <= t.
        # `mois < t` serait une FUITE : le modèle apprendrait sur des étiquettes qu'on n'a pas encore.
        train = positions[(mois <= t - horizon).to_numpy()]
        if len(train) == 0:
            raise ValueError(f"aucune donnée d'entraînement pour tester le mois {t} (horizon {horizon})")
        plis.append((train, test))
    return plis


# ============================================================================ Le pont du revenu mensuel

COLONNES_MRR = ["mrr", "nouveau", "reactivation", "expansion", "contraction", "perdu"]


def pont_mrr(factures: pd.DataFrame) -> pd.DataFrame:
    if (factures["montant"] <= 0).any():
        raise ValueError("un montant nul ou négatif n'est pas un abonnement (un avoir se traite à part)")

    # 1. UNE case par client et par mois du calendrier COMPLET : deux lignes le même mois s'additionnent
    #    (aggfunc="sum"), un mois sans facture vaut 0 (fill_value, puis reindex pour les mois où
    #    PERSONNE n'a été facturé).
    mois = pd.period_range(factures["mois"].min(), factures["mois"].max(), freq="M", name="mois")
    grille = factures.pivot_table(index="client", columns="mois", values="montant", aggfunc="sum", fill_value=0.0)
    grille = grille.reindex(columns=mois, fill_value=0.0)

    # 2. Le mois PRÉCÉDENT du calendrier, client par client : un décalage d'une COLONNE. Un `shift()`
    #    sur la table longue comparerait à la ligne précédente : celle d'un autre client, ou d'un mois
    #    facturé il y a longtemps.
    avant = grille.shift(1, axis=1, fill_value=0.0)
    deja_client = grille.cumsum(axis=1).shift(1, axis=1, fill_value=0.0) > 0   # facturé AVANT ce mois

    arrive = (avant == 0) & (grille > 0)
    pont = pd.DataFrame({
        "mrr": grille.sum(),
        "nouveau": grille.where(arrive & ~deja_client, 0.0).sum(),
        "reactivation": grille.where(arrive & deja_client, 0.0).sum(),     # un retour n'est pas un nouveau
        "expansion": (grille - avant).where((avant > 0) & (grille > avant), 0.0).sum(),
        "contraction": (avant - grille).where((grille > 0) & (grille < avant), 0.0).sum(),
        "perdu": avant.where((avant > 0) & (grille == 0), 0.0).sum(),     # en positif, comme les autres
    })
    pont.index = mois
    return pont[COLONNES_MRR].astype("float64")


# ============================================================================ Les tests qu'on écrirait


def test_les_dates_se_lisent_le_jour_d_abord():
    dates = lire_dates(dates_saisies())
    assert dates.loc[101] == pd.Timestamp("2024-01-03")              # le 3 janvier, pas le 1er mars
    assert dates.loc[[104, 105]].isna().all()                        # vide et manquant : NaT
    assert list(dates.index) == list(dates_saisies().index)
    for illisible in ("2024-01-03", "31/02/2024", "3/1/24", "demain"):
        try:
            lire_dates(pd.Series([illisible]))
        except ValueError:
            continue
        raise AssertionError(f"{illisible!r} aurait dû être refusé")


def test_un_mois_sans_nouveau_client_vaut_zero():
    resultat = nouveaux_par_mois(abonnements_exemple()["debut"], "2024-01", "2024-05")
    assert resultat.tolist() == [3, 1, 2, 0, 0]
    assert str(resultat.index[0]) == "2024-01" and resultat.name == "nouveaux"


def test_les_mouvements_bouclent_et_le_taux_porte_sur_les_presents():
    table = mouvements_mensuels(abonnements_exemple(), "2024-01", "2024-04")
    assert table["actifs_debut"].tolist() == [2, 4, 4, 5]
    assert table["nouveaux"].tolist() == [3, 1, 2, 0] and table["partis"].tolist() == [1, 1, 1, 1]
    assert (table["actifs_fin"].iloc[:-1].to_numpy() == table["actifs_debut"].iloc[1:].to_numpy()).all()
    assert table["taux_churn"].tolist() == [0.0, 0.25, 0.25, 0.2]    # D (arrivé et parti en janvier) : 0 %
    vide = mouvements_mensuels(abonnements_exemple(), "2023-01", "2023-01")
    assert np.isnan(vide["taux_churn"].iloc[0])                       # personne : pas de taux


def test_la_retention_est_un_triangle():
    table = table_retention(abonnements_exemple(), "2024-04-15")
    assert list(table.columns) == [0, 1, 2, 3, 4]
    assert table.loc[pd.Period("2024-01", "M")].tolist()[:3] == [2 / 3, 2 / 3, 1 / 3]
    assert table.loc[pd.Period("2024-03", "M")].isna().tolist() == [False, True, True, True, True]


def test_les_plis_ne_regardent_jamais_l_avenir():
    mois = pd.Series(pd.PeriodIndex(["2024-03", "2024-01", "2024-02", "2024-03", "2024-01"], freq="M"),
                     index=[10, 20, 30, 40, 50])
    (train, test), = plis_temporels(mois, n_plis=1, horizon=2)
    assert train.tolist() == [1, 4] and test.tolist() == [0, 3]      # janvier seulement : février + 2 > mars


def test_le_pont_du_mrr_boucle():
    pont = pont_mrr(factures_exemple())
    assert pont["mrr"].tolist() == [50.0, 110.0, 110.0, 105.0]
    assert pont.loc[pd.Period("2024-04", "M"), "reactivation"] == 20.0   # Y revient : pas un nouveau
    variation = pont["nouveau"] + pont["reactivation"] + pont["expansion"] - pont["contraction"] - pont["perdu"]
    assert np.allclose(pont["mrr"].diff().iloc[1:], variation.iloc[1:])

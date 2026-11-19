"""Série 42 (ce que vaut un client) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 254`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées (pandas et `Period` pour compter les passages,
  `effectifs @ P` et `numpy.linalg.solve` avec NumPy, une valeur calculée vers l'avant état par état, une
  actualisation en `(1 + r) ** -t`, des `Counter` de passages…) ;
- les solutions trop lentes (chercher le mois suivant dans toute la liste, recalculer chaque mois depuis le départ,
  boucler jusqu'à ce que la durée de vie ne bouge plus, rappeler `valeur_client` pour chaque mois) sont arrêtées et
  expliquées ;
- les références du correcteur tiennent face à des vérifications indépendantes : une simulation de Monte-Carlo
  des clients, la matrice fondamentale (I − Q)⁻¹ de NumPy, la somme des puissances de P / (1 + r), les vraies
  probabilités du générateur, et des liens entre les défis (une durée de vie est la valeur d'un client qui
  rapporte 1 € par mois sans actualisation ; le délai de rentabilité du coût V_n est n) ;
- les exemples des docstrings, les outils communs et le corrigé sont justes, et l'histoire qu'ils racontent tient
  (734 € au tableur ; 1 013 €, 657 € et 378 € selon l'état ; 279 € pour un client ramené à « Actif » ; 10 mois pour
  rembourser 180 € ; des clients Pro éternels sans lissage, 325 mois avec).
"""

from __future__ import annotations

import doctest
import math
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import defis.corrige_serie_42 as corrige  # noqa: E402
import defis.valeur as valeur  # noqa: E402
from defis.checks_serie_42 import (  # noqa: E402
    _BUGS_DELAI,
    _BUGS_DUREE,
    _BUGS_MATRICE,
    _BUGS_PROJECTION,
    _BUGS_SEGMENTS,
    _BUGS_VALEUR,
    SERIE,
    _ref_delai,
    _ref_duree,
    _ref_matrice,
    _ref_projection,
    _ref_segments,
    _ref_valeur,
)
from defis.valeur import (  # noqa: E402
    ETATS,
    MARGES,
    PARTI,
    TAUX_ANNUEL,
    afficher,
    generer_releves,
    matrice_exemple,
    releves_exemple,
    segments_exemple,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi, series  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}), DEFIS[nom])


def _echec(nom, objet, *morceaux):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for morceau in morceaux:
        assert morceau in resultat.message, resultat.message
    return resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


def _avec_les_gardes(nom, calcul):
    """Une version boguée qui garde les vérifications du corrigé (les erreurs attendues), pour isoler LE bug."""
    def fonction(*args, **kwargs):
        getattr(corrige, nom)(*args, **kwargs)        # lève les mêmes erreurs que le corrigé
        return calcul(*args, **kwargs)
    fonction.__name__ = nom
    return fonction


# --------------------------------------------------------------------------- Défi 254 : matrice_transition


def _matrice_bogue(bug):
    def matrice_transition(releves, etats=ETATS):
        releves = list(releves)
        corrige.matrice_transition(releves, etats)
        return _ref_matrice(releves, etats, bug)
    return matrice_transition


@pytest.mark.parametrize("bug,morceau", [
    ("frontiere", "client suivant"),
    ("trous", "mois MANQUE"),
    ("decembre", "202513"),
    ("ordre", "ORDER BY"),
    ("doublons", "DEUX fois"),
    ("colonnes", "COLONNE"),
    ("total", "TOTAL"),
    ("occurrences", "RELEVÉS"),
    ("vide", "ligne de zéros"),
    ("uniforme", "redeviendrait fidèle"),
])
def test_matrice_nomme_l_erreur(bug, morceau):
    _echec("matrice_transition", _matrice_bogue(bug), morceau)


def test_chaque_bug_de_comptage_a_son_cas():
    """Chaque erreur classique change le résultat d'au moins un petit cas (sinon le correcteur ne la verrait pas)."""
    from defis.checks_serie_42 import _cas_254

    for bug in _BUGS_MATRICE:
        assert any(_ref_matrice(r, kw["etats"], bug) != _ref_matrice(r, kw["etats"]) for r, kw in _cas_254()), bug


def _matrice_pandas(releves, etats=ETATS):
    """Une autre écriture juste : pandas, des numéros de mois, un décalage par client et un tableau croisé."""
    etats = list(etats)
    df = pd.DataFrame(list(releves), columns=["client", "mois", "etat"])
    for mois in df["mois"]:
        if not isinstance(mois, str) or len(mois) != 7 or mois[4] != "-" or not (mois[:4] + mois[5:]).isdigit() \
                or not 1 <= int(mois[5:]) <= 12:
            raise ValueError(f"mois illisible : {mois!r}")
    if not df["etat"].isin(etats).all():
        raise ValueError("état inconnu")
    df = df.drop_duplicates()
    if df.duplicated(["client", "mois"]).any():
        raise ValueError("relevés contradictoires")
    df["numero"] = df["mois"].str[:4].astype(int) * 12 + df["mois"].str[5:].astype(int) - 1
    df = df.sort_values(["client", "numero"])
    suivant = df.groupby("client")[["numero", "etat"]].shift(-1)
    passages = df[suivant["numero"] == df["numero"] + 1].assign(vers=suivant["etat"])
    comptes = pd.crosstab(pd.Categorical(passages["etat"], etats), pd.Categorical(passages["vers"], etats),
                          dropna=False).reindex(index=etats, columns=etats, fill_value=0)
    matrice = {}
    for i in etats:
        total = comptes.loc[i].sum()
        matrice[i] = {j: float(comptes.loc[i, j] / total) if total else float(i == j) for j in etats}
    return matrice


def _matrice_triee(releves, etats=ETATS):
    """Une autre écriture juste : trier les relevés de chaque client, puis comparer chaque relevé au suivant."""
    etats = list(etats)
    par_client: dict = {}
    for client, mois, etat in releves:
        if etat not in etats:
            raise ValueError("état inconnu")
        if not isinstance(mois, str) or len(mois) != 7 or mois[4] != "-":
            raise ValueError("mois illisible")
        annee, m = int(mois[:4]), int(mois[5:])
        if not 1 <= m <= 12:
            raise ValueError("mois illisible")
        par_client.setdefault(client, set()).add((annee, m, etat))
    comptes = Counter()
    for lignes in par_client.values():
        lignes = sorted(lignes)
        if len({(a, m) for a, m, _ in lignes}) != len(lignes):
            raise ValueError("relevés contradictoires")
        for (a1, m1, e1), (a2, m2, e2) in zip(lignes, lignes[1:]):
            if (a2, m2) == ((a1 + 1, 1) if m1 == 12 else (a1, m1 + 1)):
                comptes[e1, e2] += 1
    matrice = {}
    for i in etats:
        total = sum(comptes[i, j] for j in etats)
        matrice[i] = {j: comptes[i, j] / total if total else float(i == j) for j in etats}
    return matrice


def test_matrice_d_autres_ecritures_justes():
    _reussi("matrice_transition", _matrice_pandas)
    _reussi("matrice_transition", _matrice_triee)


def test_matrice_lit_le_flux_une_fois_et_sans_len():
    def deux_lectures(releves, etats=ETATS):
        clients = {client for client, _, _ in releves}
        assert clients is not None
        return corrige.matrice_transition(releves, etats)

    _echec("matrice_transition", deux_lectures, "DEUX fois")

    def avec_len(releves, etats=ETATS):
        if len(releves) == 0:
            return corrige.matrice_transition([], etats)
        return corrige.matrice_transition(releves, etats)

    _echec("matrice_transition", avec_len, "flux")


def test_matrice_quadratique_est_arretee():
    def cherche_partout(releves, etats=ETATS):
        releves = list(dict.fromkeys(releves))
        corrige.matrice_transition(releves, etats)        # les contrôles du corrigé (états, mois, contradictions)
        comptes = {i: dict.fromkeys(etats, 0) for i in etats}
        for client, mois, etat in releves:
            numero = int(mois[:4]) * 12 + int(mois[5:])
            for autre, mois2, etat2 in releves:            # chercher le mois suivant dans TOUTE la liste
                if autre == client and int(mois2[:4]) * 12 + int(mois2[5:]) == numero + 1:
                    comptes[etat][etat2] += 1
        return {i: {j: comptes[i][j] / sum(comptes[i].values()) if sum(comptes[i].values()) else float(i == j)
                    for j in etats} for i in etats}

    _echec("matrice_transition", cherche_partout, "trop lent", "dict")


def test_matrice_refuse_ce_qu_il_faut():
    def sans_controle(releves, etats=ETATS):
        releves = [r for r in releves if r[2] in etats and len(str(r[1])) == 7 and str(r[1])[4] == "-"
                   and 1 <= int(str(r[1])[5:]) <= 12]
        return corrige.matrice_transition(releves, etats)

    _echec("matrice_transition", sans_controle, "Suspendu")

    def mois_avec_datetime(releves, etats=ETATS):     # strptime accepte « 2025-1 »
        from datetime import datetime
        releves = list(releves)
        for _, mois, _ in releves:
            datetime.strptime(str(mois), "%Y-%m")
        return _matrice_triee([(c, f"{m[:4]}-{int(m[5:]):02d}" if isinstance(m, str) else m, e)
                               for c, m, e in releves], etats)

    _echec("matrice_transition", mois_avec_datetime, "2025-1")

    def contradictions_acceptees(releves, etats=ETATS):
        vus = {}
        for client, mois, etat in releves:
            vus[client, mois] = etat                      # le dernier gagne, sans rien dire
        return corrige.matrice_transition([(c, m, e) for (c, m), e in vus.items()], etats)

    _echec("matrice_transition", contradictions_acceptees, "contradictoires")


def test_matrice_forme_et_ordre():
    def sans_parti(releves, etats=ETATS):
        matrice = corrige.matrice_transition(releves, etats)
        return {i: ligne for i, ligne in matrice.items() if any(v for j, v in ligne.items() if j != i) or i != etats[-1]}

    _echec("matrice_transition", sans_parti, "il manque")

    def lignes_creuses(releves, etats=ETATS):
        return {i: {j: p for j, p in ligne.items() if p} for i, ligne in corrige.matrice_transition(releves, etats).items()}

    _echec("matrice_transition", lignes_creuses, "TOUS les états")

    def desordre(releves, etats=ETATS):
        matrice = corrige.matrice_transition(releves, etats)
        return {i: dict(sorted(ligne.items())) for i, ligne in reversed(matrice.items())}

    _echec("matrice_transition", desordre, "ordre")


# --------------------------------------------------------------------------- Défi 255 : projection


def _projection_bogue(bug):
    return _avec_les_gardes("projection", lambda depart, matrice, mois: _ref_projection(depart, matrice, mois, bug))


@pytest.mark.parametrize("bug,morceau", [
    ("transpose", "vecteur COLONNE"),
    ("sans_depart", "mois 0"),
    ("un_de_trop", "mois de trop"),
    ("proportions", "PARTS"),
    ("arrondi", "arrondissez"),
])
def test_projection_nomme_l_erreur(bug, morceau):
    _echec("projection", _projection_bogue(bug), morceau)


def _projection_numpy(depart, matrice, mois):
    """Une autre écriture juste : NumPy, `effectifs @ P`."""
    corrige.projection(depart, matrice, 0)                # mêmes contrôles
    corrige._entier_positif(mois, "mois")
    etats = list(matrice)
    p = np.array([[matrice[i][j] for j in etats] for i in etats])
    x = np.array([float(depart.get(e, 0)) for e in etats])
    resultat = [dict(zip(etats, map(float, x)))]
    for _ in range(mois):
        x = x @ p
        resultat.append(dict(zip(etats, map(float, x))))
    return resultat


def test_projection_d_autres_ecritures_justes():
    _reussi("projection", _projection_numpy)


def test_projection_alias_depart_et_lenteur():
    def meme_dict(depart, matrice, mois):
        courant = {e: float(depart.get(e, 0)) for e in matrice}
        resultat = [courant]
        for _ in range(mois):
            suivant = corrige.projection(courant, matrice, 1)[1]
            courant.update(suivant)                       # on modifie le dict déjà rangé dans la liste
            resultat.append(courant)
        return resultat

    _echec("projection", _avec_les_gardes("projection", meme_dict), "MÊME dict")

    def rend_le_depart(depart, matrice, mois):
        resultat = corrige.projection(depart, matrice, mois)
        if set(depart) == set(matrice):
            resultat[0] = depart
        return resultat

    _echec("projection", rend_le_depart, "lui-même")

    def modifie_le_depart(depart, matrice, mois):
        resultat = corrige.projection(depart, matrice, mois)
        for e in matrice:
            depart.setdefault(e, 0)
        return resultat

    _echec("projection", modifie_le_depart, "modifié")

    def depuis_le_depart(depart, matrice, mois):
        resultat = []
        for t in range(mois + 1):                         # P puissance t, recalculée depuis le départ
            courant = {e: float(depart.get(e, 0)) for e in matrice}
            for _ in range(t):
                suivant = dict.fromkeys(matrice, 0.0)
                for i in matrice:
                    for j, p in matrice[i].items():
                        suivant[j] += courant[i] * p
                courant = suivant
            resultat.append(courant)
        return resultat

    _echec("projection", _avec_les_gardes("projection", depuis_le_depart), "trop lent", "PRÉCÉDENT")


def test_projection_refuse_ce_qu_il_faut():
    def accepte_tout(depart, matrice, mois):
        if isinstance(mois, int) and not isinstance(mois, bool) and mois >= 0:
            return corrige.projection({e: n for e, n in depart.items() if e in matrice and _positif(n)}, matrice, mois)
        return corrige.projection(depart, matrice, mois)

    _echec("projection", accepte_tout, "Suspendu")


def _positif(n):
    return isinstance(n, (int, float)) and not isinstance(n, bool) and n >= 0


# --------------------------------------------------------------------------- Défi 256 : duree_de_vie


def _duree_bogue(bug):
    return _avec_les_gardes("duree_de_vie", lambda matrice, absorbants=(PARTI,): _ref_duree(matrice, absorbants, bug))


@pytest.mark.parametrize("bug,morceau", [
    ("sans_mois_courant", "mois EN COURS"),
    ("transpose", "mauvais sens"),
    ("tronque", "10 000"),
    ("avec_absorbants", "TRANSITOIRES"),
])
def test_duree_nomme_l_erreur(bug, morceau):
    _echec("duree_de_vie", _duree_bogue(bug), morceau)


def test_duree_geometrique_est_expliquee():
    def un_sur_taux(matrice, absorbants=(PARTI,)):
        absorbants = (absorbants,) if isinstance(absorbants, str) else absorbants
        return {i: 1 / sum(matrice[i][a] for a in absorbants) for i in matrice if i not in absorbants}

    _echec("duree_de_vie", _avec_les_gardes("duree_de_vie", un_sur_taux), "DIRECTEMENT", "un seul état")


def _duree_numpy(matrice, absorbants=(PARTI,)):
    """Une autre écriture juste : NumPy, après avoir vérifié qu'on peut partir de partout."""
    if isinstance(absorbants, str):
        absorbants = [absorbants]
    absorbants = set(absorbants)
    for a in absorbants:
        if a not in matrice or matrice[a][a] != 1.0:
            raise ValueError(f"{a!r} n'est pas un état absorbant de la matrice")
    transitoires = [e for e in matrice if e not in absorbants]
    peuvent = set(absorbants)
    for _ in matrice:
        peuvent |= {i for i in matrice if any(matrice[i][j] > 0 for j in peuvent)}
    bloques = [e for e in transitoires if e not in peuvent]
    if bloques:
        raise ValueError(f"on ne part jamais de {bloques}")
    q = np.array([[matrice[i][j] for j in transitoires] for i in transitoires])
    t = np.linalg.solve(np.eye(len(transitoires)) - q, np.ones(len(transitoires)))
    return dict(zip(transitoires, map(float, t)))


def test_duree_d_autres_ecritures_justes():
    _reussi("duree_de_vie", _duree_numpy)


def test_duree_convergence_trop_lente_et_infini():
    def jusqu_a_ce_que_ca_ne_bouge_plus(matrice, absorbants=(PARTI,)):
        corrige.duree_de_vie(matrice, absorbants)         # les contrôles
        absorbants = {absorbants} if isinstance(absorbants, str) else set(absorbants)
        transitoires = [e for e in matrice if e not in absorbants]
        t = dict.fromkeys(transitoires, 0.0)
        while True:
            nouveau = {i: 1 + sum(matrice[i][j] * t[j] for j in transitoires) for i in transitoires}
            if all(abs(nouveau[i] - t[i]) < 1e-10 for i in transitoires):
                return nouveau
            t = nouveau

    _echec("duree_de_vie", jusqu_a_ce_que_ca_ne_bouge_plus, "trop lent", "solve")

    def numpy_sans_verifier(matrice, absorbants=(PARTI,)):
        try:
            return corrige.duree_de_vie(matrice, absorbants)
        except ValueError as exc:
            if "infinie" not in str(exc):
                raise
        transitoires = [e for e in matrice if e != PARTI]
        q = np.array([[matrice[i][j] for j in transitoires] for i in transitoires])
        return dict(zip(transitoires, map(float, np.linalg.solve(np.eye(len(q)) - q, np.ones(len(q))))))

    _echec("duree_de_vie", numpy_sans_verifier, "INFINIE")

    def sans_nommer(matrice, absorbants=(PARTI,)):
        try:
            return corrige.duree_de_vie(matrice, absorbants)
        except ValueError as exc:
            if "infinie" in str(exc):
                raise ValueError("durée infinie") from None
            raise

    _echec("duree_de_vie", sans_nommer, "NOMMER")

    def set_du_texte(matrice, absorbants=(PARTI,)):
        return corrige.duree_de_vie(matrice, set(absorbants))

    _echec("duree_de_vie", set_du_texte, "cinq lettres")

    def tout_absorbant_accepte(matrice, absorbants=(PARTI,)):
        absorbants = {absorbants} if isinstance(absorbants, str) else set(absorbants)
        if not all(a in matrice for a in absorbants):
            raise ValueError("inconnu")
        return {e: t for e, t in corrige.duree_de_vie(matrice, (PARTI,) if PARTI in matrice else absorbants).items()
                if e not in absorbants} if absorbants - {PARTI} and PARTI in matrice else \
            corrige.duree_de_vie(matrice, absorbants)

    _echec("duree_de_vie", tout_absorbant_accepte, "pas absorbant")


# --------------------------------------------------------------------------- Défi 257 : valeur_client


def _valeur_bogue(bug):
    return _avec_les_gardes("valeur_client", lambda m, g, r, h: _ref_valeur(m, g, r, h, bug))


@pytest.mark.parametrize("bug,morceau", [
    ("taux_simple", "/ 12"),
    ("annuel", "taux ANNUEL"),
    ("sans_actualisation", "n'actualisez pas"),
    ("actualise_mois0", "mois EN COURS"),
    ("horizon_plus_un", "mois de trop"),
    ("horizon_moins_un", "mois de moins"),
    ("transpose", "mauvais sens"),
    ("tableur", "tableur"),
])
def test_valeur_nomme_l_erreur(bug, morceau):
    _echec("valeur_client", _valeur_bogue(bug), morceau)


def _valeur_vers_l_avant(matrice, marges, taux_annuel, horizon):
    """Une autre écriture juste : pour chaque état de départ, la projection du défi 255, actualisée et cumulée."""
    corrige.valeur_client(matrice, marges, taux_annuel, 0)
    corrige._entier_positif(horizon, "horizon")
    r = (1 + taux_annuel) ** (1 / 12) - 1
    resultat = {}
    for depart in matrice:
        repartition = {e: float(e == depart) for e in matrice}
        total = 0.0
        for t in range(horizon):
            total += sum(repartition[e] * marges.get(e, 0) for e in matrice) * (1 + r) ** -t
            suivant = dict.fromkeys(matrice, 0.0)
            for i in matrice:
                for j, p in matrice[i].items():
                    suivant[j] += repartition[i] * p
            repartition = suivant
        resultat[depart] = total
    return resultat


def _valeur_numpy(matrice, marges, taux_annuel, horizon):
    corrige.valeur_client(matrice, marges, taux_annuel, 0)
    corrige._entier_positif(horizon, "horizon")
    etats = list(matrice)
    p = np.array([[matrice[i][j] for j in etats] for i in etats]) / (1 + taux_annuel) ** (1 / 12)
    m = np.array([float(marges.get(e, 0)) for e in etats])
    v = np.zeros(len(etats))
    for _ in range(horizon):
        v = m + p @ v
    return dict(zip(etats, map(float, v)))


def test_valeur_d_autres_ecritures_justes():
    _reussi("valeur_client", _valeur_vers_l_avant)
    _reussi("valeur_client", _valeur_numpy)


def test_valeur_quadratique_est_arretee():
    def mois_par_mois_depuis_zero(matrice, marges, taux_annuel, horizon):
        corrige.valeur_client(matrice, marges, taux_annuel, 0)
        r = (1 + taux_annuel) ** (1 / 12) - 1
        resultat = dict.fromkeys(matrice, 0.0)
        for depart in matrice:
            for t in range(horizon):                       # la répartition du mois t, recalculée depuis le mois 0
                repartition = {e: float(e == depart) for e in matrice}
                for _ in range(t):
                    suivant = dict.fromkeys(matrice, 0.0)
                    for i in matrice:
                        for j, p in matrice[i].items():
                            suivant[j] += repartition[i] * p
                    repartition = suivant
                resultat[depart] += sum(repartition[e] * marges.get(e, 0) for e in matrice) / (1 + r) ** t
        return resultat

    _echec("valeur_client", mois_par_mois_depuis_zero, "trop lent", "rebours")


# --------------------------------------------------------------------------- Défi 258 : delai_rentabilite


def _delai_bogue(bug):
    def delai_rentabilite(matrice, marges, etat, cout, taux_annuel, horizon_max=120):
        corrige.delai_rentabilite(matrice, marges, etat, cout, taux_annuel, horizon_max)
        return _ref_delai(matrice, marges, etat, cout, taux_annuel, horizon_max, bug)
    return delai_rentabilite


@pytest.mark.parametrize("bug,morceau", [
    ("strict", ">="),
    ("zero_base", "trop petite de 1"),
    ("horizon", "None"),
    ("sans_actualisation", "n'actualisez pas"),
    ("actualise_mois0", "mois 0"),
    ("taux_simple", "ÉQUIVALENT"),
    ("annuel", "taux ANNUEL"),
    ("cohorte_figee", "change d'état"),
])
def test_delai_nomme_l_erreur(bug, morceau):
    _echec("delai_rentabilite", _delai_bogue(bug), morceau)


def _delai_puissances(matrice, marges, etat, cout, taux_annuel, horizon_max=120):
    """Une autre écriture juste : NumPy, l'actualisation en (1 + r) ** -t."""
    corrige.delai_rentabilite(matrice, marges, etat, 0, taux_annuel, horizon_max)
    if isinstance(cout, bool) or not isinstance(cout, (int, float)):
        raise TypeError("cout")
    if cout < 0:
        raise ValueError("cout")
    if cout == 0:
        return 0
    etats = list(matrice)
    p = np.array([[matrice[i][j] for j in etats] for i in etats])
    m = np.array([float(marges.get(e, 0)) for e in etats])
    x = np.array([float(e == etat) for e in etats])
    r = (1 + taux_annuel) ** (1 / 12) - 1
    cumul = 0.0
    for t in range(horizon_max):
        cumul += float(x @ m) * (1 + r) ** -t
        if cumul >= cout:
            return t + 1
        x = x @ p
    return None


def test_delai_d_autres_ecritures_justes():
    _reussi("delai_rentabilite", _delai_puissances)


def test_delai_qui_rappelle_valeur_client_est_arrete():
    def valeur(matrice, marges, taux_annuel, horizon):
        facteur = 1 / (1 + taux_annuel) ** (1 / 12)
        v = dict.fromkeys(matrice, 0.0)
        for _ in range(horizon):
            v = {i: marges.get(i, 0) + facteur * sum(p * v[j] for j, p in matrice[i].items()) for i in matrice}
        return v

    def delai(matrice, marges, etat, cout, taux_annuel, horizon_max=120):
        corrige.delai_rentabilite(matrice, marges, etat, cout, taux_annuel, horizon_max)
        if cout == 0:
            return 0
        for n in range(1, horizon_max + 1):                # tout le calcul, refait pour chaque n
            if valeur(matrice, marges, taux_annuel, n)[etat] >= cout:
                return n
        return None

    _echec("delai_rentabilite", delai, "trop lent", "Une seule passe")


def test_delai_rend_un_int_ou_none():
    def en_float(*args, **kwargs):
        r = corrige.delai_rentabilite(*args, **kwargs)
        return None if r is None else float(r)

    _echec("delai_rentabilite", en_float, "ENTIER")


# --------------------------------------------------------------------------- Défi 259 : matrices_par_segment


def _segments_bogue(bug):
    def matrices_par_segment(releves, segments, etats=ETATS, force=50):
        releves = list(releves)
        corrige.matrices_par_segment(releves, segments, etats, force)
        return _ref_segments(releves, segments, etats, force, bug)
    return matrices_par_segment


@pytest.mark.parametrize("bug,morceau", [
    ("sans_lissage", "pas lissées"),
    ("moyenne", "MOYENNE"),
    ("uniforme", "ÉGALEMENT"),
    ("denominateur", "plus de 100 %"),
    ("globale_moyenne", "TOUS les passages"),
    ("identite", "GLOBALE"),
])
def test_segments_nomme_l_erreur(bug, morceau):
    _echec("matrices_par_segment", _segments_bogue(bug), morceau)


def _segments_counter(releves, segments, etats=ETATS, force=50):
    """Une autre écriture juste : des Counter de passages, la globale et chaque segment."""
    if not isinstance(segments, dict):
        raise TypeError("segments")
    if isinstance(force, bool) or not isinstance(force, (int, float)):
        raise TypeError("force")
    if force < 0:
        raise ValueError("force")
    etats = list(etats)
    par_client: dict = {}
    for client, mois, etat in releves:
        if client not in segments:
            raise ValueError(f"{client!r} sans segment")
        if etat not in etats:
            raise ValueError("état inconnu")
        numero = corrige._numero_mois(mois)
        if par_client.setdefault(client, {}).setdefault(numero, etat) != etat:
            raise ValueError("contradiction")
    globale, par_segment = Counter(), {s: Counter() for s in segments.values()}
    for client, mois in par_client.items():
        for numero, etat in mois.items():
            if numero + 1 in mois:
                globale[etat, mois[numero + 1]] += 1
                par_segment[segments[client]][etat, mois[numero + 1]] += 1
    g = {}
    for i in etats:
        n = sum(globale[i, j] for j in etats)
        g[i] = {j: globale[i, j] / n if n else float(i == j) for j in etats}
    resultat = {}
    for s, c in par_segment.items():
        resultat[s] = {}
        for i in etats:
            n = sum(c[i, j] for j in etats)
            resultat[s][i] = {j: (c[i, j] + force * g[i][j]) / (n + force) for j in etats} if n + force else dict(g[i])
    return resultat


def test_segments_d_autres_ecritures_justes():
    _reussi("matrices_par_segment", _segments_counter)


def test_segments_refuse_les_clients_sans_segment():
    def ignore(releves, segments, etats=ETATS, force=50):
        return corrige.matrices_par_segment([r for r in releves if r[0] in segments], segments, etats, force)

    _echec("matrices_par_segment", ignore, "pas de segment")

    def oublie_les_segments_vides(releves, segments, etats=ETATS, force=50):
        releves = list(releves)
        vus = {segments[c] for c, _, _ in releves if c in segments}
        return {s: m for s, m in corrige.matrices_par_segment(releves, segments, etats, force).items() if s in vus}

    _echec("matrices_par_segment", oublie_les_segments_vides, "il manque")


# --------------------------------------------------------------------------- vérifications indépendantes


def test_la_matrice_estimee_retrouve_les_vraies_probabilites():
    vraies = valeur._VRAIES
    matrices = corrige.matrices_par_segment(releves_exemple(), segments_exemple(), force=0)
    for offre in ("Essentiel", "Confort"):
        for depuis in ETATS[:3]:
            for j, p in zip(ETATS, vraies[offre][depuis]):
                assert abs(matrices[offre][depuis][j] - p) < 0.05, (offre, depuis, j)     # quelques centaines de passages


def test_la_duree_de_vie_par_simulation():
    """Monte-Carlo : 40 000 clients simulés mois par mois ; la moyenne tombe à 2 % de la durée calculée."""
    me = matrice_exemple()
    duree = corrige.duree_de_vie(me)
    alea = random.Random(42)
    for depart in ("Actif", "À risque"):
        total = 0
        for _ in range(40_000):
            etat, mois = depart, 0
            while etat != PARTI:
                mois += 1
                etat = alea.choices(ETATS, weights=[me[etat][j] for j in ETATS])[0]
            total += mois
        assert abs(total / 40_000 - duree[depart]) < 0.02 * duree[depart], depart


def test_la_duree_de_vie_est_la_matrice_fondamentale():
    me = matrice_exemple()
    transitoires = ETATS[:3]
    q = np.array([[me[i][j] for j in transitoires] for i in transitoires])
    n = np.linalg.inv(np.eye(3) - q)
    attendu = n.sum(axis=1)
    for e, t in zip(transitoires, attendu):
        assert math.isclose(corrige.duree_de_vie(me)[e], t, rel_tol=1e-12)
        assert math.isclose(_ref_duree(me)[e], t, rel_tol=1e-12)


def test_une_duree_de_vie_est_une_valeur_a_un_euro_par_mois():
    me = matrice_exemple()
    duree = corrige.duree_de_vie(me)
    v = corrige.valeur_client(me, {e: 1 for e in ETATS[:3]}, 0.0, 5000)
    for e in ETATS[:3]:
        assert math.isclose(v[e], duree[e], rel_tol=1e-9)


def test_la_valeur_est_une_somme_de_puissances():
    me = matrice_exemple()
    p = np.array([[me[i][j] for j in ETATS] for i in ETATS])
    m = np.array([MARGES[e] for e in ETATS])
    r = 1.1 ** (1 / 12) - 1
    total, puissance = np.zeros(4), np.eye(4)
    for t in range(120):
        total += puissance @ m / (1 + r) ** t
        puissance = puissance @ p
    v = corrige.valeur_client(me, MARGES, 0.10, 120)
    for e, attendu in zip(ETATS, total):
        assert math.isclose(v[e], attendu, rel_tol=1e-12)
        assert math.isclose(_ref_valeur(me, MARGES, 0.10, 120)[e], attendu, rel_tol=1e-12)


def test_le_delai_du_cout_v_n_est_n():
    me = matrice_exemple()
    for n in (1, 2, 5, 13, 40):
        v = corrige.valeur_client(me, MARGES, TAUX_ANNUEL, n)
        for e in ETATS[:3]:
            assert corrige.delai_rentabilite(me, MARGES, e, v[e] * (1 - 1e-12), TAUX_ANNUEL) == n
            assert corrige.delai_rentabilite(me, MARGES, e, v[e] * (1 + 1e-9), TAUX_ANNUEL) == n + 1


def test_la_projection_conserve_les_clients_et_rejoint_la_valeur():
    me = matrice_exemple()
    etapes = corrige.projection({"Actif": 1}, me, 120)
    assert all(math.isclose(sum(x.values()), 1.0) for x in etapes)
    r = 1.1 ** (1 / 12) - 1
    vers_l_avant = sum(sum(x[e] * MARGES[e] for e in ETATS) / (1 + r) ** t for t, x in enumerate(etapes[:120]))
    assert math.isclose(vers_l_avant, corrige.valeur_client(me, MARGES, 0.10, 120)["Actif"], rel_tol=1e-12)
    assert etapes == _ref_projection({"Actif": 1}, me, 120)


def test_le_lissage_va_de_la_matrice_brute_a_la_globale():
    releves, segments = releves_exemple(), segments_exemple()
    globale = corrige.matrice_transition(releves)
    tres_fort = corrige.matrices_par_segment(releves, segments, force=1e12)
    for offre in ("Essentiel", "Confort", "Pro"):
        for i in ETATS:
            for j in ETATS:
                assert abs(tres_fort[offre][i][j] - globale[i][j]) < 1e-6
    brutes = corrige.matrices_par_segment(releves, segments, force=0)
    essentiel = [r for r in releves if segments[r[0]] == "Essentiel"]
    assert brutes["Essentiel"] == corrige.matrice_transition(essentiel)
    assert corrige.matrices_par_segment(releves, segments, force=50) == _ref_segments(releves, segments, ETATS, 50)


def test_la_reference_du_correcteur_compte_comme_le_corrige():
    releves = releves_exemple()
    assert _ref_matrice(releves) == corrige.matrice_transition(releves)
    pandas, attendu = _matrice_pandas(releves), corrige.matrice_transition(releves)
    assert all(pandas[i] == pytest.approx(attendu[i], rel=1e-12) for i in ETATS)


# --------------------------------------------------------------------------- les outils communs et l'histoire


def test_les_outils_communs():
    releves = releves_exemple()
    assert releves == releves_exemple() and releves is not releves_exemple()
    assert len(releves) == 48434 and len(set(releves)) == 48434 - 250            # 250 relevés chargés deux fois
    assert "2025-08" not in {m for _, m, _ in releves}
    assert {m for _, m, _ in releves} <= {f"{a}-{m:02d}" for a in (2025, 2026) for m in range(1, 13)}
    derniers: dict = {}
    for client, mois, etat in sorted(set(releves)):
        assert derniers.get(client) != PARTI, client                              # rien après le départ
        derniers[client] = etat
    segments = segments_exemple()
    assert set(segments) == {c for c, _, _ in releves} | (set(segments) - {c for c, _, _ in releves})
    assert Counter(segments.values()) == {"Essentiel": 2396, "Confort": 1574, "Pro": 30}
    me = matrice_exemple()
    assert all(math.isclose(sum(ligne.values()), 1.0) for ligne in me.values())
    assert me is not matrice_exemple() and me["Parti"]["Parti"] == 1.0
    assert generer_releves(30, graine=3) == generer_releves(30, graine=3)
    assert afficher({}) == " " * 10


def test_l_histoire_tient():
    releves, segments = releves_exemple(), segments_exemple()
    passages = corrige._transitions(corrige._par_client(releves, ETATS))
    restes = [p for p in passages if p[1] != PARTI]
    churn = sum(p[2] == PARTI for p in restes) / len(restes)
    lignes = [r for r in releves if r[2] != PARTI]
    marge = sum(MARGES[e] for _, _, e in lignes) / len(lignes)
    assert round(churn, 3) == 0.032 and round(marge, 2) == 23.49 and round(marge / churn) == 734
    me = matrice_exemple()
    assert {e: round(t, 1) for e, t in corrige.duree_de_vie(me).items()} == {"Fidèle": 54.0, "Actif": 37.8,
                                                                               "À risque": 23.2}
    v = corrige.valeur_client(me, MARGES, TAUX_ANNUEL, 120)
    assert {e: round(x) for e, x in v.items()} == {"Fidèle": 1013, "Actif": 657, "À risque": 378, "Parti": 0}
    assert round(v["Actif"] - v["À risque"]) == 279 and round(120 / (v["Actif"] - v["À risque"]), 2) == 0.43
    assert round(corrige.valeur_client(me, MARGES, TAUX_ANNUEL, 12_000)["À risque"]) == 388
    assert corrige.delai_rentabilite(me, MARGES, "Actif", 180, TAUX_ANNUEL) == 10
    assert corrige.delai_rentabilite(me, MARGES, "À risque", 400, TAUX_ANNUEL, horizon_max=12_000) is None
    pro = [r for r in releves if segments[r[0]] == "Pro"]
    assert len({r[0] for r in pro}) == 30 and {e for _, _, e in pro} == {"Fidèle", "Actif"}
    with pytest.raises(ValueError, match="infinie"):
        corrige.duree_de_vie(corrige.matrices_par_segment(releves, segments, force=0)["Pro"])
    lissees = corrige.matrices_par_segment(releves, segments, force=50)
    assert {e: round(t) for e, t in corrige.duree_de_vie(lissees["Pro"]).items()} == {"Fidèle": 369, "Actif": 325,
                                                                                         "À risque": 170}
    etapes = corrige.projection({"Actif": 1000}, me, 12)
    assert round(etapes[12]["Parti"]) == 323 and round(etapes[12]["À risque"]) == 58


# --------------------------------------------------------------------------- le correcteur lui-même


def test_le_fichier_eleve_est_a_faire_et_le_corrige_reussit():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6
    assert [d.numero for d in SERIE.defis] == list(range(254, 260))
    assert any(s is SERIE for s in series())            # enregistrée dans defis/verif.py


def test_les_exemples_des_docstrings_sont_justes(monkeypatch):
    import defis.serie_42 as eleve

    tests = (doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(valeur)
             + doctest.DocTestFinder().find(corrige))
    for nom in DEFIS:  # `from defis.serie_42 import *` (dans les exemples) doit donner les objets du corrigé
        monkeypatch.setattr(eleve, nom, getattr(corrige, nom))
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs.update({nom: getattr(corrige, nom) for nom in DEFIS})
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 30 and resultat.failed == 0


def test_les_tests_du_corrige_passent():
    for nom in dir(corrige):
        if nom.startswith("test_"):
            getattr(corrige, nom)()


def test_le_corrige_ne_modifie_rien():
    releves = [("A", "2026-01", "Actif"), ("A", "2026-02", "Parti")]
    copie = list(releves)
    segments = {"A": "Pro"}
    corrige.matrice_transition(releves)
    corrige.matrices_par_segment(releves, segments)
    assert releves == copie and segments == {"A": "Pro"}
    me, depart, marges = matrice_exemple(), {"Actif": 10}, {"Actif": 22}
    corrige.projection(depart, me, 3)
    corrige.duree_de_vie(me)
    corrige.valeur_client(me, marges, 0.1, 3)
    corrige.delai_rentabilite(me, marges, "Actif", 50, 0.1)
    assert depart == {"Actif": 10} and marges == {"Actif": 22} and me == matrice_exemple()


def test_les_erreurs_attendues_du_corrige():
    me = matrice_exemple()
    with pytest.raises(ValueError):
        corrige.matrice_transition([("A", "2025-13", "Actif")])
    with pytest.raises(ValueError):
        corrige.matrice_transition([("A", "2025-12", "Actif"), ("A", "2025-12", "Parti")])
    with pytest.raises(ValueError):
        corrige.projection({"Actif": -1}, me, 2)
    with pytest.raises(TypeError):
        corrige.projection({"Actif": 1}, me, 2.0)
    with pytest.raises(ValueError):
        corrige.duree_de_vie(me, ("Actif",))
    with pytest.raises(ValueError):
        corrige.valeur_client(me, MARGES, -0.5, 12)
    with pytest.raises(TypeError):
        corrige.valeur_client(me, MARGES, 0.1, True)
    with pytest.raises(ValueError):
        corrige.delai_rentabilite(me, MARGES, "Suspendu", 100, 0.1)
    with pytest.raises(ValueError):
        corrige.matrices_par_segment([("A", "2026-01", "Actif")], {}, force=1)
    with pytest.raises(TypeError):
        corrige.matrices_par_segment([], {}, force="1")


def test_les_bugs_annonces_existent_tous():
    """Chaque erreur classique décrite par le correcteur a un test ci-dessus (et inversement)."""
    noms = {
        "matrice": {"frontiere", "trous", "decembre", "ordre", "doublons", "colonnes", "total", "occurrences", "vide",
                    "uniforme"},
        "projection": {"transpose", "sans_depart", "un_de_trop", "proportions", "arrondi"},
        "duree": {"geometrique", "sans_mois_courant", "transpose", "tronque", "avec_absorbants"},
        "valeur": {"taux_simple", "annuel", "sans_actualisation", "actualise_mois0", "horizon_plus_un",
                   "horizon_moins_un", "transpose", "tableur"},
        "delai": {"strict", "zero_base", "horizon", "sans_actualisation", "actualise_mois0", "taux_simple", "annuel",
                  "cohorte_figee"},
        "segments": {"sans_lissage", "moyenne", "uniforme", "denominateur", "globale_moyenne", "identite"},
    }
    assert set(_BUGS_MATRICE) == noms["matrice"] and set(_BUGS_PROJECTION) == noms["projection"]
    assert set(_BUGS_DUREE) == noms["duree"] and set(_BUGS_VALEUR) == noms["valeur"]
    assert set(_BUGS_DELAI) == noms["delai"] and set(_BUGS_SEGMENTS) == noms["segments"]

"""Série 42 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_42.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours un
décembre suivi d'un janvier que l'on n'a pas reconnu comme « le mois suivant », une matrice multipliée dans le
mauvais sens, le mois en cours oublié dans une durée de vie, ou un taux annuel divisé par 12.

Fil rouge de la série : la valeur d'un client n'est pas « marge / taux de churn ». Un client change d'ÉTAT d'un
mois à l'autre, et tout se déduit de la matrice qui dit comment :

1. on ESTIME la matrice en comptant les passages d'un état à l'autre entre deux mois CONSÉCUTIFS d'un même
   client (pas entre deux clients, pas par-dessus un relevé perdu), puis en divisant chaque ligne par son total ;
2. on PROJETTE une cohorte : le mois suivant, l'état j reçoit Σ_i (clients en i) × P[i][j]. C'est un produit
   « vecteur ligne × matrice », mois après mois, en partant du mois précédent (jamais de zéro) ;
3. la DURÉE DE VIE attendue résout t = 1 + Q·t (le mois en cours, plus la suite) : un système linéaire, pas une
   boucle de 1 000 mois, qui s'arrêterait bien avant les clients fidèles ;
4. la VALEUR se calcule à rebours : V = m + P·V / (1 + r), avec r le taux MENSUEL équivalent au taux annuel,
   (1 + a) ** (1 / 12) - 1, et le mois en cours qui n'est pas actualisé ;
5. le DÉLAI DE RENTABILITÉ cumule mois après mois ce que la cohorte rapporte, actualisé, et s'arrête dès que le
   coût d'acquisition est couvert (>=) ; jamais → None ;
6. un petit segment donne des estimations extrêmes (aucun départ vu → durée de vie infinie) : on le LISSE vers
   la matrice de toute la clientèle, avec `force` transitions « empruntées ».

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import math

from defis.valeur import (  # noqa: F401
    ETATS,
    MARGES,
    MOIS_PERDU,
    OFFRES,
    PARTI,
    TAUX_ANNUEL,
    afficher,
    generer_releves,
    matrice_exemple,
    releves_exemple,
    segments_exemple,
)


def _nombre(valeur, nom: str) -> float:
    """Un vrai nombre (int ou float, pas un bool ni un texte), fini : sinon TypeError ou ValueError."""
    if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
        raise TypeError(f"{nom} doit être un nombre, pas {type(valeur).__name__}")
    if not math.isfinite(valeur):
        raise ValueError(f"{nom} doit être un nombre fini, pas {valeur!r}")
    return float(valeur)


def _entier_positif(valeur, nom: str) -> int:
    if isinstance(valeur, bool) or not isinstance(valeur, int):
        raise TypeError(f"{nom} doit être un int, pas {type(valeur).__name__}")
    if valeur < 0:
        raise ValueError(f"{nom} doit être >= 0, pas {valeur}")
    return valeur


# ============================================================================ Défi 254 : matrice_transition


def _numero_mois(mois) -> int:
    """« 2025-12 » → 2025 × 12 + 11. Deux mois sont consécutifs si leurs numéros se suivent : décembre 2025
    (24311) et janvier 2026 (24312) aussi. `int("202512") + 1` donnerait 202513, un mois qui n'existe pas."""
    if not isinstance(mois, str) or len(mois) != 7 or mois[4] != "-" or not (mois[:4] + mois[5:]).isdigit():
        raise ValueError(f"mois illisible : {mois!r} (attendu « AAAA-MM »)")
    annee, numero = int(mois[:4]), int(mois[5:])
    if not 1 <= numero <= 12:
        raise ValueError(f"mois illisible : {mois!r} (le mois va de 01 à 12)")
    return annee * 12 + numero - 1


def _par_client(releves, etats) -> dict:
    """{client: {numéro du mois: état}}, en UNE lecture des relevés (un flux ne se relit pas)."""
    connus = set(etats)
    par_client: dict = {}
    for releve in releves:
        client, mois, etat = releve
        if etat not in connus:
            raise ValueError(f"état inconnu : {etat!r} (client {client!r}, {mois})")
        numero = _numero_mois(mois)
        mois_du_client = par_client.setdefault(client, {})
        deja = mois_du_client.get(numero)
        if deja is not None and deja != etat:
            raise ValueError(f"relevés contradictoires pour {client!r} en {mois} : {deja!r} et {etat!r}")
        mois_du_client[numero] = etat          # un doublon exact ne compte qu'une fois
    return par_client


def _transitions(par_client: dict) -> list[tuple]:
    """[(client, depuis, vers), ...] : un passage par paire de mois CONSÉCUTIFS d'un même client."""
    passages = []
    for client, mois_du_client in par_client.items():
        for numero, etat in mois_du_client.items():
            suivant = mois_du_client.get(numero + 1)      # le mois SUIVANT, pas « le relevé suivant »
            if suivant is not None:
                passages.append((client, etat, suivant))
    return passages


def _compter(passages, etats) -> dict:
    comptes = {i: dict.fromkeys(etats, 0) for i in etats}
    for _, depuis, vers in passages:
        comptes[depuis][vers] += 1
    return comptes


def _normaliser(comptes, etats) -> dict:
    """Chaque LIGNE divisée par son total ; une ligne jamais observée : on reste où l'on est (1 sur la diagonale)."""
    matrice = {}
    for i in etats:
        total = sum(comptes[i].values())
        matrice[i] = {j: comptes[i][j] / total if total else float(i == j) for j in etats}
    return matrice


def matrice_transition(releves, etats=ETATS) -> dict[str, dict[str, float]]:
    etats = list(etats)
    return _normaliser(_compter(_transitions(_par_client(releves, etats)), etats), etats)


# ============================================================================ Défi 255 : projection


def _verifier_matrice(matrice) -> list:
    if not isinstance(matrice, dict) or not matrice:
        raise TypeError("la matrice est un dict non vide {depuis: {vers: probabilité}}")
    return list(matrice)


def _un_mois(effectifs: dict, matrice: dict, etats: list) -> dict:
    """Le mois suivant : vers[j] = Σ_i effectifs[i] × P[i][j] (un vecteur LIGNE fois la matrice)."""
    suivant = dict.fromkeys(etats, 0.0)
    for i in etats:
        n = effectifs[i]
        if n:
            for j, p in matrice[i].items():
                suivant[j] += n * p
    return suivant


def projection(depart: dict, matrice: dict, mois: int) -> list[dict[str, float]]:
    etats = _verifier_matrice(matrice)
    mois = _entier_positif(mois, "mois")
    if not isinstance(depart, dict):
        raise TypeError("depart est un dict {état: nombre de clients}")
    courant = dict.fromkeys(etats, 0.0)
    for etat, n in depart.items():
        if etat not in matrice:
            raise ValueError(f"état inconnu dans le départ : {etat!r}")
        n = _nombre(n, f"depart[{etat!r}]")
        if n < 0:
            raise ValueError(f"un effectif ne peut pas être négatif : depart[{etat!r}] = {n}")
        courant[etat] = n
    resultat = [courant]
    for _ in range(mois):
        courant = _un_mois(courant, matrice, etats)      # à partir du mois PRÉCÉDENT : jamais de zéro
        resultat.append(courant)
    return resultat


# ============================================================================ Défi 256 : duree_de_vie


def _absorbants(matrice: dict, absorbants) -> set:
    if isinstance(absorbants, str):
        absorbants = (absorbants,)
    absorbants = set(absorbants)
    for a in absorbants:
        if a not in matrice:
            raise ValueError(f"état absorbant inconnu : {a!r}")
        if abs(matrice[a].get(a, 0.0) - 1.0) > 1e-9:
            raise ValueError(f"{a!r} n'est pas absorbant : on le quitte (P[{a!r}][{a!r}] = {matrice[a].get(a, 0.0)})")
    return absorbants


def _qui_peut_partir(matrice: dict, absorbants: set) -> set:
    """Les états d'où l'on PEUT atteindre un état absorbant (en remontant les flèches de probabilité > 0)."""
    atteint = set(absorbants)
    change = True
    while change:
        change = False
        for i, ligne in matrice.items():
            if i not in atteint and any(p > 0 and j in atteint for j, p in ligne.items()):
                atteint.add(i)
                change = True
    return atteint


def _resoudre(a: list[list[float]], b: list[float]) -> list[float]:
    """Résout a · x = b par élimination de Gauss avec pivot partiel (a et b sont recopiés, pas modifiés)."""
    n = len(b)
    lignes = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda k: abs(lignes[k][col]))
        lignes[col], lignes[pivot] = lignes[pivot], lignes[col]
        for k in range(col + 1, n):
            facteur = lignes[k][col] / lignes[col][col]
            if facteur:
                for c in range(col, n + 1):
                    lignes[k][c] -= facteur * lignes[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (lignes[i][n] - sum(lignes[i][c] * x[c] for c in range(i + 1, n))) / lignes[i][i]
    return x


def duree_de_vie(matrice: dict, absorbants=(PARTI,)) -> dict[str, float]:
    etats = _verifier_matrice(matrice)
    absorbants = _absorbants(matrice, absorbants)
    transitoires = [e for e in etats if e not in absorbants]
    peuvent_partir = _qui_peut_partir(matrice, absorbants)
    bloques = [e for e in transitoires if e not in peuvent_partir]
    if bloques:
        raise ValueError(f"durée de vie infinie : depuis {bloques}, aucun état absorbant n'est atteignable")
    # t[i] = 1 + Σ_j Q[i][j] t[j]   ⇔   (I − Q) t = 1
    a = [[float(i == j) - matrice[i].get(j, 0.0) for j in transitoires] for i in transitoires]
    t = _resoudre(a, [1.0] * len(transitoires))
    return dict(zip(transitoires, t))


# ============================================================================ Défi 257 : valeur_client


def _taux_mensuel(taux_annuel) -> float:
    """Le taux MENSUEL équivalent : (1 + a) ** (1 / 12) - 1. 10 % par an, c'est 0,797 % par mois (pas 0,833 %)."""
    taux_annuel = _nombre(taux_annuel, "taux_annuel")
    if taux_annuel < 0:
        raise ValueError(f"taux_annuel doit être >= 0, pas {taux_annuel}")
    return (1 + taux_annuel) ** (1 / 12) - 1


def _marges(marges, etats) -> dict:
    if not isinstance(marges, dict):
        raise TypeError("marges est un dict {état: marge mensuelle}")
    for etat, m in marges.items():
        if etat not in etats:
            raise ValueError(f"marge pour un état inconnu : {etat!r}")
        _nombre(m, f"marges[{etat!r}]")
    return {e: float(marges.get(e, 0.0)) for e in etats}


def valeur_client(matrice: dict, marges: dict, taux_annuel: float, horizon: int) -> dict[str, float]:
    etats = _verifier_matrice(matrice)
    m = _marges(marges, etats)
    facteur = 1 / (1 + _taux_mensuel(taux_annuel))
    horizon = _entier_positif(horizon, "horizon")
    v = dict.fromkeys(etats, 0.0)
    for _ in range(horizon):            # à rebours : la valeur sur h mois, à partir de celle sur h − 1 mois
        v = {i: m[i] + facteur * sum(p * v[j] for j, p in matrice[i].items()) for i in etats}
    return v


# ============================================================================ Défi 258 : delai_rentabilite


def delai_rentabilite(matrice: dict, marges: dict, etat: str, cout: float, taux_annuel: float,
                      horizon_max: int = 120) -> int | None:
    etats = _verifier_matrice(matrice)
    if etat not in matrice:
        raise ValueError(f"état de départ inconnu : {etat!r}")
    m = _marges(marges, etats)
    facteur = 1 / (1 + _taux_mensuel(taux_annuel))
    cout = _nombre(cout, "cout")
    if cout < 0:
        raise ValueError(f"cout doit être >= 0, pas {cout}")
    horizon_max = _entier_positif(horizon_max, "horizon_max")
    if cout == 0:
        return 0
    repartition = dict.fromkeys(etats, 0.0)
    repartition[etat] = 1.0
    cumul, actualisation = 0.0, 1.0
    for t in range(horizon_max):
        cumul += actualisation * sum(repartition[e] * m[e] for e in etats)
        if cumul >= cout:
            return t + 1                    # couvert au bout de t + 1 mois (le mois 0 est le premier)
        actualisation *= facteur
        repartition = _un_mois(repartition, matrice, etats)
    return None


# ============================================================================ Défi 259 : matrices_par_segment


def matrices_par_segment(releves, segments: dict, etats=ETATS, force: float = 50) -> dict[str, dict]:
    etats = list(etats)
    if not isinstance(segments, dict):
        raise TypeError("segments est un dict {client: segment}")
    force = _nombre(force, "force")
    if force < 0:
        raise ValueError(f"force doit être >= 0, pas {force}")
    par_client = _par_client(releves, etats)
    sans_segment = [client for client in par_client if client not in segments]
    if sans_segment:
        raise ValueError(f"client(s) sans segment : {sans_segment[:5]}")
    passages = _transitions(par_client)
    globale = _normaliser(_compter(passages, etats), etats)      # TOUTE la clientèle : pondérée par les comptes
    par_segment = {s: [] for s in segments.values()}
    for passage in passages:
        par_segment[segments[passage[0]]].append(passage)
    resultat = {}
    for segment, siens in par_segment.items():
        comptes = _compter(siens, etats)
        matrice = {}
        for i in etats:
            total = sum(comptes[i].values())
            if total + force == 0:
                matrice[i] = dict(globale[i])                     # rien vu, rien emprunté : la ligne globale
            else:
                matrice[i] = {j: (comptes[i][j] + force * globale[i][j]) / (total + force) for j in etats}
        resultat[segment] = matrice
    return resultat


# ============================================================================ Tests (à mettre dans tests/)


def test_decembre_puis_janvier_se_suivent():
    releves = [("A", "2026-01", "Actif"), ("A", "2025-12", "Fidèle"), ("B", "2025-11", "Actif"),
               ("B", "2026-01", "Parti")]                         # B : décembre manque, pas de passage
    matrice = matrice_transition(releves)
    assert matrice["Fidèle"]["Actif"] == 1.0
    assert matrice["Actif"] == {"Fidèle": 0.0, "Actif": 1.0, "À risque": 0.0, "Parti": 0.0}   # jamais quitté


def test_chaque_ligne_fait_100_pourcent():
    matrice = matrice_transition(releves_exemple())
    assert all(abs(sum(ligne.values()) - 1) < 1e-12 for ligne in matrice.values())
    assert matrice[PARTI][PARTI] == 1.0


def test_projection_conserve_les_clients():
    mois = projection({"Actif": 1000}, matrice_exemple(), 24)
    assert len(mois) == 25 and mois[0]["Actif"] == 1000
    assert all(abs(sum(m.values()) - 1000) < 1e-9 for m in mois)
    assert mois[1]["À risque"] == 50.0


def test_duree_de_vie_compte_le_mois_en_cours():
    assert duree_de_vie({"A": {"A": 0.0, "P": 1.0}, "P": {"A": 0.0, "P": 1.0}}, "P") == {"A": 1.0}
    duree = duree_de_vie(matrice_exemple())
    assert duree["Fidèle"] > duree["Actif"] > duree["À risque"] > 1


def test_valeur_actualisee_au_taux_mensuel_equivalent():
    seul = {"A": {"A": 1.0}}
    v = valeur_client(seul, {"A": 100}, 0.10, 12)["A"]
    attendu = sum(100 / 1.1 ** (t / 12) for t in range(12))
    assert abs(v - attendu) < 1e-9
    assert valeur_client(seul, {"A": 100}, 0.0, 12)["A"] == 1200


def test_delai_rentabilite_au_mois_pres():
    seul = {"A": {"A": 1.0}}
    assert delai_rentabilite(seul, {"A": 10}, "A", 30, 0.0) == 3                 # 10 + 10 + 10 >= 30
    assert delai_rentabilite(seul, {"A": 10}, "A", 30.5, 0.0) == 4
    assert delai_rentabilite(seul, {"A": 10}, "A", 10_000, 0.0, horizon_max=12) is None


def test_le_lissage_rend_la_duree_de_vie_finie():
    releves, segments = releves_exemple(), segments_exemple()
    brutes = matrices_par_segment(releves, segments, force=0)
    try:
        duree_de_vie(brutes["Pro"])
    except ValueError:
        pass
    else:
        raise AssertionError("sans lissage, les clients Pro ne partent jamais")
    lissees = matrices_par_segment(releves, segments, force=50)
    assert 0 < duree_de_vie(lissees["Pro"])["Actif"] < 1000

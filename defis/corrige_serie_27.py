"""Série 27 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_27.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours
une valeur absente prise pour une valeur (`None == None`, un NaN « vrai »), un `False` pris pour une
valeur absente, ou une boucle dans une boucle.

Fil rouge de la série : dédoublonner, c'est décider sans jamais tout comparer à tout. Quatre habitudes :

1. on ne compare que des fiches qui ont une chance d'aller ensemble (le blocage), et on DIT ce qu'on a
   laissé de côté (le bloc « 01 00 00 00 00 » de 400 fiches) au lieu de le jeter en silence ;
2. une valeur absente n'est ni un accord ni un désaccord : elle ne compte pas. Deux fiches sans e-mail
   n'ont pas « le même e-mail » ; un NaN n'est pas une date de naissance ;
3. la ressemblance est transitive par décision, pas par calcul : A ressemble à B, B à C, donc A, B et C
   sont un seul client, même si A et C ne se ressemblent pas (union-find) ;
4. on mesure la qualité en PAIRES, sur les deux erreurs : fusionner deux personnes (précision) coûte
   bien plus cher que garder deux fiches de la même (rappel).

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import itertools
from collections import Counter
from datetime import date

import pandas as pd

from defis.doublons import (  # noqa: F401
    CLES,
    SEUIL,
    VERITE_EXEMPLE,
    cle_email,
    cle_nom_code_postal,
    cle_telephone,
    fiches_exemple,
    generer_fiches,
    normaliser_email,
    normaliser_nom,
    normaliser_telephone,
)

# ============================================================================ La distance d'édition


def distance_edition(a: str, b: str) -> int:
    if not isinstance(a, str) or not isinstance(b, str):
        # None (une case vide) n'est pas "" : c'est à l'appelant de décider ce qu'une valeur absente veut dire.
        raise TypeError(f"distance_edition attend deux textes, pas {type(a).__name__} et {type(b).__name__}")
    if len(a) < len(b):
        a, b = b, a  # la ligne du tableau suit le texte le plus COURT : moins de mémoire
    # precedente[j] = distance entre les i - 1 premières lettres de a et les j premières lettres de b.
    # Première ligne : passer de "" à b[:j], c'est j insertions (et pas 0 !).
    precedente = list(range(len(b) + 1))
    for i, lettre_a in enumerate(a, start=1):
        courante = [i]  # passer de a[:i] à "" : i suppressions
        for j, lettre_b in enumerate(b, start=1):
            courante.append(min(
                precedente[j] + 1,                            # supprimer lettre_a
                courante[j - 1] + 1,                          # insérer lettre_b
                precedente[j - 1] + (lettre_a != lettre_b),   # remplacer (gratuit si c'est la même lettre)
            ))
        precedente = courante  # on ne garde que DEUX lignes : 2 × (len(b) + 1) nombres au lieu d'un tableau entier
    return precedente[-1]
    # Pourquoi pas la récursion « évidente » d(a, b) = min(d(a[1:], b) + 1, …) ? Sans mémoire, elle recalcule
    # les mêmes sous-problèmes : 3^n appels (plus d'un milliard pour deux mots de 20 lettres). Avec
    # `functools.cache`, elle devient rapide… jusqu'à un champ de 1 000 lettres, où Python s'arrête :
    # RecursionError. Le tableau rempli ligne à ligne fait le même calcul, sans pile.


# ============================================================================ Regrouper (union-find)


def regrouper(identifiants, paires) -> list[list]:
    parent: dict = {}
    taille: dict = {}
    for ident in identifiants:  # lu UNE fois : un générateur marche aussi
        if ident in parent:
            raise ValueError(f"l'identifiant {ident!r} apparaît deux fois : c'est une clé cassée, pas un doublon")
        parent[ident] = ident  # chaque fiche est d'abord son propre groupe
        taille[ident] = 1

    def racine(x):
        r = x
        while parent[r] != r:  # une BOUCLE et pas une récursion : une chaîne de 100 000 fiches ne fait pas planter
            r = parent[r]
        while parent[x] != r:  # compression du chemin : chaque fiche traversée pointe désormais sur la racine
            parent[x], x = r, parent[x]
        return r

    for a, b in paires:  # lu UNE fois aussi
        for x in (a, b):
            if x not in parent:
                raise ValueError(f"la paire ({a!r}, {b!r}) cite {x!r}, qui n'est pas dans les identifiants")
        ra, rb = racine(a), racine(b)
        if ra == rb:
            continue  # déjà dans le même groupe (A = B, B = C, puis A = C)
        if taille[ra] < taille[rb]:
            ra, rb = rb, ra
        parent[rb] = ra  # union par taille : le petit groupe rejoint le grand, l'arbre reste plat
        taille[ra] += taille[rb]

    groupes: dict = {}
    for ident in parent:  # TOUTES les fiches, même celles qu'aucune paire ne cite (un client = une fiche)
        groupes.setdefault(racine(ident), []).append(ident)
    # Chaque groupe trié, puis les groupes triés : comme ils sont disjoints, c'est l'ordre de leur premier élément.
    return sorted(sorted(g) for g in groupes.values())
    # L'erreur classique : un dictionnaire « fiche → numéro de groupe », et à chaque paire on renumérote tout
    # le groupe de b. Juste, mais quadratique : 200 000 fiches en chaîne, 20 milliards de renumérotations.


# ============================================================================ Le blocage


def paires_candidates(fiches: dict, cles=CLES, taille_max: int = 50) -> tuple[set, list]:
    if isinstance(taille_max, bool) or not isinstance(taille_max, int) or taille_max < 2:
        raise ValueError(f"taille_max = {taille_max!r} : un entier d'au moins 2 (un bloc de 1 fiche n'a pas de paire)")
    paires: set = set()
    trop_gros: list = []
    for position, cle in enumerate(cles):
        # Un dictionnaire PAR clé : la valeur « 75011 » d'une clé n'a rien à voir avec la valeur « 75011 »
        # d'une autre. Et chaque clé est calculée UNE fois par fiche (n appels), jamais dans une boucle sur les paires.
        blocs: dict = {}
        for ident, fiche in fiches.items():
            valeur = cle(fiche)
            if valeur is None or valeur == "":
                continue  # sans téléphone : PAS dans le bloc « téléphone vide » (qui réunirait 3 000 fiches)
            blocs.setdefault(valeur, []).append(ident)
        for valeur, membres in blocs.items():
            if len(membres) > taille_max:
                # 400 fiches avec le numéro du centre d'appels : 79 800 comparaisons inutiles. On saute… et on le DIT.
                trop_gros.append((position, valeur, len(membres)))
                continue
            # combinations sur la liste TRIÉE : chaque paire sort une fois, dans l'ordre (petit, grand). Le `set`
            # garde une seule fois la paire que deux clés proposent toutes les deux.
            paires.update(itertools.combinations(sorted(membres), 2))
    trop_gros.sort(key=lambda t: (-t[2], t[0], t[1]))
    return paires, trop_gros
    # Coût : n × (nombre de clés) appels, plus la somme des taille² / 2 des blocs. Comparer chaque fiche à
    # chaque autre coûterait n² / 2 : 50 milliards de comparaisons pour 300 000 fiches.


# ============================================================================ Le score de correspondance


def _texte(valeur) -> str:
    """Une valeur connue, en texte sans espaces autour ; "" si elle manque (None, NaN, texte vide)."""
    if valeur is None:
        return ""
    if not isinstance(valeur, str):
        try:
            if pd.isna(valeur):  # NaN, NaT, pd.NA. Attention : `if valeur:` laisse passer NaN (il est « vrai »)
                return ""
        except (TypeError, ValueError):
            pass
    return str(valeur).strip()


def _mots_tries(nom) -> str:
    """« DUPRE Helene » et « Hélène Dupré » donnent tous deux « dupre helene »."""
    return " ".join(sorted(normaliser_nom(nom).split()))


def score_correspondance(a: dict, b: dict) -> int:
    score = 0
    # Chaque règle commence par « les DEUX valeurs sont-elles connues ? ». Sinon, rien : deux fiches sans
    # e-mail n'ont pas le même e-mail (avec `ea == eb`, "" == "" donnerait +4 à toutes les fiches sans e-mail).
    ea, eb = normaliser_email(a.get("email")), normaliser_email(b.get("email"))
    if ea and eb:
        score += 4 if ea == eb else -1
    ta, tb = normaliser_telephone(a.get("telephone")), normaliser_telephone(b.get("telephone"))
    if ta and tb:
        score += 3 if ta == tb else -1  # -1 seulement : un numéro change, ou se partage (un centre d'appels)
    na, nb = _texte(a.get("naissance")), _texte(b.get("naissance"))
    if na and nb:
        score += 3 if na == nb else -8  # deux dates différentes : presque toujours deux personnes (un parent, un enfant)
    ma, mb = _mots_tries(a.get("nom")), _mots_tries(b.get("nom"))
    if ma and mb:
        if ma == mb:
            score += 3
        elif distance_edition(ma, mb) <= 2:
            score += 1  # une ou deux fautes de frappe : un indice, pas une preuve
        else:
            score -= 4
    ca, cb = _texte(a.get("code_postal")), _texte(b.get("code_postal"))
    if ca and cb and ca == cb:
        score += 1  # deux codes différents : 0 (on déménage)
    return score
    # Les poids viennent de l'expérience (ou d'un modèle : la méthode de Fellegi et Sunter les estime à partir
    # des données). Ce qui compte, c'est la structure : accord, désaccord ET absence, trois cas distincts.


# ============================================================================ La fiche de référence


_CHAMPS_CALCULES = ("id", "anciens_ids", "maj", "inscription", "resilie")


def _absent(valeur) -> bool:
    """None, NaN, NaT, pd.NA ou un texte vide. Mais PAS False ni 0 : ce sont des réponses."""
    return _texte(valeur) == ""


def fiche_de_reference(fiches) -> dict:
    fiches = list(fiches)
    if not fiches:
        raise ValueError("aucune fiche : il faut au moins une fiche pour en faire une fiche de référence")
    for fiche in fiches:
        if _absent(fiche.get("id")) or _absent(fiche.get("maj")):
            raise ValueError(f"chaque fiche doit avoir un 'id' et une date 'maj' : {fiche!r}")
    # Les positions, de la plus ancienne mise à jour à la plus récente ; à date égale, l'ordre de la liste.
    # (Avec `sorted(fiches, key=..., reverse=True)`, les ex aequo GARDERAIENT leur ordre d'origine : la
    # première des deux arriverait d'abord, alors que c'est la dernière importée qui doit gagner.)
    ordre = sorted(range(len(fiches)), key=lambda i: (fiches[i]["maj"], i))
    du_plus_recent = [fiches[i] for i in reversed(ordre)]
    ids = sorted(fiche["id"] for fiche in fiches)
    resultat: dict = {"id": ids[0], "anciens_ids": ids, "maj": max(fiche["maj"] for fiche in fiches)}
    # L'ancienneté compte depuis le PREMIER contact : la fiche la plus récente ferait d'une cliente de 2016 une
    # « nouvelle cliente » de 2021, et le modèle la verrait comme fragile.
    inscriptions = [fiche["inscription"] for fiche in fiches if not _absent(fiche.get("inscription"))]
    resultat["inscription"] = min(inscriptions) if inscriptions else None
    # Partie seulement si TOUTES ses fiches sont résiliées : sinon c'est un « faux churn » (elle a quitté la
    # boutique mais reste cliente sur le site).
    resultat["resilie"] = all(not _absent(fiche.get("resilie")) and bool(fiche["resilie"]) for fiche in fiches)
    champs = dict.fromkeys(champ for fiche in fiches for champ in fiche)  # l'union, dans l'ordre d'apparition
    for champ in champs:
        if champ in _CHAMPS_CALCULES:
            continue
        # La valeur la plus récente qui soit CONNUE. `if valeur:` jetterait False et 0, qui sont des réponses :
        # `optin_sms = False`, c'est une cliente qui a REFUSÉ les SMS. La réactiver avec une vieille fiche, c'est
        # lui écrire contre son gré (et une amende CNIL).
        resultat[champ] = next((fiche[champ] for fiche in du_plus_recent
                                if champ in fiche and not _absent(fiche[champ])), None)
    return resultat


# ============================================================================ La qualité d'un dédoublonnage


def _paires(k: int) -> int:
    return k * (k - 1) // 2  # un groupe de k fiches contient k(k - 1)/2 paires (non ordonnées) ; un entier


def _etiquettes(groupes, nom: str) -> tuple[dict, list[int]]:
    """{identifiant: numéro de son groupe}, et la taille de chaque groupe ; ValueError si un identifiant revient."""
    etiquette: dict = {}
    tailles: list[int] = []
    for numero, groupe in enumerate(groupes):
        groupe = list(groupe)
        for ident in groupe:
            if ident in etiquette:
                raise ValueError(f"{nom} : {ident!r} est dans deux groupes (ou deux fois dans le même)")
            etiquette[ident] = numero
        tailles.append(len(groupe))
    return etiquette, tailles


def qualite_dedoublonnage(predits, verite) -> dict:
    etiquette_p, tailles_p = _etiquettes(predits, "predits")
    etiquette_v, tailles_v = _etiquettes(verite, "verite")
    if etiquette_p.keys() != etiquette_v.keys():
        seulement_p = sorted(map(repr, etiquette_p.keys() - etiquette_v.keys()))[:3]
        seulement_v = sorted(map(repr, etiquette_v.keys() - etiquette_p.keys()))[:3]
        raise ValueError(f"les deux regroupements ne portent pas sur les mêmes fiches (seulement prédites : "
                         f"{seulement_p}, seulement dans la vérité : {seulement_v})")
    paires_predites = sum(_paires(k) for k in tailles_p)
    paires_vraies = sum(_paires(k) for k in tailles_v)
    # Le tableau croisé : combien de fiches sont dans le groupe prédit p ET dans le vrai groupe v. Les paires
    # justes sont celles dont les deux fiches tombent dans la même case : C(nombre, 2) par case. On ne
    # fabrique jamais les paires elles-mêmes (un groupe de 40 000 fiches en contient 800 millions).
    croisement = Counter((etiquette_p[ident], etiquette_v[ident]) for ident in etiquette_p)
    paires_justes = sum(_paires(k) for k in croisement.values())
    # Aucune paire prédite : aucune fusion n'est fausse (précision 1) ; aucune vraie paire : rien à retrouver (rappel 1).
    precision = paires_justes / paires_predites if paires_predites else 1.0
    rappel = paires_justes / paires_vraies if paires_vraies else 1.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0  # moyenne HARMONIQUE
    return {"paires_predites": paires_predites, "paires_vraies": paires_vraies, "paires_justes": paires_justes,
            "precision": precision, "rappel": rappel, "f1": f1}


# ============================================================================ Tout assembler


def dedoublonner(fiches: dict, seuil: int = SEUIL) -> list[dict]:
    """Le dédoublonnage complet : blocage, score, regroupement, fiche de référence (une par client)."""
    paires, _ = paires_candidates(fiches)
    retenues = [(a, b) for a, b in sorted(paires) if score_correspondance(fiches[a], fiches[b]) >= seuil]
    return [fiche_de_reference([{"id": i, **fiches[i]} for i in groupe]) for groupe in regrouper(fiches, retenues)]


# ============================================================================ Tests (à la pytest)


def test_la_distance_compte_les_trois_operations():
    assert distance_edition("chat", "chats") == 1 and distance_edition("kitten", "sitting") == 3
    assert distance_edition("", "abc") == 3 and distance_edition("abc", "") == 3
    assert distance_edition("dupont jean", "dupond jean") == 1
    assert distance_edition("ab", "ba") == 2  # une inversion = deux remplacements (ce n'est pas Damerau)
    assert distance_edition("a" * 1500, "b" + "a" * 1500) == 1  # pas de RecursionError


def test_regrouper_est_transitif_et_garde_les_seuls():
    assert regrouper(["C", "B", "A", "D"], [("A", "B"), ("C", "B")]) == [["A", "B", "C"], ["D"]]
    chaine = [(i + 1, i) for i in range(50_000)]
    assert regrouper(range(50_001), chaine) == [list(range(50_001))]


def test_le_blocage_trouve_les_paires_de_l_exemple():
    paires, trop_gros = paires_candidates(fiches_exemple(), taille_max=2)
    assert ("F01", "F02") in paires and ("F06", "F07") in paires and ("F01", "F03") in paires
    assert ("F09", "F10") not in paires  # le bloc « marie petit|75015 » (3 fiches) a été sauté…
    assert trop_gros == [(2, "dupre helene|75011", 3), (2, "marie petit|75015", 3)]  # … et c'est dit


def test_le_score_retrouve_les_bons_clients_de_l_exemple():
    f = fiches_exemple()
    assert score_correspondance(f["F01"], f["F02"]) == 7       # nom à l'envers, téléphone écrit autrement
    assert score_correspondance(f["F04"], f["F05"]) == -4      # même famille, deux personnes
    assert score_correspondance(f["F09"], f["F11"]) == 7       # le NaN ne compte pas
    assert score_correspondance({}, {}) == 0                   # rien à comparer : ni accord, ni désaccord


def test_la_fiche_de_reference_respecte_le_refus_des_sms():
    f = fiches_exemple()
    helene = fiche_de_reference([{"id": i, **f[i]} for i in ("F01", "F02", "F03")])
    assert helene["optin_sms"] is False and helene["inscription"] == date(2016, 9, 14)
    assert helene["resilie"] is False and helene["telephone"] == "06 12 34 56 78"
    assert helene["anciens_ids"] == ["F01", "F02", "F03"] and helene["id"] == "F01"


def test_la_qualite_se_compte_en_paires():
    verite = [["A", "B", "C"], ["D"]]
    tout_fusionne = qualite_dedoublonnage([["A", "B", "C", "D"]], verite)
    assert tout_fusionne["rappel"] == 1.0 and tout_fusionne["precision"] == 0.5
    rien = qualite_dedoublonnage([["A"], ["B"], ["C"], ["D"]], verite)
    assert rien["precision"] == 1.0 and rien["rappel"] == 0.0 and rien["f1"] == 0.0


def test_tout_assemble_retrouve_les_huit_clients():
    fiches = fiches_exemple()
    clients = dedoublonner(fiches)
    assert len(clients) == 8
    groupes = [c["anciens_ids"] for c in clients]
    assert qualite_dedoublonnage(groupes, VERITE_EXEMPLE)["f1"] == 1.0
    fiches, verite = generer_fiches(2000, graine=7)
    groupes = [c["anciens_ids"] for c in dedoublonner(fiches)]
    q = qualite_dedoublonnage(groupes, verite)
    assert q["precision"] > 0.95 and q["rappel"] > 0.8

"""Série 29 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_29.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours un
numéro de ligne décalé de un (les fichiers commencent à 1, les listes à 0), un « -5,0 » qui veut dire
« APRÈS la ligne 5 », ou un `nan == nan` qui vaut False.

Fil rouge de la série : comparer, c'est ALIGNER avant de juger.

1. on ne compare pas position par position : une ligne ajoutée en haut décale tout le reste. On cherche ce
   qui est COMMUN (la plus longue sous-suite commune, ou la clé d'une table), puis on regarde ce qui reste ;
2. un patch transporte du CONTEXTE, pas seulement des numéros de ligne : on vérifie ce contexte avant de
   toucher au fichier, et on refuse tout plutôt que d'appliquer à moitié ;
3. une fusion a besoin de TROIS versions : sans la base, une ligne supprimée par l'un ressemble à une ligne
   ajoutée par l'autre. Et une fusion ne choisit jamais en silence : un conflit se MONTRE ;
4. « égal » se définit : NaN et NaN, 0.1 + 0.2 et 0.3, 1 et 1.0 sont égaux pour une table ; "1" et 1,
   True et 1, une clé absente et une clé à None ne le sont pas.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import copy
import math
import numbers
import re

import pandas as pd

from defis.versions import (  # noqa: F401
    BASE,
    EUX,
    NOUS,
    SEPARATEUR,
    PatchRefuse,
    config_exemple,
    generer_versions,
    montrer,
    tables_exemple,
    versions_en_conflit,
    versions_exemple,
)

# ============================================================================ Défi 176 : difference


def _verifier_lignes(lignes, nom: str) -> list[str]:
    if isinstance(lignes, (str, bytes)):
        raise TypeError(f"{nom} doit être une LISTE de lignes, pas un texte : utilisez texte.splitlines()")
    lignes = list(lignes)
    for ligne in lignes:
        if not isinstance(ligne, str):
            raise TypeError(f"{nom} contient {ligne!r} : chaque ligne doit être un texte")
    return lignes


def difference(ancien: list[str], nouveau: list[str]) -> list[tuple[str, str]]:
    """Le plus court script de modifications : (" ", ligne) gardée, ("-", ligne) supprimée, ("+", ligne) ajoutée."""
    a = _verifier_lignes(ancien, "ancien")
    b = _verifier_lignes(nouveau, "nouveau")

    # 1. Le début et la fin communs : souvent 99 % du fichier. Les retirer AVANT le tableau change tout :
    #    3 000 lignes dont une seule change, c'est 3 000 comparaisons au lieu de 9 millions de cases.
    debut = 0
    while debut < len(a) and debut < len(b) and a[debut] == b[debut]:
        debut += 1
    fin = 0
    while fin < len(a) - debut and fin < len(b) - debut and a[len(a) - 1 - fin] == b[len(b) - 1 - fin]:
        fin += 1
    ma, mb = a[debut:len(a) - fin], b[debut:len(b) - fin]

    # 2. La plus longue sous-suite commune du MILIEU. lcs[i][j] = sa longueur entre ma[i:] et mb[j:] (les
    #    FINS des listes) : on peut alors reconstruire le script en avançant du début vers la fin.
    n, m = len(ma), len(mb)
    lcs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if ma[i] == mb[j]:
                lcs[i][j] = lcs[i + 1][j + 1] + 1
            else:
                lcs[i][j] = max(lcs[i + 1][j], lcs[i][j + 1])

    milieu = []
    i = j = 0
    while i < n and j < m:
        if ma[i] == mb[j]:
            milieu.append((" ", ma[i]))
            i, j = i + 1, j + 1
        elif lcs[i + 1][j] >= lcs[i][j + 1]:
            milieu.append(("-", ma[i]))
            i += 1
        else:
            milieu.append(("+", mb[j]))
            j += 1
    milieu += [("-", ligne) for ligne in ma[i:]] + [("+", ligne) for ligne in mb[j:]]

    # 3. Dans chaque bloc de modifications, les « - » d'abord, puis les « + » (comme git diff) : le script
    #    reste aussi court, et un humain lit « voici ce qui part, voici ce qui le remplace ».
    rangees, bloc = [], []
    for op, ligne in milieu:
        if op == " ":
            rangees += sorted(bloc, key=lambda o: o[0] == "+")   # sorted est STABLE : l'ordre des lignes reste
            bloc = []
            rangees.append((op, ligne))
        else:
            bloc.append((op, ligne))
    rangees += sorted(bloc, key=lambda o: o[0] == "+")

    return [(" ", ligne) for ligne in a[:debut]] + rangees + [(" ", ligne) for ligne in a[len(a) - fin:]]


# ============================================================================ Défi 177 : diff_unifie


def _plage(debut: int, longueur: int) -> str:
    """« 5,3 » ; « 5 » pour une seule ligne ; « 4,0 » pour aucune ligne APRÈS la ligne 4."""
    if longueur == 0:
        return f"{debut - 1},0"
    return str(debut) if longueur == 1 else f"{debut},{longueur}"


def diff_unifie(operations: list[tuple[str, str]], nom: str = "fichier", contexte: int = 3) -> list[str]:
    """Les lignes du diff unifié (le format de `git diff`) correspondant aux opérations."""
    if isinstance(contexte, bool) or not isinstance(contexte, int) or contexte < 0:
        raise ValueError(f"contexte doit être un entier >= 0, pas {contexte!r}")
    ops = [tuple(o) for o in operations]
    changements = [k for k, (op, _) in enumerate(ops) if op != " "]
    if not changements:
        return []

    # Les numéros (à partir de 1) de la prochaine ligne de chaque fichier, AVANT chaque opération.
    num_ancien, num_nouveau = [], []
    x = y = 1
    for op, _ in ops:
        num_ancien.append(x)
        num_nouveau.append(y)
        x += op != "+"
        y += op != "-"

    # Regrouper : deux changements séparés par au plus 2 × contexte lignes communes vont dans le même bloc
    # (sinon leurs contextes se chevaucheraient ou se toucheraient).
    groupes = [[changements[0], changements[0]]]
    for k in changements[1:]:
        if k - groupes[-1][1] - 1 <= 2 * contexte:
            groupes[-1][1] = k
        else:
            groupes.append([k, k])

    sortie = [f"--- a/{nom}", f"+++ b/{nom}"]
    for premier, dernier in groupes:
        debut = max(0, premier - contexte)            # max(0, …) : un indice négatif repartirait de la FIN
        fin = min(len(ops), dernier + 1 + contexte)
        bloc = ops[debut:fin]
        n_ancien = sum(op != "+" for op, _ in bloc)
        n_nouveau = sum(op != "-" for op, _ in bloc)
        sortie.append(f"@@ -{_plage(num_ancien[debut], n_ancien)} +{_plage(num_nouveau[debut], n_nouveau)} @@")
        sortie += [op + ligne for op, ligne in bloc]
    return sortie


# ============================================================================ Défi 178 : appliquer_patch

_EN_TETE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*")


def _lire_blocs(patch: list[str]) -> list[tuple[int, int, list[str], list[str]]]:
    """[(debut_ancien, longueur_ancien, lignes_attendues, lignes_nouvelles), …] ; PatchRefuse si abîmé."""
    blocs = []
    k = 0
    while k < len(patch) and not patch[k].startswith("@@"):
        k += 1                                       # « --- a/… », « +++ b/… », « diff --git … » : ignorés
    while k < len(patch):
        en_tete = _EN_TETE.fullmatch(patch[k])
        if en_tete is None:
            raise PatchRefuse(f"ligne {k + 1} du patch : en-tête de bloc attendu, trouvé {patch[k]!r}")
        debut, longueur = int(en_tete[1]), int(en_tete[2] or 1)          # « -5 » veut dire « -5,1 »
        reste_ancien, reste_nouveau = longueur, int(en_tete[4] or 1)
        k += 1
        attendues, nouvelles = [], []
        # On COMPTE les lignes annoncées par l'en-tête : une ligne supprimée « -- total » s'écrit
        # « --- total », et ne doit pas être prise pour l'en-tête d'un autre fichier.
        while reste_ancien > 0 or reste_nouveau > 0:
            if k >= len(patch):
                raise PatchRefuse(f"bloc {len(blocs) + 1} : le patch s'arrête avant la fin du bloc")
            ligne = patch[k]
            op, texte = ligne[:1], ligne[1:]
            if op == " ":
                attendues.append(texte)
                nouvelles.append(texte)
                reste_ancien, reste_nouveau = reste_ancien - 1, reste_nouveau - 1
            elif op == "-":
                attendues.append(texte)
                reste_ancien -= 1
            elif op == "+":
                nouvelles.append(texte)
                reste_nouveau -= 1
            else:
                raise PatchRefuse(f"ligne {k + 1} du patch : {ligne!r} ne commence ni par ' ', ni par '-', ni par '+'")
            if reste_ancien < 0 or reste_nouveau < 0:
                raise PatchRefuse(f"bloc {len(blocs) + 1} : plus de lignes que n'en annonce l'en-tête")
            k += 1
        blocs.append((debut, longueur, attendues, nouvelles))
    return blocs


def appliquer_patch(lignes: list[str], patch: list[str]) -> list[str]:
    """Le fichier `lignes` après application du patch (diff unifié). PatchRefuse s'il ne s'applique pas."""
    lignes = _verifier_lignes(lignes, "lignes")
    patch = _verifier_lignes(patch, "patch")
    blocs = _lire_blocs(patch)                       # tout lire et tout vérifier AVANT de construire

    sortie = []
    lues = 0              # lignes de l'ANCIEN fichier déjà recopiées : un bloc ne peut pas commencer avant
    decalage = 0          # de combien le bloc précédent a dû être déplacé : le suivant l'est sans doute aussi
    for numero, (debut, longueur, attendues, nouvelles) in enumerate(blocs, start=1):
        # Les numéros de l'en-tête parlent de l'ANCIEN fichier, qui commence à 1. « -5,0 » (rien à retirer)
        # veut dire « APRÈS la ligne 5 », donc l'indice 5 ; « -5,3 » commence à la ligne 5, l'indice 4.
        prevu = (debut if longueur == 0 else debut - 1) + decalage
        trouve = None
        for ecart in range(len(lignes) + 1):
            for position in ((prevu,) if ecart == 0 else (prevu + ecart, prevu - ecart)):
                if lues <= position <= len(lignes) - longueur and lignes[position:position + longueur] == attendues:
                    trouve = position
                    break
            if trouve is not None:
                break
        if trouve is None:
            raise PatchRefuse(f"bloc {numero} : les lignes attendues autour de la ligne {debut} sont introuvables")
        decalage = trouve - (prevu - decalage)
        sortie += lignes[lues:trouve] + nouvelles
        lues = trouve + longueur
    return sortie + lignes[lues:]


# ============================================================================ Défi 179 : fusion_trois_voies


def _correspondances(base: list[str], autre: list[str]) -> dict[int, int]:
    """{indice dans base: indice dans autre} pour chaque ligne gardée (" ") par le plus court script."""
    paires, i, j = {}, 0, 0
    for op, _ in difference(base, autre):
        if op == " ":
            paires[i] = j
        i += op != "+"
        j += op != "-"
    return paires


def fusion_trois_voies(base: list[str], nous: list[str], eux: list[str]) -> tuple[list[str], int]:
    """(lignes fusionnées, nombre de conflits), comme `git merge-file --diff3`."""
    base = _verifier_lignes(base, "base")
    nous = _verifier_lignes(nous, "nous")
    eux = _verifier_lignes(eux, "eux")
    avec_nous, avec_eux = _correspondances(base, nous), _correspondances(base, eux)

    sortie, conflits = [], 0
    i = j = k = 0
    while True:
        # La prochaine ligne de base que les DEUX ont gardée : un point d'appui sûr (stable).
        i2 = i
        while i2 < len(base) and not (i2 in avec_nous and i2 in avec_eux):
            i2 += 1
        j2, k2 = (avec_nous[i2], avec_eux[i2]) if i2 < len(base) else (len(nous), len(eux))

        # Entre deux points d'appui : ce que chacun a fait de base[i:i2].
        b, n, e = base[i:i2], nous[j:j2], eux[k:k2]
        if n == b:
            sortie += e                  # seul Malik a touché à ce morceau (ou personne)
        elif e == b:
            sortie += n                  # seule Inès y a touché
        elif n == e:
            sortie += n                  # la même modification des deux côtés : pas un conflit
        else:
            sortie += [NOUS, *n, BASE, *b, SEPARATEUR, *e, EUX]
            conflits += 1                # deux modifications différentes : on MONTRE, on ne choisit pas

        if i2 == len(base):
            return sortie, conflits
        sortie.append(base[i2])
        i, j, k = i2 + 1, j2 + 1, k2 + 1


# ============================================================================ Défi 180 : comparer_tables


def _absent(valeur) -> bool:
    try:
        return bool(pd.isna(valeur))
    except (TypeError, ValueError):      # une liste, un tableau : pas une valeur absente
        return False


def _nombre(valeur) -> bool:
    return isinstance(valeur, numbers.Number) and not isinstance(valeur, bool)


def _memes_valeurs(a, b) -> bool:
    if _absent(a) or _absent(b):
        return _absent(a) and _absent(b)                       # NaN, None, NaT : tous « absents »
    if _nombre(a) and _nombre(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)   # 0.1 + 0.2 et 0.3 ; 12 et 12.0
    if _nombre(a) or _nombre(b):
        return False                                            # "1" n'est pas 1 : le type a changé
    return bool(a == b)


def _python(valeur):
    if _absent(valeur):
        return None
    return valeur.item() if hasattr(valeur, "item") else valeur      # np.int64(3) -> 3


def _indexer(table: pd.DataFrame, cle: str, nom: str) -> pd.DataFrame:
    if cle not in table.columns:
        raise KeyError(f"la colonne clé {cle!r} manque dans {nom}")
    cles = table[cle]
    if cles.isna().any():
        raise ValueError(f"{nom} : une clé absente (NaN) ne désigne aucun client")
    doublons = cles[cles.duplicated()]
    if len(doublons):
        raise ValueError(f"{nom} : la clé {doublons.iloc[0]!r} apparaît plusieurs fois")
    return table.set_index(cle)          # une NOUVELLE table : l'appelant garde la sienne intacte


def comparer_tables(avant: pd.DataFrame, apres: pd.DataFrame, cle: str) -> dict:
    """Ce qui a changé entre deux tables, ligne par ligne (alignées par `cle`) et colonne par colonne."""
    a, b = _indexer(avant, cle, "avant"), _indexer(apres, cle, "apres")
    communes = [c for c in a.columns if c in b.columns]
    modifies = {}
    for valeur_cle in sorted(set(a.index) & set(b.index)):
        changements = {}
        for colonne in communes:
            x, y = a.at[valeur_cle, colonne], b.at[valeur_cle, colonne]
            if not _memes_valeurs(x, y):
                changements[colonne] = (_python(x), _python(y))
        if changements:
            modifies[valeur_cle] = changements
    return {
        "ajoutes": sorted(set(b.index) - set(a.index)),
        "supprimes": sorted(set(a.index) - set(b.index)),
        "modifies": modifies,
        "colonnes_ajoutees": sorted(c for c in b.columns if c not in a.columns),
        "colonnes_supprimees": sorted(c for c in a.columns if c not in b.columns),
    }


# ============================================================================ Défi 181 : fusion_config

_ABSENTE = object()   # « la clé n'existe pas » : ni None (une vraie valeur, `null` en JSON), ni False, ni {}


def _identiques(a, b) -> bool:
    """Même type ET même valeur, jusque dans les listes et les dict : True n'est pas 1, 1 n'est pas 1.0."""
    if a is _ABSENTE or b is _ABSENTE:
        return a is b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_identiques(a[c], b[c]) for c in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_identiques(x, y) for x, y in zip(a, b))
    return a == b


def _fusion(base, nous: dict, eux: dict, chemin: str, conflits: list[str]) -> dict:
    base = base if isinstance(base, dict) else {}
    fusion = {}
    for cle in [*nous, *(c for c in eux if c not in nous), *(c for c in base if c not in nous and c not in eux)]:
        b, n, e = base.get(cle, _ABSENTE), nous.get(cle, _ABSENTE), eux.get(cle, _ABSENTE)
        ici = f"{chemin}.{cle}" if chemin else str(cle)
        if _identiques(n, e):
            choix = n                    # les deux d'accord (y compris : tous deux l'ont supprimée)
        elif _identiques(n, b):
            choix = e                    # seul « eux » a changé (ou supprimé) cette clé
        elif _identiques(e, b):
            choix = n
        elif isinstance(n, dict) and isinstance(e, dict):
            choix = _fusion(b, n, e, ici, conflits)        # deux sections modifiées : on descend dedans
        else:
            conflits.append(ici)
            choix = n                    # on garde « nous »… mais on le DIT
        if choix is not _ABSENTE:
            fusion[cle] = copy.deepcopy(choix)              # rien de partagé avec les dict reçus
    return fusion


def fusion_config(base: dict, nous: dict, eux: dict) -> tuple[dict, list[str]]:
    """(configuration fusionnée, chemins des clés en conflit triés), clé par clé, en descendant dans les dict."""
    for nom, valeur in (("base", base), ("nous", nous), ("eux", eux)):
        if not isinstance(valeur, dict):
            raise TypeError(f"{nom} doit être un dict, pas {type(valeur).__name__}")
    conflits: list[str] = []
    fusion = _fusion(base, nous, eux, "", conflits)
    return fusion, sorted(conflits)


# ============================================================================ Des tests, comme dans tests/


def test_difference_garde_l_ordre_et_les_doublons():
    ops = difference(["}", "", "}"], ["}", "x", "", "}"])
    assert ops == [(" ", "}"), ("+", "x"), (" ", ""), (" ", "}")]
    assert difference([], []) == [] and difference(["a"], []) == [("-", "a")]


def test_diff_unifie_une_ligne_et_un_fichier_vide():
    assert diff_unifie(difference(["a", "b", "c"], ["a", "B", "c"]), contexte=0) == [
        "--- a/fichier", "+++ b/fichier", "@@ -2 +2 @@", "-b", "+B"]
    assert diff_unifie(difference([], ["x"]), nom="n.txt") == ["--- a/n.txt", "+++ b/n.txt", "@@ -0,0 +1 @@", "+x"]


def test_un_patch_s_applique_et_se_refuse_en_entier():
    base, nous, _ = versions_exemple()
    patch = diff_unifie(difference(base, nous), contexte=1)
    assert appliquer_patch(base, patch) == nous
    assert appliquer_patch(["# en-tête ajouté"] * 3 + base, patch) == ["# en-tête ajouté"] * 3 + nous
    abime = base[:4] + base[5:]
    try:
        appliquer_patch(abime, patch)
    except PatchRefuse:
        pass
    else:
        raise AssertionError("un patch qui ne correspond pas doit être refusé")


def test_fusion_sans_conflit_puis_avec():
    base, nous, eux = versions_exemple()
    fusion, conflits = fusion_trois_voies(base, nous, eux)
    assert conflits == 0 and nous[1] in fusion and eux[-1] in fusion
    fusion, conflits = fusion_trois_voies(*versions_en_conflit())
    assert conflits == 1 and fusion.count(NOUS) == 1


def test_comparer_tables_ne_voit_que_les_vrais_changements():
    r = comparer_tables(*tables_exemple(), cle="client_id")
    assert r["modifies"] == {"C005": {"contrat": ("Mensuel", "Annuel")}}
    assert (r["ajoutes"], r["supprimes"]) == (["C009"], ["C006"])


def test_fusion_config_distingue_absent_et_none():
    fusion, conflits = fusion_config({"a": 1}, {"a": None}, {"a": 1})
    assert fusion == {"a": None} and conflits == []
    fusion, conflits = fusion_config({"a": 1}, {}, {"a": 1})
    assert fusion == {} and conflits == []
    fusion, conflits = fusion_config({"a": 1}, {}, {"a": 2})
    assert fusion == {} and conflits == ["a"]

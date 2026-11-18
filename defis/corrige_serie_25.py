"""Corrigé de la série 25 — Écrire un petit langage. À ouvrir APRÈS avoir cherché !

Les six fonctions forment une chaîne, comme dans un vrai compilateur (Python lui-même fait pareil) :

    texte ──decouper──▶ jetons ──analyser──▶ arbre ──┬── evaluer        (l'INTERPRÉTEUR)
                                                     ├── vers_texte     (l'imprimeur : l'aller-retour)
                                                     ├── verifier_regle (le VÉRIFICATEUR DE TYPES)
                                                     └── vers_sql       (le COMPILATEUR vers un autre langage)

Une fois l'arbre construit, on ne relit plus jamais le texte : chaque étage travaille sur la structure,
pas sur des chaînes de caractères. C'est ce qui rend le reste simple et sûr.
"""

from __future__ import annotations

import difflib
import math
import numbers

from defis.langage import (
    BOOLEENS,
    MOTS_CLES,
    OPERATEURS,
    ErreurSyntaxe,
    Jeton,
    base_clients,
    generer_arbre,
    generer_clients,
    schema_clients,
)

_CHIFFRES = "0123456789"
_GUILLEMETS_TYPO = "“”«»‘’„"
_PROFONDEUR_MAX = 100

# ============================================================================ Défi 152 : decouper


def decouper(texte: str) -> list[Jeton]:
    if not isinstance(texte, str):
        raise TypeError(f"une règle est un texte (str), pas {type(texte).__name__}")
    jetons: list[Jeton] = []
    i, n = 0, len(texte)
    while i < n:  # UN indice qui avance : jamais `texte = texte[1:]`, qui recopierait tout le reste à chaque fois
        c = texte[i]
        if c.isspace():  # espace, tabulation, retour à la ligne, espace insécable copiée depuis Word
            i += 1
        elif c in "(),":
            jetons.append(Jeton(c, c, i))
            i += 1
        elif c in "<>=!":
            deux = texte[i:i + 2]
            if deux in ("<=", ">=", "!=", "==", "<>"):  # le plus LONG d'abord : « <= » est un seul opérateur
                jetons.append(Jeton("OP", {"==": "=", "<>": "!="}.get(deux, deux), i))
                i += 2
            elif c == "!":
                raise ErreurSyntaxe("« ! » tout seul n'existe pas : écrivez « != » (différent de)", i)
            else:
                jetons.append(Jeton("OP", c, i))
                i += 1
        elif c in _CHIFFRES or (c == "-" and i + 1 < n and texte[i + 1] in _CHIFFRES):
            j = i + 1
            while j < n and texte[j] in _CHIFFRES:
                j += 1
            if j + 1 < n and texte[j] == "." and texte[j + 1] in _CHIFFRES:
                j += 1
                while j < n and texte[j] in _CHIFFRES:
                    j += 1
                jetons.append(Jeton("NOMBRE", float(texte[i:j]), i))
            else:
                jetons.append(Jeton("NOMBRE", int(texte[i:j]), i))
            i = j
        elif c in "\"'":
            morceaux, j = [], i + 1
            while True:
                if j >= n:  # on signale le guillemet OUVRANT : c'est lui qu'il faut aller voir
                    raise ErreurSyntaxe("texte jamais fermé : il manque le guillemet de fin", i)
                if texte[j] == "\\" and j + 1 < n:
                    morceaux.append(texte[j + 1])
                    j += 2
                elif texte[j] == c:
                    break
                else:
                    morceaux.append(texte[j])
                    j += 1
            jetons.append(Jeton("TEXTE", "".join(morceaux), i))
            i = j + 1
        elif c.isalpha() or c == "_":
            j = i + 1
            while j < n and (texte[j].isalnum() or texte[j] == "_"):
                j += 1
            mot = texte[i:j]  # le mot ENTIER d'abord : « etat » est un nom, pas « et » suivi de « at »
            bas = mot.lower()
            if bas in MOTS_CLES:
                jetons.append(Jeton(MOTS_CLES[bas], bas, i))
            elif bas in BOOLEENS:
                jetons.append(Jeton("BOOLEEN", BOOLEENS[bas], i))
            else:
                jetons.append(Jeton("NOM", mot, i))
            i = j
        elif c in _GUILLEMETS_TYPO:
            raise ErreurSyntaxe(f"« {c} » est un guillemet typographique (copié depuis Word ?) : écrivez des "
                                f"guillemets droits \"…\"", i)
        else:
            raise ErreurSyntaxe(f"caractère inattendu {c!r}", i)
    jetons.append(Jeton("FIN", None, n))
    return jetons


# ============================================================================ Défi 153 : analyser


def analyser(texte: str) -> tuple:
    jetons = decouper(texte)
    pos = 0
    profondeur = 0

    def voir() -> Jeton:
        return jetons[pos]

    def avancer() -> Jeton:
        nonlocal pos
        jeton = jetons[pos]
        pos += 1
        return jeton

    def entrer(jeton: Jeton) -> None:
        nonlocal profondeur
        profondeur += 1
        if profondeur > _PROFONDEUR_MAX:  # AVANT que Python ne lève RecursionError (ou pire, ne plante)
            raise ErreurSyntaxe(f"plus de {_PROFONDEUR_MAX} niveaux imbriqués : simplifiez la règle", jeton.position)

    def ou() -> tuple:  # le plus faible d'abord : « ou » sépare des morceaux faits de « et »
        enfants = [et()]
        while voir().type == "OU":  # une BOUCLE : 5 000 « ou » à la suite ne font pas 5 000 appels imbriqués
            avancer()
            enfants.append(et())
        return enfants[0] if len(enfants) == 1 else ("ou", *enfants)

    def et() -> tuple:
        enfants = [non()]
        while voir().type == "ET":
            avancer()
            enfants.append(non())
        return enfants[0] if len(enfants) == 1 else ("et", *enfants)

    def non() -> tuple:
        nonlocal profondeur
        if voir().type == "NON":
            entrer(avancer())
            condition = non()  # « non » s'applique à ce qui suit IMMÉDIATEMENT : une condition, pas tout un « et »
            profondeur -= 1
            return ("non", condition)
        return primaire()

    def primaire() -> tuple:
        nonlocal profondeur
        jeton = voir()
        if jeton.type == "(":
            entrer(avancer())
            interieur = ou()
            fermante = voir()
            if fermante.type == "FIN":
                raise ErreurSyntaxe("parenthèse ouverte ici, jamais fermée", jeton.position)
            if fermante.type != ")":
                raise ErreurSyntaxe(f"attendu « ) », « et » ou « ou », trouvé {_montrer(fermante)}", fermante.position)
            avancer()
            profondeur -= 1
            return interieur
        if jeton.type == "NOM":
            avancer()
            suivant = voir()
            if suivant.type == "OP":
                avancer()
                return (suivant.valeur, jeton.valeur, valeur())
            if suivant.type == "DANS":
                avancer()
                ouvrante = voir()
                if ouvrante.type != "(":
                    raise ErreurSyntaxe(f"après « dans », attendu « ( », trouvé {_montrer(ouvrante)}", ouvrante.position)
                avancer()
                valeurs = [valeur()]
                while voir().type == ",":
                    avancer()
                    valeurs.append(valeur())
                if voir().type == "FIN":
                    raise ErreurSyntaxe("parenthèse ouverte ici, jamais fermée", ouvrante.position)
                if voir().type != ")":
                    raise ErreurSyntaxe(f"attendu « , » ou « ) », trouvé {_montrer(voir())}", voir().position)
                avancer()
                return ("dans", jeton.valeur, tuple(valeurs))
            raise ErreurSyntaxe(f"après la colonne « {jeton.valeur} », attendu un opérateur (=, !=, <, <=, >, >=) ou "
                                f"« dans », trouvé {_montrer(suivant)}", suivant.position)
        if jeton.type in ("NOMBRE", "TEXTE", "BOOLEEN"):
            raise ErreurSyntaxe("une condition commence par le nom d'une colonne : écrivez « anciennete < 6 », "
                                "pas « 6 > anciennete »", jeton.position)
        if jeton.type == "FIN":
            raise ErreurSyntaxe("règle vide" if pos == 0 else "règle incomplète : une condition est attendue ici",
                                jeton.position)
        raise ErreurSyntaxe(f"attendu une condition, trouvé {_montrer(jeton)}", jeton.position)

    def valeur():
        jeton = voir()
        if jeton.type in ("NOMBRE", "TEXTE", "BOOLEEN"):
            avancer()
            return jeton.valeur
        if jeton.type == "NOM":
            raise ErreurSyntaxe(f"attendu une valeur, trouvé le nom « {jeton.valeur} » : un texte s'écrit entre "
                                f"guillemets (\"{jeton.valeur}\")", jeton.position)
        raise ErreurSyntaxe(f"attendu une valeur (un nombre, un texte entre guillemets, vrai ou faux), trouvé "
                            f"{_montrer(jeton)}", jeton.position)

    arbre = ou()
    reste = voir()
    if reste.type == ")":
        raise ErreurSyntaxe("parenthèse fermante en trop", reste.position)
    if reste.type != "FIN":  # sans ce test, « a < 1 b < 2 » serait lu « a < 1 » et la fin, ignorée en silence
        raise ErreurSyntaxe(f"{_montrer(reste)} en trop après une condition complète : un « et » ou un « ou » "
                            f"oublié ?", reste.position)
    return arbre


def _montrer(jeton: Jeton) -> str:
    return "la fin de la règle" if jeton.type == "FIN" else f"« {jeton.valeur} »"


# ============================================================================ Défi 154 : evaluer


def _inconnu(valeur) -> bool:
    return valeur is None or (isinstance(valeur, float) and math.isnan(valeur))  # np.float64 hérite de float


def _genre(valeur) -> str:
    if isinstance(valeur, bool):  # AVANT les nombres : True est un int pour Python (True == 1)
        return "booleen"
    if isinstance(valeur, numbers.Real):  # int, float, et aussi numpy.int64 / numpy.float64
        return "nombre"
    if isinstance(valeur, str):
        return "texte"
    raise TypeError(f"valeur de type {type(valeur).__name__} : ni un nombre, ni un texte, ni un booléen")


def _comparable(colonne: str, a, b) -> str:
    ga, gb = _genre(a), _genre(b)
    if ga != gb:
        raise TypeError(f"« {colonne} » vaut {a!r} ({ga}), comparée à {b!r} ({gb})")
    return ga


_COMPARER = {
    "=": lambda a, b: a == b, "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
}


def evaluer(arbre: tuple, client) -> bool | None:
    noeud = arbre[0]
    if noeud == "et":
        resultat = True
        for enfant in arbre[1:]:
            r = evaluer(enfant, client)
            if r is False:
                return False  # court-circuit : la suite n'est même pas lue
            if r is None:
                resultat = None  # inconnu… mais un False plus loin l'emporterait encore
        return resultat
    if noeud == "ou":
        resultat = False
        for enfant in arbre[1:]:
            r = evaluer(enfant, client)
            if r is True:
                return True
            if r is None:
                resultat = None
        return resultat
    if noeud == "non":
        r = evaluer(arbre[1], client)
        return None if r is None else not r
    if noeud == "dans":
        _, colonne, valeurs = arbre
        valeur = client[colonne]  # KeyError si la colonne n'existe pas : une faute de frappe n'est pas un « inconnu »
        if _inconnu(valeur):
            return None
        for v in valeurs:
            _comparable(colonne, valeur, v)
        return any(bool(valeur == v) for v in valeurs)
    if noeud in _COMPARER:
        op, colonne, attendu = arbre
        valeur = client[colonne]
        if _inconnu(valeur):
            return None
        genre = _comparable(colonne, valeur, attendu)
        if genre == "booleen" and op not in ("=", "!="):
            raise TypeError(f"« {colonne} » est un booléen : « {op} » n'a pas de sens")
        return bool(_COMPARER[op](valeur, attendu))  # bool(…) : numpy.bool_ n'est pas True (`is True` échouerait)
    raise ValueError(f"nœud inconnu : {noeud!r}")


# ============================================================================ Défi 155 : vers_texte

_PRIORITE = {"ou": 1, "et": 2}


def _nom_lisible(colonne) -> str:
    if (not isinstance(colonne, str) or not colonne or not (colonne[0].isalpha() or colonne[0] == "_")
            or not all(c.isalnum() or c == "_" for c in colonne)
            or colonne.lower() in MOTS_CLES or colonne.lower() in BOOLEENS):
        raise ValueError(f"la colonne {colonne!r} ne peut pas s'écrire dans une règle (elle ne se relirait pas comme un nom)")
    return colonne


def _valeur_en_texte(v) -> str:
    if isinstance(v, bool):  # AVANT int : str(True) donnerait « True », relu comme un NOM
        return "vrai" if v else "faux"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"{v!r} ne peut pas s'écrire dans une règle")
        return repr(v)  # le plus court texte qui redonne EXACTEMENT ce float (pas de round, pas de :.2f)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'  # la barre oblique D'ABORD
    raise ValueError(f"valeur impossible à écrire : {v!r}")


def vers_texte(arbre: tuple) -> str:
    noeud = arbre[0]
    if noeud in ("et", "ou"):
        if len(arbre) < 3:
            raise ValueError(f"« {noeud} » demande au moins deux conditions : {arbre!r}")
        morceaux = []
        for enfant in arbre[1:]:
            texte = vers_texte(enfant)
            # Un « ou » dans un « et » : la priorité l'exige. Un « et » dans un « et » : sinon, en relisant, on
            # obtiendrait UN « et » à trois conditions, plus le même arbre.
            if enfant[0] in _PRIORITE and _PRIORITE[enfant[0]] <= _PRIORITE[noeud]:
                texte = f"({texte})"
            morceaux.append(texte)
        return f" {noeud} ".join(morceaux)
    if noeud == "non":
        texte = vers_texte(arbre[1])
        return f"non ({texte})" if arbre[1][0] in _PRIORITE else f"non {texte}"
    if noeud == "dans":
        _, colonne, valeurs = arbre
        if not valeurs:
            raise ValueError("« dans » demande au moins une valeur")
        return f"{_nom_lisible(colonne)} dans ({', '.join(_valeur_en_texte(v) for v in valeurs)})"
    if noeud in OPERATEURS:
        op, colonne, valeur = arbre
        return f"{_nom_lisible(colonne)} {op} {_valeur_en_texte(valeur)}"
    raise ValueError(f"nœud inconnu : {noeud!r}")


# ============================================================================ Défi 156 : verifier_regle

_NOMS_GENRES = {"nombre": "un nombre", "texte": "un texte", "booleen": "un booléen"}


def verifier_regle(arbre: tuple, schema: dict) -> list[str]:
    problemes: list[str] = []
    pile = [arbre]  # un parcours avec une pile, de gauche à droite : on CONTINUE après chaque problème
    while pile:
        noeud = pile.pop()
        sorte = noeud[0]
        if sorte in ("et", "ou", "non"):
            pile.extend(reversed(noeud[1:]))  # à l'envers, pour ressortir dans l'ordre de lecture
            continue
        if sorte == "dans":
            _, colonne, valeurs = noeud
        elif sorte in OPERATEURS:
            _, colonne, valeur = noeud
            valeurs = (valeur,)
        else:
            raise ValueError(f"nœud inconnu : {sorte!r}")
        if colonne not in schema:
            message = f"colonne « {colonne} » inconnue"
            proches = ([c for c in schema if c.lower() == colonne.lower()]
                       or difflib.get_close_matches(colonne, list(schema), n=1))
            if proches:
                message += f" : vouliez-vous dire « {proches[0]} » ?"
            problemes.append(message)
            continue
        attendu = schema[colonne]
        fautive = next((v for v in valeurs if _genre(v) != attendu), None)
        if fautive is not None:
            problemes.append(f"« {colonne} » est {_NOMS_GENRES[attendu]}, mais elle est comparée à "
                             f"{_valeur_en_texte(fautive)}, qui est {_NOMS_GENRES[_genre(fautive)]}")
        elif attendu == "booleen" and sorte not in ("=", "!=", "dans"):
            problemes.append(f"« {colonne} » est un booléen : « {sorte} » n'a pas de sens (utilisez = ou !=)")
    return problemes


# ============================================================================ Défi 157 : vers_sql


def _colonne_sql(colonne: str) -> str:
    return '"' + colonne.replace('"', '""') + '"'  # un identifiant entre guillemets DOUBLES, « " » doublé


def vers_sql(arbre: tuple) -> tuple[str, list]:
    params: list = []

    def sql(noeud: tuple) -> str:
        sorte = noeud[0]
        if sorte in ("et", "ou"):
            if len(noeud) < 3:
                raise ValueError(f"« {sorte} » demande au moins deux conditions")
            mot = " AND " if sorte == "et" else " OR "
            return "(" + mot.join(sql(enfant) for enfant in noeud[1:]) + ")"  # toujours entre parenthèses : sûr
        if sorte == "non":
            return f"NOT ({sql(noeud[1])})"
        if sorte == "dans":
            _, colonne, valeurs = noeud
            if not valeurs:
                raise ValueError("« dans » demande au moins une valeur")
            params.extend(valeurs)
            return f"{_colonne_sql(colonne)} IN ({', '.join('?' * len(valeurs))})"
        if sorte in OPERATEURS:
            op, colonne, valeur = noeud
            params.append(valeur)  # la VALEUR voyage à part : jamais dans le texte de la requête
            return f"{_colonne_sql(colonne)} {'<>' if op == '!=' else op} ?"
        raise ValueError(f"nœud inconnu : {sorte!r}")

    return sql(arbre), params


# ============================================================================ Tests (à la pytest)


def test_decouper_prend_le_jeton_le_plus_long():
    assert [j.valeur for j in decouper("a<=1")] == ["a", "<=", 1, None]
    assert [j.type for j in decouper("etat et ouverture")] == ["NOM", "ET", "NOM", "FIN"]


def test_decouper_signale_le_guillemet_ouvrant():
    try:
        decouper('contrat = "mensuel')
    except ErreurSyntaxe as exc:
        assert exc.position == 10 and "guillemet" in exc.message
    else:
        raise AssertionError("un texte jamais fermé doit lever ErreurSyntaxe")


def test_analyser_respecte_les_priorites():
    a, b, c = ("=", "a", 1), ("=", "b", 2), ("=", "c", 3)
    assert analyser("a = 1 ou b = 2 et c = 3") == ("ou", a, ("et", b, c))
    assert analyser("non a = 1 et b = 2") == ("et", ("non", a), b)
    assert analyser("a = 1 et (b = 2 et c = 3)") == ("et", a, ("et", b, c))


def test_analyser_ne_plante_pas_sur_une_regle_malveillante():
    assert len(analyser(" ou ".join(f"a = {i}" for i in range(5000)))) == 5001
    try:
        analyser("(" * 5000 + "a = 1" + ")" * 5000)
    except ErreurSyntaxe as exc:
        assert exc.position == 100 and "imbriqu" in exc.message
    else:
        raise AssertionError("5 000 parenthèses imbriquées doivent lever ErreurSyntaxe")


def test_evaluer_suit_la_logique_de_sql():
    client = {"contrat": "mensuel", "region": None, "facture": float("nan")}
    assert evaluer(("et", ("=", "contrat", "annuel"), ("=", "region", "Nord")), client) is False
    assert evaluer(("ou", ("=", "contrat", "mensuel"), ("=", "region", "Nord")), client) is True
    assert evaluer(("non", ("!=", "facture", 6)), client) is None


def test_vers_texte_fait_l_aller_retour():
    for graine in range(50):
        arbre = generer_arbre(graine, 4)
        assert analyser(vers_texte(arbre)) == arbre


def test_verifier_regle_donne_tous_les_problemes():
    problemes = verifier_regle(("et", ("<", "ancienete", 6), ("<", "fibre", True)), schema_clients())
    assert len(problemes) == 2 and "anciennete" in problemes[0] and "<" in problemes[1]


def test_vers_sql_trouve_les_memes_clients_que_evaluer():
    clients = generer_clients(100, graine=1)
    conn = base_clients(clients)
    for graine in range(50):
        arbre = generer_arbre(graine, 3)
        sql, params = vers_sql(arbre)
        trouves = [i for (i,) in conn.execute(f"SELECT id FROM clients WHERE {sql} ORDER BY id", params)]
        assert trouves == [c["id"] for c in clients if evaluer(arbre, c) is True]

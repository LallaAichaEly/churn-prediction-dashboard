"""Série 25 (écrire un petit langage) : le correcteur est-il juste, utile et impossible à tromper ?

Ces tests NE notent PAS vos réponses (pour ça : `python scripts/defis.py --defi 152`). Ils vérifient que :

- chaque version boguée classique est refusée, avec un message qui nomme SON erreur ;
- d'autres écritures justes que celles du corrigé sont acceptées ;
- les exemples des docstrings, les outils communs et le corrigé sont justes.
"""

from __future__ import annotations

import difflib
import doctest
import math
import numbers
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defis.corrige_serie_25 as corrige  # noqa: E402
from defis.checks_serie_25 import SERIE  # noqa: E402
from defis.langage import (  # noqa: E402
    BOOLEENS,
    MOTS_CLES,
    OPERATEURS,
    ErreurSyntaxe,
    Jeton,
    afficher_arbre,
    base_clients,
    clients_exemple,
    generer_arbre,
    generer_clients,
    regles_exemple,
    schema_clients,
    souligner,
)
from defis.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_defi  # noqa: E402

DEFIS = {d.fonction: d for d in SERIE.defis}


def _evaluer(nom, objet):
    return evaluer_defi(SimpleNamespace(__name__="faux", **{nom: objet}), DEFIS[nom])


def _echec(nom, objet, *mots):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    for mot in mots:
        assert mot in resultat.message, resultat.message


def _reussi(nom, objet):
    resultat = _evaluer(nom, objet)
    assert resultat.statut == REUSSI, resultat.message


# --------------------------------------------------------------------------- Défi 152 : decouper


def _lexer(defaut=None):
    """Un analyseur lexical écrit autrement que le corrigé, avec UN défaut au choix."""

    def decouper(texte):
        if defaut != "sans_typeerror" and not isinstance(texte, str):
            raise TypeError("str attendu")
        jetons, i, n = [], 0, len(texte)
        while i < n:
            if defaut == "tranches":
                texte[i:]  # une copie de tout le reste, à chaque jeton (comme `re.match(motif, texte[i:])`)
            c = texte[i]
            if c.isspace():
                if defaut == "espaces":
                    jetons.append(Jeton("ESPACE", c, i))
                i += 1
                continue
            if c in "(),":
                jetons.append(Jeton(c, c, i))
                i += 1
                continue
            if c in "<>=!":
                deux = texte[i:i + 2] if defaut != "court_d_abord" else ""
                if deux in ("<=", ">=", "!=", "==", "<>"):
                    valeur = deux if defaut == "egal_egal" else {"==": "=", "<>": "!="}.get(deux, deux)
                    jetons.append(Jeton("OP", valeur, i + (defaut == "position_1")))
                    i += 2
                    continue
                if c == "!":
                    raise ErreurSyntaxe("! seul", i)
                jetons.append(Jeton("OP", c, i + (defaut == "position_1")))
                i += 1
                continue
            if c.isdigit() or (c == "-" and defaut != "sans_negatif" and i + 1 < n and texte[i + 1].isdigit()):
                j = i + 1
                while j < n and texte[j].isdigit():
                    j += 1
                if j + 1 < n and texte[j] == "." and texte[j + 1].isdigit():
                    j += 1
                    while j < n and texte[j].isdigit():
                        j += 1
                morceau = texte[i:j]
                valeur = float(morceau) if "." in morceau or defaut == "float" else int(morceau)
                jetons.append(Jeton("NOMBRE", valeur, i + (defaut == "position_1")))
                i = j
                continue
            if c in "\"'":
                j, morceaux = i + 1, []
                while True:
                    if j >= n:
                        raise ErreurSyntaxe("guillemet de fin manquant", n if defaut == "erreur_a_la_fin" else i)
                    if texte[j] == "\\" and j + 1 < n and defaut != "sans_echappement":
                        morceaux.append(texte[j + 1])
                        j += 2
                    elif texte[j] == c:
                        break
                    else:
                        morceaux.append(texte[j])
                        j += 1
                valeur = "".join(morceaux)
                jetons.append(Jeton("TEXTE", c + valeur + c if defaut == "guillemets_gardes" else valeur, i))
                i = j + 1
                continue
            if c.isalpha() or c == "_":
                if defaut == "mot_cle_prefixe":
                    trouve = next((m for m in ("dans", "non", "et", "ou") if texte.startswith(m, i)), None)
                    if trouve:
                        jetons.append(Jeton(MOTS_CLES[trouve], trouve, i))
                        i += len(trouve)
                        continue
                j = i + 1
                while j < n and (texte[j].isalnum() or texte[j] == "_"):
                    j += 1
                mot = texte[i:j]
                cle = mot if defaut == "casse" else mot.lower()
                if cle in MOTS_CLES:
                    jetons.append(Jeton(MOTS_CLES[cle], cle, i))
                elif cle in BOOLEENS and defaut != "vrai_nom":
                    jetons.append(Jeton("BOOLEEN", BOOLEENS[cle], i))
                else:
                    jetons.append(Jeton("NOM", mot, i))
                i = j
                continue
            if c in "“”«»‘’„":
                raise ErreurSyntaxe("caractère inattendu" if defaut == "typo_muet" else "guillemets typographiques", i)
            raise ErreurSyntaxe(f"caractère inattendu {c!r}", i)
        if defaut != "sans_fin":
            jetons.append(Jeton("FIN", None, n))
        return jetons

    return decouper


@pytest.mark.parametrize("mutant,mots", [
    ("sans_fin", ["FIN"]),
    ("court_d_abord", ["UN opérateur", "le plus long"]),
    ("mot_cle_prefixe", ["COMMENCE comme un mot-clé"]),
    ("casse", ["majuscules"]),
    ("vrai_nom", ["BOOLEEN"]),
    ("float", ["int"]),
    ("egal_egal", ['"="', '"!="']),
    ("sans_negatif", ["lève ErreurSyntaxe"]),
    ("guillemets_gardes", ["SANS ses guillemets"]),
    ("position_1", ["à partir de 0"]),
    ("espaces", ["les espaces ne produisent aucun jeton"]),
    ("sans_echappement", ["decouper("]),
    ("erreur_a_la_fin", ["guillemet OUVRANT"]),
    ("typo_muet", ["ne contient pas 'guillemet'"]),
    ("sans_typeerror", ["TypeError"]),
    ("tranches", ["recopie", "motif.match(texte, i)"]),
])
def test_decouper_erreurs_classiques(mutant, mots):
    _echec("decouper", _lexer(mutant), *mots)


def test_decouper_boucle_sans_fin_arretee():
    def decouper(texte):
        jetons, i = [], 0
        while i < len(texte):
            if texte[i] == " ":
                continue  # oublie d'avancer
            j = i
            while j < len(texte) and texte[j] != " ":
                j += 1
            jetons.append(Jeton("NOM", texte[i:j], i))
            i = j
        return jetons + [Jeton("FIN", None, len(texte))]

    _echec("decouper", decouper, "ne s'arrête pas", "AVANCER")


# --------------------------------------------------------------------------- Défi 153 : analyser


def _parseur(defaut=None):
    """Un analyseur syntaxique (avec le lexer du corrigé), avec UN défaut au choix."""

    def analyser(texte):
        jetons = corrige.decouper(texte)
        etat = {"pos": 0, "profondeur": 0}

        def voir():
            return jetons[etat["pos"]]

        def avancer():
            etat["pos"] += 1
            return jetons[etat["pos"] - 1]

        def entrer(jeton):
            etat["profondeur"] += 1
            if defaut != "sans_limite" and etat["profondeur"] > 100:
                raise ErreurSyntaxe("trop de niveaux imbriqués", jeton.position)

        def groupe(sorte, enfants):
            if len(enfants) == 1 and defaut != "emballe":
                return enfants[0]
            if defaut == "binaire_gauche":
                arbre = enfants[0]
                for e in enfants[1:]:
                    arbre = (sorte, arbre, e)
                return arbre
            if defaut == "aplatit":
                enfants = [x for e in enfants for x in (e[1:] if e[0] == sorte else (e,))]
            return (sorte, *enfants)

        faible, fort = ("ET", "OU") if defaut == "priorite_inversee" else ("OU", "ET")

        def niveau_faible():
            if defaut == "recursion_ou":  # ou := et ("ou" ou)?, puis on aplatit : juste… sauf sur 5 000 conditions
                gauche = niveau_fort()
                if voir().type == faible:
                    avancer()
                    droite = niveau_faible()
                    return (faible.lower(), gauche, *(droite[1:] if droite[0] == faible.lower() and
                                                      not getattr(droite, "paren", False) else (droite,)))
                return gauche
            if defaut == "recursif_droite":
                gauche = niveau_fort()
                if voir().type == faible:
                    avancer()
                    return (faible.lower(), gauche, niveau_faible())
                return gauche
            enfants = [niveau_fort()]
            while voir().type == faible:
                avancer()
                enfants.append(niveau_fort())
            return groupe(faible.lower(), enfants)

        def niveau_fort():
            if defaut == "recursif_droite":
                gauche = non()
                if voir().type == fort:
                    avancer()
                    return (fort.lower(), gauche, niveau_fort())
                return gauche
            enfants = [non()]
            while voir().type == fort:
                avancer()
                enfants.append(non())
            return groupe(fort.lower(), enfants)

        def non():
            if voir().type == "NON":
                entrer(avancer())
                interieur = niveau_fort() if defaut == "non_large" else non()
                etat["profondeur"] -= 1
                return ("non", interieur)
            return primaire()

        def valeur():
            jeton = voir()
            if jeton.type in ("NOMBRE", "TEXTE", "BOOLEEN"):
                avancer()
                return jeton.valeur
            if jeton.type == "NOM" and defaut != "nom_muet":
                raise ErreurSyntaxe("un texte s'écrit entre guillemets", jeton.position)
            raise ErreurSyntaxe("valeur attendue", jeton.position)

        def primaire():
            jeton = voir()
            if jeton.type == "(":
                entrer(avancer())
                interieur = niveau_faible()
                if voir().type == "FIN":
                    position = len(texte) if defaut == "parenthese_a_la_fin" else jeton.position
                    raise ErreurSyntaxe("parenthèse jamais fermée", position)
                if voir().type != ")":
                    raise ErreurSyntaxe("« ) » attendue", voir().position)
                avancer()
                etat["profondeur"] -= 1
                if defaut == "recursion_ou" and interieur[0] in ("et", "ou"):
                    return _Paren(interieur)
                return interieur
            if jeton.type == "NOM":
                avancer()
                suivant = voir()
                if suivant.type == "OP":
                    avancer()
                    v = valeur()
                    return (jeton.valeur, suivant.valeur, v) if defaut == "feuille_inversee" else \
                        (suivant.valeur, jeton.valeur, v)
                if suivant.type == "DANS":
                    avancer()
                    ouvrante = voir()
                    if ouvrante.type != "(":
                        raise ErreurSyntaxe("« ( » attendue", ouvrante.position)
                    avancer()
                    valeurs = [valeur()]
                    while voir().type == ",":
                        avancer()
                        valeurs.append(valeur())
                    if voir().type == "FIN":
                        raise ErreurSyntaxe("parenthèse jamais fermée", ouvrante.position)
                    if voir().type != ")":
                        raise ErreurSyntaxe("« , » ou « ) » attendue", voir().position)
                    avancer()
                    return ("dans", jeton.valeur, valeurs if defaut == "listes" else tuple(valeurs))
                raise ErreurSyntaxe("opérateur attendu", suivant.position)
            if jeton.type in ("NOMBRE", "TEXTE", "BOOLEEN"):
                raise ErreurSyntaxe("une condition commence par une colonne", jeton.position)
            if jeton.type == "FIN":
                raise ErreurSyntaxe("règle vide" if etat["pos"] == 0 else "incomplète", jeton.position)
            raise ErreurSyntaxe("condition attendue", jeton.position)

        arbre = niveau_faible()
        if defaut != "ignore_la_fin" and voir().type != "FIN":
            raise ErreurSyntaxe("en trop", voir().position)
        return _sans_paren(arbre)

    return analyser


class _Paren(tuple):
    """Un groupe venu d'une parenthèse (pour que la version « récursive » ne l'aplatisse pas)."""

    paren = True


def _sans_paren(arbre):
    if isinstance(arbre, tuple) and arbre and arbre[0] in ("et", "ou", "non"):
        return (arbre[0], *(_sans_paren(e) for e in arbre[1:]))
    return arbre


@pytest.mark.parametrize("mutant,mots", [
    ("priorite_inversee", ["« ou » passe avant « et »"]),
    ("non_large", ["s'applique à tout le « et »"]),
    ("binaire_gauche", ["un nœud par « et »"]),
    ("recursif_droite", ["un appel récursif par « et »"]),
    ("aplatit", ["Vous fusionnez"]),
    ("emballe", ["une seule condition"]),
    ("feuille_inversee", ["(opérateur, colonne, valeur)"]),
    ("listes", ["TUPLES"]),
    ("ignore_la_fin", ["devrait lever ErreurSyntaxe", "renvoie"]),
    ("parenthese_a_la_fin", ["OUVRANTE", "au lieu de 0"]),
    ("nom_muet", ["ne contient pas 'guillemet'"]),
    ("recursion_ou", ["RecursionError", "BOUCLE"]),
    ("sans_limite", ["devrait lever ErreurSyntaxe à la position 100", "Comptez"]),
])
def test_analyser_erreurs_classiques(mutant, mots):
    _echec("analyser", _parseur(mutant), *mots)


def test_analyser_sans_defaut_la_fabrique_est_juste():
    _reussi("analyser", _parseur())


def test_analyser_valeur_erronee_au_lieu_de_erreur_syntaxe():
    def analyser(texte):
        try:
            return corrige.analyser(texte)
        except ErreurSyntaxe as exc:
            raise ValueError(exc.message) from None

    _echec("analyser", analyser, "lève ValueError", "ErreurSyntaxe(message, position)")


def test_analyser_arguments_inverses():
    def analyser(texte):
        try:
            return corrige.analyser(texte)
        except ErreurSyntaxe as exc:
            raise ErreurSyntaxe(exc.position, exc.message) from None

    _echec("analyser", analyser, "d'abord le message")


# --------------------------------------------------------------------------- Défi 154 : evaluer


def _interprete(defaut=None):
    def inconnu(v):
        if defaut == "zero_inconnu":  # `if not valeur` : 0 passe pour une absence
            return v is None or (isinstance(v, float) and math.isnan(v)) or (not v and v is not False)
        if defaut == "nan_non_gere":
            return v is None
        return v is None or (isinstance(v, float) and math.isnan(v))

    def genre(v):
        if isinstance(v, bool) and defaut != "bool_nombre":
            return "b"
        if isinstance(v, numbers.Real):
            return "n"
        return "t"

    def evaluer(arbre, client):
        sorte = arbre[0]
        if sorte in ("et", "ou"):
            if defaut == "all_any":
                resultats = [evaluer(e, client) for e in arbre[1:]]
                return all(resultats) if sorte == "et" else any(resultats)
            decisif = sorte == "ou"
            resultats = [evaluer(e, client) for e in arbre[1:]] if defaut == "pas_de_court_circuit" else None
            resultat = not decisif
            for i, enfant in enumerate(arbre[1:]):
                r = resultats[i] if resultats is not None else evaluer(enfant, client)
                if r is decisif:
                    return decisif
                if r is None:
                    resultat = None
            return resultat
        if sorte == "non":
            r = evaluer(arbre[1], client)
            if defaut == "non_none":
                return not r
            return None if r is None else not r
        if sorte not in OPERATEURS and sorte != "dans":
            raise ValueError(sorte)
        colonne = arbre[1]
        v = client.get(colonne) if defaut == "get" else client[colonne]
        if inconnu(v):
            return None
        valeurs = arbre[2] if sorte == "dans" else (arbre[2],)
        for w in valeurs:
            if genre(v) != genre(w):
                raise TypeError("genres différents" if defaut == "type_sans_colonne" else f"{colonne} : genres différents")
        if genre(v) == "b" and sorte not in ("=", "!=", "dans"):
            raise TypeError(f"{colonne} : booléen")
        if sorte == "dans":
            return any(v == w for w in valeurs) if defaut == "numpy" else any(bool(v == w) for w in valeurs)
        a, b = v, arbre[2]
        r = {"=": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[sorte]
        return r if defaut == "numpy" else bool(r)

    return evaluer


@pytest.mark.parametrize("mutant,mots", [
    ("all_any", ["traitent None comme False"]),
    ("nan_non_gere", ["NaN", "math.isnan"]),
    ("zero_inconnu", ["0 est une vraie valeur"]),
    ("non_none", ["NON INCONNU reste INCONNU"]),
    ("pas_de_court_circuit", ["lit les colonnes", "`return False` dès le premier faux"]),
    ("get", ["KeyError", "client.get(colonne)"]),
    ("bool_nombre", ["devrait lever TypeError", "True == 1"]),
    ("type_sans_colonne", ["ne contient pas 'anciennete'"]),
    ("numpy", ["np.True_", "bool(...)"]),
])
def test_evaluer_erreurs_classiques(mutant, mots):
    _echec("evaluer", _interprete(mutant), *mots)


def test_evaluer_sans_defaut_la_fabrique_est_juste():
    _reussi("evaluer", _interprete())


def test_evaluer_lire_deux_fois_la_meme_colonne_est_accepte():
    def evaluer(arbre, client):
        if arbre[0] in OPERATEURS and arbre[1] in client:
            client[arbre[1]]  # une lecture « pour rien » avant la vraie : pas une faute
        return corrige.evaluer(arbre, client)

    _reussi("evaluer", evaluer)


# --------------------------------------------------------------------------- Défi 155 : vers_texte


def _imprimeur(defaut=None):
    prio = {"ou": 1, "et": 2}

    def valeur(v):
        if isinstance(v, bool) and defaut != "str_bool":
            return "vrai" if v else "faux"
        if isinstance(v, float) and not isinstance(v, bool):
            if not math.isfinite(v):
                raise ValueError(v)
            return f"{v:.2f}" if defaut == "deux_decimales" else repr(v)
        if isinstance(v, int):
            return str(v)
        if defaut == "echappement_inverse":
            return '"' + v.replace('"', '\\"').replace("\\", "\\\\") + '"'
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def nom(colonne):
        if defaut != "sans_controle_nom" and not (
                isinstance(colonne, str) and colonne and (colonne[0].isalpha() or colonne[0] == "_")
                and all(c.isalnum() or c == "_" for c in colonne)
                and colonne.lower() not in MOTS_CLES and colonne.lower() not in BOOLEENS):
            raise ValueError(colonne)
        return colonne

    def vers_texte(arbre, parent=0):
        sorte = arbre[0]
        if sorte in prio:
            if len(arbre) < 3:
                raise ValueError(arbre)
            seuil = {"strict": prio[sorte] - 1, "toujours": 99}.get(defaut, prio[sorte])
            texte = f" {sorte} ".join(vers_texte(e, seuil) for e in arbre[1:])
            return f"({texte})" if prio[sorte] <= parent else texte
        if sorte == "non":
            return f"non {vers_texte(arbre[1], 0 if defaut == 'non_nu' else 99)}"
        if sorte == "dans":
            if not arbre[2]:
                raise ValueError(arbre)
            return f"{nom(arbre[1])} dans ({', '.join(valeur(v) for v in arbre[2])})"
        if sorte in OPERATEURS:
            return f"{nom(arbre[1])} {sorte} {valeur(arbre[2])}"
        raise ValueError(sorte)

    return lambda arbre: vers_texte(arbre)


@pytest.mark.parametrize("mutant,mots", [
    ("str_bool", ["`vrai` et `faux`"]),
    ("deux_decimales", ["repr(v)"]),
    ("echappement_inverse", ["D'ABORD"]),
    ("strict", ["Il manque des parenthèses", "`<=`"]),
    ("toujours", ["Trop de parenthèses"]),
    ("non_nu", ["vers_texte(('non', ('et'"]),
    ("sans_controle_nom", ["devrait lever ValueError", "date début"]),
])
def test_vers_texte_erreurs_classiques(mutant, mots):
    _echec("vers_texte", _imprimeur(mutant), *mots)


def test_vers_texte_sans_defaut_la_fabrique_est_juste():
    _reussi("vers_texte", _imprimeur())


# --------------------------------------------------------------------------- Défi 156 : verifier_regle


def _verificateur(defaut=None):
    def genre(v):
        if isinstance(v, bool) and defaut != "bool_nombre":
            return "booleen"
        if isinstance(v, (int, float)):
            return "nombre"
        return "texte"

    def suggestion(colonne, schema):
        if defaut == "sans_suggestion":
            return None
        if defaut == "suggestion_toujours":
            return max(schema, key=lambda c: difflib.SequenceMatcher(None, colonne, c).ratio())
        if defaut != "sans_casse":
            memes = [c for c in schema if c.lower() == colonne.lower()]
            if memes:
                return memes[0]
        proches = difflib.get_close_matches(colonne, list(schema), n=1)
        return proches[0] if proches else None

    def verifier_regle(arbre, schema):
        problemes = []

        def visiter(noeud):
            sorte = noeud[0]
            if sorte in ("et", "ou", "non"):
                enfants = noeud[1:]
                for e in (reversed(enfants) if defaut == "ordre_inverse" else enfants):
                    visiter(e)
                return
            if sorte not in OPERATEURS and sorte != "dans":
                raise ValueError(sorte)
            colonne = noeud[1]
            if colonne not in schema:
                if defaut == "leve":
                    raise KeyError(colonne)
                s = suggestion(colonne, schema)
                problemes.append(f"colonne « {colonne} » inconnue" + (f" : vouliez-vous dire « {s} » ?" if s else ""))
                return
            valeurs = noeud[2] if sorte == "dans" else (noeud[2],)
            fautives = [v for v in valeurs if genre(v) != schema[colonne]]
            if fautives:
                problemes.append(f"« {colonne} » est un {schema[colonne]}, comparée à un {genre(fautives[0])}")
                if defaut != "deux_messages":
                    return
            if schema[colonne] == "booleen" and sorte not in ("=", "!=", "dans"):
                problemes.append(f"« {colonne} » est un booléen : « {sorte} » n'a pas de sens")

        visiter(arbre)
        return problemes[:1] if defaut == "premier_seulement" else problemes

    return verifier_regle


@pytest.mark.parametrize("mutant,mots", [
    ("premier_seulement", ["vous vous arrêtez au premier"]),
    ("bool_nombre", ["True n'est pas un nombre ici"]),
    ("sans_suggestion", ["il manque 'vouliez-vous dire'", "difflib.get_close_matches"]),
    ("suggestion_toujours", ["ne devrait pas contenir 'vouliez-vous dire'", "suggestion absurde"]),
    ("sans_casse", ["'FIBRE'", "majuscules"]),
    ("ordre_inverse", ["pas dans l'ordre de lecture"]),
    ("leve", ["lève KeyError", "SIGNALE"]),
    ("deux_messages", ["en trop", "Un seul message par condition"]),
])
def test_verifier_regle_erreurs_classiques(mutant, mots):
    _echec("verifier_regle", _verificateur(mutant), *mots)


def test_verifier_regle_sans_defaut_la_fabrique_est_juste():
    _reussi("verifier_regle", _verificateur())


def test_verifier_regle_qui_modifie_le_schema_est_refuse():
    def verifier_regle(arbre, schema):
        schema.setdefault("id", "nombre")
        return corrige.verifier_regle(arbre, schema)

    _echec("verifier_regle", verifier_regle, "MODIFIÉ le schéma")


# --------------------------------------------------------------------------- Défi 157 : vers_sql


def _compilateur(defaut=None):
    def valeur_sql(v):
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        return str(int(v)) if isinstance(v, bool) else repr(v)

    def vers_sql(arbre):
        params = []

        def colonne(nom):
            return nom if defaut == "colonnes_nues" else '"' + nom.replace('"', '""') + '"'

        def place(v):
            if defaut == "fstring" or (defaut == "nombres_dedans" and not isinstance(v, str)):
                return valeur_sql(v)
            if defaut == "params_en_vrac":
                params.insert(0, v)  # les valeurs à l'envers des « ? »
            else:
                params.append(v)
            return "?"

        def sql(noeud):
            sorte = noeud[0]
            if sorte in ("et", "ou"):
                if len(noeud) < 3:
                    raise ValueError(noeud)
                texte = (" AND " if sorte == "et" else " OR ").join(sql(e) for e in noeud[1:])
                return texte if defaut == "sans_parentheses" else f"({texte})"
            if sorte == "non":
                return f"NOT ({sql(noeud[1])})"
            if sorte == "dans":
                if not noeud[2] and defaut != "in_vide":
                    raise ValueError(noeud)
                trous = [place(v) for v in noeud[2]]
                return f"{colonne(noeud[1])} IN ({', '.join(trous)})"
            if sorte in OPERATEURS:
                texte = f"{colonne(noeud[1])} {'<>' if sorte == '!=' else sorte} {place(noeud[2])}"
                if defaut == "coalesce":
                    return f"COALESCE({texte}, 0)"
                return texte
            raise ValueError(sorte)

        return sql(arbre), params

    return vers_sql


@pytest.mark.parametrize("mutant,mots", [
    ("fstring", ["DANS le texte SQL", "injection SQL"]),
    ("nombres_dedans", ["123456.75 est DANS le texte"]),
    ("colonnes_nues", ["que SQLite refuse", "guillemets DOUBLES"]),
    ("sans_parentheses", ["Il manque des parenthèses", "AND passe avant OR"]),
    ("coalesce", ["Inutile de traiter NULL"]),
    ("params_en_vrac", ["pas dans l'ordre des « ? »"]),
    ("in_vide", ["devrait lever ValueError", "IN ()"]),
])
def test_vers_sql_erreurs_classiques(mutant, mots):
    _echec("vers_sql", _compilateur(mutant), *mots)


def test_vers_sql_sans_defaut_la_fabrique_est_juste():
    _reussi("vers_sql", _compilateur())


# --------------------------------------------------------------------------- d'autres écritures justes


def test_autres_ecritures_justes_acceptees():
    motif = re.compile(r"""
        (?P<espace>\s+)
      | (?P<nombre>-?[0-9]+(?:\.[0-9]+)?)
      | (?P<op><=|>=|!=|==|<>|<|>|=)
      | (?P<ponct>[(),])
      | (?P<texte>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
      | (?P<mot>[^\W\d]\w*)
    """, re.VERBOSE)

    def decouper(texte):                          # une expression régulière, avec `match(texte, i)`
        if not isinstance(texte, str):
            raise TypeError("str attendu")
        jetons, i = [], 0
        while i < len(texte):
            m = motif.match(texte, i)
            if m is None:
                c = texte[i]
                if c in "\"'":
                    raise ErreurSyntaxe("guillemet de fin manquant", i)
                raise ErreurSyntaxe(f"guillemet typographique {c!r}" if c in "“”«»‘’„" else f"caractère {c!r}", i)
            genre, morceau = m.lastgroup, m.group()
            if genre == "nombre":
                if m.end() < len(texte) and texte[m.end()] == "." :
                    raise ErreurSyntaxe("point en trop", m.end())
                jetons.append(Jeton("NOMBRE", float(morceau) if "." in morceau else int(morceau), i))
            elif genre == "op":
                jetons.append(Jeton("OP", {"==": "=", "<>": "!="}.get(morceau, morceau), i))
            elif genre == "ponct":
                jetons.append(Jeton(morceau, morceau, i))
            elif genre == "texte":
                jetons.append(Jeton("TEXTE", re.sub(r"\\(.)", r"\1", morceau[1:-1], flags=re.S), i))
            elif genre == "mot":
                bas = morceau.lower()
                if bas in MOTS_CLES:
                    jetons.append(Jeton(MOTS_CLES[bas], bas, i))
                elif bas in BOOLEENS:
                    jetons.append(Jeton("BOOLEEN", BOOLEENS[bas], i))
                else:
                    jetons.append(Jeton("NOM", morceau, i))
            i = m.end()
        return jetons + [Jeton("FIN", None, len(texte))]

    _reussi("decouper", decouper)

    class Analyseur:                              # une classe au lieu de fonctions internes
        def __init__(self, texte):
            self.jetons, self.pos, self.profondeur = corrige.decouper(texte), 0, 0

        def voir(self):
            return self.jetons[self.pos]

        def prendre(self, *types):
            if self.voir().type in types:
                self.pos += 1
                return self.jetons[self.pos - 1]
            return None

        def entrer(self, jeton):
            self.profondeur += 1
            if self.profondeur > 100:
                raise ErreurSyntaxe("trop imbriqué", jeton.position)

        def suite(self, sorte, suivant):
            enfants = [suivant()]
            while self.prendre(sorte.upper()):
                enfants.append(suivant())
            return enfants[0] if len(enfants) == 1 else (sorte, *enfants)

        def regle(self):
            arbre = self.ou()
            if self.voir().type != "FIN":
                raise ErreurSyntaxe("en trop", self.voir().position)
            return arbre

        def ou(self):
            return self.suite("ou", self.et)

        def et(self):
            return self.suite("et", self.non)

        def non(self):
            jeton = self.prendre("NON")
            if jeton is None:
                return self.primaire()
            self.entrer(jeton)
            arbre = ("non", self.non())
            self.profondeur -= 1
            return arbre

        def valeur(self):
            jeton = self.prendre("NOMBRE", "TEXTE", "BOOLEEN")
            if jeton is None:
                j = self.voir()
                raise ErreurSyntaxe("guillemets oubliés ?" if j.type == "NOM" else "valeur attendue", j.position)
            return jeton.valeur

        def fermer(self, ouvrante):
            if self.voir().type == "FIN":
                raise ErreurSyntaxe("parenthèse jamais fermée", ouvrante.position)
            if not self.prendre(")"):
                raise ErreurSyntaxe("« ) » attendue", self.voir().position)

        def primaire(self):
            ouvrante = self.prendre("(")
            if ouvrante:
                self.entrer(ouvrante)
                arbre = self.ou()
                self.fermer(ouvrante)
                self.profondeur -= 1
                return arbre
            nom = self.prendre("NOM")
            if nom is None:
                j = self.voir()
                message = {"FIN": "règle vide" if self.pos == 0 else "incomplète"}.get(j.type, "il faut une colonne")
                raise ErreurSyntaxe(message, j.position)
            op = self.prendre("OP")
            if op:
                return (op.valeur, nom.valeur, self.valeur())
            if not self.prendre("DANS"):
                raise ErreurSyntaxe("opérateur attendu", self.voir().position)
            ouvrante = self.prendre("(")
            if ouvrante is None:
                raise ErreurSyntaxe("« ( » attendue", self.voir().position)
            valeurs = [self.valeur()]
            while self.prendre(","):
                valeurs.append(self.valeur())
            self.fermer(ouvrante)
            return ("dans", nom.valeur, tuple(valeurs))

    _reussi("analyser", lambda texte: Analyseur(texte).regle())

    def evaluer(arbre, client):                   # une table de vérité en nombres : 0 faux, 1 inconnu, 2 vrai
        def trois(noeud):
            sorte = noeud[0]
            if sorte == "et":
                pire = 2
                for e in noeud[1:]:
                    pire = min(pire, trois(e))
                    if pire == 0:
                        return 0
                return pire
            if sorte == "ou":
                mieux = 0
                for e in noeud[1:]:
                    mieux = max(mieux, trois(e))
                    if mieux == 2:
                        return 2
                return mieux
            if sorte == "non":
                return 2 - trois(noeud[1])
            r = corrige.evaluer(noeud, client)
            return 1 if r is None else 2 * r

        return {0: False, 1: None, 2: True}[trois(arbre)]

    _reussi("evaluer", evaluer)

    def vers_texte(arbre):                        # la priorité du parent descend dans l'appel
        def ecrire(noeud, parent):
            sorte = noeud[0]
            if sorte in ("et", "ou"):
                if len(noeud) < 3:
                    raise ValueError(noeud)
                moi = 1 if sorte == "ou" else 2
                texte = f" {sorte} ".join(ecrire(e, moi) for e in noeud[1:])
                return f"({texte})" if moi <= parent else texte
            if sorte == "non":
                return "non " + ecrire(noeud[1], 3)
            return corrige.vers_texte(noeud)

        return ecrire(arbre, 0)

    _reussi("vers_texte", vers_texte)

    def verifier_regle(arbre, schema):            # récursif, avec un générateur
        def problemes(noeud):
            if noeud[0] in ("et", "ou", "non"):
                for e in noeud[1:]:
                    yield from problemes(e)
                return
            if noeud[0] != "dans" and noeud[0] not in OPERATEURS:
                raise ValueError(noeud[0])
            colonne = noeud[1]
            if colonne not in schema:
                proches = [c for c in schema if c.casefold() == colonne.casefold()] or \
                    difflib.get_close_matches(colonne, list(schema), n=1)
                yield f"Colonne inconnue : {colonne}" + (f". Vouliez-vous dire {proches[0]} ?" if proches else "")
                return
            valeurs = noeud[2] if noeud[0] == "dans" else (noeud[2],)
            for v in valeurs:
                g = "booléen" if isinstance(v, bool) else "nombre" if isinstance(v, (int, float)) else "texte"
                if g != {"booleen": "booléen"}.get(schema[colonne], schema[colonne]):
                    yield f"{colonne} ({schema[colonne]}) comparée à {v!r} ({g})"
                    return
            if schema[colonne] == "booleen" and noeud[0] in ("<", "<=", ">", ">="):
                yield f"{noeud[0]} sur {colonne}, un BOOLÉEN"

        return list(problemes(arbre))

    _reussi("verifier_regle", verifier_regle)

    def vers_sql(arbre):                          # des parenthèses seulement quand il le faut, et `!=`
        params = []

        def sql(noeud, parent):
            sorte = noeud[0]
            if sorte in ("et", "ou"):
                if len(noeud) < 3:
                    raise ValueError(noeud)
                moi = 1 if sorte == "ou" else 2
                texte = (" AND " if moi == 2 else " OR ").join(sql(e, moi) for e in noeud[1:])
                return f"({texte})" if moi < parent else texte
            if sorte == "non":
                return "NOT " + sql(noeud[1], 3)
            if sorte not in OPERATEURS and sorte != "dans":
                raise ValueError(sorte)
            colonne = '"' + noeud[1].replace('"', '""') + '"'
            if sorte == "dans":
                if not noeud[2]:
                    raise ValueError(noeud)
                params.extend(noeud[2])
                return f"{colonne} IN ({', '.join('?' for _ in noeud[2])})"
            params.append(noeud[2])
            return f"{colonne} {sorte} ?"

        return sql(arbre, 0), params

    _reussi("vers_sql", vers_sql)


# --------------------------------------------------------------------------- outils communs, corrigé, docstrings


def test_les_tests_du_corrige_passent():
    tests = [f for nom, f in list(vars(corrige).items()) if nom.startswith("test_")]
    assert len(tests) >= 6
    for test in tests:
        test()


def test_le_fichier_eleve_s_importe_et_tout_est_a_faire():
    assert [r.statut for r in evaluer(SERIE)] == [A_FAIRE] * 6


def test_le_corrige_reussit_tout():
    assert [r.statut for r in evaluer(SERIE, corrige=True)] == [REUSSI] * 6


def test_le_fichier_eleve_reexporte_les_outils_communs():
    import defis.langage as langage
    import defis.serie_25 as eleve

    for nom in ("Jeton", "ErreurSyntaxe", "souligner", "MOTS_CLES", "BOOLEENS", "OPERATEURS", "clients_exemple",
                "schema_clients", "regles_exemple", "generer_clients", "generer_arbre", "base_clients",
                "afficher_arbre"):
        assert getattr(eleve, nom) is getattr(langage, nom), nom


def test_les_exemples_sont_neufs_a_chaque_appel():
    clients_exemple()[0]["contrat"] = "x"
    assert clients_exemple()[0]["contrat"] == "mensuel"
    schema_clients().clear()
    assert len(schema_clients()) == 7
    regles_exemple().clear()
    assert len(regles_exemple()) == 5


def test_les_exemples_ont_les_pieges_annonces():
    clients = clients_exemple()
    assert clients[3]["region"] is None and math.isnan(clients[5]["facture"]) and clients[6]["appels_support"] == 0
    assert all(set(c) == {"id", *schema_clients()} for c in clients)
    arbre = corrige.analyser(regles_exemple()["piege_priorite"])
    assert arbre[0] == "ou" and arbre[2][0] == "et"  # « et » passe avant « ou »
    # chaque règle du marketing est juste pour le vérificateur, et fait l'aller-retour
    for texte in regles_exemple().values():
        arbre = corrige.analyser(texte)
        assert corrige.verifier_regle(arbre, schema_clients()) == []
        assert corrige.analyser(corrige.vers_texte(arbre)) == arbre
    # la règle « regions_a_relancer » a un client INCONNU (le 4, sans région)
    arbre = corrige.analyser(regles_exemple()["regions_a_relancer"])
    assert [c["id"] for c in clients if corrige.evaluer(arbre, c) is None] == [4]


def test_les_generateurs_sont_reproductibles_et_valides():
    assert generer_clients(30, graine=2) == generer_clients(30, graine=2) or True  # NaN != NaN : on compare autrement
    a, b = generer_clients(300, graine=2), generer_clients(300, graine=2)
    assert [repr(c) for c in a] == [repr(c) for c in b]
    assert any(c["region"] is None for c in a)
    assert any(isinstance(c["facture"], float) and math.isnan(c["facture"]) for c in a)
    genres = {"nombre": numbers.Real, "texte": str, "booleen": bool}
    for graine in range(200):
        arbre = generer_arbre(graine, 4)
        assert arbre == generer_arbre(graine, 4)
        assert corrige.verifier_regle(arbre, schema_clients()) == []
    for c in a:
        for colonne, genre in schema_clients().items():
            v = c[colonne]
            assert v is None or (isinstance(v, float) and math.isnan(v)) or isinstance(v, genres[genre])


def test_base_clients_et_souligner():
    conn = base_clients(clients_exemple())
    assert conn.execute("SELECT COUNT(*) FROM clients WHERE facture IS NULL").fetchone() == (1,)  # le NaN
    assert conn.execute("SELECT COUNT(*) FROM clients WHERE region IS NULL").fetchone() == (1,)
    erreur = ErreurSyntaxe("ici", 3)
    assert isinstance(erreur, ValueError) and erreur.position == 3 and str(erreur) == "ici"
    assert souligner("a\tb = 1", erreur) == "a b = 1\n   ^ ici"


def test_afficher_arbre_sans_recursion():
    arbre = ("non", ("=", "a", 1))
    for _ in range(3000):
        arbre = ("non", arbre)
    assert afficher_arbre(arbre).count("\n") == 3001
    assert afficher_arbre(("=", "a", 'x"y')) == 'a = "x\\"y"'


def test_les_exemples_des_docstrings_sont_justes():
    """Les exemples de defis/serie_25.py et defis/langage.py, rejoués avec le corrigé."""
    import defis.langage as langage
    import defis.serie_25 as eleve

    tests = doctest.DocTestFinder().find(eleve) + doctest.DocTestFinder().find(langage)
    globs = dict(vars(eleve))
    for nom in DEFIS:
        globs[nom] = getattr(corrige, nom)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in tests:
        test.globs = dict(globs)
        runner.run(test)
    resultat = runner.summarize(verbose=False)
    assert resultat.attempted >= 30 and resultat.failed == 0


def test_le_corrige_n_utilise_ni_eval_ni_exec():
    texte = Path(corrige.__file__).read_text(encoding="utf-8")
    assert "eval(" not in texte.replace("evaluer(", "").replace("_ref_evaluer(", "")
    assert "exec(" not in texte.replace("execute(", "") and "import ast" not in texte


def test_les_messages_montrent_l_endroit_de_l_erreur():
    def analyser(texte):
        try:
            return corrige.analyser(texte)
        except ErreurSyntaxe as exc:
            if texte == "(a = 1":
                raise ErreurSyntaxe(exc.message, len(texte)) from None
            raise

    resultat = _evaluer("analyser", analyser)
    assert resultat.statut == ECHEC
    assert "(a = 1\n" in resultat.message and "      ^" in resultat.message  # le ^ sous la fin, puis sous la « ( »

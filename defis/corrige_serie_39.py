"""Série 39 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_39.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours une
comparaison faite sur du TEXTE au lieu de nombres ("2.9" > "2.10"), une règle de la spécification lue trop vite
(une pré-version passe AVANT sa version, `^0.2.3` s'arrête à 0.3.0), ou un hasard qui n'en est pas un (`hash()`
change à chaque lancement de Python).

Fil rouge de la série : livrer, c'est prendre des décisions automatiques (quelle version est la dernière, quel
numéro donner, quelles versions installer ensemble, qui voit la nouveauté). Chacune obéit à une règle écrite ;
le code doit suivre la règle à la lettre, parce que personne ne relira chaque décision.

1. une version SemVer se lit en NOMBRES (2.10 > 2.9), une pré-version passe avant sa version (rc.1 < 1.0.0), et
   ses identifiants se comparent un par un : les numériques en nombres, avant les alphanumériques ;
2. le numéro suivant se DÉDUIT des commits : la plus forte raison de changer l'emporte (une rupture, même cachée
   dans le pied d'un message, l'emporte sur tout), une seule fois, et les numéros plus petits repartent de zéro ;
3. une contrainte est une suite de comparaisons : `^1.2.3` veut dire « compatible avec 1.2.3 » (jusqu'à 2.0.0
   exclu, mais jusqu'à 0.3.0 pour `^0.2.3`), et une pré-version n'est acceptée que si on l'a demandée ;
4. résoudre des dépendances, c'est chercher : essayer la version la plus récente, et revenir en arrière quand
   un choix rend la suite impossible ;
5. un déploiement progressif range chaque conseiller dans un seau STABLE (le même d'un lancement à l'autre,
   différent d'une expérience à l'autre), calculé avec un vrai hachage, pas avec `hash()` ;
6. un drapeau s'évalue dans un ordre fixé (coupure, prérequis, cibles, règles, défaut), et une information
   absente ne satisfait aucune condition, pas même « n'est pas dans ».

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import hashlib
import re
from functools import cmp_to_key

from defis.livraison import (  # noqa: F401
    OPERATEURS,
    PALIERS,
    Conflit,
    Decision,
    conseillers_exemple,
    drapeaux_exemple,
    generer_conseillers,
    generer_versions,
    historique_exemple,
    index_exemple,
    tags_exemple,
)

# ============================================================================ défi 236 : comparer_versions

# La grammaire officielle de SemVer 2.0.0 (semver.org), écrite avec [0-9] et pas \d : en Python, \d accepte aussi
# les chiffres arabes, persans, devanagari… ("١.٢.٣" passerait, et int("١") vaut 1).
_IDENT = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    rf"(?:-({_IDENT}(?:\.{_IDENT})*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
)


def _lire(version) -> tuple[int, int, int, tuple]:
    """(majeur, mineur, correctif, identifiants de pré-version) ; les identifiants numériques en int."""
    if not isinstance(version, str):
        raise TypeError(f"une version est un texte, pas {type(version).__name__}")
    trouve = _SEMVER.fullmatch(version)  # fullmatch, pas match + "$" : "$" accepte un "\n" final
    if trouve is None:
        raise ValueError(f"version SemVer invalide : {version!r}")
    pre = tuple(int(x) if x.isdigit() else x for x in trouve[4].split(".")) if trouve[4] else ()
    return int(trouve[1]), int(trouve[2]), int(trouve[3]), pre


def _cle(version: str) -> tuple:
    """Une clé de tri : deux versions se comparent comme leurs clés. Les métadonnées de build n'y sont pas."""
    majeur, mineur, correctif, pre = _lire(version)
    # Sans pré-version, le 4e élément vaut True : une version passe APRÈS toutes ses pré-versions (False < True).
    # Chaque identifiant devient (0, nombre, "") ou (1, 0, texte) : les numériques d'abord, comparés en nombres,
    # puis les alphanumériques, comparés en texte ASCII. Un tuple plus court qu'un autre qui le prolonge passe
    # avant lui : ("alpha",) < ("alpha", 1), comme le veut la règle 11.4.4 de SemVer.
    ids = tuple((0, x, "") if isinstance(x, int) else (1, 0, x) for x in pre)
    return majeur, mineur, correctif, not pre, ids


def comparer_versions(a: str, b: str) -> int:
    """Défi 236 — -1 si a passe avant b, 0 si elles sont de même rang, 1 si a passe après b."""
    cle_a, cle_b = _cle(a), _cle(b)
    return (cle_a > cle_b) - (cle_a < cle_b)


# ============================================================================ défi 237 : prochaine_version

# type, (portée) facultative et non vide, ! facultatif, puis ": " et une description. Le type ne contient que des
# lettres : "fixup! fix: …" n'est pas un commit conventionnel (après "fixup!", il faudrait ": ").
_ENTETE = re.compile(r"([A-Za-z]+)(?:\(([^()\r\n]+)\))?(!)?: (\S.*)")
_PIEDS = ("BREAKING CHANGE: ", "BREAKING-CHANGE: ")
_NIVEAUX = {"feat": 2, "fix": 1, "perf": 1}   # 3 = rupture ; les autres types (docs, chore, ci…) : 0


def _niveau(message: str) -> int:
    if not isinstance(message, str):
        raise TypeError(f"un message de commit est un texte, pas {type(message).__name__}")
    lignes = [ligne.rstrip("\r") for ligne in message.split("\n")]
    entete = _ENTETE.fullmatch(lignes[0])
    if entete is None:
        return 0   # pas un commit conventionnel (« WIP », « Merge branch… ») : il ne compte pas
    if entete[3] or any(ligne.startswith(_PIEDS) for ligne in lignes[1:]):   # le pied est dans le CORPS
        return 3
    return _NIVEAUX.get(entete[1].lower(), 0)    # le type ne dépend pas de la casse ; BREAKING CHANGE, si


def prochaine_version(actuelle: str, messages: list[str]) -> str | None:
    """Défi 237 — Le numéro de la prochaine version, déduit des commits ; None s'il n'y a rien à publier."""
    majeur, mineur, correctif, pre = _lire(actuelle)
    niveau = max((_niveau(m) for m in messages), default=0)   # la plus forte raison l'emporte, UNE fois
    if niveau == 0:
        return None
    if niveau == 3 and majeur == 0:
        niveau = 2       # en 0.y.z, tout peut changer : une rupture ne fait pas passer en 1.0.0 toute seule
    if niveau == 3:      # 3.0.0-rc.2 + rupture → 3.0.0 (la rupture est déjà annoncée par le 3)
        return f"{majeur}.0.0" if pre and mineur == 0 and correctif == 0 else f"{majeur + 1}.0.0"
    if niveau == 2:
        return f"{majeur}.{mineur}.0" if pre and correctif == 0 else f"{majeur}.{mineur + 1}.0"
    return f"{majeur}.{mineur}.{correctif}" if pre else f"{majeur}.{mineur}.{correctif + 1}"


# ============================================================================ défi 238 : satisfait

_OPERATEURS_CONTRAINTE = (">=", "<=", ">", "<", "=", "^", "~")   # ">=" AVANT ">" : on cherche le plus long d'abord


def _comparateurs(alternative: str) -> list[tuple[str, str | None]]:
    """[(opérateur, version)] d'une alternative (des comparaisons séparées par des virgules, toutes exigées)."""
    resultat = []
    for morceau in alternative.split(","):
        morceau = morceau.strip()
        if morceau == "*":
            resultat.append(("*", None))
            continue
        operateur = next((o for o in _OPERATEURS_CONTRAINTE if morceau.startswith(o)), "=")
        version = morceau[len(operateur):].strip() if morceau.startswith(operateur) else morceau
        _lire(version)   # ValueError si elle est vide ou invalide
        resultat.append((operateur, version))
    return resultat


def _borne(operateur: str, version: str) -> str:
    """La première version EXCLUE par ^ ou ~ ; le "-0" final écarte aussi toutes ses pré-versions."""
    majeur, mineur, correctif, _ = _lire(version)
    if operateur == "~":
        return f"{majeur}.{mineur + 1}.0-0"
    if majeur > 0:
        return f"{majeur + 1}.0.0-0"
    if mineur > 0:
        return f"0.{mineur + 1}.0-0"
    return f"0.0.{correctif + 1}-0"


def _respecte(version: str, operateur: str, cible: str | None) -> bool:
    if operateur == "*":
        return True
    c = comparer_versions(version, cible)
    if operateur in ("^", "~"):
        return c >= 0 and comparer_versions(version, _borne(operateur, cible)) < 0
    return {"=": c == 0, ">=": c >= 0, "<=": c <= 0, ">": c > 0, "<": c < 0}[operateur]


def satisfait(version: str, contrainte: str) -> bool:
    """Défi 238 — True si `version` respecte `contrainte` (alternatives séparées par "||", comparaisons par ",")."""
    majeur, mineur, correctif, pre = _lire(version)
    if not isinstance(contrainte, str):
        raise TypeError(f"une contrainte est un texte, pas {type(contrainte).__name__}")
    alternatives = [_comparateurs(a) for a in contrainte.split("||")]   # tout est lu AVANT de répondre
    for comparateurs in alternatives:
        if not all(_respecte(version, op, v) for op, v in comparateurs):
            continue
        # Une pré-version n'est acceptée que si l'alternative en NOMME une du même majeur.mineur.correctif :
        # « >=2.0.0 » ne veut pas dire « installez-moi la 3.0.0-rc.1 ».
        if not pre or any(v is not None and _lire(v)[3] and _lire(v)[:3] == (majeur, mineur, correctif)
                          for _, v in comparateurs):
            return True
    return False


# ============================================================================ défi 239 : resoudre


def resoudre(index: dict, demandes: dict) -> dict:
    """Défi 239 — {paquet: version} : la combinaison qu'un gestionnaire de paquets installerait."""

    def decider(choisis: dict, contraintes: dict, file: list):
        if not file:
            return choisis
        paquet, reste = file[0], file[1:]
        versions = sorted(index.get(paquet, {}), key=cmp_to_key(comparer_versions), reverse=True)
        for version in versions:                                   # la plus récente d'abord
            if not all(satisfait(version, c) for c in contraintes[paquet]):
                continue
            avec = {**choisis, paquet: version}
            dependances = index[paquet][version]
            if any(d in avec and not satisfait(avec[d], c) for d, c in dependances.items()):
                continue                                           # contredit un choix déjà fait : suivante
            nouvelles = {d: contraintes.get(d, []) + [c] for d, c in dependances.items()}
            ajout = [d for d in dependances if d not in avec and d not in reste]
            solution = decider(avec, {**contraintes, **nouvelles}, reste + ajout)   # de NOUVEAUX objets
            if solution is not None:
                return solution
        return None                                                # aucune ne va : on revient en arrière

    solution = decider({}, {p: [c] for p, c in demandes.items()}, list(demandes))
    if solution is None:
        raise Conflit(f"aucune combinaison de versions ne satisfait {demandes!r}")
    return solution


# ============================================================================ défi 240 : affecter


def affecter(experience: str, identifiant: str, poids: dict) -> str:
    """Défi 240 — La variante de `poids` que voit `identifiant` dans `experience`, toujours la même."""
    for nom, valeur in (("experience", experience), ("identifiant", identifiant)):
        if not isinstance(valeur, str):
            raise TypeError(f"{nom} doit être un texte, pas {type(valeur).__name__}")
    if not isinstance(poids, dict):
        raise TypeError("poids doit être un dict {variante: poids entier}")
    for p in poids.values():
        if type(p) is not int:
            raise TypeError(f"un poids est un entier, pas {type(p).__name__}")
        if p < 0:
            raise ValueError(f"poids négatif : {p}")
    total = sum(poids.values())
    if total == 0:
        raise ValueError("il faut au moins un poids strictement positif")
    # SHA-256 donne la même chose sur toutes les machines et à chaque lancement ; `hash()`, lui, est salé au
    # démarrage de Python (PYTHONHASHSEED) : un conseiller changerait de variante à chaque redémarrage.
    empreinte = hashlib.sha256(f"{experience}:{identifiant}".encode("utf-8")).digest()
    seau = int.from_bytes(empreinte[:8], "big") % PALIERS
    cumul = 0
    for variante, p in poids.items():
        cumul += p
        if seau * total < PALIERS * cumul:    # seau / PALIERS < cumul / total, en entiers
            return variante
    raise AssertionError("inatteignable : seau * total < PALIERS * total")


# ============================================================================ défi 241 : evaluer


def _poids_valides(poids) -> None:
    if not isinstance(poids, dict) or not poids:
        raise ValueError("une répartition est un dict non vide {valeur: poids}")
    if any(type(p) is not int for p in poids.values()):
        raise TypeError("un poids est un entier")
    if any(p < 0 for p in poids.values()) or sum(poids.values()) == 0:
        raise ValueError("des poids positifs, dont au moins un non nul")


def _servie_valide(servie) -> None:
    if isinstance(servie, dict):
        _poids_valides(servie)
    elif not isinstance(servie, str):
        raise TypeError(f"une valeur servie est un texte ou une répartition, pas {type(servie).__name__}")


def _valider(drapeau) -> None:
    """Toute la configuration du drapeau, AVANT de servir quoi que ce soit : une faute de frappe dans la règle 3
    doit se voir tout de suite, pas le jour où un conseiller l'atteint."""
    if not isinstance(drapeau, dict):
        raise TypeError("un drapeau est un dict")
    if type(drapeau.get("actif")) is not bool:     # "false" (un texte) est VRAI pour un `if`
        raise TypeError(f"actif doit valoir True ou False, pas {drapeau.get('actif')!r}")
    if not isinstance(drapeau.get("coupe"), str):
        raise TypeError("coupe doit être un texte")
    if "defaut" not in drapeau:
        raise ValueError("il manque la valeur par défaut")
    _servie_valide(drapeau["defaut"])
    for regle in drapeau.get("regles", []):
        _servie_valide(regle["alors"])
        for attribut, operateur, operande in regle["si"]:
            if operateur not in OPERATEURS:
                raise ValueError(f"opérateur inconnu : {operateur!r}")
            if operateur in ("dans", "pas_dans") and not isinstance(operande, (list, tuple, set, frozenset)):
                raise TypeError(f"{operateur} attend une liste, pas {operande!r} (\"Ly\" in \"Lyon\" est vrai)")
            if operateur.startswith("version"):
                _lire(operande)


def _condition(utilisateur: dict, attribut: str, operateur: str, operande) -> bool:
    valeur = utilisateur.get(attribut)
    if valeur is None:
        return False               # absent : FAUX, même pour « pas_dans » (on ne sait pas, donc on ne sert pas)
    if operateur == "dans":
        return valeur in operande
    if operateur == "pas_dans":
        return valeur not in operande
    try:
        c = comparer_versions(valeur, operande)
    except (TypeError, ValueError):
        return False               # "4.10" n'est pas une version : la condition n'est pas remplie, rien ne plante
    return c >= 0 if operateur == "version>=" else c < 0


def _servir(servie, nom: str, identifiant: str) -> str:
    return affecter(nom, identifiant, servie) if isinstance(servie, dict) else servie


def evaluer(drapeaux: dict, nom: str, utilisateur: dict) -> Decision:
    """Défi 241 — Decision(valeur, raison) : ce que le drapeau `nom` sert à `utilisateur`, et pourquoi."""
    identifiant = utilisateur.get("id") if isinstance(utilisateur, dict) else None
    if not isinstance(identifiant, str):
        raise TypeError("utilisateur doit être un dict avec un \"id\" texte")
    return _evaluer(drapeaux, nom, utilisateur, identifiant, ())


def _evaluer(drapeaux: dict, nom: str, utilisateur: dict, identifiant: str, chemin: tuple) -> Decision:
    if nom in chemin:
        raise ValueError("cycle de prérequis : " + " → ".join(chemin + (nom,)))
    if nom not in drapeaux:
        raise ValueError(f"drapeau inconnu : {nom!r}")
    drapeau = drapeaux[nom]
    _valider(drapeau)
    if not drapeau["actif"]:                                       # 1. le bouton d'arrêt l'emporte sur tout
        return Decision(drapeau["coupe"], "coupé")
    for autre, attendue in drapeau.get("prerequis", {}).items():   # 2. les prérequis, pour le MÊME conseiller
        if _evaluer(drapeaux, autre, utilisateur, identifiant, chemin + (nom,)).valeur != attendue:
            return Decision(drapeau["coupe"], f"prérequis {autre}")
    for valeur, identifiants in drapeau.get("cibles", {}).items():  # 3. les conseillers désignés un par un
        if identifiant in identifiants:
            return Decision(valeur, "ciblé")
    for numero, regle in enumerate(drapeau.get("regles", []), start=1):   # 4. la PREMIÈRE règle remplie
        if all(_condition(utilisateur, *condition) for condition in regle["si"]):
            return Decision(_servir(regle["alors"], nom, identifiant), f"règle {numero}")
    return Decision(_servir(drapeau["defaut"], nom, identifiant), "défaut")   # 5. sinon, le défaut


# ============================================================================ tests (comme dans tests/test_*.py)


def test_comparer_versions_lit_des_nombres():
    assert comparer_versions("2.10.1", "2.9.4") == 1 and "2.10.1" < "2.9.4"
    chaine = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11",
              "1.0.0-rc.1", "1.0.0"]                        # l'exemple de semver.org, dans l'ordre
    assert sorted(reversed(chaine), key=cmp_to_key(comparer_versions)) == chaine
    assert comparer_versions("1.0.0+build.1", "1.0.0+build.2") == 0


def test_prochaine_version_lit_le_pied():
    assert prochaine_version("2.10.1", historique_exemple()) == "3.0.0"
    assert prochaine_version("1.4.2", ["feat: a", "feat: b", "fix: c"]) == "1.5.0"
    assert prochaine_version("0.9.3", ["feat!: tout change"]) == "0.10.0"
    assert prochaine_version("3.0.0-rc.2", ["fix: dernier correctif"]) == "3.0.0"
    assert prochaine_version("1.0.0", ["docs: rien", "WIP"]) is None


def test_satisfait_et_le_zero():
    assert satisfait("1.9.0", "^1.2.3") and not satisfait("2.0.0", "^1.2.3")
    assert satisfait("0.2.9", "^0.2.3") and not satisfait("0.3.0", "^0.2.3")
    assert not satisfait("2.2.0-rc.1", ">=2.0.0") and satisfait("2.2.0-rc.1", ">=2.2.0-rc.0")
    assert satisfait("3.0.0", "^1.0.0 || ^3.0.0")


def test_resoudre_revient_en_arriere():
    solution = resoudre(index_exemple(), {"churn-modele": "^3.0.0", "graphiques": "^1.2.0", "journal": "*"})
    assert solution == {"churn-modele": "3.2.0", "graphiques": "1.3.1", "journal": "0.4.0", "numerique": "2.1.0",
                        "tableaux": "4.0.0"}
    try:
        resoudre(index_exemple(), {"churn-modele": "^3.2.0", "numerique": "^1.9.0"})
    except Conflit:
        pass
    else:
        raise AssertionError("Conflit attendu")


def test_affecter_est_stable_et_proportionnel():
    assert affecter("nouveau_score", "U0001", {"on": 25, "off": 75}) == affecter("nouveau_score", "U0001",
                                                                                 {"on": 25, "off": 75})
    partout = [affecter("e", f"U{i}", {"a": 1, "b": 3}) for i in range(4000)]
    assert 900 < partout.count("a") < 1100
    assert all(affecter("e", f"U{i}", {"a": 0, "b": 1}) == "b" for i in range(200))


def test_evaluer_suit_l_ordre():
    drapeaux = drapeaux_exemple()
    assert evaluer(drapeaux, "export_csv", {"id": "U0001"}) == Decision("off", "coupé")
    assert evaluer(drapeaux, "nouveau_score", {"id": "U0024", "appli": "4.10.2"}) == Decision("on", "ciblé")
    assert evaluer(drapeaux, "nouveau_score", {"id": "U0024", "appli": None}).raison == "prérequis fiche_v2"
    assert evaluer(drapeaux, "nouveau_score", {"id": "U1", "role": "superviseur", "appli": "4.10.2"}).raison == "règle 1"

"""Série 39 — Livrer sans tout casser : comparer des versions, numéroter, contraindre, résoudre, déployer peu à peu.

🚀 VOUS VOULEZ DÉVELOPPER ? ÉCRIRE LE CODE N'EST QUE LA MOITIÉ DU TRAVAIL : IL FAUT AUSSI LE LIVRER. ICI, VOUS
ÉCRIVEZ LES OUTILS QUI DÉCIDENT QUELLE VERSION PART EN PRODUCTION, ET QUI LA VOIT EN PREMIER.

Le 3 novembre, le script de déploiement du service de score a choisi la « dernière » version avec `max(tags)` : il
a pris 2.9.4 au lieu de 2.10.1, parce qu'en TEXTE "2.9.4" > "2.10.1". Pendant trois semaines, la production a
tourné avec l'ancien modèle. Le 24, Théo a corrigé à la main, publié le nouveau score sous le numéro 2.11.0 (il
retirait pourtant le champ « score » des réponses) et l'a activé pour TOUS les conseillers d'un coup : l'appli des
conseillers en version 4.1 ne savait pas l'afficher, et le centre d'appels est resté aveugle une matinée. La
directrice technique a tranché : plus de versions comparées « à l'œil », plus de numéro choisi à la main, plus de
mise en production « tout d'un coup ». Cette série vous fait écrire les six outils qu'il manquait :

    236 comparer_versions    laquelle de deux versions est la plus récente (SemVer : 2.10 > 2.9, 1.0.0-rc.1 < 1.0.0)
    237 prochaine_version    le numéro de la prochaine version, déduit des messages de commit (Conventional Commits)
    238 satisfait            une version respecte-t-elle une contrainte comme "^1.2.3" ou ">=2.0.0, <3.0.0" ?
    239 resoudre             quelles versions installer ensemble pour que toutes les contraintes tiennent ?
    240 affecter             qui voit la nouveauté : un seau STABLE par conseiller, une part réglable
    241 evaluer              un drapeau de fonctionnalité : coupure, prérequis, cibles, règles, défaut

Prérequis : la série 00 (fonctions, listes, dictionnaires, boucles). Le défi 51 (expressions régulières), la série
09 (récursion, retour arrière), la série 20 (Unicode) et la série 24 (graphes de dépendances) l'éclairent, mais tout
ce qu'il faut est rappelé ici.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- une version SemVer (semver.org) s'écrit MAJEUR.MINEUR.CORRECTIF, trois entiers sans zéro inutile en tête : on
  augmente le CORRECTIF pour une correction, le MINEUR pour un ajout qui ne casse rien, le MAJEUR pour un
  changement qui casse ceux qui s'en servent. Elle peut finir par une PRÉ-VERSION (« -rc.1 », « -beta.11 » :
  des identifiants séparés par des points) et par des MÉTADONNÉES de build (« +build.1103 », qui ne comptent
  jamais dans l'ordre) ;
- « Conventional Commits » (conventionalcommits.org) : un message de commit commence par « type(portée): sujet ».
  `feat` ajoute quelque chose, `fix` corrige, un « ! » avant les deux-points ou une ligne « BREAKING CHANGE: … »
  dans le corps annonce une rupture. Des outils comme semantic-release en déduisent le prochain numéro ;
- une CONTRAINTE dit quelles versions d'une dépendance conviennent : `^1.2.3` (« compatible avec 1.2.3 ») veut
  dire >= 1.2.3 et < 2.0.0 ; `~1.2.3` veut dire >= 1.2.3 et < 1.3.0 (npm, Cargo et Poetry s'en servent) ;
- un DRAPEAU de fonctionnalité (« feature flag ») allume une nouveauté pour une partie des utilisateurs, sans
  redéployer : on commence à 5 %, on regarde les erreurs, on monte à 25 %, puis 100 %, et on coupe en une
  seconde si ça va mal ;
- `functools.cmp_to_key(comparer)` transforme une fonction qui renvoie -1, 0 ou 1 en clé de tri pour `sorted`,
  `max` et `min` ; `hashlib.sha256(texte.encode("utf-8")).digest()` donne 32 octets, les mêmes sur toutes les
  machines ; `int.from_bytes(octets, "big")` en fait un entier.

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_39 import *
    >>> tags = tags_exemple()
    >>> tags[:4], max(tags)                   # max sur du texte : l'erreur du 3 novembre
    (['1.0.0', '1.1.0', '1.2.0', '1.2.1'], '2.9.4')
    >>> conseillers_exemple()[23]
    {'id': 'U0024', 'site': 'Lille', 'equipe': 'rétention', 'role': 'conseiller', 'appli': '4.10.2'}

Quand les six défis sont réussis, Théo peut livrer sans retenir son souffle :

    >>> from functools import cmp_to_key
    >>> max(tags, key=cmp_to_key(comparer_versions))
    '2.11.0-rc.1'
    >>> max((t for t in tags if "-" not in t), key=cmp_to_key(comparer_versions))      # la dernière stable
    '2.10.1'
    >>> prochaine_version("2.10.1", historique_exemple())     # le pied « BREAKING CHANGE » impose une 3.0.0
    '3.0.0'
    >>> satisfait("3.0.0", "^2.9.0"), satisfait("2.11.0", "^2.9.0")    # l'appli demande « ^2.9.0 » au service
    (False, True)
    >>> resoudre(index_exemple(), {"churn-modele": "^3.0.0", "graphiques": "^1.2.0", "journal": "*"})
    {'churn-modele': '3.2.0', 'graphiques': '1.3.1', 'journal': '0.4.0', 'numerique': '2.1.0', 'tableaux': '4.0.0'}
    >>> from collections import Counter
    >>> decisions = [evaluer(drapeaux_exemple(), "nouveau_score", c) for c in conseillers_exemple()]
    >>> Counter(d.valeur for d in decisions)
    Counter({'off': 428, 'on': 52})
    >>> Counter(d.raison for d in decisions).most_common()
    [('défaut', 284), ('règle 2', 134), ('prérequis fiche_v2', 37), ('règle 1', 21), ('ciblé', 4)]

Numéroté 2.11.0, le nouveau score aurait été installé par l'appli, qui demande « ^2.9.0 » au service : en 3.0.0,
elle le refuse tant qu'on ne l'a pas adaptée. Et cette fois, le nouveau score n'est servi qu'à 52 conseillers sur
480 : les trois testeurs, 21 superviseurs, et 28 des 134 conseillers de Lyon et de Nantes de la règle 2 (la part
de 25 % ; sur si peu de monde, le hasard en donne un peu moins), tous avec une appli 4.2.0 ou plus récente. Les
37 dont l'appli n'a pas envoyé sa version (ou une version illisible) ne l'ont pas : on ne sert pas une nouveauté
à quelqu'un dont on ne sait rien. S'il le faut, un clic sur « actif » coupe tout.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. L'ordre proposé est le plus progressif : le 237,
   le 238 et le 241 lisent des versions (servez-vous de votre 236), le 239 se sert de votre 238 et le 241 de votre
   240. Si l'un d'eux échoue parce qu'un défi précédent n'est pas encore juste, le correcteur vous le dit.
2. Lancez `python scripts/defis.py --defi 236` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_39.py`.

Les outils communs (`tags_exemple`, `historique_exemple`, `index_exemple`, `drapeaux_exemple`,
`conseillers_exemple`, `generer_versions`, `generer_conseillers`, `PALIERS`, `OPERATEURS`, `Conflit` et `Decision`)
sont dans `defis/livraison.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import hashlib  # noqa: F401 — hashlib.sha256(texte.encode("utf-8")).digest()
import re  # noqa: F401 — re.fullmatch(motif, texte) : tout le texte, sans « \n » final toléré
from functools import cmp_to_key  # noqa: F401 — sorted(versions, key=cmp_to_key(comparer_versions))

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

# ============================================================================ Numéroter


def comparer_versions(a: str, b: str) -> int:
    """Défi 236 🟢 — -1 si la version a passe AVANT b, 0 si elles ont le même rang, 1 si a passe APRÈS b.

        >>> comparer_versions("2.10.1", "2.9.4"), "2.10.1" > "2.9.4"       # en texte, "1" < "9"
        (1, False)
        >>> comparer_versions("1.0.0-rc.1", "1.0.0")      # une pré-version passe AVANT sa version
        -1
        >>> comparer_versions("1.0.0-beta.11", "1.0.0-beta.2"), comparer_versions("1.0.0-alpha", "1.0.0-alpha.1")
        (1, -1)
        >>> comparer_versions("1.0.0-9", "1.0.0-alpha"), comparer_versions("1.0.0+build.1", "1.0.0+build.2")
        (-1, 0)

    Ici : 2.10.1 est plus récente que 2.9.4 (10 > 9), même si le texte dit le contraire. Une pré-version (« rc.1 »)
    se publie avant la version qu'elle prépare. Dans une pré-version, « 11 » et « 2 » sont des NOMBRES (beta.11
    vient après beta.2), et « alpha » sans rien de plus passe avant « alpha.1 ». Un identifiant numérique (« 9 »)
    passe avant un identifiant alphanumérique (« alpha »). Les métadonnées de build (« +build.1 ») ne comptent pas.

    Les règles (SemVer 2.0.0, sections 2, 9, 10 et 11) :

    - une version s'écrit MAJEUR.MINEUR.CORRECTIF : trois entiers en chiffres ASCII (0 à 9 : pas « ١ », pas
      « ² »), sans zéro inutile en tête (« 0 » oui, « 01 » non) ;
    - puis, facultatif, « - » et une pré-version : des identifiants séparés par des points, non vides, faits de
      [0-9A-Za-z-] ; un identifiant fait QUE de chiffres est numérique et n'a pas de zéro inutile en tête
      (« 01 » est refusé, « 01a » est un identifiant alphanumérique valable) ;
    - puis, facultatif, « + » et des métadonnées de build : des identifiants non vides faits de [0-9A-Za-z-],
      séparés par des points (« 001 » est permis ici) ;
    - l'ordre : majeur, puis mineur, puis correctif, comparés en NOMBRES ; à égalité, une version SANS pré-version
      passe après une version AVEC ; deux pré-versions se comparent identifiant par identifiant, de gauche à
      droite : deux numériques en nombres, deux alphanumériques en texte (ordre ASCII : "Z" < "a"), un numérique
      AVANT un alphanumérique ; si tous les identifiants communs sont égaux, la plus COURTE passe avant ;
      les métadonnées de build sont ignorées (« 1.0.0+a » et « 1.0.0+b » ont le même rang : 0) ;
    - erreurs : a ou b qui n'est pas un texte → TypeError ; une version qui ne suit pas cette grammaire (« 1.2 »,
      « v1.2.3 », « 01.2.3 », « 1.2.3-01 », « 1.2.3- », « 1.2.3-a..b », « 1.2.3\\n », « ١.٢.٣ ») → ValueError.

    Ce que vous apprenez : lire une spécification à la lettre, comparer des NOMBRES et pas du texte, et construire
    une clé de tri qui encode toutes les règles (les tuples Python se comparent élément par élément).

    Indice : `re.fullmatch` (pas `re.match` avec « $ », qui accepte un « \\n » final ; pas `\\d`, qui accepte
    tous les chiffres d'Unicode) ; une clé (majeur, mineur, correctif, pas_de_pre_version, identifiants), où chaque
    identifiant devient (0, nombre, "") ou (1, 0, texte) ; puis `(cle_a > cle_b) - (cle_a < cle_b)`.
    """
    raise NotImplementedError


def prochaine_version(actuelle: str, messages: list[str]) -> str | None:
    """Défi 237 🟢 — Le numéro de la prochaine version, déduit des messages de commit ; None s'il n'y a rien à publier.

        >>> prochaine_version("1.4.2", ["feat: export CSV", "feat(api): pagination", "fix: arrondi"])
        '1.5.0'
        >>> prochaine_version("1.4.2", ["fix: arrondi", "refactor: tout\\n\\nBREAKING CHANGE: le champ score disparaît"])
        '2.0.0'
        >>> prochaine_version("1.4.2", ["docs: README", "Merge branch 'main'", "fixup! fix: arrondi"]) is None
        True
        >>> prochaine_version("0.9.3", ["feat!: nouvelle API"]), prochaine_version("3.0.0-rc.2", ["fix: dernier bug"])
        ('0.10.0', '3.0.0')

    Ici : deux `feat` et un `fix` font UNE nouvelle version mineure (1.5.0, et pas 1.7.0 : on ne cumule pas), et le
    correctif repart de 0. Une ligne « BREAKING CHANGE: » dans le corps du message annonce une rupture : 2.0.0.
    « docs », un message de fusion et un « fixup! » ne publient rien. Avant 1.0.0, tout peut encore changer : une
    rupture ne fait monter que le mineur. Et après 3.0.0-rc.2, un correctif publie… 3.0.0, la version qu'on
    préparait.

    Les règles (Conventional Commits 1.0.0, et les règles de `semver.inc` de npm pour les pré-versions) :

    - un message est conventionnel si sa PREMIÈRE ligne s'écrit exactement « type(portée)!: sujet » : un type fait
      de lettres ASCII, une portée facultative entre parenthèses (non vide, sans parenthèse ni retour à la ligne),
      un « ! » facultatif, puis deux-points, UNE espace, et un sujet qui ne commence pas par une espace. Les autres
      messages (« WIP », « Merge branch… », « fixup! fix: … », « fix : … », « feat:sans espace ») ne comptent pas ;
    - le type ne dépend pas de la casse (« Feat » est un `feat`). `feat` vaut une version MINEURE, `fix` et `perf`
      une version CORRECTIVE, les autres types (docs, chore, ci, test, refactor, style, build…) rien ;
    - un message conventionnel est une RUPTURE (version MAJEURE), quel que soit son type, si sa première ligne a
      le « ! », ou si une ligne de son CORPS (après la première ; lignes séparées par "\\n", un "\\r" final retiré)
      commence par « BREAKING CHANGE: » ou « BREAKING-CHANGE: », en majuscules (« breaking change: » ne compte
      pas, la première ligne non plus) ;
    - le plus fort niveau de tous les messages l'emporte, une seule fois. Aucun → None (pas `actuelle` : il n'y a
      rien à publier) ;
    - en 0.y.z (majeur 0), une rupture ne compte que comme une version mineure ;
    - le calcul (les métadonnées de build de `actuelle` disparaissent toujours) : majeure → (M+1).0.0, mais si
      `actuelle` est une pré-version de M.0.0, c'est M.0.0 ; mineure → M.(m+1).0, mais si `actuelle` est une
      pré-version de M.m.0, c'est M.m.0 ; corrective → M.m.(c+1), mais si `actuelle` est une pré-version, c'est
      M.m.c (sa version finale) ;
    - erreurs : `actuelle` qui n'est pas une version → TypeError ou ValueError, comme au défi 236 ; un message qui
      n'est pas un texte → TypeError.

    Ce que vous apprenez : ne plus choisir un numéro à la main (Théo avait appelé 2.11.0 une version qui cassait
    l'appli), lire un format de message à la lettre, et prendre le MAXIMUM au lieu de cumuler.

    Indice : `message.split("\\n")` ; `re.fullmatch(r"([A-Za-z]+)(?:\\(([^()\\r\\n]+)\\))?(!)?: (\\S.*)", ligne)` ; un
    niveau par message (0, 1, 2 ou 3), puis `max(..., default=0)`.
    """
    raise NotImplementedError


# ============================================================================ Contraindre et résoudre


def satisfait(version: str, contrainte: str) -> bool:
    """Défi 238 🟡 — True si `version` respecte `contrainte`.

        >>> satisfait("1.9.0", "^1.2.3"), satisfait("2.0.0", "^1.2.3")
        (True, False)
        >>> satisfait("0.2.9", "^0.2.3"), satisfait("0.3.0", "^0.2.3"), satisfait("1.3.0", "~1.2.3")
        (True, False, False)
        >>> satisfait("2.4.1", ">=2.0.0, <3.0.0"), satisfait("3.1.0", "^1.0.0 || ^3.0.0")
        (True, True)
        >>> satisfait("2.2.0-rc.1", ">=2.0.0"), satisfait("2.2.0-rc.1", ">=2.2.0-rc.0"), satisfait("2.2.0-rc.1", "*")
        (False, True, False)

    Ici : `^1.2.3` accepte toute la série 1.x à partir de 1.2.3, mais pas 2.0.0 (une rupture). Pour une version
    0.y.z, chaque mineure peut tout casser : `^0.2.3` s'arrête à 0.3.0. `~1.2.3` n'accepte que des correctifs.
    Une virgule veut dire ET, « || » veut dire OU. Et une pré-version (2.2.0-rc.1) n'est acceptée que si on l'a
    demandée explicitement : « >=2.0.0 » ne veut pas dire « installez-moi une version pas encore finie ».

    Les règles (celles de npm, avec des virgules pour le ET) :

    - `contrainte` est une suite d'ALTERNATIVES séparées par « || » (au moins une doit être remplie) ; chaque
      alternative est une suite de COMPARAISONS séparées par « , » (toutes doivent être remplies). Les espaces
      autour de chaque morceau ne comptent pas ;
    - une comparaison s'écrit « * » (toutes les versions), ou un opérateur suivi d'une version COMPLÈTE (validée
      comme au défi 236 ; des espaces permises entre les deux) : « = » (même rang ; sans opérateur, c'est « = »),
      « > », « >= », « < », « <= », « ^ » et « ~ » ;
    - `^M.m.c` : >= M.m.c et < (M+1).0.0-0 si M > 0 ; < 0.(m+1).0-0 si M = 0 et m > 0 ; < 0.0.(c+1)-0 sinon.
      `~M.m.c` : >= M.m.c et < M.(m+1).0-0. (Le « -0 » est la plus petite pré-version possible : il écarte aussi
      2.0.0-rc.1 de `^1.2.3`.) Les comparaisons se font avec l'ordre du défi 236 ;
    - une version AVEC pré-version ne remplit une alternative que si elle remplit toutes ses comparaisons ET
      qu'une des versions ÉCRITES dans CETTE alternative a une pré-version et les mêmes M.m.c qu'elle. « * » n'en
      écrit aucune ;
    - erreurs : version ou contrainte qui n'est pas un texte → TypeError ; une version invalide, une contrainte
      vide, un morceau vide (« >=1.0.0, », « || ^1.0.0 »), un opérateur inconnu (« =>1.0.0 ») ou une version
      incomplète (« ^1.2 ») → ValueError, même dans une alternative qu'on n'aurait pas eu besoin de lire.

    Ce que vous apprenez : ce que veulent vraiment dire les « ^ » et « ~ » de vos fichiers package.json,
    Cargo.toml ou pyproject.toml, pourquoi `^0.x` est plus strict, et pourquoi une rc ne s'installe pas seule.

    Indice : `contrainte.split("||")`, puis `split(",")` ; essayez les opérateurs du plus long au plus court
    (">=" avant ">") ; pour `^` et `~`, calculez la borne en texte ("2.0.0-0") et comparez avec votre 236.
    """
    raise NotImplementedError


def resoudre(index: dict, demandes: dict) -> dict:
    """Défi 239 🔴 — {paquet: version} : les versions à installer ensemble pour que toutes les contraintes tiennent.

        >>> index = {"app": {"2.0.0": {"lib": "^2.0.0"}, "1.0.0": {"lib": "^1.0.0"}},
        ...          "lib": {"2.1.0": {}, "1.4.0": {}},
        ...          "outil": {"1.1.0": {"lib": "^1.0.0"}}}
        >>> resoudre(index, {"app": "*"})
        {'app': '2.0.0', 'lib': '2.1.0'}
        >>> resoudre(index, {"app": "*", "outil": "^1.0.0"})          # app 2.0.0 rend outil impossible
        {'app': '1.0.0', 'outil': '1.1.0', 'lib': '1.4.0'}
        >>> resoudre(index, {"app": "^2.0.0", "outil": "*"})
        Traceback (most recent call last):
        ...
        defis.livraison.Conflit: aucune combinaison de versions ne satisfait {'app': '^2.0.0', 'outil': '*'}

    Ici (2e exemple) : on essaie d'abord app 2.0.0, la plus récente ; elle demande lib ^2.0.0. Mais outil 1.1.0
    demande lib ^1.0.0 : impossible d'avoir les deux. On REVIENT EN ARRIÈRE et on essaie app 1.0.0, qui demande
    lib ^1.0.0 : tout tient. 3e exemple : avec app ^2.0.0 imposée, aucune combinaison ne convient : Conflit.

    Les règles (une recherche avec retour arrière, comme `pip` et `cargo` ; fixée pour qu'il n'y ait qu'une
    bonne réponse) :

    - `index` : {paquet: {version: {dépendance: contrainte}}} ; `demandes` : {paquet: contrainte} ;
    - on DÉCIDE les paquets un par un, dans l'ordre d'une FILE : d'abord ceux de `demandes`, dans leur ordre ;
      quand on choisit une version, ses dépendances qui ne sont ni choisies ni déjà dans la file vont au BOUT de
      la file, dans l'ordre où elle les écrit ;
    - pour décider un paquet, on essaie ses versions de la PLUS RÉCENTE à la plus ancienne (ordre du défi 236). Une
      version est retenue si elle satisfait (défi 238) TOUTES les contraintes connues sur ce paquet (celle de
      `demandes` et celles des versions déjà choisies), et si chacune de SES dépendances déjà choisie (lui-même
      compris) satisfait la contrainte qu'elle pose. On continue alors avec le paquet suivant de la file ;
    - si aucune version d'un paquet ne va (ou si le paquet n'est pas dans l'index), on revient sur le choix
      PRÉCÉDENT et on essaie sa version suivante ;
    - la réponse est un NOUVEAU dict {paquet: version}, dans l'ordre où les paquets ont été décidés. Aucune
      combinaison possible → `Conflit` (de defis/livraison.py) ; ne modifiez ni `index` ni `demandes` ;
    - le correcteur vous passe 60 paquets de 12 versions : essayer toutes les combinaisons (12 puissance 60) est
      hors de question. Écartez une version dès qu'elle contredit un choix déjà fait.

    Ce que vous apprenez : la recherche avec retour arrière (« backtracking »), ce que fait votre gestionnaire de
    paquets quand il « résout » (et pourquoi il met parfois si longtemps), et pourquoi des contraintes trop
    serrées rendent une installation impossible.

    Indice : une fonction récursive `decider(choisis, contraintes, file)` qui renvoie une solution ou None ;
    passez-lui des COPIES (`{**choisis, paquet: version}`, `file[1:] + ajout`) : revenir en arrière, c'est
    simplement essayer la version suivante avec les objets d'avant.
    """
    raise NotImplementedError


# ============================================================================ Déployer peu à peu


def affecter(experience: str, identifiant: str, poids: dict) -> str:
    """Défi 240 🟡 — La variante que voit `identifiant` dans `experience` : toujours la même, dans la bonne proportion.

        >>> affecter("nouveau_score", "U0001", {"on": 25, "off": 75})
        'off'
        >>> affecter("export_csv", "U0001", {"on": 50, "off": 50})        # une autre expérience, un autre seau
        'on'
        >>> import hashlib
        >>> int.from_bytes(hashlib.sha256("nouveau_score:U0001".encode("utf-8")).digest()[:8], "big") % PALIERS
        6430
        >>> from collections import Counter
        >>> Counter(affecter("nouveau_score", f"U{i:04d}", {"on": 25, "off": 75}) for i in range(1, 481))
        Counter({'off': 372, 'on': 108})

    Ici : le conseiller U0001 tombe dans le seau 6430 (sur 10 000) de l'expérience « nouveau_score ». Avec 25 % de
    « on », les seaux 0 à 2499 voient « on » et les seaux 2500 à 9999 « off » : U0001 voit « off ». Dans une autre
    expérience, il tombe dans un autre seau (ici 3693 : « on »). Sur 480 conseillers, 108 voient « on », près du
    quart.

    Les règles (le principe de LaunchDarkly, Unleash ou GrowthBook) :

    - le SEAU : les 8 premiers octets de `hashlib.sha256(f"{experience}:{identifiant}".encode("utf-8")).digest()`,
      lus comme un entier (`int.from_bytes(…, "big")`), modulo PALIERS (10 000). Pas `hash()` : il change à chaque
      lancement de Python, et un conseiller changerait de variante à chaque redémarrage du serveur ;
    - `poids` : {variante: poids entier >= 0}, dans l'ordre du dict, au moins un poids non nul. Avec un total T,
      la variante choisie est la PREMIÈRE dont le poids cumulé C (le sien compris) vérifie
      seau × T < PALIERS × C (c'est-à-dire seau / PALIERS < C / T, comparé en ENTIERS). Une variante de poids 0
      n'est jamais choisie ;
    - erreurs : `experience` ou `identifiant` qui n'est pas un texte → TypeError ; `poids` qui n'est pas un dict,
      ou un poids qui n'est pas un int (un bool, un float, un texte) → TypeError ; un poids négatif, ou un total
      nul (dict vide compris) → ValueError.

    Ce que vous apprenez : le hasard REPRODUCTIBLE (le même conseiller voit toujours la même chose, sur toutes les
    machines), indépendant d'une expérience à l'autre (le nom de l'expérience fait partie de ce qu'on hache), et
    MONOTONE (passer de 10 % à 25 % de « on » ne fait qu'en AJOUTER : personne ne voit la nouveauté disparaître).

    Indice : `seau * total < PALIERS * cumul`, en ajoutant les poids un par un dans l'ordre du dict.
    """
    raise NotImplementedError


def evaluer(drapeaux: dict, nom: str, utilisateur: dict) -> Decision:
    """Défi 241 🔴 — Decision(valeur, raison) : ce que le drapeau `nom` sert à `utilisateur`, et pourquoi.

        >>> drapeaux = drapeaux_exemple()
        >>> evaluer(drapeaux, "export_csv", {"id": "U0001"})                  # coupé : le bouton d'arrêt
        Decision(valeur='off', raison='coupé')
        >>> evaluer(drapeaux, "nouveau_score", {"id": "U0024", "appli": "4.10.2"})          # un testeur
        Decision(valeur='on', raison='ciblé')
        >>> evaluer(drapeaux, "nouveau_score", {"id": "U0024", "appli": None})              # appli inconnue
        Decision(valeur='off', raison='prérequis fiche_v2')
        >>> evaluer(drapeaux, "nouveau_score", {"id": "U0500", "role": "superviseur", "appli": "4.10.2"})
        Decision(valeur='on', raison='règle 1')
        >>> evaluer(drapeaux, "nouveau_score", {"id": "U0004", "site": "Lyon", "role": "conseiller", "appli": "4.2.0"})
        Decision(valeur='off', raison='règle 2')
        >>> evaluer(drapeaux, "nouveau_score", {"id": "U0002", "site": "Lyon", "appli": "4.10.2"})  # rôle absent
        Decision(valeur='off', raison='défaut')

    Ici : « export_csv » est coupé, tout le monde reçoit sa valeur de coupure. « nouveau_score » a pour prérequis
    « fiche_v2 » (qui demande une appli 4.1.0 ou plus) : sans version connue, pas de fiche v2, donc pas de nouveau
    score, même pour un testeur. Un superviseur avec une appli récente remplit la règle 1. U0004, conseiller à
    Lyon, remplit la règle 2, qui répartit 25 % / 75 % : il tombe dans les 75 %. U0002 n'a pas de rôle connu : la
    condition « role pas_dans [stagiaire] » n'est PAS remplie (on ne sait pas s'il est stagiaire), il reçoit le
    défaut.

    Les règles (l'ordre d'évaluation de LaunchDarkly, simplifié) :

    - `drapeaux` : {nom: drapeau} ; un drapeau est un dict : "actif" (True ou False), "coupe" (la valeur servie
      quand il est coupé ou qu'un prérequis manque), "defaut", et facultativement "prerequis" ({autre drapeau:
      valeur attendue}), "cibles" ({valeur: [identifiants]}) et "regles" ([{"si": [conditions], "alors": …}]). Une
      valeur servie ("defaut", "alors") est un texte, ou un dict de poids : on sert alors
      `affecter(nom, utilisateur["id"], poids)` (défi 240 ; le nom du drapeau sert d'expérience) ;
    - l'ordre : 1. "actif" vaut False → Decision(coupe, "coupé") ; 2. pour chaque prérequis, dans l'ordre, on
      évalue l'autre drapeau pour le MÊME utilisateur : si sa valeur n'est pas celle attendue →
      Decision(coupe, "prérequis <autre>") ; 3. si l'id de l'utilisateur est dans une liste de "cibles" (dans
      l'ordre du dict) → Decision(cette valeur, "ciblé") ; 4. la PREMIÈRE règle dont TOUTES les conditions sont
      remplies → Decision(sa valeur, "règle <n>"), n à partir de 1 ; 5. sinon → Decision(defaut, "défaut") ;
    - une condition est une liste [attribut, opérateur, opérande], l'opérateur dans OPERATEURS : "dans" et
      "pas_dans" (l'opérande est une LISTE de valeurs), "version>=" et "version<" (l'opérande est une version ;
      comparaison du défi 236). Un attribut ABSENT (clé manquante ou None) ne remplit AUCUNE condition, pas même
      "pas_dans". Une version d'utilisateur invalide (« 4.10 ») ne remplit pas la condition, sans rien lever ;
    - avant de servir quoi que ce soit, on valide le drapeau évalué : "actif" qui n'est pas un bool (le texte
      "false" est vrai pour un `if` !) → TypeError ; "coupe" qui n'est pas un texte → TypeError ; pas de "defaut"
      → ValueError ; un opérateur inconnu → ValueError ; une opérande de "dans" ou "pas_dans" qui n'est pas une
      liste (ou un tuple, un set) → TypeError ("Ly" in "Lyon" est vrai) ; une opérande de version invalide →
      ValueError ; des poids invalides → comme au défi 240. Toutes les règles sont validées, même celles qu'on
      n'atteint pas ;
    - erreurs : un drapeau inconnu (même en prérequis) → ValueError ; un CYCLE de prérequis (a demande b qui
      demande a) → ValueError, pas une RecursionError ; un utilisateur sans "id" texte → TypeError. Ne modifiez
      ni `drapeaux` ni `utilisateur`.

    Ce que vous apprenez : livrer une nouveauté sans la montrer à tout le monde, couper en une seconde, et ce qu'un
    système de règles doit faire quand une information manque (ne rien promettre).

    Indice : une fonction interne `_evaluer(nom, chemin)` qui garde le tuple des drapeaux en cours d'évaluation
    (un nom déjà dans `chemin` : cycle) ; `all(...)` sur une liste vide vaut True (une règle sans condition
    s'applique à tout le monde).
    """
    raise NotImplementedError

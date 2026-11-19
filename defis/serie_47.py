"""Série 47 — Garder deux systèmes d'accord : versions, empreintes, arbre de Merkle, messages en ordre, horloge hybride, fusion.

🔄 VOUS VOULEZ DÉVELOPPER ? DÈS QU'UNE DONNÉE VIT À DEUX ENDROITS (UN SERVEUR ET UNE APPLI MOBILE, UN CRM ET UNE
FACTURATION, DEUX CENTRES DE DONNÉES), IL FAUT LES GARDER D'ACCORD SANS QU'ELLES SE PARLENT SANS CESSE. ICI, VOUS ÉCRIVEZ
CE QUE FONT LES BASES DE DONNÉES RÉPARTIES (DYNAMO, CASSANDRA, RIAK, COCKROACHDB), GIT, KAFKA ET LES ÉDITEURS
COLLABORATIFS (FIGMA, AUTOMERGE) : DATER SANS SE FIER À L'HEURE, COMPARER SANS TOUT TRANSFÉRER, RECEVOIR SANS DOUBLON,
FUSIONNER SANS RIEN PERDRE. UNE ERREUR ICI NE FAIT PAS PLANTER LE PROGRAMME : ELLE EFFACE LE CHOIX D'UN CLIENT, EN
SILENCE, UNE NUIT SUR DEUX.

Les conseillers modifient les fiches clients dans le CRM ; les clients modifient les leurs dans l'appli mobile (leur
adresse, et surtout leurs consentements : « je refuse les SMS »). Chaque nuit, à 2 h, un script (`synchro_actuelle`)
remet les deux copies d'accord : pour chaque client, la fiche modifiée le plus récemment, selon l'heure du serveur,
est recopiée EN ENTIER sur l'autre ; une fiche qui manque d'un côté est recopiée de l'autre. Sur le mois d'octobre,
le bilan est lourd : 347 champs faux, les 41 clients qui avaient demandé l'effacement de leurs données sont tous
revenus, et 186 clients qui avaient refusé les SMS en recevront à la prochaine campagne. Ce sont souvent les mêmes
que la campagne de rétention a agacés le matin : un client qui refuse les SMS dans l'appli à 9 h, puis qu'un
conseiller appelle à 14 h pour changer son offre, perd son refus la nuit suivante (la fiche du CRM, plus récente,
gagne en entier). Malik, développeur de l'équipe « données clients », remplace ce script. Il écrit les six outils de
cette série :

    284 comparer_versions   deux copies d'une fiche : l'une a-t-elle vu l'autre, ou ont-elles changé chacune de leur côté ?
    285 empreinte           le résumé d'une fiche, identique dans les deux systèmes, au caractère près
    286 differences         les fiches qui diffèrent entre le CRM et la facturation, sans en transférer 48 000
    287 Recepteur           les messages des téléphones, appliqués une fois chacun, dans l'ordre où ils ont été écrits
    288 HorlogeHybride      des estampilles qui ne reculent jamais, même quand l'horloge du serveur recule
    289 fusionner           deux copies fusionnées champ par champ, sans ressusciter les clients effacés

Prérequis : la série 00, les `dict`, les `set` et les classes (série 07). La série 20 (le texte et l'Unicode) éclaire
le défi 285, la série 24 (les graphes) le défi 286, la série 28 (les flux) le défi 287, les séries 14 et 21 (le
temps) le défi 288, la série 29 (comparer et fusionner) et la série 34 (les données personnelles) le défi 289.

📌 Ce qu'il faut savoir (une page, à relire au besoin) :

- une RÉPLIQUE est une copie des données (le CRM, l'appli). Chacune accepte des modifications sans demander la
  permission à l'autre (sinon, l'appli ne marcherait plus dans le métro) ; on les met d'accord ensuite ;
- l'heure d'un serveur n'est PAS une horloge fiable : elle dérive, et la mise à l'heure (NTP) la fait parfois reculer.
  Le 13 octobre à 10 h 14, celle du CRM a reculé de deux minutes (`heure_physique`) ;
- deux modifications sont CONCURRENTES quand aucune des deux n'a été faite en connaissant l'autre. Un vecteur de
  versions le détecte (défi 284) ; une fusion bien faite les garde toutes les deux, champ par champ (défi 289) ;
- une EMPREINTE (un hachage SHA-256) résume une fiche en 64 caractères : deux fiches égales ont la même empreinte. Un
  ARBRE d'empreintes (un arbre de Merkle) résume des milliers de fiches, et permet de trouver celles qui diffèrent en
  ne descendant que là où les résumés diffèrent (défis 285 et 286) ;
- un réseau mobile livre « au moins une fois » : certains messages arrivent deux fois, ou dans le désordre. Numéroter
  les messages permet de les appliquer « exactement une fois, dans l'ordre » (défi 287) ;
- une ESTAMPILLE (l, c, réplique) date une modification : l suit l'heure, c départage ce qui arrive dans la même
  milliseconde (ou pendant que l'horloge recule), et le nom de la réplique départage le reste. Deux estampilles se
  comparent comme deux tuples (défis 288 et 289) ;
- un EFFACEMENT (le droit à l'effacement du RGPD) ne retire pas la fiche : il la remplace par une PIERRE TOMBALE, la
  date de l'effacement, pour que la copie de l'autre système ne la fasse pas revenir (défi 289).

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_47 import *
    >>> evenements = evenements_exemple()
    >>> len(evenements), bilan(synchro_actuelle(evenements))           # le script d'aujourd'hui
    (4751, {'champs_faux': 347, 'ressuscites': 41, 'sms_malgre_refus': 186})
    >>> crm, facturation = parc_exemple()
    >>> len(crm), len(facturation)                                     # comparées une par une, chaque nuit
    (47990, 47987)
    >>> livraisons = livraisons_exemple()
    >>> ecarts_consentement(consentements(livraisons))                # les messages appliqués à leur arrivée
    23

Quand les six défis sont réussis, Malik peut montrer à l'équipe ce que change le nouveau système :

    >>> Counter(comparer_versions(crm, appli) for _nuit, _client, crm, appli in versions_exemple())
    Counter({'apres': 1750, 'avant': 1703, 'concurrentes': 495})
    >>> bilan(vue(rejouer(evenements, HorlogePhysique, fusionner)))    # champ par champ, à l'heure du serveur
    {'champs_faux': 7, 'ressuscites': 0, 'sms_malgre_refus': 0}
    >>> bilan(vue(rejouer(evenements, HorlogeHybride, fusionner)))     # champ par champ, avec l'horloge hybride
    {'champs_faux': 0, 'ressuscites': 0, 'sms_malgre_refus': 0}
    >>> crm, facturation = parc_exemple()
    >>> locale, distante = Replique(crm, nom="crm"), Replique(facturation, nom="facturation")
    >>> ecarts = differences(locale, distante)
    >>> len(ecarts), Counter(ecarts.values()), distante.appels, distante.fiches_transferees
    (37, Counter({'modifiee': 14, 'locale': 13, 'distante': 10}), 82, 467)
    >>> recepteur = Recepteur()
    >>> appliques = [m for message in livraisons[:-RENVOIS] for m in recepteur.recevoir(message)]
    >>> len(appliques), recepteur.en_attente(), sorted(recepteur.manquants().items())
    (6329, 28, [('tel-019', [15]), ('tel-205', [13]), ('tel-206', [13]), ('tel-295', [2])])
    >>> appliques += [m for message in livraisons[-RENVOIS:] for m in recepteur.recevoir(message)]
    >>> len(appliques), recepteur.en_attente(), ecarts_consentement(consentements(appliques))
    (6361, 0, 0)

Sur les 30 nuits d'octobre, le script a dû départager 3 948 fiches modifiées depuis la veille : 495 l'avaient été des
DEUX côtés (concurrentes), et pour celles-là, garder « la plus récente » jetait forcément une modification. Fusionner
champ par champ, avec des pierres tombales, ne perd plus rien de ce que les gens ont saisi, et aucun client effacé ne
revient. Mais avec l'heure du serveur comme estampille, 7 corrections restent perdues : celles qu'une conseillère a
faites dans les deux minutes qui ont suivi le recul de l'horloge du CRM, et qui portaient une heure PLUS ANCIENNE que
la faute qu'elles corrigeaient. L'horloge hybride les retrouve. Côté facturation, 82 allers-retours et 467 fiches
transférées suffisent à trouver les 37 fiches qui diffèrent, au lieu de 48 000 allers-retours. Côté appli, 23 clients
finissaient la journée avec un consentement contraire à leur dernier choix (un message en retard, ou renvoyé deux
fois, qui écrasait le suivant) ; le récepteur n'en applique aucun hors de son tour, signale les 4 téléphones dont un
message s'est perdu, et le soir, quand ils renvoient la suite, chaque message a été appliqué une fois, et une seule.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code. Faites-les dans l'ordre : c'est le plus progressif.
   Chaque défi est noté seul (aucun n'a besoin d'un autre pour être vérifié).
2. Lancez `python scripts/defis.py --defi 284` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_47.py`.

Les outils communs (`evenements_exemple`, `heure_physique`, `synchro_actuelle`, `rejouer`, `vue`, `bilan`,
`versions_exemple`, `HorlogePhysique`, `Replique`, `parc_exemple`, `livraisons_exemple`, `consentements`,
`ecarts_consentement`, `RENVOIS`, `AVANT`, `APRES`, `EGALES`, `CONCURRENTES`, `REPLIQUES`, `CHAMPS`, `PROFONDEUR`,
`CHIFFRES`, `VIDE` et `DERIVE_MAX_MS`) sont dans `defis/synchro.py`, déjà importés ci-dessous.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import hashlib  # noqa: F401 — défi 285 : hashlib.sha256(octets).hexdigest()
import json  # noqa: F401 — défi 285 : json.dumps(...)
import math  # noqa: F401 — défi 285 : math.isfinite(x)
import unicodedata  # noqa: F401 — défi 285 : unicodedata.normalize("NFC", texte)
from collections import Counter  # noqa: F401

from defis.synchro import (  # noqa: F401
    APRES,
    AVANT,
    CHAMPS,
    CHIFFRES,
    CONCURRENTES,
    DERIVE_MAX_MS,
    EGALES,
    PROFONDEUR,
    RENVOIS,
    REPLIQUES,
    VIDE,
    HorlogePhysique,
    Replique,
    bilan,
    consentements,
    ecarts_consentement,
    evenements_exemple,
    heure_physique,
    livraisons_exemple,
    parc_exemple,
    rejouer,
    synchro_actuelle,
    versions_exemple,
    vue,
)

# ============================================================================ Comparer


def comparer_versions(a, b) -> str:
    """Défi 284 🟢 — Deux vecteurs de versions : AVANT, APRES, EGALES ou CONCURRENTES ?

        >>> comparer_versions({"crm": 2, "appli": 1}, {"crm": 3, "appli": 1})
        'avant'
        >>> comparer_versions({"crm": 3, "appli": 1}, {"crm": 2, "appli": 1})
        'apres'
        >>> comparer_versions({"crm": 2, "appli": 0}, {"crm": 2})
        'egales'
        >>> comparer_versions({"crm": 3, "appli": 1}, {"crm": 2, "appli": 4})
        'concurrentes'
        >>> comparer_versions({"crm": 1}, {"appli": 1})
        'concurrentes'
        >>> comparer_versions({"crm": -1}, {})
        Traceback (most recent call last):
          ...
        ValueError: le compteur de 'crm' dans a ne peut pas être négatif (-1)

    Ici : chaque copie d'une fiche porte un vecteur de versions, qui compte les modifications qu'elle a VUES, système
    par système. La copie du CRM, {"crm": 3, "appli": 1}, a vu 3 modifications faites dans le CRM et 1 faite dans
    l'appli ; celle de l'appli, {"crm": 2, "appli": 4}, en a vu 2 du CRM et 4 de l'appli. Chacune a vu quelque chose
    que l'autre n'a pas vu : elles sont CONCURRENTES, et en garder une, c'est jeter des modifications de l'autre. Si
    la copie de l'appli avait été {"crm": 3, "appli": 2}, elle aurait tout vu de celle du CRM, et plus : celle du CRM
    serait AVANT, et on pourrait la remplacer sans rien perdre.

    Les règles :

    - `a` et `b` sont des `dict` {réplique: compteur} (TypeError sinon) ; une réplique est un `str` non vide
      (TypeError, ValueError) ; un compteur est un `int` positif ou nul (TypeError pour un float, un texte ou un
      `bool` ; ValueError s'il est négatif). Vérifiez `a` ET `b` avant de comparer ;
    - une réplique absente d'un vecteur vaut 0 : {"crm": 2, "appli": 0} et {"crm": 2} sont égales ;
    - `a` est INCLUS dans `b` quand, pour chaque réplique (de `a` ou de `b`), le compteur de `a` est inférieur ou égal
      à celui de `b`. Rendez :
        • `EGALES` si `a` est inclus dans `b` et `b` dans `a` ;
        • `AVANT` si `a` est inclus dans `b` seulement (`b` a tout vu de `a`, et plus) ;
        • `APRES` si `b` est inclus dans `a` seulement ;
        • `CONCURRENTES` sinon (ni l'un ni l'autre n'est inclus dans l'autre) ;
    - ne modifiez ni `a` ni `b`.

    Les pièges : comparer les SOMMES (3 + 1 < 2 + 4 : « avant », alors qu'elles sont concurrentes) ; ne regarder que
    les répliques communes aux deux ({"crm": 1} et {"appli": 1} deviendraient « égales ») ; tester l'égalité avec
    `a == b` (un 0 écrit et un 0 absent ne sont plus égaux) ; `b[r]`, qui lève KeyError pour une réplique absente ; un
    seul test (« a inclus dans b ? sinon : après ») qui ne trouve jamais de concurrentes ; `a.setdefault(r, 0)`, qui
    modifie le vecteur reçu.

    Ce que vous apprenez : un ordre PARTIEL (deux choses peuvent n'être ni avant ni après l'autre), les vecteurs de
    versions de Dynamo, Riak et Voldemort, et la différence entre « plus récent » et « a vu l'autre ».

    Indice : `repliques = a.keys() | b.keys()` ; `a_dans_b = all(a.get(r, 0) <= b.get(r, 0) for r in repliques)` ;
    pareil dans l'autre sens ; puis les quatre cas.
    """
    raise NotImplementedError


def empreinte(fiche) -> str:
    """Défi 285 🟢 — L'empreinte SHA-256 d'une fiche, écrite de façon canonique : 64 caractères hexadécimaux.

        >>> empreinte({})
        '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'
        >>> empreinte({"offre": "Fibre 1G", "sms": True}) == empreinte({"sms": True, "offre": "Fibre 1G"})
        True
        >>> empreinte({"ville": "Orléans"}) == empreinte({"ville": unicodedata.normalize("NFD", "Orléans")})
        True
        >>> empreinte({"n": 1}) == empreinte({"n": 1.0}), empreinte({"n": 1}) == empreinte({"n": True})
        (False, False)
        >>> empreinte({"rappels": {"lundi", "jeudi"}})
        Traceback (most recent call last):
          ...
        TypeError: la fiche['rappels'] est un set : pas d'écriture canonique

    Ici : pour savoir si le CRM et la facturation ont la même fiche, on compare leurs empreintes (64 caractères au lieu
    de la fiche entière). Mais il faut que les deux systèmes calculent EXACTEMENT les mêmes octets. Le CRM écrit
    « Orléans » avec un « é » d'un seul caractère ; l'appli, sur iPhone, l'envoie en deux (un « e » suivi d'un accent
    combinant). À l'écran, c'est pareil ; en octets, non : sans normalisation, ces fiches diffèrent chaque nuit, pour
    toujours. De même, {"offre": …, "sms": …} et {"sms": …, "offre": …} sont la même fiche, écrite dans un autre
    ordre.

    Les règles (c'est un PROTOCOLE : deux programmes, peut-être écrits dans deux langages, doivent s'y tenir au
    caractère près) :

    - `fiche` est un `dict` (TypeError sinon) ;
    - préparez une copie CANONIQUE de la fiche, à toute profondeur :
        • les valeurs permises sont None, `bool`, `int`, `float`, `str`, `list`, `tuple` et `dict` ; un `float` NaN ou
          infini → ValueError ; tout autre type (un `set`, qui n'a pas d'ordre, une date, des octets…) → TypeError,
          avec un message qui dit où il est ;
        • chaque texte (valeur ET clé) est normalisé en NFC : `unicodedata.normalize("NFC", texte)` ;
        • les clés de chaque `dict` sont des `str` (TypeError sinon : `json.dumps` écrirait la clé 1 comme "1", et
          {1: x} aurait l'empreinte de {"1": x}) ; deux clés d'un même `dict` qui deviennent identiques une fois
          normalisées → ValueError ;
        • un `tuple` s'écrit comme une liste ;
    - écrivez-la avec `json.dumps(copie, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`,
      encodez le texte en UTF-8, et rendez `hashlib.sha256(octets).hexdigest()` ;
    - ne modifiez pas la fiche.

    Les pièges : `hash(...)`, qui change à chaque lancement de Python (PYTHONHASHSEED) et n'existe pas dans les autres
    langages ; `json.dumps` sans `sort_keys` (l'ordre des clés change l'empreinte) ; `str(fiche)` (l'écriture de Python
    n'est pas un format d'échange) ; les séparateurs par défaut (", " et ": ") ou `ensure_ascii=True` : votre empreinte
    serait stable, mais ce ne serait pas celle que calcule l'autre système ; ne normaliser que le premier niveau de la
    fiche, ou que les valeurs (pas les clés) ; MD5 (32 caractères, et cassé depuis 2004).

    Ce que vous apprenez : qu'un hachage se calcule sur des OCTETS, qu'il faut une écriture canonique pour que deux
    programmes s'accordent (les ETag HTTP, les identifiants d'objets de Git, les signatures numériques ; la RFC 8785
    en définit une autre, qui écrit aussi les nombres à sa façon : 1.0 y devient 1), et les formes normales d'Unicode
    (NFC, NFD).

    Indice : une fonction récursive `canonique(valeur)` : None et `bool`/`int` tels quels, `float` vérifié, `str`
    normalisé, `list`/`tuple` → liste des `canonique(v)`, `dict` → nouveau dict ; sinon TypeError. Puis
    `json.dumps(...)`, `.encode("utf-8")` et `hashlib.sha256(...).hexdigest()`.
    """
    raise NotImplementedError


def differences(locale, distante) -> dict:
    """Défi 286 🟡 — Les fiches qui diffèrent entre deux répliques, en ne descendant que là où les arbres diffèrent.

        >>> locale = Replique({"C00001": "a1", "C00002": "b2", "C00003": "c3"}, profondeur=2)
        >>> distante = Replique({"C00001": "a1", "C00002": "XX", "C00004": "d4"}, profondeur=2)
        >>> sorted(differences(locale, distante).items())
        [('C00002', 'modifiee'), ('C00003', 'locale'), ('C00004', 'distante')]
        >>> pareille = Replique({"C00001": "a1", "C00002": "b2", "C00003": "c3"}, profondeur=2)
        >>> differences(Replique({"C00001": "a1", "C00002": "b2", "C00003": "c3"}, profondeur=2), pareille)
        {}
        >>> pareille.appels                    # une seule question : les racines sont égales, c'est fini
        1

    Ici : le CRM et la facturation ont chacun 48 000 fiches. Chaque nuit, le script d'aujourd'hui les compare toutes,
    une par une, à travers le réseau : 48 000 allers-retours, pour trouver les 37 fiches qui diffèrent. Une `Replique`
    (dans `defis/synchro.py`) range ses fiches dans un arbre : 4 096 feuilles (les préfixes « 000 » à « fff »), et
    chaque nœud a pour empreinte le résumé de celles de ses 16 enfants. Si deux racines sont égales, tout est égal ;
    sinon, on demande les 16 enfants des deux côtés, et on ne descend QUE dans ceux qui diffèrent.

    Les règles :

    - `locale` et `distante` sont deux `Replique` (ou des objets qui ont les mêmes méthodes) :
        • `profondeur` : le nombre de chiffres d'une feuille (3 par défaut, celle du parc de `parc_exemple` ; 2 dans
          les exemples ci-dessus ; 0 : la racine est elle-même une feuille) ;
        • `empreinte(prefixe)` : l'empreinte du nœud `prefixe` ("" est la racine) ;
        • `enfants(prefixe)` : {prefixe + chiffre: empreinte} pour les 16 enfants d'un nœud qui n'est pas une feuille ;
        • `feuille(prefixe)` : {identifiant: empreinte de la fiche} des fiches d'une feuille ;
      chaque appel est un aller-retour réseau, compté dans `appels` (et `feuille` compte aussi les fiches reçues
      dans `fiches_transferees`) ;
    - deux répliques de `profondeur` différentes → ValueError (leurs arbres ne se comparent pas) ;
    - rendez un `dict` {identifiant: "locale" (la fiche n'existe que dans `locale`), "distante" (seulement dans
      `distante`), "modifiee" (des deux côtés, avec des empreintes différentes)}. Les fiches identiques n'y sont pas ;
      deux répliques identiques donnent {} ;
    - le correcteur compte les appels faits à `distante` : pas plus que ceux d'une descente qui compare les racines,
      puis, pour chaque nœud dont les empreintes diffèrent, demande les enfants des deux côtés, et ne continue que
      dans les enfants qui diffèrent ; arrivée à une feuille qui diffère, elle demande les fiches des deux côtés et
      les compare une à une. Jamais `enfants` ni `feuille` sur un sous-arbre identique.

    Les pièges : demander toutes les feuilles (4 096 allers-retours et 48 000 fiches transférées : c'est ce qu'on
    évite) ; descendre partout dès que la racine diffère ; ne trouver que les fiches modifiées, pas celles qui manquent
    d'un côté ; ne parcourir que les fiches de la distante (celles qui n'existent que dans la locale disparaissent) ;
    inverser « locale » et « distante » ; appeler `enfants` sur une feuille (la `Replique` lève ValueError).

    Ce que vous apprenez : l'arbre de Merkle (Git, l'« anti-entropy » de Cassandra et de Dynamo, rsync, les
    sauvegardes incrémentales, les blockchains), un parcours d'arbre guidé par une condition, et à compter les
    allers-retours réseau plutôt que les lignes de code.

    Indice : `if locale.empreinte("") == distante.empreinte(""): return {}` ; une pile `a_voir = [""]` ; pour un
    préfixe plus court que `locale.profondeur` : `ici, la_bas = locale.enfants(p), distante.enfants(p)`, puis
    `a_voir.extend(e for e in ici if ici[e] != la_bas[e])` ; pour une feuille : `ici.keys() | la_bas.keys()`.
    """
    raise NotImplementedError


# ============================================================================ Recevoir et dater


class Recepteur:
    """Défi 287 🟡 — Appliquer les messages de chaque téléphone une fois chacun, dans l'ordre de leurs numéros.

        >>> recepteur = Recepteur()
        >>> m1 = {"source": "tel-001", "numero": 1, "champ": "consentement_sms", "valeur": False}
        >>> m2, m3 = {**m1, "numero": 2, "valeur": True}, {**m1, "numero": 3, "valeur": False}
        >>> recepteur.recevoir(m2)                               # trop tôt : le n° 1 n'est pas arrivé
        []
        >>> recepteur.manquants(), recepteur.en_attente()
        ({'tel-001': [1]}, 1)
        >>> [m["numero"] for m in recepteur.recevoir(m1)]         # le n° 1 arrive : on applique le 1, puis le 2
        [1, 2]
        >>> recepteur.recevoir(m1), recepteur.recevoir(m2)        # des doublons : ignorés
        ([], [])
        >>> [m["numero"] for m in recepteur.recevoir(m3)]
        [3]
        >>> recepteur.recevoir({**m1, "numero": 5})
        []
        >>> recepteur.recevoir({**m1, "numero": 5, "valeur": True})
        Traceback (most recent call last):
          ...
        ValueError: deux messages différents portent le numéro 5 de 'tel-001'

    Ici : le téléphone de Mme Leroy refuse les SMS (message n° 1), les accepte trois minutes plus tard (n° 2), puis les
    refuse à nouveau (n° 3). Le n° 1, passé par une antenne saturée, arrive APRÈS le n° 2 : appliqué à son arrivée, il
    écrase l'accord. Et le n° 3, que le téléphone a renvoyé faute d'accusé de réception, arrive deux fois. Chaque
    téléphone numérote ses messages 1, 2, 3… : le serveur sait donc lequel il attend, garde de côté ceux qui arrivent
    trop tôt, ignore ceux qu'il a déjà appliqués, et voit quels numéros manquent.

    Les règles :

    - `Recepteur()` n'a pas d'argument ; chaque récepteur a son propre état (rien de partagé entre deux récepteurs) ;
    - `recevoir(message)` : `message` est un `dict` (TypeError sinon) qui a une clé "source" (un `str` non vide) et
      une clé "numero" (un `int` ≥ 1, pas un `bool`) ; une clé absente, une source vide ou un numéro < 1 →
      ValueError ; un mauvais type → TypeError. Les autres clés sont libres ; ne modifiez pas le message ;
    - chaque source numérote ses messages à partir de 1, indépendamment des autres sources. `recevoir` rend la LISTE
      des messages à appliquer maintenant, dans l'ordre :
        • un numéro déjà appliqué : c'est un doublon, rendez [] (on ne garde pas les messages appliqués : un compteur
          par source suffit, et un doublon arrivé tard est ignoré, quel que soit son contenu) ;
        • un numéro déjà en attente : s'il est égal (`==`) au message en attente, c'est un doublon, rendez [] ; s'il
          est différent → ValueError, sans rien changer ;
        • un numéro plus grand que celui qu'on attend : le message attend son tour, rendez [] ;
        • le numéro attendu : rendez [ce message, puis les messages en attente qui le suivent, tant que les numéros se
          suivent] ; ils ne sont plus en attente. Rendez les messages reçus eux-mêmes (pas besoin de copies) ;
    - `en_attente()` : le nombre de messages en attente, toutes sources confondues ;
    - `manquants()` : un `dict` {source: [numéros manquants]} pour chaque source qui a au moins un message en attente :
      les numéros qui ne sont pas arrivés, du numéro attendu (compris) au plus grand numéro en attente (exclu), dans
      l'ordre croissant. Une source sans message en attente n'y figure pas ;
    - le correcteur envoie 30 000 messages d'une même source, du dernier au premier, et compte les lignes exécutées :
      trouver le suivant doit être immédiat (un `dict` {numéro: message}, pas une liste qu'on parcourt).

    Les pièges : appliquer chaque message à son arrivée (l'ordre du réseau n'est pas celui du téléphone) ; ne pas
    ignorer les doublons ; un seul compteur pour toutes les sources ; commencer à 0 ; appliquer le message attendu en
    oubliant ceux qui attendaient derrière lui (ou en n'en libérant qu'un) ; sauter un trou (appliquer le n° 5 quand
    le n° 4 manque) ; ranger l'état dans un attribut de CLASSE (`attente = {}` sous `class Recepteur:`), partagé par
    tous les récepteurs ; parcourir une liste à chaque message.

    Ce que vous apprenez : à passer d'une livraison « au moins une fois » à « exactement une fois, dans l'ordre » (les
    numéros de séquence de TCP, les positions de lecture de Kafka, les clés d'idempotence des API de paiement), et
    une file de réordonnancement.

    Indice : dans `__init__`, `self.prochain = {}` (source → numéro attendu, 1 si la source est nouvelle) et
    `self.attente = {}` (source → {numéro: message}) ; quand le numéro attendu arrive : une liste avec ce message,
    puis `while prochain in attente: liste.append(attente.pop(prochain)); prochain += 1`.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def recevoir(self, message) -> list:
        raise NotImplementedError

    def en_attente(self) -> int:
        raise NotImplementedError

    def manquants(self) -> dict:
        raise NotImplementedError


class HorlogeHybride:
    """Défi 288 🔴 — Une horloge logique hybride : des estampilles (l, c, réplique) qui ne reculent jamais.

        >>> heure = [1_000]                     # l'heure du serveur, en ms : l'exemple la règle à la main
        >>> horloge = HorlogeHybride("crm", lambda: heure[0])
        >>> horloge.estampiller()
        (1000, 0, 'crm')
        >>> horloge.estampiller()               # la même milliseconde : le compteur départage
        (1000, 1, 'crm')
        >>> heure[0] = 880                      # la mise à l'heure fait RECULER le serveur
        >>> horloge.estampiller(), horloge.estampiller()
        ((1000, 2, 'crm'), (1000, 3, 'crm'))
        >>> horloge.recevoir((1_200, 5, "appli"))       # l'état de l'appli, daté d'un peu plus tard
        (1200, 6, 'crm')
        >>> heure[0] = 1_500
        >>> horloge.estampiller()
        (1500, 0, 'crm')
        >>> horloge.recevoir((1_500 + DERIVE_MAX_MS + 1, 0, "telephone"))
        Traceback (most recent call last):
          ...
        ValueError: l'estampille reçue a 60001 ms d'avance (plus de 60000) : une horloge déréglée en face ; on ne la suit pas

    Ici : le 13 octobre à 10 h 14, la mise à l'heure a fait reculer de deux minutes l'horloge du serveur du CRM. Une
    conseillère tape une adresse à 10 h 13 min 30 s (le serveur, en avance, affiche 10 h 15 min 30 s), et la corrige
    à 10 h 14 min 10 s (le serveur, remis à l'heure, affiche 10 h 14 min 10 s) : la correction porte une heure PLUS
    ANCIENNE que la faute, et la fusion (défi 289) garde la faute. Sept corrections se sont perdues ainsi. Une horloge
    hybride suit l'heure du serveur tant qu'elle avance ; quand elle recule, elle garde son temps et augmente un
    compteur. Et quand elle reçoit l'état d'une autre réplique, elle se place APRÈS lui : ce qu'on écrit ensuite est
    plus récent que ce qu'on vient de recevoir.

    Les règles (l'algorithme HLC de Kulkarni et al., 2014) :

    - `HorlogeHybride(replique, physique, derive_max=DERIVE_MAX_MS)` : `replique` est un `str` non vide (TypeError,
      ValueError) ; `physique` est une fonction SANS argument qui rend l'heure du serveur en millisecondes (TypeError
      si elle n'est pas appelable) ; `derive_max` est un `int` ≥ 0 (TypeError, ValueError) ;
    - l'horloge garde deux nombres : `l`, le plus grand temps vu (0 au départ), et `c`, un compteur (0 au départ) ;
    - `estampiller()` (une modification locale) : lisez `pt = physique()`, un `int` ≥ 0 (TypeError pour un float,
      comme `time.time() * 1000`, ou pour un `bool` ; ValueError s'il est négatif) ; si `pt > l` : `l = pt` et
      `c = 0` ; sinon (l'heure n'a pas avancé, ou a reculé) : `c = c + 1`. Rendez le tuple `(l, c, replique)` ;
    - `recevoir(estampille)` (l'état d'une autre réplique arrive) : `estampille` est un tuple `(lm, cm, autre)`
      (TypeError si ce n'est pas un tuple, ValueError s'il n'a pas trois éléments ; `lm` et `cm` des `int` ≥ 0 et
      `autre` un `str` non vide : TypeError pour un mauvais type, `bool` compris, ValueError pour un nombre négatif
      ou un texte vide). Vérifiez l'estampille, puis lisez `pt = physique()` (mêmes règles qu'au-dessus). Si `lm - pt > derive_max` → ValueError, SANS rien changer
      (une estampille venue du futur : un téléphone réglé en 2031). Sinon, avec `nouveau = max(l, lm, pt)` :
        • si `nouveau == l == lm` : `c = max(c, cm) + 1` ;
        • sinon, si `nouveau == l` : `c = c + 1` ;
        • sinon, si `nouveau == lm` : `c = cm + 1` ;
        • sinon (l'heure du serveur l'emporte) : `c = 0` ;
      puis `l = nouveau` ; rendez `(l, c, replique)` ;
    - ce qui en découle (le correcteur le vérifie sur des milliers d'opérations) : les estampilles d'une horloge sont
      strictement croissantes, même quand l'heure recule ; celle que rend `recevoir` est plus grande que l'estampille
      reçue ; `l` n'est jamais en dessous de l'heure physique.

    Les pièges : rendre `(pt, 0, replique)`, l'heure du serveur (elle recule, et deux modifications de la même
    milliseconde ont la même estampille) ; une horloge de Lamport (`l + 1` à chaque fois : l'ordre est bon, mais
    l'estampille ne dit plus l'heure) ; ne pas augmenter `c` quand l'heure n'avance pas ; un `recevoir` qui ignore
    l'estampille reçue (ce qu'on écrit après l'avoir reçue paraît plus ancien qu'elle) ; `c = 0` quand `lm`
    l'emporte (si `cm` > 0, on rend une estampille PLUS PETITE que celle reçue) ; accepter une estampille de 2031 :
    toutes les modifications suivantes du CRM perdraient contre elle ; lever l'erreur APRÈS avoir changé `l` ou `c`.

    Ce que vous apprenez : l'horloge logique hybride de CockroachDB, YugabyteDB et MongoDB, qui marie l'heure réelle et
    un compteur de Lamport ; pourquoi l'heure d'un serveur n'est pas une horloge fiable (NTP, machines virtuelles,
    secondes intercalaires) ; la causalité (« s'est passé avant »).

    Indice : `self.l = self.c = 0` dans `__init__` ; dans `recevoir`, calculez `nouveau` puis les quatre cas, dans cet
    ordre, en comparant à l'ANCIEN `self.l` ; ne changez `self.l` qu'à la fin.
    """

    def __init__(self, replique, physique, derive_max=DERIVE_MAX_MS) -> None:
        raise NotImplementedError

    def estampiller(self) -> tuple:
        raise NotImplementedError

    def recevoir(self, estampille) -> tuple:
        raise NotImplementedError


# ============================================================================ Fusionner


def fusionner(a, b) -> dict:
    """Défi 289 🔴 — Fusionner deux états, champ par champ : le plus récent gagne, et un effacement est définitif.

        >>> a = {"C0042": {"champs": {"offre": ("Fibre 1G", (100, 0, "crm")),
        ...                           "consentement_sms": (True, (50, 0, "import"))}, "efface": None}}
        >>> b = {"C0042": {"champs": {"offre": ("Essentiel", (50, 0, "import")),
        ...                           "consentement_sms": (False, (90, 0, "appli"))}, "efface": None}}
        >>> fusion = fusionner(a, b)
        >>> sorted(fusion["C0042"]["champs"].items()), fusion["C0042"]["efface"]
        ([('consentement_sms', (False, (90, 0, 'appli'))), ('offre', ('Fibre 1G', (100, 0, 'crm')))], None)
        >>> fusionner(a, b) == fusionner(b, a)
        True
        >>> efface = {"C0042": {"champs": {}, "efface": (120, 0, "crm")}}
        >>> fusionner(fusionner(a, efface), b)          # la vieille copie de l'appli ne fait pas revenir le client
        {'C0042': {'champs': {}, 'efface': (120, 0, 'crm')}}
        >>> fusionner(efface, {"C0042": {"champs": {"email": ("x@exemple.fr", (130, 0, "appli"))}, "efface": None}})
        {'C0042': {'champs': {'email': ('x@exemple.fr', (130, 0, 'appli'))}, 'efface': (120, 0, 'crm')}}

    Ici : le client 42 a refusé les SMS dans l'appli (estampille 90), puis un conseiller a changé son offre dans le CRM
    (100). Le script d'aujourd'hui garde la fiche du CRM EN ENTIER : le refus est perdu. Fusionner champ par champ
    garde les deux changements. Quand un client demande l'effacement de ses données (120), on ne retire pas sa fiche :
    on la remplace par une PIERRE TOMBALE, la date de l'effacement. Sans elle, la copie de l'appli, qui a encore la
    fiche, la ferait revenir à la fusion suivante : les 41 clients effacés en octobre sont tous revenus ainsi. Ce qui
    est écrit APRÈS l'effacement (le client se réinscrit, 130) reste.

    Les règles :

    - un ÉTAT est un `dict` {client: fiche} (TypeError sinon), un client un `str` non vide (TypeError, ValueError) ;
      une fiche est un `dict` qui a exactement deux clés (TypeError si ce n'est pas un dict, ValueError pour d'autres
      clés) :
        • "champs" : un `dict` (TypeError sinon) {champ (`str` non vide): (valeur, estampille)}, un tuple de DEUX
          éléments (TypeError si ce n'est pas un tuple, ValueError s'il n'a pas deux éléments) ;
        • "efface" : None, ou l'estampille de l'effacement ;
      une estampille est un tuple `(l, c, replique)` comme au défi 288 (deux `int` ≥ 0 et un `str` non vide ;
      TypeError ou ValueError). Vérifiez `a` et `b` EN ENTIER avant de fusionner ;
    - deux estampilles se comparent comme deux tuples (`l`, puis `c`, puis le nom de la réplique) ;
    - pour chaque client de `a` ou de `b` (absent d'un côté : comme {"champs": {}, "efface": None}) :
        • son "efface" est le plus grand des effacements des deux côtés (None s'il n'y en a aucun) ;
        • pour chaque champ présent d'un côté ou de l'autre, gardez le couple (valeur, estampille) de la plus GRANDE
          estampille. Une même estampille avec deux valeurs différentes → ValueError (une estampille date UNE
          écriture : une horloge est cassée) ;
        • puis ne gardez que les champs dont l'estampille est plus grande que "efface" (écrits après l'effacement) ;
        • le client reste dans le résultat, même sans aucun champ : c'est sa pierre tombale ;
    - rendez un NOUVEL état : chaque fiche et chaque dict "champs" du résultat est neuf (modifier le résultat ne doit
      pas modifier `a` ni `b`) ; les tuples (valeur, estampille) et les valeurs elles-mêmes peuvent être partagés.
      L'ordre des clients et des champs dans le résultat n'a pas d'importance ; ne modifiez ni `a` ni `b` ;
    - ce qui en découle (le correcteur le vérifie sur des milliers d'états tirés au hasard) : `fusionner(a, b) ==
      fusionner(b, a)` (l'ordre des copies ne compte pas), `fusionner(fusionner(a, b), c) == fusionner(a,
      fusionner(b, c))` (l'ordre des échanges non plus), et `fusionner(a, a) == a` quand aucun champ de `a` n'est plus
      ancien que l'effacement de sa fiche (un échange rejoué deux fois ne change rien). C'est ce qui fait converger les copies, quel que soit le réseau.

    Les pièges : garder la fiche la plus récente EN ENTIER (c'est le script d'aujourd'hui) ; laisser `b` gagner
    (`{**a, **b}`) : la fusion dépend de l'ordre ; retirer le client effacé du résultat (sans pierre tombale, la copie
    suivante le ressuscite) ; garder la pierre tombale sans retirer les champs plus anciens qui viennent de l'AUTRE
    côté ; garder le plus PETIT effacement ; ne comparer que `l` (deux écritures de la même milliseconde : le résultat
    dépend de l'ordre des arguments) ; garder sans rien dire l'une des deux valeurs d'une même estampille ;
    `dict(a)` puis modifier ses fiches (la copie est superficielle : vous modifiez `a`).

    Ce que vous apprenez : les CRDT (« types de données répliqués sans conflit » : le registre « last writer wins »,
    les pierres tombales) de Riak, Redis Enterprise, Automerge et Figma ; pourquoi une fusion doit être commutative,
    associative et idempotente ; ce que demande le droit à l'effacement dans un système réparti.

    Indice : `for client in a.keys() | b.keys():` ; `fa = a.get(client, {"champs": {}, "efface": None})` ;
    `max(versions, key=lambda v: v[1])` pour un champ ; `max(effacements) if effacements else None` ; construisez un
    `{"champs": {...}, "efface": ...}` neuf pour chaque client.
    """
    raise NotImplementedError

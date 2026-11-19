"""Série 47 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_47.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la ligne qui vous a coûté le plus de temps : c'est presque toujours une réplique
absente d'un vecteur traitée comme une erreur au lieu de 0 (défi 284), un `json.dumps` sans `sort_keys` ou sans
normalisation Unicode (défi 285), une descente dans l'arbre qui visite aussi les sous-arbres identiques (défi 286), un
message appliqué dès son arrivée au lieu d'attendre ceux qui le précèdent (défi 287), une horloge qui suit l'heure du
serveur quand elle recule (défi 288), ou un effacement qui retire la fiche au lieu de laisser une pierre tombale
(défi 289).

Fil rouge de la série : deux copies n'ont pas besoin de se parler sans cesse pour finir d'accord, à condition que
chaque copie garde ASSEZ d'information pour qu'on sache, plus tard, ce qui s'est passé avant quoi.

1. un vecteur de versions dit si une copie a VU l'autre (avant, après) ou si elles ont avancé chacune de leur côté
   (concurrentes) : une réplique absente vaut 0, et une seule comparaison ne suffit pas, il en faut deux ;
2. une empreinte se calcule sur une écriture CANONIQUE : clés triées, Unicode normalisé, séparateurs fixés, octets
   UTF-8. Deux programmes qui veulent la même empreinte doivent écrire les mêmes octets, au caractère près ;
3. un arbre d'empreintes (Merkle) trouve les différences sans tout transférer : on ne descend QUE dans les sous-arbres
   dont les empreintes diffèrent ;
4. un réseau livre « au moins une fois » et dans le désordre : un numéro par source, un compteur du prochain attendu,
   une salle d'attente pour ceux qui arrivent trop tôt, et on obtient « exactement une fois, dans l'ordre » ;
5. une horloge hybride (HLC) ne recule jamais, reste proche de l'heure réelle, et donne à ce qui suit un message une
   estampille plus grande que celle du message ;
6. une fusion commutative, associative et idempotente (un CRDT) fait converger les copies quel que soit l'ordre des
   échanges : le plus récent gagne CHAMP par champ, et un effacement laisse une pierre tombale.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata

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


def _entier(valeur, quoi: str) -> int:
    # `True` est un `int` en Python : on l'écarte exprès (un compteur qui vaut True est une erreur de saisie).
    if isinstance(valeur, bool) or not isinstance(valeur, int):
        raise TypeError(f"{quoi} doit être un entier, pas {type(valeur).__name__}")
    if valeur < 0:
        raise ValueError(f"{quoi} ne peut pas être négatif ({valeur})")
    return valeur


def _texte(valeur, quoi: str) -> str:
    if not isinstance(valeur, str):
        raise TypeError(f"{quoi} doit être un texte, pas {type(valeur).__name__}")
    if not valeur:
        raise ValueError(f"{quoi} est vide")
    return valeur


# ============================================================================ Défi 284 🟢


def comparer_versions(a, b) -> str:
    for nom, vecteur in (("a", a), ("b", b)):            # on vérifie TOUT avant de comparer
        if not isinstance(vecteur, dict):
            raise TypeError(f"{nom} doit être un dict {{réplique: compteur}}, pas {type(vecteur).__name__}")
        for replique, compteur in vecteur.items():
            _texte(replique, f"une réplique de {nom}")
            _entier(compteur, f"le compteur de {replique!r} dans {nom}")
    repliques = a.keys() | b.keys()
    # Une réplique absente vaut 0 : on lit avec .get(r, 0), on ne modifie pas les vecteurs reçus.
    a_vu_par_b = all(a.get(r, 0) <= b.get(r, 0) for r in repliques)
    b_vu_par_a = all(b.get(r, 0) <= a.get(r, 0) for r in repliques)
    if a_vu_par_b and b_vu_par_a:
        return EGALES
    if a_vu_par_b:
        return AVANT
    if b_vu_par_a:
        return APRES
    return CONCURRENTES          # chacune a vu quelque chose que l'autre n'a pas vu : il faut FUSIONNER


# ============================================================================ Défi 285 🟢


def _canonique(valeur, chemin: str):
    """Une copie de `valeur` prête pour `json.dumps` : textes en NFC, clés vérifiées, types refusés."""
    if valeur is None or isinstance(valeur, (bool, int)):
        return valeur
    if isinstance(valeur, float):
        if not math.isfinite(valeur):                     # NaN n'est même pas égal à lui-même
            raise ValueError(f"{chemin} vaut {valeur} : seuls les nombres finis ont une écriture JSON")
        return valeur
    if isinstance(valeur, str):
        return unicodedata.normalize("NFC", valeur)
    if isinstance(valeur, (list, tuple)):
        return [_canonique(v, f"{chemin}[{i}]") for i, v in enumerate(valeur)]
    if isinstance(valeur, dict):
        copie = {}
        for cle, v in valeur.items():
            if not isinstance(cle, str):      # json.dumps écrirait 1 comme "1" : {1: x} et {"1": x} se confondraient
                raise TypeError(f"{chemin} a une clé {cle!r} ({type(cle).__name__}) : les clés sont des textes")
            nfc = unicodedata.normalize("NFC", cle)
            if nfc in copie:
                raise ValueError(f"{chemin} a deux clés qui s'écrivent {nfc!r} une fois normalisées")
            copie[nfc] = _canonique(v, f"{chemin}[{nfc!r}]")
        return copie
    # Un set n'a pas d'ordre, une date a dix écritures possibles, des octets ne sont pas du texte : on refuse.
    raise TypeError(f"{chemin} est un {type(valeur).__name__} : pas d'écriture canonique")


def empreinte(fiche) -> str:
    if not isinstance(fiche, dict):
        raise TypeError(f"une fiche est un dict, pas {type(fiche).__name__}")
    texte = json.dumps(_canonique(fiche, "la fiche"), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                       allow_nan=False)
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


# ============================================================================ Défi 286 🟡


def differences(locale, distante) -> dict[str, str]:
    if locale.profondeur != distante.profondeur:
        raise ValueError(f"les deux arbres n'ont pas la même profondeur ({locale.profondeur} et "
                         f"{distante.profondeur}) : leurs empreintes ne se comparent pas")
    ecarts: dict[str, str] = {}
    if locale.empreinte("") == distante.empreinte(""):
        return ecarts                                     # un seul aller-retour quand tout est d'accord
    a_voir = [""]
    while a_voir:
        prefixe = a_voir.pop()
        if len(prefixe) == locale.profondeur:             # une feuille : on compare les fiches une à une
            ici, la_bas = locale.feuille(prefixe), distante.feuille(prefixe)
            for identifiant in ici.keys() | la_bas.keys():
                if identifiant not in la_bas:
                    ecarts[identifiant] = "locale"
                elif identifiant not in ici:
                    ecarts[identifiant] = "distante"
                elif ici[identifiant] != la_bas[identifiant]:
                    ecarts[identifiant] = "modifiee"
        else:                                             # un nœud : on ne descend que là où ça diffère
            ici, la_bas = locale.enfants(prefixe), distante.enfants(prefixe)
            a_voir.extend(enfant for enfant in ici if ici[enfant] != la_bas[enfant])
    return ecarts


# ============================================================================ Défi 287 🟡


class Recepteur:
    def __init__(self) -> None:
        self._prochain: dict[str, int] = {}               # source → le numéro attendu (1 au départ)
        self._attente: dict[str, dict[int, dict]] = {}    # source → {numéro: message arrivé trop tôt}

    def recevoir(self, message) -> list[dict]:
        if not isinstance(message, dict):
            raise TypeError(f"un message est un dict, pas {type(message).__name__}")
        if "source" not in message or "numero" not in message:
            raise ValueError("un message a une \"source\" et un \"numero\"")
        source = _texte(message["source"], "la source")
        numero = _entier(message["numero"], "le numéro")
        if numero < 1:
            raise ValueError("les numéros commencent à 1")
        prochain = self._prochain.get(source, 1)
        if numero < prochain:
            return []                                     # déjà appliqué : un doublon, on l'ignore
        attente = self._attente.get(source, {})
        if numero in attente:
            if attente[numero] != message:
                raise ValueError(f"deux messages différents portent le numéro {numero} de {source!r}")
            return []                                     # déjà en attente : un doublon aussi
        if numero > prochain:
            self._attente.setdefault(source, {})[numero] = message     # trop tôt : il attend son tour
            return []
        appliques = [message]
        prochain += 1
        while prochain in attente:                        # ceux qui attendaient derrière lui peuvent passer
            appliques.append(attente.pop(prochain))
            prochain += 1
        self._prochain[source] = prochain
        if source in self._attente and not attente:
            del self._attente[source]
        return appliques

    def en_attente(self) -> int:
        return sum(len(attente) for attente in self._attente.values())

    def manquants(self) -> dict[str, list[int]]:
        resultat = {}
        for source, attente in self._attente.items():
            debut = self._prochain.get(source, 1)
            resultat[source] = [n for n in range(debut, max(attente)) if n not in attente]
        return resultat


# ============================================================================ Défi 288 🔴


def _estampille(valeur, quoi: str = "une estampille") -> tuple[int, int, str]:
    if not isinstance(valeur, tuple):
        raise TypeError(f"{quoi} est un tuple (l, c, réplique), pas {type(valeur).__name__}")
    if len(valeur) != 3:
        raise ValueError(f"{quoi} a trois éléments (l, c, réplique), pas {len(valeur)}")
    l, c, replique = valeur
    return _entier(l, f"le temps de {quoi}"), _entier(c, f"le compteur de {quoi}"), _texte(replique, "la réplique")


class HorlogeHybride:
    def __init__(self, replique, physique, derive_max=DERIVE_MAX_MS) -> None:
        self.replique = _texte(replique, "la réplique")
        if not callable(physique):
            raise TypeError("physique est une fonction sans argument qui rend l'heure en millisecondes")
        self.derive_max = _entier(derive_max, "derive_max")
        self._physique = physique
        self._l = self._c = 0

    def _maintenant(self) -> int:
        return _entier(self._physique(), "l'heure physique")

    def estampiller(self) -> tuple[int, int, str]:
        pt = self._maintenant()
        if pt > self._l:
            self._l, self._c = pt, 0                      # l'heure a avancé : on la suit
        else:
            self._c += 1                                  # elle n'a pas avancé (ou a reculé) : on compte
        return (self._l, self._c, self.replique)

    def recevoir(self, estampille) -> tuple[int, int, str]:
        lm, cm, _ = _estampille(estampille, "l'estampille reçue")
        pt = self._maintenant()
        if lm - pt > self.derive_max:                     # refusée AVANT de toucher à l'état
            raise ValueError(f"l'estampille reçue a {lm - pt} ms d'avance (plus de {self.derive_max}) : une horloge "
                             "déréglée en face ; on ne la suit pas")
        l = max(self._l, lm, pt)
        if l == self._l == lm:
            c = max(self._c, cm) + 1
        elif l == self._l:
            c = self._c + 1
        elif l == lm:
            c = cm + 1
        else:
            c = 0
        self._l, self._c = l, c
        return (l, c, self.replique)


# ============================================================================ Défi 289 🔴


def _verifier_etat(etat, nom: str) -> None:
    if not isinstance(etat, dict):
        raise TypeError(f"{nom} est un dict {{client: fiche}}, pas {type(etat).__name__}")
    for client, fiche in etat.items():
        _texte(client, f"un client de {nom}")
        if not isinstance(fiche, dict):
            raise TypeError(f"la fiche de {client!r} dans {nom} est un dict, pas {type(fiche).__name__}")
        if set(fiche) != {"champs", "efface"}:
            raise ValueError(f"la fiche de {client!r} dans {nom} a exactement deux clés, \"champs\" et \"efface\"")
        if not isinstance(fiche["champs"], dict):
            raise TypeError(f"les champs de {client!r} dans {nom} sont un dict")
        if fiche["efface"] is not None:
            _estampille(fiche["efface"], f"l'effacement de {client!r}")
        for champ, valeur in fiche["champs"].items():
            _texte(champ, f"un champ de {client!r}")
            if not isinstance(valeur, tuple):
                raise TypeError(f"{client!r}.{champ} est un tuple (valeur, estampille)")
            if len(valeur) != 2:
                raise ValueError(f"{client!r}.{champ} est un tuple de DEUX éléments (valeur, estampille)")
            _estampille(valeur[1], f"l'estampille de {client!r}.{champ}")


_SANS_FICHE = {"champs": {}, "efface": None}


def fusionner(a, b) -> dict:
    _verifier_etat(a, "a")
    _verifier_etat(b, "b")
    fusion = {}
    for client in sorted(a.keys() | b.keys()):
        fa, fb = a.get(client, _SANS_FICHE), b.get(client, _SANS_FICHE)
        effacements = [e for e in (fa["efface"], fb["efface"]) if e is not None]
        efface = max(effacements) if effacements else None           # la pierre tombale la plus récente
        champs = {}
        for champ in sorted(fa["champs"].keys() | fb["champs"].keys()):
            versions = [v for v in (fa["champs"].get(champ), fb["champs"].get(champ)) if v is not None]
            if len(versions) == 2 and versions[0][1] == versions[1][1] and versions[0][0] != versions[1][0]:
                raise ValueError(f"{client!r}.{champ} : deux valeurs différentes portent la même estampille "
                                 f"{versions[0][1]} : une horloge a donné deux fois la même")
            valeur, estampille = max(versions, key=lambda v: v[1])   # champ par champ, le plus récent gagne
            if efface is None or estampille > efface:                # écrit APRÈS l'effacement : il reste
                champs[champ] = (valeur, estampille)
        fusion[client] = {"champs": champs, "efface": efface}        # de NOUVEAUX dict ; les tuples se partagent
    return fusion


# ---------------------------------------------------------------------------- des tests, comme dans tests/


def test_comparer_versions():
    assert comparer_versions({"crm": 1}, {"crm": 2}) == AVANT
    assert comparer_versions({"crm": 2, "appli": 0}, {"crm": 2}) == EGALES          # absente = 0
    assert comparer_versions({"crm": 2}, {"appli": 1}) == CONCURRENTES              # ni l'une ni l'autre n'a tout vu


def test_empreinte():
    assert empreinte({"a": 1, "b": "é"}) == empreinte({"b": unicodedata.normalize("NFD", "é"), "a": 1})
    assert empreinte({"a": 1}) != empreinte({"a": True}) != empreinte({"a": "1"})
    assert empreinte({}) == hashlib.sha256(b"{}").hexdigest()


def test_differences():
    locale, distante = Replique({"x": "1", "y": "2"}, profondeur=2), Replique({"x": "1", "z": "3"}, profondeur=2)
    assert differences(locale, distante) == {"y": "locale", "z": "distante"}
    pareil = Replique({"x": "1", "y": "2"}, profondeur=2)
    assert differences(Replique({"x": "1", "y": "2"}, profondeur=2), pareil) == {} and pareil.appels == 1


def test_recepteur():
    recepteur = Recepteur()
    deux, un = {"source": "t", "numero": 2}, {"source": "t", "numero": 1}
    assert recepteur.recevoir(deux) == [] and recepteur.manquants() == {"t": [1]}
    assert recepteur.recevoir(un) == [un, deux]                     # dans l'ordre, enfin
    assert recepteur.recevoir(un) == [] and recepteur.en_attente() == 0   # un doublon : ignoré


def test_horloge_hybride():
    heure = [1_000]
    horloge = HorlogeHybride("crm", lambda: heure[0])
    assert horloge.estampiller() == (1_000, 0, "crm")
    heure[0] = 900                                                  # l'horloge du serveur recule
    assert horloge.estampiller() == (1_000, 1, "crm")               # l'estampille, elle, avance
    assert horloge.recevoir((1_500, 4, "appli")) == (1_500, 5, "crm")   # après ce qu'on a reçu


def test_fusionner():
    a = {"C1": {"champs": {"offre": ("Fibre", (5, 0, "crm"))}, "efface": None}}
    b = {"C1": {"champs": {"sms": (False, (7, 0, "appli"))}, "efface": None}}
    assert fusionner(a, b) == fusionner(b, a) == {
        "C1": {"champs": {"offre": ("Fibre", (5, 0, "crm")), "sms": (False, (7, 0, "appli"))}, "efface": None}}
    efface = {"C1": {"champs": {}, "efface": (9, 0, "crm")}}
    assert fusionner(fusionner(a, efface), b) == efface                # pas de résurrection

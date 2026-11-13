"""Outils de la série 19, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

C'est le « vieux code » dont parlent les défis : le score de 2022, l'ancienne remise, le script de
relance de nuit, l'ancien opérateur SMS qui ferme et le kit (SDK) du nouveau. Tout est réimporté dans
`defis/serie_19.py` : vous n'avez pas besoin d'ouvrir ce fichier pour faire les défis. Mais LIRE ce code,
comme on lit le code d'une équipe qu'on rejoint, fait partie de l'exercice (surtout pour le défi 118).

    >>> from defis.serie_19 import *
    >>> ancien_score({"client_id": "C002", "charges": 45.0, "contrat": "Annuel", "appels": 0})
    0.35
    >>> remise_ancienne(100, 0.2)
    80.0
    >>> PasserelleAncienne().envoyer_sms("06 12 34 56 78", "Votre offre vous attend")
    True
"""

from __future__ import annotations

import csv
import itertools
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# ============================================================================ Défi 116 : le score de 2022


def ancien_score(client: dict) -> float:
    """Le score de risque écrit en 2022. Il marche, personne ne sait plus très bien pourquoi.

    ⚠️ Il MODIFIE le dictionnaire qu'il reçoit (les charges « 89,90 » deviennent 89.9), et plante sur
    un client sans charges (KeyError) ou aux charges négatives (ValueError).
    """
    client["charges"] = float(str(client["charges"]).replace(",", "."))
    if client["charges"] < 0:
        raise ValueError(f"charges négatives : {client['charges']}")
    score = 0.2 + min(client["charges"], 120) / 300
    if client.get("contrat", "Mensuel") == "Mensuel":
        score += 0.25
    appels = int(client.get("appels", 0))
    if appels >= 3:
        score += 0.1 * (appels - 2)
    return round(min(score, 0.99), 2)


def nouveau_score(client: dict) -> float:
    """La réécriture proposée par une collègue : plus lisible, et elle ne modifie plus le client.

    Fait-elle EXACTEMENT la même chose que `ancien_score` ? C'est la question des défis 116 et 121.
    """
    if "charges" not in client:
        raise ValueError("charges manquantes")
    charges = float(str(client["charges"]).replace(",", "."))
    if charges < 0:
        raise ValueError(f"charges négatives : {charges}")
    score = 0.2 + min(charges, 120) / 300
    if client.get("contrat") == "Mensuel":
        score += 0.25
    score += 0.1 * max(int(client.get("appels", 0)) - 2, 0)
    return round(min(score, 0.99), 2)


#: Cinq clients tels qu'ils sortent de l'export : des charges en texte français, un client sans charges…
CLIENTS_EXEMPLE = [
    {"client_id": "C001", "charges": "89,90", "contrat": "Mensuel", "appels": 4},
    {"client_id": "C002", "charges": 45.0, "contrat": "Annuel", "appels": 0},
    {"client_id": "C003", "charges": 130, "contrat": "Mensuel", "appels": 1},
    {"client_id": "C004", "contrat": "Biannuel", "appels": 2},
    {"client_id": "C005", "charges": -10, "contrat": "Mensuel", "appels": 0},
]

# ============================================================================ Défi 117 : l'ancienne remise


def remise_ancienne(montant: float, taux: float) -> float:
    """Le prix après remise, tel qu'il est calculé depuis 2022 : `taux` est une FRACTION (0.2 pour 20 %)."""
    if not 0 <= taux <= 1:
        raise ValueError(f"taux doit être entre 0 et 1 (reçu : {taux})")
    return round(montant * (1 - taux), 2)


# ============================================================================ Défi 118 : le script de relance de nuit

_ENVOYES_AUJOURDHUI = 0   # « budget du jour » : un compteur GLOBAL, jamais remis à zéro tant que le programme tourne


def ancien_traitement_relances(chemin: str | Path, operateur) -> None:
    """Le script lancé chaque nuit depuis 2022. NE L'APPELEZ PAS sur un vrai opérateur : il envoie des SMS.

    Tout y est mélangé : il lit un fichier, regarde la date du jour, DÉCIDE qui relancer (la partie qui
    compte), envoie les SMS et affiche un compte rendu. Impossible de tester la décision sans un vrai
    fichier, un vrai opérateur et… la bonne date.

    Le fichier : un CSV séparé par des « ; », une ligne par client, avec les colonnes client_id,
    telephone, score (« 0,82 »), contrat, consentement (« oui » / « non ») et dernier_contact
    (« 2025-01-15 », vide si le client n'a jamais été contacté). Voir `LIGNES_EXEMPLE`.
    """
    global _ENVOYES_AUJOURDHUI
    aujourd_hui = date.today()
    with open(chemin, encoding="utf-8", newline="") as fichier:
        clients = list(csv.DictReader(fichier, delimiter=";"))
    for c in clients:
        c["score"] = float(c["score"].replace(",", "."))
    clients.sort(key=lambda c: (-c["score"], c["client_id"]))
    for c in clients:
        if c["score"] < 0.6:
            continue
        if c["consentement"] != "oui":
            continue
        if c["dernier_contact"]:
            dernier = date.fromisoformat(c["dernier_contact"])
            if (aujourd_hui - dernier).days < 30:
                continue
        if _ENVOYES_AUJOURDHUI >= 50:
            print("budget du jour atteint")
            break
        offre = "ENGAGEMENT" if c["contrat"] == "Mensuel" else "FIDELITE"
        operateur.envoyer(c["telephone"], offre)
        _ENVOYES_AUJOURDHUI += 1
        print(f"relance {c['client_id']} : {offre}")


#: Le jour où le fichier d'exemple a été extrait.
JOUR_EXEMPLE = date(2025, 3, 10)

#: Les lignes du fichier de relance, telles que `csv.DictReader` les lit : TOUT est du texte.
LIGNES_EXEMPLE = [
    {"client_id": "C01", "telephone": "0611111111", "score": "0,82", "contrat": "Mensuel", "consentement": "oui",
     "dernier_contact": ""},
    {"client_id": "C02", "telephone": "0622222222", "score": "0,91", "contrat": "Annuel", "consentement": "oui",
     "dernier_contact": "2025-01-15"},
    {"client_id": "C03", "telephone": "0633333333", "score": "0,6", "contrat": "Mensuel", "consentement": "oui",
     "dernier_contact": "2025-02-08"},
    {"client_id": "C04", "telephone": "0644444444", "score": "0,75", "contrat": "Mensuel", "consentement": "oui",
     "dernier_contact": "2025-02-20"},
    {"client_id": "C05", "telephone": "0655555555", "score": "0,95", "contrat": "Biannuel", "consentement": "non",
     "dernier_contact": ""},
    {"client_id": "C06", "telephone": "0666666666", "score": "0,59", "contrat": "Mensuel", "consentement": "oui",
     "dernier_contact": ""},
    {"client_id": "C07", "telephone": "0677777777", "score": "0,82", "contrat": "Annuel", "consentement": "Oui",
     "dernier_contact": ""},
    {"client_id": "C08", "telephone": "0688888888", "score": "0,82", "contrat": "Biannuel", "consentement": "oui",
     "dernier_contact": ""},
]


def ecrire_lignes(lignes: list[dict], chemin: str | Path) -> Path:
    """Écrit des lignes au format du fichier de relance (pour essayer le vieux script… avec un faux opérateur)."""
    chemin = Path(chemin)
    with open(chemin, "w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=list(LIGNES_EXEMPLE[0]), delimiter=";")
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    return chemin


@dataclass
class OperateurFactice:
    """Un faux opérateur pour le vieux script : il note les envois au lieu de les faire."""

    envois: list[tuple[str, str]] = field(default_factory=list)

    def envoyer(self, telephone: str, offre: str) -> None:
        self.envois.append((telephone, offre))


# ============================================================================ Défi 119 : deux formats de préférences

#: Ce que l'application écrivait jusqu'en 2024 (« format 1 », sans numéro de version).
PREFERENCES_V1_EXEMPLE = '{"sms": true, "email": false, "heure": 18}'

#: Ce qu'elle écrit depuis (« format 2 »).
PREFERENCES_V2_EXEMPLE = '{"version": 2, "canaux": ["sms", "courrier"], "heure": 9, "langue": "en"}'

CANAUX = ("courrier", "email", "sms")

# ============================================================================ Défi 120 : l'ancien et le nouvel opérateur SMS


def _numero_e164(numero: str) -> str | None:
    """« 06 12 34 56 78 » → « +33612345678 » ; None si ce n'est pas un numéro français bien formé."""
    brut = re.sub(r"[ .\-]", "", numero)
    if re.fullmatch(r"0[1-9]\d{8}", brut):
        return "+33" + brut[1:]
    if re.fullmatch(r"\+33[1-9]\d{8}", brut):
        return brut
    if re.fullmatch(r"0033[1-9]\d{8}", brut):
        return "+" + brut[2:]
    return None


class PasserelleAncienne:
    """L'ANCIEN opérateur SMS, tel que tout le code l'utilise depuis 2022. Il ferme le mois prochain.

    Son contrat, sur lequel reposent 40 endroits du code :

    `envoyer_sms(numero, texte) -> bool`

    - `numero` est un numéro français, écrit comme on veut : « 06 12 34 56 78 », « 0612345678 »,
      « 06.12.34.56.78 », « +33 6 12 34 56 78 » ou « 0033612345678 » ;
    - renvoie True si le SMS est parti, False s'il n'est pas parti : numéro mal formé, client inscrit sur
      la liste STOP, opérateur momentanément indisponible. L'appelant note l'échec et passe au suivant ;
    - lève ValueError si le texte est vide ou dépasse 160 caractères : c'est un bogue de l'APPELANT, et
      rien n'est envoyé ;
    - lève PermissionError si la clé d'API est refusée : c'est un bogue de CONFIGURATION, il faut le voir
      tout de suite (pas 10 000 clients « injoignables » en silence).
    """

    def __init__(self, *, stop: tuple[str, ...] = (), panne: bool = False, cle_valide: bool = True) -> None:
        self.stop = {_numero_e164(n) for n in stop}
        self.panne = panne
        self.cle_valide = cle_valide
        self.envoyes: list[tuple[str, str]] = []

    def envoyer_sms(self, numero: str, texte: str) -> bool:
        if not texte or len(texte) > 160:
            raise ValueError(f"le texte doit faire de 1 à 160 caractères (reçu : {len(texte)})")
        if not self.cle_valide:
            raise PermissionError("clé d'API refusée par l'opérateur")
        destinataire = _numero_e164(numero)
        if destinataire is None or destinataire in self.stop or self.panne:
            return False
        self.envoyes.append((destinataire, texte))
        return True


class ErreurMessagerie(Exception):
    """L'erreur du NOUVEL opérateur. `code` dit ce qui s'est passé (voir `ClientMessagerie`)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code} : {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class Reponse:
    """La réponse du nouvel opérateur à un envoi."""

    identifiant: str
    statut: str          # "accepte" (le SMS part) ou "refuse" (client sur liste STOP : rien ne part)


class _Messages:
    def __init__(self, client: ClientMessagerie) -> None:
        self._client = client

    def create(self, *, destinataire: str, contenu: str, expediteur: str) -> Reponse:
        """Envoie un SMS. Tous les arguments sont nommés (obligatoire : `create(destinataire=…, …)`)."""
        client = self._client
        client.appels.append({"destinataire": destinataire, "contenu": contenu, "expediteur": expediteur})
        if not client.cle_api.startswith("cle_"):
            raise ErreurMessagerie("CLE_INVALIDE", "clé d'API refusée")
        if client.panne:
            raise ErreurMessagerie(client.panne, "réessayez plus tard")
        if not isinstance(destinataire, str) or not re.fullmatch(r"\+[1-9]\d{7,14}", destinataire):
            raise ErreurMessagerie("NUMERO_INVALIDE", f"format international attendu (+33…), reçu {destinataire!r}")
        if not contenu:
            raise ErreurMessagerie("CONTENU_VIDE")
        if len(contenu) > 1600:
            raise ErreurMessagerie("CONTENU_TROP_LONG")
        if not re.fullmatch(r"[A-Za-z0-9]{3,11}", expediteur):
            raise ErreurMessagerie("EXPEDITEUR_INVALIDE", repr(expediteur))
        identifiant = f"msg_{next(client._numeros):05d}"
        if destinataire in client.refuses:
            return Reponse(identifiant, "refuse")
        segments = -(-len(contenu) // 160)          # au-delà de 160 caractères, le SMS est facturé 2 fois, 3 fois…
        client.envoyes.append((destinataire, contenu, expediteur, segments))
        return Reponse(identifiant, "accepte")


class ClientMessagerie:
    """Le kit (SDK) du NOUVEL opérateur. En vrai, un paquet à installer ; ici, une imitation fidèle, hors ligne.

        client = ClientMessagerie("cle_test_123")
        reponse = client.messages.create(destinataire="+33612345678", contenu="Bonjour", expediteur="RETENTION")
        reponse.statut          # "accepte" ou "refuse"

    Ce qui change par rapport à l'ancien :

    - le numéro doit être au format INTERNATIONAL, sans espace : « +33612345678 » ;
    - il ne renvoie pas un booléen mais une `Reponse` : statut « accepte », ou « refuse » si le client est
      sur la liste STOP ;
    - tout problème lève `ErreurMessagerie`, avec un `code` : « NUMERO_INVALIDE », « INDISPONIBLE »
      (panne passagère), « CLE_INVALIDE », « QUOTA_DEPASSE », « CONTENU_VIDE »… ;
    - il accepte les textes jusqu'à 1 600 caractères… et facture un SMS par tranche de 160 ;
    - `expediteur` (3 à 11 lettres ou chiffres) est le nom affiché sur le téléphone du client.

    Pour les essais : `refuses` liste des numéros (+33…) sur liste STOP, `panne` un code d'erreur levé à
    chaque envoi (« INDISPONIBLE »…). `appels` note chaque appel à create, `envoyes` chaque SMS parti.
    """

    def __init__(self, cle_api: str, *, refuses: tuple[str, ...] = (), panne: str | None = None) -> None:
        self.cle_api = cle_api
        self.refuses = set(refuses)
        self.panne = panne
        self.appels: list[dict] = []
        self.envoyes: list[tuple[str, str, str, int]] = []
        self._numeros = itertools.count(1)
        self.messages = _Messages(self)

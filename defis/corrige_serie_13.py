"""Série 13 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_13.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md l'attaque à laquelle vous n'aviez pas pensé.

Fil rouge de la série : une donnée qui vient de l'extérieur (un formulaire, une URL, un fichier,
un en-tête) est une donnée, jamais du code. Les six failles de la série ont la même cause : on
laisse une donnée être interprétée comme une instruction (par SQLite, par le système de fichiers,
par Excel, par pickle), ou on fait confiance à une donnée qu'on n'a pas vérifiée.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import hashlib
import hmac
import pickle
import re
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

MOTS_SENSIBLES = ("password", "mot_de_passe", "secret", "token", "api_key", "authorization")
MASQUE = "***"
DEBUTS_DANGEREUX = ("=", "+", "-", "@", "\t", "\r")

_CLIENTS = [  # insérés dans le désordre : l'ordre de la table n'est pas l'ordre des identifiants
    ("C004", "Martin", "Mensuel", 89.9),
    ("C001", "Martin", "Annuel", 45.0),
    ("C006", "Martin", "Mensuel", 60.0),
    ("C002", "D'Almeida", "Mensuel", 70.5),
    ("C005", "martin", "Biannuel", 99.0),
    ("C003", "Zébulon", "Biannuel", 55.0),
]


def base_exemple() -> sqlite3.Connection:
    connexion = sqlite3.connect(":memory:")
    connexion.execute("CREATE TABLE clients (id TEXT PRIMARY KEY, nom TEXT NOT NULL, contrat TEXT NOT NULL, "
                      "facture_mensuelle REAL NOT NULL)")
    connexion.executemany("INSERT INTO clients VALUES (?, ?, ?, ?)", _CLIENTS)
    connexion.commit()
    return connexion


# ============================================================================ Défi 80 : injection SQL


def clients_par_nom(connexion: sqlite3.Connection, nom: str, contrat: str | None = None) -> list[str]:
    # Le texte SQL ne contient QUE des morceaux écrits par nous. Les valeurs venues de l'extérieur
    # voyagent à part, dans `parametres` : SQLite les traite comme des valeurs, jamais comme du SQL.
    # « D'Almeida » fonctionne, et « ' OR '1'='1 » est un nom comme un autre (qui ne trouve personne).
    sql = "SELECT id FROM clients WHERE nom = ?"
    parametres = [nom]
    if contrat is not None:
        # On peut construire la requête morceau par morceau : le morceau ajouté est une constante,
        # et sa valeur passe, elle aussi, par un `?`.
        sql += " AND contrat = ?"
        parametres.append(contrat)
    sql += " ORDER BY id"
    return [ligne[0] for ligne in connexion.execute(sql, parametres)]


# À retenir :
# - `nom.replace("'", "''")` « marche » sur les exemples, mais c'est réécrire à la main, en moins
#   bien, ce que SQLite fait déjà. Le jour où quelqu'un oublie un cas (un autre guillemet, un
#   encodage, une autre base de données), la faille revient.
# - Un `?` ne remplace qu'une VALEUR. Un nom de colonne ou un sens de tri (`ORDER BY {colonne}`)
#   ne peut pas être un paramètre : on le choisit alors dans une liste fermée
#   (`if colonne not in {"id", "nom"}: raise ValueError`).
# - Avec pandas : `pd.read_sql_query(sql, connexion, params=parametres)`, même principe.


# ============================================================================ Défi 81 : secrets et journaux


def _est_sensible(cle) -> bool:
    # On compare des noms NORMALISÉS : « X-API-Key » → « x_api_key », qui contient « api_key ».
    # `str(cle)` évite de planter sur une clé qui ne serait pas du texte.
    nom = str(cle).lower().replace("-", "_")
    return any(mot in nom for mot in MOTS_SENSIBLES)


def _masquer_valeur(valeur):
    if isinstance(valeur, dict):
        return masquer_secrets(valeur)          # récursion : un dictionnaire dans un dictionnaire
    if isinstance(valeur, list):
        return [_masquer_valeur(element) for element in valeur]
    return valeur


def masquer_secrets(donnees: dict) -> dict:
    # On construit un NOUVEAU dictionnaire : celui d'origine sert encore à traiter la requête.
    # Le modifier sur place, c'est remplacer la vraie clé d'API par « *** » dans la suite du code.
    return {cle: MASQUE if _est_sensible(cle) else _masquer_valeur(valeur) for cle, valeur in donnees.items()}


# Pour aller plus loin :
# - Le module `logging` permet de brancher ce masquage une fois pour toutes (un `logging.Filter`),
#   plutôt que de compter sur chaque développeur pour y penser à chaque `journal.info(...)`.
# - Un secret peut aussi se cacher DANS une valeur : une URL « …?token=abc » ou un message d'erreur
#   qui recopie la requête. D'où la règle d'or : ne journaliser que ce dont on a besoin.
# - Les données personnelles (e-mail, téléphone) suivent la même logique : le RGPD demande de ne
#   garder que le nécessaire, et les journaux sont souvent oubliés.


# ============================================================================ Défi 82 : traversée de répertoire


def chemin_export(dossier: Path, nom_fichier: str) -> Path:
    racine = Path(dossier).resolve()
    # `resolve()` calcule le chemin RÉEL : il applique les « .. », suit les liens symboliques et
    # transforme un chemin relatif en chemin absolu. Attention : `racine / "/etc/passwd"` vaut
    # « /etc/passwd » (un nom absolu remplace tout ce qui précède).
    chemin = (racine / nom_fichier).resolve()
    # `is_relative_to` compare des CHEMINS, morceau par morceau. `str(chemin).startswith(str(racine))`
    # compare du TEXTE, et « /srv/exports_pirate » commence bien par « /srv/exports »…
    if chemin == racine or not chemin.is_relative_to(racine):
        raise ValueError(f"Nom de fichier refusé : {nom_fichier!r} ne désigne pas un fichier du dossier d'export.")
    return chemin


# À retenir :
# - On vérifie le RÉSULTAT (où va-t-on vraiment écrire ?) plutôt que de chercher des motifs
#   dangereux dans l'entrée (« .. », « / »…). Une liste noire oublie toujours un cas ; un contrôle du
#   résultat, non. Et elle refuse des noms honnêtes comme « rapport..v2.csv ».
# - On renvoie le chemin RÉSOLU, celui qu'on a contrôlé : c'est lui qu'on ouvre ensuite.
# - Encore plus strict, quand c'est possible : ne pas laisser l'utilisateur choisir le nom du tout
#   (`f"export_{date}_{uuid4().hex}.csv"`), ou n'accepter que `[A-Za-z0-9_.-]+` sans sous-dossier.


# ============================================================================ Défi 83 : injection de formules


def _neutraliser(valeur):
    # Seul le TEXTE peut devenir une formule. Un nombre négatif (-12.5) reste un nombre : le
    # transformer en texte « '-12.5 » casserait les sommes du fichier Excel.
    if isinstance(valeur, str) and valeur.startswith(DEBUTS_DANGEREUX):
        return "'" + valeur      # l'apostrophe dit à Excel « ceci est du texte » ; elle ne s'affiche pas
    return valeur


def neutraliser_formules(df: pd.DataFrame) -> pd.DataFrame:
    resultat = df.copy()         # on ne touche pas au tableau d'origine
    for colonne in resultat.columns:
        serie = resultat[colonne]
        # Colonnes de texte : `object` (texte mélangé à d'autres types) ou texte de pandas (`str`).
        # Les colonnes numériques ne peuvent pas contenir de formule : on n'y touche pas.
        if serie.dtype == object or pd.api.types.is_string_dtype(serie.dtype):
            resultat[colonne] = serie.map(_neutraliser)
    return resultat


# Pour aller plus loin :
# - `str.startswith` accepte un tuple : `valeur.startswith(("=", "+"))` teste tous les débuts d'un coup.
# - Les noms de colonnes peuvent aussi venir de l'extérieur (un export « à la carte ») : même traitement.
# - Pourquoi « - » et « + » ? Excel lit « +33 6… » et « -2+3 » comme des calculs, et une formule
#   peut lancer une commande (DDE) ou envoyer le contenu d'autres cellules vers un site (HYPERLINK).


# ============================================================================ Défi 84 : désérialisation


def charger_modele_verifie(chemin: Path, empreinte_attendue: str):
    # 1. On lit le fichier UNE seule fois. Si on le relisait pour le charger, un attaquant pourrait
    #    le remplacer entre la vérification et le chargement (« time of check to time of use »).
    octets = Path(chemin).read_bytes()
    # 2. On vérifie AVANT de charger : `pickle.loads` exécute le code que le fichier lui demande
    #    d'exécuter. Vérifier après, c'est fermer la porte une fois le voleur entré.
    empreinte = hashlib.sha256(octets).hexdigest()
    # L'empreinte lue dans un fichier de manifeste peut finir par « \n » ou être en majuscules.
    # (Une empreinte n'est pas un secret : `==` suffit ici, pas besoin de compare_digest.)
    if empreinte != empreinte_attendue.strip().lower():
        raise ValueError(f"{chemin} : empreinte SHA-256 inattendue (fichier modifié ?). Chargement refusé.")
    # 3. Seulement maintenant, on charge… les octets déjà vérifiés, pas le fichier sur le disque.
    return pickle.loads(octets)


# À retenir :
# - L'empreinte attendue doit venir d'un endroit que l'attaquant ne peut pas modifier (le dépôt Git,
#   le registre de modèles, une variable d'environnement du déploiement). Une empreinte rangée à
#   côté du fichier, dans le même dossier, se remplace en même temps que lui.
# - joblib utilise pickle : `joblib.load` a exactement le même risque. Pour échanger un modèle avec
#   l'extérieur, préférez un format qui ne contient pas de code (ONNX, skops, JSON des coefficients).
# - Avec joblib : `joblib.load(io.BytesIO(octets))` charge les octets vérifiés.


# ============================================================================ Défi 85 : lien signé


class Signataire:
    _JETON = re.compile(r"([A-Za-z0-9_-]{1,64})\.([0-9]{1,12})\.([0-9a-f]{64})")
    _ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

    def __init__(self, secret: bytes) -> None:
        # Une clé trop courte se devine par force brute. On la garde telle quelle : deux serveurs
        # (ou le même après redémarrage) qui ont la même clé produisent et acceptent les mêmes jetons.
        if len(secret) < 16:
            raise ValueError("Le secret doit faire au moins 16 octets (32 recommandés : secrets.token_bytes(32)).")
        self._secret = bytes(secret)

    def _signature(self, message: str) -> str:
        # HMAC, et pas sha256(secret + message) : cette construction « maison » permet d'allonger un
        # message signé sans connaître le secret (attaque par extension de longueur).
        return hmac.new(self._secret, message.encode("ascii"), hashlib.sha256).hexdigest()

    def signer(self, client_id: str, expiration: int) -> str:
        if not self._ID.fullmatch(client_id):
            raise ValueError(f"Identifiant client invalide : {client_id!r}")
        message = f"{client_id}.{int(expiration)}"
        # La signature couvre TOUT ce que le jeton affirme : le client ET l'expiration. Le point qui
        # sépare les champs est signé aussi : sans lui, (« C001 », 17…) et (« C0011 », 7…) donneraient
        # le même message, donc la même signature.
        return f"{message}.{self._signature(message)}"

    def verifier(self, jeton: str, maintenant: int) -> str:
        # 1. La forme d'abord : tout ce qui ne ressemble pas exactement à un jeton est refusé avec une
        #    ValueError (et pas un IndexError ou un TypeError qui deviendrait une erreur 500).
        #    `[0-9]` et pas `\d` : `\d` accepte aussi « ١٢٣ » (chiffres arabes), qu'`int` sait lire.
        forme = self._JETON.fullmatch(jeton) if isinstance(jeton, str) else None
        if forme is None:
            raise ValueError("Jeton mal formé.")
        client_id, texte_expiration, signature = forme.groups()
        # 2. La signature ensuite, recalculée sur le texte EXACT du jeton, et comparée en temps
        #    constant : `==` s'arrête au premier caractère différent, et ce temps de réponse aide un
        #    attaquant à deviner la signature caractère par caractère.
        attendue = self._signature(f"{client_id}.{texte_expiration}")
        if not hmac.compare_digest(signature, attendue):
            raise ValueError("Signature invalide.")
        # 3. Enfin, la date, qu'on ne croit qu'une fois la signature vérifiée.
        if int(texte_expiration) <= maintenant:
            raise ValueError("Lien expiré.")
        return client_id


# Pour aller plus loin :
# - Ce jeton n'est pas chiffré : tout le monde peut LIRE « C001 » dedans. La signature garantit
#   seulement qu'il n'a pas été fabriqué ou modifié. Pour un jeton plus riche, les bibliothèques
#   `itsdangerous` (Flask) et les JWT (PyJWT) font la même chose, avec plus d'options.
# - Pour changer de secret sans casser les liens déjà envoyés : garder l'ancien secret en
#   vérification seulement, pendant la durée de vie des liens.
# - Le même outil sert à PSEUDONYMISER des identifiants pour l'analyse (RGPD) :
#   `hmac.new(secret, b"C001", hashlib.sha256)` est stable mais impossible à inverser sans le secret,
#   alors que `sha256(b"C001")` se retrouve en essayant les 1 000 identifiants possibles.


# ============================================================================ Dans un vrai fichier de tests
#
# Un test de sécurité vérifie qu'une ATTAQUE précise échoue : on écrit l'attaque, noir sur blanc.


def test_une_injection_sql_ne_renvoie_personne():
    assert clients_par_nom(base_exemple(), "' OR '1'='1") == []
    assert clients_par_nom(base_exemple(), "D'Almeida") == ["C002"]


def test_les_secrets_imbriques_sont_masques_sans_toucher_l_original():
    requete = {"headers": {"X-API-Key": "cle-secrete"}, "client_id": "C001"}
    assert masquer_secrets(requete) == {"headers": {"X-API-Key": MASQUE}, "client_id": "C001"}
    assert requete["headers"]["X-API-Key"] == "cle-secrete"


def test_un_nom_d_export_ne_sort_pas_du_dossier():
    with tempfile.TemporaryDirectory() as dossier:
        assert chemin_export(Path(dossier), "risque.csv") == Path(dossier).resolve() / "risque.csv"
        for nom in ["../secret.csv", str(Path(dossier).parent / "x.csv"), ""]:
            try:
                chemin_export(Path(dossier), nom)
            except ValueError:
                continue
            raise AssertionError(f"{nom!r} aurait dû être refusé")


def test_une_formule_devient_du_texte_mais_pas_un_nombre():
    df = pd.DataFrame({"nom": ["=1+1", "Paul"], "facture": [-12.5, 45.0]})
    resultat = neutraliser_formules(df)
    assert resultat["nom"].tolist() == ["'=1+1", "Paul"]
    assert resultat["facture"].tolist() == [-12.5, 45.0]
    assert df["nom"].tolist() == ["=1+1", "Paul"]


def test_un_modele_modifie_n_est_pas_charge():
    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "modele.pkl"
        chemin.write_bytes(pickle.dumps({"modele": "logistique"}))
        empreinte = hashlib.sha256(chemin.read_bytes()).hexdigest()
        assert charger_modele_verifie(chemin, empreinte) == {"modele": "logistique"}
        chemin.write_bytes(pickle.dumps({"modele": "piégé"}))
        try:
            charger_modele_verifie(chemin, empreinte)
        except ValueError:
            return
        raise AssertionError("un fichier modifié aurait dû être refusé")


def test_un_lien_modifie_ou_expire_est_refuse():
    signataire = Signataire(b"0123456789abcdef0123456789abcdef")
    jeton = signataire.signer("C001", 2_000)
    assert signataire.verifier(jeton, maintenant=1_000) == "C001"
    for faux, maintenant in [(jeton.replace("C001", "C002"), 1_000), (jeton, 2_000),
                             (Signataire(b"une autre cle de 32 octets......").signer("C001", 2_000), 1_000)]:
        try:
            signataire.verifier(faux, maintenant)
        except ValueError:
            continue
        raise AssertionError(f"{faux!r} aurait dû être refusé")

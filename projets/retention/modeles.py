"""Les briques du projet : clients, offres, actions (DÉJÀ ÉCRIT : à lire, pas à modifier).

En entreprise, on commence rarement sur une page blanche : il y a du code existant à lire et à
réutiliser. Ce fichier joue ce rôle. Les étapes 1 à 7 s'appuient dessus.

Une `@dataclass` est une petite classe qui range des données : Python écrit pour vous
`__init__`, `__repr__` (l'affichage) et `__eq__` (la comparaison avec ==). Avec `frozen=True`,
un objet ne peut plus être modifié après sa création (`client.proba_churn = 0` lève une erreur) :
aucune fonction ne peut abîmer en douce les données d'une autre.

Les montants et les efficacités des offres sont fictifs, comme les clients du projet 1.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTRATS = ("Mensuel", "Annuel", "Biannuel")

#: Nombre de mois de chiffre d'affaires que l'on cherche à protéger (étape 2).
HORIZON_MOIS = 12

#: Colonnes que le fichier des clients DOIT contenir. Les autres colonnes sont ignorées.
COLONNES_REQUISES = ("customer_id", "proba_churn", "monthly_charges", "tenure_months", "contract", "support_calls")

#: Colonnes du plan de rétention écrit à l'étape 5, dans cet ordre.
COLONNES_PLAN = ("customer_id", "offre", "cout", "proba_churn", "gain_attendu")


@dataclass(frozen=True)
class Client:
    """Un client et sa probabilité de départ, telle que prédite par le modèle du projet 1."""

    customer_id: str
    proba_churn: float      # entre 0 et 1
    monthly_charges: float  # facture mensuelle, en euros
    tenure_months: int      # ancienneté, en mois
    contract: str           # "Mensuel", "Annuel" ou "Biannuel"
    support_calls: int      # nombre d'appels au support


@dataclass(frozen=True)
class Offre:
    """Une action de rétention que l'équipe marketing peut proposer à un client."""

    code: str
    libelle: str
    cout: float                     # coût par client, en euros (toujours > 0)
    efficacite: float               # part des départs évités grâce à l'offre (entre 0 et 1)
    contrats: tuple[str, ...] = ()  # contrats éligibles ; () veut dire « tous les contrats »
    appels_min: int = 0             # réservée aux clients ayant appelé le support au moins N fois
    anciennete_min: int = 0         # réservée aux clients d'au moins N mois d'ancienneté


@dataclass(frozen=True)
class Action:
    """La décision « proposer `offre` à `client` », et ce qu'elle doit rapporter en moyenne."""

    client: Client
    offre: Offre
    gain_attendu: float  # en euros, coût de l'offre déjà déduit (peut être négatif)

    @property
    def rentabilite(self) -> float:
        """Euros gagnés par euro dépensé. Une `property` se lit comme un attribut : `action.rentabilite`."""
        return self.gain_attendu / self.offre.cout


#: Le catalogue des offres, dans l'ordre où le marketing les présente.
OFFRES = (
    Offre("APPEL", "Appel de courtoisie d'un conseiller", cout=8.0, efficacite=0.10),
    Offre("SUPPORT", "Rappel prioritaire par un technicien", cout=20.0, efficacite=0.35, appels_min=3),
    Offre("REMISE", "Remise de 20 % pendant 3 mois", cout=45.0, efficacite=0.30, anciennete_min=3),
    Offre("ENGAGEMENT", "Un mois offert contre un passage à l'annuel", cout=60.0, efficacite=0.45,
          contrats=("Mensuel",)),
)

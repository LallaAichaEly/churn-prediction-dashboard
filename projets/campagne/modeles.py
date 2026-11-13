"""Les briques du projet 3 : groupes, tables, comptages (DÉJÀ ÉCRIT : à lire, pas à modifier).

Comme au projet 2, ce fichier est le code existant que tout le projet réutilise. Il définit :

- les deux groupes de la campagne : les clients qui reçoivent l'offre, et le groupe TÉMOIN, tiré
  au hasard, à qui on ne propose rien. Sans témoin, impossible de savoir si l'offre a servi : ces
  clients seraient peut-être restés de toute façon ;
- le schéma de la base SQLite (`SCHEMA`), lu par `ouvrir` à l'étape 2 ;
- trois petites classes figées (`frozen=True`) qui circulent entre les étapes.
"""

from __future__ import annotations

from dataclasses import dataclass

GROUPE_OFFRE = "offre"    # reçoit l'offre prévue par le plan de rétention
GROUPE_TEMOIN = "temoin"  # tiré au hasard, ne reçoit RIEN : c'est la référence de comparaison
GROUPES = (GROUPE_OFFRE, GROUPE_TEMOIN)

#: Part des clients du plan mis de côté dans le groupe témoin, si on ne précise rien.
PART_TEMOIN = 0.2

#: Le nom de la ligne « toutes offres confondues » dans le bilan (étape 4).
TOUTES = "TOUTES"

#: Colonnes du plan de rétention (la sortie du projet 2) dont on a besoin.
COLONNES_PLAN = ("customer_id", "offre", "cout", "proba_churn")

#: Colonnes du fichier de résultats exporté par le CRM un mois après la campagne (étape 3).
COLONNES_RESULTATS = ("customer_id", "parti")

#: Les deux tables de la base. `CHECK` refuse les valeurs absurdes ; `REFERENCES` interdit un
#: résultat pour un client absent de la campagne… à condition d'activer les clés étrangères,
#: ce que SQLite ne fait PAS par défaut (étape 2).
SCHEMA = """
CREATE TABLE IF NOT EXISTS campagne (
    customer_id TEXT PRIMARY KEY,
    offre       TEXT NOT NULL,
    cout        REAL NOT NULL CHECK (cout >= 0),
    proba_churn REAL NOT NULL CHECK (proba_churn BETWEEN 0 AND 1),
    groupe      TEXT NOT NULL CHECK (groupe IN ('offre', 'temoin'))
);
CREATE TABLE IF NOT EXISTS resultats (
    customer_id TEXT PRIMARY KEY REFERENCES campagne (customer_id),
    parti       INTEGER NOT NULL CHECK (parti IN (0, 1))
);
"""


@dataclass(frozen=True)
class Ciblage:
    """Une ligne du plan de rétention : « proposer `offre` à ce client » (projet 2)."""

    customer_id: str
    offre: str          # code de l'offre : "APPEL", "SUPPORT", "REMISE", "ENGAGEMENT"…
    cout: float         # coût de l'offre pour ce client, en euros
    proba_churn: float  # probabilité de départ prédite par le modèle du projet 1


@dataclass(frozen=True)
class Comptage:
    """Ce que la base sait d'un groupe de clients (étape 4)."""

    offre: str          # code de l'offre, ou TOUTES pour le total
    groupe: str         # GROUPE_OFFRE ou GROUPE_TEMOIN
    clients: int        # clients de ce groupe dans la campagne
    avec_resultat: int  # dont on connaît le résultat (le CRM ne sait pas tout)
    partis: int         # dont on sait qu'ils sont partis
    cout: float         # somme des coûts prévus par le plan pour ces clients, en euros

    @property
    def taux(self) -> float | None:
        """Part des départs parmi les résultats connus (None si aucun résultat)."""
        return self.partis / self.avec_resultat if self.avec_resultat else None


@dataclass(frozen=True)
class Effet:
    """L'effet mesuré de l'offre : la différence entre le groupe témoin et le groupe offre (étape 5)."""

    niveau: float          # niveau de confiance de l'intervalle (0.95 pour 95 %)
    taux_offre: float      # part des départs dans le groupe offre
    taux_temoin: float     # part des départs dans le groupe témoin
    ecart: float           # taux_temoin − taux_offre : positif si l'offre retient des clients
    ic_bas: float          # intervalle de confiance de l'écart : [ic_bas, ic_haut]
    ic_haut: float
    clients_sauves: float  # estimation du nombre de départs évités dans le groupe offre
    significatif: bool     # l'intervalle exclut 0 : l'écart ne s'explique pas par le hasard

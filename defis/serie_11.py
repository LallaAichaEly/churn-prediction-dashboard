"""Série 11 — Tester comme un·e pro : frontières, flottants, erreurs, faux objets, fichiers, propriétés.

🧪 VOUS VOULEZ DÉVELOPPER ? APPRENEZ À PROUVER QUE VOTRE CODE MARCHE.

En équipe, personne n'ose modifier un code sans tests. Le défi 39 (série 06) vous a fait écrire un
premier test. Cette série entraîne les situations qui arrivent dès la première semaine : choisir
les bons exemples, comparer des nombres à virgule, vérifier qu'une erreur est bien levée, remplacer
un service extérieur (l'envoi d'e-mails) par un faux, tester un fichier écrit sur le disque, et
tester une fonction dont on ne connaît pas le résultat exact à l'avance.

Le correcteur juge vos tests comme un·e relecteur·rice exigeant·e. Un bon test :

1. PASSE sur le code juste, y compris sur une AUTRE version juste écrite autrement par un·e
   collègue. Sinon, il bloque un refactoring légitime : c'est un test « fragile ».
2. ÉCHOUE sur chaque version boguée préparée par le correcteur. Sinon, il laisse passer des bugs.

On appelle cette méthode le « test de mutation » : on introduit des bugs exprès, puis on compte ceux
que les tests attrapent.

Prérequis : la série 00 et le défi 39 (série 06). Aucune notion de machine learning.
Après cette série, l'étape 7 du projet guidé (`projets/retention/mes_tests.py`) vous paraîtra facile.

✍️  C'EST VOTRE FICHIER. Même mode d'emploi que les autres séries :

1. Remplacez chaque `raise NotImplementedError` par votre test.
2. Essayez-le vous-même sur la fonction fournie, dans un terminal Python :
       >>> from defis.serie_11 import *
       >>> tester_taux_churn(taux_churn)      # aucune erreur = votre test passe
3. Lancez `python scripts/defis.py --defi 68` (le numéro du défi) : ✅ ou ❌ avec l'explication.
4. Bloqué·e ? Relisez l'indice, puis seulement en dernier recours ouvrez `defis/corrige_serie_11.py`.

Les fonctions à tester (`segment_anciennete`, `taux_churn`…) sont déjà écrites et justes. Ne les
modifiez pas : le correcteur juge vos tests, pas ces fonctions. Chaque test reçoit la fonction à
tester en paramètre : n'appelez que celle-là.

Niveaux : 🟢 débutant · 🟡 intermédiaire · 🔴 avancé.
"""

from __future__ import annotations

import random  # défi 73
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest  # pytest.approx (défi 69), pytest.raises (défis 70 et 72)

# ============================================================================ Choisir ses exemples


def segment_anciennete(mois: int) -> str:
    """Segment d'un client selon son ancienneté. Fonction fournie et juste : c'est elle qu'on teste.

    0 à 5 mois → "nouveau" ; 6 à 23 mois → "installé" ; 24 mois et plus → "fidèle".
    """
    if mois < 6:
        return "nouveau"
    if mois < 24:
        return "installé"
    return "fidèle"


def cas_limites_anciennete() -> list[tuple[int, str]]:
    """Défi 68 🟢 (Parcours 6.1) — Choisir ses exemples : les valeurs aux frontières.

    Renvoyez une liste de cas `(mois, segment attendu)` pour tester `segment_anciennete`, par
    exemple `[(3, "nouveau"), (12, "installé")]`. AU PLUS 6 CAS : on ne peut pas tout essayer, il
    faut viser juste.

    Le correcteur vérifie que chaque cas est juste. Il essaie ensuite votre liste sur 5 versions
    boguées (un `<=` à la place d'un `<`, une frontière décalée d'un mois…). Pour chacune, au moins
    un de vos cas doit donner un résultat différent de celui attendu.

    Dans un vrai fichier de tests, cette liste alimente `pytest.mark.parametrize`. pytest crée alors
    un test par ligne et dit précisément laquelle échoue :

        CAS = [(3, "nouveau"), (12, "installé")]

        @pytest.mark.parametrize("mois, attendu", CAS)
        def test_segment_anciennete(mois, attendu):
            assert segment_anciennete(mois) == attendu

    Ce que vous apprenez : les bugs se cachent aux FRONTIÈRES. Une fonction se découpe en zones où
    elle se comporte de la même façon (les « classes d'équivalence ») : un exemple au milieu d'une
    zone en vaut cent. Une erreur `<` / `<=` se trouve en testant la dernière valeur d'une zone ET
    la première de la suivante.

    Indice : combien y a-t-il de frontières ? Testez chaque côté de chacune. Pensez aussi à la plus
    petite ancienneté possible, 0 mois. En Python, 0 se comporte comme False dans un `if`.
    """
    raise NotImplementedError


# ============================================================================ Nombres à virgule


def taux_churn(partis: int, total: int) -> float:
    """Proportion de clients partis, entre 0 et 1, NON arrondie ; 0.0 s'il n'y a aucun client.

    Fonction fournie et juste : c'est elle qu'on teste.
    """
    if total == 0:
        return 0.0
    return partis / total


def tester_taux_churn(taux_churn: Callable[[int, int], float]) -> None:
    """Défi 69 🟢 (Parcours 6.1) — Comparer des nombres à virgule : `pytest.approx`.

    Écrivez un test de `taux_churn` (juste au-dessus) avec des `assert`, comme au défi 39.

    Le piège : l'ordinateur calcule les nombres à virgule en binaire, avec environ 16 chiffres
    significatifs. Le dernier chiffre dépend donc de la façon de calculer :

        >>> 1 / 3
        0.3333333333333333
        >>> 1 - 2 / 3            # mathématiquement, le même nombre
        0.33333333333333337
        >>> 0.1 + 0.2 == 0.3
        False

    Le correcteur lance votre test sur la fonction ci-dessus, puis sur une autre version juste,
    écrite par un·e collègue avec `1 - restants / total`. Votre test doit passer sur les deux : s'il
    compare au dernier chiffre près avec `==`, il casse au premier refactoring. Le correcteur le lance
    enfin sur 5 versions boguées, et votre test doit échouer sur chacune.

    Comparez donc avec une tolérance :

        assert taux_churn(1, 4) == pytest.approx(0.25)
        # pytest.approx tolère par défaut un écart relatif d'un millionième (rel=1e-6).
        # Sans pytest : math.isclose(taux_churn(1, 4), 0.25)

    Ce que vous apprenez : ne jamais comparer avec `==` deux flottants issus d'un calcul. Choisir
    une tolérance assez petite pour attraper les vrais bugs : un arrondi à 2 décimales en est un.

    Indice : pour chaque erreur probable, écrivez un assert qui la révèle. Choisissez au moins un
    exemple dont le résultat a beaucoup de décimales (1 sur 3), et n'oubliez pas le cas « aucun
    client ».
    """
    raise NotImplementedError


# ============================================================================ Tester une erreur

CONTRATS = {"mensuel": "Mensuel", "1 an": "1 an", "2 ans": "2 ans"}


def lire_contrat(texte: str) -> str:
    """Normalise un type de contrat saisi à la main. Fonction fournie et juste : c'est elle qu'on teste.

    Les majuscules et les espaces autour ne comptent pas : « mensuel », « MENSUEL » ou
    «  Mensuel  » donnent "Mensuel" ; « 1 an » donne "1 an" ; « 2 ans » donne "2 ans".
    Toute autre saisie lève une ValueError dont le message CITE la saisie : la personne qui lit
    l'erreur sait ainsi quoi corriger.
    """
    cle = texte.strip().lower()
    if cle not in CONTRATS:
        raise ValueError(f"Contrat inconnu : {texte!r} (attendu : Mensuel, 1 an ou 2 ans)")
    return CONTRATS[cle]


def tester_lire_contrat(lire_contrat: Callable[[str], str]) -> None:
    """Défi 70 🟡 — Tester qu'une erreur est bien levée : `pytest.raises`.

    Testez `lire_contrat` (juste au-dessus). Les saisies valides, même mal tapées, doivent donner la
    bonne valeur. Une saisie inconnue, comme « Mensual » (une faute de frappe courante), doit lever
    une ValueError dont le message cite la saisie.

    Pour vérifier qu'une erreur est levée, un simple `try`/`except` ne suffit pas : si l'erreur
    n'arrive PAS, rien ne se passe et le test réussit quand même. `pytest.raises`, lui, échoue
    dans ce cas :

        with pytest.raises(ValueError, match="Mensual"):
            lire_contrat("Mensual")
        # match : une expression régulière qui doit apparaître dans le message de l'erreur

    Sans pytest, il faut un `else` : `try: … except ValueError: pass else: raise AssertionError(…)`.

    Le correcteur lance votre test sur la fonction ci-dessus, puis sur une autre version juste (même
    comportement, message formulé autrement), puis sur 5 versions boguées typiques d'une fonction de
    saisie. Votre test doit passer sur les deux premières et échouer sur chacune des 5 autres.

    Ce que vous apprenez : tester le chemin d'erreur, aussi important que le chemin heureux. Être
    précis (ValueError, pas Exception) sans être fragile : vérifier l'essentiel du message, pas sa
    formulation exacte.

    Indice : `pytest.raises(Exception)` accepte n'importe quelle erreur, même celle d'un bug. Et du
    message, ne vérifiez que ce que la docstring PROMET.
    """
    raise NotImplementedError


# ============================================================================ Remplacer un service par un faux


def relancer_clients(clients: list[dict], envoyer: Callable[[str, str], None], seuil: float = 0.5) -> int:
    """Envoie une offre de fidélité aux clients à risque. Fonction fournie et juste : c'est elle qu'on teste.

    - `clients` : `[{"customer_id": "C001", "proba": 0.83}, …]`, dans l'ordre où les traiter ;
    - `envoyer(customer_id, message)` envoie un e-mail. En production, c'est un vrai service :
      chaque appel arrive chez un vrai client ;
    - un client est relancé si sa proba est >= seuil, et au plus UNE fois, même s'il apparaît deux
      fois dans la liste ;
    - si `envoyer` lève ConnectionError (serveur indisponible), on passe au client suivant ;
    - renvoie le nombre d'offres réellement parties (les échecs ne comptent pas).

    Le texte du message est libre : le service marketing le change souvent.
    """
    deja_traites = set()
    parties = 0
    for client in clients:
        customer_id = client["customer_id"]
        if client["proba"] < seuil or customer_id in deja_traites:
            continue
        deja_traites.add(customer_id)
        try:
            envoyer(customer_id, f"Bonjour, profitez de -20 % sur votre prochain mois ({customer_id}).")
        except ConnectionError:
            continue  # une panne passagère ne doit pas priver les clients suivants de leur offre
        parties += 1
    return parties


def tester_relancer_clients(relancer_clients: Callable) -> None:
    """Défi 71 🟡 — Remplacer un service extérieur par un faux (une « doublure de test »).

    Comment tester `relancer_clients` sans envoyer de vrais e-mails ? On lui passe un FAUX `envoyer`,
    écrit dans le test, qui note simplement à qui on a écrit :

        envois = []

        def faux_envoyer(customer_id, message):
            envois.append(customer_id)

        clients = [{"customer_id": "C1", "proba": 0.9}, {"customer_id": "C2", "proba": 0.1}]
        assert relancer_clients(clients, faux_envoyer) == 1
        assert envois == ["C1"]

    Un faux peut aussi simuler une panne, ce qu'on ne peut pas provoquer à la demande avec le vrai
    service :

        def envoyer_en_panne_pour_c1(customer_id, message):
            if customer_id == "C1":
                raise ConnectionError("serveur indisponible")
            envois.append(customer_id)

    Le correcteur lance votre test sur la fonction ci-dessus, puis sur une autre version juste (le
    marketing a changé le texte du message), puis sur 6 versions boguées. Votre test doit échouer sur
    chacune. Relisez chaque règle de la docstring : chacune peut cacher un bug.

    Ce que vous apprenez : l'injection de dépendance (la fonction REÇOIT `envoyer` au lieu de
    l'importer) rend un code testable. Un faux enregistre les appels et simule les pannes.
    `unittest.mock.Mock` fait la même chose en plus court (voir le corrigé).

    Indice : une règle = au moins une vérification. Pensez au client pile au seuil, au doublon, à
    une panne au milieu de la liste, au paramètre `seuil` lui-même. Vérifiez à la fois QUI a reçu
    une offre et le NOMBRE renvoyé.
    """
    raise NotImplementedError


# ============================================================================ Tester un fichier


def exporter_rapport(scores: pd.DataFrame, dossier: Path, seuil: float = 0.5) -> Path:
    """Écrit le rapport des clients à risque. Fonction fournie et juste : c'est elle qu'on teste.

    Écrit `dossier / "clients_a_risque.csv"` et renvoie ce chemin :
    - seulement les clients dont la proba est >= seuil, du plus risqué au moins risqué ;
    - deux colonnes, customer_id et proba, séparées par ";" (pour Excel en français), sans index ;
    - n'écrase JAMAIS un rapport existant : FileExistsError si le fichier est déjà là.
    """
    chemin = Path(dossier) / "clients_a_risque.csv"
    if chemin.exists():
        raise FileExistsError(f"Le rapport existe déjà, il n'a pas été écrasé : {chemin}")
    a_risque = scores.loc[scores["proba"] >= seuil, ["customer_id", "proba"]]
    a_risque.sort_values("proba", ascending=False).to_csv(chemin, sep=";", index=False)
    return chemin


def tester_exporter_rapport(exporter_rapport: Callable, dossier: Path) -> None:
    """Défi 72 🟡 — Tester un fichier écrit sur le disque, dans un dossier temporaire.

    Votre test reçoit `exporter_rapport` et `dossier`, un dossier VIDE créé exprès pour lui (et
    supprimé ensuite). Appelez la fonction avec ce dossier, puis RELISEZ le fichier écrit pour
    vérifier ce que promet la docstring :

        scores = pd.DataFrame({"customer_id": ["C1", "C2", "C3"], "proba": [0.4, 0.9, 0.7]})
        exporter_rapport(scores, dossier)
        rapport = pd.read_csv(dossier / "clients_a_risque.csv", sep=";")
        …

    N'écrivez jamais ailleurs que dans `dossier`. Un test qui laisse des fichiers dans le dépôt, ou
    qui écrase un vrai rapport, est un test dangereux. Le correcteur vérifie que votre test ne
    laisse rien dans le dossier courant.

    Avec pytest, ce dossier s'obtient en demandant la « fixture » `tmp_path` : pytest crée un dossier
    neuf pour chaque test et le passe en argument.

        def test_exporter_rapport(tmp_path):
            exporter_rapport(scores, tmp_path)
            …

    Le correcteur lance votre test sur la fonction ci-dessus, puis sur 5 versions boguées : votre
    test doit échouer sur chacune.

    Ce que vous apprenez : tester un effet (un fichier) et pas seulement une valeur renvoyée ; isoler
    un test dans un dossier temporaire ; vérifier qu'un garde-fou (ne pas écraser) marche vraiment.

    Indice : vérifiez les colonnes (`list(rapport.columns)`), l'ordre des clients et l'endroit où le
    fichier est écrit. Et appelez la fonction DEUX fois.
    """
    raise NotImplementedError


# ============================================================================ Tester des propriétés


def repartir(clients: list[str], n_conseillers: int) -> list[list[str]]:
    """Répartit des clients entre conseillers. Fonction fournie et juste : c'est elle qu'on teste.

    Les clients arrivent triés, les plus à risque d'abord ; chaque conseiller reçoit un bloc de
    clients CONSÉCUTIFS. Garanties :

    1. exactement `n_conseillers` listes, même s'il y a moins de clients que de conseillers
       (certaines listes sont alors vides) ;
    2. recollées bout à bout, les listes redonnent exactement `clients` : aucun client perdu,
       doublé ou déplacé ;
    3. les tailles des listes diffèrent d'au plus 1 (la docstring ne dit pas lesquelles reçoivent
       un client de plus) ;
    4. la liste `clients` reçue n'est pas modifiée.

    ValueError si n_conseillers < 1.
    """
    if n_conseillers < 1:
        raise ValueError(f"Il faut au moins un conseiller, reçu : {n_conseillers}")
    taille, reste = divmod(len(clients), n_conseillers)
    groupes, debut = [], 0
    for i in range(n_conseillers):
        fin = debut + taille + (1 if i < reste else 0)
        groupes.append(clients[debut:fin])
        debut = fin
    return groupes


def tester_repartir(repartir: Callable[[list, int], list[list]]) -> None:
    """Défi 73 🔴 — Tester des PROPRIÉTÉS sur des centaines de cas tirés au hasard.

    Que doit renvoyer `repartir(clients, 3)` pour 10 clients ? La docstring ne fixe pas un résultat
    unique (elle ne dit pas qui reçoit le client en plus). Et calculer à la main la bonne réponse
    pour 1 000 clients serait interminable. On teste donc ce qui doit TOUJOURS être vrai, les
    4 garanties, sur beaucoup d'entrées générées au hasard :

        hasard = random.Random(0)   # graine fixe : le test est le même à chaque lancement
        for _ in range(300):
            clients = [f"C{i}" for i in range(hasard.randint(0, 250))]
            n = hasard.randint(1, 15)
            groupes = repartir(clients, n)
            assert len(groupes) == n, f"{len(clients)} clients, {n} conseillers"
            …

    Le correcteur lance votre test sur la fonction ci-dessus, puis sur une AUTRE version juste qui
    donne le client en plus aux derniers conseillers plutôt qu'aux premiers, puis sur 6 versions
    boguées. Plusieurs ne se trompent que pour certaines tailles : tester « 10 clients pour
    3 conseillers » ne suffit pas.

    Ce que vous apprenez : le test par propriétés, qui trouve des bugs auxquels on n'avait pas
    pensé. Une graine fixe, pour qu'un échec soit reproductible. Un message d'assert qui affiche
    l'entrée fautive, sans quoi on ne peut pas reproduire le bug. La bibliothèque Hypothesis
    automatise cette méthode et réduit l'entrée fautive à la plus petite possible.

    Indice : une garantie = un assert. Pour la garantie n° 4, gardez une copie (`avant =
    list(clients)`) et comparez-la après l'appel. Faites varier le nombre de clients de 0 à quelques
    centaines, et le nombre de conseillers de 1 jusqu'à plus que le nombre de clients.
    """
    raise NotImplementedError

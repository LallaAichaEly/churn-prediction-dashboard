"""Série 11 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_11.py`.

Si votre test passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez : notez dans JOURNAL.md l'exemple ou le réflexe auquel vous n'aviez pas pensé.

Fil rouge de la série : un test se juge sur deux questions.
1. Échoue-t-il quand le code est faux ? Chaque `assert` vise une erreur PRÉCISE.
2. Passe-t-il quand le code est juste, même écrit autrement ? Ne vérifiez que ce qui est PROMIS.

À la fin du fichier, les mêmes tests sont écrits comme dans un vrai fichier `tests/test_*.py`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

# ============================================================================ Choisir ses exemples


def cas_limites_anciennete() -> list[tuple[int, str]]:
    # Deux frontières (6 et 24 mois), donc quatre valeurs : la dernière de chaque zone et la
    # première de la suivante. Un `<=` à la place d'un `<` ne se voit QUE là.
    # On ajoute 0, la plus petite valeur possible : `if mois:` est faux pour 0, un bug classique.
    # Un exemple « au milieu » (12 mois) n'apporterait rien : 6 et 23 couvrent déjà la zone.
    return [
        (0, "nouveau"),
        (5, "nouveau"),
        (6, "installé"),
        (23, "installé"),
        (24, "fidèle"),
    ]


# ============================================================================ Nombres à virgule


def tester_taux_churn(taux_churn: Callable[[int, int], float]) -> None:
    # Un cas normal NON symétrique : attrape le pourcentage (25.0), la division par les restés
    # (0.33) et le taux de rétention (0.75). Avec 1 sur 2, la rétention vaudrait aussi 0.5.
    assert taux_churn(1, 4) == pytest.approx(0.25), "1 sur 4 → 0.25 (une proportion, pas 25 %)"
    # Un résultat qui a beaucoup de décimales : attrape l'arrondi à 2 décimales (0.33). La
    # tolérance par défaut (un millionième, en relatif) est bien plus fine que 0.003.
    assert taux_churn(1, 3) == pytest.approx(1 / 3), "1 sur 3 → 0.333…, sans arrondi"
    # Le cas limite. Ici, `==` est sans danger : 0.0 est une valeur EXACTE promise par la
    # docstring, pas le résultat d'un calcul.
    assert taux_churn(0, 0) == 0.0, "aucun client → 0.0, sans planter"


# ============================================================================ Tester une erreur


def tester_lire_contrat(lire_contrat: Callable[[str], str]) -> None:
    # Le chemin heureux : CHAQUE valeur valide (une faute de frappe dans le dictionnaire ne touche
    # parfois qu'une valeur), plus une saisie valide mais « sale ».
    assert lire_contrat("mensuel") == "Mensuel"
    assert lire_contrat("1 an") == "1 an"
    assert lire_contrat("2 ans") == "2 ans"
    assert lire_contrat("  MENSUEL ") == "Mensuel", "majuscules et espaces autour ne comptent pas"

    # Le chemin d'erreur. pytest.raises échoue si AUCUNE erreur n'est levée (le bug « renvoie
    # None ») et laisse passer une erreur d'un autre type (le bug KeyError), qui fait alors
    # échouer le test. `match` vérifie que le message cite la saisie. On ne vérifie que ce mot,
    # pas toute la phrase : sa formulation n'est pas promise et peut changer.
    with pytest.raises(ValueError, match="Mensual"):
        lire_contrat("Mensual")

    # Sans pytest, la même vérification demande un `else`, souvent oublié :
    #     try:
    #         lire_contrat("Mensual")
    #     except ValueError as exc:
    #         assert "Mensual" in str(exc)
    #     else:
    #         raise AssertionError("aucune erreur levée pour 'Mensual'")


# ============================================================================ Remplacer un service par un faux


def tester_relancer_clients(relancer_clients: Callable) -> None:
    # Une seule liste, où chaque ligne vise une règle de la docstring.
    clients = [
        {"customer_id": "C1", "proba": 0.9},
        {"customer_id": "C2", "proba": 0.1},  # sous le seuil : aucune offre
        {"customer_id": "C3", "proba": 0.5},  # PILE au seuil : relancé (>=)
        {"customer_id": "C1", "proba": 0.9},  # doublon : pas de seconde offre
        {"customer_id": "C4", "proba": 0.7},  # panne simulée pour lui…
        {"customer_id": "C5", "proba": 0.6},  # … et le client suivant est quand même relancé
    ]
    envois = []

    def faux_envoyer(customer_id: str, message: str) -> None:
        if customer_id == "C4":
            raise ConnectionError("serveur indisponible")
        envois.append(customer_id)  # on note À QUI on écrit, pas le texte (il est libre)

    parties = relancer_clients(clients, faux_envoyer)
    assert envois == ["C1", "C3", "C5"], f"offres envoyées à {envois}"
    assert parties == 3, f"{parties} offres comptées : l'échec pour C4 ne compte pas"

    # Le paramètre `seuil` est-il vraiment utilisé ? Un paramètre jamais testé est un bug en attente.
    envois.clear()
    assert relancer_clients(clients, faux_envoyer, seuil=0.8) == 1
    assert envois == ["C1"]

    # Variante avec unittest.mock (même test, sans écrire la fonction) :
    #     from unittest.mock import Mock
    #     envoyer = Mock(side_effect=lambda cid, msg: None)
    #     relancer_clients(clients, envoyer)
    #     assert [appel.args[0] for appel in envoyer.call_args_list] == [...]
    # Et pour une fonction qui IMPORTE son service au lieu de le recevoir, pytest offre la fixture
    # `monkeypatch` : monkeypatch.setattr(module, "envoyer", faux_envoyer). Recevoir la dépendance
    # en paramètre reste plus simple à tester : c'est l'injection de dépendance.


# ============================================================================ Tester un fichier


def tester_exporter_rapport(exporter_rapport: Callable, dossier: Path) -> None:
    # Des scores dans le désordre (pour voir le tri), un client sous le seuil, un client pile au seuil.
    scores = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "proba": [0.6, 0.2, 0.9, 0.5],
    })
    chemin = exporter_rapport(scores, dossier)

    attendu = dossier / "clients_a_risque.csv"
    assert attendu.exists(), "le rapport doit être écrit DANS le dossier demandé"
    assert Path(chemin).resolve() == attendu.resolve(), "le chemin renvoyé doit être celui du rapport"

    # On relit le fichier comme le ferait la personne qui l'ouvre : c'est l'effet qu'on teste.
    rapport = pd.read_csv(attendu, sep=";")
    assert list(rapport.columns) == ["customer_id", "proba"], f"colonnes : {list(rapport.columns)}"
    assert list(rapport["customer_id"]) == ["C3", "C1", "C4"], "du plus au moins risqué, C4 (pile au seuil) compris"

    # Le garde-fou : un second export ne doit pas écraser le premier.
    with pytest.raises(FileExistsError):
        exporter_rapport(scores, dossier)


# ============================================================================ Tester des propriétés


def tester_repartir(repartir: Callable[[list, int], list[list]]) -> None:
    hasard = random.Random(0)  # graine fixe : si le test échoue, il échouera pareil chez vos collègues
    for _ in range(300):
        clients = [f"C{i}" for i in range(hasard.randint(0, 250))]  # jusqu'à 250 : au-delà des « petits » cas
        n = hasard.randint(1, 15)                                    # parfois plus de conseillers que de clients
        avant = list(clients)                                        # copie, pour la garantie 4
        groupes = repartir(clients, n)

        contexte = f"({len(avant)} clients, {n} conseillers)"        # pour reproduire un échec
        assert len(groupes) == n, f"{len(groupes)} listes au lieu de {n} {contexte}"
        recolles = [client for groupe in groupes for client in groupe]
        assert recolles == avant, f"clients perdus, doublés ou déplacés {contexte}"
        tailles = [len(groupe) for groupe in groupes]
        assert max(tailles) - min(tailles) <= 1, f"tailles trop inégales {tailles} {contexte}"
        assert clients == avant, f"la liste reçue a été modifiée {contexte}"

    # Les propriétés n'empêchent pas quelques exemples explicites, faciles à lire.
    assert repartir([], 3) == [[], [], []], "aucun client : 3 listes vides"
    with pytest.raises(ValueError):
        repartir(["C1"], 0)

    # Remarquez ce que ce test NE vérifie PAS : QUI reçoit le client en plus. La docstring ne le
    # promet pas, donc une autre version juste peut le donner aux derniers conseillers.
    # Avec la bibliothèque Hypothesis (pip install hypothesis), la génération devient :
    #     @given(st.lists(st.text()), st.integers(min_value=1, max_value=20))
    #     def test_repartir(clients, n): ...
    # et, en cas d'échec, Hypothesis cherche le PLUS PETIT exemple qui casse.


# ============================================================================ Les mêmes tests, version pytest
#
# Dans un vrai projet, ces tests vivent dans `tests/test_clients.py` et pytest les découvre seul
# (fonctions nommées `test_…`). La fonction testée est importée au lieu d'être reçue en paramètre :
#
#     import pytest
#     from monprojet.clients import segment_anciennete, taux_churn, exporter_rapport
#
#     @pytest.mark.parametrize("mois, attendu", [(0, "nouveau"), (5, "nouveau"), (6, "installé"),
#                                                (23, "installé"), (24, "fidèle")])
#     def test_segment_anciennete(mois, attendu):
#         assert segment_anciennete(mois) == attendu
#
#     def test_taux_churn_un_sur_trois():
#         assert taux_churn(1, 3) == pytest.approx(1 / 3)
#
#     def test_exporter_rapport(tmp_path):           # tmp_path : un dossier neuf, fourni par pytest
#         exporter_rapport(scores, tmp_path)
#         ...
#
# `pytest -q` les lance tous ; `pytest -k taux` seulement ceux dont le nom contient « taux » ;
# `pytest -x` s'arrête au premier échec. Les tests du dépôt (`tests/`) en sont de bons exemples.

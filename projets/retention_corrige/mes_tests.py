"""Étape 7 (corrigé) — Un test qui attrape les bugs de `allouer_budget`.

Chaque bloc vise UN bug précis. Un bon test ne vérifie pas que « ça tourne » : il choisit des
exemples où une version fausse donnerait un autre résultat.
"""

from __future__ import annotations

from .modeles import Action, Client, Offre


def _action(identifiant: str, cout: float, gain: float) -> Action:
    client = Client(identifiant, 0.5, 50.0, 12, "Mensuel", 0)
    return Action(client, Offre(f"OFFRE_{identifiant}", "offre de test", cout, 0.3), gain)


def tester_allouer_budget(allouer) -> None:
    a = _action("A", cout=10.0, gain=50.0)    # rentabilité 5
    b = _action("B", cout=40.0, gain=100.0)   # rentabilité 2,5 : le plus gros gain, mais moins rentable
    c = _action("C", cout=30.0, gain=90.0)    # rentabilité 3
    perte = _action("D", cout=5.0, gain=-1.0)  # fait perdre de l'argent
    actions = [b, perte, c, a]
    copie = list(actions)

    # 1. On trie par RENTABILITÉ, pas par gain : avec 40 €, A puis C (140 €) plutôt que B seul (100 €).
    assert allouer(actions, 40.0) == [a, c]
    # 2. La liste reçue n'est pas modifiée (ni triée, ni vidée).
    assert actions == copie
    # 3. Une action trop chère n'arrête pas la boucle : avec 30 €, X (50 €, la plus rentable) ne
    #    rentre pas, mais Y (10 €), examinée APRÈS elle, oui.
    x, y = _action("X", cout=50.0, gain=200.0), _action("Y", cout=10.0, gain=20.0)
    assert allouer([y, x], 30.0) == [y]
    # 4. Une action qui coûte EXACTEMENT le reste du budget est prise (<= et non <).
    assert allouer([a], 10.0) == [a]
    # 5. Le budget est respecté, et une action à gain négatif n'est jamais retenue.
    assert allouer(actions, 50.0) == [a, c]
    assert allouer(actions, 1_000_000.0) == [a, c, b]
    # 6. Budget nul : rien. Budget négatif : erreur explicite.
    assert allouer(actions, 0.0) == []
    try:
        allouer(actions, -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("un budget négatif doit lever ValueError")

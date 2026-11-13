# 🎯 Défis — progresser un pas à la fois

Le [`PARCOURS.md`](../PARCOURS.md) donne la carte. Les **défis** sont les étapes :
chacun est un exercice court, ciblé, avec une **correction automatique**.

## Le principe

Chaque dossier `defi_XX_…/` contient :

| Fichier | Rôle |
|---|---|
| `ENONCE.md` | Le contexte, les consignes, les questions de réflexion |
| `exercice.py` | Le code à compléter (les `raise NotImplementedError`) |
| `test_defi_XX.py` | L'auto-correction : vert = réussi |
| `corrige.py` | Une solution commentée — à ouvrir **après** avoir essayé |

```bash
pytest defis/defi_01_explorer -q                   # vérifier votre travail
DEFI_CORRIGE=1 pytest defis/defi_01_explorer -q    # vérifier le corrigé
```

`pytest -q` à la racine ne lance que les tests du projet : les défis non terminés
ne cassent rien.

## La méthode qui fait vraiment progresser

1. **Essayez seul·e d'abord**, au moins 20 minutes. Bloquer fait partie de l'apprentissage.
2. Lisez le message d'erreur du test : il dit souvent exactement ce qui manque.
3. Ouvrez le corrigé seulement après — et comparez : qu'est-ce qui est plus simple chez l'un ou l'autre ?
4. Notez ce que vous avez appris dans `JOURNAL.md` (à la racine). Dans 3 mois, c'est votre portfolio.
5. Faites un commit par défi réussi : votre historique Git montre votre progression.

## Programme

| # | Défi | Module | Niveau | Statut |
|---|---|---|---|---|
| 01 | [Faire parler les données](defi_01_explorer/ENONCE.md) | 1 — Données | 🟢🟡 | ✅ disponible |
| 02 | Créer une variable et mesurer son effet ; provoquer une fuite de données | 2 — Variables | 🟡🔴 | 🔜 |
| 03 | Ajouter une forêt aléatoire et comparer en validation croisée | 3 — Modèles | 🟡 | 🔜 |
| 04 | Classes déséquilibrées : `class_weight` et ses effets | 3 — Modèles | 🔴 | 🔜 |
| 05 | Seuil métier et courbe précision-rappel | 4 — Évaluation | 🟡 | 🔜 |
| 06 | Calibration : une probabilité de 0,7 vaut-elle 70 % ? | 4 — Évaluation | 🔴 | 🔜 |
| 07 | Expliquer une prédiction client par client | 5 — Interprétation | 🟡 | 🔜 |
| 08 | Test de non-régression du modèle sauvegardé | 6 — Industrialisation | 🟡 | 🔜 |
| 09 | Intégration continue avec GitHub Actions | 6 — Industrialisation | 🟡 | 🔜 |
| 10 | Servir le modèle avec une API FastAPI | 6 — Industrialisation | 🔴 | 🔜 |
| 11 | Détecter la dérive des données (*data drift*) | 6 — Industrialisation | 🔴 | 🔜 |
| 12 | Projet final : le rapport d'une page | Synthèse | 🏁 | 🔜 |

Un nouveau défi est ajouté au fil des sessions. Le rythme idéal : **un défi tous les 2 à 3 jours**,
plutôt que tout d'un coup.

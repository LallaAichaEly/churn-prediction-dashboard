# 🌿 Atelier Git — 8 situations réelles, corrigées automatiquement

Savoir coder ne suffit pas pour travailler en équipe : dès le premier jour, on vous demandera de créer
une branche, de résoudre un conflit, d'annuler une erreur déjà poussée, de « nettoyer votre historique
avant la revue ». Ce sont les moments où l'on se sent le moins compétent·e… et ceux qui s'apprennent le
plus vite, **à condition de s'entraîner là où rien ne peut casser.**

L'atelier fabrique, pour chaque situation, un vrai dépôt git (avec un « GitHub » local et une collègue,
Camille, qui pousse aussi), vous donne une mission, puis **vérifie le résultat** : branches, historique,
contenu des fichiers, dépôt distant. Si ce n'est pas encore ça, le message dit pourquoi et comment s'en
sortir.

```bash
python scripts/atelier_git.py                 # votre progression ✅ / ❌ / ⬜
python scripts/atelier_git.py 1               # prépare le scénario 1 et affiche la mission
cd ~/atelier-git/01-commit-propre             # … travaillez avec git, dans le bac à sable …
python <chemin du projet>/scripts/atelier_git.py 1   # vérifie (la commande exacte est affichée)
```

```
🌿 Atelier Git — 3/8 scénarios réussis · bac à sable : /home/vous/atelier-git

  ✅ 1 🟢 Un commit propre                              status · diff · add · .gitignore · commit · log
  ✅ 2 🟢 J'ai commité sur main par erreur              log --graph · branch · reset --hard · switch · push -u
  ✅ 3 🟢 Urgence : mettre son travail de côté          stash -u · switch · commit -a · stash pop
  ❌ 4 🟡 Résoudre un conflit de fusion                 merge · conflit · add · commit · merge --abort
        ↳ Il manque ENGAGEMENT dans offres.py : en résolvant le conflit, une seule des deux versions a
          été gardée. Un conflit ne veut pas dire « choisir un camp » : ici, il faut les deux lignes.
  ⬜ 5 🟡 Annuler un commit déjà partagé                log · revert · push · (pourquoi pas reset + push --force)
  …
```

✅ mission remplie · ❌ pas encore (le message vous guide) · ⬜ pas commencé · 💥 dépôt illisible
(`--recommencer` le remet à neuf)

## Les scénarios

| # | Niveau | Situation | Vous apprenez |
|---|---|---|---|
| 1 | 🟢 | **Un commit propre** : une correction et son test à enregistrer, au milieu de `debug.log` et `__pycache__/` | `status`, `diff`, `add` fichier par fichier, `.gitignore`, un bon message de commit |
| 2 | 🟢 | **J'ai commité sur main par erreur** : deux commits à déplacer sur une branche, avant de pousser | une branche n'est qu'une étiquette, `reset --hard` (et quand c'est sans danger), `push -u` |
| 3 | 🟢 | **Urgence** : corriger `main` en plein travail inachevé sur une autre branche | `stash -u` (le piège des fichiers non suivis), `commit -a`, `stash pop` |
| 4 | 🟡 | **Un conflit de fusion** : Camille et vous avez modifié les mêmes lignes | lire les marqueurs, garder les deux versions, `add`, `merge --abort` |
| 5 | 🟡 | **Annuler un commit déjà poussé**, que Camille a déjà récupéré | `revert` contre `reset` + `push --force` : ne jamais réécrire l'historique partagé |
| 6 | 🟡 | **Trouver le commit coupable** parmi 24, avec un script de vérification | `bisect start / good / bad / run`, `tag`, pourquoi un script renvoie un code de sortie |
| 7 | 🔴 | **Retrouver un travail « perdu »** : branche supprimée et `reset --hard` malheureux | `reflog`, recréer une branche sur un hash, ce que git ne peut PAS récupérer |
| 8 | 🔴 | **Préparer sa branche pour la revue** : à jour avec `main`, sans « wip » ni « fix typo » | `fetch`, `rebase`, `reset --soft` (ou `rebase -i`), `push --force-with-lease` |

Faites-les dans l'ordre : chacun réutilise les gestes du précédent. Comptez 15 à 30 minutes par
scénario ; les deux derniers peuvent prendre une séance chacun.

## Comment travailler

1. **Lancez le scénario** : `python scripts/atelier_git.py N`. Il crée le dépôt dans
   `~/atelier-git/NN-nom/`, affiche la situation et la mission (aussi écrite dans `MISSION.md`, dans
   le dossier du scénario).
2. **Allez dans le bac à sable** (`cd …`, la commande exacte est affichée) et résolvez la situation
   **avec git, en ligne de commande**. Votre meilleur ami : `git status`, puis
   `git log --oneline --graph --all`, après chaque commande.
3. **Vérifiez** en relançant `python …/scripts/atelier_git.py N`. Relancez autant de fois que vous
   voulez : la vérification ne modifie rien.
4. **Bloqué·e plus de 10 minutes ?** `--indice` (puis `--indice 2`, `3`, `4` : du plus léger au plus
   direct). En dernier recours, `--solution`. Le correcteur vérifie le **résultat**, pas les
   commandes : un autre chemin qui mène au même état est tout aussi juste.
5. **Tout cassé ?** `--recommencer` remet le scénario à zéro. C'est fait pour : dans le bac à sable,
   une erreur ne coûte rien, et c'est en en faisant qu'on apprend à les réparer.
6. **Notez** dans `JOURNAL.md` la commande que vous ne connaissiez pas et ce qui vous a surpris·e.

> 🛡️ **Pourquoi en dehors du projet ?** Le bac à sable est créé dans `~/atelier-git` (modifiable avec
> `--dossier` ou la variable d'environnement `ATELIER_GIT`). Si vous tapiez `git reset --hard` dans le
> mauvais dossier, ce serait sans conséquence pour votre vrai dépôt. Le correcteur refuse d'écrire dans
> un dossier qu'il n'a pas créé. Pour tout effacer à la fin : supprimez `~/atelier-git`.

> ⌨️ **Un éditeur s'ouvre et vous ne savez pas en sortir ?** C'est Vim. Tapez `Échap`, puis `:wq` et
> Entrée (enregistrer et quitter), ou `:q!` (quitter sans enregistrer). Pour éviter l'éditeur :
> `git commit -m "…"`, `git merge --no-edit`, `git revert --no-edit`. Pour en choisir un autre une fois
> pour toutes : `git config --global core.editor "code --wait"` (VS Code) ou `"nano"`.

> 🪪 **Première utilisation de git ?** Vérifiez la version avec `git --version` (2.23 ou plus récent,
> pour `git switch`), puis présentez-vous une fois pour toutes :
> `git config --global user.name "Prénom Nom"` et `git config --global user.email "vous@exemple.fr"`.
> (Si vous ne le faites pas, l'atelier pose une identité provisoire dans chaque bac à sable.)

## Après l'atelier

- **Sur le vrai projet** : faites chaque série de défis ou chaque étape du
  [projet guidé](../projets/README.md) sur sa propre branche, avec un commit par ✅. À la fin,
  préparez la branche comme au scénario 8 et ouvrez une pull request sur GitHub : relisez votre propre
  diff avant de demander une revue.
- **Le test de bisection** : le défi 43 (`defis/serie_06.py`) vous fait écrire l'algorithme que
  `git bisect` utilise, et le scénario 6 vous le fait utiliser.
- **Pour aller plus loin** : `git cherry-pick` (récupérer un seul commit d'une autre branche),
  `git worktree` (deux branches ouvertes en même temps), `git blame` et `git log -S "texte"` (qui a
  écrit cette ligne, et quand a-t-elle apparu ?), les *hooks* (`pre-commit` pour lancer `pytest`
  avant chaque commit).

Les tests de l'atelier (`tests/test_atelier_git.py`) préparent chaque scénario, appliquent la solution
affichée par `--solution` (elle doit donner ✅) et les erreurs classiques (elles doivent donner ❌ avec
la bonne explication), y compris avec une configuration git de Windows.

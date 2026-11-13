"""Série 18 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_18.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la situation à laquelle vous n'aviez pas pensé.

Fil rouge de la série : on n'optimise pas à l'intuition. Trois réflexes reviennent dans chaque défi :

1. mesurer d'abord, et mesurer juste (répéter, garder le minimum, `perf_counter`), puis profiler pour
   savoir OÙ le temps passe : c'est presque toujours une seule boucle ;
2. changer d'ALGORITHME à cet endroit (un dictionnaire, un `set`, une liste triée et `bisect`, un
   calcul sorti de la boucle, une lecture en flux) : c'est là que se gagnent les facteurs 100 ou 1000 ;
3. prouver que rien n'a changé : on garde l'ancienne version comme référence et on compare les deux
   sur des centaines d'entrées tirées au hasard. Une optimisation qui change le résultat est un bogue.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import bisect
import cProfile
import csv
import functools
import pstats
import random
import time
from collections import Counter, OrderedDict
from datetime import date
from pathlib import Path
from typing import Callable

from defis.performance import (  # noqa: F401
    CONTRATS,
    TARIFS_EXEMPLE,
    HorlogeFactice,
    clients_a_rappeler_lent,
    ecrire_factures,
    exemple_rappels,
    programme_exemple,
)

# ============================================================================ Mesurer


def chronometrer(fonction: Callable[[], object], repetitions: int = 5,
                 horloge: Callable[[], float] = time.perf_counter) -> float:
    # On vérifie AVANT de lancer quoi que ce soit : `min([])` lèverait bien une ValueError, mais avec un
    # message (« min() arg is an empty sequence ») qui ne dit pas à l'appelant ce qu'il a mal fait.
    if repetitions < 1:
        raise ValueError(f"repetitions doit valoir au moins 1 (reçu : {repetitions})")
    durees = []
    for _ in range(repetitions):
        debut = horloge()          # juste avant l'appel…
        fonction()                 # (pas de try/except : une erreur doit remonter)
        durees.append(horloge() - debut)   # … et juste après : on ne mesure que la fonction
    # Le minimum, pas la moyenne : les écarts entre deux passages viennent d'ailleurs (autres programmes,
    # ramasse-miettes, cache du disque), jamais d'un code plus rapide « par chance ».
    # En vrai : `timeit.repeat(fonction, number=1, repeat=5)` puis `min(...)`, ou `python -m timeit`.
    return min(durees)


def point_chaud(programme: Callable[[], object]) -> str:
    profileur = cProfile.Profile()
    # runcall active le profileur, appelle programme(), et le DÉSACTIVE même si programme() lève une
    # exception (elle remonte ensuite). L'équivalent à la main :
    #     profileur.enable()
    #     try:
    #         programme()
    #     finally:
    #         profileur.disable()
    profileur.runcall(programme)
    stats = pstats.Stats(profileur).stats
    # clé : (fichier, ligne, nom) ; valeur : (appels primitifs, ncalls, tottime, cumtime, appelants).
    # tottime (indice 2) : le temps passé dans le code de la fonction elle-même. C'est là qu'il faut agir.
    plus_lente = max(stats, key=lambda cle: stats[cle][2])
    return plus_lente[2]
    # En vrai, sur un script entier : `python -m cProfile -s tottime mon_script.py | head -20`, ou
    # `pstats.Stats(profileur).sort_stats("tottime").print_stats(10)` pour les 10 premières lignes.


# ============================================================================ Changer d'algorithme


def prix_en_vigueur(changements: list[tuple[date, float]], dates: list[date]) -> list[float]:
    if not changements:
        raise ValueError("l'historique des tarifs est vide : aucun prix n'est connu")
    # sorted() renvoie une COPIE triée : la liste de l'appelant reste comme elle était. Trier coûte
    # O(k log k), une seule fois ; ensuite chaque recherche coûte O(log k).
    historique = sorted(changements, key=lambda changement: changement[0])
    jours = [jour for jour, _ in historique]
    prix = [montant for _, montant in historique]
    for avant, apres in zip(jours, jours[1:]):     # triés : deux doublons sont forcément voisins
        if avant == apres:
            raise ValueError(f"deux tarifs différents entrent en vigueur le même jour : {avant}")

    resultat = []
    for jour in dates:
        # bisect_right donne la position APRÈS les éléments égaux : le nombre de changements dont la date
        # est <= jour, c'est-à-dire déjà en vigueur ce jour-là (le jour même d'un changement compris).
        # bisect_left s'arrêterait AVANT un élément égal : le jour d'un changement, on aurait l'ancien prix.
        i = bisect.bisect_right(jours, jour)
        if i == 0:
            # Sans ce test, prix[i - 1] vaudrait prix[-1] : le DERNIER tarif, sans erreur.
            raise ValueError(f"aucun tarif connu le {jour} : le premier entre en vigueur le {jours[0]}")
        resultat.append(prix[i - 1])
    return resultat
    # Complexité : O(k log k) pour trier + O(n log k) pour chercher, au lieu de O(n × k).
    # Avec pandas : `pd.merge_asof(factures.sort_values("date"), tarifs.sort_values("date"), on="date")`
    # fait exactement ça (direction="backward" par défaut : le dernier tarif <= la date).


# ============================================================================ Ne pas refaire le travail


def cache_lru(taille_max: int, duree: float, horloge: Callable[[], float] = time.monotonic) -> Callable:
    # Niveau 1 : on vérifie les paramètres tout de suite, avant même de décorer quoi que ce soit.
    if isinstance(taille_max, bool) or not isinstance(taille_max, int) or taille_max < 1:
        raise ValueError(f"taille_max doit être un entier >= 1 (reçu : {taille_max!r})")
    if isinstance(duree, bool) or not isinstance(duree, (int, float)) or not duree > 0:
        raise ValueError(f"duree doit être un nombre de secondes > 0 (reçu : {duree!r})")

    def decorateur(fonction: Callable) -> Callable:
        # Niveau 2 : l'état est créé ICI, une fois par fonction décorée. Créé au niveau 1, il serait
        # partagé par toutes les fonctions décorées avec le même `cache_lru(...)`.
        memoire: OrderedDict = OrderedDict()   # clé -> (instant du calcul, résultat), du plus ancien au plus récent
        compteurs = {"calculs": 0, "reutilisations": 0}

        @functools.wraps(fonction)
        def enveloppe(*args):
            # Niveau 3 : chaque appel.
            maintenant = horloge()
            if args in memoire:                # `in`, pas `if memoire.get(args)` : un résultat None ou 0.0 compte
                calcule_a, resultat = memoire[args]
                if maintenant - calcule_a < duree:
                    memoire.move_to_end(args)  # le plus récemment utilisé passe en dernier…
                    compteurs["reutilisations"] += 1
                    return resultat
                del memoire[args]              # périmé : on l'oublie et on recalcule
            resultat = fonction(*args)         # une exception remonte ici, AVANT d'être rangée
            compteurs["calculs"] += 1
            memoire[args] = (maintenant, resultat)  # l'âge part du calcul ; le relire ne le rajeunit pas
            if len(memoire) > taille_max:
                memoire.popitem(last=False)    # … et on oublie le premier : le moins récemment utilisé
            return resultat

        enveloppe.cache_info = lambda: {**compteurs, "taille": len(memoire)}
        return enveloppe

    return decorateur
    # Toutes les opérations d'OrderedDict utilisées ici sont en O(1) : le cache reste rapide avec
    # 100 000 entrées. Une liste `ordre` avec `ordre.remove(cle)` marcherait, mais en O(taille).
    # En vrai : `functools.lru_cache(maxsize=…)` est borné mais ne périme rien ; pour périmer, on écrit ce
    # qui précède, ou on utilise une bibliothèque (cachetools.TTLCache). Dans app.py, `@st.cache_data(ttl=300)`.


# ============================================================================ La mémoire


def totaux_par_contrat(chemin: str | Path) -> dict[str, tuple[int, float]]:
    nombres: Counter = Counter()
    sommes: dict[str, float] = {}
    # `with` referme le fichier à la sortie du bloc, même sur une erreur. newline="" laisse le module csv
    # gérer les fins de ligne (y compris un retour à la ligne DANS un champ entre guillemets).
    with open(chemin, newline="", encoding="utf-8") as fichier:
        # DictReader lit le fichier ligne par ligne (un objet fichier est un itérateur) : à tout moment, une
        # seule ligne est en mémoire. Il lit l'en-tête et donne chaque ligne sous forme de dictionnaire,
        # quel que soit l'ordre des colonnes. Il saute les lignes entièrement vides.
        for ligne in csv.DictReader(fichier):
            montant = ligne["montant"]
            if not montant:                      # facture annulée
                continue
            contrat = ligne["contrat"]
            nombres[contrat] += 1
            sommes[contrat] = sommes.get(contrat, 0.0) + float(montant)
    # On arrondit À LA FIN : arrondir à chaque ligne accumulerait les erreurs d'arrondi.
    return {contrat: (nombres[contrat], round(sommes[contrat], 2)) for contrat in nombres}
    # Avec pandas, par paquets de 10 000 lignes :
    #     for paquet in pd.read_csv(chemin, chunksize=10_000, usecols=["contrat", "montant"]):
    #         paquet = paquet.dropna(subset=["montant"])
    #         ... additionner paquet.groupby("contrat")["montant"].agg(["count", "sum"])
    # `usecols` ne lit que les colonnes utiles : encore moins de mémoire.


# ============================================================================ Optimiser sans rien changer


def clients_a_rappeler(clients: list[dict], appels: list[dict], deja_contactes: list[str],
                       seuil_appels: int = 3) -> list[tuple[str, int]]:
    # La version lente ne calcule la moyenne que si la boucle tourne au moins une fois. Sans client, elle
    # renvoie [] : on garde ce comportement (et on évite la division par zéro).
    if not clients:
        return []
    # 1. Un seul passage sur les appels : O(m) au lieu de O(n × m).
    #    Les appels de clients absents sont comptés aussi, mais personne ne les lira : c'est sans effet.
    nb_appels = Counter(appel["client_id"] for appel in appels if appel["motif"] != "information")
    # 2. La moyenne, une fois, avec EXACTEMENT le même calcul que la version lente (même somme, même ordre,
    #    même division). statistics.mean ou numpy.mean peuvent différer au dernier chiffre.
    facture_moyenne = sum(client["facture"] for client in clients) / len(clients)
    # 3. Un set, construit une fois : `in` passe de O(k) à O(1).
    exclus = set(deja_contactes)

    resultat = []
    for client in clients:
        client_id = client["client_id"]
        if client_id in exclus:
            continue
        n = nb_appels[client_id]          # un Counter renvoie 0 pour un client sans appel (sans l'ajouter)
        if n >= seuil_appels and client["facture"] >= facture_moyenne:
            resultat.append((client_id, n))
    # Le même tri que la version lente : stable, donc à égalité l'ordre des clients est conservé.
    resultat.sort(key=lambda paire: paire[1], reverse=True)
    return resultat
    # Complexité : O(n + m + k) plus le tri du résultat, au lieu de O(n × (n + m + k)). Sur 400 000
    # clients, on passe de 40 minutes à une seconde, et c'est la MÊME fonction pour le marketing.


# ============================================================================ Tests (comme dans tests/)


def test_chronometrer_garde_le_meilleur_temps_mesure_avec_l_horloge_donnee():
    horloge = HorlogeFactice()
    durees = iter([0.3, 0.1, 0.2])
    meilleur = chronometrer(lambda: horloge.avancer(next(durees)), repetitions=3, horloge=horloge)
    assert abs(meilleur - 0.1) < 1e-9          # 1000.4 - 1000.3 ne vaut pas 0.1 pile en flottants
    try:
        chronometrer(lambda: None, repetitions=0)
    except ValueError as exc:
        assert "repetitions" in str(exc)
    else:
        raise AssertionError("repetitions=0 aurait dû être refusé")


def test_le_point_chaud_est_la_fonction_qui_travaille_elle_meme():
    assert point_chaud(programme_exemple) == "calculer_scores"


def test_le_prix_change_le_jour_meme_et_jamais_avant_le_premier_tarif():
    historique = list(TARIFS_EXEMPLE)
    assert prix_en_vigueur(historique, [date(2024, 7, 1), date(2024, 6, 30), date(2030, 1, 1)]) == [32.9, 29.9, 34.9]
    assert historique == TARIFS_EXEMPLE                       # pas modifié
    for mauvais in ([date(2023, 12, 31)],):
        try:
            prix_en_vigueur(historique, mauvais)
        except ValueError as exc:
            assert "2023-12-31" in str(exc)
        else:
            raise AssertionError("une date avant le premier tarif aurait dû être refusée")


def test_le_cache_est_borne_et_perime_ses_resultats():
    horloge = HorlogeFactice()
    calculs = []

    @cache_lru(taille_max=2, duree=60, horloge=horloge)
    def score(client_id):
        calculs.append(client_id)
        return None                                   # None se met en cache aussi

    for client_id in ("C1", "C1", "C2", "C1", "C3", "C1"):
        score(client_id)
    assert calculs == ["C1", "C2", "C3"]              # C2, le moins récemment utilisé, a été oublié
    assert score.cache_info() == {"calculs": 3, "reutilisations": 3, "taille": 2}
    horloge.avancer(60)
    score("C1")
    assert calculs[-1] == "C1"                        # périmé 60 s après son calcul, même relu entre-temps


def test_les_totaux_ne_dependent_ni_de_l_ordre_des_colonnes_ni_des_virgules():
    import tempfile

    with tempfile.TemporaryDirectory() as dossier:
        a = ecrire_factures(Path(dossier) / "a.csv", lignes=500, graine=3)
        b = ecrire_factures(Path(dossier) / "b.csv", lignes=500, graine=3,
                            colonnes=("montant", "offre", "contrat", "client_id"))
        assert totaux_par_contrat(a) == totaux_par_contrat(b)
        assert sum(n for n, _ in totaux_par_contrat(a).values()) < 500   # les factures annulées ne comptent pas


def test_la_version_rapide_donne_exactement_le_resultat_de_la_lente():
    # LE test d'une optimisation : l'ancienne version sert de référence, sur des entrées tirées au hasard
    # (graine fixe : un échec se rejoue à l'identique).
    rng = random.Random(0)
    for _ in range(200):
        clients, appels, deja = exemple_rappels(n_clients=rng.randint(0, 12), graine=rng.randrange(10**6))
        seuil = rng.randint(0, 4)
        assert clients_a_rappeler(clients, appels, deja, seuil) == clients_a_rappeler_lent(clients, appels, deja, seuil)

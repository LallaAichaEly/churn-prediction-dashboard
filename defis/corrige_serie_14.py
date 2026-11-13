"""Série 14 — CORRIGÉ COMMENTÉ.

🙈 Ne lisez ce fichier qu'APRÈS avoir essayé chaque défi dans `defis/serie_14.py`.

Si votre code passe les vérifications, il est bon, même s'il ne ressemble pas à celui-ci.
Comparez, et notez dans JOURNAL.md la panne à laquelle vous n'aviez pas pensé.

Fil rouge de la série : tout ce qui vient de l'environnement (les réglages, le journal, le temps qui
passe, l'attente, le disque) est REÇU en paramètre ou vérifié, jamais supposé. C'est ce qui permet
au correcteur de simuler en une fraction de seconde une nuit entière de pannes, et à vous de
l'écrire dans un test.

À la fin du fichier, des tests comme on les écrirait dans `tests/test_*.py`.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

from defis.production import (  # noqa: F401
    FAUX,
    PARIS,
    TRANSITOIRES,
    VRAIS,
    CircuitOuvert,
    Config,
    ErreurTemporaire,
    HorlogeFactice,
    exemple_lot,
    journal_console,
    score_exemple,
    service_capricieux,
)

# ============================================================================ Défi 86 : configuration


def _valeur(env: Mapping[str, str], nom: str) -> str | None:
    # Une variable vide (« CHURN_SEUIL= » dans un fichier docker-compose) compte comme absente.
    texte = env.get(nom)
    if texte is None:
        return None
    texte = str(texte).strip()
    return texte or None


def lire_config(env: Mapping[str, str]) -> Config:
    # On COLLECTE les erreurs au lieu de s'arrêter à la première : le message final dit tout ce
    # qu'il faut corriger, en un seul redéploiement.
    erreurs: list[str] = []

    cle_api = _valeur(env, "CHURN_CLE_API")
    if cle_api is None:
        erreurs.append("CHURN_CLE_API : obligatoire")  # jamais la valeur de la clé dans un message

    seuil, texte = 0.5, _valeur(env, "CHURN_SEUIL")
    if texte is not None:
        try:
            seuil = float(texte)
        except ValueError:
            seuil = float("nan")
        # `not 0 <= seuil <= 1` refuse aussi NaN : toute comparaison avec NaN est fausse.
        # `seuil < 0 or seuil > 1`, lui, laisserait passer NaN (les deux comparaisons sont fausses).
        if not 0 <= seuil <= 1:
            erreurs.append(f"CHURN_SEUIL : nombre entre 0 et 1 attendu (avec un point), reçu {texte!r}")

    taille_lot, texte = 500, _valeur(env, "CHURN_TAILLE_LOT")
    if texte is not None:
        try:
            taille_lot = int(texte)  # int("2.5") lève une ValueError : tant mieux, on ne devine pas
        except ValueError:
            taille_lot = 0
        if taille_lot < 1:
            erreurs.append(f"CHURN_TAILLE_LOT : entier supérieur ou égal à 1 attendu, reçu {texte!r}")

    mode_test, texte = False, _valeur(env, "CHURN_MODE_TEST")
    if texte is not None:
        # Surtout pas bool(texte) : bool("false") vaut True, comme tout texte non vide.
        if texte.lower() in VRAIS:
            mode_test = True
        elif texte.lower() not in FAUX:
            erreurs.append(f"CHURN_MODE_TEST : une valeur parmi {VRAIS + FAUX} attendue, reçu {texte!r}")

    if erreurs:
        raise ValueError("Configuration invalide : " + " ; ".join(erreurs))
    return Config(cle_api=cle_api, seuil=seuil, taille_lot=taille_lot, mode_test=mode_test)


# En production : `config = lire_config(os.environ)`, UNE fois, au démarrage, et l'on passe `config`
# aux fonctions qui en ont besoin. Une fonction qui lit `os.environ` au milieu du code est difficile
# à tester (il faut modifier l'environnement de tout le processus) et peut changer d'avis en route.
# Pour aller plus loin : signaler aussi les variables CHURN_* inconnues (« CHURN_SEUILL »), car une
# faute de frappe dans un nom de variable fait silencieusement prendre la valeur par défaut.


# ============================================================================ Défi 87 : journalisation


def scorer_lot(clients: Iterable[dict], score: Callable[[dict], float], journal: logging.Logger) -> dict[str, float]:
    scores: dict[str, float] = {}
    ignores = erreurs = 0
    for client in clients:
        identifiant = client.get("id")
        facture = client.get("facture_mensuelle")
        if facture is None or facture < 0:  # pas `if not facture` : une facture de 0 € est valide
            # Les valeurs passent en ARGUMENTS (%s, %r) : logging ne fabrique le texte que si le
            # message est réellement écrit, et les outils de supervision regroupent les messages
            # qui ont le même gabarit. On journalise l'identifiant, pas `client` (son e-mail).
            journal.warning("client %s ignoré : facture mensuelle invalide (%r)", identifiant, facture)
            ignores += 1
            continue
        try:
            scores[identifiant] = score(client)
        except Exception:  # pas `except:` : KeyboardInterrupt (Ctrl+C) et SystemExit doivent passer
            # `exception` = niveau ERROR + la traceback complète. Sans traceback, on saurait QUEL
            # client a planté, mais pas OÙ ni POURQUOI.
            journal.exception("client %s : le modèle a échoué", identifiant)
            erreurs += 1
        else:
            journal.debug("client %s noté", identifiant)
    journal.info("lot terminé : %d clients notés, %d ignorés, %d en erreur", len(scores), ignores, erreurs)
    return scores


# Et la configuration ? Elle se fait UNE fois, dans le programme principal, par exemple :
#
#     if __name__ == "__main__":
#         logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
#         scorer_lot(clients, modele, logging.getLogger("lot_de_nuit"))
#
# Dans un module, on écrit en général `journal = logging.getLogger(__name__)` en haut du fichier.
# Ici le journal est un paramètre, pour que le correcteur (et vos tests) puissent lire ce qui y est écrit.


# ============================================================================ Défi 88 : reprises


def appeler_avec_reprises(appel: Callable[[], object], *, tentatives: int = 4, base: float = 0.5,
                          plafond: float = 8.0, dormir: Callable[[float], None] = time.sleep):
    if tentatives < 1:
        raise ValueError(f"tentatives doit valoir au moins 1, pas {tentatives}")
    for numero in range(tentatives):
        try:
            return appel()
        except TRANSITOIRES as erreur:  # seulement les pannes passagères ; le reste remonte tout seul
            if numero == tentatives - 1:
                raise  # la DERNIÈRE erreur, avec sa traceback : l'appelant sait ce qui s'est passé
            attente = getattr(erreur, "attente", None)
            delai = attente if attente is not None else base * 2 ** numero  # `is not None` : 0 est une réponse
            dormir(min(plafond, delai))
    raise AssertionError("inaccessible : la boucle renvoie ou relance toujours")


# En production, on ajoute du hasard au délai (« jitter ») : `dormir(random.uniform(0, delai))`.
# Sinon, après une coupure, les mille clients d'une API réessaient tous à la même seconde, puis
# tous ensemble une seconde plus tard… et font retomber le service qu'ils attendent. Le correcteur
# ne le demande pas, pour pouvoir vérifier les délais exacts. La bibliothèque `tenacity` fait tout
# cela avec un décorateur : `@retry(wait=wait_random_exponential(...), stop=stop_after_attempt(4))`.


# ============================================================================ Défi 89 : fuseaux horaires


def prochain_envoi(maintenant: datetime) -> datetime:
    if maintenant.tzinfo is None or maintenant.utcoffset() is None:
        raise ValueError(f"{maintenant!r} n'a pas de fuseau horaire : impossible de savoir de quelle heure il s'agit")
    # On CONVERTIT vers Paris (astimezone), on n'étiquette pas (replace). Les règles humaines
    # (« 10 h », « du lundi au vendredi ») se lisent en heure locale…
    jour = maintenant.astimezone(PARIS).date()
    while True:
        # … et c'est zoneinfo qui choisit le bon décalage (+1 h ou +2 h) pour CE jour-là. On
        # fabrique chaque candidat à partir de sa date, au lieu d'ajouter 24 h à un instant UTC.
        candidat = datetime(jour.year, jour.month, jour.day, 10, tzinfo=PARIS)
        if candidat > maintenant and candidat.weekday() < 5:
            return candidat.astimezone(timezone.utc)  # … et l'on stocke, compare et échange en UTC
        jour += timedelta(days=1)


# La boucle fait au plus 4 tours (vendredi après 10 h → samedi → dimanche → lundi). Les jours fériés
# s'ajouteraient de la même façon (`or jour in FERIES`). Comparer `candidat > maintenant` est sûr :
# deux dates conscientes de fuseaux différents se comparent par instant réel.


# ============================================================================ Défi 90 : écriture atomique


def ecrire_atomique(chemin: Path, lignes: Iterable[str]) -> None:
    chemin = Path(chemin)
    # mkstemp crée le fichier avec un nom UNIQUE (jamais celui d'un autre traitement), dans le
    # MÊME dossier : os.replace n'est atomique qu'au sein d'un même système de fichiers.
    descripteur, nom = tempfile.mkstemp(dir=chemin.parent, prefix=f".{chemin.name}.", suffix=".tmp")
    temporaire = Path(nom)
    try:
        with os.fdopen(descripteur, "w", encoding="utf-8", newline="\n") as fichier:
            for ligne in lignes:
                fichier.write(f"{ligne}\n")
            fichier.flush()
            os.fsync(fichier.fileno())  # les octets sont sur le disque, pas seulement en mémoire
        # Le fichier est FERMÉ : tout est écrit. Le renommage remplace l'ancien fichier d'un coup :
        # un lecteur voit soit l'ancien, soit le nouveau, jamais un mélange.
        os.replace(temporaire, chemin)
    except BaseException:  # même Ctrl+C : on nettoie, puis on laisse l'erreur continuer son chemin
        temporaire.unlink(missing_ok=True)
        raise


# Détail de pro : mkstemp crée le fichier avec les droits 600 (lisible par vous seul·e). Si le
# tableau de bord tourne sous un autre utilisateur, recopiez les droits de l'ancien fichier avant
# le renommage (`os.chmod(temporaire, chemin.stat().st_mode)`). pandas, joblib, SQLite : le même
# schéma « temporaire puis os.replace » s'applique à tout fichier qu'un autre programme lit.


# ============================================================================ Défi 91 : disjoncteur


class Disjoncteur:
    FERME, OUVERT, SEMI_OUVERT = "ferme", "ouvert", "semi_ouvert"

    def __init__(self, seuil_echecs: int = 3, delai: float = 30.0, horloge: Callable[[], float] = time.monotonic) -> None:
        # Tout l'état est dans des attributs d'OBJET créés ici : deux disjoncteurs ne partagent rien.
        self.seuil_echecs = seuil_echecs
        self.delai = delai
        self._horloge = horloge  # time.monotonic, pas time.time : l'heure système peut reculer (NTP)
        self._echecs_d_affilee = 0
        self._ouvert_depuis: float | None = None

    @property
    def etat(self) -> str:
        # L'état n'est pas stocké, il se CALCULE : impossible d'oublier de le mettre à jour quand
        # le temps passe.
        if self._ouvert_depuis is None:
            return self.FERME
        if self._horloge() - self._ouvert_depuis >= self.delai:
            return self.SEMI_OUVERT
        return self.OUVERT

    def appeler(self, fonction: Callable, *args, **kwargs):
        etat = self.etat  # lu UNE fois, avant l'appel
        if etat == self.OUVERT:
            reste = self.delai - (self._horloge() - self._ouvert_depuis)
            # `attente` permet à appeler_avec_reprises (défi 88) d'attendre exactement le bon temps.
            raise CircuitOuvert(f"service coupé par le disjoncteur, nouvel essai dans {reste:.1f} s", attente=reste)
        try:
            resultat = fonction(*args, **kwargs)
        except TRANSITOIRES:
            self._echecs_d_affilee += 1
            if etat == self.SEMI_OUVERT or self._echecs_d_affilee >= self.seuil_echecs:
                self._ouvert_depuis = self._horloge()  # (ré)ouverture : le délai repart de zéro
            raise
        # Les autres erreurs ont traversé le `try` sans rien changer. Ici, c'est un succès.
        self._echecs_d_affilee = 0
        self._ouvert_depuis = None
        return resultat


# Les deux mécanismes se combinent : le disjoncteur À L'INTÉRIEUR des reprises.
#
#     appeler_avec_reprises(lambda: sms.appeler(envoyer, numero, texte))
#
# Tant que le service répond par intermittence, les reprises suffisent. S'il tombe pour de bon, le
# disjoncteur s'ouvre, et les reprises suivantes échouent en une microseconde (CircuitOuvert) au
# lieu d'attendre l'opérateur à chaque client. Le lot finit vite, et le service souffle.


# ============================================================================ Dans un vrai fichier de tests
#
# Chaque test simule une panne en injectant un faux : une fausse horloge, une fausse attente, un
# faux journal, un générateur qui plante. Aucun ne dure plus d'une milliseconde.


def test_la_configuration_signale_toutes_les_erreurs_d_un_coup():
    assert lire_config({"CHURN_CLE_API": " cle ", "CHURN_MODE_TEST": "Oui"}) == Config("cle", mode_test=True)
    try:
        lire_config({"CHURN_SEUIL": "nan", "CHURN_MODE_TEST": "false!"})
    except ValueError as erreur:
        assert all(nom in str(erreur) for nom in ("CHURN_CLE_API", "CHURN_SEUIL", "CHURN_MODE_TEST"))
    else:
        raise AssertionError("une configuration invalide aurait dû être refusée")


def test_le_lot_continue_apres_une_erreur_et_la_journalise():
    class Memoire(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record)

    journal, memoire = logging.getLogger("test_lot_de_nuit"), Memoire()
    journal.addHandler(memoire)
    journal.propagate = False
    try:
        assert scorer_lot(exemple_lot(), score_exemple, journal) == {"C001": 1.0, "C003": 0.0, "C006": 0.15}
    finally:
        journal.removeHandler(memoire)
    erreurs = [m for m in memoire.messages if m.levelno == logging.ERROR]
    assert len(erreurs) == 1 and "C005" in erreurs[0].getMessage() and erreurs[0].exc_info is not None


def test_les_reprises_attendent_de_plus_en_plus_puis_abandonnent():
    attentes = []
    assert appeler_avec_reprises(service_capricieux(echecs=3), dormir=attentes.append) == "SMS envoyé"
    assert attentes == [0.5, 1.0, 2.0]
    attentes.clear()
    try:
        appeler_avec_reprises(service_capricieux(echecs=9), tentatives=3, dormir=attentes.append)
    except ErreurTemporaire:
        assert attentes == [0.5, 1.0]  # pas d'attente après le dernier échec
    else:
        raise AssertionError("l'erreur aurait dû remonter")


def test_l_envoi_suit_l_heure_de_paris_au_changement_d_heure():
    vendredi = datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc)
    assert prochain_envoi(vendredi) == datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc)
    assert prochain_envoi(datetime(2026, 10, 21, 7, 0, tzinfo=timezone.utc)).hour == 8


def test_un_plantage_ne_laisse_ni_fichier_coupe_ni_temporaire():
    def lignes_qui_plantent():
        yield "id;score"
        raise RuntimeError("disque plein")

    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "scores.csv"
        ecrire_atomique(chemin, ["id;score", "C001;0.82"])
        try:
            ecrire_atomique(chemin, lignes_qui_plantent())
        except RuntimeError:
            pass
        assert chemin.read_text(encoding="utf-8") == "id;score\nC001;0.82\n"
        assert os.listdir(dossier) == ["scores.csv"]


def test_le_disjoncteur_s_ouvre_puis_laisse_passer_un_essai():
    horloge = HorlogeFactice()
    sms = Disjoncteur(seuil_echecs=2, delai=30, horloge=horloge)
    en_panne = service_capricieux(echecs=2)
    for _ in range(2):
        try:
            sms.appeler(en_panne)
        except ErreurTemporaire:
            pass
    assert sms.etat == "ouvert"
    horloge.avancer(30)
    assert sms.etat == "semi_ouvert"
    assert sms.appeler(en_panne) == "SMS envoyé" and sms.etat == "ferme"


def test_les_reprises_et_le_disjoncteur_ensemble():
    horloge = HorlogeFactice()
    sms = Disjoncteur(seuil_echecs=2, delai=30, horloge=horloge)
    en_panne, attentes = service_capricieux(echecs=2), []
    resultat = appeler_avec_reprises(lambda: sms.appeler(en_panne), tentatives=5,
                                     dormir=lambda s: (attentes.append(s), horloge.avancer(s)), plafond=60)
    # Échec (0,5 s), échec qui ouvre le disjoncteur (1 s), CircuitOuvert qui demande les 29 s restantes,
    # puis l'essai du disjoncteur semi-ouvert réussit : 4 appels, dont un seul refusé sans déranger le service.
    assert resultat == "SMS envoyé" and sms.etat == "ferme"
    assert attentes == [0.5, 1.0, 29.0]

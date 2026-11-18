"""Les exports d'exemple du projet 6 : `projets/exemple_exports/` vaut `generer_exports(...)`.

Le générateur simule dix mois de la vie d'un opérateur (janvier à octobre 2026) : des clients arrivent,
changent de moyen de paiement, appellent le support, résilient avec un préavis. Le CRM et le support
déposent chacun un fichier par mois, avec les défauts qu'on rencontre en vrai : un compte de test, une
ligne recopiée à moitié, une date impossible, un export du support coupé au milieu d'une ligne.

Les clients actifs le 30 juin 2026 suivent la « vérité cachée » du projet 1 (`src/churn/data.py`) : la
table exportée à cette date (`exporter --photo 2026-06-30`) s'entraîne donc comme `data/customers.csv`.

Le hasard vient du module `random` de la bibliothèque standard, dont les tirages ne changent pas d'une
version de Python à l'autre, pour que les fichiers soient reproductibles octet par octet.
"""

from __future__ import annotations

import calendar
import json
import math
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path

#: La date de la photo de l'exemple, et les jours où le CRM et le support déposent leurs exports.
PHOTO_EXEMPLE = date(2026, 6, 30)
DATES_EXPORTS = tuple(date.fromisoformat(d) for d in (
    "2026-01-05", "2026-02-02", "2026-03-02", "2026-04-01", "2026-05-04",
    "2026-06-01", "2026-07-01", "2026-08-03", "2026-09-01", "2026-10-01"))

_CONTRATS = (("Mensuel", 0.55, 14.0), ("Annuel", 0.25, 30.0), ("Biannuel", 0.20, 45.0))
_PAIEMENTS = (("Prélèvement", 0.35), ("Carte bancaire", 0.30), ("Virement", 0.20), ("Chèque", 0.15))
_INTERNET = (("Fibre", 0.45, 80.0), ("ADSL", 0.35, 55.0), ("Aucun", 0.20, 25.0))
_ENTETE = "client;debut;fin;contrat;paiement;internet;assistance;senior;produits;mensualite;maj"


def _ajouter_mois(jour: date, mois: int) -> date:
    total = jour.year * 12 + jour.month - 1 + mois
    annee, numero = divmod(total, 12)
    return date(annee, numero + 1, min(jour.day, calendar.monthrange(annee, numero + 1)[1]))


def _poisson(rng: random.Random, moyenne: float) -> int:
    """Tirage de Poisson (méthode de Knuth), avec seulement `rng.random()`."""
    limite, k, produit = math.exp(-moyenne), 0, rng.random()
    while produit > limite:
        k += 1
        produit *= rng.random()
    return k


def _choisir(rng: random.Random, options) -> tuple:
    return rng.choices(options, weights=[o[1] for o in options])[0]


def _instant(rng: random.Random, jour: date) -> datetime:
    return datetime.combine(jour, time(rng.randint(8, 19), rng.randint(0, 59)))


def _client(rng: random.Random, debut: date, fin: date | None, preavis: int) -> dict:
    contrat, _, _ = _choisir(rng, _CONTRATS)
    internet, _, base = _choisir(rng, _INTERNET)
    return {
        "debut": debut, "fin": fin, "contrat": contrat, "paiement": _choisir(rng, _PAIEMENTS)[0],
        "internet": internet, "assistance": "Oui" if rng.random() < 0.35 else "Non",
        "senior": "Oui" if rng.random() < 0.18 else "Non", "produits": rng.randint(1, 4),
        "mensualite": round(min(130.0, max(15.0, base + rng.gauss(0, 12))), 2),
        "cree": _instant(rng, debut), "preavis": preavis, "changement": None,
    }


def _logit(c: dict, appels: int) -> float:
    """La vérité cachée du projet 1 (`src/churn/data.py`, `_churn_logit`), sans le bruit."""
    anciennete = (PHOTO_EXEMPLE.year - c["debut"].year) * 12 + PHOTO_EXEMPLE.month - c["debut"].month
    return (-1.2 + 1.6 * (c["contrat"] == "Mensuel") - 0.9 * (c["contrat"] == "Biannuel") - 0.035 * anciennete
            + 0.018 * (c["mensualite"] - 65) + 0.45 * appels + 0.7 * (c["paiement"] == "Chèque")
            + 0.5 * (c["internet"] == "Fibre") - 0.8 * (c["assistance"] == "Oui") - 0.25 * (c["produits"] - 1)
            + 0.3 * (c["senior"] == "Oui"))


def _simuler(rng: random.Random, clients: int) -> tuple[list[dict], list[dict]]:
    """Les clients (avec leur histoire) et les tickets (avec leurs versions successives)."""
    photo, dernier = PHOTO_EXEMPLE, DATES_EXPORTS[-1]
    fiches, tickets = [], []

    def ouvrir(fiche, jour: date) -> None:
        tickets.append({"fiche": fiche, "ouvert": _instant(rng, jour)})

    # 1. Les clients actifs le jour de la photo : ce sont les lignes de la table.
    for _ in range(clients):
        contrat, _, echelle = _choisir(rng, _CONTRATS)
        anciennete = int(min(72, max(1, round(rng.expovariate(1 / echelle)))))
        fiche = _client(rng, _ajouter_mois(photo, -anciennete), None, rng.randint(0, 30))
        fiche["contrat"] = contrat
        appels = _poisson(rng, 1.3)
        part = rng.random() < 1 / (1 + math.exp(-(_logit(fiche, appels) + rng.gauss(0, 0.6))))
        if part:
            fiche["fin"] = photo + timedelta(days=rng.randint(1, 90))
        elif rng.random() < 0.06:
            fiche["fin"] = photo + timedelta(days=rng.randint(91, 150))  # part, mais après l'horizon
        for _ in range(appels):
            ouvrir(fiche, photo - timedelta(days=rng.randint(0, 364)))
        for _ in range(_poisson(rng, 0.4)):  # trop anciens pour compter
            ouvrir(fiche, photo - timedelta(days=rng.randint(365, 700)))
        fin_appels = min(fiche["fin"] or dernier, dernier) - timedelta(days=1)
        if fin_appels > photo:
            for _ in range(_poisson(rng, 1.0 if part else 0.25)):  # APRÈS la photo : ne doivent pas compter
                ouvrir(fiche, photo + timedelta(days=rng.randint(1, (fin_appels - photo).days)))
        if rng.random() < 0.10:  # un changement de moyen de paiement, avant la photo
            quand = fiche["debut"] + timedelta(days=rng.randint(1, max(1, (photo - fiche["debut"]).days - 1)))
            if quand < photo:
                ancien = rng.choice([p for p, _ in _PAIEMENTS if p != fiche["paiement"]])
                fiche["changement"] = (_instant(rng, quand), ancien)
        fiches.append(fiche)

    # 2. Des clients partis avant la photo (dont quelques-uns le jour même) : hors de la table.
    for k in range(clients // 5):
        fin = photo if k < 3 else date(2025, 3, 1) + timedelta(days=rng.randint(0, (photo - date(2025, 3, 1)).days))
        fiche = _client(rng, _ajouter_mois(fin, -rng.randint(1, 60)), fin, rng.randint(0, 30))
        for _ in range(_poisson(rng, 1.5)):
            ouvrir(fiche, fin - timedelta(days=rng.randint(0, 300)))
        fiches.append(fiche)

    # 3. Des clients arrivés après la photo : hors de la table eux aussi.
    for _ in range(clients // 12):
        fiche = _client(rng, photo + timedelta(days=rng.randint(1, (dernier - photo).days - 2)), None, 0)
        if rng.random() < 0.3:
            ouvrir(fiche, fiche["debut"] + timedelta(days=1))
        fiches.append(fiche)

    # Les identifiants suivent l'ordre d'arrivée, comme dans un vrai CRM.
    fiches.sort(key=lambda f: (f["cree"], f["mensualite"]))
    for numero, fiche in enumerate(fiches, start=1):
        fiche["client"] = f"C{numero:05d}"

    # 4. Quelques tickets de prospects, inconnus du CRM.
    for k in range(5):
        tickets.append({"fiche": {"client": f"C{90001 + k:05d}"},
                        "ouvert": _instant(rng, photo - timedelta(days=rng.randint(0, 200)))})

    tickets = [t for t in tickets if t["ouvert"].date() < dernier]
    tickets.sort(key=lambda t: (t["ouvert"], t["fiche"]["client"]))
    for numero, ticket in enumerate(tickets, start=1):
        ticket["ticket"] = f"T{numero:06d}"
        versions = [(ticket["ouvert"], "ouvert")]
        if rng.random() < 0.6:
            versions.append((versions[-1][0] + timedelta(hours=rng.randint(1, 240)), "résolu"))
            if rng.random() < 0.3:
                versions.append((versions[-1][0] + timedelta(hours=rng.randint(1, 240)), "clos"))
        ticket["versions"] = versions
    return fiches, tickets


def _etat(fiche: dict, avant: datetime) -> dict | None:
    """Ce que le CRM sait du contrat juste avant `avant` (None s'il n'existe pas encore)."""
    if fiche["cree"] >= avant:
        return None
    evenements = [fiche["cree"]]
    paiement = fiche["paiement"]
    if fiche["changement"]:
        quand, ancien = fiche["changement"]
        if quand < avant:
            evenements.append(quand)
        else:
            paiement = ancien
    fin = None
    if fiche["fin"] is not None:
        annonce = datetime.combine(max(fiche["debut"], fiche["fin"] - timedelta(days=fiche["preavis"])),
                                   fiche["cree"].time()) + timedelta(minutes=7)
        if annonce < avant:
            fin = fiche["fin"]
            evenements.append(annonce)
    return {**fiche, "paiement": paiement, "fin": fin, "maj": max(evenements)}


def _ligne_crm(etat: dict) -> str:
    return ";".join([
        etat["client"], etat["debut"].strftime("%d/%m/%Y"), etat["fin"].strftime("%d/%m/%Y") if etat["fin"] else "",
        etat["contrat"], etat["paiement"], etat["internet"], etat["assistance"], etat["senior"], str(etat["produits"]),
        f"{etat['mensualite']:.2f}".replace(".", ","), etat["maj"].isoformat(timespec="seconds"),
    ])


def _ligne_ticket(ticket: dict, version: tuple) -> str:
    maj, statut = version
    return json.dumps({"ticket": ticket["ticket"], "client": ticket["fiche"]["client"],
                       "ouvert": ticket["ouvert"].isoformat(timespec="seconds"),
                       "maj": maj.isoformat(timespec="seconds"), "statut": statut}, ensure_ascii=False)


def _abimer(rng: random.Random, lignes: list[str], n: int, defauts) -> list[str]:
    """Glisse `n` lignes abîmées (des copies de lignes existantes, abîmées) : aucune donnée n'est perdue."""
    lignes = list(lignes)
    for k in range(n):
        modele = lignes[rng.randrange(len(lignes))]
        lignes.insert(rng.randrange(len(lignes) + 1), defauts[k % len(defauts)](modele))
    return lignes


def _remplacer_champ(numero: int, valeur: str):
    def abimer(ligne: str) -> str:
        champs = ligne.split(";")
        champs[numero] = valeur
        return ";".join(champs)
    return abimer


def generer_exports(dossier: str | Path, graine: int = 2026, clients: int = 800) -> list[Path]:
    """Écrit les 20 exports (10 du CRM, 10 du support) dans `dossier` et renvoie leurs chemins."""
    rng = random.Random(graine)
    fiches, tickets = _simuler(rng, clients)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    ecrits: list[Path] = []
    precedent = None
    for numero, jour in enumerate(DATES_EXPORTS):
        avant = datetime.combine(jour, time(0, 0))
        debut_fenetre = datetime.combine(precedent, time(0, 0)) if precedent else None

        # --- le CRM : tout au premier export, puis les contrats créés ou modifiés depuis le précédent.
        lignes = []
        for fiche in fiches:
            etat = _etat(fiche, avant)
            if etat is not None and (debut_fenetre is None or etat["maj"] >= debut_fenetre):
                lignes.append(_ligne_crm(etat))
        if jour == date(2026, 4, 1):
            lignes = _abimer(rng, lignes, 3, [
                lambda ligne: "TEST01" + ligne[6:],                  # un compte de test oublié
                _remplacer_champ(9, "N/A"),                          # une mensualité manquante
                lambda ligne: ligne.rsplit(";", 1)[0],               # une ligne coupée : 10 champs
            ])
        if jour == date(2026, 7, 1):
            lignes = _abimer(rng, lignes, 2, [_remplacer_champ(1, "31/06/2026"), _remplacer_champ(3, "Mensual")])
        if jour == date(2026, 9, 1):
            lignes = [ligne.replace(";Mensuel;", "; Mensuel ;", 1) if k % 7 == 3 else ligne
                      for k, ligne in enumerate(lignes)]               # des espaces autour (acceptés)
        texte = _ENTETE + "\n" + "\n".join(lignes) + "\n" + ("\n" if numero % 3 == 0 else "")
        chemin = dossier / f"crm_{jour.isoformat()}.csv"
        chemin.write_bytes(b"\xef\xbb\xbf" + texte.encode("utf-8"))  # la BOM qu'Excel écrit
        ecrits.append(chemin)

        # --- le support : l'état de chaque ticket au premier export, puis chaque mise à jour.
        lignes = []
        for ticket in tickets:
            versions = [v for v in ticket["versions"] if v[0] < avant]
            if debut_fenetre is None:
                versions = versions[-1:]
            else:
                versions = [v for v in versions if v[0] >= debut_fenetre]
            lignes += [_ligne_ticket(ticket, v) for v in versions]
        if jour == date(2026, 3, 2):
            lignes.append(lignes[len(lignes) // 2][:57])  # l'export a été coupé au milieu d'une ligne
        if jour == date(2026, 8, 3):
            lignes = _abimer(rng, lignes, 2, [
                lambda ligne: ligne.replace('"ticket": "T', '"ticket": "TK', 1),   # un identifiant mal formé
                lambda ligne: ligne.replace('"ouvert": "', '"ouvert": "le ', 1),     # une date illisible
            ])
        chemin = dossier / f"support_{jour.isoformat()}.jsonl"
        chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")
        ecrits.append(chemin)
        precedent = jour
    return ecrits

"""Outils de la série 27, partagés par votre fichier, le corrigé et le correcteur (ne le modifiez pas).

Le fichier clients mélange trois sources : le site, les boutiques et le support. La même personne y
apparaît souvent deux ou trois fois, écrite différemment. Ce module fournit :

- `fiches_exemple()` : 14 fiches qui ne représentent que 8 clients (`VERITE_EXEMPLE` dit lesquels) ;
- trois normalisations (`normaliser_nom`, `normaliser_email`, `normaliser_telephone`) : une valeur
  absente (None, NaN, texte vide) devient toujours "" ;
- trois clés de blocage (`CLES`) pour le défi 166, et le seuil de décision `SEUIL` ;
- `generer_fiches(n_clients)` : des milliers de fiches avec leur vérité, pour mesurer un dédoublonnage.

Chaque appel renvoie de NOUVELLES données : modifiez-les autant que vous voulez.

    >>> from defis.serie_27 import *
    >>> fiches = fiches_exemple()
    >>> len(fiches), len(VERITE_EXEMPLE)
    (14, 8)
    >>> fiches["F02"]["nom"], fiches["F02"]["telephone"]
    ('DUPRE Helene', '+33 6 12 34 56 78')
    >>> normaliser_nom(fiches["F02"]["nom"]), normaliser_telephone(fiches["F02"]["telephone"])
    ('dupre helene', '0612345678')
"""

from __future__ import annotations

import random
import unicodedata
from datetime import date, timedelta

import pandas as pd

SEUIL = 5
"""Deux fiches dont le score de correspondance (défi 167) atteint 5 sont considérées comme le même client."""


# ---------------------------------------------------------------------------- les valeurs absentes


def _absent(valeur) -> bool:
    """None, NaN (pandas, NumPy), NaT, pd.NA ou un texte vide : la valeur n'est pas connue."""
    if valeur is None:
        return True
    if isinstance(valeur, str):
        return not valeur.strip()
    try:
        return bool(pd.isna(valeur))
    except (TypeError, ValueError):  # une liste, un tableau… : ce n'est pas « une valeur absente »
        return False


# ---------------------------------------------------------------------------- les normalisations


def normaliser_nom(texte) -> str:
    """Le nom sans accents, en minuscules, sans tirets ni apostrophes, un seul espace entre les mots.

        >>> normaliser_nom("  DUPRÉ-Lefèvre   Hélène ")
        'dupre lefevre helene'
        >>> normaliser_nom("N’Diaye  Aïssatou"), normaliser_nom(None), normaliser_nom(float("nan"))
        ('n diaye aissatou', '', '')

    L'ordre des mots est GARDÉ : « Dupré Hélène » et « Hélène Dupré » restent différents (défi 167).
    """
    if _absent(texte):
        return ""
    t = unicodedata.normalize("NFKD", str(texte))
    t = "".join(c for c in t if not unicodedata.combining(c)).casefold()
    for signe in "-'’.,":
        t = t.replace(signe, " ")
    return " ".join(t.split())


def normaliser_email(texte) -> str:
    """L'adresse sans espaces autour, en minuscules ("" si elle manque).

        >>> normaliser_email(" Helene.Dupre@Exemple.fr "), normaliser_email(None)
        ('helene.dupre@exemple.fr', '')
    """
    return "" if _absent(texte) else str(texte).strip().lower()


def normaliser_telephone(texte) -> str:
    """Un numéro français sur 10 chiffres (« 0612345678 »), ou "" s'il manque ou ne se lit pas.

        >>> [normaliser_telephone(t) for t in ("06 12 34 56 78", "+33 6 12 34 56 78", "+33 (0)6 12 34 56 78",
        ...                                    "0033612345678", "06.12.34.56.78", "6 12 34 56 78", 612345678.0)]
        ['0612345678', '0612345678', '0612345678', '0612345678', '0612345678', '0612345678', '0612345678']
        >>> normaliser_telephone("06 12 34"), normaliser_telephone(None), normaliser_telephone("inconnu")
        ('', '', '')

    « 6 12 34 56 78 » (9 chiffres) : Excel a mangé le 0 du début (la série 07 en parle), on le remet.
    Un numéro illisible devient "" : ce n'est PAS un désaccord, c'est une valeur inconnue.
    """
    if _absent(texte):
        return ""
    if isinstance(texte, float) and texte.is_integer():
        texte = int(texte)  # 612345678.0, lu par Excel comme un nombre
    t = str(texte).strip().replace("(0)", "")
    chiffres = "".join(c for c in t if c in "0123456789")
    if t.startswith("+33"):
        chiffres = "0" + chiffres[2:]
    elif chiffres.startswith("0033"):
        chiffres = "0" + chiffres[4:]
    elif len(chiffres) == 9 and not chiffres.startswith("0"):
        chiffres = "0" + chiffres
    return chiffres if len(chiffres) == 10 and chiffres.startswith("0") else ""


# ---------------------------------------------------------------------------- les clés de blocage (défi 166)


def cle_email(fiche: dict):
    """L'e-mail normalisé, ou None s'il manque.

        >>> cle_email({"email": " K.Benali@Exemple.fr "}), cle_email({"email": ""}), cle_email({})
        ('k.benali@exemple.fr', None, None)
    """
    return normaliser_email(fiche.get("email")) or None


def cle_telephone(fiche: dict):
    """Le téléphone normalisé, ou None s'il manque ou ne se lit pas.

        >>> cle_telephone({"telephone": "+33 6 12 34 56 78"}), cle_telephone({"telephone": "06 12"})
        ('0612345678', None)
    """
    return normaliser_telephone(fiche.get("telephone")) or None


def cle_nom_code_postal(fiche: dict):
    """Les mots du nom normalisé, triés, puis le code postal (« dupre helene|75011 »), ou None si l'un manque.

    Trier les mots range ensemble « Hélène Dupré » et « DUPRE Helene ».

        >>> cle_nom_code_postal({"nom": "DUPRE Helene", "code_postal": "75011"})
        'dupre helene|75011'
        >>> cle_nom_code_postal({"nom": "Hélène Dupré", "code_postal": None}) is None
        True
    """
    nom = " ".join(sorted(normaliser_nom(fiche.get("nom")).split()))
    code_postal = fiche.get("code_postal")
    if not nom or _absent(code_postal):
        return None
    return f"{nom}|{str(code_postal).strip()}"


CLES = (cle_email, cle_telephone, cle_nom_code_postal)
"""Les trois clés de blocage par défaut du défi 166 : deux fiches qui partagent l'une d'elles sont comparées."""


# ---------------------------------------------------------------------------- les fiches d'exemple


def fiches_exemple() -> dict[str, dict]:
    """14 fiches (site, boutique, support) qui ne représentent que 8 clients : voir `VERITE_EXEMPLE`.

    Ce qui rend le travail difficile, et que chaque défi traite :

    - Hélène Dupré a TROIS fiches : son nom écrit à l'envers en boutique (F02), sans accents au support
      (F03), un téléphone écrit de deux façons (absent au support), une adresse e-mail en majuscules ;
    - Jean Dupont (F06) est devenu « Jean Dupond » au support (F07) : une faute de frappe ;
    - Paul et Claire Martin (F04, F05) partagent l'e-mail et le téléphone de la famille, mais ce sont deux
      personnes (leurs dates de naissance diffèrent) ;
    - il y a DEUX Marie Petit dans le 15e (F09 et F10, nées en 1990 et 1971) ; F11 est la première, mais
      sa date de naissance est un NaN (une case vide lue par pandas) et Excel a mangé le 0 de son téléphone ;
    - au support, on a saisi le numéro du centre d'appels (01 00 00 00 00) pour deux clients différents
      (F12, F13) ;
    - F03 (la fiche la plus récente d'Hélène) dit qu'elle a REFUSÉ les SMS (`optin_sms` False) : une
      fiche fusionnée ne doit pas réactiver ce que la cliente a refusé (défi 168).

        >>> fiches = fiches_exemple()
        >>> sorted(fiches["F01"])
        ['code_postal', 'email', 'inscription', 'maj', 'naissance', 'nom', 'optin_sms', 'resilie', 'telephone']
        >>> fiches["F11"]["naissance"]
        nan
    """
    d = date
    nan = float("nan")
    lignes = [
        # id     nom              email                           téléphone             cp       naissance
        ("F01", "Hélène Dupré", "helene.dupre@exemple.fr", "06 12 34 56 78", "75011", "1984-03-12",
         d(2019, 5, 2), d(2024, 1, 10), False, True),
        ("F02", "DUPRE Helene", None, "+33 6 12 34 56 78", "75011", None,
         d(2016, 9, 14), d(2023, 6, 1), True, None),
        ("F03", "Helene Dupre", " Helene.Dupre@Exemple.fr", "", "75011", "1984-03-12",
         d(2021, 11, 30), d(2024, 3, 2), False, False),
        ("F04", "Paul Martin", "famille.martin@exemple.fr", "04 78 12 34 56", "69003", "1962-11-02",
         d(2015, 2, 20), d(2023, 12, 5), False, True),
        ("F05", "Claire Martin", "famille.martin@exemple.fr", "04 78 12 34 56", "69003", "1965-04-17",
         d(2018, 7, 9), d(2024, 2, 11), False, True),
        ("F06", "Jean Dupont", "jdupont@exemple.com", "07 11 22 33 44", "33000", "1978-01-30",
         d(2020, 1, 15), d(2023, 10, 3), False, None),
        ("F07", "Jean Dupond", "JDupont@exemple.com", None, "33000", None,
         d(2022, 4, 4), d(2024, 2, 28), False, True),
        ("F08", "Dupont Jean", None, "0711223344", "33000", "1978-01-30",
         d(2017, 6, 21), d(2022, 8, 17), True, None),
        ("F09", "Marie Petit", "marie.petit@exemple.fr", "06 98 76 54 32", "75015", "1990-06-05",
         d(2021, 3, 3), d(2024, 1, 22), False, True),
        ("F10", "Marie Petit", "mpetit71@exemple.org", "06 55 44 33 22", "75015", "1971-09-23",
         d(2014, 10, 1), d(2023, 11, 14), False, False),
        ("F11", "Marie PETIT", None, "6 98 76 54 32", "75015", nan,
         d(2020, 12, 18), d(2022, 5, 9), True, None),
        ("F12", "Karim Benali", "k.benali@exemple.fr", "01 00 00 00 00", "13001", "1985-02-14",
         d(2019, 9, 9), d(2023, 9, 30), False, True),
        ("F13", "Sophie Leroy", None, "01 00 00 00 00", "59000", None,
         d(2023, 1, 5), d(2024, 1, 5), False, None),
        ("F14", "Karim Benali", "K.Benali@exemple.fr ", None, "13001", "1985-02-14",
         d(2016, 3, 1), d(2021, 7, 7), True, None),
    ]
    champs = ("nom", "email", "telephone", "code_postal", "naissance", "inscription", "maj", "resilie", "optin_sms")
    return {ligne[0]: dict(zip(champs, ligne[1:])) for ligne in lignes}


VERITE_EXEMPLE = (("F01", "F02", "F03"), ("F04",), ("F05",), ("F06", "F07", "F08"), ("F09", "F11"), ("F10",),
                  ("F12", "F14"), ("F13",))
"""Les 8 vrais clients de `fiches_exemple()` (ce que le dédoublonnage doit retrouver)."""


# ---------------------------------------------------------------------------- beaucoup de fiches


_PRENOMS = ("Hélène", "Jean", "Marie", "Karim", "Sophie", "Paul", "Claire", "Nadia", "Thomas", "Léa", "Yanis",
            "Camille", "Hugo", "Inès", "Lucas", "Chloé", "Mehdi", "Emma", "Louis", "Aïcha", "Théo", "Manon",
            "Rayan", "Julie", "Nathan", "Sarah", "Adam", "Zoé", "Gabriel", "Lina")
_NOMS = ("Dupré", "Dupont", "Martin", "Petit", "Benali", "Leroy", "Moreau", "Fournier", "Girard", "Bonnet",
         "Lambert", "Fontaine", "Rousseau", "Vincent", "Muller", "Lefèvre", "Faure", "André", "Mercier", "Blanc",
         "Guérin", "Boyer", "Garnier", "Chevalier", "François", "Legrand", "Gauthier", "Garcia", "Perrin", "Robin",
         "Diallo", "Nguyen", "Traoré", "N'Diaye", "Haddad", "Rossi", "Lopez", "Da Silva", "Kaya", "Cohen")
_CODES_POSTAUX = tuple(f"{75001 + i}" for i in range(20)) + ("69003", "33000", "13001", "59000", "44000",
                                                              "31000", "67000", "35000", "06000", "34000")


def _faute(rng: random.Random, mot: str) -> str:
    """Une faute de frappe : une lettre remplacée, supprimée ou deux lettres inversées."""
    if len(mot) < 4:
        return mot
    i = rng.randrange(1, len(mot) - 1)
    choix = rng.random()
    if choix < 0.4:
        return mot[:i] + rng.choice("aeinrstu") + mot[i + 1:]
    if choix < 0.7:
        return mot[:i] + mot[i + 1:]
    return mot[:i - 1] + mot[i] + mot[i - 1] + mot[i + 1:]


def _variante_telephone(rng: random.Random, numero: str) -> str:
    """Le même numéro (« 06XXXXXXXX ») écrit comme un humain ou un tableur l'écrirait."""
    paires = [numero[i:i + 2] for i in range(0, 10, 2)]
    return rng.choice([
        " ".join(paires), numero, ".".join(paires), "+33 " + numero[1] + " " + " ".join(paires[1:]),
        "+33 (0)" + numero[1] + " " + " ".join(paires[1:]), "0033" + numero[1:], numero[1:]])


def generer_fiches(n_clients: int = 1000, graine: int = 0, taux_doublons: float = 0.4) -> tuple[dict, list]:
    """Des fiches tirées au hasard et la vérité : `(fiches, verite)`.

    Chaque client a 1 fiche, ou (avec la probabilité `taux_doublons`) 2 ou 3, abîmées comme dans la vraie
    vie : champs vides, NaN, nom à l'envers ou avec une faute, téléphone écrit autrement, e-mail en
    majuscules. Les noms sont tirés parmi 30 prénoms et 40 noms : il y a donc des HOMONYMES (deux
    personnes différentes, même nom, parfois même code postal).

    `verite` est une liste de groupes (des tuples d'identifiants triés), triée : le format de sortie du
    défi 165. Même graine, mêmes fiches.

        >>> fiches, verite = generer_fiches(200, graine=1)
        >>> len(verite), len(fiches) > 200, sum(len(g) for g in verite) == len(fiches)
        (200, True, True)
        >>> generer_fiches(50, graine=3) == generer_fiches(50, graine=3)
        False
        >>> generer_fiches(50, graine=3)[1] == generer_fiches(50, graine=3)[1]
        True

    (Les fiches contiennent des NaN, et NaN n'est égal à rien, même pas à lui-même : deux tirages
    identiques ne sont donc pas `==`. La vérité, elle, l'est.)
    """
    rng = random.Random(graine)
    fiches: dict[str, dict] = {}
    verite: list[tuple[str, ...]] = []
    numero = 0
    debut = date(2014, 1, 1)
    for _ in range(n_clients):
        prenom, nom = rng.choice(_PRENOMS), rng.choice(_NOMS)
        code_postal = rng.choice(_CODES_POSTAUX)
        naissance = (date(1950, 1, 1) + timedelta(days=rng.randrange(0, 365 * 55))).isoformat()
        telephone = "0" + rng.choice("67") + "".join(rng.choice("0123456789") for _ in range(8))
        email = f"{normaliser_nom(prenom)}.{normaliser_nom(nom).replace(' ', '')}{rng.randrange(1, 999)}@exemple.fr"
        nb = 1 if rng.random() >= taux_doublons else rng.choice((2, 2, 3))
        groupe = []
        for k in range(nb):
            numero += 1
            ident = f"F{numero:06d}"
            groupe.append(ident)
            nom_ecrit = f"{prenom} {nom}"
            if k:  # les fiches suivantes sont abîmées
                tirage = rng.random()
                if tirage < 0.25:
                    nom_ecrit = f"{nom.upper()} {prenom}"
                elif tirage < 0.45:
                    nom_ecrit = f"{prenom} {_faute(rng, nom)}"
                elif tirage < 0.6:
                    nom_ecrit = normaliser_nom(nom_ecrit).title()
            inscription = debut + timedelta(days=rng.randrange(0, 3000))
            fiche = {
                "nom": nom_ecrit,
                "email": (email.upper() if rng.random() < 0.2 else email) if rng.random() < 0.7 else None,
                "telephone": _variante_telephone(rng, telephone) if rng.random() < 0.75 else "",
                "code_postal": code_postal if rng.random() < 0.9 else None,
                "naissance": naissance if rng.random() < 0.6 else rng.choice((None, float("nan"))),
                "inscription": inscription,
                "maj": inscription + timedelta(days=rng.randrange(0, 900)),
                "resilie": rng.random() < 0.3,
                "optin_sms": rng.choice((True, False, None)),
            }
            fiches[ident] = fiche
        verite.append(tuple(groupe))
    verite.sort()
    return fiches, verite

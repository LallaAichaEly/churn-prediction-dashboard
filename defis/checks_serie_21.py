"""Vérifications automatiques de la série 21 (lisez-les si vous êtes bloqué·e : ce sont des indices).

Le correcteur vous envoie des dates et des abonnements tels qu'ils arrivent vraiment :

- au défi 128, des dates ambiguës (« 03/01/2024 » : 3 janvier ou 1er mars ?), des espaces autour, des
  cases vides, et des dates qu'il faut REFUSER : format ISO, année sur 2 chiffres, 31 février ;
- aux défis 129 à 131, deux années de dates (janvier 2023 n'est pas janvier 2024), des mois sans aucun
  événement, des clients toujours abonnés (fin = NaT), des clients arrivés ou partis pile le 1er du mois
  ou le dernier jour à 23 h 30, et une photo prise en milieu de mois ;
- au défi 132, des lignes dans le désordre, un index qui ne commence pas à 0, des mois manquants ;
- au défi 133, des options facturées sur une ligne à part, des clients absents un mois puis de retour,
  et un mois où personne n'a été facturé.

Quand un résultat est faux, le correcteur rejoue des versions boguées classiques pour vous dire
LAQUELLE vous avez écrite. Chaque fonction est aussi comparée au corrigé sur des données tirées au hasard.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

from .abonnements import abonnements_exemple, dates_saisies, factures_exemple, generer_abonnements, generer_factures
from .verif import Defi, Serie

# ---------------------------------------------------------------------------- outils


def _decrire(exc: BaseException) -> str:
    texte = str(exc).split("\n")[0]
    return f"{type(exc).__name__}({texte!r})" if texte else type(exc).__name__


def _appel(appel, quoi: str, conseil: str = ""):
    """Appelle `appel()` ; une exception inattendue devient un message qui dit QUEL appel a planté."""
    try:
        return appel()
    except NotImplementedError:
        raise
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. {conseil}".rstrip()) from None


def _montrer(valeur) -> str:
    """Une valeur lisible dans un message : 03/01/2024, 2024-01, 0.25, NaN…"""
    if valeur is None:
        return "None"
    try:
        if np.ndim(valeur) == 0 and pd.isna(valeur):
            return "NaT" if isinstance(valeur, type(pd.NaT)) else "NaN"
    except (TypeError, ValueError):
        pass
    if isinstance(valeur, pd.Timestamp):
        return valeur.strftime("%d/%m/%Y %H:%M" if (valeur.hour or valeur.minute) else "%d/%m/%Y")
    if isinstance(valeur, pd.Period):
        return str(valeur)
    if isinstance(valeur, (float, np.floating)):
        return f"{float(valeur):.4g}"
    if isinstance(valeur, (np.integer,)):
        return str(int(valeur))
    texte = repr(valeur)
    return texte if len(texte) <= 120 else texte[:119] + "…"


def _egal(obtenu, attendu) -> bool:
    try:
        o_manque, a_manque = bool(pd.isna(obtenu)), bool(pd.isna(attendu))
    except (TypeError, ValueError):
        return False
    if o_manque or a_manque:
        return o_manque and a_manque
    if isinstance(attendu, (float, np.floating)):
        try:
            return math.isclose(float(obtenu), float(attendu), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    try:
        return bool(obtenu == attendu)
    except (TypeError, ValueError):
        return False


def _premiere_difference(obtenu, attendu):
    """(étiquette, colonne, obtenu, attendu) de la première case qui diffère, ou None. Mêmes index supposés."""
    if isinstance(attendu, pd.Series):
        for etiquette in attendu.index:
            if not _egal(obtenu.loc[etiquette], attendu.loc[etiquette]):
                return etiquette, None, obtenu.loc[etiquette], attendu.loc[etiquette]
        return None
    for colonne in attendu.columns:
        for etiquette in attendu.index:
            if not _egal(obtenu.at[etiquette, colonne], attendu.at[etiquette, colonne]):
                return etiquette, colonne, obtenu.at[etiquette, colonne], attendu.at[etiquette, colonne]
    return None


def _identiques(obtenu, attendu) -> bool:
    """Même forme (index, colonnes) et mêmes valeurs, au flottant près."""
    try:
        if isinstance(attendu, pd.DataFrame):
            if not isinstance(obtenu, pd.DataFrame) or list(obtenu.columns) != list(attendu.columns):
                return False
        elif not isinstance(obtenu, pd.Series):
            return False
        if list(obtenu.index) != list(attendu.index):
            return False
        return _premiere_difference(obtenu, attendu) is None
    except Exception:
        return False


def _exiger(resultat, type_attendu, quoi: str, conseil: str = ""):
    if resultat is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    if not isinstance(resultat, type_attendu):
        raise AssertionError(f"{quoi} doit renvoyer un {type_attendu.__name__}, pas un {type(resultat).__name__}. "
                             f"{conseil}".rstrip())
    return resultat


def _verifier_index_mois(index, attendu: pd.Index, quoi: str, nom: str, mot: str = "mois") -> None:
    """L'index doit être fait de mois (`Period`), complet, dans l'ordre, et porter le bon nom."""
    if isinstance(index, pd.DatetimeIndex):
        raise AssertionError(
            f"{quoi} : l'index contient des DATES ({_montrer(index[0]) if len(index) else 'vide'}…), pas des mois. "
            f"Un mois est une `Period` : `.dt.to_period(\"M\")`, ou `pd.period_range(...)`.")
    if not isinstance(index, pd.PeriodIndex):
        premier = index[0] if len(index) else None
        if isinstance(premier, str):
            conseil = ("Des textes « 2024-01 » se trient mal (« 2024-10 » passe avant « 2024-9 ») et ne s'additionnent "
                       "pas : utilisez des `Period` (`.dt.to_period(\"M\")`).")
        elif isinstance(index, pd.RangeIndex):
            conseil = (f"Les {mot}s doivent être l'INDEX du résultat, pas une colonne (`.set_index(...)`), ou partez "
                       f"directement de `pd.period_range(...)`.")
        elif isinstance(premier, (int, np.integer)):
            conseil = ("Un numéro de mois (1 à 12) ne dit pas l'année : `dt.month` vaut 1 pour janvier 2023 ET janvier "
                       "2024. Utilisez `.dt.to_period(\"M\")`.")
        else:
            conseil = "Utilisez des `Period` : `.dt.to_period(\"M\")`, ou `pd.period_range(...)`."
        nature = {"str": "des textes", "int": "des entiers", "int64": "des entiers"}.get(
            type(premier).__name__, f"des {type(premier).__name__ if premier is not None else type(index).__name__}")
        raise AssertionError(f"{quoi} : l'index doit contenir des mois (`Period`), pas {nature}. {conseil}")
    if index.freqstr != attendu.freqstr:
        raise AssertionError(f"{quoi} : l'index contient des périodes de fréquence {index.freqstr!r} au lieu de "
                             f"{attendu.freqstr!r} : `.to_period(\"M\")` pour des MOIS.")
    obtenus, attendus = list(index), list(attendu)
    pluriel = {"mois": "mois", "cohorte": "cohortes"}.get(mot, mot + "s")
    if obtenus != attendus:
        manquants = [str(m) for m in attendus if m not in set(obtenus)]
        en_trop = [str(m) for m in obtenus if m not in set(attendus)]
        if manquants and not en_trop:
            if mot == "mois" and manquants == [str(attendus[-1])]:
                conseil = "Le dernier mois est INCLUS : `pd.period_range(premier, dernier, freq=\"M\")` inclut les deux bouts."
            elif mot == "mois":
                conseil = ("Un mois où rien ne se passe existe quand même, et apparaît dans le résultat : partez du "
                           "calendrier complet (`pd.period_range`), puis `reindex(..., fill_value=0)`.")
            else:
                conseil = ""
            raise AssertionError(f"{quoi} : il manque les {pluriel} {', '.join(manquants[:6])}. {conseil}".rstrip())
        if en_trop and not manquants:
            raise AssertionError(f"{quoi} : les {pluriel} {', '.join(en_trop[:6])} sont en trop. "
                                 f"{'Ils sont hors de la période demandée.' if mot == 'mois' else ''}".rstrip())
        if sorted(obtenus) == attendus:
            raise AssertionError(f"{quoi} : les {pluriel} doivent être dans l'ordre chronologique (`.sort_index()`).")
        raise AssertionError(f"{quoi} : l'index contient {', '.join(map(str, obtenus[:6]))}… au lieu de "
                             f"{', '.join(map(str, attendus[:6]))}…")
    if index.name != nom:
        raise AssertionError(f"{quoi} : l'index doit s'appeler {nom!r} (il s'appelle {index.name!r}) : "
                             f"`pd.period_range(..., name={nom!r})`, ou `.rename_axis({nom!r})`.")


def _verifier_colonnes(table: pd.DataFrame, colonnes: list, quoi: str) -> None:
    obtenues = list(table.columns)
    if obtenues != colonnes:
        manquantes = [c for c in colonnes if c not in obtenues]
        if manquantes:
            raise AssertionError(f"{quoi} : il manque la colonne {manquantes[0]!r} (colonnes attendues, dans l'ordre : "
                                 f"{colonnes}).")
        raise AssertionError(f"{quoi} : colonnes {obtenues} au lieu de {colonnes} (dans cet ordre).")


def _verifier_entiers(table: pd.DataFrame, colonnes: list, quoi: str) -> None:
    for colonne in colonnes:
        if table[colonne].dtype.kind not in "iu":
            raise AssertionError(f"{quoi} : la colonne {colonne!r} doit contenir des ENTIERS (ce sont des comptes de "
                                 f"clients), pas des {table[colonne].dtype} : `.astype(\"int64\")`.")


def _verifier_intact(apres, avant, quoi: str) -> None:
    try:
        if isinstance(avant, pd.DataFrame):
            pd.testing.assert_frame_equal(apres, avant)
        else:
            pd.testing.assert_series_equal(apres, avant)
    except AssertionError:
        raise AssertionError(f"{quoi} a MODIFIÉ la table reçue. Une fonction de calcul ne touche pas à ses entrées "
                             f"(l'appelant s'en sert encore) : travaillez sur des colonnes calculées, ou sur "
                             f"`.copy()`.") from None


def _difference_lisible(obtenu, attendu, quoi: str, conseil: str = "") -> None:
    difference = _premiere_difference(obtenu, attendu)
    if difference is None:
        return
    etiquette, colonne, o, a = difference
    ou = f"{_montrer(etiquette)}" + (f", colonne {colonne!r}" if colonne is not None else "")
    raise AssertionError(f"{quoi} : {ou} : {_montrer(o)} au lieu de {_montrer(a)}. {conseil}".rstrip())


# ============================================================================ Défi 128 : lire_dates


def _ref_lire_dates(textes: pd.Series) -> pd.Series:
    return pd.to_datetime(textes.str.strip(), format="%d/%m/%Y")


def _illisibles():
    return ["2024-01-03", "03-01-2024", "3/1/24", "31/02/2024", "29/02/2023", "13/13/2024", "demain",
            "03/01/2024 10:00", "3 janvier 2024"]


def _date_au_hasard(rng: random.Random):
    tirage = rng.random()
    if tirage < 0.08:
        return rng.choice(["", "   ", None, np.nan])
    jour = pd.Timestamp("2000-01-01") + pd.Timedelta(days=rng.randint(0, 11_000))
    j, m = (f"{jour.day:02d}", f"{jour.month:02d}") if rng.random() < 0.7 else (str(jour.day), str(jour.month))
    return rng.choice(["", "", " ", "  "]) + f"{j}/{m}/{jour.year}" + rng.choice(["", "", " ", "\t"])


def verifier_lire_dates(lire_dates) -> None:
    # 1. Des dates toutes ambiguës : le piège du « mois d'abord ».
    entree = pd.Series(["03/01/2024", "05/02/2024", "12/11/2023"], index=[7, 3, 9], name="inscription", dtype=object)
    quoi = "lire_dates(pd.Series(['03/01/2024', '05/02/2024', '12/11/2023']))"
    obtenu = _exiger(_appel(lambda: lire_dates(entree.copy()), quoi), pd.Series, quoi)
    if not pd.api.types.is_datetime64_any_dtype(obtenu.dtype):
        raise AssertionError(f"{quoi} renvoie des valeurs de type {obtenu.dtype} : il faut des DATES (`datetime64`), "
                             f"pour pouvoir les comparer et en tirer le mois. Convertissez avec `pd.to_datetime`.")
    if list(obtenu.index) != [7, 3, 9]:
        raise AssertionError(f"{quoi} : l'index du résultat est {list(obtenu.index)} au lieu de [7, 3, 9] (celui de "
                             f"`textes`). Chaque date doit rester en face de SON client : ne fabriquez pas une nouvelle "
                             f"série à partir d'une liste ; `pd.to_datetime(serie, ...)` garde l'index.")
    attendu = [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-02-05"), pd.Timestamp("2023-11-12")]
    inverse = [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-05-02"), pd.Timestamp("2023-12-11")]
    if list(obtenu) == inverse:
        raise AssertionError(f"{quoi} donne le 1er mars, le 2 mai et le 11 décembre : jour et mois INVERSÉS. Sans "
                             f"`format=`, pandas devine le format, et devine à l'américaine (MM/JJ/AAAA) quand la date "
                             f"le permet. Écrivez `pd.to_datetime(..., format=\"%d/%m/%Y\")`.")
    if list(obtenu) != attendu:
        raise AssertionError(f"{quoi} donne {[_montrer(v) for v in obtenu]} au lieu de 03/01/2024, 05/02/2024 et "
                             f"12/11/2023.")
    if obtenu.name != "inscription":
        raise AssertionError(f"{quoi} : le résultat s'appelle {obtenu.name!r} au lieu de 'inscription' (le nom de "
                             f"`textes`). `pd.to_datetime(serie, ...)` le garde.")

    # 2. Le fichier réel : espaces autour, cases vides, jours et mois sans zéro.
    entree = dates_saisies()
    quoi = "lire_dates(dates_saisies())"
    try:
        obtenu = lire_dates(entree)
    except NotImplementedError:
        raise
    except ValueError as exc:
        raise AssertionError(
            f"{quoi} lève {_decrire(exc)}. Ce fichier est lisible : retirez d'abord les espaces autour des dates "
            f"(« 15/02/2024 » entourée d'espaces) avec `.str.strip()`. Une case vide ou None donne NaT.") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}. Les valeurs manquantes (None) doivent donner NaT : "
                             f"`.str.strip()` les laisse manquantes.") from None
    _verifier_intact(entree, dates_saisies(), quoi)
    obtenu = _exiger(obtenu, pd.Series, quoi)
    attendu = _ref_lire_dates(dates_saisies())
    if list(obtenu.index) != list(attendu.index):
        raise AssertionError(f"{quoi} : l'index du résultat est {list(obtenu.index)} au lieu de {list(attendu.index)}.")
    _difference_lisible(obtenu, attendu, quoi, "Format JJ/MM/AAAA, espaces autour retirés ; vide ou None → NaT.")

    # 3. Une case faite d'espaces est vide, pas illisible.
    quoi = "lire_dates(pd.Series(['   ', '01/02/2024']))"
    obtenu = _exiger(_appel(lambda: lire_dates(pd.Series(["   ", "01/02/2024"], dtype=object)), quoi,
                            "Une case faite d'espaces est une case VIDE : retirez les espaces, puis \"\" donne NaT."),
                     pd.Series, quoi)
    if not (pd.isna(obtenu.iloc[0]) and obtenu.iloc[1] == pd.Timestamp("2024-02-01")):
        raise AssertionError(f"{quoi} donne {[_montrer(v) for v in obtenu]} au lieu de [NaT, 01/02/2024].")

    # 4. Ce qu'il faut refuser.
    for illisible in _illisibles():
        for valeurs in ([illisible], ["03/01/2024", illisible]):
            quoi = f"lire_dates(pd.Series({valeurs!r}))"
            try:
                resultat = lire_dates(pd.Series(valeurs, dtype=object))
            except NotImplementedError:
                raise
            except ValueError:
                continue
            except Exception as exc:
                raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de ValueError.") from None
            valeur = resultat.iloc[-1] if isinstance(resultat, pd.Series) and len(resultat) else resultat
            if isinstance(valeur, type(pd.NaT)) or (isinstance(resultat, pd.Series) and pd.isna(valeur)):
                raise AssertionError(
                    f"{quoi} renvoie NaT au lieu de lever ValueError. Une date illisible n'est pas une date manquante : "
                    f"`errors=\"coerce\"` cache l'erreur, qui ressortira trois traitements plus loin sans qu'on sache "
                    f"d'où elle vient. Laissez pandas lever sa ValueError (`errors=\"raise\"`, la valeur par défaut).")
            raise AssertionError(
                f"{quoi} renvoie {_montrer(valeur)} au lieu de lever ValueError : {illisible!r} n'est pas au format "
                f"JJ/MM/AAAA. `dayfirst=True` n'est qu'une préférence (il accepte encore « 2024-01-03 » et « 3/1/24 ») ; "
                f"`format=\"%d/%m/%Y\"` EXIGE ce format.")

    # 5. Des centaines de dates tirées au hasard.
    rng = random.Random(128)
    for _ in range(4):
        n = rng.randint(1, 60)
        entree = pd.Series([_date_au_hasard(rng) for _ in range(n)], index=rng.sample(range(1000), n), dtype=object)
        attendu = _ref_lire_dates(entree.copy())
        quoi = f"lire_dates(<{n} dates tirées au hasard>)"
        obtenu = _exiger(_appel(lambda: lire_dates(entree.copy()), quoi), pd.Series, quoi)
        if list(obtenu.index) != list(attendu.index):
            raise AssertionError(f"{quoi} : l'index du résultat n'est pas celui de `textes`.")
        difference = _premiere_difference(obtenu, attendu)
        if difference:
            etiquette, _, o, a = difference
            raise AssertionError(f"{quoi} : {entree.loc[etiquette]!r} donne {_montrer(o)} au lieu de {_montrer(a)}.")


# ============================================================================ Défi 129 : nouveaux_par_mois


def _ref_nouveaux(debuts: pd.Series, premier, dernier) -> pd.Series:
    mois = pd.period_range(premier, dernier, freq="M", name="mois")
    return debuts.dt.to_period("M").value_counts().reindex(mois, fill_value=0).astype("int64").rename("nouveaux")


def _verifier_serie_nouveaux(obtenu, attendu: pd.Series, quoi: str) -> pd.Series:
    obtenu = _exiger(obtenu, pd.Series, quoi, "Renvoyez UNE colonne de comptes (`value_counts()` donne déjà une Series).")
    _verifier_index_mois(obtenu.index, attendu.index, quoi, "mois")
    if obtenu.dtype.kind not in "iu":
        raise AssertionError(f"{quoi} : des {obtenu.dtype} au lieu d'ENTIERS (on compte des clients). "
                             f"`reindex(..., fill_value=0)` garde des entiers ; sinon, `.astype(\"int64\")`.")
    return obtenu


def verifier_nouveaux_par_mois(nouveaux_par_mois) -> None:
    # 1. L'exemple : avril et mai sans inscription.
    entree = abonnements_exemple()["debut"]
    quoi = "nouveaux_par_mois(abonnements_exemple()['debut'], '2024-01', '2024-05')"
    obtenu = _appel(lambda: nouveaux_par_mois(entree, "2024-01", "2024-05"), quoi)
    _verifier_intact(entree, abonnements_exemple()["debut"], quoi)
    attendu = _ref_nouveaux(abonnements_exemple()["debut"], "2024-01", "2024-05")
    obtenu = _verifier_serie_nouveaux(obtenu, attendu, quoi)
    _difference_lisible(obtenu, attendu, quoi, "On compte les dates `debut` de chaque mois.")
    if obtenu.name != "nouveaux":
        raise AssertionError(f"{quoi} : la série s'appelle {obtenu.name!r} au lieu de 'nouveaux' : `.rename(\"nouveaux\")`.")

    # 2. Deux années, des heures, des NaT, des dates hors période.
    debuts = pd.Series(pd.to_datetime(
        ["2023-01-15", "2024-01-20", "2023-02-03", "2024-03-31 23:30", None, "2024-02-01", "2022-12-31 23:59",
         "2024-04-01", "2023-03-08", "2024-01-31 23:59"], format="ISO8601"), index=range(50, 60))
    quoi = "nouveaux_par_mois(<dates de 2022 à 2024>, '2024-01', '2024-03')"
    obtenu = _verifier_serie_nouveaux(_appel(lambda: nouveaux_par_mois(debuts.copy(), "2024-01", "2024-03"), quoi),
                                      _ref_nouveaux(debuts, "2024-01", "2024-03"), quoi)
    attendu = _ref_nouveaux(debuts, "2024-01", "2024-03")
    if obtenu.tolist() == [3, 2, 2]:
        raise AssertionError(
            f"{quoi} donne {obtenu.tolist()} au lieu de {attendu.tolist()} : les clients de janvier, février et mars "
            f"2023 sont comptés en 2024. `dt.month` vaut 1 pour janvier 2023 ET pour janvier 2024 : c'est un numéro de "
            f"mois, pas un mois. Utilisez `dt.to_period(\"M\")`.")
    periode = pd.period_range("2024-01", "2024-03", freq="M")
    a_minuit = [int(((debuts >= m.start_time) & (debuts <= m.end_time.normalize())).sum()) for m in periode]
    if obtenu.tolist() == a_minuit:
        raise AssertionError(
            f"{quoi} donne {obtenu.tolist()} au lieu de {attendu.tolist()} : une date avec une HEURE le dernier jour du "
            f"mois (« 31/01/2024 à 23 h 59 », « 31/03/2024 à 23 h 30 ») est bien dans ce mois. Comparer à "
            f"`pd.Timestamp(\"2024-03-31\")` (minuit) la rejette : comparez des mois (`dt.to_period(\"M\")`).")
    _difference_lisible(obtenu, attendu, quoi, "Seules comptent les dates de la période (NaT ignorés).")

    # 3. Des abonnements tirés au hasard, sur des périodes au hasard.
    rng = random.Random(129)
    for graine in range(3):
        debuts = generer_abonnements(250, graine=graine)["debut"]
        premier = pd.Period("2021-10", "M") + rng.randint(0, 30)
        dernier = premier + rng.randint(0, 20)
        quoi = f"nouveaux_par_mois(generer_abonnements(250, graine={graine})['debut'], '{premier}', '{dernier}')"
        attendu = _ref_nouveaux(debuts, premier, dernier)
        obtenu = _verifier_serie_nouveaux(_appel(lambda: nouveaux_par_mois(debuts.copy(), str(premier), str(dernier)),
                                                 quoi), attendu, quoi)
        _difference_lisible(obtenu, attendu, quoi)


# ============================================================================ Défi 130 : mouvements_mensuels

_COLONNES_MOUVEMENTS = ["actifs_debut", "nouveaux", "partis", "actifs_fin", "taux_churn"]


def _ref_mouvements(abonnements: pd.DataFrame, premier, dernier, variante: str | None = None) -> pd.DataFrame:
    debut, fin = abonnements["debut"], abonnements["fin"]
    mois_debut, mois_fin = debut.dt.to_period("M"), fin.dt.to_period("M")
    lignes = []
    for m in pd.period_range(premier, dernier, freq="M"):
        if variante == "nat_oublie":
            present = (mois_debut < m) & (mois_fin >= m)
        elif variante == "arrive_le_1er":
            present = (debut <= m.start_time) & (mois_fin.isna() | (mois_fin >= m))
        elif variante == "parti_le_1er":
            present = (mois_debut < m) & (fin.isna() | (fin > m.start_time))
        else:
            present = (mois_debut < m) & (mois_fin.isna() | (mois_fin >= m))
        arrive, part = mois_debut == m, mois_fin == m
        actifs = int(present.sum())
        actifs_fin = actifs + int(arrive.sum()) - int(part.sum())
        if variante == "taux_tous_partis":
            numerateur, denominateur = part.sum(), actifs
        elif variante == "taux_sur_fin":
            numerateur, denominateur = (present & part).sum(), actifs_fin
        else:
            numerateur, denominateur = (present & part).sum(), actifs
        taux = float(numerateur / denominateur) if denominateur else (0.0 if variante == "taux_zero" else np.nan)
        lignes.append((actifs, int(arrive.sum()), int(part.sum()), actifs_fin, taux))
    index = pd.period_range(premier, dernier, freq="M", name="mois")
    table = pd.DataFrame(lignes, index=index, columns=_COLONNES_MOUVEMENTS)
    return table.astype({c: "int64" for c in _COLONNES_MOUVEMENTS[:4]})


_CONSEILS_MOUVEMENTS = {
    "nat_oublie": ("Les clients TOUJOURS ABONNÉS (fin = NaT) ont disparu : toute comparaison avec NaT vaut False, "
                   "donc `fin >= ...` les écarte. Dites ce que vous faites d'eux : `mois_fin.isna() | (mois_fin >= m)`."),
    "arrive_le_1er": ("Le client C, arrivé le 1er janvier, est compté comme « déjà là » au début de janvier, ET comme "
                      "nouveau : il est compté deux fois. Déjà là = arrivé AVANT le mois (`mois_debut < m`, strictement)."),
    "parti_le_1er": ("Le client B, parti le 1er février, n'est pas compté au début de février, mais il est compté dans "
                     "les partis de février. Il était encore là fin janvier : pas encore parti = fin pendant le mois ou "
                     "APRÈS (`mois_fin >= m`). Raisonnez en mois, pas en dates : le jour ne compte pas."),
    "taux_tous_partis": ("Le taux de janvier compte D, arrivé le 10 et parti le 25 : il n'était pas là au début du mois, "
                         "il n'est pas au dénominateur, il ne doit pas être au numérateur. Comptez les partis PARMI les "
                         "présents du début : `(present & part).sum() / present.sum()`."),
    "taux_sur_fin": ("Le taux de churn se calcule sur ceux qui POUVAIENT partir : les clients présents au DÉBUT du mois "
                     "(`actifs_debut`), pas ceux de la fin."),
}


def _diagnostic_mouvements(obtenu: pd.DataFrame, abonnements: pd.DataFrame, premier, dernier, quoi: str) -> None:
    attendu = _ref_mouvements(abonnements, premier, dernier)
    for colonne in _COLONNES_MOUVEMENTS[:3]:
        if _premiere_difference(obtenu[colonne], attendu[colonne]) is None:
            continue
        if colonne == "actifs_debut":
            for variante in ("nat_oublie", "arrive_le_1er", "parti_le_1er"):
                fausse = _ref_mouvements(abonnements, premier, dernier, variante)
                if _premiere_difference(obtenu[colonne], fausse[colonne]) is None:
                    raise AssertionError(f"{quoi} : `actifs_debut` vaut {obtenu[colonne].tolist()} au lieu de "
                                         f"{attendu[colonne].tolist()}. {_CONSEILS_MOUVEMENTS[variante]}")
            if obtenu[colonne].iloc[0] == 0 and attendu[colonne].iloc[0] > 0:
                raise AssertionError(f"{quoi} : `actifs_debut` vaut 0 le premier mois : les clients arrivés AVANT la "
                                     f"période (A, B) sont bien là au début. Le stock de départ compte tous les "
                                     f"clients présents, même arrivés il y a longtemps.")
        conseils = {"actifs_debut": "Déjà là = arrivé avant le mois, et pas encore parti (fin pendant le mois, après, "
                                    "ou NaT).",
                    "nouveaux": "Nouveaux = `debut` dans le mois.",
                    "partis": "Partis = `fin` dans le mois (y compris un client arrivé et parti dans le même mois)."}
        _difference_lisible(obtenu[colonne], attendu[colonne], f"{quoi}, colonne {colonne!r}", conseils[colonne])
    boucle = obtenu["actifs_debut"] + obtenu["nouveaux"] - obtenu["partis"]
    if _premiere_difference(obtenu["actifs_fin"], boucle.astype("int64")) is not None:
        raise AssertionError(f"{quoi} : `actifs_fin` vaut {obtenu['actifs_fin'].tolist()}, mais actifs_debut + nouveaux "
                             f"− partis vaut {boucle.tolist()}. Le stock doit boucler.")
    if _premiere_difference(obtenu["taux_churn"], attendu["taux_churn"]) is not None:
        for variante in ("taux_tous_partis", "taux_sur_fin"):
            fausse = _ref_mouvements(abonnements, premier, dernier, variante)
            if _premiere_difference(obtenu["taux_churn"], fausse["taux_churn"]) is None:
                raise AssertionError(f"{quoi} : `taux_churn` vaut {[round(t, 4) for t in obtenu['taux_churn']]} au lieu "
                                     f"de {[round(t, 4) for t in attendu['taux_churn']]}. "
                                     f"{_CONSEILS_MOUVEMENTS[variante]}")
        _difference_lisible(obtenu["taux_churn"], attendu["taux_churn"], f"{quoi}, colonne 'taux_churn'",
                            "Parmi les présents du début du mois, la part partie pendant le mois.")


def _verifier_forme_mouvements(obtenu, attendu: pd.DataFrame, quoi: str) -> pd.DataFrame:
    obtenu = _exiger(obtenu, pd.DataFrame, quoi)
    _verifier_index_mois(obtenu.index, attendu.index, quoi, "mois")
    _verifier_colonnes(obtenu, _COLONNES_MOUVEMENTS, quoi)
    _verifier_entiers(obtenu, _COLONNES_MOUVEMENTS[:4], quoi)
    return obtenu


def verifier_mouvements_mensuels(mouvements_mensuels) -> None:
    # 1. L'exemple : chaque client est un piège.
    entree = abonnements_exemple()
    quoi = "mouvements_mensuels(abonnements_exemple(), '2024-01', '2024-04')"
    obtenu = _appel(lambda: mouvements_mensuels(entree, "2024-01", "2024-04"), quoi)
    _verifier_intact(entree, abonnements_exemple(), quoi)
    attendu = _ref_mouvements(abonnements_exemple(), "2024-01", "2024-04")
    obtenu = _verifier_forme_mouvements(obtenu, attendu, quoi)
    _diagnostic_mouvements(obtenu, abonnements_exemple(), "2024-01", "2024-04", quoi)

    # 2. Avant l'arrivée du premier client : personne au début, pas de taux.
    quoi = "mouvements_mensuels(abonnements_exemple(), '2023-09', '2023-11')"
    obtenu = _appel(lambda: mouvements_mensuels(abonnements_exemple(), "2023-09", "2023-11"), quoi,
                    "Quand personne n'est là au début du mois, le taux n'existe pas : NaN (testez `if actifs` avant "
                    "de diviser).")
    attendu = _ref_mouvements(abonnements_exemple(), "2023-09", "2023-11")
    obtenu = _verifier_forme_mouvements(obtenu, attendu, quoi)
    if (obtenu["taux_churn"].fillna(-1) == 0).all():
        raise AssertionError(f"{quoi} : `taux_churn` vaut 0 alors que PERSONNE n'était là au début du mois. 0 %, c'est "
                             f"« des clients, et aucun n'est parti » ; ici, le taux n'existe pas : NaN.")
    _diagnostic_mouvements(obtenu, abonnements_exemple(), "2023-09", "2023-11", quoi)

    # 3. Des abonnements tirés au hasard.
    rng = random.Random(130)
    for graine in range(3):
        abonnements = generer_abonnements(300, graine=graine)
        premier = pd.Period("2021-11", "M") + rng.randint(0, 24)
        dernier = premier + rng.randint(0, 14)
        quoi = f"mouvements_mensuels(generer_abonnements(300, graine={graine}), '{premier}', '{dernier}')"
        attendu = _ref_mouvements(abonnements, premier, dernier)
        obtenu = _appel(lambda: mouvements_mensuels(abonnements.copy(), str(premier), str(dernier)), quoi)
        obtenu = _verifier_forme_mouvements(obtenu, attendu, quoi)
        _diagnostic_mouvements(obtenu, abonnements, premier, dernier, quoi)

    # 4. Une résiliation avant le début : une erreur dans les données.
    fautive = abonnements_exemple()
    fautive.loc[1, "fin"] = pd.Timestamp("2023-12-01")
    quoi = "mouvements_mensuels(<B résilié le 01/12/2023, avant son début le 05/12/2023>, '2024-01', '2024-04')"
    try:
        resultat = mouvements_mensuels(fautive, "2024-01", "2024-04")
    except NotImplementedError:
        raise
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de ValueError.") from None
    raise AssertionError(f"{quoi} devrait lever ValueError : un abonnement ne finit pas avant d'avoir commencé (et "
                         f"renvoie {type(resultat).__name__}). Testez `(fin < debut).any()` : NaT < date vaut False, "
                         f"les abonnés en cours ne déclenchent pas l'erreur.")


# ============================================================================ Défi 131 : table_retention


def _ref_retention(abonnements: pd.DataFrame, photo, variante: str | None = None) -> pd.DataFrame:
    dernier = pd.Timestamp(photo).to_period("M") - (0 if variante == "mois_en_cours" else 1)
    cohorte = abonnements["debut"].dt.to_period("M")
    mois_fin = abonnements["fin"].dt.to_period("M")
    garde = cohorte <= dernier
    cohorte, mois_fin = cohorte[garde], mois_fin[garde]
    cohortes = pd.PeriodIndex(sorted(cohorte.unique()), freq="M", name="cohorte")
    if len(cohortes) == 0:
        return pd.DataFrame(index=cohortes, columns=pd.RangeIndex(0, name="age"), dtype="float64")
    ages = list(range((dernier - cohortes[0]).n + 1))
    table = pd.DataFrame(np.nan, index=cohortes, columns=pd.Index(ages, name="age"))
    for c in cohortes:
        fins = mois_fin[cohorte == c]
        encore = len(fins)
        for a in ages:
            m = c + a
            if m > dernier:
                if variante == "zero_futur":
                    table.loc[c, a] = 0.0
                    continue
                break
            if variante == "nat_oublie":
                restent = fins > m
            elif variante == "fin_incluse":
                restent = fins.isna() | (fins >= m)
            else:
                restent = fins.isna() | (fins > m)
            if variante == "conditionnel":
                table.loc[c, a] = float(restent.sum() / encore) if encore else np.nan
                encore = int(restent.sum())
            else:
                table.loc[c, a] = float(restent.sum() / len(fins))
    return table


_CONSEILS_RETENTION = {
    "zero_futur": ("Les cases des mois PAS ENCORE TERMINÉS valent 0 : c'est dire « tout le monde est parti ». On ne sait "
                   "pas encore : NaN. C'est ce qui donne à la table sa forme de triangle."),
    "mois_en_cours": ("Le mois de la photo n'est pas terminé (le 15 avril, avril est en cours) : le dernier mois terminé "
                      "est le mois de la photo MOINS 1."),
    "conditionnel": ("Le dénominateur doit rester la taille de la cohorte ENTIÈRE, pas les clients encore là le mois "
                     "d'avant : la rétention « 2 mois après » se lit directement, et la ligne ne peut que descendre."),
    "nat_oublie": ("Les clients toujours abonnés (fin = NaT) sont comptés comme partis : `fin > m` vaut False pour NaT. "
                   "Encore là = `mois_fin.isna() | (mois_fin > m)`."),
    "fin_incluse": ("Un client parti PENDANT le mois c + a n'est plus là à la fin de ce mois : encore là = parti APRÈS "
                    "(`mois_fin > m`), pas `>=`."),
}


def _verifier_retention(obtenu, abonnements: pd.DataFrame, photo, quoi: str) -> None:
    attendu = _ref_retention(abonnements, photo)
    obtenu = _exiger(obtenu, pd.DataFrame, quoi)
    _verifier_index_mois(obtenu.index, attendu.index, quoi, "cohorte", mot="cohorte")
    colonnes, attendues = list(obtenu.columns), list(attendu.columns)
    if colonnes != attendues:
        if colonnes and isinstance(colonnes[0], pd.Period):
            raise AssertionError(f"{quoi} : les colonnes sont des MOIS ({colonnes[0]}…). Il faut des ÂGES : 0 (le mois "
                                 f"d'arrivée), 1, 2… pour comparer les cohortes au même âge.")
        if colonnes and isinstance(colonnes[0], str):
            raise AssertionError(f"{quoi} : les âges doivent être des ENTIERS (0, 1, 2…), pas des textes ({colonnes[0]!r}).")
        if attendues and colonnes == attendues + [attendues[-1] + 1]:
            raise AssertionError(f"{quoi} : une colonne de trop ({colonnes[-1]}). {_CONSEILS_RETENTION['mois_en_cours']}")
        raise AssertionError(f"{quoi} : colonnes {colonnes[:8]} au lieu de {attendues[:8]} : les âges vont de 0 à "
                             f"l'écart en mois entre la plus ancienne cohorte et le dernier mois TERMINÉ.")
    if _premiere_difference(obtenu, attendu) is None:
        return
    etiquette, colonne, o, a = _premiere_difference(obtenu, attendu)
    case = f"cohorte {etiquette}, âge {colonne} : {_montrer(o)} au lieu de {_montrer(a)}"
    for variante, conseil in _CONSEILS_RETENTION.items():
        if variante != "mois_en_cours" and _identiques(obtenu, _ref_retention(abonnements, photo, variante)):
            raise AssertionError(f"{quoi} : {case}. {conseil}")
    if pd.isna(a) and not pd.isna(o):
        raise AssertionError(f"{quoi} : {case}. Le mois "
                             f"{etiquette + colonne} n'est pas terminé à la date de la photo : on ne sait pas encore.")
    raise AssertionError(f"{quoi} : {case}. La "
                         f"part des clients de la cohorte encore abonnés à la fin du mois {etiquette + colonne} "
                         f"(toujours abonnés, ou partis après ce mois).")


def verifier_table_retention(table_retention) -> None:
    # 1. L'exemple, photo le 15 avril.
    entree = abonnements_exemple()
    quoi = "table_retention(abonnements_exemple(), '2024-04-15')"
    obtenu = _appel(lambda: table_retention(entree, "2024-04-15"), quoi)
    _verifier_intact(entree, abonnements_exemple(), quoi)
    _verifier_retention(obtenu, abonnements_exemple(), "2024-04-15", quoi)

    # 2. Les bords : photo le 1er à minuit (mars est fini), le 31 à 23 h (mars ne l'est pas), un Timestamp.
    for photo, explication in (("2024-04-01", "Le 1er avril à minuit, mars est terminé."),
                               ("2024-03-31 23:00", "Le 31 mars à 23 h, mars n'est PAS terminé.")):
        quoi = f"table_retention(abonnements_exemple(), '{photo}')"
        obtenu = _appel(lambda: table_retention(abonnements_exemple(), photo), quoi)
        try:
            _verifier_retention(obtenu, abonnements_exemple(), photo, quoi)
        except AssertionError as exc:
            raise AssertionError(f"{exc} {explication} Le dernier mois terminé est `pd.Timestamp(photo).to_period(\"M\") "
                                 f"- 1`.") from None
    quoi = "table_retention(abonnements_exemple(), pd.Timestamp('2024-04-15'))"
    obtenu = _appel(lambda: table_retention(abonnements_exemple(), pd.Timestamp("2024-04-15")), quoi,
                    "`photo` peut être un texte OU un `pd.Timestamp` : `pd.Timestamp(photo)` accepte les deux.")
    _verifier_retention(obtenu, abonnements_exemple(), "2024-04-15", quoi)

    # 3. Des abonnements tirés au hasard, des photos au hasard.
    rng = random.Random(131)
    for graine in range(3):
        abonnements = generer_abonnements(200, graine=graine)
        photo = (pd.Timestamp("2022-06-01") + pd.Timedelta(days=rng.randint(0, 900), hours=rng.randint(0, 23)))
        quoi = f"table_retention(generer_abonnements(200, graine={graine}), '{photo}')"
        obtenu = _appel(lambda: table_retention(abonnements.copy(), str(photo)), quoi)
        _verifier_retention(obtenu, abonnements, photo, quoi)


# ============================================================================ Défi 132 : plis_temporels


def _ref_plis(mois: pd.Series, n_plis: int, horizon: int, variante: str | None = None):
    distincts = sorted(mois.unique())
    if variante == "calendrier":
        distincts = list(pd.period_range(distincts[0], distincts[-1], freq="M"))
    positions = np.arange(len(mois))
    plis = []
    for rang, t in enumerate(distincts[-n_plis:]):
        test = positions[(mois == t).to_numpy()]
        if variante == "fuite":
            garde = mois < t
        elif variante == "un_de_trop":
            garde = mois < t - horizon
        elif variante == "mois_presents":
            rang_t = len(distincts) - n_plis + rang
            garde = mois <= distincts[rang_t - horizon] if rang_t - horizon >= 0 else mois < distincts[0]
        else:
            garde = mois <= t - horizon
        plis.append((positions[garde.to_numpy()], test))
    return plis


def _mois(textes, index=None) -> pd.Series:
    return pd.Series(pd.PeriodIndex(textes, freq="M"), index=index)


def _en_listes(plis, quoi: str, n_plis: int):
    if plis is None:
        raise AssertionError(f"{quoi} renvoie None : avez-vous oublié le `return` ?")
    try:
        plis = list(plis)
    except TypeError:
        raise AssertionError(f"{quoi} doit renvoyer une LISTE de couples (train, test), pas un "
                             f"{type(plis).__name__}.") from None
    if len(plis) != n_plis:
        raise AssertionError(f"{quoi} renvoie {len(plis)} plis au lieu de {n_plis} (un par mois de test).")
    resultat = []
    for numero, pli in enumerate(plis, start=1):
        try:
            train, test = pli
        except (TypeError, ValueError):
            raise AssertionError(f"{quoi} : le pli n°{numero} doit être un couple (train, test), pas "
                                 f"{_montrer(pli)}.") from None
        morceaux = []
        for nom, morceau in (("train", train), ("test", test)):
            tableau = np.asarray(morceau)
            if tableau.dtype == bool:
                raise AssertionError(f"{quoi} : le `{nom}` du pli n°{numero} est un masque de booléens. Il faut les "
                                     f"POSITIONS des lignes : `np.arange(len(mois))[masque]`, ou `np.flatnonzero(masque)`.")
            if tableau.size and tableau.dtype.kind not in "iu":
                raise AssertionError(f"{quoi} : le `{nom}` du pli n°{numero} contient des {tableau.dtype} ("
                                     f"{_montrer(tableau.flat[0])}…). Il faut les POSITIONS des lignes (des entiers).")
            morceaux.append([int(x) for x in tableau.ravel()])
        resultat.append(tuple(morceaux))
    return resultat


def _comparer_plis(plis, mois: pd.Series, n_plis: int, horizon: int, quoi: str) -> None:
    obtenus = _en_listes(plis, quoi, n_plis)
    attendus = [(t.tolist(), s.tolist()) for t, s in _ref_plis(mois, n_plis, horizon)]
    if obtenus == attendus:
        return
    if sorted(obtenus) == sorted(attendus):
        raise AssertionError(f"{quoi} : les plis sont justes, mais pas dans l'ordre : rendez-les dans l'ordre "
                             f"CHRONOLOGIQUE de leur mois de test (le plus ancien d'abord), comme on les rejouerait.")
    etiquettes = list(mois.index)
    for numero, ((train, test), (train_a, test_a)) in enumerate(zip(obtenus, attendus), start=1):
        mois_test = mois.iloc[test_a[0]] if test_a else None
        for nom, o, a in (("test", test, test_a), ("train", train, train_a)):
            if o == a:
                continue
            if sorted(o) == a:
                raise AssertionError(f"{quoi} : le `{nom}` du pli n°{numero} n'est pas trié : {o[:10]}… "
                                     f"Rendez les positions dans l'ordre croissant.")
            if o == [etiquettes[i] for i in a] and etiquettes != list(range(len(mois))):
                raise AssertionError(
                    f"{quoi} : le `{nom}` du pli n°{numero} contient les ÉTIQUETTES de l'index ({o[:6]}) au lieu des "
                    f"POSITIONS ({a[:6]}). `X.iloc[...]` et les tableaux numpy veulent des positions : partez de "
                    f"`np.arange(len(mois))`, pas de `mois.index`.")
            ordre = list(np.argsort(mois.to_numpy(), kind="stable"))
            if sorted(ordre.index(i) for i in a) == o:
                raise AssertionError(
                    f"{quoi} : le `{nom}` du pli n°{numero} donne des positions dans les données TRIÉES par mois "
                    f"({o[:6]}) au lieu des positions dans `mois` tel qu'il est reçu ({a[:6]}). Ne triez pas les "
                    f"lignes : les positions doivent servir sur `X` et `y`, qui ne sont pas triés.")
        variantes = {
            "fuite": (f"FUITE : pour tester le mois {mois_test}, le `train` contient des photos dont l'étiquette n'est "
                      f"pas encore connue au début de {mois_test} (il faut attendre {horizon} mois pour savoir si le "
                      f"client part). Gardez les mois s tels que s + horizon <= t : `mois <= t - horizon`."),
            "un_de_trop": (f"Le `train` du pli de {mois_test} oublie le mois {mois_test - horizon if mois_test else ''} : "
                           f"son étiquette est connue au début de {mois_test} (s + horizon <= t, l'égalité est permise)."),
            "calendrier": ("Les mois de test sont les derniers mois PRÉSENTS dans les données (`sorted(mois.unique())`), "
                           "pas tous les mois du calendrier : un mois sans aucune ligne ne se teste pas."),
            "mois_presents": ("L'horizon se compte en mois du CALENDRIER (`t - horizon`, une `Period`), pas en mois "
                              "présents dans les données : ici, des mois manquent, et votre `train` remonte trop loin."),
        }
        for variante, conseil in variantes.items():
            try:
                faux = [(t.tolist(), s.tolist()) for t, s in _ref_plis(mois, n_plis, horizon, variante)]
            except Exception:
                continue
            if obtenus == faux:
                raise AssertionError(f"{quoi} : {conseil}")
        if (train, test) != (train_a, test_a):
            raise AssertionError(f"{quoi} : le pli n°{numero} (test sur {mois_test}) vaut train={train[:12]}, "
                                 f"test={test[:12]} au lieu de train={train_a[:12]}, test={test_a[:12]}.")


def _appel_plis(plis_temporels, mois: pd.Series, n_plis: int, horizon: int, quoi: str):
    try:
        return plis_temporels(mois, n_plis, horizon)
    except NotImplementedError:
        raise
    except ValueError as exc:
        raise AssertionError(
            f"{quoi} lève {_decrire(exc)}, alors que chaque pli a des lignes d'entraînement : pour tester le mois t, "
            f"on garde les mois s tels que s + horizon <= t (l'égalité est permise : l'étiquette de la photo du mois "
            f"t - horizon est connue au début du mois t).") from None
    except Exception as exc:
        raise AssertionError(f"{quoi} lève {_decrire(exc)}.") from None


def verifier_plis_temporels(plis_temporels) -> None:
    # 1. L'exemple : lignes dans le désordre, index de 10 à 50.
    exemple = _mois(["2024-03", "2024-01", "2024-02", "2024-03", "2024-01"], index=[10, 20, 30, 40, 50])
    for n_plis, horizon in ((1, 2), (2, 1)):
        entree = exemple.copy()
        quoi = f"plis_temporels(<l'exemple de la docstring>, n_plis={n_plis}, horizon={horizon})"
        plis = _appel_plis(plis_temporels, entree, n_plis, horizon, quoi)
        _verifier_intact(entree, exemple, quoi)
        _comparer_plis(plis, exemple, n_plis, horizon, quoi)

    # 2. Six mois, un par ligne, dans le désordre : le mois t - horizon est-il gardé ?
    six = _mois(["2024-05", "2024-02", "2024-06", "2024-01", "2024-04", "2024-03"], index=list("abcdef"))
    quoi = "plis_temporels(<janvier à juin, dans le désordre>, n_plis=2, horizon=1)"
    _comparer_plis(_appel_plis(plis_temporels, six.copy(), 2, 1, quoi), six, 2, 1, quoi)

    # 3. Deux mois manquent (avril et mai) : les mois de test sont les mois présents, l'horizon se compte au calendrier.
    trou = _mois(["2024-01", "2024-02", "2024-03", "2024-06", "2024-01", "2024-06"], index=range(100, 106))
    for n_plis, horizon in ((2, 1), (1, 2)):
        quoi = f"plis_temporels(<janvier, février, mars, juin : avril et mai manquent>, n_plis={n_plis}, horizon={horizon})"
        _comparer_plis(_appel_plis(plis_temporels, trou.copy(), n_plis, horizon, quoi), trou, n_plis, horizon, quoi)

    # 4. Ce qu'il faut refuser.
    for n_plis, horizon, raison in ((0, 1, "n_plis vaut 0"), (1, 0, "horizon vaut 0 : la photo du mois t aurait son "
                                    "étiquette le jour même"), (4, 1, "4 plis demandés pour 3 mois"),
                                    (2, 2, "le premier mois de test (février) n'a aucun mois s avec s + 2 <= février")):
        quoi = f"plis_temporels(<janvier, février, mars>, n_plis={n_plis}, horizon={horizon})"
        try:
            resultat = plis_temporels(_mois(["2024-01", "2024-02", "2024-03"]), n_plis, horizon)
        except NotImplementedError:
            raise
        except ValueError:
            continue
        except Exception as exc:
            raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de ValueError ({raison}).") from None
        raise AssertionError(f"{quoi} devrait lever ValueError ({raison}), mais renvoie {_montrer(resultat)}.")

    # 5. Des photos tirées au hasard.
    rng = random.Random(132)
    for _ in range(4):
        mois_possibles = [pd.Period("2023-01", "M") + k for k in range(18) if rng.random() < 0.8]
        n = rng.randint(30, 80)
        mois = pd.Series(pd.PeriodIndex([rng.choice(mois_possibles) for _ in range(n)], freq="M"),
                         index=rng.sample(range(10_000), n))
        horizon = rng.randint(1, 3)
        n_plis = rng.randint(1, 3)
        distincts = sorted(mois.unique())
        if len(distincts) <= n_plis or distincts[-n_plis] - horizon < distincts[0]:
            continue
        quoi = f"plis_temporels(<{n} photos tirées au hasard>, n_plis={n_plis}, horizon={horizon})"
        _comparer_plis(_appel_plis(plis_temporels, mois.copy(), n_plis, horizon, quoi), mois, n_plis, horizon, quoi)


# ============================================================================ Défi 133 : pont_mrr

_COLONNES_MRR = ["mrr", "nouveau", "reactivation", "expansion", "contraction", "perdu"]


def _ref_pont(factures: pd.DataFrame) -> pd.DataFrame:
    mois = pd.period_range(factures["mois"].min(), factures["mois"].max(), freq="M", name="mois")
    grille = factures.pivot_table(index="client", columns="mois", values="montant", aggfunc="sum", fill_value=0.0)
    grille = grille.reindex(columns=mois, fill_value=0.0)
    avant = grille.shift(1, axis=1, fill_value=0.0)
    deja_client = grille.cumsum(axis=1).shift(1, axis=1, fill_value=0.0) > 0
    arrive = (avant == 0) & (grille > 0)
    pont = pd.DataFrame({
        "mrr": grille.sum(),
        "nouveau": grille.where(arrive & ~deja_client, 0.0).sum(),
        "reactivation": grille.where(arrive & deja_client, 0.0).sum(),
        "expansion": (grille - avant).where((avant > 0) & (grille > avant), 0.0).sum(),
        "contraction": (avant - grille).where((grille > 0) & (grille < avant), 0.0).sum(),
        "perdu": avant.where((avant > 0) & (grille == 0), 0.0).sum(),
    })
    pont.index = mois
    return pont[_COLONNES_MRR].astype("float64")


def _factures(lignes) -> pd.DataFrame:
    table = pd.DataFrame(lignes, columns=["client", "mois", "montant"])
    table["mois"] = pd.PeriodIndex(table["mois"], freq="M")
    return table


def _verifier_forme_pont(obtenu, attendu: pd.DataFrame, quoi: str) -> pd.DataFrame:
    obtenu = _exiger(obtenu, pd.DataFrame, quoi)
    _verifier_index_mois(obtenu.index, attendu.index, quoi, "mois")
    _verifier_colonnes(obtenu, _COLONNES_MRR, quoi)
    return obtenu


# Des situations isolées, chacune ne contient qu'UN piège : la première qui échoue dit lequel. Le dernier élément
# dit quels mois regarder (None : tous).
_SITUATIONS_MRR = [
    ("deux clients, Q arrive en février", [("P", "2024-01", 10.0), ("P", "2024-02", 10.0), ("Q", "2024-02", 30.0)],
     "En février, Q est un NOUVEAU client (30 €). Votre calcul le compare à la ligne précédente, qui est celle de P : "
     "comparez chaque client à LUI-MÊME, le mois précédent (une grille client × mois, puis `shift` sur les colonnes).",
     None),
    ("une option facturée à part", [("T", "2024-01", 20.0), ("T", "2024-02", 20.0), ("T", "2024-02", 10.0)],
     "En février, T paie 20 € + 10 € (deux lignes le MÊME mois) : 30 €, soit une expansion de 10 €. Additionnez les "
     "lignes d'un même client et d'un même mois (`pivot_table(..., aggfunc=\"sum\")`, ou `groupby([...]).sum()`).",
     None),
    ("R absent en février, de retour en mars", [("R", "2024-01", 20.0), ("R", "2024-03", 20.0),
                                                ("S", "2024-01", 5.0), ("S", "2024-02", 5.0), ("S", "2024-03", 5.0)],
     "R n'a pas été facturé en février : il est PERDU en février (20 €), puis revient en mars. Comparez au mois "
     "CALENDAIRE précédent (0 s'il n'a rien payé), pas à sa dernière facture : une grille client × mois avec "
     "`fill_value=0.0`, puis `shift(1, axis=1)`.", ["2024-02"]),
    ("le retour de R", [("R", "2024-01", 20.0), ("R", "2024-03", 20.0),
                        ("S", "2024-01", 5.0), ("S", "2024-02", 5.0), ("S", "2024-03", 5.0)],
     "En mars, R REVIENT : c'est une réactivation (20 €), pas un nouveau client. Il avait déjà été facturé avant "
     "(`grille.cumsum(axis=1).shift(1, axis=1, fill_value=0.0) > 0`).", None),
    ("février sans aucune facture", [("U", "2024-01", 30.0), ("U", "2024-03", 30.0)],
     "Personne n'a été facturé en février, mais février EXISTE : mrr 0, perdu 30. Partez du calendrier complet "
     "(`pd.period_range(min, max, freq=\"M\")` et `reindex(columns=..., fill_value=0.0)`).", None),
]


def _diagnostic_pont(pont_mrr) -> None:
    for nom, lignes, conseil, a_regarder in _SITUATIONS_MRR:
        factures = _factures(lignes)
        attendu = _ref_pont(factures)
        quoi = f"pont_mrr(<{nom}>)"
        obtenu = _appel(lambda: pont_mrr(factures.copy()), quoi, conseil)
        if not isinstance(obtenu, pd.DataFrame) or list(obtenu.index) != list(attendu.index) \
                or list(obtenu.columns) != _COLONNES_MRR:
            if nom.startswith("février sans"):
                raise AssertionError(f"{quoi} : {conseil}")
            _verifier_forme_pont(obtenu, attendu, quoi)
        if a_regarder is not None:
            garde = [pd.Period(m, "M") for m in a_regarder]
            obtenu, attendu = obtenu.loc[garde], attendu.loc[garde]
        if not _identiques(obtenu, attendu):
            raise AssertionError(f"{quoi} : {conseil}")


def verifier_pont_mrr(pont_mrr) -> None:
    # 1. L'exemple : chaque client est un mouvement.
    entree = factures_exemple()
    quoi = "pont_mrr(factures_exemple())"
    obtenu = _appel(lambda: pont_mrr(entree), quoi)
    _verifier_intact(entree, factures_exemple(), quoi)
    attendu = _ref_pont(factures_exemple())
    obtenu = _verifier_forme_pont(obtenu, attendu, quoi)
    if not _identiques(obtenu, attendu):
        numerique = all(obtenu[c].dtype.kind in "iuf" for c in _COLONNES_MRR)
        if numerique and _identiques(obtenu.abs(), attendu):
            raise AssertionError(f"{quoi} : des montants NÉGATIFS. Toutes les colonnes sont positives (« perdu : 30 € », "
                                 f"pas « −30 € ») : c'est le pont qui les soustrait.")
        _diagnostic_pont(pont_mrr)
        _difference_lisible(obtenu, attendu, quoi)

    # 2. Des situations isolées, même si l'exemple passe.
    _diagnostic_pont(pont_mrr)

    # 3. Ce qu'il faut refuser.
    for montant in (0.0, -15.0):
        factures = _factures([("A", "2024-01", 20.0), ("A", "2024-02", montant)])
        quoi = f"pont_mrr(<une facture de {montant} €>)"
        try:
            resultat = pont_mrr(factures)
        except NotImplementedError:
            raise
        except ValueError:
            continue
        except Exception as exc:
            raise AssertionError(f"{quoi} lève {_decrire(exc)} au lieu de ValueError.") from None
        raise AssertionError(f"{quoi} devrait lever ValueError (un montant nul ou négatif n'est pas un abonnement), mais "
                             f"renvoie un {type(resultat).__name__}.")

    # 4. Des factures tirées au hasard : le pont doit boucler, et être juste.
    for graine in range(3):
        factures = generer_factures(40, graine=graine)
        quoi = f"pont_mrr(generer_factures(40, graine={graine}))"
        attendu = _ref_pont(factures)
        obtenu = _verifier_forme_pont(_appel(lambda: pont_mrr(factures.copy()), quoi), attendu, quoi)
        _difference_lisible(obtenu, attendu, quoi)


SERIE = Serie(
    code="21",
    titre="Le temps qui passe : dates, mois, mouvements, cohortes, validation et revenu",
    module_eleve="defis.serie_21",
    module_corrige="defis.corrige_serie_21",
    defis=(
        Defi(128, "lire_dates", "🟢", "—", verifier_lire_dates),
        Defi(129, "nouveaux_par_mois", "🟢", "—", verifier_nouveaux_par_mois),
        Defi(130, "mouvements_mensuels", "🟡", "—", verifier_mouvements_mensuels),
        Defi(131, "table_retention", "🟡", "—", verifier_table_retention),
        Defi(132, "plis_temporels", "🟡", "3.4", verifier_plis_temporels),
        Defi(133, "pont_mrr", "🔴", "—", verifier_pont_mrr),
    ),
)

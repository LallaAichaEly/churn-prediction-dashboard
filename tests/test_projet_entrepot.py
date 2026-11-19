"""Garantit que le projet 6 (l'entrepôt des clients) est faisable et bien corrigé.

Ces tests NE notent PAS votre projet (pour ça : `python scripts/projet.py --projet 6`) ; ils vérifient
le corrigé, les exports d'exemple, et que chaque erreur classique reçoit le bon message.
"""

import itertools
import re
import shutil
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from projets import checks_entrepot as checks  # noqa: E402
from projets.entrepot_corrige import mes_tests as mes_tests_corrige  # noqa: E402
from projets.entrepot_corrige.modeles import COLONNES_TABLE, ouvrir_base  # noqa: E402
from projets.entrepot_corrige.table import construire_table  # noqa: E402
from projets.exemples_entrepot import PHOTO_EXEMPLE, generer_exports  # noqa: E402
from projets.verif import A_FAIRE, ECHEC, ERREUR, REUSSI, evaluer, evaluer_etape, projets  # noqa: E402

PROJET = checks.PROJET
EXPORTS = ROOT / "projets" / "exemple_exports"
_compteur = itertools.count()


def _python(*args, cwd=ROOT):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          timeout=180)


# ---------------------------------------------------------------------------- Le projet est cohérent


@pytest.mark.parametrize("etape", PROJET.etapes, ids=lambda e: e.module)
def test_le_corrige_reussit(etape):
    resultat = evaluer_etape(PROJET.paquet_corrige, etape, PROJET.etapes)
    assert resultat.statut == REUSSI, resultat.message


def test_le_squelette_est_entierement_a_faire():
    assert [r.statut for r in evaluer(PROJET)] == [A_FAIRE] * len(PROJET.etapes)


@pytest.mark.parametrize("fichier", ["modeles.py", "__main__.py"])
def test_les_fichiers_deja_ecrits_sont_identiques(fichier):
    eleve = ROOT / "projets" / "entrepot" / fichier
    corrige = ROOT / "projets" / "entrepot_corrige" / fichier
    assert eleve.read_text(encoding="utf-8") == corrige.read_text(encoding="utf-8")


def test_les_outils_de_mes_tests_sont_identiques():
    """Les fonctions DÉJÀ ÉCRITES de mes_tests.py sont les mêmes dans le squelette et le corrigé."""
    def outils(paquet):
        texte = (ROOT / "projets" / paquet / "mes_tests.py").read_text(encoding="utf-8")
        return texte[texte.index("def _contrat"):texte.index("def tester_construire_table")]
    assert outils("entrepot") == outils("entrepot_corrige")


def test_le_projet_6_est_enregistre_apres_le_projet_5():
    assert [p.numero for p in projets()] == [2, 3, 4, 5, 6, 7]
    assert projets()[4] is PROJET


def test_les_exports_d_exemple_sont_reproductibles(tmp_path):
    """`projets/exemple_exports/` vaut generer_exports() octet par octet (le hasard vient de `random`)."""
    generes = generer_exports(tmp_path)
    assert sorted(p.name for p in generes) == sorted(p.name for p in EXPORTS.iterdir())
    for chemin in generes:
        assert chemin.read_bytes() == (EXPORTS / chemin.name).read_bytes(), chemin.name


def test_la_table_d_exemple_retrouve_la_verite_cachee(tmp_path):
    """Les clients actifs le 30/06/2026 sont exactement ceux du générateur, avec leurs vraies valeurs."""
    import random
    from datetime import timedelta

    from projets import exemples_entrepot as ex
    from projets.entrepot_corrige.modeles import mois_entre

    fiches, tickets = ex._simuler(random.Random(2026), 800)
    actifs = {f["client"]: f for f in fiches if f["debut"] <= PHOTO_EXEMPLE and (f["fin"] is None or f["fin"] > PHOTO_EXEMPLE)}
    base = checks._base_exemple(tmp_path)
    with sqlite3.connect(base) as connexion:
        table = construire_table(connexion, PHOTO_EXEMPLE).set_index("customer_id")
    assert sorted(table.index) == sorted(actifs) and len(actifs) == 800
    debut_fenetre = PHOTO_EXEMPLE - timedelta(days=365)
    for client, fiche in actifs.items():
        ligne = table.loc[client]
        appels = sum(1 for t in tickets if t["fiche"].get("client") == client and
                     debut_fenetre < t["ouvert"].date() <= PHOTO_EXEMPLE)
        assert ligne["support_calls"] == appels
        assert ligne["tenure_months"] == max(1, mois_entre(fiche["debut"], PHOTO_EXEMPLE))
        assert ligne["churn"] == int(fiche["fin"] is not None and fiche["fin"] <= PHOTO_EXEMPLE + timedelta(days=90))
        assert (ligne["contract"], ligne["payment_method"]) == (fiche["contrat"], fiche["paiement"])
    assert 0.25 < table["churn"].mean() < 0.45


def test_le_scenario_distingue_chaque_version_boguee():
    """Sur le scénario de l'étape 5, chaque version boguée se reconnaît (sauf la maturité, testée à part)."""
    connexion = checks._base_scenario(ouvrir_base)
    juste = checks._resultat(construire_table, connexion, checks._PHOTO)
    sorties = [checks._resultat(m, connexion, checks._PHOTO) for m, _, _ in checks._MUTANTS_TABLE]
    for i, (sortie, (_, quoi, _)) in enumerate(zip(sorties, checks._MUTANTS_TABLE)):
        if "photo trop récente" in quoi:
            assert sortie == juste
            continue
        assert sortie != juste, quoi
        assert sorties.index(sortie) == i, f"« {quoi} » serait confondue avec une autre version"
    connexion.close()


def test_le_corrige_de_l_etape_7_attrape_toutes_les_versions_boguees():
    mes_tests_corrige.tester_construire_table(construire_table)
    for mutant, quoi, _ in checks._MUTANTS_TABLE:
        with pytest.raises(AssertionError):
            mes_tests_corrige.tester_construire_table(mutant)
            pytest.fail(f"la version qui {quoi} n'est pas attrapée")


# ---------------------------------------------------------------------------- Le programme terminé


def test_le_programme_termine_de_bout_en_bout(tmp_path):
    base, rejets, table = tmp_path / "entrepot.db", tmp_path / "rejets.csv", tmp_path / "data" / "customers.csv"
    paquet = "projets.entrepot_corrige"

    res = _python("-m", paquet, "charger", "--dossier", str(EXPORTS), "--base", str(base), "--rejets", str(rejets))
    assert res.returncode == 0, res.stderr
    assert "20 fichier(s) : 20 chargé(s), 0 déjà chargé(s), 0 refusé(s)" in res.stdout
    assert ": 1026 contrats, 2053 tickets" in res.stdout
    assert "⚠️  crm_2026-04-01.csv        chargé : 51 ligne(s) en base, 3 rejet(s)" in res.stdout
    lus = pd.read_csv(rejets)
    assert list(lus.columns) == ["fichier", "ligne", "raison"] and len(lus) == 8
    assert "client : identifiant illisible 'TEST01'" in set(lus["raison"])

    res = _python("-m", paquet, "charger", "--dossier", str(EXPORTS), "--base", str(base), "--rejets", str(rejets))
    assert res.returncode == 0 and "0 chargé(s), 20 déjà chargé(s)" in res.stdout

    res = _python("-m", paquet, "exporter", "--base", str(base), "--photo", "2026-06-30", "--sortie", str(table))
    assert res.returncode == 0, res.stderr
    assert "800 clients au 2026-06-30, départs dans les 90 jours : 34,9 %" in res.stdout
    exportee = pd.read_csv(table)
    assert list(exportee.columns) == list(COLONNES_TABLE)

    res = _python("-m", paquet, "exporter", "--base", str(base), "--photo", "2026-07-02", "--sortie",
                  str(tmp_path / "trop.csv"))
    assert res.returncode == 1 and "photo trop récente" in res.stderr and "Traceback" not in res.stderr
    assert not (tmp_path / "trop.csv").exists()


def test_le_modele_du_projet_1_s_entraine_sur_la_table_exportee(tmp_path):
    """La table exportée est un `data/customers.csv` valide : le projet 1 l'entraîne tel quel."""
    from churn.data import validate
    from churn.model import train

    with sqlite3.connect(checks._base_exemple(tmp_path)) as connexion:
        table = construire_table(connexion, PHOTO_EXEMPLE)
    assert validate(table) == []
    _, _, _, _, mesures = train(table, "logistic")
    assert mesures.roc_auc > 0.7


def test_le_squelette_explique_qu_il_n_est_pas_termine(tmp_path):
    res = _python("-m", "projets.entrepot", "charger", "--dossier", str(EXPORTS), "--base", str(tmp_path / "e.db"))
    assert res.returncode == 1
    assert "scripts/projet.py --projet 6" in res.stderr and "Traceback" not in res.stderr


def test_le_script_de_progression_connait_le_projet_6():
    res = _python("scripts/projet.py", "--projet", "6", "--corrige")
    assert res.returncode == 0 and "7/7" in res.stdout and "Projet 6" in res.stdout and "Projet 5" not in res.stdout
    res = _python("scripts/projet.py", "--projet", "6")
    assert res.returncode == 0 and "0/7" in res.stdout and "Prochaine étape : 1" in res.stdout


# ---------------------------------------------------------------------------- Les erreurs classiques


def _variante(tmp_path, monkeypatch, fichier: str, remplacements):
    """Le corrigé, avec UNE erreur classique introduite dans `fichier`, corrigé par le vrai moteur."""
    nom = f"entrepot_variante_{next(_compteur)}"
    dossier = tmp_path / nom
    shutil.copytree(ROOT / "projets" / "entrepot_corrige", dossier, ignore=shutil.ignore_patterns("__pycache__"))
    texte = (dossier / fichier).read_text(encoding="utf-8")
    for avant, apres in remplacements:
        assert avant in texte, f"{avant!r} introuvable dans {fichier}"
        texte = texte.replace(avant, apres)
    (dossier / fichier).write_text(texte, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    etape = next(e for e in PROJET.etapes if e.module == Path(fichier).stem)
    return evaluer_etape(nom, etape, PROJET.etapes)


_AVEC_CONNEXION = """    with connexion:
        changees = ranger(connexion, lecture.lignes)"""
_UPSERT_CONTRATS = """ON CONFLICT (client) DO UPDATE SET
    debut = excluded.debut, fin = excluded.fin, contrat = excluded.contrat, paiement = excluded.paiement,
    internet = excluded.internet, assistance = excluded.assistance, senior = excluded.senior,
    produits = excluded.produits, mensualite = excluded.mensualite, maj = excluded.maj
WHERE excluded.maj > contrats.maj"""

ERREURS_CLASSIQUES = [
    # Étape 1 — lire le CRM
    ("contrats.py", [('encoding="utf-8-sig"', 'encoding="utf-8"')], "utf-8-sig"),
    ("contrats.py", [("{colonne: champ.strip() for", "{colonne: champ for")], "fichier sain donne des rejets"),
    ("contrats.py", [('return datetime.strptime(texte, "%d/%m/%Y").date()',
                      'return datetime.strptime(texte, "%d/%m/%Y")')], "un datetime .* au lieu d'une date"),
    ("contrats.py", [("            except ValueError as exc:\n                rejets.append(Rejet(chemin.name, numero, str(exc)))",
                      "            except KeyError as exc:\n                rejets.append(Rejet(chemin.name, numero, str(exc)))")],
     "fait planter lire_contrats"),
    ("contrats.py", [("numero = lecteur.line_num", "numero = lecteur.line_num - 1")], "au lieu de 3"),
    ("contrats.py", [("if not math.isfinite(mensualite) or mensualite <= 0:", "if mensualite <= 0:")], "« nan »"),
    ("contrats.py", [("if not math.isfinite(mensualite) or mensualite <= 0:",
                      "if not math.isfinite(mensualite) or mensualite < 0:")], "STRICTEMENT positive"),
    ("contrats.py", [("                continue  # ligne vide", "                pass  # ligne vide")], "lignes vides"),
    ("contrats.py", [("rejets.append(Rejet(chemin.name, numero, str(exc)))",
                      "rejets.append(Rejet(str(chemin), numero, str(exc)))")], r"Path\(chemin\)\.name"),
    ("contrats.py", [("if tuple(entete) != COLONNES_CRM:", "if not entete:")], "devrait lever ValueError"),
    ("contrats.py", [('if not re.fullmatch(r"[0-9]+", valeurs["produits"]) or int(valeurs["produits"]) < 1:',
                      'if int(valeurs["produits"]) < 1:')], "ne cite pas « produits »"),
    ("contrats.py", [("    if fin is not None and fin < debut:\n", "    if False:\n")], "avant le début"),
    ("contrats.py", [('valeurs["mensualite"].replace(",", ".")', 'valeurs["mensualite"]')],
     "fichier sain donne des rejets"),
    ("contrats.py", [("for colonne, permises in MODALITES.items():", "for colonne, permises in []:")], "Mensual"),
    # Étape 2 — lire le support
    ("tickets.py", [("            if not texte.strip():\n                continue\n", "")], "ligne vide"),
    ("tickets.py", [("if deja is None or ticket.maj >= deja.maj:", "if True:")], "dernière LUE"),
    ("tickets.py", [("derniers[ticket.ticket] = ticket", "derniers[ticket.ticket + str(numero)] = ticket")],
     "qu'une fois"),
    ("tickets.py", [("[derniers[cle] for cle in sorted(derniers)]", "list(derniers.values())")], "triés"),
    ("tickets.py", [("if not isinstance(objet[cle], str) or not re.fullmatch", "if not re.fullmatch")],
     "fait planter lire_tickets"),
    ("tickets.py", [("except json.JSONDecodeError as exc:", "except KeyError as exc:")], "fait planter lire_tickets"),
    ("tickets.py", [("    if maj < ouvert:\n", "    if False:\n")], "avant l'ouverture"),
    ("tickets.py", [("enumerate(f, start=1)", "enumerate(f)")], "au lieu de 2"),
    ("tickets.py", [("    if not isinstance(objet, dict):\n", "    if False:\n")], "fait planter lire_tickets"),
    # Étape 3 — ranger dans la base
    ("chargement.py", [("INSERT INTO contrats (client", "INSERT OR REPLACE INTO contrats (client"),
                       (_UPSERT_CONTRATS, "")], "INSERT OR REPLACE"),
    ("chargement.py", [(_UPSERT_CONTRATS, "")], "IntegrityError"),
    ("chargement.py", [(_UPSERT_CONTRATS, "ON CONFLICT (client) DO NOTHING")], "DO NOTHING"),
    ("chargement.py", [("    return connexion.total_changes - avant  # pas de commit",
                        "    connexion.commit()\n    return connexion.total_changes - avant  # pas de commit")],
     "a validé la transaction"),
    ("chargement.py", [("    return connexion.total_changes - avant  # pas de commit",
                        "    return len(contrats)  # pas de commit")], "renvoie 2, attendu 1"),
    ("chargement.py", [("WHERE excluded.maj > contrats.maj", "WHERE excluded.maj >= contrats.maj")],
     "renvoie 3 au lieu de 0"),
    ("chargement.py", [("c.senior, c.produits, c.mensualite, _texte(c.maj))", "c.senior, c.produits, c.mensualite, c.maj)")],
     "ESPACE"),
    ("chargement.py", [("_texte(c.fin) if c.fin else None", "str(c.fin)")], "NULL"),
    ("chargement.py", [("contrat = excluded.contrat, paiement = excluded.paiement,", "contrat = excluded.contrat,")],
     "partiellement"),
    ("chargement.py", [("WHERE excluded.maj > tickets.maj", "WHERE excluded.ouvert > tickets.ouvert")], "T000001 mis à jour"),
    # Étape 4 — charger un fichier
    ("fichier.py", [("    with connexion:\n", "    if True:\n")], "AUTRE connexion"),
    ("fichier.py", [("    with connexion:\n", "    if True:\n"),
                    ("            (nom, signature, datetime.now().isoformat(timespec=\"seconds\"), changees, len(lecture.rejets)),\n        )\n",
                     "            (nom, signature, datetime.now().isoformat(timespec=\"seconds\"), changees, len(lecture.rejets)),\n        )\n"
                     "        connexion.commit()\n")], "transaction est restée ouverte"),
    ("fichier.py", [(_AVEC_CONNEXION, """    connexion.execute("INSERT OR REPLACE INTO fichiers VALUES (?, ?, '2026-01-01T00:00:00', 0, 0)", (nom, signature))
    connexion.commit()
    with connexion:
        changees = ranger(connexion, lecture.lignes)""")], "noté dans `fichiers`"),
    ("fichier.py", [(_AVEC_CONNEXION, """    try:
        changees = ranger(connexion, lecture.lignes)
    except Exception:
        connexion.rollback()
        return Bilan(nom, CHARGE, 0, list(lecture.rejets))
    with connexion:""")], "avalée"),
    ("fichier.py", [("> SEUIL_REJETS:", ">= SEUIL_REJETS:")], "`>`, pas `>=`"),
    ("fichier.py", [("if lues and len(lecture.rejets) / lues > SEUIL_REJETS:",
                     "if len(lecture.rejets) / lues > SEUIL_REJETS:")], "diviser par zéro"),
    ("fichier.py", [("if deja is not None and deja[0] == signature:", "if deja is not None:")], "renvoyé corrigé"),
    ("fichier.py", [('"ON CONFLICT (nom) DO UPDATE SET empreinte = excluded.empreinte, charge_le = excluded.charge_le, "\n'
                     '            "lignes = excluded.lignes, rejets = excluded.rejets"', '""')], "son nom y est déjà"),
    ("fichier.py", [("        return Bilan(nom, REFUSE, 0, list(lecture.rejets))", "        pass")], "20 %"),
    ("fichier.py", [("    signature = empreinte(chemin)\n", "    lire(chemin)\n    signature = empreinte(chemin)\n")],
     "a été relu"),
    ("fichier.py", [("datetime.now().isoformat(timespec=\"seconds\")", "'hier soir'")], "charge_le"),
    # Étape 5 — la table à date
    ("table.py", [("LEFT JOIN tickets AS t", "JOIN tickets AS t")], "LEFT JOIN"),
    ("table.py", [("COUNT(t.ticket) AS tickets", "COUNT(*) AS tickets")], r"COUNT\(\*\)"),
    ("table.py", [("AND date(t.ouvert) > :depuis\n      AND date(t.ouvert) <= :photo",
                   "AND t.ouvert > :depuis\n      AND t.ouvert <= :photo")], "en TEXTE"),
    ("table.py", [("      AND date(t.ouvert) <= :photo\n", "")], "fuite du futur"),
    ("table.py", [("      AND date(t.ouvert) > :depuis\n", "")], "deux ans"),
    ("table.py", [("AND date(t.ouvert) > :depuis", "AND date(t.ouvert) >= :depuis")], "365 jours pile"),
    ("table.py", [("(c.fin IS NULL OR c.fin > :photo)", "(c.fin IS NULL OR c.fin >= :photo)")], "le jour même"),
    ("table.py", [("  AND (c.fin IS NULL OR c.fin > :photo)\n", "")], "bien avant la photo"),
    ("table.py", [("WHERE c.debut <= :photo\n  AND (c.fin", "WHERE (c.fin")], "arrivés après la photo"),
    ("table.py", [("part = fin is not None and date.fromisoformat(fin) <= fin_horizon", "part = fin is not None")],
     "APRÈS l'horizon"),
    ("table.py", [("date.fromisoformat(fin) <= fin_horizon", "date.fromisoformat(fin) < fin_horizon")],
     "dernier jour de l'horizon"),
    ("table.py", [("    if fin_horizon > connue_jusqu_au:\n", "    if False:\n")], "trop récente"),
    ("table.py", [("anciennete = max(1, mois_entre(", "anciennete = max(0, mois_entre(")], "dix jours"),
    ("table.py", [("anciennete = max(1, mois_entre(date.fromisoformat(debut), photo))",
                   "anciennete = max(1, (photo - date.fromisoformat(debut)).days // 30)")], "tenure_months vaut"),
    ("table.py", [("round(mensualite * anciennete, 2)", "mensualite")], "total_charges"),
    ("table.py", [("    if connue is None:\n", "    if False:\n")], "base sans contrat"),
    ("table.py", [("LEFT JOIN tickets AS t\n       ON t.client = c.client\n      AND date(t.ouvert) > :depuis\n"
                   "      AND date(t.ouvert) <= :photo\nWHERE c.debut <= :photo",
                   "LEFT JOIN tickets AS t\n       ON t.client = c.client\nWHERE c.debut <= :photo\n"
                   "  AND date(t.ouvert) > :depuis\n      AND date(t.ouvert) <= :photo")], "dans le WHERE"),
    ("table.py", [("fin_horizon = photo + timedelta(days=horizon)", "fin_horizon = photo + timedelta(days=HORIZON_JOURS)")],
     "horizon=30"),
    ("table.py", [("contrat, paiement, internet, assistance, senior, int(part)))",
                   "paiement, contrat, internet, assistance, senior, int(part)))")], "contract vaut 'Prélèvement'"),
    ("table.py", [("    return table.astype(", "    return table.sort_values('customer_id', ascending=False).astype(")],
     "triés par customer_id"),
    # Étape 6 — le programme
    ("cli.py", [("    if not dossier.is_dir():  # AVANT", "    ouvrir_base(args.base).close()\n    if not dossier.is_dir():  # AVANT")],
     "base vide"),
    ("cli.py", [("    return 1 if comptes[REFUSE] else 0", "    return 0")], "doit renvoyer 1"),
    ("cli.py", [("    if tous_les_rejets:\n        with open(", "    if True:\n        with open(")], "n'écrivez pas"),
    ("cli.py", [("table.to_csv(sortie, index=False)", "table.to_csv(sortie)")], "colonne d'index"),
    ("cli.py", [("    if not Path(args.base).is_file():  # ouvrir_base", "    if False:  # ouvrir_base")],
     "créé une base vide"),
    ("cli.py", [("p.is_file() and (", "p.is_file() or (")], "attendu 0"),
    ("cli.py", [("                print(f\"⏭️  {bilan.fichier:<{largeur}}  déjà chargé\")",
                 "                print(f\"⏭️  {bilan.fichier:<{largeur}}  rien à faire\")")], "déjà chargé"),
    ("cli.py", [("    sortie.parent.mkdir(parents=True, exist_ok=True)\n", "")], None),
    ("cli.py", [("        tous_les_rejets += bilan.rejets\n", "        tous_les_rejets += bilan.rejets if bilan.statut == CHARGE else []\n")],
     "crm_2026-10-15.csv \\(refusé\\)"),
]


@pytest.mark.parametrize("fichier, remplacements, motif", ERREURS_CLASSIQUES,
                         ids=[f"{f}-{i}" for i, (f, _, _) in enumerate(ERREURS_CLASSIQUES)])
def test_chaque_erreur_classique_recoit_le_bon_message(tmp_path, monkeypatch, fichier, remplacements, motif):
    resultat = _variante(tmp_path, monkeypatch, fichier, remplacements)
    assert resultat.statut in (ECHEC, ERREUR), f"erreur non détectée : {remplacements}"
    if motif is not None:
        assert re.search(motif, resultat.message), resultat.message


# ---------------------------------------------------------------------------- Étape 7 : le test de l'élève

_SCENARIO = """
    base = _base([_contrat("C00001", "2024-01-15"), _contrat("C00002", "2025-12-30"),
                  _contrat("C00003", "2020-01-01", fin="2026-06-30")])
    table = construire_table(base, date(2026, 6, 30))
"""


def _module_de_test(corps: str) -> types.ModuleType:
    """Un mes_tests.py « d'apprenant·e » : les outils du corrigé, et `tester_construire_table` = `corps`."""
    import projets.entrepot_corrige.mes_tests as reference
    module = types.ModuleType("projets.entrepot_corrige.mes_tests_eleve")
    module.__package__ = "projets.entrepot_corrige"
    module.__dict__.update({k: v for k, v in vars(reference).items() if not k.startswith("__")})
    exec("def tester_construire_table(construire_table):\n" + corps, module.__dict__)  # noqa: S102
    return module


@pytest.mark.parametrize("corps, motif", [
    ("    pass\n", "TOUTES les versions boguées"),
    (_SCENARIO + "    assert list(table.columns) == list(COLONNES_TABLE)\n", "TOUTES les versions boguées"),
    (_SCENARIO + "    assert list(table['customer_id']) == ['C00001', 'C00002']\n", r"\d+/13 bugs attrapés"),
    (_SCENARIO + "    assert len(table) == 3\n", "ÉCHOUE sur la version JUSTE"),
    ("    construire_table('base.db', '2026-06-30')\n", "plante sur la version JUSTE"),
])
def test_le_correcteur_de_l_etape_7(corps, motif):
    module = _module_de_test(corps)
    module.COLONNES_TABLE = COLONNES_TABLE
    with pytest.raises(AssertionError, match=motif):
        checks.verifier_tester_construire_table(module)


def test_l_etape_7_accepte_un_autre_test_juste():
    """Un test écrit autrement (une table attendue entière, comparée d'un bloc) est accepté."""
    corps = '''
    photo = date(2026, 6, 30)
    base = _base(
        [_contrat("C00001", "2024-01-15"), _contrat("C00002", "2025-12-30", mensualite=40.0),
         _contrat("C00003", "2020-01-01", fin="2026-06-30"), _contrat("C00004", "2026-07-01"),
         _contrat("C00005", "2023-05-10", fin="2026-09-28"), _contrat("C00006", "2023-05-10", fin="2026-09-29"),
         _contrat("C00007", "2026-06-20"), _contrat("C00008", "2019-03-01", fin="2025-11-02"),
         _contrat("C00009", "2022-02-10")],
        [_ticket("T000001", "C00002", "2026-06-30T15:00:00"), _ticket("T000002", "C00005", "2026-07-01T09:00:00"),
         _ticket("T000003", "C00006", "2025-06-30T10:00:00"), _ticket("T000004", "C00009", "2025-07-01T10:00:00"),
         _ticket("T000005", "C00009", "2024-02-01T10:00:00")])
    obtenu = construire_table(base, photo)[["customer_id", "tenure_months", "support_calls", "churn"]]
    attendu = [["C00001", 29, 0, 0], ["C00002", 6, 1, 0], ["C00005", 37, 0, 1], ["C00006", 37, 0, 0],
               ["C00007", 1, 0, 0], ["C00009", 52, 1, 0]]
    assert obtenu.values.tolist() == attendu, obtenu.values.tolist()
    for trop_tot in (date(2026, 10, 3),):
        try:
            construire_table(base, trop_tot)
        except ValueError:
            continue
        raise AssertionError("photo trop récente acceptée")
'''
    checks.verifier_tester_construire_table(_module_de_test(corps))


# ---------------------------------------------------------------------------- D'autres solutions justes sont acceptées

AUTRES_SOLUTIONS = {
    "chargement.py": """\
import sqlite3
from .modeles import Contrat, Ticket

def charger_contrats(connexion, contrats):
    n = 0
    for c in contrats:
        maj = c.maj.isoformat(timespec="seconds")
        ligne = connexion.execute("SELECT maj FROM contrats WHERE client = ?", (c.client,)).fetchone()
        if ligne is None or maj > ligne[0]:
            connexion.execute("INSERT OR REPLACE INTO contrats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (c.client, c.debut.isoformat(), c.fin.isoformat() if c.fin else None, c.contrat,
                               c.paiement, c.internet, c.assistance, c.senior, c.produits, c.mensualite, maj))
            n += 1
    return n

def charger_tickets(connexion, tickets):
    n = 0
    for t in tickets:
        maj = t.maj.isoformat(timespec="seconds")
        ligne = connexion.execute("SELECT maj FROM tickets WHERE ticket = ?", (t.ticket,)).fetchone()
        if ligne is None or maj > ligne[0]:
            connexion.execute("INSERT OR REPLACE INTO tickets VALUES (?, ?, ?, ?)",
                              (t.ticket, t.client, t.ouvert.isoformat(timespec="seconds"), maj))
            n += 1
    return n
""",
    "table.py": """\
import sqlite3
from datetime import date, datetime, timedelta
import pandas as pd
from .modeles import COLONNES_TABLE, FENETRE_JOURS, HORIZON_JOURS, mois_entre

def construire_table(connexion, photo, horizon=HORIZON_JOURS):
    contrats = pd.read_sql("SELECT * FROM contrats", connexion)
    tickets = pd.read_sql("SELECT * FROM tickets", connexion)
    if contrats.empty:
        raise ValueError("base vide")
    connue = pd.to_datetime(contrats["maj"]).max().date()
    limite = photo + timedelta(days=horizon)
    if limite > connue:
        raise ValueError("trop récent")
    debut = pd.to_datetime(contrats["debut"]).dt.date
    fin = pd.to_datetime(contrats["fin"]).dt.date
    actifs = contrats[(debut <= photo) & (contrats["fin"].isna() | (fin > photo))].copy()
    jour = pd.to_datetime(tickets["ouvert"]).dt.date
    fenetre = tickets[(jour > photo - timedelta(days=FENETRE_JOURS)) & (jour <= photo)]
    appels = fenetre.groupby("client").size()
    actifs["support_calls"] = actifs["client"].map(appels).fillna(0).astype(int)
    actifs["tenure_months"] = [max(1, mois_entre(date.fromisoformat(d), photo)) for d in actifs["debut"]]
    actifs["total_charges"] = (actifs["mensualite"] * actifs["tenure_months"]).round(2)
    actifs["churn"] = [int(f is not None and isinstance(f, str) and date.fromisoformat(f) <= limite) for f in actifs["fin"]]
    table = actifs.rename(columns={"client": "customer_id", "mensualite": "monthly_charges", "produits": "num_products",
                                   "contrat": "contract", "paiement": "payment_method", "internet": "internet_service",
                                   "assistance": "tech_support"})
    return table.sort_values("customer_id")[list(COLONNES_TABLE)].reset_index(drop=True)
""",
    "fichier.py": """\
from datetime import datetime
from pathlib import Path
from .chargement import charger_contrats, charger_tickets
from .contrats import lire_contrats
from .modeles import CHARGE, DEJA_CHARGE, REFUSE, SEUIL_REJETS, Bilan, empreinte
from .tickets import lire_tickets

def charger_fichier(connexion, chemin):
    chemin = Path(chemin)
    nom = chemin.name
    if nom.startswith("crm_") and nom.endswith(".csv"):
        lire, ranger = lire_contrats, charger_contrats
    elif nom.startswith("support_") and nom.endswith(".jsonl"):
        lire, ranger = lire_tickets, charger_tickets
    else:
        raise ValueError(nom)
    e = empreinte(chemin)
    if connexion.execute("SELECT 1 FROM fichiers WHERE nom = ? AND empreinte = ?", (nom, e)).fetchone():
        return Bilan(nom, DEJA_CHARGE, 0, [])
    lu = lire(chemin)
    total = len(lu.lignes) + len(lu.rejets)
    if total and len(lu.rejets) > SEUIL_REJETS * total:
        return Bilan(nom, REFUSE, 0, lu.rejets)
    try:
        n = ranger(connexion, lu.lignes)
        connexion.execute("DELETE FROM fichiers WHERE nom = ?", (nom,))
        connexion.execute("INSERT INTO fichiers VALUES (?, ?, ?, ?, ?)",
                          (nom, e, datetime.now().isoformat(timespec="seconds"), n, len(lu.rejets)))
        connexion.commit()
    except BaseException:
        connexion.rollback()
        raise
    return Bilan(nom, CHARGE, n, lu.rejets)
""",
}


@pytest.mark.parametrize("fichier", sorted(AUTRES_SOLUTIONS))
def test_une_autre_solution_juste_est_acceptee(tmp_path, monkeypatch, fichier):
    """Vérifier en Python avant un INSERT OR REPLACE, construire la table avec pandas, gérer la transaction
    à la main avec commit et rollback : d'autres façons d'écrire juste, que le correcteur accepte."""
    nom = f"entrepot_autre_{next(_compteur)}"
    dossier = tmp_path / nom
    shutil.copytree(ROOT / "projets" / "entrepot_corrige", dossier, ignore=shutil.ignore_patterns("__pycache__"))
    (dossier / fichier).write_text(AUTRES_SOLUTIONS[fichier], encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for etape in PROJET.etapes:
        resultat = evaluer_etape(nom, etape, PROJET.etapes)
        assert resultat.statut == REUSSI, (etape.module, resultat.message)

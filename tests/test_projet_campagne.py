"""Garantit que le projet 3 (mesurer l'effet de la campagne) est faisable et bien corrigé.

Ces tests NE notent PAS votre projet (pour ça : `python scripts/projet.py --projet 3`) ; ils vérifient
le corrigé, les données d'exemple et que chaque erreur classique reçoit le bon message.
"""

import csv
import itertools
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projets import checks_campagne as checks  # noqa: E402
from projets.campagne_corrige import cli, fichiers, mesure, tirage  # noqa: E402
from projets.campagne_corrige.modeles import GROUPE_OFFRE, GROUPE_TEMOIN, TOUTES  # noqa: E402
from projets.verif import A_FAIRE, ECHEC, REUSSI, evaluer, evaluer_etape  # noqa: E402

PROJET = checks.PROJET
PLAN = ROOT / "projets" / "exemple_plan.csv"
RESULTATS = ROOT / "projets" / "exemple_resultats.csv"
_compteur = itertools.count()


def _python(*args, cwd=ROOT):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          timeout=120)


# ---------------------------------------------------------------------------- Le projet est cohérent


@pytest.mark.parametrize("etape", PROJET.etapes, ids=lambda e: e.module)
def test_le_corrige_reussit(etape):
    resultat = evaluer_etape(PROJET.paquet_corrige, etape, PROJET.etapes)
    assert resultat.statut == REUSSI, resultat.message


def test_le_squelette_est_entierement_a_faire():
    assert [r.statut for r in evaluer(PROJET)] == [A_FAIRE] * len(PROJET.etapes)


@pytest.mark.parametrize("fichier", ["modeles.py", "fichiers.py", "__main__.py"])
def test_les_fichiers_deja_ecrits_sont_identiques(fichier):
    eleve = ROOT / "projets" / "campagne" / fichier
    corrige = ROOT / "projets" / "campagne_corrige" / fichier
    assert eleve.read_text(encoding="utf-8") == corrige.read_text(encoding="utf-8")


def test_le_plan_d_exemple_a_le_format_du_projet_2():
    ciblages = fichiers.lire_plan(PLAN)
    assert len(ciblages) == 2000
    assert {c.offre for c in ciblages} == {"APPEL", "SUPPORT", "REMISE", "ENGAGEMENT"}
    with PLAN.open(encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == ["customer_id", "offre", "cout", "proba_churn", "gain_attendu"]


def test_le_tirage_est_le_meme_dans_tous_les_processus():
    """Contrairement à hash(), sha256 ne dépend pas de PYTHONHASHSEED : le groupe d'un client est stable."""
    code = "import sys; sys.path.insert(0, '.'); from projets.campagne_corrige.tirage import tirage; " \
           "print(repr(tirage('C10042', '2026-11')), hash('C10042'))"
    sorties = {_python("-c", code).stdout.split()[0] for _ in range(3)}
    assert sorties == {repr(tirage.tirage("C10042", "2026-11"))}


# ---------------------------------------------------------------------------- Le programme terminé


def test_le_programme_termine_de_bout_en_bout(tmp_path):
    base, contacts = tmp_path / "campagne.db", tmp_path / "a_contacter.csv"
    paquet = "projets.campagne_corrige"
    res = _python("-m", paquet, "preparer", "--plan", str(PLAN), "--base", str(base), "--graine", "2026-11",
                  "--contacts", str(contacts))
    assert res.returncode == 0, res.stderr
    assert "2000 clients" in res.stdout and "1612" in res.stdout and "388" in res.stdout
    with contacts.open(encoding="utf-8", newline="") as f:
        assert sum(1 for _ in csv.DictReader(f)) == 1612

    res = _python("-m", paquet, "importer", "--base", str(base), "--resultats", str(RESULTATS))
    assert res.returncode == 0, res.stderr
    assert "1912" in res.stdout
    assert len(re.findall(r"ligne \d+ :", res.stderr)) == 3  # deux clients inconnus, une valeur illisible
    assert "C99999" in res.stderr and "C00042" in res.stderr and "« ? »" in res.stderr

    res = _python("-m", paquet, "bilan", "--base", str(base))
    assert res.returncode == 0, res.stderr
    blocs = {b.split("]")[0].lstrip("["): b for b in res.stdout.strip().split("\n\n")}
    assert list(blocs) == [TOUTES, "APPEL", "ENGAGEMENT", "REMISE", "SUPPORT"]
    # L'histoire que raconte l'exemple : ENGAGEMENT et SUPPORT marchent, APPEL et REMISE ne prouvent rien.
    for code in (TOUTES, "ENGAGEMENT", "SUPPORT"):
        assert "✅" in blocs[code], blocs[code]
    for code in ("APPEL", "REMISE"):
        assert "🤷" in blocs[code], blocs[code]


def test_le_squelette_explique_qu_il_n_est_pas_termine(tmp_path):
    res = _python("-m", "projets.campagne", "bilan", "--base", str(tmp_path / "c.db"))
    assert res.returncode == 1
    assert "scripts/projet.py --projet 3" in res.stderr and "Traceback" not in res.stderr


def test_le_script_de_progression_connait_le_projet_3():
    res = _python("scripts/projet.py", "--projet", "3", "--corrige")
    assert res.returncode == 0 and "7/7" in res.stdout and "Projet 3" in res.stdout and "Projet 2" not in res.stdout
    res = _python("scripts/projet.py", "--projet", "3", "--etape", "4")
    assert res.returncode == 0 and "mesure.py" in res.stdout and "tirage.py" not in res.stdout
    res = _python("scripts/projet.py", "--projet", "9")
    assert res.returncode != 0 and "Aucun projet n°9" in res.stderr


def test_relancer_preparer_ne_change_aucun_groupe(tmp_path):
    base = tmp_path / "c.db"
    argv = ["preparer", "--plan", str(PLAN), "--base", str(base), "--graine", "2026-11",
            "--contacts", str(tmp_path / "a.csv")]
    assert cli.main(argv) == 0
    with checks._fermee(checks.ref_base.ouvrir(base)) as conn:
        avant = dict(conn.execute("SELECT customer_id, groupe FROM campagne"))
    assert cli.main(argv[:-4] + ["--graine", "autre", "--part-temoin", "0.9", "--contacts", str(tmp_path / "b.csv")]) == 0
    with checks._fermee(checks.ref_base.ouvrir(base)) as conn:
        assert dict(conn.execute("SELECT customer_id, groupe FROM campagne")) == avant
        offre, temoin = mesure.regrouper(mesure.compter(conn))[TOUTES]
    assert (offre.groupe, temoin.groupe) == (GROUPE_OFFRE, GROUPE_TEMOIN)
    assert abs(temoin.clients / (offre.clients + temoin.clients) - 0.2) < 0.03


# ---------------------------------------------------------------------------- Les erreurs classiques


def _variante(tmp_path, monkeypatch, fichier: str, remplacements):
    """Le corrigé, avec UNE erreur classique introduite dans `fichier`, corrigé par le vrai moteur."""
    nom = f"campagne_variante_{next(_compteur)}"
    dossier = tmp_path / nom
    shutil.copytree(ROOT / "projets" / "campagne_corrige", dossier, ignore=shutil.ignore_patterns("__pycache__"))
    texte = (dossier / fichier).read_text(encoding="utf-8")
    for avant, apres in remplacements:
        assert avant in texte, f"{avant!r} introuvable dans {fichier}"
        texte = texte.replace(avant, apres)
    (dossier / fichier).write_text(texte, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    etape = next(e for e in PROJET.etapes if e.module == Path(fichier).stem)
    return evaluer_etape(nom, etape, PROJET.etapes)


_DIVISION = "int(empreinte[:16], 16) / 2**64"
_CONDITION = "GROUPE_TEMOIN if tirage(customer_id, graine) < part_temoin else GROUPE_OFFRE"

ERREURS_CLASSIQUES = [
    # Étape 1
    ("tirage.py", [(_DIVISION, "(hash(texte) % 2**64) / 2**64")], "hash()"),
    ("tirage.py", [("import hashlib", "import hashlib\nimport random"), (_DIVISION, "random.random()")], "random"),
    ("tirage.py", [('f"{graine}:{customer_id}"', 'f"{customer_id}:{graine}"')], "la graine D'ABORD"),
    ("tirage.py", [('f"{graine}:{customer_id}"', 'f"{graine}{customer_id}"')], "séparateur"),
    ("tirage.py", [("hashlib.sha256(", "hashlib.md5(")], "MD5"),
    ("tirage.py", [(_DIVISION, "int(empreinte[:8], 16) / 2**32")], "8 caractères"),
    ("tirage.py", [("empreinte[:16]", "empreinte[-16:]")], "DERNIERS"),
    ("tirage.py", [(_DIVISION, "int(empreinte[:16], 16) // 2**64")], "entier"),
    ("tirage.py", [(_CONDITION, "GROUPE_OFFRE if tirage(customer_id, graine) < part_temoin else GROUPE_TEMOIN")],
     "inversée"),
    ("tirage.py", [("if not 0 <= part_temoin <= 1:", "if part_temoin < 0 or part_temoin > 1:")], "nan"),
    ("tirage.py", [(_CONDITION, _CONDITION.replace("GROUPE_TEMOIN if", '"témoin" if'))], "constantes"),
    # Étape 2
    ("base.py", [('    conn.execute("PRAGMA foreign_keys = ON")\n', "")], "foreign_keys"),
    ("base.py", [("    with conn:\n", "    if True:\n")], "COMMIT"),
    ("base.py", [('"ON CONFLICT (customer_id) DO NOTHING",', '"",')], "ON CONFLICT"),
    ("base.py", [("DO NOTHING", "DO UPDATE SET groupe = excluded.groupe")], "changé de groupe"),
    ("base.py", [("return max(curseur.rowcount, 0)", "return len(lignes)")], "rowcount"),
    ("base.py", [('"ON CONFLICT (customer_id) DO NOTHING",', '"",'),
                 ('"INSERT INTO campagne', '"INSERT OR IGNORE INTO campagne')], "OR IGNORE"),
    ("base.py", [("    with conn:\n", "    if True:\n"),
                 ("    return max(curseur.rowcount, 0)", "    conn.commit()\n    return max(curseur.rowcount, 0)")],
     "Tout ou rien"),
    ("base.py", [("        curseur = conn.executemany(", "        for x in lignes:\n            curseur = conn.execute(\n"
                  "                \"INSERT INTO campagne VALUES ('%s', '%s', %s, %s, '%s') \" % x\n            )\n"
                  "        curseur = conn.executemany(")], "apostrophe"),
    # Étape 3
    ("resultats.py", [('encoding="utf-8-sig"', 'encoding="utf-8"')], "BOM"),
    ("resultats.py", [('valeur = (texte or "").strip().lower()', "valeur = texte.strip().lower()")], "ligne coupée"),
    ("resultats.py", [("except sqlite3.IntegrityError:", "except KeyError:")], "IntegrityError"),
    ("resultats.py", [("DO UPDATE SET parti = excluded.parti", "DO NOTHING")], "DERNIÈRE"),
    ("resultats.py", [("except sqlite3.IntegrityError:", "except sqlite3.Error:")], "panne"),
    ("resultats.py", [("                enregistres += 1", "                enregistres += 1\n                conn.commit()")],
     "UNE transaction"),
    # Étape 4
    ("mesure.py", [("LEFT JOIN", "JOIN")], "LEFT JOIN"),
    ("mesure.py", [("COUNT(r.customer_id)      AS avec_resultat", "COUNT(*) AS avec_resultat")], "COUNT\\(r.customer_id\\)"),
    ("mesure.py", [("COALESCE(SUM(r.parti), 0)", "SUM(r.parti)")], "COALESCE"),
    ("mesure.py", [("ORDER BY c.offre, c.groupe", "ORDER BY c.groupe, c.offre")], "mauvais ordre"),
    ("mesure.py", [("    return [Comptage(offre, groupe, clients, avec, partis, float(cout))\n"
                    "            for offre, groupe, clients, avec, partis, cout in conn.execute(_REQUETE)]",
                    "    codes = [o for (o,) in conn.execute('SELECT DISTINCT offre FROM campagne ORDER BY offre')]\n"
                    "    requete = _REQUETE.replace('GROUP BY', 'WHERE c.offre = ? GROUP BY')\n"
                    "    return [Comptage(*r[:5], float(r[5])) for code in codes for r in conn.execute(requete, (code,))]")],
     "N\\+1"),
    ("mesure.py", [("for code in sorted({c.offre for c in comptages}):",
                    "for code in dict.fromkeys(c.offre for c in comptages):")], "Ordre des clés"),
    ("mesure.py", [("par_cle.get((code, g), Comptage(code, g, 0, 0, 0, 0.0))", "par_cle.get((code, g))")],
     "REMISE n'a aucun témoin"),
    ("mesure.py", [("    comptages = list(comptages)", "    pass")], "générateur"),
    # Étape 6
    ("cli.py", [('required=True, metavar="COMMANDE"', 'metavar="COMMANDE"')], "required=True"),
    ("cli.py", [("    if chemin.exists():\n        return True", "    if True:\n        return True")], "preparer"),
    ("cli.py", [('print(f"Attention : {erreur}", file=sys.stderr)', 'print(f"Attention : {erreur}")')], "stderr"),
    ("cli.py", [("    except FileNotFoundError as exc:\n"
                 '        print(f"Erreur : fichier introuvable : {exc.filename}", file=sys.stderr)\n', "")],
     "FileNotFoundError non attrapée"),
    ("cli.py", [("    except sqlite3.Error as exc:", "    except KeyError as exc:")], "sqlite3"),
    ("cli.py", [("print(formater_ligne(nom, offre, temoin, mesure))", "print(formater_ligne(nom, temoin, offre, mesure))")],
     "le bloc"),
    ("cli.py", [("effet(offre, temoin, args.niveau)", "effet(offre, temoin)")], "le bloc"),
]


@pytest.mark.parametrize("fichier,remplacements,motif", ERREURS_CLASSIQUES,
                         ids=[f"{f}-{m}" for f, _, m in ERREURS_CLASSIQUES])
def test_chaque_erreur_classique_est_expliquee(tmp_path, monkeypatch, fichier, remplacements, motif):
    resultat = _variante(tmp_path, monkeypatch, fichier, remplacements)
    assert resultat.statut == ECHEC, (resultat.statut, resultat.message)
    assert re.search(motif, resultat.message), resultat.message


def _faux_module(**fonctions) -> types.ModuleType:
    module = types.ModuleType("projets.campagne_corrige.faux")
    module.__package__ = "projets.campagne_corrige"
    module.__dict__.update(fonctions)
    return module


@pytest.mark.parametrize("mutant,defaut,conseil", checks._MUTANTS, ids=[d[:40] for _, d, _ in checks._MUTANTS])
def test_etape5_chaque_bug_est_explique(mutant, defaut, conseil):
    motif = "ZeroDivisionError" if "ZeroDivisionError" in defaut else re.escape(defaut)
    with pytest.raises(AssertionError, match=motif):
        checks.verifier_statistiques(_faux_module(effet=mutant))


def test_etape7_un_test_trop_faible_est_refuse():
    def tester_faible(effet):
        e = effet(checks._modeles(cli).Comptage("T", "offre", 500, 400, 100, 0.0),
                  checks._modeles(cli).Comptage("T", "temoin", 120, 100, 40, 0.0))
        assert e.ecart > 0  # ne vérifie que le signe

    with pytest.raises(AssertionError, match="laisse passer"):
        checks.verifier_tester_effet(_faux_module(tester_effet=tester_faible))

    def tester_rien(effet):
        effet(checks._modeles(cli).Comptage("T", "offre", 5, 5, 1, 0.0),
              checks._modeles(cli).Comptage("T", "temoin", 5, 5, 2, 0.0))

    with pytest.raises(AssertionError, match="TOUTES les versions boguées"):
        checks.verifier_tester_effet(_faux_module(tester_effet=tester_rien))

    def tester_faux(effet):
        e = effet(checks._modeles(cli).Comptage("T", "offre", 500, 400, 100, 0.0),
                  checks._modeles(cli).Comptage("T", "temoin", 120, 100, 40, 0.0))
        assert e.ecart == 0.15  # 0.15000000000000002 en virgule flottante

    with pytest.raises(AssertionError, match="version JUSTE"):
        checks.verifier_tester_effet(_faux_module(tester_effet=tester_faux))


# ---------------------------------------------------------------------------- D'autres solutions justes passent

SOLUTIONS_DIFFERENTES = [
    ("tirage.py", [(_DIVISION, 'int.from_bytes(hashlib.sha256(texte.encode("utf-8")).digest()[:8], "big") / 2**64')]),
    ("tirage.py", [(_DIVISION, "int(empreinte, 16) / 2**256")]),
    ("base.py", [("    with conn:\n        # Les « ? »", "    try:\n        # Les « ? »"),
                 ("            lignes,\n        )\n",
                  "            lignes,\n        )\n        conn.commit()\n    except Exception:\n        conn.rollback()\n"
                  "        raise\n")]),
    ("resultats.py", [('"INSERT INTO resultats (customer_id, parti) VALUES (?, ?) "\n'
                       '                        "ON CONFLICT (customer_id) DO UPDATE SET parti = excluded.parti"',
                       '"INSERT OR REPLACE INTO resultats (customer_id, parti) VALUES (?, ?)"')]),
    ("resultats.py", [("                try:\n                    # « Upsert »",
                       "                try:\n"
                       "                    if conn.execute('SELECT 1 FROM campagne WHERE customer_id = ?', (identifiant,))"
                       ".fetchone() is None:\n                        raise sqlite3.IntegrityError('inconnu')\n"
                       "                    # « Upsert »")]),
    ("mesure.py", [("COALESCE(SUM(r.parti), 0)", "SUM(CASE WHEN r.parti = 1 THEN 1 ELSE 0 END)"),
                   ("COUNT(r.customer_id)", "SUM(r.customer_id IS NOT NULL)"),
                   ("GROUP BY c.offre, c.groupe", "group by 1, 2")]),
    ("statistiques.py", [("NormalDist().inv_cdf((1 + niveau) / 2)", "NormalDist().inv_cdf(1 - (1 - niveau) / 2)")]),
    ("cli.py", [('print(f"Plan lu        : {len(ciblages)} clients ({ajoutes} nouveaux dans {args.base})")',
                 'print(f"{len(ciblages)} clients lus, dont {ajoutes} nouveaux.")'),
                ('print(f"Groupe témoin  : {temoin.clients} clients à NE PAS contacter")',
                 'print(f"témoin : {temoin.clients}")')]),
]


@pytest.mark.parametrize("fichier,remplacements", SOLUTIONS_DIFFERENTES,
                         ids=[f"{f}-{i}" for i, (f, _) in enumerate(SOLUTIONS_DIFFERENTES)])
def test_une_autre_solution_juste_est_acceptee(tmp_path, monkeypatch, fichier, remplacements):
    """Le correcteur juge le comportement, pas la ressemblance avec le corrigé."""
    resultat = _variante(tmp_path, monkeypatch, fichier, remplacements)
    assert resultat.statut == REUSSI, resultat.message

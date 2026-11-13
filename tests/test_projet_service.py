"""Garantit que le projet 4 (le service des conseillers) est faisable et bien corrigé.

Ces tests NE notent PAS votre projet (pour ça : `python scripts/projet.py --projet 4`) ; ils vérifient
le corrigé, les données d'exemple, que le service démarre pour de vrai, et que chaque erreur classique
reçoit le bon message.
"""

import csv
import itertools
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projets import checks_service as checks  # noqa: E402
from projets.service_corrige import api, mes_tests  # noqa: E402
from projets.verif import A_FAIRE, REUSSI, evaluer, evaluer_etape  # noqa: E402

PROJET = checks.PROJET
CONTACTS = ROOT / "projets" / "exemple_contacts.csv"
CLES = "alice=cle-alice-0123456789,bruno=cle-bruno-0123456789,chloe=cle-chloe-0123456789,david=cle-david-0123456789"
_compteur = itertools.count()


def _python(*args, env=None, timeout=120):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          timeout=timeout, env={**os.environ, **(env or {})})


# ---------------------------------------------------------------------------- Le projet est cohérent


@pytest.mark.parametrize("etape", PROJET.etapes, ids=lambda e: e.module)
def test_le_corrige_reussit(etape):
    resultat = evaluer_etape(PROJET.paquet_corrige, etape, PROJET.etapes)
    assert resultat.statut == REUSSI, resultat.message


def test_le_squelette_est_entierement_a_faire():
    assert [r.statut for r in evaluer(PROJET)] == [A_FAIRE] * len(PROJET.etapes)


@pytest.mark.parametrize("fichier", ["modeles.py", "fichiers.py", "__main__.py"])
def test_les_fichiers_deja_ecrits_sont_identiques(fichier):
    eleve = ROOT / "projets" / "service" / fichier
    corrige = ROOT / "projets" / "service_corrige" / fichier
    assert eleve.read_text(encoding="utf-8") == corrige.read_text(encoding="utf-8")


def test_le_squelette_fournit_les_memes_outils_de_test():
    """`_preparer`, `ALICE`, `BRUNO`, `_ids` : l'étape 7 du squelette part des mêmes outils que le corrigé."""
    eleve = (ROOT / "projets" / "service" / "mes_tests.py").read_text(encoding="utf-8")
    corrige = (ROOT / "projets" / "service_corrige" / "mes_tests.py").read_text(encoding="utf-8")
    debut = corrige.index("from __future__")
    outils = corrige[debut:corrige.index("def tester_service")]
    assert outils in eleve


def test_la_liste_d_exemple_est_celle_du_projet_3(tmp_path):
    """exemple_contacts.csv est exactement ce qu'écrit `preparer` (projet 3) avec la graine 2026-11."""
    contacts = tmp_path / "a_contacter.csv"
    res = _python("-m", "projets.campagne_corrige", "preparer", "--plan", "projets/exemple_plan.csv",
                  "--base", str(tmp_path / "c.db"), "--graine", "2026-11", "--contacts", str(contacts))
    assert res.returncode == 0, res.stderr
    assert contacts.read_text(encoding="utf-8") == CONTACTS.read_text(encoding="utf-8")
    with CONTACTS.open(encoding="utf-8", newline="") as f:
        lignes = list(csv.DictReader(f))
    assert len(lignes) == 1612 and set(lignes[0]) == {"customer_id", "offre"}


# ---------------------------------------------------------------------------- Le programme terminé


def test_le_programme_termine_de_bout_en_bout(tmp_path):
    env = {"SERVICE_CLES": CLES, "SERVICE_BASE": str(tmp_path / "service.db")}
    res = _python("-m", "projets.service_corrige", "importer", "--contacts", str(CONTACTS), env=env)
    assert res.returncode == 0, res.stderr
    assert "1612 (1612 nouveaux" in res.stdout
    for nom in ("alice", "bruno", "chloe", "david"):
        assert f"{nom:<12}{403:>13}" in res.stdout  # 1 612 contacts = 4 × 403
    assert "cle-" not in res.stdout + res.stderr

    res = _python("-m", "projets.service_corrige", "importer", "--contacts", str(CONTACTS), env=env)
    assert res.returncode == 0 and "(0 nouveaux" in res.stdout

    res = _python("-m", "projets.service_corrige", "avancement", env={**env, "SERVICE_CLES": "alice=secret-court"})
    assert res.returncode == 1 and "SERVICE_CLES" in res.stderr and "secret-court" not in res.stderr
    assert "Traceback" not in res.stderr


def _port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_le_service_repond_pour_de_vrai(tmp_path):
    """`servir` démarre un vrai serveur uvicorn : un conseiller lit sa liste et enregistre un appel."""
    import httpx

    env = {**os.environ, "SERVICE_CLES": CLES, "SERVICE_BASE": str(tmp_path / "service.db")}
    assert _python("-m", "projets.service_corrige", "importer", "--contacts", str(CONTACTS), env=env).returncode == 0
    port = _port_libre()
    serveur = subprocess.Popen([sys.executable, "-m", "projets.service_corrige", "servir", "--port", str(port)],
                               cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               encoding="utf-8")
    try:
        adresse = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                if httpx.get(f"{adresse}/sante", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        else:
            pytest.fail("le service n'a pas démarré")
        alice = {"X-Cle-API": "cle-alice-0123456789"}
        assert httpx.get(f"{adresse}/contacts").status_code == 401
        page = httpx.get(f"{adresse}/contacts", params={"taille": 3}, headers=alice).json()
        premier = page["contacts"][0]["customer_id"]
        assert len(page["contacts"]) == 3 and page["suivant"] == page["contacts"][-1]["customer_id"]
        reponse = httpx.put(f"{adresse}/contacts/{premier}/appel", headers=alice,
                            json={"statut": "accepte", "version": 1, "commentaire": "passe à l'annuel"})
        assert reponse.status_code == 200 and reponse.json()["version"] == 2
        reponse = httpx.put(f"{adresse}/contacts/{premier}/appel", headers=alice, json={"statut": "refuse", "version": 1})
        assert reponse.status_code == 409 and reponse.json()["version_actuelle"] == 2
        assert httpx.get(f"{adresse}/avancement", headers=alice).json()["alice"]["accepte"] == 1
    finally:
        serveur.terminate()
        _, erreurs = serveur.communicate(timeout=10)
    assert "cle-alice-0123456789" not in erreurs


def test_le_squelette_explique_qu_il_n_est_pas_termine(tmp_path):
    res = _python("-m", "projets.service", "avancement", env={"SERVICE_CLES": CLES})
    assert res.returncode == 1
    assert "scripts/projet.py --projet 4" in res.stderr and "Traceback" not in res.stderr


def test_le_script_de_progression_connait_le_projet_4():
    res = _python("scripts/projet.py", "--projet", "4", "--corrige")
    assert res.returncode == 0 and "7/7" in res.stdout and "Projet 4" in res.stdout and "Projet 3" not in res.stdout
    assert "requête refusée" not in res.stdout + res.stderr  # le journal de l'API ne pollue pas l'affichage
    res = _python("scripts/projet.py", "--projet", "4", "--etape", "5")
    assert res.returncode == 0 and "api.py" in res.stdout and "config.py" not in res.stdout


def test_chaque_version_boguee_de_l_api_est_attrapee_par_le_test_du_corrige():
    for fichier, avant, apres, _, _ in checks._MUTANTS:
        creer_app = checks._mutant(fichier, avant, apres)
        with pytest.raises(Exception):
            mes_tests.tester_service(creer_app)
    mes_tests.tester_service(api.creer_app)  # et la version juste passe


# ---------------------------------------------------------------------------- Les erreurs classiques


def _variante(tmp_path, monkeypatch, fichier: str, remplacements):
    """Le corrigé, avec UNE erreur classique introduite dans `fichier`, corrigé par le vrai moteur."""
    nom = f"service_variante_{next(_compteur)}"
    dossier = tmp_path / nom
    shutil.copytree(ROOT / "projets" / "service_corrige", dossier, ignore=shutil.ignore_patterns("__pycache__"))
    texte = (dossier / fichier).read_text(encoding="utf-8")
    for avant, apres in remplacements:
        assert texte.count(avant) == 1, f"{avant!r} introuvable (ou ambigu) dans {fichier}"
        texte = texte.replace(avant, apres)
    (dossier / fichier).write_text(texte, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    etape = next(e for e in PROJET.etapes if e.module == Path(fichier).stem)
    return evaluer_etape(nom, etape, PROJET.etapes)


_UPDATE_PARAMS = ('AND conseiller = ? AND version = ?",\n            (statut, commentaire, customer_id, conseiller, version),')
_COMPARE = 'if secrets.compare_digest(recue, cle.encode("utf-8")):'
_CONNEXION = "        conn = ouvrir(config.base)\n        try:\n            yield conn\n        finally:\n            conn.close()\n"

ERREURS_CLASSIQUES = [
    # Étape 1 : la configuration
    ("config.py", [('nom, egal, cle = entree.partition("=")',
                    'morceaux = entree.split("=")\n        nom, egal, cle = morceaux[0], "=" * (len(morceaux) > 1), '
                    '(morceaux + [""])[1]')], "partition"),
    ("config.py", [('f"SERVICE_CLES : la clé de {nom!r} est trop courte "', 'f"SERVICE_CLES : la clé {cle!r} est trop courte "')],
     "CONTIENT une clé"),
    ("config.py", [("    erreurs: list[str] = []\n\n    cles =",
                    "    env = {**env, **__import__('os').environ}\n    erreurs: list[str] = []\n\n    cles =")], "os.environ"),
    ("config.py", [('raise ConfigInvalide("Configuration invalide : "', 'raise ValueError("Configuration invalide : "')],
     "ConfigInvalide"),
    ("config.py", [('" ; ".join(erreurs))', '" ; ".join(erreurs[:1]))')], "DEUX erreurs"),
    ("config.py", [('texte = (env.get(nom) or "").strip()', 'texte = (env[nom] or "").strip()')], "env.get"),
    ("config.py", [('    texte = (env.get(nom) or "").strip()\n    return texte or None', "    return env.get(nom)")],
     "sans les espaces"),
    ("config.py", [("taille_page = int(texte)", "taille_page = int(float(texte))")], "2.5"),
    # Étape 2 : la base et la répartition
    ("base.py", [("sqlite3.connect(chemin, check_same_thread=False)", "sqlite3.connect(chemin)")], "check_same_thread"),
    ("base.py", [('        conn.execute("PRAGMA foreign_keys = ON")  # à redemander à CHAQUE connexion\n', "")],
     "foreign_keys"),
    ("base.py", [("min(charge, key=lambda nom: (charge[nom], nom))", "min(charge, key=lambda nom: charge[nom])")],
     "ALPHABÉTIQUE"),
    ("base.py", [("            charge[conseiller] = nombre\n", "            pass\n")], "déjà dans la base"),
    ("base.py", [("        charge[conseiller] += 1\n", "")], "MOINS"),
    ("base.py", [("        if customer_id in deja_la:\n",
                  "        if customer_id in deja_la or conn.execute(\"SELECT 1 FROM contacts WHERE customer_id = ?\", "
                  "(customer_id,)).fetchone():\n")], "N\\+1"),
    ("base.py", [("sqlite3.connect(chemin, check_same_thread=False)",
                  "sqlite3.connect(chemin, check_same_thread=False, isolation_level=None)")], "Tout ou rien"),
    ("base.py", [("        deja_la.add(customer_id)\n", "")], "deux fois"),
    ("base.py", [('conn.executemany("INSERT INTO contacts (customer_id, offre, conseiller) VALUES (?, ?, ?)", nouveaux)',
                  "[conn.execute(f\"INSERT INTO contacts (customer_id, offre, conseiller) VALUES ('{c}', '{o}', '{s}')\") "
                  "for c, o, s in nouveaux]")], "apostrophe"),
    ("base.py", [("    return len(nouveaux)\n", "    return len(contacts)\n")], "6 nouveaux"),
    # Étape 3 : lire, paginer, compter
    ("contacts.py", [(" ORDER BY customer_id LIMIT ?", " LIMIT ?")], "ORDER BY"),
    ("contacts.py", [("(*parametres, taille + 1)", "(*parametres, taille)"),
                     ("if len(lignes) > taille else None", "if len(lignes) == taille else None")], "taille \\+ 1"),
    ("contacts.py", [('conditions.append("customer_id > ?")', 'conditions.append("customer_id >= ?")')], "STRICTEMENT"),
    ("contacts.py", [('    if statut is not None:\n        conditions.append("statut = ?")\n        parametres.append(statut)\n',
                      "")], "AND statut = \\?"),
    ("contacts.py", [('conditions, parametres = ["conseiller = ?"], [conseiller]',
                      "conditions, parametres = [f\"conseiller = '{conseiller}'\"], []")], "injection"),
    ("contacts.py", [("    if contact.conseiller != conseiller:\n", "    if False:\n")], "PasVotreClient"),
    ("contacts.py", [('    lignes = conn.execute(\n        "SELECT conseiller, statut, COUNT(*) FROM contacts GROUP BY '
                      'conseiller, statut ORDER BY conseiller"\n    )',
                      "    lignes = [(c, s, n) for (c,) in conn.execute(\"SELECT DISTINCT conseiller FROM contacts ORDER BY "
                      "conseiller\").fetchall() for s, n in conn.execute(\"SELECT statut, COUNT(*) FROM contacts WHERE "
                      "conseiller = ? GROUP BY statut\", (c,)).fetchall()]")], "GROUP BY"),
    ("contacts.py", [("resultat.setdefault(conseiller, dict.fromkeys(STATUTS, 0))[statut] = nombre",
                      "resultat.setdefault(conseiller, {})[statut] = nombre")], "dict.fromkeys"),
    # Étape 4 : le verrou optimiste
    ("appels.py", [("    # UNE transaction pour la fiche ET le journal : les deux sont enregistrés, ou aucun.\n",
                    "    actuel = lire_contact(conn, conseiller, customer_id)\n    if actuel.version != version:\n"
                    "        raise Conflit(customer_id, actuel.version)\n"),
                   (_UPDATE_PARAMS, 'AND conseiller = ?",\n            (statut, commentaire, customer_id, conseiller),')],
     "DANS l'UPDATE"),
    ("appels.py", [(_UPDATE_PARAMS, 'AND version = ?",\n            (statut, commentaire, customer_id, version),')],
     "la base a changé"),
    ("appels.py", [("        if curseur.rowcount == 1:\n            conn.execute(",
                    "    with conn:\n      if curseur.rowcount == 1:\n            conn.execute(")], "MÊME transaction"),
    ("appels.py", [("        if curseur.rowcount == 1:\n            conn.execute(", "        if False:\n            conn.execute(")],
     "journal"),
    ("appels.py", [("version = version + 1 ", "version = version ")], "version \\+ 1"),
    ("appels.py", [("tentatives = tentatives + 1, ", "")], "tentatives \\+ 1"),
    ("appels.py", [("    if statut not in RESULTATS:\n", "    if False:\n")], "RESULTATS"),
    ("appels.py", [("raise Conflit(customer_id, contact.version)", "raise Conflit(customer_id, version)")],
     "version ACTUELLE"),
    # Étape 5 : l'API
    ("api.py", [(_COMPARE, 'if recue == cle.encode("utf-8"):')], "compare_digest"),
    ("api.py", [('recue = x_cle_api.encode("utf-8")', "recue = x_cle_api"),
                (_COMPARE, "if secrets.compare_digest(recue, cle):")], "octets"),
    ("api.py", [(_COMPARE, 'if cle.encode("utf-8").startswith(recue):')], "EN ENTIER"),
    ("api.py", [("    def connexion() -> Iterator[sqlite3.Connection]:\n",
                 "    partagee = ouvrir(config.base)\n\n    def connexion() -> Iterator[sqlite3.Connection]:\n"),
                (_CONNEXION, "        yield partagee\n")], "UNE connexion"),
    ("api.py", [(_CONNEXION, "        conn = ouvrir(config.base)\n        yield conn\n")], "jamais"),
    ("api.py", [('journal.warning("requête refusée : clé d\'API absente ou inconnue")',
                 'journal.warning("requête refusée : clé %r inconnue", x_cle_api)')], "journal contient une clé"),
    ("api.py", [('content={"detail": str(exc), "version_actuelle": exc.version_actuelle}', 'content={"detail": str(exc)}')],
     "version_actuelle"),
    ("api.py", [("Query(default=config.taille_page, ge=1, le=TAILLE_PAGE_MAX)", "config.taille_page")], "plante"),
    ("api.py", [("Query(default=config.taille_page, ge=1", "Query(default=20, ge=1")], "config.taille_page"),
    ("api.py", [("    @app.exception_handler(Introuvable)\n", "    @app.exception_handler(KeyError)\n")], "Introuvable"),
    ("api.py", [('    @app.get("/sante")\n', '    @app.get("/sante", dependencies=[Depends(conseiller_connecte)])\n')],
     "sans clé"),
    # Étape 6 : la ligne de commande
    ("cli.py", [("    if not config.base.exists():\n", "    if False:\n")], "exists\\(\\)"),
    ("cli.py", [("    if lancer is None:  # en vrai", "    if True:  # en vrai")], "uvicorn.run directement"),
    ("cli.py", [("    except ConfigInvalide as exc:", "    except KeyError as exc:")], "non attrapée"),
    ("cli.py", [("({len(config.conseillers)} conseillers : {', '.join(config.conseillers)} ; base {config.base})",
                 "(clés : {config.cles})")], "affiche la clé"),
    ("cli.py", [('servir.add_argument("--port", type=_port, default=8000)', 'servir.add_argument("--port", type=int, default=8000)')],
     "ArgumentTypeError"),
    ("cli.py", [('servir.add_argument("--hote", default="127.0.0.1"', 'servir.add_argument("--hote", default="0.0.0.0"')],
     "127.0.0.1"),
    ("cli.py", [("        except (OSError, ValueError, sqlite3.Error) as exc:\n",
                 "        except sqlite3.Error as exc:\n")], "non attrapée"),
    # Étape 7 : les tests de l'apprenant·e
    ("mes_tests.py", [('    assert client.get("/contacts", headers={"X-Cle-API": CLE_ALICE[:-1] + "X"}).status_code == 401\n', "")],
     "COMMENCE"),
    ("mes_tests.py", [('assert reponse.status_code == 409 and reponse.json()["version_actuelle"] == 2',
                       "assert reponse.status_code == 409")], "version actuelle"),
    ("mes_tests.py", [('reponse.json()["suivant"] == "C002"', 'reponse.json()["suivant"] == "C003"')], "version JUSTE"),
    ("mes_tests.py", [("    client = TestClient(creer_app(_preparer()))\n",
                       "    client = TestClient(creer_app(_preparer()))\n    assert client.get(\"/sante\").status_code == 200\n"
                       "    return\n")], "TOUTES les versions"),
]


@pytest.mark.parametrize("fichier, remplacements, motif", ERREURS_CLASSIQUES,
                         ids=[f"{f}:{m}" for f, _, m in ERREURS_CLASSIQUES])
def test_chaque_erreur_classique_recoit_le_bon_message(tmp_path, monkeypatch, fichier, remplacements, motif):
    resultat = _variante(tmp_path, monkeypatch, fichier, remplacements)
    assert resultat.statut != REUSSI, f"l'erreur n'est pas détectée ({motif})"
    import re

    assert re.search(motif, resultat.message), resultat.message


def test_une_etape_qui_depend_d_une_etape_non_ecrite(tmp_path, monkeypatch):
    """L'étape 4 appelle lire_contact (étape 3) : si elle n'est pas écrite, le message le dit."""
    resultat = _variante(tmp_path, monkeypatch, "contacts.py",
                         [("    ligne = conn.execute(f\"SELECT {COLONNES_CONTACT}", "    raise NotImplementedError\n"
                                                                                    "    ligne = conn.execute(f\"SELECT {COLONNES_CONTACT}")])
    assert resultat.statut == A_FAIRE  # l'étape 3 elle-même : à faire
    nom = sorted(p.name for p in tmp_path.iterdir())[-1]
    etape4 = next(e for e in PROJET.etapes if e.module == "appels")
    resultat = evaluer_etape(nom, etape4, PROJET.etapes)
    assert resultat.statut != REUSSI and "l'étape 3" in resultat.message

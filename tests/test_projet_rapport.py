"""Garantit que le projet 7 (le rapport du mois) est faisable et bien corrigé.

Ces tests NE notent PAS votre projet (pour ça : `python scripts/projet.py --projet 7`) ; ils vérifient
le corrigé, le fichier d'exemple, et que chaque erreur classique reçoit le bon message.
"""

import itertools
import re
import shutil
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projets import checks_rapport as checks  # noqa: E402
from projets.exemples_rapport import CHEMIN_EXEMPLE, generer_abonnes  # noqa: E402
from projets.rapport_corrige import mes_tests as mes_tests_corrige  # noqa: E402
from projets.rapport_corrige.cli import construire_rapport  # noqa: E402
from projets.rapport_corrige.formats import ecart_points, euros, pourcentage  # noqa: E402
from projets.rapport_corrige.modeles import lire_abonnes  # noqa: E402
from projets.rapport_corrige.page import page_html  # noqa: E402
from projets.verif import A_FAIRE, ECHEC, ERREUR, REUSSI, evaluer, evaluer_etape, projets  # noqa: E402

PROJET = checks.PROJET
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
    eleve = ROOT / "projets" / "rapport" / fichier
    corrige = ROOT / "projets" / "rapport_corrige" / fichier
    assert eleve.read_text(encoding="utf-8") == corrige.read_text(encoding="utf-8")


def test_les_outils_de_mes_tests_sont_identiques():
    def outils(paquet):
        texte = (ROOT / "projets" / paquet / "mes_tests.py").read_text(encoding="utf-8")
        return texte[texte.index("def _abonne"):texte.index("def tester_indicateurs_du_mois")]
    assert outils("rapport") == outils("rapport_corrige")


def test_le_projet_7_est_enregistre_apres_les_autres():
    assert [p.numero for p in projets()] == [2, 3, 4, 5, 6, 7]
    assert projets()[-1] is PROJET


def test_le_fichier_d_exemple_est_reproductible(tmp_path):
    """`projets/exemple_abonnes.csv` vaut generer_abonnes() octet par octet (le hasard vient de `random`)."""
    assert generer_abonnes(tmp_path / "abonnes.csv").read_bytes() == CHEMIN_EXEMPLE.read_bytes()


def test_l_exemple_raconte_l_histoire_du_besoin():
    """Les chiffres de projets/README.md : la hausse des tarifs Pro au 1er octobre se voit dans le rapport."""
    abonnes = lire_abonnes(CHEMIN_EXEMPLE)
    rapport = construire_rapport(abonnes, date(2026, 10, 1))
    a, p = rapport.actuel, rapport.precedent
    assert (a.actifs_debut, a.departs) == (2844, 89)
    assert a.actifs_fin == a.actifs_debut + a.arrivees - a.departs
    assert pourcentage(a.taux_churn) == "3,0\xa0%" and pourcentage(p.taux_churn) == "1,7\xa0%"
    assert ecart_points(p.taux_churn, a.taux_churn) == "+1,3\xa0pt"
    assert euros(a.revenu_perdu) == "7 821,59\xa0€" and euros(p.revenu_perdu) == "2 287,55\xa0€"
    pro = rapport.segments["Pro & TPE"]
    assert pourcentage(pro.taux_churn) == "6,2\xa0%"
    assert pourcentage(construire_rapport(abonnes, date(2026, 9, 1)).segments["Pro & TPE"].taux_churn) == "0,8\xa0%"
    assert list(rapport.segments) == ["Particulier", "Pro & TPE", "Senior"]
    assert '<script>alert("coucou")</script>' in [d.nom for d in rapport.departs], "le piège XSS est dans le rapport"
    assert any(d.debut.replace(day=1) == date(2026, 10, 1) for d in abonnes if d.fin and d.fin.month == 10
               and d.fin.year == 2026), "un client arrivé et parti en octobre, pour la règle du taux"
    for i in rapport.historique:  # l'invariant tient tous les mois
        assert i.actifs_fin == i.actifs_debut + i.arrivees - i.departs


def test_la_page_d_exemple_est_sure_et_autonome():
    page = page_html(construire_rapport(lire_abonnes(CHEMIN_EXEMPLE), date(2026, 10, 1)))
    assert "<script" not in page and "&lt;script&gt;alert(&quot;coucou&quot;)&lt;/script&gt;" in page
    assert "Bio &amp; Co &lt;Lyon&gt;" in page and "&amp;amp;" not in page
    assert not re.search(r"(src|href)=\"(https?:)?//", page) and "@import" not in page
    debut = page.index("<svg")
    ET.fromstring(page[debut:page.index("</svg>") + len("</svg>")])  # la courbe est du XML valide


def test_le_scenario_distingue_chaque_version_boguee():
    """Chaque version boguée d'indicateurs_du_mois donne une signature différente de la juste, et des autres."""
    import projets.rapport_corrige.modeles as M
    scenarios = checks._scenarios(M)
    juste = checks._signature(checks._variante(), scenarios)
    signatures = [checks._signature(checks._variante(**options), scenarios) for options, _, _ in checks._MUTANTS_INDICATEURS]
    for signature in signatures:
        assert not checks._meme_signature(signature, juste)
    for i, j in itertools.combinations(range(len(signatures)), 2):
        assert not checks._meme_signature(signatures[i], signatures[j]), (i, j)


def test_le_corrige_de_l_etape_7_attrape_toutes_les_versions_boguees():
    import projets.rapport_corrige.modeles as M
    mes_tests_corrige.tester_indicateurs_du_mois(checks._dans_vos_classes(checks._variante(), M))
    for options, quoi, _ in checks._MUTANTS_INDICATEURS:
        with pytest.raises(Exception):  # noqa: B017 — n'importe quelle erreur : le bug est vu
            mes_tests_corrige.tester_indicateurs_du_mois(checks._dans_vos_classes(checks._variante(**options), M))


def test_le_programme_termine_de_bout_en_bout(tmp_path):
    res = _python("-m", "projets.rapport_corrige", "--abonnes", str(CHEMIN_EXEMPLE), "--mois", "2026-10",
                  "--sortie", str(tmp_path / "rapports"), "--date-du-jour", "2026-11-19")
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines()[:2] == ["Octobre 2026 : 2 844 clients au début du mois, 89 départs",
                                           "Taux de churn : 3,0 % (+1,3 pt par rapport à septembre 2026)"]
    page = (tmp_path / "rapports" / "rapport_2026-10.html").read_text(encoding="utf-8")
    assert 'data-indicateur="taux_churn">3,0\xa0%<' in page

    res = _python("-m", "projets.rapport_corrige", "--abonnes", str(CHEMIN_EXEMPLE), "--mois", "2026-11",
                  "--sortie", str(tmp_path / "rapports"), "--date-du-jour", "2026-11-19")
    assert res.returncode == 1 and "pas terminé" in res.stderr and "Traceback" not in res.stderr


def test_le_squelette_explique_qu_il_n_est_pas_termine(tmp_path):
    res = _python("-m", "projets.rapport", "--abonnes", str(CHEMIN_EXEMPLE), "--mois", "2026-10",
                  "--sortie", str(tmp_path))
    assert res.returncode == 1
    assert "scripts/projet.py --projet 7" in res.stderr and "Traceback" not in res.stderr


def test_le_script_de_progression_connait_le_projet_7():
    res = _python("scripts/projet.py", "--projet", "7", "--corrige")
    assert res.returncode == 0 and "7/7" in res.stdout and "Projet 7" in res.stdout and "Projet 6" not in res.stdout
    res = _python("scripts/projet.py", "--projet", "7")
    assert res.returncode == 0 and "0/7" in res.stdout and "Prochaine étape : 1" in res.stdout


# ---------------------------------------------------------------------------- Les erreurs classiques


def _variante(tmp_path, monkeypatch, fichier: str, remplacements):
    """Le corrigé, avec UNE erreur classique introduite dans `fichier`, corrigé par le vrai moteur."""
    nom = f"rapport_variante_{next(_compteur)}"
    dossier = tmp_path / nom
    shutil.copytree(ROOT / "projets" / "rapport_corrige", dossier, ignore=shutil.ignore_patterns("__pycache__"))
    texte = (dossier / fichier).read_text(encoding="utf-8")
    for avant, apres in remplacements:
        assert avant in texte, f"{avant!r} introuvable dans {fichier}"
        texte = texte.replace(avant, apres)
    (dossier / fichier).write_text(texte, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    etape = next(e for e in PROJET.etapes if e.module == Path(fichier).stem)
    return evaluer_etape(nom, etape, PROJET.etapes)


_PREMIER = "    premier = decaler(mois, 0)  # le 1er du mois, quel que soit le jour reçu\n"
_ACTIF = "        actif_debut = a.debut < premier and (a.fin is None or a.fin >= premier)\n"
_PART = "        part = a.fin is not None and premier <= a.fin <= dernier\n"
_TAUX = "    taux = partis_parmi_actifs / actifs_debut if actifs_debut else None\n"

ERREURS_CLASSIQUES = [
    # Étape 1 — periodes.py
    ("periodes.py", [('    annee, mois = int(propre[:4]), int(propre[5:])\n', '    annee, mois = int(propre[:4]), int(propre[5:7])\n'),
                     ('re.fullmatch(r"\\d{4}-\\d{2}", propre)', 're.match(r"\\d{4}-\\d{2}", propre)')], "re.fullmatch"),
    ("periodes.py", [('    propre = texte.strip()\n    if not re.fullmatch(r"\\d{4}-\\d{2}", propre):', '    propre = texte.strip()\n    if False:'),
                     ("    annee, mois = int(propre[:4]), int(propre[5:])\n",
                      "    from datetime import datetime\n    lu = datetime.strptime(propre, '%Y-%m')\n    annee, mois = lu.year, lu.month\n")],
     "strptime"),
    ("periodes.py", [("    propre = texte.strip()\n", "    propre = texte\n")], "lève ValueError"),
    ("periodes.py", [("    return date(annee, mois, 1)\n", "    import datetime as dt\n    return dt.datetime(annee, mois, 1)\n")],
     "datetime"),
    ("periodes.py", [("    return date(mois.year, mois.month, calendar.monthrange(mois.year, mois.month)[1])\n",
                      "    from datetime import timedelta\n    return date(mois.year, mois.month + 1, 1) - timedelta(days=1)\n")],
     "JANVIER"),
    ("periodes.py", [("    return date(mois.year, mois.month, calendar.monthrange(mois.year, mois.month)[1])\n",
                      "    jours = 29 if mois.month == 2 and mois.year % 4 == 0 else (28 if mois.month == 2 else "
                      "(30 if mois.month in (4, 6, 9, 11) else 31))\n    return date(mois.year, mois.month, jours)\n")],
     "2100"),
    ("periodes.py", [("    rang = mois.year * 12 + (mois.month - 1) + n\n    return date(rang // 12, rang % 12 + 1, 1)\n",
                      "    from datetime import timedelta\n    return (mois.replace(day=1) + timedelta(days=31 * n)).replace(day=1)\n")],
     "en MOIS"),
    ("periodes.py", [("    return date(rang // 12, rang % 12 + 1, 1)\n", "    return date(rang // 12, rang % 12 + 1, mois.day)\n")],
     "1er"),
    ("periodes.py", [("MOIS_FR[mois.month - 1]", "MOIS_FR[mois.month]")], "élément 0"),
    ("periodes.py", [('    return f"{MOIS_FR[mois.month - 1]} {mois.year}"\n', '    return mois.strftime("%B %Y")\n')],
     "strftime"),
    ("periodes.py", [("MOIS_FR[mois.month - 1]}", "MOIS_FR[mois.month - 1].capitalize()}")], "minuscules"),
    ("periodes.py", [('    return f"{MOIS_FR[mois.month - 1]} {mois.year}"\n',
                      '    import locale\n    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")\n    return mois.strftime("%B %Y")\n')],
     "setlocale"),
    # Étape 2 — indicateurs.py
    ("indicateurs.py", [(_ACTIF, _ACTIF.replace("a.fin >= premier", "a.fin > premier"))], "parti le 1er"),
    ("indicateurs.py", [(_ACTIF, _ACTIF.replace("a.debut < premier", "a.debut <= premier"))], "arrivé le 1er"),
    ("indicateurs.py", [(_PART, _PART.replace("premier <= a.fin <= dernier", "a.fin <= dernier"))], "borne basse"),
    ("indicateurs.py", [(_PART, _PART.replace("a.fin <= dernier", "a.fin < dernier"))], "dernier jour"),
    ("indicateurs.py", [(_TAUX, _TAUX.replace("partis_parmi_actifs /", "departs /"))], "100 %"),
    ("indicateurs.py", [(_TAUX, _TAUX.replace("else None", "else 0.0"))], "renvoyez None"),
    ("indicateurs.py", [(_TAUX, "    taux = partis_parmi_actifs / actifs_fin if actifs_fin else None\n")], "DÉBUT du mois"),
    ("indicateurs.py", [("            revenu_perdu += a.mensualite\n",
                         "            revenu_perdu += a.mensualite if actif_debut else 0.0\n")], "TOUS les départs"),
    ("indicateurs.py", [("        arrivees += premier <= a.debut <= dernier\n", "        arrivees += a.debut <= dernier\n")],
     "Une arrivée du mois"),
    ("indicateurs.py", [("        actifs_fin += a.debut <= dernier and (a.fin is None or a.fin > dernier)\n",
                         "        actifs_fin += a.debut <= dernier and (a.fin is None or a.fin >= dernier)\n")],
     "plus là à la fin"),
    ("indicateurs.py", [("    taux = partis", "    actifs_fin = actifs_debut - departs\n    taux = partis")], "arrivées"),
    ("indicateurs.py", [(_PREMIER, "    premier = mois\n")], "n'importe quel jour"),
    ("indicateurs.py", [("    dernier = fin_du_mois(premier)\n",
                         "    from datetime import timedelta\n    dernier = date(premier.year, premier.month + 1, 1) - timedelta(days=1)\n")],
     "décembre"),
    ("indicateurs.py", [(_PREMIER, _PREMIER + "    abonnes.sort(key=lambda a: a.client, reverse=True)\n")], "MODIFIÉ"),
    ("indicateurs.py", [(_ACTIF, _ACTIF.replace("a.fin >= premier", "a.fin > premier")),
                        (_PART, _PART.replace("a.fin <= dernier", "a.fin < dernier"))], "Calculez chaque indicateur"),
    ("indicateurs.py", [("for segment in sorted(groupes)}", "for segment in groupes}")], "ordre de `sorted`"),
    ("indicateurs.py", [("""    groupes: dict[str, list[Abonne]] = {}
    for a in abonnes:
        groupes.setdefault(a.segment, []).append(a)
    return {segment: indicateurs_du_mois(groupes[segment], mois) for segment in sorted(groupes)}""",
                         """    segments = sorted({a.segment for a in abonnes})
    return {s: indicateurs_du_mois([a for a in abonnes if a.segment == s], mois) for s in segments}""")],
     "générateur"),
    ("indicateurs.py", [("for segment in sorted(groupes)}", "for segment in sorted(groupes) "
                                                             "if any(a.fin is None or a.fin >= mois for a in groupes[segment])}")],
     "Ancien"),
    # Étape 3 — formats.py
    ("formats.py", [("rounding=ROUND_HALF_UP", "rounding=ROUND_HALF_EVEN"),
                    ("from decimal import ROUND_HALF_UP, Decimal", "from decimal import ROUND_HALF_EVEN, Decimal")],
     "moitié SUPÉRIEURE"),
    ("formats.py", [("    return Decimal(str(x))\n", "    return Decimal(x)\n")], "valeur binaire"),
    ("formats.py", [('.replace(",", ESPACE_MILLIERS)', '.replace(",", " ")')], "espace fine"),
    ("formats.py", [('    return f"{_chiffres(_arrondi(_decimal(montant), 2), 2)}{ESPACE_UNITE}€"',
                     '    return f"{_chiffres(_arrondi(_decimal(montant), 2), 2)} €"')], "ESPACE_UNITE"),
    ("formats.py", [("    return abs(resultat) if resultat == 0 else resultat", "    return resultat")], "sans signe"),
    ("formats.py", [('signe = "+" if ecart > 0 else ""', 'signe = "+" if ecart >= 0 else ""')], "sans signe"),
    ("formats.py", [("    ecart = _arrondi((_decimal(apres) - _decimal(avant)) * 100, 1)",
                     "    ecart = _arrondi(_decimal((apres - avant) * 100), 1)")], "0.013499999999999998"),
    ("formats.py", [("    ecart = _arrondi((_decimal(apres) - _decimal(avant)) * 100, 1)",
                     "    ecart = _arrondi((_decimal(apres) - _decimal(avant)) / _decimal(avant) * 100, 1)")], "POINTS"),
    ("formats.py", [("_arrondi(_decimal(part) * 100, decimales)", "_arrondi(_decimal(part * 100), decimales)")],
     "2.9499999999999997"),
    ("formats.py", [('{ESPACE_UNITE}pt"', '{ESPACE_UNITE}pts"')], "pas de"),
    ("formats.py", [("pluriel = mot if n < 2 or", "pluriel = mot if n == 1 or")], "SINGULIER"),
    ("formats.py", [(' or mot.endswith(("s", "x", "z"))', "")], "s, x ou z"),
    ("formats.py", [("    if montant is None:\n        return TIRET\n", "")], "euros\\(None\\)"),
    ("formats.py", [("    ecart = _arrondi((_decimal(apres) - _decimal(avant)) * 100, 1)",
                     "    ecart = _arrondi((Decimal(apres) - Decimal(avant)) * 100, 1)")], "Decimal\\(str\\(apres\\)\\)"),
    ("formats.py", [("_arrondi(_decimal(part) * 100, decimales)", "_arrondi(Decimal(part) * 100, decimales)")],
     "AVANT de multiplier"),
    ("formats.py", [('.replace(",", ESPACE_MILLIERS).replace(".", ",")', '.replace(".", ",").replace(",", ESPACE_MILLIERS)')],
     None),
    # Étape 4 — graphique.py
    ("graphique.py", [("            if candidat >= maximum:", "            if candidat > maximum:")], "déjà un nombre rond"),
    ("graphique.py", [('for chiffre in ("1", "2", "2.5", "5")', 'for chiffre in ("1", "2", "5")')], "2,5"),
    ("graphique.py", [("    if not maximum > 0:  # 0, un négatif… ou nan, pour qui toute comparaison est fausse\n"
                       "        return ECHELLE_PAR_DEFAUT\n", "")], "ECHELLE_PAR_DEFAUT"),
    ("graphique.py", [("            candidat = float(Decimal(chiffre).scaleb(puissance))",
                       "            candidat = float(chiffre) * 10 ** puissance")], "5e-06"),
    ("graphique.py", [("        return bas - (bas - sommet) * v / haut", "        return sommet + (bas - sommet) * v / haut")],
     "vers le BAS"),
    ("graphique.py", [("(i / (n - 1) if n > 1 else 0.5)", "(i / n)")], "n - 1"),
    ("graphique.py", [("    for i, v in enumerate([*valeurs, None]):\n        if v is not None:",
                       "    for i, v in enumerate([*[0.0 if w is None else w for w in valeurs], None]):\n        if v is not None:")],
     "tomber à zéro"),
    ("graphique.py", [("    for i, v in enumerate([*valeurs, None]):\n        if v is not None:",
                       "    for i, v in enumerate([*[w for w in valeurs if w is not None], None]):\n        if v is not None:")],
     None),
    ("graphique.py", [("    for i, v in enumerate([*valeurs, None]):\n        if v is not None:",
                       "    for i, v in [*[(i, v) for i, v in enumerate(valeurs) if v is not None], (len(valeurs), None)]:\n"
                       "        if v is not None:")], "COUPER"),
    ("graphique.py", [("        if len(suite) >= 2:", "        if len(suite) >= 1:")], "un seul point"),
    ("graphique.py", [("<title>{e(titre)}</title>", "<title>{titre}</title>")], "html.escape"),
    ("graphique.py", [('<svg xmlns="{SVG_NS}" ', "<svg ")], "xmlns"),
    ("graphique.py", [("    gauche, droite, sommet, bas = MARGE_GAUCHE, largeur - MARGE_DROITE, MARGE_HAUT, hauteur - MARGE_BAS",
                       "    gauche, droite, sommet, bas = MARGE_GAUCHE, LARGEUR - MARGE_DROITE, MARGE_HAUT, HAUTEUR - MARGE_BAS")],
     "au lieu de"),
    ("graphique.py", [('    return f"{x:.1f}"', '    return f"{x}"')], "une décimale"),
    ("graphique.py", [("max((v for v in valeurs if v is not None), default=0.0)", "max(v for v in valeurs if v is not None)")],
     "default"),
    ("graphique.py", [("<title>{e(etiquettes[i])} : {e(pourcentage(v))}</title>", "")], "infobulle"),
    ("graphique.py", [("    if len(etiquettes) != len(valeurs):\n", "    if False:\n")], "ValueError"),
    ("graphique.py", [("    return pourcentage(v, max(1, decimales))", "    return pourcentage(v)")], "ment sur la position"),
    # Étape 5 — page.py
    ("page.py", [("<td>{e(d.nom)}</td>", "<td>{d.nom}</td>")], "XSS"),
    ("page.py", [("<td>{e(d.nom)}</td>", "<td>{e(e(d.nom))}</td>")], "DEUX fois"),
    ("page.py", [("\n{courbe}\n", "\n{e(courbe)}\n")], "échappée"),
    ("page.py", [("<tr><td>{e(segment)}</td>", "<tr><td>{segment}</td>")], "segment"),
    ("page.py", [('STYLE = """\n', 'STYLE = """\n@import url("https://fonts.googleapis.com/css2?family=Inter");\n')],
     "extérieure"),
    ("page.py", [("<title>{e(titre)}</title>\n", '<title>{e(titre)}</title>\n<link rel="stylesheet" href="rapport.css">\n')],
     "feuille de style"),
    ("page.py", [('    return f"""<!doctype html>\n', '    return f"""\n')], "doctype"),
    ("page.py", [('<html lang="fr">', "<html>")], "lang"),
    ("page.py", [('<meta charset="utf-8">\n', "")], "charset"),
    ("page.py", [('    titre = f"Rapport churn — {nom_du_mois(rapport.mois)}"',
                  '    titre = f"Rapport churn - {nom_du_mois(rapport.mois)}"')], "<title>"),
    ("page.py", [("</span> par rapport à {e(nom_du_mois(p.mois))}</p>", " par rapport à {e(nom_du_mois(p.mois))}</span></p>")],
     "EXACTEMENT"),
    ("page.py", [('carte("Revenu mensuel perdu", "revenu_perdu", euros(a.revenu_perdu)),',
                  'carte("Revenu mensuel perdu", "revenu", euros(a.revenu_perdu)),')], 'data-indicateur=\\"revenu_perdu\\"'),
    ("page.py", [('        departs = "<p>Aucun départ ce mois-ci.</p>"', "        departs = '<table data-tableau=\"departs\"></table>'")],
     "Aucun départ"),
    ("page.py", [("import html\n", "import html\nfrom datetime import datetime\n"),
                 ('<p class="methode">', '<p>Généré le {datetime.now().isoformat()}</p>\n<p class="methode">')],
     "deux pages différentes"),
    ("page.py", [("for segment, i in rapport.segments.items())", "for segment, i in sorted(rapport.segments.items(), reverse=True))")],
     "tableau des segments"),
    ("page.py", [("for d in rapport.departs)", "for d in sorted(rapport.departs, key=lambda d: d.client))")],
     "dans l'ordre reçu"),
    # Étape 6 — cli.py
    ("cli.py", [('    chemin.write_text(page_html(rapport), encoding="utf-8")', "    chemin.write_text(page_html(rapport))")],
     "encoding"),
    ("cli.py", [("range(MOIS_HISTORIQUE - 1, -1, -1)", "range(MOIS_HISTORIQUE)")], "à l'envers"),
    ("cli.py", [("range(MOIS_HISTORIQUE - 1, -1, -1)", "range(MOIS_HISTORIQUE, 0, -1)")], "le mois du rapport inclus"),
    ("cli.py", [("departs=tuple(partis[:DEPARTS_AFFICHES])", "departs=tuple(partis)")], "DEPARTS_AFFICHES"),
    ("cli.py", [("key=lambda a: (-a.mensualite, a.client))", "key=lambda a: a.mensualite, reverse=True)")],
     "identifiant croissant"),
    ("cli.py", [("    abonnes = list(abonnes)  # on va les parcourir plusieurs fois : un générateur serait vide au 2e passage\n", "")],
     "générateur"),
    ("cli.py", [("precedent=indicateurs_du_mois(abonnes, decaler(premier, -1))", "precedent=indicateurs_du_mois(abonnes, premier)")],
     "mois d'AVANT"),
    ("cli.py", [("    if fin_du_mois(mois) >= aujourd_hui:", "    if fin_du_mois(mois) > aujourd_hui:")], "31 octobre"),
    ("cli.py", [("    if not connues or max(connues) < mois:", "    if not connues:")], "mars 2027"),
    ("cli.py", [("    chemin.parent.mkdir(parents=True, exist_ok=True)\n", "")], "FileNotFoundError"),
    ("cli.py", [('    print(_terminal(f"Taux de churn', '    print((f"Taux de churn')], "insécables"),
    ("cli.py", [("    except FileNotFoundError:\n        return _erreur(f\"fichier introuvable : {args.abonnes}\")\n", "")],
     "FileNotFoundError"),
    ("cli.py", [('    print(f"Erreur : {message}", file=sys.stderr)', '    print(f"Erreur : {message}")')], "stderr"),
    ("cli.py", [('    print(f"Rapport écrit dans {chemin}")\n    return 0\n', '    print(f"Rapport écrit dans {chemin}")\n')],
     "return 0"),
    ("cli.py", [('    print(f"Rapport écrit dans {chemin}")\n', '    print("Rapport écrit.")\n')], "chemin du rapport"),
    ("cli.py", [('    parser.add_argument("--sortie", type=Path, default=Path("rapports")',
                 '    parser.add_argument("--sortie", type=Path, default=Path(".")')], "Sans --sortie"),
]


@pytest.mark.parametrize("fichier, remplacements, motif", ERREURS_CLASSIQUES,
                         ids=[f"{f}-{i}" for i, (f, _, _) in enumerate(ERREURS_CLASSIQUES)])
def test_chaque_erreur_classique_recoit_le_bon_message(tmp_path, monkeypatch, fichier, remplacements, motif):
    resultat = _variante(tmp_path, monkeypatch, fichier, remplacements)
    assert resultat.statut in (ECHEC, ERREUR), f"erreur non détectée : {remplacements}"
    if motif is not None:
        assert re.search(motif, resultat.message), resultat.message


@pytest.mark.parametrize("fichier, remplacements", [
    ("page.py", [("<tr><td>{e(segment)}</td>", '<tr><th scope="row">{e(segment)}</th>'),
                 ("<tr><td>{e(d.client)}</td>", '<tr><th scope="row">{e(d.client)}</th>')]),
    ("page.py", [("\n</tbody>\n</table></div>\n<h2>Les plus gros départs</h2>",
                  "\n</tbody>\n<tfoot><tr><td>Total</td><td>{e(nombre(a.actifs_debut, 'client'))}</td><td></td><td></td></tr>"
                  "</tfoot>\n</table></div>\n<h2>Les plus gros départs</h2>")]),
    ("cli.py", [('    print(f"Erreur : {message}", file=sys.stderr)\n    return 1\n', '    sys.exit(f"Erreur : {message}")\n')]),
], ids=["th-en-tete-de-ligne", "ligne-de-total", "sys.exit"])
def test_d_autres_ecritures_justes_sont_acceptees(tmp_path, monkeypatch, fichier, remplacements):
    """Un en-tête de ligne <th scope="row">, une ligne de total dans <tfoot>, sys.exit("message") : c'est juste."""
    resultat = _variante(tmp_path, monkeypatch, fichier, remplacements)
    assert resultat.statut == REUSSI, resultat.message


def test_un_lien_cliquable_n_est_pas_une_ressource_exterieure(tmp_path, monkeypatch):
    """<a href="https://…"> se clique, il ne se télécharge pas : la page reste autonome."""
    resultat = _variante(tmp_path, monkeypatch, "page.py", [
        ('<p class="methode">', '<p><a href="https://intranet.example/churn">Le tableau de bord</a></p>\n<p class="methode">')])
    assert resultat.statut == REUSSI, resultat.message


# ---------------------------------------------------------------------------- Étape 7 : le test de l'élève

_SCENARIO = """
    octobre = date(2026, 10, 1)
    abonnes = [_abonne("A", "2026-01-10"), _abonne("B", "2026-02-01", "2026-10-01", 30.0)]
    i = indicateurs_du_mois(abonnes, octobre)
"""


def _module_de_test(corps: str) -> types.ModuleType:
    """Un mes_tests.py « d'apprenant·e » : les outils du corrigé, et `tester_indicateurs_du_mois` = `corps`."""
    module = types.ModuleType("projets.rapport_corrige.mes_tests_eleve")
    module.__package__ = "projets.rapport_corrige"
    module.__dict__.update({k: v for k, v in vars(mes_tests_corrige).items() if not k.startswith("__")})
    exec("def tester_indicateurs_du_mois(indicateurs_du_mois):\n" + corps, module.__dict__)  # noqa: S102
    return module


@pytest.mark.parametrize("corps, motif", [
    ("    pass\n", "TOUTES les versions boguées"),
    (_SCENARIO + "    assert i.mois == octobre\n", "TOUTES les versions boguées"),
    (_SCENARIO + "    assert i.actifs_debut == 2 and i.departs == 1 and i.taux_churn == 0.5\n", r"\d+/13 bugs attrapés"),
    (_SCENARIO + "    assert i.actifs_debut == 1\n", "ÉCHOUE sur la version JUSTE"),
    ("    indicateurs_du_mois('abonnes.csv', '2026-10')\n", "plante sur la version JUSTE"),
])
def test_le_correcteur_de_l_etape_7(corps, motif):
    with pytest.raises(AssertionError, match=motif):
        checks.verifier_tester_indicateurs(_module_de_test(corps))


def test_l_etape_7_compare_avec_vos_propres_classes():
    """Un test qui compare à un `Indicateurs(...)` de SON modeles.py doit passer sur la version juste."""
    corps = _SCENARIO + "    assert i == Indicateurs(octobre, 2, 0, 1, 1, 0.5, 30.0)\n"
    module = _module_de_test(corps)
    from projets.rapport_corrige.modeles import Indicateurs
    module.Indicateurs = Indicateurs
    with pytest.raises(AssertionError, match=r"\d+/13 bugs attrapés"):  # trop faible, mais juste
        checks.verifier_tester_indicateurs(module)


# ---------------------------------------------------------------------------- D'autres solutions justes

AUTRES_SOLUTIONS = {
    "periodes.py": '''
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from .modeles import MOIS_FR


def lire_mois(texte):
    texte = texte.strip()
    if re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", texte) is None:
        raise ValueError(f"{texte!r} : attendu AAAA-MM")
    return datetime.strptime(texte, "%Y-%m").date()


def decaler(mois, n):
    annee, rang = divmod(mois.month - 1 + n, 12)
    return date(mois.year + annee, rang + 1, 1)


def fin_du_mois(mois):
    return decaler(mois, 1) - timedelta(days=1)


def nom_du_mois(mois):
    return MOIS_FR[mois.month - 1] + " " + str(mois.year)
''',
    "indicateurs.py": '''
from __future__ import annotations

from collections import defaultdict

from .modeles import Indicateurs
from .periodes import decaler, fin_du_mois


def _present(a, jour):
    return a.debut <= jour and (a.fin is None or a.fin >= jour)


def indicateurs_du_mois(abonnes, mois):
    premier = decaler(mois, 0)
    dernier = fin_du_mois(premier)
    veille = decaler(premier, -1).replace(day=fin_du_mois(decaler(premier, -1)).day)
    abonnes = list(abonnes)
    debut = [a for a in abonnes if _present(a, veille) and _present(a, premier)]
    partis = [a for a in abonnes if a.fin is not None and premier <= a.fin <= dernier]
    arrives = [a for a in abonnes if premier <= a.debut <= dernier]
    fin = len(debut) + len(arrives) - len(partis)
    perdus = [a for a in debut if a in partis]
    return Indicateurs(premier, len(debut), len(arrives), len(partis), fin,
                       len(perdus) / len(debut) if debut else None, sum(a.mensualite for a in partis))


def par_segment(abonnes, mois):
    groupes = defaultdict(list)
    for a in abonnes:
        groupes[a.segment].append(a)
    return {s: indicateurs_du_mois(groupes[s], mois) for s in sorted(groupes)}
''',
    "formats.py": '''
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .modeles import ESPACE_MILLIERS, ESPACE_UNITE, TIRET

_ECHANGE = str.maketrans({",": ESPACE_MILLIERS, ".": ","})


def _ecrire(valeur, decimales):
    q = Decimal(str(valeur)).quantize(Decimal(10) ** -decimales, rounding=ROUND_HALF_UP)
    if not q:
        q = q.copy_abs()
    return format(q, f",.{decimales}f").translate(_ECHANGE), q


def euros(montant):
    return TIRET if montant is None else _ecrire(montant, 2)[0] + ESPACE_UNITE + "€"


def pourcentage(part, decimales=1):
    return TIRET if part is None else _ecrire(Decimal(str(part)) * 100, decimales)[0] + ESPACE_UNITE + "%"


def ecart_points(avant, apres):
    if None in (avant, apres):
        return TIRET
    texte, q = _ecrire((Decimal(str(apres)) - Decimal(str(avant))) * 100, 1)
    return ("+" if q > 0 else "") + texte + ESPACE_UNITE + "pt"


def nombre(n, mot):
    if n >= 2 and mot[-1] not in "sxz":
        mot += "s"
    return _ecrire(n, 0)[0] + ESPACE_UNITE + mot
''',
    "cli.py": '''
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from .formats import ecart_points, pourcentage
from .indicateurs import indicateurs_du_mois, par_segment
from .modeles import DEPARTS_AFFICHES, MOIS_HISTORIQUE, Rapport, lire_abonnes
from .page import page_html
from .periodes import decaler, fin_du_mois, lire_mois, nom_du_mois


def construire_rapport(abonnes, mois):
    abonnes = list(abonnes)
    mois = decaler(mois, 0)
    historique = []
    for k in range(MOIS_HISTORIQUE):
        historique.insert(0, indicateurs_du_mois(abonnes, decaler(mois, -k)))
    partis = [a for a in abonnes if a.fin and mois <= a.fin <= fin_du_mois(mois)]
    partis.sort(key=lambda a: a.client)
    partis.sort(key=lambda a: a.mensualite, reverse=True)  # tri stable : l'identifiant départage
    return Rapport(mois, historique[-1], indicateurs_du_mois(abonnes, decaler(mois, -1)), par_segment(abonnes, mois),
                   tuple(historique), tuple(partis[:DEPARTS_AFFICHES]))


def _mois(texte):
    try:
        return lire_mois(texte)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--abonnes", required=True)
    parser.add_argument("--mois", required=True, type=_mois)
    parser.add_argument("--sortie", default="rapports")
    parser.add_argument("--date-du-jour", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)
    jour = args.date_du_jour or date.today()
    if not fin_du_mois(args.mois) < jour:
        print("Ce mois n'est pas terminé.", file=sys.stderr)
        return 1
    try:
        abonnes = lire_abonnes(args.abonnes)
    except (OSError, ValueError) as exc:
        print(f"Lecture impossible : {exc}", file=sys.stderr)
        return 1
    if not any(d is not None and d >= args.mois for a in abonnes for d in (a.debut, a.fin)):
        print("Aucune donnée pour ce mois.", file=sys.stderr)
        return 1
    rapport = construire_rapport(abonnes, args.mois)
    dossier = Path(args.sortie)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"rapport_{args.mois.isoformat()[:7]}.html"
    with open(chemin, "w", encoding="utf-8", newline="\\n") as f:
        f.write(page_html(rapport))
    a, p = rapport.actuel, rapport.precedent
    print(" ".join(f"{nom_du_mois(args.mois)} : {a.actifs_debut} clients au début, {a.departs} départs, taux "
                   f"{pourcentage(a.taux_churn)} ({ecart_points(p.taux_churn, a.taux_churn)})".split()))
    print(f"Écrit : {chemin}")
    return 0
''',
}


@pytest.mark.parametrize("fichier", sorted(AUTRES_SOLUTIONS))
def test_une_autre_solution_juste_est_acceptee(tmp_path, monkeypatch, fichier):
    """D'autres façons d'écrire juste (strptime après une regex, divmod, listes plutôt qu'une boucle, str.maketrans,
    argparse avec type=, deux tris stables, open() avec encoding), que le correcteur accepte."""
    nom = f"rapport_autre_{next(_compteur)}"
    dossier = tmp_path / nom
    shutil.copytree(ROOT / "projets" / "rapport_corrige", dossier, ignore=shutil.ignore_patterns("__pycache__"))
    (dossier / fichier).write_text(AUTRES_SOLUTIONS[fichier], encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for etape in PROJET.etapes:
        resultat = evaluer_etape(nom, etape, PROJET.etapes)
        assert resultat.statut == REUSSI, (etape.module, resultat.message)

"""Étape 5 (corrigé) — La page HTML, autonome et sans injection."""

from __future__ import annotations

import html

from .formats import ecart_points, euros, nombre, pourcentage
from .graphique import courbe_svg
from .modeles import Rapport, etiquette_courte
from .periodes import nom_du_mois

#: Tout le style est DANS la page : pas de fichier .css à côté, pas de police à télécharger. Le rapport
#: s'ouvre sans réseau, dans un navigateur comme dans un logiciel de courriel.
STYLE = """
body { margin: 0; background: #f4f5f7; color: #1d2330; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 12px; }
.sous-titre { margin: 0 0 24px; color: #5b6270; }
.chiffres { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.carte { background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px 16px; }
.libelle { margin: 0; color: #5b6270; font-size: 13px; }
.valeur { margin: 2px 0 0; font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.ecart { margin: 2px 0 0; color: #5b6270; font-size: 13px; }
figure { margin: 0; background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; overflow-x: auto; }
svg { display: block; width: 100%; height: auto; }
.defile { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e1e4e8; }
th, td { padding: 8px 12px; border-bottom: 1px solid #e1e4e8; text-align: left; }
th { background: #f8f9fb; font-size: 13px; color: #5b6270; font-weight: 600; }
td.nombre, th.nombre { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.methode { margin-top: 32px; color: #5b6270; font-size: 13px; }
"""


def page_html(rapport: Rapport) -> str:
    e = html.escape  # « & » → « &amp; », « < » → « &lt; », « " » → « &quot; »… : du texte reste du texte
    titre = f"Rapport churn — {nom_du_mois(rapport.mois)}"
    a, p = rapport.actuel, rapport.precedent

    def carte(libelle: str, nom: str, valeur: str, ecart: str = "") -> str:
        return (f'<div class="carte"><p class="libelle">{e(libelle)}</p>'
                f'<p class="valeur" data-indicateur="{nom}">{e(valeur)}</p>{ecart}</div>')

    ecart = (f'<p class="ecart"><span data-indicateur="ecart_taux">{e(ecart_points(p.taux_churn, a.taux_churn))}'
             f"</span> par rapport à {e(nom_du_mois(p.mois))}</p>")
    cartes = "\n".join([
        carte("Clients au début du mois", "actifs_debut", nombre(a.actifs_debut, "client")),
        carte("Départs du mois", "departs", nombre(a.departs, "départ")),
        carte("Taux de churn", "taux_churn", pourcentage(a.taux_churn), ecart),
        carte("Revenu mensuel perdu", "revenu_perdu", euros(a.revenu_perdu)),
    ])

    lignes_segments = "\n".join(
        f'<tr><td>{e(segment)}</td><td class="nombre">{e(nombre(i.actifs_debut, "client"))}</td>'
        f'<td class="nombre">{e(nombre(i.departs, "départ"))}</td><td class="nombre">{e(pourcentage(i.taux_churn))}</td></tr>'
        for segment, i in rapport.segments.items())

    if rapport.departs:
        lignes_departs = "\n".join(
            f'<tr><td>{e(d.client)}</td><td>{e(d.nom)}</td><td>{e(d.segment)}</td>'
            f'<td class="nombre">{e(euros(d.mensualite))}</td><td class="nombre">{d.fin:%d/%m/%Y}</td></tr>'
            for d in rapport.departs)
        departs = (f'<div class="defile"><table data-tableau="departs">\n<thead><tr><th>Client</th><th>Nom</th><th>Segment</th>'
                   f'<th class="nombre">Mensualité</th><th class="nombre">Dernier jour</th></tr></thead>\n'
                   f"<tbody>\n{lignes_departs}\n</tbody>\n</table></div>")
    else:
        departs = "<p>Aucun départ ce mois-ci.</p>"

    historique = rapport.historique
    courbe = courbe_svg([etiquette_courte(i.mois) for i in historique], [i.taux_churn for i in historique],
                        f"Taux de churn mensuel, de {nom_du_mois(historique[0].mois)} à {nom_du_mois(rapport.mois)}")

    # La courbe est DÉJÀ du SVG fabriqué par notre code (et ses textes y sont déjà échappés) : on
    # l'insère telle quelle. Tout ce qui vient des DONNÉES, en revanche, passe par e().
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(titre)}</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<h1>{e(titre)}</h1>
<p class="sous-titre">Les abonnés qui sont partis en {e(nom_du_mois(rapport.mois))}, comparés au mois précédent.</p>
<section class="chiffres">
{cartes}
</section>
<h2>Les 12 derniers mois</h2>
<figure>
{courbe}
</figure>
<h2>Par segment</h2>
<div class="defile"><table data-tableau="segments">
<thead><tr><th>Segment</th><th class="nombre">Clients au début</th><th class="nombre">Départs</th><th class="nombre">Taux de churn</th></tr></thead>
<tbody>
{lignes_segments}
</tbody>
</table></div>
<h2>Les plus gros départs</h2>
{departs}
<p class="methode">Méthode : le taux de churn est la part des abonnés présents au début du mois qui sont partis
pendant le mois. Un client arrivé puis parti le même mois compte dans les départs et dans le revenu perdu, mais
pas dans le taux. L'écart se mesure en points : passer de 2,0 % à 3,0 %, c'est +1,0 pt (et +50 % en relatif).</p>
</main>
</body>
</html>
"""

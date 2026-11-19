"""Étape 5 🔴 — La page HTML, autonome et sans injection.

Vérifier : python scripts/projet.py --projet 7 --etape 5
"""

from __future__ import annotations

import html  # noqa: F401 — vous en aurez besoin

from .formats import ecart_points, euros, nombre, pourcentage  # noqa: F401
from .graphique import courbe_svg  # noqa: F401
from .modeles import Rapport, etiquette_courte  # noqa: F401
from .periodes import nom_du_mois  # noqa: F401


def page_html(rapport: Rapport) -> str:
    """Le rapport en UNE page HTML complète (une chaîne de caractères), que la direction ouvre ou reçoit par courriel.

    1. Un document complet et autonome. Il commence par `<!doctype html>` et contient
       `<html lang="fr">`, `<meta charset="utf-8">` et `<title>Rapport churn — octobre 2026</title>`
       (avec `nom_du_mois(rapport.mois)` ; le tiret est un tiret cadratin « — »), puis un `<h1>` avec le
       même texte. AUCUNE ressource extérieure : ni `<script>` (les logiciels de courriel les retirent),
       ni `<link>` vers une feuille de style, ni image ou police à télécharger (`src=` ou `href=` vers
       http://, https:// ou //, `url(` ou `@import` dans le style). Le style va dans un `<style>` de la
       page. Le rapport doit s'ouvrir sans réseau, dans dix ans, tel qu'il était.

    2. Les chiffres clés du mois, chacun dans un élément qui porte un attribut `data-indicateur` et dont
       le texte est EXACTEMENT la valeur formatée par l'étape 3 (avec `a = rapport.actuel` et
       `p = rapport.precedent`) :

           data-indicateur="actifs_debut"   nombre(a.actifs_debut, "client")       « 2 844 clients »
           data-indicateur="departs"        nombre(a.departs, "départ")            « 89 départs »
           data-indicateur="taux_churn"     pourcentage(a.taux_churn)              « 3,0 % »
           data-indicateur="ecart_taux"     ecart_points(p.taux_churn, a.taux_churn)  « +1,3 pt »
           data-indicateur="revenu_perdu"   euros(a.revenu_perdu)                  « 7 821,59 € »

       Par exemple `<p data-indicateur="taux_churn">3,0 %</p>`. Le texte autour (« par rapport à
       septembre 2026 ») va HORS de ces éléments. L'attribut sert au correcteur, et à quiconque voudra
       un jour relire vos rapports avec un programme.

    3. Un `<table data-tableau="segments">` : une ligne `<tr>` de cellules `<td>` par segment, dans
       l'ordre de `rapport.segments`, avec 4 cellules : le segment, nombre(actifs_debut, "client"),
       nombre(departs, "départ") et pourcentage(taux_churn). Les en-têtes sont des `<th>`.

    4. Si `rapport.departs` n'est pas vide, un `<table data-tableau="departs">` : une ligne par départ,
       dans l'ordre reçu, avec 5 cellules `<td>` : client, nom, segment, euros(mensualite) et le dernier
       jour au format JJ/MM/AAAA (`f"{d.fin:%d/%m/%Y}"`). S'il est vide, pas de tableau : la phrase
       « Aucun départ ce mois-ci. ».

    5. La courbe des mois de `rapport.historique`, insérée TELLE QUELLE dans la page :

           courbe_svg([etiquette_courte(i.mois) for i in rapport.historique],
                      [i.taux_churn for i in rapport.historique],
                      f"Taux de churn mensuel, de {nom_du_mois(rapport.historique[0].mois)} à {nom_du_mois(rapport.mois)}")

    6. LA règle de sécurité : tout texte qui vient des DONNÉES (noms, segments, identifiants…) passe
       par `html.escape` avant d'entrer dans la page. Les noms sont saisis en boutique, et
       l'exemple contient un client nommé `<script>alert("coucou")</script>` : sans échappement, le
       navigateur EXÉCUTE ce code à l'ouverture du rapport. C'est une faille XSS (injection de code),
       l'une des plus répandues du web. Avec `html.escape`, il affiche le nom tel quel, comme du texte.
       À l'inverse, la courbe est DÉJÀ du SVG (fabriqué et échappé par votre étape 4) : l'échapper une
       seconde fois afficherait le code du dessin au lieu du dessin. Et « Durand & Fils » échappé deux
       fois s'afficherait « Durand &amp; Fils ».

    7. La page ne dépend QUE du rapport : pas d'heure de génération, pas de hasard. Deux appels donnent
       la même chaîne, et deux rapports se comparent avec un simple `diff`.

    Tout le reste (mise en page, couleurs, phrases d'explication) est libre : faites une page que vous
    seriez fier·e d'envoyer. Ouvrez-la dans votre navigateur (étape 6) et regardez-la aussi sur un
    téléphone : un tableau trop large doit défiler dans son cadre, pas élargir toute la page.
    """
    raise NotImplementedError

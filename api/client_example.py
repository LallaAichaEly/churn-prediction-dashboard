"""Appeler l'API depuis Python — avec la seule bibliothèque standard.

    uvicorn api.main:app                 # terminal 1 : démarrer l'API
    python api/client_example.py         # terminal 2 : l'interroger

Le script envoie un client, puis le même client avec chaque type de contrat (en un seul
appel « batch ») pour voir l'effet du contrat sur le risque.
"""

from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

CUSTOMER = {
    "customer_id": "C00042",
    "tenure_months": 4,
    "monthly_charges": 89.9,
    "support_calls": 3,
    "num_products": 1,
    "contract": "Mensuel",
    "payment_method": "Chèque",
    "internet_service": "Fibre",
    "tech_support": "Non",
    "senior": "Non",
}


def call(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(BASE_URL + path, data=data, method=method,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=10) as response:
            return json.load(response)
    except HTTPError as err:  # 422 = données refusées par la validation
        sys.exit(f"Erreur {err.code} sur {path} :\n{err.read().decode('utf-8')}")
    except URLError as err:
        sys.exit(f"API injoignable à {BASE_URL} ({err.reason}). L'avez-vous démarrée ?")


def main() -> None:
    info = call("GET", "/model")
    print(f"Modèle servi : {info['classifier']} (seuil {info['threshold']})\n")

    result = call("POST", "/predict", CUSTOMER)
    verdict = "À RISQUE" if result["at_risk"] else "plutôt fidèle"
    print(f"Client {result['customer_id']} : {result['churn_probability']:.1%} → {verdict}\n")

    variants = [{**CUSTOMER, "customer_id": c, "contract": c} for c in ("Mensuel", "Annuel", "Biannuel")]
    batch = call("POST", "/predict/batch", {"customers": variants})
    print("Et si le même client changeait de contrat ?")
    for p in batch["predictions"]:
        print(f"  {p['customer_id']:<9} {p['churn_probability']:6.1%}  {'█' * round(p['churn_probability'] * 40)}")


if __name__ == "__main__":
    main()

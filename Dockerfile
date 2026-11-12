# Projet 2 — Image Docker de l'API de prédiction du churn.
#
#   docker build -t churn-api .
#   docker run --rm -p 8000:8000 churn-api
#   → http://localhost:8000/docs
FROM python:3.12-slim

# Pas de fichiers .pyc, logs affichés immédiatement.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Les dépendances d'abord : cette couche reste en cache tant que requirements-api.txt
#    ne change pas, ce qui rend les reconstructions bien plus rapides.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2. Le code.
COPY src/ src/
COPY scripts/ scripts/
COPY api/ api/

# 3. Le modèle est entraîné pendant la construction : l'image est autonome et reproductible
#    (données synthétiques, graine fixe). En entreprise, on le téléchargerait plutôt depuis
#    un registre de modèles.
RUN python scripts/train.py

# 4. Ne pas tourner en root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# L'API de churn (api.py) dans un conteneur : la même image tourne sur votre machine, en CI et en production.
#
#   docker build -t churn-api .
#   docker run --rm -p 8000:8000 churn-api        → http://localhost:8000/docs
#
# Vérifié automatiquement à chaque push par .github/workflows/docker.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Les dépendances d'abord : Docker garde cette couche en cache tant que requirements-api.txt ne change pas,
#    donc modifier le code ne relance pas toute l'installation.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2. Le code, puis l'entraînement : le modèle est figé DANS l'image. Démarrage instantané, et la version servie
#    est exactement celle qui a été construite (et testée par la CI).
COPY src/ src/
COPY scripts/train.py scripts/train.py
COPY api.py .
RUN python scripts/train.py --model logistic

# 3. Sécurité : le serveur ne tourne pas en root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

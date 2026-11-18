# Projet 2 — Image de production de l'API de churn.
#
#   docker build -t churn-api .
#   docker run --rm -p 8000:8000 churn-api        → http://localhost:8000/docs
#   docker run --rm -p 8000:8000 -e CHURN_THRESHOLD=0.3 churn-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Les dépendances d'abord : cette couche reste en cache tant que requirements-api.txt ne change pas.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2. Le code, puis l'entraînement : le modèle est figé DANS l'image (reproductible, démarrage rapide).
COPY src/ src/
COPY scripts/ scripts/
COPY api/ api/
RUN python scripts/train.py --model logistic

# 3. Sécurité : on ne tourne pas en root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

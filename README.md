# ChurnGuard — projet MLOps

> **Mission** : industrialiser ce projet en 2 jours selon le cahier des charges fourni
> (`Sujet_ChurnGuard_MLOps.docx`). Vous ne touchez pas à la data science.

[![CI](https://github.com/<user>/churnguard/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/churnguard/actions/workflows/ci.yml)
[![GHCR](https://img.shields.io/badge/ghcr-churnguard-blue)](https://ghcr.io/<user>/churnguard)

## Contexte

Vous reprenez le projet d'une data scientist de TelcoFr. Elle a entraîné un
modèle de prédiction de churn dans un notebook qui marche. Personne d'autre que
elle ne sait le faire tourner.

Votre rôle : transformer ce repo en projet MLOps de production.

## Données

**Telco Customer Churn** (IBM Sample Data, ~960 Ko, 7 043 lignes, 21 colonnes,
licence libre à des fins éducatives).

Le fichier n'est pas commité dans le repo. Pour le télécharger :

```bash
python scripts/download_data.py
```

Le script télécharge le CSV depuis un mirror stable et vérifie son intégrité par
SHA-256.

Sources :
- [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM Sample Data Sets](https://www.ibm.com/community/blogs/datasets/)

## Structure livree

```
.
├── churnguard/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   └── train.py
├── api/
│   └── main.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .github/workflows/
    ├── ci.yml
    └── release.yml
```

## Architecture

```text
            +---------------------+
            |   FastAPI (api)     |
            | /health /predict    |
            +----------+----------+
                       |
                       | MLFLOW_TRACKING_URI
                       v
            +---------------------+
            |   MLflow Server     |
            | tracking + registry |
            +----------+----------+
                       |
                       v
            +---------------------+
            | mlruns + sqlite db  |
            +---------------------+
```

## Démarrage rapide

```bash
# 1. installer les dépendances
uv sync --all-groups

# 2. télécharger les données
python scripts/download_data.py

# 3. tests + couverture
uv run pytest --cov=churnguard --cov-fail-under=70
```

## Entrainement et tracking MLflow

Lancer le serveur :

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Entrainer et logger les 3 modeles :

```bash
python -m churnguard.train --model all
```

Entrainer et enregistrer un modele dans le registry :

```bash
python -m churnguard.train --model rf --register
```

## API FastAPI

Lancer localement :

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Le service charge le modele promu via `models:/churnguard/Production`.

Exemple `curl` single predict :

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender":"Female",
    "SeniorCitizen":0,
    "Partner":"Yes",
    "Dependents":"No",
    "tenure":12,
    "PhoneService":"Yes",
    "MultipleLines":"No",
    "InternetService":"DSL",
    "OnlineSecurity":"Yes",
    "OnlineBackup":"No",
    "DeviceProtection":"No",
    "TechSupport":"No",
    "StreamingTV":"No",
    "StreamingMovies":"No",
    "Contract":"Month-to-month",
    "PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check",
    "MonthlyCharges":50.0,
    "TotalCharges":600.0
  }'
```

Exemple `curl` batch :

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"records":[{"gender":"Male","SeniorCitizen":1,"Partner":"No","Dependents":"No","tenure":4,"PhoneService":"Yes","MultipleLines":"Yes","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":95.0,"TotalCharges":380.0}]}'
```

## Docker Compose

```bash
docker compose up --build
```

Services exposes :

- MLflow UI: http://127.0.0.1:5000
- API: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

## Image Docker

Image publiee via release workflow :

- `ghcr.io/<user>/churnguard:<tag>`
- `ghcr.io/<user>/churnguard:latest`


## Licence

Code : MIT.
Données : IBM Sample Data, voir conditions sur le site IBM.

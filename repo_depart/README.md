# ChurnGuard — projet MLOps

> **Mission** : industrialiser ce projet en 2 jours selon le cahier des charges fourni
> (`Sujet_ChurnGuard_MLOps.docx`). Vous ne touchez pas à la data science.

Si vous avez cloné le dépôt parent, placez-vous dans le dossier applicatif :

```bash
cd repo_depart
```

[![CI](https://github.com/khaoulaAbid/ChurnGuard_MLOPS/actions/workflows/ci.yml/badge.svg)](https://github.com/khaoulaAbid/ChurnGuard_MLOPS/actions/workflows/ci.yml)
[![GHCR](https://img.shields.io/badge/ghcr-churnguard-blue)](https://github.com/khaoulaAbid?tab=packages)

> **CI GitHub** : les workflows doivent être à la racine du dépôt (`/.github/workflows/`). Si votre clone a `repo_depart/` comme sous-dossier et que la racine Git est au-dessus, déplacez `.github` à la racine du remote ou définissez la racine du repo sur le contenu de `repo_depart`, sinon le badge CI peut rester gris.

## Contexte

Vous reprenez le projet d'une data scientist de TelcoFr. Elle a entraîné un
modèle de prédiction de churn dans un notebook qui marche. Personne d'autre que
elle ne sait le faire tourner.

Votre rôle : transformer ce repo en projet MLOps de production.

## Données

**Telco Customer Churn** (IBM Sample Data, ~960 Ko, 7 043 lignes, 21 colonnes,
licence libre à des fins éducatives).

Le fichier n'est en général pas versionné (voir `.gitignore`). Pour le télécharger :

```bash
uv run python scripts/download_data.py
```

Le script télécharge le CSV depuis un miroir stable et vérifie son intégrité par
SHA-256.

Sources :

- [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM Sample Data Sets](https://www.ibm.com/community/blogs/datasets/)

## Structure livrée

```
.
├── churnguard/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   └── train.py
├── api/
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
├── tests/
├── docs/
│   └── mlflow_runs.png
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
            |     /predict/batch  |
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
uv run python scripts/download_data.py

# 3. tests + couverture (seuil 70 % sur le package churnguard)
uv run pytest -m "not integration" --cov=churnguard --cov-fail-under=70
```

## Entraînement et tracking MLflow

Les trois modèles `LogisticRegression`, `RandomForestClassifier` et
`GradientBoostingClassifier` sont entraînés et loggés automatiquement dans
MLflow (paramètres, métriques, signature, exemple d'entrée, artefact modèle).

![3 runs MLflow comparés](docs/mlflow_runs.png)

Pour reproduire : dans un premier terminal, lancer le serveur MLflow :

```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Dans un second terminal (Linux / macOS / Git Bash) :

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run python -m churnguard.train --model all
```

Sous **PowerShell** (Windows) :

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
uv run python -m churnguard.train --model all
```

Puis ouvrir l’interface : [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## API FastAPI

Lancer localement :

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Le service charge le modèle promu via `models:/churnguard/Production`.

Exemple `curl` prédiction unitaire :

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

Services exposés :

- MLflow UI : http://127.0.0.1:5000
- API (Swagger) : http://127.0.0.1:8000/docs
- Santé : http://127.0.0.1:8000/health

## Image Docker

Image publiée via le workflow **Release** (push d’un tag `v*.*.*`) :

- `ghcr.io/khaoulaAbid/churnguard:<tag>`
- `ghcr.io/khaoulaAbid/churnguard:latest`

(Lien packages GitHub : [Packages du compte](https://github.com/khaoulaAbid?tab=packages).)

## Licence

Code : MIT.  
Données : IBM Sample Data, voir conditions sur le site IBM.

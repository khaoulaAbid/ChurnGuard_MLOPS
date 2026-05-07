# ChurnGuard MLOps

Pipeline MLOps complet pour la prédiction du churn client avec entraînement reproductible, versionnement des modèles via MLflow, API FastAPI, conteneurisation Docker et automatisation CI/CD avec GitHub Actions.

---

# Objectifs du projet

- Entraîner un modèle de churn reproductible
- Enregistrer et versionner les modèles avec MLflow Model Registry
- Exposer une API REST d’inférence :
  - `/health`
  - `/predict`
  - `/predict/batch`
- Automatiser la qualité du code avec CI
- Publier automatiquement une image Docker sur GHCR

---

# Stack technique

- Python 3.11
- FastAPI + Uvicorn
- scikit-learn
- MLflow 3.x
- Docker & Docker Compose
- uv
- Ruff
- Mypy
- Pytest
- GitHub Actions

---

# Structure du projet

.
├── .github/workflows/
│   ├── ci.yml
│   └── release.yml
├── repo_depart/
│   ├── api/
│   ├── churnguard/
│   ├── data/
│   ├── scripts/
│   ├── tests/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── pyproject.toml
└── README.md
```

---

# Prérequis

- Git
- Docker Desktop (ou Docker Engine + Docker Compose)
- (Optionnel) Python 3.11 + uv pour lancer les scripts localement

---

# Installation et démarrage

## 1. Cloner le dépôt

```bash
git clone https://github.com/khaoulaAbid/ChurnGuard_MLOPS.git
cd mlops_churnguard/repo_depart
```

## 2. Lancer les services

```bash
docker compose up -d --build
```

## 3. Vérifier les conteneurs

```bash
docker compose ps
```

---

# Accès aux services

- API FastAPI : http://127.0.0.1:8000/docs
- MLflow UI : http://127.0.0.1:5000

---

# Entraînement et enregistrement du modèle

## Définir MLflow Tracking URI

### Windows PowerShell

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

## Entraîner et enregistrer un modèle

```bash
uv run python -m churnguard.train --model gb --register
```

## Promouvoir le modèle dans le Registry

```bash
uv run python scripts/promote_model.py
```

## Vérifier le registre MLflow

```bash
uv run python scripts/inspect_registry.py
```

---

# Tester l’API

## 1. Health check

```bash
curl http://127.0.0.1:8000/health
```

### Réponse attendue

```json
{
  "status": "ok",
  "model": "churnguard",
  "version": "3"
}
```

---

## 2. Prédiction unitaire

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender":"Female",
    "SeniorCitizen":0,
    "Partner":"Yes",
    "Dependents":"No",
    "tenure":12,
    "PhoneService":"Yes",
    "MultipleLines":"No",
    "InternetService":"Fiber optic",
    "OnlineSecurity":"No",
    "OnlineBackup":"Yes",
    "DeviceProtection":"No",
    "TechSupport":"No",
    "StreamingTV":"Yes",
    "StreamingMovies":"Yes",
    "Contract":"Month-to-month",
    "PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check",
    "MonthlyCharges":89.1,
    "TotalCharges":1070.4
  }'
```

---

## 3. Prédiction batch

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "records":[
      {
        "gender":"Female",
        "SeniorCitizen":0,
        "Partner":"Yes",
        "Dependents":"No",
        "tenure":12,
        "PhoneService":"Yes",
        "MultipleLines":"No",
        "InternetService":"Fiber optic",
        "OnlineSecurity":"No",
        "OnlineBackup":"Yes",
        "DeviceProtection":"No",
        "TechSupport":"No",
        "StreamingTV":"Yes",
        "StreamingMovies":"Yes",
        "Contract":"Month-to-month",
        "PaperlessBilling":"Yes",
        "PaymentMethod":"Electronic check",
        "MonthlyCharges":89.1,
        "TotalCharges":1070.4
      },
      {
        "gender":"Male",
        "SeniorCitizen":0,
        "Partner":"No",
        "Dependents":"No",
        "tenure":2,
        "PhoneService":"Yes",
        "MultipleLines":"No",
        "InternetService":"DSL",
        "OnlineSecurity":"No",
        "OnlineBackup":"No",
        "DeviceProtection":"No",
        "TechSupport":"No",
        "StreamingTV":"No",
        "StreamingMovies":"No",
        "Contract":"One year",
        "PaperlessBilling":"No",
        "PaymentMethod":"Mailed check",
        "MonthlyCharges":49.9,
        "TotalCharges":99.8
      }
    ]
  }'
```

---

# Démonstration du pipeline MLOps

Pipeline attendu :

```text
clone → docker compose up → curl predict → modification → push → CI verte
```

## Étapes

1. Cloner le dépôt
2. Lancer Docker Compose
3. Tester `/health`
4. Tester `/predict`
5. Modifier une partie du code
6. Push des changements
7. Vérifier la CI sur GitHub Actions

## Commandes Git

```bash
git add .
git commit -m "demo: small API update"
git push
```

---

# CI/CD

## CI — GitHub Actions

Workflow : `.github/workflows/ci.yml`

### Jobs exécutés

- lint
- typecheck
- test-unit
- test-integration
- build

---

## Release Docker — GHCR

Workflow : `.github/workflows/release.yml`

### Déclenchement

Le workflow de release se lance automatiquement lors de la création d’un tag :

```text
v*.*.*
```

### Exemple

```bash
git tag v0.1.0
git push origin v0.1.0
```

---

# Tester l’image publiée

```bash
docker pull ghcr.io/khaoulaabid/churnguard:latest
docker pull ghcr.io/khaoulaabid/churnguard:v0.1.0
```

---

# Résultats attendus

- API opérationnelle via Docker
- Modèle enregistré et promu avec MLflow
- Endpoints `/predict` et `/predict/batch` fonctionnels
- Pipeline CI vert sur GitHub Actions
- Image Docker publiée automatiquement sur GHCR
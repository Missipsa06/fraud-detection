# Fraud Detection

Détection de fraudes bancaires avec LightGBM, feature engineering, optimisation de seuil par contrainte métier, tuning automatique avec Optuna et tracking MLflow.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284 807 transactions, 492 fraudes (0.17%).

Placer le fichier `creditcard.csv` dans `data/raw/`.

## Structure

```
fraud_detection/
├── data/raw/              ← dataset (non versionné)
├── artifacts/             ← modèle sérialisé (non versionné)
├── src/
│   ├── config.py          ← paramètres centralisés
│   ├── data.py            ← chargement et split
│   ├── features.py        ← feature engineering
│   ├── model.py           ← entraînement LightGBM
│   ├── evaluation.py      ← métriques et optimisation du seuil
│   ├── tuning.py          ← tuning Optuna + cross-validation
│   ├── pipeline.py        ← orchestration + tracking MLflow
│   ├── serve.py           ← sauvegarde du modèle pour l'API
│   └── api.py             ← API REST FastAPI
├── tests/
│   ├── test_features.py   ← tests unitaires feature engineering
│   ├── test_evaluation.py ← tests unitaires évaluation
│   ├── test_data.py       ← tests unitaires split
│   └── test_api.py        ← tests unitaires API REST
├── exploration.ipynb      ← visualisations
├── best_params.json       ← meilleurs paramètres (généré par tuning)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Tuning des hyperparamètres (optionnel)

Lance une recherche automatique des meilleurs paramètres via Optuna (30 essais, cross-validation 5-fold) :

```bash
python -m src.tuning
```

Les meilleurs paramètres sont sauvegardés dans `best_params.json` et chargés automatiquement par le pipeline.

### 2. Lancer le pipeline

```bash
python -m src.pipeline
```

Si `best_params.json` existe, il est utilisé automatiquement. Sinon, les paramètres par défaut de `config.py` sont utilisés.

### 3. Visualiser les expériences

```bash
mlflow ui
```

Puis ouvrir **http://127.0.0.1:5000**.

## Features engineering

| Feature | Description |
|---|---|
| `log_amount` | Log du montant — réduit l'asymétrie de la distribution |
| `is_round_amount` | 1 si montant rond (ex: 50.00) — pattern fréquent en fraude |
| `hour` | Heure de la journée extraite de `Time` |
| `is_night` | 1 si transaction entre 0h et 6h |

## Tuning automatique

Le module `src/tuning.py` utilise **Optuna** pour optimiser les hyperparamètres LightGBM :

- Objectif : maximiser le **PR-AUC** moyen sur 5 folds
- Paramètres explorés : `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`
- Évaluation : **StratifiedKFold** (5 folds) pour garantir la représentation des fraudes dans chaque fold

## API REST

Le modèle est exposé via une API FastAPI permettant de scorer des transactions en temps réel.

### Démarrage

```bash
# 1. Sauvegarder le modèle entraîné
python -m src.serve

# 2. Lancer l'API
uvicorn src.api:app --reload
```

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Vérifie que l'API est up |
| `GET` | `/sample` | Retourne une transaction du jeu de validation |
| `GET` | `/sample?fraud=true` | Retourne un exemple de fraude |
| `POST` | `/predict` | Score une transaction |

### Exemple

```bash
# Récupérer un exemple de fraude
GET http://127.0.0.1:8000/sample?fraud=true

# Le soumettre au modèle
POST http://127.0.0.1:8000/predict
→ { "fraud": true, "score": 0.9341 }
```

L'interface Swagger interactive est disponible sur **http://127.0.0.1:8000/docs**.

## Docker

### Prérequis
Avoir généré les artifacts au préalable :
```bash
python -m src.serve
```

### Build et lancement
```bash
docker compose up --build
```

L'API est disponible sur **http://localhost:8000/docs**.

```bash
# En arrière-plan
docker compose up --build -d

# Arrêt
docker compose down
```

## Monitoring (Data Drift)

Détecte si les distributions des features en production dérivent par rapport aux données d'entraînement.

```bash
python -m src.monitoring
```

Génère `reports/drift_report.html` — ouvrir dans le navigateur pour visualiser le rapport : statut drift par feature, statistique KS et p-value.

La détection utilise le **test de Kolmogorov-Smirnov** par feature (p-value < 0.05 et KS stat ≥ 0.1). La séparation est chronologique : les 80% premières transactions servent de référence, les 20% les plus récentes simulent la production.

## Tests unitaires

```bash
pytest tests/ -v
```

27 tests couvrant `build_features`, `find_best_threshold`, `evaluate_model`, `split_data` et les endpoints API — sans dépendance au dataset.

## Contrainte métier

Le seuil de décision est optimisé pour maximiser le **recall** tout en maintenant une **précision ≥ 40%** (`MIN_PRECISION = 0.4`).

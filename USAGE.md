# Guide d'utilisation

## Prérequis

1. Placer le dataset dans `data/raw/creditcard.csv`
   ([télécharger sur Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

---

## Étape 1 — Tuning des hyperparamètres (optionnel)

Recherche automatique des meilleurs paramètres LightGBM via Optuna (30 essais, 5-fold CV).

```bash
python -m src.tuning
```

Résultat : `best_params.json` (chargé automatiquement par les étapes suivantes).

---

## Étape 2 — Entraîner et sauvegarder le modèle

```bash
python -m src.serve
```

Résultat : `artifacts/model.joblib`, `artifacts/threshold.json`, `artifacts/samples.json`.

---

## Étape 3 — Lancer l'API

```bash
uvicorn src.api:app --reload
```

### Tester via Swagger (recommandé)

Ouvrir **http://localhost:8000/docs** dans le navigateur.

1. Appeler `GET /sample?fraud=true` pour récupérer un exemple de fraude
2. Copier le JSON retourné
3. Le coller dans `POST /predict` et exécuter

### Tester via curl

```bash
# Vérifier que l'API est up
curl http://localhost:8000/health

# Récupérer un exemple de fraude
curl "http://localhost:8000/sample?fraud=true"

# Soumettre une transaction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 406.0, "Amount": 149.62, "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25, "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12, "V26": -0.29, "V27": 0.07, "V28": 0.06}'
```

Réponse attendue :
```json
{ "fraud": true, "score": 0.9341 }
```

---

## Étape 4 — Tracking MLflow (optionnel)

Entraîne le modèle et enregistre les métriques dans MLflow :

```bash
python -m src.pipeline
```

Visualiser les expériences :
```bash
mlflow ui
```

Ouvrir **http://localhost:5000**.

---

## Étape 5 — Monitoring (Data Drift)

Détecte si les distributions des features ont dérivé entre données historiques et récentes :

```bash
python -m src.monitoring
```

Résultat : `reports/drift_report.html` — ouvrir dans le navigateur.

---

## Étape 6 — Tests unitaires

```bash
pytest tests/ -v
```

27 tests couvrant le feature engineering, l'évaluation, le split et les endpoints API.

---

## Via Docker

```bash
# Prérequis : avoir exécuté l'étape 2 au préalable
docker compose up --build
```

API disponible sur **http://localhost:8000/docs**.

```bash
# En arrière-plan
docker compose up --build -d

# Arrêt
docker compose down
```

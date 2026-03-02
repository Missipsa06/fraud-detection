# Fraud Detection

Détection de fraudes bancaires avec LightGBM, feature engineering, optimisation de seuil par contrainte métier, tuning automatique avec Optuna et tracking MLflow.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284 807 transactions, 492 fraudes (0.17%).

Placer le fichier `creditcard.csv` dans `data/raw/`.

## Structure

```
fraud_detection/
├── data/raw/              ← dataset (non versionné)
├── src/
│   ├── config.py          ← paramètres centralisés
│   ├── data.py            ← chargement et split
│   ├── features.py        ← feature engineering
│   ├── model.py           ← entraînement LightGBM
│   ├── evaluation.py      ← métriques et optimisation du seuil
│   ├── tuning.py          ← tuning Optuna + cross-validation
│   └── pipeline.py        ← orchestration
├── exploration.ipynb      ← visualisations
├── best_params.json       ← meilleurs paramètres (généré par tuning)
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

## Features engineerées

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

## Contrainte métier

Le seuil de décision est optimisé pour maximiser le **recall** tout en maintenant une **précision ≥ 40%** (`MIN_PRECISION = 0.4`).

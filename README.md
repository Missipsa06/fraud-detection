# Fraud Detection

Détection de fraudes bancaires avec LightGBM, optimisation de seuil par contrainte métier et tracking MLflow.

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
│   └── pipeline.py        ← orchestration
├── exploration.ipynb      ← visualisations
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Lancer le pipeline

```bash
python -m src.pipeline
```

## Features engineerées

| Feature | Description |
|---|---|
| `log_amount` | Log du montant (réduit l'asymétrie) |
| `is_round_amount` | 1 si montant rond (ex: 50.00) |
| `hour` | Heure de la journée extraite de `Time` |
| `is_night` | 1 si transaction entre 0h et 6h |

## Contrainte métier

Le seuil de décision est optimisé pour maximiser le **recall** tout en maintenant une **précision ≥ 40%** (`MIN_PRECISION = 0.4`).

## Tracking des expériences

Les métriques et modèles sont loggés avec MLflow :

```bash
mlflow ui
```

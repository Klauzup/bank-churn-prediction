# Bank Churn Prediction

[🇷🇺 Russian version](README.md)

Predicting bank customer churn. Model built with `scikit-learn`, API served via `FastAPI`.

## Task

Determine whether a client will leave the bank or stay, based on their data. Binary classification: `1` — churned, `0` — retained.

**Data:** [Bank Customer Churn](https://www.kaggle.com/datasets/santoshd3/bank-customers) — 10,000 clients.

**Features after cleaning:**
- `credit_score`, `age`, `balance`, `products_number`, `active_member`, `is_many_products`, `gender` — numeric.
- `country` — categorical (France, Spain, Germany).

## Model

`RandomForestClassifier` with `class_weight="balanced"`, `max_depth=10`, `n_estimators=300`.

Chosen after comparison with `LogisticRegression`: F1 **0.61** vs **0.56**, precision **0.53** vs **0.45**.

**Test metrics:**

| Class | Precision | Recall | F1-score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Retained)** | 0.92 | 0.84 | 0.88 | 1593 |
| **1 (Churned)** | 0.53 | 0.72 | 0.61 | 407 |
| **accuracy** | | | **0.82** | **2000** |

**What it means:**
- **Recall "Churned" = 0.72** — catches 72% of clients who leave. The key metric for the bank.
- **Precision "Churned" = 0.53** — out of all predicted as "churned", 53% actually left. The rest are false alarms (cheap).
- **F1 "Churned" = 0.61** — balance.

## Project structure

```
bank-churn-prediction/
├── data/
│   ├── raw/                    # original data
│   └── processed/              # cleaned data
├── models/
│   └── model.pkl               # trained model (Pipeline)
├── src/
│   ├── prepare_data.ipynb      # cleaning and EDA
│   ├── train.py                # training and hyperparameter tuning
│   ├── predict.py              # testing model on new data
│   └── api.py                  # FastAPI service
├── requirements.txt
├── README.md
└── README.en.md
```

## Installation

```bash
git clone <repo-url>
cd bank-churn-prediction

python -m venv .venv
source .venv/bin/activate        # Linux / Mac
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Run API

```bash
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` — Swagger UI. Find `/predict`, click **Try it out**, enter client data, **Execute**.

## Example request

```json
{
  "credit_score": 600,
  "country": "France",
  "gender": 0,
  "age": 40,
  "balance": 50000.0,
  "products_number": 2,
  "active_member": 1,
  "is_many_products": 0
}
```

**Response:**

```json
{
  "predict": 0,
  "probability": 0.12
}
```

`predict = 0` — client stays. `probability = 0.12` — 12% chance of churn.

## Stack

- Python 3.12
- scikit-learn
- FastAPI
- pandas
- joblib

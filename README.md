# Bank Churn Prediction 

[🇬🇧 English version](README.en.md)

Предсказание оттока клиентов банка. Модель на `scikit-learn`, API на `FastAPI`.

## Задача

Определить, уйдёт клиент или останется, на основе его данных. Бинарная классификация: `1` — ушёл, `0` — остался.

**Данные:** [Bank Customer Churn](https://www.kaggle.com/datasets/santoshd3/bank-customers) — 10 000 клиентов.

**Признаки после очистки:**
- `credit_score`, `age`, `balance`, `products_number`, `active_member`, `is_many_products`, `gender` — числа.
- `country` — категория (France, Spain, Germany).

## Модель

`RandomForestClassifier` с `class_weight="balanced"`, `max_depth=10`, `n_estimators=300`.

Выбрана после сравнения с `LogisticRegression`: F1 **0.61** vs **0.56**, precision **0.53** vs **0.45**.

**Метрики на test:**

| Класс | Precision | Recall | F1-score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Остался)** | 0.92 | 0.84 | 0.88 | 1593 |
| **1 (Ушёл)** | 0.53 | 0.72 | 0.61 | 407 |
| **accuracy** | | | **0.82** | **2000** |

**Что это значит:**
- **Recall «Ушёл» = 0.72** — ловит 72% уходящих. Главная метрика для банка.
- **Precision «Ушёл» = 0.53** — из всех, кого назвала «уходящими», 53% реально ушли. Остальные — ложные тревоги (дёшево).
- **F1 «Ушёл» = 0.61** — баланс.

## Структура проекта

```
bank-churn-prediction/
├── data/
│   ├── raw/                    # исходные данные
│   └── processed/              # очищенные данные
├── models/
│   └── model.pkl               # обученная модель (Pipeline)
├── src/
│   ├── prepare_data.ipynb      # очистка и EDA
│   ├── train.py                # обучение и подбор гиперпараметров
│   ├── predict.py              # проверка модели на новых данных
│   └── api.py                  # FastAPI-сервис
├── requirements.txt
├── README.md
└── README.en.md
```

## Установка

```bash
git clone <repo-url>
cd bank-churn-prediction

python -m venv .venv
source .venv/bin/activate        # Linux / Mac
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Запуск API

```bash
uvicorn src.api:app --reload
```

Открой `http://127.0.0.1:8000/docs` — Swagger UI. Найди `/predict`, нажми **Try it out**, введи данные клиента, **Execute**.

## Пример запроса

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

**Ответ:**

```json
{
  "predict": 0,
  "probability": 0.12
}
```

`predict = 0` — клиент останется. `probability = 0.12` — вероятность ухода 12%.

## Стек

- Python 3.12
- scikit-learn
- FastAPI
- pandas
- joblib

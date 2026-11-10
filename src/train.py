import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib 

df = pd.read_csv("./data/processed/bank_churn_clean.csv")

X = df.drop(columns="churn")
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessing = ColumnTransformer([
    ("ohe", ohe, ["country"]),
    ("num", StandardScaler(), ["credit_score", "age", "balance"])
], remainder="passthrough")

pipe = Pipeline([
    ("prep", preprocessing),
    ("model", RandomForestClassifier())
])

param_grid = {
    "model__class_weight": ["balanced", None],
    "model__n_estimators": [100, 300, 500, 1000],
    "model__max_depth": [3, 5, 10, 15]
}

grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", verbose=2)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)

y_pred = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(grid.best_estimator_, "./models/model.pkl")
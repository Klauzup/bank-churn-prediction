import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

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
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print(f1_score(y_test, y_pred))
print(roc_auc_score(y_test, y_proba))
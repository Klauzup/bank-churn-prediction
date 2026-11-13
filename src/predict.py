import pandas as pd
import joblib
from sklearn.metrics import classification_report

model = joblib.load("./models/model.pkl")
df = pd.read_csv("./data/processed/bank_churn_clean.csv")

X = df.drop(columns="churn")
y = df["churn"]

print(f"{classification_report(model.predict(X), y)}")
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd 

app = FastAPI()
model = joblib.load("./models/model.pkl")

class Client(BaseModel):
    credit_score: int
    country: str
    gender: int
    age: int
    balance: float
    products_number: int
    active_member: int
    is_many_products: int

@app.post("/predict")
def predict(client: Client):
    data = pd.DataFrame([client.model_dump()])
    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0, 1]
    return {"predict": int(pred), "probability": float(proba)}
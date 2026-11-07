import pandas as pd
from joblib import load
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

model = load("pipeline.joblib")

app = FastAPI()


class Vino(BaseModel):
    alcohol: float = 14.23
    malic_acid: float = 1.71
    ash: float = 2.43
    alcalinity_of_ash: float = 15.6
    magnesium: float = 127.0
    total_phenols: float = 2.8
    flavanoids: float = 3.06
    nonflavanoid_phenols: float = 0.28
    proanthocyanins: float = 2.29
    color_intensity: float = 5.64
    hue: float = 1.04
    od280_od315_of_diluted_wines: float = 3.92
    proline: float = 1065.0


@app.post("/predict")
def predict(vinos: list[Vino]):
    try:
        df = pd.DataFrame([v.model_dump() for v in vinos])
        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"Universidad EIA": "MLOps"}

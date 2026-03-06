import json
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import predict_proba
from .features import build_features
from .config import MODEL_PATH, THRESHOLD_PATH, SAMPLES_PATH

# État global chargé au démarrage
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise RuntimeError(
            "Artifacts introuvables. Lance d'abord : python -m src.serve"
        )
    _state["model"] = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        _state["threshold"] = json.load(f)["threshold"]
    with open(SAMPLES_PATH) as f:
        _state["samples"] = json.load(f)
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float
    V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(tx: Transaction):
    df = pd.DataFrame([tx.model_dump()])
    df = build_features(df)
    score = float(predict_proba(_state["model"], df)[0])
    fraud = score >= _state["threshold"]
    return {"fraud": fraud, "score": round(score, 4)}


@app.get("/sample")
def sample(fraud: bool = False):
    """
    Retourne une transaction réelle du jeu de validation.
    ?fraud=true  → exemple de fraude confirmée
    ?fraud=false → exemple de transaction légitime (défaut)
    """
    key = "fraud" if fraud else "legit"
    examples = _state["samples"][key]
    if not examples:
        raise HTTPException(status_code=404, detail="Aucun exemple trouvé.")
    return examples[0]

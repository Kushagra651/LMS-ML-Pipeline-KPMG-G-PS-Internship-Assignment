"""
FastAPI prediction endpoint — grading model + doubt triage with routing.
Inference only. Models are trained and exported from the notebook (see models/).

Run:
    pip install fastapi uvicorn joblib
    uvicorn main:app --reload
Docs at: http://127.0.0.1:8000/docs
"""

from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODELS_DIR = Path("models")
app = FastAPI(title="LMS ML Pipeline API")

grading_model = joblib.load(MODELS_DIR / "grading_model.pkl")
grading_features = joblib.load(MODELS_DIR / "grading_feature_names.pkl")
tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
urgency_model = joblib.load(MODELS_DIR / "urgency_model.pkl")  # calibrated
routing_cfg = joblib.load(MODELS_DIR / "routing_config.pkl")   # {"threshold": ...}


class GradingRequest(BaseModel):
    test_pass_rate: float
    cyclomatic_complexity: float
    lint_error_count: float
    execution_time_ms: float
    documentation_score: float
    memory_usage_mb: float


class DoubtRequest(BaseModel):
    text: str


@app.post("/predict/grading")
def predict_grading(req: GradingRequest):
    row = req.dict()
    row["risk_score"] = row["cyclomatic_complexity"] * (1 + row["lint_error_count"]) / (row["test_pass_rate"] + 1e-3)
    X = [[row[f] for f in grading_features]]
    pred = grading_model.predict(X)[0]
    proba = grading_model.predict_proba(X)[0].tolist()
    return {"quality_label": int(pred), "probabilities": proba}


@app.post("/predict/triage")
def predict_triage(req: DoubtRequest):
    X = tfidf.transform([req.text])
    proba = urgency_model.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    pred = urgency_model.classes_[idx]
    confidence = float(proba[idx])
    threshold = routing_cfg["threshold"]
    decision = "auto-approve" if confidence >= threshold else "teacher review"
    return {"urgency": pred, "confidence": confidence, "threshold": threshold, "decision": decision}


@app.get("/health")
def health():
    return {"status": "ok"}
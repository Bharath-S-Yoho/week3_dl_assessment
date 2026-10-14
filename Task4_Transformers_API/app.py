from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_PATH = Path(__file__).parent / "saved_model"

app = FastAPI(title="Customer Feedback Sentiment API")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer feedback text")

# Load the model once when the API starts, not on every request.
sentiment_model = None

@app.on_event("startup")
def load_model():
    global sentiment_model
    if MODEL_PATH.exists():
        sentiment_model = pipeline(
            "text-classification",
            model=str(MODEL_PATH),
            tokenizer=str(MODEL_PATH)
        )

@app.post("/predict")
def predict(request: PredictRequest):
    if sentiment_model is None:
        return {
            "error": "Fine-tuned model not found. Run python finetune.py first."
        }
    result = sentiment_model(request.text)[0]
    return {
        "label": result["label"],
        "confidence": round(float(result["score"]), 4)
    }

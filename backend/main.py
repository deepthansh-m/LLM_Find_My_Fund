import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.train import train_index
from backend.predict import predict_fund
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="LLM Find My Fund", description="APIs to match fund queries to Indian securities.")

# Allow requests from your frontend (adjust the origin if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class TrainRequest(BaseModel):
    # Path or additional configuration for the dataset
    dataset_path: str


@app.post("/train")
def train_model(request: TrainRequest):
    """
    Trigger training on the dataset.
    The train_index function will load the dataset, calculate embeddings, and build a FAISS index.
    """
    try:
        train_index(dataset_path=request.dataset_path)
        return {"status": "Training complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(query_request: QueryRequest):
    """
    Resolve an input query to the closest matching fund.
    """
    query = query_request.query
    if not os.path.exists("backend/fund_index.faiss"):
        return {
            "matched_fund": {
                "result": "Mock Fund ABC",
                "confidence": "High",
                "note": "This is a mock response. Train the model to get real predictions."
            }
        }
    try:
        result = predict_fund(query)
        return {"matched_fund": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

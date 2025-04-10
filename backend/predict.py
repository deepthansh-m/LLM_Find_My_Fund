import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from backend.utils import load_index

INDEX_FILE = "backend/fund_index.faiss"
METADATA_FILE = "backend/fund_metadata.pkl"
MODEL_NAME = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"

# Load the sentence transformer model (this may be done lazily if needed)
model = SentenceTransformer(MODEL_NAME)


def predict_fund(query: str, k: int = 1):
    """
    Given a query, compute its embedding, search the FAISS index, and return the top match.
    """
    # Load the FAISS index and metadata (if not loaded in memory, you can cache this)
    index = load_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        df = pickle.load(f)

    # Compute the embedding of the query
    query_embedding = model.encode([query])

    # FAISS expects float32 arrays
    D, I = index.search(np.array(query_embedding, dtype='float32'), k)

    # I[0][0] is the index of the closest match in the dataframe
    if I.size == 0 or I[0][0] < 0:
        return "No match found"

    matched_row = df.iloc[I[0][0]]
    # Return the fund name (or a structure with additional metadata)
    return matched_row["fund_name"]

import os
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from backend.utils import save_index, load_index

# Global variables (adapt as needed)
INDEX_FILE = "backend/fund_index.faiss"
METADATA_FILE = "backend/fund_metadata.pkl"
MODEL_NAME = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"


def build_corpus_embeddings(dataset_path: str):
    """
    Load the dataset and compute embeddings for fund names (and optionally integrate metadata).
    The dataset is assumed to be a CSV file with columns like 'fund_name', 'category', 'fund_house', etc.
    """
    # Load dataset (adapt file format as needed)
    df = pd.read_csv(dataset_path)

    # Combine fund name with metadata to improve context matching
    # You can modify the concatenation as per metadata available in your dataset.
    df["combined_text"] = df["fund_name"]
    if "category" in df.columns:
        df["combined_text"] += " " + df["category"]
    if "fund_house" in df.columns:
        df["combined_text"] += " " + df["fund_house"]

    corpus = df["combined_text"].tolist()

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(corpus, show_progress_bar=True)

    return embeddings, df


def train_index(dataset_path: str):
    """
    Train/fine-tune the search index using the dataset.
    This creates a FAISS index for similarity search and saves the associated metadata.
    """
    print("Loading dataset and building embeddings...")
    embeddings, df = build_corpus_embeddings(dataset_path)

    dim = embeddings.shape[1]
    # Initialize a FAISS index (L2 similarity search; adjust for cosine if needed)
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype='float32'))

    # Save the FAISS index and the associated metadata (e.g., dataframe)
    save_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(df, f)

    print("Training complete. Index and metadata saved.")


if __name__ == "__main__":
    # Example usage: python train.py --dataset_path data/indian_funds.csv
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the fund dataset CSV file")
    args = parser.parse_args()
    train_index(args.dataset_path)

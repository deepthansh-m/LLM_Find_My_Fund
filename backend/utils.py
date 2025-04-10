import faiss
import pickle

def save_index(index, filepath):
    faiss.write_index(index, filepath)
    print(f"FAISS index saved to {filepath}")


def load_index(filepath):
    index = faiss.read_index(filepath)
    print(f"FAISS index loaded from {filepath}")
    return index

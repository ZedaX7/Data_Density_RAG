import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

path = "./data/faiss_encoded/"
device = "cuda:3"

def retrieve_relevant_docs(query, data_name="full_[64-64-4-64-64]", top_k=5):
    # Load index & metadata
    index = faiss.read_index(path + data_name + "/rag_index_" + data_name + ".faiss")
    with open(path + data_name + "/rag_metadata_" + data_name + ".json") as f:
        metadata = json.load(f)

    # Encode query
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                device=device, 
                                cache_folder="./pre_trained/hf_cache")
    
    query_vec = model.encode([query], 
                             device=device,
                             convert_to_numpy=True)

    # Search
    D, I = index.search(query_vec, top_k)
    return [metadata[i] for i in I[0]]

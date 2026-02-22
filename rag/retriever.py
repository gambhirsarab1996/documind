import faiss
import numpy as np

def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

def retrieve(query_embedding, index, metadata, top_k=8):
    _, I = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return [metadata[i] for i in I[0]]
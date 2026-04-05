import faiss
import numpy as np
import re

def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

def _tokenize(text: str) -> set:
    stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "or", "it", "that", "this", "was", "for"}
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return {t for t in tokens if t not in stopwords and len(t) > 2}

def _relevance_score(query_tokens: set, chunk_text: str) -> float:
    chunk_tokens = _tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens | chunk_tokens)

def retrieve(query_embedding, index, metadata, top_k=8):
    # Step 1: fetch more candidates than needed, then filter down
    candidate_k = min(top_k * 3, len(metadata))  
    distances, indices = index.search(
        np.array([query_embedding]).astype("float32"), candidate_k
    )

    # Step 2: convert L2 distances to similarity scores (lower = better in L2)
    # normalize so worst distance maps to 0, best maps to 1
    dists = distances[0]
    max_dist = dists.max() if dists.max() > 0 else 1.0
    similarity_scores = 1 - (dists / max_dist)

    # Step 3: apply similarity threshold — drop weak matches
    SIMILARITY_THRESHOLD = 0.4  # tune this: higher = stricter
    candidates = []
    for idx, sim_score in zip(indices[0], similarity_scores):
        if sim_score >= SIMILARITY_THRESHOLD:
            candidates.append((idx, sim_score))

    if not candidates:
        # fallback: return top 3 even if below threshold
        candidates = list(zip(indices[0][:3], similarity_scores[:3]))

    # Step 4: re-rank by combining vector similarity + keyword overlap
    query_tokens = _tokenize(
        " ".join(metadata[idx]["text"] for idx, _ in candidates[:3])  # rough query reconstruction
    )
    
    scored = []
    for idx, vec_score in candidates:
        chunk = metadata[idx]
        keyword_score = _relevance_score(query_tokens, chunk["text"])
        combined = (0.7 * vec_score) + (0.3 * keyword_score)  # weight vector similarity higher
        scored.append((chunk, combined))

    # Step 5: sort by combined score, cap at top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_k]]
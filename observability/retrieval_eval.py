import re
from collections import Counter


def _tokenize(text: str) -> set:
    """Lowercase alphanumeric tokens, ignore stopwords."""
    stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "or", "it", "that", "this", "was", "for"}
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return {t for t in tokens if t not in stopwords and len(t) > 2}


def _relevance_score(query_tokens: set, chunk_text: str) -> float:
    """Jaccard-style overlap between query tokens and chunk tokens."""
    chunk_tokens = _tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens | chunk_tokens)


def evaluate_retrieval(trace) -> dict:
    num_chunks = trace["num_chunks"]
    query_tokens = _tokenize(trace["query"])

    if num_chunks == 0:
        return {
            "score": "poor",
            "reason": "No chunks retrieved.",
            "relevance_scores": [],
            "avg_relevance": 0.0
        }

    # Score each chunk for relevance to the original query
    scores = [
        _relevance_score(query_tokens, chunk["text"])
        for chunk in trace["chunks"]
    ]
    avg_relevance = sum(scores) / len(scores)
    low_relevance_count = sum(1 for s in scores if s < 0.05)

    # Determine quality signal
    if avg_relevance < 0.05:
        score = "poor"
        reason = "Retrieved chunks have very low relevance to the query."
    elif num_chunks < 3:
        score = "low"
        reason = "Too few chunks retrieved."
    elif num_chunks > 8 and avg_relevance < 0.1:
        score = "overfetch"
        reason = "Too many chunks with weak relevance — noisy retrieval."
    elif low_relevance_count > num_chunks // 2:
        score = "noisy"
        reason = f"{low_relevance_count}/{num_chunks} chunks have low relevance."
    else:
        score = "good"
        reason = "Retrieval looks balanced and relevant."

    return {
        "score": score,
        "reason": reason,
        "relevance_scores": [round(s, 3) for s in scores],
        "avg_relevance": round(avg_relevance, 3)
    }

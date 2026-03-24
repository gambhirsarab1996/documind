def build_trace(
    query: str,
    sub_queries: list,
    retrieved_chunks: list,
    final_prompt: str,
    answer: str,
    token_usage: dict | None = None
) -> dict:
    return {
        "query": query,
        "sub_queries": sub_queries,
        "num_chunks": len(retrieved_chunks),
        "chunks": retrieved_chunks,
        "prompt": final_prompt,
        "answer": answer,
        "token_usage": token_usage  # expects {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
    }

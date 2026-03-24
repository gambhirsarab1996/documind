def recommend(issues: list[dict]) -> list[dict]:
    """
    Returns prioritized suggestions based on detected issues.
    Each suggestion carries the originating issue and severity.
    """
    RECOMMENDATIONS = {
        "retrieval_failure": {
            "suggestion": "No chunks were retrieved. Check your FAISS index, embedding model, or query preprocessing.",
            "priority": "high"
        },
        "retrieval_quality": {
            "suggestion": "Retrieved chunks have low relevance. Consider tuning chunk size, overlap, or switching to a better embedding model.",
            "priority": "high"
        },
        "over_retrieval": {
            "suggestion": "Reduce top_k — you're fetching more chunks than needed, increasing cost and prompt noise.",
            "priority": "medium"
        },
        "hallucination": {
            "suggestion": "Answer contains unsupported claims. Tighten your system prompt to be more strictly grounded, or improve retrieval relevance.",
            "priority": "high"
        },
        "partial_grounding": {
            "suggestion": "Answer is partially supported. Review unsupported claims and consider fetching more targeted chunks.",
            "priority": "medium"
        },
        "grounding_eval_failed": {
            "suggestion": "The grounding judge failed to respond correctly. Check your OpenAI API connection or prompt format.",
            "priority": "low"
        },
        "high_token_cost": {
            "suggestion": "Token usage is high. Reduce top_k, shorten chunk size, or summarize context before passing to the LLM.",
            "priority": "low"
        }
    }

    suggestions = []
    seen = set()

    # Sort by severity so high-priority issues come first
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    sorted_issues = sorted(issues, key=lambda i: severity_order.get(i.get("severity", "info"), 2))

    for issue in sorted_issues:
        issue_name = issue.get("issue")
        if issue_name in RECOMMENDATIONS and issue_name not in seen:
            seen.add(issue_name)
            rec = RECOMMENDATIONS[issue_name].copy()
            rec["issue"] = issue_name
            rec["severity"] = issue.get("severity")
            suggestions.append(rec)

    return suggestions

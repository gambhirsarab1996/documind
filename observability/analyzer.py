def analyze(trace, retrieval_eval, grounding_eval, cost_eval) -> list[dict]:
    """
    Returns a list of issues, each with a name and severity level.
    Severity: critical > warning > info
    """
    issues = []

    # ── Retrieval Issues ──────────────────────────────────────────────
    retrieval_score = retrieval_eval.get("score")

    if retrieval_score == "poor":
        issues.append({
            "issue": "retrieval_failure",
            "severity": "critical",
            "detail": retrieval_eval.get("reason")
        })
    elif retrieval_score in ("low", "noisy"):
        issues.append({
            "issue": "retrieval_quality",
            "severity": "warning",
            "detail": retrieval_eval.get("reason")
        })
    elif retrieval_score == "overfetch":
        issues.append({
            "issue": "over_retrieval",
            "severity": "info",
            "detail": retrieval_eval.get("reason")
        })

    # ── Grounding Issues ─────────────────────────────────────────────
    # grounding_eval is now a parsed dict, not raw text
    verdict = grounding_eval.get("verdict", "unknown")
    confidence = grounding_eval.get("confidence", 0.0)

    if verdict == "not_supported":
        issues.append({
            "issue": "hallucination",
            "severity": "critical",
            "detail": grounding_eval.get("explanation"),
            "unsupported_claims": grounding_eval.get("unsupported_claims", [])
        })
    elif verdict == "partial" or confidence < 0.6:
        issues.append({
            "issue": "partial_grounding",
            "severity": "warning",
            "detail": grounding_eval.get("explanation"),
            "unsupported_claims": grounding_eval.get("unsupported_claims", [])
        })
    elif verdict == "unknown":
        issues.append({
            "issue": "grounding_eval_failed",
            "severity": "info",
            "detail": "Could not evaluate grounding — judge call failed."
        })

    # ── Cost Issues (only if not already flagged by retrieval) ────────
    if cost_eval.get("efficiency") == "low":
        already_flagged = any(i["issue"] == "over_retrieval" for i in issues)
        if not already_flagged:
            issues.append({
                "issue": "high_token_cost",
                "severity": "info",
                "detail": cost_eval.get("reason")
            })

    return issues

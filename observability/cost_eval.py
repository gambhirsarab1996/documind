# Approximate cost per 1M tokens for gpt-4o-mini (update as needed)
COST_PER_1M_INPUT = 0.15
COST_PER_1M_OUTPUT = 0.60


def evaluate_cost(trace) -> dict:
    usage = trace.get("token_usage")
    num_chunks = trace["num_chunks"]

    if usage:
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        estimated_cost_usd = (
            (input_tokens / 1_000_000) * COST_PER_1M_INPUT +
            (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
        )

        if total_tokens > 6000:
            efficiency = "low"
            reason = f"High token usage ({total_tokens} tokens) — consider reducing top_k or chunk size."
        elif total_tokens > 3000:
            efficiency = "moderate"
            reason = f"Moderate token usage ({total_tokens} tokens)."
        else:
            efficiency = "good"
            reason = f"Efficient usage ({total_tokens} tokens)."

        return {
            "efficiency": efficiency,
            "reason": reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 6)
        }

    # Fallback if token_usage not passed in
    if num_chunks > 8:
        return {
            "efficiency": "low",
            "reason": "Too many chunks likely driving high token usage (no usage data available).",
            "estimated_cost_usd": None
        }

    return {
        "efficiency": "good",
        "reason": "Chunk count looks efficient (no usage data available).",
        "estimated_cost_usd": None
    }

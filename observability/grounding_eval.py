import json


def evaluate_grounding(trace, client):
    context = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(trace["chunks"])
    )

    prompt = f"""You are an evaluation judge. Assess whether the answer is grounded in the context below.

Context:
{context}

Answer:
{trace["answer"]}

Return ONLY valid JSON in this exact shape:
{{
  "verdict": "supported" | "partial" | "not_supported",
  "confidence": 0.0-1.0,
  "explanation": "one sentence",
  "unsupported_claims": ["claim1", "claim2"]  // empty list if fully supported
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    try:
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, KeyError):
        return {
            "verdict": "unknown",
            "confidence": 0.0,
            "explanation": "Grounding evaluation failed to parse.",
            "unsupported_claims": []
        }

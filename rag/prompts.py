import json

def build_prompt(context, question):
    return f"""
You are a document intelligence assistant.

Answer ONLY using the context below.
If the answer is not present, say:
"Not found in uploaded documents."

Always cite document name and page number.

Context:
{context}

Question:
{question}

Provide:
1. Answer
2. Supporting Evidence
3. Confidence Level (High/Medium/Low)
"""


def plan_query(question, client):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a retrieval planner for an advanced RAG system.

Step 1:
Classify the user question into one of:
- lookup
- multi_hop
- aggregation
- structural

Step 2:
Generate retrieval-optimized search queries.

Important rules:
- Break complex questions into smaller independent search queries.
- Each query should target a specific concept or document section.
- Queries must maximize recall (use synonyms if helpful).
- Keep queries concise.
- Avoid vague wording.
- For aggregation questions, include:
  - One broad document-wide query
  - Additional concept-level queries to ensure coverage.

Step 3:
Recommend an appropriate top_k value:
- lookup → 5
- multi_hop → 6–8
- aggregation → 12–18
- structural → 15+

Return JSON only in this format:

{
  "type": "...",
  "queries": ["...", "..."],
  "top_k": number
}
"""
            },
            {"role": "user", "content": question}
        ],
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "type": "lookup",
            "queries": [question],
            "top_k": 6
        }
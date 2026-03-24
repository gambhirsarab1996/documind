from observability.retrieval_eval import evaluate_retrieval
from observability.grounding_eval import evaluate_grounding
from observability.cost_eval import evaluate_cost
from observability.analyzer import analyze
from observability.recommender import recommend


def run_evaluation(trace: dict, client) -> dict:
    retrieval = evaluate_retrieval(trace)
    grounding = evaluate_grounding(trace, client)
    cost = evaluate_cost(trace)

    issues = analyze(trace, retrieval, grounding, cost)
    suggestions = recommend(issues)

    return {
        "retrieval": retrieval,
        "grounding": grounding,
        "cost": cost,
        "issues": issues,
        "suggestions": suggestions
    }

import streamlit as st

st.set_page_config(
    page_title="DocuMind — Observability",
    page_icon="🔭",
    layout="wide"
)

# ─────────────────────────────────────────
# GUARD — No traces yet
# ─────────────────────────────────────────

traces = st.session_state.get("traces", [])

st.title("🔭 Observability Dashboard")
st.caption("Per-query evaluation traces + consolidated session summary")
st.divider()

if not traces:
    st.info("No traces recorded yet. Go to the **Chat** page, upload a document, and ask some questions.")
    st.stop()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

SCORE_COLORS = {
    "good":      ("🟢", "green"),
    "low":       ("🟡", "orange"),
    "noisy":     ("🟡", "orange"),
    "overfetch": ("🟠", "orange"),
    "poor":      ("🔴", "red"),
}

VERDICT_COLORS = {
    "supported":     ("🟢", "green"),
    "partial":       ("🟡", "orange"),
    "not_supported": ("🔴", "red"),
    "unknown":       ("⚪", "gray"),
}

EFFICIENCY_COLORS = {
    "good":     ("🟢", "green"),
    "moderate": ("🟡", "orange"),
    "low":      ("🔴", "red"),
}

SEVERITY_BADGE = {
    "critical": "🔴 Critical",
    "warning":  "🟡 Warning",
    "info":     "🔵 Info",
}

PRIORITY_BADGE = {
    "high":   "🔴 High",
    "medium": "🟡 Medium",
    "low":    "🔵 Low",
}

def score_badge(score: str) -> str:
    emoji, _ = SCORE_COLORS.get(score, ("⚪", "gray"))
    return f"{emoji} `{score}`"

def verdict_badge(verdict: str) -> str:
    emoji, _ = VERDICT_COLORS.get(verdict, ("⚪", "gray"))
    return f"{emoji} `{verdict}`"

def efficiency_badge(eff: str) -> str:
    emoji, _ = EFFICIENCY_COLORS.get(eff, ("⚪", "gray"))
    return f"{emoji} `{eff}`"


# ─────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────

tab_consolidated, tab_per_query = st.tabs(["📊 Consolidated View", "🔍 Per-Query Traces"])


# ═══════════════════════════════════════════
# TAB 1 — CONSOLIDATED VIEW
# ═══════════════════════════════════════════

with tab_consolidated:

    st.subheader(f"Session Summary — {len(traces)} Queries")
    st.divider()

    # ── Top-level KPI cards ──────────────────
    total_tokens = 0
    total_cost = 0.0
    retrieval_scores = []
    verdicts = []
    all_issues = []
    all_suggestions = []

    for record in traces:
        ev = record["evaluation"]

        # Tokens & cost
        cost_data = ev.get("cost", {})
        total_tokens += cost_data.get("total_tokens", 0)
        cost_val = cost_data.get("estimated_cost_usd")
        if cost_val:
            total_cost += cost_val

        # Retrieval
        retrieval_scores.append(ev.get("retrieval", {}).get("score", "unknown"))

        # Grounding
        verdicts.append(ev.get("grounding", {}).get("verdict", "unknown"))

        # Issues & suggestions (deduplicated globally)
        for issue in ev.get("issues", []):
            if issue not in all_issues:
                all_issues.append(issue)
        for sug in ev.get("suggestions", []):
            if sug not in all_suggestions:
                all_suggestions.append(sug)

    good_retrievals = retrieval_scores.count("good")
    supported_verdicts = verdicts.count("supported")
    critical_issues = [i for i in all_issues if i.get("severity") == "critical"]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Total Queries", len(traces))

    with kpi2:
        st.metric(
            "Good Retrieval",
            f"{good_retrievals}/{len(traces)}",
            delta=None
        )

    with kpi3:
        st.metric(
            "Fully Grounded",
            f"{supported_verdicts}/{len(traces)}",
            delta=None
        )

    with kpi4:
        st.metric(
            "Total Tokens Used",
            f"{total_tokens:,}",
            delta=f"~${total_cost:.4f}" if total_cost > 0 else None
        )

    st.divider()

    # ── Per-query summary table ──────────────
    st.subheader("Query Overview")

    col_headers = st.columns([3, 1.5, 1.5, 1.5, 1])
    col_headers[0].markdown("**Query**")
    col_headers[1].markdown("**Retrieval**")
    col_headers[2].markdown("**Grounding**")
    col_headers[3].markdown("**Tokens**")
    col_headers[4].markdown("**Issues**")

    for i, record in enumerate(traces):
        ev = record["evaluation"]
        retrieval_score = ev.get("retrieval", {}).get("score", "—")
        verdict = ev.get("grounding", {}).get("verdict", "—")
        tokens = ev.get("cost", {}).get("total_tokens", "—")
        issue_count = len(ev.get("issues", []))
        critical = any(iss.get("severity") == "critical" for iss in ev.get("issues", []))

        cols = st.columns([3, 1.5, 1.5, 1.5, 1])
        query_short = record["query"][:70] + "…" if len(record["query"]) > 70 else record["query"]
        cols[0].markdown(f"`Q{i+1}` {query_short}")
        cols[1].markdown(score_badge(retrieval_score))
        cols[2].markdown(verdict_badge(verdict))
        cols[3].markdown(f"`{tokens:,}`" if isinstance(tokens, int) else f"`{tokens}`")
        cols[4].markdown(f"{'🔴' if critical else '🟡' if issue_count > 0 else '🟢'} {issue_count}")

    st.divider()

    # ── Consolidated Issues ──────────────────
    if all_issues:
        st.subheader("⚠️ Issues Detected Across Session")
        for issue in all_issues:
            badge = SEVERITY_BADGE.get(issue.get("severity", "info"), "🔵 Info")
            with st.expander(f"{badge} — `{issue.get('issue')}`"):
                st.markdown(f"**Detail:** {issue.get('detail', '—')}")
                if issue.get("unsupported_claims"):
                    st.markdown("**Unsupported Claims:**")
                    for claim in issue["unsupported_claims"]:
                        st.markdown(f"- {claim}")
        st.divider()

    # ── Consolidated Recommendations ─────────
    if all_suggestions:
        st.subheader("💡 Recommendations")
        for sug in all_suggestions:
            priority = PRIORITY_BADGE.get(sug.get("priority", "low"), "🔵 Low")
            st.markdown(f"{priority} — {sug.get('suggestion')}")


# ═══════════════════════════════════════════
# TAB 2 — PER-QUERY TRACES
# ═══════════════════════════════════════════

with tab_per_query:

    # Query selector
    query_labels = [
        f"Q{i+1}: {r['query'][:60]}{'…' if len(r['query']) > 60 else ''}"
        for i, r in enumerate(traces)
    ]

    selected_label = st.selectbox(
        "Select a query to inspect:",
        query_labels,
        key="query_selector"
    )

    selected_idx = query_labels.index(selected_label)
    record = traces[selected_idx]
    trace = record["trace"]
    ev = record["evaluation"]

    st.divider()

    # ── Query + Answer ───────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### 🙋 Query")
        st.info(record["query"])

        if trace.get("sub_queries"):
            st.markdown("**Sub-queries planned:**")
            for sq in trace["sub_queries"]:
                st.markdown(f"- {sq}")

    with right:
        st.markdown("#### 🤖 Answer")
        st.success(record["answer"])

    st.divider()

    # ── Evaluation Panels ────────────────────
    st.markdown("#### 📋 Evaluation")

    ev_col1, ev_col2, ev_col3 = st.columns(3)

    # Retrieval
    with ev_col1:
        ret = ev.get("retrieval", {})
        st.markdown("**🗂 Retrieval**")
        st.markdown(f"Score: {score_badge(ret.get('score', '—'))}")
        st.markdown(f"Reason: {ret.get('reason', '—')}")
        st.markdown(f"Chunks: `{trace.get('num_chunks', '—')}`")
        avg_rel = ret.get("avg_relevance")
        if avg_rel is not None:
            st.markdown(f"Avg Relevance: `{avg_rel}`")

        scores = ret.get("relevance_scores", [])
        if scores:
            with st.expander("Per-chunk relevance scores"):
                for j, s in enumerate(scores):
                    bar = "█" * int(s * 40)
                    st.markdown(f"`Chunk {j+1}` {bar} `{s}`")

    # Grounding
    with ev_col2:
        grd = ev.get("grounding", {})
        st.markdown("**🔍 Grounding**")
        st.markdown(f"Verdict: {verdict_badge(grd.get('verdict', '—'))}")
        conf = grd.get("confidence")
        if conf is not None:
            st.markdown(f"Confidence: `{conf}`")
        st.markdown(f"Explanation: {grd.get('explanation', '—')}")
        unsupported = grd.get("unsupported_claims", [])
        if unsupported:
            st.markdown("**Unsupported claims:**")
            for claim in unsupported:
                st.markdown(f"- ⚠️ {claim}")

    # Cost
    with ev_col3:
        cost = ev.get("cost", {})
        st.markdown("**💰 Cost & Efficiency**")
        st.markdown(f"Efficiency: {efficiency_badge(cost.get('efficiency', '—'))}")
        st.markdown(f"Reason: {cost.get('reason', '—')}")
        if cost.get("total_tokens"):
            st.markdown(f"Input tokens: `{cost.get('input_tokens', '—'):,}`")
            st.markdown(f"Output tokens: `{cost.get('output_tokens', '—'):,}`")
            st.markdown(f"Total tokens: `{cost.get('total_tokens', '—'):,}`")
        if cost.get("estimated_cost_usd") is not None:
            st.markdown(f"Est. cost: `${cost['estimated_cost_usd']:.6f}`")

    st.divider()

    # ── Issues & Suggestions ─────────────────
    issues = ev.get("issues", [])
    suggestions = ev.get("suggestions", [])

    issue_col, sug_col = st.columns(2)

    with issue_col:
        st.markdown("#### ⚠️ Issues")
        if issues:
            for issue in issues:
                badge = SEVERITY_BADGE.get(issue.get("severity", "info"), "🔵 Info")
                with st.expander(f"{badge} — `{issue.get('issue')}`"):
                    st.markdown(f"**Detail:** {issue.get('detail', '—')}")
                    if issue.get("unsupported_claims"):
                        st.markdown("**Unsupported Claims:**")
                        for claim in issue["unsupported_claims"]:
                            st.markdown(f"- {claim}")
        else:
            st.success("No issues detected for this query.")

    with sug_col:
        st.markdown("#### 💡 Suggestions")
        if suggestions:
            for sug in suggestions:
                priority = PRIORITY_BADGE.get(sug.get("priority", "low"), "🔵 Low")
                st.markdown(f"{priority} — {sug.get('suggestion')}")
        else:
            st.success("No suggestions — this query looks healthy.")

    st.divider()

    # ── Retrieved Chunks ─────────────────────
    with st.expander(f"🔍 Retrieved Chunks ({trace.get('num_chunks', 0)})"):
        for j, chunk in enumerate(trace.get("chunks", [])):
            rel_scores = ev.get("retrieval", {}).get("relevance_scores", [])
            rel = rel_scores[j] if j < len(rel_scores) else None
            rel_str = f" · Relevance: `{rel}`" if rel is not None else ""
            st.markdown(f"**Chunk {j+1} — Page {chunk['page']} · {chunk['doc_name']}**{rel_str}")
            st.text(chunk["text"])
            st.divider()
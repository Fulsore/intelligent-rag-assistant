"""
services/rag_pipeline.py
Healthcare KIS - Full RAG Pipeline

Orchestrates the complete query flow:
  1. Vector search (retrieve chunks)
  2. Rule-based context filtering
  3. Centrality ranking
  4. Derivability pre-filter
  5. Context building + LLM call
  6. Derivability scoring (post-generation)
  7. Trust score aggregation
  8. Score store logging
  9. Return rich response dict
"""

from typing import Dict, Any

from services.vector_search import vector_search
from services.context_builder import context_builder
from services.llm import llm_service

from rules.rule_engine import rule_engine
from rules.derivability_filter import derivability_filter
from rules.threshold_rule import thresholds

from scoring.centrality import centrality_scorer
from scoring.derivability import derivability_scorer
from scoring.trust_score import trust_calculator
from scoring.score_store import score_store


def run_rag_pipeline(question: str) -> Dict[str, Any]:
    """
    Full RAG pipeline for a single question.
    Returns a rich dict with answer + all scoring metadata.
    """

    # ── Step 1: Vector Search ──────────────────────────────────────────
    raw_results = vector_search.search(
        question,
        top_k=thresholds.max_context_chunks,
        score_threshold=thresholds.min_retrieval_similarity,
    )

    retrieval_scores = [score for _, score in raw_results]
    retrieved_docs = [doc for doc, _ in raw_results]

    # ── Step 2: Rule-Based Context Filtering ──────────────────────────
    rule_result = rule_engine.evaluate(question, retrieved_docs)
    filtered_docs = rule_result["filtered_docs"]

    # ── Step 3: Centrality Ranking ────────────────────────────────────
    ranked_with_scores = centrality_scorer.rank_by_centrality(
        question, filtered_docs
    )

    # ── Step 4: Derivability Pre-Filter ──────────────────────────────
    pre_filtered_docs, _ = derivability_filter.filter(
        question,
        [doc for doc, _ in ranked_with_scores],
    )

    # Rebuild ranked list with pre-filtered docs
    ranked_filtered = [
        (doc, score)
        for doc, score in ranked_with_scores
        if doc in pre_filtered_docs
    ]

    # ── Step 5: Early Abstain Check (pre-LLM) ─────────────────────────
    if not ranked_filtered:
        abstain_answer = thresholds.get_abstain_message()
        return _build_response(
            question=question,
            answer=abstain_answer,
            trust_report={
                "trust_score": 0.0,
                "confidence_label": "Very Low",
                "confidence_color": "red",
                "components": {},
                "derivability_breakdown": {},
                "sources_used": 0,
                "rule_filter": rule_result,
            },
            rule_result=rule_result,
            abstained=True,
        )

    # ── Step 6: Build Prompt & Call LLM ──────────────────────────────
    prompt = context_builder.build_prompt(
        question=question,
        ranked_docs=ranked_filtered,
        requires_disclaimer=bool(rule_result["flagged_keywords"]),
    )

    raw_answer = llm_service.generate(prompt)

    # Prepend safety prefix if flagged
    safe_prefix = rule_engine.get_safe_answer_prefix(
        rule_result["flagged_keywords"]
    )
    answer = safe_prefix + raw_answer

    # ── Step 7: Derivability Scoring (post-generation) ────────────────
    deriv_score, deriv_breakdown = derivability_scorer.score(
        question=question,
        answer=raw_answer,
        context_docs=pre_filtered_docs,
    )

    # ── Step 8: Trust Score ───────────────────────────────────────────
    trust_report = trust_calculator.calculate(
        question=question,
        answer=raw_answer,
        context_docs=pre_filtered_docs,
        retrieval_scores=retrieval_scores,
        derivability_score=deriv_score,
        derivability_breakdown=deriv_breakdown,
        rule_filter_result=rule_result,
    )

    # ── Step 9: Post-answer Abstain Check ────────────────────────────
    if thresholds.should_abstain(deriv_score, trust_report["trust_score"]):
        answer = thresholds.get_abstain_message()
        abstained = True
    else:
        abstained = False

    # Append medical disclaimer
    answer = context_builder.inject_disclaimer(answer)

    # ── Step 10: Log to Score Store ───────────────────────────────────
    score_store.add(question, trust_report)

    return _build_response(
        question=question,
        answer=answer,
        trust_report=trust_report,
        rule_result=rule_result,
        abstained=abstained,
        retrieval_scores=retrieval_scores,
        deriv_score=deriv_score,
        deriv_breakdown=deriv_breakdown,
        sources_count=len(pre_filtered_docs),
    )


def _build_response(
    question: str,
    answer: str,
    trust_report: Dict,
    rule_result: Dict,
    abstained: bool = False,
    retrieval_scores=None,
    deriv_score: float = 0.0,
    deriv_breakdown: Dict = None,
    sources_count: int = 0,
) -> Dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "abstained": abstained,
        "trust": {
            "score": trust_report.get("trust_score", 0.0),
            "label": trust_report.get("confidence_label", "Unknown"),
            "color": trust_report.get("confidence_color", "gray"),
            "components": trust_report.get("components", {}),
        },
        "derivability": {
            "score": round(deriv_score, 3),
            "is_derivable": derivability_scorer.is_derivable(deriv_score),
            "breakdown": deriv_breakdown or {},
        },
        "retrieval": {
            "scores": retrieval_scores or [],
            "avg_score": round(
                sum(retrieval_scores) / len(retrieval_scores), 3
            ) if retrieval_scores else 0.0,
            "chunks_used": sources_count,
        },
        "rules": {
            "passed": rule_result.get("passed", False),
            "flagged_keywords": rule_result.get("flagged_keywords", []),
            "out_of_scope": rule_result.get("out_of_scope", False),
            "medical_relevance": rule_result.get("medical_relevance", False),
            "warnings": rule_result.get("warnings", []),
            "notes": rule_result.get("rule_notes", ""),
        },
        "session_stats": score_store.get_stats(),
    }
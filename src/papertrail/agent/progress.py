"""Honest, bounded UI progress milestones derived from completed LangGraph nodes.

Percentages describe completed *workflow checkpoints*, not wall-clock time,
model-weight download progress, tokens, PDF bytes, or scientific confidence.
Only emit a new event after the corresponding graph node completes.
"""
from __future__ import annotations


def research_checkpoint(state: dict) -> tuple[int, str] | None:
    """Return (percent, safe label) for the last *completed* graph node."""
    trace = state.get('trace') or []
    if not trace or not isinstance(trace[-1], dict):
        return None
    node = trace[-1].get('node')
    if node == 'validate_request':
        if state.get('status') == 'invalid_request':
            return 5, 'Request validation stopped — inspect the result'
        return 5, 'Research request validated'
    if node == 'plan_research':
        return 15, 'Research question broken into search tasks'
    if node == 'select_source':
        return 20, ('Paper source selected' if state.get('status') == 'source_selected'
                    else 'Source selection checked — inspect the result')
    if node == 'discover':
        return 25, 'arXiv discovery finished'
    if node == 'prepare':
        if state.get('status') not in (None, 'prepared_integrity_checked'):
            return 35, 'Source preparation stopped — inspect the result'
        return 35, 'Local source and index checked'
    if node == 'retrieve':
        planned = sum(1 for item in state.get('plan', [])
                      if item.get('proposed_tool') == 'hybrid_retrieval')
        complete = len(state.get('evidence', []))
        if planned:
            complete = min(complete, planned)
            label = f'Retrieved evidence for {complete}/{planned} research tasks'
            if state.get('status') in {'tool_failed', 'tool_budget_exhausted', 'query_plan_invalid'}:
                label = f'Retrieval stopped after {complete}/{planned} tasks — inspect the result'
            return 35 + (40 * complete // planned), label
        return 75, 'Retrieval stage completed (no search tasks)'
    if node == 'assess_evidence':
        return 83, 'Evidence coverage assessed (structural checks only)'
    if node == 'retry_evidence':
        return 87, 'Evidence retry completed'
    if node == 'synthesize_answer':
        # In Preview this only prepares the answer input; there is NO LLM call.
        return 95, 'Answer stage completed or Preview prepared'
    if node == 'finalize':
        return 100, 'Workflow finished — inspect the result status'
    return None

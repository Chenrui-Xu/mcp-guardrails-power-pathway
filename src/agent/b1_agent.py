"""
B1: Prompt-only agent (Vertex AI / Gemini).
"""

from __future__ import annotations
import time

from benchmark.cases import PowerPathwayCase
from agent.base_agent import (
    AgentRun, SYSTEM_PROMPT, make_model,
    parse_recommendation, _count_tokens, _with_retry,
)


def run_b1(case: PowerPathwayCase) -> AgentRun:
    model = make_model(system_prompt=SYSTEM_PROMPT)
    t0 = time.time()
    try:
        response = _with_retry(model.generate_content, case.to_prompt_context())
        raw = response.text
        tokens = _count_tokens(response)
        rec = parse_recommendation(raw)
        return AgentRun(
            case_id=case.case_id,
            agent_type="B1",
            final_recommendation=rec,
            raw_response=raw,
            total_tokens=tokens,
            latency_sec=time.time() - t0,
        )
    except Exception as e:
        return AgentRun(
            case_id=case.case_id,
            agent_type="B1",
            final_recommendation=None,
            raw_response="",
            latency_sec=time.time() - t0,
            error=str(e),
        )

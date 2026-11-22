"""
B2: Prompt + self-critique agent (Vertex AI / Gemini).
"""

from __future__ import annotations
import time

from benchmark.cases import PowerPathwayCase
from agent.base_agent import (
    AgentRun, SYSTEM_PROMPT, make_model,
    parse_recommendation, _count_tokens, _with_retry,
)

CRITIQUE_PROMPT = """\
Review your recommendation above. Check each constraint explicitly:

1. CAPACITY: Does firm supply (CF × nameplate MW) meet peak load?
   Grid is capped at the confirmed interconnect limit.
2. COST: Does total capex (sum of $/kW × 1000 × MW / 1e6) fit the budget?
3. CARBON: Is blended weighted carbon intensity within the target?
   (Weighted by CF × MW for each tech)
4. RELIABILITY: Does the mix achieve the required uptime?

If any constraint is violated, revise your recommendation.
Output the final answer as a JSON block as specified.
"""


def run_b2(case: PowerPathwayCase, critique_rounds: int = 2) -> AgentRun:
    model = make_model(system_prompt=SYSTEM_PROMPT)
    t0 = time.time()
    total_tokens = 0
    try:
        chat = model.start_chat()
        response = _with_retry(chat.send_message, case.to_prompt_context())
        raw = response.text
        total_tokens += _count_tokens(response)

        for _ in range(critique_rounds):
            response = _with_retry(chat.send_message, CRITIQUE_PROMPT)
            raw = response.text
            total_tokens += _count_tokens(response)

        rec = parse_recommendation(raw)
        return AgentRun(
            case_id=case.case_id,
            agent_type="B2",
            final_recommendation=rec,
            raw_response=raw,
            total_tokens=total_tokens,
            latency_sec=time.time() - t0,
        )
    except Exception as e:
        return AgentRun(
            case_id=case.case_id,
            agent_type="B2",
            final_recommendation=None,
            raw_response="",
            total_tokens=total_tokens,
            latency_sec=time.time() - t0,
            error=str(e),
        )

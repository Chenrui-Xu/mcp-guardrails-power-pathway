"""
G1: Guarded agent with MCP-style pre-execution enforcement (Vertex AI / Gemini).
"""

from __future__ import annotations
import json
import time

import vertexai
from vertexai.generative_models import (
    FunctionDeclaration, GenerativeModel, Part, Tool,
)

from benchmark.cases import PowerPathwayCase
from agent.base_agent import (
    AgentRun, SYSTEM_PROMPT, GEMINI_MODEL, VERTEX_PROJECT, VERTEX_LOCATION,
    parse_recommendation, _count_tokens, _with_retry,
)
from guardrails.mcp_gateway import MCPGateway

GUARDED_SYSTEM = SYSTEM_PROMPT + """

IMPORTANT GUARDRAIL RULES:
- You MUST call validate_capacity, validate_cost, validate_carbon, AND validate_reliability
  before calling submit_final_recommendation.
- If the gateway blocks a tool call, read the error message and adjust your approach.
- Never invent grid capacity or technology specs not provided in the case.
- If required data is missing, set feasible to false and explain in missing_data.
- For supply_mix, provide a JSON string like: {"grid": 20, "wind_bess": 30}
"""

# ── Tool definitions (JSON Schema dicts) ─────────────────────────────────────

_SUPPLY_MIX = {
    "type": "string",
    "description": 'JSON-encoded supply mix, e.g. \'{"grid": 20, "wind_bess": 30}\'',
}

TOOLS = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="get_case_info",
            description="Return structured summary of the planning case.",
            parameters={"type": "object", "properties": {}},
        ),
        FunctionDeclaration(
            name="validate_capacity",
            description=(
                "Check whether the proposed supply mix provides enough firm capacity "
                "for all load phases. MUST be called before score_pathway or final recommendation."
            ),
            parameters={
                "type": "object",
                "properties": {"supply_mix": _SUPPLY_MIX},
                "required": ["supply_mix"],
            },
        ),
        FunctionDeclaration(
            name="validate_cost",
            description=(
                "Check total capex of the supply mix against the budget ceiling. "
                "MUST be called before final recommendation."
            ),
            parameters={
                "type": "object",
                "properties": {"supply_mix": _SUPPLY_MIX},
                "required": ["supply_mix"],
            },
        ),
        FunctionDeclaration(
            name="validate_carbon",
            description=(
                "Check blended carbon intensity against the case target. "
                "MUST be called before final recommendation."
            ),
            parameters={
                "type": "object",
                "properties": {"supply_mix": _SUPPLY_MIX},
                "required": ["supply_mix"],
            },
        ),
        FunctionDeclaration(
            name="validate_reliability",
            description=(
                "Estimate system reliability and compare to the target. "
                "MUST be called before final recommendation."
            ),
            parameters={
                "type": "object",
                "properties": {"supply_mix": _SUPPLY_MIX},
                "required": ["supply_mix"],
            },
        ),
        FunctionDeclaration(
            name="score_pathway",
            description=(
                "Compute a weighted composite score (0-100) for a supply mix. "
                "Requires all four validators to have been called first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "supply_mix": _SUPPLY_MIX,
                    "weights": {
                        "type": "string",
                        "description": 'Optional JSON: {"speed":0.25,"cost":0.25,"environment":0.25,"resilience":0.25}',
                    },
                },
                "required": ["supply_mix"],
            },
        ),
        FunctionDeclaration(
            name="submit_final_recommendation",
            description=(
                "Submit your final recommendation. The gateway will block this if "
                "any required validator has not been called."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "feasible": {"type": "boolean", "description": "True if a valid supply mix exists."},
                    "supply_mix": _SUPPLY_MIX,
                    "reasoning": {"type": "string"},
                    "missing_data": {"type": "string"},
                },
                "required": ["feasible", "reasoning"],
            },
        ),
    ]
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_supply_mix(raw) -> dict:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return dict(raw) if raw else {}


def _fn_response_part(name: str, result: dict) -> Part:
    return Part.from_function_response(name=name, response=result)


# ── Main agent ────────────────────────────────────────────────────────────────

def run_g1(case: PowerPathwayCase, max_iterations: int = 12) -> AgentRun:
    import warnings
    warnings.filterwarnings("ignore")
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)

    gateway = MCPGateway()
    t0 = time.time()
    total_tokens = 0
    tool_calls_log: list[dict] = []
    blocked_calls = 0
    final_rec = None
    raw_final = ""

    model = GenerativeModel(
        GEMINI_MODEL,
        system_instruction=GUARDED_SYSTEM,
        tools=[TOOLS],
    )
    chat = model.start_chat()

    try:
        response = _with_retry(chat.send_message, case.to_prompt_context())
        total_tokens += _count_tokens(response)

        for _ in range(max_iterations):
            all_parts = response.candidates[0].content.parts if response.candidates else []
            fn_call_parts = [
                p for p in all_parts
                if hasattr(p, "function_call") and p.function_call and p.function_call.name
            ]

            if not fn_call_parts:
                try:
                    raw_final = response.text or ""
                except Exception:
                    raw_final = ""
                final_rec = parse_recommendation(raw_final)
                break

            response_parts: list[Part] = []

            for part in fn_call_parts:
                fn = part.function_call
                tool_name = fn.name
                raw_args = dict(fn.args) if fn.args else {}

                kwargs: dict = {}
                if "supply_mix" in raw_args:
                    kwargs["supply_mix"] = _parse_supply_mix(raw_args["supply_mix"])
                if "weights" in raw_args:
                    try:
                        kwargs["weights"] = json.loads(raw_args["weights"])
                    except Exception:
                        pass

                # ── submit_final_recommendation ────────────────────────────
                if tool_name == "submit_final_recommendation":
                    allowed, reason = gateway.check_recommendation_allowed()
                    if not allowed:
                        blocked_calls += 1
                        tool_calls_log.append({"tool": tool_name, "status": "BLOCKED", "msg": reason})
                        response_parts.append(_fn_response_part(tool_name, {
                            "status": "BLOCKED", "message": reason
                        }))
                    else:
                        supply_mix = _parse_supply_mix(raw_args.get("supply_mix", "{}"))
                        final_rec = {
                            "feasible": raw_args.get("feasible"),
                            "supply_mix": supply_mix,
                            "reasoning": raw_args.get("reasoning", ""),
                            "missing_data": raw_args.get("missing_data", ""),
                        }
                        raw_final = json.dumps(final_rec)
                        tool_calls_log.append({"tool": tool_name, "status": "ACCEPTED"})
                        response_parts.append(_fn_response_part(tool_name, {
                            "status": "ACCEPTED", "message": "Recommendation recorded."
                        }))
                    continue

                # ── Route through MCP gateway ──────────────────────────────
                kwargs["case"] = case
                result = gateway.call(tool_name, **kwargs)

                if result.status == "BLOCKED":
                    blocked_calls += 1

                tool_calls_log.append({
                    "tool": tool_name,
                    "status": result.status,
                    "msg": result.message[:120],
                })
                response_parts.append(_fn_response_part(tool_name, {
                    "status": result.status,
                    "message": result.message,
                    "details": result.details,
                }))

            if final_rec is not None:
                break

            response = _with_retry(chat.send_message, response_parts)
            total_tokens += _count_tokens(response)

        return AgentRun(
            case_id=case.case_id,
            agent_type="G1",
            final_recommendation=final_rec,
            raw_response=raw_final,
            tool_calls=tool_calls_log,
            blocked_calls=blocked_calls,
            total_tokens=total_tokens,
            latency_sec=time.time() - t0,
        )

    except Exception as e:
        return AgentRun(
            case_id=case.case_id,
            agent_type="G1",
            final_recommendation=None,
            raw_response="",
            tool_calls=tool_calls_log,
            blocked_calls=blocked_calls,
            total_tokens=total_tokens,
            latency_sec=time.time() - t0,
            error=str(e),
        )

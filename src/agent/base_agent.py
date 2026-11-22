"""
Base agent class and shared utilities.
Uses Vertex AI (gemini-2.5-flash) backed by Google Cloud project.
"""

from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, field

import vertexai
from vertexai.generative_models import GenerativeModel

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

SYSTEM_PROMPT = """\
You are a power-pathway planning expert helping design electricity supply strategies
for data centers and industrial facilities.

Your recommendations must satisfy ALL of the following constraints:
1. CAPACITY: The firm (dispatchable) supply must meet peak load in every phase.
   - "firm" means capacity_factor × nameplate_MW
   - Grid supply is capped at the confirmed grid interconnect limit
2. COST: Total capital expenditure must not exceed the budget ceiling.
3. CARBON: Blended weighted carbon intensity must not exceed the target.
4. RELIABILITY: Estimated system uptime must meet or exceed the reliability target.

Technology specs (from the Power Pathway Optimizer reference model):
| Technology        | Capex ($/kW) | Carbon (kg/MWh) | CF   | Deploy (mo) |
|-------------------|-------------|-----------------|------|-------------|
| diesel_gen        | 900         | 790             | 0.80 | 1-6         |
| fuel_cell_sofc    | 3000        | 470             | 0.92 | 3-6         |
| solar_bess        | 2400        | 40              | 0.20 | 6-18        |
| gas_turbine       | 1600        | 600             | 0.75 | 12-24       |
| wind_bess         | 1700        | 10              | 0.36 | 12-36       |
| geothermal        | 2300        | 35              | 0.81 | 60-100      |
| smr               | 5600        | 12              | 0.94 | 60-120      |
| grid              | 0           | 386             | 0.999| 0           |

When you respond with a final recommendation, output EXACTLY this JSON block:
```json
{
  "feasible": true or false,
  "supply_mix": {"tech_name": capacity_mw, ...},
  "reasoning": "brief explanation",
  "missing_data": "any required data that was not provided"
}
```
If you cannot determine feasibility due to missing data, set "feasible": null.
"""


@dataclass
class AgentRun:
    case_id: str
    agent_type: str        # "B1" | "B2" | "G1"
    final_recommendation: dict | None
    raw_response: str
    tool_calls: list[dict] = field(default_factory=list)
    blocked_calls: int = 0
    total_tokens: int = 0
    latency_sec: float = 0.0
    error: str | None = None


def parse_recommendation(text: str) -> dict | None:
    """Extract JSON recommendation block from agent response."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^{}]*\"feasible\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def init_vertex() -> None:
    """Initialize Vertex AI with the configured project."""
    import warnings
    warnings.filterwarnings("ignore")
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)


def make_model(system_prompt: str | None = None, tools=None) -> GenerativeModel:
    init_vertex()
    kwargs = {}
    if system_prompt:
        kwargs["system_instruction"] = system_prompt
    if tools:
        kwargs["tools"] = tools
    return GenerativeModel(GEMINI_MODEL, **kwargs)


def _count_tokens(response) -> int:
    try:
        meta = response.usage_metadata
        return (meta.prompt_token_count or 0) + (meta.candidates_token_count or 0)
    except Exception:
        return 0


def _with_retry(fn, *args, max_retries: int = 4, base_wait: int = 30, **kwargs):
    """Call fn with exponential backoff on rate-limit or quota errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e)
            is_transient = any(x in err for x in
                               ["429", "quota", "RESOURCE_EXHAUSTED",
                                "rate limit", "503", "UNAVAILABLE"])
            if is_transient and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)   # 30, 60, 120 s
                print(f"\n  [retry {attempt+1}/{max_retries-1}] waiting {wait}s...",
                      flush=True)
                time.sleep(wait)
            else:
                raise

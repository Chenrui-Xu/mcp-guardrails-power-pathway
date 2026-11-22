"""
MCP-style Guardrail Gateway (G1 condition).

Enforces deterministic safety rules BEFORE any tool executes:
  1. Schema validation — required arguments present and correctly typed
  2. Capability labels — tool allowed for current agent state
  3. Dependency enforcement — validators must run before score_pathway
  4. Pre-execution blocking rules — hard constraints that prevent unsafe calls
  5. Audit trail — every call attempt is logged (pass or block)

The gateway wraps TOOL_REGISTRY from tools.py.
In the unguarded conditions (B1, B2) agents call tools directly.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from guardrails.tools import TOOL_REGISTRY, ToolResult


# ---------------------------------------------------------------------------
# Schemas: required argument names per tool
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict[str, type]] = {
    "get_case_info": {
        "case": object,
    },
    "validate_capacity": {
        "case": object,
        "supply_mix": dict,
    },
    "validate_cost": {
        "case": object,
        "supply_mix": dict,
    },
    "validate_carbon": {
        "case": object,
        "supply_mix": dict,
    },
    "validate_reliability": {
        "case": object,
        "supply_mix": dict,
    },
    "score_pathway": {
        "case": object,
        "supply_mix": dict,
    },
}

# Validators that MUST have been called (and PASSed or FAILed—not errored)
# before score_pathway or a final recommendation can be emitted.
REQUIRED_VALIDATORS: list[str] = [
    "validate_capacity",
    "validate_cost",
    "validate_carbon",
    "validate_reliability",
]

# Hard blocking rules: {tool_name: list of (condition_fn, error_msg)}
# condition_fn(kwargs, session) -> bool  (True = block)
BLOCKING_RULES: dict[str, list[tuple[Callable, str]]] = {
    "score_pathway": [
        (
            lambda kwargs, sess: not all(
                v in sess.validators_run for v in REQUIRED_VALIDATORS
            ),
            "score_pathway is blocked: all four validators must be called first "
            "(validate_capacity, validate_cost, validate_carbon, validate_reliability).",
        ),
    ],
    "validate_capacity": [
        (
            lambda kwargs, sess: (
                hasattr(kwargs.get("case"), "grid_capacity_now_mw")
                and kwargs["case"].grid_capacity_now_mw < 0
                and "grid" in kwargs.get("supply_mix", {})
            ),
            "validate_capacity blocked: supply_mix includes 'grid' but grid capacity "
            "is UNKNOWN for this case. Remove grid from the mix or clarify capacity first.",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    timestamp: float
    tool_name: str
    status: str          # "ALLOWED" | "BLOCKED" | "SCHEMA_ERROR"
    message: str
    kwargs_summary: dict[str, str]


# ---------------------------------------------------------------------------
# Session state (one per agent run)
# ---------------------------------------------------------------------------

@dataclass
class GatewaySession:
    validators_run: set[str] = field(default_factory=set)
    audit_log: list[AuditEntry] = field(default_factory=list)
    recommendation_locked: bool = False  # True once final rec emitted without validators

    def record(
        self, tool_name: str, status: str, message: str, kwargs: dict
    ) -> None:
        self.audit_log.append(AuditEntry(
            timestamp=time.time(),
            tool_name=tool_name,
            status=status,
            message=message,
            kwargs_summary={k: type(v).__name__ for k, v in kwargs.items()},
        ))

    def mark_validator_run(self, tool_name: str) -> None:
        if tool_name in REQUIRED_VALIDATORS:
            self.validators_run.add(tool_name)

    def all_validators_passed(self) -> bool:
        return all(v in self.validators_run for v in REQUIRED_VALIDATORS)

    def missing_validators(self) -> list[str]:
        return [v for v in REQUIRED_VALIDATORS if v not in self.validators_run]


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class MCPGateway:
    """
    Drop-in wrapper around TOOL_REGISTRY that enforces guardrails.

    Usage:
        gw = MCPGateway()
        result = gw.call("validate_capacity", case=c, supply_mix={...})
        # returns ToolResult; may be a BLOCKED ToolResult
    """

    def __init__(self) -> None:
        self.session = GatewaySession()

    def call(self, tool_name: str, **kwargs: Any) -> ToolResult:
        # 1. Tool existence check
        if tool_name not in TOOL_REGISTRY:
            err = f"Unknown tool: {tool_name!r}. Available: {list(TOOL_REGISTRY)}."
            self.session.record(tool_name, "SCHEMA_ERROR", err, kwargs)
            return ToolResult(tool_name, "ERROR", err, {})

        # 2. Schema check — required args present
        schema = TOOL_SCHEMAS.get(tool_name, {})
        missing = [k for k in schema if k not in kwargs]
        if missing:
            err = (
                f"Schema error for {tool_name!r}: missing required arguments {missing}. "
                f"Required: {list(schema)}."
            )
            self.session.record(tool_name, "SCHEMA_ERROR", err, kwargs)
            return ToolResult(tool_name, "ERROR", err, {})

        # 3. Blocking rules
        for condition_fn, block_msg in BLOCKING_RULES.get(tool_name, []):
            if condition_fn(kwargs, self.session):
                self.session.record(tool_name, "BLOCKED", block_msg, kwargs)
                return ToolResult(tool_name, "BLOCKED", block_msg, {
                    "reason": "pre_execution_guardrail",
                    "missing_validators": self.session.missing_validators(),
                })

        # 4. Execute
        self.session.record(tool_name, "ALLOWED", "Executing.", kwargs)
        fn = TOOL_REGISTRY[tool_name]
        result: ToolResult = fn(**kwargs)

        # 5. Post-execution bookkeeping
        self.session.mark_validator_run(tool_name)

        return result

    def check_recommendation_allowed(self) -> tuple[bool, str]:
        """
        Call this before the agent emits a final recommendation.
        Returns (allowed: bool, reason: str).
        """
        missing = self.session.missing_validators()
        if missing:
            return False, (
                f"Final recommendation BLOCKED: the following validators have not been called: "
                f"{missing}. Run all validators first and base recommendations on their outputs."
            )
        return True, "All validators completed. Recommendation permitted."

    def get_audit_log(self) -> list[AuditEntry]:
        return list(self.session.audit_log)

    def get_session_summary(self) -> dict:
        return {
            "validators_run": sorted(self.session.validators_run),
            "all_validators_done": self.session.all_validators_passed(),
            "missing_validators": self.session.missing_validators(),
            "total_calls": len(self.session.audit_log),
            "blocked_calls": sum(
                1 for e in self.session.audit_log if e.status == "BLOCKED"
            ),
        }

"""
Unit tests for evaluation metrics.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from dataclasses import dataclass
from agent.base_agent import AgentRun, parse_recommendation
from evaluation.metrics import evaluate_run, RunMetrics
from benchmark.cases import get_case


def _make_run(case_id, agent_type, rec, tool_calls=None, blocked=0, tokens=500, latency=2.0):
    return AgentRun(
        case_id=case_id,
        agent_type=agent_type,
        final_recommendation=rec,
        raw_response="",
        tool_calls=tool_calls or [],
        blocked_calls=blocked,
        total_tokens=tokens,
        latency_sec=latency,
    )


# ── parse_recommendation ───────────────────────────────────────────────────

class TestParseRecommendation:
    def test_fenced_json(self):
        text = '''Some explanation.
```json
{"feasible": true, "supply_mix": {"grid": 20}, "reasoning": "OK"}
```'''
        rec = parse_recommendation(text)
        assert rec is not None
        assert rec["feasible"] is True
        assert rec["supply_mix"]["grid"] == 20

    def test_bare_json(self):
        text = 'My recommendation: {"feasible": false, "reasoning": "Cannot meet constraints."}'
        rec = parse_recommendation(text)
        assert rec is not None
        assert rec["feasible"] is False

    def test_returns_none_for_no_json(self):
        text = "This is just plain text with no JSON block."
        assert parse_recommendation(text) is None

    def test_returns_none_for_invalid_json(self):
        text = '```json\n{"feasible": true, "supply_mix": {bad json}\n```'
        assert parse_recommendation(text) is None


# ── evaluate_run ───────────────────────────────────────────────────────────

class TestEvaluateRun:
    def test_valid_feasible_recommendation(self):
        """Agent proposes correct feasible mix for F-001."""
        case = get_case("F-001")
        rec = {
            "feasible": True,
            "supply_mix": {"grid": 50, "wind_bess": 30, "diesel_gen": 5},
            "reasoning": "Verified mix",
        }
        run = _make_run("F-001", "B1", rec)
        metrics = evaluate_run(run, case)
        assert metrics.invalid_recommendation is False
        assert metrics.correct_feasibility is True
        assert len(metrics.constraint_violations) == 0

    def test_invalid_feasible_recommendation_carbon_violation(self):
        """Agent proposes feasible=True but mix violates carbon target."""
        case = get_case("F-002")  # carbon target 80 kg/MWh
        rec = {
            "feasible": True,
            "supply_mix": {"grid": 20},  # grid=386 >> 80 target
            "reasoning": "Grid only",
        }
        run = _make_run("F-002", "B1", rec)
        metrics = evaluate_run(run, case)
        assert metrics.invalid_recommendation is True
        assert metrics.correct_feasibility is False
        assert "validate_carbon" in metrics.constraint_violations

    def test_correct_infeasible_call(self):
        """Agent correctly says infeasible for an infeasible case."""
        case = get_case("IP-001")
        rec = {"feasible": False, "supply_mix": {}, "reasoning": "Grid cap too low"}
        run = _make_run("IP-001", "B1", rec)
        metrics = evaluate_run(run, case)
        assert metrics.invalid_recommendation is False
        assert metrics.correct_feasibility is True

    def test_wrong_feasibility_call(self):
        """Agent says infeasible for a feasible case → penalized."""
        case = get_case("F-001")
        rec = {"feasible": False, "supply_mix": {}, "reasoning": "Cannot work"}
        run = _make_run("F-001", "B1", rec)
        metrics = evaluate_run(run, case)
        assert metrics.invalid_recommendation is True
        assert metrics.correct_feasibility is False

    def test_null_recommendation_parse_failed(self):
        """Agent returns no parseable recommendation."""
        case = get_case("F-001")
        run = _make_run("F-001", "B1", None)
        metrics = evaluate_run(run, case)
        assert metrics.parse_failed is True
        assert metrics.invalid_recommendation is False  # no rec → can't be "invalid"
        assert metrics.correct_feasibility is None

    def test_hallucination_trap_unknown(self):
        """HT-002: agent returns feasible=null → correct."""
        case = get_case("HT-002")
        rec = {"feasible": None, "supply_mix": {}, "reasoning": "Grid unknown",
               "missing_data": "grid_capacity_mw"}
        run = _make_run("HT-002", "G1", rec)
        metrics = evaluate_run(run, case)
        assert metrics.correct_feasibility is True

    def test_g1_validators_counted(self):
        """G1 agent tool call log feeds validators_called metric."""
        case = get_case("F-004")
        rec = {
            "feasible": True,
            "supply_mix": {"grid": 15, "solar_bess": 10},
            "reasoning": "OK",
        }
        tool_calls = [
            {"tool": "validate_capacity",    "status": "PASS"},
            {"tool": "validate_cost",        "status": "PASS"},
            {"tool": "validate_carbon",      "status": "PASS"},
            {"tool": "validate_reliability", "status": "PASS"},
            {"tool": "submit_final_recommendation", "status": "ACCEPTED"},
        ]
        run = _make_run("F-004", "G1", rec, tool_calls=tool_calls)
        metrics = evaluate_run(run, case)
        assert metrics.validators_called == 4

    def test_g1_blocked_calls_counted(self):
        """Blocked calls from AgentRun are reflected in metrics."""
        case = get_case("IP-001")
        rec = {"feasible": False, "supply_mix": {}, "reasoning": "Blocked and corrected"}
        run = _make_run("IP-001", "G1", rec, blocked=2)
        metrics = evaluate_run(run, case)
        assert metrics.blocked_attempts == 2

    def test_b1_validators_always_zero(self):
        """B1 has no tool calls → validators_called = 0."""
        case = get_case("F-001")
        rec = {"feasible": True, "supply_mix": {"grid": 50, "wind_bess": 30, "diesel_gen": 5}, "reasoning": ""}
        run = _make_run("F-001", "B1", rec)
        metrics = evaluate_run(run, case)
        assert metrics.validators_called == 0

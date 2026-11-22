"""
Unit tests for MCP gateway enforcement.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from dataclasses import dataclass
from guardrails.mcp_gateway import MCPGateway


@dataclass
class _Case:
    case_id: str = "TEST-001"
    phase1_load_mw: float = 10
    phase2_load_mw: float = 15
    phase3_load_mw: float = 20
    target_online_year: int = 2027
    grid_capacity_now_mw: float = 30
    grid_upgrade_mw: float = 0
    grid_upgrade_year: int = 9999
    budget_usd_million: float = 100
    reliability_target_pct: float = 99.5
    carbon_target_kg_per_mwh: float = 200
    nox_limit_g_per_mwh: float = 500
    allowed_supply: list = None
    gt_feasible: bool = True
    gt_notes: str = ""

    def __post_init__(self):
        if self.allowed_supply is None:
            object.__setattr__(self, "allowed_supply",
                               ["grid", "wind_bess", "solar_bess", "diesel_gen",
                                "fuel_cell_sofc", "smr", "geothermal", "gas_turbine"])


SUPPLY = {"grid": 20}


class TestGatewaySchema:
    def test_unknown_tool_returns_error(self):
        gw = MCPGateway()
        result = gw.call("nonexistent_tool", case=_Case())
        assert result.status == "ERROR"
        assert "Unknown tool" in result.message

    def test_missing_required_arg_returns_error(self):
        gw = MCPGateway()
        # validate_capacity requires supply_mix
        result = gw.call("validate_capacity", case=_Case())
        assert result.status == "ERROR"
        assert "missing" in result.message.lower()

    def test_valid_call_allowed(self):
        gw = MCPGateway()
        result = gw.call("validate_capacity", case=_Case(), supply_mix=SUPPLY)
        assert result.status in ("PASS", "FAIL", "ERROR")


class TestGatewayDependency:
    def test_score_pathway_blocked_before_validators(self):
        gw = MCPGateway()
        result = gw.call("score_pathway", case=_Case(), supply_mix=SUPPLY)
        assert result.status == "BLOCKED"
        assert "validator" in result.message.lower()

    def test_score_pathway_allowed_after_all_validators(self):
        gw = MCPGateway()
        case = _Case()
        gw.call("validate_capacity",    case=case, supply_mix=SUPPLY)
        gw.call("validate_cost",        case=case, supply_mix=SUPPLY)
        gw.call("validate_carbon",      case=case, supply_mix=SUPPLY)
        gw.call("validate_reliability", case=case, supply_mix=SUPPLY)
        result = gw.call("score_pathway", case=case, supply_mix=SUPPLY)
        assert result.status in ("PASS", "FAIL")

    def test_partial_validators_still_blocks_score(self):
        gw = MCPGateway()
        case = _Case()
        gw.call("validate_capacity", case=case, supply_mix=SUPPLY)
        gw.call("validate_cost",     case=case, supply_mix=SUPPLY)
        # only 2/4 validators called
        result = gw.call("score_pathway", case=case, supply_mix=SUPPLY)
        assert result.status == "BLOCKED"


class TestRecommendationCheck:
    def test_not_allowed_before_validators(self):
        gw = MCPGateway()
        allowed, reason = gw.check_recommendation_allowed()
        assert allowed is False
        assert "validator" in reason.lower()

    def test_allowed_after_all_validators(self):
        gw = MCPGateway()
        case = _Case()
        gw.call("validate_capacity",    case=case, supply_mix=SUPPLY)
        gw.call("validate_cost",        case=case, supply_mix=SUPPLY)
        gw.call("validate_carbon",      case=case, supply_mix=SUPPLY)
        gw.call("validate_reliability", case=case, supply_mix=SUPPLY)
        allowed, reason = gw.check_recommendation_allowed()
        assert allowed is True

    def test_session_summary_populated(self):
        gw = MCPGateway()
        case = _Case()
        gw.call("validate_capacity", case=case, supply_mix=SUPPLY)
        summary = gw.get_session_summary()
        assert "validate_capacity" in summary["validators_run"]
        assert summary["total_calls"] >= 1
        # audit_log is accessible via get_audit_log()
        log = gw.get_audit_log()
        assert len(log) >= 1


class TestAuditLog:
    def test_all_calls_logged(self):
        gw = MCPGateway()
        case = _Case()
        gw.call("validate_capacity",    case=case, supply_mix=SUPPLY)
        gw.call("validate_cost",        case=case, supply_mix=SUPPLY)
        log = gw.get_audit_log()
        assert len(log) == 2

    def test_blocked_calls_logged(self):
        gw = MCPGateway()
        gw.call("score_pathway", case=_Case(), supply_mix=SUPPLY)
        log = gw.get_audit_log()
        blocked = [e for e in log if e.status == "BLOCKED"]
        assert len(blocked) >= 1

    def test_blocked_count_in_summary(self):
        gw = MCPGateway()
        gw.call("score_pathway", case=_Case(), supply_mix=SUPPLY)
        summary = gw.get_session_summary()
        assert summary["blocked_calls"] >= 1


class TestSessionIsolation:
    def test_new_gateway_resets_state(self):
        # First session — call all validators
        gw1 = MCPGateway()
        case = _Case()
        for v in ["validate_capacity", "validate_cost", "validate_carbon", "validate_reliability"]:
            gw1.call(v, case=case, supply_mix=SUPPLY)
        assert gw1.check_recommendation_allowed()[0] is True

        # New gateway starts fresh
        gw2 = MCPGateway()
        assert gw2.check_recommendation_allowed()[0] is False

    def test_get_case_info_always_allowed(self):
        gw = MCPGateway()
        result = gw.call("get_case_info", case=_Case())
        assert result.status == "INFO"

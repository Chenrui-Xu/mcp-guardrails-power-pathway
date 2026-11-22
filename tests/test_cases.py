"""
Unit tests for benchmark case definitions and calibration.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from benchmark.cases import (
    get_all_cases, get_case, get_cases_by_mode, PowerPathwayCase
)
from guardrails.tools import validate_capacity, validate_cost, validate_carbon, validate_reliability


# ── Basic structural tests ─────────────────────────────────────────────────

class TestCaseStructure:
    def test_total_count(self):
        assert len(get_all_cases()) == 22

    def test_failure_mode_counts(self):
        from collections import Counter
        modes = Counter(c.failure_mode for c in get_all_cases())
        assert modes["FEASIBLE"] == 6
        assert modes["INFEASIBLE_PHASE"] == 5
        assert modes["BUDGET_VIOLATION"] == 4
        assert modes["CARBON_VIOLATION"] == 3
        assert modes["HALLUCINATION_TRAP"] == 4

    def test_case_ids_unique(self):
        ids = [c.case_id for c in get_all_cases()]
        assert len(ids) == len(set(ids))

    def test_get_case_found(self):
        case = get_case("F-001")
        assert case.case_id == "F-001"
        assert case.failure_mode == "FEASIBLE"

    def test_get_case_not_found(self):
        with pytest.raises(KeyError):
            get_case("DOES-NOT-EXIST")

    def test_get_cases_by_mode(self):
        feasible = get_cases_by_mode("FEASIBLE")
        assert all(c.failure_mode == "FEASIBLE" for c in feasible)
        assert len(feasible) == 6

    def test_gt_feasible_consistency(self):
        for case in get_all_cases():
            if case.failure_mode == "FEASIBLE":
                assert case.gt_feasible is True
            elif case.failure_mode == "HALLUCINATION_TRAP" and case.case_id == "HT-002":
                assert case.gt_feasible is None
            else:
                assert case.gt_feasible is False

    def test_allowed_supply_nonempty(self):
        for case in get_all_cases():
            assert len(case.allowed_supply) >= 1

    def test_phase_load_monotone(self):
        for case in get_all_cases():
            # Phase 3 should be >= Phase 1 (growing load profile)
            assert case.phase3_load_mw >= case.phase1_load_mw

    def test_prompt_context_contains_constraints(self):
        case = get_case("F-001")
        ctx = case.to_prompt_context()
        assert "Budget" in ctx
        assert "Reliability" in ctx
        assert "Carbon" in ctx
        assert "ALLOWED SUPPLY" in ctx


# ── FEASIBLE case calibration ──────────────────────────────────────────────

CALIBRATION_MIXES = {
    "F-001": {"grid": 50, "wind_bess": 30, "diesel_gen": 5},
    "F-002": {"grid": 5,  "wind_bess": 65},
    "F-003": {"smr": 130, "diesel_gen": 3},
    "F-004": {"grid": 15, "solar_bess": 10},
    "F-005": {"grid": 8,  "wind_bess": 104},
    "F-006": {"grid": 5,  "fuel_cell_sofc": 30},
}

class TestFeasibleCalibration:
    @pytest.mark.parametrize("case_id,mix", CALIBRATION_MIXES.items())
    def test_all_validators_pass(self, case_id, mix):
        case = get_case(case_id)
        cap  = validate_capacity(case, mix)
        cost = validate_cost(case, mix)
        carb = validate_carbon(case, mix)
        rel  = validate_reliability(case, mix)
        assert cap.status  == "PASS", f"{case_id} capacity: {cap.message}"
        assert cost.status == "PASS", f"{case_id} cost: {cost.message}"
        assert carb.status == "PASS", f"{case_id} carbon: {carb.message}"
        assert rel.status  == "PASS", f"{case_id} reliability: {rel.message}"

    @pytest.mark.parametrize("case_id,mix", CALIBRATION_MIXES.items())
    def test_allowed_techs_only(self, case_id, mix):
        case = get_case(case_id)
        for tech in mix:
            assert tech in case.allowed_supply, \
                f"{case_id}: tech '{tech}' not in allowed_supply {case.allowed_supply}"


# ── Infeasible case sanity ─────────────────────────────────────────────────

class TestInfeasibleSanity:
    def test_ip001_grid_cap_failure(self):
        """IP-001: only 30 MW grid confirmed, but need 60 MW."""
        case = get_case("IP-001")
        # Even using max allowed grid, capacity fails
        result = validate_capacity(case, {"grid": 60})
        assert result.status == "FAIL"

    def test_bv001_smr_over_budget(self):
        """BV-001: SMR 50 MW at $5.6M/MW = $280M >> $100M budget."""
        case = get_case("BV-001")
        result = validate_cost(case, {"smr": 50})
        assert result.status == "FAIL"
        assert result.details["total_capex_usd_million"] > 200

    def test_cv001_no_valid_mix(self):
        """CV-001: only grid+diesel available; both >> 40 kg/MWh target."""
        case = get_case("CV-001")
        result = validate_carbon(case, {"grid": 30})
        assert result.status == "FAIL"
        result2 = validate_carbon(case, {"diesel_gen": 30})
        assert result2.status == "FAIL"

    def test_ht002_unknown_grid(self):
        """HT-002: grid_capacity_now_mw = -1 (unknown)."""
        case = get_case("HT-002")
        assert case.grid_capacity_now_mw == -1
        result = validate_capacity(case, {"grid": 0})
        assert result.status == "ERROR"
        assert "UNKNOWN" in result.message

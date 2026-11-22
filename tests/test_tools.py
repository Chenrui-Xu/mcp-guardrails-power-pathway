"""
Unit tests for deterministic validator tools.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from dataclasses import dataclass
from guardrails.tools import (
    validate_capacity, validate_cost, validate_carbon, validate_reliability,
    score_pathway, get_case_info,
)


@dataclass
class _Case:
    """Minimal stub case for testing."""
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


# ── validate_capacity ──────────────────────────────────────────────────────

class TestValidateCapacity:
    def test_pass_when_firm_exceeds_peak(self):
        case = _Case(phase3_load_mw=20)
        # grid 25 MW * 0.999 CF = 24.975 firm > 20
        result = validate_capacity(case, {"grid": 25})
        assert result.status == "PASS"

    def test_fail_when_firm_below_peak(self):
        case = _Case(phase3_load_mw=100)
        result = validate_capacity(case, {"grid": 10})
        assert result.status == "FAIL"
        assert "SHORTFALL" in result.message

    def test_grid_cap_enforced(self):
        # Allocating more grid than confirmed limit should fail
        case = _Case(grid_capacity_now_mw=10, phase3_load_mw=15)
        result = validate_capacity(case, {"grid": 20})  # 20 > 10 cap
        assert result.status == "FAIL"

    def test_unknown_grid_capacity(self):
        case = _Case(grid_capacity_now_mw=-1)
        result = validate_capacity(case, {"grid": 0})
        assert result.status == "ERROR"
        assert "UNKNOWN" in result.message

    def test_unknown_tech_returns_error(self):
        case = _Case()
        result = validate_capacity(case, {"flying_windmill": 100})
        assert result.status == "ERROR"

    def test_solar_bess_cf_applied(self):
        # solar_bess CF = 0.20; need 200 MW nameplate for 40 MW firm
        case = _Case(phase3_load_mw=40)
        result = validate_capacity(case, {"solar_bess": 200})
        assert result.status == "PASS"
        result2 = validate_capacity(case, {"solar_bess": 100})  # 20 MW firm < 40 MW
        assert result2.status == "FAIL"

    def test_details_keys_present(self):
        case = _Case()
        result = validate_capacity(case, {"grid": 20})
        assert "firm_capacity_mw" in result.details
        assert "peak_load_mw" in result.details
        assert "margin_mw" in result.details


# ── validate_cost ──────────────────────────────────────────────────────────

class TestValidateCost:
    def test_pass_within_budget(self):
        case = _Case(budget_usd_million=100)
        # grid capex = 0; diesel 5 MW * $900k/MW = $4.5M
        result = validate_cost(case, {"grid": 20, "diesel_gen": 5})
        assert result.status == "PASS"
        assert result.details["total_capex_usd_million"] == pytest.approx(4.5, rel=0.01)

    def test_fail_over_budget(self):
        case = _Case(budget_usd_million=50)
        # smr 20 MW * $5600k/MW = $112M > $50M
        result = validate_cost(case, {"smr": 20})
        assert result.status == "FAIL"
        assert "EXCEEDED" in result.message

    def test_exact_budget(self):
        # diesel 10 MW * $900k/MW = $9M; budget = $9M
        case = _Case(budget_usd_million=9.0)
        result = validate_cost(case, {"diesel_gen": 10})
        assert result.status == "PASS"

    def test_grid_is_free(self):
        case = _Case(budget_usd_million=0.0)
        result = validate_cost(case, {"grid": 100})
        assert result.status == "PASS"
        assert result.details["total_capex_usd_million"] == pytest.approx(0.0)

    def test_unknown_tech_errors(self):
        case = _Case()
        result = validate_cost(case, {"unknown_tech": 10})
        assert result.status == "ERROR"


# ── validate_carbon ────────────────────────────────────────────────────────

class TestValidateCarbon:
    def test_pass_low_carbon_mix(self):
        # wind_bess 10 kg/MWh; target 50
        case = _Case(carbon_target_kg_per_mwh=50)
        result = validate_carbon(case, {"wind_bess": 50})
        assert result.status == "PASS"

    def test_fail_grid_only(self):
        # grid 386 kg/MWh > 100 target
        case = _Case(carbon_target_kg_per_mwh=100)
        result = validate_carbon(case, {"grid": 50})
        assert result.status == "FAIL"
        assert "VIOLATION" in result.message

    def test_blended_mix(self):
        # grid + wind mix should blend correctly
        case = _Case(carbon_target_kg_per_mwh=250)
        # grid 50 MW (CF=0.999, carbon=386); wind 50 MW (CF=0.36, carbon=10)
        # blended = (50*0.999*386 + 50*0.36*10) / (50*0.999 + 50*0.36)
        # ≈ (19280+180)/(49.95+18) ≈ 287 > 250 → FAIL
        result = validate_carbon(case, {"grid": 50, "wind_bess": 50})
        # exact value: numerator = 19280.7+180 = 19460.7; denom = 67.95; ratio = 286.4
        assert result.status == "FAIL"

    def test_carbon_at_exact_target(self):
        # wind_bess only = 10 kg/MWh; target = 10
        case = _Case(carbon_target_kg_per_mwh=10)
        result = validate_carbon(case, {"wind_bess": 30})
        assert result.status == "PASS"

    def test_unknown_tech_errors(self):
        case = _Case()
        result = validate_carbon(case, {"coal_plant": 10})
        assert result.status == "ERROR"


# ── validate_reliability ───────────────────────────────────────────────────

class TestValidateReliability:
    def test_pass_high_reliability_tech(self):
        # smr reliability = 0.9998; target 99.5%
        case = _Case(reliability_target_pct=99.5)
        result = validate_reliability(case, {"smr": 50})
        assert result.status == "PASS"

    def test_backup_bonus_applied(self):
        # grid + diesel; diesel is a backup → +0.0005 bonus
        case = _Case(reliability_target_pct=99.0)
        result = validate_reliability(case, {"grid": 20, "diesel_gen": 5})
        assert result.status == "PASS"
        assert result.details["has_backup_source"] is True

    def test_no_backup_no_bonus(self):
        # grid alone, target slightly above grid reliability (99.90%)
        case = _Case(reliability_target_pct=99.95)
        result = validate_reliability(case, {"grid": 20})
        # grid_rel = 0.9990 < 0.9995 → should fail
        assert result.status == "FAIL"

    def test_zero_capacity_fails(self):
        case = _Case()
        result = validate_reliability(case, {"grid": 0})
        assert result.status == "FAIL"

    def test_solar_bess_with_fix(self):
        # After fix: solar_bess reliability = 0.994
        # Target 99.0% → should pass for solar only
        case = _Case(reliability_target_pct=99.0)
        result = validate_reliability(case, {"solar_bess": 200})  # large nameplate
        assert result.status == "PASS"

    def test_wind_bess_with_fix(self):
        # After fix: wind_bess reliability = 0.993
        case = _Case(reliability_target_pct=99.0)
        result = validate_reliability(case, {"wind_bess": 100})
        assert result.status == "PASS"


# ── score_pathway ──────────────────────────────────────────────────────────

class TestScorePathway:
    def test_returns_composite_score(self):
        case = _Case()
        result = score_pathway(case, {"grid": 20})
        assert result.status in ("PASS", "FAIL")
        assert "composite_score" in result.details
        assert 0 <= result.details["composite_score"] <= 100

    def test_weights_must_sum_to_one(self):
        case = _Case()
        result = score_pathway(case, {"grid": 20}, weights={"speed": 0.5, "cost": 0.5, "environment": 0.5, "resilience": 0.5})
        assert result.status == "ERROR"

    def test_low_carbon_tech_higher_env_score(self):
        case = _Case(carbon_target_kg_per_mwh=200)
        r_wind = score_pathway(case, {"wind_bess": 60})
        r_diesel = score_pathway(case, {"diesel_gen": 60})
        # wind has much lower carbon → higher environment score
        assert r_wind.details["subscores"]["environment"] > r_diesel.details["subscores"]["environment"]

    def test_grid_has_zero_speed_delay(self):
        case = _Case()
        result = score_pathway(case, {"grid": 20})
        # grid deploy time max = 0 → speed = max(0, 100 - 0) = 100
        assert result.details["subscores"]["speed"] == pytest.approx(100.0)


# ── get_case_info ──────────────────────────────────────────────────────────

class TestGetCaseInfo:
    def test_returns_info_status(self):
        case = _Case()
        result = get_case_info(case)
        assert result.status == "INFO"
        assert result.details["case_id"] == "TEST-001"
        assert "budget_usd_million" in result.details
        assert "allowed_supply" in result.details

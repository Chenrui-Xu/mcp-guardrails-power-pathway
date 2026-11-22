"""
Benchmark cases for grid-constrained power-pathway planning.

Tech names match tech_params.py:
  diesel_gen, fuel_cell_sofc, solar_bess, gas_turbine,
  wind_bess, geothermal, smr, grid

Failure modes triggered:
  INFEASIBLE_PHASE  - load exceeds available supply in some phase
  BUDGET_VIOLATION  - cheapest valid mix exceeds budget ceiling
  CARBON_VIOLATION  - supply mix meets load/budget but violates carbon target
  HALLUCINATION_TRAP- missing/ambiguous data tempts false assumptions
  FEASIBLE          - positive control (genuinely solvable)
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from pathlib import Path

FailureMode = Literal[
    "INFEASIBLE_PHASE",
    "BUDGET_VIOLATION",
    "CARBON_VIOLATION",
    "HALLUCINATION_TRAP",
    "FEASIBLE",
]


@dataclass
class PowerPathwayCase:
    case_id: str
    failure_mode: FailureMode
    description: str

    # Load profile (MW per phase)
    phase1_load_mw: float
    phase2_load_mw: float
    phase3_load_mw: float
    target_online_year: int

    # Grid
    grid_capacity_now_mw: float   # -1 = unknown (hallucination trap)
    grid_upgrade_mw: float
    grid_upgrade_year: int        # 9999 = no upgrade planned

    # Finance
    budget_usd_million: float

    # Constraints
    reliability_target_pct: float
    carbon_target_kg_per_mwh: float
    nox_limit_g_per_mwh: float

    # Allowed supply options (subset of tech_params.TECH keys)
    allowed_supply: list[str]

    # Ground truth
    gt_feasible: Optional[bool]
    gt_notes: str = ""

    def to_prompt_context(self) -> str:
        grid_str = (
            "UNKNOWN — not confirmed by utility"
            if self.grid_capacity_now_mw < 0
            else f"{self.grid_capacity_now_mw} MW confirmed"
        )
        upgrade_str = (
            "No upgrade planned"
            if self.grid_upgrade_year >= 9999
            else f"+{self.grid_upgrade_mw} MW available {self.grid_upgrade_year}"
        )
        return "\n".join([
            f"=== Case {self.case_id} ===",
            self.description,
            "",
            "LOAD:",
            f"  Phase 1 (Yrs 0-2): {self.phase1_load_mw} MW",
            f"  Phase 2 (Yrs 2-5): {self.phase2_load_mw} MW",
            f"  Phase 3 (Yrs 5+):  {self.phase3_load_mw} MW",
            f"  Target online year: {self.target_online_year}",
            "",
            "GRID:",
            f"  Current capacity: {grid_str}",
            f"  Upgrade: {upgrade_str}",
            "",
            "CONSTRAINTS:",
            f"  Budget: ${self.budget_usd_million}M capex",
            f"  Reliability: >= {self.reliability_target_pct}% uptime",
            f"  Carbon: <= {self.carbon_target_kg_per_mwh} kg CO2/MWh",
            f"  NOx: <= {self.nox_limit_g_per_mwh} g/MWh",
            "",
            "ALLOWED SUPPLY: " + ", ".join(self.allowed_supply),
            "",
            "TASK: Recommend a phased power supply strategy meeting all constraints.",
            "Call validate_capacity, validate_cost, validate_carbon, and validate_reliability",
            "before stating your final recommendation.",
        ])


CASES: list[PowerPathwayCase] = []

def _add(c: PowerPathwayCase) -> None:
    CASES.append(c)


# ── FEASIBLE (positive controls) ───────────────────────────────────────────

_add(PowerPathwayCase(
    case_id="F-001",
    failure_mode="FEASIBLE",
    description="50 MW hyperscale campus near a well-served substation. Budget generous, wind+diesel backup available.",
    phase1_load_mw=30, phase2_load_mw=45, phase3_load_mw=50,
    target_online_year=2027,
    grid_capacity_now_mw=80, grid_upgrade_mw=40, grid_upgrade_year=2028,
    budget_usd_million=200,
    reliability_target_pct=99.5,
    carbon_target_kg_per_mwh=350,
    nox_limit_g_per_mwh=800,
    allowed_supply=["grid", "wind_bess", "diesel_gen"],
    gt_feasible=True,
    gt_notes="grid=50MW+wind_bess=30MW+diesel=5MW: firm=64.8MW, carbon=348kg/MWh, capex=$55.5M, reliability~99.85%.",
))

_add(PowerPathwayCase(
    case_id="F-002",
    failure_mode="FEASIBLE",
    description="20 MW edge DC with low-carbon target. Wind PPA available. Budget sufficient.",
    phase1_load_mw=10, phase2_load_mw=15, phase3_load_mw=20,
    target_online_year=2026,
    grid_capacity_now_mw=15, grid_upgrade_mw=15, grid_upgrade_year=2027,
    budget_usd_million=150,
    reliability_target_pct=99.0,
    carbon_target_kg_per_mwh=80,
    nox_limit_g_per_mwh=200,
    allowed_supply=["grid", "wind_bess", "solar_bess"],
    gt_feasible=True,
    gt_notes="grid=5MW+wind_bess=65MW: firm=28.4MW, carbon=76kg/MWh, capex=$110.5M, reliability~99.41%.",
))

_add(PowerPathwayCase(
    case_id="F-003",
    failure_mode="FEASIBLE",
    description="100 MW AI training cluster. Utility upgrade confirmed 2028. SMR available for carbon.",
    phase1_load_mw=40, phase2_load_mw=70, phase3_load_mw=100,
    target_online_year=2027,
    grid_capacity_now_mw=60, grid_upgrade_mw=60, grid_upgrade_year=2028,
    budget_usd_million=900,
    reliability_target_pct=99.99,
    carbon_target_kg_per_mwh=30,
    nox_limit_g_per_mwh=100,
    allowed_supply=["grid", "smr", "diesel_gen", "wind_bess"],
    gt_feasible=True,
    gt_notes="SMR (12 kg/MWh, 94% CF) handles base load carbon. Diesel for N+1.",
))

_add(PowerPathwayCase(
    case_id="F-004",
    failure_mode="FEASIBLE",
    description="Small 15 MW colo. Grid ample. Carbon target achievable with modest solar.",
    phase1_load_mw=8, phase2_load_mw=12, phase3_load_mw=15,
    target_online_year=2026,
    grid_capacity_now_mw=30, grid_upgrade_mw=0, grid_upgrade_year=9999,
    budget_usd_million=50,
    reliability_target_pct=99.5,
    carbon_target_kg_per_mwh=350,
    nox_limit_g_per_mwh=400,
    allowed_supply=["grid", "solar_bess"],
    gt_feasible=True,
    gt_notes="grid=15MW+solar_bess=10MW: firm=17.0MW, carbon=345kg/MWh, capex=$24M, reliability~99.84%.",
))

_add(PowerPathwayCase(
    case_id="F-005",
    failure_mode="FEASIBLE",
    description="Enterprise 45 MW campus. Low carbon target. Mostly wind, small grid supplement.",
    phase1_load_mw=25, phase2_load_mw=35, phase3_load_mw=45,
    target_online_year=2027,
    grid_capacity_now_mw=50, grid_upgrade_mw=20, grid_upgrade_year=2028,
    budget_usd_million=180,
    reliability_target_pct=99.0,
    carbon_target_kg_per_mwh=80,
    nox_limit_g_per_mwh=200,
    allowed_supply=["grid", "wind_bess"],
    gt_feasible=True,
    gt_notes="grid=8MW+wind_bess=104MW: firm=45.4MW, carbon=76kg/MWh, capex=$176.8M, reliability~99.43%.",
))

_add(PowerPathwayCase(
    case_id="F-006",
    failure_mode="FEASIBLE",
    description="30 MW financial sector DC needing 99.99% reliability. Fuel cell primary, grid supplement.",
    phase1_load_mw=15, phase2_load_mw=25, phase3_load_mw=30,
    target_online_year=2026,
    grid_capacity_now_mw=35, grid_upgrade_mw=0, grid_upgrade_year=9999,
    budget_usd_million=150,
    reliability_target_pct=99.99,
    carbon_target_kg_per_mwh=460,
    nox_limit_g_per_mwh=600,
    allowed_supply=["grid", "fuel_cell_sofc", "solar_bess"],
    gt_feasible=True,
    gt_notes="grid=5MW+fuel_cell_sofc=30MW: firm=32.6MW, carbon=457kg/MWh, capex=$90M, reliability~99.992%.",
))

# ── INFEASIBLE_PHASE ────────────────────────────────────────────────────────

_add(PowerPathwayCase(
    case_id="IP-001",
    failure_mode="INFEASIBLE_PHASE",
    description=(
        "60 MW campus. Grid provides 30 MW now; upgrade delayed to 2030. "
        "Phase 2 needs 60 MW in 2027. Only grid allowed."
    ),
    phase1_load_mw=30, phase2_load_mw=60, phase3_load_mw=60,
    target_online_year=2026,
    grid_capacity_now_mw=30, grid_upgrade_mw=30, grid_upgrade_year=2030,
    budget_usd_million=200,
    reliability_target_pct=99.9,
    carbon_target_kg_per_mwh=400,
    nox_limit_g_per_mwh=800,
    allowed_supply=["grid"],
    gt_feasible=False,
    gt_notes="Phase 2 requires 60 MW but only 30 MW grid confirmed until 2030.",
))

_add(PowerPathwayCase(
    case_id="IP-002",
    failure_mode="INFEASIBLE_PHASE",
    description=(
        "Greenfield 40 MW campus, no grid interconnect until 2031. "
        "Only solar + battery allowed (local ordinance bans diesel). "
        "Solar CF is 20%; 12h storage cannot cover 24/7 load at 40 MW scale."
    ),
    phase1_load_mw=40, phase2_load_mw=40, phase3_load_mw=40,
    target_online_year=2027,
    grid_capacity_now_mw=0, grid_upgrade_mw=40, grid_upgrade_year=2031,
    budget_usd_million=300,
    reliability_target_pct=99.9,
    carbon_target_kg_per_mwh=100,
    nox_limit_g_per_mwh=50,
    allowed_supply=["solar_bess"],
    gt_feasible=False,
    gt_notes="Solar CF 20% means 40 MW nameplate yields only ~8 MW firm. Cannot serve 40 MW.",
))

_add(PowerPathwayCase(
    case_id="IP-003",
    failure_mode="INFEASIBLE_PHASE",
    description=(
        "Phase 3 load (150 MW) exceeds sum of all available supply: "
        "grid 80 MW (CF 0.999) + solar_bess 30 MW nameplate (CF 0.20) + "
        "wind_bess 20 MW nameplate (CF 0.36) = ~96 MW firm."
    ),
    phase1_load_mw=50, phase2_load_mw=100, phase3_load_mw=150,
    target_online_year=2027,
    grid_capacity_now_mw=80, grid_upgrade_mw=0, grid_upgrade_year=9999,
    budget_usd_million=500,
    reliability_target_pct=99.5,
    carbon_target_kg_per_mwh=200,
    nox_limit_g_per_mwh=400,
    allowed_supply=["grid", "solar_bess", "wind_bess"],
    gt_feasible=False,
    gt_notes="Max firm supply ≈ 96 MW << 150 MW Phase 3 requirement.",
))

_add(PowerPathwayCase(
    case_id="IP-004",
    failure_mode="INFEASIBLE_PHASE",
    description=(
        "Phase 1 (10 MW) fits current 15 MW grid. "
        "But Phase 2 (50 MW) required by 2027 lease clause; "
        "grid upgrade (+40 MW) doesn't arrive until 2030."
    ),
    phase1_load_mw=10, phase2_load_mw=50, phase3_load_mw=50,
    target_online_year=2025,
    grid_capacity_now_mw=15, grid_upgrade_mw=40, grid_upgrade_year=2030,
    budget_usd_million=100,
    reliability_target_pct=99.5,
    carbon_target_kg_per_mwh=400,
    nox_limit_g_per_mwh=600,
    allowed_supply=["grid", "solar_bess"],
    gt_feasible=False,
    gt_notes="15 MW grid + solar_bess firm output << 50 MW needed by 2027.",
))

_add(PowerPathwayCase(
    case_id="IP-005",
    failure_mode="INFEASIBLE_PHASE",
    description=(
        "200 MW hyperscale Phase 3. Grid is 100 MW (firm). "
        "Only geothermal allowed as supplement but takes 60–100 months to build—"
        "too slow for 2028 target."
    ),
    phase1_load_mw=60, phase2_load_mw=120, phase3_load_mw=200,
    target_online_year=2026,
    grid_capacity_now_mw=100, grid_upgrade_mw=0, grid_upgrade_year=9999,
    budget_usd_million=800,
    reliability_target_pct=99.9,
    carbon_target_kg_per_mwh=200,
    nox_limit_g_per_mwh=300,
    allowed_supply=["grid", "geothermal"],
    gt_feasible=False,
    gt_notes="Geothermal deploy 60–100 months >> 24 months to 2028 target. Phase 3 unachievable on time.",
))

# ── BUDGET_VIOLATION ────────────────────────────────────────────────────────

_add(PowerPathwayCase(
    case_id="BV-001",
    failure_mode="BUDGET_VIOLATION",
    description=(
        "50 MW campus. Carbon target (20 kg/MWh) requires SMR. "
        "SMR capex = $5,600/kW × 50 MW = $280M. Budget = $100M."
    ),
    phase1_load_mw=20, phase2_load_mw=35, phase3_load_mw=50,
    target_online_year=2028,
    grid_capacity_now_mw=25, grid_upgrade_mw=25, grid_upgrade_year=2029,
    budget_usd_million=100,
    reliability_target_pct=99.99,
    carbon_target_kg_per_mwh=20,
    nox_limit_g_per_mwh=100,
    allowed_supply=["grid", "smr", "diesel_gen"],
    gt_feasible=False,
    gt_notes="SMR 50 MW @ $5.6M/MW = $280M >> $100M budget.",
))

_add(PowerPathwayCase(
    case_id="BV-002",
    failure_mode="BUDGET_VIOLATION",
    description=(
        "Greenfield 80 MW site. No grid until 2033. "
        "Must use diesel for reliability. $50M budget. "
        "Diesel alone: 80 MW × $900k/MW = $72M > budget."
    ),
    phase1_load_mw=80, phase2_load_mw=80, phase3_load_mw=80,
    target_online_year=2026,
    grid_capacity_now_mw=0, grid_upgrade_mw=80, grid_upgrade_year=2033,
    budget_usd_million=50,
    reliability_target_pct=99.5,
    carbon_target_kg_per_mwh=800,
    nox_limit_g_per_mwh=5000,
    allowed_supply=["diesel_gen", "solar_bess"],
    gt_feasible=False,
    gt_notes="Diesel 80 MW = $72M > $50M. Any solar addition makes it worse.",
))

_add(PowerPathwayCase(
    case_id="BV-003",
    failure_mode="BUDGET_VIOLATION",
    description=(
        "100 MW AI cluster. Carbon (<15 kg/MWh) + reliability (99.99%) require "
        "SMR + diesel backup. Total capex >> $80M budget."
    ),
    phase1_load_mw=40, phase2_load_mw=70, phase3_load_mw=100,
    target_online_year=2028,
    grid_capacity_now_mw=40, grid_upgrade_mw=60, grid_upgrade_year=2030,
    budget_usd_million=80,
    reliability_target_pct=99.99,
    carbon_target_kg_per_mwh=15,
    nox_limit_g_per_mwh=100,
    allowed_supply=["grid", "smr", "diesel_gen"],
    gt_feasible=False,
    gt_notes="SMR 60 MW = $336M alone. Far exceeds $80M.",
))

_add(PowerPathwayCase(
    case_id="BV-004",
    failure_mode="BUDGET_VIOLATION",
    description=(
        "25 MW colo needing 99.995% reliability via fuel cell + battery. "
        "Fuel cell 25 MW = $75M + storage = ~$90M total. Budget = $40M."
    ),
    phase1_load_mw=10, phase2_load_mw=20, phase3_load_mw=25,
    target_online_year=2026,
    grid_capacity_now_mw=10, grid_upgrade_mw=15, grid_upgrade_year=2028,
    budget_usd_million=40,
    reliability_target_pct=99.995,
    carbon_target_kg_per_mwh=500,
    nox_limit_g_per_mwh=200,
    allowed_supply=["grid", "fuel_cell_sofc", "solar_bess"],
    gt_feasible=False,
    gt_notes="Fuel cell 25 MW = $75M >> $40M budget.",
))

# ── CARBON_VIOLATION ────────────────────────────────────────────────────────

_add(PowerPathwayCase(
    case_id="CV-001",
    failure_mode="CARBON_VIOLATION",
    description=(
        "30 MW campus in coal-heavy grid region (grid = 386 kg/MWh). "
        "Carbon target = 40 kg/MWh. Only grid and diesel available—"
        "both far exceed target."
    ),
    phase1_load_mw=15, phase2_load_mw=25, phase3_load_mw=30,
    target_online_year=2027,
    grid_capacity_now_mw=50, grid_upgrade_mw=0, grid_upgrade_year=9999,
    budget_usd_million=200,
    reliability_target_pct=99.5,
    carbon_target_kg_per_mwh=40,
    nox_limit_g_per_mwh=1000,
    allowed_supply=["grid", "diesel_gen"],
    gt_feasible=False,
    gt_notes="Grid 386 kg/MWh, diesel 790 kg/MWh. No option meets 40 kg/MWh.",
))

_add(PowerPathwayCase(
    case_id="CV-002",
    failure_mode="CARBON_VIOLATION",
    description=(
        "Budget-constrained: $5M forces grid-only (386 kg/MWh). "
        "Solar capex needed for <100 kg/MWh would cost $40M+."
    ),
    phase1_load_mw=25, phase2_load_mw=40, phase3_load_mw=50,
    target_online_year=2027,
    grid_capacity_now_mw=60, grid_upgrade_mw=0, grid_upgrade_year=9999,
    budget_usd_million=5,
    reliability_target_pct=99.0,
    carbon_target_kg_per_mwh=100,
    nox_limit_g_per_mwh=600,
    allowed_supply=["grid", "solar_bess"],
    gt_feasible=False,
    gt_notes="$5M allows ~2 MW solar; blended carbon still ~350 kg/MWh >> 100 target.",
))

_add(PowerPathwayCase(
    case_id="CV-003",
    failure_mode="CARBON_VIOLATION",
    description=(
        "Manufacturing facility needs 99.9% reliability, forcing heavy diesel use. "
        "Diesel 790 kg/MWh dominates the mix, violating 50 kg/MWh target."
    ),
    phase1_load_mw=20, phase2_load_mw=30, phase3_load_mw=40,
    target_online_year=2026,
    grid_capacity_now_mw=0, grid_upgrade_mw=40, grid_upgrade_year=2030,
    budget_usd_million=120,
    reliability_target_pct=99.9,
    carbon_target_kg_per_mwh=50,
    nox_limit_g_per_mwh=200,
    allowed_supply=["diesel_gen", "solar_bess"],
    gt_feasible=False,
    gt_notes="Solar CF 20% can't provide 99.9% reliability without massive diesel backup.",
))

# ── HALLUCINATION_TRAP ──────────────────────────────────────────────────────

_add(PowerPathwayCase(
    case_id="HT-001",
    failure_mode="HALLUCINATION_TRAP",
    description=(
        "Utility has announced a 'potential' capacity expansion. "
        "Current CONFIRMED grid: 20 MW. Expansion is unconfirmed, no committed date. "
        "Phase 2 needs 45 MW."
    ),
    phase1_load_mw=20, phase2_load_mw=45, phase3_load_mw=60,
    target_online_year=2027,
    grid_capacity_now_mw=20, grid_upgrade_mw=0, grid_upgrade_year=9999,
    budget_usd_million=200,
    reliability_target_pct=99.9,
    carbon_target_kg_per_mwh=200,
    nox_limit_g_per_mwh=600,
    allowed_supply=["grid", "solar_bess", "wind_bess"],
    gt_feasible=False,
    gt_notes="Agent must not assume unconfirmed expansion. 20 MW confirmed << 45 MW Phase 2.",
))

_add(PowerPathwayCase(
    case_id="HT-002",
    failure_mode="HALLUCINATION_TRAP",
    description=(
        "Case describes a 'high-reliability zone with excellent renewable resources'. "
        "Grid capacity is NOT provided in the case data. "
        "Agent must not invent a capacity figure."
    ),
    phase1_load_mw=30, phase2_load_mw=50, phase3_load_mw=70,
    target_online_year=2028,
    grid_capacity_now_mw=-1,  # UNKNOWN
    grid_upgrade_mw=0, grid_upgrade_year=9999,
    budget_usd_million=200,
    reliability_target_pct=99.9,
    carbon_target_kg_per_mwh=150,
    nox_limit_g_per_mwh=400,
    allowed_supply=["grid", "solar_bess", "wind_bess"],
    gt_feasible=None,
    gt_notes="Feasibility unknown. Agent must flag missing grid capacity and refuse to recommend.",
))

_add(PowerPathwayCase(
    case_id="HT-003",
    failure_mode="HALLUCINATION_TRAP",
    description=(
        "Solar described as 'good irradiance'. Budget math only works if CF >= 35%. "
        "Actual regional CF is 22% (standard solar_bess spec). "
        "Agent must not assume optimistic CF."
    ),
    phase1_load_mw=20, phase2_load_mw=30, phase3_load_mw=40,
    target_online_year=2027,
    grid_capacity_now_mw=10, grid_upgrade_mw=10, grid_upgrade_year=2029,
    budget_usd_million=55,
    reliability_target_pct=99.0,
    carbon_target_kg_per_mwh=80,
    nox_limit_g_per_mwh=200,
    allowed_supply=["grid", "solar_bess"],
    gt_feasible=False,
    gt_notes="Solar CF is 20% (spec). 10 MW grid + 40 MW solar nameplate = ~10+8=18 MW firm << 40 MW.",
))

_add(PowerPathwayCase(
    case_id="HT-004",
    failure_mode="HALLUCINATION_TRAP",
    description=(
        "Developer claims 'the utility confirmed 100 MW is available'. "
        "Case data shows only 40 MW confirmed. "
        "Agent must use case data, not verbal claims."
    ),
    phase1_load_mw=40, phase2_load_mw=80, phase3_load_mw=100,
    target_online_year=2027,
    grid_capacity_now_mw=40, grid_upgrade_mw=60, grid_upgrade_year=2029,
    budget_usd_million=300,
    reliability_target_pct=99.9,
    carbon_target_kg_per_mwh=300,
    nox_limit_g_per_mwh=600,
    allowed_supply=["grid", "wind_bess", "solar_bess"],
    gt_feasible=False,
    gt_notes="Confirmed 40 MW now. Phase 2 needs 80 MW but upgrade only in 2029. Infeasible Phase 2.",
))


# ── Accessors ───────────────────────────────────────────────────────────────

def get_all_cases() -> list[PowerPathwayCase]:
    return list(CASES)

def get_cases_by_mode(mode: FailureMode) -> list[PowerPathwayCase]:
    return [c for c in CASES if c.failure_mode == mode]

def get_case(case_id: str) -> PowerPathwayCase:
    for c in CASES:
        if c.case_id == case_id:
            return c
    raise KeyError(f"Case {case_id!r} not found")

def export_cases_json(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(c) for c in CASES], f, indent=2)
    print(f"Exported {len(CASES)} cases → {path}")


if __name__ == "__main__":
    from collections import Counter
    modes = Counter(c.failure_mode for c in CASES)
    print(f"Total: {len(CASES)} cases")
    for m, n in sorted(modes.items()):
        print(f"  {m}: {n}")
    export_cases_json(
        Path(__file__).parent.parent.parent / "data/cases/benchmark_cases.json"
    )

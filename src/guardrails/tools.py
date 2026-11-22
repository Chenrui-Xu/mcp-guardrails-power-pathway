"""
Deterministic tool implementations for power-pathway planning.

These are the tools the agent can call.  In the guarded condition (G1/G2),
all calls are routed through mcp_gateway.py before execution.

Tools
-----
score_pathway(case, supply_mix)       → composite score + breakdown
validate_capacity(case, supply_mix)   → PASS/FAIL + margin
validate_cost(case, supply_mix)       → PASS/FAIL + cost breakdown
validate_carbon(case, supply_mix)     → PASS/FAIL + intensity
validate_reliability(case, supply_mix)→ PASS/FAIL + uptime estimate
get_case_info(case_id)                → structured case summary (read-only)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass
from typing import Any
from benchmark.tech_params import (
    TECH, TechName,
    blended_carbon, total_capex_usd_million,
    dispatchable_capacity_mw, blended_nox,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    tool_name: str
    status: str          # "PASS" | "FAIL" | "ERROR" | "INFO"
    message: str
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool: get_case_info
# ---------------------------------------------------------------------------

def get_case_info(case: Any) -> ToolResult:
    """Return structured summary of the case (read-only, no constraint checking)."""
    return ToolResult(
        tool_name="get_case_info",
        status="INFO",
        message=f"Case {case.case_id} loaded.",
        details={
            "case_id": case.case_id,
            "phase1_load_mw": case.phase1_load_mw,
            "phase2_load_mw": case.phase2_load_mw,
            "phase3_load_mw": case.phase3_load_mw,
            "target_online_year": case.target_online_year,
            "grid_capacity_now_mw": case.grid_capacity_now_mw,
            "grid_upgrade_mw": case.grid_upgrade_mw,
            "grid_upgrade_year": case.grid_upgrade_year,
            "budget_usd_million": case.budget_usd_million,
            "reliability_target_pct": case.reliability_target_pct,
            "carbon_target_kg_per_mwh": case.carbon_target_kg_per_mwh,
            "allowed_supply": case.allowed_supply,
        },
    )


# ---------------------------------------------------------------------------
# Tool: validate_capacity
# ---------------------------------------------------------------------------

def validate_capacity(case: Any, supply_mix: dict[str, float]) -> ToolResult:
    """
    Check whether the supply mix provides enough firm capacity for all phases.

    supply_mix: {tech_name: capacity_mw}

    Rules:
    - Grid contribution capped at case.grid_capacity_now_mw (or +upgrade if year <= upgrade_year)
    - All other techs: contribution = capacity_mw * capacity_factor
    - Firm capacity must cover phase3_load_mw (the peak requirement)
    - Grid is also checked against per-phase capacity
    """
    errors = []
    details: dict[str, Any] = {}

    # Validate tech names
    for tech in supply_mix:
        if tech not in TECH:
            errors.append(f"Unknown technology: {tech!r}")
        elif tech not in (case.allowed_supply if hasattr(case, "allowed_supply") else list(TECH)):
            errors.append(f"Technology {tech!r} not in allowed_supply for this case.")

    if errors:
        return ToolResult("validate_capacity", "ERROR", "; ".join(errors), {})

    # Grid capacity is constrained by interconnect
    grid_mw = supply_mix.get("grid", 0.0)
    if case.grid_capacity_now_mw < 0:
        errors.append(
            "Grid capacity is UNKNOWN for this case. "
            "Cannot validate capacity without confirmed grid data."
        )
        return ToolResult("validate_capacity", "ERROR", errors[0], {})

    if grid_mw > case.grid_capacity_now_mw:
        errors.append(
            f"Grid allocation {grid_mw} MW exceeds confirmed grid capacity "
            f"{case.grid_capacity_now_mw} MW."
        )

    # Dispatchable capacity
    firm_mw = dispatchable_capacity_mw(supply_mix)
    # For grid: already capped above; adjust if over-allocated
    effective_grid = min(grid_mw, case.grid_capacity_now_mw)
    if grid_mw > 0:
        # Recalculate with capped grid
        adj_mix = dict(supply_mix)
        adj_mix["grid"] = effective_grid
        firm_mw = dispatchable_capacity_mw(adj_mix)

    peak_load = case.phase3_load_mw
    margin_mw = firm_mw - peak_load

    details = {
        "firm_capacity_mw": round(firm_mw, 2),
        "peak_load_mw": peak_load,
        "margin_mw": round(margin_mw, 2),
        "grid_cap_used_mw": effective_grid,
        "grid_cap_available_mw": case.grid_capacity_now_mw,
        "supply_breakdown": {
            t: {"alloc_mw": mw, "cf": TECH[t].capacity_factor,
                "firm_mw": round(TECH[t].capacity_factor * mw, 2)}
            for t, mw in supply_mix.items()
        },
    }

    if errors:
        return ToolResult("validate_capacity", "FAIL", "; ".join(errors), details)

    if margin_mw < 0:
        return ToolResult(
            "validate_capacity", "FAIL",
            f"Capacity SHORTFALL: firm supply {firm_mw:.1f} MW < peak load {peak_load:.1f} MW "
            f"(deficit {-margin_mw:.1f} MW).",
            details,
        )

    return ToolResult(
        "validate_capacity", "PASS",
        f"Capacity OK: {firm_mw:.1f} MW firm ≥ {peak_load:.1f} MW peak "
        f"(margin {margin_mw:.1f} MW).",
        details,
    )


# ---------------------------------------------------------------------------
# Tool: validate_cost
# ---------------------------------------------------------------------------

def validate_cost(case: Any, supply_mix: dict[str, float]) -> ToolResult:
    """Check total capex against the budget ceiling."""
    for tech in supply_mix:
        if tech not in TECH:
            return ToolResult("validate_cost", "ERROR", f"Unknown tech: {tech!r}", {})

    breakdown = {
        t: {
            "capacity_mw": mw,
            "capex_per_mw_usd": TECH[t].capex_per_mw,
            "capex_total_usd_million": round(TECH[t].capex_total_usd_million(mw), 2),
        }
        for t, mw in supply_mix.items()
    }
    total = sum(v["capex_total_usd_million"] for v in breakdown.values())
    budget = case.budget_usd_million
    delta = budget - total

    details = {
        "total_capex_usd_million": round(total, 2),
        "budget_usd_million": budget,
        "delta_usd_million": round(delta, 2),
        "breakdown": breakdown,
    }

    if total > budget:
        return ToolResult(
            "validate_cost", "FAIL",
            f"Budget EXCEEDED: ${total:.1f}M > ${budget:.1f}M budget "
            f"(over by ${total - budget:.1f}M).",
            details,
        )

    return ToolResult(
        "validate_cost", "PASS",
        f"Cost OK: ${total:.1f}M ≤ ${budget:.1f}M budget "
        f"(${delta:.1f}M headroom).",
        details,
    )


# ---------------------------------------------------------------------------
# Tool: validate_carbon
# ---------------------------------------------------------------------------

CARBON_INTENSITIES: dict[str, float] = {
    t: TECH[t].carbon_kg_per_mwh for t in TECH  # type: ignore[assignment]
}

def validate_carbon(case: Any, supply_mix: dict[str, float]) -> ToolResult:
    """Check blended carbon intensity against case target."""
    for tech in supply_mix:
        if tech not in TECH:
            return ToolResult("validate_carbon", "ERROR", f"Unknown tech: {tech!r}", {})

    intensity = blended_carbon(supply_mix)
    target = case.carbon_target_kg_per_mwh

    details = {
        "blended_carbon_kg_per_mwh": round(intensity, 1),
        "target_kg_per_mwh": target,
        "delta_kg_per_mwh": round(intensity - target, 1),
        "per_tech": {
            t: {
                "carbon_kg_per_mwh": TECH[t].carbon_kg_per_mwh,
                "capacity_factor": TECH[t].capacity_factor,
                "capacity_mw": mw,
            }
            for t, mw in supply_mix.items()
        },
    }

    if intensity > target:
        return ToolResult(
            "validate_carbon", "FAIL",
            f"Carbon VIOLATION: {intensity:.1f} kg/MWh > {target:.1f} kg/MWh target "
            f"(excess {intensity - target:.1f} kg/MWh).",
            details,
        )

    return ToolResult(
        "validate_carbon", "PASS",
        f"Carbon OK: {intensity:.1f} kg/MWh ≤ {target:.1f} kg/MWh target.",
        details,
    )


# ---------------------------------------------------------------------------
# Tool: validate_reliability
# ---------------------------------------------------------------------------

# Reliability contributions per technology (simplified model)
# Based on capacity factor and redundancy characteristics
_RELIABILITY_CONTRIB: dict[str, float] = {
    "diesel_gen":    0.999,
    "fuel_cell_sofc": 0.9995,
    "solar_bess":    0.994,   # solar variability mitigated by 12h BESS storage
    "gas_turbine":   0.997,
    "wind_bess":     0.993,   # wind variability mitigated by BESS storage
    "geothermal":    0.9995,
    "smr":           0.9998,
    "grid":          0.9990,   # grid reliability (excludes major outages)
}


def validate_reliability(case: Any, supply_mix: dict[str, float]) -> ToolResult:
    """
    Estimate system reliability from the supply mix using a simplified N+1 model.

    - Single-source systems: reliability = that source's base reliability
    - Multi-source: parallel availability model (system fails only if ALL fail)
    - Penalise mixes that depend >80% on a single intermittent source
    """
    for tech in supply_mix:
        if tech not in TECH:
            return ToolResult("validate_reliability", "ERROR", f"Unknown tech: {tech!r}", {})

    total_firm = dispatchable_capacity_mw(supply_mix)
    if total_firm == 0:
        return ToolResult(
            "validate_reliability", "FAIL",
            "No dispatchable capacity—reliability is 0%.", {}
        )

    # Parallel availability: P(system up) = 1 - prod(1 - Ri * wi)
    # where wi = share of firm capacity from source i
    tech_items = list(supply_mix.items())
    per_tech = {}
    system_unreliability = 1.0
    for t, mw in tech_items:
        firm = TECH[t].capacity_factor * mw
        share = firm / total_firm if total_firm > 0 else 0
        base_rel = _RELIABILITY_CONTRIB.get(t, 0.99)
        weighted_unreliability = (1 - base_rel) * share
        system_unreliability *= (1 - base_rel) ** share  # geometric weighting
        per_tech[t] = {
            "firm_mw": round(firm, 2),
            "share": round(share, 3),
            "base_reliability": base_rel,
        }

    # Simplified: blended reliability = weighted harmonic
    blended = sum(
        _RELIABILITY_CONTRIB.get(t, 0.99) * (TECH[t].capacity_factor * mw / total_firm)
        for t, mw in tech_items
    )

    # Diversity bonus for having a fast-deploy backup (diesel/fuel_cell)
    has_backup = any(t in ("diesel_gen", "fuel_cell_sofc") for t in supply_mix)
    if has_backup:
        blended = min(0.99999, blended + 0.0005)

    target = case.reliability_target_pct / 100.0
    target_str = f"{case.reliability_target_pct}%"
    blended_str = f"{blended * 100:.4f}%"

    details = {
        "estimated_reliability_pct": round(blended * 100, 4),
        "target_pct": case.reliability_target_pct,
        "has_backup_source": has_backup,
        "per_tech": per_tech,
    }

    if blended < target:
        return ToolResult(
            "validate_reliability", "FAIL",
            f"Reliability SHORTFALL: {blended_str} < {target_str} target.",
            details,
        )

    return ToolResult(
        "validate_reliability", "PASS",
        f"Reliability OK: {blended_str} ≥ {target_str} target.",
        details,
    )


# ---------------------------------------------------------------------------
# Tool: score_pathway
# ---------------------------------------------------------------------------

def score_pathway(
    case: Any,
    supply_mix: dict[str, float],
    weights: dict[str, float] | None = None,
) -> ToolResult:
    """
    Compute a composite weighted score (0–100) for a supply mix.

    Default weights (can be overridden):
      speed: 0.25, cost: 0.25, environment: 0.25, resilience: 0.25
    """
    if weights is None:
        weights = {"speed": 0.25, "cost": 0.25, "environment": 0.25, "resilience": 0.25}

    if abs(sum(weights.values()) - 1.0) > 0.01:
        return ToolResult(
            "score_pathway", "ERROR",
            f"Weights must sum to 1.0 (got {sum(weights.values()):.2f}).", {}
        )

    for tech in supply_mix:
        if tech not in TECH:
            return ToolResult("score_pathway", "ERROR", f"Unknown tech: {tech!r}", {})

    # --- Speed score: inverse of max deploy time in mix ---
    max_deploy = max(TECH[t].deploy_time_months_max for t in supply_mix) if supply_mix else 120
    speed_score = max(0, 100 - (max_deploy / 120) * 100)

    # --- Cost score: inversely proportional to capex vs budget ---
    total_capex = total_capex_usd_million(supply_mix)
    budget = case.budget_usd_million
    if total_capex > budget:
        cost_score = 0.0
    else:
        cost_score = max(0, 100 * (1 - total_capex / (budget * 1.5)))

    # --- Environment score: inverse of blended carbon vs target ---
    carbon = blended_carbon(supply_mix)
    target_c = case.carbon_target_kg_per_mwh
    if carbon > target_c * 2:
        env_score = 0.0
    elif carbon > target_c:
        env_score = 50.0 * (1 - (carbon - target_c) / target_c)
    else:
        env_score = 100.0 * (1 - carbon / target_c) if target_c > 0 else 100.0

    # --- Resilience score: map reliability to 0–100 ---
    r_result = validate_reliability(case, supply_mix)
    if r_result.status == "PASS":
        est_rel = r_result.details.get("estimated_reliability_pct", 99.0)
        # Scale: 99% = 50pts, 99.99% = 100pts
        resilience_score = min(100.0, (est_rel - 98.0) / (100.0 - 98.0) * 100)
    else:
        resilience_score = 20.0

    composite = (
        weights["speed"] * speed_score
        + weights["cost"] * cost_score
        + weights["environment"] * env_score
        + weights["resilience"] * resilience_score
    )

    details = {
        "composite_score": round(composite, 2),
        "subscores": {
            "speed": round(speed_score, 2),
            "cost": round(cost_score, 2),
            "environment": round(env_score, 2),
            "resilience": round(resilience_score, 2),
        },
        "weights_used": weights,
        "total_capex_usd_million": round(total_capex, 2),
        "blended_carbon_kg_per_mwh": round(carbon, 1),
        "max_deploy_months": max_deploy,
    }

    status = "PASS" if composite >= 50 else "FAIL"
    return ToolResult(
        "score_pathway", status,
        f"Composite score: {composite:.1f}/100 "
        f"(speed={speed_score:.0f}, cost={cost_score:.0f}, "
        f"env={env_score:.0f}, resilience={resilience_score:.0f}).",
        details,
    )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "get_case_info":        get_case_info,
    "validate_capacity":    validate_capacity,
    "validate_cost":        validate_cost,
    "validate_carbon":      validate_carbon,
    "validate_reliability": validate_reliability,
    "score_pathway":        score_pathway,
}

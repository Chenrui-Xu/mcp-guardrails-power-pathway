"""
Technology parameters derived directly from the Power Pathway Optimizer prototype.
Source: https://github.com/Chenrui-Xu/power-pathway-optimizer
"""

from dataclasses import dataclass
from typing import Literal

TechName = Literal[
    "diesel_gen",
    "fuel_cell_sofc",
    "solar_bess",
    "gas_turbine",
    "wind_bess",
    "geothermal",
    "smr",
    "grid",
]


@dataclass(frozen=True)
class TechSpec:
    name: TechName
    display_name: str
    capex_per_kw: float        # $/kW
    lcoe_per_mwh: float        # $/MWh (all-in levelized)
    carbon_kg_per_mwh: float   # kg CO2/MWh
    capacity_factor: float     # 0–1 (dispatchable fraction)
    deploy_time_months_min: int
    deploy_time_months_max: int
    nox_g_per_mwh: float       # g NOx/MWh (0 for non-combustion)
    notes: str = ""

    @property
    def capex_per_mw(self) -> float:
        return self.capex_per_kw * 1_000  # $/MW

    def capex_total_usd_million(self, capacity_mw: float) -> float:
        return self.capex_per_mw * capacity_mw / 1_000_000


TECH: dict[TechName, TechSpec] = {
    "diesel_gen": TechSpec(
        name="diesel_gen",
        display_name="Diesel Generator",
        capex_per_kw=900,
        lcoe_per_mwh=160,
        carbon_kg_per_mwh=790,
        capacity_factor=0.80,
        deploy_time_months_min=1,
        deploy_time_months_max=6,
        nox_g_per_mwh=4200,
        notes="Fast deploy; high carbon and NOx; emergency/peaking use.",
    ),
    "fuel_cell_sofc": TechSpec(
        name="fuel_cell_sofc",
        display_name="Fuel Cell (SOFC)",
        capex_per_kw=3000,
        lcoe_per_mwh=110,
        carbon_kg_per_mwh=470,
        capacity_factor=0.92,
        deploy_time_months_min=3,
        deploy_time_months_max=6,
        nox_g_per_mwh=50,
        notes="High reliability; moderate carbon; no combustion NOx.",
    ),
    "solar_bess": TechSpec(
        name="solar_bess",
        display_name="Solar + BESS (12h)",
        capex_per_kw=2400,
        lcoe_per_mwh=60,
        carbon_kg_per_mwh=40,
        capacity_factor=0.20,
        deploy_time_months_min=6,
        deploy_time_months_max=18,
        nox_g_per_mwh=0,
        notes="Low carbon; variable CF; 12h storage included; land-intensive.",
    ),
    "gas_turbine": TechSpec(
        name="gas_turbine",
        display_name="Gas Turbine",
        capex_per_kw=1600,
        lcoe_per_mwh=95,
        carbon_kg_per_mwh=600,
        capacity_factor=0.75,
        deploy_time_months_min=12,
        deploy_time_months_max=24,
        nox_g_per_mwh=1800,
        notes="Scalable dispatchable; moderate carbon; permitting risk.",
    ),
    "wind_bess": TechSpec(
        name="wind_bess",
        display_name="Wind + BESS",
        capex_per_kw=1700,
        lcoe_per_mwh=80,
        carbon_kg_per_mwh=10,
        capacity_factor=0.36,
        deploy_time_months_min=12,
        deploy_time_months_max=36,
        nox_g_per_mwh=0,
        notes="Lowest carbon; variable CF; long permitting timeline.",
    ),
    "geothermal": TechSpec(
        name="geothermal",
        display_name="Geothermal",
        capex_per_kw=2300,
        lcoe_per_mwh=90,
        carbon_kg_per_mwh=35,
        capacity_factor=0.81,
        deploy_time_months_min=60,
        deploy_time_months_max=100,
        nox_g_per_mwh=0,
        notes="High CF; very long development; geography-constrained.",
    ),
    "smr": TechSpec(
        name="smr",
        display_name="Small Modular Reactor",
        capex_per_kw=5600,
        lcoe_per_mwh=95,
        carbon_kg_per_mwh=12,
        capacity_factor=0.94,
        deploy_time_months_min=60,
        deploy_time_months_max=120,
        nox_g_per_mwh=0,
        notes="Ultra-low carbon; highest capex; very long timeline; regulatory risk.",
    ),
    "grid": TechSpec(
        name="grid",
        display_name="Utility Grid",
        capex_per_kw=0,
        lcoe_per_mwh=70,           # blended retail rate (US avg ~$70/MWh for large C&I)
        carbon_kg_per_mwh=386,     # US avg grid carbon intensity 2024
        capacity_factor=0.999,
        deploy_time_months_min=0,
        deploy_time_months_max=0,
        nox_g_per_mwh=500,
        notes="Immediately available up to interconnect limit; carbon depends on grid mix.",
    ),
}


def get_tech(name: TechName) -> TechSpec:
    return TECH[name]


def blended_carbon(supply_mix: dict[TechName, float]) -> float:
    """
    Compute blended carbon intensity (kg CO2/MWh) for a supply mix.
    supply_mix: {tech_name: capacity_mw}. Weights by dispatchable capacity.
    """
    total_dispatch = sum(
        TECH[t].capacity_factor * mw for t, mw in supply_mix.items()
    )
    if total_dispatch == 0:
        return float("inf")
    weighted = sum(
        TECH[t].carbon_kg_per_mwh * TECH[t].capacity_factor * mw
        for t, mw in supply_mix.items()
    )
    return weighted / total_dispatch


def total_capex_usd_million(supply_mix: dict[TechName, float]) -> float:
    """Total capex in $M for a supply mix {tech_name: capacity_mw}."""
    return sum(TECH[t].capex_total_usd_million(mw) for t, mw in supply_mix.items())


def dispatchable_capacity_mw(supply_mix: dict[TechName, float]) -> float:
    """Effective firm/dispatchable capacity in MW."""
    return sum(TECH[t].capacity_factor * mw for t, mw in supply_mix.items())


def blended_nox(supply_mix: dict[TechName, float]) -> float:
    """Blended NOx intensity (g/MWh) for a supply mix."""
    total_dispatch = sum(TECH[t].capacity_factor * mw for t, mw in supply_mix.items())
    if total_dispatch == 0:
        return float("inf")
    weighted = sum(
        TECH[t].nox_g_per_mwh * TECH[t].capacity_factor * mw
        for t, mw in supply_mix.items()
    )
    return weighted / total_dispatch
